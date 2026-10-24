#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/native_head.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>

namespace strata::core {

NativeHead::~NativeHead() {
    if (weights_) strata::kernels::native_q6_k_unpack(weights_);
    if (scratch_) sycl::free(scratch_, dpct::get_in_order_queue());
    if (weights_) sycl::free(weights_, dpct::get_in_order_queue());
}

bool NativeHead::load(const std::vector<std::string> &shards, int64_t n_in,
                      int64_t n_out, std::string &err) try {
    if (loaded()) { err = "native head is already loaded"; return false; }
    if (n_in <= 0 || n_out <= 0 || n_in > INT_MAX || n_out > INT_MAX || n_in % 256) {
        err = "native head requires positive int32 dimensions and whole 256-value rows";
        return false;
    }
    try {
        // The architecture is the metadata shard's; output.weight comes from whichever shard holds it (shard 2
        // of Unsloth's UD-Q4_K_XL, whose shard 1 holds no tensor).  GgufModel refuses a duplicate across shards.
        const strata::GgufModel model(shards);
        err = strata::check_engine_architecture(model.meta());
        if (!err.empty()) return false;
        size_t at = 0;
        const strata::TensorInfo* tensor = model.find("output.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!tensor || !strata::kernels::native_mmvq_supported((int) tensor->type) || tensor->shape.size() != 2 ||
            tensor->shape[0] != (uint64_t) n_in || tensor->shape[1] != (uint64_t) n_out) {
            err = "native head: expected a natively supported output.weight with the canonical head dimensions";
            return false;
        }
        const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) tensor->type, (int) n_in, (int) n_out);
        const uint64_t payload = gguf.file_size() - gguf.data_start();
        if (tensor->offset > payload || bytes > payload - tensor->offset) {
            err = "native head: truncated output.weight payload";
            return false;
        }
        void* weights = nullptr;
        void* scratch = nullptr;
        dpct::err0 status =
            DPCT_CHECK_ERROR(weights = (void *)strata::malloc_device_guarded(
                                 bytes, dpct::get_in_order_queue()));
        if (status == 0)
            status = DPCT_CHECK_ERROR(
                scratch = (void *)strata::malloc_device_guarded(
                    strata::kernels::native_q8_1_bytes((int)n_in, 1),
                    dpct::get_in_order_queue()));
        strata::platform::advise_willneed(gguf.tensor_data(*tensor), bytes);
        if (status == 0)
            status = DPCT_CHECK_ERROR(
                dpct::get_in_order_queue()
                    .memcpy(weights, gguf.tensor_data(*tensor), bytes)
                    .wait());
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (status != 0) {
            if (scratch) sycl::free(scratch, dpct::get_in_order_queue());
            if (weights) sycl::free(weights, dpct::get_in_order_queue());
            /*
            DPCT1009: SYCL reports errors using exceptions and does not use
            error codes. Please replace the "get_error_string_dummy(...)" with a
            real error-handling function.
            */
            /*
            DPCT1001: The statement could not be removed.
            */
            err = std::string("native head upload: ") +
                  dpct::get_error_string_dummy(status);
            return false;
        }
        weights_ = weights;
        scratch_ = scratch;
        bytes_ = bytes;
        n_in_ = (int) n_in;
        n_out_ = (int) n_out;
        type_ = (int) tensor->type;
        if (type_ == 14 && strata::kernels::native_q6_k_packed_enabled())   // STRATA_Q6_PACKED=1
            strata::kernels::native_q6_k_pack(weights_, n_in_, n_out_, "output head");
        if (type_ == 14 && !strata::kernels::native_q6_k_packed_enabled() && strata::kernels::native_q6_k_planar_enabled())   // STRATA_Q6_PLANAR: stored as planes (same bytes)
            weights_ = strata::kernels::native_q6_k_planar_adopt(weights_, n_in_, n_out_, "output head");
        return true;
    } catch (const std::exception& error) {
        err = std::string("native head: ") + error.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool NativeHead::run(const float *mixed, float *logits, void *stream,
                     std::string &err) const try {
    if (!loaded() || !mixed || !logits || !stream) {
        err = "native head requires loaded weights, device buffers and an explicit stream";
        return false;
    }
    try {
        if (type_ == 13) {
            strata::kernels::native_q5_k_f32(weights_, mixed, scratch_, logits, n_in_, n_out_, 1, stream);
        } else {
            strata::kernels::native_quantize_q8_1(mixed, scratch_, n_in_, 1, stream);
            strata::kernels::native_mmvq(type_, weights_, scratch_, logits, n_in_, n_out_, 1, stream);
        }
    } catch (const std::exception& error) {
        err = std::string("native head launch: ") + error.what();
        return false;
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaPeekAtLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 status = 0;
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (status != 0) {
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        err = std::string("native head launch: ") +
              dpct::get_error_string_dummy(status);
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// ================================ plan v0.3 P6: THE NATIVE EMBEDDING ================================

namespace {
const NativeEmbed* g_embed = nullptr;
}
void set_native_embed(const NativeEmbed* e) { g_embed = e; }
const NativeEmbed* native_embed() { return g_embed; }

NativeEmbed::~NativeEmbed() {
    if (host_) sycl::free(host_, dpct::get_in_order_queue());
    else if (dev_)
        sycl::free((void *) dev_, dpct::get_in_order_queue()); // the VRAM fallback below
}

bool NativeEmbed::load(const std::vector<std::string> &shards, int64_t n_embd,
                       int64_t n_vocab, std::string &err) try {
    try {
        const strata::GgufModel model(shards);
        // --embd-gguf's one-tensor file (tools/embd_bf16_pack.py) says "strata-embd": only its tensor is checked
        const strata::MetaValue* arch = model.meta().get("general.architecture");
        err = arch != nullptr && arch->s == "strata-embd" ? std::string() : strata::check_engine_architecture(model.meta());
        if (!err.empty()) { err = "native embedding: " + err; return false; }
        size_t at = 0;
        const strata::TensorInfo* t = model.find("token_embd.weight", &at);
        const strata::GgufFile& gguf = model.shard(at);
        if (!t || t->shape.size() != 2 || t->shape[0] != (uint64_t) n_embd || t->shape[1] != (uint64_t) n_vocab ||
            !strata::kernels::embed_type_supported((int) t->type) || n_embd % 256) {
            err = "native embedding: token_embd.weight is absent, of another shape, or of a type without a GPU "
                  "dequantizer";
            return false;
        }
        row_ = strata::kernels::iq_row_bytes((int) t->type, n_embd);
        bytes_ = (uint64_t) row_ * (uint64_t) n_vocab;
        // the table is copied out of the mapping below: a truncated shard must be an error, not a read past EOF
        if (!model.in_bounds(*t, at) || strata::tensor_payload_bytes(*t) != bytes_) {
            err = "native embedding: token_embd.weight's payload is truncated or not " + std::to_string(bytes_) +
                  " B (" + gguf.path() + ")";
            bytes_ = 0;
            return false;
        }
        strata::platform::advise_willneed(gguf.tensor_data(*t), bytes_);   // copied out of the mapping below
        /*
        DPCT1048: The original value cudaHostAllocMapped is not meaningful in
        the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        /*
        DPCT1048: The original value cudaHostAllocPortable is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        if (DPCT_CHECK_ERROR(host_ = (void *)sycl::malloc_host(
                                 bytes_, dpct::get_in_order_queue())) != 0) {
            // Under WSL2 the driver's pinned/mapped host budget (~1 GiB) can be spent by the GPU contexts
            // themselves (three cards). The table is only gathered from, so keep it in the current device's VRAM
            // instead: it costs its size there and reads faster than over PCIe.
            /*
            DPCT1026: The call to cudaGetLastError was removed because this
            functionality is redundant in SYCL.
            */
            host_ = nullptr;
            void* d = nullptr;
            if (DPCT_CHECK_ERROR(d = (void *)strata::malloc_device_guarded(
                                     bytes_, dpct::get_in_order_queue())) !=
                    0 ||
                /*
                DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
                assuming in the original code the source host memory is pageable
                memory. If the memory is not pageable, call wait() on event
                return by memcpy API to ensure synchronization behavior.
                */
                DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                    d, gguf.tensor_data(*t), bytes_).wait()) != 0) {
                if (d) sycl::free(d, dpct::get_in_order_queue());
                /*
                DPCT1026: The call to cudaGetLastError was removed because
                this functionality is redundant in SYCL.
                */
                err = "native embedding: cannot pin " +
                      std::to_string(bytes_ >> 20) +
                      " MiB, nor place it in VRAM";
                return false;
            }
            std::fprintf(stderr, "strata: native embedding: cannot pin %llu MiB, kept in VRAM instead\n",
                         (unsigned long long) (bytes_ >> 20));
            dev_ = d;
        } else {
            std::memcpy(host_, gguf.tensor_data(*t), bytes_);
            void* d = nullptr;
#if defined(STRATA_USE_HIP)
            // #325: the Windows HIP stack can refuse the device alias of a mapped allocation (and, when it gives
            // one, it is the host address itself - unified addressing; kernels read it correctly there, a
            // device-to-device copy into it does not land: tests/hip/mapped_alias). The table is only gathered from,
            // so without an alias it goes into VRAM like the unpinnable case above, instead of failing the start.
            if (cudaHostGetDevicePointer(&d, host_, 0) != cudaSuccess || d == nullptr) {
                cudaGetLastError();
                d = nullptr;
                if (cudaMalloc(&d, bytes_) != cudaSuccess ||
                    cudaMemcpy(d, gguf.tensor_data(*t), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
                    if (d) cudaFree(d);
                    cudaGetLastError();
                    err = "native embedding: no device alias for the mapped table, and no VRAM to copy it into";
                    return false;
                }
                cudaFreeHost(host_);   // the destructor frees dev_ when host_ is null
                host_ = nullptr;
                std::fprintf(stderr, "strata: native embedding: no device alias for the mapped table, kept in VRAM\n");
            }
#else
            if (DPCT_CHECK_ERROR(d = (void *)host_) != 0) {
                err = "native embedding: no device alias for the mapped table";
                return false;
            }
#endif
            dev_ = d;
        }
        type_ = (int) t->type;
        n_embd_ = n_embd;
        n_vocab_ = n_vocab;
        return true;
    } catch (const std::exception& e) {
        err = std::string("native embedding: ") + e.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out, void* stream) const {
    strata::kernels::iq_embed_rows(type_, dev_, row_, tokens, n_tok, n_embd_, out, stream);
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    strata::kernels::iq_dequant_f32(type_, (const uint8_t*) dev_ + (size_t) token * row_, n_embd_, out, stream);
}

}  // namespace strata::core
