#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/core/native_dense.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/platform/memory.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>
#include <cmath>
#include <cstring>
#include <vector>

namespace strata::core {
namespace {
int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1: every layer
bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}
// S23 experiment (STRATA_HC_Q8=1): the hyper-connection projections' Q8_0 bytes for the verify read
bool hc_q8_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_HC_Q8"); return v != nullptr && v[0] == '1'; }();
    return on;
}
bool hc_q8_name(const std::string& name) {
    if (name == "output_hc_down.weight" || name == "output_hc_up.weight") return true;   // S25: the final mixer too
    if (name.rfind("blk.", 0) != 0) return false;
    static const char* suffixes[] = {".hc_attn_down.weight", ".hc_attn_up.weight", ".hc_attn_inject.weight",
                                     ".hc_ffn_down.weight", ".hc_ffn_up.weight", ".hc_ffn_inject.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
// S25: the GGUF keeps the inject rows as F32 - quantized here to Q8_0 (ggml's reference rounding) so the Q8_0 read
// covers every hc projection
uint16_t half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e <= 0) {                                   // subnormal or zero
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - e);
        uint32_t h = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) (sign | h);
}
std::vector<uint8_t> q8_0_of(const float* x, uint64_t n) {
    std::vector<uint8_t> out((size_t) (n / 32 * 34));
    for (uint64_t b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = (std::max)(amax, std::fabs(x[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        uint8_t* o = out.data() + b * 34;
        const uint16_t hb = half_bits(d);
        std::memcpy(o, &hb, 2);
        for (int j = 0; j < 32; ++j) o[2 + j] = (uint8_t) (int8_t) std::lround(x[b * 32 + j] * id);
    }
    return out;
}
bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree {
    void
    operator()(void *p) const {
        if (p) sycl::free(p, dpct::get_in_order_queue());
    }
};
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
    DevicePtr packed;   // STRATA_Q8_PACKED=1 copy, or null
};
}

// the shared header declares the static member (b60aab1b); this port only needs the env switch
bool NativeDense::hc_q8_requested() { return hc_q8_env(); }

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors())
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    tensor.shape.size() == 2)
                    out.insert(tensor.name);
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    for (const void* p : packed_keys_) strata::kernels::native_q8_0_packed_unregister(p);
    if (scratch_) sycl::free(scratch_, dpct::get_in_order_queue());
    for (void *p : weights_)
        DPCT_CHECK_ERROR(sycl::free(p, dpct::get_in_order_queue()));
}

bool NativeDense::load(const std::vector<std::string> &shards,
                       WeightTable &table, std::string &err,
                       bool include_ple_key, int64_t layer_lo,
                       int64_t layer_hi) try {
    auto outside = [&](const std::string& name) {   // a blk.<l>. tensor of another stage's layers
        if (layer_hi < 0 || name.rfind("blk.", 0) != 0) return false;
        const long l = std::strtol(name.c_str() + 4, nullptr, 10);
        return l < layer_lo || l >= layer_hi;
    };
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0, hc_q8_bytes = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                err = strata::check_engine_architecture(gguf);
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
                // the uploads below read these: ask for them now so the reads overlap
                if (eligible(tensor, include_ple_key) && !outside(tensor.name) &&
                    strata::kernels::native_mmvq_supported(tensor.type))
                    strata::platform::advise_willneed(gguf.tensor_data(tensor), bytes);
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                void* allocation = nullptr;
                auto status =
                    DPCT_CHECK_ERROR(allocation = (void *)strata::malloc_device_guarded(
                                         bytes, dpct::get_in_order_queue()));
                DevicePtr data(allocation);
                if (status == 0)
                    status = DPCT_CHECK_ERROR(
                        dpct::get_in_order_queue()
                            .memcpy(data.get(), gguf.tensor_data(tensor), bytes)
                            .wait());
                /*
                DPCT1000: Error handling if-stmt was detected but could not
                be rewritten.
                */
                if (status != 0) {
                    /*
                    DPCT1009: SYCL reports errors using exceptions and does
                    not use error codes. Please replace the
                    "get_error_string_dummy(...)" with a real error-handling
                    function.
                    */
                    /*
                    DPCT1001: The statement could not be removed.
                    */
                    err = "native dense upload " + tensor.name + ": " +
                          dpct::get_error_string_dummy(status);
                        return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                // STRATA_Q8_PACKED=1: a second, packed copy of an eligible Q8_0 matrix for the decode MMVQ (the GGUF
                // copy stays: the prompt path's GEMMs read it).
                DevicePtr packed;
                if (tensor.type == 8 && tensor.name != "blk.1.ple_key.weight" &&   // the PLE has its own kernel
                    strata::kernels::native_q8_0_packed_enabled() &&
                    strata::kernels::native_q8_0_packed_eligible((int) ref.ne0, (int) ref.ne1)) {
                    std::vector<uint8_t> host(bytes);
                    strata::kernels::native_q8_0_pack_host(gguf.tensor_data(tensor), host.data(), (int) ref.ne0,
                                                           (int) ref.ne1);
                    void* packed_allocation = nullptr;
                    auto packed_status = DPCT_CHECK_ERROR(
                        packed_allocation = (void *)strata::malloc_device_guarded(
                            bytes, dpct::get_in_order_queue()));
                    packed.reset(packed_allocation);
                    if (packed_status == 0)
                        packed_status = DPCT_CHECK_ERROR(
                            dpct::get_in_order_queue()
                                .memcpy(packed.get(), host.data(), bytes)
                                .wait());
                    /*
                    DPCT1000: Error handling if-stmt was detected but could
                    not be rewritten.
                    */
                    if (packed_status != 0) {
                        /*
                        DPCT1009: SYCL reports errors using exceptions and
                        does not use error codes. Please replace the
                        "get_error_string_dummy(...)" with a real error-handling
                        function.
                        */
                        /*
                        DPCT1001: The statement could not be removed.
                        */
                        err = "native dense packed upload " + tensor.name +
                              ": " +
                              dpct::get_error_string_dummy(packed_status);
                        return false;
                    }
                }
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data), std::move(packed)});
            }
            if (hc_q8_requested())
                for (const auto& tensor : gguf.tensors()) {
                    // S25: the F32 inject rows hold BF16-exact values: the read takes the pack's BF16 rows unless
                    // STRATA_HC_Q8_INJECT=1 asks for a Q8_0 copy
                    static const bool q8_inject = [] { const char* v = std::getenv("STRATA_HC_Q8_INJECT"); return v && v[0] == '1'; }();
                    const bool f32_inject = q8_inject && tensor.type == 0 && tensor.name.ends_with("_inject.weight");
                    if ((tensor.type != 8 && !f32_inject) || !hc_q8_name(tensor.name) || tensor.shape.size() != 2) continue;
                    auto found = table.table_.find(tensor.name);
                    if (found == table.table_.end() || found->second.hc_q8 != nullptr) continue;
                    auto& ref = found->second;
                    if (tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1 || ref.ne0 % 32 != 0) {
                        err = "native dense (STRATA_HC_Q8): incompatible matrix " + tensor.name; return false;
                    }
                    const uint64_t bytes = (uint64_t) ref.ne0 / 32 * 34 * (uint64_t) ref.ne1;
                    void* p = nullptr;
                    std::vector<uint8_t> q8;
                    if (f32_inject) q8 = q8_0_of((const float*) gguf.tensor_data(tensor), (uint64_t) ref.ne0 * (uint64_t) ref.ne1);
                    if (DPCT_CHECK_ERROR(
                            p = (void *)strata::malloc_device_guarded(
                                bytes, dpct::get_in_order_queue())) != 0 ||
                        /*
                        DPCT1114: cudaMemcpy is migrated to
                        asynchronization memcpy, assuming in the original code
                        the source host memory is pageable memory. If the memory
                        is not pageable, call wait() on event return by memcpy
                        API to ensure synchronization behavior.
                        */
                        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
                            p,
                            f32_inject ? (const void *)q8.data()
                                       : gguf.tensor_data(tensor),
                            bytes).wait()) != 0) {
                        err = "native dense (STRATA_HC_Q8) upload " + tensor.name; return false;
                    }
                    ref.hc_q8 = p;
                    weights_.push_back(p);
                    hc_q8_bytes += bytes;
                }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        const auto status =
            DPCT_CHECK_ERROR(allocation = (void *)strata::malloc_device_guarded(
                                 strata::kernels::native_q8_1_bytes(max_in),
                                 dpct::get_in_order_queue()));
        DevicePtr scratch(allocation);
        /*
        DPCT1009: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        /*
        DPCT1001: The statement could not be removed.
        */
        /*
        DPCT1000: Error handling if-stmt was detected but could not be
        rewritten.
        */
        if (status != 0) {
            err = std::string("native dense scratch: ") +
                  dpct::get_error_string_dummy(status);
            return false;
        }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        uint64_t packed_bytes = 0;
        size_t packed_count = 0;
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            if (item.packed) {
                strata::kernels::native_q8_0_packed_register(item.data.get(), item.packed.get(), (int) item.ref->ne0,
                                                             (int) item.ref->ne1);
                packed_keys_.push_back(item.data.get());
                packed_bytes += item.bytes;
                ++packed_count;
                weights_.push_back(item.packed.release());
            }
            weights_.push_back(item.data.release());
        }
        if (packed_count)
            std::fprintf(stderr, "native dense: STRATA_Q8_PACKED=1 packed %zu Q8_0 matrices for decode (+%.1f MiB)\n",
                         packed_count, packed_bytes / 1048576.0);
        scratch_ = scratch.release();
        bytes_ = total + hc_q8_bytes;
        if (hc_q8_requested())
            std::fprintf(stderr, "strata: STRATA_HC_Q8=1: %.2f GiB of Q8_0 hyper-connection projections for the verify read\n",
                         (double) hc_q8_bytes / 1073741824.0);
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
} // namespace strata::core
