#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/core/remote_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    dpct::err0 status = 0;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = DPCT_CHECK_ERROR(previous = dpct::get_current_device_id());
        if (status != 0) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        /*
        DPCT1093: The "device" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        status = DPCT_CHECK_ERROR(dpct::select_device(device));
        if (status != 0) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    /*
    DPCT1093: The "previous" device may be not the one intended for use.
    Adjust the selected device if needed.
    */
    ~DeviceScope() { if (previous >= 0) dpct::select_device(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) +
               " experts: " + (failed_step ? failed_step : "device switch") +
               /*
               DPCT1009: SYCL reports errors using exceptions and does not
               use error codes. Please replace the "get_error_string_dummy(...)"
               with a real error-handling function.
               */
               "(" +
               std::to_string(device) +
               ") failed: " + dpct::get_error_string_dummy(status) +
               " (CUDA error " + std::to_string((int)status) + ")";
    }
};

bool check(dpct::err0 result, const char *what, std::string &err, int device) {
    if (result == 0) return true;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " +
          dpct::get_error_string_dummy(result);
    return false;
}
} // namespace

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double &free_gib,
                              std::string &err) try {
    int count = 0;
    if (!check(DPCT_CHECK_ERROR(count = dpct::device_count()),
               "cudaGetDeviceCount", err, device)) return false;
    if (device < 1 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
    /*
    DPCT1007: Migration of cudaInitDevice is not supported.
    */
    (void) spin;   // SYCL: no cudaInitDevice scheduling flags; the runtime picks its own wait policy
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    if (!check(DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(
                   free_bytes, total_bytes)),
               "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

void RemoteExperts::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) stream_->wait();
        cache_.close();
        if (d_x_) sycl::free(d_x_, dpct::get_in_order_queue());
        if (d_out_) sycl::free(d_out_, dpct::get_in_order_queue());
        if (d_q8_) sycl::free(d_q8_, dpct::get_in_order_queue());
        if (d_scales_) sycl::free(d_scales_, dpct::get_in_order_queue());
        if (d_scratch_) sycl::free(d_scratch_, dpct::get_in_order_queue());
        if (d_meta_) sycl::free(d_meta_, dpct::get_in_order_queue());
        if (h_x_) sycl::free(h_x_, dpct::get_in_order_queue());
        if (h_out_) sycl::free(h_out_, dpct::get_in_order_queue());
        if (h_meta_) sycl::free(h_meta_, dpct::get_in_order_queue());
        if (stream_) dpct::get_current_device().destroy_queue(stream_);
    }
    device_ = -1;
    stream_ = &dpct::get_in_order_queue();
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>> &ranked,
                         const ExpertCache &primary, ExpertSource &source,
                         std::vector<uint8_t> &claimed, std::string &err, bool /*auto_size*/) try {
    close();
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this helper does not rotate the expert input yet (SYCL)
        err = "second-GPU experts: Hadamard-folded (Gyro) experts are not supported on a second Intel GPU yet";
        return false;
    }
    int count = 0;
    if (!check(DPCT_CHECK_ERROR(count = dpct::device_count()),
               "cudaGetDeviceCount", err, device)) return false;
    if (device < 1 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts) {
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) slots);
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            selected.push_back(pair);
            picked[index] = 1;
            if ((int) selected.size() >= slots) break;
        }
    }
    if (selected.empty()) { err = "CUDA" + std::to_string(device) + " experts: no unclaimed experts remain"; close(); return false; }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    size_t free_bytes = 0, total_bytes = 0;
    /*
    DPCT1106: 'cudaMemGetInfo' was migrated with the Intel extensions for
    device information which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    if (!check(DPCT_CHECK_ERROR(dpct::get_current_device().get_memory_info(
                   free_bytes, total_bytes)),
               "free memory", err, device)) {
        close(); return false;
    }
    uint64_t needed = 0;
    for (const auto& pair : selected)
        needed += lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + (512ull << 20) > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, layers, experts, err)
                                     : cache_.open((int64_t) selected.size(), layers, experts,
                                                   (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const bool allocated =
        /*
        DPCT1025: The SYCL queue is created ignoring the flag and priority
        options.
        */
        check(DPCT_CHECK_ERROR(
                  stream_ = dpct::get_current_device().create_queue(true)),
              "stream", err, device) &&
        /*
        DPCT1048: The original value cudaHostAllocPortable is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        /*
        DPCT1048: The original value cudaHostAllocMapped is not meaningful in
        the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        check(
            DPCT_CHECK_ERROR(h_x_ = sycl::malloc_host<float>(
                                 (size_t)CAP * H, dpct::get_in_order_queue())),
            "input staging", err, device) &&
        /*
        DPCT1048: The original value cudaHostAllocPortable is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        /*
        DPCT1048: The original value cudaHostAllocMapped is not meaningful in
        the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        check(
            DPCT_CHECK_ERROR(h_out_ = sycl::malloc_host<float>(
                                 (size_t)CAP * H, dpct::get_in_order_queue())),
            "result staging", err, device) &&
        /*
        DPCT1048: The original value cudaHostAllocPortable is not meaningful
        in the migrated code and was removed or replaced with 0. You may need to
        check the migrated code.
        */
        check(DPCT_CHECK_ERROR(
                  h_meta_ = (void *)sycl::malloc_host(
                      sizeof(RemoteMeta), dpct::get_in_order_queue())),
              "metadata staging", err, device) &&
        check(
            DPCT_CHECK_ERROR(d_x_ = sycl::malloc_device<float>(
                                 (size_t)CAP * H, dpct::get_in_order_queue())),
            "input", err, device) &&
        check(
            DPCT_CHECK_ERROR(d_out_ = sycl::malloc_device<float>(
                                 (size_t)CAP * H, dpct::get_in_order_queue())),
            "result", err, device) &&
        check(DPCT_CHECK_ERROR(
                  d_q8_ = (uint8_t *)sycl::malloc_device(
                      (size_t)CAP * (H / 32) * 36, dpct::get_in_order_queue())),
              "activation", err, device) &&
        check(DPCT_CHECK_ERROR(
                  d_scales_ = sycl::malloc_device<float>(
                      (size_t)CAP * (H / 32), dpct::get_in_order_queue())),
              "activation scales", err, device) &&
        check(DPCT_CHECK_ERROR(d_scratch_ = (void *)sycl::malloc_device(
                                   scratch, dpct::get_in_order_queue())),
              "scratch", err, device) &&
        check(DPCT_CHECK_ERROR(
                  d_meta_ = (void *)sycl::malloc_device(
                      sizeof(RemoteMeta), dpct::get_in_order_queue())),
              "group metadata", err, device);
    if (!allocated) { close(); return false; }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0') &&
                 DPCT_CHECK_ERROR(*(void **)&z_x_ = (float *)h_x_) == 0 &&
                 DPCT_CHECK_ERROR(*(void **)&z_out_ = (float *)h_out_) == 0;
    /*
    DPCT1026: The call to cudaGetLastError was removed because this
    functionality is redundant in SYCL.
    */
    auto *meta = (RemoteMeta *)d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool RemoteExperts::begin(int64_t layer, const float *x, const int32_t *ids,
                          int64_t n_tok, int64_t k, const int32_t *kind,
                          const int32_t *primary_res, std::string &err) try {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    groups_ = (int32_t) group_id_.size();
    const dpct::queue_ptr s = stream_;
    const bool staged =
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        (zero_copy_ ||
         check(DPCT_CHECK_ERROR(
                   s->memcpy(d_x_, h_x_, (size_t)n_tok * H * sizeof(float))),
               "copy input", err, device_)) &&
        /*
        DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        check(DPCT_CHECK_ERROR(s->memcpy(d_meta_, h_meta_, sizeof(RemoteMeta))),
              "copy group metadata", err, device_);
    if (!staged) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        strata::kernels::quantize_q8_1_rows(zero_copy_ ? z_x_ : d_x_, n_tok, H, d_q8_, s);
        const auto& fmt = lay.fmt[(size_t) layer];
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, zero_copy_ ? z_out_ : d_out_, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                        groups_, (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, zero_copy_ ? z_out_ : d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t) dst_.size() * H * sizeof(float);
    /*
    DPCT1124: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    if (!zero_copy_ && !check(DPCT_CHECK_ERROR(s->memcpy(
                                  h_out_, d_out_, (size_t)compact_bytes)),
                              "copy results", err, device_)) return false;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * H * sizeof(float);
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool RemoteExperts::finish(float *out, std::string &err) try {
    if (group_id_.empty()) return true;
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    if (!check(DPCT_CHECK_ERROR(stream_->wait()), "finish", err,
               device_)) return false;
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

} // namespace strata::core
