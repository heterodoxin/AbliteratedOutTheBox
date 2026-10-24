// src/prefill/ggml_cuda_host.cu - prompt-speed plan step 2b: the host-side symbols of llama.cpp's ggml-cuda that its MMQ
// and quantize code reference, for the MMQ kernels compiled into strata_mmq without the rest of ggml-cuda.cu.
#include "common.cuh"

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

[[noreturn]] void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    std::fprintf(stderr, "ggml-cuda (strata mmq): %s: %s\n  in %s at %s:%d\n", msg, stmt, func, file, line);
    std::abort();
}

int ggml_cuda_get_device() {
    int id = 0;
    CUDA_CHECK(DPCT_CHECK_ERROR(id = dpct::get_current_device_id()));
    return id;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = [] {
        ggml_cuda_device_info in = {};
        int n = 0;
        if (DPCT_CHECK_ERROR(n = dpct::device_count()) != 0) n = 0;
        n = n > GGML_CUDA_MAX_DEVICES ? GGML_CUDA_MAX_DEVICES : n;
        in.device_count = n;
        in.physical_device_count = n;
        for (int id = 0; id < n; ++id) {
            dpct::device_info prop;
            CUDA_CHECK(
                DPCT_CHECK_ERROR(dpct::get_device(id).get_device_info(prop)));
            auto & d = in.devices[id];
            /*
            DPCT1005: The SYCL device version is different from CUDA Compute
            Compatibility. You may need to rewrite this code.
            */
            d.cc =
                100 * prop.get_major_version() + 10 * prop.get_minor_version();
            d.nsm(prop.set_max_compute_units);
            /*
            DPCT1019: local_mem_size in SYCL is not a complete equivalent of
            sharedMemPerBlock in CUDA. You may need to adjust the code.
            */
            d.smpb(prop.set_local_mem_size);
            /*
            DPCT1019: local_mem_size in SYCL is not a complete equivalent of
            sharedMemPerBlockOptin in CUDA. You may need to adjust the code.
            */
            d.smpbo(prop.set_local_mem_size);
            /*
            DPCT1006: SYCL does not provide a standard API to differentiate
            between integrated and discrete GPU devices.
            */
            d.integrated = prop.get_integrated() != 0;
            d.vmm = false;
            d.total_vram(prop.set_global_mem_size);
            d.warp_size(prop.set_max_sub_group_size);
            d.supports_cooperative_launch = prop.cooperativeLaunch != 0;
            d.physical_device = id;
            d.physical_share_count = 1;
            d.virtual_index = 0;
        }
        return in;
    }();
    return info;
}

namespace {
// Buffers are kept and reused: MMQ asks for the same few sizes every launch (its stream-k fixup tiles).
struct CachingPool : ggml_cuda_pool {
    struct Buf { void * p; size_t size; bool used; };
    std::vector<Buf> bufs;
    std::mutex mu;
    void * alloc(size_t size, size_t * actual_size) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (!b.used && b.size >= size) { b.used = true; *actual_size = b.size; return b.p; }
        void * p = nullptr;
        CUDA_CHECK(DPCT_CHECK_ERROR(
            p = (void *)sycl::malloc_device(size, dpct::get_in_order_queue())));
        bufs.push_back({p, size, true});
        *actual_size = size;
        return p;
    }
    void free(void * ptr, size_t) override {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & b : bufs)
            if (b.p == ptr) { b.used = false; return; }
    }
    ~CachingPool() override {
        for (auto & b : bufs) cudaFree(b.p);
    }
};
}  // namespace

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int, int) {
    return std::unique_ptr<ggml_cuda_pool>(new CachingPool());
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {}
