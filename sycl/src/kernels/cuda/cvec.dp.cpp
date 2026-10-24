// src/kernels/cuda/cvec.cu - see include/strata/kernels/cvec.hpp.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/cvec.hpp"

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;                 // the description; its device pointers are the uploading device's
bool g_on_host = false;
// the tables on every device that holds them (a layer split applies the vector on several)
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
int cur_device() try {
    int d = 0;
    if (DPCT_CHECK_ERROR(d = dpct::get_current_device_id()) != 0 || d < 0 ||
        d >= kDevices) d = 0;
    return d;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
bool upload_here(std::string &err) try {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;
    if (DPCT_CHECK_ERROR(t.dir = sycl::malloc_device<float>(
                             g_dir_host.size(), dpct::get_in_order_queue())) !=
            0 ||
        DPCT_CHECK_ERROR(t.s = sycl::malloc_device<float>(
                             g_s_host.size(), dpct::get_in_order_queue())) !=
            0 ||
        DPCT_CHECK_ERROR(t.on = sycl::malloc_device<int>(
                             1, dpct::get_in_order_queue())) != 0 ||
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
            t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float)).wait()) !=
            0 ||
        /*
        DPCT1114: cudaMemcpy is migrated to asynchronization memcpy,
        assuming in the original code the source host memory is pageable memory.
        If the memory is not pageable, call wait() on event return by memcpy API
        to ensure synchronization behavior.
        */
        DPCT_CHECK_ERROR(dpct::get_in_order_queue().memcpy(
            t.s, g_s_host.data(), g_s_host.size() * sizeof(float)).wait()) != 0 ||
        DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                             .memcpy(t.on, &flag, sizeof(int))
                             .wait()) != 0) {
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
    }
    return true;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// the fused hyper-connection read's gate (fused_gr.cu), so a write done here is bitwise the one it would have folded
__dpct_inline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + sycl::native::exp(-x));
}

// one block per (stream, token): the pending write, then h . v over the stream, then the update
/*
DPCT1110: The total declared local variable size in device function
cvec_kernel exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void
cvec_kernel(float *__restrict__ R, const float *__restrict__ dir,
            const float *__restrict__ s_l, const int *__restrict__ on, int mode,
            int64_t layer, int n, int hc, int64_t r_ld,
            const float *__restrict__ bo, int64_t bo_ld,
            const float *__restrict__ inj, int64_t inj_ld, int write) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int c = item_ct1.get_group(2);
    const int64_t t = item_ct1.get_group(1);
    float* r = R + t * r_ld + (int64_t) c * n;
    const float s = s_l[layer];
    const bool steer = *on != 0 && s != 0.0f;   // uniform over the block
    if (!steer && !write) return;
    const float* v = dir + layer * n;
    const float w = write ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
    const float* b = write ? bo + t * bo_ld : nullptr;
    float x[MAXK];
    float dot = 0.0f;
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item_ct1.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = r[d];
            if (write) xv = sycl::fma((float)(b[d]), (float)w, xv);
            x[k] = xv;
            if (steer && mode == 0) dot = sycl::fma(xv, (float)(v[d]), dot);
        }
    }
    if (steer && mode == 0) {
        auto &part = *sycl::ext::oneapi::group_local_memory_for_overwrite<
            float[THREADS / 32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
#pragma unroll
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        for (int o = 16; o > 0; o >>= 1) dot +=
            dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                dot, o);
        if ((item_ct1.get_local_id(2) & 31) == 0)
            part[item_ct1.get_local_id(2) >> 5] = dot;
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) < 32) {
            float p = item_ct1.get_local_id(2) < THREADS / 32
                          ? part[item_ct1.get_local_id(2)]
                          : 0.0f;
#pragma unroll
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            for (int o = 16; o > 0; o >>= 1) p +=
                dpct::experimental::permute_sub_group_by_xor(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), p, o);
            if (item_ct1.get_local_id(2) == 0) part[0] = p;
        }
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        dot = part[0] * s;   // s (h . v)
    }
#pragma unroll
    for (int k = 0; k < MAXK; ++k) {
        const int d = item_ct1.get_local_id(2) + k * THREADS;
        if (d < n) {
            float xv = x[k];
            if (steer) xv =
                mode == 0 ? sycl::fma(-dot, (float)(v[d]), xv) : xv + v[d];
            r[d] = xv;
        }
    }
}

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    int prev = 0;   // a new vector replaces the old one on every device
    prev = dpct::get_current_device_id();
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        /*
        DPCT1093: The "d" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(d);
        dpct::get_current_device().queues_wait_and_throw();
        sycl::free(g_dev[d].dir, dpct::get_in_order_queue());
        sycl::free(g_dev[d].s, dpct::get_in_order_queue());
        sycl::free(g_dev[d].on, dpct::get_in_order_queue());
        g_dev[d] = DevTables{};
    }
    /*
    DPCT1093: The "prev" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    dpct::select_device(prev);
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    int prev = 0;
    prev = dpct::get_current_device_id();
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        /*
        DPCT1093: The "d" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(d);
        dpct::get_current_device()
            .queues_wait_and_throw(); // nothing in flight may still read the
                                      // flag
        dpct::get_in_order_queue().memcpy(g_dev[d].on, &v, sizeof(int)).wait();
    }
    /*
    DPCT1093: The "prev" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    dpct::select_device(prev);
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

bool cvec_tables(const float** dir, const float** s, const int** on) {
    if (!g_cvec.loaded()) return false;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) return false;
    *dir = t.dir; *s = t.s; *on = t.on;
    return true;
}

void cvec_apply(float *R, int64_t layer, int64_t T, int64_t r_ld,
                const float *bo, int64_t bo_ld, const float *inj,
                int64_t inj_ld, bool write, void *stream) try {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& t = g_dev[cur_device()];
    if (t.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const dpct::dim3 grid((unsigned)g_cvec.hc, (unsigned)T);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto g_cvec_mode_ct4 = g_cvec.mode;
                auto g_cvec_n_embd_ct6 = (int)g_cvec.n_embd;
                auto g_cvec_hc_ct7 = (int)g_cvec.hc;

                cgh.parallel_for<dpct_kernel_name<class cvec_kernel_a17328>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            cvec_kernel(R, t.dir, t.s, t.on, g_cvec_mode_ct4,
                                        layer, g_cvec_n_embd_ct6, g_cvec_hc_ct7,
                                        r_ld, bo, bo_ld, inj, inj_ld,
                                        write ? 1 : 0);
                        });
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaPeekAtLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    if (0 != 0) throw std::runtime_error("cvec_apply: launch failed");
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
