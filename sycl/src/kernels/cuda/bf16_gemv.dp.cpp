// src/kernels/cuda/bf16_gemv.cu - the BF16 GEMV.  See the header for why it is not `s_gemv`.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

/// One thread per output row, walking it contiguously.  Kept as the naive reference the split version is
/// checked against, and used directly for the shapes where the output width is already large.
__dpct_inline__ void bf16_gemv_naive_kernel(const uint16_t *__restrict__ x,
                                            const uint16_t *__restrict__ w,
                                            float *__restrict__ y,
                                            long long n_in, long long n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long o =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (o >= n_out) return;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
#pragma unroll
    for (long long i = 0; i < n_in; ++i)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    y[o] = acc;
}

/// One WARP per output row, lanes striding the reduction axis.  For a fixed `i` consecutive lanes touch
/// consecutive addresses in this layout, so the load is coalesced - the property that took `gr_read` from
/// 262 ms/token to 8.9.
__dpct_inline__ void bf16_gemv_warp_kernel(const uint16_t *__restrict__ x,
                                           const uint16_t *__restrict__ w,
                                           float *__restrict__ y,
                                           long long n_in, long long n_out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int warps_per_block = (int)(item_ct1.get_local_range(2) >> 5);
    const long long o = (long long)item_ct1.get_group(2) * warps_per_block +
                        (item_ct1.get_local_id(2) >> 5);
    if (o >= n_out) return;
    const int lane = item_ct1.get_local_id(2) & 31;
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
#pragma unroll
    for (long long i = lane; i < n_in; i += 32)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    /*
DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc +=
        dpct::experimental::shift_sub_group_left(
            0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            acc, off);
    if (lane == 0) y[o] = acc;
}

/// TPR lanes cooperate on ONE row's reduction: `threads_per_row` threads each take a strided slice and the
/// block reduces through shared memory.  Used when the output width is too small to fill the machine.
__dpct_inline__ void bf16_gemv_split_kernel(const uint16_t *__restrict__ x,
                                            const uint16_t *__restrict__ w,
                                            float *__restrict__ y,
                                            long long n_in, long long n_out,
                                            int tpr, uint8_t *dpct_local) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto scratch = (float *)dpct_local;
    const long long o = item_ct1.get_group(2);
    if (o >= n_out) return;
    const int t = item_ct1.get_local_id(2); // 0 .. tpr-1
    const uint16_t* row = w + o * n_in;
    float acc = 0.0f;
#pragma unroll
    for (long long i = t; i < n_in; i += tpr)
        acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
    // block reduction; `tpr` is at most a few hundred, so a tree in shared is enough
    scratch[t] = acc;
    item_ct1.barrier(sycl::access::fence_space::local_space);
#pragma unroll
    for (int off = tpr >> 1; off > 0; off >>= 1) {
        if (t < off) scratch[t] += scratch[t + off];
        /*
        DPCT1118: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (t == 0) y[o] = scratch[0];
}

inline void finish(void *stream, const char *what) try {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    if (stream != nullptr) return;
    const dpct::err0 s =
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // **THIS USED TO ALWAYS USE THE NAIVE KERNEL, AND THE NAIVE KERNEL IS UNCOALESCED.**
    //
    // `bf16_gemv_naive_kernel` gives one THREAD per output row and walks the row contiguously, so at a fixed
    // `i` the 32 threads of a warp read `w[(o+k)*n_in + i]` - addresses `n_in * 2` bytes apart.  Every load in
    // the kernel's inner loop costs 32 transactions instead of 1.  The comment above that kernel claims it is
    // "used directly for the shapes where the output width is already large", but nothing ever made that
    // choice: this function called it unconditionally, at every shape.
    //
    // What that costs, from the nsys per-kernel table: **4.66 ms/token across 24 calls, 194 us each** - the
    // third largest kernel in the engine at the time, moving 1.19 GiB/token at ~210 GB/s where a plain read at
    // this geometry measures 641 GB/s (`bench/micro/p32_floor.cu`).
    //
    // `bf16_gemv_warp_kernel` exists, is checked against this one by `bf16_gemv_parity`, and is coalesced -
    // consecutive lanes touch consecutive addresses - which is the property `gr_read`'s comment records as
    // having taken it from 262 ms/token to 8.9.  The threshold below is where warp-per-row has enough rows to
    // fill the machine: 8 warps per block, so 64 rows is 8 blocks and 128 is 16, against 48 SMs.
    if (n_out >= 64) {
        const int warps = THREADS / 32;
        const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class bf16_gemv_warp_kernel_1fb910>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            bf16_gemv_warp_kernel(x, w, y, n_in, n_out);
                        });
        }
        finish(stream, "bf16_gemv(warp)");
        return;
    }
    // Below the threshold the warp kernel would leave most of the machine idle, and the naive one is at least
    // not wasting warps.  It is still uncoalesced, so this is the branch to revisit if a small-output caller
    // ever shows up hot in the profile.
    const unsigned grid = (unsigned) ((n_out + THREADS - 1) / THREADS);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class bf16_gemv_naive_kernel_7678d1>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    bf16_gemv_naive_kernel(x, w, y, n_in, n_out);
                });
    }
    finish(stream, "bf16_gemv");
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // WARP-PER-ROW when the caller asks for 32, because that path needs no shared memory and no barrier - and
    // with 48 output rows it is the configuration that fills the machine.  A `tpr` that is not 32 and not a
    // power of two is refused rather than quietly rounded, because the tree reduction below needs one.
    if (threads_per_row == 32) {
        const int warps = THREADS / 32;
        const unsigned grid = (unsigned) ((n_out + warps - 1) / warps);
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            strata::q_of(stream)
                ->parallel_for<
                    dpct_kernel_name<class bf16_gemv_warp_kernel_1f3015>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                          sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            bf16_gemv_warp_kernel(x, w, y, n_in, n_out);
                        });
        }
        finish(stream, "bf16_gemv_split(warp)");
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                             "warp-per-row path)\n", threads_per_row);
        std::exit(1);
    }
    const unsigned grid = (unsigned) n_out;
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                /*
                DPCT1083: The size of local memory in the migrated code may
                be different from the original code. Check that the allocated
                memory size in the migrated code is correct.
                */
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range((size_t)threads_per_row * sizeof(float)), cgh);

                cgh.parallel_for<
                    dpct_kernel_name<class bf16_gemv_split_kernel_adfacf>>(
                    sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                          sycl::range(1, 1, threads_per_row),
                                      sycl::range(1, 1, threads_per_row)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        bf16_gemv_split_kernel(
                            x, w, y, n_in, n_out, threads_per_row,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    finish(stream, "bf16_gemv_split");
}

}  // namespace strata::kernels
