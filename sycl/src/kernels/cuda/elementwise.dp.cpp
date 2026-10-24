// src/kernels/cuda/elementwise.cu - P2.S5's glue kernels.  See the header for why each exists.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;

__dpct_inline__ void embedding_gather_kernel(const uint8_t *__restrict__ codes,
                                             const float *__restrict__ scales,
                                             const float *__restrict__ offsets,
                                             int64_t n, int code_bits,
                                             int code_bias, int group_elems,
                                             float *__restrict__ out) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    const float product = (float)(code + code_bias) * scales[group];
    /*
    DPCT1013: The rounding mode could not be specified and the generated
    code may have different accuracy than the original code. Verify the
    correctness. SYCL math built-in function rounding mode is aligned with
    OpenCL C 1.2 standard.
    */
    out[i] = product + (offsets ? offsets[group] : 0.0f);
}

/// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.
///
/// The branch is not cosmetic. `exp(89)` overflows f32 and `exp(20)` is already 4.85e8 where `log1p` loses
/// relative precision; above 20 the function is `x` to within f32 anyway.
__dpct_inline__ float softplus_dev(float x) {
    return x > 20.0f ? x : sycl::log1p(sycl::native::exp(x));
}

__dpct_inline__ void gdn_gate_kernel(const float *__restrict__ alpha,
                                     const float *__restrict__ dt,
                                     const float *__restrict__ ssm_a,
                                     float *__restrict__ gate, int64_t h_v) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= h_v) return;
    const int64_t t = i / h_v;      // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void) t;
}

__dpct_inline__ void scale_kernel(float *__restrict__ x, int64_t n, float s) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) x[i] *= s;
}

// `dst[i] += src[i]`.  See the header: this is what lets R4's GPU half and CPU half run at the same
// time and still add up to one `parts` buffer.
__dpct_inline__ void add_kernel(float *__restrict__ dst,
                                const float *__restrict__ src, long long n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long i =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) dst[i] += src[i];
}

__dpct_inline__ void to_f16_kernel(const float *__restrict__ x,
                                   uint16_t *__restrict__ y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = f16_from_f32(x[i]);
}

__dpct_inline__ void to_bf16_kernel(const float *__restrict__ x,
                                    uint16_t *__restrict__ y, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) y[i] = bf16_from_f32(x[i]);
}

__dpct_inline__ void silu_kernel(float *__restrict__ x, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy: its arrays are f32 but `np.exp` on an f32 array is
    // computed to f32 precision by a different algorithm than `expf`, and the reference is the oracle.  The
    // difference is in the last bits and this is one line.
    const double v = (double) x[i];
    x[i] = (float)(v / (1.0 + sycl::exp(-v)));
}

/// A bounded launch: a zero-length grid is illegal, and `n == 0` is a real call (an empty selection).
inline unsigned grid_for(int64_t n) { return (unsigned) ((n + THREADS - 1) / THREADS); }

/// One WARP per row, reduced through shuffles.  The rows here are short and few - (24, 256), (2, 256),
/// (4, 128) - so a block-per-row tree would spend its time in `__syncthreads` for 8 warps of 32, and the whole
/// call is 30 rows.  A warp reduction with NO shared memory and NO barrier is the shape that fits.
__dpct_inline__ void rms_norm_weighted_kernel(float *__restrict__ x,
                                              const float *__restrict__ w,
                                              int64_t rows, int64_t cols,
                                              float eps) {

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int lane = item_ct1.get_local_id(2) & 31;
    const int64_t row =
        ((int64_t)item_ct1.get_group(2) * (item_ct1.get_local_range(2) >> 5)) +
        (item_ct1.get_local_id(2) >> 5);
    // **THE ROW GUARD IS NOT DECORATION, AND ITS ABSENCE WAS THE QSA BUG.**  The launcher rounds the grid up to
    // whole 4-warp blocks, so a call with `rows` = 2 - the QSA k-norm, the only non-multiple-of-4 row count in
    // the engine - runs EIGHT warps: rows 0 and 1 are the data, and rows 2 and 3 write 2*cols floats PAST the
    // end of `b.kcur`, which in the arena is exactly where `b.vcur` begins.  `b.vcur` was therefore silently
    // replaced by two rows of `rms_norm(...) * attn_k_norm`: a normalized vector with l2 ~20.6 against V's
    // ~5.7, which the attention then attended to.  `rope_neox_kernel` has had this guard all along.
    if (row >= rows) return;
    float* r = x + row * cols;
    float acc = 0.0f;
#pragma unroll
    for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
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
    // The MEAN, not the sum: `ref/qsa.py::rms_norm` divides by `np.mean(np.square(x))`.  Broadcasting the
    // reciprocal from lane 0 keeps all 32 lanes on the same value - computing `rsqrt` per lane would be the
    // same number but a needless 32-way divergence in the last bit.
    float inv = 0.0f;
    if (lane == 0) inv = sycl::rsqrt(acc / (float)cols + eps);
    /*
    DPCT1108: '__shfl_sync' was migrated with the experimental feature masked
    sub_group function which may not be supported by all compilers or runtimes.
    You may need to adjust the code.
    */
    inv = dpct::experimental::select_from_sub_group(
        0xFFFFFFFFu, sycl::ext::oneapi::this_work_item::get_sub_group(), inv,
        0);
#pragma unroll
    for (int64_t c = lane; c < cols; c += 32)
        r[c] = (w ? r[c] * w[c] : r[c]) * inv;
}

void sync_if_needed(void *stream, const char *what) try {
    if (stream != nullptr) return;
    const dpct::err0 e =
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw());
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

bool check_launch(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    return true;
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class embedding_gather_kernel_40b387>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid_for(n)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    embedding_gather_kernel(codes, scales, offsets, n,
                                            code_bits, code_bias, group_elems,
                                            out);
                });
    }
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class gdn_gate_kernel_de22c0>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid_for(n)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    gdn_gate_kernel(alpha, dt, ssm_a, gate, h_v);
                });
    }
    check_launch("gdn_gate");
    sync_if_needed(stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class scale_kernel_501571>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid_for(n)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    scale_kernel(x, n, s);
                });
    }
    check_launch("scale_inplace");
    sync_if_needed(stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class add_kernel_95886e>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid_for(n)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    add_kernel(dst, src, n);
                });
    }
    check_launch("add_inplace");
    sync_if_needed(stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class to_f16_kernel_ba0968>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid_for(n)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    to_f16_kernel(x, y, n);
                });
    }
    check_launch("f32_to_f16_bulk");
    sync_if_needed(stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class to_bf16_kernel_7b22d1>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid_for(n)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    to_bf16_kernel(x, y, n);
                });
    }
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp64});

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class silu_kernel_4d1386>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid_for(n)) *
                                      sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    silu_kernel(x, n);
                });
    }
    check_launch("silu_inplace");
    sync_if_needed(stream, "silu_inplace");
}

/// THE DOORBELL.  One thread, one INCREMENT - the cost is the launch, and inside a graph that is paid once.
///
/// **IT INCREMENTS THE MEMORY, AND IT DOES NOT TAKE THE VALUE AS AN ARGUMENT.**  The first version did -
/// doorbell_ring(d_seq, *(h_seq) + 1u) - and that is a CLONED LITERAL: the expression is evaluated on the HOST
/// at CAPTURE time, so the graph rings 1 on every replay and a host waiting for a change waits forever.  It
/// would have looked like a working doorbell on the first token, which is the worst way for it to be wrong.
/// **THE FENCE IS NOT DECORATION, AND ITS ABSENCE WAS COSTING ~10 ms PER TOKEN ON THE HOST SIDE.**
///
/// The ring PUBLISHES three buffers the host is about to read - `h_x_f`, `h_ids`, `h_weights` - so the increment
/// must be ordered after their writes, or the host can observe the ring and then read a payload that has not
/// landed.  `__threadfence_system()` is what orders them, and it covers the HOST as well as the device, which is
/// the whole point: the reader is the CPU.
///
/// Without it the host loop was compensating with a driver call.  `session_loop` polled with `cudaEventQuery`
/// on EVERY spin iteration, because round 195 had measured that a memory-only spin never saw the datum flip -
/// and that measurement was right about the symptom and wrong about the cause.  The cause is this missing fence:
/// the write was not ordered into host-visible memory, so no amount of reading it would show it, and the driver
/// call was flushing the whole pipeline enough to make it appear.  A 10-22 us driver call per iteration is a
/// very expensive substitute for one fence instruction.
///
/// **THE STORE IS VOLATILE**, like `doorbell_publish_kernel`'s.  On RDNA4 (gfx1201) a plain store to mapped pinned
/// memory stays in the GPU's L2 until the stream is synchronized - the host never saw the ring (tests/hip/handoff:
/// 0 of 100 rings seen without a sync; volatile, a system-scope atomic store or a fence after the store: 100 of 100).
__dpct_inline__ void doorbell_ring_kernel(uint32_t *seq) {
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
    strata::sys_store(seq, strata::sys_load(seq) + 1u);
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
    __threadfence_system();
#endif
}

__dpct_inline__ void doorbell_wait_kernel(const volatile uint32_t *flag,
                                          const volatile uint32_t *seq, uint32_t spin_max) {
    const uint32_t want = strata::sys_load(seq);
    for (uint32_t spin = 0; spin < spin_max && strata::sys_load(flag) != want; ++spin) strata_spin_pause();
    /*
    DPCT1078: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    const uint32_t spin_max = strata::spin_max(*strata::q_of(stream));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class doorbell_wait_kernel_47b360>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    doorbell_wait_kernel(d_flag, d_seq, spin_max);
                });
    }
    check_launch("doorbell_wait");
}

/*
DPCT1052: SYCL does not support the member access for a volatile qualified
vector type. The volatile qualifier was removed. You may need to rewrite the
code.
*/
__dpct_inline__ void copy_from_mapped_kernel(sycl::float4 *__restrict__ dst,
                                             const sycl::float4 *src,
                                             int64_t n4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    for (int64_t i =
             (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
             item_ct1.get_local_id(2);
         i < n4; i += (int64_t)item_ct1.get_group_range(2) *
                      item_ct1.get_local_range(2)) {
        const sycl::float4 v = strata::load_mapped_float4(src + i);
        dst[i] = v;
    }
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself (the
// pool writes +0.0 into those, so this writes +0.0 too): block = row, the plan's hit rows `dst[0, *count)`.
/*
DPCT1052: SYCL does not support the member access for a volatile qualified
vector type. The volatile qualifier was removed. You may need to rewrite the
code.
*/
__dpct_inline__ void copy_rows_from_mapped_kernel(
    sycl::float4 *__restrict__ dst, const sycl::float4 *src, int64_t row4,
    const int32_t *__restrict__ hit_rows, const int32_t *__restrict__ count) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int row = item_ct1.get_group(2);
    auto &hit = *sycl::ext::oneapi::group_local_memory_for_overwrite<int>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    if (item_ct1.get_local_id(2) == 0) {
        int h = 0;
        const int c = *count;
#pragma unroll
        for (int i = 0; i < c; ++i) h |= hit_rows[i] == row;
        hit = h;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    sycl::float4 *d = dst + (int64_t)row * row4;
    if (hit) {
#pragma unroll
        for (int64_t i = item_ct1.get_local_id(2); i < row4;
             i += item_ct1.get_local_range(2))
            d[i] = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else {
        /*
        DPCT1052: SYCL does not support the member access for a volatile
        qualified vector type. The volatile qualifier was removed. You may need
        to rewrite the code.
        */
        const sycl::float4 *sr = src + (int64_t)row * row4;
#pragma unroll
        for (int64_t i = item_ct1.get_local_id(2); i < row4;
             i += item_ct1.get_local_range(2))
            d[i] = strata::load_mapped_float4(sr + i);
    }
}
namespace {
__dpct_inline__ void scatter_rows_kernel(const sycl::float4 *__restrict__ src,
                                         sycl::float4 *dst,
                                         const int32_t *__restrict__ rows,
                                         int64_t w4) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t r = item_ct1.get_group(2);
    const sycl::float4 *s = src + r * w4;
    sycl::float4 *d = dst + (int64_t)rows[r] * w4;
#pragma unroll
    for (int64_t i = item_ct1.get_local_id(2); i < w4;
         i += item_ct1.get_local_range(2)) d[i] = s[i];
}
}  // namespace
void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "scatter_rows_f32: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto width_ct3 = width / 4;

                cgh.parallel_for<
                    dpct_kernel_name<class scatter_rows_kernel_7d83a1>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        scatter_rows_kernel((const sycl::float4 *)src,
                                            (sycl::float4 *)dst, rows,
                                            width_ct3);
                    });
            });
    }
}
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto width_ct2 = width / 4;

                cgh.parallel_for<dpct_kernel_name<
                    class copy_rows_from_mapped_kernel_9530d4>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)rows) *
                                          sycl::range(1, 1, 128),
                                      sycl::range(1, 1, 128)),
                    exp_props, [=](sycl::nd_item<3> item_ct1) {
                        copy_rows_from_mapped_kernel(
                            (sycl::float4 *)dst,
                            (const sycl::float4 *)src, width_ct2,
                            hit_rows, count);
                    });
            });
    }
}
void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_from_mapped_kernel_82951a>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_from_mapped_kernel((sycl::float4 *)dst,
                                            (const sycl::float4 *)src,
                                            n4);
                });
    }
    check_launch("copy_from_mapped");
}

namespace {
// The published payload's checksum: the wrapping sum of every 32-bit word mixed with its position (murmur3's
// finalizer).  A plain sum of the words missed a reordering ([3, 7] -> [7, 3] in the expert ids) and cancelling
// changes; mixed, a stale payload passes only with a chance near 2^-32 per read.  It detects; it does not order -
// the host keeps reading until the payload it sees matches (doorbell_payload_ready).  Positions: x 0..n-1,
// ids n..n+k-1, weights n+k..n+2k-1.
inline uint32_t dbx_payload_mix(uint32_t word, uint32_t pos) {
    uint32_t h = word ^ (pos * 0x9E3779B9u);
    h ^= h >> 16; h *= 0x85EBCA6Bu; h ^= h >> 13; h *= 0xC2B2AE35u; h ^= h >> 16;
    return h;
}
inline uint32_t dbx_payload_sum(const float* x, int64_t n, const int32_t* ids, const float* w, int64_t k,
                                bool has_weights) {
    uint32_t s = 0;
    for (int64_t j = 0; j < n; ++j) s += dbx_payload_mix(sycl::bit_cast<uint32_t>(x[j]), (uint32_t) j);
    for (int64_t j = 0; j < k; ++j) s += dbx_payload_mix((uint32_t) ids[j], (uint32_t) (n + j));
    if (has_weights)
        for (int64_t j = 0; j < k; ++j) s += dbx_payload_mix(sycl::bit_cast<uint32_t>(w[j]), (uint32_t) (n + k + j));
    return s;
}
// STRATA_SYCL_A770_FAST=1 (opt-in): pair the activation stores without changing the checksum's word/position
// order. Odd sizes or unaligned mapped pointers retain the scalar path.
__dpct_inline__ uint32_t dbx_publish_x(const float* x, float* out, int n, int lane, int threads, bool pair, bool plain) {
    uint32_t part = 0;
    if (pair && !plain && ((uintptr_t)out & 7u) == 0) {
        for (int j = 2 * lane; j + 1 < n; j += 2 * threads) {
            const uint32_t a = sycl::bit_cast<uint32_t>(x[j]);
            const uint32_t b = sycl::bit_cast<uint32_t>(x[j + 1]);
            strata::sys_store_mapped(reinterpret_cast<uint64_t*>(out + j),
                                     (uint64_t)a | ((uint64_t)b << 32));
            part += dbx_payload_mix(a, (uint32_t)j) + dbx_payload_mix(b, (uint32_t)(j + 1));
        }
        if ((n & 1) && lane == 0) {
            strata::sys_store_mapped(plain, out + n - 1, x[n - 1]);
            part += dbx_payload_mix(sycl::bit_cast<uint32_t>(x[n - 1]), (uint32_t)(n - 1));
        }
    } else {
        for (int j = lane; j < n; j += threads) {
            strata::sys_store_mapped(plain, out + j, x[j]);
            part += dbx_payload_mix(sycl::bit_cast<uint32_t>(x[j]), (uint32_t)j);
        }
    }
    return part;
}
// The ring, its tag and its checksum, in slot r % 4 of the sequence word (elementwise.hpp).  Written last, after a
// system fence over the uncached payload stores.
__dpct_inline__ void dbx_ring_commit(uint32_t* seq, uint32_t part, bool group_ok, uint32_t ring) {
#if defined(__HIPCC__)  // #697: HIP only; on RDNA4 the volatile store alone can sit in L2 until the stream syncs
    (void) group_ok;
#endif
    if (group_ok) {
        strata::sys_store(seq + 2 + 2 * (ring % 4), part);
        strata::sys_store(seq + 1 + 2 * (ring % 4), ring);
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        strata::sys_store(seq, ring);
    }
}
// The payload store (strata::sys_store_mapped(plain, ...)): uncached (sycl_doorbell.hpp) so a host that polls the ring while the kernel still runs reads a
// whole payload, or a plain store when every reader waits for the kernel to end (doorbell_plain_payload: the stepped
// verify window reads a segment's payload only after the segment has drained, and the end of a kernel makes its
// stores visible). The host checksum (doorbell_payload_ready) still guards every read.
std::atomic<bool> g_plain_payload{false};
}  // namespace

void doorbell_plain_payload(bool on) { g_plain_payload.store(on, std::memory_order_relaxed); }

__dpct_inline__ void doorbell_publish_kernel(const float *__restrict__ x,
                                             const int32_t *__restrict__ ids,
                                             const float *__restrict__ w, int n,
                                             int k, float *x_out,
                                             int32_t *ids_out, float *w_out,
                                             uint32_t *seq, bool plain, bool pair) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item_ct1.get_local_id(2);
    const int nt = (int) item_ct1.get_local_range(2);
    uint32_t part = dbx_publish_x(x, x_out, n, i, nt, pair, plain);
    if (i < k) {
        const int32_t id = ids[i];
        strata::sys_store_mapped(plain, ids_out + i, id);
        part += dbx_payload_mix((uint32_t) id, (uint32_t) (n + i));
        if (w_out != nullptr) {
            const float wv = w[i];
            strata::sys_store_mapped(plain, w_out + i, wv);
            part += dbx_payload_mix(sycl::bit_cast<uint32_t>(wv), (uint32_t) (n + k + i));
        }
    }
    const uint32_t sum = sycl::reduce_over_group(item_ct1.get_group(), part, sycl::plus<uint32_t>());
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    item_ct1.barrier();
    if (i == 0) {
        const uint32_t next = strata::sys_load(seq) + 1u;
        dbx_ring_commit(seq, sum, true, next);
    }
}

__dpct_inline__ void doorbell_publish_res_kernel(
    const float *__restrict__ x, const int32_t *__restrict__ ids,
    const int32_t *__restrict__ d_res, int n_expert, int n, int k, float *x_out,
    int32_t *ids_out, uint32_t *seq, bool full, bool plain, bool pair) {
    // full (STRATA_DOORBELL_CHECK=1): the payload is published whole and the checksum covers x and the ids (this
    // variant publishes no weights). Otherwise d_res/n_expert decide whether x is copied, as before the checksum:
    // when every routed expert is resident the host never reads x, and an uncached copy of it costs the A750 ~10%
    // of decode.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    if (!full) {
        int any_miss = 0;
        if ((int) item_ct1.get_local_id(2) < k) {
            const int32_t id = ids[item_ct1.get_local_id(2)];
            strata::sys_store_mapped(plain, ids_out + item_ct1.get_local_id(2), id);
            if (d_res == nullptr || id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
        }
        if ((item_ct1.barrier(), sycl::any_of_group(sycl::ext::oneapi::this_work_item::get_work_group<3>(), any_miss))) {
            for (int j = (int) item_ct1.get_local_id(2); j < n; j += (int) item_ct1.get_local_range(2))
                strata::sys_store_mapped(plain, x_out + j, x[j]);
        }
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        item_ct1.barrier();
        if (item_ct1.get_local_id(2) == 0) {
            const uint32_t next = strata::sys_load(seq) + 1u;
            dbx_ring_commit(seq, 0u, true, next);
        }
        return;
    }
    const int i = (int) item_ct1.get_local_id(2);
    const int nt = (int) item_ct1.get_local_range(2);
    uint32_t part = dbx_publish_x(x, x_out, n, i, nt, pair, plain);
    if (i < k) {
        const int32_t id = ids[i];
        strata::sys_store_mapped(plain, ids_out + i, id);
        part += dbx_payload_mix((uint32_t) id, (uint32_t) (n + i));
    }
    const uint32_t sum = sycl::reduce_over_group(item_ct1.get_group(), part, sycl::plus<uint32_t>());
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    item_ct1.barrier();
    if (i == 0) {
        const uint32_t next = strata::sys_load(seq) + 1u;
        dbx_ring_commit(seq, sum, true, next);
    }
}

// #649 (HIP, STRATA_DOORBELL_STORE=1): the same publish, but the ring is STORED (the step's own number, known at
// capture) instead of read-modify-written over PCIe - one store, no read of host memory from the GPU.
__dpct_inline__ void doorbell_publish_value_kernel(
    const float *__restrict__ x, const int32_t *__restrict__ ids,
    const float *__restrict__ w, int n, int k, float *x_out, int32_t *ids_out,
    float *w_out, uint32_t *seq, uint32_t value, bool plain, bool pair) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int i = (int) item_ct1.get_local_id(2);
    const int nt = (int) item_ct1.get_local_range(2);
    uint32_t part = dbx_publish_x(x, x_out, n, i, nt, pair, plain);
    if (i < k) {
        const int32_t id = ids[i];
        strata::sys_store_mapped(plain, ids_out + i, id);
        part += dbx_payload_mix((uint32_t) id, (uint32_t) (n + i));
        const float wv = w[i];
        strata::sys_store_mapped(plain, w_out + i, wv);
        part += dbx_payload_mix(sycl::bit_cast<uint32_t>(wv), (uint32_t) (n + k + i));
    }
    const uint32_t sum = sycl::reduce_over_group(item_ct1.get_group(), part, sycl::plus<uint32_t>());
    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    item_ct1.barrier();
    if (i == 0) dbx_ring_commit(seq, sum, true, value);
}

void doorbell_publish_value(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k,
                            float* x_out, int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value,
                            void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    const bool plain = strata::doorbell_plain_payload() || g_plain_payload.load(std::memory_order_relaxed);
    const bool pair = strata::a770_fast();
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class doorbell_publish_value_kernel_9bdb22>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    doorbell_publish_value_kernel(x, ids, weights, (int)n,
                                                  (int)k, x_out, ids_out,
                                                  weights_out, d_seq, value, plain, pair);
                });
    }
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    const bool plain = strata::doorbell_plain_payload() || g_plain_payload.load(std::memory_order_relaxed);
    const bool pair = strata::a770_fast();
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class doorbell_publish_kernel_8b5bad>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    doorbell_publish_kernel(x, ids, weights, (int)n, (int)k,
                                            x_out, ids_out, weights_out, d_seq, plain, pair);
                });
    }
    check_launch("doorbell_publish");
}

void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k,
                          float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish_res: k too large\n"); std::exit(1); }
    static const bool full = [] { const char* v = std::getenv("STRATA_DOORBELL_CHECK"); return v != nullptr && v[0] == '1'; }();
    const bool fl = full;
    const bool plain = strata::doorbell_plain_payload() || g_plain_payload.load(std::memory_order_relaxed);
    const bool pair = strata::a770_fast();
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class doorbell_publish_res_kernel_85b9b3>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1024),
                                  sycl::range(1, 1, 1024)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    doorbell_publish_res_kernel(x, ids, d_res, n_expert, (int)n,
                                                (int)k, x_out, ids_out, d_seq, fl, plain, pair);
                });
    }
    check_launch("doorbell_publish_res");
}

bool doorbell_payload_ready(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                            const float* h_weights, int64_t k, uint32_t want) {
    const volatile uint32_t* w = h_seq + 1 + 2 * (want % 4);   // this ring's tag and checksum (elementwise.hpp)
    if (w[0] != want) return false;
    std::atomic_thread_fence(std::memory_order_acquire);
    // the two publish shapes: with the routing weights (doorbell_publish / _value) or without (doorbell_publish_res)
    return dbx_payload_sum(h_x, n, h_ids, h_weights, k, true) == w[1] ||
           dbx_payload_sum(h_x, n, h_ids, h_weights, k, false) == w[1];
}

bool doorbell_wait_payload(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                           const float* h_weights, int64_t k, uint32_t want, int timeout_ms) {
    // Opt-in (STRATA_DOORBELL_CHECK=1): on an Arc A750 (i915) the host-side checksum of the whole payload, read from
    // uncached host memory twice per CPU-expert layer, cost 15-20% of decode (5 interleaved rounds), and the stale
    // payload it guards against was not seen there. The ring tag is still published by the GPU.
    static const bool check = [] { const char* v = std::getenv("STRATA_DOORBELL_CHECK"); return v != nullptr && v[0] == '1'; }();
    if (!check) return true;
    if (doorbell_payload_ready(h_seq, h_x, n, h_ids, h_weights, k, want)) return true;
    const auto t0 = std::chrono::steady_clock::now();
    while (!doorbell_payload_ready(h_seq, h_x, n, h_ids, h_weights, k, want)) {
        for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) return false;
    }
    return true;
}

__dpct_inline__ void copy_i32_from_mapped_kernel(int32_t *__restrict__ dst,
                                                 const volatile int32_t *src,
                                                 int n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
#pragma unroll
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) dst[i] = src[i];
}

#if !defined(__HIPCC__)
struct MappedCopies {
    MappedCopy c[8];
};

__dpct_inline__ void copy_from_mapped_multi_kernel(MappedCopies a) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const MappedCopy c = a.c[item_ct1.get_group(1)];
    const int64_t i0 = (int64_t)item_ct1.get_group(2) *
                           item_ct1.get_local_range(2) +
                       item_ct1.get_local_id(2),
                  st = (int64_t)item_ct1.get_group_range(2) *
                       item_ct1.get_local_range(2);
    if ((c.words & 3) == 0 && ((uintptr_t) c.dst & 15) == 0 && ((uintptr_t) c.src & 15) == 0) {
        /*
        DPCT1052: SYCL does not support the member access for a volatile
        qualified vector type. The volatile qualifier was removed. You may need
        to rewrite the code.
        */
        const sycl::float4 *s = (const sycl::float4 *)c.src;
#pragma unroll
        for (int64_t i = i0; i < c.words / 4; i += st)(
            (sycl::float4 *)c.dst)[i] = const_cast<const sycl::float4 *>(s)[i];
    } else {
        const volatile int32_t* s = (const volatile int32_t*) c.src;
#pragma unroll
        for (int64_t i = i0; i < c.words; i += st)((int32_t *)c.dst)[i] = s[i];
    }
}

void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream) {
    if (n <= 0) return;
    if (n > 8) { std::fprintf(stderr, "copy_from_mapped_multi: at most 8 copies\n"); std::exit(1); }
    MappedCopies a;
    int64_t most = 0;
    for (int i = 0; i < n; ++i) {
        a.c[i] = copies[i];
        most = copies[i].words > most ? copies[i].words : most;
    }
    const int64_t units = (most + 3) / 4;
    const unsigned bx = (unsigned) ((units + 255) / 256 < 32 ? (units + 255) / 256 : 32);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_from_mapped_multi_kernel_66c71e>>(
                sycl::nd_range<3>(sycl::range(1, (unsigned)n, bx < 1 ? 1 : bx) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_from_mapped_multi_kernel(a);
                });
    }
    check_launch("copy_from_mapped_multi");
}
#endif

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class copy_i32_from_mapped_kernel_a44acc>>(
                sycl::nd_range<3>(sycl::range(1, 1, 128),
                                  sycl::range(1, 1, 128)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy_i32_from_mapped_kernel(
                        dst, (const volatile int32_t *)src, (int)n);
                });
    }
    check_launch("copy_i32_from_mapped");
}

#if defined(__HIPCC__)   // AMD keeps one copy a launch
void copy_from_mapped_multi(const MappedCopy* copies, int n, void* stream) {
    for (int i = 0; i < n; ++i)
        copy_i32_from_mapped((int32_t*) copies[i].dst, (const int32_t*) copies[i].src, copies[i].words, stream);
}
#endif

void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class doorbell_ring_kernel_b862a4>>(
                sycl::nd_range<3>(sycl::range(1, 1, 1), sycl::range(1, 1, 1)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    doorbell_ring_kernel(d_seq);
                });
    }
    check_launch("doorbell_ring");
    sync_if_needed(stream, "doorbell_ring");
}

void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    // 4 warps per block, so a row count that is not a multiple of 4 wastes at most 3 warps rather than
    // launching a block per row for a 2-row call.
    const unsigned warps_per_block = 4;
    const unsigned grid = (unsigned) ((rows + warps_per_block - 1) / warps_per_block);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<
                dpct_kernel_name<class rms_norm_weighted_kernel_b603f5>>(
                sycl::nd_range<3>(sycl::range(1, 1, grid) *
                                      sycl::range(1, 1, warps_per_block * 32),
                                  sycl::range(1, 1, warps_per_block * 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        rms_norm_weighted_kernel(x, w, rows, cols, eps);
                    });
    }
    check_launch("rms_norm_weighted");
    sync_if_needed(stream, "rms_norm_weighted");
}

}  // namespace strata::kernels
