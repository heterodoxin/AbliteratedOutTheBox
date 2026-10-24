// src/kernels/cuda/router_top10.cu - P2.S2: the MoE router.
//
// P2.S2's spec: "BF16 GEMV, softmax / top-k / renormalize per docs/semantics.md; emits (expert_id, weight) x 10
// per token".  This is the softmax/top-k/renormalize half; the BF16 GEMV that produces the logits is a
// separate kernel and is NOT here.
//
// The semantics are transcribed from `ref/moe.py::router`, which is itself transcribed from llama-graph.cpp:
//
//     p   = softmax(logits)                       over ALL experts, not over the selected subset
//     ids = stable argsort(-p)[:k]                ties break by INDEX, ascending
//     w   = gather(p, ids)
//     s   = max(sum(w), 2**-14)                   ggml_clamp(..., 2**-14, INF)
//     return ids, w / s
//
// The two-step structure matters and is not cosmetic: computing softmax over the selected subset instead is
// mathematically identical (softmax is shift-invariant) but it moves the renormalisation, and the CLAMP is
// part of the renormalisation.  Reproducing the gather form is what makes the clamp land in the same place.
//
// WHY THIS IS NO LONGER "NAIVE BY DESIGN", AND WHAT THE OLD REASONING GOT WRONG.
//
// The first version was ONE THREAD PER TOKEN: k passes over the experts, each pass recomputing
// `exp((double) l[e] - mx)` for every expert.  Its comment justified that with
//
//     "5,120 operations per token against 2.36e9 weights of expert matvec - three orders of magnitude smaller
//      than the thing it feeds, so the simplest correct version is also fast enough"
//
// **That is a statement about OPERATION COUNT and it says nothing about TIME, because all 5,120 operations were
// on ONE THREAD while the matvec has thousands.**  Measured (round 218, stage-by-stage inside one block):
//
//     moe_layer   4.077 ms      router_top10   3.387 ms      bf16 gemv   0.270 ms
//
// 3.39 ms per layer x 48 layers = 163 ms of a 289 ms token - **56% of the whole forward pass, in a kernel that
// reads 2.6 MB.**  It was invisible in `router_top10_parity` because that test runs the kernel ONCE and checks
// the right ids, and invisible in every per-layer test for the same reason.
//
// WHAT IS PARALLELISED AND WHAT IS DELIBERATELY NOT:
//
//   * the MAX is a tree reduction of `fmaxf`, which is exact and order-independent - bit-identical to the
//     serial scan it replaces;
//   * the 512 EXPONENTIALS are computed ONCE each, in parallel.  The old kernel computed them ELEVEN TIMES
//     (once for the softmax, then again inside every one of the k passes).  This is where the 3.4 ms was;
//   * the DOUBLE SUM is still accumulated on ONE thread in ASCENDING order.  512 double adds is about a
//     microsecond and it keeps the accumulation order - and therefore the last bits - identical to the
//     reference-faithful serial version.  Parallelising it would be a free speedup and a silent change to the
//     values, and the exp was the cost, not this;
//   * the SELECTION is by RANK rather than by k repeated maxima:
//
//         rank(e) = #{ f : p[f] > p[e] }  +  #{ f < e : p[f] == p[e] }
//
//     which IS "stable descending argsort, ties by ascending index" - the same order the repeated-maximum loop
//     produced - computed in ONE parallel pass over the experts instead of k serial ones.  It is O(n^2)
//     comparisons and that is the right trade here: n = 512, every comparison is independent, and the old
//     version was O(k*n) with a serial `exp` inside.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/sycl_doorbell.hpp"
#include "strata/kernels/router_top10.hpp"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;

// The fork's parity-tested SYCL implementation of the HIP fast router below.
// A tree sum brackets the serial reciprocal; ambiguous rounding falls back to
// the original ascending sum. Selection and renormalisation stay in subgroup 0.
template <bool FORCE_SERIAL>
void launch_sycl_fast(sycl::queue& q, const float* logits, int n_tokens, int n_expert, int k,
                      int* ids, float* weights) {
    const int threads = (std::min(n_expert, RT_MAX_THREADS) + 31) & ~31;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> s_ex(sycl::range<1>(n_expert), h);
        sycl::local_accessor<float, 1> s_red(sycl::range<1>(RT_MAX_THREADS / 32), h);
        sycl::local_accessor<double, 1> s_dred(sycl::range<1>(RT_MAX_THREADS / 32), h);
        sycl::local_accessor<float, 1> s_inv(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)n_tokens * threads),
                                        sycl::range<1>(threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int t = it.get_group(0), tid = it.get_local_id(0);
                const int lane = tid & 31, warp = tid >> 5;
                const int nt = it.get_local_range(0), nw = (nt + 31) >> 5;
                const float* l = logits + (size_t)t * n_expert;
                auto bar = [&] { sycl::group_barrier(it.get_group()); };
                float mx = -INFINITY;
                for (int e = tid; e < n_expert; e += nt) mx = sycl::fmax(mx, l[e]);
                for (int off = 16; off > 0; off >>= 1)
                    mx = sycl::fmax(mx, sycl::shift_group_left(sg, mx, off));
                if (lane == 0) s_red[warp] = mx;
                bar();
                if (tid < 32) {
                    float v = tid < nw ? s_red[tid] : -INFINITY;
                    for (int off = 16; off > 0; off >>= 1)
                        v = sycl::fmax(v, sycl::shift_group_left(sg, v, off));
                    if (tid == 0) s_red[0] = v;
                }
                bar();
                mx = s_red[0];
                double part = 0.0;
                for (int e = tid; e < n_expert; e += nt) {
                    const double x = sycl::exp((double)l[e] - (double)mx);
                    s_ex[e] = x;
                    part += x;
                }
                for (int off = 16; off > 0; off >>= 1)
                    part += sycl::permute_group_by_xor(sg, part, off);
                if (lane == 0) s_dred[warp] = part;
                bar();
                double tot = 0.0;
                for (int w = 0; w < nw; ++w) tot += s_dred[w];
                const double lo = tot * (1.0 - 0x1p-40), hi = tot * (1.0 + 0x1p-40);
                const float f_lo = (float)(1.0 / hi), f_hi = (float)(1.0 / lo);
                float inv;
                if (!FORCE_SERIAL && tot >= 1.0 && hi < 1e300 && f_lo == f_hi) {
                    inv = f_lo;
                } else {
                    if (tid == 0) {
                        double sum = 0.0;
                        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
                        s_inv[0] = (float)(1.0 / sum);
                    }
                    bar();
                    inv = s_inv[0];
                }
                if (warp != 0) return;
                constexpr int PER = 16;
                float pv[PER];
                unsigned live = 0;
                for (int j = 0; j < PER; ++j) {
                    const int e = lane + 32 * j;
                    pv[j] = -INFINITY;
                    if (e < n_expert) { pv[j] = (float)(s_ex[e] * inv); live |= 1u << j; }
                }
                float my_w = 0.0f;
                int my_id = 0, nsel = 0;
                for (int i = 0; i < k; ++i) {
                    float bv = -INFINITY;
                    int bi = n_expert;
                    for (int j = 0; j < PER; ++j)
                        if (((live >> j) & 1u) && pv[j] > bv) { bv = pv[j]; bi = lane + 32 * j; }
                    for (int off = 16; off > 0; off >>= 1) {
                        const float ov = sycl::permute_group_by_xor(sg, bv, off);
                        const int oi = sycl::permute_group_by_xor(sg, bi, off);
                        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                    }
                    if (bi >= n_expert) break;
                    if ((bi & 31) == lane) live &= ~(1u << (bi >> 5));
                    if (lane == i) { my_w = bv; my_id = bi; }
                    nsel = i + 1;
                }
                if (nsel == k) {
                    double s = 0.0;
                    for (int i = 0; i < k; ++i) s += (double)sycl::select_from_group(sg, my_w, i);
                    const double sc = sycl::fmax(s, 6.103515625e-05);
                    if (lane < k) {
                        ids[(size_t)t * k + lane] = my_id;
                        weights[(size_t)t * k + lane] = (float)((double)my_w / sc);
                    }
                } else {
                    if (lane < nsel) { ids[(size_t)t * k + lane] = my_id; weights[(size_t)t * k + lane] = my_w; }
                    sycl::group_barrier(sg);
                    if (lane == 0) {
                        double s = 0.0;
                        for (int i = 0; i < k; ++i) s += (double)weights[(size_t)t * k + i];
                        const double sc = sycl::fmax(s, 6.103515625e-05);
                        for (int i = 0; i < k; ++i)
                            weights[(size_t)t * k + i] = (float)((double)weights[(size_t)t * k + i] / sc);
                    }
                }
            });
    });
}

/// One BLOCK per token, so the reductions have somewhere to happen.  `n_tokens` is 1 in decode; the grid keeps
/// the batch case working without a second code path.
/*
DPCT1110: The total declared local variable size in device function
router_top10_kernel exceeds 128 bytes and may cause high register pressure.
Consult with your hardware vendor to find the total register size available and
adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void router_top10_kernel(const float *__restrict__ logits,
                                         int n_tokens, int n_expert, int k,
                                         int *__restrict__ ids,
                                         float *__restrict__ weights,
                                         uint8_t *dpct_local) {
    // **`s_taken` IS FIRST SO THE OTHER TWO KEEP THEIR ALIGNMENT WITHOUT AN OFFSET PARAMETER**, and `s_p`'s
    // existing `s_ex + n_expert` stays correct because it is relative to `s_ex`.  One byte per expert.
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    auto s_raw = (unsigned char *)dpct_local;
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    unsigned char* s_taken = s_raw;
    double* s_ex = (double*) (s_raw + taken_bytes);
    auto &s_red = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[RT_MAX_THREADS / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_rid = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        int[RT_MAX_THREADS / 32]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &s_sum = *sycl::ext::oneapi::group_local_memory_for_overwrite<double>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());

    const int t = item_ct1.get_group(2);
    if (t >= n_tokens) return;
    const int tid = item_ct1.get_local_id(2);
    const int nt = item_ct1.get_local_range(2);
    const float* l = logits + (size_t) t * n_expert;

    // ---- softmax over ALL experts, for stability: the max.  A tree of `fmaxf` is EXACT and order-independent,
    // so this is bit-identical to the serial scan.
    float mx = -INFINITY;
#pragma unroll
    for (int e = tid; e < n_expert; e += nt) mx = sycl::fmax(mx, l[e]);
    /*
DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) mx = sycl::fmax(
        mx, dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                mx, off));
    if ((tid & 31) == 0) s_red[tid >> 5] = mx;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tid < 32) {
        const int nw = (nt + 31) >> 5;
        float v = (tid < nw) ? s_red[tid] : -INFINITY;
        /*
DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
masked sub_group function which may not be supported by all compilers or
runtimes. You may need to adjust the code.
*/
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = sycl::fmax(
            v, dpct::experimental::shift_sub_group_left(
                   0xffffffffu,
                   sycl::ext::oneapi::this_work_item::get_sub_group(), v, off));
        if (tid == 0) s_red[0] = v;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    mx = s_red[0];

    // ---- THE 512 EXPONENTIALS, ONCE EACH AND IN PARALLEL.  `exp` in double is software-emulated on this die
    // and was 5,632 serial calls before; it is 512 parallel ones now.
#pragma unroll
    for (int e = tid; e < n_expert; e += nt)
        s_ex[e] = sycl::exp((double)l[e] - (double)mx);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ---- the sum, ascending, on one thread: see the note above on why this is NOT parallelised.
    if (tid == 0) {
        double sum = 0.0;
#pragma unroll
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    const float inv = (float) (1.0 / s_sum);

    // ---- the selection, by rank.  `p[e] = (float)(s_ex[e] * inv)` reproduces the old expression exactly:
    // `(float)(exp((double) l[e] - (double) mx) * inv)` with `inv` a FLOAT.
    // ---- p[] ONCE.  The rank loop below compares every expert against every other, so computing
    // `(float) (s_ex[f] * inv)` INSIDE it did **512 x 512 = 262,144** double multiplies and float converts per
    // token per layer instead of 512 - and a consumer die runs FP64 at a small fraction of FP32, which is why
    // this was worth more than the branch.  The shared load stays either way; the ARITHMETIC is what goes.
    //
    // THIS IS BIT-IDENTICAL, not merely equivalent: `p[e]` is the same expression it was, computed once instead
    // of 513 times, and the comparisons then see exactly the same floats.
    float* s_p = (float*) (s_ex + n_expert);
#pragma unroll
    for (int e = tid; e < n_expert; e += nt) s_p[e] = (float)(s_ex[e] * inv);
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ================================ THE SELECTION, IN k PASSES ================================
    //
    // **THE RANK-BY-COUNTING LOOP COMPARED EVERY EXPERT AGAINST EVERY OTHER**: 512 x 512 = 262,144 shared
    // loads and compares per token per layer, to choose ten things out of 512 - measured at 34.35 us, 63% of
    // `moe_route`'s accounted cost and the largest single item in the R3 plan.  The comment above records that
    // the FP64 ARITHMETIC inside that loop was hoisted out once already ("262,144 double multiplies per token
    // per layer instead of 512"); **the O(n^2) STRUCTURE was left, and it is the structure that costs.**
    //
    // `k` passes of a block-wide argmax is 10 x 512 = 5,120 compares - **51x less work** - and it produces the
    // SAME ORDER.  That is the whole correctness argument: scanning `e` ascending with a strict `>` keeps the
    // LOWEST index on a tie, which is exactly the rule the rank loop spelled out as
    // `else if (f < e && pf == pe) ++rank`.  A stable descending top-k, ties by index.
#pragma unroll
    for (int e = tid; e < n_expert; e += nt) s_taken[e] = 0;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;               // a sentinel that loses to every real index
        for (int e = tid; e < n_expert; e += nt) {
            if (s_taken[e]) continue;
            const float pe = s_p[e];
            if (pe > bv) { bv = pe; bi = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float ov = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bv, off);
            /*
            DPCT1108: '__shfl_down_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const int oi = dpct::experimental::shift_sub_group_left(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if ((tid & 31) == 0) { s_red[tid >> 5] = bv; s_rid[tid >> 5] = bi; }
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
        if (tid < 32) {
            const int nw = (nt + 31) >> 5;
            float v = (tid < nw) ? s_red[tid] : -INFINITY;
            int ix = (tid < nw) ? s_rid[tid] : n_expert;
            for (int off = 16; off > 0; off >>= 1) {
                /*
                DPCT1108: '__shfl_down_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const float ov = dpct::experimental::shift_sub_group_left(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), v, off);
                /*
                DPCT1108: '__shfl_down_sync' was migrated with the
                experimental feature masked sub_group function which may not be
                supported by all compilers or runtimes. You may need to adjust
                the code.
                */
                const int oi = dpct::experimental::shift_sub_group_left(
                    0xffffffffu,
                    sycl::ext::oneapi::this_work_item::get_sub_group(), ix,
                    off);
                if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
            }
            if (tid == 0 && ix < n_expert) {
                ids[(size_t) t * k + i] = ix;
                weights[(size_t) t * k + i] = v;
                s_taken[ix] = 1;
            }
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
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ---- renormalise, with ggml's lower clamp.  Order preserved.
    if (tid == 0) {
        double s = 0.0;
#pragma unroll
        for (int i = 0; i < k; ++i) s += (double)weights[(size_t)t * k + i];
        const double sc = sycl::fmax(s, 6.103515625e-05); // 2**-14
#pragma unroll
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
    }
}


#if defined(STRATA_HIP_GFX906)
// AMD (wave64): the same routing in ONE wavefront per token.  The block-wide kernel above spends its time in
// barriers - 10 selection passes with two __syncthreads each over 8 logical warps - not in arithmetic (43 us per
// token on gfx906).  Here every lane holds up to RW_PER probabilities in registers and each pass is a 64-lane
// butterfly argmax: no LDS round trip, no block barrier.  Bit-identical to the kernel above: fmaxf trees are
// exact, the exponentials and the (float) products are the same expressions, the sum stays one serial ascending
// scan, and the argmax keeps the lowest index on a tie (a total order, so the butterfly's association is free).
constexpr int RW_PER = 8;   // n_expert <= 64 * 8
__global__ void __launch_bounds__(64) router_top10_wave_kernel(const float* __restrict__ logits, int n_tokens,
                                                               int n_expert, int k, int* __restrict__ ids,
                                                               float* __restrict__ weights) {
    __shared__ double s_ex[64 * RW_PER];
    __shared__ double s_sum;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;
    const float* l = logits + (size_t) t * n_expert;
    float lv[RW_PER];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        lv[j] = e < n_expert ? l[e] : -INFINITY;
        mx = fmaxf(mx, lv[j]);
    }
#pragma unroll
    for (int off = 32; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 64));
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        if (e < n_expert) s_ex[e] = exp((double) lv[j] - (double) mx);
    }
    __syncthreads();
    if (lane == 0) {
        double sum = 0.0;
        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
        s_sum = sum;
    }
    __syncthreads();
    const float inv = (float) (1.0 / s_sum);
    float p[RW_PER];
#pragma unroll
    for (int j = 0; j < RW_PER; ++j) {
        const int e = lane + 64 * j;
        p[j] = e < n_expert ? (float) (s_ex[e] * inv) : -INFINITY;
    }
    unsigned taken = 0;
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;
#pragma unroll
        for (int j = 0; j < RW_PER; ++j) {
            const int e = lane + 64 * j;
            if (e >= n_expert || (taken >> j & 1u)) continue;
            if (p[j] > bv || (p[j] == bv && e < bi)) { bv = p[j]; bi = e; }
        }
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            const float ov = __shfl_xor(bv, off, 64);
            const int oi = __shfl_xor(bi, off, 64);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi >= n_expert) break;
        if ((bi & 63) == lane) taken |= 1u << (bi >> 6);
        if (lane == 0) { ids[(size_t) t * k + i] = bi; weights[(size_t) t * k + i] = bv; }
    }
    if (lane == 0) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
        const double sc = fmax(s, 6.103515625e-05);       // 2**-14
        for (int i = 0; i < k; ++i)
            weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
    }
}
#endif  // STRATA_HIP_GFX906

#if defined(__HIPCC__)
// ---- S6, AMD: the same router, BIT-IDENTICAL, without its serial parts. Measured on RDNA4 (gfx1201) the kernel
// above took 39 us per call in decode (16% of the GPU's decode time): the ascending double sum is 512 dependent FP64
// adds (~50 cycles each there), the ten selection passes are 20 block barriers, and the renormalisation re-reads its
// ten weights from global memory one by one.  Here:
//   * the sum's ONLY use is inv = (float) (1.0 / sum).  Every term is positive and the largest is exactly 1
//     (exp(0)), so the serial sum and any other summation order both lie within 2^-43 (relative) of the exact sum
//     (|serial - exact| <= 511u, |tree - exact| <= 21u, u = 2^-53).  1.0 / x and the float cast are monotone: when
//     the bracket [tree (1 - 2^-40), tree (1 + 2^-40)] maps to ONE float, that float is the serial sum's.  Otherwise
//     (about 1 row in 1000 in the parity test, or a non-finite sum) thread 0 runs the serial sum as before;
//   * the selection runs on wave 0 alone with the probabilities in registers (16 per lane): the largest p with the
//     lowest index on ties is unique, so the butterfly finds the same expert as the block reduction did;
//   * the renormalisation sums the ten weights in rank order from registers and divides them in parallel.
// Same expressions, same operands: the ids and weights are bitwise those of router_top10_kernel (hip_router_fast
// checks it on 65,536 rows with ties, near-ties, NaN rows and the forced serial path). STRATA_HIP_ROUTER_OLD=1: the
// kernel above.
template <bool FORCE_SERIAL>
__global__ void router_top10_fast_kernel(const float* __restrict__ logits, int n_tokens, int n_expert, int k,
                                         int* __restrict__ ids, float* __restrict__ weights) {
    extern __shared__ unsigned char s_raw[];
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    double* s_ex = (double*) (s_raw + taken_bytes);
    __shared__ float s_red[RT_MAX_THREADS / 32];
    __shared__ double s_dred[RT_MAX_THREADS / 32];
    __shared__ float s_inv;
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nt = blockDim.x, nw = (nt + 31) >> 5;
    const float* l = logits + (size_t) t * n_expert;
    // the max: as router_top10_kernel (exact, order-independent)
    float mx = -INFINITY;
    for (int e = tid; e < n_expert; e += nt) mx = fmaxf(mx, l[e]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, off));
    if (lane == 0) s_red[warp] = mx;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < nw) ? s_red[tid] : -INFINITY;
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (tid == 0) s_red[0] = v;
    }
    __syncthreads();
    mx = s_red[0];
    // the exponentials (the same expression), and a tree sum of them for the bracket
    double part = 0.0;
    for (int e = tid; e < n_expert; e += nt) {
        const double x = exp((double) l[e] - (double) mx);
        s_ex[e] = x;
        part += x;
    }
    for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    if (lane == 0) s_dred[warp] = part;
    __syncthreads();
    double tot = 0.0;
    for (int w = 0; w < nw; ++w) tot += s_dred[w];
    const double lo = tot * (1.0 - 0x1p-40), hi = tot * (1.0 + 0x1p-40);
    const float f_lo = (float) (1.0 / hi), f_hi = (float) (1.0 / lo);
    float inv;
    if (!FORCE_SERIAL && tot >= 1.0 && hi < 1e300 && f_lo == f_hi) {
        inv = f_lo;
    } else {   // block-uniform (every thread computed the same tot): the serial sum, exactly as router_top10_kernel
        if (tid == 0) {
            double sum = 0.0;
            for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
            s_inv = (float) (1.0 / sum);
        }
        __syncthreads();
        inv = s_inv;
    }
    if (warp != 0) return;
    // wave 0: lane holds experts lane + 32 j; p is router_top10_kernel's expression
    constexpr int PER = 16;
    float pv[PER];
    unsigned live = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int e = lane + 32 * j;
        pv[j] = -INFINITY;
        if (e < n_expert) { pv[j] = (float) (s_ex[e] * inv); live |= 1u << j; }
    }
    float my_w = 0.0f;
    int my_id = 0, nsel = 0;
    for (int i = 0; i < k; ++i) {
        float bv = -INFINITY;
        int bi = n_expert;   // a sentinel that loses to every real index
#pragma unroll
        for (int j = 0; j < PER; ++j)
            if (((live >> j) & 1u) && pv[j] > bv) { bv = pv[j]; bi = lane + 32 * j; }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (bi >= n_expert) break;   // as router_top10_kernel: no rank i or later is written (every p left is NaN)
        if ((bi & 31) == lane) live &= ~(1u << (bi >> 5));
        if (lane == i) { my_w = bv; my_id = bi; }
        nsel = i + 1;
    }
    if (nsel == k) {
        double s = 0.0;
        for (int i = 0; i < k; ++i) s += (double) __shfl_sync(0xffffffffu, my_w, i);
        const double sc = fmax(s, 6.103515625e-05);       // 2**-14
        if (lane < k) {
            ids[(size_t) t * k + lane] = my_id;
            weights[(size_t) t * k + lane] = (float) ((double) my_w / sc);
        }
    } else {   // router_top10_kernel's tail exactly: unwritten ranks keep what the buffer held and are renormalised too
        if (lane < nsel) { ids[(size_t) t * k + lane] = my_id; weights[(size_t) t * k + lane] = my_w; }
        __threadfence();
        if (lane == 0) {
            double s = 0.0;
            for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
            const double sc = fmax(s, 6.103515625e-05);
            for (int i = 0; i < k; ++i)
                weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
        }
    }
}

void launch_generic(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights, void* stream,
                    int mode) {   // mode 0: router_top10_kernel, 1: the HIP fast kernel, 2: the fast kernel's serial path
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem = taken_bytes + (size_t) n_expert * sizeof(double) + (size_t) n_expert * sizeof(float);
    if (mode == 1) {
        router_top10_fast_kernel<false><<<(unsigned) n_tokens, threads, smem, (cudaStream_t) stream>>>(
            logits, n_tokens, n_expert, k, ids, weights);
        return;
    }
    if (mode == 2) {
        router_top10_fast_kernel<true><<<(unsigned) n_tokens, threads, smem, (cudaStream_t) stream>>>(
            logits, n_tokens, n_expert, k, ids, weights);
        return;
    }
    router_top10_kernel<<<(unsigned) n_tokens, threads, smem, (cudaStream_t) stream>>>(
        logits, n_tokens, n_expert, k, ids, weights);
}
#endif

}  // namespace

bool router_top10_variant(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                          void* stream, int variant) {
#if defined(__HIPCC__)
    if (n_tokens <= 0 || n_expert <= 64 || n_expert > 512 || k <= 0 || k > 32) return false;
    if (variant < 0 || variant > 2) return false;
    launch_generic(logits, n_tokens, n_expert, k, ids, weights, stream, variant);
    return cudaGetLastError() == cudaSuccess;
#else
    if (n_tokens <= 0 || n_expert <= 64 || n_expert > 512 || k <= 0 || k > 32 || variant < 0 || variant > 2)
        return false;
    auto& q = *strata::q_of(stream);
    if (variant == 1) launch_sycl_fast<false>(q, logits, n_tokens, n_expert, k, ids, weights);
    else if (variant == 2) launch_sycl_fast<true>(q, logits, n_tokens, n_expert, k, ids, weights);
    else {
        const int threads = (std::min(n_expert, RT_MAX_THREADS) + 31) & ~31;
        const size_t taken_bytes = ((size_t)n_expert + 15u) & ~size_t(15);
        const size_t smem = taken_bytes + n_expert * (sizeof(double) + sizeof(float));
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint8_t, 1> local(sycl::range<1>(smem), h);
            h.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, (size_t)n_tokens * threads),
                                             sycl::range<3>(1, 1, threads)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(32)]] {
                    router_top10_kernel(logits, n_tokens, n_expert, k, ids, weights,
                        local.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
    if (stream == nullptr) q.wait_and_throw();
    return true;
#endif
}

void router_top10(const float *logits, int n_tokens, int n_expert, int k,
                  int *ids, float *weights, void *stream) try {
    if (strata::a770_fast() && n_tokens > 0 && n_expert > 64 && n_expert <= 512 && k > 0 && k <= 32) {
        auto& q = *strata::q_of(stream);
        launch_sycl_fast<false>(q, logits, n_tokens, n_expert, k, ids, weights);
        if (stream == nullptr) q.wait_and_throw();
        return;
    }
#if defined(__HIPCC__)
    {
#if defined(STRATA_HIP_GFX906)
        // gfx906 (wave64): the one-wavefront kernel below stays the default; the S6 kernel is wave32-shaped and only
        // runs here on request (STRATA_HIP_ROUTER_FAST=1) until it is measured on this card
        static const bool old = std::getenv("STRATA_HIP_ROUTER_FAST") == nullptr;
#else
        static const bool old = std::getenv("STRATA_HIP_ROUTER_OLD") != nullptr;
#endif
        if (!old && n_tokens > 0 && n_expert > 64 && n_expert <= 512 && k > 0 && k <= 32) {
            launch_generic(logits, n_tokens, n_expert, k, ids, weights, stream, 1);
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                std::fprintf(stderr, "router_top10 launch: %s\n", cudaGetErrorString(e));
                std::exit(1);
            }
            if (stream == nullptr) {
                const cudaError_t s = cudaDeviceSynchronize();
                if (s != cudaSuccess) {
                    std::fprintf(stderr, "router_top10: %s\n", cudaGetErrorString(s));
                    std::exit(1);
                }
            }
            return;
        }
    }
#endif
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
#if defined(STRATA_HIP_GFX906)
    if (n_expert <= 64 * RW_PER && !std::getenv("STRATA_ROUTER_BLOCK")) {
        router_top10_wave_kernel<<<(unsigned) n_tokens, 64, 0, (cudaStream_t) stream>>>(logits, n_tokens, n_expert, k,
                                                                                       ids, weights);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "router_top10 launch: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        if (stream == nullptr && cudaDeviceSynchronize() != cudaSuccess) std::exit(1);
        return;
    }
#endif
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;                  // at least one full warp, for the reductions
    // the selection's taken-mask, then n_expert doubles for the exponentials, then n_expert floats for the
    // probabilities.  The mask is first so the two aligned arrays need no offset parameter.
    const size_t taken_bytes = ((size_t) n_expert + 15u) & ~(size_t) 15u;
    const size_t smem =
        /*
        DPCT1083: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        taken_bytes + (size_t)n_expert * sizeof(double) +
        (size_t)n_expert * sizeof(float);
    /*
    DPCT1049: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};
        dpct::has_capability_or_fail(
            strata::q_of(stream)->get_device(),
            {sycl::aspect::fp64});

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range(smem), cgh);

                cgh.parallel_for<
                    dpct_kernel_name<class router_top10_kernel_3e0655>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)n_tokens) *
                                          sycl::range(1, 1, threads),
                                      sycl::range(1, 1, threads)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                        32)]] {
                        router_top10_kernel(
                            logits, n_tokens, n_expert, k, ids, weights,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                    });
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;

    if (stream == nullptr) {
        const dpct::err0 s = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());
    }
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

}  // namespace strata::kernels
