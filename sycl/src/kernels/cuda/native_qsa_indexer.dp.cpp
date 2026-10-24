// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp; ggml/src/ggml-cuda/{set-rows.cu,norm.cu,rope.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include "strata/core/geometry.hpp"
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;
static_assert(D == strata::core::kCompiledGeometry.idx_key_dim && R == strata::core::kCompiledGeometry.idx_block && ROT == strata::core::kCompiledGeometry.n_rot,
              "native_qsa_indexer is compiled for core/geometry.hpp");
inline float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        x += dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), x,
            offset);
    return x;
}
// TAB (#280, STRATA_ROPE_TABLE=1): the angles from the session's float64 table.  The host launches <false> whenever
// no table applies - the default - so the default kernel is 0.1.31's code exactly (the table read is not in it;
// with it merely skipped at run time, the compiled default path changed its results).
template <bool TAB>
__dpct_inline__ void
append(const float *__restrict__ raw, const int32_t *__restrict__ pos_dev,
       int pos_stride, int n_steps, int pos_base,
       const float *__restrict__ gamma, float epsilon, float *__restrict__ tail,
       float *__restrict__ dead, float *__restrict__ pooled,
       int32_t *__restrict__ block_pos, int max_cells, float theta_scale,
       float freq_scale, float corr_low, float corr_high, float ext_factor,
       float mscale, const int32_t *__restrict__ mtab, RopeTab rt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
auto &values = *sycl::ext::oneapi::group_local_memory_for_overwrite<float[D]>(
    sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &partials =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    const int d = item_ct1.get_local_id(2);
    for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
        const int pos = pos_dev[(std::size_t) step_idx * pos_stride];
        if (pos >= 0 && pos < max_cells) {
            const float* raw_step = raw + (std::size_t) step_idx * D;
            const int slot = pos % R;
            float incoming = 0.0f;
            if (d < D) {
                // SET_ROWS stores F16; GET_ROWS expands those exact values to F32.
                incoming =
                    sycl::vec<sycl::half, 1>(
                        sycl::vec<float, 1>(raw_step[d])
                            .convert<sycl::half, sycl::rounding_mode::rte>()[0])
                        .convert<float, sycl::rounding_mode::automatic>()[0];
                if (slot < R - 1) tail[slot * D + d] = incoming;
            }
            if (pos == 0 || slot == R - 1) {
                float mean = 0.0f;
                if (d < D) {
                    // The spare's four gather indices all name cell zero. Completed blocks
                    // use chronological slices; each graph ADD materializes an F32 sum.
                    float sum = pos == 0 ? incoming : tail[d];
#pragma unroll
                    for (int j = 1; j < R; ++j)
                        /*
                        DPCT1013: The rounding mode could not be specified
                        and the generated code may have different accuracy than
                        the original code. Verify the correctness. SYCL math
                        built-in function rounding mode is aligned with OpenCL
                        C 1.2 standard.
                        */
                        sum = sum + (pos == 0 || j == R - 1 ? incoming
                                                             : tail[j * D + d]);
                    /*
                    DPCT1013: The rounding mode could not be specified and
                    the generated code may have different accuracy than the
                    original code. Verify the correctness. SYCL math built-in
                    function rounding mode is aligned with OpenCL C 1.2
                    standard.
                    */
                    mean = sycl::fma(0.25f, sum,
                                     0.0f); // SCALE includes a zero bias.
                }
                float square_sum = 0.0f;
                if (d < D) square_sum += mean * mean;
                square_sum = warp_sum(square_sum);
                const int lane = d % 32;
                if (lane == 0) partials[d / 32] = square_sum;
                /*
                DPCT1118: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
                square_sum = warp_sum(square_sum);
                const float scale = sycl::rsqrt(square_sum / D + epsilon);
                if (d < D) values[d] = scale * mean * gamma[d];
                /*
                DPCT1118: SYCL group functions and algorithms must be
                encountered in converged control flow. You may need to adjust
                the code.
                */
                /*
                DPCT1065: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
                if (d < D) {
                    const int b = pos / R;
                    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
                    float y = values[d];
                    if (d < ROT) {
                        const int pair = d % (ROT / 2);
                        // The spare (pos == 0) keeps its zero angle; under YaRN the mscale magnitude still rides in
                        // through cos(0) - which is exactly what the queries are scaled by, so the top-k is unmoved.
                        float c, s;
                        if (!(TAB && rope_tab_cs(rt, pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s))) {
                            const float theta_extrap =
                                (pos == 0 ? 0
                                          : mrope_pos(mtab, rope_pos, pair)) *
                                dpct::pow(theta_scale, float(pair));
                            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
                        }
                        const float a = values[pair], z = values[pair + ROT / 2];
                        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
                    }
                    pooled[std::size_t(b) * D + d] = y;
                    if (pos == 0) dead[d] = y;
                    else pooled[std::size_t(b + 1) * D + d] = dead[d];
                    if (d == 0 && pos != 0) *block_pos = rope_pos;
                }
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
        if (n_steps > 1) item_ct1.barrier();
    }
}
// ---- C-2: the batched append.  The pooled key of a completed block b (its last cell pos = 4b+3), computed with
// the single append's arithmetic: keys rounded through F16, summed tail[0]+tail[1]+tail[2]+incoming in that order,
// the RMS norm, gamma, the rotation at pos_base + 4b.  `k(j)` is the key of the block's cell j: a row of this batch
// (cell >= p0) or the tail the previous batch left (cell < p0).  Only the batch's last completed block writes the
// spare after it (every earlier one's is overwritten by the next block in order).
template <bool TAB>
__dpct_inline__ float
pooled_value(const float *values, int d, int rope_pos, float theta_scale,
             float freq_scale, float corr_low, float corr_high,
             float ext_factor, float mscale, const int32_t *mtab, bool zero_pos,
             const RopeTab &rt) {
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        // The same rotation the single append applies, rope_scaling.hpp's helper shared: under none this is
        // today's cosf/sinf; under linear/YaRN the pooled keys move with the cache's K, and the spare's zero
        // angle carries the mscale magnitude through cos(0) exactly as the single append's does.
        float c, s;
        if (!(TAB && rope_tab_cs(rt, zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair), pair, c, s))) {
            const float theta_extrap =
                (zero_pos ? 0 : mrope_pos(mtab, rope_pos, pair)) *
                dpct::pow(theta_scale, float(pair));
            rope_scaled_angle(theta_extrap, freq_scale, corr_low, corr_high, ext_factor, mscale, pair, c, s);
        }
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}
__dpct_inline__ float norm_scale(float mean, float *partials, int d) {
    float square_sum = d < D ? mean * mean : 0.0f;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    sycl::ext::oneapi::this_work_item::get_nd_item<3>().barrier();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    return square_sum;
}
// cell 0 of a sequence: the spare (every gather index names cell 0), written to pooled[0] and dead
template <bool TAB>
__dpct_inline__ void
append_first(const float *__restrict__ raw, const float *__restrict__ gamma,
             float epsilon, float *__restrict__ dead,
             float *__restrict__ pooled, float theta_scale, float freq_scale,
             float corr_low, float corr_high, float ext_factor, float mscale,
             const int32_t *__restrict__ mtab, RopeTab rt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int d = item_ct1.get_local_id(2);
    auto &values =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[D]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &partials =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float mean = 0.0f;
    if (d < D) {
        const float incoming =
            sycl::vec<sycl::half, 1>(
                sycl::vec<float, 1>(raw[d])
                    .convert<sycl::half, sycl::rounding_mode::rte>()[0])
                .convert<float, sycl::rounding_mode::automatic>()[0];
        float sum = incoming;
#pragma unroll
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        for (int j = 1; j < R; ++j) sum = sum + incoming;
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        mean = sycl::fma(0.25f, sum, 0.0f);
    }
    const float square_sum = norm_scale(mean, partials, d);
    const float scale = sycl::rsqrt(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (d >= D) return;
    const float y = pooled_value<TAB>(values, d, 0, theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale,
                                 mtab, true, rt);
    pooled[d] = y;
    dead[d] = y;
}
template <bool TAB>
__dpct_inline__ void
append_blocks(const float *__restrict__ raw, int64_t n, int64_t p0,
              int pos_base, const float *__restrict__ gamma, float epsilon,
              const float *__restrict__ tail, const float *__restrict__ dead,
              float *__restrict__ pooled, int32_t *__restrict__ block_pos,
              int64_t first_block, int64_t last_block, float theta_scale,
              float freq_scale, float corr_low, float corr_high,
              float ext_factor, float mscale, const int32_t *__restrict__ mtab,
              RopeTab rt) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t b = first_block + item_ct1.get_group(2);
    const int d = item_ct1.get_local_id(2);
    auto &values =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[D]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    auto &partials =
        *sycl::ext::oneapi::group_local_memory_for_overwrite<float[32]>(
            sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float mean = 0.0f;
    if (d < D) {
        auto key = [&](int j) -> float {
            const int64_t cell = b * R + j;
            return cell >= p0
                       ? sycl::vec<sycl::half, 1>(
                             sycl::vec<float, 1>(raw[(cell - p0) * D + d])
                                 .convert<sycl::half,
                                          sycl::rounding_mode::rte>()[0])
                             .convert<float,
                                      sycl::rounding_mode::automatic>()[0]
                       : tail[j * D + d];
        };
        float sum = key(0);
#pragma unroll
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        for (int j = 1; j < R; ++j) sum = sum + key(j);
        /*
        DPCT1013: The rounding mode could not be specified and the generated
        code may have different accuracy than the original code. Verify the
        correctness. SYCL math built-in function rounding mode is aligned with
        OpenCL C 1.2 standard.
        */
        mean = sycl::fma(0.25f, sum, 0.0f);
    }
    const float square_sum = norm_scale(mean, partials, d);
    const float scale = sycl::rsqrt(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (d >= D) return;
    const int rope_pos = pos_base + R * (int) b;
    pooled[std::size_t(b) * D + d] = pooled_value<TAB>(values, d, rope_pos, theta_scale, freq_scale, corr_low, corr_high,
                                                  ext_factor, mscale, mtab, false, rt);
    if (b == last_block) {
        pooled[std::size_t(b + 1) * D + d] = dead[d];
        if (d == 0) *block_pos = rope_pos;
    }
}
// the tail after the batch: slot s holds the key of the batch's last cell with cell % 4 == s (s < 3), if any
__dpct_inline__ void append_tail(const float *__restrict__ raw, int64_t n,
                                 int64_t p0, float *__restrict__ tail) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int s = item_ct1.get_group(2), d = item_ct1.get_local_id(2);
    if (d >= D) return;
    const int64_t last = p0 + n - 1;
    int64_t cell = last - ((last % R) - s + R) % R;       // the last cell <= last with cell % 4 == s
    if (cell < p0) return;
    tail[s * D + d] =
        sycl::vec<sycl::half, 1>(
            sycl::vec<float, 1>(raw[(cell - p0) * D + d])
                .convert<sycl::half, sycl::rounding_mode::rte>()[0])
            .convert<float, sycl::rounding_mode::automatic>()[0];
}
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
} // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_qsa_indexer_append_steps(const float* raw, const int32_t* relative_pos_device,
                                     int pos_stride, int n_steps, int32_t pos_base,
                                     const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                                     const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    if (n_steps <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency/scaling and explicit stream");
    const std::size_t pos_span = (n_steps == 1 || pos_stride <= 0) ? 4 : (std::size_t(n_steps - 1) * pos_stride + 1) * 4;
    const Span spans[] = {{raw,std::size_t(n_steps)*D*4},{relative_pos_device,pos_span},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);   // none: the identity constants
    const RopeTab rt = rope_table_for(scaling);
    if (rt.cos != nullptr)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                auto mrope_table_ct18 = mrope_table();

                cgh.parallel_for<dpct_kernel_name<class append_182f95,
                                                  dpct_kernel_scalar<true>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            append<true>(raw, relative_pos_device, pos_stride,
                                         n_steps, pos_base, gamma, epsilon,
                                         b.tail, b.dead, b.pooled, b.block_pos,
                                         int(max_cells), theta_scale,
                                         k.freq_scale, k.corr_low, k.corr_high,
                                         k.ext_factor, k.attn_factor,
                                         mrope_table_ct18, rt);
                        });
            });
    } else {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->submit([&](sycl::handler &cgh) {
                auto mrope_table_ct18 = mrope_table();

                cgh.parallel_for<dpct_kernel_name<class append_182f95,
                                                  dpct_kernel_scalar<false>>>(
                    sycl::nd_range<3>(sycl::range(1, 1, THREADS),
                                      sycl::range(1, 1, THREADS)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            append<false>(raw, relative_pos_device, pos_stride,
                                          n_steps, pos_base, gamma, epsilon,
                                          b.tail, b.dead, b.pooled, b.block_pos,
                                          int(max_cells), theta_scale,
                                          k.freq_scale, k.corr_low, k.corr_high,
                                          k.ext_factor, k.attn_factor,
                                          mrope_table_ct18, rt);
                        });
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}

void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, const RopeScaling& scaling, void* stream) {
    native_qsa_indexer_append_steps(raw, relative_pos_device, 0, 1, pos_base, gamma, epsilon, b, s, max_cells, scaling, stream);
}
void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT || p0 < 0 || p0 + n > max_cells ||
        max_cells > INT32_MAX || pos_base < 0 || pos_base % R || int64_t(pos_base) + max_cells > INT32_MAX ||
        !std::isfinite(epsilon) || epsilon <= 0.0f || rope_scaling_invalid(scaling) != nullptr)
        throw std::invalid_argument("native QSA indexer (batch): bad geometry, positions, parameters or scaling");
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / ROT);
    const RopeKernelArgs k = scaling.kernel_args(ROT);   // none: the identity constants
    const float fsf = k.freq_scale, cl = k.corr_low, ch = k.corr_high, ef = k.ext_factor, ms = k.attn_factor;
    const dpct::queue_ptr st = strata::q_of(stream);
    const int32_t* mtab = mrope_table();
    const RopeTab rt = rope_table_for(scaling);
    if (p0 == 0) {
        if (rt.cos != nullptr)
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class append_first_ea2d1f,
                                              dpct_kernel_scalar<true>>>(
                sycl::nd_range<3>(sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        append_first<true>(raw, gamma, epsilon, b.dead,
                                           b.pooled, theta_scale, fsf, cl, ch,
                                           ef, ms, mtab, rt);
                    });
        } else {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            st->parallel_for<dpct_kernel_name<class append_first_ea2d1f,
                                              dpct_kernel_scalar<false>>>(
                sycl::nd_range<3>(sycl::range(1, 1, THREADS),
                                  sycl::range(1, 1, THREADS)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        append_first<false>(raw, gamma, epsilon, b.dead,
                                            b.pooled, theta_scale, fsf, cl, ch,
                                            ef, ms, mtab, rt);
                    });
        }
    }
    // completed blocks: those whose last cell (4b+3) lies in [p0, p0 + n)
    const int64_t first = p0 <= R - 1 ? 0 : (p0 - (R - 1) + R - 1) / R;       // the smallest b with 4b+3 >= p0
    const int64_t hi = p0 + n - 1 >= R - 1 ? (p0 + n - 1 - (R - 1)) / R : -1;   // the largest b with 4b+3 <= p0+n-1
    if (hi >= first && rt.cos != nullptr)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class append_blocks_a6c94e,
                                          dpct_kernel_scalar<true>>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(hi - first + 1)) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                append_blocks<true>(raw, n, p0, pos_base, gamma, epsilon,
                                    b.tail, b.dead, b.pooled, b.block_pos,
                                    first, hi, theta_scale, fsf, cl, ch, ef, ms,
                                    mtab, rt);
            });
    } else if (hi >= first)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class append_blocks_a6c94e,
                                          dpct_kernel_scalar<false>>>(
            sycl::nd_range<3>(sycl::range(1, 1, (unsigned)(hi - first + 1)) *
                                  sycl::range(1, 1, THREADS),
                              sycl::range(1, 1, THREADS)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                append_blocks<false>(raw, n, p0, pos_base, gamma, epsilon,
                                     b.tail, b.dead, b.pooled, b.block_pos,
                                     first, hi, theta_scale, fsf, cl, ch, ef,
                                     ms, mtab, rt);
            });
    }
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        st->parallel_for<dpct_kernel_name<class append_tail_e0658e>>(
            sycl::nd_range<3>(sycl::range(1, 1, R - 1) * sycl::range(1, 1, D),
                              sycl::range(1, 1, D)),
            exp_props, [=](sycl::nd_item<3> item_ct1) {
                append_tail(raw, n, p0, b.tail);
            });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const auto error = 0;
    /*
    DPCT1009: SYCL reports errors using exceptions and does not use error
    codes. Please replace the "get_error_string_dummy(...)" with a real
    error-handling function.
    */
    /*
    DPCT1001: The statement could not be removed.
    */
    /*
    DPCT1000: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (error !=
        0) throw std::runtime_error(dpct::get_error_string_dummy(error));
}
} // namespace strata::kernels
