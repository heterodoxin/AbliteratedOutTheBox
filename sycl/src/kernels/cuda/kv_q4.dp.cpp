// src/kernels/cuda/kv_q4.cu - see include/strata/kernels/kv_q4.hpp. Q4_0 KV with Walsh-Hadamard rotation
// (from PR #21 by code-martin; KV-streaming integration and the deterministic group maximum added on merge).
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    const dpct::err0 e = 0;
}

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers.
// Orthonormal (scale 1/sqrt(256) = 1/16), so it is its own inverse.
__dpct_inline__ void fwht256_kernel(const float *__restrict__ src,
                                    float *__restrict__ dst, int64_t n_rows,
                                    float scale) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size;   // 8

    const int64_t r =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
    if (r >= n_rows) return;

    const float* row_src = src + r * N;
    float* row_dst = dst + r * N;

    float reg[el_w];
    const int lane = item_ct1.get_local_id(2);

#pragma unroll
    for (int i = 0; i < el_w; ++i) reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            /*
            DPCT1108: '__shfl_xor_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            const float val2 = dpct::experimental::permute_sub_group_by_xor(
                0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
                val, h);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i) row_dst[i * warp_size + lane] = reg[i];
}

// One 32-value group in one warp (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(x / d + 8.5), 0, 15). Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one). Ties in |x| resolve to the larger value in EVERY lane,
// so all lanes agree on d (a plain `a > amax` could leave lanes with opposite signs and one block two scales).
__dpct_inline__ uint16_t q4_group(float x, int lane, uint8_t &byte) {
    float amax = sycl::fabs(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const float a = dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            amax, o);
        /*
        DPCT1108: '__shfl_xor_sync' was migrated with the experimental
        feature masked sub_group function which may not be supported by all
        compilers or runtimes. You may need to adjust the code.
        */
        const float v = dpct::experimental::permute_sub_group_by_xor(
            0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(),
            mval, o);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = sycl::vec<float, 1>{(x * id + 8.5f)}
                .convert<int, sycl::rounding_mode::rtz>()[0];
    const uint8_t qc = (uint8_t) (q < 0 ? 0 : (q > 15 ? 15 : q));
    /*
    DPCT1108: '__shfl_down_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    const uint8_t qhi = dpct::experimental::shift_sub_group_left(
        0xffffffffu, sycl::ext::oneapi::this_work_item::get_sub_group(), qc,
        16);
    byte = (uint8_t) (qc | (qhi << 4));
    (void) lane;
    return f16_from_f32(d);
}

__dpct_inline__ void q4_store(uint8_t *pool, long long row, int b, int lane,
                              uint16_t d, uint8_t byte) {
    block_q4_0* blk = reinterpret_cast<block_q4_0*>(pool + row * (long long) sizeof(block_q4_0) * 8) + b;
    if (lane == 0) blk->d = d;
    if (lane < 16) blk->qs[lane] = byte;
}

// One block = one 32-value group of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
__dpct_inline__ void
kv_append_q4_kernel(uint8_t *__restrict__ k_q4, uint8_t *__restrict__ v_q4,
                    const int32_t *__restrict__ table,
                    const int32_t *__restrict__ step, int step_stride,
                    int planes, const float *__restrict__ kcur,
                    const float *__restrict__ vcur, int kv_heads, int head_dim,
                    int page_size, KvHostPools host) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int step_idx = (int)(item_ct1.get_group(0) / (unsigned)planes);
    const bool is_v = (item_ct1.get_group(0) % (unsigned)planes) == 1u;
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    const long long pos =
        (long long)*(step + (long long)step_idx * step_stride + kStepPos);
    const int h = item_ct1.get_group(2), b = item_ct1.get_group(1),
              t = item_ct1.get_local_id(2);
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * QK4_0 + t];
    uint8_t byte;
    const uint16_t d = q4_group(x, t, byte);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
    if (host.k_q4 != nullptr)
        q4_store(is_v ? host.v_q4 : host.k_q4, ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t,
                 d, byte);
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when given.
__dpct_inline__ void kv_append_q4_batch_kernel(
    uint8_t *__restrict__ k_q4, uint8_t *__restrict__ v_q4,
    const int32_t *__restrict__ table, int64_t pos0,
    const float *__restrict__ K, const float *__restrict__ V, int kv_heads,
    int head_dim, int page_size, int is_v_grid, KvHostPools host,
    KvHostPools stage) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long t = item_ct1.get_group(2);
    const long long pos = pos0 + t;
    const int h = item_ct1.get_group(1), b = item_ct1.get_group(0),
              th = item_ct1.get_local_id(2);
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
    uint8_t byte;
    const uint16_t d = q4_group(x, th, byte);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0) q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
    if (host.k_q4 != nullptr) q4_store(is_v ? host.v_q4 : host.k_q4, row_id, b, th, d, byte);
    if (stage.k_q4 != nullptr) q4_store(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, d, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths)
__dpct_inline__ void kv_gather_q4_kernel(
    const uint8_t *__restrict__ k_q4, const uint8_t *__restrict__ v_q4,
    const int32_t *__restrict__ table, const int32_t *__restrict__ ids,
    const int32_t *__restrict__ step, int kv_heads, int head_dim, int page_size,
    uint16_t *__restrict__ k_scratch, uint16_t *__restrict__ v_scratch) {
    /*
    DPCT1098: The '*' expression is used instead of the __ldg call. These
    two expressions do not provide the exact same functionality. Check the
    generated code for potential precision and/or performance issues.
    */
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const long long n_ids = (long long)*(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4_0;                        // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0);    // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx =
        (long long)item_ct1.get_group(2) * item_ct1.get_local_range(1) +
        item_ct1.get_local_id(1);
    if (blk_idx >= total_blocks) return;

    const int t = item_ct1.get_local_id(2);
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_q4_0* k_blk = reinterpret_cast<const block_q4_0*>(k_q4 + row * bytes_per_head) + b;
    const block_q4_0* v_blk = reinterpret_cast<const block_q4_0*>(v_q4 + row * bytes_per_head) + b;
    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);
    const int j = t < 16 ? t : (t - 16);
    const uint8_t k_byte = k_blk->qs[j];
    const uint8_t v_byte = v_blk->qs[j];
    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    if (n_rows <= 0) return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                auto ct3 = 1.0f / 16.0f;

                cgh.parallel_for<dpct_kernel_name<class fwht256_kernel_d551a1>>(
                    sycl::nd_range<3>(sycl::range(1, 1, (unsigned)num_blocks) *
                                          sycl::range(1, rows_per_block, 32),
                                      sycl::range(1, rows_per_block, 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            fwht256_kernel(src, dst, n_rows, ct3);
                        });
            });
    }
    check("fwht256 launch");
}

void kv_append_q4_steps(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                        int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                        void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4");
    const int planes = (v_q4 == nullptr || (v_q4 == k_q4 && vcur == kcur)) ? 1 : 2;
    const dpct::dim3 grid((unsigned)s.n_head_kv, (unsigned)(s.head_dim / QK4_0),
                          (unsigned)(n_steps * planes));
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->submit([&](sycl::handler &cgh) {
                KvHostPools host_host_KvHostPools_ct11 =
                    host ? *host : KvHostPools{};

                cgh.parallel_for<
                    dpct_kernel_name<class kv_append_q4_kernel_f256b8>>(
                    sycl::nd_range<3>(grid * sycl::range(1, 1, 32),
                                      sycl::range(1, 1, 32)),
                    exp_props,
                    [=](sycl::nd_item<3> item_ct1)
                        [[sycl::reqd_sub_group_size(32)]] {
                            kv_append_q4_kernel(
                                k_q4, v_q4, page_table, step, step_stride,
                                planes, kcur, vcur, (int)s.n_head_kv,
                                (int)s.head_dim, (int)s.page_size,
                                host_host_KvHostPools_ct11);
                        });
            });
    }
    check("kv_append_q4 launch");
}

void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    kv_append_q4_steps(k_q4, v_q4, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, int64_t pos0, int64_t T, const float* K,
                  const float* V, const QsaShapes& s, void* stream, const KvHostPools* host, const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4");
    const dpct::dim3 grid((unsigned)T, (unsigned)s.n_head_kv,
                          (unsigned)(s.head_dim / QK4_0));
    dpct::queue_ptr cs = strata::q_of(stream);
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        cs->parallel_for<
            dpct_kernel_name<class kv_append_q4_batch_kernel_11c37a>>(
            sycl::nd_range<3>(grid * sycl::range(1, 1, 32),
                              sycl::range(1, 1, 32)),
            exp_props,
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(32)]] {
                kv_append_q4_batch_kernel(k_q4, v_q4, page_table, pos0, K, V,
                                          (int)s.n_head_kv, (int)s.head_dim,
                                          (int)s.page_size, is_v, h, st);
            });
    }
    check("kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class kv_gather_q4_kernel_300c16>>(
                sycl::nd_range<3>(sycl::range(1, 1, num_blocks) *
                                      sycl::range(1, rows_per_block, 32),
                                  sycl::range(1, rows_per_block, 32)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    kv_gather_q4_kernel(k_q4, v_q4, page_table, ids, step,
                                        (int)s.n_head_kv, (int)s.head_dim,
                                        (int)s.page_size, k_scratch, v_scratch);
                });
    }
    check("kv_gather_q4 launch");
}

}  // namespace strata::kernels
