// src/prefill/moe_mmq.cu - see include/strata/prefill/moe_mmq.hpp.  llama.cpp's MMQ (ggml-cuda, MIT) is compiled
// from the pinned llama.cpp checkout the build already takes ggml from; src/prefill/ggml_cuda_host.cu supplies the
// few host symbols of ggml-cuda.cu it references.
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {

void ck(dpct::err0 e, const char *what) {
}

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

__dpct_inline__ void copy16_kernel(const sycl::uint4 *__restrict__ a,
                                   int64_t na,
                                   const sycl::uint4 *__restrict__ b,
                                   int64_t nb, sycl::uint4 *__restrict__ ab_dst,
                                   const sycl::uint4 *__restrict__ c,
                                   int64_t nc,
                                   sycl::uint4 *__restrict__ c_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}
__dpct_inline__ void copy1_kernel(const uint8_t *__restrict__ a, int64_t na,
                                  const uint8_t *__restrict__ b, int64_t nb,
                                  uint8_t *__restrict__ ab_dst,
                                  const uint8_t *__restrict__ c, int64_t nc,
                                  uint8_t *__restrict__ c_dst) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < na) ab_dst[i] = a[i];
    else if (i < na + nb) ab_dst[i] = b[i - na];
    else if (i < na + nb + nc) c_dst[i - na - nb] = c[i - na - nb];
}

// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down scales
// [2560][10] f16 (the layout of prefill/kernels.cu's blob_dequant_kernel).  A GGUF Q2_0 block is {f16 d; 16 code
// bytes} with the same 2-bit codes in the same order, so a block is a scale and a 16-byte run of codes.
/*
DPCT1110: The total declared local variable size in device function
strata_q2_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void strata_q2_kernel(const uint8_t *__restrict__ blob,
                                      uint16_t *__restrict__ gu,
                                      uint16_t *__restrict__ dn) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    constexpr size_t O_D_CODES = (size_t)1280 * 640,
                     O_GU_SC = O_D_CODES + (size_t)2560 * 160,
                     O_D_SC = O_GU_SC + (size_t)1280 * 40 * 2;
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2); // one GGUF block
    const int64_t n_gu = 1280LL * 40, n_d = 2560LL * 10;
    const uint8_t* codes;
    const uint16_t* scale;
    uint16_t* out;
    if (i < n_gu) {
        const int64_t row = i / 40, b = i % 40;
        codes = blob + row * 640 + b * 16;
        scale = (const uint16_t*) (blob + O_GU_SC) + row * 40 + b;
        out = gu + i * 9;
    } else if (i < n_gu + n_d) {
        const int64_t j = i - n_gu, row = j / 10, b = j % 10;
        codes = blob + O_D_CODES + row * 160 + b * 16;
        scale = (const uint16_t*) (blob + O_D_SC) + row * 10 + b;
        out = dn + j * 9;
    } else {
        return;
    }
    const sycl::uint4 q = *(const sycl::uint4 *)codes;
    const uint16_t* qh = (const uint16_t*) &q;
    out[0] = *scale;
#pragma unroll
    for (int k = 0; k < 8; ++k) out[1 + k] = qh[k];
}

__dpct_inline__ void swiglu_kernel(const float *__restrict__ gu,
                                   float *__restrict__ h, int64_t rows,
                                   int64_t n_ff, bool interleaved) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i >= rows * n_ff) return;
    const int64_t r = i / n_ff, k = i % n_ff;
    const float* row = gu + r * 2 * n_ff;
    const float g = interleaved ? row[2 * k] : row[k], u = interleaved ? row[2 * k + 1] : row[n_ff + k];
    h[i] = g / (1.0f + sycl::native::exp(-g)) * u;
}

__dpct_inline__ void iota_kernel(int32_t *dst, int64_t n) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i =
        (int64_t)item_ct1.get_group(2) * item_ct1.get_local_range(2) +
        item_ct1.get_local_id(2);
    if (i < n) dst[i] = (int32_t) i;
}

unsigned blocks(int64_t n) { return (unsigned) ((n + 255) / 256); }

}  // namespace

bool built() { return true; }

bool supported(int t) {
    switch ((ggml_type) t) {
        case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

size_t matrix_bytes(int t, int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / ggml_blck_size((ggml_type) t)) * ggml_type_size((ggml_type) t);
}

size_t q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) pad512(cols) * sizeof(block_q8_1_mmq) / (4 * QK8_1) + 128 * sizeof(block_q8_1_mmq);
}

void quantize(const float* x, const int32_t* ids, void* xq, int t, int64_t cols, int64_t ld, int64_t rows, void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) t, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "quantize");
}

Context::Context() {
    int dev = 0;
    dev = dpct::get_current_device_id();
    ctx_ = new ggml_backend_cuda_context(dev);
}
Context::~Context() { delete (ggml_backend_cuda_context*) ctx_; }

void Context::run(const Product& p, void* stream) {
    if (p.n <= 0 || p.max_rows <= 0) return;
    const ggml_type t = (ggml_type) p.type;
    const int64_t qk = ggml_blck_size(t), bpr = p.w_cols / qk;
    const mmq_args a = {(const char*) p.w, t, (const int*) p.xq, p.ids, p.bounds, p.dst, nullptr,
                        p.w_cols, p.w_rows, p.total_rows, bpr, p.total_rows, p.ld_dst,
                        p.n, p.n, (int64_t) (p.expert_bytes / ggml_type_size(t)), 0, 0,
                        1, 1, 0, 0, 0,
                        p.max_rows, p.max_rows};
    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const dpct::queue_ptr s = strata::q_of(stream);
    switch (t) {
        case GGML_TYPE_Q2_0: mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, a, s); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, a, s); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, a, s); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, a, s); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, a, s); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, a, s); break;
        default:
            std::fprintf(stderr, "prefill mmq: type %d is not covered\n", (int) t);
            std::exit(1);
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "mul_mat_q");
}

void gather_native(const void* gate, const void* up, size_t gu_half_bytes, const void* down, size_t d_bytes,
                   void* gu_dst, void* d_dst, void* stream) {
    const dpct::queue_ptr s = strata::q_of(stream);
    const bool a16 = ((uintptr_t) gate | (uintptr_t) up | (uintptr_t) down | (uintptr_t) gu_dst | (uintptr_t) d_dst |
                      gu_half_bytes | d_bytes) % 16 == 0;
    if (a16) {
        const int64_t na = (int64_t) gu_half_bytes / 16, nc = (int64_t) d_bytes / 16;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<class copy16_kernel_b12127>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(2 * na + nc)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy16_kernel(
                        (const sycl::uint4 *)gate, na, (const sycl::uint4 *)up,
                        na, (sycl::uint4 *)gu_dst, (const sycl::uint4 *)down,
                        nc, (sycl::uint4 *)d_dst);
                });
        }
    } else {
        const int64_t na = (int64_t) gu_half_bytes, nc = (int64_t) d_bytes;
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            s->parallel_for<dpct_kernel_name<class copy1_kernel_c557f3>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(2 * na + nc)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    copy1_kernel((const uint8_t *)gate, na, (const uint8_t *)up,
                                 na, (uint8_t *)gu_dst, (const uint8_t *)down,
                                 nc, (uint8_t *)d_dst);
                });
        }
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "gather_native");
}

void gather_strata_q2(const uint8_t* blob, void* gu_dst, void* d_dst, void* stream) {
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class strata_q2_kernel_793a65>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1, blocks(1280LL * 40 + 2560LL * 10)) *
                        sycl::range(1, 1, 256),
                    sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    strata_q2_kernel(blob, (uint16_t *)gu_dst,
                                     (uint16_t *)d_dst);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "gather_strata_q2");
}

void swiglu(const float* gu, float* h, int64_t rows, int64_t n_ff, bool interleaved, void* stream) {
    if (rows <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class swiglu_kernel_3f7a6f>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(rows * n_ff)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    swiglu_kernel(gu, h, rows, n_ff, interleaved);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "swiglu");
}

void iota(int32_t* dst, int64_t n, void* stream) {
    if (n <= 0) return;
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        strata::q_of(stream)
            ->parallel_for<dpct_kernel_name<class iota_kernel_9245a0>>(
                sycl::nd_range<3>(sycl::range(1, 1, blocks(n)) *
                                      sycl::range(1, 1, 256),
                                  sycl::range(1, 1, 256)),
                exp_props, [=](sycl::nd_item<3> item_ct1) {
                    iota_kernel(dst, n);
                });
    }
    /*
    DPCT1010: SYCL uses exceptions to report errors and does not use the
    error codes. The cudaGetLastError function call was replaced with 0. You
    need to rewrite this code.
    */
    ck(0, "iota");
}

// STRATA_MMQ_DIRECT (include/strata/prefill/moe_mmq.hpp): the pointer-table MMQ is CUDA only, so SYCL keeps the
// gather - these say so, for a prefill.cpp migrated from a tree that calls them
bool tab_built() { return false; }
bool tab_fits(int, int64_t) { return false; }
size_t overread_bytes(int, int64_t) { return 0; }
int direct_mode() { return 0; }
size_t direct_slot_guard() { return 0; }
void prep_group(const GuardTab&, size_t, void*, void*, size_t, void*) {}

}  // namespace strata::prefill::mmq
