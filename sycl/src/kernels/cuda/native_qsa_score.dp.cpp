// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{mmf.cuh,mma.cuh,unary.cu,binbcast.cu}.
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
#include "strata/kernels/native_qsa_score.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D=128, HEADS=4, R=4, ROWS=32, WARPS=2, STRIDE=36, COMBINE=68;
static_assert(D == strata::core::kCompiledGeometry.idx_key_dim && HEADS == strata::core::kCompiledGeometry.idx_q_heads && R == strata::core::kCompiledGeometry.idx_block,
              "native_qsa_score is compiled for core/geometry.hpp");
struct TileA { uint32_t x[4]; };
struct TileB { uint32_t x[2]; };
struct TileC { float x[4]={0.0f,0.0f,0.0f,0.0f}; };
#if !defined(__HIPCC__)
#if 0   // SYCL: inline PTX (mma/ldmatrix/cp.async) - the XMX port is pending; see tools/fixups.py
__dpct_inline__ void load_a(TileA &a, const float *p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *src = p + (item_ct1.get_local_id(2) % 16) * STRIDE +
                       (item_ct1.get_local_id(2) / 16) * 4;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3])
                 : "l"(src));
#else
    dpct::experimental::matrix::ldmatrix((uintptr_t)src, &a.x[0], &a.x[1],
                                         &a.x[2], &a.x[3]);
#endif
}
__dpct_inline__ void load_b(TileB &b, const float *p) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const float *src = p + (item_ct1.get_local_id(2) % 8) * STRIDE +
                       ((item_ct1.get_local_id(2) / 8) * 4) % 8;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                 : "=r"(b.x[0]), "=r"(b.x[1])
                 : "l"(src));
#else
    dpct::experimental::matrix::ldmatrix((uintptr_t)src, &b.x[0], &b.x[1]);
#endif
}
__dpct_inline__ void mma(TileC &c, const TileA &a, const TileB &b) {
    // Deliberately no cvt.rn.tf32: pinned mma.cuh passes raw F32 bits directly.
    /*
    DPCT1053: Migration of device assembly code is not supported.
    */
    asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c.x[0]), "+f"(c.x[1]), "+f"(c.x[2]), "+f"(c.x[3])
        : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
          "r"(b.x[1]));
}
#endif
/*
DPCT1110: The total declared local variable size in device function
score_kernel exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
__dpct_inline__ void score_kernel(const float *__restrict__ pooled,
                                  const float *__restrict__ query,
                                  const float *__restrict__ bias,
                                  const int32_t *__restrict__ step,
                                  int max_cells, float *__restrict__ cells) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int n = step[kStepNKv], full = step[kStepNBid];
    if(n<1||n>max_cells||step[kStepPos]!=n-1||full!=n/R||
       step[kStepWidth]!=(n<2051?n:2051))return;
    const int row0 = item_ct1.get_group(2) * ROWS;
    if(row0>full)return;
    const int lane = item_ct1.get_local_id(2), warp = item_ct1.get_local_id(1);
#if 1   // SYCL: the scalar path (the PTX one above is not ported yet)
    // Turing (STRATA_EXPERIMENTAL_SM75, a layer-split stage): no tf32 mma.  The same scores with FP32 FMAs, one row
    // per thread of the first warp - rounded differently from the tensor-core path (FP32 instead of TF32 inputs).
    if(warp!=0)return;
    const int row=row0+lane;
    if(row>full)return;
    float h[HEADS];
    for(int j=0;j<HEADS;++j){
        float acc=0.0f;
        for(int d=0;d<D;++d)acc=fmaf(pooled[size_t(row)*D+d],query[j*D+d],acc);
        h[j]=fmaxf(acc,0.0f);
    }
    float sum=(((h[0] + h[1]) + h[2]) + h[3]);
    if(bias)sum=(sum + bias[row]);
    sum=(sum + (row==full&&n%R?1e9f:0.0f));
    for(int i=row*R;i<n&&i<(row+1)*R;++i)cells[i]=sum;
#else
    auto &shared = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[WARPS * 16 * STRIDE]>(
        sycl::ext::oneapi::this_work_item::get_work_group<3>());
    float* tile=shared+warp*16*STRIDE;
    TileC c[2];
    // mmf's launch heuristic picks2warps for K128. Each warp accumulates
    // K[warp*32,warp*32+32), then K[warp*32+64,warp*32+96).
    for(int col=warp*32+lane;col<D;col+=WARPS*32){
        TileA a[2][4];
#pragma unroll
        for(int ia=0;ia<2;++ia){
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
            for(int i=0;i<16;++i){
                const int row=row0+ia*16+i;
                tile[i*STRIDE+lane]=row<=full?pooled[size_t(row)*D+col]:0.0f;
            }
            sycl::group_barrier(
                sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
            for(int k=0;k<4;++k)load_a(a[ia][k],tile+k*8);
        }
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
        for(int h=0;h<8;++h)tile[h*STRIDE+lane]=h<HEADS?query[h*D+col]:0.0f;
        sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
#pragma unroll
        for(int k=0;k<4;++k){
            TileB b;load_b(b,tile+k*8);
#pragma unroll
            for(int ia=0;ia<2;++ia)mma(c[ia],a[ia][k],b);
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for(int ia=0;ia<2;++ia){
#pragma unroll
        for(int l=0;l<4;++l){
            const int i=warp*ROWS+ia*16+(l/2)*8+lane/4;
            const int h=(lane%4)*2+l%2;
            shared[h*COMBINE+i]=c[ia].x[l];
        }
    }
    /*
    DPCT1065: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // In the reference: +0 then warp0 partial then warp1 partial, followed
    // by materialized ReLU, CONT(head0), ADD(head1), ADD(head2), ADD(head3).
    if(warp==0){
        const int row=row0+lane;
        if(row>full)return;
        float h[HEADS];
#pragma unroll
        for(int j=0;j<HEADS;++j){
            /*
            DPCT1013: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            float v = 0.0f + shared[j * COMBINE + lane];
            /*
            DPCT1013: The rounding mode could not be specified and the
            generated code may have different accuracy than the original code.
            Verify the correctness. SYCL math built-in function rounding mode is
            aligned with OpenCL C 1.2 standard.
            */
            v = v + shared[j * COMBINE + ROWS + lane];
            h[j] = sycl::fmax(v, 0.0f);
        }
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        float sum = h[0] + h[1] + h[2] + h[3];
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        if (bias) sum = sum + bias[row];
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        sum = sum + (row == full && n % R ? 1e9f : 0.0f);
        // The live causal mask is +0. Invalid/padded cells are never exported.
        /*
        DPCT1013: The rounding mode could not be specified and the
        generated code may have different accuracy than the original code.
        Verify the correctness. SYCL math built-in function rounding mode is
        aligned with OpenCL C 1.2 standard.
        */
        sum = sum + 0.0f;
#pragma unroll
        for (int i = row * R; i < n && i < (row + 1) * R; ++i) cells[i] = sum;
    }
#endif
}
#else
// gfx1100 has no CUDA ldmatrix/mma instruction sequence. Keep the same entry point and
// score contract with an ordered scalar F32 dot for each indexer head. The four heads
// run independently; their ReLU'd scores are then added in the documented head order.
// This path favors a well-defined fallback over pretending the CUDA PTX is portable.
__global__ void scalar_score_kernel(
        const float* __restrict__ pooled,const float* __restrict__ query,
        const float* __restrict__ bias,const int32_t* __restrict__ step,
        int max_cells,float* __restrict__ cells) {
    const int n=step[kStepNKv],full=step[kStepNBid];
    if(n<1||n>max_cells||step[kStepPos]!=n-1||full!=n/R||
       step[kStepWidth]!=(n<2051?n:2051))return;
    const int row=blockIdx.x;
    if(row>full)return;
    __shared__ float head_score[HEADS];
    const int head=threadIdx.x;
    if(head<HEADS){
        float dot=0.0f;
#pragma unroll
        for(int d=0;d<D;++d)
            dot=__fmaf_rn(pooled[size_t(row)*D+d],query[size_t(head)*D+d],dot);
        head_score[head]=dot>0.0f?dot:0.0f;
    }
    __syncthreads();
    if(head==0){
        float sum=(0.0f + head_score[0]);
        sum=(sum + head_score[1]);
        sum=(sum + head_score[2]);
        sum=(sum + head_score[3]);
        if(bias)sum=(sum + bias[row]);
        sum=(sum + (row==full&&n%R?1e9f:0.0f));
        sum=(sum + 0.0f);
        for(int i=row*R;i<n&&i<(row+1)*R;++i)cells[i]=sum;
    }
}
#endif
struct Span{const void* p;size_t n;};
void validate(Span s){
    const auto p=reinterpret_cast<uintptr_t>(s.p);
    if(!p||p%4||s.n>UINTPTR_MAX-p)throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a,Span b){
    const auto x=reinterpret_cast<uintptr_t>(a.p),y=reinterpret_cast<uintptr_t>(b.p);
    return x<y+b.n&&y<x+a.n;
}
} // namespace
void native_qsa_score_set_enabled(bool value){enabled.store(value,std::memory_order_relaxed);}
bool native_qsa_score_enabled(){return enabled.load(std::memory_order_relaxed);}
void native_qsa_score(const float* pooled,const float* query,const float* bias,
                      const QsaShapes& s,const int32_t* step,int64_t max_blocks,int64_t max_cells,
                      float* cells,void* stream){
    if(!stream||s.idx_dim!=D||s.idx_n_head!=HEADS||s.idx_block!=R||s.idx_top_k!=2048||
       max_cells<1||max_cells>INT32_MAX-3||max_blocks!=max_cells/R+1)
        throw std::invalid_argument("native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[]={{pooled,size_t(max_blocks)*D*4},{query,HEADS*D*4},
        {step,kStepCount*4},{cells,size_t(max_cells)*4},{bias,bias?size_t(max_blocks)*4:0}};
    const int count=bias?5:4;
    for(int i=0;i<count;++i)validate(spans[i]);
    for(int i=0;i<count;++i)for(int j=i+1;j<count;++j)
        if(overlaps(spans[i],spans[j]))throw std::invalid_argument("native QSA score spans overlap");
#if defined(__HIPCC__)
    scalar_score_kernel<<<unsigned(max_blocks),128,0,static_cast<cudaStream_t>(stream)>>>(
        pooled,query,bias,step,int(max_cells),cells);
#else
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        ((sycl::queue *)(strata::q_of(stream)))
            ->parallel_for<dpct_kernel_name<class score_kernel_59cd1b>>(
                sycl::nd_range<3>(
                    sycl::range(1, 1,
                                unsigned((max_blocks + ROWS - 1) / ROWS)) *
                        sycl::range(1, WARPS, 32),
                    sycl::range(1, WARPS, 32)),
                exp_props,
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(32)]] {
                        score_kernel(pooled, query, bias, step, int(max_cells),
                                     cells);
                    });
    }
#endif
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
