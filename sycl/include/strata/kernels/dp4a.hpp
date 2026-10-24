// include/strata/kernels/dp4a.hpp - the two instructions the kernels use that Pascal (sm_60) does not have.
//
// The engine is written for compute capability 7.5 and newer and refuses older cards at CMake configure time
// (a Pascal build passes -DSTRATA_EXPERIMENTAL_SM60=ON).  A GP100-class card is compute capability 6.0, which is
// missing exactly two things the i-quant kernels rely on:
//
//   * `__dp4a`, a byte-wise dot product, available from 6.1 (Pascal GP10x/GV11x).  Its fallback below is
//     llama.cpp's own (`ggml/src/ggml-cuda/common.cuh`), which is the reference for the kernels in this
//     directory: they are transcribed from llama.cpp's vecdotq.cuh, and that wrapper's operands are the same
//     sites.  It reinterprets the operands as SIGNED bytes and accumulates in int32, which is what `__dp4a`
//     does for these calls, so the fallback is bit-exact rather than merely close.
//   * `__nanosleep`, available from 7.0 (Volta).  It only paces single-thread doorbell waits, so a loop that
//     spins without it is correct, merely busier.
//
// Both are compile-time selections on `__CUDA_ARCH__`, so one source tree builds for Pascal and for RTX
// 20/30/40/50 without a #define at the call sites.  A HIP build never takes either fallback: `__CUDA_ARCH__` is
// undefined there, so `STRATA_DP4A` is `__dp4a` (which `hip_compat/intrinsics.hpp` provides) and the pause is
// HIP's own `__nanosleep`.
#pragma once

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <time.h>

#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>   // gfx906: the compat layer (__forceinline__, __nanosleep, __dp4a)
#endif

#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 610
__device__ __forceinline__ int strata_dp4a(const int a, const int b, const int c) {
    const int8_t* a8 = (const int8_t*) &a;
    const int8_t* b8 = (const int8_t*) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}
#define STRATA_DP4A(a, b, c) strata_dp4a((a), (b), (c))
#else
#define STRATA_DP4A(a, b, c) dpct::dp4a((a), (b), (c))
#endif

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract, and only pre-Volta CUDA has no `__nanosleep` at all.
__dpct_inline__ void strata_spin_pause() {
#if defined(DPCT_COMPATIBILITY_TEMP) && DPCT_COMPATIBILITY_TEMP < 700
    // sm_6x: the loop spins.  Every call site is a single-thread doorbell wait, so nothing else is delayed.
#else
    /*
    DPCT1008: __nanosleep function is not defined in SYCL. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    // SYCL port: no __nanosleep; the doorbell waits are bounded by strata::kSpinMax instead
#endif
}
