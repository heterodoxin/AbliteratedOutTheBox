// The int8 copy of the trellis codebook (src/kernels/cuda/tq_lut_i8.cuh, unchanged) for the SYCL kernels: a const
// table at namespace scope, which a SYCL kernel reads like ggml's i-quant grids.  The .cuh marks it __device__ for
// CUDA/HIP; here that qualifier is empty for the length of the include.
#pragma once
#ifdef __device__
#error "tq_lut_i8_sycl.hpp: __device__ is already defined"
#endif
#define __device__
#include "../../../../src/kernels/cuda/tq_lut_i8.cuh"
#undef __device__
