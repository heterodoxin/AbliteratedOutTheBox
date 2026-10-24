#!/bin/bash
# strata-dsv41 (the DeepSeek-V4.1 engine) for NVIDIA on Linux, built next to the release engine so package_engine.py can ship it
# beside `strata` (setup.py dsv41_exe() looks for strata-dsv41 in the engine folder).  Same toolchain and RUNPATH rule as the
# engine build: the result finds cudart/cublas beside itself ($ORIGIN), never the toolkit it was built with.
#
#   SRC=<checkout> B=<build dir> CUDA=/usr/local/cuda-13.0 GG=<llama.cpp checkout> [ARCH="86-real;120"] [JOBS=6]  packaging/build_dsv41_cuda.sh
#
# ARCH defaults to the 0.2.4 release shape: SASS for sm_86 (RTX 3090 / 3060) and sm_120 (RTX Blackwell) + PTX sm_120 (the Windows twin
# built and ran on an RTX 5070; the Linux sm_86 build is the one tested on the RTX 3060 box).  Needs CUDA 13 (the sm_120 code); for a
# CUDA 12.9 package pass ARCH="86-real;89-real;89-virtual" or leave strata-dsv41 out (the Pascal cuda12 variant has no DeepSeek path).
# Prints the binary path on the last line; collect its libraries with packaging/collect_libs.py --binary <it> --search $CUDA/lib64.
set -eu
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
B=${B:?B=<build dir>}
CUDA=${CUDA:?CUDA=<toolkit root>}
GG=${GG:?GG=<llama.cpp checkout at the pinned commit>}
ARCH=${ARCH:-86-real;120}
JOBS=${JOBS:-6}
export PATH=$CUDA/bin:$PATH
cmake -G Ninja -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA" -DSTRATA_GGML_DIR="$GG" -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_SKIP_BUILD_RPATH=OFF \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN'
ninja -j "$JOBS" -C "$B" strata-dsv41
[ -x "$B/strata-dsv41" ] || { echo "no $B/strata-dsv41" >&2; exit 1; }
echo "$B/strata-dsv41"
