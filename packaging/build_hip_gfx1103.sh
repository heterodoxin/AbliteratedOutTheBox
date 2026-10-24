#!/bin/bash
# Linux HIP engine package for the Radeon 780M (gfx1103), built on the mini PC (P2006). Needs the source tree as $HOME/split-tq/tq3-src.tar.gz
# (git archive), the TheRock gfx110X-all SDK at $HOME/strata-poc/rocm/install and the llama.cpp sources at $GGML; edit the paths for another box.
# ROCm = TheRock gfx110X-all SDK of ~/strata-poc (as the earlier P2006 builds). Shared box lock for the build; nice.
R=$HOME/split-tq
S=$R/build.status
st() { echo "$* $(date +%T)" >> $S; }
SDK=$HOME/strata-poc/rocm/install
export ROCM_PATH=$SDK HIP_PATH=$SDK HIP_PLATFORM=amd
export PATH=$SDK/bin:$SDK/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:$SDK/lib/rocm_sysdeps/lib:$SDK/lib/llvm/lib:${LD_LIBRARY_PATH:-}
GGML=$HOME/144-igpu/src-c/build/_deps/strata_llamacpp-src
mkdir -p $R; cd $R || exit 1
: > $S
st "start"
rm -rf src stage components; mkdir src; tar xzf $R/tq3-src.tar.gz -C src
cd src
cmake -S . -B b1103 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON \
  -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=$GGML -DCMAKE_HIP_ARCHITECTURES=gfx1103 \
  -DCMAKE_HIP_COMPILER=$SDK/lib/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$SDK \
  "-DCMAKE_PREFIX_PATH=$SDK;$SDK/lib/rocm_sysdeps;$SDK/lib/llvm" \
  "-DCMAKE_HIP_FLAGS=--rocm-path=$SDK --rocm-device-lib-path=$SDK/lib/llvm/amdgcn/bitcode" > $R/cfg.log 2>&1
st "configured rc=$?"
flock -s $HOME/gpu.lock nice -n 10 cmake --build b1103 --target strata -j 8 > $R/build.log 2>&1
st "built rc=$?"
[ -x b1103/strata ] || { st "ABORT no exe"; tail -5 $R/build.log >> $S; exit 1; }
STAGE=$R/stage; mkdir -p $STAGE
cp b1103/strata $STAGE/
echo '{"version": "0.1.45", "source": "f-split-tq 322e3fdfb", "archs": ["gfx1103"], "rocm": "TheRock gfx110X-all 7.14.1", "portable": true, "experimental": "Radeon 780M, unvalidated; GPU resets seen (notes-minipc.md)"}' > $STAGE/BUILD.json
python3 packaging/collect_libs.py --binary b1103/strata --search $SDK/lib --search $SDK/lib/rocm_sysdeps/lib --search $SDK/lib/llvm/lib \
  --also 'libamd_comgr.so*' --also 'libhiprtc*.so*' --out $STAGE \
  --data $SDK/lib/rocblas/library=rocblas/library --data $SDK/lib/hipblaslt/library=hipblaslt/library --data-arch gfx1103 > $R/collect.log 2>&1
st "collected rc=$?"
python3 packaging/stage_tuning.py --stage $STAGE --archs gfx1103 --rocm $SDK >> $R/stage_tuning.log 2>&1   # hipblaslt_version in BUILD.json + the matching tuning table
st "tuning tables rc=$?: $(tail -n 1 $R/stage_tuning.log)"
mkdir -p $R/components/engine
python3 packaging/package_engine.py --src $STAGE --backend hip --version 0.1.45 --out $R/components/engine \
  --arch gfx1103 --build-id 322e3fdfb+rocm7.14.1-gfx110X --tree > $R/package.log 2>&1
st "packaged rc=$?"
cat $R/package.log >> $S
st "BUILD-DONE"
