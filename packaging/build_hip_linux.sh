#!/bin/bash
# Linux AMD (HIP) engine package: strata (+ strata-dsv41 for rx9070) + the CPU picture reader (strata-vision) + the GPU picture reader (strata-vision-hip),
# the ROCm libraries both load, and the rocBLAS / hipBLASLt kernel data for the listed archs, packaged as an engine component.
# Run on a Linux box that has ROCm (TheRock Python wheels are enough); the tester packages were built in an Ubuntu 22.04 chroot
# (glibc 2.35) so they run on every distribution since.  All settings are environment variables:
#
#   VARIANT=rx9070     rx9070 (gfx1200 gfx1201) | rx7900 (gfx1100 gfx1101 gfx1102) | rx6000 (gfx1030 gfx1031) | halo (gfx1151) | igpu (gfx1103 gfx1150) | a name + ARCHS=
#   ARCHS="gfx1200;gfx1201"          the code objects (default: the variant's)
#   SRC=<this checkout>              the Strata source (default: the checkout this script is in)
#   LLAMA=<llama.cpp checkout>       at the pinned commit (setup.py LLAMA_CPP_COMMIT); mtmd and ggml come from it
#   HP=<ROCm devel root>  CORE=<ROCm core root>  LIBW=<ROCm libraries root>   (TheRock: _rocm_sdk_devel, _rocm_sdk_core, _rocm_sdk_libraries_*)
#   OUT=<work folder>  PKGVER=0.1.48  COMMIT=<source commit for the build id>  JOBS=16
#   VISION_ONLY=1                    build only the two readers (no engine, no package)
#   DSV41=1|0                        also build strata-dsv41 (the DeepSeek-V4.1 engine, tools/dsv41/build_hip_linux.sh's flags) and ship it beside
#                                    strata: default 1 for the rx9070 variant (gfx1200 gfx1201; built and run on an R9700), 0 for the others
#   NO_PACKAGE=1                     stop after the stage folder
#   VISION_HIP_DEFS="-DGGML_CUDA_FA=OFF"   extra cmake definitions for the GPU reader only
#
# Both readers get RUNPATH $ORIGIN (tools/vision/CMakeLists.txt), and every bundled library is patched to $ORIGIN, so the
# component folder runs from anywhere with nothing from the environment (packaging/binary_deps.py checks it).
set -u
V=${VARIANT:?VARIANT=rx9070|rx7900|rx6000|halo|igpu|<name> with ARCHS=}
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:?LLAMA=<llama.cpp checkout at the pinned commit>}
HP=${HP:?HP=<ROCm devel root>}
CORE=${CORE:-$HP}
LIBW=${LIBW:-$HP}
OUT=${OUT:-$PWD/work-$V}
PKGVER=${PKGVER:-0.1.46}
COMMIT=${COMMIT:-unknown}
JOBS=${JOBS:-16}
case "$V" in
  rx9070) ARCHS=${ARCHS:-gfx1200;gfx1201} ;;
  rx7900) ARCHS=${ARCHS:-gfx1100;gfx1101;gfx1102} ;;
  rx6000) ARCHS=${ARCHS:-gfx1030;gfx1031} ;;   # RDNA2: no WMMA (plain kernels), no hipBLASLt kernels; built, not tested on a real RX 6000 card
  halo)   ARCHS=${ARCHS:-gfx1151} ;;
  igpu)   ARCHS=${ARCHS:-gfx1103;gfx1150} ;;
  *)      ARCHS=${ARCHS:?ARCHS=gfx...} ;;
esac
NOTE_EXTRA=""; [ "$V" = rx6000 ] && NOTE_EXTRA=" (built, not tested on a real RX 6000 card)"
if [ -z "${DSV41:-}" ]; then DSV41=0; [ "$V" = rx9070 ] && DSV41=1; fi
ARCHLIST=${ARCHS//;/ }
DATAARCH=${ARCHS//;/,}
mkdir -p "$OUT"; S=$OUT/build.status
st() { echo "$* $(date +%T)" >> "$S"; }
: > "$S"
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP
# TheRock 7.14.x dists (the Strix Halo / gfx1151 package) keep libnuma, libz, libdrm ... in lib/rocm_sysdeps; older wheels have no such folder
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
HIPDEFS=(-DCMAKE_HIP_COMPILER=$HP/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$HP "-DCMAKE_PREFIX_PATH=$HP;$LIBW$SYSDEPS"
         "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$HP/lib/llvm/amdgcn/bitcode")
st "start $V ($ARCHS) commit $COMMIT"
cd "$SRC" || exit 1
if [ -z "${VISION_ONLY:-}" ]; then
  cmake -G Ninja -S . -B "$OUT/b" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_PREFILL_MMQ=ON "-DCMAKE_HIP_ARCHITECTURES=$ARCHS" "${HIPDEFS[@]}" -DSTRATA_GGML_DIR="$LLAMA" -DSTRATA_PORTABLE=ON \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags" > "$OUT/cfg.log" 2>&1
  st "engine configured rc=$?"
  nice -n 10 cmake --build "$OUT/b" --target strata -j "$JOBS" > "$OUT/build.log" 2>&1
  st "engine built rc=$?"
  [ -x "$OUT/b/strata" ] || { st "ABORT no strata"; exit 1; }
  if [ "$DSV41" = 1 ]; then      # own build tree: the engine's flags stay as they were; STRATA_NATIVE_EXPERTS=ON is what the DeepSeek path was built with
    cmake -G Ninja -S . -B "$OUT/bd" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF       -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON "-DCMAKE_HIP_ARCHITECTURES=$ARCHS" "${HIPDEFS[@]}" -DSTRATA_GGML_DIR="$LLAMA" -DSTRATA_PORTABLE=ON       "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags" > "$OUT/cfg-dsv41.log" 2>&1
    st "dsv41 configured rc=$?"
    nice -n 10 cmake --build "$OUT/bd" --target strata-dsv41 -j "$JOBS" > "$OUT/build-dsv41.log" 2>&1
    st "dsv41 built rc=$?"
    [ -x "$OUT/bd/strata-dsv41" ] || { st "ABORT no strata-dsv41 (DSV41=0 leaves it out)"; tail -20 "$OUT/build-dsv41.log" >> "$S"; exit 1; }
  fi
fi
# the CPU reader: portable (AVX2), the fallback that runs on any PC
cmake -G Ninja -S tools/vision -B "$OUT/bv" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$LLAMA" -DSTRATA_VISION_CUDA=OFF \
  -DSTRATA_PORTABLE=ON > "$OUT/cfg-vision.log" 2>&1
nice -n 10 cmake --build "$OUT/bv" --target strata-vision -j "$JOBS" > "$OUT/build-vision.log" 2>&1
st "cpu reader built rc=$?"
[ -x "$OUT/bv/bin/strata-vision" ] || { st "ABORT no strata-vision"; exit 1; }
# the GPU reader: ggml-hip for the same archs; the CPU kernels in it are portable too
cmake -G Ninja -S tools/vision -B "$OUT/bh" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$LLAMA" -DSTRATA_VISION_HIP=ON \
  "-DSTRATA_VISION_HIP_ARCHS=$ARCHS" -DSTRATA_PORTABLE=ON "${HIPDEFS[@]}" ${VISION_HIP_DEFS:-} > "$OUT/cfg-vision-hip.log" 2>&1
st "gpu reader configured rc=$?"
nice -n 10 cmake --build "$OUT/bh" --target strata-vision -j "$JOBS" > "$OUT/build-vision-hip.log" 2>&1
st "gpu reader built rc=$?"
[ -x "$OUT/bh/bin/strata-vision-hip" ] || { st "ABORT no strata-vision-hip"; tail -20 "$OUT/build-vision-hip.log" >> "$S"; exit 1; }
[ -n "${VISION_ONLY:-}" ] && { st "VISION-DONE"; exit 0; }
STAGE=$OUT/stage; rm -rf "$STAGE"; mkdir -p "$STAGE"
ARCHJSON=$(for a in $ARCHLIST; do printf '"%s", ' "$a"; done | sed 's/, $//')
cat > "$STAGE/BUILD.json" <<EOJ
{"version": "$PKGVER", "source": "f-amd-vision $COMMIT", "variant": "$V", "archs": [$ARCHJSON], "portable": true, "vision": "gpu", "vision_cpu": "strata-vision", "vision_gpu": "strata-vision-hip", "vision_archs": [$ARCHJSON], "dsv41": $([ "$DSV41" = 1 ] && echo true || echo false)}
EOJ
# the libraries the engine and the GPU reader load (the CPU reader needs only the system's), collected from the build outputs
DATA=(--data "$LIBW/lib/rocblas/library=rocblas/library" --data "$LIBW/lib/hipblaslt/library=hipblaslt/library" --data-arch "$DATAARCH")
BINS=("$OUT/b/strata" "$OUT/bh/bin/strata-vision-hip"); [ "$DSV41" = 1 ] && BINS+=("$OUT/bd/strata-dsv41")
for b in "${BINS[@]}"; do
  python3 packaging/collect_libs.py --binary "$b" --search "$LIBW/lib" --search "$HP/lib" --search "$CORE/lib" --search "$CORE/lib/llvm/lib" ${SYSDEPS:+--search "$HP/lib/rocm_sysdeps/lib"} \
    --also 'libamd_comgr.so*' --also 'libhiprtc*.so*' --out "$STAGE" \
    "${DATA[@]}" >> "$OUT/collect.log" 2>&1
  st "collected for $(basename "$b") rc=$?"
  DATA=()                                              # the kernel data once
done
cp "$OUT/b/strata" "$OUT/bv/bin/strata-vision" "$OUT/bh/bin/strata-vision-hip" "$STAGE/"
[ "$DSV41" = 1 ] && cp "$OUT/bd/strata-dsv41" "$STAGE/"
for f in "$STAGE"/*.so "$STAGE"/*.so.*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"; done
for f in "$STAGE/strata" "$STAGE/strata-vision" "$STAGE/strata-vision-hip" $([ "$DSV41" = 1 ] && echo "$STAGE/strata-dsv41"); do
  st "RUNPATH $(basename "$f"): $(readelf -d "$f" | grep -E 'RPATH|RUNPATH' | cut -d'[' -f2 | tr -d ']' | paste -sd' ')"
done
# the hipBLASLt version of this ROCm into BUILD.json, and the tuning tables of tools/hip that match it (architecture and version) into tuning/
python3 packaging/stage_tuning.py --stage "$STAGE" --archs "$ARCHLIST" --rocm "$HP" >> "$OUT/stage_tuning.log" 2>&1
st "tuning tables rc=$?: $(tail -n 1 "$OUT/stage_tuning.log")"
[ -n "${NO_PACKAGE:-}" ] && { st "STAGE-DONE"; exit 0; }
rm -rf "$OUT/components"; mkdir -p "$OUT/components/engine"
ARCHARGS=""; for a in $ARCHLIST; do ARCHARGS="$ARCHARGS --arch $a"; done
REQ=""; [ "$DSV41" = 1 ] && REQ="--require-dsv41"
python3 packaging/package_engine.py --src "$STAGE" --backend hip --version "$PKGVER" --out "$OUT/components/engine" $ARCHARGS \
  --variant "$V" --build-id "$COMMIT+rocm" --tree $REQ --notes "AMD $V build: engine, CPU picture reader, GPU picture reader$([ "$DSV41" = 1 ] && echo ", DeepSeek-V4.1 engine")${NOTE_EXTRA:-}" > "$OUT/package.log" 2>&1
st "packaged rc=$?"
cat "$OUT/package.log" >> "$S"
st "BUILD-DONE"
