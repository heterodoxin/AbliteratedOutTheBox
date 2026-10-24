#!/bin/sh
# Build the Linux CUDA engine package for the release feed.
#
#   VARIANT=cuda12 packaging/build_cuda_linux.sh     # CUDA 12.9: GTX 10 (Pascal), Tesla P100, V100, and the newer cards up to sm_89
#   VARIANT=cuda13 packaging/build_cuda_linux.sh     # CUDA 13: RTX 20 and newer (sm_75 .. sm_120)
#
# Run it on a Linux x86_64 box with the CUDA toolkit of that variant on PATH (nvcc), a Python 3 venv with requirements.txt and
# requirements-build.txt, and the pinned llama.cpp in third_party/llama.cpp (python -c "import setup; setup.get_llama_cpp()").
# The package lands in $OUT (default ./out-engine-<variant>). Nothing is uploaded.
#
# The cuda12 package is the one the feed and docs/DOCKER.md offer to Pascal and Volta cards, so it needs machine code for sm_60,
# sm_61 and sm_70 and the -DSTRATA_EXPERIMENTAL_SM60=ON build (#3503: the 0.2.1 package had sm_75 / 86 / 89 only, and a P100
# container restarted in a loop with "this PC's GPU is sm_60"). packaging/package_engine.py refuses a cuda12 package without them.
set -eu
variant=${VARIANT:-cuda13}
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
case "$variant" in
  cuda12) archs=${ARCHS:-60;61;70;75;80;86;89}; mindriver=575.51; sm60=ON ;;
  cuda13) archs=${ARCHS:-75;86;89;120}; mindriver=580.65; sm60=OFF ;;
  *) echo "build_cuda_linux.sh: VARIANT must be cuda12 or cuda13" >&2; exit 2 ;;
esac
out=${OUT:-$repo/out-engine-$variant}
build=${BUILD:-$repo/build-$variant}
py=${PYTHON:-python3}
cudafull=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
case "$variant:$cudafull" in
  cuda12:12.*|cuda13:13.*) ;;
  *) echo "build_cuda_linux.sh: $variant needs the matching CUDA toolkit; nvcc here is ${cudafull:-missing}" >&2; exit 2 ;;
esac

real=""; for a in $(echo "$archs" | tr ';,' '  '); do real="$real;${a}-real"; done; real=${real#;}
last=$(echo "$archs" | tr ';,' '  ' | awk '{print $NF}')
cd "$repo"
cmake -G Ninja -S . -B "$build" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DSTRATA_EXPERIMENTAL_SM60=$sm60 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual" -DSTRATA_GGML_DIR="$repo/third_party/llama.cpp"
cmake --build "$build" --target strata strata_kq -j "$(nproc)"

# The picture reader is built for the SAME architectures as the engine (#3669: a reader reused from an older build had no sm_60, and on
# a Pascal card the engine ran while the reader died with "device kernel image is invalid"). BUILD_VISION=0 ships a text-only package.
vision_dir=${BUILD_VISION_DIR:-$repo/build-vision-$variant}
if [ "${BUILD_VISION:-1}" = 1 ]; then
  cmake -G Ninja -S tools/vision -B "$vision_dir" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$repo/third_party/llama.cpp" \
      -DSTRATA_VISION_CUDA=ON -DSTRATA_PORTABLE=ON -DGGML_CUDA_NCCL=OFF "-DCMAKE_CUDA_ARCHITECTURES=${real};${last}-virtual"
  cmake --build "$vision_dir" --target strata-vision -j "$(nproc)"
fi

ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' CMakeLists.txt | head -1)
stage=$(mktemp -d "${TMPDIR:-/tmp}/strata-engine-stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
cp "$build/strata" "$stage/"
[ -f "$build/libstrata_kq.so" ] && cp "$build/libstrata_kq.so" "$stage/"
vision_json=""
if [ -f "$vision_dir/bin/strata-vision" ]; then
  cp "$vision_dir/bin/strata-vision" "$stage/"
  vision_json=", \"vision_archs\": \"$archs\""        # package_engine.py refuses a reader whose archs do not cover the engine's
fi
echo "{\"version\": \"$ver\", \"source\": \"packaging/build_cuda_linux.sh\", \"archs\": \"$archs\", \"cuda\": \"$cudafull\", \"portable\": true$vision_json}" > "$stage/BUILD.json"
"$py" packaging/collect_libs.py --binary "$build/strata" --search /usr/local/cuda/lib64 --out "$stage" --also 'libcublasLt.so*'
arch_args=""; for a in $(echo "$archs" | tr ';,' '  '); do arch_args="$arch_args --arch sm_$a"; done
mkdir -p "$out"
# shellcheck disable=SC2086
"$py" packaging/package_engine.py --src "$stage" --backend cuda --version "$ver" --out "$out" $arch_args --arch "compute_$last" \
    --min-driver nvidia=$mindriver --variant "$variant" --platform linux-x86_64 --build-id "release-${ver}-${variant}-${cudafull}" \
    --include SYSTEM-LIBS.txt --include 'libstrata_kq*'
for d in "$out"/*/; do "$py" packaging/package_engine.py --verify "$d"; "$py" packaging/binary_deps.py "$d"; done
echo "built: $out"
