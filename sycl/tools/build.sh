#!/usr/bin/env bash
# Configure + build the SYCL port inside strata-sycl-dev.  sycl/build.sh [target...]
#   AOT=bmg-g31         ahead-of-time device code (empty: SPIR-V, JIT at the first start)
#   I8_ESIMD=OFF        no fused int8 ESIMD expert kernels (stubs; the prompt's experts take the XMX path). AUTO (default) or ON otherwise.
#   ALLOW_BAD_OCLOC=1   build although this ocloc is known to crash on the Battlemage kernels (see below)
set -uo pipefail
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u

# #3219: ocloc from Compute Runtime 26.22.38646.x (IGC 2.36.3) ends with "IGC: Internal Compiler Error: Floating point exception" on the
# fused int8 expert kernels when it builds for a Battlemage device (bmg-*): 77 device images, then "gen compiler command failed with exit
# code 245" at the last link, after the long compile.  A newer ocloc builds them.  The strata-sycl-dev image keeps whatever ocloc its base image
# had when it was built (the base tag moves; `docker build` reuses a cached base unless it gets --pull).  Say it before the compile, not after.
# Other versions are not judged: the list holds what has been seen to fail.
known_bad_ocloc() {
  case "$1" in
    26.22.38646|26.22.38646.*) return 0 ;;
  esac
  return 1
}
ocloc_version() {
  command -v ocloc >/dev/null 2>&1 || return 1
  ocloc --version 2>/dev/null | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1
}
if [[ "${AOT:-}" =~ ^(bmg|lnl) ]] && [[ "${I8_ESIMD:-AUTO}" != "OFF" ]] && [[ "${ALLOW_BAD_OCLOC:-}" != "1" ]]; then
  v=$(ocloc_version || true)
  if [[ -n "$v" ]] && known_bad_ocloc "$v"; then
    cat >&2 <<EOF
build.sh: this image's ocloc is $v (IGC 2.36.3). It is known to crash while it compiles the int8 ESIMD kernels for AOT=${AOT} (#3219).
  Fix: rebuild the image with a fresh base, then build again:   docker build --pull -t strata-sycl-dev sycl/tools
  Or build without those kernels (the prompt's experts take the slower XMX path):   I8_ESIMD=OFF
  Or build anyway:   ALLOW_BAD_OCLOC=1
EOF
    exit 2
  fi
fi

repo=${REPO:-$(git rev-parse --show-toplevel)}
b=${BUILD_DIR:-$repo/build-sycl}
[ -f $b/build.ninja ] || cmake -S $repo/sycl -B $b -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    -DSTRATA_SYCL_AOT="${AOT:-}" -DSTRATA_SYCL_SPIN_MAX="${SPIN_MAX:-}" -DSTRATA_SYCL_I8_ESIMD="${I8_ESIMD:-AUTO}" 2>&1 | tail -15
# an existing build folder keeps its cache: a switch given now reaches it (only this one, so AOT and the rest stay as configured)
[ -z "${I8_ESIMD:-}" ] || cmake -S $repo/sycl -B $b -DSTRATA_SYCL_I8_ESIMD="$I8_ESIMD" >/dev/null 2>&1
cmake --build $b -j ${JOBS:-12} ${1:+--target "$@"} -- -k 0 2>&1 | tee $b/build.log | grep -E '^FAILED|error:|^ninja: build stopped|Linking|^\[[0-9]+/[0-9]+\] Linking' | tail -40
echo "BUILD EXIT ${PIPESTATUS[0]}"
grep -c 'error:' $b/build.log | sed 's/^/errors: /'
grep -E '^FAILED' $b/build.log | sed 's#.*/##' | sort | uniq | head -60
if grep -q 'IGC: Internal Compiler Error' $b/build.log 2>/dev/null; then
  echo "build.sh: the device compiler (ocloc $(ocloc_version || echo '?')) crashed: try  docker build --pull -t strata-sycl-dev sycl/tools  or  I8_ESIMD=OFF  (#3219)"
fi
