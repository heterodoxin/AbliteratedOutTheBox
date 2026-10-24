#!/usr/bin/env bash
# SYCLomatic needs CUDA SDK headers to parse the sources (no GPU, no toolkit). The pip wheels carry them.
set -euo pipefail
out=${1:-$PWD/cuda-headers}      # mount this at /cuda-headers in the dev image
py=${PYTHON:-python3}          # needs pip
mkdir -p "$out/wheels" "$out/include"
$py -m pip download -q --no-deps -d "$out/wheels" \
  nvidia-cuda-runtime-cu12==12.8.90 nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cccl-cu12==12.8.90 nvidia-curand-cu12==10.3.9.90
for w in "$out"/wheels/*.whl; do
  $py - "$w" "$out/include" <<'PY'
import sys, zipfile, os
w, dst = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(w)
n = 0
for m in z.namelist():
    if "/include/" in m and not m.endswith("/"):
        rel = m.split("/include/", 1)[1]
        p = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.write(z.read(m))
        n += 1
print(os.path.basename(w), n, "headers")
PY
done
ls "$out/include" | head -50
for h in cuda.h cuda_runtime.h cuda_runtime_api.h crt/host_defines.h cuda_fp16.h cuda_bf16.h cublas_v2.h; do
  [ -f "$out/include/$h" ] && echo "ok   $h" || echo "MISS $h"; done
grep -m1 CUDA_VERSION "$out/include/cuda.h" || true
