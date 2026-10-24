#!/usr/bin/env bash
# CUDA -> SYCL migration of the Strata engine with SYCLomatic (dpct), run inside strata-sycl-dev.
#   sycl/migrate.sh [repo] [out]      repo defaults to the Strata_B70 checkout beside this dir
# Writes a compilation database first (there is no nvcc here, so CMake cannot make one), then migrates every
# CUDA-touching source into <out> mirroring the tree. Files dpct leaves alone are not copied - the port's
# CMake falls back to the original for those, so the diff against upstream stays the port itself.
set -euo pipefail
repo=${1:-/work/Strata_B70}
out=${2:-$repo/sycl}
hdr=${CUDA_HEADERS:-/cuda-headers/include}
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true; set -u
export PATH=/opt/intel/oneapi/dpcpp-ct/latest/bin:$PATH
cd "$repo"
# the compilation database: every .cu plus every .cpp/.hpp that includes a CUDA header
db_dir=$(mktemp -d)                                        # outside the checkout: nothing to ignore in git
trap 'rm -rf "$db_dir"' EXIT
python3 - "$repo" "$hdr" "$db_dir" <<'PY'
import json, os, re, sys, subprocess
repo, hdr, db_dir = sys.argv[1], sys.argv[2], sys.argv[3]
files = subprocess.check_output(["bash", "-c",
    r"find src -name '*.cu'; grep -rl 'cuda_runtime\|cublas_v2\|cuda_fp16\|cuda\.h' src --include='*.cpp'"],
    cwd=repo, text=True).split()
if os.environ.get("MIGRATE_ONLY"):            # explicit list: host files that reach CUDA only through the headers (verify.cpp, mtp.cpp)
    files = os.environ["MIGRATE_ONLY"].split()
db = []
# not in this tree: the optional ggml MMQ prefill path needs llama.cpp's ggml-cuda sources, and one parity test needs
# ggml-cpu.h. Both stay CUDA-only for now.
skip = {"src/prefill/moe_mmq.cu", "src/prefill/ggml_cuda_host.cu", "src/kernels/native_expert_parity.cpp"}
for f in sorted(set(files) - skip):
    tool = "nvcc" if f.endswith(".cu") else "g++"
    cmd = [tool, "-std=c++20", "-Iinclude", "-Ithird_party/ggml", "-I" + hdr, "-DSTRATA_VERSION=\"port\""]
    if tool == "nvcc": cmd += ["--cuda-gpu-arch=sm_80", "-x", "cuda"]
    cmd += ["-c", f, "-o", f + ".o"]
    db.append({"directory": repo, "file": os.path.join(repo, f), "arguments": cmd})
json.dump(db, open(os.path.join(db_dir, "compile_commands.json"), "w"), indent=1)
print(len(db), "translation units")
PY
mkdir -p "$out"
# --use-experimental-features: graph (cudaGraph -> sycl_ext_oneapi_graph), matrix (mma -> joint_matrix)
# --gen-helper-function: copy the dpct helper headers next to the output so the port builds without the tool
dpct -p "$db_dir" --in-root="$repo" --out-root="$out" \
     --cuda-include-path="$hdr" \
     --use-experimental-features=graph,matrix,logical-group,masked-sub-group-operation,occupancy-calculation,free-function-queries,local-memory-kernel-scope-allocation,bindless_images,virtual_mem,root-group,non-uniform-groups \
     --gen-helper-function --always-use-async-handler --sycl-named-lambda \
     --enable-ctad --optimize-migration --assume-nd-range-dim=3 \
     --report-type=all --report-file-prefix=dpct_report --report-format=csv \
     --stop-on-parse-err=false ${DPCT_EXTRA:-} 2>&1 | tee "$out/migrate.log" | tail -40
echo "MIGRATE EXIT ${PIPESTATUS[0]}"
find "$out" -type f | wc -l
