#!/usr/bin/env bash
# normalize.sh <dpct-out-dir> <upstream-commit>: a fresh migrate.sh output in the port's shape, for a 3-way merge.
#   - dpct's `X.cpp.dp.cpp` becomes `X.cpp` (the .cu -> .dp.cpp names stay), its .yaml/report files go
#   - files dpct left byte-identical to the upstream original go too (sycl/CMakeLists.txt falls back to them)
#   - the DPCTnnnn:<serial> message numbers are dropped (they renumber on every run and would be pure noise)
#   - fixups.py runs on the result
# Updating the port after an upstream merge (docs/INTEL.md "Keeping up with upstream"):
#   migrate the OLD upstream tree to sycl-base and the merged tree to sycl-new, normalize both with their
#   commits, then for every file in both:  git merge-file sycl/F sycl-base/F sycl-new/F
set -euo pipefail
d=$(cd "$1" && pwd); rev=$2; repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$d"
find . -name '*.yaml' -delete; rm -f dpct_report.*
find . -name '*.cpp.dp.cpp' | while read -r f; do mv "$f" "${f%.cpp.dp.cpp}.cpp"; done
n=0
while read -r f; do
  rel=${f#./}; orig=$rel; [[ $rel == *.dp.cpp ]] && orig=${rel%.dp.cpp}.cu
  if git -C "$repo" cat-file -e "$rev:$orig" 2>/dev/null && cmp -s "$f" <(git -C "$repo" show "$rev:$orig"); then
    rm "$f"; n=$((n+1))
  fi
done < <(find src include third_party -type f 2>/dev/null)
find . -type d -empty -delete
find src include -type f -exec sed -i -E 's/DPCT([0-9]{4}):[0-9]+:/DPCT\1:/' {} +
mkdir -p tools && cp "$repo/sycl/tools/fixups.py" tools/
python3 tools/fixups.py > fixups.log 2>&1 || { tail -5 fixups.log; exit 1; }
echo "$d: dropped $n unchanged files, $(find src include -type f | wc -l) remain, fixups applied: $(grep -c fixed fixups.log)"
