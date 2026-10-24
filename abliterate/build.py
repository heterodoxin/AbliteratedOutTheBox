"""Build this Strata checkout with the native abliteration hook."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(*args):
    subprocess.run([str(a) for a in args], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--ggml-dir", type=Path, help="optional existing tensor-library source; otherwise Strata fetches its pinned dependency")
    parser.add_argument("--backend", choices=["hip", "cuda"], required=True)
    parser.add_argument("--arch", help="HIP architecture (e.g. gfx1201) or CUDA compute capability (e.g. 89)")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8))
    parser.add_argument("--test", action="store_true", help="build and run native GPU parity tests")
    parser.add_argument("cmake_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    source = args.source.resolve()
    if not (source / "src/kernels/cuda/abliterate.cu").is_file():
        parser.error("--source must point to this Strata fork with the native hook")
    build = source / "build-abliterate"
    options = [f"-DSTRATA_ENABLE_{args.backend.upper()}=ON", "-DCMAKE_BUILD_TYPE=Release",
               f"-DSTRATA_GGML_DIR={args.ggml_dir.resolve() if args.ggml_dir else ''}", "-DSTRATA_BUILD_TESTS=OFF",
               f"-DSTRATA_BUILD_ABLITERATE_TESTS={'ON' if args.test else 'OFF'}"]
    if args.backend == "hip":
        options += [f"-DCMAKE_HIP_ARCHITECTURES={args.arch or 'gfx1201'}"]
        compiler = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/clang++"
        if compiler.is_file():
            options += [f"-DCMAKE_HIP_COMPILER={compiler}"]
    elif args.arch:
        options += [f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}"]
    extra = args.cmake_args[1:] if args.cmake_args[:1] == ["--"] else args.cmake_args
    run("cmake", "-S", source, "-B", build, *options, *extra)
    targets = ["strata"] + (["abliterate-parity"] if args.test else [])
    run("cmake", "--build", build, "--target", *targets, "-j", str(args.jobs))
    if args.test:
        run(build / ("abliterate-parity.exe" if os.name == "nt" else "abliterate-parity"))
    print(f"Strata engine: {build / ('strata.exe' if os.name == 'nt' else 'strata')}")


if __name__ == "__main__":
    main()
