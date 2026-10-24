"""Download the pinned Strata sources, install the native hook, and build the engine."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

REVISION = "8183c6b67162340c09b67360860177c577a65f2e"
UPSTREAM = "https://github.com/Niko1221/Strata.git"
HERE = Path(__file__).resolve().parent
FILES = {
    "abliterate.hpp": "include/strata/kernels/abliterate.hpp",
    "abliterate.cu": "src/kernels/cuda/abliterate.cu",
    "abliterate-parity.cpp": "src/kernels/abliterate-parity.cpp",
}


def run(*args, cwd=None, capture=False):
    result = subprocess.run([str(a) for a in args], cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def prepare(source):
    source = source.resolve()
    if not source.exists():
        run("git", "clone", "--no-checkout", "--depth", "1", UPSTREAM, source)
        run("git", "fetch", "--depth", "1", "origin", REVISION, cwd=source)
        run("git", "checkout", "--detach", REVISION, cwd=source)
    if run("git", "rev-parse", "HEAD", cwd=source, capture=True) != REVISION:
        raise RuntimeError(f"Strata checkout must be at {REVISION}; use a new --source directory")
    patch = HERE / "patches/strata-hook.patch"
    applied = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not applied:
        run("git", "apply", "--check", patch, cwd=source)
        run("git", "apply", patch, cwd=source)
    for name, destination in FILES.items():
        target = source / destination
        data = (HERE / "native" / name).read_bytes()
        if target.exists() and target.read_bytes() != data:
            raise RuntimeError(f"Refusing to overwrite a different hook file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / ".strata-src")
    parser.add_argument("--backend", choices=["hip", "cuda"], required=True)
    parser.add_argument("--arch", help="HIP architecture (e.g. gfx1201) or CUDA compute capability (e.g. 89)")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 8))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--test", action="store_true", help="build and run native GPU parity tests")
    parser.add_argument("cmake_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    source = prepare(args.source)
    if args.prepare_only:
        print(source)
        return
    build = source / "build-abliterate"
    options = [f"-DSTRATA_ENABLE_{args.backend.upper()}=ON", "-DCMAKE_BUILD_TYPE=Release",
               f"-DSTRATA_GGML_DIR={HERE.parent}", "-DSTRATA_BUILD_TESTS=OFF",
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
    print(f"Patched Strata engine: {build / 'strata'}")


if __name__ == "__main__":
    main()
