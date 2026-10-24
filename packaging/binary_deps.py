#!/usr/bin/env python3
"""What an engine file needs from the machine it runs on: PE (Windows), ELF (Linux) and Mach-O (macOS) import lists, read
with no tools and no third-party packages, and the rule "everything imported is bundled or belongs to the OS or the GPU driver".

Why: on 2026-10-17 a tester's clean Windows 11 PC failed to start the engine with 0xC0000135 (a DLL was not found): the
package held strata.exe and strata_kq.dll but not cublas64_13.dll / cublasLt64_13.dll, which strata.exe imports. Our own
clean-profile test passed only because the build PC has the CUDA Toolkit on PATH. The packager now refuses a package whose imports
are not covered, and `--check-deps` / the test suite run the same rule on any finished package.

    python packaging/binary_deps.py <package folder>          # exit 0 when every import is bundled or system, 1 with the list

Library names are compared case-insensitively on Windows and by soname elsewhere.
"""

from __future__ import annotations

import fnmatch
import struct
import sys
from pathlib import Path

# ---- Windows ---------------------------------------------------------------------------------------------------------------
# Libraries Windows itself ships (a clean Windows 10/11 has them) or the GPU driver installs in System32. Anything else an
# engine file imports (the Visual C++ runtime, CUDA, HIP, SYCL/Level Zero loaders, OpenMP...) must travel in the package.
WINDOWS_OS = (
    "kernel32.dll", "kernelbase.dll", "user32.dll", "gdi32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll",
    "ws2_32.dll", "wsock32.dll", "mswsock.dll", "iphlpapi.dll", "bcrypt.dll", "crypt32.dll", "secur32.dll", "sechost.dll",
    "ntdll.dll", "msvcrt.dll", "ucrtbase.dll", "rpcrt4.dll", "shlwapi.dll", "setupapi.dll", "cfgmgr32.dll", "version.dll",
    "winmm.dll", "imm32.dll", "comctl32.dll", "comdlg32.dll", "dbghelp.dll", "psapi.dll", "pdh.dll", "powrprof.dll",
    "userenv.dll", "netapi32.dll", "wininet.dll", "winhttp.dll", "dxgi.dll", "d3d11.dll", "d3d12.dll", "dwmapi.dll",
    "dxcore.dll", "wtsapi32.dll", "ncrypt.dll", "normaliz.dll", "api-ms-win-*.dll", "ext-ms-win-*.dll", "hid.dll",
    "opengl32.dll", "glu32.dll", "wldap32.dll", "authz.dll", "msi.dll", "propsys.dll", "uxtheme.dll", "dnsapi.dll",
)
# Installed by the GPU driver into System32 (never redistributable by us): the CUDA driver API, the NVIDIA management library,
# and the AMD / Intel user-mode driver pieces the driver package owns.
WINDOWS_DRIVER = (
    "nvcuda.dll", "nvml.dll", "nvapi64.dll", "nvapi.dll", "nvcuvid.dll", "nvEncodeAPI64.dll", "nvfatbinaryloader.dll",
    "amdxx64.dll", "amdhip64_*.dll", "amdhip64.dll", "atiadlxx.dll", "atiadlxy.dll", "amdadlx64.dll",
    "igdrcl64.dll", "ze_intel_gpu64.dll", "ze_loader.dll", "OpenCL.dll", "vulkan-1.dll",
)

# ---- Linux -----------------------------------------------------------------------------------------------------------------
# Every distribution has these (glibc and the compiler runtime). Names are sonames or glob patterns.
LINUX_OS = (
    "linux-vdso.so.1", "ld-linux-x86-64.so.2", "ld-linux-aarch64.so.1", "libc.so.6", "libm.so.6", "libdl.so.2", "libpthread.so.0",
    "librt.so.1", "libutil.so.1", "libresolv.so.2", "libnss_*.so.*", "libgcc_s.so.1", "libstdc++.so.6", "libgomp.so.1",
    "libatomic.so.1", "libz.so.1", "libnuma.so.1", "libdrm.so.2", "libdrm_amdgpu.so.1", "libelf.so.1", "libtinfo.so.*",
    "libcrypt.so.1", "libffi.so.*", "libpci.so.3", "libudev.so.1", "libX11.so.6", "libxcb.so.1",
)
# From the GPU driver package: never bundled.
LINUX_DRIVER = (
    "libcuda.so.1", "libcuda.so", "libnvidia-*.so*", "libnvcuvid.so*", "libze_intel_gpu.so*", "libze_loader.so*",
    "libOpenCL.so*", "libvulkan.so*", "libGLX*.so*", "libEGL*.so*", "libamdgpu*.so*", "libhsa-runtime64.so*",
    "libdrm_amdgpu.so*", "libigdrcl.so*", "libnvidia-ml.so*",
)

# ---- macOS -----------------------------------------------------------------------------------------------------------------
MAC_OS_PREFIXES = ("/usr/lib/", "/System/Library/")


def _match(name: str, patterns: tuple[str, ...], fold: bool) -> bool:
    n = name.lower() if fold else name
    return any(fnmatch.fnmatchcase(n, (p.lower() if fold else p)) for p in patterns)


# ---- readers ---------------------------------------------------------------------------------------------------------------
def pe_imports(path: Path) -> list[str]:
    """DLL names a PE file imports at load time (import table) plus delay-load imports (loaded at first call)."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"{path}: not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError(f"{path}: bad PE header")
    nsec, = struct.unpack_from("<H", data, pe + 6)
    opt_size, = struct.unpack_from("<H", data, pe + 20)
    opt = pe + 24
    magic, = struct.unpack_from("<H", data, opt)
    dd = opt + (112 if magic == 0x20B else 96)
    sections = []
    sec = opt + opt_size
    for i in range(nsec):
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, sec + i * 40 + 8)
        sections.append((va, max(vsize, rsize), raw))

    def off(rva: int) -> int:
        for va, size, raw in sections:
            if va <= rva < va + size:
                return raw + rva - va
        raise ValueError(f"{path}: RVA {rva:#x} is outside every section")

    def cstr(o: int) -> str:
        end = data.index(b"\0", o)
        return data[o:end].decode("ascii", "replace")

    names: list[str] = []
    imp_rva, imp_size = struct.unpack_from("<II", data, dd + 8)
    if imp_rva and imp_size:
        o = off(imp_rva)
        while True:
            _, _, _, name_rva, _ = struct.unpack_from("<IIIII", data, o)
            if not name_rva:
                break
            names.append(cstr(off(name_rva)))
            o += 20
    delay_rva, delay_size = struct.unpack_from("<II", data, dd + 13 * 8)
    if delay_rva and delay_size:
        o = off(delay_rva)
        while True:
            attrs, name_rva = struct.unpack_from("<II", data, o)
            if not name_rva:
                break
            names.append(cstr(off(name_rva)))
            o += 32
    return names


def elf_needed(path: Path) -> list[str]:
    """DT_NEEDED sonames of an ELF file (what ldd lists); libraries opened with dlopen are not visible here."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF":
        raise ValueError(f"{path}: not an ELF file")
    is64 = data[4] == 2
    le = "<" if data[5] == 1 else ">"
    if is64:
        shoff, = struct.unpack_from(le + "Q", data, 0x28)
        shentsize, shnum = struct.unpack_from(le + "HH", data, 0x3A)
    else:
        shoff, = struct.unpack_from(le + "I", data, 0x20)
        shentsize, shnum = struct.unpack_from(le + "HH", data, 0x2E)
    out: list[str] = []
    for i in range(shnum):
        base = shoff + i * shentsize
        if is64:
            stype, = struct.unpack_from(le + "I", data, base + 4)
            offset, size, link = struct.unpack_from(le + "QQI", data, base + 0x18)
        else:
            stype, = struct.unpack_from(le + "I", data, base + 4)
            offset, size, link = struct.unpack_from(le + "III", data, base + 0x10)
        if stype != 6:      # SHT_DYNAMIC
            continue
        sbase = shoff + link * shentsize
        stroff = struct.unpack_from(le + ("Q" if is64 else "I"), data, sbase + (0x18 if is64 else 0x10))[0]
        step = 16 if is64 else 8
        for o in range(offset, offset + size, step):
            tag, val = struct.unpack_from(le + ("qQ" if is64 else "iI"), data, o)
            if tag == 0:
                break
            if tag == 1:
                s = stroff + val
                out.append(data[s:data.index(b"\0", s)].decode("ascii", "replace"))
    return out


def macho_dylibs(path: Path) -> list[str]:
    """Install names a Mach-O file loads (LC_LOAD_DYLIB, weak, reexport). A fat file reads its first slice."""
    data = path.read_bytes()
    magic = struct.unpack_from(">I", data, 0)[0]
    base = 0
    if magic in (0xCAFEBABE, 0xCAFEBABF):
        base = struct.unpack_from(">I", data, 16)[0] if magic == 0xCAFEBABE else struct.unpack_from(">Q", data, 24)[0]
        magic = struct.unpack_from(">I", data, base)[0]
    if struct.unpack_from("<I", data, base)[0] not in (0xFEEDFACF, 0xFEEDFACE):
        raise ValueError(f"{path}: not a Mach-O file")
    is64 = struct.unpack_from("<I", data, base)[0] == 0xFEEDFACF
    ncmds, = struct.unpack_from("<I", data, base + 16)
    o = base + (32 if is64 else 28)
    out = []
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, o)
        if cmd in (0xC, 0x80000018, 0x8000001F):     # LOAD_DYLIB, LOAD_WEAK_DYLIB, REEXPORT_DYLIB
            name_off, = struct.unpack_from("<I", data, o + 8)
            s = o + name_off
            out.append(data[s:data.index(b"\0", s)].decode("utf-8", "replace"))
        o += size
    return out


def imports(path: Path) -> tuple[str, list[str]]:
    """(kind, names) for a binary, or ("", []) when it is not an executable format this module reads."""
    with path.open("rb") as f:
        head = f.read(4)
    if head[:2] == b"MZ":
        return "pe", pe_imports(path)
    if head == b"\x7fELF":
        return "elf", elf_needed(path)
    if head in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"):
        return "macho", macho_dylibs(path)
    return "", []


# ---- the rule --------------------------------------------------------------------------------------------------------------
def uncovered(folder: Path, files: list[str] | None = None) -> list[tuple[str, str]]:
    """(file, imported library) pairs that are neither in the package nor the OS's nor the GPU driver's. Empty = a clean PC runs it."""
    folder = Path(folder)
    rels = files if files is not None else [p.relative_to(folder).as_posix() for p in sorted(folder.rglob("*")) if p.is_file() and not p.is_symlink()]
    # Windows loads a DLL from the program's own folder, System32 and PATH; a DLL in a sub-folder counts only when BUILD.json lists
    # that folder in "lib_dirs" (Control and the app put those folders on PATH when they start the engine). Found 2026-10-18: the
    # RX 9000 package kept rocblas.dll under rocm/bin, which only Control's PATH finds, and a plain start died with 0xC0000135.
    lib_dirs = {""}
    if any(r == "BUILD.json" for r in rels):
        try:
            import json
            lib_dirs |= {str(d).strip("/\\").replace("\\", "/") for d in json.loads((folder / "BUILD.json").read_text(encoding="utf-8")).get("lib_dirs", [])}
        except (OSError, ValueError):
            pass
    bundled_win = {Path(r).name.lower() for r in rels if ("" if Path(r).parent.as_posix() == "." else Path(r).parent.as_posix()) in lib_dirs}
    bundled = {Path(r).name for r in rels}
    problems: list[tuple[str, str]] = []
    for rel in rels:
        p = folder / rel
        if p.suffix.lower() in (".json", ".txt", ".md", ".cu", ".h", ".py", ".yaml", ".yml", ".metallib", ".bin"):
            continue
        try:
            kind, names = imports(p)
        except (ValueError, OSError, struct.error):
            continue
        for name in names:
            if kind == "pe":
                if name.lower() in bundled_win or _match(name, WINDOWS_OS, True) or _match(name, WINDOWS_DRIVER, True):
                    continue
            elif kind == "elf":
                if name in bundled or _match(name, LINUX_OS, False) or _match(name, LINUX_DRIVER, False):
                    continue
            else:
                base = name.rsplit("/", 1)[-1]
                if name.startswith(MAC_OS_PREFIXES) or base in bundled:
                    continue
            problems.append((rel, name))
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    bad = uncovered(Path(argv[1]))
    for rel, name in bad:
        print(f"{rel}: needs {name}, which is not in the package and not an OS or driver library")
    if not bad:
        print(f"{argv[1]}: every imported library is bundled or belongs to the OS / GPU driver")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
