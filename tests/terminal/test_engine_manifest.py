"""The engine component manifest (schema 2): its fields, the checks a start makes, the signature hook S6 fills, and
packaging/package_engine.py round trips. Every folder is a throwaway; no GPU tool is run."""

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import components, engine_component as ec, signing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import collect_libs  # noqa: E402
import package_engine  # noqa: E402

WINDOWS = os.name == "nt"
HERE = components.host_platform()
EXE = "strata.exe" if WINDOWS else "strata"
LIB = "cudart64_13.dll" if WINDOWS else "libcudart.so.13"


def make_source(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / EXE).write_bytes(b"engine-binary-" * 40)
    (folder / LIB).write_bytes(b"runtime-library-" * 60)
    (folder / "BUILD.json").write_text(json.dumps({"version": "0.1.45"}))
    (folder / "LICENSE.txt").write_text("MIT")
    (folder / "notes.txt").write_text("not packaged by default")
    (folder / "rocm" / "bin").mkdir(parents=True, exist_ok=True)
    (folder / "rocm" / "bin" / "kernel.co").write_bytes(b"kernel" * 100)
    return folder


class Manifest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.src = make_source(self.root / "src")
        self.out = self.root / "components" / "engine"
        self.out.mkdir(parents=True)

    def package(self, **kw):
        args = dict(archs=("sm_86", "compute_89"), min_driver={"nvidia": "580.65"}, build_id="abc123-cuda13.0")
        args.update(kw)
        return package_engine.build(self.src, "cuda", "0.1.45", HERE, self.out, **args)

    def test_the_packager_writes_every_field_the_plan_asks_for(self):
        folder = self.package()
        data = json.loads((folder / ec.MANIFEST).read_text())
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual((data["id"], data["name"], data["version"]), ("strata-engine", "strata-engine", "0.1.45"))
        self.assertEqual((data["platform"], data["backend"]), (HERE, "cuda"))
        self.assertEqual(data["archs"], ["sm_86", "compute_89"])
        self.assertEqual(data["min_driver"], {"nvidia": "580.65"})
        self.assertEqual(data["build_id"], "abc123-cuda13.0")
        self.assertIn("signature", data)
        self.assertIsNone(data["signature"])
        names = {item["path"] for item in data["files"]}
        self.assertEqual(names, {EXE, LIB, "BUILD.json", "LICENSE.txt"})        # notes.txt and rocm/ need --include / --tree
        for item in data["files"]:
            body = (folder / item["path"]).read_bytes()
            self.assertEqual((item["sha256"], item["bytes"]), (hashlib.sha256(body).hexdigest(), len(body)))
        self.assertEqual(data["entrypoint"], EXE)
        component = ec.read_manifest(folder)
        self.assertEqual((component.build_id, component.archs, component.min_driver),
                         ("abc123-cuda13.0", ("sm_86", "compute_89"), {"nvidia": "580.65"}))
        ec.verify(component, full=True)

    def test_include_and_tree_take_nested_files(self):
        with_include = self.package(include=["rocm/**"])
        self.assertIn("rocm/bin/kernel.co", {f[0] for f in ec.read_manifest(with_include).files})
        self.assertNotIn("notes.txt", {f[0] for f in ec.read_manifest(with_include).files})
        every = self.out / "tree"
        every.mkdir()
        folder = package_engine.build(self.src, "cuda", "9.9", HERE, every, archs=("sm_86",), tree=True, build_id="x")
        self.assertIn("notes.txt", {f[0] for f in ec.read_manifest(folder).files})
        self.assertEqual(ec.read_manifest(folder).files[0][0], EXE)

    def test_symlinks_are_never_packaged(self):
        if WINDOWS:
            self.skipTest("POSIX symlinks")
        os.symlink("/etc/passwd", self.src / "link.so")
        folder = self.package()
        self.assertNotIn("link.so", {f[0] for f in ec.read_manifest(folder).files})

    def test_linux_programs_are_executable_in_the_tar_even_from_a_windows_packer(self):
        import feed_tool
        folder = self.package()
        elf = bytes([0x7f]) + b"ELF" + bytes(60)
        for name in ("strata-dsv41", "strata-vision"):                    # ELF programs without execute bits (copied over from Windows)
            (folder / name).write_bytes(elf)
            os.chmod(folder / name, 0o644)
        (folder / "libx.so").write_bytes(elf)
        if WINDOWS:                                                       # a Linux package packed on Windows: the ELF header decides
            self.assertEqual(feed_tool._mode(folder / "strata-dsv41", "strata-dsv41", "strata"), 0o755)
            self.assertEqual(feed_tool._mode(folder / "strata-vision", "strata-vision", "strata"), 0o755)

    def test_strata_dsv41_travels_beside_the_engine_and_can_be_required(self):
        dsv = "strata-dsv41.exe" if WINDOWS else "strata-dsv41"
        with self.assertRaises(SystemExit):
            self.package(require_dsv41=True)                                  # a package without it is refused when it is required
        self.assertNotIn(dsv, {f[0] for f in ec.read_manifest(self.package()).files})   # optional by default (Intel, Mac, Pascal ...)
        (self.src / dsv).write_bytes(b"dsv41-binary-" * 50)
        other = self.root / "again"
        other.mkdir()
        folder = package_engine.build(self.src, "cuda", "0.1.46", HERE, other, archs=("sm_86",), build_id="x", require_dsv41=True)
        self.assertIn(dsv, {f[0] for f in ec.read_manifest(folder).files})
        self.assertEqual(package_engine.main(["--verify", str(folder), "--require-dsv41"]), 0)
        (self.src / dsv).unlink()
        plain = self.root / "plain"
        plain.mkdir()
        bare = package_engine.build(self.src, "cuda", "0.1.47", HERE, plain, archs=("sm_86",), build_id="x")
        self.assertEqual(package_engine.main(["--verify", str(bare), "--require-dsv41"]), 1)

    def test_a_gpu_package_without_archs_is_refused_by_the_packager(self):
        with self.assertRaises(SystemExit):
            self.package(archs=())

    def test_schema_2_needs_build_id_and_archs(self):
        folder = self.package()
        path = folder / ec.MANIFEST
        original = json.loads(path.read_text())
        for key, why in (("build_id", "no build_id"), ("archs", "no archs")):
            data = dict(original)
            data.pop(key)
            path.write_text(json.dumps(data))
            with self.assertRaises(ec.ComponentProblem) as caught:
                ec.read_manifest(folder)
            self.assertIn(why, str(caught.exception))
        data = dict(original, min_driver={"nvidia": 580})
        path.write_text(json.dumps(data))
        with self.assertRaises(ec.ComponentProblem):
            ec.read_manifest(folder)
        data = dict(original, signature="trust me")
        path.write_text(json.dumps(data))
        with self.assertRaises(ec.ComponentProblem):
            ec.read_manifest(folder)

    def test_a_schema_1_package_from_s3_still_loads(self):
        folder = self.package()
        path = folder / ec.MANIFEST
        data = json.loads(path.read_text())
        for key in ("name", "build_id", "min_driver", "signature", "archs"):
            data.pop(key)
        data["schema_version"] = 1
        data["cuda_archs"] = ["sm_86"]
        path.write_text(json.dumps(data))
        component = ec.read_manifest(folder)
        self.assertEqual((component.archs, component.build_id), (("sm_86",), ""))

    def test_duplicate_file_entries_are_refused(self):
        folder = self.package()
        path = folder / ec.MANIFEST
        data = json.loads(path.read_text())
        data["files"].append(dict(data["files"][1]))
        path.write_text(json.dumps(data))
        with self.assertRaises(ec.ComponentProblem):
            ec.read_manifest(folder)

    def test_verify_cli_and_package_digest(self):
        folder = self.package()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(package_engine.main(["--verify", str(folder)]), 0)
        self.assertIn(package_engine.package_sha256(folder), out.getvalue())
        (folder / LIB).write_bytes(b"x" * (folder / LIB).stat().st_size)
        with contextlib.redirect_stdout(out):
            self.assertEqual(package_engine.main(["--verify", str(folder)]), 1)


class CollectLibs(unittest.TestCase):
    def test_kernel_data_is_filtered_by_arch_and_shared_tables_stay(self):
        want = "gfx1200,gfx1201"
        self.assertTrue(collect_libs.arch_wanted("Kernels.so-000-gfx1201.hsaco", want))
        self.assertTrue(collect_libs.arch_wanted("hipblasltTransform.hsaco", want))
        self.assertFalse(collect_libs.arch_wanted("TensileLibrary_x_gfx1100.co", want))
        self.assertTrue(collect_libs.arch_wanted("anything_gfx1100.co", ""))

    def test_elf_check_and_system_split(self):
        with tempfile.TemporaryDirectory() as d:
            lib, text = Path(d) / "libx.so", Path(d) / "gdb.py"
            lib.write_bytes(collect_libs.ELF_MAGIC + b"rest")
            text.write_text("print(1)")
            self.assertTrue(collect_libs.is_elf(lib))
            self.assertFalse(collect_libs.is_elf(text))
        self.assertTrue(collect_libs.is_system("/lib/x86_64-linux-gnu/libc.so.6"))
        self.assertFalse(collect_libs.is_system("/usr/local/cuda/lib64/libcudart.so.13"))
        self.assertTrue(collect_libs.is_driver("libcuda.so.1"))
        self.assertFalse(collect_libs.is_driver("libcublas.so.13"))


class Arch(unittest.TestCase):
    def test_machine_code_and_ptx(self):
        self.assertTrue(ec.arch_covers(["sm_86"], ["sm_86"]))
        self.assertFalse(ec.arch_covers(["sm_86"], ["sm_120"]))
        self.assertTrue(ec.arch_covers(["sm_75", "compute_89"], ["sm_120"]), "PTX of an older generation runs on a newer card")
        self.assertFalse(ec.arch_covers(["sm_75", "compute_89"], ["sm_86"]), "PTX never runs on an older card")
        self.assertTrue(ec.arch_covers(["sm_86"], []), "cannot tell: not a refusal")


class Start(unittest.TestCase):
    """What a start checks: every file's hash (not only the entrypoint), the driver, the signature policy."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.out = self.root / "engine"
        self.out.mkdir()
        self.folder = package_engine.build(make_source(self.root / "src"), "cuda", "0.1.45", HERE, self.out,
                                           archs=("sm_86",), min_driver={"nvidia": "580.65"}, build_id="b1")
        self.addCleanup(ec.register_signature_verifier, None)
        patch = mock.patch.object(signing, "BUILTIN_KEYS", self.root / "no-shipped-keys.json")   # a build with no shipped key; the real file: test_release_keys
        patch.start()
        self.addCleanup(patch.stop)

    def start(self, **kw):
        with mock.patch.object(ec, "cuda_archs_present", return_value=kw.pop("cards", ["sm_86"])), \
                mock.patch.object(ec, "nvidia_driver_version", return_value=kw.pop("driver", "590.00")):
            return ec.expand("component:strata-engine", {"backend": "cuda"}, root=self.root, **kw)

    def test_a_good_package_starts(self):
        self.assertEqual(Path(self.start()).resolve(), (self.folder / EXE).resolve())

    def test_a_same_size_change_in_a_library_refuses_the_start_with_the_file_named(self):
        lib = self.folder / LIB
        body = bytearray(lib.read_bytes())
        body[7] ^= 0xFF                                           # same size, same entrypoint: only the full hash sees it
        lib.write_bytes(bytes(body))
        with mock.patch.object(ec, "cuda_archs_present", return_value=["sm_86"]),                 mock.patch.object(ec, "nvidia_driver_version", return_value=None):
            self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="1").version, "0.1.45", "the quick check passes")
        with self.assertRaises(ec.EngineIntegrityError) as caught:
            self.start()
        self.assertIn(LIB, str(caught.exception))
        self.assertIn("SHA-256", str(caught.exception))
        self.assertIn("not started", str(caught.exception))
        self.assertEqual(caught.exception.exit_code, 5)
        self.assertIsInstance(caught.exception, ec.EngineMissing)

    def test_a_changed_manifest_hash_is_refused(self):
        path = self.folder / ec.MANIFEST
        data = json.loads(path.read_text())
        data["files"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(data))
        with self.assertRaises(ec.EngineIntegrityError):
            self.start()

    def test_two_builds_of_one_version_prefer_the_newer_driver_when_both_fit(self):
        other = make_source(self.root / "src3")
        old = package_engine.build(other, "cuda", "0.1.45", HERE, self.out, archs=("sm_60", "sm_86"),
                                   min_driver={"nvidia": "575.51"}, build_id="b-12", variant="cuda12")
        self.assertEqual(old.name, f"0.1.45-{HERE}-cuda-cuda12")
        self.assertEqual(ec.read_manifest(old).variant, "cuda12")
        self.assertEqual(Path(self.start(driver="610.0")).parent.resolve(), self.folder.resolve())    # 580.65 build wins
        self.assertEqual(Path(self.start(driver="576.0")).parent.resolve(), old.resolve())            # only the CUDA 12 one fits

    def test_a_cuda12_package_without_pascal_or_volta_code_is_refused(self):
        """#3503: the Linux cuda12 package shipped with 0.2.1 had sm_75/86/89 only; P100 containers restarted in a loop."""
        other = make_source(self.root / "src4")
        with self.assertRaises(SystemExit) as caught:
            package_engine.build(other, "cuda", "0.1.45", HERE, self.out, archs=("sm_75", "sm_86", "sm_89", "compute_89"),
                                 min_driver={"nvidia": "575.51"}, build_id="b-12", variant="cuda12")
        self.assertIn("sm_60", str(caught.exception))
        package_engine.build(other, "cuda", "0.1.45", HERE, self.out, archs=("sm_70", "sm_86"), build_id="b-13", variant="cuda12")

    def test_a_good_package_does_not_hide_a_tampered_one(self):
        newer = package_engine.build(make_source(self.root / "src2"), "cuda", "0.1.46", HERE, self.out,
                                     archs=("sm_86",), build_id="b2")
        (newer / LIB).write_bytes(b"y" * (newer / LIB).stat().st_size)
        self.assertEqual(ec.read_manifest(Path(self.start()).parent).version, "0.1.45")

    def test_other_pc_reasons_are_not_integrity_errors(self):
        with self.assertRaises(ec.EngineMissing) as caught:
            self.start(cards=["sm_120"])
        self.assertNotIsInstance(caught.exception, ec.EngineIntegrityError)
        self.assertIn("sm_86", str(caught.exception))

    def test_an_old_driver_is_refused_with_both_versions(self):
        with self.assertRaises(ec.EngineMissing) as caught:
            self.start(driver="535.10")
        self.assertIn("580.65", str(caught.exception))
        self.assertIn("535.10", str(caught.exception))
        self.assertTrue(self.start(driver=None), "an unreadable driver is not a refusal")

    def test_signature_policy_and_hook(self):
        self.assertEqual(ec.verify_signature(ec.read_manifest(self.folder), {}), "unsigned")
        with self.assertRaises(ec.IntegrityProblem):
            ec.verify_signature(ec.read_manifest(self.folder), {ec.ENV_REQUIRE_SIGNED: "1"})
        with mock.patch.dict(os.environ, {ec.ENV_REQUIRE_SIGNED: "1"}):
            with self.assertRaises(ec.EngineIntegrityError):
                self.start()
        # sign it: a stand-in "signature" over the canonical payload
        path = self.folder / ec.MANIFEST
        data = json.loads(path.read_text())
        key = b"test-key"
        data["signature"] = {"alg": "test-hmac", "key_id": "k1",
                             "value": hashlib.sha256(key + ec.signing_payload(data)).hexdigest()}
        path.write_text(json.dumps(data))
        component = ec.read_manifest(self.folder)
        self.assertEqual(ec.verify_signature(component, {}), "unverified", "signed but nobody can check it")
        with self.assertRaises(ec.IntegrityProblem):
            ec.verify_signature(component, {ec.ENV_REQUIRE_SIGNED: "1"})

        def verifier(payload, signature, comp):
            if signature["value"] != hashlib.sha256(key + payload).hexdigest():
                raise ValueError("wrong")
        ec.register_signature_verifier(verifier)
        self.assertEqual(ec.verify_signature(component, {ec.ENV_REQUIRE_SIGNED: "1"}), "verified")
        self.assertTrue(self.start())
        data["version"] = "0.1.46-evil"                      # any manifest change breaks the signature
        path.write_text(json.dumps(data))
        with self.assertRaises(ec.EngineIntegrityError) as caught:
            self.start()
        self.assertIn("signature", str(caught.exception))

    def test_the_payload_is_canonical_and_excludes_the_signature(self):
        a = ec.signing_payload({"b": 1, "a": [1, 2], "signature": {"alg": "x", "value": "y"}})
        b = ec.signing_payload({"a": [1, 2], "b": 1, "signature": None})
        self.assertEqual(a, b)
        self.assertEqual(a, b'{"a":[1,2],"b":1}')


if __name__ == "__main__":
    unittest.main()
