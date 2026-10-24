"""The Docker files stay consistent: shell scripts parse, Dockerfiles use only what the legacy builder knows, the compose examples carry what
docs/DOCKER.md says they carry, and every setting the document lists is one the entry point reads."""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
SH = shutil.which("sh")
BASH = shutil.which("bash")      # the full path: on Windows a bare "bash" can start the WSL launcher in System32 instead of Git bash
CYGPATH = shutil.which("cygpath")


def sh_path(p) -> str:
    """The path as the shell reads it. Git bash on Windows wants /c/Users/..., not C:\\Users\\... (tar even takes `C:` for a host)."""
    if CYGPATH is None:
        return str(p)
    return subprocess.run([CYGPATH, "-u", str(p)], capture_output=True, text=True, check=True).stdout.strip()


class Scripts(unittest.TestCase):
    @unittest.skipIf(SH is None, "needs sh")
    def test_shell_scripts_parse(self):
        for name in ("entrypoint.sh", "strata-wrapper.sh", "healthcheck.sh", "build.sh", "publish.sh"):
            r = subprocess.run([SH, "-n", sh_path(DOCKER / name)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")

    @unittest.skipIf(BASH is None, "needs bash")
    def test_smoke_test_parses_with_bash(self):           # it is a bash script (process substitution), not POSIX sh
        r = subprocess.run([BASH, "-n", sh_path(DOCKER / "smoke-test.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipIf(SH is None or shutil.which("tar") is None, "needs sh and tar")
    def test_build_script_exits_zero_after_assembling_a_context(self):
        """--ctx-only stops after the context is made (no docker needed). The EXIT trap once returned 1 there, so a good run reported failure."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "pkg"
            pkg.mkdir()
            (pkg / "engine-component.json").write_text('{"entrypoint": "strata"}', encoding="utf-8")
            (pkg / "strata").write_text("#!/bin/sh\n", encoding="utf-8")
            (pkg / "strata").chmod(0o755)
            ctx = Path(tmp) / "ctx"
            r = subprocess.run([SH, sh_path(DOCKER / "build.sh"), "--flavor", "cuda", "--engine", sh_path(pkg), "--ctx", sh_path(ctx), "--ctx-only"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue((ctx / "Dockerfile").is_file())
            self.assertTrue((ctx / "terminal-src" / "docs" / "TERMINAL_LAUNCH.md").is_file())
            self.assertTrue((ctx / "engine" / "pkg" / "engine-component.json").is_file())

    def test_scripts_use_unix_line_ends(self):
        for name in ("entrypoint.sh", "strata-wrapper.sh", "healthcheck.sh", "build.sh", "publish.sh"):
            self.assertNotIn(b"\r", (DOCKER / name).read_bytes(), name)       # build.sh strips CR for a CRLF checkout, the files themselves stay LF


class Dockerfiles(unittest.TestCase):
    FILES = (DOCKER / "Dockerfile", ROOT / "Dockerfile")

    def test_only_legacy_builder_syntax(self):
        for f in self.FILES:
            text = f.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"<<\s*-?['\"]?\w+", f"{f.name}: a heredoc needs BuildKit")
            self.assertNotIn("--mount=", text, f.name)
            self.assertNotRegex(text, r"COPY\s+--chmod", f.name)

    def test_runtime_contract(self):
        for f in self.FILES:
            text = f.read_text(encoding="utf-8")
            self.assertIn("HEALTHCHECK", text, f.name)
            self.assertIn("STOPSIGNAL SIGTERM", text, f.name)
            self.assertIn('ENTRYPOINT ["/usr/bin/tini", "--", "/opt/strata/entrypoint.sh"]', text, f.name)
            self.assertIn("VOLUME [\"/data\"]", text, f.name)
            self.assertIn("STRATA_COMPONENTS_DIR=/opt/strata/components", text, f.name)

    def test_component_image_checks_the_package_before_use(self):
        text = (DOCKER / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("package_engine.py", text)
        self.assertIn("binary_deps.py", text)
        self.assertIn("ldd", text)                       # the finished image resolves every library of the package

    def test_binary_deps_and_package_engine_exist_where_the_dockerfile_copies_them(self):
        for name in ("package_engine.py", "binary_deps.py", "package_terminal.py"):
            self.assertTrue((ROOT / "packaging" / name).is_file(), name)


class Compose(unittest.TestCase):
    def load(self, name: str) -> dict:
        data = yaml.safe_load((DOCKER / "compose" / name).read_text(encoding="utf-8"))
        self.assertIn("strata", data["services"], name)
        return data

    def test_common_settings(self):
        for name in ("docker-compose.nvidia.yml", "docker-compose.amd.yml", "docker-compose.intel.yml"):
            svc = self.load(name)["services"]["strata"]
            self.assertIn("strata-data:/data", svc["volumes"], name)
            self.assertEqual(svc["ulimits"]["memlock"], {"soft": -1, "hard": -1}, name)
            self.assertEqual(svc["shm_size"], "2gb", name)
            self.assertEqual(svc["stop_grace_period"], "2m", name)
            self.assertTrue(svc["ports"][0].startswith("${STRATA_BIND:-127.0.0.1}:"), f"{name}: the port must default to this computer only")
            for var in ("MODEL", "MODEL_PATH", "CONTEXT", "PARALLEL", "VISION", "TUNE", "API_KEY"):
                self.assertIn(var, svc["environment"], f"{name}: {var}")
            self.assertIn("strata-data", self.load(name)["volumes"], name)

    def test_gpu_access_per_vendor(self):
        nv = self.load("docker-compose.nvidia.yml")["services"]["strata"]
        dev = nv["deploy"]["resources"]["reservations"]["devices"][0]
        self.assertEqual(dev["driver"], "nvidia")
        self.assertIn("gpu", dev["capabilities"])
        amd = self.load("docker-compose.amd.yml")["services"]["strata"]
        self.assertIn("/dev/kfd:/dev/kfd", amd["devices"])
        self.assertIn("/dev/dri:/dev/dri", amd["devices"])
        self.assertIn("video", amd["group_add"])
        self.assertTrue(any("render" in str(g).lower() or "RENDER" in str(g) for g in amd["group_add"]))
        intel = self.load("docker-compose.intel.yml")["services"]["strata"]
        self.assertIn("/dev/dri:/dev/dri", intel["devices"])
        self.assertNotIn("/dev/kfd:/dev/kfd", intel["devices"])


class Document(unittest.TestCase):
    def test_every_documented_setting_is_read_by_the_entry_point(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        section = doc.split("## Settings", 1)[1].split("###", 1)[0]
        entry = (DOCKER / "entrypoint.sh").read_text(encoding="utf-8")
        names: set[str] = set()
        for line in section.splitlines():
            if line.startswith("| `"):
                names.update(re.findall(r"`([A-Z][A-Z_]+)`", line.split("|")[1]))
        self.assertTrue({"MODEL", "CONTEXT", "API_KEY", "TUNE", "VISION", "MODEL_PATH", "STRATA_SET"} <= names, names)
        for name in sorted(names):
            self.assertIn(name, entry, f"docs/DOCKER.md lists {name} but docker/entrypoint.sh does not read it")

    def test_tags_in_the_document_match_the_compose_defaults(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        for tag in ("0.2.3-cuda13", "0.2.3-cuda12", "0.2.3-rocm-gfx120x", "0.2.3-rocm-gfx110x", "0.2.3-rocm-gfx1151", "0.2.3-sycl"):
            self.assertIn(tag, doc)
        for name, tag in (("docker-compose.nvidia.yml", "ghcr.io/niko1221/strata:0.2.3-cuda13"), ("docker-compose.amd.yml", "ghcr.io/niko1221/strata:0.2.3-rocm-gfx120x"),
                          ("docker-compose.intel.yml", "ghcr.io/niko1221/strata:0.2.3-sycl")):
            self.assertIn(tag, (DOCKER / "compose" / name).read_text(encoding="utf-8"))

    def test_the_user_doc_says_images_are_published_from_0_2_1(self):
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        self.assertNotIn("TODO(owner)", doc)
        self.assertNotIn("not available yet", doc)
        self.assertIn("published from version 0.2.1 on", doc)
        self.assertIn("Publishing the images (for maintainers)", doc)


class Publishing(unittest.TestCase):
    """The scripts and the workflow that publish the images (docker/publish.sh, docker/fetch-engines.py, .github/workflows/docker-images.yml)."""
    WORKFLOW = ROOT / ".github" / "workflows" / "docker-images.yml"

    @unittest.skipIf(SH is None, "needs sh")
    def test_publish_script_parses_and_defaults_to_not_pushing(self):
        path = DOCKER / "publish.sh"
        self.assertEqual(subprocess.run([SH, "-n", sh_path(path)], capture_output=True, text=True).returncode, 0)
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("\r", text)
        self.assertIn('push=""', text)                      # no --push, no docker push
        pushes = re.findall(r'^[^#\n]*docker push "', text, re.M)
        self.assertEqual(len(pushes), 2, pushes)             # the version tag and latest-<flavor>, both inside `if [ -n "$push" ]`

    def test_every_flavor_of_the_document_is_one_the_fetch_script_knows(self):
        src = (DOCKER / "fetch-engines.py").read_text(encoding="utf-8")
        flavors = re.findall(r'^    "([a-z0-9-]+)": \("', src, re.M)
        doc = (ROOT / "docs" / "DOCKER.md").read_text(encoding="utf-8")
        self.assertTrue({"cuda13", "cuda12", "rocm-gfx120x", "rocm-gfx110x", "rocm-gfx1151", "sycl"} <= set(flavors), flavors)
        for flavor in flavors:
            self.assertIn(f"0.2.3-{flavor}", doc)

    def test_build_script_copies_everything_the_terminal_package_needs(self):
        """docker/build.sh lists the Terminal's files by hand; the list must cover package_terminal.py's payload (it once missed docs/TERMINAL_LAUNCH.md)."""
        build = (DOCKER / "build.sh").read_text(encoding="utf-8")
        listed = re.search(r'^paths="([^"]+)"', build, re.M).group(1).split()
        pt = ast.parse((ROOT / "packaging" / "package_terminal.py").read_text(encoding="utf-8"))
        payload = None
        for node in pt.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "PAYLOAD" for t in node.targets):
                payload = ast.literal_eval(node.value)
        self.assertTrue(payload)
        for item in payload:
            self.assertTrue(any(item == p or item.startswith(p.rstrip("/") + "/") for p in listed), f"build.sh does not copy {item}")

    def test_workflow_is_manual_and_pushes_only_on_request(self):
        data = yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        triggers = data.get("on", data.get(True))            # YAML 1.1 reads a bare `on` as True
        self.assertEqual(set(triggers), {"workflow_dispatch"})
        inputs = triggers["workflow_dispatch"]["inputs"]
        self.assertIn("release_tag", inputs)
        self.assertFalse(inputs["push"]["default"])
        self.assertEqual(data["permissions"], {"contents": "read"})
        self.assertEqual(data["jobs"]["image"]["permissions"], {"contents": "read", "packages": "write"})
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ghcr.io/niko1221/strata", text)
        self.assertIn("docker/publish.sh", text)
        self.assertNotIn("${{ secrets.GITHUB_TOKEN }}\" |", text)   # the token goes through an env variable, not into the command line


if __name__ == "__main__":
    unittest.main()
