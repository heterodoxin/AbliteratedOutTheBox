"""install.ps1 / install.sh: an ARM computer is refused with the reason and the way out; the Windows script survives the usual environment trouble."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PS1 = ROOT / "packaging" / "install.ps1"
SH = ROOT / "packaging" / "install.sh"


class Text(unittest.TestCase):
    def setUp(self):
        self.ps1 = PS1.read_text(encoding="utf-8")
        self.sh = SH.read_text(encoding="utf-8")

    def test_arm_is_refused_with_a_reason_and_a_next_step(self):
        self.assertIn("Windows on ARM PC", self.ps1)
        self.assertIn("no engine for ARM", self.ps1)
        self.assertNotIn("has no Strata Terminal yet", self.ps1)
        self.assertIn("ARM processor", self.sh)
        self.assertIn("Apple silicon", self.sh)

    def test_tls12_proxy_sign_in_and_the_real_error_text(self):
        self.assertIn("Tls12", self.ps1)
        self.assertIn("DefaultNetworkCredentials", self.ps1)
        self.assertIn("$($_.Exception.Message)", self.ps1)

    def test_long_paths_junction_and_running_version(self):
        self.assertIn("tar.exe", self.ps1)
        self.assertIn("STRATA_DATA_DIR to a short folder", self.ps1)
        self.assertIn("$LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $current", self.ps1)
        self.assertIn("a program is using it", self.ps1)
        self.assertIn("Windows did not say where your user folder is", self.ps1)
        self.assertNotIn("is the disk full?", self.ps1)


@unittest.skipUnless(shutil.which("powershell"), "no PowerShell")
class Run(unittest.TestCase):
    def test_native_arm_stops_before_any_download(self):
        env = dict(os.environ, PROCESSOR_ARCHITECTURE="ARM64", STRATA_FEED="http://127.0.0.1:9")
        env.pop("PROCESSOR_ARCHITEW6432", None)
        done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1)],
                              capture_output=True, text=True, env=env, timeout=120)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("Windows on ARM PC", done.stderr + done.stdout)
        self.assertNotIn("Downloading", done.stdout)


if __name__ == "__main__":
    unittest.main()
