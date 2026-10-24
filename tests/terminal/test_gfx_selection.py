"""#3456 / #3486: an RX 9070 on Windows has no hipInfo, so the gfx list is empty and every HIP build "fits". The newest version (the rdna3
build, 0.1.46.1) then won over the rx9070 one (0.1.46). Selection must follow the GPU name or the installed variant, never the version."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_control import engine_component as ec, feed as feedmod, updater

WIN = "windows-x86_64"


def pkg(version, variant, archs):
    return feedmod._engine_package({
        "version": version, "platform": WIN, "backend": "hip", "variant": variant, "archs": list(archs), "min_driver": {},
        "min_runtime": None, "build_id": f"b-{version}-{variant}", "format": "zip", "url": f"e-{version}-{variant}.zip", "size": 10,
        "sha256": "a" * 64, "manifest_sha256": "b" * 64, "installed_bytes": 20, "notes": ""})


FEED = [pkg("0.1.46", "rx9070", ("gfx1200", "gfx1201")), pkg("0.1.46.1", "rdna3", ("gfx1100", "gfx1101", "gfx1102", "gfx1151"))]


def hw(names=(), gfx=(), installed="", wanted=""):
    return {"platform": WIN, "backend": "hip", "runtime": "0.2.1.3", "nvidia_driver": None, "cuda_archs": [], "gfx": list(gfx),
            "gpu_names": list(names), "installed_variant": installed, "variant_wanted": wanted}


class Selection(unittest.TestCase):
    def pick(self, hardware):
        best, reasons = feedmod.select_engine(FEED, hardware)
        return (best.variant if best else None), reasons

    def test_rx9070_with_empty_gfx_gets_rx9070(self):
        for name in ("AMD Radeon RX 9070 XT", "AMD Radeon RX 9070", "AMD Radeon RX 9060 XT", "AMD Radeon AI PRO R9700"):
            self.assertEqual(self.pick(hw([name]))[0], "rx9070", name)

    def test_strix_halo_and_rx7900_with_empty_gfx_get_rdna3(self):
        self.assertEqual(self.pick(hw(["AMD Radeon(TM) 8060S Graphics"]))[0], "rdna3")
        self.assertEqual(self.pick(hw(["AMD Radeon RX 7900 XTX"]))[0], "rdna3")
        self.assertEqual(self.pick(hw(["AMD Radeon RX 7600"]))[0], "rdna3")

    def test_unknown_amd_keeps_the_installed_variant(self):
        got, why = self.pick(hw(["AMD Radeon Mystery"], installed="rx9070"))
        self.assertEqual(got, "rx9070")
        self.assertTrue(any("kept" in r for r in why))
        self.assertEqual(self.pick(hw([], installed="rdna3"))[0], "rdna3")

    def test_unknown_amd_and_nothing_installed_chooses_nothing_instead_of_the_newest(self):
        got, why = self.pick(hw([]))
        self.assertIsNone(got)
        self.assertTrue(any("cannot tell" in r for r in why))

    def test_the_name_beats_a_stale_installed_variant_when_the_card_changed(self):
        self.assertEqual(self.pick(hw(["AMD Radeon RX 9070 XT"], installed="rdna3"))[0], "rx9070")

    def test_a_known_gfx_is_unchanged(self):
        self.assertEqual(self.pick(hw(gfx=["gfx1201"]))[0], "rx9070")
        self.assertEqual(self.pick(hw(gfx=["gfx1151"]))[0], "rdna3")

    def test_the_variant_can_be_asked_for(self):
        self.assertEqual(self.pick(hw([], wanted="rx9070"))[0], "rx9070")


class Names(unittest.TestCase):
    def test_names(self):
        table = {"AMD Radeon RX 9070 XT": "gfx1201", "AMD Radeon RX 9060 XT": "gfx1200", "AMD Radeon AI PRO R9700": "gfx1201",
                 "AMD Radeon RX 7900 XTX": "gfx1100", "AMD Radeon RX 7800 XT": "gfx1101", "AMD Radeon RX 7700 XT": "gfx1101",
                 "AMD Radeon RX 7600 XT": "gfx1102", "AMD Radeon(TM) 8060S Graphics": "gfx1151", "AMD Radeon 780M Graphics": "gfx1103",
                 "AMD Radeon 890M Graphics": "gfx1150", "AMD Radeon RX 6800 XT": "gfx1030", "NVIDIA GeForce RTX 4090": "", "": ""}
        for name, gfx in table.items():
            self.assertEqual(ec.gfx_from_name(name), gfx, name)

    def test_a_card_with_an_unlisted_pci_id_is_identified_by_its_name(self):
        cards = ec.windows_amd_cards([{"vendor": "1002", "device": 0x7551 + 0x100, "name": "AMD Radeon RX 9070 XT", "vram_gb": 0.0}])
        self.assertEqual([c["arch"] for c in cards], ["gfx1201"])
        self.assertEqual(ec.windows_amd_cards([{"vendor": "1002", "device": 0x9999, "name": "AMD Radeon(TM) Graphics", "vram_gb": 0.5}]), [])
        self.assertEqual(ec.windows_amd_cards([{"vendor": "1002", "device": 0x9999, "name": "AMD Radeon 890M Graphics", "vram_gb": 4.0}]), [])


class Detection(unittest.TestCase):
    def test_windows_without_hipinfo_reads_the_gfx_from_the_adapters(self):
        adapters = [{"vendor": "1002", "device": 0x7550, "name": "AMD Radeon RX 9070 XT", "vram_gb": 16.0}]
        with mock.patch.object(updater.os, "name", "nt"), mock.patch.object(updater.shutil, "which", return_value=None), \
                mock.patch.object(ec, "windows_display_adapters", return_value=adapters):
            self.assertEqual(updater._gfx_names(), ["gfx1201"])
            self.assertEqual(updater._amd_card_names(), ["AMD Radeon RX 9070 XT"])

    def test_the_feed_style_backend_name_is_accepted(self):
        with mock.patch.object(updater, "_gfx_names", return_value=[]), mock.patch.object(updater, "_amd_card_names", return_value=[]):
            got = updater.detect_hardware({ec.ENV_BACKEND: "hip-rx9070"})
        self.assertEqual((got["backend"], got["variant_wanted"]), ("hip", "rx9070"))


if __name__ == "__main__":
    unittest.main()
