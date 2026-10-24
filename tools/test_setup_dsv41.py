"""Tests for DeepSeek-V4.1-Flash in setup.py (MULTI_FILE_FAMILIES: nine pinned shards, its own engine strata-dsv41, several cards) and
for strata_control/dsv41_fit.py (where it fits, from the real numbers).  Mocked engine, downloads and processes - nothing is built,
fetched or started.

    python -m unittest tools.test_setup_dsv41
"""
from __future__ import annotations

import argparse
import contextlib
import os
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from strata_control import dsv41_fit  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")
REPO = "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
R9700 = {"name": "AMD Radeon AI PRO R9700", "vram_gb": 32.0, "vendor": "amd"}
FOUR = [{**R9700, "index": i} for i in range(4)]


def args(**kw):
    base = dict(family="deepseek41", model=None, check=False, yes=True, vision=None, models_dir=None, gguf_dir=None, context=None,
                rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto", vram_reserve_mib=None,
                host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None, gpus=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Entries(unittest.TestCase):
    def test_pinned_and_hashed(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        self.assertEqual(setup.HF_REVISIONS[REPO], "f924b94f19c1fea02c06a65c8475b0a45e2b88b1")
        self.assertRegex(fam["hf"], SHA)
        self.assertTrue(fam["hf"].endswith(f"{REPO}/resolve/f924b94f19c1fea02c06a65c8475b0a45e2b88b1/UD-Q8_K_XL/"))
        self.assertIs(setup.REPO_PINS[REPO], setup.DSV41_FILES)
        shards = fam["models"]["UD-Q8_K_XL"]["shards"]
        self.assertEqual(len(shards), 9)
        self.assertEqual(sum(setup.DSV41_FILES[n][0] for n in shards), 508_376_763_197)       # the nine shards: the pinned total
        self.assertEqual(setup.DSV41_FILES[shards[7]][0], dsv41_fit.ENGRAM_BYTES)             # shard 8 is the n-gram table
        for n, (size, sha) in setup.DSV41_FILES.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", n)
            self.assertGreater(size, 5e6)
        self.assertEqual(setup.DSV41_FILES["mmproj-BF16.gguf"][0], 972_208_288)

    def test_not_a_pack_or_single_file_family(self):
        self.assertNotIn("deepseek41", setup.FAMILIES)
        self.assertNotIn("deepseek41", setup.OWN_PATH_FAMILIES)
        self.assertNotIn("UD-Q8_K_XL", setup.MODELS)

    def test_config_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-deepseek-v4.1-flash-ud-q8_k_xl.json"
            p.write_text(json.dumps({"args": ["--pack", "x", "--max-context", "8192", "--tp", "2"], "port": 8080, "gpu": [0, 1, 2, 3]}))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"], ch["context"], ch["gpu"]), ("deepseek41", "UD-Q8_K_XL", 8192, [0, 1, 2, 3]))


class Fit(unittest.TestCase):
    def test_the_numbers_of_the_model(self):
        self.assertAlmostEqual(dsv41_fit.WEIGHTS_GB, 305.6, delta=0.1)
        self.assertAlmostEqual(dsv41_fit.EXPERTS_GB, 288.8, delta=0.1)
        self.assertAlmostEqual(dsv41_fit.ENGRAM_GB, 202.8, delta=0.1)
        self.assertAlmostEqual(dsv41_fit.DOWNLOAD_GB, 508.4, delta=0.1)
        self.assertEqual(dsv41_fit.EXPERTS, 15360)

    def test_four_amd_cards_run_the_ista_sets_with_a_plain_layer_split(self):
        # r9700, 2026-10-23: 8K prompt 370 tokens a second with --tp 1 against 196 with the engine's default (2)
        for model in ("IST-DAS-1.75bpw", "IST-DAS-2.00bpw", "IST-DAS-2.25bpw"):
            p = dsv41_fit.plan(FOUR, 251.0, 900.0, model=model)
            self.assertEqual((p["tp"], p["args"]), (1, ["--tp", "1"]), model)
        # two and three cards keep pairs; other makes keep pairs; one card needs no flag; the Unsloth set is not changed
        self.assertEqual(dsv41_fit.plan(FOUR[:2], 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], ["--tp", "2"])
        self.assertEqual(dsv41_fit.plan(FOUR[:3], 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], ["--tp", "2"])
        nv = [{"name": "NVIDIA GeForce RTX 3090", "vram_gb": 24.0, "vendor": "nvidia", "index": i} for i in range(4)]
        self.assertEqual(dsv41_fit.plan(nv, 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], ["--tp", "2"])
        self.assertEqual(dsv41_fit.plan(FOUR[:1], 251.0, 900.0, model="IST-DAS-2.00bpw")["args"], [])
        self.assertEqual(dsv41_fit.plan(FOUR, 251.0, 900.0)["args"], ["--tp", "2"])

    def test_four_r9700_match_the_measured_run(self):
        # r9700, 2026-10-20: 1,558 + 1,559 + 1,575 + 1,505 = 6,197 experts in graphics memory, 9,163 (172.3 GB) on the CPU lane
        p = dsv41_fit.plan(FOUR, 251.0, 900.0)
        self.assertEqual(p["verdict"], "fits")
        self.assertAlmostEqual(p["experts_in_vram"], 6197, delta=100)
        self.assertAlmostEqual(p["cpu_experts_gb"], 172.3, delta=2.5)
        self.assertEqual([c["layers"] for c in p["cards"]], [[0, 9], [10, 19], [20, 29], [30, 39]])
        self.assertEqual(p["args"], ["--tp", "2"])
        self.assertTrue(p["disk_ok"])

    def test_one_card_and_little_ram_are_warnings(self):
        p = dsv41_fit.plan([{**R9700, "index": 0}], 64.0, 100.0)
        self.assertEqual(p["verdict"], "spills")
        self.assertEqual(p["args"], [])
        self.assertGreater(p["spill_gb"], 200)
        text = " ".join(p["warnings"])
        self.assertIn("read from the SSD", text)
        self.assertIn("free disk space", text)
        self.assertIn("still allowed", text)
        self.assertFalse(p["disk_ok"])

    def test_a_card_too_small_for_the_dense_weights_is_said(self):
        p = dsv41_fit.plan([{"index": 0, "name": "RTX 3060", "vram_gb": 12.0}], 128.0)
        self.assertTrue(any("dense weights" in w for w in p["warnings"]))

    def test_no_card(self):
        p = dsv41_fit.plan([], 128.0)
        self.assertEqual(p["verdict"], "no_card")
        self.assertTrue(any("No graphics card" in w for w in p["warnings"]))

    def test_cards_of_two_sizes_share_the_layers_in_proportion(self):
        cards = [{**R9700, "index": 0}, {"name": "RTX A4000", "vram_gb": 16.0, "index": 1}]
        p = dsv41_fit.plan(cards, 256.0)
        a, b = p["cards"]
        self.assertGreater(a["layers"][1] - a["layers"][0], b["layers"][1] - b["layers"][0])
        self.assertEqual(b["layers"][1], 39)
        self.assertGreater(a["experts"], b["experts"])

    def test_split_layers_covers_every_layer(self):
        for w in ([1, 1, 1, 1], [32, 16], [8, 8, 8], [32, 32, 16, 12, 12]):
            r = dsv41_fit.split_layers(w)
            self.assertEqual(r[0][0], 0)
            self.assertEqual(r[-1][1], 39)
            self.assertTrue(all(a[1] + 1 == b[0] for a, b in zip(r, r[1:])))

    def test_options_like_gpu_fit(self):
        o = dsv41_fit.options(FOUR, 251.0, "amd")
        self.assertEqual(o["recommended"], "all")
        self.assertEqual([x["key"] for x in o["options"]], ["all", "single"])
        self.assertEqual(o["options"][0]["cards"], [0, 1, 2, 3])
        self.assertTrue(o["options"][0]["recommended"])


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.cfg = {}
        self.exe = self.root / "engine" / ("strata-dsv41.exe" if setup.WIN else "strata-dsv41")
        eng = self.root / "engine"

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            self.calls.append(("engine", vision))
            eng.mkdir(parents=True, exist_ok=True)
            return self.root / "llama", eng, {"version": "0.2.2"}, ["/libs"], gpu, vision

        def write_cfg(path, cfg, source=None, explicit=None):
            self.cfg = cfg
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha, own=False: self.calls.append(("sha", s.name, own))),
                   mock.patch.object(setup, "free_gb", lambda p: 900.0),
                   mock.patch.object(setup, "pip_install", lambda *a, **k: None),
                   mock.patch.object(setup, "engine_step", engine_step),
                   mock.patch.object(setup, "run", lambda cmd, **k: self.calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder", lambda d, *a, **k: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "ROOT", self.root),
                   mock.patch.object(setup, "home", lambda: self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": ""}).start()
        self.addCleanup(mock.patch.stopall)

    def install(self, ram=251.0, cards=None, engine=True, **kw):
        if engine:
            self.exe.parent.mkdir(parents=True, exist_ok=True)
            self.exe.write_text("x")
        cards = cards or FOUR
        kw.setdefault("model", "UD-Q8_K_XL")
        gpu = {"index": 0, "count": len(cards), "vram_gb": 32.0, "name": "R9700", "cards": cards}
        return run(setup.install_dsv41, args(models_dir=str(self.root / "models"), **kw), "deepseek41", gpu, True, [], 13, [], 8080, ram,
                   None, cards)

    def test_installs_the_nine_pinned_shards_and_writes_the_config(self):
        rc, out = self.install()
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        folder = self.root / "models" / "deepseek-v4.1-flash-UD-Q8_K_XL"
        self.assertEqual([c[2] for c in get], [folder / setup.DSV41_SHARD.format(i=i) for i in range(1, 10)])
        self.assertEqual(get[1][1], setup.hf(REPO) + "UD-Q8_K_XL/" + setup.DSV41_SHARD.format(i=2))
        self.assertEqual(sum(1 for c in self.calls if c[0] == "sha"), 9)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn("pack_gguf.py", " ".join(map(str, pack)))
        self.assertEqual(self.cfg["arch"], "deepseek41")
        self.assertEqual(self.cfg["exe"], str(self.exe))
        self.assertEqual(self.cfg["model_name"], "deepseek-v4.1-flash")
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--max-context") + 1], "8192")
        self.assertEqual(a[a.index("--eos") + 1], "1")
        self.assertEqual(a[a.index("--tp") + 1], "2")
        self.assertTrue(Path(a[a.index("--pack") + 1]).match("*packs/deepseek-v4.1-flash-ud-q8_k_xl"))
        self.assertEqual(self.cfg["gpu"], [0, 1, 2, 3])
        self.assertEqual(self.cfg["engine_ready_s"], 3600)
        self.assertEqual(self.cfg["backend"], "hip")
        self.assertTrue(self.cfg["tokenizer"].endswith("tokenizer"))
        self.assertNotIn("--vision", a)
        self.assertNotIn("--mtp", a)
        self.assertIn("4 cards, one block of layers on each", out)
        self.assertIn("n-gram table", out)

    def test_no_engine_stops_before_the_download(self):
        rc, out = self.install(engine=False)
        self.assertEqual(rc, 1, out)
        self.assertIn("DeepSeek's engine is not available for this computer yet", out)
        self.assertIn("RTX 30", out)
        self.assertNotIn("cmake", out)
        self.assertNotIn("STRATA_DSV41_EXE", out)
        self.assertIn("Nothing has been downloaded", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_own_files_are_checked_not_downloaded(self):
        own = self.root / "own"
        own.mkdir()
        for i in range(1, 10):
            (own / setup.DSV41_SHARD.format(i=i)).write_bytes(b"GGUF")
        rc, out = self.install(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])
        self.assertTrue(all(c[2] for c in self.calls if c[0] == "sha"))        # own=True: setup never deletes the person's file
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn(str(own), pack)

    def test_a_missing_own_shard_is_said(self):
        own = self.root / "own"
        own.mkdir()
        for i in (1, 2, 3):
            (own / setup.DSV41_SHARD.format(i=i)).write_bytes(b"GGUF")
        rc, out = self.install(gguf_dir=str(own))
        self.assertEqual(rc, 1, out)
        self.assertIn("00004-of-00009.gguf is missing", out)

    def test_the_repository_layout_is_found(self):
        own = self.root / "repo"
        (own / "UD-Q8_K_XL").mkdir(parents=True)
        for i in range(1, 10):
            (own / "UD-Q8_K_XL" / setup.DSV41_SHARD.format(i=i)).write_bytes(b"GGUF")
        rc, out = self.install(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn(str(own / "UD-Q8_K_XL"), pack)

    def test_little_ram_is_asked_and_kept_with_an_explicit_model(self):
        rc, out = self.install(ram=64.0, cards=FOUR[:1])
        self.assertEqual(rc, 0, out)
        self.assertIn("read from the SSD", out)
        self.assertIn("installing UD-Q8_K_XL with 64 GB of RAM, as you chose", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--max-context") + 1], "8192")
        self.assertNotIn("--tp", self.cfg["args"])

    def test_little_ram_with_yes_alone_stops_where_the_rule_says(self):
        rc, out = self.install(ram=24.0, cards=FOUR[:1], model=None)        # the version was not chosen on the command line
        self.assertEqual(rc, 1, out)
        self.assertIn("more RAM or cards fix it", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_a_disk_that_cannot_hold_the_files_is_the_one_hard_stop(self):
        with mock.patch.object(setup, "free_gb", lambda p: 100.0):
            rc, out = self.install()
        self.assertEqual(rc, 1, out)
        self.assertIn("not enough free disk space", out)
        self.assertIn("an NVMe drive is best", out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])

    def test_pictures_and_flash_next_settings_are_said(self):
        rc, out = self.install(vision="yes", kv="int8", context=131072)
        self.assertEqual(rc, 0, out)
        self.assertIn("cannot read pictures in this Strata version", out)
        self.assertIn("Flash-Next settings", out)
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--max-context") + 1], "131072")
        self.assertIn("checked at 4K and 8K", out)


class Catalog(unittest.TestCase):
    def test_the_mcp_catalog_lists_the_size(self):
        import strata_mcp as M
        s = M.Strata(ROOT)
        fam = next(f for f in s.model_table({"ram_gb": 251.0, "gpus": [{**c, "usable": True} for c in FOUR], "engine_vendor": "amd"})
                   if f["family"] == "deepseek41")
        self.assertEqual([x["model"] for x in fam["sizes"]], ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "UD-Q8_K_XL"])
        small_sets = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual((small_sets["IST-DAS-2.00bpw"]["download_gb"], small_sets["IST-DAS-2.25bpw"]["download_gb"]), (256.8, 273.8))
        self.assertTrue(small_sets["IST-DAS-2.00bpw"]["on_this_pc"].startswith("fits"))
        size = small_sets["UD-Q8_K_XL"]
        self.assertEqual((size["model"], size["id"], size["download_gb"], size["images"]), ("UD-Q8_K_XL", "deepseek-v4.1-flash-ud-q8_k_xl", 508.4, False))
        self.assertTrue(size["on_this_pc"].startswith("fits"))
        self.assertTrue(size["mem_total"] and size["multi_file"])
        self.assertEqual(size["gpu_options"]["recommended"], "all")
        small = next(f for f in s.model_table({"ram_gb": 32.0, "gpus": []}) if f["family"] == "deepseek41")
        self.assertTrue(all(x["on_this_pc"].startswith("does not fit") for x in small["sizes"]))      # no card

    def test_the_install_plan(self):
        import strata_mcp as M
        s = M.Strata(ROOT)
        s.hardware = lambda fresh=False: {"ram_gb": 251.0, "gpus": [{**c, "usable": True} for c in FOUR], "engine_vendor": "amd",
                                          "cpu": {"avx2": True}}
        s.disk_free = lambda p: 2000.0
        plan, a = M.Tools(s).install_plan("deepseek41", "UD-Q8_K_XL", None, None, None, None, None, None, None, None, None)
        self.assertEqual((plan["title"], plan["download_gb"], plan["context"], plan["images"]), ("DeepSeek-V4.1-Flash UD-Q8_K_XL", 508.4, 8192, "no"))
        self.assertEqual(a[:6], ["--yes", "--no-start", "--family", "deepseek41", "--model", "UD-Q8_K_XL"])
        self.assertTrue(plan["ram"].startswith("fits"))
        for flag in ("--kv", "--low-ram", "--experimental-speed-projection"):
            self.assertNotIn(flag, a)
        files = M.expected_files(s.setup_module(), "deepseek41", "UD-Q8_K_XL", Path("/m"), False)
        self.assertEqual(len(files), 9)
        self.assertEqual(sum(f["bytes"] for f in files), 508_376_763_197)


ISTA = "IST-DAS-2.00bpw"
RTX5070 = {"index": 0, "name": "NVIDIA GeForce RTX 5070", "vram_gb": 12.0, "vendor": "nvidia"}


class OfficialSets(unittest.TestCase):
    """The two official IST-DAS sets (ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF): the catalog, the pins, the pack recipe, the fit words."""

    def test_the_pinned_files(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        self.assertEqual(list(fam["models"]), ["IST-DAS-2.00bpw", "IST-DAS-2.25bpw", "IST-DAS-1.75bpw", "UD-Q8_K_XL"])         # the official sets first
        repo = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"
        self.assertEqual(setup.HF_REVISIONS[repo], "cb02422c900df489cea5833e8a92171224059b3c")
        self.assertIs(setup.REPO_PINS[repo], setup.DSV41_ISTA_FILES)
        self.assertNotIn("helcig/tmp-oct26-2", setup.HF_REVISIONS)           # the first release's repository is only remembered as local names
        self.assertEqual(setup.DSV41_ISTA_REPO, repo)
        self.assertEqual(len(setup.DSV41_ISTA_FILES), 13)                      # 3 x 4 shards + the Engram
        for n, (size, sha) in setup.DSV41_ISTA_FILES.items():
            self.assertRegex(sha, r"^[0-9a-f]{64}$", n)
        eng = setup.DSV41_ISTA_FILES["DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"]
        self.assertEqual(eng, (110_595_290_816, "eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a"))
        self.assertEqual(setup.DSV41_ALL_FILES["dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"],
                         (7_976_914_240, "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"))
        sums = {m: sum(setup.DSV41_ALL_FILES[n][0] for n in fam["models"][m]["shards"]) for m in list(fam["models"])[:3]}
        self.assertEqual(sums, {"IST-DAS-2.00bpw": 138238893056, "IST-DAS-2.25bpw": 155225824256, "IST-DAS-1.75bpw": 120721120256})
        self.assertEqual(sums["IST-DAS-1.75bpw"], dsv41_fit.ISTA_SETS["IST-DAS-1.75bpw"]["set_bytes"])
        self.assertEqual(sums["IST-DAS-2.00bpw"], dsv41_fit.ISTA_SETS["IST-DAS-2.00bpw"]["set_bytes"])
        self.assertEqual(sums["IST-DAS-2.25bpw"], dsv41_fit.ISTA_SETS["IST-DAS-2.25bpw"]["set_bytes"])
        self.assertEqual(setup.DSV41_ISTA_FILES["DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"][0], dsv41_fit.ISTA_ENGRAM_BYTES)
        for m in ("IST-DAS-2.00bpw", "IST-DAS-2.25bpw"):                        # both sets name the same shared files: the Engram is downloaded once
            self.assertEqual(fam["models"][m]["shared"], [setup.DSV41_ISTA_ENGRAM, setup.DSV41_SHARD.format(i=1), setup.DSV41_DSPARK])
            for n in fam["models"][m]["shared"]:
                self.assertIn(n, setup.DSV41_SHARED)
                self.assertIn(n, setup.DSV41_ALL_FILES)
        self.assertRegex(setup.DSV41_SHARED[setup.DSV41_ISTA_ENGRAM], r"ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/cb02422c[0-9a-f]{32}/$")
        self.assertRegex(fam["models"]["IST-DAS-2.25bpw"]["hf"], r"ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/resolve/cb02422c[0-9a-f]{32}/$")

    def test_shipped_profiles_match_their_hashes(self):
        import hashlib
        for n in ("profile-2.00.bin", "profile-2.25.bin"):
            data = (setup.DSV41_PROFILES / n).read_bytes()
            self.assertEqual(len(data), 66048)                                 # uint32 [43][384]
            want = (setup.DSV41_PROFILES / (n + ".sha256")).read_text().split()[0]
            self.assertEqual(hashlib.sha256(data).hexdigest(), want)

    def test_the_pack_recipe_is_the_plain_one(self):
        folder, pack = Path("/m/set"), Path("/d/pack")
        names = [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00004.gguf" for i in range(1, 5)]
        cmd = setup.dsv41_pack_cmd(folder, names, Path("/s/engram-IQ4_NL.gguf"), Path("/s/meta.gguf"), Path("/s/dspark.gguf"), pack)
        self.assertTrue(cmd[1].endswith("pack_gguf.py"))
        self.assertEqual(cmd[2:], ["--model", str(folder), "--shards", *names, "--engram", str(Path("/s/engram-IQ4_NL.gguf")),
                                   "--engram-meta", str(Path("/s/meta.gguf")), "--dspark-gguf", str(Path("/s/dspark.gguf")),
                                   "--dspark-noise-token", "128799", "--dspark-markov-rank", "256", "--out", str(pack)])
        for bad in ("--skip-tensors", "--dense-override", "--override-experts", "--tq-experts"):      # these give garbage text on the official sets
            self.assertNotIn(bad, cmd)

    def test_fit_words_use_the_measured_numbers(self):
        for ram, verdict in ((64.0, "tight"), (46.0, "tight"), (24.0, "spills"), (256.0, "fits")):
            p = dsv41_fit.plan([RTX5070], ram, None, model=ISTA)
            self.assertEqual(p["verdict"], verdict, ram)
            self.assertAlmostEqual(p["download_gb"], 256.8, delta=0.1)
            self.assertEqual(p["args"], [])
        p = dsv41_fit.plan([RTX5070], 64.0, 100.0, model=ISTA)
        text = " ".join(p["lines"]) + " ".join(p["warnings"])
        for needle in ("4.7", "7.2", "31 tokens a second", "free disk space", "only once" if False else "downloads it once"):
            self.assertIn(needle, text)
        self.assertFalse(p["disk_ok"])
        p225 = dsv41_fit.plan([RTX5070], 64.0, None, model="IST-DAS-2.25bpw")
        self.assertAlmostEqual(p225["download_gb"], 273.8, delta=0.1)
        self.assertEqual(dsv41_fit.plan([], 64.0, None, model=ISTA)["verdict"], "no_card")
        self.assertEqual(dsv41_fit.plan([RTX5070], 64.0, None, model="UD-Q8_K_XL")["verdict"], "spills")        # the big set is unchanged


class EnginePackage(unittest.TestCase):
    """DeepSeek's engine is its own package beside the engine; the Qwen engine folder is never written."""

    def test_the_package_folder_is_beside_the_engine(self):
        self.assertEqual(setup.dsv41_engine_dir(Path("/d/engine")), Path("/d") / "engine-dsv41")

    def test_the_own_package_is_found_before_the_old_place(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {"STRATA_DSV41_EXE": ""}):
            eng = Path(d) / "engine"
            eng.mkdir()
            name = "strata-dsv41.exe" if setup.WIN else "strata-dsv41"
            self.assertIsNone(setup.dsv41_exe(eng))
            (eng / name).write_text("old")
            self.assertEqual(setup.dsv41_exe(eng), eng / name)                 # an all-in-one engine still works
            (Path(d) / "engine-dsv41" / "0.1.49-x").mkdir(parents=True)
            (Path(d) / "engine-dsv41" / "0.1.49-x" / name).write_text("new")
            self.assertEqual(setup.dsv41_exe(eng), Path(d) / "engine-dsv41" / "0.1.49-x" / name)


class SharedFiles(unittest.TestCase):
    def test_the_drafter_beside_the_sets_folder_is_found_from_the_set_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "off" / "2.00bpw").mkdir(parents=True)
            (root / "dspark").mkdir()
            f = root / "dspark" / setup.DSV41_DSPARK
            f.write_bytes(b"x")
            a = argparse.Namespace(gguf_dir=str(root / "off" / "2.00bpw"))
            with mock.patch.dict(setup.DSV41_ALL_FILES, {setup.DSV41_DSPARK: (1, "0" * 64)}):
                got = setup.dsv41_find_shared(a, setup.DSV41_DSPARK, root / "off" / "2.00bpw", root / "models" / "shared")
            self.assertEqual(got, f)


class LegacyFiles(unittest.TestCase):
    """Files of the first release (model-2.00bpw-0000N-of-00004.gguf, engram-IQ4_NL.gguf, often in a 2.00bpw/ folder) are the same weights: accepted."""

    def test_the_old_pins_are_the_old_files_and_the_unchanged_shards_match_today_s(self):
        for i in (2, 3, 4):                                                   # same bytes: same size and SHA-256 under both names
            new = setup.DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=i)
            self.assertEqual(setup.DSV41_ALL_FILES[new], setup.DSV41_LEGACY_FILES["model-2.00bpw-%05d-of-00004.gguf" % i])
            new = setup.DSV41_ISTA_SHARD.format(quant="IQ2_XS", i=i)
            self.assertEqual(setup.DSV41_ALL_FILES[new], setup.DSV41_LEGACY_FILES["model-2.25bpw-%05d-of-00004.gguf" % i])
        self.assertNotEqual(setup.DSV41_ALL_FILES[setup.DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=1)],
                            setup.DSV41_LEGACY_FILES["model-2.00bpw-00001-of-00004.gguf"])        # shard 1 and the Engram differ in their header only
        self.assertNotIn("model-2.00bpw-00001-of-00004.gguf", setup.DSV41_ISTA_FILES)                 # (never downloaded under the old names)
        self.assertNotIn("model-2.00bpw-00001-of-00004.gguf", setup.REPO_PINS["ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"])

    def test_pick_prefers_today_s_name_then_the_old_one(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            new = setup.DSV41_ISTA_SHARD.format(quant="IQ2_XXS", i=2)
            self.assertEqual(setup.dsv41_pick(folder, new), folder / new)                  # nothing there: where it will be downloaded
            (folder / "model-2.00bpw-00002-of-00004.gguf").write_bytes(b"x")
            self.assertEqual(setup.dsv41_pick(folder, new).name, "model-2.00bpw-00002-of-00004.gguf")
            (folder / new).write_bytes(b"y")
            self.assertEqual(setup.dsv41_pick(folder, new).name, new)

    def test_the_folder_of_the_first_release_is_found_from_the_parent(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "2.00bpw"
            sub.mkdir()
            (sub / "model-2.00bpw-00001-of-00004.gguf").write_bytes(b"x")
            fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
            a = argparse.Namespace(gguf_dir=d, models_dir=d)
            self.assertEqual(setup.dsv41_folder(a, fam, "t", "IST-DAS-2.00bpw"), sub)
            a2 = argparse.Namespace(gguf_dir=str(sub), models_dir=d)
            self.assertEqual(setup.dsv41_folder(a2, fam, "t", "IST-DAS-2.00bpw"), sub)

    def test_the_old_engram_name_is_found(self):
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "2.00bpw"
            sub.mkdir()
            old = sub / "engram-IQ4_NL.gguf"
            old.write_bytes(b"e")
            a = argparse.Namespace(gguf_dir=d)
            with mock.patch.dict(setup.DSV41_ALL_FILES, {"engram-IQ4_NL.gguf": (1, "0" * 64)}):
                got = setup.dsv41_find_shared(a, setup.DSV41_ISTA_ENGRAM, sub, Path(d) / "shared")
            self.assertEqual(got, old)


class Smallest(unittest.TestCase):
    """IST-DAS 1.75 bpw (IQ1_M): the smallest official set, recommended where 2.00 is tight, never forced."""

    def test_the_files_the_pins_and_the_recipe(self):
        fam = setup.MULTI_FILE_FAMILIES["deepseek41"]
        d = fam["models"]["IST-DAS-1.75bpw"]
        self.assertEqual(d["recipe"], "ista")
        self.assertEqual(d["shards"], [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-{i:05d}-of-00004.gguf" for i in range(1, 5)])
        self.assertEqual(d["shared"], fam["models"]["IST-DAS-2.00bpw"]["shared"])              # the same Engram, drafter and metadata shard
        self.assertEqual(setup.DSV41_ISTA_FILES[d["shards"][0]],
                         (39358830464, "6e3f0d5ff53d00428ec32c079db9e8255cb410ccaa435e5342aaa8a47fe1339d"))
        self.assertEqual(setup.DSV41_ISTA_FILES[d["shards"][3]][0], 2188841824)
        self.assertEqual(sum(setup.DSV41_ISTA_FILES[n][0] for n in d["shards"]), 120_721_120_256)
        self.assertTrue(all(n not in setup.DSV41_LEGACY_NAMES.values() for n in d["shards"]))     # no first-release names for it
        cmd = setup.dsv41_pack_cmd(Path("/m"), d["shards"], Path("/s/e.gguf"), Path("/s/m.gguf"), Path("/s/d.gguf"), Path("/p"))
        self.assertIn("--dspark-noise-token", cmd)
        self.assertNotIn("--skip-tensors", cmd)

    def test_it_is_the_smallest_and_warns_never_refuses(self):
        sizes = {m: dsv41_fit.ista_download_bytes(m) / 1e9 for m in dsv41_fit.ISTA_SETS}
        self.assertEqual(min(sizes, key=sizes.get), "IST-DAS-1.75bpw")
        self.assertAlmostEqual(sizes["IST-DAS-1.75bpw"], 239.3, delta=0.1)
        for ram in (24.0, 36.0, 64.0):
            p = dsv41_fit.plan([RTX5070], ram, None, model="IST-DAS-1.75bpw")
            self.assertIn(p["verdict"], ("fits", "tight", "spills"))
            self.assertTrue(p["lines"])
        self.assertEqual(dsv41_fit.plan([RTX5070], 36.0, None, model="IST-DAS-1.75bpw")["verdict"], "tight")      # runs where 2.00 spills
        self.assertEqual(dsv41_fit.plan([RTX5070], 36.0, None, model="IST-DAS-2.00bpw")["verdict"], "spills")

    def test_it_is_recommended_where_2_00_is_tight(self):
        self.assertEqual(dsv41_fit.recommend(64.0, 900.0), "IST-DAS-2.00bpw")
        self.assertEqual(dsv41_fit.recommend(None, None), "IST-DAS-2.00bpw")                   # unknown: the usual one
        self.assertEqual(dsv41_fit.recommend(32.0, 900.0), "IST-DAS-1.75bpw")                  # less RAM than 2.00 was measured with
        self.assertEqual(dsv41_fit.recommend(64.0, 250.0), "IST-DAS-1.75bpw")                  # disk fits 1.75 (239 + 6) but not 2.00 (257 + 6)
        self.assertEqual(dsv41_fit.recommend(64.0, 100.0), "IST-DAS-2.00bpw")                  # fits neither: no reason to prefer it

    def test_the_menu_marks_the_recommendation_and_still_lets_anyone_choose(self):
        import io
        names = list(setup.MULTI_FILE_FAMILIES["deepseek41"]["models"])
        self.assertIn("IST-DAS-1.75bpw", names)
        self.assertEqual(names.index(dsv41_fit.recommend(32.0, 900.0)) + 1, 3)


class InstallOfficial(Install):
    """install_dsv41 for an official set, the engine and the tools mocked."""

    def install_ista(self, model=ISTA, ram=64.0, **kw):
        self.packinfo_ok = True
        self.smoke_ok = True
        patches = [mock.patch.object(setup, "dsv41_pack_info", lambda exe, pack, libs: (self.packinfo_ok, "files: 6, all present and long enough")),
                   mock.patch.object(setup, "dsv41_smoke", lambda *a, **k: (self.smoke_ok, "The capital of France is Paris.")
                                     if self.smoke_ok else (False, "test question failed: the answer is empty"))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return self.install(ram=ram, cards=[RTX5070], model=model, **kw)

    def test_downloads_the_set_the_engram_once_and_the_drafter(self):
        rc, out = self.install_ista()
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        folder = self.root / "models" / "deepseek-v4.1-flash-IST-DAS-2.00bpw"
        shared = self.root / "models" / "deepseek-v4.1-flash-shared"
        self.assertEqual([c[2] for c in get][:4], [folder / f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-{i:05d}-of-00004.gguf" for i in range(1, 5)])
        self.assertEqual([c[2] for c in get][4:], [shared / setup.DSV41_ISTA_ENGRAM, shared / setup.DSV41_SHARD.format(i=1),
                                                  shared / setup.DSV41_DSPARK])
        self.assertTrue(get[0][1].endswith("/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf"))
        self.assertNotIn("/2.00bpw/", get[0][1])             # the repository is flat
        self.assertEqual(sum(1 for c in self.calls if c[0] == "sha"), 7)
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        self.assertIn("--engram-meta", pack)
        self.assertNotIn("--skip-tensors", pack)

    def test_the_second_set_does_not_download_the_engram_again(self):
        shared = self.root / "models" / "deepseek-v4.1-flash-shared"
        shared.mkdir(parents=True)
        shared_names = (setup.DSV41_ISTA_ENGRAM, setup.DSV41_SHARD.format(i=1), setup.DSV41_DSPARK)
        for n in shared_names:
            (shared / n).write_bytes(b"x")
            (shared / (n + ".done")).write_text("ok")
        tiny = {n: (1, setup.DSV41_ALL_FILES[n][1]) for n in shared_names}      # the stand-in files are one byte long
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(model="IST-DAS-2.25bpw")
        self.assertEqual(rc, 0, out)
        got = [c[2].name for c in self.calls if c[0] == "get"]
        self.assertEqual(got, [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-{i:05d}-of-00004.gguf" for i in range(1, 5)])      # only the four shards

    def test_the_first_releases_files_are_used_as_they_are_nothing_is_downloaded_but_the_small_files(self):
        own = self.root / "own"
        sub = own / "2.00bpw"
        sub.mkdir(parents=True)
        for i in range(1, 5):
            (sub / f"model-2.00bpw-{i:05d}-of-00004.gguf").write_bytes(b"GGUF")
        (sub / "engram-IQ4_NL.gguf").write_bytes(b"E")
        (own / "dspark").mkdir()
        (own / "dspark" / setup.DSV41_DSPARK).write_bytes(b"D")
        tiny = {n: (4, "0" * 64) for n in setup.DSV41_LEGACY_FILES if n.startswith("model-2.00bpw")}
        tiny["engram-IQ4_NL.gguf"] = (1, "0" * 64)
        tiny[setup.DSV41_DSPARK] = (1, "0" * 64)
        with mock.patch.dict(setup.DSV41_ALL_FILES, tiny):
            rc, out = self.install_ista(gguf_dir=str(own))
        self.assertEqual(rc, 0, out)
        got = [c[2].name for c in self.calls if c[0] == "get"]
        self.assertEqual(got, [setup.DSV41_SHARD.format(i=1)])                 # only the 5.8 MB metadata shard
        pack = [c for c in self.calls if c[0] == "run"][-1][1]
        shards = pack[pack.index("--shards") + 1:pack.index("--engram")]
        self.assertEqual(shards, [f"model-2.00bpw-{i:05d}-of-00004.gguf" for i in range(1, 5)])      # packed from the files as they are
        self.assertTrue(pack[pack.index("--engram") + 1].endswith("engram-IQ4_NL.gguf"))
        self.assertIn("model", pack[pack.index("--model") + 1] + "model")
        self.assertEqual(Path(pack[pack.index("--model") + 1]), sub)

    def test_pack_info_profile_smoke_and_the_launch_flags(self):
        rc, out = self.install_ista()
        self.assertEqual(rc, 0, out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--dspark") + 1], "auto")                    # never "off": that still loads the drafter and takes the VRAM
        self.assertNotIn("off", a)
        self.assertNotIn("--prefill-chunk", a)                                  # the engine's own default is the layer-major prompt path
        self.assertEqual(a[a.index("--dspark-conf") + 1], "0.65")                # never a floor below 0.3             # the layer-major prompt path
        self.assertNotIn("--ssd-cache", a)                                      # the store sizes itself
        self.assertEqual(a[a.index("--max-context") + 1], "8192")
        self.assertIn("checked: every file the engine needs is there", out)
        self.assertIn("test question answered sensibly", out)
        self.assertTrue(Path(a[a.index("--pack") + 1]).match("*packs/deepseek-v4.1-flash-ist-das-2.00bpw"))

    def test_an_incomplete_pack_stops_with_the_missing_file(self):
        self.packinfo_ok = False
        patches = [mock.patch.object(setup, "dsv41_pack_info", lambda exe, pack, libs: (False, "  MISSING x.gguf (1.00 GB wanted)")),
                   mock.patch.object(setup, "dsv41_smoke", lambda *a, **k: (True, "Paris"))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        rc, out = self.install(ram=64.0, cards=[RTX5070], model=ISTA)
        self.assertEqual(rc, 1, out)
        self.assertIn("pack is not complete", out)
        self.assertIn("MISSING", out)

    def test_a_failed_text_check_is_said_plainly_and_stops(self):
        patches = [mock.patch.object(setup, "dsv41_pack_info", lambda exe, pack, libs: (True, "all present")),
                   mock.patch.object(setup, "dsv41_smoke", lambda *a, **k: (False, "!!!!\ntest question failed: the answer is empty"))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        rc, out = self.install(ram=64.0, cards=[RTX5070], model=ISTA)
        self.assertEqual(rc, 1, out)
        self.assertIn("does not answer sensibly", out)
        self.assertIn("run setup again", out)
        self.assertFalse(self.cfg)                                              # no start script for a model that failed its check

    def test_a_disk_that_cannot_hold_the_set_stops(self):
        with mock.patch.object(setup, "free_gb", lambda p: 100.0):
            rc, out = self.install_ista()
        self.assertEqual(rc, 1, out)
        self.assertIn("not enough free disk space", out)


class InstallSmallest(InstallOfficial):
    def test_downloads_the_iq1_m_files_and_the_shared_ones(self):
        rc, out = self.install_ista(model="IST-DAS-1.75bpw", ram=36.0)
        self.assertEqual(rc, 0, out)
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual([c[2].name for c in get][:4], [f"DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-{i:05d}-of-00004.gguf" for i in range(1, 5)])
        self.assertTrue(get[0][1].startswith(setup.hf(setup.DSV41_ISTA_REPO)))
        self.assertIn("cb02422c900df489cea5833e8a92171224059b3c", get[0][1])
        self.assertIn("239 GB", out)
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--dspark") + 1], "auto")


if __name__ == "__main__":
    unittest.main()
