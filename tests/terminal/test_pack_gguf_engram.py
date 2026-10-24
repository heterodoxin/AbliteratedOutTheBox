"""pack_gguf must never take an Engram file's metadata for the model's: ISTA-DASLab's final Engram says general.architecture 'deepseek41-engram'
(the first release's said 'engram').  The fixture is the real header of DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf (its 416 bytes: metadata and the tensor
directory, no data)."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "dsv41"))
import pack_gguf as PG  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "dsv41" / "engram-final-header.gguf"


def tiny_gguf(path: Path, arch: str, extra: dict | None = None) -> Path:
    """A GGUF with only metadata (string and uint32 values), no tensors."""
    kv = {"general.architecture": arch, **(extra or {})}
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)))
    for k, v in kv.items():
        out += struct.pack("<Q", len(k)) + k.encode()
        if isinstance(v, str):
            out += struct.pack("<I", 8) + struct.pack("<Q", len(v)) + v.encode()
        else:
            out += struct.pack("<I", 4) + struct.pack("<I", v)
    out += b"\0" * (-len(out) % 32)
    path.write_bytes(bytes(out))
    return path


class EngramHeader(unittest.TestCase):
    def test_the_real_final_engram_header_is_an_engram_file(self):
        g = PG.Gguf(FIXTURE)
        self.assertEqual(g.kv["general.architecture"], "deepseek41-engram")
        self.assertTrue(PG.is_engram_file(g))
        self.assertEqual(len(g.tensors), 2)
        self.assertEqual(g.data_start, 416)

    def test_it_is_never_the_group_the_model_metadata_comes_from(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            shard = PG.Gguf(tiny_gguf(d / "model-00001-of-00004.gguf", "deepseek41", {"split.count": 4}))
            engram = PG.Gguf(FIXTURE)
            for order in ([engram, shard], [shard, engram]):
                kv = PG.model_kv(order)
                self.assertIsNotNone(kv)
                self.assertEqual(kv["general.architecture"], "deepseek41")
            self.assertIsNone(PG.model_kv([engram]))                    # an Engram alone is not a model

    def test_the_first_releases_engram_is_one_too(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "e.gguf", "engram"))))
            self.assertFalse(PG.is_engram_file(PG.Gguf(tiny_gguf(Path(d) / "m.gguf", "deepseek41"))))


if __name__ == "__main__":
    unittest.main()
