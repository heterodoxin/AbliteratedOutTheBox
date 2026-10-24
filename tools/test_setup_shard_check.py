"""setup.check_shards: a metadata-only GGUF shard that ends at its header (no alignment padding, #2346) is whole;
a shard whose tensor data runs past its end is still refused."""
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(path, tensors=(), pad_data=0):
    """A GGUF v3 file with one string key and the given (name, n_floats) F32 tensors."""
    kv = _str("general.name") + struct.pack("<I", 8) + _str("t")
    tinfo = b""
    off = 0
    for name, n in tensors:
        tinfo += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<I", 0) + struct.pack("<Q", off)
        off += ((n * 4 + 31) // 32) * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 1) + kv + tinfo
    body = head
    if tensors:
        body += b"\0" * ((32 - len(head) % 32) % 32)
        body += b"\0" * pad_data
    path.write_bytes(body)
    return len(head)


class ShardCheck(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def test_metadata_only_shard_without_padding_is_whole(self):
        p = self.d / "m-00001-of-00006.gguf"
        n = gguf(p)
        self.assertNotEqual(n % 32, 0, "the test file must end unaligned, like the real shard")
        setup.check_shards([p])          # must not stop setup

    def test_truncated_tensor_data_is_still_refused(self):
        p = self.d / "t-00002-of-00006.gguf"
        gguf(p, tensors=[("w", 64)], pad_data=100)       # needs 256 bytes of data, has 100
        with self.assertRaises(SystemExit):
            setup.check_shards([p])


if __name__ == "__main__":
    unittest.main()
