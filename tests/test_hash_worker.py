import os
import sys
import hashlib
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hash_worker import HashWorker
from config import Config


class TestHashWorker(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(difficulty=8)
        self.w = HashWorker(self.cfg)

    def test_matches_stdlib_sha256(self):
        ok, d = self.w.hash_nonce(123456)
        ref = hashlib.sha256((123456).to_bytes(4, "little")).digest()
        self.assertEqual(d, ref)
        self.assertEqual(self.w.leading_zero_bits(ref),
                         61 if ref[0] == 0 else 8 - ref[0].bit_length())

    def test_leading_zeros(self):
        self.assertEqual(self.w.leading_zero_bits(bytes([0, 0, 1])), 23)
        self.assertEqual(self.w.leading_zero_bits(bytes([0, 0, 255])), 16)
        self.assertEqual(self.w.leading_zero_bits(bytes([255])), 0)
        self.assertEqual(self.w.leading_zero_bits(bytes(32)), 256)

    def test_difficulty_check_consistent(self):
        # compare fast check against naive counting over many nonces
        hits_fast = hits_slow = 0
        for n in range(2000):
            ok, d = self.w.hash_nonce(n)
            lz = self.w.leading_zero_bits(d)
            hits_fast += ok
            hits_slow += lz >= 8
        self.assertEqual(hits_fast, hits_slow)

    def test_batch_throughput_recorded(self):
        stats = type("S", (), {"note_hash_time": staticmethod(
            lambda n, dt: None)})()
        w = HashWorker(self.cfg, stats)
        oks, dt = w.hash_batch(list(range(500)))
        self.assertEqual(len(oks), 500)
        self.assertGreater(dt, 0.0)
        self.assertGreater(w.khs(), 0.0)

    def test_difficulty_bounds(self):
        with self.assertRaises(ValueError):
            Config(difficulty=65)


if __name__ == "__main__":
    unittest.main()
