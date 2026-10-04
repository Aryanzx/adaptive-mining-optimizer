import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from engine import AdaptiveEngine, Logger
from candidate import SearchSpace, derive_nonce, Candidate


class TestSearchSpace(unittest.TestCase):
    def test_rank_roundtrip(self):
        sp = SearchSpace((10, 10, 10, 20, 10, 100, 10))
        # NOTE: the product of the spec's example dimensions is 200,000,000
        # (the prose "20,000,000" dropped one factor of 10).
        self.assertEqual(sp.total, 200_000_000)
        for r in (0, 1, 999, 1234567, sp.total - 1):
            st = sp.rank_to_state(r)
            self.assertEqual(sp.state_to_rank(st), r)

    def test_lazy_generator_covers(self):
        import itertools
        sp = SearchSpace((3, 4))
        got = [s for _, s in itertools.islice(sp.next_candidate(start=0), 12)]
        self.assertEqual(len(got), 12)
        self.assertEqual(len(set(got)), 12)
        # wraps around lazily without storing
        gen = sp.next_candidate(start=11)
        r, s = next(gen)
        self.assertEqual(r, 11)
        r, s = next(gen)
        self.assertEqual(r, 0)


class TestNonceDerivation(unittest.TestCase):
    def test_deterministic_and_sensitive(self):
        a = derive_nonce((1, 2, 3), 0, 1, 2, salt=1)
        b = derive_nonce((1, 2, 3), 0, 1, 2, salt=1)
        c = derive_nonce((1, 2, 4), 0, 1, 2, salt=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(0 <= a < 2 ** 32)


class TestEngineEndToEnd(unittest.TestCase):
    def test_small_run_completes(self):
        cfg = Config(max_hashes=4000, batch_size=128, pool_width=2,
                     difficulty=6, log_every=10_000)
        eng = AdaptiveEngine(cfg, seed=42, log=Logger(enabled=False))
        s = eng.run()
        self.assertEqual(s["attempts"], 4000)
        self.assertGreaterEqual(s["successes"], 1)   # 4000/64 expected
        self.assertTrue(0 <= s["dup_rate"] < 0.5)
        self.assertLessEqual(s["memory_kb"], cfg.max_memory_kb)

    def test_determinism_same_seed(self):
        cfg = Config(max_hashes=3000, batch_size=128, difficulty=6,
                     deterministic=True)
        a = AdaptiveEngine(cfg, seed=7, log=Logger(enabled=False)).run()
        b = AdaptiveEngine(cfg, seed=7, log=Logger(enabled=False)).run()
        self.assertEqual(a["successes"], b["successes"])
        self.assertEqual(a["final_phase"], b["final_phase"])
        self.assertEqual(a["patterns_discovered"], b["patterns_discovered"])

    def test_control_mode_ignores_patterns(self):
        # Patterns are still *detected* (passive statistics), but in
        # control mode they never influence generation.
        cfg = Config(max_hashes=2000, batch_size=128, difficulty=6,
                     use_patterns=False)
        eng = AdaptiveEngine(cfg, seed=9, log=Logger(enabled=False))
        s = eng.run()
        self.assertEqual(eng.active_patterns, ())
        self.assertEqual(eng.bias_rel, [0.0] * 9)
        self.assertEqual(eng.bias_perm, [0.0] * 7)

    def test_memory_budget_respected(self):
        cfg = Config(max_hashes=5000, batch_size=128, difficulty=6,
                     max_memory_kb=256)
        s = AdaptiveEngine(cfg, seed=11, log=Logger(enabled=False)).run()
        self.assertLessEqual(s["memory_kb"], cfg.max_memory_kb)


if __name__ == "__main__":
    unittest.main()
