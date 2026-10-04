import os
import sys
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from permutation import PermutationEngine, apply_transform
from config import Config, PH_EXPLORE, PH_EXPLOIT, PERMUTATION_TABLE


class TestPermutation(unittest.TestCase):
    def test_all_transforms_are_permutations(self):
        rng = random.Random(7)
        src = (3, 1, 4, 1, 5, 9, 2, 6)
        for pid in range(len(PERMUTATION_TABLE)):
            out = apply_transform(pid, src, rng)
            self.assertEqual(sorted(out), sorted(src),
                             "transform %d broke multiset" % pid)

    def test_engine_score_bounds_and_record(self):
        cfg = Config()
        eng = PermutationEngine(cfg, random.Random(2))
        for p in range(7):
            s = eng.permutation_score(p, PH_EXPLORE)
            self.assertTrue(0.0 <= s <= 1.0)
        eng.record(3, 1.0)
        self.assertEqual(eng.frequency[3], 1)
        self.assertGreater(eng.quality[3], 0.0)
        self.assertEqual(eng.cur_perm, 3)
        self.assertEqual(eng.prev_perm, -1)   # first record had no prev
        eng.record(5, 1.0)
        self.assertEqual(eng.prev_perm, 3)    # prev now tracks history

    def test_exploit_phase_uses_pure_quality(self):
        eng = PermutationEngine(Config(), random.Random(3))
        eng.quality[2] = 0.9
        eng.frequency[2] = 10_000
        self.assertAlmostEqual(eng.permutation_score(2, PH_EXPLOIT), 0.9)
        # explore phase blends novelty (low here): quality-heavy
        s = eng.permutation_score(2, PH_EXPLORE)
        self.assertLess(s, 0.9)


if __name__ == "__main__":
    unittest.main()
