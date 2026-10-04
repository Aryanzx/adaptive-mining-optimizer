import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math import comb
from combination import ncr, nth_combination, iter_combinations, \
    CombinationEngine
from config import Config, PH_EXPLORE, PH_EXPLOIT


class TestCombination(unittest.TestCase):
    def test_ncr(self):
        self.assertEqual(ncr(10, 3), 120)
        self.assertEqual(ncr(4, 2), 6)
        self.assertEqual(ncr(4, 5), 0)

    def test_unrank_covers_exactly(self):
        n, r = 7, 3
        seen = set()
        for idx in range(comb(n, r)):
            seen.add(nth_combination(n, r, idx))
        self.assertEqual(len(seen), comb(n, r))
        for c in seen:
            self.assertEqual(len(c), r)
            self.assertEqual(len(set(c)), r)
            self.assertTrue(all(0 <= x < n for x in c))
            self.assertEqual(list(c), sorted(c))

    def test_unrank_first_last(self):
        self.assertEqual(nth_combination(5, 2, 0), (0, 1))
        self.assertEqual(nth_combination(5, 2, comb(5, 2) - 1), (3, 4))

    def test_bounded_lazy_stream(self):
        cfg = Config()
        rng = __import__("random").Random(1)
        eng = CombinationEngine(cfg, rng)
        out = list(eng.bounded(50, 5, 10))     # C(50,5)=2,118,760 >> 10
        self.assertEqual(len(out), 10)
        self.assertEqual(len(set(out)), 10)    # no duplicates
        full = list(iter_combinations(5, 2, 100, rng))   # 10 <= 100
        self.assertEqual(len(full), comb(5, 2))

    def test_adaptive_r(self):
        eng = CombinationEngine(Config(), __import__("random").Random(1))
        self.assertEqual(eng.adaptive_r(9, PH_EXPLOIT), 2)
        self.assertEqual(eng.adaptive_r(9, PH_EXPLORE), 3)


if __name__ == "__main__":
    unittest.main()
