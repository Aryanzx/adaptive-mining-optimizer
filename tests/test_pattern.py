import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pattern import PatternEngine, PatternTable, R_REP, R_CYCLE2, R_CYCLE3, \
    R_RUN, R_TRANS, _canon
from config import Config, PS_POSITION, PS_PHASE

S = PS_POSITION  # shorthand


class TestPattern(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.table = PatternTable(self.cfg)
        self.eng = PatternEngine(self.cfg, self.table)

    def kinds(self, keys):
        return [k[1] for k in keys]

    def test_repetition(self):
        self.eng.push(S, 3)
        ks = self.eng.push(S, 3)
        self.assertIn(R_REP, self.kinds(ks))

    def test_cycle2(self):
        self.eng.push(S, 1)
        self.eng.push(S, 2)
        ks = self.eng.push(S, 1)            # A B A
        self.assertIn(R_CYCLE2, self.kinds(ks))

    def test_cycle3(self):
        for v in (5, 6, 7):
            self.eng.push(S, v)
        ks = self.eng.push(S, 5)            # A B C A
        self.assertIn(R_CYCLE3, self.kinds(ks))

    def test_run(self):
        for v in (2, 4, 6):
            self.eng.push(S, v)
        ks = self.eng.push(S, 8)            # +2 run
        self.assertIn(R_RUN, self.kinds(ks))

    def test_transition(self):
        self.eng.push(S, 1)
        ks = self.eng.push(S, 4)
        self.assertIn(R_TRANS, self.kinds(ks))

    def test_interning_stable_and_bounded(self):
        p1, new1 = self.table.intern((S, R_REP, 1))
        p2, new2 = self.table.intern((S, R_REP, 1))
        self.assertTrue(new1 and not new2)
        self.assertEqual(p1, p2)
        self.assertEqual(self.table.freq[p1], 2)
        # hard cap honoured
        self.table.hard_cap = 4
        ok_all = True
        for i in range(10):
            pid, _ = self.table.intern((PS_PHASE, R_TRANS, i, i + 1))
            ok_all = ok_all and (pid is not None)
        self.assertFalse(ok_all)            # refused beyond cap

    def test_canonical_rotation(self):
        self.assertEqual(_canon((3, 1, 2)), (1, 2, 3))
        self.assertEqual(_canon((2, 3, 1)), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
