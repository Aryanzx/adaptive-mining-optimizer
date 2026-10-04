import os
import sys
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relation import RelationEngine, classify_delta, DELTA_TO_REL, \
    D_REPEAT, D_P1, D_M1, D_JUMP
from config import (Config, REL_UNCHANGED, REL_INC, REL_DEC, REL_SWAP,
                    REL_REVERSE, REL_ROTATE, REL_NEIGHBOR, REL_COUSIN,
                    REL_SHIFT)


class TestRelation(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.eng = RelationEngine(self.cfg, random.Random(5))
        self.state = (1, 2, 3, 4, 5, 6, 7)

    def test_relations_keep_state_in_bounds(self):
        for rel in range(9):
            for pos in range(len(self.state)):
                st = self.eng.apply(rel, self.state, pos)
                self.assertEqual(len(st), len(self.state))
                for i, v in enumerate(st):
                    self.assertTrue(0 <= v < self.cfg.dimensions[i])

    def test_specific_relations(self):
        self.assertEqual(self.eng.apply(REL_INC, self.state, 0), (2, 2, 3, 4, 5, 6, 7))
        self.assertEqual(self.eng.apply(REL_DEC, self.state, 2), (1, 2, 2, 4, 5, 6, 7))
        self.assertEqual(self.eng.apply(REL_SWAP, self.state, 6)[0],
                         self.state[6])          # swap with wrap-around
        self.assertEqual(self.eng.apply(REL_REVERSE, self.state, 2)[:3],
                         (3, 2, 1))
        self.assertEqual(self.eng.apply(REL_ROTATE, self.state, 2),
                         (3, 4, 5, 6, 7, 1, 2))
        self.assertEqual(self.eng.apply(REL_UNCHANGED, self.state, 0),
                         self.state)

    def test_delta_classification(self):
        self.assertEqual(classify_delta(0), D_REPEAT)
        self.assertEqual(classify_delta(1), D_P1)
        self.assertEqual(classify_delta(-1), D_M1)
        self.assertEqual(classify_delta(9), D_JUMP)
        self.assertEqual(DELTA_TO_REL[D_P1], REL_INC)

    def test_position_tracking(self):
        self.eng.observe_position(2)
        self.eng.observe_position(3)
        self.eng.observe_position(4)
        self.assertEqual(self.eng.recent_deltas(), (1, 1))
        self.assertEqual(self.eng.last_delta_class(), D_P1)
        nov_before = self.eng.position_novelty(4)
        self.eng.observe_position(4)
        self.assertLess(self.eng.position_novelty(4), nov_before)


if __name__ == "__main__":
    unittest.main()
