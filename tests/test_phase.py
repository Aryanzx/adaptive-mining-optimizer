import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase import PhaseManager
from config import (Config, PH_EXPLORE, PH_DISCOVER, PH_VALIDATE,
                    PH_EXPLOIT, PH_STAGNATION, PH_RESET)


class TestPhase(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(discover_min_patterns=2, validate_loops=2,
                          stagnation_window=3, exploit_max_dwell=5)
        self.pm = PhaseManager(self.cfg)

    def test_explore_to_discover_to_validate(self):
        self.assertEqual(self.pm.phase, PH_EXPLORE)
        self.pm.note_loop(0, False, 3, 0)      # patterns detected
        self.pm.note_loop(1, False, 3, 0)      # pending >= min -> DISCOVER
        self.assertEqual(self.pm.phase, PH_DISCOVER)
        self.pm.note_loop(2, False, 0, 0)
        self.pm.note_loop(3, False, 0, 0)      # 2 loops in phase
        self.assertEqual(self.pm.phase, PH_VALIDATE)

    def test_validate_to_exploit_requires_validation(self):
        pm = self.pm
        pm.note_loop(0, False, 3, 0)           # EXPLORE -> DISCOVER
        pm.note_loop(1, False, 3, 0)
        pm.note_loop(2, False, 0, 0)           # DISCOVER -> VALIDATE
        pm.note_loop(3, False, 0, 0)
        pm.note_loop(4, False, 0, 1)           # validated_now -> EXPLOIT
        self.assertEqual(pm.phase, PH_EXPLOIT)

    def test_validate_without_validation_returns_to_explore(self):
        pm = self.pm
        pm.note_loop(0, False, 3, 0)
        pm.note_loop(1, False, 3, 0)
        pm.note_loop(2, False, 0, 0)
        pm.note_loop(3, False, 0, 0)
        pm.note_loop(4, False, 0, 0)           # VALIDATE done, nothing proven
        self.assertEqual(pm.phase, PH_EXPLORE)

    def test_exploit_dwell_and_stagnation_reset(self):
        pm = self.pm
        pm.phase = PH_EXPLOIT
        for i in range(5):                     # dwell cap
            pm.note_loop(10 + i, True, 0, 0)
        self.assertEqual(pm.phase, PH_STAGNATION)
        pm.note_loop(20, True, 0, 0)
        self.assertEqual(pm.phase, PH_RESET)
        pm.note_loop(21, True, 0, 0)
        self.assertEqual(pm.phase, PH_EXPLORE)

    def test_never_stuck_exploiting(self):
        pm = PhaseManager(Config(stagnation_window=2, exploit_max_dwell=3))
        pm.phase = PH_EXPLOIT
        for i in range(50):
            pm.note_loop(i, True, 0, 0)        # endless improvement
        self.assertNotEqual(pm.phase, PH_EXPLOIT)   # dwell cap fired

    def test_exploration_rates(self):
        pm = self.pm
        pm.phase = PH_EXPLORE
        self.assertGreaterEqual(pm.exploration_rate(0.0), 0.72)
        pm.phase = PH_STAGNATION
        self.assertGreaterEqual(pm.exploration_rate(0.9), 0.90)
        pm.phase = PH_EXPLOIT
        self.assertLessEqual(pm.exploration_rate(0.9), 0.30)
        pm.phase = PH_EXPLOIT
        self.assertLessEqual(pm.exploration_rate(0.0), 0.40)  # half of 0.8


if __name__ == "__main__":
    unittest.main()
