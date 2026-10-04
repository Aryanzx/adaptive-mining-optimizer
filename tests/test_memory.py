import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory import MemoryManager
from config import Config, MEM_HOT, MEM_ACTIVE, MEM_COLD, PH_EXPLOIT, \
    PH_STAGNATION


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(discard_score=0.2, active_score=0.4)
        self.mem = MemoryManager(self.cfg)

    def test_observe_and_aging(self):
        self.mem.observe(7, 1.0, PH_EXPLOIT, 0.9)
        e = self.mem.get(7)
        self.assertIsNotNone(e)
        self.assertAlmostEqual(e.score, 1.0)
        self.mem.observe(7, 0.0, PH_EXPLOIT, 0.9)
        self.assertAlmostEqual(e.score, self.cfg.decay)   # decayed + 0

    def test_tiers_promote_demote(self):
        for _ in range(50):
            self.mem.observe(9, 1.0, PH_EXPLOIT, 0.95)
        e = self.mem.get(9)
        self.assertTrue(e.tier in (MEM_HOT, MEM_ACTIVE))
        for _ in range(400):                  # starve it (decay 0.98^n)
            self.mem.observe(9, 0.0, PH_EXPLOIT, 0.1)
        self.assertIsNone(self.mem.get(9))    # aged out -> discarded
        self.assertGreater(self.mem.discarded, 0)

    def test_caps_enforced(self):
        cfg = Config(max_hot_candidates=2, max_active_patterns=2,
                     discard_score=0.0)
        mem = MemoryManager(cfg)
        for pid in range(20):
            for _ in range(3):
                mem.observe(pid, 1.0, PH_EXPLOIT, 0.9)
            mem.observe(pid, 1.0, PH_EXPLOIT, 0.95)
        c = mem.counts()
        self.assertLessEqual(c[MEM_HOT], 2)
        self.assertLessEqual(c[MEM_ACTIVE], 2)
        self.assertLessEqual(c[MEM_COLD], 4)

    def test_prune_stagnation_forgets(self):
        self.mem.observe(3, 0.3, PH_EXPLOIT, 0.5)
        self.mem.prune(PH_STAGNATION)
        self.assertIsNone(self.mem.get(3))    # double decay kills weak

    def test_usage_within_budget(self):
        cfg = Config()
        mem = MemoryManager(cfg)
        for pid in range(cfg.max_active_patterns):
            mem.observe(pid, 1.0, PH_EXPLOIT, 0.9)
        self.assertLessEqual(mem.usage_kb(), cfg.max_memory_kb)


if __name__ == "__main__":
    unittest.main()
