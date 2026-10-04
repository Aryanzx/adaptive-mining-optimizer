import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from confidence import compute_confidence, band_label
from statistics_engine import StatisticsEngine
from config import Config


class TestConfidence(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(min_pattern_samples=100)

    def test_monotonic_in_samples(self):
        a = compute_confidence(10, 1.0, 1.5, 0.0, 0, self.cfg)
        b = compute_confidence(1000, 1.0, 1.5, 0.0, 0, self.cfg)
        self.assertGreater(b, a)

    def test_bands(self):
        self.assertEqual(band_label(0.05), "useless")
        self.assertEqual(band_label(0.30), "weak")
        self.assertEqual(band_label(0.60), "experimental")
        self.assertEqual(band_label(0.80), "strong")
        self.assertEqual(band_label(0.95), "highly_validated")

    def test_zero_effect_confined_low(self):
        c = compute_confidence(10_000, 1.0, 1.0, 0.0, 0, self.cfg)
        self.assertLessEqual(c, 0.20)      # no improvement -> useless band

    def test_not_probability_of_solution(self):
        # confidence is bounded heuristic evidence, not 2^-difficulty
        c = compute_confidence(10_000, 1.0, 3.0, 0.0, 0, self.cfg)
        self.assertLessEqual(c, 1.0)


class TestStats(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(difficulty=8)
        self.st = StatisticsEngine(self.cfg)

    def test_baseline_rate(self):
        self.st.note_attempt_batch(256, 1, (), False, 0)
        self.assertAlmostEqual(self.st.baseline_rate(), 1 / 256)
        self.assertAlmostEqual(self.st.expected, 1 / 256)

    def test_pattern_effect_and_z(self):
        st = self.st
        # 4 batches WITHOUT the pattern at expected luck, then 4 batches
        # WITH the pattern at 3x luck (global baseline != conditional).
        for _ in range(4):
            st.note_attempt_batch(256, 1, (), False, 0)
        for _ in range(4):
            st.note_attempt_batch(256, 3, (5,), False, 1)
        eff = st.pattern_effect(5)
        self.assertGreaterEqual(eff, 1.5)  # 3x luck vs blended baseline ~2x
        z = st.pattern_z(5)
        self.assertGreater(z, 0)
        # train/val split
        self.assertEqual(st.patterns[5].train_samples, 4 * 256)

    def test_holdout_streams(self):
        st = self.st
        st.note_attempt_batch(256, 1, (7,), validation=False, loop_index=1)
        st.note_attempt_batch(256, 1, (7,), validation=True, loop_index=3)
        ps = st.patterns[7]
        self.assertEqual(ps.train_samples, 256)
        self.assertEqual(ps.val_samples, 256)

    def test_first_success(self):
        st = self.st
        st.note_attempt_batch(10, 0, (), False, 0)
        st.note_attempt_batch(10, 1, (), False, 1)
        self.assertEqual(st.first_success_attempt, 11)


if __name__ == "__main__":
    unittest.main()
