"""statistics_engine.py -- hash/result statistics + pattern evidence (18, 12).

Records hash_rate, attempts, successes, difficulty and per-pattern
conditional statistics.  Every hash attempt is attributed to the currently
active pattern set, split into a TRAIN stream and a VALIDATION (holdout)
stream so the validator can detect overfitting (section 26).
"""

from collections import deque
from math import sqrt

__all__ = ["StatisticsEngine", "PatternStats"]


class PatternStats:
    __slots__ = ("pid", "samples", "successes", "train_samples",
                 "train_successes", "val_samples", "val_successes",
                 "recent", "invalid", "validated", "confidence", "last_loop")

    def __init__(self, pid):
        self.pid = pid
        self.samples = 0
        self.successes = 0
        self.train_samples = 0
        self.train_successes = 0
        self.val_samples = 0
        self.val_successes = 0
        self.recent = deque(maxlen=32)   # per-loop indicator 0/1
        self.invalid = False
        self.validated = False
        self.confidence = 0.0
        self.last_loop = 0

    @property
    def conditional_rate(self):
        return self.successes / self.samples if self.samples else 0.0


class StatisticsEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.attempts = 0
        self.successes = 0
        self.hash_time = 0.0
        self.khs_ema = 0.0
        self.expected = 2.0 ** (-cfg.difficulty)
        self.rate_ema = self.expected
        self.patterns = {}
        self.first_success_attempt = None
        self.success_log = deque(maxlen=64)   # bounded (section 29)

    # ------------------------------------------------------------- hashing
    def note_hash_time(self, n_hashes, dt):
        self.hash_time += dt
        if dt > 0:
            khs = (n_hashes / dt) / 1000.0
            self.khs_ema = (self.khs_ema * 0.8 + khs * 0.2) if self.khs_ema \
                else khs

    def note_attempt_batch(self, n, successes, active_pids, validation,
                           loop_index):
        """Attribute one hashed batch to global + active-pattern stats."""
        base = self.attempts
        self.attempts += n
        self.successes += successes
        if successes and self.first_success_attempt is None:
            self.first_success_attempt = base + 1
        rate = successes / n if n else 0.0
        self.rate_ema = self.rate_ema * 0.95 + rate * 0.05
        for pid in active_pids:
            ps = self.get(pid)
            ps.samples += n
            ps.successes += successes
            ps.last_loop = loop_index
            if validation:
                ps.val_samples += n
                ps.val_successes += successes
            else:
                ps.train_samples += n
                ps.train_successes += successes
            ps.recent.append(1 if rate >= self.expected else 0)

    def register_success(self, nonce, loop_index, phase, n_active):
        self.success_log.append((nonce, loop_index, phase, n_active))

    def get(self, pid):
        ps = self.patterns.get(pid)
        if ps is None:
            ps = PatternStats(pid)
            self.patterns[pid] = ps
            if len(self.patterns) > self.cfg.max_active_patterns * 2:
                self._evict_weakest()
        return ps

    def _evict_weakest(self):
        victims = [(ps.samples, pid) for pid, ps in self.patterns.items()
                   if not ps.validated and not ps.invalid]
        if not victims:
            return
        victims.sort()
        del self.patterns[victims[0][1]]

    # ------------------------------------------------------------- scores
    def baseline_rate(self):
        return self.successes / self.attempts if self.attempts else 0.0

    def performance_score(self):
        """Context D: recent batch success rate vs expectation, 0.5 neutral."""
        d = (self.rate_ema - self.expected) / (self.expected * 4.0)
        return min(1.0, max(0.0, 0.5 + d))

    def hashstats_score(self):
        """Context G: lifetime success rate vs expectation, 0.5 neutral."""
        r = self.baseline_rate()
        if not self.attempts:
            return 0.5
        d = (r - self.expected) / (self.expected * 4.0)
        return min(1.0, max(0.0, 0.5 + d))

    # ------------------------------------------------- pattern inference
    def pattern_effect(self, pid):
        """effect_size = P(success | pattern) / P(success)."""
        ps = self.patterns.get(pid)
        base = self.baseline_rate()
        if ps is None or not ps.samples or base <= 0.0:
            return 0.0
        return ps.conditional_rate / base

    def pattern_effect_stream(self, pid, validation):
        ps = self.patterns.get(pid)
        if ps is None:
            return 0.0
        base = self.baseline_rate()
        if validation:
            n, s = ps.val_samples, ps.val_successes
        else:
            n, s = ps.train_samples, ps.train_successes
        if not n or base <= 0.0:
            return 0.0
        return (s / n) / base

    def pattern_z(self, pid):
        """One-proportion z score of conditional vs baseline rate."""
        ps = self.patterns.get(pid)
        if ps is None or not ps.samples or not self.attempts:
            return 0.0
        p = self.baseline_rate()
        pc = ps.conditional_rate
        se = sqrt(p * (1.0 - p) / ps.samples)
        if se <= 0.0:
            return 0.0
        return (pc - p) / se

    def consistency(self, pid):
        ps = self.patterns.get(pid)
        if ps is None or not ps.recent:
            return 0.5
        return sum(ps.recent) / len(ps.recent)

    def variance(self, pid):
        c = self.consistency(pid)
        return c * (1.0 - c)

    def age(self, pid, loop_index):
        ps = self.patterns.get(pid)
        return 0 if ps is None else max(0, loop_index - ps.last_loop)

    def summary(self):
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "baseline_rate": self.baseline_rate(),
            "expected_rate": self.expected,
            "khs": self.khs_ema,
            "hash_time_s": self.hash_time,
            "first_success_at": self.first_success_attempt,
            "patterns_tracked": len(self.patterns),
        }
