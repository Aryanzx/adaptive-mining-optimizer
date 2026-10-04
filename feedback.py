"""feedback.py -- adaptive feedback loop + failure detection (27, 28).

Consumes per-loop results and adjusts:
  * batch size (hardware suggestion)
  * exploration rate (via phase manager)
  * memory pressure (forced prune on overflow)

Detects failure conditions: no improvement, memory overflow, pattern
instability, repeated candidate generation, search stagnation.
"""

from collections import deque
from config import PH_STAGNATION

__all__ = ["FeedbackEngine"]


class FeedbackEngine:
    def __init__(self, cfg, stats, phase_mgr, memory, hw):
        self.cfg = cfg
        self.stats = stats
        self.phase = phase_mgr
        self.memory = memory
        self.hw = hw
        self.batch_size = cfg.batch_size
        self.dup_rate = 0.0
        self.failures = deque(maxlen=64)
        self.failure_counts = {}
        self.loops_without_improvement = 0
        self.actions_log = deque(maxlen=64)

    def note_failure(self, code, loop):
        self.failures.append((loop, code))
        self.failure_counts[code] = self.failure_counts.get(code, 0) + 1

    def update(self, loop, successes, dups, generated):
        actions = []
        # -- hardware-driven batch adaptation
        new_batch = self.hw.suggest_batch(self.batch_size)
        if new_batch != self.batch_size:
            actions.append("batch:%d->%d" % (self.batch_size, new_batch))
            self.batch_size = new_batch
        # -- memory overflow
        if self.memory.usage_kb() > self.cfg.max_memory_kb:
            self.memory.prune(force=True)
            actions.append("memory_prune")
            self.note_failure("memory_overflow", loop)
        # -- repeated candidate generation
        self.dup_rate = dups / max(1, generated)
        if self.dup_rate > 0.10:
            self.note_failure("repeated_candidates", loop)
        # -- no improvement / stagnation
        if successes:
            self.loops_without_improvement = 0
        else:
            self.loops_without_improvement += 1
            if self.loops_without_improvement >= self.cfg.stagnation_window:
                self.note_failure("search_stagnation", loop)
        # -- pattern instability (confidence collapse of tracked patterns)
        if self.cfg.use_patterns and self.stats.patterns:
            unstable = sum(1 for ps in self.stats.patterns.values()
                           if ps.invalid)
            if unstable and unstable == len(self.stats.patterns):
                self.note_failure("pattern_instability", loop)
        # -- statistical insignificance watchdog
        if self.stats.attempts >= 10 * self.cfg.min_pattern_samples and \
                not self.stats.patterns:
            self.note_failure("statistical_insignificance", loop)
        for a in actions:
            self.actions_log.append((loop, a))
        return actions

    def phase_context(self):
        return self.phase.phase == PH_STAGNATION

    def summary(self):
        return {"failure_counts": dict(self.failure_counts),
                "batch_size": self.batch_size,
                "dup_rate": self.dup_rate}
