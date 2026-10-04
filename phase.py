"""phase.py -- phase-change state machine (sections 13, 15).

EXPLORE -> DISCOVER -> VALIDATE -> EXPLOIT -> STAGNATION -> RESET -> EXPLORE

EXPLOIT is never permanent: it decays either through stagnation or through
a hard dwell cap.  The RESET transition clears weak memory (done by the
memory manager, triggered by the engine) and boosts exploration.
"""

from collections import deque
from config import (PH_EXPLORE, PH_DISCOVER, PH_VALIDATE, PH_EXPLOIT,
                    PH_STAGNATION, PH_RESET)

__all__ = ["PhaseManager"]


class PhaseManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = PH_EXPLORE
        self.loops_in_phase = 0
        self.no_improve = 0
        self.total_improve = 0
        self.pending_patterns = 0
        self.validated_total = 0
        self.transition_count = 0
        self.history = deque(maxlen=64)

    def _to(self, new, loop):
        if new != self.phase:
            self.history.append((loop, self.phase, new))
            self.phase = new
            self.loops_in_phase = 0
            self.transition_count += 1

    def note_validated(self):
        self.validated_total += 1

    def note_loop(self, loop, improved, new_patterns, validated_now):
        self.loops_in_phase += 1
        if improved:
            self.no_improve = 0
            self.total_improve += 1
        else:
            self.no_improve += 1
        self.pending_patterns += new_patterns
        c = self.cfg
        p = self.phase
        if p == PH_EXPLORE:
            if self.pending_patterns >= c.discover_min_patterns:
                self.pending_patterns = 0
                self._to(PH_DISCOVER, loop)
            elif self.no_improve >= c.stagnation_window * 2:
                self._to(PH_STAGNATION, loop)
        elif p == PH_DISCOVER:
            if self.loops_in_phase >= 2:
                self.pending_patterns = 0
                self._to(PH_VALIDATE, loop)
        elif p == PH_VALIDATE:
            if self.loops_in_phase >= c.validate_loops:
                self._to(PH_EXPLOIT if validated_now else PH_EXPLORE, loop)
        elif p == PH_EXPLOIT:
            if self.no_improve >= c.stagnation_window or \
                    self.loops_in_phase >= c.exploit_max_dwell:
                self._to(PH_STAGNATION, loop)
        elif p == PH_STAGNATION:
            self._to(PH_RESET, loop)
        elif p == PH_RESET:
            self._to(PH_EXPLORE, loop)

    def exploration_rate(self, progress):
        """0..1 fraction of exploration for this loop (section 17)."""
        c = self.cfg
        progress = min(1.0, max(0.0, progress))
        base = c.explore_start + (c.explore_end - c.explore_start) * progress
        p = self.phase
        if p == PH_EXPLOIT:
            base *= 0.5
        elif p in (PH_STAGNATION, PH_RESET):
            base = max(base, c.explore_stagnation)
        elif p == PH_EXPLORE:
            base = max(base, c.explore_start * 0.9)
        return min(0.95, base)

    def phase_bias(self):
        """Memory selection bias for the current phase (section 15)."""
        p = self.phase
        if p == PH_EXPLORE:
            return "diversity"        # favor low-frequency entries
        if p == PH_DISCOVER:
            return "novelty"          # favor unusual patterns
        if p == PH_VALIDATE:
            return "testable"         # favor high-sample patterns
        if p == PH_EXPLOIT:
            return "performance"      # favor score*confidence
        return "forget"               # STAGNATION/RESET: drop weak entries
