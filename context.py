"""context.py -- multi-context matcher (section 10).

Seven independent contexts each produce a normalized [0,1] score:

  A position history      B relation history     C permutation history
  D performance history   E phase history        F hardware state
  G hash-result statistics

score = w1*pos + w2*rel + w3*perm + w4*perf + w5*phase + w6*hw + w7*hash

Weights are fixed here (configurable) and scores are clipped to [0,1].
NOTE (section 33): these scores rank *which candidates to try first*;
they carry no information about hash outputs beyond what the feedback
loop has measured, which the validator shows is nothing.
"""

from config import (PH_EXPLORE, PH_DISCOVER, PH_VALIDATE, PH_EXPLOIT,
                    PH_STAGNATION, PH_RESET)

__all__ = ["ContextMatcher"]

_PHASE_PRIOR = (0.60, 0.70, 0.60, 0.80, 0.30, 0.40)


class ContextMatcher:
    def __init__(self, cfg, engine):
        self.cfg = cfg
        self.e = engine
        # w1..w7, sum == 1.0
        self.weights = (0.18, 0.16, 0.16, 0.12, 0.12, 0.10, 0.16)

    def score(self, c):
        e = self.e
        s_pos = e.rel.position_novelty(c.position)
        s_rel = e.rel.relation_score(c.relation_id, e.phase.phase)
        s_perm = e.perm.permutation_score(c.permutation_id, e.phase.phase)
        s_perf = e.stats.performance_score()
        s_phase = _PHASE_PRIOR[e.phase.phase]
        s_hw = e.hw.score()
        s_hash = e.stats.hashstats_score()
        w = self.weights
        v = (w[0] * s_pos + w[1] * s_rel + w[2] * s_perm + w[3] * s_perf +
             w[4] * s_phase + w[5] * s_hw + w[6] * s_hash)
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        return v

    def breakdown(self, c):
        """Per-context scores for --debug/--statistics inspection."""
        e = self.e
        return {
            "position": e.rel.position_novelty(c.position),
            "relation": e.rel.relation_score(c.relation_id, e.phase.phase),
            "permutation": e.perm.permutation_score(c.permutation_id,
                                                    e.phase.phase),
            "performance": e.stats.performance_score(),
            "phase": _PHASE_PRIOR[e.phase.phase],
            "hardware": e.hw.score(),
            "hashstats": e.stats.hashstats_score(),
        }
