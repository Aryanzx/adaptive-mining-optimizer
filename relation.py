"""relation.py -- position/relation engine (sections 4, 8).

Relations are integer IDs (0..8) applied to a state vector at a position.
Position deltas are tracked in a bounded ring and classified into compact
pattern classes.  Relations are search heuristics only -- they are never
assumed to predict hash outputs (section 33).
"""

from collections import deque
from config import (REL_UNCHANGED, REL_INC, REL_DEC, REL_SWAP, REL_REVERSE,
                    REL_ROTATE, REL_NEIGHBOR, REL_COUSIN, REL_SHIFT,
                    PH_EXPLOIT)

__all__ = ["RelationEngine", "DELTA_NAMES", "DELTA_TO_REL",
           "D_REPEAT", "D_P1", "D_M1", "D_P2", "D_M2", "D_P3", "D_M3", "D_JUMP"]

# compact delta pattern classes (section 8)
D_REPEAT, D_P1, D_M1, D_P2, D_M2, D_P3, D_M3, D_JUMP = range(8)
DELTA_NAMES = ("repeat", "+1", "-1", "+2", "-2", "+3", "-3", "jump")

DELTA_TO_REL = {
    D_REPEAT: REL_UNCHANGED,
    D_P1: REL_INC,
    D_M1: REL_DEC,
    D_P2: REL_SHIFT,
    D_M2: REL_SHIFT,
    D_P3: REL_NEIGHBOR,
    D_M3: REL_NEIGHBOR,
    D_JUMP: REL_COUSIN,
}

_PICKABLE = (REL_UNCHANGED, REL_INC, REL_DEC, REL_SWAP, REL_REVERSE,
             REL_ROTATE, REL_NEIGHBOR, REL_SHIFT)


def classify_delta(d):
    if d == 0:
        return D_REPEAT
    if d == 1:
        return D_P1
    if d == -1:
        return D_M1
    if d == 2:
        return D_P2
    if d == -2:
        return D_M2
    if d == 3:
        return D_P3
    if d == -3:
        return D_M3
    return D_JUMP


class RelationEngine:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        nrel = 9
        self.quality = [0.0] * nrel        # EMA of loop evidence
        self.frequency = [0] * nrel
        self.prev_position = -1
        self.deltas = deque(maxlen=cfg.history_len)
        self.position_freq = [0] * len(cfg.dimensions)
        self._recent_rel = deque(maxlen=cfg.history_len)

    # ------------------------------------------------------------ transform
    def apply(self, rel_id, state, position):
        st = list(state)
        n = len(st)
        dim = self.cfg.dimensions[position % n]
        p = position % n
        if rel_id == REL_UNCHANGED:
            pass
        elif rel_id == REL_INC:
            st[p] = (st[p] + 1) % dim
        elif rel_id == REL_DEC:
            st[p] = (st[p] - 1) % dim
        elif rel_id == REL_SWAP:
            j = (p + 1) % n
            st[p], st[j] = st[j], st[p]
        elif rel_id == REL_REVERSE:
            st = st[:p + 1][::-1] + st[p + 1:]
        elif rel_id == REL_ROTATE:
            st = st[p:] + st[:p]
        elif rel_id == REL_NEIGHBOR:
            st[p] = (st[p] + self.rng.choice((-1, 1))) % dim
        elif rel_id == REL_COUSIN:
            st[p] = (st[p] + self.rng.randrange(dim)) % dim
        elif rel_id == REL_SHIFT:
            d = self.rng.choice((2, -2, 3))
            q = (p + d) % n
            st[q] = (st[q] + 1) % self.cfg.dimensions[q]
        else:
            raise ValueError("bad relation_id %r" % rel_id)
        return tuple(st)

    # ------------------------------------------------------------- scoring
    def relation_score(self, rel_id, phase):
        q = self.quality[rel_id]
        novelty = 1.0 / (1.0 + self.frequency[rel_id])
        s = q if phase == PH_EXPLOIT else 0.6 * q + 0.4 * novelty
        return min(1.0, max(0.0, s))

    def pick(self, phase, bias=None):
        w = []
        for r in _PICKABLE:
            s = self.relation_score(r, phase)
            if bias is not None:
                s += min(0.5, max(0.0, bias[r]))
            w.append(0.05 + s)
        total = sum(w)
        x = self.rng.random() * total
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if x < acc:
                return _PICKABLE[i]
        return _PICKABLE[-1]

    def record(self, rel_id, evidence):
        self.frequency[rel_id] += 1
        self.quality[rel_id] = self.quality[rel_id] * 0.9 + evidence * 0.1
        self._recent_rel.append(rel_id)

    # -------------------------------------------------- position relations
    def observe_position(self, pos):
        if self.prev_position >= 0:
            self.deltas.append(pos - self.prev_position)
        self.prev_position = pos
        self.position_freq[pos % len(self.position_freq)] += 1

    def position_novelty(self, pos):
        return 1.0 / (1.0 + self.position_freq[pos % len(self.position_freq)])

    def last_delta_class(self):
        if not self.deltas:
            return D_REPEAT
        return classify_delta(self.deltas[-1])

    def recent_deltas(self):
        return tuple(self.deltas)
