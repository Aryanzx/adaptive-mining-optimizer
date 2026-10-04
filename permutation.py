"""permutation.py -- permutation engine (section 6).

Supports swap / rotate / reverse / cyclic shift / adjacent swap / block
permutation / partial permutation on small integer tuples.  Transform
choice is scored (permutation_score) from decayed quality + novelty, so
transformations that never precede productive loops fade out.  No claim is
made that permutations predict hash outputs -- they are search heuristics
validated experimentally (section 33).
"""

from collections import deque
from config import (PERM_SWAP, PERM_ROTATE, PERM_REVERSE, PERM_CYCLIC,
                    PERM_ADJACENT, PERM_BLOCK, PERM_PARTIAL, PH_EXPLOIT)

__all__ = ["PermutationEngine", "apply_transform"]

_NPERM = 7


def apply_transform(perm_id, seq, rng):
    """Apply one permutation transform to a sequence of ints."""
    seq = list(seq)
    n = len(seq)
    if n < 2:
        return tuple(seq)
    if perm_id == PERM_SWAP:
        i, j = rng.sample(range(n), 2)
        seq[i], seq[j] = seq[j], seq[i]
    elif perm_id == PERM_ROTATE:
        k = rng.randrange(n)
        seq = seq[k:] + seq[:k]
    elif perm_id == PERM_REVERSE:
        seq.reverse()
    elif perm_id == PERM_CYCLIC:
        seq = seq[1:] + seq[:1]
    elif perm_id == PERM_ADJACENT:
        i = rng.randrange(n - 1)
        seq[i], seq[i + 1] = seq[i + 1], seq[i]
    elif perm_id == PERM_BLOCK:
        b = max(2, n // 3)
        s = rng.randrange(0, n)
        blk = seq[s:s + b]
        seq[s:s + b] = blk[::-1]
    elif perm_id == PERM_PARTIAL:
        i = rng.randrange(n - 1)
        j = rng.randrange(i + 1, n)
        seq[i:j + 1] = seq[i:j + 1][::-1]
    else:
        raise ValueError("bad perm_id %r" % perm_id)
    return tuple(seq)


class PermutationEngine:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.attempts = [0] * _NPERM
        self.frequency = [0] * _NPERM
        self.quality = [0.0] * _NPERM          # EMA of loop evidence
        self.prev_perm = -1
        self.cur_perm = -1
        self._recent = deque(maxlen=cfg.history_len)

    def pick(self, phase, bias=None):
        """Weighted choice of a transform id; bias[] adds memory evidence."""
        w = []
        for p in range(_NPERM):
            s = self.permutation_score(p, phase)
            if bias is not None:
                s += min(0.5, max(0.0, bias[p]))
            w.append(0.05 + s)
        return self._weighted(w)

    def permutation_score(self, perm_id, phase):
        base = self.quality[perm_id]
        novelty = 1.0 / (1.0 + self.frequency[perm_id])
        if phase == PH_EXPLOIT:
            s = base
        else:
            s = 0.6 * base + 0.4 * novelty
        return min(1.0, max(0.0, s))

    def record(self, perm_id, evidence):
        """evidence in {0,1}: did a loop using this transform hit a target?"""
        self.attempts[perm_id] += 1
        self.frequency[perm_id] += 1
        self.quality[perm_id] = self.quality[perm_id] * 0.9 + evidence * 0.1
        self.prev_perm = self.cur_perm
        self.cur_perm = perm_id
        self._recent.append(perm_id)

    def recent(self):
        return tuple(self._recent)

    def _weighted(self, w):
        total = sum(w)
        x = self.rng.random() * total
        acc = 0.0
        for i, wi in enumerate(w):
            acc += wi
            if x < acc:
                return i
        return len(w) - 1
