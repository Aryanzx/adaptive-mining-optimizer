"""pattern.py -- pattern detection + compact pattern table (sections 11, 23).

Streams of small integers (positions, relation ids, permutation ids,
phases, position deltas) are pushed through fixed-capacity ring buffers.
O(1) per push detection of:

    repetition            A A
    2-cycle               A B A
    3-cycle               A B C A
    arithmetic run        x, x+d, x+2d   (d != 0)
    bigram transition     x -> y

Every distinct pattern is interned once into PATTERN_TABLE and referenced
by an integer pattern_id afterwards (compression, section 23).  The value
domains of all streams are small and fixed, so the table is naturally
bounded; a hard cap is still enforced.
"""

from collections import deque
from config import (PS_POSITION, PS_RELATION, PS_PERMUTATION, PS_PHASE,
                    PS_DELTA)

__all__ = ["PatternEngine", "PatternTable",
           "R_REP", "R_CYCLE2", "R_CYCLE3", "R_RUN", "R_TRANS", "KIND_NAMES"]

R_REP, R_CYCLE2, R_CYCLE3, R_RUN, R_TRANS = range(5)
KIND_NAMES = ("rep", "cycle2", "cycle3", "run", "trans")

_NSTREAMS = 5


def _canon(t):
    """Canonical rotation so cycle keys compare equal under rotation."""
    if not t:
        return t
    m = min(t)
    i = t.index(m)
    return t[i:] + t[:i]


class PatternTable:
    """pattern key (tuple of ints) <-> integer pattern_id (section 23)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.key_to_id = {}
        self.keys = []
        self.freq = []
        self.hard_cap = 4096

    def intern(self, key):
        """Return (pattern_id, is_new) or (None, False) if table is full."""
        pid = self.key_to_id.get(key)
        if pid is not None:
            self.freq[pid] += 1
            return pid, False
        if len(self.keys) >= self.hard_cap:
            return None, False
        pid = len(self.keys)
        self.key_to_id[key] = pid
        self.keys.append(key)
        self.freq.append(1)
        return pid, True

    def key(self, pid):
        return self.keys[pid]

    def __len__(self):
        return len(self.keys)


class PatternEngine:
    def __init__(self, cfg, table):
        self.cfg = cfg
        self.table = table
        self.bufs = [deque(maxlen=cfg.history_len) for _ in range(_NSTREAMS)]
        self.detections = 0

    def push(self, stream, value):
        """Push one event; return list of newly detected pattern keys."""
        if not (0 <= stream < _NSTREAMS):
            raise ValueError("bad stream %r" % stream)
        buf = self.bufs[stream]
        buf.append(value)
        keys = []
        n = len(buf)
        a, b = buf[-1], buf[-2] if n >= 2 else None
        if n >= 2 and a == b:
            keys.append((stream, R_REP, a))
        if n >= 3 and buf[-1] == buf[-3] and buf[-2] != buf[-1]:
            keys.append((stream, R_CYCLE2, _canon((buf[-2], buf[-1]))))
        if n >= 4 and buf[-1] == buf[-4] and buf[-1] != buf[-2] \
                and buf[-2] != buf[-3]:
            keys.append((stream, R_CYCLE3, _canon((buf[-3], buf[-2], buf[-1]))))
        if n >= 3:
            d1 = buf[-1] - buf[-2]
            d2 = buf[-2] - buf[-3]
            if d1 == d2 and d1 != 0:
                keys.append((stream, R_RUN, d1))
        if n >= 2 and a != b:
            keys.append((stream, R_TRANS, b, a))
        self.detections += len(keys)
        return keys

    def recent(self, stream, k=4):
        buf = self.bufs[stream]
        return tuple(list(buf)[-k:])
