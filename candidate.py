"""candidate.py -- compact integer candidate + symbolic search-space model.

A Candidate is a __slots__ object with integer fields only (section 4).
The multidimensional space is represented symbolically: states are decoded
from an integer rank on demand, never enumerated physically (section 5).
"""

__all__ = ["Candidate", "SearchSpace", "derive_nonce"]

_M64 = (1 << 64) - 1
_GOLD = 0x9E3779B97F4A7C15


class Candidate:
    """Compact candidate record.  Integer IDs only; no strings stored."""
    __slots__ = ("id", "parent_id", "position", "relation_id",
                 "permutation_id", "phase", "score", "confidence",
                 "age", "nonce", "state")

    def __init__(self, cid, parent_id, position, relation_id, permutation_id,
                 phase, state, score=0.0, confidence=0.0, age=0, nonce=0):
        self.id = cid
        self.parent_id = parent_id
        self.position = position
        self.relation_id = relation_id
        self.permutation_id = permutation_id
        self.phase = phase
        self.score = score
        self.confidence = confidence
        self.age = age
        self.nonce = nonce
        self.state = state

    def encode(self):
        """Delta/compression-friendly tuple for logs and memory."""
        return (self.id, self.parent_id, self.position, self.relation_id,
                self.permutation_id, self.phase, self.nonce)

    def __repr__(self):  # debug only; not used in hot paths
        return ("C(id=%d,par=%d,pos=%d,rel=%d,perm=%d,ph=%d,score=%.3f,"
                "nonce=%08x)" % (self.id, self.parent_id, self.position,
                                 self.relation_id, self.permutation_id,
                                 self.phase, self.score, self.nonce))


def _mix(x):
    """SplitMix64 finalizer -- cheap deterministic integer mixer."""
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def derive_nonce(state, position, relation_id, permutation_id, salt=0):
    """Deterministic map: candidate fields -> 32-bit nonce.

    This is the symbolic bridge between the multidimensional candidate
    space and the flat nonce space the hash worker operates on.  It is
    cheap integer mixing only -- the hash function itself is untouched.
    """
    h = (_GOLD ^ (salt * 0xA0761D6478BD642F)
         ^ (position << 12) ^ (relation_id << 20) ^ (permutation_id << 26))
    h &= _M64
    for i, v in enumerate(state):
        h = _mix(h ^ ((v + 1) * (i + 0x9E37))) & _M64
    return _mix(h) & 0xFFFFFFFF


class SearchSpace:
    """Symbolic Cartesian-product model with lazy iteration (section 5)."""

    def __init__(self, dimensions):
        self.dims = tuple(dimensions)
        self.n = len(self.dims)
        total = 1
        for d in self.dims:
            total *= d
        self.total = total

    def rank_to_state(self, rank):
        """Mixed-radix decode: rank -> state tuple.  Lazy, O(n)."""
        state = [0] * self.n
        for i in range(self.n - 1, -1, -1):
            d = self.dims[i]
            state[i] = rank % d
            rank //= d
        return tuple(state)

    def state_to_rank(self, state):
        rank = 0
        for i in range(self.n):
            rank = rank * self.dims[i] + (state[i] % self.dims[i])
        return rank

    def next_candidate(self, start=0):
        """Lazy generator: yields (rank, state) without storing anything."""
        rank = start % self.total
        state = list(self.rank_to_state(rank))
        while True:
            yield rank, tuple(state)
            rank = (rank + 1) % self.total
            for i in range(self.n - 1, -1, -1):
                state[i] += 1
                if state[i] < self.dims[i]:
                    break
                state[i] = 0
