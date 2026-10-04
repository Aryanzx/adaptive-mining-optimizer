"""combination.py -- combination engine (section 7).

C(n, r) is never materialised.  The combinatorial number system un-ranks a
combination directly from its lexicographic index, giving O(r) generation
of any single combination and lazy bounded streams.
"""

from math import comb

__all__ = ["ncr", "nth_combination", "iter_combinations", "CombinationEngine"]


def ncr(n, r):
    return comb(n, r) if 0 <= r <= n else 0


def nth_combination(n, r, idx):
    """Un-rank: return the idx-th (0-based, lexicographic) combination."""
    if not 0 <= idx < ncr(n, r):
        raise IndexError("combination index out of range")
    out = []
    a = 0
    for k in range(r):
        while True:
            c = comb(n - a - 1, r - k - 1)
            if idx < c:
                break
            idx -= c
            a += 1
        out.append(a)
        a += 1
    return tuple(out)


def iter_combinations(n, r, max_count, rng):
    """Lazy bounded stream of r-combinations of range(n).

    If C(n,r) fits under max_count the full set is streamed in
    lexicographic order; otherwise indices are sampled without
    replacement via un-ranking.  Nothing is ever stored wholesale.
    """
    total = ncr(n, r)
    if total <= 0:
        return
    if total <= max_count:
        for idx in range(total):
            yield nth_combination(n, r, idx)
    else:
        seen = set()
        emitted = 0
        while emitted < max_count:
            idx = rng.randrange(total)
            if idx in seen:
                continue
            seen.add(idx)
            emitted += 1
            yield nth_combination(n, r, idx)


class CombinationEngine:
    """Bounded / weighted / sampled / adaptive combination access."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.calls = 0

    def bounded(self, n, r, max_count):
        self.calls += 1
        return iter_combinations(n, r, max_count, self.rng)

    def adaptive_r(self, n, phase):
        """Phase-dependent subset size: explore wider, exploit narrow."""
        if phase == 3:            # PH_EXPLOIT
            return 2
        return max(2, min(n - 1, n // 3))

    def sample_indices(self, n, phase, max_count=8):
        """Sample one bounded batch of coordinate-index combinations."""
        r = self.adaptive_r(n, phase)
        return self.bounded(n, r, max_count)

    def weighted_sample(self, items, r, weights):
        """Sample r distinct items with replacement-free weighting."""
        pool = list(zip(items, weights))
        out = []
        for _ in range(min(r, len(pool))):
            total = sum(w for _, w in pool)
            x = self.rng.random() * total
            acc = 0.0
            for i, (it, w) in enumerate(pool):
                acc += w
                if x < acc:
                    out.append(it)
                    pool.pop(i)
                    break
            else:
                out.append(pool.pop()[0])
        return out
