"""topk.py -- top-K candidate manager (section 16).

Fixed-capacity max-selection via a bounded min-heap.  Generated candidates
are far more numerous than K; only the K best (plus exploration fills)
proceed to hashing.  K is configurable (cfg.top_k).
"""

import heapq
import itertools

__all__ = ["TopK"]


class TopK:
    def __init__(self, k):
        self.k = max(1, int(k))
        self._heap = []
        self._seq = itertools.count()
        self.overflows = 0

    def add(self, score, item):
        seq = next(self._seq)
        entry = (score, seq, item)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)
            self.overflows += 1

    def items(self):
        return sorted(((s, it) for s, _, it in self._heap), reverse=True)

    def scores(self):
        return sorted((s for s, _, _ in self._heap), reverse=True)

    def worst_kept(self):
        return self._heap[0][0] if self._heap else None

    def __len__(self):
        return len(self._heap)

    def clear(self):
        self._heap = []
