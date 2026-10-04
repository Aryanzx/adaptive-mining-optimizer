"""memory.py -- compact 4-tier memory with aging (sections 14, 15, 22, 23).

Tiers: HOT (useful now) / ACTIVE (moderately useful) / COLD (rarely
useful) / DISCARD (deleted, counted only).  Entries age:

    new_score = old_score * decay + new_evidence

and are deleted below cfg.discard_score.  Hard caps enforce the embedded
memory budget.  Phase-dependent selection biases prevent trapping
(section 15).
"""

from config import MEM_HOT, MEM_ACTIVE, MEM_COLD

__all__ = ["MemoryManager", "MemoryEntry"]


class MemoryEntry:
    __slots__ = ("pattern_id", "score", "confidence", "frequency",
                 "last_used", "phase", "tier")

    def __init__(self, pattern_id, phase):
        self.pattern_id = pattern_id
        self.score = 0.0
        self.confidence = 0.0
        self.frequency = 0
        self.last_used = 0
        self.phase = phase
        self.tier = MEM_COLD


class MemoryManager:
    CAPS = {MEM_HOT: 0, MEM_ACTIVE: 0, MEM_COLD: 0}   # filled from cfg

    def __init__(self, cfg):
        self.cfg = cfg
        self.tiers = {MEM_HOT: {}, MEM_ACTIVE: {}, MEM_COLD: {}}
        self.CAPS = {MEM_HOT: cfg.max_hot_candidates,
                     MEM_ACTIVE: cfg.max_active_patterns,
                     MEM_COLD: cfg.max_active_patterns * 2}
        self.discarded = 0
        self.observations = 0
        self.loop_clock = 0

    # ------------------------------------------------------------- observe
    def observe(self, pattern_id, evidence, phase, confidence):
        self.observations += 1
        e = self._find(pattern_id)
        if e is None:
            if len(self.tiers[MEM_COLD]) >= self.CAPS[MEM_COLD]:
                self._evict_weakest(MEM_COLD)
            e = MemoryEntry(pattern_id, phase)
            self.tiers[MEM_COLD][pattern_id] = e
        e.score = e.score * self.cfg.decay + evidence
        e.frequency += 1
        e.last_used = self.loop_clock
        e.confidence = confidence
        e.phase = phase
        self._rebalance(pattern_id, e)

    def _find(self, pid):
        for t in (MEM_HOT, MEM_ACTIVE, MEM_COLD):
            e = self.tiers[t].get(pid)
            if e is not None:
                return e
        return None

    def get(self, pid):
        return self._find(pid)

    def _rebalance(self, pid, e):
        if e.score < self.cfg.discard_score:
            self._remove(pid, e)
            return
        if e.tier != MEM_HOT and e.score >= self.cfg.active_score * 2 and \
                e.confidence >= 0.75:
            self._move(pid, e, MEM_HOT)
        elif e.tier == MEM_COLD and e.score >= self.cfg.active_score:
            self._move(pid, e, MEM_ACTIVE)
        elif e.tier == MEM_ACTIVE and e.score < self.cfg.active_score:
            self._move(pid, e, MEM_COLD)

    def _move(self, pid, e, tier):
        del self.tiers[e.tier][pid]
        cap = self.CAPS[tier]
        if len(self.tiers[tier]) >= cap:
            self._evict_weakest(tier)
        e.tier = tier
        self.tiers[tier][pid] = e

    def _remove(self, pid, e):
        del self.tiers[e.tier][pid]
        self.discarded += 1                     # -> DISCARD tier

    def _evict_weakest(self, tier):
        if not self.tiers[tier]:
            return
        pid = min(self.tiers[tier],
                  key=lambda p: (self.tiers[tier][p].score,
                                 self.tiers[tier][p].confidence))
        self._remove(pid, self.tiers[tier][pid])

    # --------------------------------------------------------------- prune
    def tick(self, loop):
        self.loop_clock = loop

    def prune(self, phase=None, force=False):
        """Age everything one tick; extra forgetting during stagnation."""
        extra = 0.5 if phase in (4, 5) else 1.0   # PH_STAGNATION/PH_RESET
        for tier in (MEM_HOT, MEM_ACTIVE, MEM_COLD):
            dead = []
            for pid, e in self.tiers[tier].items():
                e.score *= self.cfg.decay * extra
                if force or e.score < self.cfg.discard_score:
                    dead.append(pid)
            for pid in dead:
                self._remove(pid, self.tiers[tier][pid])
        # enforce caps bottom-up
        for tier in (MEM_HOT, MEM_ACTIVE, MEM_COLD):
            while len(self.tiers[tier]) > self.CAPS[tier]:
                self._evict_weakest(tier)

    # ------------------------------------------------------ phase ranking
    def ranked(self, phase, limit, phase_name="diversity"):
        """Return up to `limit` pattern_ids ranked by phase bias (sec 15)."""
        entries = []
        for tier in (MEM_HOT, MEM_ACTIVE, MEM_COLD):
            for pid, e in self.tiers[tier].items():
                if phase_name == "diversity":
                    k = 1.0 / (1.0 + e.frequency) + e.score
                elif phase_name == "novelty":
                    k = 2.0 / (1.0 + e.frequency) + e.score
                elif phase_name == "testable":
                    k = e.frequency + e.confidence
                elif phase_name == "performance":
                    k = e.score * max(0.1, e.confidence)
                else:                      # forget: keep only strong
                    k = e.score if e.confidence >= 0.75 else -e.score
                entries.append((k, pid))
        entries.sort(reverse=True)
        return [pid for _, pid in entries[:limit]]

    # ------------------------------------------------------------- metrics
    def counts(self):
        return {t: len(self.tiers[t]) for t in self.tiers}

    def usage_kb(self, extra_bytes=0):
        n = sum(len(t) for t in self.tiers.values())
        entry_bytes = n * 160                    # slot object + dict slot est.
        return (entry_bytes + extra_bytes + 4096) // 1024

    def stats(self):
        return {"counts": self.counts(), "discarded": self.discarded,
                "observations": self.observations,
                "usage_kb": self.usage_kb()}
