"""cousin.py -- cousin function (section 9).

get_cousins(candidate) generates structurally related candidates:
position mutation, relation mutation, permutation mutation, neighbor
swap, rotation, reversal, one-field mutation, phase-specific mutation.
Cousin count is hard-limited (cfg.max_cousins) and deduplicated by nonce;
downstream top-K filtering selects which cousins survive.
"""

import random

from config import (REL_UNCHANGED, REL_SWAP, REL_ROTATE, PERM_SWAP,
                    PERM_ADJACENT, PERM_PARTIAL)
from candidate import derive_nonce

__all__ = ["CousinEngine"]


class CousinEngine:
    def __init__(self, cfg, rng, perm_engine, rel_engine, new_candidate_fn):
        self.cfg = cfg
        self.rng = rng
        self.perm = perm_engine
        self.rel = rel_engine
        self._new = new_candidate_fn     # engine.new_candidate
        self.generated = 0
        self.emitted = 0

    # ------------------------------------------------------------------ ops
    def _mutate_position(self, c):
        pos = self.rng.randrange(len(c.state))
        if pos == c.position:
            pos = (pos + 1) % len(c.state)
        return self._new(c.state, pos, c.relation_id, c.permutation_id,
                         parent_id=c.id, age=c.age + 1)

    def _mutate_relation(self, c):
        rel = self.rel.pick(c.phase)
        return self._new(c.state, c.position, rel, c.permutation_id,
                         parent_id=c.id, age=c.age + 1)

    def _mutate_permutation(self, c):
        perm = self.rng.randrange(7)
        st = self.perm.apply(perm, c.state, self.rng) if hasattr(
            self.perm, "apply") else _apply_perm(perm, c.state, self.rng)
        return self._new(st, c.position, c.relation_id, perm,
                         parent_id=c.id, age=c.age + 1)

    def _swap_neighbor(self, c):
        st = list(c.state)
        p = c.position % len(st)
        j = (p + 1) % len(st)
        st[p], st[j] = st[j], st[p]
        return self._new(tuple(st), p, REL_SWAP, c.permutation_id,
                         parent_id=c.id, age=c.age + 1)

    def _rotate_state(self, c):
        st = c.state[1:] + c.state[:1]
        return self._new(st, c.position, REL_ROTATE, c.permutation_id,
                         parent_id=c.id, age=c.age + 1)

    def _reverse_state(self, c):
        return self._new(tuple(reversed(c.state)), c.position,
                         c.relation_id, c.permutation_id,
                         parent_id=c.id, age=c.age + 1)

    def _one_field(self, c):
        st = list(c.state)
        p = self.rng.randrange(len(st))
        st[p] = (st[p] + self.rng.randrange(1, max(2, self.cfg.dimensions[p]))
                 ) % self.cfg.dimensions[p]
        return self._new(tuple(st), p, c.relation_id, c.permutation_id,
                         parent_id=c.id, age=c.age + 1)

    def _phase_mutation(self, c):
        """Phase-specific: small local step around the current position."""
        st = list(c.state)
        n = len(st)
        p = c.position % n
        d = self.rng.choice((-1, 1))
        st[(p + d) % n] = (st[(p + d) % n] + d) % self.cfg.dimensions[(p + d) % n]
        return self._new(tuple(st), p, REL_UNCHANGED, PERM_ADJACENT if d > 0
                         else PERM_PARTIAL, parent_id=c.id, age=c.age + 1)

    # ----------------------------------------------------------------- main
    def get_cousins(self, cand, limit=None):
        """Up to `limit` unique cousins of `cand` (top-K applied upstream)."""
        limit = limit or self.cfg.max_cousins
        ops = (self._mutate_position, self._mutate_relation,
               self._mutate_permutation, self._swap_neighbor,
               self._rotate_state, self._reverse_state, self._one_field,
               self._phase_mutation)
        order = list(range(len(ops)))
        self.rng.shuffle(order)
        out = []
        seen = set()
        for i in order:
            if len(out) >= limit:
                break
            self.generated += 1
            c = ops[i](cand)
            if c.nonce in seen:
                continue
            seen.add(c.nonce)
            out.append(c)
        self.emitted += len(out)
        return out


def _apply_perm(perm_id, state, rng):
    """Local fallback used when only transform ids are available."""
    st = list(state)
    n = len(st)
    if n < 2:
        return tuple(st)
    if perm_id == PERM_SWAP:
        i, j = rng.sample(range(n), 2)
        st[i], st[j] = st[j], st[i]
    elif perm_id == PERM_ADJACENT:
        i = rng.randrange(n - 1)
        st[i], st[i + 1] = st[i + 1], st[i]
    elif perm_id == PERM_PARTIAL:
        i = rng.randrange(n - 1)
        j = rng.randrange(i + 1, n)
        st[i:j + 1] = st[i:j + 1][::-1]
    return tuple(st)
