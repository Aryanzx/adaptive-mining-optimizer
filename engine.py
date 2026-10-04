"""engine.py -- AdaptiveEngine: the main adaptive feedback loop (section 27).

while running:
    candidates      = generate_candidates()
    contexts        = analyze_contexts(candidates)
    cousins         = generate_cousins(candidates)
    patterns        = detect_patterns(contexts)
    candidates      = rank_candidates(candidates, patterns)
    top_candidates  = select_top_k(candidates)
    results         = hash_candidates(top_candidates)
    statistics      = analyze_results(results)
    update_memory(statistics)
    update_phase(statistics)
    optimize_parameters(statistics)

Everything is driven by one seeded RNG -> fully reproducible runs.  The
hash worker is called with plain 32-bit nonces; the SHA-256 primitive is
never touched (section 18/33).
"""

import itertools
import time
from collections import deque

from config import *          # ID tables + Config (stdlib-only project)
from candidate import SearchSpace, derive_nonce, Candidate
from permutation import PermutationEngine
from combination import CombinationEngine
from relation import RelationEngine, classify_delta, DELTA_TO_REL
from cousin import CousinEngine
from pattern import PatternEngine, PatternTable
from context import ContextMatcher
from statistics_engine import StatisticsEngine
from confidence import compute_confidence
from phase import PhaseManager
from memory import MemoryManager
from topk import TopK
from hash_worker import HashWorker
from hardware import HardwareMonitor
from feedback import FeedbackEngine

__all__ = ["AdaptiveEngine", "Logger"]


class Logger:
    """Compact logger: TIME | PHASE | CANDIDATE | PATTERN | SCORE |
    HASHRATE | RESULT (section 29)."""

    def __init__(self, debug=False, statistics=False, memory=False,
                 profile=False, enabled=True):
        self.debug = debug
        self.statistics = statistics
        self.memory = memory
        self.profile = profile
        self.enabled = enabled
        self.t0 = time.perf_counter()

    def row(self, loop, phase_name, n_cand, n_pat, score, khs, result):
        if not self.enabled:
            return
        print("t=%7.2fs | %s | cand=%7d | pat=%4d | score=%.3f | "
              "kh/s=%7.1f | %s"
              % (time.perf_counter() - self.t0, phase_name, n_cand, n_pat,
                 score, khs, result), flush=True)


class AdaptiveEngine:
    def __init__(self, cfg, seed=None, log=None):
        self.cfg = cfg
        self.seed = int(cfg.seed if seed is None else seed) & 0xFFFFFFFF
        self.rng = random_Random(self.seed)
        self.log = log or Logger(enabled=False)

        # module stack (architecture, section 3)
        self.space = SearchSpace(cfg.dimensions)
        self.perm = PermutationEngine(cfg, self.rng)
        self.comb = CombinationEngine(cfg, self.rng)
        self.rel = RelationEngine(cfg, self.rng)
        self.ptable = PatternTable(cfg)
        self.pattern = PatternEngine(cfg, self.ptable)
        self.stats = StatisticsEngine(cfg)
        self.phase = PhaseManager(cfg)
        self.memory = MemoryManager(cfg)
        self.topk = TopK(cfg.top_k)
        self.worker = HashWorker(cfg, self.stats)
        self.hw = HardwareMonitor(cfg)
        self.feedback = FeedbackEngine(cfg, self.stats, self.phase,
                                       self.memory, self.hw)
        self.context = ContextMatcher(cfg, self)
        self.cousin = CousinEngine(cfg, self.rng, self.perm, self.rel,
                                   self.new_candidate)

        # loop state
        self._cid = itertools.count(1)
        self.loop_index = 0
        self.elites = ()
        self.active_patterns = ()
        self.seen = set()                       # bounded nonce-dedup window
        self._seen_q = deque(maxlen=cfg.dedup_window)
        self.detected_last = ()
        self.discovered = 0
        self.validated = 0
        self.rejected = 0
        self.dup_skipped = 0
        self.bias_rel = [0.0] * 9
        self.bias_perm = [0.0] * 7
        self.t_gen = 0.0
        self.t_score = 0.0
        self.t_hash = 0.0

    # ------------------------------------------------------------ plumbing
    def new_candidate(self, state, position, relation_id, permutation_id,
                      parent_id=0, age=0):
        nonce = derive_nonce(state, position, relation_id, permutation_id,
                             salt=self.seed)
        return Candidate(next(self._cid), parent_id, position, relation_id,
                         permutation_id, self.phase.phase, state,
                         nonce=nonce)

    def _seen_add(self, nonce):
        self.seen.add(nonce)
        self._seen_q.append(nonce)
        if len(self._seen_q) == self._seen_q.maxlen:
            old = self._seen_q.popleft()
            self.seen.discard(old)

    # ---------------------------------------------------- 1. generation
    def _fresh_random(self):
        state = tuple(self.rng.randrange(d) for d in self.cfg.dimensions)
        position = self.rng.randrange(len(state))
        rel = self.rel.pick(self.phase.phase, self.bias_rel)
        state = self.rel.apply(rel, state, position)
        perm = self.perm.pick(self.phase.phase, self.bias_perm)
        st = self.perm.apply(perm, state, self.rng) \
            if hasattr(self.perm, "apply") else state
        # occasional combination-indexed block variant (section 7 use)
        if self.rng.random() < 0.10:
            combos = self.comb.sample_indices(len(st), self.phase.phase,
                                              max_count=1)
            for combo in combos:
                st = self._combo_variant(st, combo)
                break
        return self.new_candidate(st, position, rel, perm)

    def _combo_variant(self, state, combo):
        st = list(state)
        vals = [st[i] for i in combo]
        self.rng.shuffle(vals)
        for i, v in zip(combo, vals):
            st[i] = v
        return tuple(st)

    def generate_candidates(self, n, exploration_rate):
        """Lazy generation: symbolic candidates only, never the full space."""
        pool = []
        n_explore = int(n * exploration_rate) + 1
        for _ in range(n_explore):
            pool.append(self._fresh_random())
        if self.elites:
            per = max(1, (n - len(pool)) // len(self.elites))
            per = min(per, max(2, self.cfg.max_cousins))
            for p in self.elites:
                if len(pool) >= n:
                    break
                pool.extend(self.cousin.get_cousins(p, limit=per))
        while len(pool) < n:
            pool.append(self._fresh_random())
        return pool[:n]

    # ------------------------------------------------ 4/5. rank + select
    def select_batch(self, pool, budget):
        pool.sort(key=lambda c: -c.score)
        out, used, dups = [], set(), 0
        for c in pool:
            if c.nonce in used or c.nonce in self.seen:
                dups += 1
                continue
            used.add(c.nonce)
            out.append(c)
            if len(out) >= budget:
                break
        while len(out) < budget:                  # exploration fill
            c = self._fresh_random()
            if c.nonce in used or c.nonce in self.seen:
                dups += 1
                continue
            used.add(c.nonce)
            out.append(c)
        self.dup_skipped += dups
        return out, dups

    # ---------------------------------------------- 6. patterns/events
    def push_events(self, batch, loop_success):
        if not batch:
            return ()
        elite = batch[0]
        keys = []
        keys += self.pattern.push(PS_POSITION, elite.position)
        self.rel.observe_position(elite.position)
        keys += self.pattern.push(PS_DELTA, self.rel.last_delta_class())
        keys += self.pattern.push(PS_RELATION, elite.relation_id)
        keys += self.pattern.push(PS_PERMUTATION, elite.permutation_id)
        keys += self.pattern.push(PS_PHASE, self.phase.phase)
        pids = []
        for key in keys:
            pid, is_new = self.ptable.intern(key)
            if pid is None:
                continue
            if is_new:
                self.discovered += 1
            pids.append(pid)
        return tuple(dict.fromkeys(pids))         # order-preserving dedup

    def choose_active_patterns(self, detected):
        """Patterns attributed to the next loop's attempts (section 12)."""
        if not self.cfg.use_patterns:
            return ()
        chosen = list(detected)
        bias = self.phase.phase_bias()
        for pid in self.memory.ranked(self.phase.phase,
                                      limit=self.cfg.max_active_pids * 2,
                                      phase_name=bias):
            if len(chosen) >= self.cfg.max_active_pids:
                break
            ps = self.stats.patterns.get(pid)
            if ps is None or ps.invalid:
                continue
            if ps.confidence >= self.cfg.confidence_active or \
                    pid in detected:
                if pid not in chosen:
                    chosen.append(pid)
        return tuple(chosen[:self.cfg.max_active_pids])

    # --------------------------------------- 7. statistical validation
    def validate_patterns(self):
        """Accept/uphold/downgrade patterns purely on statistics (sec 12)."""
        cfg = self.cfg
        validated_now = 0
        rejected_now = 0
        loop = self.loop_index
        for pid, ps in list(self.stats.patterns.items()):
            if ps.invalid:
                continue
            conf = compute_confidence(
                ps.samples, self.stats.consistency(pid),
                self.stats.pattern_effect(pid), self.stats.variance(pid),
                self.stats.age(pid, loop), cfg)
            ps.confidence = conf
            enough = ps.samples >= cfg.min_pattern_samples
            z = self.stats.pattern_z(pid)
            eff = self.stats.pattern_effect(pid)
            eff_val = self.stats.pattern_effect_stream(pid, validation=True)
            if enough and not ps.validated and z >= cfg.significance_z \
                    and eff > 1.0 and eff_val > 1.0:
                ps.validated = True
                validated_now += 1
                self.validated += 1
                self.phase.note_validated()
            elif enough and z < 0.0 and \
                    ps.samples >= cfg.min_pattern_samples * 2:
                ps.invalid = True                 # automatic rejection
                rejected_now += 1
                self.rejected += 1
                e = self.memory.get(pid)
                if e is not None:
                    e.score = 0.0
        return validated_now, rejected_now

    # --------------------------------------------- 8. generation bias
    def update_generation_bias(self):
        """Active patterns -> relation/perm pick bonuses (closed loop)."""
        br = [0.0] * 9
        bp = [0.0] * 7
        if self.cfg.use_patterns:
            for pid in self.active_patterns:
                e = self.memory.get(pid)
                s = min(0.5, max(0.0, e.score if e else 0.0))
                if s <= 0.0:
                    continue
                key = self.ptable.key(pid)
                stream = key[0]
                vals = []
                for v in key[2:]:               # flatten nested cycle tuples
                    if isinstance(v, tuple):
                        vals.extend(v)
                    else:
                        vals.append(v)
                if stream == PS_RELATION:
                    for v in vals:
                        if 0 <= v < 9 and v != REL_COUSIN:
                            br[v] += s
                elif stream == PS_PERMUTATION:
                    for v in vals:
                        if 0 <= v < 7:
                            bp[v] += s
                elif stream == PS_DELTA and vals:
                    r = DELTA_TO_REL.get(classify_delta(vals[-1]),
                                         REL_UNCHANGED)
                    if r != REL_COUSIN:
                        br[r] += s
        self.bias_rel = br
        self.bias_perm = bp

    # ------------------------------------------------------- main loop
    def run(self, max_hashes=None):
        cfg = self.cfg
        budget = min(cfg.max_hashes, max_hashes or cfg.max_hashes)
        hashed = 0
        successes_total = 0
        self.hw.start()
        last_score = 0.0
        while hashed < budget:
            t0 = time.perf_counter()
            self.memory.tick(self.loop_index)
            n = min(self.feedback.batch_size, budget - hashed)
            erate = self.phase.exploration_rate(hashed / max(1, budget))

            # 1-2. generate + context scores
            pool = self.generate_candidates(n * cfg.pool_width, erate)
            t1 = time.perf_counter()
            self.t_gen += t1 - t0
            for c in pool:
                c.score = self.context.score(c)
            t2 = time.perf_counter()
            self.t_score += t2 - t1

            # 4-5. rank, top-K, select unique batch
            batch, dups = self.select_batch(pool, n)
            for c in batch:
                self.topk.add(c.score, c)

            # 6. hash (cryptographic primitive untouched)
            oks, dt = self.worker.hash_batch([c.nonce for c in batch])
            self.t_hash += dt
            successes = sum(1 for ok in oks if ok)
            validation_stream = (self.loop_index % cfg.holdout_mod) == 0
            self.stats.note_attempt_batch(
                len(batch), successes, self.active_patterns,
                validation_stream, self.loop_index)
            hashed += len(batch)
            successes_total += successes

            for c, ok in zip(batch, oks):
                self._seen_add(c.nonce)
                if ok:
                    d = self.worker.hash_nonce(c.nonce)[1]
                    z = self.worker.leading_zero_bits(d)
                    self.stats.register_success(c.nonce, self.loop_index,
                                                self.phase.phase,
                                                len(self.active_patterns))
                    if self.log.enabled:
                        print("  FOUND nonce=%08x zeros=%d digest=%s... "
                              "phase=%s loop=%d"
                              % (c.nonce, z, d.hex()[:16],
                                 PHASE_TABLE[self.phase.phase],
                                 self.loop_index), flush=True)

            # 7. pattern events + evidence
            detected = self.push_events(batch, successes)
            for pid in detected:
                ps = self.stats.get(pid)
                conf = ps.confidence
                self.memory.observe(pid, 1.0 if successes else 0.0,
                                    self.phase.phase, conf)
            self.detected_last = detected
            v_now, r_now = self.validate_patterns()

            # 8. credit + feedback + phase + memory
            ev = 1.0 if successes else 0.0
            self.elites = tuple(batch[:cfg.elite_keep])
            for c in self.elites:
                self.rel.record(c.relation_id, ev)
                self.perm.record(c.permutation_id, ev)
            last_score = batch[0].score if batch else last_score
            self.phase.note_loop(self.loop_index, successes > 0,
                                 len(detected), v_now)
            self.feedback.update(self.loop_index, successes, dups,
                                 len(pool))
            self.memory.prune(self.phase.phase)
            self.active_patterns = self.choose_active_patterns(detected)
            self.update_generation_bias()
            self.loop_index += 1

            # compact logging (section 29)
            lg = self.log
            if lg.enabled:
                if lg.debug or self.loop_index % cfg.log_every == 0:
                    lg.row(self.loop_index,
                           PHASE_TABLE[self.phase.phase], len(pool),
                           len(self.active_patterns), last_score,
                           self.stats.khs_ema,
                           "%d hit%s" % (successes,
                                         "s" if successes != 1 else ""))
                if lg.statistics and self.loop_index % cfg.stats_every == 0:
                    s = self.stats.summary()
                    print("  STATS attempts=%d succ=%d rate=%.5f "
                          "exp=%.5f khs=%.1f patterns=%d"
                          % (s["attempts"], s["successes"],
                             s["baseline_rate"], s["expected_rate"],
                             s["khs"], s["patterns_tracked"]), flush=True)
                if lg.memory and self.loop_index % cfg.mem_every == 0:
                    m = self.memory.stats()
                    print("  MEM hot=%d active=%d cold=%d discarded=%d "
                          "usage=%dKB" % (m["counts"][MEM_HOT],
                                          m["counts"][MEM_ACTIVE],
                                          m["counts"][MEM_COLD],
                                          m["discarded"], m["usage_kb"]),
                          flush=True)

        return self.summary(budget, successes_total)

    # ---------------------------------------------------------- summary
    def summary(self, budget, successes_total):
        hw = self.hw.sample()
        wall = time.perf_counter() - self.log.t0 if self.log.enabled \
            else self.t_hash
        return {
            "mode": "adaptive",
            "seed": self.seed,
            "attempts": hashed_budget(budget),
            "successes": successes_total,
            "hash_time_s": self.stats.hash_time,
            "wall_s": wall,
            "khs": (budget / max(1e-9, self.stats.hash_time)) / 1000.0,
            "first_success_at": self.stats.first_success_attempt,
            "expected_rate": self.stats.expected,
            "success_per_khash": successes_total / max(1, budget) * 1000.0,
            "patterns_discovered": self.discovered,
            "patterns_validated": self.validated,
            "patterns_rejected": self.rejected,
            "patterns_tracked": len(self.stats.patterns),
            "phase_transitions": self.phase.transition_count,
            "final_phase": PHASE_TABLE[self.phase.phase],
            "memory_kb": self.memory.usage_kb(
                extra_bytes=len(self.seen) * 72),
            "memory_tiers": self.memory.counts(),
            "memory_discarded": self.memory.discarded,
            "dup_rate": self.feedback.dup_rate,
            "failures": dict(self.feedback.failure_counts),
            "cpu_pct": round(min(100.0, self.hw.load() * 100.0), 1),
            "ram_peak_kb": hw["rss_kb"],
            "temp_c": hw["temp_c"],
            "power_w": hw["power_w"],
            "timing": {"generate_s": round(self.t_gen, 3),
                       "score_s": round(self.t_score, 3),
                       "hash_s": round(self.t_hash, 3)},
            "topk_overflow": self.topk.overflows,
            "cousins_generated": self.cousin.generated,
            "cousins_emitted": self.cousin.emitted,
        }


def hashed_budget(b):
    return b


def random_Random(seed):
    import random
    return random.Random(seed)
