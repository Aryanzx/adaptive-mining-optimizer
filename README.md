# Adaptive Embedded Cryptographic Search & Mining Optimizer

A lightweight, modular, **CPU-only**, **stdlib-only** experimental
mining/search engine. It builds an intelligent optimization layer *around*
an **unmodified SHA-256** hash worker: permutations, combinations, position
relations, cousin states, pattern detection, statistical validation,
phase-change memory, top-K selection, and hardware-aware feedback.

> **What this is NOT:** it is not a hash breaker and not a claim of a
> cryptographic shortcut. SHA-256 is used exactly as standardized.
> Every "optimization" is treated as an experimental candidate-selection
> heuristic and must prove itself in benchmark data (section 33). As
> measured below, the honest result is:
> **"No measurable advantage detected."**

---

## Quick start

```bash
python3 main.py                       # adaptive demo run (100k hashes)
python3 main.py --hashes 50000 --profile
python3 main.py --debug --statistics --memory
python3 main.py --benchmark           # MODE A vs MODE B (section 20)
python3 main.py --validate            # cross-check / anti-overfit suite
python3 main.py --embedded            # strict embedded budgets
python3 main.py --test                # 50 unit tests
```

Everything runs on a plain CPU with Python 3.8+ and **no third-party
packages** (no pandas, no numpy, no ML framework, no GPU).

---

## Architecture

```
CONFIG
  ↓
SEARCH-SPACE MODEL (symbolic, 200,000,000 states, never enumerated)
  ↓
CANDIDATE GENERATOR ── PERMUTATION + COMBINATION + RELATION engines
  ↓
MULTI-CONTEXT MATCHER (7 contexts)      COUSIN GENERATOR
  ↓                                            ↓
PATTERN ENGINE ──► STATISTICAL RANKING ──► TOP-K (K=32)
  ↓
HASH WORKER (SHA-256, untouched)  →  TARGET TEST
  ↓
FEEDBACK → PHASE MANAGER → MEMORY UPDATE → NEXT SEARCH
```

Phase machine: `EXPLORE → DISCOVER → VALIDATE → EXPLOIT → STAGNATION →
RESET → EXPLORE`. Exploitation is never permanent (dwell cap + stagnation
window).

## Files

| File | Role | Spec |
|---|---|---|
| `config.py` | all knobs + integer ID tables (relations, phases, tiers, perms, contexts, streams) | 4, 13, 22, 23 |
| `candidate.py` | `__slots__` integer candidate, SplitMix nonce derivation, lazy `SearchSpace.next_candidate()` | 4, 5 |
| `permutation.py` | swap/rotate/reverse/cyclic/adjacent/block/partial + `permutation_score()` | 6 |
| `combination.py` | `nCr`, direct un-ranking (`nth_combination`), lazy bounded streams | 7 |
| `relation.py` | relation IDs 0-8, position deltas, delta classes | 8 |
| `cousin.py` | `get_cousins()` 8 mutation ops, hard limit, nonce dedup | 9 |
| `context.py` | contexts A-G, normalized weighted score | 10 |
| `pattern.py` | rep/cycle2/cycle3/run/transition detection, `PATTERN_TABLE` interning | 11, 23 |
| `statistics_engine.py` | hash stats, per-pattern conditional stats, train/holdout streams, z-scores | 12, 18, 19, 26 |
| `confidence.py` | confidence = f(samples, consistency, effect, variance, age) + bands | 24 |
| `phase.py` | 6-phase state machine, exploration-rate schedule, phase memory bias | 13, 15, 17 |
| `memory.py` | HOT/ACTIVE/COLD/DISCARD tiers, aging `score*decay+evidence`, hard caps | 14, 22 |
| `topk.py` | bounded top-K heap | 16 |
| `hash_worker.py` | SHA-256 interface, target test, timing/throughput | 18 |
| `engine.py` | the adaptive feedback loop + compact logging | 27, 29 |
| `benchmark.py` | MODE A vs MODE B interleaved multi-seed, section-30 report | 20, 30 |
| `hardware.py` | CPU load, RSS, thermal, power (all optional/graceful), batch suggestions | 21, 28 |
| `feedback.py` | failure detection: overflow, stagnation, dup candidates, instability | 28 |
| `validator.py` | repeatability, baseline, holdout, randomized control, independent run | 25, 26 |
| `tests/` | 50 unit + end-to-end tests | all |

## How the search space stays symbolic (section 5)

The configured space `10×10×10×20×10×100×10` has **200,000,000** states
(note: the spec prose says 20M, but the true product of those dimensions is
200M — the code asserts the correct product in tests). Nothing is ever
enumerated: states are decoded from an integer rank on demand, and a
candidate's nonce is derived deterministically from its fields with a cheap
SplitMix-style integer mixer. The hash worker only ever sees plain 32-bit
nonces, so MODE A and MODE B search the *same* nonce universe under
*identical* hashing conditions.

## Measured results (this machine, 2026-10-04)

Full logs: `BENCHMARK_OUTPUT.txt`, `VALIDATION_OUTPUT.txt`.

```
Baseline (sequential nonce):  664 kH/s, 653 hits / 180,000
Adaptive  (full optimizer):    20 kH/s, 721 hits / 180,000
adaptive_speedup = 0.030        (implementation: overhead, not speedup)
success/kHash ratio = 1.104, z = 1.84 (not significant, threshold 2.0)
=> No measurable advantage detected.
```

Validation suite (5/5 PASS):
* repeatability: same seed → byte-identical outcome
* holdout split: 32 patterns looked promising on the TRAIN stream, only 5
  survived the holdout (treated as chance) — anti-overfitting works
* randomized control: patterns on vs off, z = -1.25 → patterns carry **no**
  cryptographic signal, exactly as expected
* pattern ledger in a 50k run: 75 discovered, **0 validated**, 31
  auto-rejected
* memory: 154 KB adaptive structures (cap 256 KB)

**Interpretation (section 30/33):** the adaptive layer costs ~30×
end-to-end throughput in pure Python bookkeeping and finds solutions at
exactly the statistically expected rate. The framework's value is that it
*measures* this honestly instead of assuming an advantage.

## Embedded mode (section 22)

Default budgets: `MAX_MEMORY=256 KB`, `MAX_ACTIVE_PATTERNS=128`,
`MAX_HOT_CANDIDATES=32`, `MAX_COUSINS=16`, dedup window 2048 nonces.
Fixed-capacity deques/heaps/dicts only, integer IDs everywhere, deterministic
under seed (`--seed`, and `deterministic=True` locks hardware influence).
Porting checklist to bare-metal C: replace `deque`→ring buffers (already
fixed-capacity), `hashlib`→your SHA-256, `resource`/`/proc` sampling→your
RTOS hooks; the decision logic uses only integer/float ops.

## Scientific rules enforced in code

1. Patterns activate only with `samples >= min_pattern_samples`, `z >= 2.0`
   AND holdout-stream effect > 1 (section 12).
2. Patterns with `z < 0` after enough samples are **automatically marked
   INVALID** and forgotten (sections 25, 28).
3. Train/holdout streams split every 3rd loop (section 26).
4. Confidence 0.0 whenever measured effect ≤ 1.0, regardless of sample
   count — confidence is heuristic-evidence, never a probability of solving
   a hash (section 24).
5. The benchmark distinguishes implementation speedup from cryptographic
   shortcut and prints `"No measurable advantage detected."` unless both
   survive multi-seed significance testing (section 30).

## Future extensions (section 34 — interfaces only, not implemented)

`HashWorker.hash_batch` is the sole crypto touchpoint (swap in SIMD/GPU
kernels); `HardwareMonitor.suggest_batch` exposes parallelism control;
`ContextMatcher.weights` is the hook for an RL/Bayesian controller. None of
these are added until this CPU version's benchmarks say they are worth it.
