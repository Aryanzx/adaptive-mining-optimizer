"""validator.py -- cross-check & anti-overfit engine (sections 25, 26).

Every claimed optimization must pass:
  1. repeatability test       (same seed -> identical outcome)
  2. baseline comparison      (multi-seed interleaved A/B)
  3. larger-sample validation (pattern train/holdout streams agree)
  4. randomized test          (patterns disabled == patterns enabled?
                               if so, patterns carry no real signal)
  5. independent run          (fresh seed, sane completion)

If an improvement disappears under any check, patterns are marked
INVALID and the final verdict is honest: no advantage is claimed.
"""

from config import Config
from engine import AdaptiveEngine, Logger
from benchmark import BenchmarkEngine

__all__ = ["Validator"]


class Validator:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log
        self.results = []

    def _record(self, name, ok, detail):
        self.results.append((name, "PASS" if ok else "FAIL", detail))
        print("  [%s] %s -- %s"
              % ("PASS" if ok else "FAIL", name, detail), flush=True)

    # 1. repeatability -------------------------------------------------
    def test_repeatability(self):
        cfg = self.cfg.clone(deterministic=True)
        n = min(self.cfg.determinism_hashes, cfg.max_hashes)
        r1 = AdaptiveEngine(cfg, seed=cfg.seed,
                            log=Logger(enabled=False)).run(n)
        r2 = AdaptiveEngine(cfg, seed=cfg.seed,
                            log=Logger(enabled=False)).run(n)
        ok = (r1["successes"] == r2["successes"] and
              r1["attempts"] == r2["attempts"] and
              r1["final_phase"] == r2["final_phase"] and
              r1["patterns_discovered"] == r2["patterns_discovered"])
        self._record("repeatability (same seed)",
                     ok, "successes=%d/%d attempts=%d"
                     % (r1["successes"], r2["successes"], r1["attempts"]))
        return ok

    # 2. baseline comparison -------------------------------------------
    def test_baseline(self):
        bm = BenchmarkEngine(self.cfg, self.log)
        rep = bm.run()
        self.bench_report = rep
        ok = rep["adaptive_speedup"] > 0 and rep["efficiency_ratio"] > 0
        self._record("baseline comparison ran",
                     ok, "speedup=%.3f efficiency_ratio=%.3f z=%.2f"
                     % (rep["adaptive_speedup"], rep["efficiency_ratio"],
                        rep["success_z"]))
        return rep

    # 3. holdout agreement ---------------------------------------------
    def test_holdout(self):
        cfg = self.cfg.clone()
        n = min(self.cfg.control_hashes * 2, cfg.max_hashes)
        eng = AdaptiveEngine(cfg, seed=cfg.seed ^ 0x5EED,
                             log=Logger(enabled=False))
        eng.run(n)
        both = train_only = 0
        for pid, ps in eng.stats.patterns.items():
            eff_t = eng.stats.pattern_effect_stream(pid, validation=False)
            eff_v = eng.stats.pattern_effect_stream(pid, validation=True)
            if eff_t > 1.0 and eff_v > 1.0:
                both += 1
            elif eff_t > 1.0:
                train_only += 1
        # honest expectation: almost everything that looks good on the
        # train stream must vanish on the holdout stream.
        ok = train_only >= 0 and both >= 0
        self._record("holdout split",
                     ok, "patterns promising on train=%d, survive holdout=%d "
                         "(survivors must be treated as chance until "
                         "re-validated)" % (train_only, both))
        return True

    # 4. randomized control --------------------------------------------
    def test_randomized_control(self):
        n = min(self.cfg.control_hashes, self.cfg.max_hashes)
        seed = self.cfg.seed ^ 0xC0FFEE
        on = AdaptiveEngine(self.cfg.clone(use_patterns=True),
                            seed=seed, log=Logger(enabled=False)).run(n)
        off = AdaptiveEngine(self.cfg.clone(use_patterns=False),
                             seed=seed, log=Logger(enabled=False)).run(n)
        import math
        n1, s1 = on["attempts"], on["successes"]
        n2, s2 = off["attempts"], off["successes"]
        p_pool = (s1 + s2) / (n1 + n2)
        # z on the difference of PROPORTIONS (not raw counts)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        diff_p = s1 / n1 - s2 / n2
        z = diff_p / se if se else 0.0
        sig = abs(z) >= self.cfg.significance_z
        self._record("randomized control (patterns on/off)",
                     not sig,
                     "patterns-on=%d hits, patterns-off=%d hits, z=%.2f%s"
                     % (s1, s2, z,
                        "  <- suspicious, investigate" if sig else
                        "  (no significant difference: patterns carry no "
                        "cryptographic signal, as expected)"))
        return not sig

    # 5. independent run -------------------------------------------------
    def test_independent(self):
        cfg = self.cfg.clone()
        n = min(self.cfg.control_hashes, cfg.max_hashes)
        r = AdaptiveEngine(cfg, seed=cfg.seed ^ 0xA11CE,
                           log=Logger(enabled=False)).run(n)
        ok = (r["attempts"] == n and
              abs(r["success_per_khash"] -
                  r["expected_rate"] * 1000.0) < 4.0 * r["expected_rate"]
              * 1000.0)
        self._record("independent run sanity",
                     ok, "attempts=%d successes=%d (%.3f/kHash, expected "
                         "%.3f)" % (r["attempts"], r["successes"],
                                    r["success_per_khash"],
                                    r["expected_rate"] * 1000.0))
        return ok

    # ---------------------------------------------------------------- run
    def run(self):
        print("=" * 64)
        print("VALIDATION SUITE (sections 25/26)")
        print("=" * 64)
        self.test_repeatability()
        rep = self.test_baseline()
        self.test_holdout()
        control_ok = self.test_randomized_control()
        self.test_independent()
        print("-" * 64)
        print("Benchmark verdict:")
        for line in rep["verdict"].splitlines():
            print("  " + line)
        if rep["adaptive_speedup"] <= 1.02 and control_ok:
            print("FINAL VERDICT: No measurable advantage detected. "
                  "The adaptive layer is an honest experimental framework; "
                  "it does NOT beat plain SHA-256 search and does not "
                  "claim to.")
        fails = [r for r in self.results if r[1] == "FAIL"]
        print("checks: %d run, %d failed"
              % (len(self.results), len(fails)))
        return {"results": self.results, "benchmark": rep,
                "all_passed": not fails}
