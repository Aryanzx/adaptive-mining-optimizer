"""benchmark.py -- mandatory baseline comparison (sections 20, 30).

MODE A: plain sequential nonce search (the honest baseline).
MODE B: full adaptive engine.

Both run under equivalent conditions: identical hash worker, identical
difficulty, identical hash budget, interleaved order across seeds.  The
report distinguishes IMPLEMENTATION SPEEDUP from CRYPTOGRAPHIC SHORTCUT
and refuses to manufacture an advantage (section 33).
"""

import time

from config import Config
from hash_worker import HashWorker
from engine import AdaptiveEngine, Logger

__all__ = ["BenchmarkEngine"]


class BenchmarkEngine:
    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log or Logger(enabled=False)

    # ------------------------------------------------------------- MODE A
    def run_baseline(self, budget, seed):
        worker = HashWorker(self.cfg)
        start = (seed * 0x9E3779B1) & 0xFFFFFFFF
        mask = 0xFFFFFFFF
        shift = worker.shift
        successes = 0
        first_success_at = None
        t0 = time.perf_counter()
        for i in range(1, budget + 1):
            d = _sha_digest(start)
            start = (start + 1) & mask
            if (int.from_bytes(d[:8], "big") >> shift) == 0:
                successes += 1
                if first_success_at is None:
                    first_success_at = i
        dt = time.perf_counter() - t0
        return {"mode": "baseline", "seed": seed, "attempts": budget,
                "successes": successes, "hash_time_s": dt,
                "khs": budget / dt / 1000.0,
                "first_success_at": first_success_at,
                "success_per_khash": successes / budget * 1000.0}

    # ------------------------------------------------------------- MODE B
    def run_adaptive(self, budget, seed):
        cfg = self.cfg.clone()
        eng = AdaptiveEngine(cfg, seed=seed, log=Logger(enabled=False))
        t0 = time.perf_counter()
        res = eng.run(budget)
        res["wall_s"] = time.perf_counter() - t0
        # FAIRNESS: throughput is end-to-end wall time for BOTH modes --
        # the adaptive layer must pay for its own bookkeeping overhead.
        res["khs"] = budget / max(1e-9, res["wall_s"]) / 1000.0
        return res

    # --------------------------------------------------------------- run
    def run(self, budget=None, seeds=None):
        cfg = self.cfg
        budget = min(budget or cfg.benchmark_hashes, 2 ** 32 - 1)
        seeds = seeds if seeds is not None else cfg.benchmark_seeds
        base_runs, adapt_runs = [], []
        for i in range(seeds):
            seed = cfg.seed + i * 0x10001
            r_a = self.run_baseline(budget, seed)
            r_b = self.run_adaptive(budget, seed)
            base_runs.append(r_a)
            adapt_runs.append(r_b)
            if self.log.enabled:
                print("  seed %#x: baseline %.1f kH/s (%d hits) | "
                      "adaptive %.1f kH/s (%d hits)"
                      % (seed, r_a["khs"], r_a["successes"],
                         r_b["khs"], r_b["successes"]), flush=True)
        rep = self._aggregate(base_runs, adapt_runs, budget, seeds)
        return rep

    def _aggregate(self, base, adapt, budget, seeds):
        mkhs_a = sum(r["khs"] for r in base) / seeds
        mkhs_b = sum(r["khs"] for r in adapt) / seeds
        succ_a = sum(r["successes"] for r in base)
        succ_b = sum(r["successes"] for r in adapt)
        eff_a = succ_a / (budget * seeds) * 1000.0
        eff_b = succ_b / (budget * seeds) * 1000.0
        speedup = mkhs_b / mkhs_a if mkhs_a else 0.0
        eff_ratio = eff_b / eff_a if eff_a else 0.0
        # two-sample z of success counts (equal n per mode, pooled p)
        n = budget * seeds
        p_pool = (succ_a + succ_b) / (2.0 * n) if n else 0.0
        import math
        if 0.0 < p_pool < 1.0:
            se = math.sqrt(2.0 * n * p_pool * (1.0 - p_pool))
        else:
            se = 0.0
        z_succ = ((succ_b - succ_a) / se) if se else 0.0
        a0 = adapt[0]
        rep = {
            "budget_per_seed": budget,
            "seeds": seeds,
            "difficulty": self.cfg.difficulty,
            "baseline_khs": mkhs_a,
            "adaptive_khs": mkhs_b,
            "baseline_successes": succ_a,
            "adaptive_successes": succ_b,
            "baseline_eff": eff_a,
            "adaptive_eff": eff_b,
            "difference_khs": mkhs_b - mkhs_a,
            "relative_improvement": (mkhs_b / mkhs_a - 1.0) if mkhs_a else 0.0,
            "adaptive_speedup": speedup,
            "efficiency_ratio": eff_ratio,
            "success_z": z_succ,
            "memory_kb": a0["memory_kb"],
            "cpu_pct": a0["cpu_pct"],
            "ram_peak_kb": a0["ram_peak_kb"],
            "patterns_discovered": a0["patterns_discovered"],
            "patterns_validated": a0["patterns_validated"],
            "patterns_rejected": a0["patterns_rejected"],
            "phase_transitions": a0["phase_transitions"],
            "baseline_runs": base,
            "adaptive_runs": adapt,
        }
        rep["verdict"] = self._verdict(rep)
        return rep

    def _verdict(self, rep):
        lines = []
        lines.append("IMPLEMENTATION SPEEDUP: %s (adaptive/baseline "
                     "throughput = %.3f)"
                     % ("YES" if rep["adaptive_speedup"] > 1.02 else "NO",
                        rep["adaptive_speedup"]))
        crypto = abs(rep["success_z"]) < self.cfg.significance_z
        lines.append("CRYPTOGRAPHIC SHORTCUT: %s (success-per-hash ratio = "
                     "%.3f, z = %.2f vs baseline)"
                     % ("NO" if crypto else "INCONCLUSIVE -- investigate",
                        rep["efficiency_ratio"], rep["success_z"]))
        if rep["adaptive_speedup"] <= 1.02 and crypto:
            lines.append("=> No measurable advantage detected. "
                         "(Expected: SHA-256 is designed to resist exactly "
                         "this kind of structure exploitation.)")
        return "\n".join(lines)

    # -------------------------------------------------------------- text
    @staticmethod
    def format_report(rep):
        L = []
        L.append("=" * 64)
        L.append("BENCHMARK REPORT  (CPU only, 1 thread, SHA-256, "
                 "difficulty=%d leading zero bits)" % rep["difficulty"])
        L.append("budget=%d hashes/mode/seed, seeds=%d, interleaved A/B"
                 % (rep["budget_per_seed"], rep["seeds"]))
        L.append("=" * 64)
        L.append("Baseline (Mode A, sequential nonce):")
        L.append("  hashrate            = %10.1f kH/s (mean)"
                 % rep["baseline_khs"])
        L.append("  successes           = %d / %d attempts"
                 % (rep["baseline_successes"],
                    rep["budget_per_seed"] * rep["seeds"]))
        L.append("  successes/kHash     = %.3f" % rep["baseline_eff"])
        L.append("Adaptive (Mode B, full optimizer):")
        L.append("  hashrate            = %10.1f kH/s (mean)"
                 % rep["adaptive_khs"])
        L.append("  successes           = %d / %d attempts"
                 % (rep["adaptive_successes"],
                    rep["budget_per_seed"] * rep["seeds"]))
        L.append("  successes/kHash     = %.3f" % rep["adaptive_eff"])
        L.append("Difference  (Y - X)    = %.1f kH/s" % rep["difference_khs"])
        L.append("Relative    (Y/X - 1)  = %+.1f%%"
                 % (rep["relative_improvement"] * 100.0))
        L.append("adaptive_speedup       = %.3f" % rep["adaptive_speedup"])
        L.append("Memory: %d KB (adaptive structures, cap %d KB) | "
                 "RAM peak: %s KB | CPU: %s%%"
                 % (rep["memory_kb"], 256, rep["ram_peak_kb"],
                    rep["cpu_pct"]))
        L.append("Patterns: discovered=%d validated=%d rejected=%d | "
                 "phase transitions=%d"
                 % (rep["patterns_discovered"], rep["patterns_validated"],
                    rep["patterns_rejected"], rep["phase_transitions"]))
        L.append("-" * 64)
        L.append(rep["verdict"])
        return "\n".join(L)


def _sha_digest(nonce):
    import hashlib
    import struct
    return hashlib.sha256(struct.pack("<I", nonce)).digest()
