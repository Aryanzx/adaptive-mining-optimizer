#!/usr/bin/env python3
"""main.py -- Adaptive Embedded Cryptographic Search & Mining Optimizer.

CPU-only, stdlib-only, embedded-friendly experimental mining/search
engine around an UNMODIFIED SHA-256 worker.

Usage:
  python3 main.py                       # short adaptive demo run
  python3 main.py --benchmark           # Mode A vs Mode B comparison
  python3 main.py --validate            # full cross-check/anti-overfit suite
  python3 main.py --debug --statistics --memory --profile
  python3 main.py --embedded            # strict embedded budgets
  python3 main.py --test                # unit tests

Flags of interest:
  --hashes N        hash budget for a run           (default 100000)
  --difficulty N    leading zero bits required      (default 8)
  --seed N          RNG seed (reproducible runs)
  --topk N          top-K candidates kept per loop  (default 32)
  --batch N         hashes per adaptive loop        (default 256)
  --max-cousins N   cousins per parent              (default 16)
  --max-memory-kb N memory budget                   (default 256)
  --dims a,b,c,...  multidimensional search space
  --json            machine-readable summary output
"""

import argparse
import json
import os
import sys

from config import Config
from engine import AdaptiveEngine, Logger
from benchmark import BenchmarkEngine


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Adaptive Embedded Cryptographic Search & Mining "
                    "Optimizer (CPU only)")
    modes = p.add_argument_group("modes")
    modes.add_argument("--benchmark", action="store_true",
                       help="run baseline vs adaptive benchmark")
    modes.add_argument("--validate", action="store_true",
                       help="run full validation/anti-overfit suite")
    modes.add_argument("--embedded", action="store_true",
                       help="force strict embedded budgets")
    modes.add_argument("--test", action="store_true",
                       help="run unit test suite and exit")
    p.add_argument("--debug", action="store_true", help="log every loop")
    p.add_argument("--statistics", action="store_true",
                   help="periodic statistics dump")
    p.add_argument("--memory", action="store_true",
                   help="periodic memory-tier dump")
    p.add_argument("--profile", action="store_true",
                   help="timing breakdown at end")
    p.add_argument("--json", action="store_true",
                   help="emit final summary as JSON")
    p.add_argument("--hashes", type=int, default=None)
    p.add_argument("--difficulty", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--max-cousins", type=int, default=None)
    p.add_argument("--max-memory-kb", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None,
                   help="benchmark seed count")
    p.add_argument("--dims", type=str, default=None,
                   help="comma separated dimensions, e.g. 10,10,10,20,10,100,10")
    return p.parse_args(argv)


def build_cfg(args):
    kw = {}
    if args.hashes is not None:
        kw["max_hashes"] = args.hashes
    if args.difficulty is not None:
        kw["difficulty"] = args.difficulty
    if args.seed is not None:
        kw["seed"] = args.seed
    if args.topk is not None:
        kw["top_k"] = args.topk
        kw["max_hot_candidates"] = args.topk
    if args.batch is not None:
        kw["batch_size"] = args.batch
    if args.max_cousins is not None:
        kw["max_cousins"] = args.max_cousins
    if args.max_memory_kb is not None:
        kw["max_memory_kb"] = args.max_memory_kb
    if args.seeds is not None:
        kw["benchmark_seeds"] = args.seeds
    if args.dims:
        kw["dimensions"] = tuple(int(x) for x in args.dims.split(",")
                                 if x.strip())
    cfg = Config(**kw)
    if args.embedded:
        cfg = cfg.clone(embedded=True, top_k=32, max_hot_candidates=32,
                        max_cousins=16, max_active_patterns=128,
                        max_memory_kb=min(cfg.max_memory_kb, 256),
                        batch_size=min(cfg.batch_size, 256))
    if args.debug:
        cfg = cfg.clone(log_every=1)
    return cfg


def run_demo(cfg, args):
    log = Logger(debug=args.debug, statistics=args.statistics,
                 memory=args.memory, profile=args.profile, enabled=True)
    print("=" * 64)
    print("Adaptive Embedded Cryptographic Search & Mining Optimizer")
    print("CPU only | SHA-256 untouched | difficulty=%d zero bits | "
          "space=%d symbolic states" % (cfg.difficulty, cfg.total_space))
    print("dims=%s seed=%#x top_k=%d batch=%d cousins<=%d mem<=%dKB"
          % (list(cfg.dimensions), cfg.seed, cfg.top_k, cfg.batch_size,
             cfg.max_cousins, cfg.max_memory_kb))
    print("=" * 64)
    eng = AdaptiveEngine(cfg, log=log)
    s = eng.run(cfg.max_hashes)
    print("-" * 64)
    print("RESULT successes=%d / attempts=%d (%.3f per kHash, expected "
          "%.3f)" % (s["successes"], s["attempts"], s["success_per_khash"],
                     s["expected_rate"] * 1000.0))
    print("hash-rate %.1f kH/s | first success at attempt %s"
          % (s["khs"], s["first_success_at"]))
    print("patterns: discovered=%d validated=%d rejected=%d | phases: "
          "%d transitions, ended in %s"
          % (s["patterns_discovered"], s["patterns_validated"],
             s["patterns_rejected"], s["phase_transitions"],
             s["final_phase"]))
    print("memory: %d KB (hot=%d active=%d cold=%d discarded=%d) | "
          "dup-rate %.3f | CPU %.1f%%"
          % (s["memory_kb"], s["memory_tiers"][0], s["memory_tiers"][1],
             s["memory_tiers"][2], s["memory_discarded"], s["dup_rate"],
             s["cpu_pct"]))
    if args.profile:
        t = s["timing"]
        tot = max(1e-9, t["generate_s"] + t["score_s"] + t["hash_s"])
        print("profile: generate=%.2fs (%.0f%%) score=%.2fs (%.0f%%) "
              "hash=%.2fs (%.0f%%)"
              % (t["generate_s"], 100 * t["generate_s"] / tot,
                 t["score_s"], 100 * t["score_s"] / tot,
                 t["hash_s"], 100 * t["hash_s"] / tot))
    if args.json:
        print(json.dumps(s, indent=2, default=str))
    return s


def main(argv=None):
    args = parse_args(argv)
    if args.test:
        import unittest
        root = os.path.dirname(os.path.abspath(__file__))
        suite = unittest.defaultTestLoader.discover(
            os.path.join(root, "tests"), top_level_dir=root)
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if res.wasSuccessful() else 1

    cfg = build_cfg(args)

    if args.benchmark:
        bm = BenchmarkEngine(cfg, Logger(enabled=True))
        print("Running benchmark: %d hashes x %d seeds x 2 modes "
              "(interleaved)" % (cfg.benchmark_hashes, cfg.benchmark_seeds))
        rep = bm.run()
        print(BenchmarkEngine.format_report(rep))
        if args.json:
            rep_out = {k: v for k, v in rep.items()
                       if k not in ("baseline_runs", "adaptive_runs")}
            print(json.dumps(rep_out, indent=2, default=str))
        return 0

    if args.validate:
        from validator import Validator
        v = Validator(cfg, Logger(enabled=True))
        out = v.run()
        if args.json:
            print(json.dumps(out, indent=2, default=str))
        return 0

    run_demo(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
