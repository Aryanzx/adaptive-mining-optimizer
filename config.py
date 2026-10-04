"""config.py -- central configuration and integer-ID tables.

Everything is a plain int/float/tuple so this file can be transplanted
directly into an embedded build (section 22).  No pandas, no ML framework,
no GPU: CPU only.  Strings exist only in the *_TABLE tuples, which are used
for human-readable logging -- runtime decisions use integer IDs only.
"""

__all__ = [
    "Config",
    # relations
    "REL_UNCHANGED", "REL_INC", "REL_DEC", "REL_SWAP", "REL_REVERSE",
    "REL_ROTATE", "REL_NEIGHBOR", "REL_COUSIN", "REL_SHIFT", "RELATION_TABLE",
    # phases
    "PH_EXPLORE", "PH_DISCOVER", "PH_VALIDATE", "PH_EXPLOIT",
    "PH_STAGNATION", "PH_RESET", "PHASE_TABLE",
    # memory tiers
    "MEM_HOT", "MEM_ACTIVE", "MEM_COLD", "MEM_DISCARD", "MEM_TABLE",
    # permutation transforms
    "PERM_SWAP", "PERM_ROTATE", "PERM_REVERSE", "PERM_CYCLIC",
    "PERM_ADJACENT", "PERM_BLOCK", "PERM_PARTIAL", "PERMUTATION_TABLE",
    # contexts
    "CTX_POSITION", "CTX_RELATION", "CTX_PERMUTATION", "CTX_PERFORMANCE",
    "CTX_PHASE", "CTX_HARDWARE", "CTX_HASHSTATS",
    # pattern streams
    "PS_POSITION", "PS_RELATION", "PS_PERMUTATION", "PS_PHASE", "PS_DELTA",
]

# ---------------------------------------------------------------- relation IDs
REL_UNCHANGED, REL_INC, REL_DEC, REL_SWAP, REL_REVERSE, REL_ROTATE, \
    REL_NEIGHBOR, REL_COUSIN, REL_SHIFT = range(9)
RELATION_TABLE = ("unchanged", "+1", "-1", "swap", "reverse", "rotate",
                  "neighboring", "cousin", "positional_shift")

# ------------------------------------------------------------------- phase IDs
PH_EXPLORE, PH_DISCOVER, PH_VALIDATE, PH_EXPLOIT, PH_STAGNATION, PH_RESET = \
    range(6)
PHASE_TABLE = ("EXPLORE", "DISCOVER", "VALIDATE", "EXPLOIT", "STAGNATION",
               "RESET")

# --------------------------------------------------------------- memory tiers
MEM_HOT, MEM_ACTIVE, MEM_COLD, MEM_DISCARD = range(4)
MEM_TABLE = ("HOT", "ACTIVE", "COLD", "DISCARD")

# ------------------------------------------------------- permutation transform
PERM_SWAP, PERM_ROTATE, PERM_REVERSE, PERM_CYCLIC, PERM_ADJACENT, \
    PERM_BLOCK, PERM_PARTIAL = range(7)
PERMUTATION_TABLE = ("swap", "rotate", "reverse", "cyclic_shift",
                     "adjacent_swap", "block_perm", "partial_perm")

# ---------------------------------------------------------------- context IDs
CTX_POSITION, CTX_RELATION, CTX_PERMUTATION, CTX_PERFORMANCE, CTX_PHASE, \
    CTX_HARDWARE, CTX_HASHSTATS = range(7)

# ------------------------------------------------------------- pattern streams
PS_POSITION, PS_RELATION, PS_PERMUTATION, PS_PHASE, PS_DELTA = range(5)


class Config:
    """All tunable knobs.  Defaults satisfy EMBEDDED_MODE budgets."""

    def __init__(self, **kw):
        # ---- search space (section 5)
        self.dimensions = (10, 10, 10, 20, 10, 100, 10)   # 20,000,000 states
        self.difficulty = 8                # leading zero bits required
        # ---- budgets
        self.max_hashes = 100_000          # per run
        self.batch_size = 256              # hashes per loop
        self.pool_width = 2                # candidates generated per hashed
        self.seed = 0xC0FFEE
        # ---- EMBEDDED_MODE structure caps (section 22)
        self.embedded = True
        self.max_memory_kb = 256
        self.max_active_patterns = 128
        self.max_hot_candidates = 32       # == top_k
        self.max_cousins = 16
        self.history_len = 64              # ring-buffer length per stream
        self.dedup_window = 2048           # bounded recent-nonce window
        # ---- top-K (section 16)
        self.top_k = 32
        self.elite_keep = 4                # parents kept for cousin expansion
        self.max_active_pids = 8           # patterns attributed concurrently
        # ---- exploration / exploitation (section 17)
        self.explore_start = 0.80
        self.explore_end = 0.30
        self.explore_stagnation = 0.90
        # ---- statistics / validation (sections 12, 24, 26)
        self.min_pattern_samples = 1000    # hash attempts attributed
        self.significance_z = 2.0
        self.confidence_active = 0.50
        self.decay = 0.98
        self.holdout_mod = 3               # every 3rd loop -> validation stream
        self.use_patterns = True           # False = randomized control mode
        self.deterministic = False         # True = lock HW effects / batch size
        # ---- memory tiers (section 14)
        self.discard_score = 0.05
        self.active_score = 0.25
        # ---- phase machine (section 13)
        self.stagnation_window = 8         # loops without improvement
        self.validate_loops = 4
        self.exploit_max_dwell = 16
        self.discover_min_patterns = 2
        # ---- benchmark (section 20)
        self.benchmark_hashes = 60_000
        self.benchmark_seeds = 3
        # ---- hardware (section 21)
        self.threads = 1
        self.batch_min = 64
        self.batch_max = 2048
        # ---- logging (section 29)
        self.log_every = 25
        self.stats_every = 100
        self.mem_every = 100
        # ---- validation budgets (section 25)
        self.determinism_hashes = 20_000
        self.control_hashes = 30_000

        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError("unknown config key: %s" % k)
            setattr(self, k, v)
        self.total_space = 1
        for d in self.dimensions:
            self.total_space *= d
        if not (1 <= self.difficulty <= 64):
            raise ValueError("difficulty must be 1..64 leading zero bits")

    def clone(self, **kw):
        c = Config.__new__(Config)
        c.__dict__.update(self.__dict__)
        for k, v in kw.items():
            if not hasattr(c, k):
                raise KeyError("unknown config key: %s" % k)
            setattr(c, k, v)
        return c
