"""hash_worker.py -- cryptographic primitive interface (section 18).

The hash (SHA-256) is used exactly as standardized -- it is NEVER modified,
short-circuited, or otherwise weakened.  The worker:

  1. packs the candidate nonce (4 bytes, little-endian),
  2. computes the standard SHA-256 digest,
  3. compares against the target (leading zero bits >= difficulty),
  4. returns the result and records timing/throughput.
"""

import hashlib
import struct
import time

__all__ = ["HashWorker"]

_sha256 = hashlib.sha256
_pack = struct.Struct("<I").pack


class HashWorker:
    def __init__(self, cfg, stats=None):
        self.cfg = cfg
        self.stats = stats
        self.difficulty = cfg.difficulty
        self.shift = 64 - cfg.difficulty
        self.total_hashed = 0
        self.total_time = 0.0

    def leading_zero_bits(self, digest):
        nz = 0
        for byte in digest:
            if byte == 0:
                nz += 8
                continue
            nz += 8 - byte.bit_length()
            break
        return nz

    def hash_nonce(self, nonce):
        """Single attempt -> (ok, digest).  Plain SHA-256, no tricks."""
        d = _sha256(_pack(nonce & 0xFFFFFFFF)).digest()
        ok = (int.from_bytes(d[:8], "big") >> self.shift) == 0
        return ok, d

    def hash_batch(self, nonces):
        """Hash a batch; returns (ok_flags, elapsed_seconds).

        Local bindings keep the inner loop a pure CPU hash benchmark so
        baseline and adaptive modes execute byte-identical work.
        """
        t0 = time.perf_counter()
        shift = self.shift
        oks = [(int.from_bytes(_sha256(_pack(n)).digest()[:8], "big")
                >> shift) == 0 for n in nonces]
        dt = time.perf_counter() - t0
        self.total_hashed += len(nonces)
        self.total_time += dt
        if self.stats is not None:
            self.stats.note_hash_time(len(nonces), dt)
        return oks, dt

    def khs(self):
        return (self.total_hashed / self.total_time / 1000.0
                if self.total_time > 0 else 0.0)
