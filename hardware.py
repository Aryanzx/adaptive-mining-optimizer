"""hardware.py -- hardware monitor (section 21), CPU only.

Stdlib-only monitoring: process CPU fraction, system load (Linux
/proc/stat if present), peak RSS, thermal zone temperature, thread count.
Produces a normalized [0,1] hardware score for context F and suggests
batch-size adjustments.  All reads are guarded: on embedded targets
without /proc or resource, values degrade to None gracefully.
"""

import os
import threading
import time

try:
    import resource
except ImportError:                       # pragma: no cover (non-POSIX)
    resource = None

__all__ = ["HardwareMonitor"]


class HardwareMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cpu_count = os.cpu_count() or 1
        self.cpu_frac = 0.0
        self.sys_load = None
        self.temp_c = None
        self.rss_kb = None
        self.power_w = None
        self._p0 = time.process_time()
        self._w0 = time.perf_counter()
        self._stat0 = self._read_proc_stat()
        self._temp_at = 0.0

    def start(self):
        self._p0 = time.process_time()
        self._w0 = time.perf_counter()
        self._stat0 = self._read_proc_stat()

    @staticmethod
    def _read_proc_stat():
        try:
            with open("/proc/stat", "rb") as f:
                parts = f.readline().split()
            vals = [int(x) for x in parts[1:8]]
            idle = vals[3] + vals[4]
            total = sum(vals)
            return (idle, total)
        except Exception:
            return None

    def _read_temp(self):
        now = time.perf_counter()
        if now - self._temp_at < 5.0:
            return self.temp_c
        self._temp_at = now
        try:
            base = "/sys/class/thermal/"
            zones = [z for z in os.listdir(base)
                     if z.startswith("thermal_zone")]
            best = None
            for z in zones:
                try:
                    with open(base + z + "/temp") as f:
                        t = int(f.read().strip()) / 1000.0
                    best = t if best is None else max(best, t)
                except Exception:
                    pass
            self.temp_c = best
        except Exception:
            self.temp_c = None
        return self.temp_c

    def _read_power(self):
        try:
            base = "/sys/class/power_supply/"
            for dev in os.listdir(base):
                p = base + dev + "/power_now"
                if os.path.exists(p):
                    with open(p) as f:
                        return int(f.read().strip()) / 1_000_000.0
        except Exception:
            pass
        return None

    def sample(self):
        proc = time.process_time()
        wall = time.perf_counter()
        dw = wall - self._w0
        if dw > 0:
            frac = (proc - self._p0) / dw
            self.cpu_frac = 0.9 * self.cpu_frac + 0.1 * min(1.5, frac)
        self._p0 = proc
        self._w0 = wall
        st = self._read_proc_stat()
        if st and self._stat0:
            didle = st[0] - self._stat0[0]
            dtot = st[1] - self._stat0[1]
            if dtot > 0:
                self.sys_load = 1.0 - didle / dtot
            self._stat0 = st
        if resource is not None:
            self.rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._read_temp()
        self.power_w = self._read_power()
        return {"cpu_frac": self.cpu_frac, "sys_load": self.sys_load,
                "rss_kb": self.rss_kb, "temp_c": self.temp_c,
                "power_w": self.power_w,
                "threads": threading.active_count(),
                "cpu_count": self.cpu_count}

    def load(self):
        return self.sys_load if self.sys_load is not None else self.cpu_frac

    def score(self):
        """Context F score: 1.0 = idle/roomy, 0.0 = saturated."""
        if self.cfg.deterministic:
            return 0.5
        return min(1.0, max(0.0, 1.0 - self.load()))

    def suggest_batch(self, batch):
        """Keep CPU busy but not throttling (section 21, 28)."""
        if self.cfg.deterministic:
            return batch
        ld = self.load()
        if ld > 0.93:
            return max(self.cfg.batch_min, int(batch * 0.75))
        if ld < 0.60:
            return min(self.cfg.batch_max, int(batch * 1.25) + 1)
        return batch
