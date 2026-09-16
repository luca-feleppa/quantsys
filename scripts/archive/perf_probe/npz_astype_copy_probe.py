"""Memory probe — `astype(np.float32)` vs `astype(np.float32, copy=False)`.

Measures the process's PEAK private memory while loading `X_train` from the
npz, under `copy=True` (current `02_train.py` behaviour) and `copy=False`
(candidate). One process per branch: Windows peak counters
(`PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize`) are MONOTONIC over the
process lifetime and cannot be reset, so two measurements in the same
process would contaminate each other.

Usage:
    python scripts/archive/perf_probe/npz_astype_copy_probe.py copy_true
    python scripts/archive/perf_probe/npz_astype_copy_probe.py copy_false
"""
import ctypes
import ctypes.wintypes as wt
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch


# Win32 process memory counters struct. `PrivateUsage` (current private commit)
# is the only useful INSTANTANEOUS field: `PeakWorkingSetSize` is monotonic and
# cannot separate phases. Sampling it from a thread yields the peak of the
# operation under measurement alone.
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


# EXPLICIT restype — without it ctypes treats the pseudo-handle (-1) as c_int
# and on 64-bit the value handed to the API is truncated: the call fails
# silently and the struct stays zeroed (observed: baseline 0.000 GiB).
_k32 = ctypes.windll.kernel32
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
_k32.K32GetProcessMemoryInfo.restype = wt.BOOL
_HPROC = _k32.GetCurrentProcess()


# read current private commit in bytes; fail-fast if the API fails (a zeroed
# counter is indistinguishable from "no allocation").
def private_bytes() -> int:
    c = PROCESS_MEMORY_COUNTERS_EX()
    c.cb = ctypes.sizeof(c)
    if not _k32.K32GetProcessMemoryInfo(_HPROC, ctypes.byref(c), c.cb):
        raise OSError(f"K32GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return int(c.PrivateUsage)


# high-frequency sampler on a separate thread. A low `setswitchinterval` forces
# frequent GIL handoff so sampling does not miss the copy transient (~1-2 s on
# 2.59 GB).
class PeakSampler(threading.Thread):
    def __init__(self, interval: float = 0.001):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.n = 0
        # do NOT name it `_stop`: it is a threading.Thread method and shadowing
        # it breaks join() ("'Event' object is not callable").
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            v = private_bytes()
            self.n += 1
            if v > self.peak:
                self.peak = v
            time.sleep(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5.0)


GB = 1024 ** 3


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "copy_true"
    assert mode in ("copy_true", "copy_false"), mode
    copy_flag = (mode == "copy_true")

    npz = Path("data/lstm_dataset.npz")
    if not npz.exists():
        print(f"MISSING {npz} — run from project root")
        return 2

    sys.setswitchinterval(0.0005)

    # open npz (lazy: no member materialised yet).
    data = np.load(str(npz), allow_pickle=True)
    _ = data.files
    base = private_bytes()
    print(f"mode={mode}")
    print(f"  baseline (npz aperto, nessun array)      : {base/GB:7.3f} GiB")

    smp = PeakSampler()
    smp.start()
    t0 = time.perf_counter()
    # the line below is exactly `to_t("X_train")` from 02_train.py with only the
    # `copy` parameter varied. The later clamp_ is out of scope (in-place, zero
    # allocation) but we run it to validate writing into the buffer.
    if copy_flag:
        X = torch.from_numpy(data["X_train"].astype(np.float32))
    else:
        X = torch.from_numpy(data["X_train"].astype(np.float32, copy=False))
    dt = time.perf_counter() - t0
    smp.stop()

    after = private_bytes()
    nbytes = X.element_size() * X.nelement()
    print(f"  peak DURANTE il load (n={smp.n} campioni)  : {smp.peak/GB:7.3f} GiB")
    print(f"  steady-state dopo il load                 : {after/GB:7.3f} GiB")
    print(f"  tensore X_train                           : {nbytes/GB:7.3f} GiB")
    print(f"  DELTA peak-baseline                       : {(smp.peak-base)/GB:7.3f} GiB "
          f"({(smp.peak-base)/nbytes:.2f}x la dimensione del tensore)")
    print(f"  DELTA steady-baseline                     : {(after-base)/GB:7.3f} GiB")
    print(f"  tempo load                                : {dt:7.3f} s")

    # sanity — writeable + in-place clamp_ works without a copy.
    ptr_before = X.data_ptr()
    X.clamp_(-5.0, 5.0)
    print(f"  clamp_ in-place, ptr invariato            : {X.data_ptr() == ptr_before}")
    print(f"  post-clamp min/max                        : {float(X.min()):.3f} / {float(X.max()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
