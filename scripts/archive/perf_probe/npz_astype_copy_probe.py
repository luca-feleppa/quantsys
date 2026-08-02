"""Probe di memoria — `astype(np.float32)` vs `astype(np.float32, copy=False)`.

IT: Misura il PICCO di memoria privata del processo durante il caricamento di
    `X_train` dall'npz, nei due rami `copy=True` (comportamento attuale di
    `02_train.py`) e `copy=False` (candidato). Un processo per ramo: i contatori
    di picco Windows (`PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize`) sono
    MONOTONI sulla vita del processo e non azzerabili, quindi due misure nello
    stesso processo si contaminerebbero.
EN: Memory probe — peak private memory while loading `X_train` from the npz,
    under `copy=True` (current `02_train.py` behaviour) and `copy=False`
    (candidate). One process per branch: Windows peak counters
    (`PROCESS_MEMORY_COUNTERS_EX.PeakWorkingSetSize`) are MONOTONIC over the
    process lifetime and cannot be reset, so two measurements in the same
    process would contaminate each other.

Uso / usage:
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


# IT: struct Win32 per i contatori di memoria del processo. `PrivateUsage` (commit
#     privato corrente) e' l'unico campo ISTANTANEO utile: `PeakWorkingSetSize` e'
#     monotono e non distingue le fasi. Campionandolo da un thread otteniamo il
#     picco della SOLA operazione sotto misura.
# EN: Win32 process memory counters struct. `PrivateUsage` (current private commit)
#     is the only useful INSTANTANEOUS field: `PeakWorkingSetSize` is monotonic and
#     cannot separate phases. Sampling it from a thread yields the peak of the
#     operation under measurement alone.
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


# IT: restype ESPLICITO — senza di esso ctypes tratta lo pseudo-handle (-1) come
#     c_int e su 64-bit il valore passato alla API e' troncato: la chiamata
#     fallisce silenziosamente e la struct resta a zero (visto: baseline 0.000 GiB).
# EN: EXPLICIT restype — without it ctypes treats the pseudo-handle (-1) as c_int
#     and on 64-bit the value handed to the API is truncated: the call fails
#     silently and the struct stays zeroed (observed: baseline 0.000 GiB).
_k32 = ctypes.windll.kernel32
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.K32GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
_k32.K32GetProcessMemoryInfo.restype = wt.BOOL
_HPROC = _k32.GetCurrentProcess()


# IT: legge il commit privato corrente in byte; fail-fast se la API fallisce
#     (un contatore a zero e' indistinguibile da "nessuna allocazione").
# EN: read current private commit in bytes; fail-fast if the API fails (a zeroed
#     counter is indistinguishable from "no allocation").
def private_bytes() -> int:
    c = PROCESS_MEMORY_COUNTERS_EX()
    c.cb = ctypes.sizeof(c)
    if not _k32.K32GetProcessMemoryInfo(_HPROC, ctypes.byref(c), c.cb):
        raise OSError(f"K32GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return int(c.PrivateUsage)


# IT: campionatore ad alta frequenza su thread separato. `setswitchinterval` basso
#     forza il rilascio frequente del GIL, cosi' il campionamento non "salta" il
#     transitorio della copia (che dura ~1-2 s su 2.59 GB).
# EN: high-frequency sampler on a separate thread. A low `setswitchinterval` forces
#     frequent GIL handoff so sampling does not miss the copy transient (~1-2 s on
#     2.59 GB).
class PeakSampler(threading.Thread):
    def __init__(self, interval: float = 0.001):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.n = 0
        # IT: NON chiamarlo `_stop`: e' un metodo di threading.Thread e lo shadowing
        #     rompe join() ("'Event' object is not callable").
        # EN: do NOT name it `_stop`: it is a threading.Thread method and shadowing
        #     it breaks join() ("'Event' object is not callable").
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

    # IT: apertura npz (lazy: nessun membro materializzato).
    # EN: open npz (lazy: no member materialised yet).
    data = np.load(str(npz), allow_pickle=True)
    _ = data.files
    base = private_bytes()
    print(f"mode={mode}")
    print(f"  baseline (npz aperto, nessun array)      : {base/GB:7.3f} GiB")

    smp = PeakSampler()
    smp.start()
    t0 = time.perf_counter()
    # IT: la riga sotto e' esattamente `to_t("X_train")` di 02_train.py, col solo
    #     parametro `copy` variato. Il clamp_ successivo e' fuori misura (in-place,
    #     zero allocazione) ma lo eseguiamo per validare la scrittura sul buffer.
    # EN: the line below is exactly `to_t("X_train")` from 02_train.py with only the
    #     `copy` parameter varied. The later clamp_ is out of scope (in-place, zero
    #     allocation) but we run it to validate writing into the buffer.
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

    # IT: sanity — writeable + clamp_ in-place funziona senza copia.
    # EN: sanity — writeable + in-place clamp_ works without a copy.
    ptr_before = X.data_ptr()
    X.clamp_(-5.0, 5.0)
    print(f"  clamp_ in-place, ptr invariato            : {X.data_ptr() == ptr_before}")
    print(f"  post-clamp min/max                        : {float(X.min()):.3f} / {float(X.max()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
