"""
IT: Probe temporaneo — dimostra su dati REALI che il pre-clip di 02_train.py
    ha lo stesso risultato bit-a-bit con Tensor.clamp() (out-of-place) e
    Tensor.clamp_() (in-place), e misura il picco di memoria delle due varianti.
EN: Temporary probe — proves on REAL data that the pre-clip in 02_train.py
    yields bit-identical results with Tensor.clamp() (out-of-place) and
    Tensor.clamp_() (in-place), and measures the peak memory of both variants.

IT: Uso (dalla root di progetto):
      python scripts/archive/perf_probe/clamp_inplace_probe.py equal
      python scripts/archive/perf_probe/clamp_inplace_probe.py peak oop
      python scripts/archive/perf_probe/clamp_inplace_probe.py peak inplace
    Le due misure di picco vanno lanciate in PROCESSI SEPARATI: il
    PeakWorkingSetSize di Windows e' monotono per processo, quindi due varianti
    nello stesso processo si contaminerebbero a vicenda.
EN: Usage (from the project root):
      python scripts/archive/perf_probe/clamp_inplace_probe.py equal
      python scripts/archive/perf_probe/clamp_inplace_probe.py peak oop
      python scripts/archive/perf_probe/clamp_inplace_probe.py peak inplace
    The two peak measurements must run in SEPARATE PROCESSES: the Windows
    PeakWorkingSetSize is monotonic per process, so two variants in the same
    process would contaminate each other.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import gc
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
NPZ = ROOT / "data" / "lstm_dataset.npz"


# ── Misura del working set via Win32 (psutil non e' installato nel venv) ──────
# IT: PROCESS_MEMORY_COUNTERS di psapi: WorkingSetSize (corrente) e
#     PeakWorkingSetSize (massimo raggiunto dal processo, monotono).
# EN: psapi PROCESS_MEMORY_COUNTERS: WorkingSetSize (current) and
#     PeakWorkingSetSize (process-lifetime maximum, monotonic).
class _PMC(ctypes.Structure):
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
    ]


# IT: argtypes/restype espliciti: il default ctypes (c_int) tronca l'HANDLE a
#     64 bit e la chiamata fallisce.
# EN: explicit argtypes/restype: the ctypes default (c_int) truncates the 64-bit
#     HANDLE and the call fails.
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32.GetCurrentProcess.restype = wt.HANDLE
_k32.GetCurrentProcess.argtypes = []
_psapi.GetProcessMemoryInfo.restype = wt.BOOL
_psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]


def _mem() -> tuple[int, int, int, int]:
    """
    IT: (WS corrente, WS picco, commit privato corrente, commit privato picco) in byte.
        Il WORKING SET e' inaffidabile come misura di allocazione: Windows lo
        trimma in modo asincrono e le pagine non ancora toccate non ci compaiono.
        Il COMMIT PRIVATO (PagefileUsage) e' invece quello che il processo ha
        effettivamente riservato: e' la misura giusta per un picco di allocazione.
    EN: (current WS, peak WS, current private commit, peak private commit) in bytes.
        The WORKING SET is unreliable as an allocation measure: Windows trims it
        asynchronously and untouched pages never show up there. The PRIVATE
        COMMIT (PagefileUsage) is what the process actually reserved: the right
        counter for an allocation peak.
    """
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return (pmc.WorkingSetSize, pmc.PeakWorkingSetSize,
            pmc.PagefileUsage, pmc.PeakPagefileUsage)


# ── Lettura parziale del membro npz ──────────────────────────────────────────
def load_slice(n: int) -> np.ndarray:
    """
    IT: legge i primi n campioni di X_train dal membro .npy dentro lo zip senza
        materializzare i 2.59 GB dell'array intero (i membri sono ZIP_STORED).
    EN: reads the first n samples of X_train from the .npy member inside the zip
        without materialising the whole 2.59 GB array (members are ZIP_STORED).
    """
    with zipfile.ZipFile(NPZ) as z, z.open("X_train.npy") as f:
        major, minor = np.lib.format.read_magic(f)
        # IT: numpy 2.x non espone piu' _read_array_header — usa il reader versionato.
        # EN: numpy 2.x no longer exposes _read_array_header — use the versioned reader.
        _hdr = {(1, 0): np.lib.format.read_array_header_1_0,
                (2, 0): np.lib.format.read_array_header_2_0}[(major, minor)]
        shape, fortran, dtype = _hdr(f)
        assert not fortran, "unexpected Fortran order"
        # IT: readinto su un array preallocato — UNA sola allocazione, cosi' la
        #     fase di caricamento non gonfia il picco che vogliamo attribuire al clamp.
        # EN: readinto a preallocated array — a SINGLE allocation, so the load
        #     phase does not inflate the peak we want to attribute to the clamp.
        out = np.empty((n,) + tuple(shape[1:]), dtype=dtype)
        got = f.readinto(memoryview(out).cast("B"))
        assert got == out.nbytes, f"short read: {got} != {out.nbytes}"
    return out


def clip_bounds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IT: stessi bound di 02_train.py (p0.1/p99.9 per-feature).
    EN: same bounds as 02_train.py (per-feature p0.1/p99.9)."""
    flat = x.reshape(-1, x.shape[2])
    lo = np.nanpercentile(flat, 0.1, axis=0).astype(np.float32)
    hi = np.nanpercentile(flat, 99.9, axis=0).astype(np.float32)
    return lo, hi


# ── Prova di equivalenza bit-a-bit ───────────────────────────────────────────
def run_equal(n: int) -> int:
    x = load_slice(n)
    lo, hi = clip_bounds(x)
    lo_t, hi_t = torch.from_numpy(lo), torch.from_numpy(hi)

    # IT: due copie INDIPENDENTI degli stessi dati (astype copia sempre, come in
    #     02_train.py) — una per variante, cosi' l'in-place non contamina l'altra.
    # EN: two INDEPENDENT copies of the same data (astype always copies, as in
    #     02_train.py) — one per variant, so the in-place run cannot taint the other.
    a = torch.from_numpy(x.astype(np.float32))
    b = torch.from_numpy(x.astype(np.float32))
    assert a.data_ptr() != b.data_ptr()

    a_out = a.clamp(lo_t, hi_t)          # out-of-place (variante attuale / current)
    b_ptr = b.data_ptr()
    b.clamp_(lo_t, hi_t)                 # in-place (variante proposta / proposed)

    # IT: conteggio esatto degli elementi effettivamente clippati — senza questo
    #     la prova sarebbe vacua (due no-op sono banalmente uguali).
    # EN: exact count of actually clipped elements — without it the test would be
    #     vacuous (two no-ops are trivially equal).
    src = torch.from_numpy(x.astype(np.float32))
    n_clipped = int((a_out != src).sum().item())
    n_diff = int((a_out != b).sum().item())

    print(f"slice           : {tuple(a.shape)}  ({a.nelement()*4/1e9:.4f} GB)")
    print(f"elementi totali : {a_out.nelement()}")
    print(f"elementi clipped: {n_clipped}")
    print(f"elementi diversi tra clamp e clamp_: {n_diff}")
    print(f"torch.equal(out_of_place, in_place): {torch.equal(a_out, b)}")
    print(f"in-place ha riusato lo storage     : {b.data_ptr() == b_ptr}")
    print(f"out-of-place ha allocato nuovo     : {a_out.data_ptr() != a.data_ptr()}")
    print(f"max |diff| = {float((a_out - b).abs().max().item()):.17g}")
    return 0 if (torch.equal(a_out, b) and n_clipped > 0) else 1


# ── Misura del picco ─────────────────────────────────────────────────────────
def run_peak(n: int, variant: str) -> int:
    # IT: il PeakWorkingSetSize di Windows NON e' azzerabile (nessuna API lo
    #     resetta), quindi la fase di caricamento e' costruita per restare SOTTO
    #     il picco del clamp: una sola allocazione da n campioni, bound calcolati
    #     su una testa piccola, nessun astype intermedio. La differenza fra le
    #     due varianti resta quindi interamente attribuibile al clamp.
    # EN: the Windows PeakWorkingSetSize CANNOT be reset (no API does it), so the
    #     load phase is built to stay BELOW the clamp peak: one single n-sample
    #     allocation, bounds from a small head, no intermediate astype. The gap
    #     between the two variants is therefore entirely due to the clamp.
    x = load_slice(n)
    lo, hi = clip_bounds(x[:2000])
    lo_t, hi_t = torch.from_numpy(lo), torch.from_numpy(hi)
    t = torch.from_numpy(x)
    del x
    gc.collect()

    # IT: i contatori di PICCO di Windows (PeakWorkingSetSize, PeakPagefileUsage)
    #     sono monotoni sull'intera vita del processo e NON azzerabili: la fase di
    #     caricamento li ha gia' spinti sopra il picco del clamp, quindi sono
    #     inutilizzabili qui. Campioniamo invece il commit privato CORRENTE da un
    #     thread ad alta frequenza mentre gira il clamp: misura diretta del transitorio.
    # EN: the Windows PEAK counters (PeakWorkingSetSize, PeakPagefileUsage) are
    #     monotonic over the whole process lifetime and cannot be reset: the load
    #     phase already pushed them above the clamp peak, so they are useless here.
    #     Instead we sample the CURRENT private commit from a high-frequency thread
    #     while the clamp runs: a direct measurement of the transient.
    samples: list[int] = []
    stop = threading.Event()

    def _sampler() -> None:
        while not stop.is_set():
            samples.append(_mem()[2])

    base = _mem()[2]
    th = threading.Thread(target=_sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    if variant == "oop":
        t = t.clamp(lo_t, hi_t)          # IT/EN: alloca un tensore nuovo / allocates a new tensor
    elif variant == "inplace":
        t.clamp_(lo_t, hi_t)             # IT/EN: scrive sullo storage esistente / writes in existing storage
    else:
        raise SystemExit(f"variante ignota: {variant}")
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    after = _mem()[2]

    size_gb = t.nelement() * 4 / 1e9
    hi_s = max(samples) if samples else base
    print(f"variant={variant}  tensor={tuple(t.shape)} ({size_gb:.4f} GB)")
    print(f"  durata clamp   = {dt*1e3:8.1f} ms   ({len(samples)} campioni di memoria)")
    print(f"  priv baseline  = {base/1e6:10.1f} MB")
    print(f"  priv MAX (in-op)= {hi_s/1e6:9.1f} MB   -> extra = {(hi_s-base)/1e6:8.1f} MB")
    print(f"  priv dopo      = {after/1e6:10.1f} MB")
    print(f"  checksum = {float(t.sum().item()):.6f}")
    return 0


def _sampled(fn):
    """
    IT: esegue fn() campionando il commit privato da un thread — ritorna
        (risultato, extra_byte di picco, durata_s).
    EN: runs fn() while a thread samples the private commit — returns
        (result, peak extra bytes, elapsed_s).
    """
    samples: list[int] = []
    stop = threading.Event()

    def _s() -> None:
        while not stop.is_set():
            samples.append(_mem()[2])

    base = _mem()[2]
    th = threading.Thread(target=_s, daemon=True)
    th.start()
    t0 = time.perf_counter()
    res = fn()
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    return res, (max(samples) if samples else base) - base, dt


def run_bounds(n: int) -> int:
    """
    IT: misura il costo in memoria di np.nanpercentile su X_train appiattito —
        e' l'altra allocazione grande dello stesso blocco di 02_train.py
        (linee _clip_lo/_clip_hi). NON e' oggetto del fix: solo diagnosi.
    EN: measures the memory cost of np.nanpercentile over flattened X_train —
        the other large allocation in the same 02_train.py block (the
        _clip_lo/_clip_hi lines). NOT part of the fix: diagnosis only.
    """
    x = load_slice(n)
    flat = x.reshape(-1, x.shape[2])     # IT/EN: vista, non copia / a view, not a copy
    gc.collect()
    print(f"array = {x.nbytes/1e6:.1f} MB  flat={flat.shape}  view={flat.base is not None}")
    _, extra, dt = _sampled(lambda: np.nanpercentile(flat, 0.1, axis=0))
    print(f"  nanpercentile(0.1)  : extra picco = {extra/1e6:8.1f} MB  "
          f"({extra/x.nbytes:5.2f}x array)  in {dt:.1f}s")
    _, extra, dt = _sampled(lambda: np.nanpercentile(flat, 99.9, axis=0))
    print(f"  nanpercentile(99.9) : extra picco = {extra/1e6:8.1f} MB  "
          f"({extra/x.nbytes:5.2f}x array)  in {dt:.1f}s")
    return 0


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    mode = sys.argv[1] if len(sys.argv) > 1 else "equal"
    if mode == "equal":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
        return run_equal(n)
    if mode == "peak":
        variant = sys.argv[2] if len(sys.argv) > 2 else "oop"
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
        return run_peak(n, variant)
    if mode == "bounds":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
        return run_bounds(n)
    raise SystemExit(f"modo ignoto: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
