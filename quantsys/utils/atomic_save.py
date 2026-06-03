"""
quantsys/utils/atomic_save.py
==============================
Salvataggio atomico per file dati critici (npz, parquet, pkl).

Il problema:
  np.savez_compressed("data/lstm_dataset.npz", ...) sovrascrive il file
  direttamente. Se il processo crasha a metà scrittura (OOM, Ctrl+C, 
  crash disco), il file risulta corrotto e si perde tutto il lavoro
  della Fase 1 (download + feature engineering).

La soluzione — write-then-rename:
  1. Scrivi i dati in un file temporaneo nella stessa directory
     (es. "data/lstm_dataset.npz.tmp")
  2. Rinomina atomicamente il tmp → file definitivo
  Su Linux/Mac os.replace() è garantito atomico (syscall rename(2))
  → il file è sempre o quello vecchio completo o quello nuovo completo,
     mai parziale.

Nota: non funziona cross-device (tmp e dst devono essere sullo stesso filesystem).
"""
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# IT: Helpers per scrittura atomica write-then-rename (mai file parziali).
# EN: Helpers for atomic write-then-rename (never produces partial files).


# IT: Salva array NumPy in .npz atomicamente (opz. compresso).
# EN: Atomically saves NumPy arrays to .npz (optionally compressed).
def atomic_save_npz(path: str | Path, compressed: bool = False, **arrays) -> None:
    """Salva un file npz in modo atomico (tmp + rename).

    compressed=False (default): ~5-10x più veloce, file più grande su disco.
    compressed=True: comprime con zlib (lento su dataset >1 GB).
    """
    # IT: tmp nella STESSA dir → os.replace atomico (cross-device fallisce).
    # EN: tmp in SAME dir → os.replace is atomic (cross-device would fail).
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir    = path.parent,
        suffix = ".npz",
        prefix = f".{path.stem}_",
    )
    try:
        os.close(tmp_fd)
        _save = np.savez_compressed if compressed else np.savez
        _save(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# IT: Salva un DataFrame in parquet atomicamente (tmp + rename).
# EN: Atomically saves a DataFrame to parquet (tmp + rename).
def atomic_save_parquet(df: Any, path: str | Path, **kwargs) -> None:
    """Salva un DataFrame parquet in modo atomico (tmp + rename)."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir    = path.parent,
        suffix = ".parquet.tmp",
        prefix = f".{path.stem}_",
    )
    try:
        os.close(tmp_fd)
        df.to_parquet(tmp_path, **kwargs)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# IT: Serializza un oggetto in pickle atomicamente (tmp + rename).
# EN: Atomically serializes an object to pickle (tmp + rename).
def atomic_save_pkl(obj: Any, path: str | Path) -> None:
    """Salva un oggetto pickle in modo atomico (tmp + rename)."""
    import pickle
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir    = path.parent,
        suffix = ".pkl.tmp",
        prefix = f".{path.stem}_",
    )
    try:
        os.close(tmp_fd)
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
