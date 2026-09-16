"""
quantsys/utils/atomic_save.py
==============================
Atomic saving for critical data files (npz, parquet, pkl).

The problem:
  np.savez_compressed("data/lstm_dataset.npz", ...) overwrites the file
  directly. If the process crashes mid-write (OOM, Ctrl+C,
  disk failure), the file ends up corrupted and all of Phase 1's
  work (download + feature engineering) is lost.

The solution — write-then-rename:
  1. Write the data to a temporary file in the same directory
     (e.g. "data/lstm_dataset.npz.tmp")
  2. Atomically rename the tmp → final file
  On Linux/Mac os.replace() is guaranteed atomic (rename(2) syscall)
  → the file is always either the complete old one or the complete new one,
     never partial.

Note: does not work cross-device (tmp and dst must be on the same filesystem).
"""
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


# Helpers for atomic write-then-rename (never produces partial files).


# Atomically saves NumPy arrays to .npz (optionally compressed).
def atomic_save_npz(path: str | Path, compressed: bool = False, **arrays) -> None:
    """Saves an npz file atomically (tmp + rename).

    compressed=False (default): ~5-10x faster, larger file on disk.
    compressed=True: compresses with zlib (slow on datasets >1 GB).
    """
    # tmp in SAME dir → os.replace is atomic (cross-device would fail).
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


# Atomically saves a DataFrame to parquet (tmp + rename).
def atomic_save_parquet(df: Any, path: str | Path, **kwargs) -> None:
    """Saves a DataFrame to parquet atomically (tmp + rename)."""
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


# Atomically serializes an object to pickle (tmp + rename).
def atomic_save_pkl(obj: Any, path: str | Path) -> None:
    """Saves an object to pickle atomically (tmp + rename)."""
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
