"""
_chain_io.py — shared Deribit option-chain loader (vol line).

Single source of truth to read `data/iv/chain/*.parquet` (each vol script had its own slightly
different copy = code review item A3). Concatenates all parquet + converts timestamps to UTC datetime.
`lru_cache` keyed on the files' (name,mtime) → within one process the chain isn't re-read/re-concatenated
repeatedly (e.g. a script using it in several places). ⚠ Across-process gives no benefit
(scripts = separate processes): that would need a combined parquet (out of scope).

The returned DataFrame must be treated READ-ONLY (it is the cached object): consumers only do
boolean indexing/groupby (which copy), none reassigns columns on the whole frame.
"""
import functools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHAIN_DIR = ROOT / "data" / "iv" / "chain"


@functools.lru_cache(maxsize=4)
def _load_cached(_sig):
    # _sig only keys the cache.
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    return df


def load_chain():
    # full chain (snapshot_ts/expiry already UTC datetime), cached on parquet mtimes.
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    sig = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    df = _load_cached(sig)
    if df is None:
        sys.exit("no chain parquet in data/iv/chain")
    return df
