"""
_chain_io.py — loader condiviso della chain opzioni Deribit (vol-line).
_chain_io.py — shared Deribit option-chain loader (vol line).

IT: Single source of truth per leggere `data/iv/chain/*.parquet` (prima ogni script vol aveva la
    sua copia leggermente diversa = code review A3). Concatena tutti i parquet + converte i timestamp
    a datetime UTC. `lru_cache` su (nome,mtime) dei file → entro lo stesso processo la chain non viene
    riletta/riconcatenata più volte (es. uno script che la usa in più punti). ⚠ Across-process NON
    aiuta (script = processi separati): per quello servirebbe un parquet combinato (fuori scope).
EN: Single source of truth to read `data/iv/chain/*.parquet` (each vol script had its own slightly
    different copy = review item A3). Concatenates all parquet + converts timestamps to UTC datetime.
    `lru_cache` keyed on (name,mtime) → within one process the chain isn't re-read/re-concatenated
    repeatedly. ⚠ Across-process gives no benefit (scripts = separate processes).

IT: Il DataFrame ritornato va trattato READ-ONLY (è l'oggetto in cache): i consumer fanno solo
    indexing booleano/groupby (creano copie), nessuno riassegna colonne sull'intero frame.
EN: The returned DataFrame must be treated READ-ONLY (it is the cached object): consumers only do
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
    # IT: _sig (nome,mtime) entra solo per invalidare la cache se i file cambiano. | EN: _sig only keys the cache.
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    return df


def load_chain():
    # IT: chain completa (snapshot_ts/expiry già datetime UTC), cache-ata su mtime dei parquet.
    # EN: full chain (snapshot_ts/expiry already UTC datetime), cached on parquet mtimes.
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    sig = tuple((f.name, f.stat().st_mtime_ns) for f in files)
    df = _load_cached(sig)
    if df is None:
        sys.exit("no chain parquet in data/iv/chain")
    return df
