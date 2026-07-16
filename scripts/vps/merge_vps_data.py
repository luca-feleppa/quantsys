# IT: MERGE DATI VPS → COPIA CANONICA (lato casa). Riconcilia i file scaricati
#     da pull_vps_data.ps1 (data/vps_staging/) dentro data/iv/ e data/orderbook/:
#       - atm_30h.parquet / dvol.parquet : concat + dedup su timestamp + sort
#       - iv/chain/*.parquet (giornalieri): dedup su (snapshot_ts, instrument_name)
#       - orderbook/*.parquet (giornalieri): dedup su (timestamp, symbol)
#       - deribit_trades/*.parquet (giornalieri): dedup su trade_id
#     Il doppio poller (casa acceso + VPS) produce tick duplicati by design:
#     il dedup è la semantica di merge, non un workaround. Scritture atomiche
#     (atomic_save_parquet: tmp + os.replace — safety net CLAUDE.md).
#     HEARTBEAT: avvisa se l'ultimo tick VPS (file di staging, NON il merged)
#     è più vecchio di --stale-hours → collector giù sul VPS.
# EN: VPS DATA MERGE → CANONICAL COPY (home side). Reconciles the files pulled
#     by pull_vps_data.ps1 (data/vps_staging/) into data/iv/ and data/orderbook/:
#       - atm_30h.parquet / dvol.parquet : concat + dedup on timestamp + sort
#       - iv/chain/*.parquet (dailies): dedup on (snapshot_ts, instrument_name)
#       - orderbook/*.parquet (dailies): dedup on (timestamp, symbol)
#       - deribit_trades/*.parquet (dailies): dedup on trade_id
#     Dual polling (home on + VPS) duplicates ticks by design: dedup IS the
#     merge semantics, not a workaround. Atomic writes (atomic_save_parquet:
#     tmp + os.replace — CLAUDE.md safety net).
#     HEARTBEAT: warns when the latest VPS tick (staging file, NOT the merged
#     one) is older than --stale-hours → collector down on the VPS.
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging                      # noqa: E402
from quantsys.utils.atomic_save import atomic_save_parquet    # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vps_merge")

# IT: root di progetto (scripts/vps/ → 2 livelli sopra); path CWD-indipendenti.
# EN: project root (scripts/vps/ → 2 levels up); CWD-independent paths.
ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data" / "vps_staging"

# IT: mappa merge — (staging, canonico, chiavi dedup, colonna tempo per il sort).
# EN: merge map — (staging, canonical, dedup keys, time column for sorting).
SINGLE_FILES = [
    (STAGING / "iv" / "atm_30h.parquet", ROOT / "data" / "iv" / "atm_30h.parquet",
     ["timestamp"], "timestamp"),
    (STAGING / "iv" / "dvol.parquet", ROOT / "data" / "iv" / "dvol.parquet",
     ["timestamp"], "timestamp"),
]
DAILY_DIRS = [
    (STAGING / "iv" / "chain", ROOT / "data" / "iv" / "chain",
     ["snapshot_ts", "instrument_name"], "snapshot_ts"),
    (STAGING / "orderbook", ROOT / "data" / "orderbook",
     ["timestamp", "symbol"], "timestamp"),
    # IT: trade opzioni Deribit (01e): trade_id è la chiave naturale del venue.
    # EN: Deribit option trades (01e): trade_id is the venue's natural key.
    (STAGING / "deribit_trades", ROOT / "data" / "deribit_trades",
     ["trade_id"], "timestamp"),
]


def merge_one(staged: Path, canon: Path, keys: list[str], ts_col: str) -> tuple[int, int]:
    # IT: unisce UNO staged nel canonico: se il canonico manca è una copia pura;
    #     altrimenti concat (canonico prima → keep='first' preserva la riga di
    #     casa sui duplicati, contenuto comunque identico) + dedup + sort.
    #     Ritorna (righe canoniche prima, dopo) per il log.
    # EN: merges ONE staged file into the canonical one: pure copy if the
    #     canonical is missing; else concat (canonical first → keep='first'
    #     keeps the home row on duplicates, content identical anyway) + dedup
    #     + sort. Returns (canonical rows before, after) for logging.
    new = pd.read_parquet(staged)
    if canon.exists():
        old = pd.read_parquet(canon)
        n_before = len(old)
        merged = pd.concat([old, new], ignore_index=True)
    else:
        n_before = 0
        merged = new
    merged = (merged.drop_duplicates(subset=keys, keep="first")
                    .sort_values(ts_col).reset_index(drop=True))
    if len(merged) != n_before or n_before == 0:
        canon.parent.mkdir(parents=True, exist_ok=True)
        atomic_save_parquet(merged, canon)
    return n_before, len(merged)


def heartbeat(stale_hours: float) -> bool:
    # IT: staleness del VPS misurata sui file di STAGING (fotografano lo stato
    #     del collector remoto al momento del pull). True = tutto fresco.
    # EN: VPS staleness measured on the STAGING files (they snapshot the remote
    #     collector state at pull time). True = everything fresh.
    ok = True
    now = pd.Timestamp.now(tz="UTC")
    # IT: soglia per-check: i trade 01e si scrivono solo quando il mercato stampa
    #     fill → soglia larga (≥6h) anti falsi-WARN (audit 2026-07-16, MINOR:
    #     senza questo check un 01e morto era invisibile al pull quotidiano,
    #     e la retention API ~24h rende il buco NON ricostruibile).
    # EN: per-check threshold: 01e trades are written only when the market
    #     prints fills → wide threshold (≥6h) against false WARNs (2026-07-16
    #     audit, MINOR: without this check a dead 01e was invisible to the
    #     daily pull, and the ~24h API retention makes the gap unrecoverable).
    checks = [("IV poller (atm_30h)", STAGING / "iv" / "atm_30h.parquet", "timestamp", stale_hours)]
    ob_files = sorted((STAGING / "orderbook").glob("*.parquet"))
    if ob_files:
        checks.append(("L2 recorder (orderbook)", ob_files[-1], "timestamp", stale_hours))
    tr_files = sorted((STAGING / "deribit_trades").glob("*.parquet"))
    if tr_files:
        checks.append(("Trades recorder (deribit_trades)", tr_files[-1], "timestamp",
                       max(stale_hours, 6.0)))
    for name, path, col, thresh in checks:
        if not path.exists():
            log.warning(f"HEARTBEAT {name}: file di staging assente / staging file missing")
            ok = False
            continue
        last = pd.to_datetime(pd.read_parquet(path, columns=[col])[col].max(), utc=True)
        age_h = (now - last).total_seconds() / 3600
        if age_h > thresh:
            log.warning(f"HEARTBEAT {name}: ultimo tick VPS {age_h:.1f}h fa (> {thresh}h) "
                        f"— collector remoto probabilmente GIU' / remote collector likely DOWN")
            ok = False
        else:
            log.info(f"HEARTBEAT {name}: fresco/fresh ({age_h:.1f}h)")
    return ok


def main() -> int:
    # IT: boilerplate UTF-8 console Windows (checklist CLAUDE.md — bug cp1252).
    # EN: Windows console UTF-8 boilerplate (CLAUDE.md checklist — cp1252 bug).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Merge dati VPS → copia canonica / VPS data merge")
    ap.add_argument("--stale-hours", type=float, default=3.0,
                    help="soglia heartbeat staleness VPS (ore) / VPS staleness threshold (hours)")
    args = ap.parse_args()

    if not STAGING.exists():
        log.error(f"staging assente/missing: {STAGING} — lancia prima pull_vps_data.ps1")
        return 1

    # IT: 1) file singoli append-only.  2) giornalieri per directory.
    # EN: 1) append-only single files.  2) per-directory dailies.
    for staged, canon, keys, ts in SINGLE_FILES:
        if staged.exists():
            b, a = merge_one(staged, canon, keys, ts)
            log.info(f"merge {canon.relative_to(ROOT)}: {b} → {a} righe/rows (+{a - b})")
    for sdir, cdir, keys, ts in DAILY_DIRS:
        for staged in sorted(sdir.glob("*.parquet")) if sdir.exists() else []:
            b, a = merge_one(staged, cdir / staged.name, keys, ts)
            if a != b:
                log.info(f"merge {(cdir / staged.name).relative_to(ROOT)}: {b} → {a} (+{a - b})")

    fresh = heartbeat(args.stale_hours)
    log.info("merge completato / merge done" + ("" if fresh else " — CON WARNING HEARTBEAT / WITH HEARTBEAT WARNINGS"))
    return 0 if fresh else 2


if __name__ == "__main__":
    sys.exit(main())
