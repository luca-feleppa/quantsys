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
import json
import logging
import os
import shutil
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
    # IT: C4 (2026-07-18) — greeks straddle ATM ~30h (01c --greeks): 1 riga/tick.
    # EN: C4 (2026-07-18) — ATM ~30h straddle greeks (01c --greeks): 1 row/tick.
    (STAGING / "iv" / "atm_greeks.parquet", ROOT / "data" / "iv" / "atm_greeks.parquet",
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


# IT: ── vol-paper (04b sul VPS dal 2026-07-18) ─────────────────────────────
#     Il VPS è la sorgente AUTORITATIVA del forward test: sui duplicati vincono
#     le righe di staging (keep='last' con canonico prima — diverso dai
#     collector, dove le righe sono identiche per costruzione e keep='first'
#     equivale). I jsonl si uniscono per chiave; position/hedge_state si
#     SPECCHIANO in presenza (assente sul VPS = flat) solo col marker _pulled.ok.
# EN: ── vol-paper (04b on the VPS since 2026-07-18) ──────────────────────────
#     The VPS is the AUTHORITATIVE forward-test source: on duplicates the
#     staging rows win (keep='last' with canonical first — unlike collectors,
#     whose rows are identical by construction so keep='first' is equivalent).
#     Jsonl files are key-merged; position/hedge_state are PRESENCE-mirrored
#     (absent on VPS = flat) only under the _pulled.ok marker.
VP_STAGING = STAGING / "vol_paper"
VP_CANON = ROOT / "results" / "vol_paper"
VP_JSONL = [
    ("trades.jsonl", ("entry_ts", "settled_ts")),
    ("exec_diag.jsonl", ("ts",)),
    ("hedge_ledger.jsonl", ("ts",)),
]
VP_MIRROR = ["position.json", "hedge_state.json"]


def merge_jsonl(staged: Path, canon: Path, keys: tuple) -> tuple[int, int]:
    # IT: unione per chiave (canonico prima, staging vince sui duplicati),
    #     ordine per chiave; write atomica (.tmp + os.replace, safety net repo).
    # EN: key-based union (canonical first, staging wins on duplicates), sorted
    #     by key; atomic write (.tmp + os.replace, repo safety net).
    def _load(p: Path) -> list:
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    old, new = _load(canon), _load(staged)
    by_key = {tuple(str(r.get(k)) for k in keys): r for r in old}
    n_before = len(by_key)
    for r in new:
        by_key[tuple(str(r.get(k)) for k in keys)] = r
    rows = sorted(by_key.values(), key=lambda r: tuple(str(r.get(k)) for k in keys))
    if len(rows) != n_before or n_before == 0:
        canon.parent.mkdir(parents=True, exist_ok=True)
        tmp = canon.with_suffix(canon.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows),
                       encoding="utf-8")
        os.replace(tmp, canon)
    return n_before, len(rows)


def merge_vol_paper():
    if not VP_STAGING.exists():
        return
    # IT: forecasts: dedup su candle_ts, righe VPS vincenti (vedi header sezione).
    # EN: forecasts: dedup on candle_ts, VPS rows winning (see section header).
    staged_fc = VP_STAGING / "forecasts.parquet"
    if staged_fc.exists():
        new = pd.read_parquet(staged_fc)
        canon_fc = VP_CANON / "forecasts.parquet"
        if canon_fc.exists():
            old = pd.read_parquet(canon_fc)
            n_before = len(old)
            merged = pd.concat([old, new], ignore_index=True)
        else:
            n_before, merged = 0, new
        merged = (merged.drop_duplicates(subset="candle_ts", keep="last")
                        .sort_values("candle_ts").reset_index(drop=True))
        atomic_save_parquet(merged, canon_fc, index=False)
        log.info(f"merge vol_paper/forecasts: {n_before} → {len(merged)} righe/rows")
    for name, keys in VP_JSONL:
        staged = VP_STAGING / name
        if staged.exists():
            b, a = merge_jsonl(staged, VP_CANON / name, keys)
            log.info(f"merge vol_paper/{name}: {b} → {a} righe/rows (+{a - b})")
    # IT: mirror di presenza SOLO a pull riuscito (marker): copia o rimozione.
    # EN: presence mirror ONLY after a successful pull (marker): copy or delete.
    if (VP_STAGING / "_pulled.ok").exists():
        for name in VP_MIRROR:
            staged, canon = VP_STAGING / name, VP_CANON / name
            if staged.exists():
                canon.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged, canon)
            elif canon.exists():
                canon.unlink()
                log.info(f"vol_paper/{name}: assente sul VPS (flat) — rimosso in "
                         f"canonico / absent on VPS (flat) — canonical removed")


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
    # IT: 04b (VPS dal 2026-07-18): candle_ts è la candela CHIUSA del tick →
    #     età fisiologica ~1-2h; soglia 3.5h = tollera un tick fallito.
    # EN: 04b (VPS since 2026-07-18): candle_ts is the tick's CLOSED candle →
    #     physiological age ~1-2h; 3.5h threshold = tolerates one failed tick.
    if (VP_STAGING / "forecasts.parquet").exists():
        checks.append(("Vol-paper 04b (forecasts)", VP_STAGING / "forecasts.parquet",
                       "candle_ts", max(stale_hours, 3.5)))
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
    ap.add_argument("--keep-staging", action="store_true",
                    help="non pulire data/vps_staging dopo un merge sano (default: "
                         "pulito se heartbeat OK) / do not clean data/vps_staging "
                         "after a healthy merge (default: cleaned when heartbeat OK)")
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
    merge_vol_paper()

    fresh = heartbeat(args.stale_hours)
    # IT: igiene 2026-07-18 — lo staging duplica ~40 MB gia' fusi nei canonici:
    #     si pulisce DOPO l'heartbeat (che lo legge) e SOLO a heartbeat sano; su
    #     WARN resta su disco come evidenza di debug dell'ultimo pull.
    # EN: 2026-07-18 hygiene — staging duplicates ~40 MB already merged into the
    #     canonical copies: cleaned AFTER the heartbeat (which reads it) and ONLY
    #     when the heartbeat is healthy; on WARN it stays on disk as debug
    #     evidence of the last pull.
    if fresh and not args.keep_staging:
        shutil.rmtree(STAGING, ignore_errors=True)
        log.info("staging pulito (ricreato al prossimo pull) / staging cleaned "
                 "(recreated at next pull)")
    log.info("merge completato / merge done" + ("" if fresh else " — CON WARNING HEARTBEAT / WITH HEARTBEAT WARNINGS"))
    return 0 if fresh else 2


if __name__ == "__main__":
    sys.exit(main())
