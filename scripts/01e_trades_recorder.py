# IT: Recorder trade opzioni Deribit (production, endpoint PUBBLICO no-auth) —
#     raccolta FORWARD dei trade eseguiti sulla chain opzioni BTC per la stima
#     degli spread REALIZZATI vs mark (costi eseguibili del braccio short-vol,
#     uso post-gate-v1). Razionale (verifica 2026-07-16): la retention pubblica
#     di public/get_last_trades_by_currency_and_time è ~24h — anche con
#     include_old=true e anche per singolo strumento scaduto tornano 0 trade
#     oltre la finestra → lo storico NON è ricostruibile ex-post gratis (vendor
#     a pagamento: Tardis/Amberdata), quindi si qualifica per la raccolta
#     continua come IV (01c) e L2 (01d).
#     Feed: public/get_last_trades_by_currency_and_time (currency=BTC,
#     kind=option, count≤1000, sorting=asc) con paginazione su has_more:
#     la finestra [ultimo ts salvato − overlap, now] viene percorsa in avanti
#     avanzando start_timestamp all'ultimo trade della pagina; i trade duplicati
#     al bordo (stesso ms) sono assorbiti dal dedup su trade_id. Cold start:
#     backfill dell'intera retention (~24h). Payload per-trade: price, amount,
#     contracts, direction, iv, mark_price, index_price → spread realizzato
#     vs mark calcolabile direttamente per ogni fill.
#     Output append-only ATOMICO (tmp+os.replace, dedup su trade_id) in
#     data/deribit_trades/:
#       option_trades_YYYYMMDD.parquet — 1 riga/trade, file per GIORNO UTC del
#                                        trade (non del tick di poll).
#     Modalità: --once (smoke), --minutes N (cadenza, default 10), --currency.
# EN: Deribit option trades recorder (production, PUBLIC no-auth endpoint) —
#     FORWARD collection of executed trades on the BTC option chain to estimate
#     REALIZED spreads vs mark (executable costs of the short-vol arm,
#     post-v1-gate use). Rationale (verified 2026-07-16): the public retention
#     of public/get_last_trades_by_currency_and_time is ~24h — even with
#     include_old=true and even per expired instrument, 0 trades come back
#     beyond the window → history is NOT freely reconstructible ex-post (paid
#     vendors: Tardis/Amberdata), so it qualifies for continuous collection
#     like IV (01c) and L2 (01d).
#     Feed: public/get_last_trades_by_currency_and_time (currency=BTC,
#     kind=option, count≤1000, sorting=asc) paginated on has_more: the window
#     [last saved ts − overlap, now] is walked forward advancing
#     start_timestamp to the page's last trade; boundary duplicates (same ms)
#     are absorbed by the trade_id dedup. Cold start: backfill of the whole
#     retention (~24h). Per-trade payload: price, amount, contracts, direction,
#     iv, mark_price, index_price → realized spread vs mark directly computable
#     per fill.
#     Append-only ATOMIC output (tmp+os.replace, dedup on trade_id) under
#     data/deribit_trades/:
#       option_trades_YYYYMMDD.parquet — 1 row/trade, file per UTC DAY of the
#                                        trade (not of the poll tick).
#     Modes: --once (smoke), --minutes N (cadence, default 10), --currency.
import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantsys.utils import setup_logging                      # noqa: E402
from quantsys.utils.collect import append_parquet             # noqa: E402
# IT: helper Deribit condivisi con 01c (estratti 2026-07-16; production, no-auth:
#     la testnet NON va usata qui — trade paper, inutili per gli spread realizzati).
# EN: Deribit helpers shared with 01c (extracted 2026-07-16; production, no-auth:
#     testnet must NOT be used here — paper trades, useless for realized spreads).
from quantsys.data.deribit import deribit_public_get, parse_instrument  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.trades_recorder")

TRADES_DIR = Path("data/deribit_trades")

# IT: retention osservata dell'endpoint pubblico (~24h): il cold start riparte
#     da qui; con cadenza 10 min ogni tick copre ~1/144 della finestra.
#     In ms interi: evita pd.Timedelta (DeprecationWarning numpy "generic unit"
#     sul VPS, futuro errore → crash-loop del servizio).
# EN: observed public endpoint retention (~24h): cold start rewinds this far;
#     at a 10-min cadence each tick covers ~1/144 of the window.
#     Integer ms: avoids pd.Timedelta (numpy "generic unit" DeprecationWarning
#     on the VPS, future error → service crash-loop).
RETENTION_MS = 24 * 3600 * 1000
# IT: overlap col già-salvato a ogni tick — assorbe clock skew e trade allo
#     stesso ms del bordo pagina; i doppioni li elimina il dedup su trade_id.
# EN: per-tick overlap with what's already saved — absorbs clock skew and
#     same-ms boundary trades; duplicates are removed by the trade_id dedup.
OVERLAP_MS = 60_000
# IT: massimo consentito dall'API per pagina.
# EN: API maximum per page.
PAGE_COUNT = 1000

def fetch_trades_window(currency: str, start_ms: int, end_ms: int,
                        max_pages: int = 200) -> pd.DataFrame:
    # IT: percorre [start_ms, end_ms] in avanti (sorting=asc) paginando su
    #     has_more: start avanza all'ultimo timestamp della pagina (inclusivo:
    #     i trade allo stesso ms ricompaiono e li dedupa trade_id). max_pages è
    #     una cintura di sicurezza anti-loop (200×1000 ≫ un giorno di trade BTC).
    # EN: walks [start_ms, end_ms] forward (sorting=asc) paginating on has_more:
    #     start advances to the page's last timestamp (inclusive: same-ms trades
    #     reappear and trade_id dedup removes them). max_pages is an anti-loop
    #     safety belt (200×1000 ≫ one day of BTC trades).
    frames = []
    cur = start_ms
    for _ in range(max_pages):
        res = deribit_public_get("public/get_last_trades_by_currency_and_time", {
            "currency": currency, "kind": "option",
            "start_timestamp": cur, "end_timestamp": end_ms,
            "count": PAGE_COUNT, "sorting": "asc",
        }, timeout=20)
        trades = res.get("trades", [])
        if not trades:
            break
        frames.append(pd.DataFrame(trades))
        last_ts = int(trades[-1]["timestamp"])
        if not res.get("has_more") or last_ts >= end_ms:
            break
        # IT: avanzamento inclusivo — se una pagina intera cade sullo stesso ms
        #     (irrealistico per le opzioni) si romperebbe qui: +1 in quel caso.
        # EN: inclusive advance — if a whole page shares one ms (unrealistic for
        #     options) it would stall here: bump by +1 in that case.
        cur = last_ts if last_ts > cur else last_ts + 1
        time.sleep(0.25)   # IT: cortesia rate-limit / EN: rate-limit courtesy
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["trade_id"])

    # IT: normalizzazione colonne: ts→datetime UTC + campi parsati dal nome.
    # EN: column normalization: ts→UTC datetime + name-parsed fields.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    parsed = df["instrument_name"].map(parse_instrument)
    df["expiry"] = parsed.map(lambda p: pd.Timestamp(p[0]) if p else pd.NaT)
    df["strike"] = parsed.map(lambda p: p[1] if p else float("nan"))
    df["option_type"] = parsed.map(lambda p: p[2] if p else None)
    keep = ["timestamp", "trade_id", "trade_seq", "instrument_name",
            "expiry", "strike", "option_type", "direction", "price",
            "amount", "contracts", "iv", "mark_price", "index_price",
            "tick_direction"]
    return df[[c for c in keep if c in df.columns]].sort_values("timestamp")


def last_saved_ms(now: pd.Timestamp) -> int:
    # IT: riprende dall'ultimo trade persistito (file di oggi, poi ieri —
    #     oltre non serve: la retention API è ~24h); cold start = now − retention.
    # EN: resumes from the last persisted trade (today's file, then yesterday's —
    #     no point further back: API retention is ~24h); cold start = now − retention.
    for day in (now, now - timedelta(days=1)):
        p = TRADES_DIR / f"option_trades_{day:%Y%m%d}.parquet"
        if p.exists():
            ts = pd.read_parquet(p, columns=["timestamp"])["timestamp"].max()
            return int(pd.Timestamp(ts).timestamp() * 1000)
    return int(now.timestamp() * 1000) - RETENTION_MS


def poll_once(currency: str) -> dict:
    # IT: un tick: finestra [ultimo salvato − overlap, now] → fetch paginato →
    #     append nei parquet giornalieri (per giorno UTC del TRADE: la finestra
    #     può scavallare la mezzanotte).
    # EN: one tick: [last saved − overlap, now] window → paginated fetch →
    #     append into daily parquet (by the TRADE's UTC day: the window can
    #     straddle midnight).
    now = pd.Timestamp.now(tz="UTC")
    now_ms = int(now.timestamp() * 1000)
    start_ms = max(last_saved_ms(now) - OVERLAP_MS, now_ms - RETENTION_MS)
    df = fetch_trades_window(currency, start_ms, now_ms)
    n_new, files = len(df), []
    for day, grp in (df.groupby(df["timestamp"].dt.strftime("%Y%m%d"))
                     if not df.empty else []):
        path = TRADES_DIR / f"option_trades_{day}.parquet"
        n_tot = append_parquet(path, grp, dedup_cols=["trade_id"], sort_col="timestamp")
        files.append((path.name, len(grp), n_tot))
    return {"ts": now, "n_fetched": n_new, "files": files}


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script, CLAUDE.md — bug cp1252 ricorrente).
    # EN: UTF-8 boilerplate (new-script checklist, CLAUDE.md — recurring cp1252 bug).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Recorder trade opzioni Deribit / Deribit option trades recorder")
    ap.add_argument("--minutes", type=int, default=10,
                    help="cadenza poll in minuti / poll cadence in minutes (default 10)")
    ap.add_argument("--once", action="store_true",
                    help="un solo tick e termina (smoke) / single tick then exit (smoke)")
    ap.add_argument("--currency", default="BTC",
                    help="valuta Deribit / Deribit currency (default BTC)")
    args = ap.parse_args()

    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"trades-recorder avviato/started — currency={args.currency} "
             f"cadenza/cadence {args.minutes} min → {TRADES_DIR}/")

    while True:
        try:
            info = poll_once(args.currency)
            detail = "; ".join(f"{n} (+{k} → {t} righe/rows)" for n, k, t in info["files"]) or "nessun trade nuovo / no new trades"
            log.info(f"tick {info['ts']:%Y-%m-%d %H:%M:%S}Z: {info['n_fetched']} trade scaricati/fetched — {detail}")
        except KeyboardInterrupt:
            log.info("Interrotto dall'utente / interrupted by user")
            return
        except Exception as e:
            # IT: transient (rete, 5xx) non uccidono il recorder: la continuità
            #     dello storico è la priorità (pattern 01c/01d); l'overlap +
            #     retention 24h coprono il buco al tick successivo.
            # EN: transients (network, 5xx) don't kill the recorder: history
            #     continuity is the priority (01c/01d pattern); overlap + 24h
            #     retention cover the gap at the next tick.
            log.error(f"tick fallito/failed: {type(e).__name__}: {e}")
        if args.once:
            return
        try:
            time.sleep(args.minutes * 60)
        except KeyboardInterrupt:
            log.info("Interrotto dall'utente / interrupted by user")
            return


if __name__ == "__main__":
    main()
