# IT: Poller IV Deribit (checklist 2026-06-11, punto 1) — raccolta FORWARD della
#     volatilità implicita BTC per il gate economico futuro "NN-RV vs IV":
#     lo storico IV short-tenor NON è gratis (Tardis ≥$300), quindi il dataset
#     si costruisce da qui in avanti. Loop 5-15 min, 2 richieste NON autenticate:
#       1. public/get_book_summary_by_currency (mark_iv di TUTTI gli strumenti
#          opzione BTC in una sola chiamata, rate limit 10 req/s ≫ il nostro uso)
#       2. public/get_volatility_index_data (ultimo punto DVOL, controllo 30d)
#     Output append-only in data/iv/:
#       chain/btc_options_YYYYMMDD.parquet  — snapshot raw per-strumento (1 file/giorno)
#       atm_30h.parquet                     — 1 riga/tick: ATM IV per expiry vicine
#                                             + IV interpolata a tenor costante 30h
#       dvol.parquet                        — serie DVOL (anche target del backfill)
#     Modalità: --once (smoke), --minutes N (cadenza, default 10), --backfill-dvol
#     (punto 3 checklist: storico orario DVOL 2021-03-24→oggi, ~46 chiamate).
# EN: Deribit IV poller (2026-06-11 checklist, item 1) — FORWARD collection of BTC
#     implied volatility for the future economic gate "NN-RV vs IV": short-tenor
#     IV history is NOT free (Tardis ≥$300), so the dataset is built from now on.
#     5-15 min loop, 2 UNauthenticated requests:
#       1. public/get_book_summary_by_currency (mark_iv of ALL BTC option
#          instruments in a single call, 10 req/s rate limit ≫ our usage)
#       2. public/get_volatility_index_data (latest DVOL point, 30d control)
#     Append-only output under data/iv/:
#       chain/btc_options_YYYYMMDD.parquet  — raw per-instrument snapshot (1 file/day)
#       atm_30h.parquet                     — 1 row/tick: ATM IV per nearby expiry
#                                             + IV interpolated at constant 30h tenor
#       dvol.parquet                        — DVOL series (also the backfill target)
#     Modes: --once (smoke), --minutes N (cadence, default 10), --backfill-dvol
#     (checklist item 3: hourly DVOL history 2021-03-24→today, ~46 calls).
import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantsys.utils import setup_logging                      # noqa: E402
from quantsys.utils.atomic_save import atomic_save_parquet    # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.iv_poller")

# IT: endpoint pubblici Deribit (verificati con chiamate dirette, 2026-06-11).
# EN: Deribit public endpoints (verified with direct calls, 2026-06-11).
DERIBIT_BASE = "https://www.deribit.com/api/v2"
IV_DIR = Path("data/iv")
CHAIN_DIR = IV_DIR / "chain"
ATM_PATH = IV_DIR / "atm_30h.parquet"
DVOL_PATH = IV_DIR / "dvol.parquet"

# IT: tenor target in ore = forecast_horizon del modello vol (30 barre 1h).
# EN: target tenor in hours = the vol model's forecast_horizon (30 1h bars).
TENOR_HOURS = 30.0
# IT: quante expiry vicine tenere nel riepilogo ATM (term structure corta).
# EN: how many nearby expiries to keep in the ATM summary (short term structure).
N_EXPIRIES = 4
# IT: inizio disponibilità DVOL su Deribit (per il backfill).
# EN: DVOL availability start on Deribit (for the backfill).
DVOL_START = datetime(2021, 3, 24, tzinfo=timezone.utc)

# IT: nome strumento Deribit: BTC-13JUN26-105000-C → expiry 08:00 UTC del giorno.
# EN: Deribit instrument name: BTC-13JUN26-105000-C → expiry at 08:00 UTC that day.
_INSTR_RE = re.compile(r"^BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def _get(path: str, params: dict, timeout: int = 15) -> dict:
    # IT: GET pubblica con error-raise; il chiamante gestisce i transient nel loop.
    # EN: public GET with error-raise; caller handles transients in the loop.
    r = requests.get(f"{DERIBIT_BASE}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "result" not in payload:
        raise RuntimeError(f"Deribit risposta inattesa / unexpected response: {payload}")
    return payload["result"]


def parse_instrument(name: str):
    # IT: estrae (expiry UTC, strike, tipo) dal nome; None se non-standard.
    # EN: extracts (UTC expiry, strike, type) from the name; None if non-standard.
    m = _INSTR_RE.match(name)
    if not m:
        return None
    day, mon, yy, strike_s, opt = m.groups()
    expiry = datetime(2000 + int(yy), _MONTHS[mon], int(day), 8, 0,
                      tzinfo=timezone.utc)
    # IT: strike decimali tipo "3d5" non esistono su BTC, ma il parse non crasha.
    # EN: decimal strikes like "3d5" don't occur on BTC, but parsing won't crash.
    strike = float(strike_s.replace("d", "."))
    return expiry, strike, opt


def fetch_chain_snapshot(now: pd.Timestamp) -> pd.DataFrame:
    # IT: 1 chiamata → mark_iv di tutta la chain opzioni BTC; righe non
    #     parsabili o senza IV vengono scartate (perpetual, strumenti spenti).
    # EN: 1 call → mark_iv of the whole BTC option chain; unparsable rows or
    #     rows without IV are dropped (perpetuals, dead instruments).
    res = _get("public/get_book_summary_by_currency",
               {"currency": "BTC", "kind": "option"})
    rows = []
    for it in res:
        parsed = parse_instrument(it.get("instrument_name", ""))
        if parsed is None or it.get("mark_iv") is None:
            continue
        expiry, strike, opt = parsed
        rows.append({
            "snapshot_ts": now,
            "instrument_name": it["instrument_name"],
            "expiry": pd.Timestamp(expiry),
            "strike": strike,
            "option_type": opt,
            # IT: mark_iv in PERCENTO (convenzione Deribit, es. 45.2 = 45.2%).
            # EN: mark_iv in PERCENT (Deribit convention, e.g. 45.2 = 45.2%).
            "mark_iv": float(it["mark_iv"]),
            "underlying_price": it.get("underlying_price"),
            "mark_price": it.get("mark_price"),
            "bid_price": it.get("bid_price"),
            "ask_price": it.get("ask_price"),
            "open_interest": it.get("open_interest"),
            "volume": it.get("volume"),
        })
    return pd.DataFrame(rows)


def atm_term_structure(chain: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    # IT: per ognuna delle N_EXPIRIES expiry vive più vicine: forward = mediana
    #     dell'underlying_price per-expiry (è il future/sintetico della scadenza),
    #     strike ATM = il più vicino al forward, ATM IV = media delle mark_iv
    #     C/P a quello strike (IV dello straddle ATM).
    # EN: for each of the N_EXPIRIES nearest live expiries: forward = per-expiry
    #     median underlying_price (the expiry's future/synthetic), ATM strike =
    #     closest to forward, ATM IV = mean of C/P mark_iv at that strike
    #     (ATM straddle IV).
    live = chain[chain["expiry"] > now]
    out = []
    for expiry in sorted(live["expiry"].unique())[:N_EXPIRIES]:
        grp = live[live["expiry"] == expiry]
        fwd = grp["underlying_price"].median()
        if not np.isfinite(fwd):
            continue
        strikes = grp["strike"].unique()
        atm_k = strikes[np.argmin(np.abs(strikes - fwd))]
        iv = grp.loc[grp["strike"] == atm_k, "mark_iv"].mean()
        t_hours = (expiry - now).total_seconds() / 3600.0
        out.append({"expiry": expiry, "t_hours": t_hours,
                    "forward": fwd, "atm_strike": atm_k, "atm_iv": iv})
    return pd.DataFrame(out)


def interp_iv_at_tenor(term: pd.DataFrame, tenor_hours: float) -> float:
    # IT: interpolazione LINEARE IN VARIANZA TOTALE w(T)=σ²·T (standard per le
    #     term structure IV: preserva l'assenza di arbitraggio calendario se le
    #     w sono crescenti). Sotto la prima expiry: σ flat dalla più corta
    #     (= w lineare dall'origine); sopra l'ultima tenuta: σ flat dall'ultima.
    # EN: LINEAR interpolation IN TOTAL VARIANCE w(T)=σ²·T (standard for IV term
    #     structures: preserves calendar no-arbitrage when w is increasing).
    #     Below the first expiry: flat σ from the shortest (= w linear from the
    #     origin); above the last kept: flat σ from the longest.
    if term.empty:
        return float("nan")
    t = term["t_hours"].to_numpy(dtype=float)
    sigma = term["atm_iv"].to_numpy(dtype=float) / 100.0
    w = sigma ** 2 * t
    if tenor_hours <= t[0]:
        return float(sigma[0] * 100.0)
    if tenor_hours >= t[-1]:
        return float(sigma[-1] * 100.0)
    w_star = float(np.interp(tenor_hours, t, w))
    return float(np.sqrt(w_star / tenor_hours) * 100.0)


def fetch_dvol(start_ms: int, end_ms: int, resolution: str = "3600") -> pd.DataFrame:
    # IT: candele dell'indice DVOL BTC (max 1000 punti/chiamata); teniamo il close.
    # EN: BTC DVOL index candles (max 1000 points/call); we keep the close.
    res = _get("public/get_volatility_index_data",
               {"currency": "BTC", "start_timestamp": start_ms,
                "end_timestamp": end_ms, "resolution": resolution})
    data = res.get("data", [])
    if not data:
        return pd.DataFrame()
    arr = np.asarray(data, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(arr[:, 0].astype(np.int64), unit="ms", utc=True),
        "dvol": arr[:, 4],
    })


def append_parquet(path: Path, new_rows: pd.DataFrame, dedup_cols: list) -> int:
    # IT: append con dedup su chiave + scrittura atomica (tmp+os.replace, pattern
    #     repo): un crash a metà tick non corrompe mai lo storico accumulato.
    # EN: keyed-dedup append + atomic write (tmp+os.replace, repo pattern): a
    #     mid-tick crash can never corrupt the accumulated history.
    if new_rows.empty:
        return 0
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, new_rows], ignore_index=True)
    else:
        merged = new_rows
    merged = merged.drop_duplicates(subset=dedup_cols, keep="last")
    merged = merged.sort_values(dedup_cols[0]).reset_index(drop=True)
    atomic_save_parquet(merged, path, index=False)
    return len(merged)


def poll_once() -> dict:
    # IT: un tick completo: chain raw → ATM term structure → IV@30h → DVOL.
    # EN: one full tick: raw chain → ATM term structure → IV@30h → DVOL.
    now = pd.Timestamp.now(tz="UTC").floor("s")
    chain = fetch_chain_snapshot(now)
    if chain.empty:
        raise RuntimeError("chain vuota / empty chain dal book summary")

    chain_path = CHAIN_DIR / f"btc_options_{now:%Y%m%d}.parquet"
    n_chain = append_parquet(chain_path, chain,
                             ["snapshot_ts", "instrument_name"])

    term = atm_term_structure(chain, now)
    iv30 = interp_iv_at_tenor(term, TENOR_HOURS)

    # IT: ultimo punto DVOL (finestra 2h per coprire la candela oraria in corso).
    # EN: latest DVOL point (2h window to cover the in-progress hourly candle).
    end_ms = int(now.timestamp() * 1000)
    dvol_df = fetch_dvol(end_ms - 2 * 3600 * 1000, end_ms)
    dvol_last = float(dvol_df["dvol"].iloc[-1]) if not dvol_df.empty else float("nan")
    append_parquet(DVOL_PATH, dvol_df, ["timestamp"])

    # IT: riga di riepilogo wide: iv@30h + le N expiry ATM (tenor/iv/forward).
    # EN: wide summary row: iv@30h + the N ATM expiries (tenor/iv/forward).
    row = {"timestamp": now, "iv_30h": iv30, "dvol": dvol_last,
           "n_instruments": len(chain)}
    for i, r in term.reset_index(drop=True).iterrows():
        row[f"exp{i}_t_hours"] = r["t_hours"]
        row[f"exp{i}_atm_iv"] = r["atm_iv"]
        row[f"exp{i}_forward"] = r["forward"]
    append_parquet(ATM_PATH, pd.DataFrame([row]), ["timestamp"])

    return {"ts": str(now), "iv_30h": iv30, "dvol": dvol_last,
            "n_instruments": len(chain), "n_expiries": len(term),
            "chain_rows_today": n_chain}


def backfill_dvol() -> None:
    # IT: backfill storico DVOL orario 2021-03-24→oggi: 1000 punti/chiamata,
    #     ~46 chiamate, pausa 0.3s (ben sotto il rate limit pubblico).
    # EN: hourly DVOL history backfill 2021-03-24→today: 1000 points/call,
    #     ~46 calls, 0.3s pause (well below the public rate limit).
    step_ms = 1000 * 3600 * 1000
    start = int(DVOL_START.timestamp() * 1000)
    end = int(time.time() * 1000)
    total = 0
    cur = start
    while cur < end:
        chunk = fetch_dvol(cur, min(cur + step_ms, end))
        n = append_parquet(DVOL_PATH, chunk, ["timestamp"])
        total += len(chunk)
        log.info(f"  DVOL backfill: +{len(chunk)} punti/points "
                 f"(fino a/up to {pd.Timestamp(min(cur + step_ms, end), unit='ms', tz='UTC')}) "
                 f"— file: {n} righe/rows")
        cur += step_ms
        time.sleep(0.3)
    log.info(f"Backfill DVOL completato/done: {total} punti scaricati/points fetched → {DVOL_PATH}")


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script, CLAUDE.md — bug cp1252 ricorrente).
    # EN: UTF-8 boilerplate (new-script checklist, CLAUDE.md — recurring cp1252 bug).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Poller IV Deribit (BTC options + DVOL) / Deribit IV poller")
    ap.add_argument("--minutes", type=int, default=10,
                    help="cadenza poll in minuti / poll cadence in minutes (default 10)")
    ap.add_argument("--once", action="store_true",
                    help="un solo tick e termina (smoke) / single tick then exit (smoke)")
    ap.add_argument("--backfill-dvol", action="store_true",
                    help="backfill storico DVOL orario e termina / hourly DVOL history backfill then exit")
    args = ap.parse_args()

    CHAIN_DIR.mkdir(parents=True, exist_ok=True)

    if args.backfill_dvol:
        backfill_dvol()
        return

    if not 5 <= args.minutes <= 60 and not args.once:
        log.warning(f"cadenza {args.minutes} min fuori dal range consigliato 5-15 / "
                    f"cadence outside the recommended 5-15 range")

    log.info(f"Poller IV avviato/started — cadenza/cadence {args.minutes} min, "
             f"tenor target {TENOR_HOURS}h, output {IV_DIR}/")
    while True:
        try:
            info = poll_once()
            log.info(f"tick {info['ts']}: iv_30h={info['iv_30h']:.2f}% "
                     f"dvol={info['dvol']:.2f} chain={info['n_instruments']} strumenti/instruments "
                     f"({info['n_expiries']} expiry ATM)")
        except KeyboardInterrupt:
            log.info("Interrotto dall'utente / interrupted by user")
            return
        except Exception as e:
            # IT: i transient (rete, 5xx) non uccidono il poller: logga e riprova
            #     al tick successivo — la continuità dello storico è la priorità.
            # EN: transients (network, 5xx) don't kill the poller: log and retry
            #     next tick — history continuity is the priority.
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
