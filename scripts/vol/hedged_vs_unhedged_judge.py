# IT: GIUDICE HEDGED-VS-UNHEDGED (pre-registrazione V2 delta-hedged, STATUS.md
#     2026-07-12) — confronto WITHIN-TRADE: per ogni trade del campione hedged,
#     unhedged = PnL della sola leg opzioni (pnl_btc già loggato in trades.jsonl),
#     hedged = unhedged + PnL perp inverse ESATTO dal ledger − fee perp − funding.
#     PnL perp: per ogni intervallo di holding tra fill consecutivi,
#     pnl = H_usd·(1/s_fill − 1/s_next) (convenzione inverse; formula della
#     pre-registrazione). Funding: accrual orario dallo storico PUBBLICO Deribit
#     PROD (get_funding_rate_history: interest_1h + index_price) —
#     funding_paid_btc = Σ_h H_usd/index·interest_1h (rate>0 → i long pagano);
#     fonte PROD scelta by design (il funding testnet non è rappresentativo del
#     costo di mercato) — da ratificare nell'update di congelamento della v2.
#     Eventi 'reconcile' (fill perso per crash): il PnL del gap non è
#     attribuibile esattamente → h aggiornato senza PnL, trade FLAGGATO.
#     CONDIZIONI DI PASS (dalla pre-registrazione, valutate SOLO a n≥20):
#       1. var(hedged) ≤ 0.6·var(unhedged)
#       2. mean(hedged) ≥ mean(unhedged) − 0.25·std(unhedged)/√n
#       3. n ≥ 20 settlement CON hedge attivo (--since = ts attivazione --hedge)
#     Sotto n=20: verdict NOT_EVALUABLE, nessun numero decisionale.
#     Output: results/vols/hedged_vs_unhedged.json (per-trade + aggregati +
#     metadata). Scritto 2026-07-14 PRIMA dell'attivazione (--hedge inerte,
#     hedge_ledger inesistente): oggi può produrre solo NOT_EVALUABLE.
# EN: HEDGED-VS-UNHEDGED JUDGE (delta-hedged V2 pre-registration, STATUS.md
#     2026-07-12) — WITHIN-TRADE comparison: per hedged-sample trade,
#     unhedged = options-leg PnL only (pnl_btc already logged in trades.jsonl),
#     hedged = unhedged + EXACT inverse perp PnL from the ledger − perp fees −
#     funding. Perp PnL: per holding interval between consecutive fills,
#     pnl = H_usd·(1/s_fill − 1/s_next) (inverse convention; the
#     pre-registration formula). Funding: hourly accrual from the PUBLIC Deribit
#     PROD history (get_funding_rate_history: interest_1h + index_price) —
#     funding_paid_btc = Σ_h H_usd/index·interest_1h (rate>0 → longs pay); PROD
#     source is by design (testnet funding is not representative of the market
#     cost) — to be ratified in the v2 freezing update. 'reconcile' events
#     (fill lost to a crash): gap PnL not exactly attributable → h updated with
#     no PnL, trade FLAGGED.
#     PASS CONDITIONS (from the pre-registration, evaluated ONLY at n≥20):
#       1. var(hedged) ≤ 0.6·var(unhedged)
#       2. mean(hedged) ≥ mean(unhedged) − 0.25·std(unhedged)/√n
#       3. n ≥ 20 settlements WITH the hedge active (--since = --hedge activation ts)
#     Below n=20: NOT_EVALUABLE verdict, no decision numbers.
#     Output: results/vols/hedged_vs_unhedged.json (per-trade + aggregates +
#     metadata). Written 2026-07-14 BEFORE activation (--hedge inert, no
#     hedge_ledger on disk): today it can only return NOT_EVALUABLE.
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quantsys.utils import setup_logging                      # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.hvu_judge")

TRADES_PATH = ROOT / "results" / "vol_paper" / "trades.jsonl"
LEDGER_PATH = ROOT / "results" / "vol_paper" / "hedge_ledger.jsonl"
OUT_PATH = ROOT / "results" / "vols" / "hedged_vs_unhedged.json"
FUNDING_CACHE = ROOT / "data" / "deribit_funding_perp.parquet"
FUNDING_URL = "https://www.deribit.com/api/v2/public/get_funding_rate_history"

# IT: soglie PRE-REGISTRATE (STATUS.md 2026-07-12) — non toccarle a risultati visti.
# EN: PRE-REGISTERED thresholds (STATUS.md 2026-07-12) — do not touch after results.
VAR_RATIO_MAX = 0.6
MEAN_DRAG_SE_FRAC = 0.25
N_MIN = 20


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _pos_key(d: dict) -> tuple:
    # IT: chiave trade↔ledger = (side, strike, expiry_ms) — position_key di 04b.
    # EN: trade↔ledger key = (side, strike, expiry_ms) — 04b's position_key.
    return (int(d["side"]), float(d["strike"]), int(d["expiry_ms"]))


def fetch_funding_history(start_ms: int, end_ms: int,
                          cache_path: Path = FUNDING_CACHE,
                          no_fetch: bool = False) -> pd.DataFrame:
    # IT: storico funding perp orario (PROD, pubblico) con cache parquet
    #     append+dedup: colonne [timestamp(ms), index_price, interest_1h].
    #     Fetch a chunk di 30 giorni (l'endpoint pagina per finestra temporale).
    # EN: hourly perp funding history (PROD, public) with append+dedup parquet
    #     cache: columns [timestamp(ms), index_price, interest_1h]. 30-day
    #     chunked fetch (the endpoint pages by time window).
    cached = pd.read_parquet(cache_path) if cache_path.exists() else \
        pd.DataFrame(columns=["timestamp", "index_price", "interest_1h"])
    have_lo = int(cached["timestamp"].min()) if len(cached) else None
    have_hi = int(cached["timestamp"].max()) if len(cached) else None
    if no_fetch or (have_lo is not None and have_lo <= start_ms and have_hi >= end_ms):
        return cached
    rows = []
    chunk = 30 * 24 * 3600 * 1000
    t0 = start_ms
    while t0 < end_ms:
        t1 = min(t0 + chunk, end_ms)
        r = requests.get(FUNDING_URL, params={
            "instrument_name": "BTC-PERPETUAL",
            "start_timestamp": t0, "end_timestamp": t1}, timeout=20)
        r.raise_for_status()
        for rec in r.json().get("result", []):
            rows.append({"timestamp": int(rec["timestamp"]),
                         "index_price": float(rec["index_price"]),
                         "interest_1h": float(rec["interest_1h"])})
        t0 = t1
    df = (pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
          .drop_duplicates(subset="timestamp").sort_values("timestamp")
          .reset_index(drop=True))
    if len(df) > len(cached):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def funding_paid_btc(h_usd: float, t0_ms: int, t1_ms: int,
                     funding: pd.DataFrame) -> float:
    # IT: accrual su [t0,t1): somma dei punti orari nel range (granularità 1h
    #     dichiarata: errore ≤1h agli estremi). rate>0 → il long (H>0) PAGA.
    # EN: accrual over [t0,t1): sum of hourly points in range (declared 1h
    #     granularity: ≤1h error at the ends). rate>0 → the long (H>0) PAYS.
    if abs(h_usd) < 1e-9 or t1_ms <= t0_ms or funding.empty:
        return 0.0
    m = (funding["timestamp"] >= t0_ms) & (funding["timestamp"] < t1_ms)
    sel = funding[m]
    if sel.empty:
        return 0.0
    return float((h_usd / sel["index_price"] * sel["interest_1h"]).sum())


def perp_leg(events: list[dict], funding: pd.DataFrame,
             settle_ms: int | None = None) -> dict:
    # IT: ricostruzione della leg perp di UN trade dai suoi eventi ledger
    #     (ordinati per ts): PnL inverse esatto tra fill consecutivi, fee
    #     sommate dal ledger, funding accruato sull'holding. 'reconcile' cambia
    #     H senza fill → nessun PnL sul gap, flag. Se l'ultimo evento lascia
    #     H≠0 (flatten mancante) il trade è flaggato open_residual.
    # EN: perp-leg reconstruction for ONE trade from its ledger events (sorted
    #     by ts): exact inverse PnL between consecutive fills, fees summed from
    #     the ledger, funding accrued over the holding. 'reconcile' changes H
    #     with no fill → no gap PnL, flagged. If the last event leaves H≠0
    #     (missing flatten) the trade is flagged open_residual.
    ev = sorted(events, key=lambda e: pd.Timestamp(e["ts"]).value)
    pnl_gross, fees, fund_paid = 0.0, 0.0, 0.0
    h_cur, s_last, t_last = 0.0, None, None
    has_reconcile = False
    for e in ev:
        t_ms = pd.Timestamp(e["ts"]).value // 10**6
        price = e.get("fill_price")
        if price is not None:
            price = float(price)
            if s_last is not None and abs(h_cur) > 1e-9:
                pnl_gross += h_cur * (1.0 / s_last - 1.0 / price)
            if t_last is not None:
                fund_paid += funding_paid_btc(h_cur, t_last, t_ms, funding)
            s_last, t_last = price, t_ms
        else:
            # IT: reconcile — H cambia senza prezzo: gap non attribuibile.
            # EN: reconcile — H changes with no price: unattributable gap.
            has_reconcile = True
            if t_last is not None:
                fund_paid += funding_paid_btc(h_cur, t_last, t_ms, funding)
            t_last = t_ms
        h_cur = float(e.get("h_usd_after", h_cur))
        fees += float(e.get("fee_btc") or 0.0)
    open_residual = abs(h_cur) > 1e-9
    # IT: residuo aperto: funding accruato fino al settlement (se noto) come
    #     stima conservativa; PnL prezzo del residuo NON stimabile → flag.
    # EN: open residual: funding accrued to settlement (when known) as a
    #     conservative estimate; residual price PnL NOT estimable → flag.
    if open_residual and settle_ms and t_last is not None and settle_ms > t_last:
        fund_paid += funding_paid_btc(h_cur, t_last, settle_ms, funding)
    return {"pnl_perp_gross": pnl_gross, "fees_perp": fees,
            "funding_paid": fund_paid, "n_events": len(ev),
            "has_reconcile": has_reconcile, "open_residual": open_residual}


def evaluate(trades: list[dict], ledger: list[dict], funding: pd.DataFrame,
             since: pd.Timestamp | None) -> dict:
    # IT: campione hedged = trade settled con entry_ts ≥ since (attivazione
    #     --hedge): include i trade post-attivazione SENZA fill perp (hedge mai
    #     fuori banda = PnL perp 0, che È il dato corretto). Senza --since,
    #     fallback = solo trade con fill nel ledger (campione BIASED: esclude
    #     gli zero-rebalance — warning esplicito).
    # EN: hedged sample = settled trades with entry_ts ≥ since (--hedge
    #     activation): includes post-activation trades WITHOUT perp fills
    #     (hedge never out of band = 0 perp PnL, which IS the correct datum).
    #     Without --since, fallback = only trades with ledger fills (BIASED
    #     sample: excludes zero-rebalance ones — explicit warning).
    by_key: dict[tuple, list] = {}
    for e in ledger:
        pk = e.get("position_key")
        if pk:
            by_key.setdefault(_pos_key(pk), []).append(e)

    rows = []
    for tr in trades:
        if "pnl_btc" not in tr:
            continue
        entry = pd.Timestamp(tr["entry_ts"])
        if since is not None:
            if entry < since:
                continue
        elif _pos_key(tr) not in by_key:
            continue
        events = by_key.get(_pos_key(tr), [])
        settle_ms = int(tr["expiry_ms"])
        leg = perp_leg(events, funding, settle_ms) if events else \
            {"pnl_perp_gross": 0.0, "fees_perp": 0.0, "funding_paid": 0.0,
             "n_events": 0, "has_reconcile": False, "open_residual": False}
        pnl_u = float(tr["pnl_btc"])
        pnl_h = pnl_u + leg["pnl_perp_gross"] - leg["fees_perp"] - leg["funding_paid"]
        rows.append({"entry_ts": str(entry), "side": int(tr["side"]),
                     "strike": float(tr["strike"]), "expiry_ms": settle_ms,
                     "pnl_unhedged": pnl_u, "pnl_hedged": pnl_h, **leg})

    n = len(rows)
    rep = {"n_hedged_settlements": n, "n_min": N_MIN,
           "since": str(since) if since is not None else None,
           "rows": rows}
    if since is None and n:
        rep["warning"] = ("campione senza --since: esclusi i trade hedged "
                          "zero-rebalance — BIASED / no --since: zero-rebalance "
                          "hedged trades excluded — BIASED")
    if n < N_MIN:
        rep["verdict"] = "NOT_EVALUABLE"
        rep["reason"] = f"n={n} < {N_MIN} (condizione 3 pre-registrata)"
        return rep
    u = np.array([r["pnl_unhedged"] for r in rows])
    h = np.array([r["pnl_hedged"] for r in rows])
    var_u, var_h = float(np.var(u, ddof=1)), float(np.var(h, ddof=1))
    mean_u, mean_h = float(u.mean()), float(h.mean())
    se_u = float(u.std(ddof=1) / np.sqrt(n))
    c1 = var_h <= VAR_RATIO_MAX * var_u
    c2 = mean_h >= mean_u - MEAN_DRAG_SE_FRAC * se_u
    rep.update({
        "aggregates": {"mean_unhedged": mean_u, "mean_hedged": mean_h,
                       "var_unhedged": var_u, "var_hedged": var_h,
                       "var_ratio": var_h / var_u if var_u > 0 else np.inf,
                       "se_unhedged": se_u,
                       "fees_perp_total": float(sum(r["fees_perp"] for r in rows)),
                       "funding_paid_total": float(sum(r["funding_paid"] for r in rows)),
                       "n_reconcile_flagged": int(sum(r["has_reconcile"] for r in rows)),
                       "n_open_residual": int(sum(r["open_residual"] for r in rows))},
        "conditions": {"1_var_ratio_le_0.6": bool(c1),
                       "2_mean_drag_le_quarter_se": bool(c2),
                       "3_n_ge_20": True},
        "verdict": "PASS" if (c1 and c2) else "FAIL"})
    return rep


def main() -> int:
    # IT: boilerplate UTF-8 console Windows (checklist CLAUDE.md — bug cp1252).
    # EN: Windows console UTF-8 boilerplate (CLAUDE.md checklist — cp1252 bug).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Giudice hedged-vs-unhedged (pre-reg V2)")
    ap.add_argument("--trades", default=str(TRADES_PATH))
    ap.add_argument("--ledger", default=str(LEDGER_PATH))
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--since", default=None,
                    help="ts attivazione --hedge (definisce il campione) / --hedge activation ts (defines the sample)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="solo cache funding, nessuna chiamata rete / funding cache only, no network")
    args = ap.parse_args()

    trades = _read_jsonl(Path(args.trades))
    ledger = _read_jsonl(Path(args.ledger))
    since = pd.Timestamp(args.since) if args.since else None
    if since is not None and since.tz is None:
        since = since.tz_localize("UTC")

    # IT: range funding = min entry → max expiry dei trade nel campione.
    # EN: funding range = sample trades' min entry → max expiry.
    funding = pd.DataFrame(columns=["timestamp", "index_price", "interest_1h"])
    settled = [t for t in trades if "pnl_btc" in t]
    if settled and ledger:
        t0 = min(pd.Timestamp(t["entry_ts"]).value // 10**6 for t in settled)
        t1 = max(int(t["expiry_ms"]) for t in settled)
        funding = fetch_funding_history(t0, t1, no_fetch=args.no_fetch)

    rep = evaluate(trades, ledger, funding, since)
    rep["meta"] = {"trades_file": args.trades, "ledger_file": args.ledger,
                   "funding_source": "deribit PROD get_funding_rate_history (interest_1h)",
                   "generated": str(pd.Timestamp.now(tz="UTC").floor("s")),
                   "thresholds": {"var_ratio_max": VAR_RATIO_MAX,
                                  "mean_drag_se_frac": MEAN_DRAG_SE_FRAC,
                                  "n_min": N_MIN}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    log.info(f"verdict: {rep['verdict']} (n={rep['n_hedged_settlements']}) → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
