# IT: BASELINE ANALYSIS del forward test vol-paper (gate pre-registrato STATUS.md
#     2026-06-12, criterio 2). Calcola, sullo STESSO calendario di expiry del
#     forward test, il P&L di tre strategie:
#       - NN   : entra side=sign(edge) sse |edge|>0.25 (la regola pre-registrata);
#       - LONG : always-long-vol (compra straddle a OGNI expiry del calendario);
#       - SHORT: always-short-vol (vende straddle a OGNI expiry).
#     Le baseline isolano il TIMING del NN dal variance risk premium medio: il NN
#     deve battere ENTRAMBE per superare il gate (2).
#     Metodo: REPLAY fedele del loop di 04b_vol_paper.py sul log forecasts.parquet,
#     con premio dello straddle RICOSTRUITO dai chain snapshot (data/iv/chain/*.parquet,
#     stessa selezione di pick_straddle: expiry≈30h, strike ATM, mark call+put) e
#     delivery price dall'endpoint pubblico Deribit (cache locale). Semantica
#     identica al harness: max-1 posizione, hold-to-expiry, formula di settlement
#     inverse-option bit-identica.
#     Gate (1)+(3) [NN profittevole, hit-rate] sono calcolati dai trade REALI in
#     trades.jsonl; gate (2) dal replay ricostruito (+ cross-check NN-ricostruito
#     vs NN-reale come sanity della ricostruzione del premio).
#     NESSUNA decisione a risultati visti: lo script LEGGE soltanto, non tocca i
#     processi attivi né i modelli. Output: results/vol_paper/baseline_report.json.
# EN: BASELINE ANALYSIS of the vol-paper forward test (pre-registered gate in
#     STATUS.md 2026-06-12, criterion 2). Computes, over the SAME expiry calendar
#     as the forward test, the P&L of three strategies:
#       - NN   : enters side=sign(edge) iff |edge|>0.25 (the pre-registered rule);
#       - LONG : always-long-vol (buys the straddle at EVERY calendar expiry);
#       - SHORT: always-short-vol (sells the straddle at every expiry).
#     The baselines isolate the NN's TIMING from the average variance risk premium:
#     the NN must beat BOTH to pass gate (2).
#     Method: faithful REPLAY of the 04b_vol_paper.py loop over the forecasts.parquet
#     log, with the straddle premium RECONSTRUCTED from chain snapshots
#     (data/iv/chain/*.parquet, same selection as pick_straddle: expiry≈30h, ATM
#     strike, call+put mark) and the delivery price from the public Deribit endpoint
#     (local cache). Semantics identical to the harness: max-1 position,
#     hold-to-expiry, bit-identical inverse-option settlement formula.
#     Gate (1)+(3) [NN profitable, hit-rate] are computed from the REAL trades in
#     trades.jsonl; gate (2) from the reconstructed replay (+ a reconstructed-NN
#     vs real-NN cross-check as a sanity of the premium reconstruction).
#     NO decision after seeing results: this script only READS, touches neither the
#     active processes nor the models. Output: results/vol_paper/baseline_report.json.
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantsys.utils import setup_logging, load_config                          # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vol_baselines")

OUT_DIR = Path("results/vol_paper")
FORECASTS_PATH = OUT_DIR / "forecasts.parquet"
TRADES_PATH = OUT_DIR / "trades.jsonl"
CHAIN_DIR = Path("data/iv/chain")
DELIVERY_CACHE = OUT_DIR / "delivery_cache.json"
REPORT_PATH = OUT_DIR / "baseline_report.json"

# IT: costanti PRE-REGISTRATE — DEVONO coincidere con 04b_vol_paper.py (source of
#     truth). Sono ridefinite qui (non importate) perché 04b ha nome non-importabile
#     (inizia con cifra) ed esegue side-effect a import. Se 04b cambia, allinea qui.
# EN: PRE-REGISTERED constants — MUST match 04b_vol_paper.py (source of truth).
#     Redefined here (not imported) because 04b has a non-importable name (leading
#     digit) and runs import-time side effects. If 04b changes, mirror it here.
TENOR_HOURS = 30.0
EDGE_THRESHOLD = 0.25
SIZE_CONTRACTS = 1.0
FEE_PER_CONTRACT = 0.0003
FEE_CAP_FRAC = 0.125
SNAPSHOT_TOL_MIN = 20.0           # IT: max distanza tick↔snapshot chain | EN: max tick↔chain-snapshot gap


def fee_btc(premium: float) -> float:
    # IT: fee taker per contratto, cap al 12.5% del premio (identica a 04b).
    # EN: per-contract taker fee, capped at 12.5% of premium (identical to 04b).
    return min(FEE_PER_CONTRACT, FEE_CAP_FRAC * premium) * SIZE_CONTRACTS


def settle_pnl(side: int, delivery_price: float, strike: float,
               prem_call: float, prem_put: float) -> dict:
    # IT: P&L cash-settled inverse-option, formula bit-identica a maybe_settle (04b):
    #     payoff straddle = |S_del−K|/S_del BTC/contratto; pnl = side·(payoff−prem)−fee.
    # EN: inverse-option cash-settled P&L, formula bit-identical to maybe_settle (04b):
    #     straddle payoff = |S_del−K|/S_del BTC/contract; pnl = side·(payoff−prem)−fee.
    payoff = abs(delivery_price - strike) / delivery_price * SIZE_CONTRACTS
    premium = (prem_call + prem_put) * SIZE_CONTRACTS
    fee = fee_btc(prem_call) + fee_btc(prem_put)
    pnl = side * (payoff - premium) - fee
    return {"payoff_btc": payoff, "premium_btc": premium, "fee_btc": fee, "pnl_btc": pnl}


# ──────────────────────────── ricostruzione premio dalle chain ────────────────────────────
def load_chain() -> pd.DataFrame:
    # IT: concat di tutti gli snapshot chain; tz-aware UTC; ordinati per snapshot_ts.
    # EN: concat of all chain snapshots; tz-aware UTC; sorted by snapshot_ts.
    files = sorted(CHAIN_DIR.glob("btc_options_*.parquet"))
    if not files:
        raise RuntimeError(f"nessuno snapshot chain in {CHAIN_DIR} — il poller IV ha girato?")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for col in ("snapshot_ts", "expiry"):
        df[col] = pd.to_datetime(df[col], utc=True)
    log.info(f"chain: {len(df):,} righe da {len(files)} file, "
             f"{df['snapshot_ts'].nunique()} snapshot distinti "
             f"({df['snapshot_ts'].min()} → {df['snapshot_ts'].max()})")
    return df.sort_values("snapshot_ts").reset_index(drop=True)


def reconstruct_straddle(chain: pd.DataFrame, decision_ts: pd.Timestamp) -> dict | None:
    # IT: replica di pick_straddle+mark_price sul chain snapshot più vicino (≥ decision_ts,
    #     entro SNAPSHOT_TOL_MIN): expiry daily più vicina al tenor 30h, strike ATM
    #     (più vicino a underlying_price/forward), mark di call+put a quello strike.
    #     None se manca lo snapshot o la coppia ATM call+put.
    # EN: replica of pick_straddle+mark_price on the nearest chain snapshot
    #     (≥ decision_ts, within SNAPSHOT_TOL_MIN): daily expiry closest to the 30h
    #     tenor, ATM strike (nearest to underlying_price/forward), call+put mark at
    #     that strike. None if the snapshot or the ATM call+put pair is missing.
    fwd = chain[chain["snapshot_ts"] >= decision_ts]
    if fwd.empty:
        return None
    snap_ts = fwd["snapshot_ts"].iloc[0]
    if (snap_ts - decision_ts).total_seconds() / 60.0 > SNAPSHOT_TOL_MIN:
        return None
    snap = chain[chain["snapshot_ts"] == snap_ts]

    # IT: expiry daily (08:00 UTC) più vicina al tenor; t_hours dalla decision_ts.
    # EN: daily expiry (08:00 UTC) closest to the tenor; t_hours from decision_ts.
    exps = snap["expiry"].unique()
    t_hours = {e: (pd.Timestamp(e) - decision_ts).total_seconds() / 3600.0 for e in exps}
    exps_future = [e for e in exps if t_hours[e] > 0]
    if not exps_future:
        return None
    exp = min(exps_future, key=lambda e: abs(t_hours[e] - TENOR_HOURS))
    leg = snap[snap["expiry"] == exp]

    # IT: strike ATM = più vicino al forward (underlying_price, mediano sulla gamba).
    # EN: ATM strike = nearest to the forward (underlying_price, median over the leg).
    fwd_px = float(leg["underlying_price"].median())
    strikes = sorted(leg["strike"].unique())
    k = min(strikes, key=lambda s: abs(s - fwd_px))
    call = leg[(leg["strike"] == k) & (leg["option_type"] == "C")]
    put = leg[(leg["strike"] == k) & (leg["option_type"] == "P")]
    if call.empty or put.empty:
        return None
    return {"expiry": pd.Timestamp(exp), "expiry_ms": int(pd.Timestamp(exp).value // 10**6),
            "t_hours": t_hours[exp], "strike": float(k), "forward": fwd_px,
            "prem_call": float(call["mark_price"].iloc[0]),
            "prem_put": float(put["mark_price"].iloc[0]),
            "snapshot_ts": snap_ts}


# ──────────────────────────── delivery prices (Deribit pubblico, cache) ────────────────────────────
def load_delivery_cache() -> dict:
    if DELIVERY_CACHE.exists():
        return json.loads(DELIVERY_CACHE.read_text(encoding="utf-8"))
    return {}


def fetch_delivery_prices(base_url: str, count: int = 30) -> dict:
    # IT: public/get_delivery_prices (no auth) → {DDMMMYY: prezzo}. Stessa sorgente
    #     testnet del harness (i trade reali settlano su questi prezzi); count limitato
    #     dall'API → si accumula in cache su run ripetute.
    # EN: public/get_delivery_prices (no auth) → {DDMMMYY: price}. Same testnet source
    #     as the harness (real trades settle on these); count is API-limited → it
    #     accumulates into the cache over repeated runs.
    try:
        r = requests.get(f"{base_url.rstrip('/')}/public/get_delivery_prices",
                         params={"index_name": "btc_usd", "count": count}, timeout=15)
        r.raise_for_status()
        data = r.json()["result"].get("data", [])
    except Exception as e:
        log.warning(f"fetch delivery prices fallito/failed: {type(e).__name__}: {e} "
                    f"— uso solo la cache / using cache only")
        return {}
    out = {}
    for rec in data:
        key = pd.Timestamp(rec["date"]).strftime("%d%b%y").upper()
        out[key] = float(rec["delivery_price"])
    return out


def delivery_for_expiry(expiry_ms: int, cache: dict) -> float | None:
    # IT: chiave DDMMMYY del giorno di settlement (08:00 UTC), come maybe_settle (04b).
    # EN: DDMMMYY key of the settlement day (08:00 UTC), as in maybe_settle (04b).
    key = datetime.fromtimestamp(expiry_ms / 1000, timezone.utc).strftime("%d%b%y").upper()
    return cache.get(key)


# ──────────────────────────── replay del loop 04b ────────────────────────────
def replay(forecasts: pd.DataFrame, chain: pd.DataFrame, delivery: dict, rule: str) -> dict:
    # IT: replay fedele del loop di 04b sul calendario forecasts, con regola d'ingresso
    #     parametrica. Semantica identica: max-1 posizione, settle quando candle_ts
    #     supera l'expiry e il delivery price è noto, hold-to-expiry.
    #       rule="nn"    : entra side=sign(edge) se |edge|>EDGE_THRESHOLD;
    #       rule="long"  : entra side=+1 a ogni occasione (nessun filtro su edge);
    #       rule="short" : entra side=−1 a ogni occasione.
    #     Un'occasione = riga con edge finito e straddle ricostruibile, nessuna pos aperta.
    # EN: faithful replay of the 04b loop over the forecasts calendar with a
    #     parametric entry rule. Identical semantics: max-1 position, settle once
    #     candle_ts passes expiry and the delivery price is known, hold-to-expiry.
    #       rule="nn"    : enter side=sign(edge) iff |edge|>EDGE_THRESHOLD;
    #       rule="long"  : enter side=+1 at every opportunity (no edge filter);
    #       rule="short" : enter side=−1 at every opportunity.
    #     An opportunity = row with finite edge and a reconstructable straddle, no open pos.
    trades, skipped_no_delivery = [], 0
    pos = None
    for _, r in forecasts.iterrows():
        ts = pd.Timestamp(r["candle_ts"])
        if pd.isna(ts.tz):
            ts = ts.tz_localize("UTC")

        # IT: settlement della posizione aperta se l'expiry è passata.
        # EN: settle the open position once its expiry has passed.
        if pos is not None and ts.value // 10**6 >= pos["expiry_ms"]:
            dp = delivery_for_expiry(pos["expiry_ms"], delivery)
            if dp is None:
                skipped_no_delivery += 1          # IT: expiry futura/non ancora pubblicata | EN: future/not-yet-published
            else:
                s = settle_pnl(pos["side"], dp, pos["strike"], pos["prem_call"], pos["prem_put"])
                trades.append({**pos, "delivery_price": dp, **s})
                pos = None

        # IT: ingresso (solo se flat, edge valido e straddle ricostruibile).
        # EN: entry (only if flat, edge finite and straddle reconstructable).
        if pos is None and np.isfinite(r.get("edge", np.nan)):
            edge = float(r["edge"])
            if rule == "nn":
                side = 1 if edge > EDGE_THRESHOLD else (-1 if edge < -EDGE_THRESHOLD else 0)
            elif rule == "long":
                side = 1
            elif rule == "short":
                side = -1
            else:
                raise ValueError(f"regola sconosciuta/unknown rule: {rule}")
            if side != 0:
                st = reconstruct_straddle(chain, ts)
                if st is not None:
                    pos = {"entry_ts": str(ts), "side": side, "edge": edge,
                           "expiry_ms": st["expiry_ms"], "strike": st["strike"],
                           "t_hours_at_entry": round(st["t_hours"], 2),
                           "prem_call": st["prem_call"], "prem_put": st["prem_put"]}

    pnls = np.array([t["pnl_btc"] for t in trades], dtype=float)
    return {
        "rule": rule, "n_trades": len(trades),
        "total_pnl_btc": float(pnls.sum()) if len(pnls) else 0.0,
        "mean_pnl_btc": float(pnls.mean()) if len(pnls) else float("nan"),
        "hit_rate": float((pnls > 0).mean()) if len(pnls) else float("nan"),
        "open_at_end": pos is not None,
        "skipped_no_delivery": skipped_no_delivery,
        "trades": trades,
    }


# ──────────────────────────── trade reali (gate 1 + 3) ────────────────────────────
def real_trade_stats() -> dict:
    # IT: statistiche dai trade REALI settlati (trades.jsonl) — sono questi a contare
    #     per il gate (1) P&L medio>0 e (3) hit-rate>0.5. Esclude i record non settlati.
    # EN: stats from the REAL settled trades (trades.jsonl) — these are what count for
    #     gate (1) mean P&L>0 and (3) hit-rate>0.5. Excludes unsettled records.
    if not TRADES_PATH.exists():
        return {"n_trades": 0, "note": "trades.jsonl assente / missing"}
    rows = [json.loads(l) for l in TRADES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    settled = [t for t in rows if "pnl_btc" in t and t.get("delivery_price") is not None]
    pnls = np.array([t["pnl_btc"] for t in settled], dtype=float)
    n_exec = sum(1 for t in settled if t.get("executed"))
    return {
        "n_trades": len(settled), "n_executed": n_exec,
        "n_simulated": len(settled) - n_exec,
        "total_pnl_btc": float(pnls.sum()) if len(pnls) else 0.0,
        "mean_pnl_btc": float(pnls.mean()) if len(pnls) else float("nan"),
        "hit_rate": float((pnls > 0).mean()) if len(pnls) else float("nan"),
    }


def main():
    # IT: boilerplate UTF-8 (checklist CLAUDE.md) | EN: UTF-8 boilerplate (CLAUDE.md checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Baseline always-long/short-vol del forward test vol-paper")
    ap.add_argument("--no-fetch", action="store_true",
                    help="non interrogare Deribit per i delivery price (usa solo la cache) / "
                         "do not query Deribit for delivery prices (cache only)")
    ap.add_argument("--min-trades", type=int, default=30,
                    help="N minimo di trade NN per dichiarare il gate valutabile (pre-reg: 30) / "
                         "min NN trades to declare the gate evaluable (pre-reg: 30)")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FORECASTS_PATH.exists():
        log.error("forecasts.parquet assente — il forward test ha girato? / missing — has the test run?")
        return
    forecasts = pd.read_parquet(FORECASTS_PATH).sort_values("candle_ts").reset_index(drop=True)
    chain = load_chain()

    # IT: delivery price: cache locale (accumulata) + refresh dall'endpoint testnet del harness.
    # EN: delivery prices: local cache (accumulated) + refresh from the harness testnet endpoint.
    delivery = load_delivery_cache()
    if not args.no_fetch:
        base = cfg.get("deribit_testnet", {}).get("endpoint")
        if base:
            fresh = fetch_delivery_prices(base)
            delivery.update(fresh)
            DELIVERY_CACHE.write_text(json.dumps(delivery, indent=2, sort_keys=True), encoding="utf-8")
            log.info(f"delivery price: {len(fresh)} freschi, {len(delivery)} in cache")
        else:
            log.warning("nessun endpoint deribit_testnet in config — solo cache / no endpoint — cache only")

    # IT: replay delle 3 strategie sullo stesso calendario.
    # EN: replay the 3 strategies over the same calendar.
    res = {rule: replay(forecasts, chain, delivery, rule) for rule in ("nn", "long", "short")}
    real = real_trade_stats()

    nn = res["nn"]
    evaluable = nn["n_trades"] >= args.min_trades
    # IT: gate (2): NN batte ENTRAMBE le baseline sul P&L totale.
    # EN: gate (2): NN beats BOTH baselines on total P&L.
    beats_long = nn["total_pnl_btc"] > res["long"]["total_pnl_btc"]
    beats_short = nn["total_pnl_btc"] > res["short"]["total_pnl_btc"]

    report = {
        "generated_ts": str(pd.Timestamp.now(tz="UTC").floor("s")),
        "n_forecast_rows": int(len(forecasts)),
        "calendar_span": [str(forecasts["candle_ts"].min()), str(forecasts["candle_ts"].max())],
        "min_trades_required": args.min_trades,
        "evaluable": bool(evaluable),
        "gates": {
            # IT: (1) e (3) sui trade REALI; (2) sul replay ricostruito.
            # EN: (1) and (3) on REAL trades; (2) on the reconstructed replay.
            "g1_real_mean_pnl_gt_0": bool(real.get("mean_pnl_btc", float("nan")) > 0)
                                     if real.get("n_trades") else None,
            "g2_nn_beats_both_baselines": bool(beats_long and beats_short),
            "g3_real_hit_rate_gt_0.5": bool(real.get("hit_rate", float("nan")) > 0.5)
                                       if real.get("n_trades") else None,
        },
        "real_trades": real,
        "reconstructed": {k: {kk: vv for kk, vv in v.items() if kk != "trades"}
                          for k, v in res.items()},
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # IT: riepilogo a console.
    # EN: console summary.
    log.info("─" * 64)
    log.info(f"CALENDARIO/CALENDAR: {len(forecasts)} righe forecast, "
             f"span {report['calendar_span'][0]} → {report['calendar_span'][1]}")
    log.info(f"TRADE REALI/REAL (trades.jsonl, gate 1+3): n={real.get('n_trades')} "
             f"(exec={real.get('n_executed', '?')}/sim={real.get('n_simulated', '?')}) "
             f"mean_pnl={real.get('mean_pnl_btc', float('nan')):+.5f} BTC "
             f"hit_rate={real.get('hit_rate', float('nan')):.3f}")
    for rule in ("nn", "long", "short"):
        v = res[rule]
        log.info(f"REPLAY {rule.upper():5s}: n={v['n_trades']:3d} "
                 f"total={v['total_pnl_btc']:+.5f} BTC mean={v['mean_pnl_btc']:+.5f} "
                 f"hit={v['hit_rate']:.3f} (skip_no_delivery={v['skipped_no_delivery']}, "
                 f"open_at_end={v['open_at_end']})")
    log.info(f"GATE (2) NN>LONG: {beats_long} · NN>SHORT: {beats_short} → "
             f"{'PASS' if (beats_long and beats_short) else 'FAIL'}")
    if not evaluable:
        log.warning(f"⚠ NON valutabile: {nn['n_trades']} trade NN ricostruiti < "
                    f"{args.min_trades} (pre-reg). Report scritto comunque (harness pronto).")
    log.info(f"→ {REPORT_PATH}")


if __name__ == "__main__":
    main()
