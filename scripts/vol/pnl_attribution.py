"""
A11 (ROADMAP_VOL_BOOK) — Attribution ex-post del PnL per trade del forward test vol.

IT: Decompone il PnL di ogni trade CHIUSO di `results/vol_paper/trades.jsonl` nei
    termini greek-driven, usando la serie diagnostica A6 (`exec_diag.jsonl`, tick
    orari con greeks venue per leg). Per ogni coppia di tick consecutivi con
    greeks completi, per leg (V_usd = mark_btc·S, convenzione inverse):

        ΔV_usd ≈ Δ·ΔS + ½·Γ·(ΔS)² + ν·Δiv + Θ·Δt_giorni + residuo

    e il residuo = ΔV_usd effettivo − spiegato (cross-greeks, salti, rumore mark
    testnet). Aggregato su leg e intervalli, ×side×amount → per-trade. Il PnL di
    gamma vs theta è LA verifica che i trade vincenti vincano per il motivo
    giusto (RV realizzata vs IV pagata), non per direzione/vega — rafforza
    l'interpretazione del gate n≥20 SENZA toccarlo (script read-only, offline).
    Coverage dichiarata: frazione dell'holding coperta da coppie di tick valide
    (i buchi = PC spento / greeks null su strike illiquidi — mai interpolati).
    ⚠ Conversione BTC: componenti USD / S di fine intervallo (approssimazione
    dichiarata, coerente al primo ordine con il PnL inverse).

EN: Decomposes each SETTLED trade's PnL from `results/vol_paper/trades.jsonl`
    into greek-driven terms, using the A6 diagnostic series (`exec_diag.jsonl`,
    hourly ticks with per-leg venue greeks). For each consecutive tick pair with
    complete greeks, per leg (V_usd = mark_btc·S, inverse convention):

        ΔV_usd ≈ Δ·ΔS + ½·Γ·(ΔS)² + ν·Δiv + Θ·Δt_days + residual

    residual = realized ΔV_usd − explained (cross-greeks, jumps, testnet mark
    noise). Aggregated over legs and intervals, ×side×amount → per-trade. The
    gamma-vs-theta PnL is THE check that winning trades win for the right reason
    (realized RV vs paid IV), not direction/vega — sharpens the n≥20 gate's
    interpretation WITHOUT touching it (read-only, offline). Declared coverage:
    fraction of the holding covered by valid tick pairs (gaps = PC off / null
    greeks on illiquid strikes — never interpolated).
    ⚠ BTC conversion: USD components / end-of-interval S (declared first-order
    approximation, consistent with inverse PnL).

Uso / Usage (dalla root di progetto / from the project root):
    python scripts/vol/pnl_attribution.py [--trades P] [--diag P] [--out P]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quantsys.utils import setup_logging                                      # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.pnl_attribution")

# IT: campi per-leg richiesti per un intervallo di attribution valido.
# EN: per-leg fields required for a valid attribution interval.
_REQ = ("mark", "underlying", "delta", "gamma", "vega", "theta", "mark_iv")


def _leg_ok(leg: dict) -> bool:
    # IT: leg utilizzabile ⇔ tutti i campi presenti e finiti (mai interpolare).
    # EN: usable leg ⇔ every field present and finite (never interpolate).
    return all(leg.get(k) is not None and np.isfinite(leg[k]) for k in _REQ)


def interval_attribution(leg0: dict, leg1: dict, dt_days: float) -> dict:
    # IT: attribution di UNA leg su UN intervallo (greeks di inizio intervallo,
    #     convenzione Taylor forward): componenti in USD per contratto.
    # EN: ONE leg over ONE interval (start-of-interval greeks, forward-Taylor
    #     convention): components in USD per contract.
    dS = leg1["underlying"] - leg0["underlying"]
    d_iv = leg1["mark_iv"] - leg0["mark_iv"]
    dv_usd = leg1["mark"] * leg1["underlying"] - leg0["mark"] * leg0["underlying"]
    out = {
        "delta_usd": leg0["delta"] * dS,
        "gamma_usd": 0.5 * leg0["gamma"] * dS ** 2,
        "vega_usd": leg0["vega"] * d_iv,
        "theta_usd": leg0["theta"] * dt_days,
        "dv_usd": dv_usd,
    }
    out["residual_usd"] = dv_usd - (out["delta_usd"] + out["gamma_usd"]
                                    + out["vega_usd"] + out["theta_usd"])
    return out


def load_diag(path: Path) -> list:
    # IT: exec_diag.jsonl → lista di record con ts parsato; righe corrotte scartate.
    # EN: exec_diag.jsonl → record list with parsed ts; corrupt lines dropped.
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                r["_ts"] = pd.Timestamp(r["ts"])
                rows.append(r)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return rows


def attribute_trade(trade: dict, diag: list) -> dict:
    # IT: attribution di un trade chiuso: filtra i tick A6 della SUA posizione
    #     (source=position, stessa strike/expiry) dentro [entry, exit], somma
    #     l'attribution per-intervallo sulle coppie valide, ×side×amount.
    #     exit = settled_ts per pin_close, expiry per settlement (payoff congelato).
    # EN: settled-trade attribution: filter the A6 ticks of ITS position
    #     (source=position, same strike/expiry) inside [entry, exit], sum the
    #     per-interval attribution over valid pairs, ×side×amount.
    #     exit = settled_ts for pin_close, expiry for settlement (frozen payoff).
    side = int(trade["side"])
    amount = float(trade.get("amount", 1.0))
    entry = pd.Timestamp(trade["entry_ts"])
    expiry = pd.Timestamp(int(trade["expiry_ms"]), unit="ms", tz="UTC")
    exit_ts = (pd.Timestamp(trade["settled_ts"])
               if trade.get("exit_mode") == "pin_close" else expiry)

    ticks = sorted((r for r in diag
                    if r.get("source") == "position"
                    and float(r.get("strike", np.nan)) == float(trade["strike"])
                    and int(r.get("expiry_ms", -1)) == int(trade["expiry_ms"])
                    and entry <= r["_ts"] <= exit_ts),
                   key=lambda r: r["_ts"])

    tot = {k: 0.0 for k in ("delta_usd", "gamma_usd", "vega_usd", "theta_usd",
                            "dv_usd", "residual_usd")}
    tot_btc = {k.replace("_usd", "_btc"): 0.0 for k in tot}
    covered_h, n_pairs = 0.0, 0
    for r0, r1 in zip(ticks, ticks[1:]):
        legs0 = {l["instrument"]: l for l in r0.get("legs", [])}
        legs1 = {l["instrument"]: l for l in r1.get("legs", [])}
        if set(legs0) != set(legs1) or not all(
                _leg_ok(legs0[i]) and _leg_ok(legs1[i]) for i in legs0):
            continue
        dt_days = (r1["_ts"] - r0["_ts"]).total_seconds() / 86400.0
        # IT: gap > 3h = tick mancanti (PC off): il Taylor per-intervallo non è
        #     più locale — l'intervallo si scarta e finisce nella non-coverage.
        # EN: gap > 3h = missing ticks (PC off): the per-interval Taylor is no
        #     longer local — the interval is dropped into non-coverage.
        if dt_days > 3.0 / 24.0:
            continue
        for inst in legs0:
            comp = interval_attribution(legs0[inst], legs1[inst], dt_days)
            s_end = legs1[inst]["underlying"]
            for k, v in comp.items():
                tot[k] += v
                tot_btc[k.replace("_usd", "_btc")] += v / s_end
        covered_h += dt_days * 24.0
        n_pairs += 1

    w = side * amount
    holding_h = max((exit_ts - entry).total_seconds() / 3600.0, 1e-9)
    return {
        "entry_ts": str(entry), "exit_mode": trade.get("exit_mode", "settlement"),
        "side": side, "amount": amount, "strike": float(trade["strike"]),
        "expiry": str(expiry), "edge_at_entry": float(trade.get("edge", np.nan)),
        "pnl_btc_realized": float(trade.get("pnl_btc", np.nan)),
        "fee_btc": float(trade.get("fee_btc", np.nan)),
        "n_pairs": n_pairs, "coverage_frac": round(covered_h / holding_h, 3),
        # IT: componenti firmate per la POSIZIONE (×side×amount).
        # EN: components signed for the POSITION (×side×amount).
        **{k: w * v for k, v in tot.items()},
        **{k: w * v for k, v in tot_btc.items()},
    }


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="A11 — attribution PnL gamma/theta/vega per trade (offline, read-only)")
    ap.add_argument("--trades", default="results/vol_paper/trades.jsonl")
    ap.add_argument("--diag", default="results/vol_paper/exec_diag.jsonl")
    ap.add_argument("--out", default="results/vol_paper/attribution.parquet")
    args = ap.parse_args()

    trades_path, diag_path = Path(args.trades), Path(args.diag)
    if not trades_path.exists():
        raise SystemExit(f"trades non trovato / not found: {trades_path}")
    if not diag_path.exists():
        raise SystemExit(f"exec_diag non trovato / not found: {diag_path}")

    trades = [json.loads(l) for l in open(trades_path, encoding="utf-8") if l.strip()]
    diag = load_diag(diag_path)
    log.info(f"{len(trades)} trade chiusi, {len(diag)} tick A6")

    rows = [attribute_trade(t, diag) for t in trades]
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("nessun trade da attribuire / no trade to attribute")
        return
    df.to_parquet(args.out, index=False)

    # IT: report console: per trade + totali (BTC, spazio del PnL realizzato).
    # EN: console report: per trade + totals (BTC, realized-PnL space).
    cols = ["entry_ts", "exit_mode", "side", "amount", "coverage_frac",
            "pnl_btc_realized", "gamma_btc", "theta_btc", "vega_btc",
            "delta_btc", "residual_btc"]
    pd.set_option("display.width", 200)
    print("\n=== A11 — attribution per trade (BTC) ===")
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:+.5f}" if abs(x) < 10 else f"{x:.2f}"))
    tot = df[["pnl_btc_realized", "gamma_btc", "theta_btc", "vega_btc",
              "delta_btc", "residual_btc"]].sum()
    n_lowcov = int((df["coverage_frac"] < 0.5).sum())
    print("\n=== totali (BTC) ===")
    print(tot.to_string(float_format=lambda x: f"{x:+.5f}"))
    print(f"\ngamma−theta (harvest RV-vs-IV): {tot['gamma_btc'] + tot['theta_btc']:+.5f} BTC")
    if n_lowcov:
        print(f"⚠ {n_lowcov} trade con coverage <50%: attribution parziale, "
              f"non conclusiva su quei trade / partial attribution on those trades")
    log.info(f"attribution scritta / written: {args.out}")


if __name__ == "__main__":
    main()
