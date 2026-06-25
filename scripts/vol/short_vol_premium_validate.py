"""
short_vol_premium_validate.py — verifica di robustezza del backtest storico short-vol.
short_vol_premium_validate.py — robustness check for the historical short-vol backtest.

IT: Il backtest `short_vol_hist_backtest.py` prezza il premio con FHS GJR-GARCH (no superficie IV
    storica). Punto debole: il break-even VRP=0 dipende dal realismo di quel premio modellato. Qui
    lo rendiamo *hard*: sui 12 giorni di chain REALE (`data/iv/chain`) confrontiamo, per le stesse
    scadenze/strike che venderemmo, il FAIR VALUE FHS contro il MARK e il BID Deribit reali.
      • FHS ≈ mark  → premio storico AFFIDABILE, il backtest regge.
      • FHS ≫/≪ mark → bias, da ricalibrare.
    Misura anche l'half-spread reale (mark−bid)/mark = haircut bid da applicare al backtest storico
    per renderlo net-of-cost e apples-to-apples col braccio live.
EN: The backtest prices the premium with FHS GJR-GARCH (no historical IV surface). Weak point: the
    VRP=0 break-even hinges on that modeled premium being realistic. Here we make it *hard*: over the
    12 real chain days we compare, for the same expiries/strikes we'd sell, the FHS FAIR VALUE vs the
    real Deribit MARK and BID. FHS≈mark → premium trustworthy; else bias. Also measures the real
    half-spread = bid haircut to apply to the historical backtest.

IT: CAUSALE: σ_entry e residui FHS solo da candele ≤ snapshot d'ingresso. n piccolo (~overlap candele
    ↔ chain) ⇒ è un check di BIAS, non una statistica large-sample.
EN: CAUSAL: σ_entry and FHS residuals only from candles ≤ entry snapshot. Small n (candle↔chain
    overlap) ⇒ a BIAS check, not a large-sample statistic.

Uso / usage:  python scripts/vol/short_vol_premium_validate.py
"""
import sys
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_vol_hist_backtest import (  # noqa: E402  (riuso kernel identico al backtest | reuse identical kernel)
    fit_gjr, fhs_fair_value, TENOR_H, EXPIRY_HOUR,
)

ROOT = Path(__file__).resolve().parents[2]
CANDLES = ROOT / "data" / "raw_candles.parquet"
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
OUT = ROOT / "results" / "vols" / "short_vol_premium_validate.json"
ENTRY_TOL_H = 6.0
N_PATHS = 6000


def load_chain():
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    if not files:
        sys.exit("no chain parquet in data/iv/chain")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    return df


def pick_leg(snap, opt_type, target_K):
    # IT: leg col tipo richiesto, strike più vicino, mark valido; tiene mark+bid. | EN: nearest valid leg, mark+bid.
    cand = snap[(snap["option_type"].str.upper().str[0] == opt_type) & (snap["mark_price"] > 0)]
    if cand.empty:
        return None
    row = cand.iloc[(cand["strike"] - target_K).abs().argmin()]
    bid = float(row["bid_price"]) if pd.notna(row["bid_price"]) else np.nan
    return {"K": float(row["strike"]), "mark": float(row["mark_price"]), "bid": bid}


def causal_garch_at(r, entry_idx, fit_window=24 * 365 * 2):
    # IT: fit GJR causale (≤ entry, finestra cappata) + recursion fino a entry → (σ_entry, z_pool, params).
    # EN: causal GJR fit (≤ entry, capped window) + recursion up to entry → (σ_entry, z_pool, params).
    lo = max(1, entry_idx - fit_window)
    prm = fit_gjr(r[lo:entry_idx])
    if prm is None:
        return None
    sig = np.empty(entry_idx + 1)
    sig[lo] = np.sqrt(prm["uncond"])
    for t in range(lo + 1, entry_idx + 1):
        neg = 1.0 if r[t - 1] < 0 else 0.0
        v = prm["omega"] + (prm["alpha"] + prm["gamma"] * neg) * r[t - 1] ** 2 + prm["beta"] * sig[t - 1] ** 2
        sig[t] = np.sqrt(max(v, 1e-12))
    z = r[lo + 1:entry_idx + 1] / sig[lo + 1:entry_idx + 1]
    z = z[np.isfinite(z)]
    return float(sig[entry_idx]), z, prm


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    rng = np.random.default_rng(7)

    cnd = pd.read_parquet(CANDLES)[["open_time", "close"]].copy()
    cnd["open_time"] = pd.to_datetime(cnd["open_time"], utc=True)
    cnd = cnd.sort_values("open_time").reset_index(drop=True)
    times = cnd["open_time"]
    close = cnd["close"].to_numpy(float)
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    cand_end = times.iloc[-1]

    chain = load_chain()
    snap_min, snap_max = chain["snapshot_ts"].min(), chain["snapshot_ts"].max()
    print("=== PREMIUM VALIDATION (FHS fair-value vs Deribit mark/bid reale) ===")
    print(f"  candele fino a {cand_end} | chain {snap_min.date()}→{snap_max.date()} | n_paths={N_PATHS}\n")

    widths = [("straddle", 0.0), ("strangle", 0.04), ("strangle", 0.06),
              ("strangle", 0.08), ("strangle", 0.10)]
    exps = sorted(e for e in chain["expiry"].unique() if pd.Timestamp(e).hour == EXPIRY_HOUR)

    rows = []
    for exp in exps:
        exp = pd.Timestamp(exp)
        entry_t = exp - timedelta(hours=TENOR_H)
        if entry_t < snap_min or entry_t > cand_end:        # serve candela ≤ entry per σ | need candle ≤ entry
            continue
        sub = chain[chain["expiry"] == exp]
        snaps = sub["snapshot_ts"].unique()
        if len(snaps) == 0:
            continue
        s_entry = min(snaps, key=lambda s: abs((pd.Timestamp(s) - entry_t).total_seconds()))
        if abs((pd.Timestamp(s_entry) - entry_t).total_seconds()) / 3600.0 > ENTRY_TOL_H:
            continue
        snap = sub[sub["snapshot_ts"] == s_entry]
        spot = float(snap["underlying_price"].iloc[0])
        # IT: barra candela più vicina allo snapshot (≤) per σ causale | EN: nearest candle bar (≤) for causal σ
        entry_idx = int((times <= pd.Timestamp(s_entry)).to_numpy().nonzero()[0][-1])
        g = causal_garch_at(r, entry_idx)
        if g is None:
            continue
        sig_entry, zpool, prm = g
        if len(zpool) < 200:
            continue
        for struct, w in widths:
            Kc, Kp = (spot * (1 + w), spot * (1 - w)) if struct == "strangle" else (spot, spot)
            lc, lp = pick_leg(snap, "C", Kc), pick_leg(snap, "P", Kp)
            if not lc or not lp:
                continue
            mark = lc["mark"] + lp["mark"]
            bid = (lc["bid"] if np.isfinite(lc["bid"]) else lc["mark"]) + \
                  (lp["bid"] if np.isfinite(lp["bid"]) else lp["mark"])
            # IT: FHS prezzato sugli strike REALI scelti dalla chain (non i target teorici).
            # EN: FHS priced on the REAL chosen strikes (not the theoretical targets).
            fv = fhs_fair_value(spot, lc["K"], lp["K"], sig_entry, zpool, prm, N_PATHS, rng)
            rows.append({"expiry": exp.isoformat(), "struct": struct, "width": w,
                         "fhs": fv, "mark": mark, "bid": bid,
                         "fhs_over_mark": fv / mark if mark > 0 else np.nan,
                         "half_spread": (mark - bid) / mark if mark > 0 else np.nan})

    if not rows:
        sys.exit("nessun overlap candela↔chain simulabile")
    df = pd.DataFrame(rows)

    print(f"  {'struct':<13} {'n':>2} | {'FHS':>9} {'mark':>9} {'bid':>9} | {'FHS/mark':>9} {'half-spr':>9}")
    print("  " + "-" * 70)
    summ = []
    for (struct, w), g in df.groupby(["struct", "width"]):
        tag = f"strangle {w:.0%}" if struct == "strangle" else "straddle ATM"
        rr = {"struct": struct, "width": w, "n": int(len(g)),
              "fhs_mean": float(g["fhs"].mean()), "mark_mean": float(g["mark"].mean()),
              "bid_mean": float(g["bid"].mean()),
              "fhs_over_mark_median": float(g["fhs_over_mark"].median()),
              "half_spread_median": float(g["half_spread"].median())}
        summ.append(rr)
        print(f"  {tag:<13} {rr['n']:>2} | {rr['fhs_mean']:>9.5f} {rr['mark_mean']:>9.5f} "
              f"{rr['bid_mean']:>9.5f} | {rr['fhs_over_mark_median']:>8.2f}x {rr['half_spread_median']:>8.1%}")

    overall_ratio = float(df["fhs_over_mark"].median())
    overall_hs = float(df["half_spread"].median())
    print(f"\n  → FHS/mark mediano COMPLESSIVO = {overall_ratio:.2f}x | half-spread mediano = {overall_hs:.1%}")
    if 0.85 <= overall_ratio <= 1.15:
        verdict = "FHS ≈ mark reale (±15%) → premio storico AFFIDABILE, backtest regge."
    elif overall_ratio < 0.85:
        verdict = "FHS SOTTO il mark → backtest CONSERVATIVO (premio reale > modellato)."
    else:
        verdict = "FHS SOPRA il mark → backtest OTTIMISTA (premio modellato gonfiato), ricalibrare."
    print(f"  VERDETTO: {verdict}")
    print(f"  Haircut bid da applicare al backtest storico ≈ {overall_hs:.1%} del premio.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"overall": {"fhs_over_mark_median": overall_ratio,
                                           "half_spread_median": overall_hs, "verdict": verdict},
                               "by_struct": summ, "rows": rows}, indent=2))
    print(f"\n  dettaglio in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
