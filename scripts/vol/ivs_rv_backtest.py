"""
ivs_rv_backtest.py — NET-of-spread backtest of smile-residual reversion (IVS relative value).

The scout (ivs_scout.py) found persistent/mean-reverting per-strike smile residuals (lag-1
autocorr 0.77) but a THIN margin over the round-trip bid-ask (~1.25×). Here I measure the
strategy's REAL PnL: per snapshot/expiry fit a quadratic smile in log-moneyness (liquid,
|log(K/F)|≤0.30), residual r_i = σ_i − fit_i; open |r_i|>θ (SHORT rich r>0, LONG cheap r<0,
1 vega-unit/leg), hold h hours, close. PnL_leg (vol-pts) = sign(r_i)·(r_i(t) − r_i(t+h))
[residual reversion, vega-neutral to the common smile shift] MINUS round-trip cost =
2×half-spread converted price→vol-pts via BS vega. Sweep θ×h. Blunt question: is net mean > 0?

Usage:  python scripts/vol/ivs_rv_backtest.py
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
OUT = ROOT / "results" / "vols" / "ivs_rv_backtest.json"

M_CAP = 0.30          # max |log(K/F)| in the smile fit
OI_MIN = 50.0         # liquidity gate (OI)
YEAR_H = 365.25 * 24  # hours per year
HALF_CAP = 10.0       # drop legs with IV half-spread >10 vol-pts = vega~0, NOT IV-tradable.


def norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def load_hourly():
    # load chain, keep 1 snapshot/hour (clean grid for the h-hour horizon).
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    if not files:
        sys.exit("no chain parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
    df["hour"] = df["snapshot_ts"].dt.floor("h")
    # one snapshot per hour (the first)
    first = df.groupby("hour")["snapshot_ts"].min().rename("keep")
    df = df.merge(first, left_on="hour", right_index=True)
    df = df[df["snapshot_ts"] == df["keep"]].copy()
    return df


def residuals_at(snap: pd.DataFrame) -> pd.DataFrame:
    # per expiry fit quadratic smile on liquids, return residuals + per-instrument half-spread
    # in vol-pts (price→IV via BS vega).
    out = []
    F = float(snap["underlying_price"].iloc[0])
    hour = snap["hour"].iloc[0]
    for exp, g in snap.groupby("expiry"):
        T_h = (exp - hour).total_seconds() / 3600.0
        if T_h < 2:                       # drop degenerate near-expiry
            continue
        T = T_h / YEAR_H
        g = g[(g["mark_iv"] > 0) & (g["open_interest"] >= OI_MIN) &
              (g["bid_price"] > 0) & (g["ask_price"] > g["bid_price"])].copy()
        if len(g) < 6:
            continue
        g["m"] = np.log(g["strike"] / F)
        g = g[g["m"].abs() <= M_CAP]
        if len(g) < 6:
            continue
        # quadratic fit (vol-pts)
        A = np.vstack([np.ones(len(g)), g["m"], g["m"] ** 2]).T
        coef, *_ = np.linalg.lstsq(A, g["mark_iv"].values, rcond=None)
        g["resid"] = g["mark_iv"].values - A @ coef
        # BS vega per unit of σ (decimal), in BTC (inverse options: vega_BTC = φ(d1)·√T)
        sig = g["mark_iv"].values / 100.0
        d1 = (-g["m"].values + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T) + 1e-12)
        vega_btc = norm_pdf(d1) * np.sqrt(T)            # ∂price_BTC/∂σ (per 1.00 of σ)
        vega_per_volpt = vega_btc / 100.0 + 1e-12       # per 1 vol-pt
        half_price = (g["ask_price"].values - g["bid_price"].values) / 2.0
        g["half_volpt"] = half_price / vega_per_volpt   # half-spread in vol-pt
        # drop untradeable low-vega legs
        g = g[g["half_volpt"] <= HALF_CAP]
        # columnar build, not iterrows (A2).
        out.append(pd.DataFrame({"hour": hour, "inst": g["instrument_name"].to_numpy(),
                                 "resid": g["resid"].to_numpy(float),
                                 "half_volpt": g["half_volpt"].to_numpy(float)}))
    if not out:
        return pd.DataFrame(columns=["hour", "inst", "resid", "half_volpt"])
    return pd.concat(out, ignore_index=True)


def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    df = load_hourly()
    hours = sorted(df["hour"].unique())
    print(f"=== IVS RV BACKTEST · {len(hours)} ore, {df['expiry'].nunique()} scadenze ===")

    # precompute residuals+spread per hour → dict {hour: DataFrame indexed by inst}.
    by_hour = {}
    for h in hours:
        r = residuals_at(df[df["hour"] == h])
        if not r.empty:
            by_hour[h] = r.set_index("inst")
    hrs = sorted(by_hour.keys())
    hr_set = {h: i for i, h in enumerate(hrs)}

    results = []
    for H in (1, 3, 6):                       # holding horizon (h)
        for TH in (0.0, 0.5, 1.0, 1.5, 2.0):  # |residual| threshold (vol-pts)
            legs_gross, legs_net = [], []
            for h0 in hrs:
                h1 = h0 + pd.Timedelta(hours=H)
                if h1 not in by_hour:
                    continue                  # coverage gap → skip
                e0, e1 = by_hour[h0], by_hour[h1]
                common = e0.index.intersection(e1.index)
                if len(common) == 0:
                    continue
                # vectorized .loc[common] instead of per-inst .at[] (A2) — bit-identical, same
                # |r0|>TH filter, same gross/cost; mean/sum/std are order-invariant.
                r0 = e0.loc[common, "resid"].to_numpy()
                r1 = e1.loc[common, "resid"].to_numpy()
                hv = e0.loc[common, "half_volpt"].to_numpy()
                sel = np.abs(r0) > TH                          # residual reversion (vol-pts)
                if not sel.any():
                    continue
                gross = np.sign(r0[sel]) * (r0[sel] - r1[sel])
                cost = 2.0 * hv[sel]                           # round-trip half-spread
                legs_gross.extend(gross.tolist())
                legs_net.extend((gross - cost).tolist())
            n = len(legs_net)
            if n < 10:
                continue
            g = np.array(legs_gross); nt = np.array(legs_net)
            results.append({
                "H_h": H, "theta": TH, "n": n,
                "gross_mean": float(g.mean()), "net_mean": float(nt.mean()),
                "net_total": float(nt.sum()), "hit_net": float((nt > 0).mean()),
                "sharpe_net": float(nt.mean() / (nt.std() + 1e-9)),
                "mean_cost": float((g - nt).mean()),
            })

    res = pd.DataFrame(results)
    print("\n  H(h) θ(vp)   n   gross_mean  net_mean   net_tot  hit%  Sharpe  cost")
    for _, r in res.iterrows():
        print(f"   {int(r.H_h):2d}  {r.theta:4.1f} {int(r.n):5d}  "
              f"{r.gross_mean:+8.3f}  {r.net_mean:+8.3f} {r.net_total:+8.2f} "
              f"{100*r.hit_net:4.0f}  {r.sharpe_net:+6.3f} {r.mean_cost:5.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(res.to_json(orient="records", indent=2))
    print(f"\n  → {OUT.relative_to(ROOT)}")
    # synthetic verdict
    best = res.loc[res["net_mean"].idxmax()] if not res.empty else None
    if best is not None:
        print(f"  BEST net_mean: {best.net_mean:+.3f} vol-pt/leg @ H={int(best.H_h)}h θ={best.theta} "
              f"(n={int(best.n)}, hit {100*best.hit_net:.0f}%, Sharpe {best.sharpe_net:+.2f})")


if __name__ == "__main__":
    main()
