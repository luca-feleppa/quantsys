"""
short_vol_hist_backtest.py — STRUCTURAL HISTORICAL backtest of the SHORT-VOL arm (vol line).

Complements the live study `short_vol_arm.py` (capped at n≈4 collected expiries). Here we use
7 years of hourly BTC candles (2019→2026, ~2700 daily 08:00-UTC expiries) to measure whether
selling 30h strangles/straddles survives the HISTORICAL tails (2020/21/22 crashes) — exactly
what the 12 calm days of the live test cannot show.

DATA CONSTRAINT: no historical BTC implied-vol surface exists (only 12 collected days). So this
is NOT "I sold at the real mark" but a STRUCTURAL TAIL study + VRP SWEEP:
  • PAYOFF side  = real, from candles (no model, tails included).
  • PREMIUM side = fat-tailed risk-neutral-ish fair value × (1 + VRP), VRP swept.
Kernel = FHS on GJR-GARCH(1,1): NO Gaussian assumption (≠ Black-Scholes, whose flat vol misprices
smile and tails = exactly where the OTM strangle payoff lives). REAL standardized residuals are
bootstrapped → empirical tails/skew/clustering. Fully CAUSAL (params/residuals only ≤ entry;
expanding refit on a cadence, no look-ahead).

Deliverable = BREAK-EVEN VRP curve per (struct × width): "the W% strangle over 2019-26 is net
positive iff implied/realized ≥ X", with tail distribution and worst-trade contribution.

SIGNIFICANCE (FIX ①+②, 2026-06-26). DAILY expiries (24h) but 30h tenor ⇒ overlapping windows and
shared vol-clustering ⇒ NON-i.i.d. PnL: the mean's SE is understated. (a) Sharpe now annualized
with sqrt(trades/year) from REAL timestamps (≈365/yr), not sqrt(8760/30)≈sqrt(292); (b) the HONEST
statistic is the MOVING BLOCK BOOTSTRAP (block≈21 trades≈1 month, B=2000) → 5–95% CI of mean and
Sharpe + N_eff = N·(1−ρ₁)/(1+ρ₁). Measured: PnL ρ₁ ≈ −0.01 (near-zero) ⇒ N_eff ≈ N and CIs stay
away from 0 (e.g. straddle VRP=0: mean [+0.003,+0.014], Sharpe [+0.7,+2.4]) → the edge survives the
overlap correction. ⚠ Caveat: the bootstrap does NOT capture TEMPORAL concentration (2020-21 ≈90%
of PnL); the real gate remains live n≥20.

MARTINGALE CORRECTION (FIX ⑤, inert flag --mart-correct, default OFF). The Gaussian drift −0.5σ²
enforces E[S_T]=spot only for normal z; with fat-tailed FHS residuals the flag swaps the drift for
the cumulant correction −(½σ²+skew·σ³/6+exkurt·σ⁴/24). MEASURED OUTCOME (expected ~1e-4, REFUTED):
Δmean@VRP=0 ≈ +5e-3 (64–85% of fair value!), NOT negligible. Diagnosed cause (NOT a bug): BTC z
residuals have exkurt≈19.7; though the entry-vol correction is ~1e-7, inside the sim var_t evolves
via GJR and the high-vol paths (which DOMINATE the OTM strangle's tail-driven fair value) get drift
shifts up to ~0.4 — the σ⁴ term explodes exactly where the 4th-order cumulant truncation is INVALID.
Conclusion: the correction OVER-corrects on extreme-tail data ⇒ default OFF confirmed correct; the
flag stays inert/documented.

Usage:  python scripts/vol/short_vol_hist_backtest.py
        python scripts/vol/short_vol_hist_backtest.py --n-paths 6000 --refit-days 60
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
CANDLES = ROOT / "data" / "raw_candles.parquet"
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
OUT = ROOT / "results" / "vols" / "short_vol_hist_backtest.json"

TENOR_H = 30                 # tenor in hourly bars
EXPIRY_HOUR = 8             # Deribit daily expiries 08:00 UTC
FEE_PER_LEG = 0.0003        # ~3e-4 BTC/leg
FEE_CAP_FRAC = 0.125        # Deribit cap = 12.5% of premium/leg
VRP_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]  # VRP grid
ANNUAL_BARS = 24 * 365      # bars/year for annualized Sharpe


# ─────────────────────────────────────────────────────────────────────────────
# GJR-GARCH(1,1) — Gaussian QMLE with variance targeting (causal params for FHS)
# ─────────────────────────────────────────────────────────────────────────────
def gjr_recursion(r, omega, alpha, gamma, beta, var0):
    # σ²_t = ω + (α + γ·I[r_{t-1}<0])·r²_{t-1} + β·σ²_{t-1}. Returns σ_t (NOT σ²).
    n = len(r)
    v = np.empty(n)
    v[0] = var0
    for t in range(1, n):
        shock = r[t - 1] ** 2
        neg = 1.0 if r[t - 1] < 0 else 0.0
        v[t] = omega + (alpha + gamma * neg) * shock + beta * v[t - 1]
    return np.sqrt(np.maximum(v, 1e-12))


def fit_gjr(r, warm=None):
    # estimate (α, γ, β) via QMLE; ω from variance targeting → ω = σ̄²·(1 − α − γ/2 − β).
    # Gaussian QML = consistent even under non-normal innovations (FHS supplies the tails).
    # warm = previous-refit (α,γ,β) for warm-start (single optimization → ~20× faster than the
    # multi-restart on an expanding window, which used to blow the timeout).
    r = np.asarray(r, dtype=float)
    uncond = float(np.var(r))
    if uncond <= 0 or not np.isfinite(uncond):
        return None

    def nll(theta):
        alpha, gamma, beta = theta
        persist = alpha + 0.5 * gamma + beta
        if alpha < 0 or beta < 0 or (alpha + gamma) < 0 or persist >= 0.999:
            return 1e10
        omega = uncond * (1.0 - persist)
        if omega <= 0:
            return 1e10
        sig = gjr_recursion(r, omega, alpha, gamma, beta, uncond)
        # Gaussian (QML) negative log-likelihood
        ll = -0.5 * np.sum(np.log(2 * np.pi * sig ** 2) + (r / sig) ** 2)
        return -ll if np.isfinite(ll) else 1e10

    # warm-start else repo 1m params.
    starts = [warm] if warm is not None else [[0.05, 0.065, 0.875], [0.03, 0.10, 0.86]]
    best = None
    for x0 in starts:
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})
        if best is None or res.fun < best.fun:
            best = res
    alpha, gamma, beta = best.x
    persist = alpha + 0.5 * gamma + beta
    omega = uncond * (1.0 - persist)
    return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
            "uncond": uncond, "persist": persist}


# ─────────────────────────────────────────────────────────────────────────────
# FHS pricing: P-measure fair value of the 30h inverse strangle/straddle
# ─────────────────────────────────────────────────────────────────────────────
def fhs_fair_value(spot, Kc, Kp, sig_entry, z_pool, params, n_paths, rng,
                   mart_correct=False, skew=0.0, exkurt=0.0):
    # simulate n_paths × TENOR_H steps. Each step ε=σ·z (z bootstrap from REAL residuals ≤ entry),
    # drift = −0.5σ² (martingale: E[S]≈spot), then GJR variance update. Fair value = mean inverse
    # payoff at terminal. Inverse option ⇒ BTC payoff = intrinsic/S_T.
    # FIX ⑤ (inert flag, default OFF). The Gaussian drift −0.5σ² enforces E[exp(ε)]=1 ONLY for
    # normal z; with fat-tailed/skewed bootstrap residuals (the whole point of FHS) E[S_T]≠spot.
    # The empirical cumulant correction subtracts skew/kurtosis terms → E[S_T]≈spot off-normal.
    # skew/exkurt = standardized cumulants of the CAUSAL z pool (≤entry).
    omega, alpha, gamma, beta = params["omega"], params["alpha"], params["gamma"], params["beta"]
    var_t = np.full(n_paths, sig_entry ** 2)
    cum = np.zeros(n_paths)
    z_idx = rng.integers(0, len(z_pool), size=(TENOR_H, n_paths))
    for t in range(TENOR_H):
        sig = np.sqrt(var_t)
        eps = sig * z_pool[z_idx[t]]
        if mart_correct:
            # drift = −(½σ² + skew·σ³/6 + exkurt·σ⁴/24): Cornish-Fisher/cumulant expansion
            # of log E[exp(ε)] truncated at 4th order.
            cum += -(0.5 * var_t + skew * sig ** 3 / 6.0 + exkurt * sig ** 4 / 24.0) + eps
        else:
            cum += -0.5 * var_t + eps                   # Gaussian martingale drift + shock
        neg = (eps < 0).astype(float)
        var_t = omega + (alpha + gamma * neg) * eps ** 2 + beta * var_t
        var_t = np.clip(var_t, 1e-12, 0.5 ** 2)
    S_T = spot * np.exp(cum)
    pay_c = np.maximum(0.0, S_T - Kc) / S_T            # inverse call
    pay_p = np.maximum(0.0, Kp - S_T) / S_T            # inverse put
    return float(np.mean(pay_c + pay_p))


def realized_payoff(Kc, Kp, S_del):
    # real inverse payoff at delivery.
    return max(0.0, S_del - Kc) / S_del + max(0.0, Kp - S_del) / S_del


def fee_btc(leg_premium):
    # per-leg fee, 12.5% cap.
    return min(FEE_PER_LEG, FEE_CAP_FRAC * max(leg_premium, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────
# causal-series cache: depends ONLY on (refit_days, min_train) → identical across configs
# (struct/width) → avoids re-fitting the 7y GARCH N times (review item A1).
_CAUSAL_CACHE = {}


def precompute_causal(refit_days, min_train):
    # computes ONCE candles, log-ret, causal σ_t and residuals z_t + expiry indices.
    # Heavy (expanding GJR refit) but independent of struct/width → cacheable.
    key = (refit_days, min_train)
    if key in _CAUSAL_CACHE:
        return _CAUSAL_CACHE[key]
    df = pd.read_parquet(CANDLES)[["open_time", "close"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    close = df["close"].to_numpy(float)
    r = np.diff(np.log(close))                          # hourly log-ret
    r = np.concatenate([[0.0], r])                      # align length to close
    times = df["open_time"]

    # indices of 08:00-UTC bars = daily expiries; entry = TENOR_H bars earlier.
    exp_idx = np.where((times.dt.hour == EXPIRY_HOUR).to_numpy())[0]

    # CAUSAL σ_t and standardized residuals z_t: refit params every refit_days on an expanding
    # window, propagate the recursion forward with current params (no look-ahead).
    n = len(r)
    sig = np.full(n, np.nan)
    params_at = [None] * n
    cur = None
    last_fit = -10 ** 9
    refit_bars = refit_days * 24
    fit_window = 24 * 365 * 2                            # fit on last 2y
    for t in range(n):
        if t >= min_train and (cur is None or t - last_fit >= refit_bars):
            lo = max(1, t - fit_window)
            warm = [cur["alpha"], cur["gamma"], cur["beta"]] if cur is not None else None
            fit = fit_gjr(r[lo:t], warm=warm)           # past only, capped window
            if fit is not None:
                cur = fit
                last_fit = t
        params_at[t] = cur
        if cur is None:
            continue
        if not np.isfinite(sig[t - 1]):
            sig[t] = np.sqrt(cur["uncond"])
        else:
            neg = 1.0 if r[t - 1] < 0 else 0.0
            v = cur["omega"] + (cur["alpha"] + cur["gamma"] * neg) * r[t - 1] ** 2 + cur["beta"] * sig[t - 1] ** 2
            sig[t] = np.sqrt(max(v, 1e-12))
    z = np.where(np.isfinite(sig) & (sig > 0), r / sig, np.nan)  # std residuals

    pre = {"times": times, "close": close, "sig": sig, "z": z,
           "params_at": params_at, "exp_idx": exp_idx, "min_train": min_train}
    _CAUSAL_CACHE[key] = pre
    return pre


def price_trades(pre, width, struct, n_paths, rng, mart_correct=False):
    # prices expiries on an ALREADY-computed causal series (only strike/FHS-pricing vary).
    # rng stays PER-CONFIG (caller-supplied) → FHS paths bit-identical to legacy behaviour.
    times, close = pre["times"], pre["close"]
    sig, z, params_at = pre["sig"], pre["z"], pre["params_at"]
    min_train = pre["min_train"]
    trades = []
    for ei in pre["exp_idx"]:
        entry = ei - TENOR_H
        if entry < min_train or not np.isfinite(sig[entry]):
            continue
        prm = params_at[entry]
        if prm is None:
            continue
        zp = z[1:entry]
        zp = zp[np.isfinite(zp)]
        if len(zp) < 200:                               # thin residual pool
            continue
        # FIX ⑤. Standardized cumulants of the CAUSAL z pool (≤entry): skew + excess kurtosis
        # computed ONCE per expiry from the pool, fed to the drift correction (no look-ahead).
        if mart_correct:
            zc = zp - zp.mean()
            m2 = float(np.mean(zc ** 2))
            skew = float(np.mean(zc ** 3) / m2 ** 1.5) if m2 > 0 else 0.0
            exkurt = float(np.mean(zc ** 4) / m2 ** 2 - 3.0) if m2 > 0 else 0.0
        else:
            skew = exkurt = 0.0
        spot = close[entry]
        # delivery proxy = single 08:00-UTC close. Real Deribit delivery is a 30-min TWAP →
        # non-distorting noise (zero-mean vs the close), not a systematic bias.
        S_del = close[ei]
        if struct == "strangle":
            Kc, Kp = spot * (1 + width), spot * (1 - width)
        else:                                           # straddle ATM
            Kc = Kp = spot
        fv = fhs_fair_value(spot, Kc, Kp, sig[entry], zp, prm, n_paths, rng,
                            mart_correct=mart_correct, skew=skew, exkurt=exkurt)
        rp = realized_payoff(Kc, Kp, S_del)
        trades.append({"t_entry": times.iloc[entry].isoformat(), "spot": spot, "S_del": S_del,
                       "fair_value": fv, "realized_payoff": rp,
                       "moved_pct": 100 * (S_del / spot - 1), "sig_entry": float(sig[entry])})
    return pd.DataFrame(trades)


def run(width, struct, n_paths, refit_days, min_train, seed, mart_correct=False):
    # back-compat wrapper.
    rng = np.random.default_rng(seed)
    pre = precompute_causal(refit_days, min_train)
    return price_trades(pre, width, struct, n_paths, rng, mart_correct=mart_correct)


def block_bootstrap_ci(pnl, ann_factor, block=21, B=2000, seed=12345):
    # FIX ①. Expiries are DAILY (24h spacing) but tenor is 30h → overlapping windows and shared
    # vol-clustering ⇒ PnL are NOT i.i.d.: the mean's SE (hence Sharpe/hit) is understated and
    # significance overstated (nominal n ≫ effective N; 2020-21 ≈90% of PnL). Moving block bootstrap
    # on TIME-ORDERED PnL: resample ceil(N/block) contiguous blocks of length `block` (≈21 trades
    # ≈1 month, preserves intra-block clustering) with replacement, trim to N, recompute mean and
    # annualized Sharpe; B repeats → 5–95% CI. N_eff = N·(1−ρ₁)/(1+ρ₁) (ρ₁ = lag-1 PnL autocorr).
    pnl = np.asarray(pnl, dtype=float)
    n = len(pnl)
    # lag-1 autocorr → N_eff (clamp [1,N]).
    xc = pnl - pnl.mean()
    denom = float(np.sum(xc * xc))
    rho1 = float(np.sum(xc[1:] * xc[:-1]) / denom) if (n > 1 and denom > 0) else 0.0
    n_eff = n * (1.0 - rho1) / (1.0 + rho1) if (1.0 + rho1) > 0 else float(n)
    n_eff = float(min(max(n_eff, 1.0), n))
    if n < 2:
        m = float(pnl.mean()) if n else 0.0
        return {"mean_ci": [m, m], "sharpe_ci": [0.0, 0.0], "n_eff": float(n), "rho1": rho1}
    blk = int(min(block, n))                              # block ≤ N
    n_blocks = int(np.ceil(n / blk))
    max_start = n - blk                                  # last valid block start
    rng = np.random.default_rng(seed)
    # draw ALL B×n_blocks starts at once → indices (B, n_blocks·blk) trimmed to N (vectorized).
    starts = rng.integers(0, max_start + 1, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, n_blocks * blk)[:, :n]
    samp = pnl[idx]                                      # (B, N) bootstrap replicates
    means = samp.mean(axis=1)
    sds = samp.std(axis=1, ddof=1)
    sharpes = np.where(sds > 0, means / sds * ann_factor, 0.0)
    return {"mean_ci": [float(np.percentile(means, 5)), float(np.percentile(means, 95))],
            "sharpe_ci": [float(np.percentile(sharpes, 5)), float(np.percentile(sharpes, 95))],
            "n_eff": n_eff, "rho1": rho1}


def vrp_table(tr, struct, width):
    # per VRP grid point → premium = fair_value·(1+VRP), net PnL, stats + break-even.
    out = []
    fv = tr["fair_value"].to_numpy()
    rp = tr["realized_payoff"].to_numpy()
    # FIX ②. trades/year from REAL timestamps (DAILY cadence ≈365/yr), NOT the 292 30h-cycles
    # implied by sqrt(ANNUAL_BARS/TENOR_H), which is inconsistent with the real spacing. i.i.d.
    # annualization = sqrt(trades_per_year). ⚠ Overlap (30h tenor > 24h spacing) induces positive
    # autocorrelation that INFLATES the i.i.d. Sharpe → the HONEST statistic is the block-bootstrap
    # Sharpe-CI (FIX ①), not this point estimate. ANNUAL_BARS kept defined (external imports).
    t_entry = pd.to_datetime(tr["t_entry"], utc=True)
    span_years = (t_entry.max() - t_entry.min()).total_seconds() / (365.25 * 24 * 3600.0)
    trades_per_year = (len(tr) / span_years) if span_years > 0 else (ANNUAL_BARS / TENOR_H)
    ann_factor = float(np.sqrt(trades_per_year))
    for vrp in VRP_GRID:
        prem = fv * (1.0 + vrp)
        # fee on 2 legs ≈ premium/2 each (12.5% cap), vectorized. The prem/2 split ignores the
        # inter-strike skew (call/put at different IVs) → negligible: fees are tiny and capped at
        # 12.5% of premium/leg.
        fees = 2.0 * np.minimum(FEE_PER_LEG, FEE_CAP_FRAC * np.maximum(prem / 2.0, 0.0))
        pnl = prem - rp - fees
        mean, sd = float(pnl.mean()), float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = float(mean / sd * ann_factor) if sd > 0 else 0.0
        worst = float(np.sort(pnl)[:5].sum())           # sum of 5 worst
        # overlap/clustering-robust CI + N_eff (FIX ①).
        boot = block_bootstrap_ci(pnl, ann_factor)
        out.append({"vrp": vrp, "n": int(len(pnl)), "tot": float(pnl.sum()), "mean": mean,
                    "hit": float(100 * (pnl > 0).mean()), "sharpe_ann": sharpe,
                    "worst5_sum": worst, "p05": float(np.percentile(pnl, 5)),
                    "trades_per_year": float(trades_per_year),
                    "mean_ci05": boot["mean_ci"][0], "mean_ci95": boot["mean_ci"][1],
                    "sharpe_ci05": boot["sharpe_ci"][0], "sharpe_ci95": boot["sharpe_ci"][1],
                    "n_eff": boot["n_eff"], "rho1": boot["rho1"]})
    return out


def breakeven_vrp(table):
    # minimal VRP s.t. mean PnL ≥ 0 (linear interp between the two points straddling 0).
    xs = [(row["vrp"], row["mean"]) for row in table]
    for (v0, m0), (v1, m1) in zip(xs, xs[1:]):
        if m0 < 0 <= m1:
            return round(v0 + (v1 - v0) * (-m0) / (m1 - m0), 4)
    if xs[0][1] >= 0:
        return 0.0
    return None                                         # not reached in grid


def empirical_vrp_anchor():
    # anchor the sweep to the VRP observed over the 12 REAL days (implied mark_iv vs 30h realized).
    # Returns mean implied/realized over available ATM snapshots (None if no chain).
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    if not files:
        return None
    try:
        ch = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        atm = ch[ch["mark_iv"] > 0]
        if atm.empty:
            return None
        # mean ATM implied (proxy) — coherent 30h realized lives in the backtest; here just IV level.
        return {"mean_mark_iv_pct": float(atm["mark_iv"].median())}
    except Exception:
        return None


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=4000, help="path MC per scadenza")
    ap.add_argument("--refit-days", type=int, default=90, help="cadenza refit GJR-GARCH (giorni)")
    ap.add_argument("--min-train", type=int, default=24 * 180, help="barre minime prima del 1° trade")
    ap.add_argument("--seed", type=int, default=42)
    # FIX ⑤ (inert flag, default OFF): empirical cumulant drift correction from the z pool.
    ap.add_argument("--mart-correct", action="store_true",
                    help="correzione martingala empirica (cumulanti pool z); default OFF, numeri invariati")
    args = ap.parse_args()

    anchor = empirical_vrp_anchor()
    print("=== SHORT-VOL HISTORICAL BACKTEST (FHS GJR-GARCH · 2019→2026) ===")
    print(f"  n_paths={args.n_paths} refit_days={args.refit_days} tenor={TENOR_H}h | "
          f"VRP grid={['%d%%' % (v*100) for v in VRP_GRID]} | mart_correct={args.mart_correct}")
    if anchor:
        print(f"  àncora IV reale (12gg): mediana mark_iv ≈ {anchor['mean_mark_iv_pct']:.1f}%")
    print()

    configs = [("straddle", 0.0)] + [("strangle", w) for w in (0.04, 0.06, 0.08, 0.10)]
    report = {"meta": {"n_paths": args.n_paths, "refit_days": args.refit_days, "tenor_h": TENOR_H,
                       "vrp_grid": VRP_GRID, "iv_anchor": anchor,
                       "mart_correct": bool(args.mart_correct)}, "configs": []}

    for struct, w in configs:
        tr = run(w, struct, args.n_paths, args.refit_days, args.min_train, args.seed,
                 mart_correct=args.mart_correct)
        if tr.empty:
            print(f"  {struct} {w:.0%}: n=0 (skip)")
            continue
        table = vrp_table(tr, struct, w)
        be = breakeven_vrp(table)
        tag = f"strangle {w:.0%}" if struct == "strangle" else "straddle ATM"
        be_s = f"{be:.1%}" if be is not None else ">50% (mai)"
        row0 = table[0]                                  # VRP=0 row for the CI block
        print(f"  ── {tag} | n={row0['n']} scadenze | break-even VRP = {be_s} | "
              f"trades/yr≈{row0['trades_per_year']:.0f} N_eff≈{row0['n_eff']:.0f} (ρ₁={row0['rho1']:+.3f})")
        print(f"     {'VRP':>5} {'totPnL':>10} {'mean':>10} {'hit':>5} {'Sharpe':>7} {'p05':>9} {'worst5':>10}")
        for row in table:
            print(f"     {row['vrp']:>4.0%} {row['tot']:>+10.4f} {row['mean']:>+10.5f} "
                  f"{row['hit']:>4.0f}% {row['sharpe_ann']:>7.2f} {row['p05']:>+9.5f} {row['worst5_sum']:>+10.4f}")
        # FIX ①. Block-bootstrap CI (5–95%) at VRP=0 = the HONEST statistic on non-i.i.d. PnL:
        # a mean-CI spanning 0 ⇒ edge indistinguishable from noise once overlap is accounted for.
        print(f"     └─ block-boot CI(5-95%) @VRP=0: mean [{row0['mean_ci05']:+.5f}, {row0['mean_ci95']:+.5f}] "
              f"Sharpe [{row0['sharpe_ci05']:+.2f}, {row0['sharpe_ci95']:+.2f}]")
        report["configs"].append({"struct": struct, "width": w, "n": table[0]["n"],
                                  "breakeven_vrp": be, "table": table,
                                  "moved_pct_p99": float(np.percentile(tr["moved_pct"].abs(), 99)),
                                  "moved_pct_max": float(tr["moved_pct"].abs().max())})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n  → report in {OUT.relative_to(ROOT)}")
    print("  NB: VRP=0 ⇒ vendita a fair-value P (atteso ≈0 − fee − coda); il break-even VRP è la")
    print("      soglia implied/realized minima perché lo short-vol sopravviva alle code 2019-26.")


if __name__ == "__main__":
    main()
