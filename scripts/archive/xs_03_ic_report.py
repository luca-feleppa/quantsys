#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xs_03_ic_report.py — Cross-Sectional IC Report (GO/NO-GO for the cross-asset pivot).

Decisive go/no-go test for the cross-asset pivot. Given a long panel
`data/xs/mu_panel.parquet` with [open_time, symbol, mu_raw, sigma_raw, fwd_ret_30],
it measures whether the existing model's predicted μ has CROSS-SECTIONAL RANK
skill across assets (per-timestamp Spearman Information Coefficient). It
aggregates IC_t into mean/std/ICIR/t-stat, isolates OOS (last 30% by date),
splits OOS into K=5 NON-OVERLAPPING sub-periods (project's robust methodology:
the rolling window was inflated 30× by autocorrelation, see
ic_metric_fix_2026_06_02), builds a top-minus-bottom portfolio for tradability
sanity (gross + net @13bps/leg) and emits a pre-registered KILL / PROCEED verdict.

NOTE: read-only on the rest of the repo; only creates `results/xs/ic_report.json`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# force UTF-8 on stdout/stderr (Windows console = cp1252 → UnicodeEncodeError
# on chars like "→"). Consistent with existing UTF-8 fixes in the repo.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# pre-registered scientific parameters (must NOT be tuned ex-post).
N_MIN_SYMBOLS = 8          # min symbols per timestamp
OOS_FRACTION = 0.30        # last 30% by date = OOS
K_SUBPERIODS = 5           # non-overlapping sub-periods
COST_BPS_PER_LEG = 13.0    # 13 bps round-trip cost per leg
TSTAT_THRESHOLD = 2.0      # |t| threshold
SIGN_CONSISTENCY_OK = 4    # ≥4/5 sub-periods positive
TOPK = 5                   # top/bottom 5 if ≥10 symbols
# 30-candle grid on 1m → step every 30 min (~1440 1m candles/day); the Sharpe of the
# IC_t signal and the spreads are annualized with steps/year of the 30-candle cadence.
STEPS_PER_YEAR = 365.0 * 24.0 * 60.0 / 30.0  # ≈ 17520 steps/year (30-min cadence)

PANEL_PATH = Path("data/xs/mu_panel.parquet")
OUT_PATH = Path("results/xs/ic_report.json")


# ----------------------------------------------------------------------------
# statistical core — per-timestamp cross-sectional IC.
# ----------------------------------------------------------------------------
def cross_sectional_ic_series(panel: pd.DataFrame, n_min: int = N_MIN_SYMBOLS) -> pd.DataFrame:
    """
    For each open_time with ≥ n_min symbols (non-NaN fwd_ret_30) compute the
    cross-sectional Spearman between mu_raw and fwd_ret_30 → IC_t series.
    Returns DataFrame indexed/ordered by open_time with columns [open_time, ic, n_sym].
    """
    # drop rows without realized fwd return; skill is only measurable on observed outcomes.
    df = panel.dropna(subset=["fwd_ret_30", "mu_raw"]).copy()

    records = []
    for ts, grp in df.groupby("open_time", sort=True):
        n_sym = grp["symbol"].nunique()
        if n_sym < n_min:
            # cross-section too thin → unstable IC, skip.
            continue
        # Spearman degenerates if mu or fwd is constant across the section.
        mu = grp["mu_raw"].to_numpy()
        fwd = grp["fwd_ret_30"].to_numpy()
        if np.ptp(mu) == 0.0 or np.ptp(fwd) == 0.0:
            continue
        with np.errstate(invalid="ignore"):
            ic, _ = spearmanr(mu, fwd)
        if ic is None or (isinstance(ic, float) and math.isnan(ic)):
            continue
        records.append({"open_time": ts, "ic": float(ic), "n_sym": int(n_sym)})

    return pd.DataFrame.from_records(records, columns=["open_time", "ic", "n_sym"])


def aggregate_ic(ic_values: np.ndarray) -> dict:
    """
    Aggregate an IC_t series into mean/std/ICIR/t-stat. The t-stat uses sqrt(n)
    because the 30-candle cadence makes steps quasi-independent (minimal overlap
    by construction → no strong Newey-West correction needed).
    """
    ic_values = np.asarray(ic_values, dtype=float)
    n = int(ic_values.size)
    if n == 0:
        return {"n": 0, "mean_ic": 0.0, "std_ic": 0.0, "icir": 0.0, "t_stat": 0.0}
    mean_ic = float(np.mean(ic_values))
    # sample std (ddof=1); with n=1 std undefined → 0 and t-stat 0.
    std_ic = float(np.std(ic_values, ddof=1)) if n > 1 else 0.0
    icir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
    t_stat = float(icir * math.sqrt(n)) if std_ic > 0 else 0.0
    return {"n": n, "mean_ic": mean_ic, "std_ic": std_ic, "icir": icir, "t_stat": t_stat}


def subperiod_ic(ic_df: pd.DataFrame, k: int = K_SUBPERIODS) -> dict:
    """
    Split the time-ordered IC_t series into K NON-overlapping slices, report
    mean IC per slice and sign-consistency (#slices with mean>0). Project's
    robust methodology (NO rolling window).
    """
    ics = ic_df["ic"].to_numpy()
    n = ics.size
    if n < k:
        # too few timestamps for K slices → degraded sign-consistency.
        sub_means = [float(x) for x in ics] if n > 0 else []
        n_pos = int(np.sum(np.asarray(sub_means) > 0)) if sub_means else 0
        return {"k_effective": len(sub_means), "sub_ic": sub_means,
                "n_positive": n_pos}
    # contiguous split into K roughly-equal parts (np.array_split).
    chunks = np.array_split(ics, k)
    sub_means = [float(np.mean(c)) for c in chunks]
    n_pos = int(np.sum(np.asarray(sub_means) > 0))
    return {"k_effective": k, "sub_ic": sub_means, "n_positive": n_pos}


# ----------------------------------------------------------------------------
# tradability sanity — top-minus-bottom portfolio by mu_raw rank.
# ----------------------------------------------------------------------------
def top_bottom_spread(panel: pd.DataFrame, n_min: int = N_MIN_SYMBOLS,
                      topk: int = TOPK, cost_bps_per_leg: float = COST_BPS_PER_LEG) -> dict:
    """
    Per timestamp go long the top-quantile and short the bottom-quantile by
    mu_raw rank; measure the mean realized fwd_ret_30 spread (GROSS) and a NET
    version after a 13bps round-trip cost per leg. If ≥10 symbols use top/bottom
    `topk`, otherwise terciles.
    """
    df = panel.dropna(subset=["fwd_ret_30", "mu_raw"]).copy()
    per_ts_gross = []
    per_ts_net = []

    # per-leg cost in log-return space (bps→fraction). Long+short = 2 legs; the
    # contract says "round-trip cost of 13bps per leg" → 13bps charged once per
    # leg (long, short) as the full round-trip cost for that leg.
    cost_frac = cost_bps_per_leg / 1e4
    total_cost = 2.0 * cost_frac  # 2 legs

    for ts, grp in df.groupby("open_time", sort=True):
        n_sym = grp["symbol"].nunique()
        if n_sym < n_min:
            continue
        g = grp.sort_values("mu_raw")
        fwd = g["fwd_ret_30"].to_numpy()
        m = fwd.size
        if n_sym >= 10:
            q = min(topk, m // 2)  # avoid overlap if m is small and odd
        else:
            # terciles
            q = max(1, m // 3)
        if q < 1 or 2 * q > m:
            continue
        bottom = fwd[:q]   # lowest mu → short
        top = fwd[-q:]     # highest mu → long
        gross = float(np.mean(top) - np.mean(bottom))
        net = gross - total_cost
        per_ts_gross.append(gross)
        per_ts_net.append(net)

    if not per_ts_gross:
        return {"n": 0, "mean_gross": 0.0, "mean_net": 0.0,
                "ann_gross": 0.0, "ann_net": 0.0}

    g_arr = np.asarray(per_ts_gross)
    n_arr = np.asarray(per_ts_net)
    mean_gross = float(np.mean(g_arr))
    mean_net = float(np.mean(n_arr))
    # ~linear annualization over steps/year (log-returns are additive).
    return {
        "n": int(g_arr.size),
        "mean_gross": mean_gross,
        "mean_net": mean_net,
        "ann_gross": float(mean_gross * STEPS_PER_YEAR),
        "ann_net": float(mean_net * STEPS_PER_YEAR),
    }


def ic_sharpe(ic_values: np.ndarray) -> float:
    """
    (Annualized) Sharpe of the IC_t series treated as the signal's PnL:
    mean/std × sqrt(steps/year). Diagnoses the stability of the rank edge.
    """
    ic_values = np.asarray(ic_values, dtype=float)
    if ic_values.size < 2:
        return 0.0
    mu = np.mean(ic_values)
    sd = np.std(ic_values, ddof=1)
    if sd <= 0:
        return 0.0
    return float(mu / sd * math.sqrt(STEPS_PER_YEAR))


# ----------------------------------------------------------------------------
# orchestration + pre-registered verdict.
# ----------------------------------------------------------------------------
def split_oos(panel: pd.DataFrame, frac: float = OOS_FRACTION) -> pd.DataFrame:
    """
    Defines OOS as the last `frac` by DATE (over unique timestamps, not rows —
    so densely-sampled symbols don't shift the cutoff).
    """
    uniq = np.sort(panel["open_time"].unique())
    if uniq.size == 0:
        return panel.iloc[0:0]
    cut_idx = int(math.floor(uniq.size * (1.0 - frac)))
    cut_idx = min(max(cut_idx, 0), uniq.size - 1)
    cutoff = uniq[cut_idx]
    return panel[panel["open_time"] >= cutoff].copy()


def build_report(panel: pd.DataFrame) -> dict:
    """
    Compute all metrics (FULL + OOS), tradability spread and verdict.
    """
    n_symbols = int(panel["symbol"].nunique())

    # FULL
    ic_full_df = cross_sectional_ic_series(panel)
    agg_full = aggregate_ic(ic_full_df["ic"].to_numpy())

    # OOS (last 30% by date) — IC_t computed ONLY on OOS timestamps
    oos_panel = split_oos(panel, OOS_FRACTION)
    ic_oos_df = cross_sectional_ic_series(oos_panel)
    agg_oos = aggregate_ic(ic_oos_df["ic"].to_numpy())
    sub = subperiod_ic(ic_oos_df, K_SUBPERIODS)
    spread = top_bottom_spread(oos_panel)
    sharpe_oos = ic_sharpe(ic_oos_df["ic"].to_numpy())

    # ------------------------------------------------------------------
    # pre-registered VERDICT (decision rule fixed before seeing data).
    # ------------------------------------------------------------------
    t_ok = abs(agg_oos["t_stat"]) >= TSTAT_THRESHOLD and agg_oos["mean_ic"] > 0
    sign_ok = sub["n_positive"] >= SIGN_CONSISTENCY_OK
    spread_ok = spread["mean_net"] > 0

    # KILL if IC indistinguishable from 0 OR sign-inconsistent (≤2/5)
    # OR net spread ≤0. PROCEED only if ALL conditions hold.
    kill = (abs(agg_oos["t_stat"]) < TSTAT_THRESHOLD) \
        or (sub["n_positive"] <= 2) \
        or (spread["mean_net"] <= 0)
    proceed = t_ok and sign_ok and spread_ok

    if proceed:
        verdict = "PROCEED"
    elif kill:
        verdict = "KILL"
    else:
        # grey zone — fails PROCEED but no KILL trigger fires (e.g. t≥2 and
        # spread>0 but only 3/5 positive). Conservative default: KILL.
        verdict = "KILL"

    report = {
        "n_symbols": n_symbols,
        "n_min_symbols": N_MIN_SYMBOLS,
        "oos_fraction": OOS_FRACTION,
        "cost_bps_per_leg": COST_BPS_PER_LEG,
        "full": {
            "n_timestamps": agg_full["n"],
            "mean_ic": agg_full["mean_ic"],
            "std_ic": agg_full["std_ic"],
            "icir": agg_full["icir"],
            "t_stat": agg_full["t_stat"],
        },
        "oos": {
            "n_timestamps": agg_oos["n"],
            "mean_ic": agg_oos["mean_ic"],
            "std_ic": agg_oos["std_ic"],
            "icir": agg_oos["icir"],
            "t_stat": agg_oos["t_stat"],
            "ic_sharpe_ann": sharpe_oos,
        },
        "oos_subperiods": {
            "k": K_SUBPERIODS,
            "k_effective": sub["k_effective"],
            "sub_ic": sub["sub_ic"],
            "n_positive": sub["n_positive"],
        },
        "tradability": {
            "n_timestamps": spread["n"],
            "mean_gross_spread": spread["mean_gross"],
            "mean_net_spread": spread["mean_net"],
            "ann_gross_spread": spread["ann_gross"],
            "ann_net_spread": spread["ann_net"],
        },
        "decision_rule": {
            "tstat_threshold": TSTAT_THRESHOLD,
            "sign_consistency_proceed": SIGN_CONSISTENCY_OK,
            "t_ok": bool(t_ok),
            "sign_ok": bool(sign_ok),
            "spread_ok": bool(spread_ok),
        },
        "verdict": verdict,
    }
    return report


def print_verdict(report: dict) -> None:
    """Human-readable console verdict."""
    oos = report["oos"]
    sub = report["oos_subperiods"]
    trad = report["tradability"]
    full = report["full"]

    print("=" * 72)
    print("  CROSS-SECTIONAL IC REPORT — GO/NO-GO per il pivot cross-asset")
    print("=" * 72)
    print(f"  n_symbols                 : {report['n_symbols']}")
    print(f"  N_min symbols / timestamp : {report['n_min_symbols']}")
    print("-" * 72)
    print("  FULL PANEL")
    print(f"    n_timestamps : {full['n_timestamps']}")
    print(f"    mean IC      : {full['mean_ic']:+.5f}   ICIR: {full['icir']:+.4f}   "
          f"t-stat: {full['t_stat']:+.3f}")
    print("-" * 72)
    print(f"  OOS (last {report['oos_fraction']*100:.0f}% by date)")
    print(f"    n_timestamps : {oos['n_timestamps']}")
    print(f"    mean IC      : {oos['mean_ic']:+.5f}   ICIR: {oos['icir']:+.4f}   "
          f"t-stat: {oos['t_stat']:+.3f}")
    print(f"    IC Sharpe~ann: {oos['ic_sharpe_ann']:+.3f}")
    sub_str = ", ".join(f"{x:+.4f}" for x in sub["sub_ic"])
    print(f"    sub-period IC ({sub['k_effective']}): [{sub_str}]")
    print(f"    sign-consistency: {sub['n_positive']}/{sub['k_effective']} positive")
    print("-" * 72)
    print("  TRADABILITY (top-minus-bottom by mu_raw rank)")
    print(f"    n_timestamps : {trad['n_timestamps']}")
    print(f"    gross spread : {trad['mean_gross_spread']:+.6f} / step "
          f"(ann ~ {trad['ann_gross_spread']:+.4f})")
    print(f"    net   spread : {trad['mean_net_spread']:+.6f} / step "
          f"(ann ~ {trad['ann_net_spread']:+.4f})  [13bps/leg]")
    print("=" * 72)
    dr = report["decision_rule"]
    print(f"  decision gates → |t|>=2: {dr['t_ok']} | >=4/5 positive: {dr['sign_ok']} | "
          f"net>0: {dr['spread_ok']}")
    verdict = report["verdict"]
    banner = ">>> VERDICT: " + verdict + " <<<"
    print()
    if verdict == "PROCEED":
        print("  " + banner)
        print("  IT: μ ha skill di rango cross-sezionale OOS robusta → costruisci il")
        print("      portfolio layer (universe+survivorship + neutral constructor +")
        print("      panel fee/turnover backtester).")
        print("  EN: μ has robust OOS cross-sectional rank skill → build the portfolio")
        print("      layer (universe+survivorship + neutral constructor + backtester).")
    else:
        print("  " + banner)
        print("  IT: μ NON ha skill di rango cross-sezionale OOS robusta. STOP: non")
        print("      costruire il portfolio layer (IC indistinguibile da 0, oppure")
        print("      sign-inconsistente, oppure spread netto <= 0).")
        print("  EN: μ has NO robust OOS cross-sectional rank skill. STOP: do not build")
        print("      the portfolio layer (IC indistinguishable from 0, or sign-")
        print("      inconsistent, or net spread <= 0).")
    print("=" * 72)


def run_on_panel(panel_path: Path = PANEL_PATH, out_path: Path = OUT_PATH) -> int:
    """
    Real pipeline: load parquet, compute, write JSON, print verdict.
    Returns exit-code (0 ok, 2 panel missing).
    """
    if not panel_path.exists():
        print(f"[xs_03_ic_report] panel not found: {panel_path}")
        print("  IT: produci prima il panel (agente parallelo) → "
              "data/xs/mu_panel.parquet")
        print("  EN: produce the panel first (parallel agent) → "
              "data/xs/mu_panel.parquet")
        return 2

    panel = pd.read_parquet(panel_path)
    # minimal column contract.
    required = {"open_time", "symbol", "mu_raw", "sigma_raw", "fwd_ret_30"}
    missing = required - set(panel.columns)
    if missing:
        print(f"[xs_03_ic_report] panel missing columns: {sorted(missing)}")
        return 3
    # ensure sortable UTC datetime.
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)

    report = build_report(panel)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # atomic write (.tmp + replace), consistent with repo checkpointing.
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    tmp.replace(out_path)
    print(f"[xs_03_ic_report] wrote {out_path}")
    print_verdict(report)
    return 0


# ----------------------------------------------------------------------------
# synthetic VALIDATION — recover a known injected IC + null, check the verdict
# flips correctly. Run with `--self-test`.
# ----------------------------------------------------------------------------
def _make_synthetic_panel(n_symbols: int, n_timestamps: int, ic_strength: float,
                          seed: int = 0) -> pd.DataFrame:
    """
    Synthetic panel with injected cross-sectional IC. fwd = ic*mu_std + noise:
    at each timestamp we standardize mu within the section, fwd is a linear
    combination of mu+noise so the expected Spearman ≈ ic_strength.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-01-01", tz="UTC")
    rows = []
    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    for t in range(n_timestamps):
        ts = base + pd.Timedelta(minutes=30 * t)  # 30-candle grid
        mu = rng.normal(0.0, 1.0, n_symbols)
        # standardize mu within section to control correlation.
        mu_s = (mu - mu.mean()) / (mu.std() + 1e-12)
        noise = rng.normal(0.0, 1.0, n_symbols)
        # fwd with linear correlation ~ic_strength to mu (scaled log-ret space).
        fwd = (ic_strength * mu_s + math.sqrt(max(1e-9, 1 - ic_strength**2)) * noise) * 0.01
        for j, sym in enumerate(symbols):
            rows.append({
                "open_time": ts,
                "symbol": sym,
                "mu_raw": float(mu[j] * 0.005),
                "sigma_raw": 0.01,
                "fwd_ret_30": float(fwd[j]),
            })
    df = pd.DataFrame(rows)
    # inject realistic NaNs at series ends (last 30 steps per symbol).
    df.loc[df["open_time"] >= base + pd.Timedelta(minutes=30 * (n_timestamps - 1)),
           "fwd_ret_30"] = np.nan
    return df


def self_test() -> int:
    """
    Two scenarios — SIGNAL (injected IC ~0.20) and NULL (IC=0). Check the report
    recovers ~the injected IC and the verdict flips (PROCEED vs KILL).
    """
    print("\n########## SELF-TEST: SIGNAL scenario (injected IC ~0.20) ##########")
    sig = _make_synthetic_panel(n_symbols=15, n_timestamps=500, ic_strength=0.20, seed=1)
    rep_sig = build_report(sig)
    print_verdict(rep_sig)

    print("\n########## SELF-TEST: NULL scenario (injected IC = 0) ##########")
    nul = _make_synthetic_panel(n_symbols=15, n_timestamps=500, ic_strength=0.0, seed=2)
    rep_nul = build_report(nul)
    print_verdict(rep_nul)

    # recovery assertions.
    ok = True
    sig_ic = rep_sig["full"]["mean_ic"]
    nul_ic = rep_nul["full"]["mean_ic"]
    print("\n########## SELF-TEST SUMMARY ##########")
    print(f"  SIGNAL full mean IC : {sig_ic:+.4f}  (expected ~+0.20)")
    print(f"  NULL   full mean IC : {nul_ic:+.4f}  (expected ~0.00)")
    print(f"  SIGNAL verdict      : {rep_sig['verdict']}  (expected PROCEED)")
    print(f"  NULL   verdict      : {rep_nul['verdict']}  (expected KILL)")

    if not (0.12 <= sig_ic <= 0.28):
        print("  [FAIL] SIGNAL IC not recovered in [0.12, 0.28]")
        ok = False
    if not (abs(nul_ic) <= 0.05):
        print("  [FAIL] NULL IC not near 0 (|IC|<=0.05)")
        ok = False
    if rep_sig["verdict"] != "PROCEED":
        print("  [FAIL] SIGNAL verdict != PROCEED")
        ok = False
    if rep_nul["verdict"] != "KILL":
        print("  [FAIL] NULL verdict != KILL")
        ok = False

    print("  RESULT:", "PASS — logic recovers known IC and flips verdict correctly"
          if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return self_test()
    return run_on_panel()


if __name__ == "__main__":
    raise SystemExit(main())
