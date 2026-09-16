#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MFIV-COMPARATOR v2 JUDGE (pre-reg STATUS 2026-07-20) — PAIRED comparison of
04b's edge comparator: live 30h-interpolated ATM IV vs model-free MFIV@30h
(the var-swap rate). Per qualifying daily expiry it rebuilds the theoretical
entry (first hourly tick in the [E-30h, E-22h] tenor window with fresh MFIV
<=30' AND a forecast <=60'), the ATM short-straddle PnL under 04b's EXACT
conventions, and evaluates the gate:
  (1) delta_rho = rho_MFIV - rho_ATM >= +0.05, rho_x = Spearman(-edge_x, PnL)
  (2) same sign of delta_rho on both chronological halves
  (3) n >= 40 qualifying expiries — BELOW 40 NO RUN (fail-fast: the
      metrics are never even computed, peeking is forbidden)
OFFLINE, CPU-only: zero writes into 04b's live path (JSON report only).

Usage (from the project root):
    python scripts/vol/mfiv_comparator_judge.py            # real run (only at n>=40)
    python scripts/vol/mfiv_comparator_judge.py --smoke    # synthetic smoke, zero real data
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# project root (scripts/vol/ -> two levels up) — scripts/vol convention.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantsys.utils import setup_logging  # noqa: E402
from quantsys.data.deribit import delivery_price_cached as _delivery_cached  # noqa: E402

# import 04b as a module (digit-leading name -> importlib): the pre-registered
# constants (tenor, fees, conventions) are READ, never duplicated.
_spec = importlib.util.spec_from_file_location(
    "vol_paper_04b", ROOT / "scripts" / "04b_vol_paper.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

MFIV_PATH = ROOT / "data" / "iv" / "mfiv_30h.parquet"
FORECASTS_PATH = ROOT / "results" / "vol_paper" / "forecasts.parquet"
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
DELIVERY_CACHE = ROOT / "results" / "vol_paper" / "delivery_cache.json"
REPORT_PATH = ROOT / "results" / "vols" / "mfiv_comparator_report.json"

# PRE-REGISTERED gate constants (STATUS 2026-07-20) — do not touch after seeing numbers.
N_MIN = 40
DELTA_RHO_MIN = 0.05
MFIV_MAX_AGE_MIN = 30.0
FC_MAX_AGE_MIN = 60.0
TENOR_HI_H = 30       # far edge of the entry window
TENOR_LO_H = 22       # near edge
CHAIN_MAX_AGE_MIN = 30.0


def _asof(series_ts: pd.Series, t: pd.Timestamp, max_age_min: float):
    # index of the latest tick <= t aged <= max_age (causal as-of rule, same
    # semantics as the live read_iv). None when absent/stale.
    sub = series_ts[series_ts <= t]
    if sub.empty:
        return None
    idx = sub.index[-1]
    if (t - series_ts.loc[idx]).total_seconds() / 60.0 > max_age_min:
        return None
    return idx


def _straddle_at(t: pd.Timestamp, expiry: pd.Timestamp):
    # ATM straddle for the FIXED expiry from the latest chain snapshot <= t
    # (age <= 30'): strike on the median underlying, premia = call+put mark
    # in BTC/contract (same selection as pick_straddle / vol_paper_replay).
    day_files = [CHAIN_DIR / f"btc_options_{d.strftime('%Y%m%d')}.parquet"
                 for d in (t.normalize(), (t - pd.Timedelta(days=1)).normalize())]
    frames = [pd.read_parquet(p) for p in day_files if p.exists()]
    if not frames:
        return None
    ch = pd.concat(frames, ignore_index=True)
    ch["snapshot_ts"] = pd.to_datetime(ch["snapshot_ts"], utc=True)
    ch = ch[ch["snapshot_ts"] <= t]
    if ch.empty:
        return None
    snap_ts = ch["snapshot_ts"].max()
    if (t - snap_ts).total_seconds() / 60.0 > CHAIN_MAX_AGE_MIN:
        return None
    snap = ch[ch["snapshot_ts"] == snap_ts].copy()
    snap["expiry"] = pd.to_datetime(snap["expiry"], utc=True)
    sub = snap[snap["expiry"] == expiry]
    if sub.empty:
        return None
    und = float(sub["underlying_price"].median())
    k = min(sorted(sub["strike"].unique()), key=lambda s: abs(s - und))
    ot = sub["option_type"].astype(str).str.upper().str[0]
    call = sub[(sub["strike"] == k) & (ot == "C")]
    put = sub[(sub["strike"] == k) & (ot == "P")]
    if call.empty or put.empty:
        return None
    prem = float(call["mark_price"].iloc[0]) + float(put["mark_price"].iloc[0])
    if not np.isfinite(prem):
        return None
    return {"strike": float(k), "premium": prem}


def build_sample(count_only: bool = False) -> pd.DataFrame:
    # builds the paired per-expiry sample under the pre-registered deterministic
    # rule. Returns one row per qualifying expiry. count_only=True: tick
    # qualification ONLY (timestamps) — no chain, no delivery, no PnL/edge:
    # safe progress monitoring toward n_min.
    mfiv = pd.read_parquet(MFIV_PATH)
    mfiv["timestamp"] = pd.to_datetime(mfiv["timestamp"], utc=True)
    mts = mfiv["timestamp"]
    fc = pd.read_parquet(FORECASTS_PATH)
    fc["candle_ts"] = pd.to_datetime(fc["candle_ts"], utc=True)
    fts = fc["candle_ts"]

    start = mts.min().normalize()
    end = mts.max().normalize()
    expiries = (pd.date_range(start + pd.Timedelta(days=1), end + pd.Timedelta(days=1),
                              freq="D", tz="UTC") + pd.Timedelta(hours=8))
    expiries = expiries[(expiries - pd.Timedelta(hours=TENOR_HI_H)) >= mts.min()]

    rows = []
    for e in expiries:
        for k in range(TENOR_HI_H - TENOR_LO_H + 1):
            t = e - pd.Timedelta(hours=TENOR_HI_H - k)
            im = _asof(mts, t, MFIV_MAX_AGE_MIN)
            jf = _asof(fts, t, FC_MAX_AGE_MIN)
            if im is None or jf is None:
                continue
            rv_pred = float(fc.loc[jf, "rv_pred"])
            var_atm = float(fc.loc[jf, "var_iv"])
            if not (np.isfinite(rv_pred) and np.isfinite(var_atm) and
                    rv_pred > 0 and var_atm > 0):
                continue
            # same annualization convention as live (fixed 30h tenor).
            mf = float(mfiv.loc[im, "mfiv_30h"])
            var_mfiv = (mf / 100.0) ** 2 * (M.TENOR_HOURS / M.HOURS_PER_YEAR)
            if not (np.isfinite(var_mfiv) and var_mfiv > 0):
                continue
            if count_only:
                rows.append({"expiry": e, "entry_ts": t})
                break
            std = _straddle_at(t, e)
            if std is None:
                continue
            s_t = _delivery_cached(e, DELIVERY_CACHE)
            if s_t is None:
                break  # unsettled -> excluded
            # 1-contract short PnL, 04b conventions: mark premia - inverse payoff
            # |S_T - K|/S_T - taker fee x 2 legs (BTC/contract).
            payoff = abs(s_t - std["strike"]) / s_t
            pnl_short = (std["premium"] - payoff) * M.SIZE_CONTRACTS \
                - 2 * M.FEE_PER_CONTRACT
            rows.append({
                "expiry": e, "entry_ts": t,
                "edge_atm": float(np.log(rv_pred / var_atm)),
                "edge_mfiv": float(np.log(rv_pred / var_mfiv)),
                "log_wedge": float(np.log(var_mfiv / var_atm)),
                "pnl_short": float(pnl_short),
            })
            break  # first valid tick -> entry fixed
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, n_min: int = N_MIN) -> dict:
    # evaluates the pre-registered gate. BINDING ORDER: (3) is checked BEFORE any
    # correlation is computed (below n_min peeking is structurally impossible).
    n = len(df)
    if n < n_min:
        return {"n": n, "n_min": n_min, "verdict": "NO_RUN",
                "note": "campione insufficiente: metriche NON calcolate / "
                        "insufficient sample: metrics NOT computed"}
    from scipy.stats import spearmanr
    df = df.sort_values("entry_ts").reset_index(drop=True)

    def rho(sub: pd.DataFrame) -> tuple[float, float]:
        r_a = float(spearmanr(-sub["edge_atm"], sub["pnl_short"]).statistic)
        r_m = float(spearmanr(-sub["edge_mfiv"], sub["pnl_short"]).statistic)
        return r_a, r_m

    r_atm, r_mfiv = rho(df)
    d_rho = r_mfiv - r_atm
    half = n // 2
    d1 = rho(df.iloc[:half])
    d2 = rho(df.iloc[half:])
    d_rho_1, d_rho_2 = d1[1] - d1[0], d2[1] - d2[0]
    cond1 = d_rho >= DELTA_RHO_MIN
    cond2 = np.sign(d_rho_1) == np.sign(d_rho_2)
    return {
        "n": n, "n_min": n_min,
        "rho_atm": r_atm, "rho_mfiv": r_mfiv, "delta_rho": d_rho,
        "delta_rho_half1": d_rho_1, "delta_rho_half2": d_rho_2,
        "cond1_delta_rho_ge_0.05": bool(cond1),
        "cond2_sign_consistent": bool(cond2),
        "verdict": "PASS" if (cond1 and cond2) else "FAIL",
        # NON-gating descriptive: log-variance wedge -> break-even re-estimate.
        "descr_log_wedge_median": float(df["log_wedge"].median()),
        "descr_log_wedge_p10": float(df["log_wedge"].quantile(0.10)),
        "descr_log_wedge_p90": float(df["log_wedge"].quantile(0.90)),
    }


def _write_report(payload: dict):
    # atomic write (.tmp + os.replace — repo safety net).
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, REPORT_PATH)


def _smoke() -> int:
    # SYNTHETIC smoke (no real data, no network): checks (a) rank invariance —
    # CONSTANT log-var wedge -> exactly delta_rho == 0; (b) PASS path on a built
    # sample where wedge variation is informative; (c) n<n_min guard -> NO_RUN
    # with no metrics.
    rng = np.random.default_rng(42)
    n = 60
    ts = pd.date_range("2026-06-13 02:00", periods=n, freq="D", tz="UTC")

    # (a) constant wedge -> same rankings -> delta_rho = 0
    edge = rng.normal(0, 1, n)
    pnl = -0.3 * edge + rng.normal(0, 0.5, n)
    df_a = pd.DataFrame({"expiry": ts, "entry_ts": ts, "edge_atm": edge,
                         "edge_mfiv": edge - 0.25, "log_wedge": 0.25,
                         "pnl_short": pnl})
    res_a = evaluate(df_a)
    assert abs(res_a["delta_rho"]) < 1e-12, f"rank invariance violata: {res_a['delta_rho']}"

    # (b) informative variable wedge: edge_mfiv = true signal + little noise,
    #     edge_atm = signal + lots of noise -> delta_rho > 0 and stable sign
    signal = rng.normal(0, 1, n)
    pnl_b = -0.8 * signal + rng.normal(0, 0.3, n)
    df_b = pd.DataFrame({"expiry": ts, "entry_ts": ts,
                         "edge_atm": signal + rng.normal(0, 2.0, n),
                         "edge_mfiv": signal + rng.normal(0, 0.1, n),
                         "log_wedge": rng.normal(0.25, 0.05, n),
                         "pnl_short": pnl_b})
    res_b = evaluate(df_b)
    assert res_b["verdict"] == "PASS", f"path PASS non raggiunto: {res_b}"

    # (c) n_min guard: no metrics below threshold
    res_c = evaluate(df_b.iloc[:10])
    assert res_c["verdict"] == "NO_RUN" and "delta_rho" not in res_c, res_c

    print("SMOKE OK — (a) rank-invariance delta_rho=0, (b) path PASS, (c) guard NO_RUN")
    return 0


def main() -> int:
    # UTF-8 boilerplate (recurring cp1252 bug on Windows consoles).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Giudice MFIV-comparatore v2 / judge")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke sintetico, zero dati reali / synthetic smoke only")
    ap.add_argument("--count-only", action="store_true",
                    help="solo conteggio qualificati (timestamp), zero numeri "
                         "decisionali / qualification count only, no decision numbers")
    args = ap.parse_args()
    setup_logging()
    log = logging.getLogger("quantsys.script.mfiv_judge")

    if args.smoke:
        return _smoke()

    if args.count_only:
        n = len(build_sample(count_only=True))
        print(f"expiry qualificati/qualifying (tick-rule): {n}  (n_min={N_MIN}; "
              f"run one-shot alla prima sessione con n>={N_MIN})")
        return 0

    df = build_sample()
    n = len(df)
    log.info(f"expiry qualificati/qualifying: {n} (n_min pre-registrato: {N_MIN})")
    res = evaluate(df)
    if res["verdict"] == "NO_RUN":
        # below n_min: count ONLY, no report written (no peeking).
        print(f"NESSUN RUN / NO RUN — n={n} < {N_MIN}: il gate si valuta alla prima "
              f"sessione con n>={N_MIN} (pre-reg STATUS 2026-07-20)")
        return 2
    payload = {"prereg": "MFIV-comparatore v2 (STATUS 2026-07-20)",
               "generated_utc": str(pd.Timestamp.utcnow()), **res}
    _write_report(payload)
    print(json.dumps(payload, indent=2, default=str))
    print(f"VERDETTO: {res['verdict']}  → {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
