# IVS-SCOUT — READ-ONLY scouting of the Deribit BTC options IV surface (data/iv/chain/*).
# Goal: measure whether exploitable RELATIVE-VALUE structure exists (mispricing across
# strikes/expiries), NOT a bet on the vol LEVEL. Evidence-gathering, no trading.
# Four quantitative measures:
#   1. TERM STRUCTURE — ATM IV per tenor over time; is the front-vs-back slope persistent/
#      mean-reverting (calendar RV signal) or noise? Time-series stats + autocorrelation.
#   2. SKEW — within a ~30d tenor band, IV asymmetry OTM puts vs OTM calls at matched
#      moneyness (±10% and ~25-delta proxy). Is skew level/dynamics persistent?
#   3. SURFACE RESIDUALS — per snapshot fit a smooth surface IV ~ f(log(K/F), tenor)
#      (low-order polynomial), residual per liquid option (mark_iv - fit). Are residuals
#      PERSISTENT across consecutive snapshots (per-instrument lag-1 autocorr) = a
#      mean-reversion RV signal, vs i.i.d. noise? LIQUID options only (OI/volume thr, valid b/a).
#   4. EXECUTION REALITY CHECK — typical bid-ask half-spread (IV vol-points and price) on the
#      liquid set: an RV edge must exceed the round-trip spread to be real.
# Output: results/vols/ivs_scout.json + printed summary. READ-ONLY: does NOT touch the live
# processes (04b_vol_paper / 01c_iv_poller / 01d_orderbook), no GPU, no downloads.

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# project root = two levels up (scripts/vol/ -> repo). Run from the project root.
ROOT = Path(__file__).resolve().parents[2]

# scouting parameters (all documented, explicit thresholds).
SUBSAMPLE_PER_HOUR = 1        # 1 snapshot/hour for speed
OI_MIN = 50.0                 # open-interest liquidity threshold
MIN_TENOR_DAYS = 1.0 / 24     # drop expiries <1h (mark_iv blows up)
MAX_TENOR_DAYS = 120.0        # beyond = too illiquid
RESID_LOGM_MAX = 0.30         # moneyness band for residuals (outside, the smile is strongly convex and a low-order poly breaks -> RMSE from misspecification, NOT mispricing)
RESID_MIN_PER_SMILE = 6       # min options per per-expiry smile
SKEW_TENOR_LO = 20.0          # ~30d skew tenor band
SKEW_TENOR_HI = 45.0
MONEYNESS_OTM = 0.10          # ±10% for put/call skew
# tenor buckets (days) for term-structure slope: front vs back.
FRONT_TENOR = (2.0, 9.0)     # ~1 week
BACK_TENOR = (25.0, 45.0)    # ~1 month


def _autocorr_lag1(x):
    # lag-1 autocorrelation (NaN-safe); returns None if <3 valid points.
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return None
    a, b = x[:-1], x[1:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _atm_iv(grp, underlying):
    # ATM IV = mark_iv of the strike nearest the forward (proxy = underlying) for that tenor.
    if grp.empty:
        return np.nan
    idx = (grp["strike"] - underlying).abs().idxmin()
    return float(grp.loc[idx, "mark_iv"])


def load_snapshots():
    # load all parquet, subsample to 1 snapshot/hour (first snapshot of each UTC hour).
    # Adds tenor_days and log-moneyness log(K/F) with F≈underlying (forward proxy, inverse-coin).
    files = sorted(glob.glob(str(ROOT / "data" / "iv" / "chain" / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no chain parquet under {ROOT/'data'/'iv'/'chain'}")
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
        df["expiry"] = pd.to_datetime(df["expiry"], utc=True)
        # hourly key = first snapshot per hour
        df["hour_key"] = df["snapshot_ts"].dt.floor("h")
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    # per hour keep the minimum snapshot_ts (1/h subsample).
    keep = (
        full.groupby("hour_key")["snapshot_ts"].min().reset_index(name="snap_keep")
    )
    full = full.merge(keep, on="hour_key")
    full = full[full["snapshot_ts"] == full["snap_keep"]].copy()

    full["tenor_days"] = (
        (full["expiry"] - full["snapshot_ts"]).dt.total_seconds() / 86400.0
    )
    # log-moneyness; forward proxy = underlying_price (negligible carry over short crypto tenors).
    full["logm"] = np.log(full["strike"] / full["underlying_price"])
    full["iv"] = full["mark_iv"] / 100.0  # % -> fraction
    # IV half-spread not directly available; estimated in #4 from prices.
    full = full[(full["tenor_days"] >= MIN_TENOR_DAYS) & (full["tenor_days"] <= MAX_TENOR_DAYS)]
    full = full[full["mark_iv"] > 0]
    return full, len(keep), [Path(f).name for f in files]


def liquid_mask(df):
    # liquidity mask: OI above threshold, bid>0, ask>bid (valid spread).
    return (
        (df["open_interest"] >= OI_MIN)
        & (df["bid_price"] > 0)
        & (df["ask_price"] > df["bid_price"])
    )


def analyze_term_structure(df):
    # #1 TERM STRUCTURE. Per snapshot compute ATM IV in FRONT and BACK buckets,
    # slope = back - front. Time series -> stats + lag-1 autocorr (persistence).
    rows = []
    for ts, snap in df.groupby("snapshot_ts"):
        und = float(snap["underlying_price"].iloc[0])
        front = snap[(snap["tenor_days"] >= FRONT_TENOR[0]) & (snap["tenor_days"] <= FRONT_TENOR[1])]
        back = snap[(snap["tenor_days"] >= BACK_TENOR[0]) & (snap["tenor_days"] <= BACK_TENOR[1])]
        # ATM per bucket = nearest-forward strike, at the bucket's median tenor.
        atm_f = _atm_iv(front, und)
        atm_b = _atm_iv(back, und)
        if np.isnan(atm_f) or np.isnan(atm_b):
            continue
        rows.append({"ts": ts, "atm_front": atm_f, "atm_back": atm_b, "slope": atm_b - atm_f})
    tsdf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    slope = tsdf["slope"].values  # in vol-points (% IV)
    out = {
        "n_snapshots": int(len(tsdf)),
        "atm_front_mean_pct": float(np.nanmean(tsdf["atm_front"])) if len(tsdf) else None,
        "atm_back_mean_pct": float(np.nanmean(tsdf["atm_back"])) if len(tsdf) else None,
        "slope_mean_pct": float(np.nanmean(slope)) if len(tsdf) else None,
        "slope_std_pct": float(np.nanstd(slope)) if len(tsdf) else None,
        "slope_min_pct": float(np.nanmin(slope)) if len(tsdf) else None,
        "slope_max_pct": float(np.nanmax(slope)) if len(tsdf) else None,
        "slope_frac_inverted": float(np.mean(slope < 0)) if len(tsdf) else None,
        "slope_autocorr_lag1": _autocorr_lag1(slope),
        "slope_sign_flips_per_snap": float(np.mean(np.abs(np.diff(np.sign(slope))) > 0)) if len(slope) > 1 else None,
    }
    return out, tsdf


def analyze_skew(df):
    # #2 SKEW. ~30d tenor band. Per snapshot take put at logm≈-10% and call at logm≈+10%
    # (nearest in moneyness), skew_pct = IV_put_OTM - IV_call_OTM (>0 = put-rich risk reversal).
    # Time series -> mean level + autocorr (dynamics persistence).
    band = df[(df["tenor_days"] >= SKEW_TENOR_LO) & (df["tenor_days"] <= SKEW_TENOR_HI)]
    rows = []
    for ts, snap in band.groupby("snapshot_ts"):
        puts = snap[snap["option_type"] == "P"]
        calls = snap[snap["option_type"] == "C"]
        if puts.empty or calls.empty:
            continue
        # OTM put ~ -10% moneyness (strike below forward)
        p_otm = puts.iloc[(puts["logm"] - (-MONEYNESS_OTM)).abs().argsort()[:1]]
        c_otm = calls.iloc[(calls["logm"] - (MONEYNESS_OTM)).abs().argsort()[:1]]
        # drop if nearest is too far from target moneyness (sparse chain).
        if abs(p_otm["logm"].iloc[0] + MONEYNESS_OTM) > 0.06 or abs(c_otm["logm"].iloc[0] - MONEYNESS_OTM) > 0.06:
            continue
        skew = float(p_otm["mark_iv"].iloc[0] - c_otm["mark_iv"].iloc[0])
        rows.append({"ts": ts, "iv_put_otm": float(p_otm["mark_iv"].iloc[0]),
                     "iv_call_otm": float(c_otm["mark_iv"].iloc[0]), "skew": skew})
    skdf = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    sk = skdf["skew"].values if len(skdf) else np.array([])
    out = {
        "tenor_band_days": [SKEW_TENOR_LO, SKEW_TENOR_HI],
        "moneyness_target": MONEYNESS_OTM,
        "n_snapshots": int(len(skdf)),
        "skew_mean_pct": float(np.nanmean(sk)) if len(sk) else None,
        "skew_std_pct": float(np.nanstd(sk)) if len(sk) else None,
        "skew_min_pct": float(np.nanmin(sk)) if len(sk) else None,
        "skew_max_pct": float(np.nanmax(sk)) if len(sk) else None,
        "skew_frac_put_rich": float(np.mean(sk > 0)) if len(sk) else None,
        "skew_autocorr_lag1": _autocorr_lag1(sk),
    }
    return out, skdf


def _fit_surface_residuals(snap):
    # per-expiry SMILE fit (standard RV decomposition) for ONE snapshot, on LIQUID options
    # within the RESID_LOGM_MAX moneyness band:
    #   iv ~ b0 + b1*logm + b2*logm^2   (quadratic per expiry, fit separately).
    # Per-expiry avoids the crude cross-tenor √T coupling (which injects misspecification
    # RMSE, NOT mispricing). The residual (mark_iv - smile_fit) is the genuine RV candidate.
    # Aggregates residuals of all the snapshot's smiles.
    liq = snap[liquid_mask(snap)].copy()
    liq = liq[liq["logm"].abs() <= RESID_LOGM_MAX]
    if liq.empty:
        return None
    parts = []
    sq_err = []
    for _exp, g in liq.groupby("expiry"):
        if len(g) < RESID_MIN_PER_SMILE:  # too sparse a smile
            continue
        m = g["logm"].values
        y = g["iv"].values  # as fraction
        X = np.column_stack([np.ones_like(m), m, m ** 2])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        r = (y - X @ beta) * 100.0  # residual in vol-points
        gg = g[["snapshot_ts", "instrument_name"]].copy()
        gg["resid_iv"] = r
        parts.append(gg)
        sq_err.append(r ** 2)
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res["fit_rmse_pct"] = float(np.sqrt(np.mean(np.concatenate(sq_err))))
    return res[["snapshot_ts", "instrument_name", "resid_iv", "fit_rmse_pct"]]


def analyze_surface_residuals(df):
    # #3 SURFACE RESIDUALS. Per-snapshot fit, gather per-instrument residuals, then measure
    # lag-1 autocorr of each instrument's residuals across CONSECUTIVE (hourly) snapshots.
    # Distribution of per-instrument autocorrs -> strong median>0 = exploitable mean-reversion;
    # ≈0 = noise.
    parts = []
    rmses = []
    for ts, snap in df.groupby("snapshot_ts"):
        r = _fit_surface_residuals(snap)
        if r is not None and len(r):
            parts.append(r)
            rmses.append(float(r["fit_rmse_pct"].iloc[0]))
    if not parts:
        return {"error": "no liquid snapshots fittable"}, None
    res = pd.concat(parts, ignore_index=True)

    # pivot instrument×time, per-instrument lag-1 autocorr with ≥6 observations.
    autocorrs = []
    autocorrs_dm = []  # on per-instrument DEMEANED residual
    resid_std = []
    for inst, g in res.sort_values("snapshot_ts").groupby("instrument_name"):
        series = g["resid_iv"].values
        if len(series) >= 6:
            ac = _autocorr_lag1(series)
            if ac is not None:
                autocorrs.append(ac)
                resid_std.append(float(np.std(series)))
                # RAW autocorr is inflated by a STATIC per-strike offset (fixed basis, NOT a
                # tradeable mean-reverting signal). Removing the per-instrument mean isolates
                # the DYNAMICS: demeaned autocorr ~0 => persistence is a static basis, NOT an
                # RV signal.
                ac_dm = _autocorr_lag1(series - series.mean())
                if ac_dm is not None:
                    autocorrs_dm.append(ac_dm)
    autocorrs = np.array(autocorrs)
    autocorrs_dm = np.array(autocorrs_dm)
    all_resid = res["resid_iv"].values
    out = {
        "n_snapshots_fitted": int(len(parts)),
        "n_liquid_resid_obs": int(len(res)),
        "n_instruments_with_series": int(len(autocorrs)),
        "fit_rmse_pct_median": float(np.median(rmses)),
        "resid_abs_median_pct": float(np.median(np.abs(all_resid))),
        "resid_abs_p90_pct": float(np.percentile(np.abs(all_resid), 90)),
        "resid_std_pct": float(np.std(all_resid)),
        "resid_autocorr_lag1_median": float(np.median(autocorrs)) if len(autocorrs) else None,
        "resid_autocorr_lag1_mean": float(np.mean(autocorrs)) if len(autocorrs) else None,
        "resid_autocorr_lag1_demeaned_median": float(np.median(autocorrs_dm)) if len(autocorrs_dm) else None,
        "resid_autocorr_lag1_demeaned_mean": float(np.mean(autocorrs_dm)) if len(autocorrs_dm) else None,
        "resid_autocorr_lag1_p25": float(np.percentile(autocorrs, 25)) if len(autocorrs) else None,
        "resid_autocorr_lag1_p75": float(np.percentile(autocorrs, 75)) if len(autocorrs) else None,
        "resid_autocorr_frac_positive": float(np.mean(autocorrs > 0)) if len(autocorrs) else None,
        "per_instrument_resid_std_median_pct": float(np.median(resid_std)) if resid_std else None,
    }
    return out, res


def analyze_execution(df):
    # #4 EXECUTION REALITY CHECK. On LIQUID options: price half-spread (BTC) as (ask-bid)/2 and
    # relative to mid, and IV-vol-point half-spread estimated via a local numeric vega:
    # dIV ≈ half_spread_price / vega, with vega ≈ d(mark_price)/d(IV); proxied by
    # (half_spread_price/mid)*mark_iv (for ATM, ∂price/∂σ·σ ≈ price·elasticity~1).
    # Crude proxy but enough for order-of-magnitude vs #3 residuals.
    liq = df[liquid_mask(df)].copy()
    if liq.empty:
        return {"error": "no liquid options"}
    liq["mid"] = (liq["ask_price"] + liq["bid_price"]) / 2.0
    liq["half_spread_price"] = (liq["ask_price"] - liq["bid_price"]) / 2.0
    liq = liq[liq["mid"] > 0]
    liq["rel_half_spread"] = liq["half_spread_price"] / liq["mid"]
    # IV-vol-point half-spread: (rel_half_spread) * mark_iv (% IV). Elasticity~1 proxy.
    liq["half_spread_ivpts"] = liq["rel_half_spread"] * liq["mark_iv"]
    # restrict to near-ATM, where elasticity~1 is least wrong
    atm = liq[liq["logm"].abs() <= 0.05]
    out = {
        "n_liquid": int(len(liq)),
        "n_liquid_atm": int(len(atm)),
        "half_spread_price_btc_median": float(liq["half_spread_price"].median()),
        "rel_half_spread_median": float(liq["rel_half_spread"].median()),
        "rel_half_spread_p25": float(liq["rel_half_spread"].quantile(0.25)),
        "rel_half_spread_p75": float(liq["rel_half_spread"].quantile(0.75)),
        "half_spread_ivpts_median_all": float(liq["half_spread_ivpts"].median()),
        "half_spread_ivpts_median_atm": float(atm["half_spread_ivpts"].median()) if len(atm) else None,
        "half_spread_ivpts_p75_atm": float(atm["half_spread_ivpts"].quantile(0.75)) if len(atm) else None,
    }
    return out


def build_verdict(term, skew, surf, execu):
    # honest verdict. A real RV edge requires: (a) persistent structure (slope/skew/residual
    # autocorr substantial, >~0.3) AND (b) residual magnitude > round-trip half-spread (2× half).
    notes = []
    # key compare: typical residual (#3 abs p90) vs round-trip cost in IV vol-points (2× ATM half).
    resid_p90 = surf.get("resid_abs_p90_pct")
    half_atm = execu.get("half_spread_ivpts_median_atm")
    roundtrip = (2.0 * half_atm) if half_atm is not None else None
    edge_vs_cost = None
    if resid_p90 is not None and roundtrip:
        edge_vs_cost = resid_p90 / roundtrip
        notes.append(
            f"resid abs p90 = {resid_p90:.2f} IV-pts vs round-trip cost ~{roundtrip:.2f} IV-pts "
            f"(ratio {edge_vs_cost:.2f}x)"
        )
    rac = surf.get("resid_autocorr_lag1_median")
    if rac is not None:
        notes.append(f"residual lag-1 autocorr median = {rac:.2f} (raw, includes static per-strike basis)")
    # the HONEST mean-reversion test is the DEMEANED autocorr (removes the static basis).
    rac_dm = surf.get("resid_autocorr_lag1_demeaned_median")
    if rac_dm is not None:
        notes.append(f"residual lag-1 autocorr median DEMEANED = {rac_dm:.2f} "
                     f"({'genuine mean-reverting dynamics' if rac_dm > 0.3 else 'near-zero -> persistence is a STATIC basis, not a tradeable RV signal'})")
    sac = term.get("slope_autocorr_lag1")
    if sac is not None:
        notes.append(f"term-slope lag-1 autocorr = {sac:.2f}")
    skac = skew.get("skew_autocorr_lag1")
    if skac is not None:
        notes.append(f"skew lag-1 autocorr = {skac:.2f}")

    # synthetic criterion — the persistence THAT MATTERS is the DEMEANED one
    # (reverting dynamics, not the static per-strike basis).
    persistent = (rac_dm is not None and rac_dm > 0.3)
    beats_cost = (edge_vs_cost is not None and edge_vs_cost > 1.0)
    if persistent and beats_cost:
        verdict = "EVIDENCE of exploitable relative-value structure (persistent residuals AND magnitude > round-trip spread)"
    elif persistent and not beats_cost:
        verdict = "STRUCTURE present but UNEXPLOITABLE: residuals persist but do not exceed the bid-ask round-trip cost"
    elif (not persistent) and beats_cost:
        verdict = "Large residuals but NOT persistent (i.i.d.-like): looks like quote noise, not a tradeable signal"
    else:
        verdict = "EFFICIENT / NOISE: residuals neither persistent nor larger than the spread"
    return {"verdict": verdict, "edge_vs_roundtrip_cost_ratio": edge_vs_cost, "notes": notes,
            "CAVEAT": "Testnet/recorded data cannot validate live execution (fills, latency, "
                      "depth at quote) of these strategies; this is a structural scout only."}


def main():
    # UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("[ivs_scout] loading chain snapshots (READ-ONLY)...")
    df, n_hours, files = load_snapshots()
    print(f"[ivs_scout] files={len(files)} hourly_snapshots={n_hours} rows={len(df)}")

    term, _ = analyze_term_structure(df)
    skew, _ = analyze_skew(df)
    surf, _ = analyze_surface_residuals(df)
    execu = analyze_execution(df)
    verdict = build_verdict(term, skew, surf, execu)

    report = {
        "meta": {
            "n_files": len(files), "files": files, "n_hourly_snapshots": n_hours,
            "n_rows_subsampled": int(len(df)),
            "params": {
                "subsample_per_hour": SUBSAMPLE_PER_HOUR, "oi_min": OI_MIN,
                "front_tenor_days": FRONT_TENOR, "back_tenor_days": BACK_TENOR,
                "skew_tenor_band": [SKEW_TENOR_LO, SKEW_TENOR_HI], "moneyness_otm": MONEYNESS_OTM,
            },
        },
        "1_term_structure": term,
        "2_skew": skew,
        "3_surface_residuals": surf,
        "4_execution": execu,
        "verdict": verdict,
    }

    out_path = ROOT / "results" / "vols" / "ivs_scout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    # concise summary
    print("\n===== IVS-SCOUT SUMMARY =====")
    print(f"#1 TERM STRUCTURE  slope(back-front) mean={term.get('slope_mean_pct'):.2f} "
          f"std={term.get('slope_std_pct'):.2f} IV-pts | autocorr={term.get('slope_autocorr_lag1')} | "
          f"frac inverted={term.get('slope_frac_inverted')}")
    print(f"#2 SKEW (~30d)     put-call mean={skew.get('skew_mean_pct'):.2f} "
          f"std={skew.get('skew_std_pct'):.2f} IV-pts | autocorr={skew.get('skew_autocorr_lag1')} | "
          f"frac put-rich={skew.get('skew_frac_put_rich')}")
    print(f"#3 SURF RESIDUALS  fit RMSE med={surf.get('fit_rmse_pct_median'):.2f} | "
          f"|resid| med={surf.get('resid_abs_median_pct'):.2f} p90={surf.get('resid_abs_p90_pct'):.2f} IV-pts | "
          f"lag1 autocorr med raw={surf.get('resid_autocorr_lag1_median'):.2f} "
          f"demeaned={surf.get('resid_autocorr_lag1_demeaned_median'):.2f}")
    print(f"#4 EXECUTION       rel half-spread med={execu.get('rel_half_spread_median'):.3f} | "
          f"IV half-spread ATM med={execu.get('half_spread_ivpts_median_atm')} IV-pts "
          f"(round-trip ~{2*execu.get('half_spread_ivpts_median_atm'):.2f} IV-pts)")
    print(f"\nVERDICT: {verdict['verdict']}")
    for n in verdict["notes"]:
        print(f"  - {n}")
    print(f"  CAVEAT: {verdict['CAVEAT']}")
    print(f"\n[ivs_scout] report -> {out_path}")


if __name__ == "__main__":
    main()
