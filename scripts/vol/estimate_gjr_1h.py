"""
scripts/vol/estimate_gjr_1h.py
==============================
Re-estimation of the Monte Carlo GJR-GARCH(1,1) parameters on 1h returns
(TODO closed 2026-07-15 — 1h parameters now in config/default.yaml → montecarlo,
the 1m-era ones in config/interval/1m.yaml; ω has units [variance/step] and is
NOT transferable across timeframes).

Method: Gaussian QMLE with variance targeting — reuses `fit_gjr` from
`scripts/vol/short_vol_hist_backtest.py` (single source of truth for the fitter,
already validated by the FHS backtest). FULL-SAMPLE estimate: the parameters feed
a forward simulator (monte_carlo_forecast), not a strategy judged OOS — same
status as the 1m parameters they replace.

Output: printout + `results/vols/gjr_params_1h.json` (parameters, persistence,
half-life, unconditional σ, conditional-σ percentiles, suggested MC σ cap —
the 1m-era clip of 0.01/bar saturates at 1h).

Run from the project root:
  python scripts/vol/estimate_gjr_1h.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# project root (scripts/vol/ → 2 levels up) + shared fitter import.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantsys.utils import load_config, setup_logging, interval_minutes_from_cfg  # noqa: E402
from short_vol_hist_backtest import fit_gjr, gjr_recursion                        # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.estimate_gjr_1h")

CANDLES = ROOT / "data" / "raw_candles.parquet"
OUT     = ROOT / "results" / "vols" / "gjr_params_1h.json"


def main():
    # UTF-8 reconfigure (repo checklist: the cp1252 bug recurred 5 times).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config("config/default.yaml")
    interval_min = interval_minutes_from_cfg(cfg)
    # the estimate is timeframe-specific: fail fast if config is not 1h.
    if interval_min != 60:
        raise RuntimeError(
            f"data.interval={cfg['data']['interval']} (≠1h): questa stima è per "
            f"rendimenti 1h — a timeframe diverso ω NON è trasferibile / this "
            f"estimate targets 1h returns — ω does not transfer across timeframes"
        )

    df = pd.read_parquet(CANDLES)
    r  = np.log(df["close"] / df["close"].shift(1)).dropna().to_numpy()
    span = (df["open_time"].min(), df["open_time"].max())
    log.info(f"Rendimenti 1h: {len(r):,} barre  ({span[0]} → {span[1]})")

    fit = fit_gjr(r)
    if fit is None:
        raise RuntimeError("fit_gjr ha ritornato None (varianza degenere?)")

    # diagnostics — per-bar unconditional σ, annualized, half-life,
    # conditional-σ percentiles (to size the MC cap).
    sigma_bar   = float(np.sqrt(fit["uncond"]))
    bars_year   = 24 * 365
    sigma_ann   = sigma_bar * np.sqrt(bars_year)
    half_life_h = float(np.log(0.5) / np.log(fit["persist"]))
    sig_cond    = gjr_recursion(r, fit["omega"], fit["alpha"], fit["gamma"],
                                fit["beta"], fit["uncond"])
    pct = {p: float(np.percentile(sig_cond, p)) for p in (50, 99, 99.9, 100)}
    old = cfg.get("montecarlo", {})

    print(f"""
{'═'*64}
  GJR-GARCH(1,1) su rendimenti 1h — QMLE + variance targeting
  Campione: {len(r):,} barre  {span[0]} → {span[1]}
{'═'*64}
  {'':14}  {'1m (config attuale)':>20}  {'1h (stimato)':>16}
  omega       {old.get('gjr_omega', float('nan')):>22.3e}  {fit['omega']:>16.3e}
  alpha       {old.get('gjr_alpha', float('nan')):>22.4f}  {fit['alpha']:>16.4f}
  gamma       {old.get('gjr_gamma', float('nan')):>22.4f}  {fit['gamma']:>16.4f}
  beta        {old.get('gjr_beta',  float('nan')):>22.4f}  {fit['beta']:>16.4f}

  persistence (α+γ/2+β) : {fit['persist']:.5f}   half-life: {half_life_h:.0f}h ({half_life_h/24:.1f}d)
  σ incondizionata/barra: {sigma_bar*100:.3f}%    annualizzata: {sigma_ann*100:.1f}%
  σ condizionata p50/p99/p99.9/max: {pct[50]*100:.3f}% / {pct[99]*100:.2f}% / {pct[99.9]*100:.2f}% / {pct[100]*100:.2f}%

  ⚠ Cap σ del MC (forecast.py): il clip 1m-era è 1.0%/barra — a 1h la σ
    condizionata lo supera (p99.9={pct[99.9]*100:.2f}%): cap suggerito ≥ {np.ceil(pct[100]*100*1.5)/100:.2f}
{'═'*64}
""")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "estimated_at":   pd.Timestamp.now("UTC").isoformat(),
        "interval":       "1h",
        "n_obs":          int(len(r)),
        "span":           [str(span[0]), str(span[1])],
        "method":         "gaussian QMLE + variance targeting (fit_gjr, short_vol_hist_backtest)",
        "omega":          fit["omega"],
        "alpha":          fit["alpha"],
        "gamma":          fit["gamma"],
        "beta":           fit["beta"],
        "persistence":    fit["persist"],
        "half_life_hours": half_life_h,
        "sigma_uncond_bar": sigma_bar,
        "sigma_cond_pct": {str(k): v for k, v in pct.items()},
        "old_params_1m":  {k: old.get(k) for k in
                           ("gjr_omega", "gjr_alpha", "gjr_gamma", "gjr_beta")},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Parametri salvati → {OUT}")


if __name__ == "__main__":
    main()
