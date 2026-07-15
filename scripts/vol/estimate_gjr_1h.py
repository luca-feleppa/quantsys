"""
scripts/vol/estimate_gjr_1h.py
==============================
Ri-stima dei parametri GJR-GARCH(1,1) del Monte Carlo su rendimenti 1h
(TODO documentato in MODEL_IMPROVEMENTS §2 e config/default.yaml → montecarlo:
i parametri correnti sono calibrati su rendimenti BTC 1m; ω ha unità
[varianza/passo] e NON è trasferibile tra timeframe).

Metodo: QMLE gaussiano con variance targeting — riusa `fit_gjr` di
`scripts/vol/short_vol_hist_backtest.py` (single source of truth del fitter,
già validato dal backtest FHS). Stima FULL-SAMPLE: i parametri alimentano un
simulatore forward (monte_carlo_forecast), non una strategia giudicata OOS —
stesso status dei parametri 1m che sostituiscono.

Output: stampa + `results/vols/gjr_params_1h.json` (parametri, persistence,
half-life, σ incondizionata, percentili della σ condizionata, suggerimento
per il cap di σ del MC — il clip 1m-era 0.01/barra satura a 1h).

Lanciare dalla root di progetto:
  python scripts/vol/estimate_gjr_1h.py
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# IT: root progetto (scripts/vol/ → 2 livelli sopra) + import del fitter condiviso.
# EN: project root (scripts/vol/ → 2 levels up) + shared fitter import.
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
    # IT: reconfigure UTF-8 (checklist repo: il bug cp1252 è ricorso 5 volte).
    # EN: UTF-8 reconfigure (repo checklist: the cp1252 bug recurred 5 times).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config("config/default.yaml")
    interval_min = interval_minutes_from_cfg(cfg)
    # IT: la stima è specifica del timeframe: fail-fast se la config non è a 1h.
    # EN: the estimate is timeframe-specific: fail fast if config is not 1h.
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

    # IT: diagnostiche — σ incondizionata per barra, annualizzata, half-life,
    #     percentili della σ condizionata (per dimensionare il cap del MC).
    # EN: diagnostics — per-bar unconditional σ, annualized, half-life,
    #     conditional-σ percentiles (to size the MC cap).
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
