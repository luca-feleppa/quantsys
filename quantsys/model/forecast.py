"""
quantsys/model/forecast.py
==========================
Monte Carlo autoregressivo LSTM-guided con aggiornamento multi-feature.

FIX CONCETTUALE (versione precedente):
  Al passo t aggiornava SOLO log_ret nella finestra, lasciando le altre
  54 features (VWAP deviation, RSI, lag, vol_std, ecc.) congelate
  all'ultimo valore reale osservato. Dopo 30 step la finestra era
  internamente incoerente: log_ret simulato + tutto il resto storico.

FIX ATTUALE:
  Aggiorna autoregressivamente tutte le features derivabili dal log_ret
  (lag, vol_std rolling, vol_ratio, vwap_dev approssimata).
  Le features non derivabili (volume reale, VP POC, taker ratio)
  rimangono all'ultimo valore — semplificazione consapevole, molto
  meno grave del congelamento totale.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("quantsys.model.forecast")


# IT: Mappa nome→indice colonna per le feature aggiornate nel rollout MC.
# EN: Name→column-index map for features updated during the MC rollout.
def build_feature_idx_map(feature_names: list[str]) -> dict:
    """
    Costruisce il dizionario {nome_feature: indice_colonna} dalla lista
    dei nomi salvata in lstm_dataset.npz (feature_names).

    Da chiamare prima di monte_carlo_forecast per abilitare
    l'aggiornamento multi-feature.

    Esempio:
        data = np.load("data/lstm_dataset.npz", allow_pickle=True)
        feat_names = list(data["feature_names"])
        idx_map = build_feature_idx_map(feat_names)
        result = monte_carlo_forecast(model, x, price, feature_idx_map=idx_map)
    """
    return {name: i for i, name in enumerate(feature_names)}


# IT: Rollout MC autoregressivo: GJR-GARCH + t-Student per il drift LSTM.
# EN: Autoregressive MC rollout: GJR-GARCH + t-Student sampling around LSTM drift.
def monte_carlo_forecast(
    model:               "torch.nn.Module",
    x_price_seed:        np.ndarray,          # (1, window, n_price_features)
    last_price:          float,
    n_steps:             int             = 30,
    n_paths:             int             = 1500,
    x_macro_seed:        Optional[np.ndarray] = None,
    device:              Optional["torch.device"] = None,
    feature_idx_log_ret: int             = 0,
    feature_idx_map:     Optional[dict]  = None,
    gjr_omega:           float           = 1.2e-5,
    gjr_alpha:           float           = 0.05,
    gjr_gamma:           float           = 0.065,
    gjr_beta:            float           = 0.875,
) -> dict:
    """
    Genera n_paths traiettorie di n_steps passi con aggiornamento
    multi-feature autoregressivo.

    Args:
        feature_idx_map: dizionario {nome_feature: indice} da build_feature_idx_map().
                         Se None, aggiorna solo log_ret (comportamento minimale).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    has_macro = x_macro_seed is not None

    x_batch  = np.repeat(x_price_seed, n_paths, axis=0).astype(np.float32)
    xm_batch = np.repeat(x_macro_seed, n_paths, axis=0).astype(np.float32) \
               if has_macro else None

    idx_lr  = feature_idx_log_ret
    idx_map = feature_idx_map or {}

    # IT: Indici delle feature derivabili da log_ret (le altre restano frozen).
    # EN: Indices of features derivable from log_ret (others stay frozen).
    idx_lag    = [idx_map[f"lag_ret_{i}"] for i in range(1, 6)
                  if f"lag_ret_{i}" in idx_map]
    idx_vol5   = idx_map.get("vol_std_5")
    idx_vol20  = idx_map.get("vol_std_20")
    idx_ratio  = idx_map.get("vol_ratio_5_20")
    idx_vwapd  = idx_map.get("vwap_dev")

    n_feat_log = len(idx_lag) + sum(x is not None for x in [idx_vol5, idx_vol20, idx_ratio, idx_vwapd])
    log.debug(f"Monte Carlo: {n_feat_log + 1} features aggiornate autoregressivamente")

    # IT: Inizializza σ GARCH con la σ predetta dal modello sul seed.
    # EN: Initializes GARCH σ with the model-predicted σ on the seed window.
    with torch.no_grad():
        xb   = torch.tensor(x_batch[:1], dtype=torch.float32, device=device)
        xm   = torch.tensor(xm_batch[:1], dtype=torch.float32, device=device) if has_macro else None
        out0 = model(xb, xm) if has_macro else model(xb)
        s2_init = float((F.softplus(out0[1]) + 1e-6).sqrt().mean().item())

    garch_vol  = np.full(n_paths, s2_init, dtype=np.float32)
    prices     = np.full((n_steps + 1, n_paths), last_price, dtype=np.float32)
    mu_path    = np.zeros(n_steps, dtype=np.float32)
    sigma_path = np.zeros(n_steps, dtype=np.float32)
    nu_path    = np.zeros(n_steps, dtype=np.float32)

    # IT: Buffer rolling log_ret per path: feeds vol rolling + lag features.
    # EN: Rolling log_ret buffer per path: feeds rolling vol + lag features.
    lr_buf = x_batch[:, :, idx_lr].copy()   # (n_paths, window)

    # IT: GJR-GARCH(1,1): σ²_t = ω + (α + γ·I_neg)·ε²_{t-1} + β·σ²_{t-1}.
    # EN: GJR-GARCH(1,1): σ²_t = ω + (α + γ·I_neg)·ε²_{t-1} + β·σ²_{t-1}.
    # IT: γ cattura l'asimmetria (drawdown aumentano la vol più dei rialzi).
    # EN: γ captures asymmetry (drawdowns increase vol more than upticks).

    for t in range(n_steps):
        # IT: Forward pass batched su tutti i path (vettorizzato).
        # EN: Batched forward pass over all paths (vectorized).
        xb  = torch.tensor(x_batch,  dtype=torch.float32, device=device)
        xm_ = torch.tensor(xm_batch, dtype=torch.float32, device=device) if has_macro else None

        with torch.no_grad():
            out   = model(xb, xm_) if has_macro else model(xb)
            mu_t  = out[0].cpu().numpy()
            sig_t = (F.softplus(out[1]) + 1e-6).sqrt().cpu().numpy()
            nu_t  = (F.softplus(out[2]) + 2.0  + 1e-6).cpu().numpy()

        # IT: σ_eff = 0.6·σ_model + 0.4·σ_GARCH (trend vs shock locali).
        # EN: σ_eff = 0.6·σ_model + 0.4·σ_GARCH (trend vs local shocks).
        sigma_eff = 0.6 * sig_t + 0.4 * garch_vol

        # IT: t-Student sample = N(0,1)/√(χ²_ν/ν) via Box-Muller + Gamma.
        # EN: t-Student sample = N(0,1)/√(χ²_ν/ν) via Box-Muller + Gamma.
        u1      = np.random.uniform(1e-9, 1.0, n_paths)
        u2      = np.random.uniform(0.0,  1.0, n_paths)
        z       = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        nu_clip = np.clip(nu_t, 3.0, 15.0).astype(np.float64)
        chi2    = np.random.gamma(nu_clip / 2.0, 2.0).astype(np.float32)
        log_ret = np.clip(mu_t + sigma_eff * (z / np.sqrt(chi2 / nu_clip)), -0.05, 0.05)

        prices[t + 1] = prices[t] * np.exp(log_ret)

        # IT: Step GJR-GARCH | EN: GJR-GARCH update step
        neg_shock = (log_ret < 0).astype(np.float32)   # IT: I_neg | EN: I_neg
        garch_var = np.maximum(
            gjr_omega
            + (gjr_alpha + gjr_gamma * neg_shock) * (log_ret**2)
            + gjr_beta * (garch_vol**2),
            1e-8,
        )
        garch_vol = np.sqrt(np.clip(garch_var, 1e-10, 0.01**2))
        mu_path[t]    = float(mu_t.mean())
        sigma_path[t] = float(sig_t.mean())
        nu_path[t]    = float(nu_t.mean())

        # IT: Aggiornamento autoregressivo delle feature derivabili da log_ret.
        # EN: Autoregressive update of features derivable from log_ret.
        new_step = x_batch[:, -1:, :].copy()   # (n_paths, 1, n_feat)

        new_step[:, 0, idx_lr] = log_ret
        lr_buf = np.concatenate([lr_buf[:, 1:], log_ret[:, np.newaxis]], axis=1)

        # IT: Lag features: lag_ret_i = log_ret di i step fa.
        # EN: Lag features: lag_ret_i = log_ret from i steps ago.
        for lag_i, col_i in enumerate(idx_lag, 1):
            if lag_i == 1:
                new_step[:, 0, col_i] = log_ret
            else:
                new_step[:, 0, col_i] = x_batch[:, -(lag_i - 1), idx_lr]

        # IT: Vol std rolling ricomputate sui log_ret simulati.
        # EN: Rolling vol std recomputed on simulated log_ret.
        if idx_vol5 is not None:
            vs5 = lr_buf[:, -5:].std(axis=1).astype(np.float32)
            new_step[:, 0, idx_vol5] = vs5

        if idx_vol20 is not None:
            vs20 = lr_buf[:, -20:].std(axis=1).astype(np.float32)
            new_step[:, 0, idx_vol20] = vs20
            if idx_ratio is not None and idx_vol5 is not None:
                new_step[:, 0, idx_ratio] = (vs5 / np.maximum(vs20, 1e-9)).astype(np.float32)

        # IT: VWAP deviation approx: somma log_ret 60min = proxy drift.
        # EN: Approx VWAP deviation: 60-min log_ret sum = drift proxy.
        if idx_vwapd is not None:
            cum_drift = lr_buf[:, -60:].sum(axis=1).astype(np.float32)
            new_step[:, 0, idx_vwapd] = cum_drift

        # IT: Sliding window forward di 1 step | EN: Sliding window forward by 1 step
        x_batch = np.concatenate([x_batch[:, 1:, :], new_step], axis=1)

    # ── Percentili ────────────────────────────────────────────────────────
    # IT: Aggrega i path in percentili per step → bande di confidenza.
    # EN: Aggregates paths into per-step percentiles → confidence bands.
    future = prices[1:]   # (n_steps, n_paths)
    pct_levels = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    result = {
        f"p{p:02d}": np.percentile(future, p, axis=1).tolist()
        for p in pct_levels
    }
    result["mean"]         = future.mean(axis=1).tolist()
    result["std"]          = future.std(axis=1).tolist()
    result["paths_sample"] = prices[:, np.random.choice(n_paths, min(50, n_paths), replace=False)].tolist()
    result["mu_path"]      = mu_path.tolist()
    result["sigma_path"]   = sigma_path.tolist()
    result["nu_path"]      = nu_path.tolist()

    n_updated = 1 + len(idx_lag) + sum(x is not None for x in [idx_vol5, idx_vol20, idx_ratio, idx_vwapd])
    total_feat = x_price_seed.shape[-1]
    log.info(
        f"Monte Carlo: {n_steps}step × {n_paths}path | "
        f"{n_updated}/{total_feat} features aggiornate | "
        f"p50 finale=${float(np.percentile(future[-1], 50)):,.0f}"
    )
    return result


# IT: Formatta un riepilogo testuale del forecast (p50/p05/p95, CI90, μ/σ).
# EN: Formats a textual forecast summary (p50/p05/p95, CI90, μ/σ).
def summarize_forecast(result: dict, last_price: float, steps: int = 30) -> str:
    p50  = result["p50"][-1]
    p05  = result["p05"][-1]
    p95  = result["p95"][-1]
    mu   = float(np.mean(result["mu_path"]))
    sig  = float(np.mean(result["sigma_path"]))
    return (
        f"\n  Forecast {steps} minuti\n"
        f"  {'─'*44}\n"
        f"  Prezzo attuale   : ${last_price:>10,.1f}\n"
        f"  Mediana (p50)    : ${p50:>10,.1f}  ({(p50/last_price-1)*100:+.2f}%)\n"
        f"  Pessimista (p05) : ${p05:>10,.1f}  ({(p05/last_price-1)*100:+.2f}%)\n"
        f"  Ottimista  (p95) : ${p95:>10,.1f}  ({(p95/last_price-1)*100:+.2f}%)\n"
        f"  Ampiezza CI90    : ${p95-p05:>10,.0f}  ({(p95-p05)/last_price*100:.2f}%)\n"
        f"  μ LSTM medio     : {mu:+.6f}  σ LSTM medio: {sig:.6f}\n"
    )
