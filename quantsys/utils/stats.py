# IT: Statistiche di performance condivise — estratte VERBATIM da
#     scripts/03_backtest.py il 2026-07-16 (step 2 refactor boy-scout):
#     corpi identici, golden test in tests/test_stats_utils.py bloccano
#     l'equivalenza numerica (seed 42 del bootstrap preservato).
# EN: Shared performance statistics — extracted VERBATIM from
#     scripts/03_backtest.py on 2026-07-16 (boy-scout refactor step 2):
#     identical bodies, golden tests in tests/test_stats_utils.py lock the
#     numerical equivalence (bootstrap seed 42 preserved).
import numpy as np


# IT: Intervalli bootstrap su Sharpe/Sortino — `annualize` = barre/anno del timeframe
#     corrente (525600 a 1m, 8760 a 1h). Il default 525600 resta per retro-compatibilità,
#     ma ogni call site DEVE passare annualize=bars_per_year. Bootstrap i.i.d.
#     percentile (5000 resample, seed fisso 42); sotto 30 trade nessuna stima.
# EN: Bootstrap CI for Sharpe/Sortino — `annualize` = bars/year of the current timeframe
#     (525600 at 1m, 8760 at 1h). Default 525600 kept for backwards compatibility, but
#     every call site MUST pass annualize=bars_per_year. Percentile i.i.d. bootstrap
#     (5000 resamples, fixed seed 42); below 30 trades no estimate.
def bootstrap_sharpe_ci(pnl_list, n_boot=5000, confidence=0.95, annualize=525600):
    """Bootstrap confidence interval per Sharpe e Sortino (annualizzati a barre/anno)."""
    if len(pnl_list) < 30:
        return {"sharpe_ci_low": None, "sharpe_ci_high": None,
                "sortino_ci_low": None, "sortino_ci_high": None}
    arr = np.array(pnl_list, dtype=np.float64)
    rng = np.random.default_rng(42)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1) + 1e-10
    scale = np.sqrt(annualize / len(arr))
    sharpes = (means / stds) * scale
    neg_mask = samples < 0
    neg_counts = neg_mask.sum(axis=1)
    neg_sums = np.where(neg_mask, samples, 0.0)
    neg_means = np.where(neg_counts > 0, neg_sums.sum(axis=1) / np.maximum(neg_counts, 1), 0.0)
    neg_sq_sums = np.where(neg_mask, samples**2, 0.0).sum(axis=1)
    neg_var = np.where(neg_counts > 1,
                       neg_sq_sums / neg_counts - neg_means**2,
                       0.0)
    neg_std = np.sqrt(np.maximum(neg_var, 0.0)) + 1e-10
    dstds = np.where(neg_counts > 1, neg_std, stds)
    sortinos = (means / dstds) * scale
    alpha = (1 - confidence) / 2
    return {
        "sharpe_ci_low":  float(np.percentile(sharpes, alpha*100)),
        "sharpe_ci_high": float(np.percentile(sharpes, (1-alpha)*100)),
        "sortino_ci_low":  float(np.percentile(sortinos, alpha*100)),
        "sortino_ci_high": float(np.percentile(sortinos, (1-alpha)*100)),
    }


# IT: Durata e recovery time del max drawdown sull'equity curve.
# EN: Duration and recovery time of the max drawdown along the equity curve.
def mdd_stats(equity_arr):
    """Calcola durata e recovery time del max drawdown."""
    eq = np.array(equity_arr, dtype=np.float64)
    peak_idx, trough_idx, recovery_idx = 0, 0, 0
    max_dd_val = 0.0
    running_peak = eq[0]; running_peak_idx = 0
    for i in range(1, len(eq)):
        if eq[i] >= running_peak:
            running_peak = eq[i]; running_peak_idx = i
        dd = (running_peak - eq[i]) / running_peak if running_peak > 0 else 0
        if dd > max_dd_val:
            max_dd_val = dd
            peak_idx = running_peak_idx; trough_idx = i
    # IT: Recovery = prima candela post-trough che riallinea il peak precedente.
    # EN: Recovery = first post-trough bar that recovers the previous peak.
    recovery_idx = len(eq) - 1  # IT: default non recuperato | EN: default not recovered
    for i in range(trough_idx, len(eq)):
        if eq[i] >= eq[peak_idx]:
            recovery_idx = i; break
    recovered = eq[recovery_idx] >= eq[peak_idx]
    return {
        "mdd_duration_candles": int(trough_idx - peak_idx),
        "mdd_recovery_candles": int(recovery_idx - trough_idx) if recovered else None,
        "mdd_recovered": bool(recovered),
    }
