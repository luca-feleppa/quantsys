# IT: Metriche della linea VOLATILITÀ (target log_rv) — QLIKE su RV in livelli +
#     inversione z→raw del target. SINGLE SOURCE OF TRUTH condivisa tra il giudice
#     `scripts/vol/dev_vols_qlike.py` (singolo split) e l'harness walk-forward
#     `scripts/02b_walkforward_validate.py` (fold-metric vol) → un solo punto in
#     cui vive la formula QLIKE e l'inversione, niente divergenze fra i due path.
# EN: VOLATILITY-line metrics (log_rv target) — QLIKE on RV levels + z→raw target
#     inversion. SINGLE SOURCE OF TRUTH shared by the `dev_vols_qlike.py` judge
#     (single split) and the `02b_walkforward_validate.py` harness (vol fold-metric)
#     → one place for the QLIKE formula and the inversion, no divergence between
#     the two paths.
import numpy as np
import pandas as pd

# IT: stesso ε del target in FeatureBuilder | EN: same ε as the FeatureBuilder target
EPS = 1e-12


# IT: QLIKE su RV in livelli — loss canonica per la valutazione di varianza
#     (Patton 2011: robusta al rumore nella proxy di RV).
# EN: QLIKE on RV levels — canonical variance-forecast loss
#     (Patton 2011: robust to noise in the RV proxy).
def qlike(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    rv_true = np.asarray(rv_true, dtype=np.float64)
    rv_pred = np.asarray(rv_pred, dtype=np.float64)
    r = rv_true / np.maximum(rv_pred, EPS)
    return float(np.mean(r - np.log(r) - 1.0))


# IT: inversione COMPLETA z→raw del target log-RV: log_rv = z·scale + center.
#     NB `denormalize_predictions` (solo z·scale) NON basta: il log-RV ha mediana
#     ≈ −7, serve anche il centro del RobustScaler persistito nel PipelineState.
# EN: FULL z→raw inversion of the log-RV target: log_rv = z·scale + center.
#     NB `denormalize_predictions` (z·scale only) is NOT enough: log-RV has median
#     ≈ −7, the RobustScaler center persisted in PipelineState is required too.
def invert_log_rv(z: np.ndarray, center: float, scale: float) -> np.ndarray:
    return np.asarray(z, dtype=np.float64) * float(scale) + float(center)


# IT: fold-metric vol — da predizione μ e target y ENTRAMBI in spazio z, ricostruisce
#     log-RV (center+scale) ed esponenzia → RV in livelli, poi QLIKE + MSE(log).
#     Usata dall'harness walk-forward per giudicare i fold senza ri-derivare la RV
#     dai raw candle (l'`y` dell'npz È già il log-RV z-scorato col medesimo scaler).
# EN: vol fold-metric — from μ prediction and y target BOTH in z-space, reconstructs
#     log-RV (center+scale) and exponentiates → RV levels, then QLIKE + MSE(log).
#     Used by the walk-forward harness to judge folds without re-deriving RV from
#     raw candles (the npz `y` IS already the log-RV z-scored with the same scaler).
def qlike_from_z(y_true_z: np.ndarray, mu_pred_z: np.ndarray,
                 center: float, scale: float) -> dict:
    log_true = invert_log_rv(y_true_z, center, scale)
    log_pred = invert_log_rv(mu_pred_z, center, scale)
    return {
        "qlike":   qlike(np.exp(log_true), np.exp(log_pred)),
        "mse_log": float(np.mean((log_true - log_pred) ** 2)),
    }


# IT: normalizza un indice temporale a tz-naive (gestisce sia naive sia tz-aware UTC),
#     per l'allineamento HAR↔split coerente col giudice dev_vols_qlike.py.
# EN: normalize a time index to tz-naive (handles both naive and tz-aware UTC),
#     for HAR↔split alignment consistent with the dev_vols_qlike.py judge.
def _to_naive(values) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


# IT: componenti HAR-RV (Corsi 2009) + target fwd dai raw candles. STESSA definizione
#     del giudice dev_vols_qlike.py (single source): RV trailing h-barre/7d/30d in
#     log, componenti settimana/mese riscalate all'orizzonte h; target = log(RV fwd h).
# EN: HAR-RV components (Corsi 2009) + fwd target from raw candles. SAME definition as
#     the dev_vols_qlike.py judge (single source): trailing h-bar/7d/30d RV in log,
#     weekly/monthly components rescaled to the h-bar horizon; target = log(fwd h RV).
def build_har_frame(raw: pd.DataFrame, h: int, bars_day: int) -> pd.DataFrame:
    raw = raw.sort_values("open_time").reset_index(drop=True)
    lr2 = np.log(raw["close"] / raw["close"].shift(1)) ** 2
    rv_h = lr2.rolling(h).sum()                      # IT: RV trailing h barre | EN: trailing h-bar RV
    rv_w = lr2.rolling(7 * bars_day).sum() / 7       # IT: media giornaliera 7gg | EN: 7d daily mean
    rv_m = lr2.rolling(30 * bars_day).sum() / 30     # IT: media giornaliera 30gg | EN: 30d daily mean
    rv_fwd = lr2.rolling(h).sum().shift(-h)          # IT/EN: target (formula del FeatureBuilder)
    har = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":  np.log(rv_fwd + EPS),
        "xh": np.log(rv_h + EPS),
        "xw": np.log(rv_w * (h / bars_day) + EPS),
        "xm": np.log(rv_m * (h / bars_day) + EPS),
    }).dropna().set_index("open_time")
    har.index = _to_naive(har.index)
    return har


# IT: fit OLS dell'HAR sui timestamp di train del fold, valuta sui timestamp held-out:
#     QLIKE su RV in livelli (HAR) + naive persistence (RV trailing h). Same info set
#     del NN per fold → confronto fair. NaN se allineamento insufficiente.
# EN: OLS-fit HAR on the fold's train timestamps, evaluate on the held-out timestamps:
#     QLIKE on RV levels (HAR) + naive persistence (trailing h-bar RV). Same info set
#     as the per-fold NN → fair comparison. NaN if alignment is insufficient.
def har_fold_qlike(har: pd.DataFrame, t_train, t_eval) -> dict:
    tr_idx = _to_naive(t_train)
    ev_idx = _to_naive(t_eval)
    tr = har.loc[har.index.intersection(tr_idx)]
    ev = har.loc[har.index.intersection(ev_idx)]
    if len(tr) < 50 or len(ev) < 1:
        return {"qlike_har": float("nan"), "qlike_naive": float("nan"),
                "n_har": int(len(ev)), "n_eval": int(len(ev_idx))}
    Xtr = np.column_stack([np.ones(len(tr)), tr[["xh", "xw", "xm"]].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    Xev = np.column_stack([np.ones(len(ev)), ev[["xh", "xw", "xm"]].values])
    rv_true = np.exp(ev["y"].values)
    return {
        "qlike_har":   qlike(rv_true, np.exp(Xev @ beta)),
        "qlike_naive": qlike(rv_true, np.exp(ev["xh"].values)),
        "n_har": int(len(ev)), "n_eval": int(len(ev_idx)),
    }
