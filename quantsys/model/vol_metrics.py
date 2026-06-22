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
