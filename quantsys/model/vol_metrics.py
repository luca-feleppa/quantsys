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
    return float(np.mean(qlike_series(rv_true, rv_pred)))


# IT: QLIKE PER-CAMPIONE (loss non aggregata) — serve al test di Diebold-Mariano,
#     che opera sui differenziali di loss e non sulle medie. `qlike()` è la media
#     di questa serie: un solo punto in cui vive la formula.
# EN: PER-SAMPLE QLIKE (unaggregated loss) — needed by the Diebold-Mariano test,
#     which operates on loss differentials rather than means. `qlike()` is the mean
#     of this series: the formula lives in exactly one place.
def qlike_series(rv_true: np.ndarray, rv_pred: np.ndarray) -> np.ndarray:
    rv_true = np.asarray(rv_true, dtype=np.float64)
    rv_pred = np.asarray(rv_pred, dtype=np.float64)
    r = rv_true / np.maximum(rv_pred, EPS)
    return r - np.log(r) - 1.0


# IT: TEST DI DIEBOLD-MARIANO (1995) con varianza HAC e correzione small-sample di
#     Harvey-Leybourne-Newbold (1997). Risponde a: "la differenza di loss media fra
#     due forecast è distinguibile da zero?" — indispensabile qui perché il target
#     somma h barre, quindi le finestre si SOVRAPPONGONO e i differenziali sono
#     fortemente autocorrelati: con la varianza iid lo standard error sarebbe
#     sottostimato di ~sqrt(h) e ogni p-value sarebbe fittizio.
#     Convenzione: d_t = loss_a − loss_b → DM < 0 ⇒ il forecast A perde MENO (è
#     migliore). Lag HAC di default q = h−1 (scelta canonica per forecast a h passi);
#     kernel di Bartlett (pesi 1 − j/(q+1)) → varianza garantita non negativa.
#     N_eff ≈ n/h è la numerosità che governa davvero l'incertezza, riportata perché
#     è la prima obiezione che un lettore competente muove al confronto.
# EN: DIEBOLD-MARIANO (1995) TEST with HAC variance and the Harvey-Leybourne-Newbold
#     (1997) small-sample correction. Answers: "is the mean loss difference between
#     two forecasts distinguishable from zero?" — essential here because the target
#     sums h bars, so windows OVERLAP and the differentials are strongly
#     autocorrelated: under an iid variance the standard error would be understated
#     by ~sqrt(h) and any p-value would be fictitious.
#     Convention: d_t = loss_a − loss_b → DM < 0 ⇒ forecast A loses LESS (is better).
#     Default HAC lag q = h−1 (canonical for h-step forecasts); Bartlett kernel
#     (weights 1 − j/(q+1)) → guarantees a non-negative variance.
#     N_eff ≈ n/h is the sample size that truly governs the uncertainty, reported
#     because it is the first objection a competent reader raises.
def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int,
                    lag: int | None = None) -> dict:
    from scipy import stats  # IT/EN: import locale (scipy non serve al path training)

    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = int(d.size)

    # IT: contratto di ritorno UNIFORME — ogni ramo (anche i degeneri) restituisce le
    #     stesse chiavi con nan, così i consumer non devono difendersi da KeyError.
    # EN: UNIFORM return contract — every branch (degenerate ones included) returns
    #     the same keys with nan, so consumers need no KeyError defence.
    def _null(note: str, lag_used: int = -1) -> dict:
        return {"dm": float("nan"), "dm_hln": float("nan"), "p_value": float("nan"),
                "mean_diff": float(d.mean()) if n else float("nan"),
                "hac_lag": int(lag_used), "n": n,
                "n_eff": float(n / max(h, 1)), "better": None, "note": note}

    if n < 10:
        return _null("campione troppo piccolo / sample too small")

    q = int(h - 1 if lag is None else lag)
    q = max(0, min(q, n - 1))
    d_bar = float(d.mean())
    dev = d - d_bar

    # IT: varianza HAC (Newey-West, kernel di Bartlett) della MEDIA di d.
    # EN: HAC (Newey-West, Bartlett kernel) variance of the MEAN of d.
    gamma0 = float(np.mean(dev ** 2))
    var_hac = gamma0
    for j in range(1, q + 1):
        gamma_j = float(np.mean(dev[j:] * dev[:-j]))
        var_hac += 2.0 * (1.0 - j / (q + 1.0)) * gamma_j
    var_hac = max(var_hac, 0.0)

    # IT: differenziale costante (varianza nulla) o identicamente zero → nessuna
    #     statistica definita. Caso reale: due forecast che differiscono di una
    #     costante additiva sulla loss.
    # EN: constant differential (null variance) or identically zero → no statistic is
    #     defined. Real case: two forecasts differing by an additive loss constant.
    if var_hac <= 0.0 or d_bar == 0.0:
        return _null("varianza HAC nulla / null HAC variance", lag_used=q)

    dm = d_bar / np.sqrt(var_hac / n)

    # IT: correzione HLN 1997 — con n finito e orizzonte h la DM è sovradimensionata;
    #     il fattore corregge la statistica e si confronta con una t di Student a n−1
    #     gradi di libertà invece della normale.
    # EN: HLN 1997 correction — with finite n and horizon h the DM statistic is
    #     oversized; the factor rescales it and it is compared against a Student-t
    #     with n−1 degrees of freedom rather than the normal.
    hln = (n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n
    dm_hln = dm * np.sqrt(hln) if hln > 0 else float("nan")
    stat = dm_hln if np.isfinite(dm_hln) else dm
    p = float(2.0 * stats.t.sf(abs(stat), df=n - 1))

    return {
        "dm": float(dm),                       # IT/EN: statistica non corretta / uncorrected
        "dm_hln": float(dm_hln),               # IT/EN: corretta HLN / HLN-corrected
        "p_value": p,                          # IT/EN: bilaterale sulla t / two-sided, t-dist
        "mean_diff": d_bar,                    # IT/EN: loss_a − loss_b medio / mean
        "hac_lag": q,
        "n": n,
        "n_eff": float(n / max(h, 1)),         # IT/EN: osservazioni non sovrapposte / non-overlapping
        "better": ("a" if d_bar < 0 else "b"), # IT/EN: chi perde meno / who loses less
    }


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
