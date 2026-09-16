"""
quantsys/macro/regime.py
========================
Two-stage Economic Regime Detection:

  STAGE 1 — unsupervised HMM (Hidden Markov Model)
    Automatically finds N latent regimes in historical macro data.
    Output: probability of being in each regime for every day.

  STAGE 2 — MacroEncoder (MLP)
    Lightweight neural network that turns the raw macro features
    into a dense 16-dimensional embedding.
    Trained TOGETHER with the main LSTM (end-to-end).

Why an HMM?
  · Explicitly models that the regime changes over time (transitions)
  · The 4 regimes emerge from the data — we do not define them a priori
  · Regime probabilities (e.g. [0.8, 0.1, 0.05, 0.05]) are
    interpretable and robust even with missing data

The 4 regimes the HMM typically discovers in US data:
  0 = Moderate expansion     (stable growth, low inflation)
  1 = Overheating            (high growth, rising inflation)
  2 = Stagflation / crisis   (low growth, high inflation)
  3 = Recession / Fed pivot  (negative growth, Fed cutting)

Note: the labels are interpretive — the HMM only learns patterns.
"""

import logging
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

log = logging.getLogger("quantsys.macro.regime")

# Key macro columns for the HMM (the most informative about the regime)
# Order: [inflation, growth, labor_market, financial_conditions, rates]
HMM_CORE_FEATURES = [
    "cpi_yoy_yoy",
    "core_cpi_yoy_yoy",
    "infl_exp_5y",
    "gdp_growth",
    "lei",
    "unemployment",
    "nfp_mom",
    "claims_chg",
    "fed_funds",
    "yield_curve_2_10",
    "real_rate_10y",
    "credit_spread_hy",
    "vix",
    "nfci",
]


# ─── STAGE 1: HMM REGIME DETECTOR ───────────────────────────────────────────

# Walk-forward Gaussian HMM (expanding window) labeling macro regimes without look-ahead.
class RegimeHMM:
    """
    Gaussian HMM with N latent states for economic regime detection.

    LOOK-AHEAD BIAS FIX — Walk-Forward Expanding Window:
    ─────────────────────────────────────────────────────
    The original problem: the HMM was trained on the whole history
    (e.g. 2018-2024) and then applied to classify every day, including
    those of 2018. But the HMM "already knew" how things ended in 2024
    when classifying 2018 — subtle but real look-ahead bias.

    The solution — expanding window:
      · For every date t in the dataset, the HMM is trained ONLY on the data
        available up to t-1 (no future information).
      · In practice: we train the HMM on an initial burn-in (e.g. 2 years),
        then update it incrementally every N days with new data.
      · Each day's regime labels are generated with the model that
        existed at that moment, not with the final one.

    Trade-off:
      · Slower (K trainings instead of 1, where K = n_updates).
      · More realistic: the regime probabilities the MacroEncoder
        receives during training match those available
        at that instant, with no information from the future.
    """

    # Sets regime count, EM iterations and restarts; the scaler is fit at fit time.
    def __init__(self, n_regimes: int = 4, n_iter: int = 100, random_state: int = 42,
                 n_restarts: int = 5):
        self.n_regimes    = n_regimes
        self.n_iter       = n_iter
        self.random_state = random_state
        self.n_restarts   = n_restarts
        self.model        = None   # current model (the last one trained)
        self.scaler       = RobustScaler()
        self.feature_cols: list[str] = []

    # Picks the available HMM_CORE_FEATURES; falls back to the first 20 numeric columns.
    def _select_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Select the available columns among the ideal ones for the HMM."""
        available = [c for c in HMM_CORE_FEATURES if c in df.columns]
        if len(available) < 4:
            # Fallback: use all available numeric columns (up to 20)
            available = [c for c in df.select_dtypes("number").columns][:20]
        log.info(f"HMM: {len(available)} features selezionate")
        return df[available].values, available

    # Fits the HMM over n_restarts seeds; keeps the max log-likelihood model (early-exit on convergence).
    def _fit_single(self, X_norm: np.ndarray):
        """
        Train the HMM with n_restarts different seeds, return the model with maximum log-likelihood.

        Why restarts: GaussianHMM with covariance_type="full" has 14×14×4=784 covariance
        parameters. With short windows (365-500 days) EM often gets stuck in local
        optima with decreasing log-likelihood. Trying N different seeds and keeping the
        highest-scoring model yields a stable solution without changing the
        statistical model.
        Early exit: if a restart converges before exhausting n_iter, no need to try others.
        """
        try:
            from hmmlearn import hmm as hmmlearn_hmm
        except ImportError:
            raise ImportError("pip install hmmlearn")

        import warnings

        best_model = None
        best_score = -np.inf

        for k in range(self.n_restarts):
            seed = self.random_state + k
            model = hmmlearn_hmm.GaussianHMM(
                n_components    = self.n_regimes,
                covariance_type = "full",
                n_iter          = self.n_iter,
                random_state    = seed,
                verbose         = False,
                tol             = 1e-3,
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_norm)
                score = model.score(X_norm)
                if score > best_score:
                    best_score = score
                    best_model = model
                if model.monitor_.converged:
                    break   # converged → no point trying other seeds
            except Exception:
                continue

        return best_model

    # Produces day-by-day regime labels using only past data (no look-ahead).
    def fit_predict_walkforward(
        self,
        df_macro:     pd.DataFrame,
        burn_in_days: int = 365,   # minimum days of history before predicting
        retrain_days: int = 90,    # retrain every N days
    ) -> pd.DataFrame:
        """
        Walk-forward expanding window:
        for every day t generates regime probabilities using ONLY
        the data available up to t (no look-ahead).

        Args:
            df_macro:     daily DataFrame with macro features
            burn_in_days: minimum days before starting to classify
            retrain_days: retraining frequency (e.g. every 90 days)

        Returns:
            DataFrame with columns regime_prob_0..K and regime_dominant,
            indexed like df_macro. The first burn_in_days rows
            have uniform probabilities (1/n_regimes) — burn-in period.
        """
        if "open_time" in df_macro.columns:
            df_daily = df_macro.groupby(
                df_macro["open_time"].dt.date
            ).first().reset_index(drop=True)
        else:
            df_daily = df_macro.copy()

        X_raw, self.feature_cols = self._select_features(df_daily)
        n = len(X_raw)

        if n < burn_in_days + 10:
            raise ValueError(
                f"Troppo pochi dati ({n} giorni) per walk-forward con burn_in={burn_in_days}."
            )

        # Global normalization: we fit the scaler on the whole history
        # (the scaler does not "know the future" — it only uses location/scale statistics)
        # Stricter alternative: expanding scaler. But RobustScaler is
        # already robust to outliers, and the normalization carries no look-ahead
        # into the regime probabilities.
        mask_valid = ~np.isnan(X_raw).any(axis=1)
        self.scaler.fit(X_raw[mask_valid])
        X_norm = np.clip(self.scaler.transform(
            np.nan_to_num(X_raw, nan=0.0)
        ), -5, 5)

        # Result storage — starts with uniform probabilities (no-info prior)
        probs_all = np.full((n, self.n_regimes), 1.0 / self.n_regimes)

        current_model = None
        last_retrain  = -1

        log.info(
            f"Walk-forward HMM: {n} giorni, burn_in={burn_in_days}, "
            f"retrain ogni {retrain_days} gg ..."
        )

        for t in range(burn_in_days, n):
            # Retrain if: first model, or the retrain period has elapsed
            if current_model is None or (t - last_retrain) >= retrain_days:
                # Use ONLY data up to t (excludes t itself and the future)
                X_train = X_norm[:t]
                mask_t  = ~np.isnan(X_raw[:t]).any(axis=1)
                X_train = X_train[mask_t]

                if len(X_train) >= 30:
                    try:
                        current_model = self._fit_single(X_train)
                        last_retrain  = t
                        log.debug(f"  HMM riaddestrato a t={t} ({t} giorni di storia)")
                    except Exception as e:
                        log.warning(f"  HMM fit fallito a t={t}: {e}")

            # Generate probabilities for day t with the current model
            if current_model is not None:
                x_t = X_norm[t:t+1]
                if not np.isnan(X_raw[t]).any():
                    try:
                        probs_all[t] = current_model.predict_proba(x_t)[0]
                    except Exception:
                        pass  # keep uniform probabilities

        # Save the final model (trained on the whole history)
        if current_model is not None:
            self.model = current_model

        # Build the result DataFrame aligned to df_macro's index
        result = pd.DataFrame(
            probs_all,
            index  = df_daily.index,
            columns= [f"regime_prob_{i}" for i in range(self.n_regimes)],
        )
        result["regime_dominant"] = probs_all.argmax(axis=1)
        # The first burn_in_days rows have uniform probabilities
        result["regime_burn_in"] = False
        result.iloc[:burn_in_days, -1] = True

        log.info(
            f"Walk-forward completato. "
            f"Burn-in: {burn_in_days} giorni. "
            f"Classificati: {n - burn_in_days} giorni."
        )
        self._describe_regimes_wf(X_raw, probs_all, burn_in_days)
        return result

    # Logs each regime's mean characteristics over the classified period (diagnostics).
    def _describe_regimes_wf(self, X_raw, probs_all, burn_in_days):
        """Regime analysis over the classified period (post burn-in)."""
        labels = probs_all[burn_in_days:].argmax(axis=1)
        X_post = X_raw[burn_in_days:]
        log.info("─── Caratteristiche regimi (post burn-in) ──────────────")
        for r in range(self.n_regimes):
            mask = labels == r
            if mask.sum() == 0:
                continue
            means    = X_post[mask].mean(axis=0)
            feat_str = "  ".join(
                f"{self.feature_cols[i].replace('macro_','')[:12]}={means[i]:.2f}"
                for i in range(min(4, len(self.feature_cols)))
            )
            log.info(f"  Regime {r} ({mask.sum()} gg, {mask.mean():.0%}): {feat_str}")

    # Final fit on the full history for the production model (NOT for training labels).
    def fit(self, df_macro: pd.DataFrame) -> "RegimeHMM":
        """
        Final training on the whole history (for the production model).
        Use ONLY for the model running in production/live — NOT to
        generate training labels (use fit_predict_walkforward).
        """
        if "open_time" in df_macro.columns:
            df_daily = df_macro.groupby(
                df_macro["open_time"].dt.date
            ).first().reset_index(drop=True)
        else:
            df_daily = df_macro

        X_raw, self.feature_cols = self._select_features(df_daily)
        mask = ~np.isnan(X_raw).any(axis=1)
        X_raw = X_raw[mask]

        if len(X_raw) < 50:
            raise ValueError(f"Troppo pochi dati ({len(X_raw)} giorni).")

        X = np.clip(self.scaler.fit_transform(X_raw), -5, 5)
        log.info(f"HMM fit finale: {self.n_regimes} regimi, {len(X)} osservazioni")
        self.model = self._fit_single(X)
        return self

    # Per-row regime probabilities via the trained HMM's predict_proba.
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities (uses the current model)."""
        if self.model is None:
            raise RuntimeError("HMM non addestrato.")
        X_raw = df[self.feature_cols].values if self.feature_cols else df.values
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        X     = np.clip(self.scaler.transform(X_raw), -5, 5)
        return self.model.predict_proba(X)

    # Serializes model + scaler + metadata to disk (pickle).
    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "scaler": self.scaler,
                "feature_cols": self.feature_cols, "n_regimes": self.n_regimes,
            }, f)
        log.info(f"HMM salvato → {path}")

    # Reconstructs a RegimeHMM from a saved pickle.
    @classmethod
    def load(cls, path: str) -> "RegimeHMM":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(n_regimes=data["n_regimes"])
        obj.model        = data["model"]
        obj.scaler       = data["scaler"]
        obj.feature_cols = data["feature_cols"]
        return obj


# ─── STAGE 1b: MARKOV-SWITCHING REGIME DETECTOR (Hamilton 1989) ─────────────

# Markov-Switching (Hamilton 1989) on PC1 with expanding-window PCA + walk-forward Hamilton filter.
class RegimeMarkovSwitching:
    """
    Markov-Switching Regression (Hamilton 1989) for regime detection.

    Standard quant-finance pipeline:
      1. RobustScaler on the macro features (outlier resistant)
      2. PCA → n_pca principal components (reduces multicollinearity)
      3. MarkovRegression on PC1 with switching mean + variance
      4. Hamilton filter for sequential regime probabilities

    Advantages over GaussianHMM:
      · Stable convergence (EM + scoring optimizer, not EM alone)
      · Published econometric standard (Hamilton 1989, Kim & Nelson 1999)
      · PCA removes multicollinearity among the 14 macro features
      · switching_variance captures volatility changes across regimes
      · Fewer parameters → less overfitting with short windows

    Walk-forward approach:
      Identical to RegimeHMM — expanding window with periodic retraining.
      Between retrains, uses the Hamilton filter (O(1) per day)
      instead of refitting the model.
    """

    # Sets regime/PCA/restart counts; scaler, PCA and param cache are populated at fit time.
    # `max_fit_failure_ratio`: walk-forward abort threshold (see the guard in
    # fit_predict_walkforward). 1.0 = no ratio-based abort, but the abort on ZERO
    # successful fits still stands — it is not disableable, because a walk-forward
    # without a single fit yields no information, it yields the prior.
    def __init__(self, n_regimes: int = 3, n_iter: int = 300,
                 random_state: int = 42, n_pca: int = 3,
                 n_restarts: int = 5, max_fit_failure_ratio: float = 0.5):
        if n_regimes < 2:
            raise ValueError(f"n_regimes deve essere >= 2, ricevuto {n_regimes}")
        if not 0.0 <= max_fit_failure_ratio <= 1.0:
            raise ValueError(
                f"max_fit_failure_ratio deve essere in [0,1], "
                f"ricevuto {max_fit_failure_ratio}"
            )
        self.n_regimes    = n_regimes
        self.n_iter       = n_iter
        self.random_state = random_state
        self.n_pca        = n_pca
        self.n_restarts   = n_restarts
        self.max_fit_failure_ratio = max_fit_failure_ratio
        # last walk-forward diagnostics (filled by the guard) — readable by
        # callers to log rebuild quality without re-parsing the logs.
        self.last_fit_diagnostics: dict | None = None
        self.model        = None
        self.pca          = None
        self.scaler       = RobustScaler()
        self.feature_cols: list[str] = []
        self._params_cache: dict | None = None
        # walk-forward chain state (B7): populated at the end of fit_predict_walkforward,
        # serialized into the checkpoint for incremental continuation.
        self._wf_state: dict | None = None

    # Like the HMM version, but also drops all-NaN columns (PCA cannot handle them).
    def _select_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        available = [c for c in HMM_CORE_FEATURES if c in df.columns]
        if len(available) < 4:
            available = [c for c in df.select_dtypes("number").columns][:20]
        available = [c for c in available
                     if not df[c].isna().all()]
        log.info(f"MarkovSwitching: {len(available)} features selezionate")
        return df[available].values, available

    # Fits the PCA and projects X_norm onto principal components (logs explained variance).
    def _pca_fit_transform(self, X_norm: np.ndarray) -> np.ndarray:
        n_comp = min(self.n_pca, X_norm.shape[1], X_norm.shape[0])
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X_norm)
        var_explained = self.pca.explained_variance_ratio_
        log.info(
            f"PCA: {n_comp} componenti, "
            f"varianza spiegata = {var_explained.sum():.1%} "
            f"(PC1={var_explained[0]:.1%})"
        )
        return X_pca

    # Projects new data onto the already-fitted PCA (no refit).
    def _pca_transform(self, X_norm: np.ndarray) -> np.ndarray:
        return self.pca.transform(X_norm)

    # Fits MarkovRegression on PC1 (switching mean+variance) over n_restarts seeds; keeps the max llf.
    def _fit_single(self, pc1: np.ndarray):
        """
        Fit MarkovRegression on PC1 with switching mean + variance.
        Tries n_restarts starting values for robustness.
        """
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

        endog = pd.Series(pc1, name='pc1')

        best_result = None
        best_llf    = -np.inf

        saved_rng_state = np.random.get_state()
        try:
            for k in range(self.n_restarts):
                try:
                    np.random.seed(self.random_state + k)
                    mod = MarkovRegression(
                        endog,
                        k_regimes         = self.n_regimes,
                        trend             = 'c',
                        switching_variance = True,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = mod.fit(
                            maxiter     = self.n_iter,
                            disp        = False,
                            search_reps = 20,
                        )
                    if np.isfinite(res.llf) and res.llf > best_llf:
                        best_llf    = res.llf
                        best_result = res
                except Exception as e:
                    log.debug(f"  MarkovRegression restart {k} fallito: {e}")
                    continue
        finally:
            np.random.set_state(saved_rng_state)

        if best_result is not None:
            log.debug(f"  MarkovRegression fit: llf={best_llf:.1f}")

        return best_result

    # Extracts transitions, means and variances from the statsmodels result for the manual Hamilton filter.
    def _extract_params(self, result) -> dict:
        """
        Extract the parameters for the manual Hamilton filter.

        statsmodels convention: regime_transition[i,j] = P(S_t=i | S_{t-1}=j)
        (columns sum to 1).
        """
        k = self.n_regimes
        param_dict = dict(zip(result.model.param_names, result.params))

        trans = result.regime_transition
        if trans.ndim == 3:
            trans = trans[:, :, 0]

        means     = np.array([param_dict[f'const[{i}]']  for i in range(k)])
        variances = np.array([param_dict[f'sigma2[{i}]'] for i in range(k)])
        variances = np.maximum(variances, 1e-10)

        return {'trans': trans, 'means': means, 'variances': variances}

    # One Hamilton-filter step (predict + log-space emission + normalized update).
    def _hamilton_filter_step(self, y_t: float, params: dict,
                              prev_filtered: np.ndarray) -> np.ndarray:
        """
        Single step of the Hamilton (1989) filter.

        P(S_t=j | Y_{1:t}) ∝ f(y_t | S_t=j) · P(S_t=j | Y_{1:t-1})

        where P(S_t=j | Y_{1:t-1}) = Σ_i P(S_t=j | S_{t-1}=i) · P(S_{t-1}=i | Y_{1:t-1})

        Emission computed in log-space to avoid underflow on outliers.
        """
        trans = params['trans']

        # Prediction step: trans @ prev (columns sum to 1)
        predicted = trans @ prev_filtered
        predicted = np.maximum(predicted, 1e-20)

        # Emission in log-space (avoids underflow for |y_t - μ_j| >> σ_j)
        diff = y_t - params['means']
        var  = params['variances']
        log_emission = -0.5 * diff**2 / var - 0.5 * np.log(2.0 * np.pi * var)
        log_emission -= log_emission.max()
        emission = np.exp(log_emission)

        # Update step (the subtracted constant cancels out in the normalization)
        joint = emission * predicted
        total = joint.sum()
        if total < 1e-300:
            return np.full(self.n_regimes, 1.0 / self.n_regimes)
        return joint / total

    # Walk-forward regime labels: each retrain refits PCA+MS (sign-aligned), then Hamilton-filters until the next.
    def fit_predict_walkforward(
        self,
        df_macro:     pd.DataFrame,
        burn_in_days: int = 365,
        retrain_days: int = 90,
        _stop_at:     int | None = None,
    ) -> pd.DataFrame:
        """
        Walk-forward expanding window with Markov-Switching + PCA.

        For every day t generates regime probabilities using ONLY
        the data available up to t (no look-ahead).

        Pipeline for each retrain at t:
          1. PCA fit on X_norm[:t] (expanding window, sign-aligned)
          2. MarkovRegression on PC1[:t]
          3. Hamilton filter for the following days until the next retrain,
             transforming each x_t with the current PCA

        `_stop_at`: ONLY for the B7 bit-parity tests (default None = production
        behavior bit-unchanged): truncates the loop at the given bar leaving
        scaler and storage identical to the full run, so `continue_walkforward` can
        be compared bit-for-bit with the full run on the remaining bars.
        """
        if "open_time" in df_macro.columns:
            df_daily = df_macro.groupby(
                df_macro["open_time"].dt.date
            ).first().reset_index(drop=True)
        else:
            df_daily = df_macro.copy()

        X_raw, self.feature_cols = self._select_features(df_daily)
        n = len(X_raw)

        if n < burn_in_days + 10:
            raise ValueError(
                f"Troppo pochi dati ({n} giorni) per walk-forward "
                f"con burn_in={burn_in_days}."
            )

        # Global normalization (RobustScaler: location/scale statistics,
        # negligible look-ahead — median and IQR are stable over time)
        mask_valid = ~np.isnan(X_raw).any(axis=1)
        self.scaler.fit(X_raw[mask_valid])
        X_norm = np.clip(self.scaler.transform(
            np.nan_to_num(X_raw, nan=0.0)
        ), -5, 5)

        n_comp = min(self.n_pca, X_norm.shape[1], X_norm.shape[0])

        # Result storage
        probs_all = np.full((n, self.n_regimes), 1.0 / self.n_regimes)

        current_params  = None
        current_pca     = None
        pc1_cache       = None   # PC1 of all X_norm under current PCA (A6)
        prev_components = None
        last_filtered   = np.full(self.n_regimes, 1.0 / self.n_regimes)
        last_retrain    = -1

        log.info(
            f"Walk-forward MarkovSwitching: {n} giorni, "
            f"burn_in={burn_in_days}, retrain ogni {retrain_days} gg, "
            f"{self.n_regimes} regimi, PCA expanding window ..."
        )

        # n_eff = loop end (B7: _stop_at truncates for the golden test; None = n).
        n_eff = n if _stop_at is None else min(int(_stop_at), n)

        # Counters for the anti-silent-degradation guard. Previously EVERY fit
        # failure produced one log.warning per timestep and the loop carried on:
        # with current_params=None, probs_all[t] was never written and the
        # walk-forward returned the UNIFORM PRIOR instead of filtered
        # probabilities, without anything failing. `n_filtered` measures actual
        # coverage (timesteps with genuinely computed probabilities), which is the
        # quantity to check: failed fits are the cause, coverage is the effect.
        # The counters do NOT touch the numeric path: with successful fits the
        # output is bit-identical (B7 bit-parity golden test unchanged).
        n_fit_attempts = 0
        n_fit_ok       = 0
        n_filtered     = 0
        n_steps        = max(0, n_eff - burn_in_days)

        for t in range(burn_in_days, n_eff):
            if current_params is None or (t - last_retrain) >= retrain_days:
                if t >= 50:
                    n_fit_attempts += 1
                    try:
                        # PCA expanding window: fit only on data[:t]
                        pca_t = PCA(n_components=n_comp,
                                    random_state=self.random_state)
                        pca_t.fit(X_norm[:t])

                        # Sign alignment: PC1 must point in the same
                        # direction from one retrain to the next, otherwise the
                        # Hamilton filter receives an inverted signal
                        if prev_components is not None:
                            if np.dot(pca_t.components_[0],
                                      prev_components[0]) < 0:
                                pca_t.components_[0] *= -1
                        prev_components = pca_t.components_.copy()
                        current_pca = pca_t
                        # project ALL X_norm once per retrain (A6): between retrains the PCA is fixed
                        # → pc1_cache[t] == transform(X_norm[t:t+1]) bit-identical, but avoids ~n
                        # row-by-row sklearn calls. CAUSAL: only [t] is indexed, PCA fit on [:t].
                        pc1_cache = current_pca.transform(X_norm)[:, 0]

                        pc1_train = pc1_cache[:t]

                        result = self._fit_single(pc1_train)
                        if result is not None:
                            n_fit_ok += 1
                            current_params = self._extract_params(result)
                            fmp = result.filtered_marginal_probabilities
                            if isinstance(fmp, pd.DataFrame):
                                last_filtered = fmp.values[-1].copy()
                            else:
                                last_filtered = fmp[-1].copy()
                            last_retrain = t
                            log.info(
                                f"  Retrain a t={t}/{n}: "
                                f"llf={result.llf:.1f}, "
                                f"PCA var={pca_t.explained_variance_ratio_[0]:.1%}"
                            )
                        else:
                            # _fit_single returns None when ALL n_restarts fail:
                            # this used to be a silent branch (no logging at all).
                            log.warning(
                                f"  MarkovSwitching: nessun restart converge a t={t} "
                                f"(su {self.n_restarts} tentativi)"
                            )
                    except Exception as e:
                        log.warning(
                            f"  MarkovSwitching fit fallito a t={t}: {e}"
                        )

            if current_params is not None and current_pca is not None:
                pc1_t = pc1_cache[t]   # O(1) cache lookup (A6)
                last_filtered = self._hamilton_filter_step(
                    pc1_t, current_params, last_filtered
                )
                probs_all[t] = last_filtered
                n_filtered += 1

        # ── Anti-silent-degradation guard ─────────────────────────────────────
        # Two abort conditions, deliberately differing in severity.
        # ① ZERO successful fits → the result is the uniform prior dressed up as
        #    regime probabilities: always fatal, not disableable. This is the case
        #    that shows up with statsmodels missing or broken.
        # ② Too many failed fits (> max_fit_failure_ratio) → the walk-forward runs
        #    on stale parameters for long stretches: partial degradation,
        #    configurable threshold because on short series some failures are
        #    physiological.
        # Diagnostics are always persisted, even when not aborting: a degraded
        # rebuild must leave a trace readable downstream.
        fail_ratio = (
            0.0 if n_fit_attempts == 0
            else (n_fit_attempts - n_fit_ok) / n_fit_attempts
        )
        coverage = 0.0 if n_steps == 0 else n_filtered / n_steps
        self.last_fit_diagnostics = {
            "fit_attempts": n_fit_attempts,
            "fit_ok":       n_fit_ok,
            "fail_ratio":   fail_ratio,
            "coverage":     coverage,
            "n_steps":      n_steps,
        }
        log.info(
            f"Walk-forward completato: fit {n_fit_ok}/{n_fit_attempts} riusciti, "
            f"copertura probabilità filtrate {coverage:.1%} ({n_filtered}/{n_steps})"
        )
        if n_fit_attempts > 0 and n_fit_ok == 0:
            raise RuntimeError(
                f"RegimeMarkovSwitching: NESSUN fit riuscito su {n_fit_attempts} "
                f"tentativi — le probabilità restituite sarebbero la prior uniforme "
                f"(1/{self.n_regimes}), non regimi stimati. Cause tipiche: statsmodels "
                f"assente o incompatibile, serie degenere, n_regimes troppo alto per "
                f"i dati. Controllare i warning 'MarkovSwitching fit fallito' sopra. / "
                f"NO successful fit out of {n_fit_attempts} attempts — the returned "
                f"probabilities would be the uniform prior, not estimated regimes."
            )
        if n_fit_attempts > 0 and fail_ratio > self.max_fit_failure_ratio:
            raise RuntimeError(
                f"RegimeMarkovSwitching: {n_fit_attempts - n_fit_ok}/{n_fit_attempts} "
                f"fit falliti (rapporto {fail_ratio:.1%} > soglia "
                f"{self.max_fit_failure_ratio:.1%}) — il walk-forward avrebbe girato "
                f"su parametri stantii per larga parte della storia. Alzare "
                f"max_fit_failure_ratio solo con una ragione esplicita. / "
                f"{n_fit_attempts - n_fit_ok}/{n_fit_attempts} fits failed (ratio "
                f"{fail_ratio:.1%} > threshold {self.max_fit_failure_ratio:.1%})."
            )

        # B7 — chain-state stash at loop end (BEFORE the final fit, which overwrites
        # self.pca/self.model with the full-sample fit): everything that
        # continue_walkforward needs to resume appending without redoing history.
        # Defensive copies: the final fit and consumers must not mutate the stash.
        self._wf_state = {
            "params":          None if current_params is None else {
                k: np.array(v, copy=True) for k, v in current_params.items()
            },
            "pca":             current_pca,
            "prev_components": None if prev_components is None
                               else prev_components.copy(),
            "last_filtered":   last_filtered.copy(),
            "last_retrain":    last_retrain,
            "n_bars":          n_eff,
            "burn_in_bars":    burn_in_days,
            "retrain_bars":    retrain_days,
        }

        # Final fit on the whole history (for production/predict_proba)
        self._pca_fit_transform(X_norm)
        if prev_components is not None:
            if np.dot(self.pca.components_[0], prev_components[0]) < 0:
                self.pca.components_[0] *= -1
        pc1_full = self.pca.transform(X_norm)[:, 0]

        if current_params is not None:
            self._params_cache = current_params
            final_result = self._fit_single(pc1_full)
            if final_result is not None:
                self.model = final_result
                self._params_cache = self._extract_params(self.model)

        # Result DataFrame
        result_df = pd.DataFrame(
            probs_all,
            index   = df_daily.index,
            columns = [f"regime_prob_{i}" for i in range(self.n_regimes)],
        )
        result_df["regime_dominant"] = probs_all.argmax(axis=1)
        result_df["regime_burn_in"]  = False
        result_df.iloc[:burn_in_days, -1] = True

        log.info(
            f"Walk-forward completato. "
            f"Burn-in: {burn_in_days} giorni. "
            f"Classificati: {n - burn_in_days} giorni."
        )
        self._describe_regimes_wf(X_raw, probs_all, burn_in_days)
        return result_df

    # Diagnostics: logs per-regime means + the estimated μ/σ²/P(stay) parameters.
    def _describe_regimes_wf(self, X_raw, probs_all, burn_in_days):
        labels = probs_all[burn_in_days:].argmax(axis=1)
        X_post = X_raw[burn_in_days:]
        log.info("─── Caratteristiche regimi MarkovSwitching (post burn-in) ─")
        for r in range(self.n_regimes):
            mask = labels == r
            if mask.sum() == 0:
                continue
            means = X_post[mask].mean(axis=0)
            feat_str = "  ".join(
                f"{self.feature_cols[i].replace('macro_','')[:12]}={means[i]:.2f}"
                for i in range(min(4, len(self.feature_cols)))
            )
            log.info(
                f"  Regime {r} ({mask.sum()} gg, {mask.mean():.0%}): "
                f"{feat_str}"
            )
        if self.model is not None:
            params = self._extract_params(self.model)
            log.info("─── Parametri Markov-Switching ─────────────────────────")
            for r in range(self.n_regimes):
                log.info(
                    f"  Regime {r}: μ={params['means'][r]:+.3f}, "
                    f"σ²={params['variances'][r]:.3f}, "
                    f"P(stay)={params['trans'][r,r]:.2%}"
                )

    # B7 — incremental walk-forward continuation from a persisted chain state.
    def continue_walkforward(self, df_macro: pd.DataFrame,
                             state: dict) -> pd.DataFrame:
        """
        Extend the walk-forward to the new bars only (append), resuming from the
        chain state `state` (format of `self._wf_state`), WITHOUT redoing the
        historical refits. Bit-parity guaranteed against the **original run extended
        with a FROZEN scaler** (which is what the golden test measures via `_stop_at`):
        a full rebuild over the extended span would refit the global RobustScaler
        → different X_norm on all bars → slight divergence, expected and
        documented (re-anchoring via periodic full rebuild). The frozen
        scaler is strictly causal on the new bars (more so than the rebuild's
        global fit): no lookahead.

        ⚠ MIRROR of the retrain+filter block of `fit_predict_walkforward`: any
        change there must be replicated here (golden test in tests/test_regime_incremental.py
        fails on divergence). The per-bar Hamilton step is the SAME
        method (`_hamilton_filter_step`) — do-not-touch zone, no copy.

        Args:
            df_macro: FULL history (old + new bars), same feature construction
                      as the full run. Needed in full: retrains are
                      expanding-window over [:t].
            state:    chain state (keys of `_wf_state`); UPDATED
                      in place at the end of the run (n_bars/last_* /params/pca) for
                      the checkpoint re-save.

        Returns:
            DataFrame with ONLY the new rows (positional [n_old:n)) and columns
            regime_prob_i / regime_dominant / regime_burn_in=False.
        """
        # same pre-processing as the full run, but with columns and scaler FROZEN
        # from the state (no refit: this is what makes the append deterministic).
        if "open_time" in df_macro.columns:
            df_daily = df_macro.groupby(
                df_macro["open_time"].dt.date
            ).first().reset_index(drop=True)
        else:
            df_daily = df_macro

        missing = [c for c in self.feature_cols if c not in df_daily.columns]
        if missing:
            raise RuntimeError(
                f"continue_walkforward: colonne mancanti nel df nuovo: {missing}"
            )
        X_raw = df_daily[self.feature_cols].values
        n     = len(X_raw)
        n_old = int(state["n_bars"])
        if n < n_old:
            raise RuntimeError(
                f"continue_walkforward: storia nuova ({n} barre) più corta dello "
                f"stato persistito ({n_old}): dati troncati o checkpoint stale."
            )
        if state["params"] is None or state["pca"] is None:
            raise RuntimeError(
                "continue_walkforward: stato senza parametri/PCA (walk-forward "
                "originale mai riuscito a fittare): serve un full rebuild."
            )
        X_norm = np.clip(self.scaler.transform(
            np.nan_to_num(X_raw, nan=0.0)
        ), -5, 5)

        retrain_days    = int(state["retrain_bars"])
        current_params  = state["params"]
        current_pca     = state["pca"]
        prev_components = state["prev_components"]
        last_filtered   = np.array(state["last_filtered"], copy=True)
        last_retrain    = int(state["last_retrain"])

        # PC1 projection of the whole history under the current PCA (same A6
        # cache pattern as the full run: causal, only [t] is indexed).
        pc1_cache = current_pca.transform(X_norm)[:, 0]

        probs_new = np.full((n - n_old, self.n_regimes), 1.0 / self.n_regimes)
        n_retrains = 0

        log.info(
            f"Walk-forward INCREMENTALE: {n - n_old} barre nuove "
            f"(da {n_old} a {n}), ultimo retrain a t={last_retrain}, "
            f"prossimo a t={last_retrain + retrain_days}"
        )

        # MIRROR of the full-run counters. Degradation here is milder but more
        # insidious: current_params is never None, so a failed retrain does NOT
        # produce the uniform prior — it produces filtered probabilities from
        # STALE parameters, indistinguishable downstream from good ones. Counting
        # them is the only way to notice.
        n_fit_attempts = 0

        for t in range(n_old, n):
            # ── MIRROR of the full-run retrain block (difference: current_params
            #    is never None, verified above)
            if (t - last_retrain) >= retrain_days:
                if t >= 50:
                    n_fit_attempts += 1
                    try:
                        pca_t = PCA(n_components=current_pca.n_components_,
                                    random_state=self.random_state)
                        pca_t.fit(X_norm[:t])
                        if prev_components is not None:
                            if np.dot(pca_t.components_[0],
                                      prev_components[0]) < 0:
                                pca_t.components_[0] *= -1
                        prev_components = pca_t.components_.copy()
                        current_pca = pca_t
                        pc1_cache = current_pca.transform(X_norm)[:, 0]

                        result = self._fit_single(pc1_cache[:t])
                        if result is not None:
                            current_params = self._extract_params(result)
                            fmp = result.filtered_marginal_probabilities
                            if isinstance(fmp, pd.DataFrame):
                                last_filtered = fmp.values[-1].copy()
                            else:
                                last_filtered = fmp[-1].copy()
                            last_retrain = t
                            n_retrains += 1
                            log.info(
                                f"  Retrain (incrementale) a t={t}/{n}: "
                                f"llf={result.llf:.1f}, "
                                f"PCA var={pca_t.explained_variance_ratio_[0]:.1%}"
                            )
                        else:
                            log.warning(
                                f"  MarkovSwitching: nessun restart converge a t={t} "
                                f"(su {self.n_restarts} tentativi) — l'append "
                                f"prosegue su parametri STANTII"
                            )
                    except Exception as e:
                        log.warning(
                            f"  MarkovSwitching fit fallito a t={t}: {e}"
                        )

            last_filtered = self._hamilton_filter_step(
                pc1_cache[t], current_params, last_filtered
            )
            probs_new[t - n_old] = last_filtered

        # MIRROR of the full-run guard. Zero ATTEMPTED retrains is legitimate (the
        # append may not cross a refit boundary): not an error. Zero successful out
        # of ≥1 attempted means the append crossed a refit boundary and carried on
        # with parameters frozen from the checkpoint — silently.
        self.last_fit_diagnostics = {
            "fit_attempts": n_fit_attempts,
            "fit_ok":       n_retrains,
            "fail_ratio":   (0.0 if n_fit_attempts == 0
                             else (n_fit_attempts - n_retrains) / n_fit_attempts),
            "incremental":  True,
        }
        if n_fit_attempts > 0 and n_retrains == 0:
            raise RuntimeError(
                f"continue_walkforward: {n_fit_attempts} retrain dovuti, NESSUNO "
                f"riuscito — l'append avrebbe prodotto probabilità filtrate con i "
                f"parametri congelati nel checkpoint, indistinguibili da quelle "
                f"valide. Controllare i warning sopra (statsmodels assente o serie "
                f"degenere) e rifare un full rebuild se necessario. / "
                f"{n_fit_attempts} retrains due, NONE succeeded — the append would "
                f"have produced filtered probabilities from checkpoint-frozen "
                f"parameters, indistinguishable from valid ones."
            )
        # second mirror branch — the full run also aborts on an excessive
        # failure ratio, not only on zero successes. Without this, an append
        # with a long backlog (months of pause → several retrain boundaries
        # crossed) failing 3 refits out of 4 would pass silently, while the
        # equivalent full run would have aborted: same degradation, different
        # threshold depending on which method was called. With
        # n_fit_attempts==1 this branch coincides with the previous one (no
        # behaviour change in the common case: weekly incremental cadence
        # against a 90-day retrain ⇒ 0 or 1 attempt).
        _fail_ratio = self.last_fit_diagnostics["fail_ratio"]
        if n_fit_attempts > 0 and _fail_ratio > self.max_fit_failure_ratio:
            raise RuntimeError(
                f"continue_walkforward: {n_fit_attempts - n_retrains}/"
                f"{n_fit_attempts} refit falliti (rapporto {_fail_ratio:.1%} > "
                f"soglia {self.max_fit_failure_ratio:.1%}) — l'append avrebbe "
                f"girato su parametri stantii per larga parte dello span nuovo. / "
                f"{n_fit_attempts - n_retrains}/{n_fit_attempts} refits failed "
                f"(ratio {_fail_ratio:.1%} > threshold "
                f"{self.max_fit_failure_ratio:.1%})."
            )

        # update the state in place for the checkpoint re-save (append succeeded).
        state.update({
            "params":          {k: np.array(v, copy=True)
                                for k, v in current_params.items()},
            "pca":             current_pca,
            "prev_components": None if prev_components is None
                               else prev_components.copy(),
            "last_filtered":   last_filtered.copy(),
            "last_retrain":    last_retrain,
            "n_bars":          n,
        })

        result_df = pd.DataFrame(
            probs_new,
            index   = df_daily.index[n_old:],
            columns = [f"regime_prob_{i}" for i in range(self.n_regimes)],
        )
        result_df["regime_dominant"] = probs_new.argmax(axis=1)
        # new bars are always post burn-in (burn-in lives in the original run).
        result_df["regime_burn_in"]  = False

        log.info(
            f"Incrementale completato: {n - n_old} barre, {n_retrains} retrain."
        )
        return result_df

    # Final fit on the full history (PCA+MS) for production/live; populates the param cache.
    def fit(self, df_macro: pd.DataFrame) -> "RegimeMarkovSwitching":
        """
        Final training on the whole history (production/live).
        """
        if "open_time" in df_macro.columns:
            df_daily = df_macro.groupby(
                df_macro["open_time"].dt.date
            ).first().reset_index(drop=True)
        else:
            df_daily = df_macro

        X_raw, self.feature_cols = self._select_features(df_daily)
        mask = ~np.isnan(X_raw).any(axis=1)
        X_raw = X_raw[mask]

        if len(X_raw) < 50:
            raise ValueError(f"Troppo pochi dati ({len(X_raw)} giorni).")

        X_norm = np.clip(self.scaler.fit_transform(X_raw), -5, 5)
        X_pca  = self._pca_fit_transform(X_norm)
        pc1    = X_pca[:, 0]

        log.info(
            f"MarkovSwitching fit finale: "
            f"{self.n_regimes} regimi, {len(pc1)} osservazioni"
        )
        self.model = self._fit_single(pc1)
        if self.model is None:
            raise RuntimeError(
                "MarkovRegression fit fallito per tutti i restart. "
                "Dati con varianza zero o troppo pochi campioni."
            )
        self._params_cache = self._extract_params(self.model)
        return self

    # Regime probabilities by applying the Hamilton filter sequentially, row by row.
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime probabilities with the Hamilton filter.

        Applies one filter step per row sequentially,
        so the probabilities account for the preceding history.
        """
        if self.model is None:
            raise RuntimeError("MarkovSwitching non addestrato.")
        if self._params_cache is None:
            self._params_cache = self._extract_params(self.model)

        X_raw = df[self.feature_cols].values if self.feature_cols else df.values
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        X_norm = np.clip(self.scaler.transform(X_raw), -5, 5)
        X_pca  = self._pca_transform(X_norm)
        pc1    = X_pca[:, 0]

        n = len(pc1)
        probs = np.zeros((n, self.n_regimes))
        filtered = np.full(self.n_regimes, 1.0 / self.n_regimes)

        for t in range(n):
            filtered = self._hamilton_filter_step(
                pc1[t], self._params_cache, filtered
            )
            probs[t] = filtered

        return probs

    # Serializes model + PCA + scaler + param cache to disk (pickle).
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model":        self.model,
                "pca":          self.pca,
                "scaler":       self.scaler,
                "feature_cols": self.feature_cols,
                "n_regimes":    self.n_regimes,
                "n_pca":        self.n_pca,
                "params_cache": self._params_cache,
            }, f)
        log.info(f"MarkovSwitching salvato → {path}")

    # Reconstructs a RegimeMarkovSwitching from a saved pickle.
    @classmethod
    def load(cls, path: str) -> "RegimeMarkovSwitching":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            n_regimes=data["n_regimes"],
            n_pca=data.get("n_pca", 3),
        )
        obj.model         = data["model"]
        obj.pca           = data["pca"]
        obj.scaler        = data["scaler"]
        obj.feature_cols  = data["feature_cols"]
        obj._params_cache = data.get("params_cache")
        return obj

    # Compares candidate regime counts k via BIC and returns {k: BIC}, logging the winner.
    def select_n_regimes(
        self, df_macro: pd.DataFrame,
        candidates: list[int] = [2, 3, 4],
    ) -> dict:
        """
        Select the optimal number of regimes via BIC.

        Fits the model for each k in candidates and returns
        {k: BIC}, logging the winner.
        """
        if "open_time" in df_macro.columns:
            df_daily = df_macro.groupby(
                df_macro["open_time"].dt.date
            ).first().reset_index(drop=True)
        else:
            df_daily = df_macro

        X_raw, _ = self._select_features(df_daily)
        mask = ~np.isnan(X_raw).any(axis=1)
        X_raw = X_raw[mask]
        X_norm = np.clip(RobustScaler().fit_transform(X_raw), -5, 5)

        n_comp = min(self.n_pca, X_norm.shape[1], X_norm.shape[0])
        pca = PCA(n_components=n_comp, random_state=self.random_state)
        pc1 = pca.fit_transform(X_norm)[:, 0]

        results = {}
        for k in candidates:
            try:
                old_k = self.n_regimes
                self.n_regimes = k
                res = self._fit_single(pc1)
                self.n_regimes = old_k
                if res is not None:
                    results[k] = res.bic
                    log.info(f"  k={k}: BIC={res.bic:.1f}, llf={res.llf:.1f}")
            except Exception as e:
                log.warning(f"  k={k}: fallito ({e})")

        if results:
            best_k = min(results, key=results.get)
            log.info(f"BIC selection: best k={best_k} (BIC={results[best_k]:.1f})")
        return results


# ─── STAGE 1c: SESSION-BASED REGIME DETECTOR (Asia/EU/US) ───────────────────

# Intraday regime based on trading session (Asia/EU/US) — replaces the
# degenerate macro Markov-Switching by aligning the regime timescale with
# the trading horizon (1m, h=30).
class RegimeSession:
    """
    Deterministic "session-based" regime detector, drop-in for
    `RegimeMarkovSwitching`.

    Mapping rule (UTC):
        regime 0 = Asia       [00:00, 08:00)
        regime 1 = EU/London  [08:00, 16:00)
        regime 2 = US         [16:00, 24:00)
    equivalent to `regime = hour_utc // 8`.

    Advantages over the macro Markov-Switching:
      · Timescale consistent with the trading horizon (1-min, h=30)
      · Always 3 well-balanced clusters (~33% each) — no collapse
      · No dependence on EM/convergence/look-ahead
      · Zero parameters to fit → reproducible and robust

    Interface notes:
      Returns the same schema as the Markov-Switching
      (`regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`)
      so that consumers (02_train.py, dashboard) need not be touched.
    """

    # Minimal config — no actual parameters to learn.
    def __init__(self, n_regimes: int = 3):
        # we force n_regimes=3 (Asia/EU/US); we accept the kwarg only for
        # signature compatibility with RegimeMarkovSwitching.
        if n_regimes != 3:
            log.warning(
                f"RegimeSession: n_regimes={n_regimes} ignorato, forzato a 3 "
                f"(Asia/EU/US sono fisse)."
            )
        self.n_regimes = 3
        # placeholder fields for drop-in pickle compatibility with MS.
        self.model = None
        self.pca = None
        self.scaler = None
        self.feature_cols: list[str] = []

    # Builds the hourly UTC index over df_macro's range and computes regimes.
    def fit_predict_walkforward(
        self,
        df_macro: pd.DataFrame,
        burn_in_days: int = 0,   # unused — accepted for compat
        retrain_days: int = 0,   # unused — accepted for compat
        **kwargs,                # absorbs any future kwargs
    ) -> pd.DataFrame:
        """
        Compute session-based regimes over a UTC hourly range.

        Args:
            df_macro:     used ONLY to determine the time range
                          (min/max of the index). Its content is not read.
            burn_in_days: ignored (no fit, no burn-in needed).
            retrain_days: ignored (deterministic, no retrain).

        Returns:
            DataFrame indexed on UTC hourly timestamps with columns:
              · regime_dominant (int 0/1/2)
              · regime_burn_in  (bool, always False)
              · regime_prob_0   (float, one-hot)
              · regime_prob_1   (float, one-hot)
              · regime_prob_2   (float, one-hot)
        """
        # extracts the time range from df_macro (tolerates non-tz / empty index).
        if df_macro is None or len(df_macro) == 0:
            raise ValueError("RegimeSession: df_macro vuoto, impossibile derivare il range.")

        idx_raw = pd.to_datetime(df_macro.index)
        # normalize to UTC (if naive, assume UTC; if tz-aware, convert).
        if idx_raw.tz is None:
            idx_utc = idx_raw.tz_localize("UTC")
        else:
            idx_utc = idx_raw.tz_convert("UTC")

        t_min = idx_utc.min().floor("h")
        t_max = idx_utc.max().ceil("h")

        # hourly UTC index spanning the full range [t_min, t_max].
        hourly_idx = pd.date_range(start=t_min, end=t_max, freq="h", tz="UTC")

        # hour → regime mapping (0=Asia, 1=EU, 2=US).
        hours = hourly_idx.hour.to_numpy()
        regime_dominant = (hours // 8).astype(np.int64)

        # one-hot probabilities to match the MS schema.
        n = len(hourly_idx)
        prob_0 = (regime_dominant == 0).astype(np.float32)
        prob_1 = (regime_dominant == 1).astype(np.float32)
        prob_2 = (regime_dominant == 2).astype(np.float32)

        out = pd.DataFrame(
            {
                "regime_dominant": regime_dominant,
                "regime_burn_in": np.zeros(n, dtype=bool),
                "regime_prob_0": prob_0,
                "regime_prob_1": prob_1,
                "regime_prob_2": prob_2,
            },
            index=hourly_idx,
        )

        log.info(
            f"RegimeSession: generati {n} timestamp orari "
            f"({t_min} → {t_max}), distribuzione "
            f"{{0: {int(prob_0.sum())}, 1: {int(prob_1.sum())}, 2: {int(prob_2.sum())}}}"
        )
        return out

    # Pickles the few attributes (no fitted parameters).
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_regimes": self.n_regimes,
                    "kind": "session",
                    "model": self.model,
                    "pca": self.pca,
                    "scaler": self.scaler,
                    "feature_cols": self.feature_cols,
                },
                f,
            )
        log.info(f"RegimeSession salvato → {path}")

    # Reconstructs a RegimeSession from a pickle.
    @classmethod
    def load(cls, path: str) -> "RegimeSession":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(n_regimes=data.get("n_regimes", 3))
        obj.model = data.get("model")
        obj.pca = data.get("pca")
        obj.scaler = data.get("scaler")
        obj.feature_cols = data.get("feature_cols", [])
        return obj


# ─── STAGE 1d: MARKOV-SWITCHING ON BTC REALIZED VOL (intraday) ──────────────

# Markov-Switching on hourly BTC realized volatility — "Variant 3" (2026-06-03 decision, see THEORY.md §regime).
# Replaces RegimeSession by aligning the regime timescale (switches every 3-8h)
# with the trading timeframe. The regime clock is HOURLY by design, regardless of the
# candle interval (≤1h, aggregated to 1h). Uses BTC data ONLY, no more US macro.
class RegimeMarkovBTC:
    """
    Markov-Switching (Hamilton 1989) on BTC intraday realized volatility.

    Pipeline:
      1. Load BTC candles at any interval ≤1h from `data/raw_candles.parquet`
         (with 1h input the resample is an identity; input >1h → ValueError fail-fast)
      2. Aggregate to 1 hour: log_ret_h (sum of log_ret per bucket) + log_rv (log of the
         realized variance = log(Σ log_ret²) clipped for stability)
      3. Global RobustScaler (median/IQR, negligible look-ahead)
      4. Walk-forward expanding window with `RegimeMarkovSwitching` as the engine:
         · PCA(n_pca=1) combines log_ret_h + log_rv into a single signal
         · MarkovRegression with switching mean + variance on PC1
         · O(1) Hamilton filter between retrains

    Rationale:
      - "Variant 3" of the 2026-06-03 decision (replace the macro MS with an
        intraday BTC regime). The macro MS was degenerate (monthly regimes
        vs 1m trading); the session-based one was informationally empty. Hourly BTC
        realized vol changes 3-8 times/day → matches the forecast horizon h=30.

    Drop-in interface with RegimeSession / RegimeMarkovSwitching:
      - `fit_predict_walkforward(df_macro=...)` accepts but IGNORES df_macro.
      - Returns a DataFrame with a UTC hourly index and columns
        `regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`.
    """

    # Configures the detector; the MS engine is composed (not subclassed) for clean reuse.
    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 300,
        random_state: int = 42,
        n_restarts: int = 5,
        candles_path: str = "data/raw_candles.parquet",
    ):
        if n_regimes < 2:
            raise ValueError(f"n_regimes deve essere >= 2, ricevuto {n_regimes}")
        self.n_regimes = n_regimes
        self.candles_path = candles_path
        # n_pca=1 → PCA reduces (log_ret_h, log_rv) to a single informative direction.
        self._engine = RegimeMarkovSwitching(
            n_regimes=n_regimes,
            n_iter=n_iter,
            random_state=random_state,
            n_pca=1,
            n_restarts=n_restarts,
        )
        # mirror fields for save/load drop-in with consumers.
        self.model = None
        self.pca = None
        self.scaler = None
        self.feature_cols: list[str] = []

    # Aggregates candles at any interval ≤1h into hourly features (log-return + log RV).
    def _build_btc_hourly_df(self) -> pd.DataFrame:
        """
        Load `raw_candles.parquet` (candles at any interval ≤1h; with 1h input
        the hourly resample is an identity) and produce a UTC hourly DataFrame with:
          · log_ret_h: sum of the bucket's log-returns (hourly return)
          · log_rv   : log of the hourly realized variance = log(Σ log_ret²)

        With 1h input each bucket contains ONE observation only → rv = log_ret² of the
        single bar: a poor but valid RV proxy; the 1e-12 clip avoids systematic log(0)
        on hours with no variation. Input >1h → ValueError (fail-fast:
        the regime clock is hourly by design and cannot be reconstructed).

        The log transform on `rv` is essential: realized variance is strongly
        right-skewed → without the log, the MarkovRegression collapses on outliers.
        """
        from pathlib import Path as _Path
        path = _Path(self.candles_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RegimeMarkovBTC: {path} non trovato. "
                f"Esegui prima `python scripts/01_download_data.py`."
            )

        # read + UTC time-index normalization.
        candles = pd.read_parquet(path, columns=["open_time", "close"])
        # pd.api.types handles both naive and tz-aware datetime (np.issubdtype doesn't).
        if not pd.api.types.is_datetime64_any_dtype(candles["open_time"]):
            candles["open_time"] = pd.to_datetime(
                candles["open_time"], unit="ms", utc=True,
            )
        candles = candles.sort_values("open_time").set_index("open_time")
        if candles.index.tz is None:
            candles.index = candles.index.tz_localize("UTC")
        else:
            candles.index = candles.index.tz_convert("UTC")

        # fail-fast if the data interval is >1h: the regime clock is HOURLY by design
        # (Markov on hourly realized vol) and >1h bars would yield empty buckets /
        # degenerate RV under resample("1h"). Step inferred from the MEDIAN diff.
        _step = candles.index.to_series().diff().median()
        if pd.notna(_step) and _step > pd.Timedelta("1h"):
            raise ValueError(
                f"RegimeMarkovBTC: intervallo candele inferito {_step} > 1h — "
                f"il regime detector richiede candele a intervallo ≤1h "
                f"(aggregate a 1h; con input 1h il resample è identità)."
            )

        # per-bar log-return (close-to-close, at any interval ≤1h);
        # inf/NaN dropped downstream.
        log_ret = np.log(candles["close"]).diff()
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan)

        # hourly aggregation — sum of log-returns + sum of squares (realized var).
        # With 1h input the resample is an identity: rv = the single bar's log_ret²
        # (a poor but valid proxy of hourly RV).
        log_ret_h = log_ret.resample("1h").sum()
        rv = log_ret.pow(2).resample("1h").sum()
        # clip at 1e-12 to avoid log(0) on hours with no variation (or, with 1h
        # input, on bars with unchanged close).
        log_rv = np.log(rv.clip(lower=1e-12))

        out = pd.DataFrame({"log_ret_h": log_ret_h, "log_rv": log_rv})
        out = out.dropna()
        log.info(
            f"RegimeMarkovBTC: aggregate {len(out)} ore "
            f"({out.index.min()} → {out.index.max()}); "
            f"log_rv mean={out['log_rv'].mean():.2f} std={out['log_rv'].std():.2f}"
        )
        return out

    # Walk-forward on the BTC df; df_macro is ignored (drop-in interface).
    def fit_predict_walkforward(
        self,
        df_macro: pd.DataFrame = None,
        burn_in_days: int = 30,
        retrain_days: int = 30,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Walk-forward expanding window with Markov-Switching on BTC realized vol.

        Args:
            df_macro:     ignored (used ONLY for signature compatibility).
            burn_in_days: converted to hours (×24). Default 30d = 720h.
            retrain_days: converted to hours (×24). Default 30d = 720h.

        Returns:
            DataFrame indexed on UTC hours with columns:
              · regime_dominant   (int 0..n_regimes-1)
              · regime_burn_in    (bool)
              · regime_prob_{i}   (float, sum=1 per row)
        """
        df_btc = self._build_btc_hourly_df()

        # the MS engine reasons in "days" as positional units; here a "step"
        # is one hour — convert burn_in and retrain to hours.
        burn_in_h = burn_in_days * 24
        retrain_h = retrain_days * 24

        log.info(
            f"RegimeMarkovBTC walk-forward: {len(df_btc)} ore "
            f"({len(df_btc)/24:.0f} gg), burn_in={burn_in_h}h "
            f"({burn_in_days}gg), retrain ogni {retrain_h}h "
            f"({retrain_days}gg), {self.n_regimes} regimi"
        )

        result = self._engine.fit_predict_walkforward(
            df_btc,
            burn_in_days=burn_in_h,
            retrain_days=retrain_h,
        )

        # the engine resets the index to positional 0..N-1; restore UTC hourly index.
        result.index = df_btc.index

        # mirror engine fields for save/load consumers.
        self.model = self._engine.model
        self.pca = self._engine.pca
        self.scaler = self._engine.scaler
        self.feature_cols = self._engine.feature_cols

        # final post-burn-in distribution log for quick diagnostics.
        post = result.iloc[burn_in_h:]
        counts = post["regime_dominant"].value_counts().sort_index()
        log.info("─── Distribuzione regimi BTC (post burn-in) ───")
        for r, c in counts.items():
            pct = c / len(post) * 100
            log.info(f"  Regime {r}: {c} ore ({pct:.1f}%)")

        return result

    # Delegates pickling to the MS engine (same schema, drop-in MS loader).
    def save(self, path: str) -> None:
        self._engine.save(path)
        log.info(f"RegimeMarkovBTC salvato → {path}  (schema MS engine)")

    # Reconstructs a RegimeMarkovBTC by reusing the MS engine loader.
    @classmethod
    def load(cls, path: str) -> "RegimeMarkovBTC":
        engine = RegimeMarkovSwitching.load(path)
        obj = cls(n_regimes=engine.n_regimes)
        obj._engine = engine
        obj.model = engine.model
        obj.pca = engine.pca
        obj.scaler = engine.scaler
        obj.feature_cols = engine.feature_cols
        return obj

    # ── B7 — incremental walk-forward checkpoint ─────────────────────────────
    # The checkpoint persists the walk-forward CHAIN (last-retrain params,
    # sign-aligned PCA, filtered posterior, cadence, frozen scaler): what the
    # production pkl does NOT contain (that one only has the final full-sample
    # fit for predict_proba). With the checkpoint, extending the parquet costs
    # 0-1 MLE fits instead of ~30 (minutes vs hours).

    _WF_CKPT_SCHEMA = 1

    # Atomic checkpoint write (the safety-net .tmp + os.replace pattern).
    @staticmethod
    def save_wf_checkpoint(ckpt: dict, path: str) -> None:
        import os
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(ckpt, f)
        os.replace(tmp, str(path))
        log.info(
            f"Checkpoint walk-forward → {path} "
            f"(n_bars={ckpt['chain']['n_bars']}, "
            f"last_retrain={ckpt['chain']['last_retrain']}, "
            f"last_ts={ckpt['last_timestamp']})"
        )

    # Builds the checkpoint dict from the engine state after a run (full or incremental).
    def build_wf_checkpoint(self, chain: dict,
                            last_timestamp: pd.Timestamp) -> dict:
        eng = self._engine
        return {
            "schema_version": self._WF_CKPT_SCHEMA,
            "chain":          chain,
            "scaler":         eng.scaler,
            "feature_cols":   eng.feature_cols,
            "n_regimes":      eng.n_regimes,
            "n_pca":          eng.n_pca,
            "n_iter":         eng.n_iter,
            "n_restarts":     eng.n_restarts,
            "random_state":   eng.random_state,
            "last_timestamp": last_timestamp,
        }

    # Incremental continuation: fresh candles + checkpoint → NEW rows only.
    def continue_from_checkpoint(self, checkpoint_path: str,
                                 expected_index: pd.Index | None = None,
                                 ) -> tuple[pd.DataFrame, dict]:
        """
        Extend the walk-forward to the hourly bars after the checkpoint.

        Fail-fast (RuntimeError) on: checkpoint schema/n_regimes, boundary
        timestamp and — if `expected_index` is given (the index of the existing
        parquet) — on the whole index of the covered span. NO silent
        fallback: a wrong append would poison the parquet.
        ⚠ The CADENCE used is the one frozen in the checkpoint (chain
        consistency); config↔checkpoint validation is the caller's job
        (see `run_regime_incremental` in 01b).

        Returns:
            (df_new, ckpt): new rows (UTC hourly index) + UPDATED checkpoint
            (to be re-saved via save_wf_checkpoint AFTER saving the parquet:
            the parquet→checkpoint order guarantees a crash leaves at worst a
            stale checkpoint, never a parquet ahead of the checkpoint).
        """
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        if ckpt.get("schema_version") != self._WF_CKPT_SCHEMA:
            raise RuntimeError(
                f"Checkpoint schema {ckpt.get('schema_version')} ≠ "
                f"{self._WF_CKPT_SCHEMA}: rigenera con --regime-bootstrap-checkpoint."
            )
        if ckpt["n_regimes"] != self.n_regimes:
            raise RuntimeError(
                f"n_regimes checkpoint ({ckpt['n_regimes']}) ≠ config "
                f"({self.n_regimes}): full rebuild richiesto."
            )

        # engine configured FROM the checkpoint (single source of truth:
        # frozen scaler, columns, fit hyper-parameters).
        eng              = self._engine
        eng.scaler       = ckpt["scaler"]
        eng.feature_cols = ckpt["feature_cols"]
        eng.n_iter       = ckpt["n_iter"]
        eng.n_restarts   = ckpt["n_restarts"]
        eng.random_state = ckpt["random_state"]

        df_btc = self._build_btc_hourly_df()
        n_old  = int(ckpt["chain"]["n_bars"])
        if len(df_btc) < n_old:
            raise RuntimeError(
                f"Candele aggregate ({len(df_btc)} ore) meno del checkpoint "
                f"({n_old}): storia troncata — full rebuild richiesto."
            )
        # the boundary MUST match: hourly aggregation of old bars is
        # deterministic, a mismatch = candles changed under our feet.
        if df_btc.index[n_old - 1] != ckpt["last_timestamp"]:
            raise RuntimeError(
                f"Frontiera disallineata: checkpoint @ {ckpt['last_timestamp']}, "
                f"candele @ {df_btc.index[n_old - 1]} — full rebuild o re-bootstrap."
            )
        # (audit MINOR-2) with expected_index validate the WHOLE covered span,
        # not just the boundary: catches in-place candle-history revisions with
        # identical row-count and last timestamp.
        if expected_index is not None and not df_btc.index[:n_old].equals(expected_index):
            raise RuntimeError(
                "Index candele-aggregate ≠ index atteso sullo span coperto "
                "(storia candele modificata in-place?) — full rebuild richiesto."
            )

        df_new = eng.continue_walkforward(df_btc, ckpt["chain"])
        if len(df_new):
            ckpt["last_timestamp"] = df_new.index[-1]
        return df_new, ckpt

    # One-off checkpoint bootstrap from EXISTING pkl+parquet (no rebuild):
    # reconstructs the last-retrain state with ONE MLE fit and VALIDATES it by
    # replaying the tail against the production parquet (built-in golden test).
    def bootstrap_wf_checkpoint(
        self,
        hmm_path:        str,
        parquet_path:    str,
        checkpoint_path: str,
        burn_in_days:    int = 30,
        retrain_days:    int = 90,
        atol:            float = 1e-9,
    ) -> dict:
        """
        Assumptions (validated by the replay, fail-fast otherwise):
          · the original run used the same fit hyper-parameters as the current
            engine (n_iter/n_restarts/random_state — production defaults);
          · all scheduled retrains succeeded (regular cadence from burn-in);
          · the scaler persisted in the pkl is the run's (global fit over the span).
        The sign of the reconstructed PCA is aligned to the persisted final PCA, which
        the original run aligned to the same chain → identical orientation.
        """
        engine = RegimeMarkovSwitching.load(hmm_path)
        if engine.n_regimes != self.n_regimes:
            raise RuntimeError(
                f"n_regimes pkl ({engine.n_regimes}) ≠ config ({self.n_regimes})."
            )

        df_btc    = self._build_btc_hourly_df()
        probs_old = pd.read_parquet(parquet_path)
        n = len(probs_old)
        if len(df_btc) < n:
            raise RuntimeError(
                f"Candele ({len(df_btc)} ore) meno del parquet ({n}): "
                f"storia troncata."
            )
        if not df_btc.index[:n].equals(probs_old.index):
            raise RuntimeError(
                "Index candele-aggregate ≠ index parquet sul span coperto: "
                "aggregazione non riproducibile — full rebuild richiesto."
            )

        burn_in_h = burn_in_days * 24
        retrain_h = retrain_days * 24
        if n <= burn_in_h + retrain_h:
            raise RuntimeError("Storia troppo corta per il bootstrap: full rebuild.")
        # last scheduled retrain: first at burn_in, then every retrain_h bars.
        last_retrain = burn_in_h + retrain_h * ((n - 1 - burn_in_h) // retrain_h)

        X_raw  = df_btc[engine.feature_cols].values[:n]
        X_norm = np.clip(engine.scaler.transform(
            np.nan_to_num(X_raw, nan=0.0)
        ), -5, 5)

        pca_k = PCA(n_components=engine.pca.n_components_,
                    random_state=engine.random_state)
        pca_k.fit(X_norm[:last_retrain])
        if np.dot(pca_k.components_[0], engine.pca.components_[0]) < 0:
            pca_k.components_[0] *= -1
        pc1 = pca_k.transform(X_norm)[:, 0]

        log.info(
            f"Bootstrap checkpoint: fit MLE su [:{last_retrain}] "
            f"(unico fit, ~minuti) + replay di {n - last_retrain} barre ..."
        )
        result = engine._fit_single(pc1[:last_retrain])
        if result is None:
            raise RuntimeError("Bootstrap: fit MLE fallito su tutti i restart.")
        params = engine._extract_params(result)
        fmp = result.filtered_marginal_probabilities
        last_filtered = (fmp.values[-1] if isinstance(fmp, pd.DataFrame)
                         else fmp[-1]).copy()

        # Hamilton replay of the tail [last_retrain, n) compared to the parquet:
        # golden test against production — on divergence the checkpoint is NOT
        # written (no poisoned artifact).
        prob_cols = [f"regime_prob_{i}" for i in range(engine.n_regimes)]
        ref    = probs_old[prob_cols].values
        replay = np.empty((n - last_retrain, engine.n_regimes))
        lf = last_filtered
        for i, t in enumerate(range(last_retrain, n)):
            lf = engine._hamilton_filter_step(pc1[t], params, lf)
            replay[i] = lf

        diff     = np.abs(replay - ref[last_retrain:])
        max_diff = float(diff.max())
        exact    = bool((replay == ref[last_retrain:]).all())
        if max_diff > atol:
            raise RuntimeError(
                f"Bootstrap golden FAIL: max|Δprob|={max_diff:.3e} > atol={atol:.0e} "
                f"su {n - last_retrain} barre replay — checkpoint NON salvato "
                f"(iperparametri/versioni divergenti dal run originale?): "
                f"full rebuild richiesto."
            )

        # engine mirror for build_wf_checkpoint (scaler/cols from the run's pkl).
        self._engine.scaler       = engine.scaler
        self._engine.feature_cols = engine.feature_cols
        self._engine.n_iter       = engine.n_iter
        self._engine.n_restarts   = engine.n_restarts
        self._engine.random_state = engine.random_state

        chain = {
            "params":          {k: np.array(v, copy=True)
                                for k, v in params.items()},
            "pca":             pca_k,
            "prev_components": pca_k.components_.copy(),
            "last_filtered":   lf.copy(),
            "last_retrain":    last_retrain,
            "n_bars":          n,
            "burn_in_bars":    burn_in_h,
            "retrain_bars":    retrain_h,
        }
        ckpt = self.build_wf_checkpoint(chain, probs_old.index[-1])
        self.save_wf_checkpoint(ckpt, checkpoint_path)

        report = {"max_diff": max_diff, "exact": exact,
                  "n_validated": n - last_retrain,
                  "last_retrain": last_retrain, "n_bars": n}
        log.info(
            f"Bootstrap golden PASS: max|Δprob|={max_diff:.3e} "
            f"(bit-exact={exact}) su {report['n_validated']} barre."
        )
        return report


# ─── STAGE 2: MacroEncoder ────────────────────────────────────────────────────

# Lightweight MLP compressing the macro snapshot into a bounded [-1,1] dense embedding, trained end-to-end with the LSTM.
class MacroEncoder(nn.Module):
    """
    Lightweight MLP that turns the full macro vector
    into a dense `embed_dim`-dimensional embedding.

    Trained TOGETHER with the LSTM (end-to-end backprop).
    The embedding is concatenated to the GRU output before the parametric
    output head.

    Input: (batch, n_macro_features)  — macro snapshot of the current day
    Output:(batch, embed_dim)         — context vector for the LSTM

    Architecture:
        Linear(n_macro → 64) → LayerNorm → SiLU
        Linear(64 → 32)       → LayerNorm → SiLU → Dropout
        Linear(32 → embed_dim)→ Tanh          ← bounded [-1, 1]
    """

    # Builds the Linear→LN→SiLU→Tanh stack with conservative init (gain 0.5) so it doesn't dominate the gradient.
    def __init__(self, n_macro_features: int, embed_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            nn.Linear(n_macro_features, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, embed_dim),
            nn.Tanh(),   # bounded: prevents the embedding from dominating the gradient
        )

        # Conservative initialization: small embedding at the start
        # → the network learns from price data first, then integrates the macro
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    # Replaces NaNs with 0 and projects the macro snapshot into the embedding.
    def forward(self, x_macro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_macro: (batch, n_macro_features) — NaNs replaced with 0 first

        Returns:
            (batch, embed_dim)
        """
        x = torch.nan_to_num(x_macro, nan=0.0)
        return self.net(x)


# ─── FULL NETWORK WITH MACRO ─────────────────────────────────────────────────

# Full network: price branch (LSTM→GRU, optional dual-stream) fused with the macro embedding → t-Student head (μ, log σ², log ν).
class QuantLSTMWithMacro(nn.Module):
    """
    Full architecture with macro embedding:

        Candele (batch, 60, n_price_features)
               ↓
        LSTM(256) → GRU(128) → h_price (128)
                                        │
        Macro snapshot (batch, n_macro) │
               ↓                        │
        MacroEncoder → h_macro (16)    │
                                  concat │
                                    (144)│
                                        ↓
                              MLP residual (144 → 64 → 32)
                                        ↓
                           [μ,  log_σ²,  log_ν]
                           (t-Student parametrica)

    The 16-dim macro embedding adds ~0.5% of the total parameters
    but can significantly improve calibration in periods of
    macroeconomic stress (e.g. SVB crisis 2023, Fed pivot 2022).
    """

    # Builds the price branch (single/dual-stream), macro encoder and fusion head with residual + clip buffer.
    def __init__(
        self,
        n_price_features:  int,
        n_macro_features:  int,
        lstm_hidden:       int   = 256,
        gru_hidden:        int   = 128,
        mlp_hidden:        int   = 64,
        macro_embed_dim:   int   = 16,
        n_lstm_layers:     int   = 2,
        dropout:           float = 0.2,
        n_dynamic_features: int  = None,   # Improvement 9: dual-stream
    ):
        super().__init__()
        self.n_price_features  = n_price_features
        self.n_macro_features  = n_macro_features
        self.macro_embed_dim   = macro_embed_dim
        self.loss_type         = "t_student"
        self.use_multitask     = False
        self.n_dynamic         = n_dynamic_features
        self.dual_stream       = (n_dynamic_features is not None and
                                  n_dynamic_features < n_price_features)

        # ── Price branch — single or dual stream ──────────────────────────────
        if self.dual_stream:
            n_struct = n_price_features - n_dynamic_features
            self.input_norm_dyn = nn.LayerNorm(n_dynamic_features)
            self.input_proj_dyn = nn.Linear(n_dynamic_features, lstm_hidden)
            from quantsys.model import StructuralEncoder
            self.struct_encoder = StructuralEncoder(n_struct, lstm_hidden, dropout)
        else:
            self.input_norm = nn.LayerNorm(n_price_features)
            self.input_proj = nn.Linear(n_price_features, lstm_hidden)

        self.lstm      = nn.LSTM(lstm_hidden, lstm_hidden, n_lstm_layers,
                                 dropout=dropout if n_lstm_layers > 1 else 0.0,
                                 batch_first=True)
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.gru       = nn.GRU(lstm_hidden, gru_hidden, batch_first=True)
        self.gru_norm  = nn.LayerNorm(gru_hidden)

        # ── Macro branch ──────────────────────────────────────────────────────
        self.macro_encoder = MacroEncoder(n_macro_features, macro_embed_dim, dropout)

        # ── Fusion head ───────────────────────────────────────────────────────
        fusion_dim = gru_hidden + macro_embed_dim
        self.fc1           = nn.Linear(fusion_dim, mlp_hidden)
        self.fc1_norm      = nn.LayerNorm(mlp_hidden)
        self.dropout       = nn.Dropout(dropout)
        self.fc2           = nn.Linear(mlp_hidden, mlp_hidden // 2)
        self.residual_proj = nn.Linear(fusion_dim, mlp_hidden // 2)

        self.out_mu      = nn.Linear(mlp_hidden // 2, 1)
        self.out_logsig2 = nn.Linear(mlp_hidden // 2, 1)
        self.out_lognu   = nn.Linear(mlp_hidden // 2, 1)
        self.register_buffer("clip_lo", torch.full((n_price_features,), -500.0))
        self.register_buffer("clip_hi", torch.full((n_price_features,), +500.0))
        self._init_weights(math)

    # Weight init: Xavier on input/linear, orthogonal on recurrent, log ν bias at ~ν=5.
    def _init_weights(self, math):
        for name, p in self.named_parameters():
            if "weight_ih" in name:    nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:  nn.init.orthogonal_(p)
            elif "bias" in name:       nn.init.zeros_(p)
            elif "weight" in name and p.dim() == 2:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.out_lognu.bias.fill_(math.log(5.0 - 2.0))

    # Forward: clip price → price branch → fuse with macro embedding (zeros if missing) → (μ, log σ², log ν).
    def forward(
        self,
        x_price: torch.Tensor,
        x_macro: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_price = x_price.clamp(self.clip_lo, self.clip_hi)

        # ── Price branch (single or dual stream) ─────────────────────────
        if self.dual_stream:
            x_dyn   = x_price[:, :, :self.n_dynamic]
            x_str   = x_price[:, :, self.n_dynamic:]
            xp_dyn  = F.silu(self.input_proj_dyn(self.input_norm_dyn(x_dyn)))
            xp_str  = self.struct_encoder(x_str)
            xp      = self.struct_encoder.fuse(xp_dyn, xp_str)
        else:
            xp = F.silu(self.input_proj(self.input_norm(x_price)))

        lo, _ = self.lstm(xp); lo = self.lstm_norm(lo)
        go, _ = self.gru(lo);  h_price = self.gru_norm(go[:, -1, :])

        # ── Macro branch ───────────────────────────────────────────────────
        # If x_macro is None (e.g. macro unavailable in live or backtest),
        # use a zero tensor — the MacroEncoder learns to ignore it
        if x_macro is None:
            x_macro = torch.zeros(
                x_price.shape[0], self.macro_encoder.net[0].in_features,
                device=x_price.device, dtype=x_price.dtype,
            )
        h_macro = self.macro_encoder(x_macro)            # (batch, embed_dim)

        # ── Fusion ─────────────────────────────────────────────────────────
        h   = torch.cat([h_price, h_macro], dim=-1)
        out = F.silu(self.fc1_norm(self.fc1(h)))
        out = F.silu(self.fc2(self.dropout(out))) + self.residual_proj(h)

        mu       = self.out_mu(out).squeeze(-1)
        log_sig2 = self.out_logsig2(out).squeeze(-1)
        log_nu   = self.out_lognu(out).squeeze(-1)

        return mu, log_sig2, log_nu

    # No-grad inference: maps raw outputs to {mu, sigma, nu} on CPU/numpy.
    @torch.no_grad()
    def predict(self, x_price: torch.Tensor,
                x_macro: torch.Tensor | None = None) -> dict:
        self.eval()
        mu, ls2, lnu = self.forward(x_price, x_macro)
        sigma2 = F.softplus(ls2) + 1e-6
        nu     = F.softplus(lnu) + 2.0 + 1e-6
        return {
            "mu":    mu.cpu().numpy(),
            "sigma": sigma2.sqrt().cpu().numpy(),
            "nu":    nu.cpu().numpy(),
        }


# ─── MACRO NORMALIZER ────────────────────────────────────────────────────────

# RobustScaler dedicated to macro features (separate from price scalers), clipped to ±5.
class MacroNormalizer:
    """
    Normalize the macro features for the MacroEncoder.
    Uses RobustScaler (resistant to macro-shock outliers).
    Saved separately from the price-feature scalers.
    """

    # Initializes an empty scaler (fitted=False until fit_transform is called).
    def __init__(self):
        self.scaler      = RobustScaler()
        self.feature_cols: list[str] = []
        self.fitted      = False
        # label of the macro vintage fitted on (None = not pinned). See save().
        self.pinned_vintage: str | None = None

    # Stores the columns, fits the scaler and returns normalized data (NaN→0, clip ±5).
    def fit_transform(self, df: pd.DataFrame,
                      macro_cols: list[str]) -> np.ndarray:
        self.feature_cols = macro_cols
        X = df[macro_cols].fillna(0).values.astype(np.float32)
        X = np.clip(X, -1e6, 1e6)
        result = self.scaler.fit_transform(X)
        result = np.clip(result, -5, 5).astype(np.float32)
        self.fitted = True
        return result

    # Applies the already-fitted scaler to new data (NaN→0, clip ±5).
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].fillna(0).values.astype(np.float32)
        X = np.clip(X, -1e6, 1e6)
        result = self.scaler.transform(X)
        return np.clip(result, -5, 5).astype(np.float32)

    # Serializes scaler + columns to disk (pickle). `pinned_vintage` is an OPTIONAL
    # field (label of the macro vintage the normalizer was fitted on): it serves
    # the vol line's PINNED normalizer, where knowing which vintage the instrument
    # is frozen at is half the information. Absent = legacy pickle.
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_cols": self.feature_cols,
                         "pinned_vintage": getattr(self, "pinned_vintage", None)}, f)

    # Reconstructs an already-fitted MacroNormalizer from a saved pickle.
    # Backward compatible: pickles written before the `pinned_vintage` field
    # load with None, no migration needed.
    @classmethod
    def load(cls, path: str) -> "MacroNormalizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.scaler         = data["scaler"]
        obj.feature_cols   = data["feature_cols"]
        obj.pinned_vintage = data.get("pinned_vintage")
        obj.fitted         = True
        return obj
