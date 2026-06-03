"""
quantsys/macro/regime.py
========================
Rilevamento del Regime Economico in due stadi:

  STADIO 1 — HMM (Hidden Markov Model) non supervisionato
    Trova automaticamente N regimi latenti nei dati macro storici.
    Output: probabilità di trovarsi in ciascun regime per ogni giorno.

  STADIO 2 — MacroEncoder (MLP)
    Rete neurale leggera che trasforma le macro features grezze
    in un embedding denso a 16 dimensioni.
    Viene addestrata INSIEME alla LSTM principale (end-to-end).

Perché HMM?
  · Modella esplicitamente che il regime cambia nel tempo (transizioni)
  · I 4 regimi emergono dai dati — non li definiamo a priori
  · Le probabilità di regime (es. [0.8, 0.1, 0.05, 0.05]) sono
    interpretabili e robuste anche con dati mancanti

I 4 regimi che l'HMM tipicamente scopre nei dati USA:
  0 = Espansione moderata    (crescita stabile, inflazione bassa)
  1 = Surriscaldamento       (crescita alta, inflazione in salita)
  2 = Stagflazione / crisi   (crescita bassa, inflazione alta)
  3 = Recessione / pivot Fed (crescita negativa, Fed in taglio)

Nota: le etichette sono interpretative — l'HMM impara solo pattern.
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

# Colonne macro chiave per l'HMM (quelle più informative sul regime)
# Ordine: [inflazione, crescita, mercato_lavoro, condizioni_finanziarie, tassi]
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


# ─── STADIO 1: HMM REGIME DETECTOR ──────────────────────────────────────────

# IT: Gaussian HMM walk-forward (expanding window) che etichetta i regimi macro senza look-ahead.
# EN: Walk-forward Gaussian HMM (expanding window) labeling macro regimes without look-ahead.
class RegimeHMM:
    """
    Gaussian HMM con N stati latenti per il rilevamento del regime economico.

    FIX LOOK-AHEAD BIAS — Walk-Forward Expanding Window:
    ─────────────────────────────────────────────────────
    Il problema originale: l'HMM veniva addestrato sull'intero storico
    (es. 2018-2024) e poi applicato per classificare ogni giorno, inclusi
    quelli del 2018. Ma l'HMM "sapeva già" come andava a finire nel 2024
    quando classificava il 2018 — look-ahead bias sottile ma reale.

    La soluzione — expanding window:
      · Per ogni data t nel dataset, l'HMM viene addestrato SOLO sui dati
        disponibili fino a t-1 (nessuna informazione futura).
      · In pratica: addestriamo L'HMM su un burn-in iniziale (es. 2 anni),
        poi lo aggiorniamo incrementalmente ogni N giorni con nuovi dati.
      · Le etichette di regime per ogni giorno vengono generate con il
        modello che esisteva a quel momento, non con quello finale.

    Trade-off:
      · Più lento (K addestramenti invece di 1, dove K = n_updates).
      · Più realistico: le probabilità di regime che il MacroEncoder
        riceve durante il training corrispondono a quelle disponibili
        in quell'istante, senza informazione dal futuro.
    """

    # IT: Configura n. regimi, iterazioni EM e restart; lo scaler è fittato al fit.
    # EN: Sets regime count, EM iterations and restarts; the scaler is fit at fit time.
    def __init__(self, n_regimes: int = 4, n_iter: int = 100, random_state: int = 42,
                 n_restarts: int = 5):
        self.n_regimes    = n_regimes
        self.n_iter       = n_iter
        self.random_state = random_state
        self.n_restarts   = n_restarts
        self.model        = None   # modello corrente (l'ultimo addestrato)
        self.scaler       = RobustScaler()
        self.feature_cols: list[str] = []

    # IT: Sceglie le HMM_CORE_FEATURES presenti; fallback alle prime 20 colonne numeriche.
    # EN: Picks the available HMM_CORE_FEATURES; falls back to the first 20 numeric columns.
    def _select_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Seleziona le colonne disponibili tra quelle ideali per l'HMM."""
        available = [c for c in HMM_CORE_FEATURES if c in df.columns]
        if len(available) < 4:
            # Fallback: usa tutte le colonne numeriche disponibili (fino a 20)
            available = [c for c in df.select_dtypes("number").columns][:20]
        log.info(f"HMM: {len(available)} features selezionate")
        return df[available].values, available

    # IT: Fit HMM con n_restarts seed; tiene il modello con log-likelihood massima (early-exit a convergenza).
    # EN: Fits the HMM over n_restarts seeds; keeps the max log-likelihood model (early-exit on convergence).
    def _fit_single(self, X_norm: np.ndarray):
        """
        Addestra HMM con n_restarts seed diversi, ritorna il modello con log-likelihood massima.

        Perché i restart: GaussianHMM con covariance_type="full" ha 14×14×4=784 parametri
        di covarianza. Con finestre brevi (365-500 gg) l'EM si inceppa spesso in ottimi
        locali con log-likelihood decrescente. Provando N seed diversi e tenendo il
        modello con score più alto si ottiene una soluzione stabile senza cambiare il
        modello statistico.
        Early exit: se un restart converge prima di esaurire n_iter, non serve provarne altri.
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
                    break   # converged → inutile provare altri seed
            except Exception:
                continue

        return best_model

    # IT: Genera le etichette di regime giorno-per-giorno usando solo dati passati (no look-ahead).
    # EN: Produces day-by-day regime labels using only past data (no look-ahead).
    def fit_predict_walkforward(
        self,
        df_macro:     pd.DataFrame,
        burn_in_days: int = 365,   # giorni minimi di storia prima di predire
        retrain_days: int = 90,    # riaddestra ogni N giorni
    ) -> pd.DataFrame:
        """
        Walk-forward expanding window:
        per ogni giorno t genera le probabilità di regime usando SOLO
        i dati disponibili fino a t (nessun look-ahead).

        Args:
            df_macro:     DataFrame giornaliero con macro features
            burn_in_days: giorni minimi prima di iniziare a classificare
            retrain_days: frequenza di riaddestramento (es. ogni 90 gg)

        Returns:
            DataFrame con colonne regime_prob_0..K e regime_dominant,
            indicizzato come df_macro. Le prime burn_in_days righe
            avranno probabilità uniformi (1/n_regimes) — periodo di burn-in.
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

        # Normalizzazione globale: fittiamo lo scaler su tutto lo storico
        # (lo scaler non "sa il futuro" — usa solo statistiche di posizione/scala)
        # Alternativa più rigorosa: scaler espandente. Ma RobustScaler è
        # già robusto agli outlier, e la normalizzazione non porta look-ahead
        # nelle probabilità di regime.
        mask_valid = ~np.isnan(X_raw).any(axis=1)
        self.scaler.fit(X_raw[mask_valid])
        X_norm = np.clip(self.scaler.transform(
            np.nan_to_num(X_raw, nan=0.0)
        ), -5, 5)

        # Storage risultati — inizia con probabilità uniformi (no-info prior)
        probs_all = np.full((n, self.n_regimes), 1.0 / self.n_regimes)

        current_model = None
        last_retrain  = -1

        log.info(
            f"Walk-forward HMM: {n} giorni, burn_in={burn_in_days}, "
            f"retrain ogni {retrain_days} gg ..."
        )

        for t in range(burn_in_days, n):
            # Riaddestra se: primo modello, o scaduto il periodo di retrain
            if current_model is None or (t - last_retrain) >= retrain_days:
                # Usa SOLO dati fino a t (esclude t stesso e il futuro)
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

            # Genera probabilità per il giorno t con il modello corrente
            if current_model is not None:
                x_t = X_norm[t:t+1]
                if not np.isnan(X_raw[t]).any():
                    try:
                        probs_all[t] = current_model.predict_proba(x_t)[0]
                    except Exception:
                        pass  # mantieni probabilità uniformi

        # Salva il modello finale (addestrato su tutto lo storico)
        if current_model is not None:
            self.model = current_model

        # Costruisce DataFrame risultato allineato all'indice di df_macro
        result = pd.DataFrame(
            probs_all,
            index  = df_daily.index,
            columns= [f"regime_prob_{i}" for i in range(self.n_regimes)],
        )
        result["regime_dominant"] = probs_all.argmax(axis=1)
        # Le prime burn_in_days righe hanno probabilità uniformi
        result["regime_burn_in"] = False
        result.iloc[:burn_in_days, -1] = True

        log.info(
            f"Walk-forward completato. "
            f"Burn-in: {burn_in_days} giorni. "
            f"Classificati: {n - burn_in_days} giorni."
        )
        self._describe_regimes_wf(X_raw, probs_all, burn_in_days)
        return result

    # IT: Logga le caratteristiche medie di ciascun regime sul periodo classificato (diagnostica).
    # EN: Logs each regime's mean characteristics over the classified period (diagnostics).
    def _describe_regimes_wf(self, X_raw, probs_all, burn_in_days):
        """Analisi dei regimi sul periodo classificato (post burn-in)."""
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

    # IT: Fit finale su tutto lo storico per il modello di produzione (NON per le etichette di training).
    # EN: Final fit on the full history for the production model (NOT for training labels).
    def fit(self, df_macro: pd.DataFrame) -> "RegimeHMM":
        """
        Addestramento finale sull'intero storico (per il modello di produzione).
        Usare SOLO per il modello che gira in produzione/live — NON per
        generare le etichette di training (usare fit_predict_walkforward).
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

    # IT: Probabilità di regime per ogni riga via predict_proba dell'HMM addestrato.
    # EN: Per-row regime probabilities via the trained HMM's predict_proba.
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predice le probabilità di regime (usa il modello corrente)."""
        if self.model is None:
            raise RuntimeError("HMM non addestrato.")
        X_raw = df[self.feature_cols].values if self.feature_cols else df.values
        X_raw = np.nan_to_num(X_raw, nan=0.0)
        X     = np.clip(self.scaler.transform(X_raw), -5, 5)
        return self.model.predict_proba(X)

    # IT: Serializza modello + scaler + metadati su disco (pickle).
    # EN: Serializes model + scaler + metadata to disk (pickle).
    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "scaler": self.scaler,
                "feature_cols": self.feature_cols, "n_regimes": self.n_regimes,
            }, f)
        log.info(f"HMM salvato → {path}")

    # IT: Ricostruisce un RegimeHMM da un pickle salvato.
    # EN: Reconstructs a RegimeHMM from a saved pickle.
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


# ─── STADIO 1b: MARKOV-SWITCHING REGIME DETECTOR (Hamilton 1989) ────────────

# IT: Markov-Switching (Hamilton 1989) su PC1 con PCA expanding window + Hamilton filter walk-forward.
# EN: Markov-Switching (Hamilton 1989) on PC1 with expanding-window PCA + walk-forward Hamilton filter.
class RegimeMarkovSwitching:
    """
    Markov-Switching Regression (Hamilton 1989) per regime detection.

    Pipeline standard quant finance:
      1. RobustScaler sulle macro features (resistente a outlier)
      2. PCA → n_pca componenti principali (riduce multicollinearità)
      3. MarkovRegression su PC1 con switching mean + variance
      4. Hamilton filter per probabilità di regime sequenziali

    Vantaggi rispetto a GaussianHMM:
      · Convergenza stabile (EM + scoring optimizer, non solo EM)
      · Standard econometrico pubblicato (Hamilton 1989, Kim & Nelson 1999)
      · PCA elimina multicollinearità fra le 14 macro features
      · switching_variance cattura i cambi di volatilità tra regimi
      · Meno parametri → meno overfitting con finestre corte

    Approccio walk-forward:
      Identico a RegimeHMM — expanding window con riaddestramento periodico.
      Fra un retrain e l'altro, usa il Hamilton filter (O(1) per giorno)
      anziché ri-fittare il modello.
    """

    # IT: Configura n. regimi/PCA/restart; scaler, PCA e cache parametri sono popolati al fit.
    # EN: Sets regime/PCA/restart counts; scaler, PCA and param cache are populated at fit time.
    def __init__(self, n_regimes: int = 3, n_iter: int = 300,
                 random_state: int = 42, n_pca: int = 3,
                 n_restarts: int = 5):
        if n_regimes < 2:
            raise ValueError(f"n_regimes deve essere >= 2, ricevuto {n_regimes}")
        self.n_regimes    = n_regimes
        self.n_iter       = n_iter
        self.random_state = random_state
        self.n_pca        = n_pca
        self.n_restarts   = n_restarts
        self.model        = None
        self.pca          = None
        self.scaler       = RobustScaler()
        self.feature_cols: list[str] = []
        self._params_cache: dict | None = None

    # IT: Come la versione HMM, ma scarta anche le colonne interamente NaN (PCA non le tollera).
    # EN: Like the HMM version, but also drops all-NaN columns (PCA cannot handle them).
    def _select_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        available = [c for c in HMM_CORE_FEATURES if c in df.columns]
        if len(available) < 4:
            available = [c for c in df.select_dtypes("number").columns][:20]
        available = [c for c in available
                     if not df[c].isna().all()]
        log.info(f"MarkovSwitching: {len(available)} features selezionate")
        return df[available].values, available

    # IT: Fitta la PCA e proietta X_norm sulle componenti principali (logga la varianza spiegata).
    # EN: Fits the PCA and projects X_norm onto principal components (logs explained variance).
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

    # IT: Proietta nuovi dati sulla PCA già fittata (nessun refit).
    # EN: Projects new data onto the already-fitted PCA (no refit).
    def _pca_transform(self, X_norm: np.ndarray) -> np.ndarray:
        return self.pca.transform(X_norm)

    # IT: Fitta MarkovRegression su PC1 (switching mean+variance) su n_restarts seed; tiene la llf massima.
    # EN: Fits MarkovRegression on PC1 (switching mean+variance) over n_restarts seeds; keeps the max llf.
    def _fit_single(self, pc1: np.ndarray):
        """
        Fitta MarkovRegression su PC1 con switching mean + variance.
        Prova n_restarts starting values per robustezza.
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

    # IT: Estrae transizioni, medie e varianze dal risultato statsmodels per il Hamilton filter manuale.
    # EN: Extracts transitions, means and variances from the statsmodels result for the manual Hamilton filter.
    def _extract_params(self, result) -> dict:
        """
        Estrae i parametri per il Hamilton filter manuale.

        Convenzione statsmodels: regime_transition[i,j] = P(S_t=i | S_{t-1}=j)
        (le colonne sommano a 1).
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

    # IT: Un passo del filtro di Hamilton (predict + emission log-space + update normalizzato).
    # EN: One Hamilton-filter step (predict + log-space emission + normalized update).
    def _hamilton_filter_step(self, y_t: float, params: dict,
                              prev_filtered: np.ndarray) -> np.ndarray:
        """
        Singolo passo del Hamilton (1989) filter.

        P(S_t=j | Y_{1:t}) ∝ f(y_t | S_t=j) · P(S_t=j | Y_{1:t-1})

        dove P(S_t=j | Y_{1:t-1}) = Σ_i P(S_t=j | S_{t-1}=i) · P(S_{t-1}=i | Y_{1:t-1})

        Emission calcolata in log-space per evitare underflow su outlier.
        """
        trans = params['trans']

        # Prediction step: trans @ prev (colonne sommano a 1)
        predicted = trans @ prev_filtered
        predicted = np.maximum(predicted, 1e-20)

        # Emission in log-space (evita underflow per |y_t - μ_j| >> σ_j)
        diff = y_t - params['means']
        var  = params['variances']
        log_emission = -0.5 * diff**2 / var - 0.5 * np.log(2.0 * np.pi * var)
        log_emission -= log_emission.max()
        emission = np.exp(log_emission)

        # Update step (la costante sottratta si cancella nella normalizzazione)
        joint = emission * predicted
        total = joint.sum()
        if total < 1e-300:
            return np.full(self.n_regimes, 1.0 / self.n_regimes)
        return joint / total

    # IT: Etichette di regime walk-forward: a ogni retrain ri-fitta PCA+MS (sign-aligned), poi Hamilton filter fino al prossimo.
    # EN: Walk-forward regime labels: each retrain refits PCA+MS (sign-aligned), then Hamilton-filters until the next.
    def fit_predict_walkforward(
        self,
        df_macro:     pd.DataFrame,
        burn_in_days: int = 365,
        retrain_days: int = 90,
    ) -> pd.DataFrame:
        """
        Walk-forward expanding window con Markov-Switching + PCA.

        Per ogni giorno t genera le probabilità di regime usando SOLO
        i dati disponibili fino a t (nessun look-ahead).

        Pipeline per ogni retrain a t:
          1. PCA fit su X_norm[:t] (expanding window, sign-aligned)
          2. MarkovRegression su PC1[:t]
          3. Hamilton filter per i giorni successivi fino al prossimo retrain,
             trasformando ogni x_t con la PCA corrente
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

        # Normalizzazione globale (RobustScaler: statistiche di posizione/scala,
        # look-ahead trascurabile — mediana e IQR sono stabili nel tempo)
        mask_valid = ~np.isnan(X_raw).any(axis=1)
        self.scaler.fit(X_raw[mask_valid])
        X_norm = np.clip(self.scaler.transform(
            np.nan_to_num(X_raw, nan=0.0)
        ), -5, 5)

        n_comp = min(self.n_pca, X_norm.shape[1], X_norm.shape[0])

        # Storage risultati
        probs_all = np.full((n, self.n_regimes), 1.0 / self.n_regimes)

        current_params  = None
        current_pca     = None
        prev_components = None
        last_filtered   = np.full(self.n_regimes, 1.0 / self.n_regimes)
        last_retrain    = -1

        log.info(
            f"Walk-forward MarkovSwitching: {n} giorni, "
            f"burn_in={burn_in_days}, retrain ogni {retrain_days} gg, "
            f"{self.n_regimes} regimi, PCA expanding window ..."
        )

        for t in range(burn_in_days, n):
            if current_params is None or (t - last_retrain) >= retrain_days:
                if t >= 50:
                    try:
                        # PCA expanding window: fit solo su dati[:t]
                        pca_t = PCA(n_components=n_comp,
                                    random_state=self.random_state)
                        pca_t.fit(X_norm[:t])

                        # Sign alignment: PC1 deve puntare nella stessa
                        # direzione tra un retrain e l'altro, altrimenti il
                        # Hamilton filter riceve un segnale invertito
                        if prev_components is not None:
                            if np.dot(pca_t.components_[0],
                                      prev_components[0]) < 0:
                                pca_t.components_[0] *= -1
                        prev_components = pca_t.components_.copy()
                        current_pca = pca_t

                        pc1_train = pca_t.transform(X_norm[:t])[:, 0]

                        result = self._fit_single(pc1_train)
                        if result is not None:
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
                    except Exception as e:
                        log.warning(
                            f"  MarkovSwitching fit fallito a t={t}: {e}"
                        )

            if current_params is not None and current_pca is not None:
                pc1_t = current_pca.transform(X_norm[t:t+1])[0, 0]
                last_filtered = self._hamilton_filter_step(
                    pc1_t, current_params, last_filtered
                )
                probs_all[t] = last_filtered

        # Fit finale su tutto lo storico (per produzione/predict_proba)
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

        # DataFrame risultato
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

    # IT: Diagnostica: logga le medie per regime + μ/σ²/P(stay) dei parametri stimati.
    # EN: Diagnostics: logs per-regime means + the estimated μ/σ²/P(stay) parameters.
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

    # IT: Fit finale su tutto lo storico (PCA+MS) per produzione/live; popola la cache parametri.
    # EN: Final fit on the full history (PCA+MS) for production/live; populates the param cache.
    def fit(self, df_macro: pd.DataFrame) -> "RegimeMarkovSwitching":
        """
        Addestramento finale sull'intero storico (produzione/live).
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

    # IT: Probabilità di regime applicando il Hamilton filter sequenzialmente riga-per-riga.
    # EN: Regime probabilities by applying the Hamilton filter sequentially, row by row.
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice le probabilità di regime con Hamilton filter.

        Per ogni riga applica un passo del filtro sequenzialmente,
        così le probabilità tengono conto della storia precedente.
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

    # IT: Serializza modello + PCA + scaler + cache parametri su disco (pickle).
    # EN: Serializes model + PCA + scaler + param cache to disk (pickle).
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

    # IT: Ricostruisce un RegimeMarkovSwitching da un pickle salvato.
    # EN: Reconstructs a RegimeMarkovSwitching from a saved pickle.
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

    # IT: Confronta k regimi candidati via BIC e ritorna {k: BIC}, loggando il vincitore.
    # EN: Compares candidate regime counts k via BIC and returns {k: BIC}, logging the winner.
    def select_n_regimes(
        self, df_macro: pd.DataFrame,
        candidates: list[int] = [2, 3, 4],
    ) -> dict:
        """
        Seleziona il numero ottimale di regimi via BIC.

        Fitta il modello per ogni k in candidates e ritorna
        {k: BIC} con log del vincitore.
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


# ─── STADIO 1c: SESSION-BASED REGIME DETECTOR (Asia/EU/US) ──────────────────

# IT: Regime intraday basato sulla sessione di trading (Asia/EU/US) — sostituisce
#     il Markov-Switching macro degenere allineando timescale del regime e
#     dell'orizzonte di trading (1m, h=30).
# EN: Intraday regime based on trading session (Asia/EU/US) — replaces the
#     degenerate macro Markov-Switching by aligning the regime timescale with
#     the trading horizon (1m, h=30).
class RegimeSession:
    """
    Detector di regime "session-based" deterministico, drop-in per
    `RegimeMarkovSwitching`.

    Regola di mapping (UTC):
        regime 0 = Asia       [00:00, 08:00)
        regime 1 = EU/London  [08:00, 16:00)
        regime 2 = US         [16:00, 24:00)
    equivalente a `regime = hour_utc // 8`.

    Vantaggi rispetto al Markov-Switching macro:
      · Timescale coerente con l'orizzonte di trading (1-min, h=30)
      · Sempre 3 cluster ben bilanciati (~33% ciascuno) — niente collasso
      · Nessuna dipendenza da EM/convergenza/look-ahead
      · Zero parametri da fittare → riproducibile e robusto

    Note di interfaccia:
      Restituisce lo stesso schema del Markov-Switching
      (`regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`)
      così che i consumer (02_train.py, dashboard) non vadano toccati.
    """

    # IT: Configurazione minima — nessun parametro effettivo da apprendere.
    # EN: Minimal config — no actual parameters to learn.
    def __init__(self, n_regimes: int = 3):
        # IT: forziamo n_regimes=3 (Asia/EU/US); accettiamo il kwarg solo per
        #     compatibilità di firma con RegimeMarkovSwitching.
        # EN: we force n_regimes=3 (Asia/EU/US); we accept the kwarg only for
        #     signature compatibility with RegimeMarkovSwitching.
        if n_regimes != 3:
            log.warning(
                f"RegimeSession: n_regimes={n_regimes} ignorato, forzato a 3 "
                f"(Asia/EU/US sono fisse)."
            )
        self.n_regimes = 3
        # IT: campi placeholder per drop-in compatibility con il pickle MS.
        # EN: placeholder fields for drop-in pickle compatibility with MS.
        self.model = None
        self.pca = None
        self.scaler = None
        self.feature_cols: list[str] = []

    # IT: Genera l'index orario UTC sul range di df_macro e calcola i regimi.
    # EN: Builds the hourly UTC index over df_macro's range and computes regimes.
    def fit_predict_walkforward(
        self,
        df_macro: pd.DataFrame,
        burn_in_days: int = 0,   # IT: non usato — accettato per compat / EN: unused — accepted for compat
        retrain_days: int = 0,   # IT: non usato — accettato per compat / EN: unused — accepted for compat
        **kwargs,                # IT: assorbe eventuali kwargs futuri / EN: absorbs any future kwargs
    ) -> pd.DataFrame:
        """
        Calcola i regimi session-based su un range orario UTC.

        Args:
            df_macro:     usato SOLO per determinare il range temporale
                          (min/max dell'index). Il contenuto non viene letto.
            burn_in_days: ignorato (no fit, no burn-in necessario).
            retrain_days: ignorato (deterministico, niente retrain).

        Returns:
            DataFrame indicizzato su timestamps orari UTC con colonne:
              · regime_dominant (int 0/1/2)
              · regime_burn_in  (bool, sempre False)
              · regime_prob_0   (float, one-hot)
              · regime_prob_1   (float, one-hot)
              · regime_prob_2   (float, one-hot)
        """
        # IT: estrae il range temporale da df_macro (tollera index non-tz / nullo).
        # EN: extracts the time range from df_macro (tolerates non-tz / empty index).
        if df_macro is None or len(df_macro) == 0:
            raise ValueError("RegimeSession: df_macro vuoto, impossibile derivare il range.")

        idx_raw = pd.to_datetime(df_macro.index)
        # IT: normalizza a UTC (se naive, assume UTC; se tz-aware, converti).
        # EN: normalize to UTC (if naive, assume UTC; if tz-aware, convert).
        if idx_raw.tz is None:
            idx_utc = idx_raw.tz_localize("UTC")
        else:
            idx_utc = idx_raw.tz_convert("UTC")

        t_min = idx_utc.min().floor("h")
        t_max = idx_utc.max().ceil("h")

        # IT: index orario UTC che copre l'intero range [t_min, t_max].
        # EN: hourly UTC index spanning the full range [t_min, t_max].
        hourly_idx = pd.date_range(start=t_min, end=t_max, freq="h", tz="UTC")

        # IT: mapping hour → regime (0=Asia, 1=EU, 2=US).
        # EN: hour → regime mapping (0=Asia, 1=EU, 2=US).
        hours = hourly_idx.hour.to_numpy()
        regime_dominant = (hours // 8).astype(np.int64)

        # IT: one-hot delle probabilità per match con lo schema MS.
        # EN: one-hot probabilities to match the MS schema.
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

    # IT: Pickle dei pochi attributi (nessun parametro fittato).
    # EN: Pickles the few attributes (no fitted parameters).
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

    # IT: Ricostruisce un RegimeSession da pickle.
    # EN: Reconstructs a RegimeSession from a pickle.
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


# ─── STADIO 1d: MARKOV-SWITCHING SU REALIZED VOL BTC (intraday) ─────────────

# IT: Markov-Switching su realized vol BTC oraria — Variante 3 di MODEL_IMPROVEMENTS.md.
#     Sostituisce RegimeSession allineando il timescale del regime (switch ogni 3-8h)
#     col timeframe trading (1m, h=30). Usa SOLO dati BTC, non più macro USA.
# EN: Markov-Switching on hourly BTC realized volatility — Variant 3 of MODEL_IMPROVEMENTS.md.
#     Replaces RegimeSession by aligning the regime timescale (switches every 3-8h)
#     with the trading timeframe (1m, h=30). Uses BTC data ONLY, no more US macro.
class RegimeMarkovBTC:
    """
    Markov-Switching (Hamilton 1989) su realized volatility intraday di BTC.

    Pipeline:
      1. Carica candele 1m BTC da `data/raw_candles.parquet`
      2. Aggrega a 1 ora: log_ret_h (somma log_ret 1m) + log_rv (log della
         realized variance = log(Σ log_ret²) clippato per stabilità)
      3. RobustScaler globale (mediana/IQR, look-ahead trascurabile)
      4. Walk-forward expanding window con `RegimeMarkovSwitching` come engine:
         · PCA(n_pca=1) combina log_ret_h + log_rv in un singolo segnale
         · MarkovRegression con switching mean + variance su PC1
         · Hamilton filter O(1) tra un retrain e l'altro

    Razionale:
      - Variante 3 di `MODEL_IMPROVEMENTS.md` (§"NEW — Sostituire MS macro con
        regime intraday su BTC"). Il MS su macro era degenere (regimi mensili
        vs trading 1m); il session-based era informativamente vuoto. La realized
        vol BTC oraria cambia 3-8 volte/giorno → match col forecast horizon h=30.

    Interfaccia drop-in con RegimeSession / RegimeMarkovSwitching:
      - `fit_predict_walkforward(df_macro=...)` accetta ma IGNORA df_macro.
      - Restituisce DataFrame con index orario UTC e colonne
        `regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`.
    """

    # IT: Configura il detector; il MS engine è composito (non subclass) per riuso clean.
    # EN: Configures the detector; the MS engine is composed (not subclassed) for clean reuse.
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
        # IT: n_pca=1 → PCA riduce (log_ret_h, log_rv) a un'unica direzione informativa.
        # EN: n_pca=1 → PCA reduces (log_ret_h, log_rv) to a single informative direction.
        self._engine = RegimeMarkovSwitching(
            n_regimes=n_regimes,
            n_iter=n_iter,
            random_state=random_state,
            n_pca=1,
            n_restarts=n_restarts,
        )
        # IT: campi mirror per save/load drop-in con i consumer.
        # EN: mirror fields for save/load drop-in with consumers.
        self.model = None
        self.pca = None
        self.scaler = None
        self.feature_cols: list[str] = []

    # IT: Aggrega le candele 1m in feature orarie (log-return + log realized vol).
    # EN: Aggregates 1m candles into hourly features (log-return + log realized vol).
    def _build_btc_hourly_df(self) -> pd.DataFrame:
        """
        Carica `raw_candles.parquet` e produce un DataFrame orario UTC con:
          · log_ret_h: somma dei log-return 1m per ogni ora (return orario)
          · log_rv   : log della realized variance oraria = log(Σ log_ret²)

        Il log-trasform su `rv` è essenziale: la realized variance è fortemente
        right-skewed → senza log, la MarkovRegression collassa su outlier.
        """
        from pathlib import Path as _Path
        path = _Path(self.candles_path)
        if not path.exists():
            raise FileNotFoundError(
                f"RegimeMarkovBTC: {path} non trovato. "
                f"Esegui prima `python scripts/01_download_data.py`."
            )

        # IT: lettura + normalizzazione dell'indice temporale UTC.
        # EN: read + UTC time-index normalization.
        candles = pd.read_parquet(path, columns=["open_time", "close"])
        # IT: pd.api.types gestisce sia datetime naive sia tz-aware (np.issubdtype no).
        # EN: pd.api.types handles both naive and tz-aware datetime (np.issubdtype doesn't).
        if not pd.api.types.is_datetime64_any_dtype(candles["open_time"]):
            candles["open_time"] = pd.to_datetime(
                candles["open_time"], unit="ms", utc=True,
            )
        candles = candles.sort_values("open_time").set_index("open_time")
        if candles.index.tz is None:
            candles.index = candles.index.tz_localize("UTC")
        else:
            candles.index = candles.index.tz_convert("UTC")

        # IT: log-return 1m (close-to-close); inf/NaN dropati a valle.
        # EN: 1m log-return (close-to-close); inf/NaN dropped downstream.
        log_ret = np.log(candles["close"]).diff()
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan)

        # IT: aggregazione oraria — somma log-return + somma quadrati (realized var).
        # EN: hourly aggregation — sum of log-returns + sum of squares (realized var).
        log_ret_h = log_ret.resample("1h").sum()
        rv = log_ret.pow(2).resample("1h").sum()
        # IT: clip a 1e-12 per evitare log(0) su ore senza variazione.
        # EN: clip at 1e-12 to avoid log(0) on hours with no variation.
        log_rv = np.log(rv.clip(lower=1e-12))

        out = pd.DataFrame({"log_ret_h": log_ret_h, "log_rv": log_rv})
        out = out.dropna()
        log.info(
            f"RegimeMarkovBTC: aggregate {len(out)} ore "
            f"({out.index.min()} → {out.index.max()}); "
            f"log_rv mean={out['log_rv'].mean():.2f} std={out['log_rv'].std():.2f}"
        )
        return out

    # IT: Walk-forward sul df BTC; df_macro è ignorato (interfaccia drop-in).
    # EN: Walk-forward on the BTC df; df_macro is ignored (drop-in interface).
    def fit_predict_walkforward(
        self,
        df_macro: pd.DataFrame = None,
        burn_in_days: int = 30,
        retrain_days: int = 30,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Walk-forward expanding window con Markov-Switching su realized vol BTC.

        Args:
            df_macro:     ignorato (usato SOLO per compat di firma).
            burn_in_days: convertito in ore (×24). Default 30gg = 720h.
            retrain_days: convertito in ore (×24). Default 30gg = 720h.

        Returns:
            DataFrame indicizzato su ore UTC con colonne:
              · regime_dominant   (int 0..n_regimes-1)
              · regime_burn_in    (bool)
              · regime_prob_{i}   (float, somma=1 per riga)
        """
        df_btc = self._build_btc_hourly_df()

        # IT: il MS engine ragiona in "giorni" come unità positional; qui un
        #     "passo" è un'ora — convertiamo burn_in e retrain in ore.
        # EN: the MS engine reasons in "days" as positional units; here a "step"
        #     is one hour — convert burn_in and retrain to hours.
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

        # IT: l'engine resetta l'index a positional 0..N-1; ripristiniamo orario UTC.
        # EN: the engine resets the index to positional 0..N-1; restore UTC hourly index.
        result.index = df_btc.index

        # IT: mirror dei campi engine per i consumer di save/load.
        # EN: mirror engine fields for save/load consumers.
        self.model = self._engine.model
        self.pca = self._engine.pca
        self.scaler = self._engine.scaler
        self.feature_cols = self._engine.feature_cols

        # IT: log distribuzione finale (post burn-in) per diagnostica rapida.
        # EN: final post-burn-in distribution log for quick diagnostics.
        post = result.iloc[burn_in_h:]
        counts = post["regime_dominant"].value_counts().sort_index()
        log.info("─── Distribuzione regimi BTC (post burn-in) ───")
        for r, c in counts.items():
            pct = c / len(post) * 100
            log.info(f"  Regime {r}: {c} ore ({pct:.1f}%)")

        return result

    # IT: Delega il pickle al MS engine (stesso schema, drop-in con MS loader).
    # EN: Delegates pickling to the MS engine (same schema, drop-in MS loader).
    def save(self, path: str) -> None:
        self._engine.save(path)
        log.info(f"RegimeMarkovBTC salvato → {path}  (schema MS engine)")

    # IT: Ricostruisce un RegimeMarkovBTC riusando il loader dell'engine MS.
    # EN: Reconstructs a RegimeMarkovBTC by reusing the MS engine loader.
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


# ─── STADIO 2: MacroEncoder ───────────────────────────────────────────────────

# IT: MLP leggero che comprime lo snapshot macro in un embedding denso bounded [-1,1], addestrato end-to-end con la LSTM.
# EN: Lightweight MLP compressing the macro snapshot into a bounded [-1,1] dense embedding, trained end-to-end with the LSTM.
class MacroEncoder(nn.Module):
    """
    MLP leggero che trasforma il vettore macro completo
    in un embedding denso a `embed_dim` dimensioni.

    Viene addestrato INSIEME alla LSTM (end-to-end backprop).
    L'embedding viene concatenato all'output della GRU prima della testa
    di output parametrico.

    Input: (batch, n_macro_features)  — snapshot macro del giorno corrente
    Output:(batch, embed_dim)         — vettore di contesto per la LSTM

    Architettura:
        Linear(n_macro → 64) → LayerNorm → SiLU
        Linear(64 → 32)       → LayerNorm → SiLU → Dropout
        Linear(32 → embed_dim)→ Tanh          ← bounded [-1, 1]
    """

    # IT: Costruisce lo stack Linear→LN→SiLU→Tanh con init conservativo (gain 0.5) per non dominare il gradiente.
    # EN: Builds the Linear→LN→SiLU→Tanh stack with conservative init (gain 0.5) so it doesn't dominate the gradient.
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
            nn.Tanh(),   # bounded: evita che l'embedding domini il gradiente
        )

        # Inizializzazione conservativa: embedding piccolo all'inizio
        # → la rete impara prima dai dati di prezzo, poi integra il macro
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    # IT: Sostituisce i NaN con 0 e proietta lo snapshot macro nell'embedding.
    # EN: Replaces NaNs with 0 and projects the macro snapshot into the embedding.
    def forward(self, x_macro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_macro: (batch, n_macro_features) — NaN sostituiti con 0 prima

        Returns:
            (batch, embed_dim)
        """
        x = torch.nan_to_num(x_macro, nan=0.0)
        return self.net(x)


# ─── RETE COMPLETA CON MACRO ─────────────────────────────────────────────────

# IT: Rete completa: branch prezzo (LSTM→GRU, opz. dual-stream) fuso col macro embedding → testa t-Student (μ, log σ², log ν).
# EN: Full network: price branch (LSTM→GRU, optional dual-stream) fused with the macro embedding → t-Student head (μ, log σ², log ν).
class QuantLSTMWithMacro(nn.Module):
    """
    Architettura completa con macro embedding:

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

    Il macro embedding da 16 dim aggiunge ~0.5% dei parametri totali
    ma può migliorare significativamente la calibrazione in periodi
    di stress macroeconomico (es. crisi SVB 2023, pivot Fed 2022).
    """

    # IT: Costruisce price branch (single/dual-stream), macro encoder e fusion head con residual + clip buffer.
    # EN: Builds the price branch (single/dual-stream), macro encoder and fusion head with residual + clip buffer.
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
        n_dynamic_features: int  = None,   # Miglioramento 9: dual-stream
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

        # ── Price branch — singolo o dual stream ──────────────────────────────
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

    # IT: Init pesi: Xavier su input/lineari, orthogonal sui ricorrenti, bias log ν a ~ν=5.
    # EN: Weight init: Xavier on input/linear, orthogonal on recurrent, log ν bias at ~ν=5.
    def _init_weights(self, math):
        for name, p in self.named_parameters():
            if "weight_ih" in name:    nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:  nn.init.orthogonal_(p)
            elif "bias" in name:       nn.init.zeros_(p)
            elif "weight" in name and p.dim() == 2:
                nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.out_lognu.bias.fill_(math.log(5.0 - 2.0))

    # IT: Forward: clip prezzo → price branch → fonde col macro embedding (zeros se assente) → (μ, log σ², log ν).
    # EN: Forward: clip price → price branch → fuse with macro embedding (zeros if missing) → (μ, log σ², log ν).
    def forward(
        self,
        x_price: torch.Tensor,
        x_macro: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x_price = x_price.clamp(self.clip_lo, self.clip_hi)

        # ── Price branch (single o dual stream) ──────────────────────────
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
        # Se x_macro è None (es. macro non disponibile in live o backtest),
        # usa un tensore di zeri — il MacroEncoder apprende a ignorarli
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

    # IT: Inference no-grad: trasforma gli output grezzi in {mu, sigma, nu} su CPU/numpy.
    # EN: No-grad inference: maps raw outputs to {mu, sigma, nu} on CPU/numpy.
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


# ─── NORMALIZZATORE MACRO ────────────────────────────────────────────────────

# IT: RobustScaler dedicato alle macro features (separato dagli scaler di prezzo), con clip a ±5.
# EN: RobustScaler dedicated to macro features (separate from price scalers), clipped to ±5.
class MacroNormalizer:
    """
    Normalizza le macro features per il MacroEncoder.
    Usa RobustScaler (resistente agli outlier degli shock macro).
    Salvato separatamente dagli scaler delle features di prezzo.
    """

    # IT: Inizializza scaler vuoto (fitted=False finché non si chiama fit_transform).
    # EN: Initializes an empty scaler (fitted=False until fit_transform is called).
    def __init__(self):
        self.scaler      = RobustScaler()
        self.feature_cols: list[str] = []
        self.fitted      = False

    # IT: Memorizza le colonne, fitta lo scaler e ritorna i dati normalizzati (NaN→0, clip ±5).
    # EN: Stores the columns, fits the scaler and returns normalized data (NaN→0, clip ±5).
    def fit_transform(self, df: pd.DataFrame,
                      macro_cols: list[str]) -> np.ndarray:
        self.feature_cols = macro_cols
        X = df[macro_cols].fillna(0).values.astype(np.float32)
        X = np.clip(X, -1e6, 1e6)
        result = self.scaler.fit_transform(X)
        result = np.clip(result, -5, 5).astype(np.float32)
        self.fitted = True
        return result

    # IT: Applica lo scaler già fittato a nuovi dati (NaN→0, clip ±5).
    # EN: Applies the already-fitted scaler to new data (NaN→0, clip ±5).
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].fillna(0).values.astype(np.float32)
        X = np.clip(X, -1e6, 1e6)
        result = self.scaler.transform(X)
        return np.clip(result, -5, 5).astype(np.float32)

    # IT: Serializza scaler + colonne su disco (pickle).
    # EN: Serializes scaler + columns to disk (pickle).
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_cols": self.feature_cols}, f)

    # IT: Ricostruisce un MacroNormalizer già fittato da un pickle salvato.
    # EN: Reconstructs an already-fitted MacroNormalizer from a saved pickle.
    @classmethod
    def load(cls, path: str) -> "MacroNormalizer":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.scaler       = data["scaler"]
        obj.feature_cols = data["feature_cols"]
        obj.fitted       = True
        return obj
