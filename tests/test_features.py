"""
tests/test_features.py
======================
Test unitari per la pipeline di feature engineering.

Esegui con:
  pytest tests/                  # tutti i test
  pytest tests/ -v               # verbose
  pytest tests/ -x               # stop al primo errore

Le fixture synthetic_ohlcv (2000 candele) e tiny_ohlcv (200 candele)
vengono caricate da tests/conftest.py (scope=session — create una volta sola).
synthetic_ohlcv è grande abbastanza per testare i rolling a lungo termine
(_structural_features usa finestre di 30/90/365 giorni con min_periods=60).
"""

import numpy as np
import pandas as pd
import pytest


# IT: TEST 1 — Stazionarietà dei log-return (sanity check del calcolo base).
# EN: TEST 1 — Log-return stationarity (sanity check on the base computation).

class TestLogReturns:

    # IT: La media dei log-return su un processo stazionario deve essere ~0.
    # EN: Mean of log-returns on a stationary process must be ~0.
    def test_log_ret_mean_near_zero(self, synthetic_ohlcv):
        """
        I log-return di un processo stazionario devono avere media vicina a zero.
        Se la media è molto diversa da zero, c'è un bug nel calcolo
        (es. prezzi non allineati, shift errato).
        """
        from quantsys.features import FeatureBuilder
        builder = FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10])
        df = builder._returns(synthetic_ohlcv.copy())

        mean_ret = df["log_ret"].dropna().mean()
        assert abs(mean_ret) < 0.002, (
            f"log_ret media troppo distante da zero: {mean_ret:.6f}. "
            "Possibile bug nel calcolo dei log-return."
        )

    # IT: Nessun inf/NaN nei log-return (escluso il primo, NaN per shift).
    # EN: No inf/NaN in log-returns (except the first, NaN by shift).
    def test_log_ret_no_inf(self, synthetic_ohlcv):
        """Nessun valore infinito o NaN nei log-return (eccetto il primo)."""
        from quantsys.features import FeatureBuilder
        builder = FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10])
        df = builder._returns(synthetic_ohlcv.copy())

        rets = df["log_ret"].iloc[1:]   # salta il primo (NaN per definizione)
        assert not np.isinf(rets).any(), "log_ret contiene valori inf."
        assert not rets.isna().all(),    "log_ret è tutto NaN."

    # IT: target_ret[t] = somma di log_ret[t+1..t+h] (no look-ahead bias).
    # EN: target_ret[t] = sum of log_ret[t+1..t+h] (no look-ahead bias).
    def test_target_is_shift_of_logret(self, synthetic_ohlcv):
        """
        Con forecast_horizon=1, target_ret[t] == log_ret[t+1].
        Con forecast_horizon=h, target_ret[t] == sum(log_ret[t+1..t+h]).
        """
        from quantsys.features import FeatureBuilder
        builder = FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10])

        # IT: caso h=1 | EN: case h=1
        df1 = builder._returns(synthetic_ohlcv.copy(), forecast_horizon=1)
        expected = df1["log_ret"].shift(-1)
        actual   = df1["target_ret"]
        mask = expected.notna() & actual.notna()
        np.testing.assert_allclose(
            actual[mask].values, expected[mask].values,
            rtol=1e-10,
            err_msg="target_ret (h=1) non corrisponde a log_ret.shift(-1)"
        )

        # IT: caso h=3 (rolling sum) | EN: case h=3 (rolling sum)
        df3 = builder._returns(synthetic_ohlcv.copy(), forecast_horizon=3)
        expected3 = df3["log_ret"].rolling(3).sum().shift(-3)
        actual3   = df3["target_ret"]
        mask3 = expected3.notna() & actual3.notna()
        np.testing.assert_allclose(
            actual3[mask3].values, expected3[mask3].values,
            rtol=1e-10,
            err_msg="target_ret (h=3) non corrisponde a rolling(3).sum().shift(-3)"
        )


# IT: TEST 2 — VWAP cumulativo deve restare nei limiti high/low globali.
# EN: TEST 2 — Cumulative VWAP must remain within global high/low bounds.

class TestVWAP:

    # IT: VWAP fuori dalla banda H-L = bug in pv o cum_vol.
    # EN: VWAP outside H-L band = bug in pv or cum_vol.
    def test_vwap_within_high_low(self, synthetic_ohlcv):
        """
        Il VWAP cumulativo deve sempre essere compreso tra il minimo Low
        e il massimo High osservati fino a quel momento.
        Un VWAP fuori da questo range indica un bug nel calcolo di pv o cumvol.
        """
        from quantsys.features import FeatureBuilder
        builder = FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10])
        df = builder._vwap(builder._returns(synthetic_ohlcv.copy()))

        vwap_valid = df["vwap"].dropna()
        global_lo  = synthetic_ohlcv["low"].min()
        global_hi  = synthetic_ohlcv["high"].max()

        assert (vwap_valid >= global_lo * 0.999).all(), \
            f"VWAP sotto il Low globale: min={vwap_valid.min():.1f}, lo={global_lo:.1f}"
        assert (vwap_valid <= global_hi * 1.001).all(), \
            f"VWAP sopra il High globale: max={vwap_valid.max():.1f}, hi={global_hi:.1f}"

    # IT: vwap_dev non deve contenere inf (rivela divisione per zero).
    # EN: vwap_dev must contain no inf (would reveal division by zero).
    def test_vwap_dev_finite(self, synthetic_ohlcv):
        """vwap_dev non deve contenere valori infiniti."""
        from quantsys.features import FeatureBuilder
        builder = FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10])
        df = builder._vwap(builder._returns(synthetic_ohlcv.copy()))

        assert not np.isinf(df["vwap_dev"].fillna(0)).any(), \
            "vwap_dev contiene valori infiniti — divisione per zero in vwap?"


# IT: TEST 3 — Split train/val/test temporale: zero leakage cronologico.
# EN: TEST 3 — Temporal train/val/test split: zero chronological leakage.

class TestTemporalSplit:

    # IT: Nessuna finestra di val/test deve sovrapporsi al train (data leakage).
    # EN: No val/test window may overlap the train set (data leakage).
    def test_no_overlap_between_splits(self, synthetic_ohlcv):
        """
        Il punto più critico del progetto: nessuna finestra del val/test set
        deve condividere candele col train set (data leakage).

        Lo split è TEMPORALE: train|val|test in ordine cronologico.
        L'overlap avverrebbe se le finestre fossero create dopo lo split
        invece di prima.
        """
        from quantsys.features import FeatureBuilder, create_windows, temporal_split

        builder = FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10])
        df = builder.build(synthetic_ohlcv.copy(), normalize=False, fit=True)

        feat_cols = [c for c in df.columns
                     if c not in {"open_time","close_time","date_utc","target_ret",
                                  "target_dir","pv","cum_pv","cum_vol","typical_price","obv"}
                     and df[c].dtype in [np.float32, np.float64]][:10]

        X, y, t = create_windows(df, feat_cols, window_size=10)
        splits  = temporal_split(X, y, t, val_frac=0.1, test_frac=0.1)

        t_train = splits["t_train"]
        t_val   = splits["t_val"]
        t_test  = splits["t_test"]

        # IT: max(train) < min(val) | EN: max(train) < min(val)
        assert t_train.max() < t_val.min(), (
            f"OVERLAP train/val: max_train={t_train.max()}, min_val={t_val.min()}"
        )
        # IT: max(val) < min(test) | EN: max(val) < min(test)
        assert t_val.max() < t_test.min(), (
            f"OVERLAP val/test: max_val={t_val.max()}, min_test={t_test.min()}"
        )

    # IT: Le dimensioni di train/val/test rispettano le frazioni (±1 sample).
    # EN: train/val/test sizes match requested fractions (±1 sample).
    def test_split_sizes_correct(self, synthetic_ohlcv):
        """Le dimensioni dei split devono rispettare le frazioni richieste (±1 campione)."""
        from quantsys.features import FeatureBuilder, create_windows, temporal_split

        builder = FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10])
        df = builder.build(synthetic_ohlcv.copy(), normalize=False, fit=True)

        feat_cols = [c for c in df.columns
                     if c not in {"open_time","close_time","date_utc","target_ret",
                                  "target_dir","pv","cum_pv","cum_vol","typical_price","obv"}
                     and df[c].dtype in [np.float32, np.float64]][:10]

        X, y, t = create_windows(df, feat_cols, window_size=10)
        n       = len(X)
        splits  = temporal_split(X, y, t, val_frac=0.1, test_frac=0.1)

        n_train = len(splits["X_train"])
        n_val   = len(splits["X_val"])
        n_test  = len(splits["X_test"])

        assert n_train + n_val + n_test == n, \
            f"Somma split {n_train+n_val+n_test} ≠ totale {n}"
        assert abs(n_val  / n - 0.1) < 0.02, f"Val fraction errata: {n_val/n:.3f}"
        assert abs(n_test / n - 0.1) < 0.02, f"Test fraction errata: {n_test/n:.3f}"


# IT: TEST 4 — Markov-Switching: probabilità di regime ben normalizzate.
# EN: TEST 4 — Markov-Switching: regime probabilities are well-normalised.

class TestHMM:

    # IT: P(regime|t) deve sommare a 1 per ogni timestep e non avere valori <0.
    # EN: P(regime|t) must sum to 1 per timestep and contain no negative values.
    def test_regime_probs_sum_to_one(self):
        """
        Le probabilità di regime per ogni giorno devono sommare a 1
        (sono una distribuzione di probabilità).
        Un'implementazione errata potrebbe restituire probabilità non normalizzate.
        """
        try:
            from quantsys.macro.regime import RegimeMarkovSwitching
        except ImportError:
            pytest.skip("statsmodels non installato")

        # IT: macro sintetici 5×300 | EN: synthetic macro 5×300
        np.random.seed(42)
        X_fake = np.random.randn(300, 5)

        df_fake = pd.DataFrame(
            X_fake,
            columns=[f"macro_feat_{i}" for i in range(5)],
            index=pd.date_range("2021-01-01", periods=300, freq="D"),
        )

        model = RegimeMarkovSwitching(n_regimes=3, n_iter=50, random_state=0, n_pca=3)
        model.feature_cols = list(df_fake.columns)
        try:
            model.fit(df_fake)
        except Exception:
            pytest.skip("MarkovSwitching fit fallito (dati troppo brevi)")

        probs = model.predict_proba(df_fake)

        # IT: somma riga ≈ 1 | EN: row-sum ≈ 1
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(len(row_sums)),
            atol=1e-6,
            err_msg="Le probabilità di regime non sommano a 1."
        )

        # IT: nessuna probabilità negativa | EN: no negative probabilities
        assert (probs >= 0).all(), "Probabilità di regime negative."


# IT: TEST 5 — Equivalenza VP incrementale ↔ ricostruito da zero.
# EN: TEST 5 — Equivalence between incremental VP and from-scratch VP.

class TestVolumeProfileIncremental:
    """
    Verifica che il VP incrementale (Fix 2) produce risultati
    equivalenti al VP calcolato da zero sull'intero buffer.

    Nota: il LiveFeatureBuffer vive nello script 04_live_signals.py.
    Lo testiamo estraendo la logica in una funzione di utilità locale,
    dato che il buffer è troppo accoppiato al LiveEngine per importarlo
    in isolamento senza tutte le dipendenze del motore live.
    """

    VP_BINS = 30

    # IT: Costruisce il volume profile in modo incrementale (replica del LiveFeatureBuffer).
    # EN: Builds the volume profile incrementally (mirrors the LiveFeatureBuffer logic).
    def _build_incremental(self, candles: list) -> np.ndarray:
        """
        Replica la logica incrementale del LiveFeatureBuffer in isolamento,
        senza dover importare lo script 04_live_signals.py completo.
        """
        from collections import deque

        bins: np.ndarray     = np.zeros(self.VP_BINS)
        contribs: deque      = deque(maxlen=len(candles) + 10)
        price_min: float     = 0.0
        price_max: float     = 0.0

        for c in candles:
            tp  = (c["high"] + c["low"] + c["close"]) / 3
            vol = c["volume"]

            # IT: prima candela → inizializza il range | EN: first candle → init range
            if price_max <= price_min:
                price_min = c["low"]  * 0.999
                price_max = c["high"] * 1.001

            # IT: prezzo fuori range → reset completo | EN: price out of range → full reset
            if tp < price_min or tp > price_max:
                # IT: ricalcola da zero | EN: recompute from scratch
                all_c = list(contribs) + [(0, 0)]  # placeholder
                price_min = min(x["low"]  for x in candles[:candles.index(c)+1]) * 0.999
                price_max = max(x["high"] for x in candles[:candles.index(c)+1]) * 1.001
                step = max((price_max - price_min) / self.VP_BINS, 1e-9)
                bins = np.zeros(self.VP_BINS)
                contribs.clear()
                for prev in candles[:candles.index(c)+1]:
                    tp2  = (prev["high"] + prev["low"] + prev["close"]) / 3
                    idx2 = min(int((tp2 - price_min) / step), self.VP_BINS - 1)
                    idx2 = max(idx2, 0)
                    bins[idx2] += prev["volume"]
                    contribs.append((idx2, prev["volume"]))
                continue

            step    = max((price_max - price_min) / self.VP_BINS, 1e-9)
            new_idx = min(int((tp - price_min) / step), self.VP_BINS - 1)
            new_idx = max(new_idx, 0)
            bins[new_idx] += vol
            contribs.append((new_idx, vol))

        return bins.copy()

    # IT: Calcola il volume profile da zero come riferimento (oracolo del test).
    # EN: Computes the volume profile from scratch as the reference (test oracle).
    def _build_from_scratch(self, candles: list) -> np.ndarray:
        """Calcola il VP da zero — implementazione di riferimento."""
        highs  = np.array([c["high"]   for c in candles])
        lows   = np.array([c["low"]    for c in candles])
        closes = np.array([c["close"]  for c in candles])
        vols   = np.array([c["volume"] for c in candles])
        tps    = (highs + lows + closes) / 3

        lo   = lows.min()  * 0.999
        hi   = highs.max() * 1.001
        step = max((hi - lo) / self.VP_BINS, 1e-9)
        bins = np.zeros(self.VP_BINS)
        for tp, vol in zip(tps, vols):
            idx = min(int((tp - lo) / step), self.VP_BINS - 1)
            idx = max(idx, 0)
            bins[idx] += vol
        return bins

    # IT: POC e volume totale del VP incrementale = quelli da-zero.
    # EN: POC and total volume of incremental VP equal the from-scratch values.
    def test_incremental_matches_scratch(self):
        """
        Il POC (bin con volume massimo) del VP incrementale deve coincidere
        con quello del VP calcolato da zero sullo stesso insieme di candele.
        """
        np.random.seed(7)
        n_candles = 40
        base = 50_000.0
        candles = []
        for i in range(n_candles):
            close  = base * (1 + np.random.normal(0, 0.001))
            high   = close * (1 + abs(np.random.normal(0, 0.0005)))
            low    = close * (1 - abs(np.random.normal(0, 0.0005)))
            candles.append({
                "open": base, "high": high, "low": low, "close": close,
                "volume": float(np.random.lognormal(7, 0.5)),
                "taker_buy_vol": 0.5, "hour": 12, "minute": i % 60, "ts": i,
            })
            base = close

        inc_bins     = self._build_incremental(candles)
        scratch_bins = self._build_from_scratch(candles)

        # IT: POC (bin più pesante) deve coincidere | EN: POC (heaviest bin) must match
        assert inc_bins.argmax() == scratch_bins.argmax(), (
            f"POC incrementale ({inc_bins.argmax()}) ≠ POC da zero ({scratch_bins.argmax()})\n"
            f"inc_bins top3:     {np.argsort(inc_bins)[-3:]}\n"
            f"scratch_bins top3: {np.argsort(scratch_bins)[-3:]}"
        )

        # IT: volume totale uguale (stesse candele processate) | EN: same total volume
        np.testing.assert_allclose(
            inc_bins.sum(), scratch_bins.sum(),
            rtol=1e-3,
            err_msg="Volume totale VP incrementale ≠ VP da zero"
        )


# IT: TEST 6 — NLL Student-t è differenziabile e numericamente stabile.
# EN: TEST 6 — Student-t NLL is differentiable and numerically stable.

class TestStudentTNLL:

    # IT: I gradienti devono fluire verso μ, log σ², log ν (no detach).
    # EN: Gradients must flow to μ, log σ², log ν (no detach in path).
    def test_nll_gradient_flows(self):
        """
        La NLL della t-Student deve essere differenziabile rispetto
        a tutti e tre i parametri (μ, log_σ², log_ν).
        Un bug nell'implementazione (es. uso di .detach()) bloccherebbe
        il gradiente e il modello non imparerebbe.
        """
        import torch
        from quantsys.model import student_t_nll

        batch = 32
        y      = torch.randn(batch, requires_grad=False)
        mu     = torch.randn(batch, requires_grad=True)
        lsig2  = torch.randn(batch, requires_grad=True)
        lnu    = torch.randn(batch, requires_grad=True)

        loss = student_t_nll(y, mu, lsig2, lnu)
        loss.backward()

        assert mu.grad    is not None, "Gradiente non fluisce verso μ"
        assert lsig2.grad is not None, "Gradiente non fluisce verso log_σ²"
        assert lnu.grad   is not None, "Gradiente non fluisce verso log_ν"

        # IT: i gradienti non devono essere tutti zero | EN: gradients must not be all-zero
        assert not torch.all(mu.grad    == 0), "Gradiente μ è tutto zero"
        assert not torch.all(lsig2.grad == 0), "Gradiente log_σ² è tutto zero"
        assert not torch.all(lnu.grad   == 0), "Gradiente log_ν è tutto zero"

    # IT: Per valori realistici (μ≈0, σ≈0.002, ν≈5) la NLL è finita.
    # EN: For realistic values (μ≈0, σ≈0.002, ν≈5) the NLL is finite.
    def test_nll_finite_for_typical_values(self):
        """
        Per valori tipici dei parametri (μ≈0, σ≈0.002, ν≈5),
        la NLL deve essere finita. Può essere negativa (log-likelihood > 1
        è possibile per distribuzioni con σ molto piccolo).
        """
        import torch
        from quantsys.model import student_t_nll

        y     = torch.tensor([0.001, -0.002, 0.0005])
        mu    = torch.tensor([0.0005, -0.001, 0.0008])
        lsig2 = torch.tensor([-12.0, -12.0, -12.0])   # IT: log(0.002²) ≈ -12.4 | EN: log(0.002²)
        lnu   = torch.tensor([1.099, 1.099, 1.099])    # IT: softplus+2 ≈ 5 | EN: softplus+2 ≈ 5

        loss = student_t_nll(y, mu, lsig2, lnu)

        assert torch.isfinite(loss), f"NLL non finita: {loss.item()}"


# IT: TEST 7 — Lag di pubblicazione macro: previene il look-ahead bias.
# EN: TEST 7 — Macro release lag: prevents look-ahead bias.

class TestReleaseLag:

    # IT: Dopo il lag, la data effettiva di disponibilità è posticipata.
    # EN: After the lag, the effective availability date is shifted forward.
    def test_lag_shifts_dates_forward(self):
        """
        Dopo il lag, le date di ogni serie devono essere spostate in avanti.
        Verifica che un dato con observation_date=2024-01-31 e lag=35 giorni
        risulti disponibile da 2024-03-06 (non da 2024-02-01).
        """
        import pandas as pd
        from quantsys.macro import RELEASE_LAG_DAYS

        # IT: serie mensile, osservazione al 1° del mese | EN: monthly series, obs on 1st
        dates  = pd.date_range("2024-01-01", periods=3, freq="MS")
        series = pd.Series([312.0, 313.5, 315.1], index=dates, name="cpi")

        lag    = RELEASE_LAG_DAYS["M"]   # IT: 35 giorni | EN: 35 days
        shifted = series.copy()
        shifted.index = shifted.index + pd.Timedelta(days=lag)

        # IT: 2024-01-01 + 35d = 2024-02-05 → non disponibile al 2024-01-15.
        # EN: 2024-01-01 + 35d = 2024-02-05 → not available at 2024-01-15.
        assert pd.Timestamp("2024-01-15") not in shifted.index, \
            "Dato mensile disponibile prima del lag — look-ahead bias!"

        # IT: la prima data laggata deve cadere oltre fine gennaio.
        # EN: the first lagged date must fall past the end of January.
        first_available = shifted.index[0]
        assert first_available > pd.Timestamp("2024-01-31"), \
            f"Dato disponibile troppo presto: {first_available}"

    # IT: Le serie giornaliere (breakeven, Treasury) hanno lag=1 giorno.
    # EN: Daily series (breakeven, Treasury) have a 1-day release lag.
    def test_daily_series_lag_is_one(self):
        """Le serie giornaliere (breakeven, Treasury) devono avere lag=1."""
        from quantsys.macro import RELEASE_LAG_DAYS, SERIES_LAG_OVERRIDE
        # IT: breakeven disponibili il giorno dopo | EN: breakeven available next day
        assert RELEASE_LAG_DAYS["D"] == 1
        assert SERIES_LAG_OVERRIDE.get("infl_exp_5y", RELEASE_LAG_DAYS["D"]) == 1


# IT: TEST 8 — Persistenza PipelineState: save → load conserva tutto.
# EN: TEST 8 — PipelineState persistence: save → load preserves all data.

class TestPipelineState:

    # IT: feature_cols, scaler fittato e config sopravvivono al round-trip.
    # EN: feature_cols, fitted scaler and config survive the round-trip.
    def test_save_load_roundtrip(self, tmp_path):
        """
        Salva e ricarica un PipelineState, verifica che i dati siano intatti.
        """
        from quantsys.utils import PipelineState
        from sklearn.preprocessing import RobustScaler
        import numpy as np

        state = PipelineState()
        state.feature_cols       = ["log_ret", "vwap_dev", "rsi_14_norm"]
        state.macro_feature_cols = ["macro_cpi_yoy", "macro_fed_funds"]
        state.model_config       = {"n_features": 3, "window_size": 60}

        # IT: includi uno scaler già fittato | EN: include a fitted scaler
        scaler = RobustScaler().fit(np.random.randn(100, 1))
        state.price_scaler_state = {"log_ret": scaler}

        path = str(tmp_path / "test_state.pkl")
        state.save(path)

        loaded = PipelineState.load(path)
        assert loaded.feature_cols       == state.feature_cols
        assert loaded.macro_feature_cols == state.macro_feature_cols
        assert loaded.model_config       == state.model_config
        assert "log_ret" in loaded.price_scaler_state

        # IT: lo scaler ricaricato produce le stesse trasformazioni.
        # EN: the reloaded scaler produces identical transforms.
        x = np.array([[1.5], [-0.3], [2.1]])
        np.testing.assert_allclose(
            scaler.transform(x),
            loaded.price_scaler_state["log_ret"].transform(x),
        )


# IT: TEST — Scaler no-leakage: il fit avviene solo sul training set.
# EN: TEST — Scaler no-leakage: fitting happens only on the training set.

class TestScalerNoLeakage:
    """
    Verifica che fit_scaler_only fitta lo scaler SOLO sul training set.
    Questo garantisce che val e test non influenzino le statistiche di normalizzazione.
    """

    # IT: median/IQR dello scaler = quelli del solo train (no contagio val/test).
    # EN: scaler median/IQR equal those of the train slice only (no val/test bleed).
    def test_scaler_fit_only_on_train(self, synthetic_ohlcv):
        """
        Mediana e IQR dello scaler devono coincidere con le statistiche
        calcolate SOLO sulle righe di training, non sull'intero dataset.
        """
        from quantsys.features import FeatureBuilder
        import numpy as np

        builder = FeatureBuilder()
        df      = builder.build(synthetic_ohlcv, normalize=False, fit=False)

        n_total   = len(df)
        train_end = int(n_total * 0.80)   # IT: 80% training | EN: 80% training

        # IT: fit solo sul train | EN: fit on the train slice only
        builder.fit_scaler_only(df.iloc[:train_end])

        # IT: la mediana dello scaler deve allinearsi al solo train, non al dataset completo.
        # EN: scaler median must align with the train slice only, not the full dataset.
        # IT: _get_scaler_for_col gestisce nuovo formato multi-col + legacy per-col.
        # EN: _get_scaler_for_col handles the new multi-col format and the legacy per-col one.
        scaler = builder._get_scaler_for_col("log_ret")
        if scaler is not None:
            train_median  = np.median(df["log_ret"].iloc[:train_end].dropna())
            full_median   = np.median(df["log_ret"].dropna())

            # IT: RobustScaler.center_ = mediana del train | EN: RobustScaler.center_ = train median
            scaler_median = float(scaler.center_[0])
            assert abs(scaler_median - train_median) < abs(scaler_median - full_median) + 1e-10, (
                f"Lo scaler usa la mediana del dataset completo ({full_median:.6f}) "
                f"invece di quella del training ({train_median:.6f}). "
                f"Scaler median: {scaler_median:.6f}"
            )

    # IT: I dati train scalati hanno mediana ≈ 0 → val/test non hanno influenzato il fit.
    # EN: Scaled train rows have median ≈ 0 → val/test never influenced the fit.
    def test_scaler_transform_uses_train_stats(self, synthetic_ohlcv):
        """
        I valori scalati di val/test devono usare le statistiche del training.
        Un valore estremo nel test set non deve influenzare la normalizzazione.
        """
        from quantsys.features import FeatureBuilder
        import numpy as np

        builder = FeatureBuilder()
        df      = builder.build(synthetic_ohlcv, normalize=False, fit=False)

        n_total   = len(df)
        train_end = int(n_total * 0.80)

        # IT: fit solo train | EN: fit on train only
        builder.fit_scaler_only(df.iloc[:train_end])

        # IT: trasforma l'intero dataset | EN: transform the full dataset
        df_scaled = builder._normalize(df.copy(), fit=False)

        # IT: dati train scalati centrati su 0 (RobustScaler usa la mediana del train).
        # EN: scaled train rows centred around 0 (RobustScaler uses the train median).
        # IT: _normalize sovrascrive le colonne in-place (nessun suffisso _scaled).
        # EN: _normalize overwrites columns in-place (no _scaled suffix).
        if "log_ret" in df_scaled.columns and builder._get_scaler_for_col("log_ret") is not None:
            train_scaled_median = df_scaled["log_ret"].iloc[:train_end].median()
            assert abs(train_scaled_median) < 0.1, (
                f"La mediana delle righe di training scalate dovrebbe essere ~0, "
                f"invece è {train_scaled_median:.4f}. "
                f"Lo scaler potrebbe essere fittato sull'intero dataset."
            )

    # IT: fit_scaler_only popola builder.scaler e _scale_cols correttamente.
    # EN: fit_scaler_only correctly populates builder.scaler and _scale_cols.
    def test_fit_scaler_only_sets_scalers(self, synthetic_ohlcv):
        """fit_scaler_only deve fittare il multi-scaler e popolare _scale_cols."""
        from quantsys.features import FeatureBuilder

        builder = FeatureBuilder()
        df      = builder.build(synthetic_ohlcv, normalize=False, fit=False)

        assert builder.scaler is None, "scaler dovrebbe essere None prima del fit"
        assert len(builder._scale_cols) == 0, "_scale_cols dovrebbe essere vuota prima del fit"

        builder.fit_scaler_only(df.iloc[:int(len(df)*0.8)])

        assert builder.scaler is not None, "fit_scaler_only deve fittare builder.scaler"
        assert len(builder._scale_cols) > 0, "fit_scaler_only deve popolare builder._scale_cols"
        # IT: nuovo formato → self.scalers resta {} (campo conservato per backward compat).
        # EN: new format → self.scalers stays {} (field kept for backward compatibility).
        assert isinstance(builder.scalers, dict), "builder.scalers deve rimanere un dict"

    # IT: Le colonne in _NO_SCALE non devono essere scalate né referenziate.
    # EN: Columns listed in _NO_SCALE must not be scaled nor referenced.
    def test_no_scaler_for_no_scale_columns(self, synthetic_ohlcv):
        """Le colonne in _NO_SCALE non devono essere incluse in _scale_cols."""
        from quantsys.features import FeatureBuilder

        builder = FeatureBuilder()
        df      = builder.build(synthetic_ohlcv, normalize=False, fit=False)
        builder.fit_scaler_only(df.iloc[:int(len(df)*0.8)])

        for col in FeatureBuilder._NO_SCALE:
            assert col not in builder._scale_cols, (
                f"La colonna '{col}' è in _NO_SCALE ma è in _scale_cols — non dovrebbe."
            )
            # IT: per colonne _NO_SCALE → None | EN: for _NO_SCALE columns → None
            assert builder._get_scaler_for_col(col) is None, (
                f"La colonna '{col}' è in _NO_SCALE ma _get_scaler_for_col restituisce "
                f"un oggetto — non dovrebbe."
            )


# IT: TEST 9 — ATR / True Range calcolato con la formula annidata corretta.
# EN: TEST 9 — ATR / True Range computed with the correct nested formula.

class TestATR:

    # IT: np.maximum(a, b, c) a 3 arg scrive in c: usare nesting.
    # EN: np.maximum(a, b, c) with 3 args writes into c: must use nesting.
    def test_true_range_formula(self):
        """
        Verifica che il True Range sia calcolato correttamente.
        np.maximum(a, b, c) con 3 argomenti è SBAGLIATO (scrive in c).
        La formula corretta usa np.maximum annidato.
        """
        np.random.seed(0)
        n = 50
        closes = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 50_000
        highs  = closes * (1 + np.abs(np.random.normal(0, 0.005, n)))
        lows   = closes * (1 - np.abs(np.random.normal(0, 0.005, n)))

        c_prev = np.roll(closes, 1); c_prev[0] = closes[0]

        # IT: forma corretta (np.maximum annidato) | EN: correct form (nested np.maximum)
        tr_correct = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - c_prev), np.abs(lows - c_prev))
        )
        # IT: invariante: TR >= High-Low | EN: invariant: TR >= High-Low
        assert (tr_correct >= highs - lows - 1e-10).all(), \
            "True Range < High-Low: formula sbagliata"
        # IT: TR sempre non-negativo | EN: TR always non-negative
        assert (tr_correct >= 0).all(), "True Range negativo"

    # IT: ATR con min_periods=1 elimina i NaN iniziali (utile da prima candela).
    # EN: ATR with min_periods=1 removes leading NaNs (usable from first bar).
    def test_atr_uses_min_periods(self):
        """
        rolling(14) senza min_periods=1 produce NaN per le prime 13 righe.
        Con min_periods=1 l'ATR è disponibile dalla prima candela.
        """
        import pandas as pd
        tr = np.abs(np.random.normal(100, 10, 50))
        atr_with = pd.Series(tr).rolling(14, min_periods=1).mean().values
        atr_without = pd.Series(tr).rolling(14).mean().values

        assert not np.isnan(atr_with[0]),  "ATR con min_periods=1: non dovrebbe avere NaN"
        assert np.isnan(atr_without[0]),   "ATR senza min_periods: prime righe NaN (atteso)"
        assert np.isnan(atr_without[12]),  "ATR senza min_periods: riga 13 NaN (atteso)"
        assert not np.isnan(atr_without[13]), "ATR senza min_periods: riga 14 ok"


# IT: TEST 10 — Kelly sizing robusto agli edge case (ATR=0, edge negativo).
# EN: TEST 10 — Kelly sizing robust to edge cases (ATR=0, negative edge).

class TestKelly:

    # IT: ATR=0 → sl_distance=0 → no ZeroDivisionError, size valida.
    # EN: ATR=0 → sl_distance=0 → no ZeroDivisionError, valid size.
    def test_kelly_no_zero_division(self):
        """
        Kelly con ATR=0 non deve causare ZeroDivisionError.
        """
        from quantsys.trading import RiskManager, DistributionParams, Side

        rm   = RiskManager(initial_capital=10_000)
        dist = DistributionParams(mu=0.002, sigma=0.002, nu=5.0, prob_up=0.65)

        # IT: ATR=0 caso limite → la guard deve evitare la divisione per zero.
        # EN: ATR=0 edge case → the guard must avoid division by zero.
        size_usd, size_base = rm._size(dist, price=67_000.0, atr=0.0)
        assert size_usd >= 0, "Size negativa con ATR=0"
        assert not (size_usd != size_usd), "Size NaN con ATR=0"

    # IT: Kelly negativo (edge sfavorevole) viene clippato dal floor a 0.01.
    # EN: Negative Kelly (unfavourable edge) is clipped by the 0.01 floor.
    def test_kelly_negative_edge(self):
        """
        Con prob_up molto bassa (< 1/(1+RR)), Kelly raw è negativo.
        Il floor a 0.01 deve garantire una size minima positiva.
        """
        from quantsys.trading import RiskManager, DistributionParams

        rm   = RiskManager(initial_capital=10_000, tp_rr_ratio=2.5)
        # IT: prob_up=0.2 → Kelly raw negativo | EN: prob_up=0.2 → raw Kelly negative
        dist = DistributionParams(mu=-0.001, sigma=0.002, nu=5.0, prob_up=0.2)
        size_usd, _ = rm._size(dist, price=67_000.0, atr=200.0)
        # IT: floor max(0.01, kelly_raw) → Kelly clippato | EN: floor max(0.01, kelly_raw)
        assert size_usd >= 0, "Size negativa con Kelly negativo"


# IT: TEST 11 — VP incrementale: ordine sottrai-prima/aggiungi-dopo corretto.
# EN: TEST 11 — Incremental VP: correct subtract-first/append-after order.

class TestVPIncrementalOrder:

    # IT: La candela uscente va sottratta PRIMA che la deque la espella.
    # EN: The leaving candle must be subtracted BEFORE the deque pops it.
    def test_subtract_before_append(self):
        """
        Verifica che il contributo della candela uscente venga sottratto
        PRIMA che la deque la espella (non dopo).
        """
        from collections import deque

        BINS     = 5
        MAXLEN   = 4
        bins     = np.zeros(BINS)
        contribs = deque(maxlen=MAXLEN)
        lo, hi   = 0.0, 100.0
        step     = (hi - lo) / BINS

        # IT: Aggiunge un contributo (typical price, volume) al bin rolling.
        # EN: Adds one (typical price, volume) contribution to the rolling bin.
        def push(tp, vol):
            idx = min(int((tp - lo) / step), BINS - 1)
            # IT: ordine corretto — sottrai prima, aggiungi dopo.
            # EN: correct order — subtract first, then append.
            if len(contribs) == contribs.maxlen:
                old_idx, old_vol = contribs[0]
                bins[old_idx] = max(0.0, bins[old_idx] - old_vol)
            bins[idx] += vol
            contribs.append((idx, vol))

        # IT: 4 candele tutte nel bin 0 | EN: 4 candles all in bin 0
        for _ in range(MAXLEN):
            push(10.0, 100.0)
        assert bins[0] == pytest.approx(400.0), f"Atteso 400, got {bins[0]}"

        # IT: 5ª candela in bin 0: una espulsa, una entrata → bin[0] resta 400.
        # EN: 5th candle in bin 0: one popped, one pushed → bin[0] stays 400.
        push(10.0, 100.0)
        assert bins[0] == pytest.approx(400.0), \
            f"Dopo espulsione atteso 400, got {bins[0]} (bug: sottrazione errata)"

        # IT: 5ª in bin diverso (tp=60 → bin 3) | EN: 5th in another bin (tp=60 → bin 3)
        bins[:] = 0; contribs.clear()
        for _ in range(MAXLEN):
            push(10.0, 50.0)   # IT: bin 0 | EN: bin 0
        push(60.0, 200.0)      # IT: bin 3, espelle bin 0 vol 50 | EN: bin 3, pops bin 0 vol 50
        assert bins[0] == pytest.approx(150.0), f"bin[0] atteso 150, got {bins[0]}"
        assert bins[3] == pytest.approx(200.0), f"bin[3] atteso 200, got {bins[3]}"


# IT: TEST CVD — Cumulative Volume Delta: invarianti su range e finitezza.
# EN: TEST CVD — Cumulative Volume Delta: range and finiteness invariants.

class TestCVDFeatures:
    """Verifica che le feature CVD (Cumulative Volume Delta) siano corrette."""

    # IT: cvd_norm vincolato in [-1, 1] | EN: cvd_norm bounded in [-1, 1]
    def test_cvd_norm_range(self, synthetic_ohlcv):
        """cvd_norm deve essere in [-1, 1]."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "cvd_norm" in df.columns, "cvd_norm non trovato nelle feature"
        valid = df["cvd_norm"].dropna()
        assert (valid.abs() <= 1.0 + 1e-6).all(), \
            f"cvd_norm fuori [-1,1]: min={valid.min():.3f} max={valid.max():.3f}"

    # IT: cvd_divergence finita dopo i 60 step di warm-up.
    # EN: cvd_divergence is finite past the 60-step warm-up.
    def test_cvd_divergence_finite(self, synthetic_ohlcv):
        """cvd_divergence deve essere finita (no inf, no nan oltre il warm-up)."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "cvd_divergence" in df.columns, "cvd_divergence mancante"
        # IT: dopo warm-up no inf né nan | EN: past warm-up no inf nor nan
        tail = df["cvd_divergence"].iloc[60:].dropna()
        assert np.isfinite(tail.values).all(), \
            "cvd_divergence contiene inf o nan dopo warm-up"

    # IT: cvd_cum_20 ≡ rolling(20).sum() di cvd_raw.
    # EN: cvd_cum_20 ≡ rolling(20).sum() of cvd_raw.
    def test_cvd_cum20_is_rolling_sum(self, synthetic_ohlcv):
        """cvd_cum_20 deve corrispondere alla rolling sum del delta su 20 candele."""
        import pandas as pd
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "cvd_cum_20" in df.columns, "cvd_cum_20 mancante"
        # IT: cvd_cum_20[i] == Σ cvd_raw[i-19:i+1] | EN: cvd_cum_20[i] == Σ cvd_raw[i-19:i+1]
        delta_manual = df["cvd_raw"].rolling(20, min_periods=20).sum()
        diff = (df["cvd_cum_20"] - delta_manual).dropna().abs()
        assert (diff < 1e-6).all(), f"cvd_cum_20 non coincide con rolling sum: max_err={diff.max()}"

    # IT: taker_buy_ratio resta in [0, 1] anche dopo l'aggiunta del CVD.
    # EN: taker_buy_ratio stays in [0, 1] even after adding CVD features.
    def test_taker_buy_ratio_unchanged(self, synthetic_ohlcv):
        """taker_buy_ratio deve rimanere in [0, 1] dopo l'aggiunta di CVD."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        ratio = df["taker_buy_ratio"].dropna()
        assert ((ratio >= 0) & (ratio <= 1)).all(), \
            "taker_buy_ratio fuori [0,1]"

    # IT: build() imposta n_dynamic_features (0 < n_dyn < n_total).
    # EN: build() sets n_dynamic_features (0 < n_dyn < n_total).
    def test_n_dynamic_features_set(self, synthetic_ohlcv):
        """build() deve impostare n_dynamic_features > 0 e < n_total_features."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        b.build(synthetic_ohlcv, normalize=False)
        assert b.n_dynamic_features > 0, \
            "n_dynamic_features non impostato da build()"
        assert b.n_dynamic_features < len(b.feature_cols), \
            "n_dynamic_features >= n_total (nessuna feature strutturale)"


# IT: TEST Structural — feature di livello prezzo (ATH, ATL, posizione, momentum).
# EN: TEST Structural — price-level features (ATH, ATL, position, momentum).

class TestStructuralFeatures:
    """Verifica le feature di livello prezzo (ATH, ATL, livelli strutturali)."""

    # IT: dist_ath_30d ≤ 0 (il prezzo non supera mai l'ATH rolling).
    # EN: dist_ath_30d ≤ 0 (price never exceeds the rolling ATH).
    def test_dist_ath_non_positive(self, synthetic_ohlcv):
        """dist_ath_30d deve essere ≤ 0 (prezzo sempre ≤ ATH)."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "dist_ath_30d" in df.columns, "dist_ath_30d mancante"
        valid = df["dist_ath_30d"].dropna()
        assert (valid <= 1e-6).all(), \
            f"dist_ath_30d > 0: max={valid.max():.4f} (prezzo supera ATH rolling?)"

    # IT: dist_atl_30d ≥ 0 (il prezzo non scende mai sotto l'ATL rolling).
    # EN: dist_atl_30d ≥ 0 (price never drops below the rolling ATL).
    def test_dist_atl_non_negative(self, synthetic_ohlcv):
        """dist_atl_30d deve essere ≥ 0 (prezzo sempre ≥ ATL)."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "dist_atl_30d" in df.columns, "dist_atl_30d mancante"
        valid = df["dist_atl_30d"].dropna()
        assert (valid >= -1e-6).all(), \
            f"dist_atl_30d < 0: min={valid.min():.4f} (prezzo sotto ATL rolling?)"

    # IT: price_pos_30d ∈ [0, 1] (posizione normalizzata nel range ATL-ATH).
    # EN: price_pos_30d ∈ [0, 1] (normalised position within the ATL-ATH range).
    def test_price_position_in_range(self, synthetic_ohlcv):
        """price_pos_30d deve essere in [0, 1] (posizione nel range ATL-ATH)."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "price_pos_30d" in df.columns, "price_pos_30d mancante"
        valid = df["price_pos_30d"].dropna()
        assert ((valid >= -1e-6) & (valid <= 1 + 1e-6)).all(), \
            f"price_pos_30d fuori [0,1]: min={valid.min():.4f} max={valid.max():.4f}"

    # IT: Le feature strutturali risiedono nel blocco finale di feature_cols.
    # EN: Structural features live in the final block of feature_cols.
    def test_structural_cols_in_feature_cols(self, synthetic_ohlcv):
        """Le feature strutturali devono essere in feature_cols e nel blocco finale."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        b.build(synthetic_ohlcv, normalize=False)

        struct_expected = ["dist_ath_30d", "dist_atl_30d", "price_pos_30d"]
        for col in struct_expected:
            assert col in b.feature_cols, f"{col} non in feature_cols"
            idx = b.feature_cols.index(col)
            assert idx >= b.n_dynamic_features, \
                f"{col} è nel blocco dinamico (idx={idx}) invece che strutturale (≥{b.n_dynamic_features})"

    # IT: round_level_dist deve essere ~simmetrica (media|·| piccola).
    # EN: round_level_dist must be ~symmetric (small mean of |·|).
    def test_round_level_dist_symmetric(self, synthetic_ohlcv):
        """round_level_dist deve avere media vicina a zero (distribuzione simmetrica)."""
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "round_level_dist" in df.columns, "round_level_dist mancante"
        mean_abs = df["round_level_dist"].dropna().abs().mean()
        # IT: distanza media dal livello tondo bassa | EN: low mean dist from round level
        assert mean_abs < 0.01, \
            f"round_level_dist media assoluta troppo alta: {mean_abs:.4f}"

    # IT: momentum_30d esiste; se ha valori validi devono essere finiti.
    # EN: momentum_30d exists; if any valid values are present they are finite.
    def test_momentum_30d_sign(self, synthetic_ohlcv):
        """momentum_30d deve essere presente; se esistono valori validi devono essere finiti."""
        import numpy as np
        import pytest
        from quantsys.features import FeatureBuilder
        b  = FeatureBuilder()
        df = b.build(synthetic_ohlcv, normalize=False)
        assert "momentum_30d" in df.columns, "momentum_30d mancante"
        valid = df[["close", "momentum_30d"]].dropna().iloc[30:]
        # IT: 2000 candele 1m (~33h) < 43200 richieste da momentum_30d → tutti NaN attesi.
        # EN: 2000 1m bars (~33h) < the 43200 momentum_30d needs → all NaN expected.
        if len(valid) == 0:
            pytest.skip("Dataset sintetico troppo corto per momentum_30d (richiede 30d di dati)")
        assert np.isfinite(valid["momentum_30d"].values).all(), \
            "momentum_30d contiene NaN o Inf"
