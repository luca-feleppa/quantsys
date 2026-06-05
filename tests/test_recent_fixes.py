"""
IT: Regression test per i fix critici delle sessioni 2026-05-19, 23, 24, 28.
    Copre: denormalizzazione z-score, RevIN exclude raw-returns, BLOCKER #1 live engine.
EN: Regression tests for critical fixes from sessions 2026-05-19, 23, 24, 28.
    Covers: z-score denormalization, RevIN exclude raw-returns, BLOCKER #1 live engine.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from quantsys.features import FeatureBuilder
from quantsys.utils import PipelineState


ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# IT: Fix 2026-05-23 — denormalizzazione z-score (breakthrough Sharpe -256 -> +18.7)
# EN: Fix 2026-05-23 — z-score denormalization (Sharpe -256 -> +18.7 breakthrough)
# ─────────────────────────────────────────────────────────────────────────────
class TestZscoreDenormalization:
    """
    Bug strutturale risolto il 2026-05-23: il modello prediceva mu/sigma in spazio
    z-score (output del RobustScaler), ma il trading layer li interpretava come raw
    log-return. Risultato: SL/TP irrealistici, Sharpe -256 → +18.7 dopo il fix.
    """

    def _make_state(self, target_scale: float = 0.001, use_revin: bool = False) -> PipelineState:
        # IT: stub PipelineState con scaler che produce target_scale richiesto
        # EN: stub PipelineState whose scaler yields the requested target_scale
        state = PipelineState()
        scaler = RobustScaler()
        # IT: scaler fittato su [-target_scale, +target_scale] -> scale_ = target_scale*2
        #     per ottenere target_scale=0.001 fittiamo su un range simmetrico opportuno
        # EN: scaler fit such that scale_ matches target_scale
        x = np.array([[-target_scale, 0.0], [target_scale, 0.0]])
        scaler.fit(x)
        scaler.scale_ = np.array([target_scale, 1.0])
        state.scaler = scaler
        state.scale_cols = ["target_ret", "other_feature"]
        state.training_config = {"model": {"use_revin": use_revin}}
        return state

    def test_target_scale_reads_scaler_scale(self):
        state = self._make_state(target_scale=0.005)
        assert state.target_scale == pytest.approx(0.005)

    def test_target_scale_safe_default_when_no_scaler(self):
        state = PipelineState()
        # IT: senza scaler restituisce 1.0 (no-op safe)
        assert state.target_scale == 1.0

    def test_target_scale_safe_default_when_target_not_in_scale_cols(self):
        state = PipelineState()
        scaler = RobustScaler()
        scaler.fit(np.array([[0.0], [1.0]]))
        state.scaler = scaler
        state.scale_cols = ["other"]   # IT: target_ret assente | EN: target_ret missing
        assert state.target_scale == 1.0

    def test_denormalize_zscore_path_multiplies_by_target_scale(self):
        # IT: percorso senza RevIN: mu_raw = mu_z * target_scale
        # EN: non-RevIN path: mu_raw = mu_z * target_scale
        state = self._make_state(target_scale=0.001, use_revin=False)
        mu_z, sigma_z = 0.5, 1.2
        mu_raw, sigma_raw = state.denormalize_predictions(mu_z, sigma_z)
        assert mu_raw == pytest.approx(0.5 * 0.001)
        assert sigma_raw == pytest.approx(1.2 * 0.001)

    def test_denormalize_revin_is_noop(self):
        # IT: con RevIN attivo, denormalize_predictions deve essere identita'
        # EN: with RevIN on, denormalize_predictions must be identity
        state = self._make_state(target_scale=0.001, use_revin=True)
        mu_in, sigma_in = 0.5, 1.2
        mu_out, sigma_out = state.denormalize_predictions(mu_in, sigma_in)
        assert mu_out == mu_in and sigma_out == sigma_in

    def test_denormalize_preserves_ndarray_type(self):
        state = self._make_state(target_scale=0.002)
        mu = np.array([0.1, -0.2, 0.3])
        sigma = np.array([1.0, 1.0, 2.0])
        mu_r, sigma_r = state.denormalize_predictions(mu, sigma)
        assert isinstance(mu_r, np.ndarray)
        np.testing.assert_allclose(mu_r, mu * 0.002)
        np.testing.assert_allclose(sigma_r, sigma * 0.002)

    def test_denormalize_idempotent_with_unit_scale(self):
        # IT: target_scale=1.0 (fallback) -> denormalize e' no-op
        # EN: target_scale=1.0 (fallback) -> denormalize is a no-op
        state = PipelineState()
        mu, sigma = state.denormalize_predictions(0.5, 1.0)
        assert mu == 0.5 and sigma == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# IT: Fix 2026-05-19 — RevIN exclude raw-returns dal RobustScaler globale
# EN: Fix 2026-05-19 — RevIN excludes raw-returns from the global RobustScaler
# ─────────────────────────────────────────────────────────────────────────────
class TestRevinExcludeRawReturns:
    """
    Fix concettuale 2026-05-19: con RevIN attivo, i log_ret/lag_ret raw devono
    essere esclusi dal RobustScaler globale, altrimenti RevIN opera su feature
    gia' scalate e la denormalizzazione predice in spazio z anziche' raw.
    """

    def test_default_no_revin_excludes_only_natural_scale_cols(self):
        # IT: senza RevIN, log_ret va scalato (e' nel set to-scale)
        # EN: without RevIN, log_ret should be scaled (in the to-scale set)
        b = FeatureBuilder(use_revin=False, lag_periods=5)
        no_scale = b._no_scale_set()
        assert "log_ret" not in no_scale
        assert "lag_ret_1" not in no_scale
        # IT: feature naturalmente in [0,1] / cicliche restano sempre escluse
        # EN: naturally [0,1] / cyclic features stay excluded
        assert "hour_sin" in no_scale and "body_ratio" in no_scale

    def test_revin_on_adds_raw_returns_to_no_scale(self):
        # IT: con RevIN, log_ret/log_ret_high/low/vol e lag_ret_* sono no-scale
        # EN: with RevIN, log_ret/log_ret_high/low/vol and lag_ret_* are no-scale
        b = FeatureBuilder(use_revin=True, lag_periods=5)
        no_scale = b._no_scale_set()
        for col in ("log_ret", "log_ret_high", "log_ret_low", "log_ret_vol"):
            assert col in no_scale, f"{col} deve essere escluso con RevIN"
        for k in range(1, 6):
            assert f"lag_ret_{k}" in no_scale

    def test_revin_lag_periods_respected(self):
        # IT: lag_periods=3 -> solo lag_ret_1..3 esclusi
        # EN: lag_periods=3 -> only lag_ret_1..3 excluded
        b = FeatureBuilder(use_revin=True, lag_periods=3)
        no_scale = b._no_scale_set()
        for k in (1, 2, 3):
            assert f"lag_ret_{k}" in no_scale
        for k in (4, 5):
            assert f"lag_ret_{k}" not in no_scale


# ─────────────────────────────────────────────────────────────────────────────
# IT: BLOCKER #1 (2026-05-28) — feature mismatch live vs training
# EN: BLOCKER #1 (2026-05-28) — live vs training feature mismatch
# ─────────────────────────────────────────────────────────────────────────────
def _load_live_buffer_class():
    """Carica LiveFeatureBuffer da scripts/04_live_signals.py senza eseguire l'engine."""
    spec = importlib.util.spec_from_file_location(
        "live_signals_mod", ROOT / "scripts" / "04_live_signals.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LiveFeatureBuffer


@pytest.fixture(scope="module")
def live_buffer_cls():
    return _load_live_buffer_class()


@pytest.fixture(scope="module")
def training_feature_names() -> list[str]:
    """Carica feature_names attese dal modello (da lstm_dataset.npz)."""
    p = ROOT / "data" / "lstm_dataset.npz"
    if not p.exists():
        pytest.skip("lstm_dataset.npz non disponibile")
    npz = np.load(p, allow_pickle=True)
    return [str(n) for n in npz["feature_names"]]


def _synthetic_candles(n: int = 400) -> list[dict]:
    # IT: candele sintetiche deterministiche (random walk con seed fisso)
    # EN: deterministic synthetic candles (seeded random walk)
    rng = np.random.default_rng(seed=42)
    price = 70000.0
    candles = []
    for i in range(n):
        ret = rng.normal(0, 0.0005)
        new_close = price * np.exp(ret)
        high = max(price, new_close) * (1 + abs(rng.normal(0, 0.0002)))
        low  = min(price, new_close) * (1 - abs(rng.normal(0, 0.0002)))
        vol  = abs(rng.normal(2.0, 0.5))
        candles.append({
            "open":          float(price),
            "high":          float(high),
            "low":           float(low),
            "close":         float(new_close),
            "volume":        float(vol),
            "taker_buy_vol": float(vol * 0.5),
            "hour":          (i // 60) % 24,
            "minute":        i % 60,
        })
        price = new_close
    return candles


class TestLiveFeatureBufferGolden:
    """
    Golden snapshot del LiveFeatureBuffer. Documenta lo stato attuale del
    BLOCKER #1 ed entra in regression-test: se qualcuno fixa il mismatch, il
    test della parity FALLISCE e va aggiornato (oppure il fix lo risolve).
    """

    WINDOW = 60
    EXPECTED_LIVE_FEATURES = 39

    def test_buffer_produces_window_with_expected_shape(self, live_buffer_cls):
        buf = live_buffer_cls(window=self.WINDOW)
        for c in _synthetic_candles(self.WINDOW + 100):
            buf.push(c)
        win = buf.get_window()
        assert win is not None
        assert win.shape == (self.WINDOW, self.EXPECTED_LIVE_FEATURES)

    def test_buffer_returns_none_before_warmup(self, live_buffer_cls):
        # IT: insufficienti candele -> None (no inferenza prematura)
        # EN: not enough candles -> None (no premature inference)
        buf = live_buffer_cls(window=self.WINDOW)
        for c in _synthetic_candles(self.WINDOW // 2):
            buf.push(c)
        assert buf.get_window() is None

    def test_buffer_output_is_clipped_in_range(self, live_buffer_cls):
        # IT: normalizzazione robusta + clip ±5σ
        # EN: robust normalization + ±5σ clip
        buf = live_buffer_cls(window=self.WINDOW)
        for c in _synthetic_candles(self.WINDOW + 100):
            buf.push(c)
        win = buf.get_window()
        assert win.min() >= -5.0 - 1e-6
        assert win.max() <= +5.0 + 1e-6

    def test_buffer_deterministic_for_same_input(self, live_buffer_cls):
        candles = _synthetic_candles(self.WINDOW + 100)
        b1 = live_buffer_cls(window=self.WINDOW)
        b2 = live_buffer_cls(window=self.WINDOW)
        for c in candles:
            b1.push(c); b2.push(c)
        np.testing.assert_allclose(b1.get_window(), b2.get_window())


class TestBlocker1Documentation:
    """
    IT: BLOCKER #1 RISOLTO (2026-06-05, Stage 5). Il path di produzione live è ora
        `FeatureAssembler` → `FeatureBuilder.build()` (104 feature canoniche, stesso scaler),
        con parity feature E segnale verificata in `tests/test_live_training_parity.py`
        (Gate 1 + Gate 2, bit-perfect). Il vecchio `LiveFeatureBuffer` resta SOLO come
        utility ATR/sanity, NON è più il percorso di feature: questi test lo documentano.
    EN: BLOCKER #1 RESOLVED (2026-06-05, Stage 5). Production live path is now
        FeatureAssembler → FeatureBuilder.build() (104 canonical features, same scaler), with
        feature AND signal parity verified in tests/test_live_training_parity.py. The old
        LiveFeatureBuffer survives ONLY as an ATR/sanity helper, no longer the feature path.
    """

    def test_legacy_buffer_is_deprecated_not_feature_path(self, live_buffer_cls, training_feature_names):
        # IT: Il buffer LEGACY produce ancora un set ridotto (≠104): è deprecato e NON usato
        #     per le feature. La parità di produzione è coperta da test_live_training_parity.py.
        # EN: The LEGACY buffer still yields a reduced set (≠104): deprecated, NOT the feature
        #     path. Production parity is covered by test_live_training_parity.py.
        buf = live_buffer_cls(window=60)
        for c in _synthetic_candles(200):
            buf.push(c)
        win = buf.get_window()
        assert win.shape[1] != len(training_feature_names), (
            f"Il buffer legacy ora produce {win.shape[1]} == {len(training_feature_names)} feature: "
            f"se è diventato il path di produzione, consolidare su FeatureAssembler e rimuovere il legacy."
        )

    def test_training_canonical_starts_with_raw_ohlcv(self, training_feature_names):
        # IT: Le 104 canoniche iniziano con open/high/low/close/volume — ora FORNITE in live dal
        #     FeatureAssembler (a differenza del legacy buffer). Sanity sull'ordine canonico.
        # EN: The 104 canonical features start with raw OHLCV — now SUPPLIED live by the
        #     FeatureAssembler (unlike the legacy buffer). Canonical-order sanity check.
        first_five = set(training_feature_names[:5])
        assert {"open", "high", "low", "close", "volume"}.issubset(first_five)
