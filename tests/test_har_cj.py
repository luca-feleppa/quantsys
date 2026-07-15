"""
tests/test_har_cj.py
====================
Test per le feature HAR-CJ (A4 roadmap vol): decomposizione bipower/jump
variation come INPUT, config-gated (`features.har_cj`, default OFF).

Copre:
  · inerzia del lever (default OFF = nessuna colonna nuova, output invariato)
  · correttezza matematica (J ≥ 0, ratio ∈ [0,1], BV > 0, no inf)
  · CAUSALITÀ (troncamento: nessuna dipendenza dal futuro)
  · sensibilità ai salti (uno shock isolato alza J e jump_ratio)

Esegui con:
  pytest tests/test_har_cj.py -v
"""

import numpy as np
import pandas as pd
import pytest

from quantsys.features import FeatureBuilder


# IT: Colonne attese quando il lever è ON (2 scale × 3 feature).
# EN: Expected columns when the lever is ON (2 scales × 3 features).
HAR_CJ_COLS = {
    "bv_1d", "jump_1d", "jump_ratio_1d",
    "bv_1w", "jump_1w", "jump_ratio_1w",
}


# IT: Helper — builder minimale a 1h (finestre 1d/1w = 24/168 barre).
# EN: Helper — minimal builder at 1h (1d/1w windows = 24/168 bars).
def _builder(use_har_cj: bool) -> FeatureBuilder:
    return FeatureBuilder(vp_bins=10, vp_lookback=50, windows=[5, 10],
                          interval_minutes=60, use_har_cj=use_har_cj)


# IT: TEST 1 — Inerzia: default OFF = nessuna colonna HAR-CJ, output bit-identico.
# EN: TEST 1 — Inertia: default OFF = no HAR-CJ column, bit-identical output.
class TestHarCjInertia:

    # IT: Il default del costruttore DEVE essere False (lever inerte).
    # EN: The constructor default MUST be False (inert lever).
    def test_default_is_off(self):
        assert FeatureBuilder().use_har_cj is False

    # IT: Con flag OFF la build non produce colonne bv_*/jump_* né in df né in feature_cols.
    # EN: With the flag OFF the build yields no bv_*/jump_* columns in df nor feature_cols.
    def test_off_no_new_columns(self, synthetic_ohlcv):
        b  = _builder(False)
        df = b.build(synthetic_ohlcv.copy(), normalize=False, fit=False)
        assert not (HAR_CJ_COLS & set(df.columns))
        assert not (HAR_CJ_COLS & set(b.feature_cols))

    # IT: ON aggiunge ESATTAMENTE le 6 colonne e NON perturba le colonne condivise
    #     (bit-identiche a parità di input): garanzia di inerzia sul path production.
    # EN: ON adds EXACTLY the 6 columns and does NOT perturb shared columns
    #     (bit-identical on the same input): inertia guarantee on the production path.
    def test_on_adds_columns_without_perturbing_shared(self, synthetic_ohlcv):
        df_off = _builder(False).build(synthetic_ohlcv.copy(), normalize=False, fit=False)
        df_on  = _builder(True).build(synthetic_ohlcv.copy(),  normalize=False, fit=False)
        assert set(df_on.columns) - set(df_off.columns) == HAR_CJ_COLS
        shared = [c for c in df_off.columns if c in df_on.columns]
        pd.testing.assert_frame_equal(df_off[shared], df_on[shared],
                                      check_exact=True)


# IT: TEST 2 — Correttezza matematica della decomposizione.
# EN: TEST 2 — Mathematical correctness of the decomposition.
class TestHarCjMath:

    def _built(self, synthetic_ohlcv):
        b  = _builder(True)
        df = b._returns(synthetic_ohlcv.copy())
        return b._har_cj(df)

    # IT: J = max(RV−BV,0) ≥ 0; ratio ∈ [0,1]; BV > 0 post-warmup; nessun inf.
    # EN: J = max(RV−BV,0) ≥ 0; ratio ∈ [0,1]; BV > 0 post-warmup; no inf.
    def test_bounds(self, synthetic_ohlcv):
        df = self._built(synthetic_ohlcv)
        for scale in ("1d", "1w"):
            bv, j, r = (df[f"bv_{scale}"].dropna(),
                        df[f"jump_{scale}"].dropna(),
                        df[f"jump_ratio_{scale}"].dropna())
            assert len(bv) > 0, f"tutte NaN su scala {scale} (warmup errato?)"
            assert (j >= 0).all()
            assert ((r >= 0) & (r <= 1)).all()
            assert (bv > 0).all()
            for s in (bv, j, r):
                assert np.isfinite(s.to_numpy()).all()

    # IT: Il warmup rispetta la finestra: prime w+1 barre NaN (rolling w su shift 1).
    # EN: Warmup respects the window: first w+1 bars NaN (rolling w over shift 1).
    def test_warmup_length(self, synthetic_ohlcv):
        df = self._built(synthetic_ohlcv)
        w_1d = 1440 // 60  # 24 barre a 1h / 24 bars at 1h
        assert df["bv_1d"].iloc[:w_1d].isna().all()
        assert df["bv_1d"].iloc[w_1d + 1:].notna().all()


# IT: TEST 3 — CAUSALITÀ: troncare il futuro non cambia i valori passati.
# EN: TEST 3 — CAUSALITY: truncating the future does not change past values.
class TestHarCjCausality:

    def test_truncation_invariance(self, synthetic_ohlcv):
        b = _builder(True)
        full  = b._har_cj(b._returns(synthetic_ohlcv.copy()))
        k     = 1200
        trunc = b._har_cj(b._returns(synthetic_ohlcv.iloc[:k].copy()))
        for col in HAR_CJ_COLS:
            a = full[col].iloc[:len(trunc)].to_numpy()
            t = trunc[col].to_numpy()
            # IT: NaN allineati (warmup) e valori identici dove definiti.
            # EN: aligned NaNs (warmup) and identical values where defined.
            assert np.array_equal(np.isnan(a), np.isnan(t)), col
            mask = ~np.isnan(a)
            np.testing.assert_array_equal(a[mask], t[mask], err_msg=col)


# IT: TEST 4 — Un salto isolato deve gonfiare RV più di BV → jump_ratio alto.
# EN: TEST 4 — An isolated jump must inflate RV more than BV → high jump_ratio.
class TestHarCjJumpDetection:

    def test_isolated_spike_raises_jump_ratio(self):
        rng = np.random.default_rng(7)
        n   = 400
        r   = rng.normal(0, 1e-3, n)
        r[300] = 0.08  # IT: salto isolato ~80σ | EN: isolated ~80σ jump
        df = pd.DataFrame({"log_ret": r})
        b  = _builder(True)
        df = b._har_cj(df)
        # IT: dentro la finestra 1d post-salto il ratio deve dominare il baseline pre-salto.
        # EN: inside the 1d window after the jump the ratio must dominate the pre-jump baseline.
        base  = df["jump_ratio_1d"].iloc[250:300].mean()
        spike = df["jump_ratio_1d"].iloc[301]
        assert spike > 0.5, f"jump_ratio al salto troppo basso: {spike:.3f}"
        assert spike > base + 0.3, f"salto non separato dal baseline ({spike:.3f} vs {base:.3f})"
