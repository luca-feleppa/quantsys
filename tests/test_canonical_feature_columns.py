# IT: TEST C2 refactor 2ter (2026-07-18) — `canonical_feature_columns()`:
#     ① golden di LOGICA: l'espressione VECCHIA (congelata qui sotto, com'era
#        duplicata in 01_download/01_update/04b/vol_paper_replay/paper_01)
#        e la funzione condivisa devono produrre la STESSA lista, bit-per-bit,
#        su un DataFrame sintetico che esercita tutti e 5 i filtri;
#     ② golden LISTA-104: sui dati production (features.parquet + PipelineState
#        canonico) la derivazione deve riprodurre ESATTAMENTE i feature_names
#        dell'npz (single source of truth) — skip se gli artefatti mancano.
# EN: C2 2ter refactor TEST (2026-07-18) — `canonical_feature_columns()`:
#     ① LOGIC golden: the OLD expression (frozen below, as previously duplicated
#        across 01_download/01_update/04b/vol_paper_replay/paper_01) and the
#        shared function must produce the SAME list, bit-for-bit, on a synthetic
#        DataFrame exercising all 5 filters;
#     ② 104-LIST golden: on production data (features.parquet + canonical
#        PipelineState) the derivation must EXACTLY reproduce the npz
#        feature_names (single source of truth) — skipped if artifacts missing.
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantsys.features import (CANONICAL_EXCLUDE, LIVE_DROP_FEATURES,
                               canonical_feature_columns,
                               get_canonical_feature_names)

ROOT = Path(__file__).resolve().parents[1]


def _old_expression(feature_cols, feat, nan_thresh=0.5):
    # IT: replica CONGELATA dell'espressione pre-refactor (variante 04b, la più
    #     completa: include il check `c in feat.columns`). NON aggiornarla: è il
    #     golden contro cui si misura la funzione condivisa.
    # EN: FROZEN replica of the pre-refactor expression (04b variant, the most
    #     complete: includes the `c in feat.columns` check). Do NOT update it:
    #     it is the golden the shared function is measured against.
    exclude = {"open_time", "close_time", "date_utc", "pv", "cum_pv", "cum_vol",
               "typical_price", "obv", "target_ret", "target_dir"}
    cols = [c for c in feature_cols
            if c not in exclude and c in feat.columns
            and feat[c].dtype in ["float64", "float32"]
            and c not in LIVE_DROP_FEATURES]
    cols = [c for c in cols if feat[c].isna().mean() <= nan_thresh]
    cols = [c for c in cols if not np.isinf(feat[c].values).any()]
    return cols


def _synthetic():
    # IT: DataFrame che esercita OGNI filtro: exclude, dtype, C-funding,
    #     NaN>50%, Inf, colonna assente dal df; ordine non alfabetico
    #     per verificare la preservazione dell'ordine del builder.
    # EN: DataFrame exercising EVERY filter: exclude, dtype, C-funding,
    #     NaN>50%, Inf, column missing from the df; non-alphabetical order
    #     to verify builder-order preservation.
    n = 10
    rng = np.random.default_rng(0)
    feat = pd.DataFrame({
        "zeta_ok": rng.normal(size=n),                       # float64, resta/kept
        "open_time": pd.date_range("2026-01-01", periods=n, freq="1h"),  # exclude
        "int_col": np.arange(n),                             # int64 → dtype filter
        "momentum_7d": rng.normal(size=n),                   # LIVE_DROP_FEATURES
        "nan_heavy": [np.nan] * 6 + [1.0] * 4,               # 60% NaN → filtro NaN
        "inf_col": [1.0] * (n - 1) + [np.inf],               # Inf → filtro Inf
        "alpha_ok": rng.normal(size=n).astype(np.float32),   # float32, resta/kept
        "obv": rng.normal(size=n),                           # exclude
    })
    feature_cols = ["zeta_ok", "open_time", "int_col", "momentum_7d",
                    "nan_heavy", "inf_col", "alpha_ok", "obv", "ghost_col"]
    return feature_cols, feat


class TestLogicGolden:

    def test_new_equals_old_bitwise(self):
        feature_cols, feat = _synthetic()
        old = _old_expression(feature_cols, feat)
        new = canonical_feature_columns(feature_cols, feat)
        assert new == old == ["zeta_ok", "alpha_ok"]

    def test_order_is_builder_order(self):
        # IT: l'ordine è quello di feature_cols, NON alfabetico né del df.
        # EN: order follows feature_cols, NOT alphabetical nor df order.
        feature_cols, feat = _synthetic()
        assert canonical_feature_columns(feature_cols, feat) == ["zeta_ok", "alpha_ok"]
        reordered = ["alpha_ok"] + [c for c in feature_cols if c != "alpha_ok"]
        assert canonical_feature_columns(reordered, feat) == ["alpha_ok", "zeta_ok"]

    def test_diag_matches_filters(self):
        feature_cols, feat = _synthetic()
        diag: dict = {}
        canonical_feature_columns(feature_cols, feat, diag=diag)
        assert diag["dropped_live"] == ["momentum_7d"]
        assert [c for c, _ in diag["dropped_nan"]] == ["nan_heavy"]
        assert diag["dropped_inf"] == ["inf_col"]
        # IT: nan_ratios copre solo le colonne sopravvissute ai primi 3 filtri.
        # EN: nan_ratios covers only columns surviving the first 3 filters.
        assert set(diag["nan_ratios"]) == {"zeta_ok", "nan_heavy", "inf_col", "alpha_ok"}

    def test_exclude_set_frozen(self):
        # IT: regression sul contenuto di CANONICAL_EXCLUDE (era hardcoded in 5 script).
        # EN: regression on CANONICAL_EXCLUDE content (was hardcoded in 5 scripts).
        assert CANONICAL_EXCLUDE == frozenset({
            "open_time", "close_time", "date_utc", "pv", "cum_pv", "cum_vol",
            "typical_price", "obv", "target_ret", "target_dir"})


@pytest.mark.skipif(
    not ((ROOT / "data" / "features.parquet").exists()
         and (ROOT / "models" / "pipeline_state.pkl").exists()
         and (ROOT / "data" / "lstm_dataset.npz").exists()),
    reason="artefatti production assenti (features.parquet / pipeline_state.pkl / npz)")
def test_golden_lista_104_production():
    # IT: golden LISTA-104 — la derivazione condivisa su features.parquet +
    #     PipelineState canonico riproduce ESATTAMENTE (nomi E ordine) i
    #     feature_names dell'npz production. Qualsiasi drift = FAIL.
    # EN: 104-LIST golden — the shared derivation on features.parquet + the
    #     canonical PipelineState EXACTLY reproduces (names AND order) the
    #     production npz feature_names. Any drift = FAIL.
    from quantsys.utils import PipelineState
    ps = PipelineState.load(str(ROOT / "models" / "pipeline_state.pkl"))
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    cols = canonical_feature_columns(list(ps.feature_cols), df)
    golden = get_canonical_feature_names(str(ROOT / "data" / "lstm_dataset.npz"))
    assert tuple(cols) == golden
    assert len(cols) == 104
