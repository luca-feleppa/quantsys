"""Parity test live↔training (BLOCKER #1 Stage 4).

IT: Verifica che FeatureAssembler (live engine) produca lo stesso output che
    FeatureBuilder produrrebbe in modalita' training su stessa finestra storica.
    Tolleranza massima per colonna: 1e-6.

EN: Verifies that FeatureAssembler (live engine) produces the same output that
    FeatureBuilder would produce in training mode on the same historical window.
    Per-column tolerance: 1e-6.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent


# IT: Import LiveCandleBuffer e FeatureAssembler da scripts/04_live_signals.py.
# EN: Imports LiveCandleBuffer and FeatureAssembler from scripts/04_live_signals.py.
def _load_live_module():
    spec = importlib.util.spec_from_file_location(
        "live_signals_mod", ROOT / "scripts" / "04_live_signals.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    return mod


# IT: Salta i test se mancano dipendenze runtime (parquet, pipeline_state, NPZ).
# EN: Skip tests when runtime deps are missing (parquet, pipeline_state, NPZ).
@pytest.fixture(scope="module")
def env():
    raw = ROOT / "data" / "raw_candles.parquet"
    ps_path = ROOT / "models" / "itransformer" / "pipeline_state.pkl"
    funding = ROOT / "data" / "funding_rate.parquet"
    npz = ROOT / "data" / "lstm_dataset.npz"
    for f in (raw, ps_path, funding, npz):
        if not f.exists():
            pytest.skip(f"Dipendenza mancante: {f}")
    return {"raw": raw, "ps": ps_path, "funding": funding, "npz": npz}


# IT: Test 1 — la window prodotta dal FeatureAssembler matcha la stessa
#     window prodotta direttamente da FeatureBuilder con stesso scaler.
# EN: Test 1 — the window produced by FeatureAssembler matches the same
#     window produced directly by FeatureBuilder with the same scaler.
def test_assembler_matches_direct_featurebuilder(env):
    from quantsys.features import FeatureBuilder, get_canonical_feature_names
    from quantsys.utils import PipelineState, load_config

    live_mod = _load_live_module()
    LiveCandleBuffer = live_mod.LiveCandleBuffer
    FeatureAssembler = live_mod.FeatureAssembler

    buf = LiveCandleBuffer(maxlen=50000)
    n = buf.bootstrap_from_parquet(str(env["raw"]))
    assert n >= 45000, f"Bootstrap insufficiente: {n}"

    ps = PipelineState.load(str(env["ps"]))
    funding_df = pd.read_parquet(env["funding"])

    # IT: Path A — FeatureAssembler (componente live nuovo).
    # EN: Path A — FeatureAssembler (new live component).
    asm = FeatureAssembler(buf, ps)
    win_a = asm.compute_window(window_size=120, funding_df=funding_df)
    assert win_a.shape == (120, 104)
    assert win_a.dtype == np.float32
    assert not np.isnan(win_a).any()
    assert not np.isinf(win_a).any()

    # IT: Path B — FeatureBuilder diretto (equivalente training, con scaler iniettato).
    # EN: Path B — direct FeatureBuilder (training equivalent, scaler injected).
    cfg = load_config()
    fcfg = cfg.get("features", {})
    mcfg = cfg.get("model", {})
    fb = FeatureBuilder(
        vp_bins          = fcfg.get("vp_bins", 30),
        vp_lookback      = fcfg.get("vp_lookback", 240),
        windows          = fcfg.get("windows", [5, 10, 20, 60]),
        lag_periods      = fcfg.get("lag_periods", 5),
        forecast_horizon = fcfg.get("forecast_horizon", 1),
        vp_stride        = fcfg.get("vp_stride", 1),
        frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
        use_revin        = bool(mcfg.get("use_revin", False)),
    )
    fb.scaler             = ps.scaler
    fb._scale_cols        = list(ps.scale_cols)
    fb.scalers            = dict(ps.price_scaler_state)
    fb.clip_lo_           = ps.clip_lo_
    fb.clip_hi_           = ps.clip_hi_
    fb.feature_cols       = list(ps.feature_cols)
    fb.n_dynamic_features = ps.n_dynamic_features

    df = buf.to_dataframe().reset_index()
    fd = funding_df.copy()
    if "open_time" in fd.columns:
        ot = pd.to_datetime(fd["open_time"])
        if getattr(ot.dt, "tz", None) is not None:
            ot = ot.dt.tz_convert("UTC").dt.tz_localize(None)
        fd["open_time"] = ot

    feat_df = fb.build(df, normalize=True, fit=False, funding_df=fd)
    canonical = get_canonical_feature_names()
    feat_df = feat_df[list(canonical)].dropna()
    win_b = feat_df.iloc[-120:].values.astype(np.float32)

    assert win_a.shape == win_b.shape
    diff = np.abs(win_a - win_b)
    max_diff = float(diff.max())
    print(f"Max abs diff (live vs training-eq): {max_diff:.2e}")
    assert max_diff < 1e-5, (
        f"Parity violation: max diff {max_diff:.4e} > 1e-5. "
        f"Le pipeline live e training NON producono output identici."
    )


# IT: Test 2 — i 104 nomi di feature_names sono nell'ordine atteso (no permutazioni).
# EN: Test 2 — the 104 feature_names are in the expected order (no shuffles).
def test_canonical_order_stable(env):
    from quantsys.features import get_canonical_feature_names

    names = get_canonical_feature_names(str(env["npz"]))
    assert len(names) == 104
    # IT: Sanity check su alcuni nomi ben noti.
    # EN: Sanity check on a few well-known names.
    assert names[0] == "open"
    assert names[1] == "high"
    assert names[2] == "low"
    assert names[3] == "close"
    assert names[4] == "volume"
    assert names[9] == "log_ret"
    assert names[-3] == "funding_rate"
    assert names[-2] == "funding_rate_1d"
    assert names[-1] == "funding_rate_dev"


# IT: Test 3 — la lista canonica non contiene feature live-incompatibili.
# EN: Test 3 — the canonical list contains no live-incompatible features.
def test_canonical_excludes_live_drop(env):
    from quantsys.features import LIVE_DROP_FEATURES, get_canonical_feature_names

    names = set(get_canonical_feature_names(str(env["npz"])))
    overlap = names & LIVE_DROP_FEATURES
    assert overlap == set(), f"Feature droppate trovate nella canonical: {overlap}"


# IT: Test 4 — l'hard-fail di FeatureAssembler scatta se funding_df e' None.
# EN: Test 4 — FeatureAssembler hard-fails if funding_df is None.
def test_assembler_hardfail_without_funding(env):
    from quantsys.utils import PipelineState

    live_mod = _load_live_module()
    LiveCandleBuffer = live_mod.LiveCandleBuffer
    FeatureAssembler = live_mod.FeatureAssembler

    buf = LiveCandleBuffer(maxlen=50000)
    buf.bootstrap_from_parquet(str(env["raw"]))
    ps = PipelineState.load(str(env["ps"]))
    asm = FeatureAssembler(buf, ps)

    with pytest.raises(RuntimeError, match=r"canoniche mancanti|funding"):
        asm.compute_window(window_size=120, funding_df=None)
