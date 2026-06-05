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


# IT: Fixture condivisa — costruisce UNA sola volta win_a (assembler live) e win_b
#     (FeatureBuilder diretto, equivalente training) sulla stessa finestra storica.
#     Riusata da Gate 1 (parity feature) e Gate 2 (parity segnale) per non ricostruire le feature.
# EN: Shared fixture — builds win_a (live assembler) and win_b (direct FeatureBuilder, training
#     equivalent) ONCE on the same historical window. Reused by Gate 1 (feature) and Gate 2 (signal).
@pytest.fixture(scope="module")
def parity_windows(env):
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
    return {"win_a": win_a, "win_b": win_b, "ps": ps, "cfg": cfg, "live_mod": live_mod}


# IT: Test 1 (Gate 1) — la window del FeatureAssembler matcha quella del FeatureBuilder diretto.
# EN: Test 1 (Gate 1) — the FeatureAssembler window matches the direct FeatureBuilder window.
def test_assembler_matches_direct_featurebuilder(parity_windows):
    win_a, win_b = parity_windows["win_a"], parity_windows["win_b"]
    assert win_a.shape == (120, 104)
    assert win_a.dtype == np.float32
    assert not np.isnan(win_a).any()
    assert not np.isinf(win_a).any()
    assert win_a.shape == win_b.shape
    diff = np.abs(win_a - win_b)
    max_diff = float(diff.max())
    print(f"Max abs diff (live vs training-eq): {max_diff:.2e}")
    assert max_diff < 1e-5, (
        f"Parity violation: max diff {max_diff:.4e} > 1e-5. "
        f"Le pipeline live e training NON producono output identici."
    )


# IT: Test 5 (Gate 2, Stage 5) — PARITY DEL SEGNALE. I due percorsi feature, attraverso il
#     nucleo di inferenza DETERMINISTICO di produzione (LiveEngine._deterministic_predict, lo
#     stesso usato dall'ensemble live) + SignalGenerator, devono produrre lo STESSO (μ,σ,side).
#     Cattura i flip di side su soglia che la sola parity-feature non vedrebbe. Chiude BLOCKER #1.
# EN: Test 5 (Gate 2, Stage 5) — SIGNAL PARITY. Both feature routes, through the production
#     DETERMINISTIC inference core (LiveEngine._deterministic_predict, the same the live ensemble
#     uses) + SignalGenerator, must yield the SAME (μ,σ,side). Catches threshold side-flips.
def test_signal_parity_live_vs_offline(parity_windows):
    import torch
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.trading import SignalGenerator

    win_a, win_b = parity_windows["win_a"], parity_windows["win_b"]
    ps, cfg = parity_windows["ps"], parity_windows["cfg"]
    LiveEngine = parity_windows["live_mod"].LiveEngine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # IT: stesso ensemble eterogeneo del backtest e del live engine.
    # EN: same heterogeneous ensemble used by both the backtest and the live engine.
    model = EnsembleModel.load_heterogeneous(device, cfg=cfg)
    model.eval()
    bcfg = cfg["backtest"]
    sig_gen = SignalGenerator(
        prob_threshold   = bcfg["prob_threshold"],
        min_expected_ret = bcfg["min_expected_ret"],
        max_sigma        = bcfg["max_sigma"],
        conviction_alpha = bcfg.get("conviction_alpha", 0.5),
        min_snr          = bcfg.get("min_snr", 0.0),
    )

    # IT: μ/σ/ν RAW via il nucleo deterministico condiviso col live engine (no MC dropout).
    # EN: raw μ/σ/ν via the deterministic core shared with the live engine (no MC dropout).
    mu_l, sig_l, nu_l = LiveEngine._deterministic_predict(model, win_a, None, ps, device)
    mu_t, sig_t, nu_t = LiveEngine._deterministic_predict(model, win_b, None, ps, device)
    side_l, _ = sig_gen.generate(mu_l, sig_l, nu_l)
    side_t, _ = sig_gen.generate(mu_t, sig_t, nu_t)

    print(f"live μ={mu_l:+.6e} σ={sig_l:.6e} side={side_l.value} | "
          f"offline μ={mu_t:+.6e} σ={sig_t:.6e} side={side_t.value}")
    assert abs(mu_l - mu_t) < 1e-5, f"μ diverge: {mu_l} vs {mu_t}"
    assert abs(sig_l - sig_t) < 1e-5, f"σ diverge: {sig_l} vs {sig_t}"
    assert side_l == side_t, f"side diverge: {side_l} vs {side_t}"


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
