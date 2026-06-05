"""
IT: Replay storico — alimenta LiveCandleBuffer + FeatureAssembler con candele storiche
    e verifica che il vettore feature prodotto sia identico a FeatureBuilder diretto.
    BLOCKER #1 Stage 4 (2026-06-02): nuovo engine vs training → atteso 0 mismatch.

EN: Historical replay — feeds LiveCandleBuffer + FeatureAssembler with historical candles
    and verifies the produced feature vector is identical to direct FeatureBuilder output.
    BLOCKER #1 Stage 4 (2026-06-02): new engine vs training → expected 0 mismatches.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# IT: Forza UTF-8 su stdout per evitare crash con cp1252 su Windows.
# EN: Force UTF-8 on stdout to avoid cp1252 crash on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# IT: import dal nuovo live engine | EN: import from the new live engine
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("live_signals_mod", ROOT / "scripts" / "04_live_signals.py")
live_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_mod)
LiveCandleBuffer = live_mod.LiveCandleBuffer
FeatureAssembler = live_mod.FeatureAssembler


# IT: Esegue il replay e stampa un report di verifica.
# EN: Runs the replay and prints a verification report.
def main(n_candles: int = 50000, data_dir: Path = ROOT / "data") -> int:
    print("=" * 70)
    print("REPLAY LIVE ENGINE (Stage 4) vs TRAINING FEATURE BUILDER")
    print("=" * 70)

    from quantsys.features import FeatureBuilder, get_canonical_feature_names
    from quantsys.utils import PipelineState, load_config

    # IT: 1. carica feature canoniche dal NPZ | EN: load canonical features from NPZ
    canonical = get_canonical_feature_names(str(data_dir / "lstm_dataset.npz"))
    print(f"\n[TRAINING] feature_names canoniche: {len(canonical)}")
    print(f"           prime 5: {list(canonical[:5])}")
    print(f"           ultime 5: {list(canonical[-5:])}")

    # IT: 2. alimenta LiveCandleBuffer dal parquet | EN: feed LiveCandleBuffer from parquet
    buf = LiveCandleBuffer(maxlen=n_candles)
    n_loaded = buf.bootstrap_from_parquet(str(data_dir / "raw_candles.parquet"))
    print(f"\n[BUFFER] LiveCandleBuffer alimentato: {n_loaded} candele")
    print(f"         intervallo: {buf.to_dataframe(2).index[0]} -> {buf.latest['open_time']}")

    # IT: 3. carica PipelineState + funding | EN: load PipelineState + funding
    ps = PipelineState.load(str(ROOT / "models" / "itransformer" / "pipeline_state.pkl"))
    funding_df = pd.read_parquet(data_dir / "funding_rate.parquet")
    print(f"\n[STATE] PipelineState.scaler.n_features_in_={ps.scaler.n_features_in_}")
    print(f"        funding rate obs disponibili: {len(funding_df)}")

    # IT: 4. compute window via FeatureAssembler (path live) | EN: compute window via FeatureAssembler (live path)
    asm = FeatureAssembler(buf, ps)
    win_live = asm.compute_window(window_size=120, funding_df=funding_df)
    print(f"\n[LIVE] FeatureAssembler.compute_window(120) → shape={win_live.shape} dtype={win_live.dtype}")
    print(f"       NaN: {np.isnan(win_live).any()}  Inf: {np.isinf(win_live).any()}")
    print(f"       stats: mean={win_live.mean():+.4f} std={win_live.std():.4f} "
          f"range=[{win_live.min():.2f}, {win_live.max():.2f}]")

    # IT: 5. compute window via FeatureBuilder diretto (training equivalent)
    # EN: compute window via direct FeatureBuilder (training equivalent)
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
    missing = set(canonical) - set(feat_df.columns)
    if missing:
        print(f"\n[FAIL] FeatureBuilder.build NON ha prodotto: {sorted(missing)[:10]}")
        return 1
    feat_df = feat_df[list(canonical)].dropna()
    win_train = feat_df.iloc[-120:].values.astype(np.float32)

    # IT: 6. confronto parity | EN: parity comparison
    print("\n" + "-" * 70)
    print("ANALISI PARITY")
    print("-" * 70)

    if win_live.shape != win_train.shape:
        print(f"\n  [FAIL] shape mismatch: live={win_live.shape} vs train={win_train.shape}")
        return 1

    diff = np.abs(win_live - win_train)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    per_col_max = diff.max(axis=0)
    worst_idx = int(np.argmax(per_col_max))

    print(f"\n  Shape:      {win_live.shape} == {win_train.shape}  ✓")
    print(f"  Max diff:   {max_diff:.3e}")
    print(f"  Mean diff:  {mean_diff:.3e}")
    print(f"  Worst col:  {canonical[worst_idx]} (idx={worst_idx}, max_diff={per_col_max[worst_idx]:.3e})")

    feat_ok = max_diff < 1e-5
    if feat_ok:
        print(f"\n  ✅ GATE 1 (feature parity) OK — tolleranza 1e-5")
    else:
        cols_over_threshold = np.where(per_col_max > 1e-5)[0]
        print(f"\n  ❌ GATE 1 VIOLATION — {len(cols_over_threshold)} colonne con diff > 1e-5")
        for idx in cols_over_threshold[:10]:
            print(f"       - {canonical[idx]}: max_diff={per_col_max[idx]:.3e}")

    # IT: ── GATE 2 (Stage 5) — PARITY DEL SEGNALE: i due percorsi feature (assembler live vs
    #     FeatureBuilder diretto) devono produrre lo STESSO (μ,σ,side) attraverso il nucleo di
    #     inferenza deterministico di produzione + SignalGenutore. Se le feature sono bit-identiche
    #     ma una micro-differenza < tol flippa il side su soglia → la parity feature non basta.
    # EN: ── GATE 2 (Stage 5) — SIGNAL PARITY: the two feature routes (live assembler vs direct
    #     FeatureBuilder) must yield the SAME (μ,σ,side) through the production deterministic
    #     inference core + SignalGenerator. Catches threshold-flips that feature parity alone misses.
    print("\n" + "-" * 70)
    print("GATE 2 — PARITY SEGNALE (Stage 5: live↔offline)")
    print("-" * 70)

    import torch
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.trading import SignalGenerator
    LiveEngine = live_mod.LiveEngine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load_heterogeneous(device, cfg=cfg)   # IT: stesso modello di backtest+live
    model.eval()
    bcfg = cfg["backtest"]
    sig_gen = SignalGenerator(
        prob_threshold   = bcfg["prob_threshold"],
        min_expected_ret = bcfg["min_expected_ret"],
        max_sigma        = bcfg["max_sigma"],
        conviction_alpha = bcfg.get("conviction_alpha", 0.5),
        min_snr          = bcfg.get("min_snr", 0.0),
    )
    # IT: nucleo deterministico CONDIVISO col live engine (no MC dropout: l'ensemble non lo espone).
    # EN: deterministic core SHARED with the live engine (no MC dropout: the ensemble lacks it).
    mu_l, sig_l, nu_l = LiveEngine._deterministic_predict(model, win_live,  None, ps, device)
    mu_t, sig_t, nu_t = LiveEngine._deterministic_predict(model, win_train, None, ps, device)
    side_l, _ = sig_gen.generate(mu_l, sig_l, nu_l)
    side_t, _ = sig_gen.generate(mu_t, sig_t, nu_t)

    d_mu, d_sig = abs(mu_l - mu_t), abs(sig_l - sig_t)
    print(f"\n  LIVE   (assembler): μ={mu_l:+.6e}  σ={sig_l:.6e}  side={side_l.value}")
    print(f"  OFFLINE(builder)  : μ={mu_t:+.6e}  σ={sig_t:.6e}  side={side_t.value}")
    print(f"  Δμ={d_mu:.3e}  Δσ={d_sig:.3e}  side_match={side_l == side_t}")

    sig_ok = (d_mu < 1e-5) and (d_sig < 1e-5) and (side_l == side_t)
    if sig_ok:
        print(f"\n  ✅ GATE 2 (signal parity) OK — μ/σ entro 1e-5 e side identico")
    else:
        print(f"\n  ❌ GATE 2 VIOLATION — il segnale live diverge dall'offline")

    print("\n" + "=" * 70)
    if feat_ok and sig_ok:
        print("✅ BLOCKER #1 Stage 5 — ENTRAMBI I GATE VERDI: live engine allineato al backtest.")
        return 0
    print("❌ BLOCKER #1 Stage 5 — almeno un gate fallito (vedi sopra).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
