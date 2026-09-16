"""
Script 01 — Download data from Binance and build features.
Run from PyCharm or from a terminal in the project root.

PyCharm run configuration:
  Script: scripts/01_download_data.py
  Working dir: <project root>
"""
import logging
import os
import time
from pathlib import Path

import numpy as np

from quantsys.utils import load_config, setup_logging, ensure_dirs, PipelineState, interval_minutes_from_cfg
from quantsys.utils.atomic_save import atomic_save_npz, atomic_save_parquet
from quantsys.data import fetch_klines, fetch_funding_rate
from quantsys.features import FeatureBuilder, create_windows, temporal_split, canonical_feature_columns

setup_logging()
log = logging.getLogger("quantsys.script.01")


# full pipeline: download → features → split → windows → PipelineState.
def main():
    # Windows console defaults to cp1252 — unicode chars in the final banner
    # crash the print (5th occurrence of this bug). Reconfigure UTF-8 like 02/04.
    import sys as _sys
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cfg  = load_config("config/default.yaml")
    dcfg = cfg["data"]
    fcfg = cfg["features"]
    mcfg = cfg["model"]

    ensure_dirs(dcfg["output_dir"])
    out = Path(dcfg["output_dir"])

    # 1. download OHLCV candles from Binance (REST klines)
    t0 = time.time()
    log.info("Fase 1: download candele ...")
    start_time = dcfg.get("start_time", None)
    df_raw = fetch_klines(
        dcfg["symbol"], dcfg["interval"], dcfg["limit"],
        start_time=start_time,
    )
    log.info(
        f"Fase 1 completata in {time.time()-t0:.1f}s — "
        f"Candele scaricate: {len(df_raw):,}  "
        f"[{df_raw['open_time'].iloc[0].date()} → {df_raw['open_time'].iloc[-1].date()}]"
    )

    # 1b. perpetual futures funding rate (best-effort, non-blocking)
    try:
        funding_df = fetch_funding_rate(
            symbol     = dcfg["symbol"],
            start_time = dcfg.get("start_time", "2021-01-01"),
            output_dir = dcfg["output_dir"],
        )
        log.info(f"Funding rate: {len(funding_df)} osservazioni scaricate")
    except Exception as _e:
        log.warning(f"Download funding rate fallito ({_e}) — continuo senza.")
        funding_df = None

    # holdout - drop data after holdout_start; keep this set untouched until final test
    holdout_start = cfg.get("training", {}).get("holdout_start", None)
    if holdout_start:
        import pandas as pd
        cutoff    = pd.Timestamp(holdout_start, tz="UTC")
        n_before  = len(df_raw)
        df_raw    = df_raw[df_raw["open_time"] < cutoff].copy()
        n_removed = n_before - len(df_raw)
        log.warning(
            f"HOLDOUT ATTIVO: rimossi {n_removed:,} campioni dopo {holdout_start}. "
            f"Questi dati sono bloccati per il test finale."
        )

    # persist raw OHLCV (no features) for later incremental updates
    raw_path = out / "raw_candles.parquet"
    raw_cols = ["open_time","close_time","open","high","low","close","volume",
                "quote_vol","trades","taker_buy_vol","taker_buy_quote_vol"]
    atomic_save_parquet(df_raw[raw_cols], raw_path, index=False)
    log.info(f"Raw candles → {raw_path}  ({len(df_raw):,} candele, {raw_path.stat().st_size//1024//1024} MB)")

    # 2. raw feature engineering (normalization happens after the split)
    t0 = time.time()
    log.info(f"Fase 2: feature engineering su {len(df_raw):,} candele ...")
    # with use_revin=True raw return columns are excluded from the global RobustScaler
    # so RevIN normalizes raw features and mu/log_var stay aligned with the target
    _use_revin = bool(mcfg.get("use_revin", False))
    # interval_minutes from data.interval — the FeatureBuilder's TIME-semantic
    # windows are converted to bars (identity at 1m).
    builder = FeatureBuilder(
        vp_bins          = fcfg["vp_bins"],
        vp_lookback      = fcfg["vp_lookback"],
        windows          = fcfg["windows"],
        lag_periods      = fcfg["lag_periods"],
        forecast_horizon = fcfg.get("forecast_horizon", 1),
        vp_stride        = fcfg.get("vp_stride", 1),
        frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
        use_revin        = _use_revin,
        interval_minutes = interval_minutes_from_cfg(cfg),
        # target_type from config (default "ret" = legacy directional; "log_rv" = vol-S).
        target_type      = fcfg.get("target_type", "ret"),
        # A4 HAR-CJ — inert lever (default false = 104 features bit-invariant).
        use_har_cj       = bool(fcfg.get("har_cj", False)),
    )
    df_feat = builder.build(df_raw, normalize=False, fit=False, funding_df=funding_df)
    log.info(f"Fase 2 completata in {time.time()-t0:.1f}s — {len(df_feat):,} righe valide")

    # 3. compute the training cutoff BEFORE fitting the scaler (leakage guard)
    n_total   = len(df_feat)
    val_frac  = cfg["training"]["val_fraction"]
    test_frac = cfg["training"]["test_fraction"]
    train_end = int(n_total * (1 - val_frac - test_frac))

    log.info(
        f"Split temporale: train=[0,{train_end}) "
        f"val+test=[{train_end},{n_total})  "
        f"({train_end/n_total:.0%} training)"
    )

    # 4. fit the scaler on train only, then transform the whole dataset
    if fcfg["normalize"]:
        t0 = time.time()
        log.info("Fase 3: scaler fit+transform ...")
        builder.fit_scaler_only(df_feat.iloc[:train_end])   # fit on train only
        df_feat = builder._normalize(df_feat, fit=False)    # transform on all
        log.info(f"Fase 3 completata in {time.time()-t0:.1f}s")

        # RevIN sanity-check: log_ret must stay raw (~[-0.05,+0.05] for BTC 1m).
        # If it lands in [-3,+3] it was scaled by mistake and RevIN would be broken.
        if _use_revin and "log_ret" in df_feat.columns:
            _lr = df_feat["log_ret"].dropna()
            if len(_lr) > 0:
                _lo, _hi = float(_lr.quantile(0.001)), float(_lr.quantile(0.999))
                _scaled = "log_ret" in builder._scale_cols
                log.info(
                    f"RevIN diagnostic: use_revin=True | log_ret scaled={_scaled} "
                    f"| range[0.1%,99.9%]=[{_lo:.5f},{_hi:.5f}] | mean={float(_lr.mean()):.2e}"
                )
                if _scaled:
                    log.error(
                        "RevIN: log_ret risulta SCALATO dal RobustScaler — "
                        "RevIN.denormalize_mu non sarà allineato al target raw. "
                        "Verifica _no_scale_set() in quantsys/features/__init__.py."
                    )
                elif abs(_hi) > 0.5 or abs(_lo) > 0.5:
                    log.warning(
                        f"RevIN: log_ret range [{_lo:.4f},{_hi:.4f}] è inatteso per "
                        f"return 1m raw — controlla la pipeline di feature engineering."
                    )

    # persist the feature parquet atomically (crash-safe)
    feat_path = out / "features.parquet"
    atomic_save_parquet(df_feat, feat_path, index=False)
    log.info(f"Features → {feat_path}  ({feat_path.stat().st_size//1024} KB)")

    # 5. shared canonical feature list (C2 2ter): non-feature exclude →
    # float dtype → C-funding → NaN>50% → Inf, in quantsys.features.
    # `diag` preserves this script's historical logging.
    nan_thresh = 0.5
    diag: dict = {}
    feat_cols = canonical_feature_columns(builder.feature_cols, df_feat,
                                          nan_thresh=nan_thresh, diag=diag)
    if diag["dropped_live"]:
        log.info(f"Set C-funding: scartate {len(diag['dropped_live'])} feature "
                 f"live-incompatibili: {diag['dropped_live']}")
    log.info(f"Feature valide (≤{nan_thresh*100:.0f}% NaN): "
             f"{len(feat_cols) + len(diag['dropped_inf'])}")
    if diag["dropped_inf"]:
        log.warning(f"Escluse {len(diag['dropped_inf'])} colonne con valori Inf: "
                    f"{diag['dropped_inf']}")

    # recompute n_dynamic after the NaN/Inf filter to avoid TFT dual-stream mismatch
    _struct_names = set(builder.feature_cols[builder.n_dynamic_features:])
    n_dynamic_final = sum(1 for c in feat_cols if c not in _struct_names)
    if n_dynamic_final != builder.n_dynamic_features:
        log.warning(
            f"n_dynamic_features aggiornato: {builder.n_dynamic_features} → {n_dynamic_final} "
            f"(alcune colonne dinamiche escluse per NaN/Inf)"
        )

    if diag["dropped_nan"]:
        log.warning(
            f"Feature escluse per NaN > {nan_thresh*100:.0f}% "
            f"({len(diag['dropped_nan'])}):"
        )
        for col, pct in diag["dropped_nan"]:
            log.warning(f"  {col}: {pct*100:.1f}% NaN")

    t0 = time.time()
    log.info("Fase 4: create_windows (stride_tricks) ...")
    X, y, t = create_windows(df_feat, feat_cols, window_size=mcfg["window_size"],
                             window_stride=mcfg.get("window_stride", 1))
    log.info(f"Fase 4 completata in {time.time()-t0:.1f}s — X={X.shape}")
    splits   = temporal_split(X, y, t,
                              val_frac =cfg["training"]["val_fraction"],
                              test_frac=cfg["training"]["test_fraction"])

    t0 = time.time()
    log.info("Fase 5: salvataggio dataset NN ...")
    npz_path = out / "lstm_dataset.npz"
    atomic_save_npz(
        npz_path, **splits,
        feature_names       = np.array(feat_cols),
        n_dynamic_features  = np.array([n_dynamic_final]),
    )
    log.info(
        f"Fase 5 completata in {time.time()-t0:.1f}s — "
        f"Dataset → {npz_path}  "
        f"({builder.n_dynamic_features} dyn + "
        f"{len(feat_cols)-builder.n_dynamic_features} struct features)"
    )

    # 6. persist unified PipelineState (scaler + columns + config) for inference
    ensure_dirs("models")
    state = (
        PipelineState()
        .from_feature_builder(builder)
        .set_training_config(cfg)
    )
    state.model_config = {
        "n_features":          len(feat_cols),
        "n_dynamic_features":  builder.n_dynamic_features,
        "window_size":         mcfg["window_size"],
    }
    # record dataset metadata (timeframe, sample count, frequency) for diagnostics
    state.set_dataset_info(df_feat, n_train=len(splits["X_train"]))
    _ps_arch = os.environ.get("QUANTSYS_ARCH", "lstm")
    _ps_dir  = Path("models") / _ps_arch
    _ps_dir.mkdir(parents=True, exist_ok=True)
    _ps_file = str(_ps_dir / "pipeline_state.pkl")
    state.save(_ps_file)
    log.info(f"PipelineState salvato → {_ps_file}")
    # CANONICAL copy at models/pipeline_state.pkl — the dataset (scaler/features/interval)
    # is arch-independent; without it a 02_train run with a different QUANTSYS_ARCH
    # would only find its arch dir's stale pkl (2026-06-10 bug: 1m state re-saved
    # under a 1h dataset → interval guard tripped in backtest).
    _ps_canon = str(Path("models") / "pipeline_state.pkl")
    state.save(_ps_canon)
    log.info(f"PipelineState canonico → {_ps_canon}")

    print(f"""
═══════════════════════════════════════════
  01 · DOWNLOAD & FEATURES · COMPLETATO
═══════════════════════════════════════════
  Simbolo       : {dcfg['symbol']} {dcfg['interval']}
  Candele raw   : {len(df_raw):,}
  Candele valide: {len(df_feat):,}
  Features      : {len(feat_cols)}
  Window        : {mcfg['window_size']} barre ({mcfg['window_size'] * interval_minutes_from_cfg(cfg)} min)
  Train samples : {len(splits['X_train']):,}
  Val samples   : {len(splits['X_val']):,}
  Test samples  : {len(splits['X_test']):,}
  Shape X_train : {splits['X_train'].shape}

  Raw candles : {raw_path}
  → Aggiornamenti futuri: python scripts/01_update_data.py
    (scarica solo il delta, molto più veloce)
  → Prossimo: python scripts/02_train.py
""")


if __name__ == "__main__":
    main()
