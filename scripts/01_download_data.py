"""
Script 01 — Download dati da Binance e costruzione features.
Esegui da PyCharm o da terminale nella root del progetto.

Run configuration PyCharm:
  Script: scripts/01_download_data.py
  Working dir: <root del progetto>
"""
import logging
import os
import time
from pathlib import Path

import numpy as np

from quantsys.utils import load_config, setup_logging, ensure_dirs, PipelineState, interval_minutes_from_cfg
from quantsys.utils.atomic_save import atomic_save_npz, atomic_save_parquet
from quantsys.data import fetch_klines, fetch_funding_rate
from quantsys.features import FeatureBuilder, create_windows, temporal_split, LIVE_DROP_FEATURES

setup_logging()
log = logging.getLogger("quantsys.script.01")


# IT: pipeline completa: download → feature → split → windows → PipelineState.
# EN: full pipeline: download → features → split → windows → PipelineState.
def main():
    # IT: Console Windows default cp1252 — i caratteri unicode del banner finale
    #     crashano il print (5ª occorrenza del bug). Reconfigure UTF-8 come 02/04.
    # EN: Windows console defaults to cp1252 — unicode chars in the final banner
    #     crash the print (5th occurrence of this bug). Reconfigure UTF-8 like 02/04.
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

    # IT: 1. download candele OHLCV da Binance (REST klines)
    # EN: 1. download OHLCV candles from Binance (REST klines)
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

    # IT: 1b. funding rate dei futures perpetui (best-effort, non bloccante)
    # EN: 1b. perpetual futures funding rate (best-effort, non-blocking)
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

    # IT: holdout - tronca i dati dopo holdout_start; il set resta intatto fino al test finale
    # EN: holdout - drop data after holdout_start; keep this set untouched until final test
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

    # IT: salva OHLCV raw (no features) per gli aggiornamenti incrementali futuri
    # EN: persist raw OHLCV (no features) for later incremental updates
    raw_path = out / "raw_candles.parquet"
    raw_cols = ["open_time","close_time","open","high","low","close","volume",
                "quote_vol","trades","taker_buy_vol","taker_buy_quote_vol"]
    atomic_save_parquet(df_raw[raw_cols], raw_path, index=False)
    log.info(f"Raw candles → {raw_path}  ({len(df_raw):,} candele, {raw_path.stat().st_size//1024//1024} MB)")

    # IT: 2. feature engineering raw (la normalizzazione e' fatta dopo lo split)
    # EN: 2. raw feature engineering (normalization happens after the split)
    t0 = time.time()
    log.info(f"Fase 2: feature engineering su {len(df_raw):,} candele ...")
    # IT: con use_revin=True le colonne return raw sono escluse dal RobustScaler
    #     globale cosi' RevIN normalizza feature raw e mu/log_var restano allineati al target
    # EN: with use_revin=True raw return columns are excluded from the global RobustScaler
    #     so RevIN normalizes raw features and mu/log_var stay aligned with the target
    _use_revin = bool(mcfg.get("use_revin", False))
    # IT: interval_minutes da data.interval — le finestre TIME-semantic del
    #     FeatureBuilder vengono convertite in barre (identità a 1m).
    # EN: interval_minutes from data.interval — the FeatureBuilder's TIME-semantic
    #     windows are converted to bars (identity at 1m).
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
        # IT: target_type da config (default "ret" = direzionale legacy; "log_rv" = vol-S).
        # EN: target_type from config (default "ret" = legacy directional; "log_rv" = vol-S).
        target_type      = fcfg.get("target_type", "ret"),
    )
    df_feat = builder.build(df_raw, normalize=False, fit=False, funding_df=funding_df)
    log.info(f"Fase 2 completata in {time.time()-t0:.1f}s — {len(df_feat):,} righe valide")

    # IT: 3. determina il confine training PRIMA del fit dello scaler (anti-leakage)
    # EN: 3. compute the training cutoff BEFORE fitting the scaler (leakage guard)
    n_total   = len(df_feat)
    val_frac  = cfg["training"]["val_fraction"]
    test_frac = cfg["training"]["test_fraction"]
    train_end = int(n_total * (1 - val_frac - test_frac))

    log.info(
        f"Split temporale: train=[0,{train_end}) "
        f"val+test=[{train_end},{n_total})  "
        f"({train_end/n_total:.0%} training)"
    )

    # IT: 4. fit dello scaler solo su train, poi transform sull'intero dataset
    # EN: 4. fit the scaler on train only, then transform the whole dataset
    if fcfg["normalize"]:
        t0 = time.time()
        log.info("Fase 3: scaler fit+transform ...")
        builder.fit_scaler_only(df_feat.iloc[:train_end])   # IT: fit solo su train | EN: fit on train only
        df_feat = builder._normalize(df_feat, fit=False)    # IT: transform su tutto | EN: transform on all
        log.info(f"Fase 3 completata in {time.time()-t0:.1f}s")

        # IT: sanity-check RevIN: log_ret deve restare in scala raw (~[-0.05,+0.05] per BTC 1m).
        #     Se appare in [-3,+3] e' stato scalato per errore e RevIN sarebbe rotto.
        # EN: RevIN sanity-check: log_ret must stay raw (~[-0.05,+0.05] for BTC 1m).
        #     If it lands in [-3,+3] it was scaled by mistake and RevIN would be broken.
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

    # IT: salva il parquet delle feature in modo atomico (crash-safe)
    # EN: persist the feature parquet atomically (crash-safe)
    feat_path = out / "features.parquet"
    atomic_save_parquet(df_feat, feat_path, index=False)
    log.info(f"Features → {feat_path}  ({feat_path.stat().st_size//1024} KB)")

    # IT: 5. costruisce le windows; esclude colonne non-feature e non-float
    # EN: 5. build the sliding windows; drop non-feature and non-float columns
    exclude = {"open_time","close_time","date_utc","pv","cum_pv","cum_vol",
               "typical_price","obv","target_ret","target_dir"}
    feat_cols = [c for c in builder.feature_cols
                 if c not in exclude
                 and df_feat[c].dtype in ["float64","float32"]]

    # IT: set "C-funding" — scarta le feature a ROI ≤0 / lookback >30g non calcolabili in live
    #     (importance 2026-05-28). Garantisce consistenza training↔live per costruzione.
    # EN: "C-funding" set — drop ROI ≤0 / >30d-lookback features not computable live
    #     (2026-05-28 importance). Guarantees training↔live consistency by construction.
    _dropped_live = [c for c in feat_cols if c in LIVE_DROP_FEATURES]
    if _dropped_live:
        feat_cols = [c for c in feat_cols if c not in LIVE_DROP_FEATURES]
        log.info(f"Set C-funding: scartate {len(_dropped_live)} feature live-incompatibili: {_dropped_live}")

    # IT: scarta feature con piu' del 50% di NaN (es. momentum_90d su <90gg storia)
    # EN: drop features with more than 50% NaN (e.g. momentum_90d on <90d history)
    nan_thresh = 0.5
    feat_cols_before = feat_cols[:]
    nan_ratios = {c: df_feat[c].isna().mean() for c in feat_cols_before}
    feat_cols = [c for c in feat_cols
                 if nan_ratios[c] <= nan_thresh]
    log.info(f"Feature valide (≤{nan_thresh*100:.0f}% NaN): {len(feat_cols)}")

    # IT: scarta colonne con Inf (divisioni per zero non protette a monte)
    # EN: drop columns containing Inf (divisions by zero not guarded upstream)
    inf_cols = [c for c in feat_cols if np.isinf(df_feat[c].values).any()]
    if inf_cols:
        log.warning(f"Escluse {len(inf_cols)} colonne con valori Inf: {inf_cols}")
        feat_cols = [c for c in feat_cols if c not in inf_cols]

    # IT: ricalcola n_dynamic dopo NaN/Inf filter per evitare mismatch TFT dual-stream
    # EN: recompute n_dynamic after the NaN/Inf filter to avoid TFT dual-stream mismatch
    _struct_names = set(builder.feature_cols[builder.n_dynamic_features:])
    n_dynamic_final = sum(1 for c in feat_cols if c not in _struct_names)
    if n_dynamic_final != builder.n_dynamic_features:
        log.warning(
            f"n_dynamic_features aggiornato: {builder.n_dynamic_features} → {n_dynamic_final} "
            f"(alcune colonne dinamiche escluse per NaN/Inf)"
        )

    excluded = [(c, nan_ratios[c]) for c in feat_cols_before if c not in feat_cols]
    if excluded:
        log.warning(
            f"Feature escluse per NaN > {nan_thresh*100:.0f}% "
            f"({len(excluded)} su {len(feat_cols_before)}):"
        )
        for col, pct in excluded:
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

    # IT: 6. salva PipelineState unificato (scaler + colonne + config) per inference
    # EN: 6. persist unified PipelineState (scaler + columns + config) for inference
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
    # IT: registra metadati dataset (timeframe, n campioni, frequenza) per la diagnostica
    # EN: record dataset metadata (timeframe, sample count, frequency) for diagnostics
    state.set_dataset_info(df_feat, n_train=len(splits["X_train"]))
    _ps_arch = os.environ.get("QUANTSYS_ARCH", "lstm")
    _ps_dir  = Path("models") / _ps_arch
    _ps_dir.mkdir(parents=True, exist_ok=True)
    _ps_file = str(_ps_dir / "pipeline_state.pkl")
    state.save(_ps_file)
    log.info(f"PipelineState salvato → {_ps_file}")
    # IT: copia CANONICA in models/pipeline_state.pkl — il dataset (scaler/feature/interval)
    #     è arch-independent; senza questa copia un 02_train con QUANTSYS_ARCH diversa
    #     troverebbe solo il pkl stale della sua arch dir (bug 2026-06-10: state 1m
    #     ri-salvato sotto dataset 1h → guard interval scattato in backtest).
    # EN: CANONICAL copy at models/pipeline_state.pkl — the dataset (scaler/features/interval)
    #     is arch-independent; without it a 02_train run with a different QUANTSYS_ARCH
    #     would only find its arch dir's stale pkl (2026-06-10 bug: 1m state re-saved
    #     under a 1h dataset → interval guard tripped in backtest).
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
