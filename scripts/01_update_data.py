"""
Script 01_update — Aggiornamento incrementale del dataset.
Scarica solo le candele mancanti dall'ultimo aggiornamento ad oggi,
poi ricalcola features e lstm_dataset su tutto lo storico.

Prerequisito: eseguire prima scripts/01_download_data.py (primo avvio).

Run configuration PyCharm:
  Script: scripts/01_update_data.py
  Working dir: <root del progetto>
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np

from quantsys.utils import load_config, setup_logging, ensure_dirs, PipelineState
from quantsys.utils.atomic_save import atomic_save_npz, atomic_save_parquet
from quantsys.data import fetch_klines_incremental, fetch_funding_rate
from quantsys.features import FeatureBuilder, create_windows, temporal_split, LIVE_DROP_FEATURES

setup_logging()
log = logging.getLogger("quantsys.script.01_update")


# IT: aggiornamento incrementale: scarica solo il delta e ricostruisce il dataset.
# EN: incremental update: fetch only the delta and rebuild the dataset.
def main():
    cfg  = load_config("config/default.yaml")
    dcfg = cfg["data"]
    fcfg = cfg["features"]
    mcfg = cfg["model"]

    ensure_dirs(dcfg["output_dir"])
    out = Path(dcfg["output_dir"])

    # IT: path del parquet OHLCV raw (override via default.yaml)
    # EN: path of the raw OHLCV parquet (override via default.yaml)
    raw_path = Path(dcfg.get("raw_path", "./data/raw_candles.parquet"))

    # IT: prerequisito - raw_candles.parquet deve esistere (creato da 01_download_data)
    # EN: prerequisite - raw_candles.parquet must exist (created by 01_download_data)
    if not raw_path.exists():
        print(
            f"\n[ERRORE] File non trovato: {raw_path}\n"
            f"\nEsegui prima il download completo:\n"
            f"  python scripts/01_download_data.py\n"
            f"\n01_update_data.py scarica solo il delta dall'ultimo aggiornamento.\n"
            f"Per il primo avvio è necessario 01_download_data.py.\n"
        )
        sys.exit(1)

    # IT: 1. download incrementale - solo il delta dall'ultimo timestamp salvato
    # EN: 1. incremental download - only the delta from the last saved timestamp
    t0 = time.time()
    log.info("Fase 1: aggiornamento incrementale candele ...")

    import pyarrow.parquet as _pq
    n_before = _pq.ParquetFile(raw_path).metadata.num_rows
    log.info(f"Raw candles esistenti: {n_before:,}")

    df_raw = fetch_klines_incremental(
        raw_path  = str(raw_path),
        symbol    = dcfg["symbol"],
        interval  = dcfg["interval"],
    )
    n_after = len(df_raw)
    n_new   = n_after - n_before

    log.info(
        f"Fase 1 completata in {time.time()-t0:.1f}s — "
        f"Candele: {n_before:,} → {n_after:,}  (+{n_new:,} nuove)  "
        f"[{df_raw['open_time'].iloc[0].date()} → {df_raw['open_time'].iloc[-1].date()}]"
    )

    # IT: 1b. funding rate (fetch_funding_rate gestisce il delta internamente)
    # EN: 1b. funding rate (fetch_funding_rate handles delta internally)
    try:
        funding_df = fetch_funding_rate(
            symbol     = dcfg["symbol"],
            start_time = dcfg.get("start_time", "2021-01-01"),
            output_dir = dcfg["output_dir"],
        )
        log.info(f"Funding rate: {len(funding_df)} osservazioni disponibili")
    except Exception as _e:
        log.warning(f"Aggiornamento funding rate fallito ({_e}) — continuo senza.")
        funding_df = None

    # IT: nessuna nuova candela -> esce subito senza ricalcolare features
    # EN: no new candles -> exit early without recomputing features
    if n_new == 0:
        print(
            f"\n═══════════════════════════════════════════\n"
            f"  01_update · GIÀ AGGIORNATO\n"
            f"═══════════════════════════════════════════\n"
            f"  Simbolo  : {dcfg['symbol']} {dcfg['interval']}\n"
            f"  Candele  : {n_before:,} (nessuna nuova)\n"
            f"  Ultimo ts: {df_raw['open_time'].iloc[-1]}\n"
        )
        sys.exit(0)

    # IT: salva il parquet OHLCV aggiornato in modo atomico
    # EN: persist the updated OHLCV parquet atomically
    raw_cols = ["open_time","close_time","open","high","low","close","volume",
                "quote_vol","trades","taker_buy_vol","taker_buy_quote_vol"]
    atomic_save_parquet(df_raw[raw_cols], raw_path, index=False)
    log.info(f"Raw candles aggiornato → {raw_path}  ({n_after:,} candele, {raw_path.stat().st_size//1024//1024} MB)")

    # IT: 2. holdout - taglia i dati dopo holdout_start (test set intoccato)
    # EN: 2. holdout - drop data after holdout_start (test set untouched)
    holdout_start = cfg.get("training", {}).get("holdout_start", None)
    if holdout_start:
        import pandas as pd
        cutoff    = pd.Timestamp(holdout_start, tz="UTC")
        n_raw_before = len(df_raw)
        df_raw    = df_raw[df_raw["open_time"] < cutoff].copy()
        n_removed = n_raw_before - len(df_raw)
        log.warning(
            f"HOLDOUT ATTIVO: rimossi {n_removed:,} campioni dopo {holdout_start}. "
            f"Questi dati sono bloccati per il test finale."
        )

    # IT: 3. feature engineering raw (la normalizzazione avviene dopo lo split)
    # EN: 3. raw feature engineering (normalization happens after the split)
    t0 = time.time()
    log.info(f"Fase 2: feature engineering su {len(df_raw):,} candele ...")
    builder = FeatureBuilder(
        vp_bins          = fcfg["vp_bins"],
        vp_lookback      = fcfg["vp_lookback"],
        windows          = fcfg["windows"],
        lag_periods      = fcfg["lag_periods"],
        forecast_horizon = fcfg.get("forecast_horizon", 1),
        vp_stride        = fcfg.get("vp_stride", 1),
        frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
    )
    df_feat = builder.build(df_raw, normalize=False, fit=False, funding_df=funding_df)
    log.info(f"Fase 2 completata in {time.time()-t0:.1f}s — {len(df_feat):,} righe valide")

    # IT: 4. determina il confine training PRIMA del fit scaler (anti-leakage)
    # EN: 4. compute the training cutoff BEFORE fitting the scaler (leakage guard)
    n_total   = len(df_feat)
    val_frac  = cfg["training"]["val_fraction"]
    test_frac = cfg["training"]["test_fraction"]
    train_end = int(n_total * (1 - val_frac - test_frac))

    log.info(
        f"Split temporale: train=[0,{train_end}) "
        f"val+test=[{train_end},{n_total})  "
        f"({train_end/n_total:.0%} training)"
    )

    # IT: 5. fit scaler solo su train, poi transform su tutto il dataset
    # EN: 5. fit scaler on train only, then transform the whole dataset
    if fcfg["normalize"]:
        t0 = time.time()
        log.info("Fase 3: scaler fit+transform ...")
        builder.fit_scaler_only(df_feat.iloc[:train_end])   # IT: fit solo su train | EN: fit on train only
        df_feat = builder._normalize(df_feat, fit=False)    # IT: transform su tutto | EN: transform on all
        log.info(f"Fase 3 completata in {time.time()-t0:.1f}s")

    # IT: salva il parquet delle feature in modo atomico
    # EN: persist the feature parquet atomically
    feat_path = out / "features.parquet"
    atomic_save_parquet(df_feat, feat_path, index=False)
    log.info(f"Features → {feat_path}  ({feat_path.stat().st_size//1024} KB)")

    # IT: 6. costruisce le windows; esclude colonne non-feature e non-float
    # EN: 6. build the sliding windows; drop non-feature and non-float columns
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

    # IT: scarta feature con piu' del 50% di NaN
    # EN: drop features with more than 50% NaN
    nan_thresh = 0.5
    feat_cols_before = feat_cols[:]
    nan_ratios = {c: df_feat[c].isna().mean() for c in feat_cols_before}
    feat_cols = [c for c in feat_cols
                 if nan_ratios[c] <= nan_thresh]
    log.info(f"Feature valide (≤{nan_thresh*100:.0f}% NaN): {len(feat_cols)}")

    excluded = [(c, nan_ratios[c]) for c in feat_cols_before if c not in feat_cols]
    if excluded:
        log.warning(
            f"Feature escluse per NaN > {nan_thresh*100:.0f}% "
            f"({len(excluded)} su {len(feat_cols_before)}):"
        )
        for col, pct in excluded:
            log.warning(f"  {col}: {pct*100:.1f}% NaN")

    inf_cols = [c for c in feat_cols if np.isinf(df_feat[c].values).any()]
    if inf_cols:
        log.warning(f"Escluse {len(inf_cols)} colonne con valori Inf: {inf_cols}")
        feat_cols = [c for c in feat_cols if c not in inf_cols]

    _struct_names   = set(builder.feature_cols[builder.n_dynamic_features:])
    n_dynamic_final = sum(1 for c in feat_cols if c not in _struct_names)
    if n_dynamic_final != builder.n_dynamic_features:
        log.warning(
            f"n_dynamic_features aggiornato: {builder.n_dynamic_features} → {n_dynamic_final}"
        )

    t0 = time.time()
    log.info("Fase 4: create_windows (stride_tricks) ...")
    X, y, t = create_windows(df_feat, feat_cols, window_size=mcfg["window_size"],
                             window_stride=mcfg.get("window_stride", 1))
    log.info(f"Fase 4 completata in {time.time()-t0:.1f}s — X={X.shape}")
    splits = temporal_split(X, y, t,
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
        f"({n_dynamic_final} dyn + "
        f"{len(feat_cols)-n_dynamic_final} struct features)"
    )

    # IT: 7. salva PipelineState unificato per l'inference
    # EN: 7. persist unified PipelineState for inference
    ensure_dirs("models")
    state = (
        PipelineState()
        .from_feature_builder(builder)
        .set_training_config(cfg)
    )
    state.model_config = {
        "n_features":          len(feat_cols),
        "n_dynamic_features":  n_dynamic_final,
        "window_size":         mcfg["window_size"],
    }
    state.set_dataset_info(df_feat, n_train=len(splits["X_train"]))
    import os as _os
    _ps_arch = _os.environ.get("QUANTSYS_ARCH", "lstm")
    _ps_dir  = Path("models") / _ps_arch
    _ps_dir.mkdir(parents=True, exist_ok=True)
    _ps_file = str(_ps_dir / "pipeline_state.pkl")
    state.save(_ps_file)
    log.info(f"PipelineState salvato → {_ps_file}")

    print(f"""
═══════════════════════════════════════════
  01_update · AGGIORNAMENTO COMPLETATO
═══════════════════════════════════════════
  Simbolo       : {dcfg['symbol']} {dcfg['interval']}
  Candele prima : {n_before:,}
  Candele dopo  : {n_after:,}
  Nuove candele : +{n_new:,}
  Arco temporale: {df_raw['open_time'].iloc[0].date()} → {df_raw['open_time'].iloc[-1].date()}
  Candele valide: {len(df_feat):,}
  Features      : {len(feat_cols)}
  Window        : {mcfg['window_size']} minuti
  Train samples : {len(splits['X_train']):,}
  Val samples   : {len(splits['X_val']):,}
  Test samples  : {len(splits['X_test']):,}
  Shape X_train : {splits['X_train'].shape}

  → Re-training opzionale: python scripts/02_train.py
""")


if __name__ == "__main__":
    main()
