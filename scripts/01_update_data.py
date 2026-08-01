"""
Script 01_update — Aggiornamento incrementale del dataset.
Scarica solo le candele mancanti dall'ultimo aggiornamento ad oggi,
poi ricalcola features e lstm_dataset su tutto lo storico.

⚠ Il run completo NON è un'operazione neutra: rifitta il RobustScaler sullo split
   train allungato, riscrive features.parquet + lstm_dataset.npz e salva un nuovo
   PipelineState sotto models/{QUANTSYS_ARCH|lstm}/. Su una linea con modelli
   CONGELATI (la vol production ha target_scale persistito nel suo state) questo
   rompe il contratto train↔inference. Se serve solo la storia OHLCV aggiornata
   — p.es. per un giudice offline che calcola la RV realizzata dalle barre —
   usare `--candles-only`, che si ferma dopo il parquet raw.

⚠ The full run is NOT a neutral operation: it refits the RobustScaler on the
   extended train split, rewrites features.parquet + lstm_dataset.npz and saves a
   new PipelineState under models/{QUANTSYS_ARCH|lstm}/. On a line with FROZEN
   models (the production vol one has target_scale persisted in its state) this
   breaks the train↔inference contract. When only the refreshed OHLCV history is
   needed — e.g. an offline judge computing realized RV from bars — use
   `--candles-only`, which stops right after the raw parquet.

Prerequisito: eseguire prima scripts/01_download_data.py (primo avvio).

Run configuration PyCharm:
  Script: scripts/01_update_data.py
  Working dir: <root del progetto>
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from quantsys.utils import load_config, setup_logging, ensure_dirs, PipelineState, interval_minutes_from_cfg
from quantsys.utils.atomic_save import atomic_save_npz, atomic_save_parquet
from quantsys.data import fetch_klines_incremental, fetch_funding_rate
from quantsys.features import FeatureBuilder, create_windows, temporal_split, canonical_feature_columns

setup_logging()
log = logging.getLogger("quantsys.script.01_update")


# IT: aggiornamento incrementale: scarica solo il delta e ricostruisce il dataset.
# EN: incremental update: fetch only the delta and rebuild the dataset.
def main():
    # IT: boilerplate UTF-8 — il banner contiene box-drawing e frecce, che su una
    #     console Windows cp1252 farebbero crashare la stampa finale.
    # EN: UTF-8 boilerplate — the banner contains box-drawing and arrows, which
    #     would crash the final print on a cp1252 Windows console.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Aggiornamento incrementale del dataset")
    # IT: --candles-only: estende SOLO data/raw_candles.parquet e si ferma. Nessun
    #     re-fit dello scaler, nessun npz, nessun PipelineState riscritto — è il
    #     path sicuro quando i modelli a valle sono congelati. Stesso pattern già
    #     usato in produzione dal bootstrap gap-aware di VolForecaster.
    # EN: --candles-only: extends ONLY data/raw_candles.parquet and stops. No
    #     scaler refit, no npz, no PipelineState rewritten — the safe path when
    #     downstream models are frozen. Same pattern already used in production by
    #     VolForecaster's gap-aware bootstrap.
    ap.add_argument("--candles-only", action="store_true",
                    help="aggiorna solo raw_candles.parquet, non tocca scaler/npz/state "
                         "/ refresh raw_candles.parquet only, leaves scaler/npz/state alone")
    args = ap.parse_args()

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

    # IT: 1b. funding rate (fetch_funding_rate gestisce il delta internamente).
    #     Saltato in --candles-only: il funding entra solo nel feature engineering,
    #     che in quella modalità non viene eseguito.
    # EN: 1b. funding rate (fetch_funding_rate handles delta internally). Skipped
    #     in --candles-only: funding feeds feature engineering only, which that
    #     mode does not run.
    funding_df = None
    if args.candles_only:
        log.info("--candles-only: funding rate non aggiornato / funding rate not refreshed")
    else:
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

    # IT: uscita anticipata di --candles-only: da qui in poi si ricalcolano feature,
    #     si RIFITTA lo scaler e si riscrive il PipelineState. Fermarsi PRIMA della
    #     Fase 2 è l'intero punto del flag: la storia OHLCV è aggiornata, tutto il
    #     resto resta bit-invariato.
    # EN: --candles-only early exit: from here on features are recomputed, the
    #     scaler is REFIT and the PipelineState rewritten. Stopping BEFORE phase 2
    #     is the whole point of the flag: OHLCV history is refreshed, everything
    #     else stays bit-invariant.
    if args.candles_only:
        print(f"""
═══════════════════════════════════════════
  01_update · SOLO CANDELE
═══════════════════════════════════════════
  Simbolo       : {dcfg['symbol']} {dcfg['interval']}
  Candele prima : {n_before:,}
  Candele dopo  : {n_after:,}
  Nuove candele : +{n_new:,}
  Arco temporale: {df_raw['open_time'].iloc[0].date()} → {df_raw['open_time'].iloc[-1].date()}

  Non toccati   : features.parquet · lstm_dataset.npz · scaler · PipelineState
""")
        return

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
    # IT: interval_minutes da data.interval — finestre TIME-semantic convertite
    #     in barre dal FeatureBuilder (identità a 1m).
    # EN: interval_minutes from data.interval — TIME-semantic windows converted
    #     to bars by the FeatureBuilder (identity at 1m).
    builder = FeatureBuilder(
        vp_bins          = fcfg["vp_bins"],
        vp_lookback      = fcfg["vp_lookback"],
        windows          = fcfg["windows"],
        lag_periods      = fcfg["lag_periods"],
        forecast_horizon = fcfg.get("forecast_horizon", 1),
        vp_stride        = fcfg.get("vp_stride", 1),
        frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
        interval_minutes = interval_minutes_from_cfg(cfg),
        # IT: A4 HAR-CJ — lever inerte (default false = 104 feature bit-invariate).
        # EN: A4 HAR-CJ — inert lever (default false = 104 features bit-invariant).
        use_har_cj       = bool(fcfg.get("har_cj", False)),
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

    # IT: 6. lista feature canonica condivisa (C2 2ter): exclude non-feature →
    #     dtype float → C-funding → NaN>50% → Inf, in quantsys.features.
    # EN: 6. shared canonical feature list (C2 2ter): non-feature exclude →
    #     float dtype → C-funding → NaN>50% → Inf, in quantsys.features.
    nan_thresh = 0.5
    diag: dict = {}
    feat_cols = canonical_feature_columns(builder.feature_cols, df_feat,
                                          nan_thresh=nan_thresh, diag=diag)
    if diag["dropped_live"]:
        log.info(f"Set C-funding: scartate {len(diag['dropped_live'])} feature "
                 f"live-incompatibili: {diag['dropped_live']}")
    log.info(f"Feature valide (≤{nan_thresh*100:.0f}% NaN): "
             f"{len(feat_cols) + len(diag['dropped_inf'])}")
    if diag["dropped_nan"]:
        log.warning(
            f"Feature escluse per NaN > {nan_thresh*100:.0f}% "
            f"({len(diag['dropped_nan'])}):"
        )
        for col, pct in diag["dropped_nan"]:
            log.warning(f"  {col}: {pct*100:.1f}% NaN")
    if diag["dropped_inf"]:
        log.warning(f"Escluse {len(diag['dropped_inf'])} colonne con valori Inf: "
                    f"{diag['dropped_inf']}")

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
  Window        : {mcfg['window_size']} barre ({mcfg['window_size'] * interval_minutes_from_cfg(cfg)} min)
  Train samples : {len(splits['X_train']):,}
  Val samples   : {len(splits['X_val']):,}
  Test samples  : {len(splits['X_test']):,}
  Shape X_train : {splits['X_train'].shape}

  → Re-training opzionale: python scripts/02_train.py
""")


if __name__ == "__main__":
    main()
