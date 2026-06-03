"""
Script 01b — Download dati macro + costruzione regime session-based (Asia/EU/US).
Da eseguire UNA VOLTA dopo 01_download_data.py.
I dati macro vengono poi usati automaticamente da 02_train.py.

Run configuration PyCharm:
  Script: scripts/01b_download_macro.py
  Working dir: <root del progetto>

API Key FRED (gratuita):
  1. Vai su https://fred.stlouisfed.org/docs/api/api_key.html
  2. Registrati (gratis)
  3. Inserisci la key in config/default.yaml → macro.fred_api_key
  Senza key: funziona lo stesso ma con rate limit più stretto.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from quantsys.utils import load_config, setup_logging, ensure_dirs
from quantsys.utils.atomic_save import atomic_save_npz, atomic_save_parquet
from quantsys.macro import (
    FREDDownloader, MacroFeatureBuilder,
    FRED_SERIES, YFINANCE_TICKERS, fetch_yfinance,
)
from quantsys.macro.regime import (
    RegimeMarkovSwitching, RegimeSession, RegimeMarkovBTC, MacroNormalizer,
)

setup_logging()
log = logging.getLogger("quantsys.script.01b")


# IT: pipeline macro — download FRED/yfinance, regime MS, normalizer, merge nel dataset NN
# EN: macro pipeline — download FRED/yfinance, MS regime, normalizer, merge into NN dataset
def main():
    cfg   = load_config("config/default.yaml")
    mcfg  = cfg.get("macro", {})
    dcfg  = cfg["data"]
    out   = Path(dcfg["output_dir"])
    ensure_dirs(str(out))

    fred_key  = mcfg.get("fred_api_key", None)
    start     = mcfg.get("history_start", "2018-01-01")
    n_regimes = mcfg.get("n_regimes", 3)

    print(f"""
{'═'*60}
  01b · DOWNLOAD DATI MACRO
  Periodo    : {start} → oggi
  FRED key   : {'✓ configurata' if fred_key else '⚠ non configurata (rate limit)'}
  N. regimi  : {n_regimes}
{'═'*60}
""")

    # IT: 1. download delle serie macro da FRED
    # EN: 1. download macro series from FRED
    log.info("Download serie FRED ...")
    fred    = FREDDownloader(api_key=fred_key)
    df_fred = fred.fetch_all(FRED_SERIES, start=start)

    fred_path = out / "macro_fred.parquet"
    atomic_save_parquet(df_fred, fred_path)
    log.info(f"FRED → {fred_path}  ({df_fred.shape[1]} serie, {len(df_fred)} giorni)")

    # IT: 2. download dati mercato (indici, VIX, USD, ...) via yfinance
    # EN: 2. download market data (indices, VIX, USD, ...) via yfinance
    log.info("Download dati mercato (yfinance) ...")
    df_yf = fetch_yfinance(YFINANCE_TICKERS, start=start)
    yf_path = out / "macro_yfinance.parquet"
    if not df_yf.empty:
        atomic_save_parquet(df_yf, yf_path)
        log.info(f"yfinance → {yf_path}  ({df_yf.shape[1]} serie)")
    else:
        log.warning("yfinance vuoto — controlla la connessione o installa yfinance.")

    # IT: 3. feature engineering macro (combina FRED + yfinance)
    # EN: 3. macro feature engineering (combines FRED + yfinance)
    log.info("Costruzione macro features ...")
    builder  = MacroFeatureBuilder()
    df_macro = builder.build(df_fred, df_yf)
    macro_path = out / "macro_features.parquet"
    atomic_save_parquet(df_macro, macro_path)
    log.info(f"Macro features → {macro_path}  ({df_macro.shape[1]} colonne)")

    # IT: 4. regime detection Markov-Switching su realized vol BTC (Variante 3 MODEL_IMPROVEMENTS)
    # EN: 4. Markov-Switching regime detection on BTC realized vol (Variant 3 MODEL_IMPROVEMENTS)
    log.info("Regime detector: Markov-Switching su realized vol BTC oraria ...")
    regime_model = RegimeMarkovBTC(n_regimes=n_regimes)
    try:
        # IT: df_macro è ignorato dal detector — usa direttamente raw_candles.parquet.
        # EN: df_macro is ignored by the detector — uses raw_candles.parquet directly.
        regime_df = regime_model.fit_predict_walkforward(
            df_macro, burn_in_days=30, retrain_days=30,
        )
        # IT: stesso filename del MS per backward compat (consumer non toccati).
        # EN: same filename as MS for backward compat (consumers untouched).
        hmm_path = out / "regime_hmm.pkl"
        regime_model.save(str(hmm_path))
        log.info(f"RegimeMarkovBTC salvato → {hmm_path}")

        regime_path = out / "regime_probs.parquet"
        atomic_save_parquet(regime_df, regime_path)
        log.info(f"Probabilità regime → {regime_path}")

        # IT: analisi distribuzione regimi (post burn-in 30gg = 720h)
        # EN: regime distribution analysis (post 30d=720h burn-in)
        post = regime_df[~regime_df["regime_burn_in"]] if "regime_burn_in" in regime_df else regime_df
        counts = post["regime_dominant"].value_counts().sort_index()
        print("\n  Distribuzione regimi Markov-Switching su realized vol BTC (hourly UTC, post burn-in):")
        for r, cnt in counts.items():
            pct = cnt / len(post) * 100
            bar = "█" * int(pct / 2)
            print(f"    Regime {r}  {cnt:>5} ore  ({pct:5.1f}%)  {bar}")
        print()

    except Exception as e:
        log.warning(f"RegimeMarkovBTC fallito ({e}) — fallback a DataFrame vuoto")
        regime_df = pd.DataFrame()

    # IT: 5. fit del MacroNormalizer (RobustScaler con clipping)
    # EN: 5. fit the MacroNormalizer (RobustScaler with clipping)
    log.info("Fitting MacroNormalizer ...")
    macro_cols = list(df_macro.columns)
    normalizer = MacroNormalizer()
    X_macro_norm = normalizer.fit_transform(df_macro, macro_cols)
    norm_path = out / "macro_normalizer.pkl"
    normalizer.save(str(norm_path))
    log.info(f"Normalizer salvato → {norm_path}")

    # IT: aggiorna PipelineState con il MacroNormalizer (per inference live)
    # EN: update PipelineState with the MacroNormalizer (for live inference)
    import os as _os
    _ps_arch = _os.environ.get("QUANTSYS_ARCH", "lstm")
    pipeline_state_path = str(Path("models") / _ps_arch / "pipeline_state.pkl")
    try:
        from quantsys.utils import PipelineState
        state = PipelineState.load(pipeline_state_path)
        state.from_macro_normalizer(normalizer)
        state.save(pipeline_state_path)
        log.info(f"PipelineState aggiornato con MacroNormalizer → {pipeline_state_path}")
    except FileNotFoundError:
        log.warning(f"{pipeline_state_path} non trovato — esegui prima 01_download_data.py")
    except Exception as e:
        log.warning(f"PipelineState update fallito (non critico): {e}")

    # IT: 6. merge delle feature macro nel dataset NN gia' esistente
    # EN: 6. merge macro features into the existing NN dataset
    npz_path = out / "lstm_dataset.npz"
    if npz_path.exists():
        log.info("Merge macro con dataset LSTM esistente ...")
        with np.load(npz_path, allow_pickle=True) as npz:
            # IT: carica in memoria e chiudi il file (evita file-lock su Windows)
            # EN: load into memory and close the file (avoid Windows file lock)
            splits_out = {k: np.array(npz[k]) for k in npz.files}

        for split in ["train", "val", "test"]:
            t_key = f"t_{split}"
            if t_key not in splits_out:
                continue
            timestamps = pd.to_datetime(splits_out[t_key])
            dates      = timestamps.normalize()

            # IT: mappa ogni timestamp al giorno corrispondente (forward-fill macro)
            # EN: map each timestamp to its day (forward-fill macro features)
            if split == "train":
                macro_daily = df_macro.copy()
                macro_daily.index = pd.to_datetime(macro_daily.index, utc=True).normalize()
                macro_daily = macro_daily[~macro_daily.index.duplicated(keep="last")]
                macro_daily = macro_daily[macro_cols].sort_index()

            dates_utc = pd.DatetimeIndex(dates, tz="UTC")
            all_dates = macro_daily.index.append(dates_utc).drop_duplicates().sort_values()
            merged = macro_daily.reindex(all_dates).ffill().loc[dates_utc]
            # IT: leading NaN (date prima della prima macro) -> 0
            # EN: leading NaNs (dates before first macro observation) -> 0
            merged = merged.fillna(0.0)

            X_macro_split = merged.values.astype(np.float32)
            # IT: stesso scaler fittato sopra, clip a +/-5 sigma
            # EN: same scaler fitted above, clipped at +/-5 sigma
            X_macro_split = np.clip(
                normalizer.scaler.transform(X_macro_split), -5, 5
            ).astype(np.float32)

            splits_out[f"X_macro_{split}"] = X_macro_split
            log.info(f"  {split}: X_macro shape = {X_macro_split.shape}")

        splits_out["macro_feature_names"] = np.array(macro_cols)
        splits_out["n_macro_features"]    = np.array([len(macro_cols)])

        atomic_save_npz(npz_path, **splits_out)
        log.info(f"Dataset LSTM aggiornato con macro → {npz_path}")
    else:
        log.warning(f"{npz_path} non trovato — esegui prima 01_download_data.py")

    print(f"""
{'═'*60}
  01b · MACRO DOWNLOAD · COMPLETATO
{'═'*60}
  Serie FRED          : {df_fred.shape[1]}
  Serie yfinance      : {df_yf.shape[1] if not df_yf.empty else 0}
  Macro features      : {df_macro.shape[1]}
  Regimi (Markov-BTC) : {n_regimes} su realized vol BTC oraria
  MacroEncoder input  : {len(macro_cols)} features → 16 dim embedding

  File generati in {out}/:
    ✓ macro_fred.parquet
    ✓ macro_yfinance.parquet
    ✓ macro_features.parquet
    ✓ regime_hmm.pkl  (RegimeMarkovBTC, filename kept for backward compat)
    ✓ regime_probs.parquet
    ✓ macro_normalizer.pkl
    ✓ lstm_dataset.npz   (aggiornato con X_macro_*)

  → Prossimo: python scripts/02_train.py
    (rileva automaticamente X_macro_* e usa QuantLSTMWithMacro)
{'═'*60}
""")


if __name__ == "__main__":
    main()
