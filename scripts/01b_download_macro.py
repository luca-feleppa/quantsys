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
import argparse
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


# IT: sezione regime (step 4) — condivisa fra pipeline completa e --regime-only.
#     Il detector ignora df_macro (legge raw_candles.parquet): accetta None.
# EN: regime section (step 4) — shared between full pipeline and --regime-only.
#     The detector ignores df_macro (reads raw_candles.parquet): accepts None.
def run_regime_detection(mcfg: dict, out: Path, df_macro=None) -> pd.DataFrame:
    n_regimes = mcfg.get("n_regimes", 3)
    log.info("Regime detector: Markov-Switching su realized vol BTC oraria ...")
    regime_model = RegimeMarkovBTC(n_regimes=n_regimes)
    try:
        # IT: cadenza walk-forward da config (hmm_burn_in_days/hmm_retrain_days):
        #     su storie multi-anno il refit expanding è O(t) — vedi commento in default.yaml.
        # EN: walk-forward cadence from config (hmm_burn_in_days/hmm_retrain_days):
        #     on multi-year histories the expanding refit is O(t) — see comment in default.yaml.
        regime_df = regime_model.fit_predict_walkforward(
            df_macro,
            burn_in_days = mcfg.get("hmm_burn_in_days", 30),
            retrain_days = mcfg.get("hmm_retrain_days", 90),
        )
        # IT: stesso filename del MS per backward compat (consumer non toccati).
        # EN: same filename as MS for backward compat (consumers untouched).
        hmm_path = out / "regime_hmm.pkl"
        regime_model.save(str(hmm_path))
        log.info(f"RegimeMarkovBTC salvato → {hmm_path}")

        regime_path = out / "regime_probs.parquet"
        atomic_save_parquet(regime_df, regime_path)
        log.info(f"Probabilità regime → {regime_path}")

        # IT: B7 — persisti anche il checkpoint della catena walk-forward: rende
        #     possibile il refresh incrementale (--regime-incremental, minuti vs ore).
        #     Fallimento non fatale: il full rebuild resta valido anche senza checkpoint.
        # EN: B7 — also persist the walk-forward chain checkpoint: enables the
        #     incremental refresh (--regime-incremental, minutes vs hours).
        #     Non-fatal on failure: the full rebuild stays valid without a checkpoint.
        try:
            chain = regime_model._engine._wf_state
            if chain is not None and len(regime_df):
                ckpt = regime_model.build_wf_checkpoint(chain, regime_df.index[-1])
                regime_model.save_wf_checkpoint(ckpt, str(out / "regime_wf_checkpoint.pkl"))
        except Exception as e:
            log.warning(f"Checkpoint walk-forward NON salvato ({e}) — "
                        f"l'incrementale richiederà un bootstrap.")

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
    return regime_df


# IT: refresh incrementale del regime (B7): checkpoint + parquet esistenti → append
#     delle sole barre nuove. FAIL-FAST su ogni incoerenza (niente try/except-inghiotti
#     come nel full rebuild: un append sbagliato avvelenerebbe il parquet).
# EN: incremental regime refresh (B7): existing checkpoint + parquet → append of the
#     new bars only. FAIL-FAST on any inconsistency (no swallow-all try/except like
#     the full rebuild: a wrong append would poison the parquet).
def run_regime_incremental(mcfg: dict, out: Path) -> int:
    import pickle
    n_regimes   = mcfg.get("n_regimes", 3)
    ckpt_path   = out / "regime_wf_checkpoint.pkl"
    regime_path = out / "regime_probs.parquet"
    if not ckpt_path.exists():
        raise RuntimeError(
            f"{ckpt_path} assente: lancia prima --regime-bootstrap-checkpoint "
            f"(o un full rebuild --regime-only, che ora lo salva)."
        )
    if not regime_path.exists():
        raise RuntimeError(f"{regime_path} assente: serve un full rebuild (--regime-only).")

    probs_old = pd.read_parquet(regime_path)

    # IT: coerenza checkpoint↔parquet PRIMA di girare: n_bars del checkpoint deve
    #     coincidere con le righe del parquet (un crash tra i due save li disallinea);
    #     idem la cadenza config vs quella congelata nel checkpoint.
    # EN: checkpoint↔parquet coherence BEFORE running: checkpoint n_bars must match
    #     the parquet rows (a crash between the two saves misaligns them); same for
    #     config cadence vs the one frozen in the checkpoint.
    with open(ckpt_path, "rb") as f:
        _ck = pickle.load(f)
    if int(_ck["chain"]["n_bars"]) != len(probs_old):
        raise RuntimeError(
            f"Checkpoint (n_bars={_ck['chain']['n_bars']}) ≠ parquet "
            f"({len(probs_old)} righe): rilancia --regime-bootstrap-checkpoint."
        )
    # IT: guard anti-stale (audit MINOR-1): il posteriore filtrato del checkpoint
    #     DEVE coincidere bit-per-bit con l'ultima riga del parquet (invariante by
    #     construction del run che li ha scritti entrambi). Un rebuild ri-lanciato
    #     sullo STESSO span e crashato tra i due save passerebbe il check n_bars
    #     ma non questo: catene diverse → append avvelenato evitato.
    # EN: anti-stale guard (audit MINOR-1): the checkpoint's filtered posterior
    #     MUST match the parquet's last row bit-for-bit (by-construction invariant
    #     of the run that wrote both). A rebuild re-run on the SAME span crashing
    #     between the two saves would pass the n_bars check but not this one:
    #     different chains → poisoned append avoided.
    _prob_cols = [f"regime_prob_{i}" for i in range(_ck["n_regimes"])]
    if not np.array_equal(np.asarray(_ck["chain"]["last_filtered"]),
                          probs_old[_prob_cols].iloc[-1].values):
        raise RuntimeError(
            "Checkpoint STALE: posteriore filtrato ≠ ultima riga del parquet "
            "(catena di un run diverso — crash tra i due save?): rilancia "
            "--regime-bootstrap-checkpoint."
        )
    cfg_burn, cfg_ret = mcfg.get("hmm_burn_in_days", 30) * 24, mcfg.get("hmm_retrain_days", 90) * 24
    if (int(_ck["chain"]["burn_in_bars"]), int(_ck["chain"]["retrain_bars"])) != (cfg_burn, cfg_ret):
        raise RuntimeError(
            f"Cadenza checkpoint (burn={_ck['chain']['burn_in_bars']}h, "
            f"retrain={_ck['chain']['retrain_bars']}h) ≠ config ({cfg_burn}h, {cfg_ret}h): "
            f"cadenza cambiata → full rebuild richiesto."
        )

    regime_model = RegimeMarkovBTC(n_regimes=n_regimes)
    # IT: expected_index (audit MINOR-2): valida che l'aggregazione oraria dello
    #     span vecchio riproduca ESATTAMENTE l'index del parquet (revisioni
    #     in-place della storia candele → fail-fast, non solo la frontiera).
    # EN: expected_index (audit MINOR-2): validates that the hourly aggregation of
    #     the old span reproduces EXACTLY the parquet index (in-place candle
    #     history revisions → fail-fast, not just the boundary).
    df_new, ckpt = regime_model.continue_from_checkpoint(
        str(ckpt_path), expected_index=probs_old.index,
    )
    if len(df_new) == 0:
        log.info("Nessuna barra nuova oltre il checkpoint: parquet già aggiornato.")
        return 0

    combined = pd.concat([probs_old, df_new])
    # IT: invarianti dell'append: index strettamente crescente, zero duplicati.
    # EN: append invariants: strictly increasing index, zero duplicates.
    if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
        raise RuntimeError("Append incoerente (duplicati o index non monotono): abort.")

    # IT: ordine parquet→checkpoint: un crash nel mezzo lascia al peggio un checkpoint
    #     stale (rilevato dal check n_bars al giro dopo), MAI un parquet avanti.
    # EN: parquet→checkpoint order: a mid-crash leaves at worst a stale checkpoint
    #     (caught by the n_bars check next run), NEVER a parquet ahead.
    atomic_save_parquet(combined, regime_path)
    log.info(f"Probabilità regime (append {len(df_new)} righe) → {regime_path}")
    regime_model.save_wf_checkpoint(ckpt, str(ckpt_path))
    return len(df_new)


# IT: pipeline macro — download FRED/yfinance, regime MS, normalizer, merge nel dataset NN
# EN: macro pipeline — download FRED/yfinance, MS regime, normalizer, merge into NN dataset
def main():
    # IT: Console Windows default cp1252 — i caratteri unicode dei banner (═, ✓, █)
    #     crashano il print. Reconfigure UTF-8 (stesso fix di 01/02/04).
    # EN: Windows console defaults to cp1252 — unicode banner chars (═, ✓, █)
    #     crash the print. Reconfigure UTF-8 (same fix as 01/02/04).
    import sys as _sys
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # IT: --regime-only rigenera SOLO regime_probs.parquet + regime_hmm.pkl (il detector
    #     legge raw_candles.parquet, ignora df_macro). Salta download FRED/yfinance, refit
    #     del MacroNormalizer, update PipelineState e merge npz: nessun artefatto consumato
    #     dai modelli production viene toccato. Default (senza flag) = pipeline completa,
    #     comportamento bit-invariato.
    # EN: --regime-only regenerates ONLY regime_probs.parquet + regime_hmm.pkl (the detector
    #     reads raw_candles.parquet, ignores df_macro). Skips FRED/yfinance download,
    #     MacroNormalizer refit, PipelineState update and npz merge: no artifact consumed
    #     by production models is touched. Default (no flag) = full pipeline, bit-identical.
    parser = argparse.ArgumentParser(description="01b — macro download + regime detection")
    # IT: i tre modi regime sono mutuamente esclusivi (default senza flag = pipeline completa).
    # EN: the three regime modes are mutually exclusive (default with no flag = full pipeline).
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--regime-only", action="store_true",
                      help="rigenera solo il regime detector (skip macro/normalizer/npz) "
                           "/ regenerate only the regime detector")
    mode.add_argument("--regime-incremental", action="store_true",
                      help="B7: estende regime_probs.parquet alle sole barre nuove dal "
                           "checkpoint (minuti, 0-1 fit MLE) / extend to new bars only")
    mode.add_argument("--regime-bootstrap-checkpoint", action="store_true",
                      help="B7: ricostruisce il checkpoint walk-forward da pkl+parquet "
                           "esistenti con golden test integrato (1 fit MLE, no rebuild) "
                           "/ rebuild the checkpoint from existing artifacts")
    args = parser.parse_args()

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
  01b · DOWNLOAD DATI MACRO{' · MODE: --regime-only' if args.regime_only else ''}
  Periodo    : {start} → oggi
  FRED key   : {'✓ configurata' if fred_key else '⚠ non configurata (rate limit)'}
  N. regimi  : {n_regimes}
{'═'*60}
""")

    # IT: path --regime-incremental (B7): append delle barre nuove dal checkpoint, poi exit.
    #     Fail-fast: le eccezioni PROPAGANO (exit != 0), nessun fallback silenzioso.
    # EN: --regime-incremental path (B7): append new bars from the checkpoint, then exit.
    #     Fail-fast: exceptions PROPAGATE (exit != 0), no silent fallback.
    if args.regime_incremental:
        n_new = run_regime_incremental(mcfg, out)
        # IT: con 0 barre nuove NESSUN file viene scritto (return anticipato).
        # EN: with 0 new bars NO file is written (early return).
        files_line = (f"""  File aggiornati in {out}/:
    ✓ regime_probs.parquet   (append)
    ✓ regime_wf_checkpoint.pkl""" if n_new else
                      "  Nessun file toccato (parquet già alla frontiera candele).")
        print(f"""
{'═'*60}
  01b · REGIME-INCREMENTAL · COMPLETATO
{'═'*60}
  Barre nuove appese  : {n_new}
{files_line}
  ⚠ regime_hmm.pkl NON aggiornato (fit finale full-sample: solo il
    full rebuild --regime-only lo rigenera; predict_proba lo usa live).
{'═'*60}
""")
        return

    # IT: path --regime-bootstrap-checkpoint (B7): una-tantum, ricostruisce il checkpoint
    #     dagli artefatti esistenti e lo valida contro il parquet (golden). Exit != 0 su FAIL.
    # EN: --regime-bootstrap-checkpoint path (B7): one-off, rebuilds the checkpoint from
    #     existing artifacts and validates it against the parquet (golden). Exit != 0 on FAIL.
    if args.regime_bootstrap_checkpoint:
        regime_model = RegimeMarkovBTC(n_regimes=n_regimes)
        report = regime_model.bootstrap_wf_checkpoint(
            hmm_path        = str(out / "regime_hmm.pkl"),
            parquet_path    = str(out / "regime_probs.parquet"),
            checkpoint_path = str(out / "regime_wf_checkpoint.pkl"),
            burn_in_days    = mcfg.get("hmm_burn_in_days", 30),
            retrain_days    = mcfg.get("hmm_retrain_days", 90),
        )
        print(f"""
{'═'*60}
  01b · REGIME-BOOTSTRAP-CHECKPOINT · COMPLETATO (golden PASS)
{'═'*60}
  Barre validate replay : {report['n_validated']}  (da t={report['last_retrain']})
  max|Δprob| vs parquet : {report['max_diff']:.3e}  (bit-exact: {report['exact']})
  File generato in {out}/:
    ✓ regime_wf_checkpoint.pkl
{'═'*60}
""")
        return

    # IT: path --regime-only: solo step 4 (regime), poi exit. Nessun file macro/npz/state toccato.
    # EN: --regime-only path: step 4 (regime) only, then exit. No macro/npz/state file touched.
    if args.regime_only:
        regime_df = run_regime_detection(mcfg, out, df_macro=None)
        n_rows = len(regime_df)
        last_ts = regime_df.index.max() if n_rows else "n/d"
        # IT: (audit MINOR-5) banner checkpoint condizionale all'esito reale del
        #     save (il fallimento è non-fatale ma NON va dichiarato "✓"): fresco =
        #     esiste ed è coevo/posteriore al parquet appena scritto.
        # EN: (audit MINOR-5) checkpoint banner conditional on the actual save
        #     outcome (failure is non-fatal but must NOT be reported as "✓"):
        #     fresh = exists and is coeval/newer than the just-written parquet.
        _ck, _rp = out / "regime_wf_checkpoint.pkl", out / "regime_probs.parquet"
        ckpt_line = (
            "✓ regime_wf_checkpoint.pkl  (B7: abilita --regime-incremental)"
            if _ck.exists() and _rp.exists()
               and _ck.stat().st_mtime >= _rp.stat().st_mtime - 5
            else "⚠ regime_wf_checkpoint.pkl NON salvato o STALE (warning nel log) "
                 "— per l'incrementale: --regime-bootstrap-checkpoint"
        )
        print(f"""
{'═'*60}
  01b · REGIME-ONLY · COMPLETATO
{'═'*60}
  Regimi (Markov-BTC) : {n_regimes} su realized vol BTC oraria
  Righe orarie        : {n_rows}  (ultima: {last_ts})

  File generati in {out}/:
    ✓ regime_hmm.pkl
    ✓ regime_probs.parquet
    {ckpt_line}

  ⚠ Macro features / normalizer / lstm_dataset.npz NON toccati (by design).
{'═'*60}
""")
        return

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
    #     — df_macro è ignorato dal detector (usa raw_candles.parquet); sezione condivisa col
    #     path --regime-only (helper run_regime_detection).
    # EN: 4. Markov-Switching regime detection on BTC realized vol (Variant 3 MODEL_IMPROVEMENTS)
    #     — df_macro is ignored by the detector (uses raw_candles.parquet); section shared with
    #     the --regime-only path (run_regime_detection helper).
    regime_df = run_regime_detection(mcfg, out, df_macro=df_macro)

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
