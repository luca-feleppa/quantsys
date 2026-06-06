"""
Script 04 — Live Signals Engine.
Connette il modello LSTM addestrato al feed WebSocket di Binance,
genera segnali BUY/SELL/HOLD in tempo reale e logga tutto su file.

Run configuration PyCharm:
  Script: scripts/04_live_signals.py
  Working dir: <root del progetto>
  Environment: CUDA_VISIBLE_DEVICES=0

NOTA: questo script NON esegue ordini reali. Genera segnali e li
      logga in results/live_signals.jsonl per analisi successive.
      Per il trading reale è necessaria una API key Binance con permessi
      di trading — non inclusa in questo progetto per sicurezza.

Interrompi con: Ctrl+C
"""
import asyncio
import json
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# IT: cap thread BLAS/OMP prima di importare numpy/torch
# EN: cap BLAS/OMP threads before importing numpy/torch
import yaml as _yaml
with open(Path(__file__).resolve().parent.parent / "config" / "default.yaml", encoding="utf-8") as _f:
    _cpu_frac = _yaml.safe_load(_f).get("hardware", {}).get("cpu_fraction", 0.5)
_cpu_limit = str(max(1, int(os.cpu_count() * _cpu_frac)))
os.environ.setdefault("OMP_NUM_THREADS", _cpu_limit)
os.environ.setdefault("MKL_NUM_THREADS", _cpu_limit)

import numpy as np
import pandas as pd
import requests
import torch

torch.set_num_threads(int(_cpu_limit))

import threading

from quantsys.utils import load_config, setup_logging, setup_device, ensure_dirs
from quantsys.model.ensemble import EnsembleModel
from quantsys.trading import SignalGenerator, RiskManager, Side, CloseReason

setup_logging(logging.INFO)
log = logging.getLogger("quantsys.live")


# IT: codici colore ANSI per output console leggibile
# EN: ANSI color codes for readable console output
GRN  = "\033[92m"; RED  = "\033[91m"; YEL = "\033[93m"
CYN  = "\033[96m"; DIM  = "\033[2m";  RST = "\033[0m"; BOLD = "\033[1m"

# IT: formatta BUY/SELL/HOLD con badge colorato
# EN: format BUY/SELL/HOLD with colored badge
def colored_signal(sig: str) -> str:
    if sig == "BUY":  return f"{GRN}{BOLD}▲ BUY {RST}"
    if sig == "SELL": return f"{RED}{BOLD}▼ SELL{RST}"
    return f"{YEL}◆ HOLD{RST}"


# IT: sanity check candela WS — scarta dati corrotti (high<low, prezzi<=0, spike/drop)
# EN: WS candle sanity check — drops corrupt data (high<low, prices<=0, spike/drop)
def _is_valid_candle(c: dict) -> bool:
    """
    Sanity check su una candela dal WebSocket Binance.
    Scarta candele con dati palesemente corrotti:
      · high < low (impossibile fisicamente)
      · prezzi zero o negativi (feed error o halt)
      · spike > 10x o drop > 90% rispetto al close (errore di feed)
      · volume negativo

    Nota: non scarta candele con volume = 0 (esistono nei mercati poco liquidi).
    """
    try:
        o = c["open"]; h = c["high"]; lo = c["low"]; cl = c["close"]
        v = c["volume"]

        if any(x <= 0 for x in [o, h, lo, cl]):   return False
        if h < lo:                                  return False   # IT: high < low | EN: high < low
        if v < 0:                                   return False   # IT: volume negativo | EN: negative volume
        if h > cl * 10:                             return False   # IT: spike >10x feed error | EN: spike >10x feed error
        if lo < cl * 0.1:                           return False   # IT: drop >90% | EN: drop >90%
        if not all(map(lambda x: x == x, [o, h, lo, cl, v])):  # IT: check NaN (x!=x) | EN: NaN check (x!=x)
            return False
        return True
    except (KeyError, TypeError):
        return False


# IT: feature builder leggero per inferenza live (sottoinsieme di 01_*)
# EN: lightweight feature builder for live inference (subset of 01_*)
# IT: Buffer ring di candele OHLCV grezze per il nuovo live engine (BLOCKER #1 Stage 4).
#     Sostituisce LiveFeatureBuffer come buffer raw, delegando l'intero feature engineering
#     a quantsys.features.FeatureBuilder (single source of truth condivisa col training).
# EN: Ring buffer of raw OHLCV candles for the new live engine (BLOCKER #1 Stage 4).
#     Replaces LiveFeatureBuffer as the raw buffer, delegating all feature engineering
#     to quantsys.features.FeatureBuilder (single source of truth shared with training).
class LiveCandleBuffer:
    """Buffer raw OHLCV per il live engine — feature engineering delegato a FeatureBuilder.

    Diversamente da LiveFeatureBuffer (legacy, calcolava 39 feature a mano), questo
    componente mantiene solo le candele grezze. FeatureAssembler (step 4.5) le
    consumerà chiamando FeatureBuilder.build() — garantisce parity esatta col training.

    Capacità default 50000 ≈ 35 giorni: sufficiente per warmup completo di tutte le
    feature 30d (dist_ath_30d, momentum_30d, price_vs_ma200m a 43200 candele).
    Memoria: ~5 MB con dict-of-floats, trascurabile.
    """

    # IT: Campi richiesti per essere compatibili col FeatureBuilder (training schema).
    # EN: Required fields for FeatureBuilder compatibility (training schema).
    REQUIRED_FIELDS = (
        "open", "high", "low", "close", "volume",
        "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol",
        "open_time",
    )

    # IT: Inizializza il deque a capacita' fissa (FIFO automatico su overflow).
    # EN: Initializes the deque with fixed capacity (auto-FIFO on overflow).
    def __init__(self, maxlen: int = 50000):
        self._candles: deque = deque(maxlen=maxlen)

    # IT: Pre-carica le ultime n_last candele da raw_candles.parquet (warmup boot).
    # IT: open_time → Timestamp tz-naive UTC. Uniforma le sorgenti: parquet (Timestamp, spesso
    #     tz-AWARE) e WS/REST (`ts`=epoch-ms int). Senza uniformazione il buffer mischia tz-aware
    #     e tz-naive → ValueError "Cannot mix tz-aware with tz-naive" nel build (bug smoke 2026-06-05).
    # EN: open_time → tz-naive UTC Timestamp. Uniforms sources: parquet (Timestamp, often tz-AWARE)
    #     and WS/REST (`ts`=epoch-ms int). Otherwise the buffer mixes tz-aware/naive → ValueError.
    @staticmethod
    def _norm_ts(ot) -> "pd.Timestamp":
        if isinstance(ot, (int, float)):
            return pd.Timestamp(ot, unit="ms")                  # epoch-ms → tz-naive UTC
        t = pd.Timestamp(ot)
        return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t

    # EN: Pre-loads the last n_last candles from raw_candles.parquet (warmup boot).
    def bootstrap_from_parquet(self, path: str, n_last: int | None = None) -> int:
        """Carica le ultime n_last candele da disco. Ritorna n caricate.

        Se path non esiste: warning + ritorna 0 (buffer parte vuoto, dovrà ricostruirsi
        via REST/WS, ma molte feature 30d saranno NaN finché il buffer non si riempie).
        """
        p = Path(path)
        if not p.exists():
            log.warning(f"LiveCandleBuffer.bootstrap: {path} non trovato — buffer parte vuoto")
            return 0
        df = pd.read_parquet(p)
        n_last = n_last or self._candles.maxlen
        df = df.iloc[-n_last:]
        for _, row in df.iterrows():
            self._candles.append({
                "open_time":           self._norm_ts(row["open_time"]),
                "open":                float(row["open"]),
                "high":                float(row["high"]),
                "low":                 float(row["low"]),
                "close":               float(row["close"]),
                "volume":              float(row["volume"]),
                "quote_vol":           float(row.get("quote_vol", 0.0)),
                "trades":              int(row.get("trades", 0)),
                "taker_buy_vol":       float(row.get("taker_buy_vol", 0.0)),
                "taker_buy_quote_vol": float(row.get("taker_buy_quote_vol", 0.0)),
            })
        log.info(f"LiveCandleBuffer: bootstrap {len(self._candles)} candele da {path}")
        return len(self._candles)

    # IT: Aggiunge una candela (normalizza schema, default 0 per campi assenti).
    # EN: Appends a candle (normalizes schema, defaults to 0 for missing fields).
    def append(self, candle: dict) -> None:
        """Append una candela. Campi mancanti → default 0 (quote_vol/trades/taker_buy_quote_vol).

        Il WS Binance kline-1m ritorna tutti i campi richiesti; il caller deve solo
        estrarli dai k[] del payload (vedi WS handler in LiveEngine).
        """
        # IT: Normalizza open_time a Timestamp tz-naive UTC. WS/REST passano `ts`=epoch-ms (int);
        #     il bootstrap da parquet passa già Timestamp. Senza coercizione il buffer mischia
        #     int e Timestamp → index `object` → `.dt` crasha nel FeatureBuilder (bug smoke 2026-06-05).
        # EN: Normalize open_time to a tz-naive UTC Timestamp. WS/REST pass `ts`=epoch-ms (int); the
        #     parquet bootstrap passes Timestamps. Without coercion the buffer mixes int and Timestamp
        #     → object index → `.dt` accessor crashes in FeatureBuilder (smoke-test bug 2026-06-05).
        _raw_ot = candle.get("open_time")
        if _raw_ot is None:
            _raw_ot = candle.get("ts")
        normalized = {
            "open_time":           self._norm_ts(_raw_ot),
            "open":                float(candle["open"]),
            "high":                float(candle["high"]),
            "low":                 float(candle["low"]),
            "close":               float(candle["close"]),
            "volume":              float(candle["volume"]),
            "quote_vol":           float(candle.get("quote_vol", 0.0)),
            "trades":              int(candle.get("trades", 0)),
            "taker_buy_vol":       float(candle.get("taker_buy_vol", 0.0)),
            "taker_buy_quote_vol": float(candle.get("taker_buy_quote_vol", 0.0)),
        }
        self._candles.append(normalized)

    # IT: Numero di candele attualmente in buffer | EN: Number of candles currently buffered
    def __len__(self) -> int:
        return len(self._candles)

    # IT: Ritorna le ultime n_last candele come DataFrame (index=open_time tz-naive).
    # EN: Returns the last n_last candles as a DataFrame (index=open_time tz-naive).
    def to_dataframe(self, n_last: int | None = None) -> pd.DataFrame:
        """Ritorna le candele come DataFrame compatibile con FeatureBuilder.build().

        Schema output: index=open_time (datetime tz-naive),
        colonne = open/high/low/close/volume/quote_vol/trades/taker_buy_vol/taker_buy_quote_vol.
        """
        if not self._candles:
            return pd.DataFrame()
        items = list(self._candles)[-n_last:] if n_last else list(self._candles)
        df = pd.DataFrame(items)
        # IT: FeatureBuilder vuole open_time come colonna o index; lo mettiamo come index.
        # EN: FeatureBuilder expects open_time as column or index; we set it as index.
        if "open_time" in df.columns:
            df = df.set_index("open_time")
            # IT: garantisce un DatetimeIndex anche se l'index arriva object (difesa: append
            #     coercizza già a Timestamp, ma questo protegge ogni altra fonte).
            # EN: guarantee a DatetimeIndex even if the index arrives as object (append already
            #     coerces to Timestamp; this protects any other source).
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # IT: tz-naive UTC per evitare mismatch con merge_asof di FeatureBuilder.
            # EN: tz-naive UTC to avoid merge_asof mismatch in FeatureBuilder.
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    # IT: Ultima candela in buffer (None se vuoto) | EN: Last candle in buffer (None if empty)
    @property
    def latest(self) -> dict | None:
        return self._candles[-1] if self._candles else None


# IT: Assembla il vettore feature (120, 104) live usando FeatureBuilder come single source of truth.
#     Garantisce parity con training: stesso codice, stessi parametri, stesso scaler.
# EN: Assembles the live feature vector (120, 104) using FeatureBuilder as single source of truth.
#     Guarantees training parity: same code, same parameters, same scaler.
class FeatureAssembler:
    """Produce il tensore feature pronto per il modello (BLOCKER #1 Stage 4).

    Pipeline interna:
      1. df = LiveCandleBuffer.to_dataframe() — tutto il buffer (~50k candele)
      2. FeatureBuilder.build(df, fit=False, normalize=True, funding_df=funding)
         usa lo scaler caricato da PipelineState → parity esatta col training
      3. Verifica canonical_names ⊆ feat_df.columns (HARD-FAIL su mancanze)
      4. Riordina + filtra solo le 104 colonne canoniche
      5. Drop NaN warmup
      6. Estrai ultime `window_size` righe → np.ndarray (window_size, 104)

    Sostituisce il _compute_features di LiveFeatureBuffer (legacy 39-feature).
    """

    # IT: Configura FeatureBuilder con parametri da config + scaler da PipelineState.
    # EN: Configures FeatureBuilder with config params + scaler from PipelineState.
    def __init__(self, buffer: "LiveCandleBuffer", pipeline_state,
                 config: dict | None = None):
        """Args:
            buffer: LiveCandleBuffer già popolato (bootstrap + append da WS)
            pipeline_state: PipelineState caricato (deve avere scaler fittato)
            config: dict completo (da load_config). Se None → carica da default.yaml.
        """
        from quantsys.features import FeatureBuilder, get_canonical_feature_names
        from quantsys.utils import load_config

        self.buffer = buffer
        self.ps = pipeline_state

        if config is None:
            config = load_config()
        fcfg = config.get("features", {})
        mcfg = config.get("model", {})

        # IT: FeatureBuilder con stessi parametri usati in training (da config).
        # EN: FeatureBuilder with same params used at training (from config).
        self.fb = FeatureBuilder(
            vp_bins          = fcfg.get("vp_bins", 30),
            vp_lookback      = fcfg.get("vp_lookback", 240),
            windows          = fcfg.get("windows", [5, 10, 20, 60]),
            lag_periods      = fcfg.get("lag_periods", 5),
            forecast_horizon = fcfg.get("forecast_horizon", 1),
            vp_stride        = fcfg.get("vp_stride", 1),
            frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
            use_revin        = bool(mcfg.get("use_revin", False)),
        )
        # IT: Inietta stato scaler pre-fittato → build(fit=False) lo riusa senza re-fittare.
        # EN: Inject pre-fitted scaler state → build(fit=False) reuses it without re-fitting.
        self.fb.scaler             = pipeline_state.scaler
        self.fb._scale_cols        = list(pipeline_state.scale_cols)
        self.fb.scalers            = dict(pipeline_state.price_scaler_state)
        self.fb.clip_lo_           = pipeline_state.clip_lo_
        self.fb.clip_hi_           = pipeline_state.clip_hi_
        self.fb.feature_cols       = list(pipeline_state.feature_cols)
        self.fb.n_dynamic_features = pipeline_state.n_dynamic_features

        # IT: Lista canonica delle 104 feature attese dal modello (single source of truth).
        # EN: Canonical list of the 104 features expected by the model (single source of truth).
        self.canonical_names: tuple[str, ...] = get_canonical_feature_names()
        log.info(f"FeatureAssembler: pronto per {len(self.canonical_names)} feature canoniche")

    # IT: Costruisce il window (window_size, 104) chiamando FeatureBuilder sul buffer corrente.
    # EN: Builds the (window_size, 104) window by calling FeatureBuilder on the current buffer.
    def compute_window(self, window_size: int = 120,
                       funding_df: pd.DataFrame | None = None) -> np.ndarray:
        """Ritorna il tensore (window_size, 104) pronto per il modello.

        Args:
            window_size: numero di candele finali da restituire (default 120, matches training)
            funding_df: DataFrame funding rate da FundingRatePoller (opzionale; senza, le 3
                        feature funding_rate* saranno NaN → mancheranno dal canonical → HARD-FAIL)

        Raises:
            RuntimeError: se buffer insufficiente, se feature canoniche mancanti, o se
                          dopo drop NaN restano meno di window_size righe valide.
        """
        if len(self.buffer) < window_size + 60:
            raise RuntimeError(
                f"FeatureAssembler: buffer insufficiente: {len(self.buffer)} < {window_size + 60} "
                f"(serve warmup completo prima di compute_window)"
            )

        df = self.buffer.to_dataframe()
        # IT: FeatureBuilder.build vuole open_time come colonna (non solo index).
        # EN: FeatureBuilder.build wants open_time as a column (not just index).
        if df.index.name == "open_time":
            df = df.reset_index()

        # IT: Normalizza funding_df a tz-naive per coerenza con buffer (FeatureBuilder
        #     fa reindex su open_time e crasha su dtype mismatch tz-aware vs tz-naive).
        # EN: Normalize funding_df to tz-naive for buffer coherence (FeatureBuilder
        #     reindexes on open_time and fails on dtype mismatch tz-aware vs tz-naive).
        if funding_df is not None and len(funding_df) > 0:
            funding_df = funding_df.copy()
            if "open_time" in funding_df.columns:
                ot = pd.to_datetime(funding_df["open_time"])
                if getattr(ot.dt, "tz", None) is not None:
                    ot = ot.dt.tz_convert("UTC").dt.tz_localize(None)
                funding_df["open_time"] = ot
            elif isinstance(funding_df.index, pd.DatetimeIndex) and funding_df.index.tz is not None:
                funding_df.index = funding_df.index.tz_convert("UTC").tz_localize(None)

        feat_df = self.fb.build(df, normalize=True, fit=False, funding_df=funding_df)

        # IT: Verifica hard-fail che tutte le 104 feature canoniche siano presenti.
        # EN: Hard-fail check that all 104 canonical features are present.
        missing = set(self.canonical_names) - set(feat_df.columns)
        if missing:
            sample = sorted(missing)[:10]
            raise RuntimeError(
                f"FeatureAssembler: {len(missing)} feature canoniche mancanti dall'output "
                f"FeatureBuilder.build. Sample: {sample}. "
                f"Verifica funding_df (3 feature) e warmup buffer (>43200 candele per 30d)."
            )

        # IT: Riordina nell'ordine canonico (no pad/truncate posizionale).
        # EN: Reorder in canonical order (no positional pad/truncate).
        feat_df = feat_df[list(self.canonical_names)]

        # IT: Drop righe con NaN (warmup iniziale).
        # EN: Drop rows with NaN (initial warmup).
        feat_df = feat_df.dropna()

        if len(feat_df) < window_size:
            raise RuntimeError(
                f"FeatureAssembler: solo {len(feat_df)} righe valide post-NaN, "
                f"servono {window_size}. Buffer warmup insufficiente."
            )

        window = feat_df.iloc[-window_size:].values.astype(np.float32)
        return window


class LiveFeatureBuffer:
    """
    Buffer circolare che mantiene le ultime `window` candele e costruisce
    le features necessarie per l'inferenza della LSTM in tempo reale.

    DEPRECATED 2026-06-02 (Stage 4 BLOCKER #1): produce solo 39 feature disallineate
    vs le 104 attese dal training. Sostituito da LiveCandleBuffer + FeatureAssembler.
    Mantenuto temporaneamente come fallback durante la migrazione.

    CORREZIONE — Volume Profile incrementale:
      La versione precedente ricalcolava il VP da zero ad ogni candela
      iterando su tutto il buffer (O(N) per ogni tick).
      Ora usiamo un aggiornamento incrementale O(1):
        · Quando arriva una nuova candela → aggiungi il suo contributo al bin
        · Quando una candela esce dal buffer → sottrai il suo contributo
      Il VP è sempre aggiornato senza riscansionare tutto lo storico.
    """

    VP_BINS = 30  # IT: bin Volume Profile | EN: Volume Profile bins

    def __init__(self, window: int = 60):
        self.window      = window
        # IT: n_features rilevato dinamicamente al primo compute (no hardcoding)
        # EN: n_features detected dynamically on first compute (no hardcoding)
        self.n_features  = 0

        # IT: lookback >= max rolling usato (ma200m -> 200) + window + margine
        # EN: lookback >= max rolling used (ma200m -> 200) + window + margin
        self._lookback   = max(window + 60, 260)
        self.candles: deque = deque(maxlen=self._lookback)

        # IT: stato incrementale Volume Profile (O(1) per push invece di O(N))
        # EN: incremental Volume Profile state (O(1) per push instead of O(N))
        self._vp_bins:   np.ndarray   = np.zeros(self.VP_BINS)
        # IT: deque (bin_idx, volume) per sottrarre contributi quando escono
        # EN: deque (bin_idx, volume) to subtract contributions on eviction
        self._vp_contribs: deque      = deque(maxlen=self._lookback)
        self._vp_price_min: float     = 0.0
        self._vp_price_max: float     = 0.0
        # IT: full-reset periodico per evitare drift accumulato di float
        # EN: periodic full-reset to avoid accumulated float drift
        self._vp_reset_every: int     = 60
        self._vp_since_reset: int     = 0

        self._feat_names: list[str] = []

    # IT: legge feature_names dal dataset di training per matching live/training
    # EN: read feature_names from training dataset for live/training alignment
    def load_scalers(self, data_dir: str = "data"):
        npz = np.load(f"{data_dir}/lstm_dataset.npz", allow_pickle=True)
        self._feat_names = list(npz["feature_names"])
        self.n_features  = len(self._feat_names)
        log.info(f"Features attese: {self.n_features}")

    # IT: full-reset VP (boot iniziale o periodicamente per evitare drift)
    # EN: VP full-reset (initial boot or periodic to avoid drift)
    def _vp_full_reset(self):
        """Ricalcola il VP da zero. Chiamato solo all'avvio o ogni ~60 candele."""
        c_arr = list(self.candles)
        if len(c_arr) < 2:
            return
        highs  = np.array([c["high"]  for c in c_arr])
        lows   = np.array([c["low"]   for c in c_arr])
        vols   = np.array([c["volume"] for c in c_arr])
        closes = np.array([c["close"] for c in c_arr])
        tps    = (highs + lows + closes) / 3

        self._vp_price_min = float(lows.min())
        self._vp_price_max = float(highs.max())
        step = max((self._vp_price_max - self._vp_price_min) / self.VP_BINS, 1e-9)

        self._vp_bins = np.zeros(self.VP_BINS)
        self._vp_contribs.clear()
        for tp, vol in zip(tps, vols):
            idx = min(int((tp - self._vp_price_min) / step), self.VP_BINS - 1)
            idx = max(idx, 0)
            self._vp_bins[idx] += vol
            self._vp_contribs.append((idx, vol))

        self._vp_since_reset = 0

    # IT: espande range VP rimappando i bin (O(BINS), non O(N))
    # EN: expand VP range by remapping bins (O(BINS), not O(N))
    def _vp_expand_range(self, direction: str) -> bool:
        """
        Espande il range del VP del 10% nella direzione indicata
        rimappando i bin esistenti senza riscansionare il buffer (O(BINS)).
        Ritorna True se l'espansione ha avuto successo.
        """
        price_range = self._vp_price_max - self._vp_price_min
        if price_range <= 0:
            return False

        old_step = price_range / self.VP_BINS
        expand   = price_range * 0.10   # IT: +10% range per espansione | EN: +10% range per expansion

        if direction == "down":
            new_min  = self._vp_price_min - expand
            new_step = (self._vp_price_max - new_min) / self.VP_BINS
            shift    = int(expand / new_step)
            new_bins = np.zeros(self.VP_BINS)
            if shift < self.VP_BINS:
                new_bins[shift:] = self._vp_bins[:self.VP_BINS - shift]
            self._vp_bins = new_bins
            new_contribs  = deque(maxlen=self._vp_contribs.maxlen)
            for (idx, v) in self._vp_contribs:
                new_contribs.append((min(idx + shift, self.VP_BINS - 1), v))
            self._vp_contribs  = new_contribs
            self._vp_price_min = new_min

        else:  # "up"
            new_max  = self._vp_price_max + expand
            new_step = (new_max - self._vp_price_min) / self.VP_BINS
            new_bins = np.zeros(self.VP_BINS)
            for i in range(self.VP_BINS):
                old_price = self._vp_price_min + (i + 0.5) * old_step
                new_idx   = min(int((old_price - self._vp_price_min) / new_step), self.VP_BINS - 1)
                new_bins[new_idx] += self._vp_bins[i]
            self._vp_bins = new_bins
            new_contribs  = deque(maxlen=self._vp_contribs.maxlen)
            for (idx, v) in self._vp_contribs:
                old_price = self._vp_price_min + (idx + 0.5) * old_step
                new_idx   = min(int((old_price - self._vp_price_min) / new_step), self.VP_BINS - 1)
                new_contribs.append((max(new_idx, 0), v))
            self._vp_contribs  = new_contribs
            self._vp_price_max = new_max

        return True

    # IT: aggiorna VP incrementale O(1) + espansione preventiva range (T11)
    # EN: incremental O(1) VP update + preemptive range expansion (T11)
    def _vp_update(self, candle: dict):
        """
        Aggiorna il VP con la nuova candela senza riscansionare tutto il buffer.

        T11 — Espansione preventiva del range:
          Invece di fare un full reset O(N) ogni volta che il prezzo esce dal range,
          espande preventivamente il range del 10% quando il prezzo è entro il 2%
          dal bordo. Il remapping dei bin è O(BINS)=O(30) invece di O(N_candele).
          Su mercati con trend forte riduce i reset da ogni candela a pochi all'ora.
        """
        tp  = (candle["high"] + candle["low"] + candle["close"]) / 3
        vol = candle["volume"]

        if self._vp_price_max <= self._vp_price_min:
            self._vp_full_reset()
            return

        price_range = self._vp_price_max - self._vp_price_min
        margin      = price_range * 0.02   # IT: 2% bordo -> espansione preventiva | EN: 2% edge -> preemptive expand

        # IT: prezzo vicino al fondo del range -> espandi giu'
        # EN: price near range floor -> expand downward
        if tp < self._vp_price_min + margin:
            self._vp_expand_range("down")

        # IT: prezzo vicino al top del range -> espandi su
        # EN: price near range ceiling -> expand upward
        elif tp > self._vp_price_max - margin:
            self._vp_expand_range("up")

        # IT: ancora fuori range dopo espansione (gap forte) -> full reset
        # EN: still out of range after expand (large gap) -> full reset
        if tp < self._vp_price_min or tp > self._vp_price_max:
            self._vp_full_reset()
            return

        step    = (self._vp_price_max - self._vp_price_min) / self.VP_BINS
        new_idx = min(int((tp - self._vp_price_min) / step), self.VP_BINS - 1)
        new_idx = max(new_idx, 0)

        # IT: ORDINE CRITICO — sottrai vecchio prima di aggiungere nuovo
        # EN: CRITICAL ORDER — subtract old before adding new
        if len(self._vp_contribs) == self._vp_contribs.maxlen:
            old_idx, old_vol = self._vp_contribs[0]
            self._vp_bins[old_idx] = max(0.0, self._vp_bins[old_idx] - old_vol)

        self._vp_bins[new_idx] += vol
        self._vp_contribs.append((new_idx, vol))

        self._vp_since_reset += 1
        if self._vp_since_reset >= self._vp_reset_every:
            self._vp_full_reset()

    # IT: API pubblica — aggiunge candela e mantiene VP coerente
    # EN: public API — appends candle and keeps VP consistent
    def push(self, candle: dict):
        """Aggiunge una nuova candela al buffer e aggiorna il VP incrementalmente."""
        self.candles.append(candle)
        self._vp_update(candle)

    # IT: estrae 4 scalari dal VP (POC, VAH, VAL distance + concentration)
    # EN: extract 4 scalars from VP (POC, VAH, VAL distance + concentration)
    def _vp_features(self, current_price: float) -> tuple[float, float, float, float]:
        """
        Estrae le 4 feature scalari dal Volume Profile corrente:
          poc_dist      = distanza % dal Point of Control
          vah_dist      = distanza % dalla Value Area High  (70% del volume)
          val_dist      = distanza % dalla Value Area Low
          concentration = % del volume nel bin POC (misura di liquidità)
        """
        total = self._vp_bins.sum()
        if total < 1e-9 or self._vp_price_max <= self._vp_price_min:
            return 0.0, 0.0, 0.0, 0.0

        poc_idx   = int(self._vp_bins.argmax())
        step      = (self._vp_price_max - self._vp_price_min) / self.VP_BINS
        poc_price = self._vp_price_min + (poc_idx + 0.5) * step

        # IT: Value Area = bin che coprono 70% del volume attorno al POC
        # EN: Value Area = bins covering 70% of volume around POC
        sorted_idx = np.argsort(self._vp_bins)[::-1]
        cum, va_bins = 0.0, []
        for idx in sorted_idx:
            if cum / total >= 0.70:   # IT: soglia 70% standard VP | EN: 70% standard VP threshold
                break
            va_bins.append(idx); cum += self._vp_bins[idx]
        va_lo = self._vp_price_min + min(va_bins) * step
        va_hi = self._vp_price_min + (max(va_bins) + 1) * step

        safe_price = max(current_price, 1e-9)
        return (
            (current_price - poc_price) / safe_price,   # IT: distanza % dal POC | EN: % distance from POC
            (current_price - va_hi)     / safe_price,   # IT: distanza % da VAH | EN: % distance from VAH
            (current_price - va_lo)     / safe_price,   # IT: distanza % da VAL | EN: % distance from VAL
            float(self._vp_bins[poc_idx] / total),       # IT: concentrazione volume al POC | EN: volume concentration at POC
        )

    # IT: costruisce la finestra (window, n_features) per l'inferenza
    # EN: builds the (window, n_features) tensor for inference
    def _compute_features(self) -> np.ndarray | None:
        """
        Costruisce una finestra (window, n_features) dalle candele in buffer.
        Tutte le rolling statistics sono calcolate con pandas (no convolve),
        il VP usa lo stato incrementale già mantenuto in _vp_bins.
        """
        if len(self.candles) < self.window + 20:
            return None

        c_arr  = list(self.candles)
        closes = np.array([c["close"]   for c in c_arr])
        highs  = np.array([c["high"]    for c in c_arr])
        lows   = np.array([c["low"]     for c in c_arr])
        vols   = np.array([c["volume"]  for c in c_arr])
        # IT: taker_buy_vol calcolato una volta, riusato per taker ratio + CVD
        # EN: taker_buy_vol computed once, reused for taker ratio + CVD
        taker_buy = np.array([
            c.get("taker_buy_vol", c["volume"] * 0.5) for c in c_arr
        ])
        taker     = np.where(vols > 0, taker_buy / vols, 0.5)

        s_closes = pd.Series(closes)
        s_vols   = pd.Series(vols)

        # IT: log-returns con prepend per allineare la lunghezza
        # EN: log-returns with prepend to keep array length aligned
        log_ret  = np.log(np.maximum(closes, 1e-9))
        log_ret  = np.diff(log_ret, prepend=log_ret[0])
        s_rets   = pd.Series(log_ret)

        # IT: deviazione dal VWAP cumulativo (proxy di mean reversion)
        # EN: deviation from cumulative VWAP (mean-reversion proxy)
        tp       = (highs + lows + closes) / 3
        vwap     = np.cumsum(tp * vols) / np.maximum(np.cumsum(vols), 1e-9)
        vwap_dev = (closes - vwap) / np.maximum(vwap, 1e-9)

        # IT: z-score volume su 20 candele
        # EN: 20-bar volume z-score
        vol_mu  = s_vols.rolling(20, min_periods=1).mean()
        vol_std = s_vols.rolling(20, min_periods=1).std().fillna(1)
        vol_z   = ((s_vols - vol_mu) / vol_std.replace(0, 1)).values

        # IT: vol short/long ratio (regime breakout)
        # EN: short/long vol ratio (regime breakout)
        vol_std5  = s_rets.rolling(5,  min_periods=1).std().fillna(0).values
        vol_std20 = s_rets.rolling(20, min_periods=1).std().fillna(1).values
        vol_ratio = vol_std5 / np.maximum(vol_std20, 1e-9)

        # IT: orario del giorno via encoding ciclico (sin/cos)
        # EN: time-of-day via cyclic encoding (sin/cos)
        hours = np.array([c.get("hour", 12) + c.get("minute", 0) / 60 for c in c_arr])
        h_sin = np.sin(2 * np.pi * hours / 24)
        h_cos = np.cos(2 * np.pi * hours / 24)

        # IT: lag returns + momentum a 5/60 candele
        # EN: lag returns + 5/60-bar momentum
        lag1 = s_rets.shift(1).fillna(0).values
        lag2 = s_rets.shift(2).fillna(0).values
        lag3 = s_rets.shift(3).fillna(0).values
        lag4 = s_rets.shift(4).fillna(0).values
        lag5 = s_rets.shift(5).fillna(0).values
        mom5 = s_closes.pct_change(5).fillna(0).values
        mom60= s_closes.pct_change(min(60, len(c_arr)-1)).fillna(0).values

        # IT: CVD = Cumulative Volume Delta (pressione buy vs sell)
        # EN: CVD = Cumulative Volume Delta (buy vs sell pressure)
        delta_cvd   = taker_buy - (vols - taker_buy)
        cvd_norm    = delta_cvd / np.maximum(vols, 1e-9)
        s_delta     = pd.Series(delta_cvd)
        cvd_cum20   = s_delta.rolling(20, min_periods=1).sum()
        cvd_pct20   = (cvd_cum20 / s_vols.rolling(20, min_periods=1).sum().replace(0, np.nan)).fillna(0).values
        cvd_trend   = s_delta.rolling(20, min_periods=1).sum().fillna(0)
        price_trend = s_rets.rolling(20, min_periods=1).sum().fillna(0)
        cvd_std     = cvd_trend.rolling(60, min_periods=10).std().replace(0, np.nan).fillna(1)
        p_std       = price_trend.rolling(60, min_periods=10).std().replace(0, np.nan).fillna(1)
        cvd_div     = ((cvd_trend / cvd_std) - (price_trend / p_std)).fillna(0).values
        delta_accel = (s_delta.diff(5).fillna(0) / s_vols.rolling(5, min_periods=1).sum().replace(0, np.nan)).fillna(0).values

        # IT: microstruttura candela (sostituisce indicatori classici RSI/MACD/...)
        # EN: candle microstructure (replaces classic RSI/MACD/... indicators)
        hl_arr    = highs - lows
        hl_safe   = np.where(hl_arr > 1e-9, hl_arr, 1.0)
        opens_arr = np.array([c["open"] for c in c_arr])

        body_ratio   = np.abs(closes - opens_arr) / hl_safe
        upper_shadow = (highs - np.maximum(closes, opens_arr)) / hl_safe
        lower_shadow = (np.minimum(closes, opens_arr) - lows) / hl_safe
        close_vs_open= np.where(opens_arr > 0, (closes - opens_arr) / opens_arr, 0.0)
        intraday_pos = (closes - lows) / hl_safe

        price_vel   = np.zeros(len(c_arr))
        price_accel = np.zeros(len(c_arr))
        for i in range(3, len(c_arr)):
            if closes[i-3] > 0:
                price_vel[i] = (closes[i] - closes[i-3]) / 3 / closes[i-3]
        for i in range(1, len(c_arr)):
            price_accel[i] = price_vel[i] - price_vel[i-1]

        vwap_slope = np.zeros(len(c_arr))
        for i in range(5, len(c_arr)):
            if vwap[i-5] > 0:
                vwap_slope[i] = (vwap[i] - vwap[i-5]) / vwap[i-5]

        spread_proxy = np.where(vols > 0, hl_arr / vols, 0.0)

        # IT: session_pos in [-0.5, +0.5] = posizione vs range 4h (240 candele)
        # EN: session_pos in [-0.5, +0.5] = position vs 4h range (240 bars)
        h4 = pd.Series(highs).rolling(240, min_periods=1).max().values
        l4 = pd.Series(lows).rolling(240, min_periods=1).min().values
        r4 = h4 - l4
        session_pos = np.where(r4 > 0, (closes - (h4 + l4) / 2) / r4, 0.0)

        vwap_skew = np.zeros(len(c_arr))
        for i in range(20, len(c_arr)):
            sl = slice(i-20, i)
            v = vols[sl]; r = log_ret[sl]; vs = v.sum()
            if vs > 0:
                wm = (r * v).sum() / vs
                dev= r - wm
                wvar = ((dev**2) * v).sum() / vs
                if wvar > 1e-12:
                    vwap_skew[i] = ((dev**3) * v).sum() / (vs * wvar**1.5)

        # IT: feature strutturali (ATH/ATL su finestra giornaliera 1440 candele)
        # EN: structural features (ATH/ATL on daily window of 1440 bars)
        ath_buf     = pd.Series(highs).rolling(min(len(c_arr), 1440), min_periods=1).max().values
        atl_buf     = pd.Series(lows).rolling(min(len(c_arr), 1440),  min_periods=1).min().values
        pr_range    = np.maximum(ath_buf - atl_buf, 1e-9)
        dist_ath    = (closes - ath_buf) / np.maximum(ath_buf, 1e-9)
        dist_atl    = (closes - atl_buf) / np.maximum(atl_buf, 1e-9)
        price_pos   = (closes - atl_buf) / pr_range
        # IT: distanza % dal numero tondo piu' vicino (multipli di 1000$)
        # EN: % distance from nearest round number (multiples of $1000)
        round_level = (pd.Series(closes) / 1000).round() * 1000
        round_dist  = ((pd.Series(closes) - round_level) / pd.Series(closes).replace(0, np.nan)).fillna(0).values

        # IT: feature Volume Profile (broadcast scalari sull'intera finestra)
        # EN: Volume Profile features (broadcast scalars across the window)
        current_price = closes[-1]
        poc_d, vah_d, val_d, conc = self._vp_features(current_price)
        vp_poc = np.full(len(c_arr), poc_d)
        vp_vah = np.full(len(c_arr), vah_d)
        vp_val = np.full(len(c_arr), val_d)
        vp_conc= np.full(len(c_arr), conc)

        # IT: assembla — Stream A (dinamiche tempo-varianti) | Stream B (strutturali)
        # EN: assemble — Stream A (time-varying dynamics) | Stream B (structural)
        feat_mat = np.stack([
            # IT: Stream A — feature dinamiche | EN: Stream A — dynamic features
            log_ret, vwap_dev, vol_z, vol_ratio, h_sin, h_cos, taker,
            lag1, lag2, lag3, lag4, lag5, mom5, mom60,
            cvd_norm, cvd_pct20, cvd_div, delta_accel,
            vol_std5, vol_std20,
            body_ratio, upper_shadow, lower_shadow, close_vs_open,
            price_vel, price_accel, vwap_slope, spread_proxy, vwap_skew,
            intraday_pos,
            # IT: Stream B — feature strutturali | EN: Stream B — structural features
            vp_poc, vp_vah, vp_val, vp_conc,
            dist_ath, dist_atl, price_pos, round_dist,
            session_pos,
        ], axis=1)

        win = feat_mat[-self.window:]
        if win.shape[0] < self.window:
            return None

        # IT: aggiorna n_features al primo compute o se cambia
        # EN: update n_features on first compute or on change
        if self.n_features != win.shape[1]:
            self.n_features = win.shape[1]
            log.debug(f"LiveFeatureBuffer: {self.n_features} feature rilevate automaticamente")

        # IT: normalizzazione robusta vettorizzata (mediana + IQR per colonna)
        # EN: vectorized robust normalization (per-column median + IQR)
        med = np.median(win, axis=0)
        q1_q3 = np.percentile(win, [25, 75], axis=0)
        iqr = q1_q3[1] - q1_q3[0]
        mask = iqr > 1e-9
        win[:, mask] = (win[:, mask] - med[mask]) / iqr[mask]

        # IT: clip ±5σ — robusto agli outlier in inferenza live
        # EN: clip at ±5σ — robust to live-time outliers
        return np.clip(win, -5, 5).astype(np.float32)

    # IT: accessor pubblico — restituisce la finestra feature pronta per l'inferenza
    # EN: public accessor — returns the inference-ready feature window
    def get_window(self) -> np.ndarray | None:
        return self._compute_features()

    # IT: ATR semplificato sulle ultime 15 candele (TR ≈ high-low)
    # EN: simplified ATR over the last 15 candles (TR ≈ high-low)
    @property
    def atr(self) -> float:
        if len(self.candles) < 2:
            return 0.0
        c   = list(self.candles)[-15:]
        hl  = [x["high"] - x["low"] for x in c]
        return float(np.mean(hl))


# IT: motore live — WS Binance + inferenza + paper trading + persistenza stato
# EN: live engine — Binance WS + inference + paper trading + state persistence
STATE_MAX_AGE_SEC = 300   # IT: 5 min — oltre lo stato salvato e' stale | EN: 5 min — stale state threshold


class LiveEngine:
    """
    Orchestratore principale del motore live:
      1. Mantiene il buffer delle candele aggiornato via WebSocket
      2. Ad ogni candela chiusa → inferenza LSTM → segnale
      3. Logga tutto su JSONL + stampa a schermo
      4. Paper trading: traccia P&L simulato senza ordini reali

    CORREZIONE — Persistenza stato:
      Lo stato critico (buffer candele, portfolio, posizione aperta, candle_idx)
      viene serializzato su disco ad ogni candela chiusa.
      Al riavvio (o dopo un crash del WS), lo stato viene ripristinato
      automaticamente se abbastanza recente (< STATE_MAX_AGE_SEC).
      In questo modo le posizioni aperte e il P&L paper sopravvivono
      a disconnessioni di rete, riavvii del processo, crash del sistema.
    """

    # IT: setup engine — carica PipelineState/modello, avvia thread funding+macro, init RM/sig_gen
    # EN: engine setup — load PipelineState/model, start funding+macro threads, init RM/sig_gen
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.device = device
        dcfg = cfg["data"]; mcfg = cfg["model"]; rcfg = cfg["risk"]; bcfg = cfg["backtest"]

        self.symbol   = dcfg["symbol"]
        self.interval = dcfg["interval"]

        # IT: directory arch-specifiche (models/<arch>, results/<arch>)
        # EN: arch-specific directories (models/<arch>, results/<arch>)
        self._models_dir  = Path(cfg["training"]["output_dir"])
        self._state_file  = Path(bcfg["output_dir"]) / "live_engine_state.json"

        # IT: PipelineState = scaler + colonne + config training (per coerenza live/training)
        # EN: PipelineState = scaler + columns + training config (live/training parity)
        self.pipeline_state = None
        _ps_candidates = [
            self._models_dir / "pipeline_state.pkl",
            Path("models/pipeline_state.pkl"),
        ]
        for _ps_candidate in _ps_candidates:
            if _ps_candidate.exists():
                try:
                    from quantsys.utils import PipelineState
                    self.pipeline_state = PipelineState.load(str(_ps_candidate))
                    log.info(f"PipelineState caricato da {_ps_candidate}: {self.pipeline_state}")
                    # IT: copia in dir arch per accelerare prossimi avvii
                    # EN: cache in arch dir to speed up next startups
                    _arch_ps = self._models_dir / "pipeline_state.pkl"
                    if _ps_candidate != _arch_ps and not _arch_ps.exists():
                        import shutil as _sh_ps
                        _sh_ps.copy(_ps_candidate, _arch_ps)
                except Exception as e:
                    log.warning(f"PipelineState load fallito da {_ps_candidate} ({e})")
                break
        if self.pipeline_state is None:
            log.warning(f"pipeline_state.pkl non trovato in nessun path — scaler non disponibili.")
        else:
            # IT: hard-fail se forecast_horizon config != training (segnali invalidi)
            # EN: hard-fail if forecast_horizon config != training (invalid signals)
            _cfg_h = cfg.get("features", {}).get("forecast_horizon",
                       dcfg.get("forecast_horizon", 15))
            _state_h = self.pipeline_state.forecast_horizon
            if _cfg_h != _state_h:
                raise RuntimeError(
                    f"forecast_horizon mismatch: config={_cfg_h}, training={_state_h}. "
                    f"Il modello è stato addestrato per orizzonte {_state_h}; live signals a {_cfg_h} "
                    f"produce segnali invalidi. Allinea config/default.yaml o rigenera il modello."
                )
            # IT: max_hold_candles deve >= forecast_horizon (altrimenti TP/SL e' rumore)
            # EN: max_hold_candles must >= forecast_horizon (otherwise TP/SL is noise)
            _max_hold = rcfg.get("max_hold_candles", 0)
            if _max_hold < _state_h:
                log.warning(
                    f"max_hold_candles ({_max_hold}) < forecast_horizon ({_state_h}). "
                    f"Il TP/SL potrebbe non avere tempo di triggerare prima del MAX_HOLD."
                )

        # IT: funding rate — load iniziale + refresh ogni 8h via thread daemon
        # EN: funding rate — initial load + 8h refresh via daemon thread
        self._funding_df = [None]   # IT: lista mutabile per scrittura cross-thread | EN: mutable list for cross-thread write
        self._funding_lock = threading.Lock()  # IT: protegge accesso cross-thread | EN: guards cross-thread access
        _funding_path = Path("data/funding_rate.parquet")
        if _funding_path.exists():
            _initial_df = pd.read_parquet(_funding_path)
            with self._funding_lock:
                self._funding_df[0] = _initial_df
            log.info(f"Funding rate caricato: {len(_initial_df)} osservazioni")
        else:
            log.warning("data/funding_rate.parquet non trovato — funding rate feature disabilitata")

        # IT: thread daemon — aggiorna SUBITO al primo giro, poi sleep 8h
        # EN: daemon thread — refresh IMMEDIATELY on first iter, then sleep 8h
        def _funding_rate_updater():
            _first = True
            while True:
                if not _first:
                    time.sleep(28800)  # IT: 8 ore = intervallo funding Binance | EN: 8h = Binance funding interval
                _first = False
                try:
                    from quantsys.data import fetch_funding_rate
                    new_df = fetch_funding_rate(
                        symbol     = dcfg["symbol"],
                        start_time = "2021-01-01",
                        output_dir = dcfg["output_dir"],
                    )
                    with self._funding_lock:
                        self._funding_df[0] = new_df
                    log.info(f"Funding rate aggiornato: {len(new_df)} osservazioni")
                except Exception as e:
                    log.warning(f"Funding rate update fallito: {e}")

        t_fr = threading.Thread(target=_funding_rate_updater, daemon=True)
        t_fr.start()

        # IT: refresh orario snapshot macro (yfinance + FRED) per il MacroEncoder
        # EN: hourly macro snapshot refresh (yfinance + FRED) for MacroEncoder
        self.macro_updater = None
        has_macro_cols = (
            self.pipeline_state is not None and
            len(self.pipeline_state.macro_feature_cols) > 0 and
            self.pipeline_state.macro_normalizer is not None
        )
        if has_macro_cols:
            try:
                from quantsys.macro.live_snapshot import MacroSnapshotUpdater
                self.macro_updater = MacroSnapshotUpdater(
                    normalizer         = self.pipeline_state.macro_normalizer,
                    macro_feature_cols = self.pipeline_state.macro_feature_cols,
                    update_interval_sec= 3600,
                    fred_api_key       = cfg.get("macro", {}).get("fred_api_key", ""),
                )
                self.macro_updater.start()
                log.info("MacroSnapshotUpdater avviato — macro reale in inference live")
            except Exception as e:
                log.warning(f"MacroSnapshotUpdater non avviato ({e}) — uso zeros come fallback")
                self.macro_updater = None
        else:
            log.info("Modello senza macro branch — MacroSnapshotUpdater non necessario")

        # IT: preferenza ensemble eterogeneo (>=2 arch presenti), fallback a omogeneo
        # EN: prefer heterogeneous ensemble (>=2 archs present), fallback to homogeneous
        try:
            from quantsys.model.ensemble import get_distillation_archs
            _archs = get_distillation_archs(cfg)
            _het_available = sum(1 for a in _archs
                                if (Path("models") / a / "best_model.pt").exists())
            if _het_available >= 2:
                self.model = EnsembleModel.load_heterogeneous(device, cfg=cfg)
                log.info(f"Ensemble ETEROGENEO: {self.model.n_members} architetture "
                         f"[{', '.join(self.model.arch_names)}]")
            else:
                self.model = EnsembleModel.load(str(self._models_dir), device)
                log.info(f"Modello caricato: {self.model.n_members} membro/i ensemble")
            self.use_model = True
        except FileNotFoundError:
            log.warning(f"Nessun checkpoint trovato in {self._models_dir}/ — uso rolling stats.")
            self.model     = None
            self.use_model = False

        # IT: BLOCKER #1 Stage 4.6 — buffer DEPRECATED solo per ATR + state persistence + sanity di candles.
        # EN: BLOCKER #1 Stage 4.6 — DEPRECATED buffer kept only for ATR + state persistence + candle sanity.
        self.buf = LiveFeatureBuffer(window=mcfg["window_size"])

        # IT: BLOCKER #1 Stage 4.6 — nuovo buffer raw + assembler che usa FeatureBuilder
        #     come single source of truth. Produce le 104 feature canoniche col medesimo
        #     scaler del training (parity garantita da tests/test_live_training_parity.py).
        # EN: BLOCKER #1 Stage 4.6 — new raw buffer + assembler relying on FeatureBuilder
        #     as single source of truth. Produces the 104 canonical features with the same
        #     training scaler (parity guaranteed by tests/test_live_training_parity.py).
        self.candle_buffer = LiveCandleBuffer(maxlen=50_000)
        # IT: bootstrap da raw_candles.parquet (~30d storia) per warmup feature 30d-lookback.
        # EN: bootstrap from raw_candles.parquet (~30d history) for 30d-lookback feature warmup.
        _raw_path = Path("data/raw_candles.parquet")
        if _raw_path.exists():
            self.candle_buffer.bootstrap_from_parquet(str(_raw_path), n_last=50_000)
        else:
            log.warning("data/raw_candles.parquet non trovato — LiveCandleBuffer parte vuoto.")
        # IT: funding_df letto una volta al boot (workaround Stage 4.4: niente Poller).
        # EN: funding_df read once at boot (Stage 4.4 workaround: no Poller).
        with self._funding_lock:
            self.funding_df = self._funding_df[0]
        # IT: instanzia l'assembler solo se PipelineState disponibile (richiede scaler).
        # EN: instantiate assembler only if PipelineState is available (requires scaler).
        if self.pipeline_state is not None:
            self.feature_assembler = FeatureAssembler(
                self.candle_buffer, self.pipeline_state, config=cfg
            )
        else:
            self.feature_assembler = None
            log.warning("PipelineState assente — FeatureAssembler disabilitato (fallback rolling stats).")
        self.sig_gen = SignalGenerator(
            prob_threshold   = bcfg["prob_threshold"],
            min_expected_ret = bcfg["min_expected_ret"],
            max_sigma        = bcfg["max_sigma"],
            conviction_alpha = bcfg.get("conviction_alpha", 0.5),
        )
        self.rm = RiskManager(
            initial_capital    = rcfg["initial_capital"],
            max_risk_per_trade = rcfg["max_risk_per_trade"],
            sl_atr_mult        = rcfg["sl_atr_mult"],
            tp_rr_ratio        = rcfg["tp_rr_ratio"],
            max_position_pct   = rcfg["max_position_pct"],
            max_drawdown_stop  = rcfg["max_drawdown_stop"],
            max_hold_candles   = rcfg["max_hold_candles"],
            use_trailing_stop  = rcfg["use_trailing_stop"],
            trailing_atr_mult  = rcfg["trailing_atr_mult"],
            fee_rate           = bcfg["fee_rate"],
            slippage_rate             = bcfg["slippage_rate"],
            correlation_window        = rcfg.get("correlation_window", 10),
            max_directional_exposure  = rcfg.get("max_directional_exposure", 0.6),
        )

        _results_dir = Path(bcfg["output_dir"])
        _results_dir.mkdir(parents=True, exist_ok=True)
        ensure_dirs(str(_results_dir))
        self.log_path        = _results_dir / "live_signals.jsonl"
        self.candle_idx      = 0
        self.last_signal:    dict = {}
        self.last_forecast:  dict | None = None
        # IT: cadenza Monte Carlo forecast (ogni N candele chiuse)
        # EN: Monte Carlo forecast cadence (every N closed candles)
        self._forecast_every = cfg.get("montecarlo", {}).get("live_forecast_every", 10)
        self._forecast_tick  = 0
        self.session_start   = time.time()

        # IT: candela parziale (k.x=False) tenuta separata — scartata al reconnect WS
        # EN: partial candle (k.x=False) kept aside — dropped on WS reconnect
        self._pending_candle: dict | None = None

    # IT: persistenza stato — sopravvive a crash/disconnessioni brevi
    # EN: state persistence — survives crashes/brief disconnects

    # IT: serializza buffer+portfolio+posizione su disco (write atomico temp+rename)
    # EN: serialize buffer+portfolio+position to disk (atomic temp+rename write)
    def _save_state(self):
        """
        Serializza su disco:
          · ultime 200 candele del buffer (sufficiente per il warm-up)
          · stato del portfolio (cash, equity, peak, drawdown)
          · posizione aperta (se esiste)
          · candle_idx corrente

        Il file viene scritto in modo atomico (write temp + rename)
        per evitare di lasciare un JSON corrotto in caso di crash.
        """
        pos_data = None
        if self.rm.position:
            p = self.rm.position
            pos_data = {
                "side":          p.side.value,
                "entry_price":   p.entry_price,
                "size_usd":      p.size_usd,
                "size_base":     p.size_base,
                "entry_candle":  p.entry_candle,
                "stop_loss":     p.stop_loss,
                "take_profit":   p.take_profit,
                "trailing_atr":  p.trailing_atr,
                "peak_price":    p.peak_price,
            }

        port = self.rm.portfolio
        state = {
            "saved_at":   time.time(),
            "candle_idx": self.candle_idx,
            "candles":    list(self.buf.candles)[-200:],  # IT: ultimi 200 per warm-up rapido | EN: last 200 for fast warm-up
            "portfolio": {
                "equity":        port.equity,
                "cash":          port.cash,
                "peak_equity":   port.peak_equity,
                "drawdown":      port.drawdown,
                "max_drawdown":  port.max_drawdown,
                "n_trades":      port.n_trades,
                "n_wins":        port.n_wins,
                "gross_profit":  port.gross_profit,
                "gross_loss":    port.gross_loss,
            },
            "position": pos_data,
            "trades_count": len(self.rm.trades),
        }

        # IT: write temp + rename = scrittura atomica (no JSON corrotti su crash)
        # EN: write temp + rename = atomic write (no corrupt JSON on crash)
        tmp = self._state_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        tmp.replace(self._state_file)

    # IT: ripristina stato da disco se fresco (< STATE_MAX_AGE_SEC), altrimenti warm-up fresco
    # EN: restore state from disk if fresh (< STATE_MAX_AGE_SEC), else fresh warm-up
    def _load_state(self) -> bool:
        """
        Ripristina lo stato da disco se esiste ed è recente.
        Ritorna True se il ripristino è riuscito, False altrimenti.
        """
        if not self._state_file.exists():
            return False

        try:
            with open(self._state_file, encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Stato su disco corrotto ({e}) — warm-up fresco.")
            return False

        age = time.time() - state.get("saved_at", 0)
        if age > STATE_MAX_AGE_SEC:
            log.info(f"Stato su disco troppo vecchio ({age:.0f}s > {STATE_MAX_AGE_SEC}s) — warm-up fresco.")
            return False

        log.info(f"Ripristino stato da disco (età {age:.0f}s) ...")

        # IT: ripopola il buffer candele (riattiva anche il VP incrementale)
        # EN: repopulate the candle buffer (also rebuilds incremental VP)
        for c in state.get("candles", []):
            self.buf.push(c)
        self.candle_idx = state.get("candle_idx", 0)

        # IT: ripristina cash/equity/drawdown del paper portfolio
        # EN: restore paper portfolio cash/equity/drawdown
        pdata = state.get("portfolio", {})
        port  = self.rm.portfolio
        port.equity       = pdata.get("equity",       self.rm.icap)
        port.cash         = pdata.get("cash",         self.rm.icap)
        port.peak_equity  = pdata.get("peak_equity",  self.rm.icap)
        port.drawdown     = pdata.get("drawdown",     0.0)
        port.max_drawdown = pdata.get("max_drawdown", 0.0)
        port.n_trades     = pdata.get("n_trades",     0)
        port.n_wins       = pdata.get("n_wins",       0)
        port.gross_profit = pdata.get("gross_profit", 0.0)
        port.gross_loss   = pdata.get("gross_loss",   0.0)

        # IT: ripristina posizione aperta (mantiene SL/TP/trailing originali)
        # EN: restore open position (keeps original SL/TP/trailing)
        pos_data = state.get("position")
        if pos_data:
            from quantsys.trading import Position, Side as _Side
            self.rm.position = Position(
                side         = _Side(pos_data["side"]),
                entry_price  = pos_data["entry_price"],
                size_usd     = pos_data["size_usd"],
                size_base    = pos_data["size_base"],
                entry_candle = pos_data["entry_candle"],
                stop_loss    = pos_data["stop_loss"],
                take_profit  = pos_data["take_profit"],
                trailing_atr = pos_data["trailing_atr"],
                peak_price   = pos_data["peak_price"],
            )
            log.info(f"Posizione ripristinata: {self.rm.position.side.value} "
                     f"entry={self.rm.position.entry_price:,.1f} "
                     f"SL={self.rm.position.stop_loss:,.1f}")

        log.info(f"Stato ripristinato: {len(self.buf.candles)} candele, "
                 f"candle_idx={self.candle_idx}, "
                 f"equity=${port.equity:,.2f}")
        return True

    # IT: warm-up — riempie il buffer prima di abilitare le inferenze live
    # EN: warm-up — fills the buffer before enabling live inferences
    def warmup(self):
        """
        Riempie il buffer prima di avviare lo stream live.

        Strategia a due passi:
          1. Tenta di ripristinare lo stato da disco (riavvio rapido dopo crash
             o riconnessione entro STATE_MAX_AGE_SEC=5 min). Se il file è fresco,
             recuperiamo buffer + portfolio + posizione aperta senza toccare la REST API.
          2. Scarica dalla REST API le candele mancanti per portare il buffer
             a window_size + lookback (≥ 120 candele) prima che il WS inizi.

        Il numero di candele da scaricare è derivato da window_size della config
        (non hardcoded) con un overhead di +60 per rolling features stabili.
        Con window_size=60 → necessarie 120 candele minimo.
        La REST API Binance restituisce al massimo 1000 candele per chiamata.
        """
        # IT: min candele = window + 60 lookback rolling + 10 margine scarti
        # EN: min candles = window + 60 rolling lookback + 10 discard margin
        window_size  = self.cfg["model"]["window_size"]
        min_candles  = window_size + 60 + 10

        restored = self._load_state()

        if restored:
            # IT: riavvio rapido — scarica solo il gap dallo snapshot su disco
            # EN: fast restart — only fetch the gap since on-disk snapshot
            n_in_buf = len(self.buf.candles)
            needed   = max(10, min_candles - n_in_buf)
            log.info(
                f"Warm-up: stato ripristinato da disco "
                f"({n_in_buf} candele salvate). "
                f"Richiesta REST per colmare il gap ({needed} candele recenti) ..."
            )
        else:
            # IT: cold start — buffer pieno da REST
            # EN: cold start — full buffer via REST
            needed = min_candles
            log.info(
                f"Warm-up: avvio freddo — scaricamento {needed} candele storiche "
                f"(window_size={window_size} + lookback=60 + margine=10) ..."
            )

        # IT: ── A1 — CATCH-UP CONTIGUO del candle_buffer (sorgente del FeatureAssembler) ──
        #     Il bootstrap da raw_candles.parquet può essere vecchio di giorni; senza colmare
        #     il gap fino a "ora" le feature a lookback lungo (ma200m, vp, 30d) attraversano un
        #     buco temporale (bug osservato nello smoke test 2026-06-05). Scarica via REST le
        #     candele mancanti dall'ultima del buffer fino a ora (paginazione in fetch_klines) e
        #     le appende in modo CONTIGUO. Dedup su open_time. Best-effort: se fallisce, il WS
        #     colmerà gradualmente (col mirror dedup-safe sotto come fallback).
        # EN: A1 — CONTIGUOUS catch-up of candle_buffer (the FeatureAssembler source). The parquet
        #     bootstrap can be days old; without bridging the gap to "now", long-lookback features
        #     span a temporal hole (smoke-test bug 2026-06-05). Fetch the missing range via REST
        #     and append contiguously. Dedup on open_time. Best-effort.
        try:
            from quantsys.data import fetch_klines
            _cb_last = self.candle_buffer.latest
            if _cb_last is not None:
                _last_ts = self.candle_buffer._norm_ts(_cb_last["open_time"])
                _df_cb = fetch_klines(self.symbol, self.interval, 0,
                                      start_time=_last_ts.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                _last_ts = None
                _df_cb = fetch_klines(self.symbol, self.interval, min_candles)
            _n_cb = 0
            for _, _row in _df_cb.iterrows():
                _ot = self.candle_buffer._norm_ts(_row["open_time"])
                if _last_ts is not None and _ot <= _last_ts:
                    continue                                   # IT: dedup — già in buffer
                self.candle_buffer.append({
                    "open_time":           _ot,
                    "open":                float(_row["open"]),
                    "high":                float(_row["high"]),
                    "low":                 float(_row["low"]),
                    "close":               float(_row["close"]),
                    "volume":              float(_row["volume"]),
                    "quote_vol":           float(_row.get("quote_vol", 0.0)),
                    "trades":              int(_row.get("trades", 0)),
                    "taker_buy_vol":       float(_row.get("taker_buy_vol", 0.0)),
                    "taker_buy_quote_vol": float(_row.get("taker_buy_quote_vol", 0.0)),
                })
                _n_cb += 1
            if self.candle_buffer.latest is not None:
                log.info(
                    f"Catch-up candle_buffer (A1): +{_n_cb} candele REST → "
                    f"{len(self.candle_buffer)} candele, ultima {self.candle_buffer.latest['open_time']}"
                )
        except Exception as e:
            log.warning(f"Catch-up candle_buffer (A1) fallito: {e} — il WS colmerà gradualmente")

        try:
            r = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol":   self.symbol,
                    "interval": self.interval,
                    "limit":    min(needed, 1000),   # IT: limite REST Binance | EN: Binance REST cap
                },
                timeout=10,
            )
            r.raise_for_status()
            n_pushed  = 0
            n_skipped = 0
            for k in r.json():
                ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                candle = {
                    "open": float(k[1]), "high": float(k[2]),
                    "low":  float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]), "taker_buy_vol": float(k[9]),
                    "hour": ts.hour, "minute": ts.minute, "ts": k[0],
                }
                if _is_valid_candle(candle):
                    self.buf.push(candle)
                    # IT: mirror nel candle_buffer SOLO se più recente dell'ultima — evita
                    #     duplicati col catch-up A1 sopra; resta fallback se il catch-up è fallito.
                    # EN: mirror into candle_buffer ONLY if newer than the last one — avoids dupes
                    #     with the A1 catch-up above; stays a fallback if the catch-up failed.
                    _cb_last = self.candle_buffer.latest
                    if (_cb_last is None or
                            self.candle_buffer._norm_ts(k[0]) >
                            self.candle_buffer._norm_ts(_cb_last["open_time"])):
                        self.candle_buffer.append(candle)
                    n_pushed += 1
                else:
                    n_skipped += 1

            if n_skipped:
                log.warning(
                    f"Warm-up: {n_skipped} candele corrotte scartate "
                    f"su {n_pushed + n_skipped} totali"
                )

            log.info(
                f"Warm-up completato: {n_pushed} candele storiche caricate "
                f"(REST {self.symbol} {self.interval})  |  "
                f"buffer totale = {len(self.buf.candles)} candele"
            )

        except Exception as e:
            log.error(f"Warm-up REST fallito: {e}")
            if not restored:
                # IT: no stato + no REST -> impossibile inizializzare buffer
                # EN: no state + no REST -> cannot initialize buffer
                raise

        # IT: hard check — buffer minimo per emettere il primo segnale
        # EN: hard check — minimum buffer to emit the first signal
        n_buf = len(self.buf.candles)
        if n_buf < window_size + 20:
            log.warning(
                f"Buffer post-warm-up insufficiente: {n_buf} candele < "
                f"{window_size + 20} minimo. "
                f"Le prime candele live saranno ignorate finché il buffer non si riempie."
            )
        else:
            log.info(
                f"Buffer pronto: {n_buf} candele  |  "
                f"prima finestra valida disponibile  |  "
                f"candle_idx={self.candle_idx}  |  "
                f"equity=${self.rm.portfolio.equity:,.2f}"
            )

    # IT: inferenza modello — restituisce (mu, sigma, nu) in spazio raw
    # EN: model inference — returns (mu, sigma, nu) in raw space
    def _predict(self, window: np.ndarray) -> tuple[float, float, float]:
        """
        Predice (μ, σ, ν) dalla finestra corrente.
        Usa il vero snapshot macro (aggiornato ogni ora) invece di zeros.
        """
        if self.use_model and self.model is not None:
            # IT: Stage 4.7 — strict assertion sostituisce il vecchio _pad_or_truncate.
            #     FeatureAssembler garantisce 104 feature canoniche o solleva.
            # EN: Stage 4.7 — strict assertion replaces the old _pad_or_truncate shim.
            #     FeatureAssembler guarantees 104 canonical features or raises.
            assert window.shape[-1] == 104, (
                f"feature mismatch: window has {window.shape[-1]} features, expected 104"
            )

            xb = torch.tensor(window[None], dtype=torch.float32).to(self.device)

            # IT: snapshot macro reale se aggiornato, altrimenti zeros (no crash)
            # EN: real macro snapshot if fresh, otherwise zeros (no crash)
            xm = None
            has_macro = (self.pipeline_state is not None and
                         len(self.pipeline_state.macro_feature_cols) > 0)
            if has_macro:
                if self.macro_updater is not None:
                    xm = self.macro_updater.get_tensor(self.device)
                    if not self.macro_updater.is_fresh:
                        log.debug("Snapshot macro non aggiornato di recente — potrebbe essere datato")
                else:
                    # IT: fallback zeros — branch macro neutro
                    # EN: fallback zeros — neutral macro branch
                    n_macro = len(self.pipeline_state.macro_feature_cols)
                    xm = torch.zeros(1, n_macro, dtype=torch.float32).to(self.device)

            # IT: MC Dropout n=10 — uncertainty epistemica, SOLO per modelli singoli che la
            #     espongono. NB: l'EnsembleModel di produzione NON ha predict_with_uncertainty
            #     → il path live cade sempre sul ramo DETERMINISTICO sotto, bit-identico al
            #     backtest offline (questa è la base della parity Stage 5).
            # EN: MC Dropout n=10 — epistemic uncertainty, ONLY for single models exposing it.
            #     The production EnsembleModel lacks predict_with_uncertainty → the live path
            #     always takes the DETERMINISTIC branch below, bit-identical to the offline
            #     backtest (this is what the Stage-5 parity relies on).
            if hasattr(self.model, "predict_with_uncertainty"):
                result = self.model.predict_with_uncertainty(xb, xm, n_samples=10)
                # IT: atleast_1d protegge da scalar 0-dim quando batch=1
                # EN: atleast_1d guards against 0-dim scalars when batch=1
                mu     = float(np.atleast_1d(result["mu"])[0])
                sigma  = float(np.atleast_1d(result["sigma"])[0])
                nu     = float(np.atleast_1d(result["nu"])[0])
                conf   = float(np.atleast_1d(result["confidence_score"])[0])
                # IT: boost sigma se confidence bassa — penalizza segnali incerti
                # EN: boost sigma when confidence is low — penalises uncertain signals
                if conf < 0.3:
                    sigma *= (1.0 + (0.3 - conf) * 2)
                # IT: denormalizza z-score -> spazio raw (centralizzato in PipelineState)
                # EN: denormalize z-score -> raw space (centralized in PipelineState)
                if self.pipeline_state is not None:
                    mu, sigma = self.pipeline_state.denormalize_predictions(mu, sigma)
                return mu, sigma, nu

            # IT: Path DETERMINISTICO (ensemble di produzione) — nucleo condiviso col parity
            #     test Stage 5 (vedi _deterministic_predict) → il test esercita il path reale.
            # EN: DETERMINISTIC path (production ensemble) — shared core with the Stage-5 parity
            #     test (see _deterministic_predict) → the test exercises the real path.
            return self._deterministic_predict(self.model, window, xm,
                                               self.pipeline_state, self.device)

        # IT: fallback senza modello — usa rolling stats sui returns
        # EN: no-model fallback — use rolling stats on returns
        rets  = window[:, 0]
        mu    = float(rets[-5:].mean() * 0.5 + rets[-20:].mean() * 0.5)
        sigma = float(max(rets[-20:].std(), 1e-5))
        return mu, sigma, 5.0

    # IT: Nucleo di inferenza DETERMINISTICO (no MC Dropout) + denormalizzazione z→raw.
    #     Condiviso da _predict (ramo ensemble) e dal parity test Stage 5: garantisce che il
    #     test eserciti ESATTAMENTE il path di produzione, senza re-implementarlo (zero drift).
    #     window: (T,104) np.ndarray → ritorna (μ,σ,ν) in spazio RAW.
    # EN: DETERMINISTIC inference core (no MC dropout) + z→raw denorm. Shared by _predict
    #     (ensemble branch) and the Stage-5 parity test so the test exercises the exact
    #     production path without re-implementing it. Returns raw (μ,σ,ν).
    @staticmethod
    def _deterministic_predict(model, window: np.ndarray, xm,
                               pipeline_state, device) -> tuple[float, float, float]:
        xb = torch.tensor(window[None], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(xb, xm) if xm is not None else model(xb)
        mu, sigma, nu = float(out[0].item()), float(out[1].item()), float(out[2].item())
        if pipeline_state is not None:
            mu, sigma = pipeline_state.denormalize_predictions(mu, sigma)
        return mu, sigma, nu

    # IT: Monte Carlo forecast con dinamica GJR-GARCH (chiamato ogni N candele)
    # EN: Monte Carlo forecast with GJR-GARCH dynamics (called every N candles)
    def _run_forecast(self, window: np.ndarray, price: float) -> dict | None:
        """
        Esegue monte_carlo_forecast con i parametri LSTM reali.
        Chiamato ogni `_forecast_every` candele — non ad ogni tick.
        Risultato salvato in self.last_forecast per il log e la dashboard.
        """
        if not self.use_model:
            return None
        try:
            from quantsys.model.forecast import monte_carlo_forecast, summarize_forecast, build_feature_idx_map

            n_model = self.pipeline_state.model_config.get(
                "n_features", window.shape[1]
            ) if self.pipeline_state else window.shape[1]
            win = window.copy()
            if win.shape[1] < n_model:
                win = np.concatenate([win, np.zeros((win.shape[0], n_model - win.shape[1]), dtype=np.float32)], axis=1)
            elif win.shape[1] > n_model:
                win = win[:, :n_model]

            # IT: feature_idx_map per aggiornare multi-feature dentro i percorsi MC
            # EN: feature_idx_map for multi-feature updates inside MC paths
            feat_names = list(self.pipeline_state.feature_cols) if self.pipeline_state else []
            idx_map    = build_feature_idx_map(feat_names) if feat_names else None

            mc = self.cfg["montecarlo"]
            result = monte_carlo_forecast(
                model              = self.model,
                x_price_seed       = win[np.newaxis],
                last_price         = price,
                n_steps            = mc["n_steps"],
                n_paths            = min(500, mc["n_paths"]),   # IT: cap a 500 paths in live per latenza | EN: cap at 500 paths live for latency
                device             = self.device,
                feature_idx_map    = idx_map,
                gjr_omega          = mc.get("gjr_omega", 1.2e-5),
                gjr_alpha          = mc.get("gjr_alpha", 0.05),
                gjr_gamma          = mc.get("gjr_gamma", 0.065),
                gjr_beta           = mc.get("gjr_beta",  0.875),
            )
            summary = summarize_forecast(result, price, self.cfg["montecarlo"]["n_steps"])
            log.info(summary)
            return result
        except Exception as e:
            log.debug(f"Forecast fallito (non critico): {e}")
            return None

    # IT: callback principale — eseguito ad ogni candela 1m chiusa
    # EN: main callback — runs on every closed 1m candle
    def on_closed_candle(self, k: dict):
        """Chiamato ogni volta che una candela 1m si chiude."""
        self.candle_idx += 1
        price = k["close"]
        # IT: ATR floor a 5 bps per evitare SL troppo stretto in mercati calmi
        # EN: ATR floor at 5bps to avoid too-tight SL in quiet markets
        atr   = max(self.buf.atr, price * 0.0005)

        # IT: aggiorna trailing stop SE in posizione (prima del check_exit)
        # EN: update trailing stop IF in position (before check_exit)
        if self.rm.position:
            self.rm.update_trailing(price, atr)

        # IT: Stage 4.6 — calcola finestra via FeatureAssembler (parity col training).
        #     funding_df letto dallo stato cross-thread (aggiornato ogni 8h dal daemon).
        # EN: Stage 4.6 — compute window via FeatureAssembler (training parity).
        #     funding_df read from the cross-thread state (refreshed every 8h by daemon).
        if self.feature_assembler is None:
            # IT: fallback rolling stats (no PipelineState/modello) -> path legacy.
            # EN: fallback rolling stats (no PipelineState/model) -> legacy path.
            window = self.buf.get_window()
            if window is None:
                log.debug(f"Buffer insufficiente ({len(self.buf.candles)} candele)")
                return
        else:
            try:
                with self._funding_lock:
                    _fd = self._funding_df[0]
                window = self.feature_assembler.compute_window(
                    window_size=self.cfg["model"]["window_size"],
                    funding_df=_fd,
                )
            except RuntimeError as e:
                # IT: warmup ancora incompleto (buffer/feature 30d) -> skip silenzioso.
                # EN: warmup still incomplete (buffer/30d features) -> silent skip.
                log.debug(f"FeatureAssembler non pronto: {e}")
                return

        # IT: inferenza modello + generazione segnale BUY/SELL/HOLD
        # EN: model inference + BUY/SELL/HOLD signal generation
        mu, sigma, nu = self._predict(window)
        side, dist    = self.sig_gen.generate(mu, sigma, nu)

        # IT: Monte Carlo forecast a cadenza ridotta (costoso)
        # EN: Monte Carlo forecast at lower cadence (expensive)
        self._forecast_tick += 1
        if self._forecast_tick >= self._forecast_every:
            self._forecast_tick  = 0
            self.last_forecast   = self._run_forecast(window, price)

        # IT: check uscita — SL/TP/MAX_HOLD/reverse signal
        # EN: exit check — SL/TP/MAX_HOLD/reverse signal
        if self.rm.position:
            reason = self.rm.check_exit(k["high"], k["low"], price, self.candle_idx, side)
            if reason:
                ep = self.rm.position.stop_loss if reason == CloseReason.STOP_LOSS else \
                     self.rm.position.take_profit if reason == CloseReason.TAKE_PROFIT else price
                trade = self.rm.close_position(reason, ep, self.candle_idx)
                if trade:
                    pnl_color = GRN if trade.net_pnl > 0 else RED
                    print(f"  {pnl_color}[CLOSE {trade.side.value} | {reason.value}]  "
                          f"exit={ep:,.1f}  P&L={trade.net_pnl:+.2f}$  "
                          f"({trade.pnl_pct:+.2%}){RST}")

        # IT: apre nuova posizione solo se flat (no piramidazione)
        # EN: open a new position only when flat (no pyramiding)
        if side != Side.NONE and not self.rm.position:
            self.rm.open_position(side, price, self.candle_idx, atr, dist)

        # IT: equity mark-to-market = cash + uPnL + size_usd posizione aperta
        # EN: mark-to-market equity = cash + uPnL + open position size_usd
        mtm = self.rm.portfolio.cash
        if self.rm.position:
            mtm += self.rm.position.unrealized_pnl(price) + self.rm.position.size_usd
        pnl_tot = mtm - self.rm.icap

        # IT: log riga colorata sulla console
        # EN: colored console log line
        pos_str = ""
        if self.rm.position:
            upnl = self.rm.position.unrealized_pnl(price)
            pos_str = (f"  {CYN}[{self.rm.position.side.value} "
                       f"SL={self.rm.position.stop_loss:,.0f} "
                       f"TP={self.rm.position.take_profit:,.0f} "
                       f"uPnL={upnl:+.1f}$]{RST}")

        sig_col  = colored_signal(side.value)
        pnl_col  = GRN if pnl_tot >= 0 else RED
        ts_str   = datetime.now().strftime("%H:%M:%S")
        print(
            f"  {DIM}{ts_str}{RST}  "
            f"{BOLD}${price:>10,.1f}{RST}  "
            f"{sig_col}  "
            f"μ={mu:+.5f}  σ={sigma:.5f}  ν={nu:.1f}  "
            f"P↑={dist.prob_up:.0%}  "
            f"{pnl_col}eq=${mtm:,.0f} ({pnl_tot:+.1f}$){RST}"
            f"{pos_str}"
        )

        # IT: persiste segnale su JSONL (consumato da 05_analyze + dashboard)
        # EN: persist signal to JSONL (consumed by 05_analyze + dashboard)
        record = {
            "ts":        datetime.now(timezone.utc).isoformat(),
            "price":     price,
            "signal":    side.value,
            "mu":        round(mu, 7),
            "sigma":     round(sigma, 7),
            "nu":        round(nu, 3),
            "prob_up":   round(dist.prob_up, 4),
            "equity":    round(mtm, 2),
            "n_trades":  self.rm.portfolio.n_trades,
            "in_position": self.rm.position is not None,
        }
        # IT: rotazione log a 50MB (try/except per file lock Windows)
        # EN: 50MB log rotation (try/except for Windows file locks)
        if self.log_path.exists() and self.log_path.stat().st_size > 50 * 1024 * 1024:
            ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive   = self.log_path.with_name(f"live_signals_{ts_str}.jsonl")
            try:
                self.log_path.rename(archive)
                log.info(f"Live signals log ruotato → {archive}")
            except (OSError, PermissionError) as _e:
                log.warning(f"Log rotation fallita ({_e.__class__.__name__}: {_e}); proseguo senza ruotare")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self.last_signal = record

        # IT: snapshot stato su disco -> ripristino rapido al riavvio
        # EN: state snapshot on disk -> fast recovery on restart
        try:
            self._save_state()
        except Exception as e:
            log.warning(f"_save_state fallito (non critico): {e}")

    # IT: handler WebSocket Binance con reconnect exponential backoff
    # EN: Binance WebSocket handler with exponential-backoff reconnect
    async def _ws_handler(self):
        import websockets

        url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
        log.info(f"WebSocket: {url}")

        # IT: una sessione WS — consuma kline finché la connessione regge
        # EN: a single WS session — consumes klines until the connection drops
        async def connect():
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                log.info("WebSocket connesso. In attesa di candele ...")
                async for raw in ws:
                    data = json.loads(raw)
                    k    = data.get("k", {})

                    # IT: parse messaggio Binance kline (sempre UTC)
                    # EN: parse Binance kline message (always UTC)
                    ts = datetime.fromtimestamp(k["t"]/1000, tz=timezone.utc)
                    candle = {
                        "open": float(k["o"]), "high": float(k["h"]),
                        "low": float(k["l"]),  "close": float(k["c"]),
                        "volume": float(k["v"]), "taker_buy_vol": float(k["V"]),
                        "hour": ts.hour, "minute": ts.minute, "ts": k["t"],
                    }

                    # IT: scarta candele corrotte (sanity check pre-buffer)
                    # EN: drop corrupted candles (sanity check before buffer)
                    if not _is_valid_candle(candle):
                        log.warning(
                            f"Candela corrotta scartata: "
                            f"O={candle['open']:.1f} H={candle['high']:.1f} "
                            f"L={candle['low']:.1f} C={candle['close']:.1f} "
                            f"V={candle['volume']:.0f}"
                        )
                        continue

                    if k.get("x", False):
                        # IT: candela CHIUSA -> push buffer + segnale
                        # EN: CLOSED candle -> push buffer + emit signal
                        self._pending_candle = None
                        self.buf.push(candle)
                        # IT: Stage 4.6 — mirror append nel nuovo LiveCandleBuffer (raw OHLCV).
                        # EN: Stage 4.6 — mirror append to the new LiveCandleBuffer (raw OHLCV).
                        self.candle_buffer.append(candle)
                        self.on_closed_candle(candle)
                    else:
                        # IT: candela in formazione -> tenuta separata dal buffer chiuse
                        # EN: forming candle -> kept aside from the closed buffer
                        self._pending_candle = candle

        # IT: reconnect loop con exponential backoff (5s -> 5min)
        # EN: reconnect loop with exponential backoff (5s -> 5min)
        _backoff = 5.0
        while True:
            try:
                await connect()
                _backoff = 5.0  # IT: reset backoff al successo | EN: reset backoff on success
            except Exception as e:
                log.warning(
                    f"WS disconnesso ({e.__class__.__name__}: {e}) "
                    f"— riconnessione in {_backoff:.0f}s ..."
                )
                # IT: scarta candela parziale — il nuovo feed potrebbe saltarla
                # EN: drop partial candle — new feed may skip it on resume
                if self._pending_candle is not None:
                    log.info(
                        f"Reconnect: scarto candela parziale "
                        f"ts={self._pending_candle.get('ts')}"
                    )
                    self._pending_candle = None
                await asyncio.sleep(_backoff)
                _backoff = min(_backoff * 2, 300.0)  # IT: cap 5 minuti | EN: cap at 5 minutes

    # IT: status loop — riepilogo ogni 10 minuti su console
    # EN: status loop — console summary every 10 minutes
    async def _status_loop(self):
        """Stampa un riepilogo ogni 10 minuti."""
        while True:
            await asyncio.sleep(600)   # IT: 600s = 10 minuti | EN: 600s = 10 minutes
            m = self.rm.metrics() if self.rm.trades else {}
            elapsed = (time.time() - self.session_start) / 60
            macro_status = self.macro_updater.status if self.macro_updater else "non attivo"
            print(f"\n{'═'*60}")
            print(f"  STATUS  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  (sessione: {elapsed:.0f} min)")
            print(f"  Trade   : {self.rm.portfolio.n_trades}  |  Win rate: {m.get('win_rate',0):.1%}")
            print(f"  Equity  : ${self.rm.portfolio.equity:,.2f}  |  DD: {self.rm.portfolio.drawdown:.1%}")
            print(f"  Macro   : {macro_status}")
            print(f"  Segnali → {self.log_path}")
            print(f"{'═'*60}\n")

    # IT: orchestrazione async — warm-up + WS handler + status loop
    # EN: async orchestration — warm-up + WS handler + status loop
    async def run(self):
        self.warmup()

        print(f"""
{'═'*60}
  QUANTSYS · LIVE SIGNALS ENGINE
  Symbol  : {self.symbol} · {self.interval}
  Device  : {self.device}
  Modello : {'LSTM (addestrato)' if self.use_model else 'Rolling stats (fallback)'}
  Capital : ${self.rm.icap:,.0f}  (paper trading)
  Macro   : {'✓ snapshot live (yfinance+FRED)' if self.macro_updater else '✗ zeros (no normalizer)'}
  Log     : {self.log_path}
  Stop    : Ctrl+C
{'═'*60}
  {'ts':8s}  {'price':>12s}  {'signal':6s}  {'μ':>9s}  {'σ':>9s}  {'ν':>5s}  {'P↑':>5s}  equity
{'─'*60}
""")
        await asyncio.gather(
            self._ws_handler(),
            self._status_loop(),
        )

    # IT: shutdown ordinato — chiude posizione aperta + stampa riepilogo
    # EN: graceful shutdown — closes open position + prints summary
    def shutdown(self):
        """Chiamato a Ctrl+C — chiude la posizione aperta e stampa riepilogo finale."""
        print(f"\n\n{'═'*60}  SHUTDOWN  {'═'*60}")

        # IT: prima ferma il thread macro per evitare race
        # EN: stop macro thread first to avoid races
        if self.macro_updater is not None:
            self.macro_updater.stop()
            log.info(f"MacroSnapshotUpdater fermato. {self.macro_updater.status}")
        if self.rm.position:
            log.info("Chiusura posizione aperta ...")
            try:
                # IT: prezzo spot reale via REST, fallback a entry_price se REST giu'
                # EN: real spot price via REST, fallback to entry_price if REST down
                r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol}", timeout=3)
                last = float(r.json()["price"])
            except Exception:
                last = self.rm.position.entry_price
            self.rm.close_position(CloseReason.END_OF_DATA, last, self.candle_idx)

        m = self.rm.metrics() if self.rm.trades else {}
        print(f"""
  SESSIONE TERMINATA
  Trade totali    : {self.rm.portfolio.n_trades}
  Win rate        : {m.get('win_rate', 0):.1%}
  Profit factor   : {m.get('profit_factor', 0):.2f}
  Equity finale   : ${self.rm.portfolio.equity:,.2f}
  P&L sessione    : ${self.rm.portfolio.equity - self.rm.icap:+,.2f}
  Max drawdown    : {self.rm.portfolio.max_drawdown:.1%}
  Segnali salvati : {self.log_path}
""")
        # IT: snapshot di sessione (metriche + capitale) per audit storico
        # EN: session snapshot (metrics + capital) for historical audit
        summary_path = self.log_path.parent / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({**m, "initial_capital": self.rm.icap,
                       "final_equity": self.rm.portfolio.equity}, f, indent=2)
        log.info(f"Riepilogo → {summary_path}")


# IT: entry point — config + SN policy + event loop dedicato
# EN: entry point — config + SN policy + dedicated event loop
def main():
    # IT: Forza UTF-8 su stdout/stderr — evita UnicodeEncodeError (cp1252 di Windows) sui banner
    #     con box-drawing/emoji quando l'output è rediretto su file/pipe. Bug trovato dallo smoke
    #     test 2026-06-05 (crash in run() riga ~1628). Stesso pattern di 99_replay_live_vs_training.
    # EN: Force UTF-8 on stdout/stderr — avoids Windows cp1252 UnicodeEncodeError on Unicode
    #     banners when output is redirected (found by the 2026-06-05 smoke test).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cfg    = load_config("config/default.yaml")
    # IT: SN-policy deve matchare quella di training (coerenza load_model)
    # EN: SN policy must match the training one (load_model consistency)
    from quantsys.model import set_sn_on_mu_only
    set_sn_on_mu_only(bool(cfg.get("training", {}).get("sn_on_mu_only", False)))
    device = setup_device(cfg)
    engine = LiveEngine(cfg, device)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(engine.run())
    except KeyboardInterrupt:
        engine.shutdown()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
