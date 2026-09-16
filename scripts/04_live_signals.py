"""
Script 04 — Live Signals Engine.
Connects the trained LSTM model to the Binance WebSocket feed,
generates BUY/SELL/HOLD signals in real time and logs everything to file.

PyCharm run configuration:
  Script: scripts/04_live_signals.py
  Working dir: <project root>
  Environment: CUDA_VISIBLE_DEVICES=0

NOTE: this script does NOT place real orders. It generates signals and
      logs them to results/live_signals.jsonl for later analysis.
      Real trading requires a Binance API key with trading
      permissions — not included in this project for security.

Stop with: Ctrl+C
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

# cap BLAS/OMP threads before importing numpy/torch
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

from quantsys.utils import (load_config, setup_logging, setup_device, ensure_dirs,
                            interval_minutes_from_cfg)
from quantsys.model.ensemble import EnsembleModel
from quantsys.trading import SignalGenerator, RiskManager, Side, CloseReason

setup_logging(logging.INFO)
log = logging.getLogger("quantsys.live")


# ANSI color codes for readable console output
GRN  = "\033[92m"; RED  = "\033[91m"; YEL = "\033[93m"
CYN  = "\033[96m"; DIM  = "\033[2m";  RST = "\033[0m"; BOLD = "\033[1m"

# format BUY/SELL/HOLD with colored badge
def colored_signal(sig: str) -> str:
    if sig == "BUY":  return f"{GRN}{BOLD}▲ BUY {RST}"
    if sig == "SELL": return f"{RED}{BOLD}▼ SELL{RST}"
    return f"{YEL}◆ HOLD{RST}"


# WS candle sanity check — drops corrupt data (high<low, prices<=0, spike/drop)
def _is_valid_candle(c: dict) -> bool:
    """
    Sanity check on a candle from the Binance WebSocket.
    Drops candles with plainly corrupted data:
      · high < low (physically impossible)
      · zero or negative prices (feed error or halt)
      · spike > 10x or drop > 90% relative to close (feed error)
      · negative volume

    Note: candles with volume = 0 are not dropped (they occur in illiquid markets).
    """
    try:
        o = c["open"]; h = c["high"]; lo = c["low"]; cl = c["close"]
        v = c["volume"]

        if any(x <= 0 for x in [o, h, lo, cl]):   return False
        if h < lo:                                  return False   # high < low
        if v < 0:                                   return False   # negative volume
        if h > cl * 10:                             return False   # spike >10x feed error
        if lo < cl * 0.1:                           return False   # drop >90%
        if not all(map(lambda x: x == x, [o, h, lo, cl, v])):  # NaN check (x!=x)
            return False
        return True
    except (KeyError, TypeError):
        return False


# lightweight feature builder for live inference (subset of 01_*)
# Ring buffer of raw OHLCV candles for the new live engine (BLOCKER #1 Stage 4).
# Replaces LiveFeatureBuffer as the raw buffer, delegating all feature engineering
# to quantsys.features.FeatureBuilder (single source of truth shared with training).
class LiveCandleBuffer:
    """Raw OHLCV buffer for the live engine — feature engineering delegated to FeatureBuilder.

    Unlike LiveFeatureBuffer (legacy, computed 39 features by hand), this
    component keeps only the raw candles. FeatureAssembler (step 4.5) consumes
    them by calling FeatureBuilder.build() — guarantees exact parity with training.

    Capacity: the 50000 default equals ≈35 days ONLY at 1m interval; the caller
    (LiveEngine) computes an interval-aware capacity from config: 35 days × bars/day
    + margin (≈51900 at 1m, 2340 at 1h) — enough for full warmup of all 30d
    features (dist_ath_30d, momentum_30d, price_vs_ma200m).
    Memory: ~5 MB with dict-of-floats at 1m, negligible.
    """

    # Required fields for FeatureBuilder compatibility (training schema).
    REQUIRED_FIELDS = (
        "open", "high", "low", "close", "volume",
        "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol",
        "open_time",
    )

    # Initializes the deque with fixed capacity (auto-FIFO on overflow).
    def __init__(self, maxlen: int = 50000):
        self._candles: deque = deque(maxlen=maxlen)

    # open_time → tz-naive UTC Timestamp. Uniforms sources: parquet (Timestamp, often tz-AWARE)
    # and WS/REST (`ts`=epoch-ms int). Otherwise the buffer mixes tz-aware/naive → ValueError
    # "Cannot mix tz-aware with tz-naive" in build (smoke-test bug 2026-06-05).
    @staticmethod
    def _norm_ts(ot) -> "pd.Timestamp":
        if isinstance(ot, (int, float)):
            return pd.Timestamp(ot, unit="ms")                  # epoch-ms → tz-naive UTC
        t = pd.Timestamp(ot)
        return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t

    # Pre-loads the last n_last candles from raw_candles.parquet (warmup boot).
    def bootstrap_from_parquet(self, path: str, n_last: int | None = None) -> int:
        """Loads the last n_last candles from disk. Returns the number loaded.

        If path does not exist: warning + returns 0 (buffer starts empty and must be rebuilt
        via REST/WS, but many 30d features will be NaN until the buffer fills up).
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

    # Appends a candle (normalizes schema, defaults to 0 for missing fields).
    def append(self, candle: dict) -> None:
        """Appends a candle. Missing fields → default 0 (quote_vol/trades/taker_buy_quote_vol).

        The Binance kline WS (interval from config) returns all required fields; the caller only has to
        extract them from the payload's k[] (see the WS handler in LiveEngine).
        """
        # Normalize open_time to a tz-naive UTC Timestamp. WS/REST pass `ts`=epoch-ms (int); the
        # parquet bootstrap passes Timestamps. Without coercion the buffer mixes int and Timestamp
        # → object index → `.dt` accessor crashes in FeatureBuilder (smoke-test bug 2026-06-05).
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

    # Number of candles currently buffered
    def __len__(self) -> int:
        return len(self._candles)

    # Returns the last n_last candles as a DataFrame (index=open_time tz-naive).
    def to_dataframe(self, n_last: int | None = None) -> pd.DataFrame:
        """Returns the candles as a DataFrame compatible with FeatureBuilder.build().

        Output schema: index=open_time (tz-naive datetime),
        columns = open/high/low/close/volume/quote_vol/trades/taker_buy_vol/taker_buy_quote_vol.
        """
        if not self._candles:
            return pd.DataFrame()
        items = list(self._candles)[-n_last:] if n_last else list(self._candles)
        df = pd.DataFrame(items)
        # FeatureBuilder expects open_time as column or index; we set it as index.
        if "open_time" in df.columns:
            df = df.set_index("open_time")
            # guarantee a DatetimeIndex even if the index arrives as object (append already
            # coerces to Timestamp; this protects any other source).
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            # tz-naive UTC to avoid merge_asof mismatch in FeatureBuilder.
            if getattr(df.index, "tz", None) is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df

    # Last candle in buffer (None if empty)
    @property
    def latest(self) -> dict | None:
        return self._candles[-1] if self._candles else None


# Assembles the live feature vector (120, 104) using FeatureBuilder as single source of truth.
# Guarantees training parity: same code, same parameters, same scaler.
class FeatureAssembler:
    """Produces the model-ready feature tensor (BLOCKER #1 Stage 4).

    Internal pipeline:
      1. df = LiveCandleBuffer.to_dataframe() — the whole buffer (~50k candles)
      2. FeatureBuilder.build(df, fit=False, normalize=True, funding_df=funding)
         uses the scaler loaded from PipelineState → exact parity with training
      3. Checks canonical_names ⊆ feat_df.columns (HARD-FAIL on missing ones)
      4. Reorders + keeps only the 104 canonical columns
      5. Drops warmup NaNs
      6. Extracts the last `window_size` rows → np.ndarray (window_size, 104)

    Replaces LiveFeatureBuffer's _compute_features (legacy 39-feature).
    """

    # Configures FeatureBuilder with config params + scaler from PipelineState.
    def __init__(self, buffer: "LiveCandleBuffer", pipeline_state,
                 config: dict | None = None):
        """Args:
            buffer: already populated LiveCandleBuffer (bootstrap + WS appends)
            pipeline_state: loaded PipelineState (must have a fitted scaler)
            config: full dict (from load_config). If None → loads default.yaml.
        """
        from quantsys.features import FeatureBuilder, get_canonical_feature_names
        from quantsys.utils import load_config

        self.buffer = buffer
        self.ps = pipeline_state

        if config is None:
            config = load_config()
        fcfg = config.get("features", {})
        mcfg = config.get("model", {})

        # FeatureBuilder with same params used at training (from config).
        # interval_minutes from the PIPELINE STATE (train↔inference contract), NOT
        # from the current config: TIME-semantic windows must replicate training.
        # Config↔state mismatch is blocked upstream (guard in LiveEngine/main).
        self.fb = FeatureBuilder(
            vp_bins          = fcfg.get("vp_bins", 30),
            vp_lookback      = fcfg.get("vp_lookback", 240),
            windows          = fcfg.get("windows", [5, 10, 20, 60]),
            lag_periods      = fcfg.get("lag_periods", 5),
            forecast_horizon = fcfg.get("forecast_horizon", 1),
            vp_stride        = fcfg.get("vp_stride", 1),
            frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
            use_revin        = bool(mcfg.get("use_revin", False)),
            interval_minutes = getattr(pipeline_state, "interval_minutes", 1),
            # A4 HAR-CJ — same config as training (live↔training parity).
            use_har_cj       = bool(fcfg.get("har_cj", False)),
        )
        # Inject pre-fitted scaler state → build(fit=False) reuses it without re-fitting.
        self.fb.scaler             = pipeline_state.scaler
        self.fb._scale_cols        = list(pipeline_state.scale_cols)
        self.fb.scalers            = dict(pipeline_state.price_scaler_state)
        self.fb.clip_lo_           = pipeline_state.clip_lo_
        self.fb.clip_hi_           = pipeline_state.clip_hi_
        self.fb.feature_cols       = list(pipeline_state.feature_cols)
        self.fb.n_dynamic_features = pipeline_state.n_dynamic_features

        # Canonical list of the 104 features expected by the model (single source of truth).
        self.canonical_names: tuple[str, ...] = get_canonical_feature_names()
        log.info(f"FeatureAssembler: pronto per {len(self.canonical_names)} feature canoniche")

    # Builds the (window_size, 104) window by calling FeatureBuilder on the current buffer.
    def compute_window(self, window_size: int = 120,
                       funding_df: pd.DataFrame | None = None) -> np.ndarray:
        """Returns the model-ready (window_size, 104) tensor.

        Args:
            window_size: number of trailing candles to return (default 120, matches training)
            funding_df: funding-rate DataFrame from FundingRatePoller (optional; without it the 3
                        funding_rate* features are NaN → missing from the canonical set → HARD-FAIL)

        Raises:
            RuntimeError: if the buffer is insufficient, canonical features are missing, or
                          fewer than window_size valid rows remain after dropping NaNs.
        """
        if len(self.buffer) < window_size + 60:
            raise RuntimeError(
                f"FeatureAssembler: buffer insufficiente: {len(self.buffer)} < {window_size + 60} "
                f"(serve warmup completo prima di compute_window)"
            )

        df = self.buffer.to_dataframe()
        # FeatureBuilder.build wants open_time as a column (not just index).
        if df.index.name == "open_time":
            df = df.reset_index()

        # Normalize funding_df to tz-naive for buffer coherence (FeatureBuilder
        # reindexes on open_time and fails on dtype mismatch tz-aware vs tz-naive).
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

        # Hard-fail check that all 104 canonical features are present.
        missing = set(self.canonical_names) - set(feat_df.columns)
        if missing:
            sample = sorted(missing)[:10]
            raise RuntimeError(
                f"FeatureAssembler: {len(missing)} feature canoniche mancanti dall'output "
                f"FeatureBuilder.build. Sample: {sample}. "
                f"Verifica funding_df (3 feature) e warmup buffer (>43200 candele per 30d)."
            )

        # Reorder in canonical order (no positional pad/truncate).
        feat_df = feat_df[list(self.canonical_names)]

        # Drop rows with NaN (initial warmup).
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
    Circular buffer that keeps the last `window` candles and builds
    the features needed for real-time LSTM inference.

    DEPRECATED 2026-06-02 (Stage 4 BLOCKER #1): produces only 39 features, misaligned
    vs the 104 expected by training. Replaced by LiveCandleBuffer + FeatureAssembler.
    Kept temporarily as a fallback during the migration.

    FIX — Incremental Volume Profile:
      The previous version recomputed the VP from scratch on every candle
      by iterating over the whole buffer (O(N) per tick).
      Now we use an O(1) incremental update:
        · When a new candle arrives → add its contribution to the bin
        · When a candle leaves the buffer → subtract its contribution
      The VP is always up to date without rescanning the whole history.
    """

    VP_BINS = 30  # Volume Profile bins

    def __init__(self, window: int = 60, interval_minutes: int = 1):
        self.window      = window
        # bars/day derived from the interval — makes the "1 day" windows (ATH/ATL)
        # interval-agnostic; default 1 = legacy 1m (1440 bars/day).
        self.bars_per_day = max(1, 1440 // max(1, int(interval_minutes)))
        # n_features detected dynamically on first compute (no hardcoding)
        self.n_features  = 0

        # lookback >= max rolling used (ma200m -> 200) + window + margin
        self._lookback   = max(window + 60, 260)
        self.candles: deque = deque(maxlen=self._lookback)

        # incremental Volume Profile state (O(1) per push instead of O(N))
        self._vp_bins:   np.ndarray   = np.zeros(self.VP_BINS)
        # deque (bin_idx, volume) to subtract contributions on eviction
        self._vp_contribs: deque      = deque(maxlen=self._lookback)
        self._vp_price_min: float     = 0.0
        self._vp_price_max: float     = 0.0
        # periodic full-reset to avoid accumulated float drift
        self._vp_reset_every: int     = 60
        self._vp_since_reset: int     = 0

        self._feat_names: list[str] = []

    # read feature_names from training dataset for live/training alignment
    def load_scalers(self, data_dir: str = "data"):
        npz = np.load(f"{data_dir}/lstm_dataset.npz", allow_pickle=True)
        self._feat_names = list(npz["feature_names"])
        self.n_features  = len(self._feat_names)
        log.info(f"Features attese: {self.n_features}")

    # VP full-reset (initial boot or periodic to avoid drift)
    def _vp_full_reset(self):
        """Recomputes the VP from scratch. Called only at startup or every ~60 candles."""
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

    # expand VP range by remapping bins (O(BINS), not O(N))
    def _vp_expand_range(self, direction: str) -> bool:
        """
        Expands the VP range by 10% in the given direction
        by remapping the existing bins without rescanning the buffer (O(BINS)).
        Returns True if the expansion succeeded.
        """
        price_range = self._vp_price_max - self._vp_price_min
        if price_range <= 0:
            return False

        old_step = price_range / self.VP_BINS
        expand   = price_range * 0.10   # +10% range per expansion

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

    # incremental O(1) VP update + preemptive range expansion (T11)
    def _vp_update(self, candle: dict):
        """
        Updates the VP with the new candle without rescanning the whole buffer.

        T11 — Preemptive range expansion:
          Instead of an O(N) full reset every time the price leaves the range,
          it preemptively expands the range by 10% when the price is within 2%
          of the edge. Bin remapping is O(BINS)=O(30) instead of O(N_candles).
          In strongly trending markets it cuts resets from every candle to a few per hour.
        """
        tp  = (candle["high"] + candle["low"] + candle["close"]) / 3
        vol = candle["volume"]

        if self._vp_price_max <= self._vp_price_min:
            self._vp_full_reset()
            return

        price_range = self._vp_price_max - self._vp_price_min
        margin      = price_range * 0.02   # 2% edge -> preemptive expand

        # price near range floor -> expand downward
        if tp < self._vp_price_min + margin:
            self._vp_expand_range("down")

        # price near range ceiling -> expand upward
        elif tp > self._vp_price_max - margin:
            self._vp_expand_range("up")

        # still out of range after expand (large gap) -> full reset
        if tp < self._vp_price_min or tp > self._vp_price_max:
            self._vp_full_reset()
            return

        step    = (self._vp_price_max - self._vp_price_min) / self.VP_BINS
        new_idx = min(int((tp - self._vp_price_min) / step), self.VP_BINS - 1)
        new_idx = max(new_idx, 0)

        # CRITICAL ORDER — subtract old before adding new
        if len(self._vp_contribs) == self._vp_contribs.maxlen:
            old_idx, old_vol = self._vp_contribs[0]
            self._vp_bins[old_idx] = max(0.0, self._vp_bins[old_idx] - old_vol)

        self._vp_bins[new_idx] += vol
        self._vp_contribs.append((new_idx, vol))

        self._vp_since_reset += 1
        if self._vp_since_reset >= self._vp_reset_every:
            self._vp_full_reset()

    # public API — appends candle and keeps VP consistent
    def push(self, candle: dict):
        """Adds a new candle to the buffer and updates the VP incrementally."""
        self.candles.append(candle)
        self._vp_update(candle)

    # extract 4 scalars from VP (POC, VAH, VAL distance + concentration)
    def _vp_features(self, current_price: float) -> tuple[float, float, float, float]:
        """
        Extracts the 4 scalar features from the current Volume Profile:
          poc_dist      = % distance from the Point of Control
          vah_dist      = % distance from the Value Area High  (70% of volume)
          val_dist      = % distance from the Value Area Low
          concentration = % of volume in the POC bin (liquidity measure)
        """
        total = self._vp_bins.sum()
        if total < 1e-9 or self._vp_price_max <= self._vp_price_min:
            return 0.0, 0.0, 0.0, 0.0

        poc_idx   = int(self._vp_bins.argmax())
        step      = (self._vp_price_max - self._vp_price_min) / self.VP_BINS
        poc_price = self._vp_price_min + (poc_idx + 0.5) * step

        # Value Area = bins covering 70% of volume around POC
        sorted_idx = np.argsort(self._vp_bins)[::-1]
        cum, va_bins = 0.0, []
        for idx in sorted_idx:
            if cum / total >= 0.70:   # 70% standard VP threshold
                break
            va_bins.append(idx); cum += self._vp_bins[idx]
        va_lo = self._vp_price_min + min(va_bins) * step
        va_hi = self._vp_price_min + (max(va_bins) + 1) * step

        safe_price = max(current_price, 1e-9)
        return (
            (current_price - poc_price) / safe_price,   # % distance from POC
            (current_price - va_hi)     / safe_price,   # % distance from VAH
            (current_price - va_lo)     / safe_price,   # % distance from VAL
            float(self._vp_bins[poc_idx] / total),       # volume concentration at POC
        )

    # builds the (window, n_features) tensor for inference
    def _compute_features(self) -> np.ndarray | None:
        """
        Builds a (window, n_features) window from the buffered candles.
        All rolling statistics are computed with pandas (no convolve),
        the VP uses the incremental state already kept in _vp_bins.
        """
        if len(self.candles) < self.window + 20:
            return None

        c_arr  = list(self.candles)
        closes = np.array([c["close"]   for c in c_arr])
        highs  = np.array([c["high"]    for c in c_arr])
        lows   = np.array([c["low"]     for c in c_arr])
        vols   = np.array([c["volume"]  for c in c_arr])
        # taker_buy_vol computed once, reused for taker ratio + CVD
        taker_buy = np.array([
            c.get("taker_buy_vol", c["volume"] * 0.5) for c in c_arr
        ])
        taker     = np.where(vols > 0, taker_buy / vols, 0.5)

        s_closes = pd.Series(closes)
        s_vols   = pd.Series(vols)

        # log-returns with prepend to keep array length aligned
        log_ret  = np.log(np.maximum(closes, 1e-9))
        log_ret  = np.diff(log_ret, prepend=log_ret[0])
        s_rets   = pd.Series(log_ret)

        # deviation from cumulative VWAP (mean-reversion proxy)
        tp       = (highs + lows + closes) / 3
        vwap     = np.cumsum(tp * vols) / np.maximum(np.cumsum(vols), 1e-9)
        vwap_dev = (closes - vwap) / np.maximum(vwap, 1e-9)

        # 20-bar volume z-score
        vol_mu  = s_vols.rolling(20, min_periods=1).mean()
        vol_std = s_vols.rolling(20, min_periods=1).std().fillna(1)
        vol_z   = ((s_vols - vol_mu) / vol_std.replace(0, 1)).values

        # short/long vol ratio (regime breakout)
        vol_std5  = s_rets.rolling(5,  min_periods=1).std().fillna(0).values
        vol_std20 = s_rets.rolling(20, min_periods=1).std().fillna(1).values
        vol_ratio = vol_std5 / np.maximum(vol_std20, 1e-9)

        # time-of-day via cyclic encoding (sin/cos)
        hours = np.array([c.get("hour", 12) + c.get("minute", 0) / 60 for c in c_arr])
        h_sin = np.sin(2 * np.pi * hours / 24)
        h_cos = np.cos(2 * np.pi * hours / 24)

        # lag returns + 5/60-bar momentum
        lag1 = s_rets.shift(1).fillna(0).values
        lag2 = s_rets.shift(2).fillna(0).values
        lag3 = s_rets.shift(3).fillna(0).values
        lag4 = s_rets.shift(4).fillna(0).values
        lag5 = s_rets.shift(5).fillna(0).values
        mom5 = s_closes.pct_change(5).fillna(0).values
        mom60= s_closes.pct_change(min(60, len(c_arr)-1)).fillna(0).values

        # CVD = Cumulative Volume Delta (buy vs sell pressure)
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

        # candle microstructure (replaces classic RSI/MACD/... indicators)
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

        # session_pos in [-0.5, +0.5] = position vs 4h range (240 bars)
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

        # structural features (ATH/ATL on a daily window = bars_per_day bars,
        # interval-agnostic: 1440 at 1m, 24 at 1h)
        ath_buf     = pd.Series(highs).rolling(min(len(c_arr), self.bars_per_day), min_periods=1).max().values
        atl_buf     = pd.Series(lows).rolling(min(len(c_arr), self.bars_per_day),  min_periods=1).min().values
        pr_range    = np.maximum(ath_buf - atl_buf, 1e-9)
        dist_ath    = (closes - ath_buf) / np.maximum(ath_buf, 1e-9)
        dist_atl    = (closes - atl_buf) / np.maximum(atl_buf, 1e-9)
        price_pos   = (closes - atl_buf) / pr_range
        # % distance from nearest round number (multiples of $1000)
        round_level = (pd.Series(closes) / 1000).round() * 1000
        round_dist  = ((pd.Series(closes) - round_level) / pd.Series(closes).replace(0, np.nan)).fillna(0).values

        # Volume Profile features (broadcast scalars across the window)
        current_price = closes[-1]
        poc_d, vah_d, val_d, conc = self._vp_features(current_price)
        vp_poc = np.full(len(c_arr), poc_d)
        vp_vah = np.full(len(c_arr), vah_d)
        vp_val = np.full(len(c_arr), val_d)
        vp_conc= np.full(len(c_arr), conc)

        # assemble — Stream A (time-varying dynamics) | Stream B (structural)
        feat_mat = np.stack([
            # Stream A — dynamic features
            log_ret, vwap_dev, vol_z, vol_ratio, h_sin, h_cos, taker,
            lag1, lag2, lag3, lag4, lag5, mom5, mom60,
            cvd_norm, cvd_pct20, cvd_div, delta_accel,
            vol_std5, vol_std20,
            body_ratio, upper_shadow, lower_shadow, close_vs_open,
            price_vel, price_accel, vwap_slope, spread_proxy, vwap_skew,
            intraday_pos,
            # Stream B — structural features
            vp_poc, vp_vah, vp_val, vp_conc,
            dist_ath, dist_atl, price_pos, round_dist,
            session_pos,
        ], axis=1)

        win = feat_mat[-self.window:]
        if win.shape[0] < self.window:
            return None

        # update n_features on first compute or on change
        if self.n_features != win.shape[1]:
            self.n_features = win.shape[1]
            log.debug(f"LiveFeatureBuffer: {self.n_features} feature rilevate automaticamente")

        # vectorized robust normalization (per-column median + IQR)
        med = np.median(win, axis=0)
        q1_q3 = np.percentile(win, [25, 75], axis=0)
        iqr = q1_q3[1] - q1_q3[0]
        mask = iqr > 1e-9
        win[:, mask] = (win[:, mask] - med[mask]) / iqr[mask]

        # clip at ±5σ — robust to live-time outliers
        return np.clip(win, -5, 5).astype(np.float32)

    # public accessor — returns the inference-ready feature window
    def get_window(self) -> np.ndarray | None:
        return self._compute_features()

    # simplified ATR over the last 15 candles (TR ≈ high-low)
    @property
    def atr(self) -> float:
        if len(self.candles) < 2:
            return 0.0
        c   = list(self.candles)[-15:]
        hl  = [x["high"] - x["low"] for x in c]
        return float(np.mean(hl))


# live engine — Binance WS + inference + paper trading + state persistence
STATE_MAX_AGE_SEC = 300   # 5 min — stale state threshold


class LiveEngine:
    """
    Main orchestrator of the live engine:
      1. Keeps the candle buffer up to date via WebSocket
      2. On every closed candle → LSTM inference → signal
      3. Logs everything to JSONL + prints to screen
      4. Paper trading: tracks simulated P&L without real orders

    FIX — State persistence:
      The critical state (candle buffer, portfolio, open position, candle_idx)
      is serialized to disk on every closed candle.
      On restart (or after a WS crash), the state is restored
      automatically if recent enough (< STATE_MAX_AGE_SEC).
      This way open positions and paper P&L survive
      network disconnects, process restarts and system crashes.
    """

    # engine setup — load PipelineState/model, start funding+macro threads, init RM/sig_gen
    def __init__(self, cfg: dict, device: torch.device):
        self.cfg    = cfg
        self.device = device
        dcfg = cfg["data"]; mcfg = cfg["model"]; rcfg = cfg["risk"]; bcfg = cfg["backtest"]

        self.symbol   = dcfg["symbol"]
        self.interval = dcfg["interval"]

        # arch-specific directories (models/<arch>, results/<arch>)
        self._models_dir  = Path(cfg["training"]["output_dir"])
        self._state_file  = Path(bcfg["output_dir"]) / "live_engine_state.json"

        # PipelineState = scaler + columns + training config (live/training parity)
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
                    # cache in arch dir to speed up next startups
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
            # hard-fail if forecast_horizon config != training (invalid signals)
            _cfg_h = cfg.get("features", {}).get("forecast_horizon",
                       dcfg.get("forecast_horizon", 15))
            _state_h = self.pipeline_state.forecast_horizon
            if _cfg_h != _state_h:
                raise RuntimeError(
                    f"forecast_horizon mismatch: config={_cfg_h}, training={_state_h}. "
                    f"Il modello è stato addestrato per orizzonte {_state_h}; live signals a {_cfg_h} "
                    f"produce segnali invalidi. Allinea config/default.yaml o rigenera il modello."
                )
            # hard-fail if candle interval config != training (1m→1h pivot):
            # the WS streams candles at cfg.interval but scalers/TIME-semantic
            # windows belong to training → out-of-distribution features, invalid signals.
            _cfg_im = interval_minutes_from_cfg(cfg)
            _state_im = getattr(self.pipeline_state, "interval_minutes", 1)
            if _cfg_im != _state_im:
                raise RuntimeError(
                    f"interval mismatch: config={_cfg_im}min, training={_state_im}min. "
                    f"Il modello è stato addestrato su candele {_state_im}m; il live a {_cfg_im}m "
                    f"è una combinazione invalida. Allinea config/default.yaml o ri-addestra."
                )
            # max_hold_candles must >= forecast_horizon (otherwise TP/SL is noise)
            _max_hold = rcfg.get("max_hold_candles", 0)
            if _max_hold < _state_h:
                log.warning(
                    f"max_hold_candles ({_max_hold}) < forecast_horizon ({_state_h}). "
                    f"Il TP/SL potrebbe non avere tempo di triggerare prima del MAX_HOLD."
                )

        # funding rate — initial load + 8h refresh via daemon thread
        self._funding_df = [None]   # mutable list for cross-thread write
        self._funding_lock = threading.Lock()  # guards cross-thread access
        _funding_path = Path("data/funding_rate.parquet")
        if _funding_path.exists():
            _initial_df = pd.read_parquet(_funding_path)
            with self._funding_lock:
                self._funding_df[0] = _initial_df
            log.info(f"Funding rate caricato: {len(_initial_df)} osservazioni")
        else:
            log.warning("data/funding_rate.parquet non trovato — funding rate feature disabilitata")

        # daemon thread — refresh IMMEDIATELY on first iter, then sleep 8h
        def _funding_rate_updater():
            _first = True
            while True:
                if not _first:
                    time.sleep(28800)  # 8h = Binance funding interval
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

        # hourly macro snapshot refresh (yfinance + FRED) for MacroEncoder
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

        # prefer heterogeneous ensemble (>=2 archs present), fallback to homogeneous
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

        # run interval — from PipelineState when available (train↔inference contract,
        # inference-side convention), config fallback; the two were already
        # validated identical above (hard-fail on mismatch).
        _interval_minutes = (
            getattr(self.pipeline_state, "interval_minutes", 1)
            if self.pipeline_state is not None
            else interval_minutes_from_cfg(cfg)
        )
        _bars_per_day = 1440 // _interval_minutes

        # BLOCKER #1 Stage 4.6 — DEPRECATED buffer kept only for ATR + state persistence + candle sanity
        # (interval_minutes wired: "1 day" ATH/ATL windows are interval-agnostic).
        self.buf = LiveFeatureBuffer(window=mcfg["window_size"], interval_minutes=_interval_minutes)

        # BLOCKER #1 Stage 4.6 — new raw buffer + assembler relying on FeatureBuilder
        # as single source of truth. Produces the 104 canonical features with the same
        # training scaler (parity guaranteed by tests/test_live_training_parity.py).
        # interval-aware capacity (1m→1h pivot): 35 days of bars + 1500 margin.
        # At 1m → 35×1440+1500 = 51900 (≈ legacy 50000 equivalent); at 1h → 2340.
        # CAPACITY only, not semantics: 1m identity is preserved.
        # (_interval_minutes/_bars_per_day computed above, PipelineState-first.)
        _cb_maxlen = 35 * _bars_per_day + 1500
        self.candle_buffer = LiveCandleBuffer(maxlen=_cb_maxlen)
        # bootstrap from raw_candles.parquet (~35d history) for 30d-lookback feature warmup.
        _raw_path = Path("data/raw_candles.parquet")
        if _raw_path.exists():
            self.candle_buffer.bootstrap_from_parquet(str(_raw_path), n_last=_cb_maxlen)
        else:
            log.warning("data/raw_candles.parquet non trovato — LiveCandleBuffer parte vuoto.")
        # funding_df read once at boot (Stage 4.4 workaround: no Poller).
        with self._funding_lock:
            self.funding_df = self._funding_df[0]
        # instantiate assembler only if PipelineState is available (requires scaler).
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
        # Monte Carlo forecast cadence (every N closed candles)
        self._forecast_every = cfg.get("montecarlo", {}).get("live_forecast_every", 10)
        self._forecast_tick  = 0
        self.session_start   = time.time()

        # partial candle (k.x=False) kept aside — dropped on WS reconnect
        self._pending_candle: dict | None = None

    # state persistence — survives crashes/brief disconnects

    # serialize buffer+portfolio+position to disk (atomic temp+rename write)
    def _save_state(self):
        """
        Serializes to disk:
          · last 200 buffer candles (enough for warm-up)
          · portfolio state (cash, equity, peak, drawdown)
          · open position (if any)
          · current candle_idx

        The file is written atomically (write temp + rename)
        to avoid leaving a corrupted JSON behind on a crash.
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
            "candles":    list(self.buf.candles)[-200:],  # last 200 for fast warm-up
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

        # write temp + rename = atomic write (no corrupt JSON on crash)
        tmp = self._state_file.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        tmp.replace(self._state_file)

    # restore state from disk if fresh (< STATE_MAX_AGE_SEC), else fresh warm-up
    def _load_state(self) -> bool:
        """
        Restores the state from disk if it exists and is recent.
        Returns True if the restore succeeded, False otherwise.
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

        # repopulate the candle buffer (also rebuilds incremental VP)
        for c in state.get("candles", []):
            self.buf.push(c)
        self.candle_idx = state.get("candle_idx", 0)

        # restore paper portfolio cash/equity/drawdown
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

        # restore open position (keeps original SL/TP/trailing)
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

    # warm-up — fills the buffer before enabling live inferences
    def warmup(self):
        """
        Fills the buffer before starting the live stream.

        Two-step strategy:
          1. Try to restore the state from disk (fast restart after a crash
             or reconnect within STATE_MAX_AGE_SEC=5 min). If the file is fresh,
             we recover buffer + portfolio + open position without touching the REST API.
          2. Download the missing candles from the REST API to bring the buffer
             to window_size + lookback (≥ 120 candles) before the WS starts.

        The number of candles to download is derived from the config's window_size
        (not hardcoded) with a +60 overhead for stable rolling features.
        With window_size=60 → at least 120 candles are needed.
        The Binance REST API returns at most 1000 candles per call.
        """
        # min candles = window + 60 + 10. Quantities are in BARS (bar-semantic),
        # NOT minutes: 60 = max rolling feature window (windows=[5,10,20,60] in bars),
        # 10 = discard margin. Interval-invariant (1m or 1h).
        window_size  = self.cfg["model"]["window_size"]
        min_candles  = window_size + 60 + 10

        restored = self._load_state()

        if restored:
            # fast restart — only fetch the gap since on-disk snapshot
            n_in_buf = len(self.buf.candles)
            needed   = max(10, min_candles - n_in_buf)
            log.info(
                f"Warm-up: stato ripristinato da disco "
                f"({n_in_buf} candele salvate). "
                f"Richiesta REST per colmare il gap ({needed} candele recenti) ..."
            )
        else:
            # cold start — full buffer via REST
            needed = min_candles
            log.info(
                f"Warm-up: avvio freddo — scaricamento {needed} candele storiche "
                f"(window_size={window_size} + lookback=60 + margine=10) ..."
            )

        # A1 — CONTIGUOUS catch-up of candle_buffer (the FeatureAssembler source). The parquet
        # bootstrap can be days old; without bridging the gap to "now", long-lookback features
        # (ma200m, vp, 30d) span a temporal hole (smoke-test bug 2026-06-05). Fetch the missing
        # candles via REST from the buffer's last one up to now (pagination in fetch_klines)
        # and append contiguously. Dedup on open_time. Best-effort: on failure the WS fills
        # the gap gradually (with the dedup-safe mirror below as fallback).
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
                    continue                                   # dedup — already in buffer
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
                    "limit":    min(needed, 1000),   # Binance REST cap
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
                    # mirror into candle_buffer ONLY if newer than the last one — avoids dupes
                    # with the A1 catch-up above; stays a fallback if the catch-up failed.
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
                # no state + no REST -> cannot initialize buffer
                raise

        # hard check — minimum buffer to emit the first signal
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

    # model inference — returns (mu, sigma, nu) in raw space
    def _predict(self, window: np.ndarray) -> tuple[float, float, float]:
        """
        Predicts (μ, σ, ν) from the current window.
        Uses the real macro snapshot (refreshed hourly) instead of zeros.
        """
        if self.use_model and self.model is not None:
            # Stage 4.7 — strict assertion replaces the old _pad_or_truncate shim.
            # FeatureAssembler guarantees 104 canonical features or raises.
            assert window.shape[-1] == 104, (
                f"feature mismatch: window has {window.shape[-1]} features, expected 104"
            )

            xb = torch.tensor(window[None], dtype=torch.float32).to(self.device)

            # real macro snapshot if fresh, otherwise zeros (no crash)
            xm = None
            has_macro = (self.pipeline_state is not None and
                         len(self.pipeline_state.macro_feature_cols) > 0)
            if has_macro:
                if self.macro_updater is not None:
                    xm = self.macro_updater.get_tensor(self.device)
                    if not self.macro_updater.is_fresh:
                        log.debug("Snapshot macro non aggiornato di recente — potrebbe essere datato")
                else:
                    # fallback zeros — neutral macro branch
                    n_macro = len(self.pipeline_state.macro_feature_cols)
                    xm = torch.zeros(1, n_macro, dtype=torch.float32).to(self.device)

            # MC Dropout n=10 — epistemic uncertainty, ONLY for single models exposing it.
            # The production EnsembleModel lacks predict_with_uncertainty → the live path
            # always takes the DETERMINISTIC branch below, bit-identical to the offline
            # backtest (this is what the Stage-5 parity relies on).
            if hasattr(self.model, "predict_with_uncertainty"):
                result = self.model.predict_with_uncertainty(xb, xm, n_samples=10)
                # atleast_1d guards against 0-dim scalars when batch=1
                mu     = float(np.atleast_1d(result["mu"])[0])
                sigma  = float(np.atleast_1d(result["sigma"])[0])
                nu     = float(np.atleast_1d(result["nu"])[0])
                conf   = float(np.atleast_1d(result["confidence_score"])[0])
                # boost sigma when confidence is low — penalises uncertain signals
                if conf < 0.3:
                    sigma *= (1.0 + (0.3 - conf) * 2)
                # denormalize z-score -> raw space (centralized in PipelineState)
                if self.pipeline_state is not None:
                    mu, sigma = self.pipeline_state.denormalize_predictions(mu, sigma)
                return mu, sigma, nu

            # DETERMINISTIC path (production ensemble) — shared core with the Stage-5 parity
            # test (see _deterministic_predict) → the test exercises the real path.
            return self._deterministic_predict(self.model, window, xm,
                                               self.pipeline_state, self.device)

        # no-model fallback — use rolling stats on returns
        rets  = window[:, 0]
        mu    = float(rets[-5:].mean() * 0.5 + rets[-20:].mean() * 0.5)
        sigma = float(max(rets[-20:].std(), 1e-5))
        return mu, sigma, 5.0

    # DETERMINISTIC inference core (no MC dropout) + z→raw denorm. Shared by _predict
    # (ensemble branch) and the Stage-5 parity test so the test exercises the exact
    # production path without re-implementing it (zero drift).
    # window: (T,104) np.ndarray → returns (μ,σ,ν) in RAW space.
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

    # Monte Carlo forecast with GJR-GARCH dynamics (called every N candles)
    def _run_forecast(self, window: np.ndarray, price: float) -> dict | None:
        """
        Runs monte_carlo_forecast with the real LSTM parameters.
        Called every `_forecast_every` candles — not on every tick.
        Result stored in self.last_forecast for the log and the dashboard.
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

            # feature_idx_map for multi-feature updates inside MC paths
            feat_names = list(self.pipeline_state.feature_cols) if self.pipeline_state else []
            idx_map    = build_feature_idx_map(feat_names) if feat_names else None

            mc = self.cfg["montecarlo"]
            result = monte_carlo_forecast(
                model              = self.model,
                x_price_seed       = win[np.newaxis],
                last_price         = price,
                n_steps            = mc["n_steps"],
                n_paths            = min(500, mc["n_paths"]),   # cap at 500 paths live for latency
                device             = self.device,
                feature_idx_map    = idx_map,
                gjr_omega          = mc.get("gjr_omega", 1.2e-5),
                gjr_alpha          = mc.get("gjr_alpha", 0.05),
                gjr_gamma          = mc.get("gjr_gamma", 0.065),
                gjr_beta           = mc.get("gjr_beta",  0.875),
                # per-bar σ cap from config (2026-07-15: parametric, 1h=0.13).
                gjr_sigma_cap      = mc.get("gjr_sigma_cap", 0.01),
            )
            summary = summarize_forecast(result, price, self.cfg["montecarlo"]["n_steps"])
            log.info(summary)
            return result
        except Exception as e:
            log.debug(f"Forecast fallito (non critico): {e}")
            return None

    # main callback — runs on every closed candle (interval from config)
    def on_closed_candle(self, k: dict):
        """Called every time a candle (interval from config) closes."""
        self.candle_idx += 1
        price = k["close"]
        # ATR floor at 5bps to avoid too-tight SL in quiet markets
        atr   = max(self.buf.atr, price * 0.0005)

        # update trailing stop IF in position (before check_exit)
        if self.rm.position:
            self.rm.update_trailing(price, atr)

        # Stage 4.6 — compute window via FeatureAssembler (training parity).
        # funding_df read from the cross-thread state (refreshed every 8h by daemon).
        if self.feature_assembler is None:
            # fallback rolling stats (no PipelineState/model) -> legacy path.
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
                # warmup still incomplete (buffer/30d features) -> silent skip.
                log.debug(f"FeatureAssembler non pronto: {e}")
                return

        # model inference + BUY/SELL/HOLD signal generation
        mu, sigma, nu = self._predict(window)
        side, dist    = self.sig_gen.generate(mu, sigma, nu)

        # Monte Carlo forecast at lower cadence (expensive)
        self._forecast_tick += 1
        if self._forecast_tick >= self._forecast_every:
            self._forecast_tick  = 0
            self.last_forecast   = self._run_forecast(window, price)

        # exit check — SL/TP/MAX_HOLD/reverse signal
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

        # open a new position only when flat (no pyramiding)
        if side != Side.NONE and not self.rm.position:
            self.rm.open_position(side, price, self.candle_idx, atr, dist)

        # mark-to-market equity = cash + uPnL + open position size_usd
        mtm = self.rm.portfolio.cash
        if self.rm.position:
            mtm += self.rm.position.unrealized_pnl(price) + self.rm.position.size_usd
        pnl_tot = mtm - self.rm.icap

        # colored console log line
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

        # persist signal to JSONL (consumed by 05_analyze + dashboard)
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
        # 50MB log rotation (try/except for Windows file locks)
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

        # state snapshot on disk -> fast recovery on restart
        try:
            self._save_state()
        except Exception as e:
            log.warning(f"_save_state fallito (non critico): {e}")

    # Binance WebSocket handler with exponential-backoff reconnect
    async def _ws_handler(self):
        import websockets

        url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.interval}"
        log.info(f"WebSocket: {url}")

        # a single WS session — consumes klines until the connection drops
        async def connect():
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                log.info("WebSocket connesso. In attesa di candele ...")
                async for raw in ws:
                    data = json.loads(raw)
                    k    = data.get("k", {})

                    # parse Binance kline message (always UTC)
                    ts = datetime.fromtimestamp(k["t"]/1000, tz=timezone.utc)
                    candle = {
                        "open": float(k["o"]), "high": float(k["h"]),
                        "low": float(k["l"]),  "close": float(k["c"]),
                        "volume": float(k["v"]), "taker_buy_vol": float(k["V"]),
                        "hour": ts.hour, "minute": ts.minute, "ts": k["t"],
                    }

                    # drop corrupted candles (sanity check before buffer)
                    if not _is_valid_candle(candle):
                        log.warning(
                            f"Candela corrotta scartata: "
                            f"O={candle['open']:.1f} H={candle['high']:.1f} "
                            f"L={candle['low']:.1f} C={candle['close']:.1f} "
                            f"V={candle['volume']:.0f}"
                        )
                        continue

                    if k.get("x", False):
                        # CLOSED candle -> push buffer + emit signal
                        self._pending_candle = None
                        self.buf.push(candle)
                        # Stage 4.6 — mirror append to the new LiveCandleBuffer (raw OHLCV).
                        self.candle_buffer.append(candle)
                        self.on_closed_candle(candle)
                    else:
                        # forming candle -> kept aside from the closed buffer
                        self._pending_candle = candle

        # reconnect loop with exponential backoff (5s -> 5min)
        _backoff = 5.0
        while True:
            try:
                await connect()
                _backoff = 5.0  # reset backoff on success
            except Exception as e:
                log.warning(
                    f"WS disconnesso ({e.__class__.__name__}: {e}) "
                    f"— riconnessione in {_backoff:.0f}s ..."
                )
                # drop partial candle — new feed may skip it on resume
                if self._pending_candle is not None:
                    log.info(
                        f"Reconnect: scarto candela parziale "
                        f"ts={self._pending_candle.get('ts')}"
                    )
                    self._pending_candle = None
                await asyncio.sleep(_backoff)
                _backoff = min(_backoff * 2, 300.0)  # cap at 5 minutes

    # status loop — console summary every 10 minutes
    async def _status_loop(self):
        """Prints a summary every 10 minutes."""
        while True:
            await asyncio.sleep(600)   # 600s = 10 minutes
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

    # async orchestration — warm-up + WS handler + status loop
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

    # graceful shutdown — closes open position + prints summary
    def shutdown(self):
        """Called on Ctrl+C — closes the open position and prints the final summary."""
        print(f"\n\n{'═'*60}  SHUTDOWN  {'═'*60}")

        # stop macro thread first to avoid races
        if self.macro_updater is not None:
            self.macro_updater.stop()
            log.info(f"MacroSnapshotUpdater fermato. {self.macro_updater.status}")
        if self.rm.position:
            log.info("Chiusura posizione aperta ...")
            try:
                # real spot price via REST, fallback to entry_price if REST down
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
        # session snapshot (metrics + capital) for historical audit
        summary_path = self.log_path.parent / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({**m, "initial_capital": self.rm.icap,
                       "final_equity": self.rm.portfolio.equity}, f, indent=2)
        log.info(f"Riepilogo → {summary_path}")


# entry point — config + SN policy + dedicated event loop
def main():
    # Force UTF-8 on stdout/stderr — avoids Windows cp1252 UnicodeEncodeError on Unicode
    # box-drawing/emoji banners when output is redirected to file/pipe (found by the
    # 2026-06-05 smoke test: crash in run() line ~1628). Same pattern as 99_replay_live_vs_training.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    cfg    = load_config("config/default.yaml")
    # SN policy must match the training one (load_model consistency)
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
