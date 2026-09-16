"""Phase 2 — Feature Engineering: OHLCV → normalized features for the LSTM."""
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

log = logging.getLogger("quantsys.features")


# Single source of truth for the model's canonical feature names (post C-funding).
# Read from the NPZ dataset produced by training. Used by the live engine to:
#   - align the order of columns produced by FeatureBuilder
#   - hard-fail if an expected feature is missing (no positional pad/truncate)
# Caching avoids re-reading the NPZ file on every call.
@lru_cache(maxsize=4)
def get_canonical_feature_names(npz_path: str = "data/lstm_dataset.npz") -> tuple[str, ...]:
    """Returns the model's canonical feature names (104 names, fixed order).

    Single source of truth = `data/lstm_dataset.npz['feature_names']`.
    NB: `PipelineState.feature_cols` holds 121 entries (pre-filter, includes LIVE_DROP_FEATURES
    and the targets target_ret/target_dir) — it is NOT the model's canonical list.

    Args:
        npz_path: path to the NPZ dataset produced by `scripts/01_download_data.py`.

    Returns:
        tuple of strings (immutable, cache-friendly), fixed order.

    Raises:
        FileNotFoundError: if the NPZ does not exist.
        KeyError: if the NPZ does not contain 'feature_names'.
    """
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ canonico non trovato: {npz_path}. Esegui prima 01_download_data.py")
    with np.load(p, allow_pickle=True) as data:
        if "feature_names" not in data.files:
            raise KeyError(f"'feature_names' assente in {npz_path}. NPZ corrotto o vecchia versione.")
        names = tuple(str(n) for n in data["feature_names"])
    return names


# Features dropped from the "C-funding" set (single source of truth training↔live).
# Reason: 2026-05-28 permutation importance → ROI ≤ 0 (noise or harmful) AND/OR lookback > 30d
# not computable in the live buffer. The 30d + funding features (positive ROI) are kept instead.
# See THEORY.md (C-funding filter) and THEORY.md §3 ("104 feature" rule).
LIVE_DROP_FEATURES = frozenset({
    "dist_ath_90d", "dist_atl_90d", "price_pos_90d",
    "dist_ath_365d", "dist_atl_365d", "price_pos_365d",
    "momentum_90d", "momentum_7d",
    "frac_diff_close", "frac_diff_volume",
    "vp_poc_dist_long", "vp_vah_dist_long", "vp_val_dist_long", "vp_concentration_long",
    "vp_poc_convergence",
})


# non-feature columns excluded from the canonical derivation (builder
# intermediates + targets). C2 2ter refactor (2026-07-18): was duplicated in 5 scripts.
CANONICAL_EXCLUDE = frozenset({
    "open_time", "close_time", "date_utc", "pv", "cum_pv", "cum_vol",
    "typical_price", "obv", "target_ret", "target_dir",
})


def canonical_feature_columns(feature_cols, feat: pd.DataFrame,
                              nan_thresh: float = 0.5,
                              diag: Optional[dict] = None) -> list:
    """
    CANONICAL derivation of the model feature list (C2 2ter refactor,
    2026-07-18) — single implementation of the filter sequence previously
    duplicated across 01_download/01_update/04b/vol_paper_replay/paper_01:
    ① non-feature exclude ② float dtype ③ C-funding (LIVE_DROP_FEATURES)
    ④ NaN > nan_thresh ⑤ columns containing Inf. `feature_cols` order is
    PRESERVED (= builder order). `diag`, when given, is filled with caller
    logging detail (keys: dropped_live, nan_ratios, dropped_nan,
    dropped_inf) — the function itself never logs.
    """
    cols = [c for c in feature_cols
            if c not in CANONICAL_EXCLUDE and c in feat.columns
            and feat[c].dtype in ["float64", "float32"]]
    dropped_live = [c for c in cols if c in LIVE_DROP_FEATURES]
    if dropped_live:
        cols = [c for c in cols if c not in LIVE_DROP_FEATURES]
    before = cols[:]
    nan_ratios = {c: feat[c].isna().mean() for c in before}
    cols = [c for c in cols if nan_ratios[c] <= nan_thresh]
    dropped_inf = [c for c in cols if np.isinf(feat[c].values).any()]
    if dropped_inf:
        cols = [c for c in cols if c not in dropped_inf]
    if diag is not None:
        diag["dropped_live"] = dropped_live
        diag["nan_ratios"] = nan_ratios
        diag["dropped_nan"] = [(c, nan_ratios[c]) for c in before
                               if nan_ratios[c] > nan_thresh]
        diag["dropped_inf"] = dropped_inf
    return cols


# Feature-engineering pipeline OHLCV → normalized matrix for the model.
class FeatureBuilder:
    """Full pipeline: raw OHLCV → numpy arrays ready for the LSTM."""

    # Stores feature hyperparameters (VP, lag, horizon, FFD, RevIN, scaler).
    # Timeframe contract: TIME-semantic windows (calendar days/hours/minutes:
    # structural, MA200m, 4h session_position, 24h funding) are converted to
    # bars via _tbars/bars_per_day; BAR-semantic windows (windows, CVD, vwap
    # rolling, lags, VP scales) stay in BARS and translate with the timeframe.
    # At interval_minutes=1 everything is identical to legacy behavior.
    def __init__(self, vp_bins: int = 30, vp_lookback: int = 240,
                 windows: list[int] = None, lag_periods: int = 5,
                 forecast_horizon: int = 1, vp_stride: int = 1,
                 frac_diff_d: float = 0.0, use_revin: bool = False,
                 interval_minutes: int = 1, target_type: str = "ret",
                 use_har_cj: bool = False):
        self.vp_bins             = vp_bins
        self.vp_lookback         = vp_lookback
        self.windows             = windows or [5, 10, 20, 60]
        self.lag_periods         = lag_periods
        self.forecast_horizon    = forecast_horizon   # bars ahead (=minutes at 1m, hours at 1h)
        # Target type — "ret" (sum of log-returns, directional, legacy) or
        # "log_rv" (log realized variance Σr² over h bars — vol-S experiment 2026-06-10).
        # Default "ret": the directional path stays bit-invariant.
        if target_type not in ("ret", "log_rv", "log_rs_ratio"):
            raise ValueError(f"target_type '{target_type}' non riconosciuto / unknown (ret|log_rv|log_rs_ratio)")
        self.target_type         = target_type
        self.vp_stride           = vp_stride          # VP subsample stride (O(n)→O(n/stride))
        self.frac_diff_d         = frac_diff_d         # FFD order (0=skip)
        # Bar duration in minutes (1=legacy 1m, 60=1h) + bars per calendar day.
        # Basis for TIME-semantic → bars conversions.
        self.interval_minutes    = max(1, int(interval_minutes))
        self.bars_per_day        = max(1, 1440 // self.interval_minutes)
        # RevIN fix — exclude raw returns from the global RobustScaler so RevIN runs in
        # raw scale and denormalize_mu yields predictions aligned to the target.
        self.use_revin           = use_revin
        # A4 (vol roadmap) — HAR-CJ features (bipower/jump decomposition as INPUT).
        # Default False = INERT lever: the step is skipped and the 104 features stay
        # bit-invariant. Activation only at a planned retrain with a pre-registered gate.
        self.use_har_cj          = use_har_cj
        self.scalers:            dict[str, RobustScaler] = {}
        self.scaler:             Optional[RobustScaler]  = None   # multi-column RobustScaler
        self._scale_cols:        list[str]               = []     # columns scaled by the multi-scaler
        self.feature_cols:       list[str] = []
        self.n_dynamic_features: int       = 0
        # Clip bounds fitted on training (P0.1/P99.9 per feature) — adaptive vs fixed ±20.
        self.clip_lo_: Optional[np.ndarray] = None
        self.clip_hi_: Optional[np.ndarray] = None

    def _tbars(self, minutes: int, min_bars: int = 2) -> int:
        # Converts a window expressed in MINUTES to a bar count, with an
        # anti-degeneracy floor. Identity at 1m (minutes//1 = minutes).
        return max(min_bars, minutes // self.interval_minutes)

    # ── Log-returns ──────────────────────────────────────────────────────────
    # Computes OHLCV log-returns and multi-step target (sum of next h candles).
    def _returns(self, df, forecast_horizon: int = 1):
        """
        Computes log-returns and the target.

        Improvement — multi-step target:
          The original target was log_ret.shift(-1): the return of the next
          single minute. With 0.1% fees per trade, to be profitable the
          signal must predict moves of at least 0.2%. On 1-minute candles,
          moves that large are rare and noisy.

          With forecast_horizon=15: the target is the sum of the log-returns
          of the next 15 candles = cumulative 15-minute return.
          This lowers trading frequency (fewer fees), strengthens the
          signal and gives weight to the macro features. Moves of
          0.3-1.0% over 15 min are common and well above fees.

          The model still sees the 1-minute window — the horizon changes
          only the target, not the features.

        NB — the horizon is expressed in BARS of the current timeframe
        (h=30 → 30 minutes at 1m, 30 hours at 1h), not absolute minutes.
        """
        df["log_ret"]      = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_high"] = np.log(df["high"]  / df["high"].shift(1))
        df["log_ret_low"]  = np.log(df["low"]   / df["low"].shift(1))
        df["log_ret_vol"]  = np.log(
            df["volume"].replace(0, np.nan) /
            df["volume"].shift(1).replace(0, np.nan)
        )

        h = max(1, forecast_horizon)
        if getattr(self, "target_type", "ret") == "log_rv":
            # Vol-S experiment — target = log realized variance of the next h bars:
            # log(Σᵢ₌₁..ₕ r²ₜ₊ᵢ + ε). The log makes the distribution ~Gaussian (RV tails
            # are extreme) → RobustScaler/NLL/denorm work unchanged downstream.
            # target_dir = vol-up/down: future RV > trailing h-bar RV (causal at t).
            _eps = 1e-12
            sq = df["log_ret"] ** 2
            rv_fwd  = sq.rolling(h).sum().shift(-h)
            rv_trail = sq.rolling(h).sum()
            df["target_ret"] = np.log(rv_fwd + _eps)
            df["target_dir"] = (rv_fwd > rv_trail).astype(int)
        elif getattr(self, "target_type", "ret") == "log_rs_ratio":
            # Semivariance probe (pre-reg 2026-06-11) — target = signed asymmetry of future
            # realized semivariance (Barndorff-Nielsen et al. 2010, Patton-Sheppard 2015):
            # log((RS⁺+ε)/(RS⁻+ε)) with RS± = Σᵢ₌₁..ₕ r²ₜ₊ᵢ·1[rₜ₊ᵢ≷0].
            # A vol moment (NOT direction): the "sign of variance" via signed jump variation.
            # target_dir = 1[RS⁺_fwd > RS⁻_fwd] (upside asymmetry, causal at t).
            _eps = 1e-12
            sq_pos = (df["log_ret"].clip(lower=0.0)) ** 2
            sq_neg = (df["log_ret"].clip(upper=0.0)) ** 2
            rs_pos_fwd = sq_pos.rolling(h).sum().shift(-h)
            rs_neg_fwd = sq_neg.rolling(h).sum().shift(-h)
            df["target_ret"] = np.log(rs_pos_fwd + _eps) - np.log(rs_neg_fwd + _eps)
            df["target_dir"] = (rs_pos_fwd > rs_neg_fwd).astype(int)
        else:
            # Legacy target = sum of next h log-returns (rolling+shift, no temp Series loop).
            df["target_ret"] = df["log_ret"].rolling(h).sum().shift(-h)
            df["target_dir"] = (df["target_ret"] > 0).astype(int)
        return df

    # ── VWAP ─────────────────────────────────────────────────────────────────
    # Intraday VWAP + rolling 20/60 and price-vs-VWAP deviations.
    def _vwap(self, df):
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["pv"]            = df["typical_price"] * df["volume"]
        df["date_utc"]      = df["open_time"].dt.date
        # single groupby (one factorization) for both cumsums (A8).
        _g = df.groupby("date_utc")
        df["cum_pv"]        = _g["pv"].cumsum()
        df["cum_vol"]       = _g["volume"].cumsum()
        df["vwap"]          = df["cum_pv"] / df["cum_vol"].replace(0, np.nan)
        df["vwap_dev"]      = (df["close"] - df["vwap"]) / df["vwap"]
        for w in [20, 60]:
            rpv = df["pv"].rolling(w).sum()
            rv  = df["volume"].rolling(w).sum()
            df[f"vwap_{w}"]     = rpv / rv.replace(0, np.nan)
            df[f"vwap_{w}_dev"] = (df["close"] - df[f"vwap_{w}"]) / df[f"vwap_{w}"]
        return df

    # ── Volume Profile ────────────────────────────────────────────────────────
    # Volume Profile for one lookback: POC/VAH/VAL/concentration (strided).
    def _vp_single(self, tp_arr, vl_arr, lo_arr, hi_arr, cl_arr,
                   lookback: int, suffix: str, df_len: int,
                   vp_stride: int = 1) -> dict:
        """
        Computes the VP for a single lookback. Returns arrays for the 4 features.

        Optimization with vp_stride > 1:
          Instead of computing the VP for every single candle (O(n × lookback)),
          it computes it every `vp_stride` candles and linearly interpolates the
          intermediate values. With vp_stride=5 the cost drops from O(n) to O(n/5):
            · Scale 60:   2.1M × 60 / 5   = 25M  operations  (from 126M)
            · Scale 1440: 2.1M × 1440 / 5 = 605M operations  (from 3B)
          The VP at the long scale (1440 BARS) barely changes from one bar
          to the next → the approximation is negligible relative to market
          noise. NB: lookbacks are in BARS (bar-semantic) and translate
          with the timeframe — see _volume_profile.

        Structure:
          1. Compute the VP only at the sampled indices (i = lookback, lookback+stride, ...)
          2. Fill the results into a full-size array at the sampled indices
          3. Linearly interpolate the gaps (forward/backward fill at the edges)
        """
        poc_dist_sampled = {}
        vah_dist_sampled = {}
        val_dist_sampled = {}
        vol_conc_sampled = {}

        # Sampled indices every vp_stride starting from `lookback`.
        sampled_indices = list(range(lookback, df_len, max(1, vp_stride)))

        # B2 — rolling min/max precomputed ONCE per scale (were recomputed per window: ~605M ops
        # at scale 1440 = FeatureBuilder CPU bottleneck #1). roll[i-1] spans EXACTLY
        # lo_arr[i-lookback:i]; min/max select one element (no float accumulation, no reorder)
        # → bit-identical to the per-window .min()/.max(). (No-NaN prices ⇒ rolling.min skipna
        # semantics == numpy.min.)
        roll_lo = pd.Series(lo_arr).rolling(lookback, min_periods=lookback).min().to_numpy()
        roll_hi = pd.Series(hi_arr).rolling(lookback, min_periods=lookback).max().to_numpy()

        for i in sampled_indices:
            sl  = slice(i - lookback, i)
            tp  = tp_arr[sl]; vol = vl_arr[sl]
            lo_ = roll_lo[i - 1]; hi_ = roll_hi[i - 1]
            if hi_ <= lo_:
                continue

            step    = (hi_ - lo_) / self.vp_bins
            idx_arr = np.clip(((tp - lo_) / step).astype(int), 0, self.vp_bins - 1)
            # Volume-per-bin histogram via np.bincount (vectorized segmented sum).
            # Replaces np.zeros+np.add.at, whose unbuffered path (~1 C loop/element) is
            # 10-40× slower in the VP inner loop. idx_arr is already clipped to [0, vp_bins-1]
            # → non-negative ints; minlength pins the length to vp_bins even if the last bin is
            # empty. bincount accumulates in float64 (like np.zeros), so numerically identical
            # (reduction reorder ≤1 ULP).
            bin_vol = np.bincount(idx_arr, weights=vol, minlength=self.vp_bins)

            poc_idx   = int(bin_vol.argmax())
            poc_price = lo_ + (poc_idx + 0.5) * step
            total     = bin_vol.sum()

            sorted_idx  = np.argsort(bin_vol)[::-1]
            cum_sorted  = np.cumsum(bin_vol[sorted_idx])
            n_va        = int(np.searchsorted(cum_sorted, 0.70 * total)) + 1
            va          = sorted_idx[:n_va]
            va_lo = lo_ + int(va.min()) * step
            va_hi = lo_ + (int(va.max()) + 1) * step

            curr = cl_arr[i]; safe = max(curr, 1e-9)
            poc_dist_sampled[i] = (curr - poc_price) / safe
            vah_dist_sampled[i] = (curr - va_hi)     / safe
            val_dist_sampled[i] = (curr - va_lo)     / safe
            vol_conc_sampled[i] = bin_vol[poc_idx]   / (total + 1e-9)

        # Full-size reconstruction via forward-fill (no look-ahead).
        def _fill_interp(sampled_dict: dict, n: int) -> np.ndarray:
            arr = np.full(n, np.nan)
            if not sampled_dict:
                return arr
            idxs = np.array(sorted(sampled_dict.keys()), dtype=np.int64)
            vals = np.array([sampled_dict[k] for k in idxs], dtype=np.float64)
            arr[idxs] = vals
            # numpy ffill — avoids ×12 temporary pd.Series (4 feats × 3 scales).
            mask = np.isnan(arr)
            idx  = np.where(~mask, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            return arr[idx]

        poc_dist = _fill_interp(poc_dist_sampled, df_len)
        vah_dist = _fill_interp(vah_dist_sampled, df_len)
        val_dist = _fill_interp(val_dist_sampled, df_len)
        vol_conc = _fill_interp(vol_conc_sampled, df_len)

        return {
            f"vp_poc_dist{suffix}": poc_dist,
            f"vp_vah_dist{suffix}": vah_dist,
            f"vp_val_dist{suffix}": val_dist,
            f"vp_concentration{suffix}": vol_conc,
        }

    # Multi-scale Volume Profile (1h/4h/1d) + POC convergence feature.
    def _volume_profile(self, df):
        """
        Multi-scale Volume Profile: short + medium + long term.

        CONCEPTUAL FIX — a fixed VP lookback does not adapt to the regime:
        ─────────────────────────────────────────────────────────────
        With a fixed lookback of 240 (4 hours):
          · In high volatility: 4 hours are not enough for the liquidity nodes
          · In low volatility: 4 hours already cover a "mature" market
          · The POC varies radically with the chosen period

        Solution — three scales in BARS (bar-semantic, deliberate choice):
          · Short  (60 bars):   at 1m = 1h,  at 1h = 60h  — recent liquidity
          · Medium (240 bars):  at 1m = 4h,  at 1h = 10d  — session structure
          · Long   (1440 bars): at 1m = 1d,  at 1h = 60d  — long technical levels
        The scales translate with the timeframe: profiles stay relative
        to the trading horizon (h bars), not to fixed calendar durations.
        NB: vp_*_long is in LIVE_DROP_FEATURES anyway (excluded from the model).

        The LSTM sees all three scales → it learns which is most relevant
        in each regime. In high vol the short term dominates; in low vol
        the long term is more stable as support/resistance.

        Computational cost triples but stays acceptable (~15-30s total).
        """
        tp_arr = df["typical_price"].values
        vl_arr = df["volume"].values
        lo_arr = df["low"].values
        hi_arr = df["high"].values
        cl_arr = df["close"].values
        n      = len(df)

        # Performance warning for large datasets.
        if n > 500_000:
            log.info(
                f"Volume Profile: dataset grande ({n:,} candele), "
                f"vp_stride={self.vp_stride} "
                f"(se il calcolo è lento, aumenta vp_stride in config/default.yaml)"
            )

        # Three VP scales in BARS (bar-semantic, deliberate): at 1m = 1h/4h/1d,
        # at 1h = 60h/10d/60d — profiles relative to the trading horizon;
        # vp_*_long is LIVE_DROP anyway. Do NOT convert via _tbars.
        scales = [
            (60,   "_short"),        # 60 bars (1h @1m, 60h @1h)
            (self.vp_lookback, ""),  # default 240 bars (legacy name)
            (1440, "_long"),         # 1440 bars (1d @1m, 60d @1h)
        ]

        all_vp = {}
        for lookback, suffix in scales:
            # skip if there is not enough data for the scale.
            if lookback > n - 10:
                log.warning(f"VP scale {lookback}: troppo pochi dati ({n}), skip.")
                continue
            effective_windows = max(1, (n - lookback) // self.vp_stride)
            log.debug(f"  VP lookback={lookback}{suffix} | stride={self.vp_stride} | ~{effective_windows:,} finestre")
            arrays = self._vp_single(tp_arr, vl_arr, lo_arr, hi_arr, cl_arr,
                                     lookback, suffix, n, vp_stride=self.vp_stride)
            all_vp.update(arrays)

        if all_vp:
            df = pd.concat([df, pd.DataFrame(all_vp, index=df.index)], axis=1)

        # Composite — POC convergence short vs long (strong level).
        if "vp_poc_dist_short" in df.columns and "vp_poc_dist_long" in df.columns:
            df["vp_poc_convergence"] = 1.0 - np.abs(
                df["vp_poc_dist_short"].fillna(0) - df["vp_poc_dist_long"].fillna(0)
            ).clip(0, 1)

        return df

    # ── Technical indicators ──────────────────────────────────────────────────
    # Zero-lag microstructure: candle anatomy, velocity, spread, skew.
    def _technicals(self, df):
        """
        Microstructure features — RSI, MACD, Bollinger Width and ATR removed.

        Removed because lagging and redundant:
          · RSI      → already captured by vol_std + lag_ret
          · MACD     → already captured by momentum + vol_ratio
          · BB Width → identical to vol_std_20 / vol_std_60
          · ATR      → already in vol_std; used separately by the RiskManager

        Replaced with instantaneous or near-instantaneous microstructure features:

          body_ratio      Directional strength of the candle (0=doji, 1=marubozu)
          upper_shadow    Rejection of the high price (seller pressure)
          lower_shadow    Rejection of the low price (buyer pressure)
          close_vs_open   Direction and strength of the single candle
          intraday_pos    Where the price closes within the H-L range

          price_velocity  Price velocity (close diff over 3 steps, normalized)
          price_accel     Price acceleration (derivative of velocity)

          vwap_slope      VWAP trend over the last 5 min (intraday bias)
          spread_proxy    (high-low)/volume — proxy for bid-ask spread / liquidity
          high_of_day_dist Distance from the high of the last 4 hours (session)
          vwap_ret_skew   Volume-weighted return asymmetry (pressure)
        """
        hl = (df["high"] - df["low"]).replace(0, np.nan)
        # fillna(1) once + vectorized open-close max/min (ufunc), not a sub-frame reduce (A12).
        hl_f   = hl.fillna(1)
        oc_max = np.maximum(df["open"], df["close"])
        oc_min = np.minimum(df["open"], df["close"])

        # ── Candle anatomy (zero-lag microstructure)
        df["body_ratio"]    = (df["close"] - df["open"]).abs() / hl_f
        df["upper_shadow"]  = (df["high"] - oc_max) / hl_f
        df["lower_shadow"]  = (oc_min - df["low"])  / hl_f
        df["close_vs_open"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        df["intraday_pos"]  = (df["close"] - df["low"])  / hl_f

        # ── Velocity & acceleration ────
        velocity             = df["close"].diff(3) / 3 / df["close"].shift(3).replace(0, np.nan)
        df["price_velocity"] = velocity.fillna(0)
        df["price_accel"]    = velocity.diff(1).fillna(0)

        # ── VWAP slope (intraday directional bias)
        if "vwap" in df.columns:
            vwap_diff        = df["vwap"].diff(5)
            df["vwap_slope"] = (vwap_diff / df["vwap"].shift(5).replace(0, np.nan)).fillna(0)
        else:
            df["vwap_slope"] = 0.0

        # Spread proxy — instant liquidity proxy (high = illiquid).
        df["spread_proxy"] = (hl / df["volume"].replace(0, np.nan)).fillna(0)

        # Session position in [-0.5,+0.5] within the 4h range (mid_4h centered).
        # TIME-semantic window: 240 minutes → bars via _tbars (240 at 1m, 4 at 1h).
        _w_4h              = self._tbars(240)
        _mp_4h             = self._tbars(10)
        high_4h            = df["high"].rolling(_w_4h, min_periods=_mp_4h).max()
        low_4h             = df["low"].rolling(_w_4h,  min_periods=_mp_4h).min()
        range_4h           = (high_4h - low_4h).replace(0, np.nan)
        mid_4h             = (high_4h + low_4h) / 2
        df["session_position"] = (df["close"] - mid_4h) / range_4h

        # Vol-weighted return skew (20) — >0 = bullish pressure.
        if "log_ret" in df.columns:
            vol_s        = df["volume"]
            ret_s        = df["log_ret"]
            roll_vol     = vol_s.rolling(20, min_periods=10).sum().replace(0, np.nan)
            wret_mean    = (ret_s * vol_s).rolling(20, min_periods=10).sum() / roll_vol
            dev          = ret_s - wret_mean
            roll_var     = (dev**2 * vol_s).rolling(20, min_periods=10).sum() / roll_vol
            df["vwap_ret_skew"] = (
                ((dev**3) * vol_s).rolling(20, min_periods=10).sum() /
                (roll_vol * roll_var.replace(0, np.nan)**1.5 + 1e-12)
            ).fillna(0)
        else:
            df["vwap_ret_skew"] = 0.0

        return df

    # ── Volume features ───────────────────────────────────────────────────────
    # Volume features: taker ratio, z-score, OBV ROC, money flow.
    def _volume_features(self, df):
        df["taker_buy_ratio"] = (df["taker_buy_vol"] / df["volume"].replace(0, np.nan)).clip(0, 1)
        for w in [20, 60]:
            mu  = df["volume"].rolling(w).mean()
            sig = df["volume"].rolling(w).std().replace(0, np.nan)
            df[f"vol_zscore_{w}"] = (df["volume"] - mu) / sig

        direction = np.sign(df["close"].diff())

        # OBV Rate-of-Change (stationary) — avoids decade-long cumulative drift.
        obv_raw             = (direction * df["volume"]).cumsum()
        df["obv_roc_20"]    = obv_raw.diff(20)
        df["obv_roc_60"]    = obv_raw.diff(60)
        # normalized by mean volume
        vol_ma_20           = df["volume"].rolling(20, min_periods=1).mean().replace(0, np.nan)
        vol_ma_60           = df["volume"].rolling(60, min_periods=1).mean().replace(0, np.nan)
        df["obv_roc_20_n"]  = df["obv_roc_20"] / (vol_ma_20 * 20)
        df["obv_roc_60_n"]  = df["obv_roc_60"] / (vol_ma_60 * 60)

        candle_sz = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
        mf = direction * df["volume"] * candle_sz
        df["money_flow_norm"] = mf.rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-9)
        return df

    # Cumulative Volume Delta: order-flow pressure, divergence, acceleration.
    def _cvd_features(self, df):
        """
        Improvement 4 — Cumulative Volume Delta (CVD).

        Volume delta is the difference between aggressive buy volume
        (taker buy) and aggressive sell volume (taker sell). It measures the
        directional pressure of participants who "cross the spread".

        CVD = Σ(taker_buy - taker_sell) accumulated over time.
        Rising CVD while price is flat = hidden upward pressure.
        Falling CVD while price is high = distribution.

        Derived features:
          · cvd_raw:        absolute delta value (in BTC units)
          · cvd_norm:       volume-normalized delta [-1, +1]
          · cvd_divergence: difference between CVD trend and price trend
          · delta_accel:    delta acceleration (second derivative)
        """
        taker_sell = df["volume"] - df["taker_buy_vol"]
        delta      = df["taker_buy_vol"] - taker_sell
        cvd        = delta.cumsum()

        df["cvd_raw"]   = delta                                    # instant delta
        df["cvd_norm"]  = delta / df["volume"].replace(0, np.nan)  # normalized [-1,1]

        # Rolling CVD 20/60 with min_periods=w (warmup vs steady-state parity).
        for w in [20, 60]:
            df[f"cvd_cum_{w}"]  = delta.rolling(w, min_periods=w).sum()
            vol_sum = df["volume"].rolling(w, min_periods=w).sum().replace(0, np.nan)
            df[f"cvd_pct_{w}"]  = df[f"cvd_cum_{w}"] / vol_sum     # % of volume

        # CVD-norm vs log_ret-norm divergence (rolling 20) — explicit min_periods
        # to avoid distribution shift on the first live-buffer samples.
        cvd_trend  = delta.rolling(20, min_periods=20).sum().fillna(0)
        price_trend= df["log_ret"].rolling(20, min_periods=20).sum().fillna(0)
        cvd_std    = cvd_trend.rolling(60, min_periods=60).std().replace(0, np.nan)
        price_std  = price_trend.rolling(60, min_periods=60).std().replace(0, np.nan)
        df["cvd_divergence"] = (cvd_trend / cvd_std) - (price_trend / price_std)

        # delta acceleration = order-flow momentum
        df["delta_accel"] = delta.diff(5) / df["volume"].rolling(5, min_periods=5).sum().replace(0, np.nan)

        return df

    # Aligns funding rate (8h) to the candle index and derives mean/deviation.
    def _funding_features(self, df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds funding rate features to the main DataFrame.

        The funding rate has 8h frequency; it is aligned to the candle index
        of the main df (any timeframe) with forward-fill. Leading NaNs
        (pre-2020 data or initial gaps) are filled with 0.

        Added features:
          · funding_rate:     instantaneous value (ffill from 8h)
          · funding_rate_1d:  24h moving average (bars_per_day bars) — baseline level
          · funding_rate_dev: deviation from the mean — contrarian signal
        """
        funding_df = funding_df.copy()
        if not isinstance(funding_df.index, pd.DatetimeIndex):
            funding_df = funding_df.set_index("open_time")

        funding_series = funding_df["funding_rate"]

        # align to df index (ffill 8h freq)
        df_index = df["open_time"]
        aligned = funding_series.reindex(df_index, method="ffill").values

        df["funding_rate"]     = aligned
        df["funding_rate"]     = df["funding_rate"].fillna(0)

        # TIME-semantic window: 24h = bars_per_day bars (1440 at 1m, 24 at 1h).
        df["funding_rate_1d"]  = df["funding_rate"].rolling(self.bars_per_day, min_periods=1).mean()
        df["funding_rate_dev"] = df["funding_rate"] - df["funding_rate_1d"]

        return df

    # Structural price-level features: ATH/ATL distance, momentum, MA200m.
    def _structural_features(self, df):
        """
        Improvement 3 — absolute price-level features.

        Without these features the LSTM cannot tell whether BTC is at $30k (near
        the lows) or $70k (near all-time highs). Structural context is crucial
        to understand where the support/resistance levels are.

        Features:
          · dist_ath_{30,90,365}:  % distance from the period ATH
          · dist_atl_{30,90,365}:  % distance from the period ATL
          · price_position_{30,90}: position in the [ATL, ATH] range → [0, 1]
          · momentum_{30,90}:       % performance vs N days ago
          · round_level_dist:       distance from the nearest round (psychological) level
                                    ($60k, $65k, $70k, etc.)

        Note: these features change slowly (weeks) — they are ideal for the
        StructuralEncoder (stream B of the dual-stream) because they do not share
        the dynamics of the trading features (stream A).
        """
        close = df["close"]

        for days in [30, 90, 365]:
            # TIME-semantic window: calendar days → bars via bars_per_day
            # (days*1440 at 1m, days*24 at 1h). Identity at 1m.
            w = days * self.bars_per_day
            ath = close.rolling(w, min_periods=self._tbars(60)).max()
            atl = close.rolling(w, min_periods=self._tbars(60)).min()

            df[f"dist_ath_{days}d"]   = (close - ath) / ath.replace(0, np.nan)   # ≤0
            df[f"dist_atl_{days}d"]   = (close - atl) / atl.replace(0, np.nan)   # ≥0
            price_range = (ath - atl).replace(0, np.nan)
            df[f"price_pos_{days}d"]  = (close - atl) / price_range              # [0,1]

        # momentum = log-return vs N days ago
        # TIME-semantic: days → bars via bars_per_day (identity at 1m).
        for days in [7, 30, 90]:
            w = days * self.bars_per_day
            df[f"momentum_{days}d"] = np.log(
                close / close.shift(w).replace(0, np.nan)
            )

        # Round psychological levels (multiples of $1000 for BTC).
        round_level = (close / 1000).round() * 1000
        df["round_level_dist"] = (close - round_level) / close.replace(0, np.nan)

        # price vs 200-MINUTE MA (~3.3h, intraday) — NOT 200 days.
        # TIME-semantic: 200 min → bars via _tbars (200 at 1m, 3 at 1h ≈ 3h:
        # the name stays time-accurate).
        df["price_vs_ma200m"] = close / close.rolling(self._tbars(200), min_periods=self._tbars(50)).mean() - 1

        return df

    # ── Volatility / regime ───────────────────────────────────────────────────
    # Realized volatility, cross-scale ratios, return skew and kurtosis.
    def _volatility(self, df):
        for w in self.windows:
            df[f"vol_std_{w}"]  = df["log_ret"].rolling(w).std()
            df[f"vol_mean_{w}"] = df["log_ret"].rolling(w).mean()
        for w1, w2 in [(5, 20), (5, 60), (20, 60)]:
            c1, c2 = f"vol_std_{w1}", f"vol_std_{w2}"
            if c1 in df.columns and c2 in df.columns:
                df[f"vol_ratio_{w1}_{w2}"] = df[c1] / df[c2].replace(0, np.nan)
        df["realized_var_5"]  = (df["log_ret"] ** 2).rolling(5).mean()
        df["realized_var_20"] = (df["log_ret"] ** 2).rolling(20).mean()
        df["ret_skew_20"]     = df["log_ret"].rolling(20).skew()
        df["ret_kurt_20"]     = df["log_ret"].rolling(20).kurt()
        return df

    # ── HAR-CJ features (A4, config-gated) ───────────────────────────────────
    # Continuous/jump decomposition of realized variance as INPUT
    # (Andersen–Bollerslev–Diebold 2007): BV = (π/2)·mean(|r_t|·|r_{t-1}|)
    # estimates the continuous component (jump-robust); J = max(RV−BV, 0)
    # the jump component; jump_ratio = J/RV ∈ [0,1]. C and J have different
    # persistence → separating them helps EVEN-moment forecasts (the
    # semivariance probe failed as a TARGET, not as input — orthogonal kill).
    # TIME-semantic 1d/1w scales via _tbars (calendar HAR windows).
    # CAUSAL: trailing rollings on already-closed log_ret only (r_t·r_{t-1}).
    def _har_cj(self, df):
        abs_r = df["log_ret"].abs()
        # adjacent product |r_t|·|r_{t-1}| — at index t uses only the past.
        bipow = abs_r * abs_r.shift(1)
        for scale, minutes in (("1d", 1440), ("1w", 10080)):
            w  = self._tbars(minutes)
            rv = (df["log_ret"] ** 2).rolling(w).mean()
            # μ₁⁻² = π/2 makes BV an unbiased estimator of jump-free RV.
            bv   = (np.pi / 2.0) * bipow.rolling(w).mean()
            jump = (rv - bv).clip(lower=0.0)
            # ratio=0 where RV=0 (flat bars); warmup NaNs preserved.
            ratio = (jump / rv.replace(0.0, np.nan)).mask(rv == 0.0, 0.0)
            df[f"bv_{scale}"]         = bv
            df[f"jump_{scale}"]       = jump
            df[f"jump_ratio_{scale}"] = ratio
        return df

    # ── Time features ─────────────────────────────────────────────────────────
    # Cyclic encoding (hour/day/month) + trading-session flags.
    def _time_features(self, df):
        hour  = df["open_time"].dt.hour + df["open_time"].dt.minute / 60
        dow   = df["open_time"].dt.dayofweek
        month = df["open_time"].dt.month
        df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
        df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["session_asia"]    = ((hour >= 0)  & (hour < 8)).astype(float)
        df["session_london"]  = ((hour >= 8)  & (hour < 16)).astype(float)
        df["session_ny"]      = ((hour >= 13) & (hour < 21)).astype(float)
        df["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)
        return df

    # ── Lag features ──────────────────────────────────────────────────────────
    # Lags 1..N of returns, volume z-score and taker ratio (short memory).
    def _lags(self, df):
        for lag in range(1, self.lag_periods + 1):
            df[f"lag_ret_{lag}"]   = df["log_ret"].shift(lag)
            df[f"lag_vol_{lag}"]   = df["vol_zscore_20"].shift(lag)
            df[f"lag_taker_{lag}"] = df["taker_buy_ratio"].shift(lag)
        return df

    # ── Fractional Differencing (López de Prado, AFML) ──────────────────────
    # Truncated binomial weights for FFD (stationarity keeping memory).
    @staticmethod
    def _frac_diff_weights(d: float, thresh: float = 1e-5) -> np.ndarray:
        """
        Binomial-series weights for fractional differencing.

        w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k
        Weights are truncated when |w_k| < thresh (Fixed-width FFD).
        Returns the reversed weights for direct convolution with np.convolve.
        """
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < thresh:
                break
            w.append(w_k)
            k += 1
            if k > 5000:   # safety cap
                break
        return np.array(w[::-1])   # reversed for convolution

    # Vectorized FFD of log(close) and log(volume+1); skipped if d=0.
    def _frac_diff(self, df):
        """
        Fractional differencing of log(close) and log(volume+1).

        With standard d = 1.0, log-returns remove ALL of the series'
        memory. With 0 < d < 1 (typically 0.3-0.7) one obtains
        stationarity while preserving long-range autocorrelation.

        Uses the FFD method (Fixed-width Fractional Differencing):
          frac_diff(x, d) = sum_{k=0}^{K} w_k * x_{t-k}
        where weights are truncated when |w_k| < 1e-5.
        The convolution is vectorized with np.convolve (no Python loop over rows).

        Added features:
          · frac_diff_close:  FFD of log(close)
          · frac_diff_volume: FFD of log(volume + 1)

        Skipped if frac_diff_d == 0.0 (backward compatible).
        """
        d = self.frac_diff_d
        if d == 0.0:
            return df

        weights = self._frac_diff_weights(d)
        width = len(weights)
        log.info(f"  Fractional diff: d={d}, window={width} pesi")

        # ── log(close) ──────────────────────────────────────────────────
        log_close = np.log(df["close"].values.astype(np.float64))
        conv = np.convolve(log_close, weights, mode="full")[:len(log_close)]
        # The first (width - 1) observations lack enough history → NaN
        result_close = np.empty(len(log_close), dtype=np.float64)
        result_close[:width - 1] = np.nan
        result_close[width - 1:] = conv[width - 1:]
        df["frac_diff_close"] = result_close

        # ── log(volume + 1) ─────────────────────────────────────────────
        log_vol = np.log1p(df["volume"].values.astype(np.float64))
        conv_v = np.convolve(log_vol, weights, mode="full")[:len(log_vol)]
        result_vol = np.empty(len(log_vol), dtype=np.float64)
        result_vol[:width - 1] = np.nan
        result_vol[width - 1:] = conv_v[width - 1:]
        df["frac_diff_volume"] = result_vol

        return df

    # ── Normalization ─────────────────────────────────────────────────────────
    # Columns already in natural scale (cyclic, [0,1], binary) — no scaler.
    _NO_SCALE = {
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
        "session_asia","session_london","session_ny","session_overlap",
        "taker_buy_ratio","intraday_pos","target_dir",
        # New microstructure features already in [0,1]
        "body_ratio","upper_shadow","lower_shadow",
    }

    # No-scale set; with RevIN adds raw returns (handled per-instance).
    def _no_scale_set(self) -> set:
        """
        Set of columns NOT to scale with the global RobustScaler.

        Default behavior: returns _NO_SCALE (backward compatible).

        When self.use_revin=True, dynamically adds the raw return columns
        (log_ret, log_ret_high, log_ret_low, log_ret_vol and all
        lag_ret_{N}). RevIN normalizes these features per instance, and
        denormalizing the predictions must bring mu back into the same
        space as the raw target (target_ret = sum of log_ret). If the global
        scaler standardized them, RevIN would operate on an already-scaled
        feature and denormalize into scaled space — misaligned with the
        raw target (~1e-4), and the affine (gamma, beta) would end up absorbing
        the mismatch instead of leaving RevIN its original role of
        removing local distribution shift.
        """
        no_scale = set(self._NO_SCALE)
        if getattr(self, "use_revin", False):
            no_scale.update({
                "log_ret", "log_ret_high", "log_ret_low", "log_ret_vol",
            })
            # lag_ret_{N} derive from log_ret.shift(N): same raw scale
            for lag in range(1, self.lag_periods + 1):
                no_scale.add(f"lag_ret_{lag}")
        return no_scale

    # Scaler fit on training rows only — prevents data leakage.
    def fit_scaler_only(self, df: pd.DataFrame) -> "FeatureBuilder":
        """
        Fits a single multi-column RobustScaler on the rows of df without transforming anything.
        Used to avoid data leakage: fit only on the training rows,
        then transform the whole dataset with _normalize(fit=False).

        Args:
            df: DataFrame with the training rows (index 0..train_end)

        Returns:
            self (per chaining)
        """
        no_scale = self._no_scale_set()
        to_scale = [c for c in self.feature_cols if c not in no_scale and c in df.columns]
        self._scale_cols = to_scale
        self.scalers = {}   # empty — back-compat with old pkl

        X = df[to_scale].values.astype(np.float64)
        # Impute NaN with per-column median for quantile fit only.
        with np.errstate(all="ignore"):
            col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        X_imp = np.where(np.isnan(X), col_medians, X)
        self.scaler = RobustScaler()
        self.scaler.fit(X_imp)

        # Clip [P0.1, P99.9] fitted on training only (adaptive, no leakage).
        X_scaled_tr = self.scaler.transform(X_imp)
        with np.errstate(all="ignore"):
            self.clip_lo_ = np.nanpercentile(X_scaled_tr, 0.1, axis=0)
            self.clip_hi_ = np.nanpercentile(X_scaled_tr, 99.9, axis=0)

        log.info(
            f"Scaler multi-colonna fittato su {len(df):,} righe di training "
            f"({len(to_scale)} colonne)"
        )
        return self

    # Applies (and optionally fits) the multi-column RobustScaler + clip, in-place.
    def _normalize(self, df, fit: bool = True):
        """
        Normalizes the features in-place, overwriting the original columns.

        RAM FIX — in-place overwrite instead of *_scaled columns:
          The old approach created col + "_scaled" for every scaled column,
          doubling the DataFrame's RAM. With 2M rows and 55+ features,
          this meant holding both raw and scaled values in memory.

          Now the normalized values directly overwrite the original columns.
          Raw columns are not needed after normalization: the backtest loads
          open/high/low/close/volume directly from the parquet (OHLCV columns),
          not the engineered features.

          Savings: ~55 columns × 2M rows × 4 bytes (float32) ≈ 440 MB.

        OPTIMIZATION — multi-column RobustScaler (single object instead of 60+):
          Instead of 60 separate RobustScalers (one per column), uses a single
          RobustScaler fitted on the whole matrix. Removes the Python loop for the
          transform and reduces ~60 objects to 1 in PipelineState.pkl.
          NaNs are preserved: they are imputed with 0 for the transform
          (neutral value after centering), then restored from the mask.
        """
        no_scale = self._no_scale_set()
        to_scale = [c for c in self.feature_cols if c not in no_scale and c in df.columns]
        if fit:
            self._scale_cols = to_scale
            self.scalers = {}
            X = df[to_scale].values.astype(np.float64)
            with np.errstate(all="ignore"):
                col_medians = np.nanmedian(X, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            X_imp = np.where(np.isnan(X), col_medians, X)
            self.scaler = RobustScaler()
            self.scaler.fit(X_imp)
            # adaptive clip fitted on training only
            X_scaled_tr = self.scaler.transform(X_imp)
            with np.errstate(all="ignore"):
                self.clip_lo_ = np.nanpercentile(X_scaled_tr, 0.1, axis=0)
                self.clip_hi_ = np.nanpercentile(X_scaled_tr, 99.9, axis=0)

        if self.scaler is not None and self._scale_cols:
            # intersect _scale_cols with present columns
            cols = [c for c in self._scale_cols if c in df.columns]
            X = df[cols].values.astype(np.float64)
            nan_mask = np.isnan(X)
            # Impute NaN with 0 = RobustScaler median after centering (neutral).
            X_imp = np.where(nan_mask, 0.0, X)

            if cols == self._scale_cols:
                X_scaled = self.scaler.transform(X_imp)
            else:
                # partial scaler via fit indices
                col_idx = [self._scale_cols.index(c) for c in cols]
                import sklearn.preprocessing as _skpp
                sub_scaler = _skpp.RobustScaler()
                sub_scaler.center_ = self.scaler.center_[col_idx]
                sub_scaler.scale_  = self.scaler.scale_[col_idx]
                sub_scaler.n_features_in_ = len(col_idx)
                X_scaled = sub_scaler.transform(X_imp)

            X_scaled[nan_mask] = np.nan
            # per-feature winsorization with training bounds
            if self.clip_lo_ is not None:
                if len(self.clip_lo_) == X_scaled.shape[1]:
                    X_scaled = np.clip(X_scaled, self.clip_lo_, self.clip_hi_)
                elif cols != self._scale_cols:
                    col_idx = [self._scale_cols.index(c) for c in cols]
                    X_scaled = np.clip(X_scaled, self.clip_lo_[col_idx], self.clip_hi_[col_idx])
            df[cols] = X_scaled

        return df

    # Extracts a single-column scaler (new multi or legacy per-column).
    def _get_scaler_for_col(self, col: str) -> Optional[RobustScaler]:
        """
        Backward compatibility: returns a single-column RobustScaler, working
        with both the new multi-column format and the old per-column one.

        New format (multi-column self.scaler):
          Extracts center and scale for the requested column from the multi-scaler
          parameters and builds a "virtual" RobustScaler for that single column.

        Old format (self.scalers dict):
          Returns self.scalers.get(col) directly, as before.

        Used by PipelineState and the live engine for single-column transforms.
        """
        if self.scaler is not None and col in self._scale_cols:
            idx = self._scale_cols.index(col)
            s = RobustScaler()
            s.center_         = np.array([self.scaler.center_[idx]])
            s.scale_          = np.array([self.scaler.scale_[idx]])
            s.n_features_in_  = 1
            return s
        # fallback for pre-refactor pkl
        return self.scalers.get(col)

    # ── PUBLIC ────────────────────────────────────────────────────────────────
    # Orchestrator: runs all steps, dual-stream split and normalization.
    def build(self, df: pd.DataFrame, normalize: bool = True, fit: bool = True,
              funding_df: "Optional[pd.DataFrame]" = None) -> pd.DataFrame:
        df = df.copy()
        steps = [
            ("log-returns",   lambda d: self._returns(d, self.forecast_horizon)),
            ("VWAP",          self._vwap),
            ("technical",     self._technicals),
            ("volume",        self._volume_features),
            ("CVD",           self._cvd_features),
            ("volatility",    self._volatility),
            # A4 HAR-CJ — step present ONLY when the flag is on: default OFF =
            # step list and output bit-identical to the production path (104 features).
            *([("HAR-CJ", self._har_cj)] if self.use_har_cj else []),
            ("time",          self._time_features),
            ("lags",          self._lags),
            ("frac_diff",     self._frac_diff),           # López de Prado FFD
            ("structural",    self._structural_features),
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented",
                                    category=pd.errors.PerformanceWarning)
            for i, (name, fn) in enumerate(steps):
                log.info(f"  → {name}")
                df = fn(df)
                # B3 — removed the intermediate defrag df.copy() (i==4): value-identical (a copy
                # doesn't change values) and redundant with the final defrag before normalization.
            log.info("  → volume profile (multi-scale, ~30-60s) ...")
            df = self._volume_profile(df)

        if funding_df is not None and len(funding_df) > 0:
            log.info("  → funding rate features")
            df = self._funding_features(df, funding_df)
        else:
            log.warning(
                "Funding rate non disponibile — feature funding_rate* escluse. "
                "Esegui scripts/01_download_data.py per scaricarle."
            )

        # feature interactions for regime detection
        # safe column access — on short datasets some base features may be missing.
        # Audit #28 (2026-06-03): added outer try/except as a safety-net against
        # dtype mismatch / Series×scalar edge cases on very short datasets (no KeyError).
        g = lambda c: df[c] if c in df.columns else 0.0
        try:
            df["vol_x_pos"]          = g("vol_ratio_5_20") * g("price_pos_30d")
            df["momentum_x_funding"] = g("momentum_30d")   * g("funding_rate_dev")
            df["cvd_x_vol"]          = g("cvd_norm")        * g("vol_std_20")
        except Exception as _e:
            log.warning(
                f"Feature interactions skipped on short dataset ({_e}) — "
                "vol_x_pos / momentum_x_funding / cvd_x_vol set to 0.0"
            )
            for _c in ("vol_x_pos", "momentum_x_funding", "cvd_x_vol"):
                if _c not in df.columns:
                    df[_c] = 0.0

        exclude = {"open_time","close_time","date_utc","cum_pv","cum_vol","pv",
                   "typical_price","obv","obv_roc_20","obv_roc_60"}
        all_cols = [c for c in df.columns if c not in exclude]

        # Dual-stream split via explicit set (no fragile prefix matching).
        # Stream B (structural): slow features (days/hours) — market context.
        # Stream A (dynamic):   everything else, changes every minute.
        _STRUCTURAL_COLS = {
            "vp_poc_dist", "vp_vah_dist", "vp_val_dist", "vp_concentration",
            "vp_poc_dist_short", "vp_vah_dist_short", "vp_val_dist_short", "vp_concentration_short",
            "vp_poc_dist_long",  "vp_vah_dist_long",  "vp_val_dist_long",  "vp_concentration_long",
            "vp_poc_convergence",
            "dist_ath_30d",   "dist_atl_30d",   "price_pos_30d",
            "dist_ath_90d",   "dist_atl_90d",   "price_pos_90d",
            "dist_ath_365d",  "dist_atl_365d",  "price_pos_365d",
            "momentum_7d", "momentum_30d", "momentum_90d",
            "round_level_dist", "price_vs_ma200m",
            "session_position",
            "funding_rate", "funding_rate_1d", "funding_rate_dev",
        }
        structural_cols = [c for c in all_cols if c in _STRUCTURAL_COLS]
        dynamic_cols    = [c for c in all_cols if c not in _STRUCTURAL_COLS]

        # dyn first then struct → first N_DYN cols = stream A
        ordered_cols = dynamic_cols + structural_cols
        self.feature_cols     = ordered_cols
        self.n_dynamic_features = len(dynamic_cols)

        log.info(
            f"Features: {len(dynamic_cols)} dinamiche (stream A) + "
            f"{len(structural_cols)} strutturali (stream B) = "
            f"{len(ordered_cols)} totale"
        )

        # Defragment before normalization (100+ column inserts fragment the DF).
        df = df.copy()

        if normalize:
            df = self._normalize(df, fit=fit)

        n_before = len(df)
        df = df.dropna(subset=["target_ret"]).reset_index(drop=True)
        log.info(f"Rimosse {n_before - len(df)} righe NaN — {len(df)} valide")
        return df


# Builds sliding windows (n, window, feat) via stride_tricks, drops NaNs.
def create_windows(df: pd.DataFrame, feature_cols: list[str],
                   window_size: int = 60, target_col: str = "target_ret",
                   window_stride: int = 1):
    """
    Creates windows (n, window, features) for the LSTM.

    Uses numpy stride_tricks instead of a Python loop: with 2M+ candle
    datasets the original loop would take minutes, stride_tricks is O(1)
    in time and creates a view (zero-copy up to the NaN filter).

    window_stride: sample 1 window every N candles.
      stride=1  → all windows (default, max samples, high RAM)
      stride=10 → 10x fewer windows (~7 GB for 2.8M candles)
      stride=20 → 20x fewer windows (~3.7 GB, recommended for datasets >1M candles)
      Training is equivalent because adjacent windows are nearly identical
      (they differ by only 1 candle) — the stride reduces redundancy.
    """
    scaled_cols = [c for c in feature_cols if c in df.columns]

    feat  = df[scaled_cols].values.astype(np.float32)
    tgt   = df[target_col].values.astype(np.float32)
    times = df["open_time"].values

    n, n_feat = feat.shape

    # RAM estimate pre-alloc (warn if OOM likely)
    stride_eff   = max(1, int(window_stride))
    n_windows_est = (n - window_size) // stride_eff
    ram_gb_est    = n_windows_est * window_size * n_feat * 4 / 1e9
    if ram_gb_est > 8.0:
        log.warning(
            f"create_windows: stima RAM = {ram_gb_est:.1f} GB "
            f"({n_windows_est:,} windows × {window_size} × {n_feat} × float32). "
            f"Se OOM, aumenta window_stride nel config (ora={stride_eff})."
        )
    else:
        log.info(
            f"create_windows: stima RAM = {ram_gb_est:.1f} GB "
            f"({n_windows_est:,} windows, stride={stride_eff})"
        )

    # sliding_window_view zero-copy → shape (n-w+1, 1, w, f)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(feat, window_shape=(window_size, n_feat))[:, 0, :, :]

    max_idx = min(len(windows), n - window_size - 1)
    # stride on zero-copy view before final copy
    wins  = windows[:max_idx:stride_eff]
    y_raw = tgt   [window_size: window_size + max_idx: stride_eff]
    t_raw = times [window_size: window_size + max_idx: stride_eff]

    # drop NaN windows (incomplete rolling at start)
    valid = ~np.isnan(wins).any(axis=(1, 2))
    X = np.ascontiguousarray(wins[valid], dtype=np.float32)
    y = y_raw[valid].astype(np.float32)
    t = t_raw[valid]

    log.info(f"Windows: X={X.shape}  y={y.shape}  ({X.shape[-1]} features)  "
             f"stride={stride_eff}  ({(~valid).sum()} finestre NaN scartate)")
    return X, y, t


# Temporal train/val/test split (no shuffle) for final training.
def temporal_split(X, y, t, val_frac=0.10, test_frac=0.10):
    """
    Simple TEMPORAL split — used in production for final training.
    Kept for compatibility with 01_download_data.py.
    For robust model evaluation use walk_forward_folds().
    """
    n = len(X)
    iv = int(n * (1 - val_frac - test_frac))
    it = int(n * (1 - test_frac))
    return {
        "X_train": X[:iv],   "y_train": y[:iv],   "t_train": t[:iv],
        "X_val":   X[iv:it], "y_val":   y[iv:it], "t_val":   t[iv:it],
        "X_test":  X[it:],   "y_test":  y[it:],   "t_test":  t[it:],
    }


# Purged walk-forward k-fold with embargo (robust estimate, no look-ahead).
def walk_forward_folds(
    X:             np.ndarray,
    y:             np.ndarray,
    t:             np.ndarray,
    n_folds:       int = 3,
    embargo_steps: int = 60,
    val_frac:      float = 0.10,
) -> list[dict]:
    """
    Purged Walk-Forward k-Fold with embargo period.

    CONCEPTUAL FIX — masked temporal overfitting:
    ────────────────────────────────────────────────────
    The problem with a single split (80/10/10):
      The model is evaluated on a single test period (the last 10%).
      If that period happens to be favorable (bull run, low volatility),
      the metrics look good but do not generalize.
      Conversely, a difficult market in the last 10% can make the
      model look worse than it is.

    The solution — Purged Walk-Forward:
      Splits the dataset into K temporal folds. For each fold:
        · Train:    all data before the fold (expanding window)
        · Embargo:  `embargo_steps` samples discarded between train and val
                    (prevents the gradient of the last training candle
                     from leaking into validation through autocorrelation)
        · Val:      the current fold
      Metrics are averaged over all folds → robust estimate.

    Embargo period:
      With 60-minute windows, candle t of the val set is predicted
      using features that include candles up to t-1. If train ends
      at t-window, the last training windows and the first validation
      windows partially overlap. The embargo discards these samples.

    Args:
        n_folds:       number of folds (3 recommended with ≤10k candles)
        embargo_steps: samples to discard between end of training and start of val
        val_frac:      fraction used for validation inside each fold

    Returns:
        List of dicts, one per fold:
          {fold, X_train, y_train, t_train, X_val, y_val, t_val,
           train_end_idx, val_start_idx, val_end_idx}
    """
    n      = len(X)
    folds  = []

    # Fold 0 = earliest val period; fold K-1 = classic test set.
    fold_size = n // (n_folds + 1)   # +1 so the first fold has training data

    for k in range(n_folds):
        # fold k = window [val_start, val_end)
        val_start = (k + 1) * fold_size
        val_end   = min(val_start + fold_size, n)

        if val_end - val_start < 50:
            log.warning(f"Fold {k}: troppo pochi campioni ({val_end-val_start}), skip.")
            continue

        # train = everything before val, minus embargo
        train_end = val_start - embargo_steps
        if train_end < fold_size:
            # structural note — fold 0 has val_start=fold_size, so
            # train_end=fold_size-embargo<fold_size whenever embargo>0.
            # Expected: declared n_folds → n_folds-1 effective folds.
            # To get K effective folds, set n_folds=K+1.
            log.warning(
                f"Fold {k}: training troppo corto ({train_end} < fold_size={fold_size}), "
                f"skip. [embargo={embargo_steps}; per K fold effettivi usa n_folds=K+1]"
            )
            continue

        # inner val = last val_frac of train for early stopping
        iv = int(train_end * (1 - val_frac))

        folds.append({
            "fold":           k,
            "X_train":        X[:iv],
            "y_train":        y[:iv],
            "t_train":        t[:iv],
            "X_val_internal": X[iv:train_end],          # early stopping
            "y_val_internal": y[iv:train_end],
            "t_val_internal": t[iv:train_end],
            "X_val":          X[val_start:val_end],     # held-out test fold
            "y_val":          y[val_start:val_end],
            "t_val":          t[val_start:val_end],
            "train_end_idx":  train_end,
            "val_start_idx":  val_start,
            "val_end_idx":    val_end,
            "embargo_steps":  embargo_steps,
        })

        log.info(
            f"Fold {k}: train=[0,{iv}] | embargo=[{iv},{val_start}] "
            f"| val=[{val_start},{val_end}] ({val_end-val_start} campioni)"
        )

    if not folds:
        raise ValueError(
            f"Nessun fold valido con n={n}, n_folds={n_folds}, embargo={embargo_steps}. "
            f"Scarica più dati (consigliato: almeno {n_folds * fold_size * 3} campioni)."
        )

    log.info(
        f"Walk-forward: {len(folds)} fold validi | "
        f"embargo={embargo_steps} candele | fold_size≈{fold_size}"
    )
    return folds
