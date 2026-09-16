"""Phase 5 — Trading: risk manager, position sizing, signals."""
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import t as t_dist

log = logging.getLogger("quantsys.trading")


# Enums for position direction and close reason.
class Side(Enum):
    LONG = "LONG"; SHORT = "SHORT"; NONE = "NONE"

# Reason a position was closed (for trade logging/analysis).
class CloseReason(Enum):
    STOP_LOSS = "STOP_LOSS"; TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_SL = "TRAILING_SL"; SIGNAL = "SIGNAL"
    MAX_HOLD = "MAX_HOLD"; DRAWDOWN = "DRAWDOWN"; END_OF_DATA = "END_OF_DATA"


# Predicted distribution params (t-Student) + conviction for position sizing.
@dataclass
class DistributionParams:
    """t-Student parameters + conviction score for proportional sizing."""
    mu:         float
    sigma:      float
    nu:         float
    prob_up:    float = 0.5
    conviction: float = 1.0   # [0,1] continuously scales Kelly

# Open position: entry, size, SL/TP and peak price for the trailing stop.
@dataclass
class Position:
    side: Side; entry_price: float; size_usd: float; size_base: float
    entry_candle: int; stop_loss: float; take_profit: float
    trailing_atr: float; peak_price: float = 0.0

    # True if the position is open (side != NONE).
    @property
    def is_open(self): return self.side != Side.NONE

    # Unrealized PnL at current price (sign depends on side).
    def unrealized_pnl(self, price: float) -> float:
        if self.side == Side.LONG:  return (price - self.entry_price) * self.size_base
        if self.side == Side.SHORT: return (self.entry_price - price) * self.size_base
        return 0.0

# Closed trade: immutable record with gross/net PnL and close reason.
@dataclass
class Trade:
    side: Side; entry_price: float; exit_price: float; size_usd: float
    size_base: float; entry_candle: int; exit_candle: int
    close_reason: CloseReason; gross_pnl: float; fees: float
    net_pnl: float; pnl_pct: float; hold_candles: int

# Portfolio state: equity/cash/peak + counters for drawdown and metrics.
@dataclass
class Portfolio:
    equity: float; cash: float; peak_equity: float
    drawdown: float = 0.0; max_drawdown: float = 0.0
    n_trades: int = 0; n_wins: int = 0
    gross_profit: float = 0.0; gross_loss: float = 0.0


class SignalGenerator:
    """
    Generates trading signals from the predicted t-Student distribution.

    CONCEPTUAL FIX — Continuous sizing instead of a binary signal:
    ─────────────────────────────────────────────────────────────
    The problem: the old approach turned the continuous probability
    (0.0 → 1.0) into a binary signal (HOLD/BUY/SELL) with a fixed
    threshold at 0.58. This created a cliff: prob=0.57 → size $0,
    prob=0.59 → size $355k. No confidence information was passed
    to the risk manager beyond the single BUY/SELL bit.

    The fix — sizing proportional to conviction:
      The order size is scaled linearly with the prediction's
      "conviction", defined as:
        conviction = (prob_up - 0.5) * 2   for LONG  (range 0→1)
        conviction = (0.5 - prob_up) * 2   for SHORT (range 0→1)
      The Kelly size is multiplied by conviction^alpha (alpha=0.5
      to smooth — keeps high convictions from dominating too much).

      This removes the discontinuous cliff and allows partial positions
      on uncertain signals instead of ignoring them entirely.

    Minimum threshold:
      We keep a minimum conviction threshold (default 0.55)
      below which no position is opened — the market-maker
      spread and fees require a minimum edge to be
      profitable even with reduced sizing.
    """

    # Initializes generator thresholds (all in raw post-denorm space).
    def __init__(self, prob_threshold: float = 0.55,
                 min_expected_ret: float = 0.0002,
                 max_sigma: float = 0.006,
                 conviction_alpha: float = 0.5,
                 min_snr: float = 0.2):
        self.prob_threshold   = prob_threshold    # open threshold
        self.min_expected_ret = min_expected_ret  # min |μ|
        self.max_sigma        = max_sigma         # max vol
        self.conviction_alpha = conviction_alpha  # smoothing exponent
        # Minimum SNR |μ|/σ — extra gate against entries indistinguishable from noise
        self.min_snr          = min_snr           # minimum signal-to-noise ratio

    # regime threshold removed 2026-06-03 — re-calibrate post paper-trading

    # P(log_ret > 0) from the parametric t-Student CDF.
    def prob_up(self, mu: float, sigma: float, nu: float) -> float:
        """P(log_ret > 0) from the t-Student CDF."""
        return float(1 - t_dist.cdf(-mu / (sigma + 1e-10), df=nu))

    # Conviction [0,1] from prob_up→threshold distance, smoothed by ^alpha.
    def conviction(self, prob_up: float, side: "Side") -> float:
        """
        Conviction score [0, 1] — how strong the signal is.
        Used to scale the Kelly size continuously.

        conviction = 0   → minimum threshold (prob_up = threshold)
        conviction = 1   → maximum certainty (prob_up = 1.0 or 0.0)
        """
        # Conviction = (target_prob - threshold) / (1 - threshold).
        if side.value == "LONG":
            raw = (prob_up - self.prob_threshold) / (1.0 - self.prob_threshold)
        elif side.value == "SHORT":
            raw = ((1 - prob_up) - self.prob_threshold) / (1.0 - self.prob_threshold)
        else:
            return 0.0
        # ^alpha<1 smooths: prevents size blow-up on extreme signals.
        return float(np.clip(raw, 0.0, 1.0) ** self.conviction_alpha)

    # Decides the side (LONG/SHORT/NONE) and records conviction in the result.
    def generate(self, mu: float, sigma: float,
                 nu: float) -> tuple["Side", "DistributionParams"]:
        """
        Generates the signal and computes the conviction score.
        The conviction is stored in DistributionParams.prob_up
        to be used by the RiskManager for sizing.
        """
        p_up = self.prob_up(mu, sigma, nu)
        dist = DistributionParams(mu=mu, sigma=sigma, nu=nu, prob_up=p_up)

        # No-trade zone: vol too high (uncontrolled risk).
        if sigma > self.max_sigma:
            return Side.NONE, dist

        # SNR filter — reject low |μ|/σ signals (entry indistinguishable from noise)
        if sigma > 1e-9 and abs(mu) / sigma < self.min_snr:
            return Side.NONE, dist

        # Side decision: prob ≥ threshold AND |μ| ≥ min_expected_ret.
        if p_up >= self.prob_threshold and mu >= self.min_expected_ret:
            side = Side.LONG
        elif (1 - p_up) >= self.prob_threshold and mu <= -self.min_expected_ret:
            side = Side.SHORT
        else:
            return Side.NONE, dist

        # Conviction → size multiplier consumed by the RiskManager.
        conv = self.conviction(p_up, side)
        dist = DistributionParams(mu=mu, sigma=sigma, nu=nu,
                                  prob_up=p_up, conviction=conv)
        return side, dist


class RiskManager:
    """
    Fractional Kelly + dynamic ATR stop loss + trailing stop + circuit breaker.
    Binance fees: 0.1% maker/taker. Slippage: 0.03% (or sqrt market impact).
    """

    # Initializes capital, risk constraints, slippage model and portfolio.
    def __init__(self, initial_capital=10_000.0, max_risk_per_trade=0.01,
                 sl_atr_mult=2.0, tp_rr_ratio=2.5, max_position_pct=0.25,
                 max_drawdown_stop=0.15, max_hold_candles=120,
                 use_trailing_stop=True, trailing_atr_mult=1.5,
                 fee_rate=0.001, slippage_rate=0.0003,
                 correlation_window: int = 10,
                 max_directional_exposure: float = 0.6,
                 slippage_model: str = "fixed",
                 autocorr_window: int = 50,
                 bars_per_year: int = 525_600):
        """
        correlation_window:         how many recent trades to consider for autocorrelation
        max_directional_exposure:   maximum cumulative directional exposure [0,1]
        slippage_model:             "fixed" = static base_slip,
                                    "sqrt"  = Almgren-Chriss sqrt market impact:
                                              slip = base_slip * sqrt(trade_size / ADV_1m)
        autocorr_window:            how many recent trades to use for the Kelly
                                    autocorrelation estimate (Fix 11). Default 50. With < 10
                                    trades available the correction is not applied.
        bars_per_year:              bars per year for Sharpe/Sortino annualization.
                                    Default 525_600 (1m timeframe → identity with the past);
                                    at 1h pass 8_760.
        """
        self.icap             = initial_capital
        self.max_risk         = max_risk_per_trade
        self.sl_mult          = sl_atr_mult
        self.tp_rr            = tp_rr_ratio
        self.max_pos_pct      = max_position_pct
        self.max_dd_stop      = max_drawdown_stop
        self.max_hold         = max_hold_candles
        self.trailing         = use_trailing_stop
        self.trail_mult       = trailing_atr_mult
        self.fee              = fee_rate
        self.slip             = slippage_rate
        self.slip_model       = slippage_model
        # Directional exposure tracking (side autocorrelation).
        self.corr_window      = correlation_window
        self.max_dir_exp      = max_directional_exposure
        self._recent_sides: list[int] = []   # +1=LONG -1=SHORT
        # Kelly corrected for trade-return autocorrelation (Vince 1992).
        self.autocorr_window  = autocorr_window
        self._recent_trade_returns: list[float] = []
        self._autocorr_cache: Optional[float] = None  # autocorr-factor memo (A5)
        # Bars/year used to annualize Sharpe/Sortino (525_600 at 1m, 8_760 at 1h).
        self.bars_per_year    = bars_per_year
        self.portfolio        = Portfolio(equity=initial_capital, cash=initial_capital,
                                          peak_equity=initial_capital)
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.circuit_breaker  = False
        self.circuit_breaker_triggered_at_dd: float = 0.0
        self.circuit_breaker_candle: int = 0

    # Risk presets for the current `RegimeMarkovBTC` regimes (Quiet/Trending/Stress,
    # 2026-06-03 in quantsys/macro/regime.py). Legacy macro keys are kept as a fallback
    # for the historical ATR-proxy mapping (03_backtest.py pre-fix).
    _REGIME_RISK_PARAMS = {
        # ── Data-driven BTC regimes (preferred, both int and string alias) ────
        0:          {"prob_threshold": 0.54, "max_risk": 0.008, "sl_mult": 1.5, "tp_rr": 2.5},  # Quiet
        1:          {"prob_threshold": 0.52, "max_risk": 0.012, "sl_mult": 2.0, "tp_rr": 3.0},  # Trending
        2:          {"prob_threshold": 0.58, "max_risk": 0.005, "sl_mult": 2.5, "tp_rr": 1.8},  # Stress
        "Quiet":    {"prob_threshold": 0.54, "max_risk": 0.008, "sl_mult": 1.5, "tp_rr": 2.5},
        "Trending": {"prob_threshold": 0.52, "max_risk": 0.012, "sl_mult": 2.0, "tp_rr": 3.0},
        "Stress":   {"prob_threshold": 0.58, "max_risk": 0.005, "sl_mult": 2.5, "tp_rr": 1.8},
        # ── Legacy macro keys (pre-2026-05-23 preset, calibrated in z-space) ──
        "expansion":    {"prob_threshold": 0.53, "max_risk": 0.012, "sl_mult": 1.8, "tp_rr": 3.5},
        "overheating":  {"prob_threshold": 0.60, "max_risk": 0.008, "sl_mult": 2.5, "tp_rr": 2.0},
        "stagflation":  {"prob_threshold": 0.65, "max_risk": 0.005, "sl_mult": 3.0, "tp_rr": 1.5},
        "recession":    {"prob_threshold": 0.60, "max_risk": 0.006, "sl_mult": 2.0, "tp_rr": 2.5},
    }

    # Applies the current regime's risk preset (no-op if unknown).
    # Accepts both int (RegimeMarkovBTC ID: 0=Quiet, 1=Trending, 2=Stress)
    # and string ("Quiet"/"Trending"/"Stress" or legacy "expansion"/...).
    def set_regime(self, regime_id) -> None:
        """Adapts the risk parameters to the current regime (int or str)."""
        params = self._REGIME_RISK_PARAMS.get(regime_id)
        if params is None:
            return
        self.max_risk = params["max_risk"]
        self.sl_mult  = params["sl_mult"]
        self.tp_rr    = params["tp_rr"]
        # preset prob_threshold NOT applied — see set_regime_threshold removal
        # note (SignalGenerator), 2026-06-03. Re-calibrate post paper-trading.
        # The field stays in _REGIME_RISK_PARAMS for historical reference but
        # is not consumed.

    # Directional exposure = |mean(sides)| ∈ [0,1].
    def _directional_exposure(self, new_side: Side) -> float:
        """
        Improvement 10 — Cumulative directional exposure.

        Fractional Kelly assumes independent trades. But consecutive signals
        in the same direction (e.g. 5 LONGs in a row) are not independent:
        the market is trending, and the true risk exposure is larger than
        the sum of the individual positions because they all lose together
        on a reversal.

        This method computes the "directional bias" of the last N trades:
          - 7 LONG and 0 SHORT out of 7 trades → exposure = 1.0
          - 4 LONG and 3 SHORT → exposure = 0.14
          - Exposure 0 = balanced trades (low correlation risk)

        Sizing is reduced proportionally when the exposure
        exceeds max_directional_exposure:
          multiplier = 1.0 if exposure ≤ threshold
          multiplier = (1 - exposure) if exposure > threshold
          → gradual reduction, never zeroed out completely
        """
        if len(self._recent_sides) < 3:
            return 0.0   # insufficient data

        new_val = 1 if new_side == Side.LONG else -1
        recent  = self._recent_sides[-self.corr_window:]

        # 0 = balanced (low corr), 1 = uniformly directional (high corr).
        exposure = abs(float(np.mean(recent + [new_val])))
        return exposure

    # Kelly discount factor for trade-return autocorrelation (Fix 11).
    def _autocorr_kelly_factor(self) -> float:
        # memoize the factor: it depends ONLY on _recent_trade_returns, which mutates only at
        # close_position → recomputing it per entry is wasted (A5). Cache invalidated there.
        if self._autocorr_cache is not None:
            return self._autocorr_cache
        self._autocorr_cache = self._compute_autocorr_kelly_factor()
        return self._autocorr_cache

    def _compute_autocorr_kelly_factor(self) -> float:
        """
        Fix 11 — Kelly corrected for trade-return autocorrelation.

        Classic Kelly assumes independent trades. But consecutive trades
        in the same market regime are correlated (trend persistence):
        during an uptrend consecutive LONG trades have positively
        correlated returns, inflating the optimal Kelly.

        Correction (Vince 1992, Thorp 2006):
          f_adj = f / (1 + 2 * sum(rho_k for k=1..K))
        where rho_k = autocorrelation of trade returns at lag k.

        If the sum of autocorrelations is positive (win/loss streaks),
        Kelly is reduced. If negative (mean reversion), it would be increased
        (but capped at 1.0 for safety).

        Returns:
            Multiplicative factor in [0.2, 1.0] to apply to Kelly.
            1.0 = no correction (independent trades or insufficient data).
        """
        n = len(self._recent_trade_returns)
        if n < 10:
            return 1.0  # insufficient data

        returns = np.array(self._recent_trade_returns[-self.autocorr_window:])
        n_used = len(returns)
        if n_used < 10:
            return 1.0

        # K = √N (heuristic), capped at 10 to avoid noisy estimates.
        K = min(int(np.sqrt(n_used)), 10, n_used // 3)
        if K < 1:
            return 1.0

        mean_r = returns.mean()
        var_r = returns.var()
        if var_r < 1e-12:
            return 1.0  # var ≈ 0 → no estimable corr

        # Σ ρ_k for k=1..K (autocorrelations).
        rho_sum = 0.0
        centered = returns - mean_r
        for k in range(1, K + 1):
            cov_k = np.mean(centered[:-k] * centered[k:])
            rho_k = cov_k / var_r
            rho_sum += rho_k

        # f_adj = f / (1 + 2·Σρ_k). If denom≤0 (strong mean-rev) → no boost.
        denominator = 1.0 + 2.0 * rho_sum
        if denominator <= 0:
            factor = 1.0
        else:
            factor = 1.0 / denominator

        # Clamp [0.2, 1.0]: never zeroed, never boosted.
        factor = float(np.clip(factor, 0.2, 1.0))

        if factor < 0.9:
            log.debug(
                f"Fix 11: autocorr Kelly factor={factor:.3f} "
                f"(rho_sum={rho_sum:+.3f}, K={K}, N={n_used})"
            )

        return factor

    # ── Sizing ────────────────────────────────────────────────────────────────
    # Computes position size via continuous Kelly from the predicted distribution.
    def _size(self, dist: DistributionParams, price: float, atr: float,
              side: Side = Side.NONE):
        """
        Continuous Kelly based on the distribution predicted by the LSTM.

        Improvement — dynamic Kelly f* = μ / σ²:
          The previous discrete Kelly used only prob_up and a fixed TP/RR,
          ignoring σ (predicted volatility) which the LSTM estimates explicitly.

          Correct formula for a continuous distribution:
            f* = μ / σ²
          where μ = expected drift and σ² = predicted variance.
          This is the exact solution of the Kelly optimization problem
          for normally distributed returns (a valid approximation for
          the t-Student with ν > 4).

          Advantages over discrete Kelly:
          · Uses BOTH μ and σ — strong signals with low volatility → large size
          · Strong signals with high volatility → automatically reduced size
          · Removes the dependence on the fixed TP/RR ratio (2.5)
          · Conviction score keeps the scaling proportional to the signal

          Conservative fractioning: divide by 4 (standard in the literature)
          to avoid the risk of ruin with imprecise estimates of μ and σ.
        """
        if self.circuit_breaker:
            # Recovery handled externally in _check_circuit_recovery.
            return 0.0, 0.0
        eq  = self.portfolio.equity
        slp = max(self.sl_mult * atr / max(price, 1e-9), 1e-4)

        # Continuous Kelly f* = μ/σ², capped at 0.5 and /4 (Thorp 2006).
        mu_abs  = abs(dist.mu)
        sigma2  = max(dist.sigma ** 2, 1e-8)
        kelly_raw   = mu_abs / sigma2
        kelly_base  = min(kelly_raw, 0.5) / 4

        # 0.5% floor to avoid tiny sizes on very small μ.
        kelly_base = max(kelly_base, 0.005)

        # Scale with conviction ∈ [0,1]
        kelly = kelly_base * max(0.0, min(1.0, dist.conviction))

        # Autocorrelation correction (Σρ_k > 0 → reduces Kelly).
        autocorr_factor = self._autocorr_kelly_factor()
        kelly *= autocorr_factor

        # Reduction if directional skew > max_dir_exp.
        dir_exp = self._directional_exposure(side)
        if dir_exp > self.max_dir_exp:
            corr_mult = 1.0 - 0.5 * (dir_exp - self.max_dir_exp) / (1.0 - self.max_dir_exp)
            kelly    *= max(0.3, corr_mult)
            log.debug(f"Esposizione direzionale {dir_exp:.2f} → size ×{corr_mult:.2f}")

        # 4 constraints: Kelly, max_risk, max_pos%, available cash — min wins.
        size = min(
            eq * kelly / slp,
            eq * self.max_risk / slp,
            eq * self.max_pos_pct,
            self.portfolio.cash * 0.95,
        )
        size = max(0.0, size) if not (size != size) else 0.0
        return size, size / max(price, 1e-9)

    # ── Slippage ──────────────────────────────────────────────────────────────
    def _compute_slippage(self, price: float, trade_size_usd: float = 0.0,
                          adv_1m: float = 0.0) -> float:
        """
        Computes the slippage rate for the current trade.

        "sqrt" model (Almgren-Chriss 2001, square-root market impact):
          slippage = base_slip * sqrt(trade_size / ADV_1m)

          Intuition: market impact grows with the square root of the
          fraction of volume traded. An order equal to 100% of the
          average 1-minute volume takes the full base slippage; an
          order equal to 25% of the volume takes half the base slippage.

        "fixed" model:
          slippage = base_slip  (independent of size and volume)

        If adv_1m is unavailable (= 0), fall back to the fixed model.
        """
        # Almgren-Chriss √-law: slip ∝ √(size/ADV)
        if self.slip_model == "sqrt" and adv_1m > 0.0 and trade_size_usd > 0.0:
            ratio = trade_size_usd / max(adv_1m, 1.0)
            return self.slip * math.sqrt(ratio)
        return self.slip

    # ── SL/TP ─────────────────────────────────────────────────────────────────
    def _sl_tp(self, side, price, atr, dist):
        """
        SL adaptive to predicted σ + dynamic TP based on the volatility regime.

        Improvement — dynamic TP:
          The fixed TP/RR of 2.5 assumed the market always moves
          2.5× the stop loss — regardless of regime.
          In high volatility the market can move 5× or 10× the SL before reversing.
          In low volatility it rarely reaches 2.5×.

          New approach: TP = max(SL × 2.0, predicted_σ × price × tp_sigma_mult)
          The model itself says how much movement it expects → the TP adapts.

          tp_sigma_mult=3.0: the TP is placed 3σ from entry, which corresponds
          to 99.7% of the normal distribution — it rides the trend without waiting
          for the improbable. Clamped between 2.0× and 5.0× the SL for safety.
        """
        # Clamp σ > 0: guards against upstream bugs (NaN, negative scale).
        sigma = max(float(dist.sigma), 1e-6)
        # SL = max(historical ATR, predicted σ·price·1.5).
        sigma_price  = sigma * price * 1.5
        # σ·price > 5% of price signals σ still in z-space (denorm bug).
        if sigma_price > price * 0.05 and not getattr(self, "_warned_scale", False):
            log.warning(
                f"_sl_tp: σ*price*1.5={sigma_price:.0f} > 5%×price={price*0.05:.0f}. "
                f"Probabile σ in z-score non denormalizzato. Vedi PipelineState.denormalize_predictions."
            )
            self._warned_scale = True
        effective_atr= max(atr, sigma_price)
        sl_d         = self.sl_mult * effective_atr
        # 1 bp floor prevents SL=TP=entry when atr=0 (market halt).
        sl_d         = max(sl_d, price * 1e-4)
        if atr == 0 and not getattr(self, "_warned_atr_zero", False):
            log.warning(f"_sl_tp: atr=0 (mercato halt o dati sporchi). SL floor a {price*1e-4:.2f}")
            self._warned_atr_zero = True

        # TP = 3σ from entry (≈99.7% normal), clamped to [2,5]×SL.
        tp_from_sigma = sigma * price * 3.0
        tp_d          = float(np.clip(tp_from_sigma, sl_d * 2.0, sl_d * 5.0))

        if side == Side.LONG:
            return round(price - sl_d, 2), round(price + tp_d, 2)
        return round(price + sl_d, 2), round(price - tp_d, 2)

    # ── Circuit breaker recovery (extracted from _size for bug fix #2) ──────
    # Re-enables trading once drawdown recovers to 70% of the threshold.
    def _check_circuit_recovery(self) -> None:
        """
        Circuit breaker recovery: if DD falls to 70% of the threshold,
        trading is re-enabled. E.g.: 15% threshold → re-enabled when DD < 10.5%.

        Extracted from _size to avoid state mutation between the 2 calls
        to _size inside open_position (bug #2). Must be called ONCE per
        candle before evaluating sizing/slippage.
        """
        if not self.circuit_breaker:
            return
        # Recovery at 70% of threshold (e.g. 15% → 10.5%).
        recovery_threshold = self.max_dd_stop * 0.70
        if self.portfolio.drawdown < recovery_threshold:
            self.circuit_breaker = False
            log.info(
                f"✔ CIRCUIT BREAKER DISATTIVATO: DD={self.portfolio.drawdown:.1%} "
                f"< soglia recovery {recovery_threshold:.1%} — trading ripreso"
            )

    # ── Open ──────────────────────────────────────────────────────────────────
    # Opens position: 2-step (pre-size → slippage → exec_p → final size).
    def open_position(self, side, price, candle_idx, atr, dist,
                      adv_1m: float = 0.0) -> Optional[Position]:
        # Explicit NaN/Inf guards on critical inputs (math.isfinite covers both)
        # → never open on corrupted data. Replaces the cryptic `v != v` check.
        _critical = {"price": price, "atr": atr, "mu": dist.mu, "sigma": dist.sigma}
        if not all(math.isfinite(float(v)) for v in _critical.values()):
            log.warning(
                f"open_position: input NaN/Inf rejected ({_critical}) → skip"
            )
            return None
        # Recovery evaluated ONCE only (avoids race between 2 _size calls).
        self._check_circuit_recovery()
        if self.circuit_breaker or (self.position and self.position.is_open): return None
        # Only "sqrt" slippage (Almgren-Chriss) depends on trade_size → needs the pre-size.
        # "fixed" (default) is size-independent → skip pre-size, no double _size (A4).
        # Bit-identical: in every non-sqrt case _compute_slippage returns self.slip anyway.
        if self.slip_model == "sqrt" and adv_1m > 0.0:
            sz_usd_est, _ = self._size(dist, price, atr, side=side)
            slip_rate = self._compute_slippage(price, sz_usd_est, adv_1m)
        else:
            slip_rate = self._compute_slippage(price, 0.0, adv_1m)
        exec_p = price + price*slip_rate*(1 if side==Side.LONG else -1)
        sz_usd, sz_base = self._size(dist, exec_p, atr, side=side)
        if sz_usd < 10: return None
        sl, tp = self._sl_tp(side, exec_p, atr, dist)
        if side==Side.LONG and sl>=exec_p: return None
        if side==Side.SHORT and sl<=exec_p: return None
        self.portfolio.cash -= sz_usd + sz_usd*self.fee
        self.position = Position(side=side, entry_price=exec_p, size_usd=sz_usd,
            size_base=sz_base, entry_candle=candle_idx, stop_loss=sl, take_profit=tp,
            trailing_atr=atr, peak_price=exec_p)
        log.debug(f"OPEN {side.value} {exec_p:,.1f}  SL={sl:,.1f}  TP={tp:,.1f}  ${sz_usd:,.0f}")
        return self.position

    # ── Trailing stop ─────────────────────────────────────────────────────────
    # Mark-to-market + ATR-based dynamic trailing stop.
    def update_trailing(self, price, atr):
        if not self.position: return
        # MtM critical in live: without it the CB does not fire intra-trade.
        unrealized = self.position.unrealized_pnl(price)
        mtm_equity = self.portfolio.cash + self.position.size_usd + unrealized
        self.portfolio.equity = mtm_equity
        if mtm_equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = mtm_equity
        self.portfolio.drawdown = (
            (self.portfolio.peak_equity - mtm_equity) / self.portfolio.peak_equity
            if self.portfolio.peak_equity > 0 else 0.0
        )
        if self.portfolio.drawdown > self.portfolio.max_drawdown:
            self.portfolio.max_drawdown = self.portfolio.drawdown

        # Trailing executed only if enabled
        if not self.trailing: return
        d = self.trail_mult * atr
        if self.position.side == Side.LONG:
            self.position.peak_price = max(self.position.peak_price, price)
            new_sl = self.position.peak_price - d
            if new_sl > self.position.stop_loss: self.position.stop_loss = round(new_sl, 2)
        elif self.position.side == Side.SHORT:
            self.position.peak_price = min(self.position.peak_price, price)
            new_sl = self.position.peak_price + d
            if new_sl < self.position.stop_loss: self.position.stop_loss = round(new_sl, 2)

    # ── Check exit ────────────────────────────────────────────────────────────
    # Checks SL/TP/opposite-signal/max-hold on the candle and returns the reason.
    def check_exit(self, high, low, close, candle_idx, new_signal=None) -> Optional[CloseReason]:
        if not self.position: return None
        pos, hold = self.position, candle_idx - self.position.entry_candle
        if pos.side==Side.LONG:
            if low  <= pos.stop_loss:   return CloseReason.STOP_LOSS
            if high >= pos.take_profit: return CloseReason.TAKE_PROFIT
        elif pos.side==Side.SHORT:
            if high >= pos.stop_loss:   return CloseReason.STOP_LOSS
            if low  <= pos.take_profit: return CloseReason.TAKE_PROFIT
        if new_signal and new_signal!=pos.side and new_signal!=Side.NONE: return CloseReason.SIGNAL
        if hold >= self.max_hold: return CloseReason.MAX_HOLD
        return None

    # ── Close ─────────────────────────────────────────────────────────────────
    # Closes the position: applies slippage/fee, updates equity/DD and circuit breaker.
    def close_position(self, reason, price, candle_idx,
                       adv_1m: float = 0.0) -> Optional[Trade]:
        if not self.position: return None
        pos = self.position
        slip_rate = self._compute_slippage(price, pos.size_usd, adv_1m)
        exec_p = price - price*slip_rate*(1 if pos.side==Side.LONG else -1)
        gross  = (exec_p - pos.entry_price)*pos.size_base if pos.side==Side.LONG else (pos.entry_price - exec_p)*pos.size_base
        fees   = pos.size_usd * self.fee * 2
        net    = gross - fees
        self.portfolio.cash += pos.size_usd + gross - pos.size_usd*self.fee
        self.portfolio.equity = self.portfolio.cash
        self.portfolio.n_trades += 1
        if net > 0: self.portfolio.n_wins+=1; self.portfolio.gross_profit+=net
        else:        self.portfolio.gross_loss+=abs(net)
        if self.portfolio.equity > self.portfolio.peak_equity: self.portfolio.peak_equity=self.portfolio.equity
        self.portfolio.drawdown = (self.portfolio.peak_equity-self.portfolio.equity)/self.portfolio.peak_equity
        if self.portfolio.drawdown > self.portfolio.max_drawdown: self.portfolio.max_drawdown=self.portfolio.drawdown
        if self.portfolio.drawdown >= self.max_dd_stop:
            self.circuit_breaker = True
            self.circuit_breaker_triggered_at_dd = self.portfolio.drawdown
            self.circuit_breaker_candle = candle_idx
            log.warning(
                f"⚠ CIRCUIT BREAKER ATTIVATO: DD={self.portfolio.drawdown:.1%} "
                f"(soglia={self.max_dd_stop:.1%}) — trading sospeso"
            )
        trade = Trade(side=pos.side, entry_price=pos.entry_price, exit_price=exec_p,
            size_usd=pos.size_usd, size_base=pos.size_base, entry_candle=pos.entry_candle,
            exit_candle=candle_idx, close_reason=reason, gross_pnl=gross, fees=fees,
            net_pnl=net, pnl_pct=net/pos.size_usd if pos.size_usd>0 else 0,
            hold_candles=candle_idx-pos.entry_candle)
        self.trades.append(trade); self.position = None
        # Direction/return history for autocorrelation estimates (sliding window).
        self._recent_sides.append(1 if trade.side == Side.LONG else -1)
        if len(self._recent_sides) > self.corr_window * 2:
            self._recent_sides = self._recent_sides[-self.corr_window:]
        self._recent_trade_returns.append(trade.pnl_pct)
        if len(self._recent_trade_returns) > self.autocorr_window * 2:
            self._recent_trade_returns = self._recent_trade_returns[-self.autocorr_window:]
        self._autocorr_cache = None  # invalidate memo: history changed (A5)
        return trade

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Computes backtest metrics (Sharpe, Sortino, Calmar, PF, DD, ...).
    def metrics(self) -> dict:
        if not self.trades: return {}
        pnl   = np.array([t.net_pnl for t in self.trades])
        pcts  = np.array([t.pnl_pct for t in self.trades])
        holds = np.array([t.hold_candles for t in self.trades])
        wins  = pnl[pnl>0]; losses = pnl[pnl<0]

        eq = np.concatenate([[self.icap], self.icap + np.cumsum(pnl)])
        rm = np.maximum.accumulate(eq); dd = (rm-eq)/rm; max_dd=float(dd.max())

        avg_hold = holds.mean() if len(holds) else 1
        # Annualization on TOTAL time in position (no iid assumption).
        # hold_candles is a count of BARS; self.bars_per_year (default
        # 525_600 = 1m bars/year) converts bars into a fraction of a year.
        total_bars_exposed = sum(t.hold_candles for t in self.trades) if self.trades else 0
        if total_bars_exposed > 0:
            tpy = self.bars_per_year / max(total_bars_exposed / max(len(self.trades), 1), 1.0)
        else:
            tpy = 1.0
        sharpe = float((pcts.mean()/(pcts.std()+1e-9)) * math.sqrt(tpy)) if len(pcts)>1 else 0.0
        neg = pcts[pcts<0]
        sortino= float((pcts.mean()/(neg.std()+1e-9))*math.sqrt(tpy)) if len(neg)>1 else 0.0
        total_ret = float(eq[-1]/self.icap - 1)
        calmar = total_ret/max_dd if max_dd>0 else float("inf")

        return {
            "n_trades": len(self.trades), "win_rate": len(wins)/len(self.trades),
            "profit_factor": wins.sum()/(abs(losses.sum())+1e-9),
            "avg_win_usd": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss_usd": float(losses.mean()) if len(losses) else 0.0,
            "avg_hold_candles": float(avg_hold), "total_return": total_ret,
            "final_equity": float(eq[-1]), "max_drawdown": max_dd,
            "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
            "total_fees": float(sum(t.fees for t in self.trades)),
            "gross_profit": float(wins.sum()) if len(wins) else 0.0,
            "gross_loss": float(abs(losses.sum())) if len(losses) else 0.0,
            "net_profit": float(pnl.sum()),
            "close_reasons": dict(Counter(t.close_reason.value for t in self.trades)),
            "equity_curve": eq.tolist(),
            "circuit_breaker_triggered": bool(self.circuit_breaker_triggered_at_dd > 0),
            "circuit_breaker_dd_at_trigger": float(self.circuit_breaker_triggered_at_dd),
        }
