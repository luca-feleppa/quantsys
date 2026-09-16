"""
A7 — Greeks-aware risk layer for the options book (ROADMAP_VOL_BOOK).

The legacy `RiskManager` is delta-one (Kelly on μ/σ, ATR SL, notional-drawdown
circuit breaker): it sees neither vega/gamma nor Deribit's inverse margin.
This module is the SKELETON of the options-book layer:
  1. **Net-vega cap** (and net-delta cap) per book: an order that would push
     exposure past the cap is scaled down (scale ∈ [0,1]) or rejected —
     exposure-REDUCING orders always pass;
  2. **Mark-to-market vega-loss circuit breaker**: drawdown of the book's
     MtM PnL (BTC) with recovery hysteresis (same pattern as the delta-one
     15% DD breaker: trip at threshold, recover below a fraction);
  3. **Deribit inverse margin simulation** (IM/MM for short options + perp),
     parametric: constants from Deribit's public schedule, to be VALIDATED
     against `private/get_account_summary` before decisional use (portfolio
     margin is NOT modeled — standard margin only, which is conservative).
NOT wired into `04b_vol_paper.py`: it enters the critical path ONLY when
sizing moves from fixed 1 contract to Kelly-on-edge (post-gate n≥20, v2).

Unit conventions
----------------
- per-leg greeks = Deribit venue convention (ticker.greeks): delta ∂V_usd/∂S;
  vega = ΔUSD per +1 IV vol-pt; theta = ΔUSD per day — ALL per contract.
- mark/premium in BTC per contract (inverse options); margins in BTC.
- `side` = +1 long / −1 short; `amount` = contracts (≥0).
"""
import logging
from dataclasses import dataclass, field

log = logging.getLogger("quantsys.trading.greeks_risk")


# one leg's greeks as an immutable snapshot (from venue ticker or BS calc).
@dataclass(frozen=True)
class OptionLegGreeks:
    instrument: str
    option_type: str          # "call" | "put"
    side: int                 # +1 long, -1 short
    amount: float             # contracts (>= 0)
    strike: float
    underlying: float         # index/forward USD price
    mark_btc: float           # mark premium BTC per contract
    delta: float              # per contract, venue convention
    gamma: float = 0.0
    vega: float = 0.0         # USD per vol-pt per contract
    theta: float = 0.0

    def __post_init__(self):
        # fail-fast on incoherent inputs — the risk layer must never guess.
        if self.side not in (+1, -1):
            raise ValueError(f"side deve essere ±1 / side must be ±1, got {self.side}")
        if self.amount < 0:
            raise ValueError(f"amount negativo / negative amount: {self.amount}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type sconosciuto / unknown: {self.option_type}")


# book limits — ALL parametric (no pre-registered constant here: values are
# to be decided at the v2 sizing pre-registration).
@dataclass
class GreeksLimits:
    max_net_vega_usd: float = 500.0     # |Σ side·amount·vega| cap (USD/vol-pt)
    max_net_delta: float = 5.0          # |Σ side·amount·delta| cap (BTC-eq per $1-rel)
    cb_max_loss_btc: float = 0.05       # breaker trip: book MtM DD in BTC
    cb_recovery_frac: float = 0.7       # re-arm below this fraction of the threshold
    # A13b — NET book gamma cap (∂delta/∂S, per $1). None = no cap (default,
    # legacy behavior unchanged): expiry-concentrated short convexity (daily
    # ATM Γ explodes in the final hours) is the short-vol arm's risk, not the
    # long straddle's. Value frozen at the v2 sizing pre-registration.
    max_net_gamma: float | None = None


# outcome of an order check: fully allowed, scaled, or rejected (scale=0).
@dataclass
class GreeksCheck:
    allowed: bool
    scale: float              # admissible order fraction ∈ [0,1]
    reasons: list = field(default_factory=list)


def net_greeks(legs: list) -> dict:
    # net book greeks: side×amount-weighted sums (short vega = negative).
    out = {"delta": 0.0, "gamma": 0.0, "vega_usd": 0.0, "theta_usd": 0.0}
    for l in legs:
        w = l.side * l.amount
        out["delta"] += w * l.delta
        out["gamma"] += w * l.gamma
        out["vega_usd"] += w * l.vega
        out["theta_usd"] += w * l.theta
    return out


# ─────────────────── margin simulation Deribit inverse ───────────────────
# Deribit BTC options standard-margin schedule (public, 2026) — PARAMETERS,
# not truth: validate against get_account_summary before decisional use.
# Long option: zero margin (premium paid upfront). Short call/put (per contract):
#   IM = max(0.15 − OTM_amount/underlying, 0.10) + mark_btc
#   MM = 0.075 + mark_btc
# (IM≥MM floor applied to calls AND puts — the venue schedule has it for
# short puts only: deliberate conservative approximation). Perp: IM 2%,
# MM 1% of notional.
IM_BASE_FRAC = 0.15
IM_FLOOR_FRAC = 0.10
MM_BASE_FRAC = 0.075
PERP_IM_FRAC = 0.02
PERP_MM_FRAC = 0.01


def option_margin_btc(leg: OptionLegGreeks) -> dict:
    # (IM, MM) margin in BTC for the WHOLE leg (amount included).
    if leg.side > 0:
        return {"initial": 0.0, "maintenance": 0.0}
    if leg.option_type == "call":
        otm = max(leg.strike - leg.underlying, 0.0)
    else:
        otm = max(leg.underlying - leg.strike, 0.0)
    mm = (MM_BASE_FRAC + leg.mark_btc) * leg.amount
    im = (max(IM_BASE_FRAC - otm / leg.underlying, IM_FLOOR_FRAC) + leg.mark_btc) * leg.amount
    return {"initial": max(im, mm), "maintenance": mm}


def perp_margin_btc(h_usd: float, price: float) -> dict:
    # inverse perp margin: notional fractions in BTC (|H|/S).
    notional_btc = abs(h_usd) / price if price > 0 else 0.0
    return {"initial": PERP_IM_FRAC * notional_btc,
            "maintenance": PERP_MM_FRAC * notional_btc}


def book_margin_btc(legs: list, h_usd: float = 0.0, perp_price: float = 0.0) -> dict:
    # total standard margin (per-leg sum — NO portfolio-margin netting:
    # declared CONSERVATIVE approximation).
    im = mm = 0.0
    for l in legs:
        m = option_margin_btc(l)
        im += m["initial"]; mm += m["maintenance"]
    if h_usd:
        m = perp_margin_btc(h_usd, perp_price)
        im += m["initial"]; mm += m["maintenance"]
    return {"initial": im, "maintenance": mm}


# ─────────────────────────── risk manager ───────────────────────────
class GreeksRiskManager:
    """
    Pre-trade gate (net vega/delta caps) + MtM circuit breaker with
    hysteresis. Stateless over greeks (current book passed in), stateful
    ONLY over MtM PnL (breaker peak/drawdown) — delta-one RiskManager pattern.
    """

    def __init__(self, limits: GreeksLimits | None = None):
        self.limits = limits or GreeksLimits()
        # breaker state: peak of the book's cumulative MtM PnL (BTC) + trip flag.
        self._peak_pnl_btc = 0.0
        self._circuit_open = False

    # ── MtM vega-loss circuit breaker ──
    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    def update_mtm(self, book_pnl_btc: float) -> bool:
        # update the breaker with the book's CUMULATIVE MtM PnL (BTC,
        # options+hedge). Trips when drawdown from peak exceeds
        # cb_max_loss_btc; re-arms with hysteresis once back below
        # recovery_frac×threshold (avoids flip-flopping). Returns True if
        # trading is allowed.
        self._peak_pnl_btc = max(self._peak_pnl_btc, book_pnl_btc)
        dd = self._peak_pnl_btc - book_pnl_btc
        if not self._circuit_open and dd >= self.limits.cb_max_loss_btc:
            self._circuit_open = True
            log.warning(f"CIRCUIT BREAKER vega-loss OPEN: DD MtM {dd:.5f} BTC "
                        f">= {self.limits.cb_max_loss_btc:.5f}")
        elif self._circuit_open and dd < self.limits.cb_recovery_frac * self.limits.cb_max_loss_btc:
            self._circuit_open = False
            log.info(f"circuit breaker riarmato/re-armed: DD {dd:.5f} BTC")
        return not self._circuit_open

    # ── pre-trade gate ──
    @staticmethod
    def _cap_scale(current: float, added: float, cap: float) -> float:
        # max s ∈ [0,1] with an admissible final exposure. Rules:
        # (a) |final| ≤ cap → full; (b) same-sign reduction → full even if
        # still past the cap (closing is always allowed); (c) otherwise
        # scale to the cap edge being approached. MINOR-4 audit: a sign-flip
        # landing past the opposite cap is NOT a reduction → falls into (c):
        # the policy is monotone in order size.
        new = current + added
        if abs(new) <= cap:
            return 1.0
        same_sign = (new >= 0.0) == (current >= 0.0)
        if same_sign and abs(new) <= abs(current):
            return 1.0
        if added == 0.0:
            return 1.0
        # cap edge on the side exposure is moving toward.
        target = cap if new > 0 else -cap
        s = (target - current) / added
        return min(max(s, 0.0), 1.0)

    def check_order(self, book_legs: list, new_legs: list) -> GreeksCheck:
        # evaluates the NEW legs against the book: open breaker → hard
        # reject; vega/delta caps → scale = min of both constraints (0 = reject).
        reasons = []
        if self._circuit_open:
            return GreeksCheck(False, 0.0, ["circuit_breaker_open"])

        cur = net_greeks(book_legs)
        add = net_greeks(new_legs)
        s_vega = self._cap_scale(cur["vega_usd"], add["vega_usd"],
                                 self.limits.max_net_vega_usd)
        s_delta = self._cap_scale(cur["delta"], add["delta"],
                                  self.limits.max_net_delta)
        if s_vega < 1.0:
            reasons.append(f"net_vega_cap(scale={s_vega:.3f})")
        if s_delta < 1.0:
            reasons.append(f"net_delta_cap(scale={s_delta:.3f})")
        scale = min(s_vega, s_delta)
        # A13b — optional gamma cap, same _cap_scale policy (reductions
        # always allowed); inert with max_net_gamma=None.
        if self.limits.max_net_gamma is not None:
            s_gamma = self._cap_scale(cur["gamma"], add["gamma"],
                                      self.limits.max_net_gamma)
            if s_gamma < 1.0:
                reasons.append(f"net_gamma_cap(scale={s_gamma:.3f})")
            scale = min(scale, s_gamma)
        return GreeksCheck(scale > 0.0, scale, reasons)
