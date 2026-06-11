"""Fase 5 — Trading: risk manager, position sizing, segnali."""
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import t as t_dist

log = logging.getLogger("quantsys.trading")


# IT: Enums per direzione posizione e motivo di chiusura.
# EN: Enums for position direction and close reason.
class Side(Enum):
    LONG = "LONG"; SHORT = "SHORT"; NONE = "NONE"

# IT: Motivo di chiusura di una posizione (per log/analisi trade).
# EN: Reason a position was closed (for trade logging/analysis).
class CloseReason(Enum):
    STOP_LOSS = "STOP_LOSS"; TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_SL = "TRAILING_SL"; SIGNAL = "SIGNAL"
    MAX_HOLD = "MAX_HOLD"; DRAWDOWN = "DRAWDOWN"; END_OF_DATA = "END_OF_DATA"


# IT: Parametri distribuzione predetta (t-Student) + conviction per il sizing.
# EN: Predicted distribution params (t-Student) + conviction for position sizing.
@dataclass
class DistributionParams:
    """Parametri t-Student + conviction score per sizing proporzionale."""
    mu:         float
    sigma:      float
    nu:         float
    prob_up:    float = 0.5
    conviction: float = 1.0   # IT: [0,1] scala Kelly continuamente | EN: [0,1] continuously scales Kelly

# IT: Posizione aperta: entry, size, SL/TP e peak per il trailing stop.
# EN: Open position: entry, size, SL/TP and peak price for the trailing stop.
@dataclass
class Position:
    side: Side; entry_price: float; size_usd: float; size_base: float
    entry_candle: int; stop_loss: float; take_profit: float
    trailing_atr: float; peak_price: float = 0.0

    # IT: True se la posizione è aperta (side != NONE).
    # EN: True if the position is open (side != NONE).
    @property
    def is_open(self): return self.side != Side.NONE

    # IT: PnL non realizzato al prezzo corrente (segno dipende dal side).
    # EN: Unrealized PnL at current price (sign depends on side).
    def unrealized_pnl(self, price: float) -> float:
        if self.side == Side.LONG:  return (price - self.entry_price) * self.size_base
        if self.side == Side.SHORT: return (self.entry_price - price) * self.size_base
        return 0.0

# IT: Trade chiuso: record immutabile con PnL lordo/netto e motivo chiusura.
# EN: Closed trade: immutable record with gross/net PnL and close reason.
@dataclass
class Trade:
    side: Side; entry_price: float; exit_price: float; size_usd: float
    size_base: float; entry_candle: int; exit_candle: int
    close_reason: CloseReason; gross_pnl: float; fees: float
    net_pnl: float; pnl_pct: float; hold_candles: int

# IT: Stato portafoglio: equity/cash/peak + contatori per drawdown e metriche.
# EN: Portfolio state: equity/cash/peak + counters for drawdown and metrics.
@dataclass
class Portfolio:
    equity: float; cash: float; peak_equity: float
    drawdown: float = 0.0; max_drawdown: float = 0.0
    n_trades: int = 0; n_wins: int = 0
    gross_profit: float = 0.0; gross_loss: float = 0.0


class SignalGenerator:
    """
    Genera segnali di trading dalla distribuzione t-Student predetta.

    FIX CONCETTUALE — Sizing continuo invece di segnale binario:
    ─────────────────────────────────────────────────────────────
    Il problema: il vecchio approccio convertiva la probabilità continua
    (0.0 → 1.0) in un segnale binario (HOLD/BUY/SELL) con un threshold
    fisso a 0.58. Questo creava un cliff: prob=0.57 → size $0,
    prob=0.59 → size $355k. Nessuna informazione sulla confidenza
    veniva trasmessa al risk manager oltre al singolo bit BUY/SELL.

    Il fix — sizing proporzionale alla conviction:
      La size dell'ordine viene scalata linearmente con la "conviction"
      della predizione, definita come:
        conviction = (prob_up - 0.5) * 2   per LONG  (range 0→1)
        conviction = (0.5 - prob_up) * 2   per SHORT (range 0→1)
      La size Kelly viene moltiplicata per conviction^alpha (alpha=0.5
      per smussare — evita che alte conviction dominino troppo).

      Questo elimina il cliff discontinuo e permette posizioni parziali
      su segnali incerti invece di ignorarli completamente.

    Threshold minimo:
      Manteniamo un threshold minimo di convincimento (default 0.55)
      sotto il quale non si apre nessuna posizione — il market maker
      spread e le commissioni richiedono un edge minimo per essere
      profittevoli anche con sizing ridotto.
    """

    # IT: Inizializza le soglie del generatore (tutte in spazio raw post-denorm).
    # EN: Initializes generator thresholds (all in raw post-denorm space).
    def __init__(self, prob_threshold: float = 0.55,
                 min_expected_ret: float = 0.0002,
                 max_sigma: float = 0.006,
                 conviction_alpha: float = 0.5,
                 min_snr: float = 0.2):
        self.prob_threshold   = prob_threshold    # IT: soglia per aprire | EN: open threshold
        self.min_expected_ret = min_expected_ret  # IT: |μ| minimo | EN: min |μ|
        self.max_sigma        = max_sigma         # IT: vol massima | EN: max vol
        self.conviction_alpha = conviction_alpha  # IT: esponente smussamento | EN: smoothing exponent
        # IT: SNR minimo |μ|/σ — gate aggiuntivo contro entry indistinguibili dal rumore
        # EN: Minimum SNR |μ|/σ — extra gate against entries indistinguishable from noise
        self.min_snr          = min_snr           # IT: rapporto segnale/rumore minimo | EN: minimum signal-to-noise ratio

    # IT: regime threshold rimosso 2026-06-03 — calibrazione da rifare post-paper-trading
    # EN: removed — re-calibrate post paper-trading

    # IT: P(log_ret > 0) dalla CDF della t-Student parametrica.
    # EN: P(log_ret > 0) from the parametric t-Student CDF.
    def prob_up(self, mu: float, sigma: float, nu: float) -> float:
        """P(log_ret > 0) dalla CDF della t-Student."""
        return float(1 - t_dist.cdf(-mu / (sigma + 1e-10), df=nu))

    # IT: Conviction [0,1] dalla distanza prob_up→threshold, smussata ^alpha.
    # EN: Conviction [0,1] from prob_up→threshold distance, smoothed by ^alpha.
    def conviction(self, prob_up: float, side: "Side") -> float:
        """
        Conviction score [0, 1] — quanto è forte il segnale.
        Usato per scalare la size del Kelly in modo continuo.

        conviction = 0   → soglia minima (prob_up = threshold)
        conviction = 1   → certezza massima (prob_up = 1.0 o 0.0)
        """
        # IT: Conviction = (prob_target - threshold) / (1 - threshold).
        # EN: Conviction = (target_prob - threshold) / (1 - threshold).
        if side.value == "LONG":
            raw = (prob_up - self.prob_threshold) / (1.0 - self.prob_threshold)
        elif side.value == "SHORT":
            raw = ((1 - prob_up) - self.prob_threshold) / (1.0 - self.prob_threshold)
        else:
            return 0.0
        # IT: ^alpha<1 smussa: evita size esplosive su segnali estremi.
        # EN: ^alpha<1 smooths: prevents size blow-up on extreme signals.
        return float(np.clip(raw, 0.0, 1.0) ** self.conviction_alpha)

    # IT: Decide il side (LONG/SHORT/NONE) e annota la conviction nel risultato.
    # EN: Decides the side (LONG/SHORT/NONE) and records conviction in the result.
    def generate(self, mu: float, sigma: float,
                 nu: float) -> tuple["Side", "DistributionParams"]:
        """
        Genera segnale e calcola conviction score.
        La conviction viene memorizzata in DistributionParams.prob_up
        per essere usata dal RiskManager nel sizing.
        """
        p_up = self.prob_up(mu, sigma, nu)
        dist = DistributionParams(mu=mu, sigma=sigma, nu=nu, prob_up=p_up)

        # IT: No-trade zone: vol troppo alta (rischio non controllabile).
        # EN: No-trade zone: vol too high (uncontrolled risk).
        if sigma > self.max_sigma:
            return Side.NONE, dist

        # IT: filtro SNR — rifiuta segnali con rapporto |μ|/σ basso (entry indistinguibile da rumore)
        # EN: SNR filter — reject low |μ|/σ signals (entry indistinguishable from noise)
        if sigma > 1e-9 and abs(mu) / sigma < self.min_snr:
            return Side.NONE, dist

        # IT: Side decision: prob ≥ threshold AND |μ| ≥ min_expected_ret.
        # EN: Side decision: prob ≥ threshold AND |μ| ≥ min_expected_ret.
        if p_up >= self.prob_threshold and mu >= self.min_expected_ret:
            side = Side.LONG
        elif (1 - p_up) >= self.prob_threshold and mu <= -self.min_expected_ret:
            side = Side.SHORT
        else:
            return Side.NONE, dist

        # IT: Conviction → moltiplicatore size per il RiskManager.
        # EN: Conviction → size multiplier consumed by the RiskManager.
        conv = self.conviction(p_up, side)
        dist = DistributionParams(mu=mu, sigma=sigma, nu=nu,
                                  prob_up=p_up, conviction=conv)
        return side, dist


class RiskManager:
    """
    Kelly frazionato + stop loss dinamico ATR + trailing stop + circuit breaker.
    Commissioni Binance: 0.1% maker/taker. Slippage: 0.03% (o sqrt market impact).
    """

    # IT: Inizializza capitale, vincoli di rischio, slippage model e portfolio.
    # EN: Initializes capital, risk constraints, slippage model and portfolio.
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
        correlation_window:         quanti trade recenti considerare per autocorrelazione
        max_directional_exposure:   massima esposizione direzionale cumulata [0,1]
        slippage_model:             "fixed" = statico base_slip,
                                    "sqrt"  = Almgren-Chriss sqrt market impact:
                                              slip = base_slip * sqrt(trade_size / ADV_1m)
        autocorr_window:            quanti trade recenti usare per stima autocorrelazione
                                    Kelly (Fix 11). Default 50. Se < 10 trade disponibili
                                    la correzione non viene applicata.
        bars_per_year:              barre per anno per l'annualizzazione Sharpe/Sortino.
                                    Default 525_600 (timeframe 1m → identità col passato);
                                    a 1h passare 8_760.
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
        # IT: Tracking esposizione direzionale (autocorrelazione side).
        # EN: Directional exposure tracking (side autocorrelation).
        self.corr_window      = correlation_window
        self.max_dir_exp      = max_directional_exposure
        self._recent_sides: list[int] = []   # IT: +1=LONG -1=SHORT | EN: +1=LONG -1=SHORT
        # IT: Kelly corretto per autocorrelazione trade returns (Vince 1992).
        # EN: Kelly corrected for trade-return autocorrelation (Vince 1992).
        self.autocorr_window  = autocorr_window
        self._recent_trade_returns: list[float] = []
        # IT: Barre/anno per annualizzare Sharpe/Sortino (525_600 a 1m, 8_760 a 1h).
        # EN: Bars/year used to annualize Sharpe/Sortino (525_600 at 1m, 8_760 at 1h).
        self.bars_per_year    = bars_per_year
        self.portfolio        = Portfolio(equity=initial_capital, cash=initial_capital,
                                          peak_equity=initial_capital)
        self.position: Optional[Position] = None
        self.trades: list[Trade] = []
        self.circuit_breaker  = False
        self.circuit_breaker_triggered_at_dd: float = 0.0
        self.circuit_breaker_candle: int = 0

    # IT: Preset di rischio per i regimi correnti `RegimeMarkovBTC` (Quiet/Trending/Stress,
    #     implementato 2026-06-03 in quantsys/macro/regime.py). Le vecchie chiavi macro
    #     restano come legacy fallback per il proxy ATR storico (03_backtest.py pre-fix).
    # EN: Risk presets for the current `RegimeMarkovBTC` regimes (Quiet/Trending/Stress,
    #     2026-06-03 in quantsys/macro/regime.py). Legacy macro keys are kept as a fallback
    #     for the historical ATR-proxy mapping (03_backtest.py pre-fix).
    _REGIME_RISK_PARAMS = {
        # ── Regimi data-driven BTC (preferiti, sia int che alias stringa) ─────
        0:          {"prob_threshold": 0.54, "max_risk": 0.008, "sl_mult": 1.5, "tp_rr": 2.5},  # Quiet
        1:          {"prob_threshold": 0.52, "max_risk": 0.012, "sl_mult": 2.0, "tp_rr": 3.0},  # Trending
        2:          {"prob_threshold": 0.58, "max_risk": 0.005, "sl_mult": 2.5, "tp_rr": 1.8},  # Stress
        "Quiet":    {"prob_threshold": 0.54, "max_risk": 0.008, "sl_mult": 1.5, "tp_rr": 2.5},
        "Trending": {"prob_threshold": 0.52, "max_risk": 0.012, "sl_mult": 2.0, "tp_rr": 3.0},
        "Stress":   {"prob_threshold": 0.58, "max_risk": 0.005, "sl_mult": 2.5, "tp_rr": 1.8},
        # ── Chiavi macro legacy (preset pre-2026-05-23, calibrato in z-space) ─
        "expansion":    {"prob_threshold": 0.53, "max_risk": 0.012, "sl_mult": 1.8, "tp_rr": 3.5},
        "overheating":  {"prob_threshold": 0.60, "max_risk": 0.008, "sl_mult": 2.5, "tp_rr": 2.0},
        "stagflation":  {"prob_threshold": 0.65, "max_risk": 0.005, "sl_mult": 3.0, "tp_rr": 1.5},
        "recession":    {"prob_threshold": 0.60, "max_risk": 0.006, "sl_mult": 2.0, "tp_rr": 2.5},
    }

    # IT: Applica il preset di rischio del regime corrente (no-op se ignoto).
    #     Accetta sia int (ID di RegimeMarkovBTC: 0=Quiet, 1=Trending, 2=Stress)
    #     sia stringa ("Quiet"/"Trending"/"Stress" o legacy "expansion"/...).
    # EN: Applies the current regime's risk preset (no-op if unknown).
    #     Accepts both int (RegimeMarkovBTC ID: 0=Quiet, 1=Trending, 2=Stress)
    #     and string ("Quiet"/"Trending"/"Stress" or legacy "expansion"/...).
    def set_regime(self, regime_id) -> None:
        """Adatta i parametri di rischio al regime corrente (int o str)."""
        params = self._REGIME_RISK_PARAMS.get(regime_id)
        if params is None:
            return
        self.max_risk = params["max_risk"]
        self.sl_mult  = params["sl_mult"]
        self.tp_rr    = params["tp_rr"]
        # IT: prob_threshold del preset NON applicato — vedi commento rimozione
        #     set_regime_threshold (SignalGenerator), 2026-06-03. Calibrazione da
        #     rifare post-paper-trading. Il campo resta nel preset _REGIME_RISK_PARAMS
        #     per riferimento storico ma non viene letto.
        # EN: preset prob_threshold NOT applied — see set_regime_threshold removal
        #     note (SignalGenerator), 2026-06-03. Re-calibrate post paper-trading.
        #     The field stays in _REGIME_RISK_PARAMS for historical reference but
        #     is not consumed.

    # IT: Esposizione direzionale = |mean(sides)| ∈ [0,1].
    # EN: Directional exposure = |mean(sides)| ∈ [0,1].
    def _directional_exposure(self, new_side: Side) -> float:
        """
        Miglioramento 10 — Esposizione direzionale cumulata.

        Il Kelly frazionato assume trade indipendenti. Ma segnali consecutivi
        nella stessa direzione (es. 5 LONG di fila) non sono indipendenti:
        il mercato è in trend, e la vera esposizione al rischio è maggiore
        della somma delle posizioni individuali perché tutte perdono insieme
        in un'inversione.

        Questo metodo calcola il "bias direzionale" degli ultimi N trade:
          - Se abbiamo avuto 7 LONG e 0 SHORT su 7 trade → esposizione = 1.0
          - Se abbiamo avuto 4 LONG e 3 SHORT → esposizione = 0.14
          - Esposizione 0 = trade bilanciati (basso rischio di correlazione)

        Il sizing viene ridotto proporzionalmente quando l'esposizione
        supera max_directional_exposure:
          multiplier = 1.0 se exposure ≤ threshold
          multiplier = (1 - exposure) se exposure > threshold
          → riduzione graduale, mai azzeramento completo
        """
        if len(self._recent_sides) < 3:
            return 0.0   # IT: dati insufficienti | EN: insufficient data

        new_val = 1 if new_side == Side.LONG else -1
        recent  = self._recent_sides[-self.corr_window:]

        # IT: 0 = bilanciato (low corr), 1 = uniformly directional (high corr).
        # EN: 0 = balanced (low corr), 1 = uniformly directional (high corr).
        exposure = abs(float(np.mean(recent + [new_val])))
        return exposure

    # IT: Fattore di sconto Kelly per autocorrelazione dei trade returns (Fix 11).
    # EN: Kelly discount factor for trade-return autocorrelation (Fix 11).
    def _autocorr_kelly_factor(self) -> float:
        """
        Fix 11 — Kelly corretto per autocorrelazione dei trade returns.

        Il Kelly classico assume trade indipendenti. Ma trade consecutivi
        nello stesso regime di mercato sono correlati (trend persistence):
        durante un trend rialzista i trade LONG consecutivi hanno rendimenti
        positivamente correlati, gonfiando il Kelly ottimale.

        Correzione (Vince 1992, Thorp 2006):
          f_adj = f / (1 + 2 * sum(rho_k for k=1..K))
        dove rho_k = autocorrelazione dei trade returns al lag k.

        Se la somma delle autocorrelazioni e' positiva (streak di wins/losses),
        il Kelly viene ridotto. Se negativa (mean-reversion), viene aumentato
        (ma capped a 1.0 per sicurezza).

        Returns:
            Fattore moltiplicativo in [0.2, 1.0] da applicare al Kelly.
            1.0 = nessuna correzione (trade indipendenti o dati insufficienti).
        """
        n = len(self._recent_trade_returns)
        if n < 10:
            return 1.0  # IT: dati insufficienti | EN: insufficient data

        returns = np.array(self._recent_trade_returns[-self.autocorr_window:])
        n_used = len(returns)
        if n_used < 10:
            return 1.0

        # IT: K = √N (euristica), cap a 10 per evitare stime rumorose.
        # EN: K = √N (heuristic), capped at 10 to avoid noisy estimates.
        K = min(int(np.sqrt(n_used)), 10, n_used // 3)
        if K < 1:
            return 1.0

        mean_r = returns.mean()
        var_r = returns.var()
        if var_r < 1e-12:
            return 1.0  # IT: var ≈ 0 → no correlazione stimabile | EN: var ≈ 0 → no estimable corr

        # IT: Σ ρ_k per k=1..K (autocorrelazioni).
        # EN: Σ ρ_k for k=1..K (autocorrelations).
        rho_sum = 0.0
        centered = returns - mean_r
        for k in range(1, K + 1):
            cov_k = np.mean(centered[:-k] * centered[k:])
            rho_k = cov_k / var_r
            rho_sum += rho_k

        # IT: f_adj = f / (1 + 2·Σρ_k). Se denom≤0 (mean-rev forte) → no boost.
        # EN: f_adj = f / (1 + 2·Σρ_k). If denom≤0 (strong mean-rev) → no boost.
        denominator = 1.0 + 2.0 * rho_sum
        if denominator <= 0:
            factor = 1.0
        else:
            factor = 1.0 / denominator

        # IT: Clamp [0.2, 1.0]: non azzera, non aumenta.
        # EN: Clamp [0.2, 1.0]: never zeroed, never boosted.
        factor = float(np.clip(factor, 0.2, 1.0))

        if factor < 0.9:
            log.debug(
                f"Fix 11: autocorr Kelly factor={factor:.3f} "
                f"(rho_sum={rho_sum:+.3f}, K={K}, N={n_used})"
            )

        return factor

    # ── Sizing ────────────────────────────────────────────────────────────────
    # IT: Calcola la size della posizione via Kelly continuo dalla distribuzione predetta.
    # EN: Computes position size via continuous Kelly from the predicted distribution.
    def _size(self, dist: DistributionParams, price: float, atr: float,
              side: Side = Side.NONE):
        """
        Kelly continuo basato sulla distribuzione predetta dalla LSTM.

        Miglioramento — Kelly dinamico f* = μ / σ²:
          Il Kelly discreto precedente usava solo prob_up e un TP/RR fisso,
          ignorando σ (volatilità predetta) che la LSTM stima esplicitamente.

          Formula corretta per una distribuzione continua:
            f* = μ / σ²
          dove μ = drift atteso e σ² = varianza predetta.
          Questa è la soluzione esatta del problema di ottimizzazione di Kelly
          per rendimenti normalmente distribuiti (approssimazione valida per
          la t-Student con ν > 4).

          Vantaggi rispetto al Kelly discreto:
          · Usa ENTRAMBI μ e σ — segnali forti con bassa volatilità → size grande
          · Segnali forti con alta volatilità → size ridotta automaticamente
          · Elimina la dipendenza dal TP/RR ratio fisso (2.5)
          · Conviction score mantiene la scalatura proporzionale al segnale

          Frazionamento conservativo: dividiamo per 4 (standard in letteratura)
          per evitare il rischio di rovina con stime imprecise di μ e σ.
        """
        if self.circuit_breaker:
            # IT: Recovery gestito esternamente in _check_circuit_recovery.
            # EN: Recovery handled externally in _check_circuit_recovery.
            return 0.0, 0.0
        eq  = self.portfolio.equity
        slp = max(self.sl_mult * atr / max(price, 1e-9), 1e-4)

        # IT: Kelly continuo f* = μ/σ², cap 0.5 e fraz/4 (Thorp 2006).
        # EN: Continuous Kelly f* = μ/σ², capped at 0.5 and /4 (Thorp 2006).
        mu_abs  = abs(dist.mu)
        sigma2  = max(dist.sigma ** 2, 1e-8)
        kelly_raw   = mu_abs / sigma2
        kelly_base  = min(kelly_raw, 0.5) / 4

        # IT: Floor 0.5% per evitare size irrisorie su μ piccolissimi.
        # EN: 0.5% floor to avoid tiny sizes on very small μ.
        kelly_base = max(kelly_base, 0.005)

        # IT: Scala con conviction ∈ [0,1] | EN: Scale with conviction ∈ [0,1]
        kelly = kelly_base * max(0.0, min(1.0, dist.conviction))

        # IT: Correzione autocorrelazione (Σρ_k > 0 → riduce Kelly).
        # EN: Autocorrelation correction (Σρ_k > 0 → reduces Kelly).
        autocorr_factor = self._autocorr_kelly_factor()
        kelly *= autocorr_factor

        # IT: Riduzione se sbilanciamento direzionale > max_dir_exp.
        # EN: Reduction if directional skew > max_dir_exp.
        dir_exp = self._directional_exposure(side)
        if dir_exp > self.max_dir_exp:
            corr_mult = 1.0 - 0.5 * (dir_exp - self.max_dir_exp) / (1.0 - self.max_dir_exp)
            kelly    *= max(0.3, corr_mult)
            log.debug(f"Esposizione direzionale {dir_exp:.2f} → size ×{corr_mult:.2f}")

        # IT: 4 vincoli: Kelly, max_risk, max_pos%, cash disp. — min vince.
        # EN: 4 constraints: Kelly, max_risk, max_pos%, available cash — min wins.
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
        Calcola lo slippage rate per il trade corrente.

        Modello "sqrt" (Almgren-Chriss 2001, square-root market impact):
          slippage = base_slip * sqrt(trade_size / ADV_1m)

          L'intuizione: l'impatto di mercato cresce con la radice quadrata
          della frazione di volume scambiata. Un ordine pari al 100% del
          volume medio di 1 minuto subisce lo slippage base pieno; un
          ordine pari al 25% del volume subisce metà dello slippage base.

        Modello "fixed":
          slippage = base_slip  (indipendente da size e volume)

        Se adv_1m non è disponibile (= 0), fallback al modello fisso.
        """
        # IT: Almgren-Chriss √-law: slip ∝ √(size/ADV) | EN: Almgren-Chriss √-law: slip ∝ √(size/ADV)
        if self.slip_model == "sqrt" and adv_1m > 0.0 and trade_size_usd > 0.0:
            ratio = trade_size_usd / max(adv_1m, 1.0)
            return self.slip * math.sqrt(ratio)
        return self.slip

    # ── SL/TP ─────────────────────────────────────────────────────────────────
    def _sl_tp(self, side, price, atr, dist):
        """
        SL adattivo a σ predetto + TP dinamico basato sul regime di volatilità.

        Miglioramento — TP dinamico:
          Il TP/RR fisso a 2.5 assumeva che il mercato facesse sempre movimenti
          di 2.5× lo stop loss — indipendente dal regime.
          In alta volatilità il mercato può fare 5× o 10× il SL prima di invertire.
          In bassa volatilità raramente arriva a 2.5×.

          Nuovo approccio: TP = max(SL × 2.0, σ_predetto × price × tp_sigma_mult)
          Il modello stesso indica quanto movimento si aspetta → il TP si adatta.

          tp_sigma_mult=3.0: il TP viene posto a 3σ dalla entry, che corrisponde
          al 99.7% della distribuzione normale — prende il trend ma non aspetta
          l'improbabile. Clampato tra 2.0× e 5.0× il SL per sicurezza.
        """
        # IT: Clamp σ > 0: protegge da bug upstream (NaN, scale negativa).
        # EN: Clamp σ > 0: guards against upstream bugs (NaN, negative scale).
        sigma = max(float(dist.sigma), 1e-6)
        # IT: SL = max(ATR_storico, σ_predetto·price·1.5).
        # EN: SL = max(historical ATR, predicted σ·price·1.5).
        sigma_price  = sigma * price * 1.5
        # IT: σ·price > 5% del prezzo segnala σ ancora in z-space (bug denorm).
        # EN: σ·price > 5% of price signals σ still in z-space (denorm bug).
        if sigma_price > price * 0.05 and not getattr(self, "_warned_scale", False):
            log.warning(
                f"_sl_tp: σ*price*1.5={sigma_price:.0f} > 5%×price={price*0.05:.0f}. "
                f"Probabile σ in z-score non denormalizzato. Vedi PipelineState.denormalize_predictions."
            )
            self._warned_scale = True
        effective_atr= max(atr, sigma_price)
        sl_d         = self.sl_mult * effective_atr
        # IT: Floor 1 bp evita SL=TP=entry quando atr=0 (mercato halt).
        # EN: 1 bp floor prevents SL=TP=entry when atr=0 (market halt).
        sl_d         = max(sl_d, price * 1e-4)
        if atr == 0 and not getattr(self, "_warned_atr_zero", False):
            log.warning(f"_sl_tp: atr=0 (mercato halt o dati sporchi). SL floor a {price*1e-4:.2f}")
            self._warned_atr_zero = True

        # IT: TP = 3σ dalla entry (≈99.7% normale), clampato [2,5]×SL.
        # EN: TP = 3σ from entry (≈99.7% normal), clamped to [2,5]×SL.
        tp_from_sigma = sigma * price * 3.0
        tp_d          = float(np.clip(tp_from_sigma, sl_d * 2.0, sl_d * 5.0))

        if side == Side.LONG:
            return round(price - sl_d, 2), round(price + tp_d, 2)
        return round(price + sl_d, 2), round(price - tp_d, 2)

    # ── Circuit breaker recovery (estratto da _size per fix bug #2) ──────────
    # IT: Riattiva il trading quando il drawdown rientra al 70% della soglia.
    # EN: Re-enables trading once drawdown recovers to 70% of the threshold.
    def _check_circuit_recovery(self) -> None:
        """
        Recovery del circuit breaker: se il DD scende al 70% della soglia,
        riattiva il trading. Es: soglia 15% → riattiva quando DD < 10.5%.

        Estratto da _size per evitare mutazione di stato tra le 2 chiamate
        a _size dentro open_position (bug #2). Va chiamato UNA VOLTA per
        candela prima di valutare sizing/slippage.
        """
        if not self.circuit_breaker:
            return
        # IT: Recovery a 70% della soglia (es. 15% → 10.5%).
        # EN: Recovery at 70% of threshold (e.g. 15% → 10.5%).
        recovery_threshold = self.max_dd_stop * 0.70
        if self.portfolio.drawdown < recovery_threshold:
            self.circuit_breaker = False
            log.info(
                f"✔ CIRCUIT BREAKER DISATTIVATO: DD={self.portfolio.drawdown:.1%} "
                f"< soglia recovery {recovery_threshold:.1%} — trading ripreso"
            )

    # ── Open ──────────────────────────────────────────────────────────────────
    # IT: Apre posizione: 2-step (pre-size → slippage → exec_p → size finale).
    # EN: Opens position: 2-step (pre-size → slippage → exec_p → final size).
    def open_position(self, side, price, candle_idx, atr, dist,
                      adv_1m: float = 0.0) -> Optional[Position]:
        # IT: Guard NaN/Inf espliciti sugli input critici (math.isfinite copre entrambi)
        #     → nessuna apertura su dati corrotti. Sostituisce il vecchio `v != v` criptico.
        # EN: Explicit NaN/Inf guards on critical inputs (math.isfinite covers both)
        #     → never open on corrupted data. Replaces the cryptic `v != v` check.
        _critical = {"price": price, "atr": atr, "mu": dist.mu, "sigma": dist.sigma}
        if not all(math.isfinite(float(v)) for v in _critical.values()):
            log.warning(
                f"open_position: input NaN/Inf rejected ({_critical}) → skip"
            )
            return None
        # IT: Recovery valutato UNA volta sola (evita race tra 2 call a _size).
        # EN: Recovery evaluated ONCE only (avoids race between 2 _size calls).
        self._check_circuit_recovery()
        if self.circuit_breaker or (self.position and self.position.is_open): return None
        # IT: Pre-size per stimare slippage (sqrt-law dipende da trade_size).
        # EN: Pre-size to estimate slippage (sqrt-law depends on trade_size).
        sz_usd_est, _ = self._size(dist, price, atr, side=side)
        slip_rate = self._compute_slippage(price, sz_usd_est, adv_1m)
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
    # IT: Mark-to-market + trailing stop dinamico ATR.
    # EN: Mark-to-market + ATR-based dynamic trailing stop.
    def update_trailing(self, price, atr):
        if not self.position: return
        # IT: MtM critico in live: senza questo il CB non scatta intra-trade.
        # EN: MtM critical in live: without it the CB does not fire intra-trade.
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

        # IT: Trailing eseguito solo se abilitato | EN: Trailing executed only if enabled
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
    # IT: Valuta SL/TP/segnale opposto/max-hold sulla candela e ritorna il motivo.
    # EN: Checks SL/TP/opposite-signal/max-hold on the candle and returns the reason.
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
    # IT: Chiude la posizione: applica slippage/fee, aggiorna equity/DD e circuit breaker.
    # EN: Closes the position: applies slippage/fee, updates equity/DD and circuit breaker.
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
        # IT: Storico direzione/return per stime di autocorrelazione (sliding window).
        # EN: Direction/return history for autocorrelation estimates (sliding window).
        self._recent_sides.append(1 if trade.side == Side.LONG else -1)
        if len(self._recent_sides) > self.corr_window * 2:
            self._recent_sides = self._recent_sides[-self.corr_window:]
        self._recent_trade_returns.append(trade.pnl_pct)
        if len(self._recent_trade_returns) > self.autocorr_window * 2:
            self._recent_trade_returns = self._recent_trade_returns[-self.autocorr_window:]
        return trade

    # ── Metrics ───────────────────────────────────────────────────────────────
    # IT: Calcola le metriche di backtest (Sharpe, Sortino, Calmar, PF, DD, ...).
    # EN: Computes backtest metrics (Sharpe, Sortino, Calmar, PF, DD, ...).
    def metrics(self) -> dict:
        if not self.trades: return {}
        pnl   = np.array([t.net_pnl for t in self.trades])
        pcts  = np.array([t.pnl_pct for t in self.trades])
        holds = np.array([t.hold_candles for t in self.trades])
        wins  = pnl[pnl>0]; losses = pnl[pnl<0]

        eq = np.concatenate([[self.icap], self.icap + np.cumsum(pnl)])
        rm = np.maximum.accumulate(eq); dd = (rm-eq)/rm; max_dd=float(dd.max())

        avg_hold = holds.mean() if len(holds) else 1
        # IT: Annualizzazione su tempo TOTALE in posizione (no assunzione iid).
        # EN: Annualization on TOTAL time in position (no iid assumption).
        # IT: hold_candles è un conteggio di BARRE; self.bars_per_year (default
        #     525_600 = barre 1m/anno) converte le barre in frazione d'anno.
        # EN: hold_candles is a count of BARS; self.bars_per_year (default
        #     525_600 = 1m bars/year) converts bars into a fraction of a year.
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
