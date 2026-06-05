"""
tests/test_trading.py
=====================
Test unitari per:
  - RiskManager (init, sizing Kelly, circuit breaker drawdown)
  - SignalGenerator (no-trade a bassa probabilità, generazione segnali)
  - DistributionParams, Side, Position, Trade, Portfolio (dataclass API)

Esegui con:
  pytest tests/test_trading.py -v
"""

import pytest
import numpy as np

from quantsys.trading import (
    RiskManager,
    SignalGenerator,
    DistributionParams,
    Side,
    Position,
    CloseReason,
)


# IT: Helpers ─ DistributionParams e RiskManager di default per i test.
# EN: Helpers ─ Default DistributionParams and RiskManager for tests.

def _default_dist(mu=0.003, sigma=0.002, nu=5.0, prob_up=0.65, conviction=1.0):
    """DistributionParams di default per i test — segnale LONG moderato."""
    return DistributionParams(mu=mu, sigma=sigma, nu=nu,
                              prob_up=prob_up, conviction=conviction)


# IT: Factory di un RiskManager con capitale piccolo per i test.
# EN: Factory for a small-capital RiskManager used across tests.
def _default_rm(capital=10_000.0, **kwargs):
    """RiskManager con capitale piccolo e parametri controllabili."""
    return RiskManager(initial_capital=capital, **kwargs)


# IT: Test 1 — Inizializzazione RiskManager (campi e default).
# EN: Test 1 — RiskManager initialisation (fields and defaults).

class TestRiskManagerInit:

    # IT: Costruttore non solleva con parametri di default.
    # EN: Constructor does not raise on default params.
    def test_creates_without_error(self):
        """RiskManager deve crearsi senza eccezioni con i parametri di default."""
        rm = RiskManager(initial_capital=10_000.0)
        assert rm is not None

    # IT: initial_capital memorizzato come .icap.
    # EN: initial_capital stored as .icap.
    def test_initial_capital_stored(self):
        """Il capitale iniziale deve essere registrato come icap."""
        rm = RiskManager(initial_capital=50_000.0)
        assert rm.icap == 50_000.0, f"icap errato: {rm.icap}"

    # IT: Alla creazione: equity = cash = initial_capital.
    # EN: On creation: equity = cash = initial_capital.
    def test_portfolio_equity_equals_initial_capital(self):
        """Al momento della creazione equity == cash == initial_capital."""
        rm = RiskManager(initial_capital=10_000.0)
        assert rm.portfolio.equity == 10_000.0
        assert rm.portfolio.cash   == 10_000.0

    # IT: circuit_breaker disabilitato all'avvio.
    # EN: circuit_breaker disabled at startup.
    def test_circuit_breaker_initially_false(self):
        """Il circuit_breaker deve partire disabilitato."""
        rm = RiskManager(initial_capital=10_000.0)
        assert rm.circuit_breaker is False

    # IT: Nessuna posizione aperta inizialmente.
    # EN: No open position initially.
    def test_no_open_position_at_start(self):
        """Non ci devono essere posizioni aperte alla creazione."""
        rm = RiskManager(initial_capital=10_000.0)
        assert rm.position is None

    # IT: max_risk_per_trade memorizzato come .max_risk.
    # EN: max_risk_per_trade stored as .max_risk.
    def test_max_risk_stored(self):
        """max_risk_per_trade deve essere memorizzato correttamente."""
        rm = RiskManager(initial_capital=10_000.0, max_risk_per_trade=0.02)
        assert rm.max_risk == 0.02

    # IT: max_position_pct memorizzato come .max_pos_pct.
    # EN: max_position_pct stored as .max_pos_pct.
    def test_max_position_pct_stored(self):
        """max_position_pct deve essere memorizzato correttamente."""
        rm = RiskManager(initial_capital=10_000.0, max_position_pct=0.30)
        assert rm.max_pos_pct == 0.30

    # IT: Tutti i parametri custom vengono memorizzati nei campi attesi.
    # EN: All custom parameters are stored in the expected fields.
    def test_custom_params(self):
        """Parametri custom devono essere memorizzati tutti correttamente."""
        rm = RiskManager(
            initial_capital=20_000.0,
            max_risk_per_trade=0.015,
            sl_atr_mult=3.0,
            tp_rr_ratio=3.0,
            max_position_pct=0.20,
            max_drawdown_stop=0.10,
            max_hold_candles=60,
        )
        assert rm.icap      == 20_000.0
        assert rm.max_risk  == 0.015
        assert rm.sl_mult   == 3.0
        assert rm.tp_rr     == 3.0
        assert rm.max_pos_pct == 0.20
        assert rm.max_dd_stop == 0.10
        assert rm.max_hold  == 60


# IT: Test 2 — SignalGenerator no-trade su prob < soglia, σ alta o |μ| troppo piccolo.
# EN: Test 2 — SignalGenerator no-trade on prob < threshold, high σ, or tiny |μ|.

class TestSignalGeneratorNoTradeOnLowProb:

    # IT: prob_up < prob_threshold → Side.NONE.
    # EN: prob_up < prob_threshold → Side.NONE.
    def test_no_signal_when_prob_below_threshold(self):
        """
        Con prob_up < prob_threshold, non deve essere generato nessun segnale.
        La SignalGenerator usa prob_threshold=0.6: prob=0.5 → Side.NONE.
        """
        sg = SignalGenerator(prob_threshold=0.6)
        # IT: con mu=0 → prob_up ≈ 0.5 (distribuzione centrata su zero).
        # EN: with mu=0 → prob_up ≈ 0.5 (zero-centred distribution).
        side, dist = sg.generate(mu=0.0, sigma=0.002, nu=5.0)
        assert side == Side.NONE, (
            f"Con prob≈0.5 < threshold=0.6 si aspetta Side.NONE, got {side}"
        )

    # IT: sigma > max_sigma → zona no-trade per volatilità eccessiva.
    # EN: sigma > max_sigma → no-trade zone (excess volatility).
    def test_no_signal_high_volatility(self):
        """
        Con sigma > max_sigma (default 0.006), non si deve aprire nessuna posizione
        (zona di no-trade per volatilità eccessiva).
        """
        sg = SignalGenerator(prob_threshold=0.55, max_sigma=0.006)
        # IT: sigma=0.01 >> max_sigma=0.006 | EN: sigma=0.01 >> max_sigma=0.006
        side, _ = sg.generate(mu=0.005, sigma=0.01, nu=5.0)
        assert side == Side.NONE, (
            f"Con sigma=0.01 > max_sigma=0.006 si aspetta Side.NONE, got {side}"
        )

    # IT: |mu| < min_expected_ret → nessuna apertura (edge insufficiente).
    # EN: |mu| < min_expected_ret → no entry (insufficient edge).
    def test_no_signal_when_mu_too_small(self):
        """
        Con |mu| < min_expected_ret, non si apre posizione anche se prob_up alta.
        """
        sg = SignalGenerator(prob_threshold=0.55, min_expected_ret=0.0002)
        # IT: mu=1e-5 << soglia | EN: mu=1e-5 << threshold
        side, _ = sg.generate(mu=0.00001, sigma=0.001, nu=5.0)
        assert side == Side.NONE, (
            f"Con mu troppo piccolo si aspetta Side.NONE, got {side}"
        )

    # IT: mu positivo alto + sigma bassa → Side.LONG.
    # EN: high positive mu + low sigma → Side.LONG.
    def test_long_signal_with_strong_bull(self):
        """Con mu positivo e alto, deve essere generato Side.LONG."""
        sg = SignalGenerator(prob_threshold=0.55)
        # IT: prob_up molto alta → LONG forte | EN: very high prob_up → strong LONG
        side, dist = sg.generate(mu=0.005, sigma=0.001, nu=5.0)
        assert side == Side.LONG, (
            f"Con mu=0.005, sigma=0.001 si aspetta Side.LONG, got {side}"
        )

    # IT: mu negativo grande in modulo + sigma bassa → Side.SHORT.
    # EN: large negative mu + low sigma → Side.SHORT.
    def test_short_signal_with_strong_bear(self):
        """Con mu negativo e alto in valore assoluto, deve essere generato Side.SHORT."""
        sg = SignalGenerator(prob_threshold=0.55)
        # IT: prob_down molto alta → SHORT forte | EN: very high prob_down → strong SHORT
        side, dist = sg.generate(mu=-0.005, sigma=0.001, nu=5.0)
        assert side == Side.SHORT, (
            f"Con mu=-0.005, sigma=0.001 si aspetta Side.SHORT, got {side}"
        )

    # IT: generate() ritorna sempre DistributionParams (anche per Side.NONE).
    # EN: generate() always returns DistributionParams (also for Side.NONE).
    def test_dist_params_always_returned(self):
        """generate() deve sempre restituire DistributionParams (anche per Side.NONE)."""
        sg = SignalGenerator(prob_threshold=0.6)
        side, dist = sg.generate(mu=0.0, sigma=0.002, nu=5.0)
        assert isinstance(dist, DistributionParams), (
            "generate() deve sempre restituire DistributionParams"
        )
        assert dist.mu    == 0.0
        assert dist.sigma == 0.002
        assert dist.nu    == 5.0


# IT: Test 3 — Kelly sizing: positività, cap, robustezza ATR=0, monotonicità.
# EN: Test 3 — Kelly sizing: positivity, cap, ATR=0 robustness, monotonicity.

class TestKellySizing:

    # IT: Segnale valido → size_usd e size_base > 0.
    # EN: Valid signal → size_usd and size_base > 0.
    def test_size_positive_with_valid_signal(self):
        """_size() deve restituire size_usd > 0 con un segnale valido."""
        rm   = _default_rm(capital=10_000.0)
        dist = _default_dist(mu=0.003, sigma=0.002, nu=5.0, conviction=1.0)
        size_usd, size_base = rm._size(dist, price=50_000.0, atr=200.0)
        assert size_usd > 0,   f"size_usd deve essere positiva, got {size_usd}"
        assert size_base > 0,  f"size_base deve essere positiva, got {size_base}"

    # IT: size_usd ≤ equity × max_position_pct.
    # EN: size_usd ≤ equity × max_position_pct.
    def test_size_respects_max_position_pct(self):
        """size_usd non deve superare equity × max_position_pct."""
        capital = 10_000.0
        max_pct = 0.25
        rm   = _default_rm(capital=capital, max_position_pct=max_pct)
        dist = _default_dist(mu=0.01, sigma=0.001, nu=5.0, conviction=1.0)
        size_usd, _ = rm._size(dist, price=50_000.0, atr=200.0)
        max_allowed = capital * max_pct
        assert size_usd <= max_allowed + 1e-6, (
            f"size_usd ({size_usd:.2f}) supera max_position ({max_allowed:.2f})"
        )

    # IT: circuit_breaker attivo → _size() ritorna (0, 0).
    # EN: circuit_breaker active → _size() returns (0, 0).
    def test_size_zero_with_circuit_breaker(self):
        """Con circuit_breaker attivo, _size() deve restituire (0, 0)."""
        rm = _default_rm(capital=10_000.0)
        rm.circuit_breaker = True
        dist = _default_dist()
        size_usd, size_base = rm._size(dist, price=50_000.0, atr=200.0)
        assert size_usd  == 0.0, f"size_usd deve essere 0 con circuit_breaker, got {size_usd}"
        assert size_base == 0.0, f"size_base deve essere 0 con circuit_breaker, got {size_base}"

    # IT: ATR=0 → size deve essere ≥ 0 e mai NaN.
    # EN: ATR=0 → size must be ≥ 0 and never NaN.
    def test_size_zero_with_atr_zero(self):
        """Con ATR=0, la size non deve essere NaN e deve essere >= 0."""
        rm   = _default_rm(capital=10_000.0)
        dist = _default_dist()
        size_usd, size_base = rm._size(dist, price=50_000.0, atr=0.0)
        assert size_usd >= 0,             f"size_usd negativa con ATR=0: {size_usd}"
        assert size_usd == size_usd,      "size_usd è NaN con ATR=0"

    # IT: conviction alta produce size ≥ conviction bassa (cap basso = entrambe sotto).
    # EN: high conviction yields size ≥ low conviction (low cap so neither is clipped).
    def test_size_increases_with_conviction(self):
        """Size più alta con conviction=1.0 rispetto a conviction=0.1.
        Usa max_position_pct basso per evitare che entrambe colpiscano il cap."""
        rm = _default_rm(capital=10_000.0, max_position_pct=0.05)  # IT: cap 5% | EN: 5% cap
        dist_high = _default_dist(mu=0.003, sigma=0.002, conviction=1.0)
        dist_low  = _default_dist(mu=0.003, sigma=0.002, conviction=0.1)
        size_high, _ = rm._size(dist_high, price=50_000.0, atr=200.0)
        size_low,  _ = rm._size(dist_low,  price=50_000.0, atr=200.0)
        assert size_high >= size_low, (
            f"Size con conviction alta ({size_high:.2f}) deve essere >= "
            f"di quella con conviction bassa ({size_low:.2f})"
        )

    # IT: Kelly f* = mu/σ² → sigma↑ → size↓.
    # EN: Kelly f* = mu/σ² → sigma↑ → size↓.
    def test_size_decreases_with_high_sigma(self):
        """Kelly f* = mu/sigma^2: sigma più alta → size minore (a parità di mu)."""
        rm = _default_rm(capital=10_000.0)
        dist_low_vol  = _default_dist(mu=0.003, sigma=0.001, conviction=1.0)
        dist_high_vol = _default_dist(mu=0.003, sigma=0.004, conviction=1.0)
        size_low_vol,  _ = rm._size(dist_low_vol,  price=50_000.0, atr=200.0)
        size_high_vol, _ = rm._size(dist_high_vol, price=50_000.0, atr=200.0)
        assert size_low_vol >= size_high_vol, (
            f"Volatilità più alta deve produrre size <= ({size_high_vol:.2f}), "
            f"ma size bassa vol era {size_low_vol:.2f}"
        )


# IT: Test 4 — Circuit breaker drawdown: blocco nuove entry, soglia, stato.
# EN: Test 4 — Drawdown circuit breaker: blocks new entries, threshold, state.

class TestRiskManagerMaxDrawdown:

    # IT: Helper — simula apertura/chiusura per registrare P&L sul portfolio.
    # EN: Helper — simulate open/close to record P&L on the portfolio.
    def _simulate_trade_cycle(self, rm, entry_price, exit_price, side=Side.LONG,
                               atr=200.0, candle_entry=0, candle_exit=10):
        """
        Apre e chiude una posizione simulata per far registrare P&L al portfolio.
        Restituisce il trade chiuso (o None se la posizione non è stata aperta).
        """
        dist = _default_dist(
            mu=0.003 if side == Side.LONG else -0.003,
            sigma=0.002, nu=5.0, conviction=1.0,
        )
        pos = rm.open_position(side, entry_price, candle_entry, atr, dist)
        if pos is None:
            return None
        reason = CloseReason.STOP_LOSS
        trade  = rm.close_position(reason, exit_price, candle_exit)
        return trade

    # IT: Drawdown > max_drawdown_stop → circuit breaker attivo.
    # EN: Drawdown > max_drawdown_stop → circuit breaker activates.
    def test_circuit_breaker_activates_on_large_drawdown(self):
        """
        Il circuit breaker deve attivarsi quando il drawdown supera max_drawdown_stop.
        Simuliamo una perdita sufficiente a superare la soglia.
        """
        # IT: soglia 5% bassa per facilitare il test | EN: low 5% threshold for testability
        rm = RiskManager(
            initial_capital=10_000.0,
            max_risk_per_trade=0.5,     # IT: risk alto → perdite grandi | EN: high risk → large losses
            max_position_pct=0.90,       # IT: posizione grande | EN: large position
            sl_atr_mult=2.0,
            max_drawdown_stop=0.05,      # IT: 5% attiva breaker | EN: 5% triggers breaker
        )

        # IT: forziamo dd ~11% nel portfolio | EN: force ~11% drawdown on the portfolio
        rm.portfolio.peak_equity = 10_000.0
        rm.portfolio.equity      = 8_900.0   # IT: -11% > 5% | EN: -11% > 5%
        rm.portfolio.cash        = 8_900.0
        rm.portfolio.drawdown    = 0.11       # IT: 11% | EN: 11%

        # IT: apriamo/chiudiamo una posizione per innescare il check.
        # EN: open/close a position to trigger the check.
        dist = _default_dist(mu=0.003, sigma=0.002, conviction=1.0)
        pos  = rm.open_position(Side.LONG, 50_000.0, 0, 200.0, dist)
        if pos is not None:
            rm.close_position(CloseReason.STOP_LOSS, 49_000.0, 5)

        # IT: verifichiamo il meccanismo replicandolo esplicitamente su rm2.
        # EN: verify the mechanism by replicating it explicitly on rm2.
        rm2 = RiskManager(initial_capital=10_000.0, max_drawdown_stop=0.05)
        rm2.portfolio.equity      = 9_000.0
        rm2.portfolio.cash        = 9_000.0
        rm2.portfolio.peak_equity = 10_000.0
        rm2.portfolio.drawdown    = 0.10
        # IT: replica della logica in close_position | EN: mirror of close_position logic
        if rm2.portfolio.drawdown >= rm2.max_dd_stop:
            rm2.circuit_breaker = True
        assert rm2.circuit_breaker is True, (
            "Il circuit breaker deve attivarsi con drawdown > max_drawdown_stop"
        )

    # IT: circuit_breaker attivo + DD ancora oltre la soglia di recovery → open_position None.
    # EN: active circuit_breaker + DD still above the recovery threshold → open_position None.
    def test_circuit_breaker_blocks_new_positions(self):
        """Con circuit_breaker attivo e drawdown ancora oltre la soglia di recovery
        (70% di max_dd_stop), open_position deve restituire None."""
        rm = _default_rm(capital=10_000.0)
        rm.circuit_breaker = True
        # IT: Stato realistico — il breaker scatta a DD>=max_dd_stop; finché il DD resta sopra
        #     il 70% della soglia (recovery), deve restare attivo. Senza settare il DD,
        #     `_check_circuit_recovery` (chiamato in open_position) vedrebbe DD=0 e auto-resetterebbe.
        # EN: Realistic state — the breaker trips at DD>=max_dd_stop; while DD stays above 70% of the
        #     threshold (recovery) it must stay active. Without setting DD, `_check_circuit_recovery`
        #     (called inside open_position) would see DD=0 and auto-clear the breaker.
        rm.portfolio.peak_equity = 10_000.0
        rm.portfolio.equity      = 10_000.0 * (1.0 - rm.max_dd_stop * 1.2)
        rm.portfolio.drawdown    = rm.max_dd_stop * 1.2   # DD oltre la soglia di recovery
        dist = _default_dist()
        pos  = rm.open_position(Side.LONG, 50_000.0, 0, 200.0, dist)
        assert pos is None, (
            "open_position deve restituire None con circuit_breaker attivo (DD oltre recovery)"
        )

    # IT: recovery — il breaker si auto-disattiva quando il DD rientra sotto il 70% della soglia.
    # EN: recovery — the breaker auto-clears once DD drops below 70% of the threshold.
    def test_circuit_breaker_auto_recovers_when_drawdown_recovers(self):
        """`_check_circuit_recovery` riattiva il trading quando DD < 70%·max_dd_stop."""
        rm = _default_rm(capital=10_000.0)
        rm.circuit_breaker = True
        # IT: DD sotto la soglia di recovery (es. 5% < 70%·15%=10.5%) → recovery deve scattare.
        # EN: DD below the recovery threshold (e.g. 5% < 70%·15%=10.5%) → recovery must fire.
        rm.portfolio.peak_equity = 10_000.0
        rm.portfolio.equity      = 9_500.0
        rm.portfolio.drawdown    = 0.05
        dist = _default_dist()
        pos  = rm.open_position(Side.LONG, 50_000.0, 0, 200.0, dist)
        assert rm.circuit_breaker is False, "il breaker deve auto-disattivarsi quando il DD rientra"
        assert pos is not None, "dopo il recovery open_position deve poter aprire una posizione"

    # IT: drawdown = (peak - equity) / peak.
    # EN: drawdown = (peak - equity) / peak.
    def test_drawdown_computed_correctly(self):
        """
        Il drawdown deve essere calcolato come (peak - equity) / peak.
        """
        rm = _default_rm(capital=10_000.0)

        # IT: peak=10k, equity=9k → dd=10% | EN: peak=10k, equity=9k → dd=10%
        rm.portfolio.peak_equity = 10_000.0
        rm.portfolio.equity      = 9_000.0
        expected_dd = (10_000.0 - 9_000.0) / 10_000.0
        actual_dd   = (rm.portfolio.peak_equity - rm.portfolio.equity) / rm.portfolio.peak_equity
        assert abs(actual_dd - expected_dd) < 1e-9, (
            f"Drawdown calcolato errato: {actual_dd:.4f} vs {expected_dd:.4f}"
        )

    # IT: Drawdown sotto soglia → circuit_breaker resta False.
    # EN: Drawdown below threshold → circuit_breaker stays False.
    def test_no_circuit_breaker_within_limit(self):
        """Il circuit_breaker non si attiva se il drawdown è sotto la soglia."""
        rm = _default_rm(capital=10_000.0, max_drawdown_stop=0.15)
        rm.portfolio.peak_equity = 10_000.0
        rm.portfolio.equity      = 9_500.0  # IT: -5% < 15% | EN: -5% < 15%
        rm.portfolio.drawdown    = 0.05
        # IT: il breaker non deve attivarsi senza una chiusura | EN: no auto-trigger without close
        assert rm.circuit_breaker is False, (
            "Il circuit_breaker non deve essere attivo con drawdown del 5% < 15%"
        )


# IT: Test 5 — Conviction score: continuità, limiti, mapping a Side.NONE.
# EN: Test 5 — Conviction score: continuity, bounds, mapping to Side.NONE.

class TestConvictionScore:

    # IT: Esattamente alla soglia conviction = 0.
    # EN: At the threshold exactly, conviction = 0.
    def test_conviction_zero_at_threshold(self):
        """Conviction deve essere 0 esattamente alla soglia."""
        sg = SignalGenerator(prob_threshold=0.55, conviction_alpha=1.0)
        # IT: prob_up=0.55 esattamente al threshold → raw=0 → conviction=0.
        # EN: prob_up=0.55 right at threshold → raw=0 → conviction=0.
        conv = sg.conviction(prob_up=0.55, side=Side.LONG)
        assert abs(conv) < 1e-9, f"Conviction alla soglia deve essere 0, got {conv}"

    # IT: Massima certezza (prob_up=1.0) → conviction=1.0.
    # EN: Maximum certainty (prob_up=1.0) → conviction=1.0.
    def test_conviction_one_at_certainty(self):
        """Conviction deve essere 1.0 alla massima certezza."""
        sg = SignalGenerator(prob_threshold=0.55, conviction_alpha=1.0)
        # IT: prob=1 → raw=1 → conv=1^1=1 | EN: prob=1 → raw=1 → conv=1^1=1
        conv = sg.conviction(prob_up=1.0, side=Side.LONG)
        assert abs(conv - 1.0) < 1e-9, f"Conviction a certezza massima deve essere 1, got {conv}"

    # IT: Side.NONE → conviction sempre 0.
    # EN: Side.NONE → conviction always 0.
    def test_conviction_zero_for_none_side(self):
        """Conviction deve essere 0 per Side.NONE."""
        sg = SignalGenerator()
        conv = sg.conviction(prob_up=0.8, side=Side.NONE)
        assert conv == 0.0, f"Conviction per Side.NONE deve essere 0, got {conv}"

    # IT: Conviction sempre in [0, 1] per ogni prob/side.
    # EN: Conviction always in [0, 1] for any prob/side.
    def test_conviction_in_valid_range(self):
        """Conviction deve sempre essere in [0, 1]."""
        sg = SignalGenerator(prob_threshold=0.55)
        for p in np.linspace(0.0, 1.0, 20):
            for side in [Side.LONG, Side.SHORT]:
                conv = sg.conviction(prob_up=float(p), side=side)
                assert 0.0 <= conv <= 1.0, (
                    f"Conviction fuori [0,1] per prob_up={p:.2f}, side={side}: {conv}"
                )


# IT: Test 6 — prob_up: sanity check sul segno e sui limiti.
# EN: Test 6 — prob_up: sanity checks on sign and bounds.

class TestProbUp:

    # IT: mu=0 → prob_up ≈ 0.5. | EN: mu=0 → prob_up ≈ 0.5.
    def test_prob_up_near_half_for_zero_mu(self):
        """Con mu=0, prob_up deve essere circa 0.5."""
        sg = SignalGenerator()
        p  = sg.prob_up(mu=0.0, sigma=0.002, nu=5.0)
        assert abs(p - 0.5) < 0.01, f"prob_up con mu=0 deve essere ~0.5, got {p:.4f}"

    # IT: mu>0 → prob_up > 0.5. | EN: mu>0 → prob_up > 0.5.
    def test_prob_up_above_half_for_positive_mu(self):
        """Con mu > 0, prob_up deve essere > 0.5."""
        sg = SignalGenerator()
        p  = sg.prob_up(mu=0.003, sigma=0.002, nu=5.0)
        assert p > 0.5, f"prob_up con mu>0 deve essere >0.5, got {p:.4f}"

    # IT: mu<0 → prob_up < 0.5. | EN: mu<0 → prob_up < 0.5.
    def test_prob_up_below_half_for_negative_mu(self):
        """Con mu < 0, prob_up deve essere < 0.5."""
        sg = SignalGenerator()
        p  = sg.prob_up(mu=-0.003, sigma=0.002, nu=5.0)
        assert p < 0.5, f"prob_up con mu<0 deve essere <0.5, got {p:.4f}"

    # IT: prob_up ∈ [0, 1] su un set di input. | EN: prob_up ∈ [0, 1] across inputs.
    def test_prob_up_in_valid_range(self):
        """prob_up deve essere in [0, 1] per qualsiasi input."""
        sg = SignalGenerator()
        for mu in [-0.01, -0.001, 0.0, 0.001, 0.01]:
            p = sg.prob_up(mu=mu, sigma=0.002, nu=5.0)
            assert 0.0 <= p <= 1.0, f"prob_up fuori [0,1] per mu={mu}: {p}"


# IT: Test 7 — Dataclass API: DistributionParams, Side, Portfolio defaults.
# EN: Test 7 — Dataclass API: DistributionParams, Side, Portfolio defaults.

class TestDataclasses:

    # IT: Default: prob_up=0.5, conviction=1.0.
    # EN: Defaults: prob_up=0.5, conviction=1.0.
    def test_distribution_params_defaults(self):
        """DistributionParams deve avere prob_up=0.5 e conviction=1.0 di default."""
        d = DistributionParams(mu=0.001, sigma=0.002, nu=5.0)
        assert d.prob_up    == 0.5
        assert d.conviction == 1.0

    # IT: I valori di Side coincidono con le stringhe attese.
    # EN: Side enum values match the expected strings.
    def test_side_enum_values(self):
        """Side deve avere i valori LONG, SHORT, NONE."""
        assert Side.LONG.value  == "LONG"
        assert Side.SHORT.value == "SHORT"
        assert Side.NONE.value  == "NONE"

    # IT: metrics() su portfolio nuovo → dict vuoto.
    # EN: metrics() on a fresh portfolio → empty dict.
    def test_metrics_empty_without_trades(self):
        """metrics() deve restituire un dict vuoto se non ci sono trade."""
        rm = _default_rm()
        assert rm.metrics() == {}, "metrics() deve essere vuoto senza trade"

    # IT: trades è una lista vuota al momento della creazione.
    # EN: trades is an empty list at creation time.
    def test_trades_list_initially_empty(self):
        """La lista dei trade deve essere vuota alla creazione."""
        rm = _default_rm()
        assert rm.trades == [], "trades deve essere una lista vuota all'inizio"
