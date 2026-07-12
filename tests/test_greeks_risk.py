# IT: Test A7 — risk layer greeks-aware (quantsys/trading/greeks_risk.py):
#     aggregazione greeks netti, cap vega/delta (deny/scale/riduzione sempre
#     ammessa), circuit breaker MtM con isteresi, margin simulation inverse
#     (long=0, IM≥MM, monotonia OTM, perp lineare nel nozionale).
# EN: A7 tests — greeks-aware risk layer: net-greeks aggregation, vega/delta
#     caps (deny/scale/reduction always allowed), MtM circuit breaker with
#     hysteresis, inverse margin simulation (long=0, IM≥MM, OTM monotonicity,
#     perp linear in notional).
import pytest

from quantsys.trading.greeks_risk import (
    GreeksCheck, GreeksLimits, GreeksRiskManager, OptionLegGreeks,
    book_margin_btc, net_greeks, option_margin_btc, perp_margin_btc,
)


def leg(side=+1, opt="call", amount=1.0, strike=64000, under=64000,
        mark=0.02, delta=0.5, gamma=1e-5, vega=25.0, theta=-30.0):
    # IT: factory di comodo per una leg sintetica | EN: convenience synthetic-leg factory
    return OptionLegGreeks(instrument=f"BTC-{opt}", option_type=opt, side=side,
                           amount=amount, strike=strike, underlying=under,
                           mark_btc=mark, delta=delta, gamma=gamma,
                           vega=vega, theta=theta)


class TestNetGreeks:
    def test_short_straddle_vega_negativa(self):
        # IT: short straddle → vega netta NEGATIVA (side-weighted).
        # EN: short straddle → NEGATIVE net vega (side-weighted).
        legs = [leg(side=-1, opt="call", delta=0.5, vega=25.0),
                leg(side=-1, opt="put", delta=-0.5, vega=25.0)]
        g = net_greeks(legs)
        assert g["vega_usd"] == pytest.approx(-50.0)
        assert g["delta"] == pytest.approx(0.0)   # straddle ATM ≈ delta-neutro / delta-neutral

    def test_amount_pesato(self):
        g = net_greeks([leg(side=+1, amount=3.0, vega=10.0)])
        assert g["vega_usd"] == pytest.approx(30.0)

    def test_input_invalidi(self):
        # IT: fail-fast su side/amount/tipo invalidi | EN: fail-fast on bad side/amount/type
        with pytest.raises(ValueError):
            leg(side=0)
        with pytest.raises(ValueError):
            leg(amount=-1.0)
        with pytest.raises(ValueError):
            leg(opt="straddle")


class TestCheckOrder:
    def test_dentro_i_cap_passa_pieno(self):
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=100.0, max_net_delta=5.0))
        chk = rm.check_order([], [leg(side=-1, vega=25.0)])
        assert chk.allowed and chk.scale == 1.0 and chk.reasons == []

    def test_oltre_cap_vega_scala(self):
        # IT: book a −80 vega, ordine −40 → cap 100 ⇒ scale=(−100−(−80))/(−40)=0.5.
        # EN: book at −80 vega, −40 order → 100 cap ⇒ scale 0.5.
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=100.0, max_net_delta=1e9))
        book = [leg(side=-1, amount=1.0, vega=80.0, delta=0.0)]
        chk = rm.check_order(book, [leg(side=-1, amount=1.0, vega=40.0, delta=0.0)])
        assert chk.allowed
        assert chk.scale == pytest.approx(0.5)
        assert any("net_vega_cap" in r for r in chk.reasons)

    def test_book_gia_oltre_cap_ordine_aumentativo_rifiutato(self):
        # IT: esposizione già oltre il cap: un ordine che la AUMENTA ha scale 0.
        # EN: exposure already past the cap: an INCREASING order gets scale 0.
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=100.0, max_net_delta=1e9))
        book = [leg(side=-1, vega=150.0, delta=0.0)]
        chk = rm.check_order(book, [leg(side=-1, vega=10.0, delta=0.0)])
        assert not chk.allowed and chk.scale == 0.0

    def test_ordine_riduttivo_sempre_ammesso(self):
        # IT: anche oltre il cap, chiudere/ridurre l'esposizione è SEMPRE permesso.
        # EN: even past the cap, closing/reducing exposure is ALWAYS allowed.
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=100.0, max_net_delta=1e9))
        book = [leg(side=-1, vega=150.0, delta=0.0)]
        chk = rm.check_order(book, [leg(side=+1, vega=50.0, delta=0.0)])
        assert chk.allowed and chk.scale == 1.0

    def test_cap_delta(self):
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=1e9, max_net_delta=1.0))
        chk = rm.check_order([], [leg(side=+1, amount=4.0, delta=0.5, vega=0.0)])
        assert chk.allowed
        assert chk.scale == pytest.approx(0.5)   # 4×0.5=2.0 → cap 1.0 ⇒ 0.5

    def test_sign_flip_oltre_cap_scalato(self):
        # IT: audit MINOR-4 — book +150 (oltre cap 100), ordine −280: il flip a
        #     −130 NON è "riduttivo" → scala al bordo −100 (s≈0.893), e la
        #     policy è monotona nella size (a −400 atterrava già a −100).
        # EN: MINOR-4 audit — +150 book (past the 100 cap), −280 order: the
        #     flip to −130 is NOT a reduction → scaled to the −100 edge; the
        #     policy is monotone in order size.
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=100.0, max_net_delta=1e9))
        book = [leg(side=+1, vega=150.0, delta=0.0)]
        chk = rm.check_order(book, [leg(side=-1, amount=1.0, vega=280.0, delta=0.0)])
        assert chk.allowed
        assert chk.scale == pytest.approx((-100.0 - 150.0) / -280.0)
        # IT: riduzione stesso segno oltre cap resta ammessa piena (chiudere è lecito).
        # EN: same-sign reduction past the cap stays fully allowed (closing is fine).
        chk2 = rm.check_order(book, [leg(side=-1, amount=1.0, vega=30.0, delta=0.0)])
        assert chk2.allowed and chk2.scale == 1.0

    def test_breaker_aperto_rifiuta_tutto(self):
        rm = GreeksRiskManager(GreeksLimits(cb_max_loss_btc=0.01))
        rm.update_mtm(0.0)
        rm.update_mtm(-0.02)                      # trip
        chk = rm.check_order([], [leg()])
        assert not chk.allowed and chk.reasons == ["circuit_breaker_open"]


class TestCircuitBreaker:
    def test_trip_e_recovery_isteresi(self):
        # IT: trip a DD≥soglia; riarmo SOLO sotto recovery_frac×soglia (isteresi).
        # EN: trip at DD≥threshold; re-arm ONLY below recovery_frac×threshold.
        rm = GreeksRiskManager(GreeksLimits(cb_max_loss_btc=0.10, cb_recovery_frac=0.7))
        assert rm.update_mtm(0.05)                # peak 0.05
        assert not rm.update_mtm(-0.06)           # DD 0.11 ≥ 0.10 → open
        assert rm.circuit_open
        assert not rm.update_mtm(-0.03)           # DD 0.08 ≥ 0.07 → resta aperto / stays open
        assert rm.update_mtm(-0.01)               # DD 0.06 < 0.07 → riarmato / re-armed
        assert not rm.circuit_open

    def test_peak_monotono(self):
        # IT: il peak non scende mai: DD misurato dal massimo storico.
        # EN: peak never decreases: DD measured from the all-time high.
        rm = GreeksRiskManager(GreeksLimits(cb_max_loss_btc=0.10))
        rm.update_mtm(0.20)
        assert not rm.update_mtm(0.05)            # DD 0.15 dal peak 0.20 → trip


class TestMargin:
    def test_long_zero_margine(self):
        m = option_margin_btc(leg(side=+1))
        assert m == {"initial": 0.0, "maintenance": 0.0}

    def test_short_im_maggiore_uguale_mm(self):
        m = option_margin_btc(leg(side=-1, opt="call", strike=64000, under=64000, mark=0.02))
        assert m["initial"] >= m["maintenance"] > 0.0

    def test_monotonia_otm_short_call(self):
        # IT: più OTM → meno IM (fino al floor 0.10) | EN: deeper OTM → lower IM (down to the 0.10 floor)
        atm = option_margin_btc(leg(side=-1, opt="call", strike=64000, under=64000, mark=0.02))
        otm = option_margin_btc(leg(side=-1, opt="call", strike=67200, under=64000, mark=0.01))
        deep = option_margin_btc(leg(side=-1, opt="call", strike=80000, under=64000, mark=0.002))
        assert atm["initial"] > otm["initial"] > deep["initial"]
        # IT: floor: (0.15−0.25)→0.10 + mark | EN: floor kicks in deep OTM
        assert deep["initial"] == pytest.approx(0.10 + 0.002)

    def test_short_put_otm_amount(self):
        # IT: put: OTM quando strike < spot | EN: put: OTM when strike < spot
        m = option_margin_btc(leg(side=-1, opt="put", strike=57600, under=64000, mark=0.01))
        assert m["initial"] == pytest.approx(max(0.15 - 6400 / 64000, 0.10) + 0.01)

    def test_perp_lineare_e_book_somma(self):
        p1 = perp_margin_btc(10_000, 64000.0)
        p2 = perp_margin_btc(20_000, 64000.0)
        assert p2["initial"] == pytest.approx(2 * p1["initial"])
        legs = [leg(side=-1, opt="call"), leg(side=-1, opt="put", delta=-0.5)]
        tot = book_margin_btc(legs, h_usd=10_000, perp_price=64000.0)
        somma = (option_margin_btc(legs[0])["initial"]
                 + option_margin_btc(legs[1])["initial"] + p1["initial"])
        assert tot["initial"] == pytest.approx(somma)
