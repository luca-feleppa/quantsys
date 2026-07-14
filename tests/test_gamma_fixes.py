# IT: Test A11-A14 (funzioni gamma, ROADMAP_VOL_BOOK 2026-07-14) — tutti offline:
#     ww_band (Whalley–Wilmott: monotonia in Γ, clip di sicurezza), pin_close_due
#     (pin region: tempo E banda, mai a expiry passata), vega_sized_amount
#     (round/floor/cap, degeneri → 0), gamma cap opzionale in GreeksRiskManager
#     (None = comportamento storico invariato), maybe_pin_close end-to-end su
#     FakeDB + tmp_path (nessun file production toccato), attribution per
#     intervallo (identità Taylor sui termini puri).
# EN: A11-A14 tests (gamma functions, ROADMAP_VOL_BOOK 2026-07-14) — all offline:
#     ww_band (Whalley–Wilmott: Γ monotonicity, safety clip), pin_close_due
#     (pin region: time AND band, never past expiry), vega_sized_amount
#     (round/floor/cap, degenerate → 0), optional gamma cap in GreeksRiskManager
#     (None = legacy behavior unchanged), end-to-end maybe_pin_close on FakeDB +
#     tmp_path (no production file touched), per-interval attribution
#     (Taylor identity on pure terms).
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from quantsys.trading.greeks_risk import GreeksLimits, GreeksRiskManager, OptionLegGreeks

ROOT = Path(__file__).resolve().parents[1]


def _import_from_path(name: str, path: Path):
    # IT: import da path (nomi con cifre iniziali / fuori package).
    # EN: path-based import (digit-leading names / outside packages).
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vp():
    return _import_from_path("volpaper_04b_gamma", ROOT / "scripts" / "04b_vol_paper.py")


@pytest.fixture(scope="module")
def attr():
    return _import_from_path("pnl_attribution",
                             ROOT / "scripts" / "vol" / "pnl_attribution.py")


# ──────────────────────────── A12 — ww_band ────────────────────────────
class TestWWBand:
    def test_monotona_in_gamma(self, vp):
        # IT: banda ∝ Γ^(2/3): più gamma → banda più larga (meno churn vicino ATM).
        # EN: band ∝ Γ^(2/3): more gamma → wider band (less near-ATM churn).
        b_lo = vp.ww_band(5e-4, 64000.0, 1e-4, 1e-3, band_ref=0.20)
        b_hi = vp.ww_band(5e-4, 64000.0, 4e-4, 1e-3, band_ref=0.20)
        assert b_hi > b_lo

    def test_clip_inferiore_e_superiore(self, vp):
        # IT: greek assurdo → mai sotto band/4 né sopra 4·band (guard MINOR-2-like).
        # EN: absurd greek → never below band/4 nor above 4·band (MINOR-2-like guard).
        assert vp.ww_band(5e-4, 64000.0, 0.0, 1e-3, band_ref=0.20) == pytest.approx(0.05)
        assert vp.ww_band(5e-4, 64000.0, 10.0, 1e-3, band_ref=0.20) == pytest.approx(0.80)

    def test_formula_asintotica(self, vp):
        # IT: valore interno al clip = (3·k·S·Γ²/2λ)^(1/3) esatto.
        # EN: value inside the clip = exact (3·k·S·Γ²/2λ)^(1/3).
        k, S, g, lam = 5e-4, 64000.0, 3e-4, 1e-3
        expected = (1.5 * k * S * g ** 2 / lam) ** (1 / 3)
        assert 0.05 < expected < 0.80
        assert vp.ww_band(k, S, g, lam, 0.20) == pytest.approx(expected)


# ──────────────────────────── A13a — pin_close_due ────────────────────────────
class TestPinCloseDue:
    NOW = 1_700_000_000_000.0

    def _due(self, vp, strike=64000.0, s=64100.0, hours_left=2.0,
             max_h=3.0, band=0.01):
        return vp.pin_close_due(strike, s, self.NOW + hours_left * 3.6e6,
                                self.NOW, max_h, band)

    def test_dentro_pin_region(self, vp):
        assert self._due(vp) is True

    def test_fuori_banda_spot(self, vp):
        # IT: |S−K|/S ≈ 3% > band 1% → non è pin, si tiene la posizione.
        # EN: |S−K|/S ≈ 3% > 1% band → not pinned, keep the position.
        assert self._due(vp, s=66000.0) is False

    def test_troppo_presto(self, vp):
        assert self._due(vp, hours_left=10.0) is False

    def test_expiry_passata_compete_a_settle(self, vp):
        # IT: a expiry passata il payoff è congelato → giurisdizione di maybe_settle.
        # EN: past expiry the payoff is frozen → maybe_settle's jurisdiction.
        assert self._due(vp, hours_left=-1.0) is False


# ──────────────────────────── A14 — vega_sized_amount ────────────────────────────
class TestVegaSizedAmount:
    def test_round_a_step_01(self, vp):
        # IT: 100 USD target / 42 USD·vol-pt struttura → 2.4 contratti (round .1).
        # EN: 100 USD target / 42 USD·vol-pt structure → 2.4 contracts (.1 round).
        assert vp.vega_sized_amount(42.0, 100.0, 10.0) == pytest.approx(2.4)

    def test_floor_e_cap(self, vp):
        assert vp.vega_sized_amount(1000.0, 1.0, 10.0) == pytest.approx(0.1)
        assert vp.vega_sized_amount(1.0, 1000.0, 10.0) == pytest.approx(10.0)

    def test_input_degeneri(self, vp):
        # IT: vega nulla/negativa/NaN o target ≤0 → 0.0 (fallback size fissa nel caller).
        # EN: zero/negative/NaN vega or target ≤0 → 0.0 (fixed-size fallback in caller).
        assert vp.vega_sized_amount(0.0, 100.0, 10.0) == 0.0
        assert vp.vega_sized_amount(-5.0, 100.0, 10.0) == 0.0
        assert vp.vega_sized_amount(float("nan"), 100.0, 10.0) == 0.0
        assert vp.vega_sized_amount(50.0, 0.0, 10.0) == 0.0


# ──────────────────────────── A13b — gamma cap ────────────────────────────
def _leg(side=+1, gamma=1e-4, vega=25.0, delta=0.0, amount=1.0):
    return OptionLegGreeks(instrument="BTC-X", option_type="call", side=side,
                           amount=amount, strike=64000, underlying=64000,
                           mark_btc=0.02, delta=delta, gamma=gamma, vega=vega)


class TestGammaCap:
    def test_default_none_nessun_effetto(self):
        # IT: max_net_gamma=None (default) = comportamento storico bit-invariato.
        # EN: max_net_gamma=None (default) = bit-unchanged legacy behavior.
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=1e9, max_net_delta=1e9))
        chk = rm.check_order([], [_leg(gamma=100.0)])
        assert chk.allowed and chk.scale == 1.0 and chk.reasons == []

    def test_cap_scala_l_ordine(self):
        # IT: book Γ=8e-4, ordine +8e-4, cap 1e-3 → scale = (1e-3−8e-4)/8e-4 = 0.25.
        # EN: book Γ=8e-4, +8e-4 order, 1e-3 cap → scale = 0.25.
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=1e9, max_net_delta=1e9,
                                            max_net_gamma=1e-3))
        chk = rm.check_order([_leg(gamma=8e-4)], [_leg(gamma=8e-4)])
        assert chk.scale == pytest.approx(0.25)
        assert any("net_gamma_cap" in r for r in chk.reasons)

    def test_riduzione_sempre_ammessa(self):
        # IT: chiudere convessità oltre cap è sempre permesso (policy _cap_scale b).
        # EN: closing past-cap convexity is always allowed (_cap_scale policy b).
        rm = GreeksRiskManager(GreeksLimits(max_net_vega_usd=1e9, max_net_delta=1e9,
                                            max_net_gamma=1e-4))
        chk = rm.check_order([_leg(side=-1, gamma=1e-3)], [_leg(side=+1, gamma=5e-4)])
        assert chk.allowed and chk.scale == 1.0


# ──────────────────────────── A13a — maybe_pin_close e2e ────────────────────────────
class FakePinDB:
    # IT: doppio Deribit per il pin-close: index + mark + ordini in memoria.
    # EN: Deribit double for pin close: in-memory index + marks + orders.
    def __init__(self, index, call_mark, put_mark):
        self._index, self._marks = index, {"C": call_mark, "P": put_mark}
        self.orders = []

    def index_price(self):
        return self._index

    def mark_price(self, instrument):
        return self._marks[instrument]

    def market_order(self, instrument, side, amount):
        self.orders.append({"instrument": instrument, "side": side, "amount": amount})
        return self._marks[instrument]


class TestMaybePinClose:
    def _pos(self, expiry_ms, side=+1):
        return {"entry_ts": "2026-07-14 10:00:00+00:00", "side": side,
                "executed": False, "expiry_ms": expiry_ms, "strike": 64000.0,
                "call": "C", "put": "P", "amount": 1.0,
                "prem_call": 0.0085, "prem_put": 0.0065, "fee_btc": 0.0006,
                "edge": 0.4, "rv_pred": 6.7e-4, "var_iv": 4.2e-4}

    def test_chiude_e_registra(self, vp, tmp_path, monkeypatch):
        monkeypatch.setattr(vp, "POSITION_PATH", tmp_path / "position.json")
        monkeypatch.setattr(vp, "TRADES_PATH", tmp_path / "trades.jsonl")
        expiry = time.time() * 1000 + 2 * 3.6e6            # 2h alla scadenza / to expiry
        pos = self._pos(expiry)
        vp.save_position(pos)
        db = FakePinDB(index=64100.0, call_mark=0.003, put_mark=0.002)
        closed = vp.maybe_pin_close(db, pos, {"hours": 3.0, "band": 0.01}, execute=False)
        assert closed is True
        assert not (tmp_path / "position.json").exists()   # posizione rimossa / removed
        rec = json.loads((tmp_path / "trades.jsonl").read_text().strip())
        assert rec["exit_mode"] == "pin_close"
        # IT: long: PnL = (0.005 − 0.015) − fee entry − fee exit < 0 (theta burn).
        # EN: long: PnL = (0.005 − 0.015) − entry fee − exit fee < 0 (theta burn).
        assert rec["pnl_btc"] < -0.009

    def test_fuori_regione_non_chiude(self, vp, tmp_path, monkeypatch):
        monkeypatch.setattr(vp, "POSITION_PATH", tmp_path / "position.json")
        monkeypatch.setattr(vp, "TRADES_PATH", tmp_path / "trades.jsonl")
        expiry = time.time() * 1000 + 20 * 3.6e6           # 20h > max 3h
        pos = self._pos(expiry)
        vp.save_position(pos)
        db = FakePinDB(index=64100.0, call_mark=0.003, put_mark=0.002)
        assert vp.maybe_pin_close(db, pos, {"hours": 3.0, "band": 0.01},
                                  execute=False) is False
        assert (tmp_path / "position.json").exists()

    def test_errore_lascia_posizione_intatta(self, vp, tmp_path, monkeypatch):
        # IT: fail-soft: db che esplode → False, posizione e trades intatti.
        # EN: fail-soft: exploding db → False, position and trades untouched.
        monkeypatch.setattr(vp, "POSITION_PATH", tmp_path / "position.json")
        monkeypatch.setattr(vp, "TRADES_PATH", tmp_path / "trades.jsonl")
        pos = self._pos(time.time() * 1000 + 2 * 3.6e6)
        vp.save_position(pos)

        class BoomDB:
            def index_price(self):
                raise RuntimeError("rete giù / network down")

        assert vp.maybe_pin_close(BoomDB(), pos, {"hours": 3.0, "band": 0.01},
                                  execute=False) is False
        assert (tmp_path / "position.json").exists()
        assert not (tmp_path / "trades.jsonl").exists()


# ──────────────────────────── A11 — interval_attribution ────────────────────────────
class TestIntervalAttribution:
    def _leg(self, mark, S, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, iv=45.0):
        return {"mark": mark, "underlying": S, "delta": delta, "gamma": gamma,
                "vega": vega, "theta": theta, "mark_iv": iv}

    def test_termine_gamma_puro(self, attr):
        # IT: solo Γ: componente = ½·Γ·ΔS², il resto finisce nel residuo.
        # EN: Γ only: component = ½·Γ·ΔS², everything else lands in the residual.
        l0 = self._leg(0.010, 64000.0, gamma=2e-4)
        l1 = self._leg(0.011, 64500.0, gamma=2e-4)
        comp = attr.interval_attribution(l0, l1, dt_days=1 / 24)
        assert comp["gamma_usd"] == pytest.approx(0.5 * 2e-4 * 500.0 ** 2)
        assert comp["delta_usd"] == 0.0 and comp["theta_usd"] == 0.0

    def test_identita_taylor(self, attr):
        # IT: dv = Σ componenti + residuo, per costruzione (nessuna perdita di massa).
        # EN: dv = Σ components + residual, by construction (no mass loss).
        l0 = self._leg(0.010, 64000.0, delta=0.5, gamma=2e-4, vega=25.0, theta=-30.0)
        l1 = self._leg(0.012, 64800.0, delta=0.6, gamma=1e-4, vega=20.0, theta=-40.0, iv=47.0)
        c = attr.interval_attribution(l0, l1, dt_days=1 / 24)
        assert c["dv_usd"] == pytest.approx(c["delta_usd"] + c["gamma_usd"]
                                            + c["vega_usd"] + c["theta_usd"]
                                            + c["residual_usd"])
