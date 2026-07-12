# IT: Test V2 (B2/A1) — leg delta-hedge perp di scripts/04b_vol_paper.py, offline:
#     FakeDB (nessun contatto col testnet), path di stato/ledger/posizione
#     monkeypatchati su tmp_path (nessun contatto coi file production).
#     Copre: no-op da flat, open oltre banda con nozionale corretto
#     (H* = −side·δ_conv·S, arrotondato al contratto da 10 USD), isteresi
#     (dentro banda → zero churn), rebalance oltre banda, flatten al settlement,
#     e convenzione δ raw vs adj.
# EN: V2 (B2/A1) tests — perp delta-hedge leg of scripts/04b_vol_paper.py, offline:
#     FakeDB (no testnet contact), state/ledger/position paths monkeypatched to
#     tmp_path (no production files touched). Covers: flat no-op, beyond-band
#     open with correct notional (H* = −side·δ_conv·S, rounded to the 10 USD
#     contract), hysteresis (inside band → zero churn), beyond-band rebalance,
#     flatten at settlement, and raw-vs-adj δ convention.
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vp():
    # IT: importa 04b come modulo (il nome inizia per cifra → importlib da path).
    # EN: import 04b as a module (name starts with a digit → importlib from path).
    spec = importlib.util.spec_from_file_location(
        "volpaper_04b", ROOT / "scripts" / "04b_vol_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["volpaper_04b"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeDB:
    # IT: doppio del client Deribit: ticker/ordini in memoria, zero rete.
    # EN: Deribit client double: in-memory ticker/orders, zero network.
    def __init__(self, call_delta, put_delta, call_mark, put_mark, S, perp_mark):
        self._legs = {
            "C": {"best_bid_price": call_mark * 0.9, "best_ask_price": call_mark * 1.1,
                  "best_bid_amount": 1, "best_ask_amount": 1, "mark_price": call_mark,
                  "mark_iv": 45.0, "bid_iv": 44.0, "ask_iv": 46.0,
                  "underlying_price": S,
                  "greeks": {"delta": call_delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}},
            "P": {"best_bid_price": put_mark * 0.9, "best_ask_price": put_mark * 1.1,
                  "best_bid_amount": 1, "best_ask_amount": 1, "mark_price": put_mark,
                  "mark_iv": 45.0, "bid_iv": 44.0, "ask_iv": 46.0,
                  "underlying_price": S,
                  "greeks": {"delta": put_delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}},
        }
        self.perp_mark = perp_mark
        self.orders = []

    def ticker(self, instrument):
        if instrument == "BTC-PERPETUAL":
            return {"mark_price": self.perp_mark}
        return self._legs[instrument]

    def market_order(self, instrument, side, amount):
        self.orders.append({"instrument": instrument, "side": side, "amount": amount})
        return self.perp_mark


@pytest.fixture()
def paths(vp, tmp_path, monkeypatch):
    # IT: redirige TUTTI i file di stato su tmp — i production restano intoccati.
    # EN: redirect ALL state files to tmp — production files stay untouched.
    pos = tmp_path / "position.json"
    st = tmp_path / "hedge_state.json"
    led = tmp_path / "hedge_ledger.jsonl"
    monkeypatch.setattr(vp, "POSITION_PATH", pos)
    monkeypatch.setattr(vp, "HEDGE_STATE_PATH", st)
    monkeypatch.setattr(vp, "HEDGE_LEDGER_PATH", led)
    return {"pos": pos, "state": st, "ledger": led}


def write_position(paths, side=+1, strike=64000.0, expiry_ms=1_800_000_000_000):
    paths["pos"].write_text(json.dumps({
        "side": side, "strike": strike, "expiry_ms": expiry_ms,
        "call": "C", "put": "P"}), encoding="utf-8")


HCFG = {"band": 0.20, "conv": "adj", "fee": 5e-4}


def ledger_rows(paths):
    if not paths["ledger"].exists():
        return []
    return [json.loads(l) for l in
            paths["ledger"].read_text(encoding="utf-8").strip().splitlines()]


class TestMaybeHedge:
    def test_flat_senza_stato_e_noop(self, vp, paths):
        # IT: book flat, nessun hedge in essere → nessun file creato, zero ordini.
        # EN: flat book, no live hedge → no file created, zero orders.
        db = FakeDB(0.5, -0.5, 0.02, 0.02, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)
        assert not paths["state"].exists() and not paths["ledger"].exists()
        assert db.orders == []

    def test_open_oltre_banda_nozionale_corretto(self, vp, paths):
        # IT: LONG straddle put-ITM: δ_raw=−0.87, m=0.03 → δ_adj=−0.90 →
        #     H* = −(+1)·(−0.90)·64000 = +57600 USD (buy, multiplo di 10).
        # EN: LONG put-ITM straddle: δ_adj=−0.90 → H* = +57,600 USD (buy).
        write_position(paths, side=+1)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)
        st = json.loads(paths["state"].read_text(encoding="utf-8"))
        assert st["h_usd"] == pytest.approx(57600.0)
        assert st["h_usd"] % 10 == 0
        assert db.orders == [{"instrument": "BTC-PERPETUAL", "side": "buy",
                              "amount": pytest.approx(57600.0)}]
        rows = ledger_rows(paths)
        assert len(rows) == 1 and rows[0]["event"] == "open"
        assert rows[0]["book_delta_pre"] == pytest.approx(-0.90)

    def test_isteresi_dentro_banda_zero_churn(self, vp, paths):
        # IT: hedge in essere e drift piccolo del delta → dentro la banda → NIENTE
        #     ordini (la lezione anti-churn del dry-run 07-10).
        # EN: live hedge and small delta drift → inside the band → NO orders.
        write_position(paths, side=+1)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)          # open a −0.90
        db.orders.clear()
        # IT: drift a δ_adj=−1.03 → book_delta = −1.03+0.90 = −0.13, |·|<0.20.
        # EN: drift to δ_adj=−1.03 → book_delta −0.13, inside the 0.20 band.
        db2 = FakeDB(0.05, -1.05, 0.003, 0.027, 64000.0, 64000.0)
        vp.maybe_hedge(db2, HCFG, execute=True)
        assert db2.orders == []
        assert len(ledger_rows(paths)) == 1             # solo l'open / open only

    def test_rebalance_oltre_banda(self, vp, paths):
        write_position(paths, side=+1)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)          # H=+57600
        # IT: rally S 64k→80k: h_btc = 57600/80000 = 0.72; put a δ=−1.0 →
        #     δ_adj = −0.98−0.051 = −1.031 → book_delta = −0.311 ≥ 0.20 →
        #     target H* = 1.031·80000 = 82480 → dh = +24880 (multiplo di 10).
        #     (δ fisici: |δ_raw| ≤ 1 — il bound MINOR-2 resta rispettato.)
        # EN: S rally 64k→80k: h_btc shrinks to 0.72; δ_adj = −1.031 →
        #     book_delta −0.311 ≥ 0.20 → rebalance dh = +24,880 (10-multiple).
        db2 = FakeDB(0.02, -1.0, 0.001, 0.05, 80000.0, 80000.0)
        vp.maybe_hedge(db2, HCFG, execute=True)
        st = json.loads(paths["state"].read_text(encoding="utf-8"))
        assert st["h_usd"] == pytest.approx(82480.0)
        rows = ledger_rows(paths)
        assert [r["event"] for r in rows] == ["open", "rebalance"]
        assert rows[1]["dh_usd"] % 10 == 0

    def test_flatten_al_settlement(self, vp, paths):
        write_position(paths, side=+1)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)
        paths["pos"].unlink()                            # settlement: book flat
        db.orders.clear()
        vp.maybe_hedge(db, HCFG, execute=True)
        assert not paths["state"].exists()
        assert db.orders == [{"instrument": "BTC-PERPETUAL", "side": "sell",
                              "amount": pytest.approx(57600.0)}]
        rows = ledger_rows(paths)
        assert rows[-1]["event"] == "flatten" and rows[-1]["reason"] == "settled"

    def test_convenzione_raw_vs_adj(self, vp, paths):
        # IT: con conv=raw il nozionale usa δ_raw=−0.87 (non −0.90): H*=+55680.
        # EN: with conv=raw the notional uses δ_raw=−0.87 (not −0.90): H*=+55,680.
        write_position(paths, side=+1)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, {**HCFG, "conv": "raw"}, execute=True)
        st = json.loads(paths["state"].read_text(encoding="utf-8"))
        assert st["h_usd"] == pytest.approx(-(+1) * (-0.87) * 64000.0)

    def test_short_side_segno_opposto(self, vp, paths):
        # IT: SHORT straddle (side=−1): δ_book = −δ_adj = +0.90 → hedge SELL.
        # EN: SHORT straddle (side=−1): book δ = +0.90 → SELL hedge.
        write_position(paths, side=-1)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)
        st = json.loads(paths["state"].read_text(encoding="utf-8"))
        assert st["h_usd"] == pytest.approx(-57600.0)
        assert db.orders[0]["side"] == "sell"

    def test_greeks_mancanti_skip_failsoft(self, vp, paths):
        # IT: greeks assenti su una leg → skip del ribilancio, MAI un raise.
        # EN: missing greeks on one leg → rebalance skipped, NEVER a raise.
        write_position(paths, side=+1)
        db = FakeDB(0.08, None, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)
        assert not paths["state"].exists() and db.orders == []

    def test_flatten_a_expiry_anche_senza_delivery(self, vp, paths):
        # IT: audit MAJOR-1 — expiry passata ma position.json ancora presente
        #     (delivery price non pubblicato): l'hedge va chiuso comunque, il
        #     delta post-expiry è NUDO e contaminerebbe il gate hedged-vs-unhedged.
        # EN: MAJOR-1 audit — expiry passed but position.json still present
        #     (delivery price unpublished): the hedge must be closed anyway.
        import time as _time
        write_position(paths, side=+1,
                       expiry_ms=int(_time.time() * 1000) + 3_600_000)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)          # open a −0.90
        # IT: la stessa struttura ora risulta scaduta (expiry nel passato).
        # EN: the same structure is now expired (past expiry).
        write_position(paths, side=+1,
                       expiry_ms=int(_time.time() * 1000) - 1000)
        db.orders.clear()
        vp.maybe_hedge(db, HCFG, execute=True)
        assert not paths["state"].exists()
        assert db.orders == [{"instrument": "BTC-PERPETUAL", "side": "sell",
                              "amount": pytest.approx(57600.0)}]
        rows = ledger_rows(paths)
        assert rows[-1]["event"] == "flatten" and rows[-1]["reason"] == "expired"

    def test_delta_implausibile_skip(self, vp, paths):
        # IT: audit MINOR-2 — greek testnet assurdo (|δ_adj| > 1+Σm+0.10) →
        #     skip fail-soft, nessun hedge macroscopico.
        # EN: MINOR-2 audit — absurd testnet greek → fail-soft skip.
        write_position(paths, side=+1)
        db = FakeDB(0.08, -5.0, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=True)
        assert not paths["state"].exists() and db.orders == []

    def test_sim_mode_nessun_ordine(self, vp, paths):
        # IT: execute=False → fill simulato al mark del perp, zero market order.
        # EN: execute=False → simulated fill at the perp mark, zero market orders.
        write_position(paths, side=+1)
        db = FakeDB(0.08, -0.95, 0.005, 0.025, 64000.0, 64000.0)
        vp.maybe_hedge(db, HCFG, execute=False)
        assert db.orders == []
        rows = ledger_rows(paths)
        assert rows[0]["executed"] is False
        assert rows[0]["fill_price"] == pytest.approx(64000.0)
