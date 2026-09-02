# IT: leva --adaptive di scripts/04b_vol_paper.py, offline e senza rete. Tre famiglie:
#     (1) INERZIA — con adaptive_cfg=None il tick non legge il DVOL e apre lo straddle
#         dal segnale come v1; la formula di settlement a 2 gambe è invariata;
#     (2) MECCANICA — pick_butterfly (expiry più vicina al tenor, ali strettamente OTM),
#         open_butterfly (4 ordini ali-PRIMA, corpo dopo; posizione a 4 premi e 4 fee),
#         regola di completamento (gamba fallita → flatten inverso, record `incomplete`,
#         nessuna posizione), settlement a 4 gambe con aritmetica verificata a mano;
#     (3) FAIL-FAST — DVOL stale → None mai un default; --adaptive senza soglia/k o con
#         un altro lever v2 → SystemExit.
# EN: --adaptive lever of scripts/04b_vol_paper.py, offline and network-free. Three
#     families: (1) INERTIA — with adaptive_cfg=None the tick never reads the DVOL and
#     opens the straddle from the signal as v1; the 2-leg settlement formula is unchanged;
#     (2) MECHANICS — pick_butterfly, open_butterfly (4 orders, wings FIRST), completion
#     rule (failed leg → reverse flatten, `incomplete` record, no position), 4-leg
#     settlement checked by hand; (3) FAIL-FAST — stale DVOL → None never a default;
#     --adaptive without threshold/k or with another v2 lever → SystemExit.
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vp():
    spec = importlib.util.spec_from_file_location(
        "volpaper_04b_adapt", ROOT / "scripts" / "04b_vol_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["volpaper_04b_adapt"] = mod
    spec.loader.exec_module(mod)
    return mod


NOW_MS = time.time() * 1000
EXP_7D = int(NOW_MS + 168 * 3.6e6)
EXP_14D = int(NOW_MS + 336 * 3.6e6)
EXP_1D = int(NOW_MS + 30 * 3.6e6)
STRIKES = [60000.0, 64000.0, 68000.0, 72000.0, 76000.0, 80000.0, 84000.0, 88000.0, 92000.0, 96000.0]


def _inst(exp, k, typ):
    tag = {EXP_7D: "7D", EXP_14D: "14D", EXP_1D: "1D"}[exp]
    return {"instrument_name": f"BTC-{tag}-{int(k)}-{typ[0].upper()}", "expiration_timestamp": exp,
            "strike": k, "option_type": typ}


class FakeDB:
    # IT: doppio del client Deribit: strumenti, index, ticker e ordini in memoria.
    #     `fail_on` fa fallire l'ordine sul nome strumento dato (regola di completamento).
    # EN: Deribit client double: instruments, index, tickers and orders in memory.
    #     `fail_on` makes the order fail on the given instrument (completion rule).
    def __init__(self, index=78000.0, mark=0.01, fail_on=None):
        self.index, self.mark, self.fail_on = index, mark, fail_on
        self.orders = []
        self.instruments = [_inst(e, k, t) for e in (EXP_1D, EXP_7D, EXP_14D)
                            for k in STRIKES for t in ("call", "put")]

    def get(self, path, params, private=False):
        if path == "public/get_instruments":
            return self.instruments
        if path == "public/get_index_price":
            return {"index_price": self.index}
        if path == "public/ticker":
            return {"mark_price": self.mark}
        raise AssertionError(path)

    def pick_straddle(self, tenor_hours):
        return {"expiry_ms": EXP_1D, "t_hours": 30.0, "strike": 76000.0, "index": self.index,
                "call": "BTC-1D-76000-C", "put": "BTC-1D-76000-P"}

    def mark_price(self, instrument):
        return self.mark

    def ticker(self, instrument):
        return {"best_bid_price": 0.009, "best_ask_price": 0.011, "mark_price": self.mark,
                "underlying_price": self.index, "greeks": {"delta": 0.5}}

    def market_order(self, instrument, side, amount):
        if self.fail_on and instrument == self.fail_on and len(self.orders) < 4:
            raise RuntimeError("no liquidity")
        self.orders.append({"instrument": instrument, "side": side, "amount": amount})
        return self.mark

    def delivery_price(self, expiry_ms):
        return self.index


@pytest.fixture()
def paths(vp, tmp_path, monkeypatch):
    # IT/EN: tutti i file di stato su tmp — la produzione resta intoccata.
    p = {"pos": tmp_path / "position.json", "trades": tmp_path / "trades.jsonl",
         "alog": tmp_path / "adaptive.jsonl", "dvol": tmp_path / "dvol.parquet",
         "fc": tmp_path / "forecasts.parquet", "iv": tmp_path / "atm.parquet",
         "diag": tmp_path / "exec_diag.jsonl"}
    monkeypatch.setattr(vp, "POSITION_PATH", p["pos"])
    monkeypatch.setattr(vp, "TRADES_PATH", p["trades"])
    monkeypatch.setattr(vp, "ADAPTIVE_LOG_PATH", p["alog"])
    monkeypatch.setattr(vp, "DVOL_PATH", p["dvol"])
    monkeypatch.setattr(vp, "FORECASTS_PATH", p["fc"])
    monkeypatch.setattr(vp, "IV_PATH", p["iv"])
    monkeypatch.setattr(vp, "EXEC_DIAG_PATH", p["diag"])
    return p


def write_dvol(path, dvol_pct, age_h=0.1):
    ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=age_h)
    pd.DataFrame({"timestamp": [ts], "dvol": [dvol_pct]}).to_parquet(path, index=False)


class FakeFC:
    def __init__(self):
        rng = np.random.default_rng(0)
        close = 78000.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 1000)))
        self.candles = pd.DataFrame({"open_time": pd.date_range("2026-01-01", periods=1000, freq="h", tz="UTC"),
                                     "close": close})

    def forecast(self):
        return {"candle_ts": pd.Timestamp("2026-09-04 08:00", tz="UTC"), "mu_z": 0.0,
                "log_rv": -7.0, "rv_pred": 5e-4, "rv_trail": 4e-4}


ACFG = {"threshold": 0.561, "k": 1.5, "fill_timeout_s": 120.0, "tenor_hours": 168.0}


# ───────────────────────────── (1) inerzia ─────────────────────────────
def test_tick_v1_never_reads_dvol_and_opens_from_signal(vp, paths, monkeypatch):
    # IT: adaptive_cfg=None → read_dvol NON è chiamata; con IV assente il tick resta
    #     NO_IV (v1). Con IV e edge < −soglia apre SHORT dal segnale, come v1.
    # EN: adaptive_cfg=None → read_dvol is NOT called; missing IV → NO_IV (v1). With IV
    #     and edge < −threshold it opens SHORT from the signal, as v1.
    monkeypatch.setattr(vp, "read_dvol", lambda *a, **k: (_ for _ in ()).throw(AssertionError("letto")))
    db = FakeDB()
    vp.tick(FakeFC(), db, execute=False)
    fc = pd.read_parquet(paths["fc"])
    assert fc["action"].iloc[-1] == "NO_IV" and db.orders == [] and not paths["pos"].exists()
    # IT/EN: IV fresca con var_iv grande → edge molto negativo → SHORT v1
    pd.DataFrame({"timestamp": [pd.Timestamp.now(tz="UTC")], "iv_30h": [400.0]}).to_parquet(paths["iv"], index=False)
    vp.tick(FakeFC(), db, execute=False)
    pos = json.loads(paths["pos"].read_text(encoding="utf-8"))
    assert pos["side"] == -1 and "wings" not in pos and pd.read_parquet(paths["fc"])["action"].iloc[-1] == "SHORT"


def test_settlement_two_leg_formula_unchanged(vp, paths, monkeypatch):
    # IT/EN: formula v1: pnl = side·(|S−K|/S·amt − (pc+pp)·amt) − fee, verificata a mano
    pos = {"side": -1, "strike": 76000.0, "expiry_ms": int(NOW_MS - 3.6e6), "amount": 1.0,
           "prem_call": 0.010, "prem_put": 0.012, "fee_btc": 0.0006, "call": "C", "put": "P"}
    paths["pos"].write_text(json.dumps(pos), encoding="utf-8")
    db = FakeDB(index=80000.0)
    assert vp.maybe_settle(db, pos)
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    payoff = abs(80000.0 - 76000.0) / 80000.0
    assert rec["pnl_btc"] == pytest.approx(-1 * (payoff - 0.022) - 0.0006)
    assert "payoff_wings_btc" not in rec and not paths["pos"].exists()


# ───────────────────────────── (2) meccanica ─────────────────────────────
def test_pick_butterfly_expiry_and_strictly_otm_wings(vp):
    db = FakeDB(index=78000.0)
    # IT/EN: FakeDB non ha pick_butterfly: si chiama il metodo VERO sul doppio
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert pick["expiry_ms"] == EXP_7D and pick["strike"] == 76000.0
    kc, kp = pick["wing_strikes"]
    # IT/EN: target S·exp(±1.5·0.5·√(168/8760)) = 78000·exp(±0.1039) ≈ 86540 / 70300
    assert kc == 88000.0 and kp == 72000.0 and kc > 76000.0 > kp
    assert pick["wing_call"] == "BTC-7D-88000-C" and pick["wing_put"] == "BTC-7D-72000-P"
    assert 1.0 < pick["k_eff"][0] < 2.0


def test_open_butterfly_wings_first_then_body(vp, paths):
    db = FakeDB(index=78000.0, mark=0.01)
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    pos = vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    assert [o["side"] for o in db.orders] == ["buy", "buy", "sell", "sell"]
    assert [o["instrument"] for o in db.orders] == [pick["wing_call"], pick["wing_put"], pick["call"], pick["put"]]
    assert pos["structure"] == "iron_butterfly" and pos["side"] == -1 and pos["wings"] == [pick["wing_call"], pick["wing_put"]]
    assert pos["fee_btc"] == pytest.approx(4 * min(vp.FEE_PER_CONTRACT, vp.FEE_CAP_FRAC * 0.01))
    assert paths["pos"].exists() and pos["fill_span_s"] >= 0.0


def test_completion_rule_flattens_and_records_incomplete(vp, paths):
    # IT: la 3ª gamba (corpo call) fallisce → le 2 ali comprate vengono rivendute in
    #     ordine inverso, nessuna posizione, record `incomplete` in trades.jsonl.
    # EN: the 3rd leg (body call) fails → the 2 bought wings are sold back in reverse
    #     order, no position, `incomplete` record in trades.jsonl.
    db = FakeDB(index=78000.0, fail_on="BTC-7D-76000-C")
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    pos = vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    assert pos is None and not paths["pos"].exists()
    sides = [(o["instrument"], o["side"]) for o in db.orders]
    assert sides == [(pick["wing_call"], "buy"), (pick["wing_put"], "buy"),
                     (pick["wing_put"], "sell"), (pick["wing_call"], "sell")]
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["exit_mode"] == "incomplete" and rec["legs_filled"] == ["wing_call", "wing_put"]
    assert len(rec["flatten"]) == 2 and all("error" not in f for f in rec["flatten"])


def test_settlement_four_legs_hand_arithmetic(vp, paths):
    # IT/EN: S_del=90000, K=76000, ali 88000/72000, amt 1: corpo |S−K|/S = 0.15556,
    #        ala call (90000−88000)/90000 = 0.02222, put 0; premi corpo 0.022, ali 0.004
    pos = {"side": -1, "strike": 76000.0, "expiry_ms": int(NOW_MS - 3.6e6), "amount": 1.0,
           "prem_call": 0.010, "prem_put": 0.012, "prem_wing_call": 0.002, "prem_wing_put": 0.002,
           "wings": ["WC", "WP"], "wing_strikes": [88000.0, 72000.0], "fee_btc": 0.0012,
           "call": "C", "put": "P", "structure": "iron_butterfly"}
    paths["pos"].write_text(json.dumps(pos), encoding="utf-8")
    db = FakeDB(index=90000.0)
    assert vp.maybe_settle(db, pos)
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    body, wings = 14000.0 / 90000.0, 2000.0 / 90000.0
    assert rec["payoff_body_btc"] == pytest.approx(body) and rec["payoff_wings_btc"] == pytest.approx(wings)
    assert rec["pnl_btc"] == pytest.approx(-1 * ((body - wings) - (0.022 - 0.004)) - 0.0012)
    assert not paths["pos"].exists()


def test_adaptive_entry_bands(vp, paths, monkeypatch):
    fc, sig = FakeFC(), {"edge": 0.1, "rv_pred": 5e-4, "var_iv": 4e-4}
    # IT/EN: DVOL alto → straddle daily short col macchinario v1
    write_dvol(paths["dvol"], 65.0)
    db = FakeDB()
    # IT/EN: il doppio delega al metodo VERO di pick_butterfly (get_instruments/index finti)
    db.pick_butterfly = lambda *a: vp.DeribitTestnet.pick_butterfly(db, *a)
    act = vp.adaptive_entry(fc, db, False, ACFG, sig, pd.Timestamp("2026-09-02 08:01", tz="UTC"))
    pos = json.loads(paths["pos"].read_text(encoding="utf-8"))
    assert act == "ADAPT_SHORT_DAILY" and pos["side"] == -1 and "wings" not in pos
    paths["pos"].unlink()
    # IT/EN: DVOL basso, mercoledì → WAIT; venerdì 08 UTC → farfalla
    write_dvol(paths["dvol"], 37.0)
    assert vp.adaptive_entry(fc, db, False, ACFG, sig, pd.Timestamp("2026-09-02 08:01", tz="UTC")) == "ADAPT_WAIT_FRIDAY"
    assert not paths["pos"].exists()
    act = vp.adaptive_entry(fc, db, False, ACFG, sig, pd.Timestamp("2026-09-04 08:01", tz="UTC"))
    pos = json.loads(paths["pos"].read_text(encoding="utf-8"))
    assert act == "ADAPT_FLY" and pos["structure"] == "iron_butterfly" and pos["band"] == "fly"
    rows = [json.loads(l) for l in paths["alog"].read_text(encoding="utf-8").strip().splitlines()]
    assert [r["action"] for r in rows] == ["ADAPT_SHORT_DAILY", "ADAPT_WAIT_FRIDAY", "ADAPT_FLY"]


# ───────────────────────────── (3) fail-fast ─────────────────────────────
def test_read_dvol_stale_or_missing_is_none(vp, paths):
    assert vp.read_dvol() is None
    write_dvol(paths["dvol"], 50.0, age_h=2.0)
    assert vp.read_dvol() is None
    write_dvol(paths["dvol"], 50.0, age_h=0.2)
    assert vp.read_dvol()["dvol"] == pytest.approx(0.50)


def test_adaptive_no_dvol_stays_flat(vp, paths):
    act = vp.adaptive_entry(FakeFC(), FakeDB(), False, ACFG, {"edge": 0.0, "rv_pred": 1e-4, "var_iv": 1e-4},
                            pd.Timestamp("2026-09-04 08:01", tz="UTC"))
    assert act == "ADAPT_NO_DVOL" and not paths["pos"].exists()


def _args(**kw):
    base = dict(adaptive=False, adaptive_dvol_threshold=None, adaptive_k=None,
                adaptive_fill_timeout=120.0, adaptive_tenor_hours=168.0,
                hedge=False, pin_close_hours=None, size_mode="contracts")
    base.update(kw)
    return argparse.Namespace(**base)


def test_build_adaptive_cfg_fail_fast(vp):
    assert vp.build_adaptive_cfg(_args()) is None
    with pytest.raises(SystemExit):
        vp.build_adaptive_cfg(_args(adaptive=True))
    with pytest.raises(SystemExit):
        vp.build_adaptive_cfg(_args(adaptive=True, adaptive_dvol_threshold=0.561, adaptive_k=1.5, hedge=True))
    cfg = vp.build_adaptive_cfg(_args(adaptive=True, adaptive_dvol_threshold=0.561, adaptive_k=1.5))
    assert cfg == {"threshold": 0.561, "k": 1.5, "fill_timeout_s": 120.0, "tenor_hours": 168.0}
