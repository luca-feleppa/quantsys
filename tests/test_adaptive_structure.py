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
    # IT: 04b chiama setup_logging() a livello di modulo (RotatingFileHandler sul root
    #     logger, scrive sotto logs/): va neutralizzata SOLO per la durata dell'import.
    #     Il fixture `monkeypatch` di pytest è function-scoped e non si può usare qui
    #     (fixture module-scoped) — si usa il context manager di MonkeyPatch, che
    #     ripristina l'attributo originale all'uscita del blocco `with`, anche in caso
    #     di eccezione.
    # EN: 04b calls setup_logging() at module level (RotatingFileHandler on the root
    #     logger, writes under logs/): it must be neutralised ONLY for the import's
    #     duration. pytest's `monkeypatch` fixture is function-scoped and cannot be used
    #     here (module-scoped fixture) — use MonkeyPatch's context manager instead, which
    #     restores the original attribute on exit of the `with` block, exception included.
    import quantsys.utils as qutils
    spec = importlib.util.spec_from_file_location(
        "volpaper_04b_adapt", ROOT / "scripts" / "04b_vol_paper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["volpaper_04b_adapt"] = mod
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(qutils, "setup_logging", lambda *a, **k: None)
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
    def __init__(self, index=78000.0, mark=0.01, fail_on=None, trade_map=None,
                 partial_on=None, raise_on=None, reverse_fail_on=None):
        self.index, self.mark, self.fail_on = index, mark, fail_on
        self.orders = []
        # IT: knob della famiglia (6), tutti selettivi per LABEL (entry vs reverse):
        #     partial_on = {strumento: quantità eseguita} → fill parziale TERMINALE;
        #     raise_on = strumento la cui submit di entry solleva (risposta PERSA);
        #     reverse_fail_on = strumento la cui gamba di COMPENSAZIONE risponde
        #     senza "order" (ambigua). None su tutti = comportamento invariato.
        # EN: family-(6) knobs, all keyed on the LABEL (entry vs reverse):
        #     partial_on = {instrument: filled quantity} → TERMINAL partial fill;
        #     raise_on = instrument whose entry submission raises (LOST response);
        #     reverse_fail_on = instrument whose COMPENSATING leg answers without
        #     "order" (ambiguous). All None = unchanged behaviour.
        self.partial_on = partial_on or {}
        self.raise_on, self.reverse_fail_on = raise_on, reverse_fail_on
        # IT: BUG A tests — trade_map: instrument -> lista di timestamp ms (trade
        #     sintetici VALIDI, quantità divisa in parti uguali fra loro — copre il
        #     caso multi-fill) OPPURE lista di dict COMPLETI forniti dal test (anche
        #     deliberatamente malformati: id duplicati, strumento/ordine sbagliati,
        #     quantità parziale). None (default) = un solo trade valido, come prima.
        # EN: BUG A tests — trade_map: instrument -> list of ms timestamps (VALID
        #     synthetic trades, quantity split evenly — covers the multi-fill case)
        #     OR a list of COMPLETE dicts supplied by the test (possibly malformed
        #     on purpose: duplicate ids, wrong instrument/order, partial quantity).
        #     None (default) = a single valid trade, as before.
        self.trade_map = trade_map or {}
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

    # IT: doppio dell'executor strutturato (execute=True): order_detailed/cancel/
    #     get_order_state. `fail_on` restituisce un rigetto DEFINITIVO a fill zero
    #     ("rejected", filled_amount=0 — MAI un'eccezione generica), che è ciò che
    #     classify_order riconosce come terminal_partial e abilita la compensazione;
    #     l'ordine rigettato non entra in `orders` (stesso invariante della vecchia
    #     eccezione: la gamba fallita non lascia traccia di esecuzione).
    # EN: double of the structured executor (execute=True): order_detailed/cancel/
    #     get_order_state. `fail_on` returns a DEFINITIVE zero-fill rejection
    #     ("rejected", filled_amount=0 — NEVER a generic exception), which is what
    #     classify_order recognises as terminal_partial and enables compensation;
    #     the rejected order does not enter `orders` (same invariant as the old
    #     exception: the failed leg leaves no execution trace).
    def order_detailed(self, instrument, side, amount, label):
        oid = f"o{len(self.orders)}"
        # IT/EN: risposta PERSA sulla submit di entry / LOST response on entry submit
        if self.raise_on == instrument and "-entry-" in label:
            raise RuntimeError("risposta persa / lost response")
        # IT/EN: risposta senza "order" sulla compensazione / no "order" on the reverse
        if self.reverse_fail_on == instrument and "-reverse-" in label:
            return {}
        if self.fail_on and instrument == self.fail_on and len(self.orders) < 4:
            return {"order": {"order_id": oid, "order_state": "rejected",
                              "instrument_name": instrument, "direction": side,
                              "filled_amount": 0.0}, "trades": []}
        self.orders.append({"instrument": instrument, "side": side, "amount": amount})
        # IT/EN: fill parziale terminale SOLO sulle gambe di entry / terminal partial
        #        fill ONLY on entry legs
        filled = self.partial_on.get(instrument, amount) if "-entry-" in label else amount
        order = {"order_id": oid, "order_state": ("filled" if filled >= amount else "cancelled"),
                 "instrument_name": instrument,
                 "direction": side, "filled_amount": filled, "average_price": self.mark}
        spec = self.trade_map.get(instrument)
        if spec is None:
            # IT/EN: default = un solo trade REALE completo e valido (come prima).
            trades = [{"trade_id": f"{instrument}-{oid}", "order_id": oid,
                      "instrument_name": instrument, "direction": side, "amount": filled,
                      "timestamp": time.time() * 1000, "fee": 0.0, "fee_currency": "BTC"}]
        elif spec and isinstance(spec[0], dict):
            trades = spec   # IT/EN: record già completi forniti dal test (anche malformati apposta)
        else:
            n = len(spec)
            qty = filled / n
            trades = [{"trade_id": f"{instrument}-{oid}-{i}", "order_id": oid,
                      "instrument_name": instrument, "direction": side, "amount": qty,
                      "timestamp": ts, "fee": 0.0, "fee_currency": "BTC"}
                     for i, ts in enumerate(spec)]
        return {"order": order, "trades": trades}

    def cancel_order(self, order_id):
        return {"order_id": order_id, "order_state": "cancelled"}

    def get_order_state(self, order_id):
        return {"order_id": order_id, "order_state": "filled",
                "filled_amount": None, "average_price": None}


@pytest.fixture()
def paths(vp, tmp_path, monkeypatch):
    # IT/EN: tutti i file di stato su tmp — la produzione resta intoccata.
    p = {"pos": tmp_path / "position.json", "trades": tmp_path / "trades.jsonl",
         "alog": tmp_path / "adaptive.jsonl", "dvol": tmp_path / "dvol.parquet",
         "fc": tmp_path / "forecasts.parquet", "iv": tmp_path / "atm.parquet",
         "diag": tmp_path / "exec_diag.jsonl", "journal": tmp_path / "adaptive_journal.json"}
    monkeypatch.setattr(vp, "POSITION_PATH", p["pos"])
    monkeypatch.setattr(vp, "TRADES_PATH", p["trades"])
    monkeypatch.setattr(vp, "ADAPTIVE_LOG_PATH", p["alog"])
    monkeypatch.setattr(vp, "DVOL_PATH", p["dvol"])
    monkeypatch.setattr(vp, "FORECASTS_PATH", p["fc"])
    monkeypatch.setattr(vp, "IV_PATH", p["iv"])
    monkeypatch.setattr(vp, "EXEC_DIAG_PATH", p["diag"])
    # IT: SENZA questo il journal adattivo (begin/present/clear) leggerebbe/
    #     scriverebbe sotto results/vol_paper/ di PRODUZIONE ad ogni test che apre
    #     una struttura — deve applicarsi PRIMA di ogni test.
    # EN: WITHOUT this the adaptive journal (begin/present/clear) would read/write
    #     under PRODUCTION results/vol_paper/ on every test that opens a
    #     structure — must apply BEFORE every test.
    monkeypatch.setattr(vp, "ADAPTIVE_JOURNAL_PATH", p["journal"])
    # IT: patchare la sola costante EXEC_DIAG_PATH NON basta — log_exec_diag lega il
    #     default del suo parametro `path` al MOMENTO della definizione della funzione
    #     (`def log_exec_diag(db, path=EXEC_DIAG_PATH)`), quindi punta ancora al path di
    #     produzione. Si avvolge la funzione stessa per forzare sempre il path tmp.
    # EN: patching the EXEC_DIAG_PATH constant alone is NOT enough — log_exec_diag binds
    #     its `path` parameter's default at FUNCTION-DEFINITION time
    #     (`def log_exec_diag(db, path=EXEC_DIAG_PATH)`), so it still points at the
    #     production path. Wrap the function itself to always force the tmp path.
    orig_log_exec_diag = vp.log_exec_diag
    monkeypatch.setattr(vp, "log_exec_diag",
                         lambda db, *a, **k: orig_log_exec_diag(db, path=p["diag"]))
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


# ─────────────────── (4) BUG A — timing/fee di fill verificati ───────────────────
# IT: BUG A era: observed_fill_ts teneva SOLO l'ultimo trade per gamba, e _fill_timing
#     faceva max-min dei soli ultimi → con una gamba multi-fill lo span veniva
#     sottostimato. FIX: _verify_trade_history verifica la COPERTURA (id univoci,
#     identità ordine/strumento/verso, quantità che sommano al filled_amount, ts
#     finiti) e ritorna first+last; _fill_timing usa il first/last GLOBALE.
# EN: BUG A was: observed_fill_ts kept ONLY the last trade per leg, and _fill_timing
#     took max-min of those lasts → with a multi-fill leg the span was undercounted.
#     FIX: _verify_trade_history checks COVERAGE (unique ids, order/instrument/side
#     identity, quantities summing to filled_amount, finite ts) and returns first+
#     last; _fill_timing uses the GLOBAL first/last.
def test_fill_timing_global_first_to_last_not_per_leg_last(vp):
    # IT/EN: gamba "a" con due fill (t0, t5), gamba "b" con un fill a t20 — lo span
    #     corretto è 20s (globale), non 15s (max-min dei soli ULTIMI fill per gamba,
    #     il bug di prima).
    fills = {"a": {"exchange_fill_ts_first": 0.0, "exchange_fill_ts": 5.0},
             "b": {"exchange_fill_ts_first": 20.0, "exchange_fill_ts": 20.0}}
    span, source = vp._fill_timing(fills, ("a", "b"))
    assert span == 20.0 and source == "exchange_trades"
    naive_old_bug = (max(f["exchange_fill_ts"] for f in fills.values())
                    - min(f["exchange_fill_ts"] for f in fills.values()))
    assert naive_old_bug == 15.0 != span


def test_fill_timing_unavailable_on_missing_leg_timestamp(vp):
    fills = {"a": {"exchange_fill_ts_first": 0.0, "exchange_fill_ts": 5.0},
             "b": {"exchange_fill_ts_first": None, "exchange_fill_ts": None}}
    assert vp._fill_timing(fills, ("a", "b")) == (None, "unavailable")


def test_verify_trade_history_rejects_malformed_coverage(vp):
    good = [{"trade_id": "t1", "order_id": "o1", "instrument_name": "I", "direction": "sell",
             "amount": 1.0, "timestamp": 1000.0, "fee": 0.0, "fee_currency": "BTC"}]
    assert vp._verify_trade_history({"trades": good}, "I", "sell", "o1", 1.0) == \
        {"fee_btc": 0.0, "first_ts": 1.0, "last_ts": 1.0}
    # IT/EN: nessun trade / risposta senza "trades"
    assert vp._verify_trade_history({"trades": []}, "I", "sell", "o1", 1.0) is None
    assert vp._verify_trade_history({}, "I", "sell", "o1", 1.0) is None
    # IT/EN: id duplicato
    dup = good + [dict(good[0])]
    assert vp._verify_trade_history({"trades": dup}, "I", "sell", "o1", 2.0) is None
    # IT/EN: strumento sbagliato
    wrong_inst = [{**good[0], "instrument_name": "OTHER"}]
    assert vp._verify_trade_history({"trades": wrong_inst}, "I", "sell", "o1", 1.0) is None
    # IT/EN: order_id sbagliato
    wrong_oid = [{**good[0], "order_id": "o_other"}]
    assert vp._verify_trade_history({"trades": wrong_oid}, "I", "sell", "o1", 1.0) is None
    # IT/EN: quantità parziale (non somma al filled_amount noto)
    partial = [{**good[0], "trade_id": "t2", "amount": 0.5}]
    assert vp._verify_trade_history({"trades": partial}, "I", "sell", "o1", 1.0) is None


def test_open_butterfly_fill_span_global_first_to_last(vp, paths):
    # IT/EN: integrazione end-to-end — wing_call multi-fill (t0,t5), le altre 3 gambe
    #     single-fill fino a t20: fill_span_s atteso 20s (globale), non 15/18s.
    db = FakeDB(index=78000.0, mark=0.01, trade_map={
        "BTC-7D-88000-C": [1_700_000_000_000.0, 1_700_000_005_000.0],
        "BTC-7D-72000-P": [1_700_000_002_000.0],
        "BTC-7D-76000-C": [1_700_000_010_000.0],
        "BTC-7D-76000-P": [1_700_000_020_000.0],
    })
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    pos = vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    assert pos["fill_span_s"] == pytest.approx(20.0) and pos["fill_timing_source"] == "exchange_trades"
    assert pos["fee_btc_observed"] == pytest.approx(0.0)


def test_open_butterfly_fee_and_timing_unknown_on_malformed_leg_trades(vp, paths):
    # IT/EN: la gamba "put" ha due trade con lo STESSO trade_id (duplicato) — la
    #     struttura si completa comunque (il fill non dipende dai trade), ma fee/
    #     timing osservati diventano ignoti, mai zero/inventati.
    bad_trades = [
        {"trade_id": "dup", "order_id": "o3", "instrument_name": "BTC-7D-76000-P",
         "direction": "sell", "amount": 1.0, "timestamp": 1_700_000_020_000.0,
         "fee": 0.0, "fee_currency": "BTC"},
        {"trade_id": "dup", "order_id": "o3", "instrument_name": "BTC-7D-76000-P",
         "direction": "sell", "amount": 0.0, "timestamp": 1_700_000_021_000.0,
         "fee": 0.0, "fee_currency": "BTC"},
    ]
    db = FakeDB(index=78000.0, mark=0.01, trade_map={"BTC-7D-76000-P": bad_trades})
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    pos = vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    assert pos is not None
    assert pos["fee_btc_observed"] is None
    assert pos["fill_span_s"] is None and pos["fill_timing_source"] == "unavailable"


def test_reclassify_after_cancel_marks_fee_and_timing_unknown(vp):
    # IT/EN: query-after-cancel — get_order_state riporta "filled" con quantità nota
    #     ma NON porta trade: fee/timing (first e last) devono restare ignoti, mai
    #     dedotti dalla risposta di stato.
    class DB2:
        def cancel_order(self, order_id):
            return {"order_id": order_id, "order_state": "cancelled"}

        def get_order_state(self, order_id):
            return {"order_id": order_id, "order_state": "filled",
                    "instrument_name": "I", "direction": "sell",
                    "filled_amount": 1.0, "average_price": 0.01}
    resolved, cancel_err, state_err = vp._adaptive_reclassify_nonterminal(
        DB2(), "o1", "I", "sell", 1.0, 0.0)
    assert resolved["kind"] == "filled" and resolved["filled_amount"] == 1.0
    assert resolved["fee_observed_btc"] is None
    assert resolved["exchange_fill_ts"] is None and resolved["exchange_fill_ts_first"] is None
    assert cancel_err is None and state_err is None


# ─────────────────── (5) BUG B — coda diagnostica su journal sopravvissuto ───────────────────
# IT: BUG B era: se l'entry adattiva di un tick creava un journal bloccato (entry
#     fallita), la coda del tick chiamava comunque log_exec_diag, che vedeva
#     position=None e registrava uno snapshot "flat" nonostante l'esposizione reale
#     fosse ignota. FIX: guardia in log_exec_diag stesso + skip esplicito della coda
#     diagnostica/hedge in tick() quando il journal sopravvive all'entry adattiva.
# EN: BUG B was: if a tick's adaptive entry created a blocked journal (failed entry),
#     the tick's tail still called log_exec_diag, which saw position=None and logged
#     a "flat" snapshot despite the real exposure being unknown. FIX: a guard inside
#     log_exec_diag itself + an explicit skip of the diagnostic/hedge tail in tick()
#     when the journal survives the adaptive entry.
def test_log_exec_diag_guards_against_surviving_journal(vp, paths):
    paths["journal"].write_text(json.dumps({"status": "blocked_operator_review"}),
                                encoding="utf-8")
    # IT: si CONTANO gli accessi al client invece di farli esplodere. log_exec_diag è
    #     fail-soft (`except Exception`): un raise dal client veniva inghiottito e il
    #     file restava assente ANCHE a guardia disattivata — il vecchio test passava per
    #     la ragione sbagliata (revisione 2026-09-15). La prova è "nessuna chiamata".
    # EN: client accesses are COUNTED instead of raising. log_exec_diag is fail-soft
    #     (`except Exception`): a raise from the client was swallowed and the file
    #     stayed absent EVEN with the guard disabled — the old test passed for the wrong
    #     reason (2026-09-15 review). The evidence is "no call at all".
    calls = []

    class SpyDB:
        def __getattr__(self, name):
            calls.append(name)
            raise AssertionError(f"venue call attempted: {name}")
    vp.log_exec_diag(SpyDB())
    assert calls == [] and not paths["diag"].exists()
    paths["journal"].unlink()


def test_tick_skips_diagnostic_and_hedge_tail_when_adaptive_entry_blocks(vp, paths, monkeypatch):
    def _blocked_entry(fc, db, execute, acfg, sig, now):
        paths["journal"].write_text(json.dumps({"status": "blocked_operator_review"}),
                                    encoding="utf-8")
        return "ADAPT_BLOCKED"

    def _boom(*a, **k):
        raise AssertionError("must not run when the journal survives this tick")
    monkeypatch.setattr(vp, "adaptive_entry", _blocked_entry)
    monkeypatch.setattr(vp, "log_exec_diag", _boom)
    monkeypatch.setattr(vp, "maybe_hedge", _boom)
    hedge_cfg = {"band": 0.20, "conv": "raw", "fee": 5e-4, "band_mode": "fixed", "ww_lambda": None}
    vp.tick(FakeFC(), FakeDB(), execute=False, hedge_cfg=hedge_cfg, adaptive_cfg=ACFG)
    fc_df = pd.read_parquet(paths["fc"])
    assert fc_df["action"].iloc[-1] == "ADAPT_BLOCKED"
    assert not paths["diag"].exists()
    paths["journal"].unlink()


# ────────── (6) identità dei record e durabilità del journal (2026-09-15) ──────────
# IT: famiglia aggiunta perché il conteggio 18/18 NON copriva questi contratti: (a)
#     identità dei trade RICHIESTA e degradazione a ignoto (mai un raise) su metadati
#     malformati; (b) strumento e verso CONCRETI conservati nel record di recovery dopo
#     la cancellazione del journal; (c) durabilità del journal (scritto prima del primo
#     ordine, mai sovrascritto), risposte perse, fill parziali, timeout nelle due
#     posizioni, ordine e arresto della compensazione, blocco al riavvio.
# EN: family added because the 18/18 count did NOT cover these contracts: (a) trade
#     identity REQUIRED and degradation to unknown (never a raise) on malformed
#     metadata; (b) CONCRETE instrument and side retained in the recovery record after
#     the journal is cleared; (c) journal durability (written before the first order,
#     never overwritten), lost responses, partial fills, timeouts in both positions,
#     compensation ordering and halting, restart blocking.
TRADE_OK = {"trade_id": "t1", "order_id": "o1", "instrument_name": "I", "direction": "sell",
            "amount": 1.0, "timestamp": 1000.0, "fee": 0.0, "fee_currency": "BTC"}


def _fake_monotonic(values):
    # IT/EN: seam monotono deterministico — valori in ordine, poi l'ultimo si ripete.
    state = {"i": 0, "v": values[-1]}

    def _m():
        if state["i"] < len(values):
            state["v"] = values[state["i"]]
            state["i"] += 1
        return state["v"]
    return _m


def test_verify_trade_history_requires_explicit_identity(vp):
    # IT: un trade SENZA order_id/strumento/verso non è verificabile. Prima passava
    #     (confronto `not in (None, atteso)`): la copertura risultava verificata pur
    #     non avendo nulla da verificare, ed è l'identità che il record conserva.
    # EN: a trade WITHOUT order_id/instrument/side is not verifiable. It used to pass
    #     (`not in (None, expected)`): coverage came out verified with nothing to
    #     verify — and that is the identity the record retains.
    assert vp._verify_trade_history({"trades": [TRADE_OK]}, "I", "sell", "o1", 1.0) == \
        {"fee_btc": 0.0, "first_ts": 1.0, "last_ts": 1.0}
    for missing in ("order_id", "instrument_name", "direction"):
        t = {k: v for k, v in TRADE_OK.items() if k != missing}
        assert vp._verify_trade_history({"trades": [t]}, "I", "sell", "o1", 1.0) is None
    # IT/EN: nessun order_id lato ORDINE → nessuna identità contro cui verificare
    assert vp._verify_trade_history({"trades": [TRADE_OK]}, "I", "sell", None, 1.0) is None


def test_verify_trade_history_unhashable_trade_id_is_unknown_not_crash(vp):
    # IT: un trade_id lista/dict sollevava TypeError su `in seen_ids`, DENTRO
    #     _adaptive_submit_leg che non è protetto: il raise avrebbe interrotto il
    #     journaling dopo che gli ordini erano partiti. Ora è ignoto, non un'eccezione.
    # EN: a list/dict trade_id raised TypeError on `in seen_ids`, INSIDE the
    #     unprotected _adaptive_submit_leg: the raise would have interrupted journaling
    #     after orders were submitted. Now it is unknown, not an exception.
    for bad in (["not", "hashable"], {"a": 1}, None, 3.5):
        t = {**TRADE_OK, "trade_id": bad}
        assert vp._verify_trade_history({"trades": [t]}, "I", "sell", "o1", 1.0) is None


def test_open_butterfly_malformed_trade_id_does_not_interrupt_journaling(vp, paths):
    # IT/EN: end-to-end — la struttura si completa (il fill non dipende dai trade), fee
    #        e timing restano ignoti, il journal viene chiuso: nessuna eccezione.
    bad = [{**TRADE_OK, "trade_id": ["x"], "order_id": "o3",
            "instrument_name": "BTC-7D-76000-P", "timestamp": 1_700_000_000_000.0}]
    db = FakeDB(index=78000.0, mark=0.01, trade_map={"BTC-7D-76000-P": bad})
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    pos = vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    assert pos is not None and pos["fee_btc_observed"] is None
    assert pos["fill_span_s"] is None and pos["fill_timing_source"] == "unavailable"
    assert not paths["journal"].exists()


def test_incomplete_record_retains_instrument_and_side(vp, paths):
    # IT: il journal è l'UNICO posto dove vivevano strumento e verso delle reverse, e su
    #     verified_flat viene cancellato: il record deve portarli, altrimenti resta solo
    #     il nome logico della gamba e la prova di recovery non è ricostruibile.
    # EN: the journal was the ONLY place holding the reverses' instrument and side, and
    #     it is cleared on verified_flat: the record must carry them, else only the
    #     leg's logical name survives and the recovery evidence is not reconstructable.
    db = FakeDB(index=78000.0, fail_on="BTC-7D-76000-C")
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0,
                             meta={"band": "fly"}) is None
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    flat = {f["leg"]: f for f in rec["flatten"]}
    assert flat["wing_call"]["instrument"] == pick["wing_call"] and flat["wing_call"]["side"] == "sell"
    assert flat["wing_put"]["instrument"] == pick["wing_put"] and flat["wing_put"]["side"] == "sell"
    # IT/EN: anche le gambe di ENTRY portano l'identità concreta (verso originale)
    assert rec["legs"]["wing_call"]["instrument"] == pick["wing_call"]
    assert rec["legs"]["wing_call"]["side"] == "buy"
    assert rec["legs"]["call"]["instrument"] == pick["call"] and rec["legs"]["call"]["side"] == "sell"
    assert not paths["journal"].exists()


def test_journal_written_before_the_first_order_and_cleared_on_success(vp, paths, monkeypatch):
    # IT/EN: durabilità — a ogni submit il journal esiste già su disco e contiene la
    #        gamba in corso con status "submitting" (mai un ordine senza intento scritto).
    seen = []
    orig = vp._adaptive_submit_leg

    def spy(db, execute, instrument, verb, amount, label):
        assert paths["journal"].exists(), "ordine senza journal / order without journal"
        j = json.loads(paths["journal"].read_text(encoding="utf-8"))
        assert j["legs"][-1]["label"] == label and j["legs"][-1]["status"] == "submitting"
        assert j["legs"][-1]["instrument"] == instrument and j["legs"][-1]["side"] == verb
        seen.append(instrument)
        return orig(db, execute, instrument, verb, amount, label)
    monkeypatch.setattr(vp, "_adaptive_submit_leg", spy)
    db = FakeDB(index=78000.0)
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    pos = vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    assert pos is not None and len(seen) == 4 and not paths["journal"].exists()


def test_journal_begin_refuses_to_overwrite_an_existing_attempt(vp, paths):
    # IT/EN: un journal presente = tentativo non chiuso: un secondo begin lo cancellerebbe
    #        in silenzio. Deve sollevare PRIMA di qualunque ordine, lasciandolo intatto.
    paths["journal"].write_text(json.dumps({"status": "blocked_operator_review",
                                            "attempt_id": "old"}), encoding="utf-8")
    db = FakeDB(index=78000.0)
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    with pytest.raises(RuntimeError):
        vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    assert db.orders == []
    assert json.loads(paths["journal"].read_text(encoding="utf-8"))["attempt_id"] == "old"
    paths["journal"].unlink()


def test_lost_response_blocks_without_compensation(vp, paths):
    # IT: risposta PERSA sul corpo call (eccezione sulla submit): il fill è IGNOTO, non
    #     zero → nessuna compensazione al buio, journal RITENUTO, record scritto.
    # EN: LOST response on the body call (submit raises): the fill is UNKNOWN, not zero
    #     → no blind compensation, journal RETAINED, record written.
    db = FakeDB(index=78000.0, raise_on="BTC-7D-76000-C")
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0,
                             meta={"band": "fly"}) is None
    assert [(o["instrument"], o["side"]) for o in db.orders] == \
        [(pick["wing_call"], "buy"), (pick["wing_put"], "buy")]
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["recovery"] == "blocked_operator_review" and rec["flatten"] == []
    assert rec["legs_filled"] == ["wing_call", "wing_put"] and rec["reason"] == "unresolved_call"
    j = json.loads(paths["journal"].read_text(encoding="utf-8"))
    assert j["status"] == "blocked_operator_review" and j["block_reason"] == "unresolved_call"
    paths["journal"].unlink()


def test_partial_fill_compensates_exactly_the_known_quantity(vp, paths):
    # IT/EN: fill parziale terminale 0.4 sul corpo call → si rivende 0.4 (non 1.0) e poi
    #        le due ali intere, in ordine di entry INVERSO; esito verified_flat.
    db = FakeDB(index=78000.0, partial_on={"BTC-7D-76000-C": 0.4})
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0,
                             meta={"band": "fly"}) is None
    assert [(o["instrument"], o["side"], o["amount"]) for o in db.orders[3:]] == \
        [(pick["call"], "buy", 0.4), (pick["wing_put"], "sell", 1.0),
         (pick["wing_call"], "sell", 1.0)]
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["recovery"] == "verified_flat" and rec["reason"] == "terminal_partial_call"
    assert rec["legs"]["call"]["filled_amount"] == pytest.approx(0.4)
    assert not paths["journal"].exists()


def test_timeout_before_next_leg_stops_and_compensates(vp, paths, monkeypatch):
    # IT/EN: il timeout scatta PRIMA della 3ª gamba → il corpo non viene mai sottomesso e
    #        le due ali già comprate vengono rivendute in ordine inverso.
    monkeypatch.setattr(vp, "_monotonic", _fake_monotonic([0.0, 0.0, 0.0, 0.0, 0.0, 200.0]))
    db = FakeDB(index=78000.0)
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0,
                             meta={"band": "fly"}) is None
    assert [(o["instrument"], o["side"]) for o in db.orders] == \
        [(pick["wing_call"], "buy"), (pick["wing_put"], "buy"),
         (pick["wing_put"], "sell"), (pick["wing_call"], "sell")]
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["reason"] == "timeout_before_call" and rec["recovery"] == "verified_flat"
    assert not paths["journal"].exists()


def test_timeout_after_last_leg_invalidates_the_structure(vp, paths, monkeypatch):
    # IT: tutte e 4 le gambe sono "filled" ma l'ultima risposta arriva OLTRE il timeout:
    #     la struttura NON è valida e passa comunque dalla regola di flatten — nessuna
    #     posizione salvata.
    # EN: all 4 legs are "filled" but the last response lands PAST the timeout: the
    #     structure is NOT valid and still goes through the flatten rule — no position.
    monkeypatch.setattr(vp, "_monotonic",
                        _fake_monotonic([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0]))
    db = FakeDB(index=78000.0)
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0,
                             meta={"band": "fly"}) is None
    assert not paths["pos"].exists()
    assert [o["side"] for o in db.orders] == ["buy", "buy", "sell", "sell",
                                              "buy", "buy", "sell", "sell"]
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["reason"] == "timeout_after_put" and rec["recovery"] == "verified_flat"
    assert len(rec["flatten"]) == 4 and not paths["journal"].exists()


def test_reverse_failure_blocks_and_stops_the_remaining_reverses(vp, paths):
    # IT: la PRIMA reverse non verificata ferma la compensazione: l'ala restante NON va
    #     rivenduta al buio mentre una gamba resta in stato ignoto. Journal RITENUTO.
    # EN: the FIRST unverified reverse halts compensation: the remaining wing must NOT
    #     be sold blind while a leg stays in an unknown state. Journal RETAINED.
    db = FakeDB(index=78000.0, fail_on="BTC-7D-76000-C", reverse_fail_on="BTC-7D-72000-P")
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0,
                             meta={"band": "fly"}) is None
    assert [(o["instrument"], o["side"]) for o in db.orders] == \
        [(pick["wing_call"], "buy"), (pick["wing_put"], "buy")]
    rec = json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert [f["leg"] for f in rec["flatten"]] == ["wing_put"]
    assert rec["flatten"][0]["error"] == "ambiguous"
    assert rec["flatten"][0]["instrument"] == pick["wing_put"] and rec["flatten"][0]["side"] == "sell"
    assert rec["recovery"] == "blocked_operator_review"
    j = json.loads(paths["journal"].read_text(encoding="utf-8"))
    assert j["block_reason"] == "reverse_failed_wing_put"
    paths["journal"].unlink()


def test_restart_guard_blocks_tick_and_main(vp, paths, monkeypatch):
    # IT/EN: un journal presente blocca il tick (nessuna chiamata al venue, nessuna riga
    #        di forecast) e l'avvio (SystemExit PRIMA di leggere config o modello).
    paths["journal"].write_text(json.dumps({"status": "blocked_operator_review"}),
                                encoding="utf-8")

    class BoomDB:
        def __getattr__(self, name):
            raise AssertionError(f"venue call attempted: {name}")
    vp.tick(FakeFC(), BoomDB(), execute=False, adaptive_cfg=ACFG)
    assert not paths["fc"].exists() and not paths["diag"].exists()
    monkeypatch.setattr(sys, "argv", ["04b_vol_paper.py", "--once"])
    monkeypatch.setattr(vp, "load_config",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("config letta")))
    with pytest.raises(SystemExit):
        vp.main()
    paths["journal"].unlink()


# ────────── (7) verificatore dell'ordine e ramo nonterminal (revisione 2026-09-15) ──────────
# IT: famiglia aggiunta dalla revisione indipendente del diff: i due punti in cui la regola
#     "mai compensare al buio" viene DECISA — classify_order e la ri-verifica post-cancel —
#     erano coperti solo dal caso felice; un'inversione di condizione non faceva cadere
#     nessun test. Più le due istanze residue della classe "raise non protetto a ordine
#     inviato" (trades non iterabile, order non-dict).
# EN: family added by the independent diff review: the two places where the "never
#     compensate blind" rule is DECIDED — classify_order and the post-cancel
#     re-verification — were covered only by the happy path; an inverted condition would
#     not fail any test. Plus the two residual instances of the "unprotected raise after
#     submission" class (non-iterable trades, non-dict order).
BODY_CALL, WING_PUT = "BTC-7D-76000-C", "BTC-7D-72000-P"
_DROP = object()


def _order(**kw):
    # IT/EN: ordine valido pieno; kw sovrascrive, _DROP rimuove la chiave.
    base = {"order_id": "o1", "order_state": "filled", "instrument_name": "I",
            "direction": "sell", "filled_amount": 1.0, "average_price": 0.01}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not _DROP}


def test_classify_order_contract(vp):
    # IT: tabella dei casi — ogni caso NON esplicitamente riconosciuto è ambiguous.
    # EN: case table — every case NOT explicitly recognised is ambiguous.
    cases = [
        (_order(), "filled"),
        (_order(order_state="cancelled", filled_amount=0.4), "terminal_partial"),
        (_order(order_state="rejected", filled_amount=0.0, average_price=None), "terminal_partial"),
        (_order(order_state="open", filled_amount=0.0, average_price=None), "nonterminal"),
        (_order(order_state="untriggered", filled_amount=0.0, average_price=None), "nonterminal"),
        # IT/EN: identità assente o diversa / missing or different identity
        (_order(instrument_name=_DROP), "ambiguous"),
        (_order(direction=_DROP), "ambiguous"),
        (_order(instrument_name="J"), "ambiguous"),
        (_order(direction="buy"), "ambiguous"),
        (_order(order_id=_DROP), "ambiguous"),
        # IT/EN: "filled" parziale = risposta contraddittoria / contradictory response
        (_order(filled_amount=0.4), "ambiguous"),
        (_order(filled_amount=1.5), "ambiguous"),
        (_order(average_price=None), "ambiguous"),
        (_order(filled_amount=None), "ambiguous"),
        (_order(order_state="weird"), "ambiguous"),
        (["not", "a", "dict"], "ambiguous"),
    ]
    for order, kind in cases:
        assert vp.classify_order(order, "I", "sell", 1.0)["kind"] == kind, order
    for bad_amount in (0.0, -1.0, float("nan"), float("inf")):
        assert vp.classify_order(_order(), "I", "sell", bad_amount)["kind"] == "ambiguous"


def test_verify_trade_history_non_list_trades_is_unknown_not_crash(vp):
    # IT/EN: 1/True/3.5 sollevavano TypeError sul `for` / used to raise TypeError on `for`
    for bad in (1, True, 3.5, "abc", {"trade_id": "t1"}):
        assert vp._verify_trade_history({"trades": bad}, "I", "sell", "o1", 1.0) is None


def test_submit_leg_non_dict_order_is_ambiguous_not_crash(vp):
    # IT/EN: `order.get` su una lista/stringa sollevava AttributeError a ordine inviato.
    for bad in (["x"], "x", 7):
        class DB:
            def order_detailed(self, instrument, side, amount, label, _bad=bad):
                return {"order": _bad, "trades": []}
        r = vp._adaptive_submit_leg(DB(), True, "I", "sell", 1.0, "lbl")
        assert r["kind"] == "ambiguous" and r["filled_amount"] is None


class NonterminalDB(FakeDB):
    # IT: doppio con ordini che restano APERTI alla submit. open_on =
    #     {(strumento, "entry"|"reverse"): (fill_iniziale, override_post_cancel)};
    #     get_order_state restituisce l'ordine con gli override (_DROP rimuove la chiave).
    # EN: double whose orders stay OPEN at submission. open_on =
    #     {(instrument, "entry"|"reverse"): (initial_fill, post_cancel_overrides)};
    #     get_order_state returns the order with the overrides (_DROP removes the key).
    def __init__(self, open_on, **kw):
        super().__init__(**kw)
        self.open_on, self.post, self.cancelled = open_on, {}, []

    def order_detailed(self, instrument, side, amount, label):
        phase = "entry" if "-entry-" in label else "reverse"
        spec = self.open_on.get((instrument, phase))
        if spec is None:
            return super().order_detailed(instrument, side, amount, label)
        initial, post = spec
        oid = f"open{len(self.orders)}"
        self.orders.append({"instrument": instrument, "side": side, "amount": amount})
        base = {"order_id": oid, "instrument_name": instrument, "direction": side,
                "average_price": self.mark}
        self.post[oid] = {k: v for k, v in {**base, **post}.items() if v is not _DROP}
        return {"order": {**base, "order_state": "open", "filled_amount": initial,
                          "average_price": self.mark if initial > 0 else None},
                "trades": []}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return super().cancel_order(order_id)

    def get_order_state(self, order_id):
        return self.post[order_id]


def _open_fly(vp, db):
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert pick["call"] == BODY_CALL and pick["wing_put"] == WING_PUT
    pos = vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"})
    return pick, pos


def _last_record(paths):
    return json.loads(paths["trades"].read_text(encoding="utf-8").strip().splitlines()[-1])


def test_nonterminal_persisting_after_cancel_blocks_without_compensation(vp, paths):
    # IT/EN: ancora "open" dopo il cancel → quantità finale ignota → blocked, zero reverse.
    db = NonterminalDB({(BODY_CALL, "entry"): (0.0, {"order_state": "open", "filled_amount": 0.0})},
                       index=78000.0)
    _, pos = _open_fly(vp, db)
    assert pos is None and len(db.orders) == 3 and len(db.cancelled) == 1
    rec = _last_record(paths)
    assert rec["recovery"] == "blocked_operator_review" and rec["reason"] == "unresolved_call"
    assert rec["flatten"] == []
    paths["journal"].unlink()


def test_nonterminal_then_cancelled_reverses_the_post_cancel_quantity(vp, paths):
    # IT: fill iniziale 0.1, post-cancel "cancelled" 0.4 → autorevole è 0.4: si ricompra
    #     0.4 (non 0.1, non 1.0), poi le ali intere in ordine inverso. Fee/timing ignoti.
    # EN: initial fill 0.1, post-cancel "cancelled" 0.4 → 0.4 is authoritative: buy back
    #     0.4 (not 0.1, not 1.0), then the full wings in reverse order. Fee/timing unknown.
    db = NonterminalDB({(BODY_CALL, "entry"): (0.1, {"order_state": "cancelled",
                                                     "filled_amount": 0.4})}, index=78000.0)
    pick, pos = _open_fly(vp, db)
    assert pos is None
    assert [(o["instrument"], o["side"], o["amount"]) for o in db.orders[3:]] == \
        [(pick["call"], "buy", 0.4), (pick["wing_put"], "sell", 1.0),
         (pick["wing_call"], "sell", 1.0)]
    rec = _last_record(paths)
    assert rec["recovery"] == "verified_flat" and rec["reason"] == "terminal_partial_call"
    assert rec["legs"]["call"]["filled_amount"] == pytest.approx(0.4)
    assert rec["legs"]["call"]["fee_observed_btc"] is None
    assert not paths["journal"].exists()


@pytest.mark.parametrize("initial,post", [
    (0.5, {"order_state": "cancelled", "filled_amount": 0.2}),                           # fill regredito
    (0.1, {"order_state": "cancelled", "filled_amount": 0.4, "instrument_name": _DROP}),  # senza strumento
    (0.1, {"order_state": "cancelled", "filled_amount": 0.4, "direction": _DROP}),        # senza verso
    (0.1, {"order_state": "cancelled", "filled_amount": 0.4, "direction": "buy"}),        # verso opposto
], ids=["regressed_fill", "no_instrument", "no_direction", "wrong_direction"])
def test_nonterminal_unverifiable_post_cancel_blocks(vp, paths, initial, post):
    # IT: una risposta post-cancel che non si può verificare NON autorizza una reverse
    #     dimensionata sulla sua quantità: blocked, nessuna compensazione. ⚠ Nei casi di
    #     identità il fill iniziale (0.1) è SOTTO quello post-cancel (0.4): con 0.5 il
    #     caso cadeva in ambiguous per fill regredito e il test passava anche col bug
    #     di identità (accertato con una prova di mutazione).
    # EN: an unverifiable post-cancel response does NOT authorise a reverse sized on its
    #     quantity: blocked, no compensation. ⚠ In the identity cases the initial fill
    #     (0.1) is BELOW the post-cancel one (0.4): with 0.5 the case fell into ambiguous
    #     via regressed fill and the test passed even with the identity bug (established
    #     by a mutation check).
    db = NonterminalDB({(BODY_CALL, "entry"): (initial, post)}, index=78000.0)
    _, pos = _open_fly(vp, db)
    assert pos is None and len(db.orders) == 3
    rec = _last_record(paths)
    assert rec["recovery"] == "blocked_operator_review" and rec["flatten"] == []
    paths["journal"].unlink()


def test_nonterminal_reverse_leg_is_reverified_before_continuing(vp, paths):
    # IT: la reverse dell'ala put resta aperta; la ri-verifica la chiude "filled" → la
    #     compensazione prosegue sull'ala call. Il record porta la risoluzione RI-VERIFICATA
    #     con strumento e verso inverso (il dict viene sostituito dal reclassify).
    # EN: the put wing's reverse stays open; re-verification resolves it "filled" →
    #     compensation continues on the call wing. The record carries the RE-VERIFIED
    #     resolution with instrument and reversed side (reclassify replaces the dict).
    db = NonterminalDB({(WING_PUT, "reverse"): (0.0, {"order_state": "filled",
                                                      "filled_amount": 1.0})},
                       index=78000.0, fail_on=BODY_CALL)
    pick, pos = _open_fly(vp, db)
    assert pos is None and len(db.cancelled) == 1
    assert [(o["instrument"], o["side"]) for o in db.orders] == \
        [(pick["wing_call"], "buy"), (pick["wing_put"], "buy"),
         (pick["wing_put"], "sell"), (pick["wing_call"], "sell")]
    rec = _last_record(paths)
    assert rec["recovery"] == "verified_flat"
    flat = {f["leg"]: f for f in rec["flatten"]}
    assert "error" not in flat["wing_put"] and flat["wing_put"]["order_id"].startswith("open")
    assert flat["wing_put"]["instrument"] == pick["wing_put"] and flat["wing_put"]["side"] == "sell"
    assert not paths["journal"].exists()
