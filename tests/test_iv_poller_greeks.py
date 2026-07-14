# IT: test dell'estensione --greeks di 01c_iv_poller (STATUS 2bis-①) su dati
#     SINTETICI: selezione expiry/strike identica a pick_straddle di 04b,
#     schema riga, fail su leg mancante, campi non-finiti → NaN. La rete è
#     monkeypatchata (_get) — nessuna chiamata reale.
# EN: tests for 01c_iv_poller's --greeks extension (STATUS 2bis-①) on SYNTHETIC
#     data: expiry/strike selection identical to 04b's pick_straddle, row
#     schema, missing-leg failure, non-finite fields → NaN. The network is
#     monkeypatched (_get) — no real calls.
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "iv_poller", ROOT / "scripts" / "01c_iv_poller.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

NOW = pd.Timestamp("2026-08-01 12:00:00+00:00")


def _chain():
    # IT: 2 expiry (20h e 44h dal NOW) × 2 strike × C/P — la 44h è la più
    #     vicina al tenor 30h (|44−30|=14 vs |20−30|=10... no: 10<14 → vince la
    #     20h). Costruita per rendere la scelta NON banale.
    # EN: 2 expiries (20h and 44h from NOW) × 2 strikes × C/P — 20h wins the
    #     30h tenor (|20−30|=10 < |44−30|=14). Built to make the choice
    #     non-trivial.
    rows = []
    for hrs in (20, 44):
        exp = NOW + pd.Timedelta(hours=hrs)
        for k in (60_000.0, 62_000.0):
            for opt in ("C", "P"):
                rows.append({"snapshot_ts": NOW,
                             "instrument_name": f"BTC-TEST-{hrs}-{int(k)}-{opt}",
                             "expiry": exp, "strike": k, "option_type": opt,
                             "mark_iv": 40.0, "underlying_price": 61_000.0,
                             "mark_price": 0.01, "bid_price": 0.009,
                             "ask_price": 0.011, "open_interest": 1.0,
                             "volume": 1.0})
    return pd.DataFrame(rows)


def _fake_get_factory(index_price=61_100.0, delta=0.5, drop_greeks=False):
    def fake_get(path, params, timeout=15):
        if path == "public/get_index_price":
            return {"index_price": index_price}
        if path == "public/ticker":
            g = {} if drop_greeks else {"delta": delta, "gamma": 1e-5,
                                        "vega": 12.0, "theta": -30.0}
            return {"greeks": g, "mark_price": 0.0123, "best_bid_price": 0.012,
                    "best_ask_price": 0.0126, "mark_iv": 41.5,
                    "underlying_price": 61_050.0,
                    "instrument_name": params["instrument_name"]}
        raise AssertionError(f"path inatteso/unexpected: {path}")
    return fake_get


def test_selection_tenor_and_atm_strike(monkeypatch):
    # IT: expiry a 20h (più vicina a 30h di quella a 44h) e strike 62000
    #     (più vicino all'index 61100 di 60000) — criterio di pick_straddle.
    # EN: 20h expiry (closer to 30h than 44h) and strike 62000 (closer to the
    #     61100 index than 60000) — pick_straddle's criterion.
    monkeypatch.setattr(P, "_get", _fake_get_factory(index_price=61_100.0))
    df = P.fetch_atm_greeks(_chain(), NOW)
    assert len(df) == 1
    r = df.iloc[0]
    assert r["t_hours"] == pytest.approx(20.0)
    assert r["strike"] == 62_000.0
    assert r["call_instrument"].endswith("-62000-C")
    assert r["put_instrument"].endswith("-62000-P")
    assert r["call_delta"] == pytest.approx(0.5)
    assert r["call_mark"] == pytest.approx(0.0123)


def test_missing_leg_raises(monkeypatch):
    # IT: chain senza la put allo strike ATM → RuntimeError (fail esplicito,
    #     che poll_once converte in warning fail-soft).
    # EN: chain missing the ATM-strike put → RuntimeError (explicit failure,
    #     converted to a fail-soft warning by poll_once).
    monkeypatch.setattr(P, "_get", _fake_get_factory(index_price=61_100.0))
    ch = _chain()
    ch = ch[~((ch["strike"] == 62_000.0) & (ch["option_type"] == "P"))]
    with pytest.raises(RuntimeError):
        P.fetch_atm_greeks(ch, NOW)


def test_absent_greeks_become_nan(monkeypatch):
    # IT: ticker senza blocco greeks (strike illiquido testnet) → NaN, non crash.
    # EN: ticker without the greeks block (illiquid testnet strike) → NaN, no crash.
    monkeypatch.setattr(P, "_get", _fake_get_factory(drop_greeks=True))
    df = P.fetch_atm_greeks(_chain(), NOW)
    assert np.isnan(df["call_delta"].iloc[0])
    assert np.isnan(df["put_vega"].iloc[0])
    assert df["call_mark"].iloc[0] == pytest.approx(0.0123)


def test_no_live_expiry_raises(monkeypatch):
    monkeypatch.setattr(P, "_get", _fake_get_factory())
    ch = _chain()
    with pytest.raises(RuntimeError):
        P.fetch_atm_greeks(ch, NOW + pd.Timedelta(days=10))
