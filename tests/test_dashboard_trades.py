# Trades tab of scripts/06_dashboard.py, offline and network-free: record normalization for
# every 04b record shape (v1 straddle, iron butterfly, incomplete structured attempt),
# the expired-but-not-settled flag, and the perp hedge leg, which must be the judge's
# own reconstruction (same function, not a copy) and must not leak into open positions.
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dash():
    spec = importlib.util.spec_from_file_location("dashboard_06", ROOT / "scripts" / "06_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _straddle(**kw):
    rec = {"entry_ts": "2026-08-10 08:01:00+00:00", "side": -1, "executed": True,
           "strike": 60000.0, "index_at_entry": 60100.0, "prem_call": 0.006, "prem_put": 0.007,
           "fee_btc": 0.0006, "amount": 1.0, "expiry_ms": 1786003200000,
           "call": "BTC-11AUG26-60000-C", "put": "BTC-11AUG26-60000-P"}
    rec.update(kw)
    return rec


def _write(tmp_path, dash, monkeypatch, trades, position=None, ledger=None, funding=None):
    tp, pp, lp, fp = (tmp_path / n for n in ("trades.jsonl", "position.json",
                                            "hedge_ledger.jsonl", "funding.parquet"))
    tp.write_text("\n".join(json.dumps(t) for t in trades) + "\n", encoding="utf-8")
    if position is not None:
        pp.write_text(json.dumps(position), encoding="utf-8")
    if ledger is not None:
        lp.write_text("\n".join(json.dumps(e) for e in ledger) + "\n", encoding="utf-8")
    if funding is not None:
        funding.to_parquet(fp)
    for name, path in (("TRADES_PATH", tp), ("POSITION_PATH", pp),
                       ("LEDGER_PATH", lp), ("FUNDING_CACHE", fp)):
        monkeypatch.setattr(dash, name, path)
    dash._HEDGE_CACHE.clear()


def test_butterfly_premium_is_net_of_wings(dash):
    rec = _straddle(structure="iron_butterfly", wing_strikes=[65000.0, 55000.0],
                    prem_wing_call=0.001, prem_wing_put=0.002)
    row = dash._trade_row(rec, now_ms=0)
    assert row["structure"] == "iron_butterfly"
    assert row["premium"] == pytest.approx(0.006 + 0.007 - 0.001 - 0.002)
    assert row["wing_strikes"] == [65000.0, 55000.0]


def test_incomplete_attempt_is_never_a_position(dash, tmp_path, monkeypatch):
    inc = {"entry_ts": "2026-08-10 08:01:00+00:00", "structure": "iron_butterfly",
           "exit_mode": "incomplete", "executed": True, "recovery": "verified_flat",
           "legs": {}, "flatten": []}
    settled = _straddle(delivery_price=60500.0, payoff_btc=0.0083, pnl_btc=0.004,
                        settled_ts="2026-08-11 08:05:00+00:00", exit_mode="settlement")
    _write(tmp_path, dash, monkeypatch, [settled, inc])
    out = dash.build_trades()
    s = out["summary"]
    assert s["n_incomplete"] == 1 and s["n_settled"] == 1
    assert s["total_pnl"] == pytest.approx(0.004)
    row = out["trades"][1]
    assert row["incomplete"] and not row["settled"] and not row["expired_pending"]
    assert row["strike"] is None and row["pnl_total_btc"] is None


def test_expired_position_is_flagged_not_open(dash):
    pos = _straddle()
    assert dash._trade_row(pos, now_ms=pos["expiry_ms"] - 1)["expired_pending"] is False
    assert dash._trade_row(pos, now_ms=pos["expiry_ms"] + 1)["expired_pending"] is True


def test_hedge_leg_matches_judge_and_skips_open_positions(dash, tmp_path, monkeypatch):
    key = {"side": -1, "strike": 60000.0, "expiry_ms": 1786003200000}
    ledger = [
        {"ts": "2026-08-10 09:00:00+00:00", "event": "hedge", "h_usd_after": 5000.0,
         "fill_price": 60000.0, "fee_btc": 0.00004, "executed": True, "position_key": key},
        {"ts": "2026-08-10 20:00:00+00:00", "event": "flatten", "h_usd_after": 0.0,
         "fill_price": 61000.0, "fee_btc": 0.00004, "executed": True, "position_key": key},
    ]
    t0 = pd.Timestamp("2026-08-10 09:00:00+00:00").value // 10**6
    funding = pd.DataFrame({"timestamp": [t0 + h * 3600_000 for h in range(24)],
                            "interest_1h": [1e-5] * 24, "index_price": [60000.0] * 24})
    settled = _straddle(delivery_price=60500.0, payoff_btc=0.0083, pnl_btc=0.004,
                        settled_ts="2026-08-11 08:05:00+00:00", exit_mode="settlement")
    _write(tmp_path, dash, monkeypatch, [settled], position=_straddle(expiry_ms=1786089600000),
           ledger=ledger, funding=funding)
    out = dash.build_trades()

    judge_spec = importlib.util.spec_from_file_location(
        "hvu_judge_ref", ROOT / "scripts" / "vol" / "hedged_vs_unhedged_judge.py")
    judge = importlib.util.module_from_spec(judge_spec)
    judge_spec.loader.exec_module(judge)
    leg = judge.perp_leg(ledger, funding, key["expiry_ms"])
    expected = leg["pnl_perp_gross"] - leg["fees_perp"] - leg["funding_paid"]
    assert expected != 0.0

    closed, opened = out["trades"]
    assert closed["hedge_pnl_btc"] == pytest.approx(expected, abs=1e-15)
    assert closed["pnl_total_btc"] == pytest.approx(0.004 + expected, abs=1e-15)
    assert opened["hedge_pnl_btc"] is None and opened["pnl_total_btc"] is None
    assert out["summary"]["hedge_pnl"] == pytest.approx(expected, abs=1e-15)
    assert out["summary"]["n_hedged"] == 1
