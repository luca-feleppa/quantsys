# IT: test del giudice hedged-vs-unhedged (scripts/vol/hedged_vs_unhedged_judge.py)
#     su dati SINTETICI — nessuna rete, nessun file production. Verificano:
#     aritmetica PnL inverse esatta, segno/accrual del funding, composizione
#     hedged = unhedged + perp − fee − funding, gate n≥20, flag reconcile e
#     residuo aperto, condizioni di PASS pre-registrate.
# EN: hedged-vs-unhedged judge tests (scripts/vol/hedged_vs_unhedged_judge.py)
#     on SYNTHETIC data — no network, no production files. They verify: exact
#     inverse-PnL arithmetic, funding sign/accrual, hedged = unhedged + perp −
#     fee − funding composition, the n≥20 gate, reconcile/open-residual flags,
#     and the pre-registered PASS conditions.
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "hvu_judge", ROOT / "scripts" / "vol" / "hedged_vs_unhedged_judge.py")
J = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(J)

# IT: funding sintetico orario: rate costante 1e-5, index costante 50k.
# EN: synthetic hourly funding: constant 1e-5 rate, constant 50k index.
T0 = pd.Timestamp("2026-08-01 00:00:00+00:00")


def _funding(hours: int = 100, rate: float = 1e-5, index: float = 50_000.0):
    ts = [int((T0 + pd.Timedelta(hours=i)).value // 10**6) for i in range(hours)]
    return pd.DataFrame({"timestamp": ts,
                         "index_price": index, "interest_1h": rate})


def _ev(hours: float, event: str, dh: float, h_after: float, price,
        fee: float = 0.0, pk: dict | None = None):
    return {"ts": str(T0 + pd.Timedelta(hours=hours)), "event": event,
            "dh_usd": dh, "h_usd_after": h_after, "fill_price": price,
            "fee_btc": fee, "executed": False,
            "position_key": pk or {"side": -1, "strike": 50_000.0,
                                   "expiry_ms": int((T0 + pd.Timedelta(hours=30)).value // 10**6)}}


def test_inverse_pnl_single_interval():
    # IT: long +3000 USD @50k chiuso @48k → pnl = 3000·(1/50000−1/48000) < 0.
    # EN: +3000 USD long @50k closed @48k → pnl = 3000·(1/50000−1/48000) < 0.
    ev = [_ev(0, "open", 3000, 3000, 50_000.0),
          _ev(10, "flatten", -3000, 0.0, 48_000.0)]
    leg = J.perp_leg(ev, _funding(rate=0.0))
    expected = 3000 * (1 / 50_000 - 1 / 48_000)
    assert leg["pnl_perp_gross"] == pytest.approx(expected, rel=1e-12)
    assert expected < 0
    assert not leg["open_residual"] and not leg["has_reconcile"]


def test_inverse_pnl_rebalance_chain():
    # IT: catena open→rebalance→flatten: somma degli intervalli inverse.
    # EN: open→rebalance→flatten chain: sum of the inverse intervals.
    ev = [_ev(0, "open", 3000, 3000, 50_000.0),
          _ev(5, "rebalance", 2000, 5000, 52_000.0),
          _ev(20, "flatten", -5000, 0.0, 51_000.0)]
    leg = J.perp_leg(ev, _funding(rate=0.0))
    expected = 3000 * (1 / 50_000 - 1 / 52_000) + 5000 * (1 / 52_000 - 1 / 51_000)
    assert leg["pnl_perp_gross"] == pytest.approx(expected, rel=1e-12)


def test_fees_summed_from_ledger():
    ev = [_ev(0, "open", 3000, 3000, 50_000.0, fee=1e-5),
          _ev(20, "flatten", -3000, 0.0, 50_000.0, fee=2e-5)]
    leg = J.perp_leg(ev, _funding(rate=0.0))
    assert leg["fees_perp"] == pytest.approx(3e-5)
    assert leg["pnl_perp_gross"] == pytest.approx(0.0, abs=1e-15)


def test_funding_sign_and_accrual():
    # IT: +5000 USD per 10h a rate 1e-5, index 50k → paga 10·(5000/50000)·1e-5;
    #     lo short (H<0) RICEVE (segno opposto).
    # EN: +5000 USD for 10h at 1e-5 rate, 50k index → pays 10·(5000/50000)·1e-5;
    #     the short (H<0) RECEIVES (opposite sign).
    f = _funding()
    paid_long = J.funding_paid_btc(
        5000, int(T0.value // 10**6),
        int((T0 + pd.Timedelta(hours=10)).value // 10**6), f)
    assert paid_long == pytest.approx(10 * (5000 / 50_000) * 1e-5, rel=1e-12)
    paid_short = J.funding_paid_btc(
        -5000, int(T0.value // 10**6),
        int((T0 + pd.Timedelta(hours=10)).value // 10**6), f)
    assert paid_short == pytest.approx(-paid_long, rel=1e-12)


def test_reconcile_flag_no_gap_pnl():
    # IT: reconcile senza prezzo: H cambia, nessun PnL sul gap, trade flaggato.
    # EN: priceless reconcile: H changes, no gap PnL, trade flagged.
    ev = [_ev(0, "open", 3000, 3000, 50_000.0),
          {**_ev(5, "reconcile", 0.0, 4000, None), "fill_price": None},
          _ev(20, "flatten", -4000, 0.0, 50_000.0)]
    leg = J.perp_leg(ev, _funding(rate=0.0))
    assert leg["has_reconcile"]
    # IT: PnL solo sull'intervallo prezzato con l'H post-reconcile (50k→50k = 0).
    # EN: PnL only over the priced interval with post-reconcile H (50k→50k = 0).
    assert leg["pnl_perp_gross"] == pytest.approx(0.0, abs=1e-15)


def test_open_residual_flagged():
    ev = [_ev(0, "open", 3000, 3000, 50_000.0)]
    leg = J.perp_leg(ev, _funding(),
                     settle_ms=int((T0 + pd.Timedelta(hours=30)).value // 10**6))
    assert leg["open_residual"]
    # IT: funding accruato fino al settlement anche senza flatten.
    # EN: funding accrued to settlement even without a flatten.
    assert leg["funding_paid"] == pytest.approx(30 * (3000 / 50_000) * 1e-5, rel=1e-12)


def _synthetic_sample(n: int, rng: np.random.Generator):
    # IT: n trade sintetici con hedge che DIMEZZA la deviazione (var/4) a media
    #     invariata → condizioni 1 e 2 attese PASS.
    # EN: n synthetic trades whose hedge HALVES the deviation (var/4) with
    #     unchanged mean → conditions 1 and 2 expected PASS.
    trades, ledger = [], []
    for i in range(n):
        entry = T0 + pd.Timedelta(days=i)
        expiry = entry + pd.Timedelta(hours=30)
        pk = {"side": -1, "strike": 50_000.0 + i,
              "expiry_ms": int(expiry.value // 10**6)}
        eps = float(rng.normal(0, 0.004))
        pnl_u = 0.001 + eps
        trades.append({"entry_ts": str(entry), "side": -1, "strike": pk["strike"],
                       "expiry_ms": pk["expiry_ms"], "prem_call": 0.01,
                       "prem_put": 0.01, "fee_btc": 0.0, "pnl_btc": pnl_u,
                       "delivery_price": 50_000.0, "payoff_btc": 0.0,
                       "settled_ts": str(expiry)})
        # IT: leg perp che compensa metà dell'eps: fill costruiti per dare
        #     pnl_perp = −eps/2 esatto con s0=50k (h = −eps/2 / (1/s0−1/s1)).
        # EN: perp leg offsetting half the eps: fills built to yield exactly
        #     pnl_perp = −eps/2 with s0=50k (h = −eps/2 / (1/s0−1/s1)).
        s0, s1 = 50_000.0, 49_000.0
        h = (-eps / 2) / (1 / s0 - 1 / s1)
        ts_e = (entry - T0).total_seconds() / 3600
        ledger.append(_ev(ts_e, "open", h, h, s0, fee=0.0, pk=pk))
        ledger.append(_ev(ts_e + 30, "flatten", -h, 0.0, s1, fee=0.0, pk=pk))
    return trades, ledger


def test_composition_and_pass_conditions():
    rng = np.random.default_rng(7)
    trades, ledger = _synthetic_sample(25, rng)
    rep = J.evaluate(trades, ledger, _funding(hours=26 * 31 * 24, rate=0.0),
                     since=T0)
    assert rep["n_hedged_settlements"] == 25
    # IT: composizione esatta per ogni riga: hedged = unhedged + perp − fee − funding.
    # EN: exact per-row composition: hedged = unhedged + perp − fee − funding.
    for r in rep["rows"]:
        assert r["pnl_hedged"] == pytest.approx(
            r["pnl_unhedged"] + r["pnl_perp_gross"] - r["fees_perp"]
            - r["funding_paid"], rel=1e-12)
    # IT: hedge = metà deviazione → var_ratio ≈ 0.25 ≤ 0.6, media invariata.
    # EN: hedge = half deviation → var_ratio ≈ 0.25 ≤ 0.6, unchanged mean.
    assert rep["aggregates"]["var_ratio"] == pytest.approx(0.25, abs=0.02)
    assert rep["conditions"]["1_var_ratio_le_0.6"]
    assert rep["conditions"]["2_mean_drag_le_quarter_se"]
    assert rep["verdict"] == "PASS"


def test_not_evaluable_below_n_min():
    rng = np.random.default_rng(3)
    trades, ledger = _synthetic_sample(5, rng)
    rep = J.evaluate(trades, ledger, _funding(rate=0.0), since=T0)
    assert rep["verdict"] == "NOT_EVALUABLE"
    assert "aggregates" not in rep
    assert "conditions" not in rep


def test_since_includes_zero_rebalance_trades():
    # IT: trade post-attivazione SENZA fill perp: nel campione con perp=0.
    # EN: post-activation trade WITHOUT perp fills: in-sample with perp=0.
    rng = np.random.default_rng(11)
    trades, ledger = _synthetic_sample(3, rng)
    entry = T0 + pd.Timedelta(days=10)
    expiry = entry + pd.Timedelta(hours=30)
    trades.append({"entry_ts": str(entry), "side": 1, "strike": 60_000.0,
                   "expiry_ms": int(expiry.value // 10**6), "pnl_btc": 0.002,
                   "settled_ts": str(expiry)})
    rep = J.evaluate(trades, ledger, _funding(rate=0.0), since=T0)
    assert rep["n_hedged_settlements"] == 4
    zr = [r for r in rep["rows"] if r["strike"] == 60_000.0][0]
    assert zr["n_events"] == 0
    assert zr["pnl_hedged"] == pytest.approx(zr["pnl_unhedged"])


def test_pre_activation_trades_excluded_by_since():
    rng = np.random.default_rng(5)
    trades, ledger = _synthetic_sample(3, rng)
    rep = J.evaluate(trades, ledger, _funding(rate=0.0),
                     since=T0 + pd.Timedelta(days=1, hours=1))
    # IT: i trade con entry < since escono dal campione (qui restano i giorni 2+).
    # EN: trades with entry < since leave the sample (days 2+ remain here).
    assert rep["n_hedged_settlements"] == 1


def test_report_roundtrip_json(tmp_path):
    # IT: il report deve essere JSON-serializzabile (bool numpy → bool nativi).
    # EN: the report must be JSON-serializable (numpy bools → native bools).
    rng = np.random.default_rng(2)
    trades, ledger = _synthetic_sample(21, rng)
    rep = J.evaluate(trades, ledger, _funding(rate=0.0), since=T0)
    p = tmp_path / "rep.json"
    p.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    back = json.loads(p.read_text(encoding="utf-8"))
    assert back["verdict"] in ("PASS", "FAIL")
