# Tests for the FT1 judge (scripts/vol/ft1_execution_judge.py). Four families:
# (1) SENTINELS on the constants frozen by the pre-registration and amendment 1 —
#     changing one after seeing numbers is goalpost-moving;
# (2) COUNTING RULES P3/P4 on synthetic ledgers — what is an attempt, the first
#     incomplete closes FAIL, reading at the 8th complete roll only, 16-week window;
# (3) MEASUREMENT RULES P5/P6 — ③b tolerance, ②b minimum measured rolls, c_roll
#     arithmetic by hand and the ±10 min snapshot;
# (4) REAL RECORD SHAPES — records produced by 04b's own executor (FakeDB double)
#     are classified correctly, so the judge does not rely on assumed fields.
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    import quantsys.utils as qutils
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # both scripts call setup_logging() at import: neutralised for the import only.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(qutils, "setup_logging", lambda *a, **k: None)
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def J():
    return _load("ft1_execution_judge", ROOT / "scripts" / "vol" / "ft1_execution_judge.py")


GO_LIVE = pd.Timestamp("2026-09-17 12:00", tz="UTC")   # a Thursday
FRIDAYS = [pd.Timestamp("2026-09-18 08:00", tz="UTC") + pd.Timedelta(weeks=i) for i in range(20)]


def _alog(horizon, fridays=(), action="ADAPT_FLY", band="fly", dvol=0.34):
    recs = [{"ts": str(GO_LIVE), "dvol": dvol, "band": band, "action": "ADAPT_WAIT_FRIDAY"}]
    recs += [{"ts": str(f + pd.Timedelta(seconds=5)), "dvol": dvol, "band": band, "action": action}
             for f in fridays]
    recs.append({"ts": str(horizon), "dvol": dvol, "band": band, "action": "ADAPT_WAIT_FRIDAY"})
    return recs


def _roll(friday, span=10.0, measured=True, fee_err=0.0, fee_none=False, executed=True):
    # settled iron-butterfly record with the fields 04b persists.
    marks = {"wing_call": 0.002, "wing_put": 0.002, "call": 0.02, "put": 0.02}
    t0 = friday + pd.Timedelta(seconds=30)
    legs = {}
    for n, m in marks.items():
        fee = min(0.0003, 0.125 * m) + fee_err
        legs[n] = {"order_id": f"o-{n}", "order_state": "filled", "instrument": f"BTC-X-{n}",
                   "side": "buy" if n.startswith("wing") else "sell", "filled_amount": 1.0,
                   "average_price": m, "label": "x", "fee_observed_btc": None if fee_none else fee}
    return {"entry_ts": str(friday + pd.Timedelta(seconds=40)), "structure": "iron_butterfly",
            "side": -1, "executed": executed, "expiry_ms": int((friday + pd.Timedelta(days=7)).value // 10**6),
            "strike": 76000.0, "amount": 1.0, "exec_legs": legs,
            "receipt_ts": {n: str(t0 + pd.Timedelta(seconds=i)) for i, n in enumerate(legs)},
            "receipt_span_s": 3.0, "fill_span_s": span if measured else None,
            "fill_timing_source": "exchange_trades" if measured else "unavailable",
            "exit_mode": "settlement", "settled_ts": str(friday + pd.Timedelta(days=7, minutes=1))}


def _incomplete(friday, recovery="verified_flat", structure="iron_butterfly", attempt_id="a1"):
    return {"entry_ts": str(friday + pd.Timedelta(seconds=40)), "structure": structure,
            "exit_mode": "incomplete", "attempt_id": attempt_id, "executed": True,
            "recovery": recovery, "reason": "terminal_partial_call",
            "legs_filled": ["wing_call", "wing_put"], "legs": {}, "flatten": [], "band": "fly"}


def _inp(adaptive, trades, position=None, journal=None, exec_diag=None):
    return {"adaptive": adaptive, "trades": trades, "position": position,
            "journal": journal, "exec_diag": exec_diag or []}


# ───────────────────────────── (1) sentinels ─────────────────────────────
def test_preregistered_constants_sentinel(J):
    assert J.DVOL_THRESHOLD == 0.561 and J.K_WINGS == 1.5 and J.FILL_TIMEOUT_S == 120.0
    assert J.C_STAR == 0.25153 and J.N_ROLLS_READ == 8 and J.WINDOW_WEEKS == 16
    assert J.N_MEASURED_MIN == 5 and J.FEE_TOL_BTC == 1e-8 and J.FILL_SPAN_MAX_S == 30.0
    assert J.SNAPSHOT_TOL_MIN == 10.0 and (J.ENTRY_WEEKDAY, J.ENTRY_HOUR) == (4, 8)
    assert J.C_ROLL_EX_ANTE == {"first_friday_snapshot_ge_08utc": 0.065, "all_08utc_snapshots": 0.072}
    assert J.C_ROLL_SURPRISE == 0.15
    assert J.FEE_PER_CONTRACT == 0.0003 and J.FEE_CAP_FRAC == 0.125


def test_incomplete_rate_bound_is_the_exact_one_sided_95pct_bound(J):
    # P2: 0 failures in 8 → p ≤ 1 − 0.05^(1/8) = 31.2%; ≤ 10% needs 29 clean rolls.
    assert round(1 - 0.05 ** (1 / 8), 2) == J.INCOMPLETE_RATE_BOUND
    assert 1 - 0.05 ** (1 / 29) <= 0.10 < 1 - 0.05 ** (1 / 28)


# ─────────────────────────── (2) counting rules ───────────────────────────
def test_not_started_without_adaptive_log(J):
    assert J.evaluate(_inp([], []))["verdict"].startswith("NON AVVIATO")


def test_first_incomplete_closes_fail_2a_even_with_eight_completes_after(J, tmp_path):
    trades = [_incomplete(FRIDAYS[0])] + [_roll(f) for f in FRIDAYS[1:9]]
    res = J.evaluate(_inp(_alog(FRIDAYS[9], FRIDAYS[:9]), trades), chain_dir=tmp_path)
    assert res["verdict"].startswith("CHIUSO FAIL su ②a")
    assert res["fail_2a"]["recovery"] == "verified_flat" and "cond_1" not in res


def test_blocked_journal_without_record_is_incomplete_and_counted_once(J, tmp_path):
    journal = {"attempt_id": "zz", "structure": "iron_butterfly", "status": "in_progress",
               "created_ts": str(FRIDAYS[1] + pd.Timedelta(seconds=10)), "execute": True, "legs": []}
    trades = [_roll(FRIDAYS[0])]
    res = J.evaluate(_inp(_alog(FRIDAYS[2], FRIDAYS[:2]), trades, journal=journal), chain_dir=tmp_path)
    assert res["verdict"].startswith("CHIUSO FAIL su ②a") and res["fail_2a"]["source"] == "journal_only"
    # a blocked RECORD plus its retained journal (same attempt_id) is ONE attempt.
    trades = [_incomplete(FRIDAYS[0], recovery="blocked_operator_review", attempt_id="zz")]
    journal["created_ts"] = str(FRIDAYS[0] + pd.Timedelta(seconds=10))
    res = J.evaluate(_inp(_alog(FRIDAYS[1], FRIDAYS[:1]), trades, journal=journal), chain_dir=tmp_path)
    assert res["fly_attempts"] == 1 and res["fly_incomplete"] == 1


def test_daily_incomplete_no_dvol_and_non_executed_are_not_attempts(J, tmp_path):
    adaptive = _alog(FRIDAYS[3], FRIDAYS[:1])
    adaptive.append({"ts": str(FRIDAYS[1] + pd.Timedelta(seconds=5)), "dvol": None, "band": None,
                     "action": "ADAPT_NO_DVOL"})
    trades = [_roll(FRIDAYS[0]),
              _incomplete(FRIDAYS[1] - pd.Timedelta(days=1), structure="adaptive_daily_straddle"),
              _roll(FRIDAYS[2], executed=False)]
    res = J.evaluate(_inp(adaptive, trades), chain_dir=tmp_path)
    assert res["fly_attempts"] == 1 and res["fly_incomplete"] == 0
    assert res["excluded_not_executed"] == 1
    assert res["daily_incomplete"]["verified_flat"] == 1
    acts = {f["friday"]: f["action"] for f in res["fridays"]}
    assert acts[str(FRIDAYS[1])] == "ADAPT_NO_DVOL" and acts[str(FRIDAYS[2])] == "no_record"
    assert res["verdict"].startswith("IN CORSO")


def test_attempts_outside_the_16_week_window_are_ignored(J, tmp_path):
    # an incomplete BEFORE go-live (v1 era cannot produce one, but a replayed
    # ledger could) and one AFTER the window never close the gate.
    before = _incomplete(GO_LIVE - pd.Timedelta(days=6), attempt_id="b")
    after = _incomplete(FRIDAYS[16], attempt_id="c")
    res = J.evaluate(_inp(_alog(FRIDAYS[17]), [before, after]), chain_dir=tmp_path)
    assert res["fly_attempts"] == 0
    assert res["verdict"].startswith("NESSUNA CONCLUSIONE — < 8")


def test_in_progress_until_the_data_horizon_passes_the_window_end(J, tmp_path):
    trades = [_roll(f) for f in FRIDAYS[:3]]
    res = J.evaluate(_inp(_alog(FRIDAYS[5], FRIDAYS[:3]), trades), chain_dir=tmp_path)
    assert res["verdict"].startswith("IN CORSO") and "cond_2b" not in res
    end = GO_LIVE + pd.Timedelta(weeks=16)
    res = J.evaluate(_inp(_alog(end, FRIDAYS[:3]), trades), chain_dir=tmp_path)
    assert res["verdict"].startswith("NESSUNA CONCLUSIONE — < 8")


def test_reading_uses_exactly_the_first_eight_complete_rolls(J, tmp_path):
    # first 8: measured spans 10,10,40,40,40 (+3 unmeasured) → median 40 → FAIL ②b.
    # Rolls 9-10 (10 s) would pull the median to 10 if they were included.
    spans = [10, 10, 40, 40, 40, None, None, None, 10, 10]
    trades = [_roll(f, span=s or 0.0, measured=s is not None) for f, s in zip(FRIDAYS, spans)]
    res = J.evaluate(_inp(_alog(FRIDAYS[10], FRIDAYS[:10]), trades), chain_dir=tmp_path)
    assert res["cond_2b"]["n_rolls"] == 8 and res["cond_2b"]["median_s"] == 40.0
    # no chain on disk → ① unavailable; a measured FAIL still closes the gate.
    assert res["cond_1"]["status"] == "NESSUNA CONCLUSIONE"
    assert res["verdict"].startswith("CHIUSO FAIL su ②b")
    assert "≤ 31%" in res["cond_2a"]["claim"]


# ─────────────────────────── (3) measurement rules ───────────────────────────
def test_3b_absolute_tolerance_and_none_fee(J):
    assert J.fee_check_3b(_roll(FRIDAYS[0], fee_err=0.9e-8))["passed"]
    assert not J.fee_check_3b(_roll(FRIDAYS[0], fee_err=1.1e-8))["passed"]
    assert not J.fee_check_3b(_roll(FRIDAYS[0], fee_none=True))["passed"]


def test_3b_failure_on_first_complete_roll_closes_no_conclusion(J, tmp_path):
    trades = [_roll(FRIDAYS[0], fee_none=True), _incomplete(FRIDAYS[1])]
    res = J.evaluate(_inp(_alog(FRIDAYS[2], FRIDAYS[:2]), trades), chain_dir=tmp_path)
    assert res["verdict"].startswith("NESSUNA CONCLUSIONE — ③b")


def test_2b_needs_five_measured_rolls_and_threshold_is_inclusive(J):
    rolls = [_roll(f, span=10.0) for f in FRIDAYS[:4]] + [_roll(f, measured=False) for f in FRIDAYS[4:8]]
    assert J.fill_span_2b(rolls)["status"] == "NESSUNA CONCLUSIONE"
    rolls = [_roll(f, span=30.0) for f in FRIDAYS[:5]] + [_roll(f, measured=False) for f in FRIDAYS[5:8]]
    assert J.fill_span_2b(rolls)["status"] == "PASS"
    rolls[0]["fill_span_s"] = rolls[1]["fill_span_s"] = rolls[2]["fill_span_s"] = 30.001
    assert J.fill_span_2b(rolls)["status"] == "FAIL"


def _quotes(snap_ts, spread=0.001, expiry_days=7.0):
    def q(k, mark, iv):
        return {"mark_price": mark, "bid_price": mark - spread / 2, "ask_price": mark + spread / 2,
                "mark_iv": iv, "underlying_price": 76000.0, "strike": k,
                "expiry": snap_ts + pd.Timedelta(days=expiry_days), "snapshot_ts": snap_ts}
    return {"call": q(76000.0, 0.02, 40.0), "put": q(76000.0, 0.02, 40.0),
            "wing_call": q(82000.0, 0.005, 45.0), "wing_put": q(70000.0, 0.005, 48.0)}


def test_c_roll_hand_arithmetic(J):
    ts = pd.Timestamp("2026-09-18 08:01", tz="UTC")
    c = J.c_roll(_quotes(ts), 1.0, 20.0)
    fee4 = 0.0003 * 4                        # 0.125·0.005 = 0.000625 > cap → 0.0003
    hs = 4 * 0.0005
    t_y, tau_y = 7 * 24 / 8760.0, 20.0 / 3600.0 / 8760.0
    leg = 0.0
    for k, iv in ((82000.0, 0.45), (70000.0, 0.48)):
        d1 = (math.log(76000.0 / k) + 0.5 * iv * iv * t_y) / (iv * math.sqrt(t_y))
        leg += 0.5 * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) * iv * tau_y / math.sqrt(t_y)
    assert c["net_premium_btc"] == pytest.approx(0.03)
    assert c["legging_btc"] == pytest.approx(leg, rel=1e-12)
    assert c["c_roll"] == pytest.approx((fee4 + hs + leg) / 0.03, rel=1e-12)
    # degenerate inputs → None, never a partial cost.
    bad = _quotes(ts)
    bad["put"]["bid_price"] = float("nan")
    assert J.c_roll(bad, 1.0, 20.0) is None
    bad = _quotes(ts)
    bad["wing_call"]["mark_price"] = 0.04   # net premium 0.04 − 0.045 < 0
    assert J.c_roll(bad, 1.0, 20.0) is None


def _write_chain(tmp_path, snaps, roll, spread=0.001):
    rows = []
    names = {n: roll["exec_legs"][n]["instrument"] for n in J_FLY_LEGS}
    for s in snaps:
        for n, q in _quotes(s, spread=spread).items():
            rows.append({"snapshot_ts": s, "instrument_name": names[n], "expiry": q["expiry"],
                         "strike": q["strike"], "option_type": "C" if "call" in n else "P",
                         "mark_iv": q["mark_iv"], "underlying_price": q["underlying_price"],
                         "mark_price": q["mark_price"], "bid_price": q["bid_price"],
                         "ask_price": q["ask_price"], "open_interest": 1.0, "volume": 1.0})
    df = pd.DataFrame(rows)
    for d, g in df.groupby(df["snapshot_ts"].dt.strftime("%Y%m%d")):
        path = tmp_path / f"btc_options_{d}.parquet"
        if path.exists():
            g = pd.concat([pd.read_parquet(path), g], ignore_index=True)
        g.to_parquet(path, index=False)


J_FLY_LEGS = ("wing_call", "wing_put", "call", "put")


def test_snapshot_nearest_to_last_fill_within_ten_minutes(J, tmp_path):
    roll = _roll(FRIDAYS[0])
    t_last = J.last_fill_ts(roll)
    _write_chain(tmp_path, [t_last - pd.Timedelta(minutes=3), t_last + pd.Timedelta(minutes=6)], roll)
    quotes, cause = J.roll_quotes(roll, tmp_path)
    assert cause is None and quotes["call"]["snapshot_ts"] == t_last - pd.Timedelta(minutes=3)
    other = tmp_path / "far"
    other.mkdir()
    _write_chain(other, [t_last + pd.Timedelta(minutes=11)], roll)
    quotes, cause = J.roll_quotes(roll, other)
    assert quotes is None and "±10" in cause


def test_condition_1_pass_and_fail_on_chain_snapshots(J, tmp_path):
    trades = [_roll(f) for f in FRIDAYS[:8]]
    for r in trades:
        _write_chain(tmp_path, [J.last_fill_ts(r) - pd.Timedelta(minutes=2)], r)
    res = J.evaluate(_inp(_alog(FRIDAYS[8], FRIDAYS[:8]), trades), chain_dir=tmp_path)
    assert res["cond_1"]["n_available"] == 8 and res["cond_1"]["median_c_roll"] < J.C_STAR
    assert res["verdict"].startswith("CHIUSO PASS")
    assert res["cond_1"]["robustness_tau_120s"] >= res["cond_1"]["median_c_roll"]
    wide = tmp_path / "wide"
    wide.mkdir()
    for r in trades:
        _write_chain(wide, [J.last_fill_ts(r) - pd.Timedelta(minutes=2)], r, spread=0.004)
    res = J.evaluate(_inp(_alog(FRIDAYS[8], FRIDAYS[:8]), trades), chain_dir=wide)
    assert res["verdict"].startswith("CHIUSO FAIL su ①")


def test_integrity_errors_block_the_verdict(J, tmp_path):
    # band inconsistent with the frozen threshold (wrong CLI at go-live).
    res = J.evaluate(_inp(_alog(FRIDAYS[1], FRIDAYS[:1], band="fly", dvol=0.60), []), chain_dir=tmp_path)
    assert res["verdict"].startswith("INTEGRITY_ERROR")
    # a butterfly attempt outside the Friday 08 UTC slot.
    res = J.evaluate(_inp(_alog(FRIDAYS[1]), [_roll(FRIDAYS[0] + pd.Timedelta(days=1))]), chain_dir=tmp_path)
    assert res["verdict"].startswith("INTEGRITY_ERROR")


def test_count_only_writes_no_report(J, tmp_path, monkeypatch, capsys):
    vp = tmp_path / "vp"
    vp.mkdir()
    (vp / "adaptive.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _alog(FRIDAYS[1], FRIDAYS[:1])),
                                       encoding="utf-8")
    (vp / "trades.jsonl").write_text(json.dumps(_roll(FRIDAYS[0])) + "\n", encoding="utf-8")
    monkeypatch.setattr(J, "VP_DIR", vp)
    monkeypatch.setattr(J, "OUT_PATH", tmp_path / "out.json")
    monkeypatch.setattr(sys, "argv", ["ft1", "--count-only"])
    assert J.main() == 0
    assert not (tmp_path / "out.json").exists()
    assert "COUNT_ONLY" in capsys.readouterr().out


# ─────────────────────────── (4) real record shapes ───────────────────────────
@pytest.fixture(scope="module")
def adaptive_mod():
    return _load("adaptive_structure_tests_ft1", ROOT / "tests" / "test_adaptive_structure.py")


@pytest.fixture()
def vp04b(adaptive_mod, tmp_path, monkeypatch):
    vp = _load("volpaper_04b_ft1", ROOT / "scripts" / "04b_vol_paper.py")
    p = {"pos": tmp_path / "position.json", "trades": tmp_path / "trades.jsonl",
         "journal": tmp_path / "adaptive_entry_journal.json"}
    monkeypatch.setattr(vp, "POSITION_PATH", p["pos"])
    monkeypatch.setattr(vp, "TRADES_PATH", p["trades"])
    monkeypatch.setattr(vp, "ADAPTIVE_JOURNAL_PATH", p["journal"])
    monkeypatch.setattr(vp, "ADAPTIVE_LOG_PATH", tmp_path / "adaptive.jsonl")
    return vp, p


def _executor_inputs(J, p):
    return J.load_inputs(p["pos"].parent)


def test_real_executor_records_are_classified(J, adaptive_mod, vp04b):
    vp, p = vp04b
    FakeDB = adaptive_mod.FakeDB
    # complete structure → position.json with 4 exec_legs carrying instrument.
    db = FakeDB(index=78000.0, mark=0.01)
    pick = vp.DeribitTestnet.pick_butterfly(db, 168.0, 1.5, 0.50)
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"}) is not None
    att = J.collect_attempts(_executor_inputs(J, p))
    assert [(a["source"], a["complete"]) for a in att] == [("position_open", True)]
    assert set(att[0]["rec"]["exec_legs"]) == set(J.FLY_LEGS)
    assert all(att[0]["rec"]["exec_legs"][n]["instrument"] for n in J.FLY_LEGS)
    assert J.last_fill_ts(att[0]["rec"]) is not None
    assert J.fee_check_3b(att[0]["rec"])["legs"]["call"]["fee_observed_btc"] == 0.0
    p["pos"].unlink()
    # failed body leg → verified_flat record.
    db = FakeDB(index=78000.0, fail_on=pick["call"])
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"}) is None
    # lost response → blocked record AND retained journal: one attempt.
    db = FakeDB(index=78000.0, raise_on=pick["call"])
    assert vp.open_butterfly(db, pick, execute=True, timeout_s=120.0, meta={"band": "fly"}) is None
    att = J.collect_attempts(_executor_inputs(J, p))
    assert [(a["source"], a["complete"], a["recovery"]) for a in att] == [
        ("trades_incomplete", False, "verified_flat"),
        ("trades_incomplete", False, "blocked_operator_review")]
    assert p["journal"].exists()
