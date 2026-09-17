# FT1 EXECUTION JUDGE (pre-registration STATUS.md 2026-09-02 + amendment 1 of
# 2026-09-15). READ-ONLY on results/vol_paper/ (adaptive.jsonl, trades.jsonl,
# position.json, adaptive_entry_journal.json, exec_diag.jsonl) and on the mainnet
# chain snapshots in data/iv/chain. FT1 validates EXECUTION of the adaptive
# structure (--adaptive of 04b), not its PnL. Where the amendment and the
# original tables diverge, the amendment rules. Implemented rules:
#   P3  attempts = butterfly entries only (Friday 08 UTC, fly band, --execute).
#       Any non-complete executor outcome (verified_flat OR blocked, including a
#       journal left on disk without a ledger record) is INCOMPLETE. NO_DVOL,
#       daily band, service down or a leftover journal are NOT attempts: they
#       are reported per Friday with their cause.
#   P4  read ONCE at the 8th complete butterfly roll; the FIRST incomplete
#       attempt closes FAIL on ②a immediately; < 8 complete at 16 weeks with no
#       incomplete = NO CONCLUSION.
#   P5  ②b = median fill_span_s over rolls with fill_timing_source ==
#       "exchange_trades", >= 5 measured of 8 or NO CONCLUSION. ③b = on the FIRST
#       complete roll, every leg |fee_observed - schedule| <= 1e-8 BTC; a None
#       fee on any leg = not passed = NO CONCLUSION.
#   P6  c_roll frozen definition (see c_roll()); ① needs c_roll on >= 5 of 8.
#   P1  ① is a pre-computed CONTROL (still gating): ex-ante value reported.
#   P2  a PASS on ②a is stated only together with its <= 31% bound.
# --count-only prints counts only: no c_roll, no fill span, no verdict, no report.
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quantsys.utils import setup_logging                      # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.ft1_judge")

VP_DIR = ROOT / "results" / "vol_paper"
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
OUT_PATH = ROOT / "results" / "vols" / "ft1_execution.json"

# PRE-REGISTERED constants (FT1 2026-09-02 + amendment 1) — never touched after
# numbers are seen; sentinel in tests/test_ft1_execution_judge.py.
DVOL_THRESHOLD = 0.561           # band threshold (fraction), read once at entry
K_WINGS = 1.5                    # wing distance in sigma_trail*sqrt(T) units
FILL_TIMEOUT_S = 120.0           # completion rule; also the unmeasured-tau fallback
C_STAR = 0.25153                 # ① break-even as a fraction of the net premium
N_ROLLS_READ = 8                 # ④ reading moment: 8th complete butterfly roll
WINDOW_WEEKS = 16                # window from go-live
N_MEASURED_MIN = 5               # ①/②b minimum rolls with the quantity available
FEE_TOL_BTC = 1e-8               # ③b absolute tolerance per leg
FILL_SPAN_MAX_S = 30.0           # ②b threshold on the median
SNAPSHOT_TOL_MIN = 10.0          # c_roll: chain snapshot within ±10 min of last fill
ENTRY_WEEKDAY, ENTRY_HOUR = 4, 8  # Friday 08 UTC
# ① ex-ante values frozen by amendment 1 (P1), reported next to the verdict.
C_ROLL_EX_ANTE = {"first_friday_snapshot_ge_08utc": 0.065, "all_08utc_snapshots": 0.072}
C_ROLL_SURPRISE = 0.15           # declared surprising outcome (measurement problem)
INCOMPLETE_RATE_BOUND = 0.31     # P2: 0/8 bounds the per-roll rate at <= 31% (95% one-sided)
# Deribit option fee schedule, same constants as 04b.
FEE_PER_CONTRACT = 0.0003
FEE_CAP_FRAC = 0.125
HOURS_PER_YEAR = 8760.0

FLY = "iron_butterfly"
DAILY = "adaptive_daily_straddle"
FLY_LEGS = ("wing_call", "wing_put", "call", "put")
# a leg SELLS in the body and BUYS in the wings (execution order wings first).
LEG_SIDE = {"wing_call": "buy", "wing_put": "buy", "call": "sell", "put": "sell"}


# ─────────────────────────────── inputs ───────────────────────────────
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_json(path: Path):
    # a corrupt journal is still a journal: its presence is what matters (04b
    # never parses it to decide); an unparsable one is returned as a marker.
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"_corrupt": True}


def load_inputs(vp_dir: Path | None = None) -> dict:
    # paths resolved at CALL time (a default bound at definition would ignore a
    # patched module constant).
    vp_dir = VP_DIR if vp_dir is None else vp_dir
    return {"adaptive": _read_jsonl(vp_dir / "adaptive.jsonl"),
            "trades": _read_jsonl(vp_dir / "trades.jsonl"),
            "position": _read_json(vp_dir / "position.json"),
            "journal": _read_json(vp_dir / "adaptive_entry_journal.json"),
            "exec_diag": _read_jsonl(vp_dir / "exec_diag.jsonl")}


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _is_entry_slot(t: pd.Timestamp) -> bool:
    return t.weekday() == ENTRY_WEEKDAY and t.hour == ENTRY_HOUR


# ─────────────────────────── attempts (P3) ───────────────────────────
def collect_attempts(inp: dict) -> list[dict]:
    # Builds every structured-entry attempt from the LEDGER (trades.jsonl records,
    # the open position, a surviving journal), not from adaptive.jsonl: the ledger
    # is written BEFORE the adaptive log line, so it is the record that survives a
    # crash in between. adaptive.jsonl is used for go-live, horizon, the Friday
    # calendar and consistency checks.
    out = []
    seen_attempt_ids = set()
    for r in inp["trades"]:
        s = r.get("structure")
        if s not in (FLY, DAILY):
            continue
        if r.get("exit_mode") == "incomplete":
            seen_attempt_ids.add(r.get("attempt_id"))
            out.append({"structure": s, "entry_ts": _ts(r["entry_ts"]), "complete": False,
                        "executed": bool(r.get("executed")), "source": "trades_incomplete",
                        "recovery": r.get("recovery"), "reason": r.get("reason"),
                        "attempt_id": r.get("attempt_id"), "rec": r})
        elif s == FLY:
            out.append({"structure": s, "entry_ts": _ts(r["entry_ts"]), "complete": True,
                        "executed": bool(r.get("executed")), "source": "trades_settled",
                        "rec": r})
    pos = inp["position"]
    if isinstance(pos, dict) and pos.get("structure") == FLY:
        key = (str(pos.get("entry_ts")), pos.get("expiry_ms"))
        if not any(a["source"] == "trades_settled" and
                   (str(a["rec"].get("entry_ts")), a["rec"].get("expiry_ms")) == key for a in out):
            out.append({"structure": FLY, "entry_ts": _ts(pos["entry_ts"]), "complete": True,
                        "executed": bool(pos.get("executed")), "source": "position_open",
                        "rec": pos})
    # a journal with no ledger record = an attempt interrupted before its record
    # was written (crash mid-execution, or after a complete fill but before
    # save_position): 04b stays blocked, so the executor did not close complete
    # → incomplete; the journal status is kept in the reason.
    j = inp["journal"]
    if isinstance(j, dict) and j.get("attempt_id") not in seen_attempt_ids:
        corrupt = bool(j.get("_corrupt"))
        out.append({"structure": None if corrupt else j.get("structure"),
                    "entry_ts": None if corrupt or not j.get("created_ts") else _ts(j["created_ts"]),
                    "complete": False, "executed": None if corrupt else bool(j.get("execute")),
                    "source": "journal_only", "recovery": "blocked_operator_review",
                    "reason": f"journal_without_ledger_record:{j.get('status')}", "attempt_id": j.get("attempt_id"),
                    "rec": j})
    out.sort(key=lambda a: (a["entry_ts"] is None, a["entry_ts"] or pd.Timestamp(0, tz="UTC")))
    return out


def friday_calendar(adaptive: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    # every Friday 08 UTC in [start, end): its adaptive.jsonl action, or
    # "no_record" (service down, or a journal skipping the tick — 04b writes no
    # adaptive line on a skipped tick). Reported, never counted as attempts.
    by_slot = {}
    for r in adaptive:
        t = _ts(r["ts"])
        if _is_entry_slot(t):
            by_slot[t.floor("h")] = r
    out = []
    d = start.floor("D") + pd.Timedelta(hours=ENTRY_HOUR)
    while d < end:
        if d.weekday() == ENTRY_WEEKDAY and d >= start.floor("h"):
            r = by_slot.get(d)
            out.append({"friday": str(d), "action": r.get("action") if r else "no_record",
                        "band": r.get("band") if r else None})
        d += pd.Timedelta(days=1)
    return out


def integrity_issues(inp: dict, attempts: list[dict], go_live: pd.Timestamp) -> list[str]:
    # data-integrity guards, NOT gate rules: a violation makes the ledger
    # unreadable as the pre-registered experiment (wrong CLI at go-live, attempts
    # outside the entry slot, a corrupt journal) → INTEGRITY_ERROR, manual review.
    issues = []
    for r in inp["adaptive"]:
        dv, band = r.get("dvol"), r.get("band")
        if dv is not None and band is not None and \
           band != ("daily" if dv >= DVOL_THRESHOLD else "fly"):
            issues.append(f"band {band} incoerente con dvol {dv} a soglia {DVOL_THRESHOLD} ({r.get('ts')})")
    for a in attempts:
        if a["entry_ts"] is None or a["structure"] is None:
            issues.append("journal corrotto o senza timestamp / corrupt or timestamp-less journal")
            continue
        if a["entry_ts"] < go_live:
            continue
        if a["structure"] == FLY and not _is_entry_slot(a["entry_ts"]):
            issues.append(f"tentativo farfalla fuori dallo slot venerdì 08 UTC: {a['entry_ts']}")
        if a["structure"] == FLY and a["complete"]:
            legs = a["rec"].get("exec_legs") or {}
            if set(legs) != set(FLY_LEGS):
                issues.append(f"roll completo senza le 4 exec_legs: {a['entry_ts']}")
    return issues


# ────────────────────────────── ③b (P5) ──────────────────────────────
def schedule_fee(price: float, amount: float) -> float:
    return min(FEE_PER_CONTRACT, FEE_CAP_FRAC * price) * amount


def fee_check_3b(roll: dict) -> dict:
    legs = roll.get("exec_legs") or {}
    per_leg, passed = {}, True
    for n in FLY_LEGS:
        lg = legs.get(n) or {}
        obs, px, amt = lg.get("fee_observed_btc"), lg.get("average_price"), lg.get("filled_amount")
        if obs is None or px is None or amt is None:
            per_leg[n] = {"fee_observed_btc": obs, "abs_err_btc": None, "ok": False}
            passed = False
            continue
        err = abs(float(obs) - schedule_fee(float(px), float(amt)))
        ok = err <= FEE_TOL_BTC
        per_leg[n] = {"fee_observed_btc": obs, "schedule_btc": schedule_fee(float(px), float(amt)),
                      "abs_err_btc": err, "ok": ok}
        passed = passed and ok
    return {"passed": passed, "legs": per_leg}


# ────────────────────────────── ②b (P5) ──────────────────────────────
def fill_span_2b(rolls: list[dict]) -> dict:
    spans = [float(r["fill_span_s"]) for r in rolls
             if r.get("fill_timing_source") == "exchange_trades" and r.get("fill_span_s") is not None]
    receipt = [float(r["receipt_span_s"]) for r in rolls if r.get("receipt_span_s") is not None]
    out = {"n_measured": len(spans), "n_rolls": len(rolls), "spans_s": spans,
           "receipt_span_median_s": float(np.median(receipt)) if receipt else None}
    if len(spans) < N_MEASURED_MIN:
        out.update({"median_s": None, "status": "NESSUNA CONCLUSIONE"})
    else:
        med = float(np.median(spans))
        out.update({"median_s": med, "status": "PASS" if med <= FILL_SPAN_MAX_S else "FAIL"})
    return out


# ─────────────────────────── c_roll (P6) ───────────────────────────
def last_fill_ts(roll: dict) -> pd.Timestamp | None:
    # time of the LAST entry fill. The position persists per-leg local RECEIPT
    # times (exchange fill times are not persisted per leg): the receipt of the
    # last leg follows its fill by one round-trip, far below the ±10 min
    # snapshot tolerance.
    rt = roll.get("receipt_ts") or {}
    ts = [_ts(v) for v in rt.values() if v is not None]
    return max(ts) if ts else None


def load_snapshot(t: pd.Timestamp, instruments: list[str], chain_dir: Path | None = None) -> pd.DataFrame | None:
    # rows of the chain snapshot NEAREST to t within ±SNAPSHOT_TOL_MIN, restricted
    # to the given instruments; None if no snapshot is close enough.
    chain_dir = CHAIN_DIR if chain_dir is None else chain_dir
    tol = pd.Timedelta(minutes=SNAPSHOT_TOL_MIN)
    days = sorted({(t - tol).strftime("%Y%m%d"), (t + tol).strftime("%Y%m%d")})
    frames = [pd.read_parquet(p) for d in days if (p := chain_dir / f"btc_options_{d}.parquet").exists()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    snaps = df["snapshot_ts"].drop_duplicates()
    snaps = snaps[(snaps - t).abs() <= tol]
    if snaps.empty:
        return None
    best = snaps.iloc[int(np.argmin((snaps - t).abs().to_numpy()))]
    return df[(df["snapshot_ts"] == best) & df["instrument_name"].isin(instruments)]


def c_roll(quotes: dict, amount: float, tau_s: float, half_spread_btc: dict | None = None) -> dict | None:
    # FROZEN definition (amendment 1, P6). quotes[leg] = snapshot row with mark_price,
    # bid_price, ask_price, mark_iv (percent), underlying_price, strike, expiry,
    # snapshot_ts. Premium and schedule fees at the snapshot MARKS; half-spread
    # (ask − bid)/2 per leg (or the override, for the robustness reading);
    # second-order legging ½·φ(d₁)·σ·τ/√T on the two WINGS (BTC per contract,
    # τ and T in years). c_roll = (fee₄ + Σ half-spread + legging) / net premium.
    # None when any input is missing or degenerate — never a partial cost.
    if set(quotes) != set(FLY_LEGS):
        return None
    for n in FLY_LEGS:
        q = quotes[n]
        vals = [q.get("mark_price"), q.get("bid_price"), q.get("ask_price"), q.get("mark_iv"),
                q.get("underlying_price"), q.get("strike")]
        if any(v is None or not np.isfinite(float(v)) for v in vals):
            return None
        if half_spread_btc is None and not (float(q["bid_price"]) > 0.0 and
                                            float(q["ask_price"]) >= float(q["bid_price"])):
            return None
    net = (quotes["call"]["mark_price"] + quotes["put"]["mark_price"]
           - quotes["wing_call"]["mark_price"] - quotes["wing_put"]["mark_price"]) * amount
    if not net > 0.0:
        return None
    fee4 = sum(schedule_fee(float(quotes[n]["mark_price"]), amount) for n in FLY_LEGS)
    if half_spread_btc is None:
        hs = sum((float(quotes[n]["ask_price"]) - float(quotes[n]["bid_price"])) / 2.0 * amount
                 for n in FLY_LEGS)
    else:
        if any(half_spread_btc.get(n) is None for n in FLY_LEGS):
            return None
        hs = sum(float(half_spread_btc[n]) * amount for n in FLY_LEGS)
    tau_y = tau_s / 3600.0 / HOURS_PER_YEAR
    legging = 0.0
    for n in ("wing_call", "wing_put"):
        q = quotes[n]
        t_y = (_ts(q["expiry"]) - _ts(q["snapshot_ts"])).total_seconds() / 3600.0 / HOURS_PER_YEAR
        sig = float(q["mark_iv"]) / 100.0
        if not (t_y > 0.0 and sig > 0.0):
            return None
        d1 = (math.log(float(q["underlying_price"]) / float(q["strike"])) + 0.5 * sig * sig * t_y) \
            / (sig * math.sqrt(t_y))
        phi = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        legging += 0.5 * phi * sig * tau_y / math.sqrt(t_y) * amount
    return {"c_roll": (fee4 + hs + legging) / net, "net_premium_btc": net, "fee4_btc": fee4,
            "half_spread_btc": hs, "legging_btc": legging, "tau_s": tau_s}


def roll_quotes(roll: dict, chain_dir: Path | None = None) -> tuple[dict | None, str | None]:
    # the snapshot quotes of the FILLED instruments; (None, cause) if unavailable.
    legs = roll.get("exec_legs") or {}
    inst = {n: (legs.get(n) or {}).get("instrument") for n in FLY_LEGS}
    if any(v is None for v in inst.values()):
        return None, "strumenti dei fill assenti / fill instruments missing"
    t = last_fill_ts(roll)
    if t is None:
        return None, "timestamp dell'ultimo fill assente / last fill timestamp missing"
    snap = load_snapshot(t, list(inst.values()), chain_dir)
    if snap is None:
        return None, f"nessuno snapshot entro ±{SNAPSHOT_TOL_MIN:.0f} min / no snapshot within tolerance"
    rows = {r["instrument_name"]: r for r in snap.to_dict("records")}
    if any(v not in rows for v in inst.values()):
        return None, "strumento assente dallo snapshot / instrument missing from snapshot"
    return {n: rows[inst[n]] for n in FLY_LEGS}, None


def testnet_fill_vs_book(roll: dict, exec_diag: list[dict]) -> dict | None:
    # NOT gating. From the first exec_diag row logged on the open position in the
    # entry tick: per leg, the testnet REALISED half-spread = signed distance of the
    # fill from the testnet mid (sell below mid / buy above mid = positive cost),
    # used by the ① robustness reading; and the reported fill − testnet mark Δ.
    t0 = _ts(roll["entry_ts"])
    for r in exec_diag:
        if r.get("source") != "position" or r.get("expiry_ms") != roll.get("expiry_ms") \
           or float(r.get("strike", -1)) != float(roll.get("strike", -2)):
            continue
        dt = (_ts(r["ts"]) - t0).total_seconds()
        if not (-60.0 <= dt <= 900.0):
            continue
        by_inst = {l.get("instrument"): l for l in r.get("legs") or []}
        hs, dmark = {}, {}
        for n in FLY_LEGS:
            lg = (roll.get("exec_legs") or {}).get(n) or {}
            q = by_inst.get(lg.get("instrument"))
            if q is None or q.get("bid") is None or q.get("ask") is None \
               or q.get("mark") is None or lg.get("average_price") is None:
                return None
            mid = (q["bid"] + q["ask"]) / 2.0
            px = float(lg["average_price"])
            hs[n] = (mid - px) if LEG_SIDE[n] == "sell" else (px - mid)
            dmark[n] = px - q["mark"]
        return {"half_spread_btc": hs, "fill_minus_mark_btc": dmark}
    return None


def band_report(adaptive: list[dict], exec_diag: list[dict], go_live: pd.Timestamp) -> dict:
    # NOT gating (pre-reg «reported»): time share per band, band transitions, flat
    # days (no tick holding or opening a position), daily-band entries and — only
    # if the daily band activated — its entry half-spread against the pre-go-live
    # exec_diag history.
    recs = sorted(adaptive, key=lambda r: _ts(r["ts"]))
    bands = [r["band"] for r in recs if r.get("band")]
    share = pd.Series(bands).value_counts(normalize=True).round(4).to_dict() if bands else {}
    transitions = int(sum(a != b for a, b in zip(bands, bands[1:])))
    busy = {"ADAPT_HOLD", "ADAPT_FLY", "ADAPT_SHORT_DAILY", "ADAPT_INCOMPLETE", "ADAPT_BLOCKED"}
    days = {}
    for r in recs:
        d = _ts(r["ts"]).strftime("%Y-%m-%d")
        days[d] = days.get(d, False) or r.get("action") in busy
    daily_ts = [_ts(r["ts"]) for r in recs if r.get("action") == "ADAPT_SHORT_DAILY"]
    out = {"band_share": share, "band_transitions": transitions,
           "flat_days": sum(not v for v in days.values()), "days_observed": len(days),
           "daily_entries": len(daily_ts)}
    if daily_ts:
        def _hs(rows):
            v = [r["half_spread_frac"] for r in rows if r.get("half_spread_frac") is not None]
            return float(np.median(v)) if v else None
        entry_rows = [r for r in exec_diag if r.get("source") == "position" and
                      any(0.0 <= (_ts(r["ts"]) - t).total_seconds() <= 900.0 for t in daily_ts)]
        hist_rows = [r for r in exec_diag if _ts(r["ts"]) < go_live]
        out["daily_entry_half_spread_frac_median"] = _hs(entry_rows)
        out["historical_half_spread_frac_median"] = _hs(hist_rows)
    return out


# ─────────────────────────── evaluation (P4) ───────────────────────────
def evaluate(inp: dict, chain_dir: Path | None = None, count_only: bool = False) -> dict:
    res = {"preregistration": "FT1 2026-09-02 + emendamento 1 2026-09-15"}
    if not inp["adaptive"]:
        res["verdict"] = "NON AVVIATO / NOT STARTED"
        return res
    go_live = min(_ts(r["ts"]) for r in inp["adaptive"])
    horizon = max(_ts(r["ts"]) for r in inp["adaptive"])
    window_end = go_live + pd.Timedelta(weeks=WINDOW_WEEKS)
    attempts = collect_attempts(inp)
    in_win = [a for a in attempts if a["entry_ts"] is not None and go_live <= a["entry_ts"] < window_end]
    fly = [a for a in in_win if a["structure"] == FLY and a["executed"]]
    res.update({"go_live": str(go_live), "window_end": str(window_end), "data_horizon": str(horizon),
                "fridays": friday_calendar(inp["adaptive"], go_live, min(horizon + pd.Timedelta(hours=1), window_end)),
                "fly_attempts": len(fly), "fly_complete": sum(a["complete"] for a in fly),
                "fly_incomplete": sum(not a["complete"] for a in fly),
                "excluded_not_executed": sum(1 for a in in_win if a["structure"] == FLY and not a["executed"]),
                "daily_incomplete": {k: sum(1 for a in in_win if a["structure"] == DAILY
                                            and a.get("recovery") == k)
                                     for k in ("verified_flat", "blocked_operator_review")},
                "reported": band_report(inp["adaptive"], inp["exec_diag"], go_live),
                "journal_on_disk": inp["journal"] is not None})
    issues = integrity_issues(inp, attempts, go_live)
    if issues:
        res.update({"integrity_issues": issues,
                    "verdict": "INTEGRITY_ERROR — review manuale, nessun verdetto / manual review, no verdict"})
        return res
    if count_only:
        res["verdict"] = "COUNT_ONLY"
        return res

    # chronological walk: the first closing event wins (P4).
    completes = []
    for a in fly:
        if not a["complete"]:
            res.update({"verdict": "CHIUSO FAIL su ②a / CLOSED FAIL on ②a",
                        "fail_2a": {"entry_ts": str(a["entry_ts"]), "recovery": a.get("recovery"),
                                    "reason": a.get("reason"), "attempt_id": a.get("attempt_id"),
                                    "source": a["source"],
                                    "legs_filled": a["rec"].get("legs_filled")}})
            return res
        completes.append(a["rec"])
        if len(completes) == 1:
            fc = fee_check_3b(completes[0])
            res["check_3b"] = fc
            if not fc["passed"]:
                res["verdict"] = "NESSUNA CONCLUSIONE — ③b non superato / NO CONCLUSION — ③b not passed"
                return res
        if len(completes) == N_ROLLS_READ:
            break

    if len(completes) < N_ROLLS_READ:
        res["verdict"] = ("NESSUNA CONCLUSIONE — < 8 roll farfalla completi in 16 settimane / "
                          "NO CONCLUSION — < 8 complete butterfly rolls in 16 weeks"
                          if horizon >= window_end else "IN CORSO / IN PROGRESS")
        return res

    # reading at the 8th complete roll.
    rolls = completes
    per_roll, costs, costs_tau120, costs_testnet = [], [], [], []
    for r in rolls:
        measured = r.get("fill_timing_source") == "exchange_trades" and r.get("fill_span_s") is not None
        tau = float(r["fill_span_s"]) if measured else FILL_TIMEOUT_S
        quotes, cause = roll_quotes(r, chain_dir)
        amt = float(r.get("amount", 1.0))
        c = c_roll(quotes, amt, tau) if quotes else None
        c120 = c_roll(quotes, amt, FILL_TIMEOUT_S) if quotes else None
        tn = testnet_fill_vs_book(r, inp["exec_diag"])
        ctn = c_roll(quotes, amt, tau, tn["half_spread_btc"]) if quotes and tn else None
        per_roll.append({"entry_ts": r.get("entry_ts"), "tau_s": tau, "tau_measured": measured,
                         "c_roll": c, "unavailable": cause if quotes is None else
                         (None if c else "input degenere / degenerate input"),
                         "testnet": tn})
        if c:
            costs.append(c["c_roll"])
        if c120:
            costs_tau120.append(c120["c_roll"])
        if ctn:
            costs_testnet.append(ctn["c_roll"])
    med = float(np.median(costs)) if len(costs) >= N_MEASURED_MIN else None
    cond1 = ("NESSUNA CONCLUSIONE" if med is None else ("PASS" if med <= C_STAR else "FAIL"))
    cond2b = fill_span_2b(rolls)
    res.update({"rolls": per_roll,
                "cond_1": {"status": cond1, "median_c_roll": med, "n_available": len(costs),
                           "threshold": C_STAR, "ex_ante": C_ROLL_EX_ANTE,
                           "surprising": bool(med is not None and med > C_ROLL_SURPRISE),
                           "robustness_tau_120s": float(np.median(costs_tau120)) if costs_tau120 else None,
                           "robustness_testnet_half_spread": float(np.median(costs_testnet)) if costs_testnet else None,
                           "robustness_n": [len(costs_tau120), len(costs_testnet)]},
                "cond_2a": {"status": "PASS", "claim": (
                    f"0 roll farfalla incompleti su {N_ROLLS_READ}, compatibile con un tasso di "
                    f"incompletezza per roll ≤ {INCOMPLETE_RATE_BOUND:.0%} (95% unilaterale) / "
                    f"0 incomplete butterfly rolls out of {N_ROLLS_READ}, compatible with a per-roll "
                    f"incompleteness rate ≤ {INCOMPLETE_RATE_BOUND:.0%} (95% one-sided)")},
                "cond_2b": cond2b})
    statuses = {"①": cond1, "②b": cond2b["status"]}
    fails = [k for k, v in statuses.items() if v == "FAIL"]
    undecided = [k for k, v in statuses.items() if v == "NESSUNA CONCLUSIONE"]
    if fails:
        res["verdict"] = f"CHIUSO FAIL su {' e '.join(fails)} / CLOSED FAIL on {' and '.join(fails)}"
    elif undecided:
        res["verdict"] = f"NESSUNA CONCLUSIONE su {' e '.join(undecided)} / NO CONCLUSION on {' and '.join(undecided)}"
    else:
        res["verdict"] = "CHIUSO PASS / CLOSED PASS"
    return res


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Giudice FT1 (esecuzione struttura adattiva) / FT1 judge")
    # --count-only = safe monitoring: attempt counts and the Friday calendar only,
    # no cost, no timing, no verdict and no report, at ANY sample size.
    ap.add_argument("--count-only", action="store_true",
                    help="solo conteggi, nessun verdetto e nessun report / counts only, no verdict, no report")
    args = ap.parse_args()

    res = evaluate(load_inputs(), count_only=args.count_only)
    print("=" * 78)
    print("FT1 — ESECUZIONE DELLA STRUTTURA ADATTIVA / ADAPTIVE STRUCTURE EXECUTION")
    print("=" * 78)
    if "go_live" in res:
        print(f"go-live {res['go_live']} · fine finestra/window end {res['window_end']} · "
              f"dati fino a/data through {res['data_horizon']}")
        print(f"tentativi farfalla/butterfly attempts {res['fly_attempts']} · completi/complete "
              f"{res['fly_complete']} (lettura/reading at {N_ROLLS_READ}) · incompleti/incomplete "
              f"{res['fly_incomplete']} · non eseguiti esclusi/non-executed excluded {res['excluded_not_executed']}")
        causes = pd.Series([f["action"] for f in res["fridays"]]).value_counts().to_dict() if res["fridays"] else {}
        print(f"venerdì 08 UTC per esito/Fridays by outcome: {causes}")
        print(f"banda daily incompleti/daily band incomplete: {res['daily_incomplete']} · "
              f"journal su disco/on disk: {res['journal_on_disk']}")
        print(f"riportati, non gating/reported, not gating: {res['reported']}")
    for k in ("integrity_issues",):
        for i in res.get(k, []):
            print(f"  ! {i}")
    if args.count_only:
        if res["verdict"] != "COUNT_ONLY":
            print(f"\nSTATO / STATE: {res['verdict']}")
        print("\nCOUNT_ONLY — nessun costo, nessun tempo, nessun verdetto, nessun report scritto.")
        return 0
    if "cond_1" in res:
        c1 = res["cond_1"]
        print(f"\n① c_roll mediano {c1['median_c_roll']} (n={c1['n_available']}, soglia {C_STAR}) → {c1['status']} "
              f"· ex-ante {C_ROLL_EX_ANTE} · robustezza τ=120s {c1['robustness_tau_120s']} · "
              f"½-spread testnet {c1['robustness_testnet_half_spread']}")
        c2 = res["cond_2b"]
        print(f"②b fill_span mediano {c2['median_s']} s (misurati {c2['n_measured']}/{c2['n_rolls']}) → {c2['status']}")
    if "check_3b" in res:
        print(f"③b fee per gamba / per-leg fee: {'PASS' if res['check_3b']['passed'] else 'NON SUPERATO'}")
    print(f"\nVERDETTO / VERDICT: {res['verdict']}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(f"report -> {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
