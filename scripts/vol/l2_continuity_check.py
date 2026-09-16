# L2 RECORDER CONTINUITY MONITOR — watches the one project resource that grows with
# time rather than with experiments.
# Why FRESHNESS is not enough: the order-book line's usable sample is not made of
# hours collected but of CONTIGUOUS hours. A window needs T+h consecutive bars, so
# ONE hour of gap does not cost one hour: it costs T+h-1 windows, i.e. ~149 windows
# = 6.2 days of accrual at T=120/h=30. The routine's block ③ only checked the last
# file's freshness: a fresh file with a gap INSIDE passed silently.
# The precedent is documented: the "home" epoch (2026-06-16 -> 07-18) collected 32
# days at 29% coverage and produced ZERO usable windows - no contiguous run ever
# reached 150 hours. That data does not exist.
# Read-only, timestamps only: no feature values, no relationship with the target.
#
# ⚠ WHAT THIS CHECK ACTUALLY MEASURES (clarified 2026-08-03, after a false
# positive that cost a wrong STATUS.md entry): it measures the continuity of the
# LOCAL MIRROR, not that of the VPS recorder. They agree only on hours already
# CONSOLIDATED. Two ways the mirror's tail lies:
#   (a) the IN-PROGRESS hour is partial by construction — the span's last hour is
#       the one containing the last tick, so it holds fewer than 720 snapshots;
#       with a 360 threshold the verdict depended on the MINUTE the routine ran
#       (before minute ~30 -> "current run 0h", after -> all green);
#   (b) the pull fetches dailies over scp with no remote atomicity, so the last
#       hours can lag by one pull.
# Hence: the in-progress hour is EXCLUDED from the span, and gaps falling in the
# recent tail are flagged PROVISIONAL and excluded from the cost estimate. A gap
# is a fact only after it survives a second pull.
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

L2_GLOB = "data/orderbook/l2_features_*.parquet"
MIN_SNAP_PER_HOUR = 360     # same threshold as the B1 judge
T_WIN, H = 120, 30          # production configuration
# width of the tail in which a gap stays PROVISIONAL (pull lag, not data loss).
PROVISIONAL_H = 6


def runs_of_true(mask: np.ndarray) -> list[int]:
    out, c = [], 0
    for v in mask:
        if v:
            c += 1
        else:
            out.append(c)
            c = 0
    out.append(c)
    return [r for r in out if r > 0]


def analyze(ts: pd.DatetimeIndex, days: int, provisional_h: int = PROVISIONAL_H) -> dict:
    # split out of main() to be testable on synthetic series: the defect behind the
    # false positive lived in the span construction, not in the I/O.
    per_h = pd.Series(1, index=ts).resample("1h").sum()
    span = pd.date_range(ts[0].floor("h"), ts[-1].floor("h"), freq="1h")
    # the span's last hour CONTAINS the last tick -> in progress, never complete.
    span, in_progress = span[:-1], span[-1]
    if len(span) == 0:
        return {"empty": True, "in_progress": in_progress}

    ok = per_h.reindex(span, fill_value=0).values >= MIN_SNAP_PER_HOUR
    runs = runs_of_true(ok)

    # the CURRENT run (ending at the last consolidated hour) is the only one that can
    # still grow: it is the quantity to protect, not the historical maximum.
    cur = 0
    for v in ok[::-1]:
        if not v:
            break
        cur += 1

    def windows(rs):
        return int(sum(max(r - (T_WIN + H) + 1, 0) for r in rs))

    idx = np.where(~ok & (span >= span[-1] - pd.Timedelta(days=days)))[0]
    blocks: list[tuple[int, int]] = []
    for i in idx:
        if blocks and i == blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], i)
        else:
            blocks.append((int(i), int(i)))
    # a gap touching the recent tail may be mere pull lag -> provisional.
    tail_start = span[-1] - pd.Timedelta(hours=provisional_h - 1)
    firm = [g for g in blocks if span[g[1]] < tail_start]
    prov = [g for g in blocks if span[g[1]] >= tail_start]

    return {
        "empty": False, "span": span, "in_progress": in_progress, "ok": ok,
        "coverage": float(ok.mean()), "cur": cur, "max_run": max(runs) if runs else 0,
        "windows": windows(runs), "n_eff": windows(runs) / H,
        "firm": firm, "prov": prov,
    }


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Monitor di continuita' del recorder L2 / L2 recorder continuity monitor")
    ap.add_argument("--days", type=int, default=7,
                    help="finestra recente per l'elenco buchi / recent window for the gap list")
    ap.add_argument("--provisional-hours", type=int, default=PROVISIONAL_H,
                    help="coda in cui un buco resta PROVVISORIO (ritardo di pull) / "
                         "tail in which a gap stays PROVISIONAL (pull lag)")
    args = ap.parse_args()

    files = sorted(glob.glob(L2_GLOB))
    if not files:
        print("[l2] nessun file order-book / no order-book file")
        return 0

    ts = pd.concat([pd.read_parquet(f, columns=["timestamp"]) for f in files])["timestamp"]
    ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).sort_values()
    r = analyze(ts, args.days, args.provisional_hours)
    if r["empty"]:
        print("[l2] meno di un'ora consolidata / less than one consolidated hour")
        return 0
    span, gaps, prov = r["span"], r["firm"], r["prov"]

    print(f"[l2] copertura/coverage {r['coverage']:.1%} su {len(span)} ore consolidate/"
          f"consolidated · run corrente/current run {r['cur']}h · run max {r['max_run']}h "
          f"(ora in corso {r['in_progress']:%H:%M} UTC esclusa/in-progress hour excluded)")
    print(f"[l2] finestre T={T_WIN}/h={H}: {r['windows']} · n_eff {r['n_eff']:.1f} "
          f"(gate vol: 216) · costo di 1h di buco: {T_WIN + H - 1} finestre "
          f"= {(T_WIN + H - 1) / H / (24 / H):.1f} giorni/days")

    # ── recent gaps ─────────────────────────────────────────────────────────
    if prov:
        # NOT an alarm: an unconsolidated tail. Judge it at the next pull. Printing it
        # as a loss is the 2026-08-02 mistake.
        tot_p = sum(b - a + 1 for a, b in prov)
        print(f"[l2] ~ {len(prov)} buco/i PROVVISORIO/I nelle ultime {args.provisional_hours}h "
              f"({tot_p}h) — ritardo di pull finche' non sopravvive a un secondo pull / "
              f"pull lag until it survives a second pull:")
        for a, b in prov:
            print(f"[l2]    {span[a]:%Y-%m-%d %H:%M} UTC · {b - a + 1}h (provvisorio/provisional)")
    if gaps:
        tot = sum(b - a + 1 for a, b in gaps)
        print(f"[l2] ⚠ {len(gaps)} BUCHI CONSOLIDATI negli ultimi {args.days}g ({tot}h perse) — "
              f"ogni buco spezza il run / each consolidated gap breaks the run:")
        for a, b in gaps[-10:]:
            print(f"[l2]    {span[a]:%Y-%m-%d %H:%M} UTC · {b - a + 1}h")
        # cost is counted PER BLOCK, not per hour: a 41-hour gap breaks the run ONCE,
        # destroying the T+H-1 windows that would have straddled it, not 41 times
        # that. Add the hours' own lost production (1 window/hour). Counting per hour
        # overstated by an order of magnitude and would have made the warning useless
        # because visibly absurd.
        lost = len(gaps) * (T_WIN + H - 1) + tot
        print(f"[l2] ⚠ costo stimato/estimated cost: ~{lost} finestre = ~{lost / H:.0f} n_eff "
              f"= ~{lost / H / (24 / H):.0f} giorni di accumulo / days of accrual "
              f"({len(gaps)} rotture x {T_WIN + H - 1} + {tot}h non prodotte)")
    elif not prov:
        print(f"[l2] nessun buco negli ultimi {args.days}g / no gap in the last {args.days}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
