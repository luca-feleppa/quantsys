"""
short_vol_arm.py — OFFLINE simulator of the forward test's 2nd SHORT-VOL arm (vol line).

Does NOT touch the live forward test (04b): reads ONLY already-collected data — IV chain
(data/iv/chain/*.parquet), model forecasts (results/vol_paper/forecasts.parquet) and
delivery prices (cache + Deribit public). For each daily expiry D (08:00 UTC) with a known
delivery, enters ~TENOR_H hours earlier SELLING an OTM STRANGLE (call at +w%, put at −w%)
or an ATM STRADDLE, settling at delivery (INVERSE options: payoff_leg = max(0, S−K)/S_del
for the call, max(0, K−S)/S_del for the put; short PnL = premium_collected − payoff − fee).
Compares (A) ALWAYS-SHORT (the baseline to beat) vs (B) NN-TIMED.
Hold-to-expiry ⇒ deterministic settlement ⇒ sim == what the testnet would give.

Usage:  python scripts/vol/short_vol_arm.py [--width 0.06] [--struct strangle|straddle]
        python scripts/vol/short_vol_arm.py --sweep   # full sensitivity table
"""
import sys, json, argparse
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chain_io import load_chain  # noqa: E402  (shared chain loader, A3)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.data.deribit import fetch_delivery_prices  # noqa: E402  (C2 2ter)

ROOT = Path(__file__).resolve().parents[2]
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
FC_PATH = ROOT / "results" / "vol_paper" / "forecasts.parquet"
# C2 2ter (2026-07-18) — OWN cache file (it used to share
# results/vol_paper/delivery_cache.json with 04c, which writes the paper
# venue's TESTNET deliveries there: different venues never in one file).
# First run with the new file: harmless refetch (production paging).
DELIV_CACHE = ROOT / "results" / "vol_paper" / "delivery_cache_prod.json"
OUT = ROOT / "results" / "vols" / "short_vol_arm.json"

TENOR_H = 30.0           # target tenor (h)
FEE_PER_LEG = 0.0003     # ~0.0003 BTC/contract/leg
FEE_CAP_FRAC = 0.125     # Deribit cap = 12.5% of premium per leg
SIZE = 1.0               # contracts per leg
ENTRY_TOL_H = 6.0        # snapshot↔entry_t match tolerance


def fee_btc(premium: float) -> float:
    # per-leg taker fee, capped at 12.5% of premium (Deribit schema, identical to 04b live).
    # REQUIRED: on cheap OTM legs (prem ~3e-4 BTC) the flat fee = 100% of premium → bug.
    return min(FEE_PER_LEG, FEE_CAP_FRAC * premium) * SIZE


def fetch_delivery(date_key: str, cache: dict) -> float | None:
    # delivery price (08:00 UTC) — in-memory cache (reused by the sweep)
    # then the shared C2 2ter helper (quantsys.data.deribit, production).
    if date_key in cache:
        return float(cache[date_key])
    try:
        cache.update(fetch_delivery_prices(count=60))
        return cache.get(date_key)
    except Exception as e:
        print(f"  ! delivery fetch fail {date_key}: {e}")
        return None


def settle_payoff(opt_type: str, K: float, S_del: float) -> float:
    # INVERSE option payoff per contract (BTC).
    intrinsic = max(0.0, S_del - K) if opt_type == "C" else max(0.0, K - S_del)
    return intrinsic / S_del if S_del > 0 else 0.0


def pick_leg(snap: pd.DataFrame, opt_type: str, target_K: float):
    # instrument of the requested type with strike nearest target and a valid mark premium.
    # keeps both mark and bid: a real SHORT fill is at the BID (we sell). missing bid → NaN.
    cand = snap[(snap["option_type"].str.upper().str[0] == opt_type) & (snap["mark_price"] > 0)]
    if cand.empty:
        return None
    row = cand.iloc[(cand["strike"] - target_K).abs().argmin()]
    bid = float(row["bid_price"]) if pd.notna(row["bid_price"]) else np.nan
    return {"K": float(row["strike"]), "prem": float(row["mark_price"]),
            "bid": bid, "iv": float(row["mark_iv"])}


def simulate(width: float, struct: str, chain=None, cache=None, fc=None):
    # chain/cache/fc optional for reuse in the sweep (avoids re-loading 12 parquet per config).
    if chain is None:
        chain = load_chain()
    if cache is None:
        cache = json.loads(DELIV_CACHE.read_text()) if DELIV_CACHE.exists() else {}
    if fc is None:
        fc = pd.read_parquet(FC_PATH) if FC_PATH.exists() else None
        if fc is not None:
            fc["candle_ts"] = pd.to_datetime(fc["candle_ts"], utc=True)

    # candidate daily expiries = expiries present in the chain, at 08:00 UTC.
    exps = sorted(e for e in chain["expiry"].unique()
                  if pd.Timestamp(e).hour == 8)
    snap_min, snap_max = chain["snapshot_ts"].min(), chain["snapshot_ts"].max()

    trades = []
    for exp in exps:
        exp = pd.Timestamp(exp)
        entry_t = exp - timedelta(hours=TENOR_H)
        if entry_t < snap_min or exp > snap_max + timedelta(hours=1):
            continue
        date_key = exp.strftime("%d%b%y").upper()
        S_del = fetch_delivery(date_key, cache)
        if S_del is None:
            continue
        # entry snapshot closest to entry_t for THIS expiry
        sub = chain[chain["expiry"] == exp]
        snaps = sub["snapshot_ts"].unique()
        snaps = snaps[(snaps >= snap_min)]
        if len(snaps) == 0:
            continue
        s_entry = min(snaps, key=lambda s: abs((pd.Timestamp(s) - entry_t).total_seconds()))
        dt_entry_h = (pd.Timestamp(s_entry) - entry_t).total_seconds() / 3600.0
        if abs(dt_entry_h) > ENTRY_TOL_H:
            continue  # no snapshot within ±ENTRY_TOL_H of the target tenor
        # actual tenor (h) at the chosen snapshot — varies because the poller is sparse.
        t_hours = (exp - pd.Timestamp(s_entry)).total_seconds() / 3600.0
        snap = sub[sub["snapshot_ts"] == s_entry]
        spot = float(snap["underlying_price"].iloc[0])
        if struct == "strangle":
            kc, kp = spot * (1 + width), spot * (1 - width)
        else:  # straddle ATM
            kc = kp = spot
        leg_c = pick_leg(snap, "C", kc)
        leg_p = pick_leg(snap, "P", kp)
        if not leg_c or not leg_p:
            continue
        # MARK premium (reference) and BID premium (realistic short fill).
        prem = (leg_c["prem"] + leg_p["prem"]) * SIZE
        bid_c, bid_p = leg_c["bid"], leg_p["bid"]
        prem_bid = ((bid_c if np.isfinite(bid_c) else leg_c["prem"]) +
                    (bid_p if np.isfinite(bid_p) else leg_p["prem"])) * SIZE
        payoff = (settle_payoff("C", leg_c["K"], S_del) +
                  settle_payoff("P", leg_p["K"], S_del)) * SIZE
        # fee with 12.5% per-leg cap (on mark premium, as the taker book).
        fees = fee_btc(leg_c["prem"]) + fee_btc(leg_p["prem"])
        pnl_short = prem - payoff - fees            # short = collects premium, pays payoff
        pnl_short_bid = prem_bid - payoff - fees    # robustness: fill at the bid
        # model edge at the tick closest to entry (for the NN-timed variant)
        edge = np.nan
        if fc is not None:
            near = fc.iloc[(fc["candle_ts"] - pd.Timestamp(s_entry)).abs().argmin()]
            if abs((near["candle_ts"] - pd.Timestamp(s_entry)).total_seconds()) <= 3 * 3600:
                edge = float(near["edge"])
        trades.append({
            "expiry": exp.isoformat(), "entry": pd.Timestamp(s_entry).isoformat(),
            "t_hours": t_hours, "spot": spot, "S_del": S_del,
            "Kc": leg_c["K"], "Kp": leg_p["K"], "iv_c": leg_c["iv"], "iv_p": leg_p["iv"],
            "prem": prem, "prem_bid": prem_bid, "payoff": payoff, "fees": fees,
            "pnl_short": pnl_short, "pnl_short_bid": pnl_short_bid, "edge": edge,
            "moved_pct": 100 * (S_del / spot - 1),
        })

    return pd.DataFrame(trades)


def stats(df: pd.DataFrame, col: str = "pnl_short") -> dict:
    # trade-set stats on a PnL column.
    if df.empty:
        return {"n": 0, "tot": 0.0, "mean": 0.0, "hit": 0.0}
    p = df[col]
    return {"n": len(df), "tot": float(p.sum()), "mean": float(p.mean()),
            "hit": float(100 * (p > 0).mean())}


def report(df: pd.DataFrame, label: str, col: str = "pnl_short"):
    s = stats(df, col)
    if s["n"] == 0:
        print(f"  [{label}] n=0"); return
    extra = ""
    if not df.empty:
        extra = f" | premIncass={df['prem'].sum():.4f} | payoutTot={df['payoff'].sum():.4f}"
    print(f"  [{label}] n={s['n']} | totPnL={s['tot']:+.5f} BTC | "
          f"mean={s['mean']:+.5f} | hit={s['hit']:.0f}%{extra}")


def nn_timed(df: pd.DataFrame, q: float = 0.50, win: int = 0):
    # NN-timing REDESIGN. Edge is almost always >0 (long-biased) ⇒ an absolute threshold
    # NEVER fires short. Correct rule: go SHORT only when edge is LOW relative to its OWN
    # distribution (model relatively LEAST long-vol) → causal percentile within the rolling
    # window of entries. win=0 → expanding empirical median (all past entries, causal).
    d = df.dropna(subset=["edge"]).sort_values("entry").reset_index(drop=True).copy()
    if len(d) < 3:
        return d.iloc[0:0], np.array([], dtype=bool)
    take = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        hist = d["edge"].iloc[max(0, i - win) if win else 0: i]  # causal: ONLY previous entries
        if len(hist) < 2:
            continue  # warm-up: no bet without history (no look-ahead)
        thr = hist.quantile(q)
        take[i] = d["edge"].iloc[i] <= thr  # short if edge ≤ causal percentile
    return d[take], take


def run_sweep(chain, cache, fc):
    # SENSITIVITY sweep.
    print("\n=== SENSITIVITY SWEEP (always-short) ===")
    hdr = f"  {'struct':<9} {'width':>6} | {'n':>2} | {'tot(mark)':>10} {'mean':>9} {'hit':>4} | {'tot(bid)':>10} {'mean':>9} {'hit':>4}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    configs = [("straddle", 0.0)] + [("strangle", w) for w in (0.04, 0.06, 0.08, 0.10)]
    for struct, w in configs:
        df = simulate(w, struct, chain=chain, cache=cache, fc=fc)
        sm, sb = stats(df, "pnl_short"), stats(df, "pnl_short_bid")
        tag = f"{w:.0%}" if struct == "strangle" else "ATM"
        print(f"  {struct:<9} {tag:>6} | {sm['n']:>2} | {sm['tot']:>+10.5f} {sm['mean']:>+9.5f} {sm['hit']:>3.0f}% | "
              f"{sb['tot']:>+10.5f} {sb['mean']:>+9.5f} {sb['hit']:>3.0f}%")
        rows.append({"struct": struct, "width": w, "mark": sm, "bid": sb})
    return rows


def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=0.06, help="OTM width (frazione) per lo strangle")
    ap.add_argument("--struct", default="strangle", choices=["strangle", "straddle"])
    ap.add_argument("--sweep", action="store_true", help="esegue lo sweep di sensitività completo")
    ap.add_argument("--nn-q", type=float, default=0.50, help="percentile causale per il gate NN-timed")
    ap.add_argument("--nn-win", type=int, default=0, help="finestra rolling NN-timed (0=espandente)")
    args = ap.parse_args()

    # load once, reuse.
    chain = load_chain()
    cache = json.loads(DELIV_CACHE.read_text()) if DELIV_CACHE.exists() else {}
    fc = pd.read_parquet(FC_PATH) if FC_PATH.exists() else None
    if fc is not None:
        fc["candle_ts"] = pd.to_datetime(fc["candle_ts"], utc=True)

    print(f"=== SHORT-VOL ARM (offline sim) · struct={args.struct} width={args.width:.0%} tenor={TENOR_H:.0f}h ===")
    df = simulate(args.width, args.struct, chain=chain, cache=cache, fc=fc)
    if df.empty:
        DELIV_CACHE.write_text(json.dumps(cache, indent=2))
        sys.exit("nessuna scadenza simulabile (chain/delivery insufficienti)")

    # A) ALWAYS-SHORT (baseline to beat) — mark and bid
    report(df, "ALWAYS-SHORT (mark)", "pnl_short")
    report(df, "ALWAYS-SHORT (bid) ", "pnl_short_bid")

    # B) NN-TIMED redesign: short only when edge ≤ causal percentile of its own distribution
    nn, _ = nn_timed(df, q=args.nn_q, win=args.nn_win)
    win_tag = "expand" if args.nn_win == 0 else f"win{args.nn_win}"
    report(nn, f"NN-TIMED (edge≤Q{args.nn_q:.0%} causale, {win_tag}, mark)", "pnl_short")
    report(nn, f"NN-TIMED (edge≤Q{args.nn_q:.0%} causale, {win_tag}, bid) ", "pnl_short_bid")

    if args.sweep:
        run_sweep(chain, cache, fc)

    DELIV_CACHE.write_text(json.dumps(cache, indent=2))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(df.to_json(orient="records", indent=2))
    print(f"\n  → dettaglio in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
