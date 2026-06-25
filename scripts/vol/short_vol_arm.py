"""
short_vol_arm.py — simulatore OFFLINE del 2° braccio SHORT-VOL del forward test (vol-line).
short_vol_arm.py — OFFLINE simulator of the forward test's 2nd SHORT-VOL arm (vol line).

IT: NON tocca il forward test live (04b): legge SOLO dati già raccolti — chain IV
    (data/iv/chain/*.parquet), forecasts del modello (results/vol_paper/forecasts.parquet)
    e delivery prices (cache + Deribit public). Per ogni scadenza giornaliera D (08:00 UTC)
    con delivery noto, entra ~TENOR_H ore prima vendendo uno STRANGLE OTM (call a +w%, put
    a −w%) o uno STRADDLE ATM, e regola al delivery (opzioni INVERSE: payoff_leg = max(0,
    S−K)/S_del per la call, max(0, K−S)/S_del per la put; PnL short = premio_incassato −
    payoff − fee). Confronta: (A) ALWAYS-SHORT (baseline da battere) vs (B) NN-TIMED.
    Hold-to-expiry ⇒ il settlement è deterministico ⇒ la sim == ciò che darebbe il testnet.
EN: does NOT touch the live forward test (04b): reads ONLY already-collected data — IV chain,
    model forecasts and delivery prices. For each daily expiry D (08:00 UTC) with a known
    delivery, enters ~TENOR_H hours earlier SELLING an OTM STRANGLE (call at +w%, put at −w%)
    or an ATM STRADDLE, settling at delivery (INVERSE options). Compares ALWAYS-SHORT (the
    baseline to beat) vs NN-TIMED. Hold-to-expiry ⇒ deterministic settlement ⇒ sim == testnet.

Uso / usage:  python scripts/vol/short_vol_arm.py [--width 0.06] [--struct strangle|straddle]
              python scripts/vol/short_vol_arm.py --sweep   # tabella sensitività completa
"""
import sys, json, argparse
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chain_io import load_chain  # noqa: E402  (loader chain condiviso, A3 | shared chain loader)

ROOT = Path(__file__).resolve().parents[2]
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
FC_PATH = ROOT / "results" / "vol_paper" / "forecasts.parquet"
DELIV_CACHE = ROOT / "results" / "vol_paper" / "delivery_cache.json"
OUT = ROOT / "results" / "vols" / "short_vol_arm.json"

TENOR_H = 30.0           # IT: tenor target (h) come il forward test | EN: target tenor (h)
FEE_PER_LEG = 0.0003     # IT: fee testnet ~0.0003 BTC/contratto/leg | EN: ~0.0003 BTC/contract/leg
FEE_CAP_FRAC = 0.125     # IT: cap Deribit = 12.5% del premio per leg | EN: Deribit cap = 12.5% of premium per leg
SIZE = 1.0               # contratti per leg / contracts per leg
ENTRY_TOL_H = 6.0        # IT: tolleranza match snapshot↔entry_t | EN: snapshot↔entry_t match tolerance


def fee_btc(premium: float) -> float:
    # IT: fee taker per leg, cap al 12.5% del premio (schema Deribit, identico a 04b live).
    # EN: per-leg taker fee, capped at 12.5% of premium (Deribit schema, identical to 04b live).
    # IT: NECESSARIO: su leg OTM economiche (prem ~3e-4 BTC) la fee flat = 100% del premio → bug.
    # EN: REQUIRED: on cheap OTM legs (prem ~3e-4 BTC) the flat fee = 100% of premium → bug.
    return min(FEE_PER_LEG, FEE_CAP_FRAC * premium) * SIZE


def fetch_delivery(date_key: str, cache: dict) -> float | None:
    # IT: delivery price (08:00 UTC) — cache poi Deribit public. | EN: delivery price, cache then public.
    if date_key in cache:
        return float(cache[date_key])
    try:
        url = "https://www.deribit.com/api/v2/public/get_delivery_prices?index_name=btc_usd&count=60"
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())["result"]["data"]
        for rec in data:
            d = pd.Timestamp(rec["date"]).strftime("%d%b%y").upper()
            cache[d] = float(rec["delivery_price"])
        return cache.get(date_key)
    except Exception as e:
        print(f"  ! delivery fetch fail {date_key}: {e}")
        return None


def settle_payoff(opt_type: str, K: float, S_del: float) -> float:
    # IT: payoff opzione INVERSA per contratto (BTC). | EN: INVERSE option payoff per contract (BTC).
    intrinsic = max(0.0, S_del - K) if opt_type == "C" else max(0.0, K - S_del)
    return intrinsic / S_del if S_del > 0 else 0.0


def pick_leg(snap: pd.DataFrame, opt_type: str, target_K: float):
    # IT: strumento col tipo richiesto e strike più vicino al target, con premio mark valido.
    # EN: instrument of the requested type with strike nearest target and a valid mark premium.
    # IT: salva sia mark sia bid: il fill SHORT reale avviene al BID (vendiamo). bid mancante → NaN.
    # EN: keeps both mark and bid: a real SHORT fill is at the BID (we sell). missing bid → NaN.
    cand = snap[(snap["option_type"].str.upper().str[0] == opt_type) & (snap["mark_price"] > 0)]
    if cand.empty:
        return None
    row = cand.iloc[(cand["strike"] - target_K).abs().argmin()]
    bid = float(row["bid_price"]) if pd.notna(row["bid_price"]) else np.nan
    return {"K": float(row["strike"]), "prem": float(row["mark_price"]),
            "bid": bid, "iv": float(row["mark_iv"])}


def simulate(width: float, struct: str, chain=None, cache=None, fc=None):
    # IT: chain/cache/fc opzionali per riuso nello sweep (evita re-load 12 parquet per config).
    # EN: chain/cache/fc optional for reuse in the sweep (avoids re-loading 12 parquet per config).
    if chain is None:
        chain = load_chain()
    if cache is None:
        cache = json.loads(DELIV_CACHE.read_text()) if DELIV_CACHE.exists() else {}
    if fc is None:
        fc = pd.read_parquet(FC_PATH) if FC_PATH.exists() else None
        if fc is not None:
            fc["candle_ts"] = pd.to_datetime(fc["candle_ts"], utc=True)

    # IT: scadenze giornaliere candidate = expiry presenti nella chain, 08:00 UTC.
    # EN: candidate daily expiries = expiries present in the chain, at 08:00 UTC.
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
        # snapshot di ingresso più vicino a entry_t per QUESTA scadenza
        sub = chain[chain["expiry"] == exp]
        snaps = sub["snapshot_ts"].unique()
        snaps = snaps[(snaps >= snap_min)]
        if len(snaps) == 0:
            continue
        s_entry = min(snaps, key=lambda s: abs((pd.Timestamp(s) - entry_t).total_seconds()))
        dt_entry_h = (pd.Timestamp(s_entry) - entry_t).total_seconds() / 3600.0
        if abs(dt_entry_h) > ENTRY_TOL_H:
            continue  # nessuno snapshot entro ±ENTRY_TOL_H dal tenor target
        # IT: tenor effettivo (h) dallo snapshot scelto — varia perché il poller è rado.
        # EN: actual tenor (h) at the chosen snapshot — varies because the poller is sparse.
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
        # IT: premio MARK (riferimento) e premio BID (fill short realistico).
        # EN: MARK premium (reference) and BID premium (realistic short fill).
        prem = (leg_c["prem"] + leg_p["prem"]) * SIZE
        bid_c, bid_p = leg_c["bid"], leg_p["bid"]
        prem_bid = ((bid_c if np.isfinite(bid_c) else leg_c["prem"]) +
                    (bid_p if np.isfinite(bid_p) else leg_p["prem"])) * SIZE
        payoff = (settle_payoff("C", leg_c["K"], S_del) +
                  settle_payoff("P", leg_p["K"], S_del)) * SIZE
        # IT: fee con cap 12.5% per leg (sul premio mark, come il book taker).
        # EN: fee with 12.5% per-leg cap (on mark premium, as the taker book).
        fees = fee_btc(leg_c["prem"]) + fee_btc(leg_p["prem"])
        pnl_short = prem - payoff - fees            # short = incassa premio, paga payoff
        pnl_short_bid = prem_bid - payoff - fees    # robustness: fill al bid
        # edge del modello al tick più vicino all'ingresso (per la variante NN-timed)
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
    # IT: statistiche di un set di trade su una colonna PnL. | EN: trade-set stats on a PnL column.
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
    # IT: REDESIGN NN-timing. L'edge è quasi-sempre >0 (long-biased) ⇒ una soglia assoluta
    #     non scatta MAI short. Regola corretta: SHORT solo quando l'edge è BASSO rispetto
    #     alla SUA stessa distribuzione (il modello è relativamente MENO long-vol = meno
    #     convinto che RV>IV) → percentile causale entro la finestra rolling delle entry.
    #     win=0 → mediana empirica espandente (tutte le entry passate, causale).
    # EN: NN-timing REDESIGN. Edge is almost always >0 (long-biased) ⇒ an absolute threshold
    #     NEVER fires short. Correct rule: go SHORT only when edge is LOW relative to its OWN
    #     distribution (model relatively LEAST long-vol) → causal percentile within the rolling
    #     window of entries. win=0 → expanding empirical median (all past entries, causal).
    d = df.dropna(subset=["edge"]).sort_values("entry").reset_index(drop=True).copy()
    if len(d) < 3:
        return d.iloc[0:0], np.array([], dtype=bool)
    take = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        hist = d["edge"].iloc[max(0, i - win) if win else 0: i]  # causale: SOLO entry precedenti
        if len(hist) < 2:
            continue  # warm-up: senza storia non si scommette (no look-ahead)
        thr = hist.quantile(q)
        take[i] = d["edge"].iloc[i] <= thr  # short se edge ≤ percentile causale
    return d[take], take


def run_sweep(chain, cache, fc):
    # IT: SENSITIVITÀ — struct × width, PnL mark vs bid (fill realistico). | EN: SENSITIVITY sweep.
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

    # IT: load una volta sola, riusa nello sweep (12 parquet ~450k righe). | EN: load once, reuse.
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

    # A) ALWAYS-SHORT (baseline da battere) — mark e bid
    report(df, "ALWAYS-SHORT (mark)", "pnl_short")
    report(df, "ALWAYS-SHORT (bid) ", "pnl_short_bid")

    # B) NN-TIMED redesign: short solo quando edge ≤ percentile causale della sua distribuzione
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
