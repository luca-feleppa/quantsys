# IT: A13a — CONDIZIONE (3) EX-ANTE SUL PIN-CLOSE: quante posizioni AVREBBERO
#     attivato la chiusura anticipata, su una griglia di (max_hours, pin_band)?
#     Perche' PRIMA e non dopo: un gate forward su A13 costa ~1 mese di campione
#     (~1 posizione/giorno) e non e' avviabile prima di meta' agosto (due campioni
#     forward aperti). Se il pin-close scatta su 3 posizioni su 30, quel mese
#     produrrebbe "NESSUNA CONCLUSIONE" per difetto di conteggio - ed e' esattamente
#     la trappola che il protocollo impone di evitare misurando la condizione di
#     conteggio quando e' MODEL-INDEPENDENT (successo B2/B3 il 2026-07-19, e budget
#     campionario L2 il 2026-07-31).
#     ⚠ QUESTO NON E' UN GATE E NON CALCOLA NESSUNA PnL. Conta eventi, e basta:
#     strike, expiry e prezzo del sottostante sono fatti registrati, indipendenti
#     dal modello. Nessun controfattuale di rendimento viene prodotto qui, perche'
#     il campione e' gia' stato guardato (gate leg chiuso FAIL 0/3 il 2026-07-30) e
#     un numero di PnL calcolato ora non sarebbe evidenza, sarebbe post-hoc.
#     ⚠ La cella (X, f) del gate eventuale NON va scelta massimizzando qualcosa su
#     questa superficie: va fissata da una REGOLA A PRIORI dichiarata nella
#     pre-registrazione (es. la X e la f piu' piccole che garantiscono n >= n_min).
#     Il predicato e' importato da scripts/04b_vol_paper.py: si conta con la
#     funzione di PRODUZIONE, non con una sua copia che potrebbe divergere.
# EN: A13a — EX-ANTE CONDITION (3) ON PIN-CLOSE: how many positions WOULD have
#     triggered the early close, over a grid of (max_hours, pin_band)?
#     Why BEFORE and not after: a forward gate on A13 costs ~1 month of sample
#     (~1 position/day) and cannot start before mid-August (two open forward
#     samples). If pin-close fires on 3 positions out of 30, that month would yield
#     "NO CONCLUSION" for lack of count - exactly the trap the protocol avoids by
#     measuring the counting condition while it is MODEL-INDEPENDENT (B2/B3 success
#     on 2026-07-19, L2 sample budget on 2026-07-31).
#     ⚠ THIS IS NOT A GATE AND COMPUTES NO PnL. It counts events, nothing else:
#     strike, expiry and underlying price are recorded facts, independent of the
#     model. No return counterfactual is produced here, because the sample has
#     already been looked at (leg gate closed FAIL 0/3 on 2026-07-30) and a PnL
#     number computed now would not be evidence, it would be post-hoc.
#     ⚠ The eventual gate's (X, f) cell must NOT be picked by maximizing anything
#     on this surface: it must be fixed by an A PRIORI RULE declared in the
#     pre-registration (e.g. the smallest X and f granting n >= n_min).
#     The predicate is imported from scripts/04b_vol_paper.py: the count runs on the
#     PRODUCTION function, not on a copy of it that could drift.
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRADES = ROOT / "results" / "vol_paper" / "trades.jsonl"
CHAIN_GLOB = str(ROOT / "data" / "iv" / "chain" / "btc_options_*.parquet")
CANDLES_1M = ROOT / "data" / "raw_candles_1m_l2.parquet"
OUT_JSON = ROOT / "results" / "vol_paper" / "pin_close_feasibility.json"

# IT: cadenza reale del loop 04b: tick orario a hh:00 + 90s (AVVIO.md 5.3bis).
#     La condizione va valutata SOLO quando il processo puo' agire: valutarla in
#     continuo sovrastimerebbe gli inneschi.
# EN: real 04b loop cadence: hourly tick at hh:00 + 90s. The condition must be
#     evaluated ONLY when the process can act: evaluating it continuously would
#     overstate the triggers.
TICK_OFFSET_S = 90
SNAP_TOL = pd.Timedelta("10min")

# IT: griglia esplorativa, NON una scelta di parametri.
# EN: exploratory grid, NOT a parameter choice.
GRID_HOURS = [1.0, 2.0, 3.0, 4.0, 6.0]
GRID_BAND = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]

# IT: confine di epoca — il VPS ha preso i collector il 2026-07-14 (copertura
#     100%); prima il poller di casa girava al 18.6% delle ore. Una posizione
#     entrata nell'epoca "casa" puo' avere tick NON OSSERVABILI, e contarli come
#     "nessun innesco" falserebbe il conteggio verso il basso.
# EN: epoch boundary - the VPS took over the collectors on 2026-07-14 (100%
#     coverage); before that the home poller ran at 18.6% of hours. A position
#     entered in the "home" epoch may have UNOBSERVABLE ticks, and counting those
#     as "no trigger" would bias the count downward.
VPS_EPOCH = pd.Timestamp("2026-07-14", tz="UTC")


def _import_from_path(name: str, path: Path):
    # IT: import da path (nomi con cifre iniziali / fuori package).
    # EN: path-based import (digit-leading names / outside packages).
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_positions() -> pd.DataFrame:
    # IT: campione = solo executed=True, la stessa definizione pre-dichiarata del
    #     gate v1 (il trade #0 e' uno smoke pre-lancio designato non-campione da
    #     note contemporanee PRE-settlement).
    # EN: sample = executed=True only, the same pre-declared definition as gate v1
    #     (trade #0 is a pre-launch smoke, designated non-sample by contemporary
    #     PRE-settlement notes).
    rows = [json.loads(l) for l in TRADES.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["expiry"] = pd.to_datetime(df["expiry_ms"], unit="ms", utc=True)
    excluded = int((~df["executed"].astype(bool)).sum())
    df = df[df["executed"].astype(bool)].reset_index(drop=True)
    df.attrs["excluded"] = excluded
    return df


def load_underlying() -> pd.DataFrame:
    # IT: prezzo del sottostante per (snapshot, expiry) dal raw chain registrato.
    #     `underlying_price` di Deribit e' il FORWARD di quella expiry: per tenor
    #     <= 30h il basis vs indice spot e' di pochi bps, e la sensibilita' e'
    #     quantificata sotto con due proxy alternativi.
    # EN: underlying price per (snapshot, expiry) from the recorded raw chain.
    #     Deribit's `underlying_price` is that expiry's FORWARD: for tenors <= 30h
    #     the basis vs the spot index is a few bps, and sensitivity is quantified
    #     below with two alternative proxies.
    parts = []
    for f in sorted(glob.glob(CHAIN_GLOB)):
        d = pd.read_parquet(f, columns=["snapshot_ts", "expiry", "underlying_price"])
        parts.append(d.groupby(["snapshot_ts", "expiry"], as_index=False)["underlying_price"].first())
    d = pd.concat(parts, ignore_index=True)
    d["snapshot_ts"] = pd.to_datetime(d["snapshot_ts"], utc=True)
    d["expiry"] = pd.to_datetime(d["expiry"], utc=True)
    return d.sort_values("snapshot_ts").reset_index(drop=True)


def load_front_and_spot(chain: pd.DataFrame):
    # IT: due proxy alternativi dell'indice per la sensibilita':
    #     P2 = forward della expiry PIU' VICINA a ogni snapshot (il piu' prossimo
    #          allo spot fra i forward disponibili);
    #     P3 = close 1m di Binance (indice costruito su un basket diverso).
    #     Se il conteggio non cambia fra i tre, la conclusione non dipende dal proxy.
    # EN: two alternative index proxies for sensitivity:
    #     P2 = forward of the NEAREST expiry at each snapshot (closest to spot
    #          among the available forwards);
    #     P3 = Binance 1m close (an index built on a different basket).
    #     If the count is stable across the three, the conclusion is proxy-independent.
    front = (chain.sort_values(["snapshot_ts", "expiry"])
                  .groupby("snapshot_ts", as_index=False)
                  .first()[["snapshot_ts", "underlying_price"]]
                  .rename(columns={"underlying_price": "px"}))
    spot = None
    if CANDLES_1M.exists():
        c = pd.read_parquet(CANDLES_1M)
        idx = pd.to_datetime(c.index, utc=True) if not isinstance(c.index, pd.RangeIndex) else None
        if idx is not None:
            spot = pd.DataFrame({"snapshot_ts": idx, "px": c["close"].to_numpy()})
            spot = spot.sort_values("snapshot_ts").reset_index(drop=True)
    return front, spot


def ticks_for(entry: pd.Timestamp, expiry: pd.Timestamp) -> pd.DatetimeIndex:
    # IT: tick orari in cui la posizione e' aperta E il processo puo' agire.
    # EN: hourly ticks at which the position is open AND the process can act.
    first = (entry.ceil("h") + pd.Timedelta(seconds=TICK_OFFSET_S))
    if first <= entry:
        first = first + pd.Timedelta(hours=1)
    return pd.date_range(first, expiry, freq="h", inclusive="left")


def price_at(ticks: pd.DatetimeIndex, src: pd.DataFrame) -> np.ndarray:
    # IT: prezzo osservato piu' vicino a ogni tick entro SNAP_TOL; NaN se il
    #     collector non copriva quel tick (tick NON OSSERVABILE, non "nessun
    #     innesco": la distinzione e' il punto del conteggio per epoche).
    # EN: nearest observed price within SNAP_TOL of each tick; NaN when the
    #     collector did not cover that tick (UNOBSERVABLE tick, not "no trigger":
    #     the distinction is the whole point of the per-epoch count).
    if src is None or len(src) == 0 or len(ticks) == 0:
        return np.full(len(ticks), np.nan)
    # IT: i parquet del chain arrivano in us, i tick li costruiamo in ns:
    #     merge_asof pretende la stessa risoluzione su entrambe le chiavi.
    # EN: chain parquets come in us while ticks are built in ns: merge_asof
    #     requires both keys at the same resolution.
    left = pd.DataFrame({"snapshot_ts": pd.DatetimeIndex(ticks).as_unit("ns")})
    src = src.copy()
    src["snapshot_ts"] = pd.DatetimeIndex(src["snapshot_ts"]).as_unit("ns")
    m = pd.merge_asof(left, src, on="snapshot_ts", direction="nearest", tolerance=SNAP_TOL)
    return m["px"].to_numpy(dtype=float)


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    vp = _import_from_path("volpaper_04b_pinfeas", ROOT / "scripts" / "04b_vol_paper.py")
    pos = load_positions()
    chain = load_underlying()
    front, spot = load_front_and_spot(chain)

    print("=" * 78)
    print("A13a — CONDIZIONE (3) EX-ANTE SUL PIN-CLOSE / ex-ante counting condition")
    print("        conteggio di EVENTI, nessuna PnL / EVENT count, no PnL")
    print("=" * 78)
    print(f"posizioni executed / executed positions : {len(pos)}"
          f"   (escluse/excluded: {pos.attrs['excluded']} non-executed)")
    print(f"finestra / window                       : {pos['entry_ts'].min():%Y-%m-%d} "
          f"-> {pos['entry_ts'].max():%Y-%m-%d}")
    print(f"snapshot chain / chain snapshots        : {chain['snapshot_ts'].nunique()}")

    # IT: per ogni posizione e ogni proxy: serie (t_left, moneyness) sui tick.
    # EN: per position and proxy: (t_left, moneyness) series over the ticks.
    proxies = {"P1_own_expiry_fwd": None, "P2_front_fwd": front, "P3_binance_1m": spot}
    per_pos = []
    for _, p in pos.iterrows():
        tk = ticks_for(p["entry_ts"], p["expiry"])
        own = (chain[chain["expiry"] == p["expiry"]][["snapshot_ts", "underlying_price"]]
               .rename(columns={"underlying_price": "px"}))
        rec = {"entry_ts": p["entry_ts"], "expiry": p["expiry"], "strike": float(p["strike"]),
               "side": int(p["side"]), "n_ticks": int(len(tk)),
               "epoch": "vps" if p["entry_ts"] >= VPS_EPOCH else "home"}
        for name, src in proxies.items():
            px = price_at(tk, own if src is None else src)
            rec[f"{name}__obs"] = int(np.isfinite(px).sum())
            rec[f"{name}__px"] = px
            rec[f"{name}__t"] = ((p["expiry"] - tk).total_seconds() / 3600.0).to_numpy()
        per_pos.append(rec)

    # IT: superficie di conteggio. Una posizione "innesca" se ESISTE un tick
    #     osservabile che soddisfa il predicato di produzione.
    # EN: count surface. A position "triggers" if THERE EXISTS an observable tick
    #     satisfying the production predicate.
    surface = {}
    for name in proxies:
        for X in GRID_HOURS:
            for f in GRID_BAND:
                trig = trig_vps = n_vps = 0
                unobs = 0
                for r in per_pos:
                    px, tl = r[f"{name}__px"], r[f"{name}__t"]
                    ok = np.isfinite(px)
                    # IT: tick rilevanti = quelli dentro la finestra temporale X.
                    # EN: relevant ticks = those inside the X time window.
                    rel = (tl > 0) & (tl <= X)
                    if rel.sum() > 0 and ok[rel].sum() == 0:
                        unobs += 1
                    hit = any(vp.pin_close_due(r["strike"], float(px[i]),
                                               r["expiry"].value / 1e6,
                                               r["expiry"].value / 1e6 - tl[i] * 3.6e6, X, f)
                              for i in range(len(tl)) if ok[i])
                    trig += bool(hit)
                    if r["epoch"] == "vps":
                        n_vps += 1
                        trig_vps += bool(hit)
                surface[f"{name}|{X}|{f}"] = {"n": len(per_pos), "trig": trig,
                                              "unobservable": unobs,
                                              "n_vps": n_vps, "trig_vps": trig_vps}

    # IT: stampa la superficie del proxy primario; gli altri due servono al
    #     controllo di robustezza stampato sotto.
    # EN: print the primary proxy's surface; the other two feed the robustness
    #     check printed below.
    prim = "P1_own_expiry_fwd"
    print(f"\nSUPERFICIE DI CONTEGGIO — proxy {prim} / count surface")
    print("righe = max_hours X, colonne = pin_band f / rows = X, cols = f")
    print("cella = inneschi su TUTTE le posizioni (di cui epoca VPS) / cell = triggers over ALL positions (of which VPS epoch)")
    hdr = "  X\\f  " + "".join(f"{f*100:>13.2f}%" for f in GRID_BAND)
    print(hdr)
    for X in GRID_HOURS:
        cells = []
        for f in GRID_BAND:
            s = surface[f"{prim}|{X}|{f}"]
            cells.append(f"{s['trig']:>5}/{s['n']:<3} ({s['trig_vps']:>2})")
        print(f"{X:>5.0f}h " + "".join(f"{c:>14}" for c in cells))

    print(f"\nROBUSTEZZA AL PROXY (inneschi totali) / proxy robustness (total triggers)")
    for X in GRID_HOURS:
        for f in GRID_BAND:
            vals = [surface[f"{n}|{X}|{f}"]["trig"] for n in proxies]
            if len(set(vals)) > 1:
                print(f"  X={X:g}h f={f*100:.2f}%: P1={vals[0]} P2={vals[1]} P3={vals[2]}  <-- DIVERGENTE/DIVERGING")
    if all(len({surface[f'{n}|{X}|{f}']['trig'] for n in proxies}) == 1
           for X in GRID_HOURS for f in GRID_BAND):
        print("  nessuna divergenza fra i tre proxy su tutta la griglia / no divergence across the three proxies")

    n_home = sum(1 for r in per_pos if r["epoch"] == "home")
    print(f"\nCOPERTURA / coverage: epoca casa {n_home} posizioni, epoca VPS {len(per_pos)-n_home}")
    worst = max(surface[f"{prim}|{X}|{f}"]["unobservable"] for X in GRID_HOURS for f in GRID_BAND)
    print(f"  posizioni con FINESTRA X interamente non osservata (max sulla griglia) / "
          f"positions with a fully unobserved X window (grid max): {worst}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "n_positions": len(per_pos),
        "n_excluded_non_executed": pos.attrs["excluded"],
        "window": [str(pos["entry_ts"].min()), str(pos["entry_ts"].max())],
        "grid_hours": GRID_HOURS, "grid_band": GRID_BAND,
        "tick_offset_s": TICK_OFFSET_S, "snap_tolerance": str(SNAP_TOL),
        "surface": surface,
        "note": "EVENT COUNT ONLY - no PnL, no counterfactual return, not a gate",
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nreport -> {OUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
