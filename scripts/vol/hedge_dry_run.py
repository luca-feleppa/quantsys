# IT: DRY-RUN RETROSPETTIVO DELL'HEDGE (pre-studio B2/A1, ROADMAP_VOL_BOOK) — offline,
#     read-only su results/vol_paper/exec_diag.jsonl (serie A6). Simula la leg
#     delta-hedge sul perp Deribit COME SE fosse stata ribilanciata a ogni tick
#     orario, sui trade/posizioni già osservati:
#       1. PnL opzioni per intervallo: Δm_t = Δ(Σ mark leg) in BTC (side-adjusted);
#       2. PnL perp inverse per intervallo con posizione h_t BTC-equivalenti:
#          pnl = H_usd·(1/S_t − 1/S_{t+1}), H_usd = h_t·S_t (formula esatta inverse);
#       3. DUE convenzioni di hedge ratio a confronto (caveat B2 "delta convention"):
#          δ_raw  = delta del venue (greeks.delta, ∂V_usd/∂S)
#          δ_adj  = δ_raw − m_btc  (BTC-terms: m=V/S → dm/dS=(Δ_usd−m)/S)
#       4. VERIFICA EMPIRICA: regressione Δm su r=ΔS/S → slope = delta efficace
#          osservato, da confrontare con mean(δ_raw) e mean(δ_adj);
#       5. metriche: varianza per-intervallo hedged vs unhedged, fee stimate del
#          ribilanciamento (taker perp, parametrico), PnL cumulato.
#     NON tocca 04b né la regola pre-registrata: è il pre-studio che dimensiona
#     il gate hedged-vs-unhedged della v2 (no-trade band, drag atteso).
# EN: RETROSPECTIVE HEDGE DRY-RUN (B2/A1 pre-study, ROADMAP_VOL_BOOK) — offline,
#     read-only over results/vol_paper/exec_diag.jsonl (A6 series). Simulates the
#     Deribit perp delta-hedge leg AS IF rebalanced at every hourly tick, on the
#     already-observed positions: per-interval option PnL Δm (BTC), exact inverse
#     perp PnL with h_t BTC-equivalent position, TWO delta conventions compared
#     (venue δ_raw vs BTC-terms δ_adj = δ_raw − mark), empirical hedge-ratio
#     regression (slope of Δm on r=ΔS/S), hedged-vs-unhedged variance, taker-fee
#     drag (parametric). Does NOT touch 04b nor the pre-registered rule: it is the
#     pre-study sizing the v2 hedged-vs-unhedged gate (no-trade band, expected drag).
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging                                     # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.hedge_dry_run")

EXEC_DIAG_PATH = Path("results/vol_paper/exec_diag.jsonl")
OUT_PATH = Path("results/vols/hedge_dry_run.json")

# IT: fee taker perp Deribit (frazione del nozionale scambiato) — ASSUNZIONE
#     parametrica (CLI), non costante pre-registrata: al design v2 va letta dal venue.
# EN: Deribit perp taker fee (fraction of traded notional) — parametric ASSUMPTION
#     (CLI), not a pre-registered constant: v2 design must read it from the venue.
DEFAULT_PERP_FEE = 5e-4


def load_ticks(path: Path) -> list:
    # IT: tick validi = source 'position' (posizione in essere) con mark/underlying/
    #     delta presenti su ENTRAMBE le leg (A6 è fail-soft: i campi possono essere None).
    #     I tick 'flat' (straddle ipotetico) sono un'ALTRA struttura → esclusi.
    # EN: valid ticks = source 'position' (live position) with mark/underlying/delta
    #     present on BOTH legs (A6 is fail-soft: fields may be None). 'flat' ticks
    #     (hypothetical straddle) are a DIFFERENT structure → excluded.
    ticks = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("source") != "position":
            continue
        legs = r.get("legs", [])
        if len(legs) != 2 or any(
                l.get(k) is None for l in legs for k in ("mark", "underlying", "delta")):
            continue
        ticks.append({
            "ts": r["ts"], "side": int(r["side"]), "strike": float(r["strike"]),
            "expiry_ms": int(r["expiry_ms"]),
            "S": float(np.mean([l["underlying"] for l in legs])),   # forward per-expiry
            "m": float(sum(l["mark"] for l in legs)),               # BTC/contratto · per contract
            "delta_raw": float(sum(l["delta"] for l in legs)),      # convenzione venue · venue convention
        })
    return ticks


def perp_pnl_btc(h_btc: float, s0: float, s1: float) -> float:
    # IT: PnL in BTC del perp inverse: posizione h BTC-equivalenti aperta a s0,
    #     chiusa a s1 → H_usd·(1/s0 − 1/s1), H_usd = h·s0. Esatto, non linearizzato.
    # EN: inverse perp PnL in BTC: h BTC-equivalent position opened at s0, closed
    #     at s1 → H_usd·(1/s0 − 1/s1), H_usd = h·s0. Exact, not linearized.
    return h_btc * s0 * (1.0 / s0 - 1.0 / s1)


def simulate(ticks: list, fee: float) -> dict:
    # IT: gruppo per struttura (side/strike/expiry): il ribilanciamento è intra-holding,
    #     MAI a cavallo di due posizioni diverse.
    # EN: group by structure (side/strike/expiry): rebalancing is intra-holding,
    #     NEVER across two different positions.
    n = len(ticks)
    d_opt, d_hraw, d_hadj, rets, fees_raw, fees_adj = [], [], [], [], [], []
    # IT: δ sided accumulati sullo STESSO campione della regressione (audit MINOR-3:
    #     con più strutture le medie su tutti i tick divergerebbero dagli intervalli usati).
    # EN: sided δ accumulated on the SAME sample as the regression (audit MINOR-3:
    #     with multiple structures whole-tick means would diverge from the used intervals).
    delta_raw_sided, delta_adj_sided = [], []
    prev_h_raw = prev_h_adj = None

    def _close_hedge_fees():
        # IT: fee di CHIUSURA della leg hedge a fine struttura (audit MINOR-2: senza,
        #     fees_total sottostima di ~1 ribilanciamento per struttura).
        # EN: hedge-leg CLOSING fee at structure end (audit MINOR-2: without it,
        #     fees_total understates by ~1 rebalance per structure).
        if prev_h_raw is not None:
            fees_raw.append(fee * abs(prev_h_raw))
            fees_adj.append(fee * abs(prev_h_adj))

    for a, b in zip(ticks[:-1], ticks[1:]):
        if (a["side"], a["strike"], a["expiry_ms"]) != (b["side"], b["strike"], b["expiry_ms"]):
            _close_hedge_fees()
            prev_h_raw = prev_h_adj = None
            continue
        side = a["side"]
        dm = side * (b["m"] - a["m"])                 # PnL opzioni (BTC) · option PnL
        # IT: hedge = −delta netto della posizione, nelle due convenzioni.
        # EN: hedge = −net position delta, in both conventions.
        h_raw = -side * a["delta_raw"]
        h_adj = -side * (a["delta_raw"] - a["m"])     # δ_adj = δ_usd − m (inverse)
        d_opt.append(dm)
        d_hraw.append(dm + perp_pnl_btc(h_raw, a["S"], b["S"]))
        d_hadj.append(dm + perp_pnl_btc(h_adj, a["S"], b["S"]))
        rets.append(b["S"] / a["S"] - 1.0)
        delta_raw_sided.append(side * a["delta_raw"])
        delta_adj_sided.append(side * (a["delta_raw"] - a["m"]))
        # IT: fee sul nozionale RIBILANCIATO (|Δh|; il primo tick paga l'apertura piena).
        # EN: fee on the REBALANCED notional (|Δh|; first tick pays the full opening).
        fees_raw.append(fee * abs(h_raw - (prev_h_raw or 0.0)))
        fees_adj.append(fee * abs(h_adj - (prev_h_adj or 0.0)))
        prev_h_raw, prev_h_adj = h_raw, h_adj
    _close_hedge_fees()                               # IT/EN: chiusura serie / end of series

    d_opt, d_hraw, d_hadj = map(np.asarray, (d_opt, d_hraw, d_hadj))
    rets = np.asarray(rets)
    k = len(d_opt)
    if k < 2:
        raise SystemExit(f"solo {k} intervalli utilizzabili — serve più storia A6 / "
                         f"only {k} usable intervals — need more A6 history")

    # IT: delta efficace empirico = slope OLS di Δm su r (side-adjusted già in Δm):
    #     atteso ≈ side·mean(δ_usd − m)·? — il confronto dice QUALE convenzione hedgia.
    # EN: empirical effective delta = OLS slope of Δm on r (Δm already side-adjusted):
    #     comparison against both conventions says WHICH one actually hedges.
    slope, intercept = np.polyfit(rets, d_opt, 1)
    r2 = float(np.corrcoef(rets, d_opt)[0, 1] ** 2)
    # IT: SE dello slope (OLS classico) — con n piccolo decide se lo scarto tra
    #     slope empirico e convenzioni δ è segnale o rumore. A k=2 i gradi di
    #     libertà sono 0 → SE indefinita: None, non un nan silenzioso (audit MINOR-1).
    # EN: slope SE (classic OLS) — with small n it decides whether the gap between
    #     the empirical slope and the δ conventions is signal or noise. At k=2 the
    #     degrees of freedom are 0 → SE undefined: None, not a silent nan (audit MINOR-1).
    slope_se = None
    if k >= 3:
        resid = d_opt - (slope * rets + intercept)
        slope_se = float(np.sqrt(np.sum(resid ** 2) / (k - 2) / np.sum((rets - rets.mean()) ** 2)))

    def stats(x):
        return {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)),
                "total": float(np.sum(x))}

    out = {
        "n_ticks": n, "n_intervals": k,
        "span": [ticks[0]["ts"], ticks[-1]["ts"]],
        "perp_fee_assumed": fee,
        "empirical": {"slope_dm_on_r": float(slope), "slope_se": slope_se,
                      "intercept": float(intercept), "r2": r2,
                      "mean_delta_raw_sided": float(np.mean(delta_raw_sided)),
                      "mean_delta_adj_sided": float(np.mean(delta_adj_sided))},
        "unhedged": stats(d_opt),
        # IT: total_net = PnL hedged al netto delle fee (apertura+ribilanci+chiusura).
        # EN: total_net = hedged PnL net of fees (open+rebalances+close).
        "hedged_raw": {**stats(d_hraw), "fees_total": float(np.sum(fees_raw)),
                       "total_net": float(np.sum(d_hraw) - np.sum(fees_raw))},
        "hedged_adj": {**stats(d_hadj), "fees_total": float(np.sum(fees_adj)),
                       "total_net": float(np.sum(d_hadj) - np.sum(fees_adj))},
        "variance_reduction": {
            "raw": 1.0 - float(np.var(d_hraw, ddof=1) / np.var(d_opt, ddof=1)),
            "adj": 1.0 - float(np.var(d_hadj, ddof=1) / np.var(d_opt, ddof=1))},
        "caveats": [
            "campione = soli intervalli con A6 attivo (NON il tenor completo) / sample = A6-active intervals only",
            "Δm include theta decay e Δvega (non solo delta) / Δm includes theta decay and vega moves",
            "funding perp NON incluso (serie troppo corta) / perp funding NOT included (series too short)",
            "fee = assunzione parametrica, non costante pre-registrata / fee = parametric assumption",
            "S = forward per-expiry delle opzioni usato anche come prezzo del perp: basis perp-forward NON modellata / options per-expiry forward also used as the perp price: perp-forward basis NOT modeled",
        ],
    }
    return out


def main():
    # IT: boilerplate UTF-8 (checklist CLAUDE.md — bug cp1252 ricorrente).
    # EN: UTF-8 boilerplate (CLAUDE.md checklist — recurring cp1252 bug).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Dry-run retrospettivo delta-hedge (pre-studio B2/A1)")
    ap.add_argument("--fee", type=float, default=DEFAULT_PERP_FEE,
                    help="fee taker perp (frazione nozionale) / perp taker fee (notional fraction)")
    args = ap.parse_args()

    if not EXEC_DIAG_PATH.exists():
        raise SystemExit(f"{EXEC_DIAG_PATH} assente — A6 non ha ancora loggato / missing — A6 has not logged yet")
    ticks = load_ticks(EXEC_DIAG_PATH)
    log.info(f"tick 'position' validi/valid: {len(ticks)} da/from {EXEC_DIAG_PATH}")
    res = simulate(ticks, args.fee)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")

    e, u, hr, ha = res["empirical"], res["unhedged"], res["hedged_raw"], res["hedged_adj"]
    log.info("═" * 72)
    log.info(f"  HEDGE DRY-RUN — {res['n_intervals']} intervalli orari, span {res['span'][0]} → {res['span'][1]}")
    _se = f"± {e['slope_se']:.4f}" if e["slope_se"] is not None else "(SE indefinita, k<3 / SE undefined)"
    log.info(f"  delta efficace (OLS Δm~r): {e['slope_dm_on_r']:+.4f} {_se}  (R²={e['r2']:.3f})")
    log.info(f"  convenzioni: δ_raw={e['mean_delta_raw_sided']:+.4f} · δ_adj={e['mean_delta_adj_sided']:+.4f}")
    log.info(f"  per-intervallo (BTC):  unhedged μ={u['mean']:+.5f} σ={u['std']:.5f}")
    log.info(f"                        hedged_raw μ={hr['mean']:+.5f} σ={hr['std']:.5f} "
             f"(fee {hr['fees_total']:.5f} → tot net {hr['total_net']:+.5f})")
    log.info(f"                        hedged_adj μ={ha['mean']:+.5f} σ={ha['std']:.5f} "
             f"(fee {ha['fees_total']:.5f} → tot net {ha['total_net']:+.5f})")
    log.info(f"  riduzione varianza: raw {res['variance_reduction']['raw']*100:.1f}% · "
             f"adj {res['variance_reduction']['adj']*100:.1f}%")
    log.info(f"  report → {OUT_PATH}")
    log.info("═" * 72)


if __name__ == "__main__":
    main()
