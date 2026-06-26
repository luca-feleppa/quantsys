"""
short_vol_regime_decomp.py — decomposizione per REGIME/anno + equity/drawdown del backtest short-vol.
short_vol_regime_decomp.py — REGIME/year decomposition + equity/drawdown of the short-vol backtest.

IT: Il backtest storico (`short_vol_hist_backtest.py`) dice che l'edge short-vol esiste IN MEDIA su
    2019-26. Domanda critica prima di promuovere il braccio (classico "raccogliere centesimi davanti
    allo schiacciasassi"): l'edge REGGE in tutti i regimi o COLLASSA nello Stress (dove ti travolge)?
    Qui taggo ogni scadenza col regime dominante all'ingresso (causale, da data/regime_probs.parquet —
    R0 Quiet / R1 Trending / R2 Stress) e decompongo:
      • PnL per REGIME (mean/hit/worst): se la coda è tutta nello Stress ma il segno regge → robusto;
        se lo Stress flippa negativo → serve un FILTRO di regime.
      • PnL per ANNO (chi ha portato l'edge; il 2022 lo ha rotto?).
      • EQUITY curve + MAX DRAWDOWN della strategia come portafoglio (profondità/recupero DD).
      • Variante REGIME-GATED (short solo Quiet+Trending, flat in Stress) vs always-short.
EN: The historical backtest shows the short-vol edge exists ON AVERAGE over 2019-26. Critical question
    before promoting the arm ("picking up pennies in front of a steamroller"): does the edge HOLD across
    regimes or COLLAPSE in Stress (where it runs you over)? We tag each expiry with the dominant regime
    at entry (causal) and decompose by regime, by year, equity/max-drawdown, and a regime-gated variant.

IT: PnL realistico NET-of-bid: premio = fair-value FHS × (1+VRP) × (1−haircut_bid), con haircut per
    struttura dalla validazione (ATM 3.5%, strangle ~16%). VRP=0 ≈ "vendo al mark" (vedi validazione).
EN: Realistic NET-of-bid PnL: premium = FHS fair-value × (1+VRP) × (1−bid_haircut), per-structure
    haircut from the validation (ATM 3.5%, strangle ~16%). VRP=0 ≈ "sell at mark".

IT: HAIRCUT REGIME-DIPENDENTE (FIX ③, 2026-06-26). L'haircut base deriva da n=3 osservazioni di UN
    solo finestrino calmo (12 giorni): applicarlo COSTANTE allo Stress sottostima lo spread bid-ask
    reale (che nello Stress esplode ≫16%) → PnL net-of-bid dello Stress OTTIMISTICO, e gonfia proprio
    le due conclusioni headline ("edge più alto nello Stress", "non filtrare il regime"). Il fix
    applica l'haircut PER-TRADE DOPO il tagging causale del regime: base × `--stress-haircut-mult`
    (default 2.5) sui soli trade R2 Stress; R0/R1 e i trade a regime n/d (NaN: burn-in o post-fine
    serie regime) tengono l'haircut base. Segue una SENSITIVITY esplicita delle due headline.
EN: REGIME-DEPENDENT HAIRCUT (FIX ③, 2026-06-26). The base haircut comes from n=3 observations of ONE
    calm 12-day window: applying it CONSTANT in Stress understates the real bid-ask spread (which
    blows up ≫16% in Stress) → optimistic net-of-bid Stress PnL, inflating exactly the two headline
    claims ("edge highest in Stress", "don't filter the regime"). The fix applies the haircut PER-TRADE
    AFTER causal regime tagging: base × `--stress-haircut-mult` (default 2.5) on R2 Stress trades only;
    R0/R1 and n/d-regime trades (NaN: burn-in or past regime-series end) keep the base haircut. An
    explicit SENSITIVITY of the two headlines follows.

IT: ⚠ CAVEAT n=3 LOAD-BEARING. Sia i LIVELLI di haircut sia l'equivalenza "VRP=0 ≈ vendo al mark"
    poggiano sull'unica validazione premio a n=3 (overlap candele↔chain di `short_vol_premium_validate.py`):
    è un'ANCORA small-sample che regge l'INTERA tesi PnL net-of-bid, non un check accessorio.
EN: ⚠ LOAD-BEARING n=3 CAVEAT. Both the haircut LEVELS and the "VRP=0 ≈ sell at mark" equivalence rest
    on the single n=3 premium validation (candle↔chain overlap in `short_vol_premium_validate.py`): a
    small-sample ANCHOR holding up the ENTIRE net-of-bid PnL thesis, not an accessory check.

Uso / usage:  python scripts/vol/short_vol_regime_decomp.py [--vrp 0.0] [--stress-haircut-mult 2.5]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_vol_hist_backtest import (  # noqa: E402
    precompute_causal, price_trades, block_bootstrap_ci,
    TENOR_H, ANNUAL_BARS, FEE_PER_LEG, FEE_CAP_FRAC,
)

ROOT = Path(__file__).resolve().parents[2]
REGIME = ROOT / "data" / "regime_probs.parquet"
OUT = ROOT / "results" / "vols" / "short_vol_regime_decomp.json"

REGIME_NAME = {0: "Quiet", 1: "Trending", 2: "Stress"}
# IT: haircut bid BASE per struttura (validazione FHS-vs-mark, n=3 LOAD-BEARING su 12gg calmi → solo
#     regime Quiet); nello Stress va MOLTIPLICATO (--stress-haircut-mult), vedi pnl_series_regime.
# EN: BASE per-struct bid haircut (FHS-vs-mark validation, n=3 LOAD-BEARING on a calm 12-day window →
#     Quiet regime only); in Stress it is SCALED UP (--stress-haircut-mult), see pnl_series_regime.
BID_HAIRCUT = {"straddle": 0.035, "strangle": 0.16}


def pnl_series(tr, vrp, haircut):
    # IT: PnL per-trade NET-of-bid in BTC. premio incassato = fair_value·(1+VRP)·(1−haircut);
    #     fee su 2 leg ≈ premio/2 ciascuna (cap 12.5%). PnL = premio − payoff_reale − fee.
    #     `haircut` può essere SCALARE o un VETTORE per-trade (FIX ③: haircut regime-dipendente) →
    #     numpy broadcasta (1−haircut) riga-per-riga. | EN: `haircut` may be SCALAR or a PER-TRADE
    #     vector (FIX ③: regime-dependent haircut) → numpy broadcasts (1−haircut) row-wise.
    # EN: per-trade NET-of-bid PnL (BTC). premium received = fair_value·(1+VRP)·(1−haircut);
    #     fee on 2 legs ≈ premium/2 each (12.5% cap). PnL = premium − real_payoff − fee.
    prem = tr["fair_value"].to_numpy() * (1.0 + vrp) * (1.0 - np.asarray(haircut, dtype=float))
    # IT: fee 2 leg ≈ premio/2 (cap 12.5%), vettoriale | EN: 2-leg fee ≈ premium/2 (12.5% cap), vectorized
    fees = 2.0 * np.minimum(FEE_PER_LEG, FEE_CAP_FRAC * np.maximum(prem / 2.0, 0.0))
    return prem - tr["realized_payoff"].to_numpy() - fees


def regime_haircut_vector(m, base_hc, stress_mult):
    # IT: FIX ③. Vettore haircut PER-TRADE = base × (stress_mult se regime==2 Stress, altrimenti 1.0).
    #     I trade a regime n/d (NaN: burn-in o post-fine serie regime_probs) → confronto ==2 è False →
    #     ricadono sull'haircut BASE (scelta prudente documentata: niente penalità Stress su regime ignoto).
    # EN: FIX ③. PER-TRADE haircut vector = base × (stress_mult if regime==2 Stress, else 1.0). n/d-regime
    #     trades (NaN: burn-in or past regime_probs end) → ==2 is False → fall back to the BASE haircut
    #     (documented prudent choice: no Stress penalty on unknown regime). m must already be regime-tagged.
    reg = m["regime_dominant"].to_numpy()
    stress = (reg == 2.0)                                # NaN==2 → False (NaN→base) | NaN==2 → False (NaN→base)
    return base_hc * np.where(stress, float(stress_mult), 1.0)


def tag_regime(tr):
    # IT: regime dominante all'ingresso (merge_asof backward = solo regime ≤ entry, CAUSALE).
    #     Esclude il burn-in (regime inaffidabile). regime_probs finisce 2026-06-10 → trade dopo = NaN.
    # EN: dominant regime at entry (backward merge_asof = regime ≤ entry only, CAUSAL). Excludes burn-in.
    reg = pd.read_parquet(REGIME)
    reg.index = pd.to_datetime(reg.index, utc=True)
    reg = reg.sort_index().reset_index().rename(columns={"index": "ts"})
    if "ts" not in reg.columns:
        reg = reg.rename(columns={reg.columns[0]: "ts"})
    # IT: allinea la risoluzione datetime (regime=ms, entry=us) → ns, altrimenti merge_asof fallisce.
    # EN: align datetime resolution (regime=ms, entry=us) → ns, else merge_asof raises.
    reg["ts"] = reg["ts"].astype("datetime64[ns, UTC]")
    t = tr.copy()
    t["entry_ts"] = pd.to_datetime(t["t_entry"], utc=True).astype("datetime64[ns, UTC]")
    t = t.sort_values("entry_ts").reset_index(drop=True)
    m = pd.merge_asof(t, reg[["ts", "regime_dominant", "regime_burn_in"]],
                      left_on="entry_ts", right_on="ts", direction="backward")
    m.loc[m["regime_burn_in"] == True, "regime_dominant"] = np.nan  # noqa: E712
    return m


def stats(pnl, ann_factor):
    # IT: stat di un vettore PnL (BTC). Sharpe annualizzato = mean/sd × ann_factor, con
    #     ann_factor = sqrt(trades_per_year) DERIVATO dallo span temporale reale dei trade (FIX coerenza
    #     col kernel: cadenza DAILY ≈365/anno, NON sqrt(ANNUAL_BARS/TENOR_H)=sqrt(292), che assume 292
    #     cicli da 30h non sovrapposti). ⚠ L'overlap (tenor 30h > spacing 24h) induce autocorrelazione
    #     positiva che GONFIA il Sharpe i.i.d. → la statistica ONESTA è la CI block-bootstrap (overall).
    # EN: PnL vector stats. Annualized Sharpe = mean/sd × ann_factor, ann_factor = sqrt(trades_per_year)
    #     DERIVED from the real trade time-span (kernel-consistent: DAILY cadence ≈365/yr, NOT
    #     sqrt(ANNUAL_BARS/TENOR_H)=sqrt(292), which assumes 292 non-overlapping 30h cycles). ⚠ Overlap
    #     (30h tenor > 24h spacing) induces positive autocorrelation that INFLATES the i.i.d. Sharpe →
    #     the HONEST statistic is the block-bootstrap CI (overall).
    pnl = np.asarray(pnl, dtype=float)
    if len(pnl) == 0:
        return {"n": 0}
    sd = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    eq = np.cumsum(pnl)
    dd = eq - np.maximum.accumulate(eq)               # drawdown vs picco | drawdown vs running peak
    maxdd = float(dd.min())
    tot = float(pnl.sum())
    return {"n": int(len(pnl)), "tot": tot, "mean": float(pnl.mean()), "median": float(np.median(pnl)),
            "hit": float(100 * (pnl > 0).mean()),
            "sharpe_ann": float(pnl.mean() / sd * ann_factor) if sd > 0 else 0.0,
            "worst": float(pnl.min()), "worst5_sum": float(np.sort(pnl)[:5].sum()),
            "max_drawdown": maxdd, "calmar": float(tot / abs(maxdd)) if maxdd < 0 else float("inf")}


def ann_factor_from_trades(m):
    # IT: FIX coerenza Sharpe. trades/anno dallo span REALE dei timestamp d'ingresso (cadenza DAILY) →
    #     ann_factor = sqrt(trades_per_year); fallback al fattore legacy sqrt(ANNUAL_BARS/TENOR_H) se
    #     span nullo (1 trade). | EN: Sharpe-coherence fix. trades/year from the REAL entry-timestamp span
    #     (DAILY cadence) → ann_factor = sqrt(trades_per_year); legacy fallback if the span is null.
    ts = pd.to_datetime(m["entry_ts"], utc=True)
    span_years = (ts.max() - ts.min()).total_seconds() / (365.25 * 24 * 3600.0)
    tpy = (len(m) / span_years) if span_years > 0 else (ANNUAL_BARS / TENOR_H)
    return float(np.sqrt(tpy)), float(tpy)


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--vrp", type=float, default=0.0, help="VRP applicato (0 ≈ vendo al mark)")
    ap.add_argument("--n-paths", type=int, default=4000)
    ap.add_argument("--refit-days", type=int, default=90)
    # IT: FIX ③. Moltiplicatore dell'haircut bid SOLO sui trade in regime Stress (R2): nello Stress lo
    #     spread bid-ask reale esplode ≫ del livello base calibrato su 12gg calmi. 1.0 = baseline vecchio.
    # EN: FIX ③. Bid-haircut multiplier on Stress-regime (R2) trades ONLY: real bid-ask spread blows up
    #     in Stress, far above the base level calibrated on a calm 12-day window. 1.0 = old baseline.
    ap.add_argument("--stress-haircut-mult", type=float, default=2.5,
                    help="moltiplicatore haircut bid sui soli trade Stress R2 (default 2.5; 1.0 = baseline vecchio)")
    args = ap.parse_args()

    configs = [("straddle", 0.0), ("strangle", 0.06), ("strangle", 0.08)]
    print("=== SHORT-VOL · DECOMPOSIZIONE REGIME/ANNO + EQUITY/DRAWDOWN (NET-of-bid) ===")
    print(f"  VRP={args.vrp:.0%} (≈ vendo al mark) | haircut bid BASE: straddle {BID_HAIRCUT['straddle']:.1%}, "
          f"strangle {BID_HAIRCUT['strangle']:.1%} | Stress×{args.stress_haircut_mult:.2f} | n_paths={args.n_paths}")
    # IT: caveat n=3 LOAD-BEARING stampato a inizio run (ancora small-sample che regge tutta la tesi PnL).
    # EN: load-bearing n=3 caveat printed at run start (small-sample anchor holding the whole PnL thesis).
    print("  ⚠ CAVEAT n=3 LOAD-BEARING: livelli haircut + equivalenza 'VRP=0 ≈ vendo al mark' poggiano")
    print("    sull'UNICA validazione premio a n=3 (overlap candele↔chain, 12gg calmi) → ancora che regge")
    print("    l'INTERA tesi PnL net-of-bid; l'haircut Stress è un'estrapolazione, NON un dato misurato.\n")

    report = {"meta": {"vrp": args.vrp, "bid_haircut": BID_HAIRCUT,
                       "stress_haircut_mult": args.stress_haircut_mult,
                       "caveat_n3": "haircut levels + VRP=0≈mark rest on the n=3 premium validation"},
              "configs": [], "sensitivity": []}
    # IT: serie causale GARCH calcolata UNA volta, riusata dalle 3 config (A1). rng fresco per-config
    #     → path FHS bit-identici al comportamento legacy (un run() separato per config).
    # EN: causal GARCH series computed ONCE, reused by the 3 configs (A1). Fresh per-config rng
    #     → FHS paths bit-identical to legacy (one separate run() per config).
    pre = precompute_causal(args.refit_days, 24 * 180)
    for struct, w in configs:
        tag = f"strangle {w:.0%}" if struct == "strangle" else "straddle ATM"
        tr = price_trades(pre, w, struct, args.n_paths, np.random.default_rng(42))
        # IT: FIX ③. PRIMA tagga il regime (causale), POI calcola l'haircut PER-TRADE (Stress×mult) e il
        #     PnL → tutte le decomposizioni a valle usano l'haircut regime-dipendente.
        # EN: FIX ③. FIRST tag the regime (causal), THEN compute the PER-TRADE haircut (Stress×mult) and
        #     the PnL → every downstream decomposition uses the regime-dependent haircut.
        m = tag_regime(tr)
        hc_vec = regime_haircut_vector(m, BID_HAIRCUT[struct], args.stress_haircut_mult)
        m["pnl"] = pnl_series(m, args.vrp, hc_vec)
        # IT: ann_factor coerente col kernel (sqrt(trades/anno) dallo span reale), riusato da ogni stats().
        # EN: kernel-consistent ann_factor (sqrt(trades/year) from the real span), reused by every stats().
        ann_factor, tpy = ann_factor_from_trades(m)
        overall = stats(m["pnl"].to_numpy(), ann_factor)
        # IT: CI block-bootstrap (overlap/clustering-robust) dell'OVERALL su PnL ordinato per tempo (m è
        #     già sorted per entry_ts in tag_regime) — DRY, riusa l'helper del kernel.
        # EN: block-bootstrap CI (overlap/clustering-robust) of the OVERALL on time-ordered PnL (m is
        #     already sorted by entry_ts in tag_regime) — DRY, reuses the kernel helper.
        boot = block_bootstrap_ci(m["pnl"].to_numpy(), ann_factor)
        overall.update({"mean_ci05": boot["mean_ci"][0], "mean_ci95": boot["mean_ci"][1],
                        "sharpe_ci05": boot["sharpe_ci"][0], "sharpe_ci95": boot["sharpe_ci"][1],
                        "n_eff": boot["n_eff"], "rho1": boot["rho1"], "trades_per_year": tpy})

        print(f"┌─ {tag} | n={overall['n']} | tot={overall['tot']:+.3f} BTC | "
              f"Sharpe={overall['sharpe_ann']:.2f} | maxDD={overall['max_drawdown']:+.3f} | "
              f"Calmar={overall['calmar']:.2f} | trades/yr≈{tpy:.0f} N_eff≈{boot['n_eff']:.0f}")
        # IT: CI ONESTA dell'overall (5–95%): mean-CI che include 0 ⇒ edge non distinguibile dal rumore.
        # EN: honest overall CI (5–95%): a mean-CI spanning 0 ⇒ edge indistinguishable from noise.
        print(f"│  block-boot CI(5-95%): mean [{overall['mean_ci05']:+.5f}, {overall['mean_ci95']:+.5f}] "
              f"Sharpe [{overall['sharpe_ci05']:+.2f}, {overall['sharpe_ci95']:+.2f}] (ρ₁={overall['rho1']:+.3f})")

        # ── per REGIME
        print(f"│  {'REGIME':<10} {'n':>4} {'tot':>9} {'mean':>9} {'hit':>5} {'Sharpe':>7} {'worst':>9} {'maxDD':>9}")
        per_regime = {}
        for rcode, rname in REGIME_NAME.items():
            sub = m[m["regime_dominant"] == rcode]
            st = stats(sub["pnl"].to_numpy(), ann_factor)
            per_regime[rname] = st
            if st["n"] > 0:
                print(f"│  {rname:<10} {st['n']:>4} {st['tot']:>+9.3f} {st['mean']:>+9.5f} "
                      f"{st['hit']:>4.0f}% {st['sharpe_ann']:>7.2f} {st['worst']:>+9.4f} {st['max_drawdown']:>+9.3f}")
        n_nan = int(m["regime_dominant"].isna().sum())
        if n_nan:
            print(f"│  (regime n/d: {n_nan} trade — post 2026-06-10 o burn-in; haircut BASE applicato)")

        # ── per ANNO
        m["year"] = m["entry_ts"].dt.year
        print(f"│  {'ANNO':<10} {'n':>4} {'tot':>9} {'mean':>9} {'hit':>5} {'worst':>9}")
        per_year = {}
        for y, g in m.groupby("year"):
            st = stats(g["pnl"].to_numpy(), ann_factor)
            per_year[int(y)] = st
            print(f"│  {y:<10} {st['n']:>4} {st['tot']:>+9.3f} {st['mean']:>+9.5f} {st['hit']:>4.0f}% {st['worst']:>+9.4f}")

        # ── variante REGIME-GATED: flat in Stress (R2)
        gated = m[m["regime_dominant"] != 2]            # esclude Stress (NaN inclusi = prudente skip)
        gated = gated.dropna(subset=["regime_dominant"])
        st_g = stats(gated["pnl"].to_numpy(), ann_factor)
        print(f"└─ REGIME-GATED (no Stress): n={st_g['n']} tot={st_g['tot']:+.3f} "
              f"Sharpe={st_g['sharpe_ann']:.2f} maxDD={st_g['max_drawdown']:+.3f} Calmar={st_g['calmar']:.2f}  "
              f"vs always tot={overall['tot']:+.3f} Sharpe={overall['sharpe_ann']:.2f}")

        # ── SENSITIVITY delle due headline all'haircut Stress allargato (FIX ③)
        # IT: (a) lo Stress resta l'edge migliore? = mean Stress > 0 E ≥ mean di Quiet/Trending.
        #     (b) "non filtrare il regime" regge? = always-short (overall.tot) ≥ regime-gated (st_g.tot).
        # EN: (a) is Stress still the best edge? = Stress mean > 0 AND ≥ Quiet/Trending mean.
        #     (b) does "don't filter the regime" hold? = always-short (overall.tot) ≥ regime-gated (st_g.tot).
        st_stress = per_regime["Stress"]
        others = [per_regime[r]["mean"] for r in ("Quiet", "Trending") if per_regime[r].get("n", 0) > 0]
        max_other = max(others) if others else float("-inf")
        a_stress_pos = (st_stress.get("n", 0) > 0 and st_stress["mean"] > 0 and st_stress["sharpe_ann"] > 0)
        a_stress_best = a_stress_pos and (st_stress["mean"] >= max_other)
        b_no_filter = overall["tot"] >= st_g["tot"]
        va = ("REGGE — Stress mean/Sharpe > 0 ed è il regime migliore" if a_stress_best
              else ("PARZIALE — Stress > 0 ma NON più il migliore" if a_stress_pos
                    else "CADE — Stress NON più positivo netto"))
        vb = ("REGGE — always-short ≥ regime-gated" if b_no_filter
              else "CADE — conviene filtrare lo Stress (gated > always)")
        print(f"   SENSITIVITY @Stress×{args.stress_haircut_mult:.2f}:")
        print(f"     (a) Stress resta l'edge migliore? → {va} "
              f"[mean Stress {st_stress.get('mean', float('nan')):+.5f} vs max(Q,T) {max_other:+.5f}]")
        print(f"     (b) NON filtrare il regime regge?  → {vb} "
              f"[always tot {overall['tot']:+.3f} vs gated tot {st_g['tot']:+.3f}]\n")

        report["configs"].append({"struct": struct, "width": w, "overall": overall,
                                  "per_regime": per_regime, "per_year": per_year,
                                  "regime_gated_no_stress": st_g})
        report["sensitivity"].append({"struct": struct, "width": w,
                                      "stress_haircut_mult": args.stress_haircut_mult,
                                      "a_stress_still_best": bool(a_stress_best),
                                      "a_stress_positive": bool(a_stress_pos),
                                      "b_no_regime_filter_holds": bool(b_no_filter),
                                      "stress_mean": st_stress.get("mean"),
                                      "max_other_mean": (None if max_other == float("-inf") else max_other),
                                      "always_tot": overall["tot"], "gated_tot": st_g["tot"]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"  → report in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
