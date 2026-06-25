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

Uso / usage:  python scripts/vol/short_vol_regime_decomp.py [--vrp 0.0]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from short_vol_hist_backtest import run, fee_btc, TENOR_H, ANNUAL_BARS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REGIME = ROOT / "data" / "regime_probs.parquet"
OUT = ROOT / "results" / "vols" / "short_vol_regime_decomp.json"

REGIME_NAME = {0: "Quiet", 1: "Trending", 2: "Stress"}
# IT: haircut bid per struttura (dalla validazione FHS-vs-mark, n=3 → bias-check) | EN: per-struct bid haircut
BID_HAIRCUT = {"straddle": 0.035, "strangle": 0.16}


def pnl_series(tr, struct, vrp, haircut):
    # IT: PnL per-trade NET-of-bid in BTC. premio incassato = fair_value·(1+VRP)·(1−haircut);
    #     fee su 2 leg ≈ premio/2 ciascuna (cap 12.5%). PnL = premio − payoff_reale − fee.
    # EN: per-trade NET-of-bid PnL (BTC). premium received = fair_value·(1+VRP)·(1−haircut);
    #     fee on 2 legs ≈ premium/2 each (12.5% cap). PnL = premium − real_payoff − fee.
    prem = tr["fair_value"].to_numpy() * (1.0 + vrp) * (1.0 - haircut)
    fees = np.array([2 * fee_btc(p / 2) for p in prem])
    return prem - tr["realized_payoff"].to_numpy() - fees


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


def stats(pnl):
    # IT: stat di un vettore PnL (BTC). Sharpe annualizzato su barre/anno e tenor. | EN: PnL vector stats.
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
            "sharpe_ann": float(pnl.mean() / sd * np.sqrt(ANNUAL_BARS / TENOR_H)) if sd > 0 else 0.0,
            "worst": float(pnl.min()), "worst5_sum": float(np.sort(pnl)[:5].sum()),
            "max_drawdown": maxdd, "calmar": float(tot / abs(maxdd)) if maxdd < 0 else float("inf")}


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
    args = ap.parse_args()

    configs = [("straddle", 0.0), ("strangle", 0.06), ("strangle", 0.08)]
    print("=== SHORT-VOL · DECOMPOSIZIONE REGIME/ANNO + EQUITY/DRAWDOWN (NET-of-bid) ===")
    print(f"  VRP={args.vrp:.0%} (≈ vendo al mark) | haircut bid: straddle {BID_HAIRCUT['straddle']:.1%}, "
          f"strangle {BID_HAIRCUT['strangle']:.1%} | n_paths={args.n_paths}\n")

    report = {"meta": {"vrp": args.vrp, "bid_haircut": BID_HAIRCUT}, "configs": []}
    for struct, w in configs:
        tag = f"strangle {w:.0%}" if struct == "strangle" else "straddle ATM"
        tr = run(w, struct, args.n_paths, args.refit_days, 24 * 180, 42)
        hc = BID_HAIRCUT[struct]
        tr["pnl"] = pnl_series(tr, struct, args.vrp, hc)
        m = tag_regime(tr)
        overall = stats(m["pnl"].to_numpy())

        print(f"┌─ {tag} | n={overall['n']} | tot={overall['tot']:+.3f} BTC | "
              f"Sharpe={overall['sharpe_ann']:.2f} | maxDD={overall['max_drawdown']:+.3f} | "
              f"Calmar={overall['calmar']:.2f}")

        # ── per REGIME
        print(f"│  {'REGIME':<10} {'n':>4} {'tot':>9} {'mean':>9} {'hit':>5} {'Sharpe':>7} {'worst':>9} {'maxDD':>9}")
        per_regime = {}
        for rcode, rname in REGIME_NAME.items():
            sub = m[m["regime_dominant"] == rcode]
            st = stats(sub["pnl"].to_numpy())
            per_regime[rname] = st
            if st["n"] > 0:
                print(f"│  {rname:<10} {st['n']:>4} {st['tot']:>+9.3f} {st['mean']:>+9.5f} "
                      f"{st['hit']:>4.0f}% {st['sharpe_ann']:>7.2f} {st['worst']:>+9.4f} {st['max_drawdown']:>+9.3f}")
        n_nan = int(m["regime_dominant"].isna().sum())
        if n_nan:
            print(f"│  (regime n/d: {n_nan} trade — post 2026-06-10 o burn-in)")

        # ── per ANNO
        m["year"] = m["entry_ts"].dt.year
        print(f"│  {'ANNO':<10} {'n':>4} {'tot':>9} {'mean':>9} {'hit':>5} {'worst':>9}")
        per_year = {}
        for y, g in m.groupby("year"):
            st = stats(g["pnl"].to_numpy())
            per_year[int(y)] = st
            print(f"│  {y:<10} {st['n']:>4} {st['tot']:>+9.3f} {st['mean']:>+9.5f} {st['hit']:>4.0f}% {st['worst']:>+9.4f}")

        # ── variante REGIME-GATED: flat in Stress (R2)
        gated = m[m["regime_dominant"] != 2]            # esclude Stress (NaN inclusi = prudente skip)
        gated = gated.dropna(subset=["regime_dominant"])
        st_g = stats(gated["pnl"].to_numpy())
        print(f"└─ REGIME-GATED (no Stress): n={st_g['n']} tot={st_g['tot']:+.3f} "
              f"Sharpe={st_g['sharpe_ann']:.2f} maxDD={st_g['max_drawdown']:+.3f} Calmar={st_g['calmar']:.2f}  "
              f"vs always tot={overall['tot']:+.3f} Sharpe={overall['sharpe_ann']:.2f}\n")

        report["configs"].append({"struct": struct, "width": w, "overall": overall,
                                  "per_regime": per_regime, "per_year": per_year,
                                  "regime_gated_no_stress": st_g})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"  → report in {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
