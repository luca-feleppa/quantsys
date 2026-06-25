"""
short_vol_hist_backtest.py — backtest STORICO strutturale del braccio SHORT-VOL (vol-line).
short_vol_hist_backtest.py — STRUCTURAL HISTORICAL backtest of the SHORT-VOL arm (vol line).

IT: Complementa lo studio live `short_vol_arm.py` (limitato a n≈4 scadenze raccolte). Qui
    usiamo 7 anni di candele orarie BTC (2019→2026, ~2700 scadenze giornaliere 08:00 UTC) per
    misurare se vendere strangle/straddle a 30h sopravvive alle CODE storiche (crash 2020/21/22),
    proprio ciò che i 12 giorni calmi del live NON mostrano.
EN: Complements the live study `short_vol_arm.py` (capped at n≈4 collected expiries). Here we use
    7 years of hourly BTC candles (2019→2026, ~2700 daily 08:00-UTC expiries) to measure whether
    selling 30h strangles/straddles survives the HISTORICAL tails (2020/21/22 crashes) — exactly
    what the 12 calm days of the live test cannot show.

IT: VINCOLO DATI: non esiste superficie IV implicita storica BTC (solo 12 giorni raccolti). Quindi
    il backtest NON è "ho venduto al mark reale" ma uno STUDIO STRUTTURALE TAIL + SWEEP del VRP:
      • lato PAYOFF  = reale, dalle candele (zero modello, code incluse).
      • lato PREMIO  = fair value risk-neutral-ish sotto kernel FAT-TAILED × (1 + VRP), VRP swept.
    Kernel = FHS su GJR-GARCH(1,1): NESSUNA assunzione gaussiana (≠ Black-Scholes, che a vol piatta
    sbaglia smile e code = proprio dove vive il payoff dello strangle OTM). I residui standardizzati
    REALI vengono bootstrappati → code/asimmetria/clustering empirici. Tutto CAUSALE (params/residui
    solo ≤ entry; refit expanding a cadenza, no look-ahead → no lookahead bias / stazionarietà ok).
EN: DATA CONSTRAINT: no historical BTC implied-vol surface exists (only 12 collected days). So this
    is NOT "I sold at the real mark" but a STRUCTURAL TAIL study + VRP SWEEP:
      • PAYOFF side  = real, from candles (no model, tails included).
      • PREMIUM side = fat-tailed risk-neutral-ish fair value × (1 + VRP), VRP swept.
    Kernel = FHS on GJR-GARCH(1,1): NO Gaussian assumption (≠ Black-Scholes, whose flat vol misprices
    smile and tails = exactly where the OTM strangle payoff lives). REAL standardized residuals are
    bootstrapped → empirical tails/skew/clustering. Fully CAUSAL (params/residuals only ≤ entry;
    expanding refit on a cadence, no look-ahead).

IT: Deliverable = curva di BREAK-EVEN VRP per (struct × width): "lo strangle W% su 2019-26 è netto
    positivo sse implied/realized ≥ X", con distribuzione di coda e contributo dei worst trade.
EN: Deliverable = BREAK-EVEN VRP curve per (struct × width): "the W% strangle over 2019-26 is net
    positive iff implied/realized ≥ X", with tail distribution and worst-trade contribution.

Uso / usage:  python scripts/vol/short_vol_hist_backtest.py
              python scripts/vol/short_vol_hist_backtest.py --n-paths 6000 --refit-days 60
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
CANDLES = ROOT / "data" / "raw_candles.parquet"
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
OUT = ROOT / "results" / "vols" / "short_vol_hist_backtest.json"

TENOR_H = 30                 # IT: tenor in barre orarie (=30h) come il forward test | EN: tenor in hourly bars
EXPIRY_HOUR = 8             # IT: scadenze daily Deribit 08:00 UTC | EN: Deribit daily expiries 08:00 UTC
FEE_PER_LEG = 0.0003        # IT: fee testnet ~3e-4 BTC/leg | EN: ~3e-4 BTC/leg
FEE_CAP_FRAC = 0.125        # IT: cap Deribit = 12.5% del premio/leg | EN: Deribit cap = 12.5% of premium/leg
VRP_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]  # IT: griglia VRP implied/realized | EN: VRP grid
ANNUAL_BARS = 24 * 365      # IT: barre/anno per Sharpe annualizzato | EN: bars/year for annualized Sharpe


# ─────────────────────────────────────────────────────────────────────────────
# GJR-GARCH(1,1) — QMLE gaussiano con variance targeting (params causali per FHS)
# GJR-GARCH(1,1) — Gaussian QMLE with variance targeting (causal params for FHS)
# ─────────────────────────────────────────────────────────────────────────────
def gjr_recursion(r, omega, alpha, gamma, beta, var0):
    # IT: σ²_t = ω + (α + γ·I[r_{t-1}<0])·r²_{t-1} + β·σ²_{t-1}. Ritorna σ_t (NON σ²).
    # EN: σ²_t = ω + (α + γ·I[r_{t-1}<0])·r²_{t-1} + β·σ²_{t-1}. Returns σ_t (NOT σ²).
    n = len(r)
    v = np.empty(n)
    v[0] = var0
    for t in range(1, n):
        shock = r[t - 1] ** 2
        neg = 1.0 if r[t - 1] < 0 else 0.0
        v[t] = omega + (alpha + gamma * neg) * shock + beta * v[t - 1]
    return np.sqrt(np.maximum(v, 1e-12))


def fit_gjr(r, warm=None):
    # IT: stima (α, γ, β) via QMLE; ω da variance targeting → ω = σ̄²·(1 − α − γ/2 − β).
    #     QML gaussiano = consistente anche con innovazioni non-normali (le code le mette l'FHS).
    #     warm = (α,γ,β) del refit precedente per warm-start (1 sola ottimizzazione → ~20× più veloce
    #     del multi-restart su finestra espandente, che faceva sforare il timeout).
    # EN: estimate (α, γ, β) via QMLE; ω from variance targeting → ω = σ̄²·(1 − α − γ/2 − β).
    #     Gaussian QML = consistent even under non-normal innovations (FHS supplies the tails).
    #     warm = previous-refit (α,γ,β) for warm-start (single optimization → ~20× faster than the
    #     multi-restart on an expanding window, which used to blow the timeout).
    r = np.asarray(r, dtype=float)
    uncond = float(np.var(r))
    if uncond <= 0 or not np.isfinite(uncond):
        return None

    def nll(theta):
        alpha, gamma, beta = theta
        persist = alpha + 0.5 * gamma + beta
        if alpha < 0 or beta < 0 or (alpha + gamma) < 0 or persist >= 0.999:
            return 1e10
        omega = uncond * (1.0 - persist)
        if omega <= 0:
            return 1e10
        sig = gjr_recursion(r, omega, alpha, gamma, beta, uncond)
        # IT: −logL gaussiano (QML) | EN: Gaussian (QML) negative log-likelihood
        ll = -0.5 * np.sum(np.log(2 * np.pi * sig ** 2) + (r / sig) ** 2)
        return -ll if np.isfinite(ll) else 1e10

    # IT: warm-start dal refit precedente, fallback ai params 1m del repo. | EN: warm-start else repo 1m params.
    starts = [warm] if warm is not None else [[0.05, 0.065, 0.875], [0.03, 0.10, 0.86]]
    best = None
    for x0 in starts:
        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 600})
        if best is None or res.fun < best.fun:
            best = res
    alpha, gamma, beta = best.x
    persist = alpha + 0.5 * gamma + beta
    omega = uncond * (1.0 - persist)
    return {"omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta,
            "uncond": uncond, "persist": persist}


# ─────────────────────────────────────────────────────────────────────────────
# Pricing FHS: fair value P-measure dello strangle/straddle inverso a 30h
# FHS pricing: P-measure fair value of the 30h inverse strangle/straddle
# ─────────────────────────────────────────────────────────────────────────────
def fhs_fair_value(spot, Kc, Kp, sig_entry, z_pool, params, n_paths, rng):
    # IT: simula n_paths × TENOR_H passi. Ad ogni step ε=σ·z (z bootstrap dai residui REALI ≤ entry),
    #     drift = −0.5σ² (martingala: E[S]≈spot), poi aggiorna la varianza GJR. Fair value = media
    #     dei payoff inversi sul terminale. Opzione inversa ⇒ payoff in BTC = intrinsic/S_T.
    # EN: simulate n_paths × TENOR_H steps. Each step ε=σ·z (z bootstrap from REAL residuals ≤ entry),
    #     drift = −0.5σ² (martingale: E[S]≈spot), then GJR variance update. Fair value = mean inverse
    #     payoff at terminal. Inverse option ⇒ BTC payoff = intrinsic/S_T.
    omega, alpha, gamma, beta = params["omega"], params["alpha"], params["gamma"], params["beta"]
    var_t = np.full(n_paths, sig_entry ** 2)
    cum = np.zeros(n_paths)
    z_idx = rng.integers(0, len(z_pool), size=(TENOR_H, n_paths))
    for t in range(TENOR_H):
        sig = np.sqrt(var_t)
        eps = sig * z_pool[z_idx[t]]
        cum += -0.5 * var_t + eps                      # drift martingala + shock
        neg = (eps < 0).astype(float)
        var_t = omega + (alpha + gamma * neg) * eps ** 2 + beta * var_t
        var_t = np.clip(var_t, 1e-12, 0.5 ** 2)
    S_T = spot * np.exp(cum)
    pay_c = np.maximum(0.0, S_T - Kc) / S_T            # call inversa | inverse call
    pay_p = np.maximum(0.0, Kp - S_T) / S_T            # put inversa  | inverse put
    return float(np.mean(pay_c + pay_p))


def realized_payoff(Kc, Kp, S_del):
    # IT: payoff inverso reale al delivery. | EN: real inverse payoff at delivery.
    return max(0.0, S_del - Kc) / S_del + max(0.0, Kp - S_del) / S_del


def fee_btc(leg_premium):
    # IT: fee per leg con cap 12.5% del premio (identico a short_vol_arm/04b). | EN: per-leg fee, 12.5% cap.
    return min(FEE_PER_LEG, FEE_CAP_FRAC * max(leg_premium, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────────────────
# IT: cache della serie causale: dipende SOLO da (refit_days, min_train) → identica tra le config
#     (struct/width) → evita di ri-rifittare il GARCH 7-anni N volte (A1 della code review).
# EN: causal-series cache: depends ONLY on (refit_days, min_train) → identical across configs
#     (struct/width) → avoids re-fitting the 7y GARCH N times (review item A1).
_CAUSAL_CACHE = {}


def precompute_causal(refit_days, min_train):
    # IT: calcola UNA volta candele, log-ret, σ_t e residui z_t CAUSALI + indici scadenza.
    #     Pesante (refit GJR expanding) ma indipendente da struct/width → cache-abile.
    # EN: computes ONCE candles, log-ret, causal σ_t and residuals z_t + expiry indices.
    #     Heavy (expanding GJR refit) but independent of struct/width → cacheable.
    key = (refit_days, min_train)
    if key in _CAUSAL_CACHE:
        return _CAUSAL_CACHE[key]
    df = pd.read_parquet(CANDLES)[["open_time", "close"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    close = df["close"].to_numpy(float)
    r = np.diff(np.log(close))                          # log-ret orari | hourly log-ret
    r = np.concatenate([[0.0], r])                      # allinea len a close | align length
    times = df["open_time"]

    # IT: indici delle barre 08:00 UTC = scadenze daily; entry = TENOR_H barre prima.
    # EN: indices of 08:00-UTC bars = daily expiries; entry = TENOR_H bars earlier.
    exp_idx = np.where((times.dt.hour == EXPIRY_HOUR).to_numpy())[0]

    # IT: σ_t e residui standardizzati z_t CAUSALI: refit params ogni refit_days su finestra
    #     espandente, propaga la recursion in avanti coi params correnti (no look-ahead).
    # EN: CAUSAL σ_t and standardized residuals z_t: refit params every refit_days on an expanding
    #     window, propagate the recursion forward with current params (no look-ahead).
    n = len(r)
    sig = np.full(n, np.nan)
    params_at = [None] * n
    cur = None
    last_fit = -10 ** 9
    refit_bars = refit_days * 24
    fit_window = 24 * 365 * 2                            # IT: fit su ultimi 2 anni (params stabili, causale, veloce) | EN: fit on last 2y
    for t in range(n):
        if t >= min_train and (cur is None or t - last_fit >= refit_bars):
            lo = max(1, t - fit_window)
            warm = [cur["alpha"], cur["gamma"], cur["beta"]] if cur is not None else None
            fit = fit_gjr(r[lo:t], warm=warm)           # solo passato, finestra cappata | past only, capped window
            if fit is not None:
                cur = fit
                last_fit = t
        params_at[t] = cur
        if cur is None:
            continue
        if not np.isfinite(sig[t - 1]):
            sig[t] = np.sqrt(cur["uncond"])
        else:
            neg = 1.0 if r[t - 1] < 0 else 0.0
            v = cur["omega"] + (cur["alpha"] + cur["gamma"] * neg) * r[t - 1] ** 2 + cur["beta"] * sig[t - 1] ** 2
            sig[t] = np.sqrt(max(v, 1e-12))
    z = np.where(np.isfinite(sig) & (sig > 0), r / sig, np.nan)  # residui standardizzati | std residuals

    pre = {"times": times, "close": close, "sig": sig, "z": z,
           "params_at": params_at, "exp_idx": exp_idx, "min_train": min_train}
    _CAUSAL_CACHE[key] = pre
    return pre


def price_trades(pre, width, struct, n_paths, rng):
    # IT: prezza le scadenze su una serie causale GIÀ calcolata (varia solo strike/FHS-pricing).
    #     rng resta PER-CONFIG (passato dal chiamante) → path FHS bit-identici al comportamento legacy.
    # EN: prices expiries on an ALREADY-computed causal series (only strike/FHS-pricing vary).
    #     rng stays PER-CONFIG (caller-supplied) → FHS paths bit-identical to legacy behaviour.
    times, close = pre["times"], pre["close"]
    sig, z, params_at = pre["sig"], pre["z"], pre["params_at"]
    min_train = pre["min_train"]
    trades = []
    for ei in pre["exp_idx"]:
        entry = ei - TENOR_H
        if entry < min_train or not np.isfinite(sig[entry]):
            continue
        prm = params_at[entry]
        if prm is None:
            continue
        zp = z[1:entry]
        zp = zp[np.isfinite(zp)]
        if len(zp) < 200:                               # pool residui insufficiente | thin residual pool
            continue
        spot = close[entry]
        S_del = close[ei]                               # IT: proxy delivery = close 08:00 UTC | EN: delivery proxy
        if struct == "strangle":
            Kc, Kp = spot * (1 + width), spot * (1 - width)
        else:                                           # straddle ATM
            Kc = Kp = spot
        fv = fhs_fair_value(spot, Kc, Kp, sig[entry], zp, prm, n_paths, rng)
        rp = realized_payoff(Kc, Kp, S_del)
        trades.append({"t_entry": times.iloc[entry].isoformat(), "spot": spot, "S_del": S_del,
                       "fair_value": fv, "realized_payoff": rp,
                       "moved_pct": 100 * (S_del / spot - 1), "sig_entry": float(sig[entry])})
    return pd.DataFrame(trades)


def run(width, struct, n_paths, refit_days, min_train, seed):
    # IT: wrapper retro-compatibile: precompute (cache) + pricing per-config. | EN: back-compat wrapper.
    rng = np.random.default_rng(seed)
    pre = precompute_causal(refit_days, min_train)
    return price_trades(pre, width, struct, n_paths, rng)


def vrp_table(tr, struct, width):
    # IT: per ogni VRP della griglia → premio = fair_value·(1+VRP), PnL netto, stats + break-even.
    # EN: per VRP grid point → premium = fair_value·(1+VRP), net PnL, stats + break-even.
    out = []
    fv = tr["fair_value"].to_numpy()
    rp = tr["realized_payoff"].to_numpy()
    for vrp in VRP_GRID:
        prem = fv * (1.0 + vrp)
        # IT: fee su 2 leg ≈ premio/2 ciascuna (cap 12.5%), vettoriale | EN: fee on 2 legs ≈ premium/2 each (12.5% cap), vectorized
        fees = 2.0 * np.minimum(FEE_PER_LEG, FEE_CAP_FRAC * np.maximum(prem / 2.0, 0.0))
        pnl = prem - rp - fees
        mean, sd = float(pnl.mean()), float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
        sharpe = float(mean / sd * np.sqrt(ANNUAL_BARS / TENOR_H)) if sd > 0 else 0.0
        worst = float(np.sort(pnl)[:5].sum())           # somma 5 trade peggiori | sum of 5 worst
        out.append({"vrp": vrp, "n": int(len(pnl)), "tot": float(pnl.sum()), "mean": mean,
                    "hit": float(100 * (pnl > 0).mean()), "sharpe_ann": sharpe,
                    "worst5_sum": worst, "p05": float(np.percentile(pnl, 5))})
    return out


def breakeven_vrp(table):
    # IT: VRP minimo per cui mean PnL ≥ 0 (interpolazione lineare tra i due punti che incrociano 0).
    # EN: minimal VRP s.t. mean PnL ≥ 0 (linear interp between the two points straddling 0).
    xs = [(row["vrp"], row["mean"]) for row in table]
    for (v0, m0), (v1, m1) in zip(xs, xs[1:]):
        if m0 < 0 <= m1:
            return round(v0 + (v1 - v0) * (-m0) / (m1 - m0), 4)
    if xs[0][1] >= 0:
        return 0.0
    return None                                         # IT: non raggiunto nella griglia | EN: not reached in grid


def empirical_vrp_anchor():
    # IT: àncora il sweep al VRP osservato sui 12 giorni REALI (implied mark_iv vs realized a 30h).
    #     Ritorna implied/realized medio sugli snapshot ATM disponibili (None se chain assente).
    # EN: anchor the sweep to the VRP observed over the 12 REAL days (implied mark_iv vs 30h realized).
    #     Returns mean implied/realized over available ATM snapshots (None if no chain).
    files = sorted(CHAIN_DIR.glob("*.parquet"))
    if not files:
        return None
    try:
        ch = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        atm = ch[ch["mark_iv"] > 0]
        if atm.empty:
            return None
        # IT: implied media ATM (proxy) — la realized 30h coerente è già nel backtest; qui solo livello IV.
        # EN: mean ATM implied (proxy) — coherent 30h realized lives in the backtest; here just IV level.
        return {"mean_mark_iv_pct": float(atm["mark_iv"].median())}
    except Exception:
        return None


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=4000, help="path MC per scadenza")
    ap.add_argument("--refit-days", type=int, default=90, help="cadenza refit GJR-GARCH (giorni)")
    ap.add_argument("--min-train", type=int, default=24 * 180, help="barre minime prima del 1° trade")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    anchor = empirical_vrp_anchor()
    print("=== SHORT-VOL HISTORICAL BACKTEST (FHS GJR-GARCH · 2019→2026) ===")
    print(f"  n_paths={args.n_paths} refit_days={args.refit_days} tenor={TENOR_H}h | "
          f"VRP grid={['%d%%' % (v*100) for v in VRP_GRID]}")
    if anchor:
        print(f"  àncora IV reale (12gg): mediana mark_iv ≈ {anchor['mean_mark_iv_pct']:.1f}%")
    print()

    configs = [("straddle", 0.0)] + [("strangle", w) for w in (0.04, 0.06, 0.08, 0.10)]
    report = {"meta": {"n_paths": args.n_paths, "refit_days": args.refit_days, "tenor_h": TENOR_H,
                       "vrp_grid": VRP_GRID, "iv_anchor": anchor}, "configs": []}

    for struct, w in configs:
        tr = run(w, struct, args.n_paths, args.refit_days, args.min_train, args.seed)
        if tr.empty:
            print(f"  {struct} {w:.0%}: n=0 (skip)")
            continue
        table = vrp_table(tr, struct, w)
        be = breakeven_vrp(table)
        tag = f"strangle {w:.0%}" if struct == "strangle" else "straddle ATM"
        be_s = f"{be:.1%}" if be is not None else ">50% (mai)"
        print(f"  ── {tag} | n={table[0]['n']} scadenze | break-even VRP = {be_s}")
        print(f"     {'VRP':>5} {'totPnL':>10} {'mean':>10} {'hit':>5} {'Sharpe':>7} {'p05':>9} {'worst5':>10}")
        for row in table:
            print(f"     {row['vrp']:>4.0%} {row['tot']:>+10.4f} {row['mean']:>+10.5f} "
                  f"{row['hit']:>4.0f}% {row['sharpe_ann']:>7.2f} {row['p05']:>+9.5f} {row['worst5_sum']:>+10.4f}")
        report["configs"].append({"struct": struct, "width": w, "n": table[0]["n"],
                                  "breakeven_vrp": be, "table": table,
                                  "moved_pct_p99": float(np.percentile(tr["moved_pct"].abs(), 99)),
                                  "moved_pct_max": float(tr["moved_pct"].abs().max())})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\n  → report in {OUT.relative_to(ROOT)}")
    print("  NB: VRP=0 ⇒ vendita a fair-value P (atteso ≈0 − fee − coda); il break-even VRP è la")
    print("      soglia implied/realized minima perché lo short-vol sopravviva alle code 2019-26.")


if __name__ == "__main__":
    main()
