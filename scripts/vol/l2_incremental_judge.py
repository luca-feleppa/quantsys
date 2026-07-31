# IT: B1 STADIO 1 — giudice pre-registrato (STATUS 2026-07-31): l'order-book L2 porta
#     informazione incrementale sulla RV a 3 ore, oltre a quella gia' nei lag di RV?
#     Due OLS ANNIDATE sugli stessi punti: baseline = HAR-C (la baseline di riferimento
#     adottata da C3), candidato = HAR-C + tre feature L2 all'ora t. Il confronto e'
#     appaiato in senso stretto: cambia solo il set di regressori.
#     ⚠ COSTANTI HARDCODED DI PROPOSITO (stesso pattern di hedged_vs_unhedged_judge):
#     sono pre-registrate, e leggerle da config le renderebbe modificabili a risultati
#     visti. Qualunque variante = NUOVA pre-registrazione.
#     ⚠ Perche' h=3 e non l'orizzonte di produzione (30): con un solo run L2 contiguo
#     di 410 ore, a h=30 restano n_eff=12.7 osservazioni effettive. A h=3 sono 96.3.
#     Questo stadio risponde a "L2 predice la RV a breve", NON a "L2 migliora la linea
#     vol" — quella e' lo stadio 2 e richiede anni di raccolta.
# EN: B1 STAGE 1 — pre-registered judge (STATUS 2026-07-31): does the L2 order book
#     carry incremental information about 3-hour RV, beyond what RV's own lags hold?
#     Two NESTED OLS on identical points: baseline = HAR-C (the reference baseline
#     adopted by C3), candidate = HAR-C + three L2 features at hour t. The comparison
#     is paired in the strict sense: only the regressor set changes.
#     ⚠ CONSTANTS HARDCODED ON PURPOSE (same pattern as hedged_vs_unhedged_judge):
#     they are pre-registered, and reading them from config would make them editable
#     once results are seen. Any variant = NEW pre-registration.
#     ⚠ Why h=3 and not the production horizon (30): with a single contiguous 410-hour
#     L2 run, h=30 leaves n_eff=12.7 effective observations. h=3 gives 96.3. This stage
#     answers "does L2 predict short-horizon RV", NOT "does L2 improve the vol line" —
#     that is stage 2 and needs years of collection.
import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging                                     # noqa: E402
from quantsys.model.vol_metrics import qlike, qlike_series, diebold_mariano, EPS  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.l2_judge")

# ── costanti PRE-REGISTRATE · PRE-REGISTERED constants ──────────────────────
H = 3                      # IT/EN: orizzonte in ore / horizon in hours
BURN = 120                 # IT/EN: burn-in della finestra espansiva / expanding-window burn-in
ALPHA = 0.01               # IT/EN: soglia dei DM / DM threshold
RATIO_MAX = 0.97           # IT/EN: materialita' (-3%) / materiality
N_MIN = 240                # IT/EN: previsioni OOS minime / minimum OOS forecasts
L2_COLS = ["ofi_abs", "log_depth", "dimb25_abs"]
HAR_C_COLS = ["xc_h", "xc_w", "xc_m"]
MIN_SNAP_PER_HOUR = 360    # IT/EN: soglia di copertura oraria L2 / L2 hourly coverage threshold

PX_PATH = "data/raw_candles_1m_l2.parquet"
L2_GLOB = "data/orderbook/l2_features_*.parquet"
OUT = Path("results/vols/l2_incremental_stage1.json")


# IT: RV e componenti HAR-C da barre a 1 MINUTO. Il target somma H ore SUCCESSIVE:
#     a 1m sono ~180 quadrati per osservazione contro i 3 che darebbero le barre
#     orarie — e' la ragione per cui il target e' costruito qui e non riusato dal
#     dataset di produzione. C = min(RV, BV) con BV = (pi/2)*sum|r_i||r_{i-1}|,
#     stessa definizione di `build_har_cj_frame` (single source concettuale).
# EN: RV and HAR-C components from 1-MINUTE bars. The target sums the NEXT H hours:
#     at 1m that is ~180 squares per observation against the 3 hourly bars would give
#     — the reason the target is built here rather than reused from the production
#     dataset. C = min(RV, BV) with BV = (pi/2)*sum|r_i||r_{i-1}|, the same definition
#     as `build_har_cj_frame` (conceptual single source).
def build_price_frame(px_path: str = PX_PATH) -> pd.DataFrame:
    px = pd.read_parquet(px_path, columns=["open_time", "close"])
    px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
    px = px.sort_values("open_time").set_index("open_time")
    lr = np.log(px["close"] / px["close"].shift(1)).dropna()

    rv = (lr ** 2).resample("1h").sum()
    bv = (lr.abs() * lr.abs().shift(1)).resample("1h").sum() * (np.pi / 2)
    c = np.minimum(rv, bv)                       # IT/EN: componente continua jump-robust

    # IT: target = RV sulle H ore SUCCESSIVE all'ora t (nessuna sovrapposizione con t).
    # EN: target = RV over the H hours AFTER hour t (no overlap with t itself).
    y = np.log(rv.shift(-1).rolling(H).sum().shift(-(H - 1)) + EPS)
    k = H / 24.0                                  # IT/EN: riscalamento all'orizzonte h
    return pd.DataFrame({
        "y":     y,
        "xc_h":  np.log(c.rolling(H).sum() + EPS),
        "xc_w":  np.log(c.rolling(7 * 24).sum() / 7 * k + EPS),
        "xc_m":  np.log(c.rolling(30 * 24).sum() / 30 * k + EPS),
        # IT/EN: naive persistence — RV trailing H ore, nota a fine ora t / trailing H-hour RV
        "naive": np.log(rv.rolling(H).sum() + EPS),
    })


# IT: feature L2 orarie — MEDIA degli snapshot a 5s dentro l'ora, versioni NON FIRMATE
#     (la varianza e' un momento pari). Le tre colonne sono pre-registrate e scelte da
#     diagnostiche target-free: `spread_bps` fu scartato per SNR 0.0, `imbalance_L*` per
#     collinearita' (rho 0.974). Si usa SOLO il run contiguo piu' lungo: le finestre a
#     cavallo di un buco accoppierebbero un book vecchio con un target nuovo.
# EN: hourly L2 features — MEAN of the 5s snapshots within the hour, UNSIGNED versions
#     (variance is an even moment). The three columns are pre-registered and were chosen
#     by target-free diagnostics: `spread_bps` was dropped for SNR 0.0, `imbalance_L*`
#     for collinearity (rho 0.974). ONLY the longest contiguous run is used: windows
#     straddling a gap would pair a stale book with a fresh target.
def build_l2_frame(pattern: str = L2_GLOB) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    cols = ["timestamp", "depth_imb_25bps", "ofi_best", "total_bid_qty", "total_ask_qty"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(pattern))])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")

    cnt = df.resample("1h").size()
    span = pd.date_range(cnt.index[0], cnt.index[-1], freq="1h")
    ok = cnt.reindex(span, fill_value=0) >= MIN_SNAP_PER_HOUR
    best, end, cur = 0, None, 0
    for i, v in enumerate(ok.values):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    run = span[end - best + 1: end + 1]
    sl = df.loc[(df.index >= run[0]) & (df.index < run[-1] + pd.Timedelta("1h"))]

    g = sl.groupby(sl.index.floor("h"))
    out = pd.DataFrame({
        "ofi_abs":    g["ofi_best"].apply(lambda s: s.abs().mean()),
        "log_depth":  g.apply(lambda d: np.log(d["total_bid_qty"] + d["total_ask_qty"]).mean()),
        "dimb25_abs": g["depth_imb_25bps"].apply(lambda s: s.abs().mean()),
    })
    return out.loc[out.index.isin(run)], run


# IT: PREVISIONI OOS A FINESTRA ESPANSIVA con EMBARGO — il punto delicato del giudice.
#     Per prevedere l'osservazione i (fine dell'ora t_i) si puo' usare solo cio' che a
#     quel momento e' NOTO. Il target dell'osservazione j copre le ore j+1..j+H, quindi
#     e' osservabile solo a fine ora j+H: la condizione e' j + H <= i, cioe' il train
#     arriva a i-H e le ultime H-1 osservazioni sono in EMBARGO. Senza questo taglio il
#     modello verrebbe addestrato su target che al momento della previsione non sono
#     ancora accaduti — leakage puro, e con finestre sovrapposte e' l'errore facile da
#     commettere e difficile da vedere (il numero resta plausibile).
# EN: EXPANDING-WINDOW OOS FORECASTS with EMBARGO — the judge's delicate point.
#     To forecast observation i (end of hour t_i) only what is KNOWN by then may be
#     used. Observation j's target covers hours j+1..j+H, so it is observable only at
#     the end of hour j+H: the condition is j + H <= i, i.e. training stops at i-H and
#     the last H-1 observations are EMBARGOED. Without this cut the model would be
#     trained on targets that have not happened yet at forecast time — pure leakage,
#     and with overlapping windows it is the easy mistake to make and the hard one to
#     see (the number stays plausible).
def expanding_oos(frame: pd.DataFrame, cols: list[str],
                  burn: int = BURN, h: int = H) -> np.ndarray:
    y = frame["y"].values
    X = np.column_stack([np.ones(len(frame)), frame[cols].values])
    pred = np.full(len(frame), np.nan)
    for i in range(burn, len(frame)):
        stop = i - h + 1                      # IT/EN: train = [0, stop) -> j <= i-h
        Xtr, ytr = X[:stop], y[:stop]
        beta, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        pred[i] = X[i] @ beta
    return pred


def main() -> int:
    # IT/EN: boilerplate UTF-8 (checklist nuovo script) / UTF-8 boilerplate
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="B1 stadio 1: giudice pre-registrato L2 incrementale su RV a 3h / "
                    "B1 stage 1: pre-registered incremental-L2 judge on 3h RV")
    ap.add_argument("--px", default=PX_PATH, help="parquet klines 1m / 1m klines parquet")
    ap.parse_args()

    price = build_price_frame()
    l2, run = build_l2_frame()
    log.info(f"run L2 contiguo / contiguous L2 run: {len(run)} ore "
             f"{run[0]:%Y-%m-%d %H:%M} -> {run[-1]:%Y-%m-%d %H:%M} UTC")

    df = price.join(l2, how="inner").dropna()
    # IT: guard sull'identita' del campione: i due modelli DEVONO vedere le stesse righe
    #     nello stesso ordine, altrimenti il confronto appaiato non e' appaiato.
    # EN: sample-identity guard: both models MUST see the same rows in the same order,
    #     otherwise the paired comparison is not paired.
    if not (df.index.is_monotonic_increasing and df.index.is_unique):
        raise RuntimeError("indice non monotono o con duplicati / non-monotonic or duplicated index")
    if not np.isfinite(df.values).all():
        raise RuntimeError("valori non finiti nel frame / non-finite values in the frame")
    log.info(f"osservazioni / observations: {len(df)}  "
             f"({df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M} UTC)")

    p_base = expanding_oos(df, HAR_C_COLS)
    p_cand = expanding_oos(df, HAR_C_COLS + L2_COLS)
    m = ~np.isnan(p_base)
    if not np.array_equal(m, ~np.isnan(p_cand)):
        raise RuntimeError("i due modelli non sono valutati sugli stessi punti / "
                           "the two models are not evaluated on the same points")

    rv_true = np.exp(df["y"].values[m])
    l_base = qlike_series(rv_true, np.exp(p_base[m]))
    l_cand = qlike_series(rv_true, np.exp(p_cand[m]))
    l_naive = qlike_series(rv_true, np.exp(df["naive"].values[m]))
    q_base, q_cand, q_naive = qlike(rv_true, np.exp(p_base[m])), \
        qlike(rv_true, np.exp(p_cand[m])), qlike(rv_true, np.exp(df["naive"].values[m]))
    n_eval = int(m.sum())

    dm = {}
    try:
        dm["cand_vs_base"] = diebold_mariano(l_cand, l_base, h=H)
        dm["base_vs_naive"] = diebold_mariano(l_base, l_naive, h=H)
    except Exception as e:  # noqa: BLE001
        log.warning(f"blocco DM fallito / DM block failed: {e}")

    ratio = q_cand / q_base
    d1 = dm.get("cand_vs_base", {})
    d4 = dm.get("base_vs_naive", {})
    cond = {
        # IT/EN: ① significativita' a favore del candidato / significance favouring the candidate
        "cond1_significant": bool(d1.get("p_value", 1.0) < ALPHA and d1.get("better") == "a"),
        # IT/EN: ② materialita' -3% / materiality
        "cond2_material": bool(ratio <= RATIO_MAX),
        # IT/EN: ③ validita' campione / sample validity
        "cond3_n_obs": bool(n_eval >= N_MIN),
        # IT/EN: ④ CONTROLLO POSITIVO — la baseline batte la naive / POSITIVE CONTROL
        "cond4_positive_control": bool(d4.get("p_value", 1.0) < ALPHA and d4.get("better") == "a"),
    }
    # IT: se ④ cade l'esito NON e' FAIL ma "nessuna conclusione": una baseline che non
    #     batte la persistenza non e' un metro con cui misurare alcunche'.
    # EN: if ④ fails the outcome is NOT a FAIL but "no conclusion": a baseline that does
    #     not beat persistence is not a yardstick for anything.
    verdict = ("NESSUNA_CONCLUSIONE" if not cond["cond4_positive_control"]
               else "PASS" if all(cond.values()) else "FAIL")

    rep = {
        "gate": "B1_stage1_l2_incremental", "pre_reg": "STATUS.md 2026-07-31",
        "horizon_hours": H, "burn_in": BURN, "alpha": ALPHA,
        "l2_features": L2_COLS, "baseline_cols": HAR_C_COLS,
        "run_hours": int(len(run)),
        "run_start": str(run[0]), "run_end": str(run[-1]),
        "n_obs": int(len(df)), "n_eval": n_eval, "n_eff": n_eval / H,
        "qlike_baseline": float(q_base), "qlike_candidate": float(q_cand),
        "qlike_naive": float(q_naive), "ratio_cand_over_base": float(ratio),
        "mse_log_baseline": float(np.mean((df["y"].values[m] - p_base[m]) ** 2)),
        "mse_log_candidate": float(np.mean((df["y"].values[m] - p_cand[m]) ** 2)),
        "diebold_mariano": dm, "conditions": cond, "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    print(f"\n══════ B1 STADIO 1 — L2 incrementale su RV {H}h ══════")
    print(f"  run L2: {len(run)} ore contigue · osservazioni {len(df)} · "
          f"previsioni OOS {n_eval} (n_eff {n_eval / H:.1f})")
    print(f"  baseline  HAR-C      QLIKE={q_base:.5f}  MSE(log)={rep['mse_log_baseline']:.4f}")
    print(f"  candidato HAR-C+L2   QLIKE={q_cand:.5f}  MSE(log)={rep['mse_log_candidate']:.4f}")
    print(f"  naive persistence    QLIKE={q_naive:.5f}")
    print(f"  ratio candidato/baseline = {ratio:.4f}   (soglia ② ≤ {RATIO_MAX})")
    for k, lab in (("cand_vs_base", "① cand vs base "), ("base_vs_naive", "④ base vs naive")):
        r = dm.get(k)
        if r and np.isfinite(r.get("dm_hln", float("nan"))):
            print(f"  DM {lab} stat={r['dm_hln']:+7.3f}  p={r['p_value']:.2e}  "
                  f"migliore/better={r['better']}")
    for k, v in cond.items():
        print(f"     {k:24s} = {v}")
    print(f"  VERDETTO: {verdict}   → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
