# IT: PAPER — BASELINE ECONOMETRICHE DIREZIONALI (pre-registrate in STATUS.md
#     2026-06-12). Completano il corpus del paper "Are price and volume enough?":
#     la claim "i momenti DISPARI (direzione) sono impredicibili OOS" è dimostrata
#     per il NN; qui si misura se vale anche per i predittori econometrici classici
#     → attribuisce il risultato all'INFORMAZIONE, non alla classe di modello.
#     Baseline (fit SOLO su train, valutazione su val E test, zero iterazioni):
#       (a) OLS "HAR-mean": y ~ [1, r_h, r_7d→h, r_30d→h] — l'analogo mean-equation
#           dell'HAR-RV (Corsi 2009) sui rendimenti aggregati;
#       (b) logit sul segno: P(y>0) ~ stessi regressori;
#       (c) momentum persistence: ŷ = r_h trailing (continuazione);
#       (d) train-mean costante (null di non-informatività).
#     Perimetro IDENTICO al NN: target raw y = Σ log-ret delle prossime h=30 barre
#     1h; split ricostruito replicando ESATTAMENTE il path di 01_download_data
#     (FeatureBuilder fit=False col wiring di 04b → canonico 104 → maschera
#     finestre NaN su T=120 → temporal_split 0.8/0.1/0.1), con verifica sui
#     conteggi noti del dataset (51130/6391/6392).
# EN: PAPER — DIRECTIONAL ECONOMETRIC BASELINES (pre-registered in STATUS.md
#     2026-06-12). They complete the corpus of the "Are price and volume enough?"
#     paper: the claim "ODD moments (direction) are unpredictable OOS" is proven
#     for the NN; here we measure whether it also holds for classical econometric
#     predictors → attributes the result to the INFORMATION, not the model class.
#     Baselines (fit on train ONLY, evaluated on val AND test, zero iterations):
#       (a) "HAR-mean" OLS: y ~ [1, r_h, r_7d→h, r_30d→h] — the mean-equation
#           analogue of HAR-RV (Corsi 2009) on aggregated returns;
#       (b) sign logit: P(y>0) ~ same regressors;
#       (c) momentum persistence: ŷ = trailing r_h (continuation);
#       (d) constant train-mean (non-informativeness null).
#     Perimeter IDENTICAL to the NN: raw target y = Σ log-ret of the next h=30
#     1h bars; split rebuilt by EXACTLY replicating the 01_download_data path
#     (FeatureBuilder fit=False with the 04b wiring → canonical 104 → NaN-window
#     mask over T=120 → temporal_split 0.8/0.1/0.1), verified against the known
#     dataset counts (51130/6391/6392).
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, PipelineState           # noqa: E402
from quantsys.features import FeatureBuilder, LIVE_DROP_FEATURES               # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.paper_dir")

EPS = 1e-12
# IT: conteggi attesi sul raw corrente (65.191 candele, run rs 2026-06-11). NB i
#     numeri citati in STATUS (51130/6391/6392) erano del probe 06-10 (65.159
#     candele): +32 candele = +26/+3/+3 finestre — verificato 2026-06-12.
# EN: expected counts on the current raw (65,191 candles, 2026-06-11 rs run). NB
#     the STATUS-quoted numbers (51130/6391/6392) were from the 06-10 probe
#     (65,159 candles): +32 candles = +26/+3/+3 windows — verified 2026-06-12.
EXPECTED_SPLITS = (51156, 6394, 6395)


def rebuild_split_timestamps(cfg: dict) -> dict:
    # IT: ricostruisce t_train/t_val/t_test replicando il path di 01: build con
    #     scaler iniettato (wiring 04b) → filtri canonici di 01 → logica finestre
    #     di create_windows (y/t presi alla riga window_size+j; finestra valida ⇔
    #     zero righe-NaN nelle T righe precedenti, via rolling-sum O(M)) → split.
    # EN: rebuilds t_train/t_val/t_test replicating the 01 path: build with the
    #     injected scaler (04b wiring) → 01 canonical filters → create_windows
    #     window logic (y/t taken at row window_size+j; window valid ⇔ zero
    #     NaN-rows in the previous T rows, via O(M) rolling sum) → split.
    ps = PipelineState.load("models/itransformer/pipeline_state.pkl")
    if str(cfg["data"]["interval"]) != str(ps.interval):
        raise RuntimeError(f"interval mismatch: config={cfg['data']['interval']} "
                           f"vs PipelineState={ps.interval}")
    fcfg, mcfg = cfg.get("features", {}), cfg.get("model", {})
    h = int(fcfg.get("forecast_horizon", 30))
    T = int(mcfg.get("window_size", 120))

    fb = FeatureBuilder(
        vp_bins=fcfg.get("vp_bins", 30), vp_lookback=fcfg.get("vp_lookback", 240),
        windows=fcfg.get("windows", [5, 10, 20, 60]),
        lag_periods=fcfg.get("lag_periods", 5), forecast_horizon=h,
        vp_stride=fcfg.get("vp_stride", 1), frac_diff_d=fcfg.get("frac_diff_d", 0.0),
        use_revin=bool(mcfg.get("use_revin", False)),
        interval_minutes=ps.interval_minutes)
    fb.scaler, fb._scale_cols = ps.scaler, list(ps.scale_cols)
    fb.scalers, fb.clip_lo_, fb.clip_hi_ = dict(ps.price_scaler_state), ps.clip_lo_, ps.clip_hi_
    fb.feature_cols, fb.n_dynamic_features = list(ps.feature_cols), ps.n_dynamic_features

    candles = pd.read_parquet("data/raw_candles.parquet").sort_values("open_time").reset_index(drop=True)
    funding = pd.read_parquet("data/funding_rate.parquet")
    feat = fb.build(candles, fit=False, normalize=True, funding_df=funding)

    exclude = {"open_time", "close_time", "date_utc", "pv", "cum_pv", "cum_vol",
               "typical_price", "obv", "target_ret", "target_dir"}
    cols = [c for c in fb.feature_cols
            if c not in exclude and c in feat.columns
            and feat[c].dtype in ["float64", "float32"] and c not in LIVE_DROP_FEATURES]
    cols = [c for c in cols if feat[c].isna().mean() <= 0.5]
    cols = [c for c in cols if not np.isinf(feat[c].values).any()]
    log.info(f"canonico ricostruito: {len(cols)} feature")

    # IT: replica di create_windows senza materializzare X: finestra j copre le
    #     righe j..j+T-1; max_idx = M-T-1; t[j] = open_time[T+j]; valida ⇔ nessuna
    #     riga con NaN tra le sue T righe.
    # EN: create_windows replica without materializing X: window j covers rows
    #     j..j+T-1; max_idx = M-T-1; t[j] = open_time[T+j]; valid ⇔ none of its
    #     T rows has a NaN.
    M = len(feat)
    rownan = feat[cols].isna().any(axis=1).to_numpy().astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(rownan)])
    max_idx = M - T - 1
    win_nan = csum[T:T + max_idx] - csum[0:max_idx]   # IT/EN: n. righe NaN nella finestra j
    valid = win_nan == 0
    t_all = pd.to_datetime(feat["open_time"].values[T:T + max_idx][valid])

    n = len(t_all)
    iv = int(n * 0.8)
    it = int(n * 0.9)
    counts = (iv, it - iv, n - it)
    log.info(f"split ricostruito: train/val/test = {counts} (attesi {EXPECTED_SPLITS})")
    if counts != EXPECTED_SPLITS:
        log.warning("conteggi DIVERSI dagli attesi — scostamento riportato nel JSON, "
                    "verificare l'allineamento prima dell'uso nel paper")
    return {"t_train": t_all[:iv], "t_val": t_all[iv:it], "t_test": t_all[it:],
            "counts": counts, "h": h}


def main():
    # IT: boilerplate UTF-8 (checklist CLAUDE.md) | EN: UTF-8 boilerplate (CLAUDE.md checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config("config/default.yaml")
    splits = rebuild_split_timestamps(cfg)
    h = splits["h"]
    bars_day = 24

    # IT: target raw + regressori trailing dai raw candles (stessa metodologia dei
    #     giudici vol: tutto indicizzato per open_time, allineato per intersezione).
    # EN: raw target + trailing regressors from raw candles (same methodology as
    #     the vol judges: everything indexed by open_time, intersection-aligned).
    raw = pd.read_parquet("data/raw_candles.parquet").sort_values("open_time").reset_index(drop=True)
    lr = np.log(raw["close"] / raw["close"].shift(1))
    df = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":  lr.rolling(h).sum().shift(-h),                    # IT/EN: Σ log-ret prossime h barre
        "rh": lr.rolling(h).sum(),                               # IT/EN: trailing h barre
        "rw": lr.rolling(7 * bars_day).sum() * (h / (7 * bars_day)),    # IT/EN: 7d riscalato a h
        "rm": lr.rolling(30 * bars_day).sum() * (h / (30 * bars_day)),  # IT/EN: 30d riscalato a h
    }).dropna().set_index("open_time")
    df.index = pd.to_datetime(df.index).tz_localize(None)

    tr = df.loc[df.index.intersection(pd.DatetimeIndex(splits["t_train"]).tz_localize(None))]
    log.info(f"allineamento train: {len(tr)}/{len(splits['t_train'])}")

    # IT: fit train-only: OLS chiuso + logit sul segno (stessi regressori).
    # EN: train-only fit: closed-form OLS + sign logit (same regressors).
    REG = ["rh", "rw", "rm"]
    Xtr = np.column_stack([np.ones(len(tr)), tr[REG].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    log.info(f"OLS beta: const={beta[0]:.2e} rh={beta[1]:.4f} rw={beta[2]:.4f} rm={beta[3]:.4f}")
    from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(max_iter=1000)
    logit.fit(tr[REG].values, (tr["y"].values > 0).astype(int))
    y_mean_train = float(tr["y"].mean())
    base_rate_train = float((tr["y"] > 0).mean())

    out_dir = Path("results/paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        ev = df.loc[df.index.intersection(pd.DatetimeIndex(splits[f"t_{split}"]).tz_localize(None))]
        y = ev["y"].values
        n = len(y)
        Xev = np.column_stack([np.ones(n), ev[REG].values])
        preds = {
            "ols_har_mean": Xev @ beta,
            "logit_sign":   logit.predict_proba(ev[REG].values)[:, 1] - 0.5,
            "momentum":     ev["rh"].values,
            "train_mean":   np.full(n, y_mean_train),
        }
        res = {}
        for name, p in preds.items():
            # IT: Spearman indefinito sulla costante → 0 by convention (null).
            # EN: Spearman undefined on the constant → 0 by convention (null).
            rho = 0.0 if np.allclose(p, p[0]) else float(spearmanr(p, y).statistic)
            sign_da = float(np.mean(np.sign(p if name != "train_mean" else np.full(n, y_mean_train)) == np.sign(y)))
            res[name] = {
                "spearman": rho,
                "sign_da":  sign_da,
                "mse":      float(np.mean((y - p) ** 2)) if name != "logit_sign" else None,
            }
            log.info(f"[{split}] {name:13s} ρ={rho:+.4f}  signDA={sign_da:.4f}")
        report = {
            "split": split, "n_obs": n, "h": h,
            "split_counts": list(splits["counts"]),
            "split_counts_expected": list(EXPECTED_SPLITS),
            "spearman_signif_2se": 2.0 / np.sqrt(n),
            "base_rate_up_train": base_rate_train,
            "ols_beta": list(map(float, beta)),
            "metrics": res,
        }
        out = out_dir / f"dir_baselines_1h_{split}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n══════ BASELINE DIREZIONALI [1h·{split}] (n={n}, 2/√n={2/np.sqrt(n):.4f}) ══════")
        for name in preds:
            r = res[name]
            print(f"  {name:13s} ρ={r['spearman']:+.4f}  signDA={r['sign_da']:.4f}")
        print(f"  → {out}")


if __name__ == "__main__":
    main()
