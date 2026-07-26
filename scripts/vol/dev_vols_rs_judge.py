# IT: PROBE SEMIVARIANZA — GIUDICE HAR-RS (pre-registrato in STATUS.md 2026-06-11).
#     Confronta su val/test la previsione dell'ASIMMETRIA della semivarianza realizzata
#     futura a h barre: y = log((RS⁺_fwd+ε)/(RS⁻_fwd+ε)), RS± = Σᵢ₌₁..ₕ r²ₜ₊ᵢ·1[rₜ₊ᵢ≷0]
#     (Barndorff-Nielsen–Kinnebrock–Shephard 2010; Patton–Sheppard 2015).
#       · NN (ensemble iTransformer, target log_rs_ratio, inversione COMPLETA z→raw
#         μ·IQR + centro dal RobustScaler persistito — lezione del run log-RV)
#       · HAR-RS (stile Patton–Sheppard): OLS su [1, lratio_h, lratio_7d, lratio_30d,
#         log_rv_h] trailing, fit SOLO su train (stesso information set del NN).
#         NB: il log-ratio è scale-free → niente riscalatura h/bars_day delle
#         componenti weekly/monthly (serviva solo per i LIVELLI di RV).
#       · naive persistence: lratio trailing h barre
#       · train-mean: costante = media del target su train (null di non-informatività)
#     Metrica primaria: MSE sul log-ratio (QLIKE non applicabile: il log-ratio non è
#     positivo-definito). Secondarie: Spearman, sign-DA su 1[RS⁺>RS⁻].
#     GATE PRIMARIO (test): MSE_NN ≤ 0.95·MSE_HAR-RS  E  MSE_NN < MSE_naive
#                            E  MSE_NN < MSE_train-mean.
#     GATE SECONDARIO (economico): sign-DA_NN ≥ 0.55  E  sign-DA_NN > sign-DA_HAR-RS.
#     Protocollo: val-first; il test si valuta UNA volta; zero iterazioni su FAIL.
# EN: SEMIVARIANCE PROBE — HAR-RS JUDGE (pre-registered in STATUS.md 2026-06-11).
#     Compares h-bar forecasts of the future realized-semivariance ASYMMETRY on
#     val/test: y = log((RS⁺_fwd+ε)/(RS⁻_fwd+ε)), RS± = Σᵢ₌₁..ₕ r²ₜ₊ᵢ·1[rₜ₊ᵢ≷0]
#     (Barndorff-Nielsen–Kinnebrock–Shephard 2010; Patton–Sheppard 2015).
#       · NN (iTransformer ensemble, log_rs_ratio target, FULL z→raw inversion
#         μ·IQR + center from the persisted RobustScaler — log-RV run lesson)
#       · HAR-RS (Patton–Sheppard style): OLS on [1, lratio_h, lratio_7d, lratio_30d,
#         log_rv_h] trailing, fit on train ONLY (same information set as the NN).
#         NB: the log-ratio is scale-free → no h/bars_day rescaling of the
#         weekly/monthly components (that was needed for RV LEVELS only).
#       · naive persistence: trailing h-bar lratio
#       · train-mean: constant = train target mean (no-sign-information null)
#     Primary metric: MSE on the log-ratio (QLIKE not applicable: the log-ratio is
#     not positive-definite). Secondary: Spearman, sign-DA on 1[RS⁺>RS⁻].
#     PRIMARY GATE (test): MSE_NN ≤ 0.95·MSE_HAR-RS  AND  MSE_NN < MSE_naive
#                           AND  MSE_NN < MSE_train-mean.
#     SECONDARY GATE (economic): sign-DA_NN ≥ 0.55  AND  sign-DA_NN > sign-DA_HAR-RS.
#     Protocol: val-first; test evaluated ONCE; zero iterations on FAIL.
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, interval_minutes_from_cfg  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vols_rs_judge")

EPS = 1e-12  # IT: stesso ε del target in FeatureBuilder | EN: same ε as the FeatureBuilder target
H = 30       # IT: fallback orizzonte (effettivo da config) | EN: horizon fallback (effective from config)


# IT: metriche per un set di predizioni vs verità — MSE (primaria), Spearman, sign-DA.
# EN: metrics for one prediction set vs truth — MSE (primary), Spearman, sign-DA.
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho = spearmanr(y_true, y_pred).statistic if np.std(y_pred) > 0 else 0.0
    return {
        "mse":      float(np.mean((y_true - y_pred) ** 2)),
        "spearman": float(rho),
        "sign_da":  float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rs_ratio", \
        "config features.target_type deve essere log_rs_ratio per il giudice semivarianza"

    h = int(cfg["features"].get("forecast_horizon", H))
    interval = cfg["data"]["interval"]
    bars_day = 1440 // interval_minutes_from_cfg(cfg)
    log.info(f"horizon h={h} barre · interval={interval} · bars/day={bars_day}")

    # IT: split da giudicare — val di default (val-first); test SOLO a sanity val superata.
    # EN: split to judge — val by default (val-first); test ONLY once val sanity passes.
    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    assert split in ("val", "test")

    # ── Ground truth + componenti HAR-RS dai raw candles (stessa definizione del target) ──
    raw = pd.read_parquet("data/raw_candles.parquet")
    raw = raw.sort_values("open_time").reset_index(drop=True)
    lr = np.log(raw["close"] / raw["close"].shift(1))
    sq_pos = lr.clip(lower=0.0) ** 2
    sq_neg = lr.clip(upper=0.0) ** 2

    # IT: log-ratio trailing su finestre h / 7d / 30d (scale-free: la lunghezza finestra
    #     si cancella nel rapporto) + livello log-RV trailing h (regressore pre-registrato).
    # EN: trailing log-ratio over h / 7d / 30d windows (scale-free: window length cancels
    #     in the ratio) + trailing h-bar log-RV level (pre-registered regressor).
    def lratio(win: int) -> pd.Series:
        return np.log(sq_pos.rolling(win).sum() + EPS) - np.log(sq_neg.rolling(win).sum() + EPS)

    rv_h = (sq_pos + sq_neg).rolling(h).sum()
    y_fwd = (np.log(sq_pos.rolling(h).sum().shift(-h) + EPS)
             - np.log(sq_neg.rolling(h).sum().shift(-h) + EPS))  # IT/EN: stessa formula del FeatureBuilder

    har = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":   y_fwd,
        "xh":  lratio(h),
        "xw":  lratio(7 * bars_day),
        "xm":  lratio(30 * bars_day),
        "xrv": np.log(rv_h + EPS),
    }).dropna().set_index("open_time")

    # ── Allineamento ai timestamp degli split del dataset NN ────────────────────
    d = np.load("data/lstm_dataset.npz", allow_pickle=True)
    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)
    t_eval  = pd.to_datetime(d[f"t_{split}"]).tz_localize(None)
    har.index = pd.to_datetime(har.index).tz_localize(None)

    tr = har.loc[har.index.intersection(t_train)]
    ev = har.loc[har.index.intersection(t_eval)]
    log.info(f"HAR-RS rows: train {len(tr)}/{len(t_train)}  {split} {len(ev)}/{len(t_eval)}")
    assert len(ev) >= 0.95 * len(t_eval), "allineamento HAR-RS↔split insufficiente"

    cols = ["xh", "xw", "xm", "xrv"]
    # ── Baseline 1: HAR-RS (OLS chiuso, fit su train) ───────────────────────────
    Xtr = np.column_stack([np.ones(len(tr)), tr[cols].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    Xev = np.column_stack([np.ones(len(ev)), ev[cols].values])
    pred_har = Xev @ beta
    log.info("HAR-RS beta: " + " ".join(f"{n}={b:.3f}" for n, b in zip(["const"] + cols, beta)))

    # ── Baseline 2: naive persistence · Baseline 3: train-mean ──────────────────
    pred_naive = ev["xh"].values
    pred_mean  = np.full(len(ev), float(tr["y"].mean()))

    # ── NN: ensemble forward su X_{split} → z → log-ratio raw (centro+scala) ────
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.utils import PipelineState
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load("models/itransformer", device)
    ps = PipelineState.load("models/itransformer/pipeline_state.pkl")
    idx = ps.scale_cols.index("target_ret")
    c, s = float(ps.scaler.center_[idx]), float(ps.scaler.scale_[idx])
    log.info(f"target_ret scaler: center={c:.3f} scale={s:.3f} (log-ratio ⇒ quasi-centrato)")
    # IT: sanity: il log-ratio è quasi-centrato (|c|<2) — un centro ≈ −7 indicherebbe
    #     un PipelineState stale del run log-RV.
    # EN: sanity: the log-ratio is near-centered (|c|<2) — a center ≈ −7 would flag
    #     a stale PipelineState from the log-RV run.
    assert abs(c) < 2, "center fuori range → PipelineState non è del dataset log_rs_ratio (stale?)"

    X = torch.tensor(d[f"X_{split}"], dtype=torch.float32)
    Xm = torch.tensor(d[f"X_macro_{split}"], dtype=torch.float32) if f"X_macro_{split}" in d.files else None
    mus = []
    # IT: EnsembleModel.__call__ → (mu_ens, sigma_ens, nu_ens) in spazio z (AMP off interno).
    # EN: EnsembleModel.__call__ → (mu_ens, sigma_ens, nu_ens) in z-space (AMP off internally).
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i + 256].to(device)
            xmb = Xm[i:i + 256].to(device) if Xm is not None else None
            mu, _, _ = model(xb, xmb)
            mus.append(mu.detach().cpu().numpy().ravel())
    mu_z = np.concatenate(mus)
    # IT: inversione COMPLETA z→raw: μ·IQR + centro. | EN: FULL z→raw inversion: μ·IQR + center.
    pred_nn_full = mu_z * s + c

    # IT: riallinea le predizioni NN ai soli timestamp presenti in `ev`.
    # EN: re-align NN predictions to the timestamps present in `ev`.
    pos = {ts: k for k, ts in enumerate(t_eval)}
    sel = np.array([pos[ts] for ts in ev.index])
    pred_nn = pred_nn_full[sel]

    # ── Giudizio ────────────────────────────────────────────────────────────────
    y = ev["y"].values
    res = {name: metrics(y, p) for name, p in
           [("nn", pred_nn), ("har_rs", pred_har), ("naive", pred_naive), ("train_mean", pred_mean)]}
    for name, m in res.items():
        log.info(f"{name:10s} MSE={m['mse']:.5f}  ρ={m['spearman']:+.4f}  signDA={m['sign_da']:.4f}")

    gate = {
        "split": split,
        "nn_vs_har_mse_ratio": res["nn"]["mse"] / res["har_rs"]["mse"],
        "beats_har_5pct":   bool(res["nn"]["mse"] <= 0.95 * res["har_rs"]["mse"]),
        "beats_naive":      bool(res["nn"]["mse"] < res["naive"]["mse"]),
        "beats_train_mean": bool(res["nn"]["mse"] < res["train_mean"]["mse"]),
        "secondary_sign_da_055":  bool(res["nn"]["sign_da"] >= 0.55),
        "secondary_beats_har_da": bool(res["nn"]["sign_da"] > res["har_rs"]["sign_da"]),
        "n_obs": int(len(ev)),
    }
    gate["primary_verdict"] = "PASS" if (gate["beats_har_5pct"] and gate["beats_naive"]
                                         and gate["beats_train_mean"]) else "FAIL"
    gate["secondary_verdict"] = "PASS" if (gate["secondary_sign_da_055"]
                                           and gate["secondary_beats_har_da"]) else "FAIL"

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
    # IT: report suffissato per interval+split — run a risoluzioni diverse non si sovrascrivono.
    # EN: report suffixed by interval+split — runs at different resolutions do not overwrite.
    out_path = out_dir / f"rs_report_{interval}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": res, "gate": gate, "har_beta": list(map(float, beta))}, f, indent=2)

    print(f"\n══════ SEMIVARIANCE RS-RATIO [{interval}·{split}] ══════")
    for name in ("nn", "har_rs", "naive", "train_mean"):
        m = res[name]
        print(f"  {name:10s} MSE={m['mse']:.5f}  ρ={m['spearman']:+.4f}  signDA={m['sign_da']:.4f}")
    print(f"  NN/HAR-RS MSE ratio: {gate['nn_vs_har_mse_ratio']:.4f}  (gate ≤ 0.95)")
    print(f"  VERDETTO PRIMARIO [{split}]: {gate['primary_verdict']}"
          f"   · SECONDARIO (sign): {gate['secondary_verdict']}   → {out_path}")


if __name__ == "__main__":
    main()
