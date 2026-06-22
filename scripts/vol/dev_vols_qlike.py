# IT: VOL-S — GIUDICE QLIKE (pre-registrato in STATUS.md 2026-06-10).
#     Confronta su val/test la previsione di realized variance a h barre
#     (h = features.forecast_horizon; risoluzione barra = data.interval, parametrici):
#       · NN (ensemble iTransformer, target log-RV, z-score → raw via center+scale
#         del RobustScaler: NB denormalize_predictions NON basta, il log-RV ha
#         mediana ≈ −7, serve anche il centro)
#       · HAR-RV (Corsi 2009): OLS su log-RV con componenti trailing h-barre/7d/30d,
#         fit SOLO su train (stesso information set del NN)
#       · naive persistence: RV_pred = RV trailing h barre (floor di sanità)
#     Giudice primario: QLIKE su RV in livelli, exp(log_pred) per TUTTI (stessa
#     trasformazione → confronto fair). Secondario: MSE su log-RV.
#     GATE (test): QLIKE_NN ≤ 0.95·QLIKE_HAR  E  QLIKE_NN < QLIKE_naive.
#     Protocollo: val-first; il test si valuta UNA volta.
# EN: VOL-S — QLIKE JUDGE (pre-registered in STATUS.md 2026-06-10).
#     Compares h-bar realized-variance forecasts on val/test
#     (h = features.forecast_horizon; bar resolution = data.interval, both parametric):
#       · NN (iTransformer ensemble, log-RV target, z-score → raw via the
#         RobustScaler's center+scale: NB denormalize_predictions is NOT enough,
#         log-RV has median ≈ −7, the center is required too)
#       · HAR-RV (Corsi 2009): OLS on log-RV with trailing h-bar/7d/30d components,
#         fit on train ONLY (same information set as the NN)
#       · naive persistence: RV_pred = trailing h-bar RV (sanity floor)
#     Primary judge: QLIKE on RV levels, exp(log_pred) for ALL (same transform →
#     fair comparison). Secondary: MSE on log-RV.
#     GATE (test): QLIKE_NN ≤ 0.95·QLIKE_HAR  AND  QLIKE_NN < QLIKE_naive.
#     Protocol: val-first; test is evaluated ONCE.
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, interval_minutes_from_cfg, models_root  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vols_qlike")

EPS = 1e-12  # IT: stesso ε del target in FeatureBuilder | EN: same ε as the FeatureBuilder target
# IT: default/fallback dell'orizzonte in barre — il valore effettivo viene letto
#     da cfg["features"]["forecast_horizon"] dentro main().
# EN: default/fallback for the horizon in bars — the effective value is read
#     from cfg["features"]["forecast_horizon"] inside main().
H = 30


# IT: QLIKE su RV in livelli — loss canonica per la valutazione di varianza
#     (Patton 2011: robusta al rumore nella proxy di RV).
# EN: QLIKE on RV levels — canonical variance-forecast loss
#     (Patton 2011: robust to noise in the RV proxy).
def qlike(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    r = rv_true / np.maximum(rv_pred, EPS)
    return float(np.mean(r - np.log(r) - 1.0))


def main():
    # IT: boilerplate UTF-8 (checklist CLAUDE.md) | EN: UTF-8 boilerplate (CLAUDE.md checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # IT: argparse minimale — arch del modello da giudicare (models/{arch}); flag
    #     CLI esplicito, NON env QUANTSYS_ARCH — default itransformer = run storica
    #     bit-identica.
    # EN: minimal argparse — model arch to judge (models/{arch}); explicit CLI flag,
    #     NOT the QUANTSYS_ARCH env var — default itransformer = bit-identical
    #     legacy run.
    ap = argparse.ArgumentParser(description="Giudice QLIKE vol-S (NN vs HAR-RV vs naive) / "
                                             "VOL-S QLIKE judge (NN vs HAR-RV vs naive)")
    ap.add_argument("--arch", default="itransformer",
                    choices=["itransformer", "nhits", "tcnmamba", "lstm"],
                    help="architettura del modello vol da caricare (models/{arch}) / "
                         "vol model architecture to load (models/{arch})")
    args = ap.parse_args()
    # IT: root env-aware (QUANTSYS_MODELS_ROOT) — giudica la sandbox isolata se attiva.
    # EN: env-aware root (QUANTSYS_MODELS_ROOT) — judges the isolated sandbox if set.
    model_dir = models_root() / args.arch
    log.info(f"dir modelli effettiva / effective model dir: {model_dir} (arch={args.arch})")

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config features.target_type deve essere log_rv per il giudice vol-S"

    # IT: orizzonte e risoluzione barra dalla config (H modulo = solo fallback) —
    #     il giudice resta coerente con il dataset/target correnti senza hardcode.
    # EN: horizon and bar resolution from the config (module H = fallback only) —
    #     keeps the judge consistent with the current dataset/target, no hardcoding.
    h = int(cfg["features"].get("forecast_horizon", H))
    interval = cfg["data"]["interval"]
    bars_day = 1440 // interval_minutes_from_cfg(cfg)  # IT: barre/giorno dall'interval | EN: bars/day from interval
    log.info(f"horizon h={h} barre · interval={interval} · bars/day={bars_day}")

    # IT: split da giudicare — val di default (val-first); test SOLO a sanity val superata.
    # EN: split to judge — val by default (val-first); test ONLY once val sanity passes.
    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    assert split in ("val", "test")

    # ── Ground truth + feature HAR dai raw candles (stessa definizione del target) ──
    raw = pd.read_parquet("data/raw_candles.parquet")
    raw = raw.sort_values("open_time").reset_index(drop=True)
    lr2 = np.log(raw["close"] / raw["close"].shift(1)) ** 2

    rv_h = lr2.rolling(h).sum()                      # IT: RV trailing su h barre | EN: trailing h-bar RV
    rv_w = lr2.rolling(7 * bars_day).sum() / 7       # IT: media giornaliera su 7gg | EN: 7d daily mean
    rv_m = lr2.rolling(30 * bars_day).sum() / 30     # IT: media giornaliera su 30gg | EN: 30d daily mean
    rv_fwd = lr2.rolling(h).sum().shift(-h)          # IT/EN: target (stessa formula del FeatureBuilder)

    har = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":  np.log(rv_fwd + EPS),
        "xh": np.log(rv_h + EPS),
        # IT: componenti weekly/monthly riscalate all'orizzonte h per coerenza dimensionale
        # EN: weekly/monthly components rescaled to the h-bar horizon for dimensional consistency
        "xw": np.log(rv_w * (h / bars_day) + EPS),
        "xm": np.log(rv_m * (h / bars_day) + EPS),
    }).dropna().set_index("open_time")

    # ── Allineamento ai timestamp degli split del dataset NN ────────────────────
    d = np.load("data/lstm_dataset.npz", allow_pickle=True)
    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)
    t_eval  = pd.to_datetime(d[f"t_{split}"]).tz_localize(None)
    har.index = pd.to_datetime(har.index).tz_localize(None)

    tr = har.loc[har.index.intersection(t_train)]
    ev = har.loc[har.index.intersection(t_eval)]
    log.info(f"HAR rows: train {len(tr)}/{len(t_train)}  {split} {len(ev)}/{len(t_eval)}")
    assert len(ev) >= 0.95 * len(t_eval), "allineamento HAR↔split insufficiente"

    # ── Baseline 1: HAR-RV (OLS chiuso, fit su train) ───────────────────────────
    Xtr = np.column_stack([np.ones(len(tr)), tr[["xh", "xw", "xm"]].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    Xev = np.column_stack([np.ones(len(ev)), ev[["xh", "xw", "xm"]].values])
    log_pred_har = Xev @ beta
    log.info(f"HAR beta: const={beta[0]:.3f} h={beta[1]:.3f} w={beta[2]:.3f} m={beta[3]:.3f}")

    # ── Baseline 2: naive persistence ───────────────────────────────────────────
    log_pred_naive = ev["xh"].values

    # ── NN: ensemble forward su X_{split} → z → log-RV raw (center+scale) ───────
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.utils import PipelineState
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load(str(model_dir), device)
    ps = PipelineState.load(str(model_dir / "pipeline_state.pkl"))
    idx = ps.scale_cols.index("target_ret")
    c, s = float(ps.scaler.center_[idx]), float(ps.scaler.scale_[idx])
    log.info(f"target_ret scaler: center={c:.3f} scale={s:.3f} (deve essere ~log-RV, non ~0)")
    assert c < -3, "center ≈ 0 → il PipelineState non è del dataset log-RV (stale?)"

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
    # IT: inversione COMPLETA z→raw: μ·IQR + centro (vedi nota in testa).
    # EN: FULL z→raw inversion: μ·IQR + center (see header note).
    log_pred_nn_full = mu_z * s + c

    # IT: riallinea le predizioni NN ai soli timestamp presenti in `ev`.
    # EN: re-align NN predictions to the timestamps present in `ev`.
    pos = {ts: k for k, ts in enumerate(t_eval)}
    sel = np.array([pos[ts] for ts in ev.index])
    log_pred_nn = log_pred_nn_full[sel]

    # ── Giudizio ────────────────────────────────────────────────────────────────
    rv_true = np.exp(ev["y"].values)  # IT/EN: = rv_fwd + EPS
    res = {}
    for name, lp in [("nn", log_pred_nn), ("har", log_pred_har), ("naive", log_pred_naive)]:
        res[name] = {
            "qlike":   qlike(rv_true, np.exp(lp)),
            "mse_log": float(np.mean((ev["y"].values - lp) ** 2)),
        }
        log.info(f"{name:6s} QLIKE={res[name]['qlike']:.5f}  MSE(log)={res[name]['mse_log']:.4f}")

    gate = {
        "split": split,
        "nn_vs_har_ratio": res["nn"]["qlike"] / res["har"]["qlike"],
        "beats_har_5pct": bool(res["nn"]["qlike"] <= 0.95 * res["har"]["qlike"]),
        "beats_naive":    bool(res["nn"]["qlike"] < res["naive"]["qlike"]),
        "n_obs": int(len(ev)),
    }
    gate["verdict"] = "PASS" if (gate["beats_har_5pct"] and gate["beats_naive"]) else "FAIL"

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
    # IT: report suffissato per interval+split — run a risoluzioni diverse non si sovrascrivono.
    # EN: report suffixed by interval+split — runs at different resolutions do not overwrite.
    out_path = out_dir / f"qlike_report_{interval}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": res, "gate": gate, "har_beta": list(map(float, beta))}, f, indent=2)

    print(f"\n══════ VOL-S QLIKE [{interval}·{split}] ══════")
    for name in ("nn", "har", "naive"):
        print(f"  {name:6s} QLIKE={res[name]['qlike']:.5f}  MSE(log)={res[name]['mse_log']:.4f}")
    print(f"  NN/HAR ratio: {gate['nn_vs_har_ratio']:.4f}  (gate ≤ 0.95)")
    print(f"  VERDETTO [{split}]: {gate['verdict']}   → {out_path}")


if __name__ == "__main__":
    main()
