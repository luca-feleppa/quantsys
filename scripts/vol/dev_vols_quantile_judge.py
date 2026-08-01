# IT: A2a — GIUDICE CALIBRAZIONE QUANTILI log-RV (pre-registrato in STATUS.md 2026-07-08).
#     Il modello vol PASS è già loss_type=quantile (QUANTILES=[.1,.25,.5,.75,.9]):
#     questo giudice ESTRAE i quantili per-membro mai usati (EnsembleModel li
#     collassa in μ=q50), li fonde cross-membro (Vincentization pesata) e ne
#     misura su val/test:
#       · coverage empirica per livello: P(y ≤ q_τ) vs τ (spazio log-RV; la
#         trasformazione exp è monotona → identica in spazio RV)
#       · pinball loss per livello vs baseline HAR-quantile (punto HAR OLS su
#         train + quantili empirici dei residui di train — stesso information set)
#     GATE A2a (val): coverage q90∈[.85,.95] E q10∈[.05,.15]; q50∈[.45,.55];
#     pinball_NN(q90) ≤ pinball_HAR(q90); n ≥ 0.95·len(t_split).
#     ZERO retrain, read-only sui checkpoint: alimenta solo il design v2 post-gate.
# EN: A2a — log-RV QUANTILE CALIBRATION JUDGE (pre-registered in STATUS.md 2026-07-08).
#     The PASS vol model is already loss_type=quantile: this judge EXTRACTS the
#     never-used per-member quantiles (EnsembleModel collapses them to μ=q50),
#     fuses them cross-member (weighted Vincentization) and measures on val/test:
#       · empirical coverage per level: P(y ≤ q_τ) vs τ (log-RV space; exp is
#         monotone → identical in RV space)
#       · pinball loss per level vs a HAR-quantile baseline (train-fit HAR OLS
#         point + empirical train-residual quantiles — same information set)
#     A2a GATE (val): q90 coverage ∈[.85,.95] AND q10∈[.05,.15]; q50∈[.45,.55];
#     NN q90 pinball ≤ HAR q90 pinball; n ≥ 0.95·len(t_split).
#     ZERO retrain, checkpoint read-only: feeds the post-gate v2 design only.
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
from quantsys.model.vol_metrics import EPS  # noqa: E402
from quantsys.model import QUANTILES  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vols_quantile_judge")


# IT: pinball loss per un singolo livello τ, in spazio log-RV (media sui campioni).
# EN: pinball loss for a single level τ, in log-RV space (mean over samples).
def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Giudice calibrazione quantili vol-S (A2a) / "
                                             "VOL-S quantile calibration judge (A2a)")
    ap.add_argument("--arch", default="itransformer",
                    choices=["itransformer", "nhits", "tcnmamba", "lstm"],
                    help="architettura del modello vol (models/{arch}) / vol model arch")
    # IT: A2-CONFORME (pre-registrato STATUS.md 2026-07-10) — inerte di default:
    #     senza flag il giudice A2a resta bit-identico e il suo report non è toccato.
    # EN: A2-CONFORMAL (pre-registered STATUS.md 2026-07-10) — inert by default:
    #     without the flag the A2a judge stays bit-identical, its report untouched.
    ap.add_argument("--conformal", action="store_true",
                    help="ricalibrazione split-conformal (fit prefisso, giudizio suffisso; "
                         "report separato) / split-conformal recalibration (prefix fit, "
                         "suffix judgment; separate report)")
    # IT: guard scaler modello<->dataset (2026-08-01) — via di fuga ESPLICITA,
    #     flag e mai env: un run cross-vintage produce un numero non confrontabile.
    # EN: model<->dataset scaler guard (2026-08-01) — EXPLICIT escape, flag never
    #     env: a cross-vintage run produces a non-comparable number.
    ap.add_argument("--allow-scaler-mismatch", action="store_true",
                    help="procedi anche se lo scaler del modello != scaler del dataset: "
                         "il numero NON e' confrontabile / proceed even if the model "
                         "scaler != dataset scaler: the number is NOT comparable")
    args = ap.parse_args()
    model_dir = models_root() / args.arch
    log.info(f"dir modelli effettiva / effective model dir: {model_dir} (arch={args.arch})")

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config features.target_type deve essere log_rv / must be log_rv"

    h = int(cfg["features"].get("forecast_horizon", 30))
    interval = cfg["data"]["interval"]
    bars_day = 1440 // interval_minutes_from_cfg(cfg)
    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    assert split in ("val", "test")
    log.info(f"h={h} barre · interval={interval} · split={split}")

    # ── Ground truth + regressori HAR (STESSO data-path del giudice QLIKE) ──────
    # IT: definizioni identiche a dev_vols_qlike.py — un solo modo di calcolare y.
    # EN: identical definitions to dev_vols_qlike.py — one single way to compute y.
    raw = pd.read_parquet("data/raw_candles.parquet").sort_values("open_time").reset_index(drop=True)
    lr2 = np.log(raw["close"] / raw["close"].shift(1)) ** 2
    rv_h = lr2.rolling(h).sum()
    rv_w = lr2.rolling(7 * bars_day).sum() / 7
    rv_m = lr2.rolling(30 * bars_day).sum() / 30
    rv_fwd = rv_h.shift(-h)
    har = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":  np.log(rv_fwd + EPS),
        "xh": np.log(rv_h + EPS),
        "xw": np.log(rv_w * (h / bars_day) + EPS),
        "xm": np.log(rv_m * (h / bars_day) + EPS),
    }).dropna().set_index("open_time")

    d = np.load("data/lstm_dataset.npz", allow_pickle=True)
    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)
    t_eval = pd.to_datetime(d[f"t_{split}"]).tz_localize(None)
    har.index = pd.to_datetime(har.index).tz_localize(None)
    tr = har.loc[har.index.intersection(t_train)]
    ev = har.loc[har.index.intersection(t_eval)]
    log.info(f"righe: train {len(tr)}/{len(t_train)}  {split} {len(ev)}/{len(t_eval)}")
    n_ok = len(ev) >= 0.95 * len(t_eval)

    # ── Baseline HAR-quantile: punto OLS (fit train) + quantili empirici residui ──
    Xtr = np.column_stack([np.ones(len(tr)), tr[["xh", "xw", "xm"]].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    resid_tr = tr["y"].values - Xtr @ beta
    Xev = np.column_stack([np.ones(len(ev)), ev[["xh", "xw", "xm"]].values])
    har_point = Xev @ beta
    # IT: q_τ^HAR = punto + quantile empirico dei residui di train (QR baseline standard).
    # EN: HAR q_τ = point + empirical train-residual quantile (standard QR baseline).
    har_q = {tau: har_point + float(np.quantile(resid_tr, tau)) for tau in QUANTILES}

    # ── NN: quantili per-membro → Vincentization pesata → inversione z→raw ──────
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.utils import PipelineState
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load(str(model_dir), device)
    ps = PipelineState.load(str(model_dir / "pipeline_state.pkl"))
    idx = ps.scale_cols.index("target_ret")
    c, s = float(ps.scaler.center_[idx]), float(ps.scaler.scale_[idx])
    assert c < -3, "center ≈ 0 → PipelineState non log-RV (stale?)"

    # IT: GUARD SCALER MODELLO<->DATASET — l'assert sopra cattura solo uno stato
    #     grossolanamente sbagliato (center ~ 0), NON uno plausibile ma di un altro
    #     vintage del dataset. Quel caso e' gia' successo e ha prodotto un numero
    #     credibile e sbagliato: vedi TEORIA.md 12.2, provenienza del numeratore.
    # EN: MODEL<->DATASET SCALER GUARD — the assert above only catches a grossly
    #     wrong state (center ~ 0), NOT a plausible one from another dataset
    #     vintage. That case already happened and produced a credible wrong number.
    from quantsys.utils import assert_model_dataset_scaler, dataset_npz_path
    assert_model_dataset_scaler(ps, model_dir=model_dir, arch=args.arch,
                                npz=dataset_npz_path(),
                                allow_mismatch=args.allow_scaler_mismatch, logger=log)


    # IT: tutti i membri DEVONO essere quantile (il giudice estrae la testa quantile).
    # EN: every member MUST be quantile-type (this judge extracts the quantile head).
    members = model._models
    for m in members:
        assert getattr(m, "loss_type", "") == "quantile", \
            f"membro non-quantile ({getattr(m, 'loss_type', '?')}) — A2a non applicabile"
    w = np.asarray(model.weights, dtype=np.float64)  # IT: pesi correnti (uniformi) | EN: current weights

    X = torch.tensor(d[f"X_{split}"], dtype=torch.float32)
    Xm = torch.tensor(d[f"X_macro_{split}"], dtype=torch.float32) if f"X_macro_{split}" in d.files else None
    qp_members = []  # IT: lista (n_obs, Q) per membro | EN: per-member (n_obs, Q) list
    with torch.no_grad():
        for mi, m in enumerate(members):
            outs = []
            for i in range(0, len(X), 256):
                xb = X[i:i + 256].to(device)
                xmb = Xm[i:i + 256].to(device) if Xm is not None else None
                out = m(xb, xmb) if xmb is not None else m(xb)
                qp, _ = out[0].sort(dim=-1)   # IT: sort = non-crossing (pattern ensemble) | EN: sort = non-crossing
                outs.append(qp.detach().cpu().numpy())
            qp_members.append(np.concatenate(outs, axis=0))
            log.info(f"membro {mi}: quantili estratti {qp_members[-1].shape}")
    qp_all = np.stack(qp_members, axis=0)                       # (N, n_obs, Q)
    # IT: Vincentization = media pesata dei quantili cross-membro (livello per livello).
    # EN: Vincentization = weighted cross-member quantile average (level by level).
    qp_ens_z = np.tensordot(w, qp_all, axes=(0, 0))             # (n_obs, Q)
    # IT: inversione monotona affine z→raw log-RV: i quantili si trasformano direttamente.
    # EN: monotone affine z→raw log-RV inversion: quantiles transform directly.
    qp_ens = qp_ens_z * s + c

    pos = {ts: k for k, ts in enumerate(t_eval)}
    sel = np.array([pos[ts] for ts in ev.index])
    qp_ens = qp_ens[sel]
    y = ev["y"].values

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)

    if args.conformal:
        # ── A2-CONFORME: split temporale prefisso/suffisso di ev (già in ordine ────
        #    cronologico: har.index è monotono e sel lo preserva) ──────────────────
        # IT: shift additivo per livello δ_τ = quantile_τ(y − q_τ) stimato sul
        #     prefisso (calibrazione), applicato IDENTICAMENTE a NN e HAR-quantile
        #     (stesso information set → confronto fair); giudizio SOLO sul suffisso.
        #     UNICA formula primaria pre-registrata: niente sweep di varianti.
        # EN: per-level additive shift δ_τ = quantile_τ(y − q_τ) fit on the prefix
        #     (calibration), applied IDENTICALLY to NN and HAR-quantile (same
        #     information set → fair comparison); judged on the suffix ONLY.
        #     Single pre-registered primary formula: no variant sweeps.
        assert ev.index.is_monotonic_increasing, "ev non cronologico — split conforme invalido"
        n_cal = len(y) // 2
        y_cal, y_jud = y[:n_cal], y[n_cal:]
        conf = {"n_calib": int(n_cal), "n_judge": int(len(y) - n_cal),
                "delta_nn": {}, "delta_har": {},
                "coverage_raw": {}, "coverage_conf": {}, "coverage_har_conf": {},
                "pinball_nn_conf": {}, "pinball_har_conf": {}}
        for j, tau in enumerate(QUANTILES):
            t = str(tau)
            d_nn = float(np.quantile(y_cal - qp_ens[:n_cal, j], tau))
            d_har = float(np.quantile(y_cal - har_q[tau][:n_cal], tau))
            conf["delta_nn"][t], conf["delta_har"][t] = d_nn, d_har
            q_nn = qp_ens[n_cal:, j] + d_nn
            q_har = har_q[tau][n_cal:] + d_har
            conf["coverage_raw"][t] = float(np.mean(y_jud <= qp_ens[n_cal:, j]))
            conf["coverage_conf"][t] = float(np.mean(y_jud <= q_nn))
            conf["coverage_har_conf"][t] = float(np.mean(y_jud <= q_har))
            conf["pinball_nn_conf"][t] = pinball(y_jud, q_nn, tau)
            conf["pinball_har_conf"][t] = pinball(y_jud, q_har, tau)
        # IT: diagnostica NON decisionale: larghezza q90−q10 pre/post sul suffisso.
        # EN: NON-decisional diagnostic: q90−q10 width pre/post on the suffix.
        conf["width_q10_q90"] = {
            "raw": float(np.mean(qp_ens[n_cal:, -1] - qp_ens[n_cal:, 0])),
            "conf": float(np.mean(qp_ens[n_cal:, -1] + conf["delta_nn"]["0.9"]
                                  - qp_ens[n_cal:, 0] - conf["delta_nn"]["0.1"]))}

        cc = conf["coverage_conf"]
        gate = {
            "split": split, "n_judge": conf["n_judge"],
            "cov_q90_ok": bool(0.85 <= cc["0.9"] <= 0.95),
            "cov_q10_ok": bool(0.05 <= cc["0.1"] <= 0.15),
            "cov_q50_ok": bool(0.45 <= cc["0.5"] <= 0.55),
            "pinball_q90_beats_har": bool(conf["pinball_nn_conf"]["0.9"]
                                          <= conf["pinball_har_conf"]["0.9"]),
            "n_ok": bool(conf["n_judge"] >= 3000),
        }
        gate["verdict"] = "PASS" if all(gate[k] for k in
                                        ("cov_q90_ok", "cov_q10_ok", "cov_q50_ok",
                                         "pinball_q90_beats_har", "n_ok")) else "FAIL"

        out_path = out_dir / f"quantile_conformal_report_{interval}_{split}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"quantiles": QUANTILES, "conformal": conf, "gate": gate,
                       "scaler": {"center": c, "scale": s}}, f, indent=2)

        print(f"\n══════ A2-CONFORME [{interval}·{split}] — calib {conf['n_calib']} / "
              f"giudizio {conf['n_judge']} ══════")
        print(f"  {'τ':>5s}  {'cov raw':>8s}  {'cov conf':>9s}  {'cov HARc':>9s}  "
              f"{'δ_NN':>8s}  {'pinb NNc':>9s}  {'pinb HARc':>10s}")
        for tau in QUANTILES:
            t = str(tau)
            print(f"  {tau:>5.2f}  {conf['coverage_raw'][t]:>8.4f}  {cc[t]:>9.4f}  "
                  f"{conf['coverage_har_conf'][t]:>9.4f}  {conf['delta_nn'][t]:>+8.4f}  "
                  f"{conf['pinball_nn_conf'][t]:>9.5f}  {conf['pinball_har_conf'][t]:>10.5f}")
        print(f"  width q10-q90 raw {conf['width_q10_q90']['raw']:.4f} → "
              f"conf {conf['width_q10_q90']['conf']:.4f} (diagnostica)")
        print(f"  VERDETTO [{split}]: {gate['verdict']}   → {out_path}")
        return

    # ── Metriche: coverage + pinball (NN vs HAR-quantile) ───────────────────────
    res = {"coverage": {}, "pinball_nn": {}, "pinball_har": {}}
    for j, tau in enumerate(QUANTILES):
        res["coverage"][str(tau)] = float(np.mean(y <= qp_ens[:, j]))
        res["pinball_nn"][str(tau)] = pinball(y, qp_ens[:, j], tau)
        res["pinball_har"][str(tau)] = pinball(y, har_q[tau], tau)

    cov = res["coverage"]
    gate = {
        "split": split,
        "n_obs": int(len(ev)),
        "cov_q90_ok": bool(0.85 <= cov["0.9"] <= 0.95),
        "cov_q10_ok": bool(0.05 <= cov["0.1"] <= 0.15),
        "cov_q50_ok": bool(0.45 <= cov["0.5"] <= 0.55),
        "pinball_q90_beats_har": bool(res["pinball_nn"]["0.9"] <= res["pinball_har"]["0.9"]),
        "n_ok": bool(n_ok),
    }
    gate["verdict"] = "PASS" if all(gate[k] for k in
                                    ("cov_q90_ok", "cov_q10_ok", "cov_q50_ok",
                                     "pinball_q90_beats_har", "n_ok")) else "FAIL"

    out_path = out_dir / f"quantile_report_{interval}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"quantiles": QUANTILES, "metrics": res, "gate": gate,
                   "scaler": {"center": c, "scale": s}}, f, indent=2)

    print(f"\n══════ A2a QUANTILE CALIBRATION [{interval}·{split}] ══════")
    print(f"  {'τ':>5s}  {'coverage':>9s}  {'pinball NN':>11s}  {'pinball HAR':>12s}")
    for tau in QUANTILES:
        t = str(tau)
        print(f"  {tau:>5.2f}  {cov[t]:>9.4f}  {res['pinball_nn'][t]:>11.5f}  {res['pinball_har'][t]:>12.5f}")
    print(f"  VERDETTO [{split}]: {gate['verdict']}   → {out_path}")


if __name__ == "__main__":
    main()
