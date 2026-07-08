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
    # IT: boilerplate UTF-8 (checklist CLAUDE.md) | EN: UTF-8 boilerplate (CLAUDE.md checklist)
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

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
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
