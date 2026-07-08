# IT: A5 — PESI MEMBRO PER-QLIKE (pre-registrato in STATUS.md 2026-07-08).
#     I 5 seed dell'ensemble vol di produzione hanno pesi uniformi (0.2); per la
#     linea vol il giudice canonico è QLIKE (robusto alla proxy RV, penalizza
#     asimmetricamente l'under-prediction = l'errore costoso per lo short-vol).
#     Protocollo anti val-selection: pesi FITTATI sulla PRIMA metà temporale
#     dello split (w_i ∝ 1/QLIKE_i, UNICA formula primaria), VALUTATI sulla
#     seconda metà. Diagnostiche loggate ma NON decisionali: softmax(−QLIKE/2),
#     inverse-QLIKE².
#     GATE A5 (2ª metà): QLIKE(pesato) ≤ 0.97·QLIKE(uniforme), n_eval ≥ 3000.
#     Read-only sui checkpoint: NESSUNA promozione prima del gate live n≥20.
# EN: A5 — QLIKE-BASED MEMBER WEIGHTS (pre-registered in STATUS.md 2026-07-08).
#     The production vol ensemble's 5 seeds use uniform weights (0.2); for the
#     vol line the canonical judge is QLIKE (RV-proxy-robust, asymmetric penalty
#     on under-prediction = the costly error for short-vol).
#     Anti val-selection protocol: weights FITTED on the FIRST temporal half of
#     the split (w_i ∝ 1/QLIKE_i, the ONLY primary formula), EVALUATED on the
#     second half. Logged-only diagnostics: softmax(−QLIKE/2), inverse-QLIKE².
#     A5 GATE (2nd half): weighted QLIKE ≤ 0.97·uniform QLIKE, n_eval ≥ 3000.
#     Checkpoint read-only: NO promotion before the live n≥20 gate closes.
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
from quantsys.model.vol_metrics import qlike, EPS  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vols_member_weights")


def main():
    # IT: boilerplate UTF-8 (checklist CLAUDE.md) | EN: UTF-8 boilerplate (CLAUDE.md checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Pesi membro per-QLIKE linea vol (A5) / "
                                             "QLIKE member weights, vol line (A5)")
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
    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    assert split in ("val", "test")
    _ = interval_minutes_from_cfg(cfg)  # IT: fail-fast su interval ignoto | EN: fail-fast on unknown interval
    log.info(f"h={h} barre · interval={interval} · split={split}")

    # ── Ground truth (STESSO data-path del giudice QLIKE) ───────────────────────
    raw = pd.read_parquet("data/raw_candles.parquet").sort_values("open_time").reset_index(drop=True)
    lr2 = np.log(raw["close"] / raw["close"].shift(1)) ** 2
    rv_fwd = lr2.rolling(h).sum().shift(-h)
    gt = pd.DataFrame({"open_time": raw["open_time"],
                       "y": np.log(rv_fwd + EPS)}).dropna().set_index("open_time")

    d = np.load("data/lstm_dataset.npz", allow_pickle=True)
    t_eval = pd.to_datetime(d[f"t_{split}"]).tz_localize(None)
    gt.index = pd.to_datetime(gt.index).tz_localize(None)
    ev = gt.loc[gt.index.intersection(t_eval)]
    log.info(f"righe {split}: {len(ev)}/{len(t_eval)}")

    # ── Forward per-membro: μ = q50 (o mu diretto per membri t-Student) ─────────
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.utils import PipelineState
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load(str(model_dir), device)
    ps = PipelineState.load(str(model_dir / "pipeline_state.pkl"))
    idx = ps.scale_cols.index("target_ret")
    c, s = float(ps.scaler.center_[idx]), float(ps.scaler.scale_[idx])
    assert c < -3, "center ≈ 0 → PipelineState non log-RV (stale?)"

    X = torch.tensor(d[f"X_{split}"], dtype=torch.float32)
    Xm = torch.tensor(d[f"X_macro_{split}"], dtype=torch.float32) if f"X_macro_{split}" in d.files else None
    mu_members = []
    with torch.no_grad():
        for mi, m in enumerate(model._models):
            lt = getattr(m, "loss_type", "t_student")
            outs = []
            for i in range(0, len(X), 256):
                xb = X[i:i + 256].to(device)
                xmb = Xm[i:i + 256].to(device) if Xm is not None else None
                out = m(xb, xmb) if xmb is not None else m(xb)
                if lt == "quantile":
                    qp, _ = out[0].sort(dim=-1)
                    mu = qp[:, 2]                 # IT: mediana q50 (pattern ensemble) | EN: q50 median
                else:
                    mu = out[0]
                outs.append(mu.detach().cpu().numpy().ravel())
            mu_members.append(np.concatenate(outs))
            log.info(f"membro {mi} ({lt}): μ estratto, n={len(mu_members[-1])}")
    mu_all = np.stack(mu_members, axis=0) * s + c            # (N, n_obs) log-RV raw

    pos = {ts: k for k, ts in enumerate(t_eval)}
    sel = np.array([pos[ts] for ts in ev.index])
    mu_all = mu_all[:, sel]
    y = ev["y"].values
    rv_true = np.exp(y)

    # ── Split temporale: fit pesi su 1ª metà, giudizio su 2ª metà ───────────────
    n = len(y)
    mid = n // 2
    fit_sl, eval_sl = slice(0, mid), slice(mid, n)
    log.info(f"fit: {mid} obs · eval: {n - mid} obs (split temporale, nessun overlap)")

    ql_fit = np.array([qlike(rv_true[fit_sl], np.exp(mu_all[i, fit_sl]))
                       for i in range(mu_all.shape[0])])
    # IT: formula primaria PRE-REGISTRATA: w ∝ 1/QLIKE (normalizzati).
    # EN: PRE-REGISTERED primary formula: w ∝ 1/QLIKE (normalized).
    w_inv = (1.0 / ql_fit); w_inv /= w_inv.sum()
    # IT: diagnostiche NON decisionali | EN: non-decisional diagnostics
    w_sm = np.exp(-ql_fit / 2.0); w_sm /= w_sm.sum()
    w_inv2 = (1.0 / ql_fit ** 2); w_inv2 /= w_inv2.sum()
    n_mem = len(ql_fit)
    w_unif = np.full(n_mem, 1.0 / n_mem)

    def ens_qlike(w: np.ndarray, sl: slice) -> float:
        # IT: μ_ens = media pesata dei μ per-membro (come il blend production), poi exp.
        # EN: ens μ = weighted per-member μ mean (as the production blend), then exp.
        mu_e = np.tensordot(w, mu_all[:, sl], axes=(0, 0))
        return qlike(rv_true[sl], np.exp(mu_e))

    res = {
        "qlike_fit_per_member": [float(q) for q in ql_fit],
        "weights_primary_inv": [float(x) for x in w_inv],
        "weights_diag_softmax": [float(x) for x in w_sm],
        "weights_diag_inv2": [float(x) for x in w_inv2],
        "eval_uniform": ens_qlike(w_unif, eval_sl),
        "eval_primary_inv": ens_qlike(w_inv, eval_sl),
        "eval_diag_softmax": ens_qlike(w_sm, eval_sl),
        "eval_diag_inv2": ens_qlike(w_inv2, eval_sl),
        "eval_best_single_fit": float(qlike(
            rv_true[eval_sl], np.exp(mu_all[int(np.argmin(ql_fit)), eval_sl]))),
    }
    gate = {
        "split": split, "n_fit": int(mid), "n_eval": int(n - mid),
        "ratio_primary_vs_uniform": res["eval_primary_inv"] / res["eval_uniform"],
        "beats_uniform_3pct": bool(res["eval_primary_inv"] <= 0.97 * res["eval_uniform"]),
        "n_eval_ok": bool((n - mid) >= 3000),
    }
    gate["verdict"] = "PASS" if (gate["beats_uniform_3pct"] and gate["n_eval_ok"]) else "FAIL"

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"member_weights_{interval}_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": res, "gate": gate}, f, indent=2)

    print(f"\n══════ A5 MEMBER WEIGHTS per-QLIKE [{interval}·{split}] ══════")
    print(f"  QLIKE fit per-membro : {[f'{q:.4f}' for q in ql_fit]}")
    print(f"  pesi primari (1/Q)   : {[f'{x:.3f}' for x in w_inv]}")
    print(f"  eval uniforme        : {res['eval_uniform']:.5f}")
    print(f"  eval primari         : {res['eval_primary_inv']:.5f}  "
          f"(ratio {gate['ratio_primary_vs_uniform']:.4f}, gate ≤ 0.97)")
    print(f"  eval best-single(fit): {res['eval_best_single_fit']:.5f}  [diagnostica]")
    print(f"  VERDETTO [{split}]: {gate['verdict']}   → {out_path}")


if __name__ == "__main__":
    main()
