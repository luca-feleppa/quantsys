"""Export per-bar vol-S predictions for one model/split, for plotting only.

Read-only with respect to models and data: it reproduces, without modifying it,
the NN path of `scripts/vol/dev_vols_qlike.py` (same npz, same eval index, same
log-RV inversion `mu_z * IQR + center`, same scaler identity guard) and writes the
per-bar series that the judge keeps in memory and never persists.

Output: `results/vols/vols_predictions_{interval}_{split}_{arch}.parquet` with
  open_time, rv_true, rv_nn_q50, rv_nn_q10, rv_nn_q90
where RV is the forward realized variance over `forecast_horizon` bars.

Quantile band: per-member q10/q90 in z-space, averaged across members with the
ensemble weights (quantile averaging), then inverted like the median. Quantiles
commute with the monotone map exp(q * IQR + center), so the band is exact per
member; the cross-member average is a combination rule, not the ensemble's
total-variance sigma.

Consistency check (fail-fast): the QLIKE of `rv_nn_q50` recomputed on the
exported rows must match `metrics.nn.qlike` of the judge's report for the same
arch/split, otherwise nothing is written.

Usage (from the project root; GPU optional):
    python scripts/vol/export_vols_predictions.py --arch canonical_1h_vols --split test
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import (setup_logging, load_config, interval_minutes_from_cfg,  # noqa: E402
                            models_root, dataset_npz_path)
from quantsys.model.vol_metrics import qlike, EPS  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.export_vols_predictions")

# Positions in QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9] (same indexing as EnsembleModel).
Q10, Q50, Q90 = 0, 2, 4
# Relative tolerance of the QLIKE cross-check against the judge's report.
QLIKE_RTOL = 1e-6


def main():
    # Windows console: force UTF-8 so unicode in log lines cannot crash on cp1252.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Export per-bar vol-S predictions (plotting only)")
    ap.add_argument("--arch", default="canonical_1h_vols",
                    help="model directory under the models root (models/{arch})")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    cfg = load_config()
    if cfg["features"].get("target_type") != "log_rv":
        raise RuntimeError("features.target_type must be log_rv")
    h = int(cfg["features"]["forecast_horizon"])
    interval = cfg["data"]["interval"]
    bars_day = 1440 // interval_minutes_from_cfg(cfg)
    model_dir = models_root() / args.arch

    # The judge's report is the source of truth for the cross-check; without it
    # the exported series cannot be tied to a published number.
    from importlib import util as _ilu  # noqa: PLC0415
    _spec = _ilu.spec_from_file_location("dev_vols_qlike",
                                         Path(__file__).resolve().parent / "dev_vols_qlike.py")
    _judge = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_judge)
    report_path = Path("results/vols") / _judge.report_filename(interval, args.split, args.arch,
                                                                models_root().name)
    if not report_path.exists():
        raise FileNotFoundError(f"judge report missing: {report_path} - run dev_vols_qlike.py first")
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # Eval index built exactly as in the judge: trailing-h RV frame, dropna, ∩ split timestamps.
    raw = pd.read_parquet("data/raw_candles.parquet").sort_values("open_time").reset_index(drop=True)
    lr2 = np.log(raw["close"] / raw["close"].shift(1)) ** 2
    rv_h = lr2.rolling(h).sum()
    frame = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":  np.log(rv_h.shift(-h) + EPS),
        "xh": np.log(rv_h + EPS),
        "xw": np.log(lr2.rolling(7 * bars_day).sum() / 7 * (h / bars_day) + EPS),
        "xm": np.log(lr2.rolling(30 * bars_day).sum() / 30 * (h / bars_day) + EPS),
    }).dropna().set_index("open_time")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)

    d = np.load(str(dataset_npz_path()), allow_pickle=True)
    t_eval = pd.to_datetime(d[f"t_{args.split}"]).tz_localize(None)
    ev = frame.loc[frame.index.intersection(t_eval)]

    from quantsys.model.ensemble import EnsembleModel  # noqa: PLC0415
    from quantsys.utils import PipelineState, assert_model_dataset_scaler  # noqa: PLC0415
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            if (json.load(f).get("head_type", "single") or "single") != "single":
                raise RuntimeError("only single-head models are supported")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load(str(model_dir), device)
    ps = PipelineState.load(str(model_dir / "pipeline_state.pkl"))
    idx = ps.scale_cols.index("target_ret")
    c, s = float(ps.scaler.center_[idx]), float(ps.scaler.scale_[idx])
    if c >= -3:
        raise RuntimeError("target center ~0: pipeline_state is not from the log-RV dataset")
    # Same fail-fast scaler/macro identity guard as the judge, no escape hatch.
    assert_model_dataset_scaler(ps, model_dir=model_dir, arch=args.arch, npz=dataset_npz_path(),
                                allow_mismatch=False, npz_arrays=d,
                                allow_macro_mismatch=False, logger=log)

    X = torch.tensor(d[f"X_{args.split}"], dtype=torch.float32)
    Xm = (torch.tensor(d[f"X_macro_{args.split}"], dtype=torch.float32)
          if f"X_macro_{args.split}" in d.files else None)

    # Per-member sorted quantiles (as EnsembleModel does), combined with its weights; AMP off.
    w = np.asarray(model.weights if hasattr(model, "weights") else model._weights, dtype=np.float64)
    qz = []
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
        for i in range(0, len(X), 256):
            xb = X[i:i + 256].to(device)
            xmb = Xm[i:i + 256].to(device) if Xm is not None else None
            members = []
            for m in model._models:
                if getattr(m, "loss_type", None) != "quantile":
                    raise RuntimeError("quantile band requires loss_type=quantile members")
                qp, _ = m(xb, xmb)[0].sort(dim=-1)
                members.append(qp.double().cpu().numpy())
            qz.append(np.tensordot(w, np.stack(members), axes=1))
    qz = np.concatenate(qz)

    pos = {ts: k for k, ts in enumerate(t_eval)}
    sel = np.array([pos[ts] for ts in ev.index])
    out = pd.DataFrame({
        "open_time": ev.index,
        "rv_true":   np.exp(ev["y"].values),
        "rv_nn_q50": np.exp(qz[sel, Q50] * s + c),
        "rv_nn_q10": np.exp(qz[sel, Q10] * s + c),
        "rv_nn_q90": np.exp(qz[sel, Q90] * s + c),
    })

    # Tie the export to the published number before writing anything.
    q_export = qlike(out["rv_true"].values, out["rv_nn_q50"].values)
    q_report = float(report["metrics"]["nn"]["qlike"])
    log.info(f"QLIKE export={q_export:.8f} report={q_report:.8f} n={len(out)}")
    if not np.isclose(q_export, q_report, rtol=QLIKE_RTOL, atol=0.0):
        raise RuntimeError(f"export does not reproduce the judge: {q_export:.8f} vs {q_report:.8f}")

    out_path = Path("results/vols") / f"vols_predictions_{interval}_{args.split}_{args.arch}.parquet"
    out.to_parquet(out_path, index=False)
    log.info(f"written {out_path} ({len(out)} rows, source report {report_path})")


if __name__ == "__main__":
    main()
