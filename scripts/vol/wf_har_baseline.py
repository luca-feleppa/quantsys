# IT: b1 — BASELINE HAR-RV PER-FOLD per il walk-forward vol (CPU-only, no GPU, no
#     retrain). Colma il gap di `02b_walkforward_validate.py`, che misura la QLIKE del
#     NN per fold ma NON la HAR. Ricostruisce gli STESSI fold (walk_forward_folds con
#     n_folds/embargo/val_frac di config), fitta l'HAR (OLS) sul train di ogni fold e
#     la valuta sull'held-out, poi confronta con i QLIKE NN già salvati in
#     results/{arch}/walkforward_metrics_log_rv.json. La HAR è arch-indipendente →
#     una sola passata serve da baseline per tutti e 3 gli archi.
#     Gate per-fold (stesso del giudice single-split): QLIKE_NN <= 0.95*QLIKE_HAR.
# EN: b1 — PER-FOLD HAR-RV BASELINE for the vol walk-forward (CPU-only, no GPU, no
#     retrain). Fills the gap of `02b_walkforward_validate.py`, which measures the
#     per-fold NN QLIKE but NOT HAR. Rebuilds the SAME folds (walk_forward_folds with
#     config n_folds/embargo/val_frac), OLS-fits HAR on each fold's train and evaluates
#     on the held-out, then compares to the NN QLIKE already saved in
#     results/{arch}/walkforward_metrics_log_rv.json. HAR is arch-independent → a single
#     pass is the baseline for all 3 archs. Per-fold gate: QLIKE_NN <= 0.95*QLIKE_HAR.
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, interval_minutes_from_cfg  # noqa: E402
from quantsys.features import walk_forward_folds                                  # noqa: E402
from quantsys.model.vol_metrics import build_har_frame, har_fold_qlike           # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.wf_har")


def main():
    # IT: boilerplate UTF-8 (checklist CLAUDE.md) | EN: UTF-8 boilerplate (CLAUDE.md checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Baseline HAR-RV per-fold per il WF vol / per-fold HAR-RV baseline for the vol WF")
    ap.add_argument("--archs", nargs="+", default=["itransformer", "nhits", "tcnmamba"],
                    help="archi di cui leggere la QLIKE NN salvata / archs whose saved NN QLIKE to read")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config features.target_type deve essere log_rv"
    h = int(cfg["features"].get("forecast_horizon", 30))
    interval = cfg["data"]["interval"]
    bars_day = 1440 // interval_minutes_from_cfg(cfg)

    # IT: ricostruisce gli STESSI fold del 02b — solo timestamp (X/y placeholder per la
    #     lunghezza, niente caricamento del tensore da 3.2 GB).
    # EN: rebuild the SAME folds as 02b — timestamps only (placeholder X/y for length,
    #     no 3.2 GB tensor load).
    d = np.load("data/lstm_dataset.npz", allow_pickle=True)
    t = np.concatenate([d["t_train"], d["t_val"], d["t_test"]])
    n = len(t)
    X_dummy = np.zeros((n, 1), dtype=np.float32)
    y_dummy = np.zeros(n, dtype=np.float32)

    val_cfg = cfg.get("validation", {})
    n_folds = val_cfg.get("n_folds", 6)
    embargo = val_cfg.get("embargo_steps", 168)
    folds = walk_forward_folds(X_dummy, y_dummy, t,
                               n_folds=n_folds, embargo_steps=embargo,
                               val_frac=cfg["training"]["val_fraction"])

    # IT: HAR frame una sola volta dai raw candles | EN: HAR frame once from raw candles
    raw = pd.read_parquet("data/raw_candles.parquet")
    har = build_har_frame(raw, h=h, bars_day=bars_day)
    log.info(f"HAR frame: {len(har)} righe · h={h} · bars/day={bars_day} · interval={interval}")

    # ── HAR per-fold (arch-indipendente) ───────────────────────────────────────
    har_by_fold = {}
    for fold in folds:
        k = fold["fold"]
        m = har_fold_qlike(har, fold["t_train"], fold["t_val"])
        har_by_fold[k] = m
        log.info(f"  Fold {k}: QLIKE_HAR={m['qlike_har']:.4f}  QLIKE_naive={m['qlike_naive']:.4f}  "
                 f"(n_har={m['n_har']}/{m['n_eval']})")

    har_vals = [v["qlike_har"] for v in har_by_fold.values() if np.isfinite(v["qlike_har"])]
    naive_vals = [v["qlike_naive"] for v in har_by_fold.values() if np.isfinite(v["qlike_naive"])]
    har_mean = float(np.mean(har_vals)) if har_vals else float("nan")
    naive_mean = float(np.mean(naive_vals)) if naive_vals else float("nan")

    # ── Confronto con la QLIKE NN salvata, per arch ────────────────────────────
    report = {"interval": interval, "h": h, "n_folds_eff": len(folds),
              "har_by_fold": har_by_fold, "har_mean": har_mean, "naive_mean": naive_mean,
              "archs": {}}

    print(f"\n══════ b1 · WF HAR BASELINE [{interval}] ══════")
    print(f"  HAR  QLIKE medio cross-fold = {har_mean:.4f}   |   naive = {naive_mean:.4f}")
    print(f"  (gate per-fold: QLIKE_NN <= 0.95 * QLIKE_HAR)\n")

    for a in args.archs:
        nn_path = Path("results") / a / "walkforward_metrics_log_rv.json"
        if not nn_path.exists():
            log.warning(f"{a}: {nn_path} mancante — salto")
            continue
        nn = json.load(open(nn_path, encoding="utf-8"))
        nn_by_fold = {r["fold"]: r["qlike"] for r in nn["fold_results"]}

        rows, ratios, beats = [], [], 0
        for k in sorted(set(nn_by_fold) & set(har_by_fold)):
            q_nn = nn_by_fold[k]
            q_har = har_by_fold[k]["qlike_har"]
            ratio = q_nn / q_har if (q_har and np.isfinite(q_har)) else float("nan")
            beat = bool(np.isfinite(ratio) and q_nn <= 0.95 * q_har)
            beats += int(beat)
            ratios.append(ratio)
            rows.append({"fold": k, "qlike_nn": q_nn, "qlike_har": q_har,
                         "ratio_nn_har": ratio, "beats_har_5pct": beat})

        nn_mean = float(np.mean([r["qlike_nn"] for r in rows])) if rows else float("nan")
        ratio_mean = float(np.nanmean(ratios)) if ratios else float("nan")
        verdict = "PASS" if (np.isfinite(ratio_mean) and ratio_mean <= 0.95 and beats >= len(rows) - 1) else "FAIL"
        report["archs"][a] = {"folds": rows, "nn_mean": nn_mean, "ratio_mean": ratio_mean,
                              "beats_har_folds": beats, "n_folds": len(rows), "verdict": verdict}

        print(f"  ── {a} ──  NN medio={nn_mean:.4f}  vs HAR={har_mean:.4f}  "
              f"ratio medio={ratio_mean:.3f}  batte HAR in {beats}/{len(rows)} fold  → {verdict}")
        for r in rows:
            flag = "✓" if r["beats_har_5pct"] else "·"
            print(f"      fold {r['fold']}: NN={r['qlike_nn']:.4f}  HAR={r['qlike_har']:.4f}  "
                  f"ratio={r['ratio_nn_har']:.3f} {flag}")

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"wf_har_baseline_{interval}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report → {out_path}\n")


if __name__ == "__main__":
    main()
