# IT: STEP 0 KILL-CHECK (pre-registrato in STATUS.md 2026-06-22) — correlazione
#     degli ERRORI cross-architettura sul target vol (log_rv), split VAL.
#     Razionale: se i 3 archi sbagliano in modo quasi identico (corr ≥ kill-thresh,
#     come ≈0.995 sul direzionale) la riduzione di varianza di ensembling/distill è
#     ≈0 → KILL prima di spendere il walk-forward k-fold completo (il caro). PROCEDI
#     solo se almeno una coppia ha corr ≤ proceed-thresh (diversità sfruttabile).
#     SOLO lettura modelli + forward su val: nessun training, nessuna scrittura su
#     models/. L'errore è in spazio z (= s·errore_raw): la correlazione è invariante
#     all'inversione affine, quindi center/scale non servono.
# EN: STEP 0 KILL-CHECK (pre-registered in STATUS.md 2026-06-22) — cross-architecture
#     ERROR correlation on the vol target (log_rv), VAL split.
#     Rationale: if the 3 archs err near-identically (corr ≥ kill-thresh, as ≈0.995
#     on the directional target) the ensembling/distill variance reduction is ≈0 →
#     KILL before spending the full k-fold walk-forward (the expensive part). PROCEED
#     only if at least one pair has corr ≤ proceed-thresh (exploitable diversity).
#     READ-ONLY models + forward on val: no training, no writes to models/. The error
#     is in z-space (= s·raw_error): correlation is invariant to the affine inversion,
#     so center/scale are not needed.
import argparse
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, models_root  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.step0_xarch")


# IT: forward dell'ensemble di una singola arch → μ in spazio z (mediana q2 se quantile).
# EN: single-arch ensemble forward → μ in z-space (q2 median if quantile).
def _forward_mu_z(arch: str, X: torch.Tensor, Xm, device) -> np.ndarray:
    from quantsys.model.ensemble import EnsembleModel
    mdir = models_root() / arch
    model = EnsembleModel.load(str(mdir), device)
    mus = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i + 256].to(device)
            xmb = Xm[i:i + 256].to(device) if Xm is not None else None
            mu, _, _ = model(xb, xmb)
            mus.append(mu.detach().cpu().numpy().ravel())
    return np.concatenate(mus)


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="STEP 0 kill-check: correlazione errori cross-arch (vol) / "
                    "STEP 0 kill-check: cross-arch error correlation (vol)")
    ap.add_argument("--archs", nargs="+", default=["itransformer", "nhits", "tcnmamba"],
                    help="archi da confrontare (cartelle in models_root) / archs to compare")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="split su cui valutare (val-first) / split to evaluate (val-first)")
    ap.add_argument("--kill-thresh", type=float, default=0.99,
                    help="corr media ≥ soglia → KILL / mean corr ≥ thresh → KILL")
    ap.add_argument("--proceed-thresh", type=float, default=0.97,
                    help="almeno una coppia ≤ soglia → PROCEED / at least one pair ≤ thresh → PROCEED")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config features.target_type deve essere log_rv per lo STEP 0 vol"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"models_root={models_root()}  archs={args.archs}  split={args.split}  device={device}")

    d = np.load("data/lstm_dataset.npz", allow_pickle=True)
    X = torch.tensor(d[f"X_{args.split}"], dtype=torch.float32)
    Xm = (torch.tensor(d[f"X_macro_{args.split}"], dtype=torch.float32)
          if f"X_macro_{args.split}" in d.files else None)
    y = d[f"y_{args.split}"].ravel().astype(np.float64)

    # IT: errore per-campione di ogni arch (spazio z) | EN: per-sample error of each arch (z-space)
    errs = {}
    for a in args.archs:
        mu = _forward_mu_z(a, X, Xm, device).astype(np.float64)
        if len(mu) != len(y):
            raise RuntimeError(f"{a}: len(mu)={len(mu)} ≠ len(y)={len(y)} — modello/npz disallineati")
        errs[a] = mu - y
        log.info(f"  {a:12s}  err mean={errs[a].mean():+.4f}  std={errs[a].std():.4f}")

    # IT: Pearson degli errori a coppie | EN: pairwise Pearson of errors
    pairs = {}
    for a, b in combinations(args.archs, 2):
        r = float(np.corrcoef(errs[a], errs[b])[0, 1])
        pairs[f"{a}|{b}"] = r

    vals = list(pairs.values())
    mean_corr = float(np.mean(vals))
    min_corr  = float(np.min(vals))
    max_corr  = float(np.max(vals))

    if mean_corr >= args.kill_thresh:
        verdict = "KILL"
    elif min_corr <= args.proceed_thresh:
        verdict = "PROCEED"
    else:
        verdict = "BORDERLINE"

    report = {
        "split": args.split,
        "archs": args.archs,
        "pairwise_corr": pairs,
        "mean_corr": mean_corr,
        "min_corr": min_corr,
        "max_corr": max_corr,
        "kill_thresh": args.kill_thresh,
        "proceed_thresh": args.proceed_thresh,
        "verdict": verdict,
        "n_obs": int(len(y)),
    }

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step0_xarch_corr_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n══════ STEP 0 · CROSS-ARCH ERROR CORR [{args.split}] ══════")
    for k, v in pairs.items():
        print(f"  {k:<28} ρ_err = {v:+.4f}")
    print(f"  mean={mean_corr:+.4f}  min={min_corr:+.4f}  max={max_corr:+.4f}")
    print(f"  soglie: KILL≥{args.kill_thresh}  PROCEED≤{args.proceed_thresh}")
    print(f"  VERDETTO: {verdict}   → {out_path}")
    if verdict == "KILL":
        print("  ⇒ ensembling/distill inutile (varianza ≈ irriducibile): NON girare il k-fold.")
    elif verdict == "PROCEED":
        print("  ⇒ diversità sufficiente: ha senso procedere al walk-forward k-fold.")
    else:
        print("  ⇒ zona grigia: decisione manuale (vicino alle soglie).")


if __name__ == "__main__":
    main()
