# IT: A10 — SONDA DELL'ENTROPIA DELL'ATTENTION (manipulation check ④ della
#     pre-registrazione STATUS.md 2026-07-28).
#     Misura H_norm = entropia media delle mappe di attention, normalizzata per log N
#     (N = token della sequenza), su un campione FISSO dello split scelto, mediata su
#     righe, teste, layer, batch e membri dell'ensemble.
#     A cosa serve: la condizione ④ pretende che H_norm del candidato sia almeno il 5%
#     più bassa di quella della baseline. Se non lo è, il run ha misurato
#     l'implementazione e non l'ipotesi → esito "NESSUNA CONCLUSIONE", non FAIL. Serve
#     quindi su ENTRAMBI i bracci, baseline compresa, che è addestrata a λ=0 e non
#     registra nulla durante il training: qui la misura viene accesa a posteriori
#     (`set_attn_entropy`), senza toccare pesi né predizioni.
#     Sola lettura: nessun file di modello viene scritto.
# EN: A10 — ATTENTION ENTROPY PROBE (manipulation check ④ of the STATUS.md 2026-07-28
#     pre-registration).
#     Measures H_norm = mean attention-map entropy normalized by log N (N = sequence
#     tokens) over a FIXED sample of the chosen split, averaged over rows, heads,
#     layers, batches and ensemble members.
#     Why: condition ④ requires the candidate's H_norm to be at least 5% below the
#     baseline's. If it is not, the run measured the implementation rather than the
#     hypothesis → outcome "NO CONCLUSION", not FAIL. It is therefore needed on BOTH
#     arms, including the baseline, which is trained at λ=0 and records nothing during
#     training: here measurement is switched on after the fact (`set_attn_entropy`),
#     touching neither weights nor predictions.
#     Read-only: no model file is written.
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, models_root, dataset_npz_path  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.attn_entropy_probe")


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="A10 — sonda H_norm dell'attention su un checkpoint / "
                    "A10 — attention H_norm probe on a checkpoint")
    ap.add_argument("--arch", default="itransformer",
                    help="sottodir di models_root() da sondare / models_root() subdir to probe")
    # IT: campione FISSO e limitato: la misura è una media su mappe, non serve tutto lo
    #     split; il cap va tenuto IDENTICO fra i due bracci (default 2048).
    # EN: FIXED, capped sample: the measure averages over maps, the whole split is not
    #     needed; keep the cap IDENTICAL across the two arms (default 2048).
    ap.add_argument("--n-samples", type=int, default=2048,
                    help="finestre usate (stesse per entrambi i bracci) / windows used (same for both arms)")
    args = ap.parse_args()

    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    assert split in ("val", "test")
    model_dir = models_root() / args.arch
    log.info(f"dir modelli / model dir: {model_dir} · split={split} · n={args.n_samples}")

    cfg = load_config("config/default.yaml")
    d = np.load(str(dataset_npz_path()), allow_pickle=True)
    n = min(int(args.n_samples), len(d[f"X_{split}"]))
    X = torch.tensor(d[f"X_{split}"][:n], dtype=torch.float32)
    Xm = (torch.tensor(d[f"X_macro_{split}"][:n], dtype=torch.float32)
          if f"X_macro_{split}" in d.files else None)

    from quantsys.model.ensemble import EnsembleModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ens = EnsembleModel.load(str(model_dir), device)

    # IT: misura per-membro. `set_attn_entropy(True)` accende la materializzazione
    #     SOLO in questa sonda: `load_model` la lascia spenta, così il giudice QLIKE
    #     valuta i due bracci con la stessa implementazione (Flash).
    # EN: per-member measurement. `set_attn_entropy(True)` enables materialization ONLY
    #     inside this probe: `load_model` leaves it off so the QLIKE judge evaluates
    #     both arms with the same implementation (Flash).
    per_member = {}
    for name, m in zip(ens._arch_names, ens._models):
        if not hasattr(m, "set_attn_entropy"):
            log.warning(f"{name}: arch senza attention iTransformer, saltato / "
                        f"non-iTransformer attention, skipped")
            continue
        m.eval(); m.set_attn_entropy(True)
        vals = []
        with torch.no_grad():
            for i in range(0, n, 256):
                xb = X[i:i + 256].to(device)
                xmb = Xm[i:i + 256].to(device) if Xm is not None else None
                m(xb, xmb)
                pen = m.attn_entropy_penalty()
                if pen is not None:
                    vals.append(float(pen))
        m.set_attn_entropy(False)   # IT/EN: ripristino stato / restore state
        if vals:
            per_member[name] = float(np.mean(vals))
            log.info(f"{name}: H_norm={per_member[name]:.5f}")

    if not per_member:
        raise RuntimeError("nessun membro misurabile / no measurable member")

    h_mean = float(np.mean(list(per_member.values())))
    interval = cfg["data"]["interval"]
    _root = models_root()
    _sandbox = f"_{_root.name}" if _root.name != "models" else ""
    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"attn_entropy_{args.arch}_{interval}_{split}{_sandbox}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"arch": args.arch, "split": split, "n_samples": n,
                   "h_norm_mean": h_mean, "per_member": per_member}, f, indent=2)

    print(f"\n══════ A10 attention H_norm [{args.arch}·{interval}·{split}] ══════")
    for k, v in per_member.items():
        print(f"  {k:10s} H_norm={v:.5f}")
    # IT: ④ confronta questo numero fra i due bracci: candidato ≤ 0.95 · baseline.
    # EN: ④ compares this number across arms: candidate ≤ 0.95 · baseline.
    print(f"  MEDIA/MEAN H_norm = {h_mean:.5f}  (n={n})   → {out_path}")


if __name__ == "__main__":
    main()
