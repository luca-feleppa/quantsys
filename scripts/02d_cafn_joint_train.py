"""
02d — Ottimizzazione CONGIUNTA end-to-end: CAFN + iTransformer + TCN-Mamba + N-HiTS.

IT: Loop di training end-to-end. La CAFN estrae un latente causale dal tensore
    feature di mercato e i tre modelli a valle si allenano IN CONTEMPORANEA su quel
    segnale. Loss congiunta = Σ_arch loss_predittiva(pred_arch, y) + λ·penalità_causale
    della CAFN. Un solo backward propaga i gradienti attraverso i tre modelli E la
    CAFN: i modelli massimizzano l'accuratezza, la CAFN stabilizza le relazioni
    causali (prossimità + stabilità dell'attenzione mascherata causalmente).

    ⚠ PROBE PRE-REGISTRATO, INERTE DI DEFAULT (TEORIA.md §12.1 (protocollo sperimentale)):
      • Output ISOLATO in `models/cafn/` e `results/cafn/` — NON tocca i modelli
        production `models/{arch}` né la parity live (BLOCKER #1).
      • La CAFN si addestra sul tensore CANONICO 104-feature (storia 2019→oggi).
        I dati Deribit grezzi (greche/book/IV) sono forward-collected (giorni di
        storia, storico short-tenor non gratis) → entrano SOLO come canale `extra`
        OPZIONALE futuro (flag `--deribit-extra`, inerte finché non c'è storia),
        MAI come input storico di training (sarebbe lookahead / dataset inesistente).
      • PRIOR onesto: è una variante di CLASSE-MODELLO; il progetto ha ripetutamente
        mostrato che ciò NON sposta il soffitto direzionale OOS (anti-corr val→test,
        cross-arch err≈0.995, distill 06-06 OOS≡baseline). GATE pre-registrato sotto.
      • Memoria: training simultaneo dei 3 modelli su 8GB → rischio OOM (il repo
        impone training 3-arch SEQUENZIALE). Default piccoli + `--smoke` CPU.

EN: End-to-end training loop. The CAFN extracts a causal latent from the market
    feature tensor and the three downstream models train SIMULTANEOUSLY on that
    signal. Joint loss = Σ_arch predictive_loss(pred_arch, y) + λ·CAFN causal penalty.
    A single backward propagates gradients through the three models AND the CAFN:
    the models maximize accuracy, the CAFN stabilizes causal relationships
    (proximity + stability of the causally-masked attention).

    ⚠ PRE-REGISTERED PROBE, INERT BY DEFAULT (TEORIA.md §12.1 (experimental protocol)):
      • Output ISOLATED in `models/cafn/` and `results/cafn/` — does NOT touch the
        production models `models/{arch}` nor the live parity (BLOCKER #1).
      • The CAFN trains on the CANONICAL 104-feature tensor (history 2019→today).
        Raw Deribit data (greeks/book/IV) is forward-collected (days of history,
        short-tenor history not free) → it enters ONLY as an OPTIONAL future `extra`
        channel (`--deribit-extra` flag, inert until history exists), NEVER as a
        historical training input (that would be lookahead / a non-existent dataset).
      • Honest PRIOR: this is a MODEL-CLASS variation; the project has repeatedly
        shown this does NOT move the OOS directional ceiling (val→test anti-corr,
        cross-arch err≈0.995, 06-06 distill OOS≡baseline). Pre-registered GATE below.
      • Memory: simultaneous 3-model training on 8GB → OOM risk (repo mandates
        SEQUENTIAL 3-arch training). Small defaults + `--smoke` CPU.

GATE PRE-REGISTRATO / PRE-REGISTERED GATE (valutazione val-first, su `X_val`):
    PASS sse il setup CAFN-congiunto batte il baseline NO-CAFN (stessi modelli,
    stesse seed/epoche, `latent=None`) di ≥3% in MSE-mu su val PER ALMENO 2 dei 3
    modelli. FAIL → CAFN come coordinatore è KILL (documentare in STATUS, flag
    inerte). NESSUNA iterazione a risultato visto; il test split si tocca solo a
    gate val superato.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantsys.utils import setup_logging, load_config           # noqa: E402
from quantsys.model import (                                     # noqa: E402
    CausalAttentionFlowNetwork, QuantiTransformer, QuantNHiTS, QuantTCNMamba,
)

setup_logging()
log = logging.getLogger("quantsys.script.cafn_joint")

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lstm_dataset.npz"
OUT_DIR = ROOT / "models" / "cafn"
RES_DIR = ROOT / "results" / "cafn"


# IT: estrae μ scalare dall'output (gestisce t_student tuple e quantile (B,Q)).
# EN: extract scalar μ from output (handles t_student tuple and quantile (B,Q)).
def _mu_of(out) -> torch.Tensor:
    head = out[0]
    if head.ndim == 2:                       # quantile_preds (B, Q) → mediana / median
        return head[:, head.shape[1] // 2]
    return head                              # (B,) mu


# IT: costruisce i 3 modelli a valle alla larghezza AUMENTATA n_feat+d_latent.
# EN: builds the 3 downstream models at the AUGMENTED width n_feat+d_latent.
def build_downstream(archs, n_in, T, mcfg, device):
    models = {}
    lt = mcfg.get("loss_type", "t_student")
    ne = int(mcfg.get("n_output_experts", 1))
    dm = int(mcfg.get("cafn_downstream_d_model", 64))
    for a in archs:
        if a == "itransformer":
            m = QuantiTransformer(n_features=n_in, T=T, n_dynamic=n_in, n_macro=0,
                                  d_model=dm, n_heads=4, n_layers=2,
                                  loss_type=lt, n_output_experts=ne)
        elif a == "nhits":
            m = QuantNHiTS(n_features=n_in, T=T, n_dynamic_features=n_in, n_macro=0,
                           d_model=dm, hidden=2 * dm, n_stacks=2,
                           loss_type=lt, n_output_experts=ne)
        elif a == "tcnmamba":
            m = QuantTCNMamba(n_features=n_in, d_model=dm, tcn_layers=3,
                              mamba_layers=2, n_dynamic_features=n_in,
                              loss_type=lt, n_output_experts=ne)
        else:
            raise ValueError(f"arch sconosciuta / unknown arch: {a}")
        models[a] = m.to(device)
    return models


# IT: una epoca (train o eval). use_cafn=False → baseline latent=None (per il gate).
# EN: one epoch (train or eval). use_cafn=False → latent=None baseline (for the gate).
def run_epoch(cafn, models, X, y, opt, device, lam, batch, train=True,
              use_cafn=True, max_steps=None):
    N = X.shape[0]
    idx = torch.randperm(N) if train else torch.arange(N)
    if cafn is not None:
        cafn.train(train and use_cafn)
    for m in models.values():
        m.train(train)
    agg = {a: 0.0 for a in models}
    agg["penalty"], n_batch = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi in range(0, N, batch):
            sel = idx[bi:bi + batch]
            xb = X[sel].to(device)
            yb = y[sel].to(device).float()
            latent, pen = (cafn(xb) if use_cafn else (None, xb.new_zeros(())))
            loss = lam * pen if (use_cafn and train) else xb.new_zeros(())
            for a, m in models.items():
                out = m(xb, None, latent)
                pl = F.mse_loss(_mu_of(out), yb)
                loss = loss + pl
                agg[a] += float(pl.detach())
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in ([cafn] if use_cafn else []) + list(models.values())
                     for p in g.parameters()], 5.0)
                opt.step()
            agg["penalty"] += float(pen.detach()) if use_cafn else 0.0
            n_batch += 1
            if max_steps and n_batch >= max_steps:
                break
    return {k: v / max(n_batch, 1) for k, v in agg.items()}


def load_real_data():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset {DATASET} assente (cleanup 2026-06-12). Rigenera con "
            f"`python scripts/01_download_data.py` (+ `scripts/vol/dev_vols_macro_append.py` "
            f"per il target vol) PRIMA di lanciare il training reale. Oppure usa "
            f"`--smoke` per validare il loop end-to-end su dati sintetici / "
            f"Regenerate the dataset before real training, or use `--smoke`.")
    d = np.load(DATASET, allow_pickle=True)
    to_t = lambda k: torch.from_numpy(np.asarray(d[k], dtype=np.float32))
    return to_t("X_train"), to_t("y_train"), to_t("X_val"), to_t("y_val")


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="CAFN joint end-to-end trainer")
    ap.add_argument("--smoke", action="store_true",
                    help="dati sintetici su CPU per validare il loop / synthetic CPU smoke")
    ap.add_argument("--archs", nargs="+", default=["itransformer", "tcnmamba", "nhits"],
                    choices=["itransformer", "tcnmamba", "nhits"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-latent", type=int, default=16)
    ap.add_argument("--cafn-d-model", type=int, default=64)
    ap.add_argument("--cafn-layers", type=int, default=2)
    ap.add_argument("--lambda-causal", type=float, default=0.1,
                    help="peso λ della penalità causale / weight of the causal penalty")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap batch/epoca (debug) / cap batches per epoch (debug)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-gate", action="store_true",
                    help="salta il baseline NO-CAFN del gate / skip the NO-CAFN gate baseline")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    mcfg = dict(cfg.get("model", {}))
    # IT: overlay opzionale config/cafn.yaml (se presente) per i parametri CAFN.
    # EN: optional config/cafn.yaml overlay (if present) for CAFN params.
    cafn_yaml = ROOT / "config" / "cafn.yaml"
    if cafn_yaml.exists():
        try:
            mcfg.update(load_config(str(cafn_yaml)).get("cafn", {}))
        except Exception:
            pass

    device = torch.device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Dati / Data ───────────────────────────────────────────────────────────
    if args.smoke:
        log.info("SMOKE: dati sintetici / synthetic data (CPU-safe, loop validation)")
        torch.manual_seed(0)
        T, Fdim, Ntr, Nvl = 24, 16, 256, 96
        Xtr = torch.randn(Ntr, T, Fdim); Xvl = torch.randn(Nvl, T, Fdim)
        # IT: target con dipendenza causale debole dal passato (segnale apprendibile).
        # EN: target with weak causal dependence on the past (learnable signal).
        ytr = Xtr[:, -2, 0] * 0.5 + 0.1 * torch.randn(Ntr)
        yvl = Xvl[:, -2, 0] * 0.5 + 0.1 * torch.randn(Nvl)
    else:
        Xtr, ytr, Xvl, yvl = load_real_data()
        T, Fdim = Xtr.shape[1], Xtr.shape[2]
        log.warning("Training REALE: 3 modelli + CAFN in simultanea → rischio OOM su "
                    "8GB. Riduci --batch / --cafn-d-model se necessario.")

    n_in = Fdim + args.d_latent
    log.info(f"Dataset: X_train={tuple(Xtr.shape)} X_val={tuple(Xvl.shape)} | "
             f"T={T} F={Fdim} d_latent={args.d_latent} → n_in(augmented)={n_in}")
    log.info(f"Archs a valle / downstream: {args.archs} | λ_causal={args.lambda_causal} "
             f"| device={device}")

    # ── Costruzione moduli / Module construction ──────────────────────────────
    cafn = CausalAttentionFlowNetwork(
        n_features=Fdim, d_model=args.cafn_d_model, n_heads=4,
        n_layers=args.cafn_layers, d_latent=args.d_latent, max_len=max(T + 1, 64)
    ).to(device)
    models = build_downstream(args.archs, n_in, T, mcfg, device)
    n_params = sum(p.numel() for g in [cafn, *models.values()] for p in g.parameters())
    log.info(f"Parametri totali / total params: {n_params:,} "
             f"(CAFN {sum(p.numel() for p in cafn.parameters()):,})")

    all_params = [p for g in [cafn, *models.values()] for p in g.parameters()]
    opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)

    # ── Loop di training congiunto / Joint training loop ──────────────────────
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr = run_epoch(cafn, models, Xtr, ytr, opt, device, args.lambda_causal,
                       args.batch, train=True, use_cafn=True, max_steps=args.max_steps)
        vl = run_epoch(cafn, models, Xvl, yvl, opt, device, args.lambda_causal,
                       args.batch, train=False, use_cafn=True)
        history.append({"epoch": ep, "train": tr, "val": vl})
        msg = "  ".join(f"{a}={vl[a]:.5f}" for a in args.archs)
        log.info(f"[ep {ep}/{args.epochs}] val MSE-mu: {msg}  penalty={tr['penalty']:.4f}")

    # ── Gate val-first: baseline NO-CAFN (latent=None) ────────────────────────
    gate = {"evaluated": False}
    if not args.no_gate:
        log.info("Gate: baseline NO-CAFN (latent=None) — stessi modelli, ri-inizializzati.")
        base_models = build_downstream(args.archs, Fdim, T, mcfg, device)  # larghezza ORIGINALE
        base_opt = torch.optim.AdamW(
            [p for m in base_models.values() for p in m.parameters()],
            lr=args.lr, weight_decay=1e-4)
        for ep in range(1, args.epochs + 1):
            run_epoch(None, base_models, Xtr, ytr, base_opt, device, 0.0,
                      args.batch, train=True, use_cafn=False, max_steps=args.max_steps)
        base_vl = run_epoch(None, base_models, Xvl, yvl, None, device, 0.0,
                            args.batch, train=False, use_cafn=False)
        cafn_vl = history[-1]["val"]
        wins = 0
        per_arch = {}
        for a in args.archs:
            impr = (base_vl[a] - cafn_vl[a]) / max(base_vl[a], 1e-9)
            per_arch[a] = {"cafn": cafn_vl[a], "baseline": base_vl[a], "improvement": impr}
            wins += int(impr >= 0.03)
            log.info(f"  {a}: CAFN={cafn_vl[a]:.5f} vs baseline={base_vl[a]:.5f} "
                     f"→ {impr*100:+.2f}%  {'WIN' if impr>=0.03 else ''}")
        passed = wins >= 2
        gate = {"evaluated": True, "rule": ">=3% MSE-mu val su >=2/3 archi",
                "wins": wins, "passed": bool(passed), "per_arch": per_arch}
        log.info(f"GATE: {wins}/{len(args.archs)} archi ≥3% → "
                 f"{'PASS' if passed else 'FAIL (CAFN-coordinatore KILL come da prior)'}")

    # ── Salvataggio ISOLATO / ISOLATED save ───────────────────────────────────
    torch.save(cafn.state_dict(), OUT_DIR / "cafn.pt")
    for a, m in models.items():
        torch.save(m.state_dict(), OUT_DIR / f"downstream_{a}.pt")
    report = {"smoke": args.smoke, "archs": args.archs, "T": T, "F": Fdim,
              "d_latent": args.d_latent, "lambda_causal": args.lambda_causal,
              "epochs": args.epochs, "elapsed_s": round(time.time() - t0, 1),
              "history": history, "gate": gate}
    (RES_DIR / "cafn_joint_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    log.info(f"Salvato / saved → {OUT_DIR}/  ·  report → {RES_DIR}/cafn_joint_report.json")
    log.info(f"Completato in / done in {report['elapsed_s']}s")


if __name__ == "__main__":
    main()
