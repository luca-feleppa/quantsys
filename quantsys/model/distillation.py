"""Knowledge Distillation utilities for teacher-student training.

Supports transfer of output head weights from teacher to student and
distillation loss computation (KL divergence between t-Student distributions).
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantsys.utils import models_root as _models_root

log = logging.getLogger("quantsys.model.distillation")


# IT: Mappa naming heads (canonico → attributo arch-specifico).
# EN: Head naming map (canonical → arch-specific attribute name).
_HEAD_NAMES = {
    "itransformer": ("out_mu", "out_logsig2", "out_lognu"),
    "lstm":         ("out_mu", "out_logsig2", "out_lognu"),
    "tcnmamba":     ("mu_head", "ls2_head", "lnu_head"),
}


# IT: Rileva la naming convention delle output heads dell'arch.
# EN: Detects the output-head naming convention of the architecture.
def _get_head_names(model) -> tuple:
    """Detect which output head naming convention the model uses."""
    if hasattr(model, "mu_head"):
        return ("mu_head", "ls2_head", "lnu_head")
    return ("out_mu", "out_logsig2", "out_lognu")


# IT: Copia weight/bias gestendo mismatch su in_features (slice min_in).
# EN: Copies weight/bias handling in_features mismatch (slice to min_in).
def _copy_linear(t_head: nn.Linear, s_head: nn.Linear, label: str) -> int:
    """Copia weight/bias da t_head a s_head, gestendo d_model differenti.

    Se shapes esattamente uguali → copia completa.
    Se in_features differiscono → copia bias (calibrazione) + slice min_in del weight.
    """
    if t_head.out_features != s_head.out_features:
        log.warning(f"  {label}: out_features mismatch ({t_head.out_features} vs {s_head.out_features}) — skip")
        return 0
    n = 0
    if t_head.in_features == s_head.in_features:
        s_head.weight.data.copy_(t_head.weight.data)
        n += s_head.weight.numel()
        if t_head.bias is not None and s_head.bias is not None:
            s_head.bias.data.copy_(t_head.bias.data)
            n += s_head.bias.numel()
        log.info(f"  Transferred {label} (exact copy, {tuple(s_head.weight.shape)})")
    else:
        if t_head.bias is not None and s_head.bias is not None:
            s_head.bias.data.copy_(t_head.bias.data)
            n += s_head.bias.numel()
        min_in = min(t_head.in_features, s_head.in_features)
        s_head.weight.data[:, :min_in] = t_head.weight.data[:, :min_in]
        n += s_head.weight.shape[0] * min_in
        log.info(f"  Transferred {label} (partial: {min_in}/{s_head.in_features} features)")
    return n


# IT: Trasferisce heads teacher→student (warm-start della testa di output).
# EN: Transfers heads teacher→student (warm-start of the output head).
def transfer_output_heads(teacher: nn.Module, student: nn.Module) -> int:
    """Copy output head weights from teacher to student.

    Supporta sia loss_type='t_student' (mu/ls2/lnu heads) sia 'quantile'
    (quantile_head per single-expert, expert_gate + expert_heads per MoE).

    Pre-condizioni:
      - teacher e student devono avere stesso loss_type
      - se MoE: stesso n_output_experts

    Se le pre-condizioni non sono soddisfatte, skip e ritorna 0.
    Returns: numero di parametri effettivamente copiati.
    """
    t_loss = getattr(teacher, "loss_type", "t_student")
    s_loss = getattr(student, "loss_type", "t_student")
    t_moe = getattr(teacher, "n_output_experts", 1) or 1
    s_moe = getattr(student, "n_output_experts", 1) or 1

    if t_loss != s_loss:
        log.warning(f"Transfer heads: teacher loss={t_loss} != student loss={s_loss} — skip")
        return 0
    if t_moe != s_moe:
        log.warning(f"Transfer heads: teacher MoE={t_moe} != student MoE={s_moe} — skip")
        return 0

    # IT: Branch quantile — gestisce sia single-head sia MoE.
    # EN: Quantile branch — handles both single-head and MoE cases.
    if t_loss == "quantile":
        n_transferred = 0
        if t_moe > 1:
            # IT: MoE quantile: copia gate + ogni expert head | EN: MoE quantile: copy gate + each expert head
            t_gate = getattr(teacher, "expert_gate", None)
            s_gate = getattr(student, "expert_gate", None)
            if isinstance(t_gate, nn.Linear) and isinstance(s_gate, nn.Linear):
                n_transferred += _copy_linear(t_gate, s_gate, "expert_gate")
            t_heads = getattr(teacher, "expert_heads", None)
            s_heads = getattr(student, "expert_heads", None)
            if t_heads is not None and s_heads is not None and len(t_heads) == len(s_heads):
                for i, (th, sh) in enumerate(zip(t_heads, s_heads)):
                    if isinstance(th, nn.Linear) and isinstance(sh, nn.Linear):
                        n_transferred += _copy_linear(th, sh, f"expert_heads[{i}]")
        else:
            # IT: Single-expert: una sola head Linear(d_model, Q=5).
            # EN: Single-expert: a single Linear(d_model, Q=5) head.
            t_qh = getattr(teacher, "quantile_head", None)
            s_qh = getattr(student, "quantile_head", None)
            if isinstance(t_qh, nn.Linear) and isinstance(s_qh, nn.Linear):
                n_transferred += _copy_linear(t_qh, s_qh, "quantile_head")
            else:
                log.warning("Transfer quantile: quantile_head non trovato in teacher/student — skip")
        return n_transferred

    # IT: Branch t-Student | EN: t-Student branch
    if t_moe > 1:
        # IT: MoE t-Student: identico a MoE quantile ma expert out=3 (μ/ls2/lnu).
        # EN: MoE t-Student: identical to MoE quantile but expert out=3 (μ/ls2/lnu).
        n_transferred = 0
        t_gate = getattr(teacher, "expert_gate", None)
        s_gate = getattr(student, "expert_gate", None)
        if isinstance(t_gate, nn.Linear) and isinstance(s_gate, nn.Linear):
            n_transferred += _copy_linear(t_gate, s_gate, "expert_gate")
        t_heads = getattr(teacher, "expert_heads", None)
        s_heads = getattr(student, "expert_heads", None)
        if t_heads is not None and s_heads is not None and len(t_heads) == len(s_heads):
            for i, (th, sh) in enumerate(zip(t_heads, s_heads)):
                if isinstance(th, nn.Linear) and isinstance(sh, nn.Linear):
                    n_transferred += _copy_linear(th, sh, f"expert_heads[{i}]")
        return n_transferred

    # IT: Single-expert t-Student: copia μ/ls2/lnu head-by-head (legacy path).
    # EN: Single-expert t-Student: copy μ/ls2/lnu head-by-head (legacy path).
    t_names = _get_head_names(teacher)
    s_names = _get_head_names(student)
    n_transferred = 0
    for t_attr, s_attr in zip(t_names, s_names):
        t_head = getattr(teacher, t_attr, None)
        s_head = getattr(student, s_attr, None)
        if t_head is None or s_head is None:
            log.warning(f"Head not found: teacher.{t_attr}={t_head is not None}, "
                        f"student.{s_attr}={s_head is not None}")
            continue
        if not isinstance(t_head, nn.Linear) or not isinstance(s_head, nn.Linear):
            log.warning(f"Skipping non-Linear head: {t_attr}")
            continue
        n_transferred += _copy_linear(t_head, s_head, f"{t_attr}→{s_attr}")
    return n_transferred


# IT: MSE scale-normalized su (μ, σ, ν) — evita dominanza di ν.
# EN: Scale-normalized MSE on (μ, σ, ν) — prevents ν from dominating.
def distillation_loss_t_student(
    student_mu, student_ls2, student_lnu,
    teacher_mu, teacher_ls2, teacher_lnu,
) -> torch.Tensor:
    """Distillation loss: scale-normalized MSE between teacher and student params.

    Each component is normalized by the variance of the teacher's values so that
    mu, sigma and nu contribute equally regardless of their absolute scale.
    Without normalization, nu (~5) dominates and mu (~1e-5) is irrelevant.
    """
    s_sigma = (F.softplus(student_ls2) + 1e-6).sqrt()
    t_sigma = (F.softplus(teacher_ls2) + 1e-6).sqrt()

    s_nu = F.softplus(student_lnu) + 2.0 + 1e-6
    t_nu = F.softplus(teacher_lnu) + 2.0 + 1e-6

    # IT: unbiased=False: evita NaN su batch finale N=1 (var con N-1 indefinita).
    # EN: unbiased=False: avoids NaN on final N=1 batch (N-1 correction undefined).
    scale_mu  = (teacher_mu.detach().var(unbiased=False) + 1e-10)
    scale_sig = (t_sigma.detach().var(unbiased=False) + 1e-10)
    scale_nu  = (t_nu.detach().var(unbiased=False) + 1e-10)

    loss_mu    = F.mse_loss(student_mu, teacher_mu) / scale_mu
    loss_sigma = F.mse_loss(s_sigma, t_sigma)       / scale_sig
    loss_nu    = F.mse_loss(s_nu, t_nu)             / scale_nu

    return 0.5 * loss_mu + 0.3 * loss_sigma + 0.2 * loss_nu


# IT: Distillation loss quantile: MSE diretta tra quantili predetti.
# EN: Quantile distillation loss: plain MSE between predicted quantiles.
def distillation_loss_quantile(student_preds, teacher_preds) -> torch.Tensor:
    """Distillation loss for quantile regression: MSE between predicted quantiles."""
    return F.mse_loss(student_preds, teacher_preds)


# IT: MC Dropout: Dropout in train, BN/LN in eval (stats apprese).
# EN: MC Dropout: Dropout in train, BN/LN in eval (learned stats).
def _enable_mc_dropout(model: nn.Module) -> None:
    """Attiva i layer Dropout in train mode lasciando BN/LN in eval.

    Necessario per MC Dropout: durante l'inferenza vogliamo che il Dropout
    campioni maschere stocastiche (per catturare l'uncertainty del modello)
    ma che BatchNorm/LayerNorm usino le statistiche apprese (eval mode).
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


# IT: Genera soft labels del teacher (deterministico o MC Dropout per uncertainty).
# EN: Generates teacher soft labels (deterministic or MC Dropout for uncertainty).
def generate_teacher_predictions(teacher, dataloader, device, has_macro=False,
                                 mc_samples: int = 1):
    """Run teacher on entire dataset, return cached predictions.

    Args:
        teacher:     modello teacher caricato
        dataloader:  dataloader (deve essere shuffle=False per allineamento sample_idx)
        device:      torch device
        has_macro:   se True il batch contiene anche x_macro
        mc_samples:  numero di forward stocastici (MC Dropout). Se >1, abilita
                     Dropout in train mode (BN/LN restano in eval) e fa N forward
                     poi media le predizioni. Cattura l'uncertainty del teacher.

    Returns dict with keys:
      - 'mu': (N,) tensor of predicted means
      - 'ls2': (N,) tensor of log_sigma2 (raw, before softplus)
      - 'lnu': (N,) tensor of log_nu (raw, before softplus)

    For quantile models, returns:
      - 'quantiles': (N, Q) tensor of predicted quantiles
    """
    if mc_samples > 1:
        _enable_mc_dropout(teacher)
        log.info(f"  MC Dropout attivo: {mc_samples} forward per batch")
    else:
        teacher.eval()
    loss_type = getattr(teacher, "loss_type", "t_student")

    all_outputs = {"mu": [], "ls2": [], "lnu": [], "quantiles": []}

    # IT: no_grad (non inference_mode): spectral_norm richiede autograd attivo.
    # EN: no_grad (not inference_mode): spectral_norm requires autograd enabled.
    nograd_ctx = torch.no_grad() if mc_samples > 1 else torch.inference_mode()
    with nograd_ctx:
        for batch in dataloader:
            if has_macro:
                Xb, Xm, yb = [x.to(device, non_blocking=True) for x in batch]
                forward_args = (Xb, Xm)
            else:
                Xb, yb = [x.to(device, non_blocking=True) for x in batch]
                forward_args = (Xb,)

            # IT: K forward in GPU + media finale (un solo trasferimento → CPU).
            # EN: K forwards on GPU + final average (single GPU→CPU transfer).
            mu_acc = None; ls2_acc = None; lnu_acc = None; q_acc = None
            for _ in range(mc_samples):
                out = teacher(*forward_args)
                if loss_type == "quantile":
                    qp = out[0]
                    q_acc = qp if q_acc is None else q_acc + qp
                else:
                    if mu_acc is None:
                        mu_acc, ls2_acc, lnu_acc = out[0], out[1], out[2]
                    else:
                        mu_acc  = mu_acc  + out[0]
                        ls2_acc = ls2_acc + out[1]
                        lnu_acc = lnu_acc + out[2]

            if loss_type == "quantile":
                qp = q_acc / mc_samples
                all_outputs["quantiles"].append(qp.cpu())
                all_outputs["mu"].append(qp[:, 2].cpu())
                sig = (qp[:, 4] - qp[:, 0]).clamp(min=1e-6)
                all_outputs["ls2"].append(sig.cpu())
                all_outputs["lnu"].append(torch.full((len(qp),), 5.0))
            else:
                all_outputs["mu"].append((mu_acc  / mc_samples).cpu())
                all_outputs["ls2"].append((ls2_acc / mc_samples).cpu())
                all_outputs["lnu"].append((lnu_acc / mc_samples).cpu())

    result = {}
    for k, v in all_outputs.items():
        if v:
            result[k] = torch.cat(v, dim=0)
    return result


# IT: Carica il best checkpoint del teacher per l'arch indicata.
# EN: Loads the best teacher checkpoint for the given architecture.
def load_teacher(teacher_arch: str, device: torch.device):
    """Load the best teacher checkpoint for the given architecture."""
    from quantsys.model import load_model

    teacher_dir = _models_root() / teacher_arch
    ckpt = teacher_dir / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {ckpt}\n"
            f"Train the teacher first: python run_all.py --arch {teacher_arch}"
        )
    model = load_model(str(ckpt)).to(device)
    model.eval()
    log.info(f"Teacher loaded: {teacher_arch} from {ckpt}")
    return model


# IT: Heuristic overfit detector: val-train gap al best epoch > threshold.
# EN: Heuristic overfit detector: val-train gap at best epoch > threshold.
def _is_teacher_overfit(arch: str, overfit_gap_threshold: float = 1.0) -> bool:
    """Determina se un teacher è overfit ispezionando history.json.

    Criterio: `val_nll[best_val_epoch] - train_nll[best_val_epoch] > threshold`
    significa che train è significativamente più basso (= migliore) di val al
    best epoch → memorizzazione. NLL student-t può essere negativa: usiamo la
    differenza, non un rapporto.

    Returns True se overfit (da escludere dal pool teacher), False altrimenti.
    Restituisce False (= non escludere) se non riusciamo a leggere history.
    """
    import json
    hist_path = _models_root() / arch / "history.json"
    if not hist_path.exists():
        return False
    try:
        with open(hist_path, encoding="utf-8") as f:
            hist = json.load(f)
        train_nll = hist.get("train_nll", [])
        val_nll   = hist.get("val_nll",   [])
        if not train_nll or not val_nll or len(train_nll) != len(val_nll):
            return False
        # IT: Best epoch = argmin(val_nll) | EN: Best epoch = argmin(val_nll)
        best_idx = min(range(len(val_nll)), key=lambda i: val_nll[i])
        gap_at_best = val_nll[best_idx] - train_nll[best_idx]
        if gap_at_best > overfit_gap_threshold:
            log.warning(
                f"  Teacher {arch} ESCLUSO (overfit): val_nll={val_nll[best_idx]:+.3f}, "
                f"train_nll={train_nll[best_idx]:+.3f}, gap={gap_at_best:+.3f} > {overfit_gap_threshold}"
            )
            return True
        return False
    except Exception as e:
        log.warning(f"  _is_teacher_overfit({arch}): impossibile leggere history — {e}")
        return False


# IT: Soft labels da pool eterogeneo, blended con pesi normalizzati.
# EN: Soft labels from heterogeneous pool, blended with normalized weights.
def generate_multi_teacher_predictions(
    all_archs: list[str],
    arch_weights: dict[str, float],
    dataloader,
    device: torch.device,
    has_macro: bool = False,
    overfit_gap_threshold: float = 1.0,
    mc_samples: int = 1,
) -> dict[str, torch.Tensor]:
    """Generate weighted soft labels from multiple teachers.

    Instead of using only the best model as teacher, combine predictions
    from all trained architectures weighted by their normalized scores.

    Args:
        all_archs:    list of architecture names (e.g. ["itransformer", "nhits", "tcnmamba"])
        arch_weights: {arch_name: weight} — weights should sum to 1.0
        dataloader:   ordered dataloader (no shuffle) over training set
        device:       torch device
        has_macro:    whether dataloader includes macro features
        overfit_gap_threshold: teacher esclusi se val_nll - train_nll > threshold
                               al best epoch (default 1.0 NLL unit). Set a None
                               per disabilitare il filtro.
        mc_samples: numero di forward stocastici per teacher (MC Dropout). Se >1,
                    abilita Dropout in train mode durante l'inferenza e media K
                    forward — soft labels riflettono l'uncertainty del teacher
                    invece di un'unica predizione deterministica.

    Returns:
        dict with 'mu', 'ls2', 'lnu' tensors — weighted average of all teachers
    """
    teacher_preds = {}
    skipped_overfit = []
    for arch in all_archs:
        ckpt = _models_root() / arch / "best_model.pt"
        if not ckpt.exists():
            log.warning(f"Multi-teacher: {arch} checkpoint non trovato — skip")
            continue
        # IT: Esclude teacher overfit (train << val al best epoch).
        # EN: Excludes overfit teachers (train << val at best epoch).
        if overfit_gap_threshold is not None and _is_teacher_overfit(arch, overfit_gap_threshold):
            skipped_overfit.append(arch)
            continue
        try:
            model = load_teacher(arch, device)
            preds = generate_teacher_predictions(model, dataloader, device, has_macro,
                                                 mc_samples=mc_samples)
            teacher_preds[arch] = preds
            del model
            torch.cuda.empty_cache()
            log.info(f"  Multi-teacher: {arch} predictions generate ({preds['mu'].shape[0]} campioni)")
        except Exception as e:
            log.warning(f"  Multi-teacher: {arch} fallito — {e}")

    if skipped_overfit:
        log.warning(f"  Multi-teacher: esclusi per overfit: {skipped_overfit}")

    if not teacher_preds:
        raise RuntimeError(
            "Nessun teacher disponibile per multi-teacher distillation "
            "(tutti esclusi per overfit o checkpoint mancanti)"
        )

    available = {a: w for a, w in arch_weights.items() if a in teacher_preds}
    w_sum = sum(available.values())
    if w_sum < 1e-10:
        available = {a: 1.0 / len(teacher_preds) for a in teacher_preds}
        w_sum = 1.0
    norm_weights = {a: w / w_sum for a, w in available.items()}

    log.info("  Multi-teacher weights (normalizzati): "
             + ", ".join(f"{a}={w:.2f}" for a, w in norm_weights.items()))

    result = {}
    for key in ("mu", "ls2", "lnu"):
        blended = None
        for arch, w in norm_weights.items():
            t = teacher_preds[arch][key]
            if blended is None:
                blended = w * t
            else:
                blended = blended + w * t
        result[key] = blended

    return result


# IT: Pesi (val_loss, spearman, dir_acc) dello scoring teacher — TARGET-AWARE.
#     Single source of truth condivisa con `_select_best_teacher` (run_all.py).
#     • target direzionale (`ret`) o di SEGNO (`log_rs_ratio`): pesi storici
#       0.40 val_loss + 0.35 spearman + 0.25 dir_acc (il segno È il segnale).
#     • target di VOLATILITÀ (`log_rv`): la directional accuracy è il segno della
#       varianza-vs-mediana, NON un segnale tradabile (lo straddle è direction-
#       neutral) → si AZZERA e si ribilancia su val_loss (QLIKE/NLL, il momento
#       PARI che generalizza OOS) + spearman (qualità di RANGO della vol). Selezionare
#       un teacher per "dir_acc della vol" è scientificamente scorretto su questa linea.
# EN: Teacher-scoring weights (val_loss, spearman, dir_acc) — TARGET-AWARE.
#     Single source of truth shared with `_select_best_teacher` (run_all.py).
#     • directional (`ret`) or SIGN target (`log_rs_ratio`): historical weights
#       0.40 val_loss + 0.35 spearman + 0.25 dir_acc (the sign IS the signal).
#     • VOLATILITY target (`log_rv`): directional accuracy is the sign of
#       variance-vs-median, NOT a tradable signal (the straddle is direction-
#       neutral) → ZERO it and rebalance onto val_loss (QLIKE/NLL, the EVEN moment
#       that generalizes OOS) + spearman (rank quality of vol). Picking a teacher by
#       "vol dir_acc" is scientifically wrong on this line.
def teacher_score_weights(target_type: str = "ret") -> tuple[float, float, float]:
    """Return (w_val_loss, w_spearman, w_dir_acc) for the teacher scoring, by target."""
    if target_type == "log_rv":
        return (0.65, 0.35, 0.0)
    return (0.40, 0.35, 0.25)


# IT: Pesi teacher da metriche salvate (scoring target-aware via teacher_score_weights).
# EN: Teacher weights from saved metrics (target-aware scoring via teacher_score_weights).
def compute_teacher_weights(all_archs: list[str],
                            target_type: str = "ret") -> dict[str, float]:
    """Compute teacher weights from saved metrics (config.json).

    Uses the same normalized, TARGET-AWARE scoring as _select_best_teacher in
    run_all.py (see teacher_score_weights): on the vol target the dir_acc term is
    dropped. Returns weights proportional to scores (softmax-like).
    """
    import json
    import numpy as np

    metrics = {}
    for arch in all_archs:
        cfg_path = _models_root() / arch / "config.json"
        if not cfg_path.exists():
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        val_loss = cfg.get("best_val_loss", cfg.get("val_loss", 1e6))
        spearman = cfg.get("best_spearman", cfg.get("spearman", 0.0))
        da = cfg.get("best_da", cfg.get("directional_acc", 0.5))
        metrics[arch] = {"val_loss": val_loss, "spearman": spearman, "da": da}

    if len(metrics) < 2:
        return {a: 1.0 / len(all_archs) for a in all_archs}

    archs = list(metrics.keys())
    losses = np.array([metrics[a]["val_loss"] for a in archs])
    spearmans = np.array([metrics[a]["spearman"] for a in archs])
    das = np.array([metrics[a]["da"] for a in archs])

    # IT: Min-max normalize in [0,1]; se range≈0 ritorna pesi uniformi.
    # EN: Min-max normalize to [0,1]; if range≈0 returns uniform weights.
    def _norm(arr):
        r = arr.max() - arr.min()
        if r < 1e-10:
            return np.ones_like(arr) / len(arr)
        return (arr - arr.min()) / r

    w_vl, w_sp, w_da = teacher_score_weights(target_type)
    loss_norm = 1.0 - _norm(losses)
    spe_norm = _norm(spearmans)
    da_norm = _norm(das)
    scores = w_vl * loss_norm + w_sp * spe_norm + w_da * da_norm

    # IT: Softmax con T=2 — amplifica gap tra teacher (più peso al migliore).
    # EN: Softmax with T=2 — amplifies gap between teachers (more to the best).
    temp = 2.0
    exp_scores = np.exp(scores * temp)
    weights = exp_scores / exp_scores.sum()

    result = {a: float(weights[i]) for i, a in enumerate(archs)}
    for a in all_archs:
        if a not in result:
            result[a] = 0.0

    log.info("Teacher weights computed: " +
             ", ".join(f"{a}={result[a]:.3f}" for a in all_archs))
    return result
