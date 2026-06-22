"""
IT: Test dello scoring teacher TARGET-AWARE della pipeline distillation.
    Verifica che sul target di volatilità (`log_rv`) il termine directional-accuracy
    sia azzerato (lo straddle è direction-neutral: il segno della varianza-vs-mediana
    non è un segnale tradabile), mentre resti attivo sul target direzionale (`ret`).
EN: Tests for the TARGET-AWARE teacher scoring of the distillation pipeline.
    Verifies that on the volatility target (`log_rv`) the directional-accuracy term
    is zeroed (the straddle is direction-neutral: the sign of variance-vs-median is
    not a tradable signal), while it stays active on the directional target (`ret`).
"""
import json
import os

import pytest

from quantsys.model.distillation import teacher_score_weights, compute_teacher_weights


# ─── teacher_score_weights ─────────────────────────────────────────────────────
def test_weights_vol_drops_dir_acc():
    # IT: target vol → dir_acc=0, ribilanciato su val_loss+spearman, somma 1.
    # EN: vol target → dir_acc=0, rebalanced onto val_loss+spearman, sums to 1.
    w_vl, w_sp, w_da = teacher_score_weights("log_rv")
    assert w_da == 0.0
    assert (w_vl, w_sp, w_da) == (0.65, 0.35, 0.0)
    assert abs(w_vl + w_sp + w_da - 1.0) < 1e-9


def test_weights_directional_keeps_dir_acc():
    # IT: target direzionale (ret) e di segno (log_rs_ratio) → pesi storici.
    # EN: directional (ret) and sign (log_rs_ratio) targets → historical weights.
    for t in ("ret", "log_rs_ratio", "unknown_default"):
        w = teacher_score_weights(t)
        assert w == (0.40, 0.35, 0.25)
        assert abs(sum(w) - 1.0) < 1e-9


# ─── compute_teacher_weights (target-aware blend) ───────────────────────────────
def _write_arch_cfg(root, arch, val_loss, spearman, da):
    d = root / "models" / arch
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "config.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_loss": val_loss, "best_spearman": spearman,
                   "best_da": da}, f)


def test_blend_ignores_dir_acc_on_vol(tmp_path, monkeypatch):
    # IT: 2 arch identici tranne la dir_acc. Su 'log_rv' i pesi DEVONO restare
    #     uniformi (dir_acc ininfluente); su 'ret' l'arch con dir_acc alta pesa di più.
    # EN: 2 archs identical but for dir_acc. On 'log_rv' the weights MUST stay
    #     uniform (dir_acc irrelevant); on 'ret' the high-dir_acc arch weighs more.
    monkeypatch.chdir(tmp_path)
    _write_arch_cfg(tmp_path, "itransformer", val_loss=0.20, spearman=0.10, da=0.50)
    _write_arch_cfg(tmp_path, "nhits",        val_loss=0.20, spearman=0.10, da=0.90)
    archs = ["itransformer", "nhits"]

    w_vol = compute_teacher_weights(archs, target_type="log_rv")
    assert w_vol["itransformer"] == pytest.approx(w_vol["nhits"], abs=1e-6)

    w_dir = compute_teacher_weights(archs, target_type="ret")
    assert w_dir["nhits"] > w_dir["itransformer"]


def test_blend_prefers_lower_val_loss_on_vol(tmp_path, monkeypatch):
    # IT: su vol il val_loss (QLIKE/NLL) domina: l'arch con val_loss più basso pesa di più.
    # EN: on vol val_loss (QLIKE/NLL) dominates: the lower-val_loss arch weighs more.
    monkeypatch.chdir(tmp_path)
    _write_arch_cfg(tmp_path, "itransformer", val_loss=0.15, spearman=0.10, da=0.50)
    _write_arch_cfg(tmp_path, "nhits",        val_loss=0.40, spearman=0.10, da=0.50)
    w = compute_teacher_weights(["itransformer", "nhits"], target_type="log_rv")
    assert w["itransformer"] > w["nhits"]
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_default_target_type_is_directional():
    # IT: default back-compat = direzionale (firma invariata per i chiamanti legacy).
    # EN: back-compat default = directional (signature unchanged for legacy callers).
    assert teacher_score_weights() == (0.40, 0.35, 0.25)
