"""EnsembleModel: inference over N checkpoints, homogeneous or heterogeneous.

Modes:
  - Homogeneous (legacy): N checkpoints best_model_0..N-1.pt of the same architecture
  - Heterogeneous (distillation): 1 checkpoint per architecture (itransformer, nhits, tcnmamba)
"""
import json
import math
from pathlib import Path
import logging
import torch
import torch.nn.functional as F

from quantsys.utils import models_root as _models_root

log = logging.getLogger("quantsys.model.ensemble")

# Softmax temperature for inverse-NLL weights (lower = sharper).
DEFAULT_NLL_TEMPERATURE = 0.05

# Default heterogeneous ensemble composition (override via config.yaml).
HETEROGENEOUS_ARCHS = ["itransformer", "nhits", "tcnmamba"]


# Resolves the archs list for distill/ensemble (cfg override → constant fallback).
def get_distillation_archs(cfg: dict = None) -> list:
    """Returns the list of architectures for the distill/heterogeneous
    ensemble pipeline.

    Reads `cfg["distillation"]["archs"]` if present, otherwise falls back
    to HETEROGENEOUS_ARCHS. Silently filters out non-string or empty entries.
    Allows changing the composition (e.g. swap lstm/nhits, add a 4th
    model) by editing a single line in `config/default.yaml`.
    """
    if cfg is not None and isinstance(cfg.get("distillation"), dict):
        archs = cfg["distillation"].get("archs")
        if isinstance(archs, (list, tuple)) and archs:
            cleaned = [str(a).strip().lower() for a in archs if a]
            cleaned = [a for a in cleaned if a]
            if cleaned:
                return cleaned
    return list(HETEROGENEOUS_ARCHS)

# Tolerance (seconds) between sequential writes of the same training run:
# within this window best_model.pt and numbered members are considered coherent.
STALE_MEMBERS_TOLERANCE_S = 60.0


# Non-fatal anti-stale guard (2026-06-10 bug): if best_model.pt is much newer than
# the numbered members, those are likely leftovers from a previous run
# (different config/target/interval) that load() will prefer, ignoring the new best.
def _stale_members_warning(base: Path) -> str | None:
    """Returns a warning message if the numbered members look stale, otherwise None.

    Condition: `best_model_[0-9]*.pt` members AND `best_model.pt` exist AND
    mtime(best_model.pt) > max(mtime(members)) + STALE_MEMBERS_TOLERANCE_S.
    Warning-only: does not alter checkpoint selection in any way.
    """
    members = sorted(base.glob("best_model_[0-9]*.pt"))
    single = base / "best_model.pt"
    # without numbered members or without the single best there is no ambiguity → no warning.
    if not members or not single.exists():
        return None
    try:
        newest_member = max(p.stat().st_mtime for p in members)
        single_mtime = single.stat().st_mtime
    except OSError:
        # filesystem race (file removed between glob and stat) → degrade silently.
        return None
    if single_mtime > newest_member + STALE_MEMBERS_TOLERANCE_S:
        return (
            f"Possibili checkpoint ensemble STALE in {base}: best_model.pt è più recente "
            f"di {single_mtime - newest_member:.0f}s rispetto al più nuovo dei "
            f"{len(members)} membri numerati (best_model_*.pt). EnsembleModel.load "
            f"preferisce i membri numerati, quindi il best singolo più recente "
            f"(es. da un run --n-ensemble 1) viene IGNORATO. Rimedio: rimuovi/archivia "
            f"i membri numerati oppure ri-allena con n-ensemble pieno. | Possibly STALE "
            f"ensemble checkpoints in {base}: best_model.pt is newer than the newest of "
            f"the {len(members)} numbered members; load() prefers numbered members, so "
            f"the more recent single best is SILENTLY IGNORED. Remedy: remove/archive "
            f"the numbered members or retrain with full n-ensemble."
        )
    return None


# Per-architecture default weights; overridable via arch_weights kwarg.
DEFAULT_ARCH_WEIGHTS = {
    "itransformer": 1.0,
    "nhits":        1.0,
    "tcnmamba":     1.0,
    "lstm":         0.5,
    "tft":          1.0,
}


# Data-driven weights via softmax(-val_nll/T) — temperature-scaled Bayesian Model Averaging.
def _compute_dynamic_weights(arch_names: list,
                             models_root: Path = None,
                             temperature: float = DEFAULT_NLL_TEMPERATURE) -> dict:
    """Computes per-architecture weights using an inverse-NLL softmax over the
    best val_nll read from `models/{arch}/history.json`.

    Formula (Strategy C, principled BMA):
      w_i = exp(-(NLL_i - NLL_min) / T) / Z

    Subtracting NLL_min is only for numerical stability (the exponentials
    stay in [0,1]); the final weights are identical to exp(-NLL_i/T)/Z.

    Returns dict {arch: raw_weight} (NOT normalized — final normalization
    stays in __init__). If a history.json is missing or malformed, that
    arch gets weight 1.0 (partial uniform fallback). If ALL are missing,
    returns an empty dict → the caller will use DEFAULT_ARCH_WEIGHTS.
    """
    # default = env-aware root (QUANTSYS_MODELS_ROOT) for isolated experiments.
    if models_root is None:
        models_root = _models_root()

    nlls = {}
    for arch in arch_names:
        hist_path = models_root / arch / "history.json"
        if not hist_path.exists():
            log.warning(f"_compute_dynamic_weights: {hist_path} non trovato, "
                        f"{arch} userà peso default")
            continue
        try:
            with open(hist_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            vnll = hist.get("val_nll", [])
            # Filter out NaN/Inf and non-finite values (early epochs may diverge).
            vnll = [float(v) for v in vnll
                    if v is not None and isinstance(v, (int, float))
                    and math.isfinite(float(v))]
            if not vnll:
                log.warning(f"_compute_dynamic_weights: history.json di {arch} "
                            f"non ha val_nll finiti, peso default")
                continue
            nlls[arch] = min(vnll)
        except Exception as e:
            log.warning(f"_compute_dynamic_weights: errore lettura {hist_path}: {e}")
            continue

    if not nlls:
        # No metric available → fallback to default uniform weights.
        return {}

    # Numerically stable softmax: subtract min before exponentiating.
    nll_min = min(nlls.values())
    T = max(float(temperature), 1e-6)                              # avoid div/0
    raw = {a: math.exp(-(v - nll_min) / T) for a, v in nlls.items()}
    Z   = sum(raw.values())
    if Z <= 0:
        return {}
    weights = {a: r / Z for a, r in raw.items()}

    # For archs missing history.json assign the median of computed weights
    # (compromise: neither favors nor penalizes the unknown arch).
    if len(weights) < len(arch_names):
        median_w = sorted(weights.values())[len(weights) // 2]
        for arch in arch_names:
            weights.setdefault(arch, median_w)

    log.info(
        f"_compute_dynamic_weights (inverse-NLL softmax, T={T:g}): "
        + ", ".join(f"{a}=val_nll {nlls.get(a, float('nan')):.4f}→w {weights[a]:.3f}"
                    for a in arch_names)
    )
    return weights


class EnsembleModel:
    """
    Loads N models and takes the (weighted) average of their predictions.

    Uncertainty combination formula (law of total variance, weighted):
      mu_ens    = Sum_i w_i * mu_i
      sigma_ens = sqrt(Sum_i w_i * sigma_i^2 + Sum_i w_i * (mu_i - mu_ens)^2)
    where w_i are the weights normalized to sum to 1 (one per model).
    """

    # Builds ensemble from preloaded models + resolves normalized weights.
    def __init__(self, models: list, device: torch.device,
                 arch_names: list = None, arch_weights: dict = None):
        self._models = models
        self._device = device
        self._arch_names = arch_names or [f"model_{i}" for i in range(len(models))]
        # Defaults to DEFAULT_ARCH_WEIGHTS; 1.0 for unknown archs.
        wmap = dict(DEFAULT_ARCH_WEIGHTS)
        if arch_weights:
            wmap.update(arch_weights)
        raw = [float(wmap.get(a, 1.0)) for a in self._arch_names]
        s = sum(raw)
        if s <= 0:
            raw = [1.0] * len(self._models)
            s = float(len(self._models))
        self._weights = [w / s for w in raw]                      # sum=1
        log.info(
            "EnsembleModel pesi: "
            + ", ".join(f"{a}={w:.3f}" for a, w in zip(self._arch_names, self._weights))
        )

    # Loads a homogeneous ensemble from best_model_*.pt; falls back to a single best_model.pt.
    @classmethod
    def load(cls, models_dir: str, device: torch.device) -> "EnsembleModel":
        """Loads all available best_model_*.pt. Falls back to best_model.pt."""
        from quantsys.model import load_model

        base = Path(models_dir)

        # Warning-only anti-stale guard (2026-06-10 bug): flags numbered members
        # possibly left over from a previous run; does NOT change selection.
        _stale_msg = _stale_members_warning(base)
        if _stale_msg is not None:
            log.warning(_stale_msg)

        ckpts = sorted(base.glob("best_model_[0-9]*.pt"),
                       key=lambda p: int(p.stem.split("_")[-1]))

        if len(ckpts) >= 2:
            models = []
            for ckpt in ckpts:
                m = load_model(str(ckpt)).to(device)
                m.eval()
                models.append(m)
            log.info(f"EnsembleModel: {len(models)} membri caricati da {base}/")
        else:
            if ckpts:
                log.warning(
                    f"Trovato solo 1 checkpoint ensemble ({ckpts[0].name}); "
                    "uso best_model.pt come modello singolo."
                )
            fallback = base / "best_model.pt"
            m = load_model(str(fallback)).to(device)
            m.eval()
            models = [m]
            log.info(f"EnsembleModel: 1 membro caricato (fallback a {fallback})")

        # arch_names not passed → __init__ uses default ["model_0", ...]. Safe because
        # in homogeneous ensemble (load()) all members share the same arch, and the
        # DEFAULT_ARCH_WEIGHTS.get(a, 1.0) fallback to 1.0 → correct uniform average.
        return cls(models, device)

    # Loads one checkpoint per architecture from models/{arch}/ (heterogeneous ensemble); skips missing ones.
    @classmethod
    def load_heterogeneous(cls, device: torch.device,
                           archs: list = None,
                           cfg:   dict = None) -> "EnsembleModel":
        """Loads one model per available architecture (heterogeneous ensemble).

        Looks for best_model.pt in models/{arch}/ for each architecture.
        Skips architectures without a checkpoint.

        archs list resolution (in priority order):
          1. explicit `archs` parameter
          2. `cfg["distillation"]["archs"]` if cfg is given
          3. HETEROGENEOUS_ARCHS constant
        """
        from quantsys.model import load_model

        if archs is None:
            archs = get_distillation_archs(cfg)

        # env-aware root — the heterogeneous ensemble reads from the isolated sandbox if set.
        _mroot = _models_root()
        models = []
        arch_names = []
        for arch in archs:
            ckpt = _mroot / arch / "best_model.pt"
            if not ckpt.exists():
                log.warning(f"Ensemble eterogeneo: {ckpt} non trovato, skip {arch}")
                continue
            try:
                m = load_model(str(ckpt)).to(device)
                m.eval()
                models.append(m)
                arch_names.append(arch)
                log.info(f"  Caricato {arch}: {sum(p.numel() for p in m.parameters()):,} params")
            except Exception as e:
                log.warning(f"Errore caricamento {arch}: {e}")

        if not models:
            raise FileNotFoundError(
                "Nessun checkpoint trovato per ensemble eterogeneo. "
                "Addestra almeno un modello."
            )

        log.info(f"EnsembleModel eterogeneo: {len(models)} architetture "
                 f"[{', '.join(arch_names)}]")

        # Strategy C — data-driven weights via inverse-NLL softmax on best
        # val_nll. Replaces the uniform default (1/n) from DEFAULT_ARCH_WEIGHTS.
        # Optional temperature via cfg["distillation"]["ensemble_nll_temperature"].
        temperature = DEFAULT_NLL_TEMPERATURE
        if isinstance(cfg, dict) and isinstance(cfg.get("distillation"), dict):
            tcfg = cfg["distillation"].get("ensemble_nll_temperature")
            if isinstance(tcfg, (int, float)) and tcfg > 0:
                temperature = float(tcfg)
        dyn_weights = _compute_dynamic_weights(arch_names, models_root=_mroot,
                                               temperature=temperature)
        # If empty → fallback to DEFAULT_ARCH_WEIGHTS (uniform). Otherwise override.
        return cls(models, device, arch_names,
                   arch_weights=dyn_weights if dyn_weights else None)

    # (device,dtype)-cached weights tensor — weights don't change after init.
    def _weights_tensor(self, device, dtype):
        cache = getattr(self, "_weights_cache", None)
        if cache is None:
            cache = {}
            self._weights_cache = cache
        key = (device, dtype)
        t = cache.get(key)
        if t is None:
            t = torch.tensor(self._weights, device=device, dtype=dtype).view(-1, 1)
            cache[key] = t
        return t

    # Forward across all members + total-variance-law fusion.
    def __call__(self, *args, **kwargs):
        """Forwards to all models and combines the output."""
        mus, sigs, nus_list = [], [], []

        with torch.no_grad(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=False,   # AMP off: avoids NaN (spectral_norm + Mamba scan)
        ):
            for m in self._models:
                loss_type = getattr(m, "loss_type", "t_student")
                out = m(*args, **kwargs)
                if loss_type == "quantile":
                    qp    = out[0]
                    qp, _ = qp.sort(dim=-1)
                    mu_i  = qp[:, 2]
                    sig_i = (qp[:, 4] - qp[:, 0]).clamp(min=1e-6)
                    nu_i  = torch.full_like(mu_i, 5.0)
                else:
                    mu_i  = out[0]
                    sig_i = (F.softplus(out[1]) + 1e-6).sqrt()
                    nu_i  = F.softplus(out[2]) + 2.0 + 1e-6
                mus.append(mu_i)
                sigs.append(sig_i)
                nus_list.append(nu_i)

        mus_t  = torch.stack(mus,      dim=0)                # (N, B)
        sigs_t = torch.stack(sigs,     dim=0)
        nus_t  = torch.stack(nus_list, dim=0)

        # Weights broadcast to (N,1) for tensor-friendly fusion. Cached per (device,dtype):
        # weights are immutable post-init → avoids rebuilding the tensor on every forward (A7).
        w = self._weights_tensor(mus_t.device, mus_t.dtype)

        mu_ens = (w * mus_t).sum(dim=0)
        nu_ens = (w * nus_t).sum(dim=0)

        # Total variance law: E[σ²] + Var[μ_i] = within + between models.
        sig2_mean = (w * sigs_t ** 2).sum(dim=0)
        mu_var    = (w * (mus_t - mu_ens.unsqueeze(0)) ** 2).sum(dim=0)
        sigma_ens = (sig2_mean + mu_var).clamp(min=1e-12).sqrt()

        return mu_ens, sigma_ens, nu_ens

    # Number of members in the ensemble.
    @property
    def n_members(self) -> int:
        return len(self._models)

    # Normalized per-member weights (sum to 1).
    @property
    def weights(self) -> list:
        """Normalized weights (summing to 1) for each ensemble member."""
        return list(self._weights)

    # Member architecture names.
    @property
    def arch_names(self) -> list:
        return self._arch_names

    # True if the ensemble mixes different architectures.
    @property
    def is_heterogeneous(self) -> bool:
        return len(set(self._arch_names)) > 1

    # Puts all members into eval mode.
    def eval(self):
        for m in self._models:
            m.eval()
        return self

    # Puts all members into train/eval mode (mode bool).
    def train(self, mode: bool = True):
        for m in self._models:
            m.train(mode)
        return self
