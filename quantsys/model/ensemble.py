"""EnsembleModel: inferenza su N checkpoint, omogeneo o eterogeneo.

Modalita':
  - Omogeneo (legacy): N checkpoint best_model_0..N-1.pt della stessa architettura
  - Eterogeneo (distillation): 1 checkpoint per architettura (itransformer, nhits, tcnmamba)
"""
import json
import math
from pathlib import Path
import logging
import torch
import torch.nn.functional as F

log = logging.getLogger("quantsys.model.ensemble")

# IT: Temperatura softmax per pesi inverse-NLL (più bassa = più discriminativa).
# EN: Softmax temperature for inverse-NLL weights (lower = sharper).
DEFAULT_NLL_TEMPERATURE = 0.05

# IT: Composizione default ensemble eterogeneo (override via config.yaml).
# EN: Default heterogeneous ensemble composition (override via config.yaml).
HETEROGENEOUS_ARCHS = ["itransformer", "nhits", "tcnmamba"]


# IT: Risolve la lista archi per distill/ensemble (cfg override → fallback costante).
# EN: Resolves the archs list for distill/ensemble (cfg override → constant fallback).
def get_distillation_archs(cfg: dict = None) -> list:
    """Restituisce la lista di architetture per la pipeline distill/ensemble
    eterogeneo.

    Legge da `cfg["distillation"]["archs"]` se presente, altrimenti fallback
    a HETEROGENEOUS_ARCHS. Filtra silenziosamente entry non-stringa o vuote.
    Permette di cambiare composizione (es. swap lstm/nhits, aggiungere 4°
    modello) modificando una sola riga in `config/default.yaml`.
    """
    if cfg is not None and isinstance(cfg.get("distillation"), dict):
        archs = cfg["distillation"].get("archs")
        if isinstance(archs, (list, tuple)) and archs:
            cleaned = [str(a).strip().lower() for a in archs if a]
            cleaned = [a for a in cleaned if a]
            if cleaned:
                return cleaned
    return list(HETEROGENEOUS_ARCHS)

# IT: Pesi default per architettura; sovrascrivibili via arch_weights kwarg.
# EN: Per-architecture default weights; overridable via arch_weights kwarg.
DEFAULT_ARCH_WEIGHTS = {
    "itransformer": 1.0,
    "nhits":        1.0,
    "tcnmamba":     1.0,
    "lstm":         0.5,
    "tft":          1.0,
}


# IT: Pesi data-driven via softmax(-val_nll/T) — Bayesian Model Averaging stile temperature-scaled.
# EN: Data-driven weights via softmax(-val_nll/T) — temperature-scaled Bayesian Model Averaging.
def _compute_dynamic_weights(arch_names: list,
                             models_root: Path = None,
                             temperature: float = DEFAULT_NLL_TEMPERATURE) -> dict:
    """Calcola pesi per architettura usando inverse-NLL softmax sui best
    val_nll letti da `models/{arch}/history.json`.

    Formula (Strategia C, principled BMA):
      w_i = exp(-(NLL_i - NLL_min) / T) / Z

    La sottrazione di NLL_min serve solo a stabilità numerica (gli esponenziali
    rimangono in [0,1]); i pesi finali sono identici a exp(-NLL_i/T)/Z.

    Restituisce dict {arch: peso_raw} (NON normalizzato — la normalizzazione
    finale resta in __init__). Se una history.json manca o è malformata,
    quell'arch riceve peso 1.0 (fallback uniforme parziale). Se TUTTE
    mancano, restituisce dict vuoto → caller userà DEFAULT_ARCH_WEIGHTS.
    """
    if models_root is None:
        models_root = Path("models")

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
            # IT: Filtra NaN/Inf e valori non finiti (early epochs possono divergere).
            # EN: Filter out NaN/Inf and non-finite values (early epochs may diverge).
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
        # IT: Nessuna metrica disponibile → fallback a pesi default uniformi.
        # EN: No metric available → fallback to default uniform weights.
        return {}

    # IT: Softmax stabile numericamente: sottraggo il minimo prima di esponenziare.
    # EN: Numerically stable softmax: subtract min before exponentiating.
    nll_min = min(nlls.values())
    T = max(float(temperature), 1e-6)                              # IT: evita div/0 | EN: avoid div/0
    raw = {a: math.exp(-(v - nll_min) / T) for a, v in nlls.items()}
    Z   = sum(raw.values())
    if Z <= 0:
        return {}
    weights = {a: r / Z for a, r in raw.items()}

    # IT: Per archi senza history.json assegna la mediana dei pesi calcolati
    #     (compromesso: non favorisce né penalizza l'arch sconosciuto).
    # EN: For archs missing history.json assign the median of computed weights
    #     (compromise: neither favors nor penalizes the unknown arch).
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
    Carica N modelli e fa media (pesata) delle previsioni.

    Formula combinazione incertezza (legge della varianza totale, weighted):
      mu_ens    = Sum_i w_i * mu_i
      sigma_ens = sqrt(Sum_i w_i * sigma_i^2 + Sum_i w_i * (mu_i - mu_ens)^2)
    dove w_i sono i pesi normalizzati a sommare 1 (uno per modello).
    """

    # IT: Costruisce ensemble da modelli già caricati + risolve pesi normalizzati.
    # EN: Builds ensemble from preloaded models + resolves normalized weights.
    def __init__(self, models: list, device: torch.device,
                 arch_names: list = None, arch_weights: dict = None):
        self._models = models
        self._device = device
        self._arch_names = arch_names or [f"model_{i}" for i in range(len(models))]
        # IT: Default DEFAULT_ARCH_WEIGHTS; 1.0 per arch ignote.
        # EN: Defaults to DEFAULT_ARCH_WEIGHTS; 1.0 for unknown archs.
        wmap = dict(DEFAULT_ARCH_WEIGHTS)
        if arch_weights:
            wmap.update(arch_weights)
        raw = [float(wmap.get(a, 1.0)) for a in self._arch_names]
        s = sum(raw)
        if s <= 0:
            raw = [1.0] * len(self._models)
            s = float(len(self._models))
        self._weights = [w / s for w in raw]                      # IT: somma=1 | EN: sum=1
        log.info(
            "EnsembleModel pesi: "
            + ", ".join(f"{a}={w:.3f}" for a, w in zip(self._arch_names, self._weights))
        )

    # IT: Carica ensemble omogeneo dai best_model_*.pt; fallback a best_model.pt singolo.
    # EN: Loads a homogeneous ensemble from best_model_*.pt; falls back to a single best_model.pt.
    @classmethod
    def load(cls, models_dir: str, device: torch.device) -> "EnsembleModel":
        """Carica tutti i best_model_*.pt disponibili. Fallback a best_model.pt."""
        from quantsys.model import load_model

        base = Path(models_dir)
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

        # IT: arch_names non passato → __init__ usa default ["model_0", ...]. Safe perché
        #     in ensemble omogeneo (load()) tutti i membri condividono la stessa arch, e
        #     i pesi DEFAULT_ARCH_WEIGHTS.get(a, 1.0) fallback a 1.0 → media uniforme corretta.
        # EN: arch_names not passed → __init__ uses default ["model_0", ...]. Safe because
        #     in homogeneous ensemble (load()) all members share the same arch, and the
        #     DEFAULT_ARCH_WEIGHTS.get(a, 1.0) fallback to 1.0 → correct uniform average.
        return cls(models, device)

    # IT: Carica un checkpoint per architettura da models/{arch}/ (ensemble eterogeneo); salta le mancanti.
    # EN: Loads one checkpoint per architecture from models/{arch}/ (heterogeneous ensemble); skips missing ones.
    @classmethod
    def load_heterogeneous(cls, device: torch.device,
                           archs: list = None,
                           cfg:   dict = None) -> "EnsembleModel":
        """Carica un modello per ogni architettura disponibile (ensemble eterogeneo).

        Cerca best_model.pt in models/{arch}/ per ogni architettura.
        Salta le architetture senza checkpoint.

        Risoluzione lista archs (in ordine di priorità):
          1. parametro `archs` esplicito
          2. `cfg["distillation"]["archs"]` se cfg fornito
          3. costante HETEROGENEOUS_ARCHS
        """
        from quantsys.model import load_model

        if archs is None:
            archs = get_distillation_archs(cfg)

        models = []
        arch_names = []
        for arch in archs:
            ckpt = Path("models") / arch / "best_model.pt"
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

        # IT: Strategia C — pesi data-driven via inverse-NLL softmax sui best
        #     val_nll. Sostituisce il default uniforme (1/n) di DEFAULT_ARCH_WEIGHTS.
        #     Temperatura opzionale da cfg["distillation"]["ensemble_nll_temperature"].
        # EN: Strategy C — data-driven weights via inverse-NLL softmax on best
        #     val_nll. Replaces the uniform default (1/n) from DEFAULT_ARCH_WEIGHTS.
        #     Optional temperature via cfg["distillation"]["ensemble_nll_temperature"].
        temperature = DEFAULT_NLL_TEMPERATURE
        if isinstance(cfg, dict) and isinstance(cfg.get("distillation"), dict):
            tcfg = cfg["distillation"].get("ensemble_nll_temperature")
            if isinstance(tcfg, (int, float)) and tcfg > 0:
                temperature = float(tcfg)
        dyn_weights = _compute_dynamic_weights(arch_names, temperature=temperature)
        # IT: Se vuoto → fallback a DEFAULT_ARCH_WEIGHTS (uniforme). Altrimenti override.
        # EN: If empty → fallback to DEFAULT_ARCH_WEIGHTS (uniform). Otherwise override.
        return cls(models, device, arch_names,
                   arch_weights=dyn_weights if dyn_weights else None)

    # IT: Forward su tutti i membri + fusione con legge della varianza totale.
    # EN: Forward across all members + total-variance-law fusion.
    def __call__(self, *args, **kwargs):
        """Forwarda a tutti i modelli e combina l'output."""
        mus, sigs, nus_list = [], [], []

        with torch.no_grad(), torch.amp.autocast(
            device_type=self._device.type,
            enabled=False,   # IT: AMP off: evita NaN (spectral_norm + Mamba scan) | EN: AMP off: avoids NaN
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

        # IT: Pesi broadcast su (N,1) per fusione tensor-friendly.
        # EN: Weights broadcast to (N,1) for tensor-friendly fusion.
        w = torch.tensor(self._weights, device=mus_t.device, dtype=mus_t.dtype).view(-1, 1)

        mu_ens = (w * mus_t).sum(dim=0)
        nu_ens = (w * nus_t).sum(dim=0)

        # IT: Total variance law: E[σ²] + Var[μ_i] = within + between models.
        # EN: Total variance law: E[σ²] + Var[μ_i] = within + between models.
        sig2_mean = (w * sigs_t ** 2).sum(dim=0)
        mu_var    = (w * (mus_t - mu_ens.unsqueeze(0)) ** 2).sum(dim=0)
        sigma_ens = (sig2_mean + mu_var).clamp(min=1e-12).sqrt()

        return mu_ens, sigma_ens, nu_ens

    # IT: Numero di membri nell'ensemble.
    # EN: Number of members in the ensemble.
    @property
    def n_members(self) -> int:
        return len(self._models)

    # IT: Pesi normalizzati (sommano a 1) per membro.
    # EN: Normalized per-member weights (sum to 1).
    @property
    def weights(self) -> list:
        """Pesi normalizzati (sommano a 1) per ogni membro dell'ensemble."""
        return list(self._weights)

    # IT: Nomi delle architetture dei membri.
    # EN: Member architecture names.
    @property
    def arch_names(self) -> list:
        return self._arch_names

    # IT: True se l'ensemble mischia architetture diverse.
    # EN: True if the ensemble mixes different architectures.
    @property
    def is_heterogeneous(self) -> bool:
        return len(set(self._arch_names)) > 1

    # IT: Mette tutti i membri in eval mode.
    # EN: Puts all members into eval mode.
    def eval(self):
        for m in self._models:
            m.eval()
        return self

    # IT: Mette tutti i membri in train/eval mode (mode bool).
    # EN: Puts all members into train/eval mode (mode bool).
    def train(self, mode: bool = True):
        for m in self._models:
            m.train(mode)
        return self
