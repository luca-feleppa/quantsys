"""EnsembleModel: inferenza su N checkpoint, omogeneo o eterogeneo.

Modalita':
  - Omogeneo (legacy): N checkpoint best_model_0..N-1.pt della stessa architettura
  - Eterogeneo (distillation): 1 checkpoint per architettura (itransformer, nhits, tcnmamba)
"""
from pathlib import Path
import logging
import torch
import torch.nn.functional as F

log = logging.getLogger("quantsys.model.ensemble")

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
        return cls(models, device, arch_names)

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
