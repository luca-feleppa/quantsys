"""
RevIN — Reversible Instance Normalization for time series.

Kim et al. ICLR 2022, "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift".

Mitiga la non-stazionarietà locale: ogni window (B, T, F) viene normalizzata
con la propria media/std (calcolate sull'asse T), passata al modello, e le
predizioni vengono denormalizzate usando le stats della stessa istanza.

Per crypto/finance dove il regime cambia su scale di ore-giorni, RevIN
permette al modello di apprendere pattern invarianti rispetto a media/vol
locale, riducendo il distribution shift train→test.

Riferimento standard usato anche in iTransformer (Liu et al. 2024) e
PatchTST (Nie et al. 2023).
"""
import torch
import torch.nn as nn


# IT: Reversible Instance Normalization: normalizza/denormalizza per-istanza con affine.
# EN: Reversible Instance Normalization: per-instance normalize/denormalize with affine.
class RevIN(nn.Module):
    """
    Reversible Instance Normalization con affine learnable.

    Uso:
        revin = RevIN(n_features=119, target_idx=0)
        x_norm, stats = revin.normalize(x)                 # (B, T, F)
        out = model(x_norm)                                # model in spazio norm
        mu_orig    = revin.denormalize_mu(out.mu, stats)   # (B,) o (B, Q)
        logvar_orig = revin.denormalize_log_var(out.lv, stats)

    L'idea:
      * `target_idx` indica la colonna feature che corrisponde meglio al
        target (di solito `log_ret`). Le sue statistiche istanza vengono usate
        per denormalizzare la predizione scalare.
      * L'affine (gamma, beta) è learnable per-feature e applicato dopo la
        normalizzazione: dà al modello la libertà di riscalare ogni canale.
        L'inverso viene applicato in `denormalize_*` solo per `target_idx`.

    NOTA: la denormalizzazione assume che il target sia nella stessa scala
    della colonna `target_idx`. Se nel tuo dataset `log_ret` è a un indice
    diverso, configuralo via `model.revin_target_idx` in config/default.yaml.
    """

    # IT: Inizializza parametri affine learnable per-feature (se affine=True).
    # EN: Initializes learnable per-feature affine parameters (if affine=True).
    def __init__(self, n_features: int, target_idx: int = 0,
                 affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.n_features = n_features
        self.target_idx = target_idx
        self.affine = affine
        self.eps = eps
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(n_features))
            self.affine_bias = nn.Parameter(torch.zeros(n_features))

    # IT: Normalizza per-istanza sull'asse T; affine learnable per-feature.
    # EN: Per-instance normalization on T axis; learnable per-feature affine.
    def normalize(self, x: torch.Tensor):
        """
        x: (B, T, F) → x_norm: (B, T, F), stats: (mean_t, std_t) di shape (B,).

        mean/std calcolate sull'asse temporale (T), per-istanza, per-feature.
        Vengono detached: i gradienti non fluiscono attraverso le stats
        (l'affine ha i propri parametri learnable).
        """
        # IT: detach: gradienti NON fluiscono attraverso mean/std (solo affine).
        # EN: detach: gradients do NOT flow through mean/std (only affine).
        mean = x.mean(dim=1, keepdim=True).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps).detach()

        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        # IT: Estrae stats della colonna target per denormalizzare le predizioni.
        # EN: Extracts target-column stats to denormalize the predictions.
        mean_t = mean[:, 0, self.target_idx]
        std_t = std[:, 0, self.target_idx]
        return x_norm, (mean_t, std_t)

    # IT: Inverte normalize+affine sulla predizione di media (μ).
    # EN: Inverts normalize+affine on the mean prediction (μ).
    def denormalize_mu(self, mu: torch.Tensor, stats) -> torch.Tensor:
        """
        Denormalizza una predizione di media verso lo spazio originale.
        mu: (B,) scalare o (B, Q) per quantile_preds. Stats = (mean_t, std_t).
        """
        mean_t, std_t = stats
        if self.affine:
            w = self.affine_weight[self.target_idx]
            b = self.affine_bias[self.target_idx]
            mu = (mu - b) / (w + self.eps)
        if mu.dim() == 1:
            return mu * std_t + mean_t
        return mu * std_t.unsqueeze(-1) + mean_t.unsqueeze(-1)

    # IT: Inverte normalize sul log-var: var scala con std_t^2 → +2·log(std_t).
    # EN: Inverts normalize on log-var: var scales with std_t^2 → +2·log(std_t).
    def denormalize_log_var(self, log_var: torch.Tensor, stats) -> torch.Tensor:
        """
        Denormalizza log(sigma^2) verso lo spazio originale.
        Variance scala con std_t^2 → log_var_orig = log_var_norm + 2*log(std_t).
        Affine: dividi per w^2 (logaritmicamente: -2*log|w|).
        """
        _, std_t = stats
        out = log_var + 2.0 * torch.log(std_t + self.eps)
        if self.affine:
            w_abs = torch.abs(self.affine_weight[self.target_idx]) + self.eps
            out = out - 2.0 * torch.log(w_abs)
        return out
