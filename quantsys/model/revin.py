"""
RevIN — Reversible Instance Normalization for time series.

Kim et al. ICLR 2022, "Reversible Instance Normalization for Accurate
Time-Series Forecasting against Distribution Shift".

Mitigates local non-stationarity: each window (B, T, F) is normalized with
its own mean/std (computed along the T axis), passed to the model, and the
predictions are denormalized using the stats of the same instance.

For crypto/finance, where the regime changes on hour-to-day scales, RevIN
lets the model learn patterns invariant to local mean/vol, reducing the
train→test distribution shift.

Standard reference also used in iTransformer (Liu et al. 2024) and
PatchTST (Nie et al. 2023).
"""
import torch
import torch.nn as nn


# Reversible Instance Normalization: per-instance normalize/denormalize with affine.
class RevIN(nn.Module):
    """
    Reversible Instance Normalization with learnable affine.

    Usage:
        revin = RevIN(n_features=119, target_idx=0)
        x_norm, stats = revin.normalize(x)                 # (B, T, F)
        out = model(x_norm)                                # model in norm space
        mu_orig    = revin.denormalize_mu(out.mu, stats)   # (B,) or (B, Q)
        logvar_orig = revin.denormalize_log_var(out.lv, stats)

    The idea:
      * `target_idx` is the feature column that best matches the target
        (usually `log_ret`). Its per-instance statistics are used to
        denormalize the scalar prediction.
      * The affine (gamma, beta) is learnable per-feature and applied after
        normalization: it lets the model rescale each channel.
        The inverse is applied in `denormalize_*` only for `target_idx`.

    NOTE: denormalization assumes the target is on the same scale as the
    `target_idx` column. If `log_ret` sits at a different index in your
    dataset, configure it via `model.revin_target_idx` in config/default.yaml.
    """

    # Initializes learnable per-feature affine parameters (if affine=True).
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

    # Per-instance normalization on T axis; learnable per-feature affine.
    def normalize(self, x: torch.Tensor):
        """
        x: (B, T, F) → x_norm: (B, T, F), stats: (mean_t, std_t) of shape (B,).

        mean/std computed along the time axis (T), per-instance, per-feature.
        They are detached: gradients do not flow through the stats
        (the affine has its own learnable parameters).
        """
        # detach: gradients do NOT flow through mean/std (only affine).
        mean = x.mean(dim=1, keepdim=True).detach()
        var = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps).detach()

        x_norm = (x - mean) / std
        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        # Extracts target-column stats to denormalize the predictions.
        mean_t = mean[:, 0, self.target_idx]
        std_t = std[:, 0, self.target_idx]
        return x_norm, (mean_t, std_t)

    # Inverts normalize+affine on the mean prediction (μ).
    def denormalize_mu(self, mu: torch.Tensor, stats) -> torch.Tensor:
        """
        Denormalizes a mean prediction back to the original space.
        mu: (B,) scalar or (B, Q) for quantile_preds. Stats = (mean_t, std_t).
        """
        mean_t, std_t = stats
        if self.affine:
            w = self.affine_weight[self.target_idx]
            b = self.affine_bias[self.target_idx]
            mu = (mu - b) / (w + self.eps)
        if mu.dim() == 1:
            return mu * std_t + mean_t
        return mu * std_t.unsqueeze(-1) + mean_t.unsqueeze(-1)

    # Inverts normalize on log-var: var scales with std_t^2 → +2·log(std_t).
    def denormalize_log_var(self, log_var: torch.Tensor, stats) -> torch.Tensor:
        """
        Denormalizes log(sigma^2) back to the original space.
        Variance scales with std_t^2 → log_var_orig = log_var_norm + 2*log(std_t).
        Affine: divide by w^2 (in log terms: -2*log|w|).
        """
        _, std_t = stats
        out = log_var + 2.0 * torch.log(std_t + self.eps)
        if self.affine:
            w_abs = torch.abs(self.affine_weight[self.target_idx]) + self.eps
            out = out - 2.0 * torch.log(w_abs)
        return out
