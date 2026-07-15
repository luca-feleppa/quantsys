"""QuantNHiTS — Neural Hierarchical Interpolation for Time Series.

Architettura pure-MLP a stack gerarchici (Challu et al. 2022, "N-HiTS: Neural
Hierarchical Interpolation for Time Series").  Adattata al contratto QUANTSYS:
input multivariato (B, T=120, F=116), output (mu, log_sigma2, log_nu) per
t-Student NLL.

Differenze rispetto al paper originale:
  * Multivariato: input proj Linear(F, d_model) prima degli stack.
  * Forecast latente (B, d_model) invece di forecast esplicito su horizon —
    serviamo QUANTSYS, che vuole un singolo step probabilistico, non H step.
  * Output configurabile via loss_type: "t_student" (mu, ls2, lnu) o "quantile"
    (B, Q=5 quantili). MoE supportato via n_output_experts (refactor 2026-05-15).
  * Heads (mu, ls2, lnu) con spectral_norm di default (legacy). Opt-in SN-solo-mu
    via config training.sn_on_mu_only=true (anti-overfit 2026-05-15).
  * Macro embedding additivo opzionale.
  * Dual-stream parameter accettato per compat ma ignorato (pure-MLP non
    differenzia dynamic vs structural — il gradient impara da solo).

Compatibile con EnsembleModel, distillation pipeline e train loop esistenti.
"""
from __future__ import annotations
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from quantsys.model import QUANTILES

log = logging.getLogger("quantsys.model.nhits")


# IT: Blocco N-HiTS a una risoluzione fissa (pool_kernel = scala temporale).
# EN: Single N-HiTS block at a fixed resolution (pool_kernel = temporal scale).
class NHiTSBlock(nn.Module):
    """Single N-HiTS block at a fixed temporal resolution.

    Pipeline:
      x (B, T, D)
        ─► Avg/MaxPool1d(k) along T          → (B, T_p, D)   (downsample)
        ─► flatten + MLP(hidden)             → (B, hidden)
        ─► linear → backcast (B, T*D)        (subtracted from residual)
        ─► linear → forecast latent (B, D)   (summed across stacks)

    Pooling kernel k controlla la risoluzione: k grande = pattern a lungo
    termine (trend), k=1 = pattern a brevissimo termine.
    pool_type: "avg" (default, passa-basso — design storico bit-invariato) o
    "max" (A9: preserva gli spike → componente jump della RV).
    """

    # IT: Costruisce pool + MLP + teste backcast/forecast per un blocco a risoluzione fissa.
    # EN: Builds the pool + MLP + backcast/forecast heads for a single fixed-resolution block.
    def __init__(
        self,
        input_len:   int,
        d_model:     int,
        hidden:      int,
        pool_kernel: int,
        n_layers:    int   = 2,
        dropout:     float = 0.1,
        pool_type:   str   = "avg",
    ):
        super().__init__()
        self.input_len   = input_len
        self.d_model     = d_model
        self.pool_kernel = max(1, pool_kernel)

        # IT: A9 — "avg" (passa-basso, default = bit-identico al design storico) o
        #     "max" (preserva gli spike: sensore della componente jump). Fail-fast
        #     su valori ignoti (pattern MINOR-3: mai default silenziosi su typo).
        # EN: A9 — "avg" (low-pass, default = bit-identical to the historical design)
        #     or "max" (spike-preserving: jump-component sensor). Fail-fast on
        #     unknown values (MINOR-3 pattern: never silently default on typos).
        if pool_type not in ("avg", "max"):
            raise ValueError(f"pool_type '{pool_type}' non riconosciuto / unknown (avg|max)")
        self.pool_type = pool_type
        _pool_cls = nn.AvgPool1d if pool_type == "avg" else nn.MaxPool1d

        # IT: ceil_mode=True copre completamente T anche se k non divide T.
        # EN: ceil_mode=True covers T fully even when k does not divide T.
        self.pool = _pool_cls(self.pool_kernel,
                              stride=self.pool_kernel,
                              ceil_mode=True)
        pooled_len = (input_len + self.pool_kernel - 1) // self.pool_kernel
        self.pooled_len = pooled_len

        layers = []
        in_dim = pooled_len * d_model
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden
        self.mlp = nn.Sequential(*layers)

        # IT: backcast → ricostruisce input per residual decomp; forecast → latente.
        # EN: backcast → reconstructs input for residual decomp; forecast → latent.
        self.backcast_head = nn.Linear(hidden, input_len * d_model)
        self.forecast_head = nn.Linear(hidden, d_model)

        # IT: std=0.02 per residual stabili (paper).
        # EN: std=0.02 for stable residuals (paper).
        for m in (self.backcast_head, self.forecast_head):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    # IT: Pool su T → MLP → backcast (B,T,D) + forecast latente (B,D).
    # EN: Pool over T → MLP → backcast (B,T,D) + latent forecast (B,D).
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_td   = x.transpose(1, 2).contiguous()      # (B, D, T)
        x_pool = self.pool(x_td)                      # (B, D, T_p)
        x_flat = x_pool.transpose(1, 2).reshape(B, -1)  # (B, T_p * D)

        h        = self.mlp(x_flat)                          # (B, hidden)
        backcast = self.backcast_head(h).view(B, T, D)       # (B, T, D)
        forecast = self.forecast_head(h)                     # (B, D)
        return backcast, forecast


# IT: N-HiTS multivariato per forecasting probabilistico BTC/USDT 1m.
# EN: Multivariate N-HiTS for probabilistic BTC/USDT 1-min forecasting.
class QuantNHiTS(nn.Module):
    """N-HiTS adattato a forecasting probabilistico BTC/USDT 1m.

    Args:
        n_features: numero feature in input (F).
        T: lunghezza finestra temporale (default 120).
        n_dynamic_features: indice split dynamic/structural — ACCETTATO ma
            ignorato (N-HiTS è pure-MLP, non beneficia dello split semantico;
            il parametro esiste solo per compatibilità API con le altre arch).
        n_macro: dimensione vettore macro (0 = no macro).
        d_model: dimensione interna proiezione feature.
        hidden: hidden size MLP nei blocchi.
        n_stacks: numero stack (default 3).
        pool_kernels: kernel di pooling per ogni stack — di default
            (8, 4, 1): long → mid → short term decomposition.
        n_blocks_per_stack: blocchi MLP per stack (default 1, paper usa 1-2).
        dropout: dropout in MLP e prima delle teste.
        loss_type: "t_student" supportato (quantile non implementato per ora).
        use_multitask: aggiunge dir_head (B, 3) come 4° output.
        n_output_experts: MoE — non implementato in questa arch (sempre 1).
        use_max_pool_block: A9 — blocco MaxPool parallelo (sensore jump additivo
            sul forecast latente; backcast scartato). Default False = inerte.
        max_pool_kernel: kernel del blocco MaxPool parallelo (default 8).
    """

    # IT: Costruisce input proj, gli stack N-HiTS multi-risoluzione, macro embedding e teste di output.
    # EN: Builds input projection, the multi-resolution N-HiTS stacks, macro embedding and output heads.
    def __init__(
        self,
        n_features:         int,
        T:                  int   = 120,
        n_dynamic_features: int   = None,
        n_macro:            int   = 0,
        d_model:            int   = 128,
        hidden:             int   = 256,
        n_stacks:           int   = 3,
        pool_kernels:       tuple = (8, 4, 1),
        n_blocks_per_stack: int   = 1,
        n_mlp_layers:       int   = 2,
        dropout:            float = 0.1,
        loss_type:          str   = "t_student",
        use_multitask:      bool  = False,
        n_output_experts:   int   = 1,
        use_revin:          bool  = False,
        revin_target_idx:   int   = 0,
        use_max_pool_block: bool  = False,
        max_pool_kernel:    int   = 8,
    ):
        super().__init__()

        # IT: Allinea pool_kernels a n_stacks (pad con 1 = nessun pool).
        # EN: Aligns pool_kernels to n_stacks (pad with 1 = no pooling).
        if len(pool_kernels) != n_stacks:
            pool_kernels = (list(pool_kernels) + [1] * n_stacks)[:n_stacks]

        self.n_features       = n_features
        self.T                = T
        self.n_dynamic        = n_dynamic_features        # IT: solo metadata (no split semantico) | EN: metadata only
        self.n_macro          = n_macro
        self.d_model          = d_model
        self.loss_type        = loss_type
        self.use_multitask    = use_multitask
        self.n_output_experts = max(1, n_output_experts)
        self.use_revin        = use_revin
        if use_revin:
            from quantsys.model.revin import RevIN
            self.revin = RevIN(n_features=n_features, target_idx=revin_target_idx)

        # IT: Clip bounds adattivi (set via set_clip_bounds dopo training).
        # EN: Adaptive clip bounds (set via set_clip_bounds after training).
        self.register_buffer("clip_lo", torch.full((n_features,), -500.0))
        self.register_buffer("clip_hi", torch.full((n_features,), +500.0))

        # IT: F → d_model per-timestep | EN: F → d_model per timestep
        self.input_proj = nn.Linear(n_features, d_model)
        self.input_drop = nn.Dropout(dropout)

        # IT: Macro come bias additivo broadcast su T | EN: Macro as additive bias broadcast over T
        if n_macro > 0:
            self.macro_proj = nn.Linear(n_macro, d_model)
        else:
            self.macro_proj = None

        # IT: Stack gerarchico: kernel da grande (trend) a piccolo (short-term).
        # EN: Hierarchical stack: kernel from large (trend) to small (short-term).
        blocks = []
        for s in range(n_stacks):
            k = pool_kernels[s]
            for _ in range(n_blocks_per_stack):
                blocks.append(NHiTSBlock(
                    input_len   = T,
                    d_model     = d_model,
                    hidden      = hidden,
                    pool_kernel = k,
                    n_layers    = n_mlp_layers,
                    dropout     = dropout,
                ))
        self.blocks = nn.ModuleList(blocks)
        self.n_stacks = n_stacks
        self.pool_kernels = tuple(pool_kernels)

        # IT: A9 (roadmap vol) — blocco MaxPool PARALLELO: legge lo stesso input h
        #     (post proiezione+macro) e SOMMA il suo forecast latente; il backcast è
        #     scartato → la catena residuale AvgPool resta INVARIATA (sensore jump
        #     additivo, non partecipa alla decomposizione). Default False = lever
        #     INERTE: zero parametri nuovi, state_dict e forward bit-identici
        #     (checkpoint esistenti compatibili in entrambe le direzioni).
        # EN: A9 (vol roadmap) — PARALLEL MaxPool block: reads the same input h
        #     (post projection+macro) and ADDS its latent forecast; the backcast is
        #     discarded → the AvgPool residual chain stays UNCHANGED (additive jump
        #     sensor, does not join the decomposition). Default False = INERT lever:
        #     zero new parameters, bit-identical state_dict and forward
        #     (existing checkpoints compatible both ways).
        self.use_max_pool_block = bool(use_max_pool_block)
        if self.use_max_pool_block:
            self.jump_block = NHiTSBlock(
                input_len   = T,
                d_model     = d_model,
                hidden      = hidden,
                pool_kernel = max_pool_kernel,
                n_layers    = n_mlp_layers,
                dropout     = dropout,
                pool_type   = "max",
            )
        else:
            self.jump_block = None

        self.head_drop = nn.Dropout(dropout)

        # IT: Output heads — pattern allineato alle altre arch (MoE o single).
        # EN: Output heads — pattern aligned with other archs (MoE or single).
        if self.n_output_experts > 1:
            out_dim_per_expert = 3 if loss_type == "t_student" else len(QUANTILES)
            self.expert_gate  = nn.Linear(d_model, self.n_output_experts)
            self.expert_heads = nn.ModuleList([
                nn.Linear(d_model, out_dim_per_expert)
                for _ in range(self.n_output_experts)
            ])
        else:
            if loss_type == "quantile":
                self.quantile_head = nn.Linear(d_model, len(QUANTILES))
                nn.init.normal_(self.quantile_head.weight, std=0.01)
                nn.init.zeros_(self.quantile_head.bias)
            else:
                mu_head  = nn.Linear(d_model, 1)
                ls2_head = nn.Linear(d_model, 1)
                lnu_head = nn.Linear(d_model, 1)
                self._init_output_heads(mu_head, ls2_head, lnu_head)
                # IT: SN-on-mu-only (opt-in): riduce overfit lasciando σ/ν liberi.
                # EN: SN-on-mu-only (opt-in): reduces overfit, leaves σ/ν unconstrained.
                from quantsys.model import _QS_SN_ON_MU_ONLY
                self.mu_head  = spectral_norm(mu_head)
                if _QS_SN_ON_MU_ONLY:
                    self.ls2_head = ls2_head
                    self.lnu_head = lnu_head
                else:
                    self.ls2_head = spectral_norm(ls2_head)
                    self.lnu_head = spectral_norm(lnu_head)

        # IT: Testa direzionale multitask opzionale (up/flat/down).
        # EN: Optional multitask directional head (up/flat/down).
        if use_multitask:
            self.dir_head = nn.Linear(d_model, 3)
            nn.init.normal_(self.dir_head.weight, std=0.01)
            nn.init.zeros_(self.dir_head.bias)

        log.info(
            f"QuantNHiTS init: F={n_features} T={T} d_model={d_model} "
            f"hidden={hidden} stacks={n_stacks} kernels={self.pool_kernels} "
            f"max_pool_block={self.use_max_pool_block} "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    # IT: bias di ν tale che softplus(bias)+2 ≈ 5 (ν=5 = default ragionevole).
    # EN: ν bias such that softplus(bias)+2 ≈ 5 (ν=5 = reasonable default).
    @staticmethod
    def _init_output_heads(mu_h, ls2_h, lnu_h):
        for h in (mu_h, ls2_h, lnu_h):
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)
        with torch.no_grad():
            lnu_h.bias.fill_(math.log(math.expm1(3.0)))

    # IT: Imposta i clip bounds adattivi nei buffer (post-training).
    # EN: Sets the adaptive clip bounds into buffers (post-training).
    def set_clip_bounds(self, lo: torch.Tensor, hi: torch.Tensor) -> None:
        """Imposta clip bounds adattivi (chiamato dal train loop dopo
        calcolo percentili su X_train).
        """
        with torch.no_grad():
            self.clip_lo.copy_(torch.as_tensor(lo,
                dtype=self.clip_lo.dtype, device=self.clip_lo.device))
            self.clip_hi.copy_(torch.as_tensor(hi,
                dtype=self.clip_hi.dtype, device=self.clip_hi.device))

    # IT: Forward gerarchico: residual decomp, somma forecast latenti, output.
    # EN: Hierarchical forward: residual decomp, sum latent forecasts, output.
    def forward(self, x: torch.Tensor, x_macro: torch.Tensor = None,
                latent: torch.Tensor = None) -> tuple:
        """Forward.

        x:       (B, T, F)
        x_macro: (B, n_macro) opzionale
        latent:  (B, T, d_latent) latente CAFN OPZIONALE / OPTIONAL CAFN latent

        Returns:
            (mu, log_sigma2, log_nu) shape (B,) each
            o (..., dir_logits) se use_multitask.
        """
        # IT: latente CAFN concatenato sull'asse feature. latent=None → identico
        #     (parity). Costruire il modulo con n_features+=d_latent se usato.
        # EN: CAFN latent concatenated on the feature axis. latent=None → identical
        #     (parity). Build the module with n_features+=d_latent if used.
        if latent is not None:
            x = torch.cat([x, latent], dim=-1)
        _revin_stats = None
        if self.use_revin:
            x, _revin_stats = self.revin.normalize(x)
        x = x.clamp(self.clip_lo, self.clip_hi)

        h = self.input_proj(x)                  # (B, T, D)
        h = self.input_drop(h)

        if self.macro_proj is not None and x_macro is not None:
            m = self.macro_proj(x_macro)        # (B, D)
            h = h + m.unsqueeze(1)              # IT: broadcast su T | EN: broadcast over T

        # IT: Decomp residuale: ogni blocco sottrae il proprio backcast.
        # EN: Residual decomp: each block subtracts its own backcast.
        residual = h
        agg_forecast = None
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            agg_forecast = forecast if agg_forecast is None else agg_forecast + forecast

        # IT: A9 — ramo jump parallelo su h ORIGINALE (non sul residuo): il MaxPool
        #     vede gli spike prima che i blocchi AvgPool li sottraggano. Solo il
        #     forecast è usato; backcast scartato (catena residuale invariata).
        # EN: A9 — parallel jump branch on the ORIGINAL h (not the residual): the
        #     MaxPool sees spikes before the AvgPool blocks subtract them. Only the
        #     forecast is used; backcast discarded (residual chain unchanged).
        if self.jump_block is not None:
            _, jump_forecast = self.jump_block(h)
            agg_forecast = agg_forecast + jump_forecast

        feat = self.head_drop(agg_forecast)     # (B, D)

        # IT: Output computation (MoE → gate softmax; else single head).
        # EN: Output computation (MoE → softmax gate; else single head).
        if self.n_output_experts > 1:
            gate_w      = F.softmax(self.expert_gate(feat), dim=-1)    # (B, E)
            expert_outs = torch.stack(
                [eh(feat) for eh in self.expert_heads], dim=1
            )                                                           # (B, E, out)
            h_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)    # (B, out)

            if self.loss_type == "quantile":
                quantile_preds = h_out                                  # (B, Q)
                if self.use_revin:
                    quantile_preds = self.revin.denormalize_mu(quantile_preds, _revin_stats)
                if self.use_multitask:
                    return (quantile_preds, self.dir_head(feat))
                return (quantile_preds,)
            else:
                mu       = h_out[:, 0]
                log_sig2 = h_out[:, 1]
                log_nu   = h_out[:, 2]
                if self.use_revin:
                    mu       = self.revin.denormalize_mu(mu, _revin_stats)
                    log_sig2 = self.revin.denormalize_log_var(log_sig2, _revin_stats)
                if self.use_multitask:
                    return (mu, log_sig2, log_nu, self.dir_head(feat))
                return (mu, log_sig2, log_nu)

        elif self.loss_type == "quantile":
            quantile_preds = self.quantile_head(feat)                   # (B, Q)
            if self.use_revin:
                quantile_preds = self.revin.denormalize_mu(quantile_preds, _revin_stats)
            if self.use_multitask:
                return (quantile_preds, self.dir_head(feat))
            return (quantile_preds,)

        else:
            mu  = self.mu_head(feat).squeeze(-1)
            ls2 = self.ls2_head(feat).squeeze(-1)
            lnu = self.lnu_head(feat).squeeze(-1)
            if self.use_revin:
                mu  = self.revin.denormalize_mu(mu, _revin_stats)
                ls2 = self.revin.denormalize_log_var(ls2, _revin_stats)
            if self.use_multitask:
                return mu, ls2, lnu, self.dir_head(feat)
            return mu, ls2, lnu

    # IT: Inferenza single-pass: ritorna (mu, sigma, nu) in spazio naturale.
    # EN: Single-pass inference: returns (mu, sigma, nu) in natural space.
    @torch.no_grad()
    def predict(self, x: torch.Tensor, x_macro: torch.Tensor = None) -> dict:
        """Inferenza single-pass (no dropout). Ritorna dict con tensori
        in spazio naturale (sigma, nu) per consumo trading."""
        self.eval()
        out = self.forward(x, x_macro)
        if self.loss_type == "quantile":
            qp    = out[0]
            qp, _ = qp.sort(dim=-1)
            mu    = qp[:, 2]
            sigma = (qp[:, 3] - qp[:, 1]).clamp(min=1e-6)
            nu    = torch.full_like(mu, 10.0)
        else:
            mu, ls2, lnu = out[0], out[1], out[2]
            sigma = (F.softplus(ls2) + 1e-6).sqrt()
            nu    = F.softplus(lnu) + 2.0 + 1e-6
        return {"mu": mu, "sigma": sigma, "nu": nu}

    # IT: MC Dropout: K forward stocastici, σ include incertezza epistemica.
    # EN: MC Dropout: K stochastic forwards, σ includes epistemic uncertainty.
    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor,
                                  x_macro: torch.Tensor = None,
                                  n_samples: int = 20) -> dict:
        """MC Dropout: tiene il dropout attivo, accumula sample sul device,
        un solo trasferimento GPU→CPU alla fine.
        """
        was_training = self.training
        self.train()  # IT: attiva dropout | EN: enable dropout
        try:
            mus, sigs, nus = [], [], []
            for _ in range(n_samples):
                out = self.forward(x, x_macro)
                if self.loss_type == "quantile":
                    qp = out[0]
                    qp, _ = qp.sort(dim=-1)
                    mus.append(qp[:, 2])
                    sigs.append((qp[:, 3] - qp[:, 1]).clamp(min=1e-6))
                    nus.append(torch.full_like(qp[:, 2], 10.0))
                else:
                    mu, ls2, lnu = out[0], out[1], out[2]
                    mus.append(mu)
                    sigs.append((F.softplus(ls2) + 1e-6).sqrt())
                    nus.append(F.softplus(lnu) + 2.0 + 1e-6)
            mu_s    = torch.stack(mus, dim=0)
            sigma_s = torch.stack(sigs, dim=0)
            nu_s    = torch.stack(nus, dim=0)
            mu_mean    = mu_s.mean(0)
            # IT: σ_total² = E[σ²] (aleatoric) + Var[μ] (epistemic).
            # EN: σ_total² = E[σ²] (aleatoric) + Var[μ] (epistemic).
            sigma_mean = (sigma_s.pow(2).mean(0) + mu_s.var(0)).sqrt()
            nu_mean    = nu_s.mean(0)
            epi = mu_s.var(0)
            conf = 1.0 / (1.0 + epi)  # IT: confidence ∝ 1/Var | EN: confidence ∝ 1/Var
        finally:
            self.train(was_training)
        return {
            "mu":         mu_mean,
            "sigma":      sigma_mean,
            "nu":         nu_mean,
            "confidence": conf,
        }
