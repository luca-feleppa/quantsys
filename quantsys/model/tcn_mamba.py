"""TCN+Mamba hybrid architecture for QUANTSYS.

Combines a Temporal Convolutional Network (TCN) branch with a simplified
SSM/Mamba branch, fused via a learned gate.

Output:
  * loss_type="t_student": (mu, log_sigma_sq, log_nu)
  * loss_type="quantile":  (quantile_preds,) — (B, Q) con Q=5 quantili standard
  * MoE: se n_output_experts > 1, gate softmax + experts (come QuantiTransformer)

2026-05-15: refactor per supportare quantile+MoE (prima hardcoded a t_student).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from quantsys.model import DropPath, QUANTILES


# ─── TCN Branch ──────────────────────────────────────────────────────────────

# IT: Conv1d con padding solo a sinistra → causalità temporale.
# EN: Conv1d with left-only padding → temporal causality.
class CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) padding.

    Applies (kernel_size - 1) * dilation padding to the left so the output at
    position t depends only on positions ≤ t.  The right padding that PyTorch
    would add is stripped to keep the output length equal to the input length.
    """
    # IT: Conv1d con padding sinistro pre-calcolato per garantire la causalità.
    # EN: Conv1d with pre-computed left padding to enforce causality.
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0, **kwargs,
        )

    # IT: Pad solo a sinistra → output al tempo t dipende solo da t' ≤ t.
    # EN: Left-only padding → output at time t depends only on t' ≤ t.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


# IT: Blocco TCN residuale con causal conv + GELU + DropPath.
# EN: Residual TCN block with causal conv + GELU + DropPath.
class TCNBlock(nn.Module):
    """Dilated causal residual block.

    Input/output shape: (B, T, d_model).  Internally transposes to (B, C, T)
    for the convolution, then transposes back.
    """
    # IT: Costruisce causal conv + norm + GELU + dropout + DropPath del blocco residuale.
    # EN: Builds the residual block's causal conv + norm + GELU + dropout + DropPath.
    def __init__(self, d_model: int, kernel_size: int = 3, dilation: int = 1,
                 dropout: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.conv  = CausalConv1d(d_model, d_model, kernel_size, dilation)
        self.norm  = nn.LayerNorm(d_model)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.skip  = nn.Identity()

    # IT: Transpose per conv1d su asse T, poi back. Residual + DropPath.
    # EN: Transpose for conv1d over T axis, then back. Residual + DropPath.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = x.transpose(1, 2)                  # (B, d_model, T)
        h = self.conv(h).transpose(1, 2)        # (B, T, d_model)
        h = self.act(self.norm(h))
        h = self.drop(h)
        return self.drop_path(h) + residual


# IT: TCN branch: dilations [1,2,4,8] → receptive field 31 step (≈30 min).
# EN: TCN branch: dilations [1,2,4,8] → receptive field 31 steps (~30 min).
class TCNBranch(nn.Module):
    """Projects features to d_model, then applies 4 dilated TCN blocks.

    Dilations [1, 2, 4, 8] give a receptive field of
    1 + (3-1)*(1+2+4+8) = 31 steps — enough to cover 30-min patterns in T=120.
    Output is the global average over time: (B, d_model).
    """
    # IT: Costruisce input proj + i blocchi TCN dilatati con DropPath crescente.
    # EN: Builds the input projection + the dilated TCN blocks with increasing DropPath.
    def __init__(self, n_features: int, d_model: int, tcn_layers: int = 4,
                 kernel_size: int = 3, dropout: float = 0.1, drop_path_rate: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        # IT: Dilation esponenziali [1,2,4,..] + DropPath crescente per blocco.
        # EN: Exponential dilations [1,2,4,..] + increasing DropPath per block.
        dilations = [2 ** i for i in range(tcn_layers)]
        dp_rates = [drop_path_rate * i / max(tcn_layers - 1, 1) for i in range(tcn_layers)]
        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size, d, dropout, dp) for d, dp in zip(dilations, dp_rates)
        ])

    # IT: Proietta a d_model, applica i blocchi TCN, pool medio su T.
    # EN: Projects to d_model, applies TCN blocks, mean-pools over T.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                        # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        return h.mean(dim=1)                    # IT: global avg pool su T | EN: global avg pool over T


# ─── Mamba Branch ────────────────────────────────────────────────────────────

# IT: Mamba SSM semplificato pure-PyTorch (A diagonale → scan vettorizzato).
# EN: Pure-PyTorch simplified Mamba SSM (diagonal A → vectorized scan).
class SimplifiedMambaBlock(nn.Module):
    """Pure-PyTorch simplified SSM (no mamba-ssm package required).

    Implements the Mamba selective-state-space pattern with a diagonal A matrix,
    which allows the recurrence scan to be vectorised via cumulative products
    within chunks instead of a sequential per-step loop.

    Vectorised diagonal scan derivation
    ------------------------------------
    With a diagonal A the recurrence is element-wise:

        h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊙ x_t

    Closed-form within a chunk of length L (with h_0 = carry from previous
    chunk):

        P_t = ∏_{r=1..t} A_bar_r                      # cumprod, length L
        h_t = P_t ⊙ (carry + Σ_{s=1..t} u_s / P_s)
            = P_t ⊙ (carry + cumsum(u / P)[t])

    Each chunk is therefore computed with **two vectorised PyTorch ops**
    (cumprod + cumsum) instead of L sequential kernel launches. Chunk size
    is kept small (16) so that ``P_t`` does not underflow in FP32 even for
    fast-decaying recurrences (A_bar close to 0). See
    ``_parallel_scan_chunk`` for the implementation.
    """

    # IT: Costruisce le proiezioni SSM (in/B/C/dt), conv depthwise e parametro log_A del blocco Mamba.
    # EN: Builds the Mamba block's SSM projections (in/B/C/dt), depthwise conv and log_A parameter.
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state

        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)

        # IT: Depthwise causal conv (k=4, pad=3 a sinistra).
        # EN: Depthwise causal conv (k=4, left pad=3).
        self.dw_conv  = nn.Conv1d(d_inner, d_inner, kernel_size=4,
                                  groups=d_inner, padding=0, bias=True)

        # IT: log_A=0 → A=1 iniziale; converge a A<1 durante il training.
        # EN: log_A=0 → initial A=1; converges to A<1 during training.
        self.log_A = nn.Parameter(torch.zeros(d_inner, d_state))
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        # IT: dt input-dependent: time-step per canale | EN: input-dependent per-channel dt
        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

        self._init_weights()

    # IT: dt bias t.c. softplus(bias) ≈ 0.01 → step iniziali piccoli.
    # EN: dt bias s.t. softplus(bias) ≈ 0.01 → small initial steps.
    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.constant_(self.dt_proj.bias, math.log(math.expm1(0.01)))

    @staticmethod
    def _parallel_scan_chunk(A_bar_chunk: torch.Tensor,
                              u_chunk:     torch.Tensor,
                              carry:       torch.Tensor) -> tuple:
        """Vectorised within-chunk associative scan (Fix #1, 2026-05-12).

        Solves the recurrence h_t = A_bar_t ⊙ h_{t-1} + u_t  with h_0 = carry
        in closed form:

            P_t = ∏_{r=1..L} A_bar_r              # cumprod within chunk
            h_t = P_t ⊙ (carry + Σ_{s=1..t} u_s / P_s)
                = P_t ⊙ (carry + cumsum(u / P))

        Replaces a Python ``for`` loop over time with two fully-vectorised
        tensor ops (cumprod + cumsum). Chunk size kept small (≤16) to avoid
        FP32 underflow of P_t for tightly-decaying recurrences.

        All math is performed in float32 for numerical safety even when the
        surrounding model runs under AMP autocast (FP16). Result is cast back
        to ``u_chunk.dtype`` on return.
        """
        # IT: Promuove a FP32: cumprod/cumsum sensibili a underflow in FP16.
        # EN: Promotes to FP32: cumprod/cumsum sensitive to FP16 underflow.
        orig_dtype = u_chunk.dtype
        A_f = A_bar_chunk.float()
        u_f = u_chunk.float()
        c_f = carry.float()

        # IT: h_t = P_t·(carry + cumsum(u/P)_t) — closed-form vettorizzato.
        # EN: h_t = P_t·(carry + cumsum(u/P)_t) — vectorized closed form.
        P     = A_f.cumprod(dim=1)                # (B, L, d_inner, d_state)
        P_inv = 1.0 / P.clamp(min=1e-20)
        v     = u_f * P_inv                       # u / P
        S     = v.cumsum(dim=1)
        h     = P * (c_f.unsqueeze(1) + S)        # (B, L, d_inner, d_state)

        new_carry = h[:, -1]
        return h.to(orig_dtype), new_carry.to(orig_dtype)

    # IT: Mamba forward: SSM selettivo con gating + parallel scan a chunk.
    # EN: Mamba forward: selective SSM with gating + chunked parallel scan.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        residual = x
        x = self.norm(x)

        # IT: in_proj split in (xb, z): z è il gating SiLU.
        # EN: in_proj split into (xb, z): z is the SiLU gating.
        xz     = self.in_proj(x)
        xb, z  = xz.chunk(2, dim=-1)

        # IT: Causal depthwise conv (pad=3 a sinistra) | EN: Causal depthwise conv (left pad=3)
        xb_t   = xb.transpose(1, 2)
        xb_t   = F.pad(xb_t, (3, 0))
        xb_t   = self.dw_conv(xb_t)
        xb     = F.silu(xb_t.transpose(1, 2))

        # IT: B/C/dt input-dependent (selettività Mamba).
        # EN: Input-dependent B/C/dt (Mamba selectivity).
        B_ssm  = self.B_proj(xb)                          # (B, T, d_state)
        C_ssm  = self.C_proj(xb)                          # (B, T, d_state)
        dt     = F.softplus(self.dt_proj(xb)).clamp(min=0.001, max=1.0)  # (B, T, d_inner)

        # IT: Discretizzazione ZOH: A_bar = exp(dt·A), A negative-definite.
        # EN: ZOH discretization: A_bar = exp(dt·A), A negative-definite.
        A      = -torch.exp(self.log_A)                   # (d_inner, d_state)
        A_bar  = torch.exp(dt.unsqueeze(-1) * A)          # (B, T, d_inner, d_state)

        # IT: B_bar = dt·B, u = B_bar·x | EN: B_bar = dt·B, u = B_bar·x
        B_bar  = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)    # (B, T, d_inner, d_state)
        u      = B_bar * xb.unsqueeze(-1)                 # (B, T, d_inner, d_state)

        # IT: Chunk=32 (mai modificare: il modello è trained con questo valore;
        #     cambiare l'ordine di cumprod altera l'accumulo numerico FP32).
        # EN: Chunk=32 (never change: model is trained with this value; changing
        #     cumprod ordering alters FP32 numerical accumulation).
        # IT: h pre-allocato + write in-place evita torch.cat finale.
        # EN: Pre-allocated h + in-place write avoids final torch.cat.
        chunk_size = 32
        n_chunks = (T + chunk_size - 1) // chunk_size
        carry = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=u.dtype)
        h     = torch.empty(B, T, self.d_inner, self.d_state, device=x.device, dtype=u.dtype)

        for c in range(n_chunks):
            t0 = c * chunk_size
            t1 = min(t0 + chunk_size, T)
            h_chunk, carry = self._parallel_scan_chunk(
                A_bar[:, t0:t1], u[:, t0:t1], carry
            )
            h[:, t0:t1] = h_chunk

        # IT: Output y_t = Σ_n C_t[n]·h_t[i,n] | EN: Output y_t = Σ_n C_t[n]·h_t[i,n]
        y = (C_ssm.unsqueeze(2) * h).sum(-1)                 # (B, T, d_inner)

        # IT: Gating SiLU(z) | EN: SiLU(z) gating
        y = y * F.silu(z)

        out = self.drop(self.out_proj(y))                    # (B, T, d_model)
        return out + residual


# IT: Mamba branch: stack di SimplifiedMambaBlock; output = last token.
# EN: Mamba branch: stack of SimplifiedMambaBlock; output = last token.
class MambaBranch(nn.Module):
    """Projects features, applies a stack of SimplifiedMambaBlocks, takes last token.

    Output: (B, d_model).
    """
    # IT: Costruisce input proj + lo stack di SimplifiedMambaBlock.
    # EN: Builds the input projection + the stack of SimplifiedMambaBlocks.
    def __init__(self, n_features: int, d_model: int, mamba_layers: int = 3,
                 d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj   = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            SimplifiedMambaBlock(d_model, d_state, expand, dropout)
            for _ in range(mamba_layers)
        ])

    # IT: Proietta a d_model, applica i blocchi Mamba, prende l'ultimo token.
    # EN: Projects to d_model, applies Mamba blocks, takes the last token.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)                        # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        return h[:, -1, :]                      # IT: last-token pooling | EN: last-token pooling


# ─── Gated Fusion ────────────────────────────────────────────────────────────

# IT: Gated fusion: σ(W·[tcn;mamba]) sceglie per-canale tcn vs mamba.
# EN: Gated fusion: σ(W·[tcn;mamba]) selects per-channel tcn vs mamba.
class GatedFusion(nn.Module):
    """Learns a soft gate to blend TCN and Mamba representations.

    gate = σ(W · [tcn; mamba])
    fused = gate ⊙ tcn + (1 − gate) ⊙ mamba
    """
    # IT: Costruisce la proiezione del gate + LayerNorm per la fusione.
    # EN: Builds the gate projection + LayerNorm for the fusion.
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.norm      = nn.LayerNorm(d_model)

    # IT: Concatena le due rappresentazioni, calcola il gate, blenda + norm.
    # EN: Concatenates both representations, computes the gate, blends + norm.
    def forward(self, tcn_out: torch.Tensor, mamba_out: torch.Tensor) -> torch.Tensor:
        # both: (B, d_model)
        combined = torch.cat([tcn_out, mamba_out], dim=-1)  # (B, d_model*2)
        gate     = torch.sigmoid(self.gate_proj(combined))   # (B, d_model)
        fused    = gate * tcn_out + (1.0 - gate) * mamba_out
        return self.norm(fused)                              # (B, d_model)


# ─── QuantTCNMamba ────────────────────────────────────────────────────────────

# IT: Architettura ibrida TCN+Mamba per forecasting probabilistico BTC/USDT 1m.
# EN: Hybrid TCN+Mamba architecture for probabilistic BTC/USDT 1-min forecasting.
class QuantTCNMamba(nn.Module):
    """Hybrid TCN + Mamba architecture for BTC/USDT 1-min probabilistic forecasting.

    Input  : x of shape (B, T, F) — T=120 candles, F=116 features.
    Output : (mu, log_sigma_sq, log_nu) — t-Student distribution parameters,
             compatible with student_t_nll() in quantsys/model/__init__.py.

    If n_dynamic_features is set, x is split into dynamic (0..n_dyn-1) and
    structural (n_dyn..) features; both are concatenated back before the two
    branches so each branch sees the full feature set with semantic ordering
    preserved.  This mirrors QuantLSTM's dual-stream handling.

    If use_multitask=True, also returns dir_logits (B, 3) as the 4th element
    for directional classification (up / flat / down).
    """

    def __init__(
        self,
        n_features:         int,
        d_model:            int   = 128,
        tcn_layers:         int   = 4,
        tcn_kernel:         int   = 3,
        mamba_layers:       int   = 3,
        mamba_d_state:      int   = 16,
        mamba_expand:       int   = 2,
        dropout:            float = 0.1,
        drop_path_rate:     float = 0.0,
        n_dynamic_features: int   = None,
        use_multitask:      bool  = False,
        loss_type:          str   = "t_student",
        n_output_experts:   int   = 1,
        use_revin:          bool  = False,
        revin_target_idx:   int   = 0,
    ):
        # IT: Costruisce branch TCN+Mamba, fusione gated e teste di output.
        # EN: Builds TCN+Mamba branches, gated fusion and output heads.
        super().__init__()
        self.n_dynamic        = n_dynamic_features
        self.use_multitask    = use_multitask
        self.loss_type        = loss_type
        self.n_output_experts = max(1, n_output_experts)
        self.use_revin        = use_revin
        if use_revin:
            from quantsys.model.revin import RevIN
            self.revin = RevIN(n_features=n_features, target_idx=revin_target_idx)

        self.tcn_branch   = TCNBranch(n_features, d_model, tcn_layers, tcn_kernel,
                                       dropout, drop_path_rate)
        self.mamba_branch = MambaBranch(n_features, d_model, mamba_layers,
                                        mamba_d_state, mamba_expand, dropout)
        self.fusion       = GatedFusion(d_model)
        self.head_drop    = nn.Dropout(dropout)

        # ── Output heads ─────────────────────────────────────────────────────
        # IT: Stesso pattern di QuantLSTM/QuantiTransformer: MoE → expert_gate+heads,
        #     single → quantile_head O (mu/ls2/lnu) in base a loss_type.
        # EN: Same pattern as QuantLSTM/QuantiTransformer: MoE → expert_gate+heads,
        #     single → quantile_head OR (mu/ls2/lnu) depending on loss_type.
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
                self.mu_head  = nn.Linear(d_model, 1)
                self.ls2_head = nn.Linear(d_model, 1)
                self.lnu_head = nn.Linear(d_model, 1)

        if use_multitask:
            self.dir_head = nn.Linear(d_model, 3)
            nn.init.normal_(self.dir_head.weight, std=0.01)
            nn.init.zeros_(self.dir_head.bias)

        self._init_output_heads()
        # IT: Spectral norm: solo nel ramo t_student single-expert (legacy);
        #     MoE/quantile heads non parametrizzate per non interferire col gating.
        # EN: Spectral norm: only in single-expert t_student branch (legacy);
        #     MoE/quantile heads left unparametrized to avoid gating interference.
        if self.n_output_experts <= 1 and loss_type == "t_student":
            from quantsys.model import _QS_SN_ON_MU_ONLY
            self.mu_head  = spectral_norm(self.mu_head)
            if not _QS_SN_ON_MU_ONLY:
                self.ls2_head = spectral_norm(self.ls2_head)
                self.lnu_head = spectral_norm(self.lnu_head)

    # IT: Init teste mu/ls2/lnu (solo ramo t_student single-expert).
    # EN: Inits mu/ls2/lnu heads (single-expert t_student branch only).
    def _init_output_heads(self):
        # IT: Gli altri rami fanno init inline (quantile_head) o non hanno teste (MoE).
        # EN: Other branches init inline (quantile_head) or have no heads (MoE).
        if self.n_output_experts > 1 or self.loss_type == "quantile":
            return
        for head in (self.mu_head, self.ls2_head, self.lnu_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
        # IT: softplus^-1(3)=log(e^3-1) → softplus(bias)+2 ≈ 5 gradi di libertà.
        # EN: softplus^-1(3)=log(e^3-1) → softplus(bias)+2 ≈ 5 degrees of freedom.
        with torch.no_grad():
            self.lnu_head.bias.fill_(math.log(math.expm1(3.0)))

    # IT: No-op: questa arch non usa clip bounds (solo compat train loop).
    # EN: No-op: this arch does not use clip bounds (train-loop compat only).
    def set_clip_bounds(self, lo, hi) -> None:
        """No-op — clip bounds are not used in this architecture.

        Exists for compatibility with the training loop which calls
        set_clip_bounds() on every architecture after computing percentiles
        from X_train.
        """

    # IT: Forward: TCN + Mamba in parallelo, fusi via gating, poi heads.
    # EN: Forward: TCN + Mamba in parallel, fused via gating, then heads.
    def forward(self, x: torch.Tensor, x_macro=None,
                latent: torch.Tensor = None) -> tuple:
        # IT: x_macro accettato per API uniforme ma ignorato (no macro qui).
        #     latent = latente CAFN OPZIONALE (B,T,d_latent) concatenato sull'asse
        #     feature. latent=None → path identico (parity). Costruire il modulo
        #     con n_features+=d_latent se usato.
        # EN: x_macro accepted for API uniformity but ignored (no macro here).
        #     latent = OPTIONAL CAFN latent (B,T,d_latent) concatenated on the
        #     feature axis. latent=None → identical path (parity). Build the module
        #     with n_features+=d_latent if used.
        if latent is not None:
            x = torch.cat([x, latent], dim=-1)
        _revin_stats = None
        if self.use_revin:
            x, _revin_stats = self.revin.normalize(x)
        x_in = x

        tcn_out   = self.tcn_branch(x_in)                              # (B, d_model)
        mamba_out = self.mamba_branch(x_in)                            # (B, d_model)
        fused     = self.head_drop(self.fusion(tcn_out, mamba_out))    # (B, d_model)

        # IT: Output computation (MoE softmax gate o single head).
        # EN: Output computation (MoE softmax gate or single head).
        if self.n_output_experts > 1:
            gate_w      = F.softmax(self.expert_gate(fused), dim=-1)   # (B, E)
            expert_outs = torch.stack(
                [eh(fused) for eh in self.expert_heads], dim=1
            )                                                           # (B, E, out)
            h_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)    # (B, out)

            if self.loss_type == "quantile":
                quantile_preds = h_out                                  # (B, Q)
                if self.use_revin:
                    quantile_preds = self.revin.denormalize_mu(quantile_preds, _revin_stats)
                if self.use_multitask:
                    return (quantile_preds, self.dir_head(fused))
                return (quantile_preds,)
            else:
                mu       = h_out[:, 0]
                log_sig2 = h_out[:, 1]
                log_nu   = h_out[:, 2]
                if self.use_revin:
                    mu       = self.revin.denormalize_mu(mu, _revin_stats)
                    log_sig2 = self.revin.denormalize_log_var(log_sig2, _revin_stats)
                if self.use_multitask:
                    return (mu, log_sig2, log_nu, self.dir_head(fused))
                return (mu, log_sig2, log_nu)

        elif self.loss_type == "quantile":
            quantile_preds = self.quantile_head(fused)                  # (B, Q)
            if self.use_revin:
                quantile_preds = self.revin.denormalize_mu(quantile_preds, _revin_stats)
            if self.use_multitask:
                return (quantile_preds, self.dir_head(fused))
            return (quantile_preds,)

        else:
            mu  = self.mu_head(fused).squeeze(-1)
            ls2 = self.ls2_head(fused).squeeze(-1)
            lnu = self.lnu_head(fused).squeeze(-1)
            if self.use_revin:
                mu  = self.revin.denormalize_mu(mu, _revin_stats)
                ls2 = self.revin.denormalize_log_var(ls2, _revin_stats)
            if self.use_multitask:
                return mu, ls2, lnu, self.dir_head(fused)
            return mu, ls2, lnu
