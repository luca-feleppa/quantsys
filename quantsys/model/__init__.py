"""Fase 3 — Modello: Dual-Stream LSTM → t-Student parametrica.

Miglioramento 9 — Architettura Dual-Stream:
  Invece di concatenare tutte le 55+ feature in un unico vettore,
  separiamo le feature in due stream semanticamente diversi:

  Stream A — Dinamico (log-return, volatilità, lag, indicatori momentum):
    Feature stazionarie che catturano la dinamica dei prezzi a breve termine.
    LSTM standard → bravo a modellare autocorrelazioni temporali.

  Stream B — Strutturale (VP POC/VAH/VAL, ATH distance, CVD, livelli prezzo):
    Feature che descrivono la struttura di liquidità e il contesto di mercato.
    Cambia lentamente → meglio modellata da un encoder MLP con residual.

  I due stream si fondono tramite un gate lineare (Variable Selection Network
  ispirato a TFT) prima della GRU: la rete impara quanto peso dare
  al contesto strutturale vs alla dinamica locale ad ogni passo.

  Vantaggio: il gradiente fluisce separatamente attraverso i due encoder,
  evitando che la volatilità a breve (segnale forte) soffochi il segnale
  strutturale (segnale debole ma importante per timing). +2-3% parametri.
"""
import math
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

log = logging.getLogger("quantsys.model")

# IT: Split feature: 0..N_DYN-1 = dinamiche, N_DYN.. = strutturali; None = single-stream.
# EN: Feature split: 0..N_DYN-1 = dynamic, N_DYN.. = structural; None = single-stream.
_DEFAULT_N_DYNAMIC = None

# IT: Se True spectral_norm solo su mu_head (anti-overfit). False = legacy SN su tutte le heads.
# EN: If True spectral_norm only on mu_head (anti-overfit). False = legacy SN on all heads.
_QS_SN_ON_MU_ONLY = False

# IT: Toggle globale: applica spectral_norm solo su mu_head (anti-overfit).
# EN: Global toggle: apply spectral_norm only on mu_head (anti-overfit).
def set_sn_on_mu_only(flag: bool) -> None:
    """Imposta globalmente se spectral_norm va applicato solo su mu_head."""
    global _QS_SN_ON_MU_ONLY
    _QS_SN_ON_MU_ONLY = bool(flag)

# IT: Quantili per pinball loss (10/25/50/75/90 percentile).
# EN: Quantiles for pinball loss (10/25/50/75/90 percentile).
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

# IT: A3 regime-MoE — numero di regimi del gate esterno (R0 Quiet / R1 Trending /
#     R2 Stress, colonne regime_prob_0/1/2 di data/regime_probs.parquet).
# EN: A3 regime-MoE — number of regimes of the external gate (R0 Quiet / R1 Trending /
#     R2 Stress, columns regime_prob_0/1/2 of data/regime_probs.parquet).
N_REGIMES = 3


# ─── Loss ────────────────────────────────────────────────────────────────────

# IT: CRPS differenziabile per t-Student via Monte Carlo (no CDF esplicito).
# EN: Differentiable Student-t CRPS via Monte Carlo (no explicit CDF).
def crps_t_student(y: torch.Tensor, mu: torch.Tensor,
                   sigma: torch.Tensor, nu: torch.Tensor,
                   n_mc: int = 20) -> torch.Tensor:
    """CRPS via Monte Carlo con rsample (reparameterization trick).

    Differenziabile, nessun CDF esplicito, nessun gammainc.
    Forza float32 anche sotto AMP autocast (lgamma instabile in float16).
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        y_f  = y.float()
        mu_f = mu.float()
        s_f  = sigma.float().clamp(min=1e-6)
        nu_f = nu.float().clamp(min=2.01, max=200.0)

        t_dist = torch.distributions.StudentT(df=nu_f, loc=mu_f, scale=s_f)
        samples = t_dist.rsample((n_mc,)).clamp(-1e4, 1e4)

        term1 = (samples - y_f.unsqueeze(0)).abs().mean(0)
        term2 = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs().mean(dim=(0, 1))

        crps = term1 - 0.5 * term2
        return torch.nan_to_num(crps, nan=0.0).mean()


# IT: NLL t-Student con penalità asimmetrica sui sign error + CRPS opzionale.
# EN: Student-t NLL with asymmetric sign-error penalty + optional CRPS.
def student_t_nll(y, mu, log_sigma2, log_nu,
                  asymmetry_alpha: float = 2.0,
                  large_move_threshold: float = 0.002,
                  crps_weight: float = 0.0,
                  sample_weights: torch.Tensor = None):
    """
    NLL della t-Student + penalità asimmetrica + CRPS ausiliario (opzionale).

    crps_weight=0.0  → solo NLL asimmetrica (default, comportamento precedente)
    crps_weight=0.1  → 90% NLL + 10% CRPS (calibrazione leggera)
    crps_weight=0.2  → 80% NLL + 20% CRPS (calibrazione più aggressiva)

    sample_weights: (B,) pesi per-sample proporzionali a |target|.
    Se None, tutti i campioni hanno peso uguale (comportamento precedente).
    """
    # IT: Force fp32: AMP fp16 satura softplus/log su valori estremi.
    # EN: Force fp32: AMP fp16 saturates softplus/log on extreme values.
    y_f   = y.float()
    mu_f  = mu.float()
    ls2_f = torch.nan_to_num(log_sigma2.float(), nan=0.0)
    lnu_f = torch.nan_to_num(log_nu.float(), nan=0.0)

    sigma2 = F.softplus(ls2_f) + 1e-6
    nu     = F.softplus(lnu_f) + 2.0 + 1e-6

    nll = (
        0.5 * torch.log(torch.pi * nu * sigma2)
        - torch.lgamma((nu + 1) / 2)
        + torch.lgamma(nu / 2)
        + ((nu + 1) / 2) * torch.log1p((y_f - mu_f) ** 2 / (nu * sigma2))
    )

    # IT: Penalità asimmetrica: amplifica errori di segno su movimenti grandi.
    # EN: Asymmetric penalty: amplify sign errors on large moves.
    wrong_sign      = (mu_f * y_f) < 0
    large_move      = y_f.abs() > large_move_threshold
    magnitude_ratio = (y_f.abs() / large_move_threshold).clamp(max=5.0)
    penalty = torch.where(
        wrong_sign & large_move,
        1.0 + asymmetry_alpha * magnitude_ratio,
        torch.ones_like(nll),
    )
    per_sample = nll * penalty
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
    return per_sample.mean() + crps_weight * crps_t_student(y, mu, sigma2 ** 0.5, nu)


# IT: Loss direzionale: penalizza segni errati pesando per |y|, premia direzione giusta.
# EN: Directional loss: penalizes wrong signs weighted by |y|, rewards correct direction.
def direction_value_loss(y: torch.Tensor, mu: torch.Tensor,
                         lambda_dv: float = 0.5) -> torch.Tensor:
    """Direction-Value Joint Loss: penalizza errori direzionali pesati per magnitudine.

    Quando sign(mu) != sign(y), la penalità è proporzionale a |y| (i grandi
    movimenti mancati costano di più). Quando la direzione è corretta, premia
    proporzionalmente a min(|mu|, |y|) (reward per confidenza calibrata).

    L = lambda * mean(weight * indicator) dove:
      - wrong direction: weight = |y| (penalità)
      - right direction: weight = -min(|mu|, |y|) (reward)
    """
    wrong_dir = (mu * y) < 0
    right_dir = (mu * y) > 0
    cost = torch.where(
        wrong_dir,
        y.abs(),
        torch.where(right_dir, -torch.min(mu.abs(), y.abs()), torch.zeros_like(y)),
    )
    return lambda_dv * cost.mean()


# IT: Cache tensori quantili per (quantiles, device, dtype) → evita re-alloc per batch.
# EN: Quantile-tensor cache keyed by (quantiles, device, dtype) → avoids per-batch realloc.
_quantile_tensor_cache: dict = {}

# IT: Pinball loss per quantile regression, con sample_weights opzionali.
# EN: Pinball loss for quantile regression, with optional sample_weights.
def quantile_loss(y: torch.Tensor, quantile_preds: torch.Tensor,
                  quantiles: list = None,
                  sample_weights: torch.Tensor = None) -> torch.Tensor:
    """Pinball loss per quantile regression.

    sample_weights: (B,) pesi per-sample proporzionali a |target|.
    Se None, tutti i campioni hanno peso uguale (comportamento precedente).
    """
    if quantiles is None:
        quantiles = QUANTILES
    cache_key = (tuple(quantiles), y.device, y.dtype)
    qs = _quantile_tensor_cache.get(cache_key)
    if qs is None:
        qs = torch.tensor(quantiles, dtype=y.dtype, device=y.device)
        _quantile_tensor_cache[cache_key] = qs
    errors = y.unsqueeze(1) - quantile_preds          # (B, Q)
    loss   = torch.where(errors >= 0, qs * errors, (qs - 1) * errors)
    if sample_weights is not None:
        # IT: Broadcast (B,) → (B, 1) sul quantile axis.
        # EN: Broadcast (B,) → (B, 1) over quantile axis.
        loss = loss * sample_weights.unsqueeze(1)
    return loss.mean()


# IT: Converte log_sigma2/log_nu grezzi in (sigma, nu) validi via softplus.
# EN: Converts raw log_sigma2/log_nu into valid (sigma, nu) via softplus.
def student_t_params(log_sigma2, log_nu):
    return (F.softplus(log_sigma2) + 1e-6) ** 0.5, F.softplus(log_nu) + 2.0 + 1e-6


# ─── Temporal Self-Attention ─────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    # IT: Multi-head self-attention temporale con Flash Attention; ritorna ultimo timestep.
    # EN: Temporal multi-head self-attention using Flash Attention; returns last timestep.
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model; self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

    # IT: Self-attention su tutta la sequenza, ritorna contesto dell'ultimo step.
    # EN: Self-attention over the full sequence, returns last-step context.
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # IT: Flash Attention: niente matrice T×T materializzata, fp16-stabile.
        # EN: Flash Attention: no materialised T×T matrix, fp16-stable.
        context = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        context = self.norm(x + self.out_proj(context))
        # IT: Attention weights non esposti da SDPA → placeholder None.
        # EN: Attention weights not exposed by SDPA → None placeholder.
        return context[:, -1, :], None


# ─── Structural Encoder ──────────────────────────────────────────────────────

class StructuralEncoder(nn.Module):
    """
    Encoder MLP per le feature strutturali (VP, ATH distance, CVD).
    Proietta ogni step in un embedding della stessa dimensione dello stream dinamico,
    così possono essere fusi via gating point-by-point lungo la sequenza.

    Architettura per step t: Linear(n_struct → lstm_hidden) → LayerNorm → SiLU
    Applicato identicamente a ogni step della sequenza (shared weights).
    """
    # IT: Costruisce MLP per-step + gate di fusione dinamico/strutturale.
    # EN: Builds per-step MLP + dynamic/structural fusion gate.
    def __init__(self, n_struct: int, lstm_hidden: int, dropout: float = 0.1):
        super().__init__()
        mid = max(32, lstm_hidden // 4)
        self.net = nn.Sequential(
            nn.Linear(n_struct, mid),
            nn.LayerNorm(mid),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
        )
        # IT: Gate apprende mix dinamico/strutturale per ogni step.
        # EN: Gate learns dynamic/structural mix at every step.
        self.gate = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    # IT: Fusione gated: combina stream dinamico e strutturale via gate appreso.
    # EN: Gated fusion: blends dynamic and structural streams via learned gate.
    def fuse(self, h_dyn: torch.Tensor, h_struct: torch.Tensor) -> torch.Tensor:
        """
        Gated fusion: gate = σ(W[h_dyn; h_struct])
        output = gate × h_struct + (1 - gate) × h_dyn

        All'inizio del training, i pesi del gate sono piccoli → gate ≈ 0.5,
        poi impara a bilanciare i due stream per ogni contesto.
        """
        combined = torch.cat([h_dyn, h_struct], dim=-1)
        gate     = self.gate(combined)
        return gate * h_struct + (1 - gate) * h_dyn

    # IT: Encoda lo stream strutturale step-by-step (pesi condivisi su T).
    # EN: Encodes the structural stream step-by-step (weights shared over T).
    def forward(self, x_struct: torch.Tensor) -> torch.Tensor:
        """x_struct: (B, T, n_struct) → (B, T, lstm_hidden)"""
        B, T, _ = x_struct.shape
        flat     = x_struct.reshape(B * T, -1)
        enc      = self.net(flat)
        return enc.reshape(B, T, -1)


# ─── A3 — Regime Mixture-of-Universes head (config-gated, iTransformer-only) ─

# IT: Inversa numericamente stabile di softplus: x = y + log(-expm1(-y)).
#     Serve a ri-codificare la varianza mixata (spazio naturale) nello spazio
#     raw ls2 che il downstream decodifica con softplus(ls2)+1e-6 — il contratto
#     forward (mu, ls2, lnu) resta così invariato.
# EN: Numerically stable softplus inverse: x = y + log(-expm1(-y)).
#     Re-encodes the mixed variance (natural space) into the raw ls2 space that
#     downstream decodes via softplus(ls2)+1e-6 — keeping the forward contract
#     (mu, ls2, lnu) unchanged.
def _softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp(min=1e-12)
    return y + torch.log(-torch.expm1(-y))


class RegimeMoEHead(nn.Module):
    """
    IT: Testa Mixture-of-Universes per-regime (design A3, memoria
        `mixture_of_universes_design` adattata alla linea vol): 3 teste lineari
        fisse (una per regime R0/R1/R2) mescolate da un gate ESTERNO CAUSALE
        g(t) = [regime_prob_0, regime_prob_1, regime_prob_2] (filtered
        probabilities di RegimeMarkovBTC — MAI apprese: proprietà anti-overfit
        chiave). Path t_student: legge della varianza totale (stessa di
        ensemble.py) — σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)², che INFLAZIONA
        σ quando il regime è ambiguo (calibrazione σ regime-condizionata =
        l'obiettivo A3). Path quantile: media pesata per livello
        (Vincentization) + re-sort monotono di sicurezza. lnu: media pesata dal
        gate nello spazio raw.
    EN: Per-regime Mixture-of-Universes head (A3 design): 3 fixed linear heads
        (one per regime R0/R1/R2) mixed by an EXTERNAL CAUSAL gate
        g(t) = filtered regime probabilities of RegimeMarkovBTC — NEVER learned
        (key anti-overfit property). t_student path: total variance law (same
        as ensemble.py) — σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)², which
        INFLATES σ when the regime is ambiguous (regime-conditional σ
        calibration = the A3 goal). Quantile path: per-level weighted average
        (Vincentization) + monotone safety re-sort. lnu: gate-weighted average
        in raw space.

    IT: Nessuna spectral_norm sulle teste (stesso precedente del MoE appreso
        n_output_experts); init piccola std=0.01, bias lnu calibrato a df≈5.
    EN: No spectral_norm on the heads (same precedent as the learned
        n_output_experts MoE); small init std=0.01, lnu bias calibrated to df≈5.
    """

    # IT: Costruisce le n_regimes teste lineari (out = 3 t-Student o len(QUANTILES)).
    # EN: Builds the n_regimes linear heads (out = 3 Student-t or len(QUANTILES)).
    def __init__(self, out_dim: int, loss_type: str = "t_student",
                 n_regimes: int = N_REGIMES):
        super().__init__()
        self.loss_type = loss_type
        self.n_regimes = int(n_regimes)
        head_out = 3 if loss_type == "t_student" else len(QUANTILES)
        self.heads = nn.ModuleList([
            nn.Linear(out_dim, head_out) for _ in range(self.n_regimes)
        ])
        for hd in self.heads:
            nn.init.normal_(hd.weight, std=0.01)
            nn.init.zeros_(hd.bias)
            if loss_type == "t_student":
                # IT: bias lnu così softplus(bias)+2 ≈ 5 (convenzione delle teste single).
                # EN: lnu bias so softplus(bias)+2 ≈ 5 (single-head convention).
                with torch.no_grad():
                    hd.bias[2] = math.log(5.0 - 2.0)

    # IT: Mescola le teste col gate: h (B,d), g (B,K) → stesso contratto della
    #     testa singola: (quantile_preds,) o (mu, ls2, lnu).
    # EN: Mixes the heads with the gate: h (B,d), g (B,K) → same contract as the
    #     single head: (quantile_preds,) or (mu, ls2, lnu).
    def forward(self, h: torch.Tensor, g: torch.Tensor) -> tuple:
        if g.shape[-1] != self.n_regimes:
            raise ValueError(
                f"RegimeMoEHead: gate con {g.shape[-1]} colonne, attese "
                f"{self.n_regimes} / gate has {g.shape[-1]} columns, expected "
                f"{self.n_regimes}"
            )
        # IT: rinormalizza il gate al simplesso (difesa contro drift float; no-op
        #     su gate validi, one-hot inclusi).
        # EN: renormalize the gate onto the simplex (defence against float drift;
        #     no-op on valid gates, one-hot included).
        g = g.clamp(min=0.0)
        g = g / g.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        outs = torch.stack([hd(h) for hd in self.heads], dim=1)   # (B, K, out)

        if self.loss_type == "quantile":
            # IT: Vincentization — media pesata per livello quantile + re-sort
            #     monotono (i quantili mixati restano non-decrescenti per costruzione
            #     se le teste lo sono; il sort è la cintura di sicurezza).
            # EN: Vincentization — per-level gate-weighted average + monotone
            #     re-sort (mixed quantiles stay non-decreasing by construction if
            #     the heads are; the sort is the safety belt).
            qp = (g.unsqueeze(-1) * outs).sum(dim=1)               # (B, Q)
            qp, _ = qp.sort(dim=-1)
            return (qp,)

        # IT: path t_student — mixing nello spazio NATURALE della varianza
        #     (softplus(ls2)+1e-6 = σ²), poi ri-codifica in ls2 raw via inversa.
        # EN: t_student path — mixing in the NATURAL variance space
        #     (softplus(ls2)+1e-6 = σ²), then re-encode into raw ls2 via inverse.
        mu_k  = outs[..., 0]                                       # (B, K)
        ls2_k = outs[..., 1]
        lnu_k = outs[..., 2]
        var_k = F.softplus(ls2_k) + 1e-6

        mu_mix  = (g * mu_k).sum(dim=1)                            # (B,)
        # IT: legge della varianza totale: within (Σ g σ²) + between (disagreement ≥ 0).
        # EN: total variance law: within (Σ g σ²) + between (disagreement ≥ 0).
        var_mix = (g * var_k).sum(dim=1) \
                + (g * (mu_k - mu_mix.unsqueeze(1)) ** 2).sum(dim=1)
        ls2_mix = _softplus_inverse(var_mix - 1e-6)
        lnu_mix = (g * lnu_k).sum(dim=1)
        return (mu_mix, ls2_mix, lnu_mix)


# IT: Imposta clip bounds adattivi per-feature (persistenti via register_buffer).
# EN: Set adaptive per-feature clip bounds (persisted via register_buffer).
def set_clip_bounds(model: nn.Module, clip_lo: np.ndarray, clip_hi: np.ndarray) -> None:
    """Imposta i clip bounds adattivi nel modello come buffer non-trainable.
    Chiamare dopo aver calcolato i percentili da X_train e prima del training.
    I bounds vengono salvati nel checkpoint e ricaricati automaticamente al load.
    No-op per modelli che non supportano clip bounds (es. QuantTFT)."""
    if not hasattr(model, "clip_lo"):
        return
    lo = torch.from_numpy(clip_lo.astype(np.float32))
    hi = torch.from_numpy(clip_hi.astype(np.float32))
    model.clip_lo.copy_(lo)
    model.clip_hi.copy_(hi)


# ─── QuantLSTM con Dual-Stream ───────────────────────────────────────────────

# IT: Dual-Stream LSTM (dinamico+strutturale) → GRU → testa t-Student/quantile.
# EN: Dual-Stream LSTM (dynamic+structural) → GRU → Student-t/quantile head.
class QuantLSTM(nn.Module):
    """
    Dual-Stream LSTM:
      Stream A (dinamico):   log-ret, vol, momentum → LSTM → TemporalAttention
      Stream B (strutturale):VP, ATH, CVD          → StructuralEncoder (MLP)
      Fusion: gated average dei due stream per ogni step temporale
      GRU(128) → MLP residual → [μ, log_σ², log_ν]

    Se n_dynamic_features=None → single-stream (compatibilità backward).

    Nuovi parametri (backward-compatible):
      loss_type: "t_student" (default) o "quantile"
      use_multitask: aggiunge directional head (B, 3) al return
      n_output_experts: MoE output heads (1 = single head, default)
    """

    # IT: Costruisce encoder, GRU, MLP residual e teste di output configurabili.
    # EN: Builds encoders, GRU, residual MLP and configurable output heads.
    def __init__(self, n_features, lstm_hidden=256, gru_hidden=128,
                 mlp_hidden=64, n_lstm_layers=2, dropout=0.2,
                 n_attention_heads=4, use_attention=True,
                 n_dynamic_features=None,
                 loss_type: str = "t_student",
                 use_multitask: bool = False,
                 n_output_experts: int = 1):
        super().__init__()
        self.use_attention    = use_attention
        self.n_dynamic        = n_dynamic_features   # None = single stream
        self.dual_stream      = n_dynamic_features is not None and n_dynamic_features < n_features
        self.loss_type        = loss_type
        self.use_multitask    = use_multitask
        self.n_output_experts = max(1, n_output_experts)
        self.register_buffer("clip_lo", torch.full((n_features,), -500.0))
        self.register_buffer("clip_hi", torch.full((n_features,), +500.0))

        if self.dual_stream:
            n_struct = n_features - n_dynamic_features
            # IT: Stream A — feature dinamiche (log-ret, vol, momentum).
            # EN: Stream A — dynamic features (log-ret, vol, momentum).
            self.input_norm_dyn  = nn.LayerNorm(n_dynamic_features)
            self.input_proj_dyn  = nn.Linear(n_dynamic_features, lstm_hidden)
            # IT: Stream B — feature strutturali (VP, ATH, CVD) via MLP encoder.
            # EN: Stream B — structural features (VP, ATH, CVD) via MLP encoder.
            self.struct_encoder  = StructuralEncoder(n_struct, lstm_hidden, dropout)
        else:
            self.input_norm = nn.LayerNorm(n_features)
            self.input_proj = nn.Linear(n_features, lstm_hidden)

        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, n_lstm_layers,
                            dropout=dropout if n_lstm_layers > 1 else 0.0,
                            batch_first=True)
        self.lstm_norm = nn.LayerNorm(lstm_hidden)

        if use_attention:
            self.attention = TemporalAttention(
                d_model=lstm_hidden, n_heads=n_attention_heads,
                dropout=dropout * 0.5,
            )

        self.gru      = nn.GRU(lstm_hidden, gru_hidden, batch_first=True)
        self.gru_norm = nn.LayerNorm(gru_hidden)

        self.fc1           = nn.Linear(gru_hidden, mlp_hidden)
        self.fc1_norm      = nn.LayerNorm(mlp_hidden)
        self.dropout       = nn.Dropout(dropout)
        self.fc2           = nn.Linear(mlp_hidden, mlp_hidden // 2)
        self.residual_proj = nn.Linear(gru_hidden, mlp_hidden // 2)

        out_dim = mlp_hidden // 2

        # ── Output heads ─────────────────────────────────────────────────────
        if self.n_output_experts > 1:
            # IT: MoE: gate softmax + N teste expert; output = somma pesata.
            # EN: MoE: softmax gate + N expert heads; output = weighted sum.
            self.expert_gate = nn.Linear(out_dim, self.n_output_experts)
            self.expert_heads = nn.ModuleList([
                nn.Linear(out_dim, 3 if loss_type == "t_student" else len(QUANTILES))
                for _ in range(self.n_output_experts)
            ])
        else:
            if loss_type == "quantile":
                self.quantile_head = nn.Linear(out_dim, len(QUANTILES))
                nn.init.normal_(self.quantile_head.weight, std=0.01)
                nn.init.zeros_(self.quantile_head.bias)
            else:
                self.out_mu      = nn.Linear(out_dim, 1)
                self.out_logsig2 = nn.Linear(out_dim, 1)
                self.out_lognu   = nn.Linear(out_dim, 1)

        # ── Multitask directional head ────────────────────────────────────────
        if use_multitask:
            self.dir_head = nn.Linear(out_dim, 3)
            nn.init.normal_(self.dir_head.weight, std=0.01)
            nn.init.zeros_(self.dir_head.bias)

        self._init_weights()
        # IT: SN su output heads (legacy: tutte; opt-in solo-mu via _QS_SN_ON_MU_ONLY).
        # EN: Spectral norm on output heads (legacy: all; opt-in mu-only via _QS_SN_ON_MU_ONLY).
        if self.n_output_experts <= 1 and self.loss_type == "t_student":
            self.out_mu      = spectral_norm(self.out_mu)
            if not _QS_SN_ON_MU_ONLY:
                self.out_logsig2 = spectral_norm(self.out_logsig2)
                self.out_lognu   = spectral_norm(self.out_lognu)

    # IT: Init pesi: xavier/orthogonal su LSTM, bias log_nu calibrato a df≈5.
    # EN: Weight init: xavier/orthogonal on LSTM, log_nu bias calibrated to df≈5.
    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:    nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:  nn.init.orthogonal_(p)
            elif "bias" in name:       nn.init.zeros_(p)
            elif "weight" in name and p.dim() == 2:
                nn.init.xavier_uniform_(p)
        # IT: Init bias log_nu così softplus(bias)+2 ≈ 5 (df t-Student di default).
        # EN: Init log_nu bias so softplus(bias)+2 ≈ 5 (default t-Student df).
        if self.n_output_experts <= 1 and self.loss_type == "t_student":
            with torch.no_grad():
                self.out_lognu.bias.fill_(math.log(5.0 - 2.0))

    # IT: Forward dual/single-stream → fuse → LSTM → attn → GRU → MLP residual → heads.
    # EN: Forward dual/single-stream → fuse → LSTM → attn → GRU → residual MLP → heads.
    def forward(self, x, x_macro=None):
        x = x.clamp(self.clip_lo, self.clip_hi)  # IT: clip adattivo | EN: adaptive clip
        if self.dual_stream:
            x_dyn    = x[:, :, :self.n_dynamic]
            x_struct = x[:, :, self.n_dynamic:]
            xp_dyn   = F.silu(self.input_proj_dyn(self.input_norm_dyn(x_dyn)))
            xp_str   = self.struct_encoder(x_struct)
            # IT: Gated fusion point-wise lungo T.
            # EN: Point-wise gated fusion along T.
            xp = self.struct_encoder.fuse(xp_dyn, xp_str)
        else:
            xp = F.silu(self.input_proj(self.input_norm(x)))

        lo, _ = self.lstm(xp)
        lo    = self.lstm_norm(lo)

        if self.use_attention:
            attn_ctx, _ = self.attention(lo)
            lo_gru = torch.cat([lo[:, :-1, :], attn_ctx.unsqueeze(1)], dim=1)
        else:
            lo_gru = lo

        go, _ = self.gru(lo_gru)
        g     = self.gru_norm(go[:, -1, :])

        h = F.silu(self.fc1_norm(self.fc1(g)))
        h_feature = F.silu(self.fc2(self.dropout(h))) + self.residual_proj(g)

        # ── Output computation ───────────────────────────────────────────────
        if self.n_output_experts > 1:
            gate_w      = F.softmax(self.expert_gate(h_feature), dim=-1)          # (B, E)
            expert_outs = torch.stack(
                [eh(h_feature) for eh in self.expert_heads], dim=1
            )                                                                       # (B, E, out)
            h_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)                # (B, out)

            if self.loss_type == "quantile":
                quantile_preds = h_out                                              # (B, Q)
                if self.use_multitask:
                    dir_logits = self.dir_head(h_feature)
                    return (quantile_preds, dir_logits)
                return (quantile_preds,)
            else:
                mu       = h_out[:, 0]
                log_sig2 = h_out[:, 1]
                log_nu   = h_out[:, 2]
                if self.use_multitask:
                    dir_logits = self.dir_head(h_feature)
                    return (mu, log_sig2, log_nu, dir_logits)
                return (mu, log_sig2, log_nu)

        elif self.loss_type == "quantile":
            quantile_preds = self.quantile_head(h_feature)                          # (B, Q)
            if self.use_multitask:
                dir_logits = self.dir_head(h_feature)
                return (quantile_preds, dir_logits)
            return (quantile_preds,)

        else:
            mu      = self.out_mu(h_feature).squeeze(-1)
            log_sig = self.out_logsig2(h_feature).squeeze(-1)
            log_nu  = self.out_lognu(h_feature).squeeze(-1)
            if self.use_multitask:
                dir_logits = self.dir_head(h_feature)
                return (mu, log_sig, log_nu, dir_logits)
            return (mu, log_sig, log_nu)

    # IT: Inferenza eval-mode → dict numpy (mu/sigma/nu o quantili).
    # EN: Eval-mode inference → numpy dict (mu/sigma/nu or quantiles).
    @torch.no_grad()
    def predict(self, x, x_macro=None):
        self.eval()
        out = self.forward(x, x_macro)
        if self.loss_type == "quantile":
            quantile_preds = out[0]
            quantile_preds, _ = quantile_preds.sort(dim=-1)
            sigma = quantile_preds[:, 4] - quantile_preds[:, 0]
            return {
                "mu":        quantile_preds[:, 2].cpu().numpy(),
                "sigma":     sigma.cpu().numpy(),
                "quantiles": quantile_preds.cpu().numpy(),
            }
        else:
            mu, ls2, lnu = out[0], out[1], out[2]
            sigma, nu    = student_t_params(ls2, lnu)
            return {"mu": mu.cpu().numpy(), "sigma": sigma.cpu().numpy(), "nu": nu.cpu().numpy()}

        sigmas_arr = torch.stack(sigmas, dim=0).cpu().numpy()
        if nus[0] is None:
            nus_arr = np.full_like(mus_arr, float("nan"))
        else:
            nus_arr = torch.stack(nus, dim=0).cpu().numpy()
        mu_mean   = mus_arr.mean(axis=0); mu_std = mus_arr.std(axis=0)
        sig_mean  = sigmas_arr.mean(axis=0)
        nu_mean   = nus_arr.mean(axis=0)
        sig_total = np.sqrt(sig_mean**2 + mu_std**2)
        # NOTA (audit #19, 2026-05-24): confidence è scale-invariant — ratio mu_std/sig_mean.
        # In z-score o raw produce lo stesso valore (target_scale si cancella numeratore/denom).
        # Safe da usare ovunque, prima o dopo denormalize_predictions.
        # sig_total NON è scale-invariant: va denormalizzato se confrontato con scale assolute.
        confidence= np.clip(1.0 - mu_std / (sig_mean + 1e-9), 0.0, 1.0)
        return {"mu": mu_mean, "mu_std": mu_std, "sigma": sig_mean,
                "sigma_total": sig_total, "nu": nu_mean, "confidence_score": confidence}


# ─── EarlyStopping ───────────────────────────────────────────────────────────

# IT: Early stopping su val_loss con salvataggio del best checkpoint.
# EN: Early stopping on val_loss with best-checkpoint saving.
class EarlyStopping:
    # IT: Inizializza pazienza, path e tracking del miglior val_loss.
    # EN: Initializes patience, path and best-val_loss tracking.
    def __init__(self, patience=20, path="models/best_model.pt"):
        self.patience = patience; self.path = path
        self.best = float("inf"); self.counter = 0; self.triggered = False

    # IT: Aggiorna stato: salva se migliora, incrementa counter altrimenti.
    # EN: Updates state: saves if improved, increments counter otherwise.
    def __call__(self, val_loss: float, model) -> bool:
        if val_loss < self.best - 1e-6:
            self.best = val_loss; self.counter = 0
            # IT: Fix #26 — save atomico via tmp+rename: evita checkpoint corrotti su crash.
            # EN: Fix #26 — atomic save via tmp+rename: avoids corrupt checkpoints on crash.
            _tmp = f"{self.path}.tmp"
            torch.save(model.state_dict(), _tmp)
            os.replace(_tmp, self.path)
            log.info(f"  ✓ Checkpoint  val_nll={val_loss:.5f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered

    # IT: Ricarica nel modello il miglior checkpoint salvato (se esiste).
    # EN: Reloads the best saved checkpoint into the model (if it exists).
    def restore(self, model) -> None:
        import os
        if os.path.exists(self.path):
            try:
                state = torch.load(self.path, map_location="cpu", weights_only=True)
            except Exception as e:
                log.warning(f"weights_only=True failed ({e}); falling back to legacy load")
                state = torch.load(self.path, map_location="cpu", weights_only=False)
            model.load_state_dict(state)
            log.info(f"Best model ripristinato da {self.path}  (val_nll={self.best:.5f})")
        else:
            log.warning(f"Checkpoint non trovato: {self.path}")


# ─── load_model ──────────────────────────────────────────────────────────────

# IT: Factory: istanzia la giusta arch da config.json e carica i pesi del checkpoint.
# EN: Factory: instantiates the right arch from config.json and loads checkpoint weights.
def load_model(checkpoint: str, config_path: str = None):
    """
    Carica QuantLSTM o QuantLSTMWithMacro leggendo la config dal JSON.
    Supporta sia single-stream che dual-stream (n_dynamic_features).
    """
    import json, os
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint), "config.json")
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    has_macro  = cfg.get("has_macro", False)
    model_type = cfg.get("model_type", "QuantLSTM")

    if model_type == "QuantiTransformer":
        n_feat = cfg["n_features"]
        n_dyn  = int(cfg["n_dynamic_features"]) if cfg.get("n_dynamic_features") else n_feat
        n_mac  = cfg.get("n_macro", 0) if has_macro else 0
        T      = cfg.get("window_size", 120)
        model  = QuantiTransformer(
            n_features       = n_feat,
            T                = T,
            n_dynamic        = n_dyn,
            n_macro          = n_mac,
            d_model          = cfg.get("tft_d_model", 128),
            n_heads          = cfg.get("tft_n_heads", 4),
            n_layers         = cfg.get("tft_n_layers", 3),
            dropout          = 0.0,
            patch_size       = cfg.get("patch_size", 1),
            drop_path_rate   = cfg.get("drop_path_rate", 0.0),
            loss_type        = cfg.get("loss_type", "t_student"),
            use_multitask    = cfg.get("use_multitask", False),
            n_output_experts = cfg.get("n_output_experts", 1),
            use_revin        = cfg.get("use_revin", False),
            revin_target_idx = cfg.get("revin_target_idx", 0),
            # IT: A3 — chiave assente nei config.json storici → "single" = path
            #     bit-identico ai checkpoint esistenti.
            # EN: A3 — key absent from legacy config.json → "single" = bit-identical
            #     path for existing checkpoints.
            head_type        = cfg.get("head_type", "single"),
        )
        log.info(f"load_model: QuantiTransformer (F={n_feat}, T={T}, n_macro={n_mac}, "
                 f"head_type={cfg.get('head_type', 'single')})")
    elif model_type == "QuantTFT":
        n_feat = cfg["n_features"]
        n_dyn  = int(cfg["n_dynamic_features"]) if cfg.get("n_dynamic_features") else n_feat
        n_str  = n_feat - n_dyn   # 0 = single-stream
        n_mac  = cfg.get("n_macro", 0) if has_macro else 0
        model  = QuantTFT(
            n_dynamic    = n_dyn,
            n_structural = n_str,
            n_macro      = n_mac,
            d_model      = cfg.get("tft_d_model", 64),
            n_heads      = cfg.get("tft_n_heads", 4),
            dropout      = 0.0,
        )
        log.info(f"load_model: QuantTFT (n_dynamic={n_dyn}, n_structural={n_str}, n_macro={n_mac})")
    elif model_type == "QuantNHiTS":
        n_feat = cfg["n_features"]
        n_dyn  = int(cfg["n_dynamic_features"]) if cfg.get("n_dynamic_features") else n_feat
        n_mac  = cfg.get("n_macro", 0) if has_macro else 0
        T      = cfg.get("window_size", 120)
        model  = QuantNHiTS(
            n_features         = n_feat,
            T                  = T,
            n_dynamic_features = n_dyn,
            n_macro            = n_mac,
            d_model            = cfg.get("d_model", 128),
            hidden             = cfg.get("nhits_hidden", 256),
            n_stacks           = cfg.get("nhits_stacks", 3),
            pool_kernels       = tuple(cfg.get("nhits_pool_kernels", [8, 4, 1])),
            n_blocks_per_stack = cfg.get("nhits_blocks_per_stack", 1),
            n_mlp_layers       = cfg.get("nhits_mlp_layers", 2),
            dropout            = 0.0,
            loss_type          = cfg.get("loss_type", "t_student"),
            use_multitask      = cfg.get("use_multitask", False),
            n_output_experts   = cfg.get("n_output_experts", 1),
            use_revin          = cfg.get("use_revin", False),
            revin_target_idx   = cfg.get("revin_target_idx", 0),
        )
        log.info(f"load_model: QuantNHiTS (F={n_feat}, T={T}, n_macro={n_mac})")
    elif model_type == "QuantTCNMamba":
        n_feat = cfg["n_features"]
        n_dyn  = int(cfg["n_dynamic_features"]) if cfg.get("n_dynamic_features") else n_feat
        model  = QuantTCNMamba(
            n_features         = n_feat,
            d_model            = cfg.get("d_model", 128),
            tcn_layers         = cfg.get("tcn_layers", 4),
            tcn_kernel         = cfg.get("tcn_kernel", 3),
            mamba_layers       = cfg.get("mamba_layers", 3),
            mamba_d_state      = cfg.get("mamba_d_state", 16),
            mamba_expand       = cfg.get("mamba_expand", 2),
            dropout            = 0.0,
            n_dynamic_features = n_dyn,
            use_multitask      = cfg.get("use_multitask", False),
            loss_type          = cfg.get("loss_type", "t_student"),
            n_output_experts   = cfg.get("n_output_experts", 1),
            use_revin          = cfg.get("use_revin", False),
            revin_target_idx   = cfg.get("revin_target_idx", 0),
        )
        log.info(f"load_model: QuantTCNMamba (n_features={n_feat}, d_model={cfg.get('d_model', 128)})")
    elif has_macro or model_type == "QuantLSTMWithMacro":
        from quantsys.macro.regime import QuantLSTMWithMacro
        model = QuantLSTMWithMacro(
            n_price_features   = cfg["n_features"],
            n_macro_features   = cfg.get("n_macro", 1),
            lstm_hidden        = cfg.get("lstm_hidden", 256),
            gru_hidden         = cfg.get("gru_hidden", 128),
            mlp_hidden         = cfg.get("mlp_hidden", 64),
            macro_embed_dim    = cfg.get("macro_embed_dim", 16),
            n_lstm_layers      = cfg.get("lstm_layers", 2),
            dropout            = 0.0,
            n_dynamic_features = cfg.get("n_dynamic_features"),
        )
        log.info(f"load_model: QuantLSTMWithMacro (n_price={cfg['n_features']}, "
                 f"n_macro={cfg.get('n_macro','?')}, "
                 f"dual_stream={cfg.get('n_dynamic_features') is not None})")
    else:
        model = QuantLSTM(
            n_features         = cfg["n_features"],
            lstm_hidden        = cfg.get("lstm_hidden", 256),
            gru_hidden         = cfg.get("gru_hidden", 128),
            mlp_hidden         = cfg.get("mlp_hidden", 64),
            n_lstm_layers      = cfg.get("lstm_layers", 2),
            dropout            = 0.0,
            n_attention_heads  = cfg.get("n_attention_heads", 4),
            use_attention      = cfg.get("use_attention", True),
            n_dynamic_features = cfg.get("n_dynamic_features"),
            loss_type          = cfg.get("loss_type", "t_student"),
            use_multitask      = cfg.get("use_multitask", False),
            n_output_experts   = cfg.get("n_output_experts", 1),
        )
        log.info(f"load_model: QuantLSTM (n_features={cfg['n_features']}, "
                 f"dual_stream={cfg.get('n_dynamic_features') is not None})")

    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as e:
        log.warning(f"weights_only=True failed ({e}); falling back to legacy load")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.eval()


# ─── Temporal Fusion Transformer (semplificato) ──────────────────────────────

# IT: Blocco GRN (TFT): trasformazione gated + skip residuale + LayerNorm.
# EN: GRN block (TFT): gated transform + residual skip + LayerNorm.
class GatedResidualNetwork(nn.Module):
    """
    output = LayerNorm(x + Dropout(gate ⊙ dense2(ELU(dense1(x)))))
    gate   = sigmoid(dense_gate(x))

    Il gate apprende dinamicamente quanta parte dell'attivazione lasciare passare,
    permettendo al network di sopprimere trasformazioni non informative e
    comportarsi come identità quando necessario (utile nei layer profondi).
    """
    # IT: Crea dense/gate/skip; skip lineare solo se cambia dimensione.
    # EN: Builds dense/gate/skip; linear skip only when dimension changes.
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dense1      = nn.Linear(input_dim,  hidden_dim)
        self.dense2      = nn.Linear(hidden_dim, output_dim)
        self.dense_gate  = nn.Linear(input_dim,  output_dim)
        self.norm        = nn.LayerNorm(output_dim)
        self.dropout     = nn.Dropout(dropout)
        # IT: Proiezione residuale solo se le dimensioni non coincidono.
        # EN: Residual projection only when in/out dimensions differ.
        self.skip        = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()

    # IT: Applica gate sigmoide alla trasformazione + somma residuale.
    # EN: Applies sigmoid gate to the transform + residual sum.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.dense_gate(x))
        h    = self.dropout(gate * self.dense2(F.elu(self.dense1(x))))
        return self.norm(self.skip(x) + h)


# IT: VSN (TFT): pesa via softmax le feature e somma le loro trasformazioni.
# EN: VSN (TFT): softmax-weights features and sums their transforms.
class VariableSelectionNetwork(nn.Module):
    """
    Impara pesi softmax per ogni feature e restituisce la somma pesata
    delle trasformazioni per-feature. Consente al modello di ignorare
    feature irrilevanti a ogni step temporale, adattandosi al contesto.
    """
    # IT: Crea una GRN per-feature + una GRN che produce i pesi softmax.
    # EN: Builds one per-feature GRN + one GRN producing the softmax weights.
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        # IT: Una GRN indipendente per feature: scalar → d_model.
        # EN: One independent GRN per feature: scalar → d_model.
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, d_model, d_model, dropout) for _ in range(n_features)
        ])
        # IT: GRN sull'intero vettore input → logit dei pesi per-feature.
        # EN: GRN over the full input vector → per-feature weight logits.
        self.weight_grn = GatedResidualNetwork(n_features, d_model, n_features, dropout)

    # IT: Pesa via softmax e somma le trasformazioni per-feature → (B, T, d_model).
    # EN: Softmax-weights and sums per-feature transforms → (B, T, d_model).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, n_features)
        B, T, F = x.shape
        # IT: Pesi per feature: softmax lungo la dimensione feature.
        # EN: Per-feature weights: softmax along the feature dimension.
        weights = torch.softmax(self.weight_grn(x.reshape(B * T, F)).reshape(B, T, F), dim=-1)
        # IT: Trasforma ogni feature separatamente e impila.
        # EN: Transform each feature separately and stack.
        processed = torch.stack(
            [self.feature_grns[i](x[..., i:i+1]) for i in range(F)],
            dim=-2,
        )  # (batch, seq, n_features, d_model)
        # IT: Somma pesata: (B,T,F,1) * (B,T,F,d) → (B,T,d).
        # EN: Weighted sum: (B,T,F,1) * (B,T,F,d) → (B,T,d).
        return (weights.unsqueeze(-1) * processed).sum(dim=-2)


# IT: TFT semplificato: VSN dual-stream → fusion GRN → attention → testa t-Student.
# EN: Simplified TFT: dual-stream VSN → fusion GRN → attention → Student-t head.
class QuantTFT(nn.Module):
    """
    TFT semplificato per forecasting BTC/USDT 1m.
    Mantiene la divisione dual-stream (dynamic / structural) di QuantLSTM
    per permettere confronto diretto e backward compat con il pipeline esistente.
    """

    # IT: Costruisce VSN, fusione, GRN temporali, attention causale e teste output.
    # EN: Builds VSN, fusion, temporal GRNs, causal attention and output heads.
    def __init__(self, n_dynamic: int, n_structural: int, n_macro: int = 0,
                 d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_dynamic      = n_dynamic
        self.n_structural   = n_structural
        self._single_stream = (n_structural == 0)  # IT: no split → stream condividono tutte le feature | EN: no split → streams share all features
        self.n_macro        = n_macro
        self.d_model        = d_model
        self.loss_type      = "t_student"
        self.use_multitask  = False
        n_total = n_dynamic + n_structural
        self.register_buffer("clip_lo", torch.full((n_total,), -500.0))
        self.register_buffer("clip_hi", torch.full((n_total,), +500.0))

        self.vsn_dynamic    = VariableSelectionNetwork(n_dynamic, d_model, dropout)
        _n_str_vsn          = n_dynamic if self._single_stream else n_structural
        self.vsn_structural = VariableSelectionNetwork(_n_str_vsn, d_model, dropout)

        # IT: Proiezione macro: vettore statico sommato all'output VSN strutturale.
        # EN: Macro projection: static vector added to the structural VSN output.
        if n_macro > 0:
            self.macro_proj = nn.Sequential(
                nn.Linear(n_macro, d_model),
                nn.LayerNorm(d_model),
                nn.SiLU(),
            )

        # IT: Fusione dei due stream + normalizzazione prima dell'attention.
        # EN: Fuses the two streams + normalizes before attention.
        self.fusion_grn = GatedResidualNetwork(d_model * 2, d_model * 2, d_model, dropout)
        self.pre_attn_norm = nn.LayerNorm(d_model)

        # IT: Elaborazione temporale: tre GRN in sequenza su d_model.
        # EN: Temporal processing: three stacked GRNs over d_model.
        self.temporal_grns = nn.Sequential(
            GatedResidualNetwork(d_model, d_model * 2, d_model, dropout),
            GatedResidualNetwork(d_model, d_model * 2, d_model, dropout),
            GatedResidualNetwork(d_model, d_model * 2, d_model, dropout),
        )

        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # IT: GRN finale prima della head di output.
        # EN: Final GRN before the output head.
        self.output_grn = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        self.out_mu      = nn.Linear(d_model, 1)
        self.out_logsig2 = nn.Linear(d_model, 1)
        self.out_lognu   = nn.Linear(d_model, 1)

        self._init_output_heads()
        # IT: Spectral norm sulle teste (opt-in solo-mu via _QS_SN_ON_MU_ONLY).
        # EN: Spectral norm on heads (opt-in mu-only via _QS_SN_ON_MU_ONLY).
        self.out_mu      = spectral_norm(self.out_mu)
        if not _QS_SN_ON_MU_ONLY:
            self.out_logsig2 = spectral_norm(self.out_logsig2)
            self.out_lognu   = spectral_norm(self.out_lognu)

    # IT: Init teste con pesi piccoli; bias log_nu calibrato a df≈5.
    # EN: Init heads with small weights; log_nu bias calibrated to df≈5.
    def _init_output_heads(self):
        # IT: Scala piccola: evita saturazione/instabilità a inizio training.
        # EN: Small scale: avoids saturation/instability at training start.
        for layer in (self.out_mu, self.out_logsig2, self.out_lognu):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)
        # IT: Inizializza log_nu così softplus(bias)+2 ≈ 5 → df sicuri > 2.
        # EN: Init log_nu so softplus(bias)+2 ≈ 5 → safe df > 2.
        with torch.no_grad():
            self.out_lognu.bias.fill_(math.log(5.0 - 2.0))

    # IT: Forward: split → VSN → fusione → GRN → attention causale → testa output.
    # EN: Forward: split → VSN → fusion → GRN → causal attention → output head.
    def forward(self, x: torch.Tensor, x_macro: torch.Tensor = None) -> tuple:
        """
        Firma compatibile con QuantLSTM: x = (batch, seq, n_dynamic + n_structural).
        Lo split è fatto internamente con n_dynamic memorizzato nel costruttore.
        Se n_structural == 0, entrambi gli stream ricevono lo stesso input (single-stream).
        """
        x = x.clamp(self.clip_lo, self.clip_hi)
        x_dynamic    = x[:, :, :self.n_dynamic]
        x_structural = x_dynamic if self._single_stream else x[:, :, self.n_dynamic:]

        # IT: 1. VSN per stream dinamico e strutturale → (B, seq, d_model) ciascuno.
        # EN: 1. VSN for dynamic and structural streams → (B, seq, d_model) each.
        h_dyn  = self.vsn_dynamic(x_dynamic)
        h_str  = self.vsn_structural(x_structural)

        # IT: 2. Se presente, inietta il macro embedding nello stream strutturale.
        # EN: 2. If present, inject the macro embedding into the structural stream.
        if self.n_macro > 0 and x_macro is not None:
            h_macro = self.macro_proj(x_macro)           # (batch, d_model)
            h_str   = h_str + h_macro.unsqueeze(1)       # IT: broadcast su tutti i timestep | EN: broadcast over all timesteps

        # IT: 3. Fusione tramite GRN sul concat dei due stream.
        # EN: 3. Fuse via GRN over the concat of the two streams.
        h = self.fusion_grn(torch.cat([h_dyn, h_str], dim=-1))
        h = self.pre_attn_norm(h)

        # IT: 4. Stack di GRN per elaborazione temporale.
        # EN: 4. Stack of GRNs for temporal processing.
        h = self.temporal_grns(h)

        # IT: 5. Self-attention causale: ogni timestep vede solo il passato.
        # EN: 5. Causal self-attention: each timestep sees only the past.
        T = h.shape[1]
        if not hasattr(self, '_causal_mask') or self._causal_mask.shape[0] != T:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            self.register_buffer('_causal_mask', mask)
        attn_out, _ = self.attention(h, h, h, attn_mask=self._causal_mask)
        h = self.post_attn_norm(h + attn_out)

        # IT: 6. Prende solo l'ultimo timestep e applica la GRN finale.
        # EN: 6. Takes only the last timestep and applies the final GRN.
        h_last = self.output_grn(h[:, -1, :])

        mu      = self.out_mu(h_last).squeeze(-1)
        log_sig = self.out_logsig2(h_last).squeeze(-1)
        log_nu  = self.out_lognu(h_last).squeeze(-1)
        return mu, log_sig, log_nu

    # IT: Inferenza eval-mode → dict numpy con mu/sigma/nu.
    # EN: Eval-mode inference → numpy dict with mu/sigma/nu.
    @torch.no_grad()
    def predict(self, x: torch.Tensor, x_macro: torch.Tensor = None) -> dict:
        self.eval()
        mu, ls2, lnu = self.forward(x, x_macro)
        sigma, nu    = student_t_params(ls2, lnu)
        return {"mu": mu.cpu().numpy(), "sigma": sigma.cpu().numpy(), "nu": nu.cpu().numpy()}


# ─── DropPath (Stochastic Depth) ─────────────────────────────────────────────

# IT: Stochastic Depth: droppa interi sample in training per regolarizzare.
# EN: Stochastic Depth: drops whole samples during training to regularize.
class DropPath(nn.Module):
    """Stochastic Depth: durante training droppa interi sample con prob drop_prob."""
    # IT: Memorizza la probabilità di drop per-sample.
    # EN: Stores the per-sample drop probability.
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    # IT: Applica maschera per-sample con rescaling 1/keep (no-op in eval).
    # EN: Applies per-sample mask with 1/keep rescaling (no-op in eval).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # IT: add keep PRIMA di floor: floor(rand+keep)=1 con prob keep, 0 altrimenti.
        # EN: add keep BEFORE floor: floor(rand+keep)=1 with prob keep, 0 otherwise.
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x * mask / keep


# ─── iTransformer ─────────────────────────────────────────────────────────────

# IT: Layer iTransformer: attention inter-feature pre-norm + FFN + DropPath.
# EN: iTransformer layer: pre-norm inter-feature attention + FFN + DropPath.
class iTransformerLayer(nn.Module):
    """
    Pre-norm self-attention + pre-norm FFN sul dominio delle feature token.
    Nessuna causal mask: l'attenzione è bidirezionale tra feature (non tra timestep).
    Usa F.scaled_dot_product_attention (Flash Attention) per stabilità float16 con AMP.
    Supporta DropPath (Stochastic Depth) per regolarizzazione.
    """
    # IT: Costruisce qkv/out proj, FFN e DropPath; init gain ridotto per fp16.
    # EN: Builds qkv/out proj, FFN and DropPath; reduced init gain for fp16.
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.dropout_p = dropout

        self.norm1    = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        # IT: gain=0.5: riduce la varianza iniziale per evitare overflow float16.
        # EN: gain=0.5: lowers initial variance to avoid float16 overflow.
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path)

    # IT: Forward pre-norm: attention inter-feature + FFN, entrambi con DropPath residuale.
    # EN: Pre-norm forward: inter-feature attention + FFN, both with residual DropPath.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv_proj(h).chunk(3, dim=-1)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        # IT: Flash Attention: stabile in float16, nessun overflow del softmax.
        # EN: Flash Attention: stable in float16, no softmax overflow.
        a = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        a = a.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.drop_path(self.out_proj(a))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# IT: iTransformer: ogni feature è un token, attention inter-feature O(F²), multi-scala.
# EN: iTransformer: each feature is a token, O(F²) inter-feature attention, multi-scale.
class QuantiTransformer(nn.Module):
    """
    iTransformer per serie temporali finanziarie.

    Ogni feature diventa un token: l'attention impara correlazioni inter-feature
    (vol_std_5 ↔ RSI ↔ log_ret) invece di relazioni temporali.
    La storia di ciascuna feature è compressa in d_model via proiezione
    multi-scala (1min completo, pooled a 5min, pooled a 15min).

    Complessità attention: O(F²) = O(55²) = 3025  vs  O(T²) = O(120²) = 14400
    → 4.7× meno operazioni attention rispetto a TFT.

    Miglioramenti architetturali:
      - Temporal Patching: riduce la risoluzione temporale prima dell'embedding
        via adaptive_avg_pool1d (patch_size > 1). Riduce il rumore ad alta freq.
      - Learnable Positional Embedding: nn.Embedding(n_features+1, d_model)
        per dare al modello informazioni di posizione per-feature.
      - DropPath (Stochastic Depth): regolarizzazione layer-level con intensità
        crescente per layer (drop_path_rate scheduling lineare).
      - Multi-task directional head: predice up/flat/down (3 classi) in parallelo.
      - Quantile regression: supporta loss_type="quantile" con pinball loss.
      - Mixture-of-Experts: n_output_experts > 1 usa MoE con soft gating.
    """
    def __init__(
        self,
        n_features:       int,
        T:                int,
        n_dynamic:        int   = None,
        n_macro:          int   = 0,
        d_model:          int   = 128,
        n_heads:          int   = 4,
        n_layers:         int   = 3,
        dropout:          float = 0.1,
        patch_size:       int   = 1,
        drop_path_rate:   float = 0.0,
        loss_type:        str   = "t_student",
        use_multitask:    bool  = False,
        n_output_experts: int   = 1,
        use_revin:        bool  = False,
        revin_target_idx: int   = 0,
        head_type:        str   = "single",
    ):
        # IT: Costruisce embedding multi-scala, layer iTransformer e teste output.
        # EN: Builds multi-scale embeddings, iTransformer layers and output heads.
        super().__init__()
        self.n_features      = n_features
        self.n_macro         = n_macro
        self.loss_type       = loss_type
        self.use_multitask   = use_multitask
        self.n_output_experts = max(1, n_output_experts)
        self.patch_size      = max(1, patch_size)
        self.use_revin       = use_revin
        # ── A3 regime-MoE (config-gated; default "single" = path storico invariato) ──
        # IT: head_type="regime_moe" → 3 teste per-regime + gate esterno causale g.
        #     DIVERSO dal MoE appreso (n_output_experts, gate softmax da h_feature):
        #     combinazione vietata per semplicità. RevIN vietato (denorm inconsistente
        #     col mixing in spazio varianza; RevIN è comunque OFF sulla linea vol).
        # EN: head_type="regime_moe" → 3 per-regime heads + external causal gate g.
        #     DIFFERENT from the learned MoE (n_output_experts, softmax gate from
        #     h_feature): the combination is forbidden for simplicity. RevIN is
        #     forbidden too (denorm inconsistent with variance-space mixing; RevIN
        #     is OFF on the vol line anyway).
        _ht = head_type if head_type else "single"
        if _ht not in ("single", "regime_moe"):
            raise ValueError(f"head_type sconosciuto/unknown head_type: {_ht!r} "
                             f"(atteso/expected 'single' | 'regime_moe')")
        self.head_type = _ht
        if self.head_type == "regime_moe":
            if self.n_output_experts > 1:
                raise ValueError(
                    "head_type='regime_moe' incompatibile con n_output_experts>1 "
                    "(MoE appreso e regime-MoE mutuamente esclusivi) / "
                    "incompatible with n_output_experts>1 (learned MoE and "
                    "regime-MoE are mutually exclusive)"
                )
            if use_revin:
                raise ValueError(
                    "head_type='regime_moe' incompatibile con use_revin=True / "
                    "incompatible with use_revin=True"
                )
        # IT: flag warn-once per fallback gate uniforme / gate ignorato (plain attr:
        #     nessun impatto su state_dict né su RNG).
        # EN: warn-once flags for uniform-gate fallback / ignored gate (plain attrs:
        #     no impact on state_dict nor RNG).
        self._warned_uniform_gate = False
        self._warned_gate_ignored = False
        if use_revin:
            from quantsys.model.revin import RevIN
            self.revin = RevIN(n_features=n_features, target_idx=revin_target_idx)
        # IT: Clip bounds adattivi per-feature (set via set_clip_bounds); fallback ±500.
        # EN: Adaptive per-feature clip bounds (set via set_clip_bounds); fallback ±500.
        self.register_buffer("clip_lo", torch.full((n_features,), -500.0))
        self.register_buffer("clip_hi", torch.full((n_features,), +500.0))

        # IT: Temporal patching: calcola il T effettivo dopo il patching.
        # EN: Temporal patching: computes the effective T after patching.
        self.T_eff = max(1, T // self.patch_size)
        self.T5    = max(1, self.T_eff // 5)
        self.T15   = max(1, self.T_eff // 15)

        self.embed1      = nn.Linear(self.T_eff, d_model)
        self.embed2      = nn.Linear(self.T5,    d_model)
        self.embed3      = nn.Linear(self.T15,   d_model)
        self.scale_merge = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        n_dyn = n_dynamic if n_dynamic is not None else n_features
        ftype = torch.zeros(n_features, dtype=torch.long)
        ftype[n_dyn:] = 1
        self.register_buffer("feature_types", ftype)
        self.type_emb = nn.Embedding(2, d_model)

        # IT: Positional embedding learnable: +1 per il macro token opzionale.
        # EN: Learnable positional embedding: +1 for the optional macro token.
        self.pos_emb = nn.Embedding(n_features + 1, d_model)
        self.register_buffer('_pos_ids', torch.arange(n_features))

        if n_macro > 0:
            self.macro_proj = nn.Linear(n_macro, d_model)

        # IT: DropPath scheduling: intensità crescente per layer.
        # EN: DropPath scheduling: increasing intensity per layer.
        dpr = [drop_path_rate * i / max(1, n_layers - 1) for i in range(n_layers)]
        self.layers = nn.ModuleList([
            iTransformerLayer(d_model, n_heads, dropout, drop_path=dpr[i])
            for i in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

        # out_dim per QuantiTransformer è d_model
        out_dim = d_model

        # ── Output heads ─────────────────────────────────────────────────────
        if self.head_type == "regime_moe":
            # IT: A3 — 3 teste per-regime mescolate dal gate esterno causale.
            # EN: A3 — 3 per-regime heads mixed by the external causal gate.
            self.regime_head = RegimeMoEHead(out_dim, loss_type=loss_type)
        elif self.n_output_experts > 1:
            self.expert_gate = nn.Linear(out_dim, self.n_output_experts)
            self.expert_heads = nn.ModuleList([
                nn.Linear(out_dim, 3 if loss_type == "t_student" else len(QUANTILES))
                for _ in range(self.n_output_experts)
            ])
        else:
            if loss_type == "quantile":
                self.quantile_head = nn.Linear(out_dim, len(QUANTILES))
                nn.init.normal_(self.quantile_head.weight, std=0.01)
                nn.init.zeros_(self.quantile_head.bias)
            else:
                self.out_mu      = nn.Linear(out_dim, 1)
                self.out_logsig2 = nn.Linear(out_dim, 1)
                self.out_lognu   = nn.Linear(out_dim, 1)

        # ── Multitask directional head ────────────────────────────────────────
        if use_multitask:
            self.dir_head = nn.Linear(out_dim, 3)
            nn.init.normal_(self.dir_head.weight, std=0.01)
            nn.init.zeros_(self.dir_head.bias)

        # Initialize output heads with small weights for stable start
        # IT: guard head_type — con regime_moe le teste single non esistono
        #     (init/SN già gestiti dentro RegimeMoEHead).
        # EN: head_type guard — under regime_moe the single heads do not exist
        #     (init/SN already handled inside RegimeMoEHead).
        if self.head_type != "regime_moe" and self.n_output_experts <= 1 and loss_type == "t_student":
            nn.init.normal_(self.out_mu.weight, std=0.01)
            nn.init.zeros_(self.out_mu.bias)
            nn.init.normal_(self.out_logsig2.weight, std=0.01)
            nn.init.zeros_(self.out_logsig2.bias)
            nn.init.normal_(self.out_lognu.weight, std=0.01)
            with torch.no_grad():
                self.out_lognu.bias.fill_(math.log(5.0 - 2.0))
            # IT: Spectral norm sulle teste (opt-in solo-mu via _QS_SN_ON_MU_ONLY).
            # EN: Spectral norm on heads (opt-in mu-only via _QS_SN_ON_MU_ONLY).
            self.out_mu      = spectral_norm(self.out_mu)
            if not _QS_SN_ON_MU_ONLY:
                self.out_logsig2 = spectral_norm(self.out_logsig2)
                self.out_lognu   = spectral_norm(self.out_lognu)

    # IT: Forward: RevIN → multi-scala embed → attention inter-feature → testa output.
    #     g (opzionale, in coda): gate regime causale (B, 3) per head_type="regime_moe";
    #     g=None con regime_moe → RuntimeError in eval (input obbligatorio, audit
    #     MAJOR-2), fallback gate uniforme SOLO in train (warning una-tantum);
    #     g ignorato con head_type="single" (warning una-tantum). Il contratto
    #     forward(x, x_macro=None) resta invariato per tutti i caller esistenti.
    # EN: Forward: RevIN → multi-scale embed → inter-feature attention → output head.
    #     g (optional, trailing): causal regime gate (B, 3) for head_type="regime_moe";
    #     g=None under regime_moe → RuntimeError in eval (mandatory input, MAJOR-2
    #     audit), uniform-gate fallback ONLY in train (one-time warning);
    #     g ignored under head_type="single" (one-time warning). The
    #     forward(x, x_macro=None) contract is unchanged for all existing callers.
    def forward(self, x: torch.Tensor, x_macro: torch.Tensor = None,
                latent: torch.Tensor = None, g: torch.Tensor = None) -> tuple:
        # x: (B, T, F)
        # IT: latente CAFN OPZIONALE (B,T,d_latent) concatenato come feature-token
        #     extra (inverted transformer). latent=None → path byte-identico
        #     (parity con BLOCKER #1). Il modulo va costruito con n_features+=d_latent.
        # EN: OPTIONAL CAFN latent (B,T,d_latent) concatenated as extra feature
        #     tokens (inverted transformer). latent=None → byte-identical path
        #     (parity with BLOCKER #1). Build the module with n_features+=d_latent.
        if latent is not None:
            x = torch.cat([x, latent], dim=-1)
        _revin_stats = None
        if self.use_revin:
            x, _revin_stats = self.revin.normalize(x)
        xf = x.permute(0, 2, 1)                                 # (B, F, T)

        # IT: Clip adattivo per-feature (bounds da X_train, fallback ±500): no overflow fp16.
        # EN: Adaptive per-feature clip (bounds from X_train, fallback ±500): no fp16 overflow.
        # xf: (B, F, T) → clip_lo: (F,) → unsqueeze(-1) → (F, 1) per broadcast corretto su T
        xf = xf.clamp(self.clip_lo.unsqueeze(-1), self.clip_hi.unsqueeze(-1))

        # IT: Temporal patching: riduce la risoluzione temporale.
        # EN: Temporal patching: reduces temporal resolution.
        if self.patch_size > 1:
            xf = F.adaptive_avg_pool1d(xf, self.T_eff)          # (B, F, T_eff)

        xf5  = F.adaptive_avg_pool1d(xf, self.T5)               # (B, F, T5)
        xf15 = F.adaptive_avg_pool1d(xf, self.T15)              # (B, F, T15)

        e1 = self.embed1(xf)
        e2 = self.embed2(xf5)
        e3 = self.embed3(xf15)
        h  = self.scale_merge(torch.cat([e1, e2, e3], dim=-1))  # (B, F, d)

        h = h + self.type_emb(self.feature_types)               # broadcast (F, d)

        # IT: Positional embedding learnable (prima del prepend del macro token).
        # EN: Learnable positional embedding (before prepending the macro token).
        h = h + self.pos_emb(self._pos_ids)                     # (B, F, d)

        has_ctx = self.n_macro > 0 and x_macro is not None
        if has_ctx:
            ctx = self.macro_proj(x_macro).unsqueeze(1)         # (B, 1, d)
            h   = torch.cat([ctx, h], dim=1)                    # (B, F+1, d)

        for layer in self.layers:
            h = layer(h)

        if has_ctx:
            h = h[:, 1:, :]                                     # IT: rimuove macro token | EN: drops macro token

        h_pooled  = self.out_norm(h.mean(dim=1))                # (B, d)
        h_feature = h_pooled                                    # alias per leggibilità

        # ── Output computation ───────────────────────────────────────────────
        if self.head_type == "regime_moe":
            # IT: A3 — gate esterno causale. g=None in EVAL → RuntimeError (audit
            #     MAJOR-2): il modello è addestrato CONDIZIONATO sul gate — un
            #     entry-point che non lo passa (03/04/04b/ensemble) valuterebbe
            #     in covariate shift silenzioso; il burn-in è già gestito a monte
            #     da build_regime_gate (righe uniformi nel tensore). Fallback
            #     uniforme SOLO in train-mode (warning una-tantum). use_revin è
            #     vietato in ctor → nessun path RevIN qui.
            # EN: A3 — external causal gate. g=None in EVAL → RuntimeError
            #     (MAJOR-2 audit): the model is trained CONDITIONED on the gate —
            #     an entry-point not passing it (03/04/04b/ensemble) would
            #     evaluate under silent covariate shift; burn-in is already
            #     handled upstream by build_regime_gate (uniform rows in the
            #     tensor). Uniform fallback ONLY in train mode (one-time
            #     warning). use_revin is forbidden in the ctor → no RevIN here.
            if g is None:
                if not self.training:
                    raise RuntimeError(
                        "QuantiTransformer(regime_moe): g=None in eval — il gate "
                        "regime è input OBBLIGATORIO in inference (passare "
                        "g=regime_prob_0/1/2 da build_regime_gate; pattern del "
                        "guard interval/horizon) / g=None in eval — the regime "
                        "gate is a MANDATORY inference input"
                    )
                if not self._warned_uniform_gate:
                    log.warning(
                        "QuantiTransformer(regime_moe): g=None in train → gate "
                        "uniforme (1/3,1/3,1/3). Passare g=regime_prob_0/1/2 per "
                        "il mixing per-regime. / g=None in train → uniform gate; "
                        "pass g=regime_prob_0/1/2 for per-regime mixing."
                    )
                    self._warned_uniform_gate = True
                g = torch.full(
                    (h_feature.shape[0], N_REGIMES), 1.0 / N_REGIMES,
                    device=h_feature.device, dtype=h_feature.dtype,
                )
            head_out = self.regime_head(h_feature, g)
            if self.use_multitask:
                # IT: dir_head CONDIVISA tra i regimi (stesso precedente del MoE
                #     appreso): equivale alla media pesata di logits identici.
                # EN: dir_head SHARED across regimes (same precedent as the
                #     learned MoE): equivalent to the gate-weighted average of
                #     identical logits.
                return (*head_out, self.dir_head(h_feature))
            return head_out

        # IT: head_type="single" — g non pertinente: ignorato con warning una-tantum.
        # EN: head_type="single" — g not applicable: ignored with a one-time warning.
        if g is not None and not self._warned_gate_ignored:
            log.warning(
                "QuantiTransformer(single): parametro g ignorato (head_type != "
                "'regime_moe') / g argument ignored (head_type != 'regime_moe')."
            )
            self._warned_gate_ignored = True

        if self.n_output_experts > 1:
            gate_w      = F.softmax(self.expert_gate(h_feature), dim=-1)          # (B, E)
            expert_outs = torch.stack(
                [eh(h_feature) for eh in self.expert_heads], dim=1
            )                                                                       # (B, E, out)
            h_out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=1)                # (B, out)

            if self.loss_type == "quantile":
                quantile_preds = h_out
                if self.use_revin:
                    quantile_preds = self.revin.denormalize_mu(quantile_preds, _revin_stats)
                if self.use_multitask:
                    dir_logits = self.dir_head(h_feature)
                    return (quantile_preds, dir_logits)
                return (quantile_preds,)
            else:
                mu       = h_out[:, 0]
                log_sig2 = h_out[:, 1]
                log_nu   = h_out[:, 2]
                if self.use_revin:
                    mu       = self.revin.denormalize_mu(mu, _revin_stats)
                    log_sig2 = self.revin.denormalize_log_var(log_sig2, _revin_stats)
                if self.use_multitask:
                    dir_logits = self.dir_head(h_feature)
                    return (mu, log_sig2, log_nu, dir_logits)
                return (mu, log_sig2, log_nu)

        elif self.loss_type == "quantile":
            quantile_preds = self.quantile_head(h_feature)
            if self.use_revin:
                quantile_preds = self.revin.denormalize_mu(quantile_preds, _revin_stats)
            if self.use_multitask:
                dir_logits = self.dir_head(h_feature)
                return (quantile_preds, dir_logits)
            return (quantile_preds,)

        else:
            mu      = self.out_mu(h_feature).squeeze(-1)
            log_sig = self.out_logsig2(h_feature).squeeze(-1)
            log_nu  = self.out_lognu(h_feature).squeeze(-1)
            if self.use_revin:
                mu      = self.revin.denormalize_mu(mu, _revin_stats)
                log_sig = self.revin.denormalize_log_var(log_sig, _revin_stats)
            if self.use_multitask:
                dir_logits = self.dir_head(h_feature)
                return (mu, log_sig, log_nu, dir_logits)
            return (mu, log_sig, log_nu)

    # IT: Inferenza eval-mode → dict numpy (mu/sigma/nu o quantili).
    #     g opzionale in coda (regime_moe): pass-through al forward, None = legacy.
    # EN: Eval-mode inference → numpy dict (mu/sigma/nu or quantiles).
    #     Optional trailing g (regime_moe): passed through to forward, None = legacy.
    @torch.no_grad()
    def predict(self, x: torch.Tensor, x_macro: torch.Tensor = None,
                g: torch.Tensor = None) -> dict:
        self.eval()
        out = self.forward(x, x_macro, g=g)
        if self.loss_type == "quantile":
            quantile_preds = out[0]
            quantile_preds, _ = quantile_preds.sort(dim=-1)
            sigma = quantile_preds[:, 4] - quantile_preds[:, 0]
            return {
                "mu":        quantile_preds[:, 2].cpu().numpy(),
                "sigma":     sigma.cpu().numpy(),
                "quantiles": quantile_preds.cpu().numpy(),
            }
        else:
            mu, ls2, lnu = out[0], out[1], out[2]
            sigma, nu    = student_t_params(ls2, lnu)
            return {"mu": mu.cpu().numpy(), "sigma": sigma.cpu().numpy(), "nu": nu.cpu().numpy()}

    # IT: MC Dropout: N pass con dropout attivo → stima incertezza epistemica.
    # EN: MC Dropout: N passes with dropout on → estimates epistemic uncertainty.
    def predict_with_uncertainty(self, x: torch.Tensor, x_macro: torch.Tensor = None,
                                  n_samples: int = 20, g: torch.Tensor = None) -> dict:
        """MC Dropout: N forward pass con dropout attivo → epistemic uncertainty."""
        self.train()
        mus, sigmas, nus = [], [], []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x, x_macro, g=g)
                if self.loss_type == "quantile":
                    qp = out[0]
                    qp, _ = qp.sort(dim=-1)
                    mus.append(qp[:, 2])
                    sigmas.append(qp[:, 4] - qp[:, 0])
                    nus.append(None)
                else:
                    mu_i, ls2_i, lnu_i = out[0], out[1], out[2]
                    sig_i, nu_i        = student_t_params(ls2_i, lnu_i)
                    mus.append(mu_i)
                    sigmas.append(sig_i)
                    nus.append(nu_i)
        self.eval()
        # Single GPU→CPU transfer via torch.stack
        mus_arr    = torch.stack(mus, dim=0).cpu().numpy()
        sigmas_arr = torch.stack(sigmas, dim=0).cpu().numpy()
        if nus[0] is None:
            nus_arr = np.full_like(mus_arr, float("nan"))
        else:
            nus_arr = torch.stack(nus, dim=0).cpu().numpy()
        mu_mean    = mus_arr.mean(axis=0); mu_std = mus_arr.std(axis=0)
        sig_mean   = sigmas_arr.mean(axis=0)
        nu_mean    = nus_arr.mean(axis=0)
        sig_total  = np.sqrt(sig_mean**2 + mu_std**2)
        # NOTA (audit #19, 2026-05-24): confidence è scale-invariant — ratio mu_std/sig_mean.
        # In z-score o raw produce lo stesso valore. Safe da usare ovunque.
        # sig_total NON è scale-invariant: va denormalizzato se confrontato con scale assolute.
        confidence = np.clip(1.0 - mu_std / (sig_mean + 1e-9), 0.0, 1.0)
        return {"mu": mu_mean, "mu_std": mu_std, "sigma": sig_mean,
                "sigma_total": sig_total, "nu": nu_mean, "confidence_score": confidence}


# IT: Re-export pubblici: forecast, ensemble e architetture extra dal package model.
# EN: Public re-exports: forecast, ensemble and extra architectures from the model package.
from quantsys.model.forecast import monte_carlo_forecast, summarize_forecast  # noqa
from quantsys.model.ensemble import EnsembleModel  # noqa
from quantsys.model.tcn_mamba import QuantTCNMamba  # noqa
from quantsys.model.nhits import QuantNHiTS  # noqa
from quantsys.model.cafn import CausalAttentionFlowNetwork  # noqa
