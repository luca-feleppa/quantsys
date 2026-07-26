"""
CAFN — Causal Attention Flow Network.

IT: Layer di coordinamento a monte dei tre modelli predittivi del progetto
    (iTransformer, TCN+Mamba, N-HiTS). Riceve il tensore feature di mercato
    `[B, T, F]`, lo **filtra** (gate di denoising per-feature), ne estrae una
    **rappresentazione latente** causale `[B, T, d_latent]` tramite self-attention
    a **maschera strettamente causale** (il timestep t vede solo ≤ t → nessun
    lookahead), e produce in più una **penalità causale** scalare che il loop di
    training congiunto somma alla loss per *stabilizzare le relazioni causali*.

    ⚠ ONESTÀ SCIENTIFICA (TEORIA.md §12): la "penalità causale" è un **regolarizzatore**
    (prossimità temporale + stabilità del pattern di attenzione), NON una garanzia
    di causalità in senso do-calculus / Granger. Inoltre il latente si addestra
    SUL TENSORE CANONICO 104-feature (che ha storia 2019→oggi); i dati Deribit
    grezzi (greche/book/IV) sono raccolti SOLO in avanti (giorni di storia) e
    possono entrare unicamente come canale `extra` OPZIONALE futuro — mai come
    input storico di training (sarebbe lookahead / dataset inesistente).

EN: Coordination layer upstream of the project's three forecasting models
    (iTransformer, TCN+Mamba, N-HiTS). It takes the market feature tensor
    `[B, T, F]`, **filters** it (per-feature denoising gate), extracts a causal
    **latent representation** `[B, T, d_latent]` via **strictly causally-masked**
    self-attention (timestep t only sees ≤ t → no lookahead), and additionally
    returns a scalar **causal penalty** that the joint training loop adds to the
    loss to *stabilize causal relationships*.

    ⚠ SCIENTIFIC HONESTY (TEORIA.md §12): the "causal penalty" is a **regularizer**
    (temporal proximity + attention-pattern stability), NOT a do-calculus / Granger
    causality guarantee. Moreover the latent is trained ON THE CANONICAL 104-feature
    tensor (which has history 2019→today); raw Deribit data (greeks/book/IV) is
    forward-collected ONLY (days of history) and may enter solely as an OPTIONAL
    future `extra` channel — never as a historical training input (that would be
    lookahead / a non-existent dataset).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Blocco attenzione a flusso causale · Causal flow attention block ──────────
class CausalFlowAttention(nn.Module):
    # IT: multi-head self-attention con maschera strettamente causale (lower-tri).
    #     Restituisce (output, penalità causale): la penalità combina (1) prossimità
    #     — penalizza la massa di attenzione sul passato LONTANO (bias verso la
    #     causazione prossimale) e (2) stabilità — penalizza i salti del pattern di
    #     attenzione fra timestep adiacenti (relazioni causali "stabili").
    # EN: multi-head self-attention with a strictly causal (lower-tri) mask.
    #     Returns (output, causal penalty): the penalty combines (1) proximity —
    #     penalizes attention mass on the FAR past (bias toward proximal causation)
    #     and (2) stability — penalizes jumps of the attention pattern between
    #     adjacent timesteps ("stable" causal relationships).
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                         batch_first=True)

    def forward(self, h: torch.Tensor):
        # h: (B, T, d_model)
        T = h.shape[1]
        # IT: maschera bool (True = vietato) — t può attendere solo s ≤ t.
        # EN: bool mask (True = disallowed) — t may only attend to s ≤ t.
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device),
                          diagonal=1)
        out, attn_w = self.mha(h, h, h, attn_mask=mask, need_weights=True,
                               average_attn_weights=True)        # attn_w: (B, T, T)
        penalty = self._causal_flow_penalty(attn_w)
        return out, penalty

    @staticmethod
    def _causal_flow_penalty(A: torch.Tensor) -> torch.Tensor:
        # IT: A=(B,T,T), riga t è distribuzione su s≤t (somma 1 sui non-mascherati).
        # EN: A=(B,T,T), row t is a distribution over s≤t (sums to 1 over unmasked).
        B, T, _ = A.shape
        idx = torch.arange(T, device=A.device, dtype=A.dtype)
        # IT: lag normalizzato (t-s)/(T-1) ≥ 0 sul triangolo causale.
        # EN: normalized lag (t-s)/(T-1) ≥ 0 on the causal triangle.
        lag = (idx[:, None] - idx[None, :]).clamp(min=0.0) / max(T - 1, 1)
        proximity = (A * lag).sum(dim=-1).mean()                 # penalizza passato lontano / penalize far past
        if T > 1:
            stability = (A[:, 1:, :] - A[:, :-1, :]).abs().mean()  # penalizza salti / penalize jumps
        else:
            stability = A.new_zeros(())
        return proximity + stability


# ─── Rete CAFN · CAFN network ──────────────────────────────────────────────────
class CausalAttentionFlowNetwork(nn.Module):
    # IT: Filtro → proiezione → stack di blocchi causali → latente per-timestep.
    #     `forward(x, extra=None) -> (latent [B,T,d_latent], causal_penalty scalar)`.
    #     `extra` = canale OPZIONALE (es. feature Deribit forward-collected): se
    #     fornito viene concatenato a `x` — il modulo va costruito con
    #     `n_features = F + F_extra` di conseguenza.
    # EN: Filter → projection → stack of causal blocks → per-timestep latent.
    #     `forward(x, extra=None) -> (latent [B,T,d_latent], causal_penalty scalar)`.
    #     `extra` = OPTIONAL channel (e.g. forward-collected Deribit features): if
    #     provided it is concatenated to `x` — build the module with
    #     `n_features = F + F_extra` accordingly.
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_latent: int = 16, dropout: float = 0.1,
                 max_len: int = 512, ffn_mult: int = 2):
        super().__init__()
        self.n_features = n_features
        self.d_latent = d_latent

        # IT: FILTRO di denoising — LayerNorm + gate per-feature sigmoide: scala le
        #     feature grezze (rumorose) prima della proiezione → "segnale pulito".
        # EN: denoising FILTER — LayerNorm + sigmoid per-feature gate: scales raw
        #     (noisy) features before projection → "clean signal".
        self.in_norm = nn.LayerNorm(n_features)
        self.feature_gate = nn.Parameter(torch.zeros(n_features))   # sigmoid(0)=0.5 init

        self.input_proj = nn.Linear(n_features, d_model)
        # IT: positional embedding learnable (causale → l'ordine temporale conta).
        # EN: learnable positional embedding (causal → temporal order matters).
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.attn_layers = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.ffn = nn.ModuleList()
        for _ in range(n_layers):
            self.norm1.append(nn.LayerNorm(d_model))
            self.attn_layers.append(CausalFlowAttention(d_model, n_heads, dropout))
            self.norm2.append(nn.LayerNorm(d_model))
            self.ffn.append(nn.Sequential(
                nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(ffn_mult * d_model, d_model),
            ))
        self.drop = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_latent)

    def forward(self, x: torch.Tensor, extra: torch.Tensor = None):
        # x: (B, T, F)  ·  extra: (B, T, F_extra) opzionale / optional
        if extra is not None:
            x = torch.cat([x, extra], dim=-1)
        B, T, Fdim = x.shape
        if Fdim != self.n_features:
            raise ValueError(
                f"CAFN n_features={self.n_features} ma input ha {Fdim} feature "
                f"(hai passato `extra` senza ricostruire il modulo? / passed `extra` "
                f"without rebuilding the module?)")
        if T > self.pos_emb.shape[1]:
            raise ValueError(f"T={T} supera max_len={self.pos_emb.shape[1]}")

        # IT: filtro → gate per-feature (sigmoid) → proiezione + positional.
        # EN: filter → per-feature gate (sigmoid) → projection + positional.
        x = self.in_norm(x) * torch.sigmoid(self.feature_gate)
        h = self.input_proj(x) + self.pos_emb[:, :T, :]
        h = self.drop(h)

        total_penalty = h.new_zeros(())
        for n1, attn, n2, ffn in zip(self.norm1, self.attn_layers,
                                     self.norm2, self.ffn):
            a, pen = attn(n1(h))
            h = h + a
            total_penalty = total_penalty + pen
            h = h + ffn(n2(h))

        latent = self.out_proj(self.out_norm(h))                 # (B, T, d_latent)
        n_layers = max(len(self.attn_layers), 1)
        return latent, total_penalty / n_layers

    @torch.no_grad()
    def encode(self, x: torch.Tensor, extra: torch.Tensor = None) -> torch.Tensor:
        # IT: solo il latente (eval), utile per inferenza/diagnostica.
        # EN: latent only (eval), handy for inference/diagnostics.
        self.eval()
        latent, _ = self.forward(x, extra)
        return latent
