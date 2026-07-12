"""
IT: Test A3 — Regime-MoE (mixture-of-universes), 2026-07-12. CPU-only, tensori
    sintetici, NESSUN checkpoint/training. Copre: (a) inerzia del path
    head_type="single" (default assente = bit-identico); (b) gate one-hot →
    output = testa k; (c) gate uniforme + teste identiche → output = testa
    singola; (d) legge della varianza totale (σ²_mix ≥ Σ g_k σ²_k); (e) quantili
    mixati monotoni; più guard-rail (n_output_experts>1, use_revin, fallback
    g=None) e allineamento causale del gate builder.
EN: A3 tests — Regime-MoE (mixture-of-universes), 2026-07-12. CPU-only,
    synthetic tensors, NO checkpoints/training. Covers: (a) inertia of the
    head_type="single" path (absent default = bit-identical); (b) one-hot gate →
    output = head k; (c) uniform gate + identical heads → output = single head;
    (d) total variance law (σ²_mix ≥ Σ g_k σ²_k); (e) mixed quantiles monotone;
    plus guard rails (n_output_experts>1, use_revin, g=None fallback) and the
    causal alignment of the gate builder.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from quantsys.model import (
    N_REGIMES,
    QUANTILES,
    QuantiTransformer,
    RegimeMoEHead,
    _softplus_inverse,
)

# IT: tutto su CPU — la GPU è riservata ai processi live (vincolo di sessione).
# EN: everything on CPU — the GPU is reserved for the live processes.
DEVICE = torch.device("cpu")

B, T, FEAT, DMODEL = 4, 16, 8, 16


# IT: factory deterministica di QuantiTransformer piccoli (dropout=0, eval mode).
# EN: deterministic factory of small QuantiTransformer models (dropout=0, eval mode).
def _make_model(head_type=None, loss_type="quantile", seed=7, **kw):
    torch.manual_seed(seed)
    kwargs = dict(
        n_features=FEAT, T=T, n_dynamic=6, n_macro=0,
        d_model=DMODEL, n_heads=2, n_layers=1, dropout=0.0,
        loss_type=loss_type, use_multitask=False, n_output_experts=1,
    )
    if head_type is not None:
        kwargs["head_type"] = head_type
    kwargs.update(kw)
    return QuantiTransformer(**kwargs).to(DEVICE).eval()


# IT: input sintetico deterministico (B,T,F).
# EN: deterministic synthetic input (B,T,F).
def _make_x(seed=99):
    torch.manual_seed(seed)
    return torch.randn(B, T, FEAT, device=DEVICE)


# IT: gate one-hot sul regime k, shape (n,3).
# EN: one-hot gate on regime k, shape (n,3).
def _onehot(k: int, n: int) -> torch.Tensor:
    g = torch.zeros(n, N_REGIMES, device=DEVICE)
    g[:, k] = 1.0
    return g


# ─────────────────────────────────────────────────────────────────────────────
# (a) IT: Inerzia — default (chiave assente) ≡ head_type="single", bit-identico.
#     EN: Inertia — default (key absent) ≡ head_type="single", bit-identical.
# ─────────────────────────────────────────────────────────────────────────────
class TestSingleHeadInertia:

    @pytest.mark.parametrize("loss_type", ["quantile", "t_student"])
    def test_default_equals_explicit_single_bitwise(self, loss_type):
        # IT: stesso seed → stessa costruzione → stessi output BIT-identici:
        #     il param head_type di default non consuma RNG né crea moduli.
        # EN: same seed → same construction → BIT-identical outputs: the default
        #     head_type param consumes no RNG and creates no modules.
        m_default  = _make_model(head_type=None,     loss_type=loss_type, seed=7)
        m_explicit = _make_model(head_type="single", loss_type=loss_type, seed=7)
        x = _make_x()
        with torch.no_grad():
            out_d = m_default(x)
            out_e = m_explicit(x)
        for a, b in zip(out_d, out_e):
            assert torch.equal(a, b)

    def test_default_state_dict_has_no_regime_keys(self):
        # IT: nessun parametro nuovo nel path default → checkpoint storici intatti.
        # EN: no new parameter on the default path → legacy checkpoints intact.
        m = _make_model(head_type=None, loss_type="quantile")
        assert not any("regime" in k for k in m.state_dict())

    def test_single_ignores_gate_bitwise(self):
        # IT: g passato a un modello single → ignorato (output bit-identico).
        # EN: g passed to a single-head model → ignored (bit-identical output).
        m = _make_model(head_type="single", loss_type="quantile")
        x = _make_x()
        with torch.no_grad():
            out_no_g = m(x)[0]
            out_g    = m(x, g=_onehot(1, B))[0]
        assert torch.equal(out_no_g, out_g)


# ─────────────────────────────────────────────────────────────────────────────
# (b) IT: Gate one-hot su k → output = testa k (livello head e livello modello).
#     EN: One-hot gate on k → output = head k (head level and model level).
# ─────────────────────────────────────────────────────────────────────────────
class TestOneHotGate:

    def test_onehot_quantile_equals_head_k(self):
        torch.manual_seed(3)
        head = RegimeMoEHead(DMODEL, loss_type="quantile")
        # IT: pesi grandi → quantili raw non ordinati → il sort deve agire.
        # EN: large weights → unsorted raw quantiles → the sort must kick in.
        for hd in head.heads:
            torch.nn.init.normal_(hd.weight, std=1.0)
            torch.nn.init.normal_(hd.bias,   std=1.0)
        h = torch.randn(5, DMODEL)
        for k in range(N_REGIMES):
            with torch.no_grad():
                out = head(h, _onehot(k, 5))[0]
                ref, _ = head.heads[k](h).sort(dim=-1)
            assert torch.allclose(out, ref, atol=1e-6)

    def test_onehot_t_student_equals_head_k(self):
        torch.manual_seed(4)
        head = RegimeMoEHead(DMODEL, loss_type="t_student")
        for hd in head.heads:
            torch.nn.init.normal_(hd.weight, std=0.5)
            torch.nn.init.normal_(hd.bias,   std=0.5)
        h = torch.randn(5, DMODEL)
        for k in range(N_REGIMES):
            with torch.no_grad():
                mu, ls2, lnu = head(h, _onehot(k, 5))
                ref = head.heads[k](h)
            assert torch.allclose(mu,  ref[:, 0], atol=1e-6)
            # IT: ls2 è ri-codificato via softplus-inverse → confronto nello
            #     spazio NATURALE della varianza (dove vive il contratto).
            # EN: ls2 is re-encoded via softplus-inverse → compare in the NATURAL
            #     variance space (where the contract lives).
            var_mix = F.softplus(ls2) + 1e-6
            var_ref = F.softplus(ref[:, 1]) + 1e-6
            assert torch.allclose(var_mix, var_ref, atol=1e-6)
            assert torch.allclose(lnu, ref[:, 2], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# (c) IT: Gate uniforme + teste identiche (state_dict copiato) → testa singola.
#     EN: Uniform gate + identical heads (copied state_dict) → single head.
# ─────────────────────────────────────────────────────────────────────────────
class TestUniformGateIdenticalHeads:

    def test_model_level_quantile_equals_single(self):
        # IT: stesso backbone (load_state_dict strict=False) + 3 teste = copia
        #     della quantile_head singola → il mix uniforme DEVE riprodurre la
        #     testa singola (a meno del re-sort monotono, applicato a entrambi).
        # EN: same backbone (load_state_dict strict=False) + 3 heads = copies of
        #     the single quantile_head → the uniform mix MUST reproduce the
        #     single head (up to the monotone re-sort, applied to both).
        m_single = _make_model(head_type="single",     loss_type="quantile", seed=11)
        m_moe    = _make_model(head_type="regime_moe", loss_type="quantile", seed=12)
        m_moe.load_state_dict(m_single.state_dict(), strict=False)
        for hd in m_moe.regime_head.heads:
            hd.load_state_dict(m_single.quantile_head.state_dict())
        x = _make_x()
        g = torch.full((B, N_REGIMES), 1.0 / N_REGIMES)
        with torch.no_grad():
            qp_moe    = m_moe(x, g=g)[0]
            qp_single, _ = m_single(x)[0].sort(dim=-1)
        assert torch.allclose(qp_moe, qp_single, atol=1e-6)

    def test_head_level_t_student_identical_heads(self):
        torch.manual_seed(5)
        head = RegimeMoEHead(DMODEL, loss_type="t_student")
        for hd in head.heads[1:]:
            hd.load_state_dict(head.heads[0].state_dict())
        h = torch.randn(6, DMODEL)
        g = torch.full((6, N_REGIMES), 1.0 / N_REGIMES)
        with torch.no_grad():
            mu, ls2, lnu = head(h, g)
            ref = head.heads[0](h)
        # IT: teste identiche → disagreement = 0 → σ²_mix = σ²_0 esattamente.
        # EN: identical heads → zero disagreement → σ²_mix = σ²_0 exactly.
        assert torch.allclose(mu, ref[:, 0], atol=1e-6)
        assert torch.allclose(F.softplus(ls2) + 1e-6,
                              F.softplus(ref[:, 1]) + 1e-6, atol=1e-6)
        assert torch.allclose(lnu, ref[:, 2], atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# (d) IT: Legge della varianza totale — σ²_mix ≥ Σ g_k σ²_k (disagreement ≥ 0).
#     EN: Total variance law — σ²_mix ≥ Σ g_k σ²_k (disagreement ≥ 0).
# ─────────────────────────────────────────────────────────────────────────────
class TestTotalVarianceLaw:

    def test_mixture_variance_dominates_within(self):
        torch.manual_seed(6)
        head = RegimeMoEHead(DMODEL, loss_type="t_student")
        # IT: teste ben separate in μ → termine between strettamente > 0.
        # EN: well-separated μ heads → strictly positive between term.
        for i, hd in enumerate(head.heads):
            torch.nn.init.normal_(hd.weight, std=0.5)
            with torch.no_grad():
                hd.bias[0] = float(i) * 2.0     # IT: μ separati | EN: separated μ
        h = torch.randn(32, DMODEL)
        # IT: gate random sul simplesso (Dirichlet-like via softmax).
        # EN: random simplex gate (Dirichlet-like via softmax).
        g = torch.softmax(torch.randn(32, N_REGIMES), dim=-1)
        with torch.no_grad():
            mu, ls2, _ = head(h, g)
            var_mix = F.softplus(ls2) + 1e-6
            outs  = torch.stack([hd(h) for hd in head.heads], dim=1)  # (B,K,3)
            mu_k  = outs[..., 0]
            var_k = F.softplus(outs[..., 1]) + 1e-6
            within = (g * var_k).sum(dim=1)
            between = (g * (mu_k - mu.unsqueeze(1)) ** 2).sum(dim=1)
        # IT: σ²_mix = within + between, con between ≥ 0.
        # EN: σ²_mix = within + between, with between ≥ 0.
        assert (var_mix >= within - 1e-6).all()
        assert torch.allclose(var_mix, within + between, atol=1e-5)
        assert (between > 0).any()

    def test_softplus_inverse_roundtrip(self):
        # IT: l'inversa deve invertire softplus su tutto il range utile di σ².
        # EN: the inverse must invert softplus over the useful σ² range.
        y = torch.tensor([1e-8, 1e-4, 0.1, 1.0, 5.0, 30.0])
        x = _softplus_inverse(y)
        assert torch.allclose(F.softplus(x), y, rtol=1e-5, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# (e) IT: Quantili mixati monotoni non-decrescenti (re-sort di sicurezza).
#     EN: Mixed quantiles monotone non-decreasing (safety re-sort).
# ─────────────────────────────────────────────────────────────────────────────
class TestQuantileMonotonicity:

    def test_mixed_quantiles_are_sorted(self):
        torch.manual_seed(8)
        head = RegimeMoEHead(DMODEL, loss_type="quantile")
        # IT: pesi grandi → output raw quasi certamente NON ordinato per livello.
        # EN: large weights → raw output almost surely NOT level-ordered.
        for hd in head.heads:
            torch.nn.init.normal_(hd.weight, std=2.0)
            torch.nn.init.normal_(hd.bias,   std=2.0)
        h = torch.randn(64, DMODEL)
        g = torch.softmax(torch.randn(64, N_REGIMES), dim=-1)
        with torch.no_grad():
            qp = head(h, g)[0]
        assert qp.shape == (64, len(QUANTILES))
        assert (qp.diff(dim=-1) >= 0).all()

    def test_model_level_shapes_and_fallback_uniform(self):
        # IT: g=None → fallback gate uniforme ≡ gate uniforme esplicito;
        #     shape del contratto invariata (B, Q).
        # EN: g=None → uniform-gate fallback ≡ explicit uniform gate;
        #     contract shape unchanged (B, Q).
        m = _make_model(head_type="regime_moe", loss_type="quantile", seed=21)
        x = _make_x()
        g_uni = torch.full((B, N_REGIMES), 1.0 / N_REGIMES)
        with torch.no_grad():
            out_none = m(x)[0]
            out_uni  = m(x, g=g_uni)[0]
        assert out_none.shape == (B, len(QUANTILES))
        assert torch.allclose(out_none, out_uni, atol=1e-7)

    def test_model_level_t_student_contract(self):
        # IT: contratto (mu, ls2, lnu) con shape (B,) — invariato verso l'esterno.
        # EN: (mu, ls2, lnu) contract with shape (B,) — externally unchanged.
        m = _make_model(head_type="regime_moe", loss_type="t_student", seed=22)
        x = _make_x()
        with torch.no_grad():
            out = m(x, g=_onehot(2, B))
        assert len(out) == 3
        assert all(t.shape == (B,) for t in out)


# ─────────────────────────────────────────────────────────────────────────────
# IT: Guard-rail del ctor + builder del gate causale.
# EN: Ctor guard rails + causal gate builder.
# ─────────────────────────────────────────────────────────────────────────────
class TestGuardsAndGateBuilder:

    def test_regime_moe_rejects_learned_moe(self):
        # IT: MoE appreso (n_output_experts>1) e regime-MoE mutuamente esclusivi.
        # EN: learned MoE (n_output_experts>1) and regime-MoE mutually exclusive.
        with pytest.raises(ValueError):
            _make_model(head_type="regime_moe", n_output_experts=2)

    def test_regime_moe_rejects_revin(self):
        with pytest.raises(ValueError):
            _make_model(head_type="regime_moe", use_revin=True, revin_target_idx=0)

    def test_unknown_head_type_rejected(self):
        with pytest.raises(ValueError):
            _make_model(head_type="banana")

    def test_gate_wrong_width_rejected(self):
        head = RegimeMoEHead(DMODEL, loss_type="quantile")
        with pytest.raises(ValueError):
            head(torch.randn(2, DMODEL), torch.ones(2, 4) / 4.0)

    def test_build_regime_gate_backward_and_burnin(self, tmp_path):
        # IT: parquet sintetico orario; verifica allineamento BACKWARD (ultimo
        #     regime noto ≤ t, mai forward = causale), fallback uniforme pre-inizio
        #     (burn-in) e ripristino dell'ordine originale dei sample.
        # EN: synthetic hourly parquet; checks BACKWARD alignment (last known
        #     regime ≤ t, never forward = causal), uniform fallback before start
        #     (burn-in) and restoration of the original sample ordering.
        import pandas as pd
        from quantsys.model.regime_gate import build_regime_gate

        idx = pd.date_range("2026-01-01 00:00", periods=4, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "regime_prob_0": [1.0, 0.0, 0.2, 0.0],
                "regime_prob_1": [0.0, 1.0, 0.5, 0.0],
                "regime_prob_2": [0.0, 0.0, 0.3, 1.0],
            },
            index=idx,
        )
        df.index.name = "open_time"
        p = tmp_path / "regime_probs.parquet"
        df.to_parquet(p)

        ts = np.array(
            [
                "2026-01-01T02:30",   # IT: tra 02:00 e 03:00 → prende 02:00 (backward) | EN: between rows → takes 02:00
                "2025-12-31T23:00",   # IT: prima dell'inizio → uniforme | EN: before start → uniform
                "2026-01-01T01:00",   # IT: match esatto | EN: exact match
            ],
            dtype="datetime64[ms]",
        )
        G = build_regime_gate(ts, parquet_path=str(p))
        assert G.shape == (3, 3) and G.dtype == np.float32
        # IT: righe sul simplesso.
        # EN: rows on the simplex.
        assert np.allclose(G.sum(axis=1), 1.0, atol=1e-6)
        # IT: 02:30 → riga delle 02:00 (mai quella delle 03:00 = lookahead).
        # EN: 02:30 → the 02:00 row (never the 03:00 one = lookahead).
        assert np.allclose(G[0], [0.2, 0.5, 0.3], atol=1e-6)
        # IT: pre-inizio → uniforme (burn-in contract).
        # EN: before start → uniform (burn-in contract).
        assert np.allclose(G[1], [1 / 3] * 3, atol=1e-6)
        # IT: match esatto + ordine originale preservato.
        # EN: exact match + original ordering preserved.
        assert np.allclose(G[2], [0.0, 1.0, 0.0], atol=1e-6)

    def test_multitask_appends_dir_logits(self):
        # IT: use_multitask + regime_moe → (qp, dir_logits): contratto identico
        #     al path single (dir_head condivisa tra i regimi).
        # EN: use_multitask + regime_moe → (qp, dir_logits): same contract as the
        #     single path (dir_head shared across regimes).
        m = _make_model(head_type="regime_moe", loss_type="quantile",
                        use_multitask=True, seed=33)
        x = _make_x()
        with torch.no_grad():
            out = m(x, g=_onehot(0, B))
        assert len(out) == 2
        assert out[0].shape == (B, len(QUANTILES))
        assert out[1].shape == (B, 3)
