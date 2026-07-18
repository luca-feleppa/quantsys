"""
IT: Test A3 — Regime-MoE (mixture-of-universes), 2026-07-12. CPU-only, tensori
    sintetici, NESSUN checkpoint/training. Copre: (a) inerzia del path
    head_type="single" (default assente = bit-identico); (b) gate one-hot →
    output = testa k; (c) gate uniforme + teste identiche → output = testa
    singola; (d) legge della varianza totale (σ²_mix ≥ Σ g_k σ²_k); (e) quantili
    mixati monotoni; più guard-rail (n_output_experts>1, use_revin; g=None →
    raise in eval, fallback uniforme solo in train — audit MAJOR-2) e builder
    del gate: availability shift +1h (audit BLOCKER-1) + staleness bounded
    (audit MAJOR-1). NB (audit MINOR-3): il test di inerzia confronta due
    costruzioni del codice corrente — il gold standard resta un golden
    snapshot pre-diff; verifica manuale: nessun consumo RNG sul path single.
EN: A3 tests — Regime-MoE (mixture-of-universes), 2026-07-12. CPU-only,
    synthetic tensors, NO checkpoints/training. Covers: (a) inertia of the
    head_type="single" path (absent default = bit-identical); (b) one-hot gate →
    output = head k; (c) uniform gate + identical heads → output = single head;
    (d) total variance law (σ²_mix ≥ Σ g_k σ²_k); (e) mixed quantiles monotone;
    plus guard rails (n_output_experts>1, use_revin; g=None → raise in eval,
    uniform fallback in train only — MAJOR-2 audit) and the gate builder:
    +1h availability shift (BLOCKER-1 audit) + bounded staleness (MAJOR-1
    audit). NB (MINOR-3 audit): the inertia test compares two constructions of
    the current code — the gold standard remains a pre-diff golden snapshot;
    manually verified: no RNG consumption on the single path.
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

    def test_gate_none_in_eval_raises(self):
        # IT: audit MAJOR-2 — g=None in EVAL → RuntimeError: il gate è input
        #     obbligatorio in inference (covariate shift silenzioso altrimenti).
        # EN: MAJOR-2 audit — g=None in EVAL → RuntimeError: the gate is a
        #     mandatory inference input (silent covariate shift otherwise).
        m = _make_model(head_type="regime_moe", loss_type="quantile", seed=21)
        x = _make_x()
        with pytest.raises(RuntimeError, match="regime_moe"):
            with torch.no_grad():
                m(x)

    def test_gate_none_in_train_fallback_uniform(self):
        # IT: in TRAIN g=None → fallback gate uniforme ≡ gate uniforme esplicito
        #     (dropout=0 → deterministico); shape del contratto invariata (B, Q).
        # EN: in TRAIN g=None → uniform-gate fallback ≡ explicit uniform gate
        #     (dropout=0 → deterministic); contract shape unchanged (B, Q).
        m = _make_model(head_type="regime_moe", loss_type="quantile", seed=21)
        m.train()
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

    def test_build_regime_gate_availability_and_burnin(self, tmp_path):
        # IT: audit BLOCKER-1 — la riga etichettata `t` contiene la barra
        #     [t, t+1h) → disponibile SOLO a t+1h. Il builder shifta l'indice ad
        #     availability time: un sample ESATTAMENTE a t deve risolvere alla
        #     riga t−1h (mai alla riga t = lookahead del return futuro).
        #     Copre anche burn-in (pre-inizio → uniforme) e ordine originale.
        # EN: BLOCKER-1 audit — the row labeled `t` holds bar [t, t+1h) →
        #     available ONLY at t+1h. The builder shifts the index to
        #     availability time: a sample EXACTLY at t must resolve to the t−1h
        #     row (never to row t = future-return lookahead). Also covers
        #     burn-in (pre-start → uniform) and original ordering.
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

        # IT: MINOR-A (audit B1) — il fail-fast ora conta i fallback TOTALI
        #     (burn-in incluso): 5 sample freschi extra tengono la frazione bad
        #     a 2/10 = 20% (non oltre la soglia strict >20%).
        # EN: MINOR-A (B1 audit) — fail-fast now counts TOTAL fallbacks
        #     (burn-in included): 5 extra fresh samples keep the bad fraction
        #     at 2/10 = 20% (not beyond the strict >20% bound).
        ts = np.array(
            [
                "2026-01-01T02:30",   # IT: ultima disponibile ≤02:30 = riga 01:00 | EN: last available = the 01:00 row
                "2025-12-31T23:00",   # IT: prima dell'inizio → uniforme | EN: before start → uniform
                "2026-01-01T02:00",   # IT: REGRESSIONE lookahead: a t esatto → riga t−1h | EN: lookahead REGRESSION: at exact t → the t−1h row
                "2026-01-01T03:00",   # IT: riga 02:00 (disponibile a 03:00) | EN: the 02:00 row (available at 03:00)
                "2026-01-01T00:30",   # IT: nessuna riga disponibile (la 00:00 arriva a 01:00) → uniforme | EN: none available yet → uniform
                "2026-01-01T04:00", "2026-01-01T04:00", "2026-01-01T04:00",
                "2026-01-01T04:00", "2026-01-01T04:00",   # IT: filler freschi → riga 03:00 | EN: fresh fillers → the 03:00 row
            ],
            dtype="datetime64[ms]",
        )
        G = build_regime_gate(ts, parquet_path=str(p))
        assert G.shape == (10, 3) and G.dtype == np.float32
        # IT: righe sul simplesso.
        # EN: rows on the simplex.
        assert np.allclose(G.sum(axis=1), 1.0, atol=1e-6)
        # IT: 02:30 → riga 01:00 (la 02:00 diventa disponibile solo alle 03:00).
        # EN: 02:30 → the 01:00 row (02:00 becomes available only at 03:00).
        assert np.allclose(G[0], [0.0, 1.0, 0.0], atol=1e-6)
        # IT: pre-inizio → uniforme (burn-in contract).
        # EN: before start → uniform (burn-in contract).
        assert np.allclose(G[1], [1 / 3] * 3, atol=1e-6)
        # IT: sample a t=02:00 esatto → riga 01:00, MAI la 02:00 (=lookahead
        #     della barra [02:00,03:00) non ancora realizzata).
        # EN: sample at exact t=02:00 → the 01:00 row, NEVER 02:00 (=lookahead
        #     of the not-yet-realized [02:00,03:00) bar).
        assert np.allclose(G[2], [0.0, 1.0, 0.0], atol=1e-6)
        assert np.allclose(G[3], [0.2, 0.5, 0.3], atol=1e-6)
        # IT: 00:30 — la prima riga (00:00) è disponibile solo dalle 01:00.
        # EN: 00:30 — the first row (00:00) is available only from 01:00.
        assert np.allclose(G[4], [1 / 3] * 3, atol=1e-6)
        # IT: filler a 04:00 → riga 03:00 (ultima disponibile).
        # EN: fillers at 04:00 → the 03:00 row (last available).
        assert np.allclose(G[5:], [[0.0, 0.0, 1.0]] * 5, atol=1e-6)

    def test_build_regime_gate_staleness_bounded(self, tmp_path):
        # IT: audit MAJOR-1 — oltre max_age dall'ultima riga disponibile il gate
        #     DEVE tornare uniforme (mai last-known illimitato); coda stale oltre
        #     il 20% dei sample → fail-fast RuntimeError.
        # EN: MAJOR-1 audit — beyond max_age from the last available row the gate
        #     MUST fall back to uniform (never unbounded last-known); stale tail
        #     beyond 20% of samples → RuntimeError fail-fast.
        import pandas as pd
        from quantsys.model.regime_gate import build_regime_gate

        idx = pd.date_range("2026-01-01 00:00", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"regime_prob_0": [1.0] * 3, "regime_prob_1": [0.0] * 3,
             "regime_prob_2": [0.0] * 3}, index=idx)
        df.index.name = "open_time"
        p = tmp_path / "regime_probs.parquet"
        df.to_parquet(p)

        # IT: 4 sample freschi + 1 stale (61h dopo l'ultima riga, max_age=48h):
        #     stale → uniforme, freschi → riga nota; 1/5 = 20% NON supera il bound.
        # EN: 4 fresh samples + 1 stale one (61h past the last row, max_age=48h):
        #     stale → uniform, fresh → known row; 1/5 = 20% does NOT exceed the bound.
        ts = np.array(
            ["2026-01-01T01:00", "2026-01-01T02:00", "2026-01-01T03:00",
             "2026-01-01T04:00", "2026-01-03T16:00"], dtype="datetime64[ms]")
        G = build_regime_gate(ts, parquet_path=str(p), max_age="48h")
        assert np.allclose(G[:4], [[1.0, 0.0, 0.0]] * 4, atol=1e-6)
        assert np.allclose(G[4], [1 / 3] * 3, atol=1e-6)

        # IT: 3 sample stale su 4 (75% > 20%) → fail-fast.
        # EN: 3 stale samples out of 4 (75% > 20%) → fail-fast.
        ts_bad = np.array(
            ["2026-01-01T01:00", "2026-01-05T00:00", "2026-01-06T00:00",
             "2026-01-07T00:00"], dtype="datetime64[ms]")
        with pytest.raises(RuntimeError, match="fallback"):
            build_regime_gate(ts_bad, parquet_path=str(p), max_age="48h")

    def test_build_regime_gate_truncated_head_failfast(self, tmp_path):
        # IT: MINOR-A audit B1 (2026-07-18) — parquet TRONCATO IN TESTA (copertura
        #     che parte dopo la maggioranza dei sample): i fallback sono burn-in
        #     (NaT, zero stale) ma il gate sarebbe ~uniforme ovunque → il run
        #     misurerebbe un modello ~single-head, NON l'esperimento pre-registrato.
        #     Il fail-fast deve contare i fallback TOTALI, non solo gli stale.
        # EN: MINOR-A B1 audit (2026-07-18) — HEAD-TRUNCATED parquet (coverage
        #     starting after most samples): fallbacks are burn-in (NaT, zero
        #     stale) yet the gate would be ~uniform everywhere → the run would
        #     measure a ~single-head model, NOT the pre-registered experiment.
        #     The fail-fast must count TOTAL fallbacks, not just stale ones.
        import pandas as pd
        from quantsys.model.regime_gate import build_regime_gate

        idx = pd.date_range("2026-06-01 00:00", periods=3, freq="1h", tz="UTC")
        df = pd.DataFrame(
            {"regime_prob_0": [1.0] * 3, "regime_prob_1": [0.0] * 3,
             "regime_prob_2": [0.0] * 3}, index=idx)
        df.index.name = "open_time"
        p = tmp_path / "regime_probs.parquet"
        df.to_parquet(p)

        # IT: 3 sample su 4 PRIMA dell'inizio della copertura (75% burn-in > 20%).
        # EN: 3 of 4 samples BEFORE coverage starts (75% burn-in > 20%).
        ts = np.array(
            ["2026-01-01T00:00", "2026-02-01T00:00", "2026-03-01T00:00",
             "2026-06-01T02:00"], dtype="datetime64[ms]")
        with pytest.raises(RuntimeError, match="fallback"):
            build_regime_gate(ts, parquet_path=str(p))

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
