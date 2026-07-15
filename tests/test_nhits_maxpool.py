"""
tests/test_nhits_maxpool.py
===========================
Test per il blocco MaxPool parallelo in N-HiTS (A9 roadmap vol),
config-gated via `nhits_max_pool_block` (default OFF = bit-identico).

Copre:
  · inerzia del lever (OFF = zero parametri nuovi, init e forward bit-identici)
  · compatibilità checkpoint (state_dict OFF == design storico)
  · correttezza del ramo ON (shape, gradienti al jump_block, pool max reale)
  · fail-fast su pool_type ignoto

Esegui con:
  pytest tests/test_nhits_maxpool.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from quantsys.model.nhits import NHiTSBlock, QuantNHiTS


# IT: Iperparametri minimi per test rapidi su CPU.
# EN: Minimal hyperparameters for fast CPU tests.
KW = dict(n_features=12, T=32, d_model=16, hidden=24, n_stacks=2,
          pool_kernels=(4, 1), loss_type="t_student")


def _make(seed: int, **overrides) -> QuantNHiTS:
    torch.manual_seed(seed)
    return QuantNHiTS(**{**KW, **overrides})


# IT: TEST 1 — Inerzia: OFF (default) = stesso modello del design storico.
# EN: TEST 1 — Inertia: OFF (default) = same model as the historical design.
class TestMaxPoolInertia:

    # IT: Il default DEVE essere OFF e non creare il jump_block.
    # EN: The default MUST be OFF and create no jump_block.
    def test_default_is_off(self):
        m = _make(0)
        assert m.use_max_pool_block is False
        assert m.jump_block is None

    # IT: A parità di seed, esplicitare OFF non cambia né i parametri né l'init
    #     (nessun draw RNG extra): state_dict bit-identici → checkpoint compatibili.
    # EN: With the same seed, explicit OFF changes neither params nor init
    #     (no extra RNG draws): bit-identical state_dicts → compatible checkpoints.
    def test_state_dict_bit_identical_off(self):
        sd_a = _make(42).state_dict()
        sd_b = _make(42, use_max_pool_block=False).state_dict()
        assert sd_a.keys() == sd_b.keys()
        for k in sd_a:
            assert torch.equal(sd_a[k], sd_b[k]), k

    # IT: Forward OFF bit-identico (eval, no dropout stocastico).
    # EN: Bit-identical OFF forward (eval, no stochastic dropout).
    def test_forward_bit_identical_off(self):
        m_a, m_b = _make(7).eval(), _make(7, use_max_pool_block=False).eval()
        x = torch.randn(4, KW["T"], KW["n_features"], generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            out_a, out_b = m_a(x), m_b(x)
        for t_a, t_b in zip(out_a, out_b):
            assert torch.equal(t_a, t_b)


# IT: TEST 2 — Ramo ON: parametri nuovi, shape corrette, gradienti che fluiscono.
# EN: TEST 2 — ON branch: new params, correct shapes, flowing gradients.
class TestMaxPoolOn:

    def test_on_creates_max_block(self):
        m = _make(0, use_max_pool_block=True, max_pool_kernel=4)
        assert isinstance(m.jump_block, NHiTSBlock)
        assert isinstance(m.jump_block.pool, nn.MaxPool1d)
        assert m.jump_block.pool_kernel == 4

    # IT: Output (mu, ls2, lnu) shape (B,) come da forward contract.
    # EN: (mu, ls2, lnu) outputs with shape (B,) per the forward contract.
    def test_forward_contract_shapes(self):
        m = _make(0, use_max_pool_block=True).eval()
        x = torch.randn(5, KW["T"], KW["n_features"])
        with torch.no_grad():
            mu, ls2, lnu = m(x)
        for t in (mu, ls2, lnu):
            assert t.shape == (5,)
            assert torch.isfinite(t).all()

    # IT: Il jump_block partecipa alla loss: gradiente non nullo sui suoi pesi.
    # EN: The jump_block joins the loss: non-zero gradient on its weights.
    def test_gradient_flows_to_jump_block(self):
        m = _make(0, use_max_pool_block=True)
        x = torch.randn(8, KW["T"], KW["n_features"])
        mu, _, _ = m(x)
        mu.sum().backward()
        g = m.jump_block.forecast_head.weight.grad
        assert g is not None and g.abs().sum() > 0

    # IT: ON cambia davvero il forward rispetto a OFF (a pesi condivisi identici
    #     il ramo jump additivo deve spostare l'output).
    # EN: ON really changes the forward vs OFF (with identical shared weights the
    #     additive jump branch must move the output).
    def test_on_differs_from_off(self):
        m_off = _make(3).eval()
        m_on  = _make(3, use_max_pool_block=True).eval()
        # IT: copia i pesi condivisi OFF→ON (strict=False ignora il jump_block).
        # EN: copy shared weights OFF→ON (strict=False skips the jump_block).
        m_on.load_state_dict(m_off.state_dict(), strict=False)
        x = torch.randn(4, KW["T"], KW["n_features"])
        with torch.no_grad():
            mu_off, _, _ = m_off(x)
            mu_on,  _, _ = m_on(x)
        assert not torch.allclose(mu_off, mu_on)


# IT: TEST 3 — NHiTSBlock: pool_type max è MaxPool vero; fail-fast su typo.
# EN: TEST 3 — NHiTSBlock: pool_type max is real MaxPool; fail-fast on typos.
class TestBlockPoolType:

    def test_max_pool_takes_maximum(self):
        b = NHiTSBlock(input_len=8, d_model=1, hidden=4, pool_kernel=4,
                       pool_type="max")
        x = torch.tensor([[[1., 0., 9., 0., 0., 2., 0., 0.]]])  # (B=1, D=1, T=8)
        pooled = b.pool(x)
        assert torch.equal(pooled, torch.tensor([[[9., 2.]]]))

    def test_default_is_avg(self):
        b = NHiTSBlock(input_len=8, d_model=1, hidden=4, pool_kernel=4)
        assert isinstance(b.pool, nn.AvgPool1d)
        assert b.pool_type == "avg"

    def test_unknown_pool_type_fails_fast(self):
        with pytest.raises(ValueError, match="pool_type"):
            NHiTSBlock(input_len=8, d_model=1, hidden=4, pool_kernel=4,
                       pool_type="median")
