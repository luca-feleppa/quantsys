"""
Test CAFN — Causal Attention Flow Network + integrazione parity-safe.

IT: Copre (1) shape/penalità/gradiente del modulo, (2) maschera STRETTAMENTE
    causale (nessun leak futuro→passato nel latente), (3) PARITY: i tre modelli
    con `latent=None` sono bit-identici alla chiamata legacy (vincolo BLOCKER #1),
    (4) i modelli aumentati accettano il latente.
EN: Covers (1) module shape/penalty/gradient, (2) STRICTLY causal mask (no
    future→past leak in the latent), (3) PARITY: the three models with
    `latent=None` are bit-identical to the legacy call (BLOCKER #1 constraint),
    (4) augmented models accept the latent.
"""
import torch
import pytest

from quantsys.model import (
    CausalAttentionFlowNetwork, QuantiTransformer, QuantNHiTS, QuantTCNMamba,
)

B, T, F_, DL = 4, 16, 12, 8


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(B, T, F_)


@pytest.fixture
def cafn():
    torch.manual_seed(0)
    return CausalAttentionFlowNetwork(n_features=F_, d_model=32, n_heads=4,
                                      n_layers=2, d_latent=DL)


def test_shapes_and_penalty(cafn, x):
    latent, pen = cafn(x)
    assert latent.shape == (B, T, DL)
    assert pen.ndim == 0                 # penalità scalare / scalar penalty
    assert float(pen) >= 0.0             # penalità non-negativa / non-negative


def test_gradient_flows(cafn, x):
    latent, pen = cafn(x)
    (pen + latent.pow(2).mean()).backward()
    gnorm = sum(p.grad.abs().sum().item()
                for p in cafn.parameters() if p.grad is not None)
    assert gnorm > 0.0


def test_strictly_causal_no_future_leak(cafn, x):
    # IT: perturbare l'ULTIMO timestep non deve cambiare i latenti precedenti.
    # EN: perturbing the LAST timestep must not change earlier latents.
    cafn.eval()
    with torch.no_grad():
        base = cafn.encode(x.clone())
        x2 = x.clone()
        x2[:, T - 1, :] += 100.0
        pert = cafn.encode(x2)
    leak = (base[:, :T - 1] - pert[:, :T - 1]).abs().max().item()
    assert leak < 1e-4


def test_feature_filter_gate_present(cafn):
    # IT: il "filtro" di denoising = gate per-feature sigmoide.
    # EN: the denoising "filter" = sigmoid per-feature gate.
    assert hasattr(cafn, "feature_gate")
    assert cafn.feature_gate.shape == (F_,)


def test_extra_channel_dim_guard(x):
    # IT: passare `extra` senza ricostruire il modulo → ValueError esplicito.
    # EN: passing `extra` without rebuilding the module → explicit ValueError.
    c = CausalAttentionFlowNetwork(n_features=F_, d_model=16, n_layers=1, d_latent=4)
    with pytest.raises(ValueError):
        c(x, extra=torch.randn(B, T, 3))


def _make(arch, n_in):
    if arch == "itransformer":
        return QuantiTransformer(n_features=n_in, T=T, n_dynamic=n_in, n_macro=0,
                                 d_model=32, n_heads=4, n_layers=2, loss_type="t_student")
    if arch == "nhits":
        return QuantNHiTS(n_features=n_in, T=T, n_dynamic_features=n_in, n_macro=0,
                          d_model=32, hidden=64, n_stacks=2, loss_type="t_student")
    return QuantTCNMamba(n_features=n_in, d_model=32, tcn_layers=2, mamba_layers=1,
                         n_dynamic_features=n_in, loss_type="t_student")


@pytest.mark.parametrize("arch", ["itransformer", "nhits", "tcnmamba"])
def test_latent_none_parity(arch, x):
    # IT: latent=None deve essere BIT-IDENTICO alla chiamata legacy (parity).
    # EN: latent=None must be BIT-IDENTICAL to the legacy call (parity).
    torch.manual_seed(1)
    m = _make(arch, F_).eval()
    with torch.no_grad():
        a = m(x)
        b = m(x, None, None)
    dmax = max((ta - tb).abs().max().item() for ta, tb in zip(a, b))
    assert dmax == 0.0


@pytest.mark.parametrize("arch", ["itransformer", "nhits", "tcnmamba"])
def test_augmented_accepts_latent(arch, x, cafn):
    # IT: modello costruito a larghezza F+DL → accetta il latente CAFN.
    # EN: model built at width F+DL → accepts the CAFN latent.
    latent, _ = cafn(x)
    m = _make(arch, F_ + DL).eval()
    with torch.no_grad():
        out = m(x, None, latent.detach())
    assert out[0].shape[0] == B
