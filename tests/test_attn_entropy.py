# IT: A10 — test della penalità entropica sull'attention (pre-registrazione
#     2026-07-28). La responsabilità PRINCIPALE è la condizione ③ della pre-reg:
#     con `attn_entropy_lambda = 0.0` (default production) il path numerico deve
#     restare quello storico, cioè F.scaled_dot_product_attention senza mai
#     materializzare la matrice N×N, con `state_dict` invariato e i checkpoint
#     esistenti caricabili. Se questi test non passano, la pre-reg vieta il run.
#     Secondariamente si verifica che la MISURA sia corretta (H_norm ∈ [0,1], =1
#     esattamente sull'attention uniforme) e che accenderla non cambi la funzione
#     calcolata in eval — altrimenti il manipulation check ④ misurerebbe
#     l'implementazione invece dell'ipotesi.
# EN: A10 — tests for the attention entropy penalty (2026-07-28 pre-registration).
#     The PRIMARY responsibility is pre-reg condition ③: with
#     `attn_entropy_lambda = 0.0` (production default) the numeric path must stay the
#     historical one, i.e. F.scaled_dot_product_attention with no N×N matrix ever
#     materialized, `state_dict` unchanged and existing checkpoints loadable. If these
#     tests fail, the pre-reg forbids the run.
#     Secondarily it checks the MEASUREMENT is correct (H_norm ∈ [0,1], exactly 1 on
#     uniform attention) and that enabling it does not change the function computed at
#     eval time — otherwise manipulation check ④ would measure the implementation
#     rather than the hypothesis.
import math

import pytest
import torch
import torch.nn.functional as F

from quantsys.model import QuantiTransformer

N_FEAT, T, B = 8, 20, 4


def _build(lam: float = 0.0, seed: int = 0) -> QuantiTransformer:
    # IT: stesso seed ⇒ stessi pesi iniziali fra i bracci | EN: same seed ⇒ same init
    torch.manual_seed(seed)
    return QuantiTransformer(
        n_features=N_FEAT, T=T, d_model=16, n_heads=2, n_layers=2,
        dropout=0.0, loss_type="quantile", attn_entropy_lambda=lam,
    )


@pytest.fixture
def x():
    torch.manual_seed(123)
    return torch.randn(B, T, N_FEAT)


# ── ③ INERZIA A λ=0 / INERTIA AT λ=0 ─────────────────────────────────────────
def test_lambda_zero_uses_flash_attention(monkeypatch, x):
    # IT: a λ=0 il forward DEVE passare da scaled_dot_product_attention (una volta
    #     per layer) — è la prova diretta che la matrice non viene materializzata.
    # EN: at λ=0 the forward MUST go through scaled_dot_product_attention (once per
    #     layer) — direct evidence that the matrix is never materialized.
    calls = []
    real = F.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    m = _build(0.0).eval()
    with torch.no_grad():
        m(x)
    assert len(calls) == 2                       # IT/EN: un layer, una chiamata / one per layer


def test_measurement_on_bypasses_flash_attention(monkeypatch, x):
    # IT: speculare — con la misura accesa la SDPA non va chiamata affatto.
    # EN: mirror image — with measurement on, SDPA must not be called at all.
    calls = []
    real = F.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    m = _build(0.01).eval()
    with torch.no_grad():
        m(x)
    assert calls == []


def test_lambda_zero_leaves_state_dict_unchanged():
    # IT: la leva non introduce parametri/buffer: le chiavi devono coincidere fra i
    #     due bracci → i checkpoint storici restano caricabili in strict mode.
    # EN: the lever introduces no parameters/buffers: keys must match across arms →
    #     legacy checkpoints stay loadable in strict mode.
    base, cand = _build(0.0), _build(0.01)
    assert list(base.state_dict().keys()) == list(cand.state_dict().keys())


def test_legacy_checkpoint_loads_into_penalized_model():
    # IT: caricamento STRICT di un checkpoint addestrato senza la leva dentro un
    #     modello con λ>0 (e viceversa): nessuna chiave mancante o inattesa.
    # EN: STRICT load of a lever-free checkpoint into a λ>0 model (and back): no
    #     missing or unexpected keys.
    base, cand = _build(0.0), _build(0.01, seed=1)
    cand.load_state_dict(base.state_dict(), strict=True)
    base.load_state_dict(cand.state_dict(), strict=True)


def test_lambda_zero_yields_no_penalty(x):
    # IT: a λ=0 nessuna entropia viene registrata → `attn_entropy_penalty()` è None e
    #     il chiamante non può sommare per sbaglio un termine alla loss.
    # EN: at λ=0 no entropy is recorded → `attn_entropy_penalty()` is None and the
    #     caller cannot accidentally add a term to the loss.
    m = _build(0.0).eval()
    with torch.no_grad():
        m(x)
    assert all(l._last_attn_entropy is None for l in m.layers)
    assert m.attn_entropy_penalty() is None


def test_penalty_is_cleared_after_read(x):
    # IT: la penalità si consuma alla lettura — una seconda somma nello stesso step
    #     (o uno step senza forward) non può riusare un valore stantio.
    # EN: the penalty is consumed on read — a second addition within the same step
    #     (or a step with no forward) cannot reuse a stale value.
    m = _build(0.01).eval()
    with torch.no_grad():
        m(x)
    assert m.attn_entropy_penalty() is not None
    assert m.attn_entropy_penalty() is None


def test_set_attn_entropy_toggles_both_ways(monkeypatch, x):
    # IT: il toggle esplicito è ciò che permette di misurare H_norm sul braccio
    #     BASELINE (addestrato a λ=0) per il manipulation check ④.
    # EN: the explicit toggle is what allows measuring H_norm on the BASELINE arm
    #     (trained at λ=0) for manipulation check ④.
    m = _build(0.0).eval()
    m.set_attn_entropy(True)
    with torch.no_grad():
        m(x)
    assert m.attn_entropy_penalty() is not None
    m.set_attn_entropy(False)
    with torch.no_grad():
        m(x)
    assert m.attn_entropy_penalty() is None


# ── CORRETTEZZA DELLA MISURA / MEASUREMENT CORRECTNESS ───────────────────────
def test_entropy_equals_one_on_uniform_attention(x):
    # IT: azzerando qkv_proj tutti gli score sono uguali → softmax uniforme →
    #     H = log N e H_norm = 1 ESATTAMENTE. Verifica la normalizzazione per log N,
    #     che è ciò che rende λ interpretabile come costo massimo in unità di loss.
    # EN: zeroing qkv_proj makes all scores equal → uniform softmax → H = log N and
    #     H_norm = 1 EXACTLY. Validates the log-N normalization, which is what makes λ
    #     interpretable as the maximum cost in loss units.
    m = _build(0.01).eval()
    with torch.no_grad():
        for l in m.layers:
            l.qkv_proj.weight.zero_()
        m(x)
    assert m.attn_entropy_penalty().item() == pytest.approx(1.0, abs=1e-6)


def test_entropy_is_bounded_and_drops_when_concentrated(x):
    # IT: H_norm ∈ [0,1] sempre; amplificando gli score l'attention si concentra e
    #     l'entropia normalizzata DEVE scendere — è il segno che la penalità, quando
    #     morde, spinge nella direzione dichiarata dall'ipotesi.
    # EN: H_norm ∈ [0,1] always; amplifying the scores concentrates attention and the
    #     normalized entropy MUST fall — the sign that the penalty, when it bites,
    #     pushes in the direction the hypothesis declares.
    m = _build(0.01).eval()
    with torch.no_grad():
        m(x)
        h_base = m.attn_entropy_penalty().item()
        for l in m.layers:
            l.qkv_proj.weight.mul_(50.0)
        m(x)
        h_sharp = m.attn_entropy_penalty().item()
    assert 0.0 <= h_sharp <= h_base <= 1.0
    assert h_sharp < h_base


def test_penalty_is_differentiable_wrt_attention_weights(x):
    # IT: la penalità deve produrre gradiente sulle proiezioni q/k (altrimenti non
    #     regolarizza nulla e il run misurerebbe un no-op).
    # EN: the penalty must produce gradient on the q/k projections (otherwise it
    #     regularizes nothing and the run would measure a no-op).
    m = _build(0.01).train()
    m(x)
    pen = m.attn_entropy_penalty()
    assert pen.requires_grad
    pen.backward()
    g = m.layers[0].qkv_proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_measurement_does_not_change_the_function_in_eval(x):
    # IT: in eval (dropout off) il path di misura e la SDPA calcolano la STESSA
    #     funzione: la misura non deve alterare le predizioni, altrimenti H_norm e
    #     QLIKE non sarebbero confrontabili fra i bracci.
    # EN: at eval (dropout off) the measurement path and SDPA compute the SAME
    #     function: measurement must not alter predictions, otherwise H_norm and QLIKE
    #     would not be comparable across arms.
    m = _build(0.0).eval()
    with torch.no_grad():
        flash = m(x)[0]
        m.set_attn_entropy(True)
        manual = m(x)[0]
    torch.testing.assert_close(manual, flash, rtol=1e-4, atol=1e-5)


def test_normalization_uses_the_actual_token_count(x):
    # IT: N = token della sequenza di attention. Col macro token la sequenza è F+1:
    #     se il divisore restasse log F, H_norm potrebbe superare 1 e λ perderebbe
    #     l'interpretazione di costo massimo.
    # EN: N = tokens of the attention sequence. With the macro token the sequence is
    #     F+1: were the divisor left at log F, H_norm could exceed 1 and λ would lose
    #     its maximum-cost interpretation.
    torch.manual_seed(0)
    m = QuantiTransformer(n_features=N_FEAT, T=T, n_macro=3, d_model=16, n_heads=2,
                          n_layers=2, dropout=0.0, loss_type="quantile",
                          attn_entropy_lambda=0.01).eval()
    with torch.no_grad():
        for l in m.layers:
            l.qkv_proj.weight.zero_()
        m(x, torch.randn(B, 3))
    # IT: uniforme su F+1 token → esattamente 1.0 con il divisore corretto
    # EN: uniform over F+1 tokens → exactly 1.0 with the correct divisor
    assert m.attn_entropy_penalty().item() == pytest.approx(1.0, abs=1e-6)
    assert math.log(N_FEAT + 1) > math.log(N_FEAT)   # IT/EN: i due divisori differiscono / divisors differ
