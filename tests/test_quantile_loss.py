"""
IT: Golden test del ramo di loss di PRODUZIONE (loss_type=quantile, TEORIA.md 7.0).
    Copre: proprieta' del minimizzatore della pinball loss, equivalenza MAE a tau=0.5,
    e il contratto posizionale QUANTILES <-> predict() (mu = q(0.5), sigma = IDR).
EN: Golden tests for the PRODUCTION loss branch (loss_type=quantile, TEORIA.md 7.0).
    Covers: pinball-loss minimizer property, MAE equivalence at tau=0.5, and the
    positional contract QUANTILES <-> predict() (mu = q(0.5), sigma = IDR).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from quantsys.model import QUANTILES, quantile_loss


# ─────────────────────────────────────────────────────────────────────────────
# IT: Proprieta' matematiche della pinball loss
# EN: Mathematical properties of the pinball loss
# ─────────────────────────────────────────────────────────────────────────────
class TestPinballProperties:

    # IT: a tau=0.5 la pinball degenera in mezzo errore assoluto (MAE/2).
    # EN: at tau=0.5 the pinball degenerates into half the absolute error (MAE/2).
    def test_median_level_equals_half_mae(self):
        torch.manual_seed(0)
        y = torch.randn(512)
        q = torch.randn(512, 1)
        got = quantile_loss(y, q, quantiles=[0.5])
        expected = 0.5 * (y - q[:, 0]).abs().mean()
        assert torch.allclose(got, expected, atol=1e-7)

    # IT: il minimizzatore empirico della pinball a livello tau e' il quantile tau
    #     campionario: lo verifichiamo su una griglia fitta di candidati costanti.
    # EN: the empirical minimizer of the pinball at level tau is the sample tau-quantile:
    #     verified by scanning a dense grid of constant candidates.
    @pytest.mark.parametrize("tau", [0.1, 0.25, 0.5, 0.75, 0.9])
    def test_minimizer_is_the_sample_quantile(self, tau):
        rng = np.random.default_rng(42)
        # IT: distribuzione asimmetrica -> media e mediana NON coincidono (caso rilevante)
        # EN: skewed distribution -> mean and median do NOT coincide (the relevant case)
        y_np = rng.lognormal(mean=0.0, sigma=0.7, size=4000)
        y = torch.tensor(y_np, dtype=torch.float64)

        grid = torch.linspace(float(y.min()), float(y.max()), 4001, dtype=torch.float64)
        losses = torch.stack([
            quantile_loss(y, c.repeat(y.shape[0], 1), quantiles=[tau]) for c in grid
        ])
        argmin = grid[int(losses.argmin())]
        target = torch.quantile(y, tau)
        # IT: tolleranza = passo della griglia (l'argmin discreto non e' esatto)
        # EN: tolerance = grid step (the discrete argmin is not exact)
        step = float(grid[1] - grid[0])
        assert abs(float(argmin) - float(target)) <= 5 * step

    # IT: sotto-stimare a tau alto deve costare piu' che sovra-stimare (asimmetria).
    # EN: under-predicting at high tau must cost more than over-predicting (asymmetry).
    def test_asymmetry_direction(self):
        y = torch.zeros(1)
        under = quantile_loss(y, torch.tensor([[-1.0]]), quantiles=[0.9])  # y sopra la stima
        over  = quantile_loss(y, torch.tensor([[+1.0]]), quantiles=[0.9])  # y sotto la stima
        assert float(under) == pytest.approx(9.0 * float(over), rel=1e-6)

    # IT: la loss e' non negativa qualunque sia il segno dell'errore.
    # EN: the loss is non-negative regardless of the error sign.
    def test_non_negative(self):
        torch.manual_seed(1)
        y = torch.randn(256)
        q = torch.randn(256, len(QUANTILES))
        assert float(quantile_loss(y, q)) >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# IT: Contratto posizionale QUANTILES <-> predict(). predict() indicizza per
#     posizione ([:, 2] per mu, [:, 4] - [:, 0] per sigma): se QUANTILES cambia
#     senza aggiornare quelle posizioni, mu/sigma diventano silenziosamente errati.
# EN: Positional contract QUANTILES <-> predict(). predict() indexes positionally
#     ([:, 2] for mu, [:, 4] - [:, 0] for sigma): changing QUANTILES without
#     updating those positions silently corrupts mu/sigma.
# ─────────────────────────────────────────────────────────────────────────────
class TestQuantileLayoutContract:

    def test_five_levels_sorted_and_unique(self):
        assert len(QUANTILES) == 5
        assert QUANTILES == sorted(QUANTILES)
        assert len(set(QUANTILES)) == 5
        assert all(0.0 < q < 1.0 for q in QUANTILES)

    # IT: mu = q(0.5) -> la mediana DEVE stare in posizione 2.
    # EN: mu = q(0.5) -> the median MUST sit at index 2.
    def test_median_at_index_two(self):
        assert QUANTILES[2] == 0.5

    # IT: sigma = q[4] - q[0] -> gli estremi devono essere simmetrici attorno a 0.5,
    #     altrimenti l'ampiezza non e' un intervallo centrato sulla mediana.
    # EN: sigma = q[4] - q[0] -> the outer levels must be symmetric around 0.5,
    #     otherwise the range is not centred on the median.
    def test_outer_levels_symmetric(self):
        assert QUANTILES[0] + QUANTILES[4] == pytest.approx(1.0)
        assert QUANTILES[1] + QUANTILES[3] == pytest.approx(1.0)

    # IT: sigma del ramo quantile e' un'ampiezza INTERDECILE, non una deviazione
    #     standard: su una gaussiana vale ~2.563 sigma. Il test blinda il fattore,
    #     che e' la trappola di lettura piu' probabile (TEORIA.md 7.0).
    # EN: the quantile-branch sigma is an INTERDECILE range, not a standard
    #     deviation: on a Gaussian it equals ~2.563 sigma. This pins the factor,
    #     which is the most likely misreading (TEORIA.md 7.0).
    def test_interdecile_gaussian_factor(self):
        from scipy.stats import norm
        idr = norm.ppf(QUANTILES[4]) - norm.ppf(QUANTILES[0])
        assert idr == pytest.approx(2.5631, abs=1e-3)
