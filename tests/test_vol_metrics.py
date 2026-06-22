# IT: Regression test del modulo condiviso quantsys.model.vol_metrics — garantisce
#     che l'estrazione di qlike/inversione dal giudice `dev_vols_qlike.py` sia
#     numericamente IDENTICA all'implementazione inline pre-refactor (Tier 1).
# EN: Regression test for the shared module quantsys.model.vol_metrics — guarantees
#     that extracting qlike/inversion from the `dev_vols_qlike.py` judge is
#     numerically IDENTICAL to the pre-refactor inline implementation (Tier 1).
import numpy as np
import pytest

from quantsys.model.vol_metrics import EPS, qlike, invert_log_rv, qlike_from_z


def _qlike_inline(rv_true, rv_pred):
    # IT: copia ESATTA della formula inline che viveva in dev_vols_qlike.py
    # EN: EXACT copy of the inline formula that lived in dev_vols_qlike.py
    r = rv_true / np.maximum(rv_pred, EPS)
    return float(np.mean(r - np.log(r) - 1.0))


def test_qlike_matches_inline():
    rng = np.random.default_rng(0)
    rv_true = np.exp(rng.normal(-7.0, 1.0, size=2000))
    rv_pred = np.exp(rng.normal(-7.0, 1.0, size=2000))
    assert qlike(rv_true, rv_pred) == pytest.approx(_qlike_inline(rv_true, rv_pred), rel=0, abs=1e-15)


def test_qlike_zero_at_equality():
    # IT: QLIKE(x, x) = 0 (loss nulla a previsione perfetta) | EN: QLIKE(x, x) = 0
    rv = np.exp(np.random.default_rng(1).normal(-7.0, 1.0, size=500))
    assert qlike(rv, rv) == pytest.approx(0.0, abs=1e-12)


def test_qlike_nonnegative():
    # IT: QLIKE ≥ 0 sempre (r - ln r - 1 ≥ 0) | EN: QLIKE ≥ 0 always
    rng = np.random.default_rng(2)
    for _ in range(20):
        a = np.exp(rng.normal(-7, 1.5, size=300))
        b = np.exp(rng.normal(-7, 1.5, size=300))
        assert qlike(a, b) >= -1e-12


def test_invert_log_rv():
    z = np.array([0.0, 1.0, -2.0])
    c, s = -7.175, 1.4376
    out = invert_log_rv(z, c, s)
    np.testing.assert_allclose(out, z * s + c, rtol=0, atol=1e-12)


def test_qlike_from_z_reproduces_judge_path():
    # IT: replica il path del giudice (mu_z·s+c → exp → QLIKE) e verifica che
    #     qlike_from_z dia lo stesso numero. center/scale realistici del log-RV.
    # EN: replays the judge path (mu_z·s+c → exp → QLIKE) and checks qlike_from_z
    #     yields the same number. realistic log-RV center/scale.
    rng = np.random.default_rng(3)
    c, s = -7.175, 1.4376
    y_z  = rng.normal(0.0, 1.0, size=1500)            # IT/EN: target z-scorato | z-scored target
    mu_z = y_z + rng.normal(0.0, 0.3, size=1500)      # IT/EN: predizione rumorosa | noisy prediction

    log_true = y_z * s + c
    log_pred = mu_z * s + c
    expected_qlike = _qlike_inline(np.exp(log_true), np.exp(log_pred))
    expected_mse   = float(np.mean((log_true - log_pred) ** 2))

    out = qlike_from_z(y_z, mu_z, c, s)
    assert out["qlike"]   == pytest.approx(expected_qlike, rel=0, abs=1e-12)
    assert out["mse_log"] == pytest.approx(expected_mse,   rel=0, abs=1e-12)


def test_qlike_from_z_perfect_prediction():
    # IT: mu == y → QLIKE 0 e MSE 0 | EN: mu == y → QLIKE 0 and MSE 0
    y_z = np.random.default_rng(4).normal(0, 1, size=400)
    out = qlike_from_z(y_z, y_z.copy(), -7.0, 1.4)
    assert out["qlike"]   == pytest.approx(0.0, abs=1e-12)
    assert out["mse_log"] == pytest.approx(0.0, abs=1e-12)
