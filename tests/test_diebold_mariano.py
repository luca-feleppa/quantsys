# IT: Test del test di Diebold-Mariano (HAC + correzione HLN) e della parità
#     qlike_series ↔ qlike. Il DM è DESCRITTIVO nella linea vol (non gating): questi
#     test proteggono la correttezza statistica, non un verdetto di gate.
# EN: Tests for the Diebold-Mariano test (HAC + HLN correction) and the
#     qlike_series ↔ qlike parity. DM is DESCRIPTIVE in the vol line (not gating):
#     these tests protect statistical correctness, not a gate verdict.
import numpy as np
import pytest

from quantsys.model.vol_metrics import qlike, qlike_series, diebold_mariano


# IT: la media della serie per-campione DEVE essere esattamente la QLIKE aggregata
#     (single source of truth: qlike() è definita come mean(qlike_series())).
# EN: the mean of the per-sample series MUST be exactly the aggregate QLIKE
#     (single source of truth: qlike() is defined as mean(qlike_series())).
def test_qlike_series_mean_equals_qlike():
    rng = np.random.default_rng(0)
    rv_true = np.exp(rng.normal(-7.0, 1.0, size=500))
    rv_pred = np.exp(rng.normal(-7.0, 1.0, size=500))
    assert qlike_series(rv_true, rv_pred).mean() == pytest.approx(qlike(rv_true, rv_pred), rel=0, abs=0)


# IT: convenzione dei segni — d = loss_a − loss_b, quindi DM < 0 quando A perde
#     meno (A migliore) e il flag `better` deve dirlo. Scambiando gli argomenti la
#     statistica cambia segno e resta simmetrica in modulo.
# EN: sign convention — d = loss_a − loss_b, so DM < 0 when A loses less (A is
#     better) and the `better` flag must say so. Swapping arguments flips the sign
#     and keeps the magnitude symmetric.
def test_sign_convention_and_symmetry():
    rng = np.random.default_rng(1)
    loss_good = np.abs(rng.normal(1.0, 0.3, size=600))
    loss_bad = loss_good + np.abs(rng.normal(0.5, 0.3, size=600))

    r = diebold_mariano(loss_good, loss_bad, h=1)
    assert r["dm_hln"] < 0 and r["better"] == "a" and r["p_value"] < 1e-6

    r_swapped = diebold_mariano(loss_bad, loss_good, h=1)
    assert r_swapped["better"] == "b"
    assert r_swapped["dm_hln"] == pytest.approx(-r["dm_hln"], rel=1e-12)


# IT: due forecast identici → differenziale identicamente nullo: nessuna statistica
#     calcolabile (nan) e nessuna eccezione.
# EN: two identical forecasts → identically null differential: no computable
#     statistic (nan) and no exception.
def test_identical_forecasts_return_nan():
    loss = np.abs(np.random.default_rng(2).normal(1.0, 0.2, size=300))
    r = diebold_mariano(loss, loss.copy(), h=1)
    assert np.isnan(r["dm"]) and np.isnan(r["p_value"])


# IT: IL PUNTO DEL TEST — su differenziali AUTOCORRELATI (finestre sovrapposte, come
#     il target log-RV che somma h barre) la varianza HAC deve gonfiare lo standard
#     error: |DM| con lag=h−1 DEVE essere sensibilmente minore di |DM| con lag=0
#     (varianza iid). Se questa asserzione cade, i p-value del giudice sono fittizi.
# EN: THE POINT OF THIS TEST — on AUTOCORRELATED differentials (overlapping windows,
#     like the log-RV target summing h bars) the HAC variance must inflate the
#     standard error: |DM| at lag=h−1 MUST be materially smaller than |DM| at lag=0
#     (iid variance). If this assertion fails, the judge's p-values are fictitious.
def test_hac_lag_inflates_standard_error_on_overlapping_data():
    rng = np.random.default_rng(3)
    h = 30
    base = rng.normal(0.0, 1.0, size=4000 + h - 1)
    # IT: media mobile su h punti = struttura di sovrapposizione del target a h barre.
    #     Il DIFFERENZIALE (non la loss) è la quantità che deve essere autocorrelata.
    # EN: h-point moving average = the overlap structure of the h-bar target. The
    #     DIFFERENTIAL (not the loss) is the quantity that must be autocorrelated.
    d = np.convolve(base, np.ones(h) / h, mode="valid") - 0.05
    loss_a = 2.0 + np.abs(rng.normal(0.0, 0.1, size=d.size))
    loss_b = loss_a - d  # IT/EN: loss_a − loss_b = d per costruzione / by construction

    iid = diebold_mariano(loss_a, loss_b, h=h, lag=0)
    hac = diebold_mariano(loss_a, loss_b, h=h)  # IT/EN: lag di default = h−1 / default lag = h−1

    assert hac["hac_lag"] == h - 1 and iid["hac_lag"] == 0
    assert abs(hac["dm_hln"]) < abs(iid["dm_hln"])
    # IT: n_eff riflette la sovrapposizione, non la lunghezza nominale della serie
    # EN: n_eff reflects the overlap, not the nominal series length
    assert hac["n_eff"] == pytest.approx(hac["n"] / h, rel=1e-9)


# IT: la correzione HLN riduce la statistica in modulo (fattore < 1 per h > 1) e a
#     h=1 su n grande è praticamente inerte.
# EN: the HLN correction shrinks the statistic in magnitude (factor < 1 for h > 1)
#     and at h=1 with large n it is practically inert.
def test_hln_correction_shrinks_statistic():
    rng = np.random.default_rng(4)
    loss_a = np.abs(rng.normal(1.0, 0.3, size=2000))
    loss_b = loss_a + 0.1
    r_h30 = diebold_mariano(loss_a, loss_b, h=30, lag=0)
    r_h1 = diebold_mariano(loss_a, loss_b, h=1, lag=0)
    assert abs(r_h30["dm_hln"]) < abs(r_h30["dm"])
    assert r_h1["dm_hln"] == pytest.approx(r_h1["dm"], rel=1e-3)


# IT: guard sul campione minimo — nessuna statistica sotto n=10.
# EN: minimum-sample guard — no statistic below n=10.
def test_small_sample_guard():
    r = diebold_mariano(np.ones(5), np.zeros(5), h=1)
    assert np.isnan(r["dm"]) and r["n"] == 5


# IT: contratto di ritorno UNIFORME — anche i rami degeneri (differenziale costante,
#     campione minimo, forecast identici) espongono le stesse chiavi: i consumer
#     (giudice, report JSON) non devono difendersi da KeyError.
# EN: UNIFORM return contract — degenerate branches (constant differential, minimum
#     sample, identical forecasts) expose the same keys: consumers (judge, JSON
#     report) need no KeyError defence.
@pytest.mark.parametrize("loss_a,loss_b,h", [
    (np.ones(300) * 2.0, np.ones(300) * 2.0 + 0.05, 30),   # IT/EN: differenziale costante / constant
    (np.ones(5), np.zeros(5), 1),                          # IT/EN: n < 10
    (np.abs(np.random.default_rng(5).normal(1, .2, 300)),) * 2 + (1,),  # identici / identical
])
def test_uniform_return_contract(loss_a, loss_b, h):
    r = diebold_mariano(loss_a, loss_b, h=h)
    for key in ("dm", "dm_hln", "p_value", "mean_diff", "hac_lag", "n", "n_eff", "better"):
        assert key in r, f"chiave mancante / missing key: {key}"
