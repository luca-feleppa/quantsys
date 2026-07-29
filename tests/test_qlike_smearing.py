# IT: C1 — test della correzione di smearing di Duan (1983) sul giudice QLIKE
#     (pre-registrazione 2026-07-28). Due responsabilità distinte:
#       ① INERZIA: coi default (`smear=1.0`) `qlike_from_z` e `har_fold_qlike` devono
#          restituire numeri BIT-IDENTICI al path storico → il giudice acceso e il
#          giudice spento non possono divergere, e i report già pubblicati restano
#          confrontabili. È la pre-condizione dichiarata prima di spendere GPU.
#       ② CORRETTEZZA: ŝ è il moltiplicatore che MINIMIZZA la QLIKE quando la
#          previsione è una mediana e i residui in log sono indipendenti — proprietà
#          esatta, non asintotica, che è esattamente l'ipotesi ① della pre-reg.
#     + non-leakage strutturale: ŝ non deve dipendere dai campioni di valutazione.
# EN: C1 — tests for the Duan (1983) smearing correction on the QLIKE judge
#     (2026-07-28 pre-registration). Two distinct responsibilities:
#       ① INERTIA: at the defaults (`smear=1.0`) `qlike_from_z` and `har_fold_qlike`
#          must return BIT-IDENTICAL numbers to the historical path → judge-on and
#          judge-off cannot diverge and already-published reports stay comparable.
#          This is the pre-condition declared before spending any GPU.
#       ② CORRECTNESS: ŝ is the multiplier that MINIMIZES QLIKE when the forecast is a
#          median and the log residuals are independent — an exact, non-asymptotic
#          property, which is precisely hypothesis ① of the pre-reg.
#     + structural non-leakage: ŝ must not depend on the evaluation samples.
import numpy as np
import pandas as pd
import pytest

from quantsys.model.vol_metrics import (EPS, duan_smearing, har_fold_qlike, qlike,
                                        qlike_from_z)


# ── ① INERZIA / INERTIA ──────────────────────────────────────────────────────
def _qlike_legacy(rv_true, rv_pred):
    # IT: formula storica, senza alcun fattore | EN: historical formula, no factor
    r = rv_true / np.maximum(rv_pred, EPS)
    return float(np.mean(r - np.log(r) - 1.0))


def test_qlike_from_z_default_is_bit_identical():
    # IT: default `smear=1.0` ⇒ stesso bit del path storico (x·1.0 è esatto in IEEE754).
    # EN: default `smear=1.0` ⇒ same bits as the historical path (x·1.0 is exact in IEEE754).
    rng = np.random.default_rng(11)
    c, s = -7.175, 1.4376
    y_z = rng.normal(0.0, 1.0, size=3000)
    mu_z = y_z + rng.normal(0.0, 0.45, size=3000)

    legacy = _qlike_legacy(np.exp(y_z * s + c), np.exp(mu_z * s + c))
    assert qlike_from_z(y_z, mu_z, c, s)["qlike"] == legacy               # IT/EN: uguaglianza ESATTA / EXACT
    assert qlike_from_z(y_z, mu_z, c, s, smear=1.0)["qlike"] == legacy


def test_qlike_from_z_mse_log_is_untouched_by_smearing():
    # IT: la correzione vive sul LIVELLO (QLIKE); l'MSE in log giudica la mediana e
    #     NON deve muoversi — è la scelta di specificazione dichiarata nel modulo.
    # EN: the correction lives on the LEVEL (QLIKE); the log MSE judges the median and
    #     must NOT move — the specification choice documented in the module.
    rng = np.random.default_rng(12)
    y_z, mu_z = rng.normal(size=800), rng.normal(size=800)
    a = qlike_from_z(y_z, mu_z, -7.0, 1.4)
    b = qlike_from_z(y_z, mu_z, -7.0, 1.4, smear=1.35)
    assert b["mse_log"] == a["mse_log"]
    assert b["qlike"] != a["qlike"]


def _har_frame(n=900, seed=7):
    # IT: frame HAR sintetico con la struttura attesa da `har_fold_qlike`.
    # EN: synthetic HAR frame with the structure `har_fold_qlike` expects.
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    xh = rng.normal(-7.0, 0.8, size=n)
    return pd.DataFrame({
        "y":  xh + rng.normal(0.0, 0.5, size=n),
        "xh": xh,
        "xw": xh + rng.normal(0.0, 0.2, size=n),
        "xm": xh + rng.normal(0.0, 0.3, size=n),
    }, index=idx)


def _har_legacy(har, t_train, t_eval):
    # IT: copia ESATTA del corpo pre-C1 di har_fold_qlike | EN: EXACT pre-C1 body copy
    tr, ev = har.loc[har.index.intersection(t_train)], har.loc[har.index.intersection(t_eval)]
    Xtr = np.column_stack([np.ones(len(tr)), tr[["xh", "xw", "xm"]].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    Xev = np.column_stack([np.ones(len(ev)), ev[["xh", "xw", "xm"]].values])
    rv_true = np.exp(ev["y"].values)
    return {"qlike_har": qlike(rv_true, np.exp(Xev @ beta)),
            "qlike_naive": qlike(rv_true, np.exp(ev["xh"].values))}


def test_har_fold_qlike_default_is_bit_identical():
    har = _har_frame()
    t_tr, t_ev = har.index[:700], har.index[700:]
    legacy, new = _har_legacy(har, t_tr, t_ev), har_fold_qlike(har, t_tr, t_ev)
    assert new["qlike_har"] == legacy["qlike_har"]
    assert new["qlike_naive"] == legacy["qlike_naive"]


def test_har_fold_qlike_smear_is_side_selective():
    # IT: `smear` tocca SOLO la leg HAR, `smear_naive` SOLO la naive — correggere un
    #     lato solo dev'essere una scelta esplicita, mai un effetto collaterale.
    # EN: `smear` touches ONLY the HAR leg, `smear_naive` ONLY the naive one — a
    #     one-sided correction must be an explicit choice, never a side effect.
    har = _har_frame()
    t_tr, t_ev = har.index[:700], har.index[700:]
    base = har_fold_qlike(har, t_tr, t_ev)
    only_har = har_fold_qlike(har, t_tr, t_ev, smear=1.2)
    only_naive = har_fold_qlike(har, t_tr, t_ev, smear_naive=1.2)
    assert only_har["qlike_har"] != base["qlike_har"]
    assert only_har["qlike_naive"] == base["qlike_naive"]
    assert only_naive["qlike_har"] == base["qlike_har"]
    assert only_naive["qlike_naive"] != base["qlike_naive"]


def test_har_fold_qlike_degenerate_branch_keeps_contract():
    # IT: ramo degenere → stesse chiavi con nan (nessun KeyError nei consumer).
    # EN: degenerate branch → same keys with nan (no KeyError in consumers).
    har = _har_frame(n=60)
    out = har_fold_qlike(har, har.index[:10], har.index[10:])
    assert set(out) == {"qlike_har", "qlike_naive", "n_har", "n_eval",
                        "smear_har_hat", "smear_naive_hat"}
    assert np.isnan(out["smear_har_hat"])


# ── ② CORRETTEZZA DELLO STIMATORE / ESTIMATOR CORRECTNESS ────────────────────
def test_duan_smearing_identity_on_zero_residuals():
    assert duan_smearing(np.zeros(500)) == pytest.approx(1.0, abs=1e-15)
    assert duan_smearing(np.array([])) == 1.0


def test_duan_smearing_ignores_non_finite():
    e = np.array([0.1, np.nan, 0.2, np.inf, -np.inf])
    assert duan_smearing(e) == pytest.approx(float(np.mean(np.exp([0.1, 0.2]))), abs=1e-15)


def test_duan_smearing_recovers_lognormal_factor():
    # IT: sotto ε ~ N(0, σ²) il fattore teorico è exp(σ²/2); lo stimatore è non
    #     parametrico ma deve convergerci su campione grande.
    # EN: under ε ~ N(0, σ²) the theoretical factor is exp(σ²/2); the estimator is
    #     non-parametric but must converge to it on a large sample.
    sigma = 0.6
    e = np.random.default_rng(21).normal(0.0, sigma, size=400_000)
    assert duan_smearing(e) == pytest.approx(np.exp(sigma ** 2 / 2), rel=0.01)


def test_smearing_factor_is_the_qlike_optimal_multiplier():
    # IT: proprietà ESATTA che giustifica l'esperimento C1. Con RV_true = RV_pred·exp(ε),
    #     QLIKE(s) = mean(exp(ε)/s − log(exp(ε)/s) − 1) ha derivata nulla in
    #     s = mean(exp(ε)) = ŝ di Duan → il fattore di smearing È l'argmin, e correggere
    #     una previsione-mediana non può che ABBASSARE la QLIKE (ipotesi ① della pre-reg).
    # EN: the EXACT property justifying experiment C1. With RV_true = RV_pred·exp(ε),
    #     QLIKE(s) = mean(exp(ε)/s − log(exp(ε)/s) − 1) has zero derivative at
    #     s = mean(exp(ε)) = Duan's ŝ → the smearing factor IS the argmin, so correcting
    #     a median forecast can only LOWER QLIKE (hypothesis ① of the pre-reg).
    rng = np.random.default_rng(31)
    log_pred = rng.normal(-7.0, 0.9, size=20_000)
    eps = rng.normal(0.0, 0.7, size=20_000)
    rv_true, rv_pred = np.exp(log_pred + eps), np.exp(log_pred)

    s_hat = duan_smearing(eps)
    at_hat = qlike(rv_true, rv_pred * s_hat)
    assert at_hat < qlike(rv_true, rv_pred)                      # IT/EN: meglio del raw / beats raw
    grid = s_hat * np.array([0.80, 0.90, 0.95, 1.05, 1.10, 1.25])
    assert all(at_hat <= qlike(rv_true, rv_pred * g) for g in grid)   # IT/EN: è l'argmin / it is the argmin


# ── ③ NON-LEAKAGE STRUTTURALE / STRUCTURAL NON-LEAKAGE ───────────────────────
def test_smear_hat_depends_on_train_only():
    # IT: ŝ stimato in `har_fold_qlike` non deve muoversi se cambiano i dati di
    #     VALUTAZIONE — è la condizione ③ della pre-reg, verificabile nel codice.
    # EN: the ŝ estimated inside `har_fold_qlike` must not move when the EVALUATION
    #     data change — pre-reg condition ③, verifiable in code.
    har = _har_frame()
    t_tr = har.index[:700]
    a = har_fold_qlike(har, t_tr, har.index[700:800])
    b = har_fold_qlike(har, t_tr, har.index[800:])
    assert a["smear_har_hat"] == b["smear_har_hat"]
    assert a["smear_naive_hat"] == b["smear_naive_hat"]

    perturbed = har.copy()
    perturbed.iloc[700:, perturbed.columns.get_loc("y")] += 3.0   # IT/EN: solo eval / eval only
    c = har_fold_qlike(perturbed, t_tr, perturbed.index[700:])
    assert c["smear_har_hat"] == a["smear_har_hat"]
