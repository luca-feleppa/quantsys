# IT: Regression test — GUARD anti-degradazione silenziosa del walk-forward regime.
#     Prima: ogni fit Markov-Switching fallito finiva in un log.warning per timestep e
#     il loop proseguiva. Con current_params=None, probs_all[t] non veniva MAI scritto:
#     fit_predict_walkforward restituiva la PRIOR UNIFORME (1/n_regimes) travestita da
#     probabilita' di regime, senza che nulla fallisse. Il sintomo si manifestava molto
#     piu' a valle (continue_walkforward) o non si manifestava affatto.
#     Questi test bloccano il comportamento nuovo: fallimento ESPLICITO, e path di
#     successo invariato (la bit-parity B7 e' coperta da tests/test_regime_incremental.py).
# EN: Regression test — anti-silent-degradation GUARD for the regime walk-forward.
#     Before: every failed Markov-Switching fit produced one log.warning per timestep and
#     the loop carried on. With current_params=None, probs_all[t] was NEVER written:
#     fit_predict_walkforward returned the UNIFORM PRIOR (1/n_regimes) dressed up as
#     regime probabilities, without anything failing. The symptom surfaced much further
#     downstream (continue_walkforward) or not at all.
#     These tests lock in the new behaviour: EXPLICIT failure, success path unchanged
#     (B7 bit-parity is covered by tests/test_regime_incremental.py).

import numpy as np
import pandas as pd
import pytest

from quantsys.macro.regime import RegimeMarkovSwitching

N_BARS  = 1200
BURN_IN = 200
RETRAIN = 400
ENGINE_KW = dict(n_regimes=2, n_iter=50, random_state=42, n_pca=1, n_restarts=1)


# IT: stessa serie sintetica a 2 regimi di varianza usata dai test B7.
# EN: same synthetic 2-variance-regime series used by the B7 tests.
def _make_df(n: int = N_BARS, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        stay = 0.97 if state[t - 1] == 0 else 0.95
        state[t] = state[t - 1] if rng.random() < stay else 1 - state[t - 1]
    sigma = np.where(state == 0, 0.5, 2.0)
    ret   = rng.normal(0.0, sigma)
    rv    = np.log(np.maximum(ret ** 2, 1e-12))
    return pd.DataFrame({"log_ret_h": ret, "log_rv": rv})


# IT: ① il caso osservato in produzione — il fit solleva SEMPRE (statsmodels assente).
#     Prima: DataFrame di prior uniformi, zero eccezioni. Ora: RuntimeError.
# EN: ① the case observed in production — the fit ALWAYS raises (statsmodels missing).
#     Before: DataFrame of uniform priors, zero exceptions. Now: RuntimeError.
def test_zero_successful_fits_raises(monkeypatch):
    eng = RegimeMarkovSwitching(**ENGINE_KW)

    def _boom(self, pc1):
        raise ImportError("No module named 'statsmodels'")

    monkeypatch.setattr(RegimeMarkovSwitching, "_fit_single", _boom)
    with pytest.raises(RuntimeError, match="NESSUN fit riuscito"):
        eng.fit_predict_walkforward(_make_df(), burn_in_days=BURN_IN,
                                    retrain_days=RETRAIN)


# IT: ② ramo muto — _fit_single ritorna None (tutti i restart falliscono) senza
#     sollevare. Prima non produceva NEMMENO un log: il piu' silenzioso dei due.
# EN: ② the mute branch — _fit_single returns None (all restarts fail) without
#     raising. Before it produced NOT EVEN a log: the more silent of the two.
def test_all_fits_return_none_raises(monkeypatch):
    eng = RegimeMarkovSwitching(**ENGINE_KW)
    monkeypatch.setattr(RegimeMarkovSwitching, "_fit_single",
                        lambda self, pc1: None)
    with pytest.raises(RuntimeError, match="NESSUN fit riuscito"):
        eng.fit_predict_walkforward(_make_df(), burn_in_days=BURN_IN,
                                    retrain_days=RETRAIN)


# IT: ③ il messaggio deve dire cosa sarebbe stato restituito, non solo "errore":
#     e' l'informazione che rende diagnosticabile il guasto.
# EN: ③ the message must state what would have been returned, not just "error":
#     that is the information that makes the failure diagnosable.
def test_error_message_names_the_uniform_prior(monkeypatch):
    eng = RegimeMarkovSwitching(**ENGINE_KW)
    monkeypatch.setattr(RegimeMarkovSwitching, "_fit_single",
                        lambda self, pc1: None)
    with pytest.raises(RuntimeError) as ei:
        eng.fit_predict_walkforward(_make_df(), burn_in_days=BURN_IN,
                                    retrain_days=RETRAIN)
    msg = str(ei.value)
    assert "prior uniforme" in msg
    assert "statsmodels" in msg


# IT: ④ path di successo — nessuna regressione: il guard non deve scattare su un
#     walk-forward sano, e deve popolare la diagnostica.
# EN: ④ success path — no regression: the guard must not fire on a healthy
#     walk-forward, and must populate the diagnostics.
def test_healthy_walkforward_passes_and_reports():
    eng = RegimeMarkovSwitching(**ENGINE_KW)
    out = eng.fit_predict_walkforward(_make_df(), burn_in_days=BURN_IN,
                                      retrain_days=RETRAIN)
    assert len(out) == N_BARS
    d = eng.last_fit_diagnostics
    assert d is not None
    assert d["fit_ok"] >= 1
    assert d["fail_ratio"] == pytest.approx(0.0)
    # IT: copertura = frazione di timestep post-burn-in con probabilita' filtrate.
    # EN: coverage = fraction of post-burn-in timesteps with filtered probabilities.
    assert d["coverage"] > 0.99


# IT: ⑤ la soglia di rapporto e' configurabile, ma l'abort su ZERO fit non lo e':
#     un walk-forward senza un solo fit non produce informazione in nessun caso.
# EN: ⑤ the ratio threshold is configurable, but the zero-fit abort is not:
#     a walk-forward without a single fit yields no information in any case.
def test_zero_fit_abort_not_disableable_by_threshold(monkeypatch):
    eng = RegimeMarkovSwitching(**{**ENGINE_KW, "max_fit_failure_ratio": 1.0})
    monkeypatch.setattr(RegimeMarkovSwitching, "_fit_single",
                        lambda self, pc1: None)
    with pytest.raises(RuntimeError, match="NESSUN fit riuscito"):
        eng.fit_predict_walkforward(_make_df(), burn_in_days=BURN_IN,
                                    retrain_days=RETRAIN)


# IT: ⑥ validazione del parametro nuovo (fail-fast su valore assurdo).
# EN: ⑥ validation of the new parameter (fail-fast on absurd value).
@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_invalid_failure_ratio_rejected(bad):
    with pytest.raises(ValueError, match="max_fit_failure_ratio"):
        RegimeMarkovSwitching(max_fit_failure_ratio=bad)


# IT: ⑦ MIRROR incrementale — un append che attraversa un confine di refit e non
#     riesce a rifittare prosegue su parametri STANTII: deve fallire, non tacere.
# EN: ⑦ incremental MIRROR — an append crossing a refit boundary that fails to
#     refit carries on with STALE parameters: it must fail, not stay quiet.
def test_incremental_stale_params_raises(monkeypatch):
    df = _make_df()
    eng = RegimeMarkovSwitching(**ENGINE_KW)
    eng.fit_predict_walkforward(df.iloc[:800], burn_in_days=BURN_IN,
                                retrain_days=RETRAIN)
    state = eng._wf_state
    assert state is not None and state["params"] is not None

    # IT: da qui in poi ogni refit fallisce; l'append copre >RETRAIN barre nuove
    #     quindi almeno un retrain e' dovuto.
    # EN: from here every refit fails; the append spans >RETRAIN new bars so at
    #     least one retrain is due.
    monkeypatch.setattr(RegimeMarkovSwitching, "_fit_single",
                        lambda self, pc1: None)
    with pytest.raises(RuntimeError, match="NESSUNO"):
        eng.continue_walkforward(df, state)
