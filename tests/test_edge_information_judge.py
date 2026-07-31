"""
IT: Test del giudice E1 (`scripts/vol/edge_information_judge.py`).
    Due cose vanno protette. (1) Le COSTANTI PRE-REGISTRATE: la pre-reg del
    2026-07-31 congela orizzonte, finestra del tick di decisione, lag HAC, blocco
    bootstrap, alpha e n minimo — cambiarne una a risultati visti e' goalpost-moving,
    e una sentinella nei test rende la modifica impossibile in silenzio (stesso
    pattern del giudice B1). (2) Il contratto ANTI-SOTTO-CONTEGGIO di `realized_rv`:
    una finestra con un buco deve restituire None, MAI uno zero o una somma parziale
    — uno zero entrerebbe nel pannello come "varianza realizzata nulla" e sposterebbe
    sia il segno sia la QLIKE, che e' il modo piu' diretto di corrompere il gate.
EN: Tests for the E1 judge (`scripts/vol/edge_information_judge.py`).
    Two things need protecting. (1) The PRE-REGISTERED CONSTANTS: the 2026-07-31
    pre-registration freezes horizon, decision-tick window, HAC lag, bootstrap
    block, alpha and minimum n — changing one after seeing results is
    goalpost-moving, and a sentinel test makes a silent change impossible (same
    pattern as the B1 judge). (2) The ANTI-UNDER-COUNTING contract of
    `realized_rv`: a window with a gap must return None, NEVER a zero or a partial
    sum — a zero would enter the panel as "zero realized variance" and would shift
    both the sign and the QLIKE, the most direct way to corrupt the gate.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ej():
    spec = importlib.util.spec_from_file_location(
        "edge_information_judge", ROOT / "scripts" / "vol" / "edge_information_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_preregistered_constants_sentinel(ej):
    # IT: SENTINELLA. Questi valori sono nella pre-reg committata: se un test fallisce
    #     qui, o la pre-reg e' stata violata o il test va aggiornato INSIEME a una
    #     nuova pre-registrazione — mai da solo.
    # EN: SENTINEL. These values are in the committed pre-registration: a failure
    #     here means either the pre-reg was violated or the test must be updated
    #     TOGETHER with a new pre-registration — never on its own.
    assert ej.H_BARS == 30
    assert (ej.DECISION_LO, ej.DECISION_HI) == (27.0, 33.0)
    assert ej.EXPIRY_HOUR_UTC == 8
    assert ej.HAC_LAG == 1
    assert ej.DM_H == 2
    assert ej.BLOCK == 2
    assert ej.N_BOOT == 10000
    assert ej.ALPHA == 0.05
    assert ej.N_MIN_STAGE2 == 40
    assert ej.STAGE2_CUTOFF == pd.Timestamp("2026-08-01 00:00", tz="UTC")


def _closes(start, n, step=1.0):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.Series(64000.0 + step * np.arange(n), index=idx)


def test_realized_rv_uses_exactly_30_bars_from_the_tick(ej):
    # IT: commensurabilita': rv_pred e' a 30 barre e var_iv a tenor 30h, quindi la RV
    #     realizzata deve coprire 30 rendimenti = 31 close, non "fino a scadenza".
    # EN: commensurability: rv_pred is 30 bars and var_iv a 30h tenor, so realized RV
    #     must span 30 returns = 31 closes, not "until expiry".
    c = _closes("2026-07-20 00:00", 40)
    t0 = pd.Timestamp("2026-07-20 00:00", tz="UTC")
    got = ej.realized_rv(c, t0)
    want = float(np.sum(np.diff(np.log(c.iloc[:31].to_numpy())) ** 2))
    assert got == pytest.approx(want, rel=1e-12)


def test_realized_rv_returns_none_on_a_gap_never_zero(ej):
    # IT: IL test che conta. Un buco DENTRO la finestra invalida l'osservazione.
    # EN: THE test that matters. A gap INSIDE the window invalidates the observation.
    c = _closes("2026-07-20 00:00", 40)
    holed = c.drop(c.index[10])
    out = ej.realized_rv(holed, pd.Timestamp("2026-07-20 00:00", tz="UTC"))
    assert out is None, "buco riempito o somma parziale: il pannello sarebbe corrotto"


def test_realized_rv_returns_none_when_window_runs_past_the_data(ej):
    # IT: expiry troppo recente per essere liquidata: None, non una somma su meno ore.
    # EN: expiry too recent to have settled: None, not a sum over fewer hours.
    c = _closes("2026-07-20 00:00", 20)
    assert ej.realized_rv(c, pd.Timestamp("2026-07-20 00:00", tz="UTC")) is None


def test_hac_mean_test_recovers_the_mean_and_is_one_sided(ej):
    # IT: su una serie di indicatori tutti a 1 la media e' 1 e il test contro 0.5 e'
    #     a una coda a destra (p piccolo). Su tutti a 0, p deve essere ~1: il test
    #     NON deve segnalare un effetto quando il segno e' quello sbagliato.
    # EN: on an all-ones indicator series the mean is 1 and the test against 0.5 is
    #     right-tailed (small p). On all-zeros, p must be ~1: the test must NOT flag
    #     an effect when the sign is the wrong one.
    hi = ej.hac_mean_test(np.ones(30), 0.5)
    assert hi["mean"] == 1.0 and hi["p_value"] < 1e-6
    lo = ej.hac_mean_test(np.zeros(30), 0.5)
    assert lo["mean"] == 0.0 and lo["p_value"] > 0.999


def test_spearman_block_bootstrap_ci_brackets_a_strong_relation(ej):
    # IT: sanita' della macchina d'inferenza: su una relazione monotona forte l'IC
    #     deve stare tutto sopra lo zero; il blocco 2 non deve romperla.
    # EN: inference sanity: on a strong monotone relation the CI must lie entirely
    #     above zero; block 2 must not break it.
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 40)
    y = x + rng.normal(0, 0.05, 40)
    ej.N_BOOT, saved = 400, ej.N_BOOT   # IT/EN: bootstrap ridotto per il test
    try:
        lo, hi = ej.block_bootstrap_spearman(x, y, np.random.default_rng(1))
    finally:
        ej.N_BOOT = saved
    assert lo > 0.0 and hi <= 1.0
