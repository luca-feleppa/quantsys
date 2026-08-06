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


def test_count_only_computes_nothing_decisional_at_any_n(tmp_path):
    # IT: IL test che rende sicura l'automazione. `--count-only` gira nella routine di
    #     sessione a ogni avvio, quindi deve restare muto sui numeri ANCHE sopra la
    #     soglia: il guard n<40 -> NO_RUN protegge solo sotto, e un `--stage 2` nudo a
    #     n>=40 stamperebbe il verdetto confermativo per automazione. Qui si costruisce
    #     un campione ABBONDANTEMENTE sopra la soglia (60 expiry) e si pretende che lo
    #     stdout non contenga nessuna delle etichette decisionali e che nessun report
    #     venga scritto. Se un domani qualcuno spostasse l'uscita anticipata sotto il
    #     calcolo, questo test cade.
    # EN: THE test that makes the automation safe. `--count-only` runs in the session
    #     routine at every startup, so it must stay silent about numbers EVEN above
    #     threshold: the n<40 -> NO_RUN guard only protects below, and a bare
    #     `--stage 2` at n>=40 would print the confirmatory verdict by automation. Here
    #     a sample WELL above threshold (60 expiries) is built and stdout must contain
    #     none of the decisional labels, with no report written. If someone later moved
    #     the early exit below the computation, this test fails.
    import subprocess
    import sys

    n_days = 60
    start = pd.Timestamp("2026-08-02 00:00", tz="UTC")
    # IT: un tick orario continuo copre ogni finestra di decisione [E-33h, E-27h].
    # EN: a continuous hourly tick covers every decision window [E-33h, E-27h].
    idx = pd.date_range(start, periods=24 * (n_days + 3), freq="h", tz="UTC")
    rng = np.random.default_rng(7)
    fc = pd.DataFrame({
        "candle_ts": idx,
        "rv_pred": rng.uniform(1e-4, 3e-4, len(idx)),
        "var_iv": rng.uniform(1e-4, 3e-4, len(idx)),
    })
    closes = pd.DataFrame({
        "open_time": idx,
        "close": 64000.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, len(idx)))),
    })

    root = tmp_path
    (root / "results" / "vol_paper").mkdir(parents=True)
    (root / "data").mkdir()
    fc.to_parquet(root / "results" / "vol_paper" / "forecasts.parquet")
    closes.to_parquet(root / "data" / "raw_candles.parquet")
    closes.to_parquet(root / "data" / "raw_candles_1m_l2.parquet")

    # IT: il giudice risolve i path da parents[2] del PROPRIO file: si copia lo script
    #     nell'albero temporaneo invece di scriverne uno finto.
    # EN: the judge resolves paths from parents[2] of its OWN file: the script is copied
    #     into the temporary tree rather than faked.
    (root / "scripts" / "vol").mkdir(parents=True)
    src = ROOT / "scripts" / "vol" / "edge_information_judge.py"
    (root / "scripts" / "vol" / "edge_information_judge.py").write_bytes(src.read_bytes())

    out = subprocess.run(
        [sys.executable, str(root / "scripts" / "vol" / "edge_information_judge.py"),
         "--stage", "2", "--count-only"],
        capture_output=True, text=True, cwd=str(ROOT), env={**__import__("os").environ,
                                                            "PYTHONPATH": str(ROOT)})
    assert out.returncode == 0, out.stderr
    n_line = [l for l in out.stdout.splitlines() if "osservabili" in l]
    assert n_line and int(n_line[0].split(":")[-1]) >= ej_n_min(), \
        f"campione non sopra soglia, il test non prova nulla: {out.stdout}"
    for label in ("ACCORDO DI SEGNO", "SPEARMAN", "CONTROLLO POSITIVO", "VERDETTO", "PASS", "FAIL"):
        assert label not in out.stdout, f"--count-only ha stampato '{label}'"
    assert not (root / "results" / "vol_paper" / "edge_information_stage2.json").exists()


def ej_n_min():
    # IT: rilettura della soglia dalla sorgente, non una costante duplicata nel test.
    # EN: threshold re-read from the source, not a constant duplicated in the test.
    spec = importlib.util.spec_from_file_location(
        "edge_information_judge", ROOT / "scripts" / "vol" / "edge_information_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.N_MIN_STAGE2


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
