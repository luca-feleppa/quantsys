# IT: Test B7 — walk-forward regime INCREMENTALE (quantsys/macro/regime.py).
#     Golden test di bit-parity: full run vs (run troncato a _stop_at + continuazione
#     dal checkpoint di catena) devono coincidere BIT-PER-BIT sulle barre continuate.
#     Il design _stop_at garantisce scaler identico nei due run (fit sull'intero df
#     in entrambi): l'unica differenza ammessa è l'orchestrazione dei refit.
# EN: B7 tests — INCREMENTAL regime walk-forward (quantsys/macro/regime.py).
#     Bit-parity golden test: full run vs (run truncated at _stop_at + continuation
#     from the chain checkpoint) must match BIT-FOR-BIT on the continued bars.
#     The _stop_at design guarantees an identical scaler in both runs (fit on the
#     whole df in both): the only allowed difference is the refit orchestration.

import numpy as np
import pandas as pd
import pytest

from quantsys.macro.regime import RegimeMarkovBTC, RegimeMarkovSwitching

# IT: dimensioni piccole ma con ≥1 retrain PRIMA e ≥1 DOPO lo split (t=200, 800 | 1400).
# EN: small sizes but with ≥1 retrain BEFORE and ≥1 AFTER the split (t=200, 800 | 1400).
N_BARS  = 1600
BURN_IN = 200
RETRAIN = 600
SPLIT   = 1000

ENGINE_KW = dict(n_regimes=2, n_iter=50, random_state=42, n_pca=1, n_restarts=1)


# IT: serie sintetica a 2 regimi di varianza (Markov persistente) — stessa forma
#     colonne del df BTC orario (log_ret_h, log_rv).
# EN: synthetic 2-variance-regime series (persistent Markov) — same column shape
#     as the hourly BTC df (log_ret_h, log_rv).
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


# IT: fixture module-scope: i fit MLE sono il costo dominante, girano una volta sola.
# EN: module-scope fixture: MLE fits dominate the cost, run once.
@pytest.fixture(scope="module")
def runs():
    df = _make_df()

    eng_full = RegimeMarkovSwitching(**ENGINE_KW)
    full = eng_full.fit_predict_walkforward(
        df, burn_in_days=BURN_IN, retrain_days=RETRAIN,
    )

    eng_part = RegimeMarkovSwitching(**ENGINE_KW)
    part = eng_part.fit_predict_walkforward(
        df, burn_in_days=BURN_IN, retrain_days=RETRAIN, _stop_at=SPLIT,
    )
    state = eng_part._wf_state
    cont  = eng_part.continue_walkforward(df, state)

    return {"df": df, "full": full, "part": part,
            "cont": cont, "state": state, "eng_part": eng_part}


PROB_COLS = [f"regime_prob_{i}" for i in range(ENGINE_KW["n_regimes"])]


# IT: ① parity BIT-PER-BIT delle probabilità sulle barre continuate.
# EN: ① BIT-FOR-BIT probability parity on the continued bars.
def test_bit_parity_probs(runs):
    ref = runs["full"][PROB_COLS].values[SPLIT:]
    got = runs["cont"][PROB_COLS].values
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"divergenza max {np.abs(got - ref).max():.3e} — il mirror del blocco "
        f"retrain in continue_walkforward è divergito dal run pieno"
    )


# IT: ② parity delle etichette dominanti e del flag burn-in.
# EN: ② dominant-label and burn-in flag parity.
def test_bit_parity_labels(runs):
    ref = runs["full"]["regime_dominant"].values[SPLIT:]
    got = runs["cont"]["regime_dominant"].values
    assert np.array_equal(got, ref)
    assert not runs["cont"]["regime_burn_in"].any()


# IT: ③ lo stato viene aggiornato in place per il re-save del checkpoint:
#     n_bars avanza a N e l'ultimo retrain è quello atteso dalla cadenza (t=1400).
# EN: ③ the state is updated in place for the checkpoint re-save: n_bars advances
#     to N and the last retrain is the cadence-expected one (t=1400).
def test_state_advances(runs):
    st = runs["state"]
    assert st["n_bars"] == N_BARS
    expected_last = BURN_IN + RETRAIN * ((N_BARS - 1 - BURN_IN) // RETRAIN)
    assert st["last_retrain"] == expected_last
    assert st["params"] is not None
    # IT: il posteriore filtrato finale coincide con l'ultima riga del run pieno.
    # EN: the final filtered posterior matches the full run's last row.
    assert np.array_equal(
        st["last_filtered"], runs["full"][PROB_COLS].values[-1]
    )


# IT: ④ guard di continuità: storia più corta dello stato → RuntimeError (fail-fast).
# EN: ④ continuity guard: history shorter than the state → RuntimeError (fail-fast).
def test_truncated_history_fails(runs):
    eng = runs["eng_part"]
    stale = dict(runs["state"])  # IT: n_bars == N_BARS | EN: n_bars == N_BARS
    with pytest.raises(RuntimeError, match="più corta"):
        eng.continue_walkforward(runs["df"].iloc[: N_BARS - 10], stale)


# IT: ⑤ _stop_at è inerte per default (None): il path production resta bit-invariato.
# EN: ⑤ _stop_at is inert by default (None): the production path stays bit-identical.
def test_stop_at_inert_before_split(runs):
    ref = runs["full"][PROB_COLS].values[:SPLIT]
    got = runs["part"][PROB_COLS].values[:SPLIT]
    assert np.array_equal(got, ref)


# IT: ⑥ roundtrip del checkpoint via pickle atomico: arrays e metadati sopravvivono.
# EN: ⑥ checkpoint roundtrip through the atomic pickle: arrays and metadata survive.
def test_checkpoint_roundtrip(runs, tmp_path):
    import pickle

    model = RegimeMarkovBTC(n_regimes=ENGINE_KW["n_regimes"])
    model._engine = runs["eng_part"]
    ts = pd.Timestamp("2026-06-22 14:00:00", tz="UTC")
    ckpt = model.build_wf_checkpoint(runs["state"], ts)

    path = tmp_path / "regime_wf_checkpoint.pkl"
    model.save_wf_checkpoint(ckpt, str(path))
    assert path.exists() and not (tmp_path / "regime_wf_checkpoint.pkl.tmp").exists()

    with open(path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded["schema_version"] == RegimeMarkovBTC._WF_CKPT_SCHEMA
    assert loaded["last_timestamp"] == ts
    assert loaded["n_regimes"] == ENGINE_KW["n_regimes"]
    for key in ("trans", "means", "variances"):
        assert np.array_equal(loaded["chain"]["params"][key],
                              runs["state"]["params"][key])
    assert np.array_equal(loaded["chain"]["last_filtered"],
                          runs["state"]["last_filtered"])
    assert loaded["chain"]["n_bars"] == runs["state"]["n_bars"]
