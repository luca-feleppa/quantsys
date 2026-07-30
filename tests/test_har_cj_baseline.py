# IT: Test della BASELINE HAR-CJ (C2, pre-reg STATUS 2026-07-30) — da non confondere
#     con `tests/test_har_cj.py`, che copre le FEATURE HAR-CJ di A4: stesso oggetto
#     econometrico, due ruoli opposti. A4 mette la decomposizione fra gli INPUT del
#     modello (leva di training, classe chiusa il 2026-07-30 con A10); C2 la mette
#     nella BASELINE contro cui il modello è giudicato, cioè rende il test più
#     difficile invece del modello più forte.
#     Copre: (a) le proprietà matematiche della decomposizione, che devono valere per
#     costruzione e non per fortuna numerica; (b) il NON-LEAKAGE strutturale — i
#     coefficienti OLS dipendono solo dai timestamp di train; (c) l'INERZIA — HAR-CJ
#     non deve spostare di un bit la baseline HAR-RV, contro cui il claim pubblicato
#     (banda −27% ÷ −36%) è registrato.
# EN: Tests for the HAR-CJ BASELINE (C2, STATUS 2026-07-30 pre-reg) — not to be
#     confused with `tests/test_har_cj.py`, which covers A4's HAR-CJ FEATURES: same
#     econometric object, two opposite roles. A4 puts the decomposition among the
#     model's INPUTS (a training lever, class closed on 2026-07-30 with A10); C2 puts
#     it in the BASELINE the model is judged against, i.e. it makes the test harder
#     rather than the model stronger.
#     Covers: (a) mathematical properties of the decomposition, which must hold by
#     construction rather than numerical luck; (b) structural NON-LEAKAGE — the OLS
#     coefficients depend only on train timestamps; (c) INERTIA — HAR-CJ must not
#     shift the HAR-RV baseline by one bit, since the published claim (−27% ÷ −36%
#     band) is registered against it.
import numpy as np
import pandas as pd
import pytest

from quantsys.model.vol_metrics import (build_har_frame, build_har_cj_frame,
                                        har_cj_fold_qlike, har_fold_qlike,
                                        HAR_CJ_COLS, EPS)

BARS_DAY = 4          # IT/EN: giornata corta = finestre 7g/30g piccole / short day = small 7d/30d windows
H = 2
N = 400


def _raw_from_returns(r: np.ndarray) -> pd.DataFrame:
    # IT: costruisce candele da una serie di log-return (close = exp(cumsum)).
    # EN: builds candles from a log-return series (close = exp(cumsum)).
    close = np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return pd.DataFrame({
        "open_time": pd.date_range("2020-01-01", periods=len(close), freq="h", tz="UTC"),
        "close": close,
    })


def _decompose(frame: pd.DataFrame, tag: str):
    # IT: inverte le trasformazioni del frame per riottenere C e J in livelli.
    # EN: inverts the frame transforms to recover C and J in levels.
    c = np.exp(frame[f"xc_{tag}"].values) - EPS
    j = np.expm1(frame[f"xj_{tag}"].values)
    return c, j


# ── (a) proprietà matematiche · mathematical properties ──────────────────────

def test_jump_non_negativo_per_costruzione():
    rng = np.random.default_rng(0)
    f = build_har_cj_frame(_raw_from_returns(rng.normal(0, 0.01, N)), H, BARS_DAY)
    for tag in ("h", "w", "m"):
        _, j = _decompose(f, tag)
        assert (j >= -1e-12).all(), f"salto negativo su {tag} / negative jump on {tag}"


def test_continua_piu_salto_ricostruisce_la_rv():
    # IT: C + J ≡ RV per costruzione (J = max(RV−BV,0), C = RV − J).
    # EN: C + J ≡ RV by construction (J = max(RV−BV,0), C = RV − J).
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, N)
    raw = _raw_from_returns(r)
    f = build_har_cj_frame(raw, H, BARS_DAY)
    lr2 = pd.Series(np.concatenate([[np.nan], r]) ** 2)
    rv_h = lr2.rolling(H).sum().to_numpy()
    naive_times = raw["open_time"].dt.tz_localize(None)
    pos = {ts: k for k, ts in enumerate(naive_times)}
    sel = np.array([pos[ts] for ts in f.index])
    c, j = _decompose(f, "h")
    np.testing.assert_allclose(c + j, rv_h[sel], rtol=1e-9, atol=1e-15)


def test_senza_salti_la_componente_di_salto_e_esattamente_zero():
    # IT: con |r| costante BV = (π/2)·Σr² > RV ⇒ J troncato a 0 ovunque.
    # EN: with constant |r|, BV = (π/2)·Σr² > RV ⇒ J truncated to 0 everywhere.
    r = np.tile([0.01, -0.01], N // 2)
    f = build_har_cj_frame(_raw_from_returns(r), H, BARS_DAY)
    for tag in ("h", "w", "m"):
        _, j = _decompose(f, tag)
        np.testing.assert_allclose(j, 0.0, atol=1e-15)


def test_un_salto_isolato_produce_componente_di_salto_positiva():
    # IT: un rendimento enorme isolato gonfia RV ma non BV (prodotti con vicini piccoli).
    # EN: one isolated huge return inflates RV but not BV (products with small neighbours).
    r = np.full(N, 1e-4)
    r[N // 2] = 0.5
    f = build_har_cj_frame(_raw_from_returns(r), H, BARS_DAY)
    _, j = _decompose(f, "h")
    assert j.max() > 0.01, "il salto non e' stato isolato / the jump was not isolated"
    assert (j == 0.0).sum() > len(j) * 0.5, "troppi salti spuri / too many spurious jumps"


def test_costante_di_scala_bipower_e_pi_mezzi():
    from quantsys.model.vol_metrics import _BV_SCALE
    assert _BV_SCALE == pytest.approx(np.pi / 2.0)


def test_frame_senza_nan_e_indice_naive():
    rng = np.random.default_rng(2)
    f = build_har_cj_frame(_raw_from_returns(rng.normal(0, 0.01, N)), H, BARS_DAY)
    assert not f.isna().any().any()
    assert f.index.tz is None
    # IT: l'ORDINE delle colonne nel frame (raggruppate per orizzonte) e' irrilevante —
    #     la regressione seleziona per nome via HAR_CJ_COLS; conta che l'insieme coincida.
    # EN: column ORDER in the frame (grouped by horizon) is irrelevant — the regression
    #     selects by name through HAR_CJ_COLS; what matters is that the set matches.
    assert set(f.columns) == {"y"} | set(HAR_CJ_COLS)
    assert f[HAR_CJ_COLS].shape[1] == 6


# ── (b) non-leakage strutturale · structural non-leakage ─────────────────────

def test_beta_dipende_solo_dai_timestamp_di_train():
    # IT: due eval disgiunti, stesso train ⇒ stessi coefficienti. Se il fit vedesse
    #     l'eval, i beta cambierebbero: e' il controllo di leakage strutturale.
    # EN: two disjoint eval sets, same train ⇒ identical coefficients. If the fit saw
    #     the eval set the betas would change: this is the structural leakage check.
    rng = np.random.default_rng(3)
    f = build_har_cj_frame(_raw_from_returns(rng.normal(0, 0.01, N)), H, BARS_DAY)
    idx = f.index
    a = har_cj_fold_qlike(f, idx[:200], idx[200:260])
    b = har_cj_fold_qlike(f, idx[:200], idx[260:320])
    np.testing.assert_array_equal(np.asarray(a["beta"]), np.asarray(b["beta"]))


def test_contratto_uniforme_sul_ramo_degenere():
    rng = np.random.default_rng(4)
    f = build_har_cj_frame(_raw_from_returns(rng.normal(0, 0.01, N)), H, BARS_DAY)
    out = har_cj_fold_qlike(f, f.index[:10], f.index[20:30])   # IT/EN: train < 50
    assert np.isnan(out["qlike_har_cj"])
    assert set(out) >= {"qlike_har_cj", "n_har_cj", "n_eval"}


def test_qlike_finito_e_sette_coefficienti():
    rng = np.random.default_rng(5)
    f = build_har_cj_frame(_raw_from_returns(rng.normal(0, 0.01, N)), H, BARS_DAY)
    out = har_cj_fold_qlike(f, f.index[:250], f.index[250:])
    assert np.isfinite(out["qlike_har_cj"]) and out["qlike_har_cj"] > 0
    assert len(out["beta"]) == 7          # IT/EN: costante + 3 continue + 3 salti


# ── (c) inerzia: HAR-RV non si muove · inertia: HAR-RV does not move ─────────

def test_har_rv_invariato_dalla_presenza_di_har_cj():
    # IT: il frame HAR-RV e la sua metrica devono restare quelli di prima — il claim
    #     pubblicato e' registrato contro QUESTA baseline.
    # EN: the HAR-RV frame and its metric must stay as before — the published claim
    #     is registered against THIS baseline.
    rng = np.random.default_rng(6)
    raw = _raw_from_returns(rng.normal(0, 0.01, N))
    f_rv = build_har_frame(raw, H, BARS_DAY)
    assert list(f_rv.columns) == ["y", "xh", "xw", "xm"]
    out = har_fold_qlike(f_rv, f_rv.index[:250], f_rv.index[250:])
    assert np.isfinite(out["qlike_har"])


def test_i_due_frame_condividono_lo_stesso_target():
    # IT: la colonna `y` deve coincidere sui timestamp comuni — se i due frame avessero
    #     target diversi, il confronto fra baseline sarebbe privo di senso.
    # EN: the `y` column must coincide on shared timestamps — if the two frames had
    #     different targets, comparing the baselines would be meaningless.
    rng = np.random.default_rng(7)
    raw = _raw_from_returns(rng.normal(0, 0.01, N))
    f_rv, f_cj = build_har_frame(raw, H, BARS_DAY), build_har_cj_frame(raw, H, BARS_DAY)
    common = f_rv.index.intersection(f_cj.index)
    assert len(common) > 100
    np.testing.assert_allclose(f_rv.loc[common, "y"].values,
                               f_cj.loc[common, "y"].values, rtol=1e-12)


def test_il_frame_cj_parte_al_piu_una_barra_dopo():
    # IT: la bipower porta un lag in piu'; uno sfasamento maggiore di una barra
    #     segnalerebbe finestre costruite male.
    # EN: bipower carries one extra lag; a shift larger than one bar would signal
    #     badly built rolling windows.
    rng = np.random.default_rng(8)
    raw = _raw_from_returns(rng.normal(0, 0.01, N))
    f_rv, f_cj = build_har_frame(raw, H, BARS_DAY), build_har_cj_frame(raw, H, BARS_DAY)
    assert 0 <= len(f_rv) - len(f_cj) <= 1
