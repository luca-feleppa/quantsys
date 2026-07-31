# IT: Test della BASELINE HAR-C (C3, pre-reg STATUS 2026-07-31) — sole componenti
#     continue della decomposizione, senza i termini di salto. Serve a separare due
#     spiegazioni del guadagno di HAR-CJ su HAR-RV misurato da C2: la SOSTITUZIONE
#     del regressore (C = min(RV,BV) è jump-robust) oppure la DECOMPOSIZIONE vera e
#     propria (i salti portano informazione loro).
#     Copre: (a) l'ANNIDAMENTO stretto HAR-C ⊂ HAR-CJ, che è il perno del disegno —
#     in-sample HAR-CJ non può perdere, quindi ogni conclusione deve venire dal
#     fuori campione; (b) l'IDENTITÀ del campione fra le tre baseline, che il
#     confronto appaiato richiede; (c) il NON-LEAKAGE strutturale dei coefficienti;
#     (d) l'INERZIA — HAR-C non deve spostare di un bit né HAR-RV né HAR-CJ, contro
#     cui il claim pubblicato (banda −23% ÷ −32%) è oggi registrato.
# EN: Tests for the HAR-C BASELINE (C3, STATUS 2026-07-31 pre-reg) — continuous
#     components only, no jump terms. It separates two explanations of the HAR-CJ
#     over HAR-RV gain measured by C2: regressor SUBSTITUTION (C = min(RV,BV) is
#     jump-robust) or genuine DECOMPOSITION (jumps carry their own information).
#     Covers: (a) the strict nesting HAR-C ⊂ HAR-CJ, the pivot of the design —
#     in-sample HAR-CJ cannot lose, so every conclusion must come from out of
#     sample; (b) sample IDENTITY across the three baselines, required by the
#     paired comparison; (c) structural NON-LEAKAGE of the coefficients;
#     (d) INERTIA — HAR-C must not shift HAR-RV or HAR-CJ by one bit, since the
#     published claim (−23% ÷ −32% band) is currently registered against them.
import numpy as np
import pandas as pd

from quantsys.model.vol_metrics import (build_har_frame, build_har_cj_frame,
                                        har_c_fold_qlike, har_cj_fold_qlike,
                                        har_fold_qlike, HAR_C_COLS, HAR_CJ_COLS)

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


def _frames(seed: int = 0, jumps: bool = True):
    # IT: serie con salti sporadici (il caso in cui la decomposizione ha senso).
    # EN: series with sporadic jumps (the case where the decomposition is meaningful).
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, N)
    if jumps:
        r[rng.choice(N, 12, replace=False)] += rng.normal(0, 0.08, 12)
    raw = _raw_from_returns(r)
    return raw, build_har_frame(raw, H, BARS_DAY), build_har_cj_frame(raw, H, BARS_DAY)


def _split(frame: pd.DataFrame, frac: float = 0.7):
    k = int(len(frame) * frac)
    return frame.index[:k], frame.index[k:]


# ── (a) annidamento · nesting ────────────────────────────────────────────────

def test_har_c_e_sottoinsieme_stretto_di_har_cj():
    # IT: HAR_C_COLS ⊂ HAR_CJ_COLS, e strettamente (3 regressori contro 6). È la
    #     proprietà che rende in-sample il confronto privo di informazione.
    # EN: HAR_C_COLS ⊂ HAR_CJ_COLS, strictly (3 regressors vs 6). It is the property
    #     that makes the in-sample comparison uninformative.
    assert set(HAR_C_COLS) < set(HAR_CJ_COLS)
    assert len(HAR_C_COLS) == 3 and len(HAR_CJ_COLS) == 6


def test_in_sample_har_cj_non_puo_perdere():
    # IT: valutando sugli STESSI timestamp del fit, l'annidamento impone
    #     SSE(HAR-CJ) ≤ SSE(HAR-C) per costruzione. Se questo test fallisce, il
    #     confronto fuori campione non sta misurando ciò che crede.
    # EN: evaluating on the SAME timestamps used for the fit, nesting forces
    #     SSE(HAR-CJ) ≤ SSE(HAR-C) by construction. If this fails, the out-of-sample
    #     comparison is not measuring what it thinks.
    _, _, cj = _frames(seed=1)
    idx = cj.index
    y = cj["y"].values

    def _sse(cols):
        X = np.column_stack([np.ones(len(cj)), cj[cols].values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(np.sum((y - X @ beta) ** 2))

    assert _sse(HAR_CJ_COLS) <= _sse(HAR_C_COLS) + 1e-9
    assert len(idx) > 50


def test_fuori_campione_har_cj_puo_perdere():
    # IT: la controparte del test precedente — fuori campione l'annidamento NON
    #     protegge: cerchiamo almeno un seed in cui HAR-C batte HAR-CJ su held-out.
    #     Serve a provare che la domanda di C3 è empirica e non decisa a tavolino.
    # EN: the counterpart of the previous test — out of sample nesting does NOT
    #     protect: we look for at least one seed where HAR-C beats HAR-CJ on
    #     held-out data. It proves C3's question is empirical, not settled a priori.
    beaten = False
    for seed in range(12):
        _, _, cj = _frames(seed=seed)
        t_tr, t_ev = _split(cj)
        q_c = har_c_fold_qlike(cj, t_tr, t_ev)["qlike_har_c"]
        q_cj = har_cj_fold_qlike(cj, t_tr, t_ev)["qlike_har_cj"]
        if np.isfinite(q_c) and np.isfinite(q_cj) and q_c < q_cj:
            beaten = True
            break
    assert beaten, ("HAR-C non batte mai HAR-CJ fuori campione su 12 seed sintetici: "
                    "l'annidamento sembra protettivo anche OOS, il che invaliderebbe "
                    "il disegno / HAR-C never beats HAR-CJ out of sample")


# ── (b) identità del campione · sample identity ──────────────────────────────

def test_stesso_campione_delle_altre_baseline():
    # IT: HAR-C prende lo STESSO frame di HAR-CJ, quindi n coincide per costruzione;
    #     il confronto appaiato della pre-reg ③ richiede esattamente questo.
    # EN: HAR-C takes the SAME frame as HAR-CJ, so n matches by construction; pre-reg
    #     ③'s paired design requires exactly this.
    _, _, cj = _frames(seed=2)
    t_tr, t_ev = _split(cj)
    r_c = har_c_fold_qlike(cj, t_tr, t_ev)
    r_cj = har_cj_fold_qlike(cj, t_tr, t_ev)
    assert r_c["n_har_c"] == r_cj["n_har_cj"] == len(t_ev)
    assert r_c["n_eval"] == r_cj["n_eval"]


def test_har_c_ignora_le_colonne_di_salto():
    # IT: perturbando SOLO le colonne xj_* il risultato di HAR-C deve essere identico
    #     bit a bit, mentre HAR-CJ deve cambiare. È la prova diretta che le due
    #     baseline differiscono per il set di regressori e per nient'altro.
    # EN: perturbing ONLY the xj_* columns must leave HAR-C bit-identical while HAR-CJ
    #     changes. Direct proof that the two baselines differ by the regressor set and
    #     by nothing else.
    _, _, cj = _frames(seed=3)
    t_tr, t_ev = _split(cj)
    base_c = har_c_fold_qlike(cj, t_tr, t_ev)["qlike_har_c"]
    base_cj = har_cj_fold_qlike(cj, t_tr, t_ev)["qlike_har_cj"]

    perturbed = cj.copy()
    rng = np.random.default_rng(7)
    for col in ("xj_h", "xj_w", "xj_m"):
        perturbed[col] = perturbed[col].values + rng.normal(0, 0.5, len(perturbed))

    assert har_c_fold_qlike(perturbed, t_tr, t_ev)["qlike_har_c"] == base_c
    assert har_cj_fold_qlike(perturbed, t_tr, t_ev)["qlike_har_cj"] != base_cj


# ── (c) non-leakage strutturale · structural non-leakage ─────────────────────

def test_beta_dipende_solo_dai_timestamp_di_train():
    # IT: alterare `y` sui soli timestamp di EVAL non deve muovere i coefficienti:
    #     l'OLS è chiuso sul train, come nei gemelli HAR-RV/HAR-CJ.
    # EN: altering `y` on EVAL timestamps only must not move the coefficients: the OLS
    #     is closed on train, as in the HAR-RV/HAR-CJ twins.
    _, _, cj = _frames(seed=4)
    t_tr, t_ev = _split(cj)
    beta_ref = har_c_fold_qlike(cj, t_tr, t_ev)["beta"]

    tampered = cj.copy()
    tampered.loc[t_ev, "y"] = tampered.loc[t_ev, "y"].values + 3.0
    assert har_c_fold_qlike(tampered, t_tr, t_ev)["beta"] == beta_ref


def test_ramo_degenere_contratto_uniforme():
    # IT: train troppo corto → stesso contratto di ritorno dei gemelli (NaN + n),
    #     mai un'eccezione: il giudice deve poter riportare "non valutabile".
    # EN: train too short → same return contract as the twins (NaN + n), never an
    #     exception: the judge must be able to report "not evaluable".
    _, _, cj = _frames(seed=5)
    out = har_c_fold_qlike(cj, cj.index[:10], cj.index[10:20])
    assert np.isnan(out["qlike_har_c"])
    assert out["n_har_c"] == 10 and out["n_eval"] == 10


# ── (d) inerzia · inertia ────────────────────────────────────────────────────

def test_har_rv_e_har_cj_invariati():
    # IT: C3 non tocca le due baseline già registrate. Valori attesi ricalcolati
    #     dalle funzioni storiche sullo stesso input: qualunque scostamento
    #     significherebbe aver mosso il metro mentre lo si stava verificando.
    # EN: C3 does not touch the two already-registered baselines. Expected values
    #     recomputed from the historical functions on the same input: any deviation
    #     would mean moving the yardstick while checking it.
    _, rv, cj = _frames(seed=6)
    t_tr, t_ev = _split(rv)
    ref_rv = har_fold_qlike(rv, t_tr, t_ev)
    ref_cj = har_cj_fold_qlike(cj, cj.index[:len(t_tr)], cj.index[len(t_tr):])

    # IT/EN: chiamare HAR-C non ha effetti collaterali sui frame / no side effects
    har_c_fold_qlike(cj, cj.index[:len(t_tr)], cj.index[len(t_tr):])

    assert har_fold_qlike(rv, t_tr, t_ev) == ref_rv
    assert har_cj_fold_qlike(cj, cj.index[:len(t_tr)],
                             cj.index[len(t_tr):]) == ref_cj


def test_frame_cj_non_modificato_dalla_chiamata():
    # IT: `har_c_fold_qlike` deve essere puro — il frame passato non va mutato,
    #     altrimenti il run successivo di HAR-CJ nello stesso processo (è ciò che
    #     fa il giudice) leggerebbe dati alterati.
    # EN: `har_c_fold_qlike` must be pure — the frame passed in must not be mutated,
    #     otherwise the subsequent HAR-CJ run in the same process (which is what the
    #     judge does) would read altered data.
    _, _, cj = _frames(seed=8)
    snapshot = cj.copy(deep=True)
    t_tr, t_ev = _split(cj)
    har_c_fold_qlike(cj, t_tr, t_ev)
    pd.testing.assert_frame_equal(cj, snapshot)
