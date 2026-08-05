# IT: Guard di identità del VINTAGE MACRO modello↔dataset (M1, 2026-08-05).
#     Il difetto che chiude: il guard sullo scaler copre il RobustScaler dei PREZZI e
#     `target_scale`, non la normalizzazione macro — che non vive nel `PipelineState`
#     canonico, perché `01_download_data` lo scrive prima che `01b` esista. Quindi due
#     modelli addestrati su blocchi macro DIVERSI passano entrambi `matches: true`.
#     ⚠ Perché è insidioso quanto il caso dello scaler, e per certi versi peggio: la
#     macro è INPUT del modello (90 colonne, embedding attivo) ma non entra in nessuna
#     baseline HAR, che leggono solo RV/target. Il NN si sposta e le baseline restano
#     identiche cifra per cifra — cioè il controllo di vintage che si userebbe per
#     accorgersene (le baseline coincidono ⇒ "stesso dataset") CONFERMA la conclusione
#     sbagliata. Scarto misurato su un caso reale: 0.0019 sul rapporto pubblicato.
# EN: Model↔dataset MACRO VINTAGE identity guard (M1, 2026-08-05).
#     The defect it closes: the scaler guard covers the PRICE RobustScaler and
#     `target_scale`, not macro normalization — which does not live in the canonical
#     `PipelineState`, since `01_download_data` writes it before `01b` runs. So two
#     models trained on DIFFERENT macro blocks both pass `matches: true`.
#     ⚠ Why this is as insidious as the scaler case, and in one way worse: macro is a
#     model INPUT (90 columns, embedding active) but enters no HAR baseline, which read
#     RV/target only. The NN moves while baselines stay identical digit for digit — so
#     the very vintage check one would use to notice (baselines agree ⇒ "same dataset")
#     CONFIRMS the wrong conclusion. Measured gap on a real case: 0.0019 on the
#     published ratio.
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantsys.utils import (PipelineState, assert_model_dataset_scaler,  # noqa: E402
                            check_model_dataset_macro, macro_fingerprint)


class _Scaler:
    """IT: scaler minimo, solo la superficie che l'impronta legge.
    EN: minimal scaler, only the surface the fingerprint reads."""

    def __init__(self, center, scale):
        self.center_ = np.asarray(center, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)


def _npz(seed=0, n_feat=3, names=("m_a", "m_b", "m_c"), dtype=np.float32):
    # IT: npz sintetico con la sola superficie macro. `macro_fingerprint` accetta un
    #     mapping, quindi il controllo positivo NON richiede di copiare i 3.26 GB
    #     dell'npz reale — che è anche il motivo per cui è eseguibile.
    # EN: synthetic npz exposing only the macro surface. `macro_fingerprint` accepts a
    #     mapping, so the positive control does NOT require copying the real 3.26 GB
    #     npz — which is also why it is runnable at all.
    rng = np.random.default_rng(seed)
    return {"X_macro_train": rng.normal(size=(40, n_feat)).astype(dtype),
            "X_macro_val": rng.normal(size=(10, n_feat)).astype(dtype),
            "macro_feature_names": np.array(list(names)),
            "n_macro_features": np.array([n_feat])}


def _state(npz=None, source="measured"):
    st = PipelineState()
    st.scale_cols = ["a", "target_ret"]
    st.scaler = _Scaler([1.0, -7.0], [2.0, 1.4])
    if npz is not None:
        st.macro_vintage_fp = macro_fingerprint(npz)
        st.macro_vintage_fp_source = source
    return st


def _isolate(tmp_path, monkeypatch):
    # IT: il guard sullo scaler risolve il canonico da CWD: senza isolamento i test
    #     confronterebbero uno stato sintetico col `models/pipeline_state.pkl` REALE.
    # EN: the scaler guard resolves the canonical from CWD: without isolation the tests
    #     would compare a synthetic state against the REAL `models/pipeline_state.pkl`.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    with open(tmp_path / "models" / "pipeline_state.pkl", "wb") as f:
        pickle.dump(_state(), f)


# ───────────────────────── impronta / fingerprint ─────────────────────────
def test_fingerprint_is_stable_for_the_same_macro_block():
    assert macro_fingerprint(_npz(seed=1)) == macro_fingerprint(_npz(seed=1))


def test_fingerprint_catches_a_SINGLE_CELL_difference():
    # IT: il controllo positivo al livello più fine. `01b` rifitta il MacroNormalizer
    #     whole-df e riapplica a TUTTE le righe, quindi un refresh muove moltissime
    #     celle: se il guard non vedesse nemmeno una cella, non vedrebbe nulla.
    # EN: the finest-grained positive control. `01b` refits the MacroNormalizer whole-df
    #     and reapplies it to ALL rows, so a refresh moves very many cells: a guard
    #     blind to one cell would be blind to everything.
    a = _npz(seed=1)
    b = _npz(seed=1)
    b["X_macro_train"] = b["X_macro_train"].copy()
    b["X_macro_train"][7, 1] += np.float32(1e-3)
    assert macro_fingerprint(a) != macro_fingerprint(b)


def test_fingerprint_catches_a_COLUMN_REORDERING_at_equal_values():
    # IT: stesse colonne in ordine diverso sono un input diverso per un embedding
    #     posizionale, e i valori da soli non lo direbbero.
    # EN: the same columns in a different order are a different input to a positional
    #     embedding, and the values alone would not say so.
    a = _npz(seed=2, names=("m_a", "m_b", "m_c"))
    b = dict(a)
    b["macro_feature_names"] = np.array(["m_c", "m_b", "m_a"])
    assert macro_fingerprint(a)["names_md5"] != macro_fingerprint(b)["names_md5"]
    assert macro_fingerprint(a) != macro_fingerprint(b)


def test_fingerprint_catches_a_DTYPE_change_at_equal_values():
    a = _npz(seed=3, dtype=np.float32)
    b = dict(a)
    b["X_macro_train"] = a["X_macro_train"].astype(np.float64)
    b["X_macro_val"] = a["X_macro_val"].astype(np.float64)
    assert np.allclose(a["X_macro_train"], b["X_macro_train"])
    assert macro_fingerprint(a) != macro_fingerprint(b)


def test_fingerprint_reads_an_npz_from_disk_too(tmp_path):
    # IT: esercita il ramo `np.load` reale, non solo il mapping in memoria.
    # EN: exercises the real `np.load` branch, not just the in-memory mapping.
    p = tmp_path / "d.npz"
    np.savez(p, **_npz(seed=4))
    assert macro_fingerprint(p) == macro_fingerprint(_npz(seed=4))


def test_an_npz_without_macro_is_None_not_an_empty_match():
    assert macro_fingerprint({"X_train": np.zeros((3, 2))}) is None


# ───────────────── "non verificabile" non è "verificato" ─────────────────
def test_a_model_without_the_fingerprint_is_None_and_NEVER_True():
    # IT: condizione ② di M1. I modelli anteriori a M1 non hanno impronta: restano
    #     `None` per sempre. Se il guard restituisse True su di loro, l'unico caso in
    #     cui non può controllare nulla sarebbe anche quello in cui dichiara che va
    #     tutto bene — che è come è nato il problema che M1 chiude.
    # EN: M1's condition ②. Pre-M1 models carry no fingerprint: they stay `None`
    #     forever. Were the guard to return True on them, the one case where it can
    #     check nothing would also be the one where it declares all is well — which is
    #     how the problem M1 closes came about.
    out = check_model_dataset_macro(_state(), _npz(seed=5))
    assert out["matches"] is None
    assert out["model"] is None and out["dataset"] is not None


def test_a_None_npz_means_not_requested_never_a_match():
    out = check_model_dataset_macro(_state(_npz(seed=6)), None)
    assert out["matches"] is None
    assert out["dataset"] is None


def test_guard_matches_on_the_same_macro_vintage():
    d = _npz(seed=7)
    assert check_model_dataset_macro(_state(d), d)["matches"] is True


def test_the_source_of_the_fingerprint_is_carried_through():
    # IT: "declared" (backfill documentale) non deve essere indistinguibile da
    #     "measured": un backfill spacciato per misura è un'inferenza scritta come dato.
    # EN: "declared" (documentary backfill) must not be indistinguishable from
    #     "measured": a backfill passed off as a measurement is an inference written
    #     as data.
    d = _npz(seed=8)
    out = check_model_dataset_macro(_state(d, source="declared"), d)
    assert out["matches"] is True and out["model_fp_source"] == "declared"


# ─────────────── controllo positivo end-to-end / end-to-end ───────────────
def test_the_guard_FAILS_FAST_on_a_different_macro_vintage(tmp_path, monkeypatch):
    # IT: ① di M1, il livello che conta. Senza questo test un guard che ritornasse
    #     sempre `matches: true` supererebbe l'inerzia numerica in modo PERFETTO —
    #     cioè il gate sarebbe vuoto e la fiducia che produce, ingiustificata.
    # EN: M1's ①, the level that matters. Without this test a guard always returning
    #     `matches: true` would pass the numeric inertia check PERFECTLY — i.e. the
    #     gate would be empty and the confidence it produces unearned.
    _isolate(tmp_path, monkeypatch)
    st = _state(_npz(seed=10))
    with pytest.raises(RuntimeError, match="MACRO VINTAGE MISMATCH"):
        assert_model_dataset_scaler(st, model_dir="m", npz="d.npz",
                                    npz_arrays=_npz(seed=11))


def test_allow_macro_mismatch_downgrades_to_a_flagged_report(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    prov = assert_model_dataset_scaler(_state(_npz(seed=10)), model_dir="m", npz="d.npz",
                                       npz_arrays=_npz(seed=11),
                                       allow_macro_mismatch=True)
    assert prov["macro"]["matches"] is False
    assert prov["macro"]["allow_macro_mismatch"] is True


def test_the_scaler_escape_does_NOT_open_the_macro_axis(tmp_path, monkeypatch):
    # IT: i due assi di vintage sono INDIPENDENTI. Dichiarare incomparabile lo scaler
    #     non deve spegnere il controllo macro: sono due modi diversi di essere
    #     cross-vintage e accettarne uno non è accettare l'altro.
    # EN: the two vintage axes are INDEPENDENT. Declaring the scaler incomparable must
    #     not switch off the macro check: they are two different ways of being
    #     cross-vintage, and accepting one is not accepting the other.
    _isolate(tmp_path, monkeypatch)
    st = _state(_npz(seed=10))
    st.scaler = _Scaler([9.9, -1.0], [9.9, 9.9])          # scaler deliberatamente diverso
    with pytest.raises(RuntimeError, match="MACRO VINTAGE MISMATCH"):
        assert_model_dataset_scaler(st, model_dir="m", npz="d.npz",
                                    allow_mismatch=True,          # scaler dichiarato
                                    npz_arrays=_npz(seed=11))     # macro no → deve fermarsi


def test_the_macro_axis_is_checked_even_when_the_canonical_scaler_is_absent(tmp_path,
                                                                           monkeypatch):
    # IT: su un clone pulito il canonico manca e lo scaler è "non verificabile": il
    #     controllo macro deve girare lo stesso, altrimenti l'asse nuovo eredita il
    #     silenzio di quello vecchio.
    # EN: on a clean clone the canonical is absent and the scaler is "not verifiable":
    #     the macro check must still run, else the new axis inherits the old one's
    #     silence.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="MACRO VINTAGE MISMATCH"):
        assert_model_dataset_scaler(_state(_npz(seed=10)), model_dir="m", npz="d.npz",
                                    npz_arrays=_npz(seed=11))


def test_the_fingerprint_survives_save_and_load(tmp_path):
    # IT: l'impronta serve solo se ARRIVA al giudice. È scritta da `02_train` dentro il
    #     `PipelineState` e riletta da un pickle prodotto settimane prima: se il
    #     round-trip la perdesse, il guard direbbe "non verificabile" su OGNI modello
    #     — cioè fallirebbe in silenzio esattamente come il difetto che chiude.
    # EN: the fingerprint is only useful if it REACHES the judge. It is written by
    #     `02_train` into the `PipelineState` and read back from a pickle produced
    #     weeks earlier: were the round-trip to drop it, the guard would report "not
    #     verifiable" on EVERY model — failing silently just like the defect it closes.
    d = _npz(seed=20)
    st = _state(d)
    p = tmp_path / "pipeline_state.pkl"
    st.save(str(p))
    back = PipelineState.load(str(p))
    assert getattr(back, "macro_vintage_fp", None) == macro_fingerprint(d)
    assert getattr(back, "macro_vintage_fp_source", None) == "measured"
    assert check_model_dataset_macro(back, d)["matches"] is True


def test_a_legacy_pickle_without_the_new_slots_still_loads(tmp_path):
    # IT: i pkl esistenti sono stati scritti prima che i due slot esistessero. Devono
    #     caricare senza eccezione e degradare a `None`, non far cadere il giudice.
    # EN: existing pkl files predate the two slots. They must load without raising and
    #     degrade to `None`, not bring the judge down.
    st = _state()
    del st.macro_vintage_fp, st.macro_vintage_fp_source
    p = tmp_path / "legacy.pkl"
    with open(p, "wb") as f:
        pickle.dump(st, f)
    with open(p, "rb") as f:
        back = pickle.load(f)
    assert check_model_dataset_macro(back, _npz(seed=21))["matches"] is None


# ───────────────────── ③ il path LIVE non è raggiungibile ─────────────────────
LIVE_PATH_FILES = ("scripts/04b_vol_paper.py",
                   "scripts/04c_vol_paper_baselines.py",
                   "scripts/vol/vol_paper_replay.py",
                   "quantsys/model/vol_forecaster.py")


@pytest.mark.parametrize("rel", LIVE_PATH_FILES)
def test_the_LIVE_path_never_invokes_the_vintage_guard(rel):
    # IT: condizione ③ di M1, la più costosa da sbagliare. Un fail-fast raggiungibile
    #     dal live fermerebbe il forward test al bootstrap successivo (00:30 UTC)
    #     DENTRO campioni pre-registrati aperti — e lo farebbe per un disallineamento
    #     che sul live NON esiste: `VolForecaster` calcola le feature al volo
    #     iniettando scaler e colonne dal PipelineState DEL MODELLO, e non legge mai
    #     l'npz. Il costo di sbagliare qui è un campione bruciato, non un numero da
    #     riscrivere: per questo è un test e non solo un grep.
    # EN: M1's condition ③, the costliest to get wrong. A fail-fast reachable from the
    #     live path would stop the forward test at the next bootstrap (00:30 UTC)
    #     INSIDE open pre-registered samples — over a mismatch that does NOT exist
    #     live: `VolForecaster` builds features on the fly from the MODEL's own
    #     PipelineState and never reads the npz. The cost of being wrong here is a
    #     burnt sample, not a number to rewrite: hence a test, not just a grep.
    src = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    code = "\n".join(re.sub(r"#.*$", "", ln) for ln in src.splitlines())
    for sym in ("assert_model_dataset_scaler", "check_model_dataset_macro",
                "macro_fingerprint", "_assert_macro_vintage"):
        assert sym not in code, (
            f"{rel} invoca {sym}: il guard di vintage NON va sul path live — "
            f"fermerebbe 04b al bootstrap dentro un campione forward aperto")


# ─────────────── lo stato reale su disco / the real state on disk ───────────────
@pytest.mark.skipif(not (ROOT / "models" / "canonical_1h_vols" / "pipeline_state.pkl").exists(),
                    reason="artefatto canonico non versionato: assente su un clone pulito")
def test_the_canonical_artifact_declares_its_macro_vintage_or_declares_it_unknown():
    # IT: sentinella sull'artefatto di R1. Due esiti sono legittimi e vanno distinti:
    #     impronta assente (`None`) = addestrato prima di M1, provenienza macro NON
    #     verificabile — che è un fatto sulla sua provenienza, non un difetto; oppure
    #     impronta presente, e allora la FONTE deve essere dichiarata. Ciò che il test
    #     vieta è la terza possibilità: un'impronta senza fonte, cioè un backfill
    #     indistinguibile da una misura.
    # EN: sentinel on R1's artifact. Two outcomes are legitimate and must be told
    #     apart: fingerprint absent (`None`) = trained before M1, macro provenance NOT
    #     verifiable — a fact about its provenance, not a defect; or fingerprint
    #     present, in which case its SOURCE must be declared. What the test forbids is
    #     the third possibility: a fingerprint with no source, i.e. a backfill
    #     indistinguishable from a measurement.
    st = PipelineState.load(str(ROOT / "models" / "canonical_1h_vols" / "pipeline_state.pkl"))
    fp = getattr(st, "macro_vintage_fp", None)
    src = getattr(st, "macro_vintage_fp_source", None)
    if fp is None:
        assert src is None, "fonte dichiarata senza impronta: stato incoerente"
    else:
        assert src in ("measured", "declared"), \
            "impronta macro senza fonte dichiarata: un backfill non deve essere " \
            "indistinguibile da una misura"
        assert "X_macro_train" in fp.get("splits", {})
