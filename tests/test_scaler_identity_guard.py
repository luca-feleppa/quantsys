# IT: Guard di identità dello scaler modello↔dataset (2026-08-01).
#     Il difetto che chiude: `dev_vols_qlike.py` carica center/scale dal
#     `pipeline_state` del MODELLO e valuta sull'npz corrente, che è stato
#     costruito con un RobustScaler rifittato. Il modello di produzione
#     (`models/itransformer`, restore del PASS di giugno) ha
#     `target_scale = 1.4375922`; il canonico dell'npz corrente ha `1.4268685`.
#     Nulla falliva: usciva una QLIKE plausibile e ~5% peggiore (0.27470 contro
#     0.26143 di una coppia riaddestrata sullo stesso npz), cioè un artefatto di
#     scaler letto come skill — la stessa classe del −4.94% di A8.
#     ⚠ Perché `target_scale` da solo NON basta come impronta: due dataset possono
#     condividere la scala del target e differire sulle feature di INPUT. Il
#     disallineamento degli input è metà del problema e sarebbe invisibile.
# EN: Model↔dataset scaler identity guard (2026-08-01).
#     The defect it closes: `dev_vols_qlike.py` loads center/scale from the
#     MODEL's `pipeline_state` and evaluates on the current npz, which was built
#     with a refit RobustScaler. The production model (`models/itransformer`, the
#     June PASS restore) carries `target_scale = 1.4375922`; the current npz's
#     canonical state carries `1.4268685`. Nothing failed: out came a plausible
#     QLIKE ~5% worse (0.27470 against 0.26143 for a pair retrained on that same
#     npz) — a scaler artifact read as skill, the same class as A8's −4.94%.
#     ⚠ Why `target_scale` alone is NOT a sufficient fingerprint: two datasets can
#     share the target scale and differ on the INPUT features. Input misalignment
#     is half the problem and would be invisible.
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantsys.utils import (PipelineState, canonical_state_path,  # noqa: E402
                            check_model_dataset_scaler, scaler_fingerprint)


class _Scaler:
    """IT: scaler minimo con la sola superficie che l'impronta legge.
    EN: minimal scaler exposing only the surface the fingerprint reads."""

    def __init__(self, center, scale):
        self.center_ = np.asarray(center, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)


def _state(center, scale, cols=("a", "target_ret")):
    st = PipelineState()
    st.scale_cols = list(cols)
    st.scaler = _Scaler(center, scale)
    return st


def _save(tmp_path, name, st):
    p = tmp_path / name
    with open(p, "wb") as f:
        pickle.dump(st, f)
    return p


# ───────────────────────────── impronta ─────────────────────────────
def test_fingerprint_is_stable_for_identical_scalers():
    a = scaler_fingerprint(_state([1.0, -7.0], [2.0, 1.4]))
    b = scaler_fingerprint(_state([1.0, -7.0], [2.0, 1.4]))
    assert a == b


def test_fingerprint_catches_a_difference_in_the_INPUT_features_only():
    # IT: il caso che `target_scale` da solo mancherebbe: stessa scala del target,
    #     feature di input diverse. È metà del disallineamento reale — la rete
    #     riceve input normalizzati con uno scaler e ne è stata addestrata con un
    #     altro — e senza questo confronto passerebbe in silenzio.
    # EN: the case `target_scale` alone would miss: same target scale, different
    #     input features. It is half of the real misalignment and would pass
    #     silently without this comparison.
    a = _state([1.0, -7.0], [2.0, 1.4])
    b = _state([9.0, -7.0], [2.0, 1.4])
    assert a.target_scale == b.target_scale, "il presupposto del test non regge"
    assert scaler_fingerprint(a) != scaler_fingerprint(b)


# ───────────────────────────── guard ─────────────────────────────
def test_guard_flags_a_model_trained_under_another_scaler(tmp_path):
    model = _state([1.0, -7.0], [2.0, 1.4375921590454084])
    canon = _save(tmp_path, "pipeline_state.pkl",
                  _state([1.0, -7.0], [2.0, 1.4268685271051726]))
    out = check_model_dataset_scaler(model, canon)
    assert out["matches"] is False
    assert out["model"]["target_scale"] != out["canonical"]["target_scale"]


def test_guard_passes_when_the_scalers_are_the_same(tmp_path):
    st = _state([1.0, -7.0], [2.0, 1.4])
    canon = _save(tmp_path, "pipeline_state.pkl", _state([1.0, -7.0], [2.0, 1.4]))
    assert check_model_dataset_scaler(st, canon)["matches"] is True


def test_missing_canonical_is_None_and_never_True(tmp_path):
    # IT: «non verificabile» non è «verificato uguale». Un clone pulito non ha
    #     `models/`: se il guard restituisse True lì, l'unico posto in cui non può
    #     controllare nulla sarebbe anche l'unico in cui dichiara che va tutto bene.
    # EN: "not verifiable" is not "verified equal". A clean clone has no `models/`:
    #     were the guard to return True there, the one place it can check nothing
    #     would also be the one where it claims everything is fine.
    out = check_model_dataset_scaler(_state([1.0, -7.0], [2.0, 1.4]),
                                     tmp_path / "assente.pkl")
    assert out["matches"] is None
    assert out["canonical"] is None


# ─────────── risoluzione del canonico sotto sandbox (R1, 2026-08-04) ───────────
def test_canonical_is_resolved_OUTSIDE_the_sandbox(tmp_path, monkeypatch):
    # IT: il difetto che chiude. Con `QUANTSYS_MODELS_ROOT` attiva il canonico veniva
    #     cercato DENTRO la sandbox, dove non esiste mai: il guard restituiva
    #     `matches=None` e il giudice stampava un warning — cioè non controllava
    #     nulla proprio nella modalità in cui si giudicano i candidati. Un guard che
    #     tace esattamente dove serve è peggio di un guard assente, perché il report
    #     porta comunque un blocco `provenance` dall'aria rassicurante.
    # EN: the defect this closes. With `QUANTSYS_MODELS_ROOT` set, the canonical was
    #     looked up INSIDE the sandbox, where it never exists: the guard returned
    #     `matches=None` and the judge logged a warning — i.e. it checked nothing in
    #     exactly the mode where candidates are judged. A guard that goes quiet where
    #     it is needed is worse than no guard, since the report still carries a
    #     reassuring-looking `provenance` block.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    _save(tmp_path / "models", "pipeline_state.pkl", _state([1.0, -7.0], [2.0, 1.4]))
    (tmp_path / "models_x_sandbox").mkdir()
    monkeypatch.setenv("QUANTSYS_MODELS_ROOT", "models_x_sandbox")

    assert canonical_state_path() == Path("models") / "pipeline_state.pkl"
    out = check_model_dataset_scaler(_state([1.0, -7.0], [2.0, 1.4]))
    assert out["matches"] is True, "in sandbox il guard deve confrontare, non tacere"


def test_a_sandbox_local_canonical_wins_when_it_exists(tmp_path, monkeypatch):
    # IT: la precedenza non è cosmetica: un esperimento che costruisce il PROPRIO
    #     dataset dentro la sandbox (`QUANTSYS_DATASET_NPZ`) ha il proprio canonico,
    #     e confrontarlo con quello della root di default darebbe un mismatch falso.
    # EN: the precedence is not cosmetic: an experiment building its OWN dataset
    #     inside the sandbox (`QUANTSYS_DATASET_NPZ`) has its own canonical, and
    #     comparing it against the default-root one would yield a false mismatch.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    _save(tmp_path / "models", "pipeline_state.pkl", _state([1.0, -7.0], [2.0, 1.4]))
    (tmp_path / "models_y_sandbox").mkdir()
    _save(tmp_path / "models_y_sandbox", "pipeline_state.pkl",
          _state([1.0, -7.0], [2.0, 9.9]))
    monkeypatch.setenv("QUANTSYS_MODELS_ROOT", "models_y_sandbox")

    assert canonical_state_path() == Path("models_y_sandbox") / "pipeline_state.pkl"
    assert check_model_dataset_scaler(_state([1.0, -7.0], [2.0, 9.9]))["matches"] is True
    assert check_model_dataset_scaler(_state([1.0, -7.0], [2.0, 1.4]))["matches"] is False


# ─────────────── lo stato reale su disco: il caso che è successo ───────────────
@pytest.mark.skipif(not (ROOT / "models" / "itransformer" / "pipeline_state.pkl").exists()
                    or not (ROOT / "models" / "pipeline_state.pkl").exists(),
                    reason="modelli non versionati: assenti su un clone pulito")
def test_the_real_production_pair_is_detected_as_mismatched():
    # IT: non è un test sintetico. Verifica sui file veri che il guard veda il
    #     disallineamento che ha prodotto la confusione — se un giorno il modello
    #     venisse riaddestrato sull'npz corrente, questo test cade e va aggiornato
    #     con l'esito (che sarebbe una buona notizia, non una rottura).
    # EN: not a synthetic test. It checks on the real files that the guard sees the
    #     misalignment that produced the confusion — if the model is one day
    #     retrained on the current npz this test fails and must be updated with
    #     that outcome (good news, not a breakage).
    st = PipelineState.load(str(ROOT / "models" / "itransformer" / "pipeline_state.pkl"))
    out = check_model_dataset_scaler(st, ROOT / "models" / "pipeline_state.pkl")
    assert out["matches"] is False, \
        ("il modello di produzione ora combacia con l'npz canonico: il disallineamento "
         "storico è stato risolto — aggiorna questo test e la qualificazione in TEORIA.md §12.2")


@pytest.mark.skipif(not (ROOT / "models" / "canonical_1h_vols" / "pipeline_state.pkl").exists()
                    or not (ROOT / "models" / "pipeline_state.pkl").exists(),
                    reason="artefatto canonico non versionato: assente su un clone pulito")
def test_the_canonical_artifact_is_and_stays_aligned_with_the_npz():
    # IT: sentinella sull'artefatto di R1 (2026-08-04). La coppia canonica esiste per
    #     dare un numeratore VERIFICABILE alla banda pubblicata: se il suo scaler
    #     smette di combaciare col canonico, l'artefatto non è più confrontabile con
    #     l'npz corrente e la banda torna a essere un'affermazione sul protocollo.
    #     ⚠ La rottura attesa non è "il modello è cambiato" — i checkpoint sono
    #     congelati — ma "l'npz è stato ricostruito": in quel caso la risposta è una
    #     NUOVA coppia canonica, non `--allow-scaler-mismatch`.
    # EN: sentinel on R1's artifact (2026-08-04). The canonical pair exists to give
    #     the published band a VERIFIABLE numerator: if its scaler stops matching the
    #     canonical one, the artifact is no longer comparable against the current npz
    #     and the band reverts to being a statement about the protocol.
    #     ⚠ The expected breakage is not "the model changed" — checkpoints are frozen
    #     — but "the npz was rebuilt": the answer then is a NEW canonical pair, not
    #     `--allow-scaler-mismatch`.
    st = PipelineState.load(str(ROOT / "models" / "canonical_1h_vols" / "pipeline_state.pkl"))
    out = check_model_dataset_scaler(st, ROOT / "models" / "pipeline_state.pkl")
    assert out["matches"] is True, \
        ("l'artefatto canonico non combacia più con l'npz: probabile rebuild del "
         "dataset — serve una nuova coppia canonica (pre-registrata), non una via di fuga")
    assert out["model"]["target_scale"] == 1.4268685271051726
