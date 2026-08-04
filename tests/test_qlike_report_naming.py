# IT: Naming del report del giudice QLIKE — contratto ANTI-CLOBBER (2026-08-04).
#     Il difetto che chiude: il nome portava solo il suffisso della ROOT sandbox
#     (`QUANTSYS_MODELS_ROOT`), non quello dell'ARCH. Giudicare un artefatto che vive
#     come cartella-arch dentro `models/` — la coppia canonica di R1,
#     `models/canonical_1h_vols` — scriveva quindi sul nome NUDO
#     `qlike_report_1h_val.json`, cioè **sopra il report storico di produzione**,
#     quello che `TEORIA.md` §12.2 cita come QLIKE del checkpoint di giugno (0.27470).
#     ⚠ Fallimento SILENZIOSO per costruzione: il giudice esce 0, stampa un PASS
#     corretto, e l'unica traccia della distruzione è un file sovrascritto con numeri
#     plausibili. È la stessa famiglia dei numeri orfani che il blocco `provenance`
#     esiste per impedire — lì il numero perdeva il suo modello, qui il modello
#     sovrascriveva il numero di un altro.
# EN: QLIKE judge report naming — ANTI-CLOBBER contract (2026-08-04).
#     The defect it closes: the name carried only the sandbox ROOT suffix
#     (`QUANTSYS_MODELS_ROOT`), not the ARCH one. Judging an artifact living as an
#     arch-directory inside `models/` — R1's canonical pair,
#     `models/canonical_1h_vols` — therefore wrote to the BARE name
#     `qlike_report_1h_val.json`, i.e. **over the historical production report**, the
#     one `TEORIA.md` §12.2 cites as the June checkpoint's QLIKE (0.27470).
#     ⚠ SILENT failure by construction: the judge exits 0, prints a correct PASS, and
#     the only trace of the destruction is a file overwritten with plausible numbers.
#     Same family as the orphan numbers the `provenance` block exists to prevent —
#     there a number lost its model, here a model overwrote another's number.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "vol"))

from dev_vols_qlike import report_filename  # noqa: E402


def test_production_path_keeps_the_bare_historical_name():
    # IT: invariante di non-rinomina: i report storici citati nella documentazione
    #     devono restare raggiungibili al loro nome. Se questo test cade, ogni
    #     riferimento a `qlike_report_1h_{val,test}.json` nei doc diventa morto.
    # EN: no-rename invariant: historical reports cited in the documentation must
    #     stay reachable under their name. If this test falls, every reference to
    #     `qlike_report_1h_{val,test}.json` in the docs goes dead.
    assert report_filename("1h", "val") == "qlike_report_1h_val.json"
    assert report_filename("1h", "test") == "qlike_report_1h_test.json"


def test_the_canonical_artifact_never_writes_over_production():
    # IT: il caso reale che ha prodotto la patch.
    # EN: the real case that produced the patch.
    assert report_filename("1h", "val", "canonical_1h_vols") \
        == "qlike_report_1h_val_canonical_1h_vols.json"
    assert report_filename("1h", "val", "canonical_1h_vols") != report_filename("1h", "val")


def test_root_and_arch_suffixes_are_independent():
    # IT: i due assi di isolamento non si sostituiscono a vicenda — una sandbox può
    #     contenere una dir-modello non di default, e il nome deve distinguere
    #     entrambe le dimensioni.
    # EN: the two isolation axes do not substitute for each other — a sandbox may
    #     hold a non-default model dir, and the name must separate both dimensions.
    assert report_filename("1h", "val", "itransformer", "models_r1_sandbox") \
        == "qlike_report_1h_val_models_r1_sandbox.json"
    assert report_filename("1h", "val", "itransformer_a10_sparsity", "models_a10_sparsity") \
        == "qlike_report_1h_val_itransformer_a10_sparsity_models_a10_sparsity.json"


def test_split_and_interval_stay_separated():
    # IT: val e test non devono MAI collidere: il test split è one-shot e un clobber
    #     lo renderebbe irrecuperabile senza ri-consumarlo.
    # EN: val and test must NEVER collide: the test split is one-shot and a clobber
    #     would make it unrecoverable without consuming it again.
    names = {report_filename(i, s) for i in ("1h", "30min") for s in ("val", "test")}
    assert len(names) == 4
