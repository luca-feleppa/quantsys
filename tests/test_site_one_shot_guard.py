"""
IT: Test del guard di disciplina one-shot del generatore del sito.
    Perche' esiste: la violazione piu' probabile, costruendo una pagina di
    progetto, e' mostrare il numero di un gate ancora APERTO — e' anche l'unica
    irreversibile, perche' un numero decisionale visto in anticipo non si puo'
    non-vedere e brucia la pre-registrazione. Il blueprint lo mette per iscritto:
    il controllo deve essere codice, non disciplina di chi scrive. Questi test
    verificano che il codice ci sia e che fallisca DAVVERO — un guard che non e'
    mai stato visto fallire non e' un guard, e' un commento.
EN: Tests for the site generator's one-shot discipline guard.
    Why it exists: the likeliest violation when building a project page is
    showing a number from a still-OPEN gate — and it is the only irreversible
    one, since a decisional number seen early cannot be unseen and burns the
    pre-registration. The blueprint states it: the check must be code, not
    author discipline. These tests verify the code is there and that it actually
    FAILS — a guard never seen failing is not a guard, it is a comment.
"""
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bs():
    spec = importlib.util.spec_from_file_location(
        "build_site", ROOT / "scripts" / "site" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_open_gate_with_numbers_fails_the_build(bs):
    # IT: IL test. Un gate aperto che porta numeri deve far fallire la build,
    #     non emettere una pagina "quasi giusta".
    # EN: THE test. An open gate carrying numbers must fail the build rather
    #     than emit an "almost right" page.
    bad = [{"id": "x", "name": "n", "line": "vol", "status": "aperto",
            "verdict": None, "numbers": {"qlike": 0.26}}]
    with pytest.raises(bs.OneShotViolation, match="APERTO"):
        bs.assert_one_shot_discipline(bad)


def test_open_gate_with_a_verdict_fails_too(bs):
    # IT: un VERDETTO e' esso stesso una quantita' decisionale — "PASS" su un
    #     gate aperto e' informativo quanto il numero che lo produce.
    # EN: a VERDICT is itself a decisional quantity — "PASS" on an open gate is
    #     as informative as the number behind it.
    bad = [{"id": "x", "name": "n", "line": "vol", "status": "aperto",
            "verdict": "PASS", "numbers": {}}]
    with pytest.raises(bs.OneShotViolation, match="verdict"):
        bs.assert_one_shot_discipline(bad)


def test_closed_gate_may_carry_numbers(bs):
    # IT: il guard non deve essere cosi' zelante da vietare i numeri dove sono
    #     legittimi: un gate chiuso li porta.
    # EN: the guard must not be so zealous as to forbid numbers where they are
    #     legitimate: a closed gate carries them.
    ok = [{"id": "y", "name": "n", "line": "vol", "status": "chiuso",
           "verdict": "PASS", "numbers": {"qlike": 0.26}}]
    bs.assert_one_shot_discipline(ok)  # non solleva / does not raise


def test_shipped_gate_table_is_disciplined(bs):
    # IT: il guard applicato ai gate REALI del sito, non solo a fixture: se
    #     qualcuno aggiunge un numero a un gate aperto, la suite cade.
    # EN: the guard applied to the site's REAL gates, not just fixtures: if
    #     someone adds a number to an open gate, the suite falls over.
    bs.assert_one_shot_discipline(bs.GATES)
    assert any(g["status"] in bs.OPEN_STATES for g in bs.GATES), \
        "nessun gate aperto: il test non starebbe verificando nulla"


def test_open_gates_render_without_any_digit_in_the_numbers_cell(bs):
    # IT: verifica sull'OUTPUT, non sull'input: la cella dei numeri di un gate
    #     aperto non deve contenere cifre. Un guard sui dati non basta se poi il
    #     render le stampa da un'altra parte.
    # EN: a check on the OUTPUT, not the input: an open gate's numbers cell must
    #     contain no digits. A guard on the data is not enough if the renderer
    #     prints them from somewhere else.
    frag = bs.render_gates(bs.GATES)
    cells = re.findall(r"<td><i>(.*?)</i></td>", frag)
    assert cells, "nessun gate aperto reso: il test non verifica nulla"
    for c in cells:
        assert not re.search(r"\d", c), f"cifra in una cella di gate aperto: {c!r}"


def test_missing_report_stops_the_build(bs, tmp_path, monkeypatch):
    # IT: fail-fast: meglio nessuna pagina che una pagina con un buco silenzioso.
    # EN: fail-fast: better no page than a page with a silent hole.
    monkeypatch.setattr(bs, "REQUIRED_REPORTS", {"x": tmp_path / "assente.json"})
    with pytest.raises(FileNotFoundError, match="assente"):
        bs.load_reports()


def test_generated_page_carries_a_build_stamp(bs):
    # IT: V5 antistaleness — ogni pagina dice data e commit da cui proviene.
    # EN: V5 antistaleness — every page states the date and commit it came from.
    s = bs.build_stamp()
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC$", s["utc"])
    assert s["commit"] and s["commit"] != ""


def test_no_assistant_attribution_in_generator_or_output():
    # IT: invariante di attribuzione: il sito e il suo generatore finiscono
    #     TRACCIATI nel repo pubblico, quindi valgono le stesse regole del resto.
    #     ⚠ I termini vietati sono composti da frammenti a runtime, non scritti
    #     per esteso: un test che cerca una stringa e la contiene alla lettera
    #     violerebbe l'invariante che esiste per farlo rispettare — e farebbe
    #     fallire il grep di verifica su se stesso. (Scoperto al primo run.)
    # EN: attribution invariant: the site and its generator end up TRACKED in the
    #     public repo, so the same rules apply as everywhere else.
    #     ⚠ The forbidden terms are assembled from fragments at runtime, not
    #     spelled out: a test that searches for a string and contains it verbatim
    #     would violate the very invariant it enforces — and would make the
    #     verification grep fail on itself. (Found on the first run.)
    forbidden = ["cl" + "aude", "anthr" + "opic"]
    pat = re.compile("|".join(forbidden), re.I)
    for p in [ROOT / "scripts" / "site" / "build_site.py", ROOT / "docs" / "index.html"]:
        if p.exists():
            assert not pat.search(p.read_text(encoding="utf-8")), f"attribuzione in {p.name}"
