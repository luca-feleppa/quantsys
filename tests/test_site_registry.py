"""
IT: Test del registro degli esperimenti e del generatore del sito.
    Tre proprieta' da dimostrare, non assumere.
    (1) DISCIPLINA ONE-SHOT: un gate ancora aperto che mostri un numero — o anche
        solo un verdetto — brucia la pre-registrazione, ed e' l'unica violazione
        irreversibile possibile su questa pagina. Il controllo deve essere codice
        e deve essere visto fallire.
    (2) NIENTE DUPLICAZIONE DEI NUMERI: dove un numero esiste in un report su
        disco, il registro dichiara file e chiave e il valore si legge a ogni
        build. Il test verifica che il valore reso sia davvero quello del report,
        cioe' che la divergenza sia impossibile e non solo improbabile.
    (3) NIENTE FONTE PARALLELA: ogni scheda punta a una sezione ESISTENTE di
        `TEORIA.md`, che resta la fonte autorevole. Una scheda che non ci punta
        sarebbe un fatto che vive solo nel registro — cioe' una seconda verita'.
EN: Tests for the experiment registry and the site generator.
    Three properties to demonstrate, not assume: the one-shot discipline guard
    (and it must be seen failing); no duplication of numbers (a derived value is
    read from the report at build time, so divergence is impossible); and no
    parallel source of truth (every card points at an existing `TEORIA.md`
    section, which stays authoritative).
"""
import importlib.util
import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bs():
    spec = importlib.util.spec_from_file_location(
        "build_site", ROOT / "scripts" / "site" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def reg(bs):
    return yaml.safe_load((ROOT / "docs" / "experiments.yaml").read_text(encoding="utf-8"))


# ─────────────────────────── (1) disciplina one-shot ───────────────────────────
def test_open_gate_with_numbers_fails_the_build(bs):
    with pytest.raises(bs.OneShotViolation, match="APERTO"):
        bs.assert_one_shot_discipline(
            [{"id": "x", "status": "aperto", "verdict": None,
              "numbers": {"literal": {"q": 0.26}, "as_of": "2026-01-01"}}])


def test_open_gate_with_a_verdict_fails_too(bs):
    # IT: un VERDETTO e' esso stesso decisionale: "PASS" su un gate aperto e'
    #     informativo quanto il numero che lo produce.
    # EN: a VERDICT is itself decisional.
    with pytest.raises(bs.OneShotViolation, match="verdict"):
        bs.assert_one_shot_discipline(
            [{"id": "x", "status": "aperto", "verdict": "PASS", "numbers": None}])


def test_closed_gate_may_carry_numbers(bs):
    bs.assert_one_shot_discipline(
        [{"id": "y", "status": "chiuso", "verdict": "PASS",
          "numbers": {"literal": {"q": 0.26}, "as_of": "2026-01-01"}}])


def test_real_registry_is_disciplined(bs, reg):
    # IT: il guard sui dati REALI, non su fixture: se qualcuno aggiunge un numero
    #     a un gate aperto, la suite cade prima della build.
    # EN: the guard on the REAL data, not fixtures.
    bs.assert_one_shot_discipline(reg["experiments"])
    assert any(e["status"] in bs.OPEN_STATES for e in reg["experiments"]), \
        "nessun gate aperto nel registro: il test non verificherebbe nulla"


def test_open_cards_render_without_a_single_digit_in_the_numbers_area(bs, reg):
    # IT: verifica sull'OUTPUT: un guard sui dati non basta se poi il render
    #     stampa una cifra da un'altra parte della scheda.
    # EN: a check on the OUTPUT: a data guard is not enough if the renderer
    #     prints a digit elsewhere in the card.
    lines = reg["lines"]
    opened = [e for e in reg["experiments"] if e["status"] in bs.OPEN_STATES]
    assert opened
    for e in opened:
        card = bs.render_card(e, {}, lines)
        area = re.search(r'<p class="none">(.*?)</p>', card, re.S)
        assert area, f"[{e['id']}] scheda aperta senza il blocco che spiega l'assenza"
        assert not re.search(r"\d", area.group(1)), f"[{e['id']}] cifra in una scheda aperta"


# ─────────────────── (2) i numeri derivati non possono divergere ───────────────────
def test_derived_numbers_are_read_from_the_report_not_copied(bs, reg):
    # IT: IL test anti-duplicazione. Per ogni scheda con `numbers.source`, il
    #     valore reso deve coincidere con quello nel JSON: se qualcuno copiasse
    #     il numero nel registro invece di puntarlo, questo test non lo vedrebbe
    #     — ma lo schema lo vieta (source e literal sono mutuamente esclusivi) e
    #     `test_schema_rejects_both_source_and_literal` lo verifica.
    # EN: THE anti-duplication test. For every card with `numbers.source`, the
    #     rendered value must equal the one in the JSON.
    checked = 0
    for e in reg["experiments"]:
        src = e.get("numbers", {}).get("source") if e.get("numbers") else None
        if not src:
            continue
        raw = json.loads((ROOT / src).read_text(encoding="utf-8"))
        for label, path in e["numbers"]["keys"].items():
            got = bs.dig(raw, path)
            rendered = dict((k, v) for k, v, _ in bs.resolve_numbers(e, {src: raw}))
            assert rendered[label] == bs.fmt(got)
            checked += 1
    assert checked >= 5, f"solo {checked} numeri derivati verificati: coverage troppo bassa"


def test_dig_supports_keys_containing_a_dot(bs):
    # IT: alcune chiavi dei report contengono un punto nel nome, quindi la
    #     notazione puntata non basta a esprimerle: serve la forma a lista.
    #     (Trovato alla prima build, non in fase di disegno.)
    # EN: some report keys contain a dot, so dotted notation cannot express them.
    obj = {"gates": {"g3_real_hit_rate_gt_0.5": True}}
    assert bs.dig(obj, ["gates", "g3_real_hit_rate_gt_0.5"]) is True
    with pytest.raises(KeyError):
        bs.dig(obj, "gates.g3_real_hit_rate_gt_0.5")


def test_dig_error_says_where_it_stopped(bs):
    # IT: un errore di percorso deve essere diagnostico, non solo negativo:
    #     dire dove si e' fermato e cosa c'era, altrimenti costa una sessione.
    # EN: a path error must be diagnostic, not merely negative.
    with pytest.raises(KeyError) as ei:
        bs.dig({"a": {"b": 1}}, "a.zzz")
    msg = str(ei.value)
    assert "zzz" in msg and "'a'" in msg and "b" in msg


# ─────────────────── (3) nessuna fonte parallela a TEORIA.md ───────────────────
def test_every_card_points_at_an_existing_teoria_section(bs, reg):
    bs.assert_teoria_pointers(reg["experiments"])


def test_a_dangling_teoria_pointer_fails(bs):
    with pytest.raises(bs.RegistrySchemaError, match="non esiste"):
        bs.assert_teoria_pointers([{"id": "x", "teoria": "99.99"}])


# ───────────────────────────── schema del registro ─────────────────────────────
def _minimal(**kw):
    base = {"id": "x", "name": "n", "line": "vol-1h", "status": "chiuso",
            "question": "q", "teoria": "12.2"}
    base.update(kw)
    return {"schema": 1, "lines": {"vol-1h": "v"}, "experiments": [base]}


def test_schema_rejects_both_source_and_literal(bs):
    # IT: due origini per lo stesso numero SONO una duplicazione — e' il difetto
    #     che il registro esiste per evitare, quindi lo schema lo vieta.
    # EN: two origins for the same number ARE a duplication.
    with pytest.raises(bs.RegistrySchemaError, match="sia .source. sia .literal."):
        bs.validate_registry(_minimal(numbers={"source": "a.json", "keys": {},
                                               "literal": {"q": 1}}))


def test_schema_rejects_a_literal_without_a_date(bs):
    # IT: un numero scritto a mano senza la data in cui era vero non e'
    #     verificabile da nessuno, nemmeno da chi l'ha scritto.
    # EN: a hand-written number with no as-of date is unverifiable.
    with pytest.raises(bs.RegistrySchemaError, match="as_of"):
        bs.validate_registry(_minimal(numbers={"literal": {"q": 1}}))


def test_schema_rejects_duplicate_ids(bs):
    r = _minimal()
    r["experiments"] = r["experiments"] * 2
    with pytest.raises(bs.RegistrySchemaError, match="duplicato"):
        bs.validate_registry(r)


def test_schema_requires_a_teoria_pointer(bs):
    r = _minimal()
    del r["experiments"][0]["teoria"]
    with pytest.raises(bs.RegistrySchemaError, match="teoria"):
        bs.validate_registry(r)


def test_real_registry_validates(bs, reg):
    bs.validate_registry(reg)


# ──────────────────────────── build e invarianti ────────────────────────────
def test_missing_report_stops_the_build(bs):
    with pytest.raises(FileNotFoundError, match="assente"):
        bs.load_reports({"results/vols/questo_non_esiste.json"})


def test_required_reports_are_derived_from_the_registry(bs, reg):
    # IT: la lista dei report obbligatori non e' scritta a parte: si deriva dalle
    #     schede, cosi' aggiungerne una con una sorgente nuova rende quel report
    #     obbligatorio senza toccare il generatore.
    # EN: the required-reports list is derived from the cards, not maintained
    #     separately.
    need = bs.required_reports(reg["experiments"])
    assert need, "nessun report derivato: il meccanismo non starebbe funzionando"
    for rel in need:
        assert (ROOT / rel).exists(), f"report dichiarato ma assente: {rel}"


def test_build_stamp_is_present_and_well_formed(bs):
    s = bs.build_stamp()
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC$", s["utc"])
    assert s["commit"]


def test_no_assistant_attribution_in_tracked_site_files():
    # IT: i termini vietati sono composti da frammenti a runtime: un test che
    #     cerca una stringa e la contiene alla lettera violerebbe l'invariante
    #     che esiste per far rispettare. (Scoperto al primo run.)
    # EN: the forbidden terms are assembled at runtime: a test that searches for
    #     a string and contains it verbatim would violate the very invariant it
    #     enforces.
    pat = re.compile("|".join(["cl" + "aude", "anthr" + "opic"]), re.I)
    for rel in ["scripts/site/build_site.py", "docs/experiments.yaml", "docs/index.html"]:
        p = ROOT / rel
        if p.exists():
            assert not pat.search(p.read_text(encoding="utf-8")), f"attribuzione in {rel}"
