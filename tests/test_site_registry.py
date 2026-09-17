"""
Tests for the experiment registry and the project site generator.
Three properties to demonstrate, not assume:
(1) ONE-SHOT DISCIPLINE: a still-open gate showing a number, or even just a verdict,
    burns the pre-registration, the only irreversible violation possible on this page.
    The check must be code and must be seen failing.
(2) NO DUPLICATION OF NUMBERS: where a number exists in a report on disk, the registry
    declares file and key and the value is read at every build. The test checks the
    rendered value really is the report's, i.e. divergence is impossible, not unlikely.
(3) NO PARALLEL SOURCE OF TRUTH: every card points at an EXISTING `THEORY.md` section,
    which stays authoritative.
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


OPEN_CARD = {"id": "synthetic-open", "name": {"it": "Gate aperto", "en": "Open gate"},
             "line": "vol-1h", "scope": "segnale", "status": "aperto", "verdict": None,
             "dates": {"prereg": "2026-01-01"},
             "question": {"it": "domanda", "en": "question"},
             "prior": {"it": "prior", "en": "prior"},
             "threshold": {"it": "soglia", "en": "threshold"},
             "counter": {"it": "n minimo", "en": "minimum n"},
             "split": "forward", "teoria": "12.2"}


def _open_cards(bs, reg):
    # real open gates plus a synthetic one, so the guard never tests an empty set.
    return [e for e in reg["experiments"] if e["status"] in bs.OPEN_STATES] + [OPEN_CARD]


@pytest.fixture(scope="module")
def reg(bs):
    return yaml.safe_load((ROOT / "docs" / "experiments.yaml").read_text(encoding="utf-8"))


# ─────────────────────────── (1) one-shot discipline ───────────────────────────
def test_open_gate_with_numbers_fails_the_build(bs):
    with pytest.raises(bs.OneShotViolation, match="APERTO"):
        bs.assert_one_shot_discipline(
            [{"id": "x", "status": "aperto", "verdict": None,
              "numbers": {"literal": {"q": 0.26}, "as_of": "2026-01-01"}}])


def test_open_gate_with_a_verdict_fails_too(bs):
    # a VERDICT is itself decisional.
    with pytest.raises(bs.OneShotViolation, match="verdict"):
        bs.assert_one_shot_discipline(
            [{"id": "x", "status": "aperto", "verdict": "PASS", "numbers": None}])


def test_closed_gate_may_carry_numbers(bs):
    bs.assert_one_shot_discipline(
        [{"id": "y", "status": "chiuso", "verdict": "PASS",
          "numbers": {"literal": {"q": 0.26}, "as_of": "2026-01-01"}}])


def test_real_registry_is_disciplined(bs, reg):
    # the guard on the REAL data, not fixtures.
    bs.assert_one_shot_discipline(reg["experiments"])


def test_open_cards_render_without_a_single_digit_in_the_numbers_area(bs, reg):
    # a check on the OUTPUT: a data guard is not enough if the renderer
    # prints a digit elsewhere in the card.
    lines = reg["lines"]
    for e in _open_cards(bs, reg):
        card = bs.render_card(e, {}, lines)
        area = re.search(r'<p class="none">(.*?)</p>', card, re.S)
        assert area, f"[{e['id']}] scheda aperta senza il blocco che spiega l'assenza"
        assert not re.search(r"\d", area.group(1)), f"[{e['id']}] cifra in una scheda aperta"


# ─────────────────── (2) derived numbers cannot diverge ───────────────────
def test_derived_numbers_are_read_from_the_report_not_copied(bs, reg):
    # THE anti-duplication test. For every card with `numbers.source`, the
    # rendered value must equal the one in the JSON.
    checked = 0
    for e in reg["experiments"]:
        src = e.get("numbers", {}).get("source") if e.get("numbers") else None
        if not src:
            continue
        raw = json.loads((ROOT / src).read_text(encoding="utf-8"))
        for lang in bs.LANGS:
            rendered = dict((k, v) for k, v, _ in bs.resolve_numbers(e, {src: raw}, lang))
            for item in e["numbers"]["keys"]:
                assert rendered[item["label"][lang]] == bs.fmt(bs.dig(raw, item["path"]), lang)
                checked += 1
    assert checked >= 5, f"solo {checked} numeri derivati verificati: coverage troppo bassa"


def test_dig_supports_keys_containing_a_dot(bs):
    # some report keys contain a dot, so dotted notation cannot express them.
    obj = {"gates": {"g3_real_hit_rate_gt_0.5": True}}
    assert bs.dig(obj, ["gates", "g3_real_hit_rate_gt_0.5"]) is True
    with pytest.raises(KeyError):
        bs.dig(obj, "gates.g3_real_hit_rate_gt_0.5")


def test_dig_error_says_where_it_stopped(bs):
    # a path error must be diagnostic, not merely negative.
    with pytest.raises(KeyError) as ei:
        bs.dig({"a": {"b": 1}}, "a.zzz")
    msg = str(ei.value)
    assert "zzz" in msg and "'a'" in msg and "b" in msg


# ─────────────────── (3) no parallel source to THEORY.md ───────────────────
def test_every_card_points_at_an_existing_teoria_section(bs, reg):
    bs.assert_teoria_pointers(reg["experiments"])


def test_a_dangling_teoria_pointer_fails(bs):
    with pytest.raises(bs.RegistrySchemaError, match="non esiste"):
        bs.assert_teoria_pointers([{"id": "x", "teoria": "99.99"}])


# ───────────────────────────── registry schema ─────────────────────────────
def _minimal(**kw):
    base = {"id": "x", "name": "n", "line": "vol-1h", "status": "chiuso",
            "question": "q", "teoria": "12.2", "scope": "segnale"}
    base.update(kw)
    return {"schema": 2, "lines": {"vol-1h": "v"},
            "scopes": {"segnale": "s", "metodo": "m"}, "experiments": [base]}


_LIT = {"label": {"it": "q", "en": "q"}, "value": "1"}


def test_schema_requires_a_scope(bs):
    # without `scope` the page would sum PASSes answering different
    # questions and announce more predictive results than exist.
    r = _minimal()
    del r["experiments"][0]["scope"]
    with pytest.raises(bs.RegistrySchemaError, match="scope"):
        bs.validate_registry(r)


def test_schema_rejects_an_unknown_scope(bs):
    # CLOSED vocabulary: a free-form scope would reintroduce the arbitrary
    # category the field exists to remove.
    with pytest.raises(bs.RegistrySchemaError, match="scope sconosciuto"):
        bs.validate_registry(_minimal(scope="qualcosa"))


def test_schema_rejects_both_source_and_literal(bs):
    # two origins for the same number ARE a duplication.
    with pytest.raises(bs.RegistrySchemaError, match="sia .source. sia .literal."):
        bs.validate_registry(_minimal(numbers={"source": "a.json", "keys": [],
                                               "literal": [_LIT]}))


def test_schema_rejects_a_literal_without_a_date(bs):
    # a hand-written number with no as-of date is unverifiable.
    with pytest.raises(bs.RegistrySchemaError, match="as_of"):
        bs.validate_registry(_minimal(numbers={"literal": [_LIT]}))


def test_schema_rejects_a_monolingual_number_label(bs):
    # a label in one language only puts that language's text on the other page (the
    # v1 schema showed Italian labels on the English page).
    bad = {"label": {"it": "solo italiano"}, "value": "1"}
    with pytest.raises(bs.RegistrySchemaError, match="it/en"):
        bs.validate_registry(_minimal(numbers={"literal": [bad], "as_of": "2026-01-01"}))


def test_schema_accepts_a_bilingual_literal(bs):
    bs.validate_registry(_minimal(numbers={"literal": [_LIT], "as_of": "2026-01-01"}))


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


# ──────────────────────────── build and invariants ────────────────────────────
def test_missing_report_stops_the_build(bs):
    with pytest.raises(FileNotFoundError, match="assente"):
        bs.load_reports({"results/vols/questo_non_esiste.json"})


def test_required_reports_are_derived_from_the_registry(bs, reg):
    # the required-reports list is derived from the cards, not maintained
    # separately.
    need = bs.required_reports(reg["experiments"])
    assert need, "nessun report derivato: il meccanismo non starebbe funzionando"
    for rel in need:
        assert (ROOT / rel).exists(), f"report dichiarato ma assente: {rel}"


def test_build_stamp_is_present_and_well_formed(bs):
    s = bs.build_stamp()
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC$", s["utc"])
    assert s["commit"]


# ─────────────────── model facts: derived, not typed by hand ───────────────────
def test_model_facts_are_derived_and_internally_coherent(bs):
    # Section 1's counts come from the production model's `config.json`. The
    # test checks they add up: a wrong sum is what a competent reader checks
    # in three seconds.
    f = bs.load_model_facts()
    assert f["n_dynamic"] + f["n_structural"] == f["n_features"]
    assert f["n_structural"] > 0 and f["n_dynamic"] > 0
    assert f["window"] > 0 and f["horizon"] > 0


def test_dropped_features_come_from_the_code_constant(bs):
    # the list of features dropped for live use is NOT transcribed: it is the
    # constant the code actually uses, so it cannot diverge from behaviour.
    from quantsys.features import LIVE_DROP_FEATURES
    assert bs.load_model_facts()["dropped_live"] == sorted(LIVE_DROP_FEATURES)


def test_missing_model_config_stops_the_build_with_a_useful_message(bs, tmp_path, monkeypatch):
    # `models/` is not versioned, so a clean clone cannot build the site. The
    # message must say so rather than emitting invented counts.
    monkeypatch.setattr(bs, "MODEL_CONFIG", tmp_path / "config.json")
    with pytest.raises(FileNotFoundError, match="non e. versionata|assente"):
        bs.load_model_facts()


def test_section1_interpolates_the_facts_it_was_given(bs):
    # checks the prose uses the facts rather than constants baked into the
    # text: with fake numbers the page must change accordingly.
    fake = {"n_features": 7, "n_dynamic": 3, "n_structural": 4, "n_macro": 5,
            "window": 11, "loss_type": "quantile", "dropped_live": ["a_b", "c_d"],
            "interval": "1h", "horizon": 13}
    out = bs.render_section1(fake)
    for token in ["<b>7 feature</b>", "<b>3 dinamiche</b>", "<b>4 strutturali</b>",
                  "5 serie", "<b>11 barre</b>", "<code>a_b</code>", "13 ore".replace(" ", " ")]:
        assert token in out, f"la Sezione 1 non interpola {token!r}"


# ─────────────── method section: tallies follow the registry ───────────────
def test_method_facts_match_the_registry(bs, reg):
    # method tallies must come from the registry, not a third copy.
    # Recomputed here independently: if a card is added and the prose stays
    # put, this test fails.
    f = bs.method_facts(reg)
    exps = reg["experiments"]
    closed = [e for e in exps if e["status"] not in bs.OPEN_STATES]
    assert f["n_tot"] == len(exps)
    assert f["n_closed"] == len(closed)
    assert f["n_open"] == len(exps) - len(closed)
    assert f["n_tot"] == f["n_closed"] + f["n_open"]
    assert sum(f["tally"].values()) == f["n_closed"], \
        "ogni scheda chiusa deve contribuire a esattamente un esito"
    # lines are the ones actually USED, not the ones declared.
    assert f["n_lines"] == len({e["line"] for e in exps})
    assert f["n_lines"] <= len(reg["lines"])


def test_method_section_interpolates_a_fake_registry(bs):
    # same proof as Section 1 — with a fake registry the prose must change.
    fake = {"lines": {"l1": "linea uno", "l2": "linea due"}, "experiments": [
        {"id": "a", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "segnale"},
        {"id": "b", "line": "l1", "status": "chiuso", "verdict": "FAIL", "scope": "leva"},
        {"id": "c", "line": "l2", "status": "chiuso", "verdict": "FAIL", "scope": "leva"},
        {"id": "d", "line": "l2", "status": "aperto", "scope": "segnale"},
    ]}
    out = bs.render_method(fake)
    assert "Su 4 esperimenti" in out
    assert "3 chiusi e 1 ancora" in out
    assert "distribuiti su 2 linee" in out
    assert ">2</div>" in out, "il conteggio FAIL=2 non compare nel riquadro"


def test_method_section_omits_outcomes_with_no_experiments(bs):
    # a zero-count outcome must not render as a «0» tile: an empty box on a
    # results page reads as data, not as an absence.
    fake = {"lines": {"l1": "linea uno"}, "experiments": [
        {"id": "a", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "segnale"},
    ]}
    out = bs.render_method(fake)
    assert "soglia superata" in out
    assert "soglia mancata" not in out
    assert "misura non conclusiva" not in out


def test_method_section_reports_the_open_gate_count_from_data(bs):
    # the open-gate count quoted in the prose is the one the one-shot guard
    # protects: were they to diverge, the page would describe a rule other
    # than the one it enforces.
    fake = {"lines": {"l1": "x"}, "experiments": [
        {"id": "a", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "segnale"},
        {"id": "b", "line": "l1", "status": "aperto", "scope": "metodo"},
        {"id": "c", "line": "l1", "status": "in-attesa-campione", "scope": "metodo"},
    ]}
    assert bs.method_facts(fake)["n_open"] == 2
    assert "1 chiusi e 2 ancora" in bs.render_method(fake)


def test_pass_breakdown_separates_signal_from_method(bs):
    # the contract that prevents header/tally contradiction. Three PASSes of
    # which only one answers "is there a signal" must read as ONE, not three.
    fake = {"lines": {"l1": "x"},
            "scopes": {"segnale": "s", "metodo": "m"},
            "experiments": [
                {"id": "a", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "segnale"},
                {"id": "b", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "metodo"},
                {"id": "c", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "metodo"},
            ]}
    f = bs.method_facts(fake)
    assert f["tally"]["PASS"] == 3
    assert f["n_pass_signal"] == 1 and f["n_pass_other"] == 2
    out = bs.render_method(fake)
    assert "Un solo PASS riguarda" in out
    assert "gli altri 2 riguardano" in out


def test_pass_breakdown_agrees_grammatically_at_zero_and_one(bs):
    # tallies are DERIVED and may be 0 or 1: a plural-only sentence is a
    # latent defect that surfaces the day the registry changes.
    base = {"lines": {"l1": "x"}, "scopes": {"segnale": "s", "metodo": "m"}}
    only_method = dict(base, experiments=[
        {"id": "a", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "metodo"}])
    out = bs.render_method(only_method)
    assert "Nessuno dei PASS riguarda" in out and "l'altro riguarda" in out

    only_signal = dict(base, experiments=[
        {"id": "a", "line": "l1", "status": "chiuso", "verdict": "PASS", "scope": "segnale"}])
    out = bs.render_method(only_signal)
    assert "Un solo PASS riguarda" in out and "non ce ne sono altri" in out
    assert "sostituita con una più forte" not in out, \
        "senza PASS di metodo la frase sulla baseline sostituita non ha referente"


def test_real_registry_has_exactly_one_signal_pass_matching_the_header(bs, reg):
    # the header claim ("a single PASS") is a fact about the registry: if a
    # second signal gate passed, the header would need rewriting and this
    # test forces it.
    assert bs.method_facts(reg)["n_pass_signal"] == 1


# ───────────────────── two languages: the second one is a SECOND render ─────────────────────
def test_english_is_the_canonical_page_file(bs, reg):
    # same convention as every doc in the repo (`X.md` English, `X.it.md` Italian): the
    # Pages root `index.html` is the English page. The render functions keep Italian as
    # their default argument, which is what the prose tests above exercise.
    assert bs.PAGE_FILES == {"en": "index.html", "it": "index.it.html"}
    assert bs.render_method(reg) == bs.render_method(reg, "it")
    assert bs.render_limits() == bs.render_limits("it")


def test_one_shot_guard_holds_on_the_english_render_too(bs, reg):
    # THE test that justifies treating the toggle as a risk. The one-shot guard
    # was verified on ONE render; a second language is a second rendering path,
    # and "no digit in an open card" must hold there too.
    lines = reg["lines"]
    for lang in bs.LANGS:
        for e in _open_cards(bs, reg):
            card = bs.render_card(e, {}, lines, lang)
            area = re.search(r'<p class="none">(.*?)</p>', card, re.S)
            assert area, f"[{e['id']}/{lang}] scheda aperta senza il blocco che spiega l'assenza"
            assert not re.search(r"\d", area.group(1)), f"[{e['id']}/{lang}] cifra in una scheda aperta"


def test_both_languages_render_and_do_not_leak_into_each_other(bs, reg):
    # a forgotten translation breaks nothing: it produces a mixed page, which is
    # how bilingualism silently rots. The sentinels are structural sentences of
    # the sections, not isolated words that might legitimately appear in a quote.
    facts = bs.load_model_facts()
    it_only = ["Come si decide se una cosa ha funzionato", "esperimenti registrati",
               "soglia superata", "Assunzioni e limiti, dichiarati"]
    en_only = ["How it is decided whether something worked", "registered experiments",
               "threshold passed", "Assumptions and limits, declared"]
    it_page = bs.render_method(reg, "it") + bs.render_limits("it") + bs.render_section1(facts, "it")
    en_page = bs.render_method(reg, "en") + bs.render_limits("en") + bs.render_section1(facts, "en")
    for token in it_only:
        assert token in it_page and token not in en_page, f"IT/EN mescolate su {token!r}"
    for token in en_only:
        assert token in en_page and token not in it_page, f"IT/EN mescolate su {token!r}"


def test_the_two_pages_link_to_each_other(bs):
    # without the reciprocal link the EN page exists but is unreachable, which
    # is the same as not having written it.
    assert 'href="index.html"' in bs.PAGE_CHROME["it"]["other"]
    assert 'href="index.it.html"' in bs.PAGE_CHROME["en"]["other"]


def test_english_cards_show_no_italian_schema_values(bs, reg):
    # verdicts and scopes are Italian schema keys; the English page must translate them,
    # as it does the number labels.
    reports = bs.load_reports(bs.required_reports(reg["experiments"]))
    html_en = "".join(bs.render_card(e, reports, reg["lines"], "en", reg["scopes"])
                      for e in reg["experiments"])
    for token in ("NESSUNA CONCLUSIONE", "segnale", "realizzazione", "QLIKE modello"):
        assert token not in html_en, f"Italian schema text on the English page: {token!r}"


def test_hero_claim_is_read_from_the_canonical_reports(bs):
    # the headline band is 1 - NN/HAR-C from the canonical pair's reports on both splits,
    # never a typed number.
    reports = bs.load_reports({bs.HERO_GATE_REPORT, *bs.HERO_CLAIM_REPORTS.values()})
    h = bs.hero_facts(reports)
    for split, rel in bs.HERO_CLAIM_REPORTS.items():
        r = reports[rel]
        exp = (1 - r["metrics"]["nn"]["qlike"] / r["har_cj"]["har_c"]["qlike_har_c"]) * 100
        assert h[f"{split}_edge_pct"] == pytest.approx(exp)
    out = bs.render_hero(reports, "en")
    assert f"−{h['test_edge_pct']:.2f}%" in out and f"−{h['val_edge_pct']:.2f}%" in out


def test_hero_refuses_an_unverified_claim_report(bs):
    reports = bs.load_reports({bs.HERO_GATE_REPORT, *bs.HERO_CLAIM_REPORTS.values()})
    rel = bs.HERO_CLAIM_REPORTS["test"]
    broken = dict(reports, **{rel: dict(reports[rel], provenance={"matches": None})})
    with pytest.raises(bs.RegistrySchemaError, match="provenance"):
        bs.hero_facts(broken)


def test_no_open_gate_renders_an_explicit_sentence(bs, reg):
    closed_only = dict(reg, experiments=[e for e in reg["experiments"]
                                         if e["status"] not in bs.OPEN_STATES])
    reports = bs.load_reports(bs.required_reports(closed_only["experiments"]))
    assert "No pre-registered gate is open" in bs.render_experiments(closed_only, reports, "en")


def test_every_card_label_exists_in_both_languages(bs):
    # keys must match: a missing one would raise KeyError on the EN build, but
    # only for cards using that field — a failure surfacing when a card is
    # added, not when the code breaks.
    assert set(bs.CARD_LABELS["it"]) == set(bs.CARD_LABELS["en"])
    assert set(bs.PAGE_CHROME["it"]) == set(bs.PAGE_CHROME["en"])
