# IT: GENERATORE DEL SITO DI PROGETTO — HTML statico per GitHub Pages.
#     Perche' un generatore invece di scrivere l'HTML a mano: un audit del 2026-07-28
#     ha trovato due claim STALE nel README, sopravvissuti perche' erano numeri
#     trascritti a mano che nessuno ricontrollava. La regola che ne e' uscita e' che
#     un numero derivabile a macchina va DERIVATO.
#     ⚠ NIENTE TERZA COPIA DEI FATTI. Le schede degli esperimenti vivono in
#     `docs/experiments.yaml`, non qui: una prima versione di questo file aveva i
#     gate hardcoded, creando una terza copia dopo `STATUS.md` e `TEORIA.md` §12.
#     E i NUMERI, dove esistono in un report su disco, non sono nemmeno nel
#     registro: il registro dichiara file e chiave, il valore si legge a ogni
#     build. Cosi' la divergenza non e' «rilevata», e' impossibile.
#     ⚠ GUARD DELLA DISCIPLINA ONE-SHOT — un gate pre-registrato ancora APERTO non
#     deve mostrare nessuna quantita' decisionale, verdetto incluso. Un numero
#     visto in anticipo non si puo' non-vedere e brucia la pre-registrazione. Il
#     guard NON e' una raccomandazione: alza `OneShotViolation` e ferma la build.
# EN: PROJECT SITE GENERATOR — static HTML for GitHub Pages.
#     Why a generator rather than hand-written HTML: a 2026-07-28 audit found two
#     STALE claims in the README, which survived because they were hand-copied
#     numbers nobody re-checked. The rule that followed: a machine-derivable
#     number must be DERIVED.
#     ⚠ NO THIRD COPY OF THE FACTS. Experiment cards live in
#     `docs/experiments.yaml`, not here: an earlier version of this file had the
#     gates hardcoded, creating a third copy after `STATUS.md` and `TEORIA.md`
#     §12. And the NUMBERS, where a report exists on disk, are not even in the
#     registry: the registry declares file and key, the value is read at build
#     time. Divergence is therefore impossible rather than «detected».
#     ⚠ ONE-SHOT DISCIPLINE GUARD — an OPEN pre-registered gate must show no
#     decisional quantity, verdict included. It raises `OneShotViolation`.
import argparse
import html
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantsys.utils import setup_logging  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.build_site")

OUT_DIR = ROOT / "docs"
OUT_HTML = OUT_DIR / "index.html"
REGISTRY = OUT_DIR / "experiments.yaml"

LANG = "it"          # IT/EN: una sola lingua in questo incremento (toggle: passo 4)
OPEN_STATES = {"aperto", "in-attesa-campione"}
VERDICT_CLASS = {"PASS": "ok", "FAIL": "no", "NESSUNA CONCLUSIONE": "mid"}


class OneShotViolation(RuntimeError):
    """IT: build fermata per violazione della disciplina one-shot.
    EN: build stopped on a one-shot discipline violation."""


class RegistrySchemaError(RuntimeError):
    """IT: il registro degli esperimenti non rispetta lo schema.
    EN: the experiment registry violates the schema."""


# ────────────────────────────────── registro ──────────────────────────────────
def load_registry() -> dict:
    if not REGISTRY.exists():
        raise FileNotFoundError(f"registro assente: {_rel(REGISTRY)}")
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    validate_registry(reg)
    log.info(f"registro letto: {_rel(REGISTRY)} "
             f"({len(reg['experiments'])} schede, {len(reg['lines'])} linee)")
    return reg


def validate_registry(reg: dict) -> None:
    # IT: schema minimo + le due regole che impediscono la duplicazione:
    #     (a) ogni scheda punta a una sezione di TEORIA — la fonte autorevole;
    #     (b) i numeri sono O derivati da un report O letterali con una data,
    #         mai entrambi, mai letterali senza data. Un letterale senza data e'
    #         un numero che nessuno potra' piu' verificare.
    # EN: minimal schema + the two rules that prevent duplication: every card
    #     points at a TEORIA section (the authority), and numbers are EITHER
    #     derived from a report OR literal with a date, never both.
    if reg.get("schema") != 1:
        raise RegistrySchemaError(f"schema atteso 1, trovato {reg.get('schema')!r}")
    seen = set()
    for e in reg["experiments"]:
        eid = e.get("id")
        if not eid:
            raise RegistrySchemaError("scheda senza `id`")
        if eid in seen:
            raise RegistrySchemaError(f"id duplicato: {eid!r}")
        seen.add(eid)
        for field in ("name", "line", "status", "question", "teoria"):
            if not e.get(field):
                raise RegistrySchemaError(f"[{eid}] campo obbligatorio mancante: {field}")
        if e["line"] not in reg["lines"]:
            raise RegistrySchemaError(f"[{eid}] linea sconosciuta: {e['line']!r}")
        n = e.get("numbers")
        if n:
            has_src, has_lit = "source" in n, "literal" in n
            if has_src and has_lit:
                raise RegistrySchemaError(
                    f"[{eid}] `numbers` ha sia `source` sia `literal`: scegline uno — "
                    f"due origini per lo stesso numero sono una duplicazione")
            if not has_src and not has_lit:
                raise RegistrySchemaError(f"[{eid}] `numbers` senza `source` ne' `literal`")
            if has_lit and not n.get("as_of"):
                raise RegistrySchemaError(
                    f"[{eid}] `numbers.literal` senza `as_of`: un numero scritto a mano "
                    f"senza la data in cui era vero non e' verificabile")


def assert_one_shot_discipline(experiments: list) -> None:
    # IT: IL controllo che giustifica il generatore. Vale anche per `verdict`:
    #     un verdetto E' una quantita' decisionale, «PASS» su un gate aperto e'
    #     informativo quanto il numero che lo produce.
    # EN: THE check that justifies the generator. It covers `verdict` too: a
    #     verdict IS a decisional quantity.
    n_open = 0
    for e in experiments:
        if e["status"] not in OPEN_STATES:
            continue
        n_open += 1
        if e.get("numbers"):
            raise OneShotViolation(
                f"gate '{e['id']}' e' APERTO ma dichiara numeri: emetterli "
                f"brucerebbe la pre-registrazione. Build fermata.")
        if e.get("verdict"):
            raise OneShotViolation(
                f"gate '{e['id']}' e' APERTO ma dichiara verdict={e['verdict']!r}: "
                f"un verdetto e' una quantita' decisionale. Build fermata.")
    log.info(f"guard one-shot OK: {n_open} gate aperti, nessuna quantita' decisionale")


def assert_teoria_pointers(experiments: list) -> None:
    # IT: ogni scheda deve puntare a una sezione ESISTENTE di TEORIA.md. E' il
    #     meccanismo che impedisce al registro di diventare una fonte parallela:
    #     se la prosa autorevole non c'e', la scheda non e' pubblicabile.
    # EN: every card must point at an EXISTING TEORIA.md section. It is what
    #     stops the registry from becoming a parallel source of truth.
    teoria = (ROOT / "TEORIA.md").read_text(encoding="utf-8")
    for e in experiments:
        ref = str(e["teoria"])
        if f"### {ref} " not in teoria and f"## {ref}." not in teoria and f"### {ref}." not in teoria:
            raise RegistrySchemaError(
                f"[{e['id']}] punta a TEORIA.md §{ref}, che non esiste — il registro "
                f"non e' la fonte autorevole e non puo' contenere fatti senza di essa")
    log.info(f"puntatori a TEORIA.md: {len(experiments)}/{len(experiments)} risolvono")


# ────────────────────────────── numeri e derivazione ──────────────────────────────
def _rel(path: Path) -> str:
    # IT: path leggibile. `relative_to` SOLLEVA fuori dalla root, quindi usarlo in
    #     un messaggio d'errore sostituirebbe l'errore vero con un ValueError: il
    #     messaggio che spiega il guasto non puo' essere esso stesso un guasto.
    # EN: readable path. `relative_to` RAISES outside the root, so using it inside
    #     an error message would mask the real error.
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dig(obj, path):
    # IT: risolve un percorso dentro il JSON del report. DUE forme, e la seconda
    #     non e' un vezzo: alcune chiavi dei report contengono un punto nel nome
    #     (`g3_real_hit_rate_gt_0.5`), quindi la notazione puntata non basta a
    #     esprimerle. Una stringa viene spezzata sui punti; una LISTA e' presa
    #     come sequenza esplicita di segmenti.
    #     L'errore e' diagnostico e non solo negativo: dice a che livello si e'
    #     fermato e cosa c'era disponibile li', altrimenti un path sbagliato
    #     costa una sessione di debug per una virgola.
    # EN: resolves a path inside a report JSON. TWO forms, and the second is not
    #     a flourish: some report keys contain a dot in their name, so dotted
    #     notation cannot express them. A string is split on dots; a LIST is
    #     taken as an explicit sequence of segments.
    parts = list(path) if isinstance(path, (list, tuple)) else str(path).split(".")
    cur = obj
    for i, part in enumerate(parts):
        if not isinstance(cur, dict) or part not in cur:
            here = ".".join(str(p) for p in parts[:i]) or "(radice)"
            avail = ", ".join(sorted(cur)[:8]) if isinstance(cur, dict) else type(cur).__name__
            raise KeyError(
                f"percorso non risolto: {parts!r} — fermo a {here!r}, "
                f"segmento {part!r} assente. Disponibili: {avail}. "
                f"Se il nome della chiave contiene un punto, usa la forma a lista.")
        cur = cur[part]
    return cur


def required_reports(experiments: list) -> dict:
    # IT: i report necessari sono DERIVATI dal registro, non elencati a parte:
    #     aggiungere una scheda con una nuova sorgente rende quel report
    #     obbligatorio senza toccare questo file.
    # EN: required reports are DERIVED from the registry rather than listed
    #     separately: adding a card with a new source makes that report required
    #     without touching this file.
    return {e["numbers"]["source"] for e in experiments
            if e.get("numbers", {}).get("source")}


def load_reports(paths: set) -> dict:
    # IT: fail-fast — meglio nessuna pagina che una pagina con un buco silenzioso.
    # EN: fail-fast — better no page than a page with a silent hole.
    out = {}
    for rel in sorted(paths):
        p = ROOT / rel
        if not p.exists():
            raise FileNotFoundError(
                f"report atteso assente: {rel} — la pagina non viene emessa con "
                f"un buco al suo posto")
        out[rel] = json.loads(p.read_text(encoding="utf-8"))
        log.info(f"report letto: {rel}")
    return out


def fmt(v) -> str:
    if isinstance(v, bool):
        return "sì" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ")
    return str(v)


def resolve_numbers(exp: dict, reports: dict) -> list:
    # IT: ritorna [(etichetta, valore, origine)]. `derivato` = letto dal report a
    #     questa build; `2026-07-30` = scritto a mano, vero a quella data.
    # EN: returns [(label, value, provenance)].
    n = exp.get("numbers")
    if not n:
        return []
    if "source" in n:
        rep = reports[n["source"]]
        return [(k, fmt(dig(rep, path)), "derivato") for k, path in n["keys"].items()]
    return [(k, str(v), str(n["as_of"])) for k, v in n["literal"].items()]


# ──────────────────────────────────── render ────────────────────────────────────
CSS = """
:root{--bg:#fbfaf7;--fg:#1b1a17;--mut:#6b6862;--line:#e2ded5;--accent:#8a5a2b;
--card:#fff;--warn-bg:#fdf6e8;--warn-line:#d9b56a;--ok:#2f6f4f;--no:#8c3b3b;--mid:#7a6a2f}
@media (prefers-color-scheme:dark){:root{--bg:#14130f;--fg:#eae7df;--mut:#9a958a;
--line:#2e2b25;--accent:#d0975a;--card:#1c1a16;--warn-bg:#241f14;--warn-line:#6b5322;
--ok:#6fbf95;--no:#e08b8b;--mid:#cbb46a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:52rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
header{border-bottom:1px solid var(--line);padding-bottom:1.5rem;margin-bottom:2rem}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 .4rem;letter-spacing:-.01em}
h2{font-size:1.3rem;margin:3rem 0 .75rem;letter-spacing:-.005em}
.sub{color:var(--mut);margin:0;font-size:1.02rem}
.stamp{color:var(--mut);font-size:.82rem;margin-top:1rem;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.warn{background:var(--warn-bg);border:1px solid var(--warn-line);
border-radius:.5rem;padding:.9rem 1.1rem;margin:1.5rem 0;font-size:.95rem}
.warn b{color:var(--accent)}
.grid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));margin:1.5rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:.5rem;padding:1rem 1.1rem}
.card h3{margin:0 0 .35rem;font-size:.78rem;text-transform:uppercase;
letter-spacing:.09em;color:var(--mut);font-weight:600}
.big{font-size:1.55rem;font-weight:600;letter-spacing:-.01em}
.big.ok{color:var(--ok)}.big.no{color:var(--no)}.big.mid{color:var(--mid)}
.card p{margin:.35rem 0 0;font-size:.9rem;color:var(--mut);line-height:1.5}
.exp{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--line);
border-radius:.4rem;padding:1.1rem 1.2rem;margin:1.1rem 0}
.exp.ok{border-left-color:var(--ok)}.exp.no{border-left-color:var(--no)}
.exp.mid{border-left-color:var(--mid)}.exp.open{border-left-color:var(--accent)}
.exp h3{margin:0 0 .5rem;font-size:1.03rem;line-height:1.35}
.meta{font-size:.75rem;color:var(--mut);margin:0 0 .8rem;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.verdict{font-weight:600;letter-spacing:.02em}
.verdict.ok{color:var(--ok)}.verdict.no{color:var(--no)}.verdict.mid{color:var(--mid)}
.verdict.open{color:var(--accent)}
dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:.3rem .9rem;font-size:.91rem}
dt{color:var(--mut);font-size:.74rem;text-transform:uppercase;letter-spacing:.07em;
padding-top:.22rem;white-space:nowrap}
dd{margin:0}
.nums{margin:.8rem 0 0;font-size:.88rem;border-top:1px solid var(--line);padding-top:.7rem}
.nums span{display:inline-block;margin-right:1.2rem;white-space:nowrap}
.nums b{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}
.prov{font-size:.72rem;color:var(--mut);margin-top:.45rem}
.none{color:var(--mut);font-style:italic;font-size:.9rem;margin:.8rem 0 0;
border-top:1px solid var(--line);padding-top:.7rem}
footer{margin-top:4rem;padding-top:1.25rem;border-top:1px solid var(--line);
color:var(--mut);font-size:.85rem}
a{color:var(--accent)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;
background:var(--bg);border:1px solid var(--line);border-radius:.25rem;padding:.05em .3em}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def tr(field, lang=LANG) -> str:
    # IT/EN: i campi bilingui sono dizionari {it, en}; i campi semplici passano.
    return field.get(lang, field.get("it", "")) if isinstance(field, dict) else str(field)


def md(s: str) -> str:
    # IT: markdown minimo (grassetto, corsivo, codice) — evita una dipendenza.
    # EN: minimal markdown (bold, italic, code) — avoids a dependency.
    import re
    s = esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<i>\1</i>", s)
    return s


def render_hero(reports: dict) -> str:
    g = reports["results/vols/qlike_report_1h_test.json"]["gate"]
    m = reports["results/vols/qlike_report_1h_test.json"]["metrics"]
    ratio_pct = (1.0 - g["nn_vs_har_ratio"]) * 100.0
    return f"""
<div class="warn">
<b>Stato onesto, in alto e non in nota.</b> Il braccio operativo di questo progetto è
<b>paper trading su testnet</b>: nessuna esecuzione con capitale reale, nessuno slippage
reale, nessun impatto di mercato. Ciò che è misurato è <i>accuratezza previsiva</i>; la
monetizzazione è una domanda separata, e finora senza risposta positiva.
</div>

<div class="grid">
  <div class="card">
    <h3>Linea volatilità · 1h</h3>
    <div class="big ok">{esc(g["verdict"])}</div>
    <p>Previsione della varianza realizzata a 30 ore. Batte la baseline econometrica
    del <b>{ratio_pct:.0f}%</b> in QLIKE sullo split di test (n = {g["n_obs"]:,}),
    con val→test coerenti.</p>
  </div>
  <div class="card">
    <h3>Linea direzionale</h3>
    <div class="big no">nessun alpha OOS</div>
    <p>Nessuna skill direzionale fuori campione a nessun timeframe testato.
    Il codice resta vivo come <b>controllo negativo</b>: dimostra che l'apparato
    di misura funziona anche quando il segnale non c'è.</p>
  </div>
  <div class="card">
    <h3>Confronto, split di test</h3>
    <div class="big">{m["nn"]["qlike"]:.3f}</div>
    <p>QLIKE del modello, contro <b>{m["har"]["qlike"]:.3f}</b> della baseline
    econometrica e <b>{m["naive"]["qlike"]:.3f}</b> della persistenza ingenua.
    Più basso è meglio.</p>
  </div>
</div>

<p>La tesi di questo progetto non è il risultato positivo: è il <b>metodo</b>. Un singolo
PASS su un target di varianza, circondato da un corpus esteso di fallimenti
<i>pre-registrati prima di girare</i>, dice molto di più di un PASS isolato — perché
la soglia esisteva prima del numero, e i risultati negativi sono scritti con la stessa
cura di quello positivo.</p>
""".strip()


def render_card(e: dict, reports: dict, lines: dict) -> str:
    is_open = e["status"] in OPEN_STATES
    verdict = "APERTO" if is_open else str(e.get("verdict", "—"))
    cls = "open" if is_open else VERDICT_CLASS.get(verdict, "")
    d = e.get("dates", {})
    when = f"pre-reg {d.get('prereg', '—')}" + (f" → chiuso {d['closed']}" if d.get("closed") else "")

    rows = [("Domanda", tr(e["question"])), ("Prior dichiarato", tr(e["prior"]))]
    if e.get("lever"):
        rows.append(("Leva", tr(e["lever"])))
    rows.append(("Soglia", tr(e["threshold"])))
    if is_open and e.get("counter"):
        rows.append(("Campione richiesto", tr(e["counter"])))
    if e.get("consequence"):
        rows.append(("Conseguenza", tr(e["consequence"])))
    if e.get("note"):
        rows.append(("Nota", tr(e["note"])))
    if e.get("data_note"):
        rows.append(("Sui dati", tr(e["data_note"])))
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{md(v)}</dd>" for k, v in rows)

    if is_open:
        nums = ('<p class="none">Nessun numero: il gate è aperto. Mostrarne uno prima '
                'che il campione sia pieno brucerebbe la pre-registrazione — l\'assenza '
                'qui è il punto, non una mancanza.</p>')
    else:
        got = resolve_numbers(e, reports)
        if got:
            prov = got[0][2]
            cells = "".join(f'<span>{esc(k)} <b>{esc(v)}</b></span>' for k, v, _ in got)
            src = ("letti dal report dei giudici a questa build"
                   if prov == "derivato" else f"scritti a mano, veri al {prov}")
            nums = f'<div class="nums">{cells}<div class="prov">{src}</div></div>'
        else:
            nums = ""

    return f"""
<div class="exp {cls}">
  <h3>{md(tr(e["name"]))}</h3>
  <p class="meta">{esc(tr(lines[e["line"]]))} · {esc(when)} · split {esc(e.get("split", "—"))}
  · <span class="verdict {cls}">{esc(verdict)}</span> · dettaglio in TEORIA.md §{esc(e["teoria"])}</p>
  <dl>{dl}</dl>
  {nums}
</div>""".rstrip()


def render_experiments(reg: dict, reports: dict) -> str:
    lines = {k: v for k, v in reg["lines"].items()}
    exps = reg["experiments"]
    closed = [e for e in exps if e["status"] not in OPEN_STATES]
    opened = [e for e in exps if e["status"] in OPEN_STATES]
    body = "".join(render_card(e, reports, lines) for e in closed)
    body_open = "".join(render_card(e, reports, lines) for e in opened)
    return f"""
<h2>Cosa è stato provato, e com'è andata</h2>
<p>Ogni scheda riporta la domanda, il <b>prior dichiarato prima di girare</b>, la leva,
la soglia numerica scritta in anticipo e l'esito — qualunque esso sia. Il campo che dà
valore alla pagina è il prior: mostra che il criterio esisteva prima del risultato.
Dove il prior è stato falsificato <i>nella direzione</i>, è detto esplicitamente.</p>
{body}

<h2>Gate aperti — pre-registrati, non ancora giudicati</h2>
<p>Questi esperimenti hanno soglia e campione minimo scritti e committati <b>prima</b> di
girare. Finché il campione non è pieno non compare nessuna quantità decisionale: un numero
visto in anticipo non si può non-vedere.</p>
{body_open}""".rstrip()


def build_stamp() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "n/d"
    return {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "commit": git("rev-parse", "--short", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD")}


def render_page(reg: dict, reports: dict, stamp: dict) -> str:
    return f"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QUANTSYS — motore predittivo su BTC/USDT</title>
<meta name="description" content="Come funziona un motore predittivo su BTC/USDT, e la storia completa di cosa e' stato provato e com'e' andata.">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>QUANTSYS</h1>
  <p class="sub">Un motore predittivo su BTC/USDT — e la storia completa,
  positiva e negativa, di cosa è stato provato.</p>
  <p class="stamp">build {esc(stamp["utc"])} · commit {esc(stamp["commit"])}
  · branch {esc(stamp["branch"])}</p>
</header>

<h2>In una schermata</h2>
{render_hero(reports)}

{render_experiments(reg, reports)}

<footer>
<p>Le schede vengono da <code>docs/experiments.yaml</code>; i numeri marcati «letti dal
report» sono derivati a ogni build e non trascritti, quindi non possono divergere dalla
fonte. Il timbro in alto dice da quale stato del repository provengono. La fonte tecnica
autorevole resta <code>TEORIA.md</code>, che questa pagina richiama e non sostituisce:
dove le due divergessero, ha ragione TEORIA.</p>
</footer>
</div>
</body>
</html>
"""


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Genera il sito statico di progetto in docs/")
    ap.add_argument("--check", action="store_true",
                    help="guard + validazione senza scrivere nulla / guards and validation only")
    args = ap.parse_args()

    reg = load_registry()
    exps = reg["experiments"]
    # IT: i guard PRIMA di leggere i report: un fallimento di disciplina non deve
    #     poter essere confuso con un problema di dati.
    # EN: guards BEFORE loading reports: a discipline failure must not be
    #     confusable with a data problem.
    assert_one_shot_discipline(exps)
    assert_teoria_pointers(exps)
    reports = load_reports(required_reports(exps) | {"results/vols/qlike_report_1h_test.json"})
    stamp = build_stamp()

    if args.check:
        log.info("--check: guard superati, registro valido, report leggibili; nulla scritto")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
    OUT_HTML.write_text(render_page(reg, reports, stamp), encoding="utf-8")

    n_open = sum(1 for e in exps if e["status"] in OPEN_STATES)
    derived = sum(1 for e in exps if e.get("numbers", {}).get("source"))
    literal = sum(1 for e in exps if e.get("numbers", {}).get("literal"))
    log.info(f"sito scritto: {_rel(OUT_HTML)} ({OUT_HTML.stat().st_size} B)")
    print(f"\n  {_rel(OUT_HTML)}  {OUT_HTML.stat().st_size:,} B")
    print(f"  build {stamp['utc']} · commit {stamp['commit']}")
    print(f"  schede: {len(exps)} ({len(exps)-n_open} chiuse, {n_open} aperte senza numeri)")
    print(f"  numeri: {derived} schede derivate dai report · {literal} scritte a mano con data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
