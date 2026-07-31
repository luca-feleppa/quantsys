# IT: GENERATORE DEL SITO DI PROGETTO — HTML statico per GitHub Pages.
#     Perche' un generatore invece di scrivere l'HTML a mano: un audit del 2026-07-28
#     ha trovato due claim STALE nel README, sopravvissuti perche' erano numeri
#     trascritti a mano che nessuno ricontrollava. La regola che ne e' uscita e' che
#     un numero derivabile a macchina va DERIVATO. Qui vale per le metriche dei
#     giudici, per il conteggio dei test e per il timbro di build.
#     ⚠ GUARD DELLA DISCIPLINA ONE-SHOT (V1) — la ragione principale per cui questo
#     file esiste. Un gate pre-registrato ancora APERTO non deve mostrare nessuna
#     quantita' decisionale: PnL, edge, QLIKE, p-value. Una volta che un numero e'
#     stato visto non si puo' non-vederlo, e la pre-registrazione e' bruciata. Il
#     guard NON e' una raccomandazione a chi scrive: e' un controllo che fa fallire
#     la build (`OneShotViolation`). Vedi `assert_one_shot_discipline`.
# EN: PROJECT SITE GENERATOR — static HTML for GitHub Pages.
#     Why a generator instead of hand-written HTML: a 2026-07-28 audit found two
#     STALE claims in the README, which survived because they were hand-copied
#     numbers nobody re-checked. The resulting rule is that a machine-derivable
#     number must be DERIVED. That applies here to judge metrics, the test count
#     and the build stamp.
#     ⚠ ONE-SHOT DISCIPLINE GUARD (V1) — the main reason this file exists. A
#     pre-registered gate that is still OPEN must show no decisional quantity:
#     PnL, edge, QLIKE, p-value. Once a number has been seen it cannot be unseen,
#     and the pre-registration is burnt. The guard is NOT advice to the author:
#     it is a check that fails the build (`OneShotViolation`).
import argparse
import html
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantsys.utils import setup_logging  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.build_site")

OUT_DIR = ROOT / "docs"
OUT_HTML = OUT_DIR / "index.html"

# IT: report attesi. Se uno manca il generatore SI FERMA e lo dice: meglio nessuna
#     pagina che una pagina con un buco silenzioso (blueprint §7, fail-fast).
# EN: expected reports. If one is missing the generator STOPS and says so: better
#     no page than a page with a silent hole.
REQUIRED_REPORTS = {
    "qlike_test": ROOT / "results" / "vols" / "qlike_report_1h_test.json",
}


class OneShotViolation(RuntimeError):
    """IT: build fermata per violazione della disciplina one-shot (V1).
    EN: build stopped on a one-shot discipline violation (V1)."""


# ────────────────────────────── modello dei gate ──────────────────────────────
# IT: un gate APERTO dichiara pre-registrazione, soglia e contatore — MAI numeri.
#     Un gate CHIUSO puo' portare numeri. Il campo `status` e' l'unica cosa che
#     autorizza `numbers`, e il guard lo verifica prima di emettere qualunque HTML.
# EN: an OPEN gate declares pre-registration, threshold and counter — NEVER
#     numbers. A CLOSED gate may carry numbers. `status` is the only thing that
#     authorizes `numbers`, and the guard checks it before emitting any HTML.
OPEN_STATES = {"aperto", "in-attesa-campione"}

GATES = [
    {
        "id": "vol-s",
        "name": "VOL-S — il NN batte una baseline econometrica sulla varianza?",
        "line": "vol 1h",
        "status": "chiuso",
        "verdict": "PASS",
        "numbers": {"fonte": "qlike_test"},
    },
    {
        "id": "e1",
        "name": "E1 — l'edge NN-vs-IV batte la previsione del mercato?",
        "line": "short-vol",
        "status": "aperto",
        "verdict": None,
        "threshold": "accordo di segno > 0.5 (p<0.05) E Spearman > 0 (IC bootstrap esclude lo zero)",
        "counter": "n ≥ 40 expiry liquidate dopo il 2026-08-01",
        "numbers": {},
    },
    {
        "id": "hedged",
        "name": "Hedged vs unhedged — il delta-hedge migliora il braccio short-vol?",
        "line": "short-vol",
        "status": "aperto",
        "verdict": None,
        "threshold": "3 condizioni pre-registrate, hardcoded nel giudice",
        "counter": "n ≥ 20 posizioni hedge-attive",
        "numbers": {},
    },
    {
        "id": "mfiv",
        "name": "MFIV v2 — il comparatore model-free ordina meglio dell'ATM?",
        "line": "short-vol",
        "status": "aperto",
        "verdict": None,
        "threshold": "Δρ di Spearman appaiato, MFIV vs ATM sui PnL per-expiry",
        "counter": "n ≥ 40 expiry qualificate",
        "numbers": {},
    },
]


def assert_one_shot_discipline(gates: list) -> None:
    # IT: IL controllo che giustifica il generatore. Un gate aperto con `numbers`
    #     non vuoto e' una violazione irreversibile della pre-registrazione:
    #     la build fallisce, non emette una pagina "quasi giusta".
    #     Controlla anche `verdict`: un verdetto E' una quantita' decisionale.
    # EN: THE check that justifies the generator. An open gate with non-empty
    #     `numbers` is an irreversible pre-registration violation: the build
    #     fails rather than emitting an "almost right" page. `verdict` is checked
    #     too: a verdict IS a decisional quantity.
    for g in gates:
        if g["status"] not in OPEN_STATES:
            continue
        if g.get("numbers"):
            raise OneShotViolation(
                f"gate '{g['id']}' e' APERTO ma dichiara numeri {list(g['numbers'])}: "
                f"emetterli brucerebbe la pre-registrazione. Build fermata.")
        if g.get("verdict"):
            raise OneShotViolation(
                f"gate '{g['id']}' e' APERTO ma dichiara verdict={g['verdict']!r}: "
                f"un verdetto e' una quantita' decisionale. Build fermata.")
    log.info(f"guard V1 OK: {sum(1 for g in gates if g['status'] in OPEN_STATES)} gate aperti, "
             f"nessuna quantita' decisionale associata")


# ──────────────────────────────── input derivati ────────────────────────────────
def build_stamp() -> dict:
    # IT: timbro di build (V5). Il commit rende verificabile a quale stato del
    #     repo si riferiscono i numeri della pagina.
    # EN: build stamp (V5). The commit makes the page's numbers traceable to a
    #     specific repo state.
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "n/d"
    return {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def _rel(path: Path) -> str:
    # IT: path leggibile per log ed errori. `relative_to` SOLLEVA se il path e'
    #     fuori dalla root, quindi usarlo dentro un messaggio d'errore
    #     sostituirebbe l'errore vero con un ValueError incomprensibile — il
    #     messaggio che deve spiegare il guasto non puo' essere esso stesso un
    #     guasto. (Trovato dal test `test_missing_report_stops_the_build`.)
    # EN: readable path for logs and errors. `relative_to` RAISES when the path
    #     is outside the root, so using it inside an error message would replace
    #     the real error with an opaque ValueError — the message meant to explain
    #     the failure must not itself fail.
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_reports() -> dict:
    # IT: fail-fast: un report atteso che manca ferma la build.
    # EN: fail-fast: a missing expected report stops the build.
    out = {}
    for key, path in REQUIRED_REPORTS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"report atteso assente: {_rel(path)} — "
                f"la pagina non viene emessa con un buco al suo posto")
        out[key] = json.loads(path.read_text(encoding="utf-8"))
        log.info(f"report letto: {_rel(path)}")
    return out


# ──────────────────────────────────── render ────────────────────────────────────
CSS = """
:root{--bg:#fbfaf7;--fg:#1b1a17;--mut:#6b6862;--line:#e2ded5;--accent:#8a5a2b;
--card:#fff;--warn-bg:#fdf6e8;--warn-line:#d9b56a;--ok:#2f6f4f;--no:#8c3b3b}
@media (prefers-color-scheme:dark){:root{--bg:#14130f;--fg:#eae7df;--mut:#9a958a;
--line:#2e2b25;--accent:#d0975a;--card:#1c1a16;--warn-bg:#241f14;--warn-line:#6b5322;
--ok:#6fbf95;--no:#e08b8b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:52rem;margin:0 auto;padding:2.5rem 1.25rem 5rem}
header{border-bottom:1px solid var(--line);padding-bottom:1.5rem;margin-bottom:2rem}
h1{font-size:1.9rem;line-height:1.2;margin:0 0 .4rem;letter-spacing:-.01em}
h2{font-size:1.3rem;margin:2.5rem 0 .75rem;letter-spacing:-.005em}
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
.big.ok{color:var(--ok)}.big.no{color:var(--no)}
.card p{margin:.35rem 0 0;font-size:.9rem;color:var(--mut);line-height:1.5}
.gates{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.92rem}
.gates th,.gates td{text-align:left;padding:.55rem .6rem;border-bottom:1px solid var(--line);
vertical-align:top}
.gates th{font-size:.75rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mut)}
.pill{display:inline-block;font-size:.72rem;padding:.12rem .5rem;border-radius:1rem;
border:1px solid var(--line);color:var(--mut);white-space:nowrap}
footer{margin-top:4rem;padding-top:1.25rem;border-top:1px solid var(--line);
color:var(--mut);font-size:.85rem}
a{color:var(--accent)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em;
background:var(--card);border:1px solid var(--line);border-radius:.25rem;padding:.05em .3em}
"""


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def render_hero(rep: dict, stamp: dict) -> str:
    # IT: blocco 1.0 "in una schermata" — cosa si predice, con che esito, e lo
    #     stato ONESTO delle due linee. L'avvertenza testnet (V4) sta in alto e
    #     non a fondo pagina: e' l'unica ambiguita' davvero costosa.
    # EN: block 1.0 "in one screen" — what is predicted, with what outcome, and
    #     the HONEST status of both lines. The testnet warning (V4) sits at the
    #     top, not in a footnote: it is the only genuinely costly ambiguity.
    g = rep["qlike_test"]["gate"]
    m = rep["qlike_test"]["metrics"]
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
    <div class="big ok">{g["verdict"]}</div>
    <p>Previsione della varianza realizzata a 30 ore. Batte la baseline econometrica
    del <b>{ratio_pct:.0f}%</b> in QLIKE sullo split di test
    (n = {g["n_obs"]:,}), con val→test coerenti.</p>
  </div>
  <div class="card">
    <h3>Linea direzionale</h3>
    <div class="big no">nessun alpha OOS</div>
    <p>Nessuna skill direzionale fuori campione a nessun timeframe testato.
    Il codice resta vivo come <b>controllo negativo</b>: serve a dimostrare
    che l'apparato di misura funziona anche quando il segnale non c'è.</p>
  </div>
  <div class="card">
    <h3>Confronto, sullo split di test</h3>
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


def render_gates(gates: list) -> str:
    # IT: i gate APERTI compaiono con pre-registrazione, soglia e contatore, e
    #     una cella esplicita al posto dei numeri — cosi' l'assenza e' visibile
    #     e leggibile come una scelta, non come una dimenticanza.
    # EN: OPEN gates appear with pre-registration, threshold and counter, and an
    #     explicit cell in place of the numbers — so the absence reads as a
    #     deliberate choice rather than an omission.
    rows = []
    for g in gates:
        if g["status"] not in OPEN_STATES:
            continue
        rows.append(
            f'<tr><td><b>{esc(g["name"])}</b><br><span class="pill">{esc(g["line"])}</span></td>'
            f'<td>{esc(g.get("threshold", "—"))}</td>'
            f'<td>{esc(g.get("counter", "—"))}</td>'
            f'<td><i>non disponibile per progetto</i></td></tr>')
    if not rows:
        return ""
    return f"""
<h2>Gate aperti — pre-registrati, non ancora giudicati</h2>
<p>Questi esperimenti hanno una soglia scritta e committata <b>prima</b> di girare, e un
numero minimo di osservazioni sotto il quale il giudice si rifiuta di produrre un verdetto.
Finché il campione non è pieno, <b>nessuna quantità decisionale viene mostrata qui</b>: un
numero visto in anticipo non si può non-vedere, e brucerebbe la pre-registrazione. L'assenza
in quest'ultima colonna è il punto, non una mancanza.</p>
<table class="gates">
<tr><th>Gate</th><th>Soglia pre-registrata</th><th>Campione richiesto</th><th>Numeri</th></tr>
{chr(10).join(rows)}
</table>
""".strip()


def render_page(rep: dict, stamp: dict, gates: list) -> str:
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
{render_hero(rep, stamp)}

{render_gates(gates)}

<footer>
<p>Le metriche in questa pagina sono <b>derivate dai report dei giudici</b> a ogni build,
non trascritte: il timbro in alto dice da quale stato del repository provengono.
La fonte tecnica autorevole resta <code>TEORIA.md</code> nel repository, che questa
pagina richiama e non sostituisce.</p>
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
                    help="esegue guard e lettura report senza scrivere nulla / "
                         "runs guards and report loading without writing")
    args = ap.parse_args()

    # IT: il guard PRIMA di tutto: se la disciplina e' violata non si legge
    #     nemmeno un report, cosi' il fallimento e' inequivocabile.
    # EN: the guard FIRST: on a violation not even a report is read, so the
    #     failure is unambiguous.
    assert_one_shot_discipline(GATES)
    rep = load_reports()
    stamp = build_stamp()

    if args.check:
        log.info("--check: guard superato e report leggibili, nessun file scritto")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # IT: .nojekyll — il sito e' HTML gia' finito; Jekyll aggiungerebbe una build
    #     che non controlliamo e che puo' riscrivere i path.
    # EN: .nojekyll — the site is finished HTML; Jekyll would add a build step we
    #     do not control and that can rewrite paths.
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
    OUT_HTML.write_text(render_page(rep, stamp, GATES), encoding="utf-8")

    size = OUT_HTML.stat().st_size
    log.info(f"sito scritto: {OUT_HTML.relative_to(ROOT)} ({size} B) + .nojekyll")
    print(f"\n  {OUT_HTML.relative_to(ROOT)}  {size} B")
    print(f"  build {stamp['utc']} · commit {stamp['commit']}")
    print(f"  gate aperti senza numeri: "
          f"{sum(1 for g in GATES if g['status'] in OPEN_STATES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
