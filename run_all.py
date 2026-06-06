"""
QUANTSYS — Orchestratore completo.

Esegui dalla root del progetto:
  python run_all.py                  # menu interattivo (architettura + fasi)
  python run_all.py --skip-train     # modalità flag (salta menu)
  python run_all.py --only-dashboard # apre solo la dashboard

Senza flag: mostra un menu interattivo con checkbox per selezionare
architettura e fasi della pipeline (↑↓ naviga, SPAZIO seleziona, INVIO conferma).

Flag disponibili (modalità diretta, salta il menu):
  --arch            architettura modello: lstm | itransformer | tcnmamba
  --n-ensemble N    seed nell'ensemble per il training single-arch (--arch), default 5;
                    il path --distill resta SEMPRE a n_ensemble=1 (teacher+student)
  --skip-update     salta aggiornamento dati (usa dataset esistente)
  --skip-macro      salta download macro FRED/yFinance
  --skip-train      salta riaddestramento (usa modello esistente)
  --skip-backtest   salta backtest
  --skip-walkfwd    salta walk-forward validation
  --skip-analyze    salta analisi segnali live (05_analyze_signals.py)
  --skip-live       non avvia il feed live segnali
  --no-browser      non apre il browser automaticamente
  --only-dashboard  apre solo la dashboard (nessun calcolo)
  --force-download  forza download completo anche se raw_candles.parquet esiste
"""
import argparse
import os

# ─── Resource limits (prima di qualsiasi import numerico) ────────────────────
# IT: cap thread BLAS/OMP da cpu_fraction PRIMA di import numpy/torch (altrimenti ignorato).
# EN: cap BLAS/OMP threads from cpu_fraction BEFORE importing numpy/torch (else ignored).
import yaml as _yaml
with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), encoding="utf-8") as _f:
    _cpu_frac = _yaml.safe_load(_f).get("hardware", {}).get("cpu_fraction", 0.5)
_cpu_limit = str(max(1, int(os.cpu_count() * _cpu_frac)))
os.environ["OMP_NUM_THREADS"]     = _cpu_limit
os.environ["MKL_NUM_THREADS"]     = _cpu_limit
os.environ["OPENBLAS_NUM_THREADS"] = _cpu_limit
os.environ["NUMEXPR_NUM_THREADS"] = _cpu_limit

import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
MODELS_DIR    = ROOT / "models"      # root base; arch-specific: MODELS_DIR / arch
RESULTS_DIR   = ROOT / "results"     # root base; arch-specific: RESULTS_DIR / arch
RAW_CANDLES   = DATA_DIR / "raw_candles.parquet"
DASHBOARD_URL = "http://localhost:8050"
PYTHON        = sys.executable

# Arch-specific paths — set in main() after parse_args(); initialised to lstm defaults
ARCH_MODELS_DIR = MODELS_DIR / "lstm"
ARCH_RESULTS_DIR = RESULTS_DIR / "lstm"
MODEL_FILE      = ARCH_MODELS_DIR / "best_model.pt"


# IT: legge l'architettura di default dal config (fallback lstm).
# EN: reads the default architecture from config (fallback lstm).
def _default_arch() -> str:
    """Legge l'architettura di default da config/default.yaml."""
    try:
        import re
        txt = (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
        m = re.search(r'architecture:\s*["\']?(\w+)["\']?', txt)
        if m and m.group(1) in ("lstm", "itransformer", "tft", "tcnmamba", "nhits"):
            return m.group(1)
    except Exception:
        pass
    return "lstm"


# IT: costruisce l'etichetta leggibile dell'arch per i banner (con iperparametri).
# EN: builds the human-readable arch label for banners (with hyperparams).
def _arch_label(arch: str = None) -> str:
    """Legge l'architettura dal config — usata nei banner.

    Priorità di ricerca:
      1. config/arch/{arch}.yaml  (se arch è specificato e il file esiste)
      2. config/default.yaml      (fallback)
    """
    import re

    # IT: estrae l'etichetta dal testo YAML (None se l'arch non è riconosciuta).
    # EN: extracts the label from YAML text (None if the arch is not recognized).
    def _parse_label(txt: str, arch_hint: str = None) -> str | None:
        """Estrae un'etichetta leggibile dal testo YAML. Restituisce None se non trova nulla."""
        m = re.search(r'architecture:\s*["\']?(\w+)["\']?', txt)
        a = m.group(1) if m else arch_hint
        if not a:
            return None
        d  = re.search(r'tft_d_model:\s*(\d+)', txt)
        dm = d.group(1) if d else "128"
        if a == "itransformer":
            l = re.search(r'tft_n_layers:\s*(\d+)', txt)
            return f"iTransformer (d={dm}, L={l.group(1) if l else '3'})"
        if a == "tft":
            return f"TFT (d_model={dm})"
        if a == "tcnmamba":
            ml = re.search(r'mamba_layers:\s*(\d+)', txt)
            return f"TCN+Mamba (d={dm}, M={ml.group(1) if ml else '3'})"
        if a == "lstm":
            return "LSTM+GRU"
        return a.upper()

    try:
        # 1. Prova config/arch/{arch}.yaml
        if arch:
            arch_cfg = ROOT / "config" / "arch" / f"{arch}.yaml"
            if arch_cfg.exists():
                label = _parse_label(arch_cfg.read_text(encoding="utf-8"), arch)
                if label:
                    return label
        # 2. Fallback su default.yaml
        default_cfg = ROOT / "config" / "default.yaml"
        if default_cfg.exists():
            label = _parse_label(default_cfg.read_text(encoding="utf-8"), arch)
            if label:
                return label
    except Exception:
        pass

    # Ultimo fallback: usa l'arch passato oppure LSTM+GRU
    if arch == "itransformer":
        return "iTransformer"
    if arch == "tcnmamba":
        return "TCN+Mamba"
    return "LSTM+GRU"


ARCH = _arch_label()

PHASE_WIDTH = 60


# IT: prompt interattivo per selezionare l'architettura (1/2/3).
# EN: interactive prompt to select the architecture (1/2/3).
def _ask_arch() -> str:
    """Chiede interattivamente quale architettura usare."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              QUANTSYS  ·  Scegli architettura                ║
╠══════════════════════════════════════════════════════════════╣
║  [1]  LSTM+GRU       — veloce, stabile, meno VRAM            ║
║  [2]  iTransformer   — dual-scale, attention O(F²), +VRAM    ║
║  [3]  TCN+Mamba      — pattern locali + contesto lungo SSM   ║
╚══════════════════════════════════════════════════════════════╝""")
    while True:
        choice = input("  Scelta [1/2/3]: ").strip().lower()
        if choice in ("1", "lstm"):
            return "lstm"
        if choice in ("2", "itransformer"):
            return "itransformer"
        if choice in ("3", "tcnmamba"):
            return "tcnmamba"
        print("  Inserisci 1, 2 oppure 3.")


# ─── Menu interattivo pipeline ──────────────────────────────────────────────

# IT: opzioni del menu checkbox: (label, attr flag, default on, descrizione).
# EN: checkbox menu options: (label, attr flag, default on, description).
_PIPELINE_OPTIONS = [
    # (label,                           attr,             default_on, desc)
    ("Aggiorna dati price",             "skip_update",    True,  "scarica candele mancanti"),
    ("Download macro FRED/yFinance",    "skip_macro",     True,  "dati macro per regime HMM"),
    ("Training modello",                "skip_train",     True,  "riaddestra da zero"),
    ("Walk-forward validation",         "skip_walkfwd",   True,  "valida su fold temporali"),
    ("Backtest",                        "skip_backtest",  True,  "simula trading storico"),
    ("Analisi segnali live",            "skip_analyze",   True,  "statistiche su live_signals"),
    ("Live feed WebSocket",             "skip_live",      True,  "segnali in tempo reale"),
    ("Apri browser",                    "no_browser",     True,  "apre dashboard nel browser"),
    ("Force download completo",         "force_download", False, "ri-scarica tutto da zero"),
]


# IT: legge un singolo tasto raw (frecce/spazio/invio) cross-platform.
# EN: reads a single raw keystroke (arrows/space/enter) cross-platform.
def _getch():
    """Legge un singolo tasto senza attendere Enter (Windows + Unix)."""
    if sys.platform == "win32":
        import msvcrt
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            ch = msvcrt.getch()
            return {b'H': 'UP', b'P': 'DOWN'}.get(ch, '')
        if ch == b' ':
            return 'SPACE'
        if ch in (b'\r', b'\n'):
            return 'ENTER'
        if ch == b'a':
            return 'A'
        return ch.decode(errors='ignore')
    else:
        import tty, termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                sys.stdin.read(1)  # skip [
                arrow = sys.stdin.read(1)
                return {'A': 'UP', 'B': 'DOWN'}.get(arrow, '')
            if ch == ' ':
                return 'SPACE'
            if ch in ('\r', '\n'):
                return 'ENTER'
            if ch in ('a', 'A'):
                return 'A'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


# IT: menu checkbox a schermo per scegliere le fasi; ritorna i flag {attr: bool}.
# EN: on-screen checkbox menu to pick phases; returns flags {attr: bool}.
def _interactive_pipeline_menu() -> dict:
    """Menu interattivo con checkbox per selezionare le fasi della pipeline.

    Restituisce un dict di flag {attr: bool_value} da applicare ad args.
    """
    states = [opt[2] for opt in _PIPELINE_OPTIONS]  # default on/off
    cursor = 0
    n = len(_PIPELINE_OPTIONS)

    # Abilita ANSI su Windows
    if sys.platform == "win32":
        os.system("")  # attiva VT100 processing

    UP    = "\033[A"
    CLEAR = "\033[K"

    # IT: ridisegna il menu in place usando escape ANSI (cursore su + clear).
    # EN: redraws the menu in place using ANSI escapes (cursor up + clear).
    def _render(first=False):
        if not first:
            sys.stdout.write(f"\033[{n + 4}A")  # risali

        print(f"{'─' * 62}")
        print(f"  ↑↓ naviga  ·  SPAZIO seleziona  ·  A tutto on/off  ·  INVIO conferma")
        print(f"{'─' * 62}")
        for i, (label, _, _, desc) in enumerate(_PIPELINE_OPTIONS):
            mark = "■" if states[i] else " "
            arrow = "►" if i == cursor else " "
            hi = "\033[96m" if i == cursor else ""
            rst = "\033[0m" if i == cursor else ""
            print(f"  {arrow} [{mark}] {hi}{label:<34}{rst} {desc}{CLEAR}")
        print(f"{'─' * 62}{CLEAR}")
        sys.stdout.flush()

    _render(first=True)

    while True:
        key = _getch()
        if key == 'UP':
            cursor = (cursor - 1) % n
        elif key == 'DOWN':
            cursor = (cursor + 1) % n
        elif key == 'SPACE':
            states[cursor] = not states[cursor]
        elif key == 'A':
            target = not all(states)
            states[:] = [target] * n
        elif key == 'ENTER':
            break
        _render()

    result = {}
    for i, (_, attr, _, _) in enumerate(_PIPELINE_OPTIONS):
        if attr == "force_download":
            result[attr] = states[i]
        elif attr == "no_browser":
            result[attr] = not states[i]
        else:
            result[attr] = not states[i]  # "skip_X" = NOT selected
    return result

# ─── Helpers ─────────────────────────────────────────────────────────────────

# IT: stampa un banner di fase racchiuso tra righe di separazione.
# EN: prints a phase banner wrapped in separator lines.
def banner(title: str, char: str = "═") -> None:
    line = char * PHASE_WIDTH
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# IT: log di step a livello ok/warn/err (warn ed err con icone dedicate).
# EN: step-level log helpers ok/warn/err (warn and err with dedicated icons).
def step_ok(msg: str) -> None:
    print(f"  ✔  {msg}")


def step_warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def step_err(msg: str) -> None:
    print(f"  ✖  {msg}", file=sys.stderr)


# IT: lancia scripts/<script> come sottoprocesso, output live; esce se fatal e fallisce.
# EN: runs scripts/<script> as a subprocess with live output; exits if fatal and it fails.
def run_script(script: str, extra_args: list[str] = None, fatal: bool = True) -> bool:
    """
    Lancia scripts/<script> come sottoprocesso.
    L'output viene stampato in tempo reale (nessun buffering).
    Restituisce True se il processo termina con exit code 0.
    """
    cmd = [PYTHON, str(ROOT / "scripts" / script)] + (extra_args or [])
    print(f"\n  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        step_err(f"{script} terminato con errore (exit {result.returncode})")
        if fatal:
            print("\n  Pipeline interrotta. Correggi l'errore e riprova.")
            sys.exit(result.returncode)
        return False
    return True


# IT: avvia scripts/<script> in background (Popen) senza bloccare il main.
# EN: starts scripts/<script> in the background (Popen) without blocking main.
def start_background(script: str, extra_args: list[str] = None) -> subprocess.Popen:
    """Avvia scripts/<script> in background senza bloccare."""
    cmd = [PYTHON, str(ROOT / "scripts" / script)] + (extra_args or [])
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    step_ok(f"{script} avviato in background (PID {proc.pid})")
    return proc


# IT: formatta il tempo trascorso da t0 come "Xm SSs".
# EN: formats elapsed time since t0 as "Xm SSs".
def elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s//60}m{s%60:02d}s"


# ─── Fasi pipeline ───────────────────────────────────────────────────────────

# IT: Fase 1 dati price: download completo se manca il parquet, altrimenti delta.
# EN: Phase 1 price data: full download if parquet missing, else incremental delta.
def phase_data(args) -> None:
    banner("FASE 1 · DATI  (price)")
    t0 = time.time()

    if args.force_download or not RAW_CANDLES.exists():
        if args.force_download:
            step_warn("--force-download: avvio download completo da zero")
        else:
            step_warn(f"raw_candles.parquet non trovato → download completo")
        run_script("01_download_data.py")
    else:
        step_ok(f"raw_candles.parquet trovato ({RAW_CANDLES.stat().st_size//1024//1024} MB)")
        run_script("01_update_data.py")

    step_ok(f"Dati completati in {elapsed(t0)}")


# IT: Fase 1b dati macro (FRED + yFinance); non bloccante se fallisce.
# EN: Phase 1b macro data (FRED + yFinance); non-blocking if it fails.
def phase_macro(args) -> None:
    banner("FASE 1b · DATI  (macro FRED + yFinance)")
    t0 = time.time()
    ok = run_script("01b_download_macro.py", fatal=False)
    if ok:
        step_ok(f"Macro completati in {elapsed(t0)}")
    else:
        step_warn("Download macro fallito — continuo senza. Il modello userà solo price features.")


# IT: Fase 2 training singola arch (riaddestra anche se esiste un modello).
# EN: Phase 2 single-arch training (retrains even if a model already exists).
def phase_train(args) -> None:
    banner(f"FASE 2 · TRAINING  {ARCH}  (t-Student NLL)")
    t0 = time.time()

    if MODEL_FILE.exists() and not args.force_download:
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(MODEL_FILE.stat().st_mtime))
        step_warn(f"Modello esistente: {MODEL_FILE.name}  (ultimo salvataggio: {mtime})")
        step_warn("Riaddestro su nuovo dataset ...")

    # IT: training single-arch (--arch) → ensemble multi-seed (default 5, override via --n-ensemble).
    #     Il path --distill NON passa di qui: usa phase_distill con n_ensemble=1.
    # EN: single-arch training (--arch) → multi-seed ensemble (default 5, override via --n-ensemble).
    #     The --distill path does NOT go through here: it uses phase_distill with n_ensemble=1.
    _n_ens = getattr(args, "n_ensemble", 5)
    print(f"  •  Ensemble single-arch: n_ensemble={_n_ens} seed")
    run_script("02_train.py", extra_args=["--n-ensemble", str(_n_ens)])
    step_ok(f"Training completato in {elapsed(t0)}")


# IT: sceglie il teacher con score pesato (val_loss 0.4 + spearman 0.35 + dir_acc 0.25).
# EN: picks the teacher by weighted score (val_loss 0.4 + spearman 0.35 + dir_acc 0.25).
def _select_best_teacher(all_archs: list) -> str:
    """Confronta le metriche dei modelli appena addestrati e sceglie il migliore come teacher.

    Usa i valori alla best val_loss epoch (non il picco su tutte le epoche)
    e normalizza i contributi per evitare che un singolo criterio domini.
    """
    import json as _json

    raw = {}
    for arch in all_archs:
        arch_dir = ROOT / "models" / arch
        history_path = arch_dir / "history.json"

        best_val, sp_at_best, da_at_best = None, 0.0, 0.0
        if history_path.exists():
            with open(history_path, encoding="utf-8") as f:
                history = _json.load(f)
            val_losses = history.get("val_nll", [])
            spearman = history.get("val_spearman", [])
            da = history.get("val_dir_acc", [])
            if val_losses:
                best_epoch = min(range(len(val_losses)), key=lambda i: val_losses[i])
                best_val = val_losses[best_epoch]
                if spearman and best_epoch < len(spearman):
                    sp_at_best = spearman[best_epoch]
                if da and best_epoch < len(da):
                    da_at_best = da[best_epoch]

        raw[arch] = {"val_loss": best_val, "spearman": sp_at_best, "da": da_at_best}

    archs_with_data = [a for a in all_archs if raw[a]["val_loss"] is not None]
    if not archs_with_data:
        return all_archs[0]

    vl_vals = [raw[a]["val_loss"] for a in archs_with_data]
    sp_vals = [raw[a]["spearman"] for a in archs_with_data]
    da_vals = [raw[a]["da"] for a in archs_with_data]

    # IT: min-max normalize in [0,1] (range 1.0 se i valori sono costanti).
    # EN: min-max normalize to [0,1] (range 1.0 if values are constant).
    def _norm(vals):
        lo, hi = min(vals), max(vals)
        rng = hi - lo if hi - lo > 1e-12 else 1.0
        return [(v - lo) / rng for v in vals]

    vl_norm = _norm(vl_vals)
    sp_norm = _norm(sp_vals)
    da_norm = _norm(da_vals)

    scores = {}
    for i, arch in enumerate(archs_with_data):
        score = (1.0 - vl_norm[i]) * 0.4 + sp_norm[i] * 0.35 + da_norm[i] * 0.25
        scores[arch] = score
        print(f"    {arch:<18} score={score:.3f}  "
              f"(vl={raw[arch]['val_loss']:+.4f}  sp={raw[arch]['spearman']:+.4f}  "
              f"da={raw[arch]['da']:.3f})")

    best = max(scores, key=scores.get)
    return best


# IT: Fase 2 distillation: addestra candidati → sceglie teacher → riaddestra student.
# EN: Phase 2 distillation: train candidates → pick teacher → retrain students.
def phase_distill(args) -> str:
    """Pipeline completa di knowledge distillation:
    1. Addestra tutti i modelli normalmente (n_ensemble=1) — lista da config
    2. Confronta le metriche → sceglie il teacher automaticamente
    3. Riaddestra i perdenti come student con distillation

    Returns: nome dell'architettura teacher selezionata dinamicamente.

    La composizione dell'ensemble si configura in `config/default.yaml` →
    `distillation.archs` (singolo punto di modifica per swap LSTM↔N-HiTS o
    aggiungere un 4° modello).
    """
    from quantsys.model.ensemble import get_distillation_archs
    try:
        with open(ROOT / "config" / "default.yaml", encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f) or {}
    except Exception:
        _cfg = {}
    all_archs = get_distillation_archs(_cfg)
    t0_total = time.time()

    # ── Fase 2a: Addestra tutti i modelli configurati ─────────────────────
    banner(f"FASE 2a · TRAINING CANDIDATI ({len(all_archs)} arch: "
           f"{', '.join(a.upper() for a in all_archs)})")
    for arch in all_archs:
        arch_dir = ROOT / "models" / arch
        arch_dir.mkdir(parents=True, exist_ok=True)
        model_file = arch_dir / "best_model.pt"

        if model_file.exists() and not args.force_download:
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(model_file.stat().st_mtime))
            step_warn(f"  {arch}: modello esistente ({mtime}) — skip")
            continue

        banner(f"  TRAINING {arch.upper()}")
        t0 = time.time()
        os.environ["QUANTSYS_ARCH"] = arch
        run_script("02_train.py", extra_args=["--n-ensemble", "1"])
        step_ok(f"  {arch} addestrato in {elapsed(t0)}")

    # ── Fase 2b: Scoring e pesatura teacher ─────────────────────────────────
    banner("FASE 2b · MULTI-TEACHER SCORING (confronto metriche)")
    teacher = _select_best_teacher(all_archs)
    student_archs = [a for a in all_archs if a != teacher]
    step_ok(f"  BEST teacher: {teacher.upper()}")
    step_ok(f"  STUDENT da riaddestrare: {', '.join(a.upper() for a in student_archs)}")
    step_ok(f"  Multi-teacher: tutti e 3 i modelli contribuiscono come teacher (pesati per qualità)")

    # ── Fase 2c: Riaddestra gli student con multi-teacher distillation ─────
    for student_arch in student_archs:
        _s_cfg = ROOT / "models" / student_arch / "config.json"
        if _s_cfg.exists() and not args.force_download:
            import json as _json_s
            with open(_s_cfg, encoding="utf-8") as _f_s:
                _sc = _json_s.load(_f_s)
            if _sc.get("distilled") and _sc.get("teacher_arch") == "multi-teacher":
                step_warn(f"  {student_arch}: gia' distillato (multi-teacher) — skip")
                continue

        banner(f"FASE 2c · MULTI-TEACHER DISTILLATION {student_arch.upper()}")
        t0_s = time.time()

        os.environ["QUANTSYS_ARCH"] = student_arch
        run_script("02_train.py", extra_args=[
            "--distill",
            "--teacher", teacher,
            "--multi-teacher",
            "--n-ensemble", "1",
        ])
        step_ok(f"  {student_arch} (student) addestrato in {elapsed(t0_s)}")

    step_ok(f"Pipeline multi-teacher distillation completata in {elapsed(t0_total)}")
    step_ok(f"  Best: {teacher.upper()}  |  Student: {', '.join(a.upper() for a in student_archs)}")
    step_ok(f"  Ensemble eterogeneo: {' + '.join(a.upper() for a in all_archs)}")
    return teacher


# IT: Fase 2b walk-forward su fold temporali (no retrain); non bloccante.
# EN: Phase 2b walk-forward over temporal folds (no retrain); non-blocking.
def phase_walkforward(args) -> None:
    banner("FASE 2b · WALK-FORWARD VALIDATION  (no retrain)")
    t0 = time.time()
    ok = run_script("02b_walkforward_validate.py", extra_args=["--no-retrain"], fatal=False)
    if ok:
        step_ok(f"Walk-forward completato in {elapsed(t0)}")
    else:
        step_warn("Walk-forward fallito — ignorato, non blocca la pipeline")


# IT: Fase 3 backtest storico + export di dashboard_results.json.
# EN: Phase 3 historical backtest + export of dashboard_results.json.
def phase_backtest(args) -> None:
    banner("FASE 3 · BACKTEST  + export dashboard_results.json")
    t0 = time.time()
    run_script("03_backtest.py")
    step_ok(f"Backtest completato in {elapsed(t0)}")


# IT: Fase 4b analisi segnali live: fonde le statistiche live nei risultati dashboard.
# EN: Phase 4b live-signal analysis: merges live stats into the dashboard results.
def phase_analyze(args) -> None:
    """
    Legge live_signals.jsonl, calcola statistiche e fonde i dati live
    dentro dashboard_results.json. Viene eseguito una volta subito e poi
    ogni ANALYZE_INTERVAL secondi in background mentre la dashboard è aperta.
    Non blocca la pipeline se fallisce.
    """
    live_path = ARCH_RESULTS_DIR / "live_signals.jsonl"
    if not live_path.exists():
        step_warn("live_signals.jsonl non trovato — 05_analyze_signals saltato (nessun dato live ancora)")
        return
    banner("FASE 4b · ANALISI SEGNALI LIVE")
    t0 = time.time()
    ok = run_script("05_analyze_signals.py", fatal=False)
    if ok:
        step_ok(f"Analisi segnali completata in {elapsed(t0)}")
    else:
        step_warn("05_analyze_signals.py terminato con warning — continuo")


# IT: intervallo di ri-analisi in background | EN: background re-analysis interval
ANALYZE_INTERVAL = 300  # ri-esegui analisi ogni 5 minuti mentre dashboard è aperta


# IT: thread daemon che ri-esegue 05_analyze_signals ogni ANALYZE_INTERVAL secondi.
# EN: daemon thread re-running 05_analyze_signals every ANALYZE_INTERVAL seconds.
def _analyze_loop(args) -> None:
    """Thread daemon: ri-esegue 05_analyze_signals ogni ANALYZE_INTERVAL secondi."""
    while True:
        time.sleep(ANALYZE_INTERVAL)
        live_path = ARCH_RESULTS_DIR / "live_signals.jsonl"
        if live_path.exists():
            subprocess.run(
                [PYTHON, str(ROOT / "scripts" / "05_analyze_signals.py")],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


# IT: Fase 4 feed segnali live in background (04_live_signals.py).
# EN: Phase 4 live signal feed in the background (04_live_signals.py).
def phase_live(args) -> subprocess.Popen:
    banner("FASE 4 · LIVE SIGNAL FEED  (background)")
    proc = start_background("04_live_signals.py")
    step_ok(f"Feed live attivo — scrive su results/{args.arch}/live_signals.jsonl")
    return proc


# IT: Fase 5 dashboard: avvia il server, apre il browser, fa watchdog fino a Ctrl+C.
# EN: Phase 5 dashboard: starts the server, opens the browser, watchdogs until Ctrl+C.
def phase_dashboard(args, live_proc: subprocess.Popen = None) -> None:
    banner("FASE 5 · DASHBOARD")

    # Avvia il server dashboard come sottoprocesso separato
    dash_proc = start_background("06_dashboard.py")
    step_ok(f"Dashboard server avviato (PID {dash_proc.pid})")

    if not args.no_browser:
        time.sleep(1.2)   # lascia partire il server
        webbrowser.open(DASHBOARD_URL)
        step_ok("Browser aperto")

    print(f"""
  Dashboard attiva → {DASHBOARD_URL}
  PID dashboard   : {dash_proc.pid}

  Tab disponibili:
    · Backtest     — equity curve, metriche, trade table
    · Live Segnali — segnali in tempo reale (aggiornamento ogni 5s)
    · Equity Curve — P&L completo e distribuzione

  Usa il pulsante "⚡ Aggiorna" nella dashboard per rieseguire la pipeline ML.
  Premi  Ctrl+C  per fermare tutto.
""")

    try:
        while True:
            time.sleep(2)
            # Riavvia live feed se crashato
            if live_proc and live_proc.poll() is not None:
                step_warn(f"Live feed terminato (exit {live_proc.returncode}) — riavvio ...")
                live_proc = start_background("04_live_signals.py")
            # Controlla che la dashboard non sia crashata
            if dash_proc.poll() is not None:
                step_err(f"Dashboard server terminato (exit {dash_proc.returncode}).")
                break
    except KeyboardInterrupt:
        print("\n\n  Arresto in corso ...")
        dash_proc.terminate()
        try:
            dash_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            dash_proc.kill()
        if live_proc:
            live_proc.terminate()
            try:
                live_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                live_proc.kill()
        step_ok("Tutto fermato. Arrivederci.")


# ─── Main ─────────────────────────────────────────────────────────────────────

# IT: definisce e parsa i flag CLI (arch, skip-*, distill, teacher, ...).
# EN: defines and parses CLI flags (arch, skip-*, distill, teacher, ...).
def parse_args():
    p = argparse.ArgumentParser(
        description="QUANTSYS — pipeline completa: update → train → backtest → live → dashboard"
    )
    p.add_argument(
        "--arch",
        default=_default_arch(),
        choices=["lstm", "itransformer", "tcnmamba", "nhits"],
        help="Architettura modello (default: valore in config/default.yaml)",
    )
    # IT: n_ensemble per il training single-arch (--arch). Default 5 (ensemble multi-seed).
    #     Il path --distill resta a 1 (vedi phase_distill: teacher e student a n_ensemble=1).
    # EN: n_ensemble for single-arch training (--arch). Default 5 (multi-seed ensemble).
    #     The --distill path stays at 1 (see phase_distill: teacher and student at n_ensemble=1).
    p.add_argument(
        "--n-ensemble",
        type=int,
        default=5,
        metavar="N",
        help="seed nell'ensemble per il training single-arch (--arch); default 5. --distill resta a 1",
    )
    p.add_argument("--skip-update",    action="store_true", help="salta aggiornamento dati")
    p.add_argument("--skip-macro",     action="store_true", help="salta download macro")
    p.add_argument("--skip-train",     action="store_true", help="salta riaddestramento")
    p.add_argument("--skip-backtest",  action="store_true", help="salta backtest")
    p.add_argument("--skip-walkfwd",   action="store_true", help="salta walk-forward validation")
    p.add_argument("--skip-analyze",   action="store_true", help="salta analisi segnali live")
    p.add_argument("--skip-live",      action="store_true", help="non avvia feed live")
    p.add_argument("--no-browser",     action="store_true", help="non aprire il browser")
    p.add_argument("--only-dashboard", action="store_true", help="apri solo la dashboard")
    p.add_argument("--force-download", action="store_true", help="forza download completo da zero")
    p.add_argument(
        "--max-model-age-days",
        type=int,
        default=None,
        metavar="N",
        help="forza retrain se best_model.pt è più vecchio di N giorni"
    )
    p.add_argument(
        "--distill",
        action="store_true",
        help="Pipeline distillation: train teacher, poi student con knowledge distillation"
    )
    p.add_argument(
        "--teacher",
        type=str,
        default="itransformer",
        choices=["lstm", "itransformer", "tcnmamba", "nhits"],
        help="Architettura teacher per distillation (default: itransformer)"
    )
    return p.parse_args()


# IT: orchestratore: menu/flag → propaga arch via env → esegue le fasi in ordine.
# EN: orchestrator: menu/flags → propagate arch via env → run phases in order.
def main():
    # IT: Forza UTF-8 su stdout/stderr PRIMA di parse_args — l'help/usage e i banner contengono
    #     Unicode (frecce, accenti, icone) che sotto Windows cp1252 crashano con UnicodeEncodeError
    #     quando l'output è rediretto/in pipe (es. `run_all.py --help | ...`). Stesso pattern di
    #     04_live_signals.py / 99_replay_live_vs_training.py.
    # EN: Force UTF-8 on stdout/stderr BEFORE parse_args — help/usage and banners contain Unicode
    #     that crashes under Windows cp1252 when output is redirected/piped. Same pattern as the
    #     live engine and the replay script.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args()

    # ── Modalità interattiva: nessun flag CLI → mostra menu ─────────────────
    _has_flags = any(a.startswith("--") for a in sys.argv[1:])
    if not _has_flags:
        args.arch = _ask_arch()
        flags = _interactive_pipeline_menu()
        for attr, val in flags.items():
            setattr(args, attr, val)
    elif args.distill and "--arch" not in sys.argv:
        # Con --distill usa il teacher come arch principale (no prompt)
        args.arch = args.teacher
    elif "--arch" not in sys.argv:
        args.arch = _ask_arch()

    # ── Propaga arch e encoding a tutti i subprocess tramite env var ─────────
    os.environ["QUANTSYS_ARCH"]      = args.arch
    os.environ["PYTHONIOENCODING"]   = "utf-8"

    # ── Imposta path arch-dipendenti ─────────────────────────────────────────
    global ARCH, ARCH_MODELS_DIR, ARCH_RESULTS_DIR, MODEL_FILE
    ARCH_MODELS_DIR  = ROOT / "models"  / args.arch
    ARCH_RESULTS_DIR = ROOT / "results" / args.arch
    MODEL_FILE       = ARCH_MODELS_DIR  / "best_model.pt"

    # Aggiorna ARCH label in base all'arch effettivo scelto a runtime
    ARCH = _arch_label(args.arch)

    # Crea le directory arch se non esistono
    ARCH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   QUANTSYS  ·  BTC/USDT 1m  ·  {ARCH:<10}  ·  t-Student NLL  ║
╚══════════════════════════════════════════════════════════════╝""")

    if args.only_dashboard:
        live_proc = None
        if not args.skip_live:
            live_proc = phase_live(args)
        if not args.skip_analyze:
            phase_analyze(args)
            t_analyze = threading.Thread(target=_analyze_loop, args=(args,), daemon=True)
            t_analyze.start()
        phase_dashboard(args, live_proc)
        return

    # ── Fase 1: dati ─────────────────────────────────────────────────────────
    if not args.skip_update:
        phase_data(args)
    else:
        banner("FASE 1 · DATI  [SALTATA]", char="─")
        step_warn("Usando dataset esistente.")

    # ── Fase 1b: macro ───────────────────────────────────────────────────────
    if not args.skip_macro:
        phase_macro(args)
    else:
        banner("FASE 1b · MACRO  [SALTATA]", char="─")
        _npz = DATA_DIR / "lstm_dataset.npz"
        if _npz.exists():
            try:
                import numpy as _np
                with _np.load(_npz, allow_pickle=True) as _d:
                    if "X_macro_train" in _d.files:
                        _n = int(_d["n_macro_features"][0]) if "n_macro_features" in _d.files else _d["X_macro_train"].shape[1]
                        step_ok(f"Usando macro esistenti nel dataset ({_n} feature).")
                    else:
                        step_warn("Nessuna macro nel dataset — il modello userà solo price features. "
                                  "Esegui senza --skip-macro almeno una volta.")
            except Exception as _e:
                step_warn(f"Impossibile ispezionare lstm_dataset.npz ({_e}) — proseguo.")
        else:
            step_warn("lstm_dataset.npz non trovato — esegui prima la fase 1 senza --skip-update.")

    # ── Fase 2: training ─────────────────────────────────────────────────────
    # Auto-retrain: forza retrain se il modello è più vecchio di max_model_age_days
    if args.max_model_age_days is not None and MODEL_FILE.exists():
        age_days = (time.time() - MODEL_FILE.stat().st_mtime) / 86400
        if age_days > args.max_model_age_days:
            step_warn(
                f"Modello aggiornato {age_days:.1f} giorni fa "
                f"(soglia: {args.max_model_age_days}g) — retrain automatico"
            )
            args.skip_train   = False
            args.skip_walkfwd = False

    if not args.skip_train:
        if args.distill:
            selected_teacher = phase_distill(args)
            # Allinea args.arch e path al teacher effettivamente selezionato dal distill
            args.arch = selected_teacher
            os.environ["QUANTSYS_ARCH"] = args.arch
            ARCH_MODELS_DIR  = ROOT / "models"  / args.arch
            ARCH_RESULTS_DIR = ROOT / "results" / args.arch
            MODEL_FILE       = ARCH_MODELS_DIR  / "best_model.pt"
        else:
            phase_train(args)
    else:
        banner("FASE 2 · TRAINING  [SALTATA]", char="─")
        if not MODEL_FILE.exists():
            step_err(f"Nessun modello trovato in {MODEL_FILE}. Rimuovi --skip-train.")
            sys.exit(1)
        step_warn(f"Usando modello esistente: {MODEL_FILE.name}")

    # ── Fase 2b: walk-forward ─────────────────────────────────────────────────
    if not args.skip_walkfwd:
        phase_walkforward(args)
    else:
        banner("FASE 2b · WALK-FORWARD  [SALTATA]", char="─")

    # ── Fase 3: backtest ─────────────────────────────────────────────────────
    if not args.skip_backtest:
        # IT: Dopo --distill esegue un backtest STANDALONE per ogni arch
        #     addestrata (teacher + student) così la tab "Confronto Archs" della
        #     dashboard mostra metriche per-arch comparabili. Senza --distill,
        #     un singolo backtest sull'arch corrente.
        # EN: After --distill runs a STANDALONE backtest per trained arch
        #     (teacher + students) so the dashboard "Compare Archs" tab shows
        #     comparable per-arch metrics. Without --distill, a single backtest
        #     on the current arch.
        if args.distill:
            from quantsys.model.ensemble import get_distillation_archs
            with open(ROOT / "config" / "default.yaml", encoding="utf-8") as _f:
                _bt_cfg = _yaml.safe_load(_f) or {}
            _distill_archs = get_distillation_archs(_bt_cfg)
            os.environ["QUANTSYS_BACKTEST_SINGLE_ARCH"] = "1"
            try:
                for _arch in _distill_archs:
                    banner(f"FASE 3 · BACKTEST {_arch.upper()} (single-arch)")
                    os.environ["QUANTSYS_ARCH"] = _arch
                    phase_backtest(args)
            finally:
                # IT: ripristina env per le fasi successive (live/dashboard sul teacher).
                # EN: restore env for downstream phases (live/dashboard on teacher).
                os.environ.pop("QUANTSYS_BACKTEST_SINGLE_ARCH", None)
                os.environ["QUANTSYS_ARCH"] = args.arch  # teacher selezionato dal distill
        else:
            phase_backtest(args)
    else:
        banner("FASE 3 · BACKTEST  [SALTATA]", char="─")
        results_file = ARCH_RESULTS_DIR / "dashboard_results.json"
        if results_file.exists():
            step_warn("Usando risultati backtest esistenti.")
        else:
            step_warn("Nessun risultato backtest trovato — la dashboard non mostrerà metriche.")

    # ── Fase 4: live feed ────────────────────────────────────────────────────
    live_proc = None
    if not args.skip_live:
        live_proc = phase_live(args)
    else:
        banner("FASE 4 · LIVE FEED  [SALTATO]", char="─")

    # ── Fase 4b: analisi segnali (sincrona, poi loop in background) ──────────
    if not args.skip_analyze:
        phase_analyze(args)
        # Avvia thread daemon che aggiorna l'analisi ogni 5 min
        t_analyze = threading.Thread(target=_analyze_loop, args=(args,), daemon=True)
        t_analyze.start()
        step_ok(f"Analisi automatica ogni {ANALYZE_INTERVAL//60} min avviata in background")
    else:
        banner("FASE 4b · ANALISI SEGNALI  [SALTATA]", char="─")

    banner(f"PIPELINE COMPLETATA in {elapsed(t_start)}", char="═")
    print(f"  Arch     : {args.arch}  ({ARCH})")
    print(f"  Dati     : {DATA_DIR}")
    print(f"  Modello  : {ARCH_MODELS_DIR}")
    print(f"  Risultati: {ARCH_RESULTS_DIR}")

    # ── Fase 5: dashboard (bloccante fino a Ctrl+C) ──────────────────────────
    phase_dashboard(args, live_proc)


if __name__ == "__main__":
    main()
