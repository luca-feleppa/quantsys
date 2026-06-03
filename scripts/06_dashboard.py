"""
Dashboard QUANTSYS — server HTTP single-file.
Esegui: python scripts/06_dashboard.py
Apri:   http://localhost:8050
"""
import collections
import gzip as _gzip
import hmac
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

try:
    from quantsys.utils import setup_logging, load_config
    setup_logging()
    _CFG = load_config("config/default.yaml")
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    _CFG = {}
log = logging.getLogger("quantsys.dashboard")

# IT: import pandas a livello modulo (evita race condition in threading)
# EN: module-level pandas import (avoids threading race conditions)
try:
    import pandas as _pd
except Exception:
    _pd = None
# IT: serializza letture parquet (pandas non sempre thread-safe)
# EN: serializes parquet reads (pandas not always thread-safe)
_data_lock = threading.Lock()

_DCFG = _CFG.get("dashboard", {}) if isinstance(_CFG, dict) else {}


# IT: legge architettura corrente da config (fallback su lstm)
# EN: read current architecture from config (fallback to lstm)
def _default_arch() -> str:
    try:
        _root = Path(__file__).parent.parent
        txt = (_root / "config" / "default.yaml").read_text(encoding="utf-8")
        m = re.search(r'architecture:\s*["\']?(\w+)["\']?', txt)
        if m and m.group(1) in ("lstm", "itransformer", "tft", "tcnmamba", "nhits"):
            return m.group(1)
    except Exception:
        pass
    return "lstm"


ARCH = os.environ.get("QUANTSYS_ARCH") or _default_arch()
# IT: re-export env per i sottoprocessi (pipeline /api/run)
# EN: re-export env for subprocesses (pipeline /api/run)
os.environ["QUANTSYS_ARCH"] = ARCH

HOST          = _DCFG.get("host", "127.0.0.1")
PORT          = int(_DCFG.get("port", 8050))
AUTH_TOKEN    = str(_DCFG.get("auth_token", "") or "")
SUBPROC_TIMEOUT = int(_DCFG.get("subprocess_timeout_sec", 7200))
ENABLE_GZIP   = bool(_DCFG.get("enable_gzip", True))
LOG_LINES_MAX = int(_DCFG.get("log_lines_maxlen", 500))
ROOT          = Path(__file__).parent.parent
MODELS_DIR    = ROOT / "models" / ARCH
BACKTEST_FILE = ROOT / "results" / ARCH / "dashboard_results.json"
LIVE_FILE     = ROOT / "results" / ARCH / "live_signals.jsonl"
LIVE_TAIL     = 100   # IT: ultime N righe in /api/live | EN: last N rows in /api/live


# IT: risolve i path arch-specifici sanificando l'input (evita path traversal)
# EN: resolve arch-specific paths with input sanitization (prevents path traversal)
def _arch_paths(arch: str):
    """Restituisce (BACKTEST_FILE, LIVE_FILE) per l'architettura indicata."""
    safe = re.sub(r"[^a-z0-9_]", "", (arch or "").lower())
    if safe not in ("lstm", "itransformer", "tcnmamba", "tft", "nhits"):
        safe = ARCH
    return (ROOT / "results" / safe / "dashboard_results.json",
            ROOT / "results" / safe / "live_signals.jsonl")


# IT: label leggibile arch+config (legge da config.json del modello salvato)
# EN: readable arch+config label (reads from saved model config.json)
def _model_arch_label() -> str:
    """Legge l'architettura dal config.json del modello salvato (fallback: config yaml)."""
    try:
        cfg_path = MODELS_DIR / "config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                c = json.load(f)
            arch = c.get("architecture", c.get("model_type", "lstm")).lower()
            dm   = c.get("tft_d_model", 128)
            if "itransformer" in arch or arch == "quantitransformer":
                return f"iTransformer d={dm} L={c.get('tft_n_layers', 3)}"
            if "tcnmamba" in arch or arch == "quanttcnmamba":
                return f"TCN+Mamba d={c.get('d_model', 128)}"
            if "nhits" in arch or arch == "quantnhits":
                return (f"N-HiTS d={c.get('d_model', 128)} "
                        f"stacks={c.get('nhits_stacks', 3)}")
            if "tft" in arch:
                return f"TFT d={dm}"
            return "LSTM+GRU"
    except Exception:
        pass
    try:
        txt = (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
        m = re.search(r'architecture:\s*"(\w+)"', txt)
        if m:
            a = m.group(1)
            if a == "itransformer":
                return "iTransformer"
            if a == "tcnmamba":
                return "TCN+Mamba"
            if a == "nhits":
                return "N-HiTS"
            if a == "tft":
                return "TFT"
    except Exception:
        pass
    return "LSTM+GRU"


ARCH_LABEL = _model_arch_label()

# IT: mappa step pipeline -> (script + label) per il pulsante "Aggiorna"
# EN: pipeline step map -> (script + label) for the "Update" button

PIPELINE_STEPS = {
    "update":   ("01_update_data.py",                                       "Aggiornamento dati price"),
    "macro":    ("01b_download_macro.py",                                   "Download macro FRED/yFinance"),
    "train":    ("02_train.py",                                              f"Training {ARCH_LABEL}"),
    "distill":  ("02_train.py --distill --multi-teacher",                   f"Distillation multi-teacher → {ARCH_LABEL}"),
    "walkfwd":  ("02b_walkforward_validate.py --no-retrain",                "Walk-forward validation"),
    "backtest": ("03_backtest.py",                                           "Backtest + export risultati"),
    "analyze":  ("05_analyze_signals.py",                                    "Analisi segnali"),
}

# IT: stato globale del job pipeline (condiviso fra HTTP handler + worker thread)
# EN: global pipeline job state (shared between HTTP handler + worker thread)

_job_lock = threading.Lock()
_job = {
    "status": "idle",
    "step_label": "",
    "step_key": "",
    "step_idx": 0,
    "total_steps": 0,
    "started_at": None,
    "elapsed": 0.0,
    "error_msg": "",
    "log_lines": collections.deque(maxlen=LOG_LINES_MAX),
    "last_success_ts": None,
}
_process = None  # IT: subprocess.Popen corrente per /api/run DELETE | EN: current subprocess.Popen for /api/run DELETE


# IT: worker — esegue gli script della pipeline in sequenza con stream dell'output
# EN: worker — runs pipeline scripts sequentially streaming their output
def _run_job(steps: list):
    global _process
    scripts_dir = ROOT / "scripts"

    with _job_lock:
        _job["status"] = "running"
        _job["step_idx"] = 0
        _job["total_steps"] = len(steps)
        _job["started_at"] = time.time()
        _job["elapsed"] = 0.0
        _job["error_msg"] = ""
        _job["log_lines"].clear()

    for idx, key in enumerate(steps, start=1):
        script_cmd, label = PIPELINE_STEPS[key]
        parts = script_cmd.split()
        script_file = parts[0]
        extra_args = parts[1:]

        with _job_lock:
            if _job["status"] == "cancelled":
                return
            _job["step_idx"] = idx
            _job["step_key"] = key
            _job["step_label"] = label
            _job["log_lines"].append(f"[STEP {idx}/{len(steps)}] {label}")

        cmd = [sys.executable, str(scripts_dir / script_file)] + extra_args
        log.info(f"Avvio step {idx}/{len(steps)}: {' '.join(cmd)}")

        _sub_env = os.environ.copy()
        _sub_env["QUANTSYS_ARCH"] = ARCH
        _sub_env["PYTHONIOENCODING"] = "utf-8"

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                encoding="utf-8",
                errors="replace",
                env=_sub_env,
            )
        except Exception as e:
            with _job_lock:
                _job["status"] = "error"
                _job["error_msg"] = str(e)
                _job["elapsed"] = time.time() - _job["started_at"]
                _job["log_lines"].append(f"[ERRORE] {e}")
            return

        with _job_lock:
            _process = proc

        _step_start = time.time()
        for line in proc.stdout:
            line = line.rstrip("\n")
            with _job_lock:
                if _job["status"] == "cancelled":
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
                _job["log_lines"].append(line)
                _job["elapsed"] = time.time() - _job["started_at"]
            # IT: watchdog timeout — kill step se eccede SUBPROC_TIMEOUT
            # EN: watchdog timeout — kill step if it exceeds SUBPROC_TIMEOUT
            if SUBPROC_TIMEOUT > 0 and (time.time() - _step_start) > SUBPROC_TIMEOUT:
                with _job_lock:
                    _job["log_lines"].append(
                        f"[ERRORE] Timeout watchdog ({SUBPROC_TIMEOUT}s) — kill step '{key}'.")
                    _job["status"] = "error"
                    _job["error_msg"] = f"Step '{key}' superato timeout {SUBPROC_TIMEOUT}s"
                    _job["elapsed"] = time.time() - _job["started_at"]
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception:
                    pass
                return

        proc.wait()

        with _job_lock:
            _process = None
            if _job["status"] == "cancelled":
                return
            if proc.returncode != 0:
                _job["status"] = "error"
                _job["error_msg"] = f"Step '{key}' uscito con codice {proc.returncode}"
                _job["elapsed"] = time.time() - _job["started_at"]
                _job["log_lines"].append(f"[ERRORE] exit code {proc.returncode}")
                return

    with _job_lock:
        _job["status"] = "success"
        _job["elapsed"] = time.time() - _job["started_at"]
        _job["last_success_ts"] = time.time()
        _job["log_lines"].append("[OK] Pipeline completata con successo.")

    log.info("Pipeline completata con successo.")


# IT: SPA dashboard embeddata (HTML + CSS + JS in un'unica stringa)
# EN: embedded dashboard SPA (HTML + CSS + JS in a single string)

HTML = r"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QUANTSYS Dashboard</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230d1117'/%3E%3Cpath d='M5 22 L12 13 L17 18 L27 7' stroke='%2358a6ff' stroke-width='2.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='27' cy='7' r='2.5' fill='%233fb950'/%3E%3C/svg%3E">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff; --yellow: #d29922;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 18px; font-weight: 600; letter-spacing: .5px; }
  #live-price { font-size: 22px; font-weight: 700; color: var(--blue); }
  #live-badge { font-size: 11px; padding: 2px 8px; border-radius: 10px; background: #1f2d1f; color: var(--green); border: 1px solid var(--green); }
  .tabs { display: flex; gap: 4px; padding: 12px 24px 0; background: var(--surface); border-bottom: 1px solid var(--border); }
  .tab { padding: 8px 20px; border-radius: 6px 6px 0 0; cursor: pointer; color: var(--muted); border: 1px solid transparent; border-bottom: none; }
  .tab.active { color: var(--text); background: var(--bg); border-color: var(--border); }
  .page { display: none; padding: 20px 24px; }
  .page.active { display: block; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }
  .card-value { font-size: 24px; font-weight: 700; }
  .green { color: var(--green); } .red { color: var(--red); } .blue { color: var(--blue); } .yellow { color: var(--yellow); }
  .chart-box { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .chart-box h3 { font-size: 13px; color: var(--muted); margin-bottom: 12px; }
  .chart-box canvas { max-height: 260px; }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { color: var(--muted); font-weight: 500; text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
  td { padding: 6px 10px; border-bottom: 1px solid #1c2128; }
  tr:hover td { background: #1c2128; }
  .signal-feed { max-height: 380px; overflow-y: auto; }
  .sig-row { display: flex; align-items: center; gap: 10px; padding: 7px 12px; border-bottom: 1px solid #1c2128; font-size: 12px; }
  .sig-row:hover { background: #1c2128; }
  .sig-badge { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; min-width: 44px; text-align: center; }
  .BUY  { background: #1f3326; color: var(--green); }
  .SELL { background: #3c1f1f; color: var(--red); }
  .HOLD { background: #252525; color: var(--muted); }
  .sig-meta { color: var(--muted); flex: 1; }
  .sig-price { font-weight: 600; margin-left: auto; }
  #no-backtest { text-align: center; padding: 60px; color: var(--muted); }
  #no-live { text-align: center; padding: 60px; color: var(--muted); }
  .status-bar { font-size: 11px; color: var(--muted); margin-bottom: 10px; }

  /* Run controls */
  .run-btn { background:#1a3a5c; border:1px solid #58a6ff; color:#58a6ff; padding:6px 14px; border-radius:4px; cursor:pointer; font-size:12px; font-weight:600; display:inline-flex; align-items:center; gap:6px; }
  .run-btn:hover { background:#0f2b48; }
  .run-btn:disabled { opacity:0.5; cursor:default; }
  .run-btn.danger { background:#3d0f0f; border-color:#f85149; color:#f85149; }
  .run-btn.success { background:#0d3320; border-color:#3fb950; color:#3fb950; }

  /* Spinner */
  .spinner { width:12px; height:12px; border:2px solid #21262d; border-top-color:#58a6ff; border-radius:50%; animation:spin 0.8s linear infinite; flex-shrink:0; }
  @keyframes spin { to { transform:rotate(360deg); } }

  /* Run drawer (fixed bottom) */
  #run-drawer { position:fixed; bottom:0; left:0; right:0; background:#0d1117; border-top:2px solid #30363d; padding:12px 24px; z-index:100; transform:translateY(100%); transition:transform 0.25s; }
  #run-drawer.open { transform:translateY(0); }
  .drawer-header { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
  .drawer-title { font-size:13px; font-weight:600; color:#e6edf3; flex:1; }
  .drawer-meta { font-size:11px; color:#8b949e; font-family:monospace; }
  .progress-track { height:3px; background:#21262d; border-radius:2px; margin-bottom:8px; overflow:hidden; }
  .progress-fill { height:100%; background:linear-gradient(90deg,#58a6ff,#bc8cff); border-radius:2px; transition:width 0.5s ease; }
  #run-log { max-height:100px; overflow-y:auto; background:#080a0c; border:1px solid #21262d; padding:8px; font-family:monospace; font-size:10px; color:#8b949e; border-radius:4px; margin-top:6px; display:none; }
  #run-log.show { display:block; }
  .ll-ok { color:#3fb950; }
  .ll-err { color:#f85149; }
  .ll-info { color:#58a6ff; }

  /* Modal overlay */
  #run-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:200; align-items:center; justify-content:center; }
  #run-modal.open { display:flex; }
  .modal-box { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:24px; min-width:300px; max-width:420px; }
  .modal-title { font-size:14px; font-weight:600; margin-bottom:16px; color:#e6edf3; }
  .step-checks { display:flex; flex-direction:column; gap:8px; margin-bottom:16px; }
  .step-check { display:flex; align-items:center; gap:8px; font-size:12px; color:#8b949e; cursor:pointer; }
  .step-check input { accent-color:#58a6ff; }
  .step-check .step-label { flex:1; }
  .step-check .step-time { font-size:10px; color:#21262d; }
  .modal-actions { display:flex; gap:8px; }
</style>
</head>
<body>
<header>
  <div>
    <h1>&#9889; QUANTSYS</h1>
    <div style="font-size:11px;color:var(--muted)">BTC/USDT &middot; 1m &middot; {ARCH} &middot; t-Student NLL</div>
  </div>
  <select id="arch-select" style="font-size:11px;padding:3px 8px;border-radius:10px;background:#1a1f2e;color:#58a6ff;border:1px solid #58a6ff;font-weight:600;cursor:pointer" onchange="onArchChange()">
    <option value="{ARCH_BADGE}">{ARCH_BADGE}</option>
  </select>
  <span id="arch-loss-badge" title="loss_type · n_output_experts (MoE)" style="font-size:10px;padding:2px 8px;border-radius:10px;background:#0d1117;color:#8b949e;border:1px solid #30363d;font-weight:500"></span>
  <span id="live-badge">&#9679; LIVE</span>
  <span id="live-price">&mdash;</span>
  <span id="live-change" style="font-size:11px;padding:2px 8px;border-radius:10px;background:#1a1f2e;font-weight:600">&mdash;</span>
  <span id="live-volume" style="font-size:10px;color:var(--muted)" title="Volume 24h (base asset)">vol &mdash;</span>
  <div id="run-controls" style="margin-left:auto;display:flex;align-items:center;gap:10px;">
    <span id="last-run-label" style="font-size:10px;color:var(--muted);display:none"></span>
    <button class="run-btn" id="btn-run" onclick="openRunModal()">&#9889; Aggiorna</button>
  </div>
</header>

<!-- Run drawer (fixed bottom) -->
<div id="run-drawer">
  <div class="drawer-header">
    <div class="spinner" id="drawer-spinner" style="display:none"></div>
    <div class="drawer-title" id="drawer-title">Pipeline completata</div>
    <div class="drawer-meta" id="drawer-meta"></div>
    <button class="run-btn" onclick="toggleLog()" style="padding:3px 10px;font-size:10px;">Log</button>
    <button class="run-btn danger" id="btn-cancel" onclick="cancelRun()" style="display:none">&#10005; Annulla</button>
    <button class="run-btn" id="btn-close-drawer" onclick="closeDrawer()">Chiudi</button>
  </div>
  <div class="progress-track"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
  <div id="run-log"></div>
</div>

<!-- Run modal -->
<div id="run-modal">
  <div class="modal-box">
    <div class="modal-title">&#9889; Aggiorna Modello e Dati</div>
    <div class="step-checks">
      <label class="step-check"><input type="checkbox" id="s-update" checked><span class="step-label">Aggiorna dati price</span><span class="step-time">~1 min</span></label>
      <label class="step-check"><input type="checkbox" id="s-macro" checked><span class="step-label">Download macro FRED/yFinance</span><span class="step-time">~2 min</span></label>
      <label class="step-check"><input type="checkbox" id="s-train"><span class="step-label">Riaddestra LSTM</span><span class="step-time">~20 min</span></label>
      <label class="step-check"><input type="checkbox" id="s-distill" title="Richiede teacher già addestrati (tutti e 3 i modelli)"><span class="step-label">Knowledge distillation (multi-teacher)</span><span class="step-time">~15 min</span></label>
      <label class="step-check"><input type="checkbox" id="s-walkfwd"><span class="step-label">Walk-forward validation</span><span class="step-time">~10 min</span></label>
      <label class="step-check"><input type="checkbox" id="s-backtest" checked><span class="step-label">Backtest + export</span><span class="step-time">~2 min</span></label>
      <label class="step-check"><input type="checkbox" id="s-analyze"><span class="step-label">Analisi segnali (05)</span><span class="step-time">~1 min</span></label>
    </div>
    <div class="modal-actions">
      <button class="run-btn" onclick="startRun()">&#9654; Esegui</button>
      <button class="run-btn" onclick="closeRunModal()" style="background:var(--bg);border-color:var(--border);color:var(--muted)">Annulla</button>
    </div>
  </div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('backtest')">Backtest</div>
  <div class="tab" onclick="switchTab('live')">Live Segnali</div>
  <div class="tab" onclick="switchTab('macro')">Macro & Regime</div>
  <div class="tab" onclick="switchTab('equity')">Equity Curve</div>
  <div class="tab" onclick="switchTab('walkforward')">Walk-Forward</div>
  <div class="tab" onclick="switchTab('health')">Training Health</div>
  <div class="tab" onclick="switchTab('compare')">Confronto Archs</div>
  <div class="tab" onclick="switchTab('ensemble')">Ensemble</div>
</div>

<!-- ── BACKTEST ── -->
<div id="page-backtest" class="page active">
  <div id="no-backtest" style="display:none">
    <p>Nessun dato backtest trovato.</p>
    <p style="margin-top:8px">Esegui prima: <code>python scripts/03_backtest.py</code></p>
  </div>
  <div id="backtest-content">
    <div class="cards" id="metric-cards"></div>
    <div class="row2">
      <div class="chart-box"><h3>Equity Curve</h3><canvas id="chart-equity"></canvas></div>
      <div class="chart-box"><h3>Drawdown</h3><canvas id="chart-dd"></canvas></div>
    </div>
    <div class="chart-box"><h3>P&amp;L per Trade</h3><canvas id="chart-pnl"></canvas></div>
    <div class="chart-box">
      <h3 style="display:flex;align-items:center;justify-content:space-between">
        Ultimi trade
        <button onclick="exportTradesCSV()" style="font-size:11px;padding:4px 12px;border-radius:4px;background:#1a1f2e;color:#58a6ff;border:1px solid #58a6ff;cursor:pointer">⬇ Export CSV</button>
      </h3>
      <table>
        <thead><tr><th>#</th><th>Side</th><th>Entry</th><th>Exit</th><th>Size $</th><th>Hold</th><th>P&amp;L %</th></tr></thead>
        <tbody id="trade-table"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- ── LIVE ── -->
<div id="page-live" class="page">
  <div id="no-live" style="display:none">
    <p>Nessun segnale live trovato.</p>
    <p style="margin-top:8px">Esegui: <code>python scripts/04_live_signals.py</code></p>
  </div>
  <div id="live-content">
    <div class="cards" id="live-cards"></div>
    <div class="row2">
      <div class="chart-box">
        <h3>Segnali recenti</h3>
        <div class="signal-feed" id="signal-feed"></div>
      </div>
      <div class="chart-box"><h3>Equity live</h3><canvas id="chart-live-eq"></canvas></div>
    </div>
    <div class="chart-box"><h3>Prob_Up (ultimi 100)</h3><canvas id="chart-prob"></canvas></div>
  </div>
</div>

<!-- ── MACRO & REGIME ── -->
<div id="page-macro" class="page">
  <div class="cards" id="macro-cards"></div>
  <div class="row2">
    <div class="chart-box"><h3>Regime dominante (storico)</h3><canvas id="chart-regime-dom"></canvas></div>
    <div class="chart-box"><h3>Probabilità regimi (stacked)</h3><canvas id="chart-regime-prob"></canvas></div>
  </div>
  <div class="chart-box"><h3>Funding rate BTC perpetuo (8h)</h3><canvas id="chart-funding"></canvas></div>

  <!-- A11: performance per regime (da live_signals.jsonl + regime_probs) -->
  <div class="chart-box">
    <h3>Performance per regime (live signals)</h3>
    <p style="color:var(--muted);font-size:12px;margin:4px 0 12px">
      Aggrega segnali live joinati con regime_dominant via merge_asof.
    </p>
    <table>
      <thead>
        <tr><th>Regime</th><th>N</th><th>LONG</th><th>SHORT</th><th>FLAT</th><th>Prob_Up μ</th><th>σ μ</th></tr>
      </thead>
      <tbody id="regime-perf-table"><tr><td colspan="7" style="text-align:center;color:var(--muted)">…</td></tr></tbody>
    </table>
  </div>

  <div id="no-macro" style="display:none;text-align:center;padding:40px;color:var(--muted)">
    <p>Dati macro non disponibili.</p>
    <p style="margin-top:8px">Esegui: <code>python scripts/01b_download_macro.py</code></p>
  </div>
</div>

<!-- ── EQUITY ── -->
<div id="page-equity" class="page">
  <div class="status-bar" id="eq-status"></div>
  <div class="chart-box"><h3>Equity Curve completa <span style="font-size:11px;color:var(--muted);font-weight:normal">(con buy&amp;hold overlay)</span></h3><canvas id="chart-eq-full" style="max-height:400px"></canvas></div>
  <div class="row2">
    <div class="chart-box"><h3>P&amp;L serie storica</h3><canvas id="chart-pnl-series"></canvas></div>
    <div class="chart-box"><h3>Distribuzione P&amp;L</h3><canvas id="chart-pnl-hist"></canvas></div>
  </div>
  <!-- A12: rolling Sharpe + DD -->
  <div class="row2">
    <div class="chart-box"><h3>Sharpe rolling (window 30 trade)</h3><canvas id="chart-rolling-sharpe"></canvas></div>
    <div class="chart-box"><h3>Drawdown rolling</h3><canvas id="chart-rolling-dd"></canvas></div>
  </div>
</div>

<!-- ── WALK-FORWARD (A6) ── -->
<div id="page-walkforward" class="page">
  <div class="status-bar" id="wf-status">Caricamento walk-forward…</div>
  <div class="cards" id="wf-cards"></div>
  <div class="chart-box">
    <h3>Fold-by-fold metrics</h3>
    <table>
      <thead><tr><th>Fold</th><th>N samples</th><th>DA</th><th>Spearman</th><th>WHR</th><th>CI90 coverage</th><th>Val NLL</th><th>Time (s)</th></tr></thead>
      <tbody id="wf-table"></tbody>
    </table>
  </div>
  <div class="row2">
    <div class="chart-box"><h3>DA per fold</h3><canvas id="chart-wf-da"></canvas></div>
    <div class="chart-box"><h3>Spearman per fold</h3><canvas id="chart-wf-sp"></canvas></div>
  </div>
</div>

<!-- ── TRAINING HEALTH (B2+B6) ── -->
<div id="page-health" class="page">
  <div class="status-bar" id="health-status">Caricamento metriche…</div>
  <div class="cards" id="health-cards"></div>
  <div class="row2">
    <div class="chart-box"><h3>Train vs Val NLL</h3><canvas id="chart-nll"></canvas></div>
    <div class="chart-box"><h3>Gap (train-val) — overfit guard</h3><canvas id="chart-gap"></canvas></div>
  </div>
  <div class="row2">
    <div class="chart-box"><h3>Gradient Norm (mean / p95)</h3><canvas id="chart-grad"></canvas></div>
    <div class="chart-box"><h3>Directional Accuracy nel tempo</h3><canvas id="chart-da"></canvas></div>
  </div>
  <div class="chart-box" id="reliability-box" style="display:none">
    <h3>Reliability Diagram (calibrazione σ)</h3>
    <img id="reliability-img" style="max-width:100%;border-radius:4px;background:#0d1117;padding:8px">
  </div>
</div>

<!-- ── CONFRONTO ARCHS (A2) ── -->
<div id="page-compare" class="page">
  <div class="status-bar">
    ⚠️ Confronto solo su metriche scale-free (DA, Spearman, ICIR, Sharpe).
    val_nll NON è incluso perché non comparabile tra quantile-loss e t-Student-loss.
  </div>
  <div class="chart-box">
    <h3>Tabella architetture</h3>
    <table>
      <thead>
        <tr>
          <th>Arch</th><th>Loss</th><th>MoE</th>
          <th>DA</th><th>Spearman</th><th>ICIR</th>
          <th>Sharpe</th><th>Sortino</th><th>WR</th><th>PF</th><th>MaxDD</th>
        </tr>
      </thead>
      <tbody id="compare-table"></tbody>
    </table>
  </div>
  <div class="row2">
    <div class="chart-box"><h3>DA per arch</h3><canvas id="chart-cmp-da"></canvas></div>
    <div class="chart-box"><h3>ICIR per arch</h3><canvas id="chart-cmp-icir"></canvas></div>
  </div>
  <div class="row2">
    <div class="chart-box"><h3>Sharpe per arch</h3><canvas id="chart-cmp-sharpe"></canvas></div>
    <div class="chart-box"><h3>Max DD per arch</h3><canvas id="chart-cmp-dd"></canvas></div>
  </div>
</div>

<!-- ── ENSEMBLE (A7+B5) ── -->
<div id="page-ensemble" class="page">
  <div class="status-bar" id="ens-status">Caricamento composizione ensemble…</div>
  <div class="cards" id="ens-cards"></div>
  <div class="chart-box">
    <h3>Composizione ensemble eterogeneo</h3>
    <p style="color:var(--muted);font-size:13px;margin:4px 0 12px">
      Dalla config <code>distillation.archs</code> in <code>config/default.yaml</code>.
      Pesi da <code>DEFAULT_ARCH_WEIGHTS</code> in <code>quantsys/model/ensemble.py</code>.
    </p>
    <table>
      <thead>
        <tr>
          <th>Arch</th><th>Peso</th><th>Loss</th><th>MoE</th>
          <th>Distilled</th><th>Teacher</th><th>Best val</th><th>ICIR</th>
        </tr>
      </thead>
      <tbody id="ens-table"></tbody>
    </table>
  </div>
  <div class="row2">
    <div class="chart-box"><h3>Pesi ensemble</h3><canvas id="chart-ens-weights"></canvas></div>
    <div class="chart-box" id="ens-teacher-box">
      <h3>Teacher analysis (se eseguito 07_verify_teacher.py)</h3>
      <pre id="ens-teacher-json" style="background:#0d1117;padding:8px;border-radius:4px;overflow:auto;max-height:300px;font-size:12px;color:#8b949e"></pre>
    </div>
  </div>
</div>

<script>
// ─── Architecture selector (A1/G4) ────────────────────────────────────────────
let SELECTED_ARCH = '';

async function loadArchs() {
  try {
    const r = await fetch('/api/archs');
    if(!r.ok) return;
    const j = await r.json();
    SELECTED_ARCH = j.current;
    const sel = document.getElementById('arch-select');
    sel.innerHTML = (j.available||[j.current]).map(a =>
      `<option value="${a}" ${a===j.current?'selected':''}>${a.toUpperCase()}</option>`
    ).join('');
    loadArchInfo();
  } catch(e){}
}

// B1 (2026-05-15): badge che mostra loss_type · MoE accanto al selettore arch
async function loadArchInfo() {
  const badge = document.getElementById('arch-loss-badge');
  if (!badge) return;
  try {
    const r = await fetch('/api/arch-info' + _archParam());
    if (!r.ok) { badge.textContent = ''; return; }
    const i = await r.json();
    if (!i.exists) { badge.textContent = 'no model'; badge.style.color='#f85149'; return; }
    const moe = (i.n_output_experts || 1) > 1 ? `·MoE×${i.n_output_experts}` : '';
    const valStr = i.best_val_loss != null
      ? `·${Number(i.best_val_loss).toFixed(3)} ${i.scale_label==='quantile pinball'?'pinball':'NLL'}`
      : '';
    badge.textContent = `${i.loss_type}${moe}${valStr}`;
    badge.title = `Loss: ${i.scale_label}  ·  MoE experts: ${i.n_output_experts}  ·  best_val: ${i.best_val_loss}  ·  ⚠ NON confrontare numericamente val_nll tra arch con loss diverse`;
    badge.style.color = i.loss_type === 'quantile' ? '#3fb950' : '#d29922';
  } catch(e) { badge.textContent = ''; }
}

function onArchChange() {
  SELECTED_ARCH = document.getElementById('arch-select').value;
  loadBacktest();
  loadLive();
  loadArchInfo();
}

function _archParam() {
  return SELECTED_ARCH ? ('?arch=' + encodeURIComponent(SELECTED_ARCH)) : '';
}

// ─── Tab switching ────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  const tabs = ['backtest','live','macro','equity','walkforward','health','compare','ensemble'];
  const idx = tabs.indexOf(name);
  if (idx >= 0) document.querySelectorAll('.tab')[idx].classList.add('active');
  const pg = document.getElementById('page-'+name);
  if (pg) pg.classList.add('active');
  if (name === 'macro')       { loadMacro(); loadRegimePerf(); }
  if (name === 'health')      loadTrainingHealth();
  if (name === 'compare')     loadArchComparison();
  if (name === 'ensemble')    loadEnsemble();
  if (name === 'walkforward') loadWalkforward();
}

// ─── Chart helpers ────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  responsive: true,
  animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
  }
};
const charts = {};

function mkChart(id, type, labels, datasets, extraOpts={}) {
  // C1/G5: aggiorna in-place se il chart esiste e il type è invariato
  const existing = charts[id];
  if (existing && existing.config.type === type) {
    existing.data.labels = labels;
    // Aggiorna i dataset preservando i riferimenti dove possibile
    if (existing.data.datasets.length === datasets.length) {
      datasets.forEach((d, i) => {
        Object.assign(existing.data.datasets[i], d);
      });
    } else {
      existing.data.datasets = datasets;
    }
    existing.update('none');
    return;
  }
  if (existing) existing.destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx, {
    type,
    data: { labels, datasets },
    options: { ...CHART_DEFAULTS, ...extraOpts }
  });
}

function lineDs(data, color, fill=false, label='') {
  return {
    label, data,
    borderColor: color, backgroundColor: fill ? color+'22' : 'transparent',
    borderWidth: 1.5, pointRadius: 0, fill
  };
}

function barDs(data, colors) {
  return { data, backgroundColor: colors, borderRadius: 2, borderWidth: 0 };
}

// ─── Backtest ─────────────────────────────────────────────────────────────────
let btData = null;

// D2/D3: retry counter — mostra "no data" solo dopo N tentativi falliti consecutivi
let _btFails = 0, _lvFails = 0;
const FAIL_THRESHOLD = 3;

async function loadBacktest() {
  try {
    const r = await fetch('/api/backtest' + _archParam());
    if (!r.ok) {
      const j = await r.json().catch(()=>({}));
      throw new Error(j.error || r.status);
    }
    btData = await r.json();
    _btFails = 0;
    document.getElementById('no-backtest').style.display = 'none';
    document.getElementById('backtest-content').style.display = '';
    renderBacktest(btData);
  } catch(e) {
    _btFails++;
    if (_btFails >= FAIL_THRESHOLD) {
      const nb = document.getElementById('no-backtest');
      nb.style.display = '';
      nb.innerHTML = `<p>Backtest non disponibile.</p><p style="margin-top:8px;color:#f85149;font-size:11px">${(e&&e.message)||e||''}</p>`;
      document.getElementById('backtest-content').style.display = 'none';
    }
  }
}

function fmt(v, dec=2, suffix='') {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return v.toFixed(dec) + suffix;
}

function renderBacktest(d) {
  const m = d.metrics || {};
  const cards = [
    { label: 'Sharpe Ratio',    value: fmt(m.sharpe, 2),          cls: (m.sharpe||0)>0?'green':'red' },
    { label: 'Win Rate',        value: fmt((m.win_rate||0)*100,1,'%'), cls: (m.win_rate||0)>0.5?'green':'yellow' },
    { label: 'Profit Factor',   value: fmt(m.profit_factor, 2),   cls: (m.profit_factor||0)>1?'green':'red' },
    { label: 'Max Drawdown',    value: fmt((m.max_drawdown||0)*100,1,'%'), cls: 'red' },
    { label: 'N Trade',         value: m.n_trades ?? '—',    cls: 'blue' },
    { label: 'Avg Hold (min)',  value: fmt(m.avg_hold_candles,0), cls: 'blue' },
    { label: 'Calmar',          value: fmt(m.calmar, 2),          cls: (m.calmar||0)>0?'green':'red' },
    { label: 'Sortino',         value: fmt(m.sortino, 2),         cls: (m.sortino||0)>0?'green':'yellow' },
  ];
  document.getElementById('metric-cards').innerHTML = cards.map(c=>
    `<div class="card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls}">${c.value}</div></div>`
  ).join('');

  const eq = d.equity_curve || [];
  const dd = d.drawdown_curve || [];
  const pnlPerTrade = d.pnl_per_trade || d.pnl_series || [];
  const labels = eq.map((_,i)=>i);

  // A8 (2026-05-15): buy&hold overlay. Calcoliamo da pnl_series (returns implied)
  // o, se mancante, dal primo trade entry_price → ultimo trade exit_price.
  const bh = computeBuyHold(d, eq.length);

  const eqDatasets = [
    Object.assign(lineDs(eq,'#58a6ff',true), { label: 'Strategia' })
  ];
  if (bh && bh.length === eq.length) {
    eqDatasets.push({
      label: 'Buy&Hold', data: bh, borderColor:'#8b949e', backgroundColor:'transparent',
      borderWidth:1.2, pointRadius:0, borderDash:[4,3]
    });
  }
  mkChart('chart-equity','line',labels, eqDatasets, {plugins:{legend:{display:true,labels:{color:'#8b949e'}}}});
  mkChart('chart-dd','line',labels,[lineDs(dd.map(v=>v*100),'#f85149',true)]);

  const pnlColors = pnlPerTrade.map(v=>v>=0?'#3fb95055':'#f8514955');
  mkChart('chart-pnl','bar',pnlPerTrade.map((_,i)=>i),[barDs(pnlPerTrade,pnlColors)]);

  // Equity full tab (con buy&hold)
  mkChart('chart-eq-full','line',labels, eqDatasets, {plugins:{legend:{display:true,labels:{color:'#8b949e'}}}});
  mkChart('chart-pnl-series','line',labels,[lineDs(d.pnl_series||[],'#d29922',false)]);

  // A12: rolling Sharpe + DD su equity tab
  renderRollingCharts(eq, pnlPerTrade);

  // PnL histogram (20 buckets)
  const pnl = pnlPerTrade.filter(v=>!isNaN(v));
  if (pnl.length > 0) {
    const mn = Math.min(...pnl), mx = Math.max(...pnl);
    const bins = 20, step = (mx-mn)/bins || 1;
    const counts = new Array(bins).fill(0);
    const binLabels = [];
    for (let i=0;i<bins;i++) binLabels.push((mn+i*step).toFixed(2));
    pnl.forEach(v=>{ const b = Math.min(Math.floor((v-mn)/step), bins-1); counts[b]++; });
    const hColors = binLabels.map(v=>parseFloat(v)>=0?'#3fb95088':'#f8514988');
    mkChart('chart-pnl-hist','bar',binLabels,[barDs(counts,hColors)]);
  }

  // Trade table
  const trades = (d.trades || []).slice(-30).reverse();
  document.getElementById('trade-table').innerHTML = trades.map((t,i)=>{
    const pnlPct = t.entry_price ? ((t.exit_price - t.entry_price)/t.entry_price * (t.side==='LONG'?1:-1)*100) : null;
    const cls = (pnlPct||0)>=0?'green':'red';
    return `<tr>
      <td>${i+1}</td>
      <td style="color:${t.side==='LONG'?'#3fb950':'#f85149'}">${t.side}</td>
      <td>${fmt(t.entry_price,2)}</td>
      <td>${fmt(t.exit_price,2)}</td>
      <td>$${fmt(t.size_usd,0)}</td>
      <td>${t.hold_candles??'—'}m</td>
      <td class="${cls}">${fmt(pnlPct,2,'%')}</td>
    </tr>`;
  }).join('');

  const maxDd = dd.length ? dd.reduce((a,b)=>a>b?a:b, 0) : 0;
  document.getElementById('eq-status').textContent =
    `Equity: $${fmt(eq[eq.length-1],2)} | DD max: ${fmt(maxDd*100,1)}% | Trade: ${m.n_trades??0}`;
}

// ─── Live signals ─────────────────────────────────────────────────────────────
let liveChartEq = null, liveChartProb = null;
// C3 (2026-05-15): delta sync via ?since=ts. Manteniamo un buffer locale e
// chiediamo solo i nuovi segnali al server. Reset al cambio arch.
let _liveBuffer = [];
let _liveLastTs = 0;
let _liveLastArch = null;

async function loadLive() {
  try {
    // Reset buffer al cambio arch
    if (_liveLastArch !== SELECTED_ARCH) {
      _liveBuffer = [];
      _liveLastTs = 0;
      _liveLastArch = SELECTED_ARCH;
    }
    const sinceParam = _liveLastTs > 0 ? ('&since=' + _liveLastTs) : '';
    const sep = _archParam() ? '&' : '?';
    const url = '/api/live' + _archParam() + (sinceParam ? sep + sinceParam.slice(1) : '');
    const r = await fetch(url);
    if (!r.ok) throw new Error(r.status);
    const sigs = await r.json();
    // Merge nuovi segnali nel buffer
    if (sigs && sigs.length) {
      _liveBuffer = _liveBuffer.concat(sigs);
      // Bound a ~200 elementi per stabilità memoria
      if (_liveBuffer.length > 200) _liveBuffer = _liveBuffer.slice(-200);
      const lastSig = sigs[sigs.length - 1];
      const lastTs = lastSig.ts || lastSig.timestamp;
      if (lastTs) _liveLastTs = Number(lastTs);
    }
    if (!_liveBuffer.length) throw new Error('empty');
    renderLive(_liveBuffer.slice(-100));
    _lvFails = 0;
    document.getElementById('no-live').style.display = 'none';
    document.getElementById('live-content').style.display = '';
  } catch(e) {
    _lvFails++;
    if (_lvFails >= FAIL_THRESHOLD) {
      document.getElementById('no-live').style.display = '';
      document.getElementById('live-content').style.display = 'none';
    }
  }
}

function renderLive(sigs) {
  const last = sigs[sigs.length - 1];

  // Cards live
  const inPos = last.in_position;
  const sigColor = last.signal==='BUY'?'green':last.signal==='SELL'?'red':'yellow';

  // A9/G3: Latency live (tempo dall'ultimo segnale)
  let latStr = '—', latCls = 'muted';
  if (last.ts) {
    const dt = (Date.now() - new Date(last.ts).getTime()) / 1000;
    if (dt < 0 || isNaN(dt)) { latStr = '—'; }
    else if (dt < 60)   { latStr = `${dt.toFixed(0)}s fa`;  latCls = 'green'; }
    else if (dt < 300)  { latStr = `${(dt/60).toFixed(1)}m fa`; latCls = 'yellow'; }
    else                { latStr = `${(dt/60).toFixed(0)}m fa`; latCls = 'red'; }
  }

  const cards = [
    { label: 'Segnale',        value: last.signal,                    cls: sigColor },
    { label: 'Prob Up',        value: fmt((last.prob_up||0)*100,1,'%'), cls: (last.prob_up||0)>0.5?'green':'red' },
    { label: 'Equity live',    value: fmt(last.equity,2),             cls: (last.equity||0)>1000?'green':'red' },
    { label: 'N Trade',        value: last.n_trades ?? '—',      cls: 'blue' },
    { label: 'In Position',    value: inPos ? 'Sì' : 'NO',       cls: inPos?'green':'muted' },
    { label: 'Ultimo segnale', value: latStr,                    cls: latCls },
    { label: 'μ (pred)',  value: fmt(last.mu,4),                 cls: (last.mu||0)>0?'green':'red' },
    { label: 'σ (pred)',  value: fmt(last.sigma,4),              cls: 'yellow' },
    { label: 'ν (df)',    value: fmt(last.nu,1),                 cls: 'blue' },
  ];
  document.getElementById('live-cards').innerHTML = cards.map(c=>
    `<div class="card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls}">${c.value}</div></div>`
  ).join('');

  // Signal feed
  const feed = document.getElementById('signal-feed');
  feed.innerHTML = [...sigs].reverse().slice(0,40).map(s=>{
    const t = s.ts ? new Date(s.ts).toLocaleTimeString('it',{hour:'2-digit',minute:'2-digit',second:'2-digit'}) : '';
    return `<div class="sig-row">
      <span class="sig-badge ${s.signal}">${s.signal}</span>
      <span class="sig-meta">μ=${fmt(s.mu,4)} σ=${fmt(s.sigma,4)}</span>
      <span style="color:var(--muted);font-size:11px">${t}</span>
      <span class="sig-price">$${fmt(s.price,1)}</span>
    </div>`;
  }).join('');

  // Equity live chart
  const eqLabels = sigs.map((_,i)=>i);
  mkChart('chart-live-eq','line',eqLabels,[lineDs(sigs.map(s=>s.equity),'#3fb950',true)]);

  // Prob_up chart
  mkChart('chart-prob','line',eqLabels,[lineDs(sigs.map(s=>(s.prob_up||0)*100),'#58a6ff',false)],{
    scales: {
      x: { ticks:{color:'#8b949e',maxTicksLimit:8}, grid:{color:'#21262d'} },
      y: { min:0, max:100, ticks:{color:'#8b949e'}, grid:{color:'#21262d'} }
    }
  });
}

// ─── Macro & Regime (A3/G2 + A5) ──────────────────────────────────────────────
const REGIME_COLORS = ['#3fb950', '#d29922', '#f85149', '#58a6ff']; // verde=risk-on, giallo=neutro, rosso=crisis

async function loadMacro() {
  let regOk = true, frOk = true, regData = null, frData = null;
  try {
    const r = await fetch('/api/regime?n=180');
    if (!r.ok) throw new Error(r.status);
    regData = await r.json();
  } catch(e) { regOk = false; }
  try {
    const r = await fetch('/api/funding?n=180');
    if (!r.ok) throw new Error(r.status);
    frData = await r.json();
  } catch(e) { frOk = false; }

  if (!regOk && !frOk) {
    document.getElementById('no-macro').style.display = '';
    return;
  }
  document.getElementById('no-macro').style.display = 'none';

  // Cards
  const cards = [];
  if (regOk && regData.current) {
    const dom = regData.current.dominant;
    const prob = regData.current.probs[dom] || 0;
    const regLabel = ['Risk-On', 'Neutro', 'Crisis'][dom] || `R${dom}`;
    cards.push({ label:'Regime corrente', value: regLabel, cls: dom===0?'green':dom===1?'yellow':'red' });
    cards.push({ label:'Prob regime',     value: fmt(prob*100,1,'%'), cls:'blue' });
    cards.push({ label:'Aggiornato',      value: regData.current.date, cls:'muted' });
  }
  if (frOk && frData.current) {
    const v = frData.current.value;
    const ann = frData.current.annualized_pct;
    cards.push({ label:'Funding (8h)',    value: fmt(v*100,4,'%'), cls: (v||0)>=0?'green':'red' });
    cards.push({ label:'Funding annual.', value: fmt(ann,2,'%'),   cls: (ann||0)>=0?'green':'red' });
  }
  document.getElementById('macro-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="card-label">${c.label}</div><div class="card-value ${c.cls}">${c.value}</div></div>`
  ).join('');

  // Regime dominant timeline (bar chart colorato)
  if (regOk) {
    const colors = regData.dominant.map((d, i) => regData.burn_in[i] ? '#30363d' : REGIME_COLORS[d % REGIME_COLORS.length]);
    mkChart('chart-regime-dom', 'bar', regData.dates, [{ data: regData.dominant.map(()=>1), backgroundColor: colors, borderRadius: 0, borderWidth: 0 }], {
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { color: '#21262d' } },
        y: { display: false }
      },
      plugins: { legend: { display: false } }
    });
    // Probabilità regimi stacked
    const probDatasets = Object.keys(regData.probs).map((k, i) => ({
      label: 'R' + i,
      data: regData.probs[k].map(v => v * 100),
      backgroundColor: REGIME_COLORS[i % REGIME_COLORS.length] + 'aa',
      borderColor: REGIME_COLORS[i % REGIME_COLORS.length],
      borderWidth: 1, pointRadius: 0, fill: true
    }));
    mkChart('chart-regime-prob', 'line', regData.dates, probDatasets, {
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 8 }, grid: { color: '#21262d' }, stacked: true },
        y: { min: 0, max: 100, stacked: true, ticks: { color: '#8b949e' }, grid: { color: '#21262d' } }
      },
      plugins: { legend: { display: true, labels: { color: '#8b949e' } } }
    });
  }

  // Funding rate
  if (frOk) {
    const fr_pct = frData.funding_rate.map(v => v * 100);
    const colors = fr_pct.map(v => v >= 0 ? '#3fb95088' : '#f8514988');
    mkChart('chart-funding', 'bar', frData.ts, [{ data: fr_pct, backgroundColor: colors, borderRadius: 1, borderWidth: 0 }], {
      scales: {
        x: { ticks: { color: '#8b949e', maxTicksLimit: 10 }, grid: { color: '#21262d' } },
        y: { ticks: { color: '#8b949e', callback: v => v.toFixed(3) + '%' }, grid: { color: '#21262d' } }
      }
    });
  }
}

// ─── Training Health (B2+B6, 2026-05-15) ─────────────────────────────────────
async function loadTrainingHealth() {
  try {
    const r = await fetch('/api/training-health' + _archParam());
    if (!r.ok) { document.getElementById('health-status').textContent = 'Errore: ' + r.status; return; }
    const d = await r.json();
    renderTrainingHealth(d);
  } catch (e) {
    document.getElementById('health-status').textContent = 'Errore di rete: ' + e;
  }
}

function renderTrainingHealth(d) {
  const cal = d.calibration || {};
  const ece = cal.ece;
  const gradLast = (d.grad_norm_mean && d.grad_norm_mean.length)
    ? d.grad_norm_mean[d.grad_norm_mean.length-1] : null;
  const gapLast = (d.gap && d.gap.length) ? d.gap[d.gap.length-1] : null;
  const daLast = (d.val_dir_acc && d.val_dir_acc.length)
    ? d.val_dir_acc[d.val_dir_acc.length-1] : null;
  const nEp = (d.val_nll || []).length;

  document.getElementById('health-status').textContent =
    `Arch: ${d.arch}  ·  ${nEp} epoche  ·  ${cal.ece !== undefined ? 'calibration.json caricato' : 'no calibration data'}`;

  const eceCls = ece == null ? 'gray' : (ece < 0.05 ? 'green' : ece < 0.1 ? 'yellow' : 'red');
  const gapCls = gapLast == null ? 'gray' : (gapLast > -0.5 ? 'green' : 'red');

  const cards = [
    { label: 'ECE',      value: ece != null ? ece.toFixed(4) : '—', cls: eceCls,
      sub: ece != null ? (ece<0.05?'ottimo':ece<0.1?'ok':'mal calibrato') : '' },
    { label: 'Gap (last)',  value: gapLast != null ? gapLast.toFixed(3) : '—', cls: gapCls,
      sub: 'train_nll - val_nll' },
    { label: '‖∇‖ mean',    value: gradLast != null ? gradLast.toFixed(2) : '—', cls: 'blue', sub: 'pre-clip' },
    { label: 'DA (last)',   value: daLast != null ? (daLast*100).toFixed(1)+'%' : '—',
      cls: daLast == null ? 'gray' : daLast > 0.51 ? 'green' : 'yellow' },
  ];
  document.getElementById('health-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="card-label">${c.label}</div>
       <div class="card-value ${c.cls}">${c.value}</div>
       <div style="font-size:11px;color:var(--muted);margin-top:4px">${c.sub||''}</div></div>`
  ).join('');

  const xs = (d.val_nll || []).map((_,i)=>i+1);
  if (xs.length) {
    mkChart('chart-nll', 'line', xs, [
      { label:'train_nll', data:d.train_nll||[], borderColor:'#58a6ff', backgroundColor:'transparent', borderWidth:1.5, pointRadius:0 },
      { label:'val_nll',   data:d.val_nll||[],   borderColor:'#d29922', backgroundColor:'transparent', borderWidth:1.5, pointRadius:0 },
    ], { plugins:{legend:{display:true,labels:{color:'#8b949e'}}} });
    mkChart('chart-gap', 'line', xs, [
      { data:d.gap||[], borderColor:'#a371f7', backgroundColor:'transparent', borderWidth:1.5, pointRadius:0 }
    ]);
    mkChart('chart-grad', 'line', xs, [
      { label:'mean', data:d.grad_norm_mean||[], borderColor:'#58a6ff', backgroundColor:'transparent', borderWidth:1.2, pointRadius:0 },
      { label:'p95',  data:d.grad_norm_p95||[],  borderColor:'#f85149', backgroundColor:'transparent', borderWidth:1.2, pointRadius:0 },
    ], { plugins:{legend:{display:true,labels:{color:'#8b949e'}}}, scales:{y:{type:'logarithmic',ticks:{color:'#8b949e'},grid:{color:'#21262d'}}} });
    mkChart('chart-da', 'line', xs, [
      { data:(d.val_dir_acc||[]).map(v=>v*100), borderColor:'#3fb950', backgroundColor:'transparent', borderWidth:1.5, pointRadius:0 }
    ], { scales:{y:{ticks:{color:'#8b949e',callback:v=>v.toFixed(1)+'%'},grid:{color:'#21262d'}}} });
  }

  // Reliability diagram PNG
  if (d.reliability_png_b64) {
    document.getElementById('reliability-box').style.display = 'block';
    document.getElementById('reliability-img').src = 'data:image/png;base64,' + d.reliability_png_b64;
  } else {
    document.getElementById('reliability-box').style.display = 'none';
  }
}

// ─── Arch Comparison (A2, 2026-05-15) ─────────────────────────────────────────
async function loadArchComparison() {
  try {
    const r = await fetch('/api/arch-comparison');
    if (!r.ok) return;
    const d = await r.json();
    renderArchComparison(d.archs || []);
  } catch(e) { console.error('arch-comparison', e); }
}

function _fmt(v, dec=3) {
  if (v == null || v === undefined || isNaN(v)) return '—';
  return Number(v).toFixed(dec);
}

// A8 (2026-05-15): calcola serie equity buy&hold dello stesso capital iniziale
// estendendola sulla stessa lunghezza dell'equity strategia.
// Approccio: usa i prezzi entry/exit dei trade per stimare il prezzo nel tempo.
// Se non possibile, fallback a serie costante (no overlay).
function computeBuyHold(bt, lenTarget) {
  const trades = bt.trades || [];
  if (trades.length < 2) return null;
  const startPrice = trades[0].entry_price;
  const endPrice   = trades[trades.length-1].exit_price;
  const initCap    = (bt.equity_curve||[10000])[0] || 10000;
  if (!startPrice || !endPrice || startPrice <= 0) return null;
  // Interpolazione lineare in spazio log-prezzo
  const out = new Array(lenTarget);
  const logStart = Math.log(startPrice);
  const logEnd   = Math.log(endPrice);
  for (let i = 0; i < lenTarget; i++) {
    const t = lenTarget > 1 ? i / (lenTarget - 1) : 0;
    const p = Math.exp(logStart + (logEnd - logStart) * t);
    out[i] = initCap * (p / startPrice);
  }
  return out;
}
function _pct(v, dec=1) {
  if (v == null || isNaN(v)) return '—';
  return (Number(v)*100).toFixed(dec) + '%';
}

function renderArchComparison(rows) {
  if (!rows.length) {
    document.getElementById('compare-table').innerHTML =
      '<tr><td colspan="11" style="text-align:center;color:var(--muted)">Nessun modello trovato in models/</td></tr>';
    return;
  }
  document.getElementById('compare-table').innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.arch}</strong>${r.distilled?' <span style="color:#3fb950">★</span>':''}</td>
      <td>${r.loss_type || '—'}</td>
      <td>${r.n_experts || 1}</td>
      <td>${_pct(r.dir_acc)}</td>
      <td>${_fmt(r.spearman, 4)}</td>
      <td>${_fmt(r.icir, 3)}</td>
      <td>${_fmt(r.sharpe, 2)}</td>
      <td>${_fmt(r.sortino, 2)}</td>
      <td>${_pct(r.win_rate)}</td>
      <td>${_fmt(r.profit_factor, 2)}</td>
      <td>${_pct(r.max_dd)}</td>
    </tr>
  `).join('');

  const labels = rows.map(r => r.arch);
  const colors = ['#58a6ff','#3fb950','#d29922','#a371f7','#f85149'];
  const bg = labels.map((_,i) => colors[i % colors.length]);

  const _bar = (id, key, fmt) => {
    const data = rows.map(r => r[key] == null ? 0 : Number(r[key]) * (fmt==='pct'?100:1));
    mkChart(id, 'bar', labels, [{ data, backgroundColor: bg, borderWidth: 0 }], {
      scales: { y: { ticks: { color: '#8b949e', callback: fmt==='pct'? v=>v.toFixed(1)+'%' : v=>v }, grid:{color:'#21262d'}} }
    });
  };
  _bar('chart-cmp-da', 'dir_acc', 'pct');
  _bar('chart-cmp-icir', 'icir');
  _bar('chart-cmp-sharpe', 'sharpe');
  _bar('chart-cmp-dd', 'max_dd', 'pct');
}

// ─── Ensemble panel (A7+B5, 2026-05-15) ───────────────────────────────────────
async function loadEnsemble() {
  try {
    const r = await fetch('/api/ensemble');
    if (!r.ok) {
      document.getElementById('ens-status').textContent = 'Errore: /api/ensemble ' + r.status;
      return;
    }
    const d = await r.json();
    renderEnsemble(d);
  } catch(e) {
    document.getElementById('ens-status').textContent = 'Errore di rete: ' + e;
  }
}

function renderEnsemble(d) {
  const members = d.members || [];
  document.getElementById('ens-status').textContent =
    `${members.length} membri eterogenei configurati`;

  // Cards riassuntive
  const distilled = members.filter(m => m.distilled).length;
  const teachers = [...new Set(members.map(m => m.teacher_arch).filter(Boolean))];
  const cards = [
    { label: 'Membri', value: members.length, cls: 'blue' },
    { label: 'Distilled', value: `${distilled}/${members.length}`, cls: distilled>0?'green':'yellow' },
    { label: 'Teacher(s)', value: teachers.length ? teachers.join(', ') : '—', cls: 'blue' },
  ];
  document.getElementById('ens-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="card-label">${c.label}</div>
       <div class="card-value ${c.cls}">${c.value}</div></div>`
  ).join('');

  // Tabella membri
  document.getElementById('ens-table').innerHTML = members.map(m => `
    <tr>
      <td><strong>${m.arch}</strong></td>
      <td>${_fmt(m.weight, 3)}</td>
      <td>${m.loss_type || '—'}</td>
      <td>${m.n_output_experts || 1}</td>
      <td>${m.distilled ? '✓' : '—'}</td>
      <td>${m.teacher_arch || '—'}</td>
      <td>${_fmt(m.best_val_loss, 4)} <span style="color:#6e7681;font-size:11px">(${m.loss_type==='quantile'?'pinball':'NLL'})</span></td>
      <td>${_fmt(m.best_icir, 3)}</td>
    </tr>
  `).join('');

  // Bar chart pesi
  const labels = members.map(m => m.arch);
  const data = members.map(m => m.weight);
  mkChart('chart-ens-weights', 'bar', labels,
    [{ data, backgroundColor: ['#58a6ff','#3fb950','#d29922','#a371f7','#f85149'].slice(0,labels.length), borderWidth: 0 }],
    { scales:{y:{ticks:{color:'#8b949e',callback:v=>(v*100).toFixed(0)+'%'},grid:{color:'#21262d'}}} });

  // Teacher analysis JSON
  if (d.teacher_analysis) {
    document.getElementById('ens-teacher-json').textContent =
      JSON.stringify(d.teacher_analysis, null, 2);
  } else {
    document.getElementById('ens-teacher-json').textContent =
      '(esegui scripts/07_verify_teacher.py per generare models/teacher_analysis.json)';
  }
}

// ─── Walk-forward results (A6, 2026-05-15) ────────────────────────────────────
async function loadWalkforward() {
  try {
    const r = await fetch('/api/walkforward' + _archParam());
    if (!r.ok) {
      const e = await r.json().catch(()=>({error:r.status}));
      document.getElementById('wf-status').textContent = 'Walk-forward non eseguito: ' + (e.error||r.status);
      document.getElementById('wf-table').innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--muted)">Esegui: python scripts/02b_walkforward_validate.py</td></tr>';
      return;
    }
    const d = await r.json();
    renderWalkforward(d);
  } catch (e) {
    document.getElementById('wf-status').textContent = 'Errore di rete: ' + e;
  }
}

function renderWalkforward(d) {
  const folds = d.fold_results || [];
  const agg = d.aggregate || {};
  document.getElementById('wf-status').textContent =
    `${d.n_folds||folds.length} fold · embargo=${d.embargo_steps||'?'} · retrain=${d.retrained?'sì':'no'}`;

  // Aggregate cards
  const _c = (label, key, dec=3) => ({
    label, cls: 'blue',
    value: agg[key] ? `${_fmt(agg[key].mean, dec)} ± ${_fmt(agg[key].std, dec)}` : '—',
    sub: agg[key] ? `min ${_fmt(agg[key].min,dec)}  ·  max ${_fmt(agg[key].max,dec)}` : ''
  });
  const cards = [_c('DA', 'da', 4), _c('Spearman','spearman',4), _c('WHR','whr',4), _c('CI90 cov','ci90',3)];
  document.getElementById('wf-cards').innerHTML = cards.map(c =>
    `<div class="card"><div class="card-label">${c.label}</div>
       <div class="card-value ${c.cls}">${c.value}</div>
       <div style="font-size:11px;color:var(--muted);margin-top:4px">${c.sub||''}</div></div>`
  ).join('');

  document.getElementById('wf-table').innerHTML = folds.map(f => `
    <tr>
      <td>${f.fold}</td>
      <td>${(f.n||0).toLocaleString()}</td>
      <td>${_fmt(f.da, 4)}</td>
      <td>${_fmt(f.spearman, 4)}</td>
      <td>${_fmt(f.whr, 4)}</td>
      <td>${_fmt(f.ci90, 3)}</td>
      <td>${_fmt(f.val_nll, 4)}</td>
      <td>${_fmt(f.elapsed_s, 1)}</td>
    </tr>
  `).join('');

  const labels = folds.map(f => 'Fold ' + f.fold);
  const colors = folds.map((_,i)=> ['#58a6ff','#3fb950','#d29922','#a371f7','#f85149'][i%5]);
  mkChart('chart-wf-da', 'bar', labels, [{ data: folds.map(f=>f.da), backgroundColor: colors, borderWidth:0 }], {
    scales:{y:{ticks:{color:'#8b949e',callback:v=>v.toFixed(3)},grid:{color:'#21262d'}}}
  });
  mkChart('chart-wf-sp', 'bar', labels, [{ data: folds.map(f=>f.spearman), backgroundColor: colors, borderWidth:0 }], {
    scales:{y:{ticks:{color:'#8b949e',callback:v=>v.toFixed(3)},grid:{color:'#21262d'}}}
  });
}

// ─── Export trades CSV (A10) ──────────────────────────────────────────────────
function exportTradesCSV() {
  const url = '/api/export-trades.csv' + _archParam();
  // Trigger download
  const a = document.createElement('a');
  a.href = url;
  a.download = 'trades_' + (SELECTED_ARCH||'default') + '.csv';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ─── Regime performance (A11) ─────────────────────────────────────────────────
async function loadRegimePerf() {
  try {
    const r = await fetch('/api/regime-perf' + _archParam());
    const tb = document.getElementById('regime-perf-table');
    if (!r.ok) {
      tb.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted)">live_signals/regime non disponibili</td></tr>';
      return;
    }
    const d = await r.json();
    const pr = d.per_regime || [];
    if (!pr.length) {
      tb.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--muted)">Nessun match regime ↔ segnali live</td></tr>';
      return;
    }
    const regNames = {0:'Risk-off',1:'Risk-on',2:'Crisis'};
    tb.innerHTML = pr.map(r => `
      <tr>
        <td><strong>${r.regime}</strong> <span style="color:var(--muted);font-size:11px">${regNames[r.regime]||''}</span></td>
        <td>${r.n}</td>
        <td><span style="color:#3fb950">${r.n_long||0}</span></td>
        <td><span style="color:#f85149">${r.n_short||0}</span></td>
        <td><span style="color:var(--muted)">${r.n_flat||0}</span></td>
        <td>${_fmt(r.prob_up_mean, 3)}</td>
        <td>${_fmt(r.sigma_mean, 3)}</td>
      </tr>
    `).join('');
  } catch (e) { console.error('regime-perf', e); }
}

// ─── Rolling Sharpe + DD (A12) ────────────────────────────────────────────────
function computeRollingSharpe(returns, win=30) {
  // returns: array di P&L % per trade
  const out = new Array(returns.length).fill(null);
  for (let i = win - 1; i < returns.length; i++) {
    const w = returns.slice(i - win + 1, i + 1);
    const m = w.reduce((a,b)=>a+b, 0) / win;
    const v = w.reduce((a,b)=>a+(b-m)**2, 0) / (win - 1);
    const s = Math.sqrt(v);
    out[i] = s > 1e-12 ? (m / s) * Math.sqrt(252) : 0;  // annualizzato approx
  }
  return out;
}

function computeRollingDD(equity) {
  // Drawdown rolling dal peak running
  const out = new Array(equity.length).fill(0);
  let peak = equity[0] || 1;
  for (let i = 0; i < equity.length; i++) {
    if (equity[i] > peak) peak = equity[i];
    out[i] = peak > 0 ? (equity[i] - peak) / peak : 0;
  }
  return out;
}

function renderRollingCharts(equityArr, pnlPctArr) {
  if (!pnlPctArr || !pnlPctArr.length) return;
  const win = Math.min(30, Math.max(5, Math.floor(pnlPctArr.length / 6)));
  const rs = computeRollingSharpe(pnlPctArr, win);
  mkChart('chart-rolling-sharpe', 'line', rs.map((_,i)=>i),
    [{ data: rs, borderColor: '#58a6ff', backgroundColor:'transparent', borderWidth:1.5, pointRadius:0, spanGaps:true }]);
  if (equityArr && equityArr.length) {
    const dd = computeRollingDD(equityArr).map(v => v * 100);
    mkChart('chart-rolling-dd', 'line', dd.map((_,i)=>i),
      [{ data: dd, borderColor:'#f85149', backgroundColor:'rgba(248,81,73,0.15)', fill:true, borderWidth:1.2, pointRadius:0 }],
      { scales:{y:{ticks:{color:'#8b949e',callback:v=>v.toFixed(1)+'%'},grid:{color:'#21262d'}}} });
  }
}

// ─── Binance WebSocket live price ─────────────────────────────────────────────
// C7 (2026-05-15): @trade emette centinaia di msg/s. Sostituito con @kline_1m
// che emette ~1 msg/s e include open/high/low/close/volume. Risparmio bandwidth
// e CPU client del ~99%.
let _lastPrice = null, _lastTitleUpdate = 0;
function connectBinanceWS() {
  const ws = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@kline_1m');
  ws.onmessage = e => {
    const d = JSON.parse(e.data);
    const k = d.k;  // payload kline
    if (k && k.c) {
      const p = parseFloat(k.c);
      document.getElementById('live-price').textContent = '$' + p.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
      const now = Date.now();
      if (now - _lastTitleUpdate > 1000) {
        const arrow = (_lastPrice !== null) ? (p > _lastPrice ? '▲' : p < _lastPrice ? '▼' : '◆') : '◆';
        document.title = `${arrow} $${p.toLocaleString('en-US',{maximumFractionDigits:0})} · QUANTSYS`;
        _lastPrice = p;
        _lastTitleUpdate = now;
      }
    }
  };
  ws.onerror = () => { document.getElementById('live-badge').textContent = '○ OFFLINE'; document.getElementById('live-badge').style.color='#f85149'; };
  ws.onclose = () => setTimeout(connectBinanceWS, 5000);
}

// E1 (2026-05-15): Δ% 24h e volume nel header via REST /api/v3/ticker/24hr.
// Refresh ogni 60s — frequenza più che sufficiente per uno stat 24h.
async function refresh24hStats() {
  try {
    const r = await fetch('https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT');
    if (!r.ok) return;
    const j = await r.json();
    const chg = parseFloat(j.priceChangePercent);
    const vol = parseFloat(j.volume);  // base asset (BTC)
    const el = document.getElementById('live-change');
    if (el) {
      el.textContent = (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%';
      el.style.color = chg >= 0 ? '#3fb950' : '#f85149';
    }
    const ev = document.getElementById('live-volume');
    if (ev) {
      ev.textContent = 'vol ' + (vol >= 1000 ? (vol/1000).toFixed(1) + 'k' : vol.toFixed(0)) + ' BTC';
    }
  } catch(e) {}
}

// ─── Init + polling ───────────────────────────────────────────────────────────
loadArchs().then(() => { loadBacktest(); loadLive(); });
connectBinanceWS();
refresh24hStats();                 // E1: stat 24h all'avvio

setInterval(loadLive, 5000);       // aggiorna segnali ogni 5s
setInterval(loadBacktest, 60000);  // ricarica backtest ogni minuto
setInterval(refresh24hStats, 60000); // E1: refresh stat 24h ogni minuto

// ── Run Pipeline ─────────────────────────────────────────────────────────────
let _pollTimer = null;
let _logVisible = false;

function openRunModal() { document.getElementById('run-modal').classList.add('open'); }
function closeRunModal() { document.getElementById('run-modal').classList.remove('open'); }
function closeDrawer() { document.getElementById('run-drawer').classList.remove('open'); }
function toggleLog() { _logVisible=!_logVisible; document.getElementById('run-log').classList.toggle('show',_logVisible); }

async function startRun() {
  closeRunModal();
  const steps = [];
  if(document.getElementById('s-update').checked)  steps.push('update');
  if(document.getElementById('s-macro').checked)   steps.push('macro');
  if(document.getElementById('s-train').checked)   steps.push('train');
  if(document.getElementById('s-distill').checked) steps.push('distill');
  if(document.getElementById('s-walkfwd').checked) steps.push('walkfwd');
  if(document.getElementById('s-backtest').checked) steps.push('backtest');
  if(document.getElementById('s-analyze').checked)  steps.push('analyze');
  if(!steps.length){ alert('Seleziona almeno un passo.'); return; }

  // Auth token (B2): se presente in localStorage, lo invia come X-Auth-Token
  const _hdrs = {'Content-Type':'application/json'};
  const _tok = localStorage.getItem('quantsys_auth_token') || '';
  if(_tok) _hdrs['X-Auth-Token'] = _tok;

  try {
    const r = await fetch('/api/run', {method:'POST', headers:_hdrs, body:JSON.stringify({steps})});
    const j = await r.json();
    if(!r.ok){
      if(r.status===401){
        const t = prompt('Token di auth richiesto (X-Auth-Token):');
        if(t){ localStorage.setItem('quantsys_auth_token', t); alert('Token salvato — riprova.'); }
        return;
      }
      alert(j.error||'Errore avvio'); return;
    }
  } catch(e){ alert('Errore connessione server'); return; }

  document.getElementById('btn-run').disabled = true;
  openDrawer('running');
  schedulePoll();
}

function openDrawer(mode) {
  const d = document.getElementById('run-drawer');
  d.classList.add('open');
  document.getElementById('drawer-spinner').style.display = mode==='running'?'block':'none';
  document.getElementById('btn-cancel').style.display    = mode==='running'?'inline-flex':'none';
  document.getElementById('btn-close-drawer').style.display = mode!=='running'?'inline-flex':'none';
}

async function cancelRun() {
  const _hdrs = {};
  const _tok = localStorage.getItem('quantsys_auth_token') || '';
  if(_tok) _hdrs['X-Auth-Token'] = _tok;
  try { await fetch('/api/run',{method:'DELETE', headers:_hdrs}); } catch(e){}
}

function schedulePoll() {
  clearTimeout(_pollTimer);
  _pollTimer = setTimeout(pollStatus, 2000);
}

async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    if(!r.ok){ schedulePoll(); return; }
    const s = await r.json();
    applyStatus(s);
    if(s.status==='running') schedulePoll();
  } catch(e) { schedulePoll(); }
}

function applyStatus(s) {
  // Progress bar
  let pct = 0;
  if(s.total_steps>0) pct = ((s.step_idx - (s.status==='running'?0.5:0)) / s.total_steps) * 100;
  document.getElementById('progress-fill').style.width = Math.max(0,Math.min(100,pct))+'%';

  // Meta
  const elapsed = s.elapsed>0 ? `${Math.floor(s.elapsed/60)}m ${Math.round(s.elapsed%60)}s` : '';
  document.getElementById('drawer-meta').textContent = s.step_idx>0 ? `[${s.step_idx}/${s.total_steps}] ${elapsed}` : elapsed;

  // Title + colors
  const title = document.getElementById('drawer-title');
  if(s.status==='running'){
    title.textContent = s.step_label || 'In esecuzione...';
    title.style.color = '#58a6ff';
  } else if(s.status==='success'){
    title.textContent = `✓ Completato in ${elapsed}`;
    title.style.color = '#3fb950';
    document.getElementById('btn-run').disabled = false;
    document.getElementById('btn-run').className = 'run-btn success';
    openDrawer('done');
    loadBacktest(); loadLive();
    if(s.last_success_ts){ const el=document.getElementById('last-run-label'); el.textContent='Aggiornato: '+new Date(s.last_success_ts*1000).toLocaleString('it-IT'); el.style.display=''; }
  } else if(s.status==='error'){
    title.textContent = '✖ Errore: '+(s.error_msg||'sconosciuto');
    title.style.color = '#f85149';
    document.getElementById('btn-run').disabled = false;
    openDrawer('done');
  } else if(s.status==='cancelled'){
    title.textContent = '⚠ Annullato';
    title.style.color = '#d29922';
    document.getElementById('btn-run').disabled = false;
    openDrawer('done');
  }

  // Log
  if(s.log_tail && s.log_tail.length){
    const logEl = document.getElementById('run-log');
    logEl.innerHTML = s.log_tail.slice(-25).map(l => {
      const cls = /\[ERRORE\]|Error|Traceback/i.test(l) ? 'll-err' : /\[OK\]|completat/i.test(l) ? 'll-ok' : /^\[/i.test(l) ? 'll-info' : '';
      return `<div class="${cls}">${l.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
    }).join('');
    logEl.scrollTop = logEl.scrollHeight;
  }
}

// All'avvio: controlla se c'e' una pipeline gia' in esecuzione
(async()=>{
  try {
    const r = await fetch('/api/status');
    if(!r.ok) return;
    const s = await r.json();
    if(s.status==='running'){
      document.getElementById('btn-run').disabled=true;
      openDrawer('running');
      schedulePoll();
    }
    if(s.last_success_ts){
      const el=document.getElementById('last-run-label');
      el.textContent='Aggiornato: '+new Date(s.last_success_ts*1000).toLocaleString('it-IT');
      el.style.display='';
    }
  } catch(e){}
})();
</script>
</body>
</html>"""

# IT: HTML pre-renderizzato una sola volta a startup (no template per-request)
# EN: HTML rendered once at startup (no per-request templating)
_RENDERED_HTML = (HTML
                  .replace("{ARCH}", ARCH_LABEL)
                  .replace("{ARCH_BADGE}", ARCH.upper())
                  ).encode("utf-8")

# IT: ETag stabile fino a restart -> 304 Not Modified su If-None-Match
# EN: stable ETag until restart -> 304 Not Modified on If-None-Match match
import hashlib as _hashlib
_HTML_ETAG = '"' + _hashlib.md5(_RENDERED_HTML).hexdigest()[:16] + '"'

# IT: utility — compressione gzip con timestamp 0 per ETag stabile
# EN: utility — gzip compression with mtime=0 for stable ETag
def _gzip_bytes(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with _gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6, mtime=0) as gz:
        gz.write(payload)
    return buf.getvalue()


# IT: parser query string minimale (evita dipendenza da urllib per latenza)
# EN: minimal query string parser (avoids urllib dependency for latency)
def _parse_qs(path: str) -> dict:
    if "?" not in path:
        return {}
    qs = path.split("?", 1)[1]
    out = {}
    for kv in qs.split("&"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


# IT: tail watcher con cache file_pos per /api/live (no re-read se invariato)
# EN: tail watcher with file_pos cache for /api/live (no re-read if unchanged)
_live_cache_lock = threading.Lock()
_live_cache: dict = {}   # IT: {path: {pos, size, records}} | EN: {path: {pos, size, records}}


# IT: legge tail JSONL incrementalmente — supporta delta sync via since_ts
# EN: incrementally tails JSONL — supports delta sync via since_ts
def _live_tail_cached(live_path, since_ts: float = 0.0, n_tail: int = None) -> list:
    """Tail JSONL con cache file_pos. Restituisce records ordinati per ts asc.

    Se since_ts > 0, filtra solo records con ts > since_ts (delta sync).
    """
    if n_tail is None:
        n_tail = LIVE_TAIL
    key = str(live_path)
    try:
        st = live_path.stat()
        size = st.st_size
    except OSError:
        return []

    with _live_cache_lock:
        cache = _live_cache.get(key)
        if cache is None or size < cache["size"]:
            # IT: prima lettura o file ruotato/truncato -> ri-leggi tail
            # EN: first read or file rotated/truncated -> re-read tail
            cache = {"pos": 0, "size": 0, "records": []}
        elif size == cache["size"]:
            # IT: nessuna crescita -> servi cache (zero I/O)
            # EN: no growth -> serve cache (zero I/O)
            return _filter_since(cache["records"], since_ts, n_tail)

    # IT: file cresciuto -> leggi solo il delta da cache["pos"] a EOF
    # EN: file grew -> read only delta from cache["pos"] to EOF
    new_records = list(cache["records"])
    try:
        with open(live_path, "rb") as f:
            start_pos = cache["pos"]
            if start_pos == 0 and size > 50_000:
                # IT: cold start con file grande -> ultimi ~50KB (truncate partial line)
                # EN: cold start on big file -> last ~50KB (truncate partial line)
                f.seek(-50_000, 2)
                f.readline()  # IT: scarta riga parziale | EN: drop partial line
                start_pos = f.tell()
            else:
                f.seek(start_pos)
            raw = f.read().decode("utf-8", errors="replace")
            end_pos = f.tell()
    except OSError:
        return []

    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            new_records.append(json.loads(ln))
        except json.JSONDecodeError:
            pass

    # Keep only last n_tail × 2 to bound memory
    cap = max(n_tail * 2, 200)
    if len(new_records) > cap:
        new_records = new_records[-cap:]

    with _live_cache_lock:
        _live_cache[key] = {"pos": end_pos, "size": size, "records": new_records}

    return _filter_since(new_records, since_ts, n_tail)


# IT: filtra records per ts > since_ts (delta sync) e limita a n_tail elementi
# EN: filters records by ts > since_ts (delta sync) and caps to n_tail elements
def _filter_since(records: list, since_ts: float, n_tail: int) -> list:
    """Restituisce records con ts > since_ts (se >0), max n_tail elementi."""
    if since_ts > 0.0:
        out = []
        for r in records:
            ts = r.get("ts") or r.get("timestamp")
            try:
                if ts is None or float(ts) > since_ts:
                    out.append(r)
            except (TypeError, ValueError):
                out.append(r)
        return out[-n_tail:]
    return records[-n_tail:]


# IT: request handler HTTP — serve la SPA, gli endpoint /api/* e i controlli pipeline
# EN: HTTP request handler — serves the SPA, /api/* endpoints and pipeline controls
class Handler(BaseHTTPRequestHandler):

    # IT: silenzia il logging HTTP di default per request normali
    # EN: silences default HTTP logging for normal requests
    def log_message(self, fmt, *args):
        pass  # silenzia i log HTTP per request normali

    # IT: true se il client accetta gzip e la compressione è abilitata
    # EN: true if the client accepts gzip and compression is enabled
    def _accepts_gzip(self) -> bool:
        if not ENABLE_GZIP:
            return False
        ae = self.headers.get("Accept-Encoding", "") or ""
        return "gzip" in ae.lower()

    # IT: invia una risposta con CORS localhost + gzip negoziato per body > 1KB
    # EN: sends a response with localhost CORS + negotiated gzip for bodies > 1KB
    def _send(self, code: int, content_type: str, body: bytes, *, allow_gzip: bool = True) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        # CORS: limitato a localhost (B5)
        origin = self.headers.get("Origin", "") or ""
        if origin.startswith("http://localhost") or origin.startswith("http://127.0.0.1"):
            self.send_header("Access-Control-Allow-Origin", origin)
        # Compressione gzip negoziata (C2)
        if allow_gzip and len(body) > 1024 and self._accepts_gzip():
            body = _gzip_bytes(body)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # IT: serializza il dict in JSON e lo invia via _send
    # EN: serializes the dict to JSON and sends it via _send
    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self._send(code, "application/json", body)

    # IT: verifica X-Auth-Token con confronto constant-time (solo se token configurato)
    # EN: checks X-Auth-Token with constant-time compare (only if a token is configured)
    def _auth_ok(self) -> bool:
        """Auth richiesta solo se AUTH_TOKEN è configurato. (B2)"""
        if not AUTH_TOKEN:
            return True
        provided = self.headers.get("X-Auth-Token", "") or ""
        return hmac.compare_digest(provided, AUTH_TOKEN)

    # IT: dispatcher GET — SPA HTML + endpoint /api/* (backtest, live, regime, metriche…)
    # EN: GET dispatcher — SPA HTML + /api/* endpoints (backtest, live, regime, metrics…)
    def do_GET(self):
        path = self.path.split("?")[0]
        qs   = _parse_qs(self.path)
        # IT: arch da query string -> path arch-specifici (default arch corrente)
        # EN: arch from query string -> arch-specific paths (defaults to current arch)
        arch_q = qs.get("arch", "")
        bt_file, live_file = _arch_paths(arch_q) if arch_q else (BACKTEST_FILE, LIVE_FILE)

        # IT: root -> serve la SPA pre-renderizzata con ETag/304 e gzip
        # EN: root -> serves the pre-rendered SPA with ETag/304 and gzip
        if path == "/" or path == "/index.html":
            # C4: 304 Not Modified se ETag combacia
            if self.headers.get("If-None-Match", "") == _HTML_ETAG:
                self.send_response(304)
                self.send_header("ETag", _HTML_ETAG)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("ETag", _HTML_ETAG)
            self.send_header("Cache-Control", "no-cache")  # rivalida sempre via ETag
            body = _RENDERED_HTML
            if self._accepts_gzip() and len(body) > 1024:
                body = _gzip_bytes(body)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # IT: elenca le architetture con backtest disponibile + arch corrente
        # EN: lists architectures with an available backtest + current arch
        elif path == "/api/archs":
            # Restituisce le architetture per cui esiste almeno dashboard_results.json
            avail = []
            for a in ("lstm", "itransformer", "tcnmamba", "tft", "nhits"):
                if (ROOT / "results" / a / "dashboard_results.json").exists():
                    avail.append(a)
            self._send_json(200, {"current": ARCH, "available": avail})

        # IT: serve il JSON di backtest dell'arch (valida la sintassi prima di inviarlo)
        # EN: serves the arch backtest JSON (validates syntax before sending)
        elif path == "/api/backtest":
            if bt_file.exists():
                # B4: valida che sia JSON ben formato prima di servire
                try:
                    data = bt_file.read_bytes()
                    json.loads(data)
                    self._send(200, "application/json", data)
                except json.JSONDecodeError as e:
                    self._send_json(500, {"error": f"backtest JSON corrotto: {e}"})
                except OSError as e:
                    self._send_json(500, {"error": f"lettura fallita: {e}"})
            else:
                self._send(404, "application/json", b'{"error":"not found"}')

        # IT: ultime N osservazioni regime macro + probabilità + regime corrente
        # EN: last N macro-regime observations + probabilities + current regime
        elif path == "/api/regime":
            # A3/G2: ultime N osservazioni regime macro
            reg_path = ROOT / "data" / "regime_probs.parquet"
            if _pd is None:
                self._send_json(500, {"error": "pandas non disponibile"})
                return
            if not reg_path.exists():
                self._send_json(404, {"error": "regime_probs.parquet non trovato"})
                return
            try:
                with _data_lock:
                    df = _pd.read_parquet(reg_path)
                n = int(qs.get("n", "180"))
                tail = df.tail(max(1, min(n, 2000)))
                probs_cols = [c for c in tail.columns if c.startswith("regime_prob_")]
                out_payload = {
                    "dates": [str(i)[:10] for i in tail.index],
                    "dominant": tail["regime_dominant"].astype(int).tolist() if "regime_dominant" in tail else [],
                    "burn_in":  tail["regime_burn_in"].astype(bool).tolist()  if "regime_burn_in"  in tail else [],
                    "probs":    {c: [float(v) for v in tail[c].tolist()] for c in probs_cols},
                    "n_regimes": len(probs_cols),
                    "current": {
                        "date": str(tail.index[-1])[:10],
                        "dominant": int(tail["regime_dominant"].iloc[-1]) if "regime_dominant" in tail else None,
                        "probs":    [float(tail[c].iloc[-1]) for c in probs_cols],
                    },
                }
                self._send_json(200, out_payload)
            except Exception as e:
                self._send_json(500, {"error": f"regime read: {e}"})

        # IT: ultime N osservazioni funding rate + valore corrente annualizzato
        # EN: last N funding-rate observations + current annualized value
        elif path == "/api/funding":
            # A5: ultime N osservazioni funding rate
            fr_path = ROOT / "data" / "funding_rate.parquet"
            if _pd is None:
                self._send_json(500, {"error": "pandas non disponibile"})
                return
            if not fr_path.exists():
                self._send_json(404, {"error": "funding_rate.parquet non trovato"})
                return
            try:
                with _data_lock:
                    df = _pd.read_parquet(fr_path)
                n = int(qs.get("n", "180"))
                tail = df.tail(max(1, min(n, 5000)))
                if "open_time" in tail.columns:
                    ts_vals = [str(t)[:19] for t in tail["open_time"].tolist()]
                else:
                    ts_vals = [str(t)[:19] for t in tail.index]
                fr_vals = [float(v) for v in tail["funding_rate"].tolist()]
                self._send_json(200, {
                    "ts": ts_vals,
                    "funding_rate": fr_vals,
                    "current": {
                        "ts": ts_vals[-1] if ts_vals else None,
                        "value": fr_vals[-1] if fr_vals else None,
                        "annualized_pct": (fr_vals[-1] * 3 * 365 * 100) if fr_vals else None,
                    },
                })
            except Exception as e:
                self._send_json(500, {"error": f"funding read: {e}"})

        # IT: tail dei segnali live con delta sync (?since) + cache file_pos
        # EN: live-signals tail with delta sync (?since) + file_pos cache
        elif path == "/api/live":
            # C3 (2026-05-15): delta sync con ?since=<float ts>; il client passa
            # l'ultimo ts visto, server restituisce solo righe con ts > since.
            # C6: tail watcher in-memory cache: file_pos per evitare seek/read
            # ripetuti se il file non è cresciuto.
            since_q = qs.get("since", "")
            try:
                since_ts = float(since_q) if since_q else 0.0
            except ValueError:
                since_ts = 0.0
            if live_file.exists():
                parsed = _live_tail_cached(live_file, since_ts)
                self._send(200, "application/json", json.dumps(parsed).encode())
            else:
                self._send(200, "application/json", b"[]")

        # IT: metadata per-arch (loss_type, experts, best loss/metriche, scale_label)
        # EN: per-arch metadata (loss_type, experts, best loss/metrics, scale_label)
        elif path == "/api/arch-info":
            # B1 (2026-05-15): metadata per-arch — loss_type, n_output_experts,
            # best_val_loss + scale_label per evitare confronti cross-arch sbagliati.
            arch_q2 = qs.get("arch", ARCH)
            safe = re.sub(r"[^a-z0-9_]", "", arch_q2.lower())
            if safe not in ("lstm", "itransformer", "tcnmamba", "tft", "nhits"):
                safe = ARCH
            cfg_path = ROOT / "models" / safe / "config.json"
            hist_path = ROOT / "models" / safe / "history.json"
            info = {"arch": safe, "exists": cfg_path.exists()}
            try:
                if cfg_path.exists():
                    with open(cfg_path, encoding="utf-8") as f:
                        c = json.load(f)
                    info["loss_type"]        = c.get("loss_type", "t_student")
                    info["n_output_experts"] = int(c.get("n_output_experts", 1))
                    info["best_val_loss"]    = c.get("best_val_loss")
                    info["best_directional_acc"] = c.get("best_directional_acc")
                    info["best_spearman"]    = c.get("best_spearman")
                    info["best_icir"]        = c.get("best_icir")
                    info["d_model"]          = c.get("d_model") or c.get("tft_d_model")
                    info["distilled"]        = bool(c.get("distilled", False))
                    info["teacher_arch"]     = c.get("teacher_arch")
                    info["scale_label"]      = (
                        "quantile pinball" if info["loss_type"] == "quantile"
                        else "t-Student NLL"
                    )
                    # Compute best_val_loss from history if not in config
                    if info["best_val_loss"] is None and hist_path.exists():
                        with open(hist_path, encoding="utf-8") as hf:
                            h = json.load(hf)
                        vl = h.get("val_nll") or h.get("val_loss") or []
                        if vl:
                            info["best_val_loss"] = float(min(vl))
                self._send_json(200, info)
            except Exception as e:
                self._send_json(500, {"error": f"arch-info: {e}"})

        # IT: composizione ensemble eterogeneo — pesi normalizzati + teacher analysis
        # EN: heterogeneous ensemble composition — normalized weights + teacher analysis
        elif path == "/api/ensemble":
            # A7+B5 (2026-05-15): composizione ensemble + pesi + teacher analysis.
            try:
                from quantsys.model.ensemble import (
                    get_distillation_archs, DEFAULT_ARCH_WEIGHTS
                )
                archs_list = get_distillation_archs(_CFG)
                weights = {a: float(DEFAULT_ARCH_WEIGHTS.get(a, 0.0)) for a in archs_list}
                ws = sum(weights.values()) or 1.0
                weights_norm = {a: w / ws for a, w in weights.items()}
                # Optional teacher analysis
                teacher_info = None
                ta_path = ROOT / "models" / "teacher_analysis.json"
                if ta_path.exists():
                    with open(ta_path, encoding="utf-8") as tf:
                        teacher_info = json.load(tf)
                # Per-member metadata
                members = []
                for a in archs_list:
                    cp = ROOT / "models" / a / "config.json"
                    member = {"arch": a, "weight": weights_norm.get(a, 0.0)}
                    if cp.exists():
                        try:
                            with open(cp, encoding="utf-8") as cf:
                                cc = json.load(cf)
                            member["loss_type"]        = cc.get("loss_type", "t_student")
                            member["n_output_experts"] = int(cc.get("n_output_experts", 1))
                            member["distilled"]        = bool(cc.get("distilled", False))
                            member["teacher_arch"]     = cc.get("teacher_arch")
                            member["best_val_loss"]    = cc.get("best_val_loss")
                            member["best_icir"]        = cc.get("best_icir")
                        except Exception:
                            pass
                    members.append(member)
                self._send_json(200, {
                    "archs": archs_list,
                    "weights": weights_norm,
                    "members": members,
                    "teacher_analysis": teacher_info,
                })
            except Exception as e:
                self._send_json(500, {"error": f"ensemble: {e}"})

        # IT: salute del training — curve loss/gap/grad-norm + calibrazione + reliability PNG
        # EN: training health — loss/gap/grad-norm curves + calibration + reliability PNG
        elif path == "/api/training-health":
            # B2+B6 (2026-05-15): ECE + grad norm + gap + reliability PNG.
            arch_q2 = qs.get("arch", ARCH)
            safe = re.sub(r"[^a-z0-9_]", "", arch_q2.lower())
            if safe not in ("lstm", "itransformer", "tcnmamba", "tft", "nhits"):
                safe = ARCH
            mdir = ROOT / "models" / safe
            out_payload = {"arch": safe}
            try:
                hist_path = mdir / "history.json"
                if hist_path.exists():
                    with open(hist_path, encoding="utf-8") as f:
                        h = json.load(f)
                    out_payload["train_nll"]        = h.get("train_nll", [])
                    out_payload["val_nll"]          = h.get("val_nll", [])
                    out_payload["gap"]              = h.get("gap", [])
                    out_payload["grad_norm_mean"]   = h.get("grad_norm_mean", [])
                    out_payload["grad_norm_p95"]    = h.get("grad_norm_p95", [])
                    out_payload["val_dir_acc"]      = h.get("val_dir_acc", [])
                cal_path = mdir / "calibration.json"
                if cal_path.exists():
                    with open(cal_path, encoding="utf-8") as f:
                        cal = json.load(f)
                    out_payload["calibration"] = cal
                rel_path = mdir / "reliability_diagram.png"
                if rel_path.exists():
                    import base64
                    out_payload["reliability_png_b64"] = base64.b64encode(
                        rel_path.read_bytes()).decode("ascii")
                self._send_json(200, out_payload)
            except Exception as e:
                self._send_json(500, {"error": f"training-health: {e}"})

        # IT: confronto scale-free tra architetture (DA, IC, ICIR, Sharpe) — no val_nll
        # EN: scale-free cross-arch comparison (DA, IC, ICIR, Sharpe) — excludes val_nll
        elif path == "/api/arch-comparison":
            # A2 (2026-05-15): confronto SCALE-FREE tra arch (DA, IC, ICIR, Sharpe).
            # NON include val_nll perché incomparabile cross-loss-type.
            try:
                results = []
                for a in ("itransformer", "nhits", "tcnmamba", "lstm", "tft"):
                    arch_dir = ROOT / "models" / a
                    cfg_path = arch_dir / "config.json"
                    hist_path = arch_dir / "history.json"
                    bt_path  = ROOT / "results" / a / "dashboard_results.json"
                    if not cfg_path.exists() and not bt_path.exists():
                        continue
                    entry = {"arch": a}
                    if cfg_path.exists():
                        with open(cfg_path, encoding="utf-8") as f:
                            c = json.load(f)
                        entry["loss_type"]   = c.get("loss_type", "t_student")
                        entry["n_experts"]   = int(c.get("n_output_experts", 1))
                        entry["distilled"]   = bool(c.get("distilled", False))
                        entry["dir_acc"]     = c.get("best_directional_acc")
                        entry["spearman"]    = c.get("best_spearman")
                        entry["icir"]        = c.get("best_icir")
                    # Best DA/spearman from history if not in config
                    if hist_path.exists() and (entry.get("dir_acc") is None or entry.get("spearman") is None):
                        with open(hist_path, encoding="utf-8") as f:
                            h = json.load(f)
                        da = h.get("val_dir_acc", [])
                        sp = h.get("val_spearman", [])
                        if da and entry.get("dir_acc") is None:
                            entry["dir_acc"] = float(max(da))
                        if sp and entry.get("spearman") is None:
                            entry["spearman"] = float(max(sp))
                    # Backtest metrics (scale-free)
                    if bt_path.exists():
                        try:
                            with open(bt_path, encoding="utf-8") as f:
                                bt = json.load(f)
                            m = bt.get("metrics", {})
                            entry["sharpe"]      = m.get("sharpe")
                            entry["sortino"]     = m.get("sortino")
                            entry["max_dd"]      = m.get("max_drawdown")
                            entry["win_rate"]    = m.get("win_rate")
                            entry["profit_factor"] = m.get("profit_factor")
                            entry["n_trades"]    = m.get("n_trades")
                        except Exception:
                            pass
                    results.append(entry)
                self._send_json(200, {"archs": results})
            except Exception as e:
                self._send_json(500, {"error": f"arch-comparison: {e}"})

        # IT: metriche walk-forward fold-by-fold + aggregate (sanifica NaN/Inf per JS)
        # EN: walk-forward metrics fold-by-fold + aggregate (sanitizes NaN/Inf for JS)
        elif path == "/api/walkforward":
            # A6 (2026-05-15): walk-forward metrics fold-by-fold + aggregate.
            arch_q2 = qs.get("arch", ARCH)
            safe = re.sub(r"[^a-z0-9_]", "", arch_q2.lower())
            if safe not in ("lstm", "itransformer", "tcnmamba", "tft", "nhits"):
                safe = ARCH
            wf_path = ROOT / "results" / safe / "walkforward_metrics.json"
            if not wf_path.exists():
                self._send_json(404, {"error": f"walkforward_metrics.json non trovato per {safe}"})
                return
            try:
                # Read + sanitize NaN/Infinity which JS JSON.parse refuses
                raw = wf_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                # IT: ricorsivo — sostituisce NaN/Inf con None (JSON.parse JS li rifiuta)
                # EN: recursive — replaces NaN/Inf with None (JS JSON.parse rejects them)
                def _sanitize(obj):
                    if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
                    if isinstance(obj, list): return [_sanitize(v) for v in obj]
                    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == -float("inf")):
                        return None
                    return obj
                self._send_json(200, _sanitize(data))
            except Exception as e:
                self._send_json(500, {"error": f"walkforward: {e}"})

        # IT: esporta i trade del backtest come CSV scaricabile (analisi esterna)
        # EN: exports backtest trades as a downloadable CSV (external analysis)
        elif path == "/api/export-trades.csv":
            # A10 (2026-05-15): export CSV trades per analisi esterna.
            try:
                bt_q = bt_file
                if not bt_q.exists():
                    self._send_json(404, {"error": "dashboard_results.json non trovato"})
                    return
                with open(bt_q, encoding="utf-8") as f:
                    bt = json.load(f)
                trades = bt.get("trades", [])
                if not trades:
                    body = b"side,entry_price,exit_price,size_usd,hold_candles,close_reason,net_pnl,pnl_pct\n"
                else:
                    headers = ["side","entry_price","exit_price","size_usd","hold_candles",
                               "close_reason","net_pnl","pnl_pct"]
                    rows = [",".join(headers)]
                    for t in trades:
                        rows.append(",".join(str(t.get(h, "")) for h in headers))
                    body = ("\n".join(rows) + "\n").encode("utf-8")
                # Force file download
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="trades.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._send_json(500, {"error": f"export-trades: {e}"})

        # IT: performance per regime macro — join segnali live↔regime via merge_asof
        # EN: performance conditioned on macro regime — joins live signals↔regime via merge_asof
        elif path == "/api/regime-perf":
            # A11 (2026-05-15): performance condizionata al regime macro.
            # I trade in dashboard_results.json NON hanno timestamp, ma
            # live_signals.jsonl ce l'ha. Joinamo i segnali live con regime_probs
            # per timestamp; aggreghiamo per regime_dominant.
            if _pd is None:
                self._send_json(500, {"error": "pandas non disponibile"})
                return
            try:
                reg_path = ROOT / "data" / "regime_probs.parquet"
                if not reg_path.exists():
                    self._send_json(404, {"error": "regime_probs.parquet non trovato"})
                    return
                lv = live_file
                if not lv.exists():
                    self._send_json(404, {"error": "live_signals.jsonl non trovato — regime-perf richiede dati live"})
                    return
                with _data_lock:
                    df_reg = _pd.read_parquet(reg_path)
                if not isinstance(df_reg.index, _pd.DatetimeIndex):
                    tcol = next((c for c in ("open_time","timestamp","date") if c in df_reg.columns), None)
                    if tcol is not None:
                        df_reg = df_reg.set_index(_pd.to_datetime(df_reg[tcol])).sort_index()
                else:
                    df_reg = df_reg.sort_index()
                if "regime_dominant" not in df_reg.columns:
                    self._send_json(500, {"error": "regime_dominant assente"})
                    return
                # Read live signals
                records = []
                with open(lv, encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln: continue
                        try:
                            records.append(json.loads(ln))
                        except json.JSONDecodeError:
                            pass
                if not records:
                    self._send_json(200, {"per_regime": [], "n_records": 0})
                    return
                df_lv = _pd.DataFrame(records)
                ts_col = "ts" if "ts" in df_lv.columns else ("timestamp" if "timestamp" in df_lv.columns else None)
                if ts_col is None:
                    self._send_json(500, {"error": "ts/timestamp assente nei live signals"})
                    return
                # ts può essere unix epoch float o stringa
                df_lv["_t"] = _pd.to_datetime(df_lv[ts_col], unit="s", errors="coerce")
                if df_lv["_t"].isna().all():
                    df_lv["_t"] = _pd.to_datetime(df_lv[ts_col], errors="coerce")
                df_lv = df_lv.dropna(subset=["_t"]).sort_values("_t")
                # merge_asof: per ogni segnale, ultimo regime noto
                df_lv["_t_naive"] = df_lv["_t"].dt.tz_localize(None) if df_lv["_t"].dt.tz else df_lv["_t"]
                reg_naive = df_reg.copy()
                reg_naive.index = reg_naive.index.tz_localize(None) if getattr(reg_naive.index, "tz", None) else reg_naive.index
                merged = _pd.merge_asof(
                    df_lv.sort_values("_t_naive"),
                    reg_naive[["regime_dominant"]].reset_index().rename(columns={reg_naive.index.name or "index": "_t_naive"}),
                    on="_t_naive", direction="backward"
                )
                # Calcola metriche per regime: hit rate (signal == direction realized),
                # n records, prob_up media, sigma media
                per_regime = []
                for reg, grp in merged.groupby("regime_dominant"):
                    if _pd.isna(reg):
                        continue
                    row = {"regime": int(reg), "n": int(len(grp))}
                    if "prob_up" in grp.columns:
                        row["prob_up_mean"] = float(grp["prob_up"].dropna().mean())
                    if "sigma" in grp.columns:
                        row["sigma_mean"] = float(grp["sigma"].dropna().mean())
                    if "signal" in grp.columns:
                        sigs = grp["signal"].dropna().astype(str)
                        row["n_long"]  = int((sigs == "LONG").sum())
                        row["n_short"] = int((sigs == "SHORT").sum())
                        row["n_flat"]  = int(((sigs != "LONG") & (sigs != "SHORT")).sum())
                    per_regime.append(row)
                self._send_json(200, {"per_regime": per_regime, "n_records": int(len(merged))})
            except Exception as e:
                self._send_json(500, {"error": f"regime-perf: {e}"})

        # IT: stato corrente del job pipeline (status, step, elapsed, coda log)
        # EN: current pipeline-job state (status, step, elapsed, log tail)
        elif path == "/api/status":
            with _job_lock:
                elapsed = _job["elapsed"]
                if _job["status"] == "running" and _job["started_at"] is not None:
                    elapsed = time.time() - _job["started_at"]
                payload = {
                    "status": _job["status"],
                    "step_label": _job["step_label"],
                    "step_idx": _job["step_idx"],
                    "total_steps": _job["total_steps"],
                    "elapsed": round(elapsed, 1),
                    "error_msg": _job["error_msg"],
                    "log_tail": list(_job["log_lines"])[-30:],
                    "last_success_ts": _job["last_success_ts"],
                }
            self._send_json(200, payload)

        else:
            self._send(404, "text/plain", b"Not found")

    # IT: dispatcher POST — avvia la pipeline (/api/run) in un worker thread
    # EN: POST dispatcher — starts the pipeline (/api/run) on a worker thread
    def do_POST(self):
        path = self.path.split("?")[0]

        # IT: valida steps e lancia _run_job in background (409 se già in esecuzione)
        # EN: validates steps and launches _run_job in background (409 if already running)
        if path == "/api/run":
            if not self._auth_ok():
                self._send_json(401, {"error": "Unauthorized — X-Auth-Token mancante o errato"})
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "JSON non valido"})
                return

            steps = data.get("steps", [])
            if not steps:
                self._send_json(400, {"error": "Nessuno step specificato"})
                return

            invalid = [s for s in steps if s not in PIPELINE_STEPS]
            if invalid:
                self._send_json(400, {"error": f"Step non riconosciuti: {invalid}"})
                return

            with _job_lock:
                if _job["status"] == "running":
                    self._send_json(409, {"error": "Pipeline gia' in esecuzione"})
                    return
                _job["status"] = "running"

            t = threading.Thread(target=_run_job, args=(steps,), daemon=True)
            t.start()
            log.info(f"Pipeline avviata con steps: {steps}")
            self._send_json(200, {"ok": True})

        else:
            self._send(404, "text/plain", b"Not found")

    # IT: dispatcher DELETE — annulla la pipeline in esecuzione (/api/run)
    # EN: DELETE dispatcher — cancels the running pipeline (/api/run)
    def do_DELETE(self):
        global _process
        path = self.path.split("?")[0]

        # IT: segna il job come cancelled e termina il subprocess corrente
        # EN: marks the job as cancelled and terminates the current subprocess
        if path == "/api/run":
            if not self._auth_ok():
                self._send_json(401, {"error": "Unauthorized — X-Auth-Token mancante o errato"})
                return
            with _job_lock:
                if _job["status"] != "running":
                    self._send_json(400, {"error": "Nessuna pipeline in esecuzione"})
                    return
                _job["status"] = "cancelled"
                _job["log_lines"].append("[INFO] Annullamento richiesto dall'utente.")
                proc = _process

            if proc is not None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception:
                    pass

            log.info("Pipeline annullata dall'utente.")
            self._send_json(200, {"ok": True})

        else:
            self._send(404, "text/plain", b"Not found")


# ─── Threaded server ──────────────────────────────────────────────────────────

# IT: server HTTP multi-thread (un thread daemon per connessione)
# EN: multi-threaded HTTP server (one daemon thread per connection)
class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ─── Entry point ──────────────────────────────────────────────────────────────

# IT: entry point — avvia il server, logga la config e gestisce Ctrl+C
# EN: entry point — starts the server, logs the config and handles Ctrl+C
def main():
    server = ThreadedHTTPServer((HOST, PORT), Handler)
    _shown_host = "localhost" if HOST in ("127.0.0.1", "localhost") else HOST
    log.info(f"Dashboard avviata -> http://{_shown_host}:{PORT}  (bind={HOST})")
    log.info(f"  Arch:     {ARCH}  ({ARCH_LABEL})")
    log.info(f"  Models:   {MODELS_DIR}")
    log.info(f"  Backtest: {BACKTEST_FILE}")
    log.info(f"  Live:     {LIVE_FILE}")
    log.info(f"  Auth:     {'ATTIVA (X-Auth-Token)' if AUTH_TOKEN else 'disattivata'}  "
             f"Gzip: {'on' if ENABLE_GZIP else 'off'}  "
             f"Timeout step: {SUBPROC_TIMEOUT}s")
    log.info("Premi Ctrl+C per uscire.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Dashboard fermata.")
        server.server_close()


if __name__ == "__main__":
    main()
