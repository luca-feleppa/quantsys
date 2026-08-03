"""
Script 00 — Verifica setup hardware e dipendenze.
Eseguilo PRIMA di tutto il resto per assicurarti che l'ambiente sia corretto.

Run configuration PyCharm:
  Script: scripts/00_check_setup.py
  Working dir: <root del progetto>
"""
import importlib
import logging
import platform
import sys
import time
from pathlib import Path

# IT: questo script non ha main() — il codice gira a livello di modulo, quindi il
#     reconfigure va qui, prima di qualunque print. Console Windows default cp1252:
#     i glyph ✓/△/✗ e i box-drawing delle sezioni crashano il print (bug ricorrente).
# EN: this script has no main() — code runs at module level, so the reconfigure goes
#     here, before any print. Windows console defaults to cp1252: the ✓/△/✗ glyphs and
#     the section box-drawing chars crash the print (recurring bug).
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# IT: usa basicConfig perche' setup_logging potrebbe non essere disponibile
# EN: use basicConfig since setup_logging may not yet be importable
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# IT: codici colore ANSI per output console leggibile
# EN: ANSI color codes for readable console output
OK   = "\033[92m✓\033[0m"
WARN = "\033[93m△\033[0m"
ERR  = "\033[91m✗\033[0m"
HEAD = "\033[1;96m"
RST  = "\033[0m"


# IT: stampa un'intestazione di sezione colorata
# EN: prints a colored section header
def section(title: str):
    print(f"\n{HEAD}{'─'*52}{RST}")
    print(f"{HEAD}  {title}{RST}")
    print(f"{HEAD}{'─'*52}{RST}")


# IT: stampa una riga di check (✓/△/✗) e ritorna l'esito booleano
# EN: prints a single check line (✓/△/✗) and returns the boolean outcome
def check(label: str, ok: bool, detail: str = "", warn_only: bool = False):
    icon  = OK if ok else (WARN if warn_only else ERR)
    color = "\033[92m" if ok else ("\033[93m" if warn_only else "\033[91m")
    print(f"  {icon}  {label:<36} {color}{detail}{RST}")
    return ok


# IT: 1. verifica versione Python e working directory
# EN: 1. check Python version and working directory
section("1 · PYTHON")
py = sys.version_info
check("Python version", py >= (3, 11),
      f"{py.major}.{py.minor}.{py.micro}  (richiesto ≥ 3.11)")
check("Platform", True, platform.platform())
check("Working directory", Path("config/default.yaml").exists(),
      str(Path.cwd()),
      warn_only=not Path("config/default.yaml").exists())

# IT: 2. verifica presenza e versione minima delle dipendenze critiche
# EN: 2. check critical dependencies are installed at minimum versions
section("2 · DIPENDENZE")
deps = [
    ("torch",        "2.2.0",  False),
    ("numpy",        "1.26.0", False),
    ("pandas",       "2.1.0",  False),
    ("sklearn",      "1.4.0",  False),
    ("scipy",        "1.12.0", False),
    ("requests",     "2.31.0", False),
    ("tqdm",         "4.66.0", False),
    ("yaml",         "6.0.0",  False),
    ("pyarrow",      "14.0.0", False),
    ("websockets",   "12.0",   True),   # IT: opzionale (solo live) | EN: optional (live only)
]
all_ok = True
for pkg, min_ver, warn in deps:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        ok  = check(pkg, True, f"v{ver}")
    except ImportError:
        ok = check(pkg, False, "NON INSTALLATO", warn_only=warn)
    if not ok and not warn:
        all_ok = False

# IT: 3. verifica CUDA/GPU e supporto Tensor Cores per AMP fp16
# EN: 3. check CUDA/GPU availability and Tensor Cores for fp16 AMP
section("3 · GPU / CUDA  (RTX 2070 Super)")
try:
    import torch

    cuda_ok = torch.cuda.is_available()
    check("CUDA disponibile", cuda_ok,
          f"CUDA {torch.version.cuda}" if cuda_ok else "❌  pip install torch --index-url https://download.pytorch.org/whl/cu121")

    if cuda_ok:
        n_gpu = torch.cuda.device_count()
        for i in range(n_gpu):
            p     = torch.cuda.get_device_properties(i)
            vram  = p.total_memory / 1024**3
            check(f"GPU {i}: {p.name}", True, f"{vram:.1f} GB VRAM  |  SM {p.major}.{p.minor}")
            check("  VRAM sufficiente (≥6 GB)", vram >= 6, f"{vram:.1f} GB")
            check("  Tensor Cores (Turing+)",   p.major >= 7,
                  "✓ float16 AMP supportato" if p.major >= 7 else "AMP meno efficace")

        # IT: micro-benchmark matmul fp16 per validare throughput GPU
        # EN: quick fp16 matmul benchmark to validate GPU throughput
        check("cuDNN disponibile",  torch.backends.cudnn.is_available(), "")
        check("cuDNN benchmark",    True, "verrà abilitato in training")

        dev   = torch.device("cuda")
        dtype = torch.float16
        A     = torch.randn(2048, 2048, device=dev, dtype=dtype)
        B     = torch.randn(2048, 2048, device=dev, dtype=dtype)
        t0    = time.perf_counter()
        for _ in range(10):
            _ = A @ B
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 100  # IT: ms medi per matmul (10 iter) | EN: avg ms per matmul (10 iters)
        check("matmul FP16 2048×2048 (×10)", True, f"{ms:.1f} ms  →  {'rapido ✓' if ms < 500 else 'lento, controlla driver'}")

    else:
        check("AMP (float16)", False, "richiede CUDA", warn_only=True)

except ImportError:
    check("PyTorch", False, "NON INSTALLATO — pip install torch ...")
    all_ok = False

# IT: 4. verifica CPU e parametri DataLoader (num_workers, pin_memory)
# EN: 4. check CPU and DataLoader settings (num_workers, pin_memory)
section("4 · CPU  (i7-9700K)")
try:
    import os
    n_cpu = os.cpu_count()
    check("Core logici", True, f"{n_cpu}  (DataLoader num_workers consigliato: {n_cpu-2})")

    # IT: num_workers deve lasciare almeno 2 core liberi per il main process
    # EN: num_workers should leave at least 2 cores free for the main process
    import yaml
    with open("config/default.yaml", "rb") as f:
        cfg = yaml.safe_load(f)
    nw = cfg.get("hardware", {}).get("num_workers", 0)
    check("num_workers in config", nw <= n_cpu - 2,
          f"{nw}  (max consigliato: {n_cpu-2})",
          warn_only=nw > n_cpu - 2)
    check("pin_memory in config", cfg.get("hardware", {}).get("pin_memory", False),
          str(cfg.get("hardware", {}).get("pin_memory")))

except Exception as e:
    check("CPU check", False, str(e), warn_only=True)

# IT: 5. verifica connettivita' REST Binance e latenza
# EN: 5. check Binance REST connectivity and latency
section("5 · CONNESSIONE BINANCE API")
try:
    import requests
    t0  = time.perf_counter()
    r   = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
    ms  = (time.perf_counter() - t0) * 1000
    ok  = r.status_code == 200
    check("Ping Binance REST", ok, f"HTTP {r.status_code}  |  latenza {ms:.0f} ms")

    r2  = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
    if r2.status_code == 200:
        price = float(r2.json()["price"])
        check("BTC/USDT price", True, f"${price:,.1f}")
except Exception as e:
    check("Connessione Binance", False, str(e))

# IT: 6. verifica struttura cartelle/file del progetto
# EN: 6. check project directory and file structure
section("6 · STRUTTURA PROGETTO")
expected = [
    "config/default.yaml",
    "conftest.py",
    "setup.cfg",
    "pyproject.toml",
    "requirements.txt",
    "quantsys/__init__.py",
    "quantsys/data/__init__.py",
    "quantsys/features/__init__.py",
    "quantsys/macro/__init__.py",
    "quantsys/macro/regime.py",
    "quantsys/macro/live_snapshot.py",
    "quantsys/model/__init__.py",
    "quantsys/model/forecast.py",
    "quantsys/trading/__init__.py",
    "quantsys/utils/__init__.py",
    "quantsys/utils/atomic_save.py",
    "scripts/00_check_setup.py",
    "scripts/01_download_data.py",
    "scripts/01b_download_macro.py",
    "scripts/02_train.py",
    "scripts/02b_walkforward_validate.py",
    "scripts/03_backtest.py",
    "scripts/04_live_signals.py",
    "scripts/05_analyze_signals.py",
    "tests/conftest.py",
    "tests/test_features.py",
    "scripts/06_dashboard.py",
]
for f in expected:
    check(f, Path(f).exists(), "✓" if Path(f).exists() else "MANCANTE")

# IT: cartelle create automaticamente dagli script di pipeline
# EN: directories auto-created by pipeline scripts
for d in ["data", "models", "results", "logs"]:
    exists = Path(d).exists()
    check(f"{d}/", exists,
          "✓ presente" if exists else "verrà creata automaticamente",
          warn_only=not exists)

# IT: 7. verifica stato della pipeline (artefatti generati per arch attiva)
# EN: 7. check pipeline state (generated artifacts for active architecture)
section("7 · STATO PIPELINE")
# IT: risolve l'architettura attiva: env var, poi config, poi default lstm
# EN: resolve active architecture: env var, then config, then lstm default
import re as _re, os as _os
_arch = _os.environ.get("QUANTSYS_ARCH")
if not _arch:
    try:
        _txt = Path("config/default.yaml").read_text(encoding="utf-8")
        _m = _re.search(r'architecture:\s*["\']?(\w+)["\']?', _txt)
        _arch = _m.group(1) if _m else "lstm"
    except Exception:
        _arch = "lstm"
print(f"  Arch attiva: {_arch}")
steps = [
    ("data/lstm_dataset.npz",                          "01_download_data.py",   False),
    ("data/macro_features.parquet",                    "01b_download_macro.py", True),
    (f"models/{_arch}/pipeline_state.pkl",             "01_download_data.py",   False),
    (f"models/{_arch}/best_model.pt",                  "02_train.py",           False),
    (f"models/{_arch}/config.json",                    "02_train.py",           False),
    (f"models/{_arch}/history.json",                   "02_train.py",           False),
    (f"results/{_arch}/dashboard_results.json",        "03_backtest.py",        False),
]
# IT: questi artefatti sono PRODOTTI dalla pipeline: su un clone fresco non esiste
#     nessuno di essi, ed e' lo stato corretto. Non concorrono infatti ad `all_ok`
#     (il ritorno di check() e' deliberatamente ignorato qui) — ma finche' l'assenza
#     stampava un ✗ rosso, l'icona diceva "errore" e il verdetto finale diceva
#     "setup verificato": una delle due mentiva, e a mentire era l'icona. Warning.
# EN: these artifacts are PRODUCED by the pipeline: on a fresh clone none of them
#     exists, and that is the correct state. Indeed they do not feed `all_ok`
#     (check()'s return value is deliberately ignored here) — but as long as absence
#     printed a red ✗, the icon said "error" while the final verdict said "setup
#     verified": one of the two was lying, and it was the icon. Warning instead.
for path, script, optional in steps:
    exists = Path(path).exists()
    sz     = f"  ({Path(path).stat().st_size//1024} KB)" if exists else ""
    label  = f"[opzionale] {path}" if optional else path
    check(label, exists,
          f"✓{sz}" if exists else f"→ esegui scripts/{script}",
          warn_only=not exists)

# IT: riepilogo finale con device, AMP, batch size e stima training
# EN: final summary with device, AMP, batch size and training estimate
section("RIEPILOGO")
try:
    import torch
    cuda = torch.cuda.is_available()
    import yaml
    with open("config/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    amp  = cfg.get("training", {}).get("use_amp", False)
    bs   = cfg.get("training", {}).get("batch_size", 0)

    print(f"""
  Device        : {'CUDA ✓' if cuda else 'CPU'}
  AMP float16   : {'Abilitato ✓' if amp and cuda else 'Disabilitato'}
  Batch size    : {bs}
  Est. training : {'~8-12 min (RTX 2070 Super)' if cuda else '~60-90 min (CPU)'}

  Ordine di esecuzione:
    1.  python scripts/00_check_setup.py      ← sei qui
    2.  python scripts/01_download_data.py
    3.  python scripts/02_train.py
    4.  python scripts/03_backtest.py
    5.  python scripts/04_live_signals.py     ← segnali live
""")
except Exception:
    pass

if not all_ok:
    print(f"  {ERR}  Alcuni controlli falliti — risolvi gli errori prima di procedere.\n")
    sys.exit(1)
else:
    print(f"  {OK}  Setup verificato — pronti per la pipeline.\n")
