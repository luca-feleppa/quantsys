# QUANTSYS — Guida operativa di avvio · QUANTSYS — Operations & quick-start guide

🇮🇹 **Runbook operativo** del motore di forecasting BTC/USDT (`quantsys/`). Timeframe production: **candele 1h** (`data.interval: 1h`); il motore è **interval-agnostic** (tutte le conversioni di finestra sono identità a 1m). Questo file possiede i **comandi**: la derivazione matematica sta in `TEORIA.md`, la panoramica/motivazione in `README.md`, lo stato corrente e i gate aperti in `STATUS.md`.

**EN** **Operations runbook** for the BTC/USDT forecasting engine (`quantsys/`). Production timeframe: **1h candles** (`data.interval: 1h`); the engine is **interval-agnostic** (every window conversion is an identity at 1m). This file owns the **commands**: mathematical derivations live in `TEORIA.md`, the overview/motivation in `README.md`, current status and open gates in `STATUS.md`.

## Indice · Map

🇮🇹 Guida ordinata per **fase operativa**: **0. Stato & comandi rapidi** → **1. Setup e dipendenze** (installazione, Windows/PowerShell, hardware, parametri interval) → **2. Dati** → **3. Modellazione** (train / walk-forward / Optuna / distillation / CAFN) → **4. Valutazione** (backtest / verify / vol-judge / short-vol) → **5. Deploy & inferenza** (routine di sessione, live, vol-paper, collector forward, dashboard) → **Appendice: file layout**. Tutti i comandi si lanciano dalla **root** `E:\quantsys_project`.

**EN** Guide ordered by **operational phase**: **0. Status & quick commands** → **1. Setup & dependencies** (install, Windows/PowerShell, hardware, interval parameters) → **2. Data** → **3. Modeling** (train / walk-forward / Optuna / distillation / CAFN) → **4. Evaluation** (backtest / verify / vol-judge / short-vol) → **5. Deploy & inference** (session routine, live, vol-paper, forward collectors, dashboard) → **Appendix: file layout**. Every command is run from the **root** `E:\quantsys_project`.

---

## 0. Stato & comandi rapidi · Status & quick commands

### 0.1 Comandi rapidi · Quick commands

```bash
python run_all.py                                    # menu interattivo · interactive menu
python run_all.py --arch itransformer                # training singola arch · single-arch training
python run_all.py --arch nhits
python run_all.py --arch tcnmamba
python run_all.py --arch lstm                        # backward compat
python run_all.py --distill                          # Knowledge Distillation multi-teacher
python run_all.py --distill --teacher itransformer   # forza teacher · force teacher
python run_all.py --only-dashboard                   # solo Options Risk Terminal (no ML)
python run_all.py --interval 1h                      # overlay risoluzione candela · candle-resolution overlay
python scripts/07_verify_teacher.py                  # confronto architetture · architecture comparison
python scripts/99_replay_live_vs_training.py         # diagnostica parity live · live parity diagnostic
```

🇮🇹 ⚠ Su disco `models/` contiene **solo `itransformer/` (production vol-1h) e `lstm/` (legacy)**: `models/nhits` e `models/tcnmamba` sono stati eliminati col cleanup 2026-06-12 → **vanno riaddestrati** prima di qualsiasi run eterogeneo (`--distill`, backtest ensemble). Le arch restano valide come `--arch`.

**EN** ⚠ On disk `models/` holds **only `itransformer/` (1h-vol production) and `lstm/` (legacy)**: `models/nhits` and `models/tcnmamba` were removed in the 2026-06-12 cleanup → they **must be retrained** before any heterogeneous run (`--distill`, ensemble backtest). Both archs remain valid `--arch` values.

### 0.2 Stato del sistema · System status

🇮🇹
- **Production = linea vol-1h** (`features.target_type: log_rv`, iTransformer **5 membri** in `models/itransformer/`): unico segnale **PASS OOS** del progetto.
- **Braccio short-vol** in forward test su Deribit testnet via `04b_vol_paper.py` — servizio **systemd sul VPS 24/7**, non a casa (§5.3). Gate v1 (n=20) **FAIL 0/3 il 2026-07-18**; i gate successivi sono in accumulo di campione.
- **Linea direzionale** (`target_type: ret`, `03_backtest.py`/`04_live_signals.py`): **nessun alpha OOS a nessun timeframe** — codice vivo e bit-invariato, tenuto come *negative-control* scientifico. Paper-only, nessun ordine reale.
- **Invariante z-score:** ogni nuovo entry-point deve chiamare `PipelineState.denormalize_predictions(mu, sigma)` prima del trading layer; con target `log_rv` serve l'inversione completa `μ·IQR + centro`. Derivazione in `TEORIA.md` (§ *Invariante critico*).
- ⚠ **Stato corrente, gate aperti e "riparti da qui": `STATUS.md`** (periodo corrente + gate aperti; lo storico ante 2026-07-08 è in `docs/STATUS_ARCHIVE_2026H1.md`, read-only). Questa guida **non** tiene lo storico degli esperimenti.

**EN**
- **Production = the 1h vol line** (`features.target_type: log_rv`, **5-member** iTransformer in `models/itransformer/`): the project's only **OOS PASS** signal.
- **Short-vol arm** in forward test on Deribit testnet via `04b_vol_paper.py` — a **systemd service on the 24/7 VPS**, never at home (§5.3). Gate v1 (n=20) **FAILED 0/3 on 2026-07-18**; later gates are still accumulating sample.
- **Directional line** (`target_type: ret`, `03_backtest.py`/`04_live_signals.py`): **no OOS alpha at any timeframe** — code alive and bit-invariant, kept as a scientific *negative control*. Paper-only, no real orders.
- **z-score invariant:** every new entry point must call `PipelineState.denormalize_predictions(mu, sigma)` before the trading layer; with the `log_rv` target the full inversion is `μ·IQR + center`. Derivation in `TEORIA.md` (§ *Critical invariant*).
- ⚠ **Current status, open gates and "resume here": `STATUS.md`** (current period + open gates; pre-2026-07-08 history in `docs/STATUS_ARCHIVE_2026H1.md`, read-only). This guide does **not** keep the experiment log.

---

## 1. Setup e dipendenze · Setup & dependencies

### 1.1 Installazione · Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python scripts/00_check_setup.py
```

🇮🇹 `00_check_setup.py` controlla dipendenze, CUDA, Binance, FRED — risolvi gli errori prima di proseguire. Il suo § **STATO PIPELINE** è diverso: elenca artefatti **prodotti** dagli script successivi (dataset, checkpoint, report), quindi su un clone fresco sono assenti **per definizione** e compaiono come warning `△`, non come errori — non concorrono al verdetto finale e non c'è nulla da "risolvere" prima di lanciare la pipeline. `00_test_binance_testnet.py` è uno smoke opzionale della connettività al testnet Binance.

**EN** `00_check_setup.py` checks dependencies, CUDA, Binance, FRED — fix errors before proceeding. Its **PIPELINE STATE** section is different: it lists artifacts **produced** by the later scripts (dataset, checkpoints, reports), so on a fresh clone they are missing **by definition** and show up as `△` warnings, not errors — they do not feed the final verdict and there is nothing to "fix" before running the pipeline.  `00_test_binance_testnet.py` is an optional Binance-testnet connectivity smoke.

### 1.2 Note operative su Windows / PowerShell · Windows / PowerShell operational notes

🇮🇹
- **stderr:** **NON** usare `2>&1 | Tee-Object` — PS 5.1 incapsula stderr come ErrorRecord (output rosso fittizio; il logging Python va su stderr). Lo script salva già `logs/quantsys_*.log`; per un file dedicato usa `*> file.log`.
- **Redirect shell-dependent:** `*> file.log` è sintassi **PowerShell**; sotto **bash** `*` viene globbato e il log NON si scrive → lì serve `> file.log 2>&1`. Verifica sempre che il log si riempia prima di considerare avviato un job lungo.
- **Exit code degli exe nativi:** in PS 5.1 un eseguibile nativo che esce con codice **≠0 NON solleva eccezione** → il `try/catch` non basta, controlla `$LASTEXITCODE` esplicitamente dopo la chiamata (pattern usato in `avvio_sessione.ps1`).
- **Encoding dei `.ps1`:** un file senza BOM viene letto come **cp1252** e qualunque carattere unicode corrompe il parsing → tieni gli script PowerShell **ASCII-only**.
- **Quoting verso gli exe nativi:** PS 5.1 strippa le doppie virgolette negli argomenti passati a un eseguibile nativo → nei blocchi `python -c` usa **solo apici singoli**.
- **UTF-8 boilerplate:** ogni nuovo script in `scripts/` deve reconfigurare UTF-8 su stdout/stderr in `main()` (il bug cp1252 è ricorso 5 volte: qualunque unicode nel banner crasha su console Windows).
- **`set` vs `$env:`:** i blocchi `set QUANTSYS_ARCH=...` di questa guida sono sintassi **cmd.exe**; in PowerShell usa `$env:QUANTSYS_ARCH="..."`.
- **Ordine di import — pyarrow prima di torch+sklearn (risolto 2026-08-02).** Caricare pyarrow **dopo** che torch **e** scikit-learn sono già nel processo produce un'**access violation** (exit 139, nessun traceback Python) al primo `pd.read_parquet`: è il conflitto fra i runtime OpenMP che i due portano. Serve la compresenza — torch da solo o sklearn da solo non basta. `quantsys/__init__.py` ancora `import pyarrow` alla radice del package, quindi qualunque `import quantsys.*` inizializza Arrow per primo e il problema non si presenta più (test: `tests/test_import_order.py`). ⚠ Se scrivi codice che importa torch **prima** di qualunque cosa di `quantsys`, l'ancora non ti copre: in quel caso importa `pandas` (o `pyarrow`) in cima, come fanno tutti gli script numerati.

⚠ **Diagnosi rapida:** un processo Python che muore con **exit 139 / access violation senza traceback** subito dopo un `read_parquet` è quasi sempre questo, non un OOM.

**EN**
- **stderr:** do **NOT** use `2>&1 | Tee-Object` — PS 5.1 wraps stderr as an ErrorRecord (fake red output; Python logging goes to stderr). The script already writes `logs/quantsys_*.log`; for a dedicated file use `*> file.log`.
- **Shell-dependent redirect:** `*> file.log` is **PowerShell** syntax; under **bash** `*` is globbed and the log is NOT written → there you need `> file.log 2>&1`. Always verify the log is filling before assuming a long job has started.
- **Native-exe exit codes:** in PS 5.1 a native executable exiting with a **non-zero code does NOT throw** → `try/catch` is not enough, check `$LASTEXITCODE` explicitly after the call (the pattern used in `avvio_sessione.ps1`).
- **`.ps1` encoding:** a BOM-less file is read as **cp1252** and any unicode character corrupts parsing → keep PowerShell scripts **ASCII-only**.
- **Quoting toward native exes:** PS 5.1 strips double quotes in arguments passed to a native executable → inside `python -c` blocks use **single quotes only**.
- **UTF-8 boilerplate:** every new `scripts/` file must reconfigure UTF-8 on stdout/stderr in `main()` (the cp1252 bug recurred 5 times: any unicode in the banner crashes the Windows console).
- **`set` vs `$env:`:** the `set QUANTSYS_ARCH=...` blocks in this guide are **cmd.exe** syntax; in PowerShell use `$env:QUANTSYS_ARCH="..."`.
- **Import order — pyarrow before torch+sklearn (fixed 2026-08-02).** Loading pyarrow **after** both torch **and** scikit-learn are already in the process yields an **access violation** (exit 139, no Python traceback) at the first `pd.read_parquet`: it is the clash between the OpenMP runtimes the two ship. Both are required — torch alone or sklearn alone is fine. `quantsys/__init__.py` anchors `import pyarrow` at the package root, so any `import quantsys.*` initializes Arrow first and the problem no longer occurs (test: `tests/test_import_order.py`). ⚠ If you write code importing torch **before** anything from `quantsys`, the anchor does not cover you: there, import `pandas` (or `pyarrow`) at the top, as every numbered script does.

⚠ **Quick diagnosis:** a Python process dying with **exit 139 / access violation and no traceback** right after a `read_parquet` is almost always this, not an OOM.

### 1.3 Hardware

🇮🇹 **CPU** — `config/default.yaml` (valore corrente `1` = tutti i core; abbassa a 0.5 per lasciare la macchina usabile durante un training lungo):
```yaml
hardware:
  cpu_fraction: 1     # 0.3=30%, 0.5=50%, 1=100% dei core
```
Letto da tutti gli script all'avvio.

**EN** **CPU** — `config/default.yaml` (current value `1` = all cores; lower it to 0.5 to keep the machine usable during a long training): block above. Read by every script at startup.

🇮🇹 **GPU compute** (RTX 2070 Super, min=125 max=215W):
```powershell
nvidia-smi -pl 125    # limita · throttle
nvidia-smi -pl 215    # ripristina · restore
```

🇮🇹 **Sequenzialità GPU (8GB):** backtest/walk-forward in parallelo OK; il training 5-seed × 3 arch va **sequenziale** (OOM). **NON** girare live/paper-trading/vol-paper + training/inferenza in parallelo (contesa CUDA, 5 modelli residenti). TCN+Mamba è il collo di bottiglia.

**EN** **GPU sequencing (8GB):** parallel backtest/walk-forward OK; 5-seed × 3-arch training must be **sequential** (OOM). Do **NOT** run live/paper-trading/vol-paper + training/inference in parallel (CUDA contention, 5 resident models). TCN+Mamba is the bottleneck.

🇮🇹 **Setup di riferimento (RTX 2070 Super 8GB):**

| Componente | Valore |
|---|---|
| CUDA, AMP fp16 training | sì (via `setup_device`) |
| AMP inference | **off** hardcoded in `quantsys/model/ensemble.py` (evita NaN spectral_norm + Mamba scan) |
| Batch inference backtest | 256 (`BATCH_SIZE` in `scripts/03_backtest.py`) |
| Batch training | 64 (`config/default.yaml → training.batch_size`; gli `arch/*.yaml` non lo sovrascrivono) |

**EN** **Reference setup (RTX 2070 Super 8GB):**

| Component | Value |
|---|---|
| CUDA, AMP fp16 training | yes (via `setup_device`) |
| AMP inference | **off** hardcoded in `quantsys/model/ensemble.py` (avoids NaN from spectral_norm + Mamba scan) |
| Backtest inference batch | 256 (`BATCH_SIZE` in `scripts/03_backtest.py`) |
| Training batch | 64 (`config/default.yaml → training.batch_size`; the `arch/*.yaml` files do not override it) |

🇮🇹 **Altre configurazioni:** *solo CPU* → fallback automatico via `setup_device` (`autocast` diventa no-op silenzioso): training 20–50× più lento (sconsigliato), backtest ~5s → 30–60s (tollerabile), live ~50–100ms vs ~20ms (utilizzabile, domina la latency WS Binance). *VRAM 4GB* → in `config/default.yaml → training` (o override in `config/arch/<arch>.yaml`) `batch_size: 32` + `gradient_accumulation_steps: 2` (effective batch 64) e inference batch 256→128. *VRAM ≥16GB* → `batch_size: 128`, inference batch fino a 1024 (guadagno marginale). **Apple Silicon / AMD / Intel Arc: non testato** (codice `torch.cuda.*`; MPS richiederebbe modifiche a `setup_device` + kernel custom Mamba/SSM).

**EN** **Other configurations:** *CPU only* → automatic fallback via `setup_device` (`autocast` becomes a silent no-op): training 20–50× slower (not recommended), backtest ~5s → 30–60s (tolerable), live ~50–100ms vs ~20ms (usable, Binance WS latency dominates). *4GB VRAM* → in `config/default.yaml → training` (or overridden in `config/arch/<arch>.yaml`) `batch_size: 32` + `gradient_accumulation_steps: 2` (effective batch 64) and inference batch 256→128. *≥16GB VRAM* → `batch_size: 128`, inference batch up to 1024 (marginal gain). **Apple Silicon / AMD / Intel Arc: untested** (`torch.cuda.*` code; MPS would need `setup_device` changes + custom Mamba/SSM kernels).

### 1.4 Parametri interval 1h & rollback 1m · 1h interval parameters & 1m rollback

🇮🇹 Valori correnti in `config/default.yaml` (verificati sul file). La colonna **era (1m)** è **nota storica**: serve solo a spiegare la calibrazione corrente e al rollback.

| Parametro (sezione) | Valore 1h | Era (1m) | Nota |
|---|---|---|---|
| `data.interval` | `1h` | `1m` | tutte le finestre TIME-semantic derivano da qui via `interval_minutes` |
| `data.start_time` | `2019-01-01` | `2025-05-19` | storico multi-anno, ~65k barre |
| `model.window_size` | 120 | 120 | invariato in barre = **5 giorni** di contesto (a 1m erano 2h) |
| `model.window_stride` | 1 | 5 | massimizza i sample su ~65k barre |
| `features.forecast_horizon` | 30 | 30 | invariato in barre = **30 ORE** (a 1m erano 30 min) |
| `validation.embargo_steps` | 168 (1 settimana) | 1500 (~25h) | vincolo ≥ `window_size + horizon` = 150 |
| `risk.max_hold_candles` | 60 (2.5 giorni) | 240 (4h) | vincolo ≥ `forecast_horizon` = 30 |
| `backtest.min_expected_ret` | 0.0013 (13 bps) | 0.0005 | gate cost-aware, **solo linea direzionale** |
| `backtest.max_sigma` | 0.10 (≈0.015·√60) | 0.015 | soglia direzionale, da ricalibrare sui percentili post-denorm |
| `montecarlo.gjr_*` | ω 1.026e-06 · α 0.1011 · γ 0.0052 · β 0.8732 · cap 0.13 | ω 1.2e-05 · α 0.05 · γ 0.065 · β 0.875 · cap 0.01 | ri-stimati su 1h il 2026-07-15; i valori 1m sono in `config/interval/1m.yaml` |

**EN** Current values in `config/default.yaml` (checked against the file). The **was (1m)** column is a **historical note**: it only explains the current calibration and serves the rollback.

| Parameter (section) | 1h value | Was (1m) | Note |
|---|---|---|---|
| `data.interval` | `1h` | `1m` | every TIME-semantic window derives from it via `interval_minutes` |
| `data.start_time` | `2019-01-01` | `2025-05-19` | multi-year history, ~65k bars |
| `model.window_size` | 120 | 120 | unchanged in bars = **5 days** of context (2h at 1m) |
| `model.window_stride` | 1 | 5 | maximizes samples over ~65k bars |
| `features.forecast_horizon` | 30 | 30 | unchanged in bars = **30 HOURS** (30 min at 1m) |
| `validation.embargo_steps` | 168 (1 week) | 1500 (~25h) | constraint ≥ `window_size + horizon` = 150 |
| `risk.max_hold_candles` | 60 (2.5 days) | 240 (4h) | constraint ≥ `forecast_horizon` = 30 |
| `backtest.min_expected_ret` | 0.0013 (13 bps) | 0.0005 | cost-aware gate, **directional line only** |
| `backtest.max_sigma` | 0.10 (≈0.015·√60) | 0.015 | directional threshold, to recalibrate on post-denorm percentiles |
| `montecarlo.gjr_*` | ω 1.026e-06 · α 0.1011 · γ 0.0052 · β 0.8732 · cap 0.13 | ω 1.2e-05 · α 0.05 · γ 0.065 · β 0.875 · cap 0.01 | re-estimated on 1h data 2026-07-15; the 1m values live in `config/interval/1m.yaml` |

🇮🇹 **Guard interval (fail-fast operativo):** `RuntimeError` "interval mismatch" in `03_backtest.py` e `04_live_signals.py` — modello addestrato a 1m + config 1h = combinazione invalida, bloccata. I consumer live/replay derivano l'interval da `PipelineState.interval_minutes` (fallback 1 per i pkl legacy), **mai** dalla config. σ safety-net scalata a `0.05·√interval_minutes` (≈0.387 a 1h); annualizzazione `bars_per_year = 525600 // interval_minutes` (1h→8760). **Overlay di risoluzione:** `python run_all.py --interval 1h` (o `1m`) applica `config/interval/{interval}.yaml` sopra `default.yaml` (merge shallow per-sezione, dopo i secrets e prima dell'overlay arch) e propaga `QUANTSYS_INTERVAL` ai subprocess; le `choices` derivano dai file presenti in `config/interval/`.

**EN** **Interval guard (operational fail-fast):** `RuntimeError` "interval mismatch" in `03_backtest.py` and `04_live_signals.py` — a 1m-trained model + 1h config = invalid combination, blocked. Live/replay consumers derive the interval from `PipelineState.interval_minutes` (fallback 1 for legacy pkl), **never** from the config. σ safety net scaled to `0.05·√interval_minutes` (≈0.387 at 1h); annualization `bars_per_year = 525600 // interval_minutes` (1h→8760). **Resolution overlay:** `python run_all.py --interval 1h` (or `1m`) applies `config/interval/{interval}.yaml` on top of `default.yaml` (per-section shallow merge, after secrets and before the arch overlay) and propagates `QUANTSYS_INTERVAL` to subprocesses; `choices` derive from the files present in `config/interval/`.

🇮🇹 **Rollback a 1m** (procedura, 3 passi — nessuna modifica al codice: tutte le conversioni sono identità a 1m):
1. restore di `data/backup_1m/*` sulla copia canonica in `data/`;
2. config a 1m: `data.interval: 1m`, `data.start_time: '2025-05-19'` (+ i valori della colonna *era (1m)* qui sopra, o l'overlay `--interval 1m`);
3. **retrain obbligatorio** — i checkpoint 1m sono stati eliminati col cleanup 2026-06-12, su disco non ne resta nessuno. Il guard interval config↔state blocca comunque ogni combinazione incoerente prima che produca numeri.

**EN** **Rollback to 1m** (3-step procedure — no code change: every conversion is an identity at 1m):
1. restore `data/backup_1m/*` over the canonical copy in `data/`;
2. set the config to 1m: `data.interval: 1m`, `data.start_time: '2025-05-19'` (plus the *was (1m)* column above, or the `--interval 1m` overlay);
3. **mandatory retrain** — the 1m checkpoints were deleted in the 2026-06-12 cleanup, none is left on disk. The config↔state interval guard blocks any inconsistent combination anyway, before it can produce numbers.

---

## 2. Dati · Data

### 2.1 Download / update / macro

🇮🇹 La pipeline scarica e prepara i dati prima del training. Per il primo avvio lascia che `run_all.py` esegua tutte le fasi; per i run successivi salta con `--skip-update --skip-macro` (usa i dati su disco).

**EN** The pipeline downloads and prepares data before training. On first run let `run_all.py` execute all phases; on subsequent runs skip with `--skip-update --skip-macro` (use on-disk data).

| Script | Ruolo · Role |
|---|---|
| `scripts/01_download_data.py` | download completo candele Binance + funding, rebuild dataset npz · full Binance candles + funding download, npz dataset rebuild |
| `scripts/01_update_data.py` | aggiornamento incrementale delle candele a oggi, poi **rebuild** di feature + npz con **re-fit dello scaler** e riscrittura del `PipelineState` · incremental candle update to today, then feature + npz **rebuild** with **scaler refit** and `PipelineState` rewrite |
| `scripts/01_update_data.py --candles-only` | estende **solo** `data/raw_candles.parquet` e si ferma: nessuno scaler, nessun npz, nessuno state. Da usare quando i modelli a valle sono **congelati** e serve solo la storia OHLCV fresca · extends **only** `data/raw_candles.parquet` and stops: no scaler, no npz, no state. Use it when downstream models are **frozen** |
| `scripts/01b_download_macro.py` | macro FRED/yFinance + walk-forward regime `RegimeMarkovBTC` (clock orario; **~3h su 7 anni**) · FRED/yFinance macro + regime walk-forward (hourly clock; ~3h over 7 years) |

🇮🇹 **Modalità di `01b_download_macro.py`** (mutuamente esclusive; senza flag = pipeline completa):

| Flag | Effetto |
|---|---|
| `--regime-only` | rigenera SOLO `regime_probs.parquet` + `regime_hmm.pkl` + `regime_wf_checkpoint.pkl` (salta macro/normalizer/npz; il costo resta il walk-forward) |
| `--regime-incremental` | **(B7)** appende dal checkpoint le sole barre nuove: 0-1 fit MLE, minuti. `regime_hmm.pkl` NON viene aggiornato (solo il full rebuild rifà il fit finale full-sample) |
| `--regime-bootstrap-checkpoint` | ricostruzione una-tantum del checkpoint da pkl+parquet esistenti, con golden test integrato (replay vs parquet, fail-fast) |
| `--skip-regime` | pipeline macro completa lasciando il regime detector **INTATTO** — il full rebuild rimappa gli indici dei regimi: da evitare a esperimenti aperti (le barre nuove si appendono con `--regime-incremental`) |

**EN** **`01b_download_macro.py` modes** (mutually exclusive; no flag = full pipeline):

| Flag | Effect |
|---|---|
| `--regime-only` | regenerates ONLY `regime_probs.parquet` + `regime_hmm.pkl` + `regime_wf_checkpoint.pkl` (skips macro/normalizer/npz; the walk-forward remains the cost) |
| `--regime-incremental` | **(B7)** appends new bars only, from the checkpoint: 0-1 MLE fits, minutes. `regime_hmm.pkl` is NOT updated (only a full rebuild redoes the final full-sample fit) |
| `--regime-bootstrap-checkpoint` | one-off checkpoint rebuild from existing pkl+parquet, with a built-in golden test (replay vs parquet, fail-fast) |
| `--skip-regime` | full macro pipeline leaving the regime detector **UNTOUCHED** — a full rebuild remaps regime indices: avoid while experiments are open (new bars are appended via `--regime-incremental`) |

🇮🇹 **Dati prodotti:** `data/raw_candles.parquet` = candele 1h 2019→oggi (~65k barre); `data/funding_rate.parquet` = funding completo dal lancio perp 2019-09-10; `data/macro_*.parquet` = FRED/yFinance; `data/regime_probs.parquet` = probabilità regime (index orario UTC); `data/features.parquet` + `data/lstm_dataset.npz` = feature normalizzate e finestre `X/y` per il training (104 canoniche = 86 dinamiche + 18 strutturali; `X_train ≈ (51k, 120, 104)`). ⚠ `lstm_dataset.npz` è **grande (~3 GB) e rigenerabile** da `01_download_data.py`: se assente, rigeneralo prima di train/judge.

**EN** **Produced data:** `data/raw_candles.parquet` = 1h candles 2019→today (~65k bars); `data/funding_rate.parquet` = full funding since the 2019-09-10 perp launch; `data/macro_*.parquet` = FRED/yFinance; `data/regime_probs.parquet` = regime probabilities (hourly UTC index); `data/features.parquet` + `data/lstm_dataset.npz` = normalized features and `X/y` windows for training (104 canonical = 86 dynamic + 18 structural; `X_train ≈ (51k, 120, 104)`). ⚠ `lstm_dataset.npz` is **large (~3 GB) and regenerable** from `01_download_data.py`: if missing, regenerate it before train/judge.

### 2.2 Collector forward (dato non rigenerabile) · Forward collectors (non-regenerable data)

🇮🇹 Tre collector raccolgono **in avanti** storico non disponibile gratis altrove. Dal 2026-07-18 girano **solo sul VPS** (servizi systemd): a casa non va rilanciato nulla — routine di sessione in *5.3*, deploy in *5.3bis*. Il VPS always-on elimina i buchi PC-off (coverage IV misurata al 18.6% delle ore, 2026-06-12→07-14).
- **`01c_iv_poller.py`** — IV Deribit short-tenor → `data/iv/` (UNICO dato non rigenerabile).
- **`01d_orderbook_recorder.py`** — order-book L2 Binance → `data/orderbook/` (Strada B1 microstruttura).
- **`01e_trades_recorder.py`** — trade opzioni Deribit production → `data/deribit_trades/` (spread realizzati; retention API ~24h → solo forward).

**EN** Three collectors gather **forward** history not freely available elsewhere. Since 2026-07-18 they run **on the VPS only** (systemd services): nothing to relaunch at home — session routine in *5.3*, deploy in *5.3bis*. The always-on VPS removes the PC-off gaps (measured IV coverage 18.6% of hours, 2026-06-12→07-14).
- **`01c_iv_poller.py`** — Deribit short-tenor IV → `data/iv/` (the ONLY non-regenerable data).
- **`01d_orderbook_recorder.py`** — Binance L2 order-book → `data/orderbook/` (B1 microstructure track).
- **`01e_trades_recorder.py`** — Deribit production option trades → `data/deribit_trades/` (realized spreads; ~24h API retention → forward only).

---

## 3. Modellazione · Modeling

### 3.1 Pipeline completa · Full pipeline

```bash
python run_all.py                    # menu: ↑↓ naviga, SPAZIO seleziona, A toggle all, INVIO conferma
python run_all.py --arch itransformer --force-download   # modalità diretta · direct mode
```
🇮🇹 Senza flag mostra il menu interattivo e apre la dashboard su `http://localhost:8050`. Le fasi: dati → macro → train → walk-forward → backtest → live → dashboard.

**EN** Without flags it shows the interactive menu and opens the dashboard at `http://localhost:8050`. Phases: data → macro → train → walk-forward → backtest → live → dashboard.

### 3.2 Training singola arch · Single-architecture training

🇮🇹 Ogni architettura ha config in `config/arch/{arch}.yaml` e output isolati in `models/{arch}/` e `results/{arch}/`. Nessuna interferenza tra run. ⚠ I tempi sono stimati sul vecchio dataset 1m-525k; il dataset 1h (~65k, ~8× più piccolo) è proporzionalmente più veloce, **da ri-misurare**.

**EN** Each architecture has its own config in `config/arch/{arch}.yaml` and isolated outputs in `models/{arch}/` and `results/{arch}/`. No cross-run interference. ⚠ Times are estimated on the old 1m-525k dataset; the 1h dataset (~65k, ~8× smaller) is proportionally faster, **to be re-measured**.

| `--arch` | Comando · Command | Classe (in `quantsys/model/`) · Class | Tempo (1m-525k) · Time |
|---|---|---|---|
| `itransformer` | `python run_all.py --arch itransformer --skip-update --skip-macro` | `QuantiTransformer` (`__init__.py`) — attention sulle feature, baseline · feature-wise attention | ~13–40 min |
| `nhits` | `python run_all.py --arch nhits --skip-update --skip-macro` | `QuantNHiTS` (`nhits.py`) — pure-MLP gerarchico · hierarchical pure-MLP | ~6–19 min |
| `tcnmamba` | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | `QuantTCNMamba` (`tcn_mamba.py`) — conv dilatate + SSM, collo di bottiglia · bottleneck | ~80 min/seed |
| `lstm` | `python run_all.py --arch lstm --skip-update --skip-macro` | `QuantLSTM` (`__init__.py`) — legacy, sotto-performante · underperforming | (legacy) |

🇮🇹 `--skip-update --skip-macro`: usa i dati su disco senza ridownload (ometti al primo run). Equivalente CLI diretto: `$env:QUANTSYS_ARCH="<arch>"; python scripts/02_train.py --n-ensemble <N>`. ⚠ Su disco esistono solo `models/itransformer` e `models/lstm`: `nhits`/`tcnmamba` partono da zero (cleanup 2026-06-12).

**EN** Same table above. `--skip-update --skip-macro`: use on-disk data without redownload (omit on first run). Direct CLI equivalent: `$env:QUANTSYS_ARCH="<arch>"; python scripts/02_train.py --n-ensemble <N>`. ⚠ Only `models/itransformer` and `models/lstm` exist on disk: `nhits`/`tcnmamba` start from scratch (2026-06-12 cleanup).

### 3.3 Ensemble omogeneo (5× stessa arch) · Homogeneous ensemble (5× same arch)

```yaml
# config/default.yaml
training:
  n_ensemble: 5   # default single-arch = 5; --distill default = 1, override esplicito con --n-ensemble
```
🇮🇹 Output: `models/{arch}/best_model_0..4.pt`. Backtest/live li caricano via `EnsembleModel.load`. Indipendente dalla distillation (le modalità non si escludono).

**EN** Output: `models/{arch}/best_model_0..4.pt`. Backtest/live load them via `EnsembleModel.load`. Independent from distillation (modes are not mutually exclusive).

### 3.4 Walk-forward & Optuna

🇮🇹 **Walk-forward** (`scripts/02b_walkforward_validate.py`, integrato in `run_all.py`): purged k-fold con embargo anti-leakage. `n_folds=6` → **5 fold effettivi** (fold 0 scartato strutturalmente). ⚠ Le metriche walk-forward in-sample (Spearman/WHR) **anti-correlano** col backtest direzionale: non usarle per ottimizzare.

**EN** **Walk-forward** (`scripts/02b_walkforward_validate.py`, run inside `run_all.py`): purged k-fold with anti-leakage embargo. `n_folds=6` → **5 effective folds** (fold 0 structurally skipped). ⚠ In-sample walk-forward metrics (Spearman/WHR) **anti-correlate** with the directional backtest: do not use them to optimize.

```bash
set QUANTSYS_ARCH=lstm                                # cmd.exe; PowerShell: $env:QUANTSYS_ARCH="lstm"
python scripts/02c_optuna_search.py --n-trials 50 --study-name quantsys
```
🇮🇹 **Optuna** è **hardcoded su `QuantLSTM`**. `best_params.json` salvato in `models/lstm/` NON è applicato automaticamente: copia manuale in `config/arch/lstm.yaml`. Studio persistente su SQLite (`models/lstm/optuna_quantsys.db`), ripristinabile.

**EN** **Optuna** is **hardcoded to `QuantLSTM`**. `best_params.json` saved in `models/lstm/` is NOT auto-applied: copy it manually into `config/arch/lstm.yaml`. Study persists on SQLite (`models/lstm/optuna_quantsys.db`), resumable any time.

### 3.5 Distillation multi-teacher · Multi-teacher distillation

```bash
python run_all.py --distill --skip-update --skip-macro
python run_all.py --distill --teacher itransformer   # forza il primary teacher, salta lo scoring · force primary teacher, skip scoring
```

🇮🇹 **Composizione — unico punto di modifica:** `config/default.yaml → distillation.archs`.
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
Dopo la modifica, `python run_all.py --distill` addestra i mancanti → fa scoring → distilla; backtest e live usano la nuova composizione. Esempi: `["itransformer","lstm","tcnmamba"]` (rollback legacy), `["itransformer","tcnmamba"]` (solo 2).

**EN** **Composition — the single edit point:** `config/default.yaml → distillation.archs` (block above). After editing, `python run_all.py --distill` trains what is missing → scores → distills; backtest and live pick up the new composition. Examples: `["itransformer","lstm","tcnmamba"]` (legacy rollback), `["itransformer","tcnmamba"]` (only 2).

🇮🇹 **Note operative:**
- `n_ensemble = 1` di default sul path `--distill` (vale per candidati **e** student); override solo con `--n-ensemble N` esplicito sulla CLI.
- Se `models/{arch}/best_model.pt` esiste, l'arch viene skippata; per ri-addestrare/ri-distillare cancella il checkpoint o passa `--force-download`.
- **Verifica esito:** `models/{arch}/config.json` → `distilled: true`, `teacher_arch: "multi-teacher"`.
- ⚠ `models/nhits` e `models/tcnmamba` non esistono più su disco (cleanup 2026-06-12): il primo `--distill` li riaddestra da zero, tempi in §3.2.
- Teoria (scoring target-aware `teacher_score_weights`, soft labels μ/ls²/lnu, loss `(1−α)·NLL + α·distill`, legge della varianza totale, pesi `DEFAULT_ARCH_WEIGHTS` e `ensemble_nll_temperature`): **`TEORIA.md` § Knowledge Distillation**.

**EN** **Operational notes:**
- `n_ensemble = 1` by default on the `--distill` path (for candidates **and** students); override only with an explicit `--n-ensemble N` on the CLI.
- If `models/{arch}/best_model.pt` exists the arch is skipped; to retrain/re-distill delete the checkpoint or pass `--force-download`.
- **Check the outcome:** `models/{arch}/config.json` → `distilled: true`, `teacher_arch: "multi-teacher"`.
- ⚠ `models/nhits` and `models/tcnmamba` no longer exist on disk (2026-06-12 cleanup): the first `--distill` retrains them from scratch, timings in §3.2.
- Theory (target-aware `teacher_score_weights` scoring, μ/ls²/lnu soft labels, `(1−α)·NLL + α·distill` loss, law of total variance, `DEFAULT_ARCH_WEIGHTS` and `ensemble_nll_temperature`): **`TEORIA.md` § Knowledge Distillation**.

### 3.6 CAFN — training congiunto · CAFN — joint training

```bash
python scripts/02d_cafn_joint_train.py --smoke        # valida il loop (CPU, dati sintetici) · loop smoke (CPU, synthetic)
python scripts/02d_cafn_joint_train.py --epochs 20    # reale, richiede data/lstm_dataset.npz · real run, needs the npz
```

🇮🇹 **Probe pre-registrato, inerte.** Output **isolato** in `models/cafn/`: non tocca `models/{arch}` né la parity live (kwarg `latent=None` nei 3 forward → path bit-identico al legacy). **Gate pre-registrato (val-first):** PASS sse il CAFN-congiunto batte il baseline NO-CAFN (stessi modelli/seed/epoche) di **≥3% MSE-mu su val per ≥2/3 modelli**, altrimenti KILL. Flag utili: `--archs`, `--batch`, `--d-latent`, `--cafn-d-model`, `--cafn-layers`, `--lambda-causal`, `--lr`, `--max-steps`, `--device`, `--no-gate`. ⚠ 3 modelli + CAFN su 8 GB → rischio **OOM**: abbassa `--batch`/`--cafn-d-model` e non girarlo in parallelo a poller/vol-paper. Architettura e penalità causale: **`TEORIA.md` § CAFN**.

**EN** **Pre-registered, inert probe.** Output **isolated** in `models/cafn/`: it touches neither `models/{arch}` nor live parity (`latent=None` kwarg in the 3 forwards → bit-identical to legacy). **Pre-registered gate (val-first):** PASS iff joint-CAFN beats the NO-CAFN baseline (same models/seeds/epochs) by **≥3% val MSE-mu on ≥2/3 models**, else KILL. Useful flags: `--archs`, `--batch`, `--d-latent`, `--cafn-d-model`, `--cafn-layers`, `--lambda-causal`, `--lr`, `--max-steps`, `--device`, `--no-gate`. ⚠ 3 models + CAFN on 8 GB → **OOM** risk: lower `--batch`/`--cafn-d-model` and never run it alongside the poller/vol-paper. Architecture and causal penalty: **`TEORIA.md` § CAFN**.

### 3.7 Aggiungere una nuova arch · Adding a new architecture

🇮🇹 Procedura in 7 passi (dettaglio nella skill `/add-arch`) · **EN** 7-step procedure (detail in the `/add-arch` skill):

1. Classe in `quantsys/model/` con `forward(x, x_macro=None) -> (mu, ls2, lnu)` · class with that forward contract
2. Dispatcher `load_model` in `quantsys/model/__init__.py`
3. Branch `architecture == "X"` in `scripts/02_train.py`
4. `config/arch/X.yaml`
5. `choices` dei parser `--arch` e `--teacher` in `run_all.py` · `--arch` and `--teacher` parser choices
6. Whitelist in `scripts/05_analyze_signals.py` (la dashboard resta arch-independent · the dashboard stays arch-independent)
7. Opzionale · optional: `distillation.archs` in `config/default.yaml`

---

## 4. Valutazione · Evaluation

### 4.1 Backtest direzionale & analisi segnali · Directional backtest & signal analysis

🇮🇹 `scripts/03_backtest.py` (integrato in `run_all.py`) gira il backtest trading sul target direzionale (`ret`); `scripts/05_analyze_signals.py` analizza i segnali live. Env operative: `QUANTSYS_BACKTEST_SPLIT=val` (**val-first**, output suffissati `*_val.*` che non clobberano la production; il test split si tocca a gate val superato) e `QUANTSYS_BACKTEST_SINGLE_ARCH=1` (backtest omogeneo per-arch invece dell'eterogeneo). ⚠ Il backtest **non ha senso sui modelli vol** (`log_rv`/`log_rs_ratio`): usa i giudici dedicati (§4.3). ⚠ Gli altri env-flag di `03_backtest.py` sono un **corpus KILL inerte** (regime gating, entry rank-based, cadenza, esposizione continua, calibrazione-σ): tutti validati e **FALLITI OOS**, elenco ed esiti in `TEORIA.md` §12.5 — non ri-testarli. Dopo ogni sweep azzera gli env sperimentali e rilancia un backtest pulito.

**EN** `scripts/03_backtest.py` (run inside `run_all.py`) runs the trading backtest on the directional target (`ret`); `scripts/05_analyze_signals.py` analyzes live signals. Operational envs: `QUANTSYS_BACKTEST_SPLIT=val` (**val-first**, outputs suffixed `*_val.*` that do not clobber production; the test split is touched once the val gate passes) and `QUANTSYS_BACKTEST_SINGLE_ARCH=1` (per-arch homogeneous backtest instead of heterogeneous). ⚠ The backtest is **meaningless on vol models** (`log_rv`/`log_rs_ratio`): use the dedicated judges (§4.3). ⚠ The other `03_backtest.py` env flags form an **inert KILL corpus** (regime gating, rank-based entry, decision cadence, continuous exposure, σ-calibration): all validated and **FAILED OOS**, list and outcomes in `TEORIA.md` §12.5 — do not re-test them. After each sweep clear the experimental envs and re-run a clean backtest.

### 4.2 Confronto architetture & parity · Architecture comparison & parity

```bash
python scripts/07_verify_teacher.py            # tabella comparativa archs · architecture comparison table
python scripts/99_replay_live_vs_training.py   # diagnostica parity live (BLOCKER #1) · live parity diagnostic
```
🇮🇹 `07_verify_teacher.py`: param count, forward time, Sharpe, WR, n trade, max DD, total return per ogni arch con `best_model.pt`. In alternativa: `models/{arch}/config.json` (`best_val_loss`, scaler, n_params), `models/{arch}/history.json` (curva loss), `results/{arch}/dashboard_results.json` (export backtest; non più letto dalla dashboard).

**EN** `07_verify_teacher.py`: param count, forward time, Sharpe, WR, n trades, max DD, total return for every arch with `best_model.pt`. Alternatively: `models/{arch}/config.json` (`best_val_loss`, scaler, n_params), `models/{arch}/history.json` (loss curve), `results/{arch}/dashboard_results.json` (backtest export; no longer read by the dashboard).

### 4.3 Giudici famiglia vol · Vol-family judges

🇮🇹 `features.target_type` in `config/default.yaml` seleziona la famiglia del target (default codice `ret`, bit-invariato):

| `target_type` | Target | Giudice | Esito |
|---|---|---|---|
| `ret` | log-return cumulato su h barre | `03_backtest.py` (§4.1) | nessun alpha OOS |
| `log_rv` | log realized variance Σr² su h barre | `scripts/vol/dev_vols_qlike.py` (QLIKE vs HAR-RV + naive) | **PASS a 1h** (−30% QLIKE), FAIL a 1m — 2026-06-10 |
| `log_rs_ratio` | asimmetria semivarianza log(RS⁺/RS⁻) | `scripts/vol/dev_vols_rs_judge.py` (MSE vs HAR-RS + naive + train-mean) | FAIL 2026-06-11 |

**EN** `features.target_type` in `config/default.yaml` selects the target family (code default `ret`, bit-invariant):

| `target_type` | Target | Judge | Outcome |
|---|---|---|---|
| `ret` | cumulative log-return over h bars | `03_backtest.py` (§4.1) | no OOS alpha |
| `log_rv` | log realized variance Σr² over h bars | `scripts/vol/dev_vols_qlike.py` (QLIKE vs HAR-RV + naive) | **PASS at 1h** (−30% QLIKE), FAIL at 1m — 2026-06-10 |
| `log_rs_ratio` | semivariance asymmetry log(RS⁺/RS⁻) | `scripts/vol/dev_vols_rs_judge.py` (MSE vs HAR-RS + naive + train-mean) | FAIL 2026-06-11 |

🇮🇹 Pipeline comune (dalla root, PowerShell):

```powershell
python scripts/01_download_data.py                  # rebuild dataset npz
python scripts/vol/dev_vols_macro_append.py         # ri-appende X_macro senza rifare il walk-forward regime (~5s vs ~3h)
$env:QUANTSYS_ARCH="itransformer"; python scripts/02_train.py --n-ensemble 5
$env:QUANTSYS_VOLS_SPLIT="val"; python scripts/vol/dev_vols_qlike.py      # val-first; poi "test" UNA sola volta
```

🇮🇹 Report in `results/vols/`. ⚠ **NO backtest trading sui modelli vol** (`03_backtest.py` non ha senso su target log-RV). Backup autosufficienti: `models/backup_1h_vols/` (PASS 1h), `models/backup_1m_vols/` (FAIL 1m, tenuto come record). Gli altri giudici e probe della linea vol sono elencati in `scripts/README.md`. ⚠ Due leve **inerti di default** vivono su questo path e non vanno accese fuori dal loro gate pre-registrato: `QUANTSYS_QLIKE_SMEARING=1` (correzione di smearing sul giudice, C1) e `model.attn_entropy_lambda` (penalità entropica sull'attention, A10 — overlay `config/arch/itransformer_a10_sparsity.yaml`, sonda `scripts/vol/attn_entropy_probe.py`). Razionale ed esiti: `TEORIA.md` §12.2 e §12.4.

**EN** Shared pipeline (from the root, PowerShell): block above. Reports in `results/vols/`. ⚠ **NO trading backtest on vol models** (`03_backtest.py` is meaningless on a log-RV target). Self-contained backups: `models/backup_1h_vols/` (1h PASS), `models/backup_1m_vols/` (1m FAIL, kept as a record). The other vol-line judges and probes are listed in `scripts/README.md`. ⚠ Two levers that are **inert by default** live on this path and must not be enabled outside their pre-registered gate: `QUANTSYS_QLIKE_SMEARING=1` (smearing correction on the judge, C1) and `model.attn_entropy_lambda` (attention entropy penalty, A10 — overlay `config/arch/itransformer_a10_sparsity.yaml`, probe `scripts/vol/attn_entropy_probe.py`). Rationale and outcomes: `TEORIA.md` §12.2 and §12.4.

### 4.4 Short-vol arm & IVS relative-value (linea vol monetizzazione) · Short-vol arm & IVS relative-value (vol monetization line)

🇮🇹 Script di ricerca GPU-free in `scripts/vol/`, **da lanciare dalla root**: famiglia `short_vol_*` (sim offline del forward test, backtest storico FHS-GJR-GARCH 2019→2026, robustness del premio VRP, decomposizione per regime/anno) e coppia `ivs_scout.py`/`ivs_rv_backtest.py` (smile Deribit + reversione dei residui). **Stato:** short-vol = edge VRP strutturale CONFERMATO sul backtest storico, ma il gate vero è il **forward test live** di `04b` (gate v1 n=20 **FAIL 0/3** il 2026-07-18; gate successivi in accumulo — soglie e contatori in `STATUS.md`); IVS relative-value = **KILL net-of-cost** (vivrebbe solo da market-maker). Ruolo e flag di ogni script: `scripts/README.md`.

**EN** GPU-free research scripts in `scripts/vol/`, **to be launched from the root**: the `short_vol_*` family (offline forward-test sim, 2019→2026 FHS-GJR-GARCH historical backtest, VRP premium robustness, regime/year decomposition) and the `ivs_scout.py`/`ivs_rv_backtest.py` pair (Deribit smile + residual reversion). **Status:** short-vol = structural VRP edge CONFIRMED on the historical backtest, but the real gate is `04b`'s **live forward test** (gate v1 n=20 **FAILED 0/3** on 2026-07-18; later gates accumulating — thresholds and counters in `STATUS.md`); IVS relative-value = **net-of-cost KILL** (would only live as a market-maker). Each script's role and flags: `scripts/README.md`.

---

## 5. Deploy & inferenza · Deploy & inference

### 5.1 Avvii successivi · Subsequent runs

```bash
python run_all.py --only-dashboard               # solo dashboard · dashboard only
python run_all.py --skip-train --skip-walkfwd    # aggiorna dati, stessi modelli · update data, same models
python run_all.py                                 # menu
python run_all.py --distill                       # full + distillation
```

🇮🇹 **Flag utili:**

| Flag | Effetto · Effect |
|------|---------|
| `--skip-update` | usa dataset esistente, no download · use existing dataset, no download |
| `--skip-macro` | salta download FRED/yFinance · skip FRED/yFinance download |
| `--skip-train` | usa modello esistente, no retrain · use existing model, no retrain |
| `--skip-walkfwd` | salta walk-forward validation · skip walk-forward validation |
| `--skip-backtest` | salta backtest · skip backtest |
| `--skip-live` | no feed live WebSocket · no live WebSocket feed |
| `--skip-analyze` | salta `05_analyze_signals.py` · skip `05_analyze_signals.py` |
| `--only-dashboard` | solo Options Risk Terminal, no ML né live · Options Risk Terminal only |
| `--no-browser` | non aprire browser · do not auto-open browser |
| `--force-download` | ri-scarica + forza retrain · redownload + force retrain |
| `--max-model-age-days N` | retrain se modello > N giorni · retrain if model older than N days |
| `--interval {1m,1h}` | overlay risoluzione candela · candle-resolution overlay |
| `--n-ensemble N` | seed ensemble single-arch (default 5) · single-arch ensemble seeds |
| `--distill` | pipeline multi-teacher · multi-teacher pipeline |
| `--teacher ARCH` | forza primary teacher · force primary teacher |

### 5.2 Live / paper trading

🇮🇹 Avviato da `run_all.py` (fase live, salvo `--skip-live`) o da `python scripts/04_live_signals.py`. Path di produzione: `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canoniche, scaler da `PipelineState`) →`_deterministic_predict`→`denormalize_predictions`→`SignalGenerator`. **Parity feature e segnale bit-perfect col training** (BLOCKER #1 chiuso il 2026-06-05): regressione in `tests/test_live_training_parity.py`, diagnostica end-to-end `python scripts/99_replay_live_vs_training.py` (Δfeature = Δμ = Δσ = 0). Paper-only, nessun ordine reale. Residuo operativo: smoke test WS reale. ⚠ Backtest direzionale negativo OOS → il paper-trading accumula solo trade reali. **NON** girare live + training/inferenza GPU in parallelo (contesa CUDA).

**EN** Started by `run_all.py` (live phase, unless `--skip-live`) or by `python scripts/04_live_signals.py`. Production path: `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canonical, scaler from `PipelineState`) →`_deterministic_predict`→`denormalize_predictions`→`SignalGenerator`. **Bit-perfect feature and signal parity with training** (BLOCKER #1 closed 2026-06-05): regression in `tests/test_live_training_parity.py`, end-to-end diagnostic `python scripts/99_replay_live_vs_training.py` (Δfeature = Δμ = Δσ = 0). Paper-only, no real orders. Operational remainder: a real WS smoke test. ⚠ Directional backtest negative OOS → paper-trading only accumulates real trades. Do **NOT** run live + GPU training/inference in parallel (CUDA contention).

### 5.3 Routine di sessione (lato casa) · Session routine (home side)

🇮🇹 Dal 2026-07-18 il PC di casa è **passivo**: nessun processo residente locale — i collector `01c`/`01d`/`01e` e il vol-paper `04b` girano tutti come **servizi systemd sul VPS** (§5.3bis). **Unico comando a ogni sessione**, dalla root di progetto:

```powershell
.\avvio_sessione.ps1          # [-Days 7] [-SkipPull] [-SkipMonitor] [-RefreshCandles]
```

| # | Blocco · Block | Contenuto · Content |
|---|---|---|
| ① | **Pull + merge VPS** | `scripts/vps/pull_vps_data.ps1`: **archiviazione a vintage datati** di `macro_features.parquet` → VPS (vedi sotto), scp collector → `data/vps_staging/`, merge dedup nella copia canonica, **heartbeat staleness dei 4 collector** (IV / L2 / trades / 04b), staging auto-pulito · **dated-vintage archiving** of `macro_features.parquet`, scp collectors → staging, dedup merge into the canonical copy, 4-collector staleness heartbeat, auto-cleaned staging |
| ② | **Freshness regime B7** | ≥168 barre orarie oltre il checkpoint walk-forward → `01b_download_macro.py --regime-incremental` in background (anti-dup se un `01b` è già vivo); con candele congelate stampa "fresco" ed è un no-op · ≥168 hourly bars past the walk-forward checkpoint → background incremental refresh (anti-dup); a no-op on frozen candles |
| ②bis | **Estensione candele** — *solo con `-RefreshCandles`, spento di default* | `01_update_data.py --candles-only`: estende `data/raw_candles.parquet` e si ferma (nessuno scaler, nessun npz, nessuno state). **Non è un default** e il perché sta nel paragrafo dedicato sotto: automatizzarlo toglierebbe la possibilità di **congelare i dati**. Collocato **dopo** ② di proposito, così l'eventuale refresh regime parte al prossimo avvio · *only with `-RefreshCandles`, off by default*: extends the raw candle parquet and stops; deliberately placed **after** ② so any regime refresh starts at the next startup |
| ③ | **Monitoraggio linea vol** (CPU-only, **a scrittura zero**) | `scripts/vol/derive_mfiv.py` incrementale (**dopo** il merge per costruzione: legge la chain appena scaricata; stampa il wedge MFIV−ATM, che è una **diagnostica permanente**, non un contatore di gate) + contatore delle leg opzioni eseguite (`executed` di `trades.jsonl`, soglia storica n≥30: il gate è chiuso, ma resta **l'unico numero che dice se `04b` sta eseguendo**) + **copertura del file barre 1m** (vedi sotto) + **campione E1 stadio 2** (`edge_information_judge.py --stage 2 --count-only` verso n≥40, preceduto da fin dove arriva la serie dei close). ⚠ Il contatore **hedged** è stato **ritirato il 2026-08-13**, alla rimozione di `--hedge` dall'unit: il suo gate era chiuso dall'11/08 e il suo ultimo consumatore era la verifica che la banda di wind-down non aprisse hedge nuovi. Non va ripristinato "per informazione" — un numero senza consumatore invita a confrontarlo con una soglia che non è più la sua, ed è il contatore che ha letto l'unità sbagliata tre volte (eventi ≠ posizioni ≠ settlement) · incremental MFIV derivation (after the merge by construction), `--count-only` expiry count, executed-option-leg counter (the only number telling whether `04b` is executing), **E1 stage-2 sample** count preceded by the reach of the close series; the **hedged** counter was **retired on 2026-08-13** together with the `--hedge` removal and must not be restored "for information" |

🇮🇹 ⚠ **Disciplina one-shot — invariante permanente della routine.** Il blocco ③ non esegue **mai** un giudice pre-registrato: i contatori che restano calcolano **solo conteggi** (nessun PnL, nessun edge, nessuna correlazione), e ogni giudice ha comunque il proprio guard `n < n_min → NO_RUN` senza scrivere report → automatizzare un conteggio **non può produrre peeking**. ⚠ Un guard `NO_RUN` protegge però solo **sotto** soglia, e non protegge affatto gli effetti collaterali che lo precedono (fetch di rete, riscritture di cache): per questo la routine usa le varianti `--count-only` e il **run one-shot resta MANUALE**. ⚠ Il contatore `--count-only` del comparatore MFIV è stato **ritirato il 2026-08-18**, alla chiusura del suo gate (FAIL a n=41): `derive_mfiv.py` resta perché mantiene una colonna diagnostica, non perché alimenti un gate. Fail-soft: ogni passo che fallisce logga un warning senza fermare la routine (`-SkipPull` salta ①, `-SkipMonitor` salta ③; `-Days N` = finestra del pull; `-RefreshCandles` **aggiunge** il passo ②bis, vedi sotto).

🇮🇹 ⚠ **Copertura del file barre 1m (dal 2026-08-06) — continuità del recorder e disponibilità del target sono due cose diverse.** Il check L2 misura la continuità del **recorder**; questo misura se esiste il **target** con cui quelle ore verrebbero giudicate. Il giudice B1 costruisce la RV da rendimenti a **1 minuto** (`data/raw_candles_1m_l2.parquet`, ~180 quadrati per osservazione invece di 3), quindi le ore di L2 oltre la fine di quel file sono **raccolte ma prive di target**: il campione utile è il **minimo fra i due**, e fino a oggi nessuno dei due contatori lo diceva. ⚠ **Il tetto vale per le analisi a target 1m** (B1 a h=3, proxy del pin-close): il contatore `n_eff` a h=30 stampato dal check L2 usa le barre **orarie** — il target di produzione è 30 barre da rendimenti orari — e **non è toccato** da questo file. ⚠ **Il file ha tre consumatori e zero produttori — e zero produttori per decisione (dal 2026-08-10)**: `l2_incremental_judge.py`, `pin_close_feasibility.py` e `edge_information_judge.py` (come sorgente di coda) lo leggono, ma **nessuno script del repo lo genera o lo estende**. Non è lavoro arretrato: le kline a 1 minuto sono **storiche e ri-scaricabili indefinitamente**, quindi — al contrario del recorder L2, che è forward-only e irrecuperabile se si ferma — **non si perde nulla ad aspettare**, e il file scaricato oggi è identico a quello scaricato fra sei mesi. Il "ritardo" è un download non ancora fatto, non un debito che matura. Per questo il monitor **misura soltanto** e non stampa un rimedio: quando un'analisi a target 1m verrà davvero riaperta, il download si decide in quel momento e sulla finestra che serve, invece di tenere aggiornato di continuo un file che **nessun path operativo legge** (verificato il 2026-08-10: nessun riferimento sul VPS, e il contributo effettivo del file a `hourly_close()` misurato = **0 righe**, perché l'1m è *indietro* rispetto alle orarie). Stato al 2026-08-10: 87.217 barre, 2026-06-01 → 2026-07-31, **10 giorni di L2 senza target 1m**.

**EN** ⚠ **1m bar file coverage (since 2026-08-06) — recorder continuity and target availability are two different things.** The L2 check measures **recorder** continuity; this one measures whether the **target** those hours would be judged against exists. The B1 judge builds RV from **1-minute** returns (`data/raw_candles_1m_l2.parquet`, ~180 squares per observation instead of 3), so L2 hours past the end of that file are **collected but target-less**: the usable sample is the **minimum of the two**, and until today neither counter said so. ⚠ **The cap applies to 1m-target analyses** (B1 at h=3, pin-close proxies): the `n_eff` counter at h=30 printed by the L2 check uses **hourly** bars — the production target is 30 bars of hourly returns — and is **unaffected** by this file. ⚠ **The file has three consumers and zero producers — and zero producers by decision (since 2026-08-10)**: `l2_incremental_judge.py`, `pin_close_feasibility.py` and `edge_information_judge.py` (as a tail source) read it, but **no script in the repo generates or extends it**. This is not a backlog: 1-minute klines are **historical and re-downloadable indefinitely**, so — unlike the L2 recorder, which is forward-only and unrecoverable if it stops — **nothing is lost by waiting**, and the file downloaded today is identical to the one downloaded six months from now. The "lag" is a download not yet done, not a maturing debt. Hence the monitor **only measures** and prints no remedy: when a 1m-target analysis is actually reopened, the download is decided then and over the window it needs, instead of continuously keeping current a file that **no operational path reads** (verified 2026-08-10: no reference on the VPS, and the file's actual contribution to `hourly_close()` measured = **0 rows**, because the 1m data is *behind* the hourly bars). State on 2026-08-10: 87,217 bars, 2026-06-01 → 2026-07-31, **10 days of L2 without a 1m target**.

🇮🇹 ⚠ **`-RefreshCandles` (dal 2026-08-06) — estendere le barre è un atto, non un automatismo.** Il flag lancia `01_update_data.py --candles-only` (blocco ②bis, dopo il check B7 e prima del monitoraggio) ed è **spento di default**, modellato su `-PromoteMacro` per la stessa ragione per cui quello esiste. La meccanica sarebbe sicura anche automatizzata — **no-op** se non c'è nulla di nuovo (esce *prima* di scrivere, mtime invariato), dedup che tiene la riga **esistente** quindi la storia non si riscrive mai, mai la barra in formazione (`floor(now) − 1 barra`), scrittura atomica, e il file **non viene mai spedito al VPS** (l'unico trasferimento casa→VPS è la macro) — ma automatizzarla toglierebbe la possibilità di **congelare i dati** per un esperimento pre-registrato: l'invariante *«candele/npz/regime_probs non si toccano fino a chiusura gate»* passerebbe da «non fare nulla» a «ricordarsi `-SkipMonitor`», che è la forma esatta della promozione macro avvenuta **per automazione** il 31/07. E poiché lo split è una **frazione del conteggio righe**, ogni barra appesa sposta i confini train/val/test al prossimo rebuild dell'npz: con l'estensione automatica il vintage del dataset diventerebbe funzione di *quante sessioni hai aperto*, cioè non più dichiarabile. **Quando usarlo:** prima di annotare il contatore E1 se il blocco ③ stampa il warning di ritardo, e prima del run one-shot di **E1 stadio 2** (prerequisito ⑤ della sua pre-reg). Fra i gate aperti **solo E1 legge le barre** — il giudice hedged legge i ledger e il funding perp, il comparatore MFIV il chain e `forecasts.parquet`. **Quando NON usarlo:** dentro un esperimento che dichiara i dati congelati, e quando si riproduce un report storico su una finestra che assume un conteggio di barre preciso. ⚠ **Effetto a distanza di una sessione, voluto:** estendere le barre fa avanzare il contatore di staleness B7, quindi il refresh incrementale del regime parte semmai al **prossimo** avvio — le due scritture restano separate e ognuna visibile per conto suo invece di concatenarsi in silenzio.

**EN** ⚠ **`-RefreshCandles` (since 2026-08-06) — extending the bars is an act, not an automation.** The flag runs `01_update_data.py --candles-only` (block ②bis, after the B7 check and before monitoring) and is **off by default**, modelled on `-PromoteMacro` for the same reason that one exists. The mechanics would be safe even automated — **no-op** when nothing is new (it exits *before* writing, mtime unchanged), dedup keeping the **existing** row so history is never rewritten, never the in-progress bar (`floor(now) − 1 bar`), atomic write, and the file is **never shipped to the VPS** (the only home→VPS transfer is macro) — but automating it would remove the ability to **freeze the data** for a pre-registered experiment: the *"candles/npz/regime_probs untouched until the gate closes"* invariant would go from "do nothing" to "remember `-SkipMonitor`", which is exactly the shape of the macro promotion that happened **by automation** on 31/07. And since the split is a **fraction of the row count**, every appended bar shifts the train/val/test boundaries at the next npz rebuild: under automatic extension the dataset vintage would become a function of *how many sessions you opened*, i.e. no longer declarable. **When to use it:** before recording the E1 counter if block ③ prints the lag warning, and before the one-shot run of **E1 stage 2** (prerequisite ⑤ of its pre-reg). Among the open gates **only E1 reads the bars** — the hedged judge reads the ledgers and perp funding, the MFIV comparator the chain and `forecasts.parquet`. **When NOT to use it:** inside an experiment declaring frozen data, and when reproducing a historical report over a window that assumes an exact bar count. ⚠ **Deliberate one-session delayed effect:** extending the bars moves the B7 staleness counter forward, so the incremental regime refresh starts, if at all, at the **next** startup — the two writes stay separate and each visible on its own instead of chaining silently.

🇮🇹 ⚠ **Il guard `NO_RUN` protegge solo SOTTO soglia — perché E1 è entrato nella routine con `--count-only` e non con `--stage 2` (2026-08-06).** `edge_information_judge.py --stage 2` sotto n=40 stampa la conta e si ferma, ma **a n≥40 lo stesso comando calcola le tre condizioni, stampa il verdetto e scrive il report**: automatizzarlo nudo avrebbe fatto scattare il run confermativo **per automazione invece che per decisione**, il giorno in cui la soglia cade e senza che nessuno lo scegliesse. `--count-only` si ferma alla conta a **qualunque** n, e un test lo verifica su un campione costruito **sopra** la soglia (`tests/test_edge_information_judge.py`) — sotto soglia il test non proverebbe nulla, perché passerebbe anche il comando nudo. ⚠ **Il conteggio dipende dalla serie dei close**: un'expiry è osservabile solo se la sua RV è calcolabile, quindi il blocco stampa **prima** fin dove arriva la serie e avvisa oltre 6h di ritardo. Il rimedio (`01_update_data.py --candles-only`) **non è automatizzato di proposito**: è una scrittura su un file di dati, e il blocco ③ resta a scrittura zero.

**EN** ⚠ **The `NO_RUN` guard only protects BELOW threshold — why E1 entered the routine as `--count-only`, not `--stage 2` (2026-08-06).** Below n=40 `edge_information_judge.py --stage 2` prints the count and stops, but **at n≥40 the same command computes the three conditions, prints the verdict and writes the report**: automating it bare would have fired the confirmatory run **by automation rather than by decision**, on the day the threshold is met and without anyone choosing it. `--count-only` stops at the count at **any** n, and a test verifies this on a sample built **above** threshold (`tests/test_edge_information_judge.py`) — below threshold the test would prove nothing, since the bare command would pass too. ⚠ **The count depends on the close series**: an expiry is observable only if its RV is computable, so the block prints **first** how far that series reaches and warns past a 6h lag. The remedy (`01_update_data.py --candles-only`) is **deliberately not automated**: it writes to a data file, and block ③ stays write-free.

🇮🇹 ⚠ **Vintage macro datati (dal 2026-07-31) — il push non è più una decisione, promuovere lo è.** `04b` legge `data/macro_features.parquet` al bootstrap notturno e lo **congela** per tutta la giornata: sovrascriverlo dentro un campione forward pre-registrato ne cambia l'input in silenzio (successo il 2026-07-31, breakpoint datato in `STATUS.md`). Il pull quindi: (a) **archivia sempre** il file sotto `data/macro/macro_features_<YYYYMMDD>.parquet` sul VPS, dove `<YYYYMMDD>` è l'**ultima data dell'indice** (`scripts/vps/macro_vintage.py`) — append-only, 716 KB a copia, così ogni decisione forward resta riconducibile al suo vintage; (b) tratta il canonico come un **symlink** all'archivio, quindi il vintage live si legge con `readlink` e si vede in `ls -l`; (c) **non lo ripunta mai** senza `-PromoteMacro`, e a vintage divergente emette un warning lasciando il live su quello che già usava. Il nuovo vintage entra in vigore al **bootstrap 04b successivo (00:30 UTC)**: se un campione forward è aperto, la promozione va **datata in `STATUS.md`**.

```powershell
.\scripts\vps\pull_vps_data.ps1 -PromoteMacro   # atto deliberato: ripunta il canonico VPS al vintage locale
```

🇮🇹 ⚠ **Strumento vs stato — il `MacroNormalizer` pinnato (implementato 2026-07-31, INERTE).** Il vintage datato risolve *quale* macro arriva al VPS; resta che `VolForecaster` **ri-stima** il `MacroNormalizer` whole-df a ogni bootstrap, quindi allungare il parquet sposta mediana e IQR e **lo strumento di misura cambia insieme allo stato che deve misurare** (sul breakpoint del 31/07: 2.7% della variazione totale). `scripts/vol/pin_macro_normalizer.py` congela lo strumento a un vintage **dichiarato** e lo scrive in un pickle; `04b` e il replay lo accettano con `--macro-norm <path>`. **Senza il flag il comportamento è bit-identico** (verificato end-to-end sul parquet di produzione: 0 differenze su 90 colonne). ⚠ Il vintage sotto cui `models/itransformer` fu addestrato **non è ricostruibile**: il pin non lo recupera, ne **fissa** uno esplicito. Attivarlo oggi sarebbe un **no-op di contenuto** (stesso vintage → delta 0): serve a impedire la deriva *futura*, non a correggere il passato. L'artefatto è gitignored ma **riproducibile** da un vintage archiviato.

**EN** ⚠ **Instrument vs state — the pinned `MacroNormalizer` (implemented 2026-07-31, INERT).** Dated vintages settle *which* macro reaches the VPS; what remains is that `VolForecaster` **refits** the `MacroNormalizer` whole-df at every bootstrap, so extending the parquet moves median and IQR and **the measuring instrument changes together with the state it must measure** (2.7% of the 31/07 breakpoint). `scripts/vol/pin_macro_normalizer.py` freezes the instrument at a **declared** vintage; `04b` and the replay accept it via `--macro-norm <path>`. **Without the flag behavior is bit-identical** (verified end-to-end on the production parquet: 0 differences over 90 columns). ⚠ The vintage `models/itransformer` was trained under is **not reconstructible**: the pin does not recover one, it **fixes** one. Enabling it today would be a **content no-op**; it prevents *future* drift. The artifact is gitignored but **reproducible** from an archived vintage.

🇮🇹 ⚠ **EMERGENZA** — solo su `WARN IV poller` nell'heartbeat (collector VPS giù), lanciare a mano finché il VPS non torna: `.\.venv\Scripts\python.exe scripts\01c_iv_poller.py`. **`04b` NON va MAI lanciato a casa:** due `--execute` gestirebbero la stessa posizione testnet (doppi ordini). Stop di un processo d'emergenza (match sulla command line → cattura stub+worker, non tocca altri `python.exe`):

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match '01c_iv_poller|01d_orderbook_recorder' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

🇮🇹 **Salute:** conta i processi **LOGICI**, non OS — ogni `.venv\python.exe` è stub+worker = 2 processi OS (confronta i `ParentProcessId`). La crescita dei dati arriva **dal pull**, non da processi locali: `data/iv/atm_30h.parquet` e `data/iv/mfiv_30h.parquet` ~288 righe/giorno (cadenza VPS 5 min), `results/vol_paper/forecasts.parquet` ~24 righe/giorno. Log vivi in `logs/quantsys_*.log` (più recenti per mtime).

**EN** Since 2026-07-18 the home PC is **passive**: no resident local process — collectors `01c`/`01d`/`01e` and vol-paper `04b` all run as **systemd services on the VPS** (§5.3bis). **One command per session**, from the project root: block above; the three blocks are in the table above.

⚠ **One-shot discipline — a permanent invariant of the routine.** Block ③ **never** runs a pre-registered judge: the remaining counters compute **counts only** (no PnL, no edge, no correlation), and every judge still guards `n < n_min → NO_RUN` without writing a report → automating a count **cannot produce peeking**. ⚠ A `NO_RUN` guard, however, protects only **below** threshold, and does not protect the side effects preceding it at all (network fetches, cache rewrites): hence the routine uses the `--count-only` variants and the **one-shot run stays MANUAL**. ⚠ The MFIV comparator's `--count-only` counter was **retired on 2026-08-18**, when its gate closed (FAIL at n=41): `derive_mfiv.py` stays because it maintains a diagnostic column, not because it feeds a gate. Fail-soft: a failing step logs a warning without stopping the routine (`-SkipPull` skips ①, `-SkipMonitor` skips ③; `-Days N` = pull window; `-RefreshCandles` **adds** step ②bis, see below).

⚠ **Dated macro vintages (since 2026-07-31) — the push is no longer a decision, promoting is.** `04b` reads `data/macro_features.parquet` at its nightly bootstrap and **freezes** it for the whole day: overwriting it inside a pre-registered forward sample silently changes its input (happened 2026-07-31, breakpoint dated in `STATUS.md`). The pull therefore: (a) **always archives** the file as `data/macro/macro_features_<YYYYMMDD>.parquet` on the VPS, where `<YYYYMMDD>` is the **last index date** (`scripts/vps/macro_vintage.py`) — append-only, 716 KB per copy, so every forward decision stays traceable to its vintage; (b) treats the canonical as a **symlink** into the archive, so the live vintage is readable with `readlink` and visible in `ls -l`; (c) **never repoints it** without `-PromoteMacro`, and on a diverging vintage emits a warning while leaving the live path on what it was already using. The new vintage takes effect at the **next 04b bootstrap (00:30 UTC)**: with an open forward sample, the promotion must be **dated in `STATUS.md`**. Command: block above.

⚠ **EMERGENCY** — only on a `WARN IV poller` heartbeat (VPS collector down), run by hand until the VPS is back: `.\.venv\Scripts\python.exe scripts\01c_iv_poller.py`. **`04b` must NEVER run at home:** two `--execute` would manage the same testnet position (double orders). Emergency stop: block above (command-line match → catches stub+worker, leaves other `python.exe` untouched).

**Health:** count **LOGICAL** processes, not OS ones — each `.venv\python.exe` is stub+worker = 2 OS processes (compare `ParentProcessId`). Data growth comes **from the pull**, not from local processes: `data/iv/atm_30h.parquet` and `data/iv/mfiv_30h.parquet` ~288 rows/day (5-min VPS cadence), `results/vol_paper/forecasts.parquet` ~24 rows/day. Live logs in `logs/quantsys_*.log` (newest by mtime).

#### 5.3bis Collector 24/7 su VPS · 24/7 collectors on the VPS

🇮🇹 Quattro servizi **systemd** su un VPS EU always-on (deployato 2026-07-14): `quantsys-iv` (`01c`), `quantsys-ob` (`01d`), `quantsys-trades` (`01e`), `quantsys-volpaper` (`04b`). **Host/IP sono privati: SOLO in `config/secrets.yaml` → blocco `vps:`**, mai nel repo o nella doc. Deploy completo: `deploy/vps/README.md` (geo-test 451 Binance → deploy key → `setup_vps.sh` one-shot → verify). Comandi lato casa, dalla root di progetto:

```powershell
.\avvio_sessione.ps1              # tutto-in-uno di sessione: pull+merge + check B7 + monitoraggio vol (§5.3)
.\scripts\vps\pull_vps_data.ps1   # solo sync: host da secrets.yaml → scp → data/vps_staging/ + merge + heartbeat
.\scripts\vps\check_vps.ps1       # health-check on-demand via ssh: servizi / freschezza / disco / geo (-UpdateRepo = git pull remoto)
```

🇮🇹 Il merge (`scripts/vps/merge_vps_data.py`) deduplica i tick doppi e avvisa se l'ultimo tick VPS è stale (default 3h → collector remoto giù). La copia canonica dei dati resta a casa (`data/iv/`, `data/orderbook/`, `data/deribit_trades/`); il VPS garantisce continuità H24 dell'asset IV. ⚠ Il PC di casa non riavvia collector locali (§5.3).

**EN** Four **systemd** services on an always-on EU VPS (deployed 2026-07-14): `quantsys-iv` (`01c`), `quantsys-ob` (`01d`), `quantsys-trades` (`01e`), `quantsys-volpaper` (`04b`). **Host/IP are private: ONLY in `config/secrets.yaml` → `vps:` block**, never in repo or docs. Full deploy: `deploy/vps/README.md` (Binance 451 geo-test → deploy key → one-shot `setup_vps.sh` → verify). Home-side commands, from the project root: block above. The merge (`scripts/vps/merge_vps_data.py`) dedups double ticks and warns when the latest VPS tick is stale (default 3h → remote collector down). The canonical data copy stays home (`data/iv/`, `data/orderbook/`, `data/deribit_trades/`); the VPS guarantees 24/7 continuity of the IV asset. ⚠ The home PC starts no local collector (§5.3).

#### Poller IV Deribit · Deribit IV poller

🇮🇹 `python scripts/01c_iv_poller.py` — loop, cadenza default 10 min (sul VPS gira a **5 min con `--greeks`**). 2 richieste pubbliche Deribit per tick, **nessun account**. Flag: `--minutes N` (cadenza), `--once` (smoke), `--backfill-dvol` (storico DVOL orario 2021→oggi), `--greeks` (+3 chiamate/tick per i greeks di venue dello straddle ATM ~tenor-30h → `atm_greeks.parquet`, selezione identica a `pick_straddle` di `04b`). Output append-only atomico in `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (snapshot raw, ~950 strumenti/tick), `atm_30h.parquet` (ATM IV delle 4 expiry vicine + IV interpolata in varianza totale a tenor costante 30h = orizzonte del forecast vol), `dvol.parquet` (controllo 30d). Scopo: storico IV short-tenor — non disponibile gratis altrove — per il gate **NN-RV vs IV implicita**.

**EN** `python scripts/01c_iv_poller.py` — loop, default 10-min cadence (on the VPS it runs at **5 min with `--greeks`**). 2 public Deribit requests per tick, **no account**. Flags: `--minutes N` (cadence), `--once` (smoke), `--backfill-dvol` (hourly DVOL history 2021→today), `--greeks` (+3 calls/tick for the venue greeks of the ATM ~30h-tenor straddle → `atm_greeks.parquet`, selection identical to `04b`'s `pick_straddle`). Atomic append-only output under `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (raw snapshot, ~950 instruments/tick), `atm_30h.parquet` (ATM IV of the 4 nearest expiries + total-variance-interpolated IV at a constant 30h tenor = the vol forecast horizon), `dvol.parquet` (30d control). Purpose: short-tenor IV history — not free elsewhere — for the **NN-RV vs implied IV** gate.

#### Recorder order-book L2 Binance (B1) · Binance L2 order-book recorder (B1)

🇮🇹 `python scripts/01d_orderbook_recorder.py` — loop, cadenza default 5s. Flag: `--seconds N`, `--once`, `--symbol` (default `BTCUSDT`), `--levels` (profondità REST, default 1000). 1 richiesta pubblica `/api/v3/depth` per tick (no auth, weight 50/call → a 5s = 600/min ≪ 1200). Strada **B1**: raccolta FORWARD della microstruttura come fonte NUOVA per un edge direzionale a 1m (le 104 feature OHLCV sono sature). Output append-only atomico `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/giorno, dedup su `timestamp`): mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, depth cumulata 5/10/25/50 bps, total qty, **OFI best-level** (Cont-Kukanov-Stoikov) + **top-25 livelli raw/lato** come list-column. ⚠ `ofi_best` è NaN al 1° tick di ogni processo e in `--once`.

**EN** `python scripts/01d_orderbook_recorder.py` — loop, default 5s cadence. Flags: `--seconds N`, `--once`, `--symbol` (default `BTCUSDT`), `--levels` (REST depth, default 1000). 1 public `/api/v3/depth` request per tick (no auth, weight 50/call → at 5s = 600/min ≪ 1200). Track **B1**: FORWARD collection of microstructure as a NEW source for 1m directional edge (the 104 OHLCV features are saturated). Atomic append-only output `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/day, dedup on `timestamp`): mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, cumulative depth 5/10/25/50 bps, total qty, **best-level OFI** (Cont-Kukanov-Stoikov) + **top-25 raw levels/side** as list-columns. ⚠ `ofi_best` is NaN on the 1st tick of each process and in `--once`.

#### Forward test vol-paper (NN-RV vs IV, testnet Deribit) · Vol-paper forward test

🇮🇹 ⚠ **Gira come servizio sul VPS, NON a casa** (§5.3): due `--execute` gestirebbero la stessa posizione testnet. Invocazione di produzione (unit `deploy/vps/quantsys-volpaper.service`):

```bash
python scripts/04b_vol_paper.py --execute
```

🇮🇹 ⛔ **`--hedge` è FALLITO — gate v2 chiuso il 2026-08-11 (FAIL 2/3); i flag sono stati rimossi dall'unit il 2026-08-13 e l'invocazione qui sopra è quella realmente in vigore.** La varianza per-trade scendeva del 55.3% (① superata con margine) ma il drag valeva −0.647·SE contro un budget di −0.25·SE, per **76% fee perp e 0.9% funding**. Dettaglio e decomposizione: `TEORIA.md` §12.2. ⚠ **Vincolo di ORDINE nel disattivarlo, valido in entrambe le direzioni:** `maybe_hedge` e `reconcile_hedge_state` girano **solo** dentro il ramo `--hedge`, quindi togliere il flag mentre `results/vol_paper/hedge_state.json` esiste lascia la leg perp **nuda, non gestita e mai flattenata** sul testnet — vale per chiunque riaccenda il flag e poi voglia rispegnerlo. ⚠ **Non basta "disattivare dopo un settlement": la finestra pulita è larga zero.** Il settlement e l'apertura della struttura successiva avvengono nello **stesso tick**, quindi `position.json` non è mai vuoto e un nuovo hedge può partire immediatamente. Procedura corretta, in **due passi**: ① `--hedge-band 999`, che disabilita apertura e ribilanciamento ma lascia attivo il flatten (quel ramo gira **prima** del check di banda); ② dopo il flatten successivo, rimozione dei tre flag. Nel ledger il flatten può portare `reason` ∈ `{settled, expired, structure_changed}` — **tutti e tre validi**, sono i tre arm della stessa guard, e con la riapertura nello stesso tick esce `structure_changed`. La leg opzioni è identica con o senza hedge (path v1 bit-identico), quindi la disattivazione non perturba i campioni forward aperti.

**EN** ⛔ **`--hedge` has FAILED — v2 gate closed on 2026-08-11 (FAIL 2/3); the flags were removed from the unit on 2026-08-13 and the invocation above is the one actually in force.** Per-trade variance fell 55.3% (① passed with margin) but the drag was −0.647·SE against a −0.25·SE budget, made of **76% perp fees and 0.9% funding**. Detail and decomposition: `TEORIA.md` §12.2. ⚠ **An ORDERING constraint when disabling it, and it cuts both ways:** `maybe_hedge` and `reconcile_hedge_state` run **only** inside the `--hedge` branch, so dropping the flag while `results/vol_paper/hedge_state.json` exists leaves the perp leg **naked, unmanaged and never flattened** on the testnet — this applies to anyone re-enabling the flag and later turning it off. ⚠ **"Disable after a settlement" is not enough: the clean window is zero-wide.** The settlement and the opening of the next structure happen in the **same tick**, so `position.json` is never empty and a fresh hedge can start immediately. Correct procedure, in **two steps**: ① `--hedge-band 999`, which disables opening and rebalancing but keeps the flatten alive (that branch runs **before** the band check); ② after the next flatten, remove the three flags. In the ledger the flatten may carry `reason` ∈ `{settled, expired, structure_changed}` — **all three valid**, they are the three arms of the same guard, and with the same-tick reopening you get `structure_changed`. The options leg is identical with or without the hedge (bit-identical v1 path), so disabling does not perturb the open forward samples.

🇮🇹 Loop orario a hh:00+90s. Flag: `--once` (smoke), `--execute` (ordini REALI sul testnet; default = fill SIMULATI al mark price), `--arch` (dir modelli, default `itransformer`), delta-hedge `--hedge` + `--hedge-band/-conv/-fee/-band-mode/-ww-lambda`, chiusura `--pin-close-hours/-band`, sizing `--size-mode {contracts,vega}` + `--size-vega-target/-max-contracts`, normalizer macro `--macro-norm <pin>` (**inerte**: senza il flag lo strumento è ri-stimato whole-df a ogni bootstrap, comportamento storico). Logica: forecast NN-RV 30h (modello vol-1h PASS, inversione completa `μ·IQR+centro`, feature dal path parity-blessed) vs varianza implicita dal poller IV (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; `|edge| > 0.25` → straddle ATM daily ~30h LONG/SHORT, max 1 posizione, hold a scadenza (cash settlement). Richiede poller IV attivo e key in `config/secrets.yaml` blocco `deribit_testnet:` (l'URL **deve** essere `test.deribit.com` — assert anti-mainnet). Output in `results/vol_paper/`: `forecasts.parquet` (scritto anche a posizione flat, serve alle baseline), `trades.jsonl`, `position.json`. ⚠ NON girare training GPU in parallelo (5 modelli CUDA residenti).

**EN** ⚠ **Runs as a VPS service, NOT at home** (§5.3): two `--execute` would manage the same testnet position. Production invocation (unit `deploy/vps/quantsys-volpaper.service`): block above. Hourly loop at hh:00+90s. Flags: `--once` (smoke), `--execute` (REAL testnet orders; default = SIMULATED mark-price fills), `--arch` (model dir, default `itransformer`), delta-hedge `--hedge` + `--hedge-band/-conv/-fee/-band-mode/-ww-lambda`, close policy `--pin-close-hours/-band`, sizing `--size-mode {contracts,vega}` + `--size-vega-target/-max-contracts`, macro normalizer `--macro-norm <pin>` (**inert**: without the flag the instrument is refitted whole-df at every bootstrap, legacy behavior). Logic: 30h NN-RV forecast (PASS 1h-vol model, full `μ·IQR+center` inversion, parity-blessed feature path) vs implied variance from the IV poller (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; `|edge| > 0.25` → ~30h daily ATM straddle LONG/SHORT, max 1 position, hold to expiry (cash settlement). Requires a running IV poller and keys in `config/secrets.yaml` `deribit_testnet:` block (the URL **must** be `test.deribit.com` — anti-mainnet assert). Output in `results/vol_paper/`: `forecasts.parquet` (written even when flat, the baselines need it), `trades.jsonl`, `position.json`. ⚠ Do NOT run GPU training in parallel (5 CUDA-resident models).

🇮🇹 **Baseline del gate** — `python scripts/04c_vol_paper_baselines.py` (read-only, GPU-free; `--no-fetch` = solo cache delivery, `--min-trades N` = soglia di valutabilità, default 30). Verifica il gate pre-registrato (2): il P&L NN deve battere **entrambe** le baseline always-long-vol e always-short-vol sullo **stesso** calendario di expiry (isola il timing dal variance risk premium medio). Metodo: replay del loop `04b` su `forecasts.parquet`, premio ricostruito dai chain snapshot, delivery price dall'endpoint pubblico Deribit (cache `delivery_cache.json`). I gate (1) P&L medio > 0 e (3) hit-rate > 0.5 si leggono dai trade REALI in `trades.jsonl`. Output `results/vol_paper/baseline_report.json` (+ warning "non valutabile" finché n < `--min-trades`).

**EN** **Gate baselines** — `python scripts/04c_vol_paper_baselines.py` (read-only, GPU-free; `--no-fetch` = delivery cache only, `--min-trades N` = evaluability threshold, default 30). Checks pre-registered gate (2): the NN P&L must beat **both** the always-long-vol and always-short-vol baselines over the **same** expiry calendar (isolates timing from the average variance risk premium). Method: replay of the `04b` loop over `forecasts.parquet`, premium reconstructed from chain snapshots, delivery price from the public Deribit endpoint (`delivery_cache.json`). Gates (1) mean P&L > 0 and (3) hit-rate > 0.5 are read from the REAL trades in `trades.jsonl`. Output `results/vol_paper/baseline_report.json` (+ "not evaluable" warning while n < `--min-trades`).

### 5.4 Dashboard — Deribit Options Risk Terminal

```bash
python scripts/06_dashboard.py     # avvio diretto · direct launch
python run_all.py --only-dashboard # idem (no ML, no feed live · no ML, no live feed)
```

🇮🇹 `scripts/06_dashboard.py` = **terminale opzioni crypto** (server HTTP single-file + SPA Plotly.js), GPU-free e **indipendente dalla pipeline ML**: legge i dati **pubblici Deribit** (REST, no-auth, nessuna chiave). Quattro tab — *Volatility Surface* (superficie IV 3D, smile, term structure ATM), *Option Chain* (chain call/put con Greche Black-Scholes forward), *Risk & Greeks* (OI per strike, max-pain, Greche aggregate pesate per OI, PCR, DVOL), *Trades* (forward test `04b`: storico settled + posizione aperta e profilo di payoff, endpoint `/api/trades` che legge `results/vol_paper/trades.jsonl` + `position.json`). Auto-refresh ~12s. Config: `config/default.yaml → dashboard` — `host`/`port` (default `127.0.0.1:8050`), `options_currency` (BTC|ETH), `auth_token` opzionale (constant-time, header `X-Auth-Token` o `?token=`), `enable_gzip`.

**EN** `scripts/06_dashboard.py` = the **crypto options terminal** (single-file HTTP server + Plotly.js SPA), GPU-free and **decoupled from the ML pipeline**: it reads **Deribit public** data (REST, no-auth, no key). Four tabs — *Volatility Surface* (3D IV surface, smile, ATM term structure), *Option Chain* (call/put chain with forward Black-Scholes Greeks), *Risk & Greeks* (OI by strike, max-pain, OI-weighted aggregate Greeks, PCR, DVOL), *Trades* (`04b` forward test: settled history + open position and payoff profile, `/api/trades` endpoint reading `results/vol_paper/trades.jsonl` + `position.json`). Auto-refresh ~12s. Config: `config/default.yaml → dashboard` — `host`/`port` (default `127.0.0.1:8050`), `options_currency` (BTC|ETH), optional `auth_token` (constant-time, `X-Auth-Token` header or `?token=`), `enable_gzip`.

🇮🇹 ⚠ **Trappola `SO_REUSEADDR`:** un vecchio processo dashboard può tenere `:8050` e servire HTML **stale**. Prima di un nuovo smoke **uccidi il processo precedente** (stub+worker `.venv` = 1 processo logico): `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '06_dashboard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`. ⚠ **La verità finale è il browser con HARD RELOAD (Ctrl+Shift+R)**: una pagina già aperta gira il JS vecchio. Smoke server-side: l'HTML servito deve contenere `plot(` e `/api/risk` deve rispondere HTTP 200 con la chain reale. (Fix rendering definitivo 2026-06-24: asse X di `plot-oi`/`plot-payoff` passato a `type:'category'` per immunità alla corruzione SVG di Plotly al re-render; dettaglio in `STATUS.md`.)

**EN** ⚠ **`SO_REUSEADDR` trap:** a stale dashboard process can keep `:8050` and serve **stale** HTML. Before a fresh smoke **kill the previous process** (`.venv` stub+worker = 1 logical process): `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '06_dashboard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`. ⚠ **The final truth is the browser with a HARD RELOAD (Ctrl+Shift+R)**: an already-open page runs the old JS. Server-side smoke: the served HTML must contain `plot(` and `/api/risk` must return HTTP 200 with the real chain. (Definitive rendering fix 2026-06-24: `plot-oi`/`plot-payoff` X axis switched to `type:'category'` for immunity to Plotly's SVG re-render corruption; detail in `STATUS.md`.)

### 5.5 Fermare tutto · Stopping everything

🇮🇹 `Ctrl+C` nel terminale di `run_all.py` (o del server dashboard): ferma la pipeline e, in un run completo, anche il feed live WebSocket. ⚠ Se la dashboard era detached (`Start-Process`), `Ctrl+C` non basta: usa lo `Stop-Process` mirato (§5.4). I collector di sfondo non girano più a casa (VPS, §5.3bis): un'eventuale istanza locale d'emergenza si ferma col blocco *Stop* in §5.3.

**EN** `Ctrl+C` in the `run_all.py` terminal (or the dashboard server): stops the pipeline and, in a full run, the WebSocket live feed too. ⚠ If the dashboard was detached (`Start-Process`), `Ctrl+C` is not enough: use the targeted `Stop-Process` (§5.4). Background collectors no longer run at home (VPS, §5.3bis): any emergency local instance is stopped via the *Stop* block in §5.3.

---

## Appendice — File layout · Appendix — File layout

```
quantsys_project/
├── config/
│   ├── default.yaml             # config base (distillation.archs qui · here)
│   ├── secrets.yaml             # FRED + deribit_testnet keys, gitignored
│   ├── interval/                # overlay risoluzione: 1m.yaml, 1h.yaml (--interval)
│   └── arch/                    # itransformer.yaml, nhits.yaml, tcnmamba.yaml, lstm.yaml
├── data/
│   ├── raw_candles.parquet      # OHLCV storico (1h, 2019→oggi)
│   ├── features.parquet         # feature normalizzate (rigenerabile)
│   ├── lstm_dataset.npz         # windows X/y per training (~3 GB, rigenerabile da 01)
│   ├── funding_rate.parquet     # funding futures (completo dal 2019-09-10)
│   ├── macro_*.parquet          # FRED/yFinance
│   ├── regime_probs.parquet     # probabilità regime (index orario UTC)
│   ├── iv/                      # IV Deribit: chain/ (snapshot raw), atm_30h, mfiv_30h, dvol — NON rigenerabile
│   ├── orderbook/               # L2 features forward (B1) — NON rigenerabile
│   ├── deribit_trades/          # trade opzioni production (01e) — NON rigenerabile
│   └── backup_1m/               # raw_candles + regime_probs era-1m (rollback 1m = restore + retrain, §1.4)
├── models/
│   ├── pipeline_state.pkl       # copia canonica (scritta da 01, guard anti-stale in 02)
│   ├── backup_1h_vols/          # vol-1h PASS autosufficiente (5 membri + state + raw/regime 1h)
│   ├── backup_1m_vols/          # vol-1m FAIL (record)
│   ├── itransformer/            # PRODUCTION vol-1h (5 membri) — unica arch addestrata su disco
│   └── lstm/                    # legacy (+ studio Optuna). nhits/ e tcnmamba/ NON esistono: cleanup 2026-06-12
│       ├── best_model.pt        # checkpoint (best_model_0..4.pt per ensemble multi-seed)
│       ├── config.json          # iperparametri + flag distilled/teacher_arch + best_val_*
│       ├── history.json         # curva loss
│       └── pipeline_state.pkl   # scaler + feature config + target_scale + interval
├── results/
│   ├── {arch}/                  # dashboard_results.json, live_signals.jsonl
│   ├── vols/                    # report giudici vol
│   └── vol_paper/               # forecasts.parquet, trades.jsonl, position.json, baseline_report.json, exec_diag.jsonl (A6: bid/ask+greeks diagnostici / diagnostic), hedge_state.json + hedge_ledger.jsonl (v2, SOLO con --hedge / --hedge only)
├── docs/                        # STATUS_ARCHIVE_2026H1.md (storico ante 07-08, read-only), MODEL_IMPROVEMENTS, ROADMAP_VOL_BOOK, paper/
├── tests/                       # pytest (test_recent_fixes, test_live_training_parity, test_regime_incremental)
├── deploy/vps/                  # kit deploy VPS (setup_vps.sh + README)
├── avvio_sessione.ps1           # routine di sessione lato casa (§5.3)
├── scripts/
│   ├── 00_*..99_*               # spine numerato = fase della pipeline
│   ├── vol/ research/ vps/ archive/   # script non numerati, per linea — lanciare dalla ROOT
│   └── README.md                # ⇦ mappa canonica script→fase→linea
└── logs/quantsys_YYYYMMDD_HHMMSS.log
```

🇮🇹 **La mappa canonica degli script è `scripts/README.md`** (tabella completa script→fase→linea, con i flag di ognuno): questa guida non la duplica. Gli script nelle sottocartelle usano `Path(__file__).resolve().parents[2]` e vanno lanciati dalla **root di progetto**.

**EN** **The canonical script map is `scripts/README.md`** (full script→phase→line table, with each script's flags): this guide does not duplicate it. Scripts in subfolders use `Path(__file__).resolve().parents[2]` and must be launched from the **project root**.
