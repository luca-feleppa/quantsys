# QUANTSYS — Guida operativa di avvio · QUANTSYS — Operations & quick-start guide

🇮🇹 Sistema di forecasting neurale BTC/USDT con ensemble eterogeneo (iTransformer + N-HiTS + TCN+Mamba) e Knowledge Distillation multi-teacher. **Timeframe corrente: candele 1h** (pivot 2026-06-09; il precedente perimetro 1m è in backup, vedi *Setup → Pivot timeframe*). Motore **interval-agnostic**: tutte le conversioni di finestra sono identità a 1m.

**EN** BTC/USDT neural forecasting system with a heterogeneous ensemble (iTransformer + N-HiTS + TCN+Mamba) and multi-teacher Knowledge Distillation. **Current timeframe: 1h candles** (2026-06-09 pivot; the previous 1m perimeter is backed up, see *Setup → Timeframe pivot*). **Interval-agnostic** engine: every window conversion is an identity at 1m.

## Indice · Map

🇮🇹 Questa guida è ordinata per **fase operativa**: **0. Panoramica & stato** → **1. Setup e dipendenze** → **2. Dati** → **3. Modellazione** (train / walk-forward / Optuna / distillation / CAFN) → **4. Valutazione** (backtest / verify / analyze / vol-judge) → **5. Deploy & inferenza** (live, vol-paper, collector forward, dashboard). I comandi sono pensati per essere lanciati dalla **root** `E:\quantsys_project`.

**EN** This guide is ordered by **operational phase**: **0. Overview & status** → **1. Setup & dependencies** → **2. Data** → **3. Modeling** (train / walk-forward / Optuna / distillation / CAFN) → **4. Evaluation** (backtest / verify / analyze / vol-judge) → **5. Deploy & inference** (live, vol-paper, forward collectors, dashboard). All commands are meant to be run from the **root** `E:\quantsys_project`.

---

## 0. Panoramica & stato · Overview & status

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
python scripts/99_replay_live_vs_training.py         # diagnostica BLOCKER #1 · BLOCKER #1 diagnostic
```

### 0.2 Stato del sistema (pivot 1m→1h) · System status (1m→1h pivot)

🇮🇹 Pivot al timeframe **1h** (Strada 1 dopo il KILL del probe cross-sectional 2026-06-06, diagnosi "muro = magnitudine non segno"): a 1h il rapporto costo/σ per barra scende da ~1.9–3.3× a ~0.25–0.42× (il movimento di barra cresce ∝ √Δt, il costo roundtrip è fisso). Razionale econometrico in `TEORIA.md` §1, dettaglio implementativo in `docs/MODEL_IMPROVEMENTS.md` (sezione 2026-06-09).

**EN** Pivot to the **1h** timeframe (Path 1 after the 2026-06-06 cross-sectional probe KILL, diagnosis "the wall is magnitude, not sign"): at 1h the per-bar cost/σ ratio drops from ~1.9–3.3× to ~0.25–0.42× (bar move grows ∝ √Δt, roundtrip cost is fixed). Econometric rationale in `TEORIA.md` §1, implementation detail in `docs/MODEL_IMPROVEMENTS.md` (2026-06-09 section).

🇮🇹 **Stato linee di ricerca (2026-06-16):** la **vol** (`target_type: log_rv`) è l'unico segnale **PASS OOS** — il NN batte HAR-RV del 30% in QLIKE su test a 1h (FAIL a 1m, edge risoluzione-specifico) → linea pubblicabile + in forward-test live (vol-paper). Il **direzionale** (`target_type: ret`) non ha alpha OOS a nessun timeframe: il backtest grezzo resta **negativo**, l'edge a soglia/rango/regime non sopravvive OOS (corpus di KILL in `STATUS.md`) → resta negative-control per il paper "Are price and volume enough?". L'edge regime-Quiet (Spearman +0.13÷0.19 in R0) è di **rango** e non è tradabile: tutte le leve sperimentali (`QUANTSYS_QUIET_*`, `QUANTSYS_RANK_*`, `QUANTSYS_DECISION_CADENCE`, calibrazione-σ) sono validate su val e **FALLITE**, restano flag inerti documentati in `CLAUDE.md` §NOMENCLATURA.

**EN** **Research-line status (2026-06-16):** **vol** (`target_type: log_rv`) is the only **OOS PASS** signal — the NN beats HAR-RV by 30% QLIKE on the 1h test set (FAIL at 1m, resolution-specific edge) → publishable line + in live forward-test (vol-paper). **Directional** (`target_type: ret`) has no OOS alpha at any timeframe: the raw backtest stays **negative**, threshold/rank/regime edges do not survive OOS (KILL corpus in `STATUS.md`) → kept as negative-control for the "Are price and volume enough?" paper. The Quiet-regime edge (Spearman +0.13÷0.19 in R0) is a **rank** edge and is not tradable: every experimental lever (`QUANTSYS_QUIET_*`, `QUANTSYS_RANK_*`, `QUANTSYS_DECISION_CADENCE`, σ-calibration) was validated on val and **FAILED**, staying inert flags documented in `CLAUDE.md` §NOMENCLATURA.

🇮🇹 **Invariante z-score (per ogni nuovo entry point):** il modello predice in spazio z-score (RobustScaler); il trading layer opera in spazio raw. Chiama SEMPRE `PipelineState.denormalize_predictions(mu, sigma)` prima di passare le predizioni a `SignalGenerator`. Con target `log_rv` la denormalizzazione completa è `μ·IQR + centro` (mediana log-RV ≈ −7.2). Vedi `TEORIA.md` §5.

**EN** **z-score invariant (for any new entry point):** the model predicts in z-score space (RobustScaler); the trading layer operates in raw space. ALWAYS call `PipelineState.denormalize_predictions(mu, sigma)` before passing predictions to `SignalGenerator`. With the `log_rv` target the full inversion is `μ·IQR + center` (log-RV median ≈ −7.2). See `TEORIA.md` §5.

### 0.3 Live engine — stato · Live engine — status

🇮🇹 Paper-only (nessun ordine reale), **BLOCKER #1 RISOLTO (2026-06-05)**: il path live usa `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canoniche, stesso scaler) → `_deterministic_predict` → `denormalize_predictions` → `SignalGenerator`, con **parity feature E segnale bit-perfect** (`tests/test_live_training_parity.py` 5/5; `99_replay_live_vs_training.py` Δ=0). Residuo operativo: smoke test WS reale + paper-trading. ⚠ Il backtest direzionale è negativo OOS: il paper-trading serve solo ad accumulare trade reali. Vedi `TEORIA.md` §11.

**EN** Paper-only (no real orders), **BLOCKER #1 RESOLVED (2026-06-05)**: the live path uses `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canonical, same scaler) → `_deterministic_predict` → `denormalize_predictions` → `SignalGenerator`, with **bit-perfect feature AND signal parity** (`tests/test_live_training_parity.py` 5/5; `99_replay_live_vs_training.py` Δ=0). Operational remainder: real WS smoke test + paper-trading. ⚠ The directional backtest is negative OOS: paper-trading is only for accumulating real trades. See `TEORIA.md` §11.

---

## 1. Setup e dipendenze · Setup & dependencies

### 1.1 Installazione · Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python scripts/00_check_setup.py
```

🇮🇹 `00_check_setup.py` controlla dipendenze, CUDA, Binance, FRED — risolvi gli errori prima di proseguire. `00_test_binance_testnet.py` è uno smoke opzionale della connettività al testnet Binance.

**EN** `00_check_setup.py` checks dependencies, CUDA, Binance, FRED — fix errors before proceeding. `00_test_binance_testnet.py` is an optional Binance-testnet connectivity smoke.

### 1.2 Note operative su Windows / PowerShell · Windows / PowerShell operational notes

🇮🇹
- **PowerShell 5.1 stderr:** **NON** usare `2>&1 | Tee-Object` — incapsula stderr come ErrorRecord (output rosso fittizio; il logging Python va su stderr). Lo script salva già `logs/quantsys_*.log`; per un file dedicato usa `*> file.log`.
- **Redirect shell-dependent:** `*> file.log` è sintassi **PowerShell**; sotto **bash** `*` viene globbato e il log NON si scrive → usa `> file.log 2>&1`. Verifica sempre che il log si riempia prima di considerare avviato un job lungo.
- **UTF-8 boilerplate:** ogni nuovo script in `scripts/` deve reconfigurare UTF-8 su stdout/stderr in `main()` (il bug cp1252 è ricorso 5 volte: qualunque unicode nel banner crasha su console Windows).
- **`set` vs `$env:`:** i blocchi `set QUANTSYS_ARCH=...` di questa guida sono sintassi **cmd.exe**; in PowerShell usa `$env:QUANTSYS_ARCH="..."`.

**EN**
- **PowerShell 5.1 stderr:** do **NOT** use `2>&1 | Tee-Object` — it wraps stderr as an ErrorRecord (fake red output; Python logging goes to stderr). The script already writes `logs/quantsys_*.log`; for a dedicated file use `*> file.log`.
- **Shell-dependent redirect:** `*> file.log` is **PowerShell** syntax; under **bash** `*` is globbed and the log is NOT written → use `> file.log 2>&1`. Always verify the log is filling before assuming a long job has started.
- **UTF-8 boilerplate:** every new `scripts/` file must reconfigure UTF-8 on stdout/stderr in `main()` (the cp1252 bug recurred 5 times: any unicode in the banner crashes the Windows console).
- **`set` vs `$env:`:** the `set QUANTSYS_ARCH=...` blocks in this guide are **cmd.exe** syntax; in PowerShell use `$env:QUANTSYS_ARCH="..."`.

### 1.3 Hardware

🇮🇹 **CPU** — `config/default.yaml`:
```yaml
hardware:
  cpu_fraction: 0.5   # 0.3=30%, 0.5=50%, 0.8=80%
```
Default 0.5 (4 thread su 8 core). Letto da tutti gli script all'avvio.

**EN** **CPU** — `config/default.yaml`: block above. Default 0.5 (4 threads on 8 cores). Read by every script at startup.

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
| Batch inference backtest | 256 (`scripts/03_backtest.py`) |
| Batch training | 64 (default `config/arch/<arch>.yaml`) |

**EN** **Reference setup (RTX 2070 Super 8GB):**

| Component | Value |
|---|---|
| CUDA, AMP fp16 training | yes (via `setup_device`) |
| AMP inference | **off** hardcoded in `quantsys/model/ensemble.py` (avoids NaN from spectral_norm + Mamba scan) |
| Backtest inference batch | 256 (`scripts/03_backtest.py`) |
| Training batch | 64 (default `config/arch/<arch>.yaml`) |

🇮🇹 **Solo CPU:** fallback automatico via `setup_device` (`quantsys/utils/__init__.py`); `autocast(device_type="cuda")` è no-op silenzioso su CPU. Tempi: training 20–50× più lento (sconsigliato), backtest ~5s GPU → 30–60s CPU (tollerabile), live ~50–100ms vs ~20ms GPU (utilizzabile, domina la latency WS Binance). **Apple Silicon / AMD / Intel Arc:** non testato (codice `torch.cuda.*`; MPS richiederebbe modifiche a `setup_device` + kernel custom Mamba/SSM).

**EN** **CPU only:** automatic fallback via `setup_device` (`quantsys/utils/__init__.py`); `autocast(device_type="cuda")` is a silent no-op on CPU. Times: training 20–50× slower (not recommended), backtest ~5s GPU → 30–60s CPU (tolerable), live ~50–100ms vs ~20ms GPU (usable, Binance WS latency dominates). **Apple Silicon / AMD / Intel Arc:** untested (`torch.cuda.*` code; MPS would need `setup_device` changes + custom Mamba/SSM kernels).

🇮🇹 **Poca VRAM (4GB):** in `config/arch/<arch>.yaml` `batch_size: 32` + `gradient_accumulation_steps: 2` (effective batch 64); inference batch in `scripts/03_backtest.py` 256→128. **Molta VRAM (≥16GB):** `batch_size: 128`, inference batch fino a 1024 (guadagno marginale, GPU già satura).

**EN** **Low VRAM (4GB):** in `config/arch/<arch>.yaml` `batch_size: 32` + `gradient_accumulation_steps: 2` (effective batch 64); inference batch in `scripts/03_backtest.py` 256→128. **High VRAM (≥16GB):** `batch_size: 128`, inference batch up to 1024 (marginal gain, GPU already saturated).

### 1.4 Pivot timeframe 1m↔1h · 1m↔1h timeframe pivot

🇮🇹 **Config pivot (`config/default.yaml`):**

| Parametro | Valore 1h | Era (1m) | Nota |
|---|---|---|---|
| `data.interval` | `1h` | `1m` | tutte le finestre derivano da qui via `interval_minutes` |
| `data.start_time` | `2019-01-01` | `2025-05-19` | storico multi-anno, ~65k barre |
| `window_stride` | 1 | 5 | massimizza i sample su ~65k barre |
| `embargo_steps` | 168 (1 settimana) | 1500 (~25h) | ≥ window_size+horizon = 150 |
| `max_hold_candles` | 60 (2.5 giorni) | 240 (4h) | vincolo ≥ h=30 |
| `min_expected_ret` | 0.0013 (13 bps cost-aware) | 0.0005 | 2° test pre-registrato a 23 bps |
| `max_sigma` | 0.10 (≈0.015·√60) | 0.015 | da ricalibrare sui percentili post-denorm |
| `forecast_horizon` | 30 (INVARIATO) | 30 | ma ora = **30 ORE** (era 30 min) |
| `window_size` | 120 (INVARIATO) | 120 | ma ora = **5 giorni** di contesto (era 2h) |

**EN** **Pivot config (`config/default.yaml`):**

| Parameter | 1h value | Was (1m) | Note |
|---|---|---|---|
| `data.interval` | `1h` | `1m` | every window derives from it via `interval_minutes` |
| `data.start_time` | `2019-01-01` | `2025-05-19` | multi-year history, ~65k bars |
| `window_stride` | 1 | 5 | maximizes samples on ~65k bars |
| `embargo_steps` | 168 (1 week) | 1500 (~25h) | ≥ window_size+horizon = 150 |
| `max_hold_candles` | 60 (2.5 days) | 240 (4h) | constraint ≥ h=30 |
| `min_expected_ret` | 0.0013 (13 bps cost-aware) | 0.0005 | 2nd pre-registered test at 23 bps |
| `max_sigma` | 0.10 (≈0.015·√60) | 0.015 | to recalibrate on post-denorm percentiles |
| `forecast_horizon` | 30 (UNCHANGED) | 30 | but now = **30 HOURS** (was 30 min) |
| `window_size` | 120 (UNCHANGED) | 120 | but now = **5 days** of context (was 2h) |

🇮🇹 **Guard interval:** `RuntimeError` "interval mismatch" in `03_backtest.py` e `04_live_signals.py`: modello addestrato a 1m + config 1h = combinazione invalida bloccata. I consumer live/replay derivano l'interval da `PipelineState.interval_minutes` (fallback 1 per i pkl legacy), non dalla config. σ safety-net scalata a `0.05·√interval_minutes` (≈0.387 a 1h). Annualizzazione: `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760). **Risoluzione via overlay:** `python run_all.py --interval 1h` (o `1m`) applica `config/interval/{interval}.yaml` sopra `default.yaml` (merge shallow per-sezione, dopo i secrets e prima dell'overlay arch) e propaga `QUANTSYS_INTERVAL` ai subprocess; le choices derivano dai file presenti in `config/interval/`.

**EN** **Interval guard:** `RuntimeError` "interval mismatch" in `03_backtest.py` and `04_live_signals.py`: a 1m-trained model + 1h config = invalid combination, blocked. Live/replay consumers derive the interval from `PipelineState.interval_minutes` (fallback 1 for legacy pkl), not from the config. σ safety net scaled to `0.05·√interval_minutes` (≈0.387 at 1h). Annualization: `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760). **Resolution overlay:** `python run_all.py --interval 1h` (or `1m`) applies `config/interval/{interval}.yaml` on top of `default.yaml` (per-section shallow merge, after secrets and before the arch overlay) and propagates `QUANTSYS_INTERVAL` to subprocesses; choices derive from the files present in `config/interval/`.

🇮🇹 **Rollback 1m:** ripristina la config 1m (`interval: 1m`, `start_time: 2025-05-19`, stride 5, embargo 1500, max_hold 240, min_ret 0.0005, max_sigma 0.015) + restore `data/backup_1m/*`, poi **retrain** (i checkpoint direzionali-1m sono stati eliminati col cleanup 2026-06-12; sono riaddestrabili da `data/backup_1m/`). **Il codice non va toccato**: tutte le conversioni sono identità a 1m; il guard interval config↔state blocca le combinazioni incoerenti.

**EN** **1m rollback:** restore the 1m config (`interval: 1m`, `start_time: 2025-05-19`, stride 5, embargo 1500, max_hold 240, min_ret 0.0005, max_sigma 0.015) + restore `data/backup_1m/*`, then **retrain** (directional-1m checkpoints were removed in the 2026-06-12 cleanup; retrainable from `data/backup_1m/`). **No code changes needed**: every conversion is an identity at 1m; the interval config↔state guard blocks inconsistent combinations.

---

## 2. Dati · Data

### 2.1 Download / update / macro

🇮🇹 La pipeline scarica e prepara i dati prima del training. Per il primo avvio lascia che `run_all.py` esegua tutte le fasi; per i run successivi salta con `--skip-update --skip-macro` (usa i dati su disco).

**EN** The pipeline downloads and prepares data before training. On first run let `run_all.py` execute all phases; on subsequent runs skip with `--skip-update --skip-macro` (use on-disk data).

| Script | Ruolo · Role |
|---|---|
| `scripts/01_download_data.py` | download completo candele Binance + funding, rebuild dataset npz · full Binance candles + funding download, npz dataset rebuild |
| `scripts/01_update_data.py` | aggiornamento incrementale delle candele a oggi · incremental candle update to today |
| `scripts/01b_download_macro.py` | macro FRED/yFinance + walk-forward regime `RegimeMarkovBTC` (clock orario; su 7 anni ~3h) · FRED/yFinance macro + regime walk-forward (hourly clock; ~3h over 7 years) |

🇮🇹 **Dati prodotti:** `data/raw_candles.parquet` = candele 1h 2019→oggi (~65k barre); `data/funding_rate.parquet` = funding completo dal lancio perp 2019-09-10; `data/macro_*.parquet` = FRED/yFinance; `data/regime_probs.parquet` = probabilità regime (index orario UTC); `data/features.parquet` + `data/lstm_dataset.npz` = feature normalizzate e finestre `X/y` per il training (104 canoniche = 86 dinamiche + 18 strutturali; `X_train ≈ (51k, 120, 104)`). ⚠ `lstm_dataset.npz` è **grande (~3 GB) e rigenerabile** da `01_download_data.py`: se assente, rigeneralo prima di train/judge.

**EN** **Produced data:** `data/raw_candles.parquet` = 1h candles 2019→today (~65k bars); `data/funding_rate.parquet` = full funding since the 2019-09-10 perp launch; `data/macro_*.parquet` = FRED/yFinance; `data/regime_probs.parquet` = regime probabilities (hourly UTC index); `data/features.parquet` + `data/lstm_dataset.npz` = normalized features and `X/y` windows for training (104 canonical = 86 dynamic + 18 structural; `X_train ≈ (51k, 120, 104)`). ⚠ `lstm_dataset.npz` is **large (~3 GB) and regenerable** from `01_download_data.py`: if missing, regenerate it before train/judge.

### 2.2 Collector forward (dato non rigenerabile) · Forward collectors (non-regenerable data)

🇮🇹 Due collector raccolgono **in avanti** storico non disponibile gratis altrove. Sul PC di casa vanno **rilanciati dopo ogni riavvio** (non sono servizi) — comandi di avvio/stop nella sezione *5.3 Collector forward*. Dal 2026-07-14 il percorso primario è il **VPS always-on** (kit in `deploy/vps/`, sync `scripts/vps/` — vedi *5.3bis*): elimina i buchi PC-off (coverage IV misurata al 18.6% delle ore, 2026-06-12→07-14).
- **`01c_iv_poller.py`** — IV Deribit short-tenor → `data/iv/` (UNICO dato non rigenerabile).
- **`01d_orderbook_recorder.py`** — order-book L2 Binance → `data/orderbook/` (Strada B1 microstruttura).

**EN** Two collectors gather **forward** history not freely available elsewhere. On the home PC they must be **relaunched after every reboot** (not services) — start/stop commands in *5.3 Forward collectors*. Since 2026-07-14 the primary path is the **always-on VPS** (kit in `deploy/vps/`, sync in `scripts/vps/` — see *5.3bis*): it removes the PC-off gaps (measured IV coverage 18.6% of hours, 2026-06-12→07-14).
- **`01c_iv_poller.py`** — Deribit short-tenor IV → `data/iv/` (the ONLY non-regenerable data).
- **`01d_orderbook_recorder.py`** — Binance L2 order-book → `data/orderbook/` (B1 microstructure track).

---

## 3. Modellazione · Modeling

### 3.1 Pipeline completa · Full pipeline

```bash
python run_all.py                    # menu: ↑↓ naviga, SPAZIO seleziona, A toggle all, INVIO conferma
python run_all.py --arch nhits --force-download   # modalità diretta · direct mode
```
🇮🇹 Senza flag mostra il menu interattivo e apre la dashboard su `http://localhost:8050`. Le fasi: dati → macro → train → walk-forward → backtest → live → dashboard.

**EN** Without flags it shows the interactive menu and opens the dashboard at `http://localhost:8050`. Phases: data → macro → train → walk-forward → backtest → live → dashboard.

### 3.2 Training singola arch · Single-architecture training

🇮🇹 Ogni architettura ha config in `config/arch/{arch}.yaml` e output isolati in `models/{arch}/` e `results/{arch}/`. Nessuna interferenza tra run. ⚠ I tempi sono stimati sul vecchio dataset 1m-525k; il dataset 1h (~65k, ~8× più piccolo) è proporzionalmente più veloce, **da ri-misurare**.

**EN** Each architecture has its own config in `config/arch/{arch}.yaml` and isolated outputs in `models/{arch}/` and `results/{arch}/`. No cross-run interference. ⚠ Times are estimated on the old 1m-525k dataset; the 1h dataset (~65k, ~8× smaller) is proportionally faster, **to be re-measured**.

| Arch | Comando · Command | Tempo (1m-525k, RTX 2070S) · Time | Note |
|---|---|---|---|
| iTransformer | `python run_all.py --arch itransformer --skip-update --skip-macro` | ~13–40 min | Attention sulle feature, baseline |
| N-HiTS | `python run_all.py --arch nhits --skip-update --skip-macro` | ~6–19 min | Pure-MLP gerarchico (sostituisce LSTM dal 2026-05-14) |
| TCN+Mamba | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | ~80 min/seed | Conv dilatate + SSM, collo di bottiglia |
| LSTM | `python run_all.py --arch lstm --skip-update --skip-macro` | (legacy) | Backward compat (sotto-performante) |

🇮🇹 `--skip-update --skip-macro`: usa i dati su disco senza ridownload (ometti al primo run). Equivalente CLI diretto del training single-arch: `$env:QUANTSYS_ARCH="<arch>"; python scripts/02_train.py --n-ensemble <N>`.

**EN** `--skip-update --skip-macro`: use on-disk data without redownload (omit on first run). Direct CLI equivalent of single-arch training: `$env:QUANTSYS_ARCH="<arch>"; python scripts/02_train.py --n-ensemble <N>`.

🇮🇹 **Architetture disponibili:**

| Arch | Classe | File | Note |
|---|---|---|---|
| `itransformer` | `QuantiTransformer` | `quantsys/model/__init__.py` | Attention sulle feature, baseline |
| `nhits` | `QuantNHiTS` | `quantsys/model/nhits.py` | Pure-MLP gerarchico |
| `tcnmamba` | `QuantTCNMamba` | `quantsys/model/tcn_mamba.py` | TCN dilatate + SSM ibrido |
| `lstm` | `QuantLSTM` | `quantsys/model/__init__.py` | Legacy |

**EN** **Available architectures:** (same table; classes `QuantiTransformer` / `QuantNHiTS` / `QuantTCNMamba` / `QuantLSTM`). ⚠ I riferimenti `file:linea` marciscono ad ogni edit — verificali con grep prima di usarli · `file:line` references rot on every edit — verify with grep before relying on them.

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
```

🇮🇹 Ensemble eterogeneo di default: **iTransformer + N-HiTS + TCN+Mamba** (LSTM rimosso il 2026-05-14, val_NLL 5.28 vs iTransformer 0.18; codice intatto, ricaricabile per rollback). Fasi:
- **2a — Training candidati:** ogni arch in `distillation.archs` con `n_ensemble=1` di default (override SOLO con `--n-ensemble N` esplicito; vale per candidati E student). Se `models/{arch}/best_model.pt` esiste → skip; forza con `--force-download`.
- **2b — Multi-Teacher Scoring (target-aware, `teacher_score_weights`):** direzionale `ret` → 40% val_loss + 35% Spearman + 25% directional_acc; volatilità `log_rv` → 65% val_loss + 35% Spearman + 0% directional_acc (sulla varianza la dir_acc non è tradabile). Softmax(T=2) per tutti; lo score massimo = *primary teacher*, gli altri restano teacher pesati nel pool.
- **2c — Distillation:** ogni student riceve soft labels combinate (media pesata). Loss `(1−α)·NLL_reale + α·distill` con α=0.3, scala-normalizzata su μ/σ/ν. Soft labels nel TensorDataset (shuffle-safe). Epoche al 60%. Student già distillati skippati.

**EN** Default heterogeneous ensemble: **iTransformer + N-HiTS + TCN+Mamba** (LSTM removed 2026-05-14, val_NLL 5.28 vs iTransformer 0.18; code intact, reloadable for rollback). Phases:
- **2a — Candidate training:** each arch in `distillation.archs` with `n_ensemble=1` by default (override ONLY with an explicit `--n-ensemble N`; applies to candidates AND students). If `models/{arch}/best_model.pt` exists → skipped; force with `--force-download`.
- **2b — Multi-Teacher Scoring (target-aware, `teacher_score_weights`):** directional `ret` → 40% val_loss + 35% Spearman + 25% directional_acc; volatility `log_rv` → 65% val_loss + 35% Spearman + 0% directional_acc (on variance dir_acc is not tradable). Softmax(T=2) for all; the top score = *primary teacher*, the rest stay weighted teachers in the pool.
- **2c — Distillation:** each student gets combined soft labels (weighted mean). Loss `(1−α)·NLL_real + α·distill` with α=0.3, scale-normalized on μ/σ/ν. Soft labels in the TensorDataset (shuffle-safe). Epochs at 60%. Already-distilled students skipped.

🇮🇹 **Ensemble eterogeneo (inferenza), legge della varianza totale:**
- `mu_ens = Σ w_i · mu_i`
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)`

Pesi default in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`): iTransformer/N-HiTS/TCNMamba/TFT = 1.0, LSTM = 0.5. ⚠ Pesi dinamici inverse-NLL disponibili (`ensemble_nll_temperature`, default 0.05 ≈ uniforme) ma val_nll **anti-correla** col backtest: non sharpare finché lo shift val→test non è risolto.

**EN** **Heterogeneous ensemble (inference), law of total variance:** formulas above. Default weights in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`): iTransformer/N-HiTS/TCNMamba/TFT = 1.0, LSTM = 0.5. ⚠ Dynamic inverse-NLL weights available (`ensemble_nll_temperature`, default 0.05 ≈ uniform) but val_nll **anti-correlates** with the backtest: do not sharpen until the val→test shift is fixed.

🇮🇹 **Cambiare composizione (un solo posto):** `config/default.yaml → distillation.archs`:
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
Esempi: `["itransformer","lstm","tcnmamba"]` rollback legacy; `["itransformer","nhits","tcnmamba","lstm"]` ensemble a 4; `["itransformer","tcnmamba"]` solo 2. Dopo la modifica, `python run_all.py --distill` addestra i mancanti, fa scoring, distilla; backtest/live usano la nuova composizione. **Forzare un teacher:** `python run_all.py --distill --teacher itransformer` (salta lo scoring automatico). **Verifica:** in `models/{arch}/config.json` → `distilled: true`, `teacher_arch: "multi-teacher"`; per ri-distillare cancella `best_model.pt` o usa `--force-download`.

**EN** **Change composition (one spot only):** `config/default.yaml → distillation.archs` (block above). Examples: `["itransformer","lstm","tcnmamba"]` legacy rollback; `["itransformer","nhits","tcnmamba","lstm"]` 4-model; `["itransformer","tcnmamba"]` just 2. After editing, `python run_all.py --distill` trains the missing models, scores, distills; backtest/live pick up the new composition. **Force a teacher:** `python run_all.py --distill --teacher itransformer` (skips auto-scoring). **Verify:** in `models/{arch}/config.json` → `distilled: true`, `teacher_arch: "multi-teacher"`; to re-distill delete `best_model.pt` or use `--force-download`.

### 3.6 CAFN — coordinatore causale + training congiunto · causal coordinator + joint training

🇮🇹 `scripts/02d_cafn_joint_train.py` — **probe pre-registrato, inerte**. La **CAFN** (`quantsys/model/cafn.py`, attention a maschera strettamente causale + penalità causale = regolarizzatore prossimità+stabilità) estrae un **latente causale** dal tensore feature; i 3 modelli si allenano **in contemporanea** su quel segnale (loss congiunta end-to-end = Σ_arch MSE-mu + λ·penalità). Output **isolato** in `models/cafn/` (NON tocca `models/{arch}` né la parity live). Si addestra sul tensore **canonico 104-feature** (i dati Deribit grezzi sono forward-collected → solo canale `extra` futuro, no lookahead).

```bash
python scripts/02d_cafn_joint_train.py --smoke                 # valida il loop (CPU, dati sintetici)
python scripts/02d_cafn_joint_train.py --epochs 20             # reale (richiede data/lstm_dataset.npz)
```

🇮🇹 **GATE pre-registrato (val-first):** PASS sse CAFN-congiunto batte il baseline NO-CAFN (`latent=None`, stessi modelli/seed/epoche) di ≥3% MSE-mu su val per ≥2/3 modelli; altrimenti KILL. Integrazione parity-safe: kwarg `latent=None` nei 3 forward → path bit-identico al legacy. ⚠ 3 modelli + CAFN su 8GB → rischio OOM: riduci `--batch`/`--cafn-d-model`; NON in parallelo a poller/vol-paper.

**EN** `scripts/02d_cafn_joint_train.py` — **pre-registered, inert probe**. The **CAFN** (`quantsys/model/cafn.py`, strictly causal-masked attention + causal penalty = proximity+stability regularizer) extracts a **causal latent** from the feature tensor; the 3 models train **simultaneously** on it (end-to-end joint loss = Σ_arch MSE-mu + λ·penalty). Output **isolated** in `models/cafn/` (does NOT touch `models/{arch}` nor live parity). Trains on the **canonical 104-feature** tensor (raw Deribit data is forward-collected → optional future `extra` channel only, no lookahead). **Pre-registered GATE (val-first):** PASS iff joint-CAFN beats the NO-CAFN baseline (`latent=None`) by ≥3% val MSE-mu on ≥2/3 models, else KILL. Parity-safe: `latent=None` kwarg in the 3 forwards → bit-identical to legacy. ⚠ 3 models + CAFN on 8GB → OOM risk: lower `--batch`/`--cafn-d-model`; do not run alongside the poller/vol-paper.

### 3.7 Aggiungere una nuova arch · Adding a new architecture

🇮🇹
1. Classe in `quantsys/model/` con `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher in `quantsys/model/__init__.py:load_model`
3. Branch in `scripts/02_train.py` (`architecture == "X"`)
4. `config/arch/X.yaml`
5. `choices` in `run_all.py` (parser `--arch` e `--teacher`)
6. Whitelist in `scripts/05_analyze_signals.py` (la dashboard è il terminale opzioni Deribit, arch-independent)
7. (Opzionale) `distillation.archs` in `config/default.yaml`

**EN**
1. Class in `quantsys/model/` with `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher in `quantsys/model/__init__.py:load_model`
3. Branch in `scripts/02_train.py` (`architecture == "X"`)
4. `config/arch/X.yaml`
5. `choices` in `run_all.py` (`--arch` and `--teacher` parsers)
6. Whitelist in `scripts/05_analyze_signals.py` (the dashboard is the Deribit options terminal, arch-independent)
7. (Optional) `distillation.archs` in `config/default.yaml`

---

## 4. Valutazione · Evaluation

### 4.1 Backtest direzionale & analisi segnali · Directional backtest & signal analysis

🇮🇹 `scripts/03_backtest.py` (integrato in `run_all.py`) gira il backtest trading sul target direzionale (`ret`); `scripts/05_analyze_signals.py` analizza i segnali live. ⚠ Il backtest **non ha senso sui modelli vol** (`log_rv`/`log_rs_ratio`): usa i giudici dedicati (§4.3). I flag sperimentali in `03_backtest.py` sono **inerti di default** ed env-gated (`QUANTSYS_QUIET_*`, `QUANTSYS_RANK_*`, `QUANTSYS_DECISION_CADENCE`, `QUANTSYS_SIGMA_SCALE`, `QUANTSYS_MIN_EXPECTED_RET`, …): tutti validati e **FALLITI OOS** (vedi `CLAUDE.md` §NOMENCLATURA). **Val-first:** valida su `QUANTSYS_BACKTEST_SPLIT=val` (output suffissati `*_val.*`) prima di toccare il test split. **Backtest single-arch:** `QUANTSYS_BACKTEST_SINGLE_ARCH=1` (omogeneo per-arch invece dell'eterogeneo). Dopo ogni sweep azzera gli env sperimentali e rilancia un backtest pulito.

**EN** `scripts/03_backtest.py` (run inside `run_all.py`) runs the trading backtest on the directional target (`ret`); `scripts/05_analyze_signals.py` analyzes live signals. ⚠ The backtest is **meaningless on vol models** (`log_rv`/`log_rs_ratio`): use the dedicated judges (§4.3). The experimental flags in `03_backtest.py` are **inert by default** and env-gated (`QUANTSYS_QUIET_*`, `QUANTSYS_RANK_*`, `QUANTSYS_DECISION_CADENCE`, `QUANTSYS_SIGMA_SCALE`, `QUANTSYS_MIN_EXPECTED_RET`, …): all validated and **FAILED OOS** (see `CLAUDE.md` §NOMENCLATURA). **Val-first:** validate on `QUANTSYS_BACKTEST_SPLIT=val` (outputs suffixed `*_val.*`) before touching the test split. **Single-arch backtest:** `QUANTSYS_BACKTEST_SINGLE_ARCH=1` (per-arch homogeneous instead of heterogeneous). After each sweep clear the experimental envs and re-run a clean backtest.

### 4.2 Confronto architetture & parity · Architecture comparison & parity

```bash
python scripts/07_verify_teacher.py            # tabella comparativa archs · architecture comparison table
python scripts/99_replay_live_vs_training.py   # diagnostica parity live (BLOCKER #1) · live parity diagnostic
```
🇮🇹 `07_verify_teacher.py`: param count, forward time, Sharpe, WR, n trade, max DD, total return per ogni arch con `best_model.pt`. In alternativa: `models/{arch}/config.json` (`best_val_loss`, scaler, n_params), `models/{arch}/history.json` (curva loss), `results/{arch}/dashboard_results.json` (export backtest; non più letto dalla dashboard).

**EN** `07_verify_teacher.py`: param count, forward time, Sharpe, WR, n trades, max DD, total return for every arch with `best_model.pt`. Alternatively: `models/{arch}/config.json` (`best_val_loss`, scaler, n_params), `models/{arch}/history.json` (loss curve), `results/{arch}/dashboard_results.json` (backtest export; no longer read by the dashboard).

### 4.3 Giudici famiglia vol · Vol-family judges

🇮🇹 Con `features.target_type` in `config/default.yaml` il target cambia famiglia (default `ret` = direzionale, bit-invariato): `log_rv` = log-realized-variance delle prossime h barre (giudice QLIKE `scripts/vol/dev_vols_qlike.py` vs HAR-RV+naive); `log_rs_ratio` = asimmetria semivarianza `log(RS⁺/RS⁻)` (giudice MSE `scripts/vol/dev_vols_rs_judge.py` vs HAR-RS+naive+train-mean). Pipeline comune: `01_download_data.py` (rebuild dataset) → `python scripts/vol/dev_vols_macro_append.py` (ri-appende X_macro senza rifare il walk-forward regime, ~5s — il `01b` completo su 7 anni costa ~3h) → `02_train.py --n-ensemble 5` → giudice con `QUANTSYS_VOLS_SPLIT=val` (val-first; poi `=test` UNA volta). Report in `results/vols/`. Altri script vol attivi: `scripts/vol/step0_xarch_corr.py` (STEP 0 kill-check correlazione cross-arch), `scripts/vol/wf_har_baseline.py` (baseline HAR per-fold del walk-forward). ⚠ NO backtest trading sui modelli vol. **Esiti:** `log_rv` 2026-06-10 **PASS a 1h / FAIL a 1m**; `log_rs_ratio` 2026-06-11 **FAIL** (asimmetria impredicibile per NN e HAR-RS → i momenti pari generalizzano, i dispari no). Backup: vol-1h PASS in `models/backup_1h_vols/`, vol-1m FAIL in `models/backup_1m_vols/`.

**EN** Via `features.target_type` in `config/default.yaml` the target switches family (default `ret` = directional, bit-invariant): `log_rv` = log realized variance of the next h bars (QLIKE judge `scripts/vol/dev_vols_qlike.py` vs HAR-RV+naive); `log_rs_ratio` = semivariance asymmetry `log(RS⁺/RS⁻)` (MSE judge `scripts/vol/dev_vols_rs_judge.py` vs HAR-RS+naive+train-mean). Shared pipeline: `01_download_data.py` (dataset rebuild) → `python scripts/vol/dev_vols_macro_append.py` (re-appends X_macro without re-running the regime walk-forward, ~5s — the full `01b` over 7 years costs ~3h) → `02_train.py --n-ensemble 5` → judge with `QUANTSYS_VOLS_SPLIT=val` (val-first; then `=test` ONCE). Reports in `results/vols/`. Other active vol scripts: `scripts/vol/step0_xarch_corr.py` (STEP 0 cross-arch correlation kill-check), `scripts/vol/wf_har_baseline.py` (per-fold HAR walk-forward baseline). ⚠ NO trading backtest on vol models. **Outcomes:** `log_rv` 2026-06-10 **PASS at 1h / FAIL at 1m**; `log_rs_ratio` 2026-06-11 **FAIL** (asymmetry unpredictable for NN and HAR-RS alike → even moments generalize, odd ones don't). Backups: vol-1h PASS in `models/backup_1h_vols/`, vol-1m FAIL in `models/backup_1m_vols/`.

### 4.4 Short-vol arm & IVS relative-value (linea vol monetizzazione) · Short-vol arm & IVS relative-value (vol monetization line)

🇮🇹 Script di ricerca della monetizzazione vol, GPU-free, in `scripts/vol/` (lanciare dalla root): `short_vol_arm.py` (sim offline del forward test short-vol), `short_vol_hist_backtest.py` (backtest storico strutturale FHS-GJR-GARCH 2019→2026), `short_vol_premium_validate.py` (robustness del premio VRP), `short_vol_regime_decomp.py` (decomposizione regime/anno + equity/drawdown); `ivs_scout.py` (scouting smile IV Deribit) + `ivs_rv_backtest.py` (backtest net-of-cost reversione residui smile — **KILL** netto). Helper condiviso `_chain_io.py` (lettura option-chain `data/iv/chain/` con cache LRU). Stato: short-vol = edge strutturale CONFERMATO (gate vero = live n≥20); IVS = KILL net-of-cost. Dettaglio in `STATUS.md`.

**EN** Vol-monetization research scripts, GPU-free, in `scripts/vol/` (run from root): `short_vol_arm.py` (offline short-vol forward-test sim), `short_vol_hist_backtest.py` (structural historical FHS-GJR-GARCH backtest 2019→2026), `short_vol_premium_validate.py` (VRP premium robustness), `short_vol_regime_decomp.py` (regime/year decomposition + equity/drawdown); `ivs_scout.py` (Deribit IV smile scouting) + `ivs_rv_backtest.py` (net-of-cost smile-residual reversion backtest — net **KILL**). Shared helper `_chain_io.py` (option-chain reader for `data/iv/chain/` with LRU cache). Status: short-vol = structural edge CONFIRMED (true gate = live n≥20); IVS = net-of-cost KILL. Detail in `STATUS.md`.

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

🇮🇹 Avviato da `run_all.py` (fase live, salvo `--skip-live`) o da `python scripts/04_live_signals.py`. Path: `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build`→`_deterministic_predict`→`denormalize_predictions`→`SignalGenerator` (parity bit-perfect col backtest). Paper-only, nessun ordine reale. ⚠ Backtest direzionale negativo OOS → il paper-trading accumula solo trade reali. **NON** girare live + training/inferenza GPU in parallelo (contesa CUDA).

**EN** Started by `run_all.py` (live phase, unless `--skip-live`) or by `python scripts/04_live_signals.py`. Path: `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build`→`_deterministic_predict`→`denormalize_predictions`→`SignalGenerator` (bit-perfect parity with the backtest). Paper-only, no real orders. ⚠ Directional backtest negative OOS → paper-trading only accumulates real trades. Do **NOT** run live + GPU training/inference in parallel (CUDA contention).

### 5.3 Collector forward & vol-paper · Forward collectors & vol-paper

🇮🇹 I 3 processi detached (poller IV Deribit, vol-paper, recorder order-book L2) **NON sono servizi**: dopo un riavvio muoiono e vanno rilanciati. Usa il percorso `.venv` **ESPLICITO** (evita l'ambiguità `python`→interprete base):

```powershell
$py = "E:\quantsys_project\.venv\Scripts\python.exe"
Start-Process -WindowStyle Hidden -WorkingDirectory "E:\quantsys_project" -FilePath $py -ArgumentList "scripts/01c_iv_poller.py"
Start-Process -WindowStyle Hidden -WorkingDirectory "E:\quantsys_project" -FilePath $py -ArgumentList "scripts/04b_vol_paper.py","--execute"
Start-Process -WindowStyle Hidden -WorkingDirectory "E:\quantsys_project" -FilePath $py -ArgumentList "scripts/01d_orderbook_recorder.py"
```

🇮🇹 **Stop** (mirato sulla command line → cattura stub+worker, non tocca altri `python.exe`; `-Force` perché `-WindowStyle Hidden`):

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match '01c_iv_poller|04b_vol_paper|01d_orderbook_recorder' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

🇮🇹 Per fermarne **uno solo**, restringi il regex (es. `'01d_orderbook_recorder'`). Verifica: rilancia lo stesso `Get-CimInstance ... | Select-Object ProcessId` senza il `ForEach-Object` → deve tornare vuoto. **Salute:** conta i processi **LOGICI** non OS — ogni `.venv\python.exe` è stub+worker = 2 OS, attesi 3 logici (confronta i `ParentProcessId`). Crescita attesa: `data/iv/atm_30h.parquet` ~144 righe/g, `results/vol_paper/forecasts.parquet` ~24 righe/g, `data/orderbook/l2_features_*.parquet` ~17k righe/g (a 5s). Log vivi in `logs/quantsys_*.log` (più recenti per mtime), **non** i redirect `iv_poller.log`/`vol_paper.log`. ⚠ NON girare training/inferenza GPU in parallelo a `04b` senza fermarlo.

**EN** The 3 detached processes (Deribit IV poller, vol-paper, L2 order-book recorder) **are NOT services**: they die on reboot and must be relaunched. Use the **EXPLICIT** `.venv` path (avoids the `python`→base-interpreter ambiguity); blocks above. **Stop** all 3 (matched on the command line → catches stub+worker, leaves other `python.exe` untouched; `-Force` because `-WindowStyle Hidden`). To stop **a single one**, narrow the regex. Verify: re-run the same `Get-CimInstance ... | Select-Object ProcessId` without `ForEach-Object` → must come back empty. **Health:** count **LOGICAL** processes, not OS — each `.venv\python.exe` is stub+worker = 2 OS, expected 3 logical (compare `ParentProcessId`). Expected growth: `data/iv/atm_30h.parquet` ~144 rows/day, `results/vol_paper/forecasts.parquet` ~24 rows/day, `data/orderbook/l2_features_*.parquet` ~17k rows/day (at 5s). Live logs in `logs/quantsys_*.log` (newest by mtime), **not** the `iv_poller.log`/`vol_paper.log` redirects. ⚠ Do NOT run GPU training/inference in parallel with `04b` without stopping it.

#### 5.3bis Collector 24/7 su VPS · 24/7 collectors on the VPS

🇮🇹 `01c`+`01d` girano come **servizi systemd** su un VPS EU always-on (netcup, DEPLOYED 2026-07-14; **host/IP privati: SOLO in `config/secrets.yaml` → blocco `vps:`**, mai nel repo/doc). Deploy completo: `deploy/vps/README.md` (geo-test 451 Binance → deploy key → `setup_vps.sh` one-shot → verify). Sync verso casa dalla root di progetto:

```powershell
.\avvio_sessione.ps1              # TUTTO-IN-UNO alla riaccensione: pull+merge VPS + rilancio 01c/04b (anti-dup)
.\scripts\vps\pull_vps_data.ps1   # solo sync: host da secrets.yaml → scp → data/vps_staging/ + merge + heartbeat
```

🇮🇹 Il merge (`scripts/vps/merge_vps_data.py`) deduplica i tick doppi (casa accesa + VPS = by design) e avvisa se l'ultimo tick VPS è stale (default 3h → collector remoto giù). La copia canonica resta `data/iv/`+`data/orderbook/` a casa; il VPS è continuità + ridondanza dell'asset IV. Con PC acceso i collector locali possono restare attivi (`04b` legge il file locale, staleness ≤30 min).

**EN** `01c`+`01d` run as **systemd services** on an always-on EU VPS (netcup, DEPLOYED 2026-07-14; **host/IP private: ONLY in `config/secrets.yaml` → `vps:` block**, never in repo/docs). Full deploy: `deploy/vps/README.md` (Binance 451 geo-test → deploy key → one-shot `setup_vps.sh` → verify). Sync back home from the project root: block above (host from secrets → `scp → data/vps_staging/` + merge + heartbeat). The merge (`scripts/vps/merge_vps_data.py`) dedups double ticks (home on + VPS = by design) and warns when the latest VPS tick is stale (default 3h → remote collector down). The canonical copy stays home (`data/iv`+`data/orderbook`); the VPS provides continuity + redundancy for the IV asset. With the PC on, the local collectors may keep running (`04b` reads the local file, ≤30 min staleness).

#### Poller IV Deribit · Deribit IV poller

🇮🇹 `python scripts/01c_iv_poller.py` (loop, default 10 min; `--minutes N` cadenza, `--once` smoke, `--backfill-dvol` storico DVOL orario 2021→oggi, `--greeks` **inerte di default**: +3 chiamate/tick per i greeks del venue dello straddle ATM ~tenor-30h → `atm_greeks.parquet`, selezione identica a `pick_straddle` di `04b`; attivazione prevista sul VPS post-gate v1, STATUS 2bis-①) — 2 richieste pubbliche Deribit per tick, NESSUN account. Output append-only atomico in `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (snapshot raw, ~950 strumenti/tick), `atm_30h.parquet` (ATM IV delle 4 expiry vicine + IV interpolata in varianza totale a tenor costante 30h = forecast_horizon vol), `dvol.parquet` (controllo 30d). Scopo: storico IV short-tenor (non gratis altrove) per il gate **NN-RV vs IV implicita**.

**EN** `python scripts/01c_iv_poller.py` (loop, default 10 min; `--minutes N` cadence, `--once` smoke, `--backfill-dvol` hourly DVOL history 2021→today, `--greeks` **inert by default**: +3 calls/tick for the venue greeks of the ATM ~30h-tenor straddle → `atm_greeks.parquet`, selection identical to `04b`'s `pick_straddle`; planned activation on the VPS post-v1-gate, STATUS 2bis-①) — 2 public Deribit requests per tick, NO account. Atomic append-only output under `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (raw snapshot, ~950 instruments/tick), `atm_30h.parquet` (ATM IV of the 4 nearest expiries + total-variance-interpolated IV at constant 30h tenor = vol forecast_horizon), `dvol.parquet` (30d control). Purpose: short-tenor IV history (not free elsewhere) for the **NN-RV vs implied IV** gate.

#### Recorder order-book L2 Binance (B1) · Binance L2 order-book recorder (B1)

🇮🇹 `python scripts/01d_orderbook_recorder.py` (loop, default 5s; `--seconds N` cadenza, `--once` smoke, `--symbol` default BTCUSDT, `--levels` profondità REST default 1000) — 1 richiesta pubblica Binance `/api/v3/depth` per tick (no auth, weight 50/call → a 5s = 600/min ≪ 1200). Strada **B1**: raccolta FORWARD della microstruttura come fonte NUOVA per un edge direzionale a 1m (le 104 feature OHLCV sono sature). Output append-only atomico `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/giorno, dedup su `timestamp`): mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, depth cumulata 5/10/25/50 bps, total qty, **OFI best-level** (Cont-Kukanov-Stoikov) + **top-25 livelli raw/lato** come list-column. ⚠ `ofi_best` è NaN al 1° tick di ogni processo e in `--once`.

**EN** `python scripts/01d_orderbook_recorder.py` (loop, default 5s; `--seconds N` cadence, `--once` smoke, `--symbol` default BTCUSDT, `--levels` REST depth default 1000) — 1 public Binance `/api/v3/depth` request per tick (no auth, weight 50/call → at 5s = 600/min ≪ 1200). Track **B1**: FORWARD collection of microstructure as a NEW source for 1m directional edge (the 104 OHLCV features are saturated). Atomic append-only output `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/day, dedup on `timestamp`): mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, cumulative depth 5/10/25/50 bps, total qty, **best-level OFI** (Cont-Kukanov-Stoikov) + **top-25 raw levels/side** as list-columns. ⚠ `ofi_best` is NaN on the 1st tick of each process and in `--once`.

#### Forward test vol-paper (NN-RV vs IV, testnet Deribit) · Vol-paper forward test

🇮🇹 `python scripts/04b_vol_paper.py` (loop orario a hh:00+90s; `--once` smoke, `--execute` per ordini REALI sul testnet — default fill SIMULATI al mark price; `--arch` dir modelli, default `itransformer`). Forecast NN-RV 30h (modello vol-1h PASS, inversione completa `μ·IQR+centro`, feature dal path parity-blessed) vs varianza implicita dal poller IV (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; |edge|>0.25 → straddle ATM daily ~30h LONG/SHORT, max 1 posizione, hold a scadenza (cash settlement). Richiede: poller IV attivo, key in `secrets.yaml` blocco `deribit_testnet:` (URL DEVE essere test.deribit.com — assert anti-mainnet). Output: `results/vol_paper/{forecasts.parquet,trades.jsonl,position.json}` (il log forecasts si scrive anche flat, per le baseline). ⚠ NON girare training GPU in parallelo (5 modelli CUDA residenti).

**EN** `python scripts/04b_vol_paper.py` (hourly loop at hh:00+90s; `--once` smoke, `--execute` for REAL testnet orders — default SIMULATED mark-price fills; `--arch` model dir, default `itransformer`). 30h NN-RV forecast (PASS vol-1h model, full `μ·IQR+center` inversion, parity-blessed feature path) vs implied variance from the IV poller (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; |edge|>0.25 → ~30h daily ATM straddle LONG/SHORT, max 1 position, hold to expiry (cash settlement). Requires: IV poller running, keys in `secrets.yaml` `deribit_testnet:` block (URL MUST be test.deribit.com — anti-mainnet assert). Output: `results/vol_paper/{forecasts.parquet,trades.jsonl,position.json}` (the forecasts log is written even when flat, for the baselines). ⚠ Do NOT run GPU training in parallel (5 CUDA-resident models).

🇮🇹 **Baseline del gate** — `python scripts/04c_vol_paper_baselines.py` (read-only, GPU-free; `--no-fetch` = solo cache delivery, `--min-trades N` = soglia valutabilità, default 30). Gate (2) pre-registrato: il P&L NN deve battere ENTRAMBE le baseline always-long-vol e always-short-vol sullo STESSO calendario di expiry (isola il timing dal variance risk premium medio). Metodo: replay del loop `04b` su `forecasts.parquet`, premio ricostruito dai chain snapshot (`data/iv/chain/*.parquet`), delivery price dall'endpoint pubblico Deribit (cache `delivery_cache.json`). I gate (1) P&L medio>0 e (3) hit-rate>0.5 si leggono dai trade REALI in `trades.jsonl`. Output: `results/vol_paper/baseline_report.json` (scrive report + warning "non valutabile" finché n<min-trades).

**EN** **Gate baselines** — `python scripts/04c_vol_paper_baselines.py` (read-only, GPU-free; `--no-fetch` = delivery cache only, `--min-trades N` = evaluability threshold, default 30). Pre-registered gate (2): the NN P&L must beat BOTH the always-long-vol and always-short-vol baselines over the SAME expiry calendar (isolates timing from the average variance risk premium). Method: replay of the `04b` loop over `forecasts.parquet`, premium reconstructed from chain snapshots (`data/iv/chain/*.parquet`), delivery price from the public Deribit endpoint (`delivery_cache.json`). Gates (1) mean P&L>0 and (3) hit-rate>0.5 are read from the REAL trades in `trades.jsonl`. Output: `results/vol_paper/baseline_report.json` (writes report + "not evaluable" warning while n<min-trades).

### 5.4 Dashboard — Deribit Options Risk Terminal

```bash
python scripts/06_dashboard.py     # avvio diretto · direct launch
python run_all.py --only-dashboard # idem (no ML, no feed live · no ML, no live feed)
```

🇮🇹 `scripts/06_dashboard.py` è il **terminale opzioni crypto** (server HTTP single-file + SPA Plotly.js), **indipendente dalla pipeline ML** e GPU-free. Si connette ai dati **pubblici Deribit** (REST, no-auth) — nessuna chiave richiesta. Greche col segno (+/−) e nome tra parentesi (`Δ (Delta)`). Quattro tab:
1. **Volatility Surface** — superficie IV 3D (moneyness K/F × giorni), smile per scadenza, term structure ATM
2. **Option Chain** — chain call/put a doppio lato, Greche live (Black-Scholes forward: Δ, Γ, ν per +1% vol, Θ/giorno), strike ATM evidenziato
3. **Risk & Greeks** — OI per strike (call vs put), max-pain, Greche aggregate pesate per OI, put/call ratio, DVOL
4. **Trades** — forward test vol (`04b`): storico settled + **posizione aperta** (status `open`, campi settlement a `—`): lato LONG/SHORT straddle, prezzo ingresso/settlement, premio/payoff/PnL, profilo di rischio/payoff (click su una riga; formula di settlement identica a `04b`, **breakeven** nel titolo e in verde tratteggiato; la selezione sopravvive al refresh ~12s). Endpoint `/api/trades` (legge `results/vol_paper/trades.jsonl` + `position.json`)

Header live: spot BTC (index), DVOL 30d, ATM IV 30d, OI/volume totali, put/call ratio. Auto-refresh ~12s. Underlying via `config/default.yaml → dashboard.options_currency` (BTC|ETH); `auth_token` opzionale (constant-time, `X-Auth-Token` o `?token=`).

**EN** `scripts/06_dashboard.py` is the **crypto options terminal** (single-file HTTP server + Plotly.js SPA), **decoupled from the ML pipeline** and GPU-free. It connects to **Deribit public** data (REST, no-auth) — no key required. Greeks shown with sign (+/−) and name in parentheses (`Δ (Delta)`). Four tabs:
1. **Volatility Surface** — 3D IV surface (moneyness K/F × days), per-expiry smile, ATM term structure
2. **Option Chain** — two-sided call/put chain, live Greeks (Black-Scholes forward: Δ, Γ, ν per +1% vol, Θ/day), ATM strike highlighted
3. **Risk & Greeks** — OI by strike (call vs put), max-pain, OI-weighted aggregate Greeks, put/call ratio, DVOL
4. **Trades** — vol forward test (`04b`): settled history + **open position** (status `open`, settlement fields as `—`): LONG/SHORT straddle side, entry/settlement price, premium/payoff/PnL, risk/payoff profile (click a row; settlement formula identical to `04b`, **breakevens** in the title and as dashed green lines; selection survives the ~12s refresh). `/api/trades` endpoint (reads `results/vol_paper/trades.jsonl` + `position.json`)

Live header: BTC spot (index), 30d DVOL, 30d ATM IV, total OI/volume, put/call ratio. Auto-refresh ~12s. Underlying via `config/default.yaml → dashboard.options_currency` (BTC|ETH); optional `auth_token` (constant-time, `X-Auth-Token` header or `?token=`).

🇮🇹 ⚠ **Trappola `SO_REUSEADDR`:** un vecchio processo dashboard può tenere `:8050` e servire HTML **stale**. Prima di un nuovo smoke **uccidi il processo precedente** (stub+worker `.venv` = 1 processo logico): `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '06_dashboard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`. ⚠ **La verità finale è il browser con HARD RELOAD (Ctrl+Shift+R)**: una pagina già aperta gira il JS vecchio. Smoke server-side: l'HTML servito deve contenere `plot(` e `/api/risk` deve rispondere HTTP 200 con la chain reale. (Fix rendering definitivo 2026-06-24: asse X di `plot-oi`/`plot-payoff` passato a `type:'category'` per immunità alla corruzione SVG di Plotly al re-render; dettaglio in `STATUS.md`/`docs/MODEL_IMPROVEMENTS.md`.)

**EN** ⚠ **`SO_REUSEADDR` trap:** a stale dashboard process can keep `:8050` and serve **stale** HTML. Before a fresh smoke **kill the previous process** (`.venv` stub+worker = 1 logical process): `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '06_dashboard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`. ⚠ **The final truth is the browser with a HARD RELOAD (Ctrl+Shift+R)**: an already-open page runs the old JS. Server-side smoke: the served HTML must contain `plot(` and `/api/risk` must return HTTP 200 with the real chain. (Definitive rendering fix 2026-06-24: `plot-oi`/`plot-payoff` X axis switched to `type:'category'` for immunity to Plotly's SVG re-render corruption; detail in `STATUS.md`/`docs/MODEL_IMPROVEMENTS.md`.)

### 5.5 Fermare tutto · Stopping everything

🇮🇹 `Ctrl+C` nel terminale di `run_all.py` (o del server dashboard): ferma la pipeline e, in un run completo, anche il feed live WebSocket. ⚠ Se la dashboard era detached (`Start-Process`), `Ctrl+C` non basta: usa lo `Stop-Process` mirato (§5.4). I 3 processi di sfondo (poller IV, recorder L2, vol-paper) si fermano col blocco *Stop* in §5.3.

**EN** `Ctrl+C` in the `run_all.py` terminal (or the dashboard server): stops the pipeline and, in a full run, the WebSocket live feed too. ⚠ If the dashboard was detached (`Start-Process`), `Ctrl+C` is not enough: use the targeted `Stop-Process` (§5.4). The 3 background processes (IV poller, L2 recorder, vol-paper) are stopped via the *Stop* block in §5.3.

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
│   ├── raw_candles.parquet      # OHLCV storico (1h 2019→oggi dal pivot 2026-06-09)
│   ├── features.parquet         # feature normalizzate (rigenerabile)
│   ├── lstm_dataset.npz         # windows X/y per training (~3 GB, rigenerabile da 01)
│   ├── funding_rate.parquet     # funding futures (completo dal 2019-09-10)
│   ├── macro_*.parquet          # FRED/yFinance
│   ├── regime_probs.parquet     # probabilità regime (index orario UTC)
│   ├── iv/                      # IV Deribit: chain/ (snapshot raw), atm_30h.parquet, dvol.parquet — NON rigenerabile
│   ├── orderbook/              # L2 features forward (B1)
│   └── backup_1m/               # raw_candles + regime_probs era-1m (rollback 1m = restore + retrain)
├── models/
│   ├── pipeline_state.pkl       # copia canonica (scritta da 01, guard anti-stale in 02)
│   ├── backup_1h_vols/          # vol-1h PASS autosufficiente (5 membri + state + raw/regime 1h)
│   ├── backup_1m_vols/          # vol-1m FAIL (record)
│   └── {arch}/                  # itransformer/ (production vol-1h), lstm/, cafn/ (probe), …
│       ├── best_model.pt        # checkpoint (best_model_0..4.pt per ensemble multi-seed)
│       ├── config.json          # iperparametri + flag distilled/teacher_arch + best_val_*
│       ├── history.json         # curva loss
│       └── pipeline_state.pkl   # scaler + feature config + target_scale + interval
├── results/
│   ├── {arch}/                  # dashboard_results.json, live_signals.jsonl
│   ├── vols/                    # report giudici vol
│   └── vol_paper/               # forecasts.parquet, trades.jsonl, position.json, baseline_report.json, exec_diag.jsonl (A6: bid/ask+greeks diagnostici / diagnostic), hedge_state.json + hedge_ledger.jsonl (v2, SOLO con --hedge / --hedge only)
├── tests/                       # pytest (test_recent_fixes.py, test_live_training_parity.py)
├── scripts/
│   ├── 00_*..99_*              # spine numerato (fase pipeline) + 99_replay
│   ├── vol/                    # linea vol: giudici, short-vol arm, IVS, baseline HAR, _chain_io
│   ├── research/               # paper_01_dir_baselines.py (negative-control direzionale)
│   ├── archive/                # probe morti: xs_01/02/03 (KILL), dev_step0_regime_sigma
│   └── README.md               # mappa completa script→linea
└── logs/quantsys_YYYYMMDD_HHMMSS.log
```

🇮🇹 ⚠ I riferimenti `file:linea` in questa guida marciscono a ogni edit del codice: verificali con `grep` prima di affidartici. Mappa completa script→linea in `scripts/README.md`.

**EN** ⚠ The `file:line` references in this guide rot on every code edit: verify them with `grep` before relying on them. Full script→line map in `scripts/README.md`.
