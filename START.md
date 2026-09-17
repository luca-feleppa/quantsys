🇬🇧 English · [🇮🇹 Italiano](START.it.md)

# QUANTSYS — Operations & quick-start guide

> **Documentation rule:** every change to the docs must be applied to both the English and the Italian version in the same commit.

**Operations runbook** for the BTC/USDT forecasting engine (`quantsys/`). Production timeframe: **1h candles** (`data.interval: 1h`); the engine is **interval-agnostic** (every window conversion is an identity at 1m). This file owns the **commands**: mathematical derivations live in `THEORY.md`, the overview/motivation in `README.md`, current status and open gates in `STATUS.md`.

## Map

Guide ordered by **operational phase**: **0. Status & quick commands** → **1. Setup & dependencies** (install, Windows/PowerShell, hardware, interval parameters) → **2. Data** → **3. Modeling** (train / walk-forward / Optuna / distillation / CAFN) → **4. Evaluation** (backtest / verify / vol-judge / short-vol) → **5. Deploy & inference** (session routine, live, vol-paper, forward collectors, dashboard) → **Appendix: file layout**. Every command is run from the **root** `E:\quantsys_project`.

---

## 0. Status & quick commands

### 0.1 Quick commands

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

⚠ On disk `models/` holds **only `itransformer/` (1h-vol production) and `lstm/` (legacy)**: `models/nhits` and `models/tcnmamba` were removed in the 2026-06-12 cleanup → they **must be retrained** before any heterogeneous run (`--distill`, ensemble backtest). Both archs remain valid `--arch` values.

### 0.2 System status

- **Production = the 1h vol line** (`features.target_type: log_rv`, **5-member** iTransformer in `models/itransformer/`): the project's only **OOS PASS** signal.
- **Short-vol arm** in forward test on Deribit testnet via `04b_vol_paper.py` — a **systemd service on the 24/7 VPS**, never at home (§5.3). Gate v1 (n=20) **FAILED 0/3 on 2026-07-18**; later gates are still accumulating sample.
- **Directional line** (`target_type: ret`, `03_backtest.py`/`04_live_signals.py`): **no OOS alpha at any timeframe** — code alive and bit-invariant, kept as a scientific *negative control*. Paper-only, no real orders.
- **z-score invariant:** every new entry point must call `PipelineState.denormalize_predictions(mu, sigma)` before the trading layer; with the `log_rv` target the full inversion is `μ·IQR + center`. Derivation in `THEORY.md` (§ *Critical invariant*).
- ⚠ **Current status, open gates and "resume here": `STATUS.md`** (current period + open gates; pre-2026-07-08 history in `docs/STATUS_ARCHIVE_2026H1.md`, read-only). This guide does **not** keep the experiment log.

---

## 1. Setup & dependencies

### 1.1 Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python scripts/00_check_setup.py
```

`00_check_setup.py` checks dependencies, CUDA, Binance, FRED — fix errors before proceeding. Its **PIPELINE STATE** section is different: it lists artifacts **produced** by the later scripts (dataset, checkpoints, reports), so on a fresh clone they are missing **by definition** and show up as `△` warnings, not errors — they do not feed the final verdict and there is nothing to "fix" before running the pipeline.  `00_test_binance_testnet.py` is an optional Binance-testnet connectivity smoke.

### 1.2 Windows / PowerShell operational notes

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

**CPU** — `config/default.yaml` (current value `1` = all cores; lower it to 0.5 to keep the machine usable during a long training):
```yaml
hardware:
  cpu_fraction: 1     # 0.3=30%, 0.5=50%, 1=100% dei core
```
Read by every script at startup.

**GPU compute** (RTX 2070 Super, min=125 max=215W):
```powershell
nvidia-smi -pl 125    # limita · throttle
nvidia-smi -pl 215    # ripristina · restore
```

**GPU sequencing (8GB):** parallel backtest/walk-forward OK; 5-seed × 3-arch training must be **sequential** (OOM). Do **NOT** run live/paper-trading/vol-paper + training/inference in parallel (CUDA contention, 5 resident models). TCN+Mamba is the bottleneck.

**Reference setup (RTX 2070 Super 8GB):**

| Component | Value |
|---|---|
| CUDA, AMP fp16 training | yes (via `setup_device`) |
| AMP inference | **off** hardcoded in `quantsys/model/ensemble.py` (avoids NaN from spectral_norm + Mamba scan) |
| Backtest inference batch | 256 (`BATCH_SIZE` in `scripts/03_backtest.py`) |
| Training batch | 64 (`config/default.yaml → training.batch_size`; the `arch/*.yaml` files do not override it) |

**Other configurations:** *CPU only* → automatic fallback via `setup_device` (`autocast` becomes a silent no-op): training 20–50× slower (not recommended), backtest ~5s → 30–60s (tolerable), live ~50–100ms vs ~20ms (usable, Binance WS latency dominates). *4GB VRAM* → in `config/default.yaml → training` (or overridden in `config/arch/<arch>.yaml`) `batch_size: 32` + `gradient_accumulation_steps: 2` (effective batch 64) and inference batch 256→128. *≥16GB VRAM* → `batch_size: 128`, inference batch up to 1024 (marginal gain). **Apple Silicon / AMD / Intel Arc: untested** (`torch.cuda.*` code; MPS would need `setup_device` changes + custom Mamba/SSM kernels).

### 1.4 1h interval parameters & 1m rollback

Current values in `config/default.yaml` (checked against the file). The **was (1m)** column is a **historical note**: it only explains the current calibration and serves the rollback.

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

**Interval guard (operational fail-fast):** `RuntimeError` "interval mismatch" in `03_backtest.py` and `04_live_signals.py` — a 1m-trained model + 1h config = invalid combination, blocked. Live/replay consumers derive the interval from `PipelineState.interval_minutes` (fallback 1 for legacy pkl), **never** from the config. σ safety net scaled to `0.05·√interval_minutes` (≈0.387 at 1h); annualization `bars_per_year = 525600 // interval_minutes` (1h→8760). **Resolution overlay:** `python run_all.py --interval 1h` (or `1m`) applies `config/interval/{interval}.yaml` on top of `default.yaml` (per-section shallow merge, after secrets and before the arch overlay) and propagates `QUANTSYS_INTERVAL` to subprocesses; `choices` derive from the files present in `config/interval/`.

**Rollback to 1m** (3-step procedure — no code change: every conversion is an identity at 1m):
1. restore `data/backup_1m/*` over the canonical copy in `data/`;
2. set the config to 1m: `data.interval: 1m`, `data.start_time: '2025-05-19'` (plus the *was (1m)* column above, or the `--interval 1m` overlay);
3. **mandatory retrain** — the 1m checkpoints were deleted in the 2026-06-12 cleanup, none is left on disk. The config↔state interval guard blocks any inconsistent combination anyway, before it can produce numbers.

---

## 2. Data

### 2.1 Download / update / macro

The pipeline downloads and prepares data before training. On first run let `run_all.py` execute all phases; on subsequent runs skip with `--skip-update --skip-macro` (use on-disk data).

| Script | Role |
|---|---|
| `scripts/01_download_data.py` | full Binance candles + funding download, npz dataset rebuild |
| `scripts/01_update_data.py` | incremental candle update to today, then feature + npz **rebuild** with **scaler refit** and `PipelineState` rewrite |
| `scripts/01_update_data.py --candles-only` | extends **only** `data/raw_candles.parquet` and stops: no scaler, no npz, no state. Use it when downstream models are **frozen** and only fresh OHLCV history is needed |
| `scripts/01b_download_macro.py` | FRED/yFinance macro + `RegimeMarkovBTC` regime walk-forward (hourly clock; **~3h over 7 years**) |

**`01b_download_macro.py` modes** (mutually exclusive; no flag = full pipeline):

| Flag | Effect |
|---|---|
| `--regime-only` | regenerates ONLY `regime_probs.parquet` + `regime_hmm.pkl` + `regime_wf_checkpoint.pkl` (skips macro/normalizer/npz; the walk-forward remains the cost) |
| `--regime-incremental` | **(B7)** appends new bars only, from the checkpoint: 0-1 MLE fits, minutes. `regime_hmm.pkl` is NOT updated (only a full rebuild redoes the final full-sample fit) |
| `--regime-bootstrap-checkpoint` | one-off checkpoint rebuild from existing pkl+parquet, with a built-in golden test (replay vs parquet, fail-fast) |
| `--skip-regime` | full macro pipeline leaving the regime detector **UNTOUCHED** — a full rebuild remaps regime indices: avoid while experiments are open (new bars are appended via `--regime-incremental`) |

**Produced data:** `data/raw_candles.parquet` = 1h candles 2019→today (~65k bars); `data/funding_rate.parquet` = full funding since the 2019-09-10 perp launch; `data/macro_*.parquet` = FRED/yFinance; `data/regime_probs.parquet` = regime probabilities (hourly UTC index); `data/features.parquet` + `data/lstm_dataset.npz` = normalized features and `X/y` windows for training (104 canonical = 86 dynamic + 18 structural; `X_train ≈ (51k, 120, 104)`). ⚠ `lstm_dataset.npz` is **large (~3 GB) and regenerable** from `01_download_data.py`: if missing, regenerate it before train/judge.

### 2.2 Forward collectors (non-regenerable data)

Three collectors gather **forward** history not freely available elsewhere. Since 2026-07-18 they run **on the VPS only** (systemd services): nothing to relaunch at home — session routine in *5.3*, deploy in *5.3bis*. The always-on VPS removes the PC-off gaps (measured IV coverage 18.6% of hours, 2026-06-12→07-14).
- **`01c_iv_poller.py`** — Deribit short-tenor IV → `data/iv/` (the ONLY non-regenerable data).
- **`01d_orderbook_recorder.py`** — Binance L2 order-book → `data/orderbook/` (B1 microstructure track).
- **`01e_trades_recorder.py`** — Deribit production option trades → `data/deribit_trades/` (realized spreads; ~24h API retention → forward only).

---

## 3. Modeling

### 3.1 Full pipeline

```bash
python run_all.py                    # menu: ↑↓ naviga, SPAZIO seleziona, A toggle all, INVIO conferma
python run_all.py --arch itransformer --force-download   # modalità diretta · direct mode
```
Without flags it shows the interactive menu and opens the dashboard at `http://localhost:8050`. Phases: data → macro → train → walk-forward → backtest → live → dashboard.

### 3.2 Single-architecture training

Each architecture has its own config in `config/arch/{arch}.yaml` and isolated outputs in `models/{arch}/` and `results/{arch}/`. No cross-run interference. ⚠ Times are estimated on the old 1m-525k dataset; the 1h dataset (~65k, ~8× smaller) is proportionally faster, **to be re-measured**.

| `--arch` | Command | Class (in `quantsys/model/`) | Time (1m-525k) |
|---|---|---|---|
| `itransformer` | `python run_all.py --arch itransformer --skip-update --skip-macro` | `QuantiTransformer` (`__init__.py`) — feature-wise attention, baseline | ~13–40 min |
| `nhits` | `python run_all.py --arch nhits --skip-update --skip-macro` | `QuantNHiTS` (`nhits.py`) — hierarchical pure-MLP | ~6–19 min |
| `tcnmamba` | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | `QuantTCNMamba` (`tcn_mamba.py`) — dilated convolutions + SSM, bottleneck | ~80 min/seed |
| `lstm` | `python run_all.py --arch lstm --skip-update --skip-macro` | `QuantLSTM` (`__init__.py`) — legacy, underperforming | (legacy) |

`--skip-update --skip-macro`: use on-disk data without redownload (omit on first run). Direct CLI equivalent: `$env:QUANTSYS_ARCH="<arch>"; python scripts/02_train.py --n-ensemble <N>`. ⚠ Only `models/itransformer` and `models/lstm` exist on disk: `nhits`/`tcnmamba` start from scratch (2026-06-12 cleanup).

### 3.3 Homogeneous ensemble (5× same arch)

```yaml
# config/default.yaml
training:
  n_ensemble: 5   # default single-arch = 5; --distill default = 1, override esplicito con --n-ensemble
```
Output: `models/{arch}/best_model_0..4.pt`. Backtest/live load them via `EnsembleModel.load`. Independent from distillation (modes are not mutually exclusive).

### 3.4 Walk-forward & Optuna

**Walk-forward** (`scripts/02b_walkforward_validate.py`, run inside `run_all.py`): purged k-fold with anti-leakage embargo. `n_folds=6` → **5 effective folds** (fold 0 structurally skipped). ⚠ In-sample walk-forward metrics (Spearman/WHR) **anti-correlate** with the directional backtest: do not use them to optimize.

```bash
set QUANTSYS_ARCH=lstm                                # cmd.exe; PowerShell: $env:QUANTSYS_ARCH="lstm"
python scripts/02c_optuna_search.py --n-trials 50 --study-name quantsys
```
**Optuna** is **hardcoded to `QuantLSTM`**. `best_params.json` saved in `models/lstm/` is NOT auto-applied: copy it manually into `config/arch/lstm.yaml`. Study persists on SQLite (`models/lstm/optuna_quantsys.db`), resumable any time.

### 3.5 Multi-teacher distillation

```bash
python run_all.py --distill --skip-update --skip-macro
python run_all.py --distill --teacher itransformer   # forza il primary teacher, salta lo scoring · force primary teacher, skip scoring
```

**Composition — the single edit point:** `config/default.yaml → distillation.archs`.
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
After editing, `python run_all.py --distill` trains what is missing → scores → distills; backtest and live pick up the new composition. Examples: `["itransformer","lstm","tcnmamba"]` (legacy rollback), `["itransformer","tcnmamba"]` (only 2).

**Operational notes:**
- `n_ensemble = 1` by default on the `--distill` path (for candidates **and** students); override only with an explicit `--n-ensemble N` on the CLI.
- If `models/{arch}/best_model.pt` exists the arch is skipped; to retrain/re-distill delete the checkpoint or pass `--force-download`.
- **Check the outcome:** `models/{arch}/config.json` → `distilled: true`, `teacher_arch: "multi-teacher"`.
- ⚠ `models/nhits` and `models/tcnmamba` no longer exist on disk (2026-06-12 cleanup): the first `--distill` retrains them from scratch, timings in §3.2.
- Theory (target-aware `teacher_score_weights` scoring, μ/ls²/lnu soft labels, `(1−α)·NLL + α·distill` loss, law of total variance, `DEFAULT_ARCH_WEIGHTS` and `ensemble_nll_temperature`): **`THEORY.md` § Knowledge Distillation**.

### 3.6 CAFN — joint training

```bash
python scripts/02d_cafn_joint_train.py --smoke        # valida il loop (CPU, dati sintetici) · loop smoke (CPU, synthetic)
python scripts/02d_cafn_joint_train.py --epochs 20    # reale, richiede data/lstm_dataset.npz · real run, needs the npz
```

**Pre-registered, inert probe.** Output **isolated** in `models/cafn/`: it touches neither `models/{arch}` nor live parity (`latent=None` kwarg in the 3 forwards → bit-identical to legacy). **Pre-registered gate (val-first):** PASS iff joint-CAFN beats the NO-CAFN baseline (same models/seeds/epochs) by **≥3% val MSE-mu on ≥2/3 models**, else KILL. Useful flags: `--archs`, `--batch`, `--d-latent`, `--cafn-d-model`, `--cafn-layers`, `--lambda-causal`, `--lr`, `--max-steps`, `--device`, `--no-gate`. ⚠ 3 models + CAFN on 8 GB → **OOM** risk: lower `--batch`/`--cafn-d-model` and never run it alongside the poller/vol-paper. Architecture and causal penalty: **`THEORY.md` § CAFN**.

### 3.7 Adding a new architecture

7-step procedure (detail in the `/add-arch` skill):

1. Class in `quantsys/model/` with `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher `load_model` in `quantsys/model/__init__.py`
3. Branch `architecture == "X"` in `scripts/02_train.py`
4. `config/arch/X.yaml`
5. `--arch` and `--teacher` parser `choices` in `run_all.py`
6. Whitelist in `scripts/05_analyze_signals.py` (the dashboard stays arch-independent)
7. Optional: `distillation.archs` in `config/default.yaml`

---

## 4. Evaluation

### 4.1 Directional backtest & signal analysis

`scripts/03_backtest.py` (run inside `run_all.py`) runs the trading backtest on the directional target (`ret`); `scripts/05_analyze_signals.py` analyzes live signals. Operational envs: `QUANTSYS_BACKTEST_SPLIT=val` (**val-first**, outputs suffixed `*_val.*` that do not clobber production; the test split is touched once the val gate passes) and `QUANTSYS_BACKTEST_SINGLE_ARCH=1` (per-arch homogeneous backtest instead of heterogeneous). ⚠ The backtest is **meaningless on vol models** (`log_rv`/`log_rs_ratio`): use the dedicated judges (§4.3). ⚠ The other `03_backtest.py` env flags form an **inert KILL corpus** (regime gating, rank-based entry, decision cadence, continuous exposure, σ-calibration): all validated and **FAILED OOS**, list and outcomes in `THEORY.md` §12.5 — do not re-test them. After each sweep clear the experimental envs and re-run a clean backtest.

### 4.2 Architecture comparison & parity

```bash
python scripts/07_verify_teacher.py            # tabella comparativa archs · architecture comparison table
python scripts/99_replay_live_vs_training.py   # diagnostica parity live (BLOCKER #1) · live parity diagnostic
```
`07_verify_teacher.py`: param count, forward time, Sharpe, WR, n trades, max DD, total return for every arch with `best_model.pt`. Alternatively: `models/{arch}/config.json` (`best_val_loss`, scaler, n_params), `models/{arch}/history.json` (loss curve), `results/{arch}/dashboard_results.json` (backtest export; no longer read by the dashboard).

### 4.3 Vol-family judges

`features.target_type` in `config/default.yaml` selects the target family (code default `ret`, bit-invariant):

| `target_type` | Target | Judge | Outcome |
|---|---|---|---|
| `ret` | cumulative log-return over h bars | `03_backtest.py` (§4.1) | no OOS alpha |
| `log_rv` | log realized variance Σr² over h bars | `scripts/vol/dev_vols_qlike.py` (QLIKE vs HAR-RV + naive) | **PASS at 1h** (−30% QLIKE), FAIL at 1m — 2026-06-10 |
| `log_rs_ratio` | semivariance asymmetry log(RS⁺/RS⁻) | `scripts/vol/dev_vols_rs_judge.py` (MSE vs HAR-RS + naive + train-mean) | FAIL 2026-06-11 |

Shared pipeline (from the root, PowerShell):

```powershell
python scripts/01_download_data.py                  # rebuild dataset npz
python scripts/vol/dev_vols_macro_append.py         # ri-appende X_macro senza rifare il walk-forward regime (~5s vs ~3h)
$env:QUANTSYS_ARCH="itransformer"; python scripts/02_train.py --n-ensemble 5
$env:QUANTSYS_VOLS_SPLIT="val"; python scripts/vol/dev_vols_qlike.py      # val-first; poi "test" UNA sola volta
```

Reports in `results/vols/`. ⚠ **NO trading backtest on vol models** (`03_backtest.py` is meaningless on a log-RV target). Self-contained backups: `models/backup_1h_vols/` (1h PASS), `models/backup_1m_vols/` (1m FAIL, kept as a record). The other vol-line judges and probes are listed in `scripts/README.md`. ⚠ Two levers that are **inert by default** live on this path and must not be enabled outside their pre-registered gate: `QUANTSYS_QLIKE_SMEARING=1` (smearing correction on the judge, C1) and `model.attn_entropy_lambda` (attention entropy penalty, A10 — overlay `config/arch/itransformer_a10_sparsity.yaml`, probe `scripts/vol/attn_entropy_probe.py`). Rationale and outcomes: `THEORY.md` §12.2 and §12.4.

### 4.4 Short-vol arm & IVS relative-value (vol monetization line)

GPU-free research scripts in `scripts/vol/`, **to be launched from the root**: the `short_vol_*` family (offline forward-test sim, 2019→2026 FHS-GJR-GARCH historical backtest, VRP premium robustness, regime/year decomposition) and the `ivs_scout.py`/`ivs_rv_backtest.py` pair (Deribit smile + residual reversion). **Status:** short-vol = structural VRP edge CONFIRMED on the historical backtest, but the real gate is `04b`'s **live forward test** (gate v1 n=20 **FAILED 0/3** on 2026-07-18; later gates accumulating — thresholds and counters in `STATUS.md`); IVS relative-value = **net-of-cost KILL** (would only live as a market-maker). Each script's role and flags: `scripts/README.md`.

---

## 5. Deploy & inference

### 5.1 Subsequent runs

```bash
python run_all.py --only-dashboard               # solo dashboard · dashboard only
python run_all.py --skip-train --skip-walkfwd    # aggiorna dati, stessi modelli · update data, same models
python run_all.py                                 # menu
python run_all.py --distill                       # full + distillation
```

**Useful flags:**

| Flag | Effect |
|------|---------|
| `--skip-update` | use existing dataset, no download |
| `--skip-macro` | skip FRED/yFinance download |
| `--skip-train` | use existing model, no retrain |
| `--skip-walkfwd` | skip walk-forward validation |
| `--skip-backtest` | skip backtest |
| `--skip-live` | no live WebSocket feed |
| `--skip-analyze` | skip `05_analyze_signals.py` |
| `--only-dashboard` | Options Risk Terminal only, no ML or live |
| `--no-browser` | do not auto-open browser |
| `--force-download` | redownload + force retrain |
| `--max-model-age-days N` | retrain if model older than N days |
| `--interval {1m,1h}` | candle-resolution overlay |
| `--n-ensemble N` | single-arch ensemble seeds (default 5) |
| `--distill` | multi-teacher pipeline |
| `--teacher ARCH` | force primary teacher |

### 5.2 Live / paper trading

Started by `run_all.py` (live phase, unless `--skip-live`) or by `python scripts/04_live_signals.py`. Production path: `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canonical, scaler from `PipelineState`) →`_deterministic_predict`→`denormalize_predictions`→`SignalGenerator`. **Bit-perfect feature and signal parity with training** (BLOCKER #1 closed 2026-06-05): regression in `tests/test_live_training_parity.py`, end-to-end diagnostic `python scripts/99_replay_live_vs_training.py` (Δfeature = Δμ = Δσ = 0). Paper-only, no real orders. Operational remainder: a real WS smoke test. ⚠ Directional backtest negative OOS → paper-trading only accumulates real trades. Do **NOT** run live + GPU training/inference in parallel (CUDA contention).

### 5.3 Session routine (home side)

Since 2026-07-18 the home PC is **passive**: no resident local process — collectors `01c`/`01d`/`01e` and vol-paper `04b` all run as **systemd services on the VPS** (§5.3bis). **One command per session**, from the project root:

```powershell
.\avvio_sessione.ps1          # [-Days 7] [-SkipPull] [-SkipMonitor] [-RefreshCandles]
```

| # | Block | Content |
|---|---|---|
| ① | **VPS pull + merge** | `scripts/vps/pull_vps_data.ps1`: **dated-vintage archiving** of `macro_features.parquet` → VPS (see below), scp collectors → `data/vps_staging/`, dedup merge into the canonical copy, **staleness heartbeat of the 4 collectors** (IV / L2 / trades / 04b), auto-cleaned staging |
| ② | **B7 regime freshness** | ≥168 hourly bars past the walk-forward checkpoint → `01b_download_macro.py --regime-incremental` in background (anti-dup if a `01b` is already alive); with frozen candles it prints "fresh" and is a no-op |
| ②bis | **Candle extension** — *only with `-RefreshCandles`, off by default* | `01_update_data.py --candles-only`: extends `data/raw_candles.parquet` and stops (no scaler, no npz, no state). **It is not a default** and the reason is in the dedicated paragraph below: automating it would remove the ability to **freeze the data**. Placed **after** ② on purpose, so any regime refresh starts at the next startup |
| ③ | **Vol line monitoring** (CPU-only, **write-free**) | incremental `scripts/vol/derive_mfiv.py` (**after** the merge by construction: it reads the chain just downloaded; it prints the MFIV−ATM wedge, which is a **permanent diagnostic**, not a gate counter) + counter of executed option legs (`executed` in `trades.jsonl`, historical threshold n≥30: the gate is closed, but it remains **the only number telling whether `04b` is executing**) + **1m bar file coverage** (see below) + **E1 stage-2 sample** (`edge_information_judge.py --stage 2 --count-only` toward n≥40, preceded by how far the close series reaches). ⚠ The **hedged** counter was **retired on 2026-08-13**, when `--hedge` was removed from the unit: its gate had been closed since 11/08 and its last consumer was the check that the wind-down band did not open new hedges. It must not be restored "for information" — a number without a consumer invites comparison with a threshold that is no longer its own, and it is the counter that read the wrong unit three times (events ≠ positions ≠ settlements) |

⚠ **One-shot discipline — a permanent invariant of the routine.** Block ③ **never** runs a pre-registered judge: the remaining counters compute **counts only** (no PnL, no edge, no correlation), and every judge still guards `n < n_min → NO_RUN` without writing a report → automating a count **cannot produce peeking**. ⚠ A `NO_RUN` guard, however, protects only **below** threshold, and does not protect the side effects preceding it at all (network fetches, cache rewrites): hence the routine uses the `--count-only` variants and the **one-shot run stays MANUAL**. ⚠ The MFIV comparator's `--count-only` counter was **retired on 2026-08-18**, when its gate closed (FAIL at n=41): `derive_mfiv.py` stays because it maintains a diagnostic column, not because it feeds a gate. Fail-soft: a failing step logs a warning without stopping the routine (`-SkipPull` skips ①, `-SkipMonitor` skips ③; `-Days N` = pull window; `-RefreshCandles` **adds** step ②bis, see below).

⚠ **1m bar file coverage (since 2026-08-06) — recorder continuity and target availability are two different things.** The L2 check measures **recorder** continuity; this one measures whether the **target** those hours would be judged against exists. The B1 judge builds RV from **1-minute** returns (`data/raw_candles_1m_l2.parquet`, ~180 squares per observation instead of 3), so L2 hours past the end of that file are **collected but target-less**: the usable sample is the **minimum of the two**, and until today neither counter said so. ⚠ **The cap applies to 1m-target analyses** (B1 at h=3, pin-close proxies): the `n_eff` counter at h=30 printed by the L2 check uses **hourly** bars — the production target is 30 bars of hourly returns — and is **unaffected** by this file. ⚠ **The file has three consumers and zero producers — and zero producers by decision (since 2026-08-10)**: `l2_incremental_judge.py`, `pin_close_feasibility.py` and `edge_information_judge.py` (as a tail source) read it, but **no script in the repo generates or extends it**. This is not a backlog: 1-minute klines are **historical and re-downloadable indefinitely**, so — unlike the L2 recorder, which is forward-only and unrecoverable if it stops — **nothing is lost by waiting**, and the file downloaded today is identical to the one downloaded six months from now. The "lag" is a download not yet done, not a maturing debt. Hence the monitor **only measures** and prints no remedy: when a 1m-target analysis is actually reopened, the download is decided then and over the window it needs, instead of continuously keeping current a file that **no operational path reads** (verified 2026-08-10: no reference on the VPS, and the file's actual contribution to `hourly_close()` measured = **0 rows**, because the 1m data is *behind* the hourly bars). State on 2026-08-10: 87,217 bars, 2026-06-01 → 2026-07-31, **10 days of L2 without a 1m target**.

⚠ **`-RefreshCandles` (since 2026-08-06) — extending the bars is an act, not an automation.** The flag runs `01_update_data.py --candles-only` (block ②bis, after the B7 check and before monitoring) and is **off by default**, modelled on `-PromoteMacro` for the same reason that one exists. The mechanics would be safe even automated — **no-op** when nothing is new (it exits *before* writing, mtime unchanged), dedup keeping the **existing** row so history is never rewritten, never the in-progress bar (`floor(now) − 1 bar`), atomic write, and the file is **never shipped to the VPS** (the only home→VPS transfer is macro) — but automating it would remove the ability to **freeze the data** for a pre-registered experiment: the *"candles/npz/regime_probs untouched until the gate closes"* invariant would go from "do nothing" to "remember `-SkipMonitor`", which is exactly the shape of the macro promotion that happened **by automation** on 31/07. And since the split is a **fraction of the row count**, every appended bar shifts the train/val/test boundaries at the next npz rebuild: under automatic extension the dataset vintage would become a function of *how many sessions you opened*, i.e. no longer declarable. **When to use it:** before recording the E1 counter if block ③ prints the lag warning, and before the one-shot run of **E1 stage 2** (prerequisite ⑤ of its pre-reg). Among the open gates **only E1 reads the bars** — the hedged judge reads the ledgers and perp funding, the MFIV comparator the chain and `forecasts.parquet`. **When NOT to use it:** inside an experiment declaring frozen data, and when reproducing a historical report over a window that assumes an exact bar count. ⚠ **Deliberate one-session delayed effect:** extending the bars moves the B7 staleness counter forward, so the incremental regime refresh starts, if at all, at the **next** startup — the two writes stay separate and each visible on its own instead of chaining silently.

⚠ **The `NO_RUN` guard only protects BELOW threshold — why E1 entered the routine as `--count-only`, not `--stage 2` (2026-08-06).** Below n=40 `edge_information_judge.py --stage 2` prints the count and stops, but **at n≥40 the same command computes the three conditions, prints the verdict and writes the report**: automating it bare would have fired the confirmatory run **by automation rather than by decision**, on the day the threshold is met and without anyone choosing it. `--count-only` stops at the count at **any** n, and a test verifies this on a sample built **above** threshold (`tests/test_edge_information_judge.py`) — below threshold the test would prove nothing, since the bare command would pass too. ⚠ **The count depends on the close series**: an expiry is observable only if its RV is computable, so the block prints **first** how far that series reaches and warns past a 6h lag. The remedy (`01_update_data.py --candles-only`) is **deliberately not automated**: it writes to a data file, and block ③ stays write-free.

⚠ **Dated macro vintages (since 2026-07-31) — the push is no longer a decision, promoting is.** `04b` reads `data/macro_features.parquet` at its nightly bootstrap and **freezes** it for the whole day: overwriting it inside a pre-registered forward sample silently changes its input (happened 2026-07-31, breakpoint dated in `STATUS.md`). The pull therefore: (a) **always archives** the file as `data/macro/macro_features_<YYYYMMDD>.parquet` on the VPS, where `<YYYYMMDD>` is the **last index date** (`scripts/vps/macro_vintage.py`) — append-only, 716 KB per copy, so every forward decision stays traceable to its vintage; (b) treats the canonical as a **symlink** into the archive, so the live vintage is readable with `readlink` and visible in `ls -l`; (c) **never repoints it** without `-PromoteMacro`, and on a diverging vintage emits a warning while leaving the live path on what it was already using. The new vintage takes effect at the **next 04b bootstrap (00:30 UTC)**: with an open forward sample, the promotion must be **dated in `STATUS.md`**.

```powershell
.\scripts\vps\pull_vps_data.ps1 -PromoteMacro   # atto deliberato: ripunta il canonico VPS al vintage locale
```

⚠ **Instrument vs state — the pinned `MacroNormalizer` (implemented 2026-07-31, INERT).** Dated vintages settle *which* macro reaches the VPS; what remains is that `VolForecaster` **refits** the `MacroNormalizer` whole-df at every bootstrap, so extending the parquet moves median and IQR and **the measuring instrument changes together with the state it must measure** (on the 31/07 breakpoint: 2.7% of the total variation). `scripts/vol/pin_macro_normalizer.py` freezes the instrument at a **declared** vintage and writes it to a pickle; `04b` and the replay accept it via `--macro-norm <path>`. **Without the flag behavior is bit-identical** (verified end-to-end on the production parquet: 0 differences over 90 columns). ⚠ The vintage `models/itransformer` was trained under is **not reconstructible**: the pin does not recover one, it **fixes** one. Enabling it today would be a **content no-op** (same vintage → delta 0): it serves to prevent *future* drift, not to correct the past. The artifact is gitignored but **reproducible** from an archived vintage.

⚠ **EMERGENCY** — only on a `WARN IV poller` heartbeat (VPS collector down), run by hand until the VPS is back: `.\.venv\Scripts\python.exe scripts\01c_iv_poller.py`. **`04b` must NEVER run at home:** two `--execute` would manage the same testnet position (double orders). Emergency stop (command-line match → catches stub+worker, leaves other `python.exe` untouched):

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match '01c_iv_poller|01d_orderbook_recorder' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

**Health:** count **LOGICAL** processes, not OS ones — each `.venv\python.exe` is stub+worker = 2 OS processes (compare `ParentProcessId`). Data growth comes **from the pull**, not from local processes: `data/iv/atm_30h.parquet` and `data/iv/mfiv_30h.parquet` ~288 rows/day (5-min VPS cadence), `results/vol_paper/forecasts.parquet` ~24 rows/day. Live logs in `logs/quantsys_*.log` (newest by mtime).

#### 5.3bis 24/7 collectors on the VPS

Four **systemd** services on an always-on EU VPS (deployed 2026-07-14): `quantsys-iv` (`01c`), `quantsys-ob` (`01d`), `quantsys-trades` (`01e`), `quantsys-volpaper` (`04b`). **Host/IP are private: ONLY in `config/secrets.yaml` → `vps:` block**, never in repo or docs. Full deploy: `deploy/vps/README.md` (Binance 451 geo-test → deploy key → one-shot `setup_vps.sh` → verify). Home-side commands, from the project root:

```powershell
.\avvio_sessione.ps1              # tutto-in-uno di sessione: pull+merge + check B7 + monitoraggio vol (§5.3)
.\scripts\vps\pull_vps_data.ps1   # solo sync: host da secrets.yaml → scp → data/vps_staging/ + merge + heartbeat
.\scripts\vps\check_vps.ps1       # health-check on-demand via ssh: servizi / freschezza / disco / geo (-UpdateRepo = git pull remoto)
```

The merge (`scripts/vps/merge_vps_data.py`) dedups double ticks and warns when the latest VPS tick is stale (default 3h → remote collector down). The canonical data copy stays home (`data/iv/`, `data/orderbook/`, `data/deribit_trades/`); the VPS guarantees 24/7 continuity of the IV asset. ⚠ The home PC starts no local collector (§5.3).

#### Deribit IV poller

`python scripts/01c_iv_poller.py` — loop, default 10-min cadence (on the VPS it runs at **5 min with `--greeks`**). 2 public Deribit requests per tick, **no account**. Flags: `--minutes N` (cadence), `--once` (smoke), `--backfill-dvol` (hourly DVOL history 2021→today), `--greeks` (+3 calls/tick for the venue greeks of the ATM ~30h-tenor straddle → `atm_greeks.parquet`, selection identical to `04b`'s `pick_straddle`). Atomic append-only output under `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (raw snapshot, ~950 instruments/tick), `atm_30h.parquet` (ATM IV of the 4 nearest expiries + total-variance-interpolated IV at a constant 30h tenor = the vol forecast horizon), `dvol.parquet` (30d control). Purpose: short-tenor IV history — not free elsewhere — for the **NN-RV vs implied IV** gate.

#### Binance L2 order-book recorder (B1)

`python scripts/01d_orderbook_recorder.py` — loop, default 5s cadence. Flags: `--seconds N`, `--once`, `--symbol` (default `BTCUSDT`), `--levels` (REST depth, default 1000). 1 public `/api/v3/depth` request per tick (no auth, weight 50/call → at 5s = 600/min ≪ 1200). Track **B1**: FORWARD collection of microstructure as a NEW source for 1m directional edge (the 104 OHLCV features are saturated). Atomic append-only output `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/day, dedup on `timestamp`): mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, cumulative depth 5/10/25/50 bps, total qty, **best-level OFI** (Cont-Kukanov-Stoikov) + **top-25 raw levels/side** as list-columns. ⚠ `ofi_best` is NaN on the 1st tick of each process and in `--once`.

#### Vol-paper forward test

⚠ **Runs as a VPS service, NOT at home** (§5.3): two `--execute` would manage the same testnet position. Production invocation (unit `deploy/vps/quantsys-volpaper.service`):

```bash
python scripts/04b_vol_paper.py --execute
```

⚠ **No second `04b` instance on the production directory — not even simulated, not even `--once`.** The state paths (`results/vol_paper/`) are fixed and relative to the launch directory: a second instance on the VPS rewrites `forecasts.parquet` and `position.json` under the live service, and with `--adaptive` a blocked attempt leaves `adaptive_entry_journal.json`, whose mere presence halts the v1 service **too** until it is removed by hand. The block fires without `--execute` as well (a single contract for both modes, 2026-09-15): in simulation a REST error on the mark is `ambiguous` just as with real orders. An isolated test on the VPS first requires an explicit output-directory flag, which does not exist today.

⛔ **`--hedge` has FAILED — v2 gate closed on 2026-08-11 (FAIL 2/3); the flags were removed from the unit on 2026-08-13 and the invocation above is the one actually in force.** Per-trade variance fell 55.3% (① passed with margin) but the drag was −0.647·SE against a −0.25·SE budget, made of **76% perp fees and 0.9% funding**. Detail and decomposition: `THEORY.md` §12.2. ⚠ **An ORDERING constraint when disabling it, and it cuts both ways:** `maybe_hedge` and `reconcile_hedge_state` run **only** inside the `--hedge` branch, so dropping the flag while `results/vol_paper/hedge_state.json` exists leaves the perp leg **naked, unmanaged and never flattened** on the testnet — this applies to anyone re-enabling the flag and later turning it off. ⚠ **"Disable after a settlement" is not enough: the clean window is zero-wide.** The settlement and the opening of the next structure happen in the **same tick**, so `position.json` is never empty and a fresh hedge can start immediately. Correct procedure, in **two steps**: ① `--hedge-band 999`, which disables opening and rebalancing but keeps the flatten alive (that branch runs **before** the band check); ② after the next flatten, remove the three flags. In the ledger the flatten may carry `reason` ∈ `{settled, expired, structure_changed}` — **all three valid**, they are the three arms of the same guard, and with the same-tick reopening you get `structure_changed`. The options leg is identical with or without the hedge (bit-identical v1 path), so disabling does not perturb the open forward samples.

Hourly loop at hh:00+90s. Flags: `--once` (smoke), `--execute` (REAL testnet orders; default = SIMULATED mark-price fills), `--arch` (model dir, default `itransformer`), delta-hedge `--hedge` + `--hedge-band/-conv/-fee/-band-mode/-ww-lambda`, close policy `--pin-close-hours/-band`, sizing `--size-mode {contracts,vega}` + `--size-vega-target/-max-contracts`, adaptive rule `--adaptive` + `--adaptive-dvol-threshold/-k/-fill-timeout/-tenor-hours` (INERT without the flag; threshold and k required), macro normalizer `--macro-norm <pin>` (**inert**: without the flag the instrument is refitted whole-df at every bootstrap, legacy behavior). Logic: 30h NN-RV forecast (PASS 1h-vol model, full `μ·IQR+center` inversion, parity-blessed feature path) vs implied variance from the IV poller (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; `|edge| > 0.25` → ~30h daily ATM straddle LONG/SHORT, max 1 position, hold to expiry (cash settlement). Requires a running IV poller and keys in `config/secrets.yaml` `deribit_testnet:` block (the URL **must** be `test.deribit.com` — anti-mainnet assert). Output in `results/vol_paper/`: `forecasts.parquet` (written even when flat, the baselines need it), `trades.jsonl`, `position.json`. ⚠ Do NOT run GPU training in parallel (5 CUDA-resident models).

**Adaptive executor (`--adaptive`) — durability, correctness fixes and record identity (2026-09-13/15, local, not deployed).** Opening the structure (daily straddle or iron butterfly) goes through a durable on-disk journal (`adaptive_entry_journal.json`): written **before** any order, its mere presence at restart blocks `tick()` and `main()` until manual review — an interrupted attempt never resumes on its own. Two fixes: **(A) verified fill fee/timing** — `_verify_trade_history` checks the COVERAGE of the real trades (unique ids, order/instrument/side identity, quantities summing to the known `filled_amount`, finite timestamps) before treating fee or `fill_span_s` as complete; `_fill_timing` uses the GLOBAL first and last fill across all entry legs (it previously used only the last fill per leg, undercounting the span with multi-fill legs). Malformed/incomplete coverage → fee and timing become **unknown** (never zero, never the local receipt time in place of the fill). **(B) diagnostic tail on a surviving journal** — if a tick's adaptive entry leaves the journal open (`blocked_operator_review`), `log_exec_diag` and that same tick's hedge leg are now skipped: previously a "flat" snapshot was still logged while the real exposure was unknown. **(C) record identity (09-15)** — trade identity (`order_id`/instrument/side) is **required**, no longer optional: a trade missing it no longer verifies coverage, and a non-hashable `trade_id` degrades to unknown instead of raising inside the submission, where the exception would have interrupted journaling with orders already sent. Every leg also persists its **concrete instrument and side** (`instrument`, `side`) in the execution and recovery record: these used to live only in the journal, which is cleared on verified flat, leaving the record with the leg's logical name alone. No new "never naked" guarantee: manual review is still required. Regression: `tests/test_adaptive_structure.py`, 40/40 (10 pre-existing + 8 on A/B + 12 on identity, journal durability, lost responses, partial fills, timeouts, compensation halting and restart blocking + 10 from the independent diff review: order verifier with required identity and partial `filled` treated as ambiguous, nonterminal branch inside the executor, malformed `trades`/`order` degraded to unknown).

**Gate baselines** — `python scripts/04c_vol_paper_baselines.py` (read-only, GPU-free; `--no-fetch` = delivery cache only, `--min-trades N` = evaluability threshold, default 30). Checks pre-registered gate (2): the NN P&L must beat **both** the always-long-vol and always-short-vol baselines over the **same** expiry calendar (isolates timing from the average variance risk premium). Method: replay of the `04b` loop over `forecasts.parquet`, premium reconstructed from chain snapshots, delivery price from the public Deribit endpoint (`delivery_cache.json`). Gates (1) mean P&L > 0 and (3) hit-rate > 0.5 are read from the REAL trades in `trades.jsonl`. Output `results/vol_paper/baseline_report.json` (+ "not evaluable" warning while n < `--min-trades`).

**FT1 judge** — `python scripts/vol/ft1_execution_judge.py [--count-only]` (read-only, GPU-free). Implements the FT1 pre-registration with amendment 1 on the `--adaptive` ledger; `NOT STARTED` while `adaptive.jsonl` does not exist. The pull copies `adaptive.jsonl` (key-merged) and `adaptive_entry_journal.json` (presence-mirrored, like `position.json`). Output `results/vols/ft1_execution.json` (never written by `--count-only`).

### 5.4 Dashboard — Deribit Options Risk Terminal

```bash
python scripts/06_dashboard.py     # avvio diretto · direct launch
python run_all.py --only-dashboard # idem (no ML, no feed live · no ML, no live feed)
```

`scripts/06_dashboard.py` = the **crypto options terminal** (single-file HTTP server + Plotly.js SPA), GPU-free and **decoupled from the ML pipeline**: it reads **Deribit public** data (REST, no-auth, no key). Four tabs — *Volatility Surface* (3D IV surface, smile, ATM term structure), *Option Chain* (call/put chain with forward Black-Scholes Greeks), *Risk & Greeks* (OI by strike, max-pain, OI-weighted aggregate Greeks, PCR, DVOL), *Trades* (`04b` forward test: history + open position and payoff profile, `/api/trades` endpoint reading `results/vol_paper/trades.jsonl` + `position.json` + `hedge_ledger.jsonl`. It reads every record shape 04b writes — v1 straddle, iron butterfly with net premium and 4-leg payoff, pin-close exit, `incomplete` structured attempts shown without a position or PnL — and flags a position whose expiry has passed without a settlement record on disk as *expired, awaiting settlement* instead of *open*. PnL = option legs + perp hedge leg, the latter reconstructed with `perp_leg` from `scripts/vol/hedged_vs_unhedged_judge.py` (same function, funding from the local cache `data/deribit_funding_perp.parquet`, no network); test `tests/test_dashboard_trades.py`). Auto-refresh ~12s. Config: `config/default.yaml → dashboard` — `host`/`port` (default `127.0.0.1:8050`), `options_currency` (BTC|ETH), optional `auth_token` (constant-time, `X-Auth-Token` header or `?token=`), `enable_gzip`.

⚠ **`SO_REUSEADDR` trap:** a stale dashboard process can keep `:8050` and serve **stale** HTML. Before a fresh smoke **kill the previous process** (`.venv` stub+worker = 1 logical process): `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '06_dashboard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`. ⚠ **The final truth is the browser with a HARD RELOAD (Ctrl+Shift+R)**: an already-open page runs the old JS. Server-side smoke: the served HTML must contain `plot(` and `/api/risk` must return HTTP 200 with the real chain. (Definitive rendering fix 2026-06-24: `plot-oi`/`plot-payoff` X axis switched to `type:'category'` for immunity to Plotly's SVG re-render corruption; detail in `STATUS.md`.)

### 5.5 Stopping everything

`Ctrl+C` in the `run_all.py` terminal (or the dashboard server): stops the pipeline and, in a full run, the WebSocket live feed too. ⚠ If the dashboard was detached (`Start-Process`), `Ctrl+C` is not enough: use the targeted `Stop-Process` (§5.4). Background collectors no longer run at home (VPS, §5.3bis): any emergency local instance is stopped via the *Stop* block in §5.3.

---

## Appendix — File layout

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
│   └── vol_paper/               # forecasts.parquet, trades.jsonl, position.json, baseline_report.json, exec_diag.jsonl (A6: bid/ask+greeks diagnostici / diagnostic), hedge_state.json + hedge_ledger.jsonl (v2, SOLO con --hedge / --hedge only), adaptive.jsonl + adaptive_entry_journal.json (SOLO con --adaptive / --adaptive only) · record a N gambe: corpo = prime 2, campi `_all` solo oltre 2 / N-leg records: body = first 2, `_all` fields only beyond 2
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

**The canonical script map is `scripts/README.md`** (full script→phase→line table, with each script's flags): this guide does not duplicate it. Scripts in subfolders use `Path(__file__).resolve().parents[2]` and must be launched from the **project root**.
