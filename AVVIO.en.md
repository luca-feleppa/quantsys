# QUANTSYS — Quick start guide

Algorithmic trading system for BTC/USDT 1m with a heterogeneous ensemble (iTransformer + N-HiTS + TCN+Mamba) and multi-teacher Knowledge Distillation.

> An Italian version of this document is in [AVVIO.md](AVVIO.md).

## System status (updated 2026-06-04 — T=120 rollback)

After the window-size experiments (T=180 and T=240 both **regressive**) the system was rolled back to **T=120** (empirical sweet spot on ~525k 1m candles; longer windows overfit temporal noise). Current dataset 549k candles → test set **10,104** windows. 5-seed retrain of the 3 archs on this configuration.

**Current raw-backtest status (test set, single-arch):**

| Config | Sharpe | Total Return | Note |
|---|---|---|---|
| iTransformer (best arch) | ≈ −14.6 (1-seed) | −1.77% | the least-bad |
| Heterogeneous ensemble (3 archs) | ≈ −19.6 | −5.5% | does not beat the single (cross-arch error ≈0.995) |

**The raw backtest is currently negative** — no config clears the paper-trading thresholds. The 2026-05-23 metrics (Sharpe +18.71) referred to a prior model/feature-space, now superseded.

**Real edge identified (2026-06-04): it is regime-conditional.** In regime **R0 Quiet** the model has Spearman **+0.13÷0.19**, stable across **all** OOS sub-periods (val+test). It is a *rank* edge: the |μ|-threshold entry misses it (Quiet = low vol → small μ). A **regime-specific rank-based entry** (experimental, env-gated in `03_backtest.py`: `QUANTSYS_QUIET_RANK_Q` + `QUANTSYS_QUIET_MIN_SIGMA`) partially recovered it on the test set (best −0.74% return, MDD 1.5%). ⚠ **Validation on val FAILED (2026-06-05):** on the held-out split (`QUANTSYS_BACKTEST_SPLIT=val`) the same config returns **−0.22%**, PF 0.84, only 13 trades (val is 71% Stress, 14% Quiet → underpowered) → the edge **does not hold out-of-sample**, not promoted into `SignalGenerator`; it stays an inert env-flag. Production = clean heterogeneous ensemble / iTrans standalone. The Trending-regime inversion is **not robust** (OOS validation breaks it) → not deployed. ⚠ **In-sample metrics (val_nll, walkforward Spearman) anti-correlate with the backtest** (structural distribution shift): do not use them to optimize. Full detail in `docs/MODEL_IMPROVEMENTS.en.md` and `STATUS.md`.

**Harvesting the ordinal edge — 2 experimental levers (2026-06-05): validated on val and FAILED.** Two env flags were added to `03_backtest.py` (inert by default, reversible): **Fix ① decision cadence** `QUANTSYS_DECISION_CADENCE=h` (entries only every ≥`forecast_horizon` candles; exits every candle) and **Fix ② continuous rank-based exposure** `QUANTSYS_RANK_EXPOSURE=1` (direction+`conviction` from the causal μ-percentile, regime-gated on R0 Quiet, no-trade band `QUANTSYS_RANK_BAND` = hysteresis). ⚠ **Val outcome (`QUANTSYS_BACKTEST_SPLIT=val`):** clean baseline **+4.03%/PF 1.88/WR 61%** → with Fix ①② **−2.24%/PF 0.22/WR 25.7%**. The rank signal as a directional entry is **anti-predictive OOS**; continuous exposure flips (27/35 SIGNAL exits) → PnL is dominated by the SL/TP path, not the horizon return the Spearman lives on. **Not promoted**, flags stay inert. (The +4% val baseline is the favorable side of the val→test shift; the test baseline stays negative.)

All 3 architectures load the same heterogeneous ensemble via `EnsembleModel.load_heterogeneous()`, so they produce the same backtest.

**Live engine — current status**: paper-only mode (no real orders), but **BLOCKER #1 RESOLVED (2026-06-05)**: the live path now uses `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canonical, same scaler) → `_deterministic_predict` → `SignalGenerator`, with **bit-perfect feature AND signal parity** (`tests/test_live_training_parity.py` 5/5; `99_replay_live_vs_training.py` Δ=0). Paper signals now reflect the backtest — see `TEORIA.en.md` §11. Operational remainder: real WS smoke test + paper-trading. ⚠ The backtest is negative OOS: paper-trading is for accumulating real trades.

**For any new model entry point**: always call `PipelineState.denormalize_predictions(mu, sigma)` before passing predictions to `SignalGenerator`. The model predicts in z-score space (RobustScaler); the trading layer operates in raw space. See `TEORIA.en.md` §5 for the full invariant.

---

## Quick commands

```bash
python run_all.py                                    # interactive menu
python run_all.py --arch itransformer                # train single arch
python run_all.py --arch nhits
python run_all.py --arch tcnmamba
python run_all.py --arch lstm                        # backward compat
python run_all.py --distill                          # multi-teacher Knowledge Distillation
python run_all.py --distill --teacher itransformer   # force a specific teacher
python run_all.py --only-dashboard                   # dashboard + live only
python scripts/07_verify_teacher.py                  # compare architectures
python scripts/99_replay_live_vs_training.py         # BLOCKER #1 diagnostic
set QUANTSYS_ARCH=lstm
python scripts/02c_optuna_search.py --n-trials 50    # Optuna (LSTM only)
```

---

## First run

### 1. Prerequisites
```bash
python scripts/00_check_setup.py
```
Checks Python deps, CUDA, Binance, FRED. Fix any errors before continuing.

### 2. Full pipeline
```bash
python run_all.py
```
Without flags shows a checkbox menu (↑↓ navigate, SPACE select, A toggle all, ENTER confirm). Opens dashboard at `http://localhost:8050`.

Direct mode: `python run_all.py --arch nhits --force-download`.

---

## Training a single architecture

Each architecture has its own config in `config/arch/{arch}.yaml` and isolated outputs in `models/{arch}/` and `results/{arch}/`. No cross-run interference.

| Arch | Command | Time (RTX 2070 Super) | Notes |
|---|---|---|---|
| iTransformer | `python run_all.py --arch itransformer --skip-update --skip-macro` | ~25 min | Attention over features, baseline (ICIR 0.795) |
| N-HiTS | `python run_all.py --arch nhits --skip-update --skip-macro` | ~10–15 min | Hierarchical pure-MLP, replaces LSTM since 2026-05-14 |
| TCN+Mamba | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | ~20 min | Dilated conv + SSM, great for local patterns |
| LSTM | `python run_all.py --arch lstm --skip-update --skip-macro` | ~30 min | Legacy backward compat (under-performing) |

`--skip-update --skip-macro`: use on-disk data without redownload. First run: omit them.

### Inspecting results
```bash
python scripts/07_verify_teacher.py
```
Comparison table: param count, forward time, Sharpe, WR, trade count, max DD, total return for every arch with a saved `best_model.pt`. Alternatively:
- `models/{arch}/config.json` — `best_val_loss`, scaler info, n_params
- `models/{arch}/history.json` — per-epoch loss curve
- `results/{arch}/dashboard_results.json` — backtest metrics

Or from the dashboard: `python run_all.py --only-dashboard` then arch dropdown (`/api/archs` auto-detects archs with `dashboard_results.json`).

---

## Distillation with multiple models

### Default composition

Heterogeneous ensemble: **iTransformer + N-HiTS + TCN+Mamba**. LSTM was removed on 2026-05-14 (val_NLL 5.28 vs iTransformer 0.18 → structural underfitting). LSTM code is intact and reloadable for rollback.

### Pipeline

```bash
python run_all.py --distill --skip-update --skip-macro
```

**Phase 2a — Candidate training**: each arch in `distillation.archs` is trained normally with `n_ensemble=1`. If `models/{arch}/best_model.pt` exists, it is skipped. To force retrain: `--force-download`.

**Phase 2b — Multi-Teacher Scoring**: every candidate is evaluated at its best epoch with normalized scoring (40% val_loss + 35% Spearman + 25% directional accuracy). Softmax weights with temperature=2 are computed for all of them. The top score becomes the *primary teacher*; the others stay in the pool as weighted teachers.

**Phase 2c — Multi-Teacher Distillation**: each student receives soft labels combined as a weighted mean from scoring. Mixed loss `(1−α)·NLL_real + α·distill_loss` with α=0.3. Distillation loss is scale-normalized for μ/σ/ν. Soft labels are integrated into the TensorDataset (shuffle-safe). Epochs are cut to 60%. Students already distilled are skipped automatically.

**Heterogeneous ensemble (inference)**: all models predict together, the output is combined using the law of total variance:
- `mu_ens = Σ w_i · mu_i`
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)`

Default weights in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`): iTransformer/N-HiTS/TCNMamba/TFT = 1.0, LSTM = 0.5.

### Changing the composition

**One spot only**: `config/default.yaml` → `distillation.archs`:
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
Examples: `["itransformer", "lstm", "tcnmamba"]` legacy rollback; `["itransformer", "nhits", "tcnmamba", "lstm"]` 4-model ensemble; `["itransformer", "tcnmamba"]` just 2.

After editing, `python run_all.py --distill` trains the missing models, scores, distills students; backtest/live automatically pick up the new composition.

### Forcing a specific teacher
```bash
python run_all.py --distill --teacher itransformer
```
Skips automatic scoring. The others remain in the weighted pool.

### Verifying distillation

In `models/{arch}/config.json`:
- `distilled: true`
- `teacher_arch: "multi-teacher"`

An already-distilled arch is skipped in Phase 2c. To force re-distillation: delete `best_model.pt` or use `--force-download`.

---

## Subsequent runs

```bash
python scripts/06_dashboard.py            # dashboard only
python run_all.py --only-dashboard        # same
python run_all.py --skip-train --skip-walkfwd   # refresh data, same models
python run_all.py                          # menu
python run_all.py --distill                # full + distillation
```

---

## Useful flags

| Flag | Effect |
|------|--------|
| `--skip-update` | Use existing dataset, no download |
| `--skip-macro` | Skip FRED/yFinance download |
| `--skip-train` | Use existing model, no retrain |
| `--skip-walkfwd` | Skip walk-forward validation |
| `--skip-backtest` | Skip backtest |
| `--skip-live` | No live WebSocket feed |
| `--skip-analyze` | Skip `05_analyze_signals.py` |
| `--only-dashboard` | Dashboard + live only, no ML |
| `--no-browser` | Do not auto-open browser |
| `--force-download` | Redownload + force retrain |
| `--max-model-age-days N` | Retrain if model older than N days |
| `--distill` | Multi-teacher distillation pipeline |
| `--teacher ARCH` | Force primary teacher |

---

## Hardware

### CPU
`config/default.yaml`:
```yaml
hardware:
  cpu_fraction: 0.5   # 0.3=30%, 0.5=50%, 0.8=80%
```
Default 0.5 (4 threads on 8 cores). Read by every script at startup.

### GPU compute
```powershell
nvidia-smi -pl 125    # limit (RTX 2070 Super min=125 max=215W)
nvidia-smi -pl 215    # restore
```

### Reference setup (RTX 2070 Super 8GB)

| Component | Value |
|---|---|
| CUDA, AMP fp16 training | yes (via `setup_device`) |
| AMP inference | **off** hardcoded in `ensemble.py:170` (avoids NaN from spectral_norm + Mamba scan) |
| Backtest inference batch | 256 (`scripts/03_backtest.py`) |
| Training batch | 64 (default `config/arch/<arch>.yaml`) |

### CPU only
The code falls back automatically via `setup_device` (`quantsys/utils/__init__.py`). In `quantsys/model/__init__.py:67`, `autocast(device_type="cuda")` is a silent no-op on CPU. Times:
- Training: 20–50× slower (tcnmamba ~3h GPU → 2–3 days CPU). Not recommended.
- Backtest: ~5s GPU → 30–60s CPU. Tolerable.
- Live: ~50–100ms vs ~20ms GPU. Fully usable (Binance WS latency dominates).

### Apple Silicon / AMD / Intel Arc
Untested. Code uses `torch.cuda.*`. MPS support would require changes to `setup_device` and likely custom kernels for Mamba/SSM.

### Low VRAM (4GB)
`config/arch/<arch>.yaml`:
```yaml
batch_size: 32
gradient_accumulation_steps: 2   # keeps effective batch=64
```
Inference batch in `scripts/03_backtest.py` from 256 → 128.

### High VRAM (≥16GB)
```yaml
batch_size: 128
```
Inference batch up to 1024 (marginal gain, GPU already saturated).

---

## Architectures

| Arch | Class | File | Notes |
|---|---|---|---|
| `itransformer` | `QuantiTransformer` | `quantsys/model/__init__.py:1025` | Attention over features, baseline |
| `nhits` | `QuantNHiTS` | `quantsys/model/nhits.py:110` | Hierarchical pure-MLP |
| `tcnmamba` | `QuantTCNMamba` | `quantsys/model/tcn_mamba.py:341` | Dilated TCN + SSM hybrid |
| `lstm` | `QuantLSTM` | `quantsys/model/__init__.py:309` | Legacy |

### Adding a new architecture
1. Class in `quantsys/model/` with `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher in `quantsys/model/__init__.py:load_model`
3. Branch in `scripts/02_train.py` (`architecture == "X"`)
4. `config/arch/X.yaml`
5. `choices` in `run_all.py` (parser `--arch` and `--teacher`)
6. Whitelists in `06_dashboard.py`, `05_analyze_signals.py`
7. (Optional) `distillation.archs` in `config/default.yaml`

---

## Optuna

```bash
set QUANTSYS_ARCH=lstm
python scripts/02c_optuna_search.py --n-trials 50 --study-name quantsys
```
**Limits**: hardcoded on `QuantLSTM`. The `best_params.json` saved in `models/lstm/` is NOT applied to the next training automatically — copy it manually into `config/arch/lstm.yaml`.

Study persists on SQLite (`models/lstm/optuna_quantsys.db`), resumable any time.

---

## Homogeneous ensemble (5× same arch, legacy)

`config/default.yaml`:
```yaml
training:
  n_ensemble: 5   # current default = 5 (auto-overridden to 1 when --distill)
```
Output: `models/{arch}/best_model_0..4.pt`. Backtest/live load them via `EnsembleModel.load`. Independent from distillation (modes are not mutually exclusive).

---

## File layout

```
quantsys_project/
├── config/
│   ├── default.yaml             # base config (distillation.archs lives here)
│   ├── secrets.yaml             # FRED key, gitignored
│   └── arch/
│       ├── itransformer.yaml, nhits.yaml, tcnmamba.yaml, lstm.yaml
├── data/
│   ├── raw_candles.parquet      # historical OHLCV
│   ├── features.parquet         # normalized features
│   ├── lstm_dataset.npz         # training X/y windows
│   ├── funding_rate.parquet     # funding futures (8h)
│   └── macro_*.parquet          # FRED/yFinance
├── models/
│   ├── teacher_analysis.json    # 07_verify_teacher.py output
│   └── {arch}/
│       ├── best_model.pt        # checkpoint
│       ├── config.json          # hyperparams + distilled/teacher_arch flags
│       ├── history.json         # loss curve
│       └── pipeline_state.pkl   # scalers + feature config
├── results/{arch}/
│   ├── dashboard_results.json
│   └── live_signals.jsonl
├── tests/                       # pytest (test_recent_fixes.py: regression on critical fixes)
├── scripts/                     # 00–07 numbered + 99_replay_live_vs_training.py
└── logs/quantsys_YYYYMMDD_HHMMSS.log
```

---

## Dashboard — "Update" button

1. Click **Update** top right
2. Select the steps to run
3. Click **Start** → progress bar
4. When done the dashboard refreshes automatically
5. **Cancel** to stop a job

Switching arch: top dropdown (detects archs with `dashboard_results.json`).

---

## Stopping everything

`Ctrl+C` in the `run_all.py` terminal. Terminates dashboard + live feed.
