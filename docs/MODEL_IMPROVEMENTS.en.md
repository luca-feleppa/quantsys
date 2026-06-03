# QUANTSYS — Outstanding model improvements

> Italian version in [MODEL_IMPROVEMENTS.md](MODEL_IMPROVEMENTS.md).

Everything already done lives in `CHANGELOG.md` and the notes under `~/.claude/projects/E--quantsys-project/memory/`. This file lists only what remains to implement, in recommended order.

---

## 🟢 RESOLVED 2026-06-03 — Markov-Switching on BTC realized vol (Variant 3) implemented

> 🟢 Resolved 2026-06-03: see Italian version for details. `RegimeMarkovBTC` implemented (Hamilton 1989 Markov-Switching on BTC hourly realized volatility + expanding-window PCA), three data-driven regimes (R0 Quiet ~42%, R1 Trending ~18%, R2 Stress ~40%), stratified val 46% / 12% / 41% (vs previous 100% r0 collapse), val_nll spread 0.19-0.30 (well above the 0.05 "informative" threshold). Output file `data/regime_probs.parquet` and schema unchanged for backward compat. Docs (TEORIA.md/.en.md, README.md/.it.md) and session memory `session_2026_06_03_markov_btc.md` updated. Rollback decision no longer applicable (validation passed).

---

## 🔴 BLOCKER #1 — Live↔training feature alignment (Stage 2-5)

**Status:** Stage 1 done (code), Stages 2-5 pending. Paper-trading **cannot** start until the mismatch is fixed.

**Problem (verified 2026-06-02 with `scripts/99_replay_live_vs_training.py`):** the backtest uses the full `FeatureBuilder` (**119 features**); the live engine (`LiveFeatureBuffer._compute_features` in `scripts/04_live_signals.py`) builds **only 39** by hand in a different order, with per-window median/IQR normalization (not the `pipeline_state`'s `RobustScaler`), and `_predict` does blind positional pad/truncate. Three overlapping mismatches (count + order + scale) → live inputs effectively uncorrelated from training. **Current paper-trading signals do NOT reflect the backtest.**

Root cause: the reduced `LiveFeatureBuffer` exists because the full `FeatureBuilder` requires long history (ATH/ATL 365d, momentum 90d, frac-diff, vp_*_long) not available in the live rolling buffer (260 candles).

### Decision (2026-05-28): Option C-funding (~104 features)

**Rationale from permutation importance** (heterogeneous ensemble, 2500 val windows, permutation per group/feature): the 23 "live-hostile" features (lookback > buffer: 30/90/365d, frac_diff_*, vp_*_long, vp_poc_convergence, funding) have **ROI ≤ 0** for the h=30 model: bulk-permuting them *slightly improves* metrics (DA 0.529→0.532, Spearman 0.069→0.076). Sole exception: the **funding** features (`momentum_x_funding` +0.0042, `funding_rate_dev` +0.0028).

**C-funding set** = single source of truth shared by training/live:
- Drops 15 live-incompatible features (90d, 365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`).
- Keeps 30d + funding (positive ROI, computable live via a 30d ring buffer ~170 KB and a Binance funding poll).
- Target: ~104 total features (vs 119 today).

> The "full hybrid" scheme that kept all 30/90/365d features in live was documented as an alternative but **not recommended by the data** (negative ROI on the long tier) — definitively discarded.

### Stage 1 — code ✅ DONE

`LIVE_DROP_FEATURES` (15 features) in `quantsys/features/__init__.py`, filtered in `scripts/01_download_data.py` (`feat_cols = [c for c in feat_cols if c not in LIVE_DROP_FEATURES]`).

### Stage 2 — Dataset regeneration + nhits smoke ⏳ TO DO

```bash
python run_all.py --arch nhits --skip-backtest --skip-walkfwd --skip-live --no-browser
```

Pipeline: update data → apply C-funding filter → macro → train only nhits from scratch (~10-15 min).

**Go/no-go gate in the logs:**
- `Set C-funding: scartate 15 feature live-incompatibili: [...]`
- `X=(..., 120, 104)` during `01_update` / `create_windows`
- `val_nll` must land around **~0.28**. If ~8 → loss/config anomaly, stop.

### Stage 3 — Full distill retrain (hours of GPU) ⏳ TO DO

```bash
python run_all.py --distill --skip-update --skip-macro --skip-live --no-browser
```

Phase 2a retrains itransformer + tcnmamba from scratch at 104 feat (nhits already done in Stage 2). Phase 2b picks the teacher. Phase 2c distills students multi-teacher. `--skip-live` prevents the old live engine (still 39-feat) from starting mid-pipeline.

**Watch-points:**
- The walkforward (02b) may still show the nhits anomaly (val~8): it's the P2.1 script-level issue, **non-blocking** for the distilled models (validation separate, `fatal=False`).
- **Do not** run backtest/live between Stage 2 and Stage 3: the per-arch `pipeline_state` files for itransformer/tcnmamba are absent until retrain (avoids fallback to stale scalers).

### Stage 4 — Live engine rewrite ⏳ TO DO

Reuse `FeatureBuilder` + 30d rolling buffer + funding poll + `pipeline_state` scaler. Remove positional pad/truncate.

**Files to touch:**
- `quantsys/features/__init__.py` → canonical feature name list (single source of truth).
- `scripts/04_live_signals.py` → split `LiveFeatureBuffer` into `HotFeatures` + `FeatureAssembler` (assemble by NAME according to `feature_names`, no pad/truncate; hard-fail on missing name).
- `quantsys/utils` `PipelineState` → expose scaler + feature order to the live engine.
- Funding poller reusing `quantsys/data.fetch_funding_rate`.

**Startup seeding:** load the last 30 days of 1m klines (Binance pagination, one-shot, with local cache) or reuse `data/raw_candles.parquet`.

### Stage 5 — Parity test + replay backtest (go/no-go gate) ⏳ TO DO

1. **Parity test:** live vector vs `FeatureBuilder` on the same historical window, `max|Δ| < tol` per feature.
2. **Replay backtest:** historical candles through the live pipeline → signals/PnL must match the offline backtest.
3. Only after both gates green: **start paper-trading** (2-4 weeks with live Sharpe > 0.5 before considering mainnet).

---

## 🔵 Binance Futures Testnet — Phases 2-5

**Status:** Phase 1 ✅ done (`.env` + `scripts/00_test_binance_testnet.py`). Phases 2-5 pending. Remaining effort: 8-13 hours.

Goal: the live engine sends real orders on the Futures Testnet (`testnet.binancefuture.com`) in parallel with the simulated portfolio, with periodic reconciliation. Validates real execution latency, testnet slippage, exchange-side SL/TP behaviour, operational bugs the backtest doesn't cover.

> **Prerequisite:** BLOCKER #1 (Stage 2-5) resolved first, otherwise the testnet receives signals from a model with uncorrelated inputs.

### Phase 2 — Execution layer architecture (2-4h)

**New package** `quantsys/execution/`:

```
quantsys/execution/
├── __init__.py          # factory create_adapter(mode, ...)
├── base.py              # ABC ExecutionAdapter
├── paper.py             # in-memory simulated (refactor of current RiskManager behaviour)
└── binance_futures_testnet.py  # REST via python-binance.Client(testnet=True)
```

**ABC interface** (`base.py`):

```python
class ExecutionAdapter(ABC):
    @abstractmethod
    def place_market_order(self, side: Side, qty: float) -> str: ...
    @abstractmethod
    def place_stop_market(self, side: Side, qty: float, stop_price: float) -> str: ...
    @abstractmethod
    def place_take_profit_market(self, side: Side, qty: float, tp_price: float) -> str: ...
    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...
    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None: ...
    @abstractmethod
    def get_position(self, symbol: str) -> Position | None: ...
    @abstractmethod
    def get_balance(self, asset: str = "USDT") -> float: ...
    @abstractmethod
    def get_open_orders(self, symbol: str) -> list[dict]: ...
    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> None: ...
```

**New config** in `config/default.yaml`:
```yaml
live:
  execution_mode: paper        # paper | testnet_futures
  testnet_futures:
    symbol: BTCUSDT
    margin_type: ISOLATED      # ISOLATED | CROSSED
    max_leverage: 3            # conviction-modulated; 1 = no leverage
    leverage_conviction_alpha: 1.0
    # api_key/secret from env BINANCE_TESTNET_API_KEY / _SECRET (.env)
```

**Conviction-based dynamic leverage** (decided 2026-05-24):
```python
def _conviction_leverage(conviction: float, max_lev: int, alpha: float = 1.0) -> int:
    """conviction=0 → 1x, conviction=0.5 → ~max_lev/2, conviction=1 → max_lev."""
    lev = 1 + (max_lev - 1) * (conviction ** alpha)
    return max(1, min(max_lev, round(lev)))
```
Called in `RiskManager.open_position` BEFORE `place_market_order`.

### Phase 3 — RiskManager wiring (2-3h)

Changes to `quantsys/trading/__init__.py`:
- `RiskManager.__init__` accepts `execution_adapter: ExecutionAdapter | None = None` (default = paper).
- `open_position`: after computing SL/TP+size, if `self.adapter is not None`:
  1. `set_leverage(symbol, _conviction_leverage(...))`
  2. `entry_order_id = place_market_order(side, qty)`
  3. `sl_order_id = place_stop_market(opposite_side, qty, sl_price)`
  4. `tp_order_id = place_take_profit_market(opposite_side, qty, tp_price)`
  5. Persist the 3 orderIds on the `Position`.
- `update_trailing`: if SL updated and adapter: `cancel_order(sl_order_id)` + new `place_stop_market` + update `position.sl_order_id`.
- `close_position`: if adapter: `cancel_all_orders` (closes residual SL/TP) + `place_market_order(opposite_side, qty)` (at-market close).

**Edge cases:**
- Partial fill: poll until FILLED or cancel + market on the remainder.
- Liquidation: automatic recovery (close paper, log WARNING, restart clean).
- Rate limit: Binance Futures 1200 weight/min; one `open_position` ≈ 4 REST calls → max ~300 open/min theoretical (sufficient).

### Phase 4 — Paper vs testnet reconciliation (2-3h)

New module `quantsys/execution/reconciliation.py`:

```python
class Reconciler:
    async def loop(self, interval_seconds: int = 30):
        while True:
            real_pos = self.adapter.get_position(symbol)
            real_balance = self.adapter.get_balance("USDT")
            paper_equity = self.paper.equity
            delta_equity = real_balance - paper_equity
            self._log({...})  # signals/reconciliation.jsonl
            if abs(delta_equity / paper_equity) > 0.005:  # >0.5% drift
                log.warning(f"Reconciliation drift: paper=${paper_equity} vs real=${real_balance}")
            await asyncio.sleep(interval_seconds)
```

Output: `signals/reconciliation.jsonl` (~2880 records/day). Warning only on drift > 0.5%. Integrated into `04_live_signals.py` via `asyncio.gather`.

### Phase 5 — End-to-end test (2-3h)

**Pre-flight:**
1. `python scripts/00_test_binance_testnet.py` (already ✅)
2. Set `live.execution_mode: testnet_futures` in config
3. Run with `max_leverage: 1` (no leverage) as initial safety net, monitor for: Binance WS connected, first signal, first OPEN on testnet (verify orderId + position), first trailing SL update, first CLOSE, reconciliation delta < 0.5%
4. Let it run 1-3h with `max_leverage: 1`. If 3-5 trades go OK end-to-end → raise gradually (1 → 2 → 3).

**Decision criteria after 24-48h live (all 4 must pass):**
- Reconciliation drift < 0.5% on >95% of samples
- Real slippage within 2× the backtest's (`slippage_rate: 0.0003`)
- Total latency (signal gen → filled order) < 500ms
- Zero rate limit violations

If all OK → paper-trading 2-4 weeks before considering mainnet. If any fails → fix specific bugs before retrying.

---

## 🟡 Model roadmap — Fix #3, #4, #5, #6

All gated post paper-trading (which is gated post BLOCKER #1).

| # | Fix | From | To | Effort | Expected benefit |
|---|---|---|---|---|---|
| 3 | `model.window_size` (T) | 120 | **240** | config + ~30% VRAM | DA ↑ 1-2%, vol cluster captured |
| 4 | `validation.n_folds` | 3 | **5-6** | config + +50% test time | More reliable walkforward bootstrap CI |
| 5 | Multi-timeframe (1m+5m+1h) | — | new `mtf/` pkg | 6-9 weeks elapsed | DA ↑ 2-4%, 24h context |
| 6 | `mamba-ssm` (CUDA kernel) | pure-PyTorch | official kernel | ~1h setup + retrain | TCN+Mamba speedup 3-5× |

### Fix #3 — Window size T 120 → 240

**Rationale:** vol clustering on BTC 1m has half-life ~2-6h (Engle 1986, Bollerslev 1986). With T=120 (2h) you only see half a cluster. Literature (PatchTST Nie 2023, iTransformer Liu 2024) tests lookback 96-720; plateau around 192-384.

```yaml
model:
  window_size: 240
validation:
  embargo_steps: 3000   # from 1500 (must be ≥ window_size + horizon)
```

Then: `python scripts/01_download_data.py` (rebuilds the npz) + `python run_all.py --distill --skip-update --skip-macro --no-browser`.

**Preliminary smoke test** on iTransformer only to validate VRAM:
```powershell
$env:QUANTSYS_ARCH = "itransformer"
python scripts\02_train.py --n-ensemble 1
```
If OOM on 8GB: `training.batch_size: 64 → 32` + `gradient_accumulation_steps: 2 → 4` (keep effective batch=128).

Impacts: training VRAM ~+30%, time per epoch +30-50%, usable samples −1% (more wasted candles for warmup).

### Fix #4 — Walkforward folds 3 → 5-6

**Rationale:** 3 folds give wide bootstrap CIs (Sharpe [+0.78, +74.70] on 42 trades). Finance-ML literature (López de Prado 2018, AFML ch. 7): for crypto, **5-6 folds** are the standard.

```yaml
validation:
  n_folds: 6
  embargo_steps: 3000   # if fix #3 applied, else 1500
```

Then: `python scripts/02b_walkforward_validate.py`. **No** retrain required. Time +50% (~30-45 min total).

**Checks in `results/{arch}/walkforward_metrics.json`:**
- `da_per_fold`: all > 0.51, std < 0.005
- `spearman_per_fold`: all positive
- `sharpe_per_fold` (bootstrap): CI excludes zero in **at least 4 out of 6 folds**

Wide divergence (std DA > 0.01) → model not stable across regimes → go back to more data.

### Fix #5 — Multi-timeframe (1m + 5m + 1h)

**Status:** highest-potential improvement still open. **Prerequisite:** ≥ 7-14 days of paper-trading data for a real baseline.

**Proposed architecture:**
```
1m  → 120 candles (micro-patterns, existing)
5m  →  24 candles (intraday swing, 2h context)
1h  →  24 candles (daily trend, 24h context)
```
3 separate encoders of the same family, final fusion via cross-attention or gated concat.

### Strategy: isolated parallel experiment in a new `mtf/` package

To avoid breaking production code and allow instant rollback, develop in a parallel directory that reuses as much as possible via imports.

```
mtf/                    # NEW package, isolated
├── __init__.py
├── data_builder.py     # resample 1m→5m, 1m→1h + dataset build (3 right-bound aligned streams)
├── models.py           # Quant*Mtf wrappers composing the quantsys/ models
├── train.py            # training loop with DataLoader yielding 3 X tensors
├── backtest.py
├── live_signals.py     # only if it goes to production
└── run.py
```

Parallel dataset/models/results: `data/mtf_dataset.npz`, `models/mtf_{arch}/`, `results/mtf_{arch}/`.

**Reuse** (import from `quantsys/`): loss functions (`student_t_nll`, `quantile_loss`, `direction_value_loss`), utilities (`load_config`, `setup_device`, `PipelineState`), risk manager, signal generator, macro encoder, regime detector, FeatureBuilder (called 3× on the 3 timeframes, same code).

**Create new** only where the shapes change: data_builder, models wrapper, training loop with 3 X tensors.

**Parallel-structure benefits:** instant rollback (`rm mtf/`), zero production regressions, clean A/B validation, no conflicts with ongoing single-tf paper-trading.

### Expected impacts

| Aspect | Single-tf (today) | Multi-tf | Delta |
|---|---|---|---|
| Storage `lstm_dataset.npz` | (107480, 120, 119) ~6.1 GB | + (107480, 24, 119)×2 | **+40%** (~8.5 GB) |
| iTrans training 200 epochs | ~6h | ~10-14h | +60-100% |
| Full distill pipeline | ~2-3h | **~30-50h GPU** | **10-20×** |
| TCN+Mamba VRAM batch 64 | ~4 GB | ~5-6 GB | ⚠ tight on 8GB |

⚠ **Slow iteration**: each experiment takes 6-12h. Plan accordingly.

| Metric | Single-tf | Multi-tf expected | Confidence |
|---|---|---|---|
| Directional Accuracy | 51.7-53.2% | **53-56%** | high |
| Spearman ρ | 0.034-0.062 | **0.07-0.12** | high |
| Backtest Sharpe | +18.71 | +20-40% | low (calibration) |
| Win rate | ~64% | 65-70% | medium |
| Max drawdown | 0.83% | expected similar or better | medium |

**What it specifically captures:**
1. Daily trend (1m with T=120 only sees the last 2h)
2. Funding rate cycle 8h (1h × 24 captures 3 full cycles)
3. Volatility regime shifts lasting hours (distinguishes "compression before breakout" from "ongoing range")
4. Daily seasonality history (US/EU/Asia opens)

**Risks:**
1. **Resample data leakage**: if the 5m bar at minute T:00 erroneously includes T+1..T+4 → you're predicting the future. **Critical test**: shuffle X_train and verify the model does NOT learn.
2. Live ↔ backtest warmup mismatch (live waits 24h, backtest skips 1440 candles).
3. Curse of dimensionality (3 encoders ≈ 3M params vs 107k samples = 28× unfavourable).
4. 6-12h iteration cost.

**Worth doing if:** paper-trading confirms live Sharpe > 0.5 for 2+ weeks AND you want to push ICIR 0.79 → 0.9+. **No** if you're after a quick win or the system isn't live-validated yet.

**Total estimated cost:** 6-9 weeks elapsed (1-2 weeks coding + 1 week debugging + 30-50h GPU first training + 2-3 weeks tuning + 1-2 weeks validation).

### Fix #6 — mamba-ssm package (CUDA Toolkit + official kernel)

**Expected speedup:** +3-5× on the Mamba branch (on top of the +1.4-1.6× already obtained via AMP off + chunk pre-alloc).

**Rationale:** the current implementation in `quantsys/model/tcn_mamba.py` is **pure-PyTorch** (`SimplifiedMambaBlock._parallel_scan_chunk`). The `mamba-ssm` package by Tri Dao implements a fused CUDA kernel (selective scan) that: loads `(A, B, C, Δ, x)` into shared memory once, runs the scan in register/SRAM without writing intermediates to HBM, uses parallel prefix scan (Blelloch) on block tiles, recomputes state in backward (memory-efficient à la Flash Attention). O(L) compute with small constant, O(1) HBM memory per token.

**Prerequisites on this machine:**
- ✅ RTX 2070 SUPER (Turing 7.5), Python 3.12, PyTorch 2.5.1+cu121, CUDA runtime 12.1
- ❌ **CUDA Toolkit (dev) 12.1.x** missing — must match `torch.version.cuda`. Mismatch (e.g. CUDA 12.4 with torch+cu121) → linker errors.
- ⚠ **MSVC Build Tools 2022** probably missing
- ❌ `CUDA_HOME` env var to set

**Procedure:**
1. **MSVC Build Tools 2022** from https://visualstudio.microsoft.com/downloads/?q=build+tools — workload "Desktop development with C++" (MSVC v143, Windows 11 SDK, CMake) ~6 GB.
2. **CUDA Toolkit 12.1** from https://developer.nvidia.com/cuda-12-1-1-download-archive (Windows x86_64, exe local, ~3 GB). Custom install: ☑ Development + Runtime; ☐ Driver components (you already have the RTX driver).
3. **Env vars** (PowerShell):
   ```powershell
   [Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1", "User")
   [Environment]::SetEnvironmentVariable("CUDA_PATH", "$env:CUDA_HOME", "User")
   # add $CUDA_HOME\bin to Path
   ```
4. **Install** (no build isolation, forces compile against installed PyTorch):
   ```powershell
   pip install causal-conv1d>=1.2.0 --no-build-isolation
   pip install mamba-ssm --no-build-isolation
   ```
5. **Edit** `quantsys/model/tcn_mamba.py`: replace `MambaBranch` with a conditional import of the official kernel, keep `SimplifiedMambaBlock` as fallback.
6. **Retrain** (checkpoints are NOT compatible): backup `models/tcnmamba` + `Remove-Item models\tcnmamba\best_model*.pt` + retrain TCN+Mamba from scratch. Verify `val_nll` converges to similar values (±5%). Expected time/epoch: ~1.5-2 min vs ~4.5 min currently.

**Rollback:** `pip uninstall mamba-ssm causal-conv1d` → the code auto-detects `_HAS_MAMBA_SSM = False` and uses the fallback.

**When to do it:** frequent TCN+Mamba retrains (ablation studies), `mamba_layers > 3`, longer sequences (T > 240 for multi-tf). **Not** if current training is "fast enough" or you're about to swap architecture.

---

## 🟢 Audit residue (low-priority, non-blocking)

4 MEDIUM issues + 1 INFRASTRUCTURE from the 2026-05-23 grand audit (8/8 CRITICAL + 8/8 HIGH + 5/9 MEDIUM already closed):

| # | File:line | Issue | Proposed fix | Effort |
|---|---|---|---|---|
| 21 | `quantsys/trading/__init__.py:395` | Cryptic NaN check `x != x`, only on `size` | Explicit NaN guard at the top of `open_position` with log warning | 10 min |
| 23 | `quantsys/data/__init__.py:48` | OHLCV sanity `high > close * 10` can discard legitimate flash crashes | Relax threshold or use previous candle price as reference | 15 min |
| 27 | `quantsys/model/ensemble.py:104-114` | `arch_names` not set in `load` fallbacks | Non-critical, default OK | 5 min |
| 28 | `quantsys/features/__init__.py:251` | `vol_x_pos` crashes if columns absent on short dataset | `.get(col, 0)` or try/except with log | 10 min |
| #5 ⚠ | `quantsys/trading/__init__.py:122` + `scripts/03_backtest.py:571-576` | `SignalGenerator.set_regime_threshold` exists but call sites DISABLED | Empirically calibrate regime thresholds on post-denorm-fix data (1-2h + retest), or remove dead code | 1-2h |

**#5 context:** 2026-05-24 bisect showed that hardcoded regime thresholds (overheating +3pp, stagflation +5pp over the 0.52 default) cut Sharpe from +18.71 to −4.44 (filtered 27/42 winning trades). Infrastructure stays but dead code.

Total close-out effort: ~1h (4 mediums) + 1-2h (#5 if you decide to calibrate).

---

## 📋 Paper-trading promotion thresholds

Independent of the fixes, must be satisfied SIMULTANEOUSLY before going live (3/4 met on 2026-05-23):

- ✅ Sharpe bootstrap CI (5000 iter): lower bound > 0 (+0.78)
- ✅ Stress test (`pessimistic_fee`, `flash_crash_vol`): at least break-even (+7.22 / +12.30)
- ⚠ **Walkforward WHR (3+ folds): > 0.53 stable** — iTransformer 0.567, but N-HiTS/TCN+Mamba 0.50-0.53 (per-fold models under-trained with max_epochs=40, recalibration post paper-trading via fix #4)
- ✅ Fee/gross ratio: < 30% (30.3% borderline)

These thresholds remain valid post BLOCKER #1 retrain; they must be reassessed on the new 104-feature models.

---

## 🧭 Golden rule

**One fix at a time, every change validated with bootstrap CI backtest.** Changing multiple things at once makes it impossible to causally attribute the delta.

Recommended pattern:
1. Apply a single fix
2. Full retrain (one base model if possible, e.g. iTrans, for a smoke test)
3. Compare `val_nll`, `DA`, `Spearman`, `Sharpe CI` with pre-fix baseline
4. If ≥2% improvement → keep and move to the next
5. If worse or unchanged → rollback and analyze before trying the next

**2026-05-24 lesson:** enabling "complete" fixes without pre-merge validation can activate uncalibrated dead state (case #5). Fast bisect (one fix at a time) finds the culprit in 2 iterations even on a complex codebase.

---

## 💡 Consolidated insights (long-term valid)

1. **Predictive model healthy in all setups**: walkforward DA 0.53-0.54, Spearman 0.08-0.09, well-calibrated σ. When problems emerge, they're almost always in the **trading layer** (scale, thresholds, SL/TP), not in the model — see the 2026-05-23 session for the paradigmatic case (Sharpe −256 → +18.7 from one missing multiplication).
2. **h=15 is structurally a losing setup**: roundtrip cost 26 bps ≈ |mean realized return| 25 bps. h=30 doubles the signal while keeping cost constant. Already applied.
3. **`max_sigma` must always be sized on the specific model's σ distribution** (e.g. p99 of σ_test). Arbitrary values are useless.
4. **`trailing_atr_mult: 1.5`** ≈ 11 bps of trail on BTC 1m → closes on noise (< 26 bps cost). On 1m bars `use_trailing_stop: false` (current) beats any tuned trailing.
5. **Verify scales unit-by-unit before retraining**: for 6+ sessions in May 2026 we hunted fixes on model weights (RevIN, h, stride, multi-teacher) — the actual bug was 1 missing multiplication across 2 files (z-score → raw denormalization).
