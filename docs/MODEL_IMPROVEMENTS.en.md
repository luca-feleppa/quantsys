# QUANTSYS — Outstanding model improvements

> Italian version in [MODEL_IMPROVEMENTS.md](MODEL_IMPROVEMENTS.md).

Everything already done lives in `CHANGELOG.md` and the notes under `~/.claude/projects/E--quantsys-project/memory/`. This file lists only what remains to implement, in recommended order.

---

## 🔴 NEXT — Diagnostics on the negative post-distill backtest 2026-06-03

**Context:** the `run_all.py --distill` that finished at 00:11 on 2026-06-03 produced worrying backtests:

| Arch | Sharpe | Win Rate | N trades | Return | Final equity |
|---|---|---|---|---|---|
| TCN+Mamba (teacher) | **-21.05** | 38.1% | 21 | -3.3% | $9,670 |
| iTransformer (student) | **-13.79** | 46.9% | 32 | -3.2% | $9,685 |
| NHits *(stale file 23-05)* | +18.71 | 64.3% | 42 | +3.7% | $10,367 |

**Diagnosed discrepancy:**
- Val (best epoch 2): TCN+Mamba DA **0.541**, Spearman **+0.102**
- Test set: DA **0.516** (-2.5%), Spearman **+0.023** (-77%)
- Test Spearman p-value = 0.022 → weak but statistically significant signal
- Backtest → **Sharpe -21** → the sign edge dissolves once translated into P&L

**⚠ ARTIFACT AVAILABILITY (verified 2026-06-03 00:30):** `run_all.py --distill` runs the backtest **only on the selected teacher** (`run_all.py:803-810`: `args.arch = selected_teacher` then `phase_backtest`). So:
- ✅ `results/tcnmamba/dashboard_results.json` (mtime 00:16) = real backtest of the distill run
- ❌ `results/itransformer/dashboard_results.json` (mtime 22:55 yesterday) = backtest of the **manual** pre-distillation iTrans training, NOT of tonight's distilled model
- ❌ `results/nhits/dashboard_results.json` (mtime May 23) = pre C-funding fix, completely stale

**To get valid backtests on the distilled students:**
```powershell
$env:QUANTSYS_ARCH = "itransformer"; python scripts/03_backtest.py
$env:QUANTSYS_ARCH = "nhits";       python scripts/03_backtest.py
```
~1 min each. Overwrites `results/{arch}/dashboard_results.json`. Required before drawing any conclusions about distillation performance.

**Hypothesised causes (ordered by likelihood):**
1. **Strong val→test distribution shift**: the recent-days test set captures a market regime different from the val one. Confirmation: Spearman collapses -77% val→test.
2. **Edge below fees**: WHR 0.508 vs 0.5 random → edge ~0.8% per trade. Round-trip fee 0.2% + estimated slippage 0.1% = 0.3%. Net residual edge ~0.5% per trade → thin margin.
3. **Signal generator not tuned for the new 104-feat models**: BUY/SELL/HOLD thresholds inherited from the previous setup (NHits 119 feat with Sharpe +18.7) produce too many trades during signal-less periods.
4. **Low trade frequency on test**: 21-32 trades over ~10k samples = 0.2-0.3% time in market → each trade weighs heavily, high statistical risk.

### Diagnostic steps (recommended order)

#### Step A — Verify RegimeSession (high priority, ~7 min, independent)
Confirm that Option C works: the new intraday regime detector produces a 33/33/33 val stratification instead of r0=100%.
```powershell
python run_all.py --arch itransformer --skip-update --skip-macro --force-download
```
Expected in the logs:
- `Stratified val: distribuzione regime: r0≈33%, r1≈33%, r2≈33%`
- `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=>0`

Even if the backtest will be ugly (expected), validating the Option C fix is independent from performance and must be closed out.

#### Step B — Distribution-shift investigation (medium priority, ~30 min)
Load the backup checkpoints `models/*/_bak_119feat_20260528/best_model.pt` and re-run the backtest on the **same current test set** (104 feat). Three possible scenarios:

- **Scenario B1**: old 119-feat models → also negative Sharpe → **market distribution shift**, not a pipeline regression. Decision: accept the new regime, possibly retrain with higher weights on recent samples (recency weighting in `02_train.py`).
- **Scenario B2**: old 119-feat models → positive Sharpe → **regression caused by C-funding** (the 15 dropped features carried signal that the C-funding score missed). Decision: revisit the 2026-05-28 C-funding decision, consider C-minimal or a sub-set of the 15.
- **Scenario B3**: old models with shape mismatch (119 != 104) → load fails → retrain a 119-feat model for a controlled comparison.

#### Step C — Signal generator audit (medium priority, ~20 min)
Search `quantsys/trading/` for BUY/SELL/HOLD thresholds and sizing parameters. Check whether they are hardcoded from a previous setup or adapt to the model's average CI. Possible fixes:
- Adaptive thresholds based on `σ_pred` (enter only if `μ_pred / σ_pred > threshold`)
- Min CI lower-bound > 0 filter (enter only if the interval doesn't cross zero)
- Position sizing inversely proportional to σ_pred

#### Step D — Paper-trading after Stage 4 integration (low priority, ~6-48h)
Only AFTER closing BLOCKER #1 Stage 4 (`LiveCandleBuffer` + `FeatureAssembler` integration into `LiveEngine`). Run paper-trading for 12-48h, accumulate 50-200 trades, compare live metrics with backtest. If live metrics persistently diverge from backtest → bug in signal generator or live/training matching, NOT in the model.

**Resume point for a new session:**
- Catastrophic backtest output documented here (numbers from `results/{arch}/dashboard_results.json`)
- Step A not yet executed (missing `Stratified val: r0≈33%` verification)
- Step B not yet executed (requires reloading the `_bak_119feat_20260528` backup)
- Step C not yet executed (requires grep on `quantsys/trading/` for signal thresholds)
- Operational decision pending: do A (verify) or B (investigate) first?

---

## 🟢 RESOLVED 2026-06-03 — Markov-Switching on BTC realized vol (Variant 3) implemented

**Status:** proposal, not implemented. Origin: 2026-06-02, iTransformer training shows `Stratified val: distribuzione regime: r0=10056 (100%)` on every validation → the current detector is degenerate (collapses to 1 cluster) and the per-regime diagnostics (`val_nll spread=0.000`) carry no information.

**Structural problem (not just a detector bug):**
- Hamilton 1989 on FRED + yFinance daily macro features → regimes change every **months**. Trading runs at 1-min with horizon h=30. A 4-5 order-of-magnitude mismatch between regime-detector scale and the model's operational scale.
- Verified in `scripts/02_train.py:577-1096`: the regime label is NOT an input feature to the model, it is used only for val stratification + diagnostic logging. So the current "bug" is cosmetic, but even fixing it (n_regimes 3→2) the added value for 1-min trading stays low.
- The 90 raw macro features + `MacroEncoder` (16-dim) already give the model the implicit "macro regime" — the aggregated label is redundant.

**Decision Option C — intraday regime detector on BTC:**

Replace the Markov-Switching on macro PC1 with a detector that observes **BTC microstructure directly** at a scale consistent with the trading timeframe (switching every ~1-4h, not months). Three candidate variants, ordered by increasing cost:

1. **Session regime (simplest)**: lookup on UTC `hour` → {Asia 00-08, EU 08-16, US 16-24}. Three regimes, deterministic, zero cost, the literature's ground truth for crypto (low-vol Asia, high-vol EU/US).
2. **Volatility regime via threshold**: rolling 4h percentile of realized volatility → {low / mid / high}. Switches 5-10 times a day, perfect match with h=30. Simple implementation (no EM, no PCA).
3. **HMM/Markov-Switching on BTC**: the same engine as today but observed on intraday realized volatility (rolling 1h log_ret²) instead of macro PC1. Switches 3-8 times per day. Reuses the existing `RegimeMarkovSwitching` infrastructure, only the input feature changes.

**Rationale for the final choice:** start from variant 1 (session) as the baseline, measure per-regime NLL spread on val. If spread > 0.05 NLL → regime is informative, worth moving to 2/3. If still 0.000 → the model doesn't discriminate between regimes (uniform signal), and the regime detector can simply be removed.

**Advantages vs current:**
- Switch frequency consistent with h=30 timeframe
- `val_nll per regime` diagnostic becomes informative again
- Effective val stratification (no more degenerate r0=100%)
- Possible future feature: regime label as model input (currently NOT used as feature)

**Files to touch (baseline session implementation):**
- `quantsys/macro/regime.py` → new `RegimeIntraday` class or `RegimeSession` variant (`session = floor(hour_utc / 8)`)
- `scripts/01b_download_macro.py` → fit/serialize the new detector (probably trivial, no EM)
- `scripts/02_train.py:385,577` → load the new regime for `_load_val_regimes` and stratification

**Validation:**
- After retrain, verify `Stratified val: distribuzione regime` has all regimes with ~25-40% coverage each (no longer 100% in r0)
- `val_nll per regime` spread > 0 (signal: the model does worse in some regimes)
- Backtest unchanged or improved (regime detector ≠ regression)

> ⚠ **After implementation, update `AVVIO.md`, `TEORIA.md` (§ "Markov-Switching"), `README.md` (and `.en.md` counterparts)** with the new regime-detector architecture. The current section in `TEORIA.en.md` (`statsmodels.MarkovRegression` on macro PC1) will need to be replaced with the description of the chosen intraday detector.

### 🚧 Implementation status (live tracker — session-updated)

**Chosen approach:** Variant 1 — **session-based regime** (Asia/EU/US via `hour_utc // 8`). Deterministic baseline, zero cost, no EM. If the next training still shows ~0 per-regime NLL spread, consider variant 2 (volatility threshold) or drop the detector entirely.

**2026-06-02 22:35 session — code + docs completed via 3-subagent fan-out:**

| Task | File | Status | Notes |
|---|---|---|---|
| New `RegimeSession` class | `quantsys/macro/regime.py:848-1003` | ✅ done | Added as "STAGE 1c", drop-in with `fit_predict_walkforward` / `save` / `load`. Smoke test 9097 rows, distribution 3033/3032/3032 (~33% each). `RegimeMarkovSwitching` and `RegimeHMM` untouched as fallbacks |
| Pipeline switch | `scripts/01b_download_macro.py:28,89-115,208` | ✅ done | Import + use `RegimeSession(n_regimes=3)`. Output filenames `regime_hmm.pkl` and `regime_probs.parquet` unchanged (backward compat with `_load_val_regimes` consumer) |
| TEORIA.md (IT) | §4 lines 76-83, §9 line 214, diagram ~277 | ✅ done | Section rewritten from scratch, ASCII diagram separates macro path from regime path |
| README.it.md (IT) | bullet line 36, diagram 92, tree 144/160 | ✅ done | Bullet "Rilevamento regimi" → session-based + fallback note |
| TEORIA.en.md (EN) | §"Macro regime detection" lines 76-83, diagram ~277 | ✅ done | Mirror of IT — new UTC-session detector description |
| README.md (EN) | bullet line 36, diagram 92, tree 144/160 | ✅ done | Mirror of README.it.md |
| AVVIO.md (IT) | no descriptive regime section | ⊘ skip | File is quickstart launch — does not describe the regime detector, nothing to update |
| AVVIO.en.md (EN) | no descriptive regime section | ⊘ skip | Same reason as AVVIO.md |
| Tests | `tests/test_features.py:212-226` | ⊘ skip | Existing `RegimeMarkovSwitching` test stays valid (class not removed); a `RegimeSession` test is optional, non-blocking |
| Real-pipeline smoke test | `python scripts/01b_download_macro.py` | ✅ done 2026-06-02 22:54 | `data/regime_probs.parquet` regenerated: 73777 hourly rows 2018-01-01→2026-06-02, distribution **33.3% / 33.3% / 33.3%** (24593/24592/24592). Time ~129s |
| End-to-end verification Phase 1+1b+2+3+4+5 | 3-subagent fan-out | ✅ done 2026-06-02 22:55 | All scripts run by `run_all.py --distill` verified: syntax+imports+smoke OK. IC fix sanity check: `ic_mean=0.3728 ≈ spearman=0.3726` on a 30% skill signal (mathematically consistent) |
| Verification iTransformer retrain | `run_all.py --arch itransformer --skip-update --skip-macro --force-download` | ✅ done 2026-06-03 | Stratified val **46% / 12% / 41%** (vs previous 100% r0 collapse), `val_nll` spread **0.19-0.30** (>> 0.05 "informative" threshold), 5/5 ensemble models converge stably |

**Resume points for the next session (in case of out-of-tokens):**
1. **Next step:** run `python scripts/01b_download_macro.py` to regenerate `data/regime_probs.parquet` (1-2 minutes). NB: it also downloads FRED/yFinance data — if those are already up to date, redoing the download is fine (idempotent).
2. **After step 1:** rerun iTransformer training to verify. Command: `python run_all.py --arch itransformer --skip-update --skip-macro --force-download` (~7 min). Look for `Stratified val: distribuzione regime` in the logs — expected ~33%/33%/33%, NO longer `r0=100%`.
3. **Final validation:** look for `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=Z` every 5 epochs. If `spread > 0.05`, regime is informative and worth keeping. If still `spread ≈ 0`, consider variant 2 (volatility threshold) or full removal.
4. **Optional unit test:** add `test_regime_session.py` in `tests/` verifying determinism + balanced distribution (non-blocking for merge).

**Rollback decision:** if the new `RegimeSession` does not improve per-regime NLL spread within one full training, consider variant 2 (rolling 4h volatility threshold). The `RegimeMarkovSwitching` class stays in the codebase as fallback (do not remove).

**Closure 2026-06-03:**

- ✅ **Variant 3 implemented**: new `RegimeMarkovBTC` class in `quantsys/macro/regime.py` (Hamilton 1989 Markov-Switching on BTC hourly realized volatility + expanding-window PCA, ~65-73% variance explained). `scripts/01b_download_macro.py` now uses it instead of `RegimeSession`. Output file (`data/regime_probs.parquet`) and schema unchanged for backward compat with `_load_val_regimes`.
- ✅ **3 data-driven regimes emerged** on ~9100 hours of BTC (post 30d burn-in): **R0 Quiet ~42%** (σ²(PC1)=0.56, drift≈0, P(stay)=89%), **R1 Trending ~18%** (σ²=0.12, drift=+0.08, P(stay)=92%), **R2 Stress ~40%** (σ²=3.79, drift=−0.12, P(stay)=79%, high vol + dump bias). Typical switching 3-8 times/day, consistent with h=30.
- ✅ **Val stratification no longer degenerate**: iTransformer retrain (5/5 ensemble) shows distribution **46% / 12% / 41%** (vs previous 100% r0 collapse). Per-regime `val_nll` spread **0.19-0.30** stable (>> 0.05 "informative" threshold) → regime is effectively informative for the model.
- ✅ **Docs + memory updated**: `TEORIA.md` + `TEORIA.en.md` (Markov-Switching section rewritten), `README.md` + `README.it.md` (bullet "Regime detection" updated), session memory `session_2026_06_03_markov_btc.md`.
- ⊘ **Rollback decision no longer applicable**: validation passed (variant 3 yields NLL spread >> 0.05 and balanced stratification), no fallback to variants 1/2 needed. `RegimeMarkovSwitching` and `RegimeSession` remain in the codebase as alternative classes but no longer on the production path.

---

## 🔴 BLOCKER #1 — Live↔training feature alignment (Stage 2-5)

**Status:** Stage 1 done (code), Stages 2-3 done, Stage 4 in progress, Stage 5 pending. Paper-trading **cannot** start until the mismatch is fully resolved.

**Problem (verified 2026-06-02 with `scripts/99_replay_live_vs_training.py`):** the backtest uses the C-funding-filtered `FeatureBuilder` (**104 features** post Stage 2); the live engine (`LiveFeatureBuffer._compute_features` in `scripts/04_live_signals.py`) builds **only 39** by hand in a different order, with per-window median/IQR normalization (not the `pipeline_state`'s `RobustScaler`), and `_predict` does blind positional pad/truncate. Three overlapping mismatches (count + order + scale) → live inputs effectively uncorrelated from training. **Current paper-trading signals do NOT reflect the backtest.**

Root cause: the reduced `LiveFeatureBuffer` exists because the full `FeatureBuilder` requires long history (ATH/ATL 365d, momentum 90d, frac-diff, vp_*_long) not available in the live rolling buffer (260 candles).

### Decision (2026-05-28): Option C-funding (~104 features)

**Rationale from permutation importance** (heterogeneous ensemble, 2500 val windows, permutation per group/feature): the 23 "live-hostile" features (lookback > buffer: 30/90/365d, frac_diff_*, vp_*_long, vp_poc_convergence, funding) have **ROI ≤ 0** for the h=30 model: bulk-permuting them *slightly improves* metrics (DA 0.529→0.532, Spearman 0.069→0.076). Sole exception: the **funding** features (`momentum_x_funding` +0.0042, `funding_rate_dev` +0.0028).

**C-funding set** = single source of truth shared by training/live:
- Drops 15 live-incompatible features (90d, 365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`).
- Keeps 30d + funding (positive ROI, computable live via a 30d ring buffer ~170 KB and a Binance funding poll).
- Target: ~104 total features (vs 119 before).

> The "full hybrid" scheme that kept all 30/90/365d features in live was documented as an alternative but **not recommended by the data** (negative ROI on the long tier) — definitively discarded.

### Stage 1 — code ✅ DONE

`LIVE_DROP_FEATURES` (15 features) in `quantsys/features/__init__.py`, filtered in `scripts/01_download_data.py` (`feat_cols = [c for c in feat_cols if c not in LIVE_DROP_FEATURES]`).

### Stage 2 — Dataset regeneration at 104 feat ✅ DONE 2026-06-02

Performed automatically inside `run_all.py --distill`: the dataset was regenerated as `(80390, 120, 104)` train + `(10049, 120, 104)` val + `(10049, 120, 104)` test, with the C-funding filter correctly applied (15 features dropped, programmatically verified).

### Stage 3 — Full distill retrain ✅ DONE 2026-06-02

Executed in the same `run_all.py --distill` of 2026-06-02: all 3 models (iTransformer, N-HiTS, TCN+Mamba) retrained from scratch at 104 features. Multi-teacher distillation applied to the students selected by automatic scoring (see `models/{arch}/config.json` for the `distilled: true, teacher_arch: "multi-teacher"` flags).

> Backtest metrics for the 104-feat models: to be re-read from `results/{arch}/dashboard_results.json` after the run completes (may differ from the +18.71 Sharpe of the 119-feat setup).

### Stage 4 — Live engine rewrite 🚧 IN PROGRESS (2026-06-02 23:10 session)

**Architectural decision:** instead of duplicating feature-engineering logic in `LiveFeatureBuffer`, **directly reuse `quantsys/features.FeatureBuilder.build()`** on the live buffer. Automatic single source of truth → parity test guaranteed by design.

**Rationale:**
- The live↔training feature delta is ~65 features (live has 39, training has 104). Hand-rewriting these 65 to match `FeatureBuilder` carries high silent-drift risk.
- `FeatureBuilder.build()` on 43200 rows × ~120 columns takes ~1-3s on CPU. Run at every candle close (60s budget) it fits comfortably.
- Memory: 43200 candles × 104 float32 = 18 MB. Negligible.
- The 30d features (dist_ath_30d, momentum_30d, price_vs_ma200m) require 43200 candles of history → a "warm" buffer must be bootstrapped from the historical parquet at boot.

**New live engine architecture:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ LiveCandleBuffer (50,000 raw OHLCV candles, ring buffer)            │
│  ├─ bootstrap: reads raw_candles.parquet[-50000:] at boot           │
│  └─ append(candle): push new, pop old (FIFO maxlen)                  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FundingRatePoller (deque(maxlen=30d ÷ 8h = 90), poll every 1h)      │
│  └─ uses quantsys.data.fetch_funding_rate                           │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FeatureAssembler (called at every candle close)                     │
│  1. df = pd.DataFrame(LiveCandleBuffer.tail(43200))                  │
│  2. df_with_funding = merge_asof(df, funding_history)                │
│  3. feat_df = FeatureBuilder.build(df, fit=False, normalize=True)    │
│  4. Assert feat_df.columns == PipelineState.feature_names (HARD-FAIL)│
│  5. Filter LIVE_DROP_FEATURES (already in build)                     │
│  6. Extract window[-120:, :] → (120, 104) np.ndarray                 │
└──────────────────────────────────────────────────────────────────────┘
```

**Files to touch:**
- `quantsys/features/__init__.py` → expose `get_canonical_feature_names(npz_path)` as single source of truth
- `quantsys/utils/__init__.py` `PipelineState` → add `feature_names: list[str]` attribute (persisted in pickle)
- `scripts/04_live_signals.py` → replace `LiveFeatureBuffer` with `LiveCandleBuffer` + `FeatureAssembler` + `FundingRatePoller` integration; remove `_pad_or_truncate` from `_predict`
- `tests/test_live_training_parity.py` → new: parity test (live output == FeatureBuilder on historical window with 1e-6 tolerance)
- `scripts/99_replay_live_vs_training.py` → update to use the new engine

### 🚧 Stage 4 implementation tracker (live — updated at every milestone)

**Active session:** 2026-06-02 23:10 (parallel to the ongoing distill, GPU unaffected since live engine is CPU-only)

| Step | File | Status | Resume notes |
|---|---|---|---|
| 4.1 — Expose canonical `feature_names` | `quantsys/features/__init__.py:13-58` | ✅ done 2026-06-02 23:25 | `get_canonical_feature_names(npz_path)` added. lru_cache(maxsize=4). Hard-fail on FileNotFoundError/KeyError. Smoke test: 104 correct names, cache active (2nd call <0.001ms), zero overlap with LIVE_DROP_FEATURES. |
| 4.2 — `PipelineState.feature_names` | `quantsys/utils/__init__.py:152` | ⊘ skip | Verified 2026-06-02 23:20: `PipelineState.feature_cols` has 121 elements (pre-filter, includes `LIVE_DROP_FEATURES` + `target_ret`/`target_dir`), `scale_cols` has 105 (104+target_ret). Not the canonical single source of truth. Do not add a new attribute — the NPZ remains authoritative, PipelineState provides `scaler` + `clip_lo_/hi_` + `n_dynamic_features` + macro state. |
| 4.3 — `LiveCandleBuffer` | `scripts/04_live_signals.py:97-185` | ✅ done 2026-06-02 23:18 | Smoke test: bootstrap of 50000 candles from parquet, to_dataframe(120) shape (120,9), append FIFO works, default fields=0 for missing fields. Tz-naive UTC as required by FeatureBuilder. |
| 4.4 — `FundingRatePoller` | `scripts/04_live_signals.py` (new class) | ⏳ pending | Thread/asyncio task calling `quantsys.data.fetch_funding_rate` every 1h. Deque(maxlen=90) for 30d × 3 rates/day. Expose `to_dataframe()` for merge. **Temporary workaround**: read `data/funding_rate.parquet` from disk at boot (already works, see test 4.5). The poller is only needed for real-time refresh. |
| 4.5 — `FeatureAssembler` | `scripts/04_live_signals.py:188-308` | ✅ done 2026-06-02 23:22 | **End-to-end OK**: compute_window(120) produces (120, 104) float32, no NaN, no Inf. Hard-fail on missing features. Tz-normalization of funding_df inside compute_window. Output stats: mean=-0.05, std=1.73, range [-24.8, +25.8] (consistent with training RobustScaler+clip). |
| 4.6 — Swap in `LiveEngine.__init__` | `scripts/04_live_signals.py` | ⏳ pending | Remove `self.buffer = LiveFeatureBuffer(...)`, replace with the 3 new components. Update the main loop. **Do NOT delete** the legacy `LiveFeatureBuffer` class (comment "DEPRECATED — legacy 39-feature implementation"). |
| 4.7 — Remove pad/truncate | `scripts/04_live_signals.py:982-988` | ⏳ pending | In `_predict()`, replace the `if n_live < n_model: pad ...` block with `assert window.shape[1] == n_model, "feature mismatch"`. The window already arrives at 104. |
| 4.8 — Parity test | `tests/test_live_training_parity.py` | ✅ done 2026-06-02 23:30 | **4/4 tests pass in 12.8s**. (1) Assembler output == direct FeatureBuilder output with `max abs diff < 1e-5`, (2) canonical order stable, (3) zero overlap with LIVE_DROP_FEATURES, (4) hard-fail on funding=None. Parity mathematically verified. |
| 4.9 — Update replay script | `scripts/99_replay_live_vs_training.py` | ✅ done 2026-06-02 23:33 | Rewritten from scratch to use `LiveCandleBuffer` + `FeatureAssembler`. Run output: **Max diff: 0.000e+00** (bit-perfect parity). Was 3 mismatches, now 0. |
| 4.10 — Live smoke test | `python scripts/04_live_signals.py` | ⏳ pending | WS connection, full warmup, first signal within 2 min. Verify no "feature mismatch" warnings. |
| 4.11 — Update doc `MODEL_IMPROVEMENTS.md` | this file | ⏳ pending | Move Stage 4 from pending to done, update `BLOCKER #1` header to `🟢 RESOLVED` if Stage 5 also closed |

**Resume points for session reset:**
- **If at step 4.1-4.2**: non-destructive work, can resume from anywhere
- **If at 4.3-4.7**: `scripts/04_live_signals.py` is in an intermediate state — the legacy `LiveFeatureBuffer` class must stay in place until all new components are tested. Before committing, verify the script at least imports (`python -m py_compile`)
- **If at 4.8-4.10**: testing-only, safe
- **Final post-implementation verification**: `scripts/99_replay_live_vs_training.py` must produce "✅ 0 mismatches"

**Current state (updated 2026-06-02 23:35):**
- ✅ Core components implemented: `get_canonical_feature_names`, `LiveCandleBuffer`, `FeatureAssembler`
- ✅ Parity test mathematically verified: **Max diff 0.000e+00** on 50k-candle replay
- ✅ 4/4 unit tests in `tests/test_live_training_parity.py` pass in 12.8s
- ✅ Script `99_replay_live_vs_training.py` updated: previously 3 mismatches → now 0
- **NEXT STEPS (for new session):**
  1. **4.6 — Integrate into `LiveEngine.__init__`**: replace `self.buffer = LiveFeatureBuffer(...)` with the 3 new components. The main loop in `LiveEngine._on_candle_close()` (or similarly named) calls `compute_window()` on `FeatureAssembler` instead of `get_window()` on `LiveFeatureBuffer`. Grep `self.buffer` to find all call sites.
  2. **4.7 — Remove `_pad_or_truncate`**: in `_predict` (lines ~964-988) replace the `if n_live < n_model: pad ...` block with `assert window.shape[1] == 104, "feature mismatch"`. The window already arrives at 104 from the new path.
  3. **4.4 — `FundingRatePoller`**: implement poller via asyncio task + `quantsys.data.fetch_funding_rate` every 1h. For now workaround: `LiveEngine.__init__` reads `data/funding_rate.parquet` at boot and passes a copy to `FeatureAssembler.compute_window()`.
  4. **4.10 — Live smoke test**: `python scripts/04_live_signals.py` → WS connection, warmup, first signal within 2 min. Verify no "feature mismatch" warnings.
  5. **4.11 — Final doc update**: mark `BLOCKER #1 ✅ DONE` and update TEORIA.md/AVVIO.md/README if they still mention "39 mismatched live features".

**Files modified in this session (commit-ready):**
- `quantsys/features/__init__.py` (+50 lines: `get_canonical_feature_names`)
- `scripts/04_live_signals.py` (+~220 lines: `LiveCandleBuffer` + `FeatureAssembler`; legacy `LiveFeatureBuffer` intact with DEPRECATED tag)
- `scripts/99_replay_live_vs_training.py` (rewritten: was pad-trunc check, now parity diff)
- `tests/test_live_training_parity.py` (new: 4 tests, 12.8s)

---

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
| Storage `lstm_dataset.npz` | (107480, 120, 104) ~6.1 GB | + (107480, 24, 104)×2 | **+40%** (~8.5 GB) |
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
