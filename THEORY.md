🇬🇧 English · [🇮🇹 Italiano](THEORY.it.md)

# QUANTSYS — How the system works

A descriptive walkthrough of the full flow, from raw material (candles) to the operating signal (BUY / SELL / HOLD with size and stop loss).

> **Orientation — two lines, only one in production.** The same engine (data → 104 features → 120×104 windows → probabilistic archs → ensemble/distill) serves two `target_type`s. **Production: VOLATILITY at 1h** (`target_type: log_rv`, `interval: 1h`, 5-member iTransformer) — the only signal with a verified OOS edge, feeding the options line (`04b_vol_paper.py`). **DIRECTIONAL line** (`target_type: ret`, Part V): **legacy / KILLED OOS**, code path bit-invariant as a baseline and negative control. Overview and motivation: `README.md`.

---

# Table of contents

The document follows the ML-standard pipeline: **(0) Theoretical setup** → **(I) Data** (target, log-returns, features, normalization / z-score invariant) → **(II) Modeling** (architectures, loss, ensemble, distillation, regime) → **(III) Evaluation** (walk-forward, QLIKE/Spearman, distribution shift, cross-arch ensembling) → **(IV) Inference** (Monte Carlo, denormalization) → **(V) Directional trading layer** (signal, risk, live — *legacy, KILLED OOS*) → **(VI) Cross-cutting scientific reading** (what generalizes, and why). Cross-references use Part numbers. This file is **the mathematics**: operating commands live in `START.md`, the overview in `README.md`, the experiment chronicle in `CHANGELOG.md` / `STATUS.md`.

---

# Part 0 — Minimal theoretical setup

**Stochastic object.** Let `p_t` be the BTC/USDT price at bar `t` and `r_t = log(p_t / p_{t−1})` the log-return (stationary, symmetric). The engine estimates the **predictive density** of the process moments at horizon `h=30` bars: the directional line predicts the **first moment** (μ of the cumulative return `Σ_{i=1}^{h} r_{t+i}`), the vol line predicts the **second moment** (the realized variance `RV = Σ r²`). The project's central empirical thesis is that **even moments (volatility level) generalize out-of-sample, odd moments (direction sign, signed semivariance asymmetry) do not.**

**Notation conventions.** μ/σ/ν = parameters of the predictive Student-t density (mean, scale, degrees of freedom). All forecasts are emitted in **z-score space** (standardized by the global RobustScaler) and mapped back to **raw space** before any operational/evaluative use (Part I, invariant section). Windows are expressed in **bars**; their wall-clock duration depends on `data.interval` (1h current, 1m legacy) — see the interval contract in Part I.

---

# Part I — Data

## 1. Price data collection

Starting point: Binance, OHLCV (Open, High, Low, Close, Volume) candles on BTC/USDT. The current timeframe is **1 hour** (`data.interval: 1h`), history window **2019-01-01 → today** (multi-year, ~65k bars) via `data.start_time`. History is stored in Parquet (columnar, compressed); subsequent runs download only the delta from the last local candle. The previous 1m perimeter (2025-05-19 → today, ~525k candles) survives **as raw data only** in `data/backup_1m/`: the 1m checkpoints were deleted on 2026-06-12, so a 1m rollback requires a retrain.

**Econometric rationale of the move to 1h — and its outcome.** The round-trip cost (fee + slippage, ~26 bps) is **fixed** per trade, while the standard deviation of a bar's move grows **∝ √Δt**: the cost/σ ratio therefore scales as **1/√Δt**. At 1m it was ~1.9–3.3×, at 1h it drops to **~0.25–0.42×**. The necessary counterpart is the multi-year history: at 1h a single year would yield only ~8.7k bars, insufficient for training. ⚠ **Measured outcome (2026-06-10): at 1h the cost wall falls but no directional skill emerges OOS** → the pivot *as a directional strategy* is **KILLED** (Part VI). The 1h perimeter stays because it is the natural setting of the vol line (hourly RV, HAR-RV/QLIKE), which is the current production.

---

## 2. Log returns

Raw prices are converted to **log returns**: instead of absolute price, the system works on the percentage change between consecutive candles in log scale. Advantages:
- Stationary (no rising trend).
- Symmetric (a +10% and a −10% have the same absolute weight).

The **target** is the sum of log returns over the next **30 bars** (`forecast_horizon: 30`). The horizon is defined in **bars**, so its duration in time depends on the timeframe: **30 hours at the current 1h timeframe** (it was 30 minutes at 1m). Horizon h and timeframe Δt are the two levers of the same cost/σ ∝ 1/√(h·Δt) argument of §1: lengthening h at fixed Δt raises the expected move while the cost stays fixed.

**Target type (`features.target_type`).** Three definitions over the same h future bars:
- `ret` — directional sum of log-returns `Σ_{i=1}^{h} r_{t+i}` (legacy bit-invariant path, **first moment**);
- `log_rv` — **production**: `log(Σ r² + 10⁻¹²)`, log realized variance (**second moment**), with `target_dir` = future RV > trailing h-bar RV (causal);
- `log_rs_ratio` — `log((RS⁺+ε)/(RS⁻+ε))` with `RS± = Σ r²·1[r ≷ 0]`: signed asymmetry of realized semivariance (Barndorff-Nielsen et al. 2010 / Patton–Sheppard 2015), a **third-moment** ratio.

The log makes the distribution ≈ Gaussian → RobustScaler / NLL / denormalization work unchanged. ⚠ With `log_rv` the target median is ≈ −7.2 (not ≈ 0): the z→raw inversion requires `μ·IQR + center`; `denormalize_predictions` alone is **not enough** (§5, invariant); the log-ratio is instead near-centered (|center| < 2). **Outcomes:** `log_rv` PASS at 1h (NN beats HAR-RV by ~30% in test QLIKE), FAIL at RV-30min → resolution-specific edge; `log_rs_ratio` FAIL (unpredictable for the NN *and* HAR-RS). This is the **even / odd moment** dichotomy (Part VI). No trading backtest on vol models: the target is not a return.

---

## 3. Feature engineering — what the model sees

**104 features** per candle (**86 dynamic + 18 structural**). The count is **verified on the dataset, not assumed**: the canonical list is derived once by `canonical_feature_columns` (`quantsys/features/__init__.py`) reading `feature_names` from the npz, protected by a golden test. It is after the **C-funding** filter (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`, decision 2026-05-28): 15 live-incompatible features with ROI ≤ 0 were removed (90d/365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`) — rationale (permutation importance over 2500 val windows): the long-lookback tier has ROI ≤ 0, the funding features being the only exceptions.

### Timeframe contract — TIME-semantic vs BAR-semantic windows

Since the 2026-06-09 pivot the `FeatureBuilder` is **interval-agnostic**: it receives `interval_minutes` (derived from `data.interval` via `interval_minutes_from_cfg` in `quantsys/utils`, fail-fast `ValueError` on unknown intervals), computes `bars_per_day = 1440 // interval_minutes` and converts minutes to bars with the `_tbars(minutes)` helper (anti-degeneration floor of 2 bars). Windows fall into two classes:
- **TIME-semantic** — keep their meaning in **calendar time**, converted to bars: structural ATH/ATL 30d/90d/365d and momentum 7d/30d/90d (`days × bars_per_day`), `funding_rate_1d` (24h = `bars_per_day` bars), `session_position` (240 minutes), `price_vs_ma200m` (200 minutes).
- **BAR-semantic** — deliberately **unchanged in bar counts** (they shift with the timeframe): rolling windows [5, 10, 20, 60], CVD, VWAP, Volume Profile at 60/240/1440 **bars** (at 1m = 1h/4h/1 day; at 1h = 60h/10 days/60 days), lag returns.

At `interval_minutes=1` every conversion is an **identity** → the legacy 1m behavior is preserved exactly.

**Per-interval config overlay (2026-06-10).** The interval-dependent keys (`data.interval`/`start_time`, `model.window_stride`, `validation.embargo_steps`, `risk.max_hold_candles`, `backtest.min_expected_ret`/`max_sigma`) are factored into `config/interval/{1m,1h}.yaml`. `load_config` merges them per-section (shallow) **after secrets and before the arch overlay** — hierarchy: default → secrets → interval → arch. Activated via the `QUANTSYS_INTERVAL` env var or `run_all.py --interval`.

### Feature catalogue

(each window's BAR-/TIME-semantics is the one from the contract above)
- **Trend and momentum** — rolling moving averages over 5/10/20/60 bars, ratios across scales, momentum/volatility ratios. RSI/MACD removed: the same information already sits in the `vol_std` + `lag_ret` + microstructure mix.
- **Volatility** — rolling std of log-returns over 5/10/20/60 bars. ATR removed from the inputs (redundant with `vol_std`); still computed by the `RiskManager` for dynamic stop sizing (§10).
- **VWAP** — volume-weighted average price; the price−VWAP distance measures the deviation from the session equilibrium.
- **Multi-scale Volume Profile** — volume distribution per price level over **60/240/1440 bars** (at 1m = 1h/4h/1 day; at 1h = 60h/10 days/60 days); 4 features per window (distance from POC, VAH, VAL + concentration at the POC). In high volatility the short scale dominates, in low volatility the long one is more stable. ⚠ The **1440** scale is computed but **does not enter the 104**: `vp_*_long` (and `vp_poc_convergence`, which uses it) are in `LIVE_DROP_FEATURES` → the model sees **8** VP features, on the 60 and 240 scales only.
- **CVD** — cumulative taker-buy minus taker-sell volume: a CVD↑ / price↓ divergence flags buying pressure not yet priced in.
- **Microstructure** — 10 features from candle geometry (body ratio, upper/lower shadow, price velocity, price acceleration): **instantaneous** information, without the construction lag of classic oscillators.
- **Funding rate** — BTC/USDT perps every 8h → `funding_rate`, `funding_rate_1d`, `funding_rate_dev`. High funding = crowded longs (downside short-squeeze risk); negative = crowded shorts.
- **Temporal** — time of day, day of week, day of month: crypto liquidity has intraday and weekly seasonality.
- **Lags** — log-returns of the last 5 bars as direct regressors (explicit autoregressive component).
- **Interactions** — 3 explicit products (`vol_x_pos` = volatility × VWAP position, `momentum_x_funding`, `cvd_x_vol`): second-order terms the model would otherwise have to learn implicitly.

**HAR-CJ (A4, inert lever — `features.har_cj`, default OFF).** Continuous/jump decomposition of realized variance as input (Andersen–Bollerslev–Diebold 2007): bipower variation `BV = (π/2)·mean(|r_t|·|r_{t-1}|)` — jump-robust estimator of the continuous component — then `J = max(RV − BV, 0)` and `jump_ratio = J/RV ∈ [0,1]`, on TIME-semantic 1d/1w scales. With the flag OFF (production default) the 104 features stay bit-invariant; activation only at a planned retrain with a pre-registered gate.

---

## 4. Macro data (FRED + yFinance)

Low-frequency exogenous regressors (daily/monthly), processed separately and merged with the price stream at training time (90 columns → 16-dim `MacroEncoder`): **DXY** (BTC tends to move inversely to the dollar), **VIX** (risk-aversion spillover from traditional markets), **rates** Fed Funds / Treasury (cost of capital), **gold** (comparable safe-haven asset). The frequency mismatch is the binding constraint: macro information is nearly constant over the h=30-bar horizon, so it acts as slow conditioning, not as a per-bar predictor.

### Regime detection (Markov-Switching on BTC realized volatility, hourly)

**Object.** The detector runs **directly on BTC data**, not on US macros: `RegimeMarkovBTC` (`quantsys/macro/regime.py`) is a Markov-Switching model (Hamilton 1989) on BTC realized volatility aggregated **hourly**. It supersedes the MS on PC1 of daily macros (regimes switching on a monthly scale — incompatible with h=30) and the `RegimeSession` baseline (deterministic Asia/EU/US: **temporal** clusters, not market clusters, informationally empty).

**Features.** From `data/raw_candles.parquet` (candles at any interval **≤1h** — the regime clock is HOURLY by design, independent of the trading timeframe; >1h input → fail-fast `ValueError`), aggregated per hour:
- `log_ret_h` = sum of per-bar log-returns over the hour (with 1h input the resample is an identity);
- `log_rv` = `log(Σ log_ret²)` clipped at 1e-12 — raw RV is heavily right-skewed, the log stabilizes it for `MarkovRegression`. With 1h input each bucket holds a single observation → rv = the bar's log_ret² (a poor but valid proxy of hourly RV).

**Statistics.** Global RobustScaler (median/IQR, declared-negligible look-ahead) → expanding-window PCA with `n_pca=1`, collapsing `log_ret_h` + `log_rv` into a single **motion-intensity** signal (PC1 explains ~65–73% of the joint variance) → `MarkovRegression` on PC1 with switching **mean + variance** → manual Hamilton filter, O(1) per hourly step between retrains. **Expanding** walk-forward, cadence from `macro.hmm_burn_in_days` / `hmm_retrain_days` (30-day burn-in, retrain every 90 days): the expanding refit is O(t), a full rebuild over 7 years costs ~3h (monthly cadence ~9h).

⚠ **Why the detector is CAUSAL — `filtered_marginal_probabilities`, NEVER the `smoothed_` ones.** The Hamilton filter yields at each step the **filtered** probability via a predict → update recursion:

```
predict:  P(S_t=j | y_{1:t-1}) = Σ_i P(S_t=j | S_{t-1}=i) · P(S_{t-1}=i | y_{1:t-1})
update:   P(S_t=j | y_{1:t})   ∝ f(y_t | S_t=j) · P(S_t=j | y_{1:t-1})
```

Conditioning is on `y_{1:t}`: **only information available at time t**, exactly what live would have. The *smoothed* probabilities (Kim 1994) are instead `P(S_t=j | y_{1:T})`, produced by a **backward** recursion conditioning on the whole sample, `y_{t+1..T}` included: they are better *ex post* regime estimators, but using them to stratify val, as a gate or as a feature would inject the future into the decision at `t` — **lookahead bias**, and irreproducible in production by construction. For the same reason the walk-forward chain re-estimates the MS parameters on past data only, and the filter stays forward-only between retrains.

**Incremental refresh (B7).** Besides `regime_hmm.pkl` (only the final full-sample fit for `predict_proba`), the full rebuild persists `data/regime_wf_checkpoint.pkl` = the walk-forward **chain** state (last-retrain parameters, sign-aligned PCA, filtered posterior, cadence, frozen scaler). Resuming the chain extends the parquet to the new bars only with 0-1 MLE fits (minutes instead of hours), **bit-parity by construction** (golden test `tests/test_regime_incremental.py`). The only accepted, documented deviation: the scaler stays frozen at the full rebuild → a periodic full rebuild re-anchors. Commands (`01b --regime-incremental`, `--regime-bootstrap-checkpoint`) and runbook: **`START.md`**.

⚠ **Regime indices have NO fixed semantics across runs.** The `R0/R1/R2` ordering is an MLE-fit artifact (label switching): it changes at every full rebuild. The Quiet/Trending/Stress names of the 1m 2025-26 span (R0 42% / R1 18% / R2 40%) do **not** map onto the current indices. Semantics must always be **re-derived from the current run's variances** `σ²`. 2026-07-15 run (7 hourly years 2019→2026, post burn-in; parameters from the final full-sample fit):

| Regime | Frequency | PC1 σ² | μ | P(stay) | Current semantics |
|---|---|---|---|---|---|
| **R0** | 31.1% | ≈ 0.12 | ≈ −0.01 | ≈ 92% | low vol ("quiet") |
| **R1** | 36.3% | ≈ 4.76 | ≈ 0 | ≈ 83% | high vol ("stress") |
| **R2** | 32.7% | ≈ 0.61 | ≈ 0 | ≈ 89% | mid vol (intermediate) |

**Use in the model.** The regime is **not an input feature**: it serves to (a) stratify the validation split (market clusters, not session clusters) and (b) feed the `val_nll per regime` diagnostic — if one regime shows systematically worse NLL, the model has a calibration gap on that micro-condition. Probabilities are persisted in `data/regime_probs.parquet` (UTC hourly index, columns `regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`). The US macros (FRED + yFinance) are decoupled from the detector but still consumed by the 16-dim `MacroEncoder`. The earlier detectors (`RegimeMarkovSwitching`, `RegimeSession`, `RegimeHMM`) survive in the module as **non-wired** fallbacks.

**Label-switching corollary.** The per-regime risk profiles of the directional `RiskManager` (§10) are **keyed on the INDEX**: until the index→profile mapping is re-derived from the current run's variances, any hardcoded rule like "regime=2 → cut sizing" is **STALE** by construction. Irrelevant for production (the vol line does not use the directional `RiskManager`), binding for anyone re-activating Part V.

---

## 5. Training dataset

Features are organized into **temporal windows**: every example is a `120×104` matrix (last 120 **bars** of context = **5 days** at the current 1h timeframe; it was 2 hours at 1m). The target attached to the window is the one selected by `features.target_type` over the next 30 bars (§2): log-RV in production, the sum of log-returns on the legacy line.

On the 1h dataset (2019→today) with `window_stride: 1` the current on-disk npz is `X_train (51882, 120, 104)`, split **51,882 / 6,485 / 6,486** (train/val/test ≈ 80/10/10, temporal and unshuffled) plus `X_macro_* (·, 90)`. The split grows with every raw-data extension: these figures must be **read from the npz**, not assumed.

**Why T=120 and not more.** The 120-bar window is an empirical sweet spot verified on 2026-06-04 **on the 1m perimeter**: experiments at **T=180 and T=240 both regressed** (monotone degradation of walkforward Spearman and backtest, collapsed μ_pred, below-random WHR). The ~525k 1m dataset lacked the informational depth for longer contexts (overfitting to temporal noise; the 192-384 literature plateau holds for multi-year multi-asset datasets). T=120 was **kept unchanged** in the 1h pivot (now = 5 days of context): do not increase it without new empirical evidence.

Normalization with a **global multi-column RobustScaler**, less sensitive to price spikes than the standard scaler. Parameters are persisted in `PipelineState` (`models/{arch}/pipeline_state.pkl`) to reapply the same transform at inference time.

### Critical invariant — z-score vs raw space

The target (`target_ret`, whatever `target_type`) is scaled by the RobustScaler along with the other features. The RobustScaler persists **two** per-column parameters in `PipelineState`: the **center** (raw median) and the **scale** (`target_scale` = IQR of the raw target on the training set); both vary with dataset, forecast horizon and `target_type`. Therefore:
- **The model predicts μ, σ, ν in z-score space** (standardized fraction). σ = 1.0 means "one IQR of the target", not "1% of price".
- **The trading/evaluation layer operates in raw space**: thresholds `min_expected_ret`, `max_sigma`, SL/TP (directional line) and QLIKE/RV inversion (vol line) assume raw target values.

**z→raw inversion — depends on `target_type`.**
- **Directional target (`ret`, legacy line).** The raw target is ≈ centered (log-ret median ≈ 0), so reconciliation reduces to `PipelineState.denormalize_predictions(mu, sigma) -> (mu_raw, sigma_raw)`, which multiplies by `target_scale` (the center is negligible on μ and structurally null on σ). Historical `target_scale` ≈ 0.002707 (2026-06-02 run, 1m, h=30). **Both `03_backtest.py` and `04_live_signals.py` apply it right after the forward pass**, before the `SignalGenerator`. Without it, SL/TP `σ × price × multiplier` become macroscopic (σ_z=1 × $42k × 1.5 = $63k) — structural bug fixed on 2026-05-23 (Sharpe −256 → +18.7).
- **Volatility target (`log_rv`, PRODUCTION line).** Here `denormalize_predictions` (only `μ·scale`) **is insufficient**: the log-RV median is ≈ −7.2, so the **full inversion** `log_rv = μ_z · scale + center` is required (with `center`/`scale` from the persisted RobustScaler), then `RV = exp(log_rv)` to return to levels. It lives in `quantsys/model/vol_metrics.py` (`invert_log_rv(z, center, scale)`, `qlike_from_z(...)`), the single source of truth shared by the judges (`scripts/vol/dev_vols_qlike.py`, walk-forward `02b`, `step0_xarch_corr.py`). Skipping the `center` shifts the estimated RV by a factor `exp(7.2)` ≈ **1300×**. The production model's `target_scale` = **1.4376** (IQR of log-RV); the canonical copy `models/pipeline_state.pkl` tracks the latest dataset regeneration and may differ by a few 1e-3. The robust discriminator is the **order of magnitude**: an IQR of ~1e-3 flags the directional target, ~1.4 the log-RV one.

**For any new entry point**: always call `denormalize_predictions` before interpreting μ/σ. Safety nets against regressions:
- `RuntimeError` in `03_backtest.py` if `σ_max ≥ 0.05·√interval_minutes` (0.05 at 1m, ≈0.387 at 1h; `raise` instead of `assert` survives `python -O`). The √Δt scaling preserves the guard's intent: it catches the z→raw denormalization bug (~30–100×), not the legitimate √60 growth of σ at a 30-hourly-bar horizon.
- Runtime warning in `_sl_tp` if `σ × price × 1.5 > 5% × price`.
- `PipelineState.forecast_horizon` validated in backtest + live: if `cfg.data.forecast_horizon != state.forecast_horizon` → `RuntimeError` (prevents using a h=30 model with a h=15 backtest).
- `PipelineState.interval_minutes` (property, fallback 1 for legacy pkl) validated in backtest + live with the same pattern: a 1m-trained model + 1h config → `RuntimeError` "interval mismatch". Live/replay consumers derive the interval from the `PipelineState`, not from the config.
- `merge_asof` between test set and raw_candles validated with `len(merged) == n_test_orig`, otherwise `RuntimeError` (prevents SL/TP triggered on wrong candles due to Binance gaps/halts).
- `update_trailing` updates `portfolio.equity` mark-to-market every candle (cash + size_usd + unrealized_pnl): the circuit breaker fires on intra-trade DD in live too.
- Floor `sl_d = max(sl_d, price × 1e-4)` in `_sl_tp` to avoid silent SL=TP=entry when ATR=0 (market halt).

---

# Part II — Modeling

*Sections 6–7 cover the model: probabilistic architectures, loss, ensemble (law of total variance), target-aware multi-teacher distillation. Regime detection (Markov-Switching) is described in §4 because it shares the macro/BTC ingestion pipeline, but conceptually it belongs to modeling (stratification + diagnostics, not an input feature).*

## 6. Available architectures

Four architectures selectable via `--arch`: `lstm`, `itransformer`, `tcnmamba`, `nhits`. The heterogeneous ensemble composition lives in `config/default.yaml` → `distillation.archs` (iTransformer + N-HiTS + TCNMamba; LSTM available but outside the ensemble due to structural under-performance). **Production model (vol line): the 5-member iTransformer** in `models/itransformer/` (target `log_rv`, `interval=1h`, QLIKE PASS validated twice OOS — single-split test and k-fold on the data-rich folds). ⚠ **On disk only `itransformer` and `lstm` have checkpoints**: `models/{nhits,tcnmamba}` were deleted in the 2026-06-12 cleanup, so any heterogeneous run (ensemble or distill) requires retraining them first. The 5-seed multi-teacher vol distill stays a gated experiment (low prior: note at the end of §7).

### LSTM dual-stream (`--arch lstm`, legacy)

Recurrent net with the 104 features split into two streams:
- **Dynamic stream** (86 features): log returns, CVD, volume delta, microstructure — short-term momentum. Processed by an LSTM.
- **Structural stream** (18 features): VWAP, Volume Profile (short + mid), temporal features, ATH/ATL 30d, momentum_30d, funding rate — market context. Processed by a GRU.

The two streams are fused and passed through **temporal attention** (it weighs the window's bars differently). Removed from the ensemble after val_NLL 5.28 vs the iTransformer's 0.18: structural under-performance, not a tuning issue.

### iTransformer (`--arch itransformer`)

**Inverted** Transformer: instead of attention on timesteps, attention on **features** (each feature becomes a "token"). With 104 features, complexity O(104²)≈10,800 vs O(120²)=14,400 of the classic Transformer — better suited to tabular data because it explicitly models inter-feature correlations.

**Multi-scale** embedding: the 120-bar window is compressed into 3 views via ×1/×5/×15-bar average pooling (BAR-semantic: at 1m = 1m/5m/15m, at 1h = 1h/5h/15h), capturing both fast and slow structures without doubling parameters.

### TCN+Mamba hybrid (`--arch tcnmamba`)

Two parallel branches for local patterns (5–15 candles) and long context (120 candles):
- **TCN** (Temporal Convolutional Network): six blocks of causal convolution with growing dilations (1, 2, 4, 8, 16, 32) → receptive field **127 candles** (1 + 2·(1+2+4+8+16+32)), covers the entire input window. Captures technical figures (double tops, breakouts, consolidations). Output: global average over time.
- **Mamba** (State Space Model): hidden state evolving by discrete differential equations with **input-dependent** parameters — the model decides at each step how much to remember. Dynamic information selection over 120 candles without attention's quadratic overhead. Pure PyTorch (no external deps). **Vectorized** scan via `cumprod` + `cumsum` in 32-step chunks (AMP disabled in inference to avoid NaN on spectral_norm + Mamba edge cases, see `quantsys/model/ensemble.py`). Forward+backward speedup ~1.8× vs initial sequential scan.
- **Learned gated fusion**: `σ(W·[tcn; mamba])` learns how much weight to give local vs global per example.

It is the most expensive arch of the group (the Mamba branch's sequential scan dominates training time); timings and GPU sequencing constraints: `START.md`.

### N-HiTS (`--arch nhits`)

**Neural Hierarchical Interpolation for Time Series** (Challu et al. 2022) — implemented on 2026-05-14 to replace LSTM.

**Pure-MLP** (no recurrence, no attention, no convolution): maximal **inductive-bias diversity** vs the other 3. Pipeline:
1. **Input projection**: `Linear(104, d_model)`
2. **Three hierarchical stacks** with pooling kernel (8, 4, 1):
   - Stack 1 (k=8): long-term patterns (8× downsample, MLP, expansion to backcast)
   - Stack 2 (k=4): mid-term patterns
   - Stack 3 (k=1): very short-term patterns
3. **N-BEATS-style residual decomposition**: each stack removes from the residual the pattern it captured, leaving unexplained information for subsequent stacks
4. **Aggregation**: sum of the 3 stacks' latent forecasts → output heads

**Parallel MaxPool block (A9, inert lever — `nhits_max_pool_block`, default OFF):** stack pooling is AvgPool (low-pass); for the vol target spikes are informative (RV jumps). With the flag ON a MaxPool `NHiTSBlock` reads the same projected input (BEFORE the AvgPool stacks subtract the spikes) and adds its latent forecast; the backcast is discarded → the residual decomposition stays unchanged. OFF = zero new params, bit-compatible checkpoints.

Being pure-MLP it is the cheapest of the three to train (no recurrence and no attention to unroll).

### Probabilistic output (common to all)

Not a single number, but **a full conditional distribution**. Its form depends on `loss_type` (§7.0):

- **`quantile` (production default)** — a grid of 5 quantiles, from which `predict()` derives **μ = q(0.5)** (conditional median) and **σ = q(0.9) − q(0.1)** (interdecile range, ≈2.56 standard deviations for a Gaussian). `ν` is undefined on this branch.
- **`t_student`** — the parameters of a Student-t: mean μ (direction), σ (scale), ν (heavy tails — how likely extreme moves are).

Either way the system knows not just direction but its own uncertainty; ⚠ **μ and σ are not the same estimand across the two branches** — detail and pitfalls in §7.0.

Output in **z-score space** (target_ret normalized by the global RobustScaler, §5). Explicitly denormalized via `PipelineState.denormalize_predictions()` before the trading layer.

### CAFN — Causal Attention Flow Network (experimental probe)

**NOT an `--arch`**: it is an optional **coordination** layer upstream of the 3 models (`quantsys/model/cafn.py`, trainer `scripts/02d_cafn_joint_train.py`). It filters the feature tensor (sigmoid per-feature gate = denoising), extracts a **causal latent** `[B,T,d_latent]` via **strictly causal-masked** self-attention (t attends only to ≤t → no lookahead), and the 3 models train **simultaneously** on that latent (concatenated on the feature axis). The **causal penalty** added to the joint loss is a **regularizer** — proximity (penalizes attention on the far past → proximal causation) + stability (penalizes pattern jumps between adjacent timesteps) — **not** a do-calculus/Granger causality guarantee. **Parity-safe** integration: `latent=None` kwarg in the 3 forwards → bit-identical to legacy (BLOCKER #1 constraint). **Pre-registered, inert-by-default probe**, output isolated in `models/cafn/`; trained on the canonical 104-feature tensor (raw Deribit data is forward-collected → optional future `extra` channel only, no lookahead). Honest prior: a model-class variation → unlikely to move the OOS directional ceiling; pre-registered gate in STATUS.

### Regime-MoE — mixture-of-universes (A3, implemented-inert)

**NOT an `--arch`**: it is an alternative **output head** of the iTransformer (`model.head_type: "regime_moe"`, key **absent by default = bit-identical legacy path**). Shared backbone + **3 per-regime heads** (R0/R1/R2, semantics to be re-derived from the variances — §4) mixed by an **EXTERNAL CAUSAL soft-gate** `g(t)` = `RegimeMarkovBTC` filtered probabilities (§4) aligned via backward `merge_asof` (`quantsys/model/regime_gate.py`) — the gate is **not learned** (key anti-overfit property: no degree of freedom that could overfit val). Mixing: quantile path → gate-weighted **Vincentization** + monotone re-sort; Student-t path → **total variance law** (σ² inflated when the regime is ambiguous → regime-conditional σ calibration, A3's goal, since the short-vol edge is **not uniform across regimes**). Forward contract unchanged (`g=None` → uniform gate). **NEVER trained** (2026-07-12): QLIKE gate to pre-register + `QUANTSYS_MODELS_ROOT` sandbox before the first run. Details: `docs/MODEL_IMPROVEMENTS.md` (A3 section).

---

## 7. Training

### 7.0 The loss actually in use: pinball / quantile regression

⚠ **Read this subsection before the next three.** `loss_type` (`config/default.yaml → training`) selects **two mutually exclusive branches**, and the production default is `quantile`. The terms described under "Student-t NLL", "Asymmetric penalty" and "CRPS" **all** belong to the `t_student` branch and are **inert** on the production path, even though their config keys hold non-zero values (see the table at the end of this subsection).

**Definition.** For a quantile level τ ∈ (0,1) and an error `e = y − q̂_τ`, the **pinball loss** (or *check loss*, Koenker–Bassett 1978) is

```
L_τ(e) = τ · e          if e ≥ 0     (under-prediction: y lies above the forecast)
         (τ − 1) · e    if e < 0     (over-prediction: y lies below the forecast)
```

Both branches are non-negative (in the second, `e < 0` and `τ − 1 < 0`), but the slopes are **τ** and **1 − τ**: an asymmetrically weighted absolute error. At τ = 0.5 the slopes coincide and `L = ½·|e|`, i.e. rescaled MAE. At τ = 0.9 under-predicting costs 9× more than over-predicting, so the optimum moves up until 90% of the mass lies below the forecast.

**The property that justifies it.** The population minimizer of `E[L_τ(Y − q)]` is **exactly the τ-th quantile of Y**. It is not a surrogate: it is to quantiles what MSE is to the conditional mean and MAE to the median. Summing over several τ estimates the conditional distribution **without postulating its shape** — the substantive difference from the Student-t NLL, which estimates three parameters of an *assumed* family. The price: no density, just a grid of quantiles, and nothing in the loss prevents them from **crossing**.

**Implementation** (`quantile_loss` in `quantsys/model/__init__.py`). Five levels, `QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]`; the head emits 5 values and the loss averages over the batch axis **and** the quantile axis. Optional per-sample weights ∝ |y| (`sample_weight_alpha`, **0 in production** → disabled). **Non-crossing is not enforced in the loss**: it is fixed *post-hoc* by a `sort()` along the quantile axis inside `predict()` — the standard pragmatic choice, but a constraint training never sees.

**The two quantities the rest of the system consumes** are derived from that grid, and their definitions do **not** match the same-named objects of the Student-t branch:

- **μ = q(0.5)** — the conditional **median**, not the conditional mean.
- **σ = q(0.9) − q(0.1)** — an **interdecile range**, NOT a standard deviation: for a Gaussian that range equals ≈ **2.56 σ**. Reading `sigma` in the code as a standard deviation is off by a factor of ~2.5.

⚠ `predict()` indexes **positionally** (`[:, 2]` for μ, `[:, 4] − [:, 0]` for σ): changing `QUANTILES` without updating those positions silently corrupts μ/σ. The contract is under a golden test (`tests/test_quantile_loss.py`).

**Effective production objective.** With `loss_type: quantile` and `use_multitask: true` (default) the optimized loss is

```
L  =  multitask_alpha · pinball(y, q̂)  +  (1 − multitask_alpha) · CE(soft_label, dir_logits)
   =  0.7 · pinball                     +  0.3 · CE
```

where CE is the 3-class directional head with soft labels ∝ tanh(|y|/threshold). **Nothing else.**

**Which terms are active on which branch** (single source of truth: the branching in `scripts/02_train.py`, epoch function):

| Term | Config key (value) | `quantile` branch (**production**) | `t_student` branch |
|---|---|---|---|
| Pinball | `QUANTILES` (5 levels) | ✅ active | — |
| Student-t NLL | — | — | ✅ active |
| Asymmetric penalty | `asymmetry_alpha: 2.0` | ❌ **inert** | ✅ active |
| CRPS | `crps_weight: 0.1` | ❌ **inert** | ✅ active |
| Direction-Value | `dv_lambda: 0.3` | ❌ **inert** (explicit guard) | ✅ active |
| Multitask CE | `multitask_alpha: 0.7` | ✅ active | ✅ active |
| Weights ∝ \|y\| | `sample_weight_alpha: 0.0` | disabled | disabled |

⚠ **Tuning trap.** The three keys marked inert hold non-zero values in `config/default.yaml` and *look* like live levers: changing them on the production path changes **nothing**, and an experiment built on them would return "no effect" for implementation reasons, not scientific ones. `02_train.py` emits an **explicit warning** at start-up when the quantile branch is active and any of those keys is non-zero (logging only, numeric path bit-invariant).

### Student-t NLL loss (`loss_type: t_student` branch)

Penalizes the model when the predicted distribution is far from the observed value. **Student-t** instead of Gaussian: financial returns have heavier tails (crashes and rallies happen more often than a Gaussian would predict).

### Asymmetric penalty (`t_student` branch)

Extra penalty when the model gets the direction wrong (says "up" but it goes down). Sign errors cost more than magnitude errors: a position in the wrong direction loses money, while underestimating the magnitude only hurts returns.

### CRPS (`t_student` branch)

Continuous Ranked Probability Score — auxiliary **calibration** metric: if the model says "80% probability", it should be right ~80% of the time. A model that always says "95%" but is right 60% is dangerous due to overconfidence in trading.

### Walk-forward validation

No simple train/test split: the model is trained on a historical window, tested on the immediately following period (never seen), then the window slides forward. Simulates the real-world deployment and avoids look-ahead bias. Exact mechanics and evaluation metrics are in **Part III**.

### Knowledge Distillation (alternative to homogeneous ensemble 5× same arch)

**Phase 2a — candidates.** The archs listed in `distillation.archs` are trained normally with `n_ensemble=1`. One config line changes the composition. *(Commands: `START.md`.)*

**Phase 2b — TARGET-AWARE multi-teacher scoring.** Each arch `a` is evaluated at its best epoch; metrics are min-max normalized **across archs** (`loss` inverted, lower-is-better) and combined with `target_type`-dependent weights (`teacher_score_weights`, single source of truth in `distillation.py`):

```
score_a = w_vl·(1 − norm(val_loss_a)) + w_sp·norm(spearman_a) + w_da·norm(dir_acc_a)
π_a     = exp(T · score_a) / Σ_b exp(T · score_b),        T = 2

target `ret`     (directional) : (w_vl, w_sp, w_da) = (0.40, 0.35, 0.25)
target `log_rv`  (volatility)  : (w_vl, w_sp, w_da) = (0.65, 0.35, 0.00)
```

⚠ Temperature note: `T=2` **multiplies** the score inside the exponential, so it **sharpens** the weight distribution (it does not smooth it, as dividing by T would). ⚠ Econometric note on the `dir_acc` term: on the **variance** target directional accuracy measures the sign of variance-vs-median, which is **not a tradable signal** (the straddle is direction-neutral) → the weight is zeroed and rebalanced onto the **EVEN** moment (`val_loss`/QLIKE), the only one that generalizes OOS. All archs contribute as teachers with weight `π_a` — there is no single teacher. Best-epoch validation metrics must be persisted to `config.json` (`best_val_loss`/`best_spearman`/`best_da`): without them the blend silently degrades to uniform weights.

**Phase 2c — student with head-transfer + distillation.** Each model is retrained as a *student*: (a) output-head weights (μ, σ, ν) are copied from the best teacher — calibrated rather than random start; (b) teacher-weighted soft labels enter the `TensorDataset` (shuffle-safe: each batch holds real data and teacher predictions for **the same samples**); (c) the loss is a mixture

```
L = (1 − α)·NLL(y, ŷ) + α·L_distill
L_distill = 0.5·MSE(μ_s, μ_t)/Var(μ_t) + 0.3·MSE(σ_s, σ_t)/Var(σ_t) + 0.2·MSE(ν_s, ν_t)/Var(ν_t)
```

The **scale-normalization** by the variance of the corresponding teacher output is the non-obvious part: μ ~ 1e-5 and ν ~ 5 differ by orders of magnitude, and without it ν would dominate the gradient while μ became irrelevant. `α` is scheduled **linearly 0.6 → 0.1 over 20 epochs** (CLI default; the training function's own default, with no schedule, is 0.3): the distillation weight decays as the student sees real data. Students converge in ~60% of the normal epochs.

**Heterogeneous ensemble (inference)**: the N architectures predict together. Errors tend to be uncorrelated because they capture different patterns (N-HiTS hierarchical multi-scale, TCNMamba local + long context, iTransformer inter-feature correlations). Combination = **weighted mean** with `DEFAULT_ARCH_WEIGHTS` (`ensemble.py`):
- `mu_ens = Σ w_i · mu_i` (reduces error variance)
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)` (law of total variance: accounts both for average model uncertainty and for disagreement between point predictions)

The ensemble returns (μ, σ, ν) directly in natural space, no intermediate numerical conversions.

**⚠ How much ensembling helps depends on `target_type` (measured, not assumed).** Variance reduction is ∝ to the **decorrelation of cross-arch errors**. On the **directional** line the cross-arch error is ρ ≈ **0.995** → variance reduction ≈ 0, ensembling is **mathematically useless** (the directional edge is regime-conditioned, not fixable by combining models). On the **vol** line (`log_rv`) the 2026-06-22 STEP 0 kill-check (`scripts/vol/step0_xarch_corr.py`, val split) instead measures ρ_err ≈ **0.83** (iTrans|N-HiTS 0.78, iTrans|TCN-Mamba 0.82, N-HiTS|TCN-Mamba 0.89) → exploitable diversity, so ensemble/distill has **potential headroom** (KILL≥0.99 gate cleared → PROCEED). Consistent with the even-moment thesis: vol is predictable OOS **and** the archs disagree.

⚠ **Why PROCEED does not imply a gain.** Decorrelation is *necessary* but not sufficient: the teacher-selection criterion is `val_loss`, while the vol line's judge is **QLIKE**, and the two are not aligned (on the measured run TCN-Mamba has the best `val_nll`, iTransformer the best QLIKE). A scoring that elected the wrong teacher would distill the incumbent *downward*: the prior on the vol distill is therefore **low**, and it stays a gated experiment with a pre-registered gate (status and thresholds in `STATUS.md`).

---

# Part III — Evaluation

## 7bis. Purged walk-forward — mechanics

Walk-forward validation uses **purged k-fold with anti-leakage embargo** (`walk_forward_folds` in `quantsys/features/__init__.py`). Two structural details to know:
- **`n_folds=6` declared → 5 effective folds.** Fold 0 is skipped by construction: its `train_end = fold_size − embargo` is structurally `< fold_size`, so there is no valid training window. To get **K** effective folds you need `n_folds=K+1`. Kept at 6 for consistency with fix #4 of the 2026-05-17 plan (documented behavior, not a bug).
- **Embargo `embargo_steps=168` bars** (= 1 week at 1h). It must be `≥ window_size + forecast_horizon = 120 + 30 = 150` to prevent a test window from sharing bars with a training window's target (at 1m it was 1500 ≈ 25h). The embargo *purges* the observations straddling the train/test boundary whose label spills into the other segment.
- **`--no-retrain` is in-sample contaminated** on early folds (it loads the final `best_model`, already exposed to that data): for a clean OOS evaluation use the `val`+`test` split or temporal sub-periods, not the `--no-retrain` WF.

## 7ter. Evaluation metrics

The metric depends on the line:
- **Vol line (`log_rv`, production).** **QLIKE** (`L = RV/RV̂ − log(RV/RV̂) − 1`, scale-robust, penalizes under-estimating variance more than over-estimating) is the primary judge, shared via `quantsys/model/vol_metrics.py` (`qlike_from_z`). Comparison baselines: **HAR-RV** (Corsi 2009, regression on daily/weekly/monthly RV) and the naive (trailing RV). PASS gate 2026-06-10: NN/HAR ≤ 0.95 on test → achieved 0.257/0.368 (NN beats HAR by ~30%; naive 0.807). **Spearman** (rank-IC) as the secondary ordering metric.
- **Directional line (`ret`, legacy).** **Spearman rank-IC** over K non-overlapping sub-periods (the rolling IC at window=50 was inflated ~30× by autocorrelation, fix 2026-06-02), **directional accuracy** (sign), and downstream the backtest (Sharpe, profit factor, WHR, n_trades). ⚠ See §7quater: on this line in-sample metrics **anti-correlate** with the backtest.

**Inference on a two-forecast comparison — why HAC is required.** A QLIKE ratio says *how much*, not *whether it is distinguishable from noise*. The canonical test is **Diebold-Mariano** on the loss differential `d_t = L_A,t − L_B,t`: `DM = d̄ / √(V̂/n)`. The delicate part is `V̂`: the target sums **h bars**, so samples closer than h share realizations and `d_t` is **strongly autocorrelated**. Under an iid variance the standard error is understated by roughly `√h` and the p-value becomes fictitious. Hence a **HAC (Newey-West, Bartlett kernel)** variance with lag `q = h−1`, plus the **Harvey-Leybourne-Newbold** small-sample correction and a Student-t with `n−1` degrees of freedom. The sample size governing the uncertainty is `n_eff ≈ n/h`, not `n`: at h=30 and n≈6.5k that is **~216 independent observations**. Implementation: `diebold_mariano()` in `quantsys/model/vol_metrics.py`, surfaced by the judge as a **descriptive** output — gate thresholds remain the pre-registered QLIKE ratios, never the p-value (adding a condition once results are visible would be goalpost-moving, §12.1).

## 7quater. Distribution shift val→test

**Structural empirical fact (measured on the 1m dataset, re-confirmed at 1h on the directional line).** On the **directional** line, in-sample metrics (`val_nll`, walk-forward Spearman/WHR) **anti-correlate** with the OOS backtest: optimizing rules guided by in-sample metrics systematically worsens PnL. Operational consequences are codified in the `EXPERIMENTAL PROTOCOL` (val-first, pre-registered gates, inert flags): every trading lever was validated on `QUANTSYS_BACKTEST_SPLIT=val` and **FAILED OOS** (threshold/rank entry, σ calibration, cadence, continuous exposure: the inert-flag corpus is listed in §12.5, the numbers in `STATUS.md`). The real directional edge exists **only regime-conditioned** (R0 Quiet: Spearman +0.13÷0.19 stable OOS) but it is a **rank** edge, not captured by a |μ| threshold entry.

⚠ The anti-correlation is **specific to the directional target**: on vol (`log_rv`) val and test are coherent, which is why the vol-S PASS counts as an OOS edge and not as a test-split overfit. General statement of the fact: **Part VI, point 2**. Cross-arch error diversity and the ensembling rationale: §7.

---

# Part IV — Inference

*Inference composes the models' forward (μ/σ/ν in z-score) → z→raw denormalization (Part I, invariant) → optional Monte Carlo (§8). The trading layer consuming these quantities is in Part V (directional, legacy).*

## 8. Monte Carlo

For every new candle, the system generates **2000 alternative price scenarios** (`mc.n_paths` in config) over the next 30 bars (= 30 hours at the current 1h timeframe), using the ensemble's predictions as a guide and adding stochastic variability calibrated on current volatility.

Volatility is estimated with a **GJR-GARCH(1,1)** model: the GJR variant adds an asymmetry term that amplifies the volatility update in response to negative shocks (leverage effect). **Parameters RE-ESTIMATED on 1h returns on 2026-07-15** (`config/default.yaml → montecarlo`: `ω=1.026e-6, α=0.1011, γ=0.0052, β=0.8732`; Gaussian QMLE + variance targeting on 65,450 bars 2019→2026-06-22, script `scripts/vol/estimate_gjr_1h.py`, report `results/vols/gjr_params_1h.json`). Persistence 0.977 (30h half-life), unconditional σ 0.67%/bar (~62% annualized). Empirical note: at hourly frequency γ≈0.005 — the leverage effect is near-zero on this sample; the asymmetry emerges at daily frequencies. The per-bar σ anti-explosion cap is now parametric (`gjr_sigma_cap`: 0.13 at 1h, 1m-era 0.01 in `config/interval/1m.yaml`); the full 1m-era values live in the 1m overlay for rollback.

The result is an empirical estimate of the price distribution at horizon h: from it one reads level-crossing probabilities and tail quantiles (e.g. the 5th percentile as worst case). The MC is **not on the backtest critical path**: it is diagnostics and sizing support, not a signal generator.

---

# Part V — Directional trading layer (legacy, KILLED OOS)

## 9. Signal generation

> **Line note.** §9–§11 (Part V) describe the **DIRECTIONAL trading layer** (`target_type: ret`) — code intact as a baseline but **KILLED OOS** (see orientation at the top). The production (vol/options) line does not go through this path: from the RV-vs-IV signal (`edge = log(rv_pred / var_iv)`) the ATM Deribit straddle is run by `scripts/04b_vol_paper.py` (direction-neutral, hold-to-expiry), not by `SignalGenerator`/`RiskManager`.

The operating signal (BUY / SELL / HOLD) combines multiple elements:

**Conviction score** — direction predicted by the ensemble, magnitude of the expected move, prediction uncertainty. High conviction requires (a) clear direction, (b) expected move exceeding commissions (0.1% per side), (c) low uncertainty.

**Quality filters**:
- Expected return > minimum threshold (to cover commissions)
- Predicted volatility not too high (chaotic market → skip)
- BTC regime compatible (tighter entry thresholds in the high-σ² regime, full Kelly in the intermediate one). ⚠ The rule↔regime-index mapping is hardcoded and **STALE** until re-derived from the current run's variances (§4, label-switching corollary).

---

## 10. Risk management

**Kelly sizing**: size proportional to the estimated statistical edge and inversely proportional to variance. Strong signal + calm market → larger risk; weak signal + high vol → smaller risk. Max risk per trade: 1% of capital.

**Dynamic ATR stop loss**: not a fixed %, ATR-based. Volatile market → wider stop (no stop-out by noise). Calm market → tighter stop (limits the loss).

**Trailing stop**: once in profit, the stop rises with price protecting gains. Trailing distance also proportional to current ATR.

**Circuit breaker**: if drawdown exceeds **15%** of capital (`risk.max_drawdown_stop` in config), the system stops opening new positions. Last-resort protection against prolonged losing streaks (possible structural market change not trained on). DD computed **mark-to-market every candle** (cash + size_usd + unrealized_pnl, updated in `update_trailing`): in live it fires even if a single position has large unrealized losses, without waiting for close. Auto-recovery when DD goes back below 70% of the threshold (e.g. <10.5% with 15% threshold).

**Greeks-aware risk layer (A7, skeleton — NOT wired).** The risk manager above is delta-one: for the options book there is `quantsys/trading/greeks_risk.py` — pre-trade **net-vega** (and net-delta) caps with scaling to the cap edge (exposure-reducing orders always pass), a **mark-to-market vega-loss** circuit breaker (same trip/recovery hysteresis as the delta-one CB) and a **Deribit inverse margin simulation** (IM/MM for short options + perp; per-leg standard margin, declared conservative approximation — no portfolio margin). It enters the critical path only when sizing moves from fixed 1 contract to Kelly-on-edge (post-gate v2).

---

## 11. Live execution

The `LiveEngine` (`scripts/04_live_signals.py`) connects to Binance over WebSocket and, on every **closed** candle, runs the same chain as the backtest: features with the training normalization → last-120-bar sequence → forward → denormalization → Monte Carlo → conviction score → optional position open/close (**paper trading**, no real orders) → state and signal persistence. The macro snapshot is refreshed in the background hourly so it never blocks the feed.

### ✅ Current status: BLOCKER #1 RESOLVED (2026-06-05) — live↔training parity closed

The live production path is now aligned to training **by-design** (single source of truth `FeatureBuilder`):
`LiveCandleBuffer`(50k raw-OHLCV ring) → `FeatureAssembler` → `FeatureBuilder.build(fit=False, normalize=True)` (**104 canonical features**, same order, global scaler from `PipelineState`) → `LiveEngine._deterministic_predict` (deterministic forward + `denormalize_predictions`) → `SignalGenerator`. The production `EnsembleModel` lacks `predict_with_uncertainty`, so the MC-dropout branch never fires live and the forward is bit-identical to the backtest.

**Validation (both go/no-go gates green):** Gate 1 FEATURE parity (`tests/test_live_training_parity.py` + `scripts/99_replay_live_vs_training.py`) → max|Δ|=0; Gate 2 SIGNAL parity → Δμ=Δσ=0, identical side. The old `LiveFeatureBuffer` (39 features) is deprecated, kept only as an ATR/sanity helper.

⚠ Closed parity means live signals **faithfully reflect the backtest** — and the directional backtest is **negative OOS** (threshold/rank edge exhausted). The directional `LiveEngine` is **not running**: forward production is the vol/options line (`04b`). This path remains a verified baseline and reusable infrastructure, not a strategy to be started.

### 24/7 operational robustness

Three of these safety nets are **correctness invariants**, not operational details:
- **Lookback-buffer sizing**: `max(window_size + 60, max_rolling_window + 60) = 260` candles. It must dominate the **longest** rolling window, otherwise that feature is silently constant in live and parity with training breaks with no error (real case: buffer 180 → `price_vs_ma200m` always zero).
- **Forming vs closed candle separation**: only klines with `k.x == True` enter the buffer; partial ones sit in `_pending_candle` and are dropped on WS reconnect. This is a **causality** condition: an unclosed candle carries incomplete, non-repeatable information.
- **`forecast_horizon` and `interval` guards**: `LiveEngine.__init__` raises `RuntimeError` if `cfg.data.forecast_horizon != PipelineState.forecast_horizon` or if the config interval differs from `PipelineState.interval_minutes` — it prevents serving a model trained for a different horizon or timeframe. Same pattern in the backtest (§5).
- **Atomic checkpoints**: `EarlyStopping` writes to `.tmp` + `os.replace()` (cross-platform atomic rename) — no truncated checkpoint if the process dies mid-save.

The remaining ones (funding-refresh thread safety under `threading.Lock()`, Windows file-lock-tolerant log rotation) are operational: details in `START.md`.

---

# Part VI — Cross-cutting scientific reading

Net of the chronicle, the closed lines of work established three facts that are **theory**, not anecdote:
1. **EVEN vs ODD moments.** On this asset the **even** moments (variance, RV, QLIKE) generalize OOS; the **odd** moments (direction sign, signed semivariance asymmetry) are unpredictable **for the NN and for the econometric baselines alike** — HAR-RS on the semivariance ratio does worse than the constant. This is not a model-capacity limit: it is a property of the process.
2. **The val→test distribution shift is a property of the TARGET, not of the pipeline.** On the directional target, in-sample metrics (`val_nll`, walk-forward Spearman/WHR) **anti-correlate** with the OOS backtest; on vol (`log_rv`) val and test are **coherent** (val QLIKE predicts test QLIKE). Same pipeline, same scaler, same split scheme: only the target changes.
3. **The 1m directional wall was one of MAGNITUDE, not of sign** (~1.5 bps effect vs ~26 bps round-trip cost). At 1h the cost wall falls (cost/σ from ~1.9–3.3× to ~0.25–0.42×, raw |μ| ≈ 43 bps) **yet no directional skill emerges OOS**: the failure was not attributable to friction, and removing it produces no edge.

Operational corollary: weigh every new hypothesis with this prior and pre-register every gate **before** running. The experiment **chronicle** (dates, gates, run-by-run numbers) does not live here: `CHANGELOG.md`, `STATUS.md` (current period + open gates), `docs/STATUS_ARCHIVE_2026H1.md` (pre-2026-07-08 history, read-only).

---

## 12. Experimental protocol and negative-results corpus

This section is the **synthesis of outcomes**, not the chronicle: every closed line of work is reported with its gate metric, pre-declared threshold and realized number. It is deliberately skewed towards the failures — they carry the highest information value in this project, because they delimit where the edge is NOT.

### 12.1 Protocol — 5 mandatory steps

1. **Pre-register the gate BEFORE running**: metrics, thresholds and minimum sample size, written and committed. Forbids goalpost-moving once results are visible.
2. **Val-first**: validate on `val`; the test split is touched **exactly once**, after the val gate passes.
3. **Every experimental lever is an env-flag, inert by default**: zero impact on the production path, reversible, documented with its **outcome**.
4. **A negative outcome gets written down anyway.** Documented kills are the vaccine against involuntary re-testing.
5. **After every sweep the production state is restored**: env cleared, sandboxes deleted, clean run.

**Ex-ante count condition.** If a gate carries a minimum-sample condition (per regime, per expiry), verify it **before** spending GPU: if it is model-independent the outcome is predetermined, and the experiment returns "no conclusion" instead of a result. Two gates were closed this way without an hour of training.

**Comparisons only at equal scaler.** A candidate must be judged against a **baseline retrained on the same dataset and scaler**, never against the incumbent model: scaler distribution shift produces QLIKE differences of order 4-5%, i.e. larger than the effect being measured.

### 12.2 Active line — 1h volatility

**Which baseline backs which statement — reconciliation panel.** This section names
**four** denominators (naive, HAR-RV, HAR-CJ, HAR-C) because the baseline was progressively
strengthened by three successive gates. Two roles must **never be merged**: the denominator of
the **pre-registered gate**, frozen on 2026-06-10, and the denominator of the **published
claim**, which has changed twice since. They are different statements about different
measurements, and both are true.

| statement | denominator | numbers published below | role today |
|---|---|---|---|
| **Pre-registered gate** (2026-06-10) — `QLIKE_NN ≤ 0.95·QLIKE_HAR` **and** `< QLIKE_naive` | **HAR-RV** | test −30% (0.257 vs 0.368), naive 0.807 | **passed, and frozen**: it remains valid with respect to that baseline and is **not re-run** — swapping its denominator after the fact would be goalpost-moving in reverse |
| **Significance of the edge** (DM+HAC, 2026-07-26) | **HAR-RV** | val −26.6% `p = 7.3·10⁻⁵` · test −36.1% `p = 1.7·10⁻⁶` → band −27% to −36% | an inference measurement, **not** a gate |
| **Robustness check C2** (2026-07-30) | **HAR-CJ** | val −22.6% · test −31.6%, `p ≤ 4.3·10⁻⁴` | **intermediate** baseline, superseded by C3 — ⚠ this is where the `p ≤ 4.3·10⁻⁴` quoted elsewhere comes from, so it is **not** measured against HAR-C |
| **Published claim** (since 2026-07-31, gate C3) | **HAR-C** | **−22.42% to −31.65%** | **current** — this is the reference baseline |

⚠ **The band's endpoints are `val` and `test`, not a confidence interval:** −22.42% is val,
−31.65% is test, and the decimals identify the model/npz/config triple (PRECISION paragraph at
the end of this section). Reading them as uncertainty is the error this panel exists to prevent.

⚠ **The judge prints the role next to every number** (`scripts/vol/dev_vols_qlike.py`): the
additional baseline blocks stay outside `metrics`/`gate`, so the gate keeps being computed
against HAR-RV while the claim is read against HAR-C. Conflating them is exactly what the
three-baseline stdout panel makes impossible.

**Validated PASS.** ⚠ **Reference baseline changed on 2026-07-30 (gate C2) and respecified on 2026-07-31 (gate C3): the current claim is against HAR-C, band −22.42% to −31.65%** (re-expressed on 2026-08-05 at the canonical artifact's precision — **same numbers, not a new measurement**: until then the band read −23% to −32%, with the lower endpoint inherited from HAR-CJ and never realigned to HAR-C; see the PRECISION paragraph at the end of this section) — see the dedicated paragraphs below, **including the one on the numerator's PROVENANCE**, which since 2026-08-04 binds it to a verifiable artifact (`models/canonical_1h_vols/`, gate R1). What follows is the PASS **as registered**, against HAR-RV, and remains valid with respect to that baseline. With the `log_rv` target the NN forecast beat HAR-RV by **30% in test QLIKE** (0.257 vs 0.368; persistence naive 0.807), with coherent val→test. Judge: `scripts/vol/dev_vols_qlike.py` (pre-registered gate: `QLIKE_NN ≤ 0.95·QLIKE_HAR` **and** `< QLIKE_naive`). Model: 5-member iTransformer. ⚠ No trading backtest on the vol models: `03_backtest.py` is meaningless on a variance target.

**Comparison significance (Diebold-Mariano, 2026-07-26).** The QLIKE ratio is a point estimate: inference requires a **HAC** variance, since the target sums h=30 bars and windows overlap (§7ter). On a model/scaler pair retrained on the same dataset — 5 seeds, no test contact during training — the edge over HAR-RV is: **val −26.6%** (0.26206 vs 0.35698), DM = −3.97, **p = 7.3·10⁻⁵**; **test −36.1%** (0.23631 vs 0.36998), DM = −4.79, **p = 1.7·10⁻⁶**; HAC lag 29, n ≈ 6.5k, **n_eff ≈ 216**. The edge **survives** the overlap correction. The NN beats HAR in **every regime**, stress included and validated on test (r0 0.570 · r1 0.542 · r2 0.695 as NN/HAR ratios). ⚠ The −36.1% and the historical −30.2% are **not the same measurement**: different test windows (the dataset was extended) and a retrained model — against HAR-RV the band is **−27% to −36% depending on split and vintage**, at p ≤ 1.7·10⁻⁶. ⚠⚠ **These numbers are measured against HAR-RV, which since 2026-07-30 is NO LONGER the reference baseline:** the C2 gate showed HAR-CJ to be a significantly stronger baseline and the **published band is now −22.42% to −31.65%** (against HAR-C, after C3; written −23% to −32% until 2026-08-05) — see the next paragraph.

**Optimized estimand vs judged estimand (open note, the claim is NOT affected).** QLIKE is a proper loss for variance forecasts: its expectation is minimized by the conditional **mean** of RV. The production model instead optimizes τ=0.5, i.e. the conditional **median** of log-RV (§7.0), and the judge inverts with `exp()` without a Jensen correction (`qlike_from_z` in `quantsys/model/vol_metrics.py`): since `exp` is monotone this yields the median of RV, which under skewness sits systematically **below** the mean. The same omission applies to the baseline — `har_fold_qlike` exponentiates the OLS forecast without smearing — so **the comparison stays symmetric and the PASS holds**. Two fine asymmetries remain worth noting: (i) HAR's OLS estimates the conditional **mean** of log-RV while the NN estimates its **median**, and they coincide only if log residuals are symmetric; (ii) a **smearing** correction (Duan 1983) applied to **both** sides is an untested lever — as such it requires a pre-registration (§12.1) before any run, and must **not** be retrofitted to an already published claim. **Update 2026-07-28: the pre-registration has been written and is OPEN (gate C1 in `STATUS.md`), NEVER run** — no corrected QLIKE has been computed on any split. It also declares the expected direction ex-ante: since Duan's factor grows with log-residual dispersion and the NN has lower log-MSE than the baseline, **HAR is expected to gain more than the NN**, i.e. a **reduction** of the measured edge. Adoption is conditional on a **materiality** threshold (ratio shift ≥0.02), so that a published claim is never rewritten on noise. **Update 2026-07-29: the correction is now IMPLEMENTED as an inert, never-executed lever.** `duan_smearing` (non-parametric estimator ŝ = mean of exp(ε) over log residuals) lives in `quantsys/model/vol_metrics.py`; `qlike_from_z` and `har_fold_qlike` accept a `smear` parameter with **default 1.0 = identity**; the `dev_vols_qlike.py` judge enables it only under `QUANTSYS_QLIKE_SMEARING=1` and estimates ŝ **exclusively on train residuals** (for the NN, via an inference pass over the train split). With the flag off the report is **bit-identical** to the historical one, verified by an A/B against the pre-patch judge. `mse_log` is deliberately left **uncorrected**: on the log scale squared loss is minimized by the median, which is what the model estimates — the correction concerns only the level estimand judged by QLIKE. **Update 2026-07-30 — gate C1 RUN on val (one-shot, 5 seeds retrained in a sandbox): condition ① FAILED and the correction is NOT adopted.** Numbers, reported whatever their sign as the pre-registration required: ŝ_NN = 1.6186, ŝ_HAR = 1.5836, ŝ_naive = 1.9145 (all from train residuals only); the NN's QLIKE **worsens** 0.26143 → 0.40257, while HAR improves 0.35698 → 0.34075 and naive 0.71288 → 0.54442. The NN/HAR ratio would move from 0.7323 to 1.1814 (Δ = +0.4491, far above the 0.02 materiality threshold), but the gate is AND and ① requires `QLIKE_smear ≤ QLIKE_raw` on **both** sides: one side worsens ⇒ the correction is not a uniform specification improvement ⇒ **not adopted, judge unchanged, published band −27% to −36% unchanged**. Two readings to keep. (i) The **expected direction was falsified**: ŝ_NN < ŝ_HAR was pre-declared (the NN has lower log-MSE), yet ŝ_NN > ŝ_HAR — because ŝ = E[exp(ε)] is driven by the **right tail** of the log residuals, not by their overall dispersion, and the NN has a heavier right tail despite a lower log-MSE (0.6021 vs 0.6944). (ii) The outcome **does not show** that the NN's edge is immune to the median-vs-mean mismatch: it shows that *this* global correction does not fix it, since it inflates by a constant factor a model whose conditional residual dispersion is not constant, and QLIKE charges for the resulting overprediction. The note therefore stays **tested and not adopted**, not "resolved"; the `QUANTSYS_QLIKE_SMEARING` flag stays inert by default as executable documentation of the failure.

**The comparison baseline — open check (C2, 2026-07-30 pre-reg).** The published band is measured against **plain HAR-RV** (Corsi 2009). The first legitimate objection is whether that baseline was a *strong* one: on an asset with frequent jumps the standard variant is **HAR-CJ** (Andersen-Bollerslev-Diebold 2007), splitting realized variance into continuous and jump parts via **bipower variation** `BV = (π/2)·Σ|r_i||r_{i−1}|`, with `J = max(RV−BV, 0)` and `C = RV − J`. **OUTCOME 2026-07-30 — PASS 4/4 on val and on test: HAR-CJ is adopted as the reference baseline and the band is rewritten to −23% to −32%.** HAR-CJ beats HAR-RV on both splits (val 0.33788 vs 0.35698, −5.35%; test 0.34572 vs 0.36998, −6.56%) **significantly** (DM with HAC variance: p ≈ 2·10⁻⁴ on val, 1.5·10⁻⁵ on test) → **the pre-declared counter-argument did not materialize**: at hourly resolution the HAR-CJ specification *forecasts better* (on the mechanism, see the qualification below). Against this stronger baseline the NN keeps an edge of **22.6% on val and 31.6% on test** (ratios 0.7737 and 0.6837), at p ≤ 4.3·10⁻⁴. Reading: between **one eighth and one sixth** of the historically published edge was attributable to baseline weakness — 4.14 points out of a 26.77 edge on val (**15.5%**) and 4.48 out of 36.11 on test (**12.4%**); the rest survives. The Δratio is **nearly identical across splits**, so the effect is structural rather than idiosyncratic — the main reason the result is credible. The claim comes out **more defensible**: the first objection a competent reader raises is now closed with a number. ⚠ **Documented, non-invalidating diagnostic:** the OLS jump coefficients are numerically extreme (`J[h,w,m] = +44.164, −1.227, −89.441` against a perfectly normal `C[h,w,m] = 0.241, 0.423, 0.228`), the signature of **multicollinearity among tiny-scale regressors** — at 1h jumps are small and `log(1+J) ≈ J`. The result holds because the regression is fitted on train data only and improves QLIKE **out of sample** on both splits; a respecification (rescaling, regularization) would require a **new pre-registration**, excluded ex-ante by this one. **⚠ Qualification from the end-of-day audit (2026-07-30):** coefficient stability was measured by refitting the OLS on each half of the train set, and the jump terms are **not identified** — `J[h,w,m]` moves from `(40.3, 11.4, -179.1)` to `(105.0, -179.6, -116.8)`, with the weekly coefficient **flipping sign** and changing magnitude 16-fold; the continuous coefficients stay reasonably stable, `(0.271, 0.373, 0.259)` vs `(0.207, 0.462, 0.164)`. Design condition number: **5.4e+04** against HAR-RV's 1.2e+02, and the jump regressors have ~1000x smaller standard deviation than the continuous components (0.0013 vs 1.14). **Consequence for the interpretation, not for the outcome:** what is demonstrated is that the HAR-CJ *specification* forecasts better out of sample on two disjoint splits, NOT that the *jump* component carries information. The likelier explanation is that the gain comes from using **C instead of RV** as the regressor — C = min(RV, BV) is a jump-robust measure, i.e. a cleaner signal — and the continuous coefficients being close to HAR-RV's supports that. The experiment separating the two explanations is **HAR with C only, no J**: that is an estimator variant, **excluded ex-ante by C2's ②**, so it requires a NEW pre-registration and was not run. Operational caveat: a baseline with unidentified parameters is **fragile** to refitting on different windows — relevant if HAR-CJ is used outside this comparison. The gate followed the same discipline as C1 (one estimator, materiality threshold, explicit anti-goalpost constraint). The design is paired in the strict sense — the NN side is **constant** across the two ratios, only the denominator changes — so there is no seed-draw asymmetry and no cross-scaler risk. Implementation: `build_har_cj_frame` + `har_cj_fold_qlike` in `quantsys/model/vol_metrics.py`, tests in `tests/test_har_cj_baseline.py`. Introduced as the `QUANTSYS_HAR_CJ=1` lever, inert by default (inertia verified by A/B against the pre-patch judge); **the flag was REMOVED on 2026-07-31** when C3 closed and the baseline is now always computed — see the next paragraph. ⚠ **Not to be confused with A4**, which uses the same decomposition as **input features** (`features.har_cj`, `tests/test_har_cj.py`): same econometric object, two opposite roles — A4 makes the model stronger and sits inside the class closed in §12.4, C2 makes the **test** harder and sits outside it. Caveat declared ex-ante: at **hourly** sampling bipower variation is a noisy estimator (the decomposition performs best at 5 minutes), so it is entirely plausible that HAR-CJ does not improve at all — hence gate condition ①, which in that case returns "no conclusion" rather than a FAIL of the claim.

**Attribution of the HAR-CJ gain — CLOSED (C3, pre-registered and run 2026-07-31).** The audit qualification above left one question open: HAR-CJ forecasts better, but *why*? The two explanations were regressor **substitution** (`C = min(RV,BV)` is jump-robust, i.e. a less noisy signal than `RV`) and genuine **decomposition** (jumps carry their own information). A three-way comparison — HAR-RV vs **HAR-C** (continuous components only, 3 regressors) vs HAR-CJ (6 regressors) on the same samples — separates them. **OUTCOME: SUBSTITUTION explains everything; jumps add nothing measurable. The interpretation "at 1h the decomposition carries information" is FALSIFIED.** Numbers: HAR-C = **0.33698** on val and **0.34584** on test, against HAR-RV 0.35698/0.36998 and HAR-CJ 0.33788/0.34572. Two paired DM tests (HAC, h=30, α=0.01 with familywise Bonferroni 0.02): **Test A** `DM(C, RV)` = −3.735 (p = 1.9·10⁻⁴) on val and −4.140 (p = 3.5·10⁻⁵) on test → HAR-C beats HAR-RV **significantly on both**; **Test B** `DM(CJ, C)` = +1.155 (p = 0.25) and −0.100 (p = 0.92) → the jump terms **add nothing**, and on test the statistic is practically **zero** with n = 6486. The share of the gain attributable to substitution alone, `φ = (Q_RV − Q_C)/(Q_RV − Q_CJ)`, is **+1.047 on val and +0.995 on test**: 100%. Test B's sign **flips between splits** with magnitudes of 0.03–0.27%, i.e. two models trading places on noise. ⚠ **Test A's comparison is complexity-neutral, which is what makes it conclusive:** HAR-RV and HAR-C have **the same number of regressors** (3 + constant), only the regressor *definition* changes — there is no parsimony advantage to discount, the gain is purely informational. And HAR-CJ's 3 extra regressors, the only difference between the two, buy nothing. **Mechanism:** it is `BV` doing the work, as a **robust estimator**, not `J` as a signal — jumps are noise to *remove* from the predictor, not information to *add*. Consistent with the even/odd moment dichotomy (§12.3): the jump component is a rare, signed event, the same family of quantities that are unpredictable on this asset for the NN *and* for the econometric baselines. **Applied consequences:** (i) **HAR-C is the reference baseline** — same accuracy as HAR-CJ but **450× better conditioned** (condition number 119.9 against 5.43e+04) and with **identified** parameters: between two equivalent measuring instruments you keep the stable one; (ii) the **published band does NOT change** — NN/HAR-C = 0.7758 on val (−22.4%) and 0.6835 on test (−31.7%), against −22.6%/−31.6% versus HAR-CJ: at published precision **−23% to −32%** is identical, and no GPU was spent because the NN numerator was already recorded on the same npz (vintage control: HAR-RV and HAR-CJ reproduced digit for digit on both splits); (iii) the **HAR-CJ respecification** (jump rescaling/regularization) opened by C2 is **moot** and closed at no cost — you do not regularize coefficients of regressors that carry no information; (iv) C2's **verdict** stands: HAR-CJ *is* stronger than HAR-RV and the claim survives it, simply not for the reason attributed to it. Implementation: `HAR_C_COLS` + `har_c_fold_qlike` in `quantsys/model/vol_metrics.py` (it reuses the CJ frame, so the sample is identical by construction), inertia proven by an end-to-end A/B against the pre-patch judge (43 common keys, 0 differences), tests in `tests/test_har_c_baseline.py`. **ADOPTION 2026-07-31 — the two flags `QUANTSYS_HAR_CJ`/`QUANTSYS_HAR_C` were REMOVED, not switched on:** HAR-C and HAR-CJ are now computed on every run. An always-on flag is not a lever but one more failure mode, and the property the flags protected — reproducing the historical report — is already guaranteed by version control (`git show <commit>:scripts/vol/dev_vols_qlike.py`, the mechanism behind all three inertia proofs); duplicating it with a compatibility flag would be reimplementing git inside the application. ⚠ **Invariant preserved and verified:** the blocks stay outside `metrics`/`gate`, so the pre-registered 2026-06-10 **gate** keeps **HAR-RV** as its denominator while the **claim** uses HAR-C — two different things, and stdout prints each number's role, because confusing them is the mistake the three-baseline panel exists to make impossible. Control run without flags: 103 keys identical to the C3 report, `metrics` and `gate` unchanged. ⚠ **Peeking warning declared ex-ante:** two of the three numbers (HAR-RV and HAR-CJ) were already known from 30/07, so this gate was structurally weaker than C1/C2 — the discipline adopted was to anchor **both** decision statistics to HAR-C, the only unknown quantity, and to use `φ` as descriptive only, since its denominator was an already-seen number.

**Provenance of the published band's numerator — CLARIFIED (2026-08-01), together with a reproducibility limit that must be declared.** The **−22.42% to −31.65%** band takes as numerator the QLIKE of a **RETRAINED model/scaler pair** on the current npz (val **0.26143**, test **0.23637**), not that of the frozen checkpoint in `models/itransformer`, which is the **June PASS restore**. The distinction is not pedantry: on the very same sample the June checkpoint yields val **0.27470** and test **0.24453**, i.e. NN/HAR-C ratios of 0.8152 and 0.7071 (−18.5% and −29.3%) instead of 0.7758 and 0.6835. **The sample is not the cause:** `n`, HAR-RV, HAR-C, HAR-CJ and naive match **digit for digit** across all recent reports (val n=6485, test n=6486) — **only** the model varies. ⚠ **The reason the June checkpoint is NOT the right numerator is methodological, not convenient:** its `PipelineState` carries `target_scale = 1.4375922` and a different RobustScaler (distinct `center_`/`scale_` md5) from the current npz's canonical one (`target_scale = 1.4268685`). Evaluating it on today's dataset means feeding it inputs normalized under one scaler and denormalizing μ against another: part of the resulting difference is a **scaler artifact**, not skill — the same class as A8's −4.94%, which at equal scaler was −0.79% (§12.4). A retrained pair is trained **and** evaluated inside the same vintage, so it is the only clean comparison. ⚠ **Limit declared from 2026-08-01 to 2026-08-04, now CLOSED — see the R1 paragraph at the end.** The sandbox that had produced the numerator was **deleted** when the production state was restored (protocol step 5), so the band was **not digit-reproducible**. The explanation written at the time — "~0.2 points of **seed-draw** uncertainty", inferred from the gap between two replicas (val 0.26206 against 0.26143, ratios 0.7777 against 0.7758) — was **measured on 2026-08-04 and turned out to be wrong**: at fixed seeds, config and npz the protocol is **deterministic** and retraining dispersion is **zero** (R1 reproduces 0.26143 and 0.23637 to the tenth digit). The gap between the two replicas is real but **is not RNG-draw noise**: it stems from a difference between those two runs that was **identified on 2026-08-04: the MACRO VINTAGE inside the npz**. `01b_download_macro.py` reloads the splits and replaces **only** `X_macro_{split}`, leaving `X_*`, `y_*`, `t_*` untouched; macro is a **model input** (90 columns, macro embedding active) but enters **no HAR baseline**, which read RV/target only — hence the observed signature, the NN moving while baselines stay identical digit for digit. The temporal alignment is exact: only two npz rewrites (19/07 16:54:57 and 30/07 20:55:02); every run in the 0.26206 cluster sits between them, every run in the 0.26143 cluster after the second, and the first of those starts **87 seconds** after the rewrite. ⚠ And it is not a plain append: the `MacroNormalizer` is **refit whole-df** and reapplied to every row, so extending the series shifts median and IQR and changes macro values **for historical rows too** — the same "instrument vs state" already documented on the live path, measured here on the training side. The alternatives are **excluded, not ignored**: the seed (Δ = 0, measured), the batch (811 batches/epoch across all runs) and the code (the only two commits touching the training path in that window are one logging-only and one inert at flag-off, falling back on the same `scaled_dot_product_attention` and adding no parameters, hence consuming no RNG). ⚠ **A gap that follows, declared:** the scaler identity guard covers the price RobustScaler and `target_scale`, **not** macro normalization — which does not live in the canonical `PipelineState` (`0 macro features`). There is therefore a **second, uncovered vintage axis**: two models can both pass `matches: true` and have been trained on different macro, with a measured ratio gap of **0.0019**. Comparing NN numbers across a macro refresh is as much a cross-vintage comparison as across a different scaler. **CLOSED — gate M1 run on 2026-08-05, PASS ⓪①②③④, zero GPU.** `02_train.py` now records in the `PipelineState` the fingerprint of the macro block of the npz actually consumed (per-split md5 of `X_macro_*`, column order and count, dtype; source `measured`), and the three vol judges recompute it against the current npz and fail fast on a mismatch, with an explicit `--allow-macro-mismatch` escape kept **separate** from the scaler one because the two axes are independent. Inertia verified against R1's archived reports: **118 common keys, zero numeric differences on both splits** (the only two differences are path labels inside the `provenance` block), so the published numerator does not move by one bit. Two-level positive control in `tests/test_macro_vintage_guard.py` (21 tests): a single changed cell, a column reordering at identical values and a dtype change must each trip the guard — without it, a guard always returning `true` would have passed the inertia check perfectly. ⚠ **Two limits declared, not closed.** (i) The positive control is **synthetic**: macro vintage V1 no longer exists and rebuilding it would require re-running `01b`, i.e. rewriting the frozen npz — so what is demonstrated is that the guard separates two different macro blocks, **not** that it would have caught that historical event. (ii) Pre-M1 models — `models/itransformer`, `models/backup_1h_vols` and **the canonical artifact itself** — carry no fingerprint and stay `matches: null`, i.e. **macro provenance not verifiable**. The documentary backfill foreseen by the pre-registration was **deliberately not performed**: a fingerprint recomputed today from the current npz would match **by construction**, hence carry no evidence at all, and would merely convert an honest "unknown" into a reassuring `IDENTICAL` — precisely the failure mode the gate's condition ② existed to prevent, walking in through the front door. The artifact's macro provenance therefore remains the dated line in `PROVENANCE.md`, which is a declaration and reads as one. ⚠ **A sufficient mechanism, exhibited by R1's diagnostic arm B (2026-08-04):** retraining at `batch_size 128`/`gradient_accumulation_steps 1` instead of `64`/`2` — a **purely computational** knob, at identical data, architecture and seeds — moves the ratio from 0.7758 to **0.7690**, i.e. **Δ = 0.0068**, a full **3.5×** the gap between the two clusters. It does not prove which difference separated those two runs, but it establishes that a config choice **suffices** to produce that gap, whereas the seed, measured, does not produce it at all. **Consequence for reading the band:** its endpoints carry roughly **±0.7 percentage points of uncertainty with respect to the chosen training configuration**, and that uncertainty is **not stochastic** — it is removed by declaring the config, not by averaging over replicas. The canonical pair's config is the production one, declared in `models/canonical_1h_vols/config.json`. ⚠ The second replica's ratio had already been corrected on 2026-08-03 (the previous version read 0.7738, which is the *first* replica's ratio against **HAR-CJ**, not the second's against **HAR-C**). **Consequence for the reader:** the band is a statement about the **training protocol** applied to the current dataset, not about the individual checkpoint file sitting in `models/`; and the `metrics.nn` fields of `results/vols/qlike_report_1h_{val,test}_c3.json` **are not** the band's numerator and must not be used as such. **Fixes applied so it cannot recur:** (i) `dev_vols_qlike.py` **fails fast** when the model scaler ≠ the npz's canonical scaler, with an explicit `--allow-scaler-mismatch` escape that flags the report as non-comparable (the guard was seen failing on the production model itself); (ii) every report now carries a **`provenance`** block with model dir, arch, npz and both scaler fingerprints — a number can no longer be orphaned from the model that produced it; (iii) `tests/test_scaler_identity_guard.py`. **RESOLVED — gate R1 run on 2026-08-04, PASS on every pre-registered condition.** The canonical pair was trained at **unchanged** production config on the frozen npz and now lives as a permanent artifact at **`models/canonical_1h_vols/`** (with `PROVENANCE.md`, both judge reports and the scaler fingerprints). Outcomes: ⓪ baselines reproduced **digit for digit** on both splits (val n=6485, test n=6486) → same npz, not a new dataset; ① `provenance.matches = true` **without** `--allow-scaler-mismatch`; ② the historical gate against **HAR-RV** survives (val 0.26143 ≤ 0.33913, test 0.23637 ≤ 0.35149, both far below naive); ③④ materiality met — **NN/HAR-C = 0.7758 on val (−22.4%) and 0.6835 on test (−31.7%)**, against pre-declared thresholds of 0.80 and 0.71. **The published band does not change by a single digit**: the canonical pair's numerator matches **exactly** the one already published (val 0.26143, test 0.23637). What changes is the claim's **status** — from a statement about the training protocol to a verifiable artifact, re-judgeable with `python scripts/vol/dev_vols_qlike.py --arch canonical_1h_vols`. ⚠ **What the PASS does NOT authorize — and what was not done:** promotion into `models/itransformer` and onto the VPS. Replacing the production model would change `04b`'s input **inside open pre-registered forward samples**, the same violation as refreshing macro inside a sample: that is a separate, dated decision, conditional on those samples closing. Until then the canonical pair lives **alongside** production, not in its place — and the model feeding the live arm remains the one with `target_scale = 1.4375922`, misaligned with the current npz and correctly rejected by the guard.

**Band precision — re-expressed on 2026-08-05, with nothing re-measured.** The band now reads **−22.42% to −31.65%**, i.e. the canonical pair's NN/HAR-C ratios written out: val `0.7757926` and test `0.6834522`, both readable in `models/canonical_1h_vols/`. **No run, no GPU, no new number** — the wording changes, not the measurement. The two endpoints move for **different** reasons, worth separating. The **lower** one was **stale**: −23% is the rounding of the −22.6% C2 measured against **HAR-CJ** on 2026-07-30, carried over when C3 replaced the denominator with HAR-C and never realigned; against HAR-C the value was always −22.42%, and the gap was **conservative** (the band claimed less edge than measured), which is why it was never urgent. The **upper** one was not stale but **rounded to the point**: −31.65% → −32%, and it was the only one of the two rounded in the direction that **favors** the claim. Taking both to two decimals removes the asymmetry — publishing a precise figure where it helps and a rounded one where it helps is the kind of detail a competent reader notices first, and it costs more credibility than the half point is worth. ⚠ **The two decimals are NOT a statement of uncertainty, and reading them that way would be a serious error:** they identify *which* (model, npz, config) triple produces the number, not a confidence interval. The actual uncertainty, measured on 2026-08-04, is **~±0.7 percentage points with respect to the training configuration** (R1's arm B: `batch_size 128`/`ga 1` moves the ratio by 0.0068) plus **0.0019 with respect to the npz's macro vintage**, while with respect to the **seed it is exactly zero**. None of the three is stochastic: they are sensitivities to deterministic choices, removed by **declaring the triple** — which is precisely what the two decimals do — not by averaging replicas or widening the band.

**The edge is specific to the hourly resolution.** The cross-resolution check at 30min RV is a **val FAIL** (NN/HAR QLIKE 1.0127 > 0.95): the result does not transfer by changing bar scale.

**Monetization (short-vol arm, forward test).** The VRP edge is structurally confirmed by the 2019→2026 FHS-GJR-GARCH historical backtest (n=2538; the edge is **Trending-driven**, not Stress: the "Stress = best Sharpe" reading was a constant-haircut artifact). But the real gate is live: on Deribit testnet the **pre-registered v1 gate is FAIL 0/3**, with `always-short` at +0.0396 — i.e. VRP is positive and the arm remains sound, while **the v1 rule does not monetize it**. A permanent distinction: *measured predictive skill* ≠ *monetization*.

**V2 delta-hedged — pre-registered gate CLOSED on 2026-08-11: FAIL 2/3, and the drag is almost all fees.** Forward sample `n = 20` settlements with the hedge active (19/07 → 11/08, excluding by pre-declaration the position already open at activation). ① **variance: PASS with margin** — `var(hedged)/var(unhedged) = 0.447`, i.e. **−55.3%** against the required −40%: the hedge does exactly what it was supposed to do to the second moment. ② **cost: FAIL** — `mean(hedged) − mean(unhedged) = −0.001312 BTC = −0.647·SE`, against a pre-registered budget of −0.25·SE (2.6× the threshold). ③ n ≥ 20: PASS. **Drag decomposition** (identity `hedged − unhedged = gross − fee − funding`, residual 1.7e−18): **perp fees 75.9%** (−0.000996 BTC/trade over 76 rebalances, median 3.5/trade), gross perp −0.000305 (23.2%, mean-zero by design and indistinguishable from zero at n=20), **funding 0.9%** (−0.000012). ⚠ **The pre-declared hypothesis was right about the variable and wrong about the component:** the written risk was "churn fees **+ funding**", with funding flagged as the big unknown *never measured on the series* — measured, it is worth **less than 1%**. Zero-fee counterfactual: drag −0.156·SE → ② would pass and the gate would be PASS 3/3. ⚠ **Honest qualification, which does not move the verdict:** ② is a **budget**, not a significance test — the paired t on `hedged − unhedged` is **−1.051** (hedged better in 7/20 trades), so what is demonstrated is that the hedge **does not buy PnL** and that its realization cost at 1-contract size eats a multiple of the declared budget, **not** that the drag is negative in population. **Pre-declared consequence applied:** `--hedge` marked FAILED and disabled, the **unhedged v1 remains the production paper design**; v2 sizing (HAR-q90) and A7 greeks-aware do **not** unlock, as both were conditional on a PASS. **Hypothesis B2 (hedge = VRP purifier) dies on realization cost, not on theory** — ① is the proof that the theory held. It is the third time in this project that a measured effect fails to survive transaction costs, after the 1m cost wall and the net-of-cost KILL of IVS relative-value.

**MFIV comparator v2 — pre-registered gate CLOSED on 2026-08-18: FAIL, and the permanent value is in the descriptive part.** Question: `04b`'s edge uses the 30h-interpolated ATM IV as its comparator; the **model-free MFIV@30h** (VIX-style integral over the OTM strip) is the correct var-swap rate, which ATM understates because of smile convexity. Does it improve the **ranking** of realized PnL? Paired sample `n = 41` daily expiries (pre-registered threshold 40; identical expiries for both arms, shared `RV_pred`). ① `Δρ = ρ_MFIV − ρ_ATM = +0.0343` against the required **+0.05**: **FAIL**. ② the sign of `Δρ` is **not** consistent across the two chronological halves (−0.0015 / +0.0610): **FAIL**. ③ `n ≥ 40`: PASS. `ρ_ATM = −0.1439`, `ρ_MFIV = −0.1096` (Spearman of `−edge_x` against short PnL). **The outcome was predicted ex-ante by the pre-registration itself:** were the log-variance wedge **constant**, `edge_MFIV = edge_ATM − c` and the two Spearmans would be identical by **rank invariance** — the gate measured only whether the wedge's *time variation* adds ranking power. Measured on the 41 entry ticks: log-wedge median **0.2005**, interdecile range **0.1733** — a large, stable level with dispersion that barely moves the ordering, hence `Δρ ≈ 0`. With a Spearman SE at `n=41` ≈ 0.16 (declared ex-ante), a `Δρ` of +0.034 sits inside the noise: **FAIL, not a "near miss"**. **Pre-declared consequence applied:** the live comparator stays **ATM IV**, MFIV remains a **permanent diagnostic column**, item closed, no v3 pre-registration. ⚠ **Non-gating descriptive — short-vol break-even re-estimate** (the experiment's permanent value, due whatever the verdict): on the entry ticks MFIV prices variance **+22.2%** above the interpolated ATM (p10 +13.4%, p90 +34.9%), i.e. **+10.55%** in volatility terms (p10 +6.50%, p90 +16.14%). The historical backtest gave break-even VRP = 0% with an FHS fair value ≈ ATM IV: measured against the correct var-swap rate, that same break-even corresponds to realized variance **18.2% below** the model-free rate — the short-vol arm's cushion is wider than ATM suggested, by a median **20.1 log-points of variance**. ⚠ This is not extra PnL: `04b` sells the **ATM** straddle and collects the ATM premium; the wedge says the **premium available in the market** is richer than the comparator shows, not that the current position harvests more of it — an argument for structures selling a larger share of the strip (the historical backtest's **8% strangle** already points that way). ⚠ Non-decisional diagnostic: **both ρ are negative** and at `n=41` neither is distinguishable from zero — no detectable relation between edge and short PnL under either comparator, consistent with the v1 gate's FAIL 0/3. Machine record: `results/vols/mfiv_comparator_report.json`.

**Engineering addendum (2026-09-13/15, local, not deployed, no scientific claim touched).** The adaptive executor behind `--adaptive` (a durable on-disk journal, `adaptive_entry_journal.json`, blocking `tick()`/`main()` until verified flat or manually reconciled) had three correctness gaps, fixed without touching the pre-registered rule: (a) fee/timing observed from REAL trades were treated as complete even under partial or malformed coverage (duplicate ids, mismatched instrument/order, quantities not summing to the known fill) — `_verify_trade_history` now checks coverage and, if incomplete, fee and `fill_span_s` stay **unknown** (never zero, never the local receipt time in place of the fill); also the span used the max-min of the per-leg **last** fills, undercounting it when a leg had multiple fills — it is now the **global** first-to-last across all entry legs. (b) if a tick's adaptive entry left the journal open (`blocked_operator_review` outcome), that same tick's tail still logged a "flat" diagnostic snapshot despite the real exposure being unknown — that tail (diagnostics + hedge leg) is now skipped while the journal survives. (c) trade identity was **optional** in the verification (a trade missing `order_id`/instrument/side passed: coverage came out verified with nothing to verify) and a non-hashable `trade_id` raised inside an unprotected function, interrupting journaling with orders already sent — identity is now required and every non-verifiable case degrades to unknown; each leg's concrete instrument and side, which lived **only** in the journal cleared on verified flat, are now persisted in the execution and recovery record. None of the three fixes changes the entry rule, sizing or settlement logic; regression in `tests/test_adaptive_structure.py` (40/40, after the independent diff review: identity required in the order verifier too, partial `filled` treated as ambiguous). No VPS deploy, no FT1 start.

**SignatureHAR — pre-registered gate CLOSED on 2026-08-25: val FAIL, and the value lies in what it falsified of the prior.** Question: do *path signature* terms — the family the rough-path literature sells as a complete representation of a trajectory — add predictive power to the **HAR-C** baseline on the 1h `log_rv` target? **Construction frozen before running:** causal window `W = 24` hourly bars, **time-augmented** 2-D path, **depth-3 log-signature** (5 independent terms for `d = 2`; shuffle relations already removed by the logarithm) **+ 1** quadratic-variation term from the **lead-lag** path = 6 columns on top of HAR-C's 3, in **closed form via Chen's identity** — no new dependency, `signatory`/`iisignature` stay out of the project. Same `build_har_cj_frame`, same train, same judge QLIKE: **only the regressor set differs**. **Ex-ante condition ③:** on the lead-lag path the level-2 antisymmetric term **is** the window's quadratic variation — verified **numerically over segments**, not assumed, max relative gap `2.8e-15`, with `ρ(log QV, log RV_W) = 1.000000`. But redundancy **against the baseline** is falsified: `ρ` against `xc_h` is `0.9439` at `W = 24` and `0.9760` at `W = 30` (window matched to `h`), Spearman `0.9345`/`0.9707`, below the pre-declared `0.99`. The cause is measured, not conjectured: `xc_h` is `log(C_h)`, **not** `log(RV_h)` — C3 had replaced HAR-RV with HAR-C, which removes jumps **by decision**, and the arm had been written against the **previous** baseline's regressor; the unexplained part is the jump (`R²(log QV | HAR-C) = 0.953`, residual correlating `0.356` with `xj_h`). **Gate outcome (val, `n = 6485`):** baseline reproduced digit for digit — `QLIKE(HAR-C) = 0.336979` from both `har_c_fold_qlike` and the probe chain, gap `< 1e-12`; **positive control FIRES** (oracle column bisection-calibrated to exactly `Δ = −0.004290`, `p = 0.0028`), so the instrument resolves an effect of the size that matters and the FAIL **is interpretable**; candidate `QLIKE = 0.332913`, i.e. `Δ = −0.004066`, `DM = −3.184`, `p = 0.0015` (HAC lag 29, `n_eff = 216.2`) — **significant but SUB-MATERIAL**, `0.000223` short of the `0.33269` bar, **94.8%** of the threshold. In editorial consequence, which is what the threshold measures: the published band would move from **−22.42% to −21.47%**, i.e. **0.947 percentage points**, under the required one. ⚠ **The threshold was not touched:** it was derived from the band at numbers unseen, and this is exactly the case it existed for — the pre-registration warned that a material Δ with `p > 0.05` is a FAIL, not a near-miss, and the mirror image arrived. **Mechanism, from the descriptive per-arm decomposition** (p-values **not** multiplicity-corrected, on a split already consumed by the gate: they generate hypotheses, they do not confirm them): the **even** order-2 term contributes nothing (`Δ = −0.000829`, `p = 0.262`), while the **ordering** terms carry the bulk — Lévy area `−0.000540` (`p = 0.032`), level 3 `−0.002131` (`p = 0.039`), together `−0.002966` (`p = 0.007`). **The pre-registered prior (FAIL ~85%) got the verdict right and BOTH mechanisms wrong:** the order-2 arm is null, but through **C3**'s result — what it adds over `C` is the **jump**, and jumps are noise to remove, not signal to add — not through the predicted redundancy; the ordering arm is **falsified in direction**, and its error is the conceptual one worth recording: the even/odd dichotomy (§12.3) says odd moments are unpredictable **as TARGETS**, not that odd inputs are useless for predicting an **even** target — using the path's sign to predict variance is what a GJR leverage term does, and this repo already estimates one. **The thread closes at this scale** (the `h` window) and does not reopen with "more depth": depth 4 only adds further ordering terms, already measured here. Probe: `scripts/vol/sig_har_probe.py`, **read-only**; machine record `results/vols/sig_har_probe_1h_val.json`; **`test` was not touched** and `dev_vols_qlike.py` was not modified, since it judges open pre-registered forward samples.

### 12.3 KILL corpus — closed lines

Not to be reopened without a **new hypothesis** (not a new hyperparameter):

| Line | Outcome and number |
|---|---|
| 1m directional | **Cost** wall: ~1.5 bps effect against ~26 bps round-trip cost |
| 1m→1h pivot | At 1h the cost wall falls (cost/σ from ~1.9-3.3× to ~0.25-0.42×, \|μ\| ≈ 43 bps) yet **no OOS directional skill**: pre-registered gate failed **4/4** at both 13 and 23 bps |
| Cross-sectional probe | KILL: no exploitable cross-sectional IC |
| Signed semivariance (`log_rs_ratio`) | Test FAIL: NN/HAR-RS MSE 0.9952 (gate ≤0.95), signDA 0.459, and **HAR-RS does worse than the constant** → the asymmetry is unpredictable *for everyone* |
| IVS relative value | **Net-of-cost** KILL: real structure (residuals revert) but −2.3/−3.8 vol pt net per leg, ~50× below the spread — viable only as a market maker |

### 12.3-bis Positive control: two cases, two ways of failing

**The case that justifies the positive-control condition (B1 stage 1, pre-registered and run 2026-07-31).** Question: does the L2 order book carry incremental information about **3-hour** RV beyond RV's own lags? Design: two **nested** OLS on identical points — baseline HAR-C, candidate HAR-C + three unsigned L2 features chosen by **target-free** diagnostics — expanding-window out-of-sample forecasts with an **h−1 embargo** (the target sums the next h hours: training on the last h−1 observations would be leakage), QLIKE loss, paired Diebold-Mariano. Target built from **1-minute** bars (~180 squares per observation instead of 3). `n_eval = 289`, `n_eff = 96.3`. **OUTCOME: NO CONCLUSION.** QLIKE: baseline 0.46389 · candidate 0.48615 (ratio 1.0480) · **naive persistence 0.40188**. The pre-registered condition ④ — *the baseline must beat naive, otherwise the main comparison is uninterpretable* — **failed**: HAR-C at 3 hours loses to persistence by 13.4% in point estimate (DM +1.724, p=0.086). **Why it matters:** without that condition the reportable outcome would have been "candidate 4.8% worse → L2 carries no information", i.e. a **publishable false negative**. The candidate is worse than a baseline that is worse than doing nothing, so that 4.8% does not measure L2: it measures that the apparatus does not work at that horizon. **The defect is in the design, not the implementation:** the expanding window was chosen to maximize evaluation points, and in doing so it had 4 parameters estimated on ≤400 observations against a **zero**-parameter naive — while the production HAR is fitted on 51,882. Added to that, a prior error: HAR performs best at daily-to-weekly horizons, and at 3 hours the dominant component is the recent one, i.e. almost exactly what naive computes. **No variant was run once the numbers were seen** (the pre-registration's anti-goalpost constraint): reopening requires a new gate. **Transferable lesson:** in a two-model comparison the condition that really protects you is not the one on the candidate but the one on the **baseline** — and it must be written first, because after the fact "the baseline was weak" sounds like an excuse rather than a control.

**The second case, and it fails in the opposite way (E1 stage 2, pre-registered 2026-07-31, run 2026-09-10).** Question: does the forecaster's edge against implied — `x = log(rv_pred / var_iv)` — carry predictive content about the realized outcome `y = log(RV / var_iv)`, i.e. does the model beat the **market's** forecast and not merely an econometric baseline? Unit: daily expiry at ~30h tenor, one observation each, decision tick the last one with both quantities finite in `[E−33h, E−27h]`. Confirmatory design on expiries settling **after** the pre-registration commit, preceded by an exploratory stage with no thresholds on the already-recorded window (the PnL of those expiries had already failed a gate, so they carried partial information about the outcome). `n = 40`, window 2026-08-01 → 2026-09-09. **OUTCOME: NO CONCLUSION.** Sign agreement 0.3500 (HAC lag 1, t −1.855); Spearman −0.2644 with blocked CI95 [−0.5629, +0.0572]; **positive control: QLIKE 0.59250 (model) vs 0.97617 (30h naive persistence), DM −1.165, p 0.2510, `n_eff` 20.** **Why this is B1's complement:** there the baseline **lost** to naive in point estimate and the defect was one of **design** — 4 parameters estimated on ≤400 observations against a zero-parameter naive. Here the model **beats** naive by 39.3% in QLIKE, with the difference in the right direction; what is missing is **significance**. The control does not say the apparatus is broken: it says that at this sample size it cannot be shown to work, and without that demonstration the two primary statistics are not interpretable. **The defect lies in the pre-registration, not in the numbers:** the power section had dimensioned ex ante the minimum detectable effect of the two primary statistics (|agreement − 0.5| ≈ 0.26, ρ ≈ 0.53) and **never that of the control**, which is the condition that actually bound — the more so since the Diebold-Mariano is computed with an overlap factor that halves the sample (`n_eff = n/2`), more conservative than the one used for the other two. **Transferable lesson, and different from B1's:** a mandatory positive control is a PASS condition in full and must be power-dimensioned with the same care as the primary statistics; otherwise one builds a gate that **cannot conclude in either direction**, and finds out once the sample is spent. Both primary statistics moreover fall inside the undetectable band fixed before the run (|agreement − 0.5| = 0.15, |ρ| = 0.264), so the outcome is also a **measured upper bound**: the effect, if it exists, is below those thresholds. ⚠ **No inversion of the rule is licensed** by the negative sign of the two statistics: the condition was pre-declared one-sided, under "no conclusion" those numbers are uninterpretable by construction, and the pre-registration's anti-goalpost constraint requires a new gate for any variant — sign inversion and extension of `n` included.

### 12.4 Falsified training levers on the vol line

All judged **against a retrained baseline** on the same dataset and scaler, PASS threshold −3% in QLIKE:

- **Mixup** (`mixup_alpha` 0→0.2): −0.79% → FAIL. First-order methodological note: an earlier descriptive measure gave −4.94%, but it was a **distribution-shift artifact** (cross-scaler comparison against the incumbent). It is the origin of the rule in §12.1.
- **DVOL as a feature** (Deribit risk-neutral IV, 3 columns, causal asof): −1.02% → FAIL. Log-MSE improves by 6% but **QLIKE does not** → the signal moves the log-RV mean without improving variance calibration, which is what QLIKE penalizes. Reading: 30d→30h tenor mismatch + the NN already captures IV information through non-linearities on lagged RV.
- **Quantile calibration** and **per-QLIKE member weights**: val FAIL.

**Attention sparsity (A10) — FALSIFIED 2026-07-30, and with it the class CLOSES.** Val outcome, paired arms retrained on the same npz: **QLIKE 0.26191 (candidate) vs 0.26143 (baseline) → +0.18%**, i.e. marginally worse against a −3% threshold → FAIL. The other three conditions passed: no qualified regime destroyed (r0 0.28336 ≤ 1.05×0.28433; r2 0.25387 ≤ 1.05×0.25177), valid sample (n=6485, r0+r2 qualified), and above all the **manipulation check ④ passed by a wide margin**: mean `H_norm` 0.62725 on the candidate vs 0.82942 on the baseline, **−24.4%** against a −5% threshold. That makes the FAIL maximally informative: the penalty **did** concentrate the attention maps as intended, and QLIKE did not move → **the hypothesis is falsified, not the implementation**. Attention spreading over spurious correlations among the 104 variates is not the bottleneck; the measured bottleneck remains the **effective sample** (~1.7k independent windows). The DVOL signature also recurs: **log-MSE improves (0.5939 vs 0.6021, −1.4%) while QLIKE does not** — two independent levers moving the conditional mean of log-RV without improving variance calibration. **Consequence declared ex-ante and now applied: the "training levers on the vol line" class is closed.** A10 was the last candidate; the next gain must come from **new data** (order-book L2 under forward collection, A4 HAR-CJ) or from **monetization** (short-vol arm), not from another training variant. The penalty code stays **inert at 0.0** as executable documentation; the overlay was deleted. Technical description of the lever, for a reader who wants to replicate:

**The lever (A10, `model.attn_entropy_lambda`, inert at 0.0).** In the iTransformer the tokens are the **variates** (F=104) while the effectively independent windows are ~1.7k (29/30 target overlap): attention can spread over **spurious** inter-feature correlations. The lever adds a `λ·H_norm` term to the objective, with `H_norm` = mean row entropy of the attention maps **normalized by log N** (N = sequence tokens) → `H_norm ∈ [0,1]`, so λ is by construction the **maximum** cost in loss units; λ=0.01 is fixed by that rule, **not** by a sweep. It is **structural** regularization (where the model looks), not a capacity cut. Implemented 2026-07-29 and run 2026-07-30: at λ=0 the layer does not materialize the N×N matrix and stays on `F.scaled_dot_product_attention` (bit-identical path, unchanged `state_dict`, legacy checkpoints loadable — `tests/test_attn_entropy.py`); at λ>0 the softmax is explicit and in float32 even under AMP, giving up Flash-Attention on that path. **Measurement** is decoupled from **penalization** (`set_attn_entropy`): it is needed to measure `H_norm` on the baseline arm too, which trains at λ=0, for the gate's manipulation check — if the penalty did not lower the entropy, the run measured the implementation rather than the hypothesis. At inference measurement is **always off**, so the judge evaluates both arms with the same attention implementation. Probe: `scripts/vol/attn_entropy_probe.py`. Honest prior declared before the run: **low** — no training lever had ever passed on this line; the outcome confirmed it.

### 12.5 Realization: why a rank edge does not become PnL

On the directional line seven realization levers were implemented and validated, **all inert by default** in `scripts/03_backtest.py` and all OOS failures: regime gating (`QUANTSYS_REGIME_ALLOW`, `_INVERT`), discrete rank-based entry, decision cadence (`QUANTSYS_DECISION_CADENCE`), continuous exposure proportional to the causal percentile of μ (`QUANTSYS_RANK_EXPOSURE`), purely temporal exit (`QUANTSYS_HORIZON_EXIT`), σ recalibration (`QUANTSYS_SIGMA_SCALE`: the optimum is ≈1.0, shrinking σ makes it **worse**), cost-aware |μ| gate (`QUANTSYS_MIN_EXPECTED_RET`: non-binding at 1h). They stay in the code as **executable documentation of the failure**.

**Single reading of the corpus:** the directional edge is one of **rank**, not of threshold, and **does not survive the realization machinery** — PnL is dominated by the SL/TP path, not by the horizon return. Neither regime gating nor σ recalibration produces OOS PnL.

**Safety nets not to be removed** (deliberate fail-fast guards, not defensiveness): `RuntimeError` if `σ_max ≥ 0.05·√interval_minutes` in raw space (scales with √Δt: catches 30-100× denormalization bugs, not legitimate per-bar vol growth); validation of `forecast_horizon` **and** `interval` between training and inference; model↔dataset **scaler** identity (`check_model_dataset_scaler`); `merge_asof` test↔raw-candle alignment with `len == n_test`; floor `sl_d = max(sl_d, price×1e-4)`; atomic checkpoints (`.tmp` + `os.replace`); **regime walk-forward anti-degradation guard** (`RegimeMarkovSwitching`, 2026-08-02: `RuntimeError` if ZERO Markov-Switching fits succeed — the result would be the **uniform prior** `1/n_regimes` dressed up as regime probabilities — or if the failed-fit fraction exceeds `max_fit_failure_ratio`, default 0.5; the same guard is mirrored in `continue_walkforward`, where silent failure instead yields **stale** parameters. ⚠ **Threshold exercised on real data on 2026-08-05**, since an abort threshold never measured in the field is itself an availability risk: a read-only walk-forward over **2 years of real candles** (17,520 hours, 2024-08→2026-08) at production cadence (30d burn-in, 90d retrain) → **8 fits out of 8 succeeded, `fail_ratio` = 0.000, filtered-probability coverage 100.0%** (16,800/16,800), i.e. a **0.5 margin** to the threshold and no risk of the guard aborting a legitimate rebuild. `last_fit_diagnostics` is populated on a real run, not only in synthetic tests); **DLL initialization order** (`import pyarrow` anchored in `quantsys/__init__.py`: loading pyarrow after both torch **and** scikit-learn causes an access violation at the first `read_parquet` — the numbered scripts survived only because they import `pandas` before `torch`, a de facto invariant now made a package property and tested).

⚠ **Note on the nature of the two guards added on 2026-08-02:** both turn a **silent failure into an explicit one** and do not touch the numeric path — with successful fits and pyarrow present the output is bit-identical (the B7 incremental-regime bit-parity stays green). They are not optimizations: they remove two ways of failing unnoticed, surfaced by the performance audit (`docs/PERF_AUDIT.md` §7.1-7.2).

---

# Flow summary

```
Binance REST/WS
      │
      ▼
OHLCV 1h candles (history 2019-01-01 → today, ~65k bars — 2026-06-09 pivot)
      │
      ▼
Log returns + 104 features (VWAP, VP short/mid, CVD, momentum,
                                microstructure, funding, interactions, time, lags)
      │
      ├─── Macro features (FRED + yFinance, 90 features → 16-dim MacroEncoder)
      ├─── BTC → hourly realized vol → RegimeMarkovBTC (causal Markov-Switching,
      │                                                    filtered probs, 3 data-driven
      │                                                    regimes R0/R1/R2)
      │
      ▼
Normalized 120×104 windows (global RobustScaler) → NPZ dataset
      │
      ▼
Architecture selected with --arch (or --distill for ensemble):
      │
      ├─ lstm         → dual-stream LSTM (dyn. + struct.) + attention   (legacy)
      ├─ itransformer → attention over features (multi-scale ×1/×5/×15 bars)
      ├─ tcnmamba     → TCN (dil. 1-32, RF=127) + Mamba SSM (context 120)
      │                  └─ gated fusion → unified representation
      ├─ nhits        → Neural Hierarchical Interpolation (pure-MLP, stacks 8/4/1)
      │
      ├─ [--distill] Multi-Teacher Knowledge Distillation:
      │              archs from config/default.yaml → scoring → soft labels
      │              shuffle-safe → student with 60% epochs
      │
      ▼
Output: μ + σ + ν  in z-score  (μ = 1st moment if target=ret, log-RV if target=log_rv)
      │
      ├─ target = log_rv  (PRODUCTION):
      │     full inversion  log_rv = μ_z·scale + center  →  RV = exp(log_rv)
      │     → QLIKE vs HAR-RV  →  RV-vs-IV signal  →  Deribit straddle (04b)
      │
      └─ target = ret  (LEGACY, KILLED OOS):
            PipelineState.denormalize_predictions(μ, σ)  →  raw space
                  │
                  ▼
            Monte Carlo 2000 scenarios × 30 bars (30h at 1h; GJR-GARCH for volatility)
                  │
                  ▼
            Conviction score (direction × magnitude × calibration × regime)
                  │
                  ▼
            RiskManager (Kelly size, ATR stop, trailing, 15% MtM circuit breaker)
                  │
                  ▼
            Signal: BUY / SELL / HOLD  +  size  +  stop loss  +  take profit
```
