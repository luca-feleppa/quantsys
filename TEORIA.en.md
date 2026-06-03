# QUANTSYS — How the system works

A descriptive walkthrough of the full flow, from raw material (candles) to the operating signal (BUY / SELL / HOLD with size and stop loss).

> Italian version in [TEORIA.md](TEORIA.md).

---

## 1. Price data collection

Starting point: Binance, OHLCV (Open, High, Low, Close, Volume) candles on BTC/USDT, 1-minute timeframe. The current history window is **from 2025-05-19 to today** (~1 year, 525,000 candles) — configured in `config/default.yaml` (`data.start_time` + `data.limit`).

A single recent year was chosen based on empirical tests: a much longer history (e.g. since 2021) covers deeply different market cycles (2022 bear, 2023 recovery, 2024 halving) that the model struggles to reconcile into a single distribution, hurting generalization to the present. The current dataset is balanced between trending and ranging phases.

History is stored in Parquet (columnar, compressed). Subsequent runs download only the delta from the last local candle (a seconds-long update).

---

## 2. Log returns

Raw prices are converted to **log returns**: instead of absolute price, the system works on the percentage change between consecutive candles in log scale. Advantages:
- Stationary (no rising trend).
- Symmetric (a +10% and a −10% have the same absolute weight).

The **target** is the sum of log returns over the next **30 candles** (`forecast_horizon: 30` in `config/default.yaml`). The horizon was bumped from 15 to 30 minutes on 2026-05-20: at 15 minutes the edge/cost ratio was structurally unfavourable (avg move ~25 bps vs ~26 bps roundtrip cost); at 30 minutes the expected move roughly doubles (~42 bps) while cost stays constant.

---

## 3. Feature engineering — what the model sees

**104 features** per candle (**86 dynamic + 18 structural**, verified on the current dataset 2026-06-02), designed to give the model the same tools an experienced trader would use. The count is after the **C-funding** filter (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`, decision 2026-05-28): 15 live-incompatible features with ROI ≤ 0 were removed (90d/365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`) — see `MODEL_IMPROVEMENTS.en.md` for the full rationale.

### Trend and momentum
- **Rolling moving averages** over 5/10/20/60 minutes — trend at multiple horizons.
- **Derived momentum** — ratios between MAs at different scales, momentum/volatility ratios. Classic RSI/MACD were removed: the same information is already captured by the `vol_std` + `lag_ret` + microstructure mix.

### Volatility
Rolling std of log returns over 5/10/20/60 min. Classic ATR removed from the inputs (redundant with `vol_std`); still computed by the `RiskManager` for dynamic stop sizing (§10).

### VWAP
Volume-weighted average price. The reference used by funds and market makers. Distance price-vs-VWAP tells whether the market is temporarily over/under-valued relative to the session's equilibrium.

### Volume Profile (multi-scale)
Volume distribution per price level: high-volume levels become strong supports/resistances. Computed on **three windows** (1h, 4h, 1 day). Each window produces 4 features: distance from POC, from Value Area High, from Value Area Low, volume concentration at the POC. In high volatility the short VP dominates; in low volatility the daily VP is more stable.

### CVD — Cumulative Volume Delta
Difference between aggressive buy volume (taker) and aggressive sell volume. When CVD rises while price falls, there's hidden buy pressure — and vice versa. One of the most informative indicators about the market's real intent.

### Microstructure (candle shape)
10 instantaneous features derived from candle geometry: body ratio, upper/lower shadow, price velocity, price acceleration. Capture in real time what MACD/RSI capture only after many candles.

### Funding rate
Funding rate of BTC/USDT perpetual futures downloaded every 8h. Produces 3 structural features: `funding_rate`, `funding_rate_1d`, `funding_rate_dev`. High funding = crowded longs (risk of short squeeze down); negative funding = crowded shorts (risk of squeeze up).

### Temporal features
Time of day, day of week, day of month. Crypto markets have cyclical patterns: liquidity is lower at night, Wall Street open / futures expiries produce systematic moves.

### Lag features
Log returns of the last 5 candles fed directly as features — helps recognize bounces, trend continuations, exhaustions.

### Feature interactions
3 explicit products for regime recognition: `vol_x_pos` (volatility × VWAP position), `momentum_x_funding`, `cvd_x_vol`. Help the model see relevant combinations without learning them implicitly.

---

## 4. Macro data (FRED + yFinance)

External macroeconomic indicators:
- **DXY** — BTC tends to move inversely to the dollar.
- **VIX** — fear in traditional markets spills into crypto.
- **Rates** (Fed Funds Rate, Treasury) — cost of capital and risk appetite.
- **Gold** — correlated with BTC as a safe-haven asset.

These have much lower frequency (daily/monthly): processed separately and merged with price data at training time.

### Regime detection (Markov-Switching on BTC realized volatility, hourly)
Since **2026-06-03** the regime detector runs **directly on BTC data**, no longer on US macros. The active class is `RegimeMarkovBTC` in `quantsys/macro/regime.py` — Markov-Switching (Hamilton 1989) on intraday BTC realized volatility aggregated hourly. It supersedes both the legacy `RegimeMarkovSwitching` on PC1 of daily macros (regimes switching every months — incompatible with h=30) and the transitional `RegimeSession` baseline (deterministic Asia/EU/US, ~33% by construction but informationally empty: temporal clusters, not market clusters).

**Feature pipeline.** From `data/raw_candles.parquet` (1-min BTC) the detector aggregates per hour:
- `log_ret_h` — sum of 1m log-returns over the hour (hourly return);
- `log_rv` — `log(Σ log_ret_1m²)` clipped at 1e-12 (log realized variance; raw rv is heavily right-skewed, the log stabilizes it for MarkovRegression).

**Statistical pipeline.** Global RobustScaler (median/IQR, negligible look-ahead) → expanding-window PCA with `n_pca=1` (collapses `log_ret_h` + `log_rv` into a single motion-intensity signal) → `MarkovRegression` on PC1 with switching mean **+** switching variance → manual Hamilton filter, O(1) per hourly step between retrains. Walk-forward: 30-day burn-in (720h), retrain every 30 days (720h).

**Three regimes that emerge from BTC data (2026-06-03 run on ~9100 hours, post burn-in):**
- **R0 — Quiet** (~42%): low volatility (PC1 σ² ≈ 0.56), drift μ ≈ 0, P(stay) ≈ 89%.
- **R1 — Trending** (~18%): mid volatility (σ² ≈ 0.12), positive drift μ ≈ +0.08, P(stay) ≈ 92% (persistent directional market regime).
- **R2 — Stress** (~40%): high volatility (σ² ≈ 3.79), downside bias μ ≈ −0.12, P(stay) ≈ 79% (shock / dump regime).

Typical switch frequency is **3–8 changes per day**, matching a 1-min model with h=30. The expanding-window PCA explains ~65–73% of the variance between `log_ret_h` and `log_rv` (PC1 captures the magnitude of motion). Probabilities are persisted in `data/regime_probs.parquet` on a UTC hourly index, with the unchanged schema (`regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`) for drop-in compatibility with every consumer (`02_train.py::_load_val_regimes`, dashboard, `RiskManager`).

**How the model consumes the regime.** As before, the regime is **not an input feature**: it is used to (a) stratify the validation split (now the three clusters are market clusters, not session clusters), and (b) feed the `val_nll per regime` diagnostic in the training logs — if one of the three regimes shows systematically worse NLL, the model has a calibration gap on that micro-condition. The US macros (FRED + yFinance) are now decoupled from the regime detector but are still consumed by the 16-dim `MacroEncoder` as before.

**Legacy classes in `quantsys/macro/regime.py`** (kept as optional fallbacks, no longer wired into the pipeline):
- `RegimeMarkovSwitching` — MS on PC1 of daily FRED+yFinance macros.
- `RegimeSession` — deterministic Asia/EU/US baseline on `hour_utc // 8`.
- `RegimeHMM` — legacy Gaussian HMM (MS predecessor).

`RiskManager` keeps applying regime-specific risk profiles (§10): `regime=2` (Stress) is now read as "high vol / dump" and sizing scales down accordingly, while `regime=1` (Trending) sustains full Kelly exposure.

---

## 5. Training dataset

Features are organized into **temporal windows**: every example is a `120×104` matrix (last 120 minutes = 2 hours of context, 104 features). The target is the cumulative log-return over the next 30 candles.

On the current dataset (525k candles, 1 year) with `window_stride: 5` we get **~80,000 train + ~10,000 val + ~10,000 test examples** (exact npz counts 2026-06-02: 80,390 / 10,049 / 10,049).

Normalization with a **global multi-column RobustScaler**, less sensitive to price spikes than the standard scaler. Parameters are persisted in `PipelineState` (`models/{arch}/pipeline_state.pkl`) to reapply the same transform at inference time.

### Critical invariant — z-score vs raw space

The `target_ret` is scaled by the RobustScaler along with the other features. The scale factor (`target_scale`) is computed at runtime as the IQR of the raw target on the training set and persisted in `PipelineState`; it varies with dataset and forecast horizon (e.g. ~0.002707 on the 2026-06-02 run with `data.limit=525k`, `forecast_horizon=30`). Therefore:
- **The model predicts μ, σ, ν in z-score space** (standardized fraction). σ = 1.0 means "one IQR of the target", not "1% of price".
- **The trading layer (`SignalGenerator`, `RiskManager`) operates in raw space**: thresholds `min_expected_ret`, `max_sigma`, SL/TP math assume direct log-return fractions (`σ × price = USD distance`).

Reconciliation via `PipelineState.denormalize_predictions(mu, sigma) -> (mu_raw, sigma_raw)` which multiplies by `target_scale`. **Both `03_backtest.py` and `04_live_signals.py` apply it right after the forward pass**, before handing predictions to the `SignalGenerator`. Without denormalization, SL/TP `σ × price × multiplier` become macroscopic (σ_z=1 × $42k × 1.5 = $63k) — structural bug identified and fixed on 2026-05-23 (Sharpe −256 → +18.7).

**For any new entry point**: always call `denormalize_predictions` before interpreting μ/σ. Safety nets against regressions:
- `RuntimeError` in `03_backtest.py` if `σ_max ≥ 0.05` (impossible in raw space on BTC 1m; `raise` instead of `assert` survives `python -O`).
- Runtime warning in `_sl_tp` if `σ × price × 1.5 > 5% × price`.
- `PipelineState.forecast_horizon` validated in backtest + live: if `cfg.data.forecast_horizon != state.forecast_horizon` → `RuntimeError` (prevents using a h=30 model with a h=15 backtest).
- `merge_asof` between test set and raw_candles validated with `len(merged) == n_test_orig`, otherwise `RuntimeError` (prevents SL/TP triggered on wrong candles due to Binance gaps/halts).
- `update_trailing` updates `portfolio.equity` mark-to-market every candle (cash + size_usd + unrealized_pnl): the circuit breaker fires on intra-trade DD in live too.
- Floor `sl_d = max(sl_d, price × 1e-4)` in `_sl_tp` to avoid silent SL=TP=entry when ATR=0 (market halt).

---

## 6. Available architectures

Four architectures selectable via `--arch`: `lstm`, `itransformer`, `tcnmamba`, `nhits`. The heterogeneous ensemble composition is configurable in `config/default.yaml` → `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba; LSTM available but outside the ensemble due to structural under-performance).

### LSTM dual-stream (`--arch lstm`, legacy)
Recurrent net with the 104 features split into two streams:
- **Dynamic stream** (86 features): log returns, CVD, volume delta, microstructure — short-term momentum. Processed by an LSTM.
- **Structural stream** (18 features): VWAP, Volume Profile (short + mid), temporal features, ATH/ATL 30d, momentum_30d, funding rate — market context. Processed by a GRU.

The two streams are fused and passed through **temporal attention** (weighs the window's candles differently: some are more informative, e.g. the most recent one or the one at −15 min). Training ~30–60 min on RTX 2070 Super. Removed from the ensemble on 2026-05-14 after val_NLL 5.28 vs iTransformer 0.18.

### iTransformer (`--arch itransformer`)
**Inverted** Transformer: instead of attention on timesteps, attention on **features** (each feature becomes a "token"). With 104 features, complexity O(104²)≈10,800 vs O(120²)=14,400 of the classic Transformer — better suited to tabular data because it explicitly models inter-feature correlations.

**Multi-scale** embedding: the 120-min window is compressed into 3 views (1m, 5m, 15m) via average pooling, capturing both fast and slow structures without doubling parameters.

### TCN+Mamba hybrid (`--arch tcnmamba`)
Two parallel branches for local patterns (5–15 candles) and long context (120 candles):
- **TCN** (Temporal Convolutional Network): six blocks of causal convolution with growing dilations (1, 2, 4, 8, 16, 32) → receptive field **127 candles** (1 + 2·(1+2+4+8+16+32)), covers the entire input window. Captures technical figures (double tops, breakouts, consolidations). Output: global average over time.
- **Mamba** (State Space Model): hidden state evolving by discrete differential equations with **input-dependent** parameters — the model decides at each step how much to remember. Dynamic information selection over 120 candles without attention's quadratic overhead. Pure PyTorch (no external deps). **Vectorized** scan via `cumprod` + `cumsum` in 32-step chunks (AMP disabled in inference to avoid NaN on spectral_norm + Mamba edge cases, see `quantsys/model/ensemble.py`). Forward+backward speedup ~1.8× vs initial sequential scan.
- **Learned gated fusion**: `σ(W·[tcn; mamba])` learns how much weight to give local vs global per example.

With `d_model=128`, training ~40–70 min on RTX 2070 Super (~2.5 GB VRAM).

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

Very fast training (~10–15 min on RTX 2070 Super vs 25 min for iTransformer).

### Probabilistic output (common to all)
Not a single number, but **the parameters of a distribution**: mean μ (direction), σ (uncertainty), ν (heavy-tails parameter — how likely extreme moves are). The system knows not just the direction but its own confidence.

Output in **z-score space** (target_ret normalized by the global RobustScaler, §5). Explicitly denormalized via `PipelineState.denormalize_predictions()` before the trading layer.

---

## 7. Training

### Loss — t-Student NLL
Penalizes the model when the predicted distribution is far from the observed value. **Student-t** instead of Gaussian: financial returns have heavier tails (crashes and rallies happen more often than a Gaussian would predict).

### Asymmetric penalty
Extra penalty when the model gets the direction wrong (says "up" but it goes down). Sign errors cost more than magnitude errors: a position in the wrong direction loses money, while underestimating the magnitude only hurts returns.

### CRPS
Continuous Ranked Probability Score — auxiliary **calibration** metric: if the model says "80% probability", it should be right ~80% of the time. A model that always says "95%" but is right 60% is dangerous due to overconfidence in trading.

### Walk-forward validation
No simple train/test split: the model is trained on a historical window, tested on the immediately following period (never seen), then the window slides forward. Simulates the real-world deployment and avoids look-ahead bias.

### Knowledge Distillation (alternative to homogeneous ensemble 5× same arch)

**Phase 2a — Candidate training**: archs in `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba) trained normally with `n_ensemble=1`. A single config line changes the composition.

**Phase 2b — Multi-Teacher Scoring**: every model evaluated at its best epoch with normalized scoring (min-max across archs): **40% val_loss + 35% Spearman + 25% directional accuracy**. All of them contribute as teachers with softmax weights (temperature=2) proportional to the score — not a single teacher.

**Phase 2c — Student with transfer + distillation**: every model is retrained as "student" with three advantages:
- Output-head weights (μ, σ, ν) copied from the best teacher — calibrated start instead of random.
- Mixed loss: **70% real NLL + 30% distillation**, normalized by the variance of each teacher component (μ~1e-5, ν~5 have different scales → equal contribution).
- Soft labels weighted across all teachers, integrated into the `TensorDataset` (shuffle-safe): each batch contains both real data and teacher predictions for the same samples.

Students converge in ~60% of the normal epochs. Already-distilled students are recognized and skipped automatically.

**Heterogeneous ensemble (inference)**: the N architectures predict together. Errors tend to be uncorrelated because they capture different patterns (N-HiTS hierarchical multi-scale, TCNMamba local + long context, iTransformer inter-feature correlations). Combination = **weighted mean** with `DEFAULT_ARCH_WEIGHTS` (`ensemble.py`):
- `mu_ens = Σ w_i · mu_i` (reduces error variance)
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)` (law of total variance: accounts both for average model uncertainty and for disagreement between point predictions)

The ensemble returns (μ, σ, ν) directly in natural space, no intermediate numerical conversions.

---

## 8. Monte Carlo

For every new candle, the system generates **2000 alternative price scenarios** (`mc.n_paths` in config) over the next 30 minutes, using the ensemble's predictions as a guide and adding stochastic variability calibrated on current volatility.

Volatility is estimated with a **GJR-GARCH(1,1)** model (default params `omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875` in `quantsys/model/forecast.py`): the GJR variant adds an asymmetry term that amplifies the volatility update in response to negative shocks (leverage effect), typical of financial markets.

The result is a "fan" of scenarios with confidence intervals. It lets the system answer questions like:
- What's the probability that price is above $X in 30 minutes?
- What's the worst case in the bottom 5%?

---

## 9. Signal generation

The operating signal (BUY / SELL / HOLD) combines multiple elements:

**Conviction score** — direction predicted by the ensemble, magnitude of the expected move, prediction uncertainty. High conviction requires (a) clear direction, (b) expected move exceeding commissions (0.1% per side), (c) low uncertainty.

**Quality filters**:
- Expected return > minimum threshold (to cover commissions)
- Predicted volatility not too high (chaotic market → skip)
- BTC regime compatible (e.g. R2 Stress → tighter entry thresholds; R1 Trending → full Kelly)

---

## 10. Risk management

**Kelly sizing**: size proportional to the estimated statistical edge and inversely proportional to variance. Strong signal + calm market → larger risk; weak signal + high vol → smaller risk. Max risk per trade: 1% of capital.

**Dynamic ATR stop loss**: not a fixed %, ATR-based. Volatile market → wider stop (no stop-out by noise). Calm market → tighter stop (limits the loss).

**Trailing stop**: once in profit, the stop rises with price protecting gains. Trailing distance also proportional to current ATR.

**Circuit breaker**: if drawdown exceeds **15%** of capital (`risk.max_drawdown_stop` in config), the system stops opening new positions. Last-resort protection against prolonged losing streaks (possible structural market change not trained on). DD computed **mark-to-market every candle** (cash + size_usd + unrealized_pnl, updated in `update_trailing`): in live it fires even if a single position has large unrealized losses, without waiting for close. Auto-recovery when DD goes back below 70% of the threshold (e.g. <10.5% with 15% threshold).

---

## 11. Live execution

In live mode the system (`LiveEngine` in `scripts/04_live_signals.py`) connects to Binance via WebSocket and receives every closed candle in real time. For each candle:
1. Updates features with the training-time normalization.
2. Passes the last 120-minute sequence to the model.
3. Generates the Monte Carlo simulations.
4. Computes the conviction score.
5. If the signal passes filters, opens/closes a position (**paper trading**, no real orders).
6. Updates portfolio state and writes the signal to disk.

Every hour, the macro variables snapshot is refreshed in the background to keep the macro context up to date without blocking the live feed.

### ⚠ Current status: BLOCKER #1 (paper-only, not operational)

The `LiveEngine` has a known mismatch with the trained model:
- **Live** produces **39 features** via `LiveFeatureBuffer._compute_features` (rebuilt by hand to avoid the heavy warmup of the training pipeline).
- **Training** trained the model on **104 features** (post C-funding filter) with a fixed order and a global RobustScaler.

Three overlapping mismatches: (1) count 39 vs 104, (2) order/semantics (live starts at `log_ret`, training at `open`), (3) normalization (live uses per-window median+IQR, training uses the `pipeline_state`'s global RobustScaler). Stage 2 (104-feat dataset regeneration) and Stage 3 (distill retrain) were completed on 2026-06-02; **Stage 4** (live engine rewrite using `FeatureBuilder` + 30d buffer + funding poll) and **Stage 5** (parity test) remain to fully close BLOCKER #1.

`scripts/99_replay_live_vs_training.py` verifies this programmatically. Paper-trading signals do **NOT** reflect the backtest until the feature space is aligned (decision 2026-05-28: option C-minimal or C-funding documented in `MODEL_IMPROVEMENTS.md` §"Allineamento feature live↔training"). Mandatory validation before operational live: parity test (live vector == FeatureBuilder on a historical window) + replay backtest.

### 24/7 operational robustness

The `LiveEngine` implements several safety nets for always-on systems:

- **Dynamic lookback buffer**: sized to `max(window_size + 60, max_rolling_window + 60) = 260` candles — guarantees full warmup for all rolling features (e.g. `price_vs_ma200m` on 200 candles). Pre-2026-05-24 the buffer was 180 and this feature was silently always zero in live.
- **Forming vs closed candle separation**: only candles with `k.x == True` (closed kline) enter the buffer; partial ones live in a separate `_pending_candle` and are dropped on WS reconnect. Prevents warmup corruption after disconnections.
- **Funding thread safety**: the daemon thread refreshing funding every 8h writes `_funding_df` under `threading.Lock()`. First update executed immediately at startup (no 8h wait on possibly stale parquet).
- **Windows-tolerant log rotation**: rotation at 50 MB wrapped in `try/except` for `OSError, PermissionError` — proceeds without rotating if the file is temporarily locked.
- **forecast_horizon mismatch**: `LiveEngine.__init__` raises `RuntimeError` if `cfg.data.forecast_horizon != PipelineState.forecast_horizon`, preventing live startup with a model trained for a different horizon.
- **Atomic checkpoints**: `EarlyStopping` saves weights to `.tmp` + `os.replace()` (cross-platform atomic rename), avoiding corrupted checkpoints if the process is killed during a save.

---

## Flow summary

```
Binance REST/WS
      │
      ▼
1m OHLCV candles (history 2025-05-19 → today, ~1 year, 525k candles)
      │
      ▼
Log returns + 104 features (VWAP, VP short/mid, CVD, momentum,
                              microstructure, funding, interactions, time, lag)
      │
      ├─── Macro features (FRED + yFinance) → MacroEncoder 16-dim embedding
      │    BTC 1m → hourly realized vol → RegimeMarkovBTC (Markov-Switching,
      │                                                    3 data-driven regimes:
      │                                                    Quiet / Trending / Stress)
      │
      ▼
Normalized 120×104 windows (global RobustScaler) → NPZ dataset
      │
      ▼
Architecture selected by --arch (or --distill for the ensemble):
      │
      ├─ lstm         → LSTM dual-stream (dyn. + struct.) + attention   (legacy)
      ├─ itransformer → attention over features (multi-scale 1m/5m/15m)
      ├─ tcnmamba     → TCN (dil. 1-32, RF=127) + Mamba SSM (context 120)
      │                  └─ gated fusion → unified representation
      ├─ nhits        → Neural Hierarchical Interpolation (pure-MLP, stacks 8/4/1)
      │
      ├─ [--distill] Multi-Teacher Knowledge Distillation:
      │              archs from config/default.yaml → scoring → soft labels
      │              shuffle-safe → student with 60% epochs
      │
      ▼
Output: μ (direction) + σ (uncertainty) + ν (heavy tails)  in z-score
      │
      ▼
PipelineState.denormalize_predictions(μ, σ)  →  raw space
      │
      ▼
Monte Carlo 2000 scenarios × 30 min (GJR-GARCH for volatility)
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
