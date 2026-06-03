# QUANTSYS — Neural Forecasting Engine for BTC/USDT

End-to-end algorithmic trading system that combines **deep learning forecasting**, **probabilistic risk management**, and **multi-teacher knowledge distillation** to generate directional signals on BTC/USDT 1-minute candles.

**Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API

> Documentation is bilingual. Italian: [README.it.md](README.it.md) · [AVVIO.md](AVVIO.md) · [TEORIA.md](TEORIA.md) · [MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md). English: [AVVIO.en.md](AVVIO.en.md) · [TEORIA.en.md](TEORIA.en.md) · [MODEL_IMPROVEMENTS.en.md](docs/MODEL_IMPROVEMENTS.en.md).

> ⚠ **Live engine status:** paper-only (no real orders). A known feature mismatch (BLOCKER #1: 39 live features vs 104 training features, see `TEORIA.en.md` §11) currently makes live predictions uncorrelated with the backtest. Stage 2-3 of the alignment plan done (2026-06-02); Stage 4-5 (live engine rewrite + parity test) pending — see `MODEL_IMPROVEMENTS.en.md`.

---

## Key Features

- **3-architecture heterogeneous ensemble** (default post 2026-05-14):
  - **iTransformer** — attention over features (not time), multi-scale embedding (1m/5m/15m), O(F²)
  - **N-HiTS** — pure-MLP hierarchical interpolation, multi-scale pooling stacks (replaces LSTM)
  - **TCN+Mamba** hybrid — dilated causal convolutions (receptive field 127) + State Space Model with input-dependent parameters and gated fusion
  - **LSTM+GRU** — dual-stream with temporal attention (legacy, kept for backward compat; under-performing, see `CHANGELOG.md`)

- **Multi-teacher Knowledge Distillation** (`--distill`): trains all archs, scores them (40% val_loss + 35% Spearman + 25% directional accuracy), weights them via softmax(temperature=2), and retrains students with weighted soft labels from all teachers + output-head transfer + scale-normalized mixed loss.

- **Heterogeneous ensemble** combines structurally diverse models at inference via the Law of Total Variance (weighted):
  - `mu_ens = Σ w_i · mu_i`
  - `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)`
  Default weights in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`).

- **104 engineered features** (86 dynamic + 18 structural, post C-funding filter): VWAP + Volume Profile (short + mid scales), CVD, candle microstructure (body ratio, shadows, price velocity/acceleration), funding rate, multi-window volatility (5/10/20/60), lag returns, time encoding (sin/cos hour-of-day, day-of-week, sessions), feature interactions, structural levels (ATH/ATL 30d, momentum_30d, round-level distance, MA200m). The 15 live-incompatible features (90d/365d/long-lookback, frac-diff, vp_*_long) are filtered out — see `MODEL_IMPROVEMENTS.en.md` for the rationale.

- **Probabilistic output** — t-Student NLL with asymmetric penalty + CRPS calibration + Direction-Value joint loss (`dv_lambda=0.3`): each prediction is a full distribution (μ, σ, ν), not a point estimate. Output is in **z-score space** (RobustScaler-normalized target_ret); the trading layer operates in **raw space** — denormalization is centralized via `PipelineState.denormalize_predictions()` (see `TEORIA.en.md` §5).

- **Monte Carlo simulation** — 2000 GJR-GARCH(1,1) guided scenarios, 30-min horizon (params `omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875`).

- **Risk management** — fractional Kelly sizing, dynamic ATR stop-loss, trailing stop, 15% mark-to-market drawdown circuit breaker (intra-trade).

- **BTC regime detection** — `RegimeMarkovBTC` (Markov-Switching, Hamilton 1989) fit on hourly BTC realized volatility (`log_ret_h` + `log_rv` from `raw_candles.parquet`, PCA n_pca=1, switching mean+variance, walk-forward burn-in 30d / retrain 30d). **3 data-driven regimes** that emerged on ~9100 hours: R0 Quiet (~42%, low vol, drift 0), R1 Trending (~18%, mid vol, +drift, P(stay) 92%), R2 Stress (~40%, high vol, downside bias, P(stay) 79%). Switches 3–8 times/day, matching the h=30 horizon. Default since 2026-06-03 — supersedes the prior US-macro Markov-Switching (daily timescale, single-cluster collapse) and the transitional `RegimeSession` baseline (Asia/EU/US, informationally empty). Used for stratified val + `val_nll per regime` diagnostic; raw US macros (DXY, VIX, rates, gold) still feed `MacroEncoder` (16-dim), decoupled from the regime detector. `RegimeMarkovSwitching` and `RegimeSession` remain in the codebase as fallbacks.

- **Live paper trading** — Binance WebSocket feed with state persistence, automatic reconnection (exponential backoff), incremental Volume Profile updates, thread-safe funding refresh.

- **Walk-forward validation** — purged k-fold with embargo, no look-ahead bias.

- **Backtest engine** with stress testing (fee×2 slip×3 pessimistic; fee×1.5 slip×5 flash crash), bootstrap CI 5000 iter, regime-conditioned analysis, MDD recovery stats.

---

## Quick Start

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

# 2. Verify environment
python scripts/00_check_setup.py

# 3. Full pipeline (interactive menu)
python run_all.py

# 4. Or specify architecture directly
python run_all.py --arch itransformer
python run_all.py --arch nhits
python run_all.py --arch tcnmamba
python run_all.py --arch lstm           # legacy, backward compat only

# 5. Knowledge Distillation (train all + multi-teacher weighted + distill students)
python run_all.py --distill

# 6. Diagnostics
python scripts/07_verify_teacher.py             # compare architectures
python scripts/99_replay_live_vs_training.py    # BLOCKER #1 diagnostic
```

The pipeline downloads BTC/USDT history from Binance (default: 1 year from 2025-05-19, ~525k candles), engineers 104 features, trains the selected model, runs backtest with stress tests, starts the live WebSocket feed, and opens the dashboard at `http://localhost:8050`.

---

## Architecture

```
Binance REST/WS
      │
      ▼
1m OHLCV candles (default: 2025-05-19 → today, ~525k rows)
      │
      ▼
Feature Engineering: 104 features (VWAP, VP short/mid, CVD, microstructure,
                                   funding, time, lag, interactions)
      │
      ├─── Macro data (FRED + yFinance) → MacroEncoder 16-dim embedding
      │    BTC 1m → hourly RV → RegimeMarkovBTC (Markov-Switching, 3 regimes:
      │                                          Quiet / Trending / Stress)
      │
      ▼
Sliding windows 120×104 (2h context) → normalized dataset (RobustScaler)
      │
      ▼
Architecture (selectable):
      │
      ├─ itransformer → feature-wise attention, multi-scale (1m/5m/15m)
      ├─ nhits        → hierarchical pure-MLP (stacks 8/4/1)
      ├─ tcnmamba     → dilated TCN (RF=127) + Mamba SSM, gated fusion
      ├─ lstm         → LSTM+GRU dual-stream + temporal attention (legacy)
      │
      ├─ [--distill]  Multi-teacher Knowledge Distillation: scoring → weighted
      │                soft labels (shuffle-safe) → students with 60% epochs
      │
      ▼
Output: μ (direction) + σ (uncertainty) + ν (tail weight)   in z-score space
      │
      ▼
PipelineState.denormalize_predictions(μ, σ)   →   raw space
      │
      ▼
Monte Carlo: 2000 GJR-GARCH(1,1) scenarios × 30 min
      │
      ▼
Conviction score (direction × magnitude × calibration × regime)
      │
      ▼
Risk Manager (Kelly sizing, ATR stop, trailing, 15% MtM circuit breaker)
      │
      ▼
BUY / SELL / HOLD  +  size  +  stop loss  +  take profit
```

---

## Project Structure

```
quantsys_project/
├── config/
│   ├── default.yaml              shared parameters (data, features, model, training, risk, distillation)
│   ├── secrets.yaml.example      template for API keys (copy to secrets.yaml)
│   └── arch/
│       ├── lstm.yaml             LSTM architecture overrides
│       ├── itransformer.yaml     iTransformer overrides
│       ├── nhits.yaml            N-HiTS overrides
│       └── tcnmamba.yaml         TCN+Mamba overrides
├── quantsys/                     installable Python package
│   ├── data/                     Binance REST + WebSocket + funding rate
│   ├── features/                 FeatureBuilder (104 features post C-funding, dual-stream split)
│   ├── macro/                    FRED + yFinance + RegimeMarkovBTC (Markov-Switching on BTC realized vol)
│   ├── model/
│   │   ├── __init__.py           QuantLSTM, QuantTFT, QuantiTransformer
│   │   ├── nhits.py              QuantNHiTS (pure-MLP hierarchical)
│   │   ├── tcn_mamba.py          QuantTCNMamba (TCN branch + Mamba SSM + gated fusion)
│   │   ├── ensemble.py           EnsembleModel (homogeneous / heterogeneous, AMP off in inference)
│   │   ├── distillation.py       Multi-teacher knowledge distillation
│   │   ├── forecast.py           Monte Carlo GJR-GARCH(1,1) + neural-guided
│   │   └── revin.py              Reversible Instance Normalization (optional)
│   ├── trading/                  Kelly sizing, dynamic SL, trailing, circuit breaker
│   └── utils/                    config loader, device setup, logging, PipelineState
├── scripts/
│   ├── 00_check_setup.py         verify CUDA, dependencies, Binance connection
│   ├── 00_test_binance_testnet.py  testnet API connectivity check
│   ├── 01_download_data.py       Binance → 104 features (C-funding) → lstm_dataset.npz
│   ├── 01_update_data.py         incremental update (delta candles only)
│   ├── 01b_download_macro.py     FRED + yFinance → dataset update; runs RegimeMarkovBTC on BTC candles
│   ├── 02_train.py               training with --arch, --distill, ensemble
│   ├── 02b_walkforward_validate.py   walk-forward purged k-fold with embargo
│   ├── 02c_optuna_search.py      Bayesian hyperparameter search (LSTM only)
│   ├── 03_backtest.py            backtest + stress test + bootstrap CI
│   ├── 04_live_signals.py        WebSocket live feed + paper trading
│   ├── 05_analyze_signals.py     live session analysis
│   ├── 06_dashboard.py           HTTP dashboard server (Dash)
│   ├── 07_verify_teacher.py      architecture comparison for teacher selection
│   └── 99_replay_live_vs_training.py   BLOCKER #1 diagnostic (live vs training feature parity)
├── tests/                        pytest suite (features, NLL, PipelineState, recent fix regressions)
├── dashboard/                    React dashboard (artifact for claude.ai)
├── data/                         generated (gitignored)
├── models/                       checkpoints per architecture (gitignored)
├── results/                      backtest + live signals per architecture (gitignored)
└── logs/                         rotating logs (gitignored)
```

See [AVVIO.en.md](AVVIO.en.md) for the full operational guide and [TEORIA.en.md](TEORIA.en.md) for theoretical foundations.

---

## Knowledge Distillation Pipeline

```bash
python run_all.py --distill
```

1. **Phase 2a** — Trains all architectures listed in `config/default.yaml → distillation.archs` (default: iTransformer + N-HiTS + TCN+Mamba) independently with `n_ensemble=1`. Skips an arch if `models/{arch}/best_model.pt` already exists; force retrain with `--force-download`.
2. **Phase 2b** — Multi-Teacher Scoring: every model scored at its best validation epoch with normalized scoring (40% val_loss + 35% Spearman ρ + 25% directional accuracy). Weights via softmax(temperature=2). Top score becomes the primary teacher; the others remain in the pool as weighted teachers.
3. **Phase 2c** — Retrains each model as student with:
   - Output head weight transfer (μ, σ, ν heads) from best teacher
   - Scale-normalized mixed loss `(1−α)·NLL_real + α·distill_loss` with α=0.3, normalized by per-component teacher variance (μ~1e-5, ν~5 contribute equally)
   - Weighted soft labels from all teachers integrated in `TensorDataset` (shuffle-safe)
   - 60% epoch reduction (accelerated convergence)
   - Auto-skip if student already distilled (`config.json` contains `distilled: true`)

Change the ensemble composition by editing `distillation.archs` in `config/default.yaml`. See [AVVIO.en.md](AVVIO.en.md) for examples and operational details.

---

## Hardware

Reference setup: **RTX 2070 Super (8 GB VRAM)**.

| Parameter | Value | Source |
|---|---|---|
| Training batch size | 64 (default) | `config/default.yaml` |
| Inference batch (backtest) | 256 | `scripts/03_backtest.py` |
| AMP fp16 training | yes | via `setup_device` |
| AMP inference | **off** (hardcoded) | `quantsys/model/ensemble.py:170` (avoids NaN from spectral_norm + Mamba scan) |
| `hardware.cudnn_benchmark` | true | optimized kernels for fixed shapes |
| `hardware.pin_memory` | true | zero-copy RAM → VRAM transfer |
| TCN+Mamba VRAM | ~2.5 GB | `d_model=128`, 4 TCN blocks + 3 Mamba layers |

CPU-only fallback works (with 20-50× slowdown on training; full-speed on backtest/live). Apple Silicon (MPS) untested. See [AVVIO.en.md](AVVIO.en.md) for low-VRAM (4GB) and high-VRAM (≥16GB) tuning.

---

## FRED API key (optional)

1. Free registration: https://fred.stlouisfed.org/docs/api/api_key.html
2. Copy `config/secrets.yaml.example` to `config/secrets.yaml` and add your key.
3. Without a key: works under stricter rate limits (~120 req/min).

`config/secrets.yaml` is gitignored — it never gets committed.

---

## Development & Tests

```bash
pytest tests/                          # full suite
pytest tests/test_recent_fixes.py -v   # regression tests on critical fixes (z-score, RevIN, BLOCKER #1)
```

The test suite covers FeatureBuilder no-leakage, RobustScaler invariants, t-Student NLL gradient checks, PipelineState (de)serialization, Kelly sizing edge cases, circuit-breaker triggers, and golden snapshots of the live feature buffer.

---

## License

[MIT License](LICENSE) — research code, not financial advice.
