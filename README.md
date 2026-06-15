# QUANTSYS — Motore neurale di forecasting per BTC/USDT · QUANTSYS — Neural Forecasting Engine for BTC/USDT

🇮🇹 Sistema di trading algoritmico end-to-end che combina **deep learning forecasting**, **gestione probabilistica del rischio** e **knowledge distillation multi-teacher** per generare segnali direzionali su candele BTC/USDT. **Timeframe corrente: 1 ora** (design interval-agnostic; perimetro 1m in backup). Stato 2026-06-11: il filone direzionale è negativo OOS su 1m E 1h (pivot 1h killed); la famiglia **vol** ha prodotto l'unico segnale validato del progetto — `log_rv` batte HAR-RV del 30% in QLIKE su test a 1h (FAIL a 1m: risoluzione-specifico), mentre l'asimmetria firmata `log_rs_ratio` è impredicibile (FAIL 2026-06-11) → **i momenti pari generalizzano OOS, i dispari no**. Config corrente: `features.target_type: log_rs_ratio` (ultimo probe). Vedi `docs/MODEL_IMPROVEMENTS.md` e `STATUS.md`.

**EN** End-to-end algorithmic trading system that combines **deep learning forecasting**, **probabilistic risk management**, and **multi-teacher knowledge distillation** to generate directional signals on BTC/USDT candles. **Current timeframe: 1 hour** (interval-agnostic design; the 1m perimeter is backed up). Status 2026-06-11: the directional axis is OOS-negative at both 1m AND 1h (1h pivot killed); the **vol** family produced the project's only validated signal — `log_rv` beats HAR-RV by 30% in test QLIKE at 1h (FAIL at 1m: resolution-specific), while the signed asymmetry `log_rs_ratio` is unpredictable (FAIL 2026-06-11) → **even moments generalize OOS, odd ones don't**. Current config: `features.target_type: log_rs_ratio` (latest probe). See `docs/MODEL_IMPROVEMENTS.md` and `STATUS.md`.

🇮🇹 **Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API

**EN** **Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API

🇮🇹 > Documentazione **bilingue in un unico file** (IT + EN per paragrafo, marker 🇮🇹/**EN**): [README.md](README.md) · [AVVIO.md](AVVIO.md) · [TEORIA.md](TEORIA.md) · [docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md).

**EN** > **Single-file bilingual** documentation (IT + EN per paragraph, markers 🇮🇹/**EN**): [README.md](README.md) · [AVVIO.md](AVVIO.md) · [TEORIA.md](TEORIA.md) · [docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md).

🇮🇹 > ✅ **Stato live engine:** paper-only (nessun ordine reale), ma **BLOCKER #1 RISOLTO (2026-06-05)**. Il path live costruisce ora le **104 feature canoniche** via `FeatureBuilder` (single source of truth) con lo scaler del training, più un catch-up REST contiguo al boot — con **parity feature *e* segnale bit-perfect** vs backtest (`tests/test_live_training_parity.py`, replay Δ=0). I segnali live ora riflettono il backtest; lo smoke test live passa. Vedi `TEORIA.md` §11. ⚠ Nota: il backtest è negativo out-of-sample — il paper-trading serve ad accumulare trade forward reali, senza aspettativa di Sharpe>0 a priori.

**EN** > ✅ **Live engine status:** paper-only (no real orders), but **BLOCKER #1 RESOLVED (2026-06-05)**. The live path now builds the **104 canonical features** via `FeatureBuilder` (single source of truth) with the training scaler, plus a contiguous REST catch-up at boot — achieving **bit-perfect feature *and* signal parity** vs the backtest (`tests/test_live_training_parity.py`, replay Δ=0). Live signals now reflect the backtest; the live smoke test passes. See `TEORIA.md` §11. ⚠ Note: the backtest itself is negative out-of-sample — paper-trading is for accumulating real forward trades, with no a-priori expectation of Sharpe>0.

---

## Caratteristiche principali · Key Features

🇮🇹
- **Ensemble eterogeneo a 3 architetture** (default post 2026-05-14):
  - **iTransformer** — attention sulle feature (non sul tempo), embedding multi-scala (pooling ×1/×5/×15 barre), O(F²)
  - **N-HiTS** — interpolazione gerarchica pure-MLP, stack pooling multi-scala (sostituisce LSTM)
  - **TCN+Mamba** ibrido — convoluzioni causali dilatate (campo recettivo 127) + State Space Model con parametri input-dipendenti e fusion gated
  - **LSTM+GRU** — dual-stream con attention temporale (legacy, backward compat; sotto-performante, vedi `CHANGELOG.md`)

**EN**
- **3-architecture heterogeneous ensemble** (default post 2026-05-14):
  - **iTransformer** — attention over features (not time), multi-scale embedding (×1/×5/×15-bar pooling), O(F²)
  - **N-HiTS** — pure-MLP hierarchical interpolation, multi-scale pooling stacks (replaces LSTM)
  - **TCN+Mamba** hybrid — dilated causal convolutions (receptive field 127) + State Space Model with input-dependent parameters and gated fusion
  - **LSTM+GRU** — dual-stream with temporal attention (legacy, kept for backward compat; under-performing, see `CHANGELOG.md`)

🇮🇹
- **Knowledge Distillation multi-teacher** (`--distill`): addestra tutte le architetture, le valuta (40% val_loss + 35% Spearman + 25% directional accuracy), le pesa via softmax(temperature=2), poi riadestra gli student con soft labels pesate da tutti i teacher + transfer delle output heads + loss mista scala-normalizzata.

**EN**
- **Multi-teacher Knowledge Distillation** (`--distill`): trains all archs, scores them (40% val_loss + 35% Spearman + 25% directional accuracy), weights them via softmax(temperature=2), and retrains students with weighted soft labels from all teachers + output-head transfer + scale-normalized mixed loss.

🇮🇹
- **Ensemble eterogeneo** combina modelli strutturalmente diversi in inferenza via Legge della Varianza Totale (pesata):
  - `mu_ens = Σ w_i · mu_i`
  - `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)`
  Pesi default in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`).

**EN**
- **Heterogeneous ensemble** combines structurally diverse models at inference via the Law of Total Variance (weighted):
  - `mu_ens = Σ w_i · mu_i`
  - `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)`
  Default weights in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`).

🇮🇹
- **104 feature ingegnerizzate** (86 dinamiche + 18 strutturali, post filtro C-funding): VWAP + Volume Profile (short + mid), CVD, microstructure delle candele (body ratio, shadow, price velocity/acceleration), funding rate, volatilità multi-finestra (5/10/20/60), lag returns, encoding temporale (sin/cos hour-of-day, day-of-week, sessioni), feature interactions, livelli strutturali (ATH/ATL 30d, momentum_30d, distanza da round level, MA200m). Le 15 feature live-incompatibili (90d/365d/long-lookback, frac-diff, vp_*_long) sono filtrate — vedi `MODEL_IMPROVEMENTS.md` per il razionale.

**EN**
- **104 engineered features** (86 dynamic + 18 structural, post C-funding filter): VWAP + Volume Profile (short + mid scales), CVD, candle microstructure (body ratio, shadows, price velocity/acceleration), funding rate, multi-window volatility (5/10/20/60), lag returns, time encoding (sin/cos hour-of-day, day-of-week, sessions), feature interactions, structural levels (ATH/ATL 30d, momentum_30d, round-level distance, MA200m). The 15 live-incompatible features (90d/365d/long-lookback, frac-diff, vp_*_long) are filtered out — see `MODEL_IMPROVEMENTS.md` for the rationale.

🇮🇹
- **Output probabilistico** — t-Student NLL con penalità asimmetrica + calibrazione CRPS + Direction-Value joint loss (`dv_lambda=0.3`): ogni predizione è una distribuzione completa (μ, σ, ν), non una stima puntuale. Output in **spazio z-score** (target_ret normalizzato dal RobustScaler); il trading layer opera in **spazio raw** — denormalizzazione centralizzata via `PipelineState.denormalize_predictions()` (vedi `TEORIA.md` §5).

**EN**
- **Probabilistic output** — t-Student NLL with asymmetric penalty + CRPS calibration + Direction-Value joint loss (`dv_lambda=0.3`): each prediction is a full distribution (μ, σ, ν), not a point estimate. Output is in **z-score space** (RobustScaler-normalized target_ret); the trading layer operates in **raw space** — denormalization is centralized via `PipelineState.denormalize_predictions()` (see `TEORIA.md` §5).

🇮🇹
- **Simulazione Monte Carlo** — 2000 scenari GJR-GARCH(1,1) guidati, orizzonte 30 barre (30h a 1h; params `omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875`, da ri-stimare su rendimenti 1h — non sul critical path del backtest).

**EN**
- **Monte Carlo simulation** — 2000 GJR-GARCH(1,1) guided scenarios, 30-bar horizon (30h at 1h; params `omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875`, to be re-estimated on 1h returns — not on the backtest critical path).

🇮🇹
- **Gestione del rischio** — Kelly frazionario, stop-loss ATR dinamico, trailing stop, circuit breaker drawdown mark-to-market 15% (intra-trade).

**EN**
- **Risk management** — fractional Kelly sizing, dynamic ATR stop-loss, trailing stop, 15% mark-to-market drawdown circuit breaker (intra-trade).

🇮🇹
- **Rilevamento regimi BTC** — `RegimeMarkovBTC` (Markov-Switching, Hamilton 1989) fittato sulla realized volatility oraria di BTC (`log_ret_h` + `log_rv` da `raw_candles.parquet`, PCA n_pca=1, switching mean+variance, walk-forward burn-in 30gg / retrain 30gg). **3 regimi data-driven** emersi su ~9100 ore: R0 Quiet (~42%, bassa vol, drift 0), R1 Trending (~18%, mid vol, drift +, P(stay) 92%), R2 Stress (~40%, alta vol, bias ribasso, P(stay) 79%). Switch 3–8 volte/giorno, allineati a h=30. Default dal 2026-06-03 — sostituisce sia il Markov-Switching su macro USA daily (regimi mensili, collasso a singolo cluster) sia la baseline transitoria `RegimeSession` Asia/EU/US (informativamente vuota). Usato per stratificazione val + diagnostica `val_nll per regime`; macro USA (DXY, VIX, tassi, oro) continuano ad alimentare `MacroEncoder` (16-dim) scollegato dal regime detector. `RegimeMarkovSwitching` e `RegimeSession` restano come fallback opzionali.

**EN**
- **BTC regime detection** — `RegimeMarkovBTC` (Markov-Switching, Hamilton 1989) fit on hourly BTC realized volatility (`log_ret_h` + `log_rv` from `raw_candles.parquet`, PCA n_pca=1, switching mean+variance, walk-forward burn-in 30d / retrain 30d). **3 data-driven regimes** that emerged on ~9100 hours: R0 Quiet (~42%, low vol, drift 0), R1 Trending (~18%, mid vol, +drift, P(stay) 92%), R2 Stress (~40%, high vol, downside bias, P(stay) 79%). Switches 3–8 times/day, matching the h=30 horizon. Default since 2026-06-03 — supersedes the prior US-macro Markov-Switching (daily timescale, single-cluster collapse) and the transitional `RegimeSession` baseline (Asia/EU/US, informationally empty). Used for stratified val + `val_nll per regime` diagnostic; raw US macros (DXY, VIX, rates, gold) still feed `MacroEncoder` (16-dim), decoupled from the regime detector. `RegimeMarkovSwitching` and `RegimeSession` remain in the codebase as fallbacks.

🇮🇹
- **Live paper trading** — feed Binance WebSocket con persistenza stato, reconnect automatico (exponential backoff), aggiornamento Volume Profile incrementale, refresh funding thread-safe.

**EN**
- **Live paper trading** — Binance WebSocket feed with state persistence, automatic reconnection (exponential backoff), incremental Volume Profile updates, thread-safe funding refresh.

🇮🇹
- **Walk-forward validation** — purged k-fold con embargo, no look-ahead bias.

**EN**
- **Walk-forward validation** — purged k-fold with embargo, no look-ahead bias.

🇮🇹
- **Backtest engine** con stress test (fee×2 slip×3 pessimistic; fee×1.5 slip×5 flash crash), bootstrap CI 5000 iter, analisi per regime, statistiche di recovery MDD.

**EN**
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

# 2. Verifica ambiente
python scripts/00_check_setup.py

# 3. Pipeline completa (menu interattivo)
python run_all.py

# 4. Oppure specifica architettura
python run_all.py --arch itransformer
python run_all.py --arch nhits
python run_all.py --arch tcnmamba
python run_all.py --arch lstm           # legacy backward compat

# 5. Knowledge Distillation (train tutte + multi-teacher weighted + distill student)
python run_all.py --distill

# 6. Diagnostica
python scripts/07_verify_teacher.py             # confronto architetture
python scripts/99_replay_live_vs_training.py    # diagnostica BLOCKER #1
```

🇮🇹 La pipeline scarica lo storico BTC/USDT da Binance (default: candele **1h** multi-anno dal 2019-01-01, ~65k barre — pivot 2026-06-09), costruisce 104 feature, addestra il modello selezionato, esegue il backtest con stress test, avvia il feed live WebSocket, apre la dashboard su `http://localhost:8050`.

**EN** The pipeline downloads BTC/USDT history from Binance (default: multi-year **1h** candles from 2019-01-01, ~65k bars — 2026-06-09 pivot), engineers 104 features, trains the selected model, runs backtest with stress tests, starts the live WebSocket feed, and opens the dashboard at `http://localhost:8050`.

---

## Architettura · Architecture

```
Binance REST/WS
      │
      ▼
Candele OHLCV 1h (default: 2019-01-01 → oggi, ~65k barre — pivot 2026-06-09)
      │
      ▼
Feature Engineering: 104 feature (VWAP, VP short/mid, CVD, microstructure,
                                   funding, tempo, lag, interactions)
      │
      ├─── Macro data (FRED + yFinance) → MacroEncoder 16-dim
      ├─── BTC → realized vol oraria → RegimeMarkovBTC (Markov-Switching,
      │                                                 3 regimi: Quiet / Trending / Stress)
      │
      ▼
Sliding windows 120×104 (contesto 120 barre = 5 giorni a 1h) → dataset normalizzato (RobustScaler)
      │
      ▼
Architettura (selezionabile):
      │
      ├─ itransformer → attention sulle feature, multi-scala (×1/×5/×15 barre)
      ├─ nhits        → pure-MLP gerarchico (stack 8/4/1)
      ├─ tcnmamba     → TCN dilatate (RF=127) + Mamba SSM, gated fusion
      ├─ lstm         → LSTM+GRU dual-stream + attention temporale (legacy)
      │
      ├─ [--distill]  Multi-teacher Knowledge Distillation: scoring → soft
      │                labels pesate (shuffle-safe) → student al 60% epoche
      │
      ▼
Output: μ (direzione) + σ (incertezza) + ν (code pesanti)   in z-score
      │
      ▼
PipelineState.denormalize_predictions(μ, σ)   →   spazio raw
      │
      ▼
Monte Carlo: 2000 scenari GJR-GARCH(1,1) × 30 barre (30h a 1h)
      │
      ▼
Conviction score (direzione × ampiezza × calibrazione × regime)
      │
      ▼
Risk Manager (Kelly sizing, ATR stop, trailing, circuit breaker 15% MtM)
      │
      ▼
BUY / SELL / HOLD  +  size  +  stop loss  +  take profit
```

---

## Struttura del progetto · Project Structure

```
quantsys_project/
├── config/
│   ├── default.yaml              parametri condivisi (data, features, model, training, risk, distillation)
│   ├── secrets.yaml.example      template per API keys (copia in secrets.yaml)
│   ├── interval/                 override risoluzione candela (QUANTSYS_INTERVAL / --interval)
│   │   ├── 1m.yaml               chiavi interval-dipendenti era 1m (stride 5, embargo 1500, ...)
│   │   └── 1h.yaml               chiavi interval-dipendenti pivot 1h (stride 1, embargo 168, ...)
│   └── arch/
│       ├── lstm.yaml             override LSTM
│       ├── itransformer.yaml     override iTransformer
│       ├── nhits.yaml            override N-HiTS
│       └── tcnmamba.yaml         override TCN+Mamba
├── quantsys/                     package Python installabile
│   ├── data/                     Binance REST + WebSocket + funding rate
│   ├── features/                 FeatureBuilder (104 feature post C-funding, split dual-stream)
│   ├── macro/                    FRED + yFinance + RegimeMarkovBTC (Markov-Switching su realized vol BTC)
│   ├── model/
│   │   ├── __init__.py           QuantLSTM, QuantTFT, QuantiTransformer
│   │   ├── nhits.py              QuantNHiTS (pure-MLP gerarchico)
│   │   ├── tcn_mamba.py          QuantTCNMamba (TCN + Mamba SSM + gated fusion)
│   │   ├── ensemble.py           EnsembleModel (omogeneo / eterogeneo, AMP off in inferenza)
│   │   ├── distillation.py       Knowledge Distillation multi-teacher
│   │   ├── forecast.py           Monte Carlo GJR-GARCH(1,1) + neural-guided
│   │   └── revin.py              Reversible Instance Normalization (opzionale)
│   ├── trading/                  Kelly sizing, SL dinamico, trailing, circuit breaker
│   └── utils/                    config loader, device setup, logging, PipelineState
├── scripts/
│   ├── 00_check_setup.py         verifica CUDA, dipendenze, connessione Binance
│   ├── 00_test_binance_testnet.py  check connettività API testnet
│   ├── 01_download_data.py       Binance → 104 feature (C-funding) → lstm_dataset.npz
│   ├── 01_update_data.py         aggiornamento incrementale (solo delta candele)
│   ├── 01b_download_macro.py     FRED + yFinance → RegimeMarkovBTC su candele BTC → update dataset
│   ├── 01c_iv_poller.py          poller IV Deribit (chain opzioni BTC + DVOL) → data/iv/
│   ├── 02_train.py               training con --arch, --distill, ensemble
│   ├── 02b_walkforward_validate.py   walk-forward purged k-fold con embargo
│   ├── 02c_optuna_search.py      Bayesian hyperparameter search (solo LSTM)
│   ├── 03_backtest.py            backtest + stress test + bootstrap CI
│   ├── 04_live_signals.py        feed live WebSocket + paper trading (direzionale)
│   ├── 04b_vol_paper.py          forward test vol: NN-RV vs IV → straddle testnet Deribit
│   ├── 04c_vol_paper_baselines.py  baseline always-long/short-vol (gate pre-reg) dai chain snapshot
│   ├── 05_analyze_signals.py     analisi sessione live
│   ├── 06_dashboard.py           server dashboard HTTP (Dash)
│   ├── 07_verify_teacher.py      confronto architetture per selezione teacher
│   ├── 99_replay_live_vs_training.py   diagnostica BLOCKER #1 (parity live vs training)
│   ├── dev_vols_*.py             famiglia vol: macro_append + giudici QLIKE/RS (linea attiva)
│   ├── paper_01_dir_baselines.py baseline econometriche direzionali (paper, vedi docs/paper/)
│   └── archive/                  probe chiusi (xs cross-sectional KILL, step0 σ-recal)
├── tests/                        suite pytest (features, NLL, PipelineState, regression sui fix recenti)
├── dashboard/                    React dashboard (artifact per claude.ai)
├── data/                         generato (gitignored)
├── models/                       checkpoint per architettura (gitignored)
├── results/                      backtest + segnali live per architettura (gitignored)
└── logs/                         log rotanti (gitignored)
```

🇮🇹 Vedi [AVVIO.md](AVVIO.md) per la guida operativa completa e [TEORIA.md](TEORIA.md) per i fondamenti teorici.

**EN** See [AVVIO.md](AVVIO.md) for the full operational guide and [TEORIA.md](TEORIA.md) for theoretical foundations.

---

## Pipeline Knowledge Distillation · Knowledge Distillation Pipeline

```bash
python run_all.py --distill
```

🇮🇹
1. **Fase 2a** — Addestra tutte le architetture in `config/default.yaml → distillation.archs` (default: iTransformer + N-HiTS + TCN+Mamba) indipendentemente con `n_ensemble=1`. Skippa un'arch se `models/{arch}/best_model.pt` esiste già; forza retrain con `--force-download`.
2. **Fase 2b** — Multi-Teacher Scoring: ogni modello valutato alla best epoch con scoring normalizzato (40% val_loss + 35% Spearman ρ + 25% directional accuracy). Pesi via softmax(temperature=2). Lo score massimo diventa primary teacher; gli altri restano nel pool come teacher pesati.
3. **Fase 2c** — Riadestra ogni modello come student con:
   - Transfer dei pesi delle output heads (μ, σ, ν) dal best teacher
   - Loss mista scala-normalizzata `(1−α)·NLL_reale + α·distill_loss` con α=0.3, normalizzata per varianza per-componente del teacher (μ~1e-5, ν~5 contribuiscono equamente)
   - Soft labels pesate da tutti i teacher integrate nel `TensorDataset` (shuffle-safe)
   - 60% di riduzione epoche (convergenza accelerata)
   - Auto-skip se già distillato (`config.json` contiene `distilled: true`)

**EN**
1. **Phase 2a** — Trains all architectures listed in `config/default.yaml → distillation.archs` (default: iTransformer + N-HiTS + TCN+Mamba) independently with `n_ensemble=1`. Skips an arch if `models/{arch}/best_model.pt` already exists; force retrain with `--force-download`.
2. **Phase 2b** — Multi-Teacher Scoring: every model scored at its best validation epoch with normalized scoring (40% val_loss + 35% Spearman ρ + 25% directional accuracy). Weights via softmax(temperature=2). Top score becomes the primary teacher; the others remain in the pool as weighted teachers.
3. **Phase 2c** — Retrains each model as student with:
   - Output head weight transfer (μ, σ, ν heads) from best teacher
   - Scale-normalized mixed loss `(1−α)·NLL_real + α·distill_loss` with α=0.3, normalized by per-component teacher variance (μ~1e-5, ν~5 contribute equally)
   - Weighted soft labels from all teachers integrated in `TensorDataset` (shuffle-safe)
   - 60% epoch reduction (accelerated convergence)
   - Auto-skip if student already distilled (`config.json` contains `distilled: true`)

🇮🇹 Cambia la composizione dell'ensemble modificando `distillation.archs` in `config/default.yaml`. Vedi [AVVIO.md](AVVIO.md) per esempi e dettagli operativi.

**EN** Change the ensemble composition by editing `distillation.archs` in `config/default.yaml`. See [AVVIO.md](AVVIO.md) for examples and operational details.

---

## Hardware

🇮🇹 Setup di riferimento: **RTX 2070 Super (8 GB VRAM)**.

**EN** Reference setup: **RTX 2070 Super (8 GB VRAM)**.

🇮🇹
| Parametro | Valore | Sorgente |
|---|---|---|
| Training batch size | 64 (default) | `config/default.yaml` |
| Inference batch (backtest) | 256 | `scripts/03_backtest.py` |
| AMP fp16 training | sì | via `setup_device` |
| AMP inference | **off** (hardcoded) | `quantsys/model/ensemble.py:170` (evita NaN da spectral_norm + Mamba scan) |
| `hardware.cudnn_benchmark` | true | kernel ottimizzati per shape fisse |
| `hardware.pin_memory` | true | trasferimento RAM → VRAM zero-copy |
| TCN+Mamba VRAM | ~2.5 GB | `d_model=128`, 4 blocchi TCN + 3 layer Mamba |

**EN**
| Parameter | Value | Source |
|---|---|---|
| Training batch size | 64 (default) | `config/default.yaml` |
| Inference batch (backtest) | 256 | `scripts/03_backtest.py` |
| AMP fp16 training | yes | via `setup_device` |
| AMP inference | **off** (hardcoded) | `quantsys/model/ensemble.py:170` (avoids NaN from spectral_norm + Mamba scan) |
| `hardware.cudnn_benchmark` | true | optimized kernels for fixed shapes |
| `hardware.pin_memory` | true | zero-copy RAM → VRAM transfer |
| TCN+Mamba VRAM | ~2.5 GB | `d_model=128`, 4 TCN blocks + 3 Mamba layers |

🇮🇹 Il fallback CPU-only funziona (con rallentamento 20-50× su training; piena velocità su backtest/live). Apple Silicon (MPS) non testato. Vedi [AVVIO.md](AVVIO.md) per tuning con poca VRAM (4GB) e molta VRAM (≥16GB).

**EN** CPU-only fallback works (with 20-50× slowdown on training; full-speed on backtest/live). Apple Silicon (MPS) untested. See [AVVIO.md](AVVIO.md) for low-VRAM (4GB) and high-VRAM (≥16GB) tuning.

---

## FRED API key (opzionale) · FRED API key (optional)

🇮🇹
1. Registrazione gratuita: https://fred.stlouisfed.org/docs/api/api_key.html
2. Copia `config/secrets.yaml.example` in `config/secrets.yaml` e inserisci la chiave.
3. Senza chiave: funziona con rate limit più stretti (~120 req/min).

**EN**
1. Free registration: https://fred.stlouisfed.org/docs/api/api_key.html
2. Copy `config/secrets.yaml.example` to `config/secrets.yaml` and add your key.
3. Without a key: works under stricter rate limits (~120 req/min).

🇮🇹 `config/secrets.yaml` è gitignored — non viene mai committato.

**EN** `config/secrets.yaml` is gitignored — it never gets committed.

---

## Sviluppo & Test · Development & Tests

```bash
pytest tests/                          # suite completa
pytest tests/test_recent_fixes.py -v   # regression test sui fix critici (z-score, RevIN, BLOCKER #1)
```

🇮🇹 La suite copre FeatureBuilder no-leakage, invarianti RobustScaler, gradient check su t-Student NLL, (de)serializzazione PipelineState, edge case Kelly sizing, trigger circuit-breaker e golden snapshot del live feature buffer.

**EN** The test suite covers FeatureBuilder no-leakage, RobustScaler invariants, t-Student NLL gradient checks, PipelineState (de)serialization, Kelly sizing edge cases, circuit-breaker triggers, and golden snapshots of the live feature buffer.

---

## Licenza · License

🇮🇹 [MIT License](LICENSE) — codice di ricerca, non consulenza finanziaria.

**EN** [MIT License](LICENSE) — research code, not financial advice.
