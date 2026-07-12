# QUANTSYS — Motore neurale di forecasting per BTC/USDT · QUANTSYS — Neural Forecasting Engine for BTC/USDT

🇮🇹 Motore neurale di forecasting per BTC/USDT + analytics opzioni crypto. **Linea di produzione: volatilità @ 1 ora** (`config/default.yaml → features.target_type: log_rv`, `data.interval: 1h`; design interval-agnostic, 1m = identità, perimetro 1m in backup). La famiglia **vol** ha prodotto l'unico segnale validato del progetto — `log_rv` batte HAR-RV del 30% in QLIKE su test a 1h (confermato due volte OOS: single-split 5-seed 0.257/ratio 0.70 **e** purged k-fold sui fold data-rich; FAIL a 1m: risoluzione-specifico). Il modello di produzione è l'**iTransformer 5-seed**. Il filone **direzionale** è negativo OOS su 1m E 1h (pivot 1h killed) — resta legacy/in backup. L'asimmetria firmata `log_rs_ratio` è impredicibile (FAIL 2026-06-11) → **i momenti pari generalizzano OOS, i dispari no**. Vedi `docs/MODEL_IMPROVEMENTS.md` e `STATUS.md`.

**EN** Neural forecasting engine for BTC/USDT + crypto-options analytics. **Production line: volatility @ 1 hour** (`config/default.yaml → features.target_type: log_rv`, `data.interval: 1h`; interval-agnostic design, 1m = identity, the 1m perimeter is backed up). The **vol** family produced the project's only validated signal — `log_rv` beats HAR-RV by 30% in test QLIKE at 1h (confirmed OOS twice: single-split 5-seed 0.257/ratio 0.70 **and** purged k-fold on the data-rich folds; FAIL at 1m: resolution-specific). The production model is the **iTransformer 5-seed**. The **directional** axis is OOS-negative at both 1m AND 1h (1h pivot killed) — kept as legacy/backup. The signed asymmetry `log_rs_ratio` is unpredictable (FAIL 2026-06-11) → **even moments generalize OOS, odd ones don't**. See `docs/MODEL_IMPROVEMENTS.md` and `STATUS.md`.

🇮🇹 **Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API · Deribit public REST (dashboard/IV).

**EN** **Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API · Deribit public REST (dashboard/IV).

🇮🇹 > Documentazione **bilingue in un unico file** (IT + EN per paragrafo, marker 🇮🇹/**EN**): [README.md](README.md) · [AVVIO.md](AVVIO.md) · [TEORIA.md](TEORIA.md) · [docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md).

**EN** > **Single-file bilingual** documentation (IT + EN per paragraph, markers 🇮🇹/**EN**): [README.md](README.md) · [AVVIO.md](AVVIO.md) · [TEORIA.md](TEORIA.md) · [docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md).

---

## 1. Panoramica e obiettivi · Overview & Goals

🇮🇹 QUANTSYS è un motore di forecasting probabilistico su BTC/USDT a **intervallo candela parametrico** (`data.interval`; default corrente `1h`, perimetro `1m` legacy in backup — tutte le conversioni temporali derivano da `interval_minutes`, identità a 1m). Predice una **distribuzione completa** (μ, σ, ν di una t-Student) anziché una stima puntuale, via un **ensemble eterogeneo** di 3 architetture + **multi-teacher distillation**. Due linee di ricerca convivono sullo stesso spine:

- **Linea VOL (produzione, validata):** target `log_rv` (log realized variance su h barre). Unico segnale che generalizza OOS — batte HAR-RV del 30% in QLIKE.
- **Linea direzionale (legacy, negative-control):** target `ret`. Nessun alpha OOS a 1m né a 1h; conservata come negative-control documentato.

**EN** QUANTSYS is a probabilistic forecasting engine on BTC/USDT at a **parametric candle interval** (`data.interval`; current default `1h`, legacy `1m` perimeter backed up — all temporal conversions derive from `interval_minutes`, identity at 1m). It predicts a **full distribution** (μ, σ, ν of a Student-t) rather than a point estimate, via a **heterogeneous ensemble** of 3 architectures + **multi-teacher distillation**. Two research lines coexist on the same spine:

- **VOL line (production, validated):** target `log_rv` (log realized variance over h bars). The only signal that generalizes OOS — beats HAR-RV by 30% in QLIKE.
- **Directional line (legacy, negative-control):** target `ret`. No OOS alpha at 1m or 1h; kept as a documented negative-control.

🇮🇹 > ✅ **Stato live engine:** paper-only (nessun ordine reale), ma **BLOCKER #1 RISOLTO (2026-06-05)**. Il path live costruisce ora le **104 feature canoniche** via `FeatureBuilder` (single source of truth) con lo scaler del training, più un catch-up REST contiguo al boot — con **parity feature *e* segnale bit-perfect** vs backtest (`tests/test_live_training_parity.py`, replay Δ=0). I segnali live ora riflettono il backtest; lo smoke test live passa. Vedi `TEORIA.md` §11. ⚠ Nota: il backtest direzionale è negativo out-of-sample — il paper-trading serve ad accumulare trade forward reali, senza aspettativa di Sharpe>0 a priori.

**EN** > ✅ **Live engine status:** paper-only (no real orders), but **BLOCKER #1 RESOLVED (2026-06-05)**. The live path now builds the **104 canonical features** via `FeatureBuilder` (single source of truth) with the training scaler, plus a contiguous REST catch-up at boot — achieving **bit-perfect feature *and* signal parity** vs the backtest (`tests/test_live_training_parity.py`, replay Δ=0). Live signals now reflect the backtest; the live smoke test passes. See `TEORIA.md` §11. ⚠ Note: the directional backtest is negative out-of-sample — paper-trading is for accumulating real forward trades, with no a-priori expectation of Sharpe>0.

### 1.1 Caratteristiche principali · Key Features

🇮🇹
- **Ensemble eterogeneo a 3 architetture** + LSTM legacy (vedi §4).
- **Knowledge Distillation multi-teacher** target-aware (`--distill`, vedi §4.2).
- **104 feature ingegnerizzate** (86 dinamiche + 18 strutturali, post-filtro C-funding, vedi §3).
- **Output probabilistico** t-Student (μ, σ, ν) con loss NLL + penalità asimmetrica + CRPS + Direction-Value (vedi §4.3).
- **Rilevamento regimi BTC** `RegimeMarkovBTC` (Markov-Switching causale, vedi §3.3).
- **Simulazione Monte Carlo** 2000 scenari GJR-GARCH(1,1) (vedi §4.4).
- **Gestione del rischio** Kelly frazionario + SL ATR + trailing + circuit breaker DD 15% MtM (vedi §6).
- **Walk-forward validation** purged k-fold con embargo, no look-ahead (vedi §5).
- **Backtest engine** con stress test, bootstrap CI 5000 iter, analisi per regime, recovery MDD (vedi §5).
- **Live paper trading** feed Binance WebSocket con persistenza stato, reconnect exponential-backoff, Volume Profile incrementale, funding refresh thread-safe (vedi §6).
- **Dashboard** Deribit Options Risk Terminal, GPU-free, indipendente dalla pipeline ML (vedi §6.2).

**EN**
- **3-architecture heterogeneous ensemble** + legacy LSTM (see §4).
- **Multi-teacher target-aware Knowledge Distillation** (`--distill`, see §4.2).
- **104 engineered features** (86 dynamic + 18 structural, post C-funding filter, see §3).
- **Probabilistic Student-t output** (μ, σ, ν) with NLL loss + asymmetric penalty + CRPS + Direction-Value (see §4.3).
- **BTC regime detection** `RegimeMarkovBTC` (causal Markov-Switching, see §3.3).
- **Monte Carlo simulation** 2000 GJR-GARCH(1,1) scenarios (see §4.4).
- **Risk management** fractional Kelly + ATR SL + trailing + 15% MtM drawdown circuit breaker (see §6).
- **Walk-forward validation** purged k-fold with embargo, no look-ahead (see §5).
- **Backtest engine** with stress tests, 5000-iter bootstrap CI, regime-conditioned analysis, MDD recovery (see §5).
- **Live paper trading** Binance WebSocket feed with state persistence, exponential-backoff reconnect, incremental Volume Profile, thread-safe funding refresh (see §6).
- **Dashboard** Deribit Options Risk Terminal, GPU-free, decoupled from the ML pipeline (see §6.2).

---

## 2. Setup e dipendenze · Setup & Dependencies

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

# 2. Verifica ambiente / environment check
python scripts/00_check_setup.py
```

🇮🇹 `scripts/00_check_setup.py` verifica CUDA, dipendenze e connessione Binance. Il fallback **CPU-only** funziona (rallentamento 20-50× su training; piena velocità su backtest/live). Apple Silicon (MPS) non testato.

**EN** `scripts/00_check_setup.py` verifies CUDA, dependencies and the Binance connection. The **CPU-only** fallback works (20-50× slowdown on training; full speed on backtest/live). Apple Silicon (MPS) untested.

### 2.1 FRED API key (opzionale) · FRED API key (optional)

🇮🇹
1. Registrazione gratuita: https://fred.stlouisfed.org/docs/api/api_key.html
2. Copia `config/secrets.yaml.example` in `config/secrets.yaml` e inserisci la chiave.
3. Senza chiave: funziona con rate limit più stretti (~120 req/min).
4. `config/secrets.yaml` è gitignored — non viene mai committato.

**EN**
1. Free registration: https://fred.stlouisfed.org/docs/api/api_key.html
2. Copy `config/secrets.yaml.example` to `config/secrets.yaml` and add your key.
3. Without a key: works under stricter rate limits (~120 req/min).
4. `config/secrets.yaml` is gitignored — it never gets committed.

### 2.2 Hardware

🇮🇹 Setup di riferimento: **RTX 2070 Super (8 GB VRAM)**. Sequenzialità GPU: backtest/walkforward in parallelo OK; il training 5-seed × 3 arch va **sequenziale** (OOM). Non girare live/paper + training/inferenza in parallelo (contesa CUDA). TCN+Mamba è il collo di bottiglia (~80 min/seed sul dataset 1m-525k).

**EN** Reference setup: **RTX 2070 Super (8 GB VRAM)**. GPU sequencing: parallel backtest/walkforward OK; 5-seed × 3-arch training must run **sequentially** (OOM). Do not run live/paper + training/inference in parallel (CUDA contention). TCN+Mamba is the bottleneck (~80 min/seed on the 1m-525k dataset).

🇮🇹
| Parametro | Valore | Sorgente |
|---|---|---|
| Training batch size | 64 (default) | `config/default.yaml → training.batch_size` |
| Inference batch (backtest) | 256 | `scripts/03_backtest.py` |
| AMP fp16 training | sì | `training.use_amp: true` via `setup_device` |
| AMP inference | **off** (hardcoded) | `quantsys/model/ensemble.py` (evita NaN da spectral_norm + Mamba scan) |
| `hardware.cudnn_benchmark` | true | kernel ottimizzati per shape fisse |
| `hardware.pin_memory` | true | trasferimento RAM → VRAM zero-copy |
| `hardware.cpu_fraction` | 0.5 | limite worker/thread CPU |
| TCN+Mamba VRAM | ~2.5 GB | `d_model=128`, 4 blocchi TCN + 3 layer Mamba |

**EN**
| Parameter | Value | Source |
|---|---|---|
| Training batch size | 64 (default) | `config/default.yaml → training.batch_size` |
| Inference batch (backtest) | 256 | `scripts/03_backtest.py` |
| AMP fp16 training | yes | `training.use_amp: true` via `setup_device` |
| AMP inference | **off** (hardcoded) | `quantsys/model/ensemble.py` (avoids NaN from spectral_norm + Mamba scan) |
| `hardware.cudnn_benchmark` | true | optimized kernels for fixed shapes |
| `hardware.pin_memory` | true | zero-copy RAM → VRAM transfer |
| `hardware.cpu_fraction` | 0.5 | CPU worker/thread cap |
| TCN+Mamba VRAM | ~2.5 GB | `d_model=128`, 4 TCN blocks + 3 Mamba layers |

🇮🇹 Vedi [AVVIO.md](AVVIO.md) per tuning con poca VRAM (4GB) e molta VRAM (≥16GB).

**EN** See [AVVIO.md](AVVIO.md) for low-VRAM (4GB) and high-VRAM (≥16GB) tuning.

---

## 3. Dati · Data

🇮🇹 La pipeline scarica lo storico BTC/USDT da Binance (default: candele **1h** multi-anno dal 2019-01-01, ~65k barre — pivot 2026-06-09; split 51.130/6.391/6.392), costruisce **104 feature**, le normalizza con un **RobustScaler globale** (mediana/IQR, fittato SOLO sul training — no leakage val/test), e finestra in sequenze **120×104** (T=120 barre = 5 giorni a 1h; `window_stride: 1`). I parametri scaler + config feature + `target_scale` + `forecast_horizon` + `interval` sono persistiti in `PipelineState` (unico contratto train↔inference).

**EN** The pipeline downloads BTC/USDT history from Binance (default: multi-year **1h** candles from 2019-01-01, ~65k bars — 2026-06-09 pivot; split 51,130/6,391/6,392), engineers **104 features**, normalizes them with a **global RobustScaler** (median/IQR, fit on training ONLY — no val/test leakage), and windows into **120×104** sequences (T=120 bars = 5 days at 1h; `window_stride: 1`). The scaler params + feature config + `target_scale` + `forecast_horizon` + `interval` are persisted in `PipelineState` (the single train↔inference contract).

### 3.1 Target & log-return · Target & log-return

🇮🇹 Tutto lavora su **log-return** (stazionari, simmetrici), mai prezzi assoluti. Il target è la **somma dei log-return delle prossime 30 barre** (`features.forecast_horizon: 30` — 30h a 1h, era 30 min a 1m). Cambiarlo richiede ri-generare il dataset e ri-allineare `PipelineState.forecast_horizon` (validato a runtime con `RuntimeError` in backtest e live). Famiglie di target via `features.target_type`: `ret` (direzionale legacy), `log_rv` (produzione vol), `log_rs_ratio` (probe asimmetria, FAIL — vedi §7).

**EN** Everything works on **log-returns** (stationary, symmetric), never absolute prices. The target is the **sum of the next 30 bars' log-returns** (`features.forecast_horizon: 30` — 30h at 1h, was 30 min at 1m). Changing it requires regenerating the dataset and re-aligning `PipelineState.forecast_horizon` (validated at runtime with `RuntimeError` in backtest and live). Target families via `features.target_type`: `ret` (legacy directional), `log_rv` (vol production), `log_rs_ratio` (asymmetry probe, FAIL — see §7).

### 3.2 Le 104 feature · The 104 features

🇮🇹 **104 feature = 86 dinamiche + 18 strutturali**, post-filtro C-funding (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`): VWAP + Volume Profile (short + mid), CVD, microstructure delle candele (body ratio, shadow, price velocity/acceleration), funding rate, volatilità multi-finestra (5/10/20/60), lag returns, encoding temporale (sin/cos hour-of-day, day-of-week, sessioni), feature interactions, livelli strutturali (ATH/ATL 30d, momentum_30d, distanza da round level, MA200m). Le **15 feature live-incompatibili** (90d/365d/long-lookback, frac-diff, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`) sono filtrate — il conteggio è verificato sul dataset, non assunto. Razionale in `docs/MODEL_IMPROVEMENTS.md`.

**EN** **104 features = 86 dynamic + 18 structural**, post C-funding filter (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`): VWAP + Volume Profile (short + mid scales), CVD, candle microstructure (body ratio, shadows, price velocity/acceleration), funding rate, multi-window volatility (5/10/20/60), lag returns, time encoding (sin/cos hour-of-day, day-of-week, sessions), feature interactions, structural levels (ATH/ATL 30d, momentum_30d, round-level distance, MA200m). The **15 live-incompatible features** (90d/365d/long-lookback, frac-diff, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`) are filtered out — the count is verified on the dataset, not assumed. Rationale in `docs/MODEL_IMPROVEMENTS.md`.

### 3.3 Macro & rilevamento regimi · Macro & regime detection

🇮🇹 Le macro USA (FRED + yFinance: DXY, VIX, tassi, oro) alimentano un `MacroEncoder` (16-dim), scollegato dal regime detector. **`RegimeMarkovBTC`** (Markov-Switching, Hamilton 1989) è fittato sulla realized volatility oraria di BTC (`log_ret_h` + `log_rv` da `raw_candles.parquet`, PCA `n_pca=1`, switching mean+variance, walk-forward **burn-in 30gg / retrain 90gg**). **CAUSALE**: usa `filtered_marginal_probabilities` (NON smoothed) + Hamilton filter forward-only; il refit expanding è O(t) (su 7 anni il mensile costava ~9h → cadenza 90gg). **3 regimi data-driven**: R0 Quiet (~42%, bassa vol, drift 0), R1 Trending (~18%, mid vol, +drift, P(stay) 92%), R2 Stress (~40%, alta vol, bias ribasso, P(stay) 79%) — misurati sullo span 1m 2025-26, da ri-misurare sui 7 anni 1h. Persistiti in `data/regime_probs.parquet` (index orario UTC). Uso: **stratificazione val + diagnostica `val_nll per regime`**, NON è feature di input. `RegimeMarkovSwitching` (macro USA daily) e `RegimeSession` (Asia/EU/US) restano come fallback opzionali.

**EN** US macros (FRED + yFinance: DXY, VIX, rates, gold) feed a `MacroEncoder` (16-dim), decoupled from the regime detector. **`RegimeMarkovBTC`** (Markov-Switching, Hamilton 1989) is fit on hourly BTC realized volatility (`log_ret_h` + `log_rv` from `raw_candles.parquet`, PCA `n_pca=1`, switching mean+variance, walk-forward **30d burn-in / 90d retrain**). **CAUSAL**: uses `filtered_marginal_probabilities` (NOT smoothed) + forward-only Hamilton filter; the expanding refit is O(t) (over 7 years monthly cost ~9h → 90d cadence). **3 data-driven regimes**: R0 Quiet (~42%, low vol, drift 0), R1 Trending (~18%, mid vol, +drift, P(stay) 92%), R2 Stress (~40%, high vol, downside bias, P(stay) 79%) — measured on the 1m 2025-26 span, to be re-measured on the 7-year 1h span. Persisted in `data/regime_probs.parquet` (hourly UTC index). Use: **stratified val + `val_nll per regime` diagnostic**, NOT an input feature. `RegimeMarkovSwitching` (US-macro daily) and `RegimeSession` (Asia/EU/US) remain as optional fallbacks.

### 3.4 Invariante z-score vs raw · z-score vs raw invariant

🇮🇹 ⚠ **Il bug più costoso del progetto.** Il modello predice μ/σ/ν in **spazio z-score** (`target_ret` scalato dal RobustScaler; `target_scale` = IQR del target raw, persistito in `PipelineState`); il trading layer (`SignalGenerator`, `RiskManager`) opera in **spazio raw**. **Ogni entry-point DEVE chiamare `PipelineState.denormalize_predictions(mu, sigma)` subito dopo il forward**, prima del trading layer (bug 2026-05-23: saltarla → SL/TP macroscopici, Sharpe −256 → +18.7). Con target `log_rv` la sola `denormalize_predictions` è insufficiente (mediana log-RV ≈ −7.2): l'inversione completa è `μ·IQR + centro` dal RobustScaler. Vedi `TEORIA.md` §5.

**EN** ⚠ **The project's costliest bug.** The model predicts μ/σ/ν in **z-score space** (`target_ret` scaled by the RobustScaler; `target_scale` = raw-target IQR, persisted in `PipelineState`); the trading layer (`SignalGenerator`, `RiskManager`) operates in **raw space**. **Every entry-point MUST call `PipelineState.denormalize_predictions(mu, sigma)` right after the forward**, before the trading layer (bug 2026-05-23: skipping it → macroscopic SL/TP, Sharpe −256 → +18.7). With target `log_rv`, `denormalize_predictions` alone is insufficient (log-RV median ≈ −7.2): the full inversion is `μ·IQR + center` from the RobustScaler. See `TEORIA.md` §5.

---

## 4. Modellazione · Modeling

### 4.1 Architetture · Architectures

🇮🇹 **Ensemble eterogeneo a 3 architetture** (default post 2026-05-14), selezionabili via `--arch`:

- **iTransformer** (`QuantiTransformer`) — attention sulle feature (non sul tempo), embedding multi-scala (pooling ×1/×5/×15 barre), O(F²). **Modello di produzione (5-seed)**.
- **N-HiTS** (`QuantNHiTS`) — interpolazione gerarchica pure-MLP, stack pooling multi-scala (8/4/1) — sostituisce LSTM.
- **TCN+Mamba** (`QuantTCNMamba`) — convoluzioni causali dilatate (campo recettivo 127) + State Space Model con parametri input-dipendenti e fusion gated.
- **LSTM+GRU** (`QuantLSTM`) — dual-stream con attention temporale (legacy, backward compat; sotto-performante, vedi `CHANGELOG.md`).

**Forward contract** di ogni arch: `forward(x, x_macro=None) -> (mu, ls2, lnu)` (`+dir_logits` se multitask; `(quantile_preds, ...)` se `loss_type=quantile`, default). Output path arch-isolati: `models/{arch}/` e `results/{arch}/`.

**EN** **3-architecture heterogeneous ensemble** (default post 2026-05-14), selectable via `--arch`:

- **iTransformer** (`QuantiTransformer`) — attention over features (not time), multi-scale embedding (×1/×5/×15-bar pooling), O(F²). **Production model (5-seed)**.
- **N-HiTS** (`QuantNHiTS`) — pure-MLP hierarchical interpolation, multi-scale pooling stacks (8/4/1) — replaces LSTM.
- **TCN+Mamba** (`QuantTCNMamba`) — dilated causal convolutions (receptive field 127) + State Space Model with input-dependent parameters and gated fusion.
- **LSTM+GRU** (`QuantLSTM`) — dual-stream with temporal attention (legacy, kept for backward compat; under-performing, see `CHANGELOG.md`).

Each arch's **forward contract**: `forward(x, x_macro=None) -> (mu, ls2, lnu)` (`+dir_logits` if multitask; `(quantile_preds, ...)` if `loss_type=quantile`, the default). Arch-isolated output paths: `models/{arch}/` and `results/{arch}/`.

🇮🇹 **Ensemble eterogeneo** — combina i modelli in inferenza via Legge della Varianza Totale (pesata):
- `mu_ens = Σ wᵢ · muᵢ`
- `sigma_ens = sqrt(Σ wᵢ · sigmaᵢ² + Σ wᵢ · (muᵢ − mu_ens)²)`

Pesi default in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`), oppure dinamici inverse-NLL (`ensemble_nll_temperature`). **AMP off in inferenza** (evita NaN spectral_norm + Mamba scan). ⚠ Single source of truth della composizione: `config/default.yaml → distillation.archs`.

**EN** **Heterogeneous ensemble** — combines models at inference via the Law of Total Variance (weighted):
- `mu_ens = Σ wᵢ · muᵢ`
- `sigma_ens = sqrt(Σ wᵢ · sigmaᵢ² + Σ wᵢ · (muᵢ − mu_ens)²)`

Default weights in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`), or dynamic inverse-NLL (`ensemble_nll_temperature`). **AMP off at inference** (avoids spectral_norm + Mamba-scan NaN). ⚠ Composition single source of truth: `config/default.yaml → distillation.archs`.

### 4.2 Knowledge Distillation multi-teacher · Multi-teacher Knowledge Distillation

```bash
python run_all.py --distill
```

🇮🇹
1. **Fase 2a** — Addestra tutte le architetture in `config/default.yaml → distillation.archs` (default: iTransformer + N-HiTS + TCN+Mamba) indipendentemente con `n_ensemble=1`. Skippa un'arch se `models/{arch}/best_model.pt` esiste già; forza retrain con `--force-download`.
2. **Fase 2b** — Multi-Teacher Scoring **target-aware** (`teacher_score_weights`, single source of truth in `distillation.py`): direzionale `ret` → 40% val_loss + 35% Spearman ρ + 25% directional accuracy; volatilità `log_rv` → 65% val_loss + 35% Spearman ρ + 0% directional accuracy (sulla varianza la dir_acc è il segno-vs-mediana, non tradabile — lo straddle è direction-neutral). Pesi via softmax(temperature=2). Lo score massimo diventa primary teacher; gli altri restano nel pool come teacher pesati. Le metriche di val alla best epoch (`best_val_loss`/`best_spearman`/`best_da`) sono persistite in `config.json`.
3. **Fase 2c** — Riadestra ogni modello come student con:
   - Transfer dei pesi delle output heads (μ, σ, ν) dal best teacher
   - Loss mista scala-normalizzata `(1−α)·NLL_reale + α·distill_loss` con α=0.3, normalizzata per varianza per-componente del teacher (μ~1e-5, ν~5 contribuiscono equamente)
   - Soft labels μ/ls²/lnu = media pesata di **tutti** i teacher, integrate nel `TensorDataset` (shuffle-safe)
   - 60% di riduzione epoche (convergenza accelerata)
   - Auto-skip se già distillato (`config.json` contiene `distilled: true`)

**EN**
1. **Phase 2a** — Trains all architectures listed in `config/default.yaml → distillation.archs` (default: iTransformer + N-HiTS + TCN+Mamba) independently with `n_ensemble=1`. Skips an arch if `models/{arch}/best_model.pt` already exists; force retrain with `--force-download`.
2. **Phase 2b** — **Target-aware** Multi-Teacher Scoring (`teacher_score_weights`, single source of truth in `distillation.py`): directional `ret` → 40% val_loss + 35% Spearman ρ + 25% directional accuracy; volatility `log_rv` → 65% val_loss + 35% Spearman ρ + 0% directional accuracy (on variance dir_acc is the sign-vs-median, not tradable — the straddle is direction-neutral). Weights via softmax(temperature=2). Top score becomes the primary teacher; the others remain in the pool as weighted teachers. Best-epoch val metrics (`best_val_loss`/`best_spearman`/`best_da`) are persisted in `config.json`.
3. **Phase 2c** — Retrains each model as student with:
   - Output head weight transfer (μ, σ, ν heads) from best teacher
   - Scale-normalized mixed loss `(1−α)·NLL_real + α·distill_loss` with α=0.3, normalized by per-component teacher variance (μ~1e-5, ν~5 contribute equally)
   - Soft labels μ/ls²/lnu = weighted average of **all** teachers, integrated in the `TensorDataset` (shuffle-safe)
   - 60% epoch reduction (accelerated convergence)
   - Auto-skip if student already distilled (`config.json` contains `distilled: true`)

🇮🇹 Cambia la composizione dell'ensemble modificando `distillation.archs` in `config/default.yaml`. Vedi [AVVIO.md](AVVIO.md) per esempi.

**EN** Change the ensemble composition by editing `distillation.archs` in `config/default.yaml`. See [AVVIO.md](AVVIO.md) for examples.

### 4.3 Loss & output probabilistico · Loss & probabilistic output

🇮🇹 Ogni predizione è una **distribuzione completa** (μ, σ, ν di una t-Student), non una stima puntuale. Loss = **t-Student NLL** (code pesanti) + **penalità asimmetrica** sugli errori di segno + **CRPS** (calibrazione, `crps_weight=0.1`) + **Direction-Value joint loss** (`dv_lambda=0.3`). `loss_type=quantile` di default → `model(x)` ritorna `(quantile_preds, dir_logits)`; usa `model.predict(x)["mu"]` (mediana q2) per μ scalare. Output in **spazio z-score**, denormalizzato a monte del trading layer (vedi §3.4).

**EN** Each prediction is a **full distribution** (μ, σ, ν of a Student-t), not a point estimate. Loss = **Student-t NLL** (heavy tails) + **asymmetric penalty** on sign errors + **CRPS** (calibration, `crps_weight=0.1`) + **Direction-Value joint loss** (`dv_lambda=0.3`). `loss_type=quantile` by default → `model(x)` returns `(quantile_preds, dir_logits)`; use `model.predict(x)["mu"]` (median q2) for scalar μ. Output in **z-score space**, denormalized upstream of the trading layer (see §3.4).

🇮🇹 **Iperparametri di training salienti** (`config/default.yaml → training`): `epochs=200`, `batch_size=64`, `learning_rate=1e-4`, `weight_decay=1e-3` (anti-overfit dataset piccolo), `patience=15`, scheduler `cosine` warm-restart (T0=10, T_mult=2), `grad_clip_norm=0.5`, `n_ensemble=5`, `asymmetry_alpha=2.0`/`asymmetry_threshold=0.004`. RevIN (Kim ICLR 2022) opzionale via `model.use_revin` (default false).

**EN** **Salient training hyperparameters** (`config/default.yaml → training`): `epochs=200`, `batch_size=64`, `learning_rate=1e-4`, `weight_decay=1e-3` (small-dataset anti-overfit), `patience=15`, `cosine` warm-restart scheduler (T0=10, T_mult=2), `grad_clip_norm=0.5`, `n_ensemble=5`, `asymmetry_alpha=2.0`/`asymmetry_threshold=0.004`. RevIN (Kim ICLR 2022) optional via `model.use_revin` (default false).

### 4.4 Simulazione Monte Carlo · Monte Carlo simulation

🇮🇹 2000 scenari × 30 barre, volatilità via **GJR-GARCH(1,1)** (`omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875`, leverage effect su shock negativi). ⚠ Parametri stimati su rendimenti **1m**: a 1h ω va RI-STIMATO (ha unità [varianza/passo]). Il MC **non è sul critical path del backtest** (usa μ/σ del modello, non GARCH).

**EN** 2000 scenarios × 30 bars, volatility via **GJR-GARCH(1,1)** (`omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875`, leverage effect on negative shocks). ⚠ Params estimated on **1m** returns: at 1h ω must be RE-FITTED (units [variance/step]). MC is **not on the backtest critical path** (uses model μ/σ, not GARCH).

---

## 5. Valutazione · Evaluation

### 5.1 Giudici linea VOL · VOL-line judges

🇮🇹 Metrica primaria: **QLIKE** (loss di volatilità, robusta) + ratio NN/HAR-RV. Baseline: **HAR-RV** per-fold. Giudici (NON un backtest trading — i modelli vol non vengono mai tradati nel backtest): `scripts/vol/dev_vols_qlike.py` (QLIKE log-RV), `scripts/vol/dev_vols_rs_judge.py` (asimmetria semivarianza), `scripts/vol/wf_har_baseline.py` (HAR per-fold), `scripts/vol/step0_xarch_corr.py` (kill-check correlazione cross-arch). Split val-first via `QUANTSYS_VOLS_SPLIT=val|test`. Logica condivisa (QLIKE / inversione log-RV / HAR) in `quantsys/model/vol_metrics.py`. **Risultato chiave:** NN-log_rv 0.257 QLIKE vs HAR-RV 0.368 vs naive 0.807 su test 1h (−30%).

**EN** Primary metric: **QLIKE** (robust volatility loss) + NN/HAR-RV ratio. Baseline: per-fold **HAR-RV**. Judges (NOT a trading backtest — vol models are never traded in the backtest): `scripts/vol/dev_vols_qlike.py` (log-RV QLIKE), `scripts/vol/dev_vols_rs_judge.py` (semivariance asymmetry), `scripts/vol/wf_har_baseline.py` (per-fold HAR), `scripts/vol/step0_xarch_corr.py` (cross-arch correlation kill-check). Val-first split via `QUANTSYS_VOLS_SPLIT=val|test`. Shared logic (QLIKE / log-RV inversion / HAR) in `quantsys/model/vol_metrics.py`. **Key result:** NN-log_rv 0.257 QLIKE vs HAR-RV 0.368 vs naive 0.807 on 1h test (−30%).

### 5.2 Walk-forward & backtest · Walk-forward & backtest

🇮🇹 **Walk-forward** purged k-fold con embargo anti-leakage (`scripts/02b_walkforward_validate.py`). `n_folds=6` → **5 fold effettivi** (fold 0 scartato strutturalmente: `train_end = fold_size − embargo < fold_size`); per K effettivi → `n_folds=K+1`. `embargo_steps=168` (1 settimana a 1h, ≥ `window_size+horizon`=150). **Backtest** (`scripts/03_backtest.py`): fee model + slippage sqrt-impact, stress test (fee×2 slip×3 pessimistic; fee×1.5 slip×5 flash crash), bootstrap CI 5000 iter, analisi per regime, recovery MDD. Soglie trading in `config/default.yaml → backtest` sono in **spazio RAW** — non sovrascrivere da `arch/*.yaml` senza ricalibrare.

**EN** **Walk-forward** purged k-fold with anti-leakage embargo (`scripts/02b_walkforward_validate.py`). `n_folds=6` → **5 effective folds** (fold 0 structurally skipped: `train_end = fold_size − embargo < fold_size`); for K effective → `n_folds=K+1`. `embargo_steps=168` (1 week at 1h, ≥ `window_size+horizon`=150). **Backtest** (`scripts/03_backtest.py`): fee model + sqrt-impact slippage, stress tests (fee×2 slip×3 pessimistic; fee×1.5 slip×5 flash crash), 5000-iter bootstrap CI, regime-conditioned analysis, MDD recovery. Trading thresholds in `config/default.yaml → backtest` are in **RAW space** — do not override from `arch/*.yaml` without recalibrating.

🇮🇹 ⚠ **Distribution shift val→test (fatto strutturale, misurato a 1m).** Sul filone **direzionale** le metriche in-sample (val_nll, Spearman/WHR walkforward) **anti-correlano** col backtest — NON ottimizzare regole guidate da metriche in-sample. L'errore cross-arch è ≈0.995 → ensembling matematicamente quasi inutile (riduzione varianza ≈0). L'anti-correlazione è specifica del **target direzionale**: sul target `log_rv` val→test sono coerenti.

**EN** ⚠ **val→test distribution shift (structural fact, measured at 1m).** On the **directional** line in-sample metrics (val_nll, Spearman/WHR walkforward) **anti-correlate** with the backtest — do NOT optimize rules driven by in-sample metrics. Cross-arch error is ≈0.995 → ensembling is mathematically near-useless (variance reduction ≈0). The anti-correlation is specific to the **directional target**: on the `log_rv` target val→test are coherent.

### 5.3 Test · Tests

```bash
pytest tests/                          # suite completa / full suite
pytest tests/test_recent_fixes.py -v   # regression sui fix critici (z-score, RevIN, BLOCKER #1)
```

🇮🇹 La suite copre FeatureBuilder no-leakage, invarianti RobustScaler, gradient check su t-Student NLL, (de)serializzazione PipelineState, edge case Kelly sizing, trigger circuit-breaker, golden snapshot del live feature buffer, parity live↔training, distillation, CAFN e vol_metrics. Dopo ogni fix con impatto su shape/scaler/feature: aggiungi un regression test e ri-allinea i golden.

**EN** The suite covers FeatureBuilder no-leakage, RobustScaler invariants, t-Student NLL gradient checks, PipelineState (de)serialization, Kelly sizing edge cases, circuit-breaker triggers, golden snapshots of the live feature buffer, live↔training parity, distillation, CAFN and vol_metrics. After any fix impacting shape/scaler/features: add a regression test and re-align the golden snapshots.

---

## 6. Deploy & inferenza · Deploy & Inference

### 6.1 Pipeline & live · Pipeline & live

```bash
# Pipeline completa (menu interattivo) / Full pipeline (interactive menu)
python run_all.py

# Architettura specifica / Specific architecture
python run_all.py --arch itransformer
python run_all.py --arch nhits
python run_all.py --arch tcnmamba
python run_all.py --arch lstm           # legacy backward compat

# Knowledge Distillation (train tutte + multi-teacher weighted + distill student)
python run_all.py --distill

# Solo dashboard / Dashboard only
python run_all.py --only-dashboard

# Diagnostica / Diagnostics
python scripts/07_verify_teacher.py             # confronto architetture / arch comparison
python scripts/99_replay_live_vs_training.py    # diagnostica BLOCKER #1 / parity check
```

🇮🇹 Catena d'inferenza: forward → `PipelineState.denormalize_predictions(μ, σ)` (z-score → raw) → conviction score (direzione × ampiezza × calibrazione × regime) → **Risk Manager** (Kelly frazionario ∝ edge ∝ 1/varianza, max 1%/trade; SL ATR 3×; trailing; circuit breaker DD 15% MtM intra-trade) → BUY/SELL/HOLD + size + SL + TP. Path live di produzione: `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canoniche, scaler da `PipelineState`) → `LiveEngine._deterministic_predict` (nucleo condiviso col backtest) → `denormalize_predictions` → `SignalGenerator`. Feed Binance WebSocket con reconnect exponential-backoff, persistenza stato, Volume Profile incrementale, funding refresh thread-safe.

**EN** Inference chain: forward → `PipelineState.denormalize_predictions(μ, σ)` (z-score → raw) → conviction score (direction × magnitude × calibration × regime) → **Risk Manager** (fractional Kelly ∝ edge ∝ 1/variance, max 1%/trade; ATR SL 3×; trailing; 15% MtM intra-trade drawdown circuit breaker) → BUY/SELL/HOLD + size + SL + TP. Production live path: `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canonical, scaler from `PipelineState`) → `LiveEngine._deterministic_predict` (core shared with the backtest) → `denormalize_predictions` → `SignalGenerator`. Binance WebSocket feed with exponential-backoff reconnect, state persistence, incremental Volume Profile, thread-safe funding refresh.

🇮🇹 **Safety net da preservare (non rimuovere):** `RuntimeError` se `σ_max ≥ 0.05·√interval_minutes` raw; validazione `forecast_horizon` E `interval` train vs backtest/live (`RuntimeError` su mismatch); `merge_asof` test↔raw_candles con `len == n_test`; floor `sl_d = max(sl_d, price×1e-4)`; checkpoint atomici (`.tmp` + `os.replace`).

**EN** **Safety net to preserve (do not remove):** `RuntimeError` if `σ_max ≥ 0.05·√interval_minutes` raw; `forecast_horizon` AND `interval` train-vs-backtest/live validation (`RuntimeError` on mismatch); `merge_asof` test↔raw_candles with `len == n_test`; floor `sl_d = max(sl_d, price×1e-4)`; atomic checkpoints (`.tmp` + `os.replace`).

### 6.2 Dashboard — Deribit Options Risk Terminal

🇮🇹 `scripts/06_dashboard.py` è un **terminale istituzionale per l'analisi delle opzioni crypto**, server HTTP single-file + SPA (Plotly.js), GPU-free e **indipendente dalla pipeline ML**. Si connette ai dati **pubblici Deribit** (REST, no-auth): prezzo indice spot BTC + chain opzioni completa (mark/bid/ask, `mark_iv`, open interest, volume, forward per-expiry) + indice DVOL. Calcola le **Greche in tempo reale** sull'intera option chain (Black-Scholes forward-measure, r=0, convenzione USD: Δ delta, Γ gamma, ν vega per +1% vol, Θ theta/giorno, ρ), mostrate col segno esplicito (+/−) e il nome tra parentesi. Quattro viste: **Volatility Surface** (superficie IV 3D moneyness K/F × giorni, smile per scadenza, term structure ATM), **Option Chain** (call/put a doppio lato con Greche live, strike ATM evidenziato), **Risk & Greeks** (OI per strike, max-pain, Greche aggregate pesate per OI, put/call ratio, DVOL), **Trades** (forward test vol `04b`: storico settled da `trades.jsonl` + **posizione aperta** da `position.json`, lato LONG/SHORT straddle, prezzo ingresso/settlement, premio/payoff/PnL, profilo di rischio/payoff con formula di settlement identica a `04b` e **breakeven** espliciti; endpoint `/api/trades`). Avvio: `python scripts/06_dashboard.py` o `python run_all.py --only-dashboard` → `http://localhost:8050`. Underlying via `config/default.yaml → dashboard.options_currency` (BTC|ETH).

**EN** `scripts/06_dashboard.py` is an **institutional crypto-options analytics terminal**, single-file HTTP server + SPA (Plotly.js), GPU-free and **decoupled from the ML pipeline**. It connects to **Deribit public** data (REST, no-auth): BTC spot index price + full option chain (mark/bid/ask, `mark_iv`, open interest, volume, per-expiry forward) + DVOL index. It computes **Greeks in real time** over the whole chain (Black-Scholes forward measure, r=0, USD convention: Δ delta, Γ gamma, ν vega per +1% vol, Θ theta/day, ρ), shown with an explicit sign (+/−) and the name in parentheses. Four views: **Volatility Surface** (3D IV surface, moneyness K/F × days, per-expiry smile, ATM term structure), **Option Chain** (two-sided call/put with live Greeks, ATM strike highlighted), **Risk & Greeks** (OI by strike, max-pain, OI-weighted aggregate Greeks, put/call ratio, DVOL), **Trades** (vol forward test `04b`: settled history from `trades.jsonl` + **open position** from `position.json`, LONG/SHORT straddle side, entry/settlement price, premium/payoff/PnL, risk/payoff profile with the exact `04b` settlement formula and explicit **breakevens**; `/api/trades` endpoint). Launch: `python scripts/06_dashboard.py` or `python run_all.py --only-dashboard` → `http://localhost:8050`. Underlying via `config/default.yaml → dashboard.options_currency` (BTC|ETH).

---

## 7. Storico esperimenti · Experiment Log

🇮🇹 ⚠ I kill-record sono documentali ("vaccino contro il re-test involontario"). Ogni esito negativo è pre-registrato e conservato di proposito.

**EN** ⚠ Kill-records are documental ("a vaccine against involuntary re-testing"). Every negative outcome is pre-registered and kept on purpose.

🇮🇹
| Esperimento | Data | Esito | Sintesi |
|---|---|---|---|
| VOL-S `log_rv` @ 1h | 2026-06-10 | ✅ PASS | NN batte HAR-RV del 30% in QLIKE su test (0.257 vs 0.368; naive 0.807), val→test coerenti. B2 chiusa positiva. Giudice `dev_vols_qlike.py`. |
| VOL-S `log_rv` @ 1m | 2026-06-10 | ⚫ FAIL | Cross-risoluzione pre-registrata: NN/HAR QLIKE 1.0127 > 0.95 → l'edge vol è SPECIFICO della risoluzione 1h. |
| Semivarianza `log_rs_ratio` | 2026-06-11 | ⚫ FAIL | log(RS⁺/RS⁻) impredicibile per NN E HAR-RS (MSE 0.9952, signDA 0.459; HAR-RS peggio della costante) → **i momenti pari generalizzano OOS, i dispari no**. Giudice `dev_vols_rs_judge.py`. |
| Pivot direzionale 1m→1h | 2026-06-10 | ⚫ KILLED | Muro costi sfondato (|μ|≈43bps ≫ 26bps) ma zero skill direzionale OOS; anti-corr val→test confermata a 1h. Gate 4/4 fallito a 13 E 23 bps. |
| Probe cross-sectional (XS) | 2026-06-06 | ⚫ KILL | Muro a 1m = MAGNITUDINE (~1.5 bps effetto vs ~26 bps costo round-trip), non il segno. Script in `scripts/archive/xs_*`. |
| Entry rank-based regime Quiet | 2026-06-05 | ⚫ FAIL su val | Rank-entry su val held-out: return −0.22%, PF 0.84, 13 trade → overfit del test. Edge a soglia/rank inesistente OOS. |
| Cadenza + esposizione rank continua | 2026-06-05 | ⚫ FAIL su val | Fix ①② vs baseline (+4.03%/PF 1.88) → −2.24%/PF 0.22. Rank anti-predittivo OOS; PnL dominata dal path SL/TP. Flag inerti. |
| Calibrazione-σ (`QUANTSYS_SIGMA_SCALE`) | — | ⚫ FAIL | σ↓ peggiora il backtest; l'ottimo è ≈1.0. Flag inerte. |
| IVS relative-value (taker) | 2026-06-24 | ⚫ KILL net-of-cost | Struttura reale (residui revertono) ma netto −2.3/−3.8 vol-pt/leg (~50× sotto lo spread). Vivrebbe solo da market-maker. |
| Short-vol arm (backtest storico FHS GJR-GARCH) | 2026-06-26 | 🟡 conferma strutturale | n=2538 scadenze daily 2019→2026: break-even VRP=0% per tutte le strutture; strangle 8-10% tail-safe; edge sopravvive al bid (haircut 16%). **Audit 2026-06-26:** edge **Trending-driven** (NON Stress — la "Stress=miglior Sharpe" era artefatto dell'haircut costante, ora regime-dipendente); CI block-bootstrap>0, N_eff≈N; "non filtrare il regime" regge; martingale-correction respinta (kurtosis residui ≈19.7). NON un PASS del gate (resta live n≥20). Script `scripts/vol/short_vol_*`. |

**EN**
| Experiment | Date | Outcome | Summary |
|---|---|---|---|
| VOL-S `log_rv` @ 1h | 2026-06-10 | ✅ PASS | NN beats HAR-RV by 30% in test QLIKE (0.257 vs 0.368; naive 0.807), val→test coherent. B2 closed positive. Judge `dev_vols_qlike.py`. |
| VOL-S `log_rv` @ 1m | 2026-06-10 | ⚫ FAIL | Pre-registered cross-resolution: NN/HAR QLIKE 1.0127 > 0.95 → the vol edge is SPECIFIC to the 1h resolution. |
| Semivariance `log_rs_ratio` | 2026-06-11 | ⚫ FAIL | log(RS⁺/RS⁻) unpredictable for NN AND HAR-RS (MSE 0.9952, signDA 0.459; HAR-RS worse than the constant) → **even moments generalize OOS, odd ones don't**. Judge `dev_vols_rs_judge.py`. |
| Directional pivot 1m→1h | 2026-06-10 | ⚫ KILLED | Cost wall broken (|μ|≈43bps ≫ 26bps) but zero OOS directional skill; val→test anti-corr confirmed at 1h. Gate 4/4 failed at 13 AND 23 bps. |
| Cross-sectional probe (XS) | 2026-06-06 | ⚫ KILL | The 1m wall is MAGNITUDE (~1.5 bps effect vs ~26 bps round-trip cost), not sign. Scripts in `scripts/archive/xs_*`. |
| Rank-based entry, Quiet regime | 2026-06-05 | ⚫ FAIL on val | Rank-entry on held-out val: return −0.22%, PF 0.84, 13 trades → test overfit. Threshold/rank edge nonexistent OOS. |
| Cadence + continuous rank exposure | 2026-06-05 | ⚫ FAIL on val | Fix ①② vs baseline (+4.03%/PF 1.88) → −2.24%/PF 0.22. Rank anti-predictive OOS; PnL dominated by SL/TP path. Inert flags. |
| σ calibration (`QUANTSYS_SIGMA_SCALE`) | — | ⚫ FAIL | σ↓ worsens the backtest; optimum ≈1.0. Inert flag. |
| IVS relative-value (taker) | 2026-06-24 | ⚫ KILL net-of-cost | Real structure (residuals revert) but net −2.3/−3.8 vol-pt/leg (~50× below the spread). Would live only as a market-maker. |
| Short-vol arm (FHS GJR-GARCH historical backtest) | 2026-06-26 | 🟡 structural confirmation | n=2538 daily expiries 2019→2026: break-even VRP=0% for all structures; strangle 8-10% tail-safe; edge survives the bid (16% haircut). **Audit 2026-06-26:** edge **Trending-driven** (NOT Stress — "Stress=highest Sharpe" was a constant-haircut artifact, now regime-dependent); block-bootstrap CIs>0, N_eff≈N; "do not filter regime" holds; martingale correction refuted (residual kurtosis ≈19.7). NOT a gate PASS (live n≥20 remains). Scripts `scripts/vol/short_vol_*`. |

🇮🇹 **Flag sperimentali env-gated (inerti di default, in `03_backtest.py`):** `QUANTSYS_REGIME_ALLOW`/`_INVERT`, `QUANTSYS_QUIET_*` (rank-entry discreta), `QUANTSYS_DECISION_CADENCE`, `QUANTSYS_RANK_*` (esposizione continua), `QUANTSYS_HORIZON_EXIT`, `QUANTSYS_SIGMA_SCALE`, `QUANTSYS_MIN_EXPECTED_RET`. **Tutti validati e FALLITI OOS** — restano documentati per non ri-testarli. Env runtime non-sperimentali: `QUANTSYS_ARCH`, `QUANTSYS_MODELS_ROOT` (sandbox modelli), `QUANTSYS_VOLS_SPLIT`, `QUANTSYS_BACKTEST_SPLIT=val|test`, `QUANTSYS_BACKTEST_SINGLE_ARCH`.

**EN** **Env-gated experimental flags (inert by default, in `03_backtest.py`):** `QUANTSYS_REGIME_ALLOW`/`_INVERT`, `QUANTSYS_QUIET_*` (discrete rank-entry), `QUANTSYS_DECISION_CADENCE`, `QUANTSYS_RANK_*` (continuous exposure), `QUANTSYS_HORIZON_EXIT`, `QUANTSYS_SIGMA_SCALE`, `QUANTSYS_MIN_EXPECTED_RET`. **All validated and FAILED OOS** — kept documented to avoid re-testing. Non-experimental runtime env: `QUANTSYS_ARCH`, `QUANTSYS_MODELS_ROOT` (models sandbox), `QUANTSYS_VOLS_SPLIT`, `QUANTSYS_BACKTEST_SPLIT=val|test`, `QUANTSYS_BACKTEST_SINGLE_ARCH`.

---

## 8. Architettura del sistema · System Architecture

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

### 8.1 Struttura del progetto · Project Structure

```
quantsys_project/
├── config/
│   ├── default.yaml              parametri condivisi (data, features, model, training, risk, distillation)
│   ├── secrets.yaml.example      template per API keys (copia in secrets.yaml)
│   ├── interval/                 override risoluzione candela (QUANTSYS_INTERVAL / --interval)
│   │   ├── 1m.yaml               chiavi interval-dipendenti era 1m (stride 5, embargo 1500, ...)
│   │   └── 1h.yaml               chiavi interval-dipendenti pivot 1h (stride 1, embargo 168, ...)
│   ├── arch/
│   │   ├── lstm.yaml             override LSTM
│   │   ├── itransformer.yaml     override iTransformer
│   │   ├── nhits.yaml            override N-HiTS
│   │   └── tcnmamba.yaml         override TCN+Mamba
│   └── cafn.yaml                 overlay opzionale CAFN (probe, non letto dalla pipeline production)
├── quantsys/                     package Python installabile
│   ├── data/                     Binance REST + WebSocket + funding rate
│   ├── features/                 FeatureBuilder (104 feature post C-funding, split dual-stream)
│   ├── macro/                    FRED + yFinance + RegimeMarkovBTC (Markov-Switching su realized vol BTC)
│   ├── model/
│   │   ├── __init__.py           QuantLSTM, QuantTFT, QuantiTransformer
│   │   ├── nhits.py              QuantNHiTS (pure-MLP gerarchico)
│   │   ├── tcn_mamba.py          QuantTCNMamba (TCN + Mamba SSM + gated fusion)
│   │   ├── ensemble.py           EnsembleModel (omogeneo / eterogeneo, AMP off in inferenza)
│   │   ├── distillation.py       Knowledge Distillation multi-teacher (scoring target-aware)
│   │   ├── forecast.py           Monte Carlo GJR-GARCH(1,1) + neural-guided
│   │   ├── vol_metrics.py        QLIKE / inversione log-RV / baseline HAR-RV (linea vol, condivisi)
│   │   ├── cafn.py               CausalAttentionFlowNetwork (coordinatore causale, probe inerte)
│   │   └── revin.py              Reversible Instance Normalization (opzionale)
│   ├── trading/                  Kelly sizing, SL dinamico, trailing, circuit breaker
│   │                             + greeks_risk.py (A7: cap vega/delta, CB vega-loss, margin sim Deribit — skeleton, non cablato / not wired)
│   └── utils/                    config loader, device setup, logging, PipelineState
├── scripts/
│   ├── 00_check_setup.py         verifica CUDA, dipendenze, connessione Binance
│   ├── 00_test_binance_testnet.py  check connettività API testnet
│   ├── 01_download_data.py       Binance → 104 feature (C-funding) → lstm_dataset.npz
│   ├── 01_update_data.py         aggiornamento incrementale (solo delta candele)
│   ├── 01b_download_macro.py     FRED + yFinance → RegimeMarkovBTC su candele BTC → update dataset
│   ├── 01c_iv_poller.py          poller IV Deribit (chain opzioni BTC + DVOL) → data/iv/
│   ├── 01d_orderbook_recorder.py recorder order-book L2 Binance (B1: microstruttura — OFI/imbalance/depth) → data/orderbook/
│   ├── 02_train.py               training con --arch, --distill, ensemble
│   ├── 02b_walkforward_validate.py   walk-forward purged k-fold con embargo
│   ├── 02c_optuna_search.py      Bayesian hyperparameter search (solo LSTM)
│   ├── 02d_cafn_joint_train.py   CAFN: coordinatore causale + training congiunto 3 modelli (probe)
│   ├── 03_backtest.py            backtest + stress test + bootstrap CI
│   ├── 04_live_signals.py        feed live WebSocket + paper trading (direzionale)
│   ├── 04b_vol_paper.py          forward test vol: NN-RV vs IV → straddle testnet Deribit
│   │                             (+ leg delta-hedge perp v2 dietro --hedge, inerte di default)
│   ├── 04c_vol_paper_baselines.py  baseline always-long/short-vol (gate pre-reg) dai chain snapshot
│   ├── 05_analyze_signals.py     analisi sessione live
│   ├── 06_dashboard.py           Deribit Options Risk Terminal (HTTP single-file + Plotly)
│   ├── 07_verify_teacher.py      confronto architetture per selezione teacher
│   ├── 99_replay_live_vs_training.py   diagnostica BLOCKER #1 (parity live vs training)
│   ├── vol/                      linea vol: giudici QLIKE/RS (dev_vols_qlike, dev_vols_rs_judge),
│   │                             prep dati (dev_vols_macro_append), kill-check cross-arch
│   │                             (step0_xarch_corr), baseline HAR per-fold (wf_har_baseline),
│   │                             short-vol (short_vol_*), IVS relative-value (ivs_*)
│   ├── research/                 paper / negative-control direzionale: paper_01_dir_baselines.py
│   ├── README.md                 mappa script → linea (shared / vol / direzionale)
│   └── archive/                  probe chiusi (xs cross-sectional KILL, step0 σ-recal)
├── tests/                        suite pytest (features, NLL, PipelineState, regression sui fix recenti)
├── data/                         generato (gitignored)
├── models/                       checkpoint per architettura (gitignored)
├── results/                      backtest + segnali live per architettura (gitignored)
└── logs/                         log rotanti (gitignored)
```

🇮🇹 Vedi [AVVIO.md](AVVIO.md) per la guida operativa completa e [TEORIA.md](TEORIA.md) per i fondamenti teorici.

**EN** See [AVVIO.md](AVVIO.md) for the full operational guide and [TEORIA.md](TEORIA.md) for theoretical foundations.

---

## Licenza · License

🇮🇹 [MIT License](LICENSE) — codice di ricerca, non consulenza finanziaria.

**EN** [MIT License](LICENSE) — research code, not financial advice.
