# QUANTSYS — Motore neurale di forecasting per BTC/USDT · QUANTSYS — Neural Forecasting Engine for BTC/USDT

🇮🇹 Motore neurale di forecasting probabilistico su BTC/USDT + analytics opzioni crypto. **Linea di produzione: volatilità @ 1 ora** (`config/default.yaml → features.target_type: log_rv`, `data.interval: 1h`; design interval-agnostic, 1m = identità, perimetro 1m in backup). Il target `log_rv` è l'**unico segnale validato OOS** del progetto: batte HAR-RV del 30% in QLIKE su test (0.257 vs 0.368; naive 0.807), con val→test coerenti. Modello di produzione: **iTransformer 5 membri**. Secondo braccio attivo: **short-vol** in forward test su Deribit testnet (`scripts/04b_vol_paper.py`, servizio systemd 24/7 su VPS). Il filone **direzionale** non ha alpha OOS a nessun timeframe testato (1m e 1h): il codice resta vivo e bit-invariato come negative-control documentato.

**EN** Probabilistic neural forecasting engine for BTC/USDT + crypto-options analytics. **Production line: volatility @ 1 hour** (`config/default.yaml → features.target_type: log_rv`, `data.interval: 1h`; interval-agnostic design, 1m = identity, the 1m perimeter is backed up). The `log_rv` target is the project's **only OOS-validated signal**: it beats HAR-RV by 30% in test QLIKE (0.257 vs 0.368; naive 0.807), with coherent val→test. Production model: **5-member iTransformer**. Second active arm: **short-vol** forward test on Deribit testnet (`scripts/04b_vol_paper.py`, 24/7 systemd service on a VPS). The **directional** line has no OOS alpha at any tested timeframe (1m and 1h): the code stays alive and bit-invariant as a documented negative control.

🇮🇹 **Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API · Deribit public REST (dashboard/IV/forward test vol).

**EN** **Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API · Deribit public REST (dashboard/IV/vol forward test).

### Da dove iniziare · Start here

🇮🇹 Lettura in 60 secondi — quattro puntatori, in ordine di importanza:

| Cosa | Dove |
|---|---|
| **Il risultato.** Il forecast NN della realized variance batte HAR-RV del **30% in QLIKE su test** (0.257 vs 0.368; naive 0.807 — Diebold-Mariano HAC **p ≤ 1.7·10⁻⁶**, vedi `TEORIA.md` §12.2), con val→test coerenti | `scripts/vol/dev_vols_qlike.py` — il giudice che produce il numero, split val-first |
| **Come si decide se un'idea vive o muore.** Ogni esperimento è **pre-registrato**: metriche, soglie e n minimo scritti e committati *prima* di girare | `TEORIA.md` §12.1 (protocollo in 5 passi) · `STATUS.md` (pre-registrazioni in testa) |
| **Cosa è stato provato e NON funziona**, con i numeri: direzionale a 1m e 1h, semivarianza firmata, IVS relative-value, 4 lever di training, gating per regime | `TEORIA.md` §12.3-12.4 (corpus KILL, con i numeri) |
| **Cosa può verificare un lettore esterno** senza scaricare dati | `pytest tests/` → **307 passed, 1 skipped**: parity live↔training bit-perfect, invarianti z-score/interval, bit-parity del regime incrementale |

🇮🇹 Il progetto è organizzato attorno a un'asimmetria dichiarata: **i momenti pari (varianza, RV) generalizzano fuori campione su questo asset, i momenti dispari (segno, asimmetria) no** — per la rete *e* per le baseline econometriche. Le tre linee di codice (vol-forecasting, monetizzazione short-vol, direzionale) esistono per documentare quella asimmetria, non per nasconderla.

**EN** 60-second read — four pointers, most important first:

| What | Where |
|---|---|
| **The result.** The NN realized-variance forecast beats HAR-RV by **30% in test QLIKE** (0.257 vs 0.368; naive 0.807), val→test coherent | `scripts/vol/dev_vols_qlike.py` — the judge that produces the number, val-first split |
| **How an idea lives or dies.** Every experiment is **pre-registered**: metrics, thresholds and minimum n written and committed *before* running | `TEORIA.md` §12.1 (5-step protocol) · `STATUS.md` (pre-registrations on top) |
| **What was tried and does NOT work**, with numbers: directional at 1m and 1h, signed semivariance, IVS relative-value, 4 training levers, regime gating | `TEORIA.md` §12.3-12.4 (KILL corpus, with numbers) |
| **What an outside reader can verify** without downloading data | `pytest tests/` → **307 passed, 1 skipped**: bit-perfect live↔training parity, z-score/interval invariants, incremental-regime bit-parity |

**EN** The project is organized around a stated asymmetry: **even moments (variance, RV) generalize out-of-sample on this asset, odd moments (sign, skew) do not** — for the network *and* for the econometric baselines. The three code lines (vol forecasting, short-vol monetization, directional) exist to document that asymmetry, not to hide it.

### Riproducibilità · Reproducibility

🇮🇹 `data/`, `models/` e `results/` sono **gitignored**: pesi e parquet sono grandi e i dati di mercato non sono ridistribuibili. Cosa significa in pratica per chi clona:

- **Verificabile subito, senza dati:** `pip install -e .` → `pytest tests/` (308 test, ~30s, CPU). Include i golden test sulla lista delle 104 feature e la parity live↔training.
- **Rigenerabile:** dataset (`scripts/01_download_data.py`, Binance pubblico + una chiave FRED gratuita per la macro) → training (`scripts/02_train.py --n-ensemble 5`, ~27 min per 5 seed iTransformer su RTX 2070 Super) → giudice QLIKE (`scripts/vol/dev_vols_qlike.py`).
- **NON rigenerabile** (raccolta forward, per costruzione): `data/iv/`, `data/orderbook/`, `data/deribit_trades/`, `results/vol_paper/` — snapshot IV/book/trade e forward test su testnet. I numeri del braccio short-vol non sono riproducibili da un clone: sono un log d'esperimento, e sono presentati come tali.

**EN** `data/`, `models/` and `results/` are **gitignored**: weights and parquets are large and market data is not redistributable. What that means when you clone:

- **Verifiable immediately, no data needed:** `pip install -e .` → `pytest tests/` (308 tests, ~30s, CPU), including golden tests on the 104-feature list and live↔training parity.
- **Regenerable:** dataset (`scripts/01_download_data.py`, public Binance + a free FRED key for macro) → training (`scripts/02_train.py --n-ensemble 5`, ~27 min for 5 iTransformer seeds on an RTX 2070 Super) → QLIKE judge (`scripts/vol/dev_vols_qlike.py`).
- **NOT regenerable** (forward collection, by construction): `data/iv/`, `data/orderbook/`, `data/deribit_trades/`, `results/vol_paper/` — IV/book/trade snapshots and the testnet forward test. The short-vol arm's numbers cannot be reproduced from a clone: they are an experiment log, and are presented as such.

### Mappa della documentazione · Documentation map

🇮🇹 Documentazione **bilingue in un unico file** (IT + EN per paragrafo, marker 🇮🇹/**EN**). Ruoli disgiunti — ogni fatto vive in un solo posto:

- **[README.md](README.md)** (questo file) — *cosa* fa il sistema e *perché*: panoramica, caratteristiche, struttura del progetto, puntatori.
- **[AVVIO.md](AVVIO.md)** — **runbook**: tutti i comandi operativi, setup dettagliato, tuning hardware, routine di sessione, deploy VPS.
- **[TEORIA.md](TEORIA.md)** — **matematica**: derivazioni di loss, regime detection, Monte Carlo, distillation, trading layer.
- **[CHANGELOG.md](CHANGELOG.md)** — milestone in ordine cronologico inverso.
- **[STATUS.md](STATUS.md)** — **fonte canonica** dello stato: periodo corrente + tutti i gate pre-registrati aperti. Storico antecedente al 2026-07-08 in **[docs/STATUS_ARCHIVE_2026H1.md](docs/STATUS_ARCHIVE_2026H1.md)** (read-only).
- **[TEORIA.md](TEORIA.md) §12** — protocollo sperimentale + corpus dei risultati negativi (KILL) con le soglie di gate.
- **[docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md)** · **[docs/ROADMAP_VOL_BOOK.md](docs/ROADMAP_VOL_BOOK.md)** — backlog e item aperti.

**EN** **Single-file bilingual** documentation (IT + EN per paragraph, markers 🇮🇹/**EN**). Disjoint roles — every fact lives in exactly one place:

- **[README.md](README.md)** (this file) — *what* the system does and *why*: overview, features, project structure, pointers.
- **[AVVIO.md](AVVIO.md)** — **runbook**: every operational command, detailed setup, hardware tuning, session routine, VPS deploy.
- **[TEORIA.md](TEORIA.md)** — **mathematics**: derivations for the loss, regime detection, Monte Carlo, distillation, trading layer.
- **[CHANGELOG.md](CHANGELOG.md)** — milestones, reverse chronological.
- **[STATUS.md](STATUS.md)** — **canonical source of truth** for state: current period + every open pre-registered gate. History predating 2026-07-08 in **[docs/STATUS_ARCHIVE_2026H1.md](docs/STATUS_ARCHIVE_2026H1.md)** (read-only).
- **[TEORIA.md](TEORIA.md) §12** — experimental protocol + negative-results (KILL) corpus with gate thresholds.
- **[docs/MODEL_IMPROVEMENTS.md](docs/MODEL_IMPROVEMENTS.md)** · **[docs/ROADMAP_VOL_BOOK.md](docs/ROADMAP_VOL_BOOK.md)** — backlog and open items.

---

## 1. Panoramica e obiettivi · Overview & Goals

🇮🇹 Anziché una stima puntuale, QUANTSYS restituisce l'**intera distribuzione** del target — quantili condizionali con `loss_type: quantile` (default di produzione), oppure head parametrica t-Student μ/σ/ν con `loss_type: t_student` (§4.3) — su BTC/USDT a **intervallo candela parametrico** (`data.interval`; default corrente `1h`, perimetro `1m` legacy in backup — tutte le conversioni temporali derivano da `interval_minutes`, identità a 1m). Sullo stesso spine convivono tre linee, con esiti molto diversi:

- **Linea VOL — forecasting (produzione, validata).** Target `log_rv` (log realized variance su h barre): unico segnale che generalizza OOS, batte HAR-RV del 30% in QLIKE. Giudicata con QLIKE, **mai** tradata nel backtest direzionale.
- **Linea VOL — monetizzazione (forward test in corso).** Braccio **short-vol**: straddle su Deribit testnet guidati dal confronto RV-prevista vs IV (`04b_vol_paper.py`, systemd 24/7 su VPS, delta-hedge attivo). L'edge VRP è strutturalmente confermato sul backtest storico FHS-GJR-GARCH (2019→2026, n=2538), ma il **gate v1 pre-registrato è FAIL 0/3 (2026-07-18)**: il VRP è positivo, la regola v1 non lo monetizza. Gate successivi in accumulo — contatori e scadenze in `STATUS.md`.
- **Linea direzionale (legacy, negative-control).** Target `ret`, trading layer Kelly/SL/TP, backtest e live paper. Nessun alpha OOS né a 1m né a 1h; conservata come negative-control documentato, non ri-aprirla senza un'ipotesi nuova.

**EN** Rather than a point estimate, QUANTSYS outputs the **full distribution** of the target — conditional quantiles with `loss_type: quantile` (the production default), or a parametric Student-t head μ/σ/ν with `loss_type: t_student` (§4.3) — on BTC/USDT at a **parametric candle interval** (`data.interval`; current default `1h`, legacy `1m` perimeter backed up — all temporal conversions derive from `interval_minutes`, identity at 1m). Three lines coexist on the same spine, with very different outcomes:

- **VOL line — forecasting (production, validated).** Target `log_rv` (log realized variance over h bars): the only signal that generalizes OOS, beating HAR-RV by 30% in QLIKE. Judged via QLIKE, **never** traded in the directional backtest.
- **VOL line — monetization (forward test running).** **Short-vol** arm: Deribit-testnet straddles driven by predicted-RV vs IV (`04b_vol_paper.py`, 24/7 systemd on a VPS, delta-hedge active). The VRP edge is structurally confirmed by the FHS-GJR-GARCH historical backtest (2019→2026, n=2538), but the **pre-registered v1 gate is FAIL 0/3 (2026-07-18)**: VRP is positive, the v1 rule does not monetize it. Later gates are accruing — counters and deadlines in `STATUS.md`.
- **Directional line (legacy, negative control).** Target `ret`, Kelly/SL/TP trading layer, backtest and live paper. No OOS alpha at 1m or 1h; kept as a documented negative control, do not re-open it without a new hypothesis.

🇮🇹 ✅ **Stato live engine (direzionale):** paper-only (nessun ordine reale), **BLOCKER #1 RISOLTO (2026-06-05)**. Il path live costruisce le **104 feature canoniche** via `FeatureBuilder` con lo scaler del training, con **parity feature *e* segnale bit-perfect** vs backtest (`tests/test_live_training_parity.py`, replay Δ=0). ⚠ I segnali live riflettono un backtest che resta negativo OOS: il paper-trading direzionale accumula trade forward, senza aspettativa di Sharpe>0.

**EN** ✅ **Live engine status (directional):** paper-only (no real orders), **BLOCKER #1 RESOLVED (2026-06-05)**. The live path builds the **104 canonical features** via `FeatureBuilder` with the training scaler, achieving **bit-perfect feature *and* signal parity** vs the backtest (`tests/test_live_training_parity.py`, replay Δ=0). ⚠ Live signals mirror a backtest that stays OOS-negative: directional paper-trading accrues forward trades, with no expectation of Sharpe>0.

### 1.1 Caratteristiche principali · Key Features

🇮🇹
- **Output probabilistico** — quantili condizionali (default) o t-Student (μ, σ, ν); loss NLL + penalità asimmetrica + CRPS + Direction-Value (§4.3).
- **104 feature ingegnerizzate** post-filtro C-funding, lista canonica sotto golden test (§3.2).
- **4 architetture** intercambiabili — iTransformer, N-HiTS, TCN+Mamba, LSTM legacy — dietro un forward contract unico, con ensemble eterogeneo e **distillation multi-teacher** target-aware (§4.1-4.2).
- **Rilevamento regimi BTC** `RegimeMarkovBTC`, Markov-Switching causale con refresh incrementale (§3.3).
- **Giudici della linea vol** QLIKE vs baseline HAR-RV per-fold, split val-first (§5.1).
- **Validazione anti-leakage**: walk-forward purged k-fold con embargo, backtest con stress test, bootstrap CI, analisi per regime (§5.2).
- **Monte Carlo** 2000 scenari GJR-GARCH(1,1), parametri stimati su rendimenti orari (§4.4).
- **Trading layer** Kelly frazionario + SL ATR + trailing + circuit breaker DD 15% MtM (§6.1).
- **Forward test vol** su Deribit testnet con delta-hedge, attribuzione PnL ex-post e **collector 24/7** su VPS (§6.2).
- **Dashboard** Deribit Options Risk Terminal, GPU-free, indipendente dalla pipeline ML (§6.3).

**EN**
- **Probabilistic output** — conditional quantiles (default) or Student-t (μ, σ, ν); NLL loss + asymmetric penalty + CRPS + Direction-Value (§4.3).
- **104 engineered features** post C-funding filter, canonical list under a golden test (§3.2).
- **4 interchangeable architectures** — iTransformer, N-HiTS, TCN+Mamba, legacy LSTM — behind a single forward contract, with heterogeneous ensembling and target-aware **multi-teacher distillation** (§4.1-4.2).
- **BTC regime detection** `RegimeMarkovBTC`, causal Markov-Switching with incremental refresh (§3.3).
- **Vol-line judges** QLIKE vs per-fold HAR-RV baseline, val-first split (§5.1).
- **Anti-leakage validation**: purged k-fold walk-forward with embargo, backtest with stress tests, bootstrap CI, regime-conditioned analysis (§5.2).
- **Monte Carlo** 2000 GJR-GARCH(1,1) scenarios, params estimated on hourly returns (§4.4).
- **Trading layer** fractional Kelly + ATR SL + trailing + 15% MtM drawdown circuit breaker (§6.1).
- **Vol forward test** on Deribit testnet with delta-hedge, ex-post PnL attribution and **24/7 collectors** on a VPS (§6.2).
- **Dashboard** Deribit Options Risk Terminal, GPU-free, decoupled from the ML pipeline (§6.3).

---

## 2. Setup · Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python scripts/00_check_setup.py
```

🇮🇹 `scripts/00_check_setup.py` verifica CUDA, dipendenze e connessione Binance. Il fallback **CPU-only** funziona (rallentamento su training; piena velocità su backtest/live). Setup di riferimento: **RTX 2070 Super (8 GB VRAM)** — il training multi-seed × multi-arch va sequenziale (OOM), e live/paper non va girato in parallelo a training/inferenza (contesa CUDA). La chiave FRED è **opzionale**: copia `config/secrets.yaml.example` in `config/secrets.yaml` (gitignored, mai committato); senza chiave si lavora con rate limit più stretti.

**EN** `scripts/00_check_setup.py` verifies CUDA, dependencies and the Binance connection. The **CPU-only** fallback works (slower training; full speed on backtest/live). Reference setup: **RTX 2070 Super (8 GB VRAM)** — multi-seed × multi-arch training must run sequentially (OOM), and live/paper must not run alongside training/inference (CUDA contention). The FRED key is **optional**: copy `config/secrets.yaml.example` to `config/secrets.yaml` (gitignored, never committed); without a key you work under stricter rate limits.

🇮🇹 → **Setup dettagliato, note PowerShell/Windows, tuning VRAM (4GB / ≥16GB), tempi di training: [AVVIO.md](AVVIO.md) §1.**

**EN** → **Detailed setup, PowerShell/Windows notes, VRAM tuning (4GB / ≥16GB), training timings: [AVVIO.md](AVVIO.md) §1.**

---

## 3. Dati · Data

🇮🇹 La pipeline scarica lo storico BTC/USDT da Binance (default: candele **1h** dal 2019-01-01, ~65k barre), costruisce **104 feature**, le normalizza con un **RobustScaler globale** (mediana/IQR, fittato SOLO sul training — no leakage val/test), e finestra in sequenze **120×104** (`model.window_size: 120` = 5 giorni a 1h; `window_stride: 1`). Split temporale 80/10/10 (`training.val_fraction`/`test_fraction` = 0.1/0.1 → ~6.5k finestre per val e per test sul dataset corrente). I parametri scaler + config feature + `target_scale` + `forecast_horizon` + `interval` sono persistiti in `PipelineState` (unico contratto train↔inference).

**EN** The pipeline downloads BTC/USDT history from Binance (default: **1h** candles from 2019-01-01, ~65k bars), engineers **104 features**, normalizes them with a **global RobustScaler** (median/IQR, fit on training ONLY — no val/test leakage), and windows into **120×104** sequences (`model.window_size: 120` = 5 days at 1h; `window_stride: 1`). Temporal 80/10/10 split (`training.val_fraction`/`test_fraction` = 0.1/0.1 → ~6.5k windows each for val and test on the current dataset). Scaler params + feature config + `target_scale` + `forecast_horizon` + `interval` are persisted in `PipelineState` (the single train↔inference contract).

### 3.1 Target & log-return · Target & log-return

🇮🇹 Tutto lavora su **log-return** (stazionari, simmetrici), mai prezzi assoluti. L'orizzonte è di **30 barre** (`features.forecast_horizon: 30` — 30h a 1h). Cambiarlo richiede ri-generare il dataset e ri-allineare `PipelineState.forecast_horizon` (validato a runtime con `RuntimeError` in backtest e live). Famiglie di target via `features.target_type`: `log_rv` (produzione vol: log realized variance Σr² su h barre), `ret` (direzionale legacy: somma dei log-return futuri), `log_rs_ratio` (probe asimmetria semivarianza, FAIL — §7).

**EN** Everything works on **log-returns** (stationary, symmetric), never absolute prices. The horizon is **30 bars** (`features.forecast_horizon: 30` — 30h at 1h). Changing it requires regenerating the dataset and re-aligning `PipelineState.forecast_horizon` (validated at runtime with `RuntimeError` in backtest and live). Target families via `features.target_type`: `log_rv` (vol production: log realized variance Σr² over h bars), `ret` (legacy directional: sum of future log-returns), `log_rs_ratio` (semivariance asymmetry probe, FAIL — §7).

### 3.2 Le 104 feature · The 104 features

🇮🇹 **104 feature = 86 dinamiche + 18 strutturali**: VWAP + Volume Profile, CVD, microstructure delle candele, funding rate, volatilità multi-finestra, lag returns, encoding temporale, feature interactions, livelli strutturali. Il **filtro C-funding** (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`) rimuove 15 feature — quelle a lookback lungo (90d/365d), frazionalmente differenziate o dipendenti da Volume Profile long — per due motivi cumulabili: permutation importance con ROI ≤ 0 (rumore o dannose) e/o lookback non calcolabile nel buffer live. Tenerle rompe la parity live↔backtest senza guadagno predittivo. La lista è derivata una volta sola da `canonical_feature_columns` sotto golden test: il conteggio è **verificato sul dataset, non assunto**. Dettaglio delle famiglie in [TEORIA.md](TEORIA.md) §3.

**EN** **104 features = 86 dynamic + 18 structural**: VWAP + Volume Profile, CVD, candle microstructure, funding rate, multi-window volatility, lag returns, time encoding, feature interactions, structural levels. The **C-funding filter** (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`) drops 15 features — long-lookback (90d/365d), fractionally differenced, or long-Volume-Profile dependent — for two cumulable reasons: permutation importance with ROI ≤ 0 (noise or harmful) and/or a lookback not computable in the live buffer. Keeping them breaks live↔backtest parity with no predictive gain. The list is derived once by `canonical_feature_columns` under a golden test: the count is **verified on the dataset, not assumed**. Family-by-family detail in [TEORIA.md](TEORIA.md) §3.

### 3.3 Macro & rilevamento regimi · Macro & regime detection

🇮🇹 Le macro USA (FRED + yFinance: DXY, VIX, tassi, oro) alimentano un `MacroEncoder` (16-dim), scollegato dal regime detector. **`RegimeMarkovBTC`** (Markov-Switching, Hamilton 1989) è fittato sulla realized volatility oraria di BTC ed è **CAUSALE by design**: probabilità *filtered* (mai smoothed), Hamilton filter forward-only, walk-forward expanding con burn-in/retrain da `macro.hmm_burn_in_days`/`hmm_retrain_days`. Produce **3 regimi data-driven** persistiti in `data/regime_probs.parquet` (index orario UTC); sul full rebuild 7 anni 1h del 2026-07-15: **R0 31% (quiet, σ²≈0.12) · R1 36% (stress, σ²≈4.8) · R2 33% (mid, σ²≈0.6)**. ⚠ Gli **indici non hanno semantica fissa tra run**: vanno ri-derivati dalle varianze a ogni full rebuild. Refresh **incrementale** disponibile (`01b --regime-incremental`): appende le sole barre nuove partendo da un checkpoint walk-forward, con bit-parity garantita da test. Uso: **stratificazione della val + diagnostica `val_nll` per regime** — **NON è una feature di input**. `RegimeMarkovSwitching` (macro USA daily), `RegimeHMM` e `RegimeSession` (Asia/EU/US) restano come alternative opzionali. Derivazione completa in [TEORIA.md](TEORIA.md) §4.

**EN** US macros (FRED + yFinance: DXY, VIX, rates, gold) feed a `MacroEncoder` (16-dim), decoupled from the regime detector. **`RegimeMarkovBTC`** (Markov-Switching, Hamilton 1989) is fit on hourly BTC realized volatility and is **CAUSAL by design**: *filtered* probabilities (never smoothed), forward-only Hamilton filter, expanding walk-forward with burn-in/retrain from `macro.hmm_burn_in_days`/`hmm_retrain_days`. It yields **3 data-driven regimes** persisted in `data/regime_probs.parquet` (hourly UTC index); on the 2026-07-15 seven-year 1h full rebuild: **R0 31% (quiet, σ²≈0.12) · R1 36% (stress, σ²≈4.8) · R2 33% (mid, σ²≈0.6)**. ⚠ The **indices carry no fixed semantics across runs**: re-derive them from the variances at every full rebuild. An **incremental** refresh is available (`01b --regime-incremental`): it appends only the new bars from a walk-forward checkpoint, with test-enforced bit-parity. Use: **val stratification + per-regime `val_nll` diagnostics** — **NOT an input feature**. `RegimeMarkovSwitching` (US-macro daily), `RegimeHMM` and `RegimeSession` (Asia/EU/US) remain optional alternatives. Full derivation in [TEORIA.md](TEORIA.md) §4.

### 3.4 Invariante z-score vs raw · z-score vs raw invariant

🇮🇹 ⚠ **Il bug più costoso del progetto.** Il modello predice μ/σ/ν in **spazio z-score** (target scalato dal RobustScaler; `target_scale` = IQR del target raw, persistito in `PipelineState`); il trading layer (`SignalGenerator`, `RiskManager`) opera in **spazio raw**. **Ogni entry-point DEVE chiamare `PipelineState.denormalize_predictions(mu, sigma)` subito dopo il forward**, prima del trading layer (bug 2026-05-23: saltarla → SL/TP macroscopici, Sharpe −256 → +18.7). Con target `log_rv` la sola `denormalize_predictions` è **insufficiente** (mediana log-RV ≈ −7.2): l'inversione completa è `μ·IQR + centro` dal RobustScaler persistito. Derivazione in [TEORIA.md](TEORIA.md) §5.

**EN** ⚠ **The project's costliest bug.** The model predicts μ/σ/ν in **z-score space** (target scaled by the RobustScaler; `target_scale` = raw-target IQR, persisted in `PipelineState`); the trading layer (`SignalGenerator`, `RiskManager`) operates in **raw space**. **Every entry-point MUST call `PipelineState.denormalize_predictions(mu, sigma)` right after the forward**, before the trading layer (bug 2026-05-23: skipping it → macroscopic SL/TP, Sharpe −256 → +18.7). With the `log_rv` target, `denormalize_predictions` alone is **insufficient** (log-RV median ≈ −7.2): the full inversion is `μ·IQR + center` from the persisted RobustScaler. Derivation in [TEORIA.md](TEORIA.md) §5.

---

## 4. Modellazione · Modeling

### 4.1 Architetture · Architectures

🇮🇹 Quattro architetture selezionabili via `--arch`, dietro un forward contract unico:

- **iTransformer** (`QuantiTransformer`) — attention sulle feature (non sul tempo), embedding multi-scala, O(F²). **Arch di produzione della linea vol (5 membri).**
- **N-HiTS** (`QuantNHiTS`) — interpolazione gerarchica pure-MLP, stack pooling multi-scala (8/4/1).
- **TCN+Mamba** (`QuantTCNMamba`) — convoluzioni causali dilatate (campo recettivo 127) + State Space Model con parametri input-dipendenti e fusion gated.
- **LSTM+GRU** (`QuantLSTM`) — dual-stream con attention temporale (legacy, backward compat; sotto-performante).

**Forward contract:** `forward(x, x_macro=None) -> (mu, ls2, lnu)` (`+dir_logits` se multitask; `(quantile_preds, ...)` se `loss_type=quantile`, default). Output path arch-isolati: `models/{arch}/` e `results/{arch}/`.

**EN** Four architectures selectable via `--arch`, behind a single forward contract:

- **iTransformer** (`QuantiTransformer`) — attention over features (not time), multi-scale embedding, O(F²). **Production arch of the vol line (5 members).**
- **N-HiTS** (`QuantNHiTS`) — pure-MLP hierarchical interpolation, multi-scale pooling stacks (8/4/1).
- **TCN+Mamba** (`QuantTCNMamba`) — dilated causal convolutions (receptive field 127) + State Space Model with input-dependent parameters and gated fusion.
- **LSTM+GRU** (`QuantLSTM`) — dual-stream with temporal attention (legacy, backward compat; under-performing).

**Forward contract:** `forward(x, x_macro=None) -> (mu, ls2, lnu)` (`+dir_logits` if multitask; `(quantile_preds, ...)` if `loss_type=quantile`, the default). Arch-isolated output paths: `models/{arch}/` and `results/{arch}/`.

🇮🇹 ⚠ **Stato su disco vs capacità del codice.** L'**ensemble eterogeneo** (combinazione di archi diverse in inferenza via legge della varianza totale — `mu_ens = Σ wᵢ·muᵢ`, `sigma_ens = sqrt(Σ wᵢ·σᵢ² + Σ wᵢ·(muᵢ−mu_ens)²)`, pesi in `DEFAULT_ARCH_WEIGHTS` o dinamici inverse-NLL) è implementato e vivo, ma **la linea vol di produzione gira su solo iTransformer**: `models/nhits` e `models/tcnmamba` sono stati eliminati col cleanup del 2026-06-12 e **vanno riaddestrati prima di qualsiasi run eterogeneo**. **AMP off in inferenza** (evita NaN spectral_norm + Mamba scan). Single source of truth della composizione: `config/default.yaml → distillation.archs`. ⚠ Sul filone direzionale l'errore cross-arch è ≈0.995 → la riduzione di varianza da ensembling è ≈0.

**EN** ⚠ **On-disk state vs code capability.** The **heterogeneous ensemble** (combining different archs at inference via the Law of Total Variance — `mu_ens = Σ wᵢ·muᵢ`, `sigma_ens = sqrt(Σ wᵢ·σᵢ² + Σ wᵢ·(muᵢ−mu_ens)²)`, weights in `DEFAULT_ARCH_WEIGHTS` or dynamic inverse-NLL) is implemented and alive, but **the production vol line runs on iTransformer only**: `models/nhits` and `models/tcnmamba` were removed in the 2026-06-12 cleanup and **must be retrained before any heterogeneous run**. **AMP off at inference** (avoids spectral_norm + Mamba-scan NaN). Composition single source of truth: `config/default.yaml → distillation.archs`. ⚠ On the directional line the cross-arch error is ≈0.995 → the variance reduction from ensembling is ≈0.

### 4.2 Knowledge Distillation multi-teacher · Multi-teacher Knowledge Distillation

🇮🇹 Alternativa all'ensemble omogeneo 5× stessa arch: si addestrano tutte le architetture di `distillation.archs`, si assegna loro uno **score target-aware** (`teacher_score_weights`, single source of truth in `distillation.py`) — per il target direzionale `ret` pesano val_loss, Spearman ρ e directional accuracy; per il target di **volatilità** `log_rv` la directional accuracy pesa **0** (sulla varianza è il segno-vs-mediana, non un segnale tradabile: lo straddle è direction-neutral) — e si distilla ogni modello come **student** su soft labels μ/ls²/lnu ottenute come media pesata di *tutti* i teacher, con transfer delle output head e loss mista scala-normalizzata. Il vantaggio è che l'aggregazione esclude implicitamente gli archi in overfit invece di mediarli alla cieca. Formule dei pesi, temperatura softmax e loss dello student in [TEORIA.md](TEORIA.md) §7; comandi ed esempi in [AVVIO.md](AVVIO.md) §3.5.

**EN** An alternative to the homogeneous 5×-same-arch ensemble: all architectures in `distillation.archs` are trained, given a **target-aware score** (`teacher_score_weights`, single source of truth in `distillation.py`) — for the directional `ret` target val_loss, Spearman ρ and directional accuracy all weigh in; for the **volatility** target `log_rv` directional accuracy weighs **0** (on variance it is sign-vs-median, not a tradable signal: the straddle is direction-neutral) — and each model is then distilled as a **student** on soft labels μ/ls²/lnu formed as a weighted average of *all* teachers, with output-head transfer and a scale-normalized mixed loss. The point is that the aggregation implicitly excludes overfit archs instead of blindly averaging them. Weight formulas, softmax temperature and the student loss in [TEORIA.md](TEORIA.md) §7; commands and examples in [AVVIO.md](AVVIO.md) §3.5.

### 4.3 Loss & output probabilistico · Loss & probabilistic output

🇮🇹 Ogni predizione è una **distribuzione completa**, non una stima puntuale — la σ predetta è ciò che alimenta sia il sizing sia il confronto RV-vs-IV della linea vol. La loss combina quattro termini con ruoli distinti: **t-Student NLL** (code pesanti, il crypto non è gaussiano), **penalità asimmetrica** sugli errori di segno oltre una soglia di magnitudine, **CRPS** (calibrazione della distribuzione, non solo della media) e **Direction-Value joint loss** (accoppia il segno al valore). Con `loss_type=quantile` (default) `model(x)` ritorna `(quantile_preds, dir_logits)`: per un μ scalare usa `model.predict(x)["mu"]` (mediana q2). L'output è in **spazio z-score** e va denormalizzato a monte del trading layer (§3.4). Forme chiuse, gradienti e razionale dei pesi in [TEORIA.md](TEORIA.md) §7; iperparametri in `config/default.yaml → training`.

**EN** Each prediction is a **full distribution**, not a point estimate — the predicted σ is what feeds both position sizing and the vol line's RV-vs-IV comparison. The loss combines four terms with distinct roles: **Student-t NLL** (heavy tails, crypto is not Gaussian), an **asymmetric penalty** on sign errors beyond a magnitude threshold, **CRPS** (calibration of the whole distribution, not just the mean) and a **Direction-Value joint loss** (couples sign to value). With `loss_type=quantile` (default) `model(x)` returns `(quantile_preds, dir_logits)`: for a scalar μ use `model.predict(x)["mu"]` (median q2). Output lives in **z-score space** and must be denormalized upstream of the trading layer (§3.4). Closed forms, gradients and weight rationale in [TEORIA.md](TEORIA.md) §7; hyperparameters in `config/default.yaml → training`.

### 4.4 Simulazione Monte Carlo · Monte Carlo simulation

🇮🇹 2000 scenari × 30 barre con volatilità **GJR-GARCH(1,1)** (`config/default.yaml → montecarlo`). Parametri **ri-stimati su rendimenti orari** il 2026-07-15 (QMLE gaussiano + variance targeting su ~65k barre 2019→2026): `ω=1.026e-6, α=0.1011, γ=0.0052, β=0.8732` → persistence 0.977, half-life 30h, **γ≈0.005: leverage effect quasi nullo a 1h**. Cap σ/barra parametrico (`gjr_sigma_cap: 0.13` a 1h); i valori 1m-era restano in `config/interval/1m.yaml` per il rollback. Il MC **non è sul critical path del backtest** (che usa μ/σ del modello, non il GARCH). Derivazione in [TEORIA.md](TEORIA.md) §8.

**EN** 2000 scenarios × 30 bars with **GJR-GARCH(1,1)** volatility (`config/default.yaml → montecarlo`). Params **re-estimated on hourly returns** on 2026-07-15 (Gaussian QMLE + variance targeting over ~65k bars 2019→2026): `ω=1.026e-6, α=0.1011, γ=0.0052, β=0.8732` → persistence 0.977, 30h half-life, **γ≈0.005: near-zero leverage effect at 1h**. Parametric per-bar σ cap (`gjr_sigma_cap: 0.13` at 1h); the 1m-era values stay in `config/interval/1m.yaml` for rollback. MC is **not on the backtest critical path** (which uses model μ/σ, not GARCH). Derivation in [TEORIA.md](TEORIA.md) §8.

---

## 5. Valutazione · Evaluation

### 5.1 Giudici linea VOL · VOL-line judges

🇮🇹 Metrica primaria: **QLIKE** (loss di volatilità robusta) + ratio NN/HAR-RV; baseline **HAR-RV** per-fold. I modelli vol **non vengono mai tradati nel backtest**: il giudizio è puramente predittivo. Giudici in `scripts/vol/`: `dev_vols_qlike.py` (QLIKE log-RV, giudice principale), `dev_vols_rs_judge.py` (asimmetria semivarianza), `wf_har_baseline.py` (HAR per-fold), `step0_xarch_corr.py` (kill-check correlazione cross-arch), `mfiv_comparator_judge.py` (comparatore MFIV@30h vs IV ATM sul forward test). Split **val-first** via `QUANTSYS_VOLS_SPLIT=val|test`; la logica condivisa (QLIKE, inversione log-RV, HAR) vive in `quantsys/model/vol_metrics.py`. **Risultato chiave:** NN-log_rv 0.257 QLIKE vs HAR-RV 0.368 vs naive 0.807 su test 1h (−30%).

**EN** Primary metric: **QLIKE** (robust volatility loss) + NN/HAR-RV ratio; baseline: per-fold **HAR-RV**. Vol models are **never traded in the backtest**: judging is purely predictive. Judges in `scripts/vol/`: `dev_vols_qlike.py` (log-RV QLIKE, main judge), `dev_vols_rs_judge.py` (semivariance asymmetry), `wf_har_baseline.py` (per-fold HAR), `step0_xarch_corr.py` (cross-arch correlation kill-check), `mfiv_comparator_judge.py` (MFIV@30h vs ATM IV comparator on the forward test). **Val-first** split via `QUANTSYS_VOLS_SPLIT=val|test`; shared logic (QLIKE, log-RV inversion, HAR) lives in `quantsys/model/vol_metrics.py`. **Key result:** NN-log_rv 0.257 QLIKE vs HAR-RV 0.368 vs naive 0.807 on the 1h test split (−30%).

### 5.2 Walk-forward & backtest · Walk-forward & backtest

🇮🇹 **Walk-forward** purged k-fold con embargo anti-leakage (`scripts/02b_walkforward_validate.py`): l'embargo (`embargo_steps=168`, 1 settimana a 1h) è dimensionato ≥ `window_size+horizon` perché finestre e target si sovrappongono nel tempo — senza, il fold di test vede dati già visti in training. Meccanica dei fold in [TEORIA.md](TEORIA.md) §7bis. **Backtest direzionale** (`scripts/03_backtest.py`): fee model + slippage sqrt-impact, stress test (pessimistic e flash-crash), bootstrap CI 5000 iter, analisi per regime, recovery MDD. Le soglie trading in `config/default.yaml → backtest` sono in **spazio RAW** — non sovrascriverle da `arch/*.yaml` senza ricalibrare.

**EN** **Walk-forward** purged k-fold with anti-leakage embargo (`scripts/02b_walkforward_validate.py`): the embargo (`embargo_steps=168`, 1 week at 1h) is sized ≥ `window_size+horizon` because windows and targets overlap in time — without it the test fold sees data already seen in training. Fold mechanics in [TEORIA.md](TEORIA.md) §7bis. **Directional backtest** (`scripts/03_backtest.py`): fee model + sqrt-impact slippage, stress tests (pessimistic and flash-crash), 5000-iter bootstrap CI, regime-conditioned analysis, MDD recovery. Trading thresholds in `config/default.yaml → backtest` live in **RAW space** — do not override them from `arch/*.yaml` without recalibrating.

🇮🇹 ⚠ **Distribution shift val→test — è del TARGET, non della pipeline.** Sul filone **direzionale** le metriche in-sample (val_nll, Spearman/WHR walkforward) **anti-correlano** col backtest: non ottimizzare regole guidate da metriche in-sample. Sul target **`log_rv`** val→test sono invece **coerenti** (verificato dal PASS 2026-06-10) → i gate val-first della linea vol sono informativi. ⚠ Corollario metodologico: un lever va sempre giudicato contro una **baseline riaddestrata sullo stesso dataset/scaler**, mai contro l'incumbent production — confronti cross-scaler producono artefatti (§7).

**EN** ⚠ **val→test distribution shift — it belongs to the TARGET, not the pipeline.** On the **directional** line in-sample metrics (val_nll, Spearman/WHR walkforward) **anti-correlate** with the backtest: do not optimize rules driven by in-sample metrics. On the **`log_rv`** target val→test are instead **coherent** (verified by the 2026-06-10 PASS) → the vol line's val-first gates are informative. ⚠ Methodological corollary: a lever must always be judged against a **baseline retrained on the same dataset/scaler**, never against the production incumbent — cross-scaler comparisons produce artifacts (§7).

### 5.3 Test · Tests

```bash
pytest tests/                          # suite completa / full suite
pytest tests/test_recent_fixes.py -v   # regression sui fix critici (z-score, RevIN, BLOCKER #1)
```

🇮🇹 La suite è centrata sugli **invarianti che, se rotti, non fanno rumore**: no-leakage del `FeatureBuilder`, invarianti dello scaler, contratto `PipelineState`, parity live↔training, bit-parity del regime incrementale, golden della lista-104. Dopo ogni fix con impatto su shape/scaler/feature: aggiungi un regression test e ri-allinea i golden.

**EN** The suite targets the **invariants that fail silently when broken**: `FeatureBuilder` no-leakage, scaler invariants, the `PipelineState` contract, live↔training parity, incremental-regime bit-parity, the 104-list golden. After any fix impacting shape/scaler/features: add a regression test and re-align the golden snapshots.

---

## 6. Deploy & inferenza · Deploy & Inference

```bash
python run_all.py     # menu interattivo / interactive menu
```

🇮🇹 → **Tutti i comandi (pipeline per fase, training per arch, walk-forward, backtest, live, collector, routine di sessione, deploy VPS): [AVVIO.md](AVVIO.md).**

**EN** → **All commands (per-phase pipeline, per-arch training, walk-forward, backtest, live, collectors, session routine, VPS deploy): [AVVIO.md](AVVIO.md).**

### 6.1 Catena d'inferenza direzionale · Directional inference chain

🇮🇹 Catena: forward → `PipelineState.denormalize_predictions(μ, σ)` (z-score → raw) → conviction score (direzione × ampiezza × calibrazione × regime) → **Risk Manager** (Kelly frazionario ∝ edge ∝ 1/varianza, max 1%/trade; SL ATR 3×; trailing; circuit breaker DD 15% MtM intra-trade) → BUY/SELL/HOLD + size + SL + TP. Path live di produzione: `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canoniche, scaler da `PipelineState`) → `LiveEngine._deterministic_predict` (nucleo deterministico condiviso col backtest) → `denormalize_predictions` → `SignalGenerator`. Feed Binance WebSocket con reconnect exponential-backoff, persistenza stato, Volume Profile incrementale, funding refresh thread-safe. ⚠ Questo è il filone **legacy senza alpha OOS**: gira come negative-control, non come strategia.

**EN** Chain: forward → `PipelineState.denormalize_predictions(μ, σ)` (z-score → raw) → conviction score (direction × magnitude × calibration × regime) → **Risk Manager** (fractional Kelly ∝ edge ∝ 1/variance, max 1%/trade; ATR SL 3×; trailing; 15% MtM intra-trade drawdown circuit breaker) → BUY/SELL/HOLD + size + SL + TP. Production live path: `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canonical, scaler from `PipelineState`) → `LiveEngine._deterministic_predict` (deterministic core shared with the backtest) → `denormalize_predictions` → `SignalGenerator`. Binance WebSocket feed with exponential-backoff reconnect, state persistence, incremental Volume Profile, thread-safe funding refresh. ⚠ This is the **legacy line with no OOS alpha**: it runs as a negative control, not as a strategy.

🇮🇹 ⚠ Il path contiene una serie di **guard fail-fast deliberati** (cap su σ raw, validazione `forecast_horizon` e `interval` train-vs-inferenza, allineamento `merge_asof`, floor sullo stop, checkpoint atomici): sono lì per intercettare i bug di denormalizzazione e di contratto train↔inference, **non vanno rimossi**. Elenco puntuale in [TEORIA.md](TEORIA.md) §12.5.

**EN** ⚠ The path carries a set of **deliberate fail-fast guards** (raw-σ cap, `forecast_horizon` and `interval` train-vs-inference validation, `merge_asof` alignment, stop floor, atomic checkpoints): they exist to catch denormalization and train↔inference contract bugs and **must not be removed**. Itemized list in [TEORIA.md](TEORIA.md) §12.5.

### 6.2 Forward test vol & collector 24/7 · Vol forward test & 24/7 collectors

🇮🇹 Il braccio **short-vol** (`scripts/04b_vol_paper.py`) confronta la RV prevista dal modello con la IV implicita Deribit e apre straddle sul **testnet Deribit** quando l'edge supera la soglia pre-registrata, con leg di **delta-hedge** su perp. Gira come servizio **systemd 24/7 su VPS** (il PC di casa è passivo: lanciare `04b` in locale produrrebbe ordini doppi sulla stessa posizione testnet). Baseline di confronto always-long/always-short-vol in `04c_vol_paper_baselines.py`; attribuzione PnL ex-post delta/gamma/theta/vega in `scripts/vol/pnl_attribution.py`. **Gate v1 pre-registrato: FAIL 0/3 (2026-07-18)** — VRP positivo confermato, regola v1 non monetizzante; i gate successivi sono in accumulo di campione (stato e contatori in `STATUS.md`).

**EN** The **short-vol** arm (`scripts/04b_vol_paper.py`) compares model-predicted RV against Deribit implied vol and opens straddles on the **Deribit testnet** when the edge exceeds the pre-registered threshold, with a perp **delta-hedge** leg. It runs as a **24/7 systemd service on a VPS** (the home PC is passive: launching `04b` locally would double up orders on the same testnet position). Always-long/always-short-vol comparison baselines in `04c_vol_paper_baselines.py`; ex-post delta/gamma/theta/vega PnL attribution in `scripts/vol/pnl_attribution.py`. **Pre-registered v1 gate: FAIL 0/3 (2026-07-18)** — positive VRP confirmed, the v1 rule does not monetize it; later gates are accruing sample (state and counters in `STATUS.md`).

🇮🇹 Tre **collector forward** girano in parallelo sullo stesso VPS e producono l'unico dato **non rigenerabile** del progetto: `01c_iv_poller.py` (chain opzioni Deribit + DVOL → `data/iv/`), `01d_orderbook_recorder.py` (order-book L2 Binance, microstruttura → `data/orderbook/`), `01e_trades_recorder.py` (trade opzioni Deribit per gli spread realizzati → `data/deribit_trades/`; la retention API è ~24h, quindi la raccolta è necessariamente forward). Kit di deploy in `deploy/vps/`, sync lato casa in `scripts/vps/`.

**EN** Three **forward collectors** run in parallel on the same VPS and produce the project's only **non-regenerable** data: `01c_iv_poller.py` (Deribit option chain + DVOL → `data/iv/`), `01d_orderbook_recorder.py` (Binance L2 order book, microstructure → `data/orderbook/`), `01e_trades_recorder.py` (Deribit options trades for realized spreads → `data/deribit_trades/`; API retention is ~24h, so collection is necessarily forward). Deploy kit in `deploy/vps/`, home-side sync in `scripts/vps/`.

### 6.3 Dashboard — Deribit Options Risk Terminal

🇮🇹 `scripts/06_dashboard.py` è un terminale di analytics per opzioni crypto: server HTTP single-file + SPA Plotly, **GPU-free e indipendente dalla pipeline ML**, alimentato dai dati **pubblici Deribit** (REST, no-auth). Calcola le Greche in tempo reale sull'intera option chain (Black-Scholes forward-measure, r=0) ed espone quattro viste: Volatility Surface, Option Chain, Risk & Greeks, e **Trades** (storico settled + posizione aperta del forward test `04b`). Avvio: `python run_all.py --only-dashboard` → `http://localhost:8050`. Dettaglio delle viste, endpoint e configurazione: [AVVIO.md](AVVIO.md) §5.4.

**EN** `scripts/06_dashboard.py` is a crypto-options analytics terminal: single-file HTTP server + Plotly SPA, **GPU-free and decoupled from the ML pipeline**, fed by **Deribit public** data (REST, no-auth). It computes Greeks in real time over the whole option chain (Black-Scholes forward measure, r=0) and exposes four views: Volatility Surface, Option Chain, Risk & Greeks, and **Trades** (settled history + open position of the `04b` forward test). Launch: `python run_all.py --only-dashboard` → `http://localhost:8050`. View details, endpoints and configuration: [AVVIO.md](AVVIO.md) §5.4.

---

## 7. Esiti sperimentali · Experimental outcomes

🇮🇹 Ogni esperimento segue un protocollo pre-registrato (gate scritti PRIMA di girare, validazione val-first, lever come flag inerti di default) e **ogni esito negativo viene conservato**: i kill-record sono documentali, il "vaccino contro il re-test involontario". La sintesi di anni di gate è netta: la **linea vol** è l'unico PASS OOS del progetto (`log_rv` batte HAR-RV del 30% in QLIKE, val→test coerenti), mentre il **direzionale non ha alpha OOS a nessun timeframe testato** — a 1m il muro è il costo di transazione, a 1h il costo cade ma non emerge skill; né il gating per regime, né l'entry a soglia o a rango, né la ricalibrazione di σ producono PnL OOS. Prior trasversale che ne deriva: **i momenti pari (varianza, RV) generalizzano OOS, i dispari (segno, asimmetria firmata) no**.

**EN** Every experiment follows a pre-registered protocol (gates written BEFORE running, val-first validation, levers as inert-by-default flags) and **every negative outcome is kept**: kill-records are documental, the "vaccine against involuntary re-testing". The synthesis of years of gates is sharp: the **vol line** is the project's only OOS PASS (`log_rv` beats HAR-RV by 30% in QLIKE, coherent val→test), whereas the **directional line has no OOS alpha at any tested timeframe** — at 1m the wall is transaction cost, at 1h the cost falls away but no skill emerges; neither regime gating, nor threshold/rank entries, nor σ recalibration produce OOS PnL. The cross-cutting prior that follows: **even moments (variance, RV) generalize OOS, odd ones (sign, signed asymmetry) do not**.

🇮🇹 **Dove leggere cosa:** [CHANGELOG.md](CHANGELOG.md) per i milestone in ordine cronologico · [STATUS.md](STATUS.md) per la fonte canonica (periodo corrente + tutti i gate aperti, con i numeri decisionali) · [docs/STATUS_ARCHIVE_2026H1.md](docs/STATUS_ARCHIVE_2026H1.md) per lo storico antecedente al 2026-07-08 (scorporo letterale, read-only) · [TEORIA.md](TEORIA.md) §12 per il protocollo sperimentale, il corpus KILL con i numeri e i flag inerti da non ri-testare.

**EN** **Where to read what:** [CHANGELOG.md](CHANGELOG.md) for milestones in chronological order · [STATUS.md](STATUS.md) for the canonical source (current period + every open gate, with the decisional numbers) · [docs/STATUS_ARCHIVE_2026H1.md](docs/STATUS_ARCHIVE_2026H1.md) for history predating 2026-07-08 (literal split-off, read-only) · [TEORIA.md](TEORIA.md) §12 for the experimental protocol, the KILL corpus with numbers and the inert flags not to be re-tested.

---

## 8. Architettura del sistema · System Architecture

```
Binance REST/WS
      │
      ▼
Candele OHLCV 1h (default: 2019-01-01 → oggi, ~65k barre)
      │
      ▼
Feature Engineering: 104 feature (VWAP, VP short/mid, CVD, microstructure,
                                  funding, tempo, lag, interactions)
      │
      ├─── Macro data (FRED + yFinance) → MacroEncoder 16-dim
      ├─── BTC → realized vol oraria → RegimeMarkovBTC (Markov-Switching, 3 regimi
      │                                data-driven — semantica da ri-derivare per run)
      │
      ▼
Sliding windows 120×104 (contesto 120 barre = 5 giorni a 1h) → dataset normalizzato (RobustScaler)
      │
      ▼
Architettura (selezionabile):
      │
      ├─ itransformer → attention sulle feature, multi-scala   [produzione linea vol, 5 membri]
      ├─ nhits        → pure-MLP gerarchico (stack 8/4/1)      [da riaddestrare]
      ├─ tcnmamba     → TCN dilatate (RF=127) + Mamba SSM      [da riaddestrare]
      ├─ lstm         → LSTM+GRU dual-stream + attention (legacy)
      │
      ├─ [--distill]  Multi-teacher Knowledge Distillation: scoring target-aware →
      │                soft labels pesate (shuffle-safe) → student al 60% epoche
      │
      ▼
Output: μ + σ + ν   in spazio z-score
      │
      ▼
PipelineState.denormalize_predictions(μ, σ)   →   spazio raw
      │
      ├──────────────────────────────► LINEA VOL (produzione)
      │                                 giudizio QLIKE vs HAR-RV  ·  04b: RV_pred vs IV
      │                                 → straddle short-vol su Deribit testnet + delta-hedge
      ▼
LINEA DIREZIONALE (legacy, negative-control)
      │
      ├─ Monte Carlo: 2000 scenari GJR-GARCH(1,1) × 30 barre (off critical path)
      ├─ Conviction score (direzione × ampiezza × calibrazione × regime)
      ├─ Risk Manager (Kelly sizing, ATR stop, trailing, circuit breaker 15% MtM)
      ▼
BUY / SELL / HOLD  +  size  +  stop loss  +  take profit
```

### 8.1 Struttura del progetto · Project Structure

```
quantsys_project/
├── config/
│   ├── default.yaml              parametri condivisi (data, features, model, training, risk, distillation)
│   ├── secrets.yaml.example      template per API keys (copia in secrets.yaml, gitignored)
│   ├── interval/                 override risoluzione candela (1m.yaml legacy · 1h.yaml corrente)
│   ├── arch/                     override per architettura (lstm, itransformer, nhits, tcnmamba, regime-MoE)
│   └── cafn.yaml                 overlay opzionale CAFN (probe, non letto dalla pipeline production)
├── quantsys/                     package Python installabile (pip install -e .)
│   ├── data/                     Binance REST + WebSocket + funding · deribit.py (client pubblico + delivery cache)
│   ├── features/                 FeatureBuilder (104 feature post C-funding, canonical_feature_columns, dual-stream)
│   ├── macro/                    FRED + yFinance · RegimeMarkovBTC + fallback · MacroEncoder / MacroNormalizer
│   ├── model/
│   │   ├── __init__.py           QuantLSTM, QuantiTransformer, QuantTFT
│   │   ├── nhits.py              QuantNHiTS (pure-MLP gerarchico)
│   │   ├── tcn_mamba.py          QuantTCNMamba (TCN + Mamba SSM + gated fusion)
│   │   ├── ensemble.py           EnsembleModel (omogeneo / eterogeneo, AMP off in inferenza)
│   │   ├── distillation.py       Knowledge Distillation multi-teacher (scoring target-aware)
│   │   ├── forecast.py           Monte Carlo GJR-GARCH(1,1) + neural-guided
│   │   ├── vol_metrics.py        QLIKE / inversione log-RV / baseline HAR-RV (linea vol, condivisi)
│   │   ├── vol_forecaster.py     VolForecaster (nucleo forecast del vol-paper, promosso da 04b)
│   │   ├── regime_gate.py        build_regime_gate (gate causale: asof backward + staleness bound)
│   │   ├── cafn.py               CausalAttentionFlowNetwork (coordinatore causale, probe inerte)
│   │   └── revin.py              Reversible Instance Normalization (opzionale, use_revin)
│   ├── trading/                  Kelly sizing, SL dinamico, trailing, circuit breaker
│   │                             + greeks_risk.py (cap vega/delta, CB vega-loss, margin sim — non cablato al live)
│   └── utils/                    config loader, device setup, logging, PipelineState, atomic_save, stats
├── scripts/                      spine numerato (fase) + sottocartelle per linea — mappa: scripts/README.md
│   ├── 00_check_setup.py         verifica CUDA, dipendenze, connessione Binance
│   ├── 01_download_data.py       Binance → 104 feature → dataset npz  ·  01_update_data.py (delta incrementale)
│   ├── 01b_download_macro.py     FRED + yFinance → RegimeMarkovBTC (full / --regime-incremental)
│   ├── 01c/01d/01e_*.py          collector forward 24/7: IV Deribit · order-book L2 Binance · trade opzioni
│   ├── 02_train.py               training con --arch / --distill / ensemble  ·  02b walk-forward  ·  02c optuna  ·  02d CAFN
│   ├── 03_backtest.py            backtest direzionale + stress test + bootstrap CI
│   ├── 04_live_signals.py        feed live WebSocket + paper trading (direzionale, legacy)
│   ├── 04b_vol_paper.py          forward test vol: RV_pred vs IV → straddle testnet Deribit (+ delta-hedge)
│   ├── 04c_vol_paper_baselines.py  baseline always-long / always-short-vol per i gate pre-registrati
│   ├── 05_analyze_signals.py     analisi sessione live  ·  07_verify_teacher.py  confronto architetture
│   ├── 06_dashboard.py           Deribit Options Risk Terminal (HTTP single-file + Plotly)
│   ├── 99_replay_live_vs_training.py   replay diagnostico parity live vs training
│   ├── vol/                      linea vol: giudici QLIKE/RS/MFIV, prep dati, HAR per-fold, kill-check
│   │                             cross-arch, short-vol (backtest storico + arm), IVS, attribuzione PnL
│   ├── research/                 materiale paper / negative-control direzionale
│   ├── vps/                      sync lato casa dei collector VPS (pull scp + merge dedup + heartbeat)
│   └── archive/                  probe chiusi (cross-sectional KILL, σ-recal)
├── deploy/vps/                   kit collector 24/7 (geo-test, setup one-shot, unit systemd)
├── tests/                        suite pytest (feature, NLL, PipelineState, parity, regime, greeks, regression)
├── avvio_sessione.ps1            routine di sessione lato casa (pull VPS + freshness regime + monitoraggio vol)
├── run_all.py                    orchestratore: dati → macro → train → walkfwd → backtest → live → dashboard
├── README.md · AVVIO.md · TEORIA.md · CHANGELOG.md · STATUS.md
├── docs/                         MODEL_IMPROVEMENTS · ROADMAP_VOL_BOOK · STATUS_ARCHIVE_2026H1 · paper/
├── data/                         generato (gitignored) — ⚠ data/iv, data/orderbook, data/deribit_trades NON rigenerabili
├── models/                       checkpoint per architettura (gitignored)
├── results/                      backtest, giudici e segnali live per architettura (gitignored)
└── logs/                         log rotanti (gitignored)
```

🇮🇹 Vedi [AVVIO.md](AVVIO.md) per la guida operativa completa e [TEORIA.md](TEORIA.md) per i fondamenti teorici.

**EN** See [AVVIO.md](AVVIO.md) for the full operational guide and [TEORIA.md](TEORIA.md) for theoretical foundations.

---

## Licenza · License

🇮🇹 [MIT License](LICENSE) — codice di ricerca, **non consulenza finanziaria**.

🇮🇹 **Disclaimer.** Questo repository è un progetto di ricerca personale. Nessuna delle linee esegue ordini con capitale reale: il braccio direzionale è **paper-only** e il braccio short-vol gira su **Deribit testnet** (fondi di carta). Le metriche pubblicate sono out-of-sample dove dichiarato e provengono da gate pre-registrati — inclusi i **fallimenti**, riportati con gli stessi numeri dei successi. Nulla qui è una raccomandazione d'investimento né un'aspettativa di rendimento; il trading di derivati crypto comporta rischio di perdita totale. I dati di mercato appartengono ai rispettivi venue (Binance, Deribit) e non sono ridistribuiti in questo repo.

**EN** [MIT License](LICENSE) — research code, **not financial advice**.

**EN** **Disclaimer.** This repository is a personal research project. No line executes orders with real capital: the directional arm is **paper-only** and the short-vol arm runs on **Deribit testnet** (paper funds). Published metrics are out-of-sample where stated and come from pre-registered gates — including the **failures**, reported with the same numbers as the successes. Nothing here is investment advice or a return expectation; trading crypto derivatives carries a risk of total loss. Market data belongs to the respective venues (Binance, Deribit) and is not redistributed in this repo.
