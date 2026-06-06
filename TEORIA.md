# QUANTSYS — Come funziona il sistema · QUANTSYS — How the system works

🇮🇹 Spiegazione descrittiva del flusso completo, dalla materia prima (candele grezze) al segnale operativo (BUY / SELL / HOLD con size e stop loss).

**EN** A descriptive walkthrough of the full flow, from raw material (candles) to the operating signal (BUY / SELL / HOLD with size and stop loss).

---

## 1. Raccolta dati di prezzo · 1. Price data collection

🇮🇹 Il punto di partenza è Binance: candele OHLCV (Open, High, Low, Close, Volume) BTC/USDT su timeframe 1 minuto. La finestra storica corrente è **dal 2025-05-19 a oggi** (~1 anno, 525.000 candele) — configurata in `config/default.yaml` (`data.start_time` + `data.limit`).

**EN** Starting point: Binance, OHLCV (Open, High, Low, Close, Volume) candles on BTC/USDT, 1-minute timeframe. The current history window is **from 2025-05-19 to today** (~1 year, 525,000 candles) — configured in `config/default.yaml` (`data.start_time` + `data.limit`).

🇮🇹 La scelta di un singolo anno recente nasce da test empirici: una storia molto più lunga (es. dal 2021) include cicli di mercato profondamente diversi (bear 2022, recovery 2023, halving 2024) che il modello fatica a riconciliare in un'unica distribuzione, peggiorando la generalizzazione sul presente. Il dataset corrente è bilanciato tra fasi di trend e fasi laterali.

**EN** A single recent year was chosen based on empirical tests: a much longer history (e.g. since 2021) covers deeply different market cycles (2022 bear, 2023 recovery, 2024 halving) that the model struggles to reconcile into a single distribution, hurting generalization to the present. The current dataset is balanced between trending and ranging phases.

🇮🇹 Lo storico è salvato in Parquet (colonnare, compresso). Negli avvii successivi viene scaricato solo il delta dall'ultima candela locale (aggiornamento in secondi).

**EN** History is stored in Parquet (columnar, compressed). Subsequent runs download only the delta from the last local candle (a seconds-long update).

---

## 2. Log rendimenti · 2. Log returns

🇮🇹 Conversione dei prezzi grezzi in **log rendimenti**: invece del prezzo assoluto, il sistema lavora sulla variazione percentuale tra candele consecutive in scala logaritmica. Vantaggi:
- Stazionari (no trend crescente).
- Simmetrici (un +10% e un −10% hanno lo stesso peso assoluto).

**EN** Raw prices are converted to **log returns**: instead of absolute price, the system works on the percentage change between consecutive candles in log scale. Advantages:
- Stationary (no rising trend).
- Symmetric (a +10% and a −10% have the same absolute weight).

🇮🇹 Il **target** è la somma dei log rendimenti delle prossime **30 candele** (`forecast_horizon: 30` in `config/default.yaml`). L'orizzonte è stato portato da 15 a 30 minuti il 2026-05-20: a 15 minuti il rapporto edge/costo era strutturalmente sfavorevole (movimento medio ~25 bps vs costo roundtrip ~26 bps); a 30 minuti il movimento atteso raddoppia (~42 bps) mentre il costo resta costante.

**EN** The **target** is the sum of log returns over the next **30 candles** (`forecast_horizon: 30` in `config/default.yaml`). The horizon was bumped from 15 to 30 minutes on 2026-05-20: at 15 minutes the edge/cost ratio was structurally unfavourable (avg move ~25 bps vs ~26 bps roundtrip cost); at 30 minutes the expected move roughly doubles (~42 bps) while cost stays constant.

---

## 3. Feature engineering — cosa vede il modello · 3. Feature engineering — what the model sees

🇮🇹 **104 feature** per candela (**86 dinamiche + 18 strutturali**, verificate sul dataset corrente 2026-06-02), pensate per dare al modello gli strumenti che un trader esperto userebbe. Il conteggio è post-filtro **C-funding** (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`, decisione 2026-05-28): sono state rimosse 15 feature live-incompatibili con ROI ≤ 0 (90d/365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`) — vedi `MODEL_IMPROVEMENTS.md` per il razionale completo.

**EN** **104 features** per candle (**86 dynamic + 18 structural**, verified on the current dataset 2026-06-02), designed to give the model the same tools an experienced trader would use. The count is after the **C-funding** filter (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`, decision 2026-05-28): 15 live-incompatible features with ROI ≤ 0 were removed (90d/365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`) — see `MODEL_IMPROVEMENTS.md` for the full rationale.

### Tendenza e momentum · Trend and momentum

🇮🇹
- **Medie mobili** rolling su 5/10/20/60 minuti — trend su orizzonti diversi.
- **Momentum derivato** — rapporti tra medie mobili a scale diverse, ratio momentum/volatilità. RSI/MACD classici rimossi: l'informazione è già catturata dal mix `vol_std` + `lag_ret` + microstructure.

**EN**
- **Rolling moving averages** over 5/10/20/60 minutes — trend at multiple horizons.
- **Derived momentum** — ratios between MAs at different scales, momentum/volatility ratios. Classic RSI/MACD were removed: the same information is already captured by the `vol_std` + `lag_ret` + microstructure mix.

### Volatilità · Volatility

🇮🇹 Rolling std dei log rendimenti su 5/10/20/60 min. ATR classico rimosso dall'input perché ridondante con `vol_std`; resta calcolato dal `RiskManager` per il sizing dinamico degli stop (§10).

**EN** Rolling std of log returns over 5/10/20/60 min. Classic ATR removed from the inputs (redundant with `vol_std`); still computed by the `RiskManager` for dynamic stop sizing (§10).

### VWAP

🇮🇹 Prezzo medio ponderato per il volume. Riferimento usato da fondi e market maker. La distanza prezzo-VWAP indica se il mercato è temporaneamente sopra/sotto-valutato rispetto all'equilibrio di sessione.

**EN** Volume-weighted average price. The reference used by funds and market makers. Distance price-vs-VWAP tells whether the market is temporarily over/under-valued relative to the session's equilibrium.

### Volume Profile (multi-scala) · Volume Profile (multi-scale)

🇮🇹 Distribuzione del volume per livello di prezzo: i livelli con molto volume diventano supporti/resistenze. Calcolato su **tre finestre** (1h, 4h, 1 giorno). Ogni finestra produce 4 feature: distanza dal POC, dalla Value Area High, dalla Value Area Low, concentrazione del volume al POC. In alta volatilità domina il VP breve; in bassa volatilità il giornaliero è più stabile.

**EN** Volume distribution per price level: high-volume levels become strong supports/resistances. Computed on **three windows** (1h, 4h, 1 day). Each window produces 4 features: distance from POC, from Value Area High, from Value Area Low, volume concentration at the POC. In high volatility the short VP dominates; in low volatility the daily VP is more stable.

### CVD — Cumulative Volume Delta

🇮🇹 Differenza tra volume buy aggressivo (taker) e sell aggressivo. Quando CVD sale mentre il prezzo scende, c'è pressione di acquisto nascosta — e viceversa. Uno degli indicatori più informativi sull'intenzione reale del mercato.

**EN** Difference between aggressive buy volume (taker) and aggressive sell volume. When CVD rises while price falls, there's hidden buy pressure — and vice versa. One of the most informative indicators about the market's real intent.

### Microstructure (forma della candela) · Microstructure (candle shape)

🇮🇹 10 feature istantanee derivate dalla geometria: body ratio, upper/lower shadow, price velocity, price acceleration. Catturano in tempo reale ciò che MACD/RSI catturano solo dopo molte candele.

**EN** 10 instantaneous features derived from candle geometry: body ratio, upper/lower shadow, price velocity, price acceleration. Capture in real time what MACD/RSI capture only after many candles.

### Funding rate

🇮🇹 Funding rate dei futures perpetui BTC/USDT scaricato ogni 8h. Genera 3 feature strutturali: `funding_rate`, `funding_rate_1d`, `funding_rate_dev`. Funding alto = long affollati (rischio short squeeze al ribasso); funding negativo = short affollati (rischio squeeze al rialzo).

**EN** Funding rate of BTC/USDT perpetual futures downloaded every 8h. Produces 3 structural features: `funding_rate`, `funding_rate_1d`, `funding_rate_dev`. High funding = crowded longs (risk of short squeeze down); negative funding = crowded shorts (risk of squeeze up).

### Feature temporali · Temporal features

🇮🇹 Ora del giorno, giorno della settimana, giorno del mese. I mercati crypto hanno pattern ciclici: liquidità minore di notte, apertura Wall Street / scadenze future generano movimenti sistematici.

**EN** Time of day, day of week, day of month. Crypto markets have cyclical patterns: liquidity is lower at night, Wall Street open / futures expiries produce systematic moves.

### Lag features

🇮🇹 Log rendimenti delle ultime 5 candele come feature dirette — aiuta a riconoscere rimbalzi, continuazioni di trend, esaurimenti.

**EN** Log returns of the last 5 candles fed directly as features — helps recognize bounces, trend continuations, exhaustions.

### Feature interactions

🇮🇹 3 prodotti espliciti per riconoscimento di regime: `vol_x_pos` (volatilità × posizione VWAP), `momentum_x_funding`, `cvd_x_vol`. Aiutano il modello a riconoscere combinazioni rilevanti senza apprenderle implicitamente.

**EN** 3 explicit products for regime recognition: `vol_x_pos` (volatility × VWAP position), `momentum_x_funding`, `cvd_x_vol`. Help the model see relevant combinations without learning them implicitly.

---

## 4. Dati macro (FRED + yFinance) · 4. Macro data (FRED + yFinance)

🇮🇹 Indicatori macroeconomici esterni:
- **DXY** — BTC tende a muoversi inversamente al dollaro.
- **VIX** — la paura sui tradizionali si propaga al crypto.
- **Tassi** (Fed Funds Rate, Treasury) — costo del capitale e propensione al rischio.
- **Oro** — correlato a BTC come asset rifugio.

**EN** External macroeconomic indicators:
- **DXY** — BTC tends to move inversely to the dollar.
- **VIX** — fear in traditional markets spills into crypto.
- **Rates** (Fed Funds Rate, Treasury) — cost of capital and risk appetite.
- **Gold** — correlated with BTC as a safe-haven asset.

🇮🇹 Frequenza molto più bassa (giornaliera/mensile): processati separatamente e fusi con i dati di prezzo al training.

**EN** These have much lower frequency (daily/monthly): processed separately and merged with price data at training time.

### Rilevamento dei regimi (Markov-Switching su realized volatility BTC, oraria) · Regime detection (Markov-Switching on BTC realized volatility, hourly)

🇮🇹 Dal **2026-06-03** il detector di regime opera **direttamente sui dati BTC**, non più sulle macro USA. La classe attiva è `RegimeMarkovBTC` in `quantsys/macro/regime.py` — Markov-Switching (Hamilton 1989) su realized volatility intraday di BTC aggregata oraria. Sostituisce sia il vecchio `RegimeMarkovSwitching` su PC1 delle macro daily (regimi mensili, incompatibili con h=30) sia la baseline transitoria `RegimeSession` (Asia/EU/US deterministico, ~33% per costruzione ma informativamente vuota: cluster temporali, non di mercato).

**EN** Since **2026-06-03** the regime detector runs **directly on BTC data**, no longer on US macros. The active class is `RegimeMarkovBTC` in `quantsys/macro/regime.py` — Markov-Switching (Hamilton 1989) on intraday BTC realized volatility aggregated hourly. It supersedes both the legacy `RegimeMarkovSwitching` on PC1 of daily macros (regimes switching every months — incompatible with h=30) and the transitional `RegimeSession` baseline (deterministic Asia/EU/US, ~33% by construction but informationally empty: temporal clusters, not market clusters).

🇮🇹 **Pipeline di feature.** Da `data/raw_candles.parquet` (1-min BTC) il detector aggrega ogni ora:
- `log_ret_h` — somma dei log-return 1m sull'ora (return orario);
- `log_rv` — `log(Σ log_ret_1m²)` clippato a 1e-12 (log della realized variance; la rv grezza è fortemente right-skewed, il log la stabilizza per la MarkovRegression).

**EN** **Feature pipeline.** From `data/raw_candles.parquet` (1-min BTC) the detector aggregates per hour:
- `log_ret_h` — sum of 1m log-returns over the hour (hourly return);
- `log_rv` — `log(Σ log_ret_1m²)` clipped at 1e-12 (log realized variance; raw rv is heavily right-skewed, the log stabilizes it for MarkovRegression).

🇮🇹 **Pipeline statistica.** RobustScaler globale (mediana/IQR, look-ahead trascurabile) → PCA expanding window con `n_pca=1` (combina `log_ret_h` + `log_rv` in un singolo segnale di intensità del moto) → `MarkovRegression` su PC1 con switching mean **+** switching variance → Hamilton filter manuale O(1) per i passi orari fra un retrain e l'altro. Walk-forward: burn-in 30 giorni (720h), retrain ogni 30 giorni (720h).

**EN** **Statistical pipeline.** Global RobustScaler (median/IQR, negligible look-ahead) → expanding-window PCA with `n_pca=1` (collapses `log_ret_h` + `log_rv` into a single motion-intensity signal) → `MarkovRegression` on PC1 with switching mean **+** switching variance → manual Hamilton filter, O(1) per hourly step between retrains. Walk-forward: 30-day burn-in (720h), retrain every 30 days (720h).

🇮🇹 **Tre regimi che emergono dai dati BTC (run 2026-06-03 su ~9100 ore, post burn-in):**
- **R0 — Quiet** (~42%): bassa volatilità (σ² PC1 ≈ 0.56), drift μ ≈ 0, P(stay)≈89%.
- **R1 — Trending** (~18%): volatilità media (σ² ≈ 0.12), drift positivo μ≈+0.08, P(stay)≈92% (regime persistente di mercato direzionale).
- **R2 — Stress** (~40%): alta volatilità (σ² ≈ 3.79), bias di ribasso μ≈−0.12, P(stay)≈79% (regime di shock / dump).

**EN** **Three regimes that emerge from BTC data (2026-06-03 run on ~9100 hours, post burn-in):**
- **R0 — Quiet** (~42%): low volatility (PC1 σ² ≈ 0.56), drift μ ≈ 0, P(stay) ≈ 89%.
- **R1 — Trending** (~18%): mid volatility (σ² ≈ 0.12), positive drift μ ≈ +0.08, P(stay) ≈ 92% (persistent directional market regime).
- **R2 — Stress** (~40%): high volatility (σ² ≈ 3.79), downside bias μ ≈ −0.12, P(stay) ≈ 79% (shock / dump regime).

🇮🇹 La frequenza di switch tipica è di **3–8 cambi al giorno**, coerente con un modello a 1-min con orizzonte h=30. La PCA expanding window spiega ~65–73% della varianza fra `log_ret_h` e `log_rv` (PC1 cattura la magnitudine del moto). Le probabilità sono persistite in `data/regime_probs.parquet` su index orario UTC, schema invariato (`regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`) per drop-in con tutti i consumer (`02_train.py::_load_val_regimes`, dashboard, `RiskManager`).

**EN** Typical switch frequency is **3–8 changes per day**, matching a 1-min model with h=30. The expanding-window PCA explains ~65–73% of the variance between `log_ret_h` and `log_rv` (PC1 captures the magnitude of motion). Probabilities are persisted in `data/regime_probs.parquet` on a UTC hourly index, with the unchanged schema (`regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`) for drop-in compatibility with every consumer (`02_train.py::_load_val_regimes`, dashboard, `RiskManager`).

🇮🇹 **Uso del regime nel modello.** Come prima, **non è feature di input** del modello: serve per (a) stratificare il validation split (ora i tre cluster sono di mercato, non di sessione), e (b) come dimensione diagnostica per il `val_nll per regime` nei training log — se uno dei tre regimi mostra NLL sistematicamente peggiore, il modello ha un buco di calibrazione su quella micro-condizione. Le macro USA (FRED + yFinance) restano scollegate dal regime detector ma vengono ancora consumate dal `MacroEncoder` 16-dim come prima.

**EN** **How the model consumes the regime.** As before, the regime is **not an input feature**: it is used to (a) stratify the validation split (now the three clusters are market clusters, not session clusters), and (b) feed the `val_nll per regime` diagnostic in the training logs — if one of the three regimes shows systematically worse NLL, the model has a calibration gap on that micro-condition. The US macros (FRED + yFinance) are now decoupled from the regime detector but are still consumed by the 16-dim `MacroEncoder` as before.

🇮🇹 **Classi legacy in `quantsys/macro/regime.py`** (mantenute come fallback opzionali, non più cablate nella pipeline):
- `RegimeMarkovSwitching` — MS su PC1 delle macro daily FRED+yFinance.
- `RegimeSession` — baseline deterministica Asia/EU/US su `hour_utc // 8`.
- `RegimeHMM` — Gaussian HMM storico (predecessore di MS).

**EN** **Legacy classes in `quantsys/macro/regime.py`** (kept as optional fallbacks, no longer wired into the pipeline):
- `RegimeMarkovSwitching` — MS on PC1 of daily FRED+yFinance macros.
- `RegimeSession` — deterministic Asia/EU/US baseline on `hour_utc // 8`.
- `RegimeHMM` — legacy Gaussian HMM (MS predecessor).

🇮🇹 Il `RiskManager` continua ad applicare profili di rischio per regime (§10): ora `regime=2` (Stress) viene letto come "high vol / dump" e il sizing si riduce di conseguenza, mentre `regime=1` (Trending) sostiene esposizione full Kelly.

**EN** `RiskManager` keeps applying regime-specific risk profiles (§10): `regime=2` (Stress) is now read as "high vol / dump" and sizing scales down accordingly, while `regime=1` (Trending) sustains full Kelly exposure.

---

## 5. Dataset per il training · 5. Training dataset

🇮🇹 Le feature vengono organizzate in **finestre temporali**: ogni esempio è una matrice `120×104` (ultimi 120 minuti = 2 ore di context, 104 feature). Il target è il log-return cumulativo sulle prossime 30 candele.

**EN** Features are organized into **temporal windows**: every example is a `120×104` matrix (last 120 minutes = 2 hours of context, 104 features). The target is the cumulative log-return over the next 30 candles.

🇮🇹 Sul dataset corrente (549k candele) con `window_stride: 5` si ottengono **~80.000 esempi train + ~10.000 val + ~10.000 test** (conteggio sul npz 2026-06-04: 80824 / 10103 / 10104).

**EN** On the current dataset (525k candles, 1 year) with `window_stride: 5` we get **~80,000 train + ~10,000 val + ~10,000 test examples** (exact npz counts 2026-06-02: 80,390 / 10,049 / 10,049).

🇮🇹 **Perché T=120 e non di più.** La finestra di 120 minuti è uno sweet spot empirico verificato il 2026-06-04: esperimenti a **T=180 e T=240 hanno prodotto regressioni** (degrado monotono di Spearman walkforward e backtest, μ_pred collassata, sotto-random WHR). Il dataset 1m ~525k non ha profondità informativa per sostenere context più lunghi (over-fitting al noise temporale; il plateau di letteratura 192-384 vale su dataset multi-anno multi-asset, non su BTC singolo 1 anno). **Non aumentare `model.window_size` finché il dataset resta ~525k.**

🇮🇹 Normalizzazione con **RobustScaler globale multi-colonna**, meno sensibile a spike di prezzo rispetto allo standard scaler. I parametri sono persistiti in `PipelineState` (`models/{arch}/pipeline_state.pkl`) per riapplicare la stessa trasformazione in inferenza.

**EN** Normalization with a **global multi-column RobustScaler**, less sensitive to price spikes than the standard scaler. Parameters are persisted in `PipelineState` (`models/{arch}/pipeline_state.pkl`) to reapply the same transform at inference time.

### Invariante critico — Spazio z-score vs raw · Critical invariant — z-score vs raw space

🇮🇹 Il `target_ret` è scalato dal RobustScaler insieme alle altre feature. Il fattore di scala (`target_scale`) è calcolato runtime come IQR del target raw sul training set e persistito in `PipelineState`; varia col dataset e l'orizzonte di forecast (es. ~0.002707 sulla run 2026-06-02 con `data.limit=525k`, `forecast_horizon=30`). Quindi:
- **Il modello predice μ, σ, ν in spazio z-score** (frazione standardizzata). Una σ = 1.0 significa "una IQR del target", non "1% di prezzo".
- **Il trading layer (`SignalGenerator`, `RiskManager`) opera in spazio raw**: soglie `min_expected_ret`, `max_sigma`, calcoli SL/TP assumono frazioni di log-return dirette (`σ × price = distanza in USD`).

**EN** The `target_ret` is scaled by the RobustScaler along with the other features. The scale factor (`target_scale`) is computed at runtime as the IQR of the raw target on the training set and persisted in `PipelineState`; it varies with dataset and forecast horizon (e.g. ~0.002707 on the 2026-06-02 run with `data.limit=525k`, `forecast_horizon=30`). Therefore:
- **The model predicts μ, σ, ν in z-score space** (standardized fraction). σ = 1.0 means "one IQR of the target", not "1% of price".
- **The trading layer (`SignalGenerator`, `RiskManager`) operates in raw space**: thresholds `min_expected_ret`, `max_sigma`, SL/TP math assume direct log-return fractions (`σ × price = USD distance`).

🇮🇹 Riconciliazione tramite `PipelineState.denormalize_predictions(mu, sigma) -> (mu_raw, sigma_raw)` che moltiplica per `target_scale`. **Sia `03_backtest.py` sia `04_live_signals.py` la applicano subito dopo il forward**, prima di passare le predizioni al `SignalGenerator`. Senza denormalizzazione, SL/TP `σ × price × multiplier` diventano macroscopici (σ_z=1 × $42k × 1.5 = $63k) — bug strutturale identificato e fixato il 2026-05-23 (Sharpe −256 → +18.7).

**EN** Reconciliation via `PipelineState.denormalize_predictions(mu, sigma) -> (mu_raw, sigma_raw)` which multiplies by `target_scale`. **Both `03_backtest.py` and `04_live_signals.py` apply it right after the forward pass**, before handing predictions to the `SignalGenerator`. Without denormalization, SL/TP `σ × price × multiplier` become macroscopic (σ_z=1 × $42k × 1.5 = $63k) — structural bug identified and fixed on 2026-05-23 (Sharpe −256 → +18.7).

🇮🇹 **Per nuovi entry point**: usare sempre `denormalize_predictions` prima di interpretare μ/σ. Safety net contro regressioni:
- `RuntimeError` in `03_backtest.py` se `σ_max ≥ 0.05` (impossibile in spazio raw su BTC 1m; `raise` invece di `assert` sopravvive a `python -O`).
- Warning runtime in `_sl_tp` se `σ × price × 1.5 > 5% × price`.
- `PipelineState.forecast_horizon` validato in backtest + live: se `cfg.data.forecast_horizon != state.forecast_horizon` → `RuntimeError` (impedisce di usare un modello h=30 con backtest h=15).
- `merge_asof` tra test set e raw_candles validato con `len(merged) == n_test_orig`, altrimenti `RuntimeError` (previene SL/TP su candele sbagliate per gap/halt Binance).
- `update_trailing` aggiorna `portfolio.equity` mark-to-market ad ogni candela (cash + size_usd + unrealized_pnl): il circuit breaker scatta su DD intra-trade anche in live.
- Floor `sl_d = max(sl_d, price × 1e-4)` in `_sl_tp` per evitare SL=TP=entry silenzioso quando ATR=0 (mercato halt).

**EN** **For any new entry point**: always call `denormalize_predictions` before interpreting μ/σ. Safety nets against regressions:
- `RuntimeError` in `03_backtest.py` if `σ_max ≥ 0.05` (impossible in raw space on BTC 1m; `raise` instead of `assert` survives `python -O`).
- Runtime warning in `_sl_tp` if `σ × price × 1.5 > 5% × price`.
- `PipelineState.forecast_horizon` validated in backtest + live: if `cfg.data.forecast_horizon != state.forecast_horizon` → `RuntimeError` (prevents using a h=30 model with a h=15 backtest).
- `merge_asof` between test set and raw_candles validated with `len(merged) == n_test_orig`, otherwise `RuntimeError` (prevents SL/TP triggered on wrong candles due to Binance gaps/halts).
- `update_trailing` updates `portfolio.equity` mark-to-market every candle (cash + size_usd + unrealized_pnl): the circuit breaker fires on intra-trade DD in live too.
- Floor `sl_d = max(sl_d, price × 1e-4)` in `_sl_tp` to avoid silent SL=TP=entry when ATR=0 (market halt).

---

## 6. Architetture disponibili · 6. Available architectures

🇮🇹 Quattro architetture selezionabili con `--arch`: `lstm`, `itransformer`, `tcnmamba`, `nhits`. Composizione ensemble eterogeneo configurabile in `config/default.yaml` → `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba; LSTM disponibile ma fuori dall'ensemble per under-performance strutturale).

**EN** Four architectures selectable via `--arch`: `lstm`, `itransformer`, `tcnmamba`, `nhits`. The heterogeneous ensemble composition is configurable in `config/default.yaml` → `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba; LSTM available but outside the ensemble due to structural under-performance).

### LSTM dual-stream (`--arch lstm`, legacy)

🇮🇹 Rete ricorrente con le 104 feature divise in due stream:
- **Stream dinamico** (86 feature): log rendimenti, CVD, volume delta, microstructure — momentum di breve termine. Elaborato da una LSTM.
- **Stream strutturale** (18 feature): VWAP, Volume Profile (short + mid), feature temporali, ATH/ATL 30d, momentum_30d, funding rate — contesto di mercato. Elaborato da una GRU.

**EN** Recurrent net with the 104 features split into two streams:
- **Dynamic stream** (86 features): log returns, CVD, volume delta, microstructure — short-term momentum. Processed by an LSTM.
- **Structural stream** (18 features): VWAP, Volume Profile (short + mid), temporal features, ATH/ATL 30d, momentum_30d, funding rate — market context. Processed by a GRU.

🇮🇹 I due stream fusi e passati attraverso **attention temporale** (pesa diversamente le candele della finestra: alcune sono più informative di altre, es. quella più recente o quella a −15 min). Training ~30-60 min su RTX 2070 Super. Rimosso dall'ensemble il 2026-05-14 dopo val_NLL 5.28 vs iTransformer 0.18.

**EN** The two streams are fused and passed through **temporal attention** (weighs the window's candles differently: some are more informative, e.g. the most recent one or the one at −15 min). Training ~30–60 min on RTX 2070 Super. Removed from the ensemble on 2026-05-14 after val_NLL 5.28 vs iTransformer 0.18.

### iTransformer (`--arch itransformer`)

🇮🇹 Transformer **invertito**: invece di attention sui timestep, attention sulle **feature** (ogni feature diventa un "token"). Con 104 feature, complessità O(104²)≈10.800 vs O(120²)=14.400 del Transformer classico — più adatto a dati tabellari perché modella esplicitamente le correlazioni inter-feature.

**EN** **Inverted** Transformer: instead of attention on timesteps, attention on **features** (each feature becomes a "token"). With 104 features, complexity O(104²)≈10,800 vs O(120²)=14,400 of the classic Transformer — better suited to tabular data because it explicitly models inter-feature correlations.

🇮🇹 Embedding **multi-scala**: la finestra da 120 min condensata in 3 viste (1m, 5m, 15m) con average pooling, per catturare strutture rapide e lente senza raddoppiare i parametri.

**EN** **Multi-scale** embedding: the 120-min window is compressed into 3 views (1m, 5m, 15m) via average pooling, capturing both fast and slow structures without doubling parameters.

### TCN+Mamba ibrido (`--arch tcnmamba`) · TCN+Mamba hybrid (`--arch tcnmamba`)

🇮🇹 Due rami in parallelo per pattern locali (5-15 candele) e contesto lungo (120 candele):
- **TCN** (Temporal Convolutional Network): sei blocchi di convoluzione causale con dilatazioni crescenti (1, 2, 4, 8, 16, 32) → campo recettivo **127 candele** (1 + 2·(1+2+4+8+16+32)), copre l'intera finestra di input. Cattura figure tecniche (doppi massimi, breakout, consolidamenti). Output: media globale nel tempo.
- **Mamba** (State Space Model): stato nascosto che evolve con equazioni differenziali discrete a parametri **input-dipendenti** — il modello decide ad ogni passo quanto ricordare. Selezione dinamica dell'informazione su 120 candele senza overhead quadratico dell'attention. Puro PyTorch (no deps esterne). Scan **vettorizzato** via `cumprod` + `cumsum` in chunk di 32 step (AMP disabilitato in inference per evitare NaN su spectral_norm + Mamba edge case, vedi `quantsys/model/ensemble.py`). Speedup forward+backward ~1.8× vs scan sequenziale iniziale.
- **Fusione con gate appreso**: `σ(W·[tcn; mamba])` impara quanto peso dare a locale vs globale per ogni esempio.

**EN** Two parallel branches for local patterns (5–15 candles) and long context (120 candles):
- **TCN** (Temporal Convolutional Network): six blocks of causal convolution with growing dilations (1, 2, 4, 8, 16, 32) → receptive field **127 candles** (1 + 2·(1+2+4+8+16+32)), covers the entire input window. Captures technical figures (double tops, breakouts, consolidations). Output: global average over time.
- **Mamba** (State Space Model): hidden state evolving by discrete differential equations with **input-dependent** parameters — the model decides at each step how much to remember. Dynamic information selection over 120 candles without attention's quadratic overhead. Pure PyTorch (no external deps). **Vectorized** scan via `cumprod` + `cumsum` in 32-step chunks (AMP disabled in inference to avoid NaN on spectral_norm + Mamba edge cases, see `quantsys/model/ensemble.py`). Forward+backward speedup ~1.8× vs initial sequential scan.
- **Learned gated fusion**: `σ(W·[tcn; mamba])` learns how much weight to give local vs global per example.

🇮🇹 Con `d_model=128`, training ~40-70 min su RTX 2070 Super (~2.5 GB VRAM).

**EN** With `d_model=128`, training ~40–70 min on RTX 2070 Super (~2.5 GB VRAM).

### N-HiTS (`--arch nhits`)

🇮🇹 **Neural Hierarchical Interpolation for Time Series** (Challu et al. 2022) — implementato il 2026-05-14 come sostituto LSTM.

**EN** **Neural Hierarchical Interpolation for Time Series** (Challu et al. 2022) — implemented on 2026-05-14 to replace LSTM.

🇮🇹 **Pure-MLP** (no recurrence, no attention, no convoluzione): massima **diversità di inductive bias** vs gli altri 3. Pipeline:
1. **Input projection**: `Linear(104, d_model)`
2. **Tre stack gerarchici** con pooling kernel (8, 4, 1):
   - Stack 1 (k=8): pattern di lungo termine (downsample 8×, MLP, espansione a backcast)
   - Stack 2 (k=4): pattern di medio termine
   - Stack 3 (k=1): pattern di brevissimo termine
3. **Residual decomposition** stile N-BEATS: ogni stack rimuove dal residuo il pattern catturato, lasciando l'informazione non spiegata agli stack successivi
4. **Aggregazione**: somma dei forecast latenti dei 3 stack → output heads

**EN** **Pure-MLP** (no recurrence, no attention, no convolution): maximal **inductive-bias diversity** vs the other 3. Pipeline:
1. **Input projection**: `Linear(104, d_model)`
2. **Three hierarchical stacks** with pooling kernel (8, 4, 1):
   - Stack 1 (k=8): long-term patterns (8× downsample, MLP, expansion to backcast)
   - Stack 2 (k=4): mid-term patterns
   - Stack 3 (k=1): very short-term patterns
3. **N-BEATS-style residual decomposition**: each stack removes from the residual the pattern it captured, leaving unexplained information for subsequent stacks
4. **Aggregation**: sum of the 3 stacks' latent forecasts → output heads

🇮🇹 Training molto rapido (~10-15 min su RTX 2070 Super vs 25 min iTransformer).

**EN** Very fast training (~10–15 min on RTX 2070 Super vs 25 min for iTransformer).

### Output probabilistico (comune a tutte) · Probabilistic output (common to all)

🇮🇹 Non un singolo numero, ma **i parametri di una distribuzione**: media μ (direzione), σ (incertezza), ν (parametro di code pesanti — quanto sono probabili movimenti estremi). Il sistema conosce non solo la direzione ma anche la propria confidenza.

**EN** Not a single number, but **the parameters of a distribution**: mean μ (direction), σ (uncertainty), ν (heavy-tails parameter — how likely extreme moves are). The system knows not just the direction but its own confidence.

🇮🇹 Output in **spazio z-score** (target_ret normalizzato dal RobustScaler globale, §5). Denormalizzato esplicitamente con `PipelineState.denormalize_predictions()` prima del trading layer.

**EN** Output in **z-score space** (target_ret normalized by the global RobustScaler, §5). Explicitly denormalized via `PipelineState.denormalize_predictions()` before the trading layer.

---

## 7. Training

### Loss — t-Student NLL

🇮🇹 Penalizza il modello quando la distribuzione prevista è lontana dal valore osservato. Distribuzione **t di Student** invece di gaussiana: i rendimenti finanziari hanno code più pesanti (crash e rally violenti sono più frequenti della normale).

**EN** Penalizes the model when the predicted distribution is far from the observed value. **Student-t** instead of Gaussian: financial returns have heavier tails (crashes and rallies happen more often than a Gaussian would predict).

### Penalità asimmetrica · Asymmetric penalty

🇮🇹 Penalità extra quando il modello sbaglia direzione (dice "sale" e scende). Errori di segno costano più degli errori di ampiezza: una posizione nella direzione sbagliata perde, una sottostima dell'ampiezza peggiora solo il rendimento.

**EN** Extra penalty when the model gets the direction wrong (says "up" but it goes down). Sign errors cost more than magnitude errors: a position in the wrong direction loses money, while underestimating the magnitude only hurts returns.

### CRPS

🇮🇹 Continuous Ranked Probability Score — metrica ausiliaria di **calibrazione**: se il modello dice "80% di probabilità", dovrebbe avere ragione l'80% delle volte. Un modello che dice sempre "95%" ma azzecca il 60% è pericoloso per eccesso di fiducia.

**EN** Continuous Ranked Probability Score — auxiliary **calibration** metric: if the model says "80% probability", it should be right ~80% of the time. A model that always says "95%" but is right 60% is dangerous due to overconfidence in trading.

### Validazione walk-forward · Walk-forward validation

🇮🇹 No semplice split train/test: il modello viene addestrato su una finestra storica, testato sul periodo immediatamente successivo (mai visto), poi la finestra si sposta in avanti. Simula l'uso reale, evitando look-ahead bias.

**EN** No simple train/test split: the model is trained on a historical window, tested on the immediately following period (never seen), then the window slides forward. Simulates the real-world deployment and avoids look-ahead bias.

### Knowledge Distillation (alternativa all'ensemble omogeneo 5× stessa arch) · Knowledge Distillation (alternative to homogeneous ensemble 5× same arch)

🇮🇹 **Fase 2a — Training candidati**: architetture in `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba) addestrate normalmente con `n_ensemble=1`. Una sola riga di config cambia la composizione.

**EN** **Phase 2a — Candidate training**: archs in `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba) trained normally with `n_ensemble=1`. A single config line changes the composition.

🇮🇹 **Fase 2b — Multi-Teacher Scoring**: ogni modello valutato alla best epoch con scoring normalizzato (min-max tra arch): **40% val_loss + 35% Spearman + 25% directional accuracy**. Tutti contribuiscono come teacher con pesi softmax (temperature=2) proporzionali allo score — non singolo teacher.

**EN** **Phase 2b — Multi-Teacher Scoring**: every model evaluated at its best epoch with normalized scoring (min-max across archs): **40% val_loss + 35% Spearman + 25% directional accuracy**. All of them contribute as teachers with softmax weights (temperature=2) proportional to the score — not a single teacher.

🇮🇹 **Fase 2c — Student con transfer + distillation**: ogni modello riadestrato come "student" con tre vantaggi:
- I pesi delle output heads (μ, σ, ν) sono copiati dal best teacher — partenza calibrata invece di casuale.
- Loss mista: **70% NLL reale + 30% distillation**, normalizzata per la varianza di ogni componente teacher (μ~1e-5, ν~5 hanno scale diverse → contributo equo).
- Soft labels pesate da tutti i teacher integrate nel `TensorDataset` (shuffle-safe): ogni batch contiene sia dati reali sia predizioni teacher per gli stessi campioni.

**EN** **Phase 2c — Student with transfer + distillation**: every model is retrained as "student" with three advantages:
- Output-head weights (μ, σ, ν) copied from the best teacher — calibrated start instead of random.
- Mixed loss: **70% real NLL + 30% distillation**, normalized by the variance of each teacher component (μ~1e-5, ν~5 have different scales → equal contribution).
- Soft labels weighted across all teachers, integrated into the `TensorDataset` (shuffle-safe): each batch contains both real data and teacher predictions for the same samples.

🇮🇹 Student convergono in ~60% delle epoche normali. Student già distillati riconosciuti e skippati automaticamente.

**EN** Students converge in ~60% of the normal epochs. Already-distilled students are recognized and skipped automatically.

🇮🇹 **Ensemble eterogeneo (inferenza)**: le N architetture predicono insieme. Errori tendono a non essere correlati perché catturano pattern diversi (N-HiTS gerarchici multi-scala, TCNMamba locali + contesto lungo, iTransformer correlazioni inter-feature). Combinazione = **media pesata** con `DEFAULT_ARCH_WEIGHTS` (`ensemble.py`):
- `mu_ens = Σ w_i · mu_i` (riduce la varianza dell'errore)
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)` (legge della varianza totale: tiene conto sia dell'incertezza media sia del disaccordo tra i modelli)

**EN** **Heterogeneous ensemble (inference)**: the N architectures predict together. Errors tend to be uncorrelated because they capture different patterns (N-HiTS hierarchical multi-scale, TCNMamba local + long context, iTransformer inter-feature correlations). Combination = **weighted mean** with `DEFAULT_ARCH_WEIGHTS` (`ensemble.py`):
- `mu_ens = Σ w_i · mu_i` (reduces error variance)
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)` (law of total variance: accounts both for average model uncertainty and for disagreement between point predictions)

🇮🇹 L'ensemble restituisce direttamente (μ, σ, ν) in spazio naturale, niente conversioni intermedie.

**EN** The ensemble returns (μ, σ, ν) directly in natural space, no intermediate numerical conversions.

---

## 8. Monte Carlo

🇮🇹 Per ogni nuova candela, il sistema genera **2000 scenari di prezzo alternativi** (`mc.n_paths` in config) per i prossimi 30 minuti, usando le predizioni dell'ensemble come guida e aggiungendo variabilità stocastica calibrata sulla volatilità corrente.

**EN** For every new candle, the system generates **2000 alternative price scenarios** (`mc.n_paths` in config) over the next 30 minutes, using the ensemble's predictions as a guide and adding stochastic variability calibrated on current volatility.

🇮🇹 La volatilità è stimata con **GJR-GARCH(1,1)** (params di default `omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875` in `quantsys/model/forecast.py`): la variante GJR aggiunge un termine di asimmetria che amplifica l'update di volatilità in risposta a shock negativi (leverage effect), tipico dei mercati finanziari.

**EN** Volatility is estimated with a **GJR-GARCH(1,1)** model (default params `omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875` in `quantsys/model/forecast.py`): the GJR variant adds an asymmetry term that amplifies the volatility update in response to negative shocks (leverage effect), typical of financial markets.

🇮🇹 Risultato: "ventola" di scenari con intervalli di confidenza. Permette di rispondere a domande come:
- Con quale probabilità il prezzo è sopra X$ tra 30 minuti?
- Qual è il peggior scenario nel 5% dei casi?

**EN** The result is a "fan" of scenarios with confidence intervals. It lets the system answer questions like:
- What's the probability that price is above $X in 30 minutes?
- What's the worst case in the bottom 5%?

---

## 9. Generazione del segnale · 9. Signal generation

🇮🇹 Il segnale operativo (BUY / SELL / HOLD) combina più elementi:

**EN** The operating signal (BUY / SELL / HOLD) combines multiple elements:

🇮🇹 **Conviction score** — direzione prevista dall'ensemble, ampiezza del movimento atteso, incertezza della previsione. Alta conviction richiede che (a) la direzione sia chiara, (b) il movimento atteso superi le commissioni (0.1% per lato), (c) l'incertezza sia bassa.

**EN** **Conviction score** — direction predicted by the ensemble, magnitude of the expected move, prediction uncertainty. High conviction requires (a) clear direction, (b) expected move exceeding commissions (0.1% per side), (c) low uncertainty.

🇮🇹 **Filtri di qualità**:
- Rendimento atteso > soglia minima (per coprire commissioni)
- Volatilità prevista non troppo alta (mercato caotico → skip)
- Regime BTC compatibile (es. R2 Stress → soglie di ingresso più conservative; R1 Trending → full Kelly)

**EN** **Quality filters**:
- Expected return > minimum threshold (to cover commissions)
- Predicted volatility not too high (chaotic market → skip)
- BTC regime compatible (e.g. R2 Stress → tighter entry thresholds; R1 Trending → full Kelly)

---

## 10. Gestione del rischio · 10. Risk management

🇮🇹 **Kelly sizing**: size proporzionale all'edge statistico stimato e inversamente proporzionale alla varianza. Segnale forte + mercato calmo → rischio maggiore; segnale debole + alta vola → rischio minore. Rischio massimo per operazione: 1% del capitale.

**EN** **Kelly sizing**: size proportional to the estimated statistical edge and inversely proportional to variance. Strong signal + calm market → larger risk; weak signal + high vol → smaller risk. Max risk per trade: 1% of capital.

🇮🇹 **Stop loss dinamico (ATR)**: non % fissa ma basato sull'ATR. Mercato volatile → stop più lontano (no stoppato da rumore). Mercato calmo → stop più vicino (limita la perdita).

**EN** **Dynamic ATR stop loss**: not a fixed %, ATR-based. Volatile market → wider stop (no stop-out by noise). Calm market → tighter stop (limits the loss).

🇮🇹 **Trailing stop**: in profitto, lo stop sale col prezzo proteggendo i guadagni. Distanza proporzionale all'ATR corrente.

**EN** **Trailing stop**: once in profit, the stop rises with price protecting gains. Trailing distance also proportional to current ATR.

🇮🇹 **Circuit breaker**: se il drawdown supera il **15%** del capitale (`risk.max_drawdown_stop` in config), il sistema smette di aprire nuove posizioni. Protezione finale contro periodi prolungati di perdite (possibile cambiamento strutturale del mercato non addestrato). DD calcolato **mark-to-market ad ogni candela** (cash + size_usd + unrealized_pnl, aggiornato in `update_trailing`): in live scatta anche se una singola posizione va in forte perdita non realizzata, senza aspettare la chiusura. Recovery automatica quando il DD rientra sotto il 70% della soglia (es. <10.5% su soglia 15%).

**EN** **Circuit breaker**: if drawdown exceeds **15%** of capital (`risk.max_drawdown_stop` in config), the system stops opening new positions. Last-resort protection against prolonged losing streaks (possible structural market change not trained on). DD computed **mark-to-market every candle** (cash + size_usd + unrealized_pnl, updated in `update_trailing`): in live it fires even if a single position has large unrealized losses, without waiting for close. Auto-recovery when DD goes back below 70% of the threshold (e.g. <10.5% with 15% threshold).

---

## 11. Esecuzione live · 11. Live execution

🇮🇹 In modalità live il sistema (`LiveEngine` in `scripts/04_live_signals.py`) si connette a Binance via WebSocket e riceve ogni candela chiusa in tempo reale. Per ogni candela:
1. Aggiorna le feature con la normalizzazione del training.
2. Passa la sequenza degli ultimi 120 minuti al modello.
3. Genera le simulazioni Monte Carlo.
4. Calcola il conviction score.
5. Se il segnale supera i filtri, apre/chiude una posizione (**paper trading**, nessun ordine reale).
6. Aggiorna lo stato del portafoglio e scrive il segnale su disco.

**EN** In live mode the system (`LiveEngine` in `scripts/04_live_signals.py`) connects to Binance via WebSocket and receives every closed candle in real time. For each candle:
1. Updates features with the training-time normalization.
2. Passes the last 120-minute sequence to the model.
3. Generates the Monte Carlo simulations.
4. Computes the conviction score.
5. If the signal passes filters, opens/closes a position (**paper trading**, no real orders).
6. Updates portfolio state and writes the signal to disk.

🇮🇹 Ogni ora aggiorna in background lo snapshot delle macro per mantenere il contesto aggiornato senza bloccare il feed.

**EN** Every hour, the macro variables snapshot is refreshed in the background to keep the macro context up to date without blocking the live feed.

### ✅ Stato attuale: BLOCKER #1 RISOLTO (2026-06-05) — parity live↔training chiusa · ✅ Current status: BLOCKER #1 RESOLVED (2026-06-05) — live↔training parity closed

🇮🇹 Il path di produzione live è ora allineato al training **by-design** (single source of truth `FeatureBuilder`):
`LiveCandleBuffer`(ring 50k OHLCV grezze) → `FeatureAssembler` → `FeatureBuilder.build(fit=False, normalize=True)` (**104 feature canoniche**, stesso ordine, scaler globale dal `PipelineState`) → `LiveEngine._deterministic_predict` (forward deterministico + `denormalize_predictions`) → `SignalGenerator`. L'`EnsembleModel` di produzione non espone `predict_with_uncertainty`, quindi il ramo MC-dropout non scatta in live e il forward è bit-identico al backtest.

**EN** The live production path is now aligned to training **by-design** (single source of truth `FeatureBuilder`):
`LiveCandleBuffer`(50k raw-OHLCV ring) → `FeatureAssembler` → `FeatureBuilder.build(fit=False, normalize=True)` (**104 canonical features**, same order, global scaler from `PipelineState`) → `LiveEngine._deterministic_predict` (deterministic forward + `denormalize_predictions`) → `SignalGenerator`. The production `EnsembleModel` lacks `predict_with_uncertainty`, so the MC-dropout branch never fires live and the forward is bit-identical to the backtest.

🇮🇹 **Validazione (gate go/no-go entrambi verdi):** Gate 1 parity FEATURE (`tests/test_live_training_parity.py` + `scripts/99_replay_live_vs_training.py`) → max|Δ|=0; Gate 2 parity SEGNALE → Δμ=Δσ=0, side identico. Il vecchio `LiveFeatureBuffer` (39 feature) è deprecato, resta solo come utility ATR/sanity.

**EN** **Validation (both go/no-go gates green):** Gate 1 FEATURE parity (`tests/test_live_training_parity.py` + `scripts/99_replay_live_vs_training.py`) → max|Δ|=0; Gate 2 SIGNAL parity → Δμ=Δσ=0, identical side. The old `LiveFeatureBuffer` (39 features) is deprecated, kept only as an ATR/sanity helper.

🇮🇹 Residuo **operativo** (non di codice): smoke test WS Binance reale + avvio paper-trading. ⚠ I segnali paper ora riflettono il backtest, ma il backtest è negativo OOS (edge a soglia/rank esaurito): il paper-trading serve ad accumulare trade reali, senza aspettativa di Sharpe>0 a priori.

**EN** Operational remainder (not code): real Binance WS smoke test + paper-trading start. ⚠ Paper signals now reflect the backtest, but the backtest is negative OOS (threshold/rank edge exhausted): paper-trading is for accumulating real trades, with no a-priori expectation of Sharpe>0.

### Robustezza operativa 24/7 · 24/7 operational robustness

🇮🇹 Il `LiveEngine` implementa diverse safety net per sistemi sempre attivi:

**EN** The `LiveEngine` implements several safety nets for always-on systems:

🇮🇹
- **Buffer di lookback dinamico**: dimensionato a `max(window_size + 60, max_rolling_window + 60) = 260` candele — garantisce warmup completo per tutte le feature rolling (es. `price_vs_ma200m` su 200 candele). Pre-2026-05-24 il buffer era 180 e questa feature era silenziosamente sempre zero in live.
- **Separazione candela in formazione vs buffer chiuso**: solo le candele con `k.x == True` (kline chiusa) entrano nel buffer; le parziali stanno in `_pending_candle` separato e vengono scartate al reconnect WS. Previene corruzione del warmup post-disconnessione.
- **Thread safety funding**: il thread daemon che aggiorna il funding ogni 8h scrive `_funding_df` sotto `threading.Lock()`. Primo update eseguito immediatamente all'avvio (no attesa 8h con parquet vecchio).
- **Log rotation tollerante a file lock Windows**: rotazione a 50 MB wrappata in `try/except` per `OSError, PermissionError` — prosegue senza ruotare se il file è temporaneamente lockato.
- **Mismatch forecast_horizon**: `LiveEngine.__init__` solleva `RuntimeError` se `cfg.data.forecast_horizon != PipelineState.forecast_horizon`, impedendo di avviare il live con un modello addestrato per un orizzonte diverso.
- **Checkpoint atomici**: `EarlyStopping` salva i pesi su `.tmp` + `os.replace()` (rename atomico cross-platform), evita checkpoint corrotti se il processo è killato durante un save.

**EN**
- **Dynamic lookback buffer**: sized to `max(window_size + 60, max_rolling_window + 60) = 260` candles — guarantees full warmup for all rolling features (e.g. `price_vs_ma200m` on 200 candles). Pre-2026-05-24 the buffer was 180 and this feature was silently always zero in live.
- **Forming vs closed candle separation**: only candles with `k.x == True` (closed kline) enter the buffer; partial ones live in a separate `_pending_candle` and are dropped on WS reconnect. Prevents warmup corruption after disconnections.
- **Funding thread safety**: the daemon thread refreshing funding every 8h writes `_funding_df` under `threading.Lock()`. First update executed immediately at startup (no 8h wait on possibly stale parquet).
- **Windows-tolerant log rotation**: rotation at 50 MB wrapped in `try/except` for `OSError, PermissionError` — proceeds without rotating if the file is temporarily locked.
- **forecast_horizon mismatch**: `LiveEngine.__init__` raises `RuntimeError` if `cfg.data.forecast_horizon != PipelineState.forecast_horizon`, preventing live startup with a model trained for a different horizon.
- **Atomic checkpoints**: `EarlyStopping` saves weights to `.tmp` + `os.replace()` (cross-platform atomic rename), avoiding corrupted checkpoints if the process is killed during a save.

---

## Riepilogo del flusso · Flow summary

```
Binance REST/WS
      │
      ▼
Candele OHLCV 1m (storico 2025-05-19 → oggi, ~1 anno, 525k candele)
      │
      ▼
Log rendimenti + 104 feature (VWAP, VP short/mid, CVD, momentum,
                                microstructure, funding, interactions, tempo, lag)
      │
      ├─── Feature macro (FRED + yFinance, 90 feature → MacroEncoder 16-dim)
      ├─── BTC 1m → realized vol oraria → RegimeMarkovBTC (Markov-Switching,
      │                                                    3 regimi data-driven:
      │                                                    Quiet / Trending / Stress)
      │
      ▼
Finestre 120×104 normalizzate (RobustScaler globale) → dataset NPZ
      │
      ▼
Architettura selezionata con --arch (o --distill per ensemble):
      │
      ├─ lstm         → LSTM dual-stream (din. + strutt.) + attention   (legacy)
      ├─ itransformer → attention sulle feature (multi-scala 1m/5m/15m)
      ├─ tcnmamba     → TCN (dil. 1-32, RF=127) + Mamba SSM (contesto 120)
      │                  └─ gated fusion → rappresentazione unificata
      ├─ nhits        → Neural Hierarchical Interpolation (pure-MLP, stack 8/4/1)
      │
      ├─ [--distill] Multi-Teacher Knowledge Distillation:
      │              archs da config/default.yaml → scoring → soft labels
      │              shuffle-safe → student con 60% epoch
      │
      ▼
Output: μ (direzione) + σ (incertezza) + ν (code pesanti)  in z-score
      │
      ▼
PipelineState.denormalize_predictions(μ, σ)  →  spazio raw
      │
      ▼
Monte Carlo 2000 scenari × 30 min (GJR-GARCH per volatilità)
      │
      ▼
Conviction score (direzione × ampiezza × calibrazione × regime)
      │
      ▼
RiskManager (Kelly size, ATR stop, trailing, circuit breaker 15% MtM)
      │
      ▼
Segnale: BUY / SELL / HOLD  +  size  +  stop loss  +  take profit
```
