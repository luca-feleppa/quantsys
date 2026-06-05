# QUANTSYS — Come funziona il sistema

Spiegazione descrittiva del flusso completo, dalla materia prima (candele grezze) al segnale operativo (BUY / SELL / HOLD con size e stop loss).

> Versione inglese in [TEORIA.en.md](TEORIA.en.md).

---

## 1. Raccolta dati di prezzo

Il punto di partenza è Binance: candele OHLCV (Open, High, Low, Close, Volume) BTC/USDT su timeframe 1 minuto. La finestra storica corrente è **dal 2025-05-19 a oggi** (~1 anno, 525.000 candele) — configurata in `config/default.yaml` (`data.start_time` + `data.limit`).

La scelta di un singolo anno recente nasce da test empirici: una storia molto più lunga (es. dal 2021) include cicli di mercato profondamente diversi (bear 2022, recovery 2023, halving 2024) che il modello fatica a riconciliare in un'unica distribuzione, peggiorando la generalizzazione sul presente. Il dataset corrente è bilanciato tra fasi di trend e fasi laterali.

Lo storico è salvato in Parquet (colonnare, compresso). Negli avvii successivi viene scaricato solo il delta dall'ultima candela locale (aggiornamento in secondi).

---

## 2. Log rendimenti

Conversione dei prezzi grezzi in **log rendimenti**: invece del prezzo assoluto, il sistema lavora sulla variazione percentuale tra candele consecutive in scala logaritmica. Vantaggi:
- Stazionari (no trend crescente).
- Simmetrici (un +10% e un −10% hanno lo stesso peso assoluto).

Il **target** è la somma dei log rendimenti delle prossime **30 candele** (`forecast_horizon: 30` in `config/default.yaml`). L'orizzonte è stato portato da 15 a 30 minuti il 2026-05-20: a 15 minuti il rapporto edge/costo era strutturalmente sfavorevole (movimento medio ~25 bps vs costo roundtrip ~26 bps); a 30 minuti il movimento atteso raddoppia (~42 bps) mentre il costo resta costante.

---

## 3. Feature engineering — cosa vede il modello

**104 feature** per candela (**86 dinamiche + 18 strutturali**, verificate sul dataset corrente 2026-06-02), pensate per dare al modello gli strumenti che un trader esperto userebbe. Il conteggio è post-filtro **C-funding** (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`, decisione 2026-05-28): sono state rimosse 15 feature live-incompatibili con ROI ≤ 0 (90d/365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`) — vedi `MODEL_IMPROVEMENTS.md` per il razionale completo.

### Tendenza e momentum
- **Medie mobili** rolling su 5/10/20/60 minuti — trend su orizzonti diversi.
- **Momentum derivato** — rapporti tra medie mobili a scale diverse, ratio momentum/volatilità. RSI/MACD classici rimossi: l'informazione è già catturata dal mix `vol_std` + `lag_ret` + microstructure.

### Volatilità
Rolling std dei log rendimenti su 5/10/20/60 min. ATR classico rimosso dall'input perché ridondante con `vol_std`; resta calcolato dal `RiskManager` per il sizing dinamico degli stop (§10).

### VWAP
Prezzo medio ponderato per il volume. Riferimento usato da fondi e market maker. La distanza prezzo-VWAP indica se il mercato è temporaneamente sopra/sotto-valutato rispetto all'equilibrio di sessione.

### Volume Profile (multi-scala)
Distribuzione del volume per livello di prezzo: i livelli con molto volume diventano supporti/resistenze. Calcolato su **tre finestre** (1h, 4h, 1 giorno). Ogni finestra produce 4 feature: distanza dal POC, dalla Value Area High, dalla Value Area Low, concentrazione del volume al POC. In alta volatilità domina il VP breve; in bassa volatilità il giornaliero è più stabile.

### CVD — Cumulative Volume Delta
Differenza tra volume buy aggressivo (taker) e sell aggressivo. Quando CVD sale mentre il prezzo scende, c'è pressione di acquisto nascosta — e viceversa. Uno degli indicatori più informativi sull'intenzione reale del mercato.

### Microstructure (forma della candela)
10 feature istantanee derivate dalla geometria: body ratio, upper/lower shadow, price velocity, price acceleration. Catturano in tempo reale ciò che MACD/RSI catturano solo dopo molte candele.

### Funding rate
Funding rate dei futures perpetui BTC/USDT scaricato ogni 8h. Genera 3 feature strutturali: `funding_rate`, `funding_rate_1d`, `funding_rate_dev`. Funding alto = long affollati (rischio short squeeze al ribasso); funding negativo = short affollati (rischio squeeze al rialzo).

### Feature temporali
Ora del giorno, giorno della settimana, giorno del mese. I mercati crypto hanno pattern ciclici: liquidità minore di notte, apertura Wall Street / scadenze future generano movimenti sistematici.

### Lag features
Log rendimenti delle ultime 5 candele come feature dirette — aiuta a riconoscere rimbalzi, continuazioni di trend, esaurimenti.

### Feature interactions
3 prodotti espliciti per riconoscimento di regime: `vol_x_pos` (volatilità × posizione VWAP), `momentum_x_funding`, `cvd_x_vol`. Aiutano il modello a riconoscere combinazioni rilevanti senza apprenderle implicitamente.

---

## 4. Dati macro (FRED + yFinance)

Indicatori macroeconomici esterni:
- **DXY** — BTC tende a muoversi inversamente al dollaro.
- **VIX** — la paura sui tradizionali si propaga al crypto.
- **Tassi** (Fed Funds Rate, Treasury) — costo del capitale e propensione al rischio.
- **Oro** — correlato a BTC come asset rifugio.

Frequenza molto più bassa (giornaliera/mensile): processati separatamente e fusi con i dati di prezzo al training.

### Rilevamento dei regimi (Markov-Switching su realized volatility BTC, oraria)
Dal **2026-06-03** il detector di regime opera **direttamente sui dati BTC**, non più sulle macro USA. La classe attiva è `RegimeMarkovBTC` in `quantsys/macro/regime.py` — Markov-Switching (Hamilton 1989) su realized volatility intraday di BTC aggregata oraria. Sostituisce sia il vecchio `RegimeMarkovSwitching` su PC1 delle macro daily (regimi mensili, incompatibili con h=30) sia la baseline transitoria `RegimeSession` (Asia/EU/US deterministico, ~33% per costruzione ma informativamente vuota: cluster temporali, non di mercato).

**Pipeline di feature.** Da `data/raw_candles.parquet` (1-min BTC) il detector aggrega ogni ora:
- `log_ret_h` — somma dei log-return 1m sull'ora (return orario);
- `log_rv` — `log(Σ log_ret_1m²)` clippato a 1e-12 (log della realized variance; la rv grezza è fortemente right-skewed, il log la stabilizza per la MarkovRegression).

**Pipeline statistica.** RobustScaler globale (mediana/IQR, look-ahead trascurabile) → PCA expanding window con `n_pca=1` (combina `log_ret_h` + `log_rv` in un singolo segnale di intensità del moto) → `MarkovRegression` su PC1 con switching mean **+** switching variance → Hamilton filter manuale O(1) per i passi orari fra un retrain e l'altro. Walk-forward: burn-in 30 giorni (720h), retrain ogni 30 giorni (720h).

**Tre regimi che emergono dai dati BTC (run 2026-06-03 su ~9100 ore, post burn-in):**
- **R0 — Quiet** (~42%): bassa volatilità (σ² PC1 ≈ 0.56), drift μ ≈ 0, P(stay)≈89%.
- **R1 — Trending** (~18%): volatilità media (σ² ≈ 0.12), drift positivo μ≈+0.08, P(stay)≈92% (regime persistente di mercato direzionale).
- **R2 — Stress** (~40%): alta volatilità (σ² ≈ 3.79), bias di ribasso μ≈−0.12, P(stay)≈79% (regime di shock / dump).

La frequenza di switch tipica è di **3–8 cambi al giorno**, coerente con un modello a 1-min con orizzonte h=30. La PCA expanding window spiega ~65–73% della varianza fra `log_ret_h` e `log_rv` (PC1 cattura la magnitudine del moto). Le probabilità sono persistite in `data/regime_probs.parquet` su index orario UTC, schema invariato (`regime_dominant`, `regime_burn_in`, `regime_prob_0/1/2`) per drop-in con tutti i consumer (`02_train.py::_load_val_regimes`, dashboard, `RiskManager`).

**Uso del regime nel modello.** Come prima, **non è feature di input** del modello: serve per (a) stratificare il validation split (ora i tre cluster sono di mercato, non di sessione), e (b) come dimensione diagnostica per il `val_nll per regime` nei training log — se uno dei tre regimi mostra NLL sistematicamente peggiore, il modello ha un buco di calibrazione su quella micro-condizione. Le macro USA (FRED + yFinance) restano scollegate dal regime detector ma vengono ancora consumate dal `MacroEncoder` 16-dim come prima.

**Classi legacy in `quantsys/macro/regime.py`** (mantenute come fallback opzionali, non più cablate nella pipeline):
- `RegimeMarkovSwitching` — MS su PC1 delle macro daily FRED+yFinance.
- `RegimeSession` — baseline deterministica Asia/EU/US su `hour_utc // 8`.
- `RegimeHMM` — Gaussian HMM storico (predecessore di MS).

Il `RiskManager` continua ad applicare profili di rischio per regime (§10): ora `regime=2` (Stress) viene letto come "high vol / dump" e il sizing si riduce di conseguenza, mentre `regime=1` (Trending) sostiene esposizione full Kelly.

---

## 5. Dataset per il training

Le feature vengono organizzate in **finestre temporali**: ogni esempio è una matrice `120×104` (ultimi 120 minuti = 2 ore di context, 104 feature). Il target è il log-return cumulativo sulle prossime 30 candele.

Sul dataset corrente (549k candele) con `window_stride: 5` si ottengono **~80.000 esempi train + ~10.000 val + ~10.000 test** (conteggio sul npz 2026-06-04: 80824 / 10103 / 10104).

**Perché T=120 e non di più.** La finestra di 120 minuti è uno sweet spot empirico verificato il 2026-06-04: esperimenti a **T=180 e T=240 hanno prodotto regressioni** (degrado monotono di Spearman walkforward e backtest, μ_pred collassata, sotto-random WHR). Il dataset 1m ~525k non ha profondità informativa per sostenere context più lunghi (over-fitting al noise temporale; il plateau di letteratura 192-384 vale su dataset multi-anno multi-asset, non su BTC singolo 1 anno). **Non aumentare `model.window_size` finché il dataset resta ~525k.**

Normalizzazione con **RobustScaler globale multi-colonna**, meno sensibile a spike di prezzo rispetto allo standard scaler. I parametri sono persistiti in `PipelineState` (`models/{arch}/pipeline_state.pkl`) per riapplicare la stessa trasformazione in inferenza.

### Invariante critico — Spazio z-score vs raw

Il `target_ret` è scalato dal RobustScaler insieme alle altre feature. Il fattore di scala (`target_scale`) è calcolato runtime come IQR del target raw sul training set e persistito in `PipelineState`; varia col dataset e l'orizzonte di forecast (es. ~0.002707 sulla run 2026-06-02 con `data.limit=525k`, `forecast_horizon=30`). Quindi:
- **Il modello predice μ, σ, ν in spazio z-score** (frazione standardizzata). Una σ = 1.0 significa "una IQR del target", non "1% di prezzo".
- **Il trading layer (`SignalGenerator`, `RiskManager`) opera in spazio raw**: soglie `min_expected_ret`, `max_sigma`, calcoli SL/TP assumono frazioni di log-return dirette (`σ × price = distanza in USD`).

Riconciliazione tramite `PipelineState.denormalize_predictions(mu, sigma) -> (mu_raw, sigma_raw)` che moltiplica per `target_scale`. **Sia `03_backtest.py` sia `04_live_signals.py` la applicano subito dopo il forward**, prima di passare le predizioni al `SignalGenerator`. Senza denormalizzazione, SL/TP `σ × price × multiplier` diventano macroscopici (σ_z=1 × $42k × 1.5 = $63k) — bug strutturale identificato e fixato il 2026-05-23 (Sharpe −256 → +18.7).

**Per nuovi entry point**: usare sempre `denormalize_predictions` prima di interpretare μ/σ. Safety net contro regressioni:
- `RuntimeError` in `03_backtest.py` se `σ_max ≥ 0.05` (impossibile in spazio raw su BTC 1m; `raise` invece di `assert` sopravvive a `python -O`).
- Warning runtime in `_sl_tp` se `σ × price × 1.5 > 5% × price`.
- `PipelineState.forecast_horizon` validato in backtest + live: se `cfg.data.forecast_horizon != state.forecast_horizon` → `RuntimeError` (impedisce di usare un modello h=30 con backtest h=15).
- `merge_asof` tra test set e raw_candles validato con `len(merged) == n_test_orig`, altrimenti `RuntimeError` (previene SL/TP su candele sbagliate per gap/halt Binance).
- `update_trailing` aggiorna `portfolio.equity` mark-to-market ad ogni candela (cash + size_usd + unrealized_pnl): il circuit breaker scatta su DD intra-trade anche in live.
- Floor `sl_d = max(sl_d, price × 1e-4)` in `_sl_tp` per evitare SL=TP=entry silenzioso quando ATR=0 (mercato halt).

---

## 6. Architetture disponibili

Quattro architetture selezionabili con `--arch`: `lstm`, `itransformer`, `tcnmamba`, `nhits`. Composizione ensemble eterogeneo configurabile in `config/default.yaml` → `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba; LSTM disponibile ma fuori dall'ensemble per under-performance strutturale).

### LSTM dual-stream (`--arch lstm`, legacy)
Rete ricorrente con le 104 feature divise in due stream:
- **Stream dinamico** (86 feature): log rendimenti, CVD, volume delta, microstructure — momentum di breve termine. Elaborato da una LSTM.
- **Stream strutturale** (18 feature): VWAP, Volume Profile (short + mid), feature temporali, ATH/ATL 30d, momentum_30d, funding rate — contesto di mercato. Elaborato da una GRU.

I due stream fusi e passati attraverso **attention temporale** (pesa diversamente le candele della finestra: alcune sono più informative di altre, es. quella più recente o quella a −15 min). Training ~30-60 min su RTX 2070 Super. Rimosso dall'ensemble il 2026-05-14 dopo val_NLL 5.28 vs iTransformer 0.18.

### iTransformer (`--arch itransformer`)
Transformer **invertito**: invece di attention sui timestep, attention sulle **feature** (ogni feature diventa un "token"). Con 104 feature, complessità O(104²)≈10.800 vs O(120²)=14.400 del Transformer classico — più adatto a dati tabellari perché modella esplicitamente le correlazioni inter-feature.

Embedding **multi-scala**: la finestra da 120 min condensata in 3 viste (1m, 5m, 15m) con average pooling, per catturare strutture rapide e lente senza raddoppiare i parametri.

### TCN+Mamba ibrido (`--arch tcnmamba`)
Due rami in parallelo per pattern locali (5-15 candele) e contesto lungo (120 candele):
- **TCN** (Temporal Convolutional Network): sei blocchi di convoluzione causale con dilatazioni crescenti (1, 2, 4, 8, 16, 32) → campo recettivo **127 candele** (1 + 2·(1+2+4+8+16+32)), copre l'intera finestra di input. Cattura figure tecniche (doppi massimi, breakout, consolidamenti). Output: media globale nel tempo.
- **Mamba** (State Space Model): stato nascosto che evolve con equazioni differenziali discrete a parametri **input-dipendenti** — il modello decide ad ogni passo quanto ricordare. Selezione dinamica dell'informazione su 120 candele senza overhead quadratico dell'attention. Puro PyTorch (no deps esterne). Scan **vettorizzato** via `cumprod` + `cumsum` in chunk di 32 step (AMP disabilitato in inference per evitare NaN su spectral_norm + Mamba edge case, vedi `quantsys/model/ensemble.py`). Speedup forward+backward ~1.8× vs scan sequenziale iniziale.
- **Fusione con gate appreso**: `σ(W·[tcn; mamba])` impara quanto peso dare a locale vs globale per ogni esempio.

Con `d_model=128`, training ~40-70 min su RTX 2070 Super (~2.5 GB VRAM).

### N-HiTS (`--arch nhits`)
**Neural Hierarchical Interpolation for Time Series** (Challu et al. 2022) — implementato il 2026-05-14 come sostituto LSTM.

**Pure-MLP** (no recurrence, no attention, no convoluzione): massima **diversità di inductive bias** vs gli altri 3. Pipeline:
1. **Input projection**: `Linear(104, d_model)`
2. **Tre stack gerarchici** con pooling kernel (8, 4, 1):
   - Stack 1 (k=8): pattern di lungo termine (downsample 8×, MLP, espansione a backcast)
   - Stack 2 (k=4): pattern di medio termine
   - Stack 3 (k=1): pattern di brevissimo termine
3. **Residual decomposition** stile N-BEATS: ogni stack rimuove dal residuo il pattern catturato, lasciando l'informazione non spiegata agli stack successivi
4. **Aggregazione**: somma dei forecast latenti dei 3 stack → output heads

Training molto rapido (~10-15 min su RTX 2070 Super vs 25 min iTransformer).

### Output probabilistico (comune a tutte)
Non un singolo numero, ma **i parametri di una distribuzione**: media μ (direzione), σ (incertezza), ν (parametro di code pesanti — quanto sono probabili movimenti estremi). Il sistema conosce non solo la direzione ma anche la propria confidenza.

Output in **spazio z-score** (target_ret normalizzato dal RobustScaler globale, §5). Denormalizzato esplicitamente con `PipelineState.denormalize_predictions()` prima del trading layer.

---

## 7. Training

### Loss — t-Student NLL
Penalizza il modello quando la distribuzione prevista è lontana dal valore osservato. Distribuzione **t di Student** invece di gaussiana: i rendimenti finanziari hanno code più pesanti (crash e rally violenti sono più frequenti della normale).

### Penalità asimmetrica
Penalità extra quando il modello sbaglia direzione (dice "sale" e scende). Errori di segno costano più degli errori di ampiezza: una posizione nella direzione sbagliata perde, una sottostima dell'ampiezza peggiora solo il rendimento.

### CRPS
Continuous Ranked Probability Score — metrica ausiliaria di **calibrazione**: se il modello dice "80% di probabilità", dovrebbe avere ragione l'80% delle volte. Un modello che dice sempre "95%" ma azzecca il 60% è pericoloso per eccesso di fiducia.

### Validazione walk-forward
No semplice split train/test: il modello viene addestrato su una finestra storica, testato sul periodo immediatamente successivo (mai visto), poi la finestra si sposta in avanti. Simula l'uso reale, evitando look-ahead bias.

### Knowledge Distillation (alternativa all'ensemble omogeneo 5× stessa arch)

**Fase 2a — Training candidati**: architetture in `distillation.archs` (default: iTransformer + N-HiTS + TCNMamba) addestrate normalmente con `n_ensemble=1`. Una sola riga di config cambia la composizione.

**Fase 2b — Multi-Teacher Scoring**: ogni modello valutato alla best epoch con scoring normalizzato (min-max tra arch): **40% val_loss + 35% Spearman + 25% directional accuracy**. Tutti contribuiscono come teacher con pesi softmax (temperature=2) proporzionali allo score — non singolo teacher.

**Fase 2c — Student con transfer + distillation**: ogni modello riadestrato come "student" con tre vantaggi:
- I pesi delle output heads (μ, σ, ν) sono copiati dal best teacher — partenza calibrata invece di casuale.
- Loss mista: **70% NLL reale + 30% distillation**, normalizzata per la varianza di ogni componente teacher (μ~1e-5, ν~5 hanno scale diverse → contributo equo).
- Soft labels pesate da tutti i teacher integrate nel `TensorDataset` (shuffle-safe): ogni batch contiene sia dati reali sia predizioni teacher per gli stessi campioni.

Student convergono in ~60% delle epoche normali. Student già distillati riconosciuti e skippati automaticamente.

**Ensemble eterogeneo (inferenza)**: le N architetture predicono insieme. Errori tendono a non essere correlati perché catturano pattern diversi (N-HiTS gerarchici multi-scala, TCNMamba locali + contesto lungo, iTransformer correlazioni inter-feature). Combinazione = **media pesata** con `DEFAULT_ARCH_WEIGHTS` (`ensemble.py`):
- `mu_ens = Σ w_i · mu_i` (riduce la varianza dell'errore)
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)` (legge della varianza totale: tiene conto sia dell'incertezza media sia del disaccordo tra i modelli)

L'ensemble restituisce direttamente (μ, σ, ν) in spazio naturale, niente conversioni intermedie.

---

## 8. Monte Carlo

Per ogni nuova candela, il sistema genera **2000 scenari di prezzo alternativi** (`mc.n_paths` in config) per i prossimi 30 minuti, usando le predizioni dell'ensemble come guida e aggiungendo variabilità stocastica calibrata sulla volatilità corrente.

La volatilità è stimata con **GJR-GARCH(1,1)** (params di default `omega=1.2e-5, alpha=0.05, gamma=0.065, beta=0.875` in `quantsys/model/forecast.py`): la variante GJR aggiunge un termine di asimmetria che amplifica l'update di volatilità in risposta a shock negativi (leverage effect), tipico dei mercati finanziari.

Risultato: "ventola" di scenari con intervalli di confidenza. Permette di rispondere a domande come:
- Con quale probabilità il prezzo è sopra X$ tra 30 minuti?
- Qual è il peggior scenario nel 5% dei casi?

---

## 9. Generazione del segnale

Il segnale operativo (BUY / SELL / HOLD) combina più elementi:

**Conviction score** — direzione prevista dall'ensemble, ampiezza del movimento atteso, incertezza della previsione. Alta conviction richiede che (a) la direzione sia chiara, (b) il movimento atteso superi le commissioni (0.1% per lato), (c) l'incertezza sia bassa.

**Filtri di qualità**:
- Rendimento atteso > soglia minima (per coprire commissioni)
- Volatilità prevista non troppo alta (mercato caotico → skip)
- Regime BTC compatibile (es. R2 Stress → soglie di ingresso più conservative; R1 Trending → full Kelly)

---

## 10. Gestione del rischio

**Kelly sizing**: size proporzionale all'edge statistico stimato e inversamente proporzionale alla varianza. Segnale forte + mercato calmo → rischio maggiore; segnale debole + alta vola → rischio minore. Rischio massimo per operazione: 1% del capitale.

**Stop loss dinamico (ATR)**: non % fissa ma basato sull'ATR. Mercato volatile → stop più lontano (no stoppato da rumore). Mercato calmo → stop più vicino (limita la perdita).

**Trailing stop**: in profitto, lo stop sale col prezzo proteggendo i guadagni. Distanza proporzionale all'ATR corrente.

**Circuit breaker**: se il drawdown supera il **15%** del capitale (`risk.max_drawdown_stop` in config), il sistema smette di aprire nuove posizioni. Protezione finale contro periodi prolungati di perdite (possibile cambiamento strutturale del mercato non addestrato). DD calcolato **mark-to-market ad ogni candela** (cash + size_usd + unrealized_pnl, aggiornato in `update_trailing`): in live scatta anche se una singola posizione va in forte perdita non realizzata, senza aspettare la chiusura. Recovery automatica quando il DD rientra sotto il 70% della soglia (es. <10.5% su soglia 15%).

---

## 11. Esecuzione live

In modalità live il sistema (`LiveEngine` in `scripts/04_live_signals.py`) si connette a Binance via WebSocket e riceve ogni candela chiusa in tempo reale. Per ogni candela:
1. Aggiorna le feature con la normalizzazione del training.
2. Passa la sequenza degli ultimi 120 minuti al modello.
3. Genera le simulazioni Monte Carlo.
4. Calcola il conviction score.
5. Se il segnale supera i filtri, apre/chiude una posizione (**paper trading**, nessun ordine reale).
6. Aggiorna lo stato del portafoglio e scrive il segnale su disco.

Ogni ora aggiorna in background lo snapshot delle macro per mantenere il contesto aggiornato senza bloccare il feed.

### ✅ Stato attuale: BLOCKER #1 RISOLTO (2026-06-05) — parity live↔training chiusa

Il path di produzione live è ora allineato al training **by-design** (single source of truth `FeatureBuilder`):
`LiveCandleBuffer`(ring 50k OHLCV grezze) → `FeatureAssembler` → `FeatureBuilder.build(fit=False, normalize=True)` (**104 feature canoniche**, stesso ordine, scaler globale dal `PipelineState`) → `LiveEngine._deterministic_predict` (forward deterministico + `denormalize_predictions`) → `SignalGenerator`. L'`EnsembleModel` di produzione non espone `predict_with_uncertainty`, quindi il ramo MC-dropout non scatta in live e il forward è bit-identico al backtest.

**Validazione (gate go/no-go entrambi verdi):** Gate 1 parity FEATURE (`tests/test_live_training_parity.py` + `scripts/99_replay_live_vs_training.py`) → max|Δ|=0; Gate 2 parity SEGNALE → Δμ=Δσ=0, side identico. Il vecchio `LiveFeatureBuffer` (39 feature) è deprecato, resta solo come utility ATR/sanity.

Residuo **operativo** (non di codice): smoke test WS Binance reale + avvio paper-trading. ⚠ I segnali paper ora riflettono il backtest, ma il backtest è negativo OOS (edge a soglia/rank esaurito): il paper-trading serve ad accumulare trade reali, senza aspettativa di Sharpe>0 a priori.

### Robustezza operativa 24/7

Il `LiveEngine` implementa diverse safety net per sistemi sempre attivi:

- **Buffer di lookback dinamico**: dimensionato a `max(window_size + 60, max_rolling_window + 60) = 260` candele — garantisce warmup completo per tutte le feature rolling (es. `price_vs_ma200m` su 200 candele). Pre-2026-05-24 il buffer era 180 e questa feature era silenziosamente sempre zero in live.
- **Separazione candela in formazione vs buffer chiuso**: solo le candele con `k.x == True` (kline chiusa) entrano nel buffer; le parziali stanno in `_pending_candle` separato e vengono scartate al reconnect WS. Previene corruzione del warmup post-disconnessione.
- **Thread safety funding**: il thread daemon che aggiorna il funding ogni 8h scrive `_funding_df` sotto `threading.Lock()`. Primo update eseguito immediatamente all'avvio (no attesa 8h con parquet vecchio).
- **Log rotation tollerante a file lock Windows**: rotazione a 50 MB wrappata in `try/except` per `OSError, PermissionError` — prosegue senza ruotare se il file è temporaneamente lockato.
- **Mismatch forecast_horizon**: `LiveEngine.__init__` solleva `RuntimeError` se `cfg.data.forecast_horizon != PipelineState.forecast_horizon`, impedendo di avviare il live con un modello addestrato per un orizzonte diverso.
- **Checkpoint atomici**: `EarlyStopping` salva i pesi su `.tmp` + `os.replace()` (rename atomico cross-platform), evita checkpoint corrotti se il processo è killato durante un save.

---

## Riepilogo del flusso

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
