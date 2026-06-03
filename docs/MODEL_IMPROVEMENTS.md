# QUANTSYS — Miglioramenti modello residui

> English version in [MODEL_IMPROVEMENTS.en.md](MODEL_IMPROVEMENTS.en.md).

Tutto il "già fatto" è stato spostato in `CHANGELOG.md` e nelle note `~/.claude/projects/E--quantsys-project/memory/`. Questo file lista solo ciò che resta da implementare, in ordine raccomandato.

---

## 🔴 NEXT — Diagnostica backtest negativo post-distill 2026-06-03

**Contesto:** il `run_all.py --distill` terminato alle 00:11 del 2026-06-03 ha prodotto backtest preoccupanti:

| Arch | Sharpe | Win Rate | N trades | Return | Equity finale |
|---|---|---|---|---|---|
| TCN+Mamba (teacher) | **-21.05** | 38.1% | 21 | -3.3% | $9,670 |
| iTransformer (student) | **-13.79** | 46.9% | 32 | -3.2% | $9,685 |
| NHits *(file stale 23-05)* | +18.71 | 64.3% | 42 | +3.7% | $10,367 |

**Discordanza diagnosticata:**
- Val (best epoch 2): TCN+Mamba DA **0.541**, Spearman **+0.102**
- Test set: DA **0.516** (-2.5%), Spearman **+0.023** (-77%)
- Test p-value Spearman = 0.022 → segnale debole ma statisticamente significativo
- Backtest → **Sharpe -21** → l'edge in segno si dissolve quando convertito in P&L

**⚠ ARTIFACT AVAILABILITY (verificato 2026-06-03 00:30):** `run_all.py --distill` esegue backtest **SOLO sul teacher selezionato** (`run_all.py:803-810`: `args.arch = selected_teacher` poi `phase_backtest`). Quindi:
- ✅ `results/tcnmamba/dashboard_results.json` (mtime 00:16) = backtest reale del distill
- ❌ `results/itransformer/dashboard_results.json` (mtime 22:55 ieri) = backtest del training **manuale** iTrans pre-distillation, NON del modello distillato di stanotte
- ❌ `results/nhits/dashboard_results.json` (mtime 23 maggio) = pre-fix C-funding, completamente obsoleto

**Per ottenere backtest validi sugli student distillati:**
```powershell
$env:QUANTSYS_ARCH = "itransformer"; python scripts/03_backtest.py
$env:QUANTSYS_ARCH = "nhits";       python scripts/03_backtest.py
```
~1 min ciascuno. Sovrascrive `results/{arch}/dashboard_results.json`. Necessario PRIMA di trarre conclusioni sulla performance distillation.

**Cause ipotizzate (ordinate per probabilità):**
1. **Distribution shift val→test forte**: il test set degli ultimi giorni cattura un regime di mercato diverso da quello del val. Conferma: Spearman crolla -77% val→test.
2. **Edge inferiore alle fee**: WHR 0.508 vs 0.5 random → edge ~0.8% per trade. Fee round-trip 0.2% + slippage stimato 0.1% = 0.3%. Edge netto residuo ~0.5% per trade → margine sottile.
3. **Signal generator non tarato per nuovi modelli 104 feat**: le soglie BUY/SELL/HOLD ereditate dal setup precedente (NHits 119 feat con Sharpe +18.7) producono troppi trade in periodi senza segnale.
4. **Bassa frequenza di trade su test**: 21-32 trade su ~10k sample = 0.2-0.3% di tempo a mercato → ogni trade pesa molto, alto rischio statistico.

### Step diagnostici (ordine raccomandato)

#### Step A — Verifica RegimeSession (priorità alta, ~7 min, indipendente)
Confermare che Opzione C funzioni: nuovo regime detector intraday produce stratificazione val 33/33/33 invece di r0=100%.
```powershell
python run_all.py --arch itransformer --skip-update --skip-macro --force-download
```
Atteso nei log:
- `Stratified val: distribuzione regime: r0≈33%, r1≈33%, r2≈33%`
- `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=>0`

Anche se il backtest sarà brutto (atteso), la verifica del fix Opzione C è separata dalla performance e va chiusa.

#### Step B — Investigazione distribution shift (priorità media, ~30 min)
Caricare i checkpoint backup `models/*/_bak_119feat_20260528/best_model.pt` e ri-eseguire il backtest sullo **stesso test set attuale** (104 feat). Tre scenari possibili:

- **Scenario B1**: vecchi modelli 119 feat → Sharpe negativo anche loro → **distribution shift del mercato**, non regressione di pipeline. Decisione: accettare il regime nuovo, eventualmente retrain con weight più alto sui sample recenti (recency weighting in `02_train.py`).
- **Scenario B2**: vecchi modelli 119 feat → Sharpe positivo → **regressione causata da C-funding** (le 15 feature droppate contenevano segnale che mancava nel C-funding score). Decisione: revisitare la decisione 2026-05-28 di C-funding, considerare C-minimal o sub-set delle 15.
- **Scenario B3**: vecchi modelli con shape mismatch (119 != 104) → load fallisce → riaddestrare un modello a 119 feat per confronto controllato.

#### Step C — Audit signal generator (priorità media, ~20 min)
Cercare in `quantsys/trading/` le soglie BUY/SELL/HOLD e i parametri di sizing. Verificare se sono hardcoded da un setup precedente o si adattano al CI medio del modello. Possibili fix:
- Soglie adattive basate su `σ_pred` (entra solo se `μ_pred / σ_pred > threshold`)
- Filtro min CI lower bound > 0 (entra solo se intervallo non attraversa zero)
- Position sizing inversely proportional a σ_pred

#### Step D — Paper-trading dopo Stage 4 integration (priorità bassa, ~6-48h)
Solo DOPO aver chiuso BLOCKER #1 Stage 4 (integrazione `LiveCandleBuffer` + `FeatureAssembler` nel `LiveEngine`). Far girare il paper-trading per 12-48h, accumulare 50-200 trade, confrontare metriche live con backtest. Se le metriche live divergono dal backtest in modo persistente → bug nel signal generator o nel matching live/training, NON nel modello.

**Punto di ripresa per nuova sessione:**
- Output backtest catastrofico documentato qui (numeri da `results/{arch}/dashboard_results.json`)
- Step A non ancora eseguito (manca verifica `Stratified val: r0≈33%`)
- Step B non ancora eseguito (richiede ricaricare backup `_bak_119feat_20260528`)
- Step C non ancora eseguito (richiede grep su `quantsys/trading/` per signal thresholds)
- Decisione operativa pending: prima fare A (verifica) o B (investigazione)?

---

## 🟡 NEW — Sostituire Markov-Switching macro con regime intraday su BTC (Opzione C)

**Stato:** proposta, non implementata. Origine: 2026-06-02, training iTransformer mostra `Stratified val: distribuzione regime: r0=10056 (100%)` su tutte le validation → il detector attuale è degenere (collassa a 1 cluster) e la diagnostica per-regime (`val_nll spread=0.000`) non porta informazione.

**Problema strutturale (non solo bug del detector):**
- Hamilton 1989 su macro features FRED + yFinance daily → i regimi cambiano ogni **mesi**. Il trading è a 1-min con orizzonte h=30. Mismatch di 4-5 ordini di grandezza tra la scala del regime detector e la scala operativa del modello.
- Verificato in `scripts/02_train.py:577-1096`: il regime label NON è una feature di input al modello, è usato solo per stratificazione val + log diagnostico. Quindi il "bug" attuale è cosmetico, ma anche fixandolo (n_regimes 3→2) il valore aggiunto per trading 1-min resta basso.
- Le 90 macro features grezze + `MacroEncoder` (16-dim) già danno al modello il "regime macro" implicito — il label aggregato è ridondante.

**Decisione Opzione C — regime detector intraday su BTC:**

Sostituire il Markov-Switching su PC1 delle macro con un detector che osservi **direttamente la microstruttura BTC** a una scala coerente col timeframe trading (cambio ogni ~1-4h, non mesi). Tre varianti candidate, ordinate per costo crescente:

1. **Session regime (più semplice)**: lookup su `hour` UTC → {Asia 00-08, EU 08-16, US 16-24}. Tre regimi, deterministico, costo zero, ground truth della letteratura sui crypto (Asia low-vol, EU/US high-vol).
2. **Volatility regime via threshold**: percentile rolling 4h della realized volatility → {low / mid / high}. Cambia 5-10 volte al giorno, match perfetto con h=30. Implementazione semplice (no EM, no PCA).
3. **HMM/Markov-Switching su BTC**: stesso engine attuale ma osservato su realized volatility intraday (rolling 1h log_ret²) anziché su PC1 macro. Cambia 3-8 volte/giorno. Riusa l'infrastruttura `RegimeMarkovSwitching` esistente, cambia solo la feature di input.

**Razionale per la scelta finale:** partire dalla 1 (session) come baseline, misurare lo spread NLL per-regime sul val. Se spread > 0.05 NLL → il regime è informativo, vale la pena passare alla 2/3. Se ancora 0.000 → il modello non discrimina tra regimi (segnale uniforme), e tanto vale rimuovere il regime detector del tutto.

**Vantaggi vs attuale:**
- Frequenza di switch coerente col timeframe h=30
- Diagnostica `val_nll per regime` torna informativa
- Stratificazione val effettiva (non più degenere r0=100%)
- Possibile features future: regime label come input al modello (oggi NON usato come feature)

**File da toccare (implementazione baseline session):**
- `quantsys/macro/regime.py` → nuova classe `RegimeIntraday` o variante `RegimeSession` (`session = floor(hour_utc / 8)`)
- `scripts/01b_download_macro.py` → fittare/serializzare il nuovo detector (probabilmente trivial, no EM)
- `scripts/02_train.py:385,577` → caricare il nuovo regime per `_load_val_regimes` e stratificazione

**Validazione:**
- Dopo retrain, verificare `Stratified val: distribuzione regime` ha tutti i regimi con coverage ~25-40% ciascuno (non più 100% in r0)
- Spread `val_nll per regime` > 0 (segnale: il modello fa peggio in alcuni regimi)
- Backtest invariato o migliorato (regime detector ≠ regressione)

> ⚠ **Dopo implementazione, aggiornare `AVVIO.md`, `TEORIA.md` (§ "Markov-Switching"), `README.md` (e versioni `.en.md`)** con la nuova architettura del regime detector. La sezione attuale in `TEORIA.md` (`statsmodels.MarkovRegression` su PC1 macro) andrà sostituita con la descrizione del detector intraday scelto.

### 🚧 Implementation status (live tracker — aggiornato dalla sessione)

**Approccio scelto:** Variante 1 — **regime session-based** (Asia/EU/US via `hour_utc // 8`). Baseline deterministica, costo zero, nessuna EM. Se dopo il prossimo training lo spread NLL per-regime resta ~0, si valuta variante 2 (volatility threshold) o si rimuove del tutto il detector.

**Sessione 2026-06-02 22:35 — code + docs completati via fan-out 3 subagent:**

| Task | File | Stato | Note |
|---|---|---|---|
| Nuova classe `RegimeSession` | `quantsys/macro/regime.py:848-1003` | ✅ done | Aggiunta come "STADIO 1c", drop-in con `fit_predict_walkforward` / `save` / `load`. Smoke test 9097 righe, distribuzione 3033/3032/3032 (~33% ciascuno). `RegimeMarkovSwitching` e `RegimeHMM` intoccati come fallback |
| Switch in pipeline | `scripts/01b_download_macro.py:28,89-115,208` | ✅ done | Import + uso `RegimeSession(n_regimes=3)`. Filename output `regime_hmm.pkl` e `regime_probs.parquet` invariati (backward compat consumer `_load_val_regimes`) |
| TEORIA.md (IT) | §4 righe 76-83, §9 riga 214, diagramma ~277 | ✅ done | Sezione riscritta da capo, diagramma ASCII separa macro path da regime path |
| README.it.md (IT) | bullet riga 36, diagramma 92, tree 144/160 | ✅ done | Bullet "Rilevamento regimi" → session-based + nota fallback |
| TEORIA.en.md (EN) | §"Macro regime detection" righe 76-83, diagramma ~277 | ✅ done | Mirror IT — nuova descrizione UTC-session detector |
| README.md (EN) | bullet riga 36, diagramma 92, tree 144/160 | ✅ done | Mirror README.it.md |
| AVVIO.md (IT) | nessuna sezione regime descrittiva | ⊘ skip | File è quickstart launch — non descrive regime detector, niente da aggiornare |
| AVVIO.en.md (EN) | nessuna sezione regime descrittiva | ⊘ skip | Stesso motivo di AVVIO.md |
| Test | `tests/test_features.py:212-226` | ⊘ skip | Test esistente di `RegimeMarkovSwitching` resta valido (classe non rimossa); test per `RegimeSession` opzionale, non bloccante |
| Smoke test pipeline reale | `python scripts/01b_download_macro.py` | ✅ done 2026-06-02 22:54 | `data/regime_probs.parquet` rigenerato: 73777 righe orarie 2018-01-01→2026-06-02, distribuzione **33.3% / 33.3% / 33.3%** (24593/24592/24592). Tempo ~129s |
| Verifica end-to-end Phase 1+1b+2+3+4+5 | fan-out 3 subagent | ✅ done 2026-06-02 22:55 | Tutti gli script lanciati da `run_all.py --distill` verificati: sintassi+import+smoke OK. IC fix sanity check: `ic_mean=0.3728 ≈ spearman=0.3726` su segnale skill 30% (matematicamente coerente) |
| Retrain iTransformer di verifica | `run_all.py --arch itransformer --skip-update --skip-macro --force-download` | ⏳ **pending** | verificare `Stratified val: r0=X r1=Y r2=Z` distribuito + spread NLL > 0 |

**Punti di ripresa per la prossima sessione (in caso di out-of-tokens):**
1. **Step successivo:** lanciare `python scripts/01b_download_macro.py` per rigenerare `data/regime_probs.parquet` (1-2 minuti). NB: scarica anche dati FRED/yFinance — se quelli sono già aggiornati, è OK rifare il download (idempotente).
2. **Dopo step 1:** rilanciare training iTransformer per verificare. Comando: `python run_all.py --arch itransformer --skip-update --skip-macro --force-download` (~7 min). Cercare la riga `Stratified val: distribuzione regime` nei log — atteso ~33%/33%/33%, NON più `r0=100%`.
3. **Validazione finale:** cercare la riga `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=Z` ogni 5 epoche. Se `spread > 0.05`, il regime è informativo e vale la pena tenerlo. Se ancora `spread ≈ 0`, considerare variante 2 (volatility threshold) o rimozione totale.
4. **Test unitario opzionale:** aggiungere `test_regime_session.py` in `tests/` con verifica determinismo + distribuzione bilanciata (non bloccante per merge).

**Decisione di rollback:** se la nuova `RegimeSession` non migliora lo spread NLL per-regime entro 1 training completo, considerare variante 2 (volatility threshold percentile rolling 4h). La classe `RegimeMarkovSwitching` resta nel codice come fallback (non rimuovere).

---

## 🔴 BLOCKER #1 — Allineamento feature live↔training (Stage 2-5)

**Stato:** Stage 1 done (codice), Stage 2-5 pending. Il paper-trading **non** è eseguibile finché il mismatch non è risolto.

**Problema (verificato 2026-06-02 con `scripts/99_replay_live_vs_training.py`):** il backtest usa il `FeatureBuilder` filtrato C-funding (**104 feature** post Stage 2); il live engine (`LiveFeatureBuffer._compute_features` in `scripts/04_live_signals.py`) ne costruisce a mano **solo 39** in ordine diverso, con normalizzazione median/IQR per-window (non il `RobustScaler` del `pipeline_state`), e `_predict` fa pad/truncate posizionale cieco. Tre disallineamenti sovrapposti (conteggio + ordine + scala) → input live di fatto scorrelati dal training. **I segnali del paper-trading attuale NON riflettono il backtest.**

Causa di fondo: il `LiveFeatureBuffer` ridotto esiste perché il `FeatureBuilder` completo richiede storia lunga (ATH/ATL 365d, momentum 90d, frac-diff, vp_*_long) non disponibile nel buffer rolling live (260 candele).

### Decisione (2026-05-28): Opzione C-funding (~104 feature)

**Razionale dall'esperimento di permutation importance** (ensemble eterogeneo, 2500 finestre val, permutazione per gruppo/feature): le 23 feature "live-hostile" (lookback > buffer: 30/90/365d, frac_diff_*, vp_*_long, vp_poc_convergence, funding) hanno **ROI ≤ 0** per il modello a h=30: permutarle in blocco *migliora* leggermente le metriche (DA 0.529→0.532, Spearman 0.069→0.076). Unica eccezione: le feature **funding** (`momentum_x_funding` +0.0042, `funding_rate_dev` +0.0028).

**Set C-funding** = single source of truth condiviso training/live:
- Droppa 15 feature live-incompatibili (90d, 365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`).
- Mantiene 30d + funding (ROI positivo, calcolabili in live via ring 30d ~170 KB e poll funding Binance).
- Risultato atteso: ~104 feature totali (vs 119 attuali).

> Lo schema "ibrido completo" che manteneva tutte le 30/90/365d in live era documentato come alternativa ma **non raccomandato dai dati** (ROI negativo del tier long) — definitivamente scartato.

### Stage 1 — codice ✅ DONE

`LIVE_DROP_FEATURES` (15 feature) in `quantsys/features/__init__.py`, filtrato in `scripts/01_download_data.py` (`feat_cols = [c for c in feat_cols if c not in LIVE_DROP_FEATURES]`).

### Stage 2 — Rigenerazione dataset a 104 feat ✅ DONE 2026-06-02

Eseguita in automatico dentro `run_all.py --distill`: il dataset è stato rigenerato a `(80390, 120, 104)` train + `(10049, 120, 104)` val + `(10049, 120, 104)` test, con il filtro C-funding correttamente applicato (15 feature droppate, verificato programmaticamente).

### Stage 3 — Retrain distill completo ✅ DONE 2026-06-02

Eseguito nello stesso `run_all.py --distill` del 2026-06-02: tutti e 3 i modelli (iTransformer, N-HiTS, TCN+Mamba) riaddestrati da zero a 104 feature. Distillation multi-teacher applicata agli student selezionati dallo scoring automatico (vedi `models/{arch}/config.json` per i flag `distilled: true, teacher_arch: "multi-teacher"`).

> Metriche di backtest dei modelli a 104 feat: da rileggere in `results/{arch}/dashboard_results.json` dopo la conclusione del run (potrebbero essere diverse dal +18.71 Sharpe del setup a 119 feat).

### Stage 4 — Riscrittura live engine 🚧 IN CORSO (sessione 2026-06-02 23:10)

**Decisione architetturale:** invece di duplicare la logica feature engineering in `LiveFeatureBuffer`, **riusare direttamente `quantsys/features.FeatureBuilder.build()`** sul buffer live. Single source of truth automatica → parity test garantito by-design.

**Razionale:**
- Il delta feature live↔training è ~65 feature (live ha 39, training ha 104). Riscrivere a mano queste 65 a parità con `FeatureBuilder` è alto rischio di drift silenzioso.
- `FeatureBuilder.build()` su 43200 righe × ~120 colonne richiede ~1-3s su CPU. Eseguito al close di ogni candela (60s budget) è ampiamente nel budget.
- Memoria: 43200 candele × 104 float32 = 18 MB. Trascurabile.
- Le feature 30d (dist_ath_30d, momentum_30d, price_vs_ma200m) richiedono 43200 candele di storia → buffer "warm" deve essere bootstrapped dal parquet storico al boot.

**Architettura nuovo live engine:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ LiveCandleBuffer (50,000 candele OHLCV grezze, ring buffer)         │
│  ├─ bootstrap: legge raw_candles.parquet[-50000:] al boot           │
│  └─ append(candle): push new, pop old (FIFO maxlen)                  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FundingRatePoller (deque(maxlen=30d ÷ 8h = 90), poll ogni 1h)       │
│  └─ usa quantsys.data.fetch_funding_rate                            │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FeatureAssembler (chiamato al close di ogni candela)                │
│  1. df = pd.DataFrame(LiveCandleBuffer.tail(43200))                  │
│  2. df_with_funding = merge_asof(df, funding_history)                │
│  3. feat_df = FeatureBuilder.build(df, fit=False, normalize=True)    │
│  4. Verifica feat_df.columns == PipelineState.feature_names (HARD-FAIL)│
│  5. Filtra LIVE_DROP_FEATURES (già in build)                         │
│  6. Estrai window[-120:, :] → (120, 104) np.ndarray                  │
└──────────────────────────────────────────────────────────────────────┘
```

**File da toccare:**
- `quantsys/features/__init__.py` → esporre `get_canonical_feature_names(npz_path)` come single source of truth
- `quantsys/utils/__init__.py` `PipelineState` → aggiungere `feature_names: list[str]` attribute (persistito in pickle)
- `scripts/04_live_signals.py` → sostituire `LiveFeatureBuffer` con `LiveCandleBuffer` + `FeatureAssembler` + integrazione `FundingRatePoller`; rimuovere `_pad_or_truncate` da `_predict`
- `tests/test_live_training_parity.py` → nuovo: parity test (live output == FeatureBuilder su finestra storica con tolleranza 1e-6)
- `scripts/99_replay_live_vs_training.py` → aggiornare per usare nuovo engine

### 🚧 Stage 4 implementation tracker (live — aggiornato a ogni milestone)

**Sessione attiva:** 2026-06-02 23:10 (parallela al distill in corso, GPU non interferita perché live engine è solo CPU)

| Step | File | Stato | Note di ripresa |
|---|---|---|---|
| 4.1 — Esporre `feature_names` canonico | `quantsys/features/__init__.py:13-58` | ✅ done 2026-06-02 23:25 | `get_canonical_feature_names(npz_path)` aggiunta. lru_cache(maxsize=4). Hard-fail su FileNotFoundError/KeyError. Smoke test: 104 nomi corretti, cache attiva (2a call <0.001ms), zero overlap con LIVE_DROP_FEATURES. |
| 4.2 — `PipelineState.feature_names` | `quantsys/utils/__init__.py:152` | ⊘ skip | Verificato 2026-06-02 23:20: `PipelineState.feature_cols` ha 121 elementi (pre-filter, include `LIVE_DROP_FEATURES` + `target_ret`/`target_dir`), `scale_cols` ha 105 (104+target_ret). Non è la single source of truth canonica. Non aggiungere attributo nuovo — il NPZ resta autoritativo, PipelineState fornisce `scaler` + `clip_lo_/hi_` + `n_dynamic_features` + macro state. |
| 4.3 — `LiveCandleBuffer` | `scripts/04_live_signals.py:97-185` | ✅ done 2026-06-02 23:18 | Smoke test: bootstrap 50000 candele da parquet, to_dataframe(120) shape (120,9), append FIFO funziona, default fields=0 per campi mancanti. Tz-naive UTC come richiesto da FeatureBuilder. |
| 4.4 — `FundingRatePoller` | `scripts/04_live_signals.py` (nuova classe) | ⏳ pending | Thread/asyncio task che chiama `quantsys.data.fetch_funding_rate` ogni 1h. Deque(maxlen=90) per 30d × 3 rates/giorno. Esponi `to_dataframe()` per merge. **Workaround temporaneo**: leggere `data/funding_rate.parquet` da disk al boot (già funziona, vedi test 4.5). Poller serve solo per refresh real-time. |
| 4.5 — `FeatureAssembler` | `scripts/04_live_signals.py:188-308` | ✅ done 2026-06-02 23:22 | **End-to-end OK**: compute_window(120) produce (120, 104) float32, no NaN, no Inf. Hard-fail su feature mancanti. Tz-normalization di funding_df dentro compute_window. Stats output: mean=-0.05, std=1.73, range [-24.8, +25.8] (consistente con RobustScaler+clip del training). |
| 4.6 — Sostituire in `LiveEngine.__init__` | `scripts/04_live_signals.py` | ⏳ pending | Rimuovere `self.buffer = LiveFeatureBuffer(...)`, sostituire con i 3 nuovi componenti. Aggiornare loop principale. **NON cancellare** la classe `LiveFeatureBuffer` vecchia (commento "DEPRECATED — legacy 39-feature implementation"). |
| 4.7 — Rimuovere pad/truncate | `scripts/04_live_signals.py:982-988` | ⏳ pending | In `_predict()`, sostituire il blocco `if n_live < n_model: pad ...` con `assert window.shape[1] == n_model, "feature mismatch"`. Il window arriva già a 104. |
| 4.8 — Parity test | `tests/test_live_training_parity.py` | ✅ done 2026-06-02 23:30 | **4/4 test passano in 12.8s**. (1) Assembler output == direct FeatureBuilder output con `max abs diff < 1e-5`, (2) canonical order stabile, (3) zero overlap con LIVE_DROP_FEATURES, (4) hard-fail su funding=None. Parity matematicamente verificata. |
| 4.9 — Aggiornare replay script | `scripts/99_replay_live_vs_training.py` | ✅ done 2026-06-02 23:33 | Riscritto da zero per usare `LiveCandleBuffer` + `FeatureAssembler`. Run output: **Max diff: 0.000e+00** (parity perfetta bit-identica). Era 3 disallineamenti, ora 0. |
| 4.10 — Smoke test live | `python scripts/04_live_signals.py` | ⏳ pending | Connessione WS, warmup completo, primo segnale entro 2 min. Verificare niente warning "feature mismatch". |
| 4.11 — Update doc `MODEL_IMPROVEMENTS.md` | questo file | ⏳ pending | Spostare Stage 4 da pending a done, aggiornare `BLOCKER #1` header con `🟢 RESOLVED` se anche Stage 5 chiuso |

**Punti di ripresa per session reset:**
- **Se sei a step 4.1-4.2**: lavoro non distruttivo, può riprendere da qualsiasi punto
- **Se sei a 4.3-4.7**: il `scripts/04_live_signals.py` è in stato intermedio — la vecchia classe `LiveFeatureBuffer` va lasciata in piedi finché tutti i nuovi componenti sono testati. Prima di committare, verificare che lo script sia almeno importabile (`python -m py_compile`)
- **Se sei a 4.8-4.10**: testing-only, sicuro
- **Verifica finale post-implementazione**: lo script `scripts/99_replay_live_vs_training.py` deve produrre output "✅ 0 mismatch"

**Stato attuale (aggiornato 2026-06-02 23:35):**
- ✅ Componenti core implementati: `get_canonical_feature_names`, `LiveCandleBuffer`, `FeatureAssembler`
- ✅ Parity test verificata matematicamente: **Max diff 0.000e+00** sul replay 50k candele
- ✅ 4/4 unit test in `tests/test_live_training_parity.py` passano in 12.8s
- ✅ Script `99_replay_live_vs_training.py` aggiornato: prima 3 disallineamenti → ora 0
- **PROSSIMI STEP (per nuova sessione):**
  1. **4.6 — Integrare in `LiveEngine.__init__`**: sostituire `self.buffer = LiveFeatureBuffer(...)` con i 3 nuovi componenti. Il main loop in `LiveEngine._on_candle_close()` (o nome simile) chiama `compute_window()` su `FeatureAssembler` invece di `get_window()` su `LiveFeatureBuffer`. Cercare con grep `self.buffer` per trovare tutti i call site.
  2. **4.7 — Rimuovere `_pad_or_truncate`**: nel metodo `_predict` (riga ~964-988) sostituire il blocco `if n_live < n_model: pad ...` con `assert window.shape[1] == 104, "feature mismatch"`. Window arriva già a 104 dal nuovo path.
  3. **4.4 — `FundingRatePoller`**: implementare poller con asyncio task + `quantsys.data.fetch_funding_rate` ogni 1h. Per ora workaround: `LiveEngine.__init__` legge `data/funding_rate.parquet` al boot e ne passa una copia a `FeatureAssembler.compute_window()`.
  4. **4.10 — Smoke test live**: `python scripts/04_live_signals.py` → connessione WS, warmup, primo segnale entro 2 min. Verificare niente warning "feature mismatch".
  5. **4.11 — Update doc finale**: marcare `BLOCKER #1 ✅ DONE` e aggiornare TEORIA.md/AVVIO.md/README se citano "39 feature live mismatched".

**Files modificati in questa sessione (commit-able):**
- `quantsys/features/__init__.py` (+50 righe: `get_canonical_feature_names`)
- `scripts/04_live_signals.py` (+~220 righe: `LiveCandleBuffer` + `FeatureAssembler`; `LiveFeatureBuffer` legacy intatta con tag DEPRECATED)
- `scripts/99_replay_live_vs_training.py` (riscritto: era pad-trun check, ora parity diff)
- `tests/test_live_training_parity.py` (nuovo: 4 test, 12.8s)

---

**Seeding all'avvio:** caricare gli ultimi 30g di klines 1m (paginazione Binance, una-tantum, cache locale) o riusare `data/raw_candles.parquet`.

### Stage 5 — Parity test + replay backtest (gate go/no-go) ⏳ DA FARE

1. **Parity test:** vettore live vs `FeatureBuilder` sulla stessa finestra storica, `max|Δ| < tol` per ogni feature.
2. **Replay backtest:** candele storiche attraverso la pipeline live → segnali/PnL devono combaciare col backtest offline.
3. Solo dopo entrambi i gate verdi: **avviare paper-trading** (2-4 settimane Sharpe live > 0.5 prima di considerare il mainnet).

---

## 🔵 Binance Futures Testnet — Fasi 2-5

**Stato:** Fase 1 ✅ done (`.env` + `scripts/00_test_binance_testnet.py`). Fasi 2-5 pending. Tempo residuo: 8-13 ore.

Obiettivo: il live engine invia ordini reali sul Futures Testnet (`testnet.binancefuture.com`) parallelamente al portfolio simulato, con riconciliazione periodica. Valida latency esecuzione reale, slippage testnet, comportamento SL/TP exchange-side, bug operativi che il backtest non copre.

> **Prerequisito:** BLOCKER #1 (Stage 2-5) risolto prima, altrimenti il testnet riceve segnali da un modello con input scorrelato.

### Fase 2 — Architettura execution layer (2-4h)

**Nuovo package** `quantsys/execution/`:

```
quantsys/execution/
├── __init__.py          # factory create_adapter(mode, ...)
├── base.py              # ABC ExecutionAdapter
├── paper.py             # in-memory simulato (rifactor del comportamento attuale RiskManager)
└── binance_futures_testnet.py  # REST via python-binance.Client(testnet=True)
```

**Interface ABC** (`base.py`):

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

**Config nuova** in `config/default.yaml`:
```yaml
live:
  execution_mode: paper        # paper | testnet_futures
  testnet_futures:
    symbol: BTCUSDT
    margin_type: ISOLATED      # ISOLATED | CROSSED
    max_leverage: 3            # modulata da conviction; 1 = no leva
    leverage_conviction_alpha: 1.0
    # api_key/secret da env BINANCE_TESTNET_API_KEY / _SECRET (.env)
```

**Leva dinamica conviction-based** (decisa 2026-05-24):
```python
def _conviction_leverage(conviction: float, max_lev: int, alpha: float = 1.0) -> int:
    """conviction=0 → 1x, conviction=0.5 → ~max_lev/2, conviction=1 → max_lev."""
    lev = 1 + (max_lev - 1) * (conviction ** alpha)
    return max(1, min(max_lev, round(lev)))
```
Chiamato in `RiskManager.open_position` PRIMA del `place_market_order`.

### Fase 3 — Wiring RiskManager (2-3h)

Modifiche a `quantsys/trading/__init__.py`:
- `RiskManager.__init__` accetta `execution_adapter: ExecutionAdapter | None = None` (default = paper).
- `open_position`: dopo calcolo SL/TP+size, se `self.adapter is not None`:
  1. `set_leverage(symbol, _conviction_leverage(...))`
  2. `entry_order_id = place_market_order(side, qty)`
  3. `sl_order_id = place_stop_market(opposite_side, qty, sl_price)`
  4. `tp_order_id = place_take_profit_market(opposite_side, qty, tp_price)`
  5. Persisti i 3 orderId nella `Position`.
- `update_trailing`: se SL aggiornato e adapter: `cancel_order(sl_order_id)` + nuovo `place_stop_market` + update `position.sl_order_id`.
- `close_position`: se adapter: `cancel_all_orders` (chiude SL/TP residui) + `place_market_order(opposite_side, qty)` (chiusura at-market).

**Casi edge:**
- Partial fill: poll fino a FILLED o cancel + market sul resto.
- Liquidation: recovery automatico (chiudi paper, log WARNING, riparti pulito).
- Rate limit: Binance Futures 1200 weight/min; un `open_position` ≈ 4 chiamate REST → max ~300 open/min teorico (sufficiente).

### Fase 4 — Riconciliazione paper vs testnet (2-3h)

Nuovo modulo `quantsys/execution/reconciliation.py`:

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

Output: `signals/reconciliation.jsonl` (~2880 record/giorno). Warning solo su drift > 0.5%. Integrazione in `04_live_signals.py` via `asyncio.gather`.

### Fase 5 — Test end-to-end (2-3h)

**Pre-flight:**
1. `python scripts/00_test_binance_testnet.py` (già ✅)
2. Set `live.execution_mode: testnet_futures` in config
3. Run con `max_leverage: 1` (no leva) come safety net iniziale, monitor armato per: WS Binance connesso, primo segnale, primo OPEN sul testnet (verifica orderId + posizione), primo update SL trailing, primo CLOSE, riconciliazione delta < 0.5%
4. Lasciare girare 1-3h con `max_leverage: 1`. Se 3-5 trade OK end-to-end → alzare gradualmente (1 → 2 → 3).

**Decision criteria post 24-48h live (tutti e 4 devono essere OK):**
- Drift reconciliation < 0.5% per >95% dei sample
- Slippage reale entro 2× quello backtest (`slippage_rate: 0.0003`)
- Latency totale (signal gen → ordine fillato) < 500ms
- Zero rate limit violation

Se OK → paper-trading 2-4 settimane prima di considerare mainnet. Se fallisce uno → fix bug specifici prima di riprovare.

---

## 🟡 Roadmap modello — Fix #3, #4, #5, #6

Tutti gated post paper-trading (paper-trading è gated post BLOCKER #1).

| # | Fix | Da | A | Effort | Beneficio atteso |
|---|---|---|---|---|---|
| 3 | `model.window_size` (T) | 120 | **240** | config + ~30% VRAM | DA ↑ 1-2%, vol cluster catturato |
| 4 | `validation.n_folds` | 3 | **5-6** | config + +50% test time | CI bootstrap walkforward più affidabili |
| 5 | Multi-timeframe (1m+5m+1h) | — | nuovo pkg `mtf/` | 6-9 settimane elapsed | DA ↑ 2-4%, contesto 24h |
| 6 | `mamba-ssm` (kernel CUDA) | pure-PyTorch | kernel ufficiale | ~1h setup + retrain | speedup TCN+Mamba 3-5× |

### Fix #3 — Window size T 120 → 240

**Razionale:** vol clustering BTC 1m ha half-life ~2-6h (Engle 1986, Bollerslev 1986). Con T=120 (2h) vedi solo metà del cluster. Letteratura (PatchTST Nie 2023, iTransformer Liu 2024) testa lookback 96-720; plateau intorno 192-384.

```yaml
model:
  window_size: 240
validation:
  embargo_steps: 3000   # da 1500 (deve essere ≥ window_size + horizon)
```

Poi: `python scripts/01_download_data.py` (ricostruisce npz) + `python run_all.py --distill --skip-update --skip-macro --no-browser`.

**Smoke test preliminare** su solo iTransformer per validare VRAM:
```powershell
$env:QUANTSYS_ARCH = "itransformer"
python scripts\02_train.py --n-ensemble 1
```
Se OOM su 8GB: `training.batch_size: 64 → 32` + `gradient_accumulation_steps: 2 → 4` (mantieni effective batch=128).

Impatti: VRAM training ~+30%, tempo per epoca +30-50%, samples utilizzabili −1% (più candele wasted per warmup).

### Fix #4 — Walkforward folds 3 → 5-6

**Razionale:** 3 fold danno CI bootstrap larghi (Sharpe [+0.78, +74.70] su 42 trade). Letteratura finance ML (López de Prado 2018, AFML cap. 7): per crypto, **5-6 fold** sono lo standard.

```yaml
validation:
  n_folds: 6
  embargo_steps: 3000   # se fix #3 applicato, altrimenti 1500
```

Poi: `python scripts/02b_walkforward_validate.py`. **Non** richiede retrain. Tempo +50% (~30-45 min totali).

**Controlli in `results/{arch}/walkforward_metrics.json`:**
- `da_per_fold`: tutti > 0.51, std < 0.005
- `spearman_per_fold`: tutti positivi
- `sharpe_per_fold` (bootstrap): CI esclude zero in **almeno 4 fold su 6**

Se divergenza ampia (std DA > 0.01) → modello non stabile attraverso regime → torna a più dati.

### Fix #5 — Multi-timeframe (1m + 5m + 1h)

**Stato:** miglioramento con il potenziale più alto ancora aperto. **Prerequisito:** ≥ 7-14 giorni di paper-trading data per baseline reale.

**Architettura proposta:**
```
1m  → 120 candele (micro-pattern, esistente)
5m  →  24 candele (swing intraday, 2h contesto)
1h  →  24 candele (trend giornaliero, 24h contesto)
```
3 encoder separati della stessa famiglia, fusion finale con cross-attention o gated concat.

### Strategia: esperimento parallelo isolato in nuovo package `mtf/`

Per evitare di rompere il codice di produzione e rollback istantaneo, sviluppo in directory parallela che riusa per import il più possibile.

```
mtf/                    # NUOVO package, isolato
├── __init__.py
├── data_builder.py     # resample 1m→5m, 1m→1h + dataset build (3 stream allineati right-bound)
├── models.py           # wrapper Quant*Mtf che compongono i modelli quantsys/
├── train.py            # training loop con DataLoader che yielda 3 tensori X
├── backtest.py
├── live_signals.py     # solo se va in produzione
└── run.py
```

Dataset/modelli/risultati paralleli: `data/mtf_dataset.npz`, `models/mtf_{arch}/`, `results/mtf_{arch}/`.

**Riusa** (import da `quantsys/`): loss functions (`student_t_nll`, `quantile_loss`, `direction_value_loss`), utilities (`load_config`, `setup_device`, `PipelineState`), risk manager, signal generator, macro encoder, regime detector, FeatureBuilder (eseguito 3 volte sui 3 timeframe, stesso codice).

**Crea nuovo** solo dove cambiano le shape: data_builder, models wrapper, training loop con 3 tensori X.

**Vantaggi parallelo:** rollback istantaneo (`rm mtf/`), zero regressioni produzione, A/B validation pulita, niente conflitti con paper-trading single-tf in corso.

### Impatti attesi

| Aspetto | Single-tf (oggi) | Multi-tf | Delta |
|---|---|---|---|
| Storage `lstm_dataset.npz` | (107480, 120, 119) ~6.1 GB | + (107480, 24, 119)×2 | **+40%** (~8.5 GB) |
| Training iTrans 200 epoche | ~6h | ~10-14h | +60-100% |
| Distill pipeline completa | ~2-3h | **~30-50h GPU** | **10-20×** |
| VRAM TCN+Mamba batch 64 | ~4 GB | ~5-6 GB | ⚠ stretto su 8GB |

⚠ **Iterazione lenta**: ogni esperimento richiede 6-12h. Va pianificato.

| Metrica | Single-tf | Multi-tf atteso | Confidenza |
|---|---|---|---|
| Directional Accuracy | 51.7-53.2% | **53-56%** | alta |
| Spearman ρ | 0.034-0.062 | **0.07-0.12** | alta |
| Sharpe backtest | +18.71 | +20-40% | bassa (calibrazione) |
| Win rate | ~64% | 65-70% | media |
| Max drawdown | 0.83% | atteso simile o migliore | media |

**Cosa specifico cattura:**
1. Trend giornaliero (1m con T=120 vede solo 2h)
2. Funding rate cycle 8h (il 1h × 24 cattura 3 cicli completi)
3. Volatility regime shifts che durano ore (distingue "compressione che precede breakout" da "lateralità che continua")
4. Daily seasonality storica (apertura US/EU/Asia)

**Rischi:**
1. **Data leakage nei resample**: se il 5m bar al minuto T:00 include T+1..T+4 → predici il futuro. **Test critico**: shuffle X_train e verifica che il modello NON impari.
2. Mismatch live ↔ backtest sul warmup (live aspetta 24h, backtest skippa 1440 candele).
3. Curse of dimensionality (3 encoder ≈ 3M params vs 107k samples = 28× sfavorevole).
4. Costo iterazione 6-12h.

**Vale la pena se:** paper-trading conferma Sharpe live > 0.5 per 2+ settimane AND vuoi spingere ICIR 0.79 → 0.9+. **No** se cerchi quick win o sistema non ancora validato live.

**Costo totale stimato:** 6-9 settimane elapsed (1-2 settimane coding + 1 settimana debug + 30-50h GPU primo training + 2-3 settimane tuning + 1-2 settimane validazione).

### Fix #6 — mamba-ssm package (CUDA Toolkit + kernel ufficiale)

**Speedup atteso:** +3-5× sul Mamba branch (sopra il +1.4-1.6× già ottenuto con AMP off + chunk pre-alloc).

**Razionale:** l'implementazione attuale in `quantsys/model/tcn_mamba.py` è **pure-PyTorch** (`SimplifiedMambaBlock._parallel_scan_chunk`). Il pacchetto `mamba-ssm` di Tri Dao implementa un kernel CUDA fuso (selective scan) che: carica `(A, B, C, Δ, x)` in shared memory una sola volta, esegue lo scan in registro/SRAM senza scrivere intermedi in HBM, usa parallel prefix scan (Blelloch) su tile, ricomputa lo stato in backward (memory-efficient à la Flash Attention). O(L) compute con costante piccola, O(1) memoria HBM per token.

**Prerequisiti su questa macchina:**
- ✅ RTX 2070 SUPER (Turing 7.5), Python 3.12, PyTorch 2.5.1+cu121, CUDA runtime 12.1
- ❌ **CUDA Toolkit (dev) 12.1.x** mancante — deve matchare `torch.version.cuda`. Mismatch (es. CUDA 12.4 con torch+cu121) → linker errors.
- ⚠ **MSVC Build Tools 2022** probabilmente mancanti
- ❌ `CUDA_HOME` env var da settare

**Procedura:**
1. **MSVC Build Tools 2022** da https://visualstudio.microsoft.com/downloads/?q=build+tools — workload "Desktop development with C++" (MSVC v143, Windows 11 SDK, CMake) ~6 GB.
2. **CUDA Toolkit 12.1** da https://developer.nvidia.com/cuda-12-1-1-download-archive (Windows x86_64, exe local, ~3 GB). Custom install: ☑ Development + Runtime; ☐ Driver components (hai già quello dell'RTX).
3. **Env vars** (PowerShell):
   ```powershell
   [Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1", "User")
   [Environment]::SetEnvironmentVariable("CUDA_PATH", "$env:CUDA_HOME", "User")
   # aggiungere $CUDA_HOME\bin al Path
   ```
4. **Install** (no build isolation, forza compilazione contro PyTorch installato):
   ```powershell
   pip install causal-conv1d>=1.2.0 --no-build-isolation
   pip install mamba-ssm --no-build-isolation
   ```
5. **Modifica** `quantsys/model/tcn_mamba.py`: sostituisci `MambaBranch` con import condizionale del kernel ufficiale, mantieni `SimplifiedMambaBlock` come fallback.
6. **Retrain** (i checkpoint NON sono compatibili): backup `models/tcnmamba` + `Remove-Item models\tcnmamba\best_model*.pt` + retrain TCN+Mamba da zero. Verifica `val_nll` converga su valori simili (±5%). Tempo/epoca atteso ~1.5-2 min vs 4.5 min attuali.

**Rollback:** `pip uninstall mamba-ssm causal-conv1d` → il codice rileva automaticamente `_HAS_MAMBA_SSM = False` e usa il fallback.

**Quando farlo:** retrain frequenti TCN+Mamba (ablation studies), `mamba_layers > 3`, sequenze più lunghe (T > 240 per multi-tf). **Non** se il training corrente è "abbastanza veloce" o stai per cambiare arch.

---

## 🟢 Audit residui (low-priority, non bloccanti)

4 issue MEDIE + 1 INFRASTRUCTURE dal grand audit 2026-05-23 (8/8 CRITICHE + 8/8 ALTE + 5/9 MEDIE già chiuse):

| # | File:line | Issue | Fix proposto | Effort |
|---|---|---|---|---|
| 21 | `quantsys/trading/__init__.py:395` | NaN check `x != x` criptico, solo su `size` | NaN guard esplicito all'inizio di `open_position` con log warning | 10 min |
| 23 | `quantsys/data/__init__.py:48` | Sanity OHLCV `high > close * 10` può scartare flash crash legittimi | rilassare soglia o usare prezzo candela precedente come riferimento | 15 min |
| 27 | `quantsys/model/ensemble.py:104-114` | `arch_names` non impostato nei fallback `load` | non critico, default OK | 5 min |
| 28 | `quantsys/features/__init__.py:251` | `vol_x_pos` crash se colonne assenti su dataset corto | `.get(col, 0)` o try/except con log | 10 min |
| #5 ⚠ | `quantsys/trading/__init__.py:122` + `scripts/03_backtest.py:571-576` | `SignalGenerator.set_regime_threshold` exists ma chiamate DISABILITATE | calibrare empiricamente soglie regime sui dati post-fix denorm (1-2h + retest), oppure rimuovere dead code | 1-2h |

**Contesto fix #5:** bisect 2026-05-24 ha mostrato che le soglie regime hardcoded (overheating +3pp, stagflation +5pp sul default 0.52) riducevano Sharpe da +18.71 a −4.44 (filtravano 27/42 trade vincenti). Infrastructure resta ma dead code.

Effort totale chiusura completa: ~1h (4 medie) + 1-2h (#5 se si decide di calibrare).

---

## 📋 Soglie di promozione paper-trading

Indipendenti dai fix, da soddisfare CONTEMPORANEAMENTE prima di andare live (3/4 raggiunte 2026-05-23):

- ✅ Sharpe CI bootstrap (5000 iter): lower bound > 0 (+0.78)
- ✅ Stress test (`pessimistic_fee`, `flash_crash_vol`): almeno break-even (+7.22 / +12.30)
- ⚠ **WHR walkforward (3+ fold): > 0.53 stabile** — iTransformer 0.567, ma N-HiTS/TCN+Mamba 0.50-0.53 (modelli per-fold sotto-trained con max_epochs=40, ricalibrazione post paper-trading via fix #4)
- ✅ Fee/gross ratio: < 30% (30.3% al limite)

Le soglie restano valide anche dopo il retrain post-BLOCKER #1; vanno rivalutate sui nuovi modelli a 104 feature.

---

## 🧭 Regola d'oro

**Un fix alla volta, ogni cambio validato da backtest con CI bootstrap.** Cambiare più cose insieme rende impossibile attribuire causalmente il delta.

Pattern raccomandato:
1. Applica un fix singolo
2. Retrain completo (un solo modello base se possibile, es. iTrans, per smoke test)
3. Confronta `val_nll`, `DA`, `Spearman`, `Sharpe CI` con baseline pre-fix
4. Se ≥2% miglioramento → mantieni e passa al prossimo
5. Se peggiora o invariato → rollback e analizza prima di provare il prossimo

**Lezione 2026-05-24:** attivare fix "completi" senza validation pre-merge può accendere dead state non calibrati (caso #5). Bisect rapido (un fix alla volta) trova il colpevole in 2 iterazioni anche con codebase complesso.

---

## 💡 Insights consolidati (validi long-term)

1. **Modello predittivo sano in tutti i setup**: walkforward DA 0.53-0.54, Spearman 0.08-0.09, σ ben calibrato. Il problema, quando emerge, è quasi sempre nel **trading layer** (scala, soglie, SL/TP), non nel modello — vedi sessione 2026-05-23 per il caso paradigmatico (Sharpe −256 → +18.7 con 1 moltiplicazione mancante).
2. **h=15 è strutturalmente perdente**: cost roundtrip 26 bps ≈ |realized return medio| 25 bps. h=30 raddoppia il segnale mantenendo costo costante. Già applicato.
3. **`max_sigma` va sempre dimensionato sulla distribuzione σ del modello specifico** (es. p99 della σ_test). Valori arbitrari sono inutili.
4. **`trailing_atr_mult: 1.5`** ≈ 11 bps di trail su BTC 1m → chiude su rumore (< del cost 26 bps). Su 1m bar `use_trailing_stop: false` (attuale) batte qualsiasi trailing tunato.
5. **Verificare le scale unit-by-unit prima di retrainare**: per 6+ sessioni a maggio 2026 abbiamo cercato fix sui pesi del modello (RevIN, h, stride, multi-teacher) — il vero bug era 1 moltiplicazione mancante in 2 file (denormalizzazione z-score → raw).
