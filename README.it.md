🇮🇹 Italiano · [🇬🇧 English](README.md)

# QUANTSYS — Motore neurale di forecasting per BTC/USDT

Questo progetto pone due domande su Bitcoin. La prima: una rete neurale può prevedere quanto si muoverà il prezzo — non in che direzione, ma quanto saranno ampie le oscillazioni nelle prossime 30 ore, con la previsione aggiornata ogni ora? Su dati che il modello non ha mai visto in addestramento, le sue previsioni hanno un errore più basso dei modelli econometrici standard usati come riferimento. La seconda: quella previsione si può trasformare in profitto vendendo volatilità, cioè incassando i premi delle opzioni quando le oscillazioni attese sembrano prezzate troppo care? Finora no: le regole di vendita provate qui non hanno superato i criteri di successo e fallimento fissati prima di eseguirle.

![Errore di previsione della rete neurale e dei modelli di riferimento](docs/assets/qlike_comparison.png)
*Errore di previsione sullo split di test fuori campione, una barra per modello: più basso è meglio.*

![Varianza prevista contro varianza realizzata nelle prossime 30 ore](docs/assets/rv_pred_vs_actual.png)
*Varianza prevista contro realizzata delle prossime 30 ore sull'ultimo tratto dello split di test fuori campione, con l'intervallo 10–90% della rete.*

Motore neurale di forecasting probabilistico su BTC/USDT + analytics opzioni crypto. **Linea di produzione: volatilità @ 1 ora** (`config/default.yaml → features.target_type: log_rv`, `data.interval: 1h`; design interval-agnostic, 1m = identità, perimetro 1m in backup). Il target `log_rv` è l'**unico segnale validato OOS** del progetto: batte **HAR-C** — la variante HAR sulla sola componente continua jump-robust (`C = min(RV, BV)`), cioè la baseline econometrica più forte fra quelle testate — del **32% in QLIKE su test** (0.236 vs 0.346; naive 0.793), con val→test coerenti. Modello di produzione: **iTransformer 5 membri**. Secondo braccio attivo: **short-vol** in forward test su Deribit testnet (`scripts/04b_vol_paper.py`, servizio systemd 24/7 su VPS). Il filone **direzionale** non ha alpha OOS a nessun timeframe testato (1m e 1h): il codice resta vivo e bit-invariato come negative-control documentato.

**Stack:** Python 3.12 | PyTorch (CUDA) | NumPy/Pandas | Binance REST+WebSocket | FRED API · Deribit public REST (dashboard/IV/forward test vol).

### Da dove iniziare

Lettura in 60 secondi — quattro puntatori, in ordine di importanza:

| Cosa | Dove |
|---|---|
| **Il risultato.** Il forecast NN della realized variance batte **HAR-C** (`C = min(RV, BV)`, componente continua jump-robust — la baseline forte, non quella comoda; riferimento dal 2026-07-31, gate C3, che ha sostituito HAR-CJ a parità di accuratezza) del **32% in QLIKE su test** e del 23% su val (0.236 vs 0.346; naive 0.793 — significatività Diebold-Mariano HAC **p ≤ 4.3·10⁻⁴**, misurata contro HAR-CJ, vedi `THEORY.it.md` §12.2), val→test coerenti. Contro HAR-RV semplice la banda sarebbe −27%÷−36%: la differenza è la misura di **quanto del vantaggio veniva da una baseline debole**, quantificata con un gate pre-registrato invece che assunta. ⚠ **Il gate pre-registrato del 2026-06-10 usa HAR-RV come denominatore, il claim usa HAR-C: sono due affermazioni diverse, entrambe vere** — pannello di riconciliazione in testa a `THEORY.it.md` §12.2 | `scripts/vol/dev_vols_qlike.py` — il giudice che produce il numero, split val-first |
| **Come si decide se un'idea vive o muore.** Ogni esperimento è **pre-registrato**: metriche, soglie e n minimo scritti e committati *prima* di girare | `THEORY.it.md` §12.1 (protocollo in 5 passi) · `STATUS.md` (pre-registrazioni in testa) |
| **Cosa è stato provato e NON funziona**, con i numeri: direzionale a 1m e 1h, semivarianza firmata, IVS relative-value, 4 lever di training, gating per regime | `THEORY.it.md` §12.3-12.4 (corpus KILL, con i numeri) |
| **Cosa può verificare un lettore esterno** senza scaricare dati | `pytest tests/` → **suite intera verde su CPU**, un test saltato per costruzione: parity live↔training bit-perfect, invarianti z-score/interval, bit-parity del regime incrementale, ordine di init pyarrow, guard anti-degradazione del regime |

Il progetto è organizzato attorno a un'asimmetria dichiarata: **i momenti pari (varianza, RV) generalizzano fuori campione su questo asset, i momenti dispari (segno, asimmetria) no** — per la rete *e* per le baseline econometriche. Le tre linee di codice (vol-forecasting, monetizzazione short-vol, direzionale) esistono per documentare quella asimmetria, non per nasconderla.

### Riproducibilità

`data/`, `models/` e `results/` sono **gitignored**: pesi e parquet sono grandi e i dati di mercato non sono ridistribuibili. Cosa significa in pratica per chi clona:

- **Verificabile subito, senza dati:** `pip install -e .` → `pytest tests/` (solo CPU, circa un minuto; passa tutto tranne un test saltato per costruzione, la cui fixture sintetica è troppo corta per una feature a 30 giorni; il conteggio corrente lo stampa pytest). Include i golden test sulla lista delle 104 feature e la parity live↔training.
- **Rigenerabile:** dataset (`scripts/01_download_data.py`, Binance pubblico + una chiave FRED gratuita per la macro) → training (`scripts/02_train.py --n-ensemble 5`, ~27 min per 5 seed iTransformer su RTX 2070 Super) → giudice QLIKE (`scripts/vol/dev_vols_qlike.py`).
- **NON rigenerabile** (raccolta forward, per costruzione): `data/iv/`, `data/orderbook/`, `data/deribit_trades/`, `results/vol_paper/` — snapshot IV/book/trade e forward test su testnet. I numeri del braccio short-vol non sono riproducibili da un clone: sono un log d'esperimento, e sono presentati come tali. ⚠ **2026-09-13/15 (locale, non deployato):** tre correzioni di correttezza nell'esecutore adattivo di `scripts/04b_vol_paper.py` — fee/timing di fill trattati come ignoti (mai zero/inventati) quando la copertura dei trade reali non è verificabile, coda diagnostica/hedge saltata quando un tick lascia il journal bloccato invece di registrare uno snapshot "flat" falso, e identità dei trade richiesta con strumento/verso concreti persistiti nel record di recovery (prima vivevano solo nel journal, cancellato a flat verificato); dettaglio in `START.it.md` §5.3, `tests/test_adaptive_structure.py` 40/40.

### Mappa della documentazione

Documentazione in **due lingue, un file per lingua** (inglese `NOME.md`, italiano `NOME.it.md`, selettore di lingua in prima riga). Ruoli disgiunti — ogni fatto vive in un solo posto:

- **[README.it.md](README.it.md)** (questo file) — *cosa* fa il sistema e *perché*: panoramica, caratteristiche, struttura del progetto, puntatori.
- **[START.it.md](START.it.md)** — **runbook**: tutti i comandi operativi, setup dettagliato, tuning hardware, routine di sessione, deploy VPS.
- **[THEORY.it.md](THEORY.it.md)** — **matematica**: derivazioni di loss, regime detection, Monte Carlo, distillation, trading layer.
- **[CHANGELOG.it.md](CHANGELOG.it.md)** — milestone in ordine cronologico inverso.
- **[STATUS.md](STATUS.md)** — **fonte canonica** dello stato: periodo corrente + tutti i gate pre-registrati aperti. Storico antecedente al 2026-07-08 in **[docs/STATUS_ARCHIVE_2026H1.md](docs/STATUS_ARCHIVE_2026H1.md)** (read-only).
- **[THEORY.it.md](THEORY.it.md) §12** — protocollo sperimentale + corpus dei risultati negativi (KILL) con le soglie di gate.
- **[docs/MODEL_IMPROVEMENTS.it.md](docs/MODEL_IMPROVEMENTS.it.md)** · **[docs/ROADMAP_VOL_BOOK.it.md](docs/ROADMAP_VOL_BOOK.it.md)** — backlog e item aperti.
- **[Diagrammi delle architetture, pagina live](https://luca-feleppa.github.io/quantsys/architetture.html)** (sorgente: [docs/architetture.html](docs/architetture.html)) — **diagrammi interattivi** delle architetture, ricavati dai `forward` e non dalla doc: pipeline dei dati, iTransformer, TCN+Mamba, N-HiTS, CAFN, MoE/MoU, testa di output, e il cablaggio interno di attenzione, convoluzione, stato e decomposizione. Le forme dei tensori si aggiornano al variare dei parametri; toggle IT/EN nella pagina.

---

## 1. Panoramica e obiettivi

Anziché una stima puntuale, QUANTSYS restituisce l'**intera distribuzione** del target — quantili condizionali con `loss_type: quantile` (default di produzione), oppure head parametrica t-Student μ/σ/ν con `loss_type: t_student` (§4.3) — su BTC/USDT a **intervallo candela parametrico** (`data.interval`; default corrente `1h`, perimetro `1m` legacy in backup — tutte le conversioni temporali derivano da `interval_minutes`, identità a 1m). Sullo stesso spine convivono tre linee, con esiti molto diversi:

- **Linea VOL — forecasting (produzione, validata).** Target `log_rv` (log realized variance su h barre): unico segnale che generalizza OOS, batte **HAR-C** (baseline sulla componente continua jump-robust) del 32% in QLIKE su test. Giudicata con QLIKE, **mai** tradata nel backtest direzionale.
- **Linea VOL — monetizzazione (forward test in corso).** Braccio **short-vol**: straddle su Deribit testnet guidati dal confronto RV-prevista vs IV (`04b_vol_paper.py`, systemd 24/7 su VPS). L'edge VRP è strutturalmente confermato sul backtest storico FHS-GJR-GARCH (2019→2026, n=2538), ma **entrambi i gate pre-registrati sono falliti**: v1 **FAIL 0/3 (2026-07-18)** — il VRP è positivo, la regola v1 non lo monetizza — e v2 **delta-hedged FAIL 2/3 (2026-08-11)**, dove l'hedge riduce la varianza per-trade del **55.3%** ma il suo costo di realizzazione (**76% fee perp**, funding <1%) eccede di 2.6× il budget pre-registrato. Contatori dei gate ancora in accumulo in `STATUS.md`.
- **Linea direzionale (legacy, negative-control).** Target `ret`, trading layer Kelly/SL/TP, backtest e live paper. Nessun alpha OOS né a 1m né a 1h; conservata come negative-control documentato, non ri-aprirla senza un'ipotesi nuova.

✅ **Stato live engine (direzionale):** paper-only (nessun ordine reale), **BLOCKER #1 RISOLTO (2026-06-05)**. Il path live costruisce le **104 feature canoniche** via `FeatureBuilder` con lo scaler del training, con **parity feature *e* segnale bit-perfect** vs backtest (`tests/test_live_training_parity.py`, replay Δ=0). ⚠ I segnali live riflettono un backtest che resta negativo OOS: il paper-trading direzionale accumula trade forward, senza aspettativa di Sharpe>0.

### 1.1 Caratteristiche principali

- **Output probabilistico** — quantili condizionali (default) o t-Student (μ, σ, ν); loss NLL + penalità asimmetrica + CRPS + Direction-Value (§4.3).
- **104 feature ingegnerizzate** post-filtro C-funding, lista canonica sotto golden test (§3.2).
- **4 architetture** intercambiabili — iTransformer, N-HiTS, TCN+Mamba, LSTM legacy — dietro un forward contract unico, con ensemble eterogeneo e **distillation multi-teacher** target-aware (§4.1-4.2).
- **Rilevamento regimi BTC** `RegimeMarkovBTC`, Markov-Switching causale con refresh incrementale (§3.3).
- **Giudici della linea vol** QLIKE vs pannello a tre baseline HAR (RV = denominatore del gate, **C** = denominatore del claim, CJ = diagnostica) + HAR-RV per-fold nel walk-forward, split val-first (§5.1).
- **Validazione anti-leakage**: walk-forward purged k-fold con embargo, backtest con stress test, bootstrap CI, analisi per regime (§5.2).
- **Monte Carlo** 2000 scenari GJR-GARCH(1,1), parametri stimati su rendimenti orari (§4.4).
- **Trading layer** Kelly frazionario + SL ATR + trailing + circuit breaker DD 15% MtM (§6.1).
- **Forward test vol** su Deribit testnet con delta-hedge, attribuzione PnL ex-post e **collector 24/7** su VPS (§6.2).
- **Dashboard** Deribit Options Risk Terminal, GPU-free, indipendente dalla pipeline ML (§6.3).

---

## 2. Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python scripts/00_check_setup.py
```

`scripts/00_check_setup.py` verifica CUDA, dipendenze e connessione Binance. Il fallback **CPU-only** funziona (rallentamento su training; piena velocità su backtest/live). Setup di riferimento: **RTX 2070 Super (8 GB VRAM)** — il training multi-seed × multi-arch va sequenziale (OOM), e live/paper non va girato in parallelo a training/inferenza (contesa CUDA). La chiave FRED è **opzionale**: copia `config/secrets.yaml.example` in `config/secrets.yaml` (gitignored, mai committato); senza chiave si lavora con rate limit più stretti.

→ **Setup dettagliato, note PowerShell/Windows, tuning VRAM (4GB / ≥16GB), tempi di training: [START.it.md](START.it.md) §1.**

---

## 3. Dati

La pipeline scarica lo storico BTC/USDT da Binance (default: candele **1h** dal 2019-01-01, ~65k barre), costruisce **104 feature**, le normalizza con un **RobustScaler globale** (mediana/IQR, fittato SOLO sul training — no leakage val/test), e finestra in sequenze **120×104** (`model.window_size: 120` = 5 giorni a 1h; `window_stride: 1`). Split temporale 80/10/10 (`training.val_fraction`/`test_fraction` = 0.1/0.1 → ~6.5k finestre per val e per test sul dataset corrente). I parametri scaler + config feature + `target_scale` + `forecast_horizon` + `interval` sono persistiti in `PipelineState` (unico contratto train↔inference).

### 3.1 Target & log-return

Tutto lavora su **log-return** (stazionari, simmetrici), mai prezzi assoluti. L'orizzonte è di **30 barre** (`features.forecast_horizon: 30` — 30h a 1h). Cambiarlo richiede ri-generare il dataset e ri-allineare `PipelineState.forecast_horizon` (validato a runtime con `RuntimeError` in backtest e live). Famiglie di target via `features.target_type`: `log_rv` (produzione vol: log realized variance Σr² su h barre), `ret` (direzionale legacy: somma dei log-return futuri), `log_rs_ratio` (probe asimmetria semivarianza, FAIL — §7).

### 3.2 Le 104 feature

**104 feature = 86 dinamiche + 18 strutturali**: VWAP + Volume Profile, CVD, microstructure delle candele, funding rate, volatilità multi-finestra, lag returns, encoding temporale, feature interactions, livelli strutturali. Il **filtro C-funding** (`LIVE_DROP_FEATURES` in `quantsys/features/__init__.py`) rimuove 15 feature — quelle a lookback lungo (90d/365d), frazionalmente differenziate o dipendenti da Volume Profile long — per due motivi cumulabili: permutation importance con ROI ≤ 0 (rumore o dannose) e/o lookback non calcolabile nel buffer live. Tenerle rompe la parity live↔backtest senza guadagno predittivo. La lista è derivata una volta sola da `canonical_feature_columns` sotto golden test: il conteggio è **verificato sul dataset, non assunto**. Dettaglio delle famiglie in [THEORY.it.md](THEORY.it.md) §3.

### 3.3 Macro & rilevamento regimi

Le macro USA (FRED + yFinance: DXY, VIX, tassi, oro) alimentano un `MacroEncoder` (16-dim), scollegato dal regime detector. **`RegimeMarkovBTC`** (Markov-Switching, Hamilton 1989) è fittato sulla realized volatility oraria di BTC ed è **CAUSALE by design**: probabilità *filtered* (mai smoothed), Hamilton filter forward-only, walk-forward expanding con burn-in/retrain da `macro.hmm_burn_in_days`/`hmm_retrain_days`. Produce **3 regimi data-driven** persistiti in `data/regime_probs.parquet` (index orario UTC); sul full rebuild 7 anni 1h del 2026-07-15: **R0 31% (quiet, σ²≈0.12) · R1 36% (stress, σ²≈4.8) · R2 33% (mid, σ²≈0.6)**. ⚠ Gli **indici non hanno semantica fissa tra run**: vanno ri-derivati dalle varianze a ogni full rebuild. Refresh **incrementale** disponibile (`01b --regime-incremental`): appende le sole barre nuove partendo da un checkpoint walk-forward, con bit-parity garantita da test. Uso: **stratificazione della val + diagnostica `val_nll` per regime** — **NON è una feature di input**. `RegimeMarkovSwitching` (macro USA daily), `RegimeHMM` e `RegimeSession` (Asia/EU/US) restano come alternative opzionali. Derivazione completa in [THEORY.it.md](THEORY.it.md) §4.

### 3.4 Invariante z-score vs raw

⚠ **Il bug più costoso del progetto.** Il modello predice μ/σ/ν in **spazio z-score** (target scalato dal RobustScaler; `target_scale` = IQR del target raw, persistito in `PipelineState`); il trading layer (`SignalGenerator`, `RiskManager`) opera in **spazio raw**. **Ogni entry-point DEVE chiamare `PipelineState.denormalize_predictions(mu, sigma)` subito dopo il forward**, prima del trading layer (bug 2026-05-23: saltarla → SL/TP macroscopici, Sharpe −256 → +18.7). Con target `log_rv` la sola `denormalize_predictions` è **insufficiente** (mediana log-RV ≈ −7.2): l'inversione completa è `μ·IQR + centro` dal RobustScaler persistito. Derivazione in [THEORY.it.md](THEORY.it.md) §5.

---

## 4. Modellazione

### 4.1 Architetture

Quattro architetture selezionabili via `--arch`, dietro un forward contract unico:

- **iTransformer** (`QuantiTransformer`) — attention sulle feature (non sul tempo), embedding multi-scala, O(F²). **Arch di produzione della linea vol (5 membri).**
- **N-HiTS** (`QuantNHiTS`) — interpolazione gerarchica pure-MLP, stack pooling multi-scala (8/4/1).
- **TCN+Mamba** (`QuantTCNMamba`) — convoluzioni causali dilatate (campo recettivo 127) + State Space Model con parametri input-dipendenti e fusion gated.
- **LSTM+GRU** (`QuantLSTM`) — dual-stream con attention temporale (legacy, backward compat; sotto-performante).

**Forward contract:** `forward(x, x_macro=None) -> (mu, ls2, lnu)` (`+dir_logits` se multitask; `(quantile_preds, ...)` se `loss_type=quantile`, default). Output path arch-isolati: `models/{arch}/` e `results/{arch}/`.

⚠ **Stato su disco vs capacità del codice.** L'**ensemble eterogeneo** (combinazione di archi diverse in inferenza via legge della varianza totale — `mu_ens = Σ wᵢ·muᵢ`, `sigma_ens = sqrt(Σ wᵢ·σᵢ² + Σ wᵢ·(muᵢ−mu_ens)²)`, pesi in `DEFAULT_ARCH_WEIGHTS` o dinamici inverse-NLL) è implementato e vivo, ma **la linea vol di produzione gira su solo iTransformer**: `models/nhits` e `models/tcnmamba` sono stati eliminati col cleanup del 2026-06-12 e **vanno riaddestrati prima di qualsiasi run eterogeneo**. **AMP off in inferenza** (evita NaN spectral_norm + Mamba scan). Single source of truth della composizione: `config/default.yaml → distillation.archs`. ⚠ Sul filone direzionale l'errore cross-arch è ≈0.995 → la riduzione di varianza da ensembling è ≈0.

### 4.2 Knowledge Distillation multi-teacher

Alternativa all'ensemble omogeneo 5× stessa arch: si addestrano tutte le architetture di `distillation.archs`, si assegna loro uno **score target-aware** (`teacher_score_weights`, single source of truth in `distillation.py`) — per il target direzionale `ret` pesano val_loss, Spearman ρ e directional accuracy; per il target di **volatilità** `log_rv` la directional accuracy pesa **0** (sulla varianza è il segno-vs-mediana, non un segnale tradabile: lo straddle è direction-neutral) — e si distilla ogni modello come **student** su soft labels μ/ls²/lnu ottenute come media pesata di *tutti* i teacher, con transfer delle output head e loss mista scala-normalizzata. Il vantaggio è che l'aggregazione esclude implicitamente gli archi in overfit invece di mediarli alla cieca. Formule dei pesi, temperatura softmax e loss dello student in [THEORY.it.md](THEORY.it.md) §7; comandi ed esempi in [START.it.md](START.it.md) §3.5.

### 4.3 Loss & output probabilistico

Ogni predizione è una **distribuzione condizionale completa**, non una stima puntuale — la σ predetta è ciò che alimenta sia il sizing sia il confronto RV-vs-IV della linea vol. `loss_type` (`config/default.yaml → model`) seleziona **due rami mutuamente esclusivi**:

- **`quantile` — default di produzione.** Pinball loss su 5 livelli `[0.1, 0.25, 0.5, 0.75, 0.9]`; `model(x)` ritorna `(quantile_preds, dir_logits)` e `model.predict(x)` deriva **μ = q(0.5)** (mediana condizionale) e **σ = q(0.9) − q(0.1)** (ampiezza interdecile, **non** una deviazione standard). Obiettivo effettivo: `0.7 · pinball + 0.3 · CE` della testa direzionale multitask.
- **`t_student`.** NLL t-Student (code pesanti, il crypto non è gaussiano) + **penalità asimmetrica** sugli errori di segno oltre una soglia di magnitudine + **CRPS** (calibrazione della distribuzione, non solo della media) + **Direction-Value joint loss** (accoppia il segno al valore).

⚠ I tre termini additivi del secondo ramo sono **inerti** su quello di produzione pur avendo valori non nulli in config (`asymmetry_alpha`, `crps_weight`, `dv_lambda`) — tabella dei termini attivi per ramo, definizione della pinball e trappole di lettura di μ/σ in [THEORY.it.md](THEORY.it.md) §7.0. L'output è in **spazio z-score** e va denormalizzato a monte del trading layer (§3.4). Forme chiuse, gradienti e razionale dei pesi in [THEORY.it.md](THEORY.it.md) §7.

### 4.4 Simulazione Monte Carlo

2000 scenari × 30 barre con volatilità **GJR-GARCH(1,1)** (`config/default.yaml → montecarlo`). Parametri **ri-stimati su rendimenti orari** il 2026-07-15 (QMLE gaussiano + variance targeting su ~65k barre 2019→2026): `ω=1.026e-6, α=0.1011, γ=0.0052, β=0.8732` → persistence 0.977, half-life 30h, **γ≈0.005: leverage effect quasi nullo a 1h**. Cap σ/barra parametrico (`gjr_sigma_cap: 0.13` a 1h); i valori 1m-era restano in `config/interval/1m.yaml` per il rollback. Il MC **non è sul critical path del backtest** (che usa μ/σ del modello, non il GARCH). Derivazione in [THEORY.it.md](THEORY.it.md) §8.

---

## 5. Valutazione

### 5.1 Giudici linea VOL

Metrica primaria: **QLIKE** (loss di volatilità robusta) + ratio NN/baseline; baseline di riferimento **HAR-C** dal 2026-07-31 (gate C3, che sostituisce la HAR-CJ adottata da C2 il 2026-07-30: stessa accuratezza, condizionamento 450× migliore), con HAR-RV e HAR-CJ riportate accanto come contesto. ⚠ Il **gate** pre-registrato del 2026-06-10 continua a usare **HAR-RV** come denominatore: il claim usa HAR-C, il gate no — il giudice stampa il ruolo accanto a ogni numero. I modelli vol **non vengono mai tradati nel backtest**: il giudizio è puramente predittivo. Giudici in `scripts/vol/`: `dev_vols_qlike.py` (QLIKE log-RV, giudice principale), `dev_vols_rs_judge.py` (asimmetria semivarianza), `wf_har_baseline.py` (HAR per-fold), `step0_xarch_corr.py` (kill-check correlazione cross-arch), `mfiv_comparator_judge.py` (comparatore MFIV@30h vs IV ATM sul forward test). Split **val-first** via `QUANTSYS_VOLS_SPLIT=val|test`; la logica condivisa (QLIKE, inversione log-RV, HAR) vive in `quantsys/model/vol_metrics.py`. **Risultato chiave:** il PASS **come fu registrato** (2026-06-10) è NN-log_rv 0.257 QLIKE vs HAR-RV 0.368 vs naive 0.807 su test 1h (−30%); il **claim corrente**, contro HAR-C e su coppia modello/scaler riaddestrata sull'npz corrente, è 0.236 vs 0.346 (**−31.65%**; su val −22.42%). ⚠ I due decimali identificano la coppia (modello, npz, config) che produce il numero, **non** sono un intervallo di confidenza: l'incertezza misurata è ~±0.7 punti rispetto alla config di training, zero rispetto al seed. Dal 2026-08-04 quella coppia è un **artefatto permanente e ri-giudicabile** (`models/canonical_1h_vols/`, gate R1, gitignored come tutto `models/`): `python scripts/vol/dev_vols_qlike.py --arch canonical_1h_vols` ne riproduce il numero, con il guard di identità dello scaler che deve stampare `IDENTICO`.

### 5.2 Walk-forward & backtest

**Walk-forward** purged k-fold con embargo anti-leakage (`scripts/02b_walkforward_validate.py`): l'embargo (`embargo_steps=168`, 1 settimana a 1h) è dimensionato ≥ `window_size+horizon` perché finestre e target si sovrappongono nel tempo — senza, il fold di test vede dati già visti in training. Meccanica dei fold in [THEORY.it.md](THEORY.it.md) §7bis. **Backtest direzionale** (`scripts/03_backtest.py`): fee model + slippage sqrt-impact, stress test (pessimistic e flash-crash), bootstrap CI 5000 iter, analisi per regime, recovery MDD. Le soglie trading in `config/default.yaml → backtest` sono in **spazio RAW** — non sovrascriverle da `arch/*.yaml` senza ricalibrare.

⚠ **Distribution shift val→test — è del TARGET, non della pipeline.** Sul filone **direzionale** le metriche in-sample (val_nll, Spearman/WHR walkforward) **anti-correlano** col backtest: non ottimizzare regole guidate da metriche in-sample. Sul target **`log_rv`** val→test sono invece **coerenti** (verificato dal PASS 2026-06-10) → i gate val-first della linea vol sono informativi. ⚠ Corollario metodologico: un lever va sempre giudicato contro una **baseline riaddestrata sullo stesso dataset/scaler**, mai contro l'incumbent production — confronti cross-scaler producono artefatti (§7).

### 5.3 Test

```bash
pytest tests/                          # suite completa
pytest tests/test_recent_fixes.py -v   # regression sui fix critici (z-score, RevIN, BLOCKER #1)
```

La suite è centrata sugli **invarianti che, se rotti, non fanno rumore**: no-leakage del `FeatureBuilder`, invarianti dello scaler, contratto `PipelineState`, parity live↔training, bit-parity del regime incrementale, golden della lista-104. Dopo ogni fix con impatto su shape/scaler/feature: aggiungi un regression test e ri-allinea i golden.

---

## 6. Deploy & inferenza

```bash
python run_all.py     # menu interattivo
```

→ **Tutti i comandi (pipeline per fase, training per arch, walk-forward, backtest, live, collector, routine di sessione, deploy VPS): [START.it.md](START.it.md).**

### 6.1 Catena d'inferenza direzionale

Catena: forward → `PipelineState.denormalize_predictions(μ, σ)` (z-score → raw) → conviction score (direzione × ampiezza × calibrazione × regime) → **Risk Manager** (Kelly frazionario ∝ edge ∝ 1/varianza, max 1%/trade; SL ATR 3×; trailing; circuit breaker DD 15% MtM intra-trade) → BUY/SELL/HOLD + size + SL + TP. Path live di produzione: `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canoniche, scaler da `PipelineState`) → `LiveEngine._deterministic_predict` (nucleo deterministico condiviso col backtest) → `denormalize_predictions` → `SignalGenerator`. Feed Binance WebSocket con reconnect exponential-backoff, persistenza stato, Volume Profile incrementale, funding refresh thread-safe. ⚠ Questo è il filone **legacy senza alpha OOS**: gira come negative-control, non come strategia.

⚠ Il path contiene una serie di **guard fail-fast deliberati** (cap su σ raw, validazione `forecast_horizon` e `interval` train-vs-inferenza, allineamento `merge_asof`, floor sullo stop, checkpoint atomici): sono lì per intercettare i bug di denormalizzazione e di contratto train↔inference, **non vanno rimossi**. Elenco puntuale in [THEORY.it.md](THEORY.it.md) §12.5.

### 6.2 Forward test vol & collector 24/7

Il braccio **short-vol** (`scripts/04b_vol_paper.py`) confronta la RV prevista dal modello con la IV implicita Deribit e apre straddle sul **testnet Deribit** quando l'edge supera la soglia pre-registrata, con leg di **delta-hedge** su perp. Gira come servizio **systemd 24/7 su VPS** (il PC di casa è passivo: lanciare `04b` in locale produrrebbe ordini doppi sulla stessa posizione testnet). Baseline di confronto always-long/always-short-vol in `04c_vol_paper_baselines.py`; attribuzione PnL ex-post delta/gamma/theta/vega in `scripts/vol/pnl_attribution.py`. **Gate v1 pre-registrato: FAIL 0/3 (2026-07-18)** — VRP positivo confermato, regola v1 non monetizzante. **Gate v2 delta-hedged: FAIL 2/3 (2026-08-11)** su n=20 settlement hedge-attivi — la riduzione di varianza c'è ed è ampia (`var(hedged)/var(unhedged) = 0.447`, cioè −55.3% contro il −40% richiesto), ma il drag medio vale −0.647·SE contro un budget di −0.25·SE ed è per il **76% fee di ribilanciamento** e per lo **0.9% funding**: l'hedge compra varianza e la paga, quindi la leg perp è disattivata e il design di produzione resta la v1 unhedged. Gli altri gate sono in accumulo di campione (stato e contatori in `STATUS.md`).

Tre **collector forward** girano in parallelo sullo stesso VPS e producono l'unico dato **non rigenerabile** del progetto: `01c_iv_poller.py` (chain opzioni Deribit + DVOL → `data/iv/`), `01d_orderbook_recorder.py` (order-book L2 Binance, microstruttura → `data/orderbook/`), `01e_trades_recorder.py` (trade opzioni Deribit per gli spread realizzati → `data/deribit_trades/`; la retention API è ~24h, quindi la raccolta è necessariamente forward). Kit di deploy in `deploy/vps/`, sync lato casa in `scripts/vps/`.

### 6.3 Dashboard — Deribit Options Risk Terminal

`scripts/06_dashboard.py` è un terminale di analytics per opzioni crypto: server HTTP single-file + SPA Plotly, **GPU-free e indipendente dalla pipeline ML**, alimentato dai dati **pubblici Deribit** (REST, no-auth). Calcola le Greche in tempo reale sull'intera option chain (Black-Scholes forward-measure, r=0) ed espone quattro viste: Volatility Surface, Option Chain, Risk & Greeks, e **Trades** (storico e posizione aperta del forward test `04b` — straddle e iron butterfly, PnL delle gambe opzioni più la gamba perp di hedge dove è stata tradata, posizioni scadute ancora senza settlement segnalate come tali). Avvio: `python run_all.py --only-dashboard` → `http://localhost:8050`. Dettaglio delle viste, endpoint e configurazione: [START.it.md](START.it.md) §5.4.

---

## 7. Esiti sperimentali

Ogni esperimento segue un protocollo pre-registrato (gate scritti PRIMA di girare, validazione val-first, lever come flag inerti di default) e **ogni esito negativo viene conservato**: i kill-record sono documentali, il "vaccino contro il re-test involontario". La sintesi di anni di gate è netta: la **linea vol** è l'unico PASS OOS del progetto (`log_rv` batte HAR-RV del 30% in QLIKE, val→test coerenti), mentre il **direzionale non ha alpha OOS a nessun timeframe testato** — a 1m il muro è il costo di transazione, a 1h il costo cade ma non emerge skill; né il gating per regime, né l'entry a soglia o a rango, né la ricalibrazione di σ producono PnL OOS. Prior trasversale che ne deriva: **i momenti pari (varianza, RV) generalizzano OOS, i dispari (segno, asimmetria firmata) no**.

**Dove leggere cosa:** [CHANGELOG.it.md](CHANGELOG.it.md) per i milestone in ordine cronologico · [STATUS.md](STATUS.md) per la fonte canonica (periodo corrente + tutti i gate aperti, con i numeri decisionali) · [docs/STATUS_ARCHIVE_2026H1.md](docs/STATUS_ARCHIVE_2026H1.md) per lo storico antecedente al 2026-07-08 (scorporo letterale, read-only) · [THEORY.it.md](THEORY.it.md) §12 per il protocollo sperimentale, il corpus KILL con i numeri e i flag inerti da non ri-testare.

---

## 8. Architettura del sistema

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

### 8.1 Struttura del progetto

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
│   │   ├── vol_metrics.py        QLIKE / inversione log-RV / baseline HAR-RV, HAR-C, HAR-CJ
│   │   │                         + Diebold-Mariano HAC e smearing di Duan (linea vol, condivisi)
│   │   ├── vol_forecaster.py     VolForecaster (nucleo forecast del vol-paper, promosso da 04b) + macro_snapshot (strumento vs stato: refit legacy o normalizer pinnato)
│   │   ├── regime_gate.py        build_regime_gate (gate causale: asof backward + staleness bound)
│   │   ├── cafn.py               CausalAttentionFlowNetwork (coordinatore causale, probe inerte)
│   │   └── revin.py              Reversible Instance Normalization (opzionale, use_revin)
│   ├── trading/                  Kelly sizing, SL dinamico, trailing, circuit breaker
│   │                             + greeks_risk.py (cap vega/delta, CB vega-loss, margin sim — non cablato al live)
│   └── utils/                    config loader, device setup, logging, PipelineState, atomic_save, stats
├── scripts/                      spine numerato (fase) + sottocartelle per linea — mappa: scripts/README.it.md
│   ├── 00_check_setup.py         verifica CUDA, dipendenze, connessione Binance
│   ├── 01_download_data.py       Binance → 104 feature → dataset npz  ·  01_update_data.py (delta; --candles-only = solo OHLCV)
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
├── README.it.md · START.it.md · THEORY.it.md · CHANGELOG.it.md · STATUS.md
├── docs/                         MODEL_IMPROVEMENTS · ROADMAP_VOL_BOOK · STATUS_ARCHIVE_2026H1 · paper/
├── data/                         generato (gitignored) — ⚠ data/iv, data/orderbook, data/deribit_trades NON rigenerabili
├── models/                       checkpoint per architettura (gitignored)
├── results/                      backtest, giudici e segnali live per architettura (gitignored)
└── logs/                         log rotanti (gitignored)
```

Vedi [START.it.md](START.it.md) per la guida operativa completa e [THEORY.it.md](THEORY.it.md) per i fondamenti teorici.

---

## Licenza

[MIT License](LICENSE) — codice di ricerca, **non consulenza finanziaria**.

**Disclaimer.** Questo repository è un progetto di ricerca personale. Nessuna delle linee esegue ordini con capitale reale: il braccio direzionale è **paper-only** e il braccio short-vol gira su **Deribit testnet** (fondi di carta). Le metriche pubblicate sono out-of-sample dove dichiarato e provengono da gate pre-registrati — inclusi i **fallimenti**, riportati con gli stessi numeri dei successi. Nulla qui è una raccomandazione d'investimento né un'aspettativa di rendimento; il trading di derivati crypto comporta rischio di perdita totale. I dati di mercato appartengono ai rispettivi venue (Binance, Deribit) e non sono ridistribuiti in questo repo.
