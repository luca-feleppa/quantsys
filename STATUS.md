# STATUS.md — Continuity Log

> Aggiornato alla fine di ogni ciclo di lavoro (direttiva permanente #3 di `CLAUDE.md`).
> Prima di iniziare qualsiasi task: **leggi questo file**. Memoria di lungo periodo dettagliata: `~/.claude/projects/E--quantsys-project/memory/`.

---

## 🕒 Ultimo aggiornamento: 2026-06-13 (2 poller Deribit rilanciati post-riavvio + 1° trade SETTLATO −0.0100 BTC + harness baseline del gate `04c_vol_paper_baselines.py` costruito e validato; ⚠ PC SPENTO dall'utente a fine sessione → i 2 processi VANNO RILANCIATI alla ripresa)

## 🔴 RIPRESA 2026-06-13 — RILANCIARE I 2 PROCESSI (PC spento dall'utente)

Il PC è stato spento: i 2 processi detached sono morti. **Primo gesto alla ripresa: rilanciarli** (comandi `Start-Process` in `AVVIO.md`, sezioni "Poller IV Deribit" e "Forward test vol-paper"):
```powershell
Start-Process -WindowStyle Hidden python -ArgumentList "scripts/01c_iv_poller.py"
Start-Process -WindowStyle Hidden python -ArgumentList "scripts/04b_vol_paper.py","--execute"
```
Poi verificare salute: `Get-Content logs/iv_poller.log,logs/vol_paper.log -Tail 5` + crescita `data/iv/atm_30h.parquet` e `results/vol_paper/forecasts.parquet`. ⚠ Il forward test ha un BUCO temporale per le ore di PC spento (il calendario forecasts salta quelle candele — atteso, il replay baseline lo gestisce).

## 🟢 HARNESS BASELINE DEL GATE — COSTRUITO E VALIDATO 2026-06-13 (`scripts/04c_vol_paper_baselines.py`, lavoro parallelo mentre il poller matura)

**Cosa fa:** calcola il **gate (2) pre-registrato** (NN batte ENTRAMBE always-long-vol e always-short-vol sullo stesso calendario di expiry → isola il timing dal variance risk premium medio). Metodo: **replay fedele** del loop di `04b` sul log `forecasts.parquet`, con premio dello straddle **ricostruito dai chain snapshot** (`data/iv/chain/*.parquet`, stessa selezione di `pick_straddle`: expiry≈30h, strike ATM, mark call+put) e **delivery price** dall'endpoint pubblico Deribit (cache `results/vol_paper/delivery_cache.json`). Semantica bit-identica a `04b` (max-1, hold-to-expiry, formula inverse-option). Gate (1) P&L medio>0 e (3) hit-rate>0.5 letti dai **trade REALI** in `trades.jsonl`; gate (2) dal replay ricostruito. Solo lettura, GPU-free, zero impatto sui processi. Output: `results/vol_paper/baseline_report.json`.

**Validazione chiave (smoke su 7 righe forecast / 1 trade, NON valutabile n<30):** **NN-ricostruito (−0.00993 BTC) ≈ NN-reale (−0.01001 BTC)** a meno di **8e-5 BTC** (mark-snapshot vs mark-live a 10 min) → la ricostruzione del premio dalle chain è apples-to-apples. Sul singolo trade NN≡LONG (edge +1.16), SHORT vince (+0.00873, vol-crush); scarto LONG↔SHORT = esatto `2·fee=0.0012 BTC`. Il report si scrive comunque con warning "non valutabile" finché i trade NN ricostruiti <30 (pre-reg): **l'harness è pronto, la valutazione del gate matura col tempo-calendario del poller.** Doc sincronizzate (README albero scripts, AVVIO sezione vol-paper). Working tree NON committato.

## ▶️ STATO 2026-06-13 (poller rilanciati in sessione poi PC spento, primo trade chiuso)

- **1° trade SETTLATO** (2026-06-13 15:18 UTC): era lo straddle LONG bootstrap `executed:false` (BTC-13JUN26-63500), delivery 63779.78 (mossa +0.68% ≪ breakeven) → **PnL −0.0100 BTC**. 1 trade su ≥30 del gate: regime di rumore atteso, nessuna conclusione. ⚠ era `executed:false` (calibrazione, non segnale reale); i prossimi saranno ordini testnet reali.
- **AZIONE ESATTA ALLA RIPRESA:** (1) **rilanciare i 2 processi** (sezione rossa sopra) + salute; (2) ri-girare `04c_vol_paper_baselines.py` quando i trade reali si accumulano (matura da solo); (3) NESSUN intervento sul test fino a ≥30 trade chiusi; (4) lavoro parallelo restante: paper "price+volume enough?" (figure+tabelle dai JSON, stesura §1-§4 — `docs/paper/OUTLINE.md`) e/o B1 order-book L2.

## 📝 VALUTAZIONE SWAP iTRANSFORMER-5seed → DISTILLATO 3-ARCH SU TARGET VOL (2026-06-12 sera — analisi, NESSUN esperimento avviato)

**Domanda dell'utente:** il codice permette di passare facilmente dal 5-seed iTransformer vol-1h a un modello distillato dalle 3 architetture? **Risposta: sì, strutturalmente — la pipeline distillation è target-agnostic (`02_train` consuma l'npz, il target vive in `01`/FeatureBuilder; scoring teacher su metriche da history.json valido anche su log_rv) — con 5 frizioni concrete:**
1. **Hardcoding consumer-side:** `04b_vol_paper.py` (3 refs) e `dev_vols_qlike.py` (2 refs) puntano letteralmente a `models/itransformer` → serve parametrizzazione (~3 righe ciascuno).
2. **`--distill` è n_ensemble=1 by design** (run_all.py, teacher e student 1 seed): il PASS vol è 5-seed → confronto impari senza estensione del path distill a n>1.
3. **Trappola membri stale (bug noto 2026-06-10):** run n=1 aggiorna solo `best_model.pt`, MAI i `best_model_{0..4}.pt`; `EnsembleModel.load` preferisce i numerati → con membri vecchi su disco il nuovo best viene IGNORATO silenziosamente. Serve guard.
4. **Prerequisiti:** npz NON esiste (cleanup 06-12, rigenerare ~10 min); nhits/tcnmamba mai addestrati su log_rv e i loro yaml hanno hyperparam era-1m (rischio overfit-epoca-1 su 58k finestre; il tuning vol — lr 3e-5/dropout 0.3 — è solo in itransformer.yaml).
5. **⚠ NON ora e NON in-place:** `models/itransformer/` è il modello del forward test in corso (un retrain lì lo corromperebbe al prossimo restart di 04b); regola GPU: no training + inference oraria in parallelo. Eventuale esperimento → dir separate + pre-registrazione (gate QLIKE vs 5-seed corrente, val-first), DOPO il gate dei 30 trade.

**✅ Fix di preparazione COMPLETATI (2026-06-12 sera, fan-out 3 subagent — tutti inerti by default, zero impatto su processi attivi e modelli su disco):**
- **Fix A (FATTO):** flag CLI `--arch` (default `itransformer`, choices 4 arch → comportamento invariato; scelto flag esplicito e NON env QUANTSYS_ARCH per evitare redirect silenziosi) in `04b_vol_paper.py` (3 refs parametrizzati via `self.model_dir`, dir loggata all'avvio) + `dev_vols_qlike.py` (2 refs). Il processo 04b attivo ha il vecchio codice in memoria — invariato fino al prossimo restart, che senza flag è bit-identico.
- **Fix B (FATTO):** helper `_stale_members_warning(base)` in `ensemble.py` (warning se `best_model.pt` più recente dei membri numerati di >60s — il load preferirebbe i membri stale) chiamato in `EnsembleModel.load` + pre-check warning in `02_train` (n_ensemble < membri numerati presenti, soglia ≥2 per evitare falsi positivi col fallback single). Verificato read-only su `models/itransformer/`: guard NON scatta (6 file coerenti). 4 regression test nuovi in `test_recent_fixes.py`.
- **Fix C (FATTO):** `phase_distill` usa `distill_n = args.n_ensemble if "--n-ensemble" in sys.argv else 1` (2 punti parametrizzati: training candidati Fase 2a + retrain student Fase 2c); senza flag esplicito i comandi subprocess sono byte-identici a prima.
- **Verifica:** suite full `pytest tests/` → **122 passed, 8 skipped, 0 failed** (skip pre-esistenti: npz/dataset assenti dal cleanup). Doc sincronizzate: `AVVIO.md` (Fase 2a distill, nota n_ensemble, sezione 04b col flag `--arch`). Working tree NON committato.

## 📋 PRE-REGISTRAZIONE BASELINE DIREZIONALI PER IL PAPER (scritta PRIMA di girare, 2026-06-12 sera)

**Scopo (paper "Are price and volume enough?"):** la claim "i momenti dispari sono impredicibili" è finora dimostrata solo per il NN. Per attribuirla all'INFORMAZIONE (e non alla classe di modello) servono le baseline econometriche direzionali sullo stesso perimetro. Aspettativa pre-dichiarata: **nessuna baseline mostra skill OOS** (|Spearman| < 2/√n, signDA ≈ 0.5) — esito atteso, ma va misurato; un esito contrario falsificherebbe la tesi del paper (e andrebbe riportato com'è).

**Design (`scripts/paper_01_dir_baselines.py`, stesso pattern dei giudici vol):**
- **Perimetro:** 1h, raw candles su disco (immutate dal run rs, 65.191), target raw `y = Σ log-ret prossime h=30 barre` (stessa formula dei giudici); split ricostruito ESATTAMENTE replicando il path di 01 (build 04b-wiring → canonico 104 → maschera finestre NaN su T=120 → temporal_split 0.8/0.1/0.1) con assert sui conteggi noti (51130/6391/6392) — se l'assert fallisce, si riporta lo scostamento.
- **Baseline:** (a) **OLS "HAR-mean"** `y ~ [1, r_h, r_7d→h, r_30d→h]` (analogo mean-equation dell'HAR, fit train-only); (b) **logit sul segno** stessi regressori; (c) **momentum persistence** `ŷ = r_h trailing`; (d) **train-mean costante** (null).
- **Metriche:** Spearman, sign-DA, MSE su val E test (riportati entrambi com'è: è conferma di claim negativa, non model selection). Report `results/paper/dir_baselines_1h_{val,test}.json`. NO backtest, NO iterazioni.

**✅ ESEGUITO 2026-06-12 sera → esito COME PRE-REGISTRATO (nessuna skill).** Tutte le baseline |ρ|≤0.048 (soglia 2/√n=0.025) e le poche nominalmente sopra **flippano segno val→test** (momentum −0.048→+0.016; OLS −0.013→+0.034; logit +0.025→−0.002); signDA ≈ base rate (0.52 val / 0.48 test). → L'instabilità val→test dei momenti dispari vale anche per i modelli lineari: **il limite è dell'informazione price/volume, non della classe di modello** — il tassello che mancava al paper. Split ricostruito ESATTO sul raw corrente: (51156/6394/6395); i numeri citati prima (51130/...) erano del probe 06-10 con 32 candele in meno — spiegazione verificata, documentata nello script.

**📄 MATERIALI PAPER CREATI (2026-06-12):** `docs/paper/RESULTS_MAP.md` (inventario claim→artefatto→numeri, con note di provenienza per i risultati i cui artefatti sono stati sovrascritti) + `docs/paper/OUTLINE.md` (titolo, tesi, struttura §1-§8+appendici, venue JFDS/arXiv, TODO per il draft: 2 figure + tabelle LaTeX da JSON + stesura §1-§4).

## 🟢 FORWARD TEST VOL-PAPER — AVVIATO 2026-06-12 ~15:30 (harness `04b_vol_paper.py` --execute, detached)

**Stato operativo a fine sessione 2026-06-12 — 2 processi PERSISTENTI attivi (⚠ NON sono servizi: dopo un riavvio vanno rilanciati, comandi Start-Process in AVVIO.md):**
| Processo | PID (2026-06-12) | Cadenza | Log | Output |
|---|---|---|---|---|
| `01c_iv_poller.py` | 12740 | 10 min | `logs/iv_poller.log` | `data/iv/` |
| `04b_vol_paper.py --execute` | 3924 | orario hh:00+90s | `logs/vol_paper.log` | `results/vol_paper/` |

**Primo segnale (candela 2026-06-12 12:00 UTC):** NN-RV=1.090e-3 (≈56% ann) vs var_iv=3.4e-4 (IV 31% ann) → **edge +1.16 → LONG straddle BTC-13JUN26-63500** (fill simulato al mark — premio 0.0138 BTC, `executed:false`, aperto dallo smoke pre-lancio; i successivi saranno ordini reali testnet). Sanity: rv_pred tra trailing-30h (7.7e-4) e trailing-7d (1.29e-3) → il NN è un blend HAR-like; il mercato prezza il vol-crush del weekend a <½ della realized. Settlement: 2026-06-13 08:00 UTC. NB il primo tick del processo --execute dà HOLD (posizione aperta blocca, max-1 by design).

**Dettagli implementativi chiave di `04b` (per debugging futuro):** canonico 104 derivato a runtime replicando i filtri di `01_download_data` su `ps.feature_cols` (exclude→C-funding→NaN50%→Inf) e validato vs `config.json` del modello — l'npz NON serve; macro 90 dal parquet + refit identico `MacroNormalizer` (lo state vol non le persiste; ultima data 2026-06-10, warn se >7g); inversione completa `μ·1.438−7.175`; candele in-memory (bootstrap parquet + delta REST 48, tz-aware UTC, scarta candela non chiusa); assert anti-mainnet su URL deribit.

**▶️ AZIONE ESATTA ALLA RIPRESA:** (1) salute dei 2 processi (`Get-Content logs/{iv_poller,vol_paper}.log -Tail 5`; crescita `data/iv/atm_30h.parquet` ~144 righe/g e `results/vol_paper/forecasts.parquet` ~24 righe/g; se morti → Start-Process da AVVIO.md); (2) controllare il settlement del primo trade in `results/vol_paper/trades.jsonl` (post 2026-06-13 08:00 UTC); (3) NESSUN intervento sul test fino a ≥30 trade chiusi (gate pre-registrato sotto) — eventuale script di analisi baseline (always-long/short dal forecasts log + delivery prices) si può scrivere quando ci sono i primi trade; (4) lavoro parallelo sensato: paper "price+volume enough?" (effort S) e/o B1 order-book L2.

**🔒 DECISIONE 2026-06-12 (con l'utente): fase di validazione DERIBIT-ONLY.** Razionale: (a) test chiuso su una sola superficie (IV mark + esecuzione + delivery price Deribit → zero basis cross-venue); (b) matching segnale↔strumento nativo (24/7, dailies 08:00 UTC ≈ tenor 30h) vs i 3 confound IBIT (RTH ~20% delle ore, scadenze Mon/Wed/Fri, tracking error ETF). **Re-check IV/greeks Alpaca ESEGUITO in RTH (quote fresche): IV/greeks = None ANCHE a mercato aperto** → sul piano free il feed indicative NON espone mai IV/greeks (la ricognizione 2026-06-11 era errata su questo punto; servirebbe OPRA a pagamento). Alpaca resta DORMIENTE come unico ponte regolamentato per capitale reale (se il gate passa: strategia adattata a RTH/weekly su IBIT, greeks calcolati in proprio da bid/ask+spot); zero lavoro allocato.

## 📋 PRE-REGISTRAZIONE FORWARD TEST VOL-PAPER (scritta PRIMA di girare, 2026-06-12)

**Domanda:** il forecast NN-RV 1h (modello PASS, `models/itransformer` = restore backup_1h_vols) contiene informazione economica OLTRE la IV implicita del mercato opzioni? Test FORWARD (unbiased by construction) su testnet Deribit: straddle ATM su daily ~30h, entry sul divario NN-RV vs forward variance implicita.

**Segnale (ogni ora, a candela 1h chiusa, script `04b_vol_paper.py`):**
- `RV_pred = exp(μ_z·s + c)` — inversione COMPLETA dallo scaler persistito (pattern del giudice QLIKE); feature dal path parity-blessed (FeatureBuilder fit=False su storico full + scaler da PipelineState; macro via MacroSnapshotUpdater, fallback zeros).
- `var_iv = (iv_30h/100)² · 30/8760` dall'ultima riga di `data/iv/atm_30h.parquet` (staleness ≤ 30 min, altrimenti NO-TRADE).
- `edge = log(RV_pred / var_iv)`.

**Regola pre-registrata (simmetrica, NESSUN tuning a risultati visti):**
- `edge > +0.25` → LONG straddle ATM (buy C+P) sull'expiry daily più vicina a 30h; `edge < −0.25` → SHORT straddle; altrimenti flat.
- Max 1 posizione aperta; size 1.0 contratto/leg; hold a SCADENZA (cash settlement al delivery price — P&L deterministico, zero rumore di exit); fee taker opzioni 0.0003 BTC/contratto cap 12.5% premium, contate per leg.
- Ogni tick logga `(ts, μ_z, log_rv, rv_pred, iv_30h, var_iv, edge, azione)` su `results/vol_paper/forecasts.parquet` ANCHE quando flat: serve a calcolare le baseline sull'intero calendario.

**GATE (valutazione a ≥30 trade chiusi, non prima):** (1) P&L medio/trade > 0 al netto fee; (2) P&L totale > ENTRAMBE le baseline always-long-vol e always-short-vol sullo stesso calendario di expiry (isola il timing del NN dal variance risk premium medio); (3) hit-rate > 0.5. Esito negativo = riportato com'è; qualsiasi modifica a soglie/sizing = NUOVA pre-registrazione.

**Caveat dichiarati:** liquidità testnet simulata (fill market-order poco realistici — il test valida il SEGNALE, non lo slippage); IV mark Deribit (no bid/ask spread della vol); il lato short ha rischio illimitato (accettabile solo in paper).

## 🧹 RIORIENTAMENTO VOL-1H + CLEANUP DISCO (2026-06-12, deciso con l'utente)

**Decisione:** il progetto si orienta sulla linea vol-1h (unico segnale PASS); lo stato disco ora la riflette.
- **Restore production:** `models/itransformer/` = vol-1h PASS da `backup_1h_vols` (5 membri + best, state verificato: interval=1h, target_scale=1.4376, h=30); copia canonica `models/pipeline_state.pkl` allineata; config `target_type: log_rs_ratio → log_rv`. I modelli rs-ratio FAIL eliminati.
- **Eliminati (~4,7 GB, tutti rigenerabili o filoni morti):** `data/lstm_dataset.npz` (3,07 GB, era vol-1m FAIL) + `features.parquet` (65 MB); `data/xs/` (433 MB, probe KILL — resta `results/xs/ic_report.json`); `models/{nhits,tcnmamba}` (535 MB, duplicati byte-identici dei backup 1m); `models/backup_1m/` (611 MB, direzionale-1m morto — riaddestrabile da `data/backup_1m/` che RESTA, 36 MB). ⚠ Conseguenze: **il dataset npz NON esiste** (prima di train/judge: `01_download_data.py` + `dev_vols_macro_append.py`, ~10 min); **rollback 1m = restore data + RETRAIN** (checkpoint 1m non esistono più).
- **Script:** `xs_01/02/03` + `dev_step0_regime_sigma.py` → `scripts/archive/` (git mv). I `dev_vols_*` restano in `scripts/` (linea attiva).
- **Tenuti:** `models/backup_1h_vols/` (asset primario), `data/iv/` (non rigenerabile), `data/backup_1m/`, `models/backup_1m_vols/` (15 MB, record FAIL), `models/lstm/` (5,7 MB).
- Doc sincronizzate: CLAUDE.md (STATO NOTO + rollback), AVVIO.md (file layout), README.md (albero scripts).

## 🟢 POLLER IV DERIBIT — IMPLEMENTATO, SMOKE OK, IN ESECUZIONE (2026-06-12)

**Fatto (punti 1 e 3 della checklist 2026-06-11, in ~40 min):**
- **`scripts/01c_iv_poller.py`** (numerazione famiglia dati): loop 5-15 min (default 10), 2 req pubbliche Deribit/tick — `get_book_summary_by_currency` (mark_iv di tutta la chain opzioni BTC, ~950 strumenti) + `get_volatility_index_data` (ultimo DVOL). NESSUN account richiesto (il vincolo "solo Alpaca free" dell'utente è irrilevante qui). Output append-only ATOMICO (tmp+os.replace, dedup su chiave) in `data/iv/`:
  - `chain/btc_options_YYYYMMDD.parquet` — snapshot raw per-strumento (1 file/giorno; tutta la chain, è il dataset che Tardis vende ≥$300);
  - `atm_30h.parquet` — 1 riga/tick: ATM IV (straddle, strike più vicino al forward = mediana underlying_price per-expiry) delle 4 expiry vive più vicine + **IV interpolata in varianza totale w=σ²·T a tenor costante 30h** (= forecast_horizon del modello vol 1h);
  - `dvol.parquet` — serie DVOL (controllo tenor-30d).
- **Smoke `--once` PASSATO** al primo colpo: tick 2026-06-12 11:10 UTC → iv_30h=31.40% (interpolato correttamente tra exp0 35.27%@20.8h e exp1 28.15%@44.8h), dvol=42.14, 950 strumenti, 12 expiry vive, dailies a 08:00 UTC confermate.
- **Backfill DVOL storico ESEGUITO** (`--backfill-dvol`): 45.755 righe orarie 2021-03-24→oggi in 46 chiamate/29s, serie quasi-continua (atteso ~45.7k). → `data/iv/dvol.parquet`.
- **Poller AVVIATO detached** (PID 12740, 2026-06-12 13:12 locale, cadenza 10 min, log `logs/iv_poller.log`). ⚠ NON è un servizio: dopo un riavvio va rilanciato — comando `Start-Process` documentato in `AVVIO.md` (sezione "Poller IV Deribit"). Verificare periodicamente che `data/iv/atm_30h.parquet` cresca (~144 righe/giorno).
- Doc sincronizzate: `AVVIO.md` (sezione operativa nuova + file layout), `README.md` (albero scripts).

**✅ Punto 2 (smoke chain IBIT Alpaca) — ESEGUITO 2026-06-12, PASS.** Key paper fornite dall'utente (rigenerate dalla dashboard, in `config/secrets.yaml` blocco `alpaca:`, gitignored — `load_config` le merge-a). Risultati: account **ACTIVE, options_trading_level 3** (multi-leg → long straddle/strangle ✅), equity paper $100k, **chain IBIT visibile e tradable** con dailies 0-DTE (expiry oggi stesso = Mon/Wed/Fri confermate). Perimetro di esecuzione paper CONFERMATO. ⚠ 2 trappole trovate: (1) l'`endpoint` della dashboard include già `/v2` → normalizzare (`re.sub(r'/v2/?$','',ep)`) prima di concatenare i path; (2) feed dati `indicative` (free): campi `impliedVolatility`/`greeks` = **None fuori RTH** (smoke girato alle 7:35 ET, quote = close di ieri 19:59 ET) — **da ri-verificare a mercato aperto (15:30-22:00 CET)**; `feed=opra` → 403 "agreement not signed" (atteso sul free).

**✅ EXECUTION LAYER 24/7 — TESTNET DERIBIT VERIFICATO (2026-06-12, dopo il cleanup).** L'utente ha creato l'account su test.deribit.com; key in `secrets.yaml` blocco `deribit_testnet:` (`client_id`/`client_secret`/`endpoint`, auth OAuth2 via `public/auth` grant_type=client_credentials → Bearer token). Smoke PASS: scope `trade:read_write`, **saldo paper 100 BTC** (⚠ irrealistico: dimensionare le strategie su una frazione), **1.076 opzioni BTC attive con dailies OGNI giorno a 08:00 UTC** → tenor ~30h disponibile nativamente, 24/7 (vs IBIT: solo RTH + Mon/Wed/Fri). Testnet = venue primaria per il paper della monetizzazione vol; Alpaca/IBIT = venue regolamentata di riserva. ⚠ Liquidità testnet simulata: valida la logica, non lo slippage.

**▶️ AZIONE ESATTA ALLA RIPRESA:** (1) check salute poller (`Get-Content logs/iv_poller.log -Tail 5` + crescita `atm_30h.parquet` ~144 righe/giorno; se morto → rilancio con Start-Process da AVVIO.md); (2) **re-check IV/greeks Alpaca in RTH** (15:30-22:00 CET: snapshot ATM IBIT, atteso popolato — 1 call); (3) **prima del primo trade paper serve il dataset rigenerato** (`01_download_data.py` + `dev_vols_macro_append.py`, ~10 min — npz eliminato col cleanup) e poi il design della strategia vol (es. straddle 30h quando NN-RV ≫ IV: gate da pre-registrare quando lo storico IV del poller è sufficiente); (4) filoni paralleli: B1 order-book L2, paper "price+volume enough?". Il gate NN-RV vs IV matura col tempo-calendario del poller. NB: la sorgente IV per il gate SCIENTIFICO resta il poller Deribit pubblico; testnet/Alpaca sono solo esecuzione.

## 🕒 2026-06-11 sera (PROBE SEMIVARIANZA eseguito → **FAIL su test**, filone HD-firmato CHIUSO)

## ⚫ PROBE SEMIVARIANZA 1h — ESEGUITO → **FAIL** (primario E secondario, su test; chiuso senza appello come pre-registrato)

**Run (2026-06-11, ~30 min totali):** rebuild 1h con restore `backup_1h_vols/{raw_candles,regime_probs}` (evitate ~3h di walk-forward Markov — NB: la stima "3 min" per `01b` vale solo sul span 381gg; su 7 anni è ~3h), dataset 65.191 candele → train iTrans 5-seed (~25 min, best val_nll 0.1925) → giudice `dev_vols_rs_judge.py`. Sanity val-first `MSE_NN ≤ MSE_HAR-RS` superata per un soffio (ratio 0.9959) → test valutato UNA volta:

| Modello | MSE val | MSE test | ρ test | signDA test |
|---|---|---|---|---|
| NN (iTrans 5-seed) | 0.82489 | **0.99366** | **−0.038** | **0.459** |
| HAR-RS (Patton–Sheppard OLS) | 0.82829 | 0.99843 | −0.053 | 0.465 |
| Naive (lratio trailing 30h) | 1.64712 | 1.89025 | +0.048 | 0.525 |
| Train-mean (costante) | 0.83036 | 0.99387 | 0 | 0.472 |

- **Gate primario:** NN/HAR-RS = 0.9952 > 0.95 ✗ (e batte la train-mean di appena 0.02%). **Gate secondario:** signDA 0.459 < 0.55 ✗ (sotto il caso!). **FAIL/FAIL.**
- **Lettura scientifica (il risultato vero del probe):** l'asimmetria RS⁺/RS⁻ futura è **impredicibile per TUTTI** — su test HAR-RS fa *peggio* della costante (0.99843 vs 0.99387) e il NN la pareggia. In più il pattern val→test si ripete (ρ +0.078 → −0.038): **gli oggetti a momento DISPARI (direzione, signed jump variation) non generalizzano OOS; il momento PARI (livello RV) sì** (−30% QLIKE). La dicotomia pari/dispari è la sintesi più netta dell'intero progetto — e rafforza la tesi del paper "price+volume enough?": l'informazione in candele price/volume riguarda solo i momenti pari.
- **Filone semivarianza/HD-firmato CHIUSO senza appello** (pre-registrazione: zero iterazioni). Il ramo SVAR/historical-decomposition con identificazione via regimi eredita lo stesso verdetto: era la stessa scommessa con più ipotesi.
- **Stato disco post-probe:** `models/itransformer/` = modelli **rs-ratio 1h (FAIL)**; vol-1h PASS in `models/backup_1h_vols/`; vol-1m FAIL in `models/backup_1m_vols/`; direzionale 1m in `models/backup_1m/`+`data/backup_1m/`. Config `target_type: log_rs_ratio`. Report: `results/vols/rs_report_1h_{val,test}.json`; log `logs/rs_{01,macro_append,02_train,judge_val,judge_test}.log`.

**▶️ AZIONE ESATTA ALLA RIPRESA (checklist in ordine di priorità):**
1. **Poller IV Deribit (S, sblocca il tempo-calendario):** scrivere `scripts/iv_poller.py` (o numerazione coerente) — loop 5-15 min, 2 req non-auth (`get_book_summary_by_currency` BTC options + ultimo DVOL), append parquet `data/iv/`, ATM IV 3-4 expiry vicine + interpolazione tenor 30h. Dettagli verificati nella sezione RICOGNIZIONE sopra. Prima parte, prima si accumula lo storico per il gate **NN-RV vs IV**.
2. **Smoke chain IBIT su Alpaca (S, 1 call autenticata):** `GET /v2/options/contracts?underlying_symbols=IBIT` con le key dell'utente → conferma perimetro di esecuzione paper (long straddle/strangle mleg L3).
3. **Backfill DVOL storico (S):** ~46 call `get_volatility_index_data` orario 2021→oggi → serie di controllo 30d.
4. Poi: **B1 order-book L2** (alpha direzionale, progetto-dati a sé) e/o **paper** "Are price and volume enough?" (memoria `paper-idea-price-volume-enough`; la dicotomia momenti pari/dispari di oggi è il risultato centrale; mancano 1-2 baseline econometriche direzionali, effort S).

Per tornare a un setting operativo qualsiasi: restore modelli dal backup appropriato (`backup_1h_vols` = vol-1h PASS, `backup_1m` = direzionale 1m) + `target_type`/`interval` coerenti in config — il guard interval/horizon config↔state blocca i mix incoerenti.

## 📡 RICOGNIZIONE ALPACA + DERIBIT (2026-06-11, subagent web, fatti verificati con chiamate dirette)

**Contesto:** l'utente ha un account **Alpaca** con API key derivati (memoria `user-alpaca-account`). Obiettivo: monetizzare/testare l'edge vol 1h. Il competitor reale del forecast NN-RV non è HAR-RV ma la **IV implicita del mercato opzioni** → gate economico futuro: NN-RV batte la forward variance implicita a tenor ~30h come predittore della RV realizzata?

- **Alpaca — opzioni IBIT:** IBIT ha scadenze **Mon/Wed/Fri dal 2026-02-02** → 0–2 DTE quasi sempre disponibili. IV+greeks inclusi nel **piano gratuito** (feed indicative real-time, trade delayed 15min, 200 call/min). Paper trading opzioni attivo di default, **multi-leg Level 3** (long straddle/strangle ✅; vendita vol nuda ❌ — solo rischio-definito: iron condor/butterfly). **Vincolo duro: RTH 9:30–16:00 ET** vs segnale 24/7 (~80% delle ore non hedgeabili su IBIT; mitigazione: spot crypto Alpaca 24/7 o perps). **Smoke da fare (1 call autenticata):** `GET /v2/options/contracts?underlying_symbols=IBIT` per confermare la chain sull'account.
- **Alpaca — crypto perps Beta non-US:** endpoint dati verificato live (`data.alpaca.markets/v1beta1/crypto-perps/global/latest/trades?symbols=BTCUSDT.P`, no auth); trading via entità AlpacaX (Bahamas), Beta — verificare eligibility dell'account.
- **Deribit (sorgente IV per il test scientifico):** dailies BTC 0–2 DTE sempre listate (978 strumenti attivi); **`public/get_book_summary_by_currency?currency=BTC&kind=option` restituisce mark_iv di TUTTI gli strumenti in 1 req non autenticata** (rate limit 10 req/s — un poller a 5-15 min è 4 ordini di grandezza sotto). DVOL storico orario gratis dal 2021-03-24 (`public/get_volatility_index_data`, 1000 punti/call, tenor 30d = solo variabile di controllo). ⚠ **Lo storico IV short-tenor NON è gratis** (Tardis ≥$300, sample gratuito il 1° del mese) → **la raccolta forward è il collo di bottiglia temporale: prima parte, prima si accumula il dataset per il gate RV-vs-IV.**
- **Architettura poller minima (da costruire):** processo schedulato 5–15 min, 2 req Deribit (book summary opzioni BTC + ultimo punto DVOL), append su parquet, estrazione ATM IV delle 3-4 expiry più vicine + interpolazione a tenor costante 30h.

## 📋 PRE-REGISTRAZIONE PROBE SEMIVARIANZA 1h (scritta PRIMA di girare, 2026-06-11)

**Domanda:** la pipeline NN (identica al run vol-S 1h PASS) predice il **segno della varianza** — cioè l'asimmetria upside/downside della realized semivariance futura (Barndorff-Nielsen–Kinnebrock–Shephard 2010; Patton–Sheppard 2015 "good/bad volatility") — meglio delle baseline econometriche? È la traduzione econometrica dell'idea historical-decomposition: attribuire la varianza a shock firmati. NON è il direzionale travestito: il target è un momento di vol (famiglia che ha generalizzato val→test), non la media.

**Design (tutto identico al run vol-S 1h tranne il target):**
- **Dati:** 1h, 2019-01-01→oggi (restore da `models/backup_1h_vols/{raw_candles_1h,regime_probs_1h}.parquet`), stesso split, `window_stride: 1`.
- **Target:** `features.target_type: log_rs_ratio` → `target_ret = log((RS⁺_fwd+ε)/(RS⁻_fwd+ε))`, ε=1e-12, con `RS±_fwd = Σᵢ₌₁..ₕ r²ₜ₊ᵢ·1[rₜ₊ᵢ≷0]`, h=30 barre 1h; `target_dir = 1[RS⁺_fwd > RS⁻_fwd]` (causale).
- **Modello:** iTrans 5-seed, hyperparam INVARIATI (lr 3e-5, dropout 0.3, drop_path 0.2, wd 3e-3). Zero tuning.
- **Baseline (giudice `scripts/dev_vols_rs_judge.py`, OLS train-only):** (a) **HAR-RS** stile Patton–Sheppard: regressori `[1, lratio_h, lratio_7d, lratio_30d, log_rv_h]` trailing; (b) **naive persistence** = lratio trailing h; (c) **train-mean** (null di non-informatività del segno).
- **Metrica primaria: MSE sul log-ratio** (il QLIKE non si applica: il log-ratio non è positivo-definito). Secondarie descrittive: Spearman, sign-DA su `target_dir`.
- **Inversione z→raw:** completa, `μ·IQR + centro` dal RobustScaler persistito (lezione log_rv); sanity sul centro: |c| < 2 (il log-ratio è quasi-centrato, ≠ log-RV c≈−7.2).
- **Protocollo:** val-first (sanity: MSE_NN ≤ MSE_HAR-RS su val), test toccato UNA volta. **Zero iterazioni** in caso di fallimento. NO backtest trading.

**GATE PRIMARIO (test 1h):** `MSE_NN ≤ 0.95·MSE_HAR-RS` **E** `MSE_NN < MSE_naive` **E** `MSE_NN < MSE_train-mean`.
**GATE SECONDARIO (economico, per promuovere il follow-up risk-reversal):** `sign-DA_NN ≥ 0.55 su test` **E** `sign-DA_NN > sign-DA_HAR-RS`. Primario PASS + secondario FAIL ⇒ "asimmetria predicibile in magnitudine ma segno non tradabile": riportato com'è, nessun follow-up di esecuzione.
**FAIL primario ⇒ filone semivarianza CHIUSO** (e con esso il ramo HD-firmato), senza appello.

## 🕒 2026-06-11 (registrazione esito verifica 1m, eseguita ieri notte)

## 🔴 VERIFICA log_rv su 1m — ESEGUITA 2026-06-10 notte → **FAIL sanity val** (test 1m MAI toccato, niente iterazioni, come pre-registrato)

**Run completo end-to-end** (log: `vols1m_01.log` 20:29 → `vols1m_01b.log` 20:36 → `vols1m_02_train.log` 21:43 → `vols1m_qlike_val.log` 21:44): rebuild 1m 381gg con overlay `config/interval/1m.yaml` (dataset 4.9 GB), train iTrans 5-seed ~67 min (best val_nll 0.116), giudice QLIKE su val.

| Modello | QLIKE val 1m | MSE(log) val |
|---|---|---|
| NN (iTrans 5-seed) | 0.42678 | 0.4707 |
| **HAR-RV (OLS train-only)** | **0.42143** | 0.5619 |
| Naive (RV trailing 30m) | 0.52315 | 0.6603 |

- **NN/HAR ratio 1.0127 > 0.95 ⇒ sanity val-first FALLITA** (batte la naive, NON batte HAR — nemmeno la pareggia). Per protocollo il test 1m NON è stato valutato; per pre-registrazione: zero iterazioni di tuning.
- **Verdetto pre-registrato applicato: il risultato vol-S 1h è SPECIFICO della risoluzione.** A h=30 barre 1m (RV a 30 min) HAR-RV è già un compressore adeguato e il NN non aggiunge nulla; il vantaggio NN (−30% QLIKE) esiste solo a RV oraria (h=30h). Riportato com'è.
- ⚠ Le metriche "alte" nel train log 1m (Spearman +0.84, DA 86%, ICIR +7.7) NON contraddicono il FAIL: il target vol è fortemente autocorrelato e anche HAR le cattura — il giudice è il QLIKE *relativo* alla baseline, non la metrica assoluta.
- **Stato disco post-run:** `models/itransformer/` = modelli VOL **1m** (quelli del FAIL); vol **1h** (il PASS) preservati in `models/backup_1h_vols/`; direzionale 1m intatto in `models/backup_1m/` + `data/backup_1m/`. PipelineState arch-locale **e** canonico = `1m`, target_scale 1.5836 (coerente log-RV 1m). `data/lstm_dataset.npz` = dataset vol 1m. `config/default.yaml` resta `interval: 1h` + `target_type: log_rv` (il run 1m è passato dall'overlay interval). Artefatto: `results/vols/qlike_report_1m_val.json`.

**▶️ AZIONE ESATTA ALLA RIPRESA (aggiornata 2026-06-11):** la verifica cross-risoluzione chiude il capitolo vol-S con perimetro netto: **edge vol = solo 1h**. Restano le 3 strade di follow-up (invariate rispetto a sotto, ma con questo vincolo): (a) jump/no-trade gate difensivo col modello vol **1h** (`models/backup_1h_vols/`) — valore limitato finché il direzionale non ha alpha; (b) **B1 order-book L2** — unico filone alpha direzionale residuo; (c) studio strumenti vol-tradabili (Deribit options/variance) — ora più motivato: è l'unico segnale validato OOS del progetto, ma vive a risoluzione oraria. Inoltre: working tree con ~22 file modificati + 2 script dev non committati (pivot interval-agnostic + vol-S, coerenti e committabili — rassegna 2026-06-11); decidere commit e stato config (disco=1m-vol, default.yaml=1h).

## 📋 PRE-REGISTRAZIONE VERIFICA log_rv su 1m (scritta PRIMA di girare, 2026-06-10 notte)

**Domanda:** la predicibilità della vol sopra HAR-RV replica a risoluzione 1m (h=30 barre = RV a 30 min, dataset 381gg backup)? Test di solidità del risultato vol-S, NON tuning.
**Design: TUTTO identico al run 1h tranne l'interval** — stessi hyperparam (lr 3e-5, dropout 0.3, drop_path 0.2, wd 3e-3, 5 seed), stesso giudice (QLIKE su RV livelli, HAR-RV OLS train-only con componenti trailing h/7d/30d riscalate, naive trailing-h), stesso protocollo val-first → test UNA volta.
**GATE (test 1m):** `QLIKE_NN ≤ 0.95·QLIKE_HAR` E `QLIKE_NN < QLIKE_naive`. PASS ⇒ il segnale vol è robusto cross-risoluzione. FAIL ⇒ il risultato 1h va trattato come specifico della risoluzione (riportato com'è, niente iterazioni).

## 🟢 VOL-S ESEGUITO → **PASS su val E test** (primo gate pre-registrato superato nella storia del progetto)

**Risultato (giudice QLIKE su RV livelli, h=30 barre 1h, exp(log_pred) per tutti):**

| Modello | QLIKE val | QLIKE test | MSE(log) test |
|---|---|---|---|
| **NN (iTrans 5-seed)** | **0.27784** | **0.25716** | 0.6327 |
| HAR-RV (Corsi, OLS train-only) | 0.37326 | 0.36811 | 0.7144 |
| Naive (RV trailing 30h) | 0.73362 | 0.80673 | 1.1530 |

- **NN/HAR ratio: 0.744 (val) → 0.699 (test)** — gate ≤0.95 superato con margine 6× (−30% QLIKE su test); naive battuta ovunque. **val→test COERENTI** (nessuna anti-correlazione: il segnale vol generalizza, a differenza della direzione).
- Metriche modello su test: Spearman **+0.4532** (p≈0), DA 71.3%, IC +0.377, **ICIR +3.56** su 5 sotto-periodi (stabilissimo), coverage 95.2%.
- Artefatti: `results/vols/qlike_report_{val,test}.json`; log `logs/vols_02_train.log`; modello vol in `models/itransformer/` (⚠ è il modello VOL, non direzionale — NO backtest trading su questo).

**Implementazione (tutta reversibile):** `features.target_type: log_rv` in config (default codice `ret` = path direzionale bit-invariato; ValueError su valori ignoti) → `FeatureBuilder._returns` target `log(Σr²+1e-12)` con `target_dir`=vol-up/down causale; `scripts/dev_vols_macro_append.py` (ri-appende X_macro senza rifare il walk-forward regime, ~5s vs 3h); `scripts/dev_vols_qlike.py` (giudice: HAR-RV OLS chiuso fit su train, naive, NN con inversione z→raw **centro+scala** — NB `denormalize_predictions` da sola è SBAGLIATA per log-RV, mediana ≈ −7.2, serve `μ·IQR + centro`).

**Significato:** B2 (pivot volatilità) **chiusa POSITIVA**: la pipeline NN estrae segnale vol genuino sopra la baseline econometrica seria. La vol resta NON tradabile sul perimetro spot/perp attuale (nessuno strumento di varianza) — il valore immediato è il **follow-up pre-dichiarato: jump/no-trade gate difensivo** (usare la predizione vol per filtrare/ridurre l'esposizione del modello direzionale nei picchi previsti); il valore strategico è l'opzione futura Deribit/varianza (progetto a sé, > B1).

**▶️ AZIONE ESATTA ALLA RIPRESA:** decidere il follow-up tra: (a) **jump/no-trade gate** (S: usare il modello vol per gate difensivo del direzionale — ma il direzionale è negativo OOS, quindi valore limitato finché non c'è alpha); (b) **B1 order-book L2** (l'asse informazione-nuova per l'alpha direzionale, ora unico filone alpha residuo); (c) studio strumenti vol-tradabili (Deribit options/variance — progetto dati+esecuzione, effort L). Lo stato config è VOL-S (target_type: log_rv, 1h): per tornare al direzionale 1m → rollback completo (backup intatti); per il direzionale 1h → `target_type: ret` + rebuild + retrain.

## 🕒 2026-06-10 sera (PIVOT 1h — **KILL DEFINITIVO** dopo iterazione tuning pre-registrata)

## 📋 PRE-REGISTRAZIONE ESPERIMENTO VOL-S (scritta PRIMA di girare, 2026-06-10 sera) — chiude B2

**Domanda:** il NN (stessa pipeline, target sostituito) batte le baseline econometriche nel prevedere la realized volatility? NON è alpha (vol non tradabile sul perimetro spot/perp); valore = chiusura scientifica B2 + eventuale jump/no-trade gate difensivo.

**Setup (sui dati 1h già su disco — motivo per cui NON si fa rollback ora):**
- **Target:** `log-RV = log(Σᵢ₌₁..ₕ r²ₜ₊ᵢ + 1e-12)`, h=30 barre 1h. Config `features.target_type: log_rv` (default codice `ret` → path direzionale bit-invariato). `target_dir` ridefinito = RV futura > RV trailing 30h (vol-up/down, causale).
- **Modello:** iTrans 5-seed, hyperparam tuned correnti (lr 3e-5, dropout 0.3, drop_path 0.2, wd 3e-3) — INVARIATI, zero tuning aggiuntivo.
- **Baseline:** (a) **HAR-RV** (Corsi 2009) su log-RV: OLS su train con componenti trailing 30h/7d/30d; (b) **naive persistence** (RV trailing 30h) come floor di sanità.
- **Giudice primario: QLIKE** su RV in livelli — `mean(RV_true/RV_pred − ln(RV_true/RV_pred) − 1)` — con `RV_pred = exp(log_pred)` per TUTTI i modelli (stessa trasformazione, niente correzioni di Jensen asimmetriche). Secondario: MSE su log-RV.
- **Protocollo:** val-first (sanity: NN ≤ HAR su val), poi UNA valutazione su test. **NO backtest trading su questo modello.**
- **GATE:** su test, `QLIKE_NN ≤ 0.95 × QLIKE_HAR` (≥5% di miglioramento) **E** `QLIKE_NN < QLIKE_naive`. Successo ⇒ B2 chiusa positiva → follow-up jump/no-trade gate. Fallimento ⇒ **B2 chiusa negativa definitiva** (la letteratura dice che battere HAR-RV è difficile: esito negativo = informativo, non fallimento del metodo).

## ⚫ VERDETTO FINALE PIVOT 1h: **KILL** (2ª iterazione fallita → stop senza appello, come pre-registrato)

**Iterazione tuning eseguita come pre-registrato** (lr 3e-5, dropout 0.3, drop_path 0.2, wd 3e-3, 5 seed, 23 min):
- **Il fix meccanicistico HA funzionato:** training non collassa più a epoca 1 (best ep 4-13 per seed, val_nll 0.1928-0.197 ≤ probe), val Spearman fino a **+0.19-0.20** (probe: +0.045).
- **MA val→test anti-correlano anche a 1h:** test Spearman ensemble **−0.041 (p=0.001)**, IC −0.023, ICIR −0.12. L'anti-correlazione è del METODO, confermata sul nuovo timeframe.
- **Backtest gate (13 ≡ 23 bps): 2 trade, net −1.46%, PF 0.12, Sharpe −8.24** → 4/4 criteri falliti. Il 5-member ensemble medio-azzera μ (membri debolmente/anti-correlati) e gonfia σ_ens via legge varianza totale → conviction collassa, quasi zero entry.
- **Gate:** Sharpe≥1.0 ✗ · PF≥1.3 ✗ · ≥80 trade ✗ (2) · net>0 ✗ — a ENTRAMBI i costi. **KILL.**

**Bilancio scientifico del pivot (2 giorni, costo contenuto):** il 1h ha rimosso il muro dei costi (|μ|≈43bps ≫ 26bps — la diagnosi del KILL cross-sectional era giusta) ma ha esposto il problema più profondo: **non c'è skill direzionale OOS a nessun timeframe testato; la val non è predittiva del test.** Con 1m E 1h falliti per la stessa ragione (segnale, non costi), il filone "stesso metodo, altro timeframe" è esaurito. Restano i filoni con informazione/target NUOVI: **vol-S** (chiude B2, fattibile sui dati 1h già scaricati: RV oraria è il setting naturale per HAR-RV/QLIKE) e **B1 order-book L2**.

**Stato del repo post-KILL (decisione rollback APERTA):** config a 1h, dataset/regime/modelli 1h su disco; production 1m intatta in `data/backup_1m` + `models/backup_1m` (rollback = restore + `interval: 1m`, `start_time: '2025-05-19'`). Se la prossima strada è vol-S → conviene RESTARE su 1h (dati giusti già pronti). Se si torna al paper-trading 1m → rollback.

## 🔴 PROBE 1h ESEGUITO END-TO-END → GATE PRE-REGISTRATO **FALLITO** (4/4 criteri, a entrambi i costi)

**Pipeline completata oggi (in ordine):**
1. **Fix contratto state↔config verificato** (già in codice da ieri sera): 5/5 parity test verdi, suite full 125 passed / 1 skipped.
2. **Dati 1h:** 65.159 candele 2019-01-01→2026-06-10 (4 gap = halt Binance noti), 65.129 valide, **104 feature confermate anche a 1h**, split 51.130/6.391/6.392, X_train (51130,120,104).
3. **Macro+regime:** 38 FRED + 9 yfinance → 90 feature; **cadenza walk-forward Markov spostata 30d→90d** (refit expanding è O(t): 30d su 7 anni ≈ 9h, 90d ≈ 3h). Wiring: `macro.hmm_burn_in_days`/`hmm_retrain_days` in config (chiavi prima morte, ora consumate da `01b`). Distribuzione 7 anni: **R0 31,2% (σ²=0.12 quiet) / R1 34,9% (σ²=4.76 high-vol) / R2 34,0% (σ²=0.61 mid)** — mapping etichette DIVERSO dallo span 1m.
4. **Train probe iTrans 1-seed: 4,1 min.** Best checkpoint a **epoca 1** (val_nll 0.198, poi degrado monotono — overfit immediato). Test: Spearman +0.012 (p=0.33), DA 50,0%, IC +0.023, ICIR 0.22, coverage 97,8% (σ sovra-larga, pattern noto).
5. **Backtest test split, 13 E 23 bps** (nuovo flag `QUANTSYS_MIN_EXPECTED_RET`, inerte di default): risultati **identici** ai due costi perché il gate |μ| NON è vincolante a 1h (|μ| raw mediano ≈ 43 bps ≫ 23 bps — **il muro della magnitudine è effettivamente sfondato**).

| Criterio gate | Soglia | Esito |
|---|---|---|
| Sharpe | ≥ 1.0 | **−0.87** ✗ |
| Profit Factor | ≥ 1.3 | **0.78** ✗ |
| N° trade | ≥ 80 | **74** ✗ |
| Net return | > 0 | **−5,23%** ✗ |

(WR 51,4%; MDD 6,5%; fee $330 su $10k: il gross è ≈ −1,9% — la perdita è REALE, non erosione fee. CI bootstrap Sharpe [−3.45, +1.46] include lo 0.)

**Lettura onesta:** il pivot ha rimosso il vincolo di costo (μ ≫ fee, fee drag marginale) ma **a 1h non c'è skill direzionale** con questo modello/hyperparam: la perdita è del segnale, non della microstruttura. Caveat metodologici del probe: 1 seed, best=epoca 1 (modello quasi non-trainato: overfitta da epoca 2 su 51k finestre), hyperparam ereditati dal 1m senza retuning, LR schedule pensato per dataset 8× più grande.

**🐛 2 bug infrastrutturali trovati e chiusi oggi (entrambi figli del pivot):**
- **Membri ensemble stale:** `models/itransformer/best_model_{0..4}.pt` (1m, 4 giugno) + `best_model.pt` nuovo (1h) = mix → `EnsembleModel.load` falliva → **fallback silenzioso a SimpleSignalModel** (primo backtest era del modello giocattolo!). Rimossi (in `models/backup_1m`). ⚠ Il probe n=1 aggiorna solo `best_model.pt`, MAI i membri numerati.
- **PipelineState routing:** `01_download_data` salva lo state in `models/{QUANTSYS_ARCH}` (default `lstm`) — lanciato senza env, lo state 1h è finito in `models/lstm/` e `02_train` su itransformer ha ri-salvato quello stale 1m. **Il guard interval l'ha intercettato** (avrebbe corrotto la denorm con target_scale 1m = replica del bug Sharpe −256). Fix strutturale: (a) `01` salva anche copia canonica `models/pipeline_state.pkl`; (b) `02_train` ha guard anti-stale (interval pkl vs config → sostituisce dalla canonica o RuntimeError); (c) state itransformer riparato (interval 1h, target_scale 0.0326 ≈ 12× quello 1m, coerente con somma su 30 barre orarie).

## 📋 PRE-REGISTRAZIONE iterazione tuning 1h (decisione 1 — scritta PRIMA di girare, 2026-06-10 ~18:15)

**Ipotesi meccanicistica:** best=epoca-1 ⇒ LR 2e-4 ereditato dal 1m è troppo alto per 51k finestre stride-1 (campioni indipendenti effettivi ~1.7k per overlap target 29/30); il modello supera l'ottimo nella prima epoca e poi memorizza.
**Modifiche (config-only, `config/arch/itransformer.yaml`):** `learning_rate 2e-4 → 3e-5`; `tft_dropout 0.1 → 0.3`; `drop_path_rate 0.1 → 0.2`; `weight_decay 1e-3 → 3e-3` (override arch); `--n-ensemble 5`. TUTTO il resto invariato (patience 15, cosine T0=10, mixup off per regola repo, batch 128 eff.).
**Gate IDENTICO al probe:** Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 su test a 13 E 23 bps. **Esito negativo ⇒ KILL definitivo del timeframe 1h, senza ulteriori iterazioni.** Successo parziale (es. net>0 ma Sharpe<1) NON è promozione: verrà riportato com'è.

**▶️ DECISIONE PRESA: opzione 1 (tuning pre-registrato sopra). Le alternative erano:**
1. **Un'iterazione di tuning onesta** (pre-registrata, NON fishing): LR/wd/patience ricalibrati per 51k finestre + 5 seed (il best=epoca-1 dice che il training non è mai davvero partito). Costo ~1h GPU. Se fallisce anche questa → KILL definitivo del timeframe 1h.
2. **KILL secco** e passare a Strada 2 (vol-S, chiude B2) o B1 (order-book L2).
3. Ibrido: tuning 1h in background + avvio studio vol-S in parallelo (GPU permettendo).

**Nota dashboard/metrics:** `models/itransformer/{metrics.json,trades.csv,equity_curve.npz}` e `results/itransformer/dashboard_results.json` = run 23 bps (≡ 13 bps). `QUANTSYS_MIN_EXPECTED_RET` documentato in CLAUDE.md.

## 🕒 2026-06-09 (PIVOT 1h — implementazione)

## 🚧 PIVOT TIMEFRAME 1m→1h (Strada 1 ★) — codice interval-agnostic FATTO, pipeline dati DA LANCIARE

**Razionale (dalla diagnosi KILL del probe cross-sectional):** il muro è la MAGNITUDINE (~1.5 bps effetto vs ~26 bps costo round-trip); a 1h il cost/σ scende da ~1.9-3.3× a ~0.25-0.42× (movimento barra ∝ √Δt, costo fisso).

**Design cardine: `interval_minutes` parametrico con default 1 → tutte le conversioni sono IDENTITÀ a 1m** (path production 1m bit-perfect, reversibile). Finestre TIME-semantic (strutturali 30d/90d/365d, momentum, funding_1d, session 4h, ma200m) convertite via `bars_per_day`/`_tbars(minutes)`; finestre BAR-semantic (windows [5,10,20,60], CVD, vwap, VP scales 60/240/1440 barre) deliberatamente invariate.

**Fatto (fan-out 3 recon + 3 implementazione):**
- `quantsys/utils`: helper `interval_minutes_from_cfg` (single source of truth, ValueError fail-fast su intervalli ignoti).
- `config/default.yaml`: `interval: 1h`, `start_time: 2019-01-01` (~65k barre), `window_stride: 1` (era 5), `embargo_steps: 168` (era 1500), `max_hold_candles: 60` (era 240; vincolo ≥ h=30), `min_expected_ret: 0.0013` (gate cost-aware 13 bps; 2° test pre-registrato a 23 bps), `max_sigma: 0.10` (≈0.015×√60, da ricalibrare sui percentili post-denorm), TODO GJR-GARCH (ω va ri-stimato su rendimenti 1h; MC non è sul critical path del backtest).
- `quantsys/features/__init__.py`: `FeatureBuilder(interval_minutes=1)` + `_tbars` + conversioni TIME-semantic; wiring in `01_download_data`/`01_update_data`/`99_replay`.
- `quantsys/data/__init__.py`: passo incrementale da `_INTERVAL_SECS` (era `Timedelta("1min")`); `macro/__init__.py`: gap candele inferito dal passo mediano dei dati; `regime.py`: clock orario invariato by design, fail-fast se input >1h, con input 1h il resample è identità.
- `scripts/03_backtest.py`: `bars_per_year = 525600//interval_minutes` (annualizzazione Sharpe/CI), σ safety-net scalato `0.05·√interval` (preserva l'intento: cattura bug denorm ~30-100×, non la crescita √60 legittima); `quantsys/trading`: `RiskManager(bars_per_year=525_600)`.
- `scripts/04_live_signals.py`: FeatureBuilder wired, LiveCandleBuffer capacity interval-aware (35gg), WS già config-driven.
- **Backup 1m:** `data/backup_1m/{raw_candles,regime_probs}.parquet` + `models/backup_1m/{itransformer,nhits,tcnmamba}` (611 MB). Rollback = config 1m + restore.
- **Test:** 123/126 passed. I 2 fail in `test_live_training_parity` sono DIAGNOSTICI: espongono la combinazione incoerente *PipelineState 1m + config 1h* (il live path leggeva interval dalla config). Fix in corso: `interval_minutes` derivato da `PipelineState.interval` (campo già persistito) nei consumer + RuntimeError su mismatch config↔state (stesso pattern del guard forecast_horizon).

**▶️ PROSSIMI PASSI (in ordine):** (1) fix contratto state↔config (in corso); (2) `01_download_data.py` → 1h 2019→oggi (~65 richieste REST, sovrascrive raw/features/dataset — 1m già in backup); (3) `01b` macro+regime (walk-forward Markov su 7 anni); (4) train iTrans 1-seed (probe); (5) backtest cost-aware a 13 E 23 bps. **Gate pre-registrato:** Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 a ENTRAMBI i costi. Dopo: Strada 2 (vol-S, chiusura B2).

## 🕒 2026-06-06 (probe cross-sectional eseguito → KILL)

## 🔴 PROBE CROSS-SECTIONAL IC — ESEGUITO → VERDETTO **KILL**

- **Implementato e girato** (fan-out 3 subagent + orchestrazione): `quantsys/data/universe.py` (PerpUniverse top-N), `scripts/xs_01_download.py` (16 perp USDT scaricati: ADA AVAX BNB BTC DOGE ENA ETH FIL LINK LTC NEAR SOL SUI WLD XRP ZEC, raw+funding, stesso span), `scripts/xs_02_panel_signals.py` (applica l'ensemble esistente per-asset, denormalize z→raw, grid 30 candele → `data/xs/mu_panel.parquet`, 261.900 righe/16 simboli), `scripts/xs_03_ic_report.py` (Spearman cross-sezionale, sub-periodi non sovrapposti, verdetto pre-registrato). Report: `results/xs/ic_report.json`.
- **Risultato OOS:** mean cross-sectional IC **+0.0138**, t-stat **+1.86** (<2), ICIR 0.035, sub-periodi [+0.043, −0.016, +0.0002, +0.036, +0.005] = **4/5 positivi**. Tradability: spread lordo top-bottom **+1.5 bps/step**, **netto −0.00245/step** (~−43 ann, 13bps/leg).
- **VERDETTO KILL** (gate: |t|≥2 FALSE, ≥4/5 TRUE, net>0 FALSE → serve tutti e 3). **Non costruire il portfolio layer.**
- **Lettura:** μ ha skill di rango cross-sezionale **debolmente positiva** (direzione giusta, NON anti-predittiva come nel single-asset WR 29%), ma l'effetto (~1.5 bps) è **~17× sotto il costo** (~26 bps round-trip long-short). **Il muro è la MAGNITUDINE, non il segno.** Stesso pattern strutturale: rank reale ma troppo debole per le fee. Il de-risk economico ha risparmiato il build M.
- **Nota infra:** GPU 8GB satura con A+xs_02 in parallelo → contesa CUDA. Paper-trading A **FERMATO** per liberare la GPU (xs_02 da 28min-stallo → 28s/simbolo). A NON riavviato (vedi sotto: poco valore ora).

## ▶️ Sotto-sessione 2026-06-06 (paper-trading live + pivot)

- **STRADA A — paper-trading AVVIATO e sano (processo persistente in background).** `$env:QUANTSYS_ARCH="itransformer"; python scripts/04_live_signals.py` (log dedicato `logs/paper_trading_live.log`, log pulito `logs/quantsys_20260606_134026.log`). Warmup OK: ensemble eterogeneo 3 archi, 104 feature, funding 5941 obs, bootstrap 50k + **catch-up A1.1 +2899 candele REST** → ultima 2026-06-06 11:40, **WebSocket connesso** `btcusdt@kline_1m`, equity paper $10.000, feature in calcolo su candele live. Segnali in `results/itransformer/live_signals.jsonl`. ⚠ Se la macchina riavvia, **va rilanciato** (non è un servizio). Aspettativa onesta: piatto/negativo — è validazione forward + readiness, non scommessa Sharpe.
- **PIVOT — fan-out di 4 subagent (studi di fattibilità read-only).** Sintesi ranked (dettaglio completo in memoria `pivot_fanout_2026_06_06`):
  1. **Cross-sectional multi-asset** — **prior tradabile più alto + test più economico.** Long-short rank market-neutral top-N perp: usa solo l'ordinamento (= il nostro Spearman) e cancella il beta comune. **De-risk quasi-gratis: misurare la Spearman cross-sezionale di μ PRIMA di costruire il portfolio layer** (se ≈0 → stop). Backfillable ora. Effort M. Success: net Sharpe ≥1.0 test, +in ≥4/5 sub-periodi, fees<40% gross.
  2. **Timeframe → 1h** — cost/σ 1m=1.9-3.3× → 1h~0.25-0.42×. Serve re-download multi-anno + re-tune lookback hardcoded a 1m. Effort M. Rischio: cost-fragile; anti-correlazione val→test è del *metodo*.
  3. **Target → volatilità** — **PREDICIBILE MA NON TRADABILE** sul perimetro attuale (no strumento; Deribit = progetto > B1). Valore: esperimento S che chiude B2 (vol vs HAR-RV, QLIKE) + jump/no-trade gate difensivo per A.
  4. **Asset class → ES 1m** — inefficienza session-mechanical (documentata, causale). Dati ~free (Databento). Rewrite data-layer + ~2/3 riuso. Effort M. Rischio: già HFT-arbitraggiato; leakage roll.
- **B1 (order-book L2)** resta l'asse "informazione nuova" non spremuto, ortogonale (memoria `future_orderbook_l2`).

## 🕒 2026-06-06 (merge doc bilingue single-file)

## ✅ Sotto-sessione 2026-06-06 (doc bilingue fuse)

- **Fusi i 4 file `.md` gemelli IT/EN in un unico file bilingue paragrafo-per-paragrafo.** Coppie fuse → file base, gemelli eliminati: `AVVIO.md`(+`AVVIO.en.md`), `README.md`(+`README.it.md`), `TEORIA.md`(+`TEORIA.en.md`), `docs/MODEL_IMPROVEMENTS.md`(+`.en.md`). Formato: heading bilingue `IT · EN`; corpo con paragrafo IT (prefisso `🇮🇹`) seguito da EN (prefisso `**EN**`); blocchi di codice emessi una volta, tabelle duplicate IT/EN; puntatori "versione X in Y" rimossi; cross-reference `.en.md`/`.it.md` reindirizzati ai file base (le menzioni storiche dentro MODEL_IMPROVEMENTS restano come record).
- **Metodo:** script one-shot `scripts/_merge_bilingual.py` (poi rimosso) con allineamento **sezione-per-sezione via difflib** su chiave heading language-neutral (numeri/date/emoji/`Step X` + ancora primaria `Stage|Phase|Fase|Step|Fix #N`) e **resync corpo** via ancore inline-code/numeri. Validato con check di adiacenza marker: AVVIO 38/38 e README 26/26 perfetti; TEORIA ha 1 blocco IT-only legittimo (`Perché T=120`, assente nell'EN); MODEL ha la sola sezione IT-only `RESUME 2026-06-04` + Stage 4 con heading `✅ COMPLETATO · 🚧 IN PROGRESS` (l'EN era stale, drift pre-esistente reso esplicito).
- **Doc-convention aggiornata in `CLAUDE.md`** (direttiva #2 + nomenclatura): single-file bilingue, NON ricreare i gemelli.
- ⚠ **Drift residuo da sanare (non bloccante):** l'EN di alcune sezioni era più vecchio dell'IT (evidente in `MODEL_IMPROVEMENTS` Stage 4). Ora visibile nello stesso file → riallineare le due lingue alla prossima modifica di quelle sezioni.

## ✅ Sotto-sessione 2026-06-06

- **Student distillato MISURATO (chiude la domanda OOS):** distillato N-HiTS multi-teacher (teacher=iTrans) e backtestato single-arch (test). Risultato **IDENTICO a 4 decimali** al N-HiTS standalone: return −3.57%, Sharpe −28.96, PF 0.21, WR 35%, 17 trade. I `best_model.pt` hanno **hash diversi** (modelli genuinamente diversi) ma stesso esito di trading → conferma empirica che **la distillation non cambia l'OOS** (corr 0.995 resa manifesta). Baseline N-HiTS ripristinato da backup. **La leva NON è la variante di modello.**
- **`run_all.py`:** `--arch` → `--n-ensemble 5` (default, override via flag); `--distill` resta a 1. + fix UTF-8 `--help` (cp1252). Committato e pushato su main (`92d7beb`).
- **Roadmap A1.1 — catch-up contiguo `candle_buffer`** (`scripts/04_live_signals.py`, `warmup()`): via `fetch_klines(start_time=last)` colma il gap tra il bootstrap parquet (può essere vecchio di giorni) e "ora"; mirror legacy reso dedup-safe (fallback). **Verificato con smoke test:** "+2211 candele REST → ultima 2026-06-06 00:12", 2 segnali emessi sul buffer contiguo, zero errori. Risolve il buco temporale che le feature a lookback lungo (ma200m/vp) attraversavano.
- **Fix cp1252 in `scripts/02_train.py`** (3ª occorrenza, aveva causato l'exit 1 "failed" del distill in background — il modello era comunque salvato): reconfigure UTF-8 stdout/stderr in `main()`.
- Recon roadmap A1 (2 subagent): catch-up candele + meccanismo funding. FundingRatePoller (Stage 4.4) resta come miglioria minore (funding cambia ogni 8h, ffill'd → workaround adeguato).
- **B2 esplorato e CHIUSO negativo (2 step de-risk):**
  - **Step 0** (`scripts/dev_step0_regime_sigma.py`, no-training): la mixture-of-universes (σ regime-condizionata) **accantonata** — aggiunge solo +0.0155 nats sopra una ricalibrazione σ globale; R1 Trend resta NLL 2.05 con σ-oracolo = μ-error irriducibile. MA ha scoperto che **σ è ~3× troppo grande** (std(z)=0.37/0.665/0.41, scale globale ottimo 0.33).
  - **Step 0.5** (flag `QUANTSYS_SIGMA_SCALE` in `03_backtest.py`, sweep val): ricalibrare σ verso il basso **peggiora monotonicamente** il backtest (return 4.03%→1.33%, PF 1.88→1.16). La σ larga disabilita di fatto gli stop → hold-to-horizon, migliore per edge debole. **NLL-calibrazione e PnL in conflitto; ottimo trading ≈1.0.** Flag inerte.
  - **Bilancio:** tutti i lever model/backtest-side sono esauriti (distill, ensemble, pesi, rank-harvest, mixture, σ-recal). Restano solo **A (paper-trading = verità forward, pronto)** e **B1 (order-book L2 = informazione nuova, progetto-dati a sé, accantonato)**.

## 🕒 2026-06-05 (BLOCKER #1 Stage 5 CHIUSO + Tier-1 rank esaurito)

## ✅✅ BLOCKER #1 RISOLTO — Stage 5 chiuso (parity live↔training)

- **Stato reale > doc-tracker:** Stage 4.6/4.7 (integrazione in `LiveEngine.__init__` + rimozione `_pad_or_truncate` → `assert window.shape[-1]==104`) erano **già nel codice**. Il LiveEngine carica `EnsembleModel.load_heterogeneous` (riga 934) che **non** espone `predict_with_uncertainty` → `_predict` cade sempre sul ramo **deterministico** (MC-dropout non scatta in produzione).
- **Refactor parity-safe** (`scripts/04_live_signals.py`): estratto `LiveEngine._deterministic_predict(model, window, xm, ps, device)` — nucleo deterministico (forward + denormalizzazione z→raw) **condiviso** da `_predict` (ramo ensemble) e dal parity test → il test esercita il path reale, zero drift.
- **Gate 1 — parity FEATURE:** `tests/test_live_training_parity.py` (5/5) + `99_replay_live_vs_training.py` → **max|Δ|=0.000e+00** su finestra (120,104).
- **Gate 2 — parity SEGNALE (Stage 5, nuovo):** stessi due percorsi feature attraverso `_deterministic_predict` + `SignalGenerator` → **Δμ=0, Δσ=0, side identico**. Codificato in `test_signal_parity_live_vs_offline` + sezione Gate 2 nello script di replay.
- **Test obsoleto aggiornato:** `TestBlocker1Documentation` (in `test_recent_fixes.py`) riscritto — il legacy `LiveFeatureBuffer` (39 feat) è deprecato (solo ATR/sanity), non più il path feature. **Suite: 21/21 verdi.**
- **Smoke test live (Stage 4.10) ESEGUITO 2026-06-05 — SUPERATO.** WS Binance reale connesso e stabile, warmup OK, pipeline 104-feature end-to-end su candele live, segnali emessi in `results/itransformer/live_signals.jsonl` (es. μ=8.3e-4, σ=0.0175 raw, prob_up=0.518, side=NONE — coerente con la scala del parity test). **Lo smoke test ha trovato e risolto 3 bug live reali** (`scripts/04_live_signals.py`): (1) crash `UnicodeEncodeError` cp1252 sul banner `run()` → reconfigure UTF-8 stdout/stderr in `main()`; (2) `AttributeError .dt` su `open_time` dtype misto (WS=epoch-ms int vs bootstrap=Timestamp) → helper `LiveCandleBuffer._norm_ts`; (3) `ValueError mix tz-aware/naive` (parquet tz-aware vs WS tz-naive) → uniformati a tz-naive UTC alla sorgente. Più fix test pre-esistente `test_circuit_breaker_blocks_new_positions` (stato DD irrealistico vs nuova logica recovery) + nuovo `test_circuit_breaker_auto_recovers...`. **Suite full: 125 passed, 1 skipped, 0 failed.**
- **Residuo:** solo avvio paper-trading 2-4 settimane (richiede dati `raw_candles.parquet` aggiornati: nello smoke il buffer bootstrap era fermo al 2026-06-04, gap colmato solo dal WS live).

## ✅ Appena completato (sessione Tier-1 harvest dell'edge ordinale)

- **Fix ① — Cadenza decisionale = orizzonte** (`scripts/03_backtest.py`, env `QUANTSYS_DECISION_CADENCE`, default `0`=off, `"h"`=`forecast_horizon`).
  Razionale: un segnale a orizzonte h tradato ogni candela genera h bet sovrapposti/autocorrelati (breadth effettiva ≪ nominale, IR≈IC·√breadth). Gate causale: nuova entry solo se `i − _last_entry_i ≥ cadence`. **Gli exit (SL/TP/trailing/circuit-breaker) restano ogni candela.** Inerte di default.
- **Fix ② — Esposizione continua rank-based, regime-gated** (env `QUANTSYS_RANK_EXPOSURE=1`/`QUANTSYS_RANK_REGIME`/`QUANTSYS_RANK_BAND`/`QUANTSYS_RANK_MIN_SIGMA`/`QUANTSYS_RANK_WIN`).
  `r`=percentile causale di μ nel buffer ∈[0,1]; `s`=2r−1; no-trade band `|s|<band` (deadzone=isteresi); `conviction=(|s|−band)/(1−band)` → scala il Kelly con continuità (`dist.conviction → RiskManager._size`). Attivo SOLO nel regime target (Quiet di default), NONE altrove. Sostituisce concettualmente il rank-entry **discreto** (che distrugge l'informazione ordinale). Inerte di default.
- **Verificato:** `py_compile` OK; sanity-check matematico del mapping rank→conviction e del gate cadenza (band=0.5 ⇒ trade solo top/bottom 25%; cadence=0 ⇒ baseline invariato).
- **Doc sincronizzate:** `CLAUDE.md` (nuovi flag), `AVVIO.md`+`AVVIO.en.md` (paragrafo harvest edge ordinale).

## 🧪 Esito validazione su VAL — Fix ①② FALLISCONO decisamente (NON promossi)

Confronto su `QUANTSYS_BACKTEST_SPLIT=val`, ensemble eterogeneo 3 archi, `cadence=h`, `rank_exposure=1`, `band=0.5`, `regime=Quiet`, `min_sigma=0`:

| Metrica         | Baseline (clean) | **Fix ①②**        |
|-----------------|------------------|--------------------|
| Total return    | **+4.03%**       | **−2.24%**         |
| Sharpe          | +12.89           | **−45.31**         |
| Profit Factor   | **1.88**         | **0.22**           |
| Win Rate        | 61.0%            | **25.7%**          |
| N° trade        | 41               | 35                 |
| avg_hold        | 188 candele      | 56 candele         |
| Close reasons   | MAX_HOLD 24 / SIGNAL 11 | **SIGNAL 27** / STOP_LOSS 6 |

- **Diagnosi meccanicistica:** il segnale rank come *entry direzionale* è **anti-predittivo su val** (WR 20-27% in entrambi i bucket di vol). Inoltre l'esposizione continua flippa di segno → 27/35 chiusure per **SIGNAL** (avg_hold crolla 188→56): la PnL realizzata è dominata dal path SL/TP/flip, **NON** dal rendimento a orizzonte-30 su cui è misurato lo Spearman +0.13÷0.19. L'edge di rango NON sopravvive alla macchina di realizzazione del trade.
- ⚠ **Nota sulla baseline val (+4%, PF 1.88):** è il lato favorevole — val→test anti-correlano (shift strutturale). Il baseline su **test** resta negativo (−1.77%). Non leggere il +4% come successo.
- **DECISIONE:** Fix ①② **NON promossi**; restano env-flag **inerti/reversibili**. Stato production invariato (i file di test non sono stati toccati; le run val usano suffisso `_val`). Artefatti diagnostici: `models/itransformer/metrics_val_{baseline,fix12}.json`.

**Test di isolamento orizzonte-locked (`QUANTSYS_HORIZON_EXIT=h`, chiusura TEMPORALE pura a 30 candele, bypassa SL/TP/SIGNAL):** `close_reasons={MAX_HOLD:41}`, avg_hold 30.0 esatto. Esito val: return **−0.97%**, PF **0.49**, WR **29.3%** (low_vol bucket WR 18%, 33 trade). Migliora vs Fix①② (PF 0.22→0.49: il path SL/TP/flip era drag aggiuntivo) **ma resta negativo**. ⇒ **anche realizzando la PnL esattamente al rendimento a orizzonte-30, l'edge di rango è anti-predittivo OOS (WR<50%).** Lo Spearman in-sample NON è un edge direzionale tradabile. **Filone rank/soglia ESAURITO** (3ª conferma). Non inseguire l'inversione di segno (overfit val, inversione regime già non-robusta). Artefatto: `metrics_val_rank_hx.json`.

## ✅ Sessione precedente (val-backtest split + esito edge Quiet)

- **Verifica CLAUDE.md vs codice** — allineato. Corretto unico ref sfasato: AMP-off in inference `ensemble.py:170` → **`:275`**.
- **Passo 1 — modalità val-backtest implementata** in `scripts/03_backtest.py`:
  `QUANTSYS_BACKTEST_SPLIT=val|test` (default `test`). Carica `X_{split}/y_{split}/t_{split}` e `X_macro_{split}`,
  allinea OHLCV (`raw_candles.parquet`) e regimi (`regime_probs.parquet`) su `t_{split}`.
  Output **suffissati** per `val` (`metrics_val.json`, `equity_curve_val.npz`, `trades_val.csv`,
  `dashboard_results_val.json`) → la run val **non clobbera** i file production-clean (risolve vecchio problema #4 per il caso val).
- **Passo 2 — backtest Quiet eseguito su val** (`q=0.10, σ≥0.004`, ensemble eterogeneo 3 archi).
- **Documentazione sincronizzata:** `CLAUDE.md` (nuova env var), `AVVIO.md`+`AVVIO.en.md` (esito validazione).

## 🧪 Esito chiave — l'edge Quiet rank-entry NON regge su val (Passo 3)

| Metrica            | Test (2026-06-04) | **Val (held-out, 2026-06-05)** |
|--------------------|-------------------|--------------------------------|
| Return             | −0.74%            | **−0.22%**                     |
| N° trade           | 34                | **13**                         |
| Win Rate           | 47%               | **38.5%**                      |
| Profit Factor      | —                 | **0.84** (<1 = perdita)        |
| Sharpe             | —                 | **−4.04**                      |

- Distribuzione regimi su val: **R0 Quiet 1428 (14%) / R1 Trending 1488 / R2 Stress 7187 (71%)** → periodo dominato da Stress, pochissimo Quiet ⇒ solo 13 trade, **sotto-campione non significativo**.
- Criterio Passo 3 (`return ≥ break-even, segno coerente`) **NON soddisfatto** → l'edge è **overfit del test**.
- **DECISIONE:** rank-entry Quiet **NON promosso** a regola nel `SignalGenerator`; resta env-flag **inerte/reversibile**. Stato production = **ensemble eterogeneo pulito** (≈ iTrans standalone).

## ⚠️ Problemi aperti

1. **L'edge tradabile a soglia/rank non esiste OOS** con questo dataset. L'unico segnale stabile (Spearman Quiet) è di rango ma troppo debole/raro per coprire le fee. Direzioni residue: (a) accumulare trade reali via paper-trading; (b) ripensare l'entry (non |μ|, non rank semplice).
2. **BLOCKER #1 (live) — RISOLTO 2026-06-05 (parity codice chiusa).** Resta solo il residuo OPERATIVO: smoke test live via WS Binance reale (Stage 4.10, non eseguito in questa sessione — richiede connessione) + avvio paper-trading. ⚠ I segnali ora riflettono il backtest, ma **il backtest è negativo OOS** (l'edge a soglia/rank è esaurito): il paper-trading serve ad accumulare trade reali, non c'è aspettativa di Sharpe>0 a priori.
3. **`dashboard_results.json`/`metrics.json` di `models/itransformer`** restano dell'ultima run di **test** (production); i nuovi file `*_val.*` sono separati. Se serve uno stato test production-clean fresco, rilanciare il backtest senza env sperimentali (comando sotto).

## ▶️ Azione esatta da cui ripartire

**STATO al 2026-06-06: TUTTI i lever model/backtest-side esauriti e dimostrati negativi OOS** (distill≡baseline, ensemble corr 0.995, pesi dinamici anti-correlano, rank-harvest fallito, mixture-of-universes non vale, ricalibrazione σ peggiora). L'infrastruttura è SOLIDA: BLOCKER #1 chiuso (parity bit-perfect), live engine robusto (catch-up A1.1, smoke test superato, 3 cp1252 fixati), `run_all.py` aggiornato (`--arch`→5 / `--distill`→1). **Non restano tweak di modello con prior non-nullo.**

**▶️ STATO 2026-06-06 (aggiornato):** Strada A avviata e poi **fermata** (poco valore validare forward un modello in pivot + serviva la GPU). Pivot **studiato** (fan-out 4 assi). **Probe cross-sectional IC ESEGUITO → KILL** (vedi sezione in cima). **SESSIONE IN PAUSA dopo commit.**

**▶️ AZIONE ESATTA ALLA RIPRESA — scegliere fra le 3 strade residue (cross-sectional già escluso):**
La diagnosi del KILL è precisa: il rank di μ è cross-sezionalmente **debolmente positivo (segno giusto)** ma **~17× sotto i costi** → **il muro è la MAGNITUDINE, non il segno.** Questo ordina le opzioni:
1. **★ TIMEFRAME → 1h (RACCOMANDATO — attacca direttamente la diagnosi):** a 1h il rapporto cost/σ scende da ~1.9-3.3× (1m) a ~0.25-0.42× perché il movimento di barra cresce ~√Δt e il costo è fisso. Passi: `config/default.yaml` `interval:1h` + `start_time` multi-anno (~2019, il resample del file 381g è troppo sottile: 1h~9k barre); **re-tune dei lookback hardcoded a 1m** in `quantsys/features/__init__.py` (`_structural_features` ~l.536/547, `price_vs_ma200m` ~l.559 `rolling(200)`=minuti, `_funding_features` `rolling(1440)`→`rolling(24)`, `features.windows`) e il regime-clock in `quantsys/macro/regime.py` (`_build_btc_hourly_df` resample 1m→1h diventa no-op); rebuild dataset; train iTrans single-arch; backtest cost-aware OOS col filtro magnitudo **già nel repo** (`min_expected_ret`/`min_snr`) a 13 E 23 bps. Success: Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 a entrambi i costi. Effort M ~2-3g. Rischio: cost-fragile + l'anti-corr val→test è del *metodo*.
2. **VOL-S (ortogonale, costo S):** re-target a RV (one-line `target=Σlog_ret²` in `FeatureBuilder._returns` ~l.137), baseline HAR-RV+GJR-GARCH, giudice **QLIKE** OOS; success = batte HAR-RV ≥5% su test. Chiude scientificamente B2 + estrae il **jump/no-trade gate difensivo**. NON è alpha (vol predicibile ma non tradabile sul perimetro spot/perp).
3. **ES 1m (esci dalla crypto, effort M ~1-2 sett.):** porta l'architettura su index futures (inefficienza session-mechanical, documentata/causale). Dati ~free per le barre (Databento PAYG); rewrite data-layer + ~5-8 feature (drop funding, VWAP reset per sessione, add time-of-session/overnight-gap). Rischio: già HFT-arbitraggiato; leakage del roll continuous-contract.

**B1 order-book L2** resta l'asse informazione-NUOVA non spremuto, ortogonale a tutti e 3 (memoria `future_orderbook_l2`).
**Riavvio paper-trading A** (`$env:QUANTSYS_ARCH="itransformer"; python scripts/04_live_signals.py`): solo SE si decide di validare forward un modello (NON ora; la readiness è già verificata). ⚠ GPU 8GB NON regge A + training/inferenza in parallelo.

Memorie di riferimento per riprendere: `pivot_fanout_2026_06_06` (4 assi + esito KILL), `roadmap_post_blocker1_2026_06`, `future_orderbook_l2`, `mixture_of_universes_design`.

Comando stato **test production-clean** (azzera TUTTI gli env sperimentali prima di un backtest "production"):
`Remove-Item Env:\QUANTSYS_RANK_EXPOSURE,Env:\QUANTSYS_DECISION_CADENCE,Env:\QUANTSYS_HORIZON_EXIT,Env:\QUANTSYS_SIGMA_SCALE,Env:\QUANTSYS_QUIET_RANK_Q,Env:\QUANTSYS_QUIET_MIN_SIGMA,Env:\QUANTSYS_BACKTEST_SPLIT,Env:\QUANTSYS_BACKTEST_SINGLE_ARCH -EA SilentlyContinue; $env:QUANTSYS_ARCH="itransformer"; python scripts/03_backtest.py`
