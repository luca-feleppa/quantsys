# STATUS.md — Continuity Log

> Aggiornato alla fine di ogni ciclo di lavoro (direttiva permanente #3 di `CLAUDE.md`).
> Prima di iniziare qualsiasi task: **leggi questo file**. Memoria di lungo periodo dettagliata: `~/.claude/projects/E--quantsys-project/memory/`.

---

## 🟢 RISOLTO (definitivo) 2026-06-24 — DASHBOARD rendering: la causa era l'asse LINEARE; fix = asse CATEGORY

Chiuso **per davvero** il bug "barre OI + curva payoff spariscono/si schiacciano al cambio-tab", dopo che il primo tentativo (helper `plot()` con size-guard + width/height espliciti, sezione sotto) si è rivelato **necessario ma NON sufficiente** (riverificato col browser: barre ancora KO al rientro-tab). Diagnosi browser sistematica (Playwright + `getBoundingClientRect`, ~14 esperimenti) → **root cause vera trovata e fix verificato indipendentemente su 3 restart freschi.**

**Root cause (definitiva):** su un **re-render dopo `display:none→block`** (qualsiasi rientro-tab) Plotly **CORROMPE la mappatura-pixel dell'asse X LINEARE numerico**. Sintomo a due facce, stesso bug: (a) le trace finiscono **fuori campo** (barre a `x≈−1244px`) oppure (b) tutta la banda **compressa in ~19px**; restano solo shapes/annotation `yref:'paper'` (scala-indipendenti) → "spariscono". `_fullLayout` (range/offset/length/margin) risultava **identico** tra primo render OK e re-render KO: la corruzione è nel rendering SVG, non nei dati. **Asimmetria chiave:** il PRIMO render di ogni plot è sempre OK; ogni successivo è KO. **Solo l'asse X è colpito** (la Y numerica rende sempre bene).

**Cosa NON lo risolve (provato uno per uno, tutti KO):** `Plotly.react`, `Plotly.newPlot`, `Plotly.purge`+newPlot, `purge`+`innerHTML=''`+newPlot, **sostituzione del nodo DOM**, `Plotly.Plots.resize`, `Plotly.relayout` (width-toggle / range-toggle), `Plotly.redraw`, evento `resize`, **autorange** (sposta solo off-screen→crammato), rimozione width-barra, scala-x ÷1000, hiding via `visibility`/`position` invece di `display:none`, purge-on-leave. **Nessun rimedio via API recupera l'asse lineare corrotto.**

**Fix (verificato):** **asse X = `type:'category'`** sui due grafici colpiti (`plot-oi`, `plot-payoff`). Un asse category posiziona per **INDICE** (non per scala numerica) ed è **IMMUNE** alla corruzione — coerente col fatto che `plot-greeks` (già category) non ha MAI avuto il bug. Dettagli:
- **`renderOI`:** strike di banda → **categorie stringa** (ordinate ascendenti); via `catPos(v, ks)` (nuovo helper) Spot e Max-Pain a **indice FRAZIONARIO** così le linee restano proporzionali fra gli strike; tick diradati (`dtick≈n/9`); niente più width-barra esplicita (su category è in slot, auto). Gli strike di banda sono ~equispaziati → vista category ≈ lineare.
- **`renderPayoff`:** prezzi `xs` (linspace regolare) → categorie; Strike/Entry a indice frazionario (`catPos`); marker settlement agganciato alla categoria più vicina (curva fitta, snap trascurabile) → resta SULLA curva; Y resta numerica autorange.
- Helper `catPos(v, arr)` aggiunto (posizione frazionaria su asse category).

**Verifica (fan-out subagent, 3 restart freschi × 6 cambi-tab):** OI bars `spanPx=1467` su **OGNI** step risk inclusi i rientri (s2/s4); payoff curve `spanPx=685` su **OGNI** step trades inclusi i rientri (s3/s5); identico su a1/a2/a3, zero flicker. Screenshot ispezionati a mano: barre call/put complete + Net OI + Spot/Max-Pain ben posizionati; V del payoff completa + marker sulla curva + linee strike/entry. `py_compile` OK; HTML servito 40381 byte (`type:'category'`×2, `catPos`×9).

**Lezione metodologica (oro):** misurare il **pixel SCHERMO reale** (`getBoundingClientRect` dentro l'area di plot **+ confronto con lo screenshot**), MAI `getBBox` né il count dei path SVG (coord user-space, ignorano transform/clip → falsi positivi: riportavano "71 barre" su una tab VUOTA). Il primo "fix" sembrava ok a metriche deboli ed era rotto. **La verità è lo screenshot.**

## 🟡 (SUPERATO il 2026-06-24 — vedi sopra) RISOLTO 2026-06-24 — DASHBOARD rendering: meccanismo di render unificato (`plot()`)

> ⚠ Questa sezione descrive il PRIMO tentativo: necessario (l'helper `plot()` resta, con size-guard + rebuild-on-re-entry) ma **NON sufficiente** — il bug persisteva al rientro-tab. La causa vera (asse lineare) e il fix definitivo (asse category) sono nella sezione sopra. Lasciata come record del percorso.

Chiusi i 2 problemi residui ("OI by strike" e "Risk profile dei trade" che sparivano al cambio-tab `risk`/`trades`). Come da nota del 06-23 ("NON fare altri patch per-grafico: affrontare il MECCANISMO in modo olistico"), **rimosse TUTTE le patch per-grafico accumulate** e sostituite con **un solo helper `plot(id, traces, layout, cfg)`** in `scripts/06_dashboard.py`.

**Root cause unificante (confermata):** Plotly disegna su geometria sbagliata quando il container è a dimensione 0 o appena riflowato (tab `display:none→block`): le barre/curve nascono fuori vista o sub-pixel → spariscono, restano solo linee+shapes (scala-indipendenti). I sintomi "OI" e "payoff" erano LA STESSA classe di bug (container non misurabile al momento del render async).

**Fix (`plot()` — meccanismo olistico, deterministico):**
1. **Size-guard:** se `offsetWidth===0` (tab nascosta o reflow non ancora flushato) ritenta al frame successivo (`requestAnimationFrame`, cap 60 ≈1s, poi rinuncia: il prossimo `switchTab` ridisegna). Leggere `offsetWidth` forza già il reflow sincrono.
2. **Dimensioni esplicite:** passa `width`/`height` dai pixel reali del container + `autosize:false` → Plotly NON rimisura da solo (era la sua misura transitoria durante il reflow a corrompere la geometria).

**Pulizia (rimosso il cruft diagnostico):** eliminati `PL_CFG_2D_STATIC` (responsive:false), contatore `__roi`, `console.log` di debug, retry-rAF locale in `renderOI`, doppio-rAF resize in `switchTab`, `Plotly.purge` TEST in `loadRisk`, `.then(relayout+resize)` in `renderPayoff`. **TUTTI** i 7 render (surface/smile/term/oi/greeks/pcr/payoff) ora passano per `plot()` → unico punto di controllo della geometria. Mantenuti: guard anti-race async↔tab in `loadRisk` (skip se `page-risk` non più attiva), guard dati degradati, `Plotly.purge('plot-payoff')` quando non ci sono trade.

**Verifica:** `py_compile` OK; smoke server fresco su :8050 → HTML 37219 byte con `plot()` presente, zero residui (`PL_CFG_2D_STATIC`/`__roi`/`console.log`/`Plots.resize` = 0), `/api/risk` HTTP 200 con chain reale (3006 byte). ⚠ Trappola onorata: un vecchio processo dashboard teneva :8050 e serviva HTML stale (`SO_REUSEADDR`) → ucciso (stub+worker venv, 1 processo logico) prima dello smoke. Dashboard riavviata, live. **NB: la verità finale è il browser dell'utente con HARD RELOAD (Ctrl+Shift+R)** — la pagina già aperta gira il JS vecchio.

**Doc sincronizzate (2026-06-24):** sezioni dashboard di README/AVVIO + intero corpus `.md` (fan-out subagent — vedi sotto). Processi sfondo 01c/01d/04b vivi.

## 🕒 Ultimo aggiornamento: 2026-06-22 (DECISIONE: resta sul PASS iTrans, distill 5-seed RIMANDATO)

## 🟢 2026-06-23 — DASHBOARD: greche col segno+nome, tab Trades, e RISOLTI 2 bug di rendering Plotly

- **Greche più chiare** (richiesta utente): valori col **segno esplicito (+/−)** e **nome tra parentesi** (`Δ (Delta)` ecc.) in card aggregate, grafico a barre, header chain (+tooltip) e celle firmate. Helper `fmtS`/`fmtKS`/`GK_NAME`. Commit `1c0a1d8`.
- **Nuova tab Trades** + endpoint `/api/trades` (`build_trades` legge `results/vol_paper/trades.jsonl`): card riepilogo (PnL tot/hit-rate/avg/best-worst/n-gate), tabella trade (lato LONG/SHORT, strike, spot ingresso, premio, settlement, payoff, PnL firmato, edge, status) e **profilo di rischio/payoff** per trade (PnL@scadenza vs sottostante, opzione inversa `|S−K|/S`, costo calibrato sul realizzato, marker settlement + linee strike/entry). Commit `1c0a1d8`.
- **🐛 BUG OI-by-strike "barre spariscono dopo refresh" — ROOT CAUSE TROVATA E RISOLTA.** Non era react-vs-newPlot (tentativi precedenti falliti). Causa reale: la banda di zoom includeva *qualsiasi* strike con OI ≥0.5% del picco; con strike da 20k a 380k, quando a un refresh uno strike **far-OTM** supera la soglia il **range-x esplode** (~360k) → le barre (~500-1000 larghe) diventano **sub-pixel = invisibili**, mentre linea Net OI e shapes (scala-indipendenti) restano → esattamente la firma del bug. **Fix:** banda cappata a **spot ± 35%** + **larghezza barra esplicita** (≈85% gap mediano) così non collassa mai.
- **🐛 Profilo di rischio Trades "solo 2 linee verticali" — RISOLTO.** I dati payoff erano validi (verificato: V asimmetrica, no NaN); la curva non si disegnava ma gli shapes sì. Causa: `plot-payoff` è in `.grid2` (colonna a metà larghezza) e all'attivazione della tab (display:none→block) il reflow dava larghezza 0 al render + autorange. **Fix:** **range x/y espliciti** (come renderOI) + **`Plotly.Plots.resize` post-render** (ri-misura dopo il reflow).
- **Verifica:** py_compile OK; smoke server fresco → HTML 33003 byte con tutti i fix, `/api/trades` (7 trade) e `/api/risk` (97 strike) OK. ⚠ Trappola smoke: vecchi processi dashboard su :8050 servivano HTML stale (`SO_REUSEADDR`) — uccidere via `Stop-Process` (taskkill falliva su stub+worker venv). Doc README/AVVIO sincronizzati. **NB: confermare nel browser reale** (il test headless non riproduceva il bug OI: la verità è l'occhio dell'utente).
- **🐛 OI bars — 2ª iterazione (la diagnosi giusta era dell'utente: dati non sincronizzati).** Band-cap+width NON bastavano: causa vera = **chain Deribit parziale/vuota cachata** (`_TTLCache`, TTL 8s) → `build_risk` con `strikes=[]` → barre a zero per tutto il TTL; + `ConnectionAbortedError` (browser chiude la connessione al refresh, handler ri-crashava sul write del 500). **Fix:** (1) `fetch_option_chain` valida (≥50 strumenti) + retry + **NON cacha chain sospetta** (si solleva → frontend tiene l'ultimo buono); (2) `_send` ingoia `ConnectionAborted/Reset/BrokenPipe`; (3) `loadRisk` guard: se dati degradati (no strike / OI tutto-zero) **skip redraw** invece di svuotare. Verificato: chain completa (97 strike), guard nell'HTML servito. **NB: NON serve cambiare libreria (Plotly va bene): il problema era dati/connessione, lib-agnostico.**

## 📋 STRATEGIE OPZIONI — MODIFICHE FUTURE DA TESTARE (analisi 2026-06-23, post-review codice 04b)

Valutazione di 5 strategie proposte vs il codice attuale (`04b_vol_paper.py` = long/short straddle ATM su `edge=log(rv_pred/var_iv)`, ±0.25, direction-neutral, hold-to-expiry, max-1, testnet) e vs l'evidenza del progetto. **Dato chiave dal forward test (n=7): PnL −0.038 BTC, hit-rate 29%, trade quasi tutti LONG che perdono → la IV è stata > RV realizzata (VRP positivo) → il long-vol sanguina, il lato che paga è probabilmente lo SHORT-vol.**

**✅ DA TESTARE (in ordine di priorità):**
1. **[ORA] Chiudere il forward test straddle a 30 trade** (gate pre-registrato 2026-06-12). È l'unico edge con prior validato (momento PARI = livello vol); il resto è prematuro finché n<30. Nessuna modifica al codice.
2. **[ALTA] Short strangle / short-vol come 2° braccio** del forward test. Razionale: il VRP su BTC è strutturalmente positivo + il NN sa stimare quando IV è sovrastimata; il segnale precoce (long perde) punta lì. Implementazione = estensione MINIMA: in `pick_straddle` scegliere 2 strike OTM invece dell'ATM + flag struttura. ⚠ Rischio di coda illimitato → su capitale reale serve hedging (#sotto); su testnet ok per validare il SEGNALE. Il gate già previsto (battere always-short-vol) lo misura.
3. **[FUTURA, effort ALTO] IVS relative-value (VAE sulla superficie di vol).** L'UNICA alpha genuinamente nuova (mispricing tra strike/scadenze ≠ scommessa sul livello; informazione del mercato opzioni, NON in dead-zone). Hai i dati (`data/iv/chain/*.parquet`). ⚠ Costo d'esecuzione brutale (bid-ask opzioni mangia il mispricing relativo) + superficie BTC efficiente sui liquidi + il **testnet NON valida l'esecuzione** (liquidità simulata) → validabile solo come segnale, non come PnL netto. Progetto a sé.

**❌ NON FARE (classe già falsificata / prematuro):**
- **Wrapper DRL: OPHR multi-agente (#1), Trans-DDQN (#2), Soft-Actor-Critic deep hedging (#5).** DRL è la classe KILLED (2026-06-16: DRL+ES, News-RL): instabile, non tocca il vincolo (l'informazione, non il modello), il segnale RV-vs-IV è già una regola semplice che funziona meglio.
- **Stop-line dinamica sullo straddle (#2):** rompe l'hold-to-expiry (settlement deterministico, zero rumore exit) e reintroduce la *macchina di realizzazione path-dipendente* che ha già DISTRUTTO l'edge sul direzionale (Fix ①②). Prematuro a n<30.
- **Deep hedging (#5):** è esecuzione, non alpha; presuppone un edge short-vol validato (non c'è). Solo DOPO #2, e prima un baseline **rule-based delta-band**, non DRL.

**Caveat trasversale (già in STATUS):** la liquidità testnet è simulata → valida bene i segnali **hold-to-expiry** (settlement al delivery price, esecuzione irrilevante), MALE le strategie execution-sensitive (#3 arb, hedging frequente). Per il PnL netto reale di quelle servirebbe mainnet/Alpaca o dati L2 opzioni a pagamento.

## 🟢 2026-06-22 sera — punti 2/3/4 eseguiti (parity FIX, health, gate B1)

**Punto 2 — fail parity RISOLTO (era un bug del TEST, NON di produzione, NON serviva retrain).** Diagnosi: il test `test_live_training_parity` carica `ps` da `models/itransformer` e costruisce ENTRAMBI i path (FeatureAssembler + FeatureBuilder) dallo stesso buffer/scaler → il diff di 40 NON era data/scaler drift. Le feature divergenti erano TUTTE **time-semantic** (`momentum_x_funding`, `price_vs_ma200m`, `dist_atl/ath_30d`, `momentum_30d`, `session_position`, `funding_*`); le **bar-semantic** (VP) avevano diff 0.0 → firma di un mismatch `interval_minutes`. Causa: il fixture costruiva `FeatureBuilder` SENZA `interval_minutes` → default **1**, mentre `FeatureAssembler` usa `ps.interval_minutes`=**60** (1h). Bug latente dell'era 1m (default 1 matchava), emerso ora che l'npz rigenerato ri-attiva i test prima skippati. **Fix:** 1 riga nel fixture (`interval_minutes = ps.interval_minutes`). → parity **5/5**, suite **151 passed, 1 skipped, 0 failed**. ⚠ Conferma: BLOCKER #1 parity in PRODUZIONE è INTATTO (live usa ps.interval, training usa config interval, entrambi 1h); era solo il test stale.

**Punto 3 — forward test sano.** 3 processi vivi (poller IV, recorder L2, `04b --execute`), log freschi. Nessun intervento.

**Punto 4 — GATE B1 order-book L2 PRE-REGISTRATO (scritto PRIMA di modellare).**
- **Stato dati:** 7 file `data/orderbook/l2_features_*.parquet` (06-16→22), ~31k righe a 5s ma **con buchi PC-off** → ~43h di book effettivo. **INSUFFICIENTE** (servono settimane-mesi di copertura). ⚠ Il recorder NON è 24/7 → i buchi PC-off sono un limite strutturale: valutare host always-on o accettare copertura parziale. Schema già raccolto: mid/microprice/tilt, spread_bps, imbalance L1-20, depth bande 5-50bps, OFI best, top-25 raw.
- **Integrazione (quando i dati bastano):** nuovo stream nel `FeatureBuilder` con vincoli — (a) **parity live↔training** BLOCKER #1 (l'assembler deve ricostruire le L2 dallo stesso book in tempo reale → serve un buffer book live), (b) **causalità** (solo book ≤ t), (c) aggregazione a barra (last/mean/sum per tipo, da definire).
- **GATE (cost-aware, split DATA-RICH non media k-fold, embargo ±h):** ① predicibilità: Spearman(μ, ret_fwd) su test data-rich con **embargo ±h**, |ρ| > 2/√n_eff E coerente val→test (no flip segno — lezione separation-index/ic-metric); ② tradabilità: backtest a costi reali (~26 bps round-trip a 1m) → Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 a ENTRAMBI i costi (riuso gate probe pivot); ③ negative control: feature-selection L2 batte il floor di permutazione (embargo ±h).
- ⚠ **Prior:** il muro a 1m è il COSTO (~26bps ≫ effetto ~1.5bps price/volume). L2 è informazione NUOVA → può avere magnitudine maggiore, ma deve battere il costo; se l'edge esiste ma è sub-costo → valutare uso come TIMING/esecuzione, non alpha standalone. Esito negativo = documentato.

## ✅ DECISIONE 2026-06-22 — consolidamento sul PASS iTrans, distill = prossima cosa

Dopo la catena STEP 0 → retune → A (k-fold) → b1 (gate vs HAR) → verifica anti-bug, deciso con l'utente: **il modello vol di produzione resta l'iTransformer 5-seed PASS** (`models/itransformer`, già validato due volte OOS: single-split test QLIKE 0.257/ratio 0.70 **e** k-fold sui fold data-rich, batte HAR). Il forward test `04b` continua a maturare verso i 30 trade. **Distill 5-seed RIMANDATO** (prossima cosa da fare).

**▶️ PER RIPRENDERE IL DISTILL 5-SEED (prerequisiti esatti — NON ci sono scorciatoie):**
- ⚠ **Serve riaddestrare TUTTI E 3 gli archi a `--n-ensemble 5` sui dati correnti, in dir sandbox PULITA.** Motivi: (1) N-HiTS/TCN+Mamba a 5-seed su `log_rv` **non esistono** (i vecchi `models/{nhits,tcnmamba}` erano duplicati 1m, eliminati il 06-12; oggi esistono solo 1-seed); (2) l'iTrans 5-seed PASS è **inutilizzabile come teacher** — `config.json` SENZA `best_val_loss/spearman/da` e SENZA `history.json` (addestrato prima del fix 06-21) → `compute_teacher_weights`/`_select_best_teacher` ricadrebbero su uniforme/skip; inoltre è su scaler dati-vecchi (1.4376 vs 1.4343 npz corrente).
- **⚠ LEZIONE seed≠fold (per non riconfondersi):** `best_model_0..4.pt` = **5-seed ensemble** (stessi dati, init diversi, mediati a inference, caricati da `EnsembleModel.load` → è il modello di produzione). `wf_fold1..5_best.pt` = **5 checkpoint del walk-forward** (fette temporali diverse, throwaway diagnostici, NON caricabili come ensemble). Entrambi danno "5 file/arch" ma sono cose diverse. Oggi gli arch sandbox hanno 1-seed (`best_model.pt`) + 5 `wf_fold*` (da A), **nessun 5-seed**.
- **Comando:** dir sandbox nuova, copia canonical `pipeline_state.pkl`, `$env:QUANTSYS_MODELS_ROOT=<sandbox>; python run_all.py --distill --n-ensemble 5` (Fase 2a addestra i 3 a 5-seed, Fase 2b scoring target-aware `log_rv`→0.65/0.35/0.00, retrain student). Giudizio `dev_vols_qlike.py --arch <student>` su val. **GATE:** QLIKE_student ≤ 0.97×0.343 (battere iTrans-standalone) E ratio HAR ≤0.95. Fermare `04b` (GPU).
- **⚠ PRIOR BASSO (atteso FAIL):** iTrans è già il QLIKE-migliore sul regime di produzione; il mismatch **val_nll↔QLIKE** (tcnmamba miglior val_nll 0.155 ma iTrans miglior QLIKE 0.343) fa sì che lo scoring eleggerebbe tcnmamba teacher e distillerebbe iTrans *verso il basso*; err cross-arch ρ~0.83 → riduzione varianza modesta. Da fare solo per chiudere la domanda "combinare aiuta?" con un NO documentato.
- **Pulizia opzionale:** `models_distill_vol/` (1-seed + wf checkpoint di oggi) è throwaway/gitignored → eliminabile.

---

## 🕒 Storico 2026-06-22 (COMMIT distill + prereq + GATE pre-registrato)

## 🟢 CICLO 2026-06-22 — committato il distill target-aware + rigenerato il dataset npz + gate pre-registrato (fan-out 3 subagent)

Su richiesta utente, eseguite 3 cose in parallelo (fan-out).

**1) COMMIT del ciclo distill target-aware → `73fef66`** `feat(distill): scoring teacher TARGET-AWARE per la linea volatilita (log_rv)`. 12 file (+256/−40), working tree pulito, **NON pushato**. `pytest tests/test_distillation.py` 5/5 verde pre-commit. ⚠ Nel diff era bundlato anche un 2° concern dello **stesso ciclo**: nuovo helper **`models_root()`** (`quantsys/utils/__init__.py`) gated su env **`QUANTSYS_MODELS_ROOT`**, propagato a `distillation.py`/`ensemble.py`/`run_all.py`/`02_train.py`/`dev_vols_qlike.py` — sostituisce gli hardcoded `Path("models")` per far girare un distill vol in **sandbox isolata** senza clobberare il modello live di `04b`. Default byte-identico al comportamento precedente. Dichiarato nel body del commit.

**2) SALUTE 3 PROCESSI DETACHED → 3/3 SANI** (nessun rilancio). `01c_iv_poller` (parquet sul grid 10-min), `04b_vol_paper --execute` (tick 16:01 `edge=+0.538 → HOLD`), `01d_orderbook_recorder` (tick 5s `err=0`). Conteggio LOGICO verificato via ParentProcessId (6 OS proc = 3 logici, lezione venv stub+worker), no duplicati.

**3) PREREQUISITI DISTILL-VOL → PRONTI (NON è l'esperimento, solo la prep reversibile).**
- Config verificata read-only: `target_type: log_rv`, `interval: 1h`, `forecast_horizon: 30`. ✅
- **Dataset npz RIGENERATO** (era assente dal cleanup 06-12): `01_download_data.py` (`QUANTSYS_ARCH=lstm` per NON toccare `models/itransformer/`) + `scripts/vol/dev_vols_macro_append.py`. Risultato: `data/lstm_dataset.npz` (3.2 GB), `X_train (51364,120,104)`, split 51364/6420/6421, `X_macro_* (·,90)`, target z-scored (`target_scale=1.4343` = IQR del log_rv, log-ret avrebbe IQR ~1e-3 → conferma log_rv). Canonical `models/pipeline_state.pkl` riscritta: interval=1h, h=30. **`models/itransformer/` confermato INTATTO** (tutti i file ancora 2026-06-10 20:25 — forward test di `04b` salvo).
  - ⚠ **Trappola incontrata e risolta:** lanciato in chain PS `cmd1 *> log; if ($?){cmd2}`, lo step macro_append era stato **saltato** (npz senza `X_macro_*`): è il gotcha PS 5.1 di CLAUDE.md — `01_download` logga su stderr, sotto `*>` PowerShell marca `$?`=`$false` (NativeCommandError) anche con exit 0. Rilanciato `dev_vols_macro_append.py` in foreground → OK. **Lezione: per chain dipendenti da uno script Python che logga su stderr, NON affidarsi a `if ($?)`; sequenziare a mano o usare la shell bash.**
- **Dir sandbox create:** `models/distill_vol/`, `results/distill_vol/` (gitignored). NB la garanzia vera dell'isolamento è l'env `QUANTSYS_MODELS_ROOT`, non queste dir.
- **NESSUN training avviato** (è l'esperimento gated, non la prep; + contesa CUDA coi 3 processi live).

### 🎯 GATE PRE-REGISTRATO — esperimento DISTILL-VOL (scritto PRIMA di girare, zero iterazioni a risultato visto)

**Obiettivo:** lo student distillato multi-teacher su `log_rv` batte l'iTransformer 5-seed corrente in **QLIKE (split VAL)**? Protocollo: val-first, dir-isolata, esito negativo documentato.

**Isolamento (obbligatorio, protegge il modello live di `04b`):** `$env:QUANTSYS_MODELS_ROOT="E:\quantsys_project\models_distill_vol"` → l'intero path train→distill→judge legge/scrive in sandbox, MAI in `models/`. Seedare la sandbox copiando la canonical `pipeline_state.pkl` (log_rv) negli arch dir. Pre-condizione dura: **i 3 processi live IN PAUSA prima di QUALSIASI training** (RTX 2070S 8GB → OOM/contesa); arch **sequenziali**.

**STEP 0 — KILL-CHECK ECONOMICO PRIMA del run completo (correlazione errori cross-arch sulla vol):** allena **1 seed** di nhits e tcnmamba su `log_rv`, forward dei due + iTransformer su VAL, Pearson degli errori per-campione `(pred−y)` a coppie.
- **KILL gate:** se corr media ≥ **0.99** (come ≈0.995 sul direzionale) → ensembling/distill riduce varianza ≈0 → **STOP, FAIL in STATUS, niente GPU sprecata.**
- **PROCEDI** solo se almeno una coppia ≤ **0.97** (diversità sfruttabile).

**STEP 1–3 (solo se Step 0 passa):** train teacher nhits+tcnmamba `--n-ensemble 5` in sandbox (sequenziali) → `run_all.py --distill --multi-teacher` (scoring già target-aware: `log_rv`→pesi 0.65/0.35/0.00) → giudice `dev_vols_qlike.py` su VAL (`QUANTSYS_VOLS_SPLIT=val`).

**PASS/FAIL (split VAL):** QLIKE(student) ≤ **0.97 × QLIKE(iTrans 5-seed)** (≥3% di miglioramento) **E** il gate assoluto del giudice resta valido (QLIKE_NN ≤ 0.95·QLIKE_HAR e < naive). Campione: `n_val ≥ 0.95·len(t_val)` (~6.4k → no fragilità small-sample). **FAIL → scrivilo in STATUS, NON toccare il test, NON promuovere in `models/itransformer/`. PASS su val → test UNA sola volta** con la stessa soglia.

**Trappole da onorare:** bug membri stale (`n=1` aggiorna solo `best_model.pt`, mai i `best_model_{0..4}.pt` → usare `--n-ensemble 5` in dir PULITA); `nhits.yaml`/`tcnmamba.yaml` hanno hyperparam era-1m (rischio overfit su 65k 1h → rivedere lr/dropout prima); val-only fino a gate; test single-use.

**Prior onesto (pre-dichiarato):** sul direzionale cross-arch err≈0.995 → ensembling inutile; SI 06-16 dà la vol-bucket come unico asse non condannato ma con headroom incerto. **Aspettativa: Step 0 è il punto di kill più probabile** — documentare un FAIL lì è l'esito economico e corretto.

**▶️ AZIONE ESATTA ALLA RIPRESA (per girare l'esperimento):** (1) mettere in pausa i 3 processi live; (2) `QUANTSYS_MODELS_ROOT` sandbox + seed canonical; (3) **STEP 0** (1 seed nhits+tcnmamba, corr errori) e applicare il kill-gate; (4) se passa, STEP 1–3 val-first; (5) qualunque esito → scriverlo qui; (6) rilanciare i 3 processi (blocco rosso 2026-06-16). **Working tree:** dataset/dir sono gitignored; `STATUS.md` modificato (questa sezione) NON committato.

## 🟢 STEP 0 KILL-CHECK ESEGUITO 2026-06-22 → PROCEED (diversità cross-arch sulla vol)

Eseguiti i 3 archi 1-seed su `log_rv` in sandbox (`QUANTSYS_MODELS_ROOT=models_distill_vol`, GPU dedicata, `models/itransformer` LIVE confermato intatto) → `scripts/vol/step0_xarch_corr.py --split val`.
- **Correlazione errori cross-arch (val):** iTrans|N-HiTS 0.776 · iTrans|TCN-Mamba 0.815 · N-HiTS|TCN-Mamba 0.887 → **mean 0.826, min 0.776**. Gate KILL≥0.99 / PROCEED≤0.97 → **PROCEED** (non ucciso). Report `results/vols/step0_xarch_corr_val.json`.
- **Significato:** sul VOL gli archi disaccordano (ρ_err ~0.83), in netto contrasto col direzionale (≈0.995, ensembling inutile). Rafforza la tesi momenti PARI: la vol è predicibile OOS (batte HAR 30%) **e** ha diversità cross-arch → distill/ensemble ha headroom potenziale.
- **Best val_nll 1-seed:** TCN+Mamba 0.155 < iTrans 0.183 < N-HiTS 0.189 (TCN+Mamba miglior singolo, bias minore +0.11).
- ⚠ **CAVEAT (non sovrainterpretare):** 1 seed · 1 split · N-HiTS/TCN+Mamba su yaml era-1m → overfit (gap val −0.13÷−0.20). Parte della "diversità" può essere overfit-indotta (ognuno overfitta diverso), NON segnale complementare. Il test vero resta il gate **QLIKE k-fold** pre-registrato (sopra). PROCEED = "non morto a priori", non "l'ensemble aiuterà OOS".
- **Infra usata:** Tier 1 (modulo `quantsys/model/vol_metrics.py` QLIKE condiviso + fold-metric QLIKE/branch N-HiTS in `02b` + checkpoint sandbox-aware) e Tier 2 (`step0_xarch_corr.py`) — codice NON ancora committato. ⚠ Rigenerando l'npz sono emersi 2 fail in `test_live_training_parity.py` (prima skippati per npz assente): stato↔dati stale (`models/itransformer` vol-1h vs raw_candles ri-scaricati), NON regressione del Tier 1; 04b live sano. Known-issue da investigare a parte.
- **Prossimo (decisione aperta):** STEP 1–3 = walk-forward **k-fold distill** (harness `02b` pronto) — run GPU pesante (5 fold × multi-arch sequenziali), richiede di ri-fermare `04b`. Oppure fermarsi e valutare.

## 🟢 A — PURGED K-FOLD per-arch sul vol ESEGUITO 2026-06-22 (post-retune regolarizzazione)

Dopo il retune (commit `df4b5fb`: N-HiTS/TCN+Mamba allineati al regime vol iTransformer + `4647194` log per-epoca) → `02b_walkforward_validate.py` per i 3 archi (5 fold effettivi, embargo 168=1sett, 1-seed, sandbox `models_distill_vol`, fold-metric QLIKE). Esito QLIKE cross-fold (split = ogni fold held-out, n=9172/fold):

| arch | QLIKE mean ± std | σ/μ | range fold1→5 |
|---|---|---|---|
| **TCN+Mamba** | **0.364 ± 0.062** | **0.169** | 0.37/0.43/0.43/0.30/0.28 |
| N-HiTS | 0.401 ± 0.080 | 0.200 | 0.41/0.51/0.45/0.35/0.28 |
| iTransformer | 0.493 ± 0.160 | 0.324 | 0.61/0.71/0.52/0.30/0.32 |

- **Retune efficace:** N-HiTS/TCN+Mamba ora più stabili e MIGLIORI di iTransformer; **TCN+Mamba domina** (best + più stabile), coerente con STEP 0 (val_nll 0.155).
- **Trend temporale:** QLIKE cala fold1→fold5 (effetto expanding-window: più dati nei fold tardivi / vol più predicibile di recente). σ/μ 0.17–0.32 = stabilità moderata.
- ⚠ **GAP da colmare prima di concludere skill OOS:** `02b` WF **NON calcola HAR-RV per fold** → da questo run NON si può dire "batte HAR OOS cross-fold" (solo che la QLIKE NN è stabile e TCN+Mamba domina). Il gate vs-HAR richiede la baseline HAR per fold.
- **Prossimo (decisione aperta):** (b1) aggiungere HAR-per-fold al WF per il confronto gate; oppure (b2) k-fold ensemble/distill — testa se *combinare* batte il best single (0.364): con err cross-arch ~0.83 la riduzione varianza è modesta, da verificare se supera TCN+Mamba da solo. Report: `results/{arch}/walkforward_metrics_log_rv.json`.

## 🟢 b1 — HAR-RV PER-FOLD (gate vs HAR cross-fold) ESEGUITO 2026-06-22

Script CPU-only `scripts/vol/wf_har_baseline.py` (helper `build_har_frame`/`har_fold_qlike` in `quantsys/model/vol_metrics.py`): stessi 5 fold, HAR OLS fit-per-fold sui timestamp di train, eval sull'held-out, confronto coi QLIKE NN già salvati. Gate per-fold QLIKE_NN ≤ 0.95·QLIKE_HAR. HAR QLIKE medio cross-fold = **0.430** (naive 0.800).

| arch | NN medio | ratio NN/HAR | batte HAR | verdetto |
|---|---|---|---|---|
| **TCN+Mamba** | **0.364** | **0.863** | **4/5** | ✅ PASS |
| N-HiTS | 0.401 | 0.948 | 3/5 | ~ borderline |
| iTransformer | 0.493 | 1.170 | 3/5 | ❌ FAIL |

- **TCN+Mamba batte HAR OOS cross-fold (~14%)**, decisivo nei fold data-rich (ratio 0.75–0.83) → la skill vol **sopravvive al purged k-fold**.
- **Effetto fold-1 (strutturale):** TUTTI gli archi falliscono il fold 1, il più antico (expanding window data-starved: NN data-hungry < HAR a 3 param). NON model-failure; la skill emerge coi dati (fold 3→5 sempre PASS).
- **iTransformer 1-seed FAIL** ma era PASS a **5-seed** single-split (0.257, ratio 0.70) → NON apples-to-apples (seed-variance + fold early lo penalizzano).
- **Implicazione b2 (distill):** TCN+Mamba da solo PASSA; iTrans 1-seed è membro DEBOLE (sopra HAR) → ensemble equal-weight lo trascinerebbe, il distill quality-weighted dovrebbe pesare TCN+Mamba ma battere 0.364 con un membro debole è incerto → **b2 meno attraente**. Alternativa più sensata: **(c) TCN+Mamba 5-seed standalone** = modello vol robusto/deployabile.
- Report: `results/vols/wf_har_baseline_1h.json`. Codice b1 (vol_metrics HAR helpers + wf_har_baseline.py) NON committato.

## 🔎 VERIFICA 2026-06-22 (richiesta utente) — i risultati k-fold sono REALI (nessun bug), ma la MEDIA cross-fold è la statistica SBAGLIATA

Indagine sul perché iTrans/N-HiTS sembravano "cattivi" nel k-fold mentre settimane fa erano il PASS vol.
- **Check 1 — verità identica:** target NN (npz invertito `y·s+c`) == verità HAR (RV `rv_fwd` dai raw candle) a **rel diff 8e-7** → il confronto NN-vs-HAR NON è viziato (entrambi giudicati sulla stessa ground-truth).
- **Check 2 — giudice validato `dev_vols_qlike.py` su val (data-rich), stessi 1-seed sandbox:** iTrans QLIKE **0.343 ratio 0.917 PASS** · tcnmamba 0.356 ratio 0.951 (FAIL di un soffio) · nhits 0.396 FAIL. Riconcilia coi fold data-rich del k-fold (iTrans fold4/5 = 0.30/0.32 ≈ val 0.343). HAR val 0.374 ∈ [fold4 0.40, fold5 0.35] → anche l'HAR riconcilia.
- **CONCLUSIONE: nessun bug.** iTransformer NON è peggiorato — è il **MIGLIORE sullo split data-rich** (= regime di produzione), coerente col PASS 5-seed originale (test 0.257, ratio 0.70). Il "FAIL" nel k-fold MEAN era un **ARTEFATTO**: iTrans è il più **data-hungry** → pessimo sui fold 1-2 (expanding-window data-starved, ~fold_size campioni) che trascinano la media; tcnmamba è più robusto a pochi dati → media migliore, ma iTrans vince dove conta.
- **⚠ LEZIONE METODOLOGICA (da non dimenticare):** la **media cross-fold over-penalizza i modelli data-hungry** sui fold early, che NON rappresentano la produzione (dove hai sempre la storia piena). Statistica corretta per la produzione = **val/test split + fold data-rich**, NON la media su tutti i fold. Il k-fold resta utile per la *stabilità*/robustezza, non per il ranking assoluto.
- **Implicazione retrain:** il rationale "tcnmamba 5-seed perché domina il k-fold" **CADE**. iTrans è già il 5-seed PASS e vince sul regime di produzione. tcnmamba 5-seed resta **opzionale** (2° modello per ensemble/distill; oppure se interessa l'MSE-log, dove tcnmamba batte iTrans 0.63 vs 0.83; oppure la stabilità σ/μ 0.169 vs 0.324). Doppia conferma OOS del filone vol: single-split 5-seed + k-fold sui fold data-rich.
- Report: `results/vols/qlike_report_1h_val.json` (ultimo arch giudicato), `results/vols/wf_har_baseline_1h.json`.

## 🕒 Aggiornamento precedente: 2026-06-21 (DISTILL TARGET-AWARE per la linea VOLATILITÀ/opzioni)

## 🟢 DISTILLATION RIADATTATA AL TARGET VARIANZA (`log_rv`) — FATTO 2026-06-21

Su richiesta utente (/goal): riadattare gli script della distill a funzionare sulla **versione attuale per le opzioni** (target `log_rv` = varianza). **Problema scientifico individuato:** lo scoring teacher era cablato sul direzionale (`40% val_loss + 35% Spearman + 25% directional_acc`). Sulla linea vol la **directional accuracy è il segno della varianza-vs-mediana**, che NON è un segnale tradabile (lo straddle di `04b_vol_paper.py` è **direction-neutral**, monetizza RV vs IV = momento PARI). Pesare la selezione/blend dei teacher per la dir_acc della varianza è scorretto e poteva scegliere il teacher sbagliato.

**Cosa è stato fatto (engineering completo, suite verde):**
- **`quantsys/model/distillation.py`:** nuovo helper **`teacher_score_weights(target_type)`** = single source of truth dei pesi `(val_loss, spearman, dir_acc)`. Direzionale (`ret`/`log_rs_ratio`) → `(0.40, 0.35, 0.25)`; **vol (`log_rv`) → `(0.65, 0.35, 0.00)`** (dir_acc azzerata, ribilanciata su val_loss/QLIKE+Spearman). `compute_teacher_weights(all_archs, target_type=...)` lo usa.
- **`run_all.py`:** `_select_best_teacher(all_archs, target_type=...)` (il selettore reale, legge `history.json`) ora usa lo stesso helper; `phase_distill` legge `target_type` da config e lo passa. Log diagnostico stampa i pesi attivi.
- **`scripts/02_train.py`:** passa `target_type` a `compute_teacher_weights`; **persiste in `config.json`** `best_val_loss`/`best_spearman`/`best_da` (metriche di val alla best-val epoch). ⚠ Prima NON erano persistite → `compute_teacher_weights` (che le legge) ricadeva silenziosamente su pesi UNIFORMI: ora il blend multi-teacher è davvero quality-weighted.
- **`tests/test_distillation.py`** (5 test nuovi): vol azzera dir_acc, direzionale la mantiene, blend ignora dir_acc su vol, blend preferisce val_loss più basso su vol, default back-compat. **Suite full: 138 passed, 8 skipped, 0 failed** (era 133; +5).
- **Doc sincronizzate:** CLAUDE.md (§REGOLE), TEORIA.md, README.md (×2 lingue ×2 sezioni), AVVIO.md (Fase 2b). MODEL_IMPROVEMENTS.md non aveva la sezione scoring → nessun edit.

**Verifica integrazione:** `_select_best_teacher` su dati sintetici — arch con dir_acc 0.95 ma val_loss peggiore: vince sotto `ret` solo se domina anche val/sp; sotto `log_rv` la dir_acc contribuisce 0 (score isola val_loss+Spearman). Il resto del path distill era già target-agnostic (`transfer_output_heads`, `distillation_loss_*`, `generate_multi_teacher_predictions` gestiscono quantile+t_student).

**⚠ NON ESEGUITO (è solo l'engineering del metodo, non un esperimento):** il distill vol vero richiede i prerequisiti già noti (STATUS 2026-06-12): (1) rigenerare l'npz (`01_download_data.py`+`scripts/vol/dev_vols_macro_append.py`, eliminato col cleanup 06-12); (2) addestrare nhits/tcnmamba su `log_rv` (mai fatto; i loro yaml hanno hyperparam era-1m → rischio overfit, il tuning vol lr3e-5/drop0.3 è solo in itransformer.yaml); (3) **dir SEPARATE, non in-place** (`models/itransformer/` è il modello del forward test `04b` in corso); (4) **pre-registrare il gate prima di girare** (QLIKE val del distillato vs iTrans 5-seed corrente; misurare PRIMA la correlazione cross-arch degli errori sulla vol — se ≈0.995 come sul direzionale, il blend è inutile a priori). NON girare training in parallelo a poller/vol_paper (contesa CUDA).

**Working tree:** modificati `quantsys/model/distillation.py`, `run_all.py`, `scripts/02_train.py`, `tests/test_distillation.py` (nuovo), CLAUDE/TEORIA/README/AVVIO + STATUS. NON committato.

## 🕒 Aggiornamento precedente: 2026-06-18 (CAFN coordinatore + DASHBOARD opzioni)

## 🧪 CAFN — Causal Attention Flow Network (coordinatore dei 3 modelli) — COSTRUITO + PRE-REGISTRATO 2026-06-18

Costruito su richiesta utente (/goal) un layer di **coordinamento a monte** dei 3 modelli (iTransformer, TCN-Mamba, N-HiTS): la CAFN estrae un **latente causale** dal tensore feature e i 3 modelli si allenano **in contemporanea** su quel segnale (loss congiunta end-to-end). **Engineering COMPLETO e verificato; scientificamente è un PROBE pre-registrato, INERTE di default — NON ancora valutato su dati reali** (dataset npz assente dal cleanup 06-12).

**Cosa è stato costruito (tutto isolato, zero impatto su production/parity):**
- **`quantsys/model/cafn.py` — `CausalAttentionFlowNetwork`**: filtro denoising (gate per-feature sigmoide) → proiezione + pos-emb → stack di blocchi self-attention a **maschera STRETTAMENTE causale** (t vede solo ≤t) → latente `[B,T,d_latent]`. `forward(x, extra=None) -> (latent, causal_penalty)`. **Penalità causale = REGOLARIZZATORE** (prossimità: penalizza attenzione sul passato lontano; stabilità: penalizza salti del pattern fra t adiacenti) — NON causalità do-calculus/Granger (dichiarato nel docstring). Canale `extra` opzionale per feature Deribit forward-collected (futuro, gated).
- **3 forward parity-safe**: aggiunto kwarg `latent=None` a `QuantiTransformer`/`QuantNHiTS`/`QuantTCNMamba` (concat sull'asse feature in cima). `latent=None` → path **bit-identico** alla chiamata legacy (test di parità verde su tutti e 3 → vincolo BLOCKER #1 preservato). I modelli aumentati si costruiscono con `n_features += d_latent`.
- **`scripts/02d_cafn_joint_train.py`**: loop end-to-end, 1 optimizer su CAFN+3 modelli, loss = Σ_arch MSE-mu + λ·penalità. `--smoke` (CPU/synthetic, validato), fail-fast su npz assente. Output ISOLATO in `models/cafn/` + report `results/cafn/`. **Gate val-first** integrato (baseline NO-CAFN, `latent=None`, stessi modelli ri-inizializzati).
- **`config/cafn.yaml`** (overlay opzionale, blocco `cafn`, non letto dalla pipeline production) + export in `quantsys.model`.

**GATE PRE-REGISTRATO (scritto PRIMA di girare sul reale):** PASS sse CAFN-congiunto batte il baseline NO-CAFN (stessi modelli/seed/epoche) di **≥3% MSE-mu su val per ≥2 dei 3 modelli**. FAIL → CAFN-coordinatore KILL (flag inerte, documentato). Zero iterazioni a risultato visto; test split solo a gate val PASS.

**⚠ 3 BLOCCHI scientifici dichiarati e come risolti:**
1. **Dati Deribit grezzi come input storico = lookahead + dataset inesistente** (greche/book/IV forward-collected, giorni di storia). → CAFN si addestra sul **tensore canonico 104-feature** (storia 2019→oggi); Deribit entra solo come canale `extra` opzionale futuro.
2. **"invece del dataframe" romperebbe parity/PipelineState.** → integrazione **additiva** (concat, `latent=None`→identico), contratto forward intatto.
3. **Training simultaneo 3 modelli su 8GB = OOM** (il repo impone 3-arch sequenziale). → default piccoli + `--smoke`; caveat memoria nel trainer.

**PRIOR ONESTO:** è una variante di CLASSE-MODELLO; il progetto ha ripetutamente mostrato che ciò NON sposta il soffitto direzionale OOS (anti-corr val→test, cross-arch err≈0.995, distill 06-06 OOS≡baseline, baseline lineari senza skill → "il limite è l'informazione, non il modello"). Aspettativa pre-dichiarata: **FAIL del gate**. Lo smoke synthetic ha dato 1/3 win → FAIL (atteso, è rumore: serve solo a validare la macchina).

**Verifica:** `pytest tests/test_cafn.py` 11/11 + suite completa **133 passed, 8 skipped** (era 122; +11 CAFN, zero regressioni — parità preservata). Smoke joint end-to-end OK su CUDA (1.7s). **Working tree non committato.**

**▶️ AZIONE ALLA RIPRESA (se si vuole valutare il probe):** (1) rigenerare il dataset npz (`01_download_data.py` [+ `dev_vols_macro_append.py` se target vol]); (2) `python scripts/02d_cafn_joint_train.py --epochs <N>` (val-first, NON sul test); (3) applicare il GATE pre-registrato e, qualunque l'esito, scriverlo qui (KILL documentato se FAIL). NB: NON girare in parallelo a poller/vol_paper (contesa CUDA).

## 🟢 REFACTORING DASHBOARD — `06_dashboard.py` ora è un terminale opzioni Deribit (2026-06-18)

Riscritta **da zero** `scripts/06_dashboard.py`: da dashboard ML (metriche backtest/portafoglio, segnali live, pulsante "Aggiorna" pipeline) a **piattaforma istituzionale per l'analisi delle opzioni crypto**, single-file HTTP + SPA Plotly.js, GPU-free e **indipendente dalla pipeline ML**.
- **Data layer Deribit** (REST pubblico, no-auth, cache TTL per-chiave anti thundering-herd): `get_index_price` (spot BTC, ttl 4s), `get_book_summary_by_currency` (chain opzioni completa: mark_iv/mark/bid/ask/OI/volume/underlying_price, ttl 8s), `get_volatility_index_data` (DVOL, ttl 60s). Riusa i pattern di `01c_iv_poller.py` (parse strumento, expiry 08:00 UTC).
- **Motore Greche** (`bs_greeks`, Black-Scholes forward-measure, r=0, convenzione USD, vettoriale numpy; scipy.stats.norm con fallback math.erf): Δ delta, Γ gamma, ν vega (per +1% vol), Θ theta (per giorno), ρ + prezzo teorico. Calcolate **live sull'intera chain** ad ogni snapshot.
- **Analytics:** `build_surface` (IV interpolata su griglia comune moneyness K/F=[0.6,1.6]×41 × giorni, no extrapolazione→NaN), `build_term_structure` (ATM IV vs giorni), `build_chain_table` (call|put a doppio lato per expiry), `build_risk` (OI per strike call/put, max-pain vettoriale O(n²), Greche aggregate pesate OI, P/C ratio).
- **SPA** (Plotly.js CDN): header risk live (spot/DVOL/ATM IV 30d/OI/vol/PCR + conn status) + 3 tab: **Volatility Surface** (3D + smile selezionabile + term structure), **Option Chain** (tabella call/put con Greche, ATM evidenziato), **Risk & Greeks** (OI by strike a **profilo divergente** call ▲/put ▼ + linea **Net OI** sull'asse PRIMARIO (stessa unità delle barre — era su `yaxis2` overlay ma con `uirevision` collassava l'autorange delle barre al refresh → fix) + zoom auto sulla banda liquida + toggle contratti BTC/notional USD, spot/max-pain; Greche aggregate, P/C pie). Auto-refresh ~12s, gzip, auth opzionale constant-time. **Interazione grafici (2026-06-18):** i 5 grafici 2D (smile/term/oi/greeks/pcr) usano `PL_CFG_2D` (`scrollZoom:true`, `doubleClick:'reset'`) + layout `dragmode:'pan'`, `autosize:true`. Niente box-zoom "finestra"; zoom per-asse con la rotellina sopra l'asse (Y su/giù, X sx/dx), pan col drag, doppio-click reset. La **superficie 3D resta a interazione piena** (`PL_CFG`, orbit/zoom). Modifiche tentate (tutte committate): (1) `dragmode:false`→`dragmode:'pan'` (con `dragmode:false` Plotly disabilita ANCHE il drag sugli assi); (2) Net OI da `yaxis2` overlay all'asse primario (stessa unità delle barre); (3) `Plotly.react`→**`Plotly.newPlot`** per i 2 grafici con barre (`plot-oi`, `plot-greeks`), ipotesi: react non ridisegna le trace `bar` ai re-render; (4) rimosso `uirevision` (sospettato dello stesso bar-drop). Interazione assi: `scrollZoom:true` + `dragmode:'pan'` + doppio-click reset.

**🔴 PROBLEMA APERTO — NON RISOLTO (2026-06-18):** nel grafico **Open Interest by Strike** le barre call/put compaiono all'apertura ma **dopo poco (≈refresh 12s) spariscono**, restano solo le linee (Net OI) e gli `shapes` (spot/max-pain). ⚠ **Discrepanza diagnostica da chiarire:** il test headless Playwright/Chromium contava 198 path-barra persistenti su 2 refresh → PASS, MA **l'utente continua a vederlo sparire nel browser reale** → la verità è l'osservazione dell'utente, il test headless NON replica il fault. Ipotesi ancora da verificare per la prossima sessione: (a) il test contava i path SVG ma in-browser le barre potrebbero avere width/opacity→0 (path presenti ma invisibili) — misurare bounding-box/fill, non il count; (b) intermittenza legata al timing reale del fetch `/api/risk` (il test girava su rete forse diversa); (c) interazione con `autosize`/resize o con `hovermode:'x unified'`; (d) il fault potrebbe scattare solo dopo interazione utente (zoom/hover) prima del refresh. PROSSIMO PASSO: riprodurre col vero browser dell'utente (screenshot/devtools), misurare gli attributi reali dei `<path>` barra (non il conteggio), prima di tentare altri fix.
- **Endpoint JSON:** `/api/summary|surface|term|expiries|chain|risk`. Config: nuovo `dashboard.options_currency` (BTC|ETH); rimosse `subprocess_timeout_sec`/`log_lines_maxlen` (pipeline-runner eliminato).
- **Rimossa** l'app React orfana `dashboard/` (`git rm` + working tree: era un esperimento non cablato in pipeline/doc, sarebbe stata una 2ª dashboard in conflitto).
- **`run_all.py`:** `--only-dashboard` non avvia più live feed/analisi (la dashboard è disaccoppiata → "nessun calcolo" come da help); testo `phase_dashboard` riscritto. `00_check_setup.py` aggiornato (`scripts/06_dashboard.py` al posto del jsx rimosso).
- **Doc sincronizzate:** README (nuova sezione bilingue + albero), AVVIO (sezione dashboard riscritta + flag + checklist nuova-arch step 6), TEORIA (rimosso "dashboard" dai consumer di regime_probs), scripts/README, config.
- **Verifica:** AST OK; unit offline greche (ATM Δ call/put diff=1.0, Γ/ν>0, Θ<0) + analytics su chain sintetica JSON-serializzabili; **live end-to-end Deribit reale** (spot 62.8k, DVOL 41.9, 960 strumenti, 12 expiry, surface 12×41, greche sane); server HTTP smoke (HTML, `/api/summary`, gzip negoziato, `/api/risk`, 404). **NB: il backtest ML (`03_backtest.py`) continua a scrivere `results/{arch}/dashboard_results.json`** — non più letto dalla dashboard ma artefatto valido.
- **Working tree:** non committato.

---

## 🕒 Aggiornamento precedente: 2026-06-16 (AUDIT + B1 order-book L2 + riorganizzazione repo)

## 🚀 OTTIMIZZAZIONE P1 (Volume Profile bincount) — IMPLEMENTATA + VERIFICATA 2026-06-16

Eseguito audit diagnostico delle inefficienze computazionali dei percorsi caldi (forward ensemble, FeatureBuilder, loop backtest/walkforward, MC GJR-GARCH, regime refit). **Implementato il solo fix P1** (impatto ampio + bit-equivalente); l'audit completo e lo scaffolding di test sono stati RIMOSSI su richiesta utente (conclusione preservata qui, analisi rigenerabile — coerente col workflow "conclusione in STATUS, codice/doc throwaway via").
- **P1 (FATTO):** `quantsys/features/__init__.py` `_vp_single` — istogramma VP ora `np.bincount(idx_arr, weights=vol, minlength=vp_bins)` invece di `np.zeros`+`np.add.at` (innermost loop ×3 scale 60/240/1440) → ~10–40× sul singolo hotspot dell'engine feature; si paga al build dataset (01) **e** a ogni build feature live. Firma invariata, numericamente identico (≤1 ULP, accumulo float64). `idx_arr` già `np.clip(0, vp_bins-1)` → prerequisito non-negatività di bincount soddisfatto.
- **Verifica (fan-out 2 subagent):** (a) audit consumer → SICURO (call chain unica `_vp_single`←`_volume_profile`←`build`; 5 consumer production tutti via API pubblica: `01_download_data`, `01_update_data`, `99_replay`, `04b_vol_paper`, `04_live_signals`; nessun golden bit-exact, nessun altro `np.add.at` in production); (b) suite completa post-fix → **122 passed, 8 skipped, 0 failed** (baseline invariata; il test di parità usato per validare è stato rimosso col cleanup) + smoke VP end-to-end via `build()` OK.
- **Restanti P2–P7 (NON implementati, rigenerabili dall'audit):** P2 MC `forecast.py` copie host↔device per step (live only). P3 `03_backtest.py` `mdd_stats` loop Python→`np.maximum.accumulate`. P4 buffer rank/quiet `list.pop(0)`→`deque` (flag già FALLITI OOS). P5 `features.build` 3× `df.copy()` (985/1058 deliberati, 965 valutabile). P6 `ensemble.py:360` tensore pesi per `__call__` (cache). P7 `02b` `nanpercentile` per fold (NON bit-per-bit). Esclusi come falsi positivi (CLAUDE.md): refit expanding O(t) del regime, `predict_proba` sequenziale, batch-inference già batchata, `sliding_window_view` già zero-copy.
- **Working tree:** `quantsys/features/__init__.py` + `STATUS.md` modificati, NON committati.

## 🗂️ RIORGANIZZAZIONE REPO 2026-06-16 (script per linea, motore condiviso invariato)

Decisione con l'utente: la vol (unico PASS OOS) è la linea pubblicabile su GitHub; il direzionale serve al paper. **NON due repo/cartelle separate** (sarebbe duplicare `quantsys/` → trappola duplicati stale): un repo, motore condiviso, separazione solo degli script non numerati per linea.
- **Spostati** (`git mv`): `dev_vols_qlike.py`/`dev_vols_rs_judge.py`/`dev_vols_macro_append.py` → `scripts/vol/`; `paper_01_dir_baselines.py` → `scripts/research/`. ⚠ Fix `Path(__file__).resolve().parents[1]→parents[2]` in tutti e 4 (un livello più in profondità); lanciare dalla root (path relativi CWD-relativi). Smoke OK (import quantsys risolto, `paper_01` gira dalla nuova posizione).
- **Spine numerato `00→99`: invariato** (è la *fase*, non la linea; `02_train` è condiviso, target da `config.features.target_type`). Nessun rinumero → zero riferimenti rotti in `run_all.py`.
- **Nuovo `scripts/README.md`:** mappa bilingue script→linea (shared / vol / direzionale).
- **Doc sincronizzati** (path `scripts/vol/`·`scripts/research/`): CLAUDE.md (§NOMENCLATURA + STATO NOTO), README.md (albero), AVVIO.md, TEORIA.md, config/default.yaml, docs/MODEL_IMPROVEMENTS.md, docs/paper/{OUTLINE,RESULTS_MAP}.md. STATUS storico lasciato com'è (log). `.gitignore` già copre data/models/results/logs/secrets → nessun rischio di pubblicare artefatti.
- **Pubblicazione futura (NON ora):** vol-only su GitHub via README che mette la vol in primo piano + eventuale export `git subtree split` (mai branch divergente); pesi vol PASS via release asset (`models/` è gitignored).

## 🟢 AUDIT COMPLETO 2026-06-16 — esito + B1 AVVIATO

**Salute codice: VERDE.** `pytest tests/` → **122 passed, 8 skipped, 0 failed** (gli 8 skip sono attesi: test parity live + 3 in `test_recent_fixes` dipendono dall'npz eliminato col cleanup 06-12, non regressioni). Working tree pulito, tutto committato (harness `04c` in `03d15cf`). Guard scientifici critici (z-score denorm, interval/horizon config↔state, membri stale ensemble) coperti dai regression test e verdi.

**⚠ LEZIONE INFRASTRUTTURALE — venv = stub+worker (NON sono processi duplicati).** Su questo host `.venv\Scripts\python.exe` è uno **stub** che delega all'interprete base (Python 3.12 di sistema): **1 processo logico = 2 processi OS** (stub `.venv` = padre, worker `python.exe` di sistema = figlio; uccidere il worker fa cadere lo stub). Quindi `Get-Process python` che mostra 4 processi con `04b`+`01c` ×2 = **1 poller + 1 vol_paper**, NON duplicati. Diagnosi corretta: confrontare `ParentProcessId` (figlio sys → padre .venv) e contare i processi LOGICI, non quelli OS. (In questa sessione ho inizialmente mis-diagnosticato una "race condition" e riavviato i 2 processi: restart pulito, nessun danno — forecast append-only dedup, vol_paper senza posizione aperta — ma evitabile.)

**Forward test vol-paper:** 2 trade settlati (#1 −0.01001, #2 +0.00026 BTC), n=2 ≪ 30 → nessuna conclusione (rumore atteso). ⚠ **Velocità campione: ~2 trade in 4 giorni → i 30 del gate arrivano in ~7-8 settimane** (entry rara: `edge>0.25` scatta di rado). I 2 processi (poller IV + vol_paper) **rilanciati e sani** in questa sessione (coppia `.venv`, log vivi = `logs/quantsys_*.log` NON i redirect `iv_poller.log`/`vol_paper.log`).

**🚀 B1 ORDER-BOOK L2 — AVVIATO 2026-06-16 (deciso con l'utente: GO solo se API gratuite bastano → confermato).** Binance spot `/api/v3/depth` è **free, no-auth, 1000 livelli/lato, weight 50/call** (a 5s = 600 weight/min ≪ 1200) — nessun servizio a pagamento. Come per la IV, lo **storico L2 NON è gratis (Tardis)** → raccolta FORWARD, è il collo di bottiglia temporale.
- **`scripts/01d_orderbook_recorder.py`** (famiglia dati, gemello di `01c`): loop REST snapshot, append-only ATOMICO + dedup su `timestamp`, output `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/giorno). Persiste **feature microstrutturali derivate** (mid, microprice+tilt, spread_bps, imbalance L1/5/10/20, depth cumulata in bande 5/10/25/50 bps, total qty, **OFI best-level** cross-tick Cont-Kukanov-Stoikov) **+ top-25 livelli raw/lato** come list-column (rete di sicurezza: feature future ri-derivabili senza ri-raccogliere → schema NON bloccato). Modi `--once`/`--seconds N`/`--symbol`/`--levels`.
- **Smoke + avvio PASS:** feature sane, list-column round-trip OK, OFI valorizzato dal 2° tick del processo persistente (NaN in `--once` per design: stato `_PREV` per-processo). **Recorder detached attivo** (cadenza 5s) — ⚠ NON è un servizio, rilanciare dopo riavvio (Start-Process in AVVIO.md).
- **Prossimo passo B1 (NON ora):** dopo settimane di accumulo, integrare le feature L2 come nuovo stream nel `FeatureBuilder` (vincoli parity live↔training di BLOCKER #1 + causalità) e **pre-registrare** un gate direzionale OOS prima di modellare.

## 📚 VALUTAZIONE 3 PAPER 2026-06-16 (fan-out) → tutti KILL come alpha; codice di test RIMOSSO

Valutati 3 paper su richiesta utente; conclusioni preservate qui, **scaffolding di test rimosso** (cleanup richiesto dall'utente — coerente col workflow "conclusione in STATUS/memoria, codice rigenerabile via").
1. **DRL + Evolutionary Strategies** (CIT 2024): KILL secco. Ottimizzatore ES al posto di SGD su **OHLCV puro**, zero costi, mesi cherry-picked, nessuno Sharpe. Nessuna informazione nuova → dentro la dead-zone già provata. Nessun codice prodotto.
2. **News-Aware Direct RL** (arXiv:2510.19173): KILL il metodo (DDQN/GRPO instabili; il loro miglior risultato è il modello SENZA news, zero costi, single-regime bull). Sopravvive solo l'**idea**: news = terza fonte di informazione NUOVA (accanto a L2/IV) → eventuale data-clock futuro (poller CryptoPanic, gate-first sul vol). Nessun codice prodotto.
3. **Separation Index** (Kalhor arXiv:2406.17083): KILL come alpha (criterio in-sample su target direzionale). Usato come **diagnostico** ha dato il risultato netto sotto. Script `paper_02_separation_index.py` + output rimossi post-valutazione (ricostruibili da [[session-2026-06-16-separation-index]]).

**Risultato SI (preservato — figura per il paper "price+volume enough?"): con embargo temporale ±h nel 1-NN** (rimuove il gemello adiacente col target forward sovrapposto, stessa trappola di [[ic-metric-fix]]): **DIREZIONE test 0.701 raw → 0.495 embargo = floor permutazione 0.504** (separabilità direzionale OOS NULLA); **VOL-BUCKET test 0.826 raw → 0.672 embargo ≫ floor** (il livello di vol resta separabile OOS). Operazionalizza la dicotomia momenti pari/dispari. **Risposta alla domanda "SI feature-selector per alzare l'OOS del distill":** sui RENDIMENTI inutile a priori (separabilità=floor); sulla VOL unico caso non condannato ma headroom incerto (il vol-model già batte HAR del 30%) → non prioritario. Lezione metodologica blindata: ogni metrica di predicibilità su target forward sovrapposto richiede embargo ≥ h (vale anche per le future feature L2 di B1).

## ▶️ RIPRESA 2026-06-16 — 3 PROCESSI DA TENERE VIVI (rilanciare dopo ogni riavvio)

Dopo un riavvio del PC i 3 processi detached muoiono — rilanciarli (path `.venv` ESPLICITO per evitare l'ambiguità `python`→interprete):
```powershell
$py = "E:\quantsys_project\.venv\Scripts\python.exe"
Start-Process -WindowStyle Hidden -WorkingDirectory "E:\quantsys_project" -FilePath $py -ArgumentList "scripts/01c_iv_poller.py"
Start-Process -WindowStyle Hidden -WorkingDirectory "E:\quantsys_project" -FilePath $py -ArgumentList "scripts/04b_vol_paper.py","--execute"
Start-Process -WindowStyle Hidden -WorkingDirectory "E:\quantsys_project" -FilePath $py -ArgumentList "scripts/01d_orderbook_recorder.py"
```
Salute: contare i processi LOGICI (ParentProcessId, vedi lezione sopra) = atteso 3; crescita `data/iv/atm_30h.parquet` (~144 righe/g), `results/vol_paper/forecasts.parquet` (~24 righe/g), `data/orderbook/l2_features_*.parquet` (~17k righe/g a 5s); log vivi in `logs/quantsys_*.log` (i più recenti per mtime). ⚠ Buchi temporali nei calendari per le ore di PC spento (attesi).

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
