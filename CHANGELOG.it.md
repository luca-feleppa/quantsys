🇮🇹 Italiano · [🇬🇧 English](CHANGELOG.md)

# QUANTSYS — Changelog

Ordine cronologico inverso (la voce più recente in alto). Le "Iterazioni" 1-10 sono il blocco direzionale storico (1m); dal pivot 1h in poi le voci sono datate per sessione e tracciate in dettaglio in `STATUS.md` (lab notebook append-only). Questo file riassume i milestone; `STATUS.md` è la fonte canonica.

---

## 2026-09-02 — `exec_diag` a N gambe senza toccare il record a due

Gli aggregati di `exec_diag.jsonl` in `scripts/04b_vol_paper.py` escono da una funzione pura (`exec_diag_aggregate`) calcolata sul corpo (prime due gambe); i campi dell'intera struttura compaiono solo oltre le due gambe. `hedge_dry_run.py` isola il corpo via `body_idx`. Test di replay sulle 1236 righe storiche con uguaglianza esatta (`tests/test_exec_diag_multileg.py`). Non deployato: `04b` invariato sul VPS.

**Leva `--adaptive` in `04b`, inerte di default** — regola d'entry per banda DVOL senza segnale NN: sopra soglia short straddle daily (macchinario v1), sotto soglia short iron butterfly a ~7 giorni (4 ordini market, ali prima, regola di completamento a 120 s con flatten inverso e record `incomplete`), settlement a 4 gambe con formula a 2 gambe invariata, parametri congelati obbligatori ed espliciti, `adaptive.jsonl` solo con il flag. 10 test (`tests/test_adaptive_structure.py`), suite 504 passed. Non deployato.

## 2026-08-29 — Le architetture diventano una pagina, e il denominatore smette di vivere a memoria

**`docs/architetture.html`** — dodici viste interattive delle architetture, ricavate leggendo i `forward` e non la documentazione: pipeline dei dati, iTransformer, TCN+Mamba, N-HiTS, CAFN, MoE/MoU, testa di output, più il cablaggio interno di attenzione, convoluzione, stato e decomposizione. I diagrammi sono **dati** (riga + corsia) e il layout li dispone, quindi i blocchi non possono sovrapporsi per costruzione; le forme dei tensori si derivano dai parametri modificabili in pagina, così si vede cosa cresce con `T` e cosa no. Bilingue **in un unico file** come il resto della documentazione: ogni stringa visibile è una coppia `{it, en}`, la cornice statica si popola da un dizionario e il toggle IT/EN non azzera lo stato. ⚠ Il ramo italiano è stato diffato **campo per campo** contro l'originale (testo, topologia degli archi, corsie, passi) e tutte le 12 viste sono state renderizzate in **entrambe** le lingue contro un DOM finto: in questo tipo di lavoro un errore non produce un guasto visibile, e una traduzione che riscrive in silenzio il testo esistente sarebbe indistinguibile da un miglioramento.

**`THEORY.it.md` §12.2** — la sezione nomina quattro denominatori (naive, HAR-RV, HAR-CJ, HAR-C) perché la baseline è stata resa progressivamente più forte da tre gate successivi, e la distinzione che conta stava a metà di un paragrafo di quaranta righe. Ora c'è un **pannello di riconciliazione** in testa: una riga per affermazione, con denominatore, numeri e ruolo. Il **gate** pre-registrato del 2026-06-10 usa **HAR-RV** ed è congelato; il **claim** pubblicato usa **HAR-C** dal C3. Due cose rese esplicite: il `p ≤ 4.3·10⁻⁴` citato nel README è misurato contro **HAR-CJ** (gate C2), non contro HAR-C; e gli estremi della banda sono **val e test**, non un intervallo di confidenza. Nel `README.it.md` la distinzione entra **dove il claim viene fatto**, non solo nel blocco di metodologia duecento righe più sotto. ⚠ **Zero numeri nuovi**, verificato meccanicamente: tutte e 25 le cifre del pannello compaiono già altrove nel file.

## 2026-08-18 — Il comparatore MFIV chiude FAIL: il wedge è grande ma quasi costante, e il rango non lo vede

**Gate pre-registrato MFIV-comparatore v2 SCATTATO e CHIUSO in giornata: FAIL**, run one-shot manuale alla prima sessione con la condizione ③ soddisfatta. Campione appaiato `n = 41` expiry daily (soglia 40), verificato ex-ante in sola lettura prima di lanciare: il contatore della routine leggeva **42**, il campione reale era **41** (l'ultima expiry qualifica dai tick di oggi ma settla domani). ① `Δρ = ρ_MFIV − ρ_ATM = +0.0343` contro +0.05 richiesto: FAIL. ② segno non coerente sulle due metà cronologiche (−0.0015 / +0.0610): FAIL. ③ `n ≥ 40`: PASS. L'esito era **predetto dalla pre-registrazione stessa**: a wedge costante i due Spearman sono identici per invarianza di rango, e il wedge misurato è grande (mediana log-varianza 0.2005) ma poco disperso (interdecile 0.1733) — il gate misurava solo la sua variazione temporale, che a `n=41` (SE ≈ 0.16) non aggiunge potere di ranking. **Conseguenza pre-dichiarata applicata:** il comparatore live di `04b` resta **ATM IV**, MFIV resta colonna diagnostica permanente, item chiuso, nessuna pre-reg v3. **Descrittivo non-gating dovuto comunque — break-even short-vol ri-stimato:** la MFIV prezza la varianza **+22.2%** sopra l'ATM interpolata (+10.55% in volatilità), quindi il pareggio a VRP = 0% del backtest storico corrisponde a una varianza realizzata **18.2% sotto** il var-swap rate corretto: il cuscinetto del braccio short-vol è più ampio di quanto l'ATM mostrasse. Non è PnL in più — `04b` vende lo straddle ATM e incassa il premio ATM — ma è un argomento per strutture che vendono una porzione maggiore della strip. Ritirato dalla routine il contatore `--count-only`: il suo unico consumatore era il one-shot, ora esaurito. Dettaglio e numeri: `THEORY.it.md` §12.2, record macchina `results/vols/mfiv_comparator_report.json`.

---

## 2026-08-13 — Wind-down chiuso: la finestra pulita era larga zero

**`--hedge --hedge-band 999 --hedge-conv raw` rimossi dall'unit: `04b` gira di nuovo come v1 unhedged, che è il design di produzione.** Precondizione verificata prima di toccare qualunque cosa: flatten della leg residua alle **08:01:34 UTC** (`h_usd_after: 0.0`, fill 63836.5, fee 8.19e−05 BTC), `hedge_state.json` assente su VPS e in canonico. ⚠ **Il `reason` è uscito `structure_changed`, non `settled`/`expired` come pre-annotato ieri — ed è il terzo arm della stessa guard, non un'anomalia** (`pos is None` → *settled*, `expired` → *expired*, `position_key != pos_key` → *structure_changed*): la previsione era incompleta perché ricavata da due rami su tre. ⚠ **La causa è la conferma retroattiva della correzione di ieri:** allo **stesso tick** delle 08:01:34 la 13AUG si è settlata **e** `04b` ha aperto la 23ª struttura (BTC-14AUG26-64000, edge +0.501), quindi `position.json` non è **mai** stato vuoto e il ramo `settled` non poteva scattare. **La finestra pulita post-settlement non dura "poche ore": è larga zero** — il book viene rimpiazzato nello stesso tick in cui si liquida, e la banda di wind-down non era la strada comoda ma **l'unica**. **Inerzia della banda 999 misurata:** nessun evento nel ledger dopo il flatten, contatore hedge-attive fermo a 22. **Deploy verificato sui due lati:** prima, md5 dell'unit in vigore **identico** a `HEAD` (nessuna deriva) con processo ripartito alle 00:30:03 UTC dal timer, cioè ancora a banda 999 — la conferma che *il commit da solo non cambia la produzione*; dopo, `ExecStart` senza `--hedge`, md5 remoto = locale, `is-active`, e nel log di boot **nessun `V2 HEDGE ATTIVO`** né riconciliazione, primo tick regolare (`edge=+0.628 → HOLD`). Il commento sopra `ExecStart` passa da stato transitorio a **ragione definitiva**, col vincolo di ordine riscritto in forma **simmetrica**: vale anche per chi un domani riaccendesse il flag e volesse rispegnerlo. **Contatore hedged ritirato** dalla routine di sessione (resta il solo contatore delle leg opzioni, unico numero che dice se `04b` sta eseguendo): un contatore si ritira quando il suo **ultimo consumatore** è esaurito — non alla chiusura del gate (l'11/08 serviva ancora a provare che la banda non aprisse hedge nuovi), e non dopo (un numero senza consumatore invita a confrontarlo con una soglia che non è più la sua). ⚠ **Refresh candele verificato per-expiry e deliberatamente NON applicato:** valeva **+1** (la sola 13/08; la 14/08 matura domani), ma a 12/40 il contatore E1 non ha consumatore, la soglia cade lo stesso intorno al 9-10/09 e ogni refresh è una scrittura su un file di dati di produzione — resta obbligatorio **una volta sola**, prima del run one-shot di E1 stadio 2. **Nota di metodo:** lo smoke della routine con `2>&1 | Select-String` è morto su `NativeCommandError` — in PS 5.1 la gotcha già nota per `Tee-Object` vale per **qualunque** pipe con `2>&1` su uno script che lancia un exe nativo; sostituito da un test mirato (estrazione del blocco via AST ed esecuzione standalone), che è anche il test giusto perché isola la modifica.

---

## 2026-08-12 — Wind-down della leg hedge: la finestra pulita non esiste, la si costruisce

**Il vincolo di ordine della disattivazione non era un evento singolo: si ripresenta a ogni ciclo.** La voce dell'11/08 prescriveva «attendere il settlement delle 08:00 UTC, verificare il flatten, poi riavviare senza `--hedge`»: presupponeva che dopo il flatten lo stato hedge restasse vuoto fino al riavvio. **Falso.** Il flatten della 21ª posizione è avvenuto alle 08:01:34 (`reason: "settled"`, `h_usd_after: 0`), ma alle **10:01** `04b` ha aperto la 22ª (strike 63500, expiry 13AUG) e l'ha hedgiata, ribilanciando alle 15:01. La finestra pulita è durata **2 ore** e la sua ampiezza **non è controllabile**: dipende da quando `|book_delta|` supera la banda, e può essere il primo tick utile. Un piano che richiede di cronometrarla fallisce a intermittenza, e ogni fallimento costa un altro giorno di churn su una leva già FALLITA. **Soluzione presa dal codice, non dal calendario:** in `maybe_hedge` il ramo di flatten (`pos is None or expired or position_key != pos_key`) gira **prima** del check di banda (`if abs(book_delta) < band_eff: return`) → una banda arbitrariamente grande **disabilita apertura e ribilanciamento** ma **lascia attivo** il flatten a settlement/expiry/cambio struttura. Wind-down in due passi: oggi `ExecStart` → `--hedge-band 999` (deploy VPS, restart 15:34:54 UTC); domani, dopo il settlement del 13AUG e senza limite superiore di orario, `ExecStart` torna al solo `--execute`. La leg residua sarà quindi chiusa **dallo stesso codice che ha scritto le altre 20**, e `hedge_ledger.jsonl` resta internamente consistente — l'alternativa (ordine perp piazzato a mano) avrebbe lasciato un `open` senza il suo `flatten`, cioè un record *sbagliato* invece che *incompleto*. **Stato transitorio messo nel file tracciato e non in un drop-in systemd:** un drop-in non è ricostruibile da git (lo stato deployato per ~17h non esisterebbe in alcun artefatto versionato) e, se dimenticato, sovrascrive in silenzio qualunque unit futura; costo della variante tracciata: un commit di una riga, e un passaggio in meno. ⚠ **Non è un cambio di parametro a giudizio in corso:** il congelamento `band=0.30`/`conv=raw` valeva per la durata del gate, chiuso ieri, e il campione giudicato (n=20, ultimo settlement exp 11AUG) è frozen in `results/vols/hedged_vs_unhedged.json`; la 22ª posizione sta fuori dal campione per costruzione. ⚠ **Ri-lanciare oggi il giudice con lo stesso `--since` restituirebbe 22 settlement, non 20:** il record è il JSON archiviato, il gate è chiuso e non si ri-giudica a campione allargato. **Inerzia verificata empiricamente:** al riavvio guard `band=999.0`, nessun warning di riconciliazione, tick di bootstrap `edge=+0.280 → HOLD` e leg hedge girata **senza produrre eventi** (ledger fermo, `updated_ts` invariato). **E1 stadio 2: da 8 a 12/40** dopo `01_update_data.py --candles-only` (+97 candele). ⚠ **La regola `T+2h` è stata misurata, non ereditata, e la lettura ingenua del log la sbaglia di un'ora:** il log stampa «Caricate 98 candele → 15:00 UTC» ma il merge ne scrive **97** (la barra in formazione è scartata lì), quindi `raw_candles.parquet` finisce alle 14:00 e `hourly_close()`, che ne scarta un'altra, dà close usabile fino alle **13:00 UTC** girando alle 15:22 — due righe perse, non una. Margine effettivo sul vincolo della 12/08 (close 11:00): **2h**. ⚠ **Il monitor stampava `no_rv: 5` ma il guadagno era +4:** la quinta expiry è la 13/08, il cui tick di decisione esiste già ma la cui finestra RV si chiude domani — verificato **per-expiry ex-ante**, prima di scrivere sul file di produzione, secondo la regola che ha già risparmiato due scritture inutili il 07/08 e il 10/08.

---

## 2026-08-11 — Il gate delta-hedged chiude FAIL: l'hedge fa il suo lavoro, le fee se lo mangiano

**Gate pre-registrato hedged-vs-unhedged (v2, `04b --hedge`) SCATTATO e CHIUSO in giornata: FAIL 2/3**, run one-shot su decisione esplicita alla prima sessione con la condizione ③ soddisfatta. Campione forward `n = 20` settlement con hedge attivo (19/07 → 11/08; esclusa per pre-dichiarazione la posizione già in essere all'attivazione). ① **varianza: PASS con margine** — `var(hedged)/var(unhedged) = 0.447`, cioè **−55.3%** contro il −40% richiesto. ② **costo: FAIL** — drag `mean(h) − mean(u) = −0.001312 BTC = −0.647·SE` contro un budget pre-registrato di −0.25·SE, **2.6× la soglia**. ③ n ≥ 20: PASS. **Decomposizione** (identità `hedged − unhedged = gross − fee − funding`, residuo 1.7e−18): **fee perp 75.9%** (−0.000996 BTC/trade su 76 ribilanciamenti, 0.000262 BTC per evento), gross perp 23.2% (mean-zero by design, a n=20 non distinguibile da zero), **funding 0.9%**. ⚠ **L'ipotesi pre-dichiarata era giusta sulla variabile e sbagliata sulla componente:** il 12/07 il rischio era scritto come «fee di churn **+ funding**», col funding indicato come la grande incognita *mai misurata sulla serie* — misurato, vale **meno dell'1%** del drag. Contro-fattuale a fee nulle: drag −0.156·SE → il gate sarebbe **PASS 3/3**. ⚠ **Qualificazione che non sposta il verdetto:** la ② è un **budget**, non un test di significatività — il t appaiato è **−1.051** (hedged migliore in 7/20 trade), quindi è dimostrato che l'hedge **non compra PnL** e che il costo di realizzazione a size 1 contratto consuma un multiplo del budget, **non** che il drag sia negativo in popolazione. **Conseguenze pre-dichiarate applicate:** `--hedge` FALLITO, la **v1 unhedged resta il design paper di produzione**; sizing v2 (HAR-q90) e A7 greeks-aware **non** si sbloccano, erano condizionati al PASS. **B2 (hedge = purificatore del VRP) muore sul costo di realizzazione, non sulla teoria** — la ① lo prova. Terza volta nel progetto che un effetto misurato non sopravvive ai costi di transazione, dopo il muro dei costi a 1m e il KILL net-of-cost dell'IVS. ⚠ **Due scarti di unità trovati prima di lanciare:** il contatore di routine leggeva `21` (posizioni con ≥1 hedge eseguito, inclusa quella **ancora aperta**) contro le `20` **settlement** della pre-reg — terza variante dello stesso errore su questo contatore dopo «eventi ≠ posizioni» del 26/07; e `--since` è stato verificato **sample-invariante** su tutta la finestra ammissibile, così la sua scelta non è un grado di libertà. ⚠ **La disattivazione ha un vincolo di ORDINE non previsto dalla pre-registrazione:** `hedge_state.json` ha una leg perp aperta di +19.320 USD e `maybe_hedge`/`reconcile_hedge_state` girano **solo** sotto `--hedge` → riavviare senza il flag adesso lascerebbe quel perp **nudo e mai flattenato**. Ordine corretto: attendere il settlement del 12/08 08:00 UTC, verificare il flatten nel ledger, poi riavviare. Verificato che la disattivazione **non perturba** E1 e MFIV: senza flag il path è la v1 bit-identica e la leg opzioni è invariata.

---

## 2026-08-10 — Un file senza produttore, e il motivo per cui resta senza

**`data/raw_candles_1m_l2.parquet` non avrà uno script produttore, e la ragione è una proprietà del dato, non una priorità.** La voce era in coda da quattro sessioni come "lavoro non fatto". Misurato il peso reale prima di deciderla: il file compare in **tre soli** consumatori, **nessuno** dei quali gira sul VPS — `04b` legge IV, macro e candele **orarie**, quindi il braccio live ne è indipendente. L'unico consumatore eseguito a ogni sessione lo legge come *coda* di `hourly_close()`, e siccome l'1m è **indietro** rispetto alle orarie il suo contributo misurato è **0 righe**. Da qui la decisione: le kline a 1 minuto sono **storiche e ri-scaricabili indefinitamente**, quindi — al contrario del recorder L2, forward-only e irrecuperabile se si ferma — il "ritardo" non è un debito che matura ma un download non ancora fatto, e il file scaricato oggi è identico a quello scaricato fra sei mesi. Un produttore permanente terrebbe aggiornato un file che nessun path operativo legge: **quando servirà, il download si decide allora, sulla finestra che serve.** Il monitor continua a misurare la copertura e non stampa più un rimedio. Registrato anche quanto varrebbe, per non doverlo ri-derivare: **237 ore L2 già registrate ma prive di target 1m** (774 con ≥360 snapshot/ora, 537 sotto il tetto) ≈ **+128 punti di valutazione**, `n_eval` da 289 a ~420. **B1 stadio 1 resta chiuso** — il vincolo del 31/07 non erano i dati (`N_MIN=240` già superato con 289) ma il controllo positivo, HAR-C ristimata sulla finestra corta contro una naive a zero parametri. **Emerso di lato:** `VolForecaster._bootstrap` estende e ri-persiste `raw_candles.parquet` in modo gap-aware, quindi **sul VPS le candele orarie avanzano da sole** a ogni bootstrap — «estendere le barre è un atto, non un automatismo» vale per la copia **locale**, quella che alimenta contatori e giudici a casa.

---

## 2026-08-06 — Un campione che sembrava fermo, e il guard che protegge solo da un lato

**Il contatore del campione confermativo E1 stadio 2 non era a 0: era a 6.** Due difetti sovrapposti. ① Lo `0/40` in continuità **non era una misura** — era il valore di apertura del 01/08 riportato in avanti per cinque giorni, perché quel contatore non era nella routine di sessione e nessuno lo rileggeva. ② Anche misurandolo avrebbe detto **2, non 5**: `raw_candles.parquet` era fermo al 02/08 e `realized_rv` restituiva `None` su ogni expiry successiva. ⚠ La combinazione è peggiore della somma: ① garantisce che nessuno guardi, ② che chi guardasse veda un numero basso e **plausibile**. Diagnosi verificata, non congetturata: i **7 tick di decisione esistono tutti** per ogni expiry dal 01/08 al 07/08, mancava solo la serie dei close. Dopo `01_update_data.py --candles-only` (66.435 → 66.530 barre; `features.parquet`, `lstm_dataset.npz`, scaler e `PipelineState` **non toccati**) il conteggio passa a **6/40**, finestra 01/08 → 06/08, **1 osservazione al giorno dall'apertura**. **Nessun dato perso:** i tick vivono in `forecasts.parquet`, che si merge append-only dal VPS — il ritardo era di **osservabilità**, non di raccolta, ed è **recuperabile**, a differenza di un vintage macro promosso dentro un campione aperto. ETA invariata: n≥40 al **09-10/09**.

**Copertura del file barre 1m nel monitoraggio — continuità del recorder ≠ disponibilità del target.** Il check L2 misura la continuità del **recorder**; il nuovo check misura se esiste il **target** con cui quelle ore verrebbero giudicate (il giudice B1 costruisce la RV da rendimenti a **1 minuto**). Il campione utile è il **minimo fra i due**, e nessuno dei due contatori lo diceva — la stessa forma del difetto E1 di oggi: un numero che cresce mentre qualcos'altro lo tappa. Primo run: **87.217 barre, 2026-06-01 → 2026-07-31, 6 giorni di L2 registrati senza target 1m**. ⚠ Il messaggio **qualifica l'orizzonte**, perché senza sarebbe un allarme sul gate sbagliato: il tetto vale per le analisi a target 1m (B1 a h=3, proxy pin-close), mentre `n_eff` a h=30 usa le barre **orarie** e non è toccato. ⚠ Il file ha **tre consumatori e zero produttori** — nessuno script del repo lo genera o lo estende, fu acquisito a mano — quindi il monitor **misura soltanto** e dichiara che l'estensione è manuale via `quantsys.data.fetch_klines`, invece di stampare un rimedio inesistente. Scrivere il produttore resta un lavoro a sé, non fatto.

**`-RefreshCandles`: l'estensione delle barre diventa un atto esplicito, non un default.** Audit del codice: automatizzare il refresh sarebbe **meccanicamente sicuro** — no-op se non c'è nulla di nuovo (uscita *prima* della scrittura), dedup che tiene la riga **esistente** quindi la storia non si riscrive mai, mai la barra in formazione, scrittura atomica, e il file **non arriva mai al VPS** (l'unico trasferimento casa→VPS è la macro). Resta comunque un flag per tre ragioni scritte nel repo: (i) il blocco ③ è descritto come *off-path e a scrittura zero*, ed è una categorica su cui un lettore futuro si appoggia; (ii) l'estensione fa avanzare lo staleness B7, che a ≥168 barre lancia da solo il refresh incrementale del regime — **oggi 95/168**; (iii) **decisiva**, con l'estensione automatica **congelare i dati diventa impossibile**: l'invariante *«candele/npz/regime_probs non si toccano fino a chiusura gate»* passerebbe da «non fare nulla» a «ricordarsi `-SkipMonitor`», la forma esatta della promozione macro avvenuta per automazione il 31/07. Corollario di provenienza: lo split è una **frazione del conteggio righe**, quindi ogni barra appesa sposta i confini train/val/test al prossimo rebuild — il vintage del dataset diventerebbe funzione di *quante sessioni sono state aperte*, cioè non più dichiarabile. ⚠ Qualificazione onesta: **nessuna pre-reg aperta congela oggi le candele**, quindi l'automazione non avrebbe violato nulla di corrente. Forma adottata: passo ②bis **dopo** il check B7, così l'eventuale refresh del regime parte al **prossimo** avvio e le due scritture restano separate. Fra i gate aperti **solo E1 legge le barre** — verificato nei giudici.

**Contatore E1 aggiunto al blocco ③ della routine — e il modo ovvio di farlo era sbagliato.** Il guard `n<40 → NO_RUN` protegge **solo sotto soglia**: a n≥40 uno `--stage 2` nudo calcola le tre condizioni, stampa il verdetto e scrive il report, cioè avrebbe fatto scattare il run confermativo **per automazione invece che per decisione**, il giorno in cui la soglia cade. Aggiunto `--count-only` al giudice E1 (si ferma alla conta a **qualunque** n: nessuna statistica, nessun report), con un test che lo verifica su un campione costruito **sopra** la soglia — sotto soglia il test non proverebbe nulla, perché lo passerebbe anche il comando nudo. Costruire il pannello **è** la conta, ma `x`, `y` e le statistiche non vengono né stampate né scritte: si vede **quante** osservazioni ci sono, mai **quanto valgono**. ⚠ Il refresh delle candele **non è automatizzato di proposito**: il blocco stampa fin dove arriva la serie dei close e avvisa oltre 6h di ritardo, ma il rimedio resta un comando esplicito — automatizzarlo trasformerebbe il blocco ③ da *off-path e a scrittura zero* a *scrive un file di dati a ogni sessione*, la stessa classe di automatismo che il 31/07 ha prodotto una promozione macro non decisa. Suite **491 passed / 1 skipped**; `START.it.md` §5.3 e `scripts/README.it.md` allineati.

---

## 2026-08-05 — La banda pubblicata scritta alla precisione del suo artefatto

**Banda ri-espressa: `−23% ÷ −32%` → `−22.42% ÷ −31.65%`. Stessi numeri, nessuna ri-misura, zero GPU** — sono i rapporti NN/HAR-C della coppia canonica (val `0.7757926`, test `0.6834522`) scritti per esteso invece che arrotondati al punto. I due estremi si muovono per ragioni **diverse**: quello **inferiore** era **stale** (−23% è l'arrotondamento del −22.6% misurato da C2 contro **HAR-CJ**, mantenuto quando C3 sostituì il denominatore con HAR-C e mai riallineato — contro HAR-C il valore è sempre stato −22.42%, con lo scarto in direzione **conservativa**); quello **superiore** non era stale ma **arrotondato al punto**, ed era l'unico dei due arrotondato nella direzione che **favorisce** il claim. Allineare le precisioni toglie l'asimmetria e rende il claim **leggermente più conservativo**. ⚠ **I due decimali non sono un intervallo di confidenza:** identificano *quale* coppia (modello, npz, config) produce il numero. L'incertezza misurata resta ~±0.7 punti rispetto alla config di training, 0.0019 rispetto al vintage macro, **zero** rispetto al seed — nessuna delle tre è stocastica, e si elimina dichiarando la coppia. Propagato in `THEORY.it.md` §12.2 (con un nuovo paragrafo dedicato IT+EN), `README.it.md` §5.1 e i commenti di `tests/test_har_c_baseline.py`; **i record datati di C2 e C3 non sono stati riscritti** — dicono quale banda fu decisa allora ed erano veri quando furono scritti.

**Coda del 02/08 svuotata.** (i) `astype(np.float32, copy=False)` in `to_t()` di `02_train.py`: **−2.42 GiB di picco**, bit-identico — è la seconda metà del picco di cui il `clamp_` in-place era la prima, e le due sono corrette **solo insieme**, perché con `copy=False` il clamp scrive direttamente sul membro npz. L'invariante che lo rende sicuro (`NpzFile.__getitem__` materializza un array fresco a ogni accesso) non è assunta: la inchiodano i 7 test di `tests/test_npz_load_aliasing.py`. (ii) **Guard di fit del walk-forward regime esercitato su dati reali** — la soglia di abort `max_fit_failure_ratio` (0.5, introdotta il 02/08) non era mai stata misurata sul campo, e una soglia di abort mai vista in campo è un rischio di **disponibilità**: se i fit reali fallissero oltre il 50%, il guard renderebbe impossibile il rebuild che doveva proteggere. Probe **read-only** su 2 anni di candele vere (17.520 ore) a cadenza di produzione: **8 fit su 8, `fail_ratio` = 0.000, copertura 100.0%**, margine 0.5 dalla soglia, `last_fit_diagnostics` popolato su un run reale. Nessun file di regime riscritto.

**Refresh macro NON eseguito, deliberatamente.** Due campioni forward pre-registrati aperti dipendono dall'input del modello: hedged (16/20, ~09/08) ed **E1 stadio 2, che è a 0/40 e accumula fino a ~10/09**. Promuovere un vintage oggi spaccherebbe E1 stadio 2 fra due normalizzazioni — e il `MacroNormalizer` è rifittato **whole-df**, quindi un refresh cambia i valori macro anche delle righe storiche. Il vintage si può ri-puntare indietro, le osservazioni forward già raccolte no. Finestra pulita: dopo ~10/09.

**Gate M1 pre-registrato ed ESEGUITO in giornata: PASS ⓪①②③④, zero GPU.** L'impronta di identità train↔inference copre ora anche il **vintage macro**: `02_train.py` registra nel `PipelineState` l'md5 per split di `X_macro_*` dell'npz consumato (più ordine, conteggio colonne e dtype; fonte `measured`) e i tre giudici vol la ri-calcolano e fail-fastano, con `--allow-macro-mismatch` separata da quella dello scaler. ⓪ **inerzia bit-identica su entrambi gli split** contro i report archiviati di R1 (118 chiavi comuni, **0** differenze numeriche; le sole due sono le etichette di path del blocco `provenance`, 18 chiavi nuove tutte sotto `provenance.macro`, zero perse). ① controllo positivo a due livelli: una singola cella modificata, un riordino di colonne a valori identici, un cambio di dtype e il caso end-to-end devono tutti far scattare il guard — senza, un guard che ritornasse sempre `true` avrebbe superato l'inerzia in modo perfetto. ② i tre model dir anteriori a M1 restano `matches: null` = non verificabile, mai `true`. ③ path live irraggiungibile, verificato per grep **e** con un test parametrico su quattro file: un fail-fast raggiungibile da `04b` fermerebbe il forward test al bootstrap dentro campioni aperti. ④ suite **490 passed / 1 skipped**. ⚠ **Due limiti dichiarati e non chiusi:** il controllo positivo è **sintetico** (il vintage V1 non esiste più e ricostruirlo richiederebbe di riscrivere l'npz congelato), e i tre modelli pre-M1 — artefatto canonico incluso — non porteranno mai l'impronta. ⚠ **Il backfill previsto dalla pre-registrazione NON è stato eseguito, di proposito:** un'impronta ricalcolata oggi dall'npz corrente combacia **per costruzione**, quindi non porta evidenza e convertirebbe solo un onesto `null` in un rassicurante `IDENTICO` — la condizione ② violata dalla porta principale. Fatto meno di quanto il PASS autorizzasse, mai di più.

**Pre-registrazione M1 come fu scritta:** estendere l'impronta di identità train↔inference al **vintage macro** dell'npz. Oggi il guard copre il RobustScaler dei prezzi e `target_scale` ma non la normalizzazione macro, che non vive nel `PipelineState` canonico — quindi due modelli addestrati su macro diverse passano entrambi `matches: true`, con uno scarto misurato di `0.0019` sul rapporto pubblicato. Impronta scelta ex-ante: md5 di `X_macro_train` letto dall'npz al training e persistito nello stato del modello. Costo **zero GPU**; pre-registrata comunque perché tocca l'impronta che decide se un numeratore è confrontabile. Condizioni: inerzia bit-identica del numeratore pubblicato su entrambi gli split, **controllo positivo** (il guard deve fallire quando deve — senza, un guard che ritorna sempre `true` supererebbe l'inerzia in modo perfetto), nessun `null` confuso con "verificato", e **il path live irraggiungibile dal nuovo fail-fast** (fermerebbe il forward test dentro tre campioni aperti). ⚠ Limite dichiarato ex-ante: il vintage V1 non esiste più e ricostruirlo richiederebbe di riscrivere l'npz congelato, quindi il controllo positivo è **sintetico** — dimostra che il guard distingue due macro diverse, non che avrebbe intercettato quell'evento storico.

---

## 2026-08-04 — La banda pubblicata torna verificabile, e la sua incertezza dichiarata era una deduzione

**Gate R1 eseguito e chiuso: PASS ⓪①②③④.** La coppia canonica modello ↔ npz è stata addestrata a config di produzione **invariata** sull'npz congelato e vive come artefatto permanente in `models/canonical_1h_vols/` (5 checkpoint, `pipeline_state.pkl`, i due report del giudice, `PROVENANCE.md` col legame modello↔dataset e le impronte di scaler). Esiti: ⓪ baseline riprodotte **cifra per cifra** su entrambi gli split (val n=6485, test n=6486) → stesso npz; ① `provenance.matches = true` **senza** via di fuga; ② gate storico contro **HAR-RV** superato (val 0.26143 ≤ 0.33913; test 0.23637 ≤ 0.35149, entrambi ≪ naive); ③④ materialità rispettata contro **HAR-C** — **0.7758 su val (−22.4%) e 0.6835 su test (−31.7%)** contro soglie pre-dichiarate 0.80 e 0.71. **La banda pubblicata non cambia di una cifra**: il numeratore coincide **esattamente** con quello già pubblicato. Cambia lo **statuto** del claim — da affermazione sul protocollo di addestramento ad artefatto ri-giudicabile (`dev_vols_qlike.py --arch canonical_1h_vols`). ⚠ **Nessuna promozione**: `models/itransformer` e il VPS restano intoccati finché i campioni forward pre-registrati sono aperti.

**Ritrattazione: «~0.2 punti di incertezza di seed-draw» era una deduzione, non una misura, ed è falsa.** §12.2 attribuiva lo scarto fra due repliche (val 0.26206 vs 0.26143) al sorteggio RNG. Misurato: a seed, config e npz fissi il protocollo è **deterministico** — la coppia canonica riproduce una replica precedente alla **decima cifra su entrambi gli split** (delta 0.000e+00), quindi la dispersione di ri-addestramento è **zero**. I report su disco formano due cluster deterministici e la differenza fra i due **non è rumore di seed** ma di config o di versione del codice, **non identificata**: scritta come tale in §12.2 IT+EN. È la seconda ritrattazione in tre giorni con la stessa forma — una causa plausibile scritta senza l'esperimento che la testava.

**Tre difetti trovati eseguendo il gate, nessuno dei quali era l'esperimento.** ① **Il guard sullo scaler era cieco in sandbox:** `check_model_dataset_scaler` risolveva il canonico via `models_root()`, che sotto `QUANTSYS_MODELS_ROOT` punta **dentro** la sandbox, dove il canonico non esiste mai → `matches=None` e un warning, cioè nessun controllo **proprio nella modalità in cui si giudicano i candidati**, l'unico caso d'uso del guard. Nuova `canonical_state_path()`: canonico locale alla sandbox se esiste (esperimento con npz proprio), altrimenti quello della root di default; giudice ri-eseguito su val col fix, `nn_qlike` **bit-identico** → il fix tocca la provenienza, non la metrica. ② **Il nome del report non conteneva l'arch:** giudicare un artefatto che vive come dir-arch dentro `models/` avrebbe scritto sul nome NUDO `qlike_report_1h_val.json`, **sopra il report storico di produzione** citato in §12.2, uscendo 0 e stampando un PASS corretto — distruzione silenziosa la cui unica traccia sarebbe stato un file con numeri plausibili. Regola estratta in `report_filename()`, path production invariato nel nome, non-clobber verificato con md5. ③ **`--arch` non ammetteva l'artefatto**, che sarebbe stato un checkpoint dichiarato riproducibile e **non verificabile** — il difetto stesso che R1 chiude. Sette test nuovi: `tests/test_qlike_report_naming.py` (4, di cui uno pretende che val e test non collidano mai perché il test split è one-shot) + 3 in `tests/test_scaler_identity_guard.py` (uno gira **con la env sandbox attiva** e pretende `matches is True`; uno è la sentinella sull'artefatto canonico).

**Braccio diagnostico B (solo val): speedup 1.61×, adozione RESPINTA, e il risultato utile è il Δ.** Riaddestrando a `batch_size 128`/`ga 1` il wall-clock scende da 28.5 a 17.7 min (≥1.5× ✅) ma il rapporto NN/HAR-C passa da 0.7758 a **0.7690**, cioè `|Δ| = 0.0068` contro una soglia di adozione pre-registrata di 0.005 → **la config di produzione resta `64`/`2`**. ⚠ B è *migliore* su val e non conta: il ruolo dei bracci era fissato ex-ante, sceglierlo a risultati visti sarebbe selezione sull'esito. Il numero informativo è che un knob **puramente computazionale** — a dati, architettura e seed identici — sposta il rapporto pubblicato di **3.5× lo scarto** fra i due cluster deterministici che ieri era attribuito al seed-draw. Coppia di fatti complementari misurati nello stesso giorno: **il seed non muove il rapporto (Δ=0), la config sì (Δ=0.0068)**. Ne segue che gli estremi della banda portano **~±0.7 punti percentuali** di incertezza rispetto alla configurazione di training — un'incertezza **non stocastica**, che si elimina **dichiarando** la config invece che mediando su repliche. Config di produzione ripristinata e verificata (`git diff --exit-code` pulito); suite **469 passed, 1 skipped**.

**I due cluster deterministici: causa identificata — è il VINTAGE MACRO dell'npz, non il codice e non i seed.** Indagine a costo zero (log + git). Due soli rewrite dell'npz (19/07 16:54:57 e 30/07 20:55:02): tutte le run del cluster 0.26206 stanno fra i due, tutte quelle del cluster 0.26143 dopo il secondo, e la prima di queste parte **87 secondi** dopo la riscrittura. Meccanismo verificato nel codice: `01b_download_macro.py` sostituisce **solo** `X_macro_{split}` e lascia intatti `X_*`, `y_*`, `t_*`; la macro è **input del modello** (90 colonne) ma **non entra in nessuna baseline HAR** → NN si sposta, baseline identiche cifra per cifra, che è esattamente la firma osservata. ⚠ Non è un append: il `MacroNormalizer` è **rifittato whole-df**, quindi allungare la serie cambia i valori macro **anche delle righe storiche**. Alternative escluse: seed (Δ=0, misurato), batch (811 batch/epoca ovunque), codice (i due commit della finestra sono uno di solo logging e uno inerte a flag spento, senza `nn.Parameter` aggiunti quindi senza consumo di RNG). ⚠ **Falsa pista degna di nota:** gli SHA delle run del cluster B non esistono più (pre-rewrite del 27/07) e il confine ci cade accanto — sembrava una spiegazione e non poteva esserlo, perché un rewrite di storia non cambia il contenuto dei file. **Buco dichiarato e non chiuso:** il guard di identità copre il RobustScaler dei prezzi e `target_scale`, non la normalizzazione macro (assente dal `PipelineState` canonico), quindi due modelli possono passare `matches: true` ed essere addestrati su macro diverse — **0.0019** di scarto sul rapporto pubblicato, oggi non segnalato da nulla. **Bilancio sulla precisione della banda:** seed 0, refresh macro 0.0019, config di training 0.0068 — tutte fonti **deterministiche e dichiarabili**, nessuna stocastica.

---

## 2026-08-03 — Un allarme che era un artefatto di misura, e la coppia canonica pre-registrata

**Il "buco del recorder L2 del 02/08" non è mai esistito, e la causa è stata rimossa.** L'ora incriminata ha 720/720 snapshot e il run contiguo è 462+28=490h: mai spezzato. Il monitor `scripts/vol/l2_continuity_check.py` misurava il **mirror locale** credendo di misurare il recorder — lo span finiva su `ts[-1].floor("h")`, che è **per costruzione** l'ora contenente l'ultimo tick, quindi in corso e parziale; con soglia 360 su cadenza reale 720/h **l'esito dipendeva dal minuto** in cui girava la routine. Secondo canale: il pull scarica i giornalieri con `scp` senza atomicità remota, quindi la coda può arrivare in ritardo di un pull. Fix: ora in corso **esclusa** dallo span (conteggio conservativo), buchi nelle ultime `--provisional-hours` (default 6) marcati **PROVVISORI** ed esclusi dalla stima di costo — *un buco è un fatto solo dopo essere sopravvissuto a un secondo pull* — mentre i buchi consolidati restano un allarme pieno. Logica estratta in `analyze()` per essere testabile: `tests/test_l2_continuity_check.py` (5), fra cui uno che pretende il run corrente invariato su cinque riempimenti dell'ora in corso (1, 60, 359, 361, 720). **Sintomo diagnostico generalizzabile: se l'esito dipende dall'istante in cui giri la misura, è un artefatto di misura, non un fatto sui dati.**

**Gate R1 pre-registrato (APERTO, mai eseguito): coppia canonica modello ↔ npz.** La banda pubblicata −23% ÷ −32% ha come numeratore una coppia riaddestrata in una sandbox poi eliminata, quindi oggi è un'affermazione sul **protocollo**, non su un artefatto verificabile. R1 lo produce, a config di produzione **invariata** e npz **congelato**, con quattro condizioni pre-dichiarate: ⓪ controllo di vintage **model-independent** (le baseline devono coincidere cifra per cifra: sono fittate dentro l'npz, quindi se divergono il dataset è un altro e non c'è nulla da ri-pubblicare), ① identità dello scaler (`matches: true`, senza `--allow-scaler-mismatch`), ② sopravvivenza del gate storico contro **HAR-RV** (denominatore del *gate*, non del *claim*), ③ materialità `NN/HAR-C ≤ 0.80`. Test one-shot solo a val verde. Braccio diagnostico **Leva B** (`batch_size 128`, `ga 1`) **solo su val**, che per costruzione **non può** diventare la coppia canonica: il ruolo è fissato ex-ante perché sceglierlo a posteriori sarebbe selezione sull'esito. ⚠ Il PASS **non autorizza la promozione**: sostituire il modello di `04b` dentro campioni forward pre-registrati aperti è la stessa violazione del refresh macro dentro un campione. **Due correzioni collaterali:** un rapporto della nota di provenienza in §12.2 riportava il denominatore sbagliato (0.7738 = prima replica contro HAR-CJ, invece di 0.7777 = seconda replica contro HAR-C; lo spread ~0.2 punti e ogni claim pubblico restano invariati), e `scripts/00_check_setup.py` stampava un ✗ **rosso** per ogni artefatto di pipeline assente mentre il verdetto finale diceva "setup verificato" — su un clone fresco quegli artefatti mancano **per definizione** e infatti non concorrono al verdetto: ora sono warning.

---

## 2026-08-02 — Audit di performance: il training è launch-bound, e due fallimenti silenziosi in meno

**Ricognizione diagnostica (`docs/PERF_AUDIT.it.md`, nessuna ottimizzazione applicata) più i due fix che ne sono usciti.** Il risultato centrale ribalta l'intuizione: **il training non è compute-bound**. A batch 64 la GPU sta al 5-15% di SM e il tempo per step è dominato dal **lancio** dei kernel — passando da batch 32 a 128 il lavoro aritmetico quadruplica e il wall-clock cresce del 16%; la curva diventa lineare solo oltre batch 512 (SM 96-98%). Conseguenza: i lever che riducono l'aritmetica (`channels_last`, Numba, estensioni native, Polars) non toccano il collo di bottiglia. `torch.compile(backend="cudagraphs")` — l'unico che aggredisce i lanci — misura **1.56×** sullo step (15.30 → 9.79 ms), senza Triton (non installabile da PyPI su Windows) e senza graph break (Dynamo traccia 1 grafo, 88 op); ⚠ **non applicato**: cambia i pesi, quindi richiede un gate pre-registrato. Scartati con la ragione tecnica: `channels_last` (**nessun tensore 4D NCHW** nel progetto — solo Conv1d e attention batched), Numba (il bootstrap CI è **già** una matrice NumPy `(5000,n)`, 37 ms; l'event loop del backtest costa 0.2-0.9 s ed è pieno di Enum/dataclass), estensione nativa (nessun componente insieme pesante **e** isolato; il calcolo più lungo — regime walk-forward — vive dentro statsmodels ed è già risolto da B7), Polars nel `FeatureBuilder` (il data prep completo è **2.24 s**, di cui il 59% è il loop Python del Volume Profile che Polars non esprime; e il prototipo mostra che `ret_skew_20` — feature della lista-104 — cambierebbe del **7.7%** perché Polars usa lo stimatore **biased** e pandas l'**unbiased**: differenza di definizione, non di arrotondamento). Individuata la leva col miglior rapporto guadagno/complessità, **non applicata**: i clip bounds `np.nanpercentile` costano **31-36 s** per invocazione di `02_train` ordinando 647M celle che sono **52.001 barre distinte ripetute 120×** (stride 1); calcolarli sulle barre distinte è **200× più veloce** ma sposta 34 colonne su 104 → è una leva, non una pulizia.

**Due fallimenti silenziosi rimossi** (entrambi bit-invarianti sul path di successo, entrambi emersi dall'audit e non cercati). ① **Ordine di inizializzazione DLL:** caricare pyarrow dopo torch **e** scikit-learn provoca un'access violation (exit 139, nessun traceback) al primo `read_parquet` — conflitto fra i runtime OpenMP dei due. Gli script numerati sopravvivevano solo perché importano `pandas` alla riga 30 e `torch` alla 31: un invariante **di fatto, mai dichiarato né testato**, che uno script nuovo scritto nell'ordine naturale (progetto prima, pandas poi) violerebbe. `import pyarrow` ancorato in `quantsys/__init__.py` lo rende una proprietà del package. ② **Degradazione silenziosa del walk-forward regime:** ogni fit Markov-Switching fallito finiva in un `log.warning` per timestep e il loop proseguiva; con `current_params=None` le probabilità non venivano mai scritte e `fit_predict_walkforward` restituiva la **prior uniforme travestita da regimi**, senza che nulla fallisse (e il ramo `_fit_single → None` non produceva **nemmeno un log**). Ora `RuntimeError` su zero fit riusciti — non disattivabile, perché un walk-forward senza un solo fit non produce informazione — più abort configurabile su `max_fit_failure_ratio` (default 0.5) e diagnostica persistita in `last_fit_diagnostics`; guard rispecchiato in `continue_walkforward`, dove il fallimento produce parametri **stantii** anziché la prior. Test: `tests/test_import_order.py` (4) + `tests/test_regime_fit_guard.py` (8) → **450 passed, 1 skipped**. Bit-parity B7 del regime incrementale invariata.

---

## 2026-07-31 (3) — Strumento vs stato: `MacroNormalizer` pinnabile, gate E1 pre-registrato

**Seconda metà del problema macro.** I vintage datati (voce precedente) risolvono *quale* file arriva al VPS; restava che `VolForecaster` **ri-stimasse** il `MacroNormalizer` whole-df a ogni bootstrap, per cui allungare il parquet muove mediana e IQR e **lo strumento di misura cambia insieme allo stato che deve misurare** — 2.7% della variazione totale sul breakpoint del 31/07. Estratta `macro_snapshot()` (due rami: `refit` legacy e pin da disco), aggiunto `scripts/vol/pin_macro_normalizer.py` e il flag `--macro-norm` a `04b` **e al replay**. Parametro **esplicito, mai da env**, stesso principio di `--arch`: una env residua cambierebbe l'input del live in silenzio; e il replay deve poter scegliere il regime in base alla **data** della decisione che riproduce, non all'ambiente. **Inerzia provata end-to-end sul parquet di produzione: 0 differenze su 90 colonne.** Guard fail-fast sulle colonne del pin (ordine compreso): applicare mediana e IQR della colonna sbagliata sarebbe silenzioso e permanente. ⚠ Il vintage sotto cui `models/itransformer` fu addestrato **non è ricostruibile**: il pin ne **fissa** uno dichiarato, non lo recupera; attivarlo oggi è un **no-op di contenuto** e serve a impedire la deriva futura. **Gate E1 pre-registrato ed eseguito allo stadio 1** (esplorativo, nessun verdetto): l'edge NN-vs-IV ha contenuto predittivo sulla varianza realizzata a 30h? Stadio 2 confermativo a n≥40 expiry → ~10 settembre 2026. **Condizione ③ ex-ante su A13a** (pin-close): `n_eff = n_trig`, non `n_posizioni` — le posizioni che non innescano sono bit-identiche sotto le due regole; A13 parcheggiato con condizioni di riapertura datate. Test: 3 nuovi file (18 test). Vedi `STATUS.md` 2026-07-31 sessione 2.

---

## 2026-07-31 (2) — Lo snapshot macro del live diventa un artefatto versionato

**Il push macro casa→VPS sovrascriveva il canonico a ogni pull, incondizionatamente** — e il 2026-07-31 ha consegnato al live una macro rifrescata **dentro due campioni forward aperti**, prendendo di fatto una decisione che era in sospeso. Il difetto di fondo però non era il push: era che lo snapshot che alimenta `04b` (letto al bootstrap notturno e **congelato** per la giornata) fosse **stato mutabile** invece che artefatto versionato — tanto che il breakpoint del 31/07 non è stato misurabile direttamente, il file vecchio essendo stato sovrascritto e non essendo in git. Riscritto il blocco 0 di `pull_vps_data.ps1` in tre parti: (a) archivio **append-only** `data/macro/macro_features_<YYYYMMDD>.parquet` sul VPS, con `<YYYYMMDD>` = ultima data dell'indice (`scripts/vps/macro_vintage.py`, nuovo) — 716 KB a copia, quindi ogni decisione forward resta riconducibile al suo vintage e il replay torna riproducibile; (b) il canonico diventa un **symlink** all'archivio, così il vintage live si legge con `readlink` e si vede in `ls -l` — nessun marker che possa mentire, il puntatore **è** la verità; (c) ripuntarlo richiede `-PromoteMacro`, quindi **il push smette di essere una decisione e promuovere lo diventa**; a vintage divergente si emette un warning e il live resta dov'era. Scartata l'alternativa "far girare `01b` anche sul VPS": due fetch FRED/yfinance indipendenti divergono **sulla storia** (le serie FRED sono revisionate retroattivamente), il che romperebbe la riproducibilità di `vol_paper_replay.py` — lo strumento con cui è stata provata la parità live↔training — e toglierebbe del tutto l'umano dal loop invece di rimettercelo. Nessuna modifica al path live (`vol_forecaster.py` continua a leggere lo stesso percorso canonico), quindi l'intervento è ammissibile a campioni forward aperti. Test: `tests/test_macro_vintage.py` 4/4 (contratto CLI: stdout = **esattamente** il vintage, fallimenti puliti a stdout vuoto); branch del blocco 0 verificati su 5 casi con `ssh`/`scp` stubbati. Vedi `STATUS.md` 2026-07-31 (sessione 2).

---

## 2026-07-26 (2) — Diebold-Mariano sul confronto NN-vs-HAR + repo pronto alla pubblicazione

**Il claim "batte HAR del 30% in QLIKE" ha ora un'inferenza, non solo una stima puntuale.** Implementati `qlike_series()` (loss per-campione; `qlike()` ne è la media → formula in un unico punto) e `diebold_mariano()` con varianza **HAC Newey-West (kernel di Bartlett, lag `q = h−1 = 29`)** e correzione small-sample **Harvey-Leybourne-Newbold**: necessaria perché il target somma 30 barre, le finestre si sovrappongono e con varianza iid lo standard error sarebbe sottostimato di ~√h (`n_eff ≈ n/h` ≈ 216, non 6.5k). Esiti su una coppia modello/scaler riaddestrata (5 seed, sandbox, produzione intatta): **val −26.6%, p = 7.3e-05 · test −36.1%, p = 1.7e-06**; il NN batte HAR **in ogni regime**, stress incluso e validato su test. La banda onesta del claim è **−27% ÷ −36%** secondo split e vintage (il −36% e il −30.2% storico misurano popolazioni diverse: finestra di test estesa + modello riaddestrato). Il DM è **descrittivo, non gating**: le soglie pre-registrate restano i rapporti di QLIKE. Test: `tests/test_diebold_mariano.py` 9/9, incluso quello che verifica l'inflazione dello SE su differenziali sovrapposti — senza cui i p-value sarebbero fittizi. **Repo pronto alla pubblicazione:** audit secret PASS su tutte le revisioni, igiene OpSec, attribuzione dell'assistente rimossa (storia riscritta, 141 trailer, albero bit-identico), corpus KILL e protocollo trasferiti in `THEORY.it.md` §12. Vedi `STATUS.md` 2026-07-26 ⑦⑧.

---

## 2026-07-26 — Fix unità di misura del contatore hedged + archivio STATUS committato

Il contatore automatizzato ieri stampava `hedge_ledger: {n} eventi` accanto alla soglia `n≥20 hedge-attivi`, ma il campione pre-registrato del giudice hedged-vs-unhedged è definito in **trade aperti con hedge attivo**, non in eventi di ledger (1 posizione = open + N rebalance + flatten): con 22 eventi la riga invitava a leggere "22 ≥ 20" e a lanciare **in anticipo** un giudice one-shot. Corretto in `position_key` distinte con ≥1 hedge eseguito (**n=6** al 26/07, giudice atteso ~09-10/08) con l'esclusione pre-dichiarata della posizione parzialmente hedgiata resa **esplicita nel codice**. Un rischio di violazione introdotto dall'automazione, non un errore di dati: la lezione è che **automatizzare un contatore richiede di verificare che l'unità stampata sia quella della pre-registrazione**. Committato anche lo scorporo dello storico `STATUS.md` → `docs/STATUS_ARCHIVE_2026H1.md` (verificato **letterale**: 809 righe rimosse, 0 non ritrovate nell'archivio) con il doc-sync associato. Vedi `STATUS.md` 2026-07-26.

---

## 2026-07-25 — Routine di sessione automatizzata (blocco ③ monitoraggio vol)

`avvio_sessione.ps1` copriva solo ① pull+merge VPS e ② check freshness regime B7, mentre la routine ricorrente ne richiedeva altri 3 passi manuali (dimenticabili). Aggiunto **blocco ③ "monitoraggio ricorrente linea vol"** (`-SkipMonitor` per saltarlo): `derive_mfiv.py` incrementale — **vincolo d'ordine: dopo il merge**, perché legge la chain appena scaricata — + `mfiv_comparator_judge.py --count-only` + stampa in coda dei contatori dei due gate forward (`n executed` di `trades.jsonl`, eventi di `hedge_ledger.jsonl`). **Invariante di disciplina preservato:** `--count-only` calcola solo timestamp e il giudice ha comunque il guard `n<N_MIN=40 → NO_RUN` senza scrivere report → automatizzare il conteggio **non può** produrre peeking; il run one-shot resta MANUALE. Fail-soft con check esplicito di `$LASTEXITCODE` (in PS 5.1 un exe nativo che esce ≠0 NON solleva eccezione: il `try/catch` non basta) e file ri-verificato ASCII-only. Validazione: run `-SkipPull` end-to-end OK, blocco ③ idempotente. Vedi `STATUS.md` 2026-07-25.

---

## 2026-07-19 → 07-23 — Lever di training sulla linea vol: A8-BIS e B4-bis entrambi FAIL su val

Due gate pre-registrati eseguiti e chiusi negativi, entrambi **contro una baseline riaddestrata sullo stesso dataset esteso** (non contro l'incumbent production). **A8-BIS mixup** (`mixup_alpha` 0→0.2): baseline 0.26206 vs candidato 0.25998 = −0.79% ≪ soglia −3% → FAIL, overlay eliminato, niente test. Lezione metodologica di primo ordine: il −4.94% descrittivo misurato in Fase B era **artefatto di distribution shift** (confronto cross-scaler contro l'incumbent old-scaler) → **mai decidere un lever su confronti attraverso scaler diversi**. **B4-bis DVOL-come-feature** (pre-reg `5a6112d`, close `526659d`): `dev_vols_dvol_append.py` estende X_macro 90→93 (`dvol_log`/`dvol_chg_24h`/`dvol_avail`, asof causale cap 24h, fill mediana train-only, copertura val 1.000, npz production intatto); baseline 0.26206 vs candidato 0.25939 = −1.02% > soglia −3% → **① FAIL** (② sarebbe passata, gate in AND), nessun one-shot su test. Lettura: tenor mismatch 30d→30h + incumbent che già cattura l'informazione IV via lag RV → MSE-log migliora −6% ma non QLIKE, dove conta la calibrazione di σ². Ratificato anche il **pattern-③ standard** (condizione di conteggio per regime verificata ex-ante: se model-independent l'esito è "nessuna conclusione", non un risultato). Vedi `STATUS.md` 2026-07-19/20/23.

---

## 2026-07-18 — Gate v1 CHIUSO: FAIL 0/3 · `04b` migrato sul VPS · MFIV@30h (D4)

**Gate v1 del braccio short-vol chiuso al checkpoint pre-registrato n=20: FAIL 0/3 su ENTRAMBI i campioni** (concordi). Always-short +0.0396 → il VRP resta positivo e conferma il braccio, ma la regola v1 non lo monetizza; sbloccato `POST_GATE_V1.md`. **Migrazione `04b` sul VPS** (`quantsys-volpaper` systemd, con `--hedge --hedge-band 0.30 --hedge-conv raw`): il forward v2 parte e il PC di casa diventa **passivo** — ⚠ mai più lanciare `04b` a casa, doppi ordini sulla stessa posizione testnet. **🔴 Bug candele:** la finestra live era bucata dal ~06-24 (fixato + guard aggiunto). **D4 MFIV@30h model-free** (`scripts/vol/derive_mfiv.py`): derivazione OFFLINE retroattiva del var-swap rate VIX-style a tenor 30h + skew 25Δ RR/BF dal raw chain già registrato → colonne PARALLELE, mai nel path decisionale; wedge di convessità MFIV−ATM mediano **+3.45 vol pt** → il break-even short-vol calcolato su IV ATM era **conservativo**. `VolForecaster` promosso da `04b` a `quantsys/model/vol_forecaster.py`. Vedi `STATUS.md` 2026-07-18.

---

## 2026-07-14 → 07-16 — VPS collector 24/7 + B7 regime incrementale + GJR-1h + A11-A14

**VPS collector 24/7** (VPS EU, Ubuntu 24.04, geo-test PASS): servizi systemd `quantsys-iv` / `quantsys-ob` / `quantsys-trades` — il nuovo **`01e_trades_recorder.py`** raccoglie i trade opzioni Deribit production (retention API ~24h verificata → la raccolta è necessariamente FORWARD) per misurare gli spread realizzati vs mark. Host in `config/secrets.yaml → vps.host`, **mai leggerlo né stamparlo**. Sync lato casa: `scripts/vps/pull_vps_data.ps1` (+ fix anti-stallo ssh/scp). **B7 — regime walk-forward incrementale**: il full rebuild salva `data/regime_wf_checkpoint.pkl`, `01b --regime-incremental` appende le sole barre nuove (**minuti invece di ~3h**), `--regime-bootstrap-checkpoint` ricostruisce lo stato con replay-validation fail-fast; bit-parity garantita da `tests/test_regime_incremental.py`, bootstrap validato bit-exact. Sul rebuild 7 anni le **etichette dei regimi sono RIMAPPATE** (R1 = stress ora): ri-derivarle dalle varianze a ogni full rebuild. **GJR-1h chiuso**: ri-stima QMLE dei parametri MC su rendimenti orari (`scripts/vol/estimate_gjr_1h.py`) → γ≈0.005, leverage effect quasi nullo a 1h; cap σ/barra reso parametrico. **A11-A14 (funzioni gamma):** `pnl_attribution.py` ATTIVA (decomposizione ex-post delta/gamma/theta/vega, read-only); A12 banda WW, A13 pin-close+gamma-cap, A14 sizing-vega implementati **INERTI** (attivazione solo con pre-reg v2 post-gate). Vedi `STATUS.md` 2026-07-14/15/16.

---

## 2026-07-08 — Roadmap vol-book v2 + A6 exec-diag in `04b` + refresh macro

**Roadmap** (`docs/ROADMAP_VOL_BOOK.it.md`, sessione advisory 07-07): backlog A1-A10 dall'audit anti-overfit (verdetti chiusi: FrAug/RevIN-vol/MC-dropout/interp-N-HiTS NO) + verdetto strategico book a due strumenti — B1 futures direzionali con leva ❌ (momenti dispari falsificati OOS, E[PnL] = leva·(0−costi) < 0), B2 perp Deribit come **delta-hedge** del book opzioni ✅ (PnL = ∫½ΓS²(σ²impl−σ²real)dt = puro harvest VRP, la quantità che il NN predice), sequencing B3 vincolante. **A6 implementato** (`scripts/04b_vol_paper.py`, unico item pre-gate): `log_exec_diag()` a fine tick → `results/vol_paper/exec_diag.jsonl` (bid/ask/mark/IV/greeks Deribit per leg + delta netto + half-spread; posizione aperta → leg in essere, flat → straddle ATM ipotetico), fail-soft, **regola e costanti pre-registrate INTATTE**; smoke su testnet reale + processo live riavviato (prima riga: half-spread 18.2% ≈ haircut 16% della validazione 06-25). **Refresh macro**: `macro_features.parquet` 06-10→07-08 (sola sezione macro di `01b`, schema 90 col identico; regime/npz/normalizer deliberatamente intatti). Vedi `STATUS.md` 2026-07-08.

**A2a+A5 eseguiti e FALLITI stesso giorno (gate pre-registrati, zero retrain — la testa quantile era GIÀ nei checkpoint PASS).** A2a (`scripts/vol/dev_vols_quantile_judge.py`): coverage sopra target a tutti i livelli (q50→0.73, q90→0.97 = distribuzione shiftata in alto) e pinball q90 NN perde da HAR+quantili-residui (0.160 vs 0.144) → **A2b (retrain q95) morto**; per il sizing v2 la coda destra è HAR-q90 o conforme da pre-registrare. A5 (`scripts/vol/dev_vols_member_weights.py`, fit 1ª metà val / eval 2ª): ratio 0.9925 vs gate ≤0.97, pesi quasi-uniformi, best-single peggio dell'ensemble → **pesi uniformi confermati ottimali**. Checkpoint production read-only, test split intatto, processi live fermati ~5 min e rilanciati sani.

---

## 2026-06-26 — Audit statistico short-vol (1 conclusione corretta, 1 prior smentito) + perf B2/B3 golden Δ=0

Audit statistico/logico del backtest short-vol (causalità/lookahead verificati PULITI ovunque). Fix in `scripts/vol/`: **①** block-bootstrap CI + N-effettivo — la lag-1 autocorr dei PnL è ≈0 → N_eff≈N(2538), CI>0 (l'overlap 30h/24h NON gonfia la significatività al lag-1; il bootstrap però non cattura la concentrazione temporale 2020-21=90% PnL). **②** Sharpe annualizzato a √(trade/anno)≈√365 (era √292, incoerente con la cadenza giornaliera). **③** haircut bid REGIME-dipendente (`--stress-haircut-mult`) → **CORREGGE la conclusione regime del 06-25**: "edge più alto nello Stress" era un artefatto dell'haircut costante; con spread realistici il **Trending domina** e lo Stress è marginale (il SEGNO regge, la GERARCHIA no). "Non filtrare il regime" SOPRAVVIVE robusto. **④** caveat n=3 load-bearing reso esplicito. **⑤** martingale-correction empirica (flag `--mart-correct` default-OFF): prior SMENTITO (Δ≈5e-3 non ~1e-4 per excess-kurtosis residui ≈19.7 → sovra-corregge sui path di coda) → flag inerte confermato corretto. **A4:** regression test slippage pre-size skip. **Perf B2/B3** (`quantsys/features/__init__.py`, gated golden Δ=0): VP rolling min/max precomputato (era per-finestra, bit-identico — guadagno reale modesto ~1.1×: bincount/argsort co-dominano l'inner-loop) + rimossa `df.copy()` di defrag ridondante; verifica `build(normalize=False)` 122col×2970righe = 0 celle diverse, test `tests/test_vp_golden.py`. Nessun retrain (feature bit-identiche). Vedi `STATUS.md` 2026-06-26.

---

## 2026-06-25 — Short-vol arm: backtest storico strutturale FHS GJR-GARCH + validazione premio

Secondo braccio della linea vol (short-vol systematic). Backtest strutturale su 7 anni di candele orarie (`scripts/vol/short_vol_hist_backtest.py`): payoff REALE dalle candele (code incluse), premio = fair-value fat-tailed FHS su GJR-GARCH(1,1) × (1+VRP), VRP swept, tutto CAUSALE (refit expanding 90gg, no lookahead). Risultato (n=2538 scadenze daily 08:00 UTC): break-even VRP = **0% per tutte le strutture** → il PnL medio positivo è la raccolta del VRP storico (realized 30h < implied). Strangle 8-10% = struttura tail-safe (hit 97-98%, maxDD −0.33 BTC, Calmar 33.8). Validazione premio (`short_vol_premium_validate.py`): FHS/mark mediano 1.05×, edge sopravvive all'haircut bid 16%. Decomposizione regime/anno (`short_vol_regime_decomp.py`): edge concentrato nell'alta-vol (2020+2021 = 90% del PnL), always-short, NON filtrare il regime. **NON è un PASS del gate** (resta il live n≥20). Vedi `STATUS.md` 2026-06-25.

---

## 2026-06-24 — IVS relative-value → KILL net-of-cost

Probe smile-reversal Deribit (`scripts/vol/ivs_scout.py` + `scripts/vol/ivs_rv_backtest.py`): struttura reale (i residui dello smile revertono, autocorr 0.77) MA netto **−2.3/−3.8 vol-pt/leg** (gross +0.01/+0.04 vs costo round-trip 2.3/3.9 → ~50× sotto lo spread). Morta come price-taker; vivrebbe solo da market-maker. Tetto **economico**, non di dati.

---

## 2026-06-22 — Robustezza vol: purged k-fold + gate HAR-per-fold + diversità cross-arch

Conferma OOS della linea vol oltre il single-split. Purged k-fold QLIKE per arch (`scripts/02b_walkforward_validate.py`, embargo 168h) + baseline HAR fit-per-fold (`scripts/vol/wf_har_baseline.py`, helper in `quantsys/model/vol_metrics.py`): TCN+Mamba batte HAR di ~14% OOS (ratio 0.863, 4/5 fold), tutti gli archi falliscono solo il fold più antico data-starved (strutturale). Kill-check diversità cross-arch sugli errori vol (`scripts/vol/step0_xarch_corr.py`): ρ_err medio 0.83 (vs ≈0.995 direzionale) → la diversità è anch'essa un oggetto pari-specifico. Numeri in `docs/paper/RESULTS_MAP.md` CLAIM 2b/2c.

---

## 2026-06-12 — Riorientamento vol-1h + paper "Are price and volume enough?"

Lo stato production diventa la linea vol-1h (`models/itransformer/` = PASS vol-1h, `target_type: log_rv`). Aggiunte le baseline econometriche direzionali come negative-control del paper (`scripts/research/paper_01_dir_baselines.py`): nessuna skill coerente, ρ flippano segno val→test come il NN → il risultato è dell'informazione, non del modello. Cleanup disco ~4.7 GB (dataset/modelli 1m rigenerabili). Documenti paper avviati: `docs/paper/OUTLINE.md`, `docs/paper/RESULTS_MAP.md`. Riorganizzazione script per-linea in sottocartelle (`vol/`, `research/`, `archive/`): mappa in `scripts/README.it.md`.

---

## 2026-06-11 — Probe semivarianza firmata → FAIL

Target `log(RS⁺/RS⁻)` fwd (signed jump variation, Patton–Sheppard; giudice `scripts/vol/dev_vols_rs_judge.py`): su test NN/HAR-RS MSE 0.9952 (gate ≤0.95), signDA 0.459, e HAR-RS fa peggio della costante → **l'asimmetria è impredicibile per tutti**. Sintesi chiave: i momenti PARI generalizzano OOS, i momenti DISPARI no. Filone HD-firmato chiuso.

---

## 2026-06-10 — VOL-S PASS + pivot 1m→1h KILLED

**VOL-S PASS (B2 positiva):** target `features.target_type: log_rv`, il NN batte HAR-RV del **30% in QLIKE su test** (0.257 vs 0.368; naive 0.807), val→test coerenti (l'anti-correlazione è specifica del target direzionale). Giudice `scripts/vol/dev_vols_qlike.py`. Verifica cross-risoluzione 1m: FAIL su val (NN/HAR 1.013) → edge vol SPECIFICO della risoluzione 1h. **Pivot 1h KILLED:** il 1h sfonda il muro dei costi (|μ|≈43bps ≫ 26bps) ma NON c'è skill direzionale OOS, anti-correlazione val→test confermata anche a 1h, gate 4/4 fallito a 13 E 23 bps → filone "stesso metodo, altro timeframe" chiuso.

---

## 2026-06-05 — BLOCKER #1 (live parity) risolto

Path live = `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canoniche) → `LiveEngine._deterministic_predict` → `denormalize_predictions` → `SignalGenerator`. Parity feature E segnale bit-perfect (`tests/test_live_training_parity.py`; replay `scripts/99_replay_live_vs_training.py`: Δfeature=0, Δμ=Δσ=0). Residuo operativo: smoke WS + paper-trading. (Backtest direzionale negativo OOS — vedi sopra.)

---

> **Blocco storico — linea direzionale 1m (Iterazioni 1-10).** Conservato come record: l'alpha direzionale 1m non sopravvive OOS (vedi voci 2026-06 sopra), ma le iterazioni di pipeline/fix restano il fondamento del motore condiviso.

## Iterazione 10 — Dashboard: fix definitivo rendering — asse category (causa: asse lineare corrotto al re-render dopo display:none) (2026-06-24)

### Asse X `type:'category'` su `plot-oi` e `plot-payoff` — `scripts/06_dashboard.py`

**Il bug**: "OI by strike" (`plot-oi`) e il "profilo di rischio/payoff" dei
trade (`plot-payoff`) sparivano o si schiacciavano in una banda sottile uscendo
e rientrando da una tab del browser (risk↔trades).

**Root cause (diagnosi browser, 2026-06-24)**: ad ogni re-render dopo
`display:none→block` (qualunque rientro in tab) Plotly **corrompe la mappatura in
pixel di un asse X numerico LINEARE** — le tracce finiscono fuori vista
(x≈−1244px) oppure l'intera banda si comprime a ~19px, mentre le shapes paper-ref
restano corrette. Il **primo** render è sempre buono, ogni render **successivo**
è rotto; colpito **solo l'asse X** (Y numerico ok). `_fullLayout`
(range/offset/length/margin) è identico tra primo render buono e re-render rotto →
corruzione di rendering SVG, **non** dei dati.

**Rimedi falliti (provati e scartati)**: `Plotly.react`, `newPlot`,
`purge+newPlot`, sostituzione del nodo, `Plots.resize`, `relayout`
(toggle width/range), `redraw`, evento `resize` della finestra, `autorange` (gira
solo il fuori-schermo in schiacciato), rimozione larghezza barra esplicita,
scaling x verso il basso, nascondere via `visibility`/`position` invece di
`display:none`, purge-on-leave.

**Il fix (verificato su 3 restart server puliti)**: asse X di `plot-oi` e
`plot-payoff` passato a **`type:'category'`** (il posizionamento per-indice è
immune alla corruzione — coerente con `plot-greeks`, già category, che il bug non
l'ha mai avuto). Nuovo helper `catPos(value, sortedArr)` colloca le linee di
riferimento (Spot/Max-Pain su OI; Strike/Entry su payoff) a un **indice di
categoria frazionario**, restando proporzionali tra le categorie. OI: gli strike
di banda diventano categorie stringa (ordinate crescenti), tick diradati
(`dtick≈n/9`), nessuna larghezza barra esplicita. Payoff: i prezzi del linspace
diventano categorie (V-curve identica, griglia regolare), il marker di settlement
aggancia la categoria più vicina, Y resta numerico autorange.

**Storia (sotto-passo precedente, stesso 2026-06-24)**: l'helper unico
`plot(id, traces, layout, cfg)` aveva già unificato i 7 render con **size-guard**
(`offsetWidth===0` → ritenta al frame successivo via `requestAnimationFrame`,
cap ~1s) + dimensioni esplicite/`autosize:false`. Era contesto **necessario ma NON
sufficiente** — non fermava la corruzione al rientro in tab. L'helper resta
(size-guard + rebuild-al-rientro); il dettaglio width-esplicito/`autosize:false` è
stato rimosso e il fix risolutivo è l'asse category.

**Verifica**: `py_compile` OK; 3 restart server freschi su :8050 con hard
reload (Ctrl+Shift+R) → `plot-oi` e `plot-payoff` stabili su ogni rientro in tab,
linee di riferimento proporzionali. La verità finale resta il browser dell'utente
con hard reload.

---

## Iterazione 9 — Fix denormalizzazione z-score + paper-trading ready (2026-05-23)

### Bug strutturale: trading layer in spazio raw vs modello in z-score

Il `RobustScaler` globale scala `target_ret` insieme alle altre feature
(scale_factor=0.002707). Il modello quindi predice μ, σ, ν in spazio z-score
standardizzato. Tutto il trading layer (`SignalGenerator`, `RiskManager._sl_tp`,
`_size`, soglie config) assumeva invece spazio raw (frazioni di log-return).

**Conseguenze pre-fix**:
- `_sl_tp`: `dist.sigma * price * 1.5` con σ_z≈1 e price=$42k → SL distance $63k
  (300% del prezzo) → mai colpito. Tutti i trade chiudevano per `MAX_HOLD` invece
  che SL/TP veri.
- `_size`: Kelly `mu/sigma²` calcolato in z-score → fattore `target_scale` mancante
  → sizing sottostimato di ~370× (poi capped dal floor 0.005).
- Soglie config in scala mista: `max_sigma=2.0` (z-space) era no-op in raw space.

### Fix centralizzato

- **`quantsys/utils/__init__.py`**: nuova property `PipelineState.target_scale`
  (legge `scaler.scale_` per `target_ret`, fallback 1.0) e metodo
  `denormalize_predictions(mu, sigma) → (mu_raw, sigma_raw)`. Type-preserving
  (float / ndarray / Tensor). Single source of truth.
- **`scripts/03_backtest.py`**: dopo batch inference chiama
  `state.denormalize_predictions(all_mu, all_sigma)`. Safety assert
  `assert all_sigma.max() < 0.05` per rilevare regressioni future.
- **`scripts/04_live_signals.py`**: stessa chiamata in `_predict()` prima del
  return — critico perché senza fix il paper-trading opererebbe con SL/TP
  impossibili.
- **`quantsys/trading/__init__.py`**: warning runtime one-shot in `_sl_tp`
  se `σ*price*1.5 > 5%*price` (rileva mancata denormalizzazione).
- **`config/default.yaml` e `config/arch/*.yaml`**: pulite le soglie legacy
  in z-space (`prob_threshold: 0.52`, `min_expected_ret: 0.0001`, `max_sigma: 2.0`)
  centralizzando in `default.yaml` con valori in spazio raw + commenti che
  dichiarano l'invariante.

### Risultati

Backtest h=30 test set 7929 candele (ensemble eterogeneo, identico per i 3 archs):

| Metrica | Pre-fix | Post-fix | Delta |
|---|---|---|---|
| Sharpe | -255.9 | **+18.71** | +274 |
| Win Rate | 11.03% | **64.29%** | +53 pp |
| Total Return | -15.02% | **+3.67%** | +18.7 pp |
| Max Drawdown | 15.02% | **0.83%** | -14.2 pp |
| Fee/Gross ratio | 1010% | **30.3%** | -980 pp |
| Sharpe CI 95% lower | -48 | **+0.78** | >0 per la prima volta |
| Circuit breaker | TRIGGERED | False | risolto |

**Stress test passato**: Pessimistic (fee×2, slip×3) Sharpe +7.22, Flash Crash
(fee×1.5, slip×5) Sharpe +12.30.

**Walkforward 5-fold** conferma DA stat-sig per iTransformer (0.524 ± 0.008,
CI [0.510, 0.531]) e Spearman 0.070 ± 0.010. WHR borderline (0.504–0.517),
da migliorare con fix #3 (window_size 240) o paper-trading reale.

4/4 soglie esplicite di promozione a paper-trading raggiunte. Vedi
`MODEL_IMPROVEMENTS.it.md` per il piano dei prossimi step.

---

## Iterazione 8 — Horizon 15min + Advanced DL features

### Forecast horizon 5 → 15 → 30 minuti
Target evoluto progressivamente: 5 → 15 → 30 minuti. A h=15 (Iterazione 8 originale)
`asymmetry_threshold` era stato riscalato da 0.002 a 0.004 per compensare l'ampiezza
maggiore dei rendimenti. A h=30 (2026-05-20, parte di Iterazione 9) il movimento
atteso (~42 bps) supera con margine il costo roundtrip (~26 bps), rendendo il
trading strutturalmente profittevole. Search space Optuna `forecast_horizon`
aggiornato a [15,30,60].

### Multi-Teacher Distillation — `quantsys/model/distillation.py`, `run_all.py`, `scripts/02_train.py`
Tutti e 3 i modelli contribuiscono come teacher con pesi proporzionali allo scoring normalizzato
(softmax con temperature=2 su score 40% loss, 35% spearman, 25% DA). Sostituisce la selezione
di un singolo teacher: ogni student riceve soft labels pesate da tutti i candidati.
CLI: `--multi-teacher` flag, attivato automaticamente da `run_all.py --distill`.

### Fractional Differencing (FFD) — `quantsys/features/__init__.py`, `config/default.yaml`
Implementazione Fixed-width Fractional Differencing (López de Prado) su log(close) e log(volume+1).
Genera 2 feature additive: `frac_diff_close` e `frac_diff_volume`. d=0.4 configurabile
(`features.frac_diff_d`), pesi troncati a |w_k| < 1e-5, convoluzione vettorizzata.
d=0.0 disabilita le feature (backward compatible).

### Direction-Value Joint Loss — `quantsys/model/__init__.py`, `scripts/02_train.py`
Nuovo termine di loss che penalizza errori direzionali (sign(mu) != sign(y)) proporzionalmente
a |y|: le predizioni sbagliate su movimenti ampi costano di più. `dv_lambda=0.3` configurabile,
0.0 disabilita. Complementare alla asymmetry_penalty (che agisce sulla NLL).

### CPU fraction centralizzata — `config/default.yaml`, tutti gli script
`hardware.cpu_fraction` in `config/default.yaml` controlla la percentuale di core CPU
usata da tutti gli script (default 0.5 = 50%). Tutti i 6 script (`run_all.py`,
`02_train.py`, `02b`, `02c`, `03_backtest.py`, `04_live_signals.py`) leggono il valore
dal config all'avvio. Non serve più modificare il codice per cambiare il limite CPU.

### GPU VRAM limit rimosso — tutti gli script
Rimossa la chiamata `torch.cuda.set_per_process_memory_fraction()` da tutti gli script.
Il modello utilizza ora tutta la VRAM disponibile della GPU. Per limitare il compute GPU
(non VRAM), usare `nvidia-smi -pl <watt>` prima del training (RTX 2070 Super TDP=215W).

---

## Iterazione 7 — Ottimizzazioni pipeline distillation

### Fix critico: soft labels shuffle-safe — `scripts/02_train.py`
Le soft labels del teacher erano indicizzate sequenzialmente (`sample_idx`) ma il
dataloader di training usa `shuffle=True`. Le soft labels finivano associate ai
campioni sbagliati. Fix: soft labels integrate nel `TensorDataset` cosi' lo shuffle
le riordina insieme ai dati reali.

### Scoring teacher normalizzato — `run_all.py`
La formula di `_select_best_teacher()` era dominata dalla val_loss (contribuiva 150-200
punti vs 0.5-2.5 per spearman). Ora ogni metrica e' normalizzata in scala 0-1 (min-max
tra le 3 architetture) e pesata: 40% loss, 35% spearman, 25% DA. I valori sono presi
alla best val_loss epoch, non il picco su tutte le epoche.

### Ensemble output naturale — `quantsys/model/ensemble.py`, `scripts/03_backtest.py`, `scripts/04_live_signals.py`
`EnsembleModel.__call__()` restituisce direttamente `(mu, sigma, nu)` in spazio naturale
invece di riconvertire in log-space con `log(expm1(x))` (instabile per valori piccoli).
Rimossa la doppia softplus: backtest e live non ri-applicano piu' la conversione.
Aggiunto `torch.amp.autocast` nel forward dell'ensemble per dimezzare la VRAM su GPU.

### Loss distillation scala-normalizzata — `quantsys/model/distillation.py`
I pesi fissi (1.0 mu + 0.5 sigma + 0.1 nu) non compensavano le scale diverse:
MSE(nu)~0.1 dominava, MSE(mu)~1e-10 era irrilevante. Ora ogni componente e'
divisa per la varianza del teacher. Pesi: 0.5 mu + 0.3 sigma + 0.2 nu.

### Teacher caricato una sola volta — `scripts/02_train.py`
Il teacher veniva caricato 2 volte: una per generare soft labels, una per il transfer
delle output heads. Ora viene mantenuto in memoria e riusato.

### Stress test con segnali pre-calcolati — `scripts/03_backtest.py`
`run_stress_scenario()` non ri-esegue piu' le predizioni del modello. I segnali
`(side, dist)` vengono salvati durante il loop principale e riusati con parametri
fee/slippage diversi. Elimina ~400k iterazioni Python per 2 scenari.

### Student skip se gia' distillati — `run_all.py`
`phase_distill()` fase 2c controlla `config.json` di ogni student: se gia' distillato
dallo stesso teacher, lo salta automaticamente (a meno di `--force-download`).

### QUANTSYS_ARCH ripristinato dopo distillation — `run_all.py`
Dopo `phase_distill()`, i path arch-specifici (ARCH_MODELS_DIR, ARCH_RESULTS_DIR,
MODEL_FILE) vengono aggiornati correttamente per backtest e live.

### Feature count da dataset — `scripts/07_verify_teacher.py`
`n_feat` e `n_dynamic` letti da `data/lstm_dataset.npz` invece di essere hardcoded
(116, 85). Fallback ai valori precedenti se il dataset non esiste.

### rolling_std vectorizzata — `scripts/03_backtest.py`
La rolling std per `SimpleSignalModel` era calcolata con list comprehension Python
(una `pd.Series().rolling().std()` per campione). Sostituita con `np.std` vectorizzato
sugli ultimi 20 return di ogni finestra.

### Transfer heads warning MoE/Quantile — `quantsys/model/distillation.py`
`transfer_output_heads()` ora emette warning esplicito e restituisce 0 se il modello
usa `loss_type="quantile"` o `n_output_experts > 1` (transfer non supportato).

### Cleanup generate_teacher_predictions — `quantsys/model/distillation.py`
Rimossa allocazione lista `"quantiles"` inutile (usata solo per modelli quantile
ma allocata per tutti).

---

## Iterazione 6 — Knowledge Distillation + Ensemble Eterogeneo

### Knowledge Distillation pipeline — `scripts/02_train.py`, `quantsys/model/distillation.py`
Nuova pipeline `--distill` che addestra un teacher (iTransformer) e poi student (LSTM, TCNMamba)
con transfer dei pesi delle output heads + loss mista (0.7 reale + 0.3 distillazione).
Gli student convergono in ~60% delle epoche normali grazie alla calibrazione trasferita.

### Ensemble eterogeneo — `quantsys/model/ensemble.py`
`EnsembleModel.load_heterogeneous()` carica un modello per architettura (iTransformer + LSTM +
TCNMamba) invece di N checkpoint della stessa. Backtest e live usano automaticamente l'ensemble
eterogeneo quando almeno 2 architetture hanno un checkpoint disponibile. Diversita' strutturale
degli errori migliora la robustezza vs ensemble omogeneo (5x stesso seed).

### Script verifica teacher — `scripts/07_verify_teacher.py`
Analizza parametri, complessita', metriche backtest delle 3 architetture e raccomanda
quale usare come teacher. Salva risultati in `models/teacher_analysis.json`.

### Orchestrazione distillation — `run_all.py`
Nuovo flag `--distill` + `--teacher` per `run_all.py`. Automatizza: train teacher →
train student LSTM con distillation → train student TCNMamba con distillation.

---

## Iterazione 5 — Architetture multiple + ottimizzazioni

### iTransformer (QuantiTransformer) — `quantsys/model/__init__.py`
Nuova architettura selezionabile con `--arch itransformer`. Multi-scale embedding su
tre finestre (1min T=120, 5min T=24, 15min T=8), feature type embedding (dynamic/structural),
macro context token prepended se disponibile. N layer pre-norm attention + FFN, mean pool →
output heads. Complessità O(F²)=3025 vs O(T²)=14400 del TFT: 4.7× meno operazioni attention.

### Directory arch-specifiche — `run_all.py`, tutti gli script
`models/lstm/` e `models/itransformer/` per checkpoint separati. `results/lstm/` e
`results/itransformer/` per backtest e segnali live separati. Env var `QUANTSYS_ARCH`
propagata a tutti i subprocess. `load_config(path, arch=)` fonde base + override arch.

### Selezione architettura interattiva — `run_all.py`
Se `--arch` non è passato da CLI, `run_all.py` mostra un prompt con le due opzioni
e aspetta la scelta dell'utente prima di avviare la pipeline.

### 116 features dual-stream — `quantsys/features/__init__.py`
Da 55 a 116 feature: 85 dinamiche (stream A) + 31 strutturali (stream B). Aggiunti
VP multi-scala (short/medium/long), feature di livello assoluto ATH/ATL su 30/90/365g,
momentum lento 7/30/90g, round level, price_vs_ma200m, session position, funding rate.

### HMM multi-restart — `quantsys/macro/regime.py`
`_fit_single` prova `n_restarts=5` seed consecutivi, sopprime i warning durante ogni
tentativo con `warnings.catch_warnings()`, ritorna il modello con log-likelihood massima.
Early exit se un restart converge prima di esaurire le iterazioni. Elimina i "Model is
not converging" warning che comparivano su finestre brevi del burn-in.

### Ottimizzazioni feature engineering — `quantsys/features/__init__.py`
- VP `_fill_interp`: numpy ffill (`maximum.accumulate`) invece di `pd.Series` temporaneo (×12)
- VP value area: `cumsum + searchsorted` invece di loop Python su 420k iterazioni
- `_normalize`: bulk write `df[cols] = X_scaled` invece di loop colonna-per-colonna
- Fix `ChainedAssignmentError` funding rate (pandas 3.x CoW): `inplace` → riassegnazione
- Fix `PerformanceWarning` VP: `pd.concat` bulk invece di insert colonna per colonna

## Iterazione 4 — Ottimizzazioni training

### Flash Attention — `quantsys/model/__init__.py`
`F.scaled_dot_product_attention` al posto dell'attention manuale in `TemporalAttention`.
Abilitato automaticamente su CUDA (kernel fused → ~30% speedup attention).

### Ottimizzazioni DataLoader e eval — `scripts/02_train.py`
- `prefetch_factor=4`: pre-carica 4 batch in anticipo
- `torch.from_numpy()` zero-copy nel caricamento dataset
- `torch.inference_mode()` in `run_eval()` (~5-10% più veloce di no_grad)
- `non_blocking=True` nei `.to(device)` durante eval
- Validazione ogni 2 epoche: dimezza il costo eval sul dataset di val

## Iterazione 3 — Fix 5-10

### Fix 5 — Logging su file (`quantsys/utils/__init__.py`)
`setup_logging()` ora aggiunge un `FileHandler` con timestamp nel nome
(`logs/quantsys_YYYYMMDD_HHMMSS.log`) oltre al `StreamHandler` su stdout.
La scrittura su file è idempotente (non duplica handler se chiamata più volte).

### Fix 6 — `PipelineState` unificato (`quantsys/utils/__init__.py`)
Nuovo oggetto che aggrega in un unico `.pkl` gli scaler delle price features,
il `MacroNormalizer`, la lista ordinata delle colonne e la config del modello.
Eliminata la necessità di caricare 3-4 file separati in inference.
Viene salvato da `01_download_data.py` e aggiornato da `02_train.py`.

### Fix 7 — Spearman ρ e ICIR (`scripts/02_train.py`)
Sostituita la sola `directional_accuracy` con `prediction_metrics()` che calcola:
- **Spearman ρ**: correlazione di rango tra μ predetto e log-return reale
- **Weighted Hit Rate**: DA pesata per la grandezza del movimento
- **IC medio** e **ICIR**: consistenza del segnale su finestre rolling da 50 step
Il log per epoch ora mostra `DA=x.xxx  ρ=+x.xxxx`.

### Fix 8 — LR separato per MacroEncoder (`scripts/02_train.py`)
Il `MacroEncoder` usa `lr = lr_base / 10` per le prime epoche, evitando
che il suo gradiente rumoroso destabilizzi il branch price della LSTM.
Implementato con due param groups in `AdamW`. Attivo solo quando `has_macro=True`.

### Fix 9 — Monte Carlo guidato dalla LSTM (`quantsys/model/forecast.py`)
Nuovo modulo che sostituisce il random walk con volatilità storica.
Ad ogni step: la LSTM predice `(μ_t, σ_t, ν_t)` sull'intera batch di path
in un solo forward pass, campiona log-return dalla t-Student parametrica,
aggiorna la finestra autoregressivamente. `σ_eff = √(σ_lstm × σ_garch)` combina
la previsione della rete con il clustering GARCH della volatilità.

### Fix 10 — Test unitari (`tests/test_features.py`)
8 classi di test che coprono i bug silenti più pericolosi:
- **Log-return stazionari**: media vicina a zero, nessun inf/NaN
- **Target corretto**: `target_ret[t] == log_ret[t+1]`
- **VWAP nella banda H-L**: impossibile essere fuori range
- **Split senza overlap**: `max(t_train) < min(t_val) < min(t_test)`
- **Dimensioni split corrette**: frazioni rispettate ±2%
- **HMM probabilità**: ogni riga somma a 1, nessun valore negativo
- **NLL differenziabile**: gradiente fluisce verso μ, log_σ², log_ν
- **PipelineState round-trip**: save → load restituisce dati identici

---

## Iterazione 2 — Fix 1-4

### Fix 1 — Release lag FRED (`quantsys/macro/__init__.py`)
Aggiunto `RELEASE_LAG_DAYS` (D=1, W=4, M=35, Q=35 giorni) e
`SERIES_LAG_OVERRIDE` per serie specifiche. `fetch_all()` shifta l'indice
di ogni serie prima del `ffill`. Merge con `pd.merge_asof(direction="backward")`.

### Fix 2 — Volume Profile incrementale (`scripts/04_live_signals.py`)
`LiveFeatureBuffer` mantiene `_vp_bins` (array) e `_vp_contribs` (deque).
Ogni `push()` aggiorna in O(1) invece di O(N). Reset completo ogni 60 candele.
`np.convolve` sostituito con `pd.Series.ewm()` e `rolling(min_periods=1)`.

### Fix 3 — Persistenza stato WS (`scripts/04_live_signals.py`)
`_save_state()` (write atomica) e `_load_state()` con verifica età (< 5 min).
`warmup()` prova prima il ripristino da disco, poi colma il gap con REST API.
Posizione aperta, portfolio e buffer sopravvivono a disconnessioni del WS.

### Fix 4 — SimpleSignalModel (`scripts/03_backtest.py`)
Rolling statistics pre-calcolate sull'intero test set prima del loop.
EWM pandas per fast/slow mean, `rolling(min_periods=3).std()` per volatilità.
ν dinamico: varia con la volatilità osservata (3-12).

---

## Iterazione 1 — Progetto iniziale

Pipeline completa: Binance REST+WS, feature engineering (55 features),
LSTM→GRU→t-Student NLL, Monte Carlo GARCH, Risk Manager Kelly, Backtest,
MacroEncoder HMM, Dashboard React Bloomberg-style.
