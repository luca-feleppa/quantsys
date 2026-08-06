# QUANTSYS — Changelog

🇮🇹 Ordine cronologico inverso (la voce più recente in alto). Le "Iterazioni" 1-10 sono il blocco direzionale storico (1m); dal pivot 1h in poi le voci sono datate per sessione e tracciate in dettaglio in `STATUS.md` (lab notebook append-only). Questo file riassume i milestone; `STATUS.md` è la fonte canonica.

**EN** Reverse chronological order (newest on top). "Iterations" 1-10 are the historical directional block (1m); from the 1h pivot onward entries are dated per session and tracked in detail in `STATUS.md` (append-only lab notebook). This file summarizes milestones; `STATUS.md` is the canonical source.

---

## 2026-08-06 — Un campione che sembrava fermo, e il guard che protegge solo da un lato · A sample that looked stalled, and the guard that protects on one side only

🇮🇹 **Il contatore del campione confermativo E1 stadio 2 non era a 0: era a 6.** Due difetti sovrapposti. ① Lo `0/40` in continuità **non era una misura** — era il valore di apertura del 01/08 riportato in avanti per cinque giorni, perché quel contatore non era nella routine di sessione e nessuno lo rileggeva. ② Anche misurandolo avrebbe detto **2, non 5**: `raw_candles.parquet` era fermo al 02/08 e `realized_rv` restituiva `None` su ogni expiry successiva. ⚠ La combinazione è peggiore della somma: ① garantisce che nessuno guardi, ② che chi guardasse veda un numero basso e **plausibile**. Diagnosi verificata, non congetturata: i **7 tick di decisione esistono tutti** per ogni expiry dal 01/08 al 07/08, mancava solo la serie dei close. Dopo `01_update_data.py --candles-only` (66.435 → 66.530 barre; `features.parquet`, `lstm_dataset.npz`, scaler e `PipelineState` **non toccati**) il conteggio passa a **6/40**, finestra 01/08 → 06/08, **1 osservazione al giorno dall'apertura**. **Nessun dato perso:** i tick vivono in `forecasts.parquet`, che si merge append-only dal VPS — il ritardo era di **osservabilità**, non di raccolta, ed è **recuperabile**, a differenza di un vintage macro promosso dentro un campione aperto. ETA invariata: n≥40 al **09-10/09**.

**EN** **The E1 stage-2 confirmatory sample counter was not at 0: it was at 6.** Two overlapping defects. ① The `0/40` in the continuity log **was not a measurement** — it was the 01/08 opening value carried forward for five days, because that counter was not in the session routine and nobody re-read it. ② Even measured it would have said **2, not 5**: `raw_candles.parquet` was stuck at 02/08 and `realized_rv` returned `None` for every later expiry. ⚠ The combination is worse than the sum: ① ensures nobody looks, ② ensures whoever looks sees a low and **plausible** number. Diagnosis verified, not conjectured: all **7 decision ticks exist** for every expiry from 01/08 to 07/08; only the close series was missing. After `01_update_data.py --candles-only` (66,435 → 66,530 bars; `features.parquet`, `lstm_dataset.npz`, scaler and `PipelineState` **untouched**) the count moves to **6/40**, window 01/08 → 06/08, **one observation per day since opening**. **No data lost:** the ticks live in `forecasts.parquet`, merged append-only from the VPS — the lag was one of **observability**, not collection, and it is **recoverable**, unlike a macro vintage promoted inside an open sample. ETA unchanged: n≥40 on **09-10/09**.

🇮🇹 **`-RefreshCandles`: l'estensione delle barre diventa un atto esplicito, non un default.** Audit del codice: automatizzare il refresh sarebbe **meccanicamente sicuro** — no-op se non c'è nulla di nuovo (uscita *prima* della scrittura), dedup che tiene la riga **esistente** quindi la storia non si riscrive mai, mai la barra in formazione, scrittura atomica, e il file **non arriva mai al VPS** (l'unico trasferimento casa→VPS è la macro). Resta comunque un flag per tre ragioni scritte nel repo: (i) il blocco ③ è descritto come *off-path e a scrittura zero*, ed è una categorica su cui un lettore futuro si appoggia; (ii) l'estensione fa avanzare lo staleness B7, che a ≥168 barre lancia da solo il refresh incrementale del regime — **oggi 95/168**; (iii) **decisiva**, con l'estensione automatica **congelare i dati diventa impossibile**: l'invariante *«candele/npz/regime_probs non si toccano fino a chiusura gate»* passerebbe da «non fare nulla» a «ricordarsi `-SkipMonitor`», la forma esatta della promozione macro avvenuta per automazione il 31/07. Corollario di provenienza: lo split è una **frazione del conteggio righe**, quindi ogni barra appesa sposta i confini train/val/test al prossimo rebuild — il vintage del dataset diventerebbe funzione di *quante sessioni sono state aperte*, cioè non più dichiarabile. ⚠ Qualificazione onesta: **nessuna pre-reg aperta congela oggi le candele**, quindi l'automazione non avrebbe violato nulla di corrente. Forma adottata: passo ③bis **dopo** il check B7, così l'eventuale refresh del regime parte al **prossimo** avvio e le due scritture restano separate. Fra i gate aperti **solo E1 legge le barre** — verificato nei giudici.

**EN** **`-RefreshCandles`: extending the bars becomes an explicit act, not a default.** Code audit: automating the refresh would be **mechanically safe** — no-op when nothing is new (exit *before* writing), dedup keeping the **existing** row so history is never rewritten, never the in-progress bar, atomic write, and the file **never reaches the VPS** (the only home→VPS transfer is macro). It stays a flag for three reasons written in the repo: (i) block ③ is documented as *off-path and write-free*, a categorical a future reader leans on; (ii) extending moves the B7 staleness counter, which at ≥168 bars launches the incremental regime refresh on its own — **today 95/168**; (iii) **decisive**, under automatic extension **freezing the data becomes impossible**: the *"candles/npz/regime_probs untouched until the gate closes"* invariant would go from "do nothing" to "remember `-SkipMonitor`", the exact shape of the macro promotion that happened by automation on 31/07. Provenance corollary: the split is a **fraction of the row count**, so every appended bar shifts the train/val/test boundaries at the next rebuild — the dataset vintage would become a function of *how many sessions were opened*, i.e. no longer declarable. ⚠ Honest qualification: **no open pre-registration currently freezes the candles**, so automation would have violated nothing current. Adopted form: step ③bis **after** the B7 check, so any regime refresh starts at the **next** startup and the two writes stay separate. Among the open gates **only E1 reads the bars** — verified in the judges.

🇮🇹 **Contatore E1 aggiunto al blocco ③ della routine — e il modo ovvio di farlo era sbagliato.** Il guard `n<40 → NO_RUN` protegge **solo sotto soglia**: a n≥40 uno `--stage 2` nudo calcola le tre condizioni, stampa il verdetto e scrive il report, cioè avrebbe fatto scattare il run confermativo **per automazione invece che per decisione**, il giorno in cui la soglia cade. Aggiunto `--count-only` al giudice E1 (si ferma alla conta a **qualunque** n: nessuna statistica, nessun report), con un test che lo verifica su un campione costruito **sopra** la soglia — sotto soglia il test non proverebbe nulla, perché lo passerebbe anche il comando nudo. Costruire il pannello **è** la conta, ma `x`, `y` e le statistiche non vengono né stampate né scritte: si vede **quante** osservazioni ci sono, mai **quanto valgono**. ⚠ Il refresh delle candele **non è automatizzato di proposito**: il blocco stampa fin dove arriva la serie dei close e avvisa oltre 6h di ritardo, ma il rimedio resta un comando esplicito — automatizzarlo trasformerebbe il blocco ③ da *off-path e a scrittura zero* a *scrive un file di dati a ogni sessione*, la stessa classe di automatismo che il 31/07 ha prodotto una promozione macro non decisa. Suite **491 passed / 1 skipped**; `AVVIO.md` §5.3 e `scripts/README.md` allineati.

**EN** **E1 counter added to routine block ③ — and the obvious way to do it was wrong.** The `n<40 → NO_RUN` guard protects **below threshold only**: at n≥40 a bare `--stage 2` computes the three conditions, prints the verdict and writes the report, i.e. it would have fired the confirmatory run **by automation rather than by decision**, on the day the threshold is met. Added `--count-only` to the E1 judge (it stops at the count at **any** n: no statistics, no report), with a test verifying this on a sample built **above** threshold — below threshold the test would prove nothing, since the bare command would pass it too. Building the panel **is** the count, but `x`, `y` and the statistics are neither printed nor written: one sees **how many** observations exist, never **what they are worth**. ⚠ The candle refresh is **deliberately not automated**: the block prints how far the close series reaches and warns past a 6h lag, but the remedy stays an explicit command — automating it would turn block ③ from *off-path and write-free* into *writes a data file every session*, the same class of automation that produced an undecided macro promotion on 31/07. Suite **491 passed / 1 skipped**; `AVVIO.md` §5.3 and `scripts/README.md` aligned.

---

## 2026-08-05 — La banda pubblicata scritta alla precisione del suo artefatto · The published band written at its artifact's precision

🇮🇹 **Banda ri-espressa: `−23% ÷ −32%` → `−22.42% ÷ −31.65%`. Stessi numeri, nessuna ri-misura, zero GPU** — sono i rapporti NN/HAR-C della coppia canonica (val `0.7757926`, test `0.6834522`) scritti per esteso invece che arrotondati al punto. I due estremi si muovono per ragioni **diverse**: quello **inferiore** era **stale** (−23% è l'arrotondamento del −22.6% misurato da C2 contro **HAR-CJ**, mantenuto quando C3 sostituì il denominatore con HAR-C e mai riallineato — contro HAR-C il valore è sempre stato −22.42%, con lo scarto in direzione **conservativa**); quello **superiore** non era stale ma **arrotondato al punto**, ed era l'unico dei due arrotondato nella direzione che **favorisce** il claim. Allineare le precisioni toglie l'asimmetria e rende il claim **leggermente più conservativo**. ⚠ **I due decimali non sono un intervallo di confidenza:** identificano *quale* coppia (modello, npz, config) produce il numero. L'incertezza misurata resta ~±0.7 punti rispetto alla config di training, 0.0019 rispetto al vintage macro, **zero** rispetto al seed — nessuna delle tre è stocastica, e si elimina dichiarando la coppia. Propagato in `TEORIA.md` §12.2 (con un nuovo paragrafo dedicato IT+EN), `README.md` §5.1 e i commenti di `tests/test_har_c_baseline.py`; **i record datati di C2 e C3 non sono stati riscritti** — dicono quale banda fu decisa allora ed erano veri quando furono scritti.

**EN** **Band re-expressed: `−23% to −32%` → `−22.42% to −31.65%`. Same numbers, no re-measurement, zero GPU** — the canonical pair's NN/HAR-C ratios (val `0.7757926`, test `0.6834522`) written out rather than rounded to the point. The endpoints move for **different** reasons: the **lower** one was **stale** (−23% is the rounding of the −22.6% C2 measured against **HAR-CJ**, carried over when C3 replaced the denominator with HAR-C and never realigned — against HAR-C the value was always −22.42%, the gap being **conservative**); the **upper** one was not stale but **rounded to the point**, and it was the only one rounded in the direction that **favors** the claim. Aligning the precisions removes the asymmetry and makes the claim **slightly more conservative**. ⚠ **The two decimals are not a confidence interval:** they identify *which* (model, npz, config) triple produces the number. Measured uncertainty remains ~±0.7 points with respect to the training config, 0.0019 with respect to the macro vintage, **zero** with respect to the seed — none of them stochastic, all removed by declaring the triple. Propagated to `TEORIA.md` §12.2 (with a new dedicated IT+EN paragraph), `README.md` §5.1 and the `tests/test_har_c_baseline.py` header comments; **the dated C2 and C3 records were not rewritten** — they state which band was decided then, and were true when written.

🇮🇹 **Coda del 02/08 svuotata.** (i) `astype(np.float32, copy=False)` in `to_t()` di `02_train.py`: **−2.42 GiB di picco**, bit-identico — è la seconda metà del picco di cui il `clamp_` in-place era la prima, e le due sono corrette **solo insieme**, perché con `copy=False` il clamp scrive direttamente sul membro npz. L'invariante che lo rende sicuro (`NpzFile.__getitem__` materializza un array fresco a ogni accesso) non è assunta: la inchiodano i 7 test di `tests/test_npz_load_aliasing.py`. (ii) **Guard di fit del walk-forward regime esercitato su dati reali** — la soglia di abort `max_fit_failure_ratio` (0.5, introdotta il 02/08) non era mai stata misurata sul campo, e una soglia di abort mai vista in campo è un rischio di **disponibilità**: se i fit reali fallissero oltre il 50%, il guard renderebbe impossibile il rebuild che doveva proteggere. Probe **read-only** su 2 anni di candele vere (17.520 ore) a cadenza di produzione: **8 fit su 8, `fail_ratio` = 0.000, copertura 100.0%**, margine 0.5 dalla soglia, `last_fit_diagnostics` popolato su un run reale. Nessun file di regime riscritto.

**EN** **02/08 backlog cleared.** (i) `astype(np.float32, copy=False)` in `02_train.py`'s `to_t()`: **−2.42 GiB peak**, bit-identical — the second half of the peak whose first half was the in-place `clamp_`, and the two are correct **only together**, since under `copy=False` the clamp writes straight into the npz member. The invariant making it safe (`NpzFile.__getitem__` materialises a fresh array per access) is not assumed: it is pinned by the 7 tests in `tests/test_npz_load_aliasing.py`. (ii) **Regime walk-forward fit guard exercised on real data** — the `max_fit_failure_ratio` abort threshold (0.5, introduced 02/08) had never been measured in the field, and an abort threshold never seen in the field is an **availability** risk: were real fits to fail above 50%, the guard would make impossible the very rebuild it was meant to protect. **Read-only** probe over 2 years of real candles (17,520 hours) at production cadence: **8 fits out of 8, `fail_ratio` = 0.000, coverage 100.0%**, a 0.5 margin to the threshold, `last_fit_diagnostics` populated on a real run. No regime file rewritten.

🇮🇹 **Refresh macro NON eseguito, deliberatamente.** Due campioni forward pre-registrati aperti dipendono dall'input del modello: hedged (16/20, ~09/08) ed **E1 stadio 2, che è a 0/40 e accumula fino a ~10/09**. Promuovere un vintage oggi spaccherebbe E1 stadio 2 fra due normalizzazioni — e il `MacroNormalizer` è rifittato **whole-df**, quindi un refresh cambia i valori macro anche delle righe storiche. Il vintage si può ri-puntare indietro, le osservazioni forward già raccolte no. Finestra pulita: dopo ~10/09.

**EN** **Macro refresh deliberately NOT performed.** Two open pre-registered forward samples depend on the model input: hedged (16/20, ~09/08) and **E1 stage 2, at 0/40 and accumulating until ~10/09**. Promoting a vintage today would split E1 stage 2 across two normalizations — and the `MacroNormalizer` is refit **whole-df**, so a refresh changes macro values for historical rows too. A vintage can be re-pointed back; forward observations already collected cannot. Clean window: after ~10/09.

🇮🇹 **Gate M1 pre-registrato ed ESEGUITO in giornata: PASS ⓪①②③④, zero GPU.** L'impronta di identità train↔inference copre ora anche il **vintage macro**: `02_train.py` registra nel `PipelineState` l'md5 per split di `X_macro_*` dell'npz consumato (più ordine, conteggio colonne e dtype; fonte `measured`) e i tre giudici vol la ri-calcolano e fail-fastano, con `--allow-macro-mismatch` separata da quella dello scaler. ⓪ **inerzia bit-identica su entrambi gli split** contro i report archiviati di R1 (118 chiavi comuni, **0** differenze numeriche; le sole due sono le etichette di path del blocco `provenance`, 18 chiavi nuove tutte sotto `provenance.macro`, zero perse). ① controllo positivo a due livelli: una singola cella modificata, un riordino di colonne a valori identici, un cambio di dtype e il caso end-to-end devono tutti far scattare il guard — senza, un guard che ritornasse sempre `true` avrebbe superato l'inerzia in modo perfetto. ② i tre model dir anteriori a M1 restano `matches: null` = non verificabile, mai `true`. ③ path live irraggiungibile, verificato per grep **e** con un test parametrico su quattro file: un fail-fast raggiungibile da `04b` fermerebbe il forward test al bootstrap dentro campioni aperti. ④ suite **490 passed / 1 skipped**. ⚠ **Due limiti dichiarati e non chiusi:** il controllo positivo è **sintetico** (il vintage V1 non esiste più e ricostruirlo richiederebbe di riscrivere l'npz congelato), e i tre modelli pre-M1 — artefatto canonico incluso — non porteranno mai l'impronta. ⚠ **Il backfill previsto dalla pre-registrazione NON è stato eseguito, di proposito:** un'impronta ricalcolata oggi dall'npz corrente combacia **per costruzione**, quindi non porta evidenza e convertirebbe solo un onesto `null` in un rassicurante `IDENTICO` — la condizione ② violata dalla porta principale. Fatto meno di quanto il PASS autorizzasse, mai di più.

**EN** **Gate M1 pre-registered and RUN the same day: PASS ⓪①②③④, zero GPU.** The train↔inference identity fingerprint now also covers the **macro vintage**: `02_train.py` records into the `PipelineState` the per-split md5 of the consumed npz's `X_macro_*` (plus column order, count and dtype; source `measured`) and the three vol judges recompute it and fail fast, with `--allow-macro-mismatch` kept separate from the scaler escape. ⓪ **bit-identical inertia on both splits** against R1's archived reports (118 common keys, **0** numeric differences; the only two are path labels in the `provenance` block, 18 new keys all under `provenance.macro`, none lost). ① two-level positive control: a single changed cell, a column reordering at identical values, a dtype change and the end-to-end case must each trip the guard — without it, a guard always returning `true` would have passed the inertia check perfectly. ② the three pre-M1 model dirs stay `matches: null` = not verifiable, never `true`. ③ live path unreachable, verified by grep **and** by a parametrized test over four files: a fail-fast reachable from `04b` would stop the forward test at bootstrap inside open samples. ④ suite **490 passed / 1 skipped**. ⚠ **Two limits declared and not closed:** the positive control is **synthetic** (vintage V1 no longer exists and rebuilding it would rewrite the frozen npz), and the three pre-M1 models — canonical artifact included — will never carry the fingerprint. ⚠ **The backfill foreseen by the pre-registration was deliberately NOT performed:** a fingerprint recomputed today from the current npz matches **by construction**, hence carries no evidence and would merely convert an honest `null` into a reassuring `IDENTICAL` — condition ② violated through the front door. Less was done than the PASS authorized, never more.

🇮🇹 **Pre-registrazione M1 come fu scritta:** estendere l'impronta di identità train↔inference al **vintage macro** dell'npz. Oggi il guard copre il RobustScaler dei prezzi e `target_scale` ma non la normalizzazione macro, che non vive nel `PipelineState` canonico — quindi due modelli addestrati su macro diverse passano entrambi `matches: true`, con uno scarto misurato di `0.0019` sul rapporto pubblicato. Impronta scelta ex-ante: md5 di `X_macro_train` letto dall'npz al training e persistito nello stato del modello. Costo **zero GPU**; pre-registrata comunque perché tocca l'impronta che decide se un numeratore è confrontabile. Condizioni: inerzia bit-identica del numeratore pubblicato su entrambi gli split, **controllo positivo** (il guard deve fallire quando deve — senza, un guard che ritorna sempre `true` supererebbe l'inerzia in modo perfetto), nessun `null` confuso con "verificato", e **il path live irraggiungibile dal nuovo fail-fast** (fermerebbe il forward test dentro tre campioni aperti). ⚠ Limite dichiarato ex-ante: il vintage V1 non esiste più e ricostruirlo richiederebbe di riscrivere l'npz congelato, quindi il controllo positivo è **sintetico** — dimostra che il guard distingue due macro diverse, non che avrebbe intercettato quell'evento storico.

**EN** **Pre-registration M1 as written:** extend the train↔inference identity fingerprint to the npz's **macro vintage**. Today the guard covers the price RobustScaler and `target_scale` but not macro normalization, which does not live in the canonical `PipelineState` — so two models trained on different macro both pass `matches: true`, with a measured gap of `0.0019` on the published ratio. Fingerprint chosen ex-ante: md5 of `X_macro_train` read from the npz at training time and persisted in the model state. Cost **zero GPU**; pre-registered regardless, because it touches the fingerprint deciding whether a numerator is comparable. Conditions: bit-identical inertia of the published numerator on both splits, a **positive control** (the guard must fail when it should — without it, a guard always returning `true` would pass the inertia check perfectly), no `null` conflated with "verified", and **the live path unreachable from the new fail-fast** (it would stop the forward test inside three open samples). ⚠ Limit declared ex-ante: vintage V1 no longer exists and rebuilding it would rewrite the frozen npz, so the positive control is **synthetic** — it shows the guard distinguishes two different macro sets, not that it would have caught that historical event.

---

## 2026-08-04 — La banda pubblicata torna verificabile, e la sua incertezza dichiarata era una deduzione · The published band becomes verifiable again, and its declared uncertainty was an inference

🇮🇹 **Gate R1 eseguito e chiuso: PASS ⓪①②③④.** La coppia canonica modello ↔ npz è stata addestrata a config di produzione **invariata** sull'npz congelato e vive come artefatto permanente in `models/canonical_1h_vols/` (5 checkpoint, `pipeline_state.pkl`, i due report del giudice, `PROVENANCE.md` col legame modello↔dataset e le impronte di scaler). Esiti: ⓪ baseline riprodotte **cifra per cifra** su entrambi gli split (val n=6485, test n=6486) → stesso npz; ① `provenance.matches = true` **senza** via di fuga; ② gate storico contro **HAR-RV** superato (val 0.26143 ≤ 0.33913; test 0.23637 ≤ 0.35149, entrambi ≪ naive); ③④ materialità rispettata contro **HAR-C** — **0.7758 su val (−22.4%) e 0.6835 su test (−31.7%)** contro soglie pre-dichiarate 0.80 e 0.71. **La banda pubblicata non cambia di una cifra**: il numeratore coincide **esattamente** con quello già pubblicato. Cambia lo **statuto** del claim — da affermazione sul protocollo di addestramento ad artefatto ri-giudicabile (`dev_vols_qlike.py --arch canonical_1h_vols`). ⚠ **Nessuna promozione**: `models/itransformer` e il VPS restano intoccati finché i campioni forward pre-registrati sono aperti.

🇮🇹 **Ritrattazione: «~0.2 punti di incertezza di seed-draw» era una deduzione, non una misura, ed è falsa.** §12.2 attribuiva lo scarto fra due repliche (val 0.26206 vs 0.26143) al sorteggio RNG. Misurato: a seed, config e npz fissi il protocollo è **deterministico** — la coppia canonica riproduce una replica precedente alla **decima cifra su entrambi gli split** (delta 0.000e+00), quindi la dispersione di ri-addestramento è **zero**. I report su disco formano due cluster deterministici e la differenza fra i due **non è rumore di seed** ma di config o di versione del codice, **non identificata**: scritta come tale in §12.2 IT+EN. È la seconda ritrattazione in tre giorni con la stessa forma — una causa plausibile scritta senza l'esperimento che la testava.

🇮🇹 **Tre difetti trovati eseguendo il gate, nessuno dei quali era l'esperimento.** ① **Il guard sullo scaler era cieco in sandbox:** `check_model_dataset_scaler` risolveva il canonico via `models_root()`, che sotto `QUANTSYS_MODELS_ROOT` punta **dentro** la sandbox, dove il canonico non esiste mai → `matches=None` e un warning, cioè nessun controllo **proprio nella modalità in cui si giudicano i candidati**, l'unico caso d'uso del guard. Nuova `canonical_state_path()`: canonico locale alla sandbox se esiste (esperimento con npz proprio), altrimenti quello della root di default; giudice ri-eseguito su val col fix, `nn_qlike` **bit-identico** → il fix tocca la provenienza, non la metrica. ② **Il nome del report non conteneva l'arch:** giudicare un artefatto che vive come dir-arch dentro `models/` avrebbe scritto sul nome NUDO `qlike_report_1h_val.json`, **sopra il report storico di produzione** citato in §12.2, uscendo 0 e stampando un PASS corretto — distruzione silenziosa la cui unica traccia sarebbe stato un file con numeri plausibili. Regola estratta in `report_filename()`, path production invariato nel nome, non-clobber verificato con md5. ③ **`--arch` non ammetteva l'artefatto**, che sarebbe stato un checkpoint dichiarato riproducibile e **non verificabile** — il difetto stesso che R1 chiude. Sette test nuovi: `tests/test_qlike_report_naming.py` (4, di cui uno pretende che val e test non collidano mai perché il test split è one-shot) + 3 in `tests/test_scaler_identity_guard.py` (uno gira **con la env sandbox attiva** e pretende `matches is True`; uno è la sentinella sull'artefatto canonico).

🇮🇹 **Braccio diagnostico B (solo val): speedup 1.61×, adozione RESPINTA, e il risultato utile è il Δ.** Riaddestrando a `batch_size 128`/`ga 1` il wall-clock scende da 28.5 a 17.7 min (≥1.5× ✅) ma il rapporto NN/HAR-C passa da 0.7758 a **0.7690**, cioè `|Δ| = 0.0068` contro una soglia di adozione pre-registrata di 0.005 → **la config di produzione resta `64`/`2`**. ⚠ B è *migliore* su val e non conta: il ruolo dei bracci era fissato ex-ante, sceglierlo a risultati visti sarebbe selezione sull'esito. Il numero informativo è che un knob **puramente computazionale** — a dati, architettura e seed identici — sposta il rapporto pubblicato di **3.5× lo scarto** fra i due cluster deterministici che ieri era attribuito al seed-draw. Coppia di fatti complementari misurati nello stesso giorno: **il seed non muove il rapporto (Δ=0), la config sì (Δ=0.0068)**. Ne segue che gli estremi della banda portano **~±0.7 punti percentuali** di incertezza rispetto alla configurazione di training — un'incertezza **non stocastica**, che si elimina **dichiarando** la config invece che mediando su repliche. Config di produzione ripristinata e verificata (`git diff --exit-code` pulito); suite **469 passed, 1 skipped**.

🇮🇹 **I due cluster deterministici: causa identificata — è il VINTAGE MACRO dell'npz, non il codice e non i seed.** Indagine a costo zero (log + git). Due soli rewrite dell'npz (19/07 16:54:57 e 30/07 20:55:02): tutte le run del cluster 0.26206 stanno fra i due, tutte quelle del cluster 0.26143 dopo il secondo, e la prima di queste parte **87 secondi** dopo la riscrittura. Meccanismo verificato nel codice: `01b_download_macro.py` sostituisce **solo** `X_macro_{split}` e lascia intatti `X_*`, `y_*`, `t_*`; la macro è **input del modello** (90 colonne) ma **non entra in nessuna baseline HAR** → NN si sposta, baseline identiche cifra per cifra, che è esattamente la firma osservata. ⚠ Non è un append: il `MacroNormalizer` è **rifittato whole-df**, quindi allungare la serie cambia i valori macro **anche delle righe storiche**. Alternative escluse: seed (Δ=0, misurato), batch (811 batch/epoca ovunque), codice (i due commit della finestra sono uno di solo logging e uno inerte a flag spento, senza `nn.Parameter` aggiunti quindi senza consumo di RNG). ⚠ **Falsa pista degna di nota:** gli SHA delle run del cluster B non esistono più (pre-rewrite del 27/07) e il confine ci cade accanto — sembrava una spiegazione e non poteva esserlo, perché un rewrite di storia non cambia il contenuto dei file. **Buco dichiarato e non chiuso:** il guard di identità copre il RobustScaler dei prezzi e `target_scale`, non la normalizzazione macro (assente dal `PipelineState` canonico), quindi due modelli possono passare `matches: true` ed essere addestrati su macro diverse — **0.0019** di scarto sul rapporto pubblicato, oggi non segnalato da nulla. **Bilancio sulla precisione della banda:** seed 0, refresh macro 0.0019, config di training 0.0068 — tutte fonti **deterministiche e dichiarabili**, nessuna stocastica.

**EN** **Gate R1 run and closed: PASS on ⓪①②③④.** The canonical model ↔ npz pair was trained at **unchanged** production config on the frozen npz and lives as a permanent artifact at `models/canonical_1h_vols/` (5 checkpoints, `pipeline_state.pkl`, both judge reports, a `PROVENANCE.md` carrying the model↔dataset binding and the scaler fingerprints). Outcomes: ⓪ baselines reproduced **digit for digit** on both splits (val n=6485, test n=6486) → same npz; ① `provenance.matches = true` **without** the escape hatch; ② the historical gate against **HAR-RV** survives (val 0.26143 ≤ 0.33913; test 0.23637 ≤ 0.35149, both far below naive); ③④ materiality met against **HAR-C** — **0.7758 on val (−22.4%) and 0.6835 on test (−31.7%)** against pre-declared thresholds of 0.80 and 0.71. **The published band does not change by a single digit**: the numerator matches **exactly** the published one. What changes is the claim's **status** — from a statement about the training protocol to a re-judgeable artifact (`dev_vols_qlike.py --arch canonical_1h_vols`). ⚠ **No promotion**: `models/itransformer` and the VPS stay untouched while pre-registered forward samples are open.

**EN** **Retraction: "~0.2 points of seed-draw uncertainty" was an inference, not a measurement, and it is false.** §12.2 attributed the gap between two replicas (val 0.26206 vs 0.26143) to the RNG draw. Measured: at fixed seeds, config and npz the protocol is **deterministic** — the canonical pair reproduces an earlier replica to the **tenth digit on both splits** (delta 0.000e+00), so retraining dispersion is **zero**. The reports on disk form two deterministic clusters and the difference between them **is not seed noise** but a config or code-vintage difference, **not identified**: written as such in §12.2 IT+EN. It is the second retraction in three days with the same shape — a plausible cause written without the experiment that would test it.

**EN** **Diagnostic arm B (val only): 1.61× speedup, adoption REJECTED, and the useful result is the Δ.** Retraining at `batch_size 128`/`ga 1` cuts wall-clock from 28.5 to 17.7 min (≥1.5× ✅) but moves the NN/HAR-C ratio from 0.7758 to **0.7690**, i.e. `|Δ| = 0.0068` against a pre-registered adoption threshold of 0.005 → **production config stays at `64`/`2`**. ⚠ B is *better* on val and that does not count: the arms' roles were fixed ex ante, and picking one after seeing the results would be selection on the outcome. The informative number is that a **purely computational** knob — at identical data, architecture and seeds — moves the published ratio by **3.5× the gap** between the two deterministic clusters that was attributed to seed draw yesterday. Two complementary facts measured on the same day: **the seed does not move the ratio (Δ=0), the config does (Δ=0.0068)**. It follows that the band's endpoints carry about **±0.7 percentage points** of uncertainty with respect to the training configuration — an uncertainty that is **not stochastic** and is removed by **declaring** the config rather than averaging over replicas. Production config restored and verified (`git diff --exit-code` clean); suite **469 passed, 1 skipped**.

**EN** **The two deterministic clusters: cause identified — it is the npz's MACRO VINTAGE, not the code and not the seeds.** Zero-cost investigation (logs + git). Only two npz rewrites (19/07 16:54:57 and 30/07 20:55:02): every run in the 0.26206 cluster sits between them, every run in the 0.26143 cluster after the second, and the first of those starts **87 seconds** after the rewrite. Mechanism verified in the code: `01b_download_macro.py` replaces **only** `X_macro_{split}` and leaves `X_*`, `y_*`, `t_*` untouched; macro is a **model input** (90 columns) but enters **no HAR baseline** → the NN moves while baselines stay identical digit for digit, exactly the observed signature. ⚠ It is not an append: the `MacroNormalizer` is **refit whole-df**, so extending the series changes macro values **for historical rows too**. Alternatives excluded: seed (Δ=0, measured), batch (811 batches/epoch throughout), code (the window's two commits are one logging-only and one inert at flag-off, adding no `nn.Parameter` hence consuming no RNG). ⚠ **A false lead worth recording:** cluster B's run SHAs no longer exist (pre-rewrite, 27/07) and the boundary falls next to them — it looked like an explanation and could not be one, since rewriting history does not change file contents. **Declared, unclosed gap:** the identity guard covers the price RobustScaler and `target_scale`, not macro normalization (absent from the canonical `PipelineState`), so two models can both pass `matches: true` while trained on different macro — **0.0019** of published-ratio gap, today flagged by nothing. **Balance sheet on the band's precision:** seed 0, macro refresh 0.0019, training config 0.0068 — all **deterministic and declarable** sources, none stochastic.

**EN** **Three defects found while running the gate, none of which was the experiment.** ① **The scaler guard was blind under sandbox:** `check_model_dataset_scaler` resolved the canonical via `models_root()`, which under `QUANTSYS_MODELS_ROOT` points **inside** the sandbox, where no canonical ever exists → `matches=None` plus a warning, i.e. no check at all **in exactly the mode where candidates are judged**, the guard's only use case. New `canonical_state_path()`: sandbox-local canonical if present (an experiment with its own npz), otherwise the default-root one; the judge was re-run on val with the fix and `nn_qlike` came out **bit-identical** → the fix touches provenance, not the metric. ② **The report name did not carry the arch:** judging an artifact living as an arch-dir inside `models/` would have written to the BARE name `qlike_report_1h_val.json`, **over the historical production report** cited in §12.2, exiting 0 and printing a correct PASS — silent destruction whose only trace would have been a file holding plausible numbers. Rule extracted into `report_filename()`, production path name unchanged, non-clobber verified by md5. ③ **`--arch` did not accept the artifact**, which would have been a checkpoint declared reproducible yet **not verifiable** — the very defect R1 closes. Seven new tests: `tests/test_qlike_report_naming.py` (4, one of which demands val and test never collide because the test split is one-shot) + 3 in `tests/test_scaler_identity_guard.py` (one runs **with the sandbox env set** and demands `matches is True`; one is the sentinel on the canonical artifact).

---

## 2026-08-03 — Un allarme che era un artefatto di misura, e la coppia canonica pre-registrata · An alarm that was a measurement artifact, and the canonical pair pre-registered

🇮🇹 **Il "buco del recorder L2 del 02/08" non è mai esistito, e la causa è stata rimossa.** L'ora incriminata ha 720/720 snapshot e il run contiguo è 462+28=490h: mai spezzato. Il monitor `scripts/vol/l2_continuity_check.py` misurava il **mirror locale** credendo di misurare il recorder — lo span finiva su `ts[-1].floor("h")`, che è **per costruzione** l'ora contenente l'ultimo tick, quindi in corso e parziale; con soglia 360 su cadenza reale 720/h **l'esito dipendeva dal minuto** in cui girava la routine. Secondo canale: il pull scarica i giornalieri con `scp` senza atomicità remota, quindi la coda può arrivare in ritardo di un pull. Fix: ora in corso **esclusa** dallo span (conteggio conservativo), buchi nelle ultime `--provisional-hours` (default 6) marcati **PROVVISORI** ed esclusi dalla stima di costo — *un buco è un fatto solo dopo essere sopravvissuto a un secondo pull* — mentre i buchi consolidati restano un allarme pieno. Logica estratta in `analyze()` per essere testabile: `tests/test_l2_continuity_check.py` (5), fra cui uno che pretende il run corrente invariato su cinque riempimenti dell'ora in corso (1, 60, 359, 361, 720). **Sintomo diagnostico generalizzabile: se l'esito dipende dall'istante in cui giri la misura, è un artefatto di misura, non un fatto sui dati.**

🇮🇹 **Gate R1 pre-registrato (APERTO, mai eseguito): coppia canonica modello ↔ npz.** La banda pubblicata −23% ÷ −32% ha come numeratore una coppia riaddestrata in una sandbox poi eliminata, quindi oggi è un'affermazione sul **protocollo**, non su un artefatto verificabile. R1 lo produce, a config di produzione **invariata** e npz **congelato**, con quattro condizioni pre-dichiarate: ⓪ controllo di vintage **model-independent** (le baseline devono coincidere cifra per cifra: sono fittate dentro l'npz, quindi se divergono il dataset è un altro e non c'è nulla da ri-pubblicare), ① identità dello scaler (`matches: true`, senza `--allow-scaler-mismatch`), ② sopravvivenza del gate storico contro **HAR-RV** (denominatore del *gate*, non del *claim*), ③ materialità `NN/HAR-C ≤ 0.80`. Test one-shot solo a val verde. Braccio diagnostico **Leva B** (`batch_size 128`, `ga 1`) **solo su val**, che per costruzione **non può** diventare la coppia canonica: il ruolo è fissato ex-ante perché sceglierlo a posteriori sarebbe selezione sull'esito. ⚠ Il PASS **non autorizza la promozione**: sostituire il modello di `04b` dentro campioni forward pre-registrati aperti è la stessa violazione del refresh macro dentro un campione. **Due correzioni collaterali:** un rapporto della nota di provenienza in §12.2 riportava il denominatore sbagliato (0.7738 = prima replica contro HAR-CJ, invece di 0.7777 = seconda replica contro HAR-C; lo spread ~0.2 punti e ogni claim pubblico restano invariati), e `scripts/00_check_setup.py` stampava un ✗ **rosso** per ogni artefatto di pipeline assente mentre il verdetto finale diceva "setup verificato" — su un clone fresco quegli artefatti mancano **per definizione** e infatti non concorrono al verdetto: ora sono warning.

**EN** **The "02/08 L2 recorder gap" never existed, and its cause has been removed.** That hour holds 720/720 snapshots and the contiguous run is 462+28=490h: never broken. The monitor measured the **local mirror** while believing it measured the recorder — the span ended at `ts[-1].floor("h")`, **by construction** the hour containing the last tick, hence in progress and partial; with a 360 threshold against a real 720/h rate **the verdict depended on the minute** the routine ran. Second channel: the pull fetches dailies over `scp` with no remote atomicity, so the tail can lag by one pull. Fix: the in-progress hour is **excluded** from the span (conservative count), gaps within the last `--provisional-hours` (default 6) are flagged **PROVISIONAL** and excluded from the cost estimate — *a gap is a fact only after surviving a second pull* — while consolidated gaps stay full alarms. Logic extracted into `analyze()` to be testable: `tests/test_l2_continuity_check.py` (5), one of which demands an unchanged current run across five in-progress-hour fill levels (1, 60, 359, 361, 720). **Generalizable diagnostic symptom: if the verdict depends on the instant you take the measurement, it is a measurement artifact, not a fact about the data.**

**EN** **Gate R1 pre-registered (OPEN, never run): canonical model ↔ npz pair.** The published −23% to −32% band takes as numerator a pair retrained in a sandbox that was later deleted, so today it is a statement about the **protocol**, not about a verifiable artifact. R1 produces one, at **unchanged** production config on a **frozen** npz, with four pre-declared conditions: ⓪ a **model-independent** vintage check (the baselines must match digit for digit: they are fitted inside the npz, so if they diverge the dataset is a different one and there is nothing to re-publish), ① scaler identity (`matches: true`, without `--allow-scaler-mismatch`), ② survival of the historical gate against **HAR-RV** (the *gate*'s denominator, not the *claim*'s), ③ materiality `NN/HAR-C ≤ 0.80`. Test is one-shot, only after val passes. A diagnostic **Lever B** arm (`batch_size 128`, `ga 1`) runs **on val only** and by construction **cannot** become the canonical pair: the role is fixed ex ante because picking it afterwards would be selection on the outcome. ⚠ A PASS **does not authorize promotion**: swapping `04b`'s model inside open pre-registered forward samples is the same violation as refreshing macro inside a sample. **Two collateral corrections:** one ratio in the §12.2 provenance note carried the wrong denominator (0.7738 = first replica against HAR-CJ, instead of 0.7777 = second replica against HAR-C; the ~0.2-point spread and every public claim are unchanged), and `scripts/00_check_setup.py` printed a **red** ✗ for every missing pipeline artifact while the final verdict said "setup verified" — on a fresh clone those artifacts are missing **by definition** and indeed do not feed the verdict: they are warnings now.

---

## 2026-08-02 — Audit di performance: il training è launch-bound, e due fallimenti silenziosi in meno · Performance audit: training is launch-bound, and two fewer silent failures

🇮🇹 **Ricognizione diagnostica (`docs/PERF_AUDIT.md`, nessuna ottimizzazione applicata) più i due fix che ne sono usciti.** Il risultato centrale ribalta l'intuizione: **il training non è compute-bound**. A batch 64 la GPU sta al 5-15% di SM e il tempo per step è dominato dal **lancio** dei kernel — passando da batch 32 a 128 il lavoro aritmetico quadruplica e il wall-clock cresce del 16%; la curva diventa lineare solo oltre batch 512 (SM 96-98%). Conseguenza: i lever che riducono l'aritmetica (`channels_last`, Numba, estensioni native, Polars) non toccano il collo di bottiglia. `torch.compile(backend="cudagraphs")` — l'unico che aggredisce i lanci — misura **1.56×** sullo step (15.30 → 9.79 ms), senza Triton (non installabile da PyPI su Windows) e senza graph break (Dynamo traccia 1 grafo, 88 op); ⚠ **non applicato**: cambia i pesi, quindi richiede un gate pre-registrato. Scartati con la ragione tecnica: `channels_last` (**nessun tensore 4D NCHW** nel progetto — solo Conv1d e attention batched), Numba (il bootstrap CI è **già** una matrice NumPy `(5000,n)`, 37 ms; l'event loop del backtest costa 0.2-0.9 s ed è pieno di Enum/dataclass), estensione nativa (nessun componente insieme pesante **e** isolato; il calcolo più lungo — regime walk-forward — vive dentro statsmodels ed è già risolto da B7), Polars nel `FeatureBuilder` (il data prep completo è **2.24 s**, di cui il 59% è il loop Python del Volume Profile che Polars non esprime; e il prototipo mostra che `ret_skew_20` — feature della lista-104 — cambierebbe del **7.7%** perché Polars usa lo stimatore **biased** e pandas l'**unbiased**: differenza di definizione, non di arrotondamento). Individuata la leva col miglior rapporto guadagno/complessità, **non applicata**: i clip bounds `np.nanpercentile` costano **31-36 s** per invocazione di `02_train` ordinando 647M celle che sono **52.001 barre distinte ripetute 120×** (stride 1); calcolarli sulle barre distinte è **200× più veloce** ma sposta 34 colonne su 104 → è una leva, non una pulizia.

🇮🇹 **Due fallimenti silenziosi rimossi** (entrambi bit-invarianti sul path di successo, entrambi emersi dall'audit e non cercati). ① **Ordine di inizializzazione DLL:** caricare pyarrow dopo torch **e** scikit-learn provoca un'access violation (exit 139, nessun traceback) al primo `read_parquet` — conflitto fra i runtime OpenMP dei due. Gli script numerati sopravvivevano solo perché importano `pandas` alla riga 30 e `torch` alla 31: un invariante **di fatto, mai dichiarato né testato**, che uno script nuovo scritto nell'ordine naturale (progetto prima, pandas poi) violerebbe. `import pyarrow` ancorato in `quantsys/__init__.py` lo rende una proprietà del package. ② **Degradazione silenziosa del walk-forward regime:** ogni fit Markov-Switching fallito finiva in un `log.warning` per timestep e il loop proseguiva; con `current_params=None` le probabilità non venivano mai scritte e `fit_predict_walkforward` restituiva la **prior uniforme travestita da regimi**, senza che nulla fallisse (e il ramo `_fit_single → None` non produceva **nemmeno un log**). Ora `RuntimeError` su zero fit riusciti — non disattivabile, perché un walk-forward senza un solo fit non produce informazione — più abort configurabile su `max_fit_failure_ratio` (default 0.5) e diagnostica persistita in `last_fit_diagnostics`; guard rispecchiato in `continue_walkforward`, dove il fallimento produce parametri **stantii** anziché la prior. Test: `tests/test_import_order.py` (4) + `tests/test_regime_fit_guard.py` (8) → **450 passed, 1 skipped**. Bit-parity B7 del regime incrementale invariata.

**EN** **Diagnostic reconnaissance (`docs/PERF_AUDIT.md`, no optimization applied) plus the two fixes it surfaced.** The central result inverts the intuition: **training is not compute-bound**. At batch 64 the GPU sits at 5-15% SM and per-step time is dominated by kernel **launch** — going from batch 32 to 128 quadruples the arithmetic and grows wall-clock by 16%; the curve turns linear only beyond batch 512 (SM 96-98%). Consequence: levers reducing arithmetic (`channels_last`, Numba, native extensions, Polars) do not touch the bottleneck. `torch.compile(backend="cudagraphs")` — the only one attacking launches — measures **1.56×** per step (15.30 → 9.79 ms), without Triton (not installable from PyPI on Windows) and without graph breaks (Dynamo traces 1 graph, 88 ops); ⚠ **not applied**: it changes the weights, hence requires a pre-registered gate. Rejected with the technical reason: `channels_last` (**no 4D NCHW tensor** in the project — only Conv1d and batched attention), Numba (the bootstrap CI is **already** a `(5000,n)` NumPy matrix, 37 ms; the backtest event loop costs 0.2-0.9 s and is full of Enums/dataclasses), a native extension (no component both heavy **and** isolated; the longest computation — the regime walk-forward — lives inside statsmodels and is already solved by B7), Polars in the `FeatureBuilder` (full data prep is **2.24 s**, 59% of it the Volume Profile's Python loop which Polars cannot express; and the prototype shows `ret_skew_20` — a list-104 feature — would change by **7.7%** because Polars uses the **biased** estimator and pandas the **unbiased** one: a definitional difference, not rounding). Identified the best gain/complexity lever, **not applied**: the `np.nanpercentile` clip bounds cost **31-36 s** per `02_train` invocation, sorting 647M cells that are **52,001 distinct bars repeated 120×** (stride 1); computing them on distinct bars is **200× faster** but moves 34 of 104 columns → a lever, not a cleanup.

**EN** **Two silent failures removed** (both bit-invariant on the success path, both surfaced by the audit rather than sought). ① **DLL initialization order:** loading pyarrow after both torch **and** scikit-learn causes an access violation (exit 139, no traceback) at the first `read_parquet` — a clash between the two OpenMP runtimes. The numbered scripts survived only because they import `pandas` at line 30 and `torch` at line 31: a **de facto invariant, never stated nor tested**, which a new script written in the natural order (project first, pandas second) would violate. `import pyarrow` anchored in `quantsys/__init__.py` makes it a package property. ② **Silent regime walk-forward degradation:** every failed Markov-Switching fit produced one `log.warning` per timestep and the loop carried on; with `current_params=None` probabilities were never written and `fit_predict_walkforward` returned the **uniform prior dressed up as regimes**, without anything failing (and the `_fit_single → None` branch produced **not even a log**). Now `RuntimeError` on zero successful fits — not disableable, because a walk-forward without a single fit yields no information — plus a configurable abort on `max_fit_failure_ratio` (default 0.5) and diagnostics persisted in `last_fit_diagnostics`; the guard is mirrored in `continue_walkforward`, where failure yields **stale** parameters instead of the prior. Tests: `tests/test_import_order.py` (4) + `tests/test_regime_fit_guard.py` (8) → **450 passed, 1 skipped**. B7 incremental-regime bit-parity unchanged.

---

## 2026-07-31 (3) — Strumento vs stato: `MacroNormalizer` pinnabile, gate E1 pre-registrato · Instrument vs state: pinnable `MacroNormalizer`, E1 pre-registered

🇮🇹 **Seconda metà del problema macro.** I vintage datati (voce precedente) risolvono *quale* file arriva al VPS; restava che `VolForecaster` **ri-stimasse** il `MacroNormalizer` whole-df a ogni bootstrap, per cui allungare il parquet muove mediana e IQR e **lo strumento di misura cambia insieme allo stato che deve misurare** — 2.7% della variazione totale sul breakpoint del 31/07. Estratta `macro_snapshot()` (due rami: `refit` legacy e pin da disco), aggiunto `scripts/vol/pin_macro_normalizer.py` e il flag `--macro-norm` a `04b` **e al replay**. Parametro **esplicito, mai da env**, stesso principio di `--arch`: una env residua cambierebbe l'input del live in silenzio; e il replay deve poter scegliere il regime in base alla **data** della decisione che riproduce, non all'ambiente. **Inerzia provata end-to-end sul parquet di produzione: 0 differenze su 90 colonne.** Guard fail-fast sulle colonne del pin (ordine compreso): applicare mediana e IQR della colonna sbagliata sarebbe silenzioso e permanente. ⚠ Il vintage sotto cui `models/itransformer` fu addestrato **non è ricostruibile**: il pin ne **fissa** uno dichiarato, non lo recupera; attivarlo oggi è un **no-op di contenuto** e serve a impedire la deriva futura. **Gate E1 pre-registrato ed eseguito allo stadio 1** (esplorativo, nessun verdetto): l'edge NN-vs-IV ha contenuto predittivo sulla varianza realizzata a 30h? Stadio 2 confermativo a n≥40 expiry → ~10 settembre 2026. **Condizione ③ ex-ante su A13a** (pin-close): `n_eff = n_trig`, non `n_posizioni` — le posizioni che non innescano sono bit-identiche sotto le due regole; A13 parcheggiato con condizioni di riapertura datate. Test: 3 nuovi file (18 test). Vedi `STATUS.md` 2026-07-31 sessione 2.

**EN** **The second half of the macro problem.** Dated vintages settle *which* file reaches the VPS; what remained is that `VolForecaster` **refitted** the `MacroNormalizer` whole-df at every bootstrap, so extending the parquet moves median and IQR and **the measuring instrument changes together with the state it must measure** — 2.7% of the 31/07 breakpoint. Extracted `macro_snapshot()` (two branches: legacy `refit` and on-disk pin), added `scripts/vol/pin_macro_normalizer.py` and the `--macro-norm` flag to `04b` **and the replay**. An **explicit parameter, never from env**, same principle as `--arch`: a stale env would silently change the live input, and the replay must pick its regime by the **date** of the decision it reproduces. **Inertia proven end-to-end on the production parquet: 0 differences over 90 columns.** Fail-fast guard on the pin's columns (order included). ⚠ The vintage `models/itransformer` was trained under is **not reconstructible**: the pin **fixes** a declared one; enabling it today is a **content no-op** and prevents future drift. **Gate E1 pre-registered and run at stage 1** (exploratory, no verdict). **A13a ex-ante counting condition**: `n_eff = n_trig`; A13 parked with dated reopening conditions. Tests: 3 new files (18 tests). See `STATUS.md` 2026-07-31 session 2.

---

## 2026-07-31 (2) — Lo snapshot macro del live diventa un artefatto versionato · The live macro snapshot becomes a versioned artifact

🇮🇹 **Il push macro casa→VPS sovrascriveva il canonico a ogni pull, incondizionatamente** — e il 2026-07-31 ha consegnato al live una macro rifrescata **dentro due campioni forward aperti**, prendendo di fatto una decisione che era in sospeso. Il difetto di fondo però non era il push: era che lo snapshot che alimenta `04b` (letto al bootstrap notturno e **congelato** per la giornata) fosse **stato mutabile** invece che artefatto versionato — tanto che il breakpoint del 31/07 non è stato misurabile direttamente, il file vecchio essendo stato sovrascritto e non essendo in git. Riscritto il blocco 0 di `pull_vps_data.ps1` in tre parti: (a) archivio **append-only** `data/macro/macro_features_<YYYYMMDD>.parquet` sul VPS, con `<YYYYMMDD>` = ultima data dell'indice (`scripts/vps/macro_vintage.py`, nuovo) — 716 KB a copia, quindi ogni decisione forward resta riconducibile al suo vintage e il replay torna riproducibile; (b) il canonico diventa un **symlink** all'archivio, così il vintage live si legge con `readlink` e si vede in `ls -l` — nessun marker che possa mentire, il puntatore **è** la verità; (c) ripuntarlo richiede `-PromoteMacro`, quindi **il push smette di essere una decisione e promuovere lo diventa**; a vintage divergente si emette un warning e il live resta dov'era. Scartata l'alternativa "far girare `01b` anche sul VPS": due fetch FRED/yfinance indipendenti divergono **sulla storia** (le serie FRED sono revisionate retroattivamente), il che romperebbe la riproducibilità di `vol_paper_replay.py` — lo strumento con cui è stata provata la parità live↔training — e toglierebbe del tutto l'umano dal loop invece di rimettercelo. Nessuna modifica al path live (`vol_forecaster.py` continua a leggere lo stesso percorso canonico), quindi l'intervento è ammissibile a campioni forward aperti. Test: `tests/test_macro_vintage.py` 4/4 (contratto CLI: stdout = **esattamente** il vintage, fallimenti puliti a stdout vuoto); branch del blocco 0 verificati su 5 casi con `ssh`/`scp` stubbati. Vedi `STATUS.md` 2026-07-31 (sessione 2).

**EN** **The home→VPS macro push overwrote the canonical on every pull, unconditionally** — and on 2026-07-31 it delivered a refreshed macro to the live path **inside two open forward samples**, de facto taking a decision that was still pending. The underlying defect was not the push, though: it was that the snapshot feeding `04b` (read at the nightly bootstrap and **frozen** for the day) was **mutable state** rather than a versioned artifact — so much so that the 31/07 breakpoint could not be measured directly, the old file having been overwritten and not being in git. Block 0 of `pull_vps_data.ps1` rewritten in three parts: (a) an **append-only** archive `data/macro/macro_features_<YYYYMMDD>.parquet` on the VPS, `<YYYYMMDD>` = last index date (`scripts/vps/macro_vintage.py`, new) — 716 KB per copy, so every forward decision stays traceable to its vintage and the replay becomes reproducible again; (b) the canonical becomes a **symlink** into the archive, so the live vintage is readable with `readlink` and visible in `ls -l` — no marker that could lie, the pointer **is** the truth; (c) repointing requires `-PromoteMacro`, so **the push stops being a decision and promoting becomes one**; on a diverging vintage a warning is emitted and the live path stays put. The "run `01b` on the VPS too" alternative was rejected: two independent FRED/yfinance fetches diverge **over history** (FRED series are revised retroactively), which would break the reproducibility of `vol_paper_replay.py` — the tool that proved live↔training parity — and would remove the human from the loop entirely instead of restoring them to it. No live-path change (`vol_forecaster.py` still reads the same canonical path), so the work is admissible with forward samples open. Tests: `tests/test_macro_vintage.py` 4/4 (CLI contract: stdout = **exactly** the vintage, clean failures with empty stdout); block-0 branches verified on 5 cases with stubbed `ssh`/`scp`. See `STATUS.md` 2026-07-31 (session 2).

---

## 2026-07-26 (2) — Diebold-Mariano sul confronto NN-vs-HAR + repo pronto alla pubblicazione · Diebold-Mariano on the NN-vs-HAR comparison + repo ready to publish

🇮🇹 **Il claim "batte HAR del 30% in QLIKE" ha ora un'inferenza, non solo una stima puntuale.** Implementati `qlike_series()` (loss per-campione; `qlike()` ne è la media → formula in un unico punto) e `diebold_mariano()` con varianza **HAC Newey-West (kernel di Bartlett, lag `q = h−1 = 29`)** e correzione small-sample **Harvey-Leybourne-Newbold**: necessaria perché il target somma 30 barre, le finestre si sovrappongono e con varianza iid lo standard error sarebbe sottostimato di ~√h (`n_eff ≈ n/h` ≈ 216, non 6.5k). Esiti su una coppia modello/scaler riaddestrata (5 seed, sandbox, produzione intatta): **val −26.6%, p = 7.3e-05 · test −36.1%, p = 1.7e-06**; il NN batte HAR **in ogni regime**, stress incluso e validato su test. La banda onesta del claim è **−27% ÷ −36%** secondo split e vintage (il −36% e il −30.2% storico misurano popolazioni diverse: finestra di test estesa + modello riaddestrato). Il DM è **descrittivo, non gating**: le soglie pre-registrate restano i rapporti di QLIKE. Test: `tests/test_diebold_mariano.py` 9/9, incluso quello che verifica l'inflazione dello SE su differenziali sovrapposti — senza cui i p-value sarebbero fittizi. **Repo pronto alla pubblicazione:** audit secret PASS su tutte le revisioni, igiene OpSec, attribuzione dell'assistente rimossa (storia riscritta, 141 trailer, albero bit-identico), corpus KILL e protocollo trasferiti in `TEORIA.md` §12. Vedi `STATUS.md` 2026-07-26 ⑦⑧.

**EN** **The "beats HAR by 30% in QLIKE" claim now carries inference, not just a point estimate.** Added `qlike_series()` (per-sample loss; `qlike()` is its mean → one formula, one place) and `diebold_mariano()` with a **HAC Newey-West variance (Bartlett kernel, lag `q = h−1 = 29`)** plus the **Harvey-Leybourne-Newbold** small-sample correction: required because the target sums 30 bars, windows overlap, and under an iid variance the standard error would be understated by ~√h (`n_eff ≈ n/h` ≈ 216, not 6.5k). Outcomes on a retrained model/scaler pair (5 seeds, sandbox, production untouched): **val −26.6%, p = 7.3e-05 · test −36.1%, p = 1.7e-06**; the NN beats HAR **in every regime**, stress included and validated on test. The honest band for the claim is **−27% to −36%** depending on split and vintage (the −36% and the historical −30.2% measure different populations: extended test window + retrained model). DM is **descriptive, not gating**: pre-registered thresholds remain the QLIKE ratios. Tests: `tests/test_diebold_mariano.py` 9/9, including the one verifying SE inflation on overlapping differentials — without which the p-values would be fictitious. **Repo ready to publish:** secret audit PASS across all revisions, OpSec hygiene, assistant attribution removed (history rewritten, 141 trailers, bit-identical tree), KILL corpus and protocol moved into `TEORIA.md` §12. See `STATUS.md` 2026-07-26 ⑦⑧.

---

## 2026-07-26 — Fix unità di misura del contatore hedged + archivio STATUS committato · Hedged-counter unit fixed + STATUS archive committed

🇮🇹 Il contatore automatizzato ieri stampava `hedge_ledger: {n} eventi` accanto alla soglia `n≥20 hedge-attivi`, ma il campione pre-registrato del giudice hedged-vs-unhedged è definito in **trade aperti con hedge attivo**, non in eventi di ledger (1 posizione = open + N rebalance + flatten): con 22 eventi la riga invitava a leggere "22 ≥ 20" e a lanciare **in anticipo** un giudice one-shot. Corretto in `position_key` distinte con ≥1 hedge eseguito (**n=6** al 26/07, giudice atteso ~09-10/08) con l'esclusione pre-dichiarata della posizione parzialmente hedgiata resa **esplicita nel codice**. Un rischio di violazione introdotto dall'automazione, non un errore di dati: la lezione è che **automatizzare un contatore richiede di verificare che l'unità stampata sia quella della pre-registrazione**. Committato anche lo scorporo dello storico `STATUS.md` → `docs/STATUS_ARCHIVE_2026H1.md` (verificato **letterale**: 809 righe rimosse, 0 non ritrovate nell'archivio) con il doc-sync associato. Vedi `STATUS.md` 2026-07-26.

**EN** Yesterday's automated counter printed `hedge_ledger: {n} events` next to the `n≥20 hedge-active` threshold, but the hedged-vs-unhedged judge's pre-registered sample is defined in **trades opened with the hedge active**, not ledger events (1 position = open + N rebalance + flatten): at 22 events the line invited reading "22 ≥ 20" and launching a one-shot judge **early**. Fixed to distinct `position_key` with ≥1 executed hedge (**n=6** on 26/07, judge expected ~09-10/08), with the pre-declared exclusion of the partially-hedged position now **explicit in code**. A risk introduced by the automation, not a data error: the lesson is that **automating a counter requires checking that the printed unit is the pre-registration's unit**. Also committed the `STATUS.md` history split into `docs/STATUS_ARCHIVE_2026H1.md` (verified **literal**: 809 lines removed, 0 missing from the archive) plus the associated doc sync. See `STATUS.md` 2026-07-26.

---

## 2026-07-25 — Routine di sessione automatizzata (blocco ③ monitoraggio vol) · Session routine automated (block ③, vol monitoring)

🇮🇹 `avvio_sessione.ps1` copriva solo ① pull+merge VPS e ② check freshness regime B7, mentre la routine ricorrente ne richiedeva altri 3 passi manuali (dimenticabili). Aggiunto **blocco ③ "monitoraggio ricorrente linea vol"** (`-SkipMonitor` per saltarlo): `derive_mfiv.py` incrementale — **vincolo d'ordine: dopo il merge**, perché legge la chain appena scaricata — + `mfiv_comparator_judge.py --count-only` + stampa in coda dei contatori dei due gate forward (`n executed` di `trades.jsonl`, eventi di `hedge_ledger.jsonl`). **Invariante di disciplina preservato:** `--count-only` calcola solo timestamp e il giudice ha comunque il guard `n<N_MIN=40 → NO_RUN` senza scrivere report → automatizzare il conteggio **non può** produrre peeking; il run one-shot resta MANUALE. Fail-soft con check esplicito di `$LASTEXITCODE` (in PS 5.1 un exe nativo che esce ≠0 NON solleva eccezione: il `try/catch` non basta) e file ri-verificato ASCII-only. Validazione: run `-SkipPull` end-to-end OK, blocco ③ idempotente. Vedi `STATUS.md` 2026-07-25.

**EN** `avvio_sessione.ps1` covered only ① VPS pull+merge and ② B7 regime freshness, while the recurring routine required 3 more forgettable manual steps. Added **block ③ "recurring vol-line monitoring"** (`-SkipMonitor` to skip): incremental `derive_mfiv.py` — **ordering constraint: after the merge**, since it reads the freshly pulled chain — + `mfiv_comparator_judge.py --count-only` + a trailing printout of both forward-gate counters. **Discipline invariant preserved:** `--count-only` computes timestamps only and the judge still holds the `n<N_MIN=40 → NO_RUN` guard without writing a report → automating the count **cannot** produce peeking; the one-shot run stays MANUAL. Fail-soft with explicit `$LASTEXITCODE` checks (in PS 5.1 a native exe exiting ≠0 does NOT raise: `try/catch` is not enough) and the file re-verified ASCII-only. Validation: end-to-end `-SkipPull` run OK, block ③ idempotent. See `STATUS.md` 2026-07-25.

---

## 2026-07-19 → 07-23 — Lever di training sulla linea vol: A8-BIS e B4-bis entrambi FAIL su val · Vol-line training levers: A8-BIS and B4-bis both FAIL on val

🇮🇹 Due gate pre-registrati eseguiti e chiusi negativi, entrambi **contro una baseline riaddestrata sullo stesso dataset esteso** (non contro l'incumbent production). **A8-BIS mixup** (`mixup_alpha` 0→0.2): baseline 0.26206 vs candidato 0.25998 = −0.79% ≪ soglia −3% → FAIL, overlay eliminato, niente test. Lezione metodologica di primo ordine: il −4.94% descrittivo misurato in Fase B era **artefatto di distribution shift** (confronto cross-scaler contro l'incumbent old-scaler) → **mai decidere un lever su confronti attraverso scaler diversi**. **B4-bis DVOL-come-feature** (pre-reg `5a6112d`, close `526659d`): `dev_vols_dvol_append.py` estende X_macro 90→93 (`dvol_log`/`dvol_chg_24h`/`dvol_avail`, asof causale cap 24h, fill mediana train-only, copertura val 1.000, npz production intatto); baseline 0.26206 vs candidato 0.25939 = −1.02% > soglia −3% → **① FAIL** (② sarebbe passata, gate in AND), nessun one-shot su test. Lettura: tenor mismatch 30d→30h + incumbent che già cattura l'informazione IV via lag RV → MSE-log migliora −6% ma non QLIKE, dove conta la calibrazione di σ². Ratificato anche il **pattern-③ standard** (condizione di conteggio per regime verificata ex-ante: se model-independent l'esito è "nessuna conclusione", non un risultato). Vedi `STATUS.md` 2026-07-19/20/23.

**EN** Two pre-registered gates run and closed negative, both **against a baseline retrained on the same extended dataset** (not against the production incumbent). **A8-BIS mixup** (`mixup_alpha` 0→0.2): 0.26206 baseline vs 0.25998 candidate = −0.79% ≪ the −3% threshold → FAIL, overlay deleted, no test. First-order methodological lesson: the −4.94% descriptive figure from Phase B was a **distribution-shift artifact** (cross-scaler comparison against the old-scaler incumbent) → **never decide a lever on comparisons across different scalers**. **B4-bis DVOL-as-feature** (pre-reg `5a6112d`, close `526659d`): `dev_vols_dvol_append.py` extends X_macro 90→93 (causal asof, 24h cap, train-only median fill, val coverage 1.000, production npz untouched); 0.26206 baseline vs 0.25939 candidate = −1.02% vs the −3% threshold → **① FAIL** (② would have passed, AND-gate), no test one-shot. Reading: 30d→30h tenor mismatch + an incumbent already capturing IV information through lagged RV → log-MSE improves −6% but QLIKE does not, and QLIKE is where σ² calibration counts. The **standard ③-pattern** was also ratified (per-regime count condition verified ex-ante: if model-independent the outcome is "no conclusion", not a result). See `STATUS.md` 2026-07-19/20/23.

---

## 2026-07-18 — Gate v1 CHIUSO: FAIL 0/3 · `04b` migrato sul VPS · MFIV@30h (D4) · v1 gate CLOSED: FAIL 0/3

🇮🇹 **Gate v1 del braccio short-vol chiuso al checkpoint pre-registrato n=20: FAIL 0/3 su ENTRAMBI i campioni** (concordi). Always-short +0.0396 → il VRP resta positivo e conferma il braccio, ma la regola v1 non lo monetizza; sbloccato `POST_GATE_V1.md`. **Migrazione `04b` sul VPS** (`quantsys-volpaper` systemd, con `--hedge --hedge-band 0.30 --hedge-conv raw`): il forward v2 parte e il PC di casa diventa **passivo** — ⚠ mai più lanciare `04b` a casa, doppi ordini sulla stessa posizione testnet. **🔴 Bug candele:** la finestra live era bucata dal ~06-24 (fixato + guard aggiunto). **D4 MFIV@30h model-free** (`scripts/vol/derive_mfiv.py`): derivazione OFFLINE retroattiva del var-swap rate VIX-style a tenor 30h + skew 25Δ RR/BF dal raw chain già registrato → colonne PARALLELE, mai nel path decisionale; wedge di convessità MFIV−ATM mediano **+3.45 vol pt** → il break-even short-vol calcolato su IV ATM era **conservativo**. `VolForecaster` promosso da `04b` a `quantsys/model/vol_forecaster.py`. Vedi `STATUS.md` 2026-07-18.

**EN** **Short-vol arm v1 gate closed at the pre-registered n=20 checkpoint: FAIL 0/3 on BOTH samples** (concordant). Always-short +0.0396 → VRP stays positive and confirms the arm, but the v1 rule does not monetize it; `POST_GATE_V1.md` unblocked. **`04b` migrated to the VPS** (`quantsys-volpaper` systemd, `--hedge --hedge-band 0.30 --hedge-conv raw`): the v2 forward test starts and the home PC becomes **passive** — ⚠ never launch `04b` at home again, duplicate orders on the same testnet position. **🔴 Candle bug:** the live window had been holed since ~06-24 (fixed + guard added). **D4 model-free MFIV@30h**: OFFLINE retroactive derivation of the VIX-style var-swap rate at 30h tenor + 25Δ RR/BF skew from the already-recorded raw chain → PARALLEL columns, never in the decision path; median MFIV−ATM convexity wedge **+3.45 vol pt** → the short-vol break-even computed on ATM IV was **conservative**. `VolForecaster` promoted from `04b` into `quantsys/model/vol_forecaster.py`. See `STATUS.md` 2026-07-18.

---

## 2026-07-14 → 07-16 — VPS collector 24/7 + B7 regime incrementale + GJR-1h + A11-A14 · 24/7 VPS collector + B7 incremental regime + 1h GJR + A11-A14

🇮🇹 **VPS collector 24/7** (VPS EU, Ubuntu 24.04, geo-test PASS): servizi systemd `quantsys-iv` / `quantsys-ob` / `quantsys-trades` — il nuovo **`01e_trades_recorder.py`** raccoglie i trade opzioni Deribit production (retention API ~24h verificata → la raccolta è necessariamente FORWARD) per misurare gli spread realizzati vs mark. Host in `config/secrets.yaml → vps.host`, **mai leggerlo né stamparlo**. Sync lato casa: `scripts/vps/pull_vps_data.ps1` (+ fix anti-stallo ssh/scp). **B7 — regime walk-forward incrementale**: il full rebuild salva `data/regime_wf_checkpoint.pkl`, `01b --regime-incremental` appende le sole barre nuove (**minuti invece di ~3h**), `--regime-bootstrap-checkpoint` ricostruisce lo stato con replay-validation fail-fast; bit-parity garantita da `tests/test_regime_incremental.py`, bootstrap validato bit-exact. Sul rebuild 7 anni le **etichette dei regimi sono RIMAPPATE** (R1 = stress ora): ri-derivarle dalle varianze a ogni full rebuild. **GJR-1h chiuso**: ri-stima QMLE dei parametri MC su rendimenti orari (`scripts/vol/estimate_gjr_1h.py`) → γ≈0.005, leverage effect quasi nullo a 1h; cap σ/barra reso parametrico. **A11-A14 (funzioni gamma):** `pnl_attribution.py` ATTIVA (decomposizione ex-post delta/gamma/theta/vega, read-only); A12 banda WW, A13 pin-close+gamma-cap, A14 sizing-vega implementati **INERTI** (attivazione solo con pre-reg v2 post-gate). Vedi `STATUS.md` 2026-07-14/15/16.

**EN** **24/7 VPS collector** (EU VPS, Ubuntu 24.04, geo-test PASS): `quantsys-iv` / `quantsys-ob` / `quantsys-trades` systemd services — the new **`01e_trades_recorder.py`** records production Deribit options trades (API retention ~24h verified → collection is necessarily FORWARD) to measure realized spreads vs mark. Host lives in `config/secrets.yaml → vps.host`, **never read or print it**. Home-side sync: `scripts/vps/pull_vps_data.ps1` (+ ssh/scp anti-hang fix). **B7 — incremental walk-forward regime**: the full rebuild saves `data/regime_wf_checkpoint.pkl`, `01b --regime-incremental` appends only new bars (**minutes instead of ~3h**), `--regime-bootstrap-checkpoint` rebuilds the state with fail-fast replay validation; bit-parity enforced by `tests/test_regime_incremental.py`, bootstrap validated bit-exact. On the 7-year rebuild the **regime labels are REMAPPED** (R1 = stress now): re-derive them from the variances at every full rebuild. **1h GJR closed**: QMLE re-estimation of the MC parameters on hourly returns → γ≈0.005, near-zero leverage effect at 1h; per-bar σ cap made parametric. **A11-A14 (gamma functions):** `pnl_attribution.py` ACTIVE (ex-post delta/gamma/theta/vega decomposition, read-only); A12 WW band, A13 pin-close+gamma-cap, A14 vega-sizing implemented **INERT** (activation only under a post-gate v2 pre-registration). See `STATUS.md` 2026-07-14/15/16.

---

## 2026-07-08 — Roadmap vol-book v2 + A6 exec-diag in `04b` + refresh macro · Vol-book v2 roadmap + A6 exec-diag + macro refresh

🇮🇹 **Roadmap** (`docs/ROADMAP_VOL_BOOK.md`, sessione advisory 07-07): backlog A1-A10 dall'audit anti-overfit (verdetti chiusi: FrAug/RevIN-vol/MC-dropout/interp-N-HiTS NO) + verdetto strategico book a due strumenti — B1 futures direzionali con leva ❌ (momenti dispari falsificati OOS, E[PnL] = leva·(0−costi) < 0), B2 perp Deribit come **delta-hedge** del book opzioni ✅ (PnL = ∫½ΓS²(σ²impl−σ²real)dt = puro harvest VRP, la quantità che il NN predice), sequencing B3 vincolante. **A6 implementato** (`scripts/04b_vol_paper.py`, unico item pre-gate): `log_exec_diag()` a fine tick → `results/vol_paper/exec_diag.jsonl` (bid/ask/mark/IV/greeks Deribit per leg + delta netto + half-spread; posizione aperta → leg in essere, flat → straddle ATM ipotetico), fail-soft, **regola e costanti pre-registrate INTATTE**; smoke su testnet reale + processo live riavviato (prima riga: half-spread 18.2% ≈ haircut 16% della validazione 06-25). **Refresh macro**: `macro_features.parquet` 06-10→07-08 (sola sezione macro di `01b`, schema 90 col identico; regime/npz/normalizer deliberatamente intatti). Vedi `STATUS.md` 2026-07-08.

**EN** **Roadmap** (`docs/ROADMAP_VOL_BOOK.md`, 07-07 advisory session): A1-A10 backlog from the anti-overfit audit (closed verdicts: FrAug/vol-RevIN/MC-dropout/N-HiTS-interp NO) + two-instrument book strategic verdict — B1 leveraged directional futures ❌ (odd moments falsified OOS, E[PnL] = leverage·(0−costs) < 0), B2 Deribit perp as options-book **delta-hedge** ✅ (PnL = ∫½ΓS²(σ²impl−σ²real)dt = pure VRP harvest, exactly what the NN predicts), binding B3 sequencing. **A6 implemented** (`scripts/04b_vol_paper.py`, the only pre-gate item): `log_exec_diag()` at end of tick → `results/vol_paper/exec_diag.jsonl` (per-leg Deribit bid/ask/mark/IV/greeks + net delta + half-spread; open position → live legs, flat → hypothetical ATM straddle), fail-soft, **pre-registered rule and constants UNTOUCHED**; real-testnet smoke + live process restarted (first row: 18.2% half-spread ≈ the 16% haircut from the 06-25 validation). **Macro refresh**: `macro_features.parquet` 06-10→07-08 (macro-only section of `01b`, identical 90-col schema; regime/npz/normalizer deliberately untouched). See `STATUS.md` 2026-07-08.

🇮🇹 **A2a+A5 eseguiti e FALLITI stesso giorno (gate pre-registrati, zero retrain — la testa quantile era GIÀ nei checkpoint PASS).** A2a (`scripts/vol/dev_vols_quantile_judge.py`): coverage sopra target a tutti i livelli (q50→0.73, q90→0.97 = distribuzione shiftata in alto) e pinball q90 NN perde da HAR+quantili-residui (0.160 vs 0.144) → **A2b (retrain q95) morto**; per il sizing v2 la coda destra è HAR-q90 o conforme da pre-registrare. A5 (`scripts/vol/dev_vols_member_weights.py`, fit 1ª metà val / eval 2ª): ratio 0.9925 vs gate ≤0.97, pesi quasi-uniformi, best-single peggio dell'ensemble → **pesi uniformi confermati ottimali**. Checkpoint production read-only, test split intatto, processi live fermati ~5 min e rilanciati sani.

**EN** **A2a+A5 run and FAILED the same day (pre-registered gates, zero retrain — the quantile head was ALREADY in the PASS checkpoints).** A2a: coverage above target at every level (upward-shifted distribution) and NN q90 pinball loses to HAR+residual-quantiles (0.160 vs 0.144) → **A2b (q95 retrain) dead**; for v2 sizing the right tail is HAR-q90 or a pre-registered conformal recalibration. A5 (fit 1st val half / eval 2nd): ratio 0.9925 vs the ≤0.97 gate, near-uniform weights, best-single worse than the ensemble → **uniform weights confirmed optimal**. Production checkpoints read-only, test split untouched, live processes paused ~5 min and relaunched healthy.

---

## 2026-06-26 — Audit statistico short-vol (1 conclusione corretta, 1 prior smentito) + perf B2/B3 golden Δ=0 · Short-vol statistical audit + B2/B3 perf

🇮🇹 Audit statistico/logico del backtest short-vol (causalità/lookahead verificati PULITI ovunque). Fix in `scripts/vol/`: **①** block-bootstrap CI + N-effettivo — la lag-1 autocorr dei PnL è ≈0 → N_eff≈N(2538), CI>0 (l'overlap 30h/24h NON gonfia la significatività al lag-1; il bootstrap però non cattura la concentrazione temporale 2020-21=90% PnL). **②** Sharpe annualizzato a √(trade/anno)≈√365 (era √292, incoerente con la cadenza giornaliera). **③** haircut bid REGIME-dipendente (`--stress-haircut-mult`) → **CORREGGE la conclusione regime del 06-25**: "edge più alto nello Stress" era un artefatto dell'haircut costante; con spread realistici il **Trending domina** e lo Stress è marginale (il SEGNO regge, la GERARCHIA no). "Non filtrare il regime" SOPRAVVIVE robusto. **④** caveat n=3 load-bearing reso esplicito. **⑤** martingale-correction empirica (flag `--mart-correct` default-OFF): prior SMENTITO (Δ≈5e-3 non ~1e-4 per excess-kurtosis residui ≈19.7 → sovra-corregge sui path di coda) → flag inerte confermato corretto. **A4:** regression test slippage pre-size skip. **Perf B2/B3** (`quantsys/features/__init__.py`, gated golden Δ=0): VP rolling min/max precomputato (era per-finestra, bit-identico — guadagno reale modesto ~1.1×: bincount/argsort co-dominano l'inner-loop) + rimossa `df.copy()` di defrag ridondante; verifica `build(normalize=False)` 122col×2970righe = 0 celle diverse, test `tests/test_vp_golden.py`. Nessun retrain (feature bit-identiche). Vedi `STATUS.md` 2026-06-26.

**EN** Statistical/logical audit of the short-vol backtest (causality/lookahead verified CLEAN throughout). Fixes in `scripts/vol/`: **①** block-bootstrap CI + effective-N — PnL lag-1 autocorr ≈0 → N_eff≈N(2538), CIs>0 (the 30h/24h overlap does NOT inflate lag-1 significance; the bootstrap does not capture the 2020-21=90% temporal concentration). **②** Sharpe annualized at √(trades/yr)≈√365 (was √292, inconsistent with the daily cadence). **③** REGIME-dependent bid haircut (`--stress-haircut-mult`) → **CORRECTS the 06-25 regime conclusion**: "edge highest in Stress" was a constant-haircut artifact; with realistic spreads **Trending dominates** and Stress is marginal (SIGN holds, HIERARCHY doesn't). "Do not filter regime" SURVIVES robustly. **④** load-bearing n=3 caveat made explicit. **⑤** empirical martingale correction (default-OFF `--mart-correct` flag): prior REFUTED (Δ≈5e-3 not ~1e-4 due to residual excess-kurtosis ≈19.7 → over-corrects on tail paths) → inert flag confirmed correct. **A4:** slippage pre-size-skip regression test. **B2/B3 perf** (`quantsys/features/__init__.py`, golden-gated Δ=0): VP rolling min/max precomputed (was per-window, bit-identical — modest ~1.1× real gain: bincount/argsort co-dominate the inner loop) + removed redundant defrag `df.copy()`; `build(normalize=False)` 122col×2970rows = 0 differing cells, test `tests/test_vp_golden.py`. No retrain (bit-identical features). See `STATUS.md` 2026-06-26.

---

## 2026-06-25 — Short-vol arm: backtest storico strutturale FHS GJR-GARCH + validazione premio · Short-vol arm: structural historical backtest

🇮🇹 Secondo braccio della linea vol (short-vol systematic). Backtest strutturale su 7 anni di candele orarie (`scripts/vol/short_vol_hist_backtest.py`): payoff REALE dalle candele (code incluse), premio = fair-value fat-tailed FHS su GJR-GARCH(1,1) × (1+VRP), VRP swept, tutto CAUSALE (refit expanding 90gg, no lookahead). Risultato (n=2538 scadenze daily 08:00 UTC): break-even VRP = **0% per tutte le strutture** → il PnL medio positivo è la raccolta del VRP storico (realized 30h < implied). Strangle 8-10% = struttura tail-safe (hit 97-98%, maxDD −0.33 BTC, Calmar 33.8). Validazione premio (`short_vol_premium_validate.py`): FHS/mark mediano 1.05×, edge sopravvive all'haircut bid 16%. Decomposizione regime/anno (`short_vol_regime_decomp.py`): edge concentrato nell'alta-vol (2020+2021 = 90% del PnL), always-short, NON filtrare il regime. **NON è un PASS del gate** (resta il live n≥20). Vedi `STATUS.md` 2026-06-25.

**EN** Second arm of the vol line (systematic short-vol). Structural backtest over 7 years of hourly candles (`scripts/vol/short_vol_hist_backtest.py`): REAL payoff from candles (tails included), premium = fat-tailed FHS fair-value on GJR-GARCH(1,1) × (1+VRP), VRP swept, fully CAUSAL (90d expanding refit, no lookahead). Result (n=2538 daily 08:00 UTC expiries): break-even VRP = **0% for all structures** → positive mean PnL is the historical VRP harvest (realized 30h < implied). Strangle 8-10% = tail-safe structure (hit 97-98%, maxDD −0.33 BTC, Calmar 33.8). Premium validation (`short_vol_premium_validate.py`): median FHS/mark 1.05×, edge survives the 16% bid haircut. Regime/year decomposition (`short_vol_regime_decomp.py`): edge concentrated in high-vol (2020+2021 = 90% of PnL), always-short, do NOT filter regime. **NOT a gate PASS** (the live n≥20 gate stands). See `STATUS.md` 2026-06-25.

---

## 2026-06-24 — IVS relative-value → KILL net-of-cost · IVS relative-value → KILL net-of-cost

🇮🇹 Probe smile-reversal Deribit (`scripts/vol/ivs_scout.py` + `scripts/vol/ivs_rv_backtest.py`): struttura reale (i residui dello smile revertono, autocorr 0.77) MA netto **−2.3/−3.8 vol-pt/leg** (gross +0.01/+0.04 vs costo round-trip 2.3/3.9 → ~50× sotto lo spread). Morta come price-taker; vivrebbe solo da market-maker. Tetto **economico**, non di dati.

**EN** Deribit smile-reversal probe (`scripts/vol/ivs_scout.py` + `scripts/vol/ivs_rv_backtest.py`): real structure (smile residuals revert, autocorr 0.77) BUT net **−2.3/−3.8 vol-pt/leg** (gross +0.01/+0.04 vs round-trip cost 2.3/3.9 → ~50× below the spread). Dead as a price-taker; would only live as a market-maker. **Economic** ceiling, not a data one.

---

## 2026-06-22 — Robustezza vol: purged k-fold + gate HAR-per-fold + diversità cross-arch · Vol robustness

🇮🇹 Conferma OOS della linea vol oltre il single-split. Purged k-fold QLIKE per arch (`scripts/02b_walkforward_validate.py`, embargo 168h) + baseline HAR fit-per-fold (`scripts/vol/wf_har_baseline.py`, helper in `quantsys/model/vol_metrics.py`): TCN+Mamba batte HAR di ~14% OOS (ratio 0.863, 4/5 fold), tutti gli archi falliscono solo il fold più antico data-starved (strutturale). Kill-check diversità cross-arch sugli errori vol (`scripts/vol/step0_xarch_corr.py`): ρ_err medio 0.83 (vs ≈0.995 direzionale) → la diversità è anch'essa un oggetto pari-specifico. Numeri in `docs/paper/RESULTS_MAP.md` CLAIM 2b/2c.

**EN** OOS confirmation of the vol line beyond the single split. Purged k-fold QLIKE per arch (`scripts/02b_walkforward_validate.py`, 168h embargo) + HAR fit-per-fold baseline (`scripts/vol/wf_har_baseline.py`, helpers in `quantsys/model/vol_metrics.py`): TCN+Mamba beats HAR by ~14% OOS (ratio 0.863, 4/5 folds), all archs fail only the oldest data-starved fold (structural). Cross-arch error-diversity kill-check (`scripts/vol/step0_xarch_corr.py`): mean ρ_err 0.83 (vs ≈0.995 directional) → diversity is itself an even-moment-specific object. Numbers in `docs/paper/RESULTS_MAP.md` CLAIM 2b/2c.

---

## 2026-06-12 — Riorientamento vol-1h + paper "Are price and volume enough?" · Vol-1h reorientation + paper

🇮🇹 Lo stato production diventa la linea vol-1h (`models/itransformer/` = PASS vol-1h, `target_type: log_rv`). Aggiunte le baseline econometriche direzionali come negative-control del paper (`scripts/research/paper_01_dir_baselines.py`): nessuna skill coerente, ρ flippano segno val→test come il NN → il risultato è dell'informazione, non del modello. Cleanup disco ~4.7 GB (dataset/modelli 1m rigenerabili). Documenti paper avviati: `docs/paper/OUTLINE.md`, `docs/paper/RESULTS_MAP.md`. Riorganizzazione script per-linea in sottocartelle (`vol/`, `research/`, `archive/`): mappa in `scripts/README.md`.

**EN** Production state becomes the vol-1h line (`models/itransformer/` = vol-1h PASS, `target_type: log_rv`). Added directional econometric baselines as the paper's negative-control (`scripts/research/paper_01_dir_baselines.py`): no coherent skill, ρ sign-flip val→test like the NN → the result is about information, not the model. ~4.7 GB disk cleanup (regenerable 1m dataset/models). Paper documents started: `docs/paper/OUTLINE.md`, `docs/paper/RESULTS_MAP.md`. Per-line script reorg into subfolders (`vol/`, `research/`, `archive/`): map in `scripts/README.md`.

---

## 2026-06-11 — Probe semivarianza firmata → FAIL · Signed semivariance probe → FAIL

🇮🇹 Target `log(RS⁺/RS⁻)` fwd (signed jump variation, Patton–Sheppard; giudice `scripts/vol/dev_vols_rs_judge.py`): su test NN/HAR-RS MSE 0.9952 (gate ≤0.95), signDA 0.459, e HAR-RS fa peggio della costante → **l'asimmetria è impredicibile per tutti**. Sintesi chiave: i momenti PARI generalizzano OOS, i momenti DISPARI no. Filone HD-firmato chiuso.

**EN** Forward `log(RS⁺/RS⁻)` target (signed jump variation, Patton–Sheppard; judge `scripts/vol/dev_vols_rs_judge.py`): on test NN/HAR-RS MSE 0.9952 (gate ≤0.95), signDA 0.459, and HAR-RS underperforms the constant → **asymmetry is unpredictable for everyone**. Key synthesis: EVEN moments generalize OOS, ODD moments do not. Signed-HD line closed.

---

## 2026-06-10 — VOL-S PASS + pivot 1m→1h KILLED · VOL-S PASS + 1m→1h pivot KILLED

🇮🇹 **VOL-S PASS (B2 positiva):** target `features.target_type: log_rv`, il NN batte HAR-RV del **30% in QLIKE su test** (0.257 vs 0.368; naive 0.807), val→test coerenti (l'anti-correlazione è specifica del target direzionale). Giudice `scripts/vol/dev_vols_qlike.py`. Verifica cross-risoluzione 1m: FAIL su val (NN/HAR 1.013) → edge vol SPECIFICO della risoluzione 1h. **Pivot 1h KILLED:** il 1h sfonda il muro dei costi (|μ|≈43bps ≫ 26bps) ma NON c'è skill direzionale OOS, anti-correlazione val→test confermata anche a 1h, gate 4/4 fallito a 13 E 23 bps → filone "stesso metodo, altro timeframe" chiuso.

**EN** **VOL-S PASS (B2 positive):** target `features.target_type: log_rv`, the NN beats HAR-RV by **30% in QLIKE on test** (0.257 vs 0.368; naive 0.807), val→test coherent (the anti-correlation is specific to the directional target). Judge `scripts/vol/dev_vols_qlike.py`. 1m cross-resolution check: FAIL on val (NN/HAR 1.013) → the vol edge is SPECIFIC to 1h resolution. **1h pivot KILLED:** 1h breaks the cost wall (|μ|≈43bps ≫ 26bps) but there is NO directional OOS skill, val→test anti-correlation confirmed also at 1h, gate 4/4 failed at both 13 and 23 bps → the "same method, other timeframe" line is closed.

---

## 2026-06-05 — BLOCKER #1 (live parity) risolto · BLOCKER #1 (live parity) resolved

🇮🇹 Path live = `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canoniche) → `LiveEngine._deterministic_predict` → `denormalize_predictions` → `SignalGenerator`. Parity feature E segnale bit-perfect (`tests/test_live_training_parity.py`; replay `scripts/99_replay_live_vs_training.py`: Δfeature=0, Δμ=Δσ=0). Residuo operativo: smoke WS + paper-trading. (Backtest direzionale negativo OOS — vedi sopra.)

**EN** Live path = `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canonical) → `LiveEngine._deterministic_predict` → `denormalize_predictions` → `SignalGenerator`. Feature AND signal parity bit-perfect (`tests/test_live_training_parity.py`; replay `scripts/99_replay_live_vs_training.py`: Δfeature=0, Δμ=Δσ=0). Operational residue: WS smoke + paper-trading. (Directional backtest negative OOS — see above.)

---

> 🇮🇹 **Blocco storico — linea direzionale 1m (Iterazioni 1-10).** Conservato come record: l'alpha direzionale 1m non sopravvive OOS (vedi voci 2026-06 sopra), ma le iterazioni di pipeline/fix restano il fondamento del motore condiviso.
> **EN** **Historical block — directional 1m line (Iterations 1-10).** Kept as record: the 1m directional alpha does not survive OOS (see the 2026-06 entries above), but the pipeline/fix iterations remain the foundation of the shared engine.

## Iterazione 10 — Dashboard: fix definitivo rendering — asse category (causa: asse lineare corrotto al re-render dopo display:none) (2026-06-24)

### Asse X `type:'category'` su `plot-oi` e `plot-payoff` — `scripts/06_dashboard.py`

🇮🇹 **Il bug**: "OI by strike" (`plot-oi`) e il "profilo di rischio/payoff" dei
trade (`plot-payoff`) sparivano o si schiacciavano in una banda sottile uscendo
e rientrando da una tab del browser (risk↔trades).

🇮🇹 **Root cause (diagnosi browser, 2026-06-24)**: ad ogni re-render dopo
`display:none→block` (qualunque rientro in tab) Plotly **corrompe la mappatura in
pixel di un asse X numerico LINEARE** — le tracce finiscono fuori vista
(x≈−1244px) oppure l'intera banda si comprime a ~19px, mentre le shapes paper-ref
restano corrette. Il **primo** render è sempre buono, ogni render **successivo**
è rotto; colpito **solo l'asse X** (Y numerico ok). `_fullLayout`
(range/offset/length/margin) è identico tra primo render buono e re-render rotto →
corruzione di rendering SVG, **non** dei dati.

🇮🇹 **Rimedi falliti (provati e scartati)**: `Plotly.react`, `newPlot`,
`purge+newPlot`, sostituzione del nodo, `Plots.resize`, `relayout`
(toggle width/range), `redraw`, evento `resize` della finestra, `autorange` (gira
solo il fuori-schermo in schiacciato), rimozione larghezza barra esplicita,
scaling x verso il basso, nascondere via `visibility`/`position` invece di
`display:none`, purge-on-leave.

🇮🇹 **Il fix (verificato su 3 restart server puliti)**: asse X di `plot-oi` e
`plot-payoff` passato a **`type:'category'`** (il posizionamento per-indice è
immune alla corruzione — coerente con `plot-greeks`, già category, che il bug non
l'ha mai avuto). Nuovo helper `catPos(value, sortedArr)` colloca le linee di
riferimento (Spot/Max-Pain su OI; Strike/Entry su payoff) a un **indice di
categoria frazionario**, restando proporzionali tra le categorie. OI: gli strike
di banda diventano categorie stringa (ordinate crescenti), tick diradati
(`dtick≈n/9`), nessuna larghezza barra esplicita. Payoff: i prezzi del linspace
diventano categorie (V-curve identica, griglia regolare), il marker di settlement
aggancia la categoria più vicina, Y resta numerico autorange.

🇮🇹 **Storia (sotto-passo precedente, stesso 2026-06-24)**: l'helper unico
`plot(id, traces, layout, cfg)` aveva già unificato i 7 render con **size-guard**
(`offsetWidth===0` → ritenta al frame successivo via `requestAnimationFrame`,
cap ~1s) + dimensioni esplicite/`autosize:false`. Era contesto **necessario ma NON
sufficiente** — non fermava la corruzione al rientro in tab. L'helper resta
(size-guard + rebuild-al-rientro); il dettaglio width-esplicito/`autosize:false` è
stato rimosso e il fix risolutivo è l'asse category.

🇮🇹 **Verifica**: `py_compile` OK; 3 restart server freschi su :8050 con hard
reload (Ctrl+Shift+R) → `plot-oi` e `plot-payoff` stabili su ogni rientro in tab,
linee di riferimento proporzionali. La verità finale resta il browser dell'utente
con hard reload.

**EN** **The bug**: "OI by strike" (`plot-oi`) and the trades "risk/payoff profile"
(`plot-payoff`) vanished or got crammed into a thin strip when leaving and
re-entering a browser tab (risk↔trades).

**EN** **Root cause (browser-diagnosed, 2026-06-24)**: on every re-render after
`display:none→block` (any tab re-entry) Plotly **corrupts the pixel mapping of a
LINEAR numeric X axis** — traces land off-view (x≈−1244px) or the whole band
compresses to ~19px, while the paper-ref shapes stay correct. The **first** render
is always fine, every **subsequent** one is broken; **only the X axis** is affected
(numeric Y renders fine). `_fullLayout` (range/offset/length/margin) is identical
between the good first render and the broken re-render → an SVG-render corruption,
**not** data.

**EN** **Failed remedies (tried and rejected)**: `Plotly.react`, `newPlot`,
`purge+newPlot`, node replacement, `Plots.resize`, `relayout` (width/range toggle),
`redraw`, window `resize` event, `autorange` (only turns off-screen into crammed),
removing explicit bar width, scaling x down, hiding via `visibility`/`position`
instead of `display:none`, purge-on-leave.

**EN** **The fix (verified across 3 fresh server restarts)**: the X axis of
`plot-oi` and `plot-payoff` switched to **`type:'category'`** (index-based
positioning is immune to the corruption — consistent with `plot-greeks`, already a
category axis, which never had the bug). A new helper `catPos(value, sortedArr)`
places the reference lines (Spot/Max-Pain on OI; Strike/Entry on payoff) at a
**fractional category index**, so they stay proportional between categories. OI:
band strikes become string categories (sorted ascending), thinned ticks
(`dtick≈n/9`), no explicit bar width. Payoff: the linspace prices become categories
(V-curve identical since the grid is regular), the settlement marker snaps to the
nearest category, Y stays numeric autorange.

**EN** **History (preceding sub-step, same 2026-06-24)**: the single
`plot(id, traces, layout, cfg)` helper had already unified the 7 renders with a
**size-guard** (`offsetWidth===0` → retry next frame via `requestAnimationFrame`,
~1s cap) + explicit dimensions/`autosize:false`. That was **necessary but NOT
sufficient** context — it did not stop the re-entry corruption. The helper remains
(size-guard + rebuild-on-re-entry); the explicit-width/`autosize:false` detail was
removed and the real fix is the category axis.

**EN** **Verification**: `py_compile` OK; 3 fresh server restarts on :8050 with a
hard reload (Ctrl+Shift+R) → `plot-oi` and `plot-payoff` stable on every tab
re-entry, reference lines proportional. The final truth remains the user's browser
with a hard reload.

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
`MODEL_IMPROVEMENTS.md` per il piano dei prossimi step.

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
