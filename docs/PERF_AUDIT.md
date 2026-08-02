# PERF_AUDIT.md — Audit diagnostico di performance

> **Natura del documento.** Ricognizione conoscitiva, non piano di ottimizzazione. Nessun file di
> `quantsys/` è stato modificato; le probe usate vivono in `scripts/archive/perf_probe/` (untracked,
> eliminabili). Data: 2026-08-02. Macchina: i7-9700K (8 core / 8 thread, no HT), 15.9 GB RAM,
> RTX 2070 SUPER 8 GB, Python 3.12.10, torch 2.5.1+cu121, pandas 3.0.2, numpy 2.4.3, Windows 11.
>
> **Convenzione.** Ogni numero è marcato **[M]** = misurato in questa sessione, **[S]** = stimato
> leggendo il codice o estrapolato da una misura parziale, **[D]** = documentato altrove nel repo e
> non ri-verificato. Dove non ho una misura e non me la sento di dedurre, scrivo *da verificare*.

---

## 0. Sintesi in dieci righe

Il tempo di calcolo del progetto è **quasi tutto nel training**, e il training **non è
compute-bound**: a batch 64 la GPU sta al 5-15% di occupazione e il costo per step è dominato dal
**lancio dei kernel** (overhead CPU/dispatch), non dalla loro esecuzione. La prova diretta è che
raddoppiare il batch da 32 a 64 costa +6% di wall-clock invece di +100% **[M]**. Questo ha una
conseguenza netta sui lever proposti: quelli che riducono il *lavoro aritmetico* (channels_last,
Numba, estensioni native, Polars) non toccano il collo di bottiglia; l'unico che aggredisce il
lancio dei kernel — `torch.compile` — dà **1.56×** sullo step, misurato, col backend `cudagraphs`
**[M]**.

Il data prep, che è il candidato naturale per Polars, costa **2.2 secondi** su 66k barre **[M]**:
non c'è niente da guadagnare, e il prototipo mostra che tre delle sette colonne portate cambiano
valore, una delle quali del **7.7%** per una differenza di definizione dello stimatore **[M]**.

Sono emersi due difetti collaterali più importanti di qualunque ottimizzazione: un **crash da
ordine di import** (§7.1) e una **degradazione silenziosa del regime detector** (§7.2).

---

## 1. Come funziona il sistema oggi

### 1.1 I quattro percorsi di esecuzione

**`01_download_data.py` — rete → CPU → disco.** In ordine: `fetch_klines` (REST Binance, ~66
chiamate da 1000 candele per coprire 2019→oggi), `fetch_funding_rate` (best-effort, non bloccante),
troncamento holdout, scrittura `raw_candles.parquet`, `FeatureBuilder.build()` (11 step + Volume
Profile + funding + 3 interazioni), fit dello scaler **sul solo train** e transform su tutto,
scrittura `features.parquet`, `create_windows` (stride_tricks + materializzazione), `temporal_split`,
scrittura `lstm_dataset.npz`, doppio salvataggio del `PipelineState` (arch-locale + canonico).
Il confine rete/CPU è netto: tutto il download precede tutto il calcolo. Non c'è GPU in questo path.

**`02_train.py` — disco → CPU → GPU.** `np.load` dell'npz (3.26 GB), calcolo dei clip bounds
adattivi p0.1/p99.9 su `X_train`, pre-clip di train/val/test, costruzione dei `TensorDataset`
(tensori CPU residenti in RAM), poi il loop epoche: `run_train` (forward AMP → loss quantile+CE →
backward → clip → step ogni 2 batch) e `run_eval` sul val. Il `DataLoader` gira con
**`num_workers=0` forzato su Windows** (riga esplicita in `02_train.py`, che sovrascrive il
`num_workers: 6` del config): collate e pin avvengono nel processo principale, sincroni rispetto
allo step. Il confine CPU/GPU è per-batch e attraversato ~810 volte per epoca.

**`03_backtest.py` — GPU in blocco, poi CPU pura.** Tutte le predizioni sono calcolate **in
anticipo** in batch da 256 (`all_mu`/`all_sigma`/`all_nu`), denormalizzate una volta sola, e il
`predict()` interno all'event loop è un lookup O(1) su array. L'anello su `range(n-1)` è quindi
**interamente CPU/Python**, senza GPU e senza I/O. Il Monte Carlo non è sul critical path.

**`04b_vol_paper.py` — rete, e basta.** `while True`: un tick, poi `time.sleep` fino a `hh:00:90`.
Il tick fa alcune chiamate REST Deribit (chain, mark, index, ticker, eventuale ordine + hedge perp)
e **un** forward del modello. Il wall-clock del processo è ~99.9% `sleep`; il tempo attivo è
dominato dalla latenza di rete verso Deribit, non dal calcolo. Sul VPS l'inferenza è su CPU
(wheel torch CPU-only).

### 1.2 Dove va il tempo — ripartizione

**Suite di test [M]:** `438 passed, 1 skipped in 33.9s` (misurato due volte: 35.6s e 34.0s).
⚠ Il README dichiara *"355 passed, 1 skipped, ~30s"*: il conteggio è **stale** di 83 test.

**`01_download_data.py`, solo calcolo locale [M]** (rete esclusa — non l'ho cronometrata perché
dipende dalla banda e dal rate-limit Binance):

| Fase | Tempo | Note |
|---|---|---|
| `import` del modulo | ~2.2 s | vedi §1.3 |
| `FeatureBuilder.build()` | **1.72 s** | di cui `_volume_profile` **1.32 s (77%)** |
| `fit_scaler_only` + `_normalize` | 0.52 s | RobustScaler sklearn |
| `create_windows` | **4.31 s** | materializza 3.30 GB |
| `np.savez` del dataset | **5.82 s** | 3.30 GB → 0.57 GB/s |
| **totale calcolo locale** | **≈ 14.6 s** | |

Dentro `build()`, tutti gli step diversi dal Volume Profile stanno **sotto i 40 ms ciascuno**
(`_structural_features` 0.04 s, `_frac_diff` 0.04 s, `_vwap` 0.03 s, `_technicals` 0.03 s,
`_volatility` 0.03 s, gli altri ≤0.02 s). Il data prep è, in pratica, il Volume Profile e nient'altro.

**`02_train.py` [M]** (da un run reale sandboxed su `QUANTSYS_MODELS_ROOT`, log con timestamp):

| Fase | Tempo | Come misurato |
|---|---|---|
| `import` + config | ~2.2 s | §1.3 |
| `np.load` npz 3.26 GB | 4.15 s | probe dedicata, 0.79 GB/s |
| **clip bounds `np.nanpercentile`** | **36 s** | timestamp di log 19:03:42 → 19:04:18 |
| epoca (train + eval) | **18 s** | 10 epoche consecutive, tutte 18 s |
| ├─ di cui train (810 step) | ~14.9 s | 810 × 18.34 ms **[M]** |
| └─ di cui eval sul val | ~3 s | differenza **[S]** |
| **5 seed × ~18 epoche** | **≈ 27 min** | coerente col README **[S]** |

I 36 secondi dei clip bounds sono **due epoche intere** pagate una volta per invocazione (non per
seed: il calcolo precede il loop d'ensemble). `np.nanpercentile` fa un sort completo per colonna su
una matrice `(6.2M, 104)`.

**`03_backtest.py` [M/S]:** inferenza batch dei 5 membri su 6.485 campioni = **2.0 s [M]**.
L'event loop direzionale costa **30–129 µs/barra [M]** a seconda della densità di trade (30 µs con
pochi trade, 129 µs con 363 trade su 1020 barre — un caso estremo): sull'intero split di test sono
**0.2–0.9 s**. `bootstrap_sharpe_ci` (5000 resample) = **37 ms [M]**; `mdd_stats` = **2 ms [M]**.

⚠ **Il backtest non è eseguibile end-to-end sullo stato su disco corrente**, e questo è corretto:
il checkpoint in `models/itransformer/` è il modello **vol** (`log_rv`), quindi la σ denormalizzata
vale 1.61–2.92 in unità di log-varianza e il guard `σ ≥ 0.05·√60 = 0.387` fail-fasta come previsto
(`RuntimeError` a `03_backtest.py:512`). Ho quindi misurato l'event loop **in isolamento**, pilotando
le classi `SignalGenerator`/`RiskManager` di produzione con μ/σ sintetici, invece di aggirare il guard.

**`04b_vol_paper.py` [M/S]:** forward singolo (batch 1, `no_grad`, AMP off) = **2.74 ms** su GPU
**[M]**; batch 64 = 4.22 ms **[M]**. Sul VPS, CPU-only, sarà più lento ma resta trascurabile rispetto
a un tick orario **[S]** — non l'ho misurato sul VPS.

### 1.3 Il costo degli import

Misurato su 3 run per riga, interprete nudo ≈ 59 ms **[M]**:

| Import | Tempo |
|---|---|
| `pandas` | 472 ms |
| `pandas` + `sklearn.preprocessing` | 1 404 ms |
| `pandas` + `torch` | 2 202 ms |
| `pandas` + `quantsys.utils` | **2 222 ms** |
| `pandas` + `quantsys.features` | 1 478 ms |

`quantsys.utils` importa torch a livello di modulo, quindi **ogni script che lo tocca paga ~2.2 s
prima di fare qualunque cosa**. Il profilo py-spy di `FeatureBuilder` lo conferma dal lato opposto:
`_load_dll_libraries (torch/__init__.py:238)` raccoglie **87 campioni**, esattamente quanti la riga
più calda del Volume Profile (`__init__.py:347`, 87 campioni) — in un benchmark che non usa torch.
Sommando la macchina di import (`get_data`, `_path_stat`, `_compile_bytecode`, `realpath`) si
superano i 230 campioni, cioè più dell'intero Volume Profile.

Per `01`/`02` (decine di secondi o minuti) è rumore. Per i giudici one-shot e per la routine di
sessione, che lanciano più script brevi in sequenza, è la voce dominante.

### 1.4 Il training è launch-bound — la misura

Questo è il risultato centrale della Parte 1. Modello: `QuantiTransformer`, **675 995 parametri**,
d_model 128, 3 layer, patch_size 5 → `T_eff = 24`, F = 104 token (+1 macro). Sono kernel minuscoli
per una 2070 SUPER.

**Sweep del batch, fwd+bwd puro su batch GPU-resident (nessun DataLoader, nessun H2D) [M]:**

| batch | ms/step | sample/s | ms/step senza `.item()` |
|---|---|---|---|
| 32 | 16.69 | 1 917 | 18.19 |
| 64 (**produzione**) | 17.74 | 3 607 | 17.43 |
| 128 | 19.40 | 6 597 | 19.58 |
| 256 | 25.96 | 9 861 | 26.91 |
| 512 | 52.03 | 9 841 | 50.46 |
| 1024 | 99.84 | 10 256 | 96.92 |

Da 32 a 128 il lavoro aritmetico quadruplica e il wall-clock cresce del **16%**: sotto batch ~256 il
tempo è quasi indipendente dal lavoro, cioè **dominato dal lancio**. Sopra 512 la curva diventa
lineare — lì, e solo lì, il training è compute-bound. `nvidia-smi dmon` concorda: **SM 5-15%** a
batch piccolo, **96-98%** a batch ≥512 **[M]**.

Quantificato in modo difendibile: alla saturazione la GPU macina 10 256 sample/s, quindi 64 campioni
"valgono" 6.2 ms di calcolo reale; ne spendiamo 17.7. **Circa 11.5 ms per step (65%) sono overhead
che non scala col lavoro.**

Due ipotesi che ho testato e che **non** reggono:
- *Il `loss.item()` per step stalla la pipeline.* Costa +0.3 ms a batch 64 **[M]** — irrilevante,
  proprio perché essendo già launch-bound la GPU è comunque in attesa della CPU.
- *Il DataLoader è il collo.* Da solo costa **1.74 ms/batch** a bs=64 **[M]** (collate+pin+H2D);
  nello step completo il delta rispetto al fwd/bwd puro è ~0.6 ms. È il ~3-9% dello step, non il
  collo — anche se `num_workers=0` significa che quel costo è interamente sul critical path.

Il profilo `torch.profiler` su 20 step l'ho eseguito ma **non lo riporto come ripartizione**: con
`record_shapes` attivo l'overhead di profiling ha gonfiato il totale CPU da 0.40 s a 1.80 s e
attribuito "self CUDA time" a operazioni CPU-only (`as_strided`, `select`), producendo un'occupazione
GPU del 476% — un artefatto. Le tre misure dirette sopra sono più affidabili e dicono la stessa cosa.

---

## 2. I lever, uno per uno

### (a) `torch.compile` — **SÌ, è l'unico che aggredisce il collo reale**

**Compatibilità.** Tre verifiche, tutte misurate:

1. **`spectral_norm` non c'è sul path di produzione.** In `QuantiTransformer.__init__` la
   `spectral_norm` è applicata dentro un ramo `if ... and loss_type == "t_student"`. La produzione
   gira `loss_type: quantile`, quindi il modello **non ha alcuna parametrizzazione**: verificato con
   `torch.nn.utils.parametrize.is_parametrized` su ogni sotto-modulo → *nessuno* **[M]**. La
   preoccupazione `torch.compile` ↔ `spectral_norm` è **fuori perimetro** per l'arch di produzione.
   Resterebbe rilevante solo per il ramo `t_student` e per nhits/tcnmamba, dove la SN è applicata
   incondizionatamente.
2. **Lo scan di Mamba non è in gioco**: `tcnmamba` non è sulla linea vol (i checkpoint sono stati
   eliminati col cleanup 06-12) e non va riaddestrato per questo. Non l'ho testato — *da verificare*
   se e quando si riapre un run eterogeneo. Nota: lo scan è già vettorizzato (cumprod/cumsum, non un
   loop Python) e forza float32 internamente, quindi è un candidato plausibile ma non verificato.
3. **Dynamo traccia il modello per intero**: `graph_count=1`, **`graph_break_count=0`**,
   `op_count=88` **[M]**. Nessun graph break da risolvere.

**Il blocco vero è la toolchain, non il codice.** Il backend di default (`inductor`) fallisce:
`RuntimeError: Cannot find a working triton installation`. **Triton non è installabile da PyPI su
Windows** (`pip download triton` → `No matching distribution found`) **[M]**. Esiste il pacchetto
di terze parti `triton-windows`, che non ho installato né valutato.

**Il backend `cudagraphs` non richiede Triton e funziona:**

| | ms/step | speedup |
|---|---|---|
| eager (produzione) | 15.30 | — |
| `torch.compile(backend="cudagraphs")` | **9.79** | **1.56×** **[M]** |

Costo di compilazione: il warmup di 12 step passa da 0.52 s a 3.32 s, cioè **~2.8 s una tantum** per
processo **[M]** — trascurabile su un training da minuti, non trascurabile su uno script one-shot.

**Guadagno plausibile end-to-end [S].** Il lever tocca solo la parte train dell'epoca (~14.9 s su
18 s). A 1.56× l'epoca scenderebbe a ~13.5 s, cioè il training 5-seed da ~27 a **~20 min**: **−26%**.
Non tocca i 36 s dei clip bounds né i 4.15 s di `np.load`. È un guadagno reale e misurato, ma di
ordine "minuti", non "ore".

**Perché funziona** è esattamente ciò che dice la §1.4: CUDA Graphs cattura la sequenza di lanci e la
rieseguq come singola submission, che è la cura specifica per un carico launch-bound. Coerentemente,
il guadagno atteso a batch grande sarebbe molto minore — *non l'ho misurato a batch 512*.

**Rischio parity: alto, vedi §4.**

### (b) `channels_last` — **NO, non si applica**

`channels_last` è un memory format definito per tensori **4D NCHW** (e 5D NDHWC) e agisce
selezionando kernel cuDNN diversi per **convoluzioni e normalizzazioni spaziali**. In questa
codebase:

- `grep` su `quantsys/model/` non trova **nessun** `Conv2d`, `BatchNorm2d`, `MaxPool2d` **[M]**.
- La TCN usa `nn.Conv1d` (tensori 3D NCL) — `channels_last` non è definito per il 3D; l'analogo
  sarebbe `channels_last_1d`, che non è un formato pubblico stabile in PyTorch.
- Gli unici tensori 4D del progetto sono i `q/k/v` dell'attention, `(B, n_heads, N, d_head)`, prodotti
  da `view(...).transpose(1,2)`. Non è un layout spaziale NCHW: è un batch di matrici per
  `scaled_dot_product_attention`, che non consulta il memory format e vuole comunque il suo layout.
  Marcarli `channels_last` non cambierebbe kernel; al più aggiungerebbe una copia di rilayout.

Punto chiuso. Non c'è un tensore su cui il formato cambi qualcosa.

### (c) AMP — **già mappato correttamente, niente da fare**

| Dove | Stato | Ragione documentata |
|---|---|---|
| `02_train.py:268` (`run_train`) | **ON** (`use_amp = tcfg["use_amp"] and cuda`) | training |
| `02b_walkforward_validate.py:310` | ON | training |
| `02c_optuna_search.py:78` | ON | training |
| `02b_walkforward:314,341` (eval) | **OFF** esplicito | valutazione deterministica |
| `EnsembleModel.__call__` (`ensemble.py:355`) | **OFF** (`enabled=False`) | commento in loco: *"evita NaN (spectral_norm + Mamba scan)"* |
| `crps_t_student` (`model/__init__.py:73`) | **OFF** forzato | *"lgamma instabile in float16"* |
| `MambaSSM` scan (`tcn_mamba.py:~204`) | promozione a fp32 | *"cumprod/cumsum sensibili a underflow in FP16"* |

I tre siti OFF sono spenti per **stabilità numerica**, ognuno con la motivazione scritta accanto.
Non propongo di riaccenderli: non è un'omissione, è una scelta. Osservo solo che l'inferenza
dell'ensemble in fp32 è, alla luce della §1.4, comunque **launch-bound** (2.74 ms per un forward
batch-1 su un modello da 676k parametri), quindi l'AMP non le farebbe guadagnare granché nemmeno se
fosse sicura.

### (d) Polars al posto di pandas nel `FeatureBuilder` — **NO, doppiamente**

**Primo motivo: non c'è tempo da recuperare.** Il data prep completo è **2.24 s** su 66k barre
**[M]**, e il **59%** è `_volume_profile`, che è un **loop Python** su indici campionati con
`np.bincount`/`argsort`/`searchsorted` dentro — cioè esattamente ciò che Polars *non* esprime.
Le operazioni che Polars accelererebbe (rolling, groupby-cumsum) sommano ~0.4 s. Anche a 3× uniforme
si recuperano **~0.27 s** su un percorso che, con rete e I/O, dura decine di secondi.

**Secondo motivo: cambia i numeri, e non solo all'ultimo bit.** Ho portato 7 colonne — **tutte e
sette sono nella lista canonica delle 104**, verificato contro `feature_names` dell'npz **[M]**:

| Gruppo | speedup | scarto vs pandas |
|---|---|---|
| `vol_mean_20`, `realized_var_20` | **3.48×** | **bit-identici** (100% dei valori) |
| `vol_std_20` | (stesso gruppo) | rel_max **2.9e-12**, ULP_max **24 394**, bit-uguali **0.2%** |
| `vwap_20` (rolling sum) | **1.87×** | **bit-identico** |
| `vwap` (groupby cumsum) | (stesso gruppo) | rel_max 7.7e-16, ULP_max **6**, bit-uguali 50.1% |
| `ret_skew_20` | **0.43×** (più **lento**) | **rel_max 7.7e-2, |Δ|max 3.4e-1** |

Le prime due righe sono la storia che ci si aspetta: somme e medie riassociate danno risultati
identici o entro pochi ULP; la deviazione standard rolling usa un algoritmo di accumulo diverso
(verosimilmente Welford contro two-pass) e diverge a 1e-12 relativo — abbastanza da rompere una
parity bit-perfect, non abbastanza da cambiare una decisione.

**`ret_skew_20` è il caso serio.** Non è arrotondamento: `rolling_skew` di Polars usa lo stimatore
**biased** (denominatore *n*), `rolling(20).skew()` di pandas quello **unbiased** Fisher (*n−1*).
La differenza è **sistematica e del 7.7%** su una feature che il modello di produzione riceve in
input. Una migrazione fatta colonna per colonna, con i test verdi (i golden test controllano la
*lista* delle 104 feature e le shape, non i *valori* di ogni colonna), introdurrebbe una modifica
silenziosa dell'input del modello. E per giunta su un'operazione in cui Polars è **2.3× più lento**.

**Terzo elemento, emerso per caso ma pertinente.** Installare Polars in questo ambiente non è
gratis: porta `polars-runtime-32`, cioè un secondo runtime Arrow accanto a `pyarrow`. Durante
l'audit l'installazione/rimozione ha anche perturbato il set di dipendenze (`statsmodels` è stato
rimosso e ho dovuto reinstallarlo alla 0.14.6 per riportare la suite a `438 passed`). Ho disinstallato
Polars a fine audit; l'ambiente è tornato identico al baseline.

### (e) Numba — **NO, i candidati dichiarati non esistono o sono già vettorizzati**

Verificati uno per uno i tre candidati indicati:

1. **Event loop di `03_backtest.py`.** È sequenziale davvero (stato del `RiskManager` che dipende
   dalla barra precedente), ma costa **0.2–0.9 s sull'intero split di test** **[M]**. Anche
   un'accelerazione infinita risparmia meno di un secondo. In più è **nopython-incompatibile**:
   manipola `Enum` (`Side`, `CloseReason`), dataclass Python (`Position`, `Trade`, `DistributionParams`),
   liste di oggetti, `logging` — riscriverlo per Numba significherebbe riscrivere il risk layer in
   forma scalare, cioè toccare esattamente il codice che il manifesto vuole bit-invariato.
2. **Bootstrap CI 5000 iterazioni.** **Già completamente vettorizzato**: `rng.choice` genera una
   matrice `(5000, n)` e tutte le statistiche sono riduzioni NumPy lungo `axis=1`, senza alcun loop
   Python. Costa **37 ms** **[M]**. Non c'è niente da compilare.
3. **Delta-hedge di `04b_vol_paper.py`.** Non è un loop di calcolo: `maybe_hedge` è una manciata di
   aritmetica scalare per tick, e il tick è **orario**. Il tempo è nelle chiamate REST a Deribit.
   Numba qui non ha oggetto.

Il solo loop davvero caldo del progetto è quello del **Volume Profile** (1.32 s, e la riga più calda
del profilo py-spy). È numerico, nopython-compatibile in linea di principio — ma vale 1.32 secondi
una volta per rigenerazione del dataset. Il `mdd_stats` è un vero loop Python su 6485 punti: **2 ms**.

### (f) Estensione nativa (Rust/PyO3 o C++/pybind11) — **NO, chiaramente**

La domanda è se esista **un** componente insieme abbastanza pesante e abbastanza isolato. Passandoli
in rassegna:

- *Volume Profile* — isolato sì (funzione pura su 5 array, ritorna 4 array), pesante no: **1.32 s**.
- *Event loop del backtest* — pesante no (**<1 s**), isolato no (intreccia risk layer, enum, dataclass).
- *Training* — è il grosso del tempo, ma il calcolo è già in kernel CUDA nativi: il problema è che
  ce ne sono **troppi e troppo piccoli**, e un'estensione nativa in Python non riduce il numero di
  lanci. È precisamente il caso che `torch.compile` copre e un'estensione no.
- *Regime detector* — il full rebuild walk-forward è **[D]** dichiarato ~3 h con `hmm_retrain_days: 90`
  (~9 h a cadenza mensile) ed è di gran lunga il calcolo più lungo del progetto. Non l'ho eseguito.
  Ma il costo sta nel **fit Markov-Switching di statsmodels** (EM + ottimizzazione), non in codice
  Python del repo: sostituirlo significherebbe reimplementare filtro di Hamilton **e** stima ML in
  Rust, cioè riscrivere la parte scientificamente più delicata e meglio testata (bit-parity sotto
  test). E il problema pratico è già risolto altrimenti: **B7** (`--regime-incremental`) porta il
  refresh a minuti con bit-parity garantita da test.

**Nessun componente giustifica un'estensione nativa.** Il costo — §3 — sarebbe alto e il beneficio
misurabile in secondi.

---

## 3. Stato della toolchain

Inventario **[M]** su questa macchina:

| Componente | Stato |
|---|---|
| Visual Studio / Build Tools | **assente** — nessuna dir in `Program Files*\Microsoft Visual Studio`, nessuna chiave `HKLM\SOFTWARE\Microsoft\VisualStudio\SxS\VS7`, `vswhere.exe` assente, `cl.exe` non nel PATH, winget non trova `Microsoft.VisualStudio.2022.BuildTools` |
| Windows SDK | **assente** (`Windows Kits\10\Include` non esiste) |
| Rust | **assente** — `rustc`, `cargo`, `maturin` non trovati, nessun `~/.cargo` |
| Triton (per `torch.compile`/inductor) | **assente e non installabile da PyPI su Windows** |
| py-spy | installato durante l'audit (0.4.2), **lasciato**: profiler out-of-process, nessun conflitto di runtime |
| polars | installato e poi **disinstallato** (runtime Arrow duplicato accanto a pyarrow) |

**Cosa servirebbe al VPS Linux.** Dedotto da `deploy/vps/setup_vps.sh` e dalle unit, non indovinato:
`apt-get install -y git python3-venv python3-pip ufw unattended-upgrades curl` — **non c'è
`build-essential`, non c'è `gcc`, non ci sono header Python di sviluppo**. L'installazione è
`pip install torch --index-url .../cpu` → `pip install -r requirements-vps.txt` →
`pip install -e . --no-deps`, e `pyproject.toml` non dichiara dipendenze. Oggi il VPS costruisce
zero codice nativo: prende solo wheel. Introdurre un'estensione compilata significherebbe **o**
aggiungere una toolchain C/Rust al provisioning (e allungare `setup_vps.sh`, che è dichiarato
idempotente e one-shot), **o** costruire e distribuire wheel manylinux + win_amd64 per ogni release.

**Cosa cambierebbe per chi clona.** Oggi: `pip install -e .` funziona senza alcuna toolchain di
sistema e `pytest tests/` gira in ~34 s su CPU, che è precisamente il claim di verificabilità del
README ("verificabile subito, senza dati"). Con un'estensione nativa quel claim decade: chi clona su
Windows senza Build Tools non riesce più a installare il package, e il progetto passa da
"pip install e basta" a "pip install più un compilatore". Per un repo pubblicato a corredo di un CV
questo è un costo reputazionale concreto, non solo tecnico — ed è sproporzionato rispetto ai secondi
in gioco.

---

## 4. Rischio parity

L'invariante da proteggere è `tests/test_live_training_parity.py` (Δfeature = 0, Δμ = Δσ = 0 fra
live e training) più i golden sulle 104 feature. Per ogni lever che ha senso:

**`torch.compile` — rischio ALTO, ma delimitabile.** Meccanismi concreti:

- *Kernel diversi e ordine di riduzione.* Con `inductor` i kernel sono **generati**, non quelli di
  cuDNN/cuBLAS: fusioni, tiling e ordine di accumulo cambiano, e con essi l'ultimo bit. Con
  `cudagraphs` i kernel restano quelli di eager — è la *submission* a cambiare, non la matematica —
  quindi il rischio è molto minore, ma **AOTAutograd può ripartizionare il grafo forward/backward e
  ricomputare invece di salvare attivazioni**, il che sposta l'ordine delle operazioni nel backward.
  *Da verificare*: non ho confrontato i gradienti eager vs compiled.
- *Cattura CUDA Graph e shape statiche.* Il graph è catturato su una shape fissa. L'ultimo batch
  dell'epoca è parziale (`51882 % 64 = 42`) → ricattura o fallback. Non un problema di correttezza,
  ma di determinismo del percorso.
- *RNG.* Dropout 0.3 e drop_path 0.2 sono attivi in training. PyTorch gestisce il RNG dentro i graph
  con offset philox, ma **la sequenza di numeri casuali consumata può differire** da eager: due run
  "identici" divergerebbero. *Da verificare*.

**La delimitazione che rende il rischio accettabile:** la parity bit-perfect che il progetto
protegge è **live ↔ training**, cioè riguarda il percorso di **inferenza** (`FeatureBuilder` →
`_deterministic_predict` → denormalizzazione). `torch.compile` applicato **al solo loop di training**
non tocca quel percorso: cambierebbe i **pesi** ottenuti (un modello diverso, da ri-giudicare col
gate), non l'equivalenza fra due percorsi di inferenza sugli stessi pesi. Applicarlo invece
all'inferenza — `EnsembleModel.__call__`, `04b`, i giudici — romperebbe la parity in senso proprio e
richiederebbe di ri-verificare `test_live_training_parity.py` con tolleranza, che è esattamente ciò
che quel test esiste per non fare.

⚠ Conseguenza metodologica, non tecnica: un modello addestrato con `torch.compile` **non è
confrontabile con l'incumbent** attraverso il claim pubblicato. Vale la regola già scritta nel
manifesto — un lever si giudica contro una **baseline riaddestrata sullo stesso dataset/scaler**.
`torch.compile` è un lever di *costo*, non di *qualità*: se cambia il QLIKE, il gate va rifatto.

**Polars — rischio ALTO e non delimitabile.** Rompe la parity in due modi distinti: riassociazione
floating point (`vol_std_20`, ULP fino a 24k) e **differenza di stimatore** (`ret_skew_20`, 7.7%).
Il secondo non è un problema di tolleranza: è una feature diversa. E poiché il `FeatureBuilder` è
condiviso da training e live, un port parziale creerebbe divergenza live↔training **se e solo se**
i due path venissero migrati in momenti diversi — cioè la modalità di fallimento più probabile.

**Numba, channels_last, estensione nativa — rischio non applicabile**, perché i lever non si
applicano. Per completezza: se mai si compilasse `_vp_single` con Numba, `np.bincount` con `weights`
non è supportato in nopython e andrebbe riscritto come loop di accumulo, cambiando l'ordine di somma
→ le feature `vp_*` cambierebbero all'ultimo bit. Il commento nel codice documenta che l'attuale
`bincount` fu scelto proprio perché numericamente identico al `np.add.at` precedente.

**AMP** — già off dove serve; nessun cambiamento proposto, nessun rischio nuovo.

---

## 5. Cosa NON ha senso fare, e perché

In ordine di quanto sono sicuro:

1. **`channels_last`** — non esiste un tensore 4D NCHW nel progetto. Nessuna convoluzione 2D,
   nessuna normalizzazione spaziale. Il formato non ha su cosa agire.
2. **Numba sui candidati dichiarati** — il bootstrap è già una matrice NumPy `(5000, n)` senza loop;
   il delta-hedge è aritmetica scalare a cadenza oraria; l'event loop del backtest costa meno di un
   secondo ed è pieno di Enum e dataclass, quindi nopython-incompatibile senza riscrivere il risk layer.
3. **Estensione nativa** — nessun componente supera il secondo di costo *e* è isolato. Il calcolo
   più lungo del progetto (regime walk-forward) sta dentro statsmodels, non nel repo, ed è già
   risolto da B7 in modo bit-exact. Il costo d'ingresso è invece alto e ricade su tre macchine
   (questa, il VPS senza `build-essential`, e quella di chi clona).
4. **Polars nel `FeatureBuilder`** — recupererebbe ~0.27 s su un data prep da 2.24 s, non toccando il
   77% che è un loop Python inesprimibile in Polars, e cambierebbe il valore di almeno una delle 104
   feature del **7.7%** per differenza di stimatore.
5. **Riattivare AMP dove è off** — i tre siti sono spenti per NaN documentati (spectral_norm+Mamba,
   lgamma in fp16, underflow di cumprod). Non è un'omissione da correggere.
6. **Ottimizzare il data prep in generale** — 2.24 s. Qualunque intervento qui è rumore rispetto ai
   36 s dei clip bounds o ai 27 minuti di training.

E una cosa che non ha senso fare *in questo ordine*: inseguire `torch.compile` prima di aver guardato
i **36 secondi** di `np.nanpercentile` (§6, domanda 1), che sono gratis da recuperare e valgono più
di due epoche.

---

## 6. Domande aperte

Segnalate, non implementate.

1. **I 36 s dei clip bounds.** `np.nanpercentile(X_train.reshape(-1, 104), [0.1, 99.9], axis=0)`
   ordina completamente ~6.2M valori per colonna. È un costo fisso per invocazione di `02_train`,
   pari a due epoche. Domanda: serve la precisione esatta del percentile, o basterebbe una stima su
   un sotto-campione causale? ⚠ Non è una micro-ottimizzazione neutra: i clip bounds entrano nel
   `PipelineState` e quindi nel contratto train↔inference — cambiarli **cambia i dati** e richiede
   un gate. Da trattare come lever sperimentale, non come pulizia.
2. **`torch.compile(backend="cudagraphs")` sul solo training.** 1.56× misurato, ~2.8 s di
   compilazione, zero graph break, nessuna `spectral_norm` di mezzo. Le domande aperte sono la
   riproducibilità del RNG sotto cattura del graph e la ricomputazione di AOTAutograd nel backward
   (§4). Se si vuole aprire, va aperto come esperimento pre-registrato con baseline riaddestrata.
3. **Il batch di produzione è 64 in un regime launch-bound.** A 1024 la GPU rende 2.8× di throughput.
   Ma `batch_size` non è una leva di costo: cambia il numero di step, la traiettoria SGD e
   l'interazione con `gradient_accumulation_steps: 2` — e il config commenta che lr e dropout sono
   stati tarati sul 1h con un dataset da ~1.7k campioni indipendenti effettivi. **Non toccarlo per
   ragioni di performance.** La domanda legittima è un'altra: dato che l'ensemble è di 5 seed
   indipendenti e la GPU è al 5-15%, si potrebbero addestrare **più seed in parallelo nello stesso
   processo** invece che in sequenza? Sarebbe un guadagno da launch-bound (i lanci si sovrappongono)
   senza toccare l'iperparametro di nessun membro. Da verificare contro gli 8 GB di VRAM.
4. **`create_windows` con `window_stride: 1` materializza 3.30 GB** per 66k barre, e il pipeline lo
   scrive su disco, lo rilegge, e ne fa una copia in RAM col `clamp` (picco ~6.6 GB su 15.9 GB).
   Il fattore di espansione è 120× (ogni barra compare in 120 finestre). Domanda: c'è ragione di
   materializzare, invece di generare le finestre con una `Dataset` che indicizza la matrice
   `(66k, 104)` a costo zero? Cambierebbe l'ordine di nulla — le finestre sono viste — ma toccherebbe
   il formato dell'npz, che è il contratto fra `01` e `02`/giudici.

---

## 7. Osservazioni collaterali (trovate, non corrette)

> **Aggiornamento 2026-08-02 (stessa giornata):** i due difetti gravi di questa sezione (§7.1, §7.2)
> sono stati **corretti**, con regression test; §7.3 (doc stale) riallineata. Le sottosezioni restano
> nella forma diagnostica originale — descrivono il difetto *com'era* — con una nota di chiusura in
> testa a ciascuna. Le inefficienze minori di §7.4 sono deliberatamente **non** toccate.

### 7.1 ⚠ Crash da ordine di import: `pyarrow` deve inizializzarsi prima di torch+sklearn

> ✅ **RISOLTO 2026-08-02** — `import pyarrow` ancorato in `quantsys/__init__.py` (best-effort, non
> diventa una dipendenza dichiarata). Regression test `tests/test_import_order.py`, 4 test in
> subprocesso: il crash è un'access violation, non un'eccezione Python, quindi non catturabile
> in-process con `pytest.raises` — si verifica il **codice di uscita**.

**Riproducibile [M]**, exit code 139 (access violation in `pyarrow/dataset.py` durante il caricamento del modulo):

| Ordine | Esito |
|---|---|
| `import pandas` → `import torch, sklearn.preprocessing` → `read_parquet` | **OK** |
| `import torch, sklearn.preprocessing` → `import pandas` → `read_parquet` | **SEGFAULT** |
| come sopra, ma con `import pyarrow.dataset` esplicito prima | **SEGFAULT** |
| solo `torch` → `read_parquet` | OK |
| solo `sklearn` → `read_parquet` | OK |

Servono **entrambi** torch e sklearn prima di pyarrow perché il crash avvenga (classico conflitto fra
runtime OpenMP: torch porta `libiomp5md.dll`, scikit-learn/scipy il proprio).

**Perché la produzione non lo vede:** tutti gli script numerati importano `pandas` (riga 30 in
`03_backtest.py`) **prima** di `torch` (riga 31) e prima di `quantsys.*` (riga 37). L'invariante
regge **per accidente dell'ordine di import**, non per una regola.

**Perché è un rischio:** `quantsys.utils` importa torch a livello di modulo, e la checklist "nuovo
script" prescrive `load_config` da `quantsys.utils` senza dire nulla
sull'ordine. Uno script nuovo scritto in modo naturale (prima gli import del progetto, poi pandas)
crasha con un access violation senza traceback Python. Ci sono incappato scrivendo una probe di
questo audit. Non l'ho corretto (vincolo: nessuna modifica a `quantsys/`); se lo si volesse rendere
strutturale, il posto è un `import pyarrow` eager in cima a `quantsys/utils/__init__.py`, oppure una
riga nella checklist.

### 7.2 ⚠ Il fit del regime degrada in silenzio invece di fallire

> ✅ **RISOLTO 2026-08-02** — `RuntimeError` su zero fit riusciti (non disattivabile) + abort
> configurabile su `max_fit_failure_ratio` (default 0.5) + diagnostica in `last_fit_diagnostics`;
> guard rispecchiato in `continue_walkforward`. Aggiunto anche il log mancante sul ramo
> `_fit_single → None`, che prima era completamente muto. Regression test
> `tests/test_regime_fit_guard.py` (8 test); bit-parity B7 verificata invariata.

In `quantsys/macro/regime.py:651-653`, il fit Markov-Switching per timestep è dentro
`except Exception as e: log.warning(...)`. Con `statsmodels` mancante ho osservato **un warning per
ogni t** e il walk-forward che prosegue fino in fondo: `current_params` resta `None`, `probs_all[t]`
non viene mai scritto, e si ottiene un risultato **privo di contenuto informativo senza che nessuno
fallisca**. Solo `continue_walkforward` (il path B7 incrementale) fail-fasta a valle, con un messaggio
corretto ("serve un full rebuild").

L'ho scoperto perché durante l'audit `statsmodels` è stato rimosso dall'ambiente da una delle mie
operazioni pip (poi reinstallato alla 0.14.6; la suite è tornata a `438 passed, 1 skipped`). Il fatto
che il sintomo si presenti come "6 errori in `test_regime_incremental.py`" e non come un fallimento
esplicito del rebuild è la parte che segnalo: un full rebuild lanciato in quelle condizioni avrebbe
prodotto un `regime_probs.parquet` degradato, e la degradazione sarebbe stata visibile solo leggendo
i warning. Un contatore di fit falliti con soglia di abort sarebbe coerente col resto dei guard
fail-fast del progetto — ma è una decisione di design, non una svista da correggere di mia iniziativa.

### 7.3 Doc stale: il conteggio dei test nel README

> ✅ **RISOLTO 2026-08-02** — README riallineato a **450 passed, 1 skipped, ~45 s** (438 misurati
> all'inizio dell'audit + 12 nuovi test dei fix §7.1-7.2). ⚠ La suite è passata da ~34 s a ~45 s:
> i 4 test di ordine-import girano in **subprocesso** e ognuno paga ~2.2 s di `import torch`.
> È il prezzo di testare un invariante che si manifesta solo come crash di processo.

Il README dichiarava **355 passed, 1 skipped, ~30s** in due punti (sezione "Da dove iniziare" e
"Riproducibilità"). Il valore reale a inizio audit era **438 passed, 1 skipped, ~34 s** **[M]**.
È un claim che un lettore esterno verifica in trenta secondi, quindi valeva la pena riallinearlo.

### 7.4 Inefficienze minori, tutte sotto la soglia di rilevanza

Le elenco per completezza, con la ragione per cui **non** vale la pena toccarle:

- `create_windows` valuta `np.isnan(wins).any(axis=(1,2))` sulla **vista espansa** (3.3 GB, ogni
  barra riletta 120 volte) quando la maschera NaN è calcolabile sulla matrice `(66k, 104)` prima
  dell'espansione. Costa una frazione dei 4.31 s — ma vedi §6.4, il punto vero è la materializzazione.
- `02_train.py` fa `X_tr.clamp(...)` creando una **copia completa** dei tensori train/val/test
  (~3.3 GB transitori su 15.9 GB di RAM). `clamp_` in-place eviterebbe il picco. Non è tempo, è
  memoria — e non ho osservato swap.
- `_vp_single` accumula in **dict Python** indicizzati da intero (`poc_dist_sampled[i] = ...`) per poi
  riconvertirli in array con `np.array(sorted(dict.keys()))`. Un array pre-allocato + maschera
  eviterebbe dict e sort. Vale una frazione di 1.32 s.
- Il `FeatureBuilder` emette `PerformanceWarning: DataFrame is highly fragmented` in 6 punti
  (`_funding_features`, le 3 interazioni finali). È cosmetico: la defrag avviene comunque con
  `df.copy()` prima della normalizzazione, e il commento B3 documenta che la copia intermedia fu
  rimossa apposta perché ridondante.

---

## 8. Cosa non ho fatto

- **Non ho eseguito un training completo 5 seed**: le 27 min sono estrapolate da 10 epoche reali
  misurate a 18 s l'una più i 42 s di startup, non cronometrate end-to-end.
- **Non ho eseguito il full rebuild del regime detector** (~3 h dichiarate): il costo è **[D]**, preso
  dal commento in `config/default.yaml`, non verificato.
- **Non ho testato `torch.compile` su nhits/tcnmamba** (checkpoint assenti dal 06-12) né sul ramo
  `t_student`, dove la `spectral_norm` è invece applicata: lì la domanda di compatibilità resta aperta.
- **Non ho verificato la riproducibilità del RNG né i gradienti** sotto `cudagraphs` (§4): il 1.56×
  è una misura di velocità, non un certificato di equivalenza.
- **Non ho misurato l'inferenza sul VPS** (CPU): il forward 2.74 ms è su GPU di questa macchina.
- **Non ho valutato `triton-windows`** (pacchetto di terze parti) come via per abilitare `inductor`.
- **Non ho misurato il download di rete** di `01_download_data.py`, che dipende da banda e rate-limit.
- **Non ho toccato** `tests/test_live_training_parity.py`, i golden sulle 104 feature, i guard
  fail-fast di `TEORIA.md` §12.5 (il guard σ del backtest è anzi scattato durante l'audit e l'ho
  lasciato scattare) né il contratto `PipelineState`.

**Stato del repo a fine audit (fase diagnostica):** `git status` mostrava solo
`?? scripts/archive/perf_probe/`. Nessun file di `quantsys/` modificato; suite a
`438 passed, 1 skipped` come all'inizio.

---

## 9. Cosa è stato fatto DOPO l'audit (2026-08-02, stessa giornata)

L'audit era a perimetro read-only. Su indicazione esplicita sono stati poi implementati i soli
interventi a **guadagno zero secondi e rischio numerico zero** — quelli che rimuovono modi di
sbagliare in silenzio, non quelli che rendono il codice più veloce:

| Intervento | File | Test | Impatto numerico |
|---|---|---|---|
| `import pyarrow` ancorato alla radice del package | `quantsys/__init__.py` | `tests/test_import_order.py` (4) | **nessuno** |
| Guard anti-degradazione del walk-forward regime | `quantsys/macro/regime.py` | `tests/test_regime_fit_guard.py` (8) | **nessuno** sul path di successo (bit-parity B7 verde) |
| Riallineamento conteggio test | `README.md` | — | — |

Suite: **450 passed, 1 skipped** (~45 s). Doc aggiornate: `README.md`, `TEORIA.md` §12.5 (elenco
safety net), `AVVIO.md` §1.2 (nota Windows + diagnosi rapida dell'exit 139), `CHANGELOG.md`,
`STATUS.md`.

## 10. Leva A — clip bounds sulle barre distinte: TESTATA e NON ADOTTATA (2026-08-02)

**Esito: no-go.** Registrato qui perché un esito negativo misurato vale quanto uno positivo.

La leva sembrava la migliore per rapporto guadagno/complessità (31 s → 0.16 s, **222×**). Il test di
correttezza — *quale dei due stimatori descrive la popolazione giusta* — ha prodotto tre risultati,
il primo dei quali ha invalidato la premessa dell'implementazione ingenua.

**A) `X_train` NON è contiguo.** `create_windows` scarta le finestre contenenti NaN, quindi il
tensore è fatto di **4 blocchi** separati da 3 discontinuità (a j = 2933, 17520, 35895 sul dataset
corrente). La ricostruzione ovvia — "la barra j è `X[j,0,:]`" — è **sbagliata in silenzio**: la prova
di meccanismo l'ha intercettata (ricostruzione ≠ vista espansa, |Δ| = 0.39 dove doveva essere 0).
Gestendo i blocchi: 52.358 barre distinte, Σ molteplicità = 6.225.840 = `n_tr × W`, ricostruzione
**bit-identica**. Da cui il fatto strutturale: **la vista espansa non contiene informazione in più**,
è la stessa popolazione con i bordi sotto-pesati. E i bordi sono 8, non 2 → **952 barre (1.82%)
sotto-pesate, 0.92% di deficit di peso**: l'artefatto è ~4× la stima iniziale.

**B) La differenza fra i due stimatori è sotto il rumore dello stimatore stesso.** Bootstrap sulle
barre (B=300, l'unità campionaria vera — le finestre sono 120 copie sfalsate della stessa storia):
z mediano **0.008**, p90 0.263. Solo **9 bound su 208** distano più di 1 SD bootstrap, 4 più di 2.
Per il **95.7%** i due stimatori sono statisticamente indistinguibili.

**C) Impatto a valle trascurabile:** 0.145% delle celle clippate diversamente, |Δ| mediana
**0.0008 IQR**.

**Quale è più corretto, allora?** Concettualmente quello sulle **barre distinte**: il peso per
molteplicità è un artefatto della procedura di windowing — dipende da stride, window size *e da dove
sono capitati gli scarti NaN* — e non ha alcuna relazione col processo generatore dei dati. Il clip
bound dovrebbe descrivere la distribuzione marginale della feature nel tempo. Lo stimatore attuale
è una sua approssimazione con bias di bordo.

**Perché comunque no.** ① Il premio è 31 s su 27 min = **1.9%**. ② L'implementazione corretta
richiede di rilevare la struttura a blocchi, che cambia a ogni rebuild del dataset: è esattamente la
classe di bug silenzioso che §7.1-7.2 hanno appena rimosso. ③ Tocca il `PipelineState`, quindi
richiede un gate pre-registrato il cui costo supera di molto il premio.

**La versione buona dell'idea, se mai servisse:** calcolare i bound in `01_download_data.py` da
`df_feat`, dove la matrice a livello di barra esiste già e non c'è niente da ricostruire. Pulito ed
esattamente corretto — ma cambia il contratto `01`↔`02` e cambia comunque i numeri.

⚠ **Il risultato che vale a prescindere dalla decisione:** i clip bounds sono stimati da ~52k
osservazioni effettive, **non da 6.2M**. La SD bootstrap è 0.008 (p0.1) e **0.036 (p99.9)** dove
l'IQR mediana delle feature è 1.004 — il bound superiore porta ~3.6% di IQR di rumore campionario.
Il calcolo sui 6.2M dà **precisione fittizia**: la ridondanza 120× non aggiunge informazione.
Probe: `scripts/archive/perf_probe/test_clip_bounds_correctness.py`.

⚠ Verificato per inciso e **falsificato**: `X_train` non contiene NaN (0 su 647M), ma sostituire
`np.nanpercentile` con `np.percentile` **non aiuta** — è **0.92×, più lento**. L'ipotesi "il costo è
la gestione dei NaN" è sbagliata; il costo è l'ordinamento di 647M celle ridondanti.
