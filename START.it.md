🇮🇹 Italiano · [🇬🇧 English](START.md)

# QUANTSYS — Guida operativa di avvio

> **Regola della documentazione:** ogni modifica alla doc va applicata sia alla versione inglese sia a quella italiana nello stesso commit.

**Runbook operativo** del motore di forecasting BTC/USDT (`quantsys/`). Timeframe production: **candele 1h** (`data.interval: 1h`); il motore è **interval-agnostic** (tutte le conversioni di finestra sono identità a 1m). Questo file possiede i **comandi**: la derivazione matematica sta in `THEORY.it.md`, la panoramica/motivazione in `README.it.md`, lo stato corrente e i gate aperti in `STATUS.md`.

## Indice

Guida ordinata per **fase operativa**: **0. Stato & comandi rapidi** → **1. Setup e dipendenze** (installazione, Windows/PowerShell, hardware, parametri interval) → **2. Dati** → **3. Modellazione** (train / walk-forward / Optuna / distillation / CAFN) → **4. Valutazione** (backtest / verify / vol-judge / short-vol) → **5. Deploy & inferenza** (routine di sessione, live, vol-paper, collector forward, dashboard) → **Appendice: file layout**. Tutti i comandi si lanciano dalla **root** `E:\quantsys_project`.

---

## 0. Stato & comandi rapidi

### 0.1 Comandi rapidi

```bash
python run_all.py                                    # menu interattivo · interactive menu
python run_all.py --arch itransformer                # training singola arch · single-arch training
python run_all.py --arch nhits
python run_all.py --arch tcnmamba
python run_all.py --arch lstm                        # backward compat
python run_all.py --distill                          # Knowledge Distillation multi-teacher
python run_all.py --distill --teacher itransformer   # forza teacher · force teacher
python run_all.py --only-dashboard                   # solo Options Risk Terminal (no ML)
python run_all.py --interval 1h                      # overlay risoluzione candela · candle-resolution overlay
python scripts/07_verify_teacher.py                  # confronto architetture · architecture comparison
python scripts/99_replay_live_vs_training.py         # diagnostica parity live · live parity diagnostic
```

⚠ Su disco `models/` contiene **solo `itransformer/` (production vol-1h) e `lstm/` (legacy)**: `models/nhits` e `models/tcnmamba` sono stati eliminati col cleanup 2026-06-12 → **vanno riaddestrati** prima di qualsiasi run eterogeneo (`--distill`, backtest ensemble). Le arch restano valide come `--arch`.

### 0.2 Stato del sistema

- **Production = linea vol-1h** (`features.target_type: log_rv`, iTransformer **5 membri** in `models/itransformer/`): unico segnale **PASS OOS** del progetto.
- **Braccio short-vol** in forward test su Deribit testnet via `04b_vol_paper.py` — servizio **systemd sul VPS 24/7**, non a casa (§5.3). Gate v1 (n=20) **FAIL 0/3 il 2026-07-18**; i gate successivi sono in accumulo di campione.
- **Linea direzionale** (`target_type: ret`, `03_backtest.py`/`04_live_signals.py`): **nessun alpha OOS a nessun timeframe** — codice vivo e bit-invariato, tenuto come *negative-control* scientifico. Paper-only, nessun ordine reale.
- **Invariante z-score:** ogni nuovo entry-point deve chiamare `PipelineState.denormalize_predictions(mu, sigma)` prima del trading layer; con target `log_rv` serve l'inversione completa `μ·IQR + centro`. Derivazione in `THEORY.it.md` (§ *Invariante critico*).
- ⚠ **Stato corrente, gate aperti e "riparti da qui": `STATUS.md`** (periodo corrente + gate aperti; lo storico ante 2026-07-08 è in `docs/STATUS_ARCHIVE_2026H1.md`, read-only). Questa guida **non** tiene lo storico degli esperimenti.

---

## 1. Setup e dipendenze

### 1.1 Installazione

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
python scripts/00_check_setup.py
```

`00_check_setup.py` controlla dipendenze, CUDA, Binance, FRED — risolvi gli errori prima di proseguire. Il suo § **STATO PIPELINE** è diverso: elenca artefatti **prodotti** dagli script successivi (dataset, checkpoint, report), quindi su un clone fresco sono assenti **per definizione** e compaiono come warning `△`, non come errori — non concorrono al verdetto finale e non c'è nulla da "risolvere" prima di lanciare la pipeline. `00_test_binance_testnet.py` è uno smoke opzionale della connettività al testnet Binance.

### 1.2 Note operative su Windows / PowerShell

- **stderr:** **NON** usare `2>&1 | Tee-Object` — PS 5.1 incapsula stderr come ErrorRecord (output rosso fittizio; il logging Python va su stderr). Lo script salva già `logs/quantsys_*.log`; per un file dedicato usa `*> file.log`.
- **Redirect shell-dependent:** `*> file.log` è sintassi **PowerShell**; sotto **bash** `*` viene globbato e il log NON si scrive → lì serve `> file.log 2>&1`. Verifica sempre che il log si riempia prima di considerare avviato un job lungo.
- **Exit code degli exe nativi:** in PS 5.1 un eseguibile nativo che esce con codice **≠0 NON solleva eccezione** → il `try/catch` non basta, controlla `$LASTEXITCODE` esplicitamente dopo la chiamata (pattern usato in `avvio_sessione.ps1`).
- **Encoding dei `.ps1`:** un file senza BOM viene letto come **cp1252** e qualunque carattere unicode corrompe il parsing → tieni gli script PowerShell **ASCII-only**.
- **Quoting verso gli exe nativi:** PS 5.1 strippa le doppie virgolette negli argomenti passati a un eseguibile nativo → nei blocchi `python -c` usa **solo apici singoli**.
- **UTF-8 boilerplate:** ogni nuovo script in `scripts/` deve reconfigurare UTF-8 su stdout/stderr in `main()` (il bug cp1252 è ricorso 5 volte: qualunque unicode nel banner crasha su console Windows).
- **`set` vs `$env:`:** i blocchi `set QUANTSYS_ARCH=...` di questa guida sono sintassi **cmd.exe**; in PowerShell usa `$env:QUANTSYS_ARCH="..."`.
- **Ordine di import — pyarrow prima di torch+sklearn (risolto 2026-08-02).** Caricare pyarrow **dopo** che torch **e** scikit-learn sono già nel processo produce un'**access violation** (exit 139, nessun traceback Python) al primo `pd.read_parquet`: è il conflitto fra i runtime OpenMP che i due portano. Serve la compresenza — torch da solo o sklearn da solo non basta. `quantsys/__init__.py` ancora `import pyarrow` alla radice del package, quindi qualunque `import quantsys.*` inizializza Arrow per primo e il problema non si presenta più (test: `tests/test_import_order.py`). ⚠ Se scrivi codice che importa torch **prima** di qualunque cosa di `quantsys`, l'ancora non ti copre: in quel caso importa `pandas` (o `pyarrow`) in cima, come fanno tutti gli script numerati.

⚠ **Diagnosi rapida:** un processo Python che muore con **exit 139 / access violation senza traceback** subito dopo un `read_parquet` è quasi sempre questo, non un OOM.

### 1.3 Hardware

**CPU** — `config/default.yaml` (valore corrente `1` = tutti i core; abbassa a 0.5 per lasciare la macchina usabile durante un training lungo):
```yaml
hardware:
  cpu_fraction: 1     # 0.3=30%, 0.5=50%, 1=100% dei core
```
Letto da tutti gli script all'avvio.

**GPU compute** (RTX 2070 Super, min=125 max=215W):
```powershell
nvidia-smi -pl 125    # limita · throttle
nvidia-smi -pl 215    # ripristina · restore
```

**Sequenzialità GPU (8GB):** backtest/walk-forward in parallelo OK; il training 5-seed × 3 arch va **sequenziale** (OOM). **NON** girare live/paper-trading/vol-paper + training/inferenza in parallelo (contesa CUDA, 5 modelli residenti). TCN+Mamba è il collo di bottiglia.

**Setup di riferimento (RTX 2070 Super 8GB):**

| Componente | Valore |
|---|---|
| CUDA, AMP fp16 training | sì (via `setup_device`) |
| AMP inference | **off** hardcoded in `quantsys/model/ensemble.py` (evita NaN spectral_norm + Mamba scan) |
| Batch inference backtest | 256 (`BATCH_SIZE` in `scripts/03_backtest.py`) |
| Batch training | 64 (`config/default.yaml → training.batch_size`; gli `arch/*.yaml` non lo sovrascrivono) |

**Altre configurazioni:** *solo CPU* → fallback automatico via `setup_device` (`autocast` diventa no-op silenzioso): training 20–50× più lento (sconsigliato), backtest ~5s → 30–60s (tollerabile), live ~50–100ms vs ~20ms (utilizzabile, domina la latency WS Binance). *VRAM 4GB* → in `config/default.yaml → training` (o override in `config/arch/<arch>.yaml`) `batch_size: 32` + `gradient_accumulation_steps: 2` (effective batch 64) e inference batch 256→128. *VRAM ≥16GB* → `batch_size: 128`, inference batch fino a 1024 (guadagno marginale). **Apple Silicon / AMD / Intel Arc: non testato** (codice `torch.cuda.*`; MPS richiederebbe modifiche a `setup_device` + kernel custom Mamba/SSM).

### 1.4 Parametri interval 1h & rollback 1m

Valori correnti in `config/default.yaml` (verificati sul file). La colonna **era (1m)** è **nota storica**: serve solo a spiegare la calibrazione corrente e al rollback.

| Parametro (sezione) | Valore 1h | Era (1m) | Nota |
|---|---|---|---|
| `data.interval` | `1h` | `1m` | tutte le finestre TIME-semantic derivano da qui via `interval_minutes` |
| `data.start_time` | `2019-01-01` | `2025-05-19` | storico multi-anno, ~65k barre |
| `model.window_size` | 120 | 120 | invariato in barre = **5 giorni** di contesto (a 1m erano 2h) |
| `model.window_stride` | 1 | 5 | massimizza i sample su ~65k barre |
| `features.forecast_horizon` | 30 | 30 | invariato in barre = **30 ORE** (a 1m erano 30 min) |
| `validation.embargo_steps` | 168 (1 settimana) | 1500 (~25h) | vincolo ≥ `window_size + horizon` = 150 |
| `risk.max_hold_candles` | 60 (2.5 giorni) | 240 (4h) | vincolo ≥ `forecast_horizon` = 30 |
| `backtest.min_expected_ret` | 0.0013 (13 bps) | 0.0005 | gate cost-aware, **solo linea direzionale** |
| `backtest.max_sigma` | 0.10 (≈0.015·√60) | 0.015 | soglia direzionale, da ricalibrare sui percentili post-denorm |
| `montecarlo.gjr_*` | ω 1.026e-06 · α 0.1011 · γ 0.0052 · β 0.8732 · cap 0.13 | ω 1.2e-05 · α 0.05 · γ 0.065 · β 0.875 · cap 0.01 | ri-stimati su 1h il 2026-07-15; i valori 1m sono in `config/interval/1m.yaml` |

**Guard interval (fail-fast operativo):** `RuntimeError` "interval mismatch" in `03_backtest.py` e `04_live_signals.py` — modello addestrato a 1m + config 1h = combinazione invalida, bloccata. I consumer live/replay derivano l'interval da `PipelineState.interval_minutes` (fallback 1 per i pkl legacy), **mai** dalla config. σ safety-net scalata a `0.05·√interval_minutes` (≈0.387 a 1h); annualizzazione `bars_per_year = 525600 // interval_minutes` (1h→8760). **Overlay di risoluzione:** `python run_all.py --interval 1h` (o `1m`) applica `config/interval/{interval}.yaml` sopra `default.yaml` (merge shallow per-sezione, dopo i secrets e prima dell'overlay arch) e propaga `QUANTSYS_INTERVAL` ai subprocess; le `choices` derivano dai file presenti in `config/interval/`.

**Rollback a 1m** (procedura, 3 passi — nessuna modifica al codice: tutte le conversioni sono identità a 1m):
1. restore di `data/backup_1m/*` sulla copia canonica in `data/`;
2. config a 1m: `data.interval: 1m`, `data.start_time: '2025-05-19'` (+ i valori della colonna *era (1m)* qui sopra, o l'overlay `--interval 1m`);
3. **retrain obbligatorio** — i checkpoint 1m sono stati eliminati col cleanup 2026-06-12, su disco non ne resta nessuno. Il guard interval config↔state blocca comunque ogni combinazione incoerente prima che produca numeri.

---

## 2. Dati

### 2.1 Download / update / macro

La pipeline scarica e prepara i dati prima del training. Per il primo avvio lascia che `run_all.py` esegua tutte le fasi; per i run successivi salta con `--skip-update --skip-macro` (usa i dati su disco).

| Script | Ruolo |
|---|---|
| `scripts/01_download_data.py` | download completo candele Binance + funding, rebuild dataset npz |
| `scripts/01_update_data.py` | aggiornamento incrementale delle candele a oggi, poi **rebuild** di feature + npz con **re-fit dello scaler** e riscrittura del `PipelineState` |
| `scripts/01_update_data.py --candles-only` | estende **solo** `data/raw_candles.parquet` e si ferma: nessuno scaler, nessun npz, nessuno state. Da usare quando i modelli a valle sono **congelati** e serve solo la storia OHLCV fresca |
| `scripts/01b_download_macro.py` | macro FRED/yFinance + walk-forward regime `RegimeMarkovBTC` (clock orario; **~3h su 7 anni**) |

**Modalità di `01b_download_macro.py`** (mutuamente esclusive; senza flag = pipeline completa):

| Flag | Effetto |
|---|---|
| `--regime-only` | rigenera SOLO `regime_probs.parquet` + `regime_hmm.pkl` + `regime_wf_checkpoint.pkl` (salta macro/normalizer/npz; il costo resta il walk-forward) |
| `--regime-incremental` | **(B7)** appende dal checkpoint le sole barre nuove: 0-1 fit MLE, minuti. `regime_hmm.pkl` NON viene aggiornato (solo il full rebuild rifà il fit finale full-sample) |
| `--regime-bootstrap-checkpoint` | ricostruzione una-tantum del checkpoint da pkl+parquet esistenti, con golden test integrato (replay vs parquet, fail-fast) |
| `--skip-regime` | pipeline macro completa lasciando il regime detector **INTATTO** — il full rebuild rimappa gli indici dei regimi: da evitare a esperimenti aperti (le barre nuove si appendono con `--regime-incremental`) |

**Dati prodotti:** `data/raw_candles.parquet` = candele 1h 2019→oggi (~65k barre); `data/funding_rate.parquet` = funding completo dal lancio perp 2019-09-10; `data/macro_*.parquet` = FRED/yFinance; `data/regime_probs.parquet` = probabilità regime (index orario UTC); `data/features.parquet` + `data/lstm_dataset.npz` = feature normalizzate e finestre `X/y` per il training (104 canoniche = 86 dinamiche + 18 strutturali; `X_train ≈ (51k, 120, 104)`). ⚠ `lstm_dataset.npz` è **grande (~3 GB) e rigenerabile** da `01_download_data.py`: se assente, rigeneralo prima di train/judge.

### 2.2 Collector forward (dato non rigenerabile)

Tre collector raccolgono **in avanti** storico non disponibile gratis altrove. Dal 2026-07-18 girano **solo sul VPS** (servizi systemd): a casa non va rilanciato nulla — routine di sessione in *5.3*, deploy in *5.3bis*. Il VPS always-on elimina i buchi PC-off (coverage IV misurata al 18.6% delle ore, 2026-06-12→07-14).
- **`01c_iv_poller.py`** — IV Deribit short-tenor → `data/iv/` (UNICO dato non rigenerabile).
- **`01d_orderbook_recorder.py`** — order-book L2 Binance → `data/orderbook/` (Strada B1 microstruttura).
- **`01e_trades_recorder.py`** — trade opzioni Deribit production → `data/deribit_trades/` (spread realizzati; retention API ~24h → solo forward).

---

## 3. Modellazione

### 3.1 Pipeline completa

```bash
python run_all.py                    # menu: ↑↓ naviga, SPAZIO seleziona, A toggle all, INVIO conferma
python run_all.py --arch itransformer --force-download   # modalità diretta · direct mode
```
Senza flag mostra il menu interattivo e apre la dashboard su `http://localhost:8050`. Le fasi: dati → macro → train → walk-forward → backtest → live → dashboard.

### 3.2 Training singola arch

Ogni architettura ha config in `config/arch/{arch}.yaml` e output isolati in `models/{arch}/` e `results/{arch}/`. Nessuna interferenza tra run. ⚠ I tempi sono stimati sul vecchio dataset 1m-525k; il dataset 1h (~65k, ~8× più piccolo) è proporzionalmente più veloce, **da ri-misurare**.

| `--arch` | Comando | Classe (in `quantsys/model/`) | Tempo (1m-525k) |
|---|---|---|---|
| `itransformer` | `python run_all.py --arch itransformer --skip-update --skip-macro` | `QuantiTransformer` (`__init__.py`) — attention sulle feature, baseline | ~13–40 min |
| `nhits` | `python run_all.py --arch nhits --skip-update --skip-macro` | `QuantNHiTS` (`nhits.py`) — pure-MLP gerarchico | ~6–19 min |
| `tcnmamba` | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | `QuantTCNMamba` (`tcn_mamba.py`) — conv dilatate + SSM, collo di bottiglia | ~80 min/seed |
| `lstm` | `python run_all.py --arch lstm --skip-update --skip-macro` | `QuantLSTM` (`__init__.py`) — legacy, sotto-performante | (legacy) |

`--skip-update --skip-macro`: usa i dati su disco senza ridownload (ometti al primo run). Equivalente CLI diretto: `$env:QUANTSYS_ARCH="<arch>"; python scripts/02_train.py --n-ensemble <N>`. ⚠ Su disco esistono solo `models/itransformer` e `models/lstm`: `nhits`/`tcnmamba` partono da zero (cleanup 2026-06-12).

### 3.3 Ensemble omogeneo (5× stessa arch)

```yaml
# config/default.yaml
training:
  n_ensemble: 5   # default single-arch = 5; --distill default = 1, override esplicito con --n-ensemble
```
Output: `models/{arch}/best_model_0..4.pt`. Backtest/live li caricano via `EnsembleModel.load`. Indipendente dalla distillation (le modalità non si escludono).

### 3.4 Walk-forward & Optuna

**Walk-forward** (`scripts/02b_walkforward_validate.py`, integrato in `run_all.py`): purged k-fold con embargo anti-leakage. `n_folds=6` → **5 fold effettivi** (fold 0 scartato strutturalmente). ⚠ Le metriche walk-forward in-sample (Spearman/WHR) **anti-correlano** col backtest direzionale: non usarle per ottimizzare.

```bash
set QUANTSYS_ARCH=lstm                                # cmd.exe; PowerShell: $env:QUANTSYS_ARCH="lstm"
python scripts/02c_optuna_search.py --n-trials 50 --study-name quantsys
```
**Optuna** è **hardcoded su `QuantLSTM`**. `best_params.json` salvato in `models/lstm/` NON è applicato automaticamente: copia manuale in `config/arch/lstm.yaml`. Studio persistente su SQLite (`models/lstm/optuna_quantsys.db`), ripristinabile.

### 3.5 Distillation multi-teacher

```bash
python run_all.py --distill --skip-update --skip-macro
python run_all.py --distill --teacher itransformer   # forza il primary teacher, salta lo scoring · force primary teacher, skip scoring
```

**Composizione — unico punto di modifica:** `config/default.yaml → distillation.archs`.
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
Dopo la modifica, `python run_all.py --distill` addestra i mancanti → fa scoring → distilla; backtest e live usano la nuova composizione. Esempi: `["itransformer","lstm","tcnmamba"]` (rollback legacy), `["itransformer","tcnmamba"]` (solo 2).

**Note operative:**
- `n_ensemble = 1` di default sul path `--distill` (vale per candidati **e** student); override solo con `--n-ensemble N` esplicito sulla CLI.
- Se `models/{arch}/best_model.pt` esiste, l'arch viene skippata; per ri-addestrare/ri-distillare cancella il checkpoint o passa `--force-download`.
- **Verifica esito:** `models/{arch}/config.json` → `distilled: true`, `teacher_arch: "multi-teacher"`.
- ⚠ `models/nhits` e `models/tcnmamba` non esistono più su disco (cleanup 2026-06-12): il primo `--distill` li riaddestra da zero, tempi in §3.2.
- Teoria (scoring target-aware `teacher_score_weights`, soft labels μ/ls²/lnu, loss `(1−α)·NLL + α·distill`, legge della varianza totale, pesi `DEFAULT_ARCH_WEIGHTS` e `ensemble_nll_temperature`): **`THEORY.it.md` § Knowledge Distillation**.

### 3.6 CAFN — training congiunto

```bash
python scripts/02d_cafn_joint_train.py --smoke        # valida il loop (CPU, dati sintetici) · loop smoke (CPU, synthetic)
python scripts/02d_cafn_joint_train.py --epochs 20    # reale, richiede data/lstm_dataset.npz · real run, needs the npz
```

**Probe pre-registrato, inerte.** Output **isolato** in `models/cafn/`: non tocca `models/{arch}` né la parity live (kwarg `latent=None` nei 3 forward → path bit-identico al legacy). **Gate pre-registrato (val-first):** PASS sse il CAFN-congiunto batte il baseline NO-CAFN (stessi modelli/seed/epoche) di **≥3% MSE-mu su val per ≥2/3 modelli**, altrimenti KILL. Flag utili: `--archs`, `--batch`, `--d-latent`, `--cafn-d-model`, `--cafn-layers`, `--lambda-causal`, `--lr`, `--max-steps`, `--device`, `--no-gate`. ⚠ 3 modelli + CAFN su 8 GB → rischio **OOM**: abbassa `--batch`/`--cafn-d-model` e non girarlo in parallelo a poller/vol-paper. Architettura e penalità causale: **`THEORY.it.md` § CAFN**.

### 3.7 Aggiungere una nuova arch

Procedura in 7 passi (dettaglio nella skill `/add-arch`):

1. Classe in `quantsys/model/` con `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher `load_model` in `quantsys/model/__init__.py`
3. Branch `architecture == "X"` in `scripts/02_train.py`
4. `config/arch/X.yaml`
5. `choices` dei parser `--arch` e `--teacher` in `run_all.py`
6. Whitelist in `scripts/05_analyze_signals.py` (la dashboard resta arch-independent)
7. Opzionale: `distillation.archs` in `config/default.yaml`

---

## 4. Valutazione

### 4.1 Backtest direzionale & analisi segnali

`scripts/03_backtest.py` (integrato in `run_all.py`) gira il backtest trading sul target direzionale (`ret`); `scripts/05_analyze_signals.py` analizza i segnali live. Env operative: `QUANTSYS_BACKTEST_SPLIT=val` (**val-first**, output suffissati `*_val.*` che non clobberano la production; il test split si tocca a gate val superato) e `QUANTSYS_BACKTEST_SINGLE_ARCH=1` (backtest omogeneo per-arch invece dell'eterogeneo). ⚠ Il backtest **non ha senso sui modelli vol** (`log_rv`/`log_rs_ratio`): usa i giudici dedicati (§4.3). ⚠ Gli altri env-flag di `03_backtest.py` sono un **corpus KILL inerte** (regime gating, entry rank-based, cadenza, esposizione continua, calibrazione-σ): tutti validati e **FALLITI OOS**, elenco ed esiti in `THEORY.it.md` §12.5 — non ri-testarli. Dopo ogni sweep azzera gli env sperimentali e rilancia un backtest pulito.

### 4.2 Confronto architetture & parity

```bash
python scripts/07_verify_teacher.py            # tabella comparativa archs · architecture comparison table
python scripts/99_replay_live_vs_training.py   # diagnostica parity live (BLOCKER #1) · live parity diagnostic
```
`07_verify_teacher.py`: param count, forward time, Sharpe, WR, n trade, max DD, total return per ogni arch con `best_model.pt`. In alternativa: `models/{arch}/config.json` (`best_val_loss`, scaler, n_params), `models/{arch}/history.json` (curva loss), `results/{arch}/dashboard_results.json` (export backtest; non più letto dalla dashboard).

### 4.3 Giudici famiglia vol

`features.target_type` in `config/default.yaml` seleziona la famiglia del target (default codice `ret`, bit-invariato):

| `target_type` | Target | Giudice | Esito |
|---|---|---|---|
| `ret` | log-return cumulato su h barre | `03_backtest.py` (§4.1) | nessun alpha OOS |
| `log_rv` | log realized variance Σr² su h barre | `scripts/vol/dev_vols_qlike.py` (QLIKE vs HAR-RV + naive) | **PASS a 1h** (−30% QLIKE), FAIL a 1m — 2026-06-10 |
| `log_rs_ratio` | asimmetria semivarianza log(RS⁺/RS⁻) | `scripts/vol/dev_vols_rs_judge.py` (MSE vs HAR-RS + naive + train-mean) | FAIL 2026-06-11 |

Pipeline comune (dalla root, PowerShell):

```powershell
python scripts/01_download_data.py                  # rebuild dataset npz
python scripts/vol/dev_vols_macro_append.py         # ri-appende X_macro senza rifare il walk-forward regime (~5s vs ~3h)
$env:QUANTSYS_ARCH="itransformer"; python scripts/02_train.py --n-ensemble 5
$env:QUANTSYS_VOLS_SPLIT="val"; python scripts/vol/dev_vols_qlike.py      # val-first; poi "test" UNA sola volta
```

Report in `results/vols/`. ⚠ **NO backtest trading sui modelli vol** (`03_backtest.py` non ha senso su target log-RV). Backup autosufficienti: `models/backup_1h_vols/` (PASS 1h), `models/backup_1m_vols/` (FAIL 1m, tenuto come record). Gli altri giudici e probe della linea vol sono elencati in `scripts/README.it.md`. ⚠ Due leve **inerti di default** vivono su questo path e non vanno accese fuori dal loro gate pre-registrato: `QUANTSYS_QLIKE_SMEARING=1` (correzione di smearing sul giudice, C1) e `model.attn_entropy_lambda` (penalità entropica sull'attention, A10 — overlay `config/arch/itransformer_a10_sparsity.yaml`, sonda `scripts/vol/attn_entropy_probe.py`). Razionale ed esiti: `THEORY.it.md` §12.2 e §12.4.

### 4.4 Short-vol arm & IVS relative-value (linea vol monetizzazione)

Script di ricerca GPU-free in `scripts/vol/`, **da lanciare dalla root**: famiglia `short_vol_*` (sim offline del forward test, backtest storico FHS-GJR-GARCH 2019→2026, robustness del premio VRP, decomposizione per regime/anno) e coppia `ivs_scout.py`/`ivs_rv_backtest.py` (smile Deribit + reversione dei residui). **Stato:** short-vol = edge VRP strutturale CONFERMATO sul backtest storico, ma il gate vero è il **forward test live** di `04b` (gate v1 n=20 **FAIL 0/3** il 2026-07-18; gate successivi in accumulo — soglie e contatori in `STATUS.md`); IVS relative-value = **KILL net-of-cost** (vivrebbe solo da market-maker). Ruolo e flag di ogni script: `scripts/README.it.md`.

---

## 5. Deploy & inferenza

### 5.1 Avvii successivi

```bash
python run_all.py --only-dashboard               # solo dashboard · dashboard only
python run_all.py --skip-train --skip-walkfwd    # aggiorna dati, stessi modelli · update data, same models
python run_all.py                                 # menu
python run_all.py --distill                       # full + distillation
```

**Flag utili:**

| Flag | Effetto |
|------|---------|
| `--skip-update` | usa dataset esistente, no download |
| `--skip-macro` | salta download FRED/yFinance |
| `--skip-train` | usa modello esistente, no retrain |
| `--skip-walkfwd` | salta walk-forward validation |
| `--skip-backtest` | salta backtest |
| `--skip-live` | no feed live WebSocket |
| `--skip-analyze` | salta `05_analyze_signals.py` |
| `--only-dashboard` | solo Options Risk Terminal, no ML né live |
| `--no-browser` | non aprire browser |
| `--force-download` | ri-scarica + forza retrain |
| `--max-model-age-days N` | retrain se modello > N giorni |
| `--interval {1m,1h}` | overlay risoluzione candela |
| `--n-ensemble N` | seed ensemble single-arch (default 5) |
| `--distill` | pipeline multi-teacher |
| `--teacher ARCH` | forza primary teacher |

### 5.2 Live / paper trading

Avviato da `run_all.py` (fase live, salvo `--skip-live`) o da `python scripts/04_live_signals.py`. Path di produzione: `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canoniche, scaler da `PipelineState`) →`_deterministic_predict`→`denormalize_predictions`→`SignalGenerator`. **Parity feature e segnale bit-perfect col training** (BLOCKER #1 chiuso il 2026-06-05): regressione in `tests/test_live_training_parity.py`, diagnostica end-to-end `python scripts/99_replay_live_vs_training.py` (Δfeature = Δμ = Δσ = 0). Paper-only, nessun ordine reale. Residuo operativo: smoke test WS reale. ⚠ Backtest direzionale negativo OOS → il paper-trading accumula solo trade reali. **NON** girare live + training/inferenza GPU in parallelo (contesa CUDA).

### 5.3 Routine di sessione (lato casa)

Dal 2026-07-18 il PC di casa è **passivo**: nessun processo residente locale — i collector `01c`/`01d`/`01e` e il vol-paper `04b` girano tutti come **servizi systemd sul VPS** (§5.3bis). **Unico comando a ogni sessione**, dalla root di progetto:

```powershell
.\avvio_sessione.ps1          # [-Days 7] [-SkipPull] [-SkipMonitor] [-RefreshCandles]
```

| # | Blocco | Contenuto |
|---|---|---|
| ① | **Pull + merge VPS** | `scripts/vps/pull_vps_data.ps1`: **archiviazione a vintage datati** di `macro_features.parquet` → VPS (vedi sotto), scp collector → `data/vps_staging/`, merge dedup nella copia canonica, **heartbeat staleness dei 4 collector** (IV / L2 / trades / 04b), staging auto-pulito |
| ② | **Freshness regime B7** | ≥168 barre orarie oltre il checkpoint walk-forward → `01b_download_macro.py --regime-incremental` in background (anti-dup se un `01b` è già vivo); con candele congelate stampa "fresco" ed è un no-op |
| ②bis | **Estensione candele** — *solo con `-RefreshCandles`, spento di default* | `01_update_data.py --candles-only`: estende `data/raw_candles.parquet` e si ferma (nessuno scaler, nessun npz, nessuno state). **Non è un default** e il perché sta nel paragrafo dedicato sotto: automatizzarlo toglierebbe la possibilità di **congelare i dati**. Collocato **dopo** ② di proposito, così l'eventuale refresh regime parte al prossimo avvio |
| ③ | **Monitoraggio linea vol** (CPU-only, **a scrittura zero**) | `scripts/vol/derive_mfiv.py` incrementale (**dopo** il merge per costruzione: legge la chain appena scaricata; stampa il wedge MFIV−ATM, che è una **diagnostica permanente**, non un contatore di gate) + contatore delle leg opzioni eseguite (`executed` di `trades.jsonl`, soglia storica n≥30: il gate è chiuso, ma resta **l'unico numero che dice se `04b` sta eseguendo**) + **copertura del file barre 1m** (vedi sotto) + **campione E1 stadio 2** (`edge_information_judge.py --stage 2 --count-only` verso n≥40, preceduto da fin dove arriva la serie dei close). ⚠ Il contatore **hedged** è stato **ritirato il 2026-08-13**, alla rimozione di `--hedge` dall'unit: il suo gate era chiuso dall'11/08 e il suo ultimo consumatore era la verifica che la banda di wind-down non aprisse hedge nuovi. Non va ripristinato "per informazione" — un numero senza consumatore invita a confrontarlo con una soglia che non è più la sua, ed è il contatore che ha letto l'unità sbagliata tre volte (eventi ≠ posizioni ≠ settlement) |

⚠ **Disciplina one-shot — invariante permanente della routine.** Il blocco ③ non esegue **mai** un giudice pre-registrato: i contatori che restano calcolano **solo conteggi** (nessun PnL, nessun edge, nessuna correlazione), e ogni giudice ha comunque il proprio guard `n < n_min → NO_RUN` senza scrivere report → automatizzare un conteggio **non può produrre peeking**. ⚠ Un guard `NO_RUN` protegge però solo **sotto** soglia, e non protegge affatto gli effetti collaterali che lo precedono (fetch di rete, riscritture di cache): per questo la routine usa le varianti `--count-only` e il **run one-shot resta MANUALE**. ⚠ Il contatore `--count-only` del comparatore MFIV è stato **ritirato il 2026-08-18**, alla chiusura del suo gate (FAIL a n=41): `derive_mfiv.py` resta perché mantiene una colonna diagnostica, non perché alimenti un gate. Fail-soft: ogni passo che fallisce logga un warning senza fermare la routine (`-SkipPull` salta ①, `-SkipMonitor` salta ③; `-Days N` = finestra del pull; `-RefreshCandles` **aggiunge** il passo ②bis, vedi sotto).

⚠ **Copertura del file barre 1m (dal 2026-08-06) — continuità del recorder e disponibilità del target sono due cose diverse.** Il check L2 misura la continuità del **recorder**; questo misura se esiste il **target** con cui quelle ore verrebbero giudicate. Il giudice B1 costruisce la RV da rendimenti a **1 minuto** (`data/raw_candles_1m_l2.parquet`, ~180 quadrati per osservazione invece di 3), quindi le ore di L2 oltre la fine di quel file sono **raccolte ma prive di target**: il campione utile è il **minimo fra i due**, e fino a oggi nessuno dei due contatori lo diceva. ⚠ **Il tetto vale per le analisi a target 1m** (B1 a h=3, proxy del pin-close): il contatore `n_eff` a h=30 stampato dal check L2 usa le barre **orarie** — il target di produzione è 30 barre da rendimenti orari — e **non è toccato** da questo file. ⚠ **Il file ha tre consumatori e zero produttori — e zero produttori per decisione (dal 2026-08-10)**: `l2_incremental_judge.py`, `pin_close_feasibility.py` e `edge_information_judge.py` (come sorgente di coda) lo leggono, ma **nessuno script del repo lo genera o lo estende**. Non è lavoro arretrato: le kline a 1 minuto sono **storiche e ri-scaricabili indefinitamente**, quindi — al contrario del recorder L2, che è forward-only e irrecuperabile se si ferma — **non si perde nulla ad aspettare**, e il file scaricato oggi è identico a quello scaricato fra sei mesi. Il "ritardo" è un download non ancora fatto, non un debito che matura. Per questo il monitor **misura soltanto** e non stampa un rimedio: quando un'analisi a target 1m verrà davvero riaperta, il download si decide in quel momento e sulla finestra che serve, invece di tenere aggiornato di continuo un file che **nessun path operativo legge** (verificato il 2026-08-10: nessun riferimento sul VPS, e il contributo effettivo del file a `hourly_close()` misurato = **0 righe**, perché l'1m è *indietro* rispetto alle orarie). Stato al 2026-08-10: 87.217 barre, 2026-06-01 → 2026-07-31, **10 giorni di L2 senza target 1m**.

⚠ **`-RefreshCandles` (dal 2026-08-06) — estendere le barre è un atto, non un automatismo.** Il flag lancia `01_update_data.py --candles-only` (blocco ②bis, dopo il check B7 e prima del monitoraggio) ed è **spento di default**, modellato su `-PromoteMacro` per la stessa ragione per cui quello esiste. La meccanica sarebbe sicura anche automatizzata — **no-op** se non c'è nulla di nuovo (esce *prima* di scrivere, mtime invariato), dedup che tiene la riga **esistente** quindi la storia non si riscrive mai, mai la barra in formazione (`floor(now) − 1 barra`), scrittura atomica, e il file **non viene mai spedito al VPS** (l'unico trasferimento casa→VPS è la macro) — ma automatizzarla toglierebbe la possibilità di **congelare i dati** per un esperimento pre-registrato: l'invariante *«candele/npz/regime_probs non si toccano fino a chiusura gate»* passerebbe da «non fare nulla» a «ricordarsi `-SkipMonitor`», che è la forma esatta della promozione macro avvenuta **per automazione** il 31/07. E poiché lo split è una **frazione del conteggio righe**, ogni barra appesa sposta i confini train/val/test al prossimo rebuild dell'npz: con l'estensione automatica il vintage del dataset diventerebbe funzione di *quante sessioni hai aperto*, cioè non più dichiarabile. **Quando usarlo:** prima di annotare il contatore E1 se il blocco ③ stampa il warning di ritardo, e prima del run one-shot di **E1 stadio 2** (prerequisito ⑤ della sua pre-reg). Fra i gate aperti **solo E1 legge le barre** — il giudice hedged legge i ledger e il funding perp, il comparatore MFIV il chain e `forecasts.parquet`. **Quando NON usarlo:** dentro un esperimento che dichiara i dati congelati, e quando si riproduce un report storico su una finestra che assume un conteggio di barre preciso. ⚠ **Effetto a distanza di una sessione, voluto:** estendere le barre fa avanzare il contatore di staleness B7, quindi il refresh incrementale del regime parte semmai al **prossimo** avvio — le due scritture restano separate e ognuna visibile per conto suo invece di concatenarsi in silenzio.

⚠ **Il guard `NO_RUN` protegge solo SOTTO soglia — perché E1 è entrato nella routine con `--count-only` e non con `--stage 2` (2026-08-06).** `edge_information_judge.py --stage 2` sotto n=40 stampa la conta e si ferma, ma **a n≥40 lo stesso comando calcola le tre condizioni, stampa il verdetto e scrive il report**: automatizzarlo nudo avrebbe fatto scattare il run confermativo **per automazione invece che per decisione**, il giorno in cui la soglia cade e senza che nessuno lo scegliesse. `--count-only` si ferma alla conta a **qualunque** n, e un test lo verifica su un campione costruito **sopra** la soglia (`tests/test_edge_information_judge.py`) — sotto soglia il test non proverebbe nulla, perché passerebbe anche il comando nudo. ⚠ **Il conteggio dipende dalla serie dei close**: un'expiry è osservabile solo se la sua RV è calcolabile, quindi il blocco stampa **prima** fin dove arriva la serie e avvisa oltre 6h di ritardo. Il rimedio (`01_update_data.py --candles-only`) **non è automatizzato di proposito**: è una scrittura su un file di dati, e il blocco ③ resta a scrittura zero.

⚠ **Vintage macro datati (dal 2026-07-31) — il push non è più una decisione, promuovere lo è.** `04b` legge `data/macro_features.parquet` al bootstrap notturno e lo **congela** per tutta la giornata: sovrascriverlo dentro un campione forward pre-registrato ne cambia l'input in silenzio (successo il 2026-07-31, breakpoint datato in `STATUS.md`). Il pull quindi: (a) **archivia sempre** il file sotto `data/macro/macro_features_<YYYYMMDD>.parquet` sul VPS, dove `<YYYYMMDD>` è l'**ultima data dell'indice** (`scripts/vps/macro_vintage.py`) — append-only, 716 KB a copia, così ogni decisione forward resta riconducibile al suo vintage; (b) tratta il canonico come un **symlink** all'archivio, quindi il vintage live si legge con `readlink` e si vede in `ls -l`; (c) **non lo ripunta mai** senza `-PromoteMacro`, e a vintage divergente emette un warning lasciando il live su quello che già usava. Il nuovo vintage entra in vigore al **bootstrap 04b successivo (00:30 UTC)**: se un campione forward è aperto, la promozione va **datata in `STATUS.md`**.

```powershell
.\scripts\vps\pull_vps_data.ps1 -PromoteMacro   # atto deliberato: ripunta il canonico VPS al vintage locale
```

⚠ **Strumento vs stato — il `MacroNormalizer` pinnato (implementato 2026-07-31, INERTE).** Il vintage datato risolve *quale* macro arriva al VPS; resta che `VolForecaster` **ri-stima** il `MacroNormalizer` whole-df a ogni bootstrap, quindi allungare il parquet sposta mediana e IQR e **lo strumento di misura cambia insieme allo stato che deve misurare** (sul breakpoint del 31/07: 2.7% della variazione totale). `scripts/vol/pin_macro_normalizer.py` congela lo strumento a un vintage **dichiarato** e lo scrive in un pickle; `04b` e il replay lo accettano con `--macro-norm <path>`. **Senza il flag il comportamento è bit-identico** (verificato end-to-end sul parquet di produzione: 0 differenze su 90 colonne). ⚠ Il vintage sotto cui `models/itransformer` fu addestrato **non è ricostruibile**: il pin non lo recupera, ne **fissa** uno esplicito. Attivarlo oggi sarebbe un **no-op di contenuto** (stesso vintage → delta 0): serve a impedire la deriva *futura*, non a correggere il passato. L'artefatto è gitignored ma **riproducibile** da un vintage archiviato.

⚠ **EMERGENZA** — solo su `WARN IV poller` nell'heartbeat (collector VPS giù), lanciare a mano finché il VPS non torna: `.\.venv\Scripts\python.exe scripts\01c_iv_poller.py`. **`04b` NON va MAI lanciato a casa:** due `--execute` gestirebbero la stessa posizione testnet (doppi ordini). Stop di un processo d'emergenza (match sulla command line → cattura stub+worker, non tocca altri `python.exe`):

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match '01c_iv_poller|01d_orderbook_recorder' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

**Salute:** conta i processi **LOGICI**, non OS — ogni `.venv\python.exe` è stub+worker = 2 processi OS (confronta i `ParentProcessId`). La crescita dei dati arriva **dal pull**, non da processi locali: `data/iv/atm_30h.parquet` e `data/iv/mfiv_30h.parquet` ~288 righe/giorno (cadenza VPS 5 min), `results/vol_paper/forecasts.parquet` ~24 righe/giorno. Log vivi in `logs/quantsys_*.log` (più recenti per mtime).

#### 5.3bis Collector 24/7 su VPS

Quattro servizi **systemd** su un VPS EU always-on (deployato 2026-07-14): `quantsys-iv` (`01c`), `quantsys-ob` (`01d`), `quantsys-trades` (`01e`), `quantsys-volpaper` (`04b`). **Host/IP sono privati: SOLO in `config/secrets.yaml` → blocco `vps:`**, mai nel repo o nella doc. Deploy completo: `deploy/vps/README.it.md` (geo-test 451 Binance → deploy key → `setup_vps.sh` one-shot → verify). Comandi lato casa, dalla root di progetto:

```powershell
.\avvio_sessione.ps1              # tutto-in-uno di sessione: pull+merge + check B7 + monitoraggio vol (§5.3)
.\scripts\vps\pull_vps_data.ps1   # solo sync: host da secrets.yaml → scp → data/vps_staging/ + merge + heartbeat
.\scripts\vps\check_vps.ps1       # health-check on-demand via ssh: servizi / freschezza / disco / geo (-UpdateRepo = git pull remoto)
```

Il merge (`scripts/vps/merge_vps_data.py`) deduplica i tick doppi e avvisa se l'ultimo tick VPS è stale (default 3h → collector remoto giù). La copia canonica dei dati resta a casa (`data/iv/`, `data/orderbook/`, `data/deribit_trades/`); il VPS garantisce continuità H24 dell'asset IV. ⚠ Il PC di casa non riavvia collector locali (§5.3).

#### Poller IV Deribit

`python scripts/01c_iv_poller.py` — loop, cadenza default 10 min (sul VPS gira a **5 min con `--greeks`**). 2 richieste pubbliche Deribit per tick, **nessun account**. Flag: `--minutes N` (cadenza), `--once` (smoke), `--backfill-dvol` (storico DVOL orario 2021→oggi), `--greeks` (+3 chiamate/tick per i greeks di venue dello straddle ATM ~tenor-30h → `atm_greeks.parquet`, selezione identica a `pick_straddle` di `04b`). Output append-only atomico in `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (snapshot raw, ~950 strumenti/tick), `atm_30h.parquet` (ATM IV delle 4 expiry vicine + IV interpolata in varianza totale a tenor costante 30h = orizzonte del forecast vol), `dvol.parquet` (controllo 30d). Scopo: storico IV short-tenor — non disponibile gratis altrove — per il gate **NN-RV vs IV implicita**.

#### Recorder order-book L2 Binance (B1)

`python scripts/01d_orderbook_recorder.py` — loop, cadenza default 5s. Flag: `--seconds N`, `--once`, `--symbol` (default `BTCUSDT`), `--levels` (profondità REST, default 1000). 1 richiesta pubblica `/api/v3/depth` per tick (no auth, weight 50/call → a 5s = 600/min ≪ 1200). Strada **B1**: raccolta FORWARD della microstruttura come fonte NUOVA per un edge direzionale a 1m (le 104 feature OHLCV sono sature). Output append-only atomico `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/giorno, dedup su `timestamp`): mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, depth cumulata 5/10/25/50 bps, total qty, **OFI best-level** (Cont-Kukanov-Stoikov) + **top-25 livelli raw/lato** come list-column. ⚠ `ofi_best` è NaN al 1° tick di ogni processo e in `--once`.

#### Forward test vol-paper (NN-RV vs IV, testnet Deribit)

⚠ **Gira come servizio sul VPS, NON a casa** (§5.3): due `--execute` gestirebbero la stessa posizione testnet. Invocazione di produzione (unit `deploy/vps/quantsys-volpaper.service`):

```bash
python scripts/04b_vol_paper.py --execute
```

⚠ **Nessuna seconda istanza di `04b` sulla directory di produzione — nemmeno simulata, nemmeno `--once`.** I path di stato (`results/vol_paper/`) sono fissi e relativi alla directory di lancio: una seconda istanza sul VPS riscrive `forecasts.parquet` e `position.json` sotto il servizio live, e con `--adaptive` un tentativo bloccato lascia `adaptive_entry_journal.json`, la cui sola presenza ferma **anche** il servizio v1 finché non viene rimosso a mano. Il blocco scatta anche senza `--execute` (contratto unico per le due modalità, 2026-09-15): in simulazione un errore REST sul mark è `ambiguous` come con ordini reali. Una prova isolata sul VPS richiede prima un flag esplicito di directory di output, oggi assente.

⛔ **`--hedge` è FALLITO — gate v2 chiuso il 2026-08-11 (FAIL 2/3); i flag sono stati rimossi dall'unit il 2026-08-13 e l'invocazione qui sopra è quella realmente in vigore.** La varianza per-trade scendeva del 55.3% (① superata con margine) ma il drag valeva −0.647·SE contro un budget di −0.25·SE, per **76% fee perp e 0.9% funding**. Dettaglio e decomposizione: `THEORY.it.md` §12.2. ⚠ **Vincolo di ORDINE nel disattivarlo, valido in entrambe le direzioni:** `maybe_hedge` e `reconcile_hedge_state` girano **solo** dentro il ramo `--hedge`, quindi togliere il flag mentre `results/vol_paper/hedge_state.json` esiste lascia la leg perp **nuda, non gestita e mai flattenata** sul testnet — vale per chiunque riaccenda il flag e poi voglia rispegnerlo. ⚠ **Non basta "disattivare dopo un settlement": la finestra pulita è larga zero.** Il settlement e l'apertura della struttura successiva avvengono nello **stesso tick**, quindi `position.json` non è mai vuoto e un nuovo hedge può partire immediatamente. Procedura corretta, in **due passi**: ① `--hedge-band 999`, che disabilita apertura e ribilanciamento ma lascia attivo il flatten (quel ramo gira **prima** del check di banda); ② dopo il flatten successivo, rimozione dei tre flag. Nel ledger il flatten può portare `reason` ∈ `{settled, expired, structure_changed}` — **tutti e tre validi**, sono i tre arm della stessa guard, e con la riapertura nello stesso tick esce `structure_changed`. La leg opzioni è identica con o senza hedge (path v1 bit-identico), quindi la disattivazione non perturba i campioni forward aperti.

Loop orario a hh:00+90s. Flag: `--once` (smoke), `--execute` (ordini REALI sul testnet; default = fill SIMULATI al mark price), `--arch` (dir modelli, default `itransformer`), delta-hedge `--hedge` + `--hedge-band/-conv/-fee/-band-mode/-ww-lambda`, chiusura `--pin-close-hours/-band`, sizing `--size-mode {contracts,vega}` + `--size-vega-target/-max-contracts`, regola adattiva `--adaptive` + `--adaptive-dvol-threshold/-k/-fill-timeout/-tenor-hours` (INERTE senza flag; soglia e k obbligatori), normalizer macro `--macro-norm <pin>` (**inerte**: senza il flag lo strumento è ri-stimato whole-df a ogni bootstrap, comportamento storico). Logica: forecast NN-RV 30h (modello vol-1h PASS, inversione completa `μ·IQR+centro`, feature dal path parity-blessed) vs varianza implicita dal poller IV (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; `|edge| > 0.25` → straddle ATM daily ~30h LONG/SHORT, max 1 posizione, hold a scadenza (cash settlement). Richiede poller IV attivo e key in `config/secrets.yaml` blocco `deribit_testnet:` (l'URL **deve** essere `test.deribit.com` — assert anti-mainnet). Output in `results/vol_paper/`: `forecasts.parquet` (scritto anche a posizione flat, serve alle baseline), `trades.jsonl`, `position.json`. ⚠ NON girare training GPU in parallelo (5 modelli CUDA residenti).

**Esecutore adattivo (`--adaptive`) — durabilità, correzioni di correttezza e identità dei record (2026-09-13/15, locale, non deployato).** L'apertura della struttura (straddle daily o iron butterfly) passa da un journal durabile su disco (`adaptive_entry_journal.json`): scritto **prima** di ogni ordine, la sua sola presenza a riavvio blocca `tick()` e `main()` finché non c'è review manuale — un tentativo interrotto non riprende mai da solo. Due correzioni: **(A) fee/timing di fill verificati** — `_verify_trade_history` controlla la COPERTURA dei trade reali (id univoci, identità ordine/strumento/verso, quantità che sommano al `filled_amount` noto, timestamp finiti) prima di trattare fee o `fill_span_s` come completi; `_fill_timing` usa il primo e l'ultimo fill **globali** su tutte le gambe di entry (prima usava solo l'ultimo fill per gamba, sottostimando lo span con gambe multi-fill). Copertura malformata/incompleta → fee e timing **ignoti** (mai zero, mai il tempo di ricezione locale al posto del fill). **(B) coda diagnostica su journal sopravvissuto** — se l'entry adattiva di un tick lascia il journal aperto (`blocked_operator_review`), `log_exec_diag` e la leg hedge del **medesimo tick** vengono saltate: prima veniva comunque registrato uno snapshot "flat" mentre l'esposizione reale era ignota. **(C) identità dei record (15/09)** — l'identità dei trade (`order_id`/strumento/verso) è **richiesta** e non più opzionale: un trade che ne è privo non verifica più la copertura, e un `trade_id` non hashabile degrada a ignoto invece di sollevare dentro la sottomissione, dove l'eccezione avrebbe interrotto il journaling a ordini già partiti. Inoltre ogni gamba persiste **strumento e verso concreti** (`instrument`, `side`) nel record di esecuzione e di recovery: prima vivevano solo nel journal, che su flat verificato viene cancellato, lasciando il record col solo nome logico della gamba. Nessuna garanzia nuova di "mai nudo": resta la review manuale. Regressione: `tests/test_adaptive_structure.py`, 40/40 (10 preesistenti + 8 su A/B + 12 su identità, durabilità del journal, risposte perse, fill parziali, timeout, arresto della compensazione e blocco al riavvio + 10 dalla revisione indipendente del diff: verificatore dell'ordine con identità richiesta e `filled` parziale trattato come ambiguo, ramo nonterminal dentro l'executor, `trades`/`order` malformati degradati a ignoto).

**Baseline del gate** — `python scripts/04c_vol_paper_baselines.py` (read-only, GPU-free; `--no-fetch` = solo cache delivery, `--min-trades N` = soglia di valutabilità, default 30). Verifica il gate pre-registrato (2): il P&L NN deve battere **entrambe** le baseline always-long-vol e always-short-vol sullo **stesso** calendario di expiry (isola il timing dal variance risk premium medio). Metodo: replay del loop `04b` su `forecasts.parquet`, premio ricostruito dai chain snapshot, delivery price dall'endpoint pubblico Deribit (cache `delivery_cache.json`). I gate (1) P&L medio > 0 e (3) hit-rate > 0.5 si leggono dai trade REALI in `trades.jsonl`. Output `results/vol_paper/baseline_report.json` (+ warning "non valutabile" finché n < `--min-trades`).

### 5.4 Dashboard — Deribit Options Risk Terminal

```bash
python scripts/06_dashboard.py     # avvio diretto · direct launch
python run_all.py --only-dashboard # idem (no ML, no feed live · no ML, no live feed)
```

`scripts/06_dashboard.py` = **terminale opzioni crypto** (server HTTP single-file + SPA Plotly.js), GPU-free e **indipendente dalla pipeline ML**: legge i dati **pubblici Deribit** (REST, no-auth, nessuna chiave). Quattro tab — *Volatility Surface* (superficie IV 3D, smile, term structure ATM), *Option Chain* (chain call/put con Greche Black-Scholes forward), *Risk & Greeks* (OI per strike, max-pain, Greche aggregate pesate per OI, PCR, DVOL), *Trades* (forward test `04b`: storico settled + posizione aperta e profilo di payoff, endpoint `/api/trades` che legge `results/vol_paper/trades.jsonl` + `position.json`). Auto-refresh ~12s. Config: `config/default.yaml → dashboard` — `host`/`port` (default `127.0.0.1:8050`), `options_currency` (BTC|ETH), `auth_token` opzionale (constant-time, header `X-Auth-Token` o `?token=`), `enable_gzip`.

⚠ **Trappola `SO_REUSEADDR`:** un vecchio processo dashboard può tenere `:8050` e servire HTML **stale**. Prima di un nuovo smoke **uccidi il processo precedente** (stub+worker `.venv` = 1 processo logico): `Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '06_dashboard' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }`. ⚠ **La verità finale è il browser con HARD RELOAD (Ctrl+Shift+R)**: una pagina già aperta gira il JS vecchio. Smoke server-side: l'HTML servito deve contenere `plot(` e `/api/risk` deve rispondere HTTP 200 con la chain reale. (Fix rendering definitivo 2026-06-24: asse X di `plot-oi`/`plot-payoff` passato a `type:'category'` per immunità alla corruzione SVG di Plotly al re-render; dettaglio in `STATUS.md`.)

### 5.5 Fermare tutto

`Ctrl+C` nel terminale di `run_all.py` (o del server dashboard): ferma la pipeline e, in un run completo, anche il feed live WebSocket. ⚠ Se la dashboard era detached (`Start-Process`), `Ctrl+C` non basta: usa lo `Stop-Process` mirato (§5.4). I collector di sfondo non girano più a casa (VPS, §5.3bis): un'eventuale istanza locale d'emergenza si ferma col blocco *Stop* in §5.3.

---

## Appendice — File layout

```
quantsys_project/
├── config/
│   ├── default.yaml             # config base (distillation.archs qui · here)
│   ├── secrets.yaml             # FRED + deribit_testnet keys, gitignored
│   ├── interval/                # overlay risoluzione: 1m.yaml, 1h.yaml (--interval)
│   └── arch/                    # itransformer.yaml, nhits.yaml, tcnmamba.yaml, lstm.yaml
├── data/
│   ├── raw_candles.parquet      # OHLCV storico (1h, 2019→oggi)
│   ├── features.parquet         # feature normalizzate (rigenerabile)
│   ├── lstm_dataset.npz         # windows X/y per training (~3 GB, rigenerabile da 01)
│   ├── funding_rate.parquet     # funding futures (completo dal 2019-09-10)
│   ├── macro_*.parquet          # FRED/yFinance
│   ├── regime_probs.parquet     # probabilità regime (index orario UTC)
│   ├── iv/                      # IV Deribit: chain/ (snapshot raw), atm_30h, mfiv_30h, dvol — NON rigenerabile
│   ├── orderbook/               # L2 features forward (B1) — NON rigenerabile
│   ├── deribit_trades/          # trade opzioni production (01e) — NON rigenerabile
│   └── backup_1m/               # raw_candles + regime_probs era-1m (rollback 1m = restore + retrain, §1.4)
├── models/
│   ├── pipeline_state.pkl       # copia canonica (scritta da 01, guard anti-stale in 02)
│   ├── backup_1h_vols/          # vol-1h PASS autosufficiente (5 membri + state + raw/regime 1h)
│   ├── backup_1m_vols/          # vol-1m FAIL (record)
│   ├── itransformer/            # PRODUCTION vol-1h (5 membri) — unica arch addestrata su disco
│   └── lstm/                    # legacy (+ studio Optuna). nhits/ e tcnmamba/ NON esistono: cleanup 2026-06-12
│       ├── best_model.pt        # checkpoint (best_model_0..4.pt per ensemble multi-seed)
│       ├── config.json          # iperparametri + flag distilled/teacher_arch + best_val_*
│       ├── history.json         # curva loss
│       └── pipeline_state.pkl   # scaler + feature config + target_scale + interval
├── results/
│   ├── {arch}/                  # dashboard_results.json, live_signals.jsonl
│   ├── vols/                    # report giudici vol
│   └── vol_paper/               # forecasts.parquet, trades.jsonl, position.json, baseline_report.json, exec_diag.jsonl (A6: bid/ask+greeks diagnostici / diagnostic), hedge_state.json + hedge_ledger.jsonl (v2, SOLO con --hedge / --hedge only) · record a N gambe: corpo = prime 2, campi `_all` solo oltre 2 / N-leg records: body = first 2, `_all` fields only beyond 2
├── docs/                        # STATUS_ARCHIVE_2026H1.md (storico ante 07-08, read-only), MODEL_IMPROVEMENTS, ROADMAP_VOL_BOOK, paper/
├── tests/                       # pytest (test_recent_fixes, test_live_training_parity, test_regime_incremental)
├── deploy/vps/                  # kit deploy VPS (setup_vps.sh + README)
├── avvio_sessione.ps1           # routine di sessione lato casa (§5.3)
├── scripts/
│   ├── 00_*..99_*               # spine numerato = fase della pipeline
│   ├── vol/ research/ vps/ archive/   # script non numerati, per linea — lanciare dalla ROOT
│   └── README.md                # ⇦ mappa canonica script→fase→linea
└── logs/quantsys_YYYYMMDD_HHMMSS.log
```

**La mappa canonica degli script è `scripts/README.it.md`** (tabella completa script→fase→linea, con i flag di ognuno): questa guida non la duplica. Gli script nelle sottocartelle usano `Path(__file__).resolve().parents[2]` e vanno lanciati dalla **root di progetto**.
