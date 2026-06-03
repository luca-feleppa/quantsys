# QUANTSYS — Guida all'avvio

Sistema di trading algoritmico BTC/USDT 1m con ensemble eterogeneo (iTransformer + N-HiTS + TCN+Mamba) e Knowledge Distillation multi-teacher.

> Una versione inglese di questo documento è in [AVVIO.en.md](AVVIO.en.md).

## Stato del sistema (backtest h=30 dopo fix denormalizzazione z-score del 2026-05-23)

Test set 7929 candele, metriche centralizzate in `config/default.yaml`:

| Metrica | Valore | Soglia paper-trading |
|---|---|---|
| Sharpe | **+18.71** | > 0 ✓ |
| Sharpe CI 95% lower bound | **+0.78** | > 0 ✓ |
| Win Rate | **64.3%** | > 50% ✓ |
| Total Return | **+3.67%** | > 0 ✓ |
| Max Drawdown | **0.83%** | < 15% ✓ |
| Fee/Gross ratio | **30.3%** | < 30% ⚠ al limite |
| Stress test (fee×2, slip×3) | Sharpe +7.22 | break-even ✓ |
| Stress test flash crash (fee×1.5, slip×5) | Sharpe +12.30 | break-even ✓ |

Le 3 architetture caricano lo stesso ensemble eterogeneo via `EnsembleModel.load_heterogeneous()` quindi producono lo stesso backtest.

**Live engine — stato attuale**: in modalità paper-only (nessun ordine reale). Il `LiveEngine` ha un mismatch di feature noto (BLOCKER #1: 39 feature live vs **104** training post C-funding) — vedi `TEORIA.md` §11. Stage 2-3 del piano di allineamento ✅ completati (2026-06-02); Stage 4-5 (riscrittura live engine + parity test) pending. Da risolvere prima di usare le predizioni live operativamente.

**Per nuovi entry point al modello**: usare sempre `PipelineState.denormalize_predictions(mu, sigma)` prima di passare le predizioni a `SignalGenerator`. Il modello predice in spazio z-score (RobustScaler); il trading layer opera in spazio raw. Vedi `TEORIA.md` §5 per l'invariante completo.

---

## Comandi rapidi

```bash
python run_all.py                                    # menu interattivo
python run_all.py --arch itransformer                # training singola arch
python run_all.py --arch nhits
python run_all.py --arch tcnmamba
python run_all.py --arch lstm                        # backward compat
python run_all.py --distill                          # Knowledge Distillation multi-teacher
python run_all.py --distill --teacher itransformer   # forza teacher
python run_all.py --only-dashboard                   # solo dashboard + live
python scripts/07_verify_teacher.py                  # confronto architetture
python scripts/99_replay_live_vs_training.py         # diagnostica BLOCKER #1
set QUANTSYS_ARCH=lstm
python scripts/02c_optuna_search.py --n-trials 50    # Optuna (solo LSTM)
```

---

## Primo avvio

### 1. Prerequisiti
```bash
python scripts/00_check_setup.py
```
Controlla dipendenze, CUDA, Binance, FRED. Risolvi errori prima di proseguire.

### 2. Pipeline completa
```bash
python run_all.py
```
Senza flag mostra menu con checkbox (↑↓ naviga, SPAZIO seleziona, A toggle all, INVIO conferma). Apre dashboard su `http://localhost:8050`.

Modalità diretta: `python run_all.py --arch nhits --force-download`.

---

## Training singola arch

Ogni architettura ha config in `config/arch/{arch}.yaml` e output isolati in `models/{arch}/` e `results/{arch}/`. Nessuna interferenza tra run.

| Arch | Comando | Tempo (RTX 2070 Super) | Note |
|---|---|---|---|
| iTransformer | `python run_all.py --arch itransformer --skip-update --skip-macro` | ~25 min | Attention sulle feature, baseline (ICIR 0.795) |
| N-HiTS | `python run_all.py --arch nhits --skip-update --skip-macro` | ~10-15 min | Pure-MLP gerarchico, sostituisce LSTM dal 2026-05-14 |
| TCN+Mamba | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | ~20 min | Conv dilatate + SSM, ottimo per pattern locali |
| LSTM | `python run_all.py --arch lstm --skip-update --skip-macro` | ~30 min | Legacy backward compat (sotto-performante) |

`--skip-update --skip-macro`: usa dati su disco senza ridownload. Prima volta: ometti.

### Verifica risultati
```bash
python scripts/07_verify_teacher.py
```
Tabella comparativa: param count, forward time, Sharpe, WR, n trade, max DD, total return per ogni arch con `best_model.pt`. In alternativa:
- `models/{arch}/config.json` — `best_val_loss`, scaler, n_params
- `models/{arch}/history.json` — curva loss
- `results/{arch}/dashboard_results.json` — metriche backtest

Oppure dalla dashboard: `python run_all.py --only-dashboard` poi dropdown arch nel browser (`/api/archs` rileva quelle con `dashboard_results.json`).

---

## Distillation con più modelli

### Composizione default

Ensemble eterogeneo: **iTransformer + N-HiTS + TCN+Mamba**. LSTM rimosso il 2026-05-14 (val_NLL 5.28 vs iTransformer 0.18 → underfitting strutturale). Codice LSTM intatto, ricaricabile per rollback.

### Pipeline

```bash
python run_all.py --distill --skip-update --skip-macro
```

**Fase 2a — Training candidati**: ogni arch in `distillation.archs` addestrata con `n_ensemble=1`. Se `models/{arch}/best_model.pt` esiste, skippata. Per forzare retrain: `--force-download`.

**Fase 2b — Multi-Teacher Scoring**: ogni candidato valutato alla best epoch con score normalizzato (40% val_loss + 35% spearman + 25% directional accuracy). Pesi softmax con temperature=2 calcolati per tutti. Lo score massimo diventa *primary teacher*; gli altri restano nel pool come teacher pesati.

**Fase 2c — Multi-Teacher Distillation**: ogni student riceve soft labels combinate (media pesata da scoring). Loss mista `(1-α)·NLL_reale + α·loss_distillazione` con α=0.3. Loss distillazione scala-normalizzata per μ/σ/ν. Soft labels integrate nel TensorDataset (shuffle-safe). Epoche ridotte al 60%. Student già distillati skippati automaticamente.

**Ensemble eterogeneo (inferenza)**: tutti i modelli predicono insieme, output combinato con legge della varianza totale:
- `mu_ens = Σ w_i · mu_i`
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i - mu_ens)²)`

Pesi default in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`): iTransformer/N-HiTS/TCNMamba/TFT = 1.0, LSTM = 0.5.

### Cambiare composizione

**Un solo posto**: `config/default.yaml` → `distillation.archs`:
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
Esempi: `["itransformer", "lstm", "tcnmamba"]` rollback legacy; `["itransformer", "nhits", "tcnmamba", "lstm"]` ensemble a 4; `["itransformer", "tcnmamba"]` solo 2.

Dopo la modifica, `python run_all.py --distill`: addestra mancanti, scoring, distill student, backtest/live usano automaticamente la nuova composizione.

### Forzare un teacher specifico
```bash
python run_all.py --distill --teacher itransformer
```
Salta lo scoring automatico. Gli altri restano nel pool pesato.

### Verificare la distillation

In `models/{arch}/config.json`:
- `distilled: true`
- `teacher_arch: "multi-teacher"`

Arch già distillata viene skippata in Fase 2c. Per forzare ri-distillation: cancella `best_model.pt` o usa `--force-download`.

---

## Avvii successivi

```bash
python scripts/06_dashboard.py            # solo dashboard
python run_all.py --only-dashboard        # idem
python run_all.py --skip-train --skip-walkfwd   # aggiornamento dati, stessi modelli
python run_all.py                          # menu
python run_all.py --distill                # full + distillation
```

---

## Flag utili

| Flag | Effetto |
|------|---------|
| `--skip-update` | Usa dataset esistente, no download |
| `--skip-macro` | Salta download FRED/yFinance |
| `--skip-train` | Usa modello esistente, no retrain |
| `--skip-walkfwd` | Salta walk-forward validation |
| `--skip-backtest` | Salta backtest |
| `--skip-live` | No feed live WebSocket |
| `--skip-analyze` | Salta `05_analyze_signals.py` |
| `--only-dashboard` | Solo dashboard + live, no ML |
| `--no-browser` | Non aprire browser |
| `--force-download` | Ri-scarica + forza retrain |
| `--max-model-age-days N` | Retrain se modello > N giorni |
| `--distill` | Pipeline multi-teacher |
| `--teacher ARCH` | Forza primary teacher |

---

## Hardware

### CPU
`config/default.yaml`:
```yaml
hardware:
  cpu_fraction: 0.5   # 0.3=30%, 0.5=50%, 0.8=80%
```
Default 0.5 (4 thread su 8 core). Letto da tutti gli script all'avvio.

### GPU compute
```powershell
nvidia-smi -pl 125    # limita (RTX 2070 Super min=125 max=215W)
nvidia-smi -pl 215    # ripristina
```

### Setup di riferimento (RTX 2070 Super 8GB)

| Componente | Valore |
|---|---|
| CUDA, AMP fp16 training | sì (via `setup_device`) |
| AMP inference | **off** hardcoded in `ensemble.py:170` (evita NaN spectral_norm + Mamba scan) |
| Batch inference backtest | 256 (`scripts/03_backtest.py`) |
| Batch training | 64 (default `config/arch/<arch>.yaml`) |

### Solo CPU
Il codice fa fallback automatico via `setup_device` (`quantsys/utils/__init__.py`). Su `quantsys/model/__init__.py:67`, `autocast(device_type="cuda")` è no-op silenzioso su CPU. Tempi:
- Training: 20-50× più lento (tcnmamba ~3h GPU → 2-3 giorni CPU). Sconsigliato.
- Backtest: ~5s GPU → 30-60s CPU. Tollerabile.
- Live: ~50-100ms vs ~20ms GPU. Pienamente utilizzabile (latency WS Binance domina).

### Apple Silicon / AMD / Intel Arc
Non testato. Codice usa `torch.cuda.*`. Per MPS servono modifiche a `setup_device` e probabilmente kernel custom per Mamba/SSM.

### Poca VRAM (4GB)
`config/arch/<arch>.yaml`:
```yaml
batch_size: 32
gradient_accumulation_steps: 2   # mantiene effective batch=64
```
Inference batch in `scripts/03_backtest.py` da 256 → 128.

### Molta VRAM (≥16GB)
```yaml
batch_size: 128
```
Inference batch fino a 1024 (guadagno marginale, GPU già satura).

---

## Architetture

| Arch | Classe | File | Note |
|---|---|---|---|
| `itransformer` | `QuantiTransformer` | `quantsys/model/__init__.py:1025` | Attention sulle feature, baseline |
| `nhits` | `QuantNHiTS` | `quantsys/model/nhits.py:110` | Pure-MLP gerarchico |
| `tcnmamba` | `QuantTCNMamba` | `quantsys/model/tcn_mamba.py:341` | TCN dilatate + SSM ibrido |
| `lstm` | `QuantLSTM` | `quantsys/model/__init__.py:309` | Legacy |
| `tft` | `QuantTFT` | `quantsys/model/__init__.py:797` | Temporal Fusion Transformer |

### Aggiungere una nuova arch
1. Classe in `quantsys/model/` con `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher in `quantsys/model/__init__.py:load_model`
3. Branch in `scripts/02_train.py` (`architecture == "X"`)
4. `config/arch/X.yaml`
5. `choices` in `run_all.py` (parser `--arch` e `--teacher`)
6. Whitelist in `06_dashboard.py`, `05_analyze_signals.py`
7. (Opzionale) `distillation.archs` in `config/default.yaml`

---

## Optuna

```bash
set QUANTSYS_ARCH=lstm
python scripts/02c_optuna_search.py --n-trials 50 --study-name quantsys
```
**Limiti**: hardcoded su `QuantLSTM`. `best_params.json` salvato in `models/lstm/` NON applicato automaticamente al training successivo — copia manuale in `config/arch/lstm.yaml`.

Studio persistente su SQLite (`models/lstm/optuna_quantsys.db`), ripristinabile.

---

## Ensemble omogeneo (5× stessa arch, legacy)

`config/default.yaml`:
```yaml
training:
  n_ensemble: 5   # default attuale = 5 (override automatico a 1 in --distill)
```
Output: `models/{arch}/best_model_0..4.pt`. Backtest/live li caricano via `EnsembleModel.load`. Indipendente dalla distillation (modalità non si escludono).

---

## File layout

```
quantsys_project/
├── config/
│   ├── default.yaml             # config base (distillation.archs qui)
│   ├── secrets.yaml             # FRED key, gitignored
│   └── arch/
│       ├── itransformer.yaml, nhits.yaml, tcnmamba.yaml, lstm.yaml
├── data/
│   ├── raw_candles.parquet      # OHLCV storico
│   ├── features.parquet         # feature normalizzate
│   ├── lstm_dataset.npz         # windows X/y per training
│   ├── funding_rate.parquet     # funding futures (8h)
│   └── macro_*.parquet          # FRED/yFinance
├── models/
│   ├── teacher_analysis.json    # output 07_verify_teacher.py
│   └── {arch}/
│       ├── best_model.pt        # checkpoint
│       ├── config.json          # iperparametri + flag distilled/teacher_arch
│       ├── history.json         # curva loss
│       └── pipeline_state.pkl   # scalers + feature config
├── results/{arch}/
│   ├── dashboard_results.json
│   └── live_signals.jsonl
├── tests/                       # pytest (test_recent_fixes.py: regression fix critici)
├── scripts/                     # 00-07 numerati + 99_replay_live_vs_training.py
└── logs/quantsys_YYYYMMDD_HHMMSS.log
```

---

## Dashboard — pulsante "Aggiorna"

1. **Aggiorna** in alto a destra
2. Seleziona step da eseguire
3. **Avvia** → barra di progresso
4. Al termine dashboard si aggiorna automaticamente
5. **Annulla** per fermare un job

Switch arch: dropdown in alto (rileva arch con `dashboard_results.json`).

---

## Fermare tutto

`Ctrl+C` nel terminale di `run_all.py`. Termina dashboard + live feed.
