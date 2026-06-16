# QUANTSYS — Guida all'avvio · QUANTSYS — Quick start guide

🇮🇹 Sistema di trading algoritmico BTC/USDT con ensemble eterogeneo (iTransformer + N-HiTS + TCN+Mamba) e Knowledge Distillation multi-teacher. **Timeframe corrente: candele 1h** (pivot 2026-06-09; il precedente perimetro 1m è in backup, vedi sezione pivot).

**EN** Algorithmic trading system for BTC/USDT with a heterogeneous ensemble (iTransformer + N-HiTS + TCN+Mamba) and multi-teacher Knowledge Distillation. **Current timeframe: 1h candles** (2026-06-09 pivot; the previous 1m perimeter is backed up, see the pivot section).

## Stato del sistema (aggiornato 2026-06-09 — pivot timeframe 1m→1h) · System status (updated 2026-06-09 — 1m→1h timeframe pivot)

🇮🇹 Pivot al timeframe **1h** (Strada 1 dopo il KILL del probe cross-sectional 2026-06-06, diagnosi "muro = magnitudine non segno"): a 1h il rapporto costo/σ per barra scende da ~1.9–3.3× a ~0.25–0.42× (il movimento di barra cresce ∝ √Δt, il costo roundtrip è fisso). **Stesso motore, design interval-agnostic**: tutte le conversioni di finestra sono identità a 1m. Razionale econometrico in `TEORIA.md` §1, dettaglio implementativo in `docs/MODEL_IMPROVEMENTS.md` (sezione 2026-06-09).

**EN** Pivot to the **1h** timeframe (Path 1 after the 2026-06-06 cross-sectional probe KILL, diagnosis "the wall is magnitude, not sign"): at 1h the per-bar cost/σ ratio drops from ~1.9–3.3× to ~0.25–0.42× (bar move grows ∝ √Δt, roundtrip cost is fixed). **Same engine, interval-agnostic design**: every window conversion is an identity at 1m. Econometric rationale in `TEORIA.md` §1, implementation detail in `docs/MODEL_IMPROVEMENTS.md` (2026-06-09 section).

🇮🇹 **Config pivot (`config/default.yaml`):**

🇮🇹
| Parametro | Valore 1h | Era (1m) | Nota |
|---|---|---|---|
| `data.interval` | `1h` | `1m` | tutte le finestre derivano da qui via `interval_minutes` |
| `data.start_time` | `2019-01-01` | `2025-05-19` | storico multi-anno, ~65k barre |
| `window_stride` | 1 | 5 | massimizza i sample su ~65k barre |
| `embargo_steps` | 168 (1 settimana) | 1500 (~25h) | ≥ window_size+horizon = 150 |
| `max_hold_candles` | 60 (2.5 giorni) | 240 (4h) | vincolo ≥ h=30 |
| `min_expected_ret` | 0.0013 (13 bps cost-aware) | 0.0005 | 2° test pre-registrato a 23 bps |
| `max_sigma` | 0.10 (≈0.015·√60) | 0.015 | da ricalibrare sui percentili post-denorm |
| `forecast_horizon` | 30 (INVARIATO) | 30 | ma ora = **30 ORE** (era 30 min) |
| `window_size` | 120 (INVARIATO) | 120 | ma ora = **5 giorni** di contesto (era 2h) |

**EN**
| Parameter | 1h value | Was (1m) | Note |
|---|---|---|---|
| `data.interval` | `1h` | `1m` | every window derives from it via `interval_minutes` |
| `data.start_time` | `2019-01-01` | `2025-05-19` | multi-year history, ~65k bars |
| `window_stride` | 1 | 5 | maximizes samples on ~65k bars |
| `embargo_steps` | 168 (1 week) | 1500 (~25h) | ≥ window_size+horizon = 150 |
| `max_hold_candles` | 60 (2.5 days) | 240 (4h) | constraint ≥ h=30 |
| `min_expected_ret` | 0.0013 (13 bps cost-aware) | 0.0005 | 2nd pre-registered test at 23 bps |
| `max_sigma` | 0.10 (≈0.015·√60) | 0.015 | to recalibrate on post-denorm percentiles |
| `forecast_horizon` | 30 (UNCHANGED) | 30 | but now = **30 HOURS** (was 30 min) |
| `window_size` | 120 (UNCHANGED) | 120 | but now = **5 days** of context (was 2h) |

🇮🇹 **Dati:** `data/raw_candles.parquet` = candele 1h 2019→oggi (**65.145 barre**); funding completo dal lancio dei perp 2019-09-10 (**7.394 osservazioni**, re-download completo — il vecchio file partiva dal 2021). Dataset npz: `X_train (51120, 120, 104)` — **stessa composizione canonica 104 = 86 dinamiche + 18 strutturali**.

**EN** **Data:** `data/raw_candles.parquet` = 1h candles 2019→today (**65,145 bars**); full funding since the perp launch 2019-09-10 (**7,394 observations**, full re-download — the old file started in 2021). Npz dataset: `X_train (51120, 120, 104)` — **same canonical 104 = 86 dynamic + 18 structural composition**.

🇮🇹 **Nuovi guard:** `RuntimeError` "interval mismatch" in `03_backtest.py` e `04_live_signals.py` (stesso pattern del guard `forecast_horizon`): modello addestrato a 1m + config 1h = combinazione invalida bloccata. I consumer live/replay derivano l'interval da `PipelineState.interval_minutes` (fallback 1 per i pkl legacy), non dalla config. σ safety-net scalata a `0.05·√interval_minutes` (≈0.387 a 1h). Annualizzazione interval-aware: `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760).

**EN** **New guards:** `RuntimeError` "interval mismatch" in `03_backtest.py` and `04_live_signals.py` (same pattern as the `forecast_horizon` guard): a 1m-trained model + 1h config = invalid combination, blocked. Live/replay consumers derive the interval from `PipelineState.interval_minutes` (fallback 1 for legacy pkl), not from the config. σ safety net scaled to `0.05·√interval_minutes` (≈0.387 at 1h). Interval-aware annualization: `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760).

🇮🇹 **Rollback 1m:** ripristina la config 1m (`interval: 1m`, `start_time: 2025-05-19`, stride 5, embargo 1500, max_hold 240, min_ret 0.0005, max_sigma 0.015) + restore `data/backup_1m/*` e `models/backup_1m/*`. **Il codice non va toccato**: tutte le conversioni sono identità a 1m.

**EN** **1m rollback:** restore the 1m config (`interval: 1m`, `start_time: 2025-05-19`, stride 5, embargo 1500, max_hold 240, min_ret 0.0005, max_sigma 0.015) + restore `data/backup_1m/*` and `models/backup_1m/*`. **No code changes needed**: every conversion is an identity at 1m.

🇮🇹 **Stato:** dati e config completati; **training e backtest 1h NON ancora eseguiti** (in corso). **Gate pre-registrato del pivot:** Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 a ENTRAMBI i costi 13 e 23 bps sul test OOS.

**EN** **Status:** data and config done; **1h training and backtest NOT yet executed** (in progress). **Pre-registered pivot gate:** Sharpe≥1.0, PF≥1.3, ≥80 trades, net>0 at BOTH 13 and 23 bps costs on the OOS test set.

## Stato precedente — perimetro 1m (2026-06-04 — rollback T=120) · Previous state — 1m perimeter (2026-06-04 — T=120 rollback)

🇮🇹 Dopo gli esperimenti su window-size (T=180 e T=240 **regressivi**) il sistema è stato riportato a **T=120** (sweet spot empirico su ~525k candele 1m; finestre più lunghe over-fittano il noise temporale). Dataset corrente 549k candele → test set **10.104** finestre. Retrain 5-seed dei 3 archi su questa configurazione.

**EN** After the window-size experiments (T=180 and T=240 both **regressive**) the system was rolled back to **T=120** (empirical sweet spot on ~525k 1m candles; longer windows overfit temporal noise). Current dataset 549k candles → test set **10,104** windows. 5-seed retrain of the 3 archs on this configuration.

🇮🇹 **Stato corrente del backtest grezzo (test set, single-arch):**

**EN** **Current raw-backtest status (test set, single-arch):**

🇮🇹
| Config | Sharpe | Total Return | Note |
|---|---|---|---|
| iTransformer (miglior arch) | ≈ −14.6 (1-seed) | −1.77% | il meno-peggio |
| Ensemble eterogeneo (3 arch) | ≈ −19.6 | −5.5% | non batte il singolo (errori cross-arch ≈0.995) |

**EN**
| Config | Sharpe | Total Return | Note |
|---|---|---|---|
| iTransformer (best arch) | ≈ −14.6 (1-seed) | −1.77% | the least-bad |
| Heterogeneous ensemble (3 archs) | ≈ −19.6 | −5.5% | does not beat the single (cross-arch error ≈0.995) |

🇮🇹 **Il backtest grezzo è attualmente negativo** — nessuna config supera le soglie paper-trading. Le metriche del 2026-05-23 (Sharpe +18.71) erano relative a un modello/feature-space precedente, ormai superato.

**EN** **The raw backtest is currently negative** — no config clears the paper-trading thresholds. The 2026-05-23 metrics (Sharpe +18.71) referred to a prior model/feature-space, now superseded.

🇮🇹 **Edge reale identificato (2026-06-04): è regime-condizionato.** In regime **R0 Quiet** il modello ha Spearman **+0.13÷0.19**, stabile in **tutti** i sotto-periodi OOS (val+test). È edge di *rango*: l'entry a soglia |μ| non lo cattura (Quiet = bassa vol → μ piccole). Un entry **rank-based regime-specifico** (sperimentale, env-gated in `03_backtest.py`: `QUANTSYS_QUIET_RANK_Q` + `QUANTSYS_QUIET_MIN_SIGMA`) recuperava parzialmente sul test (best −0.74% return, MDD 1.5%). ⚠ **Validazione su val FALLITA (2026-06-05):** sullo split held-out (`QUANTSYS_BACKTEST_SPLIT=val`) la stessa config dà return **−0.22%**, PF 0.84, solo 13 trade (val è 71% Stress, 14% Quiet → sotto-campione) → l'edge **non regge fuori campione**, non promosso nel `SignalGenerator`; resta env-flag inerte. Produzione = ensemble eterogeneo pulito / iTrans standalone. L'inversione del regime Trending **non è robusta** (validazione OOS la smonta) → non deployata. ⚠ **Le metriche in-sample (val_nll, Spearman walkforward) anti-correlano col backtest** (distribution shift strutturale): non usarle per ottimizzare. Dettaglio completo in `docs/MODEL_IMPROVEMENTS.md` e `STATUS.md`.

**EN** **Real edge identified (2026-06-04): it is regime-conditional.** In regime **R0 Quiet** the model has Spearman **+0.13÷0.19**, stable across **all** OOS sub-periods (val+test). It is a *rank* edge: the |μ|-threshold entry misses it (Quiet = low vol → small μ). A **regime-specific rank-based entry** (experimental, env-gated in `03_backtest.py`: `QUANTSYS_QUIET_RANK_Q` + `QUANTSYS_QUIET_MIN_SIGMA`) partially recovered it on the test set (best −0.74% return, MDD 1.5%). ⚠ **Validation on val FAILED (2026-06-05):** on the held-out split (`QUANTSYS_BACKTEST_SPLIT=val`) the same config returns **−0.22%**, PF 0.84, only 13 trades (val is 71% Stress, 14% Quiet → underpowered) → the edge **does not hold out-of-sample**, not promoted into `SignalGenerator`; it stays an inert env-flag. Production = clean heterogeneous ensemble / iTrans standalone. The Trending-regime inversion is **not robust** (OOS validation breaks it) → not deployed. ⚠ **In-sample metrics (val_nll, walkforward Spearman) anti-correlate with the backtest** (structural distribution shift): do not use them to optimize. Full detail in `docs/MODEL_IMPROVEMENTS.md` and `STATUS.md`.

🇮🇹 **Harvest dell'edge ordinale — 2 leve sperimentali (2026-06-05): validate su val e FALLITE.** Aggiunti due flag env in `03_backtest.py` (inerti di default, reversibili): **Fix ① cadenza decisionale** `QUANTSYS_DECISION_CADENCE=h` (entry solo ogni ≥`forecast_horizon` candele; exit ogni candela) e **Fix ② esposizione continua rank-based** `QUANTSYS_RANK_EXPOSURE=1` (direzione+`conviction` dal percentile causale di μ, regime-gated su R0 Quiet, no-trade band `QUANTSYS_RANK_BAND` = isteresi). ⚠ **Esito val (`QUANTSYS_BACKTEST_SPLIT=val`):** baseline pulito **+4.03%/PF 1.88/WR 61%** → con Fix ①② **−2.24%/PF 0.22/WR 25.7%**. Il segnale rank come entry direzionale è **anti-predittivo OOS**; l'esposizione continua flippa (27/35 chiusure SIGNAL) → la PnL è dominata dal path SL/TP, non dal rendimento a orizzonte su cui vive lo Spearman. **NON promossi**, restano flag inerti. (Baseline val +4% è il lato favorevole dello shift val→test; il baseline su test resta negativo.)

**EN** **Harvesting the ordinal edge — 2 experimental levers (2026-06-05): validated on val and FAILED.** Two env flags were added to `03_backtest.py` (inert by default, reversible): **Fix ① decision cadence** `QUANTSYS_DECISION_CADENCE=h` (entries only every ≥`forecast_horizon` candles; exits every candle) and **Fix ② continuous rank-based exposure** `QUANTSYS_RANK_EXPOSURE=1` (direction+`conviction` from the causal μ-percentile, regime-gated on R0 Quiet, no-trade band `QUANTSYS_RANK_BAND` = hysteresis). ⚠ **Val outcome (`QUANTSYS_BACKTEST_SPLIT=val`):** clean baseline **+4.03%/PF 1.88/WR 61%** → with Fix ①② **−2.24%/PF 0.22/WR 25.7%**. The rank signal as a directional entry is **anti-predictive OOS**; continuous exposure flips (27/35 SIGNAL exits) → PnL is dominated by the SL/TP path, not the horizon return the Spearman lives on. **Not promoted**, flags stay inert. (The +4% val baseline is the favorable side of the val→test shift; the test baseline stays negative.)

🇮🇹 Le 3 architetture caricano lo stesso ensemble eterogeneo via `EnsembleModel.load_heterogeneous()` quindi producono lo stesso backtest.

**EN** All 3 architectures load the same heterogeneous ensemble via `EnsembleModel.load_heterogeneous()`, so they produce the same backtest.

🇮🇹 **Live engine — stato attuale**: paper-only (nessun ordine reale), ma **BLOCKER #1 RISOLTO (2026-06-05)**: il path live usa ora `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canoniche, stesso scaler) → `_deterministic_predict` → `SignalGenerator`, con **parity feature E segnale bit-perfect** (`tests/test_live_training_parity.py` 5/5; `99_replay_live_vs_training.py` Δ=0). I segnali paper ora riflettono il backtest — vedi `TEORIA.md` §11. Residuo operativo: smoke test WS reale + paper-trading. ⚠ Il backtest è negativo OOS: il paper-trading serve ad accumulare trade reali.

**EN** **Live engine — current status**: paper-only mode (no real orders), but **BLOCKER #1 RESOLVED (2026-06-05)**: the live path now uses `LiveCandleBuffer`→`FeatureAssembler`→`FeatureBuilder.build` (104 canonical, same scaler) → `_deterministic_predict` → `SignalGenerator`, with **bit-perfect feature AND signal parity** (`tests/test_live_training_parity.py` 5/5; `99_replay_live_vs_training.py` Δ=0). Paper signals now reflect the backtest — see `TEORIA.md` §11. Operational remainder: real WS smoke test + paper-trading. ⚠ The backtest is negative OOS: paper-trading is for accumulating real trades.

🇮🇹 **Per nuovi entry point al modello**: usare sempre `PipelineState.denormalize_predictions(mu, sigma)` prima di passare le predizioni a `SignalGenerator`. Il modello predice in spazio z-score (RobustScaler); il trading layer opera in spazio raw. Vedi `TEORIA.md` §5 per l'invariante completo.

**EN** **For any new model entry point**: always call `PipelineState.denormalize_predictions(mu, sigma)` before passing predictions to `SignalGenerator`. The model predicts in z-score space (RobustScaler); the trading layer operates in raw space. See `TEORIA.md` §5 for the full invariant.

---

## Comandi rapidi · Quick commands

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

## Primo avvio · First run

### 1. Prerequisiti · 1. Prerequisites

```bash
python scripts/00_check_setup.py
```
Controlla dipendenze, CUDA, Binance, FRED. Risolvi errori prima di proseguire.

### 2. Pipeline completa · 2. Full pipeline

```bash
python run_all.py
```
Senza flag mostra menu con checkbox (↑↓ naviga, SPAZIO seleziona, A toggle all, INVIO conferma). Apre dashboard su `http://localhost:8050`.

🇮🇹 Modalità diretta: `python run_all.py --arch nhits --force-download`.

**EN** Direct mode: `python run_all.py --arch nhits --force-download`.

🇮🇹 **Risoluzione candela:** `python run_all.py --interval 1h` (o `1m`) applica l'overlay `config/interval/{interval}.yaml` sopra `default.yaml` (merge shallow per-sezione, dopo secrets e prima dell'overlay arch) e propaga `QUANTSYS_INTERVAL` a tutti i subprocess. Senza flag: config as-is. Le choices sono derivate dai file presenti in `config/interval/`.

**EN** **Candle resolution:** `python run_all.py --interval 1h` (or `1m`) applies the `config/interval/{interval}.yaml` overlay on top of `default.yaml` (per-section shallow merge, after secrets and before the arch overlay) and propagates `QUANTSYS_INTERVAL` to every subprocess. Without the flag: config as-is. Choices are derived from the files present in `config/interval/`.

---

## Training singola arch · Training a single architecture

🇮🇹 Ogni architettura ha config in `config/arch/{arch}.yaml` e output isolati in `models/{arch}/` e `results/{arch}/`. Nessuna interferenza tra run.

**EN** Each architecture has its own config in `config/arch/{arch}.yaml` and isolated outputs in `models/{arch}/` and `results/{arch}/`. No cross-run interference.

🇮🇹
| Arch | Comando | Tempo (RTX 2070 Super) | Note |
|---|---|---|---|
| iTransformer | `python run_all.py --arch itransformer --skip-update --skip-macro` | ~25 min | Attention sulle feature, baseline (ICIR 0.795) |
| N-HiTS | `python run_all.py --arch nhits --skip-update --skip-macro` | ~10-15 min | Pure-MLP gerarchico, sostituisce LSTM dal 2026-05-14 |
| TCN+Mamba | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | ~20 min | Conv dilatate + SSM, ottimo per pattern locali |
| LSTM | `python run_all.py --arch lstm --skip-update --skip-macro` | ~30 min | Legacy backward compat (sotto-performante) |

**EN**
| Arch | Command | Time (RTX 2070 Super) | Notes |
|---|---|---|---|
| iTransformer | `python run_all.py --arch itransformer --skip-update --skip-macro` | ~25 min | Attention over features, baseline (ICIR 0.795) |
| N-HiTS | `python run_all.py --arch nhits --skip-update --skip-macro` | ~10–15 min | Hierarchical pure-MLP, replaces LSTM since 2026-05-14 |
| TCN+Mamba | `python run_all.py --arch tcnmamba --skip-update --skip-macro` | ~20 min | Dilated conv + SSM, great for local patterns |
| LSTM | `python run_all.py --arch lstm --skip-update --skip-macro` | ~30 min | Legacy backward compat (under-performing) |

🇮🇹 `--skip-update --skip-macro`: usa dati su disco senza ridownload. Prima volta: ometti.

**EN** `--skip-update --skip-macro`: use on-disk data without redownload. First run: omit them.

### Verifica risultati · Inspecting results

```bash
python scripts/07_verify_teacher.py
```
Tabella comparativa: param count, forward time, Sharpe, WR, n trade, max DD, total return per ogni arch con `best_model.pt`. In alternativa:
- `models/{arch}/config.json` — `best_val_loss`, scaler, n_params
- `models/{arch}/history.json` — curva loss
- `results/{arch}/dashboard_results.json` — metriche backtest

🇮🇹 Oppure dalla dashboard: `python run_all.py --only-dashboard` poi dropdown arch nel browser (`/api/archs` rileva quelle con `dashboard_results.json`).

**EN** Or from the dashboard: `python run_all.py --only-dashboard` then arch dropdown (`/api/archs` auto-detects archs with `dashboard_results.json`).

---

## Distillation con più modelli · Distillation with multiple models

### Composizione default · Default composition

🇮🇹 Ensemble eterogeneo: **iTransformer + N-HiTS + TCN+Mamba**. LSTM rimosso il 2026-05-14 (val_NLL 5.28 vs iTransformer 0.18 → underfitting strutturale). Codice LSTM intatto, ricaricabile per rollback.

**EN** Heterogeneous ensemble: **iTransformer + N-HiTS + TCN+Mamba**. LSTM was removed on 2026-05-14 (val_NLL 5.28 vs iTransformer 0.18 → structural underfitting). LSTM code is intact and reloadable for rollback.

### Pipeline

```bash
python run_all.py --distill --skip-update --skip-macro
```

🇮🇹 **Fase 2a — Training candidati**: ogni arch in `distillation.archs` addestrata con `n_ensemble=1` di default; override SOLO passando `--n-ensemble N` esplicito sulla CLI (vale per candidati E student). Se `models/{arch}/best_model.pt` esiste, skippata. Per forzare retrain: `--force-download`.

**EN** **Phase 2a — Candidate training**: each arch in `distillation.archs` is trained with `n_ensemble=1` by default; override ONLY by passing an explicit `--n-ensemble N` on the CLI (applies to candidates AND students). If `models/{arch}/best_model.pt` exists, it is skipped. To force retrain: `--force-download`.

🇮🇹 **Fase 2b — Multi-Teacher Scoring**: ogni candidato valutato alla best epoch con score normalizzato (40% val_loss + 35% spearman + 25% directional accuracy). Pesi softmax con temperature=2 calcolati per tutti. Lo score massimo diventa *primary teacher*; gli altri restano nel pool come teacher pesati.

**EN** **Phase 2b — Multi-Teacher Scoring**: every candidate is evaluated at its best epoch with normalized scoring (40% val_loss + 35% Spearman + 25% directional accuracy). Softmax weights with temperature=2 are computed for all of them. The top score becomes the *primary teacher*; the others stay in the pool as weighted teachers.

🇮🇹 **Fase 2c — Multi-Teacher Distillation**: ogni student riceve soft labels combinate (media pesata da scoring). Loss mista `(1-α)·NLL_reale + α·loss_distillazione` con α=0.3. Loss distillazione scala-normalizzata per μ/σ/ν. Soft labels integrate nel TensorDataset (shuffle-safe). Epoche ridotte al 60%. Student già distillati skippati automaticamente.

**EN** **Phase 2c — Multi-Teacher Distillation**: each student receives soft labels combined as a weighted mean from scoring. Mixed loss `(1−α)·NLL_real + α·distill_loss` with α=0.3. Distillation loss is scale-normalized for μ/σ/ν. Soft labels are integrated into the TensorDataset (shuffle-safe). Epochs are cut to 60%. Students already distilled are skipped automatically.

🇮🇹 **Ensemble eterogeneo (inferenza)**: tutti i modelli predicono insieme, output combinato con legge della varianza totale:
- `mu_ens = Σ w_i · mu_i`
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i - mu_ens)²)`

**EN** **Heterogeneous ensemble (inference)**: all models predict together, the output is combined using the law of total variance:
- `mu_ens = Σ w_i · mu_i`
- `sigma_ens = sqrt(Σ w_i · sigma_i² + Σ w_i · (mu_i − mu_ens)²)`

🇮🇹 Pesi default in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`): iTransformer/N-HiTS/TCNMamba/TFT = 1.0, LSTM = 0.5.

**EN** Default weights in `DEFAULT_ARCH_WEIGHTS` (`quantsys/model/ensemble.py`): iTransformer/N-HiTS/TCNMamba/TFT = 1.0, LSTM = 0.5.

### Cambiare composizione · Changing the composition

🇮🇹 **Un solo posto**: `config/default.yaml` → `distillation.archs`:
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
Esempi: `["itransformer", "lstm", "tcnmamba"]` rollback legacy; `["itransformer", "nhits", "tcnmamba", "lstm"]` ensemble a 4; `["itransformer", "tcnmamba"]` solo 2.

**EN** **One spot only**: `config/default.yaml` → `distillation.archs`:
```yaml
distillation:
  archs:
    - itransformer
    - nhits
    - tcnmamba
```
Examples: `["itransformer", "lstm", "tcnmamba"]` legacy rollback; `["itransformer", "nhits", "tcnmamba", "lstm"]` 4-model ensemble; `["itransformer", "tcnmamba"]` just 2.

🇮🇹 Dopo la modifica, `python run_all.py --distill`: addestra mancanti, scoring, distill student, backtest/live usano automaticamente la nuova composizione.

**EN** After editing, `python run_all.py --distill` trains the missing models, scores, distills students; backtest/live automatically pick up the new composition.

### Forzare un teacher specifico · Forcing a specific teacher

```bash
python run_all.py --distill --teacher itransformer
```
Salta lo scoring automatico. Gli altri restano nel pool pesato.

### Verificare la distillation · Verifying distillation

🇮🇹 In `models/{arch}/config.json`:
- `distilled: true`
- `teacher_arch: "multi-teacher"`

**EN** In `models/{arch}/config.json`:
- `distilled: true`
- `teacher_arch: "multi-teacher"`

🇮🇹 Arch già distillata viene skippata in Fase 2c. Per forzare ri-distillation: cancella `best_model.pt` o usa `--force-download`.

**EN** An already-distilled arch is skipped in Phase 2c. To force re-distillation: delete `best_model.pt` or use `--force-download`.

---

## Avvii successivi · Subsequent runs

```bash
python scripts/06_dashboard.py            # solo dashboard
python run_all.py --only-dashboard        # idem
python run_all.py --skip-train --skip-walkfwd   # aggiornamento dati, stessi modelli
python run_all.py                          # menu
python run_all.py --distill                # full + distillation
```

---

## Flag utili · Useful flags

🇮🇹
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

**EN**
| Flag | Effect |
|------|--------|
| `--skip-update` | Use existing dataset, no download |
| `--skip-macro` | Skip FRED/yFinance download |
| `--skip-train` | Use existing model, no retrain |
| `--skip-walkfwd` | Skip walk-forward validation |
| `--skip-backtest` | Skip backtest |
| `--skip-live` | No live WebSocket feed |
| `--skip-analyze` | Skip `05_analyze_signals.py` |
| `--only-dashboard` | Dashboard + live only, no ML |
| `--no-browser` | Do not auto-open browser |
| `--force-download` | Redownload + force retrain |
| `--max-model-age-days N` | Retrain if model older than N days |
| `--distill` | Multi-teacher distillation pipeline |
| `--teacher ARCH` | Force primary teacher |

---

## Esperimenti famiglia vol (forecasting di volatilità) · Vol-family experiments (volatility forecasting)

🇮🇹 Con `features.target_type` in `config/default.yaml` il target cambia famiglia (default `ret` = direzionale, bit-invariato): `log_rv` = log-realized-variance delle prossime h barre (giudice QLIKE `scripts/vol/dev_vols_qlike.py` vs HAR-RV+naive); `log_rs_ratio` = asimmetria semivarianza `log(RS⁺/RS⁻)` (giudice MSE `scripts/vol/dev_vols_rs_judge.py` vs HAR-RS+naive+train-mean). Pipeline comune: `01_download_data.py` (rebuild dataset) → `python scripts/vol/dev_vols_macro_append.py` (ri-appende X_macro senza rifare il walk-forward regime, ~5s — NB il `01b` completo su 7 anni costa ~3h) → `02_train.py --n-ensemble 5` → giudice con `QUANTSYS_VOLS_SPLIT=val` (val-first; poi `=test` UNA volta). Report in `results/vols/`. ⚠ NO backtest trading sui modelli vol (`03_backtest.py` non ha senso su questi target). **Esiti:** `log_rv` 2026-06-10 **PASS a 1h** (NN −30% QLIKE vs HAR-RV su test) ma **FAIL a 1m** (edge risoluzione-specifico); `log_rs_ratio` 2026-06-11 **FAIL** (asimmetria impredicibile per NN e HAR-RS → i momenti pari generalizzano, i dispari no). Backup modelli: vol-1h PASS in `models/backup_1h_vols/`, vol-1m in `models/backup_1m_vols/`, direzionale 1m in `models/backup_1m/`.

**EN** Via `features.target_type` in `config/default.yaml` the target switches family (default `ret` = directional, bit-invariant): `log_rv` = log realized variance of the next h bars (QLIKE judge `scripts/vol/dev_vols_qlike.py` vs HAR-RV+naive); `log_rs_ratio` = semivariance asymmetry `log(RS⁺/RS⁻)` (MSE judge `scripts/vol/dev_vols_rs_judge.py` vs HAR-RS+naive+train-mean). Shared pipeline: `01_download_data.py` (dataset rebuild) → `python scripts/vol/dev_vols_macro_append.py` (re-appends X_macro without re-running the regime walk-forward, ~5s — NB the full `01b` over 7 years costs ~3h) → `02_train.py --n-ensemble 5` → judge with `QUANTSYS_VOLS_SPLIT=val` (val-first; then `=test` ONCE). Reports in `results/vols/`. ⚠ NO trading backtest on vol models (`03_backtest.py` is meaningless on these targets). **Outcomes:** `log_rv` 2026-06-10 **PASS at 1h** (NN −30% QLIKE vs HAR-RV on test) but **FAIL at 1m** (resolution-specific edge); `log_rs_ratio` 2026-06-11 **FAIL** (asymmetry unpredictable for NN and HAR-RS alike → even moments generalize, odd ones don't). Model backups: vol-1h PASS in `models/backup_1h_vols/`, vol-1m in `models/backup_1m_vols/`, directional 1m in `models/backup_1m/`.

---

## Poller IV Deribit (monetizzazione vol 1h) · Deribit IV poller (1h vol monetization)

🇮🇹 `python scripts/01c_iv_poller.py` (loop, default 10 min; `--minutes N` per la cadenza, `--once` smoke, `--backfill-dvol` storico DVOL orario 2021→oggi) — 2 richieste pubbliche Deribit per tick, NESSUN account richiesto. Output append-only atomico in `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (snapshot raw, ~950 strumenti/tick), `atm_30h.parquet` (ATM IV delle 4 expiry vicine + IV interpolata in varianza totale a tenor costante 30h = forecast_horizon del modello vol), `dvol.parquet` (controllo 30d). Scopo: accumulare lo storico IV short-tenor (non gratis altrove) per il gate futuro **NN-RV vs IV implicita**. Avvio persistente (non è un servizio: va rilanciato dopo un riavvio): `Start-Process python -ArgumentList "scripts/01c_iv_poller.py" -WorkingDirectory E:\quantsys_project -WindowStyle Hidden -RedirectStandardError logs\iv_poller.log -RedirectStandardOutput logs\iv_poller.out.log`.

**EN** `python scripts/01c_iv_poller.py` (loop, default 10 min; `--minutes N` for cadence, `--once` smoke, `--backfill-dvol` hourly DVOL history 2021→today) — 2 public Deribit requests per tick, NO account required. Atomic append-only output under `data/iv/`: `chain/btc_options_YYYYMMDD.parquet` (raw snapshot, ~950 instruments/tick), `atm_30h.parquet` (ATM IV of the 4 nearest expiries + total-variance-interpolated IV at constant 30h tenor = the vol model's forecast_horizon), `dvol.parquet` (30d control). Purpose: accumulate short-tenor IV history (not free elsewhere) for the future **NN-RV vs implied IV** gate. Persistent launch (not a service: must be relaunched after a reboot): `Start-Process python -ArgumentList "scripts/01c_iv_poller.py" -WorkingDirectory E:\quantsys_project -WindowStyle Hidden -RedirectStandardError logs\iv_poller.log -RedirectStandardOutput logs\iv_poller.out.log`.

---

## Recorder order-book L2 Binance (B1 microstruttura) · Binance L2 order-book recorder (B1 microstructure)

🇮🇹 `python scripts/01d_orderbook_recorder.py` (loop, default 5s; `--seconds N` cadenza, `--once` smoke, `--symbol` default BTCUSDT, `--levels` profondità REST default 1000) — 1 sola richiesta pubblica Binance `/api/v3/depth` per tick (no auth, weight 50/call → a 5s = 600 weight/min ≪ 1200). Strada **B1**: raccolta FORWARD della microstruttura del book come fonte di informazione NUOVA per un edge direzionale a 1m (le 104 feature OHLCV sono sature). Lo storico L2 non è gratis (Tardis) → si costruisce da qui in avanti. Output append-only atomico in `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/giorno, dedup su `timestamp`): feature microstrutturali derivate (mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, depth cumulata in bande 5/10/25/50 bps, total qty, **OFI best-level** cross-tick Cont-Kukanov-Stoikov) + **top-25 livelli raw/lato** come list-column (rete di sicurezza per ri-derivare feature future senza ri-raccogliere). ⚠ L'`ofi_best` è NaN al 1° tick di ogni processo (richiede lo snapshot precedente) e in `--once` (stato per-processo). Avvio persistente (non è un servizio): `Start-Process -FilePath E:\quantsys_project\.venv\Scripts\python.exe -ArgumentList "scripts/01d_orderbook_recorder.py" -WorkingDirectory E:\quantsys_project -WindowStyle Hidden`.

**EN** `python scripts/01d_orderbook_recorder.py` (loop, default 5s; `--seconds N` cadence, `--once` smoke, `--symbol` default BTCUSDT, `--levels` REST depth default 1000) — 1 public Binance `/api/v3/depth` request per tick (no auth, weight 50/call → at 5s = 600 weight/min ≪ 1200). Track **B1**: FORWARD collection of book microstructure as a genuinely NEW information source for 1m directional edge (the 104 OHLCV features are saturated). L2 history is not free (Tardis) → built from now on. Atomic append-only output under `data/orderbook/l2_features_YYYYMMDD.parquet` (1 file/day, dedup on `timestamp`): derived microstructure features (mid, microprice + tilt bps, spread_bps, imbalance L1/5/10/20, cumulative depth in 5/10/25/50 bps bands, total qty, **best-level OFI** cross-tick Cont-Kukanov-Stoikov) + **top-25 raw levels/side** as list-columns (safety net to re-derive future features without re-collecting). ⚠ `ofi_best` is NaN on the 1st tick of each process (needs the previous snapshot) and in `--once` (per-process state). Persistent launch (not a service): `Start-Process -FilePath E:\quantsys_project\.venv\Scripts\python.exe -ArgumentList "scripts/01d_orderbook_recorder.py" -WorkingDirectory E:\quantsys_project -WindowStyle Hidden`.

---

## Forward test vol-paper (NN-RV vs IV, testnet Deribit) · Vol-paper forward test (NN-RV vs IV, Deribit testnet)

🇮🇹 `python scripts/04b_vol_paper.py` (loop orario a hh:00+90s; `--once` smoke, `--execute` per ordini REALI sul testnet — default fill SIMULATI al mark price; `--arch` per la dir modelli, default `itransformer` = comportamento storico invariato — stesso flag in `dev_vols_qlike.py`). Pre-registrato in `STATUS.md` 2026-06-12: forecast NN-RV 30h (modello vol-1h PASS, inversione completa `μ·IQR+centro`, feature dal path parity-blessed, macro dal parquet con refit identico del normalizer) vs varianza implicita dal poller IV (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; |edge|>0.25 → straddle ATM daily ~30h LONG/SHORT, max 1 posizione, hold a scadenza (cash settlement). Richiede: poller IV attivo, key in `secrets.yaml` blocco `deribit_testnet:` (l'URL DEVE essere test.deribit.com — assert anti-mainnet). Output: `results/vol_paper/{forecasts.parquet,trades.jsonl,position.json}` — il log forecasts si scrive anche quando flat (serve alle baseline always-long/short). Avvio persistente: stesso pattern `Start-Process` del poller, log `logs/vol_paper.log`. ⚠ NON girare training GPU in parallelo senza fermare il processo (5 modelli CUDA residenti).

**EN** `python scripts/04b_vol_paper.py` (hourly loop at hh:00+90s; `--once` smoke, `--execute` for REAL testnet orders — default SIMULATED mark-price fills; `--arch` selects the model dir, default `itransformer` = historical behavior unchanged — same flag in `dev_vols_qlike.py`). Pre-registered in `STATUS.md` 2026-06-12: 30h NN-RV forecast (PASS vol-1h model, full `μ·IQR+center` inversion, parity-blessed feature path, macro from the parquet with identical normalizer refit) vs implied variance from the IV poller (staleness ≤30 min) → `edge = log(RV_pred/var_iv)`; |edge|>0.25 → ~30h daily ATM straddle LONG/SHORT, max 1 position, hold to expiry (cash settlement). Requires: IV poller running, keys in `secrets.yaml` `deribit_testnet:` block (URL MUST be test.deribit.com — anti-mainnet assert). Output: `results/vol_paper/{forecasts.parquet,trades.jsonl,position.json}` — the forecasts log is written even when flat (feeds the always-long/short baselines). Persistent launch: same `Start-Process` pattern as the poller, log `logs/vol_paper.log`. ⚠ Do NOT run GPU training in parallel without stopping the process (5 CUDA-resident models).

🇮🇹 **Baseline del gate** — `python scripts/04c_vol_paper_baselines.py` (solo lettura, GPU-free; `--no-fetch` = solo cache delivery, `--min-trades N` = soglia di valutabilità, default 30). Calcola il gate (2) pre-registrato: il P&L del NN deve battere ENTRAMBE le baseline always-long-vol e always-short-vol sullo STESSO calendario di expiry (isola il timing dal variance risk premium medio). Metodo: replay fedele del loop di `04b` sul log `forecasts.parquet`, premio dello straddle ricostruito dai chain snapshot (`data/iv/chain/*.parquet`, stessa selezione di `pick_straddle`), delivery price dall'endpoint pubblico Deribit (cache `delivery_cache.json`). I gate (1) P&L medio>0 e (3) hit-rate>0.5 si leggono dai trade REALI in `trades.jsonl`; un cross-check NN-ricostruito vs NN-reale valida la ricostruzione del premio. Output: `results/vol_paper/baseline_report.json`. Lo script gira anche con pochi trade (scrive il report + warning "non valutabile" finché n<min-trades): l'harness è pronto, la valutazione del gate matura col tempo-calendario del poller.

**EN** **Gate baselines** — `python scripts/04c_vol_paper_baselines.py` (read-only, GPU-free; `--no-fetch` = delivery cache only, `--min-trades N` = evaluability threshold, default 30). Computes the pre-registered gate (2): the NN P&L must beat BOTH the always-long-vol and always-short-vol baselines over the SAME expiry calendar (isolates timing from the average variance risk premium). Method: faithful replay of the `04b` loop over the `forecasts.parquet` log, straddle premium reconstructed from chain snapshots (`data/iv/chain/*.parquet`, same selection as `pick_straddle`), delivery price from the public Deribit endpoint (`delivery_cache.json`). Gates (1) mean P&L>0 and (3) hit-rate>0.5 are read from the REAL trades in `trades.jsonl`; a reconstructed-NN vs real-NN cross-check validates the premium reconstruction. Output: `results/vol_paper/baseline_report.json`. The script runs even with few trades (writes the report + a "not evaluable" warning while n<min-trades): the harness is ready, the gate evaluation matures with the poller's calendar time.

---

## Hardware

### CPU

🇮🇹 `config/default.yaml`:
```yaml
hardware:
  cpu_fraction: 0.5   # 0.3=30%, 0.5=50%, 0.8=80%
```
Default 0.5 (4 thread su 8 core). Letto da tutti gli script all'avvio.

**EN** `config/default.yaml`:
```yaml
hardware:
  cpu_fraction: 0.5   # 0.3=30%, 0.5=50%, 0.8=80%
```
Default 0.5 (4 threads on 8 cores). Read by every script at startup.

### GPU compute

```powershell
nvidia-smi -pl 125    # limita (RTX 2070 Super min=125 max=215W)
nvidia-smi -pl 215    # ripristina
```

### Setup di riferimento (RTX 2070 Super 8GB) · Reference setup (RTX 2070 Super 8GB)

🇮🇹
| Componente | Valore |
|---|---|
| CUDA, AMP fp16 training | sì (via `setup_device`) |
| AMP inference | **off** hardcoded in `ensemble.py:170` (evita NaN spectral_norm + Mamba scan) |
| Batch inference backtest | 256 (`scripts/03_backtest.py`) |
| Batch training | 64 (default `config/arch/<arch>.yaml`) |

**EN**
| Component | Value |
|---|---|
| CUDA, AMP fp16 training | yes (via `setup_device`) |
| AMP inference | **off** hardcoded in `ensemble.py:170` (avoids NaN from spectral_norm + Mamba scan) |
| Backtest inference batch | 256 (`scripts/03_backtest.py`) |
| Training batch | 64 (default `config/arch/<arch>.yaml`) |

### Solo CPU · CPU only

🇮🇹 Il codice fa fallback automatico via `setup_device` (`quantsys/utils/__init__.py`). Su `quantsys/model/__init__.py:67`, `autocast(device_type="cuda")` è no-op silenzioso su CPU. Tempi:
- Training: 20-50× più lento (tcnmamba ~3h GPU → 2-3 giorni CPU). Sconsigliato.
- Backtest: ~5s GPU → 30-60s CPU. Tollerabile.
- Live: ~50-100ms vs ~20ms GPU. Pienamente utilizzabile (latency WS Binance domina).

**EN** The code falls back automatically via `setup_device` (`quantsys/utils/__init__.py`). In `quantsys/model/__init__.py:67`, `autocast(device_type="cuda")` is a silent no-op on CPU. Times:
- Training: 20–50× slower (tcnmamba ~3h GPU → 2–3 days CPU). Not recommended.
- Backtest: ~5s GPU → 30–60s CPU. Tolerable.
- Live: ~50–100ms vs ~20ms GPU. Fully usable (Binance WS latency dominates).

### Apple Silicon / AMD / Intel Arc

🇮🇹 Non testato. Codice usa `torch.cuda.*`. Per MPS servono modifiche a `setup_device` e probabilmente kernel custom per Mamba/SSM.

**EN** Untested. Code uses `torch.cuda.*`. MPS support would require changes to `setup_device` and likely custom kernels for Mamba/SSM.

### Poca VRAM (4GB) · Low VRAM (4GB)

🇮🇹 `config/arch/<arch>.yaml`:
```yaml
batch_size: 32
gradient_accumulation_steps: 2   # mantiene effective batch=64
```
Inference batch in `scripts/03_backtest.py` da 256 → 128.

**EN** `config/arch/<arch>.yaml`:
```yaml
batch_size: 32
gradient_accumulation_steps: 2   # keeps effective batch=64
```
Inference batch in `scripts/03_backtest.py` from 256 → 128.

### Molta VRAM (≥16GB) · High VRAM (≥16GB)

```yaml
batch_size: 128
```
Inference batch fino a 1024 (guadagno marginale, GPU già satura).

---

## Architetture · Architectures

🇮🇹
| Arch | Classe | File | Note |
|---|---|---|---|
| `itransformer` | `QuantiTransformer` | `quantsys/model/__init__.py:1025` | Attention sulle feature, baseline |
| `nhits` | `QuantNHiTS` | `quantsys/model/nhits.py:110` | Pure-MLP gerarchico |
| `tcnmamba` | `QuantTCNMamba` | `quantsys/model/tcn_mamba.py:341` | TCN dilatate + SSM ibrido |
| `lstm` | `QuantLSTM` | `quantsys/model/__init__.py:309` | Legacy |

**EN**
| Arch | Class | File | Notes |
|---|---|---|---|
| `itransformer` | `QuantiTransformer` | `quantsys/model/__init__.py:1025` | Attention over features, baseline |
| `nhits` | `QuantNHiTS` | `quantsys/model/nhits.py:110` | Hierarchical pure-MLP |
| `tcnmamba` | `QuantTCNMamba` | `quantsys/model/tcn_mamba.py:341` | Dilated TCN + SSM hybrid |
| `lstm` | `QuantLSTM` | `quantsys/model/__init__.py:309` | Legacy |

### Aggiungere una nuova arch · Adding a new architecture

🇮🇹
1. Classe in `quantsys/model/` con `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher in `quantsys/model/__init__.py:load_model`
3. Branch in `scripts/02_train.py` (`architecture == "X"`)
4. `config/arch/X.yaml`
5. `choices` in `run_all.py` (parser `--arch` e `--teacher`)
6. Whitelist in `06_dashboard.py`, `05_analyze_signals.py`
7. (Opzionale) `distillation.archs` in `config/default.yaml`

**EN**
1. Class in `quantsys/model/` with `forward(x, x_macro=None) -> (mu, ls2, lnu)`
2. Dispatcher in `quantsys/model/__init__.py:load_model`
3. Branch in `scripts/02_train.py` (`architecture == "X"`)
4. `config/arch/X.yaml`
5. `choices` in `run_all.py` (parser `--arch` and `--teacher`)
6. Whitelists in `06_dashboard.py`, `05_analyze_signals.py`
7. (Optional) `distillation.archs` in `config/default.yaml`

---

## Optuna

```bash
set QUANTSYS_ARCH=lstm
python scripts/02c_optuna_search.py --n-trials 50 --study-name quantsys
```
**Limiti**: hardcoded su `QuantLSTM`. `best_params.json` salvato in `models/lstm/` NON applicato automaticamente al training successivo — copia manuale in `config/arch/lstm.yaml`.

🇮🇹 Studio persistente su SQLite (`models/lstm/optuna_quantsys.db`), ripristinabile.

**EN** Study persists on SQLite (`models/lstm/optuna_quantsys.db`), resumable any time.

---

## Ensemble omogeneo (5× stessa arch, legacy) · Homogeneous ensemble (5× same arch, legacy)

🇮🇹 `config/default.yaml`:
```yaml
training:
  n_ensemble: 5   # default attuale = 5 (in --distill il default è 1; override esplicito con --n-ensemble)
```
Output: `models/{arch}/best_model_0..4.pt`. Backtest/live li caricano via `EnsembleModel.load`. Indipendente dalla distillation (modalità non si escludono).

**EN** `config/default.yaml`:
```yaml
training:
  n_ensemble: 5   # current default = 5 (--distill defaults to 1; explicit --n-ensemble overrides)
```
Output: `models/{arch}/best_model_0..4.pt`. Backtest/live load them via `EnsembleModel.load`. Independent from distillation (modes are not mutually exclusive).

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
│   ├── raw_candles.parquet      # OHLCV storico (1h 2019→oggi dal pivot 2026-06-09)
│   ├── features.parquet         # feature normalizzate
│   ├── lstm_dataset.npz         # windows X/y per training
│   ├── funding_rate.parquet     # funding futures (8h, completo dal 2019-09-10)
│   ├── macro_*.parquet          # FRED/yFinance
│   ├── iv/                      # IV Deribit: chain/ (snapshot raw), atm_30h.parquet, dvol.parquet — UNICO dato non rigenerabile
│   └── backup_1m/               # raw_candles + regime_probs era-1m (rollback 1m = restore + retrain)
├── models/
│   ├── pipeline_state.pkl       # copia canonica (scritta da 01, guard anti-stale in 02)
│   ├── teacher_analysis.json    # output 07_verify_teacher.py
│   ├── backup_1h_vols/          # vol-1h PASS autosufficiente (5 membri + state + raw/regime 1h)
│   ├── backup_1m_vols/          # vol-1m FAIL (record)
│   └── {arch}/
│       ├── best_model.pt        # checkpoint
│       ├── config.json          # iperparametri + flag distilled/teacher_arch
│       ├── history.json         # curva loss
│       └── pipeline_state.pkl   # scalers + feature config
├── results/{arch}/
│   ├── dashboard_results.json
│   └── live_signals.jsonl
├── tests/                       # pytest (test_recent_fixes.py: regression fix critici)
├── scripts/                     # 00-07 numerati + 99_replay + dev_vols_* (giudici vol attivi)
│   └── archive/                 # probe chiusi: xs_01/02/03 (KILL), dev_step0_regime_sigma
└── logs/quantsys_YYYYMMDD_HHMMSS.log
```

---

## Dashboard — pulsante "Aggiorna" · Dashboard — "Update" button

🇮🇹
1. **Aggiorna** in alto a destra
2. Seleziona step da eseguire
3. **Avvia** → barra di progresso
4. Al termine dashboard si aggiorna automaticamente
5. **Annulla** per fermare un job

**EN**
1. Click **Update** top right
2. Select the steps to run
3. Click **Start** → progress bar
4. When done the dashboard refreshes automatically
5. **Cancel** to stop a job

🇮🇹 Switch arch: dropdown in alto (rileva arch con `dashboard_results.json`).

**EN** Switching arch: top dropdown (detects archs with `dashboard_results.json`).

---

## Fermare tutto · Stopping everything

🇮🇹 `Ctrl+C` nel terminale di `run_all.py`. Termina dashboard + live feed.

**EN** `Ctrl+C` in the `run_all.py` terminal. Terminates dashboard + live feed.
