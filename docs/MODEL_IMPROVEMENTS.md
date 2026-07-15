# QUANTSYS — Miglioramenti modello · QUANTSYS — Model improvements

🇮🇹 Tracker dei miglioramenti al motore neural-forecasting BTC/USDT, riorganizzato per fase del ciclo ML: **Panoramica → Dati/Feature → Modellazione → Valutazione → Storico esperimenti**. Lo "stato corrente" (cosa gira in produzione, cosa è aperto) vive in `STATUS.md`; le lezioni long-term e i kill sono qui per non re-testarli. Doc bilingue single-file (marker 🇮🇹 / **EN**); allineato a `CLAUDE.md` come single source of truth.

**EN** Tracker for improvements to the BTC/USDT neural-forecasting engine, reorganized by ML lifecycle phase: **Overview → Data/Features → Modeling → Evaluation → Experiment log**. The "current state" (what runs in production, what is open) lives in `STATUS.md`; long-term lessons and kills live here so we don't re-test them. Bilingual single-file doc (markers 🇮🇹 / **EN**); aligned to `CLAUDE.md` as the single source of truth.

---

## 1. Panoramica · Overview

🇮🇹 **Linea di produzione corrente = volatilità @ 1h** (`features.target_type: log_rv`, Vol-S PASS 2026-06-10). L'infrastruttura interval-agnostica (sotto, §2) nata per il pivot direzionale 1m→1h è il fondamento di questa linea: config/dataset/modelli su disco sono 1h, ma il target è il log-RV, non la direzione. Il filone **direzionale** (1m e 1h) è **KILLED OOS** e conservato come record metodologico (§5).

**EN** **Current production line = volatility @ 1h** (`features.target_type: log_rv`, Vol-S PASS 2026-06-10). The interval-agnostic infrastructure (below, §2), born for the directional 1m→1h pivot, is the foundation of this line: on-disk config/dataset/models are 1h, but the target is log-RV, not direction. The **directional** thread (both 1m and 1h) is **KILLED OOS** and kept as a methodological record (§5).

🇮🇹 **Stato sintetico per asse** (dettaglio numerico in §4-§5 e `STATUS.md`):

**EN** **Per-axis summary** (numeric detail in §4-§5 and `STATUS.md`):

🇮🇹
| Asse · Axis | Esito · Outcome | Nota |
|---|---|---|
| Cross-sectional multi-asset | ⚫ KILL 2026-06-06 | muro = magnitudine (~1.5 bps effetto vs ~26 bps costo) |
| Timeframe direzionale → 1h | ⚫ KILL 2026-06-10 | muro costi sfondato, zero skill OOS, anti-corr val→test |
| Target → volatilità (`log_rv`) | 🟢 PASS 2026-06-10 | NN batte HAR-RV del 30% in QLIKE, val→test coerenti |
| Semivarianza firmata (`log_rs_ratio`) | ⚫ FAIL 2026-06-11 | asimmetria impredicibile per tutti (momenti dispari) |
| Asset class → ES 1m | ⊘ non avviato | already HFT-arbitraged; roll leakage |

**EN**
| Axis | Outcome | Note |
|---|---|---|
| Cross-sectional multi-asset | ⚫ KILL 2026-06-06 | wall = magnitude (~1.5 bps effect vs ~26 bps cost) |
| Directional timeframe → 1h | ⚫ KILL 2026-06-10 | cost wall broken, zero OOS skill, val→test anti-corr |
| Target → volatility (`log_rv`) | 🟢 PASS 2026-06-10 | NN beats HAR-RV by 30% in QLIKE, val→test consistent |
| Signed semivariance (`log_rs_ratio`) | ⚫ FAIL 2026-06-11 | asymmetry unpredictable for everyone (odd moments) |
| Asset class → ES 1m | ⊘ not started | already HFT-arbitraged; roll leakage |

🇮🇹 **Assi vivi** (gate da accumulare, vedi `STATUS.md`): monetizzazione vol-1h (RV vs IV — poller Deribit `01c`, opzioni IBIT via Alpaca, forward test straddle `04b`); **B1 order-book L2** (recorder `01d`, raccolta forward). **Lezione di fondo** (§6): l'informazione in price/volume riguarda i **momenti pari** (livello RV: predicibile) e non quelli **dispari** (direzione, segno della varianza: impredicibili).

**EN** **Live axes** (gates to accumulate, see `STATUS.md`): vol-1h monetization (RV vs IV — Deribit poller `01c`, IBIT options via Alpaca, `04b` straddle forward test); **B1 order-book L2** (recorder `01d`, forward collection). **Core lesson** (§6): the information in price/volume concerns the **even moments** (RV level: predictable) and not the **odd ones** (direction, sign of variance: unpredictable).

---

## 2. Dati e feature · Data and features

### 2.1 Infrastruttura interval-agnostica (pivot 1m→1h) · Interval-agnostic infrastructure (1m→1h pivot)

🇮🇹 Implementata 2026-06-09, invariante chiave: **identità a 1m** (path 1m bit-perfect, reversibile). È l'infrastruttura production della linea vol-1h, non un esperimento in corso.

**EN** Implemented 2026-06-09, key invariant: **identity at 1m** (1m path bit-perfect, reversible). It is the production infrastructure of the vol-1h line, not an experiment in progress.

🇮🇹
- **Single source of truth config-side**: `interval_minutes_from_cfg` in `quantsys/utils` (mappa `data.interval` → minuti, `ValueError` fail-fast su intervalli ignoti).
- **`FeatureBuilder(interval_minutes=...)`** con `bars_per_day = 1440 // interval_minutes` e helper `_tbars(minutes)` (floor anti-degenerazione 2 barre).
- **Finestre TIME-semantic convertite** (mantengono il significato in TEMPO): strutturali 30d/90d/365d (`days × bars_per_day`), momentum 7d/30d/90d, `funding_rate_1d`, `session_position` (240 min), `price_vs_ma200m` (200 min).
- **Finestre BAR-semantic deliberatamente invariate** (si traslano col timeframe): `features.windows` [5,10,20,60], CVD, vwap, VP scales 60/240/1440 **BARRE**, lags.
- **Inference-side l'interval si deriva da `PipelineState.interval`** (property `interval_minutes`, fallback 1 per pkl legacy), MAI dalla config. Guard `RuntimeError` "interval mismatch" in `03_backtest.py` e `04_live_signals.py` (stesso pattern del guard `forecast_horizon`): modello-1m + config-1h = combinazione bloccata.
- **Annualizzazione interval-aware**: `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760) in `bootstrap_sharpe_ci` e `RiskManager`. σ safety-net scalata `0.05·√interval_minutes` (1h→≈0.387): cattura il bug denorm z→raw ~30-100×, non la crescita √60 legittima.

**EN**
- **Config-side single source of truth**: `interval_minutes_from_cfg` in `quantsys/utils` (maps `data.interval` → minutes, fail-fast `ValueError` on unknown intervals).
- **`FeatureBuilder(interval_minutes=...)`** with `bars_per_day = 1440 // interval_minutes` and helper `_tbars(minutes)` (anti-degeneration floor of 2 bars).
- **TIME-semantic windows converted** (keep meaning in TIME): structural 30d/90d/365d (`days × bars_per_day`), momentum 7d/30d/90d, `funding_rate_1d`, `session_position` (240 min), `price_vs_ma200m` (200 min).
- **BAR-semantic windows deliberately unchanged** (they shift with the timeframe): `features.windows` [5,10,20,60], CVD, vwap, VP scales 60/240/1440 **BARS**, lags.
- **Inference-side the interval derives from `PipelineState.interval`** (`interval_minutes` property, fallback 1 for legacy pkl), NEVER from config. `RuntimeError` "interval mismatch" guard in `03_backtest.py` and `04_live_signals.py` (same pattern as the `forecast_horizon` guard): 1m-model + 1h-config = blocked.
- **Interval-aware annualization**: `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760) in `bootstrap_sharpe_ci` and `RiskManager`. Scaled σ safety net `0.05·√interval_minutes` (1h→≈0.387): catches the ~30-100× z→raw denorm bug, not the legitimate √60 growth.

🇮🇹 **Overlay `config/interval/` (2026-06-10):** le chiavi interval-dipendenti sono fattorizzate in `config/interval/{1m,1h}.yaml`, mergiate da `load_config` per-sezione shallow **dopo secrets e prima dell'overlay arch** (l'arch resta l'override più specifico). Selezione via `QUANTSYS_INTERVAL` o `python run_all.py --interval 1m|1h` (propagata a tutti i subprocess, incluso `--distill`). File mancante → warning, prosegue col solo default.yaml.

**EN** **`config/interval/` overlay (2026-06-10):** interval-dependent keys are factored into `config/interval/{1m,1h}.yaml`, merged by `load_config` per-section shallow **after secrets and before the arch overlay** (arch stays the most specific override). Selected via `QUANTSYS_INTERVAL` or `python run_all.py --interval 1m|1h` (propagated to all subprocesses, including `--distill`). Missing file → warning, then default.yaml only.

🇮🇹
| Parametro · Parameter | Valore 1h · 1h value | Era 1m · Was 1m |
|---|---|---|
| `interval` | `1h` | `1m` |
| `start_time` | `2019-01-01` | `2025-05-19` |
| `window_stride` | 1 | 5 |
| `embargo_steps` | 168 | 1500 |
| `max_hold_candles` | 60 (≥ h=30) | 240 |
| `min_expected_ret` | 0.0013 (13 bps; 2° test 23 bps) | 0.0005 |
| `max_sigma` | 0.10 (≈0.015·√60, da ricalibrare) | 0.015 |
| `forecast_horizon` | 30 INVARIATO — ora 30 ORE · now 30 HOURS | 30 (= 30 min) |
| `window_size` | 120 INVARIATO — ora 5 giorni · now 5 days | 120 (= 2h) |

🇮🇹 **Rollback 1m:** basta `--interval 1m` (overlay 1m.yaml) + restore `data/backup_1m/*` — tutte le conversioni sono identità a 1m, il codice non va toccato. ⚠ I checkpoint `models/backup_1m/` sono stati **eliminati** col cleanup 2026-06-12: il rollback 1m ora richiede retrain. **TODO GJR-GARCH: CHIUSO 2026-07-15** — parametri ri-stimati su rendimenti 1h (QMLE + variance targeting, `scripts/vol/estimate_gjr_1h.py` → `ω=1.026e-6, α=0.1011, γ=0.0052, β=0.8732`, persistence 0.977; report `results/vols/gjr_params_1h.json`); default.yaml porta i valori 1h, l'overlay 1m conserva i 1m-era (ω 1.2e-5 ecc.) per il rollback; cap σ del MC ora parametrico (`gjr_sigma_cap`).

**EN** **1m rollback:** just `--interval 1m` (1m.yaml overlay) + restore `data/backup_1m/*` — every conversion is identity at 1m, no code changes. ⚠ The `models/backup_1m/` checkpoints were **deleted** in the 2026-06-12 cleanup: 1m rollback now requires retrain. **GJR-GARCH TODO: CLOSED 2026-07-15** — parameters re-estimated on 1h returns (QMLE + variance targeting, `scripts/vol/estimate_gjr_1h.py` → `ω=1.026e-6, α=0.1011, γ=0.0052, β=0.8732`, persistence 0.977; report `results/vols/gjr_params_1h.json`); default.yaml carries the 1h values, the 1m overlay keeps the 1m-era ones (ω 1.2e-5 etc.) for rollback; the MC σ cap is now parametric (`gjr_sigma_cap`).

### 2.2 Dataset corrente · Current dataset

🇮🇹 `raw_candles.parquet` 1h 2019→oggi (**~65k barre**); funding completo dal lancio perp 2019-09-10 (**~7.4k obs**, re-download). Dataset rigenerato 2026-06-22: `X_train (51364, 120, 104)`, split 51364/6420/6421, `X_macro_* (·,90)`, target **z-scored su `log_rv`** (`target_scale=1.4343` = IQR del log-RV; il log-ret avrebbe IQR ~1e-3 → conferma che il target è `log_rv`). ⚠ Il dataset npz è gitignored: rigenerare con `01_download_data.py` + `scripts/vol/dev_vols_macro_append.py` prima di train/judge.

**EN** `raw_candles.parquet` is 1h 2019→today (**~65k bars**); full funding since perp launch 2019-09-10 (**~7.4k obs**, re-download). Dataset regenerated 2026-06-22: `X_train (51364, 120, 104)`, split 51364/6420/6421, `X_macro_* (·,90)`, target **z-scored on `log_rv`** (`target_scale=1.4343` = IQR of log-RV; log-ret would have IQR ~1e-3 → confirms the target is `log_rv`). ⚠ The npz dataset is gitignored: regenerate with `01_download_data.py` + `scripts/vol/dev_vols_macro_append.py` before train/judge.

### 2.3 Filtro feature C-funding (104 = 86 dinamiche + 18 strutturali) · C-funding feature filter

🇮🇹 **Decisione 2026-05-28** (da permutation importance, ensemble eterogeneo, 2500 finestre val): le 23 feature "live-hostile" (lookback > buffer rolling live) hanno **ROI ≤ 0** per il modello a h=30 — permutarle in blocco *migliora* leggermente le metriche (DA 0.529→0.532, Spearman 0.069→0.076). Unica eccezione: le feature **funding** (`momentum_x_funding` +0.0042, `funding_rate_dev` +0.0028).

**EN** **Decision 2026-05-28** (from permutation importance, heterogeneous ensemble, 2500 val windows): the 23 "live-hostile" features (lookback > live rolling buffer) have **ROI ≤ 0** for the h=30 model — bulk-permuting them *slightly improves* metrics (DA 0.529→0.532, Spearman 0.069→0.076). Sole exception: the **funding** features (`momentum_x_funding` +0.0042, `funding_rate_dev` +0.0028).

🇮🇹 **Set C-funding** = single source of truth `LIVE_DROP_FEATURES` (`quantsys/features/__init__.py`), filtrato in `01_download_data.py`. Droppa **15 feature** live-incompatibili (90d, 365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`); mantiene 30d + funding (ROI positivo, calcolabili in live). Risultato: **104 feature** (vs 119 pre-filtro). Lo schema "ibrido completo" (tutte le 30/90/365d in live) è stato scartato dai dati (ROI negativo del tier long).

**EN** **C-funding set** = single source of truth `LIVE_DROP_FEATURES` (`quantsys/features/__init__.py`), filtered in `01_download_data.py`. Drops **15** live-incompatible features (90d, 365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`); keeps 30d + funding (positive ROI, live-computable). Result: **104 features** (vs 119 pre-filter). The "full hybrid" scheme (all 30/90/365d live) was discarded by the data (negative long-tier ROI).

### 2.4 Regime detector — `RegimeMarkovBTC` · Regime detector

🇮🇹 **Implementato 2026-06-03** (Variante 3, dopo che il detector macro era degenere: collassava a r0=100% e dava `val_nll spread=0.000`). Markov-Switching Hamilton 1989 su realized volatility BTC oraria (`log_ret_h`+`log_rv`, PCA n_pca=1 expanding, switching mean+variance). **CAUSALE**: `filtered_marginal_probabilities` (NON smoothed) + Hamilton filter forward-only, walk-forward expanding (`macro.hmm_burn_in_days: 30` / `hmm_retrain_days: 90`). 3 regimi data-driven misurati sullo span 1m 2025-26 (da ri-misurare a 1h): **R0 Quiet ~42%** (σ²≈0.56, drift≈0, P(stay)=89%), **R1 Trending ~18%** (σ²≈0.12, drift=+0.08, P(stay)=92%), **R2 Stress ~40%** (σ²≈3.79, drift=−0.12, P(stay)=79%). Persistiti `data/regime_probs.parquet` (index orario UTC). Uso: **stratified val + `val_nll per regime` diagnostico, NON feature di input**. Le classi `RegimeMarkovSwitching` / `RegimeSession` restano nel codice come alternative non-production.

**EN** **Implemented 2026-06-03** (Variant 3, after the macro detector was degenerate: collapsing to r0=100% with `val_nll spread=0.000`). Hamilton-1989 Markov-Switching on BTC hourly realized volatility (`log_ret_h`+`log_rv`, PCA n_pca=1 expanding, switching mean+variance). **CAUSAL**: `filtered_marginal_probabilities` (NOT smoothed) + forward-only Hamilton filter, expanding walk-forward (`macro.hmm_burn_in_days: 30` / `hmm_retrain_days: 90`). 3 data-driven regimes measured on the 1m 2025-26 span (to re-measure at 1h): **R0 Quiet ~42%** (σ²≈0.56, drift≈0, P(stay)=89%), **R1 Trending ~18%** (σ²≈0.12, drift=+0.08, P(stay)=92%), **R2 Stress ~40%** (σ²≈3.79, drift=−0.12, P(stay)=79%). Persisted in `data/regime_probs.parquet` (hourly UTC index). Use: **stratified val + per-regime `val_nll` diagnostic, NOT an input feature**. The `RegimeMarkovSwitching` / `RegimeSession` classes stay in the codebase as non-production alternatives.

🇮🇹 **Validazione (retrain iTransformer 5/5 ensemble):** stratificazione val **46% / 12% / 41%** (vs precedente collasso 100% r0), spread `val_nll` per regime **0.19-0.30** stabile (>> soglia 0.05 "informativo") → il regime è effettivamente informativo. Memoria: `session_2026_06_03_markov_btc.md`.

**EN** **Validation (iTransformer 5/5 ensemble retrain):** val stratification **46% / 12% / 41%** (vs previous 100% r0 collapse), per-regime `val_nll` spread **0.19-0.30** stable (>> 0.05 "informative" threshold) → regime is effectively informative. Memory: `session_2026_06_03_markov_btc.md`.

---

## 3. Modellazione · Modeling

### 3.1 Distillation TARGET-AWARE · TARGET-AWARE distillation

🇮🇹 **DONE, su disco** (commit `73fef66`). `teacher_score_weights(target_type)` in `quantsys/model/distillation.py` = single source of truth dei pesi di scoring teacher, usata da `_select_best_teacher` + `compute_teacher_weights`:
- `ret` (direzionale) → `0.40 val_loss + 0.35 spearman + 0.25 dir_acc` (il segno È il segnale);
- **`log_rv` (vol) → `0.65 val_loss + 0.35 spearman + 0.00 dir_acc`** (la dir_acc della varianza = segno-vs-mediana, NON tradabile: lo straddle è direction-neutral; il momento PARI val_loss/QLIKE è ciò che generalizza OOS).

Metriche di val alla best epoch (`best_val_loss`/`best_spearman`/`best_da`) persistite in `config.json` (senza, il blend ricadeva su pesi uniformi). → softmax(T=2) → soft labels μ/ls²/lnu = media pesata di TUTTI gli archi → student `(1−α)·NLL + α·distill` (α=0.3, scale-normalized) + 60% epoche.

**EN** **DONE, on disk** (commit `73fef66`). `teacher_score_weights(target_type)` in `quantsys/model/distillation.py` = single source of truth for teacher-scoring weights, used by `_select_best_teacher` + `compute_teacher_weights`:
- `ret` (directional) → `0.40 val_loss + 0.35 spearman + 0.25 dir_acc` (the sign IS the signal);
- **`log_rv` (vol) → `0.65 val_loss + 0.35 spearman + 0.00 dir_acc`** (variance dir_acc = sign-vs-median, NOT tradable: the straddle is direction-neutral; the EVEN moment val_loss/QLIKE is what generalizes OOS).

Best-epoch val metrics (`best_val_loss`/`best_spearman`/`best_da`) persisted in `config.json` (without them the blend fell back to uniform weights). → softmax(T=2) → soft labels μ/ls²/lnu = weighted mean of ALL archs → student `(1−α)·NLL + α·distill` (α=0.3, scale-normalized) + 60% epochs.

### 3.2 Infra linea-vol · Vol-line infra (DONE, on disk)

🇮🇹
- **`models_root()` + env `QUANTSYS_MODELS_ROOT`** (`quantsys/utils/__init__.py`, propagato a `distillation.py`/`ensemble.py`/`run_all.py`/`02_train.py`/`dev_vols_qlike.py`): redirige TUTTE le read/write modelli su una root sandbox (default `models/`), isola un esperimento distill/k-fold dal modello LIVE di `04b`. Default byte-identico al comportamento precedente.
- **`quantsys/model/vol_metrics.py`**: helper condivisi QLIKE / inversione log-RV (centro+scala dal RobustScaler) / baseline HAR — usati da `dev_vols_qlike.py`, `02b` (fold-metric QLIKE), `step0_xarch_corr.py`.
- **CAFN coordinatore causale** (`quantsys/model/cafn.py` + `scripts/02d_cafn_joint_train.py`): probe coordinatore, **inerte** (non sul path production), da validare con gate pre-registrato.
- **Inversione completa del target vol:** con `target_type: log_rv` `denormalize_predictions` (solo μ·scale) è insufficiente (mediana log-RV ≈ −7.2): l'inversione corretta è `μ·IQR + centro` dal RobustScaler persistito (vedi `vol_metrics.py` / `TEORIA.md` §2).

**EN**
- **`models_root()` + env `QUANTSYS_MODELS_ROOT`** (`quantsys/utils/__init__.py`, propagated to `distillation.py`/`ensemble.py`/`run_all.py`/`02_train.py`/`dev_vols_qlike.py`): redirects ALL model reads/writes to a sandbox root (default `models/`), isolating a distill/k-fold experiment from `04b`'s LIVE model. Default byte-identical to prior behavior.
- **`quantsys/model/vol_metrics.py`**: shared QLIKE / log-RV inversion (center+scale from RobustScaler) / HAR baseline helpers — used by `dev_vols_qlike.py`, `02b` (fold-metric QLIKE), `step0_xarch_corr.py`.
- **CAFN causal coordinator** (`quantsys/model/cafn.py` + `scripts/02d_cafn_joint_train.py`): coordinator probe, **inert** (not on the production path), to validate with a pre-registered gate.
- **Full vol-target inversion:** with `target_type: log_rv`, `denormalize_predictions` (μ·scale only) is insufficient (log-RV median ≈ −7.2): the correct inversion is `μ·IQR + center` from the persisted RobustScaler (see `vol_metrics.py` / `TEORIA.md` §2).

### 3.3 Live engine — parity feature↔training (BLOCKER #1) · Live engine — feature parity

🇮🇹 **RISOLTO 2026-06-05 (Stage 1-5 DONE).** Problema originario: il live engine costruiva a mano **39 feature** in ordine diverso, con normalizzazione per-window (non il `RobustScaler` del `pipeline_state`) e pad/truncate posizionale cieco → input scorrelati dal training. **Decisione architetturale:** riusare direttamente `FeatureBuilder.build()` sul buffer live (single source of truth by-design).

**EN** **RESOLVED 2026-06-05 (Stages 1-5 DONE).** Original problem: the live engine hand-built **39 features** in a different order, with per-window normalization (not the `pipeline_state`'s `RobustScaler`) and blind positional pad/truncate → inputs uncorrelated from training. **Architectural decision:** directly reuse `FeatureBuilder.build()` on the live buffer (single source of truth by-design).

🇮🇹 **Path di produzione:** `LiveCandleBuffer`(50k, ring, bootstrap da `raw_candles.parquet`) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canoniche, scaler da `PipelineState`, hard-fail su mismatch nomi) → finestra `[-120:, :]` → `LiveEngine._deterministic_predict` (nucleo deterministico condiviso col backtest; l'`EnsembleModel` non espone `predict_with_uncertainty`, quindi MC-dropout non scatta) → `denormalize_predictions` → `SignalGenerator`. Il vecchio `LiveFeatureBuffer` (39 feat) resta SOLO come utility ATR/sanity. `get_canonical_feature_names(npz_path)` esposto in `quantsys/features/__init__.py` (lru_cache).

**EN** **Production path:** `LiveCandleBuffer`(50k, ring, bootstrap from `raw_candles.parquet`) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canonical, scaler from `PipelineState`, hard-fail on name mismatch) → window `[-120:, :]` → `LiveEngine._deterministic_predict` (deterministic core shared with the backtest; `EnsembleModel` lacks `predict_with_uncertainty`, so MC-dropout never fires) → `denormalize_predictions` → `SignalGenerator`. The legacy `LiveFeatureBuffer` (39 feat) stays ONLY as an ATR/sanity utility. `get_canonical_feature_names(npz_path)` exposed in `quantsys/features/__init__.py` (lru_cache).

🇮🇹 **Gate go/no-go (DONE):** Gate 1 parity FEATURE `max|Δ|=0.000e+00`; Gate 2 parity SEGNALE `Δμ=0, Δσ=0, side identico` (`tests/test_live_training_parity.py` 5/5, replay `scripts/99_replay_live_vs_training.py`, recent-fixes 16/16). **Residuo operativo (non di codice):** smoke test WS Binance reale + paper-trading per accumulare trade OOS. ⚠ I segnali paper riflettono il backtest, che è negativo OOS sul direzionale (edge soglia/rank esaurito): nessuna aspettativa di Sharpe>0 a priori.

**EN** **Go/no-go gate (DONE):** Gate 1 FEATURE parity `max|Δ|=0.000e+00`; Gate 2 SIGNAL parity `Δμ=0, Δσ=0, identical side` (`tests/test_live_training_parity.py` 5/5, replay `scripts/99_replay_live_vs_training.py`, recent-fixes 16/16). **Operational remainder (not code):** real Binance WS smoke test + paper-trading to accumulate OOS trades. ⚠ Paper signals reflect the backtest, which is negative OOS on the directional target (threshold/rank edge exhausted): no a-priori expectation of Sharpe>0.

### 3.4 Roadmap modello (legacy-direzionale, SUPERATA) · Model roadmap (legacy-directional, SUPERSEDED)

🇮🇹 > ⚠ Questa roadmap nasceva per spingere la **directional accuracy** della linea direzionale-1m (KILLED OOS: anti-corr val→test strutturale; pivot 1h KILL 2026-06-10). **Niente di questo è sul critical path della linea vol-1h.** Conservata come record. Tutto gated post paper-trading direzionale con Sharpe>0 mai raggiunto.

**EN** > ⚠ This roadmap was meant to push **directional accuracy** of the directional-1m line (KILLED OOS: structural val→test anti-correlation; 1h pivot KILL 2026-06-10). **None of this is on the vol-1h critical path.** Kept as a record. All gated behind a directional paper-trading Sharpe>0 never reached.

🇮🇹
| # | Fix | Da · From → A · To | Esito · Outcome |
|---|---|---|---|
| 3 | `window_size` (T) | 120 → 240 | ⚫ **REGREDITO** sul dataset 1m (Spearman wf 0.065→0.034, WHR 0.504; T=180/240 degrado monotono). Causa: T=240 collassa la varianza di μ_pred (3 trade vs 12); il dataset 1m non ha profondità per i param aggiuntivi. Rollback T=120 eseguito 2026-06-04. |
| 4 | `validation.n_folds` | 3 → 6 | ✅ applicato (`n_folds=6` → 5 fold effettivi, fold 0 scartato strutturalmente). |
| 5 | Multi-timeframe (1m+5m+1h) | nuovo pkg `mtf/` | ⊘ **MAI implementato** (effort 6-9 settimane, gated dietro Sharpe live>0.5; package `mtf/` non esiste su disco). Dettaglio progettuale in §5.4. |
| 6 | `mamba-ssm` kernel CUDA | pure-PyTorch → kernel ufficiale | ⊘ aperto. **Unica voce target-agnostica ancora utile** (speedup branch Mamba +3-5×). Dettaglio in §5.5. |

**EN**
| # | Fix | From → To | Outcome |
|---|---|---|---|
| 3 | `window_size` (T) | 120 → 240 | ⚫ **REGRESSED** on the 1m dataset (wf Spearman 0.065→0.034, WHR 0.504; T=180/240 monotonic degradation). Cause: T=240 collapses μ_pred variance (3 trades vs 12); the 1m dataset lacks depth for the extra params. T=120 rollback executed 2026-06-04. |
| 4 | `validation.n_folds` | 3 → 6 | ✅ applied (`n_folds=6` → 5 effective folds, fold 0 structurally dropped). |
| 5 | Multi-timeframe (1m+5m+1h) | new `mtf/` pkg | ⊘ **NEVER implemented** (6-9 weeks effort, gated behind live Sharpe>0.5; the `mtf/` package does not exist on disk). Design detail in §5.4. |
| 6 | `mamba-ssm` CUDA kernel | pure-PyTorch → official kernel | ⊘ open. **Only target-agnostic item still useful** (Mamba-branch speedup +3-5×). Detail in §5.5. |

### 3.5 Execution layer / Binance Futures Testnet (NON implementato) · Execution layer (NOT implemented)

🇮🇹 > ⚠ **SPECULATIVO — codice inesistente su disco.** Il package `quantsys/execution/` e il modulo `quantsys/execution/reconciliation.py` descritti in versioni precedenti **non esistono** (verificato 2026-06-25). Era il design (Fasi 2-5, 8-13h) per inviare ordini reali sul Futures Testnet parallelamente al portfolio simulato. Prerequisito: BLOCKER #1 risolto (✅, §3.3). Non avviato. Conservato qui solo come schema progettuale, non come stato del codice.

**EN** > ⚠ **SPECULATIVE — code does not exist on disk.** The `quantsys/execution/` package and `quantsys/execution/reconciliation.py` module described in earlier versions **do not exist** (verified 2026-06-25). It was the design (Phases 2-5, 8-13h) to send real orders to the Futures Testnet in parallel with the simulated portfolio. Prerequisite: BLOCKER #1 resolved (✅, §3.3). Not started. Kept here only as a design sketch, not as code state.

🇮🇹 **Schema (se mai ripreso):** ABC `ExecutionAdapter` (paper | testnet_futures) con `place_market_order` / `place_stop_market` / `place_take_profit_market` / `cancel_*` / `get_position` / `set_leverage`; leva dinamica conviction-based (`lev = 1 + (max_lev−1)·conviction^alpha`, decisa 2026-05-24); riconciliazione paper-vs-testnet con warning su drift > 0.5%. Fase 1 (`.env` + `scripts/00_test_binance_testnet.py`) era l'unico pezzo done.

**EN** **Sketch (if ever resumed):** ABC `ExecutionAdapter` (paper | testnet_futures) with `place_market_order` / `place_stop_market` / `place_take_profit_market` / `cancel_*` / `get_position` / `set_leverage`; conviction-based dynamic leverage (`lev = 1 + (max_lev−1)·conviction^alpha`, decided 2026-05-24); paper-vs-testnet reconciliation with a warning on >0.5% drift. Phase 1 (`.env` + `scripts/00_test_binance_testnet.py`) was the only done piece.

### 3.6 A3 — Regime-MoE (mixture-of-universes) · A3 — Regime-MoE (mixture-of-universes)

🇮🇹 **IMPLEMENTATO-INERTE 2026-07-12 — MAI addestrato, zero risultati.** Item A3 di `docs/ROADMAP_VOL_BOOK.md` (design: memoria `mixture_of_universes_design`, adattato alla linea vol). Backbone iTransformer condiviso + **3 teste-regime** (R0 Quiet / R1 Trending / R2 Stress) mescolate da un **soft-gate ESTERNO CAUSALE** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — le filtered probabilities di `RegimeMarkovBTC` in `data/regime_probs.parquet`, **mai apprese** (proprietà anti-overfit chiave). Razionale: l'edge short-vol è **Trending-driven** (audit 2026-06-26) → la calibrazione σ regime-condizionata è direttamente monetizzabile.

- **Attivazione config-gated:** chiave `model.head_type` — **assente o `"single"` = path storico bit-identico** (verificato: stesso seed → output `torch.equal`; checkpoint production caricano con `load_state_dict` strict; suite completa verde). `"regime_moe"` attiva le teste. Esempio d'uso: `config/arch/itransformer_regime_moe.yaml` (MAI in `config/default.yaml`).
- **Mixing:** path `quantile` (produzione vol) → media pesata dal gate per livello (**Vincentization**) + re-sort monotono di sicurezza; path `t_student` → **legge della varianza totale** (stessa di `ensemble.py`): `μ_mix = Σ g_k·μ_k`, `σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)²` — σ INFLAZIONATA quando il regime è ambiguo; `lnu` = media pesata dal gate; `ls2` ri-codificato via softplus-inverse (contratto `(mu, ls2, lnu)` invariato).
- **Contratto forward invariato:** `forward(x, x_macro=None, latent=None, g=None)` — `g` opzionale in coda; `g=None` con regime_moe → gate uniforme (1/3,1/3,1/3) con warning una-tantum; burn-in/gap del regime → riga uniforme. `dir_head` (multitask) CONDIVISA tra i regimi (precedente del MoE appreso).
- **Allineamento causale:** `quantsys/model/regime_gate.py → build_regime_gate()` — `merge_asof` **backward** sui timestamp (stesso meccanismo della stratificazione val di `02_train`), mai forward.
- **Scope/esclusioni (fail-fast):** iTransformer-only; mutuamente esclusivo con il MoE appreso (`n_output_experts>1`), con `use_revin` e con `--distill`.
- **Test:** `tests/test_regime_moe.py` (19 test CPU-only, sintetici): inerzia bit-identica, one-hot→testa k, gate uniforme+teste identiche→testa singola, legge varianza totale, monotonia quantili, causalità del builder.
- ⚠ **Gate QLIKE da PRE-REGISTRARE in STATUS.md prima del primo run** (protocollo sperimentale, passo 1); training in sandbox `QUANTSYS_MODELS_ROOT` (mai su `models/itransformer`); giudice `dev_vols_qlike.py` già gate-aware (legge `head_type` dal `config.json` del modello).

**EN** **IMPLEMENTED-INERT 2026-07-12 — NEVER trained, zero results.** Item A3 of `docs/ROADMAP_VOL_BOOK.md` (design: `mixture_of_universes_design` memory, adapted to the vol line). Shared iTransformer backbone + **3 regime heads** (R0 Quiet / R1 Trending / R2 Stress) mixed by an **EXTERNAL CAUSAL soft-gate** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — the filtered probabilities of `RegimeMarkovBTC` in `data/regime_probs.parquet`, **never learned** (key anti-overfit property). Rationale: the short-vol edge is **Trending-driven** (2026-06-26 audit) → regime-conditional σ calibration is directly monetizable.

- **Config-gated activation:** key `model.head_type` — **absent or `"single"` = bit-identical legacy path** (verified: same seed → `torch.equal` outputs; production checkpoints load via strict `load_state_dict`; full suite green). `"regime_moe"` enables the heads. Usage example: `config/arch/itransformer_regime_moe.yaml` (NEVER in `config/default.yaml`).
- **Mixing:** `quantile` path (vol production) → per-level gate-weighted average (**Vincentization**) + monotone safety re-sort; `t_student` path → **total variance law** (same as `ensemble.py`): `μ_mix = Σ g_k·μ_k`, `σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)²` — σ INFLATED when the regime is ambiguous; `lnu` = gate-weighted average; `ls2` re-encoded via softplus-inverse (`(mu, ls2, lnu)` contract unchanged).
- **Forward contract unchanged:** `forward(x, x_macro=None, latent=None, g=None)` — optional trailing `g`; `g=None` under regime_moe → uniform gate (1/3,1/3,1/3) with a one-time warning; regime burn-in/gaps → uniform row. `dir_head` (multitask) SHARED across regimes (learned-MoE precedent).
- **Causal alignment:** `quantsys/model/regime_gate.py → build_regime_gate()` — **backward** `merge_asof` on timestamps (same mechanism as `02_train`'s val stratification), never forward.
- **Scope/exclusions (fail-fast):** iTransformer-only; mutually exclusive with the learned MoE (`n_output_experts>1`), with `use_revin` and with `--distill`.
- **Tests:** `tests/test_regime_moe.py` (19 CPU-only synthetic tests): bit-identical inertia, one-hot→head k, uniform gate+identical heads→single head, total variance law, quantile monotonicity, builder causality.
- ⚠ **QLIKE gate to PRE-REGISTER in STATUS.md before the first run** (experimental protocol, step 1); train in a `QUANTSYS_MODELS_ROOT` sandbox (never on `models/itransformer`); the `dev_vols_qlike.py` judge is already gate-aware (reads `head_type` from the model's `config.json`).
- 🇮🇹 **Audit causality-auditor 2026-07-12 (post-implementazione): 1 BLOCKER + 2 MAJOR + 3 MINOR, fixati.** BLOCKER-1: la riga `t` di `regime_probs.parquet` contiene la barra `[t,t+1h)` → il match esatto era lookahead di 1 barra; fix = shift dell'indice ad **availability time (+1h)** prima del merge_asof (+ regression test). MAJOR-1: staleness illimitata a fine parquet → bound `max_age` (default 168h, uniforme oltre) + fail-fast se stale >20% (⚠ il parquet su disco È stale: rigenerare con 01b prima del training). MAJOR-2: `g=None` in eval ora è `RuntimeError` (input obbligatorio; fallback uniforme solo in train). MINOR-2: `02b_walkforward_validate` fail-fasta su regime_moe (gate non threadato). **MINOR-1 (nota per la pre-registrazione, NON fixato — scelta di design):** la Vincentization del path quantile NON ha il termine between (μ-disagreement) → il meccanismo "σ inflazionata su regime ambiguo" esiste SOLO sul path t_student; sul path production (quantile) il gate QLIKE misura μ, non la calibrazione σ dichiarata come obiettivo A3 — la pre-registrazione deve dichiararlo. · **EN** **2026-07-12 causality audit (post-implementation): 1 BLOCKER + 2 MAJOR + 3 MINOR, fixed.** BLOCKER-1: row `t` of the parquet holds bar `[t,t+1h)` → the exact match was a 1-bar lookahead; fix = index shift to **availability time (+1h)** before the merge_asof (+ regression test). MAJOR-1: unbounded staleness past the parquet end → `max_age` bound (168h default, uniform beyond) + fail-fast above 20% stale (⚠ the on-disk parquet IS stale: regenerate via 01b before training). MAJOR-2: `g=None` in eval is now a `RuntimeError` (mandatory input; uniform fallback in train only). MINOR-2: `02b_walkforward_validate` fail-fasts on regime_moe. **MINOR-1 (pre-registration note, NOT fixed — design choice):** the quantile-path Vincentization has NO between term (μ-disagreement) → the "σ inflated on ambiguous regime" mechanism exists ONLY on the t_student path; on the production (quantile) path the QLIKE gate measures μ, not the σ calibration stated as the A3 goal — the pre-registration must declare this.

### 3.7 V2 delta-hedged (`04b --hedge`) + risk layer greeks-aware (A7) · V2 delta-hedged + greeks-aware risk layer (A7)

🇮🇹 **IMPLEMENTATI 2026-07-12 — entrambi INERTI/NON cablati, mai attivati live.** Sequencing B3 step 2 della `ROADMAP_VOL_BOOK`, preparato in anticipo perché il gate v1 n≥20 chiude a giorni.

- **Leg delta-hedge perp (`scripts/04b_vol_paper.py --hedge`, default OFF = v1 bit-identica):** ribilanciamento SOLO oltre la no-trade band `|Δ_book|` (dry-run 2026-07-10: churn ATM = drag puro); hedge ratio = δ teorico venue, convenzione parametrica `raw`/`adj` (`adj = Σδ−Σmark`, BTC-terms, coerente con slope −0.98 mainnet); nozionale `H* = −side·δ_conv·S` sul perp inverse; flatten automatico a settlement E a expiry (audit MAJOR-1: mai delta nudo post-expiry); stato atomico + riconciliazione con la posizione venue all'avvio (`--execute`); fail-fast se `--hedge` parte senza band/conv espliciti (= congelati dalla pre-registrazione, MINOR-3); bound di plausibilità sul δ del ticker (MINOR-2). Output: `hedge_state.json` + `hedge_ledger.jsonl` (PnL inverse esatto ricostruibile offline). Gate hedged-vs-unhedged pre-registrato in DRAFT (STATUS.md 2026-07-12): attivazione SOLO post-gate v1. Test: `tests/test_hedge_leg.py` (11, FakeDB offline).
- **Risk layer greeks-aware (`quantsys/trading/greeks_risk.py`, A7 skeleton):** cap vega/delta netti pre-trade (scaling monotono al bordo del cap, riduzioni sempre ammesse), circuit breaker vega-loss MtM con isteresi, margin sim Deribit inverse (IM/MM, conservativa, da validare vs `get_account_summary`). NON cablato in 04b: serve al sizing Kelly-su-edge della v2. Test: `tests/test_greeks_risk.py` (17).
- Audit `causality-auditor` stesso giorno: 0 blocker; 1 MAJOR + 4 MINOR trovati e **tutti applicati** (expiry-flatten, write atomica+riconciliazione, δ-bound, fail-fast attivazione, cap sign-flip).

**EN** **IMPLEMENTED 2026-07-12 — both INERT/not wired, never activated live.** B3 sequencing step 2 of `ROADMAP_VOL_BOOK`, prepared ahead because the v1 n≥20 gate closes within days.

- **Perp delta-hedge leg (`scripts/04b_vol_paper.py --hedge`, default OFF = bit-identical v1):** rebalance ONLY beyond the `|book_delta|` no-trade band (2026-07-10 dry-run: ATM churn = pure drag); hedge ratio = venue theoretical delta, parametric `raw`/`adj` convention (`adj = Σδ−Σmark`, BTC-terms, consistent with the −0.98 mainnet slope); inverse-perp notional `H* = −side·δ_conv·S`; automatic flatten at settlement AND at expiry (MAJOR-1 audit: never naked delta post-expiry); atomic state + venue-position reconciliation at `--execute` startup; fail-fast if `--hedge` starts without explicit band/conv (= frozen by the pre-registration, MINOR-3); plausibility bound on ticker deltas (MINOR-2). Output: `hedge_state.json` + `hedge_ledger.jsonl` (exact inverse PnL reconstructable offline). Hedged-vs-unhedged gate pre-registered as DRAFT (STATUS.md 2026-07-12): activation ONLY post-v1-gate. Tests: `tests/test_hedge_leg.py` (11, offline FakeDB).
- **Greeks-aware risk layer (`quantsys/trading/greeks_risk.py`, A7 skeleton):** pre-trade net vega/delta caps (monotone scaling to the cap edge, reductions always allowed), MtM vega-loss circuit breaker with hysteresis, Deribit inverse margin sim (IM/MM, conservative, to validate vs `get_account_summary`). NOT wired into 04b: it serves the v2 Kelly-on-edge sizing. Tests: `tests/test_greeks_risk.py` (17).
- Same-day `causality-auditor` audit: 0 blockers; 1 MAJOR + 4 MINOR found and **all applied** (expiry-flatten, atomic write+reconciliation, δ-bound, activation fail-fast, sign-flip cap).

### 3.8 Infrastruttura forward-test 24/7 + tooling v2 (2026-07-14) · 24/7 forward-test infrastructure + v2 tooling

🇮🇹 Cinque completamenti nella finestra di attesa del gate v1 (18/20), tutti inerti rispetto ai path congelati:
- **Collector 24/7 su VPS (netcup, EU).** `01c`+`01d` come servizi systemd `Restart=always`; kit in `deploy/vps/` (geo-test 451, setup one-shot, health-check), sync casa in `scripts/vps/` (pull scp + merge dedup + heartbeat staleness). Motivazione misurata: coverage IV di casa **18.6% delle ore** → bias di selezione oraria sul campione v1 (caveat PRE-dichiarato in STATUS a 18/20); dal deploy la serie IV è H24 e ridondata su 2 macchine. Host privato SOLO in `config/secrets.yaml → vps:`.
- **Replay offline di `04b`** (`scripts/vol/vol_paper_replay.py`): simula i tick orari sulle ore PC-off dai soli dati su disco (stesso wiring/costanti di 04b via import; IV as-of con staleness live; premio dal mark chain; delivery pubblico). Validato **bit-identico** al path live troncato (Δfeature=0, Δμ=0); output separati, MAI nel gate v1. Finding collaterale: `04b` non refresha il funding per tick (staleness cresce con l'uptime) — fix pianificato post-gate. La v2 hedged NON è replayabile (greeks del venue assenti nei dati poller) → punto successivo.
- **Estensione `--greeks` di `01c`** (inerte di default): +3 chiamate pubbliche/tick → `data/iv/atm_greeks.parquet` (δ/γ/vega/θ + bid/ask/mark per leg), selezione identica a `pick_straddle` di 04b. Attivazione sul VPS post-gate → leg hedge v2 replayabile e convenzione δ validabile sul campo. Test 4/4.
- **Giudice hedged-vs-unhedged** (`scripts/vol/hedged_vs_unhedged_judge.py`, scritto PRE-attivazione): implementa le 3 condizioni pre-registrate (var ratio ≤0.6, drag medio ≤¼·SE, n≥20, hardcoded); PnL perp inverse esatto dal ledger; **funding perp misurato per la prima volta** (accrual orario da storico PROD Deribit, cache parquet, contratto API validato). Sotto n=20 → NOT_EVALUABLE. Test 11/11.
- **Pre-registrazione A3 regime-MoE** scritta a zero numeri visti (gate QLIKE val vs incumbent, soglie in STATUS; run nella finestra GPU post-gate-v1, prerequisiti P1–P3).

**EN** Five completions during the v1-gate waiting window (18/20), all inert w.r.t. frozen paths:
- **24/7 collectors on a VPS (netcup, EU).** `01c`+`01d` as `Restart=always` systemd services; kit in `deploy/vps/` (451 geo-test, one-shot setup, health-check), home sync in `scripts/vps/` (scp pull + dedup merge + staleness heartbeat). Measured motivation: home IV coverage **18.6% of hours** → hour-of-day selection bias on the v1 sample (caveat PRE-declared in STATUS at 18/20); since the deploy the IV series is 24/7 and redundant across 2 machines. Private host ONLY in `config/secrets.yaml → vps:`.
- **Offline `04b` replay** (`scripts/vol/vol_paper_replay.py`): simulates the hourly ticks over PC-off hours from on-disk data only (same 04b wiring/constants via import; as-of IV with live staleness; chain-mark premium; public delivery). Validated **bit-identical** to the truncated live path (Δfeature=0, Δμ=0); separate outputs, NEVER in the v1 gate. Side finding: `04b` does not refresh funding per tick (staleness grows with uptime) — fix planned post-gate. The hedged v2 is NOT replayable (venue greeks missing from poller data) → next item.
- **`01c` `--greeks` extension** (inert by default): +3 public calls/tick → `data/iv/atm_greeks.parquet` (δ/γ/vega/θ + per-leg bid/ask/mark), selection identical to 04b's `pick_straddle`. VPS activation post-gate → v2 hedge leg replayable and δ convention field-validatable. Tests 4/4.
- **Hedged-vs-unhedged judge** (`scripts/vol/hedged_vs_unhedged_judge.py`, written PRE-activation): implements the 3 pre-registered conditions (var ratio ≤0.6, mean drag ≤¼·SE, n≥20, hardcoded); exact inverse perp PnL from the ledger; **perp funding measured for the first time** (hourly accrual from the Deribit PROD history, parquet cache, API contract validated). Below n=20 → NOT_EVALUABLE. Tests 11/11.
- **A3 regime-MoE pre-registration** written with zero numbers seen (val QLIKE gate vs the incumbent, thresholds in STATUS; run in the post-v1-gate GPU window, prerequisites P1–P3).

---

## 4. Valutazione · Evaluation

### 4.1 Vol-S (B2) — PASS 2026-06-10 · Vol-S (B2) — PASS

🇮🇹 **Primo gate pre-registrato superato nel progetto.** Target `features.target_type: log_rv` (log-RV a h=30 barre 1h), stessa pipeline/hyperparam del direzionale. Giudice `scripts/vol/dev_vols_qlike.py`. QLIKE test **NN 0.2572 vs HAR-RV 0.3681 vs naive 0.8067** → NN/HAR = 0.699 (gate ≤ 0.95, margine 6×); val 0.744 → test 0.699 **coerenti** (l'anti-correlazione val→test è specifica del target direzionale, NON della pipeline). Modello: Spearman test +0.45, DA 71%, ICIR +3.56. **La vol è prevedibile sopra la baseline econometrica seria ma NON tradabile sul perimetro spot/perp** (NO backtest trading sui modelli vol): valore = jump/no-trade gate difensivo + opzione Deribit/varianza. ⚠ Verifica cross-risoluzione 1m (pre-registrata): **FAIL su val** (NN/HAR QLIKE 1.0127 > 0.95) → l'edge vol è **specifico della risoluzione 1h**. Memoria: `session_2026_06_10_vols_pass.md`.

**EN** **First pre-registered gate ever passed in this project.** Target `features.target_type: log_rv` (log-RV at h=30 1h-bars), same pipeline/hyperparams as the directional line. Judge `scripts/vol/dev_vols_qlike.py`. Test QLIKE **NN 0.2572 vs HAR-RV 0.3681 vs naive 0.8067** → NN/HAR = 0.699 (gate ≤ 0.95, 6× margin); val 0.744 → test 0.699 **consistent** (the val→test anti-correlation is specific to the directional target, NOT the pipeline). Model: test Spearman +0.45, DA 71%, ICIR +3.56. **Vol is predictable above the serious econometric baseline but NOT tradable on the spot/perp perimeter** (NO trading backtest on vol models): value = defensive jump/no-trade gate + Deribit/variance option. ⚠ 1m cross-resolution check (pre-registered): **FAIL on val** (NN/HAR QLIKE 1.0127 > 0.95) → the vol edge is **specific to 1h resolution**. Memory: `session_2026_06_10_vols_pass.md`.

### 4.2 Walk-forward e distribution shift · Walk-forward and distribution shift

🇮🇹
- **Walk-forward**: purged k-fold con embargo. `n_folds=6` → 5 fold effettivi (fold 0 scartato: `train_end = fold_size − embargo < fold_size`). Il `--no-retrain` è in-sample contaminato sui fold early (carica best_model finale): per OOS pulita usa val+test split o sotto-periodi temporali. Baseline HAR per-fold: `scripts/vol/wf_har_baseline.py`.
- **Distribution shift val→test (direzionale, fatto strutturale 1m)**: le metriche in-sample (val_nll, Spearman/WHR walkforward) **anti-correlano** col backtest. NON ottimizzare regole guidate da metriche in-sample. L'errore cross-arch ≈0.995 → ensembling matematicamente inutile (riduzione varianza ≈0). Edge reale solo regime-condizionato: R0 Quiet ha Spearman +0.13÷0.19 stabile su tutti i sotto-periodi OOS, ma è edge di **rango** (entry a soglia |μ| non lo cattura). Sul target **vol** (`log_rv`) val→test sono invece coerenti (§4.1).

**EN**
- **Walk-forward**: purged k-fold with embargo. `n_folds=6` → 5 effective folds (fold 0 dropped: `train_end = fold_size − embargo < fold_size`). `--no-retrain` is in-sample-contaminated on early folds (loads final best_model): for clean OOS use val+test split or temporal sub-periods. Per-fold HAR baseline: `scripts/vol/wf_har_baseline.py`.
- **Val→test distribution shift (directional, structural at 1m)**: in-sample metrics (val_nll, walkforward Spearman/WHR) **anti-correlate** with the backtest. Do NOT optimize rules guided by in-sample metrics. Cross-arch error ≈0.995 → ensembling mathematically useless (variance reduction ≈0). Real edge only regime-conditioned: R0 Quiet has stable Spearman +0.13÷0.19 across all OOS sub-periods, but it's a **rank** edge (a |μ| threshold entry doesn't capture it). On the **vol** target (`log_rv`) val→test are instead consistent (§4.1).

### 4.3 IC metric fix (2026-06-02) · IC metric fix

🇮🇹 IC/ICIR rolling window=50 era inflato ~30× per autocorrelazione (il training iTrans mostrava IC=+0.23 con Spearman=+0.008). Sostituito con Spearman su K=5 sub-periodi non sovrapposti. Sanity check: `ic_mean=0.3728 ≈ spearman=0.3726` su segnale skill 30% (coerente).

**EN** IC/ICIR rolling window=50 was inflated ~30× by autocorrelation (iTrans training showed IC=+0.23 with Spearman=+0.008). Replaced with Spearman over K=5 non-overlapping sub-periods. Sanity check: `ic_mean=0.3728 ≈ spearman=0.3726` on a 30%-skill signal (consistent).

### 4.4 Soglie di promozione paper-trading (legacy-direzionale, SUPERATA) · Paper-trading promotion thresholds (legacy, SUPERSEDED)

🇮🇹 > ⚠ Gate della linea direzionale (KILLED OOS); i numeri (Sharpe +18.71, WHR walkforward) sono in-sample 1m e NON OOS-replicabili. Il gate **attivo** oggi è il **forward test vol** (`04b_vol_paper`, straddle ATM su `edge=log(rv_pred/var_iv)`): chiudere a **30 trade** pre-registrato (2026-06-12) prima di valutare. Conservata come record metodologico. Snapshot 2026-05-23 (3/4 raggiunte): Sharpe CI lower bound +0.78; stress test +7.22/+12.30; WHR walkforward iTrans 0.567 ma N-HiTS/TCN+Mamba 0.50-0.53 (⚠ sotto soglia 0.53); fee/gross 30.3% (al limite).

**EN** > ⚠ Directional-line gate (KILLED OOS); the numbers (Sharpe +18.71, walkforward WHR) are 1m in-sample and NOT OOS-reproducible. The **active** gate today is the **vol forward test** (`04b_vol_paper`, ATM straddle on `edge=log(rv_pred/var_iv)`): close at the pre-registered **30 trades** (2026-06-12) before judging. Kept as a methodological record. 2026-05-23 snapshot (3/4 met): Sharpe CI lower bound +0.78; stress test +7.22/+12.30; walkforward WHR iTrans 0.567 but N-HiTS/TCN+Mamba 0.50-0.53 (⚠ below the 0.53 threshold); fee/gross 30.3% (borderline).

---

## 5. Storico esperimenti · Experiment log

🇮🇹 > Esperimenti FALLITI conservati come **vaccino contro il re-test involontario**. Non re-testarli senza una ragione nuova esplicita.

**EN** > FAILED experiments kept as a **vaccine against involuntary re-testing**. Do not re-test them without an explicit new reason.

### 5.1 Pivot 1h direzionale — KILL 2026-06-10 · Directional 1h pivot — KILL

🇮🇹 **KILL definitivo** dopo probe + 1 iterazione tuning pre-registrata. Razionale: a 1m il muro era la **magnitudine** (~1.5 bps effetto vs ~26 bps costo roundtrip); a 1h il cost/σ scende da ~1.9-3.3× a ~0.25-0.42× (movimento barra ∝ √Δt, costo fisso). Gate pre-registrato: Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 a ENTRAMBI i costi 13/23 bps. **Esito: gate 4/4 fallito.** Il 1h sfonda il muro dei costi (|μ| raw mediano ≈ 43 bps ≫ 26 bps) ma probe Sharpe −0.87/PF 0.78/74 trade/−5.23%; tuned 5-seed 2 trade/PF 0.12 (l'ensemble medio-azzera μ e gonfia σ via legge varianza totale). **Anti-correlazione val→test confermata anche a 1h** (val ρ +0.19 → test ρ −0.04): è del target direzionale, non del timeframe. 2 trappole infra documentate: membri stale → `SimpleSignalModel` silente; routing `PipelineState` lstm-default. Memoria: `session_2026_06_10_pivot1h_probe.md`.

**EN** **Definitive KILL** after probe + 1 pre-registered tuning iteration. Rationale: at 1m the wall was **magnitude** (~1.5 bps effect vs ~26 bps roundtrip cost); at 1h cost/σ drops from ~1.9-3.3× to ~0.25-0.42× (bar move ∝ √Δt, fixed cost). Pre-registered gate: Sharpe≥1.0, PF≥1.3, ≥80 trades, net>0 at BOTH 13/23 bps costs. **Outcome: gate failed 4/4.** 1h breaks the cost wall (median raw |μ| ≈ 43 bps ≫ 26 bps) but probe Sharpe −0.87/PF 0.78/74 trades/−5.23%; tuned 5-seed 2 trades/PF 0.12 (the ensemble averages μ toward zero and inflates σ via the total-variance law). **Val→test anti-correlation confirmed at 1h too** (val ρ +0.19 → test ρ −0.04): it belongs to the directional target, not the timeframe. 2 infra traps documented: stale members → silent `SimpleSignalModel`; lstm-default `PipelineState` routing. Memory: `session_2026_06_10_pivot1h_probe.md`.

### 5.2 Semivarianza firmata (`log_rs_ratio`) — FAIL 2026-06-11 · Signed semivariance — FAIL

🇮🇹 **Probe pre-registrato "segno della varianza":** target `log_rs_ratio` = `log(RS⁺_fwd/RS⁻_fwd)` a h=30 barre 1h (semivarianza realizzata firmata, Patton–Sheppard). Giudice `scripts/vol/dev_vols_rs_judge.py` (HAR-RS OLS train-only + naive + train-mean, metrica MSE — il QLIKE non si applica a target non positivo-definito). **FAIL:** NN/HAR-RS MSE 0.9952 > 0.95; NN batte la costante di 0.02%; signDA 0.459 < 0.55; ρ val +0.078 → test −0.038. Punto scientifico: **nessuno** predice l'asimmetria (HAR-RS fa peggio della costante su test) → l'informazione price/volume è nei **momenti pari** (livello RV: −30% QLIKE), non nei **dispari** (direzione, signed jump variation). Filone HD-firmato chiuso. Memoria: `session_2026_06_11_rs_probe_fail.md`.

**EN** **Pre-registered "sign of variance" probe:** target `log_rs_ratio` = `log(RS⁺_fwd/RS⁻_fwd)` at h=30 1h-bars (signed realized semivariance, Patton–Sheppard). Judge `scripts/vol/dev_vols_rs_judge.py` (HAR-RS OLS train-only + naive + train-mean, MSE metric — QLIKE does not apply to a non-positive-definite target). **FAIL:** NN/HAR-RS MSE 0.9952 > 0.95; NN beats the constant by 0.02%; signDA 0.459 < 0.55; ρ val +0.078 → test −0.038. Scientific point: **nobody** predicts the asymmetry (HAR-RS is worse than the constant on test) → price/volume information is in the **even moments** (RV level: −30% QLIKE), not the **odd ones** (direction, signed jump variation). HD-signed thread closed. Memory: `session_2026_06_11_rs_probe_fail.md`.

### 5.3 Lever direzionali env-gated — TUTTI FALLITI OOS · Env-gated directional levers — ALL FAILED OOS

🇮🇹 Lever implementati come **env-flag inerti di default** in `scripts/03_backtest.py` (pattern protocollo sperimentale: zero impatto production, reversibili). **Tutti validati e FALLITI OOS** — numeri in `STATUS.md`. **Non re-testarli.**
- `QUANTSYS_QUIET_RANK_Q`/`_REGIME`/`_CONVICTION` (entry rank-based discreta per regime Quiet) → overfit del test, return −0.22% su val.
- `QUANTSYS_DECISION_CADENCE` (Fix ①, cadenza entry N candele/`h`) + `QUANTSYS_RANK_EXPOSURE` (Fix ②, esposizione continua ∝ percentile causale di μ, regime-gated) → baseline val +4.03%/PF 1.88 → Fix①② −2.24%/PF 0.22; rank anti-predittivo OOS, PnL dominata dal path SL/TP.
- `QUANTSYS_SIGMA_SCALE` (Step 0.5, ricalibra σ post-denorm) → σ↓ peggiora il backtest, ottimo ≈1.0.
- `QUANTSYS_MIN_EXPECTED_RET` (sweep cost-aware 13 vs 23 bps) → a 1h NON vincolante (|μ| raw mediano ≈43 bps); gate trading comunque fallito.
- `QUANTSYS_HORIZON_EXIT`, `QUANTSYS_REGIME_ALLOW`/`_INVERT` → isolamento edge rango / regime gating, nessun PnL OOS.

**Sintesi:** l'edge a soglia/rango e la calibrazione-σ non producono PnL OOS sul direzionale; restano flag inerti documentati.

**EN** Levers implemented as **env-flags inert by default** in `scripts/03_backtest.py` (experimental-protocol pattern: zero production impact, reversible). **All validated and FAILED OOS** — numbers in `STATUS.md`. **Do not re-test them.**
- `QUANTSYS_QUIET_RANK_Q`/`_REGIME`/`_CONVICTION` (discrete rank-based entry for the Quiet regime) → test overfit, −0.22% return on val.
- `QUANTSYS_DECISION_CADENCE` (Fix ①, entry cadence N candles/`h`) + `QUANTSYS_RANK_EXPOSURE` (Fix ②, continuous exposure ∝ causal percentile of μ, regime-gated) → val baseline +4.03%/PF 1.88 → Fix①② −2.24%/PF 0.22; rank anti-predictive OOS, PnL dominated by the SL/TP path.
- `QUANTSYS_SIGMA_SCALE` (Step 0.5, recalibrate σ post-denorm) → σ↓ worsens the backtest, optimum ≈1.0.
- `QUANTSYS_MIN_EXPECTED_RET` (cost-aware sweep 13 vs 23 bps) → at 1h NOT binding (median raw |μ| ≈43 bps); trading gate failed anyway.
- `QUANTSYS_HORIZON_EXIT`, `QUANTSYS_REGIME_ALLOW`/`_INVERT` → rank-edge isolation / regime gating, no OOS PnL.

**Summary:** threshold/rank edge and σ-calibration produce no OOS PnL on the directional target; they remain documented inert flags.

### 5.4 Multi-timeframe (1m+5m+1h) — design, MAI implementato · Multi-timeframe — design, NEVER implemented

🇮🇹 > ⚠ Il package `mtf/` e gli artefatti `data/mtf_dataset.npz` / `models/mtf_*` / `results/mtf_*` **non esistono su disco** (verificato 2026-06-25). Design conservato come record; effort 6-9 settimane elapsed, era gated dietro Sharpe live>0.5 mai raggiunto. Architettura proposta: 3 encoder separati della stessa famiglia (1m→120 candele, 5m→24, 1h→24), fusion via cross-attention o gated concat, sviluppo in package parallelo isolato che riusa loss/utility/risk/FeatureBuilder da `quantsys/`. Rischio chiave: **data leakage nei resample** (se il 5m bar al minuto T:00 include T+1..T+4 → predici il futuro; test critico = shuffle X_train). Costo iterazione 6-12h/esperimento, distill completa stimata 30-50h GPU.

**EN** > ⚠ The `mtf/` package and the `data/mtf_dataset.npz` / `models/mtf_*` / `results/mtf_*` artifacts **do not exist on disk** (verified 2026-06-25). Design kept as a record; 6-9 weeks elapsed effort, was gated behind a live Sharpe>0.5 never reached. Proposed architecture: 3 separate same-family encoders (1m→120 candles, 5m→24, 1h→24), fusion via cross-attention or gated concat, developed in an isolated parallel package reusing loss/utility/risk/FeatureBuilder from `quantsys/`. Key risk: **resample data leakage** (if the 5m bar at minute T:00 includes T+1..T+4 → predicting the future; critical test = shuffle X_train). Iteration cost 6-12h/experiment, full distill estimated 30-50h GPU.

### 5.5 mamba-ssm CUDA kernel — aperto (target-agnostico) · mamba-ssm CUDA kernel — open (target-agnostic)

🇮🇹 **Unica voce della roadmap legacy ancora potenzialmente utile** (speedup, indipendente dal target). L'implementazione attuale `quantsys/model/tcn_mamba.py` è pure-PyTorch (`SimplifiedMambaBlock._parallel_scan_chunk`). Il pacchetto `mamba-ssm` (Tri Dao) implementa un kernel CUDA fuso (selective scan, prefix-scan Blelloch, ricomputo stato in backward à la Flash Attention): speedup atteso +3-5× sul branch Mamba (sopra il +1.4-1.6× già ottenuto con AMP off + chunk pre-alloc). Prerequisiti mancanti su questa macchina: CUDA Toolkit dev 12.1.x (deve matchare `torch.version.cuda`), MSVC Build Tools 2022, `CUDA_HOME`. Install `--no-build-isolation` (causal-conv1d + mamba-ssm), modifica `MambaBranch` con import condizionale (fallback `SimplifiedMambaBlock`), retrain TCN+Mamba (checkpoint NON compatibili). Rollback: `pip uninstall` → auto-detect `_HAS_MAMBA_SSM = False`. **Quando**: retrain frequenti / `mamba_layers > 3` / sequenze T > 240. Non se il training è "abbastanza veloce".

**EN** **Only legacy-roadmap item still potentially useful** (speedup, target-independent). Current implementation `quantsys/model/tcn_mamba.py` is pure-PyTorch (`SimplifiedMambaBlock._parallel_scan_chunk`). The `mamba-ssm` package (Tri Dao) implements a fused CUDA kernel (selective scan, Blelloch prefix-scan, backward state recompute à la Flash Attention): expected speedup +3-5× on the Mamba branch (on top of the +1.4-1.6× already obtained via AMP off + chunk pre-alloc). Missing prerequisites on this machine: dev CUDA Toolkit 12.1.x (must match `torch.version.cuda`), MSVC Build Tools 2022, `CUDA_HOME`. Install `--no-build-isolation` (causal-conv1d + mamba-ssm), edit `MambaBranch` with a conditional import (fallback `SimplifiedMambaBlock`), retrain TCN+Mamba (checkpoints NOT compatible). Rollback: `pip uninstall` → auto-detect `_HAS_MAMBA_SSM = False`. **When**: frequent retrains / `mamba_layers > 3` / sequences T > 240. Not if training is "fast enough".

### 5.6 Audit residui low-priority · Low-priority audit residue

🇮🇹 4 issue MEDIE + 1 INFRASTRUCTURE dal grand audit 2026-05-23 (8/8 CRITICHE + 8/8 ALTE + 5/9 MEDIE già chiuse). ⚠ I riferimenti `file:linea` marciscono a ogni edit — verificali con grep prima di agire.

**EN** 4 MEDIUM issues + 1 INFRASTRUCTURE from the 2026-05-23 grand audit (8/8 CRITICAL + 8/8 HIGH + 5/9 MEDIUM already closed). ⚠ The `file:line` references rot on every edit — verify with grep before acting.

🇮🇹
| # | File | Issue | Fix proposto · Proposed fix |
|---|---|---|---|
| 21 | `quantsys/trading/__init__.py` | NaN check `x != x` criptico, solo su `size` | NaN guard esplicito all'inizio di `open_position` |
| 23 | `quantsys/data/__init__.py` | Sanity OHLCV `high > close * 10` può scartare flash crash legittimi | rilassare soglia o usare prezzo candela precedente |
| 27 | `quantsys/model/ensemble.py` | `arch_names` non impostato nei fallback `load` | non critico, default OK |
| 28 | `quantsys/features/__init__.py` | `vol_x_pos` crash se colonne assenti su dataset corto | `.get(col, 0)` o try/except |
| #5 ⚠ | `quantsys/trading/__init__.py` + `scripts/03_backtest.py` | `SignalGenerator.set_regime_threshold` esiste ma chiamate DISABILITATE | calibrare o rimuovere dead code |

**EN**
| # | File | Issue | Proposed fix |
|---|---|---|---|
| 21 | `quantsys/trading/__init__.py` | Cryptic NaN check `x != x`, only on `size` | Explicit NaN guard at top of `open_position` |
| 23 | `quantsys/data/__init__.py` | OHLCV sanity `high > close * 10` may discard legitimate flash crashes | Relax threshold or use previous candle price |
| 27 | `quantsys/model/ensemble.py` | `arch_names` not set in `load` fallbacks | Non-critical, default OK |
| 28 | `quantsys/features/__init__.py` | `vol_x_pos` crashes if columns absent on short dataset | `.get(col, 0)` or try/except |
| #5 ⚠ | `quantsys/trading/__init__.py` + `scripts/03_backtest.py` | `SignalGenerator.set_regime_threshold` exists but call sites DISABLED | Calibrate or remove dead code |

🇮🇹 **Contesto #5:** bisect 2026-05-24 ha mostrato che le soglie regime hardcoded (overheating +3pp, stagflation +5pp sul default 0.52) riducevano Sharpe da +18.71 a −4.44 (filtravano 27/42 trade vincenti). Infrastruttura resta ma dead code.

**EN** **#5 context:** 2026-05-24 bisect showed hardcoded regime thresholds (overheating +3pp, stagflation +5pp over the 0.52 default) cut Sharpe from +18.71 to −4.44 (filtered 27/42 winning trades). Infrastructure stays but is dead code.

---

## 6. Insight consolidati e regola d'oro · Consolidated insights and golden rule

🇮🇹 > ⚠ **Caveat:** i punti 1-4 sono lezioni della linea **direzionale-1m** (KILLED OOS). Le metriche in-sample (walkforward DA/Spearman) **anti-correlano** col backtest sul target direzionale; restano insegnamenti su *trading layer* e *scale*, NON una promessa di edge. Sul target **vol** (`log_rv`) val→test sono coerenti (§4.1).

**EN** > ⚠ **Caveat:** points 1-4 are lessons from the **directional-1m** line (KILLED OOS). In-sample metrics (walkforward DA/Spearman) **anti-correlate** with the backtest on the directional target; they remain *trading-layer* and *scale* lessons, NOT an edge promise. On the **vol** target (`log_rv`) val→test are consistent (§4.1).

🇮🇹
1. **Modello sano in-sample in tutti i setup direzionali** (walkforward DA 0.53-0.54, Spearman 0.08-0.09, σ calibrato) ma queste metriche anti-correlano col PnL OOS. Quando il problema emerge in-sample è quasi sempre nel **trading layer** (scala, soglie, SL/TP), non nel modello — caso paradigmatico 2026-05-23: Sharpe −256 → +18.7 con 1 moltiplicazione mancante (la lezione vale, il +18.7 NON è OOS-replicabile).
2. **h=15 è strutturalmente perdente**: cost roundtrip 26 bps ≈ |realized return medio| 25 bps. h=30 raddoppia il segnale a costo costante. Applicato.
3. **`max_sigma` va sempre dimensionato sulla distribuzione σ del modello specifico** (es. p99 della σ_test). Valori arbitrari sono inutili.
4. **`trailing_atr_mult: 1.5`** ≈ 11 bps di trail su BTC 1m → chiude su rumore (< cost 26 bps). Su 1m `use_trailing_stop: false` batte qualsiasi trailing tunato.
5. **Verificare le scale unit-by-unit prima di retrainare**: per 6+ sessioni a maggio 2026 si è cercato il fix sui pesi del modello (RevIN, h, stride, multi-teacher) — il vero bug era 1 moltiplicazione mancante in 2 file (denormalizzazione z-score → raw).
6. **Dicotomia momenti pari/dispari** (sintesi chiave del progetto): l'informazione in price/volume riguarda i momenti **pari** (livello RV, predicibile: Vol-S PASS) e non i **dispari** (direzione, segno della varianza: KILL/FAIL).

**EN**
1. **Healthy in-sample model across all directional setups** (walkforward DA 0.53-0.54, Spearman 0.08-0.09, calibrated σ) yet these metrics anti-correlate with OOS PnL. When the problem shows in-sample it is almost always in the **trading layer** (scale, thresholds, SL/TP), not the model — paradigmatic 2026-05-23 case: Sharpe −256 → +18.7 from one missing multiplication (the lesson holds, the +18.7 is NOT OOS-reproducible).
2. **h=15 is structurally a losing setup**: roundtrip cost 26 bps ≈ |mean realized return| 25 bps. h=30 doubles the signal at constant cost. Applied.
3. **`max_sigma` must always be sized on the specific model's σ distribution** (e.g. p99 of σ_test). Arbitrary values are useless.
4. **`trailing_atr_mult: 1.5`** ≈ 11 bps of trail on BTC 1m → closes on noise (< 26 bps cost). On 1m bars `use_trailing_stop: false` beats any tuned trailing.
5. **Verify scales unit-by-unit before retraining**: for 6+ sessions in May 2026 we hunted fixes on model weights (RevIN, h, stride, multi-teacher) — the actual bug was one missing multiplication across 2 files (z-score → raw denormalization).
6. **Even/odd moment dichotomy** (project's key synthesis): the information in price/volume concerns the **even** moments (RV level, predictable: Vol-S PASS) and not the **odd** ones (direction, sign of variance: KILL/FAIL).

🇮🇹 **Regola d'oro:** un fix alla volta, ogni cambio validato da backtest con CI bootstrap. Pattern: (1) applica un fix singolo; (2) retrain completo (un solo modello base per smoke test); (3) confronta `val_nll`/`DA`/`Spearman`/`Sharpe CI` con la baseline pre-fix; (4) se ≥2% miglioramento → mantieni; (5) se peggiora o invariato → rollback. **Lezione 2026-05-24:** attivare fix "completi" senza validation pre-merge può accendere dead state non calibrati (caso #5 §5.6); un bisect rapido trova il colpevole in 2 iterazioni.

**EN** **Golden rule:** one fix at a time, every change validated with a bootstrap-CI backtest. Pattern: (1) apply a single fix; (2) full retrain (one base model for a smoke test); (3) compare `val_nll`/`DA`/`Spearman`/`Sharpe CI` with the pre-fix baseline; (4) if ≥2% improvement → keep; (5) if worse or unchanged → rollback. **2026-05-24 lesson:** enabling "complete" fixes without pre-merge validation can activate uncalibrated dead state (case #5 §5.6); a fast bisect finds the culprit in 2 iterations.
