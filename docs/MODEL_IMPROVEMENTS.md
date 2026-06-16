# QUANTSYS — Miglioramenti modello residui · QUANTSYS — Outstanding model improvements

🇮🇹 Tutto il "già fatto" è stato spostato in `CHANGELOG.md` e nelle note `~/.claude/projects/E--quantsys-project/memory/`. Questo file lista solo ciò che resta da implementare, in ordine raccomandato.

**EN** Everything already done lives in `CHANGELOG.md` and the notes under `~/.claude/projects/E--quantsys-project/memory/`. This file lists only what remains to implement, in recommended order.

---

## 🔵 2026-06-09 — Pivot timeframe 1m→1h (Strada 1 post-KILL cross-sectional) · 🔵 2026-06-09 — 1m→1h timeframe pivot (Path 1 after the cross-sectional KILL)

### Razionale · Rationale

🇮🇹 Il probe cross-sectional (2026-06-06) ha dato **KILL** con diagnosi "il muro è la **magnitudine**, non il segno" (~1.5 bps di effetto vs ~26 bps di costo roundtrip). Il costo è fisso, il movimento di barra cresce ∝ √Δt → il rapporto costo/σ scala come 1/√Δt: a 1m era ~1.9–3.3×, **a 1h scende a ~0.25–0.42×**. Pivot = **stesso motore, candele 1h**, storico multi-anno 2019→oggi (~65k barre).

**EN** The cross-sectional probe (2026-06-06) returned **KILL** with diagnosis "the wall is **magnitude**, not sign" (~1.5 bps effect vs ~26 bps roundtrip cost). The cost is fixed, the bar move grows ∝ √Δt → the cost/σ ratio scales as 1/√Δt: at 1m it was ~1.9–3.3×, **at 1h it drops to ~0.25–0.42×**. Pivot = **same engine, 1h candles**, multi-year history 2019→today (~65k bars).

### Design interval-agnostic (invariante: identità a 1m) · Interval-agnostic design (invariant: identity at 1m)

🇮🇹
- Nuovo helper `interval_minutes_from_cfg` in `quantsys/utils` (mappa `data.interval` → minuti, `ValueError` fail-fast su intervalli sconosciuti).
- `FeatureBuilder(interval_minutes=...)` con `bars_per_day = 1440 // interval_minutes` e helper `_tbars(minutes)` (floor anti-degenerazione 2 barre).
- **Finestre TIME-semantic convertite** (mantengono il significato in TEMPO): strutturali 30d/90d/365d (`days × bars_per_day`), momentum 7d/30d/90d, `funding_rate_1d` (`bars_per_day`), `session_position` (240 min), `price_vs_ma200m` (200 min).
- **Finestre BAR-semantic deliberatamente invariate** (si traslano col timeframe): windows [5, 10, 20, 60], CVD, vwap, VP scales 60/240/1440 **BARRE** (a 1m = 1h/4h/1d; a 1h = 60h/10d/60d), lags.
- A `interval_minutes=1` tutte le conversioni sono identità → comportamento legacy preservato.

**EN**
- New `interval_minutes_from_cfg` helper in `quantsys/utils` (maps `data.interval` → minutes, fail-fast `ValueError` on unknown intervals).
- `FeatureBuilder(interval_minutes=...)` with `bars_per_day = 1440 // interval_minutes` and the `_tbars(minutes)` helper (anti-degeneration floor of 2 bars).
- **TIME-semantic windows converted** (keep their meaning in TIME): structural 30d/90d/365d (`days × bars_per_day`), momentum 7d/30d/90d, `funding_rate_1d` (`bars_per_day`), `session_position` (240 min), `price_vs_ma200m` (200 min).
- **BAR-semantic windows deliberately unchanged** (they shift with the timeframe): windows [5, 10, 20, 60], CVD, vwap, VP scales 60/240/1440 **BARS** (at 1m = 1h/4h/1d; at 1h = 60h/10d/60d), lags.
- At `interval_minutes=1` every conversion is an identity → legacy behavior preserved.

### Contratto train↔inference esteso · Extended train↔inference contract

🇮🇹 `PipelineState.interval` (già persistito) ora esposto come property `interval_minutes` (fallback 1 per pkl legacy). I consumer (`FeatureAssembler` live, `99_replay`) derivano l'interval dal **PipelineState**, non dalla config. Nuovo guard `RuntimeError` "interval mismatch" in `03_backtest.py` e `04_live_signals.py` (stesso pattern del guard `forecast_horizon`): modello-1m + config-1h = combinazione invalida bloccata.

**EN** `PipelineState.interval` (already persisted) is now exposed as the `interval_minutes` property (fallback 1 for legacy pkl). Consumers (live `FeatureAssembler`, `99_replay`) derive the interval from the **PipelineState**, not from the config. New `RuntimeError` "interval mismatch" guard in `03_backtest.py` and `04_live_signals.py` (same pattern as the `forecast_horizon` guard): 1m-model + 1h-config = invalid combination, blocked.

### Annualizzazione interval-aware · Interval-aware annualization

🇮🇹 `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760) in `03_backtest.py` (`bootstrap_sharpe_ci`) e `RiskManager(bars_per_year=...)`. σ safety-net scalata: `0.05·√interval_minutes` (1h→≈0.387) — preserva l'intento del guard (cattura il bug di denormalizzazione z→raw ~30–100×, non la crescita √60 legittima della σ a orizzonte 30 barre orarie).

**EN** `bars_per_year = 525600 // interval_minutes` (1m→525600, 1h→8760) in `03_backtest.py` (`bootstrap_sharpe_ci`) and `RiskManager(bars_per_year=...)`. Scaled σ safety net: `0.05·√interval_minutes` (1h→≈0.387) — preserves the guard's intent (catches the ~30–100× z→raw denormalization bug, not the legitimate √60 growth of σ at a 30-hourly-bar horizon).

### Config pivot (`config/default.yaml`) · Pivot config (`config/default.yaml`)

🇮🇹
| Parametro | Valore 1h | Era (1m) |
|---|---|---|
| `interval` | `1h` | `1m` |
| `start_time` | `2019-01-01` | `2025-05-19` |
| `window_stride` | 1 | 5 |
| `embargo_steps` | 168 | 1500 |
| `max_hold_candles` | 60 (vincolo ≥ h=30) | 240 |
| `min_expected_ret` | 0.0013 (gate cost-aware 13 bps; 2° test pre-registrato a 23 bps) | 0.0005 |
| `max_sigma` | 0.10 (≈0.015·√60, da ricalibrare) | 0.015 |
| `forecast_horizon` | 30 INVARIATO — ora = **30 ORE** | 30 (= 30 min) |
| `window_size` | 120 INVARIATO — ora = **5 giorni** di contesto | 120 (= 2h) |

**EN**
| Parameter | 1h value | Was (1m) |
|---|---|---|
| `interval` | `1h` | `1m` |
| `start_time` | `2019-01-01` | `2025-05-19` |
| `window_stride` | 1 | 5 |
| `embargo_steps` | 168 | 1500 |
| `max_hold_candles` | 60 (constraint ≥ h=30) | 240 |
| `min_expected_ret` | 0.0013 (cost-aware 13 bps gate; 2nd pre-registered test at 23 bps) | 0.0005 |
| `max_sigma` | 0.10 (≈0.015·√60, to recalibrate) | 0.015 |
| `forecast_horizon` | 30 UNCHANGED — now = **30 HOURS** | 30 (= 30 min) |
| `window_size` | 120 UNCHANGED — now = **5 days** of context | 120 (= 2h) |

🇮🇹 TODO documentato: GJR-GARCH ω da ri-stimare su rendimenti 1h (il forecast MC non è sul critical path del backtest).

**EN** Documented TODO: GJR-GARCH ω to re-estimate on 1h returns (the MC forecast is not on the backtest critical path).

### Overlay `config/interval/` (2026-06-10) · `config/interval/` overlay (2026-06-10)

🇮🇹 Le chiavi interval-dipendenti della tabella sopra sono ora fattorizzate in **`config/interval/{1m,1h}.yaml`**, mergiate da `load_config` per-sezione shallow **dopo secrets e prima dell'overlay arch** (l'arch resta l'override più specifico). Selezione via `QUANTSYS_INTERVAL` o `python run_all.py --interval 1m|1h` (propagata a tutti i subprocess, incluso il loop `--distill`). File mancante → warning e si prosegue col solo default.yaml (stesso comportamento dell'overlay arch).

**EN** The interval-dependent keys from the table above are now factored into **`config/interval/{1m,1h}.yaml`**, merged by `load_config` per-section shallow **after secrets and before the arch overlay** (arch stays the most specific override). Selected via `QUANTSYS_INTERVAL` or `python run_all.py --interval 1m|1h` (propagated to all subprocesses, including the `--distill` loop). Missing file → warning, then default.yaml only (same behavior as the arch overlay).

### Dati · Data

🇮🇹 `raw_candles.parquet` ora 1h 2019→oggi (**65.145 barre**); funding completo dal lancio perp 2019-09-10 (**7.394 obs**, re-download completo — il vecchio file partiva dal 2021); backup 1m in `data/backup_1m/` e `models/backup_1m/`. Dataset: `X_train (51120, 120, 104)` — **STESSA composizione canonica 104 = 86 dinamiche + 18 strutturali**. Fix cp1252 (`reconfigure` UTF-8) aggiunto a `01_download_data.py` e `01b_download_macro.py`.

**EN** `raw_candles.parquet` is now 1h 2019→today (**65,145 bars**); full funding since the perp launch 2019-09-10 (**7,394 obs**, full re-download — the old file started in 2021); 1m backups in `data/backup_1m/` and `models/backup_1m/`. Dataset: `X_train (51120, 120, 104)` — **SAME canonical 104 = 86 dynamic + 18 structural composition**. cp1252 fix (UTF-8 `reconfigure`) added to `01_download_data.py` and `01b_download_macro.py`.

### Rollback 1m · 1m rollback

🇮🇹 Config 1m: basta `--interval 1m` (= overlay `config/interval/1m.yaml`: `interval: 1m`, `start_time: 2025-05-19`, stride 5, embargo 1500, max_hold 240, min_ret 0.0005, max_sigma 0.015) + restore `data/backup_1m/*` — tutte le conversioni sono identità a 1m, **il codice non va toccato**.

**EN** 1m config: just `--interval 1m` (= `config/interval/1m.yaml` overlay: `interval: 1m`, `start_time: 2025-05-19`, stride 5, embargo 1500, max_hold 240, min_ret 0.0005, max_sigma 0.015) + restore `data/backup_1m/*` — every conversion is an identity at 1m, **no code changes needed**.

### Gate pre-registrato e stato · Pre-registered gate and status

🇮🇹 **Gate del pivot:** Sharpe≥1.0, PF≥1.3, ≥80 trade, net>0 a **ENTRAMBI** i costi 13 e 23 bps sul test OOS. **Stato:** dati e config completati ✅; **training/backtest 1h NON ancora eseguiti** (in corso).

**EN** **Pivot gate:** Sharpe≥1.0, PF≥1.3, ≥80 trades, net>0 at **BOTH** 13 and 23 bps costs on the OOS test set. **Status:** data and config done ✅; **1h training/backtest NOT yet executed** (in progress).

---

## 🧭 PIVOT ROADMAP 2026-06-06 — esauriti i lever model-side, 4 assi studiati · 4 axes studied after model-side levers exhausted

🇮🇹 Dopo che tutti i lever model/backtest-side su BTC-1m sono risultati negativi OOS (distill≡baseline, ensemble corr 0.995, rank-harvest fallito, mixture/σ-recal inutili), è stata avviata la **Strada A (paper-trading live)** e studiato il pivot via fan-out di 4 subagent. Dettaglio in memoria `pivot_fanout_2026_06_06`.

**EN** After all model/backtest-side levers on BTC-1m proved negative OOS (distill≡baseline, ensemble corr 0.995, rank-harvest failed, mixture/σ-recal worthless), **Path A (live paper-trading)** was launched and the pivot studied via a 4-subagent fan-out. Full detail in memory note `pivot_fanout_2026_06_06`.

🇮🇹
| Asse | Prior edge tradabile | Test più economico | Effort | Rischio chiave |
|---|---|---|---|---|
| **Cross-sectional** multi-asset | **il più alto** | Spearman cross-sezionale di μ (ore) | M | edge BTC-idiosincratico / liquid∩costi |
| **Timeframe → 1h** | plausibile (cost/σ 0.25-0.42 vs 1.9-3.3) | nessuno (serve re-download+re-tune) | M | cost-fragile; anti-corr val→test è del metodo |
| **Target → volatilità** | **predicibile ma NON tradabile** | one-line target + QLIKE vs HAR-RV (S) | S esp. / L+ prodotto | nessuno strumento per monetizzare |
| **Asset class → ES 1m** | modesto (session-mechanical) | nessuno (rewrite data-layer) | M | già HFT-arbitraggiato; leakage roll |

**EN**
| Axis | Tradable-edge prior | Cheapest test | Effort | Key risk |
|---|---|---|---|---|
| **Cross-sectional** multi-asset | **highest** | cross-sectional Spearman of μ (hours) | M | BTC-idiosyncratic edge / liquid∩costs |
| **Timeframe → 1h** | plausible (cost/σ 0.25-0.42 vs 1.9-3.3) | none (needs re-download+re-tune) | M | cost-fragile; val→test anti-corr is method-level |
| **Target → volatility** | **predictable but NOT tradable** | one-line target + QLIKE vs HAR-RV (S) | S exp. / L+ product | no instrument to monetize |
| **Asset class → ES 1m** | modest (session-mechanical) | none (data-layer rewrite) | M | already HFT-arbitraged; roll leakage |

🇮🇹 **Sequenza raccomandata:** (1) **probe cross-sectional IC** (Spearman cross-sezionale di μ; kill pre-registrato se ≈0); (2) in parallelo **vol vs HAR-RV** (chiude B2 + jump/no-trade gate per il paper-trading A); differiti 1h poi ES. **B1 order-book L2** resta l'asse informazione-nuova ortogonale.

**EN** **Recommended sequence:** (1) **cross-sectional IC probe** (cross-sectional Spearman of μ; pre-registered kill if ≈0); (2) in parallel **vol vs HAR-RV** (closes B2 + jump/no-trade gate for paper-trading A); 1h then ES deferred. **B1 order-book L2** stays the orthogonal new-information axis.

---

## 🔴 RESUME 2026-06-04 — Fix #3 (T=240) regressione confermata: scegliere tra 4 opzioni

🇮🇹 **Stato pipeline al termine sessione 2026-06-03 → 2026-06-04 ~01:00:**

🇮🇹
- ✅ **Stage 4.6 + 4.7 (live engine)** completati: `LiveCandleBuffer + FeatureAssembler` wired in `LiveEngine.__init__`, `_pad_or_truncate` rimosso da `_predict()`, parity test verdi. → **Superato il 2026-06-05/06: Stage 5 (parity feature+segnale) e smoke test 4.10 COMPLETATI → BLOCKER #1 RISOLTO** (vedi header dedicato "✅ BLOCKER #1 ... RISOLTO" sotto). + catch-up contiguo `candle_buffer` (A1.1).
- ✅ **Quick Wins SNR + prob_threshold testati e ROLLED BACK**: bisection mostra che alzare prob_threshold 0.52→0.58 azzera trade (μ_pred troppo piccoli), 0.53 peggiora (WR 33%→18%), SNR≥0.10 filtra solo i loser (Sharpe -277). Config ripristinata a `prob_threshold: 0.52`, `min_snr: 0.0`. Wiring `min_snr` in `scripts/03_backtest.py:542-550` MANTENUTO per future calibrazioni.
- ✅ **Option B fresh data retrain** (~40 min): Sharpe -3.37%→**-1.81%**, Spearman walkforward +0.040→**+0.065** (+62%). Beneficio reale ma WHR ancora 0.517 < 0.53.
- 🔴 **Fix #3 (T=240) applicato + retrain completo + walkforward** = REGRESSIONE:
  - Walkforward Spearman crollato **+0.065 → +0.034** (-48%)
  - WHR mean **0.504 ± 0.011** (sotto random 0.50, era 0.517)
  - Backtest single-arch (ensemble eterogeneo rotto per shape mismatch N-HiTS+TCN+Mamba ancora a T=120): solo 3 trade, statisticamente non significativo
  - Smoke test 1-ensemble aveva anticipato: test DA 0.497, Spearman +0.0016 (p=0.870)
  - **Causa diagnosticata**: T=240 collassa la varianza di μ_pred → modello troppo conservativo (3 trade vs 12), dataset 525k 1m probabilmente non ha abbastanza profondità per i parametri aggiuntivi (overfitting al noise temporale, plateau letteratura 192-384 non raggiunto su BTC 1m con questa quantità di dati)

🇮🇹 **Stato config attuale** (`config/default.yaml`): `window_size: 240`, `embargo_steps: 3000`, `n_folds: 6` (MA il walkforward script gira solo 5 fold — bug da investigare). Dataset npz a T=240 (10 GB). Modelli iTrans a T=240 sovrascritti su T=120 precedenti (no backup).

🇮🇹 **Decisione da prendere domani — scegliere tra:**

🇮🇹
| # | Opzione | Effort | Probabilità successo | Pro/Contro |
|---|---|---|---|---|
| **A** | **Rollback completo a T=120** (revert config + rebuild npz + retrain iTrans/N-HiTS/TCN+Mamba) | ~40 min iTrans + 50-70 min full distill se vuoi anche N-HiTS+TCN+Mamba | ✅ alta — ripristina baseline noto (Sharpe -1.81% post-Option B) | Sicuro ma "torna indietro"; nessun guadagno di edge ulteriore |
| **B** | **Mantieni T=240 e retrain anche N-HiTS + TCN+Mamba a T=240** per riattivare ensemble eterogeneo + distillation | ~2-3h (3 archs × ~40 min ciascuno + distillation) | media — distillation potrebbe stabilizzare il segnale collassato di iTrans T=240 | Costoso ma scopre se l'ensemble è la chiave di T=240; rischio: regressione conferma anche con ensemble |
| **C** | **T intermedio = 180** (config window_size: 180, embargo_steps: 2400) | ~50 min retrain singolo arch | bassa — la cliff Spearman 0.065→0.034 a T=240 è netta, 180 è geometricamente vicino | Possibile sweet spot ma improbabile dato il pattern |
| **D** | **Investigare bug `n_folds` config not respected** + rollback T=120 + Step B3 (retrain 119-feat reference) | ~30 min audit + 40 min rollback + 2h Step B3 | n/a — diagnostico | Non migliora edge direttamente ma chiude due loose ends (Fix #4 wiring + B1/B2/B3 verdict su distribution shift) |

🇮🇹 **Raccomandazione preferita: A (rollback) + parallel D Step B3** se vuoi sfruttare le 2h notturne. Il rollback A è veloce e ripristina lo stato funzionante; D Step B3 risponde alla domanda di fondo "il distribution shift val→test è transitorio (mercato) o strutturale (C-funding)?", informando la prossima sessione.

🇮🇹 **Punti di ripresa per la prossima sessione:**
1. **Prima cosa**: scegliere fra A/B/C/D
2. **Se A**: editare `config/default.yaml` lines 53,89,90 (window_size 240→120, embargo_steps 3000→1500; n_folds tienilo a 6 — è Fix #4 valido) → `python scripts/01_download_data.py` → `python run_all.py --arch itransformer --skip-update --skip-macro --skip-walkfwd --no-browser` → walkforward + backtest comparativi con baseline post-Option B (Sharpe -1.81%, Spearman wf +0.065)
3. **Se B**: due retrain aggiuntivi `$env:QUANTSYS_ARCH="nhits"; python run_all.py ...` e `$env:QUANTSYS_ARCH="tcnmamba"; python run_all.py ...` poi backtest senza `QUANTSYS_BACKTEST_SINGLE_ARCH=1`
4. **Se D**: audit `scripts/02b_walkforward_validate.py` per `n_folds` hardcoded vs config-read; poi rollback A; poi Step B3 (richiede prima ricostruire 119-feat dataset, vedi Step B feasibility report nella sessione precedente)
5. **In tutti i casi**: completare Stage 4.10 smoke test live (`python scripts/04_live_signals.py`) e Stage 4.11 doc closure (marcare BLOCKER #1 ✅ DONE in questo file e aggiornare TEORIA.md/AVVIO.md se citano "39 feature live mismatched")

🇮🇹 **Files modificati nella sessione non committati** (verifica con `git status`):
- `quantsys/trading/__init__.py` (5 audit fixes: #21 NaN guard, #5 dead code regime threshold rimosso, SNR filter aggiunto a SignalGenerator)
- `quantsys/data/__init__.py` (audit #23: threshold flash crash 10×→50×)
- `quantsys/model/ensemble.py` (audit #27: commento documentale)
- `quantsys/features/__init__.py` (audit #28: try/except vol_x_pos)
- `scripts/03_backtest.py` (#5 dead code rimosso + wiring `min_snr` da config)
- `scripts/04_live_signals.py` (Stage 4.6+4.7 LiveEngine wiring + assert shape 104)
- `config/default.yaml` (window_size 240, embargo_steps 3000, n_folds 6, prob_threshold 0.52, min_snr 0.0)

---

## 🔴 NEXT — Diagnostica backtest negativo post-distill 2026-06-03 · 🔴 NEXT — Diagnostics on the negative post-distill backtest 2026-06-03

🇮🇹 **Contesto:** il `run_all.py --distill` terminato alle 00:11 del 2026-06-03 ha prodotto backtest preoccupanti:

**EN** **Context:** the `run_all.py --distill` that finished at 00:11 on 2026-06-03 produced worrying backtests:

🇮🇹
| Arch | Sharpe | Win Rate | N trades | Return | Equity finale |
|---|---|---|---|---|---|
| TCN+Mamba (teacher) | **-21.05** | 38.1% | 21 | -3.3% | $9,670 |
| iTransformer (student) | **-13.79** | 46.9% | 32 | -3.2% | $9,685 |
| NHits *(file stale 23-05)* | +18.71 | 64.3% | 42 | +3.7% | $10,367 |

**EN**
| Arch | Sharpe | Win Rate | N trades | Return | Final equity |
|---|---|---|---|---|---|
| TCN+Mamba (teacher) | **-21.05** | 38.1% | 21 | -3.3% | $9,670 |
| iTransformer (student) | **-13.79** | 46.9% | 32 | -3.2% | $9,685 |
| NHits *(stale file 23-05)* | +18.71 | 64.3% | 42 | +3.7% | $10,367 |

🇮🇹 **Discordanza diagnosticata:**
- Val (best epoch 2): TCN+Mamba DA **0.541**, Spearman **+0.102**
- Test set: DA **0.516** (-2.5%), Spearman **+0.023** (-77%)
- Test p-value Spearman = 0.022 → segnale debole ma statisticamente significativo
- Backtest → **Sharpe -21** → l'edge in segno si dissolve quando convertito in P&L

**EN** **Diagnosed discrepancy:**
- Val (best epoch 2): TCN+Mamba DA **0.541**, Spearman **+0.102**
- Test set: DA **0.516** (-2.5%), Spearman **+0.023** (-77%)
- Test Spearman p-value = 0.022 → weak but statistically significant signal
- Backtest → **Sharpe -21** → the sign edge dissolves once translated into P&L

🇮🇹 **⚠ ARTIFACT AVAILABILITY (verificato 2026-06-03 00:30):** `run_all.py --distill` esegue backtest **SOLO sul teacher selezionato** (`run_all.py:803-810`: `args.arch = selected_teacher` poi `phase_backtest`). Quindi:
- ✅ `results/tcnmamba/dashboard_results.json` (mtime 00:16) = backtest reale del distill
- ❌ `results/itransformer/dashboard_results.json` (mtime 22:55 ieri) = backtest del training **manuale** iTrans pre-distillation, NON del modello distillato di stanotte
- ❌ `results/nhits/dashboard_results.json` (mtime 23 maggio) = pre-fix C-funding, completamente obsoleto

**EN** **⚠ ARTIFACT AVAILABILITY (verified 2026-06-03 00:30):** `run_all.py --distill` runs the backtest **only on the selected teacher** (`run_all.py:803-810`: `args.arch = selected_teacher` then `phase_backtest`). So:
- ✅ `results/tcnmamba/dashboard_results.json` (mtime 00:16) = real backtest of the distill run
- ❌ `results/itransformer/dashboard_results.json` (mtime 22:55 yesterday) = backtest of the **manual** pre-distillation iTrans training, NOT of tonight's distilled model
- ❌ `results/nhits/dashboard_results.json` (mtime May 23) = pre C-funding fix, completely stale

🇮🇹 **Per ottenere backtest validi sugli student distillati:**
```powershell
$env:QUANTSYS_ARCH = "itransformer"; python scripts/03_backtest.py
$env:QUANTSYS_ARCH = "nhits";       python scripts/03_backtest.py
```
~1 min ciascuno. Sovrascrive `results/{arch}/dashboard_results.json`. Necessario PRIMA di trarre conclusioni sulla performance distillation.

**EN** **To get valid backtests on the distilled students:**
```powershell
$env:QUANTSYS_ARCH = "itransformer"; python scripts/03_backtest.py
$env:QUANTSYS_ARCH = "nhits";       python scripts/03_backtest.py
```
~1 min each. Overwrites `results/{arch}/dashboard_results.json`. Required before drawing any conclusions about distillation performance.

🇮🇹 **Cause ipotizzate (ordinate per probabilità):**
1. **Distribution shift val→test forte**: il test set degli ultimi giorni cattura un regime di mercato diverso da quello del val. Conferma: Spearman crolla -77% val→test.
2. **Edge inferiore alle fee**: WHR 0.508 vs 0.5 random → edge ~0.8% per trade. Fee round-trip 0.2% + slippage stimato 0.1% = 0.3%. Edge netto residuo ~0.5% per trade → margine sottile.
3. **Signal generator non tarato per nuovi modelli 104 feat**: le soglie BUY/SELL/HOLD ereditate dal setup precedente (NHits 119 feat con Sharpe +18.7) producono troppi trade in periodi senza segnale.
4. **Bassa frequenza di trade su test**: 21-32 trade su ~10k sample = 0.2-0.3% di tempo a mercato → ogni trade pesa molto, alto rischio statistico.

**EN** **Hypothesised causes (ordered by likelihood):**
1. **Strong val→test distribution shift**: the recent-days test set captures a market regime different from the val one. Confirmation: Spearman collapses -77% val→test.
2. **Edge below fees**: WHR 0.508 vs 0.5 random → edge ~0.8% per trade. Round-trip fee 0.2% + estimated slippage 0.1% = 0.3%. Net residual edge ~0.5% per trade → thin margin.
3. **Signal generator not tuned for the new 104-feat models**: BUY/SELL/HOLD thresholds inherited from the previous setup (NHits 119 feat with Sharpe +18.7) produce too many trades during signal-less periods.
4. **Low trade frequency on test**: 21-32 trades over ~10k samples = 0.2-0.3% time in market → each trade weighs heavily, high statistical risk.

### Step diagnostici (ordine raccomandato) · Diagnostic steps (recommended order)

#### Step A — Verifica RegimeSession (priorità alta, ~7 min, indipendente) · Step A — Verify RegimeSession (high priority, ~7 min, independent)

🇮🇹 Confermare che Opzione C funzioni: nuovo regime detector intraday produce stratificazione val 33/33/33 invece di r0=100%.
```powershell
python run_all.py --arch itransformer --skip-update --skip-macro --force-download
```
Atteso nei log:
- `Stratified val: distribuzione regime: r0≈33%, r1≈33%, r2≈33%`
- `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=>0`

**EN** Confirm that Option C works: the new intraday regime detector produces a 33/33/33 val stratification instead of r0=100%.
```powershell
python run_all.py --arch itransformer --skip-update --skip-macro --force-download
```
Expected in the logs:
- `Stratified val: distribuzione regime: r0≈33%, r1≈33%, r2≈33%`
- `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=>0`

🇮🇹 Anche se il backtest sarà brutto (atteso), la verifica del fix Opzione C è separata dalla performance e va chiusa.

**EN** Even if the backtest will be ugly (expected), validating the Option C fix is independent from performance and must be closed out.

#### Step B — Investigazione distribution shift (priorità media, ~30 min) · Step B — Distribution-shift investigation (medium priority, ~30 min)

🇮🇹 Caricare i checkpoint backup `models/*/_bak_119feat_20260528/best_model.pt` e ri-eseguire il backtest sullo **stesso test set attuale** (104 feat). Tre scenari possibili:

**EN** Load the backup checkpoints `models/*/_bak_119feat_20260528/best_model.pt` and re-run the backtest on the **same current test set** (104 feat). Three possible scenarios:

🇮🇹
- **Scenario B1**: vecchi modelli 119 feat → Sharpe negativo anche loro → **distribution shift del mercato**, non regressione di pipeline. Decisione: accettare il regime nuovo, eventualmente retrain con weight più alto sui sample recenti (recency weighting in `02_train.py`).
- **Scenario B2**: vecchi modelli 119 feat → Sharpe positivo → **regressione causata da C-funding** (le 15 feature droppate contenevano segnale che mancava nel C-funding score). Decisione: revisitare la decisione 2026-05-28 di C-funding, considerare C-minimal o sub-set delle 15.
- **Scenario B3**: vecchi modelli con shape mismatch (119 != 104) → load fallisce → riaddestrare un modello a 119 feat per confronto controllato.

**EN**
- **Scenario B1**: old 119-feat models → also negative Sharpe → **market distribution shift**, not a pipeline regression. Decision: accept the new regime, possibly retrain with higher weights on recent samples (recency weighting in `02_train.py`).
- **Scenario B2**: old 119-feat models → positive Sharpe → **regression caused by C-funding** (the 15 dropped features carried signal that the C-funding score missed). Decision: revisit the 2026-05-28 C-funding decision, consider C-minimal or a sub-set of the 15.
- **Scenario B3**: old models with shape mismatch (119 != 104) → load fails → retrain a 119-feat model for a controlled comparison.

#### Step C — Audit signal generator (priorità media, ~20 min) · Step C — Signal generator audit (medium priority, ~20 min)

🇮🇹 Cercare in `quantsys/trading/` le soglie BUY/SELL/HOLD e i parametri di sizing. Verificare se sono hardcoded da un setup precedente o si adattano al CI medio del modello. Possibili fix:
- Soglie adattive basate su `σ_pred` (entra solo se `μ_pred / σ_pred > threshold`)
- Filtro min CI lower bound > 0 (entra solo se intervallo non attraversa zero)
- Position sizing inversely proportional a σ_pred

**EN** Search `quantsys/trading/` for BUY/SELL/HOLD thresholds and sizing parameters. Check whether they are hardcoded from a previous setup or adapt to the model's average CI. Possible fixes:
- Adaptive thresholds based on `σ_pred` (enter only if `μ_pred / σ_pred > threshold`)
- Min CI lower-bound > 0 filter (enter only if the interval doesn't cross zero)
- Position sizing inversely proportional to σ_pred

#### Step D — Paper-trading dopo Stage 4 integration (priorità bassa, ~6-48h) · Step D — Paper-trading after Stage 4 integration (low priority, ~6-48h)

🇮🇹 Solo DOPO aver chiuso BLOCKER #1 Stage 4 (integrazione `LiveCandleBuffer` + `FeatureAssembler` nel `LiveEngine`). Far girare il paper-trading per 12-48h, accumulare 50-200 trade, confrontare metriche live con backtest. Se le metriche live divergono dal backtest in modo persistente → bug nel signal generator o nel matching live/training, NON nel modello.

**EN** Only AFTER closing BLOCKER #1 Stage 4 (`LiveCandleBuffer` + `FeatureAssembler` integration into `LiveEngine`). Run paper-trading for 12-48h, accumulate 50-200 trades, compare live metrics with backtest. If live metrics persistently diverge from backtest → bug in signal generator or live/training matching, NOT in the model.

🇮🇹 **Punto di ripresa per nuova sessione:**
- Output backtest catastrofico documentato qui (numeri da `results/{arch}/dashboard_results.json`)
- Step A non ancora eseguito (manca verifica `Stratified val: r0≈33%`)
- Step B non ancora eseguito (richiede ricaricare backup `_bak_119feat_20260528`)
- Step C non ancora eseguito (richiede grep su `quantsys/trading/` per signal thresholds)
- Decisione operativa pending: prima fare A (verifica) o B (investigazione)?

**EN** **Resume point for a new session:**
- Catastrophic backtest output documented here (numbers from `results/{arch}/dashboard_results.json`)
- Step A not yet executed (missing `Stratified val: r0≈33%` verification)
- Step B not yet executed (requires reloading the `_bak_119feat_20260528` backup)
- Step C not yet executed (requires grep on `quantsys/trading/` for signal thresholds)
- Operational decision pending: do A (verify) or B (investigate) first?

---

## 🟢 RESOLVED 2026-06-03 — Markov-Switching su BTC realized vol (Variante 3) implementato · 🟢 RESOLVED 2026-06-03 — Markov-Switching on BTC realized vol (Variant 3) implemented

🇮🇹 **Stato:** proposta, non implementata. Origine: 2026-06-02, training iTransformer mostra `Stratified val: distribuzione regime: r0=10056 (100%)` su tutte le validation → il detector attuale è degenere (collassa a 1 cluster) e la diagnostica per-regime (`val_nll spread=0.000`) non porta informazione.

**EN** **Status:** proposal, not implemented. Origin: 2026-06-02, iTransformer training shows `Stratified val: distribuzione regime: r0=10056 (100%)` on every validation → the current detector is degenerate (collapses to 1 cluster) and the per-regime diagnostics (`val_nll spread=0.000`) carry no information.

🇮🇹 **Problema strutturale (non solo bug del detector):**
- Hamilton 1989 su macro features FRED + yFinance daily → i regimi cambiano ogni **mesi**. Il trading è a 1-min con orizzonte h=30. Mismatch di 4-5 ordini di grandezza tra la scala del regime detector e la scala operativa del modello.
- Verificato in `scripts/02_train.py:577-1096`: il regime label NON è una feature di input al modello, è usato solo per stratificazione val + log diagnostico. Quindi il "bug" attuale è cosmetico, ma anche fixandolo (n_regimes 3→2) il valore aggiunto per trading 1-min resta basso.
- Le 90 macro features grezze + `MacroEncoder` (16-dim) già danno al modello il "regime macro" implicito — il label aggregato è ridondante.

**EN** **Structural problem (not just a detector bug):**
- Hamilton 1989 on FRED + yFinance daily macro features → regimes change every **months**. Trading runs at 1-min with horizon h=30. A 4-5 order-of-magnitude mismatch between regime-detector scale and the model's operational scale.
- Verified in `scripts/02_train.py:577-1096`: the regime label is NOT an input feature to the model, it is used only for val stratification + diagnostic logging. So the current "bug" is cosmetic, but even fixing it (n_regimes 3→2) the added value for 1-min trading stays low.
- The 90 raw macro features + `MacroEncoder` (16-dim) already give the model the implicit "macro regime" — the aggregated label is redundant.

🇮🇹 **Decisione Opzione C — regime detector intraday su BTC:**

**EN** **Decision Option C — intraday regime detector on BTC:**

🇮🇹 Sostituire il Markov-Switching su PC1 delle macro con un detector che osservi **direttamente la microstruttura BTC** a una scala coerente col timeframe trading (cambio ogni ~1-4h, non mesi). Tre varianti candidate, ordinate per costo crescente:

**EN** Replace the Markov-Switching on macro PC1 with a detector that observes **BTC microstructure directly** at a scale consistent with the trading timeframe (switching every ~1-4h, not months). Three candidate variants, ordered by increasing cost:

🇮🇹
1. **Session regime (più semplice)**: lookup su `hour` UTC → {Asia 00-08, EU 08-16, US 16-24}. Tre regimi, deterministico, costo zero, ground truth della letteratura sui crypto (Asia low-vol, EU/US high-vol).
2. **Volatility regime via threshold**: percentile rolling 4h della realized volatility → {low / mid / high}. Cambia 5-10 volte al giorno, match perfetto con h=30. Implementazione semplice (no EM, no PCA).
3. **HMM/Markov-Switching su BTC**: stesso engine attuale ma osservato su realized volatility intraday (rolling 1h log_ret²) anziché su PC1 macro. Cambia 3-8 volte/giorno. Riusa l'infrastruttura `RegimeMarkovSwitching` esistente, cambia solo la feature di input.

**EN**
1. **Session regime (simplest)**: lookup on UTC `hour` → {Asia 00-08, EU 08-16, US 16-24}. Three regimes, deterministic, zero cost, the literature's ground truth for crypto (low-vol Asia, high-vol EU/US).
2. **Volatility regime via threshold**: rolling 4h percentile of realized volatility → {low / mid / high}. Switches 5-10 times a day, perfect match with h=30. Simple implementation (no EM, no PCA).
3. **HMM/Markov-Switching on BTC**: the same engine as today but observed on intraday realized volatility (rolling 1h log_ret²) instead of macro PC1. Switches 3-8 times per day. Reuses the existing `RegimeMarkovSwitching` infrastructure, only the input feature changes.

🇮🇹 **Razionale per la scelta finale:** partire dalla 1 (session) come baseline, misurare lo spread NLL per-regime sul val. Se spread > 0.05 NLL → il regime è informativo, vale la pena passare alla 2/3. Se ancora 0.000 → il modello non discrimina tra regimi (segnale uniforme), e tanto vale rimuovere il regime detector del tutto.

**EN** **Rationale for the final choice:** start from variant 1 (session) as the baseline, measure per-regime NLL spread on val. If spread > 0.05 NLL → regime is informative, worth moving to 2/3. If still 0.000 → the model doesn't discriminate between regimes (uniform signal), and the regime detector can simply be removed.

🇮🇹 **Vantaggi vs attuale:**
- Frequenza di switch coerente col timeframe h=30
- Diagnostica `val_nll per regime` torna informativa
- Stratificazione val effettiva (non più degenere r0=100%)
- Possibile features future: regime label come input al modello (oggi NON usato come feature)

**EN** **Advantages vs current:**
- Switch frequency consistent with h=30 timeframe
- `val_nll per regime` diagnostic becomes informative again
- Effective val stratification (no more degenerate r0=100%)
- Possible future feature: regime label as model input (currently NOT used as feature)

🇮🇹 **File da toccare (implementazione baseline session):**
- `quantsys/macro/regime.py` → nuova classe `RegimeIntraday` o variante `RegimeSession` (`session = floor(hour_utc / 8)`)
- `scripts/01b_download_macro.py` → fittare/serializzare il nuovo detector (probabilmente trivial, no EM)
- `scripts/02_train.py:385,577` → caricare il nuovo regime per `_load_val_regimes` e stratificazione

**EN** **Files to touch (baseline session implementation):**
- `quantsys/macro/regime.py` → new `RegimeIntraday` class or `RegimeSession` variant (`session = floor(hour_utc / 8)`)
- `scripts/01b_download_macro.py` → fit/serialize the new detector (probably trivial, no EM)
- `scripts/02_train.py:385,577` → load the new regime for `_load_val_regimes` and stratification

🇮🇹 **Validazione:**
- Dopo retrain, verificare `Stratified val: distribuzione regime` ha tutti i regimi con coverage ~25-40% ciascuno (non più 100% in r0)
- Spread `val_nll per regime` > 0 (segnale: il modello fa peggio in alcuni regimi)
- Backtest invariato o migliorato (regime detector ≠ regressione)

**EN** **Validation:**
- After retrain, verify `Stratified val: distribuzione regime` has all regimes with ~25-40% coverage each (no longer 100% in r0)
- `val_nll per regime` spread > 0 (signal: the model does worse in some regimes)
- Backtest unchanged or improved (regime detector ≠ regression)

🇮🇹 > ⚠ **Dopo implementazione, aggiornare `AVVIO.md`, `TEORIA.md` (§ "Markov-Switching"), `README.md` (e versioni `.en.md`)** con la nuova architettura del regime detector. La sezione attuale in `TEORIA.md` (`statsmodels.MarkovRegression` su PC1 macro) andrà sostituita con la descrizione del detector intraday scelto.

**EN** > ⚠ **After implementation, update `AVVIO.md`, `TEORIA.md` (§ "Markov-Switching"), `README.md` (and `.en.md` counterparts)** with the new regime-detector architecture. The current section in `TEORIA.en.md` (`statsmodels.MarkovRegression` on macro PC1) will need to be replaced with the description of the chosen intraday detector.

### 🚧 Implementation status (live tracker — aggiornato dalla sessione) · 🚧 Implementation status (live tracker — session-updated)

🇮🇹 **Approccio scelto:** Variante 1 — **regime session-based** (Asia/EU/US via `hour_utc // 8`). Baseline deterministica, costo zero, nessuna EM. Se dopo il prossimo training lo spread NLL per-regime resta ~0, si valuta variante 2 (volatility threshold) o si rimuove del tutto il detector.

**EN** **Chosen approach:** Variant 1 — **session-based regime** (Asia/EU/US via `hour_utc // 8`). Deterministic baseline, zero cost, no EM. If the next training still shows ~0 per-regime NLL spread, consider variant 2 (volatility threshold) or drop the detector entirely.

🇮🇹 **Sessione 2026-06-02 22:35 — code + docs completati via fan-out 3 subagent:**

**EN** **2026-06-02 22:35 session — code + docs completed via 3-subagent fan-out:**

🇮🇹
| Task | File | Stato | Note |
|---|---|---|---|
| Nuova classe `RegimeSession` | `quantsys/macro/regime.py:848-1003` | ✅ done | Aggiunta come "STADIO 1c", drop-in con `fit_predict_walkforward` / `save` / `load`. Smoke test 9097 righe, distribuzione 3033/3032/3032 (~33% ciascuno). `RegimeMarkovSwitching` e `RegimeHMM` intoccati come fallback |
| Switch in pipeline | `scripts/01b_download_macro.py:28,89-115,208` | ✅ done | Import + uso `RegimeSession(n_regimes=3)`. Filename output `regime_hmm.pkl` e `regime_probs.parquet` invariati (backward compat consumer `_load_val_regimes`) |
| TEORIA.md (IT) | §4 righe 76-83, §9 riga 214, diagramma ~277 | ✅ done | Sezione riscritta da capo, diagramma ASCII separa macro path da regime path |
| README.it.md (IT) | bullet riga 36, diagramma 92, tree 144/160 | ✅ done | Bullet "Rilevamento regimi" → session-based + nota fallback |
| TEORIA.en.md (EN) | §"Macro regime detection" righe 76-83, diagramma ~277 | ✅ done | Mirror IT — nuova descrizione UTC-session detector |
| README.md (EN) | bullet riga 36, diagramma 92, tree 144/160 | ✅ done | Mirror README.it.md |
| AVVIO.md (IT) | nessuna sezione regime descrittiva | ⊘ skip | File è quickstart launch — non descrive regime detector, niente da aggiornare |
| AVVIO.en.md (EN) | nessuna sezione regime descrittiva | ⊘ skip | Stesso motivo di AVVIO.md |
| Test | `tests/test_features.py:212-226` | ⊘ skip | Test esistente di `RegimeMarkovSwitching` resta valido (classe non rimossa); test per `RegimeSession` opzionale, non bloccante |
| Smoke test pipeline reale | `python scripts/01b_download_macro.py` | ✅ done 2026-06-02 22:54 | `data/regime_probs.parquet` rigenerato: 73777 righe orarie 2018-01-01→2026-06-02, distribuzione **33.3% / 33.3% / 33.3%** (24593/24592/24592). Tempo ~129s |
| Verifica end-to-end Phase 1+1b+2+3+4+5 | fan-out 3 subagent | ✅ done 2026-06-02 22:55 | Tutti gli script lanciati da `run_all.py --distill` verificati: sintassi+import+smoke OK. IC fix sanity check: `ic_mean=0.3728 ≈ spearman=0.3726` su segnale skill 30% (matematicamente coerente) |
| Retrain iTransformer di verifica | `run_all.py --arch itransformer --skip-update --skip-macro --force-download` | ✅ done 2026-06-03 | Stratif val **46% / 12% / 41%** (vs precedente collasso 100% r0), spread `val_nll` **0.19-0.30** (>> soglia 0.05 "informativo"), 5/5 modelli ensemble convergono stabilmente |

**EN**
| Task | File | Status | Notes |
|---|---|---|---|
| New `RegimeSession` class | `quantsys/macro/regime.py:848-1003` | ✅ done | Added as "STAGE 1c", drop-in with `fit_predict_walkforward` / `save` / `load`. Smoke test 9097 rows, distribution 3033/3032/3032 (~33% each). `RegimeMarkovSwitching` and `RegimeHMM` untouched as fallbacks |
| Pipeline switch | `scripts/01b_download_macro.py:28,89-115,208` | ✅ done | Import + use `RegimeSession(n_regimes=3)`. Output filenames `regime_hmm.pkl` and `regime_probs.parquet` unchanged (backward compat with `_load_val_regimes` consumer) |
| TEORIA.md (IT) | §4 lines 76-83, §9 line 214, diagram ~277 | ✅ done | Section rewritten from scratch, ASCII diagram separates macro path from regime path |
| README.it.md (IT) | bullet line 36, diagram 92, tree 144/160 | ✅ done | Bullet "Rilevamento regimi" → session-based + fallback note |
| TEORIA.en.md (EN) | §"Macro regime detection" lines 76-83, diagram ~277 | ✅ done | Mirror of IT — new UTC-session detector description |
| README.md (EN) | bullet line 36, diagram 92, tree 144/160 | ✅ done | Mirror of README.it.md |
| AVVIO.md (IT) | no descriptive regime section | ⊘ skip | File is quickstart launch — does not describe the regime detector, nothing to update |
| AVVIO.en.md (EN) | no descriptive regime section | ⊘ skip | Same reason as AVVIO.md |
| Tests | `tests/test_features.py:212-226` | ⊘ skip | Existing `RegimeMarkovSwitching` test stays valid (class not removed); a `RegimeSession` test is optional, non-blocking |
| Real-pipeline smoke test | `python scripts/01b_download_macro.py` | ✅ done 2026-06-02 22:54 | `data/regime_probs.parquet` regenerated: 73777 hourly rows 2018-01-01→2026-06-02, distribution **33.3% / 33.3% / 33.3%** (24593/24592/24592). Time ~129s |
| End-to-end verification Phase 1+1b+2+3+4+5 | 3-subagent fan-out | ✅ done 2026-06-02 22:55 | All scripts run by `run_all.py --distill` verified: syntax+imports+smoke OK. IC fix sanity check: `ic_mean=0.3728 ≈ spearman=0.3726` on a 30% skill signal (mathematically consistent) |
| Verification iTransformer retrain | `run_all.py --arch itransformer --skip-update --skip-macro --force-download` | ✅ done 2026-06-03 | Stratified val **46% / 12% / 41%** (vs previous 100% r0 collapse), `val_nll` spread **0.19-0.30** (>> 0.05 "informative" threshold), 5/5 ensemble models converge stably |

🇮🇹 **Punti di ripresa per la prossima sessione (in caso di out-of-tokens):**
1. **Step successivo:** lanciare `python scripts/01b_download_macro.py` per rigenerare `data/regime_probs.parquet` (1-2 minuti). NB: scarica anche dati FRED/yFinance — se quelli sono già aggiornati, è OK rifare il download (idempotente).
2. **Dopo step 1:** rilanciare training iTransformer per verificare. Comando: `python run_all.py --arch itransformer --skip-update --skip-macro --force-download` (~7 min). Cercare la riga `Stratified val: distribuzione regime` nei log — atteso ~33%/33%/33%, NON più `r0=100%`.
3. **Validazione finale:** cercare la riga `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=Z` ogni 5 epoche. Se `spread > 0.05`, il regime è informativo e vale la pena tenerlo. Se ancora `spread ≈ 0`, considerare variante 2 (volatility threshold) o rimozione totale.
4. **Test unitario opzionale:** aggiungere `test_regime_session.py` in `tests/` con verifica determinismo + distribuzione bilanciata (non bloccante per merge).

**EN** **Resume points for the next session (in case of out-of-tokens):**
1. **Next step:** run `python scripts/01b_download_macro.py` to regenerate `data/regime_probs.parquet` (1-2 minutes). NB: it also downloads FRED/yFinance data — if those are already up to date, redoing the download is fine (idempotent).
2. **After step 1:** rerun iTransformer training to verify. Command: `python run_all.py --arch itransformer --skip-update --skip-macro --force-download` (~7 min). Look for `Stratified val: distribuzione regime` in the logs — expected ~33%/33%/33%, NO longer `r0=100%`.
3. **Final validation:** look for `↳ val_nll per regime: r0=+X r1=+Y r2=+Z spread=Z` every 5 epochs. If `spread > 0.05`, regime is informative and worth keeping. If still `spread ≈ 0`, consider variant 2 (volatility threshold) or full removal.
4. **Optional unit test:** add `test_regime_session.py` in `tests/` verifying determinism + balanced distribution (non-blocking for merge).

🇮🇹 **Decisione di rollback:** se la nuova `RegimeSession` non migliora lo spread NLL per-regime entro 1 training completo, considerare variante 2 (volatility threshold percentile rolling 4h). La classe `RegimeMarkovSwitching` resta nel codice come fallback (non rimuovere).

**EN** **Rollback decision:** if the new `RegimeSession` does not improve per-regime NLL spread within one full training, consider variant 2 (rolling 4h volatility threshold). The `RegimeMarkovSwitching` class stays in the codebase as fallback (do not remove).

🇮🇹 **Closure 2026-06-03:**

**EN** **Closure 2026-06-03:**

🇮🇹
- ✅ **Variante 3 implementata**: nuova classe `RegimeMarkovBTC` in `quantsys/macro/regime.py` (Markov-Switching Hamilton 1989 su realized volatility BTC oraria + PCA expanding window, ~65-73% varianza spiegata). `scripts/01b_download_macro.py` ora la usa al posto di `RegimeSession`. File output (`data/regime_probs.parquet`) e schema invariati per backward compat con `_load_val_regimes`.
- ✅ **3 regimi data-driven emersi** su ~9100 ore di BTC (post burn-in 30gg): **R0 Quiet ~42%** (σ²(PC1)=0.56, drift≈0, P(stay)=89%), **R1 Trending ~18%** (σ²=0.12, drift=+0.08, P(stay)=92%), **R2 Stress ~40%** (σ²=3.79, drift=−0.12, P(stay)=79%, high vol + dump bias). Switch tipico 3-8 volte/giorno, coerente con h=30.
- ✅ **Stratificazione val non più degenere**: retrain iTransformer (5/5 ensemble) mostra distribuzione **46% / 12% / 41%** (vs precedente collasso 100% in r0). Spread `val_nll` per regime **0.19-0.30** stabile (>>0.05 soglia "informativo") → il regime è effettivamente informativo per il modello.
- ✅ **Doc + memoria aggiornati**: `TEORIA.md` + `TEORIA.en.md` (sezione Markov-Switching riscritta), `README.md` + `README.it.md` (bullet "Rilevamento regimi" aggiornato), memoria sessione `session_2026_06_03_markov_btc.md`.
- ⊘ **Decisione di rollback non più applicabile**: la validazione è superata (variante 3 produce spread NLL >> 0.05 e stratificazione bilanciata), nessun fallback a varianti 1/2 necessario. `RegimeMarkovSwitching` e `RegimeSession` restano nel codice come classi alternative ma non più nel path di produzione.

**EN**
- ✅ **Variant 3 implemented**: new `RegimeMarkovBTC` class in `quantsys/macro/regime.py` (Hamilton 1989 Markov-Switching on BTC hourly realized volatility + expanding-window PCA, ~65-73% variance explained). `scripts/01b_download_macro.py` now uses it instead of `RegimeSession`. Output file (`data/regime_probs.parquet`) and schema unchanged for backward compat with `_load_val_regimes`.
- ✅ **3 data-driven regimes emerged** on ~9100 hours of BTC (post 30d burn-in): **R0 Quiet ~42%** (σ²(PC1)=0.56, drift≈0, P(stay)=89%), **R1 Trending ~18%** (σ²=0.12, drift=+0.08, P(stay)=92%), **R2 Stress ~40%** (σ²=3.79, drift=−0.12, P(stay)=79%, high vol + dump bias). Typical switching 3-8 times/day, consistent with h=30.
- ✅ **Val stratification no longer degenerate**: iTransformer retrain (5/5 ensemble) shows distribution **46% / 12% / 41%** (vs previous 100% r0 collapse). Per-regime `val_nll` spread **0.19-0.30** stable (>> 0.05 "informative" threshold) → regime is effectively informative for the model.
- ✅ **Docs + memory updated**: `TEORIA.md` + `TEORIA.en.md` (Markov-Switching section rewritten), `README.md` + `README.it.md` (bullet "Regime detection" updated), session memory `session_2026_06_03_markov_btc.md`.
- ⊘ **Rollback decision no longer applicable**: validation passed (variant 3 yields NLL spread >> 0.05 and balanced stratification), no fallback to variants 1/2 needed. `RegimeMarkovSwitching` and `RegimeSession` remain in the codebase as alternative classes but no longer on the production path.

---

## ✅ BLOCKER #1 — Allineamento feature live↔training (Stage 2-5) — RISOLTO 2026-06-05 · ✅ BLOCKER #1 — Live↔training feature alignment (Stage 2-5) — RESOLVED 2026-06-05

🇮🇹 **Stato:** Stage 1-5 **DONE**. Parity di codice (feature **e** segnale) verificata bit-perfect — vedi "Stage 5" in fondo. Residuo solo OPERATIVO: smoke test WS reale + avvio paper-trading. ⚠ I segnali paper ora riflettono il backtest, ma il backtest è negativo OOS (edge soglia/rank esaurito): il paper-trading serve ad accumulare trade reali, senza aspettativa di Sharpe>0 a priori.

**EN** **Status:** Stages 1-5 **DONE**. Code parity (feature **and** signal) verified bit-perfect — see "Stage 5" below. Only operational remainder: real WS smoke test + paper-trading start. ⚠ Paper signals now reflect the backtest, but the backtest is negative OOS (threshold/rank edge exhausted): paper-trading is to accumulate real OOS trades, with no a-priori expectation of Sharpe>0.

🇮🇹 **Problema (verificato 2026-06-02 con `scripts/99_replay_live_vs_training.py`):** il backtest usa il `FeatureBuilder` filtrato C-funding (**104 feature** post Stage 2); il live engine (`LiveFeatureBuffer._compute_features` in `scripts/04_live_signals.py`) ne costruisce a mano **solo 39** in ordine diverso, con normalizzazione median/IQR per-window (non il `RobustScaler` del `pipeline_state`), e `_predict` fa pad/truncate posizionale cieco. Tre disallineamenti sovrapposti (conteggio + ordine + scala) → input live di fatto scorrelati dal training. **I segnali del paper-trading attuale NON riflettono il backtest.**

**EN** **Problem (verified 2026-06-02 with `scripts/99_replay_live_vs_training.py`):** the backtest uses the C-funding-filtered `FeatureBuilder` (**104 features** post Stage 2); the live engine (`LiveFeatureBuffer._compute_features` in `scripts/04_live_signals.py`) builds **only 39** by hand in a different order, with per-window median/IQR normalization (not the `pipeline_state`'s `RobustScaler`), and `_predict` does blind positional pad/truncate. Three overlapping mismatches (count + order + scale) → live inputs effectively uncorrelated from training. **Current paper-trading signals do NOT reflect the backtest.**

🇮🇹 Causa di fondo: il `LiveFeatureBuffer` ridotto esiste perché il `FeatureBuilder` completo richiede storia lunga (ATH/ATL 365d, momentum 90d, frac-diff, vp_*_long) non disponibile nel buffer rolling live (260 candele).

**EN** Root cause: the reduced `LiveFeatureBuffer` exists because the full `FeatureBuilder` requires long history (ATH/ATL 365d, momentum 90d, frac-diff, vp_*_long) not available in the live rolling buffer (260 candles).

### Decisione (2026-05-28): Opzione C-funding (~104 feature) · Decision (2026-05-28): Option C-funding (~104 features)

🇮🇹 **Razionale dall'esperimento di permutation importance** (ensemble eterogeneo, 2500 finestre val, permutazione per gruppo/feature): le 23 feature "live-hostile" (lookback > buffer: 30/90/365d, frac_diff_*, vp_*_long, vp_poc_convergence, funding) hanno **ROI ≤ 0** per il modello a h=30: permutarle in blocco *migliora* leggermente le metriche (DA 0.529→0.532, Spearman 0.069→0.076). Unica eccezione: le feature **funding** (`momentum_x_funding` +0.0042, `funding_rate_dev` +0.0028).

**EN** **Rationale from permutation importance** (heterogeneous ensemble, 2500 val windows, permutation per group/feature): the 23 "live-hostile" features (lookback > buffer: 30/90/365d, frac_diff_*, vp_*_long, vp_poc_convergence, funding) have **ROI ≤ 0** for the h=30 model: bulk-permuting them *slightly improves* metrics (DA 0.529→0.532, Spearman 0.069→0.076). Sole exception: the **funding** features (`momentum_x_funding` +0.0042, `funding_rate_dev` +0.0028).

🇮🇹 **Set C-funding** = single source of truth condiviso training/live:
- Droppa 15 feature live-incompatibili (90d, 365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`).
- Mantiene 30d + funding (ROI positivo, calcolabili in live via ring 30d ~170 KB e poll funding Binance).
- Risultato atteso: ~104 feature totali (vs 119 attuali).

**EN** **C-funding set** = single source of truth shared by training/live:
- Drops 15 live-incompatible features (90d, 365d, `frac_diff_*`, `vp_*_long`, `vp_poc_convergence`, `momentum_7d/90d`).
- Keeps 30d + funding (positive ROI, computable live via a 30d ring buffer ~170 KB and a Binance funding poll).
- Target: ~104 total features (vs 119 before).

🇮🇹 > Lo schema "ibrido completo" che manteneva tutte le 30/90/365d in live era documentato come alternativa ma **non raccomandato dai dati** (ROI negativo del tier long) — definitivamente scartato.

**EN** > The "full hybrid" scheme that kept all 30/90/365d features in live was documented as an alternative but **not recommended by the data** (negative ROI on the long tier) — definitively discarded.

### Stage 1 — codice ✅ DONE · Stage 1 — code ✅ DONE

🇮🇹 `LIVE_DROP_FEATURES` (15 feature) in `quantsys/features/__init__.py`, filtrato in `scripts/01_download_data.py` (`feat_cols = [c for c in feat_cols if c not in LIVE_DROP_FEATURES]`).

**EN** `LIVE_DROP_FEATURES` (15 features) in `quantsys/features/__init__.py`, filtered in `scripts/01_download_data.py` (`feat_cols = [c for c in feat_cols if c not in LIVE_DROP_FEATURES]`).

### Stage 2 — Rigenerazione dataset a 104 feat ✅ DONE 2026-06-02 · Stage 2 — Dataset regeneration at 104 feat ✅ DONE 2026-06-02

🇮🇹 Eseguita in automatico dentro `run_all.py --distill`: il dataset è stato rigenerato a `(80390, 120, 104)` train + `(10049, 120, 104)` val + `(10049, 120, 104)` test, con il filtro C-funding correttamente applicato (15 feature droppate, verificato programmaticamente).

**EN** Performed automatically inside `run_all.py --distill`: the dataset was regenerated as `(80390, 120, 104)` train + `(10049, 120, 104)` val + `(10049, 120, 104)` test, with the C-funding filter correctly applied (15 features dropped, programmatically verified).

### Stage 3 — Retrain distill completo ✅ DONE 2026-06-02 · Stage 3 — Full distill retrain ✅ DONE 2026-06-02

🇮🇹 Eseguito nello stesso `run_all.py --distill` del 2026-06-02: tutti e 3 i modelli (iTransformer, N-HiTS, TCN+Mamba) riaddestrati da zero a 104 feature. Distillation multi-teacher applicata agli student selezionati dallo scoring automatico (vedi `models/{arch}/config.json` per i flag `distilled: true, teacher_arch: "multi-teacher"`).

**EN** Executed in the same `run_all.py --distill` of 2026-06-02: all 3 models (iTransformer, N-HiTS, TCN+Mamba) retrained from scratch at 104 features. Multi-teacher distillation applied to the students selected by automatic scoring (see `models/{arch}/config.json` for the `distilled: true, teacher_arch: "multi-teacher"` flags).

🇮🇹 > Metriche di backtest dei modelli a 104 feat: da rileggere in `results/{arch}/dashboard_results.json` dopo la conclusione del run (potrebbero essere diverse dal +18.71 Sharpe del setup a 119 feat).

**EN** > Backtest metrics for the 104-feat models: to be re-read from `results/{arch}/dashboard_results.json` after the run completes (may differ from the +18.71 Sharpe of the 119-feat setup).

### Stage 4 — Riscrittura live engine ✅ COMPLETATO (2026-06-05/06; tracker storico 2026-06-02 sotto) · Stage 4 — Live engine rewrite 🚧 IN PROGRESS (2026-06-02 23:10 session)

🇮🇹 **Decisione architetturale:** invece di duplicare la logica feature engineering in `LiveFeatureBuffer`, **riusare direttamente `quantsys/features.FeatureBuilder.build()`** sul buffer live. Single source of truth automatica → parity test garantito by-design.

**EN** **Architectural decision:** instead of duplicating feature-engineering logic in `LiveFeatureBuffer`, **directly reuse `quantsys/features.FeatureBuilder.build()`** on the live buffer. Automatic single source of truth → parity test guaranteed by design.

🇮🇹 **Razionale:**
- Il delta feature live↔training è ~65 feature (live ha 39, training ha 104). Riscrivere a mano queste 65 a parità con `FeatureBuilder` è alto rischio di drift silenzioso.
- `FeatureBuilder.build()` su 43200 righe × ~120 colonne richiede ~1-3s su CPU. Eseguito al close di ogni candela (60s budget) è ampiamente nel budget.
- Memoria: 43200 candele × 104 float32 = 18 MB. Trascurabile.
- Le feature 30d (dist_ath_30d, momentum_30d, price_vs_ma200m) richiedono 43200 candele di storia → buffer "warm" deve essere bootstrapped dal parquet storico al boot.

**EN** **Rationale:**
- The live↔training feature delta is ~65 features (live has 39, training has 104). Hand-rewriting these 65 to match `FeatureBuilder` carries high silent-drift risk.
- `FeatureBuilder.build()` on 43200 rows × ~120 columns takes ~1-3s on CPU. Run at every candle close (60s budget) it fits comfortably.
- Memory: 43200 candles × 104 float32 = 18 MB. Negligible.
- The 30d features (dist_ath_30d, momentum_30d, price_vs_ma200m) require 43200 candles of history → a "warm" buffer must be bootstrapped from the historical parquet at boot.

🇮🇹 **Architettura nuovo live engine:**

**EN** **New live engine architecture:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ LiveCandleBuffer (50,000 candele OHLCV grezze, ring buffer)         │
│  ├─ bootstrap: legge raw_candles.parquet[-50000:] al boot           │
│  └─ append(candle): push new, pop old (FIFO maxlen)                  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FundingRatePoller (deque(maxlen=30d ÷ 8h = 90), poll ogni 1h)       │
│  └─ usa quantsys.data.fetch_funding_rate                            │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ FeatureAssembler (chiamato al close di ogni candela)                │
│  1. df = pd.DataFrame(LiveCandleBuffer.tail(43200))                  │
│  2. df_with_funding = merge_asof(df, funding_history)                │
│  3. feat_df = FeatureBuilder.build(df, fit=False, normalize=True)    │
│  4. Verifica feat_df.columns == PipelineState.feature_names (HARD-FAIL)│
│  5. Filtra LIVE_DROP_FEATURES (già in build)                         │
│  6. Estrai window[-120:, :] → (120, 104) np.ndarray                  │
└──────────────────────────────────────────────────────────────────────┘
```

🇮🇹 **File da toccare:**
- `quantsys/features/__init__.py` → esporre `get_canonical_feature_names(npz_path)` come single source of truth
- `quantsys/utils/__init__.py` `PipelineState` → aggiungere `feature_names: list[str]` attribute (persistito in pickle)
- `scripts/04_live_signals.py` → sostituire `LiveFeatureBuffer` con `LiveCandleBuffer` + `FeatureAssembler` + integrazione `FundingRatePoller`; rimuovere `_pad_or_truncate` da `_predict`
- `tests/test_live_training_parity.py` → nuovo: parity test (live output == FeatureBuilder su finestra storica con tolleranza 1e-6)
- `scripts/99_replay_live_vs_training.py` → aggiornare per usare nuovo engine

**EN** **Files to touch:**
- `quantsys/features/__init__.py` → expose `get_canonical_feature_names(npz_path)` as single source of truth
- `quantsys/utils/__init__.py` `PipelineState` → add `feature_names: list[str]` attribute (persisted in pickle)
- `scripts/04_live_signals.py` → replace `LiveFeatureBuffer` with `LiveCandleBuffer` + `FeatureAssembler` + `FundingRatePoller` integration; remove `_pad_or_truncate` from `_predict`
- `tests/test_live_training_parity.py` → new: parity test (live output == FeatureBuilder on historical window with 1e-6 tolerance)
- `scripts/99_replay_live_vs_training.py` → update to use the new engine

### 📋 Stage 4 implementation tracker (snapshot storico 2026-06-02 — superato, vedi banner sotto) · 📋 Stage 4 implementation tracker (historical snapshot 2026-06-02 — superseded, see banner below)

🇮🇹 **Sessione attiva:** 2026-06-02 23:10 (parallela al distill in corso, GPU non interferita perché live engine è solo CPU)

**EN** **Active session:** 2026-06-02 23:10 (parallel to the ongoing distill, GPU unaffected since live engine is CPU-only)

🇮🇹 > ⚠ **Questo tracker è uno snapshot storico del 2026-06-02.** Superato il 2026-06-05/06: gli step **4.6, 4.7, 4.10 sono DONE** e lo **Stage 5 (parity feature+segnale) è DONE → BLOCKER #1 RISOLTO** (vedi header "✅ BLOCKER #1 ... RISOLTO" sopra e la sezione Stage 5 sotto). Lo smoke test live è passato; aggiunto catch-up contiguo candele (A1.1). Resta solo **4.4 FundingRatePoller**, come miglioria *minore* (funding cambia ogni 8h, ffill'd → workaround da disco adeguato).

**EN** > ⚠ **This tracker is a 2026-06-02 historical snapshot.** Superseded 2026-06-05/06: steps **4.6, 4.7, 4.10 are DONE** and **Stage 5 (feature+signal parity) is DONE → BLOCKER #1 RESOLVED** (see the "✅ BLOCKER #1 ... RESOLVED" header above and the Stage 5 section below). The live smoke test passed; an A1.1 contiguous candle catch-up was added. Only **4.4 FundingRatePoller** remains, as a *minor* improvement (funding changes every 8h, ffilled → the disk workaround is adequate).

🇮🇹
| Step | File | Stato | Note di ripresa |
|---|---|---|---|
| 4.1 — Esporre `feature_names` canonico | `quantsys/features/__init__.py:13-58` | ✅ done 2026-06-02 23:25 | `get_canonical_feature_names(npz_path)` aggiunta. lru_cache(maxsize=4). Hard-fail su FileNotFoundError/KeyError. Smoke test: 104 nomi corretti, cache attiva (2a call <0.001ms), zero overlap con LIVE_DROP_FEATURES. |
| 4.2 — `PipelineState.feature_names` | `quantsys/utils/__init__.py:152` | ⊘ skip | Verificato 2026-06-02 23:20: `PipelineState.feature_cols` ha 121 elementi (pre-filter, include `LIVE_DROP_FEATURES` + `target_ret`/`target_dir`), `scale_cols` ha 105 (104+target_ret). Non è la single source of truth canonica. Non aggiungere attributo nuovo — il NPZ resta autoritativo, PipelineState fornisce `scaler` + `clip_lo_/hi_` + `n_dynamic_features` + macro state. |
| 4.3 — `LiveCandleBuffer` | `scripts/04_live_signals.py:97-185` | ✅ done 2026-06-02 23:18 | Smoke test: bootstrap 50000 candele da parquet, to_dataframe(120) shape (120,9), append FIFO funziona, default fields=0 per campi mancanti. Tz-naive UTC come richiesto da FeatureBuilder. |
| 4.4 — `FundingRatePoller` | `scripts/04_live_signals.py` (nuova classe) | ⏳ pending | Thread/asyncio task che chiama `quantsys.data.fetch_funding_rate` ogni 1h. Deque(maxlen=90) per 30d × 3 rates/giorno. Esponi `to_dataframe()` per merge. **Workaround temporaneo**: leggere `data/funding_rate.parquet` da disk al boot (già funziona, vedi test 4.5). Poller serve solo per refresh real-time. |
| 4.5 — `FeatureAssembler` | `scripts/04_live_signals.py:188-308` | ✅ done 2026-06-02 23:22 | **End-to-end OK**: compute_window(120) produce (120, 104) float32, no NaN, no Inf. Hard-fail su feature mancanti. Tz-normalization di funding_df dentro compute_window. Stats output: mean=-0.05, std=1.73, range [-24.8, +25.8] (consistente con RobustScaler+clip del training). |
| 4.6 — Sostituire in `LiveEngine.__init__` | `scripts/04_live_signals.py` | ⏳ pending | Rimuovere `self.buffer = LiveFeatureBuffer(...)`, sostituire con i 3 nuovi componenti. Aggiornare loop principale. **NON cancellare** la classe `LiveFeatureBuffer` vecchia (commento "DEPRECATED — legacy 39-feature implementation"). |
| 4.7 — Rimuovere pad/truncate | `scripts/04_live_signals.py:982-988` | ⏳ pending | In `_predict()`, sostituire il blocco `if n_live < n_model: pad ...` con `assert window.shape[1] == n_model, "feature mismatch"`. Il window arriva già a 104. |
| 4.8 — Parity test | `tests/test_live_training_parity.py` | ✅ done 2026-06-02 23:30 | **4/4 test passano in 12.8s**. (1) Assembler output == direct FeatureBuilder output con `max abs diff < 1e-5`, (2) canonical order stabile, (3) zero overlap con LIVE_DROP_FEATURES, (4) hard-fail su funding=None. Parity matematicamente verificata. |
| 4.9 — Aggiornare replay script | `scripts/99_replay_live_vs_training.py` | ✅ done 2026-06-02 23:33 | Riscritto da zero per usare `LiveCandleBuffer` + `FeatureAssembler`. Run output: **Max diff: 0.000e+00** (parity perfetta bit-identica). Era 3 disallineamenti, ora 0. |
| 4.10 — Smoke test live | `python scripts/04_live_signals.py` | ⏳ pending | Connessione WS, warmup completo, primo segnale entro 2 min. Verificare niente warning "feature mismatch". |
| 4.11 — Update doc `MODEL_IMPROVEMENTS.md` | questo file | ⏳ pending | Spostare Stage 4 da pending a done, aggiornare `BLOCKER #1` header con `🟢 RESOLVED` se anche Stage 5 chiuso |

**EN**
| Step | File | Status | Resume notes |
|---|---|---|---|
| 4.1 — Expose canonical `feature_names` | `quantsys/features/__init__.py:13-58` | ✅ done 2026-06-02 23:25 | `get_canonical_feature_names(npz_path)` added. lru_cache(maxsize=4). Hard-fail on FileNotFoundError/KeyError. Smoke test: 104 correct names, cache active (2nd call <0.001ms), zero overlap with LIVE_DROP_FEATURES. |
| 4.2 — `PipelineState.feature_names` | `quantsys/utils/__init__.py:152` | ⊘ skip | Verified 2026-06-02 23:20: `PipelineState.feature_cols` has 121 elements (pre-filter, includes `LIVE_DROP_FEATURES` + `target_ret`/`target_dir`), `scale_cols` has 105 (104+target_ret). Not the canonical single source of truth. Do not add a new attribute — the NPZ remains authoritative, PipelineState provides `scaler` + `clip_lo_/hi_` + `n_dynamic_features` + macro state. |
| 4.3 — `LiveCandleBuffer` | `scripts/04_live_signals.py:97-185` | ✅ done 2026-06-02 23:18 | Smoke test: bootstrap of 50000 candles from parquet, to_dataframe(120) shape (120,9), append FIFO works, default fields=0 for missing fields. Tz-naive UTC as required by FeatureBuilder. |
| 4.4 — `FundingRatePoller` | `scripts/04_live_signals.py` (new class) | ⏳ pending | Thread/asyncio task calling `quantsys.data.fetch_funding_rate` every 1h. Deque(maxlen=90) for 30d × 3 rates/day. Expose `to_dataframe()` for merge. **Temporary workaround**: read `data/funding_rate.parquet` from disk at boot (already works, see test 4.5). The poller is only needed for real-time refresh. |
| 4.5 — `FeatureAssembler` | `scripts/04_live_signals.py:188-308` | ✅ done 2026-06-02 23:22 | **End-to-end OK**: compute_window(120) produces (120, 104) float32, no NaN, no Inf. Hard-fail on missing features. Tz-normalization of funding_df inside compute_window. Output stats: mean=-0.05, std=1.73, range [-24.8, +25.8] (consistent with training RobustScaler+clip). |
| 4.6 — Swap in `LiveEngine.__init__` | `scripts/04_live_signals.py` | ⏳ pending | Remove `self.buffer = LiveFeatureBuffer(...)`, replace with the 3 new components. Update the main loop. **Do NOT delete** the legacy `LiveFeatureBuffer` class (comment "DEPRECATED — legacy 39-feature implementation"). |
| 4.7 — Remove pad/truncate | `scripts/04_live_signals.py:982-988` | ⏳ pending | In `_predict()`, replace the `if n_live < n_model: pad ...` block with `assert window.shape[1] == n_model, "feature mismatch"`. The window already arrives at 104. |
| 4.8 — Parity test | `tests/test_live_training_parity.py` | ✅ done 2026-06-02 23:30 | **4/4 tests pass in 12.8s**. (1) Assembler output == direct FeatureBuilder output with `max abs diff < 1e-5`, (2) canonical order stable, (3) zero overlap with LIVE_DROP_FEATURES, (4) hard-fail on funding=None. Parity mathematically verified. |
| 4.9 — Update replay script | `scripts/99_replay_live_vs_training.py` | ✅ done 2026-06-02 23:33 | Rewritten from scratch to use `LiveCandleBuffer` + `FeatureAssembler`. Run output: **Max diff: 0.000e+00** (bit-perfect parity). Was 3 mismatches, now 0. |
| 4.10 — Live smoke test | `python scripts/04_live_signals.py` | ⏳ pending | WS connection, full warmup, first signal within 2 min. Verify no "feature mismatch" warnings. |
| 4.11 — Update doc `MODEL_IMPROVEMENTS.md` | this file | ⏳ pending | Move Stage 4 from pending to done, update `BLOCKER #1` header to `🟢 RESOLVED` if Stage 5 also closed |

🇮🇹 **Punti di ripresa per session reset:**
- **Se sei a step 4.1-4.2**: lavoro non distruttivo, può riprendere da qualsiasi punto
- **Se sei a 4.3-4.7**: il `scripts/04_live_signals.py` è in stato intermedio — la vecchia classe `LiveFeatureBuffer` va lasciata in piedi finché tutti i nuovi componenti sono testati. Prima di committare, verificare che lo script sia almeno importabile (`python -m py_compile`)
- **Se sei a 4.8-4.10**: testing-only, sicuro
- **Verifica finale post-implementazione**: lo script `scripts/99_replay_live_vs_training.py` deve produrre output "✅ 0 mismatch"

**EN** **Resume points for session reset:**
- **If at step 4.1-4.2**: non-destructive work, can resume from anywhere
- **If at 4.3-4.7**: `scripts/04_live_signals.py` is in an intermediate state — the legacy `LiveFeatureBuffer` class must stay in place until all new components are tested. Before committing, verify the script at least imports (`python -m py_compile`)
- **If at 4.8-4.10**: testing-only, safe
- **Final post-implementation verification**: `scripts/99_replay_live_vs_training.py` must produce "✅ 0 mismatches"

🇮🇹 **Stato attuale (aggiornato 2026-06-02 23:35):**
- ✅ Componenti core implementati: `get_canonical_feature_names`, `LiveCandleBuffer`, `FeatureAssembler`
- ✅ Parity test verificata matematicamente: **Max diff 0.000e+00** sul replay 50k candele
- ✅ 4/4 unit test in `tests/test_live_training_parity.py` passano in 12.8s
- ✅ Script `99_replay_live_vs_training.py` aggiornato: prima 3 disallineamenti → ora 0
- **PROSSIMI STEP (per nuova sessione):**
  1. **4.6 — Integrare in `LiveEngine.__init__`**: sostituire `self.buffer = LiveFeatureBuffer(...)` con i 3 nuovi componenti. Il main loop in `LiveEngine._on_candle_close()` (o nome simile) chiama `compute_window()` su `FeatureAssembler` invece di `get_window()` su `LiveFeatureBuffer`. Cercare con grep `self.buffer` per trovare tutti i call site.
  2. **4.7 — Rimuovere `_pad_or_truncate`**: nel metodo `_predict` (riga ~964-988) sostituire il blocco `if n_live < n_model: pad ...` con `assert window.shape[1] == 104, "feature mismatch"`. Window arriva già a 104 dal nuovo path.
  3. **4.4 — `FundingRatePoller`**: implementare poller con asyncio task + `quantsys.data.fetch_funding_rate` ogni 1h. Per ora workaround: `LiveEngine.__init__` legge `data/funding_rate.parquet` al boot e ne passa una copia a `FeatureAssembler.compute_window()`.
  4. **4.10 — Smoke test live**: `python scripts/04_live_signals.py` → connessione WS, warmup, primo segnale entro 2 min. Verificare niente warning "feature mismatch".
  5. **4.11 — Update doc finale**: marcare `BLOCKER #1 ✅ DONE` e aggiornare TEORIA.md/AVVIO.md/README se citano "39 feature live mismatched".

**EN** **Current state (updated 2026-06-02 23:35):**
- ✅ Core components implemented: `get_canonical_feature_names`, `LiveCandleBuffer`, `FeatureAssembler`
- ✅ Parity test mathematically verified: **Max diff 0.000e+00** on 50k-candle replay
- ✅ 4/4 unit tests in `tests/test_live_training_parity.py` pass in 12.8s
- ✅ Script `99_replay_live_vs_training.py` updated: previously 3 mismatches → now 0
- **NEXT STEPS (for new session):**
  1. **4.6 — Integrate into `LiveEngine.__init__`**: replace `self.buffer = LiveFeatureBuffer(...)` with the 3 new components. The main loop in `LiveEngine._on_candle_close()` (or similarly named) calls `compute_window()` on `FeatureAssembler` instead of `get_window()` on `LiveFeatureBuffer`. Grep `self.buffer` to find all call sites.
  2. **4.7 — Remove `_pad_or_truncate`**: in `_predict` (lines ~964-988) replace the `if n_live < n_model: pad ...` block with `assert window.shape[1] == 104, "feature mismatch"`. The window already arrives at 104 from the new path.
  3. **4.4 — `FundingRatePoller`**: implement poller via asyncio task + `quantsys.data.fetch_funding_rate` every 1h. For now workaround: `LiveEngine.__init__` reads `data/funding_rate.parquet` at boot and passes a copy to `FeatureAssembler.compute_window()`.
  4. **4.10 — Live smoke test**: `python scripts/04_live_signals.py` → WS connection, warmup, first signal within 2 min. Verify no "feature mismatch" warnings.
  5. **4.11 — Final doc update**: mark `BLOCKER #1 ✅ DONE` and update TEORIA.md/AVVIO.md/README if they still mention "39 mismatched live features".

🇮🇹 **Files modificati in questa sessione (commit-able):**
- `quantsys/features/__init__.py` (+50 righe: `get_canonical_feature_names`)
- `scripts/04_live_signals.py` (+~220 righe: `LiveCandleBuffer` + `FeatureAssembler`; `LiveFeatureBuffer` legacy intatta con tag DEPRECATED)
- `scripts/99_replay_live_vs_training.py` (riscritto: era pad-trun check, ora parity diff)
- `tests/test_live_training_parity.py` (nuovo: 4 test, 12.8s)

**EN** **Files modified in this session (commit-ready):**
- `quantsys/features/__init__.py` (+50 lines: `get_canonical_feature_names`)
- `scripts/04_live_signals.py` (+~220 lines: `LiveCandleBuffer` + `FeatureAssembler`; legacy `LiveFeatureBuffer` intact with DEPRECATED tag)
- `scripts/99_replay_live_vs_training.py` (rewritten: was pad-trunc check, now parity diff)
- `tests/test_live_training_parity.py` (new: 4 tests, 12.8s)

---

🇮🇹 **Seeding all'avvio:** caricare gli ultimi 30g di klines 1m (paginazione Binance, una-tantum, cache locale) o riusare `data/raw_candles.parquet`.

**EN** **Startup seeding:** load the last 30 days of 1m klines (Binance pagination, one-shot, with local cache) or reuse `data/raw_candles.parquet`.

### Stage 5 — Parity test + replay backtest (gate go/no-go) ✅ DONE 2026-06-05 · Stage 5 — Parity test + replay backtest (go/no-go gate) ✅ DONE 2026-06-05

🇮🇹
1. **Gate 1 — Parity FEATURE:** vettore live (`FeatureAssembler`) vs `FeatureBuilder` diretto sulla stessa finestra storica → **max|Δ| = 0.000e+00** (`tests/test_live_training_parity.py::test_assembler_matches_direct_featurebuilder` + `scripts/99_replay_live_vs_training.py`).
2. **Gate 2 — Parity SEGNALE (replay):** gli stessi due percorsi feature attraversano il nucleo di inferenza deterministico di produzione (`LiveEngine._deterministic_predict` — l'`EnsembleModel` non espone `predict_with_uncertainty`, quindi MC-dropout non scatta in live) + `SignalGenerator` → **Δμ=0, Δσ=0, side identico** (`test_signal_parity_live_vs_offline` + sezione Gate 2 dello script di replay). Suite parity 5/5, suite recent-fixes 16/16.
3. **Residuo operativo (non di codice):** smoke test WS Binance reale (Stage 4.10) + paper-trading per accumulare trade OOS.

**EN**
1. **Gate 1 — FEATURE parity:** live vector (`FeatureAssembler`) vs direct `FeatureBuilder` on the same historical window → **max|Δ| = 0.000e+00** (`tests/test_live_training_parity.py::test_assembler_matches_direct_featurebuilder` + `scripts/99_replay_live_vs_training.py`).
2. **Gate 2 — SIGNAL parity (replay):** both feature routes through the production deterministic inference core (`LiveEngine._deterministic_predict` — the `EnsembleModel` lacks `predict_with_uncertainty`, so MC-dropout never fires live) + `SignalGenerator` → **Δμ=0, Δσ=0, identical side** (`test_signal_parity_live_vs_offline` + Gate 2 section of the replay script). Parity suite 5/5, recent-fixes suite 16/16.
3. **Operational remainder (not code):** real Binance WS smoke test (Stage 4.10) + paper-trading to accumulate OOS trades.

🇮🇹 > Nota: Stage 4.6/4.7 (integrazione `LiveEngine.__init__` + rimozione `_pad_or_truncate`) risultavano **già nel codice** (tracker disallineato); confermati durante la chiusura di Stage 5. Refactor introdotto: `LiveEngine._deterministic_predict` come nucleo condiviso live↔parity-test (zero drift).

**EN** > Note: Stages 4.6/4.7 (LiveEngine.__init__ integration + `_pad_or_truncate` removal) were **already in the code** (tracker stale); confirmed while closing Stage 5. New refactor: `LiveEngine._deterministic_predict` as the shared core between live and the parity test (zero drift).

---

## 🔵 Binance Futures Testnet — Fasi 2-5 · 🔵 Binance Futures Testnet — Phases 2-5

🇮🇹 **Stato:** Fase 1 ✅ done (`.env` + `scripts/00_test_binance_testnet.py`). Fasi 2-5 pending. Tempo residuo: 8-13 ore.

**EN** **Status:** Phase 1 ✅ done (`.env` + `scripts/00_test_binance_testnet.py`). Phases 2-5 pending. Remaining effort: 8-13 hours.

🇮🇹 Obiettivo: il live engine invia ordini reali sul Futures Testnet (`testnet.binancefuture.com`) parallelamente al portfolio simulato, con riconciliazione periodica. Valida latency esecuzione reale, slippage testnet, comportamento SL/TP exchange-side, bug operativi che il backtest non copre.

**EN** Goal: the live engine sends real orders on the Futures Testnet (`testnet.binancefuture.com`) in parallel with the simulated portfolio, with periodic reconciliation. Validates real execution latency, testnet slippage, exchange-side SL/TP behaviour, operational bugs the backtest doesn't cover.

🇮🇹 > **Prerequisito:** BLOCKER #1 (Stage 2-5) risolto prima, altrimenti il testnet riceve segnali da un modello con input scorrelato.

**EN** > **Prerequisite:** BLOCKER #1 (Stage 2-5) resolved first, otherwise the testnet receives signals from a model with uncorrelated inputs.

### Fase 2 — Architettura execution layer (2-4h) · Phase 2 — Execution layer architecture (2-4h)

🇮🇹 **Nuovo package** `quantsys/execution/`:

**EN** **New package** `quantsys/execution/`:

```
quantsys/execution/
├── __init__.py          # factory create_adapter(mode, ...)
├── base.py              # ABC ExecutionAdapter
├── paper.py             # in-memory simulato (rifactor del comportamento attuale RiskManager)
└── binance_futures_testnet.py  # REST via python-binance.Client(testnet=True)
```

🇮🇹 **Interface ABC** (`base.py`):

**EN** **ABC interface** (`base.py`):

```python
class ExecutionAdapter(ABC):
    @abstractmethod
    def place_market_order(self, side: Side, qty: float) -> str: ...
    @abstractmethod
    def place_stop_market(self, side: Side, qty: float, stop_price: float) -> str: ...
    @abstractmethod
    def place_take_profit_market(self, side: Side, qty: float, tp_price: float) -> str: ...
    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...
    @abstractmethod
    def cancel_all_orders(self, symbol: str) -> None: ...
    @abstractmethod
    def get_position(self, symbol: str) -> Position | None: ...
    @abstractmethod
    def get_balance(self, asset: str = "USDT") -> float: ...
    @abstractmethod
    def get_open_orders(self, symbol: str) -> list[dict]: ...
    @abstractmethod
    def set_leverage(self, symbol: str, leverage: int) -> None: ...
```

🇮🇹 **Config nuova** in `config/default.yaml`:
```yaml
live:
  execution_mode: paper        # paper | testnet_futures
  testnet_futures:
    symbol: BTCUSDT
    margin_type: ISOLATED      # ISOLATED | CROSSED
    max_leverage: 3            # modulata da conviction; 1 = no leva
    leverage_conviction_alpha: 1.0
    # api_key/secret da env BINANCE_TESTNET_API_KEY / _SECRET (.env)
```

**EN** **New config** in `config/default.yaml`:
```yaml
live:
  execution_mode: paper        # paper | testnet_futures
  testnet_futures:
    symbol: BTCUSDT
    margin_type: ISOLATED      # ISOLATED | CROSSED
    max_leverage: 3            # conviction-modulated; 1 = no leverage
    leverage_conviction_alpha: 1.0
    # api_key/secret from env BINANCE_TESTNET_API_KEY / _SECRET (.env)
```

🇮🇹 **Leva dinamica conviction-based** (decisa 2026-05-24):
```python
def _conviction_leverage(conviction: float, max_lev: int, alpha: float = 1.0) -> int:
    """conviction=0 → 1x, conviction=0.5 → ~max_lev/2, conviction=1 → max_lev."""
    lev = 1 + (max_lev - 1) * (conviction ** alpha)
    return max(1, min(max_lev, round(lev)))
```
Chiamato in `RiskManager.open_position` PRIMA del `place_market_order`.

**EN** **Conviction-based dynamic leverage** (decided 2026-05-24):
```python
def _conviction_leverage(conviction: float, max_lev: int, alpha: float = 1.0) -> int:
    """conviction=0 → 1x, conviction=0.5 → ~max_lev/2, conviction=1 → max_lev."""
    lev = 1 + (max_lev - 1) * (conviction ** alpha)
    return max(1, min(max_lev, round(lev)))
```
Called in `RiskManager.open_position` BEFORE `place_market_order`.

### Fase 3 — Wiring RiskManager (2-3h) · Phase 3 — RiskManager wiring (2-3h)

🇮🇹 Modifiche a `quantsys/trading/__init__.py`:
- `RiskManager.__init__` accetta `execution_adapter: ExecutionAdapter | None = None` (default = paper).
- `open_position`: dopo calcolo SL/TP+size, se `self.adapter is not None`:
  1. `set_leverage(symbol, _conviction_leverage(...))`
  2. `entry_order_id = place_market_order(side, qty)`
  3. `sl_order_id = place_stop_market(opposite_side, qty, sl_price)`
  4. `tp_order_id = place_take_profit_market(opposite_side, qty, tp_price)`
  5. Persisti i 3 orderId nella `Position`.
- `update_trailing`: se SL aggiornato e adapter: `cancel_order(sl_order_id)` + nuovo `place_stop_market` + update `position.sl_order_id`.
- `close_position`: se adapter: `cancel_all_orders` (chiude SL/TP residui) + `place_market_order(opposite_side, qty)` (chiusura at-market).

**EN** Changes to `quantsys/trading/__init__.py`:
- `RiskManager.__init__` accepts `execution_adapter: ExecutionAdapter | None = None` (default = paper).
- `open_position`: after computing SL/TP+size, if `self.adapter is not None`:
  1. `set_leverage(symbol, _conviction_leverage(...))`
  2. `entry_order_id = place_market_order(side, qty)`
  3. `sl_order_id = place_stop_market(opposite_side, qty, sl_price)`
  4. `tp_order_id = place_take_profit_market(opposite_side, qty, tp_price)`
  5. Persist the 3 orderIds on the `Position`.
- `update_trailing`: if SL updated and adapter: `cancel_order(sl_order_id)` + new `place_stop_market` + update `position.sl_order_id`.
- `close_position`: if adapter: `cancel_all_orders` (closes residual SL/TP) + `place_market_order(opposite_side, qty)` (at-market close).

🇮🇹 **Casi edge:**
- Partial fill: poll fino a FILLED o cancel + market sul resto.
- Liquidation: recovery automatico (chiudi paper, log WARNING, riparti pulito).
- Rate limit: Binance Futures 1200 weight/min; un `open_position` ≈ 4 chiamate REST → max ~300 open/min teorico (sufficiente).

**EN** **Edge cases:**
- Partial fill: poll until FILLED or cancel + market on the remainder.
- Liquidation: automatic recovery (close paper, log WARNING, restart clean).
- Rate limit: Binance Futures 1200 weight/min; one `open_position` ≈ 4 REST calls → max ~300 open/min theoretical (sufficient).

### Fase 4 — Riconciliazione paper vs testnet (2-3h) · Phase 4 — Paper vs testnet reconciliation (2-3h)

🇮🇹 Nuovo modulo `quantsys/execution/reconciliation.py`:

**EN** New module `quantsys/execution/reconciliation.py`:

```python
class Reconciler:
    async def loop(self, interval_seconds: int = 30):
        while True:
            real_pos = self.adapter.get_position(symbol)
            real_balance = self.adapter.get_balance("USDT")
            paper_equity = self.paper.equity
            delta_equity = real_balance - paper_equity
            self._log({...})  # signals/reconciliation.jsonl
            if abs(delta_equity / paper_equity) > 0.005:  # >0.5% drift
                log.warning(f"Reconciliation drift: paper=${paper_equity} vs real=${real_balance}")
            await asyncio.sleep(interval_seconds)
```

🇮🇹 Output: `signals/reconciliation.jsonl` (~2880 record/giorno). Warning solo su drift > 0.5%. Integrazione in `04_live_signals.py` via `asyncio.gather`.

**EN** Output: `signals/reconciliation.jsonl` (~2880 records/day). Warning only on drift > 0.5%. Integrated into `04_live_signals.py` via `asyncio.gather`.

### Fase 5 — Test end-to-end (2-3h) · Phase 5 — End-to-end test (2-3h)

🇮🇹 **Pre-flight:**
1. `python scripts/00_test_binance_testnet.py` (già ✅)
2. Set `live.execution_mode: testnet_futures` in config
3. Run con `max_leverage: 1` (no leva) come safety net iniziale, monitor armato per: WS Binance connesso, primo segnale, primo OPEN sul testnet (verifica orderId + posizione), primo update SL trailing, primo CLOSE, riconciliazione delta < 0.5%
4. Lasciare girare 1-3h con `max_leverage: 1`. Se 3-5 trade OK end-to-end → alzare gradualmente (1 → 2 → 3).

**EN** **Pre-flight:**
1. `python scripts/00_test_binance_testnet.py` (already ✅)
2. Set `live.execution_mode: testnet_futures` in config
3. Run with `max_leverage: 1` (no leverage) as initial safety net, monitor for: Binance WS connected, first signal, first OPEN on testnet (verify orderId + position), first trailing SL update, first CLOSE, reconciliation delta < 0.5%
4. Let it run 1-3h with `max_leverage: 1`. If 3-5 trades go OK end-to-end → raise gradually (1 → 2 → 3).

🇮🇹 **Decision criteria post 24-48h live (tutti e 4 devono essere OK):**
- Drift reconciliation < 0.5% per >95% dei sample
- Slippage reale entro 2× quello backtest (`slippage_rate: 0.0003`)
- Latency totale (signal gen → ordine fillato) < 500ms
- Zero rate limit violation

**EN** **Decision criteria after 24-48h live (all 4 must pass):**
- Reconciliation drift < 0.5% on >95% of samples
- Real slippage within 2× the backtest's (`slippage_rate: 0.0003`)
- Total latency (signal gen → filled order) < 500ms
- Zero rate limit violations

🇮🇹 Se OK → paper-trading 2-4 settimane prima di considerare mainnet. Se fallisce uno → fix bug specifici prima di riprovare.

**EN** If all OK → paper-trading 2-4 weeks before considering mainnet. If any fails → fix specific bugs before retrying.

---

## 🟡 Roadmap modello — Fix #3, #4, #5, #6 · 🟡 Model roadmap — Fix #3, #4, #5, #6

🇮🇹 Tutti gated post paper-trading (paper-trading è gated post BLOCKER #1).

**EN** All gated post paper-trading (which is gated post BLOCKER #1).

🇮🇹
| # | Fix | Da | A | Effort | Beneficio atteso |
|---|---|---|---|---|---|
| 3 | `model.window_size` (T) | 120 | **240** | config + ~30% VRAM | DA ↑ 1-2%, vol cluster catturato |
| 4 | `validation.n_folds` | 3 | **5-6** | config + +50% test time | CI bootstrap walkforward più affidabili |
| 5 | Multi-timeframe (1m+5m+1h) | — | nuovo pkg `mtf/` | 6-9 settimane elapsed | DA ↑ 2-4%, contesto 24h |
| 6 | `mamba-ssm` (kernel CUDA) | pure-PyTorch | kernel ufficiale | ~1h setup + retrain | speedup TCN+Mamba 3-5× |

**EN**
| # | Fix | From | To | Effort | Expected benefit |
|---|---|---|---|---|---|
| 3 | `model.window_size` (T) | 120 | **240** | config + ~30% VRAM | DA ↑ 1-2%, vol cluster captured |
| 4 | `validation.n_folds` | 3 | **5-6** | config + +50% test time | More reliable walkforward bootstrap CI |
| 5 | Multi-timeframe (1m+5m+1h) | — | new `mtf/` pkg | 6-9 weeks elapsed | DA ↑ 2-4%, 24h context |
| 6 | `mamba-ssm` (CUDA kernel) | pure-PyTorch | official kernel | ~1h setup + retrain | TCN+Mamba speedup 3-5× |

### Fix #3 — Window size T 120 → 240

🇮🇹 **Razionale:** vol clustering BTC 1m ha half-life ~2-6h (Engle 1986, Bollerslev 1986). Con T=120 (2h) vedi solo metà del cluster. Letteratura (PatchTST Nie 2023, iTransformer Liu 2024) testa lookback 96-720; plateau intorno 192-384.

**EN** **Rationale:** vol clustering on BTC 1m has half-life ~2-6h (Engle 1986, Bollerslev 1986). With T=120 (2h) you only see half a cluster. Literature (PatchTST Nie 2023, iTransformer Liu 2024) tests lookback 96-720; plateau around 192-384.

```yaml
model:
  window_size: 240
validation:
  embargo_steps: 3000   # da 1500 (deve essere ≥ window_size + horizon)
```

🇮🇹 Poi: `python scripts/01_download_data.py` (ricostruisce npz) + `python run_all.py --distill --skip-update --skip-macro --no-browser`.

**EN** Then: `python scripts/01_download_data.py` (rebuilds the npz) + `python run_all.py --distill --skip-update --skip-macro --no-browser`.

🇮🇹 **Smoke test preliminare** su solo iTransformer per validare VRAM:
```powershell
$env:QUANTSYS_ARCH = "itransformer"
python scripts\02_train.py --n-ensemble 1
```
Se OOM su 8GB: `training.batch_size: 64 → 32` + `gradient_accumulation_steps: 2 → 4` (mantieni effective batch=128).

**EN** **Preliminary smoke test** on iTransformer only to validate VRAM:
```powershell
$env:QUANTSYS_ARCH = "itransformer"
python scripts\02_train.py --n-ensemble 1
```
If OOM on 8GB: `training.batch_size: 64 → 32` + `gradient_accumulation_steps: 2 → 4` (keep effective batch=128).

🇮🇹 Impatti: VRAM training ~+30%, tempo per epoca +30-50%, samples utilizzabili −1% (più candele wasted per warmup).

**EN** Impacts: training VRAM ~+30%, time per epoch +30-50%, usable samples −1% (more wasted candles for warmup).

### Fix #4 — Walkforward folds 3 → 5-6

🇮🇹 **Razionale:** 3 fold danno CI bootstrap larghi (Sharpe [+0.78, +74.70] su 42 trade). Letteratura finance ML (López de Prado 2018, AFML cap. 7): per crypto, **5-6 fold** sono lo standard.

**EN** **Rationale:** 3 folds give wide bootstrap CIs (Sharpe [+0.78, +74.70] on 42 trades). Finance-ML literature (López de Prado 2018, AFML ch. 7): for crypto, **5-6 folds** are the standard.

```yaml
validation:
  n_folds: 6
  embargo_steps: 3000   # se fix #3 applicato, altrimenti 1500
```

🇮🇹 Poi: `python scripts/02b_walkforward_validate.py`. **Non** richiede retrain. Tempo +50% (~30-45 min totali).

**EN** Then: `python scripts/02b_walkforward_validate.py`. **No** retrain required. Time +50% (~30-45 min total).

🇮🇹 **Controlli in `results/{arch}/walkforward_metrics.json`:**
- `da_per_fold`: tutti > 0.51, std < 0.005
- `spearman_per_fold`: tutti positivi
- `sharpe_per_fold` (bootstrap): CI esclude zero in **almeno 4 fold su 6**

**EN** **Checks in `results/{arch}/walkforward_metrics.json`:**
- `da_per_fold`: all > 0.51, std < 0.005
- `spearman_per_fold`: all positive
- `sharpe_per_fold` (bootstrap): CI excludes zero in **at least 4 out of 6 folds**

🇮🇹 Se divergenza ampia (std DA > 0.01) → modello non stabile attraverso regime → torna a più dati.

**EN** Wide divergence (std DA > 0.01) → model not stable across regimes → go back to more data.

### Fix #5 — Multi-timeframe (1m + 5m + 1h)

🇮🇹 **Stato:** miglioramento con il potenziale più alto ancora aperto. **Prerequisito:** ≥ 7-14 giorni di paper-trading data per baseline reale.

**EN** **Status:** highest-potential improvement still open. **Prerequisite:** ≥ 7-14 days of paper-trading data for a real baseline.

🇮🇹 **Architettura proposta:**
```
1m  → 120 candele (micro-pattern, esistente)
5m  →  24 candele (swing intraday, 2h contesto)
1h  →  24 candele (trend giornaliero, 24h contesto)
```
3 encoder separati della stessa famiglia, fusion finale con cross-attention o gated concat.

**EN** **Proposed architecture:**
```
1m  → 120 candles (micro-patterns, existing)
5m  →  24 candles (intraday swing, 2h context)
1h  →  24 candles (daily trend, 24h context)
```
3 separate encoders of the same family, final fusion via cross-attention or gated concat.

### Strategia: esperimento parallelo isolato in nuovo package `mtf/` · Strategy: isolated parallel experiment in a new `mtf/` package

🇮🇹 Per evitare di rompere il codice di produzione e rollback istantaneo, sviluppo in directory parallela che riusa per import il più possibile.

**EN** To avoid breaking production code and allow instant rollback, develop in a parallel directory that reuses as much as possible via imports.

```
mtf/                    # NUOVO package, isolato
├── __init__.py
├── data_builder.py     # resample 1m→5m, 1m→1h + dataset build (3 stream allineati right-bound)
├── models.py           # wrapper Quant*Mtf che compongono i modelli quantsys/
├── train.py            # training loop con DataLoader che yielda 3 tensori X
├── backtest.py
├── live_signals.py     # solo se va in produzione
└── run.py
```

🇮🇹 Dataset/modelli/risultati paralleli: `data/mtf_dataset.npz`, `models/mtf_{arch}/`, `results/mtf_{arch}/`.

**EN** Parallel dataset/models/results: `data/mtf_dataset.npz`, `models/mtf_{arch}/`, `results/mtf_{arch}/`.

🇮🇹 **Riusa** (import da `quantsys/`): loss functions (`student_t_nll`, `quantile_loss`, `direction_value_loss`), utilities (`load_config`, `setup_device`, `PipelineState`), risk manager, signal generator, macro encoder, regime detector, FeatureBuilder (eseguito 3 volte sui 3 timeframe, stesso codice).

**EN** **Reuse** (import from `quantsys/`): loss functions (`student_t_nll`, `quantile_loss`, `direction_value_loss`), utilities (`load_config`, `setup_device`, `PipelineState`), risk manager, signal generator, macro encoder, regime detector, FeatureBuilder (called 3× on the 3 timeframes, same code).

🇮🇹 **Crea nuovo** solo dove cambiano le shape: data_builder, models wrapper, training loop con 3 tensori X.

**EN** **Create new** only where the shapes change: data_builder, models wrapper, training loop with 3 X tensors.

🇮🇹 **Vantaggi parallelo:** rollback istantaneo (`rm mtf/`), zero regressioni produzione, A/B validation pulita, niente conflitti con paper-trading single-tf in corso.

**EN** **Parallel-structure benefits:** instant rollback (`rm mtf/`), zero production regressions, clean A/B validation, no conflicts with ongoing single-tf paper-trading.

### Impatti attesi · Expected impacts

🇮🇹
| Aspetto | Single-tf (oggi) | Multi-tf | Delta |
|---|---|---|---|
| Storage `lstm_dataset.npz` | (107480, 120, 119) ~6.1 GB | + (107480, 24, 119)×2 | **+40%** (~8.5 GB) |
| Training iTrans 200 epoche | ~6h | ~10-14h | +60-100% |
| Distill pipeline completa | ~2-3h | **~30-50h GPU** | **10-20×** |
| VRAM TCN+Mamba batch 64 | ~4 GB | ~5-6 GB | ⚠ stretto su 8GB |

**EN**
| Aspect | Single-tf (today) | Multi-tf | Delta |
|---|---|---|---|
| Storage `lstm_dataset.npz` | (107480, 120, 104) ~6.1 GB | + (107480, 24, 104)×2 | **+40%** (~8.5 GB) |
| iTrans training 200 epochs | ~6h | ~10-14h | +60-100% |
| Full distill pipeline | ~2-3h | **~30-50h GPU** | **10-20×** |
| TCN+Mamba VRAM batch 64 | ~4 GB | ~5-6 GB | ⚠ tight on 8GB |

🇮🇹 ⚠ **Iterazione lenta**: ogni esperimento richiede 6-12h. Va pianificato.

**EN** ⚠ **Slow iteration**: each experiment takes 6-12h. Plan accordingly.

🇮🇹
| Metrica | Single-tf | Multi-tf atteso | Confidenza |
|---|---|---|---|
| Directional Accuracy | 51.7-53.2% | **53-56%** | alta |
| Spearman ρ | 0.034-0.062 | **0.07-0.12** | alta |
| Sharpe backtest | +18.71 | +20-40% | bassa (calibrazione) |
| Win rate | ~64% | 65-70% | media |
| Max drawdown | 0.83% | atteso simile o migliore | media |

**EN**
| Metric | Single-tf | Multi-tf expected | Confidence |
|---|---|---|---|
| Directional Accuracy | 51.7-53.2% | **53-56%** | high |
| Spearman ρ | 0.034-0.062 | **0.07-0.12** | high |
| Backtest Sharpe | +18.71 | +20-40% | low (calibration) |
| Win rate | ~64% | 65-70% | medium |
| Max drawdown | 0.83% | expected similar or better | medium |

🇮🇹 **Cosa specifico cattura:**
1. Trend giornaliero (1m con T=120 vede solo 2h)
2. Funding rate cycle 8h (il 1h × 24 cattura 3 cicli completi)
3. Volatility regime shifts che durano ore (distingue "compressione che precede breakout" da "lateralità che continua")
4. Daily seasonality storica (apertura US/EU/Asia)

**EN** **What it specifically captures:**
1. Daily trend (1m with T=120 only sees the last 2h)
2. Funding rate cycle 8h (1h × 24 captures 3 full cycles)
3. Volatility regime shifts lasting hours (distinguishes "compression before breakout" from "ongoing range")
4. Daily seasonality history (US/EU/Asia opens)

🇮🇹 **Rischi:**
1. **Data leakage nei resample**: se il 5m bar al minuto T:00 include T+1..T+4 → predici il futuro. **Test critico**: shuffle X_train e verifica che il modello NON impari.
2. Mismatch live ↔ backtest sul warmup (live aspetta 24h, backtest skippa 1440 candele).
3. Curse of dimensionality (3 encoder ≈ 3M params vs 107k samples = 28× sfavorevole).
4. Costo iterazione 6-12h.

**EN** **Risks:**
1. **Resample data leakage**: if the 5m bar at minute T:00 erroneously includes T+1..T+4 → you're predicting the future. **Critical test**: shuffle X_train and verify the model does NOT learn.
2. Live ↔ backtest warmup mismatch (live waits 24h, backtest skips 1440 candles).
3. Curse of dimensionality (3 encoders ≈ 3M params vs 107k samples = 28× unfavourable).
4. 6-12h iteration cost.

🇮🇹 **Vale la pena se:** paper-trading conferma Sharpe live > 0.5 per 2+ settimane AND vuoi spingere ICIR 0.79 → 0.9+. **No** se cerchi quick win o sistema non ancora validato live.

**EN** **Worth doing if:** paper-trading confirms live Sharpe > 0.5 for 2+ weeks AND you want to push ICIR 0.79 → 0.9+. **No** if you're after a quick win or the system isn't live-validated yet.

🇮🇹 **Costo totale stimato:** 6-9 settimane elapsed (1-2 settimane coding + 1 settimana debug + 30-50h GPU primo training + 2-3 settimane tuning + 1-2 settimane validazione).

**EN** **Total estimated cost:** 6-9 weeks elapsed (1-2 weeks coding + 1 week debugging + 30-50h GPU first training + 2-3 weeks tuning + 1-2 weeks validation).

### Fix #6 — mamba-ssm package (CUDA Toolkit + kernel ufficiale) · Fix #6 — mamba-ssm package (CUDA Toolkit + official kernel)

🇮🇹 **Speedup atteso:** +3-5× sul Mamba branch (sopra il +1.4-1.6× già ottenuto con AMP off + chunk pre-alloc).

**EN** **Expected speedup:** +3-5× on the Mamba branch (on top of the +1.4-1.6× already obtained via AMP off + chunk pre-alloc).

🇮🇹 **Razionale:** l'implementazione attuale in `quantsys/model/tcn_mamba.py` è **pure-PyTorch** (`SimplifiedMambaBlock._parallel_scan_chunk`). Il pacchetto `mamba-ssm` di Tri Dao implementa un kernel CUDA fuso (selective scan) che: carica `(A, B, C, Δ, x)` in shared memory una sola volta, esegue lo scan in registro/SRAM senza scrivere intermedi in HBM, usa parallel prefix scan (Blelloch) su tile, ricomputa lo stato in backward (memory-efficient à la Flash Attention). O(L) compute con costante piccola, O(1) memoria HBM per token.

**EN** **Rationale:** the current implementation in `quantsys/model/tcn_mamba.py` is **pure-PyTorch** (`SimplifiedMambaBlock._parallel_scan_chunk`). The `mamba-ssm` package by Tri Dao implements a fused CUDA kernel (selective scan) that: loads `(A, B, C, Δ, x)` into shared memory once, runs the scan in register/SRAM without writing intermediates to HBM, uses parallel prefix scan (Blelloch) on block tiles, recomputes state in backward (memory-efficient à la Flash Attention). O(L) compute with small constant, O(1) HBM memory per token.

🇮🇹 **Prerequisiti su questa macchina:**
- ✅ RTX 2070 SUPER (Turing 7.5), Python 3.12, PyTorch 2.5.1+cu121, CUDA runtime 12.1
- ❌ **CUDA Toolkit (dev) 12.1.x** mancante — deve matchare `torch.version.cuda`. Mismatch (es. CUDA 12.4 con torch+cu121) → linker errors.
- ⚠ **MSVC Build Tools 2022** probabilmente mancanti
- ❌ `CUDA_HOME` env var da settare

**EN** **Prerequisites on this machine:**
- ✅ RTX 2070 SUPER (Turing 7.5), Python 3.12, PyTorch 2.5.1+cu121, CUDA runtime 12.1
- ❌ **CUDA Toolkit (dev) 12.1.x** missing — must match `torch.version.cuda`. Mismatch (e.g. CUDA 12.4 with torch+cu121) → linker errors.
- ⚠ **MSVC Build Tools 2022** probably missing
- ❌ `CUDA_HOME` env var to set

🇮🇹 **Procedura:**
1. **MSVC Build Tools 2022** da https://visualstudio.microsoft.com/downloads/?q=build+tools — workload "Desktop development with C++" (MSVC v143, Windows 11 SDK, CMake) ~6 GB.
2. **CUDA Toolkit 12.1** da https://developer.nvidia.com/cuda-12-1-1-download-archive (Windows x86_64, exe local, ~3 GB). Custom install: ☑ Development + Runtime; ☐ Driver components (hai già quello dell'RTX).
3. **Env vars** (PowerShell):
   ```powershell
   [Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1", "User")
   [Environment]::SetEnvironmentVariable("CUDA_PATH", "$env:CUDA_HOME", "User")
   # aggiungere $CUDA_HOME\bin al Path
   ```
4. **Install** (no build isolation, forza compilazione contro PyTorch installato):
   ```powershell
   pip install causal-conv1d>=1.2.0 --no-build-isolation
   pip install mamba-ssm --no-build-isolation
   ```
5. **Modifica** `quantsys/model/tcn_mamba.py`: sostituisci `MambaBranch` con import condizionale del kernel ufficiale, mantieni `SimplifiedMambaBlock` come fallback.
6. **Retrain** (i checkpoint NON sono compatibili): backup `models/tcnmamba` + `Remove-Item models\tcnmamba\best_model*.pt` + retrain TCN+Mamba da zero. Verifica `val_nll` converga su valori simili (±5%). Tempo/epoca atteso ~1.5-2 min vs 4.5 min attuali.

**EN** **Procedure:**
1. **MSVC Build Tools 2022** from https://visualstudio.microsoft.com/downloads/?q=build+tools — workload "Desktop development with C++" (MSVC v143, Windows 11 SDK, CMake) ~6 GB.
2. **CUDA Toolkit 12.1** from https://developer.nvidia.com/cuda-12-1-1-download-archive (Windows x86_64, exe local, ~3 GB). Custom install: ☑ Development + Runtime; ☐ Driver components (you already have the RTX driver).
3. **Env vars** (PowerShell):
   ```powershell
   [Environment]::SetEnvironmentVariable("CUDA_HOME", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1", "User")
   [Environment]::SetEnvironmentVariable("CUDA_PATH", "$env:CUDA_HOME", "User")
   # add $CUDA_HOME\bin to Path
   ```
4. **Install** (no build isolation, forces compile against installed PyTorch):
   ```powershell
   pip install causal-conv1d>=1.2.0 --no-build-isolation
   pip install mamba-ssm --no-build-isolation
   ```
5. **Edit** `quantsys/model/tcn_mamba.py`: replace `MambaBranch` with a conditional import of the official kernel, keep `SimplifiedMambaBlock` as fallback.
6. **Retrain** (checkpoints are NOT compatible): backup `models/tcnmamba` + `Remove-Item models\tcnmamba\best_model*.pt` + retrain TCN+Mamba from scratch. Verify `val_nll` converges to similar values (±5%). Expected time/epoch: ~1.5-2 min vs ~4.5 min currently.

🇮🇹 **Rollback:** `pip uninstall mamba-ssm causal-conv1d` → il codice rileva automaticamente `_HAS_MAMBA_SSM = False` e usa il fallback.

**EN** **Rollback:** `pip uninstall mamba-ssm causal-conv1d` → the code auto-detects `_HAS_MAMBA_SSM = False` and uses the fallback.

🇮🇹 **Quando farlo:** retrain frequenti TCN+Mamba (ablation studies), `mamba_layers > 3`, sequenze più lunghe (T > 240 per multi-tf). **Non** se il training corrente è "abbastanza veloce" o stai per cambiare arch.

**EN** **When to do it:** frequent TCN+Mamba retrains (ablation studies), `mamba_layers > 3`, longer sequences (T > 240 for multi-tf). **Not** if current training is "fast enough" or you're about to swap architecture.

---

## 🟢 Audit residui (low-priority, non bloccanti) · 🟢 Audit residue (low-priority, non-blocking)

🇮🇹 4 issue MEDIE + 1 INFRASTRUCTURE dal grand audit 2026-05-23 (8/8 CRITICHE + 8/8 ALTE + 5/9 MEDIE già chiuse):

**EN** 4 MEDIUM issues + 1 INFRASTRUCTURE from the 2026-05-23 grand audit (8/8 CRITICAL + 8/8 HIGH + 5/9 MEDIUM already closed):

🇮🇹
| # | File:line | Issue | Fix proposto | Effort |
|---|---|---|---|---|
| 21 | `quantsys/trading/__init__.py:395` | NaN check `x != x` criptico, solo su `size` | NaN guard esplicito all'inizio di `open_position` con log warning | 10 min |
| 23 | `quantsys/data/__init__.py:48` | Sanity OHLCV `high > close * 10` può scartare flash crash legittimi | rilassare soglia o usare prezzo candela precedente come riferimento | 15 min |
| 27 | `quantsys/model/ensemble.py:104-114` | `arch_names` non impostato nei fallback `load` | non critico, default OK | 5 min |
| 28 | `quantsys/features/__init__.py:251` | `vol_x_pos` crash se colonne assenti su dataset corto | `.get(col, 0)` o try/except con log | 10 min |
| #5 ⚠ | `quantsys/trading/__init__.py:122` + `scripts/03_backtest.py:571-576` | `SignalGenerator.set_regime_threshold` exists ma chiamate DISABILITATE | calibrare empiricamente soglie regime sui dati post-fix denorm (1-2h + retest), oppure rimuovere dead code | 1-2h |

**EN**
| # | File:line | Issue | Proposed fix | Effort |
|---|---|---|---|---|
| 21 | `quantsys/trading/__init__.py:395` | Cryptic NaN check `x != x`, only on `size` | Explicit NaN guard at the top of `open_position` with log warning | 10 min |
| 23 | `quantsys/data/__init__.py:48` | OHLCV sanity `high > close * 10` can discard legitimate flash crashes | Relax threshold or use previous candle price as reference | 15 min |
| 27 | `quantsys/model/ensemble.py:104-114` | `arch_names` not set in `load` fallbacks | Non-critical, default OK | 5 min |
| 28 | `quantsys/features/__init__.py:251` | `vol_x_pos` crashes if columns absent on short dataset | `.get(col, 0)` or try/except with log | 10 min |
| #5 ⚠ | `quantsys/trading/__init__.py:122` + `scripts/03_backtest.py:571-576` | `SignalGenerator.set_regime_threshold` exists but call sites DISABLED | Empirically calibrate regime thresholds on post-denorm-fix data (1-2h + retest), or remove dead code | 1-2h |

🇮🇹 **Contesto fix #5:** bisect 2026-05-24 ha mostrato che le soglie regime hardcoded (overheating +3pp, stagflation +5pp sul default 0.52) riducevano Sharpe da +18.71 a −4.44 (filtravano 27/42 trade vincenti). Infrastructure resta ma dead code.

**EN** **#5 context:** 2026-05-24 bisect showed that hardcoded regime thresholds (overheating +3pp, stagflation +5pp over the 0.52 default) cut Sharpe from +18.71 to −4.44 (filtered 27/42 winning trades). Infrastructure stays but dead code.

🇮🇹 Effort totale chiusura completa: ~1h (4 medie) + 1-2h (#5 se si decide di calibrare).

**EN** Total close-out effort: ~1h (4 mediums) + 1-2h (#5 if you decide to calibrate).

---

## 📋 Soglie di promozione paper-trading · 📋 Paper-trading promotion thresholds

🇮🇹 Indipendenti dai fix, da soddisfare CONTEMPORANEAMENTE prima di andare live (3/4 raggiunte 2026-05-23):

**EN** Independent of the fixes, must be satisfied SIMULTANEOUSLY before going live (3/4 met on 2026-05-23):

🇮🇹
- ✅ Sharpe CI bootstrap (5000 iter): lower bound > 0 (+0.78)
- ✅ Stress test (`pessimistic_fee`, `flash_crash_vol`): almeno break-even (+7.22 / +12.30)
- ⚠ **WHR walkforward (3+ fold): > 0.53 stabile** — iTransformer 0.567, ma N-HiTS/TCN+Mamba 0.50-0.53 (modelli per-fold sotto-trained con max_epochs=40, ricalibrazione post paper-trading via fix #4)
- ✅ Fee/gross ratio: < 30% (30.3% al limite)

**EN**
- ✅ Sharpe bootstrap CI (5000 iter): lower bound > 0 (+0.78)
- ✅ Stress test (`pessimistic_fee`, `flash_crash_vol`): at least break-even (+7.22 / +12.30)
- ⚠ **Walkforward WHR (3+ folds): > 0.53 stable** — iTransformer 0.567, but N-HiTS/TCN+Mamba 0.50-0.53 (per-fold models under-trained with max_epochs=40, recalibration post paper-trading via fix #4)
- ✅ Fee/gross ratio: < 30% (30.3% borderline)

🇮🇹 Le soglie restano valide anche dopo il retrain post-BLOCKER #1; vanno rivalutate sui nuovi modelli a 104 feature.

**EN** These thresholds remain valid post BLOCKER #1 retrain; they must be reassessed on the new 104-feature models.

---

## 🧭 Regola d'oro · 🧭 Golden rule

🇮🇹 **Un fix alla volta, ogni cambio validato da backtest con CI bootstrap.** Cambiare più cose insieme rende impossibile attribuire causalmente il delta.

**EN** **One fix at a time, every change validated with bootstrap CI backtest.** Changing multiple things at once makes it impossible to causally attribute the delta.

🇮🇹 Pattern raccomandato:
1. Applica un fix singolo
2. Retrain completo (un solo modello base se possibile, es. iTrans, per smoke test)
3. Confronta `val_nll`, `DA`, `Spearman`, `Sharpe CI` con baseline pre-fix
4. Se ≥2% miglioramento → mantieni e passa al prossimo
5. Se peggiora o invariato → rollback e analizza prima di provare il prossimo

**EN** Recommended pattern:
1. Apply a single fix
2. Full retrain (one base model if possible, e.g. iTrans, for a smoke test)
3. Compare `val_nll`, `DA`, `Spearman`, `Sharpe CI` with pre-fix baseline
4. If ≥2% improvement → keep and move to the next
5. If worse or unchanged → rollback and analyze before trying the next

🇮🇹 **Lezione 2026-05-24:** attivare fix "completi" senza validation pre-merge può accendere dead state non calibrati (caso #5). Bisect rapido (un fix alla volta) trova il colpevole in 2 iterazioni anche con codebase complesso.

**EN** **2026-05-24 lesson:** enabling "complete" fixes without pre-merge validation can activate uncalibrated dead state (case #5). Fast bisect (one fix at a time) finds the culprit in 2 iterations even on a complex codebase.

---

## ⚫🟢 2026-06-10 — Pivot 1h KILLED · Vol-S PASS (B2 chiusa positiva) · ⚫🟢 2026-06-10 — 1h pivot KILLED · Vol-S PASS (B2 closed positive)

🇮🇹 **Pivot 1h (direzionale): KILL definitivo** dopo probe + 1 iterazione tuning pre-registrata. Il 1h sfonda il muro dei costi (|μ| raw mediano ≈ 43 bps ≫ 26 bps roundtrip, gate fee non più vincolante) ma il gate trading fallisce 4/4 a entrambi i costi (13/23 bps): probe Sharpe −0.87/PF 0.78/74 trade/−5.23%; tuned 5-seed 2 trade/PF 0.12 (l'ensemble medio-azzera μ e gonfia σ via legge varianza totale). L'**anti-correlazione val→test si conferma anche a 1h** (val ρ +0.19 → test ρ −0.04): è del target direzionale, non del timeframe. Filone "stesso metodo, altro timeframe" chiuso.

**EN** **1h pivot (directional): definitive KILL** after probe + 1 pre-registered tuning iteration. 1h does break the cost wall (median raw |μ| ≈ 43 bps ≫ 26 bps roundtrip, fee gate no longer binding) but the trading gate fails 4/4 at both costs (13/23 bps): probe Sharpe −0.87/PF 0.78/74 trades/−5.23%; tuned 5-seed 2 trades/PF 0.12 (the ensemble averages μ toward zero and inflates σ via the total-variance law). The **val→test anti-correlation holds at 1h too** (val ρ +0.19 → test ρ −0.04): it belongs to the directional target, not the timeframe. The "same method, different timeframe" axis is closed.

🇮🇹 **Vol-S (B2): PASS — primo gate pre-registrato superato nel progetto.** Target `features.target_type: log_rv` (log-RV a h=30 barre 1h), stessa pipeline/hyperparam: QLIKE test **NN 0.2572 vs HAR-RV 0.3681 vs naive 0.8067** → NN/HAR = 0.699 (gate ≤ 0.95, margine 6×); val 0.744 → test 0.699 **coerenti**. Modello: Spearman test +0.45, DA 71%, ICIR +3.56. La vol è prevedibile sopra la baseline econometrica seria ma NON tradabile sul perimetro spot/perp: valore = jump/no-trade gate difensivo (follow-up) + opzione Deribit/varianza. Dettagli e trappola denorm (centro+scala) in `TEORIA.md` §2 e `STATUS.md`.

**EN** **Vol-S (B2): PASS — first pre-registered gate ever passed in this project.** Target `features.target_type: log_rv` (log-RV at h=30 1h-bars), same pipeline/hyperparams: test QLIKE **NN 0.2572 vs HAR-RV 0.3681 vs naive 0.8067** → NN/HAR = 0.699 (gate ≤ 0.95, 6× margin); val 0.744 → test 0.699 **consistent**. Model: test Spearman +0.45, DA 71%, ICIR +3.56. Vol is predictable above the serious econometric baseline but NOT tradable on the spot/perp perimeter: value = defensive jump/no-trade gate (follow-up) + Deribit/variance option. Details and the denorm trap (center+scale) in `TEORIA.md` §2 and `STATUS.md`.

---

## ⚫ 2026-06-11 — Probe semivarianza (log_rs_ratio) FAIL · ⚫ 2026-06-11 — Semivariance probe (log_rs_ratio) FAIL

🇮🇹 **Probe pre-registrato: il "segno della varianza"** — target `log_rs_ratio` = `log(RS⁺_fwd/RS⁻_fwd)` a h=30 barre 1h (semivarianza realizzata firmata; traduzione econometrica dell'idea historical-decomposition). Stessa pipeline/hyperparam del vol-S; giudice `scripts/vol/dev_vols_rs_judge.py` (HAR-RS Patton–Sheppard OLS train-only + naive + train-mean, metrica MSE — il QLIKE non si applica a un target non positivo-definito). **Esito test: FAIL** (NN/HAR-RS MSE 0.9952 > 0.95; NN batte la costante di 0.02%; signDA 0.459 < 0.55; ρ val +0.078 → test −0.038). Il punto scientifico: **nessuno** predice l'asimmetria (HAR-RS fa peggio della costante su test) → l'informazione in price/volume riguarda i **momenti pari** (livello RV: −30% QLIKE) e non quelli **dispari** (direzione, signed jump variation). Filone chiuso senza appello come pre-registrato.

**EN** **Pre-registered probe: the "sign of variance"** — target `log_rs_ratio` = `log(RS⁺_fwd/RS⁻_fwd)` at h=30 1h-bars (signed realized semivariance; the econometric translation of the historical-decomposition idea). Same pipeline/hyperparams as vol-S; judge `scripts/vol/dev_vols_rs_judge.py` (HAR-RS Patton–Sheppard OLS train-only + naive + train-mean, MSE metric — QLIKE does not apply to a non-positive-definite target). **Test outcome: FAIL** (NN/HAR-RS MSE 0.9952 > 0.95; NN beats the constant by 0.02%; signDA 0.459 < 0.55; ρ val +0.078 → test −0.038). The scientific point: **nobody** predicts the asymmetry (HAR-RS is worse than the constant on test) → the information in price/volume concerns **even moments** (RV level: −30% QLIKE) and not **odd ones** (direction, signed jump variation). Thread closed without appeal as pre-registered.

---

## 💡 Insights consolidati (validi long-term) · 💡 Consolidated insights (long-term valid)

🇮🇹
1. **Modello predittivo sano in tutti i setup**: walkforward DA 0.53-0.54, Spearman 0.08-0.09, σ ben calibrato. Il problema, quando emerge, è quasi sempre nel **trading layer** (scala, soglie, SL/TP), non nel modello — vedi sessione 2026-05-23 per il caso paradigmatico (Sharpe −256 → +18.7 con 1 moltiplicazione mancante).
2. **h=15 è strutturalmente perdente**: cost roundtrip 26 bps ≈ |realized return medio| 25 bps. h=30 raddoppia il segnale mantenendo costo costante. Già applicato.
3. **`max_sigma` va sempre dimensionato sulla distribuzione σ del modello specifico** (es. p99 della σ_test). Valori arbitrari sono inutili.
4. **`trailing_atr_mult: 1.5`** ≈ 11 bps di trail su BTC 1m → chiude su rumore (< del cost 26 bps). Su 1m bar `use_trailing_stop: false` (attuale) batte qualsiasi trailing tunato.
5. **Verificare le scale unit-by-unit prima di retrainare**: per 6+ sessioni a maggio 2026 abbiamo cercato fix sui pesi del modello (RevIN, h, stride, multi-teacher) — il vero bug era 1 moltiplicazione mancante in 2 file (denormalizzazione z-score → raw).

**EN**
1. **Predictive model healthy in all setups**: walkforward DA 0.53-0.54, Spearman 0.08-0.09, well-calibrated σ. When problems emerge, they're almost always in the **trading layer** (scale, thresholds, SL/TP), not in the model — see the 2026-05-23 session for the paradigmatic case (Sharpe −256 → +18.7 from one missing multiplication).
2. **h=15 is structurally a losing setup**: roundtrip cost 26 bps ≈ |mean realized return| 25 bps. h=30 doubles the signal while keeping cost constant. Already applied.
3. **`max_sigma` must always be sized on the specific model's σ distribution** (e.g. p99 of σ_test). Arbitrary values are useless.
4. **`trailing_atr_mult: 1.5`** ≈ 11 bps of trail on BTC 1m → closes on noise (< 26 bps cost). On 1m bars `use_trailing_stop: false` (current) beats any tuned trailing.
5. **Verify scales unit-by-unit before retraining**: for 6+ sessions in May 2026 we hunted fixes on model weights (RevIN, h, stride, multi-teacher) — the actual bug was 1 missing multiplication across 2 files (z-score → raw denormalization).
