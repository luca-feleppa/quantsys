# QUANTSYS — Changelog

🇮🇹 Ordine cronologico inverso (la voce più recente in alto). Le "Iterazioni" 1-10 sono il blocco direzionale storico (1m); dal pivot 1h in poi le voci sono datate per sessione e tracciate in dettaglio in `STATUS.md` (lab notebook append-only). Questo file riassume i milestone; `STATUS.md` è la fonte canonica.

**EN** Reverse chronological order (newest on top). "Iterations" 1-10 are the historical directional block (1m); from the 1h pivot onward entries are dated per session and tracked in detail in `STATUS.md` (append-only lab notebook). This file summarizes milestones; `STATUS.md` is the canonical source.

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
