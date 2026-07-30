# Mappa dei risultati per il paper · Results map for the paper

🇮🇹 Inventario claim→evidenza per *"Are price and volume enough?"*. Ogni claim elenca: artefatto su disco (quando esiste), riferimento canonico in `STATUS.md`/git (quando l'artefatto è stato sovrascritto dal flusso operativo), e i numeri chiave. Il lab notebook canonico dell'intero corpus è `STATUS.md` (append-only, con pre-registrazioni datate *prima* di ogni run — citarlo come registro di pre-registrazione).

**EN** Claim→evidence inventory for *"Are price and volume enough?"*. Each claim lists: on-disk artifact (when it exists), canonical reference in `STATUS.md`/git (when the artifact was overwritten by operations), and the key numbers. The canonical lab notebook for the whole corpus is `STATUS.md` (append-only, with pre-registrations dated *before* each run — cite it as the pre-registration registry).

---

## CLAIM 1 — La direzione (momento dispari, 1°) è impredicibile OOS · Direction (odd moment, 1st) is unpredictable OOS

### 1a. NN direzionale a 1h (5-seed iTransformer, tuned) · Directional NN at 1h
- **Artefatto/Artifact:** sovrascritto dai run vol; numeri in `STATUS.md` §"VERDETTO FINALE PIVOT 1h" (2026-06-10) + commit `6680027`.
- **Numeri/Numbers:** val Spearman **+0.19/+0.20** → test **−0.041 (p=0.001)**, IC −0.023, ICIR −0.12. Backtest gate (13≡23 bps): 2 trade, net −1.46%, PF 0.12, Sharpe −8.24 → 4/4 criteri pre-registrati falliti. Probe 1-seed: test Spearman +0.012 (p=0.33), DA 50.0%, 74 trade, net −5.23%, Sharpe −0.87, gross −1.9% (perdita di segnale, non di fee).
- **Punto chiave/Key point:** |μ| mediano ≈43 bps ≫ 26 bps di costo → il muro dei costi a 1h è sfondato MA non c'è skill: il fallimento è informativo, non microstrutturale.

### 1b. NN direzionale a 1m (era production) · Directional NN at 1m
- **Artefatto/Artifact:** numeri in `STATUS.md` §2026-06-04/05/06 + memoria di progetto; modelli eliminati (cleanup 2026-06-12), raw 1m in `data/backup_1m/`.
- **Numeri/Numbers:** test backtest baseline −1.77% (iTrans, migliore); edge di rango Quiet Spearman +0.13÷0.19 stabile ma NON monetizzabile: rank-entry val PF 0.84/13 trade; Fix cadenza+rank-exposure val PF 0.22; horizon-exit puro PF 0.49, WR 29% → 3 conferme indipendenti che l'edge ordinale non sopravvive alla realizzazione.
- **Fenomeno metodologico/Methodological finding:** **anti-correlazione val→test** sistematica delle metriche direzionali (distribution shift strutturale) — replicata a 1m E 1h.

### 1c. Cross-sectional (16 perp USDT) · Cross-sectional
- **Artefatto/Artifact:** `results/xs/ic_report.json` ✓ (su disco/on disk).
- **Numeri/Numbers:** mean cross-sectional IC +0.0138, t=1.86 (<2), ICIR 0.035, 4/5 sotto-periodi positivi; spread top-bottom +1.5 bps/step vs ~26 bps costo → netto −43%/ann. **Il muro è la MAGNITUDINE, non il segno.**

### 1d. Baseline econometriche direzionali (1h) · Directional econometric baselines (1h) — **NUOVO 2026-06-12**
- **Artefatto/Artifact:** `results/paper/dir_baselines_1h_{val,test}.json` ✓; script `scripts/research/paper_01_dir_baselines.py`; pre-registrazione in `STATUS.md` 2026-06-12 sera.
- **Numeri/Numbers** (soglia 2/√n = 0.025):

| Baseline | ρ val | ρ test | signDA val | signDA test |
|---|---|---|---|---|
| OLS "HAR-mean" | −0.013 | +0.034 | 0.524 | 0.491 |
| Logit segno/sign | +0.025 | −0.002 | 0.529 | 0.487 |
| Momentum (r_h trailing) | −0.048 | +0.016 | 0.479 | 0.511 |
| Train-mean (null) | 0 | 0 | 0.524 | 0.481 |

- **Punto chiave/Key point:** nessuna skill coerente; i ρ nominalmente sopra-soglia **flippano segno val→test** (stessa instabilità del NN) e le signDA seguono il base rate. → Il risultato è dell'**informazione**, non della classe di modello.

---

## CLAIM 2 — La varianza (momento pari, 2°) è predicibile OOS, e solo a risoluzione 1h · Variance (even moment, 2nd) is predictable OOS, and only at 1h resolution

### 2a. VOL-S 1h — PASS (positive control del paper / the paper's positive control)
- **Artefatto/Artifact:** `results/vols/qlike_report_1h_{val,test}.json` ✓; modelli in `models/backup_1h_vols/` (= `models/itransformer/` correnti); giudice `scripts/vol/dev_vols_qlike.py`.
- **Numeri/Numbers:** QLIKE test NN **0.2572** vs HAR-RV 0.3681 vs naive 0.8067 → **NN/HAR = 0.699 (−30%)** ⚠ **AGGIORNAMENTO 2026-07-30 (gate C2): la baseline di riferimento non e' piu' HAR-RV ma HAR-CJ** (decomposizione continua/salti). Contro HAR-CJ il vantaggio e' **22.6% su val e 31.6% su test** (NN 0.23637 vs 0.34572 su test), p <= 4.3e-04. Il paper deve citare la banda **-23% / -32%**; i numeri contro HAR-RV qui sotto restano validi come confronto storico ma NON sono il claim corrente. Dettaglio: `TEORIA.md` §12.2 e `STATUS.md` sezione C2.; val 0.744 → test 0.699 **coerenti** (niente anti-correlazione). Test: Spearman +0.4532 (p≈0), DA 71.3%, ICIR +3.56 su 5 sotto-periodi, coverage 95.2%.

### 2b. Robustezza purged k-fold + gate HAR-per-fold (1h) · Purged k-fold robustness + HAR-per-fold gate — **NUOVO 2026-06-22**
- **Artefatto/Artifact:** `results/vols/wf_har_baseline_1h.json` ✓ (script `scripts/vol/wf_har_baseline.py`, helper `build_har_frame`/`har_fold_qlike` in `quantsys/model/vol_metrics.py`); QLIKE NN per-fold in `results/{arch}/walkforward_metrics_log_rv.json` (harness `scripts/02b_walkforward_validate.py`, fold-metric QLIKE). Pre-registrazione `STATUS.md` 2026-06-22.
- **Numeri/Numbers (5 fold effettivi, embargo 168h=1 sett, 1-seed; HAR fit-per-fold; gate per-fold QLIKE_NN ≤ 0.95·QLIKE_HAR):** HAR QLIKE medio cross-fold **0.430** (naive 0.800).

| arch | NN medio | ratio NN/HAR | batte HAR | verdetto |
|---|---|---|---|---|
| **TCN+Mamba** | **0.364** | **0.863** | **4/5** | ✅ PASS |
| N-HiTS | 0.401 | 0.948 | 3/5 | ~ borderline |
| iTransformer (1-seed) | 0.493 | 1.170 | 3/5 | ❌ FAIL |

- **Punto chiave/Key point:** la skill vol **sopravvive al purged k-fold**: TCN+Mamba batte HAR di ~14% OOS, decisivo nei fold data-rich (ratio 0.75–0.83). TUTTI gli archi falliscono il fold 1 (più antico, expanding-window data-starved → NN data-hungry < HAR a 3 param): è effetto strutturale, non model-failure (la skill emerge coi dati, fold 3→5 sempre PASS). ⚠ L'iTrans 1-seed "FAIL" qui è **artefatto della MEDIA cross-fold** (over-penalizza i data-hungry sui fold early), NON apples-to-apples col PASS 5-seed single-split di 2a: sul giudice val data-rich (= regime di produzione) iTrans 5-seed resta 0.343/ratio 0.92 ✓ (verifica `STATUS.md` 2026-06-22). → doppia conferma OOS (single-split 5-seed + k-fold data-rich).

### 2c. Diversità cross-arch sugli errori vol · Cross-arch error diversity on vol — **NUOVO 2026-06-22**
- **Artefatto/Artifact:** `results/vols/step0_xarch_corr_val.json` ✓ (script `scripts/vol/step0_xarch_corr.py`, kill-check pre-distill).
- **Numeri/Numbers (Pearson errori per-campione, val, n=6420):** iTrans|N-HiTS 0.776 · iTrans|TCN-Mamba 0.815 · N-HiTS|TCN-Mamba 0.887 → **mean 0.826, min 0.776**.
- **Punto chiave/Key point:** sul vol gli archi **disaccordano** (ρ_err ~0.83), in netto contrasto col direzionale (≈0.995, dove l'ensembling riduce la varianza ≈0 → matematicamente inutile, vedi CLAIM 1). La diversità cross-arch è quindi anch'essa un **oggetto pari-specifico**: esiste solo dove l'informazione esiste. ⚠ Caveat: 1 seed/1 split, yaml era-1m → parte della diversità può essere overfit-indotta; la robustezza vera è 2b.

### 2d. Verifica cross-risoluzione 1m — FAIL (perimetro della claim / the claim's perimeter)
- **Artefatto/Artifact:** `results/vols/qlike_report_1m_val.json` ✓.
- **Numeri/Numbers:** val 1m NN/HAR QLIKE = **1.0127** (>0.95, sanity val-first fallita; test mai toccato, come pre-registrato). → L'edge vol sopra HAR esiste **solo a RV oraria** (h=30h): a RV-30min HAR è già sufficiente.

---

## CLAIM 3 — L'asimmetria firmata della varianza (oggetto dispari) è impredicibile per TUTTI · The signed variance asymmetry (odd object) is unpredictable for ALL

- **Artefatto/Artifact:** `results/vols/rs_report_1h_{val,test}.json` ✓; giudice `scripts/vol/dev_vols_rs_judge.py`; target `log(RS⁺/RS⁻)` (Barndorff-Nielsen–Kinnebrock–Shephard 2010; Patton–Sheppard 2015).
- **Numeri/Numbers (MSE log-ratio, test):** NN 0.99366, HAR-RS 0.99843, train-mean **0.99387**, naive 1.89025 → **HAR-RS fa peggio della costante**; NN/HAR 0.9952 (gate ≤0.95 fallito); signDA NN 0.459 < 0.5. ρ val +0.078 → test −0.038 (stesso flip dei momenti dispari).
- **Punto chiave/Key point:** chiude il quadrato: pari predicibile / dispari no, *anche dentro la famiglia vol*.

---

## CLAIM 4 (in accumulo / accruing) — Test economico forward: NN-RV vs IV implicita · Forward economic test: NN-RV vs implied IV

- **Artefatto/Artifact:** `results/vol_paper/{forecasts.parquet, trades.jsonl, baseline_report.json}` (harness `scripts/04b_vol_paper.py`, testnet Deribit, dal 2026-06-12); IV: `data/iv/atm_30h.parquet` (poller `01c_iv_poller.py`); DVOL storico `data/iv/dvol.parquet`.
- **Stato/Status:** pre-registrato (regola |log(RV_pred/var_iv)|>0.25, straddle ATM ~30h, hold-to-expiry direction-neutral, gate a ≥30 trade — non ancora raggiunto, `evaluable=false`). **Preliminare (n=7, da `trades.jsonl` / `STATUS.md` 2026-06-23):** PnL ≈ −0.038 BTC, hit-rate ~29%, trade quasi tutti LONG-vol che perdono → **IV > RV realizzata (VRP positivo)**; il lato che paga è plausibilmente lo SHORT-vol (atteso da un VRP strutturalmente positivo su BTC). ⚠ Il testnet valida bene il **segnale** hold-to-expiry (settlement deterministico) ma NON l'esecuzione (liquidità simulata). Per il paper: sezione "economic relevance" come outlook, da chiudere a n≥30.

---

## Note di provenienza · Provenance notes

🇮🇹 (1) I numeri senza artefatto su disco sono ricostruibili dal lab notebook `STATUS.md` alla data indicata e dai commit git (`ae31540` xs-KILL, `6680027` pivot 1h+vol-S, `f7fa33e` rs-FAIL, `482b682` cleanup+restore, `cc81131` distill target-aware). Il codice 2c/2b (k-fold QLIKE, cross-arch corr, HAR-per-fold, `vol_metrics.py` helpers) è eseguito ma NON ancora committato (artefatti JSON su disco; STATUS 2026-06-22 è il registro). (2) Split: i report vol/rs usano il dataset del run (65.159 candele il 06-10; 65.191 il 06-11 → 51156/6394/6395 finestre); le baseline direzionali ricostruiscono lo split sul raw corrente (verificato esatto, vedi commento in `paper_01_dir_baselines.py`) — scostamento di boundary ≤0,05%, irrilevante per claim a ρ≈0. (3) Tutti i gate erano pre-registrati in `STATUS.md` PRIMA delle run, protocollo val-first, test toccato una volta.

**EN** (1) Numbers without an on-disk artifact are recoverable from the `STATUS.md` lab notebook at the stated date and from git commits (`ae31540` xs-KILL, `6680027` 1h pivot+vol-S, `f7fa33e` rs-FAIL, `482b682` cleanup+restore, `cc81131` target-aware distill). The 2c/2b code (k-fold QLIKE, cross-arch corr, HAR-per-fold, `vol_metrics.py` helpers) is run but NOT yet committed (JSON artifacts on disk; STATUS 2026-06-22 is the registry). (2) Splits: vol/rs reports use the run's dataset (65,159 candles on 06-10; 65,191 on 06-11 → 51156/6394/6395 windows); the directional baselines rebuild the split on the current raw (verified exact, see comment in `paper_01_dir_baselines.py`) — boundary deviation ≤0.05%, immaterial for ρ≈0 claims. (3) All gates were pre-registered in `STATUS.md` BEFORE the runs, val-first protocol, test touched once.
