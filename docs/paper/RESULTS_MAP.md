# Mappa dei risultati per il paper · Results map for the paper

🇮🇹 Inventario claim→evidenza per *"Are price and volume enough?"*. Ogni claim elenca: artefatto su disco (quando esiste), riferimento canonico in `STATUS.md`/git (quando l'artefatto è stato sovrascritto dal flusso operativo), e i numeri chiave. Il lab notebook canonico dell'intero corpus è `STATUS.md` (append-only, con pre-registrazioni datate *prima* di ogni run — citarlo come registro di pre-registrazione).

**EN** Claim→evidence inventory for *"Are price and volume enough?"*. Each claim lists: on-disk artifact (when it exists), canonical reference in `STATUS.md`/git (when the artifact was overwritten by operations), and the key numbers. The canonical lab notebook for the whole corpus is `STATUS.md` (append-only, with pre-registrations dated *before* each run — cite it as the pre-registration registry).

---

## CLAIM 1 — La direzione (momento dispari, 1°) è impredicibile OOS · Direction (odd moment, 1st) is unpredictable OOS

### 1a. NN direzionale a 1h (5-seed iTransformer, tuned) · Directional NN at 1h
- **Artefatto/Artifact:** sovrascritto dai run vol; numeri in `STATUS.md` §"VERDETTO FINALE PIVOT 1h" (2026-06-10) + commit `21e86a8`.
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
- **Artefatto/Artifact:** `results/paper/dir_baselines_1h_{val,test}.json` ✓; script `scripts/paper_01_dir_baselines.py`; pre-registrazione in `STATUS.md` 2026-06-12 sera.
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
- **Artefatto/Artifact:** `results/vols/qlike_report_1h_{val,test}.json` ✓; modelli in `models/backup_1h_vols/` (= `models/itransformer/` correnti); giudice `scripts/dev_vols_qlike.py`.
- **Numeri/Numbers:** QLIKE test NN **0.2572** vs HAR-RV 0.3681 vs naive 0.8067 → **NN/HAR = 0.699 (−30%)**; val 0.744 → test 0.699 **coerenti** (niente anti-correlazione). Test: Spearman +0.4532 (p≈0), DA 71.3%, ICIR +3.56 su 5 sotto-periodi, coverage 95.2%.

### 2b. Verifica cross-risoluzione 1m — FAIL (perimetro della claim / the claim's perimeter)
- **Artefatto/Artifact:** `results/vols/qlike_report_1m_val.json` ✓.
- **Numeri/Numbers:** val 1m NN/HAR QLIKE = **1.0127** (>0.95, sanity val-first fallita; test mai toccato, come pre-registrato). → L'edge vol sopra HAR esiste **solo a RV oraria** (h=30h): a RV-30min HAR è già sufficiente.

---

## CLAIM 3 — L'asimmetria firmata della varianza (oggetto dispari) è impredicibile per TUTTI · The signed variance asymmetry (odd object) is unpredictable for ALL

- **Artefatto/Artifact:** `results/vols/rs_report_1h_{val,test}.json` ✓; giudice `scripts/dev_vols_rs_judge.py`; target `log(RS⁺/RS⁻)` (Barndorff-Nielsen–Kinnebrock–Shephard 2010; Patton–Sheppard 2015).
- **Numeri/Numbers (MSE log-ratio, test):** NN 0.99366, HAR-RS 0.99843, train-mean **0.99387**, naive 1.89025 → **HAR-RS fa peggio della costante**; NN/HAR 0.9952 (gate ≤0.95 fallito); signDA NN 0.459 < 0.5. ρ val +0.078 → test −0.038 (stesso flip dei momenti dispari).
- **Punto chiave/Key point:** chiude il quadrato: pari predicibile / dispari no, *anche dentro la famiglia vol*.

---

## CLAIM 4 (in accumulo / accruing) — Test economico forward: NN-RV vs IV implicita · Forward economic test: NN-RV vs implied IV

- **Artefatto/Artifact:** `results/vol_paper/{forecasts.parquet, trades.jsonl}` (harness `scripts/04b_vol_paper.py`, testnet Deribit, dal 2026-06-12); IV: `data/iv/atm_30h.parquet` (poller `01c_iv_poller.py`); DVOL storico `data/iv/dvol.parquet`.
- **Stato/Status:** pre-registrato (regola |log(RV_pred/var_iv)|>0.25, straddle ATM ~30h, gate a ≥30 trade). Per il paper: sezione "economic relevance" — risultato preliminare o follow-up, secondo i tempi.

---

## Note di provenienza · Provenance notes

🇮🇹 (1) I numeri senza artefatto su disco sono ricostruibili dal lab notebook `STATUS.md` alla data indicata e dai commit git (`6ca5676` xs-KILL, `21e86a8` pivot 1h+vol-S, `0253d2e` rs-FAIL, `590b96a` cleanup+restore). (2) Split: i report vol/rs usano il dataset del run (65.159 candele il 06-10; 65.191 il 06-11 → 51156/6394/6395 finestre); le baseline direzionali ricostruiscono lo split sul raw corrente (verificato esatto, vedi commento in `paper_01_dir_baselines.py`) — scostamento di boundary ≤0,05%, irrilevante per claim a ρ≈0. (3) Tutti i gate erano pre-registrati in `STATUS.md` PRIMA delle run, protocollo val-first, test toccato una volta.

**EN** (1) Numbers without an on-disk artifact are recoverable from the `STATUS.md` lab notebook at the stated date and from git commits (`6ca5676` xs-KILL, `21e86a8` 1h pivot+vol-S, `0253d2e` rs-FAIL, `590b96a` cleanup+restore). (2) Splits: vol/rs reports use the run's dataset (65,159 candles on 06-10; 65,191 on 06-11 → 51156/6394/6395 windows); the directional baselines rebuild the split on the current raw (verified exact, see comment in `paper_01_dir_baselines.py`) — boundary deviation ≤0.05%, immaterial for ρ≈0 claims. (3) All gates were pre-registered in `STATUS.md` BEFORE the runs, val-first protocol, test touched once.
