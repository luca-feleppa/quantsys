# POST_GATE_V1 — Piano operativo alla chiusura del gate v1 · Operating plan at v1-gate closure

> 🇮🇹 Scritto 2026-07-16 a **n=19 settlement** (PRIMA del 20°: nessun numero aggregato calcolato — le decisioni di campione qui sotto sono prese a esiti finali non visti). Questo file è la checklist esecutiva; le pre-registrazioni vincolanti restano in `STATUS.md`. Da ELIMINARE a piano completato.
> **EN** Written 2026-07-16 at **n=19 settlements** (BEFORE the 20th: no aggregate numbers computed — the sample decisions below are made without seeing final outcomes). This file is the executive checklist; the binding pre-registrations stay in `STATUS.md`. DELETE when the plan is complete.

---

## 0 · Definizione del campione (decisa ORA, pre-esito) · Sample definition (decided NOW, pre-outcome)

🇮🇹 Due ambiguità scoperte il 2026-07-16 e risolte qui, prima di vedere il verdetto:

1. **Trade #0 (`executed: false`, entry 2026-06-12 13:02).** Fu aperto dallo **smoke pre-lancio**, non dal processo `--execute`; le annotazioni contemporanee lo designano come non-campione ("aperto dallo smoke pre-lancio; i successivi saranno ordini reali", STATUS 06-12, scritta PRIMA del suo settlement; "calibrazione, non segnale reale", STATUS 06-13). **Decisione: campione primario = soli settlement `executed: true` → il gate v1 chiude al 20° settlement eseguito = 21ª riga di `trades.jsonl`** (~1 trade dopo la 20ª riga). ⚠ Trasparenza obbligatoria: il trade #0 è una PERDITA (−0.0100), quindi escluderlo favorisce il PASS — il report di chiusura calcola i criteri su ENTRAMBI i campioni (con e senza #0); se il verdetto diverge tra i due, vale il **peggiore** (fail-safe pre-dichiarato).
2. **Soglia 20 vs 30.** La pre-registrazione originale (2026-06-12) fissava la valutazione a **≥30 trade chiusi**; "n≥20" è entrato nel lessico dal gate short-vol (06-24) e da lì è diventato "gate v1" senza ri-registrazione formale. **Risoluzione: n=20 = checkpoint operativo** (sblocca le attivazioni v2 qui sotto, coerente con 3 settimane di STATUS) **ma la valutazione pre-registrata a ≥30 resta dovuta**: la leg opzioni è INVARIATA sotto `--hedge` (confronto within-trade), quindi il campione v1 continua a crescere e i 3 criteri congelati si ri-valutano a n=30 (~10 giorni dopo). **Vincolo conseguente: A13/A14 (pin-close, sizing vega) NON si attivano prima della valutazione a 30** — cambierebbero la leg opzioni e contaminerebbero il campione.

**EN** Two ambiguities discovered 2026-07-16 and resolved here, before seeing the verdict:

1. **Trade #0 (`executed: false`, entry 2026-06-12 13:02).** Opened by the **pre-launch smoke**, not by the `--execute` process; contemporaneous notes designate it as non-sample ("opened by the pre-launch smoke; the next ones will be real orders", STATUS 06-12, written BEFORE its settlement; "calibration, not a real signal", STATUS 06-13). **Decision: primary sample = `executed: true` settlements only → gate v1 closes at the 20th executed settlement = 21st row of `trades.jsonl`** (~1 trade after the 20th row). ⚠ Mandatory transparency: trade #0 is a LOSS (−0.0100), so excluding it favors PASS — the closure report computes the criteria on BOTH samples (with and without #0); if the verdicts diverge, the **worse** one stands (pre-declared fail-safe).
2. **Threshold 20 vs 30.** The original pre-registration (2026-06-12) set evaluation at **≥30 closed trades**; "n≥20" entered the lexicon from the short-vol gate (06-24) and became "gate v1" without a formal re-registration. **Resolution: n=20 = operational checkpoint** (unlocks the v2 activations below, consistent with 3 weeks of STATUS) **but the pre-registered ≥30 evaluation is still owed**: the options leg is UNCHANGED under `--hedge` (within-trade comparison), so the v1 sample keeps growing and the 3 frozen criteria are re-evaluated at n=30 (~10 days later). **Consequent constraint: A13/A14 (pin-close, vega sizing) do NOT activate before the n=30 evaluation** — they would alter the options leg and contaminate the sample.

---

## 1 · Fase A — Chiusura formale (subito, niente GPU) · Phase A — Formal closure (immediately, no GPU)

🇮🇹
| # | Azione | Test/giudice | Informazione/conferma attesa |
|---|---|---|---|
| A1 | **Report di chiusura gate v1** sui 3 criteri congelati 2026-06-12: ① PnL medio/trade > 0 net; ② PnL totale > always-long-vol E always-short-vol sullo stesso calendario; ③ hit-rate > 0.5 | `scripts/04c_vol_paper_baselines.py` (criterio ②) + aggregati da `trades.jsonl` (①③), su ENTRAMBI i campioni (§0.1) | il segnale NN-RV-vs-IV ha valore economico OLTRE il VRP medio? (② isola il timing dal premio strutturale) |
| A2 | Nel report: **caveat selezione oraria** (⑤bis, dati pre-14/07 = 18.6% coverage), **nota dedup** (18 righe→17), **nota trade #0** (§0.1), **nota 20-vs-30** (§0.2) | — | delimitano la validità ESTERNA del verdetto, non le soglie |
| A3 | **Attivare `vol_paper_replay.py` come gap-filler ufficiale** (griglia dal 2026-07-14T14Z) | parity check integrato vs tick live sovrapposti (già bit-exact in validazione) | serie forecast H24 senza bias orario → alimenta la pre-reg v2, MAI il campione v1 |
| A4 | **`hedge_dry_run.py` sulla serie A6 piena** → congelare **band** (argmax total_net su {0.10…0.30}), **λ WW** (A12, confronto offline fixed-vs-ww), **convenzione δ** (raw/adj, match slope empirico) → **update pre-reg V2** in STATUS (+ ratifica funding PROD) | dry-run offline su dati DISGIUNTI dal giudizio forward | parametri v2 congelati su dati pre-attivazione → zero gradi di libertà a giudizio in corso |

**EN**
| # | Action | Test/judge | Expected information/confirmation |
|---|---|---|---|
| A1 | **v1-gate closure report** on the 3 frozen 2026-06-12 criteria: ① mean PnL/trade > 0 net; ② total PnL > always-long-vol AND always-short-vol on the same calendar; ③ hit-rate > 0.5 | `scripts/04c_vol_paper_baselines.py` (criterion ②) + aggregates from `trades.jsonl` (①③), on BOTH samples (§0.1) | does the NN RV-vs-IV signal add economic value BEYOND the average VRP? (② isolates timing from the structural premium) |
| A2 | In the report: **hour-selection caveat** (⑤bis, pre-14/07 data = 18.6% coverage), **dedup note** (18 rows→17), **trade #0 note** (§0.1), **20-vs-30 note** (§0.2) | — | they bound the verdict's EXTERNAL validity, not the thresholds |
| A3 | **Activate `vol_paper_replay.py` as the official gap-filler** (grid from 2026-07-14T14Z) | built-in parity check vs overlapping live ticks (already bit-exact in validation) | 24/7 forecast series without hour bias → feeds the v2 pre-reg, NEVER the v1 sample |
| A4 | **`hedge_dry_run.py` on the full A6 series** → freeze **band** (argmax total_net over {0.10…0.30}), **WW λ** (A12, offline fixed-vs-ww comparison), **δ convention** (raw/adj, empirical-slope match) → **update the V2 pre-reg** in STATUS (+ ratify PROD funding) | offline dry-run on data DISJOINT from the forward judgment | v2 parameters frozen on pre-activation data → zero degrees of freedom mid-judgment |

---

## 2 · Fase B — Finestra GPU (04b fermo, poche ore) · Phase B — GPU window (04b stopped, a few hours)

🇮🇹 Ordine vincolante (pre-reg già scritte in STATUS; run indipendenti vs lo stesso incumbent; ogni interazione A3×A8×DVOL = nuova pre-reg):
| # | Azione | Test/giudice | Informazione attesa |
|---|---|---|---|
| B1 | **Audit `causality-auditor` sui file A3** (P3, mai eseguito) | audit read-only | nessun lookahead residuo nel gate/training MoE prima di spendere GPU |
| B2 | **Run A3 regime-MoE** (`QUANTSYS_ARCH=itransformer_regime_moe`, sandbox `models_a3_moe`, 5 seed) | `dev_vols_qlike.py` su val: ① QLIKE ≤0.97·incumbent ② nessun regime distrutto (≤1.05 nel peggiore) ③ n≥5000/≥800 | le 3 teste-regime col gate causale battono l'incumbent su μ? (misura μ, NON σ — MINOR-1 dichiarata) |
| B3 | **Run A8 mixup** (`QUANTSYS_ARCH=itransformer_a8_mixup`, sandbox `models_a8_mixup`, 5 seed) | stesse 3 condizioni su val | l'unica augmentation cross-feature-coerente riduce l'overfit residuo? (prior onesto: FAIL/nullo) |
| B4 | **Run probe DVOL-come-feature** (pre-reg 2026-07-17). Prerequisito CPU pre-finestra **FATTO 2026-07-17**: patch `QUANTSYS_DATASET_NPZ` (02_train + giudice) verificata INERTE (7 test `tests/test_dataset_npz_flag.py`) + `scripts/vol/dev_vols_dvol_append.py` → `lstm_dataset_dvol.npz` (assert bit-identità sul resto; npz production NON toccato). Run: `QUANTSYS_ARCH=itransformer`, sandbox `models_dvol_probe`, 5 seed | stesse 3 condizioni su val | la IV risk-neutral (DVOL 30d, unica serie IV con storia 2021→) aggiunge contenuto predittivo oltre price+volume+macro? (prior onesto: nullo/sotto soglia per tenor mismatch 30d vs 30h) |
| B5 | Esiti scritti in STATUS **comunque**; PASS → one-shot su test; poi riavvio processi (Fase C) | — | — |

**EN** Binding order (pre-regs already in STATUS; independent runs vs the same incumbent; any A3×A8×DVOL interaction = new pre-reg):
| # | Action | Test/judge | Expected information |
|---|---|---|---|
| B1 | **`causality-auditor` audit of the A3 files** (P3, never run) | read-only audit | no residual lookahead in the MoE gate/training before spending GPU |
| B2 | **A3 regime-MoE run** (`QUANTSYS_ARCH=itransformer_regime_moe`, `models_a3_moe` sandbox, 5 seeds) | `dev_vols_qlike.py` on val: ① QLIKE ≤0.97·incumbent ② no regime destroyed (≤1.05 in the worst) ③ n≥5000/≥800 | do 3 regime heads with a causal gate beat the incumbent on μ? (measures μ, NOT σ — declared MINOR-1) |
| B3 | **A8 mixup run** (`QUANTSYS_ARCH=itransformer_a8_mixup`, `models_a8_mixup` sandbox, 5 seeds) | same 3 conditions on val | does the only cross-feature-coherent augmentation reduce residual overfit? (honest prior: FAIL/null) |
| B4 | **DVOL-as-feature probe run** (pre-reg 2026-07-17). Pre-window CPU prerequisite **DONE 2026-07-17**: `QUANTSYS_DATASET_NPZ` patch (02_train + judge) verified INERT (7 tests `tests/test_dataset_npz_flag.py`) + `scripts/vol/dev_vols_dvol_append.py` → `lstm_dataset_dvol.npz` (bit-identity assert on the rest; production npz UNTOUCHED). Run: `QUANTSYS_ARCH=itransformer`, `models_dvol_probe` sandbox, 5 seeds | same 3 conditions on val | does risk-neutral IV (DVOL 30d, the only IV series with 2021→ history) add predictive content beyond price+volume+macro? (honest prior: null/below threshold due to the 30d-vs-30h tenor mismatch) |
| B5 | Outcomes written to STATUS **regardless**; PASS → one-shot on test; then restart processes (Phase C) | — | — |

---

## 3 · Fase C — Implementazioni + riavvio v2 · Phase C — Implementations + v2 restart

🇮🇹
| # | Azione | Test/verifica | Informazione/conferma attesa |
|---|---|---|---|
| C1 | **Fix funding-refresh per-tick in `04b`** (oggi solo all'avvio, `__init__`: feature funding stale con l'uptime — causa del residuo Δμ replay-vs-live) | replay A/B `vol_paper_replay.py` post-fix: il residuo di parità sui tick di processo lungo deve → 0 | live = replay bit-coerenti → il gap-filler è pienamente fedele |
| C2 | **Refactor 2ter** (stessa campagna di parità di C1): ① `canonical_feature_columns()` in `quantsys/features` + golden test lista-104; ② client Deribit pubblico unico in `quantsys/data/deribit.py` (assert anti-mainnet preservato) + delivery-cache unica; ③ `VolForecaster` promosso da 04b a `quantsys/model/` | prova bit-perfetta pattern A/B (Δfeature=0, Δμ=0) per ogni estrazione | elimina la classe di bug "lista duplicata che deriva" (classe z-score) senza cambiare i bit |
| C3 | **Riavvio `04b --execute --hedge --hedge-band <frozen> --hedge-ww-lambda <frozen> --hedge-conv <frozen>`** → parte il forward v2 | giudice `hedged_vs_unhedged_judge.py` a n≥20 hedge-attivi: ① var ratio ≤0.6 ② drag medio ≤¼·SE ③ n≥20 | l'hedge compra varianza senza pagare in media? (B2: purificazione del VRP net-of-costs) |
| C4 | **VPS:** `--greeks` su `quantsys-iv.service` (+ daemon-reload) e cadenza poller 10→5 min. ⚠ **Includere il sync casa** (nota 2026-07-16): `atm_greeks.parquet` NON è nel pull/merge — aggiungere la riga file-singolo in `pull_vps_data.ps1`, l'entry in `SINGLE_FILES` di `merge_vps_data.py` (dedup `timestamp`) e il check freshness in `health_check.sh`; senza, il VPS accumula ma casa non la vede | smoke: `atm_greeks.parquet` cresce, selezione = `pick_straddle`, e dopo un pull la copia canonica di casa cresce anch'essa | rende la leg hedge v2 replayabile offline + valida la convenzione δ sul campo |

**EN**
| # | Action | Test/verification | Expected information/confirmation |
|---|---|---|---|
| C1 | **Per-tick funding refresh fix in `04b`** (today startup-only, `__init__`: funding features go stale with uptime — cause of the replay-vs-live Δμ residual) | post-fix A/B replay `vol_paper_replay.py`: the parity residual on long-uptime ticks must → 0 | live = replay bit-coherent → the gap-filler is fully faithful |
| C2 | **2ter refactor** (same parity campaign as C1): ① `canonical_feature_columns()` in `quantsys/features` + 104-list golden test; ② single public Deribit client in `quantsys/data/deribit.py` (anti-mainnet assert preserved) + single delivery cache; ③ `VolForecaster` promoted from 04b to `quantsys/model/` | bit-perfect A/B proof (Δfeature=0, Δμ=0) per extraction | kills the "drifting duplicated list" bug class (the z-score class) without changing bits |
| C3 | **Restart `04b --execute --hedge --hedge-band <frozen> --hedge-ww-lambda <frozen> --hedge-conv <frozen>`** → v2 forward starts | `hedged_vs_unhedged_judge.py` at n≥20 hedge-active: ① var ratio ≤0.6 ② mean drag ≤¼·SE ③ n≥20 | does the hedge buy variance without paying in mean? (B2: VRP purification net-of-costs) |
| C4 | **VPS:** `--greeks` on `quantsys-iv.service` (+ daemon-reload) and poller cadence 10→5 min. ⚠ **Include the home sync** (2026-07-16 note): `atm_greeks.parquet` is NOT in the pull/merge — add the single-file line in `pull_vps_data.ps1`, the `SINGLE_FILES` entry in `merge_vps_data.py` (`timestamp` dedup) and the freshness check in `health_check.sh`; without these the VPS accumulates but home never sees it | smoke: `atm_greeks.parquet` grows, selection = `pick_straddle`, and after one pull the home canonical copy grows too | makes the v2 hedge leg offline-replayable + field-validates the δ convention |

---

## 4 · Fase D — Code (dopo C) · Phase D — Queue (after C)

🇮🇹
1. **Valutazione pre-registrata a n=30** (leg opzioni, §0.2) — stessi 3 criteri, stessi 2 campioni; SOLO dopo: pre-registrazione **sizing v2** (A13 pin-close + A14 vega-sizing + A7 cablaggio greeks-risk, coda di rischio da **HAR-q90** — esito A2 definitivo).
2. **Sblocco candele oltre il 2026-06-22 + refresh macro** (`01b`; B7 incrementale scatta da solo via `avvio_sessione`) — SOLO a esperimenti A3/A8 chiusi (lo span congelato è il loro invariante).
3. **Eventuale retrain con A4 (HAR-CJ) / A9 (MaxPool)** = rigen dataset + gate QLIKE **da pre-registrare** (nuova sezione STATUS; A10 sparsity solo se A8 delude).
4. **Derivazione OFFLINE `mfiv_30h` + skew (25Δ RR/BF) dal raw chain** (CPU-only, retroattiva su tutto il periodo di raccolta; colonne registrate parallele, MAI nel path decisionale). Eventuale promozione a comparatore dell'edge di `04b` = **NUOVA pre-reg v2** con break-even ri-stimato (wedge di convessità MFIV vs IV ATM). Indipendente dall'esito del probe DVOL (B4); dettaglio: memoria `idea_mfiv_30h`.
5. **Decisioni utente:** pubblicazione GitHub (audit secret PASS 07-14); migrazione `04b`→VPS (insieme al fix C1 già fatto); tenor ladder v2 (memoria, richiede nuova pre-reg). Promemoria: **disdetta netcup ~dicembre 2026**.
6. A piano completato: eliminare questo file (checklist esaurita, esiti in STATUS).

**EN**
1. **Pre-registered n=30 evaluation** (options leg, §0.2) — same 3 criteria, same 2 samples; ONLY afterwards: **v2 sizing** pre-registration (A13 pin-close + A14 vega-sizing + A7 greeks-risk wiring, tail risk from **HAR-q90** — definitive A2 outcome).
2. **Unfreeze candles past 2026-06-22 + macro refresh** (`01b`; incremental B7 fires by itself via `avvio_sessione`) — ONLY once the A3/A8 experiments are closed (the frozen span is their invariant).
3. **Possible retrain with A4 (HAR-CJ) / A9 (MaxPool)** = dataset regen + QLIKE gate **to pre-register** (new STATUS section; A10 sparsity only if A8 disappoints).
4. **OFFLINE derivation of `mfiv_30h` + skew (25Δ RR/BF) from the raw chain** (CPU-only, retroactive over the whole collection period; parallel recorded columns, NEVER in the decision path). Possible promotion to `04b` edge comparator = **NEW v2 pre-reg** with re-estimated break-even (MFIV-vs-ATM-IV convexity wedge). Independent of the DVOL probe outcome (B4); detail: `idea_mfiv_30h` memory.
5. **User decisions:** GitHub publication (secret audit PASS 07-14); `04b`→VPS migration (together with the already-done C1 fix); v2 tenor ladder (memory, needs a new pre-reg). Reminder: **netcup cancellation ~December 2026**.
6. When the plan is complete: delete this file (checklist exhausted, outcomes in STATUS).
