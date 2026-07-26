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
| A1 | ✅ **FATTO 2026-07-18** — **Report di chiusura gate v1** sui 3 criteri congelati 2026-06-12: ① PnL medio/trade > 0 net; ② PnL totale > always-long-vol E always-short-vol sullo stesso calendario; ③ hit-rate > 0.5 | `scripts/04c_vol_paper_baselines.py` (criterio ②) + aggregati da `trades.jsonl` (①③), su ENTRAMBI i campioni (§0.1) | **ESITO: FAIL 0/3 su entrambi i campioni** (dettaglio in STATUS 2026-07-18) — il timing batte always-long ma non lo short: VRP positivo domina |
| A2 | ✅ **FATTO 2026-07-18** (inclusi nel report STATUS) — **caveat selezione oraria** (⑤bis, dati pre-14/07 = 18.6% coverage), **nota dedup** (18 righe→17), **nota trade #0** (§0.1), **nota 20-vs-30** (§0.2) | — | delimitano la validità ESTERNA del verdetto, non le soglie |
| A3 | ✅ **FATTO 2026-07-18** — **`vol_paper_replay.py` attivato come gap-filler ufficiale** (griglia dal 2026-07-14T14Z, 95 tick) | parity check: 64% accordo segnale — il residuo QUANTIFICA il bug candele del live (finestra bucata dal ~06-24, vedi STATUS) | serie forecast H24 senza bias orario; post-migrazione VPS il live È H24 → replay solo per outage |
| A4 | ✅ **FATTO 2026-07-18** — **CONGELATI: band=0.30 fixed, conv=raw, NO ww** (dominanza assente: Δnet 4e-5=rumore, var↓ peggiore); var↓ 64.9%; slope +0.19±0.07 non matcha bene nessuna δ (caveat theta/vega) → pre-reg v2 aggiornata in STATUS | dry-run offline su dati DISGIUNTI dal giudizio forward | parametri v2 congelati su dati pre-attivazione → zero gradi di libertà a giudizio in corso |

**EN**
| # | Action | Test/judge | Expected information/confirmation |
|---|---|---|---|
| A1 | ✅ **DONE 2026-07-18** — **v1-gate closure report** on the 3 frozen 2026-06-12 criteria: ① mean PnL/trade > 0 net; ② total PnL > always-long-vol AND always-short-vol on the same calendar; ③ hit-rate > 0.5 | `scripts/04c_vol_paper_baselines.py` (criterion ②) + aggregates from `trades.jsonl` (①③), on BOTH samples (§0.1) | **OUTCOME: FAIL 0/3 on both samples** (detail in STATUS 2026-07-18) — timing beats always-long but not short: positive VRP dominates |
| A2 | ✅ **DONE 2026-07-18** (included in the STATUS report) — **hour-selection caveat** (⑤bis, pre-14/07 data = 18.6% coverage), **dedup note** (18 rows→17), **trade #0 note** (§0.1), **20-vs-30 note** (§0.2) | — | they bound the verdict's EXTERNAL validity, not the thresholds |
| A3 | ✅ **DONE 2026-07-18** — **`vol_paper_replay.py` activated as the official gap-filler** (grid from 2026-07-14T14Z, 95 ticks) | parity check: 64% signal agreement — the residual QUANTIFIES the live candle bug (holed window since ~06-24, see STATUS) | 24/7 forecast series without hour bias; after the VPS migration live IS 24/7 → replay for outages only |
| A4 | ✅ **DONE 2026-07-18** — **FROZEN: band=0.30 fixed, conv=raw, NO ww** (no dominance: Δnet 4e-5=noise, worse var↓); var↓ 64.9%; slope +0.19±0.07 matches neither δ well (theta/vega caveat) → v2 pre-reg updated in STATUS | offline dry-run on data DISJOINT from the forward judgment | v2 parameters frozen on pre-activation data → zero degrees of freedom mid-judgment |

---

## 2 · Fase B — Finestra GPU (04b fermo, poche ore) · Phase B — GPU window (04b stopped, a few hours)

🇮🇹 Ordine vincolante (pre-reg già scritte in STATUS; run indipendenti vs lo stesso incumbent; ogni interazione A3×A8×DVOL = nuova pre-reg):
| # | Azione | Test/giudice | Informazione attesa |
|---|---|---|---|
| B1 | ✅ **FATTO 2026-07-18** — **Audit `causality-auditor` sui file A3** | audit read-only | GPU-GO: nessun lookahead residuo nel gate/training MoE |
| B2 | ✅ **FATTO 2026-07-19** — **Run A3 regime-MoE** eseguito e giudicato | `dev_vols_qlike.py` su val | **ESITO: NESSUNA CONCLUSIONE** (③ model-independent, r1=657<800 sul val congelato; descrittivo −2.02%, sotto la soglia ①) → **A3-bis PARCHEGGIATO** (prior sfavorevole; condizioni di rivalutazione in STATUS 2026-07-20) |
| B3 | ✅ **FATTO 2026-07-19** — **Run A8 mixup** eseguito e giudicato | stesse condizioni su val | **ESITO: NESSUNA CONCLUSIONE** (stessa ③); descrittivo −4.94% → ri-pre-reg A8-BIS su dataset esteso → **FAIL DEFINITIVO su val 2026-07-20** (−0.79% vs baseline riaddestrata: il −4.94% era artefatto di distribution shift; STATUS 2026-07-20 ④) |
| B4 | ✅ **FATTO 2026-07-23 come B4-bis** — npz ri-derivato (`dev_vols_dvol_append.py`, X_macro 90→93, copertura val 1.000), pre-reg `e3a9e97` committata PRIMA di ogni numero, run baseline+candidato in sandbox | `dev_vols_qlike.py` su val | **ESITO: FAIL su val** — candidato 0.25939 vs baseline riaddestrata 0.26206 = **−1.02%**, soglia −3% → ① FAIL (② sarebbe passata, gate in AND), nessun one-shot su test. Il prior onesto era corretto: tenor mismatch 30d→30h + incumbent che già cattura la IV via lag RV (MSE-log −6% ma QLIKE no). **Filone DVOL-come-feature chiuso** (close `b463763`) |
| B5 | Esiti B2/B3 scritti in STATUS (fatto); lezione di processo: condizioni campionarie model-independent verificate EX-ANTE d'ora in poi (pattern-③ standard, STATUS 2026-07-20) | — | — |

**EN** Binding order (pre-regs already in STATUS; independent runs vs the same incumbent; any A3×A8×DVOL interaction = new pre-reg):
| # | Action | Test/judge | Expected information |
|---|---|---|---|
| B1 | ✅ **DONE 2026-07-18** — **`causality-auditor` audit of the A3 files** | read-only audit | GPU-GO: no residual lookahead in the MoE gate/training |
| B2 | ✅ **DONE 2026-07-19** — **A3 regime-MoE run** executed and judged | `dev_vols_qlike.py` on val | **OUTCOME: NO CONCLUSION** (model-independent ③, r1=657<800 on the frozen val; descriptive −2.02%, below threshold ①) → **A3-bis PARKED** (unfavorable prior; re-evaluation conditions in STATUS 2026-07-20) |
| B3 | ✅ **DONE 2026-07-19** — **A8 mixup run** executed and judged | same conditions on val | **OUTCOME: NO CONCLUSION** (same ③); descriptive −4.94% → A8-BIS re-pre-reg on the extended dataset → **DEFINITIVE FAIL on val 2026-07-20** (−0.79% vs the retrained baseline: the −4.94% was a distribution-shift artifact; STATUS 2026-07-20 ④) |
| B4 | **DEFERRED 2026-07-19 → B4-bis** (never run, zero numbers seen): re-derive `lstm_dataset_dvol.npz` from the NEW npz (`dev_vols_dvol_append.py`) + NEW pre-reg with the standard ③-pattern (STATUS 2026-07-20 ②) — only AFTER A8-BIS closes | same conditions on val | does risk-neutral IV (DVOL 30d) add predictive content beyond price+volume+macro? (honest prior: null/below threshold due to the 30d-vs-30h tenor mismatch) |
| B5 | B2/B3 outcomes written to STATUS (done); process lesson: model-independent sample conditions verified EX-ANTE from now on (standard ③-pattern, STATUS 2026-07-20) | — | — |

---

## 3 · Fase C — Implementazioni + riavvio v2 · Phase C — Implementations + v2 restart

🇮🇹
| # | Azione | Test/verifica | Informazione/conferma attesa |
|---|---|---|---|
| C1 | ✅ **FATTO 2026-07-18** (incluso nella migrazione VPS) — **funding-refresh per-tick in `04b`** (fail-soft; + bootstrap candele gap-aware e guard contiguità, vedi STATUS) | replay A/B sui tick futuri sovrapposti VPS-vs-replay: residuo atteso → 0 (verifica pendente) | live = replay bit-coerenti → il gap-filler è pienamente fedele |
| C2 | **Refactor 2ter** (stessa campagna di parità di C1): ① `canonical_feature_columns()` in `quantsys/features` + golden test lista-104; ② client Deribit pubblico unico in `quantsys/data/deribit.py` (assert anti-mainnet preservato) + delivery-cache unica; ③ `VolForecaster` promosso da 04b a `quantsys/model/` | prova bit-perfetta pattern A/B (Δfeature=0, Δμ=0) per ogni estrazione | elimina la classe di bug "lista duplicata che deriva" (classe z-score) senza cambiare i bit |
| C3 | ✅ **FATTO 2026-07-18** — servizio VPS riavviato con `--execute --hedge --hedge-band 0.30 --hedge-conv raw` (congelati A4; NO ww) → **forward v2 PARTITO**; posizione parzialmente hedgiata ESCLUSA dal campione (pre-dichiarato in STATUS) | giudice `hedged_vs_unhedged_judge.py` a n≥20 hedge-attivi: ① var ratio ≤0.6 ② drag medio ≤¼·SE ③ n≥20 | l'hedge compra varianza senza pagare in media? (B2: purificazione del VRP net-of-costs) |
| C4 | **VPS:** `--greeks` su `quantsys-iv.service` (+ daemon-reload) e cadenza poller 10→5 min. ⚠ **Includere il sync casa** (nota 2026-07-16): `atm_greeks.parquet` NON è nel pull/merge — aggiungere la riga file-singolo in `pull_vps_data.ps1`, l'entry in `SINGLE_FILES` di `merge_vps_data.py` (dedup `timestamp`) e il check freshness in `health_check.sh`; senza, il VPS accumula ma casa non la vede | smoke: `atm_greeks.parquet` cresce, selezione = `pick_straddle`, e dopo un pull la copia canonica di casa cresce anch'essa | rende la leg hedge v2 replayabile offline + valida la convenzione δ sul campo |

**EN**
| # | Action | Test/verification | Expected information/confirmation |
|---|---|---|---|
| C1 | ✅ **DONE 2026-07-18** (part of the VPS migration) — **per-tick funding refresh in `04b`** (fail-soft; + gap-aware candle bootstrap and contiguity guard, see STATUS) | A/B replay on future overlapping VPS-vs-replay ticks: residual expected → 0 (verification pending) | live = replay bit-coherent → the gap-filler is fully faithful |
| C2 | **2ter refactor** (same parity campaign as C1): ① `canonical_feature_columns()` in `quantsys/features` + 104-list golden test; ② single public Deribit client in `quantsys/data/deribit.py` (anti-mainnet assert preserved) + single delivery cache; ③ `VolForecaster` promoted from 04b to `quantsys/model/` | bit-perfect A/B proof (Δfeature=0, Δμ=0) per extraction | kills the "drifting duplicated list" bug class (the z-score class) without changing bits |
| C3 | ✅ **DONE 2026-07-18** — VPS service restarted with `--execute --hedge --hedge-band 0.30 --hedge-conv raw` (A4-frozen; NO ww) → **v2 forward STARTED**; partially-hedged open position EXCLUDED from the sample (pre-declared in STATUS) | `hedged_vs_unhedged_judge.py` at n≥20 hedge-active: ① var ratio ≤0.6 ② mean drag ≤¼·SE ③ n≥20 | does the hedge buy variance without paying in mean? (B2: VRP purification net-of-costs) |
| C4 | **VPS:** `--greeks` on `quantsys-iv.service` (+ daemon-reload) and poller cadence 10→5 min. ⚠ **Include the home sync** (2026-07-16 note): `atm_greeks.parquet` is NOT in the pull/merge — add the single-file line in `pull_vps_data.ps1`, the `SINGLE_FILES` entry in `merge_vps_data.py` (`timestamp` dedup) and the freshness check in `health_check.sh`; without these the VPS accumulates but home never sees it | smoke: `atm_greeks.parquet` grows, selection = `pick_straddle`, and after one pull the home canonical copy grows too | makes the v2 hedge leg offline-replayable + field-validates the δ convention |

---

## 4 · Fase D — Code (dopo C) · Phase D — Queue (after C)

🇮🇹
1. **Valutazione pre-registrata a n=30** (leg opzioni, §0.2) — stessi 3 criteri, stessi 2 campioni; SOLO dopo: pre-registrazione **sizing v2** (A13 pin-close + A14 vega-sizing + A7 cablaggio greeks-risk, coda di rischio da **HAR-q90** — esito A2 definitivo).
2. ✅ **FATTO 2026-07-19** — **Sblocco candele + refresh macro**: rebuild dataset esteso →2026-07-19 (`01b --skip-regime` label-preserving + regime incrementale +27 barre); nuovo invariante = npz esteso congelato fino a chiusura A8-BIS.
3. **Eventuale retrain con A4 (HAR-CJ) / A9 (MaxPool)** = rigen dataset + gate QLIKE **da pre-registrare** (nuova sezione STATUS). A8-BIS FALLITO 2026-07-20 → **A10 sparsity = unico candidato training residuo** (prior basso, nessuna urgenza).
4. **Derivazione `mfiv_30h`**: ✅ **FATTA** (D4 2026-07-18, incrementale periodica via `scripts/vol/derive_mfiv.py`; wedge MFIV−ATM stabile +3.39 vol pt su 1.911 tick). **Promozione a comparatore dell'edge di `04b` = pre-reg v2 PROMOSSA nella roadmap 2026-07-20** (subito dopo la chiusura A8-BIS, prima di B4-bis): break-even short-vol ri-stimato col wedge di convessità. Indipendente dall'esito del probe DVOL; dettaglio: memoria `idea_mfiv_30h`.
5. **Decisioni utente:** pubblicazione GitHub (audit secret PASS 07-14); ✅ migrazione `04b`→VPS **FATTA 2026-07-18** (con fix C1; servizio `quantsys-volpaper` attivo, health PASS); tenor ladder v2 (memoria, richiede nuova pre-reg). Promemoria: **disdetta VPS ~dicembre 2026**.
6. A piano completato: eliminare questo file (checklist esaurita, esiti in STATUS).

**EN**
1. **Pre-registered n=30 evaluation** (options leg, §0.2) — same 3 criteria, same 2 samples; ONLY afterwards: **v2 sizing** pre-registration (A13 pin-close + A14 vega-sizing + A7 greeks-risk wiring, tail risk from **HAR-q90** — definitive A2 outcome).
2. ✅ **DONE 2026-07-19** — **Candle unfreeze + macro refresh**: extended dataset rebuilt →2026-07-19 (label-preserving `01b --skip-regime` + incremental regime +27 bars); new invariant = extended npz frozen until A8-BIS closes.
3. **Possible retrain with A4 (HAR-CJ) / A9 (MaxPool)** = dataset regen + QLIKE gate **to pre-register** (new STATUS section). A8-BIS FAILED 2026-07-20 → **A10 sparsity = only residual training candidate** (low prior, no urgency).
4. **`mfiv_30h` derivation**: ✅ **DONE** (D4 2026-07-18, periodic incremental via `scripts/vol/derive_mfiv.py`; MFIV−ATM wedge stable at +3.39 vol pt over 1,911 ticks). **Promotion to `04b` edge comparator = v2 pre-reg PROMOTED in the 2026-07-20 roadmap** (right after A8-BIS closes, ahead of B4-bis): short-vol break-even re-estimated with the convexity wedge. Independent of the DVOL probe outcome; detail: `idea_mfiv_30h` memory.
5. **User decisions:** GitHub publication (secret audit PASS 07-14); ✅ `04b`→VPS migration **DONE 2026-07-18** (with the C1 fix; `quantsys-volpaper` service active, health PASS); v2 tenor ladder (memory, needs a new pre-reg). Reminder: **VPS cancellation ~December 2026**.
6. When the plan is complete: delete this file (checklist exhausted, outcomes in STATUS).
