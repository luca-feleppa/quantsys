# RIPRESA — lista residua · remaining list

> 🇮🇹 Riscritto 2026-07-20 (v4, roadmap riordinata su decisione utente). File EFFIMERO: eliminarlo a lista esaurita. Pre-registrazioni vincolanti e dettaglio: `STATUS.md` (pre-reg A8-BIS in cima + sezioni 2026-07-19/20).
> **EN** Rewritten 2026-07-20 (v4, roadmap reordered by user decision). EPHEMERAL file: delete when exhausted. Binding pre-registrations and detail: `STATUS.md` (A8-BIS pre-reg on top + 2026-07-19/20 sections).

## Fatto · Done

🇮🇹 **2026-07-19:** B2 A3-MoE e B3 A8-mixup giudicati "NESSUNA CONCLUSIONE" (③ model-independent); dataset RICOSTRUITO su span esteso; pre-reg A8-BIS committata (`ebda722`). **2026-07-20:** decisione utente pattern-③ standard + roadmap riordinata; **A8-BIS ESEGUITO E CHIUSO: FAIL su val** — baseline 0.26206 vs mixup 0.25998, ratio 0.9921 (−0.79% ≪ −3% della ①) → mixup FALLITO definitivo sul dataset esteso, overlay eliminato, niente test; il descrittivo −4.94% di B3 era artefatto di distribution shift (incumbent old-scaler). A10 unico candidato training residuo. Invarianti npz DECADUTI. Dettaglio: STATUS 2026-07-20 ④.
**EN** 2026-07-19: B2+B3 judged "no conclusion" (③); dataset rebuilt; A8-BIS pre-reg committed. 2026-07-20: standard ③-pattern decision + roadmap reordered; **A8-BIS RUN AND CLOSED: FAIL on val** (−0.79% ≪ −3%) → mixup definitively FAILED, overlay deleted, no test; the −4.94% descriptive was a distribution-shift artifact. A10 = only residual training candidate. Npz invariants LIFTED. Detail: STATUS 2026-07-20 ④.

🇮🇹 **2026-07-23:** **B4-bis DVOL-come-feature CHIUSO FAIL su val** (pre-reg `5a6112d`, close `526659d`): candidato 0.25939 vs baseline riaddestrata 0.26206 = −1.02%, soglia −3% → ① FAIL, nessun one-shot su test. Filone chiuso. **2026-07-25:** igiene disco B4-bis verificata già fatta; routine di sessione automatizzata (blocco ③ di `avvio_sessione.ps1`). **2026-07-26:** fix unità di misura del contatore hedged (posizioni hedge-attive, NON eventi di ledger: rischio one-shot introdotto dall'automazione) + commit del doc-refactor 25/07 (scorporo archivio verificato letterale, 0 righe perse). Nessun item GPU attivo residuo.
**EN** 2026-07-23: **B4-bis DVOL-as-feature CLOSED FAIL on val** (−1.02% vs the −3% threshold → ① FAIL, no test one-shot); line closed. 2026-07-25: B4-bis disk hygiene verified done; session routine automated (block ③ of `avvio_sessione.ps1`). 2026-07-26: hedged-counter unit fixed (hedge-active positions, NOT ledger events — a one-shot risk introduced by yesterday's automation) + 25/07 doc-refactor committed (archive split verified literal, 0 lines lost). No residual active GPU item.

## Da fare (ordine = roadmap corrente) · To do (order = current roadmap)

1. 🇮🇹 **MFIV-comparatore v2 — ✅ PRE-REG + GIUDICE SCRITTI 2026-07-20** (in cima a STATUS): gate Δρ Spearman appaiato MFIV-vs-ATM sui PnL short-straddle per-expiry, ③ n≥40 qualificati (**19 il 26/07**, +~1/giorno → run one-shot ~metà agosto). Residuo: **solo attesa campione** — il monitoraggio per-sessione (`derive_mfiv.py` + `--count-only`) è **automatizzato dal blocco ③ di `avvio_sessione.ps1`** dal 2026-07-25: niente da lanciare a mano. Run one-shot **MANUALE** alla prima sessione in cui la routine stampa ≥40.
   **EN** MFIV-comparator v2 — ✅ PRE-REG + JUDGE WRITTEN 2026-07-20 (**18 qualifying on 25/07**). Remaining: sample wait only — per-session monitoring is now **automated by block ③ of `avvio_sessione.ps1`** (since 2026-07-25). The one-shot run stays **MANUAL** at the first session printing ≥40.

2. 🇮🇹 **A3-bis regime-MoE: PARCHEGGIATO** (prior sfavorevole, descrittivo −2.02% < 3%; il ramo "baseline cambia con PASS mixup" è decaduto col FAIL): rivalutare SOLO se un episodio stress porta massa a r1. **CAFN: parcheggiato a prior basso**, riapribile solo con re-scope. **A10 sparsity = unico candidato training residuo** (prior basso, effort M). Razionale: STATUS 2026-07-20 ③④.
   **EN** A3-bis PARKED (the "baseline changes on mixup PASS" branch lapsed with the FAIL); CAFN parked low-prior; A10 sparsity = only residual training candidate (rationale: STATUS 2026-07-20 ③④).

3. 🇮🇹 **~29/07 (n=27 il 26/07, +~1/giorno):** valutazione pre-registrata **n≥30** leg opzioni (POST_GATE_V1 §0.2); subito dopo: refresh macro `01b --skip-regime` (rimandato per non perturbare il live a campione aperto — STATUS 25/07 ⑤); solo dopo: pre-reg sizing v2 (A13+A14+A7).
   **EN** ~29/07 (n=27 on 26/07): pre-registered n≥30 evaluation; right after: macro refresh `01b --skip-regime` (deferred to avoid perturbing the live path mid-sample — STATUS 25/07 ⑤); only then the v2 sizing pre-reg.

4. 🇮🇹 **~09-10/08:** giudice `hedged_vs_unhedged_judge.py` a n≥20 **posizioni hedge-attive** (26/07: **n=6**, +~1/giorno — il ledger ha 22 *eventi*, unità diversa: vedi STATUS 26/07 ③).
   **EN** ~09-10/08: hedged-vs-unhedged judge at n≥20 **hedge-active positions** (26/07: n=6; the ledger holds 22 *events* — different unit, see STATUS 26/07 ③).

5. 🇮🇹 A lista esaurita: eliminare questo file (esiti in STATUS).
   **EN** When exhausted: delete this file (outcomes in STATUS).
