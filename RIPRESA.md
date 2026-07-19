# RIPRESA — lista residua · remaining list

> 🇮🇹 Riscritto 2026-07-20 (v4, roadmap riordinata su decisione utente). File EFFIMERO: eliminarlo a lista esaurita. Pre-registrazioni vincolanti e dettaglio: `STATUS.md` (pre-reg A8-BIS in cima + sezioni 2026-07-19/20).
> **EN** Rewritten 2026-07-20 (v4, roadmap reordered by user decision). EPHEMERAL file: delete when exhausted. Binding pre-registrations and detail: `STATUS.md` (A8-BIS pre-reg on top + 2026-07-19/20 sections).

## Fatto · Done

🇮🇹 **2026-07-19:** B2 A3-MoE e B3 A8-mixup giudicati "NESSUNA CONCLUSIONE" (③ model-independent); dataset RICOSTRUITO su span esteso; pre-reg A8-BIS committata (`0e4a73b`). **2026-07-20:** decisione utente pattern-③ standard + roadmap riordinata; **A8-BIS ESEGUITO E CHIUSO: FAIL su val** — baseline 0.26206 vs mixup 0.25998, ratio 0.9921 (−0.79% ≪ −3% della ①) → mixup FALLITO definitivo sul dataset esteso, overlay eliminato, niente test; il descrittivo −4.94% di B3 era artefatto di distribution shift (incumbent old-scaler). A10 unico candidato training residuo. Invarianti npz DECADUTI. Dettaglio: STATUS 2026-07-20 ④.
**EN** 2026-07-19: B2+B3 judged "no conclusion" (③); dataset rebuilt; A8-BIS pre-reg committed. 2026-07-20: standard ③-pattern decision + roadmap reordered; **A8-BIS RUN AND CLOSED: FAIL on val** (−0.79% ≪ −3%) → mixup definitively FAILED, overlay deleted, no test; the −4.94% descriptive was a distribution-shift artifact. A10 = only residual training candidate. Npz invariants LIFTED. Detail: STATUS 2026-07-20 ④.

## Da fare (ordine = roadmap corrente) · To do (order = current roadmap)

1. 🇮🇹 **MFIV-comparatore v2 — ✅ PRE-REG SCRITTA 2026-07-20** (in cima a STATUS): gate Δρ Spearman appaiato MFIV-vs-ATM sui PnL short-straddle per-expiry, ③ n≥40 qualificati (oggi 15, +~1/giorno → run one-shot ~metà agosto). ✅ **Giudice SCRITTO 2026-07-20** (`scripts/vol/mfiv_comparator_judge.py`, smoke sintetico 3/3, guard n≥40 fail-fast; conteggio con regola causale = 12). Residuo: attesa campione (+~1/giorno) — a ogni sessione: `derive_mfiv.py` (derivazione incrementale) + `mfiv_comparator_judge.py --count-only`; run one-shot alla prima sessione con n≥40 (~metà agosto).
   **EN** MFIV-comparator v2 — ✅ PRE-REG WRITTEN + JUDGE WRITTEN 2026-07-20 (synthetic smoke 3/3; causal-rule count = 12). Remaining: sample wait — per session run incremental derivation + `--count-only`; one-shot at the first session with n≥40 (~mid-August).

2. 🇮🇹 **B4-bis DVOL:** ri-derivare `lstm_dataset_dvol.npz` dal npz NUOVO (`dev_vols_dvol_append.py`) + NUOVA pre-reg col pattern-③ standard (invariante npz di A8-BIS decaduto: si può procedere).
   **EN** B4-bis DVOL: re-derive the dvol npz from the NEW npz + new pre-reg with the standard ③-pattern (the A8-BIS npz invariant has lapsed: clear to proceed).

3. 🇮🇹 **A3-bis regime-MoE: PARCHEGGIATO** (prior sfavorevole, descrittivo −2.02% < 3%; il ramo "baseline cambia con PASS mixup" è decaduto col FAIL): rivalutare SOLO se un episodio stress porta massa a r1. **CAFN: parcheggiato a prior basso**, riapribile solo con re-scope. **A10 sparsity = unico candidato training residuo** (prior basso, effort M). Razionale: STATUS 2026-07-20 ③④.
   **EN** A3-bis PARKED (the "baseline changes on mixup PASS" branch lapsed with the FAIL); CAFN parked low-prior; A10 sparsity = only residual training candidate (rationale: STATUS 2026-07-20 ③④).

4. 🇮🇹 **~Fine luglio:** valutazione pre-registrata **n≥30** leg opzioni (POST_GATE_V1 §0.2); solo dopo: pre-reg sizing v2 (A13+A14+A7).
   **EN** ~End of July: pre-registered n≥30 evaluation; only then the v2 sizing pre-reg.

5. 🇮🇹 **~Metà agosto:** giudice `hedged_vs_unhedged_judge.py` a n≥20 hedge-attivi.
   **EN** ~Mid August: hedged-vs-unhedged judge at n≥20 hedge-active trades.

6. 🇮🇹 Igiene disco: ✅ COMPLETATA 2026-07-20 — 4 sandbox (73 MB) + `lstm_dataset_frozen0622.npz` + `lstm_dataset_dvol.npz` stale + `pipeline_state_frozen0622.pkl` ELIMINATI (~6,4 GB; tutti rigenerabili/obsoleti; report giudice preservati in `results/vols/`). Azzerare gli env sperimentali nel terminale (Passo D) se non già fatto.
   **EN** Disk hygiene: ✅ DONE 2026-07-20 — 4 sandboxes + both npz backups + frozen pkl DELETED (~6.4 GB; all regenerable/obsolete; judge reports preserved). Clear experimental env vars if not already done.

7. 🇮🇹 A lista esaurita: eliminare questo file (esiti in STATUS).
   **EN** When exhausted: delete this file (outcomes in STATUS).
