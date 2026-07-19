# RIPRESA — lista residua · remaining list

> 🇮🇹 Riscritto 2026-07-18 sera (v2): la checklist post-riavvio originale è ESAURITA salvo la Fase B GPU (training stoppato su decisione utente) e gli item time-gated. File EFFIMERO: eliminarlo a lista esaurita. Pre-registrazioni vincolanti e dettaglio: `STATUS.md` (sezione 2026-07-18 sera) e `POST_GATE_V1.md`.
> **EN** Rewritten 2026-07-18 evening (v2): the original post-reboot checklist is DONE except the GPU Phase B (training stopped by user decision) and the time-gated items. EPHEMERAL file: delete when exhausted. Binding pre-registrations and detail: `STATUS.md` (2026-07-18 evening section) and `POST_GATE_V1.md`.

## Fatto stasera · Done tonight

🇮🇹 Pull+merge OK (heartbeat 4/4) · B7 regime refresh eseguito · **B1 audit A3 → GPU-GO** (3 MINOR fixati, 22/22 test) · B2 training A3 partito e **STOPPATO** (sandbox `models_a3_moe` eliminata, pre-reg intatta) · **C4 FATTO** (greeks 5' attivi sul VPS + sync casa, health 14/14) · **C2 FATTO** (refactor 2ter, 3 estrazioni bit-perfette, 298 test, VPS a `6e77a23` col 04b refactorato — tick verificato) · **D4 FATTO** (`mfiv_30h.parquet`, 1.682 snapshot, wedge MFIV−ATM mediana **+3.45 vol pt**).
**EN** Pull+merge OK · B7 done · B1 audit → GPU-GO · B2 started and STOPPED (sandbox deleted, pre-reg intact) · C4 DONE (VPS greeks + home sync) · C2 DONE (2ter refactor, bit-perfect, VPS updated & verified) · D4 DONE (MFIV@30h, median wedge +3.45 vol pts).

## Da fare · To do

1. 🇮🇹 **Fase B GPU — B2 e B3 ESEGUITI 2026-07-19, entrambi "NESSUNA CONCLUSIONE"** (condizione ③ model-independent: r1=657<800 sul val congelato — era verificabile ex-ante, lezione in STATUS). Descrittivo: MoE −2.02% (① non passata), **mixup −4.94% (① sarebbe passata)** → A8-bis prioritaria alla ri-pre-reg. **B4 DVOL: RINVIATO (deciso 2026-07-19)** — esito ③ predeterminato sul val corrente; ri-pre-reg sul dataset esteso (npz dvol da ri-derivare).
   **EN** GPU Phase B — B2+B3 RUN 2026-07-19, both "no conclusion" (condition ③, model-independent). Descriptive: mixup −4.94% (① would have passed) → priority for re-pre-reg. B4 not run, user decision pending.
2. ~~🇮🇹 **Domani dopo le 08:00 UTC (19/07):** `.\avvio_sessione.ps1` → `trades.jsonl` deve avere **22 righe**~~ ✅ **FATTO 2026-07-19**: 22 righe, BTC-19JUL26-64000 settlato (PnL −0.00072 BTC, quasi-pin). n=21 executed.
   **EN** ~~Tomorrow after 08:00 UTC: pull → 22 rows.~~ ✅ DONE 2026-07-19: 22 rows, settled (near-pin, −0.00072 BTC).
3. 🇮🇹 **~Fine luglio:** valutazione pre-registrata **n≥30** leg opzioni (stessi 3 criteri, entrambi i campioni; POST_GATE_V1 §0.2). SOLO dopo: pre-reg sizing v2 (A13+A14+A7).
   **EN** ~End of July: pre-registered n≥30 options-leg evaluation; only afterwards the v2 sizing pre-reg.
4. 🇮🇹 **~Metà agosto:** giudice `hedged_vs_unhedged_judge.py` a n≥20 hedge-attivi (dal primo trade aperto con hedge).
   **EN** ~Mid August: hedged-vs-unhedged judge at n≥20 hedge-active trades.
5. 🇮🇹 **Sblocco candele oltre il 2026-06-22 + refresh macro** (`01b`) — SOLO a B2/B3 chiusi (lo span congelato è il loro invariante).
   **EN** Unfreeze candles past 2026-06-22 — ONLY once B2/B3 are closed.
6. 🇮🇹 **Eventuale pre-reg v2 MFIV-comparatore** (wedge STABILE: +3.39 mediana su 1.911 tick al 2026-07-19, era +3.45 su 1.682 → break-even short-vol da ri-stimare). Derivazione incrementale ESEGUITA 2026-07-19 (+229 snapshot); resta periodica (`python scripts/vol/derive_mfiv.py` — accodabile ad avvio_sessione se utile).
   **EN** Possible MFIV-comparator v2 pre-reg (wedge stable +3.39 on 1,911 ticks); incremental derivation RUN 2026-07-19, stays periodic.
7. 🇮🇹 A lista esaurita: eliminare questo file (esiti in STATUS).
   **EN** When exhausted: delete this file (outcomes in STATUS).
