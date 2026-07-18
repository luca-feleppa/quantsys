# RIPRESA — checklist post-riavvio · post-reboot checklist

> 🇮🇹 Scritto 2026-07-18 sera, prima del riavvio del PC. File EFFIMERO: eliminarlo a lista esaurita. Le pre-registrazioni vincolanti e il dettaglio restano in `STATUS.md` (sezione 2026-07-18) e `POST_GATE_V1.md`.
> **EN** Written 2026-07-18 evening, before the PC reboot. EPHEMERAL file: delete when the list is done. Binding pre-registrations and detail live in `STATUS.md` (2026-07-18 section) and `POST_GATE_V1.md`.

## Stato al riavvio · State at reboot

🇮🇹 **Tutto il sistema live è sul VPS** (health PASS 13/13): `quantsys-iv/ob/trades` (collector) + `quantsys-volpaper` = `04b --execute --hedge --hedge-band 0.30 --hedge-conv raw` (**forward v2 hedged ATTIVO** dal 2026-07-18 ~13:13 UTC, parametri congelati A4, restart automatico 00:30 UTC). Il PC è **passivo**: zero processi residenti (01c auto-start rimosso; emergenza = comando nel commento di `avvio_sessione.ps1`). Posizione aperta: BTC-19JUL26-64000 long straddle (parzialmente hedgiata → ESCLUSA dal campione v2, pre-dichiarato). Gate v1: CHIUSO FAIL 0/3; valutazione n≥30 dovuta ~fine luglio.
**EN** The whole live system is on the VPS (health PASS 13/13): collectors + `quantsys-volpaper` running the hedged v2 forward (frozen A4 params, daily 00:30 UTC restart). The PC is passive: no resident processes. Open position: BTC-19JUL26-64000 long straddle (partially hedged → EXCLUDED from the v2 sample, pre-declared). Gate v1: CLOSED FAIL 0/3; the n≥30 evaluation is owed ~end of July.

## Da fare, in ordine · To do, in order

1. 🇮🇹 **`.\avvio_sessione.ps1`** — solo pull+merge (porta a casa forecasts/trades/hedge-ledger dal VPS; lo staging si auto-pulisce a heartbeat sano) + check B7: al primo avvio scatterà il **refresh incrementale regime** (621 barre nuove > 168; background, minuti, innocuo by design — non altera righe storiche).
   **EN** `avvio_sessione.ps1` — pull+merge only (staging auto-cleans on healthy heartbeat) + B7 check: the incremental regime refresh will fire once (background, minutes, harmless by design).
2. 🇮🇹 **Verifica post-pull:** heartbeat 4 sorgenti fresche (IV / L2 / trades / **04b forecasts**); dopo le 08:00 UTC del 19/07 `trades.jsonl` deve avere **22 righe** (settlement BTC-19JUL26-64000, eseguito dal VPS).
   **EN** Post-pull check: 4 fresh heartbeat sources; after 08:00 UTC on 07/19 `trades.jsonl` must show **22 rows** (VPS-executed settlement).
3. 🇮🇹 **Finestra GPU (ora LIBERA: 04b non contende più CUDA)** — Fase B di `POST_GATE_V1.md` in ordine vincolante, pre-reg già scritte in `STATUS.md`:
   - **B1** audit `causality-auditor` sui file A3-MoE (mai eseguito — PRIMA di spendere GPU);
   - **B2** run A3 regime-MoE (`QUANTSYS_ARCH=itransformer_regime_moe`, sandbox `QUANTSYS_MODELS_ROOT=models_a3_moe`, 5 seed) → giudice `dev_vols_qlike.py` su val;
   - **B3** run A8 mixup (`QUANTSYS_ARCH=itransformer_a8_mixup`, sandbox `models_a8_mixup`, 5 seed) → stesso giudice;
   - **B4** probe DVOL (`QUANTSYS_ARCH=itransformer`, `QUANTSYS_DATASET_NPZ=data/lstm_dataset_dvol.npz`, `QUANTSYS_MODELS_ROOT=models_dvol_probe`, 5 seed) → stesso giudice.
   Esiti in STATUS **comunque**; PASS → one-shot su test; run indipendenti vs lo stesso incumbent; ogni interazione = nuova pre-reg.
   **EN** GPU window (now FREE) — Phase B of `POST_GATE_V1.md` in binding order: B1 causality audit of the A3-MoE files, then B2 (A3 regime-MoE), B3 (A8 mixup), B4 (DVOL probe), each 5 seeds in its own sandbox, judged by `dev_vols_qlike.py` on val. Outcomes written to STATUS regardless.
4. 🇮🇹 **In coda (dopo la Fase B):** valutazione pre-registrata **n≥30** leg opzioni (~fine luglio, stessi 3 criteri/2 campioni); giudice **hedged-vs-unhedged** a n≥20 hedge-attivi (~metà agosto); **C2** refactor 2ter; **C4** `--greeks` sul poller VPS + sync casa di `atm_greeks.parquet`; **D4** derivazione offline MFIV@30h (risposta strutturale al FAIL del gate: comparatore corretto per convessità).
   **EN** Queue (after Phase B): pre-registered n≥30 options-leg evaluation (~end of July); hedged-vs-unhedged judge at n≥20 hedge-active (~mid August); C2 refactor; C4 VPS greeks + home sync; D4 offline MFIV@30h derivation.
5. 🇮🇹 A lista esaurita: eliminare questo file (esiti in STATUS).
   **EN** When the list is done: delete this file (outcomes in STATUS).
