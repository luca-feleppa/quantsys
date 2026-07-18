# RIPRESA — lista residua · remaining list

> 🇮🇹 Riscritto 2026-07-18 sera (v2): la checklist post-riavvio originale è ESAURITA salvo la Fase B GPU (training stoppato su decisione utente) e gli item time-gated. File EFFIMERO: eliminarlo a lista esaurita. Pre-registrazioni vincolanti e dettaglio: `STATUS.md` (sezione 2026-07-18 sera) e `POST_GATE_V1.md`.
> **EN** Rewritten 2026-07-18 evening (v2): the original post-reboot checklist is DONE except the GPU Phase B (training stopped by user decision) and the time-gated items. EPHEMERAL file: delete when exhausted. Binding pre-registrations and detail: `STATUS.md` (2026-07-18 evening section) and `POST_GATE_V1.md`.

## Fatto stasera · Done tonight

🇮🇹 Pull+merge OK (heartbeat 4/4) · B7 regime refresh eseguito · **B1 audit A3 → GPU-GO** (3 MINOR fixati, 22/22 test) · B2 training A3 partito e **STOPPATO** (sandbox `models_a3_moe` eliminata, pre-reg intatta) · **C4 FATTO** (greeks 5' attivi sul VPS + sync casa, health 14/14) · **C2 FATTO** (refactor 2ter, 3 estrazioni bit-perfette, 298 test, VPS a `6e77a23` col 04b refactorato — tick verificato) · **D4 FATTO** (`mfiv_30h.parquet`, 1.682 snapshot, wedge MFIV−ATM mediana **+3.45 vol pt**).
**EN** Pull+merge OK · B7 done · B1 audit → GPU-GO · B2 started and STOPPED (sandbox deleted, pre-reg intact) · C4 DONE (VPS greeks + home sync) · C2 DONE (2ter refactor, bit-perfect, VPS updated & verified) · D4 DONE (MFIV@30h, median wedge +3.45 vol pts).

## Da fare · To do

1. 🇮🇹 **Fase B GPU (quando decidi di riaprire la finestra — ordine vincolante, pre-reg già in STATUS, audit B1 valido).** Ogni run 5 seed → giudice `dev_vols_qlike.py` su val (`QUANTSYS_VOLS_SPLIT=val`) vs incumbent `models/itransformer` (giudice incumbent: stesso script SENZA env root, `--arch itransformer`):
   - **B2** A3 regime-MoE (DA ZERO): `$env:QUANTSYS_ARCH="itransformer_regime_moe"; $env:QUANTSYS_MODELS_ROOT="models_a3_moe"; python scripts/02_train.py --n-ensemble 5` → giudice con env root + `--arch itransformer_regime_moe`. ⚠ Heads-up condizione ③: r1≈657<800 su val → possibile "nessuna conclusione" pre-registrata.
   - **B3** A8 mixup: idem con `itransformer_a8_mixup` / `models_a8_mixup`.
   - **B4** probe DVOL: `QUANTSYS_ARCH=itransformer`, `QUANTSYS_DATASET_NPZ=data/lstm_dataset_dvol.npz`, `QUANTSYS_MODELS_ROOT=models_dvol_probe`.
   Esiti in STATUS comunque; PASS → one-shot su test; ogni interazione = nuova pre-reg. Tempi: ~35-40 min/esperimento (misura seed 1: 7,5 min).
   **EN** GPU Phase B (when you reopen the window — binding order): B2 A3-MoE from scratch, B3 A8 mixup, B4 DVOL probe; 5 seeds each in its sandbox, judged on val vs the incumbent; outcomes to STATUS regardless. ~35-40 min per experiment.
2. 🇮🇹 **Domani dopo le 08:00 UTC (19/07):** `.\avvio_sessione.ps1` → `trades.jsonl` deve avere **22 righe** (settlement BTC-19JUL26-64000 eseguito dal VPS).
   **EN** Tomorrow after 08:00 UTC: pull → trades.jsonl must show 22 rows.
3. 🇮🇹 **~Fine luglio:** valutazione pre-registrata **n≥30** leg opzioni (stessi 3 criteri, entrambi i campioni; POST_GATE_V1 §0.2). SOLO dopo: pre-reg sizing v2 (A13+A14+A7).
   **EN** ~End of July: pre-registered n≥30 options-leg evaluation; only afterwards the v2 sizing pre-reg.
4. 🇮🇹 **~Metà agosto:** giudice `hedged_vs_unhedged_judge.py` a n≥20 hedge-attivi (dal primo trade aperto con hedge).
   **EN** ~Mid August: hedged-vs-unhedged judge at n≥20 hedge-active trades.
5. 🇮🇹 **Sblocco candele oltre il 2026-06-22 + refresh macro** (`01b`) — SOLO a B2/B3 chiusi (lo span congelato è il loro invariante).
   **EN** Unfreeze candles past 2026-06-22 — ONLY once B2/B3 are closed.
6. 🇮🇹 **Eventuale pre-reg v2 MFIV-comparatore** (D4 derivato: wedge +3.45 vol pt → break-even short-vol da ri-stimare) e **derivazione incrementale periodica** (`python scripts/vol/derive_mfiv.py`, appende i soli snapshot nuovi — accodabile ad avvio_sessione se utile).
   **EN** Possible MFIV-comparator v2 pre-reg (re-estimated break-even) + periodic incremental derivation.
7. 🇮🇹 A lista esaurita: eliminare questo file (esiti in STATUS).
   **EN** When exhausted: delete this file (outcomes in STATUS).
