# RIPRESA — lista residua · remaining list

> 🇮🇹 Riscritto 2026-07-19 sera (v3). File EFFIMERO: eliminarlo a lista esaurita. Pre-registrazioni vincolanti e dettaglio: `STATUS.md` (pre-reg A8-BIS in cima + sezione 2026-07-19).
> **EN** Rewritten 2026-07-19 evening (v3). EPHEMERAL file: delete when exhausted. Binding pre-registrations and detail: `STATUS.md` (A8-BIS pre-reg on top + 2026-07-19 section).

## Fatto il 2026-07-19 · Done on 2026-07-19

🇮🇹 Pull+merge OK, **22ª riga trades verificata** (settlement quasi-pin, PnL −0.00072 BTC, n=21 executed) · **MFIV incrementale** +229 → 1.911 righe, wedge stabile +3.39 vol pt · **B2 A3-MoE e B3 A8-mixup eseguiti e giudicati: entrambi "NESSUNA CONCLUSIONE"** (condizione ③ model-independent, r1<800 sul val congelato; descrittivo: MoE −2.02%, **mixup −4.94%**) · **B4 DVOL RINVIATO** (mai girato) · lezione di processo: condizioni campionarie model-independent si verificano EX-ANTE · **dataset RICOSTRUITO** sullo span esteso (candele→07-19, split 51.882/6.485/6.486, backup `lstm_dataset_frozen0622.npz`) con macro fresche (`01b --skip-regime`, flag nuovo label-preserving) e regime incrementale (+27 barre) · **pre-reg A8-BIS scritta e committata** (`0e4a73b`): 2 bracci riaddestrati, gate ② solo su regimi qualificati (val: r0+r2; r1=613 descrittivo), one-shot test con tutti e 3 (r1=818) · giudice: report anti-clobber per-sandbox + breakdown per-regime · commit `43f2d80`, `2e8b8e8`, `0e4a73b`.
**EN** 22nd trade row verified (n=21) · MFIV incremental (wedge stable +3.39) · B2+B3 run & judged: both "no conclusion" (condition ③, model-independent); B4 deferred · dataset REBUILT on extended span + fresh macro (new `--skip-regime` flag) + incremental regime · **A8-BIS pre-reg written & committed** (2 retrained arms, per-regime gate on qualified regimes only, one-shot test covers stress) · judge: per-sandbox reports + per-regime breakdown.

## Da fare · To do

1. 🇮🇹 **A8-BIS — lanciare i 2 bracci (run MAI partiti; ~70-80 min GPU totali, sequenziali).** Pattern: training → verifica `$LASTEXITCODE`=0 e 5/5 seed (se fallito: elimina la sandbox e rilancia, MAI giudicare un ensemble parziale) → giudice.

   **Braccio 1 — baseline riaddestrata:**
   ```powershell
   $env:QUANTSYS_ARCH="itransformer"; $env:QUANTSYS_MODELS_ROOT="models_base_ext"
   python scripts/02_train.py --n-ensemble 5
   # dopo verifica:
   $env:QUANTSYS_VOLS_SPLIT="val"
   python scripts/vol/dev_vols_qlike.py --arch itransformer
   ```
   **Braccio 2 — candidato mixup:**
   ```powershell
   $env:QUANTSYS_ARCH="itransformer_a8_mixup"; $env:QUANTSYS_MODELS_ROOT="models_a8_mixup_ext"
   python scripts/02_train.py --n-ensemble 5
   # dopo verifica:
   python scripts/vol/dev_vols_qlike.py --arch itransformer_a8_mixup
   Remove-Item Env:QUANTSYS_ARCH, Env:QUANTSYS_MODELS_ROOT, Env:QUANTSYS_VOLS_SPLIT
   ```
   Poi valutazione ①②③ dai report `results/vols/qlike_report_1h_val_models_base_ext.json` e `..._models_a8_mixup_ext.json` (blocco `per_regime`); esiti in STATUS comunque; a PASS val → one-shot su test (stesse condizioni, ② su tutti e 3 i regimi). ⚠ Invariante fino a chiusura gate: NON toccare candele/npz/regime_probs.
   **EN** A8-BIS — launch the 2 arms (never started; ~70-80 min GPU, sequential); judge from the suffixed per-regime reports; outcomes to STATUS regardless; on val PASS → one-shot test. Do NOT touch candles/npz/regime_probs until the gate closes.

2. 🇮🇹 **B4-bis DVOL (rinviato):** ri-derivare `lstm_dataset_dvol.npz` dal npz NUOVO (`dev_vols_dvol_append.py`) + NUOVA pre-reg con ③ verificata ex-ante — solo DOPO la chiusura di A8-BIS (npz congelato è l'invariante).
   **EN** B4-bis DVOL (deferred): re-derive the dvol npz from the NEW npz + new pre-reg with ex-ante ③ — only AFTER A8-BIS closes.

3. 🇮🇹 **Eventuale pre-reg v2 MFIV-comparatore** (wedge stabile +3.39 vol pt su 1.911 tick → break-even short-vol da ri-stimare); derivazione incrementale periodica `python scripts/vol/derive_mfiv.py`.
   **EN** Possible MFIV-comparator v2 pre-reg + periodic incremental derivation.

4. 🇮🇹 **~Fine luglio:** valutazione pre-registrata **n≥30** leg opzioni (POST_GATE_V1 §0.2); solo dopo: pre-reg sizing v2 (A13+A14+A7).
   **EN** ~End of July: pre-registered n≥30 evaluation; only then the v2 sizing pre-reg.

5. 🇮🇹 **~Metà agosto:** giudice `hedged_vs_unhedged_judge.py` a n≥20 hedge-attivi.
   **EN** ~Mid August: hedged-vs-unhedged judge at n≥20 hedge-active trades.

6. 🇮🇹 Igiene a gate A8-BIS chiuso: sandbox `models_a3_moe`/`models_a8_mixup`/`models_base_ext`/`models_a8_mixup_ext` eliminabili; valutare rimozione backup `lstm_dataset_frozen0622.npz` (3 GB) e `lstm_dataset_dvol.npz` stale (3 GB, da ri-derivare comunque).
   **EN** Hygiene once A8-BIS closes: sandboxes deletable; consider dropping the 3 GB frozen npz backup and the stale dvol npz.

7. 🇮🇹 A lista esaurita: eliminare questo file (esiti in STATUS).
   **EN** When exhausted: delete this file (outcomes in STATUS).
