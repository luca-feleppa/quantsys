🇮🇹 Italiano · [🇬🇧 English](MODEL_IMPROVEMENTS.md)

# QUANTSYS — Miglioramenti modello

> **Verifica 2026-09-10:** file conservato per i design ancora aperti o parcheggiati. Le sezioni implementative sono storiche: stato e condizioni di attivazione nella roadmap riconciliata e in `STATUS.md`, non nelle vecchie date di esecuzione. Nessun esperimento autorizzato da questo documento.

**File SNELLITO 2026-07-16 (decisione utente):** contiene SOLO gli item **aperti** (implementati-inerti in attesa di gate, oppure mai avviati e non scartati). Gli item **applicati** sono documentati in `THEORY.it.md` / `START.it.md` / `README.it.md` (script research: `scripts/README.it.md`); gli **esiti** (PASS/FAIL/KILL) e le lezioni consolidate vivono in `STATUS.md` e nella memoria di lungo periodo — non duplicarli qui. Backlog sperimentale della linea vol: `docs/ROADMAP_VOL_BOOK.it.md`. Coda operativa corrente: `STATUS.md` in testa; residui riconciliati in `docs/ROADMAP_VOL_BOOK.it.md`.

---

## A3 — Regime-MoE (mixture-of-universes)

**IMPLEMENTATO 2026-07-12 — run eseguito il 19/07, NESSUNA CONCLUSIONE (r1=657<800). ⚠ PARCHEGGIATO dal 2026-07-20** (prior sfavorevole: descrittivo −2.02%, sotto la soglia del 3%; il ramo "la baseline cambia se il mixup passa" è decaduto col FAIL di A8-BIS). **Non è in coda per la prossima finestra GPU**: rivalutare SOLO se un episodio di stress porta massa al regime r1 — la condizione ③ è oggi predeterminata dai conteggi su val, quindi il gate restituirebbe "nessuna conclusione", non un esito. La pre-registrazione storica resta immutata; una rivalutazione richiede una nuova pre-registrazione. Item A3 di `docs/ROADMAP_VOL_BOOK.it.md` (design: memoria `mixture_of_universes_design`, adattato alla linea vol). Backbone iTransformer condiviso + **3 teste-regime** (R0 Quiet / R1 Trending / R2 Stress) mescolate da un **soft-gate ESTERNO CAUSALE** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — le filtered probabilities di `RegimeMarkovBTC` in `data/regime_probs.parquet`, **mai apprese** (proprietà anti-overfit chiave). Razionale: l'edge short-vol è **Trending-driven** (audit 2026-06-26) → la calibrazione σ regime-condizionata è direttamente monetizzabile.

- **Attivazione config-gated:** chiave `model.head_type` — **assente o `"single"` = path storico bit-identico** (verificato: stesso seed → output `torch.equal`; checkpoint production caricano con `load_state_dict` strict; suite completa verde). `"regime_moe"` attiva le teste. Esempio d'uso: `config/arch/itransformer_regime_moe.yaml` (MAI in `config/default.yaml`).
- **Mixing:** path `quantile` (produzione vol) → media pesata dal gate per livello (**Vincentization**) + re-sort monotono di sicurezza; path `t_student` → **legge della varianza totale** (stessa di `ensemble.py`): `μ_mix = Σ g_k·μ_k`, `σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)²` — σ INFLAZIONATA quando il regime è ambiguo; `lnu` = media pesata dal gate; `ls2` ri-codificato via softplus-inverse (contratto `(mu, ls2, lnu)` invariato).
- **Contratto forward invariato:** `forward(x, x_macro=None, latent=None, g=None)` — `g` opzionale in coda; `g=None` con regime_moe → gate uniforme (1/3,1/3,1/3) con warning una-tantum; burn-in/gap del regime → riga uniforme. `dir_head` (multitask) CONDIVISA tra i regimi (precedente del MoE appreso).
- **Allineamento causale:** `quantsys/model/regime_gate.py → build_regime_gate()` — `merge_asof` **backward** sui timestamp (stesso meccanismo della stratificazione val di `02_train`), mai forward.
- **Scope/esclusioni (fail-fast):** iTransformer-only; mutuamente esclusivo con il MoE appreso (`n_output_experts>1`), con `use_revin` e con `--distill`.
- **Test:** `tests/test_regime_moe.py` (19 test CPU-only, sintetici): inerzia bit-identica, one-hot→testa k, gate uniforme+teste identiche→testa singola, legge varianza totale, monotonia quantili, causalità del builder.
- **Gate QLIKE PRE-REGISTRATO in `STATUS.md` il 2026-07-14** (run nella finestra GPU post-gate-v1, prerequisiti P1 ✅ regime rigenerato 07-15 / P2 04b fermo / P3 audit causality-auditor sui file A3 — da eseguire PRIMA del run); training in sandbox `QUANTSYS_MODELS_ROOT` (mai su `models/itransformer`); giudice `dev_vols_qlike.py` già gate-aware (legge `head_type` dal `config.json` del modello).

**Audit causality-auditor 2026-07-12 (post-implementazione): 1 BLOCKER + 2 MAJOR + 3 MINOR, fixati.** BLOCKER-1: la riga `t` di `regime_probs.parquet` contiene la barra `[t,t+1h)` → il match esatto era lookahead di 1 barra; fix = shift dell'indice ad **availability time (+1h)** prima del merge_asof (+ regression test). MAJOR-1: staleness illimitata a fine parquet → bound `max_age` (default 168h, uniforme oltre) + fail-fast se stale >20%. MAJOR-2: `g=None` in eval ora è `RuntimeError` (input obbligatorio; fallback uniforme solo in train). MINOR-2: `02b_walkforward_validate` fail-fasta su regime_moe (gate non threadato). **MINOR-1 (nota per la pre-registrazione, NON fixato — scelta di design):** la Vincentization del path quantile NON ha il termine between (μ-disagreement) → il meccanismo "σ inflazionata su regime ambiguo" esiste SOLO sul path t_student; sul path production (quantile) il gate QLIKE misura μ, non la calibrazione σ dichiarata come obiettivo A3 — la pre-registrazione lo dichiara.

---

## A7 — Risk layer greeks-aware, residuo

Il gate delta-hedged è **CHIUSO FAIL 2/3 dall'11/08**, wind-down completato il 13/08: nessuna attivazione o seconda lettura dovuta. Esito completo in `THEORY.it.md` §12.2; implementazione hedge in `START.it.md`. Conservato A7: `quantsys/trading/greeks_risk.py` contiene cap vega/delta/gamma, circuit breaker vega-loss con isteresi e simulazione conservativa del margine inverse, da validare contro il venue. Skeleton non cablato in `04b`; test `tests/test_greeks_risk.py`. Il FAIL hedged non autorizza il sizing HAR-q90 né l'ingresso di A7 nel critical path: serve una decisione e preregistrazione separata.

---
## Execution layer / Binance Futures Testnet (design, NON implementato)

> ⚠ **SPECULATIVO — codice inesistente su disco.** Il package `quantsys/execution/` e il modulo `quantsys/execution/reconciliation.py` descritti in versioni precedenti **non esistono** (verificato 2026-06-25). Era il design (Fasi 2-5, 8-13h) per inviare ordini reali sul Futures Testnet parallelamente al portfolio simulato. Prerequisito: BLOCKER #1 risolto (✅). Non avviato. Conservato qui solo come schema progettuale, non come stato del codice.

**Schema (se mai ripreso):** ABC `ExecutionAdapter` (paper | testnet_futures) con `place_market_order` / `place_stop_market` / `place_take_profit_market` / `cancel_*` / `get_position` / `set_leverage`; leva dinamica conviction-based (`lev = 1 + (max_lev−1)·conviction^alpha`, decisa 2026-05-24); riconciliazione paper-vs-testnet con warning su drift > 0.5%. Fase 1 (`.env` + `scripts/00_test_binance_testnet.py`) era l'unico pezzo done.

---

## mamba-ssm CUDA kernel — aperto (target-agnostico)

**Unica voce della roadmap legacy ancora potenzialmente utile** (speedup, indipendente dal target). L'implementazione attuale `quantsys/model/tcn_mamba.py` è pure-PyTorch (`SimplifiedMambaBlock._parallel_scan_chunk`). Il pacchetto `mamba-ssm` (Tri Dao) implementa un kernel CUDA fuso (selective scan, prefix-scan Blelloch, ricomputo stato in backward à la Flash Attention): speedup atteso +3-5× sul branch Mamba (sopra il +1.4-1.6× già ottenuto con AMP off + chunk pre-alloc). Prerequisiti mancanti su questa macchina: CUDA Toolkit dev 12.1.x (deve matchare `torch.version.cuda`), MSVC Build Tools 2022, `CUDA_HOME`. Install `--no-build-isolation` (causal-conv1d + mamba-ssm), modifica `MambaBranch` con import condizionale (fallback `SimplifiedMambaBlock`), retrain TCN+Mamba (checkpoint NON compatibili). Rollback: `pip uninstall` → auto-detect `_HAS_MAMBA_SSM = False`. **Quando**: retrain frequenti / `mamba_layers > 3` / sequenze T > 240. Non se il training è "abbastanza veloce".

---

## Audit residui low-priority

4 issue MEDIE + 1 INFRASTRUCTURE dal grand audit 2026-05-23 (8/8 CRITICHE + 8/8 ALTE + 5/9 MEDIE già chiuse). ⚠ I riferimenti `file:linea` marciscono a ogni edit — verificali con grep prima di agire.

| # | File | Issue | Fix proposto |
|---|---|---|---|
| 21 | `quantsys/trading/__init__.py` | NaN check `x != x` criptico, solo su `size` | NaN guard esplicito all'inizio di `open_position` |
| 23 | `quantsys/data/__init__.py` | Sanity OHLCV `high > close * 10` può scartare flash crash legittimi | rilassare soglia o usare prezzo candela precedente |
| 27 | `quantsys/model/ensemble.py` | `arch_names` non impostato nei fallback `load` | non critico, default OK |
| 28 | `quantsys/features/__init__.py` | `vol_x_pos` crash se colonne assenti su dataset corto | `.get(col, 0)` o try/except |
| #5 ⚠ | `quantsys/trading/__init__.py` + `scripts/03_backtest.py` | `SignalGenerator.set_regime_threshold` esiste ma chiamate DISABILITATE | calibrare o rimuovere dead code |

**Contesto #5:** bisect 2026-05-24 ha mostrato che le soglie regime hardcoded (overheating +3pp, stagflation +5pp sul default 0.52) riducevano Sharpe da +18.71 a −4.44 (filtravano 27/42 trade vincenti). Infrastruttura resta ma dead code.
