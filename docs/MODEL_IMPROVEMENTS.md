# QUANTSYS — Miglioramenti modello · QUANTSYS — Model improvements

> 🇮🇹 **Verifica 2026-09-10:** file conservato per i design ancora aperti o parcheggiati. Le sezioni implementative sono storiche: stato e condizioni di attivazione nella roadmap riconciliata e in `STATUS.md`, non nelle vecchie date di esecuzione. Nessun esperimento autorizzato da questo documento.
> **EN** **2026-09-10 review:** file retained for open or parked designs. Implementation sections are historical: status and activation conditions are in the reconciled roadmap and `STATUS.md`, not in old execution dates. This document authorizes no experiment.

🇮🇹 **File SNELLITO 2026-07-16 (decisione utente):** contiene SOLO gli item **aperti** (implementati-inerti in attesa di gate, oppure mai avviati e non scartati). Gli item **applicati** sono documentati in `TEORIA.md` / `AVVIO.md` / `README.md` (script research: `scripts/README.md`); gli **esiti** (PASS/FAIL/KILL) e le lezioni consolidate vivono in `STATUS.md` e nella memoria di lungo periodo — non duplicarli qui. Backlog sperimentale della linea vol: `docs/ROADMAP_VOL_BOOK.md`. Coda operativa corrente: `STATUS.md` in testa; residui riconciliati in `docs/ROADMAP_VOL_BOOK.md`.

**EN** **File SLIMMED 2026-07-16 (user decision):** it contains ONLY **open** items (implemented-inert awaiting a gate, or never started and not discarded). **Applied** items are documented in `TEORIA.md` / `AVVIO.md` / `README.md` (research scripts: `scripts/README.md`); **outcomes** (PASS/FAIL/KILL) and consolidated lessons live in `STATUS.md` and in long-term memory — do not duplicate them here. Vol-line experimental backlog: `docs/ROADMAP_VOL_BOOK.md`. Current operating queue: opening sections of `STATUS.md`; reconciled remaining work in `docs/ROADMAP_VOL_BOOK.md`.

---

## A3 — Regime-MoE (mixture-of-universes) · A3 — Regime-MoE (mixture-of-universes)

🇮🇹 **IMPLEMENTATO 2026-07-12 — run eseguito il 19/07, NESSUNA CONCLUSIONE (r1=657<800). ⚠ PARCHEGGIATO dal 2026-07-20** (prior sfavorevole: descrittivo −2.02%, sotto la soglia del 3%; il ramo "la baseline cambia se il mixup passa" è decaduto col FAIL di A8-BIS). **Non è in coda per la prossima finestra GPU**: rivalutare SOLO se un episodio di stress porta massa al regime r1 — la condizione ③ è oggi predeterminata dai conteggi su val, quindi il gate restituirebbe "nessuna conclusione", non un esito. La pre-registrazione storica resta immutata; una rivalutazione richiede una nuova pre-registrazione. Item A3 di `docs/ROADMAP_VOL_BOOK.md` (design: memoria `mixture_of_universes_design`, adattato alla linea vol). Backbone iTransformer condiviso + **3 teste-regime** (R0 Quiet / R1 Trending / R2 Stress) mescolate da un **soft-gate ESTERNO CAUSALE** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — le filtered probabilities di `RegimeMarkovBTC` in `data/regime_probs.parquet`, **mai apprese** (proprietà anti-overfit chiave). Razionale: l'edge short-vol è **Trending-driven** (audit 2026-06-26) → la calibrazione σ regime-condizionata è direttamente monetizzabile.

- **Attivazione config-gated:** chiave `model.head_type` — **assente o `"single"` = path storico bit-identico** (verificato: stesso seed → output `torch.equal`; checkpoint production caricano con `load_state_dict` strict; suite completa verde). `"regime_moe"` attiva le teste. Esempio d'uso: `config/arch/itransformer_regime_moe.yaml` (MAI in `config/default.yaml`).
- **Mixing:** path `quantile` (produzione vol) → media pesata dal gate per livello (**Vincentization**) + re-sort monotono di sicurezza; path `t_student` → **legge della varianza totale** (stessa di `ensemble.py`): `μ_mix = Σ g_k·μ_k`, `σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)²` — σ INFLAZIONATA quando il regime è ambiguo; `lnu` = media pesata dal gate; `ls2` ri-codificato via softplus-inverse (contratto `(mu, ls2, lnu)` invariato).
- **Contratto forward invariato:** `forward(x, x_macro=None, latent=None, g=None)` — `g` opzionale in coda; `g=None` con regime_moe → gate uniforme (1/3,1/3,1/3) con warning una-tantum; burn-in/gap del regime → riga uniforme. `dir_head` (multitask) CONDIVISA tra i regimi (precedente del MoE appreso).
- **Allineamento causale:** `quantsys/model/regime_gate.py → build_regime_gate()` — `merge_asof` **backward** sui timestamp (stesso meccanismo della stratificazione val di `02_train`), mai forward.
- **Scope/esclusioni (fail-fast):** iTransformer-only; mutuamente esclusivo con il MoE appreso (`n_output_experts>1`), con `use_revin` e con `--distill`.
- **Test:** `tests/test_regime_moe.py` (19 test CPU-only, sintetici): inerzia bit-identica, one-hot→testa k, gate uniforme+teste identiche→testa singola, legge varianza totale, monotonia quantili, causalità del builder.
- **Gate QLIKE PRE-REGISTRATO in `STATUS.md` il 2026-07-14** (run nella finestra GPU post-gate-v1, prerequisiti P1 ✅ regime rigenerato 07-15 / P2 04b fermo / P3 audit causality-auditor sui file A3 — da eseguire PRIMA del run); training in sandbox `QUANTSYS_MODELS_ROOT` (mai su `models/itransformer`); giudice `dev_vols_qlike.py` già gate-aware (legge `head_type` dal `config.json` del modello).

**EN** **IMPLEMENTED 2026-07-12 — run executed on 07-19, NO CONCLUSION (r1=657<800). ⚠ PARKED since 2026-07-20** (unfavorable prior: −2.02% descriptive, below the 3% threshold; the "baseline changes if mixup passes" branch lapsed with the A8-BIS FAIL). **Not queued for the next GPU window**: revisit ONLY if a stress episode brings mass to regime r1 — condition ③ is currently predetermined by the val counts, so the gate would return "no conclusion", not a result. The historical pre-registration remains unchanged; reconsideration requires a new preregistration. Item A3 of `docs/ROADMAP_VOL_BOOK.md` (design: `mixture_of_universes_design` memory, adapted to the vol line). Shared iTransformer backbone + **3 regime heads** (R0 Quiet / R1 Trending / R2 Stress) mixed by an **EXTERNAL CAUSAL soft-gate** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — the filtered probabilities of `RegimeMarkovBTC` in `data/regime_probs.parquet`, **never learned** (key anti-overfit property). Rationale: the short-vol edge is **Trending-driven** (2026-06-26 audit) → regime-conditional σ calibration is directly monetizable.

- **Config-gated activation:** key `model.head_type` — **absent or `"single"` = bit-identical legacy path** (verified: same seed → `torch.equal` outputs; production checkpoints load via strict `load_state_dict`; full suite green). `"regime_moe"` enables the heads. Usage example: `config/arch/itransformer_regime_moe.yaml` (NEVER in `config/default.yaml`).
- **Mixing:** `quantile` path (vol production) → per-level gate-weighted average (**Vincentization**) + monotone safety re-sort; `t_student` path → **total variance law** (same as `ensemble.py`): `μ_mix = Σ g_k·μ_k`, `σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)²` — σ INFLATED when the regime is ambiguous; `lnu` = gate-weighted average; `ls2` re-encoded via softplus-inverse (`(mu, ls2, lnu)` contract unchanged).
- **Forward contract unchanged:** `forward(x, x_macro=None, latent=None, g=None)` — optional trailing `g`; `g=None` under regime_moe → uniform gate (1/3,1/3,1/3) with a one-time warning; regime burn-in/gaps → uniform row. `dir_head` (multitask) SHARED across regimes (learned-MoE precedent).
- **Causal alignment:** `quantsys/model/regime_gate.py → build_regime_gate()` — **backward** `merge_asof` on timestamps (same mechanism as `02_train`'s val stratification), never forward.
- **Scope/exclusions (fail-fast):** iTransformer-only; mutually exclusive with the learned MoE (`n_output_experts>1`), with `use_revin` and with `--distill`.
- **Tests:** `tests/test_regime_moe.py` (19 CPU-only synthetic tests): bit-identical inertia, one-hot→head k, uniform gate+identical heads→single head, total variance law, quantile monotonicity, builder causality.
- **QLIKE gate PRE-REGISTERED in `STATUS.md` on 2026-07-14** (run in the post-v1-gate GPU window, prerequisites P1 ✅ regime regenerated 07-15 / P2 04b stopped / P3 causality-auditor audit of the A3 files — to run BEFORE training); train in a `QUANTSYS_MODELS_ROOT` sandbox (never on `models/itransformer`); the `dev_vols_qlike.py` judge is already gate-aware (reads `head_type` from the model's `config.json`).

🇮🇹 **Audit causality-auditor 2026-07-12 (post-implementazione): 1 BLOCKER + 2 MAJOR + 3 MINOR, fixati.** BLOCKER-1: la riga `t` di `regime_probs.parquet` contiene la barra `[t,t+1h)` → il match esatto era lookahead di 1 barra; fix = shift dell'indice ad **availability time (+1h)** prima del merge_asof (+ regression test). MAJOR-1: staleness illimitata a fine parquet → bound `max_age` (default 168h, uniforme oltre) + fail-fast se stale >20%. MAJOR-2: `g=None` in eval ora è `RuntimeError` (input obbligatorio; fallback uniforme solo in train). MINOR-2: `02b_walkforward_validate` fail-fasta su regime_moe (gate non threadato). **MINOR-1 (nota per la pre-registrazione, NON fixato — scelta di design):** la Vincentization del path quantile NON ha il termine between (μ-disagreement) → il meccanismo "σ inflazionata su regime ambiguo" esiste SOLO sul path t_student; sul path production (quantile) il gate QLIKE misura μ, non la calibrazione σ dichiarata come obiettivo A3 — la pre-registrazione lo dichiara.

**EN** **2026-07-12 causality audit (post-implementation): 1 BLOCKER + 2 MAJOR + 3 MINOR, fixed.** BLOCKER-1: row `t` of the parquet holds bar `[t,t+1h)` → the exact match was a 1-bar lookahead; fix = index shift to **availability time (+1h)** before the merge_asof (+ regression test). MAJOR-1: unbounded staleness past the parquet end → `max_age` bound (168h default, uniform beyond) + fail-fast above 20% stale. MAJOR-2: `g=None` in eval is now a `RuntimeError` (mandatory input; uniform fallback in train only). MINOR-2: `02b_walkforward_validate` fail-fasts on regime_moe. **MINOR-1 (pre-registration note, NOT fixed — design choice):** the quantile-path Vincentization has NO between term (μ-disagreement) → the "σ inflated on ambiguous regime" mechanism exists ONLY on the t_student path; on the production (quantile) path the QLIKE gate measures μ, not the σ calibration stated as the A3 goal — the pre-registration declares this.

---

## A7 — Risk layer greeks-aware, residuo · Remaining greeks-aware risk layer

🇮🇹 Il gate delta-hedged è **CHIUSO FAIL 2/3 dall'11/08**, wind-down completato il 13/08: nessuna attivazione o seconda lettura dovuta. Esito completo in `TEORIA.md` §12.2; implementazione hedge in `AVVIO.md`. Conservato A7: `quantsys/trading/greeks_risk.py` contiene cap vega/delta/gamma, circuit breaker vega-loss con isteresi e simulazione conservativa del margine inverse, da validare contro il venue. Skeleton non cablato in `04b`; test `tests/test_greeks_risk.py`. Il FAIL hedged non autorizza il sizing HAR-q90 né l'ingresso di A7 nel critical path: serve una decisione e preregistrazione separata.

**EN** The delta-hedged gate has been **CLOSED FAIL 2/3 since 08-11**, with wind-down completed on 08-13: no activation or second reading is due. Full outcome in `TEORIA.md` §12.2; hedge implementation in `AVVIO.md`. A7 retained: `quantsys/trading/greeks_risk.py` contains vega/delta/gamma caps, a hysteretic vega-loss circuit breaker and conservative inverse-margin simulation, requiring venue validation. Skeleton not wired into `04b`; tests in `tests/test_greeks_risk.py`. The hedged FAIL authorizes neither HAR-q90 sizing nor A7 critical-path integration: a separate decision and preregistration are required.

---
## Execution layer / Binance Futures Testnet (design, NON implementato) · Execution layer (design, NOT implemented)

🇮🇹 > ⚠ **SPECULATIVO — codice inesistente su disco.** Il package `quantsys/execution/` e il modulo `quantsys/execution/reconciliation.py` descritti in versioni precedenti **non esistono** (verificato 2026-06-25). Era il design (Fasi 2-5, 8-13h) per inviare ordini reali sul Futures Testnet parallelamente al portfolio simulato. Prerequisito: BLOCKER #1 risolto (✅). Non avviato. Conservato qui solo come schema progettuale, non come stato del codice.

**EN** > ⚠ **SPECULATIVE — code does not exist on disk.** The `quantsys/execution/` package and `quantsys/execution/reconciliation.py` module described in earlier versions **do not exist** (verified 2026-06-25). It was the design (Phases 2-5, 8-13h) to send real orders to the Futures Testnet in parallel with the simulated portfolio. Prerequisite: BLOCKER #1 resolved (✅). Not started. Kept here only as a design sketch, not as code state.

🇮🇹 **Schema (se mai ripreso):** ABC `ExecutionAdapter` (paper | testnet_futures) con `place_market_order` / `place_stop_market` / `place_take_profit_market` / `cancel_*` / `get_position` / `set_leverage`; leva dinamica conviction-based (`lev = 1 + (max_lev−1)·conviction^alpha`, decisa 2026-05-24); riconciliazione paper-vs-testnet con warning su drift > 0.5%. Fase 1 (`.env` + `scripts/00_test_binance_testnet.py`) era l'unico pezzo done.

**EN** **Sketch (if ever resumed):** ABC `ExecutionAdapter` (paper | testnet_futures) with `place_market_order` / `place_stop_market` / `place_take_profit_market` / `cancel_*` / `get_position` / `set_leverage`; conviction-based dynamic leverage (`lev = 1 + (max_lev−1)·conviction^alpha`, decided 2026-05-24); paper-vs-testnet reconciliation with a warning on >0.5% drift. Phase 1 (`.env` + `scripts/00_test_binance_testnet.py`) was the only done piece.

---

## mamba-ssm CUDA kernel — aperto (target-agnostico) · mamba-ssm CUDA kernel — open (target-agnostic)

🇮🇹 **Unica voce della roadmap legacy ancora potenzialmente utile** (speedup, indipendente dal target). L'implementazione attuale `quantsys/model/tcn_mamba.py` è pure-PyTorch (`SimplifiedMambaBlock._parallel_scan_chunk`). Il pacchetto `mamba-ssm` (Tri Dao) implementa un kernel CUDA fuso (selective scan, prefix-scan Blelloch, ricomputo stato in backward à la Flash Attention): speedup atteso +3-5× sul branch Mamba (sopra il +1.4-1.6× già ottenuto con AMP off + chunk pre-alloc). Prerequisiti mancanti su questa macchina: CUDA Toolkit dev 12.1.x (deve matchare `torch.version.cuda`), MSVC Build Tools 2022, `CUDA_HOME`. Install `--no-build-isolation` (causal-conv1d + mamba-ssm), modifica `MambaBranch` con import condizionale (fallback `SimplifiedMambaBlock`), retrain TCN+Mamba (checkpoint NON compatibili). Rollback: `pip uninstall` → auto-detect `_HAS_MAMBA_SSM = False`. **Quando**: retrain frequenti / `mamba_layers > 3` / sequenze T > 240. Non se il training è "abbastanza veloce".

**EN** **Only legacy-roadmap item still potentially useful** (speedup, target-independent). Current implementation `quantsys/model/tcn_mamba.py` is pure-PyTorch (`SimplifiedMambaBlock._parallel_scan_chunk`). The `mamba-ssm` package (Tri Dao) implements a fused CUDA kernel (selective scan, Blelloch prefix-scan, backward state recompute à la Flash Attention): expected speedup +3-5× on the Mamba branch (on top of the +1.4-1.6× already obtained via AMP off + chunk pre-alloc). Missing prerequisites on this machine: dev CUDA Toolkit 12.1.x (must match `torch.version.cuda`), MSVC Build Tools 2022, `CUDA_HOME`. Install `--no-build-isolation` (causal-conv1d + mamba-ssm), edit `MambaBranch` with a conditional import (fallback `SimplifiedMambaBlock`), retrain TCN+Mamba (checkpoints NOT compatible). Rollback: `pip uninstall` → auto-detect `_HAS_MAMBA_SSM = False`. **When**: frequent retrains / `mamba_layers > 3` / sequences T > 240. Not if training is "fast enough".

---

## Audit residui low-priority · Low-priority audit residue

🇮🇹 4 issue MEDIE + 1 INFRASTRUCTURE dal grand audit 2026-05-23 (8/8 CRITICHE + 8/8 ALTE + 5/9 MEDIE già chiuse). ⚠ I riferimenti `file:linea` marciscono a ogni edit — verificali con grep prima di agire.

**EN** 4 MEDIUM issues + 1 INFRASTRUCTURE from the 2026-05-23 grand audit (8/8 CRITICAL + 8/8 HIGH + 5/9 MEDIUM already closed). ⚠ The `file:line` references rot on every edit — verify with grep before acting.

🇮🇹
| # | File | Issue | Fix proposto · Proposed fix |
|---|---|---|---|
| 21 | `quantsys/trading/__init__.py` | NaN check `x != x` criptico, solo su `size` | NaN guard esplicito all'inizio di `open_position` |
| 23 | `quantsys/data/__init__.py` | Sanity OHLCV `high > close * 10` può scartare flash crash legittimi | rilassare soglia o usare prezzo candela precedente |
| 27 | `quantsys/model/ensemble.py` | `arch_names` non impostato nei fallback `load` | non critico, default OK |
| 28 | `quantsys/features/__init__.py` | `vol_x_pos` crash se colonne assenti su dataset corto | `.get(col, 0)` o try/except |
| #5 ⚠ | `quantsys/trading/__init__.py` + `scripts/03_backtest.py` | `SignalGenerator.set_regime_threshold` esiste ma chiamate DISABILITATE | calibrare o rimuovere dead code |

**EN**
| # | File | Issue | Proposed fix |
|---|---|---|---|
| 21 | `quantsys/trading/__init__.py` | Cryptic NaN check `x != x`, only on `size` | Explicit NaN guard at top of `open_position` |
| 23 | `quantsys/data/__init__.py` | OHLCV sanity `high > close * 10` may discard legitimate flash crashes | Relax threshold or use previous candle price |
| 27 | `quantsys/model/ensemble.py` | `arch_names` not set in `load` fallbacks | Non-critical, default OK |
| 28 | `quantsys/features/__init__.py` | `vol_x_pos` crashes if columns absent on short dataset | `.get(col, 0)` or try/except |
| #5 ⚠ | `quantsys/trading/__init__.py` + `scripts/03_backtest.py` | `SignalGenerator.set_regime_threshold` exists but call sites DISABLED | Calibrate or remove dead code |

🇮🇹 **Contesto #5:** bisect 2026-05-24 ha mostrato che le soglie regime hardcoded (overheating +3pp, stagflation +5pp sul default 0.52) riducevano Sharpe da +18.71 a −4.44 (filtravano 27/42 trade vincenti). Infrastruttura resta ma dead code.

**EN** **#5 context:** 2026-05-24 bisect showed hardcoded regime thresholds (overheating +3pp, stagflation +5pp over the 0.52 default) cut Sharpe from +18.71 to −4.44 (filtered 27/42 winning trades). Infrastructure stays but is dead code.
