# QUANTSYS — Miglioramenti modello · QUANTSYS — Model improvements

🇮🇹 **File SNELLITO 2026-07-16 (decisione utente):** contiene SOLO gli item **aperti** (implementati-inerti in attesa di gate, oppure mai avviati e non scartati). Gli item **applicati** sono documentati in `TEORIA.md` / `AVVIO.md` / `README.md` / `CLAUDE.md` (script research: `scripts/README.md`); gli **esiti** (PASS/FAIL/KILL) e le lezioni consolidate vivono in `STATUS.md` e nella memoria di lungo periodo — non duplicarli qui. Backlog sperimentale della linea vol: `docs/ROADMAP_VOL_BOOK.md`. Piano operativo post-gate-v1: `POST_GATE_V1.md` (root).

**EN** **File SLIMMED 2026-07-16 (user decision):** it contains ONLY **open** items (implemented-inert awaiting a gate, or never started and not discarded). **Applied** items are documented in `TEORIA.md` / `AVVIO.md` / `README.md` / `CLAUDE.md` (research scripts: `scripts/README.md`); **outcomes** (PASS/FAIL/KILL) and consolidated lessons live in `STATUS.md` and in long-term memory — do not duplicate them here. Vol-line experimental backlog: `docs/ROADMAP_VOL_BOOK.md`. Post-gate-v1 operating plan: `POST_GATE_V1.md` (root).

---

## A3 — Regime-MoE (mixture-of-universes) · A3 — Regime-MoE (mixture-of-universes)

🇮🇹 **IMPLEMENTATO-INERTE 2026-07-12 — MAI addestrato, zero risultati.** Item A3 di `docs/ROADMAP_VOL_BOOK.md` (design: memoria `mixture_of_universes_design`, adattato alla linea vol). Backbone iTransformer condiviso + **3 teste-regime** (R0 Quiet / R1 Trending / R2 Stress) mescolate da un **soft-gate ESTERNO CAUSALE** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — le filtered probabilities di `RegimeMarkovBTC` in `data/regime_probs.parquet`, **mai apprese** (proprietà anti-overfit chiave). Razionale: l'edge short-vol è **Trending-driven** (audit 2026-06-26) → la calibrazione σ regime-condizionata è direttamente monetizzabile.

- **Attivazione config-gated:** chiave `model.head_type` — **assente o `"single"` = path storico bit-identico** (verificato: stesso seed → output `torch.equal`; checkpoint production caricano con `load_state_dict` strict; suite completa verde). `"regime_moe"` attiva le teste. Esempio d'uso: `config/arch/itransformer_regime_moe.yaml` (MAI in `config/default.yaml`).
- **Mixing:** path `quantile` (produzione vol) → media pesata dal gate per livello (**Vincentization**) + re-sort monotono di sicurezza; path `t_student` → **legge della varianza totale** (stessa di `ensemble.py`): `μ_mix = Σ g_k·μ_k`, `σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)²` — σ INFLAZIONATA quando il regime è ambiguo; `lnu` = media pesata dal gate; `ls2` ri-codificato via softplus-inverse (contratto `(mu, ls2, lnu)` invariato).
- **Contratto forward invariato:** `forward(x, x_macro=None, latent=None, g=None)` — `g` opzionale in coda; `g=None` con regime_moe → gate uniforme (1/3,1/3,1/3) con warning una-tantum; burn-in/gap del regime → riga uniforme. `dir_head` (multitask) CONDIVISA tra i regimi (precedente del MoE appreso).
- **Allineamento causale:** `quantsys/model/regime_gate.py → build_regime_gate()` — `merge_asof` **backward** sui timestamp (stesso meccanismo della stratificazione val di `02_train`), mai forward.
- **Scope/esclusioni (fail-fast):** iTransformer-only; mutuamente esclusivo con il MoE appreso (`n_output_experts>1`), con `use_revin` e con `--distill`.
- **Test:** `tests/test_regime_moe.py` (19 test CPU-only, sintetici): inerzia bit-identica, one-hot→testa k, gate uniforme+teste identiche→testa singola, legge varianza totale, monotonia quantili, causalità del builder.
- **Gate QLIKE PRE-REGISTRATO in `STATUS.md` il 2026-07-14** (run nella finestra GPU post-gate-v1, prerequisiti P1 ✅ regime rigenerato 07-15 / P2 04b fermo / P3 audit causality-auditor sui file A3 — da eseguire PRIMA del run); training in sandbox `QUANTSYS_MODELS_ROOT` (mai su `models/itransformer`); giudice `dev_vols_qlike.py` già gate-aware (legge `head_type` dal `config.json` del modello).

**EN** **IMPLEMENTED-INERT 2026-07-12 — NEVER trained, zero results.** Item A3 of `docs/ROADMAP_VOL_BOOK.md` (design: `mixture_of_universes_design` memory, adapted to the vol line). Shared iTransformer backbone + **3 regime heads** (R0 Quiet / R1 Trending / R2 Stress) mixed by an **EXTERNAL CAUSAL soft-gate** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — the filtered probabilities of `RegimeMarkovBTC` in `data/regime_probs.parquet`, **never learned** (key anti-overfit property). Rationale: the short-vol edge is **Trending-driven** (2026-06-26 audit) → regime-conditional σ calibration is directly monetizable.

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

## V2 delta-hedged (`04b --hedge`) + risk layer greeks-aware (A7) · V2 delta-hedged + greeks-aware risk layer (A7)

🇮🇹 **IMPLEMENTATI 2026-07-12 — entrambi INERTI/NON cablati, mai attivati live.** Sequencing B3 step 2 della `ROADMAP_VOL_BOOK`, preparato in anticipo perché il gate v1 n≥20 chiude a giorni.

- **Leg delta-hedge perp (`scripts/04b_vol_paper.py --hedge`, default OFF = v1 bit-identica):** ribilanciamento SOLO oltre la no-trade band `|Δ_book|` (dry-run 2026-07-10: churn ATM = drag puro); hedge ratio = δ teorico venue, convenzione parametrica `raw`/`adj` (`adj = Σδ−Σmark`, BTC-terms, coerente con slope −0.98 mainnet); nozionale `H* = −side·δ_conv·S` sul perp inverse; flatten automatico a settlement E a expiry (audit MAJOR-1: mai delta nudo post-expiry); stato atomico + riconciliazione con la posizione venue all'avvio (`--execute`); fail-fast se `--hedge` parte senza band/conv espliciti (= congelati dalla pre-registrazione, MINOR-3); bound di plausibilità sul δ del ticker (MINOR-2). Output: `hedge_state.json` + `hedge_ledger.jsonl` (PnL inverse esatto ricostruibile offline). Gate hedged-vs-unhedged pre-registrato in DRAFT (STATUS.md 2026-07-12): attivazione SOLO post-gate v1. Test: `tests/test_hedge_leg.py` (11, FakeDB offline). Banda WW gamma-scalata opzionale (A12, `--hedge-band-mode ww`, default `fixed`).
- **Risk layer greeks-aware (`quantsys/trading/greeks_risk.py`, A7 skeleton):** cap vega/delta netti pre-trade (scaling monotono al bordo del cap, riduzioni sempre ammesse), circuit breaker vega-loss MtM con isteresi, margin sim Deribit inverse (IM/MM, conservativa, da validare vs `get_account_summary`). NON cablato in 04b: serve al sizing Kelly-su-edge della v2. Test: `tests/test_greeks_risk.py` (17).
- Audit `causality-auditor` stesso giorno: 0 blocker; 1 MAJOR + 4 MINOR trovati e **tutti applicati** (expiry-flatten, write atomica+riconciliazione, δ-bound, fail-fast attivazione, cap sign-flip).

**EN** **IMPLEMENTED 2026-07-12 — both INERT/not wired, never activated live.** B3 sequencing step 2 of `ROADMAP_VOL_BOOK`, prepared ahead because the v1 n≥20 gate closes within days.

- **Perp delta-hedge leg (`scripts/04b_vol_paper.py --hedge`, default OFF = bit-identical v1):** rebalance ONLY beyond the `|book_delta|` no-trade band (2026-07-10 dry-run: ATM churn = pure drag); hedge ratio = venue theoretical delta, parametric `raw`/`adj` convention (`adj = Σδ−Σmark`, BTC-terms, consistent with the −0.98 mainnet slope); inverse-perp notional `H* = −side·δ_conv·S`; automatic flatten at settlement AND at expiry (MAJOR-1 audit: never naked delta post-expiry); atomic state + venue-position reconciliation at `--execute` startup; fail-fast if `--hedge` starts without explicit band/conv (= frozen by the pre-registration, MINOR-3); plausibility bound on ticker deltas (MINOR-2). Output: `hedge_state.json` + `hedge_ledger.jsonl` (exact inverse PnL reconstructable offline). Hedged-vs-unhedged gate pre-registered as DRAFT (STATUS.md 2026-07-12): activation ONLY post-v1-gate. Tests: `tests/test_hedge_leg.py` (11, offline FakeDB). Optional gamma-scaled WW band (A12, `--hedge-band-mode ww`, default `fixed`).
- **Greeks-aware risk layer (`quantsys/trading/greeks_risk.py`, A7 skeleton):** pre-trade net vega/delta caps (monotone scaling to the cap edge, reductions always allowed), MtM vega-loss circuit breaker with hysteresis, Deribit inverse margin sim (IM/MM, conservative, to validate vs `get_account_summary`). NOT wired into 04b: it serves the v2 Kelly-on-edge sizing. Tests: `tests/test_greeks_risk.py` (17).
- Same-day `causality-auditor` audit: 0 blockers; 1 MAJOR + 4 MINOR found and **all applied** (expiry-flatten, atomic write+reconciliation, δ-bound, activation fail-fast, sign-flip cap).

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
