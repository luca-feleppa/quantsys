🇬🇧 English · [🇮🇹 Italiano](MODEL_IMPROVEMENTS.it.md)

# QUANTSYS — Model improvements

> **2026-09-10 review:** file retained for open or parked designs. Implementation sections are historical: status and activation conditions are in the reconciled roadmap and `STATUS.md`, not in old execution dates. This document authorizes no experiment.

**File SLIMMED 2026-07-16 (user decision):** it contains ONLY **open** items (implemented-inert awaiting a gate, or never started and not discarded). **Applied** items are documented in `THEORY.md` / `START.md` / `README.md` (research scripts: `scripts/README.md`); **outcomes** (PASS/FAIL/KILL) and consolidated lessons live in `STATUS.md` and in long-term memory — do not duplicate them here. Vol-line experimental backlog: `docs/ROADMAP_VOL_BOOK.md`. Current operating queue: opening sections of `STATUS.md`; reconciled remaining work in `docs/ROADMAP_VOL_BOOK.md`.

---

## A3 — Regime-MoE (mixture-of-universes)

**IMPLEMENTED 2026-07-12 — run executed on 07-19, NO CONCLUSION (r1=657<800). ⚠ PARKED since 2026-07-20** (unfavorable prior: −2.02% descriptive, below the 3% threshold; the "baseline changes if mixup passes" branch lapsed with the A8-BIS FAIL). **Not queued for the next GPU window**: revisit ONLY if a stress episode brings mass to regime r1 — condition ③ is currently predetermined by the val counts, so the gate would return "no conclusion", not a result. The historical pre-registration remains unchanged; reconsideration requires a new preregistration. Item A3 of `docs/ROADMAP_VOL_BOOK.md` (design: `mixture_of_universes_design` memory, adapted to the vol line). Shared iTransformer backbone + **3 regime heads** (R0 Quiet / R1 Trending / R2 Stress) mixed by an **EXTERNAL CAUSAL soft-gate** `g(t) = [regime_prob_0, regime_prob_1, regime_prob_2]` — the filtered probabilities of `RegimeMarkovBTC` in `data/regime_probs.parquet`, **never learned** (key anti-overfit property). Rationale: the short-vol edge is **Trending-driven** (2026-06-26 audit) → regime-conditional σ calibration is directly monetizable.

- **Config-gated activation:** key `model.head_type` — **absent or `"single"` = bit-identical legacy path** (verified: same seed → `torch.equal` outputs; production checkpoints load via strict `load_state_dict`; full suite green). `"regime_moe"` enables the heads. Usage example: `config/arch/itransformer_regime_moe.yaml` (NEVER in `config/default.yaml`).
- **Mixing:** `quantile` path (vol production) → per-level gate-weighted average (**Vincentization**) + monotone safety re-sort; `t_student` path → **total variance law** (same as `ensemble.py`): `μ_mix = Σ g_k·μ_k`, `σ²_mix = Σ g_k·σ²_k + Σ g_k·(μ_k−μ_mix)²` — σ INFLATED when the regime is ambiguous; `lnu` = gate-weighted average; `ls2` re-encoded via softplus-inverse (`(mu, ls2, lnu)` contract unchanged).
- **Forward contract unchanged:** `forward(x, x_macro=None, latent=None, g=None)` — optional trailing `g`; `g=None` under regime_moe → uniform gate (1/3,1/3,1/3) with a one-time warning; regime burn-in/gaps → uniform row. `dir_head` (multitask) SHARED across regimes (learned-MoE precedent).
- **Causal alignment:** `quantsys/model/regime_gate.py → build_regime_gate()` — **backward** `merge_asof` on timestamps (same mechanism as `02_train`'s val stratification), never forward.
- **Scope/exclusions (fail-fast):** iTransformer-only; mutually exclusive with the learned MoE (`n_output_experts>1`), with `use_revin` and with `--distill`.
- **Tests:** `tests/test_regime_moe.py` (19 CPU-only synthetic tests): bit-identical inertia, one-hot→head k, uniform gate+identical heads→single head, total variance law, quantile monotonicity, builder causality.
- **QLIKE gate PRE-REGISTERED in `STATUS.md` on 2026-07-14** (run in the post-v1-gate GPU window, prerequisites P1 ✅ regime regenerated 07-15 / P2 04b stopped / P3 causality-auditor audit of the A3 files — to run BEFORE training); train in a `QUANTSYS_MODELS_ROOT` sandbox (never on `models/itransformer`); the `dev_vols_qlike.py` judge is already gate-aware (reads `head_type` from the model's `config.json`).

**2026-07-12 causality-auditor audit (post-implementation): 1 BLOCKER + 2 MAJOR + 3 MINOR, fixed.** BLOCKER-1: row `t` of `regime_probs.parquet` holds bar `[t,t+1h)` → the exact match was a 1-bar lookahead; fix = index shift to **availability time (+1h)** before the merge_asof (+ regression test). MAJOR-1: unbounded staleness past the parquet end → `max_age` bound (168h default, uniform beyond) + fail-fast above 20% stale. MAJOR-2: `g=None` in eval is now a `RuntimeError` (mandatory input; uniform fallback in train only). MINOR-2: `02b_walkforward_validate` fail-fasts on regime_moe (gate not threaded). **MINOR-1 (pre-registration note, NOT fixed — design choice):** the quantile-path Vincentization has NO between term (μ-disagreement) → the "σ inflated on ambiguous regime" mechanism exists ONLY on the t_student path; on the production (quantile) path the QLIKE gate measures μ, not the σ calibration stated as the A3 goal — the pre-registration declares this.

---

## A7 — Remaining greeks-aware risk layer

The delta-hedged gate has been **CLOSED FAIL 2/3 since 08-11**, with wind-down completed on 08-13: no activation or second reading is due. Full outcome in `THEORY.md` §12.2; hedge implementation in `START.md`. A7 retained: `quantsys/trading/greeks_risk.py` contains vega/delta/gamma caps, a hysteretic vega-loss circuit breaker and conservative inverse-margin simulation, requiring venue validation. Skeleton not wired into `04b`; tests in `tests/test_greeks_risk.py`. The hedged FAIL authorizes neither HAR-q90 sizing nor A7 critical-path integration: a separate decision and preregistration are required.

---
## Execution layer / Binance Futures Testnet (design, NOT implemented)

> ⚠ **SPECULATIVE — code does not exist on disk.** The `quantsys/execution/` package and `quantsys/execution/reconciliation.py` module described in earlier versions **do not exist** (verified 2026-06-25). It was the design (Phases 2-5, 8-13h) to send real orders to the Futures Testnet in parallel with the simulated portfolio. Prerequisite: BLOCKER #1 resolved (✅). Not started. Kept here only as a design sketch, not as code state.

**Sketch (if ever resumed):** ABC `ExecutionAdapter` (paper | testnet_futures) with `place_market_order` / `place_stop_market` / `place_take_profit_market` / `cancel_*` / `get_position` / `set_leverage`; conviction-based dynamic leverage (`lev = 1 + (max_lev−1)·conviction^alpha`, decided 2026-05-24); paper-vs-testnet reconciliation with a warning on >0.5% drift. Phase 1 (`.env` + `scripts/00_test_binance_testnet.py`) was the only done piece.

---

## mamba-ssm CUDA kernel — open (target-agnostic)

**Only legacy-roadmap item still potentially useful** (speedup, target-independent). Current implementation `quantsys/model/tcn_mamba.py` is pure-PyTorch (`SimplifiedMambaBlock._parallel_scan_chunk`). The `mamba-ssm` package (Tri Dao) implements a fused CUDA kernel (selective scan, Blelloch prefix-scan, backward state recompute à la Flash Attention): expected speedup +3-5× on the Mamba branch (on top of the +1.4-1.6× already obtained via AMP off + chunk pre-alloc). Missing prerequisites on this machine: dev CUDA Toolkit 12.1.x (must match `torch.version.cuda`), MSVC Build Tools 2022, `CUDA_HOME`. Install `--no-build-isolation` (causal-conv1d + mamba-ssm), edit `MambaBranch` with a conditional import (fallback `SimplifiedMambaBlock`), retrain TCN+Mamba (checkpoints NOT compatible). Rollback: `pip uninstall` → auto-detect `_HAS_MAMBA_SSM = False`. **When**: frequent retrains / `mamba_layers > 3` / sequences T > 240. Not if training is "fast enough".

---

## Low-priority audit residue

4 MEDIUM issues + 1 INFRASTRUCTURE from the 2026-05-23 grand audit (8/8 CRITICAL + 8/8 HIGH + 5/9 MEDIUM already closed). ⚠ The `file:line` references rot on every edit — verify with grep before acting.

| # | File | Issue | Proposed fix |
|---|---|---|---|
| 21 | `quantsys/trading/__init__.py` | Cryptic NaN check `x != x`, only on `size` | Explicit NaN guard at top of `open_position` |
| 23 | `quantsys/data/__init__.py` | OHLCV sanity `high > close * 10` may discard legitimate flash crashes | Relax threshold or use previous candle price |
| 27 | `quantsys/model/ensemble.py` | `arch_names` not set in `load` fallbacks | Non-critical, default OK |
| 28 | `quantsys/features/__init__.py` | `vol_x_pos` crashes if columns absent on short dataset | `.get(col, 0)` or try/except |
| #5 ⚠ | `quantsys/trading/__init__.py` + `scripts/03_backtest.py` | `SignalGenerator.set_regime_threshold` exists but call sites DISABLED | Calibrate or remove dead code |

**#5 context:** 2026-05-24 bisect showed hardcoded regime thresholds (overheating +3pp, stagflation +5pp over the 0.52 default) cut Sharpe from +18.71 to −4.44 (filtered 27/42 winning trades). Infrastructure stays but is dead code.
