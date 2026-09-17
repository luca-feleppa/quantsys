🇬🇧 English · [🇮🇹 Italiano](ROADMAP_VOL_BOOK.it.md)

# ROADMAP — Vol Book: Status and remaining work

**Reconciled on 2026-09-10.** Current queue and binding preregistrations: [STATUS.md](../STATUS.md), opening sections. This file retains useful IDs and remaining work; it authorizes no training, live changes or new gate readings. Full outcomes remain in [THEORY.md](../THEORY.md) §12 and the status history.

## Current priorities

E1 stage 2 is **CLOSED, NO CONCLUSION** (09-10): no second readings or signal inversion without a new preregistration. FT1 is implemented-inert and **not started**: power audit done on 09-15, with amendment 1 adopted (① downgraded to a pre-computed control, ②a's claim with its bound, executor-outcome counting, reading time, measurable ②b/③b) — go-live stays blocked until the FT1 judge exists in code with sentinel tests; decide the macro vintage before the first fill; go live only on explicit instruction and with a flat ledger after settlement. The routine's E1 counter should be retired because it has no remaining decision consumer. None of these tasks was executed during the documentation cleanup.

## Retained work

| ID | Status and constraint | Source |
|---|---|---|
| A3 / A3-bis | Regime-MoE **run on 07-19**, no conclusion: r1=657<800. Parked; A8-BIS failed, so that reopening branch lapsed. New sample and preregistration required before reconsideration. | [Implementation](MODEL_IMPROVEMENTS.md), STATUS 19–20/07 |
| A7 | Greeks-risk skeleton not wired. The hedged FAIL authorizes neither critical-path integration nor HAR-q90 sizing; a separate design decision is needed. | MODEL_IMPROVEMENTS, THEORY §12.2 |
| A9 | Parallel N-HiTS MaxPool implemented-inert; no documented PASS. Not an automatic reopening of the closed training class: a new hypothesis and preregistration are required. | `config/arch/nhits.yaml`, `tests/test_nhits_maxpool.py`, STATUS |
| A13 / A13a | Pin-close and gamma cap inert. Pin-close parked: useful unit = triggers, historical design n_trig≥20; choose paired offline or executed forward evaluation. Inconclusive E1 neither promotes nor automatically downgrades the lever. | STATUS, `04b --pin-close-hours/--pin-close-band` |
| A14 | Vega sizing inert (`--size-mode vega`, `--size-vega-target`). Do not activate based on the old post-v1 checklist: a new decision and preregistration on the current design are required. | STATUS, `04b` |
| MacroNormalizer | Pin implemented-inert (`--macro-norm` on 04b and replay), declared reference 20260730. Normalizer pinning and parquet promotion are separate decisions; no activation is assumed. | [START](../START.md), STATUS 31/07 and current status |
| B1 / L2 | Stage 1 inconclusive because the positive control failed. No automatic h=3 reopening. At h=30 wait for n_eff=216; dates depend on continuity. No permanent 1m-kline producer. | STATUS 10/08 and current status |
| CAFN | Parked, low prior; reopen only with a revised scope and gate. | [START](../START.md), STATUS 20/07 |
| Replay C1 | The old checklist left the full comparison on future overlapping live/replay ticks pending after funding refresh. Do not confuse it with C2 refactor parity; retain it for reconciliation before claiming complete coverage. | STATUS 18/07 |
| Dashboard | D1–D5 retained (D6 done 2026-09-17): useful proposals, not preregistrations. Result views must not anticipate restricted gates or descriptives. | [DASHBOARD_IMPROVEMENTS](DASHBOARD_IMPROVEMENTS.md) |

## Exhausted items removed from the queue

| ID | Outcome |
|---|---|
| v1 n=20 / n=30 | FAIL 0/3 on 07-18 and 07-30; samples and caveats are in the history. |
| A1 / B2, hedge v2 | FAIL 2/3 on 08-11: variance −55.3%, drag −0.647 SE beyond −0.25 budget; 76% fees. Wind-down completed on 08-13. |
| A12, WW band | Inert code; pre-v2 comparison found no dominance, fixed 0.30 selected. No new activation authorized. |
| A8-BIS / B4-bis / A10 | Mixup FAIL 07-20; DVOL-feature FAIL 07-23; sparsity FAIL 07-30 with manipulation check passed. Training class closed. |
| A4 / C1–C3 | HAR-CJ inputs do not reopen the training class. Smearing not adopted; HAR-C baseline adopted; HAR-CJ respecification exhausted. |
| MFIV | Derivation completed; comparator v2 FAIL 08-18, n=41. ATM remains comparator, MFIV diagnostic; no v3. |
| C2 / C4 infrastructure | 2ter refactor and greeks+sync completed on 07-18, documented in README/START/THEORY. |
| R1 | No longer a task to run on 08-04: use the outcome and provenance constraints in STATUS/THEORY, with no implied model promotion. |

**Checklist retirement:** `POST_GATE_V1.md` and `RIPRESA.md` deleted on 09-10 after reconciliation. Remaining work is in the table above; outcomes and preregistrations remain in STATUS history and Git. Mentions in historical entries refer to files present at that date, not current instructions. Operational reminder retained: review VPS renewal/cancellation around December 2026.
