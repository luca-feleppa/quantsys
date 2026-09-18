🇬🇧 English · [🇮🇹 Italiano](CHANGELOG.it.md)

# QUANTSYS — Changelog

Reverse chronological order (newest on top). "Iterations" 1-10 are the historical directional block (1m); from the 1h pivot onward entries are dated per session and tracked in detail in `STATUS.md` (append-only lab notebook). This file summarizes milestones; `STATUS.md` is the canonical source.

---

## 2026-09-18 — The entry layer: what a reader sees before deciding to open the repo

**Link previews.** `docs/index.html`, `docs/index.it.html` and `docs/architetture.html` carried only a `<title>`: shared on LinkedIn, Reddit or X they rendered as a bare URL, so the first screen was empty where the work is densest. The three pages now declare Open Graph and Twitter-card tags (canonical URL, per-language title/description, locale plus alternate, `summary_large_image`), generated in `build_site.py` rather than hand-written. The English page's canonical is the directory URL, so `…/quantsys/` and `…/index.html` are not counted as two pages.

**`scripts/site/make_social_cards.py`** — the preview images themselves: `docs/assets/og_card.png` (1200x630, English), `og_card.it.png` (Italian), `social_preview.png` (1280x640, for the GitHub repo setting). The card states the claim, so it obeys the same rule as the site: QLIKE values, the percentage and `n` are **read from the judge's report** (`results/vols/qlike_report_1h_test_canonical_1h_vols.json`) and the script fails fast if that report's scaler provenance is not verified. The social copy in `build_site.py` is a template filled from the same reports, so page, card and preview text cannot state different figures.

**README, first screen (both languages).** A badge row and a three-row **"In 30 seconds"** table — what works, what does not, what never did — placed before the technical paragraph that used to open with a config path. The block states the method (pre-registered gates, failures published with the same numbers as the success) and points to the project site for readers who will not open the code.

**Repo metadata:** the GitHub "About" box now links the project site (`homepageUrl`), which was empty.

⚠ **Left undone, deliberately:** the repo's social preview image must be uploaded by hand in Settings (no API exists for it) — the generated 1280x640 file is ready. The repo description is unchanged: it states the gate against HAR-RV, which is correct, and rewording it would say nothing new.

## 2026-09-02 — N-leg `exec_diag` without touching the two-leg record

The `exec_diag.jsonl` aggregates in `scripts/04b_vol_paper.py` now come from a pure function (`exec_diag_aggregate`) computed on the body (first two legs); whole-structure fields appear only beyond two legs. `hedge_dry_run.py` isolates the body via `body_idx`. Replay test over the 1236 historical rows with exact equality (`tests/test_exec_diag_multileg.py`). Not deployed: `04b` unchanged on the VPS.

**`--adaptive` lever in `04b`, inert by default** — DVOL-band entry rule without the NN signal: above threshold short daily straddle (v1 machinery), below threshold short ~7-day iron butterfly (4 market orders, wings first, 120 s completion rule with reverse flatten and `incomplete` record), 4-leg settlement with the 2-leg formula unchanged, frozen parameters required and explicit, `adaptive.jsonl` only with the flag. 10 tests (`tests/test_adaptive_structure.py`), suite 504 passed. Not deployed.

## 2026-08-29 — The architectures become a page, and the denominator stops living in someone's head

**`docs/architetture.html`** — twelve interactive views of the architectures, derived by reading the `forward` passes rather than the documentation: data pipeline, iTransformer, TCN+Mamba, N-HiTS, CAFN, MoE/MoU, output head, plus the internal wiring of attention, convolution, state and decomposition. The diagrams are **data** (row + lane) and the layout places them, so blocks cannot overlap by construction; tensor shapes derive from parameters editable in the page, so you can see what grows with `T` and what does not. **Single-file bilingual** like the rest of the docs: every visible string is an `{it, en}` pair, the static chrome is populated from a dictionary, and the IT/EN toggle does not reset state. ⚠ The Italian branch was diffed **field by field** against the original (text, edge topology, lanes, walkthrough steps) and all 12 views were rendered in **both** languages against a fake DOM: in this kind of work an error produces no visible failure, and a translation that silently rewrites existing copy would be indistinguishable from an improvement.

**`THEORY.md` §12.2** — the section names four denominators (naive, HAR-RV, HAR-CJ, HAR-C) because the baseline was progressively strengthened by three successive gates, and the distinction that matters sat halfway through a forty-line paragraph. It now opens with a **reconciliation panel**: one row per statement, with denominator, numbers and role. The pre-registered **gate** of 2026-06-10 uses **HAR-RV** and is frozen; the published **claim** uses **HAR-C** since C3. Two things made explicit: the `p ≤ 4.3·10⁻⁴` quoted in the README is measured against **HAR-CJ** (gate C2), not against HAR-C; and the band's endpoints are **val and test**, not a confidence interval. In `README.md` the distinction now appears **where the claim is made**, not only in the methodology block two hundred lines below. ⚠ **No new numbers**, mechanically verified: all 25 figures in the panel already appear elsewhere in the file.

## 2026-08-18 — The MFIV comparator closes FAIL: the wedge is large but nearly constant, and ranks do not see it

**Pre-registered MFIV comparator v2 gate FIRED and CLOSED the same day: FAIL**, manual one-shot run at the first session where condition ③ was met. Paired sample `n = 41` daily expiries (threshold 40), verified ex-ante read-only before launching: the routine counter read **42**, the real sample was **41** (the last expiry qualifies on today's ticks but settles tomorrow). ① `Δρ = ρ_MFIV − ρ_ATM = +0.0343` against the required +0.05: FAIL. ② sign not consistent across the two chronological halves (−0.0015 / +0.0610): FAIL. ③ `n ≥ 40`: PASS. The outcome was **predicted by the pre-registration itself**: with a constant wedge the two Spearmans are identical by rank invariance, and the measured wedge is large (log-variance median 0.2005) but weakly dispersed (interdecile 0.1733) — the gate measured only its time variation, which at `n=41` (SE ≈ 0.16) adds no ranking power. **Pre-declared consequence applied:** `04b`'s live comparator stays **ATM IV**, MFIV remains a permanent diagnostic column, item closed, no v3 pre-registration. **Non-gating descriptive, due regardless — short-vol break-even re-estimated:** MFIV prices variance **+22.2%** above the interpolated ATM (+10.55% in vol terms), so the historical backtest's VRP = 0% break-even corresponds to realized variance **18.2% below** the correct var-swap rate: the short-vol arm's cushion is wider than ATM suggested. It is not extra PnL — `04b` sells the ATM straddle and collects the ATM premium — but it argues for structures selling a larger share of the strip. The `--count-only` counter was retired from the routine: its only consumer was the one-shot, now spent. Detail and numbers: `THEORY.md` §12.2, machine record `results/vols/mfiv_comparator_report.json`.

---

## 2026-08-13 — Wind-down closed: the clean window was zero-wide

**`--hedge --hedge-band 999 --hedge-conv raw` removed from the unit: `04b` runs again as unhedged v1, which is the production design.** Precondition verified before touching anything: residual-leg flatten at **08:01:34 UTC** (`h_usd_after: 0.0`, fill 63836.5, fee 8.19e−05 BTC), `hedge_state.json` absent on the VPS and canonically. ⚠ **The `reason` came out `structure_changed`, not `settled`/`expired` as pre-noted yesterday — and it is the third arm of the same guard, not an anomaly** (`pos is None` → *settled*, `expired` → *expired*, `position_key != pos_key` → *structure_changed*): the prediction was incomplete because it was derived from two branches out of three. ⚠ **The cause retroactively confirms yesterday's correction:** at the **same tick** of 08:01:34 the 13AUG structure settled **and** `04b` opened the 23rd (BTC-14AUG26-64000, edge +0.501), so `position.json` was **never** empty and the `settled` branch could not fire. **The post-settlement clean window does not last "a couple of hours": it is zero-wide** — the book is replaced in the very tick it liquidates, and the wind-down band was not the convenient route but **the only one**. **999-band inertness measured:** no ledger event after the flatten, hedge-active counter stuck at 22. **Deploy verified on both sides:** before, the md5 of the in-force unit **identical** to `HEAD` (no drift), with the process restarted at 00:30:03 UTC by the timer, i.e. still on band 999 — confirming that *the commit alone does not change production*; after, `ExecStart` without `--hedge`, remote md5 = local, `is-active`, and in the boot log **no `V2 HEDGE ATTIVO`** and no reconciliation, first tick regular (`edge=+0.628 → HOLD`). The comment above `ExecStart` moves from transient state to **definitive reason**, with the ordering constraint rewritten **symmetrically**: it also applies to anyone re-enabling the flag later and wanting to turn it off. **Hedged counter retired** from the session routine (only the option-leg counter remains, the one number telling whether `04b` is executing): a counter is retired when its **last consumer** is discharged — not at gate closure (on 08/11 it was still needed to prove the band opened no new hedges), and not after (a number without a consumer invites comparing it to a threshold that is no longer its own). ⚠ **Candle refresh verified per-expiry and deliberately NOT applied:** it was worth **+1** (13/08 only; 14/08 matures tomorrow), but at 12/40 the E1 counter has no consumer, the threshold still falls around 09-10/09, and every refresh is a write to a production data file — it stays mandatory **exactly once**, before the E1 stage-2 one-shot run. **Method note:** the routine smoke test with `2>&1 | Select-String` died on `NativeCommandError` — in PS 5.1 the gotcha already known for `Tee-Object` applies to **any** pipe with `2>&1` over a script launching a native exe; replaced by a targeted test (extracting the block via AST and running it standalone), which is also the right test because it isolates the change.

---

## 2026-08-12 — Hedge-leg wind-down: the clean window does not exist, you build it

**The ordering constraint on disabling was not a one-off event: it recurs every cycle.** The 08/11 entry prescribed "wait for the 08:00 UTC settlement, verify the flatten, then restart without `--hedge`": it assumed the hedge state would stay empty from the flatten until the restart. **False.** The 21st position's flatten happened at 08:01:34 (`reason: "settled"`, `h_usd_after: 0`), but at **10:01** `04b` opened the 22nd (strike 63500, 13AUG expiry) and hedged it, rebalancing at 15:01. The clean window lasted **2 hours** and its width is **not controllable**: it depends on when `|book_delta|` crosses the band, and it can be the very first tick. A plan that requires timing it fails intermittently, and each failure costs another day of churn on an already FAILED lever. **Solution taken from the code, not the calendar:** in `maybe_hedge` the flatten branch (`pos is None or expired or position_key != pos_key`) runs **before** the band check (`if abs(book_delta) < band_eff: return`) → an arbitrarily large band **disables opening and rebalancing** but **keeps** the settlement/expiry/structure-change flatten active. Two-step wind-down: today `ExecStart` → `--hedge-band 999` (VPS deploy, restart 15:34:54 UTC); tomorrow, after the 13AUG settlement and with no upper time bound, `ExecStart` returns to `--execute` alone. The residual leg will thus be closed **by the same code that wrote the other 20**, and `hedge_ledger.jsonl` stays internally consistent — the alternative (a hand-placed perp order) would have left an `open` without its `flatten`, i.e. a *wrong* record rather than an *incomplete* one. **Transient state put in the tracked file rather than a systemd drop-in:** a drop-in is not reconstructible from git (the state deployed for ~17h would exist in no versioned artifact) and, if forgotten, silently overrides any future unit; cost of the tracked variant: a one-line commit, and one step fewer. ⚠ **This is not a mid-judgment parameter change:** the `band=0.30`/`conv=raw` freeze held for the gate's duration, closed yesterday, and the judged sample (n=20, last settlement exp 11AUG) is frozen in `results/vols/hedged_vs_unhedged.json`; the 22nd position lies outside the sample by construction. ⚠ **Re-running the judge today with the same `--since` would return 22 settlements, not 20:** the record is the archived JSON, the gate is closed, and it is not re-judged on an enlarged sample. **Inertness verified empirically:** on restart, guard `band=999.0`, no reconciliation warning, bootstrap tick `edge=+0.280 → HOLD` and the hedge leg ran **without producing events** (ledger unchanged, `updated_ts` unchanged). **E1 stage 2: from 8 to 12/40** after `01_update_data.py --candles-only` (+97 candles). ⚠ **The `T+2h` rule was measured, not inherited, and reading the log naively gets it wrong by an hour:** the log prints "Loaded 98 candles → 15:00 UTC" but the merge writes **97** (the forming bar is dropped right there), so `raw_candles.parquet` ends at 14:00 and `hourly_close()`, which drops one more, yields closes usable through **13:00 UTC** when run at 15:22 — two rows lost, not one. Effective margin on the 12/08 constraint (11:00 close): **2h**. ⚠ **The monitor printed `no_rv: 5` but the gain was +4:** the fifth expiry is 13/08, whose decision tick already exists but whose RV window closes tomorrow — verified **per-expiry ex-ante**, before writing to the production file, following the rule that already saved two useless writes on 07/08 and 10/08.

---

## 2026-08-11 — The delta-hedged gate closes FAIL: the hedge does its job, the fees eat it

**Pre-registered hedged-vs-unhedged gate (v2, `04b --hedge`) FIRED and CLOSED the same day: FAIL 2/3**, one-shot run on an explicit decision at the first session where condition ③ was met. Forward sample `n = 20` settlements with the hedge active (19/07 → 11/08; the position already open at activation excluded by pre-declaration). ① **variance: PASS with margin** — `var(hedged)/var(unhedged) = 0.447`, i.e. **−55.3%** against the required −40%. ② **cost: FAIL** — drag `mean(h) − mean(u) = −0.001312 BTC = −0.647·SE` against a pre-registered budget of −0.25·SE, **2.6× the threshold**. ③ n ≥ 20: PASS. **Decomposition** (identity `hedged − unhedged = gross − fee − funding`, residual 1.7e−18): **perp fees 75.9%** (−0.000996 BTC/trade over 76 rebalances, 0.000262 BTC per event), gross perp 23.2% (mean-zero by design, indistinguishable from zero at n=20), **funding 0.9%**. ⚠ **The pre-declared hypothesis was right about the variable and wrong about the component:** on 12/07 the risk was written as "churn fees **+ funding**", with funding flagged as the big unknown *never measured on the series* — measured, it is worth **less than 1%** of the drag. Zero-fee counterfactual: drag −0.156·SE → the gate would be **PASS 3/3**. ⚠ **A qualification that does not move the verdict:** ② is a **budget**, not a significance test — the paired t is **−1.051** (hedged better in 7/20 trades), so what is demonstrated is that the hedge **does not buy PnL** and that realization cost at 1-contract size eats a multiple of the budget, **not** that the drag is negative in population. **Pre-declared consequences applied:** `--hedge` FAILED, the **unhedged v1 remains the production paper design**; v2 sizing (HAR-q90) and A7 greeks-aware do **not** unlock, both were conditional on a PASS. **B2 (hedge = VRP purifier) dies on realization cost, not on theory** — ① proves it. Third time in this project that a measured effect fails to survive transaction costs, after the 1m cost wall and the net-of-cost IVS KILL. ⚠ **Two unit mismatches caught before running:** the routine counter read `21` (positions with ≥1 executed hedge, including the one **still open**) against the pre-reg's `20` **settlements** — the third variant of the same error on this counter after "events ≠ positions" on 26/07; and `--since` was verified **sample-invariant** across the entire admissible window, so its choice is not a researcher degree of freedom. ⚠ **Disabling has an ORDERING constraint the pre-registration did not foresee:** `hedge_state.json` holds an open +19,320 USD perp leg and `maybe_hedge`/`reconcile_hedge_state` run **only** under `--hedge` → restarting without the flag now would leave that perp **naked and never flattened**. Correct order: wait for the 12/08 08:00 UTC settlement, verify the flatten in the ledger, then restart. Verified that disabling **does not perturb** E1 and MFIV: without the flag the path is the bit-identical v1 and the option leg is unchanged.

---

## 2026-08-10 — A file with no producer, and why it stays that way

**`data/raw_candles_1m_l2.parquet` will not get a producer script, and the reason is a property of the data, not a priority call.** The item had sat in the queue for four sessions as "work not done". Its actual weight was measured before deciding: the file appears in **three** consumers only, **none** of which runs on the VPS — `04b` reads IV, macro and **hourly** candles, so the live arm is independent of it. The one consumer executed every session reads it as the *tail* of `hourly_close()`, and since the 1m data is **behind** the hourly bars its measured contribution is **0 rows**. Hence the decision: 1-minute klines are **historical and re-downloadable indefinitely**, so — unlike the L2 recorder, forward-only and unrecoverable if it stops — the "lag" is not a maturing debt but a download not yet done, and the file downloaded today is identical to the one downloaded six months from now. A standing producer would keep current a file no operational path reads: **when it is needed, the download is decided then, over the window it needs.** The monitor still measures coverage and no longer prints a remedy. Its value is recorded so it need not be re-derived: **237 already-recorded L2 hours without a 1m target** (774 with ≥360 snapshots/hour, 537 under the cap) ≈ **+128 evaluation points**, `n_eval` from 289 to ~420. **B1 stage 1 stays closed** — the 31/07 constraint was not data (`N_MIN=240` already exceeded at 289) but the positive control, HAR-C re-estimated on the short window against a zero-parameter naive. **Surfaced in passing:** `VolForecaster._bootstrap` extends and re-persists `raw_candles.parquet` gap-aware, so **on the VPS the hourly candles advance on their own** at each bootstrap — "extending the bars is an act, not an automatism" holds for the **local** copy, the one feeding counters and judges at home.

---

## 2026-08-06 — A sample that looked stalled, and the guard that protects on one side only

**The E1 stage-2 confirmatory sample counter was not at 0: it was at 6.** Two overlapping defects. ① The `0/40` in the continuity log **was not a measurement** — it was the 01/08 opening value carried forward for five days, because that counter was not in the session routine and nobody re-read it. ② Even measured it would have said **2, not 5**: `raw_candles.parquet` was stuck at 02/08 and `realized_rv` returned `None` for every later expiry. ⚠ The combination is worse than the sum: ① ensures nobody looks, ② ensures whoever looks sees a low and **plausible** number. Diagnosis verified, not conjectured: all **7 decision ticks exist** for every expiry from 01/08 to 07/08; only the close series was missing. After `01_update_data.py --candles-only` (66,435 → 66,530 bars; `features.parquet`, `lstm_dataset.npz`, scaler and `PipelineState` **untouched**) the count moves to **6/40**, window 01/08 → 06/08, **one observation per day since opening**. **No data lost:** the ticks live in `forecasts.parquet`, merged append-only from the VPS — the lag was one of **observability**, not collection, and it is **recoverable**, unlike a macro vintage promoted inside an open sample. ETA unchanged: n≥40 on **09-10/09**.

**1m bar file coverage added to monitoring — recorder continuity ≠ target availability.** The L2 check measures **recorder** continuity; the new check measures whether the **target** those hours would be judged against exists (the B1 judge builds RV from **1-minute** returns). The usable sample is the **minimum of the two**, and neither counter said so — the same shape as today's E1 defect: a number growing while something else caps it. First run: **87,217 bars, 2026-06-01 → 2026-07-31, 6 days of recorded L2 without a 1m target**. ⚠ The message **qualifies the horizon**, since without it the alarm would point at the wrong gate: the cap applies to 1m-target analyses (B1 at h=3, pin-close proxies), while `n_eff` at h=30 uses **hourly** bars and is unaffected. ⚠ The file has **three consumers and zero producers** — no repo script generates or extends it, it was acquired by hand — so the monitor **only measures** and states that extension is manual via `quantsys.data.fetch_klines`, rather than printing a non-existent remedy. Writing the producer remains separate work, not done.

**`-RefreshCandles`: extending the bars becomes an explicit act, not a default.** Code audit: automating the refresh would be **mechanically safe** — no-op when nothing is new (exit *before* writing), dedup keeping the **existing** row so history is never rewritten, never the in-progress bar, atomic write, and the file **never reaches the VPS** (the only home→VPS transfer is macro). It stays a flag for three reasons written in the repo: (i) block ③ is documented as *off-path and write-free*, a categorical a future reader leans on; (ii) extending moves the B7 staleness counter, which at ≥168 bars launches the incremental regime refresh on its own — **today 95/168**; (iii) **decisive**, under automatic extension **freezing the data becomes impossible**: the *"candles/npz/regime_probs untouched until the gate closes"* invariant would go from "do nothing" to "remember `-SkipMonitor`", the exact shape of the macro promotion that happened by automation on 31/07. Provenance corollary: the split is a **fraction of the row count**, so every appended bar shifts the train/val/test boundaries at the next rebuild — the dataset vintage would become a function of *how many sessions were opened*, i.e. no longer declarable. ⚠ Honest qualification: **no open pre-registration currently freezes the candles**, so automation would have violated nothing current. Adopted form: step ②bis **after** the B7 check, so any regime refresh starts at the **next** startup and the two writes stay separate. Among the open gates **only E1 reads the bars** — verified in the judges.

**E1 counter added to routine block ③ — and the obvious way to do it was wrong.** The `n<40 → NO_RUN` guard protects **below threshold only**: at n≥40 a bare `--stage 2` computes the three conditions, prints the verdict and writes the report, i.e. it would have fired the confirmatory run **by automation rather than by decision**, on the day the threshold is met. Added `--count-only` to the E1 judge (it stops at the count at **any** n: no statistics, no report), with a test verifying this on a sample built **above** threshold — below threshold the test would prove nothing, since the bare command would pass it too. Building the panel **is** the count, but `x`, `y` and the statistics are neither printed nor written: one sees **how many** observations exist, never **what they are worth**. ⚠ The candle refresh is **deliberately not automated**: the block prints how far the close series reaches and warns past a 6h lag, but the remedy stays an explicit command — automating it would turn block ③ from *off-path and write-free* into *writes a data file every session*, the same class of automation that produced an undecided macro promotion on 31/07. Suite **491 passed / 1 skipped**; `START.md` §5.3 and `scripts/README.md` aligned.

---

## 2026-08-05 — The published band written at its artifact's precision

**Band re-expressed: `−23% to −32%` → `−22.42% to −31.65%`. Same numbers, no re-measurement, zero GPU** — the canonical pair's NN/HAR-C ratios (val `0.7757926`, test `0.6834522`) written out rather than rounded to the point. The endpoints move for **different** reasons: the **lower** one was **stale** (−23% is the rounding of the −22.6% C2 measured against **HAR-CJ**, carried over when C3 replaced the denominator with HAR-C and never realigned — against HAR-C the value was always −22.42%, the gap being **conservative**); the **upper** one was not stale but **rounded to the point**, and it was the only one rounded in the direction that **favors** the claim. Aligning the precisions removes the asymmetry and makes the claim **slightly more conservative**. ⚠ **The two decimals are not a confidence interval:** they identify *which* (model, npz, config) triple produces the number. Measured uncertainty remains ~±0.7 points with respect to the training config, 0.0019 with respect to the macro vintage, **zero** with respect to the seed — none of them stochastic, all removed by declaring the triple. Propagated to `THEORY.md` §12.2 (with a new dedicated IT+EN paragraph), `README.md` §5.1 and the `tests/test_har_c_baseline.py` header comments; **the dated C2 and C3 records were not rewritten** — they state which band was decided then, and were true when written.

**02/08 backlog cleared.** (i) `astype(np.float32, copy=False)` in `02_train.py`'s `to_t()`: **−2.42 GiB peak**, bit-identical — the second half of the peak whose first half was the in-place `clamp_`, and the two are correct **only together**, since under `copy=False` the clamp writes straight into the npz member. The invariant making it safe (`NpzFile.__getitem__` materialises a fresh array per access) is not assumed: it is pinned by the 7 tests in `tests/test_npz_load_aliasing.py`. (ii) **Regime walk-forward fit guard exercised on real data** — the `max_fit_failure_ratio` abort threshold (0.5, introduced 02/08) had never been measured in the field, and an abort threshold never seen in the field is an **availability** risk: were real fits to fail above 50%, the guard would make impossible the very rebuild it was meant to protect. **Read-only** probe over 2 years of real candles (17,520 hours) at production cadence: **8 fits out of 8, `fail_ratio` = 0.000, coverage 100.0%**, a 0.5 margin to the threshold, `last_fit_diagnostics` populated on a real run. No regime file rewritten.

**Macro refresh deliberately NOT performed.** Two open pre-registered forward samples depend on the model input: hedged (16/20, ~09/08) and **E1 stage 2, at 0/40 and accumulating until ~10/09**. Promoting a vintage today would split E1 stage 2 across two normalizations — and the `MacroNormalizer` is refit **whole-df**, so a refresh changes macro values for historical rows too. A vintage can be re-pointed back; forward observations already collected cannot. Clean window: after ~10/09.

**Gate M1 pre-registered and RUN the same day: PASS ⓪①②③④, zero GPU.** The train↔inference identity fingerprint now also covers the **macro vintage**: `02_train.py` records into the `PipelineState` the per-split md5 of the consumed npz's `X_macro_*` (plus column order, count and dtype; source `measured`) and the three vol judges recompute it and fail fast, with `--allow-macro-mismatch` kept separate from the scaler escape. ⓪ **bit-identical inertia on both splits** against R1's archived reports (118 common keys, **0** numeric differences; the only two are path labels in the `provenance` block, 18 new keys all under `provenance.macro`, none lost). ① two-level positive control: a single changed cell, a column reordering at identical values, a dtype change and the end-to-end case must each trip the guard — without it, a guard always returning `true` would have passed the inertia check perfectly. ② the three pre-M1 model dirs stay `matches: null` = not verifiable, never `true`. ③ live path unreachable, verified by grep **and** by a parametrized test over four files: a fail-fast reachable from `04b` would stop the forward test at bootstrap inside open samples. ④ suite **490 passed / 1 skipped**. ⚠ **Two limits declared and not closed:** the positive control is **synthetic** (vintage V1 no longer exists and rebuilding it would rewrite the frozen npz), and the three pre-M1 models — canonical artifact included — will never carry the fingerprint. ⚠ **The backfill foreseen by the pre-registration was deliberately NOT performed:** a fingerprint recomputed today from the current npz matches **by construction**, hence carries no evidence and would merely convert an honest `null` into a reassuring `IDENTICAL` — condition ② violated through the front door. Less was done than the PASS authorized, never more.

**Pre-registration M1 as written:** extend the train↔inference identity fingerprint to the npz's **macro vintage**. Today the guard covers the price RobustScaler and `target_scale` but not macro normalization, which does not live in the canonical `PipelineState` — so two models trained on different macro both pass `matches: true`, with a measured gap of `0.0019` on the published ratio. Fingerprint chosen ex-ante: md5 of `X_macro_train` read from the npz at training time and persisted in the model state. Cost **zero GPU**; pre-registered regardless, because it touches the fingerprint deciding whether a numerator is comparable. Conditions: bit-identical inertia of the published numerator on both splits, a **positive control** (the guard must fail when it should — without it, a guard always returning `true` would pass the inertia check perfectly), no `null` conflated with "verified", and **the live path unreachable from the new fail-fast** (it would stop the forward test inside three open samples). ⚠ Limit declared ex-ante: vintage V1 no longer exists and rebuilding it would rewrite the frozen npz, so the positive control is **synthetic** — it shows the guard distinguishes two different macro sets, not that it would have caught that historical event.

---

## 2026-08-04 — The published band becomes verifiable again, and its declared uncertainty was an inference

**Gate R1 run and closed: PASS on ⓪①②③④.** The canonical model ↔ npz pair was trained at **unchanged** production config on the frozen npz and lives as a permanent artifact at `models/canonical_1h_vols/` (5 checkpoints, `pipeline_state.pkl`, both judge reports, a `PROVENANCE.md` carrying the model↔dataset binding and the scaler fingerprints). Outcomes: ⓪ baselines reproduced **digit for digit** on both splits (val n=6485, test n=6486) → same npz; ① `provenance.matches = true` **without** the escape hatch; ② the historical gate against **HAR-RV** survives (val 0.26143 ≤ 0.33913; test 0.23637 ≤ 0.35149, both far below naive); ③④ materiality met against **HAR-C** — **0.7758 on val (−22.4%) and 0.6835 on test (−31.7%)** against pre-declared thresholds of 0.80 and 0.71. **The published band does not change by a single digit**: the numerator matches **exactly** the published one. What changes is the claim's **status** — from a statement about the training protocol to a re-judgeable artifact (`dev_vols_qlike.py --arch canonical_1h_vols`). ⚠ **No promotion**: `models/itransformer` and the VPS stay untouched while pre-registered forward samples are open.

**Retraction: "~0.2 points of seed-draw uncertainty" was an inference, not a measurement, and it is false.** §12.2 attributed the gap between two replicas (val 0.26206 vs 0.26143) to the RNG draw. Measured: at fixed seeds, config and npz the protocol is **deterministic** — the canonical pair reproduces an earlier replica to the **tenth digit on both splits** (delta 0.000e+00), so retraining dispersion is **zero**. The reports on disk form two deterministic clusters and the difference between them **is not seed noise** but a config or code-vintage difference, **not identified**: written as such in §12.2 IT+EN. It is the second retraction in three days with the same shape — a plausible cause written without the experiment that would test it.

**Diagnostic arm B (val only): 1.61× speedup, adoption REJECTED, and the useful result is the Δ.** Retraining at `batch_size 128`/`ga 1` cuts wall-clock from 28.5 to 17.7 min (≥1.5× ✅) but moves the NN/HAR-C ratio from 0.7758 to **0.7690**, i.e. `|Δ| = 0.0068` against a pre-registered adoption threshold of 0.005 → **production config stays at `64`/`2`**. ⚠ B is *better* on val and that does not count: the arms' roles were fixed ex ante, and picking one after seeing the results would be selection on the outcome. The informative number is that a **purely computational** knob — at identical data, architecture and seeds — moves the published ratio by **3.5× the gap** between the two deterministic clusters that was attributed to seed draw yesterday. Two complementary facts measured on the same day: **the seed does not move the ratio (Δ=0), the config does (Δ=0.0068)**. It follows that the band's endpoints carry about **±0.7 percentage points** of uncertainty with respect to the training configuration — an uncertainty that is **not stochastic** and is removed by **declaring** the config rather than averaging over replicas. Production config restored and verified (`git diff --exit-code` clean); suite **469 passed, 1 skipped**.

**The two deterministic clusters: cause identified — it is the npz's MACRO VINTAGE, not the code and not the seeds.** Zero-cost investigation (logs + git). Only two npz rewrites (19/07 16:54:57 and 30/07 20:55:02): every run in the 0.26206 cluster sits between them, every run in the 0.26143 cluster after the second, and the first of those starts **87 seconds** after the rewrite. Mechanism verified in the code: `01b_download_macro.py` replaces **only** `X_macro_{split}` and leaves `X_*`, `y_*`, `t_*` untouched; macro is a **model input** (90 columns) but enters **no HAR baseline** → the NN moves while baselines stay identical digit for digit, exactly the observed signature. ⚠ It is not an append: the `MacroNormalizer` is **refit whole-df**, so extending the series changes macro values **for historical rows too**. Alternatives excluded: seed (Δ=0, measured), batch (811 batches/epoch throughout), code (the window's two commits are one logging-only and one inert at flag-off, adding no `nn.Parameter` hence consuming no RNG). ⚠ **A false lead worth recording:** cluster B's run SHAs no longer exist (pre-rewrite, 27/07) and the boundary falls next to them — it looked like an explanation and could not be one, since rewriting history does not change file contents. **Declared, unclosed gap:** the identity guard covers the price RobustScaler and `target_scale`, not macro normalization (absent from the canonical `PipelineState`), so two models can both pass `matches: true` while trained on different macro — **0.0019** of published-ratio gap, today flagged by nothing. **Balance sheet on the band's precision:** seed 0, macro refresh 0.0019, training config 0.0068 — all **deterministic and declarable** sources, none stochastic.

**Three defects found while running the gate, none of which was the experiment.** ① **The scaler guard was blind under sandbox:** `check_model_dataset_scaler` resolved the canonical via `models_root()`, which under `QUANTSYS_MODELS_ROOT` points **inside** the sandbox, where no canonical ever exists → `matches=None` plus a warning, i.e. no check at all **in exactly the mode where candidates are judged**, the guard's only use case. New `canonical_state_path()`: sandbox-local canonical if present (an experiment with its own npz), otherwise the default-root one; the judge was re-run on val with the fix and `nn_qlike` came out **bit-identical** → the fix touches provenance, not the metric. ② **The report name did not carry the arch:** judging an artifact living as an arch-dir inside `models/` would have written to the BARE name `qlike_report_1h_val.json`, **over the historical production report** cited in §12.2, exiting 0 and printing a correct PASS — silent destruction whose only trace would have been a file holding plausible numbers. Rule extracted into `report_filename()`, production path name unchanged, non-clobber verified by md5. ③ **`--arch` did not accept the artifact**, which would have been a checkpoint declared reproducible yet **not verifiable** — the very defect R1 closes. Seven new tests: `tests/test_qlike_report_naming.py` (4, one of which demands val and test never collide because the test split is one-shot) + 3 in `tests/test_scaler_identity_guard.py` (one runs **with the sandbox env set** and demands `matches is True`; one is the sentinel on the canonical artifact).

---

## 2026-08-03 — An alarm that was a measurement artifact, and the canonical pair pre-registered

**The "02/08 L2 recorder gap" never existed, and its cause has been removed.** That hour holds 720/720 snapshots and the contiguous run is 462+28=490h: never broken. The monitor `scripts/vol/l2_continuity_check.py` measured the **local mirror** while believing it measured the recorder — the span ended at `ts[-1].floor("h")`, **by construction** the hour containing the last tick, hence in progress and partial; with a 360 threshold against a real 720/h rate **the verdict depended on the minute** the routine ran. Second channel: the pull fetches dailies over `scp` with no remote atomicity, so the tail can lag by one pull. Fix: the in-progress hour is **excluded** from the span (conservative count), gaps within the last `--provisional-hours` (default 6) are flagged **PROVISIONAL** and excluded from the cost estimate — *a gap is a fact only after surviving a second pull* — while consolidated gaps stay full alarms. Logic extracted into `analyze()` to be testable: `tests/test_l2_continuity_check.py` (5), one of which demands an unchanged current run across five in-progress-hour fill levels (1, 60, 359, 361, 720). **Generalizable diagnostic symptom: if the verdict depends on the instant you take the measurement, it is a measurement artifact, not a fact about the data.**

**Gate R1 pre-registered (OPEN, never run): canonical model ↔ npz pair.** The published −23% to −32% band takes as numerator a pair retrained in a sandbox that was later deleted, so today it is a statement about the **protocol**, not about a verifiable artifact. R1 produces one, at **unchanged** production config on a **frozen** npz, with four pre-declared conditions: ⓪ a **model-independent** vintage check (the baselines must match digit for digit: they are fitted inside the npz, so if they diverge the dataset is a different one and there is nothing to re-publish), ① scaler identity (`matches: true`, without `--allow-scaler-mismatch`), ② survival of the historical gate against **HAR-RV** (the *gate*'s denominator, not the *claim*'s), ③ materiality `NN/HAR-C ≤ 0.80`. Test is one-shot, only after val passes. A diagnostic **Lever B** arm (`batch_size 128`, `ga 1`) runs **on val only** and by construction **cannot** become the canonical pair: the role is fixed ex ante because picking it afterwards would be selection on the outcome. ⚠ A PASS **does not authorize promotion**: swapping `04b`'s model inside open pre-registered forward samples is the same violation as refreshing macro inside a sample. **Two collateral corrections:** one ratio in the §12.2 provenance note carried the wrong denominator (0.7738 = first replica against HAR-CJ, instead of 0.7777 = second replica against HAR-C; the ~0.2-point spread and every public claim are unchanged), and `scripts/00_check_setup.py` printed a **red** ✗ for every missing pipeline artifact while the final verdict said "setup verified" — on a fresh clone those artifacts are missing **by definition** and indeed do not feed the verdict: they are warnings now.

---

## 2026-08-02 — Performance audit: training is launch-bound, and two fewer silent failures

**Diagnostic reconnaissance (`docs/PERF_AUDIT.md`, no optimization applied) plus the two fixes it surfaced.** The central result inverts the intuition: **training is not compute-bound**. At batch 64 the GPU sits at 5-15% SM and per-step time is dominated by kernel **launch** — going from batch 32 to 128 quadruples the arithmetic and grows wall-clock by 16%; the curve turns linear only beyond batch 512 (SM 96-98%). Consequence: levers reducing arithmetic (`channels_last`, Numba, native extensions, Polars) do not touch the bottleneck. `torch.compile(backend="cudagraphs")` — the only one attacking launches — measures **1.56×** per step (15.30 → 9.79 ms), without Triton (not installable from PyPI on Windows) and without graph breaks (Dynamo traces 1 graph, 88 ops); ⚠ **not applied**: it changes the weights, hence requires a pre-registered gate. Rejected with the technical reason: `channels_last` (**no 4D NCHW tensor** in the project — only Conv1d and batched attention), Numba (the bootstrap CI is **already** a `(5000,n)` NumPy matrix, 37 ms; the backtest event loop costs 0.2-0.9 s and is full of Enums/dataclasses), a native extension (no component both heavy **and** isolated; the longest computation — the regime walk-forward — lives inside statsmodels and is already solved by B7), Polars in the `FeatureBuilder` (full data prep is **2.24 s**, 59% of it the Volume Profile's Python loop which Polars cannot express; and the prototype shows `ret_skew_20` — a list-104 feature — would change by **7.7%** because Polars uses the **biased** estimator and pandas the **unbiased** one: a definitional difference, not rounding). Identified the best gain/complexity lever, **not applied**: the `np.nanpercentile` clip bounds cost **31-36 s** per `02_train` invocation, sorting 647M cells that are **52,001 distinct bars repeated 120×** (stride 1); computing them on distinct bars is **200× faster** but moves 34 of 104 columns → a lever, not a cleanup.

**Two silent failures removed** (both bit-invariant on the success path, both surfaced by the audit rather than sought). ① **DLL initialization order:** loading pyarrow after both torch **and** scikit-learn causes an access violation (exit 139, no traceback) at the first `read_parquet` — a clash between the two OpenMP runtimes. The numbered scripts survived only because they import `pandas` at line 30 and `torch` at line 31: a **de facto invariant, never stated nor tested**, which a new script written in the natural order (project first, pandas second) would violate. `import pyarrow` anchored in `quantsys/__init__.py` makes it a package property. ② **Silent regime walk-forward degradation:** every failed Markov-Switching fit produced one `log.warning` per timestep and the loop carried on; with `current_params=None` probabilities were never written and `fit_predict_walkforward` returned the **uniform prior dressed up as regimes**, without anything failing (and the `_fit_single → None` branch produced **not even a log**). Now `RuntimeError` on zero successful fits — not disableable, because a walk-forward without a single fit yields no information — plus a configurable abort on `max_fit_failure_ratio` (default 0.5) and diagnostics persisted in `last_fit_diagnostics`; the guard is mirrored in `continue_walkforward`, where failure yields **stale** parameters instead of the prior. Tests: `tests/test_import_order.py` (4) + `tests/test_regime_fit_guard.py` (8) → **450 passed, 1 skipped**. B7 incremental-regime bit-parity unchanged.

---

## 2026-07-31 (3) — Instrument vs state: pinnable `MacroNormalizer`, E1 pre-registered

**The second half of the macro problem.** Dated vintages (previous entry) settle *which* file reaches the VPS; what remained is that `VolForecaster` **refitted** the `MacroNormalizer` whole-df at every bootstrap, so extending the parquet moves median and IQR and **the measuring instrument changes together with the state it must measure** — 2.7% of the total variation at the 31/07 breakpoint. Extracted `macro_snapshot()` (two branches: legacy `refit` and on-disk pin), added `scripts/vol/pin_macro_normalizer.py` and the `--macro-norm` flag to `04b` **and the replay**. An **explicit parameter, never from env**, same principle as `--arch`: a stale env would silently change the live input, and the replay must pick its regime by the **date** of the decision it reproduces, not by the environment. **Inertia proven end-to-end on the production parquet: 0 differences over 90 columns.** Fail-fast guard on the pin's columns (order included): applying the median and IQR of the wrong column would be silent and permanent. ⚠ The vintage `models/itransformer` was trained under is **not reconstructible**: the pin **fixes** a declared one, it does not recover it; enabling it today is a **content no-op** and prevents future drift. **Gate E1 pre-registered and run at stage 1** (exploratory, no verdict): does the NN-vs-IV edge carry predictive content on realized variance at 30h? Confirmatory stage 2 at n≥40 expiries → ~10 September 2026. **Ex-ante condition ③ on A13a** (pin-close): `n_eff = n_trig`, not `n_positions` — positions that do not trigger are bit-identical under the two rules; A13 parked with dated reopening conditions. Tests: 3 new files (18 tests). See `STATUS.md` 2026-07-31 session 2.

---

## 2026-07-31 (2) — The live macro snapshot becomes a versioned artifact

**The home→VPS macro push overwrote the canonical on every pull, unconditionally** — and on 2026-07-31 it delivered a refreshed macro to the live path **inside two open forward samples**, de facto taking a decision that was still pending. The underlying defect was not the push, though: it was that the snapshot feeding `04b` (read at the nightly bootstrap and **frozen** for the day) was **mutable state** rather than a versioned artifact — so much so that the 31/07 breakpoint could not be measured directly, the old file having been overwritten and not being in git. Block 0 of `pull_vps_data.ps1` rewritten in three parts: (a) an **append-only** archive `data/macro/macro_features_<YYYYMMDD>.parquet` on the VPS, `<YYYYMMDD>` = last index date (`scripts/vps/macro_vintage.py`, new) — 716 KB per copy, so every forward decision stays traceable to its vintage and the replay becomes reproducible again; (b) the canonical becomes a **symlink** into the archive, so the live vintage is readable with `readlink` and visible in `ls -l` — no marker that could lie, the pointer **is** the truth; (c) repointing requires `-PromoteMacro`, so **the push stops being a decision and promoting becomes one**; on a diverging vintage a warning is emitted and the live path stays put. The "run `01b` on the VPS too" alternative was rejected: two independent FRED/yfinance fetches diverge **over history** (FRED series are revised retroactively), which would break the reproducibility of `vol_paper_replay.py` — the tool that proved live↔training parity — and would remove the human from the loop entirely instead of restoring them to it. No live-path change (`vol_forecaster.py` still reads the same canonical path), so the work is admissible with forward samples open. Tests: `tests/test_macro_vintage.py` 4/4 (CLI contract: stdout = **exactly** the vintage, clean failures with empty stdout); block-0 branches verified on 5 cases with stubbed `ssh`/`scp`. See `STATUS.md` 2026-07-31 (session 2).

---

## 2026-07-26 (2) — Diebold-Mariano on the NN-vs-HAR comparison + repo ready to publish

**The "beats HAR by 30% in QLIKE" claim now carries inference, not just a point estimate.** Added `qlike_series()` (per-sample loss; `qlike()` is its mean → one formula, one place) and `diebold_mariano()` with a **HAC Newey-West variance (Bartlett kernel, lag `q = h−1 = 29`)** plus the **Harvey-Leybourne-Newbold** small-sample correction: required because the target sums 30 bars, windows overlap, and under an iid variance the standard error would be understated by ~√h (`n_eff ≈ n/h` ≈ 216, not 6.5k). Outcomes on a retrained model/scaler pair (5 seeds, sandbox, production untouched): **val −26.6%, p = 7.3e-05 · test −36.1%, p = 1.7e-06**; the NN beats HAR **in every regime**, stress included and validated on test. The honest band for the claim is **−27% to −36%** depending on split and vintage (the −36% and the historical −30.2% measure different populations: extended test window + retrained model). DM is **descriptive, not gating**: pre-registered thresholds remain the QLIKE ratios. Tests: `tests/test_diebold_mariano.py` 9/9, including the one verifying SE inflation on overlapping differentials — without which the p-values would be fictitious. **Repo ready to publish:** secret audit PASS across all revisions, OpSec hygiene, assistant attribution removed (history rewritten, 141 trailers, bit-identical tree), KILL corpus and protocol moved into `THEORY.md` §12. See `STATUS.md` 2026-07-26 ⑦⑧.

---

## 2026-07-26 — Hedged-counter unit fixed + STATUS archive committed

Yesterday's automated counter printed `hedge_ledger: {n} events` next to the `n≥20 hedge-active` threshold, but the hedged-vs-unhedged judge's pre-registered sample is defined in **trades opened with the hedge active**, not ledger events (1 position = open + N rebalance + flatten): at 22 events the line invited reading "22 ≥ 20" and launching a one-shot judge **early**. Fixed to distinct `position_key` with ≥1 executed hedge (**n=6** on 26/07, judge expected ~09-10/08), with the pre-declared exclusion of the partially-hedged position now **explicit in code**. A risk introduced by the automation, not a data error: the lesson is that **automating a counter requires checking that the printed unit is the pre-registration's unit**. Also committed the `STATUS.md` history split into `docs/STATUS_ARCHIVE_2026H1.md` (verified **literal**: 809 lines removed, 0 missing from the archive) plus the associated doc sync. See `STATUS.md` 2026-07-26.

---

## 2026-07-25 — Session routine automated (block ③, vol monitoring)

`avvio_sessione.ps1` covered only ① VPS pull+merge and ② B7 regime freshness, while the recurring routine required 3 more forgettable manual steps. Added **block ③ "recurring vol-line monitoring"** (`-SkipMonitor` to skip): incremental `derive_mfiv.py` — **ordering constraint: after the merge**, since it reads the freshly pulled chain — + `mfiv_comparator_judge.py --count-only` + a trailing printout of both forward-gate counters (`n executed` from `trades.jsonl`, `hedge_ledger.jsonl` events). **Discipline invariant preserved:** `--count-only` computes timestamps only and the judge still holds the `n<N_MIN=40 → NO_RUN` guard without writing a report → automating the count **cannot** produce peeking; the one-shot run stays MANUAL. Fail-soft with explicit `$LASTEXITCODE` checks (in PS 5.1 a native exe exiting ≠0 does NOT raise: `try/catch` is not enough) and the file re-verified ASCII-only. Validation: end-to-end `-SkipPull` run OK, block ③ idempotent. See `STATUS.md` 2026-07-25.

---

## 2026-07-19 → 07-23 — Vol-line training levers: A8-BIS and B4-bis both FAIL on val

Two pre-registered gates run and closed negative, both **against a baseline retrained on the same extended dataset** (not against the production incumbent). **A8-BIS mixup** (`mixup_alpha` 0→0.2): 0.26206 baseline vs 0.25998 candidate = −0.79% ≪ the −3% threshold → FAIL, overlay deleted, no test. First-order methodological lesson: the −4.94% descriptive figure from Phase B was a **distribution-shift artifact** (cross-scaler comparison against the old-scaler incumbent) → **never decide a lever on comparisons across different scalers**. **B4-bis DVOL-as-feature** (pre-reg `5a6112d`, close `526659d`): `dev_vols_dvol_append.py` extends X_macro 90→93 (`dvol_log`/`dvol_chg_24h`/`dvol_avail`, causal asof, 24h cap, train-only median fill, val coverage 1.000, production npz untouched); 0.26206 baseline vs 0.25939 candidate = −1.02% vs the −3% threshold → **① FAIL** (② would have passed, AND-gate), no test one-shot. Reading: 30d→30h tenor mismatch + an incumbent already capturing IV information through lagged RV → log-MSE improves −6% but QLIKE does not, and QLIKE is where σ² calibration counts. The **standard ③-pattern** was also ratified (per-regime count condition verified ex-ante: if model-independent the outcome is "no conclusion", not a result). See `STATUS.md` 2026-07-19/20/23.

---

## 2026-07-18 — v1 gate CLOSED: FAIL 0/3 · `04b` migrated to the VPS · MFIV@30h (D4)

**Short-vol arm v1 gate closed at the pre-registered n=20 checkpoint: FAIL 0/3 on BOTH samples** (concordant). Always-short +0.0396 → VRP stays positive and confirms the arm, but the v1 rule does not monetize it; `POST_GATE_V1.md` unblocked. **`04b` migrated to the VPS** (`quantsys-volpaper` systemd, `--hedge --hedge-band 0.30 --hedge-conv raw`): the v2 forward test starts and the home PC becomes **passive** — ⚠ never launch `04b` at home again, duplicate orders on the same testnet position. **🔴 Candle bug:** the live window had been holed since ~06-24 (fixed + guard added). **D4 model-free MFIV@30h** (`scripts/vol/derive_mfiv.py`): OFFLINE retroactive derivation of the VIX-style var-swap rate at 30h tenor + 25Δ RR/BF skew from the already-recorded raw chain → PARALLEL columns, never in the decision path; median MFIV−ATM convexity wedge **+3.45 vol pt** → the short-vol break-even computed on ATM IV was **conservative**. `VolForecaster` promoted from `04b` into `quantsys/model/vol_forecaster.py`. See `STATUS.md` 2026-07-18.

---

## 2026-07-14 → 07-16 — 24/7 VPS collector + B7 incremental regime + 1h GJR + A11-A14

**24/7 VPS collector** (EU VPS, Ubuntu 24.04, geo-test PASS): `quantsys-iv` / `quantsys-ob` / `quantsys-trades` systemd services — the new **`01e_trades_recorder.py`** records production Deribit options trades (API retention ~24h verified → collection is necessarily FORWARD) to measure realized spreads vs mark. Host lives in `config/secrets.yaml → vps.host`, **never read or print it**. Home-side sync: `scripts/vps/pull_vps_data.ps1` (+ ssh/scp anti-hang fix). **B7 — incremental walk-forward regime**: the full rebuild saves `data/regime_wf_checkpoint.pkl`, `01b --regime-incremental` appends only new bars (**minutes instead of ~3h**), `--regime-bootstrap-checkpoint` rebuilds the state with fail-fast replay validation; bit-parity enforced by `tests/test_regime_incremental.py`, bootstrap validated bit-exact. On the 7-year rebuild the **regime labels are REMAPPED** (R1 = stress now): re-derive them from the variances at every full rebuild. **1h GJR closed**: QMLE re-estimation of the MC parameters on hourly returns (`scripts/vol/estimate_gjr_1h.py`) → γ≈0.005, near-zero leverage effect at 1h; per-bar σ cap made parametric. **A11-A14 (gamma functions):** `pnl_attribution.py` ACTIVE (ex-post delta/gamma/theta/vega decomposition, read-only); A12 WW band, A13 pin-close+gamma-cap, A14 vega-sizing implemented **INERT** (activation only under a post-gate v2 pre-registration). See `STATUS.md` 2026-07-14/15/16.

---

## 2026-07-08 — Vol-book v2 roadmap + A6 exec-diag in `04b` + macro refresh

**Roadmap** (`docs/ROADMAP_VOL_BOOK.md`, 07-07 advisory session): A1-A10 backlog from the anti-overfit audit (closed verdicts: FrAug/vol-RevIN/MC-dropout/N-HiTS-interp NO) + two-instrument book strategic verdict — B1 leveraged directional futures ❌ (odd moments falsified OOS, E[PnL] = leverage·(0−costs) < 0), B2 Deribit perp as options-book **delta-hedge** ✅ (PnL = ∫½ΓS²(σ²impl−σ²real)dt = pure VRP harvest, exactly what the NN predicts), binding B3 sequencing. **A6 implemented** (`scripts/04b_vol_paper.py`, the only pre-gate item): `log_exec_diag()` at end of tick → `results/vol_paper/exec_diag.jsonl` (per-leg Deribit bid/ask/mark/IV/greeks + net delta + half-spread; open position → live legs, flat → hypothetical ATM straddle), fail-soft, **pre-registered rule and constants UNTOUCHED**; real-testnet smoke + live process restarted (first row: 18.2% half-spread ≈ the 16% haircut from the 06-25 validation). **Macro refresh**: `macro_features.parquet` 06-10→07-08 (macro-only section of `01b`, identical 90-col schema; regime/npz/normalizer deliberately untouched). See `STATUS.md` 2026-07-08.

**A2a+A5 run and FAILED the same day (pre-registered gates, zero retrain — the quantile head was ALREADY in the PASS checkpoints).** A2a (`scripts/vol/dev_vols_quantile_judge.py`): coverage above target at every level (q50→0.73, q90→0.97 = upward-shifted distribution) and NN q90 pinball loses to HAR+residual-quantiles (0.160 vs 0.144) → **A2b (q95 retrain) dead**; for v2 sizing the right tail is HAR-q90 or a pre-registered conformal recalibration. A5 (`scripts/vol/dev_vols_member_weights.py`, fit 1st val half / eval 2nd): ratio 0.9925 vs the ≤0.97 gate, near-uniform weights, best-single worse than the ensemble → **uniform weights confirmed optimal**. Production checkpoints read-only, test split untouched, live processes paused ~5 min and relaunched healthy.

---

## 2026-06-26 — Short-vol statistical audit (1 conclusion corrected, 1 prior refuted) + B2/B3 perf golden Δ=0

Statistical/logical audit of the short-vol backtest (causality/lookahead verified CLEAN throughout). Fixes in `scripts/vol/`: **①** block-bootstrap CI + effective-N — PnL lag-1 autocorr ≈0 → N_eff≈N(2538), CIs>0 (the 30h/24h overlap does NOT inflate lag-1 significance; the bootstrap does not capture the 2020-21=90% temporal concentration). **②** Sharpe annualized at √(trades/yr)≈√365 (was √292, inconsistent with the daily cadence). **③** REGIME-dependent bid haircut (`--stress-haircut-mult`) → **CORRECTS the 06-25 regime conclusion**: "edge highest in Stress" was a constant-haircut artifact; with realistic spreads **Trending dominates** and Stress is marginal (SIGN holds, HIERARCHY doesn't). "Do not filter regime" SURVIVES robustly. **④** load-bearing n=3 caveat made explicit. **⑤** empirical martingale correction (default-OFF `--mart-correct` flag): prior REFUTED (Δ≈5e-3 not ~1e-4 due to residual excess-kurtosis ≈19.7 → over-corrects on tail paths) → inert flag confirmed correct. **A4:** slippage pre-size-skip regression test. **B2/B3 perf** (`quantsys/features/__init__.py`, golden-gated Δ=0): VP rolling min/max precomputed (was per-window, bit-identical — modest ~1.1× real gain: bincount/argsort co-dominate the inner loop) + removed redundant defrag `df.copy()`; `build(normalize=False)` 122col×2970rows = 0 differing cells, test `tests/test_vp_golden.py`. No retrain (bit-identical features). See `STATUS.md` 2026-06-26.

---

## 2026-06-25 — Short-vol arm: structural FHS GJR-GARCH historical backtest + premium validation

Second arm of the vol line (systematic short-vol). Structural backtest over 7 years of hourly candles (`scripts/vol/short_vol_hist_backtest.py`): REAL payoff from candles (tails included), premium = fat-tailed FHS fair-value on GJR-GARCH(1,1) × (1+VRP), VRP swept, fully CAUSAL (90d expanding refit, no lookahead). Result (n=2538 daily 08:00 UTC expiries): break-even VRP = **0% for all structures** → positive mean PnL is the historical VRP harvest (realized 30h < implied). Strangle 8-10% = tail-safe structure (hit 97-98%, maxDD −0.33 BTC, Calmar 33.8). Premium validation (`short_vol_premium_validate.py`): median FHS/mark 1.05×, edge survives the 16% bid haircut. Regime/year decomposition (`short_vol_regime_decomp.py`): edge concentrated in high-vol (2020+2021 = 90% of PnL), always-short, do NOT filter regime. **NOT a gate PASS** (the live n≥20 gate stands). See `STATUS.md` 2026-06-25.

---

## 2026-06-24 — IVS relative-value → KILL net-of-cost

Deribit smile-reversal probe (`scripts/vol/ivs_scout.py` + `scripts/vol/ivs_rv_backtest.py`): real structure (smile residuals revert, autocorr 0.77) BUT net **−2.3/−3.8 vol-pt/leg** (gross +0.01/+0.04 vs round-trip cost 2.3/3.9 → ~50× below the spread). Dead as a price-taker; would only live as a market-maker. **Economic** ceiling, not a data one.

---

## 2026-06-22 — Vol robustness: purged k-fold + per-fold HAR gate + cross-arch diversity

OOS confirmation of the vol line beyond the single split. Purged k-fold QLIKE per arch (`scripts/02b_walkforward_validate.py`, 168h embargo) + HAR fit-per-fold baseline (`scripts/vol/wf_har_baseline.py`, helpers in `quantsys/model/vol_metrics.py`): TCN+Mamba beats HAR by ~14% OOS (ratio 0.863, 4/5 folds), all archs fail only the oldest data-starved fold (structural). Cross-arch error-diversity kill-check (`scripts/vol/step0_xarch_corr.py`): mean ρ_err 0.83 (vs ≈0.995 directional) → diversity is itself an even-moment-specific object. Numbers in `docs/paper/RESULTS_MAP.md` CLAIM 2b/2c.

---

## 2026-06-12 — Vol-1h reorientation + paper "Are price and volume enough?"

Production state becomes the vol-1h line (`models/itransformer/` = vol-1h PASS, `target_type: log_rv`). Added directional econometric baselines as the paper's negative-control (`scripts/research/paper_01_dir_baselines.py`): no coherent skill, ρ sign-flip val→test like the NN → the result is about information, not the model. ~4.7 GB disk cleanup (regenerable 1m dataset/models). Paper documents started: `docs/paper/OUTLINE.md`, `docs/paper/RESULTS_MAP.md`. Per-line script reorg into subfolders (`vol/`, `research/`, `archive/`): map in `scripts/README.md`.

---

## 2026-06-11 — Signed semivariance probe → FAIL

Forward `log(RS⁺/RS⁻)` target (signed jump variation, Patton–Sheppard; judge `scripts/vol/dev_vols_rs_judge.py`): on test NN/HAR-RS MSE 0.9952 (gate ≤0.95), signDA 0.459, and HAR-RS underperforms the constant → **asymmetry is unpredictable for everyone**. Key synthesis: EVEN moments generalize OOS, ODD moments do not. Signed-HD line closed.

---

## 2026-06-10 — VOL-S PASS + 1m→1h pivot KILLED

**VOL-S PASS (B2 positive):** target `features.target_type: log_rv`, the NN beats HAR-RV by **30% in QLIKE on test** (0.257 vs 0.368; naive 0.807), val→test coherent (the anti-correlation is specific to the directional target). Judge `scripts/vol/dev_vols_qlike.py`. 1m cross-resolution check: FAIL on val (NN/HAR 1.013) → the vol edge is SPECIFIC to 1h resolution. **1h pivot KILLED:** 1h breaks the cost wall (|μ|≈43bps ≫ 26bps) but there is NO directional OOS skill, val→test anti-correlation confirmed also at 1h, gate 4/4 failed at both 13 and 23 bps → the "same method, other timeframe" line is closed.

---

## 2026-06-05 — BLOCKER #1 (live parity) resolved

Live path = `LiveCandleBuffer`(50k) → `FeatureAssembler` → `FeatureBuilder.build(fit=False)` (104 canonical) → `LiveEngine._deterministic_predict` → `denormalize_predictions` → `SignalGenerator`. Feature AND signal parity bit-perfect (`tests/test_live_training_parity.py`; replay `scripts/99_replay_live_vs_training.py`: Δfeature=0, Δμ=Δσ=0). Operational residue: WS smoke + paper-trading. (Directional backtest negative OOS — see above.)

---

> **Historical block — directional 1m line (Iterations 1-10).** Kept as record: the 1m directional alpha does not survive OOS (see the 2026-06 entries above), but the pipeline/fix iterations remain the foundation of the shared engine.

## Iteration 10 — Dashboard: definitive rendering fix — category axis (cause: linear axis corrupted on re-render after display:none) (2026-06-24)

### X axis `type:'category'` on `plot-oi` and `plot-payoff` — `scripts/06_dashboard.py`

**The bug**: "OI by strike" (`plot-oi`) and the trades "risk/payoff profile"
(`plot-payoff`) vanished or got crammed into a thin strip when leaving and
re-entering a browser tab (risk↔trades).

**Root cause (browser-diagnosed, 2026-06-24)**: on every re-render after
`display:none→block` (any tab re-entry) Plotly **corrupts the pixel mapping of a
LINEAR numeric X axis** — traces land off-view (x≈−1244px) or the whole band
compresses to ~19px, while the paper-ref shapes stay correct. The **first** render
is always fine, every **subsequent** one is broken; **only the X axis** is affected
(numeric Y renders fine). `_fullLayout` (range/offset/length/margin) is identical
between the good first render and the broken re-render → an SVG-render corruption,
**not** data.

**Failed remedies (tried and rejected)**: `Plotly.react`, `newPlot`,
`purge+newPlot`, node replacement, `Plots.resize`, `relayout` (width/range toggle),
`redraw`, window `resize` event, `autorange` (only turns off-screen into crammed),
removing explicit bar width, scaling x down, hiding via `visibility`/`position`
instead of `display:none`, purge-on-leave.

**The fix (verified across 3 fresh server restarts)**: the X axis of
`plot-oi` and `plot-payoff` switched to **`type:'category'`** (index-based
positioning is immune to the corruption — consistent with `plot-greeks`, already a
category axis, which never had the bug). A new helper `catPos(value, sortedArr)`
places the reference lines (Spot/Max-Pain on OI; Strike/Entry on payoff) at a
**fractional category index**, so they stay proportional between categories. OI:
band strikes become string categories (sorted ascending), thinned ticks
(`dtick≈n/9`), no explicit bar width. Payoff: the linspace prices become categories
(V-curve identical since the grid is regular), the settlement marker snaps to the
nearest category, Y stays numeric autorange.

**History (preceding sub-step, same 2026-06-24)**: the single
`plot(id, traces, layout, cfg)` helper had already unified the 7 renders with a
**size-guard** (`offsetWidth===0` → retry next frame via `requestAnimationFrame`,
~1s cap) + explicit dimensions/`autosize:false`. That was **necessary but NOT
sufficient** context — it did not stop the re-entry corruption. The helper remains
(size-guard + rebuild-on-re-entry); the explicit-width/`autosize:false` detail was
removed and the real fix is the category axis.

**Verification**: `py_compile` OK; 3 fresh server restarts on :8050 with a
hard reload (Ctrl+Shift+R) → `plot-oi` and `plot-payoff` stable on every tab
re-entry, reference lines proportional. The final truth remains the user's browser
with a hard reload.

---

## Iteration 9 — z-score denormalization fix + paper-trading ready (2026-05-23)

### Structural bug: trading layer in raw space vs model in z-score

The global `RobustScaler` scales `target_ret` together with the other features
(scale_factor=0.002707). The model therefore predicts μ, σ, ν in standardized
z-score space. The whole trading layer (`SignalGenerator`, `RiskManager._sl_tp`,
`_size`, config thresholds) instead assumed raw space (log-return fractions).

**Pre-fix consequences**:
- `_sl_tp`: `dist.sigma * price * 1.5` with σ_z≈1 and price=$42k → SL distance $63k
  (300% of the price) → never hit. Every trade closed on `MAX_HOLD` instead
  of real SL/TP.
- `_size`: Kelly `mu/sigma²` computed in z-score → missing `target_scale` factor
  → sizing underestimated by ~370× (then capped by the 0.005 floor).
- Config thresholds on a mixed scale: `max_sigma=2.0` (z-space) was a no-op in raw space.

### Centralized fix

- **`quantsys/utils/__init__.py`**: new property `PipelineState.target_scale`
  (reads `scaler.scale_` for `target_ret`, fallback 1.0) and method
  `denormalize_predictions(mu, sigma) → (mu_raw, sigma_raw)`. Type-preserving
  (float / ndarray / Tensor). Single source of truth.
- **`scripts/03_backtest.py`**: after batch inference calls
  `state.denormalize_predictions(all_mu, all_sigma)`. Safety assert
  `assert all_sigma.max() < 0.05` to detect future regressions.
- **`scripts/04_live_signals.py`**: same call in `_predict()` before the
  return — critical because without the fix paper-trading would operate with
  impossible SL/TP.
- **`quantsys/trading/__init__.py`**: one-shot runtime warning in `_sl_tp`
  if `σ*price*1.5 > 5%*price` (detects missing denormalization).
- **`config/default.yaml` and `config/arch/*.yaml`**: cleaned up the legacy z-space
  thresholds (`prob_threshold: 0.52`, `min_expected_ret: 0.0001`, `max_sigma: 2.0`),
  centralizing in `default.yaml` with raw-space values + comments that
  state the invariant.

### Results

Backtest h=30 test set 7929 candles (heterogeneous ensemble, identical for the 3 archs):

| Metric | Pre-fix | Post-fix | Delta |
|---|---|---|---|
| Sharpe | -255.9 | **+18.71** | +274 |
| Win Rate | 11.03% | **64.29%** | +53 pp |
| Total Return | -15.02% | **+3.67%** | +18.7 pp |
| Max Drawdown | 15.02% | **0.83%** | -14.2 pp |
| Fee/Gross ratio | 1010% | **30.3%** | -980 pp |
| Sharpe CI 95% lower | -48 | **+0.78** | >0 for the first time |
| Circuit breaker | TRIGGERED | False | resolved |

**Stress test passed**: Pessimistic (fee×2, slip×3) Sharpe +7.22, Flash Crash
(fee×1.5, slip×5) Sharpe +12.30.

**5-fold walkforward** confirms stat-sig DA for iTransformer (0.524 ± 0.008,
CI [0.510, 0.531]) and Spearman 0.070 ± 0.010. WHR borderline (0.504–0.517),
to be improved with fix #3 (window_size 240) or real paper-trading.

4/4 explicit promotion thresholds to paper-trading met. See
`MODEL_IMPROVEMENTS.md` for the plan of the next steps.

---

## Iteration 8 — Horizon 15min + Advanced DL features

### Forecast horizon 5 → 15 → 30 minutes
Target evolved progressively: 5 → 15 → 30 minutes. At h=15 (original Iteration 8)
`asymmetry_threshold` had been rescaled from 0.002 to 0.004 to compensate for the larger
amplitude of returns. At h=30 (2026-05-20, part of Iteration 9) the expected
move (~42 bps) exceeds the roundtrip cost (~26 bps) by a margin, making
trading structurally profitable. Optuna search space `forecast_horizon`
updated to [15,30,60].

### Multi-Teacher Distillation — `quantsys/model/distillation.py`, `run_all.py`, `scripts/02_train.py`
All 3 models contribute as teachers with weights proportional to the normalized scoring
(softmax with temperature=2 on a score of 40% loss, 35% spearman, 25% DA). Replaces the selection
of a single teacher: every student receives soft labels weighted across all candidates.
CLI: `--multi-teacher` flag, enabled automatically by `run_all.py --distill`.

### Fractional Differencing (FFD) — `quantsys/features/__init__.py`, `config/default.yaml`
Implementation of Fixed-width Fractional Differencing (López de Prado) on log(close) and log(volume+1).
Generates 2 additional features: `frac_diff_close` and `frac_diff_volume`. d=0.4 configurable
(`features.frac_diff_d`), weights truncated at |w_k| < 1e-5, vectorized convolution.
d=0.0 disables the features (backward compatible).

### Direction-Value Joint Loss — `quantsys/model/__init__.py`, `scripts/02_train.py`
New loss term that penalizes directional errors (sign(mu) != sign(y)) proportionally
to |y|: wrong predictions on large moves cost more. `dv_lambda=0.3` configurable,
0.0 disables. Complementary to asymmetry_penalty (which acts on the NLL).

### Centralized CPU fraction — `config/default.yaml`, all scripts
`hardware.cpu_fraction` in `config/default.yaml` controls the percentage of CPU cores
used by all scripts (default 0.5 = 50%). All 6 scripts (`run_all.py`,
`02_train.py`, `02b`, `02c`, `03_backtest.py`, `04_live_signals.py`) read the value
from the config at startup. Changing the CPU limit no longer requires editing code.

### GPU VRAM limit removed — all scripts
Removed the `torch.cuda.set_per_process_memory_fraction()` call from all scripts.
The model now uses all available GPU VRAM. To limit GPU compute
(not VRAM), use `nvidia-smi -pl <watt>` before training (RTX 2070 Super TDP=215W).

---

## Iteration 7 — Distillation pipeline optimizations

### Critical fix: shuffle-safe soft labels — `scripts/02_train.py`
The teacher's soft labels were indexed sequentially (`sample_idx`) but the
training dataloader uses `shuffle=True`. Soft labels ended up associated with the
wrong samples. Fix: soft labels integrated into the `TensorDataset` so the shuffle
reorders them together with the real data.

### Normalized teacher scoring — `run_all.py`
The `_select_best_teacher()` formula was dominated by val_loss (it contributed 150-200
points vs 0.5-2.5 for spearman). Now each metric is normalized to a 0-1 scale (min-max
across the 3 architectures) and weighted: 40% loss, 35% spearman, 25% DA. Values are taken
at the best val_loss epoch, not the peak over all epochs.

### Natural ensemble output — `quantsys/model/ensemble.py`, `scripts/03_backtest.py`, `scripts/04_live_signals.py`
`EnsembleModel.__call__()` returns `(mu, sigma, nu)` directly in natural space
instead of converting back to log-space with `log(expm1(x))` (unstable for small values).
Removed the double softplus: backtest and live no longer re-apply the conversion.
Added `torch.amp.autocast` in the ensemble forward to halve VRAM on GPU.

### Scale-normalized distillation loss — `quantsys/model/distillation.py`
The fixed weights (1.0 mu + 0.5 sigma + 0.1 nu) did not compensate for the different scales:
MSE(nu)~0.1 dominated, MSE(mu)~1e-10 was irrelevant. Now each component is
divided by the teacher's variance. Weights: 0.5 mu + 0.3 sigma + 0.2 nu.

### Teacher loaded only once — `scripts/02_train.py`
The teacher was loaded 2 times: once to generate soft labels, once for the transfer
of the output heads. It is now kept in memory and reused.

### Stress test with precomputed signals — `scripts/03_backtest.py`
`run_stress_scenario()` no longer re-runs the model's predictions. The signals
`(side, dist)` are saved during the main loop and reused with different
fee/slippage parameters. Eliminates ~400k Python iterations for 2 scenarios.

### Student skip if already distilled — `run_all.py`
`phase_distill()` phase 2c checks each student's `config.json`: if already distilled
from the same teacher, it skips it automatically (unless `--force-download`).

### QUANTSYS_ARCH restored after distillation — `run_all.py`
After `phase_distill()`, the arch-specific paths (ARCH_MODELS_DIR, ARCH_RESULTS_DIR,
MODEL_FILE) are correctly updated for backtest and live.

### Feature count from dataset — `scripts/07_verify_teacher.py`
`n_feat` and `n_dynamic` read from `data/lstm_dataset.npz` instead of being hardcoded
(116, 85). Fallback to the previous values if the dataset does not exist.

### Vectorized rolling_std — `scripts/03_backtest.py`
The rolling std for `SimpleSignalModel` was computed with a Python list comprehension
(one `pd.Series().rolling().std()` per sample). Replaced with a vectorized `np.std`
over the last 20 returns of each window.

### Transfer heads warning MoE/Quantile — `quantsys/model/distillation.py`
`transfer_output_heads()` now emits an explicit warning and returns 0 if the model
uses `loss_type="quantile"` or `n_output_experts > 1` (transfer not supported).

### Cleanup generate_teacher_predictions — `quantsys/model/distillation.py`
Removed the useless `"quantiles"` list allocation (used only for quantile models
but allocated for all).

---

## Iteration 6 — Knowledge Distillation + Heterogeneous Ensemble

### Knowledge Distillation pipeline — `scripts/02_train.py`, `quantsys/model/distillation.py`
New `--distill` pipeline that trains a teacher (iTransformer) and then students (LSTM, TCNMamba)
with transfer of the output-head weights + mixed loss (0.7 real + 0.3 distillation).
Students converge in ~60% of the normal epochs thanks to the transferred calibration.

### Heterogeneous ensemble — `quantsys/model/ensemble.py`
`EnsembleModel.load_heterogeneous()` loads one model per architecture (iTransformer + LSTM +
TCNMamba) instead of N checkpoints of the same one. Backtest and live automatically use the heterogeneous
ensemble when at least 2 architectures have a checkpoint available. Structural diversity
of the errors improves robustness vs a homogeneous ensemble (5x same seed).

### Teacher verification script — `scripts/07_verify_teacher.py`
Analyzes parameters, complexity and backtest metrics of the 3 architectures and recommends
which one to use as teacher. Saves results to `models/teacher_analysis.json`.

### Distillation orchestration — `run_all.py`
New `--distill` + `--teacher` flags for `run_all.py`. Automates: train teacher →
train LSTM student with distillation → train TCNMamba student with distillation.

---

## Iteration 5 — Multiple architectures + optimizations

### iTransformer (QuantiTransformer) — `quantsys/model/__init__.py`
New architecture selectable with `--arch itransformer`. Multi-scale embedding over
three windows (1min T=120, 5min T=24, 15min T=8), feature type embedding (dynamic/structural),
macro context token prepended when available. N pre-norm attention + FFN layers, mean pool →
output heads. Complexity O(F²)=3025 vs O(T²)=14400 for TFT: 4.7× fewer attention operations.

### Arch-specific directories — `run_all.py`, all scripts
`models/lstm/` and `models/itransformer/` for separate checkpoints. `results/lstm/` and
`results/itransformer/` for separate backtest and live signals. Env var `QUANTSYS_ARCH`
propagated to all subprocesses. `load_config(path, arch=)` merges base + arch override.

### Interactive architecture selection — `run_all.py`
If `--arch` is not passed on the CLI, `run_all.py` shows a prompt with the two options
and waits for the user's choice before starting the pipeline.

### 116 dual-stream features — `quantsys/features/__init__.py`
From 55 to 116 features: 85 dynamic (stream A) + 31 structural (stream B). Added
multi-scale VP (short/medium/long), absolute-level ATH/ATL features over 30/90/365d,
slow momentum 7/30/90d, round level, price_vs_ma200m, session position, funding rate.

### HMM multi-restart — `quantsys/macro/regime.py`
`_fit_single` tries `n_restarts=5` consecutive seeds, suppresses warnings during each
attempt with `warnings.catch_warnings()`, returns the model with the maximum log-likelihood.
Early exit if a restart converges before exhausting the iterations. Eliminates the "Model is
not converging" warnings that appeared on short burn-in windows.

### Feature engineering optimizations — `quantsys/features/__init__.py`
- VP `_fill_interp`: numpy ffill (`maximum.accumulate`) instead of a temporary `pd.Series` (×12)
- VP value area: `cumsum + searchsorted` instead of a Python loop over 420k iterations
- `_normalize`: bulk write `df[cols] = X_scaled` instead of a column-by-column loop
- Fix `ChainedAssignmentError` funding rate (pandas 3.x CoW): `inplace` → reassignment
- Fix `PerformanceWarning` VP: bulk `pd.concat` instead of column-by-column insert

## Iteration 4 — Training optimizations

### Flash Attention — `quantsys/model/__init__.py`
`F.scaled_dot_product_attention` in place of the manual attention in `TemporalAttention`.
Enabled automatically on CUDA (fused kernel → ~30% attention speedup).

### DataLoader and eval optimizations — `scripts/02_train.py`
- `prefetch_factor=4`: preloads 4 batches ahead
- `torch.from_numpy()` zero-copy in dataset loading
- `torch.inference_mode()` in `run_eval()` (~5-10% faster than no_grad)
- `non_blocking=True` in `.to(device)` during eval
- Validation every 2 epochs: halves the eval cost on the val dataset

## Iteration 3 — Fix 5-10

### Fix 5 — File logging (`quantsys/utils/__init__.py`)
`setup_logging()` now adds a `FileHandler` with a timestamp in the name
(`logs/quantsys_YYYYMMDD_HHMMSS.log`) in addition to the `StreamHandler` on stdout.
File writing is idempotent (does not duplicate handlers if called multiple times).

### Fix 6 — Unified `PipelineState` (`quantsys/utils/__init__.py`)
New object aggregating into a single `.pkl` the price-feature scalers,
the `MacroNormalizer`, the ordered column list and the model config.
Eliminated the need to load 3-4 separate files at inference.
It is saved by `01_download_data.py` and updated by `02_train.py`.

### Fix 7 — Spearman ρ and ICIR (`scripts/02_train.py`)
Replaced the lone `directional_accuracy` with `prediction_metrics()`, which computes:
- **Spearman ρ**: rank correlation between predicted μ and the real log-return
- **Weighted Hit Rate**: DA weighted by the size of the move
- **Mean IC** and **ICIR**: signal consistency over rolling 50-step windows
The per-epoch log now shows `DA=x.xxx  ρ=+x.xxxx`.

### Fix 8 — Separate LR for MacroEncoder (`scripts/02_train.py`)
The `MacroEncoder` uses `lr = lr_base / 10` for the first epochs, preventing
its noisy gradient from destabilizing the LSTM's price branch.
Implemented with two param groups in `AdamW`. Active only when `has_macro=True`.

### Fix 9 — LSTM-guided Monte Carlo (`quantsys/model/forecast.py`)
New module replacing the random walk with historical volatility.
At each step: the LSTM predicts `(μ_t, σ_t, ν_t)` over the entire batch of paths
in a single forward pass, samples log-returns from the parametric t-Student,
updates the window autoregressively. `σ_eff = √(σ_lstm × σ_garch)` combines
the network's forecast with GARCH volatility clustering.

### Fix 10 — Unit tests (`tests/test_features.py`)
8 test classes covering the most dangerous silent bugs:
- **Stationary log-returns**: mean close to zero, no inf/NaN
- **Correct target**: `target_ret[t] == log_ret[t+1]`
- **VWAP within the H-L band**: cannot be out of range
- **Split without overlap**: `max(t_train) < min(t_val) < min(t_test)`
- **Correct split sizes**: fractions respected ±2%
- **HMM probabilities**: each row sums to 1, no negative values
- **Differentiable NLL**: gradient flows to μ, log_σ², log_ν
- **PipelineState round-trip**: save → load returns identical data

---

## Iteration 2 — Fix 1-4

### Fix 1 — FRED release lag (`quantsys/macro/__init__.py`)
Added `RELEASE_LAG_DAYS` (D=1, W=4, M=35, Q=35 days) and
`SERIES_LAG_OVERRIDE` for specific series. `fetch_all()` shifts the index
of each series before the `ffill`. Merge with `pd.merge_asof(direction="backward")`.

### Fix 2 — Incremental Volume Profile (`scripts/04_live_signals.py`)
`LiveFeatureBuffer` maintains `_vp_bins` (array) and `_vp_contribs` (deque).
Each `push()` updates in O(1) instead of O(N). Full reset every 60 candles.
`np.convolve` replaced with `pd.Series.ewm()` and `rolling(min_periods=1)`.

### Fix 3 — WS state persistence (`scripts/04_live_signals.py`)
`_save_state()` (atomic write) and `_load_state()` with age check (< 5 min).
`warmup()` first tries restoring from disk, then fills the gap with the REST API.
Open position, portfolio and buffer survive WS disconnections.

### Fix 4 — SimpleSignalModel (`scripts/03_backtest.py`)
Rolling statistics precomputed over the entire test set before the loop.
Pandas EWM for fast/slow mean, `rolling(min_periods=3).std()` for volatility.
Dynamic ν: varies with observed volatility (3-12).

---

## Iteration 1 — Initial project

Full pipeline: Binance REST+WS, feature engineering (55 features),
LSTM→GRU→t-Student NLL, Monte Carlo GARCH, Kelly Risk Manager, Backtest,
HMM MacroEncoder, Bloomberg-style React Dashboard.
