🇬🇧 English · [🇮🇹 Italiano](PERF_AUDIT.it.md)

# PERF_AUDIT.md — Diagnostic performance audit

> **Nature of the document.** A fact-finding survey, not an optimization plan. No file under
> `quantsys/` was modified; the probes used live in `scripts/archive/perf_probe/` (untracked,
> disposable). Date: 2026-08-02. Machine: i7-9700K (8 cores / 8 threads, no HT), 15.9 GB RAM,
> RTX 2070 SUPER 8 GB, Python 3.12.10, torch 2.5.1+cu121, pandas 3.0.2, numpy 2.4.3, Windows 11.
>
> **Convention.** Every number is tagged **[M]** = measured in this session, **[S]** = estimated
> by reading the code or extrapolated from a partial measurement, **[D]** = documented elsewhere in the repo and
> not re-verified. Where I have no measurement and am not comfortable deducing one, I write *to be verified*.

---

## 0. Summary in ten lines

The project's compute time is **almost entirely in training**, and training is **not
compute-bound**: at batch 64 the GPU sits at 5-15% utilization and the per-step cost is dominated by
**kernel launching** (CPU/dispatch overhead), not by kernel execution. The direct proof is that
doubling the batch from 32 to 64 costs +6% wall-clock instead of +100% **[M]**. This has a
clear-cut consequence for the proposed levers: those that reduce the *arithmetic work* (channels_last,
Numba, native extensions, Polars) do not touch the bottleneck; the only one that attacks
kernel launching — `torch.compile` — gives **1.56×** on the step, measured, with the `cudagraphs` backend
**[M]**.

Data prep, the natural candidate for Polars, costs **2.2 seconds** on 66k bars **[M]**:
there is nothing to gain, and the prototype shows that three of the seven ported columns change
value, one of them by **7.7%** because of a difference in estimator definition **[M]**.

Two collateral defects emerged that matter more than any optimization: an **import-order
crash** (§7.1) and a **silent degradation of the regime detector** (§7.2).

---

## 1. How the system works today

### 1.1 The four execution paths

**`01_download_data.py` — network → CPU → disk.** In order: `fetch_klines` (Binance REST, ~66
calls of 1000 candles each to cover 2019→today), `fetch_funding_rate` (best-effort, non-blocking),
holdout truncation, writing `raw_candles.parquet`, `FeatureBuilder.build()` (11 steps + Volume
Profile + funding + 3 interactions), scaler fit **on train only** and transform on everything,
writing `features.parquet`, `create_windows` (stride_tricks + materialization), `temporal_split`,
writing `lstm_dataset.npz`, double save of the `PipelineState` (arch-local + canonical).
The network/CPU boundary is clean: all of the download precedes all of the computation. There is no GPU in this path.

**`02_train.py` — disk → CPU → GPU.** `np.load` of the npz (3.26 GB), computation of the adaptive
p0.1/p99.9 clip bounds on `X_train`, pre-clipping of train/val/test, construction of the `TensorDataset`s
(CPU tensors resident in RAM), then the epoch loop: `run_train` (AMP forward → quantile+CE loss →
backward → clip → step every 2 batches) and `run_eval` on val. The `DataLoader` runs with
**`num_workers=0` forced on Windows** (explicit line in `02_train.py`, which overrides the
`num_workers: 6` in the config): collate and pin happen in the main process, synchronously with
the step. The CPU/GPU boundary is per-batch and is crossed ~810 times per epoch.

**`03_backtest.py` — GPU in bulk, then pure CPU.** All predictions are computed **up
front** in batches of 256 (`all_mu`/`all_sigma`/`all_nu`), denormalized once, and the
`predict()` inside the event loop is an O(1) array lookup. The loop over `range(n-1)` is therefore
**entirely CPU/Python**, with no GPU and no I/O. Monte Carlo is not on the critical path.

**`04b_vol_paper.py` — network, and nothing else.** `while True`: one tick, then `time.sleep` until `hh:00:90`.
The tick makes a few Deribit REST calls (chain, mark, index, ticker, possible order + perp hedge)
and **one** model forward. The process wall-clock is ~99.9% `sleep`; active time is
dominated by network latency to Deribit, not by computation. On the VPS inference runs on CPU
(CPU-only torch wheel).

### 1.2 Where the time goes — breakdown

**Test suite [M]:** `438 passed, 1 skipped in 33.9s` (measured twice: 35.6s and 34.0s).
⚠ The README states *"355 passed, 1 skipped, ~30s"*: the count is **stale** by 83 tests.

**`01_download_data.py`, local computation only [M]** (network excluded — I did not time it because
it depends on bandwidth and the Binance rate limit):

| Phase | Time | Notes |
|---|---|---|
| module `import` | ~2.2 s | see §1.3 |
| `FeatureBuilder.build()` | **1.72 s** | of which `_volume_profile` **1.32 s (77%)** |
| `fit_scaler_only` + `_normalize` | 0.52 s | sklearn RobustScaler |
| `create_windows` | **4.31 s** | materializes 3.30 GB |
| `np.savez` of the dataset | **5.82 s** | 3.30 GB → 0.57 GB/s |
| **total local computation** | **≈ 14.6 s** | |

Inside `build()`, every step other than the Volume Profile stays **under 40 ms each**
(`_structural_features` 0.04 s, `_frac_diff` 0.04 s, `_vwap` 0.03 s, `_technicals` 0.03 s,
`_volatility` 0.03 s, the others ≤0.02 s). Data prep is, in practice, the Volume Profile and nothing else.

**`02_train.py` [M]** (from a real run sandboxed via `QUANTSYS_MODELS_ROOT`, timestamped log):

| Phase | Time | How measured |
|---|---|---|
| `import` + config | ~2.2 s | §1.3 |
| `np.load` npz 3.26 GB | 4.15 s | dedicated probe, 0.79 GB/s |
| **clip bounds `np.nanpercentile`** | **36 s** | log timestamps 19:03:42 → 19:04:18 |
| epoch (train + eval) | **18 s** | 10 consecutive epochs, all 18 s |
| ├─ of which train (810 steps) | ~14.9 s | 810 × 18.34 ms **[M]** |
| └─ of which eval on val | ~3 s | difference **[S]** |
| **5 seeds × ~18 epochs** | **≈ 27 min** | consistent with the README **[S]** |

The 36 seconds of clip bounds are **two full epochs** paid once per invocation (not per
seed: the computation precedes the ensemble loop). `np.nanpercentile` does a full per-column sort on
a `(6.2M, 104)` matrix.

**`03_backtest.py` [M/S]:** batch inference of the 5 members on 6,485 samples = **2.0 s [M]**.
The directional event loop costs **30–129 µs/bar [M]** depending on trade density (30 µs with
few trades, 129 µs with 363 trades over 1020 bars — an extreme case): over the whole test split that is
**0.2–0.9 s**. `bootstrap_sharpe_ci` (5000 resamples) = **37 ms [M]**; `mdd_stats` = **2 ms [M]**.

⚠ **The backtest cannot run end-to-end on the current on-disk state**, and this is correct:
the checkpoint in `models/itransformer/` is the **vol** model (`log_rv`), so the denormalized σ
is 1.61–2.92 in log-variance units and the guard `σ ≥ 0.05·√60 = 0.387` fails fast as intended
(`RuntimeError` at `03_backtest.py:512`). I therefore measured the event loop **in isolation**, driving
the production `SignalGenerator`/`RiskManager` classes with synthetic μ/σ, instead of bypassing the guard.

**`04b_vol_paper.py` [M/S]:** single forward (batch 1, `no_grad`, AMP off) = **2.74 ms** on GPU
**[M]**; batch 64 = 4.22 ms **[M]**. On the VPS, CPU-only, it will be slower but remains negligible compared with
an hourly tick **[S]** — I did not measure it on the VPS.

### 1.3 The cost of imports

Measured over 3 runs per row, bare interpreter ≈ 59 ms **[M]**:

| Import | Time |
|---|---|
| `pandas` | 472 ms |
| `pandas` + `sklearn.preprocessing` | 1 404 ms |
| `pandas` + `torch` | 2 202 ms |
| `pandas` + `quantsys.utils` | **2 222 ms** |
| `pandas` + `quantsys.features` | 1 478 ms |

`quantsys.utils` imports torch at module level, so **every script that touches it pays ~2.2 s
before doing anything at all**. The py-spy profile of `FeatureBuilder` confirms it from the opposite side:
`_load_dll_libraries (torch/__init__.py:238)` collects **87 samples**, exactly as many as the hottest
line of the Volume Profile (`__init__.py:347`, 87 samples) — in a benchmark that does not use torch.
Adding up the import machinery (`get_data`, `_path_stat`, `_compile_bytecode`, `realpath`) exceeds
230 samples, i.e. more than the entire Volume Profile.

For `01`/`02` (tens of seconds or minutes) it is noise. For the one-shot judges and for the session
routine, which launch several short scripts in sequence, it is the dominant item.

### 1.4 Training is launch-bound — the measurement

This is the central result of Part 1. Model: `QuantiTransformer`, **675 995 parameters**,
d_model 128, 3 layers, patch_size 5 → `T_eff = 24`, F = 104 tokens (+1 macro). These are tiny kernels
for a 2070 SUPER.

**Batch sweep, pure fwd+bwd on a GPU-resident batch (no DataLoader, no H2D) [M]:**

| batch | ms/step | samples/s | ms/step without `.item()` |
|---|---|---|---|
| 32 | 16.69 | 1 917 | 18.19 |
| 64 (**production**) | 17.74 | 3 607 | 17.43 |
| 128 | 19.40 | 6 597 | 19.58 |
| 256 | 25.96 | 9 861 | 26.91 |
| 512 | 52.03 | 9 841 | 50.46 |
| 1024 | 99.84 | 10 256 | 96.92 |

From 32 to 128 the arithmetic work quadruples and wall-clock grows by **16%**: below batch ~256 the
time is almost independent of the work, i.e. **dominated by launching**. Above 512 the curve becomes
linear — there, and only there, training is compute-bound. `nvidia-smi dmon` agrees: **SM 5-15%** at
small batch, **96-98%** at batch ≥512 **[M]**.

Quantified in a defensible way: at saturation the GPU processes 10 256 samples/s, so 64 samples
"are worth" 6.2 ms of real computation; we spend 17.7. **About 11.5 ms per step (65%) is overhead
that does not scale with the work.**

Two hypotheses I tested that do **not** hold:
- *The per-step `loss.item()` stalls the pipeline.* It costs +0.3 ms at batch 64 **[M]** — irrelevant,
  precisely because, being already launch-bound, the GPU is waiting for the CPU anyway.
- *The DataLoader is the bottleneck.* On its own it costs **1.74 ms/batch** at bs=64 **[M]** (collate+pin+H2D);
  in the full step the delta relative to pure fwd/bwd is ~0.6 ms. It is ~3-9% of the step, not the
  bottleneck — even though `num_workers=0` means that cost sits entirely on the critical path.

I ran the `torch.profiler` profile over 20 steps but **do not report it as a breakdown**: with
`record_shapes` active the profiling overhead inflated total CPU from 0.40 s to 1.80 s and
attributed "self CUDA time" to CPU-only operations (`as_strided`, `select`), producing a GPU
utilization of 476% — an artifact. The three direct measurements above are more reliable and say the same thing.

---

## 2. The levers, one by one

### (a) `torch.compile` — **YES, it is the only one that attacks the real bottleneck**

**Compatibility.** Three checks, all measured:

1. **`spectral_norm` is not on the production path.** In `QuantiTransformer.__init__` the
   `spectral_norm` is applied inside an `if ... and loss_type == "t_student"` branch. Production
   runs `loss_type: quantile`, so the model **has no parametrization at all**: verified with
   `torch.nn.utils.parametrize.is_parametrized` on every submodule → *none* **[M]**. The
   `torch.compile` ↔ `spectral_norm` concern is **out of scope** for the production arch.
   It would remain relevant only for the `t_student` branch and for nhits/tcnmamba, where SN is applied
   unconditionally.
2. **The Mamba scan is not in play**: `tcnmamba` is not on the vol line (its checkpoints were
   deleted with the 06-12 cleanup) and should not be retrained for this. I did not test it — *to be verified*
   if and when a heterogeneous run is reopened. Note: the scan is already vectorized (cumprod/cumsum, not a
   Python loop) and forces float32 internally, so it is a plausible but unverified candidate.
3. **Dynamo traces the whole model**: `graph_count=1`, **`graph_break_count=0`**,
   `op_count=88` **[M]**. No graph breaks to resolve.

**The real blocker is the toolchain, not the code.** The default backend (`inductor`) fails:
`RuntimeError: Cannot find a working triton installation`. **Triton cannot be installed from PyPI on
Windows** (`pip download triton` → `No matching distribution found`) **[M]**. There is a
third-party package `triton-windows`, which I neither installed nor evaluated.

**The `cudagraphs` backend does not require Triton and works:**

| | ms/step | speedup |
|---|---|---|
| eager (production) | 15.30 | — |
| `torch.compile(backend="cudagraphs")` | **9.79** | **1.56×** **[M]** |

Compilation cost: the 12-step warmup goes from 0.52 s to 3.32 s, i.e. **~2.8 s one-off** per
process **[M]** — negligible on a training run lasting minutes, not negligible on a one-shot script.

**Plausible end-to-end gain [S].** The lever only touches the train part of the epoch (~14.9 s out of
18 s). At 1.56× the epoch would drop to ~13.5 s, i.e. 5-seed training from ~27 to **~20 min**: **−26%**.
It does not touch the 36 s of clip bounds nor the 4.15 s of `np.load`. It is a real, measured gain, but of
order "minutes", not "hours".

**Why it works** is exactly what §1.4 says: CUDA Graphs captures the sequence of launches and
re-executes it as a single submission, which is the specific cure for a launch-bound workload. Consistently,
the expected gain at large batch would be much smaller — *I did not measure it at batch 512*.

**Parity risk: high, see §4.**

### (b) `channels_last` — **NO, it does not apply**

`channels_last` is a memory format defined for **4D NCHW** tensors (and 5D NDHWC) and acts by
selecting different cuDNN kernels for **convolutions and spatial normalizations**. In this
codebase:

- `grep` on `quantsys/model/` finds **no** `Conv2d`, `BatchNorm2d`, `MaxPool2d` **[M]**.
- The TCN uses `nn.Conv1d` (3D NCL tensors) — `channels_last` is not defined for 3D; the analogue
  would be `channels_last_1d`, which is not a stable public format in PyTorch.
- The only 4D tensors in the project are the attention `q/k/v`, `(B, n_heads, N, d_head)`, produced
  by `view(...).transpose(1,2)`. This is not an NCHW spatial layout: it is a batch of matrices for
  `scaled_dot_product_attention`, which does not consult the memory format and wants its own layout anyway.
  Marking them `channels_last` would not change the kernel; at most it would add a re-layout copy.

Case closed. There is no tensor on which the format changes anything.

### (c) AMP — **already mapped correctly, nothing to do**

| Where | State | Documented reason |
|---|---|---|
| `02_train.py:268` (`run_train`) | **ON** (`use_amp = tcfg["use_amp"] and cuda`) | training |
| `02b_walkforward_validate.py:310` | ON | training |
| `02c_optuna_search.py:78` | ON | training |
| `02b_walkforward:314,341` (eval) | explicitly **OFF** | deterministic evaluation |
| `EnsembleModel.__call__` (`ensemble.py:355`) | **OFF** (`enabled=False`) | in-place comment: *"avoids NaN (spectral_norm + Mamba scan)"* |
| `crps_t_student` (`model/__init__.py:73`) | forced **OFF** | *"lgamma unstable in float16"* |
| `MambaSSM` scan (`tcn_mamba.py:~204`) | promotion to fp32 | *"cumprod/cumsum sensitive to underflow in FP16"* |

The three OFF sites are disabled for **numerical stability**, each with its rationale written next to it.
I do not propose re-enabling them: it is not an omission, it is a choice. I only observe that ensemble
inference in fp32 is, in light of §1.4, **launch-bound** anyway (2.74 ms for a batch-1 forward
on a 676k-parameter model), so AMP would not gain it much even if it
were safe.

### (d) Polars instead of pandas in `FeatureBuilder` — **NO, on two counts**

**First reason: there is no time to recover.** Full data prep is **2.24 s** on 66k bars
**[M]**, and **59%** of it is `_volume_profile`, which is a **Python loop** over sampled indices with
`np.bincount`/`argsort`/`searchsorted` inside — i.e. exactly what Polars does *not* express.
The operations Polars would accelerate (rolling, groupby-cumsum) add up to ~0.4 s. Even at a uniform 3×
one recovers **~0.27 s** on a path that, with network and I/O, lasts tens of seconds.

**Second reason: it changes the numbers, and not just in the last bit.** I ported 7 columns — **all
seven are in the canonical list of 104**, verified against the npz `feature_names` **[M]**:

| Group | speedup | deviation vs pandas |
|---|---|---|
| `vol_mean_20`, `realized_var_20` | **3.48×** | **bit-identical** (100% of values) |
| `vol_std_20` | (same group) | rel_max **2.9e-12**, ULP_max **24 394**, bit-equal **0.2%** |
| `vwap_20` (rolling sum) | **1.87×** | **bit-identical** |
| `vwap` (groupby cumsum) | (same group) | rel_max 7.7e-16, ULP_max **6**, bit-equal 50.1% |
| `ret_skew_20` | **0.43×** (**slower**) | **rel_max 7.7e-2, |Δ|max 3.4e-1** |

The first two rows are the expected story: re-associated sums and means give identical results
or results within a few ULPs; the rolling standard deviation uses a different accumulation algorithm
(presumably Welford versus two-pass) and diverges at 1e-12 relative — enough to break
bit-perfect parity, not enough to change a decision.

**`ret_skew_20` is the serious case.** This is not rounding: Polars' `rolling_skew` uses the
**biased** estimator (denominator *n*), pandas' `rolling(20).skew()` the **unbiased** Fisher one (*n−1*).
The difference is **systematic and 7.7%** on a feature that the production model receives as
input. A migration done column by column, with green tests (the golden tests check the
*list* of the 104 features and the shapes, not the *values* of each column), would introduce a silent
change to the model's input. And on top of that, on an operation where Polars is **2.3× slower**.

**Third element, found by chance but relevant.** Installing Polars in this environment is not
free: it brings `polars-runtime-32`, i.e. a second Arrow runtime next to `pyarrow`. During
the audit the install/uninstall also perturbed the dependency set (`statsmodels` was
removed and I had to reinstall it at 0.14.6 to bring the suite back to `438 passed`). I uninstalled
Polars at the end of the audit; the environment returned identical to the baseline.

### (e) Numba — **NO, the stated candidates either do not exist or are already vectorized**

The three indicated candidates, checked one by one:

1. **Event loop of `03_backtest.py`.** It is genuinely sequential (`RiskManager` state that depends
   on the previous bar), but it costs **0.2–0.9 s over the whole test split** **[M]**. Even
   an infinite speedup saves less than a second. Moreover it is **nopython-incompatible**:
   it manipulates `Enum`s (`Side`, `CloseReason`), Python dataclasses (`Position`, `Trade`, `DistributionParams`),
   lists of objects, `logging` — rewriting it for Numba would mean rewriting the risk layer in
   scalar form, i.e. touching exactly the code the manifesto wants bit-invariant.
2. **Bootstrap CI 5000 iterations.** **Already fully vectorized**: `rng.choice` generates a
   `(5000, n)` matrix and all statistics are NumPy reductions along `axis=1`, with no
   Python loop. It costs **37 ms** **[M]**. There is nothing to compile.
3. **Delta-hedge of `04b_vol_paper.py`.** It is not a compute loop: `maybe_hedge` is a handful of
   scalar arithmetic per tick, and the tick is **hourly**. The time is in the REST calls to Deribit.
   Numba has nothing to act on here.

The only truly hot loop in the project is the **Volume Profile** one (1.32 s, and the hottest line
in the py-spy profile). It is numerical, nopython-compatible in principle — but it is worth 1.32 seconds
once per dataset regeneration. `mdd_stats` is a real Python loop over 6485 points: **2 ms**.

### (f) Native extension (Rust/PyO3 or C++/pybind11) — **NO, clearly**

The question is whether there exists **a** component that is both heavy enough and isolated enough. Going through
them:

- *Volume Profile* — isolated yes (pure function over 5 arrays, returns 4 arrays), heavy no: **1.32 s**.
- *Backtest event loop* — heavy no (**<1 s**), isolated no (intertwines risk layer, enums, dataclasses).
- *Training* — it is the bulk of the time, but the computation is already in native CUDA kernels: the problem is that
  there are **too many and too small**, and a native extension in Python does not reduce the number of
  launches. This is precisely the case that `torch.compile` covers and an extension does not.
- *Regime detector* — the walk-forward full rebuild is **[D]** stated at ~3 h with `hmm_retrain_days: 90`
  (~9 h at monthly cadence) and is by far the longest computation in the project. I did not run it.
  But the cost lies in the **statsmodels Markov-Switching fit** (EM + optimization), not in the repo's Python
  code: replacing it would mean reimplementing the Hamilton filter **and** ML estimation in
  Rust, i.e. rewriting the most scientifically delicate and best-tested part (bit-parity under
  test). And the practical problem is already solved otherwise: **B7** (`--regime-incremental`) brings the
  refresh down to minutes with bit-parity guaranteed by tests.

**No component justifies a native extension.** The cost — §3 — would be high and the benefit
measurable in seconds.

---

## 3. State of the toolchain

Inventory **[M]** on this machine:

| Component | State |
|---|---|
| Visual Studio / Build Tools | **absent** — no dir in `Program Files*\Microsoft Visual Studio`, no key `HKLM\SOFTWARE\Microsoft\VisualStudio\SxS\VS7`, `vswhere.exe` absent, `cl.exe` not in PATH, winget does not find `Microsoft.VisualStudio.2022.BuildTools` |
| Windows SDK | **absent** (`Windows Kits\10\Include` does not exist) |
| Rust | **absent** — `rustc`, `cargo`, `maturin` not found, no `~/.cargo` |
| Triton (for `torch.compile`/inductor) | **absent and not installable from PyPI on Windows** |
| py-spy | installed during the audit (0.4.2), **left in place**: out-of-process profiler, no runtime conflict |
| polars | installed and then **uninstalled** (duplicate Arrow runtime next to pyarrow) |

**What the Linux VPS would need.** Deduced from `deploy/vps/setup_vps.sh` and the units, not guessed:
`apt-get install -y git python3-venv python3-pip ufw unattended-upgrades curl` — **there is no
`build-essential`, no `gcc`, no Python development headers**. Installation is
`pip install torch --index-url .../cpu` → `pip install -r requirements-vps.txt` →
`pip install -e . --no-deps`, and `pyproject.toml` declares no dependencies. Today the VPS builds
zero native code: it only takes wheels. Introducing a compiled extension would mean **either**
adding a C/Rust toolchain to provisioning (and lengthening `setup_vps.sh`, which is declared
idempotent and one-shot), **or** building and distributing manylinux + win_amd64 wheels for every release.

**What would change for whoever clones the repo.** Today: `pip install -e .` works without any system
toolchain and `pytest tests/` runs in ~34 s on CPU, which is precisely the verifiability claim of the
README ("verifiable right away, without data"). With a native extension that claim lapses: whoever clones on
Windows without Build Tools can no longer install the package, and the project goes from
"just pip install" to "pip install plus a compiler". For a repo published to accompany a CV
this is a concrete reputational cost, not just a technical one — and it is disproportionate to the seconds
at stake.

---

## 4. Parity risk

The invariant to protect is `tests/test_live_training_parity.py` (Δfeature = 0, Δμ = Δσ = 0 between
live and training) plus the golden tests on the 104 features. For each lever that makes sense:

**`torch.compile` — HIGH risk, but boundable.** Concrete mechanisms:

- *Different kernels and reduction order.* With `inductor` the kernels are **generated**, not those of
  cuDNN/cuBLAS: fusions, tiling and accumulation order change, and with them the last bit. With
  `cudagraphs` the kernels remain the eager ones — it is the *submission* that changes, not the math —
  so the risk is much lower, but **AOTAutograd may repartition the forward/backward graph and
  recompute instead of saving activations**, which shifts the order of operations in the backward.
  *To be verified*: I did not compare eager vs compiled gradients.
- *CUDA Graph capture and static shapes.* The graph is captured on a fixed shape. The last batch
  of the epoch is partial (`51882 % 64 = 42`) → re-capture or fallback. Not a correctness problem,
  but one of path determinism.
- *RNG.* Dropout 0.3 and drop_path 0.2 are active in training. PyTorch handles RNG inside graphs
  with philox offsets, but **the sequence of random numbers consumed may differ** from eager: two
  "identical" runs would diverge. *To be verified*.

**The boundary that makes the risk acceptable:** the bit-perfect parity the project
protects is **live ↔ training**, i.e. it concerns the **inference** path (`FeatureBuilder` →
`_deterministic_predict` → denormalization). `torch.compile` applied **to the training loop only**
does not touch that path: it would change the **weights** obtained (a different model, to be re-judged by the
gate), not the equivalence between two inference paths on the same weights. Applying it instead
to inference — `EnsembleModel.__call__`, `04b`, the judges — would break parity in the proper sense and
would require re-verifying `test_live_training_parity.py` with a tolerance, which is exactly what
that test exists to avoid.

⚠ Methodological consequence, not a technical one: a model trained with `torch.compile` **is not
comparable with the incumbent** through the published claim. The rule already written in the
manifesto applies — a lever is judged against a **baseline retrained on the same dataset/scaler**.
`torch.compile` is a *cost* lever, not a *quality* lever: if it changes QLIKE, the gate must be redone.

**Polars — HIGH risk and not boundable.** It breaks parity in two distinct ways: floating-point
re-association (`vol_std_20`, ULP up to 24k) and **estimator difference** (`ret_skew_20`, 7.7%).
The second is not a tolerance problem: it is a different feature. And since `FeatureBuilder` is
shared by training and live, a partial port would create live↔training divergence **if and only if**
the two paths were migrated at different times — i.e. the most likely failure mode.

**Numba, channels_last, native extension — risk not applicable**, because the levers do not
apply. For completeness: if `_vp_single` were ever compiled with Numba, `np.bincount` with `weights`
is not supported in nopython and would have to be rewritten as an accumulation loop, changing the summation order
→ the `vp_*` features would change in the last bit. The comment in the code documents that the current
`bincount` was chosen precisely because it is numerically identical to the previous `np.add.at`.

**AMP** — already off where needed; no change proposed, no new risk.

---

## 5. What does NOT make sense to do, and why

In order of how certain I am:

1. **`channels_last`** — there is no 4D NCHW tensor in the project. No 2D convolution,
   no spatial normalization. The format has nothing to act on.
2. **Numba on the stated candidates** — the bootstrap is already a NumPy `(5000, n)` matrix with no loop;
   the delta-hedge is scalar arithmetic at hourly cadence; the backtest event loop costs less than a
   second and is full of Enums and dataclasses, hence nopython-incompatible without rewriting the risk layer.
3. **Native extension** — no component both exceeds a second of cost *and* is isolated. The longest
   computation in the project (regime walk-forward) lives inside statsmodels, not in the repo, and is already
   solved by B7 in a bit-exact way. The entry cost, on the other hand, is high and falls on three machines
   (this one, the VPS without `build-essential`, and that of whoever clones).
4. **Polars in `FeatureBuilder`** — it would recover ~0.27 s on a 2.24 s data prep, without touching the
   77% that is a Python loop inexpressible in Polars, and it would change the value of at least one of the 104
   features by **7.7%** because of an estimator difference.
5. **Re-enabling AMP where it is off** — the three sites are disabled for documented NaNs (spectral_norm+Mamba,
   lgamma in fp16, cumprod underflow). It is not an omission to fix.
6. **Optimizing data prep in general** — 2.24 s. Any intervention here is noise compared with the
   36 s of clip bounds or the 27 minutes of training.

And one thing that does not make sense to do *in this order*: chasing `torch.compile` before having looked at
the **36 seconds** of `np.nanpercentile` (§6, question 1), which are free to recover and worth more
than two epochs.

---

## 6. Open questions

Flagged, not implemented.

1. **The 36 s of clip bounds.** `np.nanpercentile(X_train.reshape(-1, 104), [0.1, 99.9], axis=0)`
   fully sorts ~6.2M values per column. It is a fixed cost per invocation of `02_train`,
   equal to two epochs. Question: is the exact percentile precision needed, or would an estimate on
   a causal subsample suffice? ⚠ It is not a neutral micro-optimization: the clip bounds enter the
   `PipelineState` and therefore the train↔inference contract — changing them **changes the data** and requires
   a gate. To be treated as an experimental lever, not as a cleanup.
2. **`torch.compile(backend="cudagraphs")` on training only.** 1.56× measured, ~2.8 s of
   compilation, zero graph breaks, no `spectral_norm` involved. The open questions are
   RNG reproducibility under graph capture and AOTAutograd recomputation in the backward
   (§4). If it is to be opened, it must be opened as a pre-registered experiment with a retrained baseline.
3. **The production batch is 64 in a launch-bound regime.** At 1024 the GPU delivers 2.8× the throughput.
   But `batch_size` is not a cost lever: it changes the number of steps, the SGD trajectory and
   the interaction with `gradient_accumulation_steps: 2` — and the config comments that lr and dropout were
   tuned on 1h with a dataset of ~1.7k effective independent samples. **Do not touch it for
   performance reasons.** The legitimate question is a different one: given that the ensemble is 5
   independent seeds and the GPU is at 5-15%, could **several seeds be trained in parallel in the same
   process** instead of sequentially? It would be a launch-bound gain (the launches overlap)
   without touching any member's hyperparameters. To be verified against the 8 GB of VRAM.
4. **`create_windows` with `window_stride: 1` materializes 3.30 GB** for 66k bars, and the pipeline
   writes it to disk, reads it back, and makes an in-RAM copy of it with `clamp` (peak ~6.6 GB out of 15.9 GB).
   The expansion factor is 120× (each bar appears in 120 windows). Question: is there a reason to
   materialize, instead of generating the windows with a `Dataset` that indexes the `(66k, 104)`
   matrix at zero cost? It would change the order of nothing — the windows are views — but it would touch
   the npz format, which is the contract between `01` and `02`/judges.

---

## 7. Collateral observations (found, not fixed)

> **Update 2026-08-02 (same day):** the two serious defects in this section (§7.1, §7.2)
> have been **fixed**, with regression tests; §7.3 (stale doc) realigned. The subsections remain
> in their original diagnostic form — they describe the defect *as it was* — with a closing note at the
> top of each. The minor inefficiencies in §7.4 are deliberately **not** touched.

### 7.1 ⚠ Import-order crash: `pyarrow` must initialize before torch+sklearn

> ✅ **RESOLVED 2026-08-02** — `import pyarrow` anchored in `quantsys/__init__.py` (best-effort, it does not
> become a declared dependency). Regression test `tests/test_import_order.py`, 4 tests in a
> subprocess: the crash is an access violation, not a Python exception, so it cannot be caught
> in-process with `pytest.raises` — the **exit code** is checked.

**Reproducible [M]**, exit code 139 (access violation in `pyarrow/dataset.py` while loading the module):

| Order | Outcome |
|---|---|
| `import pandas` → `import torch, sklearn.preprocessing` → `read_parquet` | **OK** |
| `import torch, sklearn.preprocessing` → `import pandas` → `read_parquet` | **SEGFAULT** |
| as above, but with an explicit `import pyarrow.dataset` first | **SEGFAULT** |
| `torch` only → `read_parquet` | OK |
| `sklearn` only → `read_parquet` | OK |

**Both** torch and sklearn are needed before pyarrow for the crash to happen (classic conflict between
OpenMP runtimes: torch ships `libiomp5md.dll`, scikit-learn/scipy their own).

**Why production does not see it:** all numbered scripts import `pandas` (line 30 in
`03_backtest.py`) **before** `torch` (line 31) and before `quantsys.*` (line 37). The invariant
holds **by accident of import order**, not by a rule.

**Why it is a risk:** `quantsys.utils` imports torch at module level, and the "new
script" checklist prescribes `load_config` from `quantsys.utils` without saying anything
about ordering. A new script written in the natural way (project imports first, then pandas)
crashes with an access violation and no Python traceback. I ran into it while writing a probe for
this audit. I did not fix it (constraint: no changes to `quantsys/`); if one wanted to make it
structural, the place is an eager `import pyarrow` at the top of `quantsys/utils/__init__.py`, or a
line in the checklist.

### 7.2 ⚠ The regime fit degrades silently instead of failing

> ✅ **RESOLVED 2026-08-02** — `RuntimeError` on zero successful fits (cannot be disabled) + configurable
> abort on `max_fit_failure_ratio` (default 0.5) + diagnostics in `last_fit_diagnostics`;
> guard mirrored in `continue_walkforward`. Also added the missing log on the
> `_fit_single → None` branch, which was previously completely silent. Regression test
> `tests/test_regime_fit_guard.py` (8 tests); B7 bit-parity verified unchanged.

In `quantsys/macro/regime.py:651-653`, the per-timestep Markov-Switching fit is inside
`except Exception as e: log.warning(...)`. With `statsmodels` missing I observed **one warning for
each t** and the walk-forward continuing to the end: `current_params` stays `None`, `probs_all[t]`
is never written, and one gets a result **devoid of informational content without anything
failing**. Only `continue_walkforward` (the incremental B7 path) fails fast downstream, with a correct
message ("a full rebuild is needed").

I discovered it because during the audit `statsmodels` was removed from the environment by one of my
pip operations (later reinstalled at 0.14.6; the suite returned to `438 passed, 1 skipped`). The fact
that the symptom shows up as "6 errors in `test_regime_incremental.py`" and not as an explicit failure
of the rebuild is the part I am flagging: a full rebuild launched under those conditions would have
produced a degraded `regime_probs.parquet`, and the degradation would have been visible only by reading
the warnings. A failed-fit counter with an abort threshold would be consistent with the rest of the project's
fail-fast guards — but it is a design decision, not an oversight to fix on my own initiative.

### 7.3 Stale doc: the test count in the README

> ✅ **RESOLVED 2026-08-02** — README realigned to **450 passed, 1 skipped, ~45 s** (438 measured
> at the start of the audit + 12 new tests from the §7.1-7.2 fixes). ⚠ The suite went from ~34 s to ~45 s:
> the 4 import-order tests run in a **subprocess** and each pays ~2.2 s of `import torch`.
> That is the price of testing an invariant that manifests only as a process crash.

The README stated **355 passed, 1 skipped, ~30s** in two places (the "Where to start" and
"Reproducibility" sections). The actual value at the start of the audit was **438 passed, 1 skipped, ~34 s** **[M]**.
It is a claim an outside reader verifies in thirty seconds, so it was worth realigning.

### 7.4 Minor inefficiencies, all below the relevance threshold

Listed for completeness, with the reason why they are **not** worth touching:

- `create_windows` evaluates `np.isnan(wins).any(axis=(1,2))` on the **expanded view** (3.3 GB, each
  bar re-read 120 times) when the NaN mask can be computed on the `(66k, 104)` matrix before
  expansion. It costs a fraction of the 4.31 s — but see §6.4, the real point is materialization.
- ~~`02_train.py` does `X_tr.clamp(...)`, creating a **full copy** of the train/val/test tensors
  (~3.3 GB transient out of 15.9 GB of RAM). An in-place `clamp_` would avoid the peak.~~ **APPLIED on
  2026-08-02** (`f36b406`, −2.59 GB of peak). **Completed on 2026-08-05:** the other half of the same
  peak was `astype(np.float32)` in `to_t()`, which copied every npz member that was already float32 — now
  `astype(np.float32, copy=False)`, **−2.42 GiB**, bit-identical. ⚠ The two changes are safe only
  **together and in this order of reasoning**: `copy=False` returns the same ndarray as the npz
  member, so it is the in-place `clamp_` that writes over it — the invariant that makes the combination correct is
  that `NpzFile.__getitem__` materializes a fresh array on every access, and it is pinned down by
  `tests/test_npz_load_aliasing.py` (7 tests) instead of being assumed.
- `_vp_single` accumulates into **Python dicts** keyed by integer (`poc_dist_sampled[i] = ...`) and then
  converts them back into arrays with `np.array(sorted(dict.keys()))`. A pre-allocated array + mask
  would avoid dicts and sorting. Worth a fraction of 1.32 s.
- `FeatureBuilder` emits `PerformanceWarning: DataFrame is highly fragmented` at 6 points
  (`_funding_features`, the 3 final interactions). It is cosmetic: defragmentation happens anyway with
  `df.copy()` before normalization, and the B3 comment documents that the intermediate copy was
  removed on purpose because it was redundant.

---

## 8. What I did not do

- **I did not run a full 5-seed training**: the 27 min are extrapolated from 10 real epochs
  measured at 18 s each plus the 42 s of startup, not timed end-to-end.
- **I did not run the full rebuild of the regime detector** (~3 h stated): the cost is **[D]**, taken
  from the comment in `config/default.yaml`, not verified.
- **I did not test `torch.compile` on nhits/tcnmamba** (checkpoints absent since 06-12) nor on the
  `t_student` branch, where `spectral_norm` is applied: there the compatibility question remains open.
- **I did not verify RNG reproducibility or gradients** under `cudagraphs` (§4): the 1.56×
  is a speed measurement, not a certificate of equivalence.
- **I did not measure inference on the VPS** (CPU): the 2.74 ms forward is on this machine's GPU.
- **I did not evaluate `triton-windows`** (third-party package) as a way to enable `inductor`.
- **I did not measure the network download** of `01_download_data.py`, which depends on bandwidth and rate limit.
- **I did not touch** `tests/test_live_training_parity.py`, the golden tests on the 104 features, the
  fail-fast guards of `THEORY.md` §12.5 (the backtest σ guard actually fired during the audit and I
  let it fire) nor the `PipelineState` contract.

**Repo state at the end of the audit (diagnostic phase):** `git status` showed only
`?? scripts/archive/perf_probe/`. No file under `quantsys/` modified; suite at
`438 passed, 1 skipped` as at the start.

---

## 9. What was done AFTER the audit (2026-08-02, same day)

The audit had a read-only scope. On explicit instruction, only the
interventions with **zero seconds gained and zero numerical risk** were then implemented — those that remove ways of
going wrong silently, not those that make the code faster:

| Intervention | File | Test | Numerical impact |
|---|---|---|---|
| `import pyarrow` anchored at the package root | `quantsys/__init__.py` | `tests/test_import_order.py` (4) | **none** |
| Anti-degradation guard for the regime walk-forward | `quantsys/macro/regime.py` | `tests/test_regime_fit_guard.py` (8) | **none** on the success path (B7 bit-parity green) |
| Test count realignment | `README.md` | — | — |

Suite: **450 passed, 1 skipped** (~45 s). Docs updated: `README.md`, `THEORY.md` §12.5 (safety
net list), `START.md` §1.2 (Windows note + quick diagnosis of exit 139), `CHANGELOG.md`,
`STATUS.md`.

## 10. Lever A — clip bounds on distinct bars: TESTED and NOT ADOPTED (2026-08-02)

**Outcome: no-go.** Recorded here because a measured negative outcome is worth as much as a positive one.

The lever looked like the best in gain/complexity ratio (31 s → 0.16 s, **222×**). The
correctness test — *which of the two estimators describes the right population* — produced three results,
the first of which invalidated the premise of the naive implementation.

**A) `X_train` is NOT contiguous.** `create_windows` discards windows containing NaNs, so the
tensor is made of **4 blocks** separated by 3 discontinuities (at j = 2933, 17520, 35895 on the current
dataset). The obvious reconstruction — "bar j is `X[j,0,:]`" — is **silently wrong**: the mechanism
check caught it (reconstruction ≠ expanded view, |Δ| = 0.39 where it should have been 0).
Handling the blocks: 52,358 distinct bars, Σ multiplicities = 6,225,840 = `n_tr × W`, reconstruction
**bit-identical**. Hence the structural fact: **the expanded view contains no extra information**,
it is the same population with the edges under-weighted. And the edges are 8, not 2 → **952 bars (1.82%)
under-weighted, 0.92% weight deficit**: the artifact is ~4× the initial estimate.

**B) The difference between the two estimators is below the noise of the estimator itself.** Bootstrap over
bars (B=300, the true sampling unit — the windows are 120 shifted copies of the same history):
median z **0.008**, p90 0.263. Only **9 bounds out of 208** differ by more than 1 bootstrap SD, 4 by more than 2.
For **95.7%** the two estimators are statistically indistinguishable.

**C) Negligible downstream impact:** 0.145% of cells clipped differently, median |Δ|
**0.0008 IQR**.

**Which one is more correct, then?** Conceptually the one on **distinct bars**: the multiplicity
weighting is an artifact of the windowing procedure — it depends on stride, window size *and on where
the NaN discards happened to fall* — and bears no relation to the data-generating process. The clip
bound should describe the feature's marginal distribution over time. The current estimator
is an approximation of it with edge bias.

**Why not, all the same.** ① The payoff is 31 s out of 27 min = **1.9%**. ② The correct implementation
requires detecting the block structure, which changes at every dataset rebuild: that is exactly the
class of silent bug that §7.1-7.2 have just removed. ③ It touches the `PipelineState`, so it
requires a pre-registered gate whose cost far exceeds the payoff.

**The good version of the idea, should it ever be needed:** compute the bounds in `01_download_data.py` from
`df_feat`, where the bar-level matrix already exists and there is nothing to reconstruct. Clean and
exactly correct — but it changes the `01`↔`02` contract and changes the numbers anyway.

⚠ **The result that holds regardless of the decision:** the clip bounds are estimated from ~52k
effective observations, **not from 6.2M**. The bootstrap SD is 0.008 (p0.1) and **0.036 (p99.9)** where
the median feature IQR is 1.004 — the upper bound carries ~3.6% of IQR of sampling noise.
Computing on the 6.2M gives **spurious precision**: the 120× redundancy adds no information.
Probe: `scripts/archive/perf_probe/test_clip_bounds_correctness.py`.

⚠ Checked incidentally and **falsified**: `X_train` contains no NaNs (0 out of 647M), but replacing
`np.nanpercentile` with `np.percentile` **does not help** — it is **0.92×, slower**. The hypothesis "the cost is
NaN handling" is wrong; the cost is sorting 647M redundant cells.
