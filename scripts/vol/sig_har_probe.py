# SignatureHAR — pre-registered probe (STATUS.md, 2026-08-25 gate).
# This file implements the gate's EX-ANTE step: CONDITION ③, the analytic
# redundancy check of the order-2 term. On the lead-lag path the level-2
# antisymmetric term (Lévy area) equals the window's quadratic variation,
# which HAR-C already carries as `xc_h`: if redundancy holds, the order-2 arm
# closes WITHOUT estimating anything.
# READ-ONLY: reads `data/raw_candles.parquet` and the split npz, writes
# nothing. `scripts/vol/dev_vols_qlike.py` is NOT touched (it judges open
# pre-registered forward samples).
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, interval_minutes_from_cfg, dataset_npz_path  # noqa: E402
from quantsys.model.vol_metrics import (build_har_cj_frame, har_c_fold_qlike,  # noqa: E402
                                        qlike, qlike_series, diebold_mariano,
                                        HAR_C_COLS, EPS)

setup_logging()
log = logging.getLogger("quantsys.script.sig_har_probe")


# Lévy area of the LEAD-LAG path built from a return sequence, computed
# NUMERICALLY over segments (not assumed): for a piecewise-linear 2-D path the
# two level-2 iterated integrals are S12 = Σ_k [A_k·b_k + a_k·b_k/2] and
# S21 = Σ_k [B_k·a_k + a_k·b_k/2], with A/B the partial sums of increments;
# the area is (S12 − S21)/2. The lead-lag path alternates a "lead" step
# (a=r_k, b=0) and a "lag" step (a=0, b=r_k), so the mixed a_k·b_k terms vanish
# segment by segment. Returns 2·area, i.e. the pre-registration's
# quadratic-variation term. Vectorised over the N windows; the loop runs over
# the W bars, not over the data.
def leadlag_qv_term(rw: np.ndarray) -> np.ndarray:
    n, w = rw.shape
    s12 = np.zeros(n)
    s21 = np.zeros(n)
    a = np.zeros(n)   # lead partial sum
    b = np.zeros(n)   # lag partial sum
    for k in range(w):
        r = rw[:, k]
        # lead step — increment (r, 0): contributes to S21 only, via B
        s21 += b * r
        a += r
        # lag step — increment (0, r): contributes to S12 only, via A
        s12 += a * r
        b += r
    return s12 - s21          # = 2 · Lévy area


# (N, W) matrix of CAUSAL return windows ending at t. Row i = [r_{i−W+1}, …,
# r_i]; the first W−1 rows stay NaN and are dropped downstream.
def rolling_windows(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full((n, w), np.nan)
    for k in range(w):
        lag = w - 1 - k
        out[lag:, k] = x[:n - lag] if lag > 0 else x
    return out


# R² of the candidate column regressed on the HAR-C regressors (intercept
# included). This is the redundancy measure that matters for an OLS: how much
# of the new column already lies in the incumbent design's span.
def r2_on_design(y: np.ndarray, x: np.ndarray) -> float:
    xd = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
    resid = y - xd @ beta
    return float(1.0 - resid.var() / y.var())


# FULL SIGNATURE up to level 3 of the TIME-AUGMENTED 2-D path P = {(k/W, x_k)},
# via Chen's identity segment by segment. For a linear segment of increment Δ the
# signature is exp(Δ) = 1 + Δ + Δ⊗Δ/2 + Δ⊗Δ⊗Δ/6, and concatenation is the
# truncated tensor product (formulas above). ⚠ Update ORDER is binding: S³ uses
# the OLD S²/S¹, S² the old S¹ — updating S¹ first would silently produce a wrong
# signature. Component 0 = time, component 1 = return. Vectorised over N windows.
def signature_depth3(rw: np.ndarray) -> tuple:
    n, w = rw.shape
    s1 = np.zeros((n, 2))
    s2 = np.zeros((n, 2, 2))
    s3 = np.zeros((n, 2, 2, 2))
    dt = 1.0 / w
    for k in range(w):
        e1 = np.stack([np.full(n, dt), rw[:, k]], axis=1)          # (n,2)
        e2 = np.einsum("ni,nj->nij", e1, e1) / 2.0
        e3 = np.einsum("nij,nk->nijk", e2, e1) / 3.0               # = Δ⊗Δ⊗Δ/6
        s3 = (s3 + np.einsum("nij,nk->nijk", s2, e1)
                 + np.einsum("ni,njk->nijk", s1, e2) + e3)
        s2 = s2 + np.einsum("ni,nj->nij", s1, e1) + e2
        s1 = s1 + e1
    return s1, s2, s3


# LOG-SIGNATURE in tensor coordinates — log(1+X) = X − X⊗X/2 + X⊗X⊗X/3 truncated
# at level 3, X = S without its scalar part. Then extraction of the coefficients
# on the d=2, depth-3 Hall basis {e1, e2, [e1,e2], [e1,[e1,e2]], [e2,[e1,e2]]}:
# expanding the brackets gives c([e1,e2]) = L¹², c([e1,[e1,e2]]) = L¹¹² and
# c([e2,[e1,e2]]) = −L¹²². These are the pre-registration's 5 independent terms
# (shuffle relations are already removed by the logarithm — no hand-picking).
def logsig_depth3(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> dict:
    l2 = s2 - np.einsum("ni,nj->nij", s1, s1) / 2.0
    l3 = (s3
          - (np.einsum("nij,nk->nijk", s2, s1) + np.einsum("ni,njk->nijk", s1, s2)) / 2.0
          + np.einsum("ni,nj,nk->nijk", s1, s1, s1) / 3.0)
    return {
        "sig_t":    s1[:, 0],          # level 1, time
        "sig_x":    s1[:, 1],          # level 1, return
        "sig_area": l2[:, 0, 1],       # Lévy area
        "sig_ttx":  l3[:, 0, 0, 1],    # [e1,[e1,e2]]
        "sig_txx": -l3[:, 0, 1, 1],    # [e2,[e1,e2]]
        "_l2_anti": l2[:, 0, 1] + l2[:, 1, 0],   # must be ≈0
    }


# the 6 columns of the FROZEN construction: the 5 log-signature terms of the
# augmented path + the lead-lag path's quadratic-variation term.
SIG_COLS = ["sig_t", "sig_x", "sig_area", "sig_ttx", "sig_txx", "sig_qv"]


# closed-form OLS, fit on TRAIN timestamps only and evaluation on held-out rows,
# QLIKE on RV levels — mechanics IDENTICAL to `har_c_fold_qlike`, of which this is
# the arbitrary-regressor generalisation. Also returns the unaggregated loss
# series, needed by the paired DM.
def fit_eval(tr: pd.DataFrame, ev: pd.DataFrame, cols: list) -> dict:
    xtr = np.column_stack([np.ones(len(tr)), tr[cols].values])
    beta, *_ = np.linalg.lstsq(xtr, tr["y"].values, rcond=None)
    xev = np.column_stack([np.ones(len(ev)), ev[cols].values])
    rv_true = np.exp(ev["y"].values)
    rv_pred = np.exp(xev @ beta)
    # RANK alongside the condition number: with a column collinear to the
    # intercept the design is rank-deficient and `cond` blows up, yet the fit
    # stays well defined (minimum norm). Rank says HOW MANY regressors are
    # actually identified, which is the decision-relevant fact.
    return {"qlike": qlike(rv_true, rv_pred), "loss": qlike_series(rv_true, rv_pred),
            "cond": float(np.linalg.cond(xtr)), "rank": int(np.linalg.matrix_rank(xtr)),
            "ncol": int(xtr.shape[1]), "beta": beta}


# PRE-REGISTERED thresholds (STATUS.md 2026-08-25) — hardcoded BECAUSE they are
# pre-registered: a CLI parameter here would be the door goalpost-moving walks
# back in through. `W_FROZEN` is the frozen window; materiality is DERIVED from
# the published band (a 1 percentage-point move), not chosen.
W_FROZEN = 24
QLIKE_HAR_C_PUBLISHED = 0.33698      # val, C3 2026-07-31 (THEORY.md §12.2)
DELTA_MATERIAL = -0.004290           # materiality threshold
QLIKE_SIG_MAX = 0.33269              # = 0.33698 − 0.004290
P_MAX = 0.05
ORACLE_SEED = 42


# POSITIVE CONTROL (pre-reg condition ③) — oracle column built as `y + λ·sd(y)·ε`
# and CALIBRATED by bisection on λ so as to produce exactly the materiality
# improvement. It measures the INSTRUMENT's resolution, not the candidate: if the
# OLS+QLIKE chain cannot register an effect of that size, a candidate FAIL is
# uninterpretable and the pre-declared verdict is NO CONCLUSION (precedent: B1
# stage 1). Noise is drawn ONCE with a fixed seed and stays identical across
# iterations: bisection moves λ, not the sample. Monotonicity holds by
# construction — λ→0 makes the column the target itself (QLIKE→0), λ→∞ makes it
# pure noise (QLIKE→QLIKE_HAR-C) — so the objective changes sign at the ends.
def positive_control(tr, ev, frame, q_target):
    rng = np.random.default_rng(ORACLE_SEED)
    noise = pd.Series(rng.standard_normal(len(frame)), index=frame.index)
    sd_y = float(frame["y"].std())

    def run(lam):
        col = frame["y"] + lam * sd_y * noise
        tr2 = tr.assign(oracle=col.loc[tr.index].values)
        ev2 = ev.assign(oracle=col.loc[ev.index].values)
        return fit_eval(tr2, ev2, HAR_C_COLS + ["oracle"])

    lo, hi = 1e-3, 1e4                     # low λ = strong oracle
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))      # log-scale bisection
        if run(mid)["qlike"] < q_target:
            lo = mid                       # too strong → more noise
        else:
            hi = mid
    lam = float(np.sqrt(lo * hi))
    res = run(lam)
    res["lambda"] = lam
    return res


# val evaluation of the pre-registered gate. Order imposed by STATUS.md:
# (a) baseline reproduction, (b) positive control, (c) candidate. The test split
# is NOT reachable from here: it is one-shot and runs only after the val gate
# passes, as a decision brought to the user.
def run_gate(raw, lr, har_cj, d, h):
    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    if split != "val":
        raise RuntimeError(
            f"QUANTSYS_VOLS_SPLIT={split}: la sonda valuta SOLO val. Il test è one-shot e si "
            f"esegue a gate val superato, come decisione. / the probe evaluates val ONLY; "
            f"test is one-shot, after the val gate passes.")

    rw = rolling_windows(lr, W_FROZEN)
    s1, s2, s3 = signature_depth3(rw)
    cols = logsig_depth3(s1, s2, s3)
    cols["sig_qv"] = leadlag_qv_term(rw)

    # two construction checks before any estimation. (a) level-2 antisymmetry of
    # the logarithm: `L²[0,1] + L²[1,0]` must vanish. (b) the level-1 return term
    # MUST equal the path's total increment, i.e. the sum of returns in the window.
    fin = np.isfinite(cols["sig_x"])
    anti = float(np.nanmax(np.abs(cols["_l2_anti"][fin])))
    dx = np.nansum(rw, axis=1)
    scale_x = float(np.nanmax(np.abs(dx[fin])))
    max_dx = float(np.nanmax(np.abs(cols["sig_x"][fin] - dx[fin])))
    log.info(f"[costruzione] antisimmetria log-sig livello 2: max |L2[0,1]+L2[1,0]| = {anti:.3e}")
    log.info(f"[costruzione] livello 1 x vs somma rendimenti: max scarto = {max_dx:.3e} "
             f"(scala {scale_x:.3e})")
    log.info(f"[costruzione] livello 1 t (tempo): min={np.nanmin(cols['sig_t'][fin]):.6f} "
             f"max={np.nanmax(cols['sig_t'][fin]):.6f}")

    sig = pd.DataFrame({c: cols[c] for c in SIG_COLS})
    sig.index = pd.to_datetime(raw["open_time"]).dt.tz_localize(None)
    frame = har_cj.join(sig, how="left").dropna()
    # the HAR-C frame starts 30 days in, the signature after W bars: no row may be
    # lost, otherwise the sample is no longer the judge's and the two numbers are
    # not comparable.
    assert len(frame) == len(har_cj), f"campione alterato dal join: {len(frame)} vs {len(har_cj)}"

    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)
    t_eval = pd.to_datetime(d[f"t_{split}"]).tz_localize(None)
    tr = frame.loc[frame.index.intersection(t_train)]
    ev = frame.loc[frame.index.intersection(t_eval)]
    log.info(f"righe: train {len(tr)}/{len(t_train)} · {split} {len(ev)}/{len(t_eval)}")

    # standardisation of the 6 columns on TRAIN-ONLY statistics. It is a linear
    # reparametrisation: it leaves the OLS fit unchanged in exact arithmetic and
    # only helps conditioning (level-3 terms live at ~1e-9, the time component at
    # 1). Degenerate columns (zero standard deviation) are left intact and
    # DECLARED, not removed: the construction is frozen.
    # ⚠ Degeneracy must be detected in RELATIVE scale. The augmented path's time
    # component is identically 1 on every window, yet its empirical `sd` is not
    # zero: it is rounding noise at ~1e-16. An absolute test (`sd < 1e-300`) misses
    # it, and dividing by 1e-16 PROMOTES that noise to a unit-variance regressor —
    # a garbage column absent from the frozen construction, which OLS would still
    # assign a coefficient to. Degenerate columns are left RAW and declared: they
    # stay collinear with the intercept and `lstsq` (minimum-norm solution,
    # machine-precision cutoff) annihilates them cleanly.
    mu = tr[SIG_COLS].mean()
    sd = tr[SIG_COLS].std()
    degenerate = [c for c in SIG_COLS
                  if not np.isfinite(sd[c]) or sd[c] <= 1e-12 * max(1.0, abs(float(mu[c])))]
    for c in degenerate:
        log.warning(f"[costruzione] colonna DEGENERE sul train: {c} (sd={sd[c]:.3e}, "
                    f"media={float(mu[c]):.6f}) — collineare con l'intercetta, tenuta RAW "
                    f"perché la costruzione è congelata")
        mu[c], sd[c] = 0.0, 1.0
    tr = tr.assign(**{c: (tr[c] - mu[c]) / sd[c] for c in SIG_COLS})
    ev = ev.assign(**{c: (ev[c] - mu[c]) / sd[c] for c in SIG_COLS})

    # ── (a) baseline reproduction ────────────────────────────────────────────────
    ref = har_c_fold_qlike(har_cj, t_train, t_eval)
    base = fit_eval(tr, ev, HAR_C_COLS)
    log.info(f"HAR-C (funzione di produzione)  QLIKE = {ref['qlike_har_c']:.6f}  n={ref['n_har_c']}")
    log.info(f"HAR-C (catena della sonda)      QLIKE = {base['qlike']:.6f}  cond={base['cond']:.1f}  rango={base['rank']}/{base['ncol']}")
    log.info(f"HAR-C pubblicato (THEORY §12.2) QLIKE = {QLIKE_HAR_C_PUBLISHED}")
    delta_chain = abs(base["qlike"] - ref["qlike_har_c"])
    assert delta_chain < 1e-12, f"la sonda non riproduce har_c_fold_qlike: scarto {delta_chain:.3e}"

    # ── (b) positive control ─────────────────────────────────────────────────────
    q_target = base["qlike"] + DELTA_MATERIAL
    pc = positive_control(tr, ev, frame, q_target)
    dm_pc = diebold_mariano(pc["loss"], base["loss"], h)
    pc_delta = pc["qlike"] - base["qlike"]
    pc_fires = bool(abs(pc_delta - DELTA_MATERIAL) < 1e-5 and dm_pc["p_value"] <= P_MAX)
    log.info(f"[controllo positivo] λ* = {pc['lambda']:.4f} · QLIKE = {pc['qlike']:.6f} · "
             f"Δ = {pc_delta:+.6f} (bersaglio {DELTA_MATERIAL:+.6f})")
    log.info(f"[controllo positivo] DM = {dm_pc['dm_hln']:.4f} · p = {dm_pc['p_value']:.6f} · "
             f"HAC lag {dm_pc['hac_lag']} · n_eff = {dm_pc['n_eff']:.1f} · "
             f"{'SI ACCENDE' if pc_fires else 'SPENTO'}")

    # ── (c) candidate ────────────────────────────────────────────────────────────
    cand = fit_eval(tr, ev, HAR_C_COLS + SIG_COLS)
    dm = diebold_mariano(cand["loss"], base["loss"], h)
    delta = cand["qlike"] - base["qlike"]
    log.info(f"SignatureHAR (3+6 regressori)   QLIKE = {cand['qlike']:.6f}  cond={cand['cond']:.3e}  rango={cand['rank']}/{cand['ncol']}")
    log.info(f"Δ QLIKE = {delta:+.6f}  (soglia pre-registrata {DELTA_MATERIAL:+.6f})")
    log.info(f"DM appaiato = {dm['dm_hln']:.4f} · p = {dm['p_value']:.6f} · HAC lag {dm['hac_lag']} · "
             f"n_eff = {dm['n_eff']:.1f}")

    # ── (d) per-arm decomposition ────────────────────────────────────────────────
    # DESCRIPTIVE, not gating — it lives outside `conditions`/`verdict`, like the
    # judge's baseline blocks. The pre-registration's FAIL branch requires writing
    # the MECHANISM, not just the outcome, and the mechanism here is: how much of
    # the Δ comes from the EVEN term (quadratic variation, i.e. RV) and how much
    # from the ORDERING terms (Lévy area and level 3), the odd family arm ② makes
    # its prediction about. `sig_x` is kept separate because it is the window's
    # total return — odd, but level 1, hence not an ordering term.
    # ⚠ p-values are NOT multiplicity-corrected and are computed on a split the
    # gate has already consumed: they generate hypotheses, they do not confirm any.
    arms = {
        "solo_qv_pari": ["sig_qv"],                     # order 2, even
        "solo_rendimento_l1": ["sig_x"],                # level 1
        "solo_area_levy": ["sig_area"],                 # level-2 antisym.
        "solo_livello3": ["sig_ttx", "sig_txx"],        # level 3
        "solo_ordinamento": ["sig_area", "sig_ttx", "sig_txx"],
        "solo_logsig_5": ["sig_t", "sig_x", "sig_area", "sig_ttx", "sig_txx"],
    }
    decomp = {}
    for name, extra in arms.items():
        a = fit_eval(tr, ev, HAR_C_COLS + extra)
        dm_a = diebold_mariano(a["loss"], base["loss"], h)
        decomp[name] = {"cols": extra, "qlike": a["qlike"],
                        "delta": float(a["qlike"] - base["qlike"]),
                        "p_value": dm_a["p_value"], "rank": a["rank"], "ncol": a["ncol"]}
        log.info(f"[descrittiva] {name:20s} Δ = {decomp[name]['delta']:+.6f} · "
                 f"QLIKE = {a['qlike']:.6f} · p = {dm_a['p_value']:.6f} · "
                 f"rango {a['rank']}/{a['ncol']}")

    cond1 = bool(cand["qlike"] <= QLIKE_SIG_MAX)
    cond2 = bool(dm["p_value"] <= P_MAX and delta < 0)
    verdict = ("NESSUNA CONCLUSIONE" if not pc_fires
               else "PASS" if (cond1 and cond2) else "FAIL")
    log.info(f"cond① materialità (QLIKE ≤ {QLIKE_SIG_MAX}) = {cond1} · "
             f"cond② significatività (p ≤ {P_MAX}) = {cond2} · "
             f"cond③ controllo positivo = {pc_fires}")
    log.info(f"VERDETTO / VERDICT: {verdict}")

    out = {
        "gate": "SignatureHAR", "prereg": "STATUS.md 2026-08-25", "split": split,
        "window_frozen": W_FROZEN, "h": h, "n_eval": int(len(ev)), "n_train": int(len(tr)),
        "construction": {"logsig_antisymmetry_max": anti,
                         "level1_x_vs_sum_returns_max": max_dx,
                         "degenerate_columns": degenerate},
        "baseline_har_c": {"qlike_production_fn": ref["qlike_har_c"],
                           "qlike_probe_chain": base["qlike"],
                           "qlike_published": QLIKE_HAR_C_PUBLISHED, "cond": base["cond"]},
        "positive_control": {"lambda": pc["lambda"], "qlike": pc["qlike"], "delta": float(pc_delta),
                             "target_delta": DELTA_MATERIAL, "p_value": dm_pc["p_value"],
                             "fires": pc_fires},
        "candidate": {"qlike": cand["qlike"], "delta": float(delta), "cond": cand["cond"],
                      "rank": cand["rank"], "ncol": cand["ncol"],
                      "dm": dm["dm_hln"], "p_value": dm["p_value"], "hac_lag": dm["hac_lag"],
                      "n_eff": dm["n_eff"]},
        "descriptive_arms": decomp,
        "conditions": {"cond1_material": cond1, "cond2_significant": cond2,
                       "cond3_positive_control": pc_fires},
        "verdict": verdict,
    }
    path = Path("results/vols") / f"sig_har_probe_1h_{split}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info(f"report: {path}")


def main():
    # UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="SignatureHAR — condizione ③ ex-ante: ridondanza del termine di ordine 2 / "
                    "SignatureHAR — ex-ante condition ③: order-2 term redundancy")
    # windows to test — 24 is the one FROZEN in the pre-registration, 30 is the
    # one matched to `xc_h` (h = forecast_horizon). Both are reported: a gap in
    # ρ between them is attributable to the window, not to the mechanism.
    ap.add_argument("--windows", default="24,30",
                    help="finestre W separate da virgola / comma-separated W")
    # `cond3` = ex-ante check (already run 2026-08-25, kept reproducible);
    # `gate` = val evaluation of the pre-registered gate. Default unchanged:
    # running the bare probe consumes nothing of the gate's split.
    ap.add_argument("--mode", default="cond3", choices=["cond3", "gate"],
                    help="cond3 = condizione ex-ante · gate = valutazione su val / "
                         "cond3 = ex-ante condition · gate = val evaluation")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    h = int(cfg["features"].get("forecast_horizon", 30))
    bars_day = 1440 // interval_minutes_from_cfg(cfg)
    log.info(f"h={h} barre · bars/day={bars_day} · interval={cfg['data']['interval']}")

    raw = pd.read_parquet("data/raw_candles.parquet").sort_values("open_time").reset_index(drop=True)
    lr = np.log(raw["close"] / raw["close"].shift(1)).values

    # HAR-C frame from the shared module — IDENTICAL to the judge's, so `xc_h`
    # is exactly the incumbent regressor, not a reconstruction of it.
    har_cj = build_har_cj_frame(raw, h, bars_day)

    # TRAIN split from the npz: condition ③ is checked on train, with no
    # contact with val (the gate's split).
    d = np.load(str(dataset_npz_path()), allow_pickle=True)
    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)

    if args.mode == "gate":
        run_gate(raw, lr, har_cj, d, h)
        return

    for w in [int(v) for v in args.windows.split(",")]:
        rw = rolling_windows(lr, w)
        qv = leadlag_qv_term(rw)

        # IDENTITY check — the segment-computed term must equal Σ r² over the
        # window. If it does not, the bug is in the path construction and the
        # rest of the number is worthless.
        direct = np.nansum(rw ** 2, axis=1)
        ok = np.isfinite(qv) & np.isfinite(direct) & (direct > 0)
        max_rel = float(np.max(np.abs(qv[ok] - direct[ok]) / direct[ok]))
        log.info(f"[W={w}] identita lead-lag Levy vs sum(r^2): scarto relativo max = {max_rel:.3e}")

        sig = pd.DataFrame({"open_time": raw["open_time"], "qv": qv}).set_index("open_time")
        sig.index = pd.to_datetime(sig.index).tz_localize(None)

        j = har_cj.join(sig, how="inner").dropna()
        j = j.loc[j.index.intersection(t_train)]
        j = j[j["qv"] > 0]
        log.info(f"[W={w}] righe di train allineate / aligned train rows: {len(j)}")

        qv_v = j["qv"].values
        xch = j["xc_h"].values
        logqv = np.log(qv_v + EPS)

        # three ρ readings, declared BEFORE looking at them, plus the design R².
        rho_log = float(np.corrcoef(logqv, xch)[0, 1])                      # scale-matched
        rho_lvl = float(np.corrcoef(qv_v, xch)[0, 1])                       # literal
        rk_qv = pd.Series(qv_v).rank().values
        rk_xc = pd.Series(xch).rank().values
        rho_sp = float(np.corrcoef(rk_qv, rk_xc)[0, 1])                     # Spearman
        r2_lvl = r2_on_design(qv_v, j[HAR_C_COLS].values)
        r2_log = r2_on_design(logqv, j[HAR_C_COLS].values)

        log.info(f"[W={w}] rho Pearson  log(QV) vs xc_h = {rho_log:.6f}  <- primaria / primary")
        log.info(f"[W={w}] rho Spearman     QV  vs xc_h = {rho_sp:.6f}  (invariante monotona)")
        log.info(f"[W={w}] rho Pearson      QV  vs xc_h = {rho_lvl:.6f}  (letterale / literal)")
        log.info(f"[W={w}] R2  di QV      sui 3 HAR-C   = {r2_lvl:.6f}")
        log.info(f"[W={w}] R2  di log(QV) sui 3 HAR-C   = {r2_log:.6f}")

        # condition ③ DIAGNOSTIC — if ρ vs `xc_h` falls short of 0.99, the
        # question is WHAT the term is redundant against. `xc_h = log(C_h)` with
        # C = RV − J: HAR-C (C3, 31/07) DELIBERATELY dropped the jumps. The
        # order-2 term is the quadratic variation, i.e. RV = C + J, hence two
        # checks: (a) ρ against log(RV_h) on the same window — must be ≈ 1 by
        # identity; (b) how much of log(QV)'s residual on the HAR-C design is
        # explained by the jump column `xj_h`.
        rv_same = pd.Series(np.log(np.nansum(rolling_windows(lr, w) ** 2, axis=1) + EPS),
                            index=pd.to_datetime(raw["open_time"]).dt.tz_localize(None))
        xh_v = rv_same.loc[j.index].values
        rho_rv = float(np.corrcoef(logqv, xh_v)[0, 1])
        xd = np.column_stack([np.ones(len(j)), j[HAR_C_COLS].values])
        beta, *_ = np.linalg.lstsq(xd, logqv, rcond=None)
        resid = logqv - xd @ beta
        rho_jump = float(np.corrcoef(resid, j["xj_h"].values)[0, 1])
        log.info(f"[W={w}] rho Pearson  log(QV) vs log(RV_{w}) = {rho_rv:.6f}  (identita attesa)")
        log.info(f"[W={w}] rho residuo(log QV | HAR-C) vs xj_h = {rho_jump:.6f}  (colonna salti)")


if __name__ == "__main__":
    main()
