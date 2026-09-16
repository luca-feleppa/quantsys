# VOL-S — QLIKE JUDGE (pre-registered in STATUS.md 2026-06-10).
# Compares h-bar realized-variance forecasts on val/test
# (h = features.forecast_horizon; bar resolution = data.interval, both parametric):
#   · NN (iTransformer ensemble, log-RV target, z-score → raw via the
#     RobustScaler's center+scale: NB denormalize_predictions is NOT enough,
#     log-RV has median ≈ −7, the center is required too)
#   · HAR-RV (Corsi 2009): OLS on log-RV with trailing h-bar/7d/30d components,
#     fit on train ONLY (same information set as the NN)
#   · naive persistence: RV_pred = trailing h-bar RV (sanity floor)
# Primary judge: QLIKE on RV levels, exp(log_pred) for ALL (same transform →
# fair comparison). Secondary: MSE on log-RV.
# GATE (test): QLIKE_NN ≤ 0.95·QLIKE_HAR  AND  QLIKE_NN < QLIKE_naive.
# Protocol: val-first; test is evaluated ONCE.
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, interval_minutes_from_cfg, models_root, dataset_npz_path  # noqa: E402
# QLIKE + ε now live in the shared module (single source of truth with the 02b harness).
from quantsys.model.vol_metrics import (qlike, qlike_series, diebold_mariano,  # noqa: E402
                                        duan_smearing, HAR_CJ_COLS, HAR_C_COLS, EPS)

setup_logging()
log = logging.getLogger("quantsys.script.vols_qlike")

# default/fallback for the horizon in bars — the effective value is read
# from cfg["features"]["forecast_horizon"] inside main().
H = 30


# report name — single source of truth, extracted to be testable without running
# the judge. Two independent, both anti-clobber suffixes: the ROOT
# (`QUANTSYS_MODELS_ROOT`, candidate vs incumbent) and the ARCH (model dir inside
# `models/`, e.g. R1's canonical artifact). The production path — default arch and
# default root — keeps the BARE name, so historical reports are never renamed.
def report_filename(interval: str, split: str, arch: str = "itransformer",
                    root_name: str = "models") -> str:
    arch_sfx = f"_{arch}" if arch != "itransformer" else ""
    root_sfx = f"_{root_name}" if root_name != "models" else ""
    return f"qlike_report_{interval}_{split}{arch_sfx}{root_sfx}.json"


def main():
    # UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # minimal argparse — model arch to judge (models/{arch}); explicit CLI flag,
    # NOT the QUANTSYS_ARCH env var — default itransformer = bit-identical
    # legacy run.
    ap = argparse.ArgumentParser(description="Giudice QLIKE vol-S (NN vs HAR-RV vs naive) / "
                                             "VOL-S QLIKE judge (NN vs HAR-RV vs naive)")
    # MINOR-B (B1 audit 2026-07-18) — itransformer_regime_moe allowed: the A3
    # run writes to {models_root}/itransformer_regime_moe (models_a3_moe
    # sandbox); without the choice the judge cannot target the right dir.
    # 2026-07-19: same fix for itransformer_a8_mixup (models_a8_mixup sandbox,
    # B3 run) — added EX-ANTE, before any A8 training.
    # 2026-07-29: same EX-ANTE fix for itransformer_a10_sparsity (A10 pre-reg of
    # 28/07, models_a10_sparsity sandbox) — added before the first checkpoint
    # exists, so the judge cannot block the run after the GPU is already spent.
    # 2026-08-04: + canonical_1h_vols — not an architecture but the directory of
    # R1's canonical pair (`models/canonical_1h_vols`, permanent artifact). The
    # flag has long meant "model directory name", not "architecture": the three
    # choices above are sandbox dirs too. It is needed so the artifact can be
    # RE-JUDGED; otherwise it is a checkpoint declared reproducible yet not
    # verifiable — the very defect R1 closes.
    ap.add_argument("--arch", default="itransformer",
                    choices=["itransformer", "nhits", "tcnmamba", "lstm",
                             "itransformer_regime_moe", "itransformer_a8_mixup",
                             "itransformer_a10_sparsity", "canonical_1h_vols"],
                    help="architettura del modello vol da caricare (models/{arch}) / "
                         "vol model architecture to load (models/{arch})")
    # 2026-08-01 — EXPLICIT escape hatch for the scaler guard (see below). It
    # must be a flag, not an env var: a cross-vintage run produces a number NOT
    # comparable with the other reports, so its nature must stay written in the
    # invocation and end up in the report.
    ap.add_argument("--allow-scaler-mismatch", action="store_true",
                    help="procedi anche se lo scaler del modello ≠ scaler del dataset: "
                         "il numero NON è confrontabile con gli altri report / proceed even "
                         "if the model scaler ≠ dataset scaler: the number is NOT comparable")
    # M1 — escape hatch kept SEPARATE from the scaler one on purpose: they are two
    # independent vintage axes, and accepting one must not silently accept the other.
    ap.add_argument("--allow-macro-mismatch", action="store_true",
                    help="procedi anche se il vintage macro del modello ≠ quello dell'npz: "
                         "il numero NON è confrontabile / proceed even if the model's macro "
                         "vintage ≠ the npz's: the number is NOT comparable")
    args = ap.parse_args()

    # env-aware root (QUANTSYS_MODELS_ROOT) — judges the isolated sandbox if set.
    model_dir = models_root() / args.arch
    log.info(f"dir modelli effettiva / effective model dir: {model_dir} (arch={args.arch})")

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config features.target_type deve essere log_rv per il giudice vol-S"

    # horizon and bar resolution from the config (module H = fallback only) —
    # keeps the judge consistent with the current dataset/target, no hardcoding.
    h = int(cfg["features"].get("forecast_horizon", H))
    interval = cfg["data"]["interval"]
    bars_day = 1440 // interval_minutes_from_cfg(cfg)  # bars/day from interval
    log.info(f"horizon h={h} barre · interval={interval} · bars/day={bars_day}")

    # split to judge — val by default (val-first); test ONLY once val sanity passes.
    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    assert split in ("val", "test")

    # C1 (STATUS 2026-07-28 pre-reg) — Duan (1983) smearing-correction lever on the
    # judge. INERT by default: with the flag off nothing is estimated, the train
    # inference pass is skipped and the numeric path is the historical one
    # bit-for-bit (the correction NEVER touches `res`/`gate`: the smeared block
    # lives in a separate report section). This is a judge SPECIFICATION check,
    # not a model lever.
    smearing = os.environ.get("QUANTSYS_QLIKE_SMEARING", "0") == "1"
    if smearing:
        log.info("C1 SMEARING ATTIVO / ACTIVE (QUANTSYS_QLIKE_SMEARING=1): "
                 "ŝ stimato SOLO su train, riportato accanto ai valori raw / "
                 "ŝ estimated on TRAIN only, reported alongside the raw values")

    # ── Ground truth + HAR features from raw candles (same target definition) ──
    raw = pd.read_parquet("data/raw_candles.parquet")
    raw = raw.sort_values("open_time").reset_index(drop=True)
    lr2 = np.log(raw["close"] / raw["close"].shift(1)) ** 2

    rv_h = lr2.rolling(h).sum()                      # trailing h-bar RV
    rv_w = lr2.rolling(7 * bars_day).sum() / 7       # 7d daily mean
    rv_m = lr2.rolling(30 * bars_day).sum() / 30     # 30d daily mean
    rv_fwd = rv_h.shift(-h)                          # target = already-computed rolling, shifted (A-minor)

    har = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":  np.log(rv_fwd + EPS),
        "xh": np.log(rv_h + EPS),
        # weekly/monthly components rescaled to the h-bar horizon for dimensional consistency
        "xw": np.log(rv_w * (h / bars_day) + EPS),
        "xm": np.log(rv_m * (h / bars_day) + EPS),
    }).dropna().set_index("open_time")

    # ── Alignment to the NN dataset split timestamps ────────────────────────────
    # same env-aware npz as training (QUANTSYS_DATASET_NPZ, default unchanged).
    d = np.load(str(dataset_npz_path()), allow_pickle=True)
    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)
    t_eval  = pd.to_datetime(d[f"t_{split}"]).tz_localize(None)
    har.index = pd.to_datetime(har.index).tz_localize(None)

    tr = har.loc[har.index.intersection(t_train)]
    ev = har.loc[har.index.intersection(t_eval)]
    log.info(f"HAR rows: train {len(tr)}/{len(t_train)}  {split} {len(ev)}/{len(t_eval)}")
    assert len(ev) >= 0.95 * len(t_eval), "allineamento HAR↔split insufficiente"

    # ── Baseline 1: HAR-RV (closed-form OLS, fit on train) ──────────────────────
    Xtr = np.column_stack([np.ones(len(tr)), tr[["xh", "xw", "xm"]].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    Xev = np.column_stack([np.ones(len(ev)), ev[["xh", "xw", "xm"]].values])
    log_pred_har = Xev @ beta
    log.info(f"HAR beta: const={beta[0]:.3f} h={beta[1]:.3f} w={beta[2]:.3f} m={beta[3]:.3f}")

    # ── Baseline 2: naive persistence ───────────────────────────────────────────
    log_pred_naive = ev["xh"].values

    # ── Baselines 3 and 4: HAR-C and HAR-CJ — ALWAYS computed (adopted 2026-07-31) ──
    # born as C2/C3 env-gated levers, promoted to a stable part of the instrument
    # when C3 closed. The two flags `QUANTSYS_HAR_CJ`/`QUANTSYS_HAR_C` were REMOVED, not switched on: an always-on flag
    # is not a lever, it is one more failure mode (the inconsistent combination and
    # its guard disappear with them). To reproduce the historical pre-C2 report use
    # version control — `git show <commit>:scripts/vol/dev_vols_qlike.py` — which is
    # how all three inertia proofs were done: duplicating it with a compatibility
    # flag would be reimplementing git inside the application.
    # ⚠ INVARIANT PRESERVED: the blocks stay OUTSIDE `metrics`/`gate`, so the
    # pre-registered vol-S gate (2026-06-10, HAR-RV denominator) is structurally
    # uncontaminable — the promotion does not touch it.
    # The CJ frame is built SEPARATELY (bipower variation carries one extra lag: merging
    # it into the HAR-RV frame before the dropna would shift the historical baseline's
    # sample by one bar) and is then narrowed to EXACTLY the `ev` timestamps. HAR-C
    # reuses that frame: sample identity across the three baselines holds BY
    # CONSTRUCTION, not through an alignment to be checked downstream.
    from quantsys.model.vol_metrics import build_har_cj_frame                # noqa: PLC0415
    har_cj = build_har_cj_frame(raw, h, bars_day)
    tr_cj = har_cj.loc[har_cj.index.intersection(tr.index)]
    ev_cj = har_cj.loc[har_cj.index.intersection(ev.index)]
    # guard on index IDENTITY, not just count: two indices of equal length but
    # different order would pair predictions with the wrong ground truth, giving
    # a plausible-looking but false QLIKE — the kind of error you cannot spot by
    # looking at the number. (2026-07-30 audit.)
    if not ev_cj.index.equals(ev.index):
        raise RuntimeError(
            f"allineamento HAR-C/HAR-CJ ↔ HAR-RV non esatto sull'eval "
            f"({len(ev_cj)} vs {len(ev)} righe, indici {'di pari lunghezza ma diversi' if len(ev_cj)==len(ev) else 'di lunghezza diversa'}) "
            f"— il confronto appaiato richiede gli STESSI sample nello STESSO ordine / "
            f"non-exact HAR-C/HAR-CJ ↔ HAR-RV eval alignment"
        )
    Xtr_cj = np.column_stack([np.ones(len(tr_cj)), tr_cj[HAR_CJ_COLS].values])
    beta_cj, *_ = np.linalg.lstsq(Xtr_cj, tr_cj["y"].values, rcond=None)
    Xev_cj = np.column_stack([np.ones(len(ev_cj)), ev_cj[HAR_CJ_COLS].values])
    log_pred_har_cj = Xev_cj @ beta_cj

    # HAR-C — same mechanics on 3 columns instead of 6, same train, same eval: ONLY
    # the regressor set differs. It is the published claim's reference baseline
    # since 2026-07-31 (C3).
    Xtr_c = np.column_stack([np.ones(len(tr_cj)), tr_cj[HAR_C_COLS].values])
    beta_c, *_ = np.linalg.lstsq(Xtr_c, tr_cj["y"].values, rcond=None)
    Xev_c = np.column_stack([np.ones(len(ev_cj)), ev_cj[HAR_C_COLS].values])
    log_pred_har_c = Xev_c @ beta_c
    log.info(f"HAR-C  beta: const={beta_c[0]:.3f} "
             f"C[h,w,m]=({beta_c[1]:.3f},{beta_c[2]:.3f},{beta_c[3]:.3f})  "
             f"train={len(tr_cj)}")
    log.info(f"HAR-CJ beta: const={beta_cj[0]:.3f} "
             f"C[h,w,m]=({beta_cj[1]:.3f},{beta_cj[2]:.3f},{beta_cj[3]:.3f}) "
             f"J[h,w,m]=({beta_cj[4]:.3f},{beta_cj[5]:.3f},{beta_cj[6]:.3f})")

    # ── NN: ensemble forward on X_{split} → z → raw log-RV (center+scale) ───────
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.utils import PipelineState
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load(str(model_dir), device)
    ps = PipelineState.load(str(model_dir / "pipeline_state.pkl"))
    idx = ps.scale_cols.index("target_ret")
    c, s = float(ps.scaler.center_[idx]), float(ps.scaler.scale_[idx])
    log.info(f"target_ret scaler: center={c:.3f} scale={s:.3f} (deve essere ~log-RV, non ~0)")
    assert c < -3, "center ≈ 0 → il PipelineState non è del dataset log-RV (stale?)"

    # MODEL↔DATASET SCALER GUARD (2026-08-01). The assert above only catches a
    # grossly wrong state (center ≈ 0); it does NOT catch a plausible state from
    # ANOTHER dataset vintage, which is what actually happened:
    # `models/itransformer` is the June PASS restore and its QLIKE on the current
    # npz (0.27470 val) is ~5% worse than a pair retrained on that same npz
    # (0.26143) — a difference that is a scaler artifact, not skill. Comparing
    # two models across different scalers is forbidden; doing it silently is how
    # it happened. Fail-fast, not a warning: a warning inside a 600-line log has
    # stopped nobody so far.
    from quantsys.utils import assert_model_dataset_scaler
    prov = assert_model_dataset_scaler(ps, model_dir=model_dir, arch=args.arch,
                                       npz=dataset_npz_path(),
                                       allow_mismatch=args.allow_scaler_mismatch,
                                       npz_arrays=d,
                                       allow_macro_mismatch=args.allow_macro_mismatch,
                                       logger=log)

    X = torch.tensor(d[f"X_{split}"], dtype=torch.float32)
    Xm = torch.tensor(d[f"X_macro_{split}"], dtype=torch.float32) if f"X_macro_{split}" in d.files else None

    # A3 regime-MoE — if the model's config.json declares head_type=
    # "regime_moe", build the causal gate aligned to the split timestamps and
    # pass it as g= (EnsembleModel forwards kwargs to the members).
    # Key absent (all legacy models) → inert branch, bit-identical call.
    _head_type = "single"
    _mdl_cfg = model_dir / "config.json"
    if _mdl_cfg.exists():
        with open(_mdl_cfg, encoding="utf-8") as f:
            _head_type = json.load(f).get("head_type", "single") or "single"
    G = None
    if _head_type == "regime_moe":
        from quantsys.model.regime_gate import build_regime_gate
        G = torch.from_numpy(build_regime_gate(d[f"t_{split}"]))
        log.info(f"regime_moe: gate (N,3) costruito per t_{split} / gate built for t_{split}")

    # ensemble forward in 256-sized batches → μ in z-space. Extracted into a function
    # (C1) to reuse the SAME batching on the train split when ŝ must be estimated:
    # operations and order unchanged vs the historical inline loop → bit-identical.
    # EnsembleModel.__call__ → (mu_ens, sigma_ens, nu_ens) in z-space (AMP off internally).
    def _forward_mu_z(Xa, Xma, Ga) -> np.ndarray:
        mus = []
        with torch.no_grad():
            for i in range(0, len(Xa), 256):
                xb = Xa[i:i + 256].to(device)
                xmb = Xma[i:i + 256].to(device) if Xma is not None else None
                if Ga is not None:
                    mu, _, _ = model(xb, xmb, g=Ga[i:i + 256].to(device))
                else:
                    mu, _, _ = model(xb, xmb)
                mus.append(mu.detach().cpu().numpy().ravel())
        return np.concatenate(mus)

    mu_z = _forward_mu_z(X, Xm, G)
    # FULL z→raw inversion: μ·IQR + center (see header note).
    log_pred_nn_full = mu_z * s + c

    # re-align NN predictions to the timestamps present in `ev`.
    pos = {ts: k for k, ts in enumerate(t_eval)}
    sel = np.array([pos[ts] for ts in ev.index])
    log_pred_nn = log_pred_nn_full[sel]

    # ── Judgment ────────────────────────────────────────────────────────────────
    rv_true = np.exp(ev["y"].values)  # = rv_fwd + EPS
    res = {}
    losses = {}  # per-sample QLIKE losses, for DM
    for name, lp in [("nn", log_pred_nn), ("har", log_pred_har), ("naive", log_pred_naive)]:
        losses[name] = qlike_series(rv_true, np.exp(lp))
        res[name] = {
            "qlike":   qlike(rv_true, np.exp(lp)),
            "mse_log": float(np.mean((ev["y"].values - lp) ** 2)),
        }
        log.info(f"{name:6s} QLIKE={res[name]['qlike']:.5f}  MSE(log)={res[name]['mse_log']:.4f}")

    # INFERENCE on the comparison (Diebold-Mariano, added 2026-07-26) — the QLIKE
    # ratio is a POINT estimate: without HAC the standard error is understated by
    # ~sqrt(h) because the target sums h bars (overlapping windows). DESCRIPTIVE,
    # NOT GATING: the pre-registered PASS conditions remain the QLIKE ratios
    # (0.95·HAR and < naive) and are untouched by these p-values.
    # Fail-soft: an error here does not invalidate the aggregate verdict.
    dm = {}
    try:
        for label, (a, b) in {"nn_vs_har": ("nn", "har"),
                              "nn_vs_naive": ("nn", "naive")}.items():
            dm[label] = diebold_mariano(losses[a], losses[b], h=h)
            r = dm[label]
            log.info(f"DM {label}: stat={r['dm_hln']:+.3f} p={r['p_value']:.2e} "
                     f"(HAC lag={r['hac_lag']}, n={r['n']}, n_eff≈{r['n_eff']:.0f}, "
                     f"migliore/better={r['better']})")
    except Exception as e:  # noqa: BLE001
        log.warning(f"blocco Diebold-Mariano fallito / DM block failed: {e}")
    res["diebold_mariano"] = dm

    # ── C2: HAR-CJ baseline evaluation (SEPARATE block, never inside `res`) ──
    # the pre-reg's four conditions are COMPUTED but NOT decisional here: the
    # printed verdict remains the original vol-S gate (NN vs HAR-RV and naive).
    # Keeping the block outside `res`/`gate` is what makes contaminating an
    # already-registered gate impossible by construction, not by discipline.
    # `enabled` kept True for schema COMPATIBILITY with the reports already on disk
    # and the tables committed in STATUS: the key no longer has a switch behind it,
    # but removing it would make the C2/C3 reports non-diffable.
    har_cj_block = {"enabled": True}
    losses["har_cj"] = qlike_series(rv_true, np.exp(log_pred_har_cj))
    q_cj = qlike(rv_true, np.exp(log_pred_har_cj))
    q_rv = res["har"]["qlike"]
    q_nn = res["nn"]["qlike"]
    ratio_rv, ratio_cj = q_nn / q_rv, q_nn / q_cj
    dm_cj = {}
    try:
        dm_cj["nn_vs_har_cj"] = diebold_mariano(losses["nn"], losses["har_cj"], h=h)
        dm_cj["har_cj_vs_har"] = diebold_mariano(losses["har_cj"], losses["har"], h=h)
    except Exception as e:  # noqa: BLE001
        log.warning(f"DM HAR-CJ fallito / DM HAR-CJ failed: {e}")
    har_cj_block.update({
        "qlike_har_cj": float(q_cj),
        "qlike_har_rv": float(q_rv),
        "mse_log_har_cj": float(np.mean((ev["y"].values - log_pred_har_cj) ** 2)),
        "beta": [float(b) for b in beta_cj],
        "ratio_rv": float(ratio_rv), "ratio_cj": float(ratio_cj),
        "delta_ratio": float(ratio_cj - ratio_rv),
        "n_eval": int(len(ev)),
        # C2's four conditions — gate CLOSED on 2026-07-30. Kept as continuous
        # monitoring (② tells every run whether the claim still holds against the
        # strong baseline), NOT as live conditions.
        "cond1_cj_stronger": bool(q_cj <= q_rv),
        "cond2_claim_survives": bool(q_nn <= 0.95 * q_cj),
        "cond3_material": bool(abs(ratio_cj - ratio_rv) >= 0.02),
        "cond4_n_obs": bool(len(ev) >= 5000),
        "diebold_mariano": dm_cj,
    })

    # ── HAR-C: nested sub-block (schema stable since the C3 reports) ───────────
    # sitting inside a block already outside `metrics`/`gate`, the pre-registered
    # vol-S gate is uncontaminable a second time, by construction. `phi` stays
    # DESCRIPTIVE: its denominator (q_rv − q_cj) was an already-observed quantity
    # when C3 was pre-registered, and the peeking warning forbade anchoring a
    # threshold to it — the note still holds for anyone re-reading the number.
    losses["har_c"] = qlike_series(rv_true, np.exp(log_pred_har_c))
    q_c = qlike(rv_true, np.exp(log_pred_har_c))
    dm_c = {}
    try:
        # diebold_mariano(a,b) convention → NEGATIVE stat = `a` is better
        dm_c["test_a_har_c_vs_har_rv"] = diebold_mariano(losses["har_c"], losses["har"], h=h)
        dm_c["test_b_har_cj_vs_har_c"] = diebold_mariano(losses["har_cj"], losses["har_c"], h=h)
    except Exception as e:  # noqa: BLE001
        log.warning(f"DM HAR-C fallito / DM HAR-C failed: {e}")
    denom = q_rv - q_cj
    phi = float((q_rv - q_c) / denom) if abs(denom) > 1e-12 else float("nan")
    har_cj_block["har_c"] = {
        "enabled": True,
        "qlike_har_c": float(q_c),
        "mse_log_har_c": float(np.mean((ev["y"].values - log_pred_har_c) ** 2)),
        "beta": [float(b) for b in beta_c],
        # attribution — share of the CJ gain from the substitution alone
        "phi_attribution": phi,
        # conditioning of the two designs (diagnostic, never decisional): it supports
        # the instrument-STABILITY argument, not accuracy. It is why HAR-C, not
        # HAR-CJ, is the claim's baseline.
        "cond_number_har_c": float(np.linalg.cond(Xtr_c)),
        "cond_number_har_cj": float(np.linalg.cond(Xtr_cj)),
        # published CLAIM ratio (reference baseline since 2026-07-31)
        "ratio_nn_over_har_c": float(q_nn / q_c),
        "diebold_mariano": dm_c,
    }
    log.info(f"baseline: HAR-RV={q_rv:.5f} (gate) · HAR-C={q_c:.5f} (claim) · "
             f"HAR-CJ={q_cj:.5f}   ratio NN/HAR-C={q_nn / q_c:.4f}  phi={phi:+.3f}")

    # A8-bis (2026-07-19 pre-reg) — per-regime QLIKE breakdown: labels = argmax of
    # the causal gate (model-independent), aligned to the `ev` samples via `sel`.
    # Feeds per-regime gate condition ②; fail-soft with a warning (the aggregate
    # verdict does NOT depend on this block).
    per_regime = {}
    try:
        from quantsys.model.regime_gate import build_regime_gate as _brg
        lbl = _brg(d[f"t_{split}"]).argmax(axis=1)[sel]
        for r in range(3):
            m = lbl == r
            per_regime[f"r{r}"] = {
                "n": int(m.sum()),
                "qlike_nn":  float(qlike(rv_true[m], np.exp(log_pred_nn[m]))) if m.any() else None,
                "qlike_har": float(qlike(rv_true[m], np.exp(log_pred_har[m]))) if m.any() else None,
            }
        log.info("per-regime QLIKE: " + "  ".join(
            f"r{r}[n={per_regime[f'r{r}']['n']}] nn={per_regime[f'r{r}']['qlike_nn']:.5f}"
            for r in range(3) if per_regime[f"r{r}"]["n"]))
    except Exception as e:  # pragma: no cover
        log.warning(f"breakdown per-regime non disponibile / unavailable: {e}")

    # ── C1: Duan (1983) smearing correction, SYMMETRIC on both sides ────────────
    # STATUS 2026-07-28 pre-reg. ŝ = mean of exp(ε) over the TRAIN-only log
    # residuals, applied at evaluation as RV_corr = exp(log_pred)·ŝ. BOTH sides of
    # the comparison (NN and HAR) get their OWN factor: correcting one side only is
    # excluded ex-ante by the pre-registration. `naive` is descriptive. Adoption
    # conditions ①②③ are computed here but decide nothing on their own: the
    # decision (and rewriting the published band) stays manual. The `metrics` block
    # and the pre-registered 2026-06-10 `gate` are left UNTOUCHED.
    smear_block = {"enabled": bool(smearing)}
    if smearing:
        # ŝ_HAR and ŝ_naive from the IN-SAMPLE train residuals (same sample as the OLS).
        eps_har = tr["y"].values - Xtr @ beta
        eps_naive = tr["y"].values - tr["xh"].values

        # ŝ_NN — inference pass over the TRAIN split (~52k windows × 5 members).
        # `from_numpy` (zero-copy) rather than `tensor` to avoid duplicating 2.6 GB;
        # tensors freed right after. Log residual: (y_z − μ_z)·scale (the center
        # cancels in the difference — same inversion as the judge).
        import gc
        Xtr_t = torch.from_numpy(np.ascontiguousarray(d["X_train"]))
        Xmtr_t = (torch.from_numpy(np.ascontiguousarray(d["X_macro_train"]))
                  if "X_macro_train" in d.files else None)
        Gtr = None
        if _head_type == "regime_moe":
            from quantsys.model.regime_gate import build_regime_gate as _brg_tr
            Gtr = torch.from_numpy(_brg_tr(d["t_train"]))
        log.info(f"C1: forward sul train per ŝ_NN / train forward for ŝ_NN "
                 f"({len(Xtr_t)} finestre/windows)...")
        mu_z_train = _forward_mu_z(Xtr_t, Xmtr_t, Gtr)
        eps_nn = (np.asarray(d["y_train"], dtype=np.float64) - mu_z_train) * s
        del Xtr_t, Xmtr_t, Gtr
        gc.collect()

        s_nn, s_har, s_naive = (duan_smearing(eps_nn), duan_smearing(eps_har),
                                duan_smearing(eps_naive))
        log.info(f"C1 fattori di Duan / Duan factors: ŝ_NN={s_nn:.4f}  ŝ_HAR={s_har:.4f}  "
                 f"ŝ_naive={s_naive:.4f}  (n_train NN={len(eps_nn)}, HAR={len(eps_har)})")

        factors = {"nn": s_nn, "har": s_har, "naive": s_naive}
        preds = {"nn": log_pred_nn, "har": log_pred_har, "naive": log_pred_naive}
        metrics_smear = {}
        for name, lp in preds.items():
            metrics_smear[name] = {"qlike": qlike(rv_true, np.exp(lp) * factors[name])}
            log.info(f"{name:6s} QLIKE_smear={metrics_smear[name]['qlike']:.5f} "
                     f"(raw {res[name]['qlike']:.5f}, ŝ={factors[name]:.4f})")

        ratio_raw = res["nn"]["qlike"] / res["har"]["qlike"]
        ratio_smear = metrics_smear["nn"]["qlike"] / metrics_smear["har"]["qlike"]
        smear_block.update({
            "factors": {k: float(v) for k, v in factors.items()},
            "n_train": {"nn": int(len(eps_nn)), "har": int(len(eps_har))},
            "metrics_smeared": metrics_smear,
            "ratio_raw": float(ratio_raw),
            "ratio_smear": float(ratio_smear),
            "delta_ratio": float(ratio_smear - ratio_raw),
            # ① specification coherence on BOTH sides
            "cond1_coherent_both_sides": bool(
                metrics_smear["nn"]["qlike"] <= res["nn"]["qlike"]
                and metrics_smear["har"]["qlike"] <= res["har"]["qlike"]),
            # ② materiality ≥0.02 of ratio
            "cond2_material": bool(abs(ratio_smear - ratio_raw) >= 0.02),
            # ③ sample validity (non-leakage is structural in the code)
            "cond3_n_obs": bool(len(ev) >= 5000),
        })

    gate = {
        "split": split,
        "nn_vs_har_ratio": res["nn"]["qlike"] / res["har"]["qlike"],
        "beats_har_5pct": bool(res["nn"]["qlike"] <= 0.95 * res["har"]["qlike"]),
        "beats_naive":    bool(res["nn"]["qlike"] < res["naive"]["qlike"]),
        "n_obs": int(len(ev)),
    }
    gate["verdict"] = "PASS" if (gate["beats_har_5pct"] and gate["beats_naive"]) else "FAIL"

    out_dir = Path("results/vols"); out_dir.mkdir(parents=True, exist_ok=True)
    # report suffixed by interval+split — runs at different resolutions do not overwrite.
    # 2026-07-19: + sandbox suffix when QUANTSYS_MODELS_ROOT is set — candidate and
    # incumbent runs no longer clobber each other (seen on B2/B3: the candidate
    # report only survived in the logs). Production path (no env) UNCHANGED.
    # 2026-08-04 — the ARCH suffix originates here: judging an artifact living as
    # an arch-directory inside `models/` (R1's canonical pair) wrote to the BARE
    # name, i.e. over the historical production report §12.2 cites as the June
    # checkpoint's number. The sandbox root only covered `QUANTSYS_MODELS_ROOT`.
    # Rule lives in `report_filename`, with dedicated tests.
    out_path = out_dir / report_filename(interval, split, args.arch, models_root().name)
    with open(out_path, "w", encoding="utf-8") as f:
        # `provenance` (2026-08-01) — the report MUST say which model produced
        # `metrics.nn`. Without it a number is orphaned from the model that
        # generated it: that is exactly how the C3 reports (NN from
        # `models/itransformer`) were read as if they held the published band's
        # numerator (NN from a retrained pair, sandbox later deleted). The
        # sample was identical digit-for-digit across every baseline, so nothing
        # flagged the difference.
        json.dump({"metrics": res, "gate": gate, "per_regime": per_regime,
                   "har_beta": list(map(float, beta)), "smearing": smear_block,
                   "har_cj": har_cj_block, "provenance": prov}, f, indent=2)

    print(f"\n══════ VOL-S QLIKE [{interval}·{split}] ══════")
    for name in ("nn", "har", "naive"):
        print(f"  {name:6s} QLIKE={res[name]['qlike']:.5f}  MSE(log)={res[name]['mse_log']:.4f}")
    print(f"  NN/HAR ratio: {gate['nn_vs_har_ratio']:.4f}  (gate ≤ 0.95)")
    # descriptive inference (not gating) — see the DM block above.
    for label in ("nn_vs_har", "nn_vs_naive"):
        r = res.get("diebold_mariano", {}).get(label)
        if r and np.isfinite(r.get("dm_hln", float("nan"))):
            print(f"  DM {label:12s} stat={r['dm_hln']:+7.3f}  p={r['p_value']:.2e}"
                  f"  (HAC lag={r['hac_lag']}, n_eff≈{r['n_eff']:.0f}) [descrittivo/descriptive]")
    # C1 — prints raw-vs-smeared side by side; the three conditions are INFORMATIVE
    # (the adoption decision is manual, under the anti-goalpost constraint).
    if smear_block["enabled"]:
        print(f"  ── C1 smearing (Duan 1983) ── ŝ: NN={smear_block['factors']['nn']:.4f}  "
              f"HAR={smear_block['factors']['har']:.4f}  naive={smear_block['factors']['naive']:.4f}")
        for name in ("nn", "har", "naive"):
            print(f"     {name:6s} QLIKE raw={res[name]['qlike']:.5f} → "
                  f"smear={smear_block['metrics_smeared'][name]['qlike']:.5f}")
        print(f"     ratio NN/HAR: raw={smear_block['ratio_raw']:.4f} → "
              f"smear={smear_block['ratio_smear']:.4f}  (Δ={smear_block['delta_ratio']:+.4f})")
        print(f"     ① coerenza/coherence={smear_block['cond1_coherent_both_sides']}  "
              f"② materiale/material(≥0.02)={smear_block['cond2_material']}  "
              f"③ n≥5000={smear_block['cond3_n_obs']}")
    # THREE-baseline panel, each one's ROLE printed next to its number. It is the
    # main reason the computation was made unconditional: whoever runs the judge
    # must see the published claim's denominator without remembering to set a flag,
    # and must see it is NOT the pre-registered gate's denominator. Confusing the
    # two is the mistake this panel exists to make impossible.
    _hc = har_cj_block["har_c"]
    print("  ── baseline econometriche / econometric baselines ──")
    # under a scaler mismatch the NN RATIOS below are not comparable with the
    # published band, while the baseline QLIKEs are (HARs are fitted and
    # evaluated inside the npz, hence model-independent). Without this line the
    # «published CLAIM denominator» label would sit next to a ratio that is NOT
    # the claim's: the panel, built to make one confusion impossible, would
    # introduce another.
    if prov["matches"] is False:
        print("     ⚠ SCALER MISMATCH: i rapporti NN qui sotto NON sono confrontabili con la "
              "banda pubblicata / NN ratios below are NOT comparable with the published band")
        print("       (le QLIKE delle baseline restano valide: sono model-independent / "
              "baseline QLIKEs remain valid: they are model-independent)")
    print(f"     HAR-RV  QLIKE={har_cj_block['qlike_har_rv']:.5f}   ratio NN={har_cj_block['ratio_rv']:.4f}"
          f"   ← denominatore del GATE pre-registrato (2026-06-10)")
    print(f"     HAR-C   QLIKE={_hc['qlike_har_c']:.5f}   ratio NN={_hc['ratio_nn_over_har_c']:.4f}"
          f"   ← denominatore del CLAIM pubblicato (C3, 2026-07-31)")
    print(f"     HAR-CJ  QLIKE={har_cj_block['qlike_har_cj']:.5f}   ratio NN={har_cj_block['ratio_cj']:.4f}"
          f"   ← diagnostica (C2); φ={_hc['phi_attribution']:+.3f}")
    print(f"     cond(design): HAR-C={_hc['cond_number_har_c']:.1f}  "
          f"HAR-CJ={_hc['cond_number_har_cj']:.3e}  → HAR-C è la specificazione "
          f"identificata / is the identified specification")
    for label, tag in (("test_a_har_c_vs_har_rv", "C  vs RV"),
                       ("test_b_har_cj_vs_har_c", "CJ vs C ")):
        r = _hc.get("diebold_mariano", {}).get(label)
        if r and np.isfinite(r.get("dm_hln", float("nan"))):
            print(f"     DM {tag}  stat={r['dm_hln']:+7.3f}  p={r['p_value']:.2e}  "
                  f"migliore/better={r['better']}  [descrittivo/descriptive]")
    if not har_cj_block["cond2_claim_survives"]:
        print("     ⚠ il claim NON regge più contro HAR-CJ (≤0.95) — verificare / "
              "the claim no longer survives against HAR-CJ")
    print(f"  VERDETTO [{split}]: {gate['verdict']}   → {out_path}")


if __name__ == "__main__":
    main()
