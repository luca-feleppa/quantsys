# IT: VOL-S — GIUDICE QLIKE (pre-registrato in STATUS.md 2026-06-10).
#     Confronta su val/test la previsione di realized variance a h barre
#     (h = features.forecast_horizon; risoluzione barra = data.interval, parametrici):
#       · NN (ensemble iTransformer, target log-RV, z-score → raw via center+scale
#         del RobustScaler: NB denormalize_predictions NON basta, il log-RV ha
#         mediana ≈ −7, serve anche il centro)
#       · HAR-RV (Corsi 2009): OLS su log-RV con componenti trailing h-barre/7d/30d,
#         fit SOLO su train (stesso information set del NN)
#       · naive persistence: RV_pred = RV trailing h barre (floor di sanità)
#     Giudice primario: QLIKE su RV in livelli, exp(log_pred) per TUTTI (stessa
#     trasformazione → confronto fair). Secondario: MSE su log-RV.
#     GATE (test): QLIKE_NN ≤ 0.95·QLIKE_HAR  E  QLIKE_NN < QLIKE_naive.
#     Protocollo: val-first; il test si valuta UNA volta.
# EN: VOL-S — QLIKE JUDGE (pre-registered in STATUS.md 2026-06-10).
#     Compares h-bar realized-variance forecasts on val/test
#     (h = features.forecast_horizon; bar resolution = data.interval, both parametric):
#       · NN (iTransformer ensemble, log-RV target, z-score → raw via the
#         RobustScaler's center+scale: NB denormalize_predictions is NOT enough,
#         log-RV has median ≈ −7, the center is required too)
#       · HAR-RV (Corsi 2009): OLS on log-RV with trailing h-bar/7d/30d components,
#         fit on train ONLY (same information set as the NN)
#       · naive persistence: RV_pred = trailing h-bar RV (sanity floor)
#     Primary judge: QLIKE on RV levels, exp(log_pred) for ALL (same transform →
#     fair comparison). Secondary: MSE on log-RV.
#     GATE (test): QLIKE_NN ≤ 0.95·QLIKE_HAR  AND  QLIKE_NN < QLIKE_naive.
#     Protocol: val-first; test is evaluated ONCE.
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
# IT: QLIKE + ε ora vivono nel modulo condiviso (single source of truth con l'harness 02b).
# EN: QLIKE + ε now live in the shared module (single source of truth with the 02b harness).
from quantsys.model.vol_metrics import (qlike, qlike_series, diebold_mariano,  # noqa: E402
                                        duan_smearing, HAR_CJ_COLS, HAR_C_COLS, EPS)

setup_logging()
log = logging.getLogger("quantsys.script.vols_qlike")

# IT: default/fallback dell'orizzonte in barre — il valore effettivo viene letto
#     da cfg["features"]["forecast_horizon"] dentro main().
# EN: default/fallback for the horizon in bars — the effective value is read
#     from cfg["features"]["forecast_horizon"] inside main().
H = 30


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # IT: argparse minimale — arch del modello da giudicare (models/{arch}); flag
    #     CLI esplicito, NON env QUANTSYS_ARCH — default itransformer = run storica
    #     bit-identica.
    # EN: minimal argparse — model arch to judge (models/{arch}); explicit CLI flag,
    #     NOT the QUANTSYS_ARCH env var — default itransformer = bit-identical
    #     legacy run.
    ap = argparse.ArgumentParser(description="Giudice QLIKE vol-S (NN vs HAR-RV vs naive) / "
                                             "VOL-S QLIKE judge (NN vs HAR-RV vs naive)")
    # IT: MINOR-B (audit B1 2026-07-18) — itransformer_regime_moe ammesso: il run
    #     A3 scrive in {models_root}/itransformer_regime_moe (sandbox models_a3_moe);
    #     senza la choice il giudice non può puntare alla dir corretta.
    #     2026-07-19: stesso fix per itransformer_a8_mixup (sandbox models_a8_mixup,
    #     run B3) — aggiunto EX-ANTE, prima di qualsiasi training A8.
    # EN: MINOR-B (B1 audit 2026-07-18) — itransformer_regime_moe allowed: the A3
    #     run writes to {models_root}/itransformer_regime_moe (models_a3_moe
    #     sandbox); without the choice the judge cannot target the right dir.
    #     2026-07-19: same fix for itransformer_a8_mixup (models_a8_mixup sandbox,
    #     B3 run) — added EX-ANTE, before any A8 training.
    #     2026-07-29: stesso fix EX-ANTE per itransformer_a10_sparsity (pre-reg A10
    #     del 28/07, sandbox models_a10_sparsity) — aggiunto prima di scrivere il
    #     primo checkpoint, così il giudice non blocca il run a GPU già spesa.
    # EN: 2026-07-29: same EX-ANTE fix for itransformer_a10_sparsity (A10 pre-reg of
    #     28/07, models_a10_sparsity sandbox) — added before the first checkpoint
    #     exists, so the judge cannot block the run after the GPU is already spent.
    ap.add_argument("--arch", default="itransformer",
                    choices=["itransformer", "nhits", "tcnmamba", "lstm",
                             "itransformer_regime_moe", "itransformer_a8_mixup",
                             "itransformer_a10_sparsity"],
                    help="architettura del modello vol da caricare (models/{arch}) / "
                         "vol model architecture to load (models/{arch})")
    args = ap.parse_args()

    # IT: C3 (pre-reg STATUS 2026-07-31) — guard di COMBINAZIONE dei flag, anticipato
    #     qui prima di caricare npz (~3 GB) e checkpoint: una combinazione incoerente
    #     deve costare zero, non due minuti di I/O. HAR-C riusa il frame di HAR-CJ,
    #     quindi da solo non ha nulla da stimare.
    # EN: C3 (STATUS 2026-07-31 pre-reg) — flag COMBINATION guard, hoisted here before
    #     loading the npz (~3 GB) and the checkpoints: an inconsistent combination must
    #     cost zero, not two minutes of I/O. HAR-C reuses HAR-CJ's frame, so on its own
    #     it has nothing to estimate.
    if (os.environ.get("QUANTSYS_HAR_C", "0") == "1"
            and os.environ.get("QUANTSYS_HAR_CJ", "0") != "1"):
        raise RuntimeError(
            "C3: QUANTSYS_HAR_C=1 richiede QUANTSYS_HAR_CJ=1 (HAR-C riusa il frame "
            "HAR-CJ: stesse colonne xc_*, stesso campione) / QUANTSYS_HAR_C=1 "
            "requires QUANTSYS_HAR_CJ=1 (HAR-C reuses the HAR-CJ frame)"
        )

    # IT: root env-aware (QUANTSYS_MODELS_ROOT) — giudica la sandbox isolata se attiva.
    # EN: env-aware root (QUANTSYS_MODELS_ROOT) — judges the isolated sandbox if set.
    model_dir = models_root() / args.arch
    log.info(f"dir modelli effettiva / effective model dir: {model_dir} (arch={args.arch})")

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config features.target_type deve essere log_rv per il giudice vol-S"

    # IT: orizzonte e risoluzione barra dalla config (H modulo = solo fallback) —
    #     il giudice resta coerente con il dataset/target correnti senza hardcode.
    # EN: horizon and bar resolution from the config (module H = fallback only) —
    #     keeps the judge consistent with the current dataset/target, no hardcoding.
    h = int(cfg["features"].get("forecast_horizon", H))
    interval = cfg["data"]["interval"]
    bars_day = 1440 // interval_minutes_from_cfg(cfg)  # IT: barre/giorno dall'interval | EN: bars/day from interval
    log.info(f"horizon h={h} barre · interval={interval} · bars/day={bars_day}")

    # IT: split da giudicare — val di default (val-first); test SOLO a sanity val superata.
    # EN: split to judge — val by default (val-first); test ONLY once val sanity passes.
    split = os.environ.get("QUANTSYS_VOLS_SPLIT", "val")
    assert split in ("val", "test")

    # IT: C1 (pre-reg STATUS 2026-07-28) — leva della correzione di smearing di Duan
    #     (1983) sul giudice. INERTE di default: a flag spento non si stima nulla, non
    #     si esegue la passata di inferenza sul train e il path numerico è quello
    #     storico bit-per-bit (le metriche `res`/`gate` non vengono MAI toccate dalla
    #     correzione: il blocco smeared vive in una sezione separata del report).
    #     Non è una leva di modello ma un controllo di SPECIFICAZIONE del giudice.
    # EN: C1 (STATUS 2026-07-28 pre-reg) — Duan (1983) smearing-correction lever on the
    #     judge. INERT by default: with the flag off nothing is estimated, the train
    #     inference pass is skipped and the numeric path is the historical one
    #     bit-for-bit (the correction NEVER touches `res`/`gate`: the smeared block
    #     lives in a separate report section). This is a judge SPECIFICATION check,
    #     not a model lever.
    smearing = os.environ.get("QUANTSYS_QLIKE_SMEARING", "0") == "1"
    if smearing:
        log.info("C1 SMEARING ATTIVO / ACTIVE (QUANTSYS_QLIKE_SMEARING=1): "
                 "ŝ stimato SOLO su train, riportato accanto ai valori raw / "
                 "ŝ estimated on TRAIN only, reported alongside the raw values")

    # ── Ground truth + feature HAR dai raw candles (stessa definizione del target) ──
    raw = pd.read_parquet("data/raw_candles.parquet")
    raw = raw.sort_values("open_time").reset_index(drop=True)
    lr2 = np.log(raw["close"] / raw["close"].shift(1)) ** 2

    rv_h = lr2.rolling(h).sum()                      # IT: RV trailing su h barre | EN: trailing h-bar RV
    rv_w = lr2.rolling(7 * bars_day).sum() / 7       # IT: media giornaliera su 7gg | EN: 7d daily mean
    rv_m = lr2.rolling(30 * bars_day).sum() / 30     # IT: media giornaliera su 30gg | EN: 30d daily mean
    rv_fwd = rv_h.shift(-h)                          # IT/EN: target = rolling già calcolato, shiftato (A-minor)

    har = pd.DataFrame({
        "open_time": raw["open_time"],
        "y":  np.log(rv_fwd + EPS),
        "xh": np.log(rv_h + EPS),
        # IT: componenti weekly/monthly riscalate all'orizzonte h per coerenza dimensionale
        # EN: weekly/monthly components rescaled to the h-bar horizon for dimensional consistency
        "xw": np.log(rv_w * (h / bars_day) + EPS),
        "xm": np.log(rv_m * (h / bars_day) + EPS),
    }).dropna().set_index("open_time")

    # ── Allineamento ai timestamp degli split del dataset NN ────────────────────
    # IT: stesso npz env-aware del training (QUANTSYS_DATASET_NPZ, default invariato).
    # EN: same env-aware npz as training (QUANTSYS_DATASET_NPZ, default unchanged).
    d = np.load(str(dataset_npz_path()), allow_pickle=True)
    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)
    t_eval  = pd.to_datetime(d[f"t_{split}"]).tz_localize(None)
    har.index = pd.to_datetime(har.index).tz_localize(None)

    tr = har.loc[har.index.intersection(t_train)]
    ev = har.loc[har.index.intersection(t_eval)]
    log.info(f"HAR rows: train {len(tr)}/{len(t_train)}  {split} {len(ev)}/{len(t_eval)}")
    assert len(ev) >= 0.95 * len(t_eval), "allineamento HAR↔split insufficiente"

    # ── Baseline 1: HAR-RV (OLS chiuso, fit su train) ───────────────────────────
    Xtr = np.column_stack([np.ones(len(tr)), tr[["xh", "xw", "xm"]].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["y"].values, rcond=None)
    Xev = np.column_stack([np.ones(len(ev)), ev[["xh", "xw", "xm"]].values])
    log_pred_har = Xev @ beta
    log.info(f"HAR beta: const={beta[0]:.3f} h={beta[1]:.3f} w={beta[2]:.3f} m={beta[3]:.3f}")

    # ── Baseline 2: naive persistence ───────────────────────────────────────────
    log_pred_naive = ev["xh"].values

    # ── Baseline 3 (C2, pre-reg STATUS 2026-07-30): HAR-CJ, env-gated e INERTE ──
    # IT: a flag spento non viene costruito nulla e il report resta bit-identico —
    #     l'unica differenza possibile è la PRESENZA del blocco `har_cj`. Il frame CJ
    #     nasce SEPARATO (la bipower variation porta un lag in più: fonderlo nel frame
    #     HAR-RV prima del dropna sposterebbe di una barra il campione della baseline
    #     storica, cioè cambierebbe il claim mentre si cerca di verificarlo) e viene
    #     poi ristretto ESATTAMENTE ai timestamp di `ev`: il disegno appaiato della
    #     pre-reg ③ richiede che l'unica cosa a cambiare fra i due rapporti sia il
    #     denominatore. Se l'allineamento non è esatto si fallisce subito: un confronto
    #     su campioni diversi sarebbe peggio di nessun confronto.
    # EN: with the flag off nothing is built and the report stays bit-identical — the
    #     only possible difference is the PRESENCE of the `har_cj` block. The CJ frame
    #     is built SEPARATELY (bipower variation carries one extra lag: merging it into
    #     the HAR-RV frame before dropna would shift the historical baseline's sample by
    #     one bar, i.e. change the claim while trying to verify it) and is then narrowed
    #     to EXACTLY the `ev` timestamps: pre-reg ③'s paired design requires the
    #     denominator to be the only thing that changes between the two ratios. If the
    #     alignment is not exact we fail fast: comparing on different samples would be
    #     worse than not comparing at all.
    har_cj_on = os.environ.get("QUANTSYS_HAR_CJ", "0") == "1"
    # ── Baseline 4 (C3, pre-reg STATUS 2026-07-31): HAR-C, sotto-leva di C2 ─────
    # IT: sole componenti continue (`xc_*`), senza i termini di salto. Riusa lo
    #     STESSO frame CJ — stesse colonne, stesso dropna, stessi timestamp — quindi
    #     l'identità del campione fra le tre baseline è garantita per costruzione e
    #     non da un secondo allineamento da verificare. Fail-fast se acceso da solo:
    #     senza il frame CJ non c'è nulla da stimare, e un flag che tace quando non
    #     può funzionare è peggio di un errore.
    # EN: continuous components only (`xc_*`), no jump terms. It reuses the SAME CJ
    #     frame — same columns, same dropna, same timestamps — so sample identity
    #     across the three baselines holds by construction rather than through a
    #     second alignment check. Fail-fast if switched on alone: without the CJ
    #     frame there is nothing to estimate, and a flag that stays silent when it
    #     cannot work is worse than an error.
    #     La coerenza della combinazione è già stata verificata a inizio `main()`.
    #     Consistency of the combination is already checked at the top of `main()`.
    har_c_on = os.environ.get("QUANTSYS_HAR_C", "0") == "1"
    log_pred_har_cj = None
    log_pred_har_c = None
    if har_cj_on:
        from quantsys.model.vol_metrics import build_har_cj_frame            # noqa: PLC0415
        log.info("C2 HAR-CJ ATTIVO / ACTIVE (QUANTSYS_HAR_CJ=1): baseline aggiuntiva, "
                 "gate pre-registrato del vol-S INVARIATO / additional baseline, "
                 "the pre-registered vol-S gate is UNCHANGED")
        har_cj = build_har_cj_frame(raw, h, bars_day)
        tr_cj = har_cj.loc[har_cj.index.intersection(tr.index)]
        ev_cj = har_cj.loc[har_cj.index.intersection(ev.index)]
        # IT: guard sull'IDENTITÀ dell'indice, non sul solo conteggio: due indici della
        #     stessa lunghezza ma di ordine diverso accoppierebbero previsioni e verità
        #     sbagliate producendo un QLIKE plausibile ma falso — il tipo di errore che
        #     non si vede guardando il numero. (Audit 2026-07-30: verificato che nel run
        #     reale gli indici erano identici elemento per elemento; il guard debole non
        #     aveva prodotto danni, ma non lo avrebbe intercettato.)
        # EN: guard on index IDENTITY, not just count: two indices of equal length but
        #     different order would pair predictions with the wrong ground truth, giving
        #     a plausible-looking but false QLIKE — the kind of error you cannot spot by
        #     looking at the number. (2026-07-30 audit: verified the real run had
        #     element-wise identical indices; the weak guard did no harm but would not
        #     have caught it.)
        if not ev_cj.index.equals(ev.index):
            raise RuntimeError(
                f"C2: allineamento HAR-CJ↔HAR-RV non esatto sull'eval "
                f"({len(ev_cj)} vs {len(ev)} righe, indici {'di pari lunghezza ma diversi' if len(ev_cj)==len(ev) else 'di lunghezza diversa'}) "
                f"— il disegno appaiato della pre-reg ③ richiede gli STESSI sample nello "
                f"STESSO ordine / non-exact HAR-CJ↔HAR-RV eval alignment"
            )
        Xtr_cj = np.column_stack([np.ones(len(tr_cj)), tr_cj[HAR_CJ_COLS].values])
        beta_cj, *_ = np.linalg.lstsq(Xtr_cj, tr_cj["y"].values, rcond=None)
        Xev_cj = np.column_stack([np.ones(len(ev_cj)), ev_cj[HAR_CJ_COLS].values])
        log_pred_har_cj = Xev_cj @ beta_cj
        log.info(f"HAR-CJ beta: const={beta_cj[0]:.3f} "
                 f"C[h,w,m]=({beta_cj[1]:.3f},{beta_cj[2]:.3f},{beta_cj[3]:.3f}) "
                 f"J[h,w,m]=({beta_cj[4]:.3f},{beta_cj[5]:.3f},{beta_cj[6]:.3f})  "
                 f"train={len(tr_cj)}")

        # IT: C3 — stessa meccanica su 3 colonne invece di 6, sullo STESSO train e
        #     sullo STESSO eval: fra i due modelli cambia solo il set di regressori.
        # EN: C3 — same mechanics on 3 columns instead of 6, on the SAME train and
        #     the SAME eval: only the regressor set differs between the two models.
        if har_c_on:
            log.info("C3 HAR-C ATTIVO / ACTIVE (QUANTSYS_HAR_C=1): baseline a sole "
                     "componenti continue, ANNIDATA in HAR-CJ → confronto informativo "
                     "solo fuori campione / continuous-only baseline, NESTED in HAR-CJ "
                     "→ informative out of sample only")
            Xtr_c = np.column_stack([np.ones(len(tr_cj)), tr_cj[HAR_C_COLS].values])
            beta_c, *_ = np.linalg.lstsq(Xtr_c, tr_cj["y"].values, rcond=None)
            Xev_c = np.column_stack([np.ones(len(ev_cj)), ev_cj[HAR_C_COLS].values])
            log_pred_har_c = Xev_c @ beta_c
            log.info(f"HAR-C beta: const={beta_c[0]:.3f} "
                     f"C[h,w,m]=({beta_c[1]:.3f},{beta_c[2]:.3f},{beta_c[3]:.3f})  "
                     f"train={len(tr_cj)}")

    # ── NN: ensemble forward su X_{split} → z → log-RV raw (center+scale) ───────
    from quantsys.model.ensemble import EnsembleModel
    from quantsys.utils import PipelineState
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnsembleModel.load(str(model_dir), device)
    ps = PipelineState.load(str(model_dir / "pipeline_state.pkl"))
    idx = ps.scale_cols.index("target_ret")
    c, s = float(ps.scaler.center_[idx]), float(ps.scaler.scale_[idx])
    log.info(f"target_ret scaler: center={c:.3f} scale={s:.3f} (deve essere ~log-RV, non ~0)")
    assert c < -3, "center ≈ 0 → il PipelineState non è del dataset log-RV (stale?)"

    X = torch.tensor(d[f"X_{split}"], dtype=torch.float32)
    Xm = torch.tensor(d[f"X_macro_{split}"], dtype=torch.float32) if f"X_macro_{split}" in d.files else None

    # IT: A3 regime-MoE — se il config.json del modello dichiara head_type=
    #     "regime_moe", costruisci il gate causale allineato ai timestamp dello
    #     split e passalo come g= (EnsembleModel inoltra i kwargs ai membri).
    #     Chiave assente (tutti i modelli storici) → ramo inerte, call bit-identica.
    # EN: A3 regime-MoE — if the model's config.json declares head_type=
    #     "regime_moe", build the causal gate aligned to the split timestamps and
    #     pass it as g= (EnsembleModel forwards kwargs to the members).
    #     Key absent (all legacy models) → inert branch, bit-identical call.
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

    # IT: forward dell'ensemble in batch da 256 → μ in spazio z. Estratto in funzione
    #     (C1) per riusare lo STESSO batching sul train split quando serve stimare ŝ:
    #     operazioni e ordine invariati rispetto al loop inline storico → bit-identico.
    #     EnsembleModel.__call__ → (mu_ens, sigma_ens, nu_ens) in spazio z (AMP off interno).
    # EN: ensemble forward in 256-sized batches → μ in z-space. Extracted into a function
    #     (C1) to reuse the SAME batching on the train split when ŝ must be estimated:
    #     operations and order unchanged vs the historical inline loop → bit-identical.
    #     EnsembleModel.__call__ → (mu_ens, sigma_ens, nu_ens) in z-space (AMP off internally).
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
    # IT: inversione COMPLETA z→raw: μ·IQR + centro (vedi nota in testa).
    # EN: FULL z→raw inversion: μ·IQR + center (see header note).
    log_pred_nn_full = mu_z * s + c

    # IT: riallinea le predizioni NN ai soli timestamp presenti in `ev`.
    # EN: re-align NN predictions to the timestamps present in `ev`.
    pos = {ts: k for k, ts in enumerate(t_eval)}
    sel = np.array([pos[ts] for ts in ev.index])
    log_pred_nn = log_pred_nn_full[sel]

    # ── Giudizio ────────────────────────────────────────────────────────────────
    rv_true = np.exp(ev["y"].values)  # IT/EN: = rv_fwd + EPS
    res = {}
    losses = {}  # IT: loss QLIKE per-campione, per il DM | EN: per-sample QLIKE losses, for DM
    for name, lp in [("nn", log_pred_nn), ("har", log_pred_har), ("naive", log_pred_naive)]:
        losses[name] = qlike_series(rv_true, np.exp(lp))
        res[name] = {
            "qlike":   qlike(rv_true, np.exp(lp)),
            "mse_log": float(np.mean((ev["y"].values - lp) ** 2)),
        }
        log.info(f"{name:6s} QLIKE={res[name]['qlike']:.5f}  MSE(log)={res[name]['mse_log']:.4f}")

    # IT: INFERENZA sul confronto (Diebold-Mariano, aggiunto 2026-07-26) — il rapporto
    #     di QLIKE è una stima PUNTUALE: senza HAC lo standard error è sottostimato di
    #     ~sqrt(h) perché il target somma h barre (finestre sovrapposte). DESCRITTIVO,
    #     NON GATING: le condizioni di PASS pre-registrate restano quelle sui rapporti
    #     di QLIKE (0.95·HAR e < naive) e NON vengono toccate da questi p-value.
    #     Fail-soft: un errore qui non invalida il verdetto aggregato.
    # EN: INFERENCE on the comparison (Diebold-Mariano, added 2026-07-26) — the QLIKE
    #     ratio is a POINT estimate: without HAC the standard error is understated by
    #     ~sqrt(h) because the target sums h bars (overlapping windows). DESCRIPTIVE,
    #     NOT GATING: the pre-registered PASS conditions remain the QLIKE ratios
    #     (0.95·HAR and < naive) and are untouched by these p-values.
    #     Fail-soft: an error here does not invalidate the aggregate verdict.
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

    # ── C2: valutazione della baseline HAR-CJ (blocco SEPARATO, mai dentro `res`) ──
    # IT: le quattro condizioni della pre-reg sono CALCOLATE ma NON decisionali qui:
    #     il verdetto stampato resta quello del gate vol-S originale (NN vs HAR-RV e
    #     naive). Tenere il blocco fuori da `res`/`gate` è ciò che rende impossibile,
    #     per costruzione e non per disciplina, contaminare un gate già registrato.
    # EN: the pre-reg's four conditions are COMPUTED but NOT decisional here: the
    #     printed verdict remains the original vol-S gate (NN vs HAR-RV and naive).
    #     Keeping the block outside `res`/`gate` is what makes contaminating an
    #     already-registered gate impossible by construction, not by discipline.
    har_cj_block = {"enabled": bool(har_cj_on)}
    if har_cj_on:
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
            # IT/EN: ① la baseline è davvero più forte? / is the baseline actually stronger?
            "cond1_cj_stronger": bool(q_cj <= q_rv),
            # IT/EN: ② il claim sopravvive al gate originale 0.95 / claim survives the original 0.95 gate
            "cond2_claim_survives": bool(q_nn <= 0.95 * q_cj),
            # IT/EN: ③ lo spostamento della banda è materiale? / is the band shift material?
            "cond3_material": bool(abs(ratio_cj - ratio_rv) >= 0.02),
            # IT/EN: ④ validità campione / sample validity
            "cond4_n_obs": bool(len(ev) >= 5000),
            "diebold_mariano": dm_cj,
        })
        log.info(f"C2 HAR-CJ QLIKE={q_cj:.5f} (HAR-RV {q_rv:.5f})  "
                 f"ratio NN/HAR: RV={ratio_rv:.4f} → CJ={ratio_cj:.4f} "
                 f"(Δ={ratio_cj - ratio_rv:+.4f})")

        # ── C3: sub-blocco HAR-C, ANNIDATO dentro `har_cj` ─────────────────────
        # IT: essendo dentro un blocco già fuori da `metrics`/`gate`, il gate
        #     pre-registrato del vol-S è non contaminabile una seconda volta, per
        #     costruzione. Le due statistiche di decisione della pre-reg ④ sono i
        #     DM appaiati; `phi` è DESCRITTIVA e non gating, perché il suo
        #     denominatore (q_rv − q_cj) è una quantità già osservata il 30/07 —
        #     l'avvertenza di peeking ① vieta di fondarci una soglia.
        # EN: sitting inside a block already outside `metrics`/`gate`, the
        #     pre-registered vol-S gate is uncontaminable a second time, by
        #     construction. Pre-reg ④'s two decision statistics are the paired DM
        #     tests; `phi` is DESCRIPTIVE, not gating, because its denominator
        #     (q_rv − q_cj) is a quantity already observed on 30/07 — peeking
        #     warning ① forbids anchoring a threshold to it.
        if har_c_on:
            losses["har_c"] = qlike_series(rv_true, np.exp(log_pred_har_c))
            q_c = qlike(rv_true, np.exp(log_pred_har_c))
            dm_c = {}
            try:
                # IT/EN: convenzione diebold_mariano(a,b) → stat NEGATIVA = `a` migliore
                dm_c["test_a_har_c_vs_har_rv"] = diebold_mariano(
                    losses["har_c"], losses["har"], h=h)
                dm_c["test_b_har_cj_vs_har_c"] = diebold_mariano(
                    losses["har_cj"], losses["har_c"], h=h)
            except Exception as e:  # noqa: BLE001
                log.warning(f"DM HAR-C fallito / DM HAR-C failed: {e}")
            denom = q_rv - q_cj
            phi = float((q_rv - q_c) / denom) if abs(denom) > 1e-12 else float("nan")
            har_cj_block["har_c"] = {
                "enabled": True,
                "qlike_har_c": float(q_c),
                "mse_log_har_c": float(np.mean((ev["y"].values - log_pred_har_c) ** 2)),
                "beta": [float(b) for b in beta_c],
                # IT/EN: attribuzione — quota del guadagno CJ dalla sola sostituzione
                "phi_attribution": phi,
                # IT: condizionamento dei due design (diagnostico, mai decisionale):
                #     è la quantità che l'audit del 30/07 ha calcolato ad hoc, qui
                #     resa riproducibile — sostiene l'argomento di STABILITÀ dello
                #     strumento, non l'accuratezza.
                # EN: conditioning of the two designs (diagnostic, never decisional):
                #     the quantity the 30/07 audit computed ad hoc, made reproducible
                #     here — it supports the instrument-STABILITY argument, not accuracy.
                "cond_number_har_c": float(np.linalg.cond(Xtr_c)),
                "cond_number_har_cj": float(np.linalg.cond(Xtr_cj)),
                "diebold_mariano": dm_c,
            }
            log.info(f"C3 HAR-C QLIKE={q_c:.5f}  (HAR-RV {q_rv:.5f} · HAR-CJ {q_cj:.5f})  "
                     f"phi={phi:+.3f}  cond: C={np.linalg.cond(Xtr_c):.1f} "
                     f"CJ={np.linalg.cond(Xtr_cj):.3e}")
            for label in ("test_a_har_c_vs_har_rv", "test_b_har_cj_vs_har_c"):
                r = dm_c.get(label)
                if r:
                    log.info(f"DM {label}: stat={r['dm_hln']:+.3f} p={r['p_value']:.2e} "
                             f"(migliore/better={r['better']})")

    # IT: firma di inerzia di C3: a flag spento il report guadagna SOLO questa chiave,
    #     in coda (l'ordine delle chiavi C2 resta quello storico). setdefault → no-op
    #     quando il sub-blocco è già stato scritto sopra.
    # EN: C3's inertia signature: with the flag off the report gains ONLY this key, at
    #     the end (C2's key order stays historical). setdefault → no-op when the
    #     sub-block was already written above.
    har_cj_block.setdefault("har_c", {"enabled": bool(har_c_on)})

    # IT: A8-bis (pre-reg 2026-07-19) — breakdown QLIKE per-regime: label = argmax del
    #     gate causale (model-independent), allineate ai sample `ev` via `sel`. Serve
    #     alla condizione ② dei gate per-regime; fail-soft con warning (il verdetto
    #     aggregato NON dipende da questo blocco).
    # EN: A8-bis (2026-07-19 pre-reg) — per-regime QLIKE breakdown: labels = argmax of
    #     the causal gate (model-independent), aligned to the `ev` samples via `sel`.
    #     Feeds per-regime gate condition ②; fail-soft with a warning (the aggregate
    #     verdict does NOT depend on this block).
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

    # ── C1: correzione di smearing di Duan (1983), SIMMETRICA sui due lati ──────
    # IT: pre-reg STATUS 2026-07-28. ŝ = media di exp(ε) sui residui in log del solo
    #     TRAIN, applicato in valutazione come RV_corr = exp(log_pred)·ŝ. Entrambi i
    #     lati del confronto (NN e HAR) sono corretti con il PROPRIO fattore: correggere
    #     un lato solo è escluso ex-ante dalla pre-registrazione. `naive` è descrittivo.
    #     Le condizioni ①②③ di adozione sono calcolate qui ma NON decidono nulla da
    #     sole: la decisione (e la riscrittura della banda pubblicata) resta manuale.
    #     Il blocco `metrics` e il `gate` pre-registrato del 2026-06-10 restano INTATTI.
    # EN: STATUS 2026-07-28 pre-reg. ŝ = mean of exp(ε) over the TRAIN-only log
    #     residuals, applied at evaluation as RV_corr = exp(log_pred)·ŝ. BOTH sides of
    #     the comparison (NN and HAR) get their OWN factor: correcting one side only is
    #     excluded ex-ante by the pre-registration. `naive` is descriptive. Adoption
    #     conditions ①②③ are computed here but decide nothing on their own: the
    #     decision (and rewriting the published band) stays manual. The `metrics` block
    #     and the pre-registered 2026-06-10 `gate` are left UNTOUCHED.
    smear_block = {"enabled": bool(smearing)}
    if smearing:
        # IT: ŝ_HAR e ŝ_naive dai residui IN-SAMPLE di train (stesso campione dell'OLS).
        # EN: ŝ_HAR and ŝ_naive from the IN-SAMPLE train residuals (same sample as the OLS).
        eps_har = tr["y"].values - Xtr @ beta
        eps_naive = tr["y"].values - tr["xh"].values

        # IT: ŝ_NN — passata di inferenza sul TRAIN split (~52k finestre × 5 membri).
        #     `from_numpy` (zero-copy) invece di `tensor` per non duplicare i 2.6 GB;
        #     tensori liberati subito dopo. Residuo in log: (y_z − μ_z)·scale (il centro
        #     si cancella nella differenza — stessa inversione del giudice).
        # EN: ŝ_NN — inference pass over the TRAIN split (~52k windows × 5 members).
        #     `from_numpy` (zero-copy) rather than `tensor` to avoid duplicating 2.6 GB;
        #     tensors freed right after. Log residual: (y_z − μ_z)·scale (the center
        #     cancels in the difference — same inversion as the judge).
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
            # IT: ① coerenza di specificazione su ENTRAMBI i lati | EN: ① specification coherence on BOTH sides
            "cond1_coherent_both_sides": bool(
                metrics_smear["nn"]["qlike"] <= res["nn"]["qlike"]
                and metrics_smear["har"]["qlike"] <= res["har"]["qlike"]),
            # IT: ② materialità ≥0.02 di ratio | EN: ② materiality ≥0.02 of ratio
            "cond2_material": bool(abs(ratio_smear - ratio_raw) >= 0.02),
            # IT: ③ validità campione (non-leakage è strutturale nel codice) | EN: ③ sample validity
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
    # IT: report suffissato per interval+split — run a risoluzioni diverse non si sovrascrivono.
    #     2026-07-19: + suffisso sandbox quando QUANTSYS_MODELS_ROOT è attivo — il run
    #     candidato e quello incumbent non si clobberano più (visto su B2/B3: il report
    #     del candidato sopravviveva solo nei log). Path production (no env) INVARIATO.
    # EN: report suffixed by interval+split — runs at different resolutions do not overwrite.
    #     2026-07-19: + sandbox suffix when QUANTSYS_MODELS_ROOT is set — candidate and
    #     incumbent runs no longer clobber each other (seen on B2/B3: the candidate
    #     report only survived in the logs). Production path (no env) UNCHANGED.
    _root = models_root()
    _sandbox = f"_{_root.name}" if _root.name != "models" else ""
    out_path = out_dir / f"qlike_report_{interval}_{split}{_sandbox}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": res, "gate": gate, "per_regime": per_regime,
                   "har_beta": list(map(float, beta)), "smearing": smear_block,
                   "har_cj": har_cj_block}, f, indent=2)

    print(f"\n══════ VOL-S QLIKE [{interval}·{split}] ══════")
    for name in ("nn", "har", "naive"):
        print(f"  {name:6s} QLIKE={res[name]['qlike']:.5f}  MSE(log)={res[name]['mse_log']:.4f}")
    print(f"  NN/HAR ratio: {gate['nn_vs_har_ratio']:.4f}  (gate ≤ 0.95)")
    # IT: inferenza descrittiva (non gating) — vedi blocco DM sopra.
    # EN: descriptive inference (not gating) — see the DM block above.
    for label in ("nn_vs_har", "nn_vs_naive"):
        r = res.get("diebold_mariano", {}).get(label)
        if r and np.isfinite(r.get("dm_hln", float("nan"))):
            print(f"  DM {label:12s} stat={r['dm_hln']:+7.3f}  p={r['p_value']:.2e}"
                  f"  (HAC lag={r['hac_lag']}, n_eff≈{r['n_eff']:.0f}) [descrittivo/descriptive]")
    # IT: C1 — stampa raw-vs-smeared affiancati; le tre condizioni sono INFORMATIVE
    #     (la decisione di adozione è manuale, con il vincolo anti-goalpost).
    # EN: C1 — prints raw-vs-smeared side by side; the three conditions are INFORMATIVE
    #     (the adoption decision is manual, under the anti-goalpost constraint).
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
    # IT: C2 — HAR-CJ affiancata a HAR-RV; le quattro condizioni sono INFORMATIVE, la
    #     decisione è manuale sotto il vincolo anti-goalpost della pre-reg.
    # EN: C2 — HAR-CJ shown next to HAR-RV; the four conditions are INFORMATIVE, the
    #     decision is manual under the pre-reg's anti-goalpost constraint.
    if har_cj_block["enabled"]:
        print(f"  ── C2 baseline HAR-CJ (Andersen-Bollerslev-Diebold 2007) ──")
        print(f"     HAR-RV QLIKE={har_cj_block['qlike_har_rv']:.5f} → "
              f"HAR-CJ QLIKE={har_cj_block['qlike_har_cj']:.5f}  "
              f"MSE(log)={har_cj_block['mse_log_har_cj']:.4f}")
        print(f"     ratio NN/baseline: vs RV={har_cj_block['ratio_rv']:.4f} → "
              f"vs CJ={har_cj_block['ratio_cj']:.4f}  (Δ={har_cj_block['delta_ratio']:+.4f})")
        for label in ("nn_vs_har_cj", "har_cj_vs_har"):
            r = har_cj_block.get("diebold_mariano", {}).get(label)
            if r and np.isfinite(r.get("dm_hln", float("nan"))):
                print(f"     DM {label:14s} stat={r['dm_hln']:+7.3f}  p={r['p_value']:.2e}"
                      f"  [descrittivo/descriptive]")
        print(f"     ① CJ più forte/stronger={har_cj_block['cond1_cj_stronger']}  "
              f"② claim regge/survives(≤0.95)={har_cj_block['cond2_claim_survives']}  "
              f"③ materiale/material(≥0.02)={har_cj_block['cond3_material']}  "
              f"④ n≥5000={har_cj_block['cond4_n_obs']}")
    # IT/EN: C3 — attribuzione del guadagno CJ / attribution of the CJ gain
    _hc = har_cj_block.get("har_c", {})
    if _hc.get("enabled"):
        print(f"  [C3] HAR-C (sole componenti continue/continuous only) "
              f"QLIKE={_hc['qlike_har_c']:.5f}  MSE(log)={_hc['mse_log_har_c']:.4f}")
        print(f"     ordinamento/ordering: HAR-RV={har_cj_block['qlike_har_rv']:.5f} · "
              f"HAR-C={_hc['qlike_har_c']:.5f} · HAR-CJ={har_cj_block['qlike_har_cj']:.5f}"
              f"   φ={_hc['phi_attribution']:+.3f}")
        print(f"     cond(design): HAR-C={_hc['cond_number_har_c']:.1f}  "
              f"HAR-CJ={_hc['cond_number_har_cj']:.3e}  [diagnostico/diagnostic]")
        for label, tag in (("test_a_har_c_vs_har_rv", "A  C vs RV "),
                           ("test_b_har_cj_vs_har_c", "B  CJ vs C ")):
            r = _hc.get("diebold_mariano", {}).get(label)
            if r and np.isfinite(r.get("dm_hln", float("nan"))):
                print(f"     Test {tag} DM={r['dm_hln']:+7.3f}  p={r['p_value']:.2e}  "
                      f"migliore/better={r['better']}  "
                      f"{'SIG' if r['p_value'] < 0.01 else 'non sig.'} (α=0.01)")
    print(f"  VERDETTO [{split}]: {gate['verdict']}   → {out_path}")


if __name__ == "__main__":
    main()
