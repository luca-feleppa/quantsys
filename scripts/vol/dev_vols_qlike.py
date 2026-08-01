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
    # IT: 2026-08-01 — via di fuga ESPLICITA al guard sullo scaler (vedi sotto).
    #     Deve essere un flag e non una env: un run cross-vintage produce un numero
    #     NON confrontabile con gli altri report, quindi la sua natura deve restare
    #     scritta nell'invocazione e finire nel report.
    # EN: 2026-08-01 — EXPLICIT escape hatch for the scaler guard (see below). It
    #     must be a flag, not an env var: a cross-vintage run produces a number NOT
    #     comparable with the other reports, so its nature must stay written in the
    #     invocation and end up in the report.
    ap.add_argument("--allow-scaler-mismatch", action="store_true",
                    help="procedi anche se lo scaler del modello ≠ scaler del dataset: "
                         "il numero NON è confrontabile con gli altri report / proceed even "
                         "if the model scaler ≠ dataset scaler: the number is NOT comparable")
    args = ap.parse_args()

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

    # ── Baseline 3 e 4: HAR-C e HAR-CJ — SEMPRE calcolate (adottate 2026-07-31) ──
    # IT: nate come leve env-gated di C2/C3, promosse a parte stabile dello strumento
    #     alla chiusura di C3. I due flag `QUANTSYS_HAR_CJ`/`QUANTSYS_HAR_C` sono stati
    #     RIMOSSI, non accesi: un flag sempre acceso non è una leva, è una via di
    #     fallimento in più (spariscono con essi la combinazione incoerente e il suo
    #     guard). Per riprodurre il report storico pre-C2 si usa il version control —
    #     `git show <commit>:scripts/vol/dev_vols_qlike.py` — che è il meccanismo con
    #     cui le tre prove di inerzia sono state fatte: duplicarlo con un flag di
    #     compatibilità sarebbe reimplementare git dentro l'applicazione.
    #     ⚠ INVARIANTE PRESERVATO: i blocchi restano FUORI da `metrics`/`gate`, quindi
    #     il gate pre-registrato del vol-S (2026-06-10, denominatore HAR-RV) è
    #     strutturalmente non contaminabile — la promozione non lo tocca.
    #     Il frame CJ nasce SEPARATO (la bipower variation porta un lag in più: fonderlo
    #     nel frame HAR-RV prima del dropna sposterebbe di una barra il campione della
    #     baseline storica) e viene poi ristretto ESATTAMENTE ai timestamp di `ev`.
    #     HAR-C riusa quel frame: identità del campione fra le tre baseline garantita
    #     per COSTRUZIONE, non da un allineamento da verificare a valle.
    # EN: born as C2/C3 env-gated levers, promoted to a stable part of the instrument
    #     when C3 closed. The two flags were REMOVED, not switched on: an always-on flag
    #     is not a lever, it is one more failure mode (the inconsistent combination and
    #     its guard disappear with them). To reproduce the historical pre-C2 report use
    #     version control — `git show <commit>:scripts/vol/dev_vols_qlike.py` — which is
    #     how all three inertia proofs were done: duplicating it with a compatibility
    #     flag would be reimplementing git inside the application.
    #     ⚠ INVARIANT PRESERVED: the blocks stay OUTSIDE `metrics`/`gate`, so the
    #     pre-registered vol-S gate (2026-06-10, HAR-RV denominator) is structurally
    #     uncontaminable — the promotion does not touch it.
    #     The CJ frame is built SEPARATELY (bipower variation carries one extra lag) and
    #     is then narrowed to EXACTLY the `ev` timestamps. HAR-C reuses that frame:
    #     sample identity across the three baselines holds BY CONSTRUCTION.
    from quantsys.model.vol_metrics import build_har_cj_frame                # noqa: PLC0415
    har_cj = build_har_cj_frame(raw, h, bars_day)
    tr_cj = har_cj.loc[har_cj.index.intersection(tr.index)]
    ev_cj = har_cj.loc[har_cj.index.intersection(ev.index)]
    # IT: guard sull'IDENTITÀ dell'indice, non sul solo conteggio: due indici della
    #     stessa lunghezza ma di ordine diverso accoppierebbero previsioni e verità
    #     sbagliate producendo un QLIKE plausibile ma falso — il tipo di errore che
    #     non si vede guardando il numero. (Audit 2026-07-30.)
    # EN: guard on index IDENTITY, not just count: two indices of equal length but
    #     different order would pair predictions with the wrong ground truth, giving
    #     a plausible-looking but false QLIKE — the kind of error you cannot spot by
    #     looking at the number. (2026-07-30 audit.)
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

    # IT: HAR-C — stessa meccanica su 3 colonne invece di 6, stesso train, stesso eval:
    #     fra i due modelli cambia SOLO il set di regressori. È la baseline di
    #     riferimento del claim pubblicato dal 2026-07-31 (C3).
    # EN: HAR-C — same mechanics on 3 columns instead of 6, same train, same eval: ONLY
    #     the regressor set differs. It is the published claim's reference baseline
    #     since 2026-07-31 (C3).
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

    # IT: GUARD SCALER MODELLO↔DATASET (2026-08-01). L'assert sopra cattura solo lo
    #     stato grossolanamente sbagliato (center ≈ 0); NON cattura uno stato
    #     plausibile ma di un ALTRO vintage del dataset, che è il caso che si è
    #     verificato davvero: `models/itransformer` è il restore del PASS di giugno
    #     e la sua QLIKE sull'npz corrente (0.27470 val) è ~5% peggiore di quella di
    #     una coppia riaddestrata sullo stesso npz (0.26143) — differenza che è
    #     artefatto di scaler, non skill. Confrontare due modelli attraverso scaler
    #     diversi è vietato dal manifesto; farlo in silenzio è come è successo.
    #     Fail-fast, non warning: un warning in un log da 600 righe non ha fermato
    #     nessuno finora.
    # EN: MODEL↔DATASET SCALER GUARD (2026-08-01). The assert above only catches a
    #     grossly wrong state (center ≈ 0); it does NOT catch a plausible state from
    #     ANOTHER dataset vintage, which is what actually happened:
    #     `models/itransformer` is the June PASS restore and its QLIKE on the current
    #     npz (0.27470 val) is ~5% worse than a pair retrained on that same npz
    #     (0.26143) — a difference that is a scaler artifact, not skill. Comparing
    #     two models across different scalers is forbidden; doing it silently is how
    #     it happened. Fail-fast, not a warning: a warning inside a 600-line log has
    #     stopped nobody so far.
    from quantsys.utils import assert_model_dataset_scaler
    prov = assert_model_dataset_scaler(ps, model_dir=model_dir, arch=args.arch,
                                       npz=dataset_npz_path(),
                                       allow_mismatch=args.allow_scaler_mismatch,
                                       logger=log)

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
    # IT: `enabled` conservato a True per COMPATIBILITÀ di schema con i report già su
    #     disco e con le tabelle committate in STATUS: la chiave non ha più un
    #     interruttore dietro, ma toglierla renderebbe non diffabili i report di C2/C3.
    # EN: `enabled` kept True for schema COMPATIBILITY with the reports already on disk
    #     and the tables committed in STATUS: the key no longer has a switch behind it,
    #     but removing it would make the C2/C3 reports non-diffable.
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
        # IT: le quattro condizioni di C2 — gate CHIUSO il 2026-07-30. Restano calcolate
        #     come monitoraggio continuo (② dice a ogni run se il claim regge ancora
        #     contro la baseline forte), NON come condizioni vive.
        # EN: C2's four conditions — gate CLOSED on 2026-07-30. Kept as continuous
        #     monitoring (② tells every run whether the claim still holds against the
        #     strong baseline), NOT as live conditions.
        "cond1_cj_stronger": bool(q_cj <= q_rv),
        "cond2_claim_survives": bool(q_nn <= 0.95 * q_cj),
        "cond3_material": bool(abs(ratio_cj - ratio_rv) >= 0.02),
        "cond4_n_obs": bool(len(ev) >= 5000),
        "diebold_mariano": dm_cj,
    })

    # ── HAR-C: sub-blocco annidato (schema stabile dai report C3) ──────────────
    # IT: essendo dentro un blocco già fuori da `metrics`/`gate`, il gate
    #     pre-registrato del vol-S è non contaminabile una seconda volta, per
    #     costruzione. `phi` resta DESCRITTIVA: il suo denominatore (q_rv − q_cj) era
    #     una quantità già osservata quando C3 fu pre-registrato, e l'avvertenza di
    #     peeking vietava di fondarci una soglia — la nota resta valida per chiunque
    #     rilegga il numero.
    # EN: sitting inside a block already outside `metrics`/`gate`, the pre-registered
    #     vol-S gate is uncontaminable a second time, by construction. `phi` stays
    #     DESCRIPTIVE: its denominator (q_rv − q_cj) was an already-observed quantity
    #     when C3 was pre-registered, and the peeking warning forbade anchoring a
    #     threshold to it — the note still holds for anyone re-reading the number.
    losses["har_c"] = qlike_series(rv_true, np.exp(log_pred_har_c))
    q_c = qlike(rv_true, np.exp(log_pred_har_c))
    dm_c = {}
    try:
        # IT/EN: convenzione diebold_mariano(a,b) → stat NEGATIVA = `a` migliore
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
        # IT/EN: attribuzione — quota del guadagno CJ dalla sola sostituzione
        "phi_attribution": phi,
        # IT: condizionamento dei due design (diagnostico, mai decisionale): sostiene
        #     l'argomento di STABILITÀ dello strumento, non l'accuratezza. È la ragione
        #     per cui HAR-C, e non HAR-CJ, è la baseline del claim.
        # EN: conditioning of the two designs (diagnostic, never decisional): it supports
        #     the instrument-STABILITY argument, not accuracy. It is why HAR-C, not
        #     HAR-CJ, is the claim's baseline.
        "cond_number_har_c": float(np.linalg.cond(Xtr_c)),
        "cond_number_har_cj": float(np.linalg.cond(Xtr_cj)),
        # IT/EN: rapporto del CLAIM pubblicato (baseline di riferimento dal 2026-07-31)
        "ratio_nn_over_har_c": float(q_nn / q_c),
        "diebold_mariano": dm_c,
    }
    log.info(f"baseline: HAR-RV={q_rv:.5f} (gate) · HAR-C={q_c:.5f} (claim) · "
             f"HAR-CJ={q_cj:.5f}   ratio NN/HAR-C={q_nn / q_c:.4f}  phi={phi:+.3f}")

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
        # IT: `provenance` (2026-08-01) — il report DEVE dire quale modello ha
        #     prodotto `metrics.nn`. Senza, un numero è orfano del modello che lo
        #     ha generato: è esattamente così che i report C3 (NN da
        #     `models/itransformer`) sono stati letti come se contenessero il
        #     numeratore della banda pubblicata (NN da una coppia riaddestrata,
        #     sandbox poi eliminata). Il campione era identico cifra per cifra su
        #     tutte le baseline, quindi nulla segnalava la differenza.
        # EN: `provenance` (2026-08-01) — the report MUST say which model produced
        #     `metrics.nn`. Without it a number is orphaned from the model that
        #     generated it: that is exactly how the C3 reports (NN from
        #     `models/itransformer`) were read as if they held the published band's
        #     numerator (NN from a retrained pair, sandbox later deleted). The
        #     sample was identical digit-for-digit across every baseline, so nothing
        #     flagged the difference.
        json.dump({"metrics": res, "gate": gate, "per_regime": per_regime,
                   "har_beta": list(map(float, beta)), "smearing": smear_block,
                   "har_cj": har_cj_block, "provenance": prov}, f, indent=2)

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
    # IT: pannello a TRE baseline, con il RUOLO di ciascuna stampato accanto al numero.
    #     È la ragione principale per cui il calcolo è stato reso incondizionato: chi
    #     lancia il giudice deve vedere il denominatore del claim pubblicato senza
    #     doversi ricordare di accendere un flag, e deve vedere che NON è lo stesso
    #     denominatore del gate pre-registrato. Confondere i due è l'errore che questo
    #     pannello esiste per rendere impossibile.
    # EN: THREE-baseline panel, each one's ROLE printed next to its number. It is the
    #     main reason the computation was made unconditional: whoever runs the judge
    #     must see the published claim's denominator without remembering to set a flag,
    #     and must see it is NOT the pre-registered gate's denominator. Confusing the
    #     two is the mistake this panel exists to make impossible.
    _hc = har_cj_block["har_c"]
    print("  ── baseline econometriche / econometric baselines ──")
    # IT: in mismatch di scaler i RAPPORTI NN qui sotto non sono confrontabili con
    #     la banda pubblicata, mentre le QLIKE delle baseline lo restano (le HAR
    #     sono fittate e valutate dentro l'npz, quindi indipendenti dal modello).
    #     Senza questa riga l'etichetta «denominatore del CLAIM» starebbe accanto a
    #     un rapporto che NON è quello del claim: il pannello, nato per rendere
    #     impossibile una confusione, ne introdurrebbe un'altra.
    # EN: under a scaler mismatch the NN RATIOS below are not comparable with the
    #     published band, while the baseline QLIKEs are (HARs are fitted and
    #     evaluated inside the npz, hence model-independent). Without this line the
    #     «published CLAIM denominator» label would sit next to a ratio that is NOT
    #     the claim's: the panel, built to make one confusion impossible, would
    #     introduce another.
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
