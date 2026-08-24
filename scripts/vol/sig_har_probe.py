# IT: SignatureHAR — sonda pre-registrata (STATUS.md, gate 2026-08-25).
#     Questo file implementa il passo EX-ANTE del gate: la CONDIZIONE ③, cioè la
#     verifica di ridondanza analitica del termine di ordine 2. Sul percorso
#     lead-lag il termine antisimmetrico di livello 2 (area di Lévy) coincide con
#     la variazione quadratica della finestra, che HAR-C porta già come `xc_h`:
#     se la ridondanza c'è, il ramo ordine 2 si chiude SENZA stimare nulla.
#     SOLA LETTURA: legge `data/raw_candles.parquet` e l'npz degli split, non
#     scrive nulla. `scripts/vol/dev_vols_qlike.py` NON viene toccato (giudica
#     campioni forward pre-registrati aperti).
# EN: SignatureHAR — pre-registered probe (STATUS.md, 2026-08-25 gate).
#     This file implements the gate's EX-ANTE step: CONDITION ③, the analytic
#     redundancy check of the order-2 term. On the lead-lag path the level-2
#     antisymmetric term (Lévy area) equals the window's quadratic variation,
#     which HAR-C already carries as `xc_h`: if redundancy holds, the order-2 arm
#     closes WITHOUT estimating anything.
#     READ-ONLY: reads `data/raw_candles.parquet` and the split npz, writes
#     nothing. `scripts/vol/dev_vols_qlike.py` is NOT touched (it judges open
#     pre-registered forward samples).
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging, load_config, interval_minutes_from_cfg, dataset_npz_path  # noqa: E402
from quantsys.model.vol_metrics import build_har_cj_frame, HAR_C_COLS, EPS  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.sig_har_probe")


# IT: area di Lévy del percorso LEAD-LAG costruito su una sequenza di rendimenti,
#     calcolata per via NUMERICA sui segmenti (non assunta): per un percorso
#     lineare a tratti in 2-D i due integrali iterati di livello 2 sono
#     S12 = Σ_k [A_k·b_k + a_k·b_k/2] e S21 = Σ_k [B_k·a_k + a_k·b_k/2], con A/B
#     somme parziali degli incrementi; l'area è (S12 − S21)/2. Il percorso
#     lead-lag alterna un passo "lead" (a=r_k, b=0) e un passo "lag" (a=0, b=r_k),
#     quindi i termini misti a_k·b_k sono nulli segmento per segmento.
#     Restituisce 2·area, cioè il termine di variazione quadratica della pre-reg.
#     Vettorizzato sulle N finestre; il loop è sulle W barre, non sui dati.
# EN: Lévy area of the LEAD-LAG path built from a return sequence, computed
#     NUMERICALLY over segments (not assumed): for a piecewise-linear 2-D path the
#     two level-2 iterated integrals are S12 = Σ_k [A_k·b_k + a_k·b_k/2] and
#     S21 = Σ_k [B_k·a_k + a_k·b_k/2], with A/B the partial sums of increments;
#     the area is (S12 − S21)/2. The lead-lag path alternates a "lead" step
#     (a=r_k, b=0) and a "lag" step (a=0, b=r_k), so the mixed a_k·b_k terms vanish
#     segment by segment. Returns 2·area, i.e. the pre-registration's
#     quadratic-variation term. Vectorised over the N windows; the loop runs over
#     the W bars, not over the data.
def leadlag_qv_term(rw: np.ndarray) -> np.ndarray:
    n, w = rw.shape
    s12 = np.zeros(n)
    s21 = np.zeros(n)
    a = np.zeros(n)   # IT: somma parziale componente lead | EN: lead partial sum
    b = np.zeros(n)   # IT: somma parziale componente lag  | EN: lag partial sum
    for k in range(w):
        r = rw[:, k]
        # IT: passo lead — incremento (r, 0): contribuisce solo a S21, via B
        # EN: lead step — increment (r, 0): contributes to S21 only, via B
        s21 += b * r
        a += r
        # IT: passo lag — incremento (0, r): contribuisce solo a S12, via A
        # EN: lag step — increment (0, r): contributes to S12 only, via A
        s12 += a * r
        b += r
    return s12 - s21          # IT/EN: = 2 · area di Lévy / 2 · Lévy area


# IT: matrice (N, W) delle finestre CAUSALI di rendimenti che terminano in t.
#     Riga i = [r_{i−W+1}, …, r_i]; le prime W−1 righe restano NaN e cadono a valle.
# EN: (N, W) matrix of CAUSAL return windows ending at t. Row i = [r_{i−W+1}, …,
#     r_i]; the first W−1 rows stay NaN and are dropped downstream.
def rolling_windows(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.full((n, w), np.nan)
    for k in range(w):
        lag = w - 1 - k
        out[lag:, k] = x[:n - lag] if lag > 0 else x
    return out


# IT: R² della colonna candidata regredita sui regressori HAR-C (costante inclusa).
#     È la misura di ridondanza rilevante per un OLS: quanta parte della colonna
#     nuova è già nello span del disegno incumbent.
# EN: R² of the candidate column regressed on the HAR-C regressors (intercept
#     included). This is the redundancy measure that matters for an OLS: how much
#     of the new column already lies in the incumbent design's span.
def r2_on_design(y: np.ndarray, x: np.ndarray) -> float:
    xd = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(xd, y, rcond=None)
    resid = y - xd @ beta
    return float(1.0 - resid.var() / y.var())


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="SignatureHAR — condizione ③ ex-ante: ridondanza del termine di ordine 2 / "
                    "SignatureHAR — ex-ante condition ③: order-2 term redundancy")
    # IT: finestre da testare — 24 è quella CONGELATA nella pre-reg, 30 è quella
    #     appaiata a `xc_h` (h = forecast_horizon). Si riportano entrambe: uno
    #     scarto di ρ fra le due è attribuibile alla finestra, non al meccanismo.
    # EN: windows to test — 24 is the one FROZEN in the pre-registration, 30 is the
    #     one matched to `xc_h` (h = forecast_horizon). Both are reported: a gap in
    #     ρ between them is attributable to the window, not to the mechanism.
    ap.add_argument("--windows", default="24,30",
                    help="finestre W separate da virgola / comma-separated W")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    h = int(cfg["features"].get("forecast_horizon", 30))
    bars_day = 1440 // interval_minutes_from_cfg(cfg)
    log.info(f"h={h} barre · bars/day={bars_day} · interval={cfg['data']['interval']}")

    raw = pd.read_parquet("data/raw_candles.parquet").sort_values("open_time").reset_index(drop=True)
    lr = np.log(raw["close"] / raw["close"].shift(1)).values

    # IT: frame HAR-C dal modulo condiviso — IDENTICO a quello del giudice, così
    #     `xc_h` è esattamente il regressore incumbent, non una sua ricostruzione.
    # EN: HAR-C frame from the shared module — IDENTICAL to the judge's, so `xc_h`
    #     is exactly the incumbent regressor, not a reconstruction of it.
    har_cj = build_har_cj_frame(raw, h, bars_day)

    # IT: split TRAIN dall'npz: la condizione ③ si verifica sul train, nessun
    #     contatto con val (che è lo split del gate).
    # EN: TRAIN split from the npz: condition ③ is checked on train, with no
    #     contact with val (the gate's split).
    d = np.load(str(dataset_npz_path()), allow_pickle=True)
    t_train = pd.to_datetime(d["t_train"]).tz_localize(None)

    for w in [int(v) for v in args.windows.split(",")]:
        rw = rolling_windows(lr, w)
        qv = leadlag_qv_term(rw)

        # IT: controllo di IDENTITÀ — il termine calcolato sui segmenti deve
        #     coincidere con Σ r² sulla finestra. Se non coincide, il bug è nella
        #     costruzione del percorso e il resto del numero non vale nulla.
        # EN: IDENTITY check — the segment-computed term must equal Σ r² over the
        #     window. If it does not, the bug is in the path construction and the
        #     rest of the number is worthless.
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

        # IT: tre letture di ρ, dichiarate PRIMA di guardarle, più l'R² sul disegno.
        # EN: three ρ readings, declared BEFORE looking at them, plus the design R².
        rho_log = float(np.corrcoef(logqv, xch)[0, 1])                      # scale-matched
        rho_lvl = float(np.corrcoef(qv_v, xch)[0, 1])                       # letterale / literal
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

        # IT: DIAGNOSTICA della condizione ③ — se ρ vs `xc_h` non raggiunge 0.99, la
        #     domanda è CONTRO COSA il termine è ridondante. `xc_h = log(C_h)` con
        #     C = RV − J: HAR-C (C3, 31/07) ha DELIBERATAMENTE tolto i salti. Il
        #     termine di ordine 2 è la variazione quadratica, cioè RV = C + J, quindi
        #     due controlli: (a) ρ contro log(RV_h) sulla stessa finestra — deve
        #     essere ≈ 1 per identità; (b) quanto del residuo di log(QV) sul disegno
        #     HAR-C è spiegato dalla colonna dei salti `xj_h`.
        # EN: condition ③ DIAGNOSTIC — if ρ vs `xc_h` falls short of 0.99, the
        #     question is WHAT the term is redundant against. `xc_h = log(C_h)` with
        #     C = RV − J: HAR-C (C3, 31/07) DELIBERATELY dropped the jumps. The
        #     order-2 term is the quadratic variation, i.e. RV = C + J, hence two
        #     checks: (a) ρ against log(RV_h) on the same window — must be ≈ 1 by
        #     identity; (b) how much of log(QV)'s residual on the HAR-C design is
        #     explained by the jump column `xj_h`.
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
