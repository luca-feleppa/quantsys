"""
IT: STEP 0 (de-risk mixture-of-universes) — diagnostico SENZA training.
    Domanda: lo spread di NLL per-regime (r0/r1/r2) e' dovuto a sigma MISCALIBRATA per regime
    (fixabile da una testa-sigma regime-condizionata) o a errore-mu irriducibile (non fixabile)?
    Metodo: predizioni del modello esistente su VAL -> per ogni regime calcola NLL t-Student,
    lo std del residuo standardizzato z=(y-mu)/sigma (calibration ratio ~ sigma ottimale), e
    confronta NLL con scaling sigma GLOBALE vs PER-REGIME (oracolo). Se il per-regime batte il
    globale di un margine sensibile -> sigma regime-condizionata ha valore -> procedi con la mixture.

EN: STEP 0 de-risk — training-free diagnostic. Is the per-regime NLL spread due to per-regime
    sigma MIScalibration (fixable by a regime-conditioned sigma head) or irreducible mu-error?
    Compare t-Student NLL under GLOBAL vs PER-REGIME (oracle) sigma scaling on the val set.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as t_dist

os.environ.setdefault("QUANTSYS_ARCH", "itransformer")
from quantsys.utils import load_config, setup_device
from quantsys.model.ensemble import EnsembleModel

ARCH = os.environ.get("QUANTSYS_ARCH", "itransformer")


def _to_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return s.dt.tz_convert("UTC").dt.tz_localize(None) if s.dt.tz is not None else s


def nll(y, mu, sigma, nu, lam=1.0):
    # t-Student NLL per-sample: -[logpdf(z; nu) - log(s)], z=(y-mu)/s, s=lam*sigma
    s = lam * sigma
    z = (y - mu) / s
    return -(t_dist.logpdf(z, df=nu) - np.log(s))


def best_lambda(y, mu, sigma, nu, grid):
    vals = [nll(y, mu, sigma, nu, lam=g).mean() for g in grid]
    j = int(np.argmin(vals))
    return float(grid[j]), float(vals[j])


def main():
    cfg = load_config()
    device = setup_device(cfg)

    data = np.load("data/lstm_dataset.npz", allow_pickle=True)
    X = data["X_val"]
    y = data["y_val"].astype(np.float64).reshape(-1)
    t_val = data["t_val"]
    print(f"VAL: {len(X)} campioni | arch={ARCH}")

    # ---- inferenza (z-score, pre-denorm; AMP off) ----
    model = EnsembleModel.load(f"models/{ARCH}", device)
    model.eval()
    n = len(X)
    mu = np.zeros(n); sg = np.zeros(n); nu = np.zeros(n)
    with torch.no_grad():
        for i in range(0, n, 256):
            xb = torch.tensor(X[i:i + 256], dtype=torch.float32).to(device)
            mb, sb, nb = model(xb)
            mu[i:i + 256] = mb.squeeze(-1).cpu().numpy()
            sg[i:i + 256] = np.maximum(sb.squeeze(-1).cpu().numpy(), 1e-9)
            nu[i:i + 256] = np.clip(nb.squeeze(-1).cpu().numpy(), 2.1, 100.0)

    # ---- allinea regime (causale, merge_asof backward) ----
    tv = pd.DataFrame({"i": np.arange(n), "open_time": _to_naive(pd.Series(pd.to_datetime(t_val)))})
    tv = tv.sort_values("open_time")
    reg = pd.read_parquet("data/regime_probs.parquet").reset_index()
    reg["open_time"] = _to_naive(reg["open_time"])
    reg = reg.sort_values("open_time")
    merged = pd.merge_asof(tv, reg[["open_time", "regime_dominant"]], on="open_time",
                           direction="backward")
    merged = merged.sort_values("i")
    regime = merged["regime_dominant"].to_numpy()

    names = {0: "R0 Quiet", 1: "R1 Trend", 2: "R2 Stress"}
    # IT: griglia ampia (floor basso) — la sigma del modello e' globalmente troppo grande
    #     (std(z)<<1), gli ottimi cadono sotto 0.5; un floor a 0.5 li clampa e maschera lo spread.
    # EN: wide grid (low floor) — model sigma is globally too large (std(z)<<1), optima fall below
    #     0.5; a 0.5 floor clamps them and hides the per-regime spread.
    grid = np.linspace(0.15, 3.0, 286)

    # ---- baseline + scaling globale + scaling per-regime (oracolo) ----
    base_all = nll(y, mu, sg, nu).mean()
    lam_g, nll_g_all = best_lambda(y, mu, sg, nu, grid)

    print("\n" + "=" * 78)
    print("STEP 0 — CALIBRAZIONE SIGMA REGIME-CONDIZIONATA (val, no training)")
    print("=" * 78)
    print(f"{'regime':<11}{'n':>7}{'NLL base':>11}{'std(z)':>9}"
          f"{'lam*reg':>9}{'NLL@reg':>10}{'NLL@glob':>10}")
    print("-" * 78)

    nll_perreg_total = 0.0
    nll_glob_total = 0.0
    base_total = 0.0
    spread_stdz = []
    for k in (0, 1, 2):
        m = regime == k
        nk = int(m.sum())
        if nk == 0:
            continue
        yk, muk, sgk, nuk = y[m], mu[m], sg[m], nu[m]
        base_k = nll(yk, muk, sgk, nuk).mean()
        stdz_k = float(np.std((yk - muk) / sgk))
        lam_k, nll_reg_k = best_lambda(yk, muk, sgk, nuk, grid)
        nll_glob_k = nll(yk, muk, sgk, nuk, lam=lam_g).mean()
        spread_stdz.append(stdz_k)
        nll_perreg_total += nll_reg_k * nk
        nll_glob_total += nll_glob_k * nk
        base_total += base_k * nk
        print(f"{names[k]:<11}{nk:>7}{base_k:>11.4f}{stdz_k:>9.3f}"
              f"{lam_k:>9.3f}{nll_reg_k:>10.4f}{nll_glob_k:>10.4f}")

    nll_perreg_total /= n
    nll_glob_total /= n
    base_total /= n
    print("-" * 78)
    print(f"{'OVERALL':<11}{n:>7}{base_total:>11.4f}{'':>9}{'':>9}"
          f"{nll_perreg_total:>10.4f}{nll_glob_total:>10.4f}")

    gain = nll_glob_total - nll_perreg_total
    stdz_spread = (max(spread_stdz) - min(spread_stdz)) if spread_stdz else 0.0
    print("\n" + "-" * 78)
    print(f"  lambda globale ottimale            : {lam_g:.3f}")
    print(f"  std(z) spread tra regimi           : {stdz_spread:.3f}  "
          f"(0=sigma gia' ben calibrata per-regime)")
    print(f"  NLL globale-scaled                 : {nll_glob_total:.4f}")
    print(f"  NLL per-regime-scaled (oracolo)    : {nll_perreg_total:.4f}")
    print(f"  GUADAGNO regime-cond. su sigma     : {gain:.4f} nats/campione")
    print("-" * 78)
    # IT: regola di decisione (euristica): guadagno >= 0.02 nats e spread std(z) >= 0.15
    #     indicano sigma regime-miscalibrata fixabile -> la mixture ha valore.
    if gain >= 0.02 and stdz_spread >= 0.15:
        print("  VERDETTO: sigma e' REGIME-MISCALIBRATA -> la testa-sigma regime ha valore. PROCEDI.")
    elif gain >= 0.02:
        print("  VERDETTO: guadagno presente ma spread std(z) basso -> valore modesto. Valutare.")
    else:
        print("  VERDETTO: guadagno trascurabile -> lo spread NLL e' mu-error irriducibile, "
              "NON fixabile da sigma. La mixture non aiuta la calibrazione: NON procedere.")
    print("=" * 78)


if __name__ == "__main__":
    main()
