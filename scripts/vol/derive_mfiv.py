# IT: D4 — DERIVAZIONE OFFLINE MFIV@30h + SKEW 25Δ DAL RAW CHAIN (CPU-only,
#     retroattiva su tutto il periodo di raccolta 01c). Risposta strutturale al
#     FAIL del gate v1: la IV ATM interpolata (atm_30h) SOTTOSTIMA il var-swap
#     rate per la convessità dello smile → il vero comparatore dell'edge di 04b
#     è il tasso model-free (replica VIX-style del variance swap, Carr-Madan /
#     CBOE). Output PARALLELO (data/iv/mfiv_30h.parquet): MAI nel path
#     decisionale di 04b — l'eventuale promozione a comparatore = NUOVA pre-reg
#     v2 con break-even ri-stimato (wedge di convessità MFIV vs IV ATM).
#     Metodo per (snapshot_ts, expiry):
#       F      = mediana di underlying_price del gruppo (forward del venue,
#                più autoritativo della parity sui mark);
#       K0     = strike massimo ≤ F;
#       OTM    = put K<K0 + call K>K0 + media call/put a K0 (CBOE);
#       prezzi USD: Q = mark_price(BTC) × F (opzioni inverse quotate in BTC);
#       σ²_ann = (2/T)·Σ ΔK/K²·Q − (1/T)·(F/K0−1)²   (r=0, convenzione crypto);
#       tenor 30h: interpolazione LINEARE IN VARIANZA TOTALE (σ²_ann·T) tra le
#       due expiry che bracketano 30h (stessa convenzione di atm_30h in 01c);
#       skew:  smile in delta-space dai mark_iv (delta forward BS: N(d1), N(d1)−1)
#              → IV a Δcall=+0.25 e Δput=−0.25 (interp. lineare in delta) e IV
#              ATM a K=F (interp. in ln(K/F)) → RR25 = ivC−ivP,
#              BF25 = ½(ivC+ivP)−ivATM, interpolati in T a 30h.
#     Filtri qualità: mark_price>0, mark_iv>0, ≥3 strike OTM per lato; niente
#     truncation-rule sui bid (i mark Deribit sono model-smooth by design —
#     scelta dichiarata). Incrementale: gli snapshot già presenti nell'output
#     vengono saltati (--force = full rebuild). Convenzione annualizzazione:
#     8760h/anno (Deribit 365d, identica a 04b).
# EN: D4 — OFFLINE MFIV@30h + 25Δ SKEW DERIVATION FROM THE RAW CHAIN (CPU-only,
#     retroactive over the whole 01c collection). Structural response to the v1
#     gate FAIL: interpolated ATM IV (atm_30h) UNDERSTATES the var-swap rate by
#     the smile convexity → 04b's true edge comparator is the model-free rate
#     (VIX-style variance-swap replication, Carr-Madan / CBOE). PARALLEL output
#     (data/iv/mfiv_30h.parquet): NEVER in 04b's decision path — any promotion
#     to comparator = NEW v2 pre-reg with re-estimated break-even (MFIV vs ATM
#     IV convexity wedge). Method per (snapshot_ts, expiry): venue forward F =
#     median underlying_price; K0 = max strike ≤ F; CBOE OTM selection; USD
#     prices Q = BTC mark × F (inverse options); σ²_ann = (2/T)·Σ ΔK/K²·Q −
#     (1/T)·(F/K0−1)² with r=0; 30h tenor = LINEAR-IN-TOTAL-VARIANCE
#     interpolation across the two bracketing expiries (same convention as
#     atm_30h in 01c); 25Δ smile from mark_iv via forward BS deltas → RR/BF
#     interpolated in T to 30h. Quality filters: positive mark/IV, ≥3 OTM
#     strikes per side; no bid truncation rule (Deribit marks are model-smooth
#     by design — declared choice). Incremental: snapshots already in the
#     output are skipped (--force = full rebuild). Annualization: 8760h/year
#     (Deribit 365d, identical to 04b).
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.utils import setup_logging                       # noqa: E402
from quantsys.utils.atomic_save import atomic_save_parquet     # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.derive_mfiv")

ROOT = Path(__file__).resolve().parents[2]
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
OUT_PATH = ROOT / "data" / "iv" / "mfiv_30h.parquet"

HOURS_PER_YEAR = 8760.0     # IT: convenzione Deribit 365g (come 04b) | EN: Deribit 365d convention (as 04b)
TENOR_HOURS = 30.0
MIN_OTM_PER_SIDE = 3        # IT: sotto, l'integrale è troppo rado | EN: below this the integral is too sparse
MAX_TENOR_DAYS = 8.0        # IT: bastano le expiry corte per bracketare 30h | EN: short expiries suffice to bracket 30h


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # IT: CDF normale via erf (niente scipy nel critical path degli script).
    # EN: normal CDF via erf (no scipy in the scripts' critical path).
    from math import erf, sqrt
    return np.array([0.5 * (1.0 + erf(v / sqrt(2.0))) for v in np.asarray(x, dtype=float)])


def mfiv_one_expiry(grp: pd.DataFrame, t_years: float) -> dict | None:
    # IT: replica var-swap CBOE su UNA expiry: ritorna σ²_ann + diagnostica,
    #     None se il gruppo non passa i filtri qualità.
    # EN: CBOE variance-swap replication on ONE expiry: returns annualized σ² +
    #     diagnostics, None when the group fails the quality filters.
    g = grp[(grp["mark_price"] > 0) & (grp["mark_iv"] > 0)]
    if g.empty:
        return None
    F = float(g["underlying_price"].median())
    if not np.isfinite(F) or F <= 0:
        return None
    strikes = np.sort(g["strike"].unique())
    below = strikes[strikes <= F]
    if below.size == 0:
        return None
    K0 = float(below.max())

    # IT: selezione OTM (put sotto K0, call sopra, media a K0) in USD: Q=mark·F.
    # EN: OTM selection (puts below K0, calls above, K0 average) in USD: Q=mark·F.
    puts = g[(g["option_type"] == "P") & (g["strike"] < K0)]
    calls = g[(g["option_type"] == "C") & (g["strike"] > K0)]
    if len(puts) < MIN_OTM_PER_SIDE or len(calls) < MIN_OTM_PER_SIDE:
        return None
    rows = []
    for df_side in (puts, calls):
        for _, r in df_side.iterrows():
            rows.append((float(r["strike"]), float(r["mark_price"]) * F))
    at0 = g[g["strike"] == K0]
    q0 = float(at0["mark_price"].mean()) * F   # IT: media C/P se entrambe | EN: C/P average when both
    rows.append((K0, q0))
    rows.sort()
    K = np.array([r[0] for r in rows])
    Q = np.array([r[1] for r in rows])

    # IT: ΔK centrale (one-sided ai bordi), integrale 2/T Σ ΔK/K² Q − correzione K0.
    # EN: central ΔK (one-sided at the edges), 2/T Σ ΔK/K² Q integral − K0 correction.
    dK = np.empty_like(K)
    dK[1:-1] = (K[2:] - K[:-2]) / 2.0
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    var_ann = (2.0 / t_years) * float(np.sum(dK / K ** 2 * Q)) \
        - (1.0 / t_years) * (F / K0 - 1.0) ** 2
    if var_ann <= 0:
        return None

    # IT: smile 25Δ dai mark_iv (delta forward BS; put OTM per Δput, call OTM per Δcall).
    # EN: 25Δ smile from mark_iv (forward BS deltas; OTM puts for Δput, OTM calls for Δcall).
    def _iv_at_delta(side: pd.DataFrame, target: float, is_call: bool) -> float:
        iv = side["mark_iv"].values / 100.0
        sqt = np.sqrt(t_years)
        d1 = (np.log(F / side["strike"].values) + 0.5 * iv ** 2 * t_years) / (iv * sqt)
        delta = _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0
        order = np.argsort(delta)
        return float(np.interp(target, delta[order], side["mark_iv"].values[order]))

    def _iv_atm() -> float:
        gg = g.groupby("strike")["mark_iv"].mean()
        x = np.log(gg.index.values / F)
        order = np.argsort(x)
        return float(np.interp(0.0, x[order], gg.values[order]))

    try:
        iv_c25 = _iv_at_delta(calls, 0.25, is_call=True)
        iv_p25 = _iv_at_delta(puts, -0.25, is_call=False)
        iv_atm = _iv_atm()
        rr25 = iv_c25 - iv_p25
        bf25 = 0.5 * (iv_c25 + iv_p25) - iv_atm
    except Exception:
        rr25 = bf25 = iv_atm = np.nan

    return {"F": F, "K0": K0, "var_ann": var_ann, "n_otm": len(K),
            "rr25": rr25, "bf25": bf25, "iv_atm_smile": iv_atm}


def process_snapshot(snap: pd.DataFrame, ts: pd.Timestamp) -> dict | None:
    # IT: MFIV per expiry corta + interpolazione in varianza TOTALE a 30h
    #     (bracketing; None se non bracketabile). RR/BF interpolati in T.
    # EN: per-short-expiry MFIV + TOTAL-variance interpolation at 30h
    #     (bracketing; None when not bracketable). RR/BF interpolated in T.
    per_exp = []
    for exp, grp in snap.groupby("expiry"):
        t_h = (pd.Timestamp(exp) - ts).total_seconds() / 3600.0
        if not (1.0 <= t_h <= MAX_TENOR_DAYS * 24.0):
            continue
        r = mfiv_one_expiry(grp, t_h / HOURS_PER_YEAR)
        if r is not None:
            r["t_hours"] = t_h
            per_exp.append(r)
    below = [r for r in per_exp if r["t_hours"] <= TENOR_HOURS]
    above = [r for r in per_exp if r["t_hours"] > TENOR_HOURS]
    if not below or not above:
        return None
    lo = max(below, key=lambda r: r["t_hours"])
    hi = min(above, key=lambda r: r["t_hours"])

    # IT: varianza TOTALE (σ²_ann·T_anni) lineare in T → tasso var-swap 30h.
    # EN: TOTAL variance (ann σ²·T_years) linear in T → 30h var-swap rate.
    tv_lo = lo["var_ann"] * lo["t_hours"] / HOURS_PER_YEAR
    tv_hi = hi["var_ann"] * hi["t_hours"] / HOURS_PER_YEAR
    w = (TENOR_HOURS - lo["t_hours"]) / (hi["t_hours"] - lo["t_hours"])
    tv_30h = tv_lo + w * (tv_hi - tv_lo)
    if tv_30h <= 0:
        return None
    var_ann_30h = tv_30h * HOURS_PER_YEAR / TENOR_HOURS

    def _interp(key: str) -> float:
        a, b = lo[key], hi[key]
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.nan
        return a + w * (b - a)

    return {"timestamp": ts,
            "mfiv_30h": float(np.sqrt(var_ann_30h) * 100.0),   # IT: vol % ann | EN: ann vol %
            "mfiv_var_total_30h": float(tv_30h),               # IT: confrontabile con var_iv di 04b | EN: comparable to 04b's var_iv
            "rr25_30h": _interp("rr25"), "bf25_30h": _interp("bf25"),
            "iv_atm_smile_30h": _interp("iv_atm_smile"),
            "tenor_lo_h": lo["t_hours"], "tenor_hi_h": hi["t_hours"],
            "n_otm_lo": lo["n_otm"], "n_otm_hi": hi["n_otm"]}


def main():
    # IT: boilerplate UTF-8 console Windows (checklist CLAUDE.md — bug cp1252).
    # EN: Windows console UTF-8 boilerplate (CLAUDE.md checklist — cp1252 bug).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Derivazione offline MFIV@30h + skew 25Δ / "
                                             "offline MFIV@30h + 25Δ skew derivation")
    ap.add_argument("--force", action="store_true",
                    help="full rebuild (ignora l'output esistente) / full rebuild")
    args = ap.parse_args()

    done: set = set()
    old = None
    if OUT_PATH.exists() and not args.force:
        old = pd.read_parquet(OUT_PATH)
        done = set(pd.to_datetime(old["timestamp"], utc=True))
        log.info(f"output esistente: {len(old)} righe — modalità incrementale / "
                 f"existing output: {len(old)} rows — incremental mode")

    rows = []
    files = sorted(CHAIN_DIR.glob("btc_options_*.parquet"))
    log.info(f"{len(files)} file chain da processare / chain files to process")
    for f in files:
        df = pd.read_parquet(f, columns=["snapshot_ts", "expiry", "strike",
                                         "option_type", "mark_price", "mark_iv",
                                         "underlying_price"])
        n_new = 0
        for ts, snap in df.groupby("snapshot_ts"):
            ts = pd.Timestamp(ts)
            if ts in done:
                continue
            r = process_snapshot(snap, ts)
            if r is not None:
                rows.append(r)
                n_new += 1
        log.info(f"{f.name}: {n_new} snapshot derivati / snapshots derived")

    if not rows and old is None:
        log.warning("nessuno snapshot derivabile / no derivable snapshot")
        return
    new = pd.DataFrame(rows)
    out = (pd.concat([old, new], ignore_index=True) if old is not None else new)
    out = (out.drop_duplicates(subset="timestamp", keep="first")
              .sort_values("timestamp").reset_index(drop=True))
    atomic_save_parquet(out, OUT_PATH, index=False)
    log.info(f"→ {OUT_PATH} ({len(out)} righe totali, +{len(new)} nuove / "
             f"total rows, new)")

    # IT: sanity vs IV ATM del poller: il wedge di convessità DEVE essere ≥0 in
    #     mediana (MFIV ≥ ATM da disuguaglianza di Jensen sullo smile).
    # EN: sanity vs the poller's ATM IV: the convexity wedge MUST be ≥0 in the
    #     median (MFIV ≥ ATM by Jensen's inequality over the smile).
    atm_path = ROOT / "data" / "iv" / "atm_30h.parquet"
    if atm_path.exists() and len(out):
        atm = pd.read_parquet(atm_path)[["timestamp", "iv_30h"]].copy()
        atm["timestamp"] = pd.to_datetime(atm["timestamp"], utc=True)
        m = pd.merge_asof(out.sort_values("timestamp"), atm.sort_values("timestamp"),
                          on="timestamp", tolerance=pd.Timedelta("15min"),
                          direction="nearest").dropna(subset=["iv_30h"])
        if len(m):
            wedge = m["mfiv_30h"] - m["iv_30h"] * 100.0 \
                if m["iv_30h"].median() < 3 else m["mfiv_30h"] - m["iv_30h"]
            log.info(f"wedge convessità MFIV−ATM (vol pt): mediana "
                     f"{wedge.median():+.2f}, p10 {wedge.quantile(0.1):+.2f}, "
                     f"p90 {wedge.quantile(0.9):+.2f} su {len(m)} tick accoppiati")


if __name__ == "__main__":
    main()
