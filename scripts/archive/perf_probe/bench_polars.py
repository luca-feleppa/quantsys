"""
Probe temporanea (PERF AUDIT) — lever (d): Polars vs pandas nel FeatureBuilder.
Temporary probe (PERF AUDIT) — lever (d): Polars vs pandas in the FeatureBuilder.

Porta 3 gruppi di feature RAPPRESENTATIVI dei pattern realmente usati e misura
(1) lo speedup e (2) lo SCARTO NUMERICO rispetto a pandas sulle stesse colonne.
Ports 3 feature groups REPRESENTATIVE of the patterns actually used and measures
(1) speedup and (2) the NUMERIC DELTA vs pandas on the same columns.

  A) rolling std/mean/var su log_ret        (riduzioni rolling)
  B) VWAP: groupby(day).cumsum + rolling sum (groupby + cumulative)
  C) rolling skew/kurt su log_ret            (momenti superiori)

NB: NON importa torch — su questa macchina torch+sklearn prima di pyarrow
manda in access violation il caricamento parquet (vedi docs/PERF_AUDIT.md).
NB: does NOT import torch — on this box torch+sklearn before pyarrow crashes
the parquet load with an access violation (see docs/PERF_AUDIT.md).

Uso / Usage: python scripts/archive/perf_probe/bench_polars.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]


def timeit(fn, n=7):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    return (time.perf_counter() - t0) / n, out


def delta(a: np.ndarray, b: np.ndarray, name: str):
    """Scarto numerico pandas vs polars, ignorando i NaN comuni."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() == 0:
        print(f"    {name:<18} nessun valore comparabile")
        return
    d = np.abs(a[m] - b[m])
    scale = np.maximum(np.abs(a[m]), 1e-300)
    rel = d / scale
    n_exact = int((d == 0).sum())
    # IT: ULP = distanza in rappresentazioni float64 consecutive.
    # EN: ULP = distance in consecutive float64 representations.
    ulp = d / np.spacing(np.abs(a[m]))
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    print(f"    {name:<18} |Δ|max={d.max():.3e}  rel_max={rel.max():.3e}  "
          f"ULP_max={ulp.max():8.1f}  bit-uguali={n_exact}/{m.sum()} "
          f"({n_exact/m.sum()*100:5.1f}%)  NaN-mismatch={nan_mismatch}")


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        import polars as pl
    except ImportError:
        print("polars non installato: pip install polars")
        return

    print(f"pandas {pd.__version__}  |  polars {pl.__version__}\n")

    raw = pd.read_parquet(ROOT / "data/raw_candles.parquet")
    n = len(raw)
    print(f"dataset: {n:,} barre 1h\n")

    pdf = raw[["open_time", "high", "low", "close", "volume"]].copy()
    pdf["log_ret"] = np.log(pdf["close"] / pdf["close"].shift(1))
    pdf["typical_price"] = (pdf["high"] + pdf["low"] + pdf["close"]) / 3
    pdf["pv"] = pdf["typical_price"] * pdf["volume"]
    pdf["date_utc"] = pdf["open_time"].dt.date
    ldf = pl.from_pandas(pdf)

    # ── A) rolling std / mean / var ─────────────────────────────────────────
    def a_pd():
        o = {}
        for w in (5, 10, 20, 60):
            o[f"vol_std_{w}"] = pdf["log_ret"].rolling(w).std().to_numpy()
            o[f"vol_mean_{w}"] = pdf["log_ret"].rolling(w).mean().to_numpy()
        o["realized_var_20"] = (pdf["log_ret"] ** 2).rolling(20).mean().to_numpy()
        return o

    def a_pl():
        exprs = []
        for w in (5, 10, 20, 60):
            exprs.append(pl.col("log_ret").rolling_std(w).alias(f"vol_std_{w}"))
            exprs.append(pl.col("log_ret").rolling_mean(w).alias(f"vol_mean_{w}"))
        exprs.append((pl.col("log_ret") ** 2).rolling_mean(20).alias("realized_var_20"))
        r = ldf.select(exprs)
        return {c: r[c].to_numpy() for c in r.columns}

    t_pd, r_pd = timeit(a_pd)
    t_pl, r_pl = timeit(a_pl)
    print(f"A) rolling std/mean/var (9 colonne)")
    print(f"    pandas {t_pd*1000:7.2f} ms | polars {t_pl*1000:7.2f} ms "
          f"| speedup {t_pd/t_pl:5.2f}×")
    for k in ("vol_std_20", "vol_mean_20", "realized_var_20"):
        delta(r_pd[k], r_pl[k], k)

    # ── B) VWAP: groupby cumsum + rolling sum ───────────────────────────────
    def b_pd():
        g = pdf.groupby("date_utc")
        cum_pv = g["pv"].cumsum()
        cum_vol = g["volume"].cumsum()
        vwap = cum_pv / cum_vol.replace(0, np.nan)
        rpv = pdf["pv"].rolling(20).sum()
        rv = pdf["volume"].rolling(20).sum()
        vwap20 = rpv / rv.replace(0, np.nan)
        return {"vwap": vwap.to_numpy(), "vwap_20": vwap20.to_numpy()}

    def b_pl():
        r = ldf.select([
            (pl.col("pv").cum_sum().over("date_utc")
             / pl.col("volume").cum_sum().over("date_utc")).alias("vwap"),
            (pl.col("pv").rolling_sum(20)
             / pl.col("volume").rolling_sum(20)).alias("vwap_20"),
        ])
        return {c: r[c].to_numpy() for c in r.columns}

    t_pd, r_pd = timeit(b_pd)
    t_pl, r_pl = timeit(b_pl)
    print(f"\nB) VWAP groupby-cumsum + rolling sum")
    print(f"    pandas {t_pd*1000:7.2f} ms | polars {t_pl*1000:7.2f} ms "
          f"| speedup {t_pd/t_pl:5.2f}×")
    for k in ("vwap", "vwap_20"):
        delta(r_pd[k], r_pl[k], k)

    # ── C) rolling skew / kurt ──────────────────────────────────────────────
    def c_pd():
        return {"ret_skew_20": pdf["log_ret"].rolling(20).skew().to_numpy()}

    def c_pl():
        r = ldf.select(pl.col("log_ret").rolling_skew(20).alias("ret_skew_20"))
        return {"ret_skew_20": r["ret_skew_20"].to_numpy()}

    t_pd, r_pd = timeit(c_pd)
    t_pl, r_pl = timeit(c_pl)
    print(f"\nC) rolling skew (20)")
    print(f"    pandas {t_pd*1000:7.2f} ms | polars {t_pl*1000:7.2f} ms "
          f"| speedup {t_pd/t_pl:5.2f}×")
    delta(r_pd["ret_skew_20"], r_pl["ret_skew_20"], "ret_skew_20")
    print("\n    NB: rolling_skew di polars usa la definizione BIASED (n),")
    print("        pandas usa quella UNBIASED (n-1 Fisher) → differenza")
    print("        SISTEMATICA, non di arrotondamento.")


if __name__ == "__main__":
    main()
