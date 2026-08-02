"""
Probe temporanea (PERF AUDIT) — cronometra FeatureBuilder.build() step per step.
Temporary probe (PERF AUDIT) — times FeatureBuilder.build() step by step.

Nessuna modifica al codice di produzione: gli step vengono avvolti a runtime
sull'ISTANZA (monkeypatch locale), non sulla classe.
No production code is modified: steps are wrapped at runtime on the INSTANCE
(local monkeypatch), not on the class.

Uso / Usage:  python scripts/archive/perf_probe/bench_featurebuilder.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantsys.utils import load_config, interval_minutes_from_cfg  # noqa: E402
from quantsys.features import FeatureBuilder  # noqa: E402


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config(str(ROOT / "config/default.yaml"))
    fcfg = cfg["features"]

    raw = pd.read_parquet(ROOT / "data/raw_candles.parquet")
    funding = None
    fp = ROOT / "data/funding_rate.parquet"
    if fp.exists():
        funding = pd.read_parquet(fp)
    print(f"raw candles: {len(raw):,}  cols={list(raw.columns)}")

    b = FeatureBuilder(
        vp_bins=fcfg["vp_bins"],
        vp_lookback=fcfg["vp_lookback"],
        windows=fcfg["windows"],
        lag_periods=fcfg["lag_periods"],
        forecast_horizon=fcfg.get("forecast_horizon", 1),
        vp_stride=fcfg.get("vp_stride", 1),
        frac_diff_d=fcfg.get("frac_diff_d", 0.0),
        use_revin=bool(cfg["model"].get("use_revin", False)),
        interval_minutes=interval_minutes_from_cfg(cfg),
        target_type=fcfg.get("target_type", "ret"),
        use_har_cj=bool(fcfg.get("har_cj", False)),
    )

    # IT: wrappa i metodi-step sull'istanza per misurare il wall-clock di ciascuno.
    # EN: wrap step methods on the instance to measure each one's wall-clock.
    timings = {}
    step_names = [
        "_returns", "_vwap", "_technicals", "_volume_features", "_cvd_features",
        "_volatility", "_time_features", "_lags", "_frac_diff",
        "_structural_features", "_volume_profile", "_funding_features",
        "_normalize", "fit_scaler_only",
    ]
    for name in step_names:
        orig = getattr(b, name)

        def make(orig=orig, name=name):
            def wrapped(*a, **kw):
                t0 = time.perf_counter()
                out = orig(*a, **kw)
                timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0)
                return out
            return wrapped
        setattr(b, name, make())

    t0 = time.perf_counter()
    df = b.build(raw, normalize=False, fit=False, funding_df=funding)
    t_build = time.perf_counter() - t0

    n_total = len(df)
    train_end = int(n_total * (1 - cfg["training"]["val_fraction"] - cfg["training"]["test_fraction"]))
    t1 = time.perf_counter()
    b.fit_scaler_only(df.iloc[:train_end])
    df = b._normalize(df, fit=False)
    t_norm = time.perf_counter() - t1

    print(f"\n=== FeatureBuilder.build() — {len(raw):,} barre ===")
    print(f"build() totale        : {t_build:8.2f} s")
    print(f"scaler fit+transform  : {t_norm:8.2f} s")
    acc = 0.0
    for k, v in sorted(timings.items(), key=lambda x: -x[1]):
        acc += v
        print(f"  {k:24s} {v:8.2f} s   ({v/(t_build+t_norm)*100:5.1f}%)")
    print(f"  {'[somma step]':24s} {acc:8.2f} s")
    print(f"  {'[overhead/copy/altro]':24s} {t_build + t_norm - acc:8.2f} s")
    print(f"\nrighe valide: {len(df):,}  colonne: {df.shape[1]}")


if __name__ == "__main__":
    main()
