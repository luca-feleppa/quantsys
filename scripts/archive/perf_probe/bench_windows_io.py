"""
Probe temporanea (PERF AUDIT) — costo di create_windows + I/O del dataset npz.
Temporary probe (PERF AUDIT) — cost of create_windows + npz dataset I/O.

Misura: lettura features.parquet, create_windows (materializza ~3.3 GB),
scrittura npz atomica, rilettura npz (quello che fa 02_train allo start).
Measures: features.parquet read, create_windows (materializes ~3.3 GB),
atomic npz write, npz re-read (what 02_train does at startup).

Uso / Usage: python scripts/archive/perf_probe/bench_windows_io.py [--skip-write]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantsys.utils import load_config  # noqa: E402
from quantsys.features import create_windows, canonical_feature_columns, temporal_split  # noqa: E402

# IT: dir temporanea di sistema — la probe non scrive mai in models/ o data/.
# EN: system temp dir — this probe never writes into models/ or data/.
import tempfile  # noqa: E402
SCRATCH = Path(tempfile.gettempdir())


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    skip_write = "--skip-write" in sys.argv

    cfg = load_config(str(ROOT / "config/default.yaml"))
    mcfg = cfg["model"]

    t0 = time.perf_counter()
    feat = pd.read_parquet(ROOT / "data/features.parquet")
    print(f"read features.parquet     : {time.perf_counter()-t0:7.2f} s  ({len(feat):,} righe)")

    # IT: la lista canonica viene derivata come in 01 (non assunta).
    # EN: canonical list derived as in 01 (not assumed).
    all_cols = [c for c in feat.columns]
    t0 = time.perf_counter()
    cols = canonical_feature_columns(all_cols, feat)
    print(f"canonical_feature_columns : {time.perf_counter()-t0:7.2f} s  ({len(cols)} colonne)")

    t0 = time.perf_counter()
    X, y, t = create_windows(feat, cols, window_size=mcfg["window_size"],
                             window_stride=mcfg.get("window_stride", 1))
    t_win = time.perf_counter() - t0
    print(f"create_windows            : {t_win:7.2f} s  X={X.shape} "
          f"({X.nbytes/1e9:.2f} GB)")

    t0 = time.perf_counter()
    splits = temporal_split(X, y, t, val_frac=cfg["training"]["val_fraction"],
                            test_frac=cfg["training"]["test_fraction"])
    print(f"temporal_split (slicing)  : {time.perf_counter()-t0:7.2f} s")

    if not skip_write:
        out = SCRATCH / "perf_dataset.npz"
        t0 = time.perf_counter()
        np.savez(out, **splits, feature_names=np.array(cols),
                 n_dynamic_features=np.array([0]))
        t_w = time.perf_counter() - t0
        sz = out.stat().st_size / 1e9
        print(f"np.savez (scratch disk)   : {t_w:7.2f} s  ({sz:.2f} GB "
              f"→ {sz/t_w:.2f} GB/s)")

        t0 = time.perf_counter()
        with np.load(out) as d:
            _ = {k: d[k] for k in ("X_train", "X_val", "X_test")}
        print(f"np.load (materializza X)  : {time.perf_counter()-t0:7.2f} s")
        out.unlink(missing_ok=True)

    # IT: rilettura del npz DI PRODUZIONE (quella che paga 02_train allo start).
    # EN: re-read of the PRODUCTION npz (what 02_train pays at startup).
    prod = ROOT / "data/lstm_dataset.npz"
    if prod.exists():
        t0 = time.perf_counter()
        with np.load(prod, allow_pickle=True) as d:
            keys = list(d.files)
            arrs = {k: d[k] for k in keys}
        t_l = time.perf_counter() - t0
        tot = sum(a.nbytes for a in arrs.values()) / 1e9
        print(f"np.load produzione        : {t_l:7.2f} s  ({tot:.2f} GB "
              f"→ {tot/t_l:.2f} GB/s)  keys={keys}")


if __name__ == "__main__":
    main()
