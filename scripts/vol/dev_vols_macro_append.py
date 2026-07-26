# IT: VOL-S utility — ri-appende X_macro_* a data/lstm_dataset.npz dopo un rebuild
#     del dataset (es. cambio target_type), SENZA rifare FRED/yfinance/regime
#     walk-forward (~3h). Replica il passo 6 di 01b_download_macro.py: stesse
#     macro_features.parquet su disco, stesso MacroNormalizer (refit identico),
#     stesso merge daily-ffill sui timestamp t_{split}.
# EN: VOL-S utility — re-appends X_macro_* to data/lstm_dataset.npz after a dataset
#     rebuild (e.g. target_type change), WITHOUT re-running FRED/yfinance/regime
#     walk-forward (~3h). Replicates step 6 of 01b_download_macro.py: same on-disk
#     macro_features.parquet, same MacroNormalizer (identical refit), same
#     daily-ffill merge onto the t_{split} timestamps.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.macro.regime import MacroNormalizer     # noqa: E402
from quantsys.utils import setup_logging              # noqa: E402
from quantsys.utils.atomic_save import atomic_save_npz  # noqa: E402

import logging  # noqa: E402
setup_logging()
log = logging.getLogger("quantsys.script.vols_macro")


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    out = Path("data")
    df_macro = pd.read_parquet(out / "macro_features.parquet")
    macro_cols = list(df_macro.columns)

    # IT: refit del normalizer come in 01b (fit_transform sull'intero df macro —
    #     comportamento pre-esistente, identico run-su-run a parità di parquet).
    # EN: normalizer refit as in 01b (fit_transform on the whole macro df —
    #     pre-existing behavior, identical run-over-run for the same parquet).
    normalizer = MacroNormalizer()
    normalizer.fit_transform(df_macro, macro_cols)

    npz_path = out / "lstm_dataset.npz"
    with np.load(npz_path, allow_pickle=True) as npz:
        splits_out = {k: np.array(npz[k]) for k in npz.files}

    macro_daily = df_macro.copy()
    macro_daily.index = pd.to_datetime(macro_daily.index, utc=True).normalize()
    macro_daily = macro_daily[~macro_daily.index.duplicated(keep="last")]
    macro_daily = macro_daily[macro_cols].sort_index()

    for split in ["train", "val", "test"]:
        t_key = f"t_{split}"
        if t_key not in splits_out:
            continue
        timestamps = pd.to_datetime(splits_out[t_key])
        dates_utc = pd.DatetimeIndex(timestamps.normalize(), tz="UTC")
        all_dates = macro_daily.index.append(dates_utc).drop_duplicates().sort_values()
        merged = macro_daily.reindex(all_dates).ffill().loc[dates_utc].fillna(0.0)
        X = np.clip(
            normalizer.scaler.transform(merged.values.astype(np.float32)), -5, 5
        ).astype(np.float32)
        splits_out[f"X_macro_{split}"] = X
        log.info(f"  {split}: X_macro shape = {X.shape}")

    splits_out["macro_feature_names"] = np.array(macro_cols)
    splits_out["n_macro_features"] = np.array([len(macro_cols)])
    atomic_save_npz(npz_path, **splits_out)
    log.info(f"Dataset aggiornato con macro → {npz_path}")


if __name__ == "__main__":
    main()
