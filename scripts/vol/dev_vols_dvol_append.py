# IT: PROBE DVOL-COME-FEATURE (pre-reg STATUS 2026-07-17) — deriva
#     data/lstm_dataset_dvol.npz dal npz production aggiungendo 3 colonne allo
#     stream macro: dvol_log (merge_asof backward, staleness cap 24h),
#     dvol_chg_24h (Δ24h di log(DVOL)), dvol_avail (indicator 0/1).
#     Fill dove non disponibile: mediana della porzione DISPONIBILE del SOLO
#     train (no leakage) + indicator=0. Il npz production NON viene toccato
#     (READ-ONLY, congelato per A3/A8); verifica bit-identità post-save di
#     tutte le chiavi non aumentate. Lanciare dalla root di progetto.
# EN: DVOL-AS-FEATURE PROBE (STATUS pre-reg 2026-07-17) — derives
#     data/lstm_dataset_dvol.npz from the production npz by adding 3 columns to
#     the macro stream: dvol_log (backward merge_asof, 24h staleness cap),
#     dvol_chg_24h (24h Δ of log(DVOL)), dvol_avail (0/1 indicator).
#     Fill where unavailable: median of the AVAILABLE portion of the TRAIN
#     split only (no leakage) + indicator=0. The production npz is NEVER
#     touched (READ-ONLY, frozen for A3/A8); post-save bit-identity check of
#     every non-augmented key. Run from the project root.
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantsys.macro.regime import MacroNormalizer        # noqa: E402
from quantsys.utils import setup_logging                 # noqa: E402
from quantsys.utils.atomic_save import atomic_save_npz   # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vols_dvol")

# IT: costanti pre-registrate (STATUS 2026-07-17) — nessuno sweep.
# EN: pre-registered constants (STATUS 2026-07-17) — no sweep.
SRC_NPZ = Path("data/lstm_dataset.npz")
DST_NPZ = Path("data/lstm_dataset_dvol.npz")
DVOL_PARQUET = Path("data/iv/dvol.parquet")
STALENESS = pd.Timedelta(hours=24)
DVOL_COLS = ["dvol_log", "dvol_chg_24h", "dvol_avail"]


def build_dvol_features(timestamps: pd.DatetimeIndex, dvol: pd.DataFrame,
                        fill: dict | None = None) -> tuple[pd.DataFrame, dict]:
    # IT: costruisce le 3 feature RAW (pre-normalizzazione) sui timestamp dati.
    #     CAUSALE: merge_asof backward (ultimo valore ≤ t) con tolerance 24h;
    #     dvol_chg_24h = log(DVOL)_t − log(DVOL)_{t−24h} (stesso asof su t−24h).
    #     `fill=None` → calcola le mediane sulla porzione disponibile (SOLO per
    #     la chiamata train); altrimenti riusa le mediane passate (val/test).
    # EN: builds the 3 RAW features (pre-normalization) on the given timestamps.
    #     CAUSAL: backward merge_asof (last value ≤ t) with 24h tolerance;
    #     dvol_chg_24h = log(DVOL)_t − log(DVOL)_{t−24h} (same asof at t−24h).
    #     `fill=None` → computes medians on the available portion (TRAIN call
    #     only); otherwise reuses the medians passed in (val/test).
    # IT: l'ordine righe DEVE restare quello di X_{split} → input monotono richiesto.
    # EN: row order MUST match X_{split} → monotonic input required.
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps non monotoni / non-monotonic timestamps")
    dv = dvol.dropna(subset=["dvol"]).sort_values("timestamp").reset_index(drop=True)
    dv["dvol_log"] = np.log(dv["dvol"].astype(float))
    # IT: risoluzione datetime unificata a ns — l'npz è [ms], il parquet [us]:
    #     merge_asof richiede dtype identici.
    # EN: datetime resolution unified to ns — the npz is [ms], the parquet [us]:
    #     merge_asof requires identical dtypes.
    dv["timestamp"] = dv["timestamp"].astype("datetime64[ns]")
    right = dv[["timestamp", "dvol_log"]]

    left = pd.DataFrame({"t": timestamps.astype("datetime64[ns]")})
    now = pd.merge_asof(left.rename(columns={"t": "timestamp"}), right,
                        on="timestamp", direction="backward",
                        tolerance=STALENESS)["dvol_log"]
    prev_ts = left["t"] - pd.Timedelta(hours=24)
    prev = pd.merge_asof(pd.DataFrame({"timestamp": prev_ts}), right,
                         on="timestamp", direction="backward",
                         tolerance=STALENESS)["dvol_log"]

    chg = now - prev
    avail = now.notna() & prev.notna()

    if fill is None:
        # IT: mediane SOLO sulla porzione disponibile di QUESTA chiamata (train).
        # EN: medians on the available portion of THIS call only (train).
        if not avail.any():
            raise RuntimeError("nessun timestamp coperto da DVOL nel train / "
                               "no train timestamp covered by DVOL")
        fill = {"dvol_log": float(now[avail].median()),
                "dvol_chg_24h": float(chg[avail].median())}

    # IT: .to_numpy() deliberato — evita il riallineamento per indice di pandas
    #     (Series RangeIndex vs index timestamp → NaN silenziosi).
    # EN: deliberate .to_numpy() — avoids pandas index re-alignment
    #     (RangeIndex Series vs timestamp index → silent NaNs).
    out = pd.DataFrame({
        "dvol_log": now.where(avail, fill["dvol_log"]).to_numpy(np.float32),
        "dvol_chg_24h": chg.where(avail, fill["dvol_chg_24h"]).to_numpy(np.float32),
        "dvol_avail": avail.to_numpy(np.float32),
    }, index=left["t"])
    return out, fill


def main() -> None:
    # IT: boilerplate UTF-8 (checklist nuovo script, CLAUDE.md).
    # EN: UTF-8 boilerplate (new-script checklist, CLAUDE.md).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # IT: fotografia del production npz per l'assert finale di non-modifica.
    # EN: production npz snapshot for the final untouched assert.
    src_stat = SRC_NPZ.stat()

    dvol = pd.read_parquet(DVOL_PARQUET)
    dvol["timestamp"] = pd.to_datetime(dvol["timestamp"], utc=True).dt.tz_localize(None)
    log.info(f"DVOL: {len(dvol)} righe, {dvol['timestamp'].iloc[0]} → {dvol['timestamp'].iloc[-1]}")

    with np.load(SRC_NPZ, allow_pickle=True) as npz:
        arrays = {k: np.array(npz[k]) for k in npz.files}

    # IT: feature raw per split — mediane di fill dal SOLO train (no leakage).
    # EN: raw features per split — fill medians from the TRAIN split only (no leakage).
    raw, fill = {}, None
    for split in ("train", "val", "test"):
        ts = pd.DatetimeIndex(pd.to_datetime(arrays[f"t_{split}"]))
        raw[split], fill = build_dvol_features(ts, dvol, fill)
        cov = raw[split]["dvol_avail"].mean()
        log.info(f"  {split}: n={len(ts)} copertura dvol_avail={cov:.3f}")
    log.info(f"  fill (mediane train disponibile): {fill}")

    # IT: normalizzazione delle 2 colonne continue col pattern macro esistente
    #     (MacroNormalizer=RobustScaler, refit whole-df, clip ±5 — comportamento
    #     pre-esistente dichiarato in pre-reg). L'indicator resta RAW {0,1}:
    #     su una binaria quasi-costante l'IQR del RobustScaler è ~0 (scala
    #     degenere), e il MacroEncoder non richiede input scalati.
    # EN: normalization of the 2 continuous columns via the existing macro
    #     pattern (MacroNormalizer=RobustScaler, whole-df refit, ±5 clip —
    #     pre-existing behavior declared in the pre-reg). The indicator stays
    #     RAW {0,1}: on a near-constant binary the RobustScaler IQR is ~0
    #     (degenerate scale), and the MacroEncoder needs no scaled input.
    cont_cols = ["dvol_log", "dvol_chg_24h"]
    whole = pd.concat([raw[s][cont_cols] for s in ("train", "val", "test")])
    norm = MacroNormalizer()
    whole_scaled = norm.fit_transform(whole, cont_cols)

    ofs = 0
    for split in ("train", "val", "test"):
        n = len(raw[split])
        X3 = np.column_stack([whole_scaled[ofs:ofs + n],
                              raw[split]["dvol_avail"].to_numpy(np.float32)]
                             ).astype(np.float32)
        ofs += n
        key = f"X_macro_{split}"
        arrays[key] = np.concatenate([arrays[key], X3], axis=1)
        log.info(f"  {key}: → {arrays[key].shape}")

    names = list(arrays["macro_feature_names"]) + DVOL_COLS
    arrays["macro_feature_names"] = np.array(names)
    arrays["n_macro_features"] = np.array([len(names)])

    # IT: non-compresso deliberato: 5-10× più veloce, il float32 comprime ~0.
    # EN: deliberately uncompressed: 5-10× faster, float32 barely compresses.
    atomic_save_npz(DST_NPZ, compressed=False, **arrays)
    log.info(f"Salvato {DST_NPZ} ({DST_NPZ.stat().st_size / 1e9:.2f} GB)")

    # IT: VERIFICA post-save — bit-identità chiave-per-chiave vs production
    #     (lazy: una chiave alla volta, picco RAM contenuto) + production intatto.
    # EN: post-save VERIFY — key-by-key bit-identity vs production (lazy: one
    #     key at a time, bounded RAM peak) + production untouched.
    aug = {"X_macro_train", "X_macro_val", "X_macro_test",
           "macro_feature_names", "n_macro_features"}
    with np.load(SRC_NPZ, allow_pickle=True) as src, \
         np.load(DST_NPZ, allow_pickle=True) as dst:
        assert set(src.files) == set(dst.files), "set di chiavi divergente / key set diverged"
        for k in src.files:
            if k in aug:
                continue
            assert np.array_equal(src[k], dst[k]), f"chiave NON bit-identica / non-bit-identical key: {k}"
        for split in ("train", "val", "test"):
            assert np.array_equal(src[f"X_macro_{split}"],
                                  dst[f"X_macro_{split}"][:, :90]), \
                f"prime 90 colonne X_macro_{split} divergono / first 90 X_macro_{split} cols diverge"
        assert list(dst["macro_feature_names"][:90]) == list(src["macro_feature_names"]), \
            "primi 90 nomi macro divergono / first 90 macro names diverge"
        assert int(dst["n_macro_features"][0]) == 93, "n_macro_features ≠ 93"
    post = SRC_NPZ.stat()
    assert (post.st_size, post.st_mtime_ns) == (src_stat.st_size, src_stat.st_mtime_ns), \
        "il npz PRODUCTION è stato modificato / PRODUCTION npz was modified"
    log.info("VERIFICA PASS: chiavi non aumentate bit-identiche, prime 90 colonne "
             "X_macro identiche, production npz intatto / VERIFY PASS")


if __name__ == "__main__":
    main()
