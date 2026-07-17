# IT: Regression test — flag QUANTSYS_DATASET_NPZ (inerte di default) e feature
#     builder DVOL del probe (causalità asof, staleness cap, fill train-only).
#     Pre-reg STATUS 2026-07-17 (probe DVOL-come-feature).
# EN: Regression tests — QUANTSYS_DATASET_NPZ flag (inert by default) and the
#     probe's DVOL feature builder (asof causality, staleness cap, train-only
#     fill). STATUS pre-reg 2026-07-17 (DVOL-as-feature probe).
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantsys.utils import dataset_npz_path  # noqa: E402


# IT: carica il modulo appender dallo script (non è un package).
# EN: loads the appender module from the script (not a package).
def _load_appender():
    p = ROOT / "scripts" / "vol" / "dev_vols_dvol_append.py"
    spec = importlib.util.spec_from_file_location("dev_vols_dvol_append", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ────────────────────── flag QUANTSYS_DATASET_NPZ ──────────────────────

# IT: senza env → path production INVARIATO (inerzia del flag, condizione P3).
# EN: without env → production path UNCHANGED (flag inertness, P3 prerequisite).
def test_dataset_npz_default_inert(monkeypatch):
    monkeypatch.delenv("QUANTSYS_DATASET_NPZ", raising=False)
    assert str(dataset_npz_path()) == str(Path("data/lstm_dataset.npz"))


# IT: con env → override esatto (sandbox del probe).
# EN: with env → exact override (probe sandbox).
def test_dataset_npz_override(monkeypatch):
    monkeypatch.setenv("QUANTSYS_DATASET_NPZ", "data/lstm_dataset_dvol.npz")
    assert str(dataset_npz_path()) == str(Path("data/lstm_dataset_dvol.npz"))


# ────────────────────── build_dvol_features (probe) ──────────────────────

@pytest.fixture(scope="module")
def appender():
    return _load_appender()


# IT: serie DVOL sintetica oraria con valori distinti (log riconoscibili).
# EN: synthetic hourly DVOL series with distinct values (recognizable logs).
def _dvol(start="2021-01-10", periods=200):
    ts = pd.date_range(start, periods=periods, freq="1h")
    return pd.DataFrame({"timestamp": ts,
                         "dvol": 40.0 + np.arange(periods, dtype=float)})


# IT: causalità — a t tra due tick l'asof backward prende l'ULTIMO ≤ t, mai il futuro.
# EN: causality — at t between two ticks backward asof takes the LAST ≤ t, never the future.
def test_dvol_asof_is_causal(appender):
    dv = _dvol()
    t = pd.DatetimeIndex([pd.Timestamp("2021-01-12 05:30")])
    out, _ = appender.build_dvol_features(t, dv, fill={"dvol_log": 0.0, "dvol_chg_24h": 0.0})
    # IT: 2021-01-12 05:00 è il 53° tick (indice 53) → dvol=93, non 94 (06:00).
    # EN: 2021-01-12 05:00 is tick index 53 → dvol=93, not 94 (06:00).
    assert out["dvol_avail"].iloc[0] == 1.0
    assert out["dvol_log"].iloc[0] == pytest.approx(np.log(93.0), abs=1e-6)
    # IT: Δ24h = log(93) − log(69) (valore a 2021-01-11 05:30 → asof 05:00, idx 29).
    # EN: 24h Δ = log(93) − log(69) (value at 2021-01-11 05:30 → asof 05:00, idx 29).
    assert out["dvol_chg_24h"].iloc[0] == pytest.approx(np.log(93.0) - np.log(69.0), abs=1e-6)


# IT: pre-copertura (t < primo tick DVOL) → indicator 0 + fill costante.
# EN: pre-coverage (t < first DVOL tick) → indicator 0 + constant fill.
def test_dvol_precoverage_filled(appender):
    dv = _dvol()
    t = pd.DatetimeIndex([pd.Timestamp("2020-06-01 00:00")])
    fill = {"dvol_log": 3.5, "dvol_chg_24h": 0.01}
    out, _ = appender.build_dvol_features(t, dv, fill=fill)
    assert out["dvol_avail"].iloc[0] == 0.0
    assert out["dvol_log"].iloc[0] == pytest.approx(3.5)
    assert out["dvol_chg_24h"].iloc[0] == pytest.approx(0.01)


# IT: staleness — ultimo tick più vecchio del cap 24h → indicator 0.
# EN: staleness — last tick older than the 24h cap → indicator 0.
def test_dvol_staleness_cap(appender):
    dv = _dvol(periods=48)  # IT: copre fino a 2021-01-11 23:00 / EN: covers through 2021-01-11 23:00
    t = pd.DatetimeIndex([pd.Timestamp("2021-01-13 12:00")])
    out, _ = appender.build_dvol_features(t, dv, fill={"dvol_log": 0.0, "dvol_chg_24h": 0.0})
    assert out["dvol_avail"].iloc[0] == 0.0


# IT: primo giorno di copertura — v(t) esiste ma v(t−24h) no → indicator 0
#     (l'indicator richiede ENTRAMBI, coerente con la coppia di feature).
# EN: first coverage day — v(t) exists but v(t−24h) doesn't → indicator 0
#     (the indicator requires BOTH, consistent with the feature pair).
def test_dvol_first_day_unavailable(appender):
    dv = _dvol()
    t = pd.DatetimeIndex([pd.Timestamp("2021-01-10 06:00")])
    out, _ = appender.build_dvol_features(t, dv, fill={"dvol_log": 0.0, "dvol_chg_24h": 0.0})
    assert out["dvol_avail"].iloc[0] == 0.0


# IT: fill=None calcola le mediane sulla porzione disponibile e le RITORNA per
#     il riuso su val/test (no leakage: mediane del solo train).
# EN: fill=None computes medians on the available portion and RETURNS them for
#     reuse on val/test (no leakage: train-only medians).
def test_dvol_fill_medians_from_train_reused(appender):
    dv = _dvol()
    t_train = pd.date_range("2021-01-11 06:00", periods=24, freq="1h")
    out_tr, fill = appender.build_dvol_features(pd.DatetimeIndex(t_train), dv, fill=None)
    assert out_tr["dvol_avail"].all()
    avail_logs = out_tr["dvol_log"]
    assert fill["dvol_log"] == pytest.approx(float(np.median(avail_logs)), abs=1e-6)
    # IT: su val pre-copertura le costanti applicate sono ESATTAMENTE quelle del train.
    # EN: on pre-coverage val rows the constants applied are EXACTLY the train ones.
    t_val = pd.DatetimeIndex([pd.Timestamp("2019-01-01 00:00")])
    out_vl, fill2 = appender.build_dvol_features(t_val, dv, fill=fill)
    assert fill2 == fill
    assert out_vl["dvol_log"].iloc[0] == pytest.approx(fill["dvol_log"])
    assert out_vl["dvol_avail"].iloc[0] == 0.0
