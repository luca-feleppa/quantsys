# IT: Regression test — `01_update_data.py --candles-only` deve fermarsi PRIMA del
#     feature engineering. Il contratto che protegge: il run completo rifitta il
#     RobustScaler, riscrive `lstm_dataset.npz` e salva un nuovo `PipelineState`
#     sotto `models/{arch}/`. Su una linea con modelli CONGELATI (la vol
#     production ha `target_scale` persistito nel proprio state) questo rompe il
#     contratto train↔inference — e lo romperebbe **in silenzio**, perché lo
#     script termina con successo e stampa un banner rassicurante. Se qualcuno
#     spostasse l'uscita anticipata sotto la Fase 2, nulla fallirebbe: cadrebbe
#     solo questo test.
#     Simmetrico: SENZA il flag il feature engineering deve ancora essere
#     raggiunto — l'inerzia va provata in entrambe le direzioni, altrimenti un
#     early-return incondizionato passerebbe la metà "candles-only" del test.
# EN: Regression test — `01_update_data.py --candles-only` must stop BEFORE
#     feature engineering. The contract it protects: the full run refits the
#     RobustScaler, rewrites `lstm_dataset.npz` and saves a new `PipelineState`
#     under `models/{arch}/`. On a line with FROZEN models (production vol has
#     `target_scale` persisted in its own state) this breaks the train↔inference
#     contract — and would break it **silently**, since the script exits
#     successfully printing a reassuring banner. If someone moved the early exit
#     below phase 2, nothing would fail: only this test.
#     Symmetric: WITHOUT the flag feature engineering must still be reached —
#     inertness has to be proven in both directions, otherwise an unconditional
#     early return would pass the "candles-only" half of the test.
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_COLS = ["open_time", "close_time", "open", "high", "low", "close", "volume",
            "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol"]


# IT: carica lo script numerato come modulo (non è un package).
# EN: loads the numbered script as a module (it is not a package).
def _load_update():
    p = ROOT / "scripts" / "01_update_data.py"
    spec = importlib.util.spec_from_file_location("update_data_01", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FeatureEngineeringReached(RuntimeError):
    """IT: sentinella — la Fase 2 è stata raggiunta.
    EN: sentinel — phase 2 was reached."""


# IT: fixture che isola main() da rete e disco: nessuna candela viene scaricata e
#     NESSUN file viene scritto (il parquet di produzione resta intatto).
# EN: fixture isolating main() from network and disk: no candle is downloaded and
#     NO file is written (the production parquet stays intact).
@pytest.fixture
def stubbed(monkeypatch):
    mod = _load_update()
    n_before = 3

    class _FakeMeta:
        num_rows = n_before

    class _FakeParquetFile:
        def __init__(self, *a, **k):
            self.metadata = _FakeMeta()

    monkeypatch.setattr(pq, "ParquetFile", _FakeParquetFile)

    # IT: n_after = n_before + 1 → n_new > 0, così il ramo "già aggiornato" (che
    #     esce comunque) non maschera l'uscita che stiamo testando.
    # EN: n_after = n_before + 1 → n_new > 0, so the "already updated" branch
    #     (which exits anyway) cannot mask the exit under test.
    ts = pd.date_range("2026-01-01", periods=n_before + 1, freq="1h", tz="UTC")
    df = pd.DataFrame({c: 1.0 for c in RAW_COLS}, index=range(len(ts)))
    df["open_time"] = ts
    df["close_time"] = ts

    monkeypatch.setattr(mod, "fetch_klines_incremental", lambda **k: df)
    monkeypatch.setattr(mod, "fetch_funding_rate",
                        lambda **k: pytest.fail("funding non deve essere scaricato"))

    saved = []
    monkeypatch.setattr(mod, "atomic_save_parquet",
                        lambda d, p, **k: saved.append(Path(p).name))
    monkeypatch.setattr(mod, "atomic_save_npz",
                        lambda *a, **k: pytest.fail("npz riscritto in --candles-only"))

    def _boom(*a, **k):
        raise _FeatureEngineeringReached()

    monkeypatch.setattr(mod, "FeatureBuilder", _boom)
    monkeypatch.setattr(mod, "PipelineState", _boom)
    return mod, saved


# IT: con il flag → si ferma dopo il parquet raw, nessuno scaler, nessuno state.
# EN: with the flag → stops after the raw parquet, no scaler, no state.
def test_candles_only_stops_before_feature_engineering(monkeypatch, stubbed):
    mod, saved = stubbed
    monkeypatch.setattr(sys, "argv", ["01_update_data.py", "--candles-only"])
    mod.main()
    assert saved == ["raw_candles.parquet"], \
        f"deve salvare SOLO il parquet raw, invece: {saved}"


# IT: senza il flag → il path production è invariato e arriva alla Fase 2.
# EN: without the flag → the production path is unchanged and reaches phase 2.
def test_default_path_still_reaches_feature_engineering(monkeypatch, stubbed):
    mod, saved = stubbed
    # IT: il funding viene scaricato solo nel path completo: qui è atteso, quindi
    #     lo si rende innocuo invece di farlo fallire.
    # EN: funding is fetched only on the full path: expected here, so make it
    #     harmless instead of failing.
    monkeypatch.setattr(mod, "fetch_funding_rate", lambda **k: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["01_update_data.py"])
    with pytest.raises(_FeatureEngineeringReached):
        mod.main()
    assert saved == ["raw_candles.parquet"]
