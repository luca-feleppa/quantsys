"""
tests/conftest.py
-----------------
Fixture condivise per tutti i test di QUANTSYS.
pytest le carica automaticamente prima di ogni sessione di test.
"""
import numpy as np
import pandas as pd
import pytest


# IT: Helper interno — costruisce OHLCV sintetico riproducibile per i test.
# EN: Internal helper — builds reproducible synthetic OHLCV for tests.
def _make_ohlcv(n: int, seed: int = 99) -> pd.DataFrame:
    """Helper interno: genera un DataFrame OHLCV sintetico con n candele."""
    np.random.seed(seed)
    price   = 50_000.0
    records = []
    dates   = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    for ts in dates:
        ret   = np.random.normal(0, 0.001)
        close = max(price * (1 + ret), 1.0)
        noise = abs(np.random.normal(0, 0.0005))
        high  = close * (1 + noise)
        low   = close * (1 - noise)
        vol   = float(np.random.lognormal(7, 0.5))
        records.append({
            "open_time":          ts,
            "open":               price,
            "high":               high,
            "low":                low,
            "close":              close,
            "volume":             vol,
            "close_time":         ts + pd.Timedelta(seconds=59),
            "quote_vol":          close * vol,
            "trades":             int(np.random.randint(50, 300)),
            "taker_buy_vol":      vol * np.random.uniform(0.3, 0.7),
            "taker_buy_quote_vol":close * vol * 0.5,
        })
        price = close
    return pd.DataFrame(records)


# IT: 200 candele — usato dai test rapidi (log-return, VWAP, VP).
# EN: 200 candles — used by fast tests (log-returns, VWAP, VP).
@pytest.fixture(scope="session")
def tiny_ohlcv():
    """
    200 candele sintetiche — per test veloci (log-return, VWAP, lags, VP).
    Scope=session: creato una volta sola per tutta la sessione.
    """
    return _make_ohlcv(200)


# IT: 2000 candele — per rolling lunghi (structural 30/90/365d, CVD warm-up).
# EN: 2000 candles — for long rollings (structural 30/90/365d, CVD warm-up).
@pytest.fixture(scope="session")
def synthetic_ohlcv():
    """
    2000 candele sintetiche — per test che richiedono finestre lunghe:
      · _structural_features usa rolling su 30d (43200 min), 90d, 365d
        con min_periods=60 → servono almeno ~300 candele per risultati non-NaN
      · _cvd_features usa rolling su 20/60 candele con warm-up
      · VP multi-scale usa lookback 60/240/1440 min

    Scope=session: costoso da generare, viene riusato.
    """
    return _make_ohlcv(2000)


# IT: RobustScaler pre-fittato — usato dai test PipelineState round-trip.
# EN: Pre-fitted RobustScaler — used by PipelineState round-trip tests.
@pytest.fixture(scope="session")
def trained_scaler():
    """RobustScaler già fittato su dati casuali — per test PipelineState."""
    from sklearn.preprocessing import RobustScaler
    X = np.random.randn(200, 1)
    return RobustScaler().fit(X)
