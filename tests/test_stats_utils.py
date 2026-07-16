# IT: Golden test dell'estrazione bootstrap_sharpe_ci/mdd_stats (step 2 refactor
#     2026-07-16): i valori attesi sono stati CATTURATI dal codice pre-refactor
#     in scripts/03_backtest.py (stesso input, seed bootstrap 42) — qualunque
#     divergenza numerica = regressione dell'estrazione.
# EN: Golden tests for the bootstrap_sharpe_ci/mdd_stats extraction (refactor
#     step 2, 2026-07-16): expected values were CAPTURED from the pre-refactor
#     code in scripts/03_backtest.py (same input, bootstrap seed 42) — any
#     numerical divergence = extraction regression.
import numpy as np
import pytest

from quantsys.utils.stats import bootstrap_sharpe_ci, mdd_stats


def test_bootstrap_ci_golden():
    # IT: golden pre-refactor: rng(7), 60 PnL, annualize=8760 (1h).
    # EN: pre-refactor golden: rng(7), 60 PnLs, annualize=8760 (1h).
    rng = np.random.default_rng(7)
    pnl = rng.normal(0.001, 0.01, 60).tolist()
    ci = bootstrap_sharpe_ci(pnl, annualize=8760)
    assert ci["sharpe_ci_low"] == pytest.approx(-4.713284017, abs=1e-9)
    assert ci["sharpe_ci_high"] == pytest.approx(1.6024009539, abs=1e-9)
    assert ci["sortino_ci_low"] == pytest.approx(-7.1377193593, abs=1e-9)
    assert ci["sortino_ci_high"] == pytest.approx(2.8055909262, abs=1e-9)


def test_bootstrap_ci_under_30_trades_is_none():
    # IT: sotto 30 trade nessuna stima (soglia storica del backtest).
    # EN: below 30 trades no estimate (historical backtest threshold).
    ci = bootstrap_sharpe_ci([0.1] * 10)
    assert all(v is None for v in ci.values())


def test_bootstrap_ci_deterministic():
    # IT: seed fisso 42 → due chiamate identiche (riproducibilità dei report).
    # EN: fixed seed 42 → two identical calls (report reproducibility).
    pnl = list(np.linspace(-0.01, 0.02, 40))
    assert bootstrap_sharpe_ci(pnl) == bootstrap_sharpe_ci(pnl)


def test_mdd_recovered_golden():
    # IT: golden pre-refactor: peak 105 (i=1) → trough 92 (i=4) → recovery 106 (i=7).
    # EN: pre-refactor golden: peak 105 (i=1) → trough 92 (i=4) → recovery 106 (i=7).
    out = mdd_stats([100, 105, 103, 98, 92, 95, 101, 106, 104, 107])
    assert out == {"mdd_duration_candles": 3,
                   "mdd_recovery_candles": 3,
                   "mdd_recovered": True}


def test_mdd_not_recovered_golden():
    # IT: equity che non riaggancia il peak → recovery None, recovered False.
    # EN: equity never re-reaching the peak → recovery None, recovered False.
    out = mdd_stats([100, 110, 90, 95])
    assert out == {"mdd_duration_candles": 1,
                   "mdd_recovery_candles": None,
                   "mdd_recovered": False}


def test_mdd_monotone_equity():
    # IT: equity monotona crescente → nessun drawdown, durata 0.
    # EN: monotonically rising equity → no drawdown, zero duration.
    out = mdd_stats([100, 101, 102, 103])
    assert out["mdd_duration_candles"] == 0
    assert out["mdd_recovered"] is True
