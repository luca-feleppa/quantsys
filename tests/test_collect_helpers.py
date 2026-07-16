# IT: Regression test degli helper collector estratti il 2026-07-16
#     (quantsys/utils/collect.py + quantsys/data/deribit.py, ex duplicati
#     locali in 01c/01d/01e): il comportamento deve restare identico a quello
#     delle copie negli script (dedup keep-last, sort, parse strumenti).
# EN: Regression tests for the collector helpers extracted on 2026-07-16
#     (quantsys/utils/collect.py + quantsys/data/deribit.py, ex local
#     duplicates in 01c/01d/01e): behaviour must stay identical to the script
#     copies (keep-last dedup, sorting, instrument parsing).
from datetime import datetime, timezone

import pandas as pd
import pytest

from quantsys.data.deribit import parse_instrument
from quantsys.utils.collect import append_parquet


# ─── parse_instrument ─────────────────────────────────────────────────────────

def test_parse_call_standard():
    # IT: nome standard call → expiry 08:00 UTC, strike float, tipo C.
    # EN: standard call name → 08:00 UTC expiry, float strike, C type.
    exp, strike, opt = parse_instrument("BTC-13JUN26-105000-C")
    assert exp == datetime(2026, 6, 13, 8, 0, tzinfo=timezone.utc)
    assert strike == 105000.0
    assert opt == "C"


def test_parse_put_single_digit_day():
    # IT: giorno a una cifra (convenzione Deribit: BTC-9JUL26-...).
    # EN: single-digit day (Deribit convention: BTC-9JUL26-...).
    exp, strike, opt = parse_instrument("BTC-9JUL26-64000-P")
    assert exp == datetime(2026, 7, 9, 8, 0, tzinfo=timezone.utc)
    assert strike == 64000.0
    assert opt == "P"


def test_parse_decimal_strike_no_crash():
    # IT: strike "3d5" = 3.5 (non esiste su BTC ma il parse non deve crashare).
    # EN: "3d5" strike = 3.5 (doesn't occur on BTC but parsing must not crash).
    parsed = parse_instrument("ETH-1AUG26-3d5-C")
    assert parsed is not None and parsed[1] == 3.5


@pytest.mark.parametrize("name", [
    "BTC-PERPETUAL",            # perpetual: niente strike/tipo / no strike/type
    "BTC-25SEP26",              # future: idem
    "BTC_USDC-25SEP26-70000-C", # linear (underscore): fuori contratto / out of contract
    "garbage",
])
def test_parse_non_option_returns_none(name):
    assert parse_instrument(name) is None


# ─── append_parquet ───────────────────────────────────────────────────────────

def _df(rows):
    return pd.DataFrame(rows)


def test_append_create_dedup_keep_last(tmp_path):
    # IT: creazione file + dedup keep-last (il tick nuovo vince sul vecchio).
    # EN: file creation + keep-last dedup (the new tick wins over the old one).
    p = tmp_path / "x.parquet"
    n = append_parquet(p, _df([{"ts": 1, "v": 10}, {"ts": 2, "v": 20}]), ["ts"])
    assert n == 2
    n = append_parquet(p, _df([{"ts": 2, "v": 99}, {"ts": 3, "v": 30}]), ["ts"])
    assert n == 3
    out = pd.read_parquet(p)
    assert out.loc[out["ts"] == 2, "v"].item() == 99          # keep='last'
    assert out["ts"].tolist() == [1, 2, 3]                    # sort su dedup_cols[0]


def test_append_sort_col_override(tmp_path):
    # IT: dedup su una chiave, sort su un'altra (pattern 01e: trade_id/timestamp).
    # EN: dedup on one key, sort on another (01e pattern: trade_id/timestamp).
    p = tmp_path / "x.parquet"
    append_parquet(p, _df([{"tid": "b", "timestamp": 2}, {"tid": "a", "timestamp": 1}]),
                   ["tid"], sort_col="timestamp")
    out = pd.read_parquet(p)
    assert out["timestamp"].tolist() == [1, 2]


def test_append_empty_is_noop(tmp_path):
    # IT: DataFrame vuoto → 0, nessun file scritto.
    # EN: empty DataFrame → 0, no file written.
    p = tmp_path / "x.parquet"
    assert append_parquet(p, pd.DataFrame(), ["ts"]) == 0
    assert not p.exists()
