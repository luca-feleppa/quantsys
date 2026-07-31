"""
IT: Test dell'helper `scripts/vps/macro_vintage.py` — il vintage dello snapshot
    macro. L'helper e' invocato da `pull_vps_data.ps1` per NOMINARE l'archivio
    datato e per decidere se il canonico remoto va promosso: il contratto che
    conta e' che su stdout finisca ESATTAMENTE il vintage e nient'altro, perche'
    il chiamante PowerShell tratta come vintage tutto cio' che legge da stdout.
    Una riga di diagnostica sfuggita li' dentro produrrebbe un nome di archivio
    sbagliato in silenzio.
EN: Tests for the `scripts/vps/macro_vintage.py` helper - the macro snapshot
    vintage. The helper is called by `pull_vps_data.ps1` to NAME the dated
    archive and to decide whether the remote canonical must be promoted: the
    contract that matters is that stdout carries EXACTLY the vintage and nothing
    else, because the PowerShell caller treats whatever it reads from stdout as
    the vintage. A stray diagnostic line there would silently produce a wrong
    archive name.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "vps" / "macro_vintage.py"


def _run(arg):
    # IT: subprocess e non import: e' proprio l'interfaccia CLI a essere il
    #     contratto (stdout/exit code), non la funzione Python.
    # EN: subprocess, not import: the CLI surface (stdout/exit code) IS the
    #     contract here, not the Python function.
    return subprocess.run(
        [sys.executable, str(HELPER), str(arg)],
        capture_output=True, text=True,
    )


@pytest.fixture
def macro_parquet(tmp_path):
    # IT: indice giornaliero: l'ultima data e' il vintage atteso.
    # EN: daily index: the last date is the expected vintage.
    idx = pd.date_range("2026-07-01", "2026-07-30", freq="D")
    df = pd.DataFrame({"a": range(len(idx)), "b": 1.0}, index=idx)
    p = tmp_path / "macro_features.parquet"
    df.to_parquet(p)
    return p


def test_stdout_is_exactly_the_vintage(macro_parquet):
    # IT: una sola riga, formato YYYYMMDD, uguale all'ultima data dell'indice.
    # EN: a single line, YYYYMMDD, equal to the last index date.
    r = _run(macro_parquet)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "20260730"
    assert len(r.stdout.strip().splitlines()) == 1


def test_vintage_is_last_index_date_not_write_time(tmp_path):
    # IT: il vintage e' lo stato del mondo incorporato, NON il mtime del file:
    #     un ri-salvataggio senza dati nuovi non deve cambiare il vintage (e
    #     quindi non deve far scattare una promozione spuria del canonico).
    # EN: the vintage is the embedded state of the world, NOT the file mtime: a
    #     re-save with no new data must not change the vintage (hence must not
    #     trigger a spurious promotion of the canonical pointer).
    idx = pd.date_range("2019-01-01", "2026-06-15", freq="D")
    df = pd.DataFrame({"a": 1.0}, index=idx)
    p = tmp_path / "m.parquet"
    df.to_parquet(p)
    first = _run(p).stdout.strip()
    df.to_parquet(p)  # IT: ri-salvataggio identico / EN: identical re-save
    assert _run(p).stdout.strip() == first == "20260615"


def test_missing_file_fails_without_polluting_stdout(tmp_path):
    # IT: fallimento pulito: exit != 0 e stdout VUOTO, cosi' il chiamante salta
    #     il push invece di archiviare sotto un nome inventato.
    # EN: clean failure: non-zero exit and EMPTY stdout, so the caller skips the
    #     push instead of archiving under a made-up name.
    r = _run(tmp_path / "nope.parquet")
    assert r.returncode != 0
    assert r.stdout.strip() == ""
    assert "missing" in r.stderr or "assente" in r.stderr


def test_empty_parquet_fails(tmp_path):
    # IT: un parquet senza righe non ha vintage: fallire, non stampare nulla.
    # EN: a row-less parquet has no vintage: fail, print nothing.
    p = tmp_path / "empty.parquet"
    pd.DataFrame({"a": []}, index=pd.DatetimeIndex([])).to_parquet(p)
    r = _run(p)
    assert r.returncode != 0
    assert r.stdout.strip() == ""
