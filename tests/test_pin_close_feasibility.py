"""
IT: Test del conteggio ex-ante A13a (`scripts/vol/pin_close_feasibility.py`).
    Cosa e' davvero a rischio qui non e' l'aritmetica, e' il SOTTO-CONTEGGIO
    silenzioso: se un tick senza dato venisse riempito per propagazione invece che
    lasciato NaN, le posizioni dell'epoca "casa" (copertura 2-6% nella finestra del
    pin, perche' le expiry sono alle 08:00 UTC e il PC era spento di notte)
    entrerebbero nel conteggio come "nessun innesco" invece che come NON OSSERVATE.
    Il conteggio ne uscirebbe distorto verso il basso e A13 verrebbe archiviata per
    un difetto di dati scambiato per un fatto. Questi test bloccano quel percorso.
EN: Tests for the A13a ex-ante count (`scripts/vol/pin_close_feasibility.py`).
    What is actually at risk here is not the arithmetic, it is silent
    UNDER-COUNTING: if a tick with no data were forward-filled instead of left
    NaN, the "home" epoch positions (2-6% coverage inside the pin window, because
    expiries are at 08:00 UTC and the PC was off at night) would enter the count as
    "no trigger" instead of NOT OBSERVED. The count would be biased downward and
    A13 would be shelved over a data defect mistaken for a fact. These tests block
    that path.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pf():
    spec = importlib.util.spec_from_file_location(
        "pin_close_feasibility", ROOT / "scripts" / "vol" / "pin_close_feasibility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ticks_are_the_moments_the_process_can_act(pf):
    # IT: 04b agisce a hh:00+90s: la condizione va valutata li' e non in continuo,
    #     altrimenti si contano inneschi che il processo non avrebbe potuto cogliere.
    # EN: 04b acts at hh:00+90s: the condition must be evaluated there and not
    #     continuously, or one counts triggers the process could never have caught.
    entry = pd.Timestamp("2026-07-29 01:01:33", tz="UTC")
    expiry = pd.Timestamp("2026-07-30 08:00:00", tz="UTC")
    tk = pf.ticks_for(entry, expiry)
    assert tk[0] == pd.Timestamp("2026-07-29 02:01:30", tz="UTC")
    assert (tk > entry).all()
    assert (tk < expiry).all(), "nessun tick a/oltre la scadenza / no tick at or past expiry"
    # IT: confronto in Timedelta e non in interi: la risoluzione dell'indice (us o
    #     ns) dipende da come e' costruito il Timestamp e non e' parte del contratto.
    # EN: compare Timedeltas, not raw ints: the index resolution (us or ns) depends
    #     on how the Timestamp was built and is not part of the contract.
    assert ((tk[1:] - tk[:-1]) == pd.Timedelta("1h")).all()


def test_entry_exactly_on_a_tick_does_not_duplicate(pf):
    # IT: se l'entry cade esattamente su un tick, quel tick non e' valutabile per
    #     la chiusura (la posizione nasce li'): si parte dal successivo.
    # EN: if entry falls exactly on a tick, that tick is not evaluable for closing
    #     (the position is born there): start from the next one.
    entry = pd.Timestamp("2026-07-29 02:01:30", tz="UTC")
    expiry = pd.Timestamp("2026-07-30 08:00:00", tz="UTC")
    assert pf.ticks_for(entry, expiry)[0] == pd.Timestamp("2026-07-29 03:01:30", tz="UTC")


def test_missing_data_stays_nan_and_is_never_filled(pf):
    # IT: IL test che conta. Un buco piu' largo della tolleranza deve restare NaN:
    #     un forward-fill trasformerebbe "non osservato" in "osservato e negativo".
    # EN: THE test that matters. A gap wider than the tolerance must stay NaN: a
    #     forward fill would turn "not observed" into "observed and negative".
    src = pd.DataFrame({
        "snapshot_ts": pd.to_datetime(["2026-07-29 02:00:00", "2026-07-29 09:00:00"], utc=True),
        "px": [64000.0, 65000.0],
    })
    ticks = pd.DatetimeIndex(pd.to_datetime(
        ["2026-07-29 02:01:30",   # IT/EN: dentro tolleranza / within tolerance
         "2026-07-29 05:01:30",   # IT/EN: buco / gap
         "2026-07-29 06:01:30"],  # IT/EN: buco / gap
        utc=True))
    px = pf.price_at(ticks, src)
    assert px[0] == 64000.0
    assert np.isnan(px[1]) and np.isnan(px[2]), "buco riempito: il conteggio sarebbe distorto"


def test_us_and_ns_resolutions_both_work(pf):
    # IT: i parquet del chain arrivano in microsecondi, i tick sono costruiti in
    #     nanosecondi: il mismatch faceva fallire merge_asof (non silenziosamente,
    #     ma il fix va protetto perche' un cast sbagliato sarebbe silenzioso).
    # EN: chain parquets come in microseconds, ticks are built in nanoseconds: the
    #     mismatch broke merge_asof (loudly, but the fix needs guarding because a
    #     wrong cast would be silent).
    src = pd.DataFrame({
        "snapshot_ts": pd.to_datetime(["2026-07-29 02:00:00"], utc=True).as_unit("us"),
        "px": [64000.0],
    })
    ticks = pd.DatetimeIndex(pd.to_datetime(["2026-07-29 02:01:30"], utc=True)).as_unit("ns")
    assert pf.price_at(ticks, src)[0] == 64000.0


def test_predicate_is_the_production_one(pf):
    # IT: il conteggio deve girare sulla funzione di PRODUZIONE, non su una copia.
    #     Verifica dei confini: 0 < t_left <= X (expiry passata = False, competenza
    #     di maybe_settle) e |S-K|/S <= f.
    # EN: the count must run on the PRODUCTION function, not a copy. Boundary
    #     check: 0 < t_left <= X (past expiry = False, maybe_settle's job) and
    #     |S-K|/S <= f.
    vp = pf._import_from_path("volpaper_04b_testpin", ROOT / "scripts" / "04b_vol_paper.py")
    exp_ms, K = 1_785_484_800_000.0, 64_000.0
    at = lambda h: exp_ms - h * 3.6e6
    assert vp.pin_close_due(K, 64_000.0, exp_ms, at(2.0), 3.0, 0.005)      # dentro/inside
    assert not vp.pin_close_due(K, 64_000.0, exp_ms, at(4.0), 3.0, 0.005)  # troppo presto/too early
    assert not vp.pin_close_due(K, 64_000.0, exp_ms, at(-1.0), 3.0, 0.005)  # oltre scadenza/past
    assert not vp.pin_close_due(K, 64_000.0 * 1.02, exp_ms, at(2.0), 3.0, 0.005)  # fuori banda/out of band
