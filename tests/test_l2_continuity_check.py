# IT: Test del monitor di continuita' L2. Il rischio dominante di questo monitor NON e'
#     un bug aritmetico: e' che misuri il MIRROR LOCALE credendo di misurare il recorder.
#     Il falso positivo del 2026-08-02 (ora in corso letta come buco -> "run corrente 0h"
#     e una stima di ~5 n_eff persi, scritta in STATUS e poi ritrattata) nasceva tutto
#     nella costruzione dello span. I test inchiodano le tre proprieta' che lo escludono:
#     (a) l'ora in corso non spezza il run, a QUALUNQUE minuto giri la routine;
#     (b) un buco consolidato interno resta un allarme pieno;
#     (c) un buco nella coda recente e' PROVVISORIO e non entra nella stima di costo.
# EN: Tests for the L2 continuity monitor. Its dominant risk is NOT an arithmetic bug: it
#     is measuring the LOCAL MIRROR while believing it measures the recorder. The
#     2026-08-02 false positive (in-progress hour read as a gap -> "current run 0h" plus a
#     ~5 n_eff loss estimate, written into STATUS and later retracted) originated entirely
#     in the span construction. The tests pin the three properties that rule it out:
#     (a) the in-progress hour never breaks the run, whatever minute the routine runs at;
#     (b) an interior consolidated gap stays a full alarm;
#     (c) a gap in the recent tail is PROVISIONAL and does not enter the cost estimate.
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "l2_continuity", ROOT / "scripts" / "vol" / "l2_continuity_check.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["l2_continuity"] = C
_spec.loader.exec_module(C)

SNAP_PER_HOUR = 720  # IT/EN: cadenza reale del recorder (1 snapshot / 5 s) / real recorder rate


def _ts(hours: int, last_hour_snaps: int = SNAP_PER_HOUR,
        drop: tuple[int, ...] = ()) -> pd.DatetimeIndex:
    # IT: costruisce `hours` ore piene a partire da un'origine fissa, con le ore in `drop`
    #     rimosse e l'ultima ora troncata a `last_hour_snaps` (l'ora in corso).
    # EN: builds `hours` full hours from a fixed origin, with the hours in `drop` removed
    #     and the last hour truncated to `last_hour_snaps` (the in-progress hour).
    out = []
    t0 = pd.Timestamp("2026-06-01 00:00", tz="UTC")
    for h in range(hours):
        if h in drop:
            continue
        n = last_hour_snaps if h == hours - 1 else SNAP_PER_HOUR
        out.append(pd.date_range(t0 + pd.Timedelta(hours=h), periods=n, freq="5s"))
    return pd.DatetimeIndex([]).append(out).sort_values()


def test_in_progress_hour_never_breaks_the_run():
    # IT: il difetto originario: sotto la soglia di 360 snapshot l'ora in corso risultava
    #     "buco" e azzerava il run corrente. L'esito dipendeva dal minuto di esecuzione,
    #     che e' la firma di un artefatto di misura, non di un fatto sui dati.
    # EN: the original defect: below the 360-snapshot threshold the in-progress hour read
    #     as a "gap" and zeroed the current run. The verdict depended on the minute of
    #     execution, the signature of a measurement artifact rather than a fact.
    for snaps in (1, 60, 359, 361, SNAP_PER_HOUR):
        r = C.analyze(_ts(200, last_hour_snaps=snaps), days=7)
        assert r["cur"] == 199, f"run corrente rotto con {snaps} snapshot nell'ora in corso"
        assert r["firm"] == [] and r["prov"] == []


def test_consolidated_interior_gap_is_still_a_full_alarm():
    # IT: il monitor esiste per questo caso; renderlo tollerante alla coda non deve
    #     ammorbidirlo sulle ore gia' consolidate.
    # EN: the monitor exists for this case; tolerating the tail must not soften it on
    #     hours already consolidated.
    r = C.analyze(_ts(200, drop=(100,)), days=30)
    assert len(r["firm"]) == 1 and r["prov"] == []
    a, b = r["firm"][0]
    assert r["span"][a] == pd.Timestamp("2026-06-05 04:00", tz="UTC") and b == a
    # IT: il run si spezza in due -> quello corrente vale solo il tratto dopo il buco.
    # EN: the run splits in two -> the current one is only the stretch after the gap.
    assert r["cur"] == 98


def test_tail_gap_is_provisional_and_not_costed():
    # IT: coda non consolidata = ritardo di pull finche' non sopravvive a un secondo pull.
    #     Contarla come perdita e' esattamente l'errore ritrattato.
    # EN: unconsolidated tail = pull lag until it survives a second pull. Counting it as a
    #     loss is precisely the retracted mistake.
    r = C.analyze(_ts(200, drop=(196,)), days=7)
    assert r["firm"] == [] and len(r["prov"]) == 1
    # IT: la stessa ora, con una coda provvisoria piu' stretta, torna a essere un allarme:
    #     la distinzione e' la finestra dichiarata, non una tolleranza nascosta.
    # EN: the same hour, with a narrower provisional tail, becomes an alarm again: the
    #     distinction is the declared window, not a hidden tolerance.
    r2 = C.analyze(_ts(200, drop=(196,)), days=7, provisional_h=2)
    assert len(r2["firm"]) == 1 and r2["prov"] == []


def test_single_in_progress_hour_yields_no_consolidated_span():
    # IT: caso degenere reale (primo avvio del recorder): l'unica ora presente e' quella
    #     in corso, quindi non esiste NIENTE di consolidato. Deve dirlo, non dividere per
    #     zero ne' inventare un run.
    # EN: real degenerate case (recorder's first start): the only hour present is the one
    #     in progress, so nothing is consolidated. It must say so, not divide by zero nor
    #     invent a run.
    r = C.analyze(_ts(1, last_hour_snaps=200), days=7)
    assert r["empty"] is True
    assert r["in_progress"] == pd.Timestamp("2026-06-01 00:00", tz="UTC")


def test_windows_and_n_eff_count_only_contiguous_runs():
    # IT: controllo dell'aritmetica pre-esistente su un caso a mano: un run di R ore
    #     produce max(R-(T+h)+1, 0) finestre, e n_eff le riscala per h.
    # EN: check of the pre-existing arithmetic on a hand case: a run of R hours yields
    #     max(R-(T+h)+1, 0) windows, and n_eff rescales them by h.
    r = C.analyze(_ts(200), days=7)
    assert r["windows"] == 199 - (C.T_WIN + C.H) + 1
    assert r["n_eff"] == r["windows"] / C.H
    # IT: sotto T+h non esiste nemmeno una finestra -> zero, non un numero piccolo.
    # EN: below T+h not even one window exists -> zero, not a small number.
    assert C.analyze(_ts(100), days=7)["windows"] == 0
