"""
IT: Test del MacroNormalizer PINNATO (`quantsys.model.vol_forecaster.macro_snapshot`).
    Tre proprieta' vanno dimostrate, non assunte.
    (1) INERZIA: senza path esplicito il ramo legacy deve dare esattamente cio' che
        dava prima — altrimenti la modifica avrebbe cambiato l'input del live mentre
        due campioni forward sono aperti.
    (2) IL PIN PINNA DAVVERO: allungare il parquet deve muovere il risultato nel ramo
        refit e NON muoverlo nel ramo pinnato. Un test che non allunga il parquet
        passerebbe anche con un pin rotto, perche' su dati identici i due rami
        coincidono per costruzione.
    (3) IL GUARD FAIL-FAST sulle colonne: un pin applicato a un parquet con colonne
        aggiunte o riordinate userebbe mediana e IQR della colonna SBAGLIATA, in
        silenzio, su ogni tick del live. Deve alzare, non passare.
EN: Tests for the PINNED MacroNormalizer (`vol_forecaster.macro_snapshot`).
    Three properties must be demonstrated, not assumed.
    (1) INERTIA: with no explicit path the legacy branch must return exactly what it
        used to — otherwise the change would have altered the live input while two
        forward samples are open.
    (2) THE PIN ACTUALLY PINS: extending the parquet must move the refit branch and
        NOT move the pinned one. A test that does not extend the parquet would pass
        even with a broken pin, since on identical data the branches coincide by
        construction.
    (3) THE FAIL-FAST GUARD on columns: a pin applied to a parquet with added or
        reordered columns would use the WRONG column's median and IQR, silently, on
        every live tick. It must raise, not pass.
"""
import numpy as np
import pandas as pd
import pytest

from quantsys.macro.regime import MacroNormalizer
from quantsys.model.vol_forecaster import MACRO_NORM_REFIT, macro_snapshot


def _macro(n, seed=0, cols=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="D")
    return pd.DataFrame({c: rng.normal(i, 1 + i, n) for i, c in enumerate(cols)}, index=idx)


def _legacy(df, cols):
    # IT: aritmetica legacy riprodotta a mano dal codice pre-modifica.
    # EN: legacy arithmetic reproduced by hand from the pre-change code.
    n = MacroNormalizer()
    n.fit_transform(df, cols)
    last = df[cols].iloc[[-1]].fillna(0.0)
    return np.clip(n.scaler.transform(last.values.astype(np.float32)), -5, 5).astype(np.float32)


def test_default_is_refit_and_is_bit_identical_to_legacy():
    # IT: (1) inerzia. Il default DEVE essere il ramo storico, e coincidere bit a bit.
    # EN: (1) inertia. The default MUST be the legacy branch, bit for bit.
    df = _macro(400)
    cols = list(df.columns)
    got, info = macro_snapshot(df, cols)
    assert info["mode"] == MACRO_NORM_REFIT and info["pinned_vintage"] is None
    np.testing.assert_array_equal(got, _legacy(df, cols))


def test_pin_holds_the_instrument_still_while_refit_drifts(tmp_path):
    # IT: (2) il test che conta. Stesso ULTIMO dato, storia piu' lunga:
    #     - refit  -> il risultato CAMBIA (mediana/IQR ri-stimate: e' la deriva
    #       dello strumento misurata al 2.7% sul breakpoint del 31/07);
    #     - pinnato -> il risultato NON cambia (lo strumento e' fermo).
    # EN: (2) THE test. Same LAST row, longer history:
    #     - refit  -> the result CHANGES (median/IQR refitted: the instrument drift
    #       measured at 2.7% on the 31/07 breakpoint);
    #     - pinned -> the result does NOT change (the instrument is frozen).
    short = _macro(300, seed=1)
    long = pd.concat([_macro(200, seed=2), short])  # IT/EN: stessa ultima riga / same last row
    cols = list(short.columns)
    assert (short.iloc[-1] == long.iloc[-1]).all()

    pin = tmp_path / "pin.pkl"
    n = MacroNormalizer()
    n.fit_transform(short, cols)
    n.pinned_vintage = "20260730"
    n.save(str(pin))

    refit_short, _ = macro_snapshot(short, cols)
    refit_long, _ = macro_snapshot(long, cols)
    pin_short, i1 = macro_snapshot(short, cols, str(pin))
    pin_long, i2 = macro_snapshot(long, cols, str(pin))

    assert not np.allclose(refit_short, refit_long), \
        "il ramo refit non deriva: il test non prova nulla / refit does not drift"
    np.testing.assert_array_equal(pin_short, pin_long)
    np.testing.assert_allclose(pin_short, refit_short, rtol=1e-6)  # IT/EN: pin == refit sul SUO vintage
    assert i1["pinned_vintage"] == i2["pinned_vintage"] == "20260730"


def test_pin_rejects_added_or_reordered_columns(tmp_path):
    # IT: (3) guard fail-fast. Applicare mediana e IQR alla colonna sbagliata e'
    #     silenzioso e permanente: deve essere impossibile.
    # EN: (3) fail-fast guard. Applying the wrong column's median and IQR is silent
    #     and permanent: it must be impossible.
    df = _macro(200)
    cols = list(df.columns)
    pin = tmp_path / "pin.pkl"
    n = MacroNormalizer()
    n.fit_transform(df, cols)
    n.save(str(pin))

    grown = df.assign(d=1.0)
    with pytest.raises(RuntimeError, match="incompatibile|incompatible"):
        macro_snapshot(grown, list(grown.columns), str(pin))

    reordered = df[["c", "a", "b"]]
    with pytest.raises(RuntimeError, match="incompatibile|incompatible"):
        macro_snapshot(reordered, list(reordered.columns), str(pin))


def test_vintage_round_trips_and_legacy_pickles_still_load(tmp_path):
    # IT: il campo e' OPZIONALE: i pickle scritti da 01b prima della modifica devono
    #     caricarsi senza migrazione, con vintage None.
    # EN: the field is OPTIONAL: pickles written by 01b before the change must load
    #     without migration, with a None vintage.
    df = _macro(100)
    n = MacroNormalizer()
    n.fit_transform(df, list(df.columns))
    n.pinned_vintage = "20260730"
    p = tmp_path / "v.pkl"
    n.save(str(p))
    assert MacroNormalizer.load(str(p)).pinned_vintage == "20260730"

    import pickle
    legacy = tmp_path / "legacy.pkl"
    with open(legacy, "wb") as f:
        pickle.dump({"scaler": n.scaler, "feature_cols": n.feature_cols}, f)
    old = MacroNormalizer.load(str(legacy))
    assert old.pinned_vintage is None and old.fitted


def test_data_vintage_is_reported_separately_from_the_instrument(tmp_path):
    # IT: strumento e stato sono due cose diverse e il chiamante deve poterle
    #     stampare separate — confonderle e' il malinteso che il breakpoint del
    #     31/07 ha dovuto decomporre a mano.
    # EN: instrument and state are two different things and the caller must be able
    #     to log them separately — conflating them is the confusion the 31/07
    #     breakpoint had to decompose by hand.
    df = _macro(50)
    _, info = macro_snapshot(df, list(df.columns))
    assert info["data_vintage"] == str(df.index[-1].date())
