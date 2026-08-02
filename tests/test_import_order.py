# IT: Regression test — ORDINE DI INIZIALIZZAZIONE DLL (pyarrow prima di torch+sklearn).
#     Su Windows, caricare pyarrow dopo che torch E scikit-learn sono gia' nel processo
#     produce un'access violation (exit 139) al primo pd.read_parquet: non un'eccezione
#     Python, un crash del processo — quindi NON catturabile con pytest.raises.
#     Il test gira percio' in un SUBPROCESSO e verifica il codice di uscita.
#     Gli script numerati sopravvivevano solo perche' importano pandas prima di torch:
#     un invariante di fatto, mai testato. `quantsys/__init__.py` lo rende una proprieta'
#     del package (import pyarrow ancorato alla radice) e questo test lo blocca.
# EN: Regression test — DLL INITIALIZATION ORDER (pyarrow before torch+sklearn).
#     On Windows, loading pyarrow after both torch AND scikit-learn are already in the
#     process yields an access violation (exit 139) at the first pd.read_parquet: not a
#     Python exception, a process crash — hence NOT catchable with pytest.raises.
#     The test therefore runs in a SUBPROCESS and checks the exit code.
#     The numbered scripts survived only because they import pandas before torch: a de
#     facto invariant, never tested. `quantsys/__init__.py` makes it a package property
#     (pyarrow import anchored at the root) and this test locks it in.

import subprocess
import sys
import textwrap

import pytest


# IT: esegue uno snippet in un interprete pulito, ritorna (returncode, stdout+stderr).
# EN: runs a snippet in a clean interpreter, returns (returncode, stdout+stderr).
def _run(code: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=300,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# IT: ① l'ordine che crashava — import del progetto PRIMA di pandas.
#     E' il modo naturale di scrivere uno script nuovo seguendo la checklist
#     (load_config da quantsys.utils in cima), quindi il caso da proteggere.
# EN: ① the order that used to crash — project imports BEFORE pandas.
#     It is the natural way to write a new script following the checklist
#     (load_config from quantsys.utils at the top), hence the case to protect.
def test_quantsys_utils_before_pandas_parquet(tmp_path):
    pq = tmp_path / "t.parquet"
    rc, out = _run(f"""
        from quantsys.utils import load_config          # importa torch
        from sklearn.preprocessing import RobustScaler  # noqa: F401
        import pandas as pd
        pd.DataFrame({{"a": [1.0, 2.0], "b": [3.0, 4.0]}}).to_parquet(r"{pq}")
        df = pd.read_parquet(r"{pq}")
        assert len(df) == 2
        print("OK")
    """)
    assert rc == 0, (
        f"crash con exit {rc} importando quantsys.utils prima di pandas "
        f"(regressione dell'ordine di init pyarrow):\n{out}"
    )
    assert "OK" in out


# IT: ② variante esplicita — torch e sklearn importati direttamente dopo quantsys.
# EN: ② explicit variant — torch and sklearn imported directly after quantsys.
def test_quantsys_root_import_then_torch_sklearn(tmp_path):
    pq = tmp_path / "t2.parquet"
    rc, out = _run(f"""
        import quantsys                                 # noqa: F401
        import torch                                    # noqa: F401
        import sklearn.preprocessing                    # noqa: F401
        import pandas as pd
        pd.DataFrame({{"a": [1.0]}}).to_parquet(r"{pq}")
        assert len(pd.read_parquet(r"{pq}")) == 1
        print("OK")
    """)
    assert rc == 0, f"crash con exit {rc}:\n{out}"
    assert "OK" in out


# IT: ③ l'import della radice deve realmente aver caricato pyarrow — e' il
#     meccanismo del fix, non un effetto collaterale su cui fare affidamento.
# EN: ③ the root import must actually have loaded pyarrow — that is the fix's
#     mechanism, not an incidental side effect to rely on.
def test_quantsys_root_preloads_pyarrow():
    pytest.importorskip("pyarrow")
    rc, out = _run("""
        import sys
        import quantsys                                 # noqa: F401
        assert "pyarrow" in sys.modules, "quantsys non ha pre-caricato pyarrow"
        print("OK")
    """)
    assert rc == 0, f"exit {rc}:\n{out}"
    assert "OK" in out


# IT: ④ l'import di quantsys non deve fallire se pyarrow manca (VPS minimale):
#     il pre-load e' best-effort, non una dipendenza nuova.
# EN: ④ importing quantsys must not fail when pyarrow is missing (minimal VPS):
#     the preload is best-effort, not a new dependency.
def test_quantsys_import_survives_missing_pyarrow():
    rc, out = _run("""
        import sys
        import builtins
        _real = builtins.__import__
        def _blocked(name, *a, **k):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("simulazione: pyarrow assente")
            return _real(name, *a, **k)
        builtins.__import__ = _blocked
        for m in [k for k in sys.modules if k.startswith(("quantsys", "pyarrow"))]:
            del sys.modules[m]
        import quantsys                                 # noqa: F401
        builtins.__import__ = _real
        print("OK")
    """)
    assert rc == 0, (
        f"import quantsys fallito con pyarrow assente (exit {rc}) — il pre-load "
        f"deve restare best-effort:\n{out}"
    )
    assert "OK" in out
