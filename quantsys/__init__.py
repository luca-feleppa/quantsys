"""QUANTSYS — Neural Forecasting Engine for Crypto Trading."""
# IT: Versione del package, esposta per logging e checkpoint metadata.
# EN: Package version, exposed for logging and checkpoint metadata.
__version__ = "0.1.0"

# ── Ordine di inizializzazione DLL: pyarrow PRIMA di torch/sklearn ────────────
# IT: Su Windows, caricare pyarrow DOPO che torch E scikit-learn sono gia' nel
#     processo produce un'access violation (exit 139, nessun traceback Python)
#     al primo `pd.read_parquet`. Serve la compresenza di entrambi: torch da solo
#     o sklearn da solo non basta — e' il conflitto fra i runtime OpenMP che
#     ciascuno porta (`libiomp5md.dll` di torch vs quello di scipy/sklearn) a
#     rompere il caricamento tardivo delle DLL Arrow.
#     Gli script numerati sopravvivono solo perche' importano `pandas` prima di
#     `torch`: un invariante di FATTO, mai dichiarato ne' testato. Poiche'
#     `quantsys.utils` importa torch a livello di modulo, uno script nuovo che
#     importi prima il progetto e poi pandas crasherebbe.
#     Ancorando l'import qui, alla RADICE del package, qualunque `import
#     quantsys.*` inizializza Arrow per primo e l'ordine diventa una proprieta'
#     del package invece che una coincidenza del call site.
#     Best-effort: pyarrow non e' una dipendenza dichiarata (pyproject non ne
#     dichiara alcuna) e i collector VPS potrebbero non averlo — l'assenza non
#     deve impedire l'import di quantsys.
#     Regression test: tests/test_import_order.py
# EN: On Windows, loading pyarrow AFTER both torch AND scikit-learn are already
#     in the process yields an access violation (exit 139, no Python traceback)
#     at the first `pd.read_parquet`. Both are required: torch alone or sklearn
#     alone is fine — it is the clash between the OpenMP runtimes each ships
#     (torch's `libiomp5md.dll` vs scipy/sklearn's) that breaks the late load of
#     the Arrow DLLs.
#     The numbered scripts survive only because they import `pandas` before
#     `torch`: a DE FACTO invariant, never stated nor tested. Since
#     `quantsys.utils` imports torch at module level, a new script importing the
#     project first and pandas second would crash.
#     Anchoring the import here, at the package ROOT, makes any `import
#     quantsys.*` initialize Arrow first, turning the ordering into a property
#     of the package rather than a coincidence of the call site.
#     Best-effort: pyarrow is not a declared dependency (pyproject declares
#     none) and the VPS collectors may lack it — its absence must not block
#     importing quantsys.
#     Regression test: tests/test_import_order.py
#     ⚠ I due casi di fallimento NON sono equivalenti e vanno distinti:
#     · `ImportError` = pyarrow assente (VPS minimale) → legittimo, silenzioso.
#     · qualunque altra eccezione = pyarrow PRESENTE ma rotto (mismatch ABI con
#       numpy, installazione pip corrotta): inghiottirla in silenzio farebbe
#       fallire il pre-load lasciando credere che l'ancora abbia funzionato, e
#       il primo `read_parquet` ritenterebbe l'import con torch+sklearn gia' in
#       processo — cioe' esattamente l'access violation che questo blocco esiste
#       per prevenire, ma spostata piu' a valle e senza traccia. Va segnalata.
# EN: ⚠ The two failure cases are NOT equivalent and must be told apart:
#     · `ImportError` = pyarrow absent (minimal VPS) → legitimate, silent.
#     · any other exception = pyarrow PRESENT but broken (numpy ABI mismatch,
#       corrupted pip install): swallowing it silently would fail the preload
#       while suggesting the anchor worked, and the first `read_parquet` would
#       retry the import with torch+sklearn already resident — precisely the
#       access violation this block exists to prevent, only moved downstream and
#       untraceable. It must be surfaced.
try:
    import pyarrow as _pyarrow  # noqa: F401
except ImportError:  # pragma: no cover - ambienti senza pyarrow / envs without pyarrow
    pass
except Exception as _e:  # pragma: no cover - pyarrow presente ma rotto / present but broken
    import warnings as _warnings
    _warnings.warn(
        f"quantsys: pre-caricamento di pyarrow fallito con "
        f"{type(_e).__name__}: {_e}. pyarrow risulta installato ma non "
        f"importabile: un successivo read_parquet puo' crashare il processo "
        f"(access violation) invece di sollevare un'eccezione. / "
        f"quantsys: pyarrow preload failed; it is installed but not importable, "
        f"a later read_parquet may crash the process instead of raising.",
        RuntimeWarning, stacklevel=2,
    )
