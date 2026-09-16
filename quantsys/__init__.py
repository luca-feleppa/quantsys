"""QUANTSYS — Neural Forecasting Engine for Crypto Trading."""
# Package version, exposed for logging and checkpoint metadata.
__version__ = "0.1.0"

# ── DLL initialization order: pyarrow BEFORE torch/sklearn ────────────────────
# On Windows, loading pyarrow AFTER both torch AND scikit-learn are already
# in the process yields an access violation (exit 139, no Python traceback)
# at the first `pd.read_parquet`. Both are required: torch alone or sklearn
# alone is fine — it is the clash between the OpenMP runtimes each ships
# (torch's `libiomp5md.dll` vs scipy/sklearn's) that breaks the late load of
# the Arrow DLLs.
# The numbered scripts survive only because they import `pandas` before
# `torch`: a DE FACTO invariant, never stated nor tested. Since
# `quantsys.utils` imports torch at module level, a new script importing the
# project first and pandas second would crash.
# Anchoring the import here, at the package ROOT, makes any `import
# quantsys.*` initialize Arrow first, turning the ordering into a property
# of the package rather than a coincidence of the call site.
# Best-effort: pyarrow is not a declared dependency (pyproject declares
# none) and the VPS collectors may lack it — its absence must not block
# importing quantsys.
# Regression test: tests/test_import_order.py
# ⚠ The two failure cases are NOT equivalent and must be told apart:
# · `ImportError` = pyarrow absent (minimal VPS) → legitimate, silent.
# · any other exception = pyarrow PRESENT but broken (numpy ABI mismatch,
#   corrupted pip install): swallowing it silently would fail the preload
#   while suggesting the anchor worked, and the first `read_parquet` would
#   retry the import with torch+sklearn already resident — precisely the
#   access violation this block exists to prevent, only moved downstream and
#   untraceable. It must be surfaced.
try:
    import pyarrow as _pyarrow  # noqa: F401
except ImportError:  # pragma: no cover - envs without pyarrow
    pass
except Exception as _e:  # pragma: no cover - pyarrow present but broken
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
