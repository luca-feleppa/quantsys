# IT: Contratto di aliasing di NpzFile — invariante su cui poggia la sicurezza di
#     `astype(np.float32, copy=False)` seguito da un `clamp_` IN-PLACE nel
#     caricamento dataset di `02_train.py`.
#
#     Con `copy=False` su un membro già float32, `astype` restituisce lo STESSO
#     oggetto ndarray, `torch.from_numpy` ne condivide il buffer e `clamp_` lo
#     MUTA. L'operazione è sicura solo se `NpzFile.__getitem__` materializza un
#     array FRESCO e non condiviso a ogni accesso: se una versione futura di numpy
#     iniziasse a cachare il membro (o a mapparlo in memoria e a condividerlo tra
#     letture), il clamp diventerebbe una mutazione silenziosa di stato condiviso —
#     una lettura successiva della stessa chiave restituirebbe dati già clippati,
#     e il bug sarebbe invisibile (nessuna eccezione, solo numeri diversi).
#     Questi test cadono nel momento in cui quell'ipotesi smette di valere.
#
# EN: NpzFile aliasing contract — the invariant that makes
#     `astype(np.float32, copy=False)` followed by an IN-PLACE `clamp_` safe in
#     the dataset load of `02_train.py`.
#
#     With `copy=False` on an already-float32 member, `astype` returns the SAME
#     ndarray object, `torch.from_numpy` shares its buffer and `clamp_` MUTATES it.
#     That is safe only as long as `NpzFile.__getitem__` materialises a FRESH,
#     unshared array on every access: were a future numpy to cache the member (or
#     memory-map it and share it across reads), the clamp would silently mutate
#     shared state — a later read of the same key would return already-clipped
#     data, and the bug would be invisible (no exception, just different numbers).
#     These tests fail the moment that assumption stops holding.
import sys
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as npy_fmt
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# IT: chiavi caricate via `to_t()` in 02_train.py — le uniche per cui `copy=False`
#     è rilevante (le altre non finiscono in un tensore mutato in-place).
# EN: keys loaded through `to_t()` in 02_train.py — the only ones for which
#     `copy=False` matters (the rest never reach an in-place-mutated tensor).
_TO_T_KEYS = (
    "X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
    "X_macro_train", "X_macro_val", "X_macro_test",
)


# IT: npz sintetico minuscolo con la stessa forma logica del dataset di training
#     (finestra 3D + target 1D + macro 2D). Volutamente piccolo: l'invariante è
#     sul contratto di NpzFile, non sulla dimensione.
# EN: tiny synthetic npz mirroring the logical shape of the training dataset (3D
#     window + 1D target + 2D macro). Deliberately small: the invariant is about
#     the NpzFile contract, not about size.
@pytest.fixture
def small_npz(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "mini_dataset.npz"
    np.savez(
        p,
        X_train=rng.normal(size=(8, 6, 4)).astype(np.float32) * 10.0,
        y_train=rng.normal(size=8).astype(np.float32),
        X_macro_train=rng.normal(size=(8, 3)).astype(np.float32),
        t_train=np.arange("2020-01-01", "2020-01-09", dtype="datetime64[D]").astype("datetime64[ms]"),
    )
    return p


# ─────────── invariante 1: ogni accesso restituisce un array fresco ───────────

# IT: due letture consecutive della stessa chiave devono dare oggetti DISTINTI
#     con buffer DISGIUNTI. È il cardine: se cade, `clamp_` muta stato condiviso.
# EN: two consecutive reads of the same key must yield DISTINCT objects with
#     DISJOINT buffers. This is the crux: if it breaks, `clamp_` mutates shared state.
def test_npzfile_getitem_returns_fresh_unshared_array(small_npz):
    data = np.load(str(small_npz), allow_pickle=True)
    a1 = data["X_train"]
    a2 = data["X_train"]
    assert a1 is not a2, "NpzFile ha iniziato a cachare i membri / started caching members"
    assert id(a1) != id(a2)
    assert not np.shares_memory(a1, a2), (
        "due letture della stessa chiave condividono memoria: `clamp_` in-place "
        "muterebbe stato condiviso / two reads of the same key share memory: "
        "in-place `clamp_` would mutate shared state"
    )


# IT: la mutazione di una lettura non deve propagarsi né a una lettura già in vita
#     né a una successiva (il buffer su disco resta la sorgente di verità).
# EN: mutating one read must propagate neither to a read already alive nor to a
#     later one (the on-disk buffer stays the source of truth).
def test_mutating_one_read_does_not_leak_to_other_reads(small_npz):
    data = np.load(str(small_npz), allow_pickle=True)
    a1 = data["X_train"]
    a2 = data["X_train"]
    original = float(a1[0, 0, 0])
    sentinel = -12345.0
    a1[0, 0, 0] = sentinel

    assert float(a2[0, 0, 0]) == pytest.approx(original), \
        "la mutazione è filtrata in una lettura viva / mutation leaked into a live read"
    assert float(data["X_train"][0, 0, 0]) == pytest.approx(original), \
        "la mutazione è filtrata in una lettura nuova / mutation leaked into a fresh read"


# IT: gli array restituiti devono essere SCRIVIBILI. Se numpy passasse a un buffer
#     read-only (es. `frombuffer` su bytes immutabili, o un mmap read-only),
#     `torch.from_numpy` emetterebbe solo un UserWarning e `clamp_` scriverebbe
#     comunque su memoria non scrivibile — comportamento indefinito, non un errore.
# EN: returned arrays must be WRITEABLE. Were numpy to switch to a read-only buffer
#     (e.g. `frombuffer` over immutable bytes, or a read-only mmap),
#     `torch.from_numpy` would merely emit a UserWarning and `clamp_` would still
#     write into non-writeable memory — undefined behaviour, not an error.
def test_npz_arrays_are_writeable(small_npz):
    data = np.load(str(small_npz), allow_pickle=True)
    for key in ("X_train", "y_train", "X_macro_train"):
        arr = data[key]
        assert arr.flags.writeable, f"{key} non scrivibile / not writeable"
        assert arr.flags.c_contiguous, f"{key} non C-contiguo / not C-contiguous"


# ───────── invariante 2: semantica di astype(copy=False) su float32 ─────────

# IT: su un array già float32 e C-contiguo `astype(copy=False)` restituisce lo
#     STESSO oggetto (nessuna allocazione) — è il guadagno cercato, ed è anche la
#     ragione per cui il test sopra è obbligatorio. Su dtype diverso copia sempre.
# EN: on an already-float32, C-contiguous array `astype(copy=False)` returns the
#     SAME object (no allocation) — that is the sought-after saving, and also why
#     the test above is mandatory. On a different dtype it always copies.
def test_astype_copy_false_is_identity_on_float32(small_npz):
    data = np.load(str(small_npz), allow_pickle=True)
    src = data["X_train"]
    assert src.dtype == np.float32
    same = src.astype(np.float32, copy=False)
    assert same is src
    assert np.shares_memory(same, src)

    # IT: controllo positivo — copy=True (default) alloca sempre.
    # EN: positive control — copy=True (the default) always allocates.
    copied = src.astype(np.float32)
    assert copied is not src
    assert not np.shares_memory(copied, src)

    # IT: dtype diverso → copia anche con copy=False (nessuna scorciatoia silenziosa).
    # EN: different dtype → copies even with copy=False (no silent shortcut).
    f64 = src.astype(np.float64)
    conv = f64.astype(np.float32, copy=False)
    assert conv is not f64
    assert not np.shares_memory(conv, f64)


# ───────── invariante 3: equivalenza end-to-end del pattern di 02_train ─────────

# IT: replica esatta di `to_t()` + `clamp_` in-place di 02_train.py, nei due rami.
#     Il risultato deve essere BIT-identico e il membro npz deve restare intatto.
# EN: exact replica of `to_t()` + in-place `clamp_` from 02_train.py, both branches.
#     The result must be BIT-identical and the npz member must stay untouched.
def test_to_t_clamp_pattern_bit_identical_both_branches(small_npz):
    data = np.load(str(small_npz), allow_pickle=True)
    n_feat = data["X_train"].shape[2]

    flat = data["X_train"].reshape(-1, n_feat)
    lo = torch.from_numpy(np.nanpercentile(flat, 10.0, axis=0).astype(np.float32))
    hi = torch.from_numpy(np.nanpercentile(flat, 90.0, axis=0).astype(np.float32))
    del flat

    x_copy = torch.from_numpy(data["X_train"].astype(np.float32))
    x_view = torch.from_numpy(data["X_train"].astype(np.float32, copy=False))
    x_copy.clamp_(lo, hi)
    x_view.clamp_(lo, hi)

    assert torch.equal(x_copy, x_view), \
        "copy=False cambia il risultato numerico / copy=False changes the numeric result"
    # IT: il clamp ha davvero morso (altrimenti il confronto sarebbe vacuo).
    # EN: the clamp actually bit (otherwise the comparison would be vacuous).
    assert float(x_view.max()) < float(torch.from_numpy(data["X_train"]).max())
    # IT: il membro npz è ancora quello originale, non clippato.
    # EN: the npz member is still the original, unclipped one.
    assert float(data["X_train"].max()) > float(x_view.max())


# IT: nessuna chiave caricata via `to_t()` deve essere letta due volte da
#     02_train.py: sotto copy=False la seconda lettura sarebbe comunque un array
#     fresco (quindi corretta) ma pagherebbe una I/O completa in silenzio.
#     Il test è una sentinella sul codice sorgente, non sul runtime.
# EN: no key loaded through `to_t()` may be read twice by 02_train.py: under
#     copy=False the second read would still be a fresh array (hence correct) but
#     would silently pay a full I/O. This is a source-level sentinel, not a runtime one.
def test_02_train_reads_each_to_t_key_once():
    src = (ROOT / "scripts" / "02_train.py").read_text(encoding="utf-8")
    for key in _TO_T_KEYS:
        n_direct = src.count(f'data["{key}"]')
        n_to_t = src.count(f'to_t("{key}")')
        assert n_direct == 0, (
            f'{key}: accesso diretto data["{key}"] oltre a to_t() — '
            f'verificare che non sia una rilettura post-clamp / direct access '
            f'besides to_t() — check it is not a post-clamp re-read'
        )
        # IT: `== 1` e non `<= 1`: con `<= 1` il test passerebbe anche a chiave
        #     sparita, cioè sarebbe vacuo. Oggi tutte e 9 compaiono esattamente
        #     una volta (misurato); se una viene rimossa il test va aggiornato
        #     consapevolmente, non silenziosamente.
        # EN: `== 1`, not `<= 1`: with `<= 1` the test would also pass on a
        #     vanished key, i.e. be vacuous. Today all 9 appear exactly once
        #     (measured); if one is removed the test must be updated knowingly,
        #     not silently.
        assert n_to_t == 1, f"{key}: caricata {n_to_t} volte via to_t() / loaded {n_to_t} times"


# ─────────── il dataset di produzione è davvero float32 su disco ───────────

# IT: `copy=False` risparmia memoria SOLO se i membri sono già float32; su un npz
#     float64 astype copierebbe comunque. Verificato leggendo i soli header .npy
#     dallo zip — nessun array materializzato (il dataset reale pesa ~3.26 GB).
#     Skip se l'npz non è su disco: è rigenerabile e può legittimamente mancare.
# EN: `copy=False` saves memory ONLY if the members are already float32; on a
#     float64 npz astype would copy anyway. Verified by reading just the .npy
#     headers out of the zip — no array materialised (the real dataset is ~3.26 GB).
#     Skipped when the npz is absent: it is regenerable and may legitimately be missing.
def test_production_npz_members_are_float32():
    npz = ROOT / "data" / "lstm_dataset.npz"
    if not npz.exists():
        pytest.skip(f"{npz} assente (rigenerabile) / absent (regenerable)")

    dtypes = {}
    with zipfile.ZipFile(npz) as zf:
        for name in zf.namelist():
            with zf.open(name) as fh:
                version = npy_fmt.read_magic(fh)
                # IT: solo API pubbliche — `_read_array_header` è privata e ha già
                #     cambiato firma tra versioni di numpy.
                # EN: public API only — `_read_array_header` is private and has
                #     already changed signature across numpy versions.
                if version == (1, 0):
                    shape, _fortran, dtype = npy_fmt.read_array_header_1_0(fh)
                elif version == (2, 0):
                    shape, _fortran, dtype = npy_fmt.read_array_header_2_0(fh)
                else:
                    pytest.skip(f"formato .npy {version} non gestito / unhandled")
            dtypes[Path(name).stem] = (dtype, shape)

    present = [k for k in _TO_T_KEYS if k in dtypes]
    assert present, "nessuna chiave to_t nel dataset / no to_t key in the dataset"
    for key in present:
        dtype, shape = dtypes[key]
        assert dtype == np.dtype(np.float32), (
            f"{key} è {dtype} su disco: astype(copy=False) copierebbe comunque / "
            f"{key} is {dtype} on disk: astype(copy=False) would copy anyway"
        )
