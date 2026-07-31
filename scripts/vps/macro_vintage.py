"""
IT: VINTAGE DELLO SNAPSHOT MACRO — stampa su stdout l'ultima data dell'indice di
    un `macro_features.parquet` nel formato `YYYYMMDD`, e nient'altro.
    Serve a `pull_vps_data.ps1` per nominare l'archivio datato dei vintage: il
    NOME DEL FILE diventa l'identita' del vintage, cosi' lo snapshot che
    alimenta il live smette di essere uno stato mutevole sovrascritto a ogni
    pull e diventa un artefatto versionato e recuperabile (il breakpoint del
    2026-07-31 non fu misurabile direttamente proprio perche' il file vecchio
    era stato sovrascritto e non e' in git).
    Stdout PULITO by design (la sola data): ogni diagnostica va su stderr,
    altrimenti il chiamante PowerShell la scambierebbe per il vintage.
EN: MACRO SNAPSHOT VINTAGE - prints the last index date of a
    `macro_features.parquet` to stdout as `YYYYMMDD`, and nothing else.
    Used by `pull_vps_data.ps1` to name the dated vintage archive: the FILENAME
    becomes the vintage's identity, so the snapshot feeding the live path stops
    being mutable state overwritten on every pull and becomes a versioned,
    recoverable artifact (the 2026-07-31 breakpoint could not be measured
    directly precisely because the old file had been overwritten and is not in
    git).
    Stdout is CLEAN by design (the date only): diagnostics go to stderr, or the
    PowerShell caller would mistake them for the vintage.

Uso / Usage:
    python scripts/vps/macro_vintage.py data/macro_features.parquet
"""
import sys
from pathlib import Path


def main() -> int:
    # IT: boilerplate UTF-8 (console Windows cp1252: qualunque unicode crasha).
    # EN: UTF-8 boilerplate (Windows cp1252 console: any unicode would crash).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"macro_vintage: file assente / missing: {path}", file=sys.stderr)
        return 1

    # IT: import locale: l'helper e' chiamato anche quando pandas non serve al
    #     resto del pull, e un import in testa ne farebbe un prerequisito duro.
    # EN: local import: the helper is called even when the rest of the pull does
    #     not need pandas, and a top-level import would make it a hard prereq.
    import pandas as pd

    df = pd.read_parquet(path)
    if len(df) == 0:
        print(f"macro_vintage: parquet vuoto / empty: {path}", file=sys.stderr)
        return 1

    # IT: il vintage e' l'ULTIMA data dell'indice, cioe' lo stato del mondo piu'
    #     recente incorporato nel file — non la data di scrittura su disco, che
    #     cambia anche per un semplice ri-salvataggio senza dati nuovi.
    # EN: the vintage is the LAST index date, i.e. the most recent state of the
    #     world embedded in the file - not the on-disk write time, which changes
    #     even on a plain re-save with no new data.
    print(pd.Timestamp(df.index[-1]).strftime("%Y%m%d"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
