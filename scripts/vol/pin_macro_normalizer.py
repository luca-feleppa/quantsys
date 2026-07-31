# IT: PIN DEL MacroNormalizer — congela lo STRUMENTO di normalizzazione macro a un
#     vintage DICHIARATO, separandolo dallo STATO del mondo che deve misurare.
#     Il problema. `VolForecaster` ri-stima il MacroNormalizer whole-df a ogni
#     bootstrap di 04b: allungare `macro_features.parquet` sposta mediana e IQR,
#     quindi lo strumento cambia insieme al dato. Sul breakpoint del 2026-07-31 la
#     decomposizione ha attribuito il 2.7% della variazione totale alla sola deriva
#     dello strumento (L2 0.0891 su 3.2804) e il ~97% allo stato genuinamente piu'
#     fresco. Il 2.7% e' piccolo ma NON e' misura: e' rumore che si presenta come
#     segnale, e in un campione forward pre-registrato non deve esistere.
#     ⚠ IL VINTAGE DI RIFERIMENTO VA DICHIARATO, non dedotto: il vintage sotto cui
#     `models/itransformer` fu addestrato NON e' ricostruibile (il parquet di allora
#     e' stato sovrascritto e non e' in git). Questo script quindi non "recupera" il
#     vintage giusto: ne FISSA uno, esplicito e datato, e lo scrive nel pickle.
#     ⚠ INERTE finche' nessuno passa `--macro-norm` a 04b o al replay: creare il pin
#     non cambia il comportamento di nulla.
# EN: MacroNormalizer PIN — freezes the macro normalization INSTRUMENT at a DECLARED
#     vintage, separating it from the STATE of the world it must measure.
#     The problem. `VolForecaster` refits the MacroNormalizer whole-df at every 04b
#     bootstrap: extending `macro_features.parquet` moves median and IQR, so the
#     instrument moves together with the data. On the 2026-07-31 breakpoint the
#     decomposition attributed 2.7% of the total variation to instrument drift alone
#     and ~97% to genuinely fresher state. 2.7% is small but it is NOT measurement:
#     it is noise presenting as signal, and inside a pre-registered forward sample it
#     must not exist.
#     ⚠ THE REFERENCE VINTAGE MUST BE DECLARED, not inferred: the vintage
#     `models/itransformer` was trained under is NOT reconstructible (that parquet was
#     overwritten and is not in git). So this script does not "recover" the right
#     vintage: it FIXES one, explicit and dated, and writes it into the pickle.
#     ⚠ INERT until someone passes `--macro-norm` to 04b or the replay: creating the
#     pin changes nothing's behavior.
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantsys.macro.regime import MacroNormalizer  # noqa: E402

DEFAULT_SRC = ROOT / "data" / "macro_features.parquet"
DEFAULT_OUT = ROOT / "models" / "macro_normalizer_pinned.pkl"


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Pin del MacroNormalizer a un vintage dichiarato / "
                    "pin the MacroNormalizer at a declared vintage")
    ap.add_argument("--source", default=str(DEFAULT_SRC),
                    help="parquet macro su cui fittare (default: il canonico) / "
                         "macro parquet to fit on (default: the canonical one)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="pickle di destinazione / output pickle")
    ap.add_argument("--force", action="store_true",
                    help="sovrascrivi un pin esistente / overwrite an existing pin")
    args = ap.parse_args()

    src, out = Path(args.source), Path(args.out)
    if not src.exists():
        print(f"ERRORE: sorgente assente / missing source: {src}", file=sys.stderr)
        return 1
    # IT: un pin esistente NON si sovrascrive per sbaglio: e' l'artefatto che tiene
    #     fermo l'input del live, e sostituirlo in silenzio riaprirebbe esattamente
    #     il problema che il pin esiste per chiudere.
    # EN: an existing pin is NOT overwritten by accident: it is the artifact holding
    #     the live input still, and silently replacing it would reopen precisely the
    #     problem the pin exists to close.
    if out.exists() and not args.force:
        prev = MacroNormalizer.load(str(out))
        print(f"ERRORE: pin gia' presente / pin already exists: {out}\n"
              f"  vintage pinnato / pinned vintage: {getattr(prev, 'pinned_vintage', None)}\n"
              f"  sovrascrivere e' un ATTO DELIBERATO: ripassa con --force / "
              f"overwriting is a DELIBERATE act: re-run with --force", file=sys.stderr)
        return 1

    df = pd.read_parquet(src)
    cols = list(df.columns)
    vintage = pd.Timestamp(df.index[-1]).strftime("%Y%m%d")

    norm = MacroNormalizer()
    norm.fit_transform(df, cols)
    norm.pinned_vintage = vintage
    out.parent.mkdir(parents=True, exist_ok=True)
    norm.save(str(out))

    sha = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    print("=" * 74)
    print("PIN DEL MacroNormalizer / MacroNormalizer PIN")
    print("=" * 74)
    print(f"  sorgente / source        : {src.relative_to(ROOT)}  (sha256[:16] {sha})")
    print(f"  righe / rows             : {len(df)}   colonne / columns: {len(cols)}")
    print(f"  vintage DICHIARATO       : {vintage}  (ultima data dell'indice / last index date)")
    print(f"  pin scritto / pin written: {out.relative_to(ROOT)}")
    print()
    print("  INERTE: nessun consumer cambia comportamento finche' non gli si passa")
    print("  esplicitamente --macro-norm <path>. / INERT until --macro-norm is passed.")
    print(f"    python scripts/04b_vol_paper.py ... --macro-norm {out.relative_to(ROOT)}")
    print(f"    python scripts/vol/vol_paper_replay.py ... --macro-norm {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
