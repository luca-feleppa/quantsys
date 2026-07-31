# IT: MONITOR DI CONTINUITA' DEL RECORDER L2 — sorveglia la sola risorsa del progetto
#     che cresce col tempo invece che con gli esperimenti.
#     Perche' serve, e perche' la FRESCHEZZA non basta: il campione utile del filone
#     order-book non e' fatto di ore raccolte ma di ore CONTIGUE. Una finestra richiede
#     T+h barre consecutive, quindi UN'ORA di buco non costa un'ora: costa T+h-1
#     finestre, cioe' ~149 finestre = 6.2 giorni di accumulo a T=120/h=30. Il blocco ③
#     della routine controllava solo la freschezza dell'ultimo file: un file fresco con
#     un buco DENTRO passava il check senza dire nulla.
#     Il precedente e' documentato: l'epoca "casa" (16/06 -> 18/07 2026) ha raccolto 32
#     giorni al 29% di copertura e ha prodotto ZERO finestre utilizzabili - nessun run
#     contiguo ha mai raggiunto le 150 ore. Quel dato non esiste.
#     Read-only, solo timestamp: nessun valore di feature, nessuna relazione col target.
# EN: L2 RECORDER CONTINUITY MONITOR — watches the one project resource that grows with
#     time rather than with experiments.
#     Why FRESHNESS is not enough: the order-book line's usable sample is not made of
#     hours collected but of CONTIGUOUS hours. A window needs T+h consecutive bars, so
#     ONE hour of gap does not cost one hour: it costs T+h-1 windows, i.e. ~149 windows
#     = 6.2 days of accrual at T=120/h=30. The routine's block ③ only checked the last
#     file's freshness: a fresh file with a gap INSIDE passed silently.
#     The precedent is documented: the "home" epoch (2026-06-16 -> 07-18) collected 32
#     days at 29% coverage and produced ZERO usable windows - no contiguous run ever
#     reached 150 hours. That data does not exist.
#     Read-only, timestamps only: no feature values, no relationship with the target.
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

L2_GLOB = "data/orderbook/l2_features_*.parquet"
MIN_SNAP_PER_HOUR = 360     # IT/EN: stessa soglia del giudice B1 / same threshold as the B1 judge
T_WIN, H = 120, 30          # IT/EN: configurazione di produzione / production configuration


def runs_of_true(mask: np.ndarray) -> list[int]:
    out, c = [], 0
    for v in mask:
        if v:
            c += 1
        else:
            out.append(c)
            c = 0
    out.append(c)
    return [r for r in out if r > 0]


def main() -> int:
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Monitor di continuita' del recorder L2 / L2 recorder continuity monitor")
    ap.add_argument("--days", type=int, default=7,
                    help="finestra recente per l'elenco buchi / recent window for the gap list")
    args = ap.parse_args()

    files = sorted(glob.glob(L2_GLOB))
    if not files:
        print("[l2] nessun file order-book / no order-book file")
        return 0

    ts = pd.concat([pd.read_parquet(f, columns=["timestamp"]) for f in files])["timestamp"]
    ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).sort_values()
    per_h = pd.Series(1, index=ts).resample("1h").sum()
    span = pd.date_range(ts[0].floor("h"), ts[-1].floor("h"), freq="1h")
    ok = per_h.reindex(span, fill_value=0).values >= MIN_SNAP_PER_HOUR

    runs = runs_of_true(ok)
    # IT: il run CORRENTE (quello che termina all'ultima ora) e' l'unico che puo' ancora
    #     crescere: e' la quantita' da proteggere, non il massimo storico.
    # EN: the CURRENT run (the one ending at the last hour) is the only one that can
    #     still grow: it is the quantity to protect, not the historical maximum.
    cur = 0
    for v in ok[::-1]:
        if not v:
            break
        cur += 1

    def windows(rs):
        return int(sum(max(r - (T_WIN + H) + 1, 0) for r in rs))

    n_eff = windows(runs) / H
    print(f"[l2] copertura/coverage {ok.mean():.1%} su {len(span)} ore · "
          f"run corrente/current run {cur}h · run max {max(runs) if runs else 0}h")
    print(f"[l2] finestre T={T_WIN}/h={H}: {windows(runs)} · n_eff {n_eff:.1f} "
          f"(gate vol: 216) · costo di 1h di buco: {T_WIN + H - 1} finestre "
          f"= {(T_WIN + H - 1) / H / (24 / H):.1f} giorni/days")

    # ── buchi recenti · recent gaps ─────────────────────────────────────────
    lo = span[-1] - pd.Timedelta(days=args.days)
    recent = span >= lo
    idx, gaps = np.where(~ok & recent)[0], []
    for i in idx:
        if gaps and i == gaps[-1][1] + 1:
            gaps[-1] = (gaps[-1][0], i)
        else:
            gaps.append((i, i))
    if gaps:
        tot = sum(b - a + 1 for a, b in gaps)
        print(f"[l2] ⚠ {len(gaps)} BUCHI negli ultimi {args.days}g ({tot}h perse) — "
              f"ogni buco spezza il run / each gap breaks the run:")
        for a, b in gaps[-10:]:
            print(f"[l2]    {span[a]:%Y-%m-%d %H:%M} UTC · {b - a + 1}h")
        # IT: il costo si conta PER BLOCCO, non per ora: un buco di 41 ore spezza il run
        #     UNA volta, quindi distrugge le T+H-1 finestre che lo avrebbero scavalcato,
        #     non 41 volte quel numero. A cio' si somma la mancata produzione delle ore
        #     stesse (1 finestra/ora). Contarlo per ora sovrastimava di un ordine di
        #     grandezza e avrebbe reso il warning inutile perche' palesemente assurdo.
        # EN: cost is counted PER BLOCK, not per hour: a 41-hour gap breaks the run ONCE,
        #     destroying the T+H-1 windows that would have straddled it, not 41 times
        #     that. Add the hours' own lost production (1 window/hour). Counting per hour
        #     overstated by an order of magnitude and would have made the warning useless
        #     because visibly absurd.
        lost = len(gaps) * (T_WIN + H - 1) + tot
        print(f"[l2] ⚠ costo stimato/estimated cost: ~{lost} finestre = ~{lost / H:.0f} n_eff "
              f"= ~{lost / H / (24 / H):.0f} giorni di accumulo / days of accrual "
              f"({len(gaps)} rotture x {T_WIN + H - 1} + {tot}h non prodotte)")
    else:
        print(f"[l2] nessun buco negli ultimi {args.days}g / no gap in the last {args.days}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
