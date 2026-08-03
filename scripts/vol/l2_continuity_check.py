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
#
# IT: ⚠ COSA QUESTO CHECK MISURA DAVVERO (chiarito il 2026-08-03, dopo un falso
#     positivo costato una voce sbagliata in STATUS.md): misura la continuita' del
#     MIRROR LOCALE, non quella del recorder sul VPS. I due coincidono solo per le
#     ore gia' CONSOLIDATE. Due modi in cui la coda del mirror mente:
#       (a) l'ora IN CORSO e' parziale per costruzione — l'ultima ora dello span e'
#           quella che contiene l'ultimo tick, quindi ha meno di 720 snapshot; con
#           soglia 360 l'esito dipendeva dal MINUTO in cui girava la routine (prima
#           del minuto ~30 -> "run corrente 0h", dopo -> tutto verde);
#       (b) il pull scarica i giornalieri con scp SENZA atomicita' remota, quindi le
#           ultime ore possono arrivare in ritardo di un pull.
#     Percio': l'ora in corso e' ESCLUSA dallo span, e i buchi che cadono nella coda
#     recente sono marcati PROVVISORI ed esclusi dalla stima di costo. Un buco e'
#     un fatto solo dopo essere sopravvissuto a un secondo pull.
# EN: ⚠ WHAT THIS CHECK ACTUALLY MEASURES (clarified 2026-08-03, after a false
#     positive that cost a wrong STATUS.md entry): it measures the continuity of the
#     LOCAL MIRROR, not that of the VPS recorder. They agree only on hours already
#     CONSOLIDATED. Two ways the mirror's tail lies:
#       (a) the IN-PROGRESS hour is partial by construction — the span's last hour is
#           the one containing the last tick, so it holds fewer than 720 snapshots;
#           with a 360 threshold the verdict depended on the MINUTE the routine ran
#           (before minute ~30 -> "current run 0h", after -> all green);
#       (b) the pull fetches dailies over scp with no remote atomicity, so the last
#           hours can lag by one pull.
#     Hence: the in-progress hour is EXCLUDED from the span, and gaps falling in the
#     recent tail are flagged PROVISIONAL and excluded from the cost estimate. A gap
#     is a fact only after it survives a second pull.
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
# IT: ampiezza della coda in cui un buco resta PROVVISORIO (ritardo di pull, non perdita).
# EN: width of the tail in which a gap stays PROVISIONAL (pull lag, not data loss).
PROVISIONAL_H = 6


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


def analyze(ts: pd.DatetimeIndex, days: int, provisional_h: int = PROVISIONAL_H) -> dict:
    # IT: separata da main() per essere testabile su serie sintetiche: il difetto che
    #     ha prodotto il falso positivo stava nella costruzione dello span, non nell'I/O.
    # EN: split out of main() to be testable on synthetic series: the defect behind the
    #     false positive lived in the span construction, not in the I/O.
    per_h = pd.Series(1, index=ts).resample("1h").sum()
    span = pd.date_range(ts[0].floor("h"), ts[-1].floor("h"), freq="1h")
    # IT: l'ultima ora dello span CONTIENE l'ultimo tick -> e' in corso, mai completa.
    # EN: the span's last hour CONTAINS the last tick -> in progress, never complete.
    span, in_progress = span[:-1], span[-1]
    if len(span) == 0:
        return {"empty": True, "in_progress": in_progress}

    ok = per_h.reindex(span, fill_value=0).values >= MIN_SNAP_PER_HOUR
    runs = runs_of_true(ok)

    # IT: il run CORRENTE (quello che termina all'ultima ora consolidata) e' l'unico che
    #     puo' ancora crescere: e' la quantita' da proteggere, non il massimo storico.
    # EN: the CURRENT run (ending at the last consolidated hour) is the only one that can
    #     still grow: it is the quantity to protect, not the historical maximum.
    cur = 0
    for v in ok[::-1]:
        if not v:
            break
        cur += 1

    def windows(rs):
        return int(sum(max(r - (T_WIN + H) + 1, 0) for r in rs))

    idx = np.where(~ok & (span >= span[-1] - pd.Timedelta(days=days)))[0]
    blocks: list[tuple[int, int]] = []
    for i in idx:
        if blocks and i == blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], i)
        else:
            blocks.append((int(i), int(i)))
    # IT: un buco che tocca la coda recente puo' essere solo ritardo di pull -> provvisorio.
    # EN: a gap touching the recent tail may be mere pull lag -> provisional.
    tail_start = span[-1] - pd.Timedelta(hours=provisional_h - 1)
    firm = [g for g in blocks if span[g[1]] < tail_start]
    prov = [g for g in blocks if span[g[1]] >= tail_start]

    return {
        "empty": False, "span": span, "in_progress": in_progress, "ok": ok,
        "coverage": float(ok.mean()), "cur": cur, "max_run": max(runs) if runs else 0,
        "windows": windows(runs), "n_eff": windows(runs) / H,
        "firm": firm, "prov": prov,
    }


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
    ap.add_argument("--provisional-hours", type=int, default=PROVISIONAL_H,
                    help="coda in cui un buco resta PROVVISORIO (ritardo di pull) / "
                         "tail in which a gap stays PROVISIONAL (pull lag)")
    args = ap.parse_args()

    files = sorted(glob.glob(L2_GLOB))
    if not files:
        print("[l2] nessun file order-book / no order-book file")
        return 0

    ts = pd.concat([pd.read_parquet(f, columns=["timestamp"]) for f in files])["timestamp"]
    ts = pd.DatetimeIndex(pd.to_datetime(ts, utc=True)).sort_values()
    r = analyze(ts, args.days, args.provisional_hours)
    if r["empty"]:
        print("[l2] meno di un'ora consolidata / less than one consolidated hour")
        return 0
    span, gaps, prov = r["span"], r["firm"], r["prov"]

    print(f"[l2] copertura/coverage {r['coverage']:.1%} su {len(span)} ore consolidate/"
          f"consolidated · run corrente/current run {r['cur']}h · run max {r['max_run']}h "
          f"(ora in corso {r['in_progress']:%H:%M} UTC esclusa/in-progress hour excluded)")
    print(f"[l2] finestre T={T_WIN}/h={H}: {r['windows']} · n_eff {r['n_eff']:.1f} "
          f"(gate vol: 216) · costo di 1h di buco: {T_WIN + H - 1} finestre "
          f"= {(T_WIN + H - 1) / H / (24 / H):.1f} giorni/days")

    # ── buchi recenti · recent gaps ─────────────────────────────────────────
    if prov:
        # IT: NON e' un allarme: e' una coda non ancora consolidata. Si giudica al pull
        #     successivo. Stamparlo come perdita e' l'errore del 2026-08-02.
        # EN: NOT an alarm: an unconsolidated tail. Judge it at the next pull. Printing it
        #     as a loss is the 2026-08-02 mistake.
        tot_p = sum(b - a + 1 for a, b in prov)
        print(f"[l2] ~ {len(prov)} buco/i PROVVISORIO/I nelle ultime {args.provisional_hours}h "
              f"({tot_p}h) — ritardo di pull finche' non sopravvive a un secondo pull / "
              f"pull lag until it survives a second pull:")
        for a, b in prov:
            print(f"[l2]    {span[a]:%Y-%m-%d %H:%M} UTC · {b - a + 1}h (provvisorio/provisional)")
    if gaps:
        tot = sum(b - a + 1 for a, b in gaps)
        print(f"[l2] ⚠ {len(gaps)} BUCHI CONSOLIDATI negli ultimi {args.days}g ({tot}h perse) — "
              f"ogni buco spezza il run / each consolidated gap breaks the run:")
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
    elif not prov:
        print(f"[l2] nessun buco negli ultimi {args.days}g / no gap in the last {args.days}d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
