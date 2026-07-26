# IT: Recorder order-book L2 Binance (strada B1 della roadmap) — raccolta FORWARD
#     della microstruttura BTC come fonte di informazione NUOVA per un edge
#     direzionale a 1m. Razionale: le 104 feature OHLCV-derivate sono SATURE
#     (cross-arch corr ≈0.995, nessuna variante di modello sposta l'OOS — vedi
#     corpus di KILL in STATUS.md); l'unica fonte plausibile non ancora spremuta
#     è la microstruttura del book (depth, imbalance, microprice, OFI).
#     Come per la IV (poller 01c), lo storico L2 NON è gratis (Tardis lo vende),
#     quindi il dataset si costruisce da qui in avanti: questo è il collo di
#     bottiglia temporale di B1 — prima parte, prima si accumula.
#     Feed: REST pubblico /api/v3/depth (no auth, no servizi a pagamento), limit
#     1000 livelli/lato, weight 50/call → a 5s = 600 weight/min ≪ limite 1200/min.
#     Output append-only ATOMICO (tmp+os.replace, dedup su timestamp) in
#     data/orderbook/:
#       l2_features_YYYYMMDD.parquet — 1 riga/tick: feature derivate (segnale)
#                                      + top-25 livelli raw/lato (rete di sicurezza
#                                      per ri-derivare feature future senza
#                                      ri-raccogliere — lo schema NON è bloccato).
#     Modalità: --once (smoke), --seconds N (cadenza, default 5), --symbol, --levels.
# EN: Binance L2 order-book recorder (roadmap track B1) — FORWARD collection of BTC
#     microstructure as a genuinely NEW information source for 1m directional edge.
#     Rationale: the 104 OHLCV-derived features are SATURATED (cross-arch corr
#     ≈0.995, no model variant moves OOS — see KILL corpus in STATUS.md); the only
#     plausible un-squeezed source is book microstructure (depth, imbalance,
#     microprice, OFI). Like IV (poller 01c), L2 history is NOT free (Tardis sells
#     it), so the dataset is built from now on: this is B1's temporal bottleneck —
#     the sooner it starts, the sooner it accumulates.
#     Feed: public REST /api/v3/depth (no auth, no paid services), 1000 levels/side,
#     weight 50/call → at 5s = 600 weight/min ≪ the 1200/min limit.
#     Append-only ATOMIC output (tmp+os.replace, dedup on timestamp) under
#     data/orderbook/:
#       l2_features_YYYYMMDD.parquet — 1 row/tick: derived features (signal)
#                                      + top-25 raw levels/side (safety net to
#                                      re-derive future features without
#                                      re-collecting — the schema is NOT locked).
#     Modes: --once (smoke), --seconds N (cadence, default 5), --symbol, --levels.
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantsys.utils import setup_logging                      # noqa: E402
# IT: append atomico condiviso dei collector (estratto 2026-07-16, ex duplicato).
# EN: shared collector atomic append (extracted 2026-07-16, ex local duplicate).
from quantsys.utils.collect import append_parquet             # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.ob_recorder")

# IT: endpoint pubblico Binance spot (verificato no-auth da questo host, 2026-06-16).
# EN: Binance spot public endpoint (verified no-auth from this host, 2026-06-16).
BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"
OB_DIR = Path("data/orderbook")

# IT: quanti livelli raw persistere per lato (rete di sicurezza ri-derivazione);
#     il book completo a 1000 livelli × 5s sarebbe ~0,5 GB/giorno: il top-25 cattura
#     la zona economicamente attiva (>99% del flusso passa vicino al mid) a costo minimo.
# EN: how many raw levels to persist per side (re-derivation safety net); the full
#     1000-level book × 5s would be ~0.5 GB/day: top-25 captures the economically
#     active zone (>99% of flow sits near the mid) at minimal cost.
RAW_LEVELS = 25

# IT: bande in bps attorno al mid su cui sommare la depth cumulata — feature
#     robusta al numero di livelli (invariante al tick-size, confrontabile nel tempo).
# EN: bps bands around the mid for cumulative depth — a feature robust to level
#     count (tick-size invariant, comparable over time).
DEPTH_BANDS_BPS = (5.0, 10.0, 25.0, 50.0)

# IT: stato cross-tick per l'OFI di Cont et al. (richiede lo snapshot precedente).
# EN: cross-tick state for Cont et al. OFI (needs the previous snapshot).
_PREV = {"bid_px": None, "bid_qty": None, "ask_px": None, "ask_qty": None}


def fetch_depth(symbol: str, levels: int) -> dict:
    # IT: uno snapshot REST del book (best→worst, già ordinato da Binance).
    # EN: one REST snapshot of the book (best→worst, already sorted by Binance).
    r = requests.get(
        BINANCE_DEPTH_URL,
        params={"symbol": symbol, "limit": levels},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _best_level_ofi(bid_px, bid_qty, ask_px, ask_qty) -> float:
    # IT: OFI sul best level (Cont-Kukanov-Stoikov) tra due snapshot consecutivi:
    #     misura la pressione netta di flusso ordini, predittore microstrutturale
    #     classico del prossimo movimento del mid. Approssimazione snapshot→snapshot
    #     (l'OFI event-level pieno richiede il WS diff stream: v2).
    # EN: best-level OFI (Cont-Kukanov-Stoikov) between two consecutive snapshots:
    #     net order-flow pressure, a classic microstructure predictor of the next
    #     mid move. Snapshot-to-snapshot approximation (full event-level OFI needs
    #     the WS diff stream: v2).
    pb, qb = _PREV["bid_px"], _PREV["bid_qty"]
    pa, qa = _PREV["ask_px"], _PREV["ask_qty"]
    if pb is None:
        return np.nan
    # IT: lato bid — un bid che sale o tiene il prezzo aggiunge pressione d'acquisto.
    # EN: bid side — a bid that rises or holds price adds buy pressure.
    if bid_px > pb:
        e_bid = bid_qty
    elif bid_px == pb:
        e_bid = bid_qty - qb
    else:
        e_bid = -qb
    # IT: lato ask — un ask che scende o tiene aggiunge pressione di vendita.
    # EN: ask side — an ask that falls or holds adds sell pressure.
    if ask_px < pa:
        e_ask = ask_qty
    elif ask_px == pa:
        e_ask = ask_qty - qa
    else:
        e_ask = -qa
    return float(e_bid - e_ask)


def compute_features(depth: dict, symbol: str) -> pd.DataFrame:
    # IT: deriva le feature microstrutturali da uno snapshot. Tutto è point-in-time
    #     e causale (solo il book corrente + il precedente per l'OFI).
    # EN: derive microstructure features from one snapshot. Everything is
    #     point-in-time and causal (only the current book + the previous for OFI).
    ts = pd.Timestamp.now(tz="UTC").floor("s")
    bids = np.asarray(depth["bids"], dtype=float)   # [[price, qty], ...] best→worst
    asks = np.asarray(depth["asks"], dtype=float)
    if bids.size == 0 or asks.size == 0:
        raise RuntimeError("book vuoto / empty book")

    bid_px, bid_qty = bids[:, 0], bids[:, 1]
    ask_px, ask_qty = asks[:, 0], asks[:, 1]
    best_bid, best_ask = bid_px[0], ask_px[0]
    q_b1, q_a1 = bid_qty[0], ask_qty[0]

    mid = 0.5 * (best_bid + best_ask)
    spread = best_ask - best_bid

    # IT: microprice (Gatheral-Stoikov): mid pesato dalle size opposte — stima
    #     non distorta del "vero" prezzo, sbilanciata verso il lato con più liquidità.
    # EN: microprice (Gatheral-Stoikov): mid weighted by opposite sizes — an
    #     unbiased estimate of the "true" price, tilted toward the deeper side.
    microprice = (best_bid * q_a1 + best_ask * q_b1) / (q_b1 + q_a1)

    feat = {
        "timestamp": ts,
        "symbol": symbol,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread": spread,
        "spread_bps": spread / mid * 1e4,
        "microprice": microprice,
        "microprice_tilt_bps": (microprice - mid) / mid * 1e4,
    }

    # IT: imbalance cumulato a profondità crescenti — segno classico di pressione.
    # EN: cumulative imbalance at increasing depths — a classic pressure sign.
    for n in (1, 5, 10, 20):
        sb = bid_qty[:n].sum()
        sa = ask_qty[:n].sum()
        feat[f"imbalance_L{n}"] = (sb - sa) / (sb + sa) if (sb + sa) > 0 else 0.0

    # IT: depth cumulata entro bande bps dal mid — invariante al numero di livelli.
    # EN: cumulative depth within bps bands of the mid — invariant to level count.
    for bps in DEPTH_BANDS_BPS:
        lo = mid * (1 - bps / 1e4)
        hi = mid * (1 + bps / 1e4)
        db = bid_qty[bid_px >= lo].sum()
        da = ask_qty[ask_px <= hi].sum()
        tag = f"{int(bps)}bps"
        feat[f"depth_bid_{tag}"] = db
        feat[f"depth_ask_{tag}"] = da
        feat[f"depth_imb_{tag}"] = (db - da) / (db + da) if (db + da) > 0 else 0.0

    # IT: liquidità totale nello snapshot (controllo di regime/spessore del book).
    # EN: total snapshot liquidity (book-thickness / regime control).
    feat["total_bid_qty"] = bid_qty.sum()
    feat["total_ask_qty"] = ask_qty.sum()
    feat["n_bid_levels"] = len(bid_px)
    feat["n_ask_levels"] = len(ask_px)

    # IT: OFI best-level vs snapshot precedente (NaN al primo tick).
    # EN: best-level OFI vs previous snapshot (NaN on first tick).
    feat["ofi_best"] = _best_level_ofi(best_bid, q_b1, best_ask, q_a1)
    _PREV.update(bid_px=best_bid, bid_qty=q_b1, ask_px=best_ask, ask_qty=q_a1)

    # IT: top-N livelli raw/lato come list-column — rete di sicurezza: qualunque
    #     feature futura (microprice multi-livello, slope, ecc.) è ri-derivabile
    #     senza ri-raccogliere il dataset.
    # EN: top-N raw levels/side as list-columns — safety net: any future feature
    #     (multi-level microprice, slope, etc.) is re-derivable without re-collecting.
    feat["raw_bid_px"] = bid_px[:RAW_LEVELS].tolist()
    feat["raw_bid_qty"] = bid_qty[:RAW_LEVELS].tolist()
    feat["raw_ask_px"] = ask_px[:RAW_LEVELS].tolist()
    feat["raw_ask_qty"] = ask_qty[:RAW_LEVELS].tolist()

    feat["last_update_id"] = depth.get("lastUpdateId")
    return pd.DataFrame([feat])


def poll_once(symbol: str, levels: int) -> dict:
    # IT: un tick completo: snapshot REST → feature → append parquet del giorno.
    # EN: one full tick: REST snapshot → features → append to the day's parquet.
    depth = fetch_depth(symbol, levels)
    rows = compute_features(depth, symbol)
    ts = rows["timestamp"].iloc[0]
    out_path = OB_DIR / f"l2_features_{ts:%Y%m%d}.parquet"
    n = append_parquet(out_path, rows, dedup_cols=["timestamp"])
    r0 = rows.iloc[0]
    return {
        "ts": ts, "path": out_path, "n_rows": n,
        "mid": r0["mid"], "spread_bps": r0["spread_bps"],
        "imb_L5": r0["imbalance_L5"], "ofi_best": r0["ofi_best"],
    }


def main():
    # IT: reconfigure UTF-8 (bug cp1252 ricorrente su console Windows — checklist nuovo script).
    # EN: reconfigure UTF-8 (recurring cp1252 bug on Windows console — new-script checklist).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Recorder order-book L2 Binance (B1)")
    ap.add_argument("--once", action="store_true", help="un solo tick (smoke) / single tick (smoke)")
    ap.add_argument("--seconds", type=float, default=5.0, help="cadenza polling / polling cadence (s)")
    ap.add_argument("--symbol", default="BTCUSDT", help="simbolo Binance spot / Binance spot symbol")
    ap.add_argument("--levels", type=int, default=1000, help="profondità REST / REST depth (max 5000)")
    args = ap.parse_args()

    OB_DIR.mkdir(parents=True, exist_ok=True)
    log.info(
        "ob-recorder avviato/started — symbol=%s levels=%d cadenza/cadence=%.1fs "
        "raw_levels=%d → data/orderbook/",
        args.symbol, args.levels, args.seconds, RAW_LEVELS,
    )

    if args.once:
        s = poll_once(args.symbol, args.levels)
        log.info(
            "tick %s: mid=%.2f spread=%.2fbps imb_L5=%+.3f ofi=%s | %d righe/rows → %s",
            s["ts"], s["mid"], s["spread_bps"], s["imb_L5"],
            "nan" if np.isnan(s["ofi_best"]) else f"{s['ofi_best']:+.3f}",
            s["n_rows"], s["path"].name,
        )
        return

    # IT: loop persistente — NON è un servizio: dopo un riavvio va rilanciato
    #     (Start-Process, vedi AVVIO.md). Errori di rete loggati e non fatali.
    # EN: persistent loop — NOT a service: relaunch after a reboot (Start-Process,
    #     see AVVIO.md). Network errors logged and non-fatal.
    n_ok, n_err = 0, 0
    while True:
        t0 = time.time()
        try:
            s = poll_once(args.symbol, args.levels)
            n_ok += 1
            if n_ok % 12 == 1:   # IT: log ogni ~1 min a 5s / log ~every 1 min at 5s
                log.info(
                    "tick %s: mid=%.2f spread=%.2fbps imb_L5=%+.3f ofi=%s | rows=%d (ok=%d err=%d)",
                    s["ts"], s["mid"], s["spread_bps"], s["imb_L5"],
                    "nan" if np.isnan(s["ofi_best"]) else f"{s['ofi_best']:+.3f}",
                    s["n_rows"], n_ok, n_err,
                )
        except Exception as e:   # noqa: BLE001
            n_err += 1
            log.warning("tick fallito/failed (err=%d): %s", n_err, e)
        # IT: dormi il residuo per tenere la cadenza stabile nonostante la latenza REST.
        # EN: sleep the remainder to keep cadence stable despite REST latency.
        time.sleep(max(0.0, args.seconds - (time.time() - t0)))


if __name__ == "__main__":
    main()
