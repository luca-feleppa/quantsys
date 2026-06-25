"""Fase 1 — Data Ingestion: Binance REST API + WebSocket live feed.

Download storico: usa python-binance (Client.get_historical_klines) con
paginazione automatica e rispetto del rate limit weight-based di Binance.
Fallback a requests puro se python-binance non è installato.
"""
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger("quantsys.data")

BINANCE_REST       = "https://api.binance.com/api/v3"
BINANCE_FAPI_REST  = "https://fapi.binance.com/fapi/v1"
BINANCE_WS         = "wss://stream.binance.com:9443/ws"
MAX_PER_REQ        = 1000

# IT: Mappa intervallo → secondi per candela (usata per gap detection)
# EN: Interval → seconds-per-candle mapping (used for gap detection)
_INTERVAL_SECS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


# ─── Helpers ──────────────────────────────────────────────────────────────────

# IT: Converte le klines grezze Binance in DataFrame OHLCV tipizzato e ordinato.
# EN: Converts raw Binance klines into a typed, sorted OHLCV DataFrame.
def _raw_to_df(all_klines: list) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_vol","trades","taker_buy_vol","taker_buy_quote_vol","_"]
    df = pd.DataFrame(all_klines, columns=cols).drop(columns=["_"])
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume","quote_vol","taker_buy_vol","taker_buy_quote_vol"]:
        df[c] = df[c].astype(float)
    df["trades"] = df["trades"].astype(int)
    return df.sort_values("open_time").reset_index(drop=True)


# IT: Rimuove candele corrotte (OHLC invalidi) e logga gap temporali anomali.
# EN: Drops corrupted candles (invalid OHLC) and logs anomalous time gaps.
def _sanitize(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Rimuove candele corrotte e logga gap temporali anomali."""
    n_before = len(df)
    # IT: reference = open (≈ close della candela precedente): robusto a un close corrotto.
    #     Soglia rilassata 10×→50× (audit #23, 2026-06-03): un wick 10×–50× può essere un
    #     flash-crash legittimo (es. May-2021/COVID-2020), non necessariamente una candela rotta.
    #     50× resta abbastanza stretto da catturare corruzioni reali (decimal shift, bad tick).
    # EN: reference = open (≈ previous candle's close): robust to a corrupted close.
    #     Threshold relaxed 10×→50× (audit #23, 2026-06-03): a 10×–50× wick can be a
    #     legitimate flash-crash (e.g. May-2021/COVID-2020), not necessarily a broken candle.
    #     50× is still tight enough to catch real corruption (decimal shift, bad tick).
    ref_ok = df["open"] > 0
    bad = (
        (df["high"]   < df["low"])
        | (df["close"] < 0) | (df["open"] < 0) | (df["volume"] < 0)
        | (ref_ok & (df["high"] > df["open"] * 50))
        | (ref_ok & (df["low"]  < df["open"] * 0.02))
        | (df["close"] == 0) | (df["open"] == 0)
        | df["close"].isna()
    )
    if bad.any():
        n_bad = bad.sum()
        log.warning(f"Rimosse {n_bad} candele corrotte su {n_before} ({n_bad/n_before:.1%})")
        df = df[~bad].reset_index(drop=True)

    if len(df) > 1:
        expected_s = _INTERVAL_SECS.get(interval, 60)
        diffs = df["open_time"].diff().dt.total_seconds().dropna()
        big_gaps = diffs[diffs > expected_s * 5]
        if not big_gaps.empty:
            log.warning(
                f"{len(big_gaps)} gap temporali > 5× intervallo atteso "
                f"(max={big_gaps.max():.0f}s). Possibili halt o interruzioni feed."
            )

    log.info(f"Caricate {len(df):,} candele  [{df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]}]")
    return df


# ─── Download via python-binance ──────────────────────────────────────────────
# IT: Path principale: paginazione automatica + rate-limit weight-based.
# EN: Primary path: automatic pagination + weight-based rate limiting.

def _fetch_via_binance_lib(symbol: str, interval: str, start_time: str) -> list:
    """
    Usa python-binance con paginazione manuale per mostrare il progresso.
    Ogni 100 richieste (~100k candele) logga percentuale e candele scaricate.
    """
    from binance.client import Client

    client   = Client()
    start_ms = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000)
    now_ms   = int(time.time() * 1000)

    log.info(f"[python-binance] Download {symbol} {interval} da {start_time} ...")
    log.info(f"  Stima: ~{(now_ms - start_ms) // (_INTERVAL_SECS.get(interval, 60) * 1000):,} candele")

    all_klines    = []
    current_start = start_ms
    n_req         = 0

    while current_start < now_ms:
        klines = client.get_klines(
            symbol    = symbol,
            interval  = interval,
            startTime = current_start,
            limit     = MAX_PER_REQ,
        )
        if not klines:
            break
        all_klines.extend(klines)
        current_start = klines[-1][0] + 1
        n_req += 1

        if n_req % 100 == 0:
            pct = min((current_start - start_ms) / (now_ms - start_ms) * 100, 99.9)
            log.info(
                f"  Progresso: {len(all_klines):,} candele  ({pct:.1f}%)  "
                f"— fino a {pd.to_datetime(klines[-1][0], unit='ms', utc=True).date()}"
            )

    log.info(f"[python-binance] Completato: {len(all_klines):,} candele scaricate")
    return all_klines


# IT: Fallback via requests puro (no python-binance), paginazione manuale.
# EN: Pure-requests fallback (no python-binance), manual pagination.
def _fetch_via_requests(symbol: str, interval: str,
                        start_time: str = None, limit: int = 50000) -> list:
    """
    Fallback: scarica tramite requests con paginazione manuale.
    Usato se python-binance non è installato o come backup.
    """
    import requests as _req

    if start_time is not None:
        start_ms = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000)
        all_klines = []
        now_ms = int(time.time() * 1000)
        current_start = start_ms

        log.info(f"[requests] Download storico {symbol} {interval} da {start_time} ...")
        n_req = 0
        while current_start < now_ms:
            params = {
                "symbol": symbol, "interval": interval,
                "startTime": current_start, "limit": MAX_PER_REQ,
            }
            resp = _req.get(f"{BINANCE_REST}/klines", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_klines.extend(data)
            current_start = data[-1][0] + 1
            n_req += 1
            if n_req % 50 == 0:
                pct = (current_start - start_ms) / (now_ms - start_ms) * 100
                log.info(f"  Download: {len(all_klines):,} candele ({pct:.0f}%)")
            time.sleep(0.15)
        return all_klines

    else:
        all_klines, remaining, end_time = [], limit, None
        log.info(f"[requests] Scaricamento {limit} candele {symbol} {interval} ...")
        while remaining > 0:
            batch  = min(remaining, MAX_PER_REQ)
            params = {"symbol": symbol, "interval": interval, "limit": batch}
            if end_time:
                params["endTime"] = end_time
            resp = _req.get(f"{BINANCE_REST}/klines", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data: break
            all_klines = data + all_klines
            end_time   = data[0][0] - 1
            remaining -= len(data)
            time.sleep(0.35)
        return all_klines


# ─── API pubblica ─────────────────────────────────────────────────────────────

# IT: Entry point download OHLCV storico (sceglie lib o requests, poi sanitize).
# EN: Historical OHLCV download entry point (picks lib or requests, then sanitize).
def fetch_klines(symbol: str, interval: str, limit: int,
                 start_time: str = None) -> pd.DataFrame:
    """
    Scarica candele OHLCV storiche da Binance.

    Con start_time: usa python-binance se disponibile (più veloce e robusto),
    altrimenti requests con paginazione manuale.
    Senza start_time: scarica le ultime `limit` candele via requests.

    Args:
        symbol:     es. "BTCUSDT"
        interval:   es. "1m"
        limit:      candele recenti da scaricare (ignorato se start_time fornito)
        start_time: data di inizio ISO 8601, es. "2021-01-01"
                    "2021-01-01" → ~2.1M candele, ciclo completo ATH+bear+recovery
                    "2022-01-01" → ~1.3M candele, bear+recovery (più veloce)
    """
    if start_time is not None:
        # IT: Prova prima python-binance (paginazione + rate-limit automatici).
        # EN: Try python-binance first (auto pagination + rate limiting).
        try:
            raw = _fetch_via_binance_lib(symbol, interval, start_time)
        except ImportError:
            log.warning("python-binance non installato — uso requests (più lento). "
                        "pip install python-binance per velocizzare il download.")
            raw = _fetch_via_requests(symbol, interval, start_time=start_time)
        except Exception as e:
            log.warning(f"python-binance fallito ({e}), fallback a requests ...")
            raw = _fetch_via_requests(symbol, interval, start_time=start_time)
    else:
        raw = _fetch_via_requests(symbol, interval, limit=limit)

    df = _raw_to_df(raw)
    return _sanitize(df, interval)


# IT: Aggiornamento incrementale: scarica solo il delta dall'ultimo timestamp.
# EN: Incremental update: downloads only the delta since the last timestamp.
def fetch_klines_incremental(raw_path: str, symbol: str, interval: str) -> pd.DataFrame:
    """
    Aggiornamento incrementale: legge l'ultimo timestamp da raw_path,
    scarica solo le candele mancanti (delta), le appende e restituisce
    il DataFrame completo aggiornato.

    Usato da scripts/01_update_data.py per evitare di ri-scaricare
    l'intero storico ad ogni aggiornamento.

    Args:
        raw_path: percorso a data/raw_candles.parquet
        symbol:   es. "BTCUSDT"
        interval: es. "1m"

    Returns:
        DataFrame OHLCV completo (storico + delta), già sanitizzato.
        La candela corrente (non ancora chiusa) viene scartata.
    """
    df_existing = pd.read_parquet(raw_path)
    last_ts = df_existing["open_time"].max()

    # IT: Passo temporale derivato dall'intervallo (interval-agnostic: "1m"→60s, "1h"→3600s).
    #     A 1m il comportamento è identico al legacy (Timedelta("1min")).
    # EN: Time step derived from the interval (interval-agnostic: "1m"→60s, "1h"→3600s).
    #     At 1m the behavior is identical to legacy (Timedelta("1min")).
    step = pd.Timedelta(seconds=_INTERVAL_SECS.get(interval, 60))

    # IT: Riparti dalla candela successiva all'ultima disponibile.
    # EN: Resume from the candle after the last available one.
    next_ts = last_ts + step
    # IT: Scarta la candela corrente (non ancora chiusa → potrebbe essere parziale).
    # EN: Drop the in-progress candle (not yet closed → could be partial).
    now_floored = pd.Timestamp.utcnow().floor(step) - step

    if next_ts >= now_floored:
        log.info(f"Dataset già aggiornato fino a {last_ts} — nessun delta da scaricare.")
        return df_existing

    start_str = next_ts.strftime("%Y-%m-%d %H:%M:%S")
    delta_candles = (now_floored - next_ts).total_seconds() / _INTERVAL_SECS.get(interval, 60)
    log.info(
        f"Aggiornamento incrementale: {last_ts.date()} → {now_floored.date()}  "
        f"(~{delta_candles:,.0f} nuove candele)"
    )

    df_delta = fetch_klines(symbol, interval, limit=0, start_time=start_str)

    # IT: Rimuove la candela corrente se ancora aperta.
    # EN: Drop the current candle if still open.
    df_delta = df_delta[df_delta["open_time"] <= now_floored]

    df_full = (
        pd.concat([df_existing, df_delta], ignore_index=True)
        .drop_duplicates(subset="open_time")
        .sort_values("open_time")
        .reset_index(drop=True)
    )

    n_new = len(df_full) - len(df_existing)
    log.info(
        f"Merge completato: {len(df_existing):,} + {n_new:,} nuove = "
        f"{len(df_full):,} candele totali  "
        f"[{df_full['open_time'].iloc[0].date()} → {df_full['open_time'].iloc[-1].date()}]"
    )
    return df_full


# IT: Scarica e persiste il funding rate perpetui (cadenza 8h, delta-aware).
# EN: Downloads and persists perpetual funding rate (8h cadence, delta-aware).
def fetch_funding_rate(symbol: str, start_time: str, output_dir: str) -> pd.DataFrame:
    """
    Scarica il funding rate storico dei futures perpetui Binance.

    Il funding rate è pubblicato ogni 8 ore. Questo endpoint restituisce
    al massimo 1000 osservazioni per request, quindi viene paginato
    automaticamente dal start_time fino ad oggi.

    Se `output_dir/funding_rate.parquet` esiste già, carica da disco e
    scarica solo il delta dall'ultimo timestamp presente.

    Args:
        symbol:     es. "BTCUSDT"
        start_time: data ISO 8601 di inizio storico, es. "2021-01-01"
        output_dir: cartella dove salvare funding_rate.parquet

    Returns:
        DataFrame con colonne: open_time (datetime UTC), funding_rate (float)
    """
    import requests as _req
    from quantsys.utils.atomic_save import atomic_save_parquet

    out_path = Path(output_dir) / "funding_rate.parquet"

    # IT: Punto di partenza = ultimo timestamp noto, o start_time richiesto.
    # EN: Starting point = last known timestamp, or requested start_time.
    df_existing = None
    if out_path.exists():
        df_existing = pd.read_parquet(out_path)
        last_ts     = df_existing["open_time"].max()
        # IT: Prossima osservazione attesa 8h dopo l'ultima (cadenza funding).
        # EN: Next observation expected 8h after the last (funding cadence).
        next_ts     = last_ts + pd.Timedelta("8h")
        fetch_from_ms = int(next_ts.timestamp() * 1000)
        log.info(
            f"Funding rate: caricato storico da disco ({len(df_existing):,} obs). "
            f"Delta da {next_ts.date()} ..."
        )
    else:
        fetch_from_ms = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000)
        log.info(f"Funding rate: download completo da {start_time} ...")

    now_ms    = int(time.time() * 1000)
    all_rows  = []
    n_req     = 0

    while fetch_from_ms < now_ms:
        params = {
            "symbol":    symbol,
            "limit":     MAX_PER_REQ,
            "startTime": fetch_from_ms,
        }
        resp = _req.get(f"{BINANCE_FAPI_REST}/fundingRate", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        fetch_from_ms = data[-1]["fundingTime"] + 1
        n_req += 1
        if n_req % 20 == 0:
            ts = pd.to_datetime(data[-1]["fundingTime"], unit="ms", utc=True)
            log.info(f"  Funding rate: {len(all_rows):,} obs scaricate — fino a {ts.date()}")
        time.sleep(0.5)

    if not all_rows:
        if df_existing is not None:
            log.info("Funding rate: nessun delta nuovo, dataset già aggiornato.")
            return df_existing
        log.warning(f"Funding rate: nessun dato ricevuto per {symbol} da {start_time}.")
        return pd.DataFrame(columns=["open_time", "funding_rate"])

    df_new = pd.DataFrame({
        "open_time":    pd.to_datetime([r["fundingTime"] for r in all_rows], unit="ms", utc=True),
        "funding_rate": [float(r["fundingRate"]) for r in all_rows],
    })

    if df_existing is not None:
        df_full = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .drop_duplicates(subset="open_time")
            .sort_values("open_time")
            .reset_index(drop=True)
        )
    else:
        df_full = df_new.sort_values("open_time").reset_index(drop=True)

    atomic_save_parquet(df_full, out_path, index=False)
    log.info(
        f"Funding rate: {len(df_full):,} obs totali  "
        f"[{df_full['open_time'].iloc[0].date()} → {df_full['open_time'].iloc[-1].date()}]  "
        f"→ {out_path}"
    )
    return df_full


# IT: Snapshot statistiche ticker 24h (prezzo, volume, variazione %).
# EN: 24h ticker statistics snapshot (price, volume, % change).
def fetch_ticker_24hr(symbol: str) -> dict:
    import requests as _req
    r = _req.get(f"{BINANCE_REST}/ticker/24hr", params={"symbol": symbol}, timeout=5)
    r.raise_for_status()
    return r.json()


# ─── WebSocket live ───────────────────────────────────────────────────────────
# IT: Streaming candele live con reconnect automatico (backoff 3s).
# EN: Live candle streaming with automatic reconnect (3s backoff).

async def stream_klines(
    symbol:   str,
    interval: str,
    callback: Callable[[dict], None],
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Stream live di candele via WebSocket Binance.
    Chiama `callback(kline_dict)` ad ogni aggiornamento.
    """
    import websockets

    url = f"{BINANCE_WS}/{symbol.lower()}@kline_{interval}"
    log.info(f"WS connesso: {url}")

    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                async for msg in ws:
                    data = json.loads(msg)
                    if "k" in data:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data["k"])
                        else:
                            callback(data["k"])
                    if stop_event and stop_event.is_set():
                        return
        except Exception as e:
            log.warning(f"WS disconnesso ({e}), riconnessione in 3s ...")
            await asyncio.sleep(3)
