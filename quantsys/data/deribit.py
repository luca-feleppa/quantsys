# IT: Helper Deribit condivisi (01c poller IV, 01e recorder trades; dal C2 2ter
#     2026-07-18 anche delivery-cache unica per 04c/replay/short_vol_arm) —
#     estratti 2026-07-16 dai duplicati negli script. SOLO endpoint PUBBLICI
#     no-auth: le credenziali testnet (secrets.yaml) NON passano di qui.
# EN: Shared Deribit helpers (01c IV poller, 01e trades recorder; since the C2
#     2ter refactor 2026-07-18 also the single delivery cache for
#     04c/replay/short_vol_arm) — extracted 2026-07-16 from the script
#     duplicates. PUBLIC no-auth endpoints ONLY: the testnet credentials
#     (secrets.yaml) never come here.
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger("quantsys.data.deribit")

# IT: endpoint pubblico production (verificato 2026-06-11; la testnet non va
#     usata per dati di mercato: trade paper, storia non ritenuta).
# EN: production public endpoint (verified 2026-06-11; testnet must not be
#     used for market data: paper trades, no history retention).
DERIBIT_BASE = "https://www.deribit.com/api/v2"

# IT: nome strumento: BTC-13JUN26-105000-C → expiry 08:00 UTC del giorno.
#     Prefisso [A-Z]+ (non solo BTC): il parser è valuta-agnostico; perpetual
#     e future (senza -C/-P) NON matchano per costruzione.
# EN: instrument name: BTC-13JUN26-105000-C → expiry at 08:00 UTC that day.
#     [A-Z]+ prefix (not BTC-only): currency-agnostic parser; perpetuals and
#     futures (no -C/-P) do NOT match by construction.
_INSTR_RE = re.compile(r"^[A-Z]+-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:d\d+)?)-([CP])$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def parse_instrument(name: str):
    # IT: estrae (expiry UTC, strike, tipo C/P) dal nome; None se non-standard.
    #     Strike decimali tipo "3d5" non esistono su BTC, ma il parse non crasha.
    # EN: extracts (UTC expiry, strike, C/P type) from the name; None when
    #     non-standard. Decimal strikes like "3d5" don't occur on BTC, but
    #     parsing won't crash.
    m = _INSTR_RE.match(name)
    if not m:
        return None
    day, mon, yy, strike_s, opt = m.groups()
    expiry = datetime(2000 + int(yy), _MONTHS[mon], int(day), 8, 0,
                      tzinfo=timezone.utc)
    return expiry, float(strike_s.replace("d", ".")), opt


def deribit_public_get(path: str, params: dict, timeout: int = 15) -> dict:
    # IT: GET pubblica con error-raise; i transient (rete, 5xx) li gestisce il
    #     loop chiamante (pattern 01c/01e: mai uccidere il collector).
    # EN: public GET with error-raise; transients (network, 5xx) are handled by
    #     the calling loop (01c/01e pattern: never kill the collector).
    r = requests.get(f"{DERIBIT_BASE}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "result" not in payload:
        raise RuntimeError(f"Deribit risposta inattesa / unexpected response: {payload}")
    return payload["result"]


# IT: ── delivery price (C2 2ter: cache unica) ──────────────────────────────
#     Chiave canonica = DDMMMYY del giorno di settlement (08:00 UTC), la stessa
#     di maybe_settle in 04b. La cache è per-consumer (path esplicito): 04c
#     legge il delivery TESTNET (venue dei trade paper), replay/short_vol_arm
#     il production — venue diverse NON vanno mai mischiate nello stesso file.
# EN: ── delivery price (C2 2ter: single cache) ─────────────────────────────
#     Canonical key = DDMMMYY of the settlement day (08:00 UTC), same as
#     maybe_settle in 04b. The cache is per-consumer (explicit path): 04c reads
#     the TESTNET delivery (paper-trade venue), replay/short_vol_arm the
#     production one — different venues must NEVER share a cache file.
def delivery_key(expiry) -> str:
    # IT: chiave cache canonica dal timestamp/datetime di expiry.
    # EN: canonical cache key from the expiry timestamp/datetime.
    return pd.Timestamp(expiry).strftime("%d%b%y").upper()


def fetch_delivery_prices(base_url: str = DERIBIT_BASE, count: int = 10,
                          offset: int = 0, timeout: int = 15) -> dict:
    # IT: una pagina di public/get_delivery_prices → {DDMMMYY: prezzo}.
    #     Error-raise: la fail-softness la decide il chiamante.
    # EN: one page of public/get_delivery_prices → {DDMMMYY: price}.
    #     Error-raise: fail-softness is the caller's decision.
    r = requests.get(f"{base_url.rstrip('/')}/public/get_delivery_prices",
                     params={"index_name": "btc_usd", "count": count,
                             "offset": offset}, timeout=timeout)
    r.raise_for_status()
    data = r.json().get("result", {}).get("data", [])
    return {delivery_key(rec["date"]): float(rec["delivery_price"]) for rec in data}


def delivery_price_cached(expiry, cache_path, base_url: str = DERIBIT_BASE,
                          max_offset: int = 100, page_count: int = 10):
    # IT: lookup con cache JSON su disco + paging count/offset (l'endpoint torna
    #     solo gli ultimi N giorni per pagina). Fail-soft sulla rete: warning e
    #     cache-only (None se assente = non ancora pubblicato). Write atomica
    #     (.tmp + os.replace — safety net repo).
    # EN: lookup with on-disk JSON cache + count/offset paging (the endpoint
    #     returns only the last N days per page). Network fail-soft: warning and
    #     cache-only (None when absent = not yet published). Atomic write
    #     (.tmp + os.replace — repo safety net).
    key = delivery_key(expiry)
    cache_path = Path(cache_path)
    cache = (json.loads(cache_path.read_text(encoding="utf-8"))
             if cache_path.exists() else {})
    if key in cache:
        return float(cache[key])
    try:
        for offset in range(0, max_offset, page_count):
            page = fetch_delivery_prices(base_url, count=page_count, offset=offset)
            if not page:
                break
            cache.update(page)
            if key in cache:
                break
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, cache_path)
    except Exception as e:
        log.warning(f"fetch delivery prices fallito/failed: {type(e).__name__}: {e} "
                    f"— uso solo la cache / cache only")
    return float(cache[key]) if key in cache else None
