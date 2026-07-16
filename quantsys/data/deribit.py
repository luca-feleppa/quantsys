# IT: Helper Deribit condivisi (01c poller IV, 01e recorder trades) — estratti
#     2026-07-16 dai duplicati negli script. SOLO endpoint production PUBBLICI
#     no-auth: le credenziali testnet (secrets.yaml) NON passano di qui.
# EN: Shared Deribit helpers (01c IV poller, 01e trades recorder) — extracted
#     2026-07-16 from the script duplicates. Production PUBLIC no-auth
#     endpoints ONLY: the testnet credentials (secrets.yaml) never come here.
import re
from datetime import datetime, timezone

import requests

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
