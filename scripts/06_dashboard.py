"""
QUANTSYS — Deribit BTC Options Risk Terminal (server HTTP single-file).
Esegui / Run:  python scripts/06_dashboard.py
Apri / Open:   http://localhost:8050

IT: Terminale istituzionale per l'analisi delle opzioni crypto. Si connette ai
    dati pubblici Deribit (REST, no-auth) per spot BTC + chain opzioni completa
    (mark/bid/ask, mark_iv, open interest, volume, forward per-expiry), calcola
    le Greche in tempo reale (Black-Scholes forward-measure) sull'intera option
    chain e visualizza la Superficie di Volatilità (3D), gli smile per scadenza,
    la term structure ATM e la distribuzione del rischio (OI/Greche aggregate).
    La tab Trades mostra il forward test vol di 04b: storico settled da
    results/vol_paper/trades.jsonl + posizione APERTA da position.json.
    NON usa i modelli ML del progetto: è un risk terminal di mercato, GPU-free.
EN: Institutional crypto-options analytics terminal. Connects to Deribit public
    data (REST, no-auth) for BTC spot + full option chain (mark/bid/ask, mark_iv,
    open interest, volume, per-expiry forward), computes Greeks in real time
    (Black-Scholes forward-measure) over the whole chain, and renders the
    Volatility Surface (3D), per-expiry smiles, the ATM term structure and the
    risk distribution (OI / aggregate Greeks). The Trades tab shows 04b's vol
    forward test: settled history from results/vol_paper/trades.jsonl + the OPEN
    position from position.json. Does NOT touch the project's ML models: it is
    a market risk terminal, GPU-free.
"""
import gzip as _gzip
import hmac
import json
import logging
import math
import re as _re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import numpy as np
import requests

# IT: scipy è già dipendenza del progetto (statsmodels/Markov) → norm vettoriale.
#     Fallback a math.erf se assente (greche comunque corrette, solo più lente).
# EN: scipy is already a project dependency (statsmodels/Markov) → vectorized
#     normal. Fallback to math.erf if missing (greeks still correct, just slower).
try:
    from scipy.stats import norm as _scipy_norm
    _ndtr = _scipy_norm.cdf
    _npdf = _scipy_norm.pdf
except Exception:  # pragma: no cover - scipy quasi sempre presente / almost always present
    def _ndtr(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

    def _npdf(x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

try:
    from quantsys.utils import setup_logging, load_config
    setup_logging()
    _CFG = load_config("config/default.yaml")
except Exception:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    _CFG = {}
log = logging.getLogger("quantsys.dashboard")

_DCFG = _CFG.get("dashboard", {}) if isinstance(_CFG, dict) else {}
HOST        = _DCFG.get("host", "127.0.0.1")
PORT        = int(_DCFG.get("port", 8050))
AUTH_TOKEN  = str(_DCFG.get("auth_token", "") or "")
ENABLE_GZIP = bool(_DCFG.get("enable_gzip", True))
CURRENCY    = str(_DCFG.get("options_currency", "BTC")).upper()

# IT: anno solare (in secondi) per l'annualizzazione del time-to-expiry.
# EN: calendar year (seconds) for time-to-expiry annualization.
YEAR_SECONDS = 365.0 * 24.0 * 3600.0

# ═══════════════════════════════════════════════════════════════════════════════
# IT: 1) DATA LAYER DERIBIT — REST pubblico con cache TTL in-memory.
# EN: 1) DERIBIT DATA LAYER — public REST with in-memory TTL cache.
# ═══════════════════════════════════════════════════════════════════════════════

# IT: mainnet pubblica (solo dati di mercato in lettura, nessuna auth).
# EN: public mainnet (read-only market data, no auth).
DERIBIT_BASE = "https://www.deribit.com/api/v2"
_HTTP = requests.Session()
_HTTP.headers.update({"User-Agent": "quantsys-risk-terminal/1.0"})


class _TTLCache:
    # IT: cache thread-safe con scadenza: protegge i public endpoint Deribit dal
    #     polling concorrente del browser (1 fetch reale ogni `ttl` secondi).
    #     Un lock per-chiave serializza i fetch concorrenti (no thundering herd).
    # EN: thread-safe expiring cache: shields Deribit public endpoints from the
    #     browser's concurrent polling (1 real fetch every `ttl` seconds).
    #     A per-key lock serializes concurrent fetches (no thundering herd).
    def __init__(self):
        self._lock = threading.Lock()
        self._store = {}        # key -> (expires_at, value)
        self._keylocks = {}     # key -> threading.Lock

    def _keylock(self, key):
        with self._lock:
            kl = self._keylocks.get(key)
            if kl is None:
                kl = self._keylocks[key] = threading.Lock()
            return kl

    def get_or_fetch(self, key, ttl, fetch_fn):
        now = time.time()
        with self._lock:
            hit = self._store.get(key)
            if hit and hit[0] > now:
                return hit[1]
        # IT: fetch sotto lock-per-chiave → un solo thread va in rete, gli altri
        #     attendono e poi leggono la cache appena popolata.
        # EN: fetch under a per-key lock → only one thread hits the network, the
        #     others wait and then read the freshly populated cache.
        with self._keylock(key):
            now = time.time()
            with self._lock:
                hit = self._store.get(key)
                if hit and hit[0] > now:
                    return hit[1]
            value = fetch_fn()
            with self._lock:
                self._store[key] = (time.time() + ttl, value)
            return value


_CACHE = _TTLCache()


def _deribit_get(path: str, params: dict, timeout: int = 12) -> dict:
    # IT: GET pubblica con raise sugli errori; il chiamante decide il fallback.
    # EN: public GET that raises on errors; the caller decides the fallback.
    r = _HTTP.get(f"{DERIBIT_BASE}/{path}", params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    if "result" not in payload:
        raise RuntimeError(f"Deribit unexpected response: {payload}")
    return payload["result"]


def fetch_index_price(currency: str = CURRENCY) -> float:
    # IT: prezzo indice spot (media multi-exchange Deribit), tenor-0 del forward.
    # EN: spot index price (Deribit multi-exchange average), tenor-0 of the forward.
    def _f():
        res = _deribit_get("public/get_index_price",
                           {"index_name": f"{currency.lower()}_usd"})
        return float(res["index_price"])
    return _CACHE.get_or_fetch(f"index:{currency}", 4.0, _f)


def fetch_dvol(currency: str = CURRENCY) -> float:
    # IT: ultimo punto dell'indice DVOL (vol implicita 30d annualizzata, %).
    # EN: latest DVOL index point (30d annualized implied vol, %).
    def _f():
        now_ms = int(time.time() * 1000)
        res = _deribit_get("public/get_volatility_index_data",
                           {"currency": currency,
                            "start_timestamp": now_ms - 2 * 3600 * 1000,
                            "end_timestamp": now_ms, "resolution": "3600"})
        data = res.get("data", [])
        return float(data[-1][4]) if data else float("nan")
    try:
        return _CACHE.get_or_fetch(f"dvol:{currency}", 60.0, _f)
    except Exception:
        return float("nan")


def fetch_option_chain(currency: str = CURRENCY) -> list:
    # IT: 1 chiamata → riepilogo book di TUTTA la chain opzioni (mark_iv, mark,
    #     bid/ask, open_interest, volume, underlying_price per-strumento). È la
    #     sorgente unica da cui si derivano greche, surface, smile e term struct.
    # EN: 1 call → book summary of the WHOLE option chain (mark_iv, mark, bid/ask,
    #     open_interest, volume, per-instrument underlying_price). Single source
    #     from which greeks, surface, smile and term structure are derived.
    def _f():
        # IT: una chain valida ha centinaia di strumenti; un risultato vuoto/parziale
        #     (hiccup Deribit) NON va in cache, altrimenti svuoterebbe i grafici (barre
        #     OI che spariscono) per tutto il TTL. Retry singolo, poi si solleva →
        #     get_or_fetch NON cacha e il frontend tiene l'ultimo buono.
        # EN: a valid chain has hundreds of instruments; an empty/partial result
        #     (Deribit hiccup) must NOT be cached, else it blanks the charts (OI bars
        #     vanishing) for the whole TTL. Single retry, then raise → get_or_fetch
        #     does NOT cache and the frontend keeps the last good one.
        for _attempt in range(2):
            res = _deribit_get("public/get_book_summary_by_currency",
                               {"currency": currency, "kind": "option"})
            if isinstance(res, list) and len(res) >= 50:
                return res
            time.sleep(0.3)
        raise RuntimeError(f"chain Deribit vuota/parziale dopo retry "
                           f"({len(res) if isinstance(res, list) else type(res).__name__})")
    return _CACHE.get_or_fetch(f"chain:{currency}", 8.0, _f)


# IT: nome strumento Deribit: BTC-27JUN25-100000-C → (expiry 08:00 UTC, K, tipo).
# EN: Deribit instrument name: BTC-27JUN25-100000-C → (expiry 08:00 UTC, K, type).
_INSTR_RE = _re.compile(r"^[A-Z]+-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-([CP])$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
_MONTH_NAME = {v: k for k, v in _MONTHS.items()}


def _parse_instrument(name: str):
    m = _INSTR_RE.match(name or "")
    if not m:
        return None
    day, mon, yy, strike_s, opt = m.groups()
    expiry = datetime(2000 + int(yy), _MONTHS[mon], int(day), 8, 0,
                      tzinfo=timezone.utc)
    return expiry, float(strike_s), opt


# ═══════════════════════════════════════════════════════════════════════════════
# IT: 2) MOTORE GRECHE — Black-Scholes forward-measure (r=0, opzioni sul forward).
#        Deribit quota le opzioni in BTC (inverse, europee, cash-settled): le
#        Greche qui sono in convenzione "USD" (prezzo in USD, sottostante = forward
#        per-expiry), lo standard di un risk terminal di mercato. Vettorializzate.
# EN: 2) GREEKS ENGINE — Black-Scholes forward-measure (r=0, options on forward).
#        Deribit quotes options in BTC (inverse, European, cash-settled): greeks
#        here use the "USD" convention (USD price, underlying = per-expiry
#        forward), the market risk-terminal standard. Vectorized.
# ═══════════════════════════════════════════════════════════════════════════════

def bs_greeks(F, K, T, sigma, opt_type):
    # IT: F=forward, K=strike, T=anni a scadenza, sigma=vol (frazione, 0.55=55%),
    #     opt_type 'C'/'P' (array o scalare). Ritorna dict di array float.
    #     Convenzioni display: vega per +1 vol-point (1%), theta per giorno.
    # EN: F=forward, K=strike, T=years to expiry, sigma=vol (fraction, 0.55=55%),
    #     opt_type 'C'/'P' (array or scalar). Returns dict of float arrays.
    #     Display conventions: vega per +1 vol-point (1%), theta per day.
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(opt_type) == "C"

    # IT: guardia numerica: T e sigma minimi per evitare /0 nelle scadenze brevi.
    # EN: numeric guard: floor T and sigma to avoid /0 on very short expiries.
    Tg = np.maximum(T, 1e-6)
    sg = np.maximum(sigma, 1e-6)
    sqrtT = np.sqrt(Tg)
    vol_sqrtT = sg * sqrtT

    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sg * sg * Tg) / vol_sqrtT
        d2 = d1 - vol_sqrtT
    nd1 = _npdf(d1)
    Nd1 = _ndtr(d1)
    Nd2 = _ndtr(d2)

    # IT: prezzo teorico USD (r=0 → forward measure, niente sconto): riferimento
    #     accanto al mark Deribit (quotato in BTC).
    # EN: theoretical USD price (r=0 → forward measure, no discount): a reference
    #     alongside Deribit's mark (quoted in BTC).
    price = np.where(is_call, F * Nd1 - K * Nd2, K * (1.0 - Nd2) - F * (1.0 - Nd1))

    delta = np.where(is_call, Nd1, Nd1 - 1.0)
    gamma = nd1 / (F * vol_sqrtT)
    vega = F * nd1 * sqrtT / 100.0                      # per +1% vol
    theta = (-(F * nd1 * sg) / (2.0 * sqrtT)) / 365.0   # per giorno / per day
    # IT: rho ≈ 0 con r=0; riportato come sensibilità di forma (call/put).
    # EN: rho ≈ 0 with r=0; reported as a shape sensitivity (call/put).
    rho = np.where(is_call, K * Tg * Nd2, -K * Tg * (1.0 - Nd2)) / 100.0

    # IT: scadenze spirate / d1 non finiti → greche azzerate (no inf nel JSON).
    # EN: expired contracts / non-finite d1 → zeroed greeks (no inf in JSON).
    dead = (T <= 0) | ~np.isfinite(d1)
    for arr in (price, delta, gamma, vega, theta, rho):
        arr[dead] = 0.0
    return {"price": price, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho}


# ═══════════════════════════════════════════════════════════════════════════════
# IT: 3) ASSEMBLAGGIO — chain normalizzata con greche, surface, smile, risk.
# EN: 3) ASSEMBLY — normalized chain with greeks, surface, smile, risk.
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(v, default=0.0):
    # IT: None/NaN/inf → default (il JSON non serializza NaN/inf in modo robusto).
    # EN: None/NaN/inf → default (JSON does not robustly serialize NaN/inf).
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _optf(v):
    # IT: come _safe ma PRESERVA l'assenza: None/NaN/inf → None (JSON null).
    #     Per i campi opzionali (delivery_price, DVOL, …) dove 0.0 è un valore
    #     FUORVIANTE: il frontend mostra '—' su null, e un delivery_price=0.0
    #     fantasma manderebbe il profilo payoff in divisione-per-zero.
    # EN: like _safe but PRESERVES absence: None/NaN/inf → None (JSON null).
    #     For optional fields (delivery_price, DVOL, …) where 0.0 is a MISLEADING
    #     value: the frontend renders '—' on null, and a phantom delivery_price=0.0
    #     would drive the payoff profile into a division-by-zero.
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def build_market(currency: str = CURRENCY) -> dict:
    # IT: stato di mercato completo da 1 snapshot chain + index: righe
    #     per-strumento con greche calcolate, più la lista delle expiry vive.
    # EN: full market state from 1 chain snapshot + index: per-instrument rows
    #     with computed greeks, plus the list of live expiries.
    raw = fetch_option_chain(currency)
    spot = fetch_index_price(currency)
    now = datetime.now(timezone.utc)
    now_ts = now.timestamp()

    rows = []
    for it in raw:
        parsed = _parse_instrument(it.get("instrument_name", ""))
        iv = it.get("mark_iv")
        if parsed is None or iv is None:
            continue
        expiry, strike, opt = parsed
        T = (expiry.timestamp() - now_ts) / YEAR_SECONDS
        if T <= 0:
            continue
        fwd = _safe(it.get("underlying_price"), spot)
        rows.append({
            "instrument": it["instrument_name"],
            "expiry_ts": expiry.timestamp(),
            "expiry_label": f"{expiry.day}{_MONTH_NAME[expiry.month]}{str(expiry.year)[2:]}",
            "T": T,
            "days": T * 365.0,
            "strike": strike,
            "type": opt,
            "iv": _safe(iv) / 100.0,          # frazione / fraction
            "iv_pct": _safe(iv),              # percento / percent
            "forward": fwd,
            "moneyness": (strike / fwd) if fwd > 0 else float("nan"),
            "mark": _safe(it.get("mark_price")),
            "bid": _safe(it.get("bid_price")),
            "ask": _safe(it.get("ask_price")),
            "oi": _safe(it.get("open_interest")),
            "volume": _safe(it.get("volume")),
        })

    if not rows:
        return {"spot": spot, "expiries": [], "rows": [],
                "ts": now.isoformat(), "currency": currency}

    # IT: greche vettoriali su tutta la chain in un colpo solo.
    # EN: vectorized greeks over the whole chain in one shot.
    F = np.array([r["forward"] for r in rows])
    K = np.array([r["strike"] for r in rows])
    T = np.array([r["T"] for r in rows])
    sig = np.array([r["iv"] for r in rows])
    typ = np.array([r["type"] for r in rows])
    g = bs_greeks(F, K, T, sig, typ)
    for i, r in enumerate(rows):
        r["delta"] = _safe(g["delta"][i])
        r["gamma"] = _safe(g["gamma"][i])
        r["vega"] = _safe(g["vega"][i])
        r["theta"] = _safe(g["theta"][i])
        r["rho"] = _safe(g["rho"][i])
        r["theo"] = _safe(g["price"][i])

    expiries = sorted({(r["expiry_ts"], r["expiry_label"], round(r["days"], 2))
                       for r in rows})
    exp_list = [{"ts": e[0], "label": e[1], "days": e[2]} for e in expiries]
    return {"spot": spot, "expiries": exp_list, "rows": rows,
            "ts": now.isoformat(), "currency": currency}


def _atm_iv_for_expiry(rows_e: list) -> float:
    # IT: ATM IV = media mark_iv dei contratti con strike più vicino al forward.
    # EN: ATM IV = mean mark_iv of the contracts with strike closest to forward.
    if not rows_e:
        return float("nan")
    fwd = np.median([r["forward"] for r in rows_e])
    strikes = np.array(sorted({r["strike"] for r in rows_e}))
    atm_k = strikes[int(np.argmin(np.abs(strikes - fwd)))]
    ivs = [r["iv_pct"] for r in rows_e if r["strike"] == atm_k]
    return float(np.mean(ivs)) if ivs else float("nan")


def build_summary(market: dict) -> dict:
    # IT: metriche di testata del risk terminal: spot, DVOL, ATM IV ~30g,
    #     OI/volume totali, put/call ratio, conteggi.
    # EN: risk-terminal header metrics: spot, DVOL, ~30d ATM IV, total OI/volume,
    #     put/call ratio, counts.
    rows = market["rows"]
    total_oi = sum(r["oi"] for r in rows)
    total_vol = sum(r["volume"] for r in rows)
    call_oi = sum(r["oi"] for r in rows if r["type"] == "C")
    put_oi = sum(r["oi"] for r in rows if r["type"] == "P")
    pcr = (put_oi / call_oi) if call_oi > 0 else float("nan")

    # IT: ATM IV alla scadenza più vicina a 30 giorni (proxy IV "1m").
    # EN: ATM IV at the expiry nearest to 30 days (proxy for "1m" IV).
    atm30 = float("nan")
    if market["expiries"]:
        target = min(market["expiries"], key=lambda e: abs(e["days"] - 30.0))
        rows_e = [r for r in rows if r["expiry_ts"] == target["ts"]]
        atm30 = _atm_iv_for_expiry(rows_e)

    # IT: dvol/atm/pcr sono opzionali (fetch fallito / chain vuota): null → '—' nel
    #     frontend, non un fuorviante 0.0%.
    # EN: dvol/atm/pcr are optional (failed fetch / empty chain): null → '—' in the
    #     frontend, not a misleading 0.0%.
    return {
        "spot": _safe(market["spot"]),
        "dvol": _optf(fetch_dvol(market.get("currency", CURRENCY))),
        "atm_iv_30d": _optf(atm30),
        "total_oi": _safe(total_oi),
        "total_volume": _safe(total_vol),
        "call_oi": _safe(call_oi),
        "put_oi": _safe(put_oi),
        "put_call_ratio": _optf(pcr),
        "n_instruments": len(rows),
        "n_expiries": len(market["expiries"]),
        "ts": market["ts"],
    }


def build_surface(market: dict) -> dict:
    # IT: superficie IV interpolata su griglia comune di moneyness (K/F) per ogni
    #     expiry. Per scadenza: mediana mark_iv per strike (C/P collassati via
    #     parità in IV), poi np.interp sul grid; fuori dal range osservato → NaN
    #     (niente extrapolazione → buchi puliti nel rendering Plotly).
    # EN: IV surface interpolated onto a common moneyness (K/F) grid per expiry.
    #     Per expiry: median mark_iv per strike (C/P collapsed via IV parity),
    #     then np.interp onto the grid; outside the observed range → NaN (no
    #     extrapolation → clean gaps in the Plotly render).
    rows = market["rows"]
    grid = np.round(np.linspace(0.6, 1.6, 41), 4)   # K/F, ATM=1.0
    exps = sorted(market["expiries"], key=lambda e: e["days"])
    z, days, labels, smiles = [], [], [], []
    for e in exps:
        rs = [r for r in rows if r["expiry_ts"] == e["ts"]]
        if len(rs) < 3:
            continue
        fwd_e = float(np.median([r["forward"] for r in rs]))
        by_k = {}
        for r in rs:
            by_k.setdefault(round(r["moneyness"], 4), []).append(r["iv_pct"])
        m = np.array(sorted(by_k))
        iv = np.array([float(np.median(by_k[k])) for k in m])
        row = np.interp(grid, m, iv, left=np.nan, right=np.nan)
        z.append([None if not math.isfinite(v) else round(float(v), 3) for v in row])
        days.append(round(e["days"], 2))
        labels.append(e["label"])
        # IT: smile raw (per il grafico 2D) in strike reali + moneyness.
        # EN: raw smile (for the 2D chart) in real strikes + moneyness.
        smiles.append({
            "label": e["label"], "days": round(e["days"], 2), "forward": _safe(fwd_e),
            "strikes": [round(float(k * fwd_e), 0) for k in m],
            "moneyness": [round(float(k), 4) for k in m],
            "iv": [round(float(v), 3) for v in iv],
        })
    return {"moneyness": [float(x) for x in grid], "days": days,
            "labels": labels, "z": z, "smiles": smiles,
            "spot": _safe(market["spot"])}


def build_term_structure(market: dict) -> dict:
    # IT: term structure ATM IV (IV vs giorni a scadenza) + forward/OI per expiry.
    # EN: ATM IV term structure (IV vs days to expiry) + forward/OI per expiry.
    rows = market["rows"]
    out = []
    for e in sorted(market["expiries"], key=lambda x: x["days"]):
        rs = [r for r in rows if r["expiry_ts"] == e["ts"]]
        out.append({"label": e["label"], "days": round(e["days"], 2),
                    "atm_iv": _safe(_atm_iv_for_expiry(rs)),
                    "forward": _safe(np.median([r["forward"] for r in rs]) if rs else float("nan")),
                    "oi": _safe(sum(r["oi"] for r in rs))})
    return {"term": out, "dvol": _optf(fetch_dvol(market.get("currency", CURRENCY)))}


def build_chain_table(market: dict, expiry_ts) -> dict:
    # IT: chain a doppio lato (call|put per strike) per UNA scadenza, con greche.
    #     Selezione expiry più vicina al ts richiesto (robusto agli arrotondamenti);
    #     default = scadenza più vicina a 30 giorni.
    # EN: two-sided chain (call|put per strike) for ONE expiry, with greeks.
    #     Picks the expiry nearest the requested ts (robust to rounding);
    #     default = expiry nearest 30 days.
    if not market["expiries"]:
        return {"expiry": None, "rows": [], "forward": None, "spot": _safe(market["spot"])}
    if expiry_ts is None:
        target = min(market["expiries"], key=lambda e: abs(e["days"] - 30.0))
    else:
        target = min(market["expiries"], key=lambda e: abs(e["ts"] - expiry_ts))
    rs = [r for r in market["rows"] if r["expiry_ts"] == target["ts"]]
    fwd = float(np.median([r["forward"] for r in rs])) if rs else market["spot"]
    by_strike = {}
    for r in rs:
        d = by_strike.setdefault(r["strike"], {"strike": r["strike"]})
        side = "call" if r["type"] == "C" else "put"
        d[side] = {k: r[k] for k in ("bid", "ask", "mark", "iv_pct", "oi",
                                     "volume", "delta", "gamma", "vega",
                                     "theta", "rho", "theo")}
    table = [by_strike[k] for k in sorted(by_strike)]
    return {"expiry": target, "forward": _safe(fwd),
            "spot": _safe(market["spot"]), "rows": table}


def build_risk(market: dict) -> dict:
    # IT: vista rischio: OI per strike (call vs put), max-pain, greche aggregate
    #     pesate per OI (esposizione dealer-implied del book), DVOL.
    # EN: risk view: OI by strike (call vs put), max-pain, OI-weighted aggregate
    #     greeks (dealer-implied book exposure), DVOL.
    rows = market["rows"]
    spot = market["spot"]
    strikes = sorted({r["strike"] for r in rows})
    call_oi = {k: 0.0 for k in strikes}
    put_oi = {k: 0.0 for k in strikes}
    for r in rows:
        (call_oi if r["type"] == "C" else put_oi)[r["strike"]] += r["oi"]

    # IT: max-pain = strike che minimizza il payoff totale ai detentori a scadenza
    #     (somma sui contratti aperti). Vettoriale O(n_strikes²): ~poche centinaia.
    # EN: max-pain = strike minimizing total holder payoff at expiry (sum over open
    #     contracts). Vectorized O(n_strikes²): a few hundred at most.
    max_pain = float("nan")
    if strikes:
        ks = np.array(strikes, dtype=float)              # IT/EN: strike ordinati crescenti
        call_arr = np.array([call_oi[k] for k in strikes])
        put_arr = np.array([put_oi[k] for k in strikes])
        # IT: pain(p) = Σ_K (p−K)⁺·OI_call + Σ_K (K−p)⁺·OI_put, valutata su ogni strike.
        #     Forma chiusa O(n) via prefix/suffix sum (gli strike sono già ordinati):
        #       call (K≤p): p·Σ_{K≤p}OI_c − Σ_{K≤p}K·OI_c   → cumsum
        #       put  (K≥p): Σ_{K≥p}K·OI_p − p·Σ_{K≥p}OI_p   → suffix-sum
        #     La diagonale K=p contribuisce 0 in entrambi (esatto vs la vecchia O(n²)).
        # EN: pain(p) = Σ_K (p−K)⁺·OI_call + Σ_K (K−p)⁺·OI_put, evaluated at each strike.
        #     Closed-form O(n) via prefix/suffix sums (strikes already sorted):
        #       call (K≤p): p·Σ_{K≤p}OI_c − Σ_{K≤p}K·OI_c   → cumsum
        #       put  (K≥p): Σ_{K≥p}K·OI_p − p·Σ_{K≥p}OI_p   → suffix-sum
        #     The K=p diagonal contributes 0 in both (exact vs the old O(n²)).
        call_cum = np.cumsum(call_arr)
        wcall_cum = np.cumsum(ks * call_arr)
        put_suf = np.cumsum(put_arr[::-1])[::-1]
        wput_suf = np.cumsum((ks * put_arr)[::-1])[::-1]
        pain = (ks * call_cum - wcall_cum) + (wput_suf - ks * put_suf)
        max_pain = float(ks[int(np.argmin(pain))])

    agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for r in rows:
        for gk in agg:
            agg[gk] += r[gk] * r["oi"]

    return {
        "strikes": [float(k) for k in strikes],
        "call_oi": [_safe(call_oi[k]) for k in strikes],
        "put_oi": [_safe(put_oi[k]) for k in strikes],
        "max_pain": _safe(max_pain),
        "spot": _safe(spot),
        "agg_greeks": {k: _safe(v) for k, v in agg.items()},
        "dvol": _optf(fetch_dvol(market.get("currency", CURRENCY))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IT: 3b) FORWARD TEST — trade dello straddle vol (04b_vol_paper.py): storico settled
#     da trades.jsonl (append-only al settlement) + posizione APERTA da position.json
#     (04b la scrive all'open e la azzera al settle). Per ogni trade: lato (LONG/SHORT
#     straddle = long/short vol), strike, spot di ingresso, premio, prezzo di
#     settlement, payoff e PnL (BTC) + sintesi aggregata. I campi di settlement sono
#     null finché il trade è aperto (_optf, MAI 0.0 fantasma).
# EN: 3b) FORWARD TEST — vol straddle trades (04b_vol_paper.py): settled history from
#     trades.jsonl (append-only at settlement) + OPEN position from position.json
#     (04b writes it on open, clears it on settle). Per trade: side (LONG/SHORT
#     straddle = long/short vol), strike, entry spot, premium, settlement price,
#     payoff and PnL (BTC) + aggregated summary. Settlement fields stay null while
#     the trade is open (_optf, NEVER a phantom 0.0).
# ═══════════════════════════════════════════════════════════════════════════════
TRADES_PATH = Path("results/vol_paper/trades.jsonl")
POSITION_PATH = Path("results/vol_paper/position.json")


def _trade_row(t: dict) -> dict:
    # IT: normalizza un record 04b (riga trades.jsonl O position.json — stesso schema,
    #     la posizione aperta è semplicemente senza campi di settlement).
    # EN: normalize a 04b record (trades.jsonl line OR position.json — same schema,
    #     the open position simply lacks the settlement fields).
    prem = float(t.get("prem_call", 0) or 0) + float(t.get("prem_put", 0) or 0)
    return {
        "entry_ts": t.get("entry_ts"), "settled_ts": t.get("settled_ts"),
        "side": int(t.get("side", 1)),                 # IT/EN: 1 LONG straddle, -1 SHORT
        "executed": bool(t.get("executed", False)),
        "settled": t.get("settled_ts") is not None,
        "strike": _safe(t.get("strike")),
        "entry_spot": _safe(t.get("index_at_entry")),
        "delivery_price": _optf(t.get("delivery_price")),
        "prem_call": _safe(t.get("prem_call")), "prem_put": _safe(t.get("prem_put")),
        "premium": _safe(prem), "fee_btc": _safe(t.get("fee_btc")),
        "amount": _safe(t.get("amount", 1.0)),
        "payoff_btc": _optf(t.get("payoff_btc")), "pnl_btc": _optf(t.get("pnl_btc")),
        "edge": _safe(t.get("edge")), "rv_pred": _safe(t.get("rv_pred")),
        "var_iv": _safe(t.get("var_iv")), "t_hours": _safe(t.get("t_hours_at_entry")),
        "expiry_ms": _optf(t.get("expiry_ms")),
        "call": t.get("call"), "put": t.get("put"),
    }


def build_trades() -> dict:
    rows = []
    if TRADES_PATH.exists():
        for line in TRADES_PATH.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(_trade_row(json.loads(line)))
            except Exception:
                continue
    # IT: posizione aperta in coda (è sempre la più recente): status 'open' nel
    #     frontend, profilo di rischio dal premio (nessun settlement da calibrare).
    # EN: open position appended last (always the most recent): 'open' status in the
    #     frontend, risk profile from the premium (no settlement to calibrate).
    n_open = 0
    if POSITION_PATH.exists():
        try:
            pos = json.loads(POSITION_PATH.read_text(encoding="utf-8"))
            if isinstance(pos, dict) and pos.get("strike") is not None:
                rows.append(_trade_row(pos))
                n_open = 1
        except Exception:
            log.warning("position.json illeggibile — riga open omessa / unreadable, open row skipped")
    if not rows:
        return {"trades": [], "summary": {"n": 0, "n_settled": 0, "n_open": 0, "gate_trades": 30,
                                          "note": "nessun trade (forward test 04b non avviato)"}}
    settled = [r for r in rows if r["settled"] and r["pnl_btc"] is not None]
    pnls = [r["pnl_btc"] for r in settled]
    n_s = len(settled)
    wins = sum(1 for p in pnls if p > 0)
    summary = {
        "n": len(rows), "n_settled": n_s, "n_open": n_open,
        "n_executed": sum(1 for r in rows if r["executed"]),
        "total_pnl": _safe(sum(pnls)) if pnls else 0.0,
        "hit_rate": _safe(wins / n_s) if n_s else None,
        "avg_pnl": _safe(sum(pnls) / n_s) if n_s else None,
        "best": _safe(max(pnls)) if pnls else None,
        "worst": _safe(min(pnls)) if pnls else None,
        "gate_trades": 30,
    }
    return {"trades": rows, "summary": summary}


# ═══════════════════════════════════════════════════════════════════════════════
# IT: 4) FRONTEND — SPA istituzionale (Plotly.js CDN per la superficie 3D).
# EN: 4) FRONTEND — institutional SPA (Plotly.js CDN for the 3D surface).
# ═══════════════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QUANTSYS · Deribit Options Risk Terminal</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230b0e14'/%3E%3Cpath d='M4 20 C10 6 22 26 28 10' stroke='%23f0a020' stroke-width='2.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root{
    --bg:#0b0e14; --surface:#11151f; --surface2:#161b27; --border:#222a38;
    --text:#dfe6f0; --muted:#7d8aa0; --green:#2ecc71; --red:#ff5c5c;
    --amber:#f0a020; --blue:#4aa3ff; --violet:#b07cff;
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font:13px/1.4 'Segoe UI',system-ui,sans-serif;overflow-y:scroll;}
  header{background:var(--surface);border-bottom:1px solid var(--border);
    padding:8px 18px;display:flex;align-items:center;gap:18px;flex-wrap:wrap;}
  .brand{font-size:15px;font-weight:700;letter-spacing:.5px;color:var(--amber);white-space:nowrap;}
  .brand small{color:var(--muted);font-weight:400;font-size:10px;letter-spacing:0;}
  .hdr-metric{display:flex;flex-direction:column;line-height:1.2;}
  .hdr-metric .lbl{font-size:9px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);}
  .hdr-metric .val{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums;}
  #conn{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:10px;color:var(--muted);}
  #conn .dot{width:8px;height:8px;border-radius:50%;background:var(--muted);}
  #conn.ok .dot{background:var(--green);box-shadow:0 0 6px var(--green);}
  #conn.err .dot{background:var(--red);box-shadow:0 0 6px var(--red);}
  .tabs{display:flex;gap:2px;padding:0 18px;background:var(--surface);border-bottom:1px solid var(--border);}
  .tab{padding:9px 18px;cursor:pointer;color:var(--muted);font-size:12px;font-weight:600;
    border-bottom:2px solid transparent;}
  .tab:hover{color:var(--text);}
  .tab.active{color:var(--amber);border-bottom-color:var(--amber);}
  .page{display:none;padding:16px 18px;}
  .page.active{display:block;}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
  .panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;}
  .panel h3{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
    margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;gap:8px;}
  .plot{width:100%;height:340px;}
  .plot.tall{height:520px;}
  select{background:var(--surface2);color:var(--text);border:1px solid var(--border);
    border-radius:5px;padding:4px 8px;font-size:11px;}
  table{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums;}
  th{color:var(--muted);font-weight:600;text-align:right;padding:5px 7px;border-bottom:1px solid var(--border);
    position:sticky;top:0;background:var(--surface);white-space:nowrap;}
  td{padding:4px 7px;border-bottom:1px solid #161b27;text-align:right;white-space:nowrap;}
  .scroll{max-height:560px;overflow:auto;}
  .k-col{text-align:center;font-weight:700;background:var(--surface2);color:var(--text);}
  .atm-row td{background:#1d2433;}
  .call-side{color:#9fd8b6;} .put-side{color:#e6b0b0;}
  .pos{color:var(--green);} .neg{color:var(--red);} .amb{color:var(--amber);} .blu{color:var(--blue);}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:14px;}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px;}
  .card .lbl{font-size:9px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:5px;}
  .card .val{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums;}
  .card .sub{font-size:10px;color:var(--muted);margin-top:3px;}
  .ctrl-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px;}
  .ctrl-row label{font-size:11px;color:var(--muted);}
  .legend{font-size:10px;color:var(--muted);margin-top:6px;}
</style>
</head>
<body>
<header>
  <div class="brand">⚡ QUANTSYS <small>· DERIBIT OPTIONS RISK TERMINAL</small></div>
  <div class="hdr-metric"><span class="lbl">BTC Index</span><span class="val" id="h-spot">—</span></div>
  <div class="hdr-metric"><span class="lbl">DVOL (30d)</span><span class="val amb" id="h-dvol">—</span></div>
  <div class="hdr-metric"><span class="lbl">ATM IV 30d</span><span class="val blu" id="h-atm">—</span></div>
  <div class="hdr-metric"><span class="lbl">Total OI (BTC)</span><span class="val" id="h-oi">—</span></div>
  <div class="hdr-metric"><span class="lbl">24h Vol (BTC)</span><span class="val" id="h-vol">—</span></div>
  <div class="hdr-metric"><span class="lbl">Put/Call OI</span><span class="val" id="h-pcr">—</span></div>
  <div id="conn"><span class="dot"></span><span id="conn-txt">connecting…</span></div>
</header>

<div class="tabs">
  <div class="tab active" data-tab="surface" onclick="switchTab('surface')">Volatility Surface</div>
  <div class="tab" data-tab="chain" onclick="switchTab('chain')">Option Chain</div>
  <div class="tab" data-tab="risk" onclick="switchTab('risk')">Risk &amp; Greeks</div>
  <div class="tab" data-tab="trades" onclick="switchTab('trades')">Trades</div>
</div>

<!-- ── VOL SURFACE ── -->
<div id="page-surface" class="page active">
  <div class="panel" style="margin-bottom:14px;">
    <h3>Implied Volatility Surface <span style="font-weight:400;text-transform:none;letter-spacing:0">IV (%) · moneyness K/F · days to expiry</span></h3>
    <div id="plot-surface" class="plot tall"></div>
    <div class="legend">3D surface interpolated on a common moneyness grid (K/F). Gaps = strikes outside the quoted range (no extrapolation).</div>
  </div>
  <div class="grid2">
    <div class="panel">
      <h3>Volatility Smile <select id="smile-sel" onchange="renderSmile()"></select></h3>
      <div id="plot-smile" class="plot"></div>
    </div>
    <div class="panel">
      <h3>ATM Term Structure <span style="font-weight:400;text-transform:none;letter-spacing:0">IV (%) vs days</span></h3>
      <div id="plot-term" class="plot"></div>
    </div>
  </div>
</div>

<!-- ── OPTION CHAIN ── -->
<div id="page-chain" class="page">
  <div class="ctrl-row">
    <label>Expiry</label><select id="chain-sel" onchange="loadChain()"></select>
    <span id="chain-fwd" style="color:var(--muted);font-size:11px;"></span>
  </div>
  <div class="panel">
    <div class="scroll">
      <table id="chain-table">
        <thead><tr id="chain-head"></tr></thead>
        <tbody id="chain-body"></tbody>
      </table>
    </div>
    <div class="legend">Calls (left) · Strike · Puts (right). Greeks computed live (Black-Scholes, forward measure, USD convention). Δ delta · Γ gamma · ν vega (per +1% vol) · Θ theta (per day). ATM strike highlighted.</div>
  </div>
</div>

<!-- ── RISK & GREEKS ── -->
<div id="page-risk" class="page">
  <div class="cards" id="risk-cards"></div>
  <div class="panel" style="margin-bottom:14px;">
    <h3>Open Interest by Strike
      <span style="display:flex;align-items:center;gap:8px;font-weight:400;text-transform:none;letter-spacing:0">
        <span>calls ▲ / puts ▼ · net OI · spot &amp; max-pain</span>
        <select id="oi-metric" onchange="loadRisk()">
          <option value="btc">Contracts (BTC)</option>
          <option value="usd">Notional (USD)</option>
        </select>
      </span>
    </h3>
    <div id="plot-oi" class="plot tall"></div>
  </div>
  <div class="grid2">
    <div class="panel"><h3>OI-Weighted Aggregate Greeks</h3><div id="plot-greeks" class="plot"></div></div>
    <div class="panel"><h3>Put/Call OI Split</h3><div id="plot-pcr" class="plot"></div></div>
  </div>
</div>

<!-- ── TRADES (forward test vol-paper 04b) ── -->
<div id="page-trades" class="page">
  <div class="cards" id="trades-cards"></div>
  <div class="grid2">
    <div class="panel">
      <h3>Trade History <span style="font-weight:400;text-transform:none;letter-spacing:0">vol straddle · 04b forward test</span></h3>
      <div class="scroll">
        <table id="trades-table"><thead><tr id="trades-head"></tr></thead><tbody id="trades-body"></tbody></table>
      </div>
      <div class="legend">Clicca una riga per il profilo di rischio. LONG straddle = long vol (profitto se |mossa| &gt; breakeven), SHORT = short vol. Premio/payoff/PnL in BTC. status: settled (reale) · calib (bootstrap) · open.</div>
    </div>
    <div class="panel">
      <h3>Risk Profile <span id="payoff-title" style="font-weight:400;text-transform:none;letter-spacing:0"></span></h3>
      <div id="plot-payoff" class="plot tall"></div>
      <div class="legend">PnL (BTC) a scadenza vs prezzo del sottostante. ◆ = settlement reale. Linee: Strike (punteggiata) · Entry spot (ambra) · Breakeven (verde tratteggiata). Opzioni inverse Deribit: PnL = side·(|S−K|/S − premio) − fee.</div>
    </div>
  </div>
</div>

<script>
// ─── Plotly dark theme + helpers ──────────────────────────────────────────────
const PL_DARK = {
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#7d8aa0', size:10},
  margin:{l:48,r:12,t:8,b:36},
  xaxis:{gridcolor:'#1c2230', zerolinecolor:'#2a3344'},
  yaxis:{gridcolor:'#1c2230', zerolinecolor:'#2a3344'},
  showlegend:false,
};
const PL_CFG = {displayModeBar:false, responsive:true};
// IT: config 2D — niente box-zoom "finestra" (dragmode 'pan'), ma zoom assi con
//     rotellina sopra l'asse (Y su/giù, X sx/dx) + drag/pan; doppio-click reset.
// EN: 2D config — no box "window" zoom (dragmode 'pan'), but axis zoom via wheel
//     over the axis (Y up/down, X left/right) + drag/pan; double-click reset.
const PL_CFG_2D = {displayModeBar:false, responsive:true, scrollZoom:true, doubleClick:'reset'};
// IT: ── plot() — render guard UNICO (meccanismo olistico). Il bug "barre/curve
//     spariscono al cambio-tab" ha DUE cause distinte, entrambe coperte qui:
//       (a) container non ancora misurabile (tab display:none→block, reflow non
//           flushato) → la trace nasce a geometria 0/sub-pixel;
//       (b) DOPO un display:none→block, Plotly.react fa un update MINIMO e NON
//           ricostruisce i clip-path della cartesian layer (azzerati dal toggle di
//           visibilità) → le trace clippate (barre/linea payoff) spariscono mentre
//           shapes/annotation paper-ref (scala-indipendenti) restano. È esattamente
//           la firma osservata: px=0 sulle barre, asse-x senza tick, ma spot/max-pain ok.
//     Fix deterministico:
//       (1) size-guard via getBoundingClientRect: ritenta a rAF finché il box è
//           davvero DIPINTO (w,h ≥ 2px; cap 60 ≈1s) — niente render su geometria 0;
//       (2) sul RIENTRO tab (finestra <4s da switchTab) o su div mai disegnato →
//           Plotly.newPlot = rebuild COMPLETO con clip-path freschi (la react minimale
//           non li rifà; la width esplicita+autosize:false del tentativo precedente
//           anzi impediva pure a Plots.resize di ricostruirli). Sui refresh in-place
//           (stessa tab già visibile) resta Plotly.react → preserva zoom/pan.
// EN: ── plot() — SINGLE render guard (holistic). The "bars/curves vanish on tab switch"
//     bug has TWO distinct causes, both handled here:
//       (a) container not yet measurable (tab display:none→block, reflow not flushed) →
//           the trace is born at 0/sub-pixel geometry;
//       (b) AFTER a display:none→block, Plotly.react does a MINIMAL update and does NOT
//           rebuild the cartesian layer clip-paths (zeroed by the visibility toggle) →
//           clipped traces (bars/payoff line) vanish while paper-ref shapes/annotations
//           (scale-independent) remain. Exactly the observed signature: px=0 on bars,
//           x-axis without ticks, but spot/max-pain fine.
//     Deterministic fix: (1) size-guard via getBoundingClientRect: retry on rAF until
//     the box is actually PAINTED (w,h ≥ 2px; cap 60 ≈1s) — never render on 0 geometry;
//     (2) on tab RE-ENTRY (window <4s from switchTab) or a never-drawn div → Plotly.newPlot
//     = FULL rebuild with fresh clip-paths (the minimal react does not redo them; the
//     previous attempt's explicit width+autosize:false even stopped Plots.resize from
//     rebuilding them). On in-place refresh (same already-visible tab) keep Plotly.react
//     → preserves zoom/pan.
function plot(id, traces, layout, cfg, _t){
  const el = document.getElementById(id);
  if(!el) return;
  // IT: (1) size-guard schermo-reale: getBoundingClientRect riflette visibilità/transform.
  // EN: (1) real-screen size-guard: getBoundingClientRect reflects visibility/transform.
  const rc = el.getBoundingClientRect();
  if(rc.width < 2 || rc.height < 2){
    if((_t||0)<60) requestAnimationFrame(()=>plot(id,traces,layout,cfg,(_t||0)+1));
    return;
  }
  // IT: (2) rebuild completo nel rientro-tab (clip-path da rifare) o su div mai disegnato;
  //     altrimenti update in-place. Responsivo (no width esplicita) → newPlot/resize
  //     leggono il container vivo.
  // EN: (2) full rebuild on tab re-entry (clip-paths to redo) or a never-drawn div;
  //     otherwise in-place update. Responsive (no explicit width) → newPlot/resize read
  //     the live container.
  const fresh = (Date.now() - (window.__tabShownAt||0)) < 4000;
  if(fresh || !el.data){
    Plotly.newPlot(el, traces, layout, cfg);
  } else {
    Plotly.react(el, traces, layout, cfg);
  }
}
function fmt(v,d=2){ if(v==null||isNaN(v)) return '—'; return Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}); }
function fmt0(v){ return fmt(v,0); }
function fmtK(v){ if(v==null||isNaN(v)) return '—'; const a=Math.abs(v);
  if(a>=1e9) return (v/1e9).toFixed(1)+'B'; if(a>=1e6) return (v/1e6).toFixed(1)+'M';
  if(a>=1e3) return (v/1e3).toFixed(1)+'k'; return fmt(v,0); }
// IT: percentuale null-safe: '—' senza suffisso % (il backend manda null se il dato manca).
// EN: null-safe percentage: '—' without the % suffix (backend sends null when data is missing).
function fmtPct(v,d=1){ return (v==null||isNaN(v)) ? '—' : fmt(v,d)+'%'; }
function cls(v){ return v>=0?'pos':'neg'; }
// IT: formatter con SEGNO esplicito (+ per positivi; i negativi hanno già −) per le greche.
// EN: explicit-SIGN formatter (+ for positives; negatives already carry −) for the greeks.
function sgn(v){ return (v!=null && !isNaN(v) && v>0) ? '+' : ''; }
function fmtS(v,d=2){ if(v==null||isNaN(v)) return '—'; return sgn(v)+fmt(v,d); }
function fmtKS(v){ if(v==null||isNaN(v)) return '—'; return sgn(v)+fmtK(v); }
// IT: nomi estesi delle greche per header/etichette | EN: full greek names for headers/labels
const GK_NAME = {'Θ':'Theta','ν':'Vega','Γ':'Gamma','Δ':'Delta','ρ':'Rho'};

let SURFACE = null;

// ─── Tabs ─────────────────────────────────────────────────────────────────────
function switchTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.dataset.tab===name));
  document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active', p.id==='page-'+name));
  // IT: marca l'istante del cambio-tab: per i prossimi ~4s plot() farà un newPlot
  //     (rebuild dei clip-path corrotti dal display:none→block), poi torna a react.
  // EN: stamp the tab-switch instant: for the next ~4s plot() does a newPlot (rebuild the
  //     clip-paths corrupted by display:none→block), then reverts to react.
  window.__tabShownAt = Date.now();
  // IT: la pagina è ORA display:block; i loader sono async ma ogni render passa per
  //     plot() (size-guard + rebuild on re-entry) → niente resize manuale qui.
  // EN: the page is NOW display:block; loaders are async but every render goes through
  //     plot() (size-guard + rebuild on re-entry) → no manual resize here.
  if(name==='surface') loadSurface();
  else if(name==='chain')  loadChain();
  else if(name==='risk')   loadRisk();
  else if(name==='trades') loadTrades();
}

// ─── Header / summary ─────────────────────────────────────────────────────────
async function loadSummary(){
  try{
    const s = await (await fetch('/api/summary')).json();
    document.getElementById('h-spot').textContent = '$'+fmt0(s.spot);
    document.getElementById('h-dvol').textContent = fmtPct(s.dvol,1);
    document.getElementById('h-atm').textContent  = fmtPct(s.atm_iv_30d,1);
    document.getElementById('h-oi').textContent   = fmtK(s.total_oi);
    document.getElementById('h-vol').textContent  = fmtK(s.total_volume);
    document.getElementById('h-pcr').textContent  = fmt(s.put_call_ratio,2);
    setConn(true);
  }catch(e){ setConn(false); }
}
function setConn(ok){
  const c = document.getElementById('conn');
  c.className = ok?'ok':'err';
  document.getElementById('conn-txt').textContent = ok
    ? 'live · '+new Date().toLocaleTimeString('en-GB') : 'disconnected';
}

// ─── Volatility Surface ───────────────────────────────────────────────────────
async function loadSurface(){
  try{
    SURFACE = await (await fetch('/api/surface')).json();
    const sel = document.getElementById('smile-sel');
    const cur = sel.value;
    sel.innerHTML = SURFACE.smiles.map((s,i)=>`<option value="${i}">${s.label} · ${Math.round(s.days)}d</option>`).join('');
    if(cur && +cur < SURFACE.smiles.length) sel.value = cur;
    renderSurface(); renderSmile(); renderTerm();
  }catch(e){ setConn(false); }
}
function renderSurface(){
  if(!SURFACE || !SURFACE.z.length) return;
  plot('plot-surface', [{
    type:'surface', x:SURFACE.moneyness, y:SURFACE.days, z:SURFACE.z,
    colorscale:[[0,'#1a3a6b'],[0.4,'#4aa3ff'],[0.7,'#f0a020'],[1,'#ff5c5c']],
    colorbar:{title:{text:'IV %',font:{color:'#7d8aa0'}}, tickfont:{color:'#7d8aa0'}, thickness:10, len:.7},
    contours:{z:{show:true,usecolormap:true,project:{z:true}}},
    connectgaps:false, hovertemplate:'K/F %{x:.2f}<br>%{y:.0f}d<br>IV %{z:.1f}%<extra></extra>',
  }], Object.assign({}, PL_DARK, {
    margin:{l:0,r:0,t:0,b:0},
    scene:{
      xaxis:{title:'Moneyness K/F',color:'#7d8aa0',gridcolor:'#1c2230',backgroundcolor:'rgba(0,0,0,0)'},
      yaxis:{title:'Days',color:'#7d8aa0',gridcolor:'#1c2230',backgroundcolor:'rgba(0,0,0,0)'},
      zaxis:{title:'IV %',color:'#7d8aa0',gridcolor:'#1c2230',backgroundcolor:'rgba(0,0,0,0)'},
      camera:{eye:{x:1.7,y:-1.6,z:0.9}},
    }
  }), PL_CFG);
}
function renderSmile(){
  if(!SURFACE) return;
  const i = +document.getElementById('smile-sel').value || 0;
  const s = SURFACE.smiles[i]; if(!s) return;
  plot('plot-smile', [{
    type:'scatter', mode:'lines+markers', x:s.strikes, y:s.iv,
    line:{color:'#f0a020',width:2}, marker:{size:4,color:'#f0a020'},
    hovertemplate:'K %{x:,.0f}<br>IV %{y:.1f}%<extra></extra>',
  }], Object.assign({}, PL_DARK, {
    dragmode:'pan', autosize:true,
    xaxis:Object.assign({},PL_DARK.xaxis,{title:'Strike'}),
    yaxis:Object.assign({},PL_DARK.yaxis,{title:'IV %'}),
    shapes:[{type:'line',x0:s.forward,x1:s.forward,y0:0,y1:1,yref:'paper',
      line:{color:'#4aa3ff',width:1,dash:'dot'}}],
    annotations:[{x:s.forward,y:1,yref:'paper',text:'F',showarrow:false,font:{color:'#4aa3ff',size:10},yanchor:'bottom'}],
  }), PL_CFG_2D);
}
function renderTerm(){
  fetch('/api/term').then(r=>r.json()).then(t=>{
    const x = t.term.map(d=>d.days), y = t.term.map(d=>d.atm_iv), lbl = t.term.map(d=>d.label);
    plot('plot-term', [{
      type:'scatter', mode:'lines+markers', x, y, text:lbl,
      line:{color:'#b07cff',width:2}, marker:{size:5,color:'#b07cff'},
      hovertemplate:'%{text} · %{x:.0f}d<br>ATM IV %{y:.1f}%<extra></extra>',
    }], Object.assign({}, PL_DARK, {
      dragmode:'pan', autosize:true,
      xaxis:Object.assign({},PL_DARK.xaxis,{title:'Days to expiry',type:'log'}),
      yaxis:Object.assign({},PL_DARK.yaxis,{title:'ATM IV %'}),
    }), PL_CFG_2D);
  }).catch(()=>{});
}

// ─── Option Chain ─────────────────────────────────────────────────────────────
let CHAIN_EXPIRIES = [];
let CHAIN_EXP_AT = 0;
async function ensureExpiries(){
  // IT: TTL 10 min — le daily spirano alle 08:00 UTC e ne quotano di nuove: senza
  //     refresh il menu resta stale (expiry morte selezionabili, nuove assenti).
  //     La selezione corrente è preservata se l'expiry esiste ancora.
  // EN: 10-min TTL — dailies expire at 08:00 UTC and new ones get listed: without a
  //     refresh the menu goes stale (dead expiries selectable, new ones missing).
  //     Current selection is preserved if the expiry still exists.
  if(CHAIN_EXPIRIES.length && (Date.now()-CHAIN_EXP_AT) < 600e3) return;
  const m = await (await fetch('/api/expiries')).json();
  if(!m.expiries || !m.expiries.length) return;      // fetch degradata: tieni la lista corrente / degraded fetch: keep current list
  CHAIN_EXPIRIES = m.expiries; CHAIN_EXP_AT = Date.now();
  const sel = document.getElementById('chain-sel');
  const cur = sel.value;
  sel.innerHTML = CHAIN_EXPIRIES.map(e=>`<option value="${e.ts}">${e.label} · ${Math.round(e.days)}d</option>`).join('');
  if(cur && CHAIN_EXPIRIES.some(e=>String(e.ts)===cur)){ sel.value = cur; }
  else {
    let best=0,bd=1e9; CHAIN_EXPIRIES.forEach((e,i)=>{const d=Math.abs(e.days-30);if(d<bd){bd=d;best=i;}});
    sel.selectedIndex = best;
  }
}
const CH_COLS = ['oi','vol','Θ','ν','Γ','Δ','IV','bid','mark','ask'];
async function loadChain(){
  try{
    await ensureExpiries();
    const ts = document.getElementById('chain-sel').value;
    const d = await (await fetch('/api/chain?expiry='+ts)).json();
    document.getElementById('chain-fwd').textContent =
      `Forward ${fmt0(d.forward)}  ·  Spot ${fmt0(d.spot)}`;
    const head = document.getElementById('chain-head');
    head.innerHTML =
      CH_COLS.map(c=>`<th class="call-side"${GK_NAME[c]?` title="${GK_NAME[c]}"`:''}>${GK_NAME[c]?`${c} (${GK_NAME[c]})`:c}</th>`).join('') +
      `<th class="k-col">STRIKE</th>` +
      CH_COLS.slice().reverse().map(c=>`<th class="put-side"${GK_NAME[c]?` title="${GK_NAME[c]}"`:''}>${GK_NAME[c]?`${c} (${GK_NAME[c]})`:c}</th>`).join('');
    const atmK = nearest(d.rows.map(r=>r.strike), d.forward);
    const cell = (o,k,d2=2,sign=false)=>{
      if(!o||o[k]==null) return '<td>—</td>';
      let v=o[k], c = sign ? (v>=0?'pos':'neg') : '';
      return `<td class="${c}">${sign?fmtS(v,d2):fmt(v,d2)}</td>`;
    };
    document.getElementById('chain-body').innerHTML = d.rows.map(r=>{
      const c=r.call, p=r.put, atm = r.strike===atmK;
      const callCells = [
        cell(c,'oi',0), cell(c,'volume',0), cell(c,'theta',2,true),
        cell(c,'vega',2), cell(c,'gamma',5), cell(c,'delta',3,true),
        cell(c,'iv_pct',1), cell(c,'bid',4), cell(c,'mark',4), cell(c,'ask',4),
      ].join('');
      const putCells = [
        cell(p,'ask',4), cell(p,'mark',4), cell(p,'bid',4), cell(p,'iv_pct',1),
        cell(p,'delta',3,true), cell(p,'gamma',5), cell(p,'vega',2),
        cell(p,'theta',2,true), cell(p,'volume',0), cell(p,'oi',0),
      ].join('');
      return `<tr class="${atm?'atm-row':''}">${callCells}<td class="k-col">${fmt0(r.strike)}</td>${putCells}</tr>`;
    }).join('');
    setConn(true);
  }catch(e){ setConn(false); }
}
function nearest(arr,v){ let b=arr[0],bd=1e18; arr.forEach(x=>{const dd=Math.abs(x-v);if(dd<bd){bd=dd;b=x;}}); return b; }
// IT: posizione FRAZIONARIA di un valore su un asse CATEGORY (categorie = arr ordinato
//     ascendente). Plotly legge un x NUMERICO su asse category come indice 0-based
//     (frazionario ammesso) → piazza linee spot/strike PROPORZIONALI fra le categorie.
// EN: FRACTIONAL position of a value on a CATEGORY axis (categories = ascending arr).
//     Plotly reads a NUMERIC x on a category axis as a 0-based (fractional) index → places
//     spot/strike lines PROPORTIONALLY between categories.
function catPos(v, arr){
  const n=arr.length; if(!n) return 0;
  if(v<=arr[0]) return 0; if(v>=arr[n-1]) return n-1;
  for(let i=1;i<n;i++){ if(v<=arr[i]) return (i-1)+(v-arr[i-1])/((arr[i]-arr[i-1])||1); }
  return n-1;
}

// ─── Risk & Greeks ────────────────────────────────────────────────────────────
// IT: profilo OI per strike — barre DIVERGENTI (call ▲ verde sopra, put ▼ rosso
//     sotto), linea Net OI (call−put) su asse secondario per leggere lo skew del
//     book, zoom automatico sulla banda liquida, toggle contratti BTC / notional
//     USD (OI×strike). Linee spot e max-pain. Hover sempre in valori assoluti.
// EN: per-strike OI profile — DIVERGING bars (calls ▲ green up, puts ▼ red down),
//     Net OI (call−put) line on a secondary axis to read book skew, auto-zoom on
//     the liquid band, BTC-contracts / USD-notional toggle (OI×strike). Spot and
//     max-pain reference lines. Hover always shows absolute values.
function renderOI(d){
  const usd = document.getElementById('oi-metric').value === 'usd';
  const unit = usd ? 'USD' : 'BTC';
  // IT: FILTRA i dati alla banda spot±35% PRIMA di plottare (non solo la vista): X e Y
  //     si scalano sugli STESSI strike → niente barra fuori-vista (balena OTM con OI
  //     enorme) che schiaccia l'asse Y e fa "sparire" le near-money. Confermato in
  //     browser: 24.5k OI @ K=80000 fuori vista schiacciava le near-money a ~1px.
  // EN: FILTER data to the spot±35% band BEFORE plotting (not just the view): X and Y
  //     scale on the SAME strikes → no off-view bar (OTM whale, huge OI) squishing the
  //     Y axis and making the near-money "vanish". Confirmed in browser: 24.5k OI @
  //     K=80000 off-view squished the near-money to ~1px.
  const cap = (d.spot>0 ? d.spot : Math.max(...d.strikes)) * 0.35;
  let idx = []; d.strikes.forEach((k,i)=>{ if(Math.abs(k-d.spot)<=cap) idx.push(i); });
  if(!idx.length) idx = d.strikes.map((_,i)=>i);                // fallback: tutti / all
  // IT: ordina la banda per strike ASCENDENTE → le categorie dell'asse-x seguono l'ordine
  //     di strike (un asse category ordina per prima apparizione nei dati).
  // EN: sort the band by ASCENDING strike → x-axis categories follow strike order (a
  //     category axis orders by first appearance in the data).
  idx.sort((a,b)=>d.strikes[a]-d.strikes[b]);
  const ks    = idx.map(i=>d.strikes[i]);
  const ksStr = ks.map(k=>String(k));                          // categorie (label = strike) / categories
  const callV = idx.map(i=> usd ? d.call_oi[i]*d.strikes[i] : d.call_oi[i]);
  const putV  = idx.map(i=> usd ? d.put_oi[i]*d.strikes[i]  : d.put_oi[i]);
  const netV  = callV.map((v,j)=>v - putV[j]);                  // skew call−put (banda)

  // IT: ── ASSE-X CATEGORY (non lineare) = IL FIX del bug storico "barre spariscono / si
  //     schiacciano al cambio-tab". Causa reale (diagnosi browser 2026-06-24): su un
  //     re-render dopo display:none→block Plotly CORROMPE la mappatura-pixel dell'asse
  //     LINEARE numerico → le trace finiscono fuori campo (x≈−1244px) o tutta la banda
  //     compressa in ~19px, mentre shapes/annotation paper-ref restano (firma del bug).
  //     NESSUN rimedio via API lo recupera (newPlot, purge, replace-node, Plots.resize,
  //     relayout, autorange, scala-x: tutti KO, verificati uno per uno). L'asse CATEGORY
  //     posiziona per INDICE (non per scala numerica) ed è IMMUNE — coerente col fatto che
  //     plot-greeks (già category) non ha MAI avuto il bug. Spot/Max-Pain a INDICE
  //     FRAZIONARIO (catPos) per restare proporzionali; gli strike di banda sono
  //     ~equispaziati → la vista category ≈ la lineare. Niente width barra esplicita:
  //     su asse category la larghezza è in slot-categoria, auto e deterministica.
  // EN: ── CATEGORY x-axis (not linear) = THE fix for the long-standing "bars vanish /
  //     squash on tab switch" bug. Real cause (browser diagnosis 2026-06-24): on a
  //     re-render after display:none→block Plotly CORRUPTS the LINEAR numeric axis pixel
  //     mapping → traces land off-view (x≈−1244px) or the whole band is compressed into
  //     ~19px, while paper-ref shapes/annotations remain (the bug's signature). NO API
  //     remedy recovers it (newPlot, purge, replace-node, Plots.resize, relayout,
  //     autorange, x-scaling: all failed, each verified). A CATEGORY axis positions by
  //     INDEX (not numeric scale) and is IMMUNE — consistent with plot-greeks (already
  //     category) never having the bug. Spot/Max-Pain at FRACTIONAL index (catPos) to stay
  //     proportional; band strikes are ~evenly spaced → the category view ≈ the linear one.
  //     No explicit bar width: on a category axis width is in category-slots, auto/deterministic.
  const _spotX = catPos(d.spot, ks), _mpX = catPos(d.max_pain, ks);
  plot('plot-oi', [
    {type:'bar', name:'Call OI', x:ksStr, y:callV, marker:{color:'#2ecc7199'},
     customdata:callV, hovertemplate:`K %{x}<br>Call OI %{customdata:,.0f} ${unit}<extra></extra>`},
    {type:'bar', name:'Put OI', x:ksStr, y:putV.map(v=>-v), marker:{color:'#ff5c5c99'},
     customdata:putV, hovertemplate:`K %{x}<br>Put OI %{customdata:,.0f} ${unit}<extra></extra>`},
    // IT: Net OI = call−put → stessa unità/scala delle barre → asse PRIMARIO (no yaxis2).
    // EN: Net OI = call−put → same unit/scale as the bars → PRIMARY axis (no yaxis2).
    {type:'scatter', name:'Net OI', mode:'lines', x:ksStr, y:netV,
     line:{color:'#f0a020',width:1.6}, hovertemplate:`K %{x}<br>Net %{y:,.0f} ${unit}<extra></extra>`},
  ], Object.assign({}, PL_DARK, {
    dragmode:'pan',
    barmode:'relative', showlegend:true,
    legend:{font:{color:'#7d8aa0'},orientation:'h',y:1.08,x:0},
    hovermode:'x unified',
    // IT: asse category + tick diradati (~1 ogni n/9) per non sovrapporre 40+ strike.
    // EN: category axis + thinned ticks (~1 every n/9) to avoid overlapping 40+ strikes.
    xaxis:Object.assign({},PL_DARK.xaxis,{title:'Strike',type:'category',
      tickmode:'linear',tick0:0,dtick:Math.max(1,Math.round(ks.length/9))}),
    yaxis:Object.assign({},PL_DARK.yaxis,{title:`Puts ▼ / Calls ▲ / Net (${unit})`,autorange:true,zeroline:true,zerolinecolor:'#3a465c'}),
    shapes:[
      {type:'line',x0:_spotX,x1:_spotX,y0:0,y1:1,yref:'paper',line:{color:'#4aa3ff',width:1.5}},
      {type:'line',x0:_mpX,x1:_mpX,y0:0,y1:1,yref:'paper',line:{color:'#f0a020',width:1.5,dash:'dash'}},
    ],
    annotations:[
      {x:_spotX,y:1,yref:'paper',text:'Spot',showarrow:false,font:{color:'#4aa3ff',size:10},yanchor:'bottom'},
      {x:_mpX,y:1,yref:'paper',text:'Max Pain',showarrow:false,font:{color:'#f0a020',size:10},yanchor:'bottom',xanchor:'right'},
    ],
  }), PL_CFG_2D);
}

async function loadRisk(){
  try{
    const d = await (await fetch('/api/risk')).json();
    // IT: GUARD anti-race async↔tab. loadRisk è async: tra l'inizio della fetch e il suo
    //     ritorno l'utente può aver cambiato tab. Se page-risk NON è più attiva, il
    //     render atterrerebbe su un container nascosto (0×0 / reflow) → barre disegnate
    //     con geometria sbagliata = il bug "barre sparite al cambio-tab". Skip: il
    //     prossimo switch-to-risk rifà fetch+render a container visibile.
    // EN: async↔tab race GUARD. loadRisk is async: between fetch start and resolve the
    //     user may have switched tabs. If page-risk is no longer active, the render would
    //     land on a hidden container (0×0 / reflow) → bars drawn with wrong geometry =
    //     the "bars vanish on tab switch" bug. Skip: the next switch-to-risk re-fetches
    //     and re-renders into a visible container.
    if(!document.getElementById('page-risk').classList.contains('active')) return;
    // IT: guard dati — se la risposta è degradata (chain parziale/desincronizzata:
    //     niente strike o OI tutto a zero) NON ridisegnare: tieni l'ultimo grafico
    //     buono invece di svuotare le barre. Difesa frontend (oltre al fix backend).
    // EN: data guard — if the response is degraded (partial/desynced chain: no
    //     strikes or all-zero OI) do NOT redraw: keep the last good chart instead of
    //     blanking the bars. Frontend defense (on top of the backend fix).
    if(d && d.error){ setConn(false); return; }
    const _noStrk = !d || !Array.isArray(d.strikes) || d.strikes.length < 3;
    const _noOI = !_noStrk && (d.call_oi||[]).every(v=>!v) && (d.put_oi||[]).every(v=>!v);
    if(_noStrk || _noOI){ console.warn('risk: dati parziali/desincronizzati → skip redraw'); return; }
    const g = d.agg_greeks;
    document.getElementById('risk-cards').innerHTML = [
      {l:'Max Pain', v:'$'+fmt0(d.max_pain), s:'min total holder payoff', c:'amb'},
      {l:'Spot vs Max Pain', v:fmt(((d.spot-d.max_pain)/d.max_pain*100),1)+'%', s:'spot $'+fmt0(d.spot), c:(d.spot-d.max_pain)>=0?'pos':'neg'},
      {l:'Net OI Δ (Delta)', v:fmtKS(g.delta), s:'OI-weighted · BTC', c:cls(g.delta)},
      {l:'Net OI Γ (Gamma)', v:fmtS(g.gamma,4), s:'OI-weighted', c:cls(g.gamma)},
      {l:'Net OI ν (Vega)', v:fmtKS(g.vega), s:'per +1% vol', c:cls(g.vega)},
      {l:'Net OI Θ (Theta)', v:fmtKS(g.theta), s:'OI-weighted · USD/day', c:cls(g.theta)},
      {l:'DVOL', v:fmtPct(d.dvol,1), s:'30d implied', c:'amb'},
    ].map(c=>`<div class="card"><div class="lbl">${c.l}</div><div class="val ${c.c||''}">${c.v}</div><div class="sub">${c.s}</div></div>`).join('');

    renderOI(d);
    const gv=['delta','gamma','vega','theta'].map(k=>g[k]);
    plot('plot-greeks', [{type:'bar', x:['Δ (Delta)','Γ (Gamma)','ν (Vega)','Θ (Theta)'], y:gv,
      marker:{color:gv.map(v=>v>=0?'#2ecc71':'#ff5c5c')},
      hovertemplate:'%{x}: %{y:+,.2f}<extra></extra>'}],
      Object.assign({}, PL_DARK, {dragmode:'pan', yaxis:Object.assign({},PL_DARK.yaxis,{title:'OI-weighted'})}), PL_CFG_2D);

    const totC = d.call_oi.reduce((a,b)=>a+b,0), totP = d.put_oi.reduce((a,b)=>a+b,0);
    plot('plot-pcr', [{type:'pie', labels:['Calls','Puts'], values:[totC,totP],
      marker:{colors:['#2ecc71','#ff5c5c']}, hole:.55, textinfo:'label+percent',
      textfont:{color:'#dfe6f0'}, hovertemplate:'%{label}: %{value:,.0f} BTC<extra></extra>'}],
      Object.assign({}, PL_DARK, {dragmode:'pan', margin:{l:8,r:8,t:8,b:8}}), PL_CFG_2D);
    setConn(true);
  }catch(e){ setConn(false); }
}

// ─── Trades (forward test vol-paper) ────────────────────────────────────────────
let TRADES = [];
let SEL_TS = null;   // IT: entry_ts del trade selezionato (sopravvive al refresh 12s) | EN: selected trade entry_ts (survives the 12s refresh)
async function loadTrades(){
  try{
    const d = await (await fetch('/api/trades')).json();
    TRADES = d.trades||[];
    const s = d.summary||{};
    document.getElementById('trades-cards').innerHTML = [
      {l:'Total PnL (BTC)', v:fmtS(s.total_pnl,4), s:`${s.n_settled||0} settled / gate ${s.gate_trades||30}`, c:cls(s.total_pnl||0)},
      {l:'Hit-rate', v:(s.hit_rate==null?'—':fmt(s.hit_rate*100,0)+'%'), s:'PnL > 0', c:(s.hit_rate==null?'amb':(s.hit_rate>=0.5?'pos':'neg'))},
      {l:'Avg PnL/trade', v:(s.avg_pnl==null?'—':fmtS(s.avg_pnl,4)), s:'BTC', c:cls(s.avg_pnl||0)},
      {l:'Trades', v:fmt0(s.n||0), s:`${s.n_executed||0} eseguiti (real) · ${s.n_open||0} open`, c:'amb'},
      {l:'Best / Worst', v:(s.best==null?'—':fmtS(s.best,4))+' / '+(s.worst==null?'—':fmtS(s.worst,4)), s:'BTC', c:'amb'},
    ].map(c=>`<div class="card"><div class="lbl">${c.l}</div><div class="val ${c.c||''}">${c.v}</div><div class="sub">${c.s}</div></div>`).join('');

    const cols=['#','Entry','Side','Strike','Entry spot','Premium','Settle','Payoff','PnL','Edge','Status'];
    document.getElementById('trades-head').innerHTML = cols.map(c=>`<th>${c}</th>`).join('');
    document.getElementById('trades-body').innerHTML = TRADES.map((t,i)=>{
      const stat = !t.settled ? 'open' : (t.executed?'settled':'calib');
      return `<tr onclick="selectTrade(${i})" style="cursor:pointer">
        <td>${i+1}</td><td>${(t.entry_ts||'').slice(5,16)}</td>
        <td class="${t.side>0?'pos':'neg'}">${t.side>0?'LONG':'SHORT'}</td>
        <td>${fmt0(t.strike)}</td><td>${fmt0(t.entry_spot)}</td>
        <td>${fmt(t.premium,4)}</td>
        <td>${t.delivery_price==null?'—':fmt0(t.delivery_price)}</td>
        <td>${t.payoff_btc==null?'—':fmt(t.payoff_btc,4)}</td>
        <td class="${cls(t.pnl_btc||0)}">${t.pnl_btc==null?'—':fmtS(t.pnl_btc,4)}</td>
        <td>${fmt(t.edge,2)}</td><td>${stat}</td></tr>`;
    }).join('');
    // IT: ripristina la selezione dell'utente per entry_ts (il refresh 12s ricostruisce
    //     la tabella: senza questo, il click veniva sovrascritto tornando all'ultimo).
    // EN: restore the user's selection by entry_ts (the 12s refresh rebuilds the table:
    //     without this, the click was overwritten back to the last trade).
    if(TRADES.length){
      let i = TRADES.findIndex(t=>t.entry_ts===SEL_TS);
      selectTrade(i>=0 ? i : TRADES.length-1);
    }
    else { document.getElementById('payoff-title').textContent='(nessun trade)'; Plotly.purge('plot-payoff'); }
    setConn(true);
  }catch(e){ setConn(false); }
}
function selectTrade(i){
  const t=TRADES[i]; if(!t) return;
  SEL_TS = t.entry_ts;
  document.querySelectorAll('#trades-body tr').forEach((r,j)=>r.classList.toggle('atm-row', j===i));
  renderPayoff(t);
}
// IT: profilo di rischio dello straddle — PnL (BTC) a scadenza vs sottostante,
//     con la STESSA formula di settlement di 04b (maybe_settle):
//       pnl(S) = side·amt·(|S−K|/S − premium) − fee    (opzioni inverse Deribit)
//     → il marker ◆ del settlement cade ESATTAMENTE sulla curva (niente calibrazione
//     a posteriori: la vecchia `cost = pf ∓ pnl` sbagliava il segno della fee sugli
//     SHORT e divideva per delivery_price=0 sui trade aperti). Breakeven espliciti:
//     |S−K|/S = premium + side·fee/amt → S± = K/(1∓m*).
// EN: straddle risk profile — PnL (BTC) at expiry vs underlying, using the SAME
//     settlement formula as 04b (maybe_settle):
//       pnl(S) = side·amt·(|S−K|/S − premium) − fee    (inverse Deribit options)
//     → the settlement ◆ marker lands EXACTLY on the curve (no ex-post calibration:
//     the old `cost = pf ∓ pnl` got the fee sign wrong on SHORTs and divided by
//     delivery_price=0 on open trades). Explicit breakevens:
//     |S−K|/S = premium + side·fee/amt → S± = K/(1∓m*).
function renderPayoff(t){
  const K=t.strike, amt=t.amount||1, side=t.side>0?1:-1;
  const prem=t.premium||0, fee=t.fee_btc||0;
  const pnlAt = S => side*amt*(Math.abs(S-K)/S - prem) - fee;
  // IT: finestra K±20%, ESTESA a includere il settlement se cade fuori (audit MINOR-5:
  //     prima un delivery off-range veniva snappato al bordo con ascissa fuorviante).
  // EN: K±20% window, EXTENDED to include the settlement when it falls outside (audit
  //     MINOR-5: an off-range delivery used to snap to the edge with a misleading x).
  const dp = (t.delivery_price!=null && t.delivery_price>0) ? t.delivery_price : null;
  let lo=K*0.80, hi=K*1.20;
  if(dp!=null){ lo=Math.min(lo, dp*0.97); hi=Math.max(hi, dp*1.03); }
  const N=121, xs=[], ys=[];
  for(let j=0;j<N;j++){ const S=lo+(hi-lo)*j/(N-1); xs.push(S); ys.push(pnlAt(S)); }
  // IT: breakeven m* = |S−K|/S a PnL=0; S⁺=K/(1−m*) (sopra strike), S⁻=K/(1+m*).
  // EN: breakeven m* = |S−K|/S at PnL=0; S⁺=K/(1−m*) (above strike), S⁻=K/(1+m*).
  const mStar = prem + side*fee/amt;
  const beUp = (mStar>0 && mStar<1) ? K/(1-mStar) : null;
  const beDn = (mStar>0) ? K/(1+mStar) : null;
  // IT: ── ASSE-X CATEGORY anche qui (STESSO fix di renderOI). Causa (diagnosi browser
  //     2026-06-24): l'asse LINEARE numerico, ri-renderizzato dopo display:none→block,
  //     CORROMPE la mappatura-pixel e schiaccia la curva in una verticale a sinistra
  //     (≈1px), restano solo gli shapes = le 2 verticali strike/entry. xs è una griglia
  //     REGOLARE (linspace) → come categorie resta equispaziata e la V del payoff è
  //     IDENTICA alla vista lineare. Strike/Entry a indice FRAZIONARIO (catPos); il marker
  //     settlement è agganciato alla categoria più vicina (curva fitta, errore di snap
  //     ~(hi−lo)/120 trascurabile) così resta SULLA curva. L'asse-Y resta numerico in
  //     autorange (il bug è solo dell'asse ORIZZONTALE; la Y ha sempre reso bene).
  // EN: ── CATEGORY x-axis here too (SAME fix as renderOI). Cause (browser diagnosis
  //     2026-06-24): the LINEAR numeric axis, re-rendered after display:none→block,
  //     CORRUPTS the pixel mapping and squashes the curve into a left vertical line
  //     (≈1px), leaving only the shapes = the 2 strike/entry verticals. xs is a REGULAR
  //     grid (linspace) → as categories it stays evenly spaced and the payoff V is
  //     IDENTICAL to the linear view. Strike/Entry at FRACTIONAL index (catPos); the
  //     settlement marker snaps to the nearest category (dense curve, snap error
  //     ~(hi−lo)/120 negligible) so it stays ON the curve. The Y axis stays numeric
  //     autorange (the bug is HORIZONTAL-axis only; Y has always rendered fine).
  const xsStr = xs.map(s=>String(Math.round(s)));
  const _idxNear = (v)=>{ let bi=0,bd=1e18; for(let j=0;j<xs.length;j++){const dd=Math.abs(xs[j]-v); if(dd<bd){bd=dd;bi=j;}} return bi; };
  const _kX = catPos(K, xs), _eX = catPos(t.entry_spot, xs);
  const traces=[{type:'scatter',mode:'lines',x:xsStr,y:ys,line:{color:'#4aa3ff',width:2},
    hovertemplate:'S %{x}<br>PnL %{y:+,.4f} BTC<extra></extra>'}];
  // IT: marker settlement solo su trade chiusi con delivery valido (>0: da flat il
  //     backend manda null, mai 0 fantasma — doppia difesa qui).
  // EN: settlement marker only on closed trades with a valid delivery (>0: the
  //     backend sends null when open, never a phantom 0 — double defense here).
  if(t.delivery_price!=null && t.delivery_price>0 && t.pnl_btc!=null){
    traces.push({type:'scatter',mode:'markers',x:[xsStr[_idxNear(t.delivery_price)]],y:[t.pnl_btc],
      marker:{color:(t.pnl_btc>=0?'#2ecc71':'#ff5c5c'),size:12,symbol:'diamond'},
      hovertemplate:'settle %{x}<br>PnL %{y:+,.4f} BTC<extra></extra>'});
  }
  // IT: debit (long: premio+fee) / credit (short: premio−fee) + breakeven nel titolo.
  // EN: debit (long: premium+fee) / credit (short: premium−fee) + breakevens in title.
  const netCash = prem*amt + side*fee;
  document.getElementById('payoff-title').textContent =
    `${t.side>0?'LONG':'SHORT'} straddle · K ${fmt0(K)} · ${t.side>0?'debit':'credit'} ${fmt(netCash,4)} BTC`
    + (beDn&&beUp ? ` · BE ${fmtK(beDn)} / ${fmtK(beUp)}` : '')
    + (t.settled ? '' : ' · OPEN');
  // IT: render via plot() (size-guard + rebuild on re-entry); asse-x category (vedi sopra).
  // EN: render via plot() (size-guard + rebuild on re-entry); category x-axis (see above).
  plot('plot-payoff', traces, Object.assign({}, PL_DARK, {
    dragmode:'pan', showlegend:false,
    xaxis:Object.assign({},PL_DARK.xaxis,{title:'Underlying at expiry (USD)',type:'category',
      tickmode:'linear',tick0:0,dtick:Math.max(1,Math.round(N/7))}),
    yaxis:Object.assign({},PL_DARK.yaxis,{title:'PnL (BTC)',autorange:true,zeroline:true,zerolinecolor:'#3a465c'}),
    shapes:[
      {type:'line',x0:_kX,x1:_kX,y0:0,y1:1,yref:'paper',line:{color:'#7d8aa0',width:1,dash:'dot'}},
      {type:'line',x0:_eX,x1:_eX,y0:0,y1:1,yref:'paper',line:{color:'#f0a020',width:1.2}},
      // IT: breakeven (PnL=0) tratteggiati, solo se dentro la finestra ±20%.
      // EN: dashed breakevens (PnL=0), only when inside the ±20% window.
      ...[beDn,beUp].filter(b=>b!=null && b>=lo && b<=hi).map(b=>{
        const x=catPos(b,xs);
        return {type:'line',x0:x,x1:x,y0:0,y1:1,yref:'paper',line:{color:'#2ecc71',width:1,dash:'dash'}};
      }),
    ],
    // IT: niente label 'BE' (collidono coi tick sotto e con la V dentro): i valori
    //     sono nel titolo, la legenda spiega il verde tratteggiato.
    // EN: no 'BE' labels (they collide with ticks below and the V inside): values
    //     are in the title, the legend explains the dashed green.
    annotations:[
      {x:_kX,y:1,yref:'paper',text:'Strike',showarrow:false,font:{color:'#7d8aa0',size:10},yanchor:'bottom'},
      {x:_eX,y:1,yref:'paper',text:'Entry',showarrow:false,font:{color:'#f0a020',size:10},yanchor:'bottom',xanchor:'right'},
    ],
  }), PL_CFG_2D);
}

// ─── Refresh loop ─────────────────────────────────────────────────────────────
function refresh(){
  loadSummary();
  if(document.getElementById('page-surface').classList.contains('active')) loadSurface();
  if(document.getElementById('page-chain').classList.contains('active'))   loadChain();
  if(document.getElementById('page-risk').classList.contains('active'))    loadRisk();
  if(document.getElementById('page-trades').classList.contains('active'))  loadTrades();
}
loadSummary(); loadSurface();
setInterval(refresh, 12000);
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# IT: 5) SERVER HTTP — routing JSON, gzip, auth opzionale (pattern dashboard repo).
# EN: 5) HTTP SERVER — JSON routing, gzip, optional auth (repo dashboard pattern).
# ═══════════════════════════════════════════════════════════════════════════════
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # IT: silenzia il logging per-request | EN: silence per-request logging
        pass

    # IT: auth a token constant-time (se configurato in config.dashboard.auth_token).
    # EN: constant-time token auth (if set in config.dashboard.auth_token).
    def _authorized(self) -> bool:
        if not AUTH_TOKEN:
            return True
        qs = parse_qs(urlparse(self.path).query)
        tok = (qs.get("token", [""])[0]
               or self.headers.get("X-Auth-Token", ""))
        return hmac.compare_digest(tok, AUTH_TOKEN)

    def _send(self, code: int, body: bytes, ctype: str):
        accept = self.headers.get("Accept-Encoding", "") or ""
        if ENABLE_GZIP and "gzip" in accept.lower() and len(body) > 512:
            body = _gzip.compress(body)
            enc = True
        else:
            enc = False
        # IT: il client (browser) può chiudere la connessione a metà risposta quando
        #     il refresh ~12s annulla i fetch ancora in volo → ConnectionAborted/Reset/
        #     BrokenPipe. NON è un errore del server: ignora silenziosamente (prima
        #     crashava e ri-crashava provando a scrivere il 500).
        # EN: the client (browser) may drop the connection mid-response when the ~12s
        #     refresh cancels in-flight fetches → ConnectionAborted/Reset/BrokenPipe.
        #     NOT a server error: swallow it silently (it used to crash, then crash
        #     again trying to write the 500).
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            if enc:
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Vary", "Accept-Encoding")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        if not self._authorized():
            self._json({"error": "unauthorized"}, 401)
            return
        parsed = urlparse(self.path)
        route = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if route == "/api/summary":
                self._json(build_summary(build_market()))
                return
            if route == "/api/surface":
                self._json(build_surface(build_market()))
                return
            if route == "/api/term":
                self._json(build_term_structure(build_market()))
                return
            if route == "/api/expiries":
                m = build_market()
                self._json({"expiries": m["expiries"], "spot": m["spot"]})
                return
            if route == "/api/chain":
                ts = qs.get("expiry", [None])[0]
                ts = float(ts) if ts not in (None, "") else None
                self._json(build_chain_table(build_market(), ts))
                return
            if route == "/api/risk":
                self._json(build_risk(build_market()))
                return
            if route == "/api/trades":
                self._json(build_trades())
                return
            self._json({"error": "not found"}, 404)
        except requests.RequestException as e:
            log.warning(f"Deribit fetch error on {route}: {e}")
            self._json({"error": f"deribit upstream: {e}"}, 502)
        except Exception as e:
            log.exception(f"handler error on {route}")
            self._json({"error": str(e)}, 500)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script, CLAUDE.md — bug cp1252 ricorrente).
    # EN: UTF-8 boilerplate (new-script checklist, CLAUDE.md — recurring cp1252 bug).
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    _shown = "localhost" if HOST in ("127.0.0.1", "localhost") else HOST
    log.info("=" * 70)
    log.info("  QUANTSYS · Deribit BTC Options Risk Terminal")
    log.info(f"  → http://{_shown}:{PORT}   (bind={HOST})")
    log.info(f"  Data: Deribit public REST (no-auth) · currency={CURRENCY} · "
             f"auth={'ON' if AUTH_TOKEN else 'OFF'} · gzip={'on' if ENABLE_GZIP else 'off'}")
    log.info("  Ctrl+C per uscire / to quit.")
    log.info("=" * 70)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Dashboard fermata / stopped (Ctrl+C).")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
