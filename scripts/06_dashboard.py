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
    NON usa i modelli ML del progetto: è un risk terminal di mercato, GPU-free.
EN: Institutional crypto-options analytics terminal. Connects to Deribit public
    data (REST, no-auth) for BTC spot + full option chain (mark/bid/ask, mark_iv,
    open interest, volume, per-expiry forward), computes Greeks in real time
    (Black-Scholes forward-measure) over the whole chain, and renders the
    Volatility Surface (3D), per-expiry smiles, the ATM term structure and the
    risk distribution (OI / aggregate Greeks). Does NOT touch the project's ML
    models: it is a market risk terminal, GPU-free.
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
        return _deribit_get("public/get_book_summary_by_currency",
                            {"currency": currency, "kind": "option"})
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

    return {
        "spot": _safe(market["spot"]),
        "dvol": _safe(fetch_dvol(market.get("currency", CURRENCY))),
        "atm_iv_30d": _safe(atm30),
        "total_oi": _safe(total_oi),
        "total_volume": _safe(total_vol),
        "call_oi": _safe(call_oi),
        "put_oi": _safe(put_oi),
        "put_call_ratio": _safe(pcr),
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
    return {"term": out, "dvol": _safe(fetch_dvol(market.get("currency", CURRENCY)))}


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
        ks = np.array(strikes)
        call_arr = np.array([call_oi[k] for k in strikes])
        put_arr = np.array([put_oi[k] for k in strikes])
        # IT: per ogni prezzo candidato p (riga): payoff call = (p-K)+ ·OI_call,
        #     payoff put = (K-p)+ ·OI_put, sommati su tutti gli strike (colonna).
        # EN: for each candidate price p (row): call payoff = (p-K)+ ·OI_call,
        #     put payoff = (K-p)+ ·OI_put, summed over all strikes (column).
        diff = ks[:, None] - ks[None, :]
        pain = (np.maximum(diff, 0.0) * call_arr[None, :]).sum(1) \
            + (np.maximum(-diff, 0.0) * put_arr[None, :]).sum(1)
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
        "dvol": _safe(fetch_dvol(market.get("currency", CURRENCY))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# IT: 3b) FORWARD TEST — trade dello straddle vol (04b_vol_paper.py) da trades.jsonl.
#     Per ogni trade: lato (LONG/SHORT straddle = long/short vol), strike, spot di
#     ingresso, premio, prezzo di settlement, payoff e PnL (BTC) + sintesi aggregata.
# EN: 3b) FORWARD TEST — vol straddle trades (04b_vol_paper.py) from trades.jsonl.
#     Per trade: side (LONG/SHORT straddle = long/short vol), strike, entry spot,
#     premium, settlement price, payoff and PnL (BTC) + aggregated summary.
# ═══════════════════════════════════════════════════════════════════════════════
TRADES_PATH = Path("results/vol_paper/trades.jsonl")


def build_trades() -> dict:
    if not TRADES_PATH.exists():
        return {"trades": [], "summary": {"n": 0, "n_settled": 0, "gate_trades": 30,
                                          "note": "nessun trade (forward test 04b non avviato)"}}
    rows = []
    for line in TRADES_PATH.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        try:
            t = json.loads(line)
        except Exception:
            continue
        prem = float(t.get("prem_call", 0) or 0) + float(t.get("prem_put", 0) or 0)
        rows.append({
            "entry_ts": t.get("entry_ts"), "settled_ts": t.get("settled_ts"),
            "side": int(t.get("side", 1)),                 # IT/EN: 1 LONG straddle, -1 SHORT
            "executed": bool(t.get("executed", False)),
            "settled": t.get("settled_ts") is not None,
            "strike": _safe(t.get("strike")),
            "entry_spot": _safe(t.get("index_at_entry")),
            "delivery_price": _safe(t.get("delivery_price")),
            "prem_call": _safe(t.get("prem_call")), "prem_put": _safe(t.get("prem_put")),
            "premium": _safe(prem), "fee_btc": _safe(t.get("fee_btc")),
            "amount": _safe(t.get("amount", 1.0)),
            "payoff_btc": _safe(t.get("payoff_btc")), "pnl_btc": _safe(t.get("pnl_btc")),
            "edge": _safe(t.get("edge")), "rv_pred": _safe(t.get("rv_pred")),
            "var_iv": _safe(t.get("var_iv")), "t_hours": _safe(t.get("t_hours_at_entry")),
            "call": t.get("call"), "put": t.get("put"),
        })
    settled = [r for r in rows if r["settled"] and r["pnl_btc"] is not None]
    pnls = [r["pnl_btc"] for r in settled]
    n_s = len(settled)
    wins = sum(1 for p in pnls if p > 0)
    summary = {
        "n": len(rows), "n_settled": n_s,
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
  body{background:var(--bg);color:var(--text);font:13px/1.4 'Segoe UI',system-ui,sans-serif;}
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
      <div class="legend">PnL (BTC) a scadenza vs prezzo del sottostante. ◆ = settlement reale. Linee: Strike (punteggiata) · Entry spot (ambra). Opzioni inverse Deribit: payoff = |S−K|/S.</div>
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
function fmt(v,d=2){ if(v==null||isNaN(v)) return '—'; return Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}); }
function fmt0(v){ return fmt(v,0); }
function fmtK(v){ if(v==null||isNaN(v)) return '—'; if(Math.abs(v)>=1e3) return (v/1e3).toFixed(1)+'k'; return fmt(v,0); }
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
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if(name==='surface'){ loadSurface(); }
  if(name==='chain')  loadChain();
  if(name==='risk')   loadRisk();
  if(name==='trades') loadTrades();
}

// ─── Header / summary ─────────────────────────────────────────────────────────
async function loadSummary(){
  try{
    const s = await (await fetch('/api/summary')).json();
    document.getElementById('h-spot').textContent = '$'+fmt0(s.spot);
    document.getElementById('h-dvol').textContent = fmt(s.dvol,1)+'%';
    document.getElementById('h-atm').textContent  = fmt(s.atm_iv_30d,1)+'%';
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
  Plotly.react('plot-surface', [{
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
  Plotly.react('plot-smile', [{
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
    Plotly.react('plot-term', [{
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
async function ensureExpiries(){
  if(CHAIN_EXPIRIES.length) return;
  const m = await (await fetch('/api/expiries')).json();
  CHAIN_EXPIRIES = m.expiries;
  const sel = document.getElementById('chain-sel');
  sel.innerHTML = CHAIN_EXPIRIES.map(e=>`<option value="${e.ts}">${e.label} · ${Math.round(e.days)}d</option>`).join('');
  let best=0,bd=1e9; CHAIN_EXPIRIES.forEach((e,i)=>{const d=Math.abs(e.days-30);if(d<bd){bd=d;best=i;}});
  sel.selectedIndex = best;
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
  const conv = (v,i) => usd ? v * d.strikes[i] : v;             // notional o contratti
  const callV = d.call_oi.map((v,i)=>conv(v,i));
  const putV  = d.put_oi.map((v,i)=>conv(v,i));
  const netV  = callV.map((v,i)=>v - putV[i]);                  // skew call−put

  // IT: larghezza barra ESPLICITA (≈85% del gap mediano tra strike). Senza, Plotly
  //     la deriva dal gap MINIMO: se a un refresh la chain porta due strike vicini il
  //     min-gap crolla → tutte le barre larghe ~0 = invisibili (il bug "spariscono
  //     dopo 12s", restano solo linee/shapes). Esplicita = deterministica, no collasso.
  // EN: EXPLICIT bar width (≈85% of the median strike gap). Without it Plotly derives
  //     width from the MIN gap: if a refresh brings two close strikes the min-gap
  //     collapses → all bars ~0-wide = invisible (the "vanish after 12s" bug, only
  //     lines/shapes remain). Explicit = deterministic, no collapse.
  const _ks = d.strikes.slice().sort((a,b)=>a-b);
  const _gaps = []; for(let i=1;i<_ks.length;i++){ const g=_ks[i]-_ks[i-1]; if(g>0) _gaps.push(g); }
  _gaps.sort((a,b)=>a-b);
  const _barW = (_gaps.length ? _gaps[Math.floor(_gaps.length/2)] : (_ks.length?_ks[0]*0.01:1)) * 0.85;

  // IT: banda di zoom ROBUSTA — strike con OI (≥0.5% del picco) MA limitati a
  //     spot ± 35%. CAUSA DEL BUG "barre spariscono dopo il refresh": senza il cap,
  //     un singolo strike far-OTM (range 20k–380k) che supera la soglia fa esplodere
  //     il range-x a ~360k → le barre (larghe ~500-1000) diventano SUB-PIXEL =
  //     invisibili, mentre linea Net OI e shapes (scala-indipendenti) restano.
  // EN: ROBUST zoom band — strikes with OI (≥0.5% of peak) BUT capped to spot ± 35%.
  //     ROOT CAUSE of the "bars vanish after refresh" bug: without the cap, a single
  //     far-OTM strike (range 20k–380k) crossing the threshold blows the x-range to
  //     ~360k → bars (~500-1000 wide) become SUB-PIXEL = invisible, while the Net OI
  //     line and shapes (scale-independent) remain.
  const peak = Math.max(1e-9, ...callV, ...putV);
  const thr = peak * 0.005;
  const cap = (d.spot>0 ? d.spot : Math.max(...d.strikes)) * 0.35;
  let lo=null, hi=null;
  d.strikes.forEach((k,i)=>{ if((callV[i]>thr||putV[i]>thr) && Math.abs(k-d.spot)<=cap){ if(lo==null)lo=k; hi=k; } });
  if(lo==null){ lo=d.spot*0.85; hi=d.spot*1.15; }
  const pad=((hi-lo)*0.05)||lo*0.1; const xr=[lo-pad, hi+pad];

  // IT: newPlot (NON react): Plotly.react al re-render non ridisegna le trace
  //     `bar` (le barre sparivano ai refresh, restavano solo linee/shapes).
  //     newPlot ricostruisce da zero ogni volta = sempre come il primo render OK.
  // EN: newPlot (NOT react): Plotly.react fails to redraw `bar` traces on
  //     re-render (bars vanished on refresh, only lines/shapes stayed). newPlot
  //     rebuilds from scratch every time = always like the working first render.
  Plotly.newPlot('plot-oi', [
    {type:'bar', name:'Call OI', x:d.strikes, y:callV, width:_barW, marker:{color:'#2ecc7199'},
     customdata:callV, hovertemplate:`K %{x:,.0f}<br>Call OI %{customdata:,.0f} ${unit}<extra></extra>`},
    {type:'bar', name:'Put OI', x:d.strikes, y:putV.map(v=>-v), width:_barW, marker:{color:'#ff5c5c99'},
     customdata:putV, hovertemplate:`K %{x:,.0f}<br>Put OI %{customdata:,.0f} ${unit}<extra></extra>`},
    // IT: Net OI = call−put → stessa unità/scala delle barre → asse PRIMARIO
    //     (no yaxis2 overlay, più semplice e corretto). NB: niente `uirevision`
    //     su questi grafici — con Plotly.react ai refresh non ridisegnava le
    //     trace `bar` (sparivano le barre, restavano solo le linee/shapes).
    // EN: Net OI = call−put → same unit/scale as the bars → PRIMARY axis (no
    //     yaxis2 overlay, simpler and correct). NB: no `uirevision` on these
    //     charts — with Plotly.react it failed to redraw `bar` traces on refresh
    //     (bars vanished, only lines/shapes remained).
    {type:'scatter', name:'Net OI', mode:'lines', x:d.strikes, y:netV,
     line:{color:'#f0a020',width:1.6}, hovertemplate:`K %{x:,.0f}<br>Net %{y:,.0f} ${unit}<extra></extra>`},
  ], Object.assign({}, PL_DARK, {
    dragmode:'pan', autosize:true,
    barmode:'relative', showlegend:true,
    legend:{font:{color:'#7d8aa0'},orientation:'h',y:1.08,x:0},
    hovermode:'x unified',
    xaxis:Object.assign({},PL_DARK.xaxis,{title:'Strike',range:xr}),
    yaxis:Object.assign({},PL_DARK.yaxis,{title:`Puts ▼ / Calls ▲ / Net (${unit})`,zeroline:true,zerolinecolor:'#3a465c'}),
    shapes:[
      {type:'line',x0:d.spot,x1:d.spot,y0:0,y1:1,yref:'paper',line:{color:'#4aa3ff',width:1.5}},
      {type:'line',x0:d.max_pain,x1:d.max_pain,y0:0,y1:1,yref:'paper',line:{color:'#f0a020',width:1.5,dash:'dash'}},
    ],
    annotations:[
      {x:d.spot,y:1,yref:'paper',text:'Spot',showarrow:false,font:{color:'#4aa3ff',size:10},yanchor:'bottom'},
      {x:d.max_pain,y:1,yref:'paper',text:'Max Pain',showarrow:false,font:{color:'#f0a020',size:10},yanchor:'bottom',xanchor:'right'},
    ],
  }), PL_CFG_2D);
}

async function loadRisk(){
  try{
    const d = await (await fetch('/api/risk')).json();
    const g = d.agg_greeks;
    document.getElementById('risk-cards').innerHTML = [
      {l:'Max Pain', v:'$'+fmt0(d.max_pain), s:'min total holder payoff', c:'amb'},
      {l:'Spot vs Max Pain', v:fmt(((d.spot-d.max_pain)/d.max_pain*100),1)+'%', s:'spot $'+fmt0(d.spot), c:(d.spot-d.max_pain)>=0?'pos':'neg'},
      {l:'Net OI Δ (Delta)', v:fmtKS(g.delta), s:'OI-weighted · BTC', c:cls(g.delta)},
      {l:'Net OI Γ (Gamma)', v:fmtS(g.gamma,4), s:'OI-weighted', c:cls(g.gamma)},
      {l:'Net OI ν (Vega)', v:fmtKS(g.vega), s:'per +1% vol', c:cls(g.vega)},
      {l:'Net OI Θ (Theta)', v:fmtKS(g.theta), s:'OI-weighted · USD/day', c:cls(g.theta)},
      {l:'DVOL', v:fmt(d.dvol,1)+'%', s:'30d implied', c:'amb'},
    ].map(c=>`<div class="card"><div class="lbl">${c.l}</div><div class="val ${c.c||''}">${c.v}</div><div class="sub">${c.s}</div></div>`).join('');

    renderOI(d);

    const gv=['delta','gamma','vega','theta'].map(k=>g[k]);
    Plotly.newPlot('plot-greeks', [{type:'bar', x:['Δ (Delta)','Γ (Gamma)','ν (Vega)','Θ (Theta)'], y:gv,
      marker:{color:gv.map(v=>v>=0?'#2ecc71':'#ff5c5c')},
      hovertemplate:'%{x}: %{y:+,.2f}<extra></extra>'}],
      Object.assign({}, PL_DARK, {dragmode:'pan', autosize:true, yaxis:Object.assign({},PL_DARK.yaxis,{title:'OI-weighted'})}), PL_CFG_2D);

    const totC = d.call_oi.reduce((a,b)=>a+b,0), totP = d.put_oi.reduce((a,b)=>a+b,0);
    Plotly.react('plot-pcr', [{type:'pie', labels:['Calls','Puts'], values:[totC,totP],
      marker:{colors:['#2ecc71','#ff5c5c']}, hole:.55, textinfo:'label+percent',
      textfont:{color:'#dfe6f0'}, hovertemplate:'%{label}: %{value:,.0f} BTC<extra></extra>'}],
      Object.assign({}, PL_DARK, {dragmode:'pan', autosize:true, margin:{l:8,r:8,t:8,b:8}}), PL_CFG_2D);
    setConn(true);
  }catch(e){ setConn(false); }
}

// ─── Trades (forward test vol-paper) ────────────────────────────────────────────
let TRADES = [];
async function loadTrades(){
  try{
    const d = await (await fetch('/api/trades')).json();
    TRADES = d.trades||[];
    const s = d.summary||{};
    document.getElementById('trades-cards').innerHTML = [
      {l:'Total PnL (BTC)', v:fmtS(s.total_pnl,4), s:`${s.n_settled||0} settled / gate ${s.gate_trades||30}`, c:cls(s.total_pnl||0)},
      {l:'Hit-rate', v:(s.hit_rate==null?'—':fmt(s.hit_rate*100,0)+'%'), s:'PnL > 0', c:(s.hit_rate>=0.5?'pos':'neg')},
      {l:'Avg PnL/trade', v:(s.avg_pnl==null?'—':fmtS(s.avg_pnl,4)), s:'BTC', c:cls(s.avg_pnl||0)},
      {l:'Trades', v:fmt0(s.n||0), s:`${s.n_executed||0} eseguiti (real)`, c:'amb'},
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
    if(TRADES.length){ selectTrade(TRADES.length-1); }
    else { document.getElementById('payoff-title').textContent='(nessun trade)'; Plotly.purge('plot-payoff'); }
    setConn(true);
  }catch(e){ setConn(false); }
}
function selectTrade(i){
  const t=TRADES[i]; if(!t) return;
  document.querySelectorAll('#trades-body tr').forEach((r,j)=>r.classList.toggle('atm-row', j===i));
  renderPayoff(t);
}
// IT: profilo di rischio dello straddle — PnL (BTC) a scadenza vs sottostante.
//     Opzioni inverse Deribit: payoff(S)=amount·|S−K|/S. Il costo totale è calibrato
//     dal trade realizzato (payoff@settle − pnl) così la curva passa per il marker ◆.
// EN: straddle risk profile — PnL (BTC) at expiry vs underlying. Inverse Deribit
//     options: payoff(S)=amount·|S−K|/S. Total cost is calibrated from the realized
//     trade (payoff@settle − pnl) so the curve passes through the ◆ marker.
function renderPayoff(t){
  const K=t.strike, amt=t.amount||1, side=t.side>0?1:-1;
  let cost=(t.premium||0)*amt+(t.fee_btc||0);
  if(t.delivery_price!=null && t.pnl_btc!=null){
    const pf=amt*Math.abs(t.delivery_price-K)/t.delivery_price;
    cost = side>0 ? (pf - t.pnl_btc) : (pf + t.pnl_btc);
  }
  const lo=K*0.80, hi=K*1.20, N=121, xs=[], ys=[];
  for(let j=0;j<N;j++){ const S=lo+(hi-lo)*j/(N-1);
    const pf=amt*Math.abs(S-K)/S;
    ys.push(side>0 ? pf-cost : cost-pf); xs.push(S);
  }
  // IT: range ESPLICITI x/y (come renderOI) — non affidarsi all'autorange, che con
  //     il div in una tab appena attivata (reflow) può collassare e non disegnare la
  //     curva (restavano solo gli shapes = le 2 linee verticali). | EN: EXPLICIT x/y
  //     ranges (like renderOI) — don't rely on autorange, which with the div in a
  //     just-activated tab (reflow) can collapse and skip the curve (only the shapes
  //     = the 2 vertical lines remained).
  const _xpad=(hi-lo)*0.03; const _xr=[lo-_xpad, hi+_xpad];
  let _ymin=Math.min(...ys), _ymax=Math.max(...ys);
  if(t.pnl_btc!=null){ _ymin=Math.min(_ymin,t.pnl_btc); _ymax=Math.max(_ymax,t.pnl_btc); }
  const _ypad=((_ymax-_ymin)*0.10)||0.01; const _yr=[_ymin-_ypad, _ymax+_ypad];
  const traces=[{type:'scatter',mode:'lines',x:xs,y:ys,line:{color:'#4aa3ff',width:2},
    hovertemplate:'S %{x:,.0f}<br>PnL %{y:+,.4f} BTC<extra></extra>'}];
  if(t.delivery_price!=null && t.pnl_btc!=null){
    traces.push({type:'scatter',mode:'markers',x:[t.delivery_price],y:[t.pnl_btc],
      marker:{color:(t.pnl_btc>=0?'#2ecc71':'#ff5c5c'),size:12,symbol:'diamond'},
      hovertemplate:'settle %{x:,.0f}<br>PnL %{y:+,.4f} BTC<extra></extra>'});
  }
  document.getElementById('payoff-title').textContent =
    `${t.side>0?'LONG':'SHORT'} straddle · K ${fmt0(K)} · cost ${fmt(cost,4)} BTC`;
  Plotly.newPlot('plot-payoff', traces, Object.assign({}, PL_DARK, {
    dragmode:'pan', autosize:true, showlegend:false,
    xaxis:Object.assign({},PL_DARK.xaxis,{title:'Underlying at expiry (USD)',range:_xr}),
    yaxis:Object.assign({},PL_DARK.yaxis,{title:'PnL (BTC)',range:_yr,zeroline:true,zerolinecolor:'#3a465c'}),
    shapes:[
      {type:'line',x0:K,x1:K,y0:0,y1:1,yref:'paper',line:{color:'#7d8aa0',width:1,dash:'dot'}},
      {type:'line',x0:t.entry_spot,x1:t.entry_spot,y0:0,y1:1,yref:'paper',line:{color:'#f0a020',width:1.2}},
    ],
    annotations:[
      {x:K,y:1,yref:'paper',text:'Strike',showarrow:false,font:{color:'#7d8aa0',size:10},yanchor:'bottom'},
      {x:t.entry_spot,y:1,yref:'paper',text:'Entry',showarrow:false,font:{color:'#f0a020',size:10},yanchor:'bottom',xanchor:'right'},
    ],
  }), PL_CFG_2D).then(()=>{ try{ Plotly.Plots.resize('plot-payoff'); }catch(e){} });
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
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if enc:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

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
