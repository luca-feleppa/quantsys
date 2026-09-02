# IT: FORWARD TEST VOL-PAPER (pre-registrato in STATUS.md 2026-06-12) — loop orario:
#       1. forecast NN-RV a 30h col modello vol-1h PASS (inversione COMPLETA
#          z→raw: μ_z·scale + centro dal RobustScaler persistito — pattern del
#          giudice QLIKE; feature dal path parity-blessed: FeatureBuilder
#          fit=False + scaler/colonne da PipelineState, macro via
#          MacroSnapshotUpdater con fallback zeros);
#       2. confronto con la forward variance implicita a tenor 30h dal poller
#          IV (data/iv/atm_30h.parquet, staleness ≤30 min);
#       3. regola pre-registrata: edge = log(RV_pred/var_iv); >+0.25 → LONG
#          straddle ATM daily ~30h; <−0.25 → SHORT; altrimenti flat. Max 1
#          posizione, hold a SCADENZA (cash settlement al delivery price).
#     Esecuzione: default = SIMULATA (fill al mark price Deribit — zero rumore
#     di fill); --execute piazza ordini market REALI sul testnet (OAuth2 da
#     config/secrets.yaml, blocco deribit_testnet). NO mainnet: il base URL
#     viene da secrets e DEVE contenere "test.deribit.com" (assert).
#     Output: results/vol_paper/{forecasts.parquet, trades.jsonl, position.json,
#     exec_diag.jsonl}. Il log forecasts è scritto ANCHE quando flat: serve alle
#     baseline always-long/short-vol sull'intero calendario (gate pre-registrato).
#     exec_diag.jsonl (A6, ROADMAP_VOL_BOOK) = bid/ask reali + greeks per tick,
#     SOLO diagnostico: nessun input alla regola pre-registrata.
#     V2 (B2/A1, ROADMAP_VOL_BOOK) — leg delta-hedge sul perp, dietro flag
#     --hedge INERTE di default (senza flag: comportamento v1 bit-identico,
#     nessun file hedge letto/scritto). Ribilanciamento SOLO oltre la no-trade
#     band |delta_book| (dry-run 2026-07-10: churn ATM = drag puro); hedge ratio
#     = delta teorico del venue (convenzione parametrica raw/adj, MAI stimato dai
#     mark testnet — verdetto 2026-07-08); flatten automatico al settlement.
#     Output v2: hedge_state.json + hedge_ledger.jsonl (fill esatti → il PnL
#     perp inverse si ricostruisce offline: pnl = H_usd·(1/s0−1/s1)).
#     ⚠ ATTIVARE SOLO post-gate n≥20 e SOLO dopo la pre-registrazione
#     hedged-vs-unhedged (STATUS.md): il gate v1 chiude sul design congelato.
# EN: VOL-PAPER FORWARD TEST (pre-registered in STATUS.md 2026-06-12) — hourly loop:
#       1. NN-RV 30h forecast with the PASS vol-1h model (FULL z→raw inversion:
#          μ_z·scale + center from the persisted RobustScaler — QLIKE-judge
#          pattern; features from the parity-blessed path: FeatureBuilder
#          fit=False + scaler/columns from PipelineState, macro via
#          MacroSnapshotUpdater with zeros fallback);
#       2. comparison vs the implied forward variance at 30h tenor from the IV
#          poller (data/iv/atm_30h.parquet, staleness ≤30 min);
#       3. pre-registered rule: edge = log(RV_pred/var_iv); >+0.25 → LONG ATM
#          ~30h daily straddle; <−0.25 → SHORT; else flat. Max 1 position,
#          hold to EXPIRY (cash settlement at the delivery price).
#     Execution: default = SIMULATED (fills at Deribit mark price — zero fill
#     noise); --execute places REAL market orders on the testnet (OAuth2 from
#     config/secrets.yaml, deribit_testnet block). NO mainnet: the base URL
#     comes from secrets and MUST contain "test.deribit.com" (assert).
#     Output: results/vol_paper/{forecasts.parquet, trades.jsonl, position.json,
#     exec_diag.jsonl}. The forecasts log is written EVEN when flat: it feeds the
#     always-long/short-vol baselines over the full calendar (pre-registered gate).
#     exec_diag.jsonl (A6, ROADMAP_VOL_BOOK) = real bid/ask + greeks per tick,
#     diagnostic ONLY: no input to the pre-registered rule.
#     V2 (B2/A1, ROADMAP_VOL_BOOK) — perp delta-hedge leg behind the --hedge
#     flag, INERT by default (without it: bit-identical v1 behavior, no hedge
#     file is read/written). Rebalance ONLY beyond the |book_delta| no-trade
#     band (2026-07-10 dry-run: ATM churn = pure drag); hedge ratio = venue
#     theoretical delta (parametric raw/adj convention, NEVER estimated from
#     testnet marks — 2026-07-08 verdict); automatic flatten at settlement.
#     V2 output: hedge_state.json + hedge_ledger.jsonl (exact fills → inverse
#     perp PnL is reconstructable offline: pnl = H_usd·(1/s0−1/s1)).
#     ⚠ ENABLE ONLY post-gate n≥20 and ONLY after the hedged-vs-unhedged
#     pre-registration (STATUS.md): the v1 gate closes on the frozen design.
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantsys.utils import setup_logging, load_config, PipelineState          # noqa: E402
from quantsys.utils.atomic_save import atomic_save_parquet                    # noqa: E402
from quantsys.data import fetch_klines, fetch_klines_incremental, fetch_funding_rate  # noqa: E402
from quantsys.features import FeatureBuilder, canonical_feature_columns       # noqa: E402
from quantsys.model.ensemble import EnsembleModel                             # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vol_paper")

OUT_DIR = Path("results/vol_paper")
FORECASTS_PATH = OUT_DIR / "forecasts.parquet"
TRADES_PATH = OUT_DIR / "trades.jsonl"
POSITION_PATH = OUT_DIR / "position.json"
IV_PATH = Path("data/iv/atm_30h.parquet")
# IT: A6 (ROADMAP_VOL_BOOK) — log diagnostico esecuzione (bid/ask + delta), append-only.
# EN: A6 (ROADMAP_VOL_BOOK) — execution diagnostic log (bid/ask + delta), append-only.
EXEC_DIAG_PATH = OUT_DIR / "exec_diag.jsonl"

# IT: V2 (B2/A1) — leg delta-hedge perp: stato corrente (sopravvive ai restart) +
#     ledger append-only dei fill (ricostruzione PnL inverse esatta offline).
#     File toccati SOLO con --hedge attivo (default: inerti, mai creati).
# EN: V2 (B2/A1) — perp delta-hedge leg: current state (survives restarts) +
#     append-only fill ledger (exact offline inverse-PnL reconstruction).
#     Files touched ONLY with --hedge active (default: inert, never created).
HEDGE_STATE_PATH = OUT_DIR / "hedge_state.json"
HEDGE_LEDGER_PATH = OUT_DIR / "hedge_ledger.jsonl"
PERP_INSTRUMENT = "BTC-PERPETUAL"
# IT: taglia contratto perp Deribit (10 USD) — gli ordini vanno arrotondati al multiplo.
# EN: Deribit perp contract size (10 USD) — orders must be rounded to the multiple.
PERP_CONTRACT_USD = 10.0
# IT: default PARAMETRICI (CLI), NON costanti pre-registrate: band e convenzione
#     delta vengono CONGELATE nella pre-registrazione hedged-vs-unhedged della v2
#     (dimensionate sul dry-run A6 a serie matura) PRIMA di attivare --hedge.
#     band = soglia |delta_book| in BTC-equivalenti (dry-run 07-10: sotto ~0.17 il
#     ribilanciamento non riduce varianza e paga fee); fee = taker perp (frazione
#     del nozionale, stessa assunzione del dry-run — dal venue al design finale).
# EN: PARAMETRIC defaults (CLI), NOT pre-registered constants: band and delta
#     convention get FROZEN in the v2 hedged-vs-unhedged pre-registration (sized
#     on the matured A6 dry-run) BEFORE --hedge is ever enabled.
#     band = |book_delta| threshold in BTC-equivalents (07-10 dry-run: below ~0.17
#     rebalancing reduces no variance and pays fees); fee = perp taker fraction.
DEFAULT_HEDGE_BAND = 0.20
DEFAULT_HEDGE_FEE = 5e-4

# IT: costanti PRE-REGISTRATE (STATUS.md 2026-06-12) — non toccarle a risultati visti.
# EN: PRE-REGISTERED constants (STATUS.md 2026-06-12) — do not touch after seeing results.
TENOR_HOURS = 30.0
EDGE_THRESHOLD = 0.25
SIZE_CONTRACTS = 1.0
HOURS_PER_YEAR = 8760.0          # IT: convenzione 365gg di Deribit | EN: Deribit 365-day convention
IV_MAX_AGE_MIN = 30.0
FEE_PER_CONTRACT = 0.0003        # IT: taker opzioni, BTC/contratto | EN: options taker, BTC/contract
FEE_CAP_FRAC = 0.125             # IT: cap 12.5% del premio | EN: 12.5% premium cap


# ──────────────────────────── Deribit testnet client ────────────────────────────
class DeribitTestnet:
    # IT: client minimo REST — OAuth2 client_credentials con refresh del token;
    #     SOLO testnet (assert sull'URL: un --execute non può mai toccare il mainnet).
    # EN: minimal REST client — OAuth2 client_credentials with token refresh;
    #     testnet ONLY (URL assert: --execute can never touch mainnet).
    def __init__(self, cfg: dict):
        d = cfg["deribit_testnet"]
        self.base = d["endpoint"].rstrip("/")
        assert "test.deribit.com" in self.base, \
            "endpoint deribit non-testnet — esecuzione VIETATA / non-testnet endpoint — execution FORBIDDEN"
        self._cid, self._csec = d["client_id"], d["client_secret"]
        self._token, self._token_exp = None, 0.0

    def _headers(self) -> dict:
        # IT: rinnova il Bearer token se mancano <60s alla scadenza.
        # EN: refresh the Bearer token when <60s to expiry.
        if self._token is None or time.time() > self._token_exp - 60:
            r = requests.get(f"{self.base}/public/auth", params={
                "grant_type": "client_credentials",
                "client_id": self._cid, "client_secret": self._csec}, timeout=15)
            r.raise_for_status()
            res = r.json()["result"]
            self._token = res["access_token"]
            self._token_exp = time.time() + float(res.get("expires_in", 900))
        return {"Authorization": f"Bearer {self._token}"}

    def get(self, path: str, params: dict, private: bool = False) -> dict:
        h = self._headers() if private else {}
        r = requests.get(f"{self.base}/{path}", params=params, headers=h, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if "result" not in payload:
            raise RuntimeError(f"Deribit: {payload}")
        return payload["result"]

    # IT: sceglie l'expiry daily più vicina al tenor e lo strike ATM (più vicino all'index).
    # EN: picks the daily expiry closest to the tenor and the ATM strike (nearest to index).
    def pick_straddle(self, tenor_hours: float) -> dict:
        ins = self.get("public/get_instruments",
                       {"currency": "BTC", "kind": "option", "expired": "false"})
        now_ms = time.time() * 1000
        by_exp = {}
        for i in ins:
            by_exp.setdefault(i["expiration_timestamp"], []).append(i)
        exp = min(by_exp, key=lambda e: abs((e - now_ms) / 3.6e6 - tenor_hours))
        idx = float(self.get("public/get_index_price",
                             {"index_name": "btc_usd"})["index_price"])
        strikes = sorted({i["strike"] for i in by_exp[exp]})
        k = min(strikes, key=lambda s: abs(s - idx))
        call = next(i["instrument_name"] for i in by_exp[exp]
                    if i["strike"] == k and i["option_type"] == "call")
        put = next(i["instrument_name"] for i in by_exp[exp]
                   if i["strike"] == k and i["option_type"] == "put")
        return {"expiry_ms": int(exp), "t_hours": (exp - now_ms) / 3.6e6,
                "strike": float(k), "index": idx, "call": call, "put": put}

    def mark_price(self, instrument: str) -> float:
        return float(self.get("public/ticker",
                              {"instrument_name": instrument})["mark_price"])

    # IT: prezzo indice BTC/USD corrente — input del check pin-risk (A13a).
    # EN: current BTC/USD index price — input of the pin-risk check (A13a).
    def index_price(self) -> float:
        return float(self.get("public/get_index_price",
                              {"index_name": "btc_usd"})["index_price"])

    # IT: ticker completo (bid/ask/mark/IV/greeks) — base del logging diagnostico A6.
    # EN: full ticker (bid/ask/mark/IV/greeks) — basis of the A6 diagnostic logging.
    def ticker(self, instrument: str) -> dict:
        return self.get("public/ticker", {"instrument_name": instrument})

    # IT: ordine market sul testnet; ritorna il prezzo medio di fill (BTC/contratto).
    # EN: testnet market order; returns the average fill price (BTC/contract).
    def market_order(self, instrument: str, side: str, amount: float) -> float:
        res = self.get(f"private/{side}", {"instrument_name": instrument,
                                           "amount": amount, "type": "market"},
                       private=True)
        return float(res["order"]["average_price"])

    # IT: posizione perp REALE sul venue (USD firmati, 0 se flat) — base della
    #     riconciliazione dello stato hedge all'avvio (audit MINOR-1).
    # EN: REAL venue perp position (signed USD, 0 if flat) — basis of the hedge
    #     state reconciliation at startup (MINOR-1 audit).
    def perp_position_usd(self, instrument: str = "BTC-PERPETUAL") -> float:
        res = self.get("private/get_position", {"instrument_name": instrument},
                       private=True)
        return float(res.get("size") or 0.0)

    # IT: delivery price del giorno di settlement (08:00 UTC) — None se non ancora pubblicato.
    # EN: settlement-day delivery price (08:00 UTC) — None if not yet published.
    def delivery_price(self, expiry_ms: int):
        date = datetime.fromtimestamp(expiry_ms / 1000, timezone.utc).strftime("%d%b%y").upper()
        res = self.get("public/get_delivery_prices",
                       {"index_name": "btc_usd", "count": 10})
        for rec in res.get("data", []):
            d = pd.Timestamp(rec["date"]).strftime("%d%b%y").upper()
            if d == date:
                return float(rec["delivery_price"])
        return None


# IT: VolForecaster PROMOSSO in quantsys/model/vol_forecaster.py (C2 2ter
#     2026-07-18, corpo invariato, prova A/B bit-perfetta in STATUS) - 04b
#     e vol_paper_replay lo consumano da li'.
# EN: VolForecaster PROMOTED to quantsys/model/vol_forecaster.py (C2 2ter
#     2026-07-18, unchanged body, bit-perfect A/B proof in STATUS) - 04b
#     and vol_paper_replay consume it from there.
from quantsys.model.vol_forecaster import VolForecaster, MACRO_NORM_REFIT                       # noqa: E402


# ──────────────────────────── IV + segnale ────────────────────────────
def read_iv() -> dict | None:
    # IT: ultima riga del poller; None se file assente o stale > IV_MAX_AGE_MIN.
    # EN: latest poller row; None if the file is missing or stale > IV_MAX_AGE_MIN.
    if not IV_PATH.exists():
        return None
    row = pd.read_parquet(IV_PATH).iloc[-1]
    ts = pd.Timestamp(row["timestamp"])
    age_min = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 60
    if age_min > IV_MAX_AGE_MIN or not np.isfinite(row["iv_30h"]):
        return None
    iv = float(row["iv_30h"])
    # IT: IV annualizzata (%) → varianza implicita sulla finestra di 30h.
    # EN: annualized IV (%) → implied variance over the 30h window.
    var_iv = (iv / 100.0) ** 2 * (TENOR_HOURS / HOURS_PER_YEAR)
    return {"iv_ts": ts, "iv_30h": iv, "var_iv": var_iv, "iv_age_min": age_min}


# ──────────────────────────── persistenza ────────────────────────────
def append_forecast(row: dict):
    df_new = pd.DataFrame([row])
    if FORECASTS_PATH.exists():
        df = pd.concat([pd.read_parquet(FORECASTS_PATH), df_new], ignore_index=True)
        df = df.drop_duplicates(subset="candle_ts", keep="last")
    else:
        df = df_new
    atomic_save_parquet(df.sort_values("candle_ts").reset_index(drop=True),
                        FORECASTS_PATH, index=False)


def load_position() -> dict | None:
    if POSITION_PATH.exists():
        return json.loads(POSITION_PATH.read_text(encoding="utf-8"))
    return None


def save_position(pos: dict | None):
    if pos is None:
        POSITION_PATH.unlink(missing_ok=True)
    else:
        POSITION_PATH.write_text(json.dumps(pos, indent=2, default=str), encoding="utf-8")


def fee_btc(premium: float, amount: float = SIZE_CONTRACTS) -> float:
    # IT: fee taker per contratto, cap al 12.5% del premio (schema Deribit opzioni).
    #     `amount` parametrico per il sizing v2 (A14); default = costante v1.
    # EN: per-contract taker fee, capped at 12.5% of premium (Deribit options schema).
    #     `amount` parametric for v2 sizing (A14); default = v1 constant.
    return min(FEE_PER_CONTRACT, FEE_CAP_FRAC * premium) * amount


# ──────────────────────── diagnostica esecuzione (A6) ────────────────────────
def _leg_snapshot(db: DeribitTestnet, instrument: str) -> dict:
    # IT: snapshot per-leg dal ticker: bid/ask reali (metà-spread = il costo che il
    #     fill al mark ignora), mark, IV e greeks Deribit (delta teorico BS della
    #     venue — stessa convenzione inverse/coin-settled del margin engine).
    # EN: per-leg ticker snapshot: real bid/ask (half-spread = the cost mark-price
    #     fills ignore), mark, IV and Deribit greeks (venue BS theoretical delta —
    #     same inverse/coin-settled convention as the margin engine).
    t = db.ticker(instrument)
    g = t.get("greeks") or {}

    def _f(v):
        # IT: float finito o None (il testnet può dare campi assenti/null su strike
        #     illiquidi — mai crashare, il delta si ricalcola offline dal mark_iv).
        # EN: finite float or None (testnet may return missing/null fields on
        #     illiquid strikes — never crash, delta is recomputable offline from mark_iv).
        try:
            v = float(v)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    return {"instrument": instrument,
            "bid": _f(t.get("best_bid_price")), "ask": _f(t.get("best_ask_price")),
            "bid_size": _f(t.get("best_bid_amount")), "ask_size": _f(t.get("best_ask_amount")),
            "mark": _f(t.get("mark_price")), "mark_iv": _f(t.get("mark_iv")),
            "bid_iv": _f(t.get("bid_iv")), "ask_iv": _f(t.get("ask_iv")),
            "underlying": _f(t.get("underlying_price")),
            "delta": _f(g.get("delta")), "gamma": _f(g.get("gamma")),
            "vega": _f(g.get("vega")), "theta": _f(g.get("theta"))}


def exec_diag_aggregate(legs: list, side: int, n_body: int = 2) -> dict:
    # IT: aggregati di struttura per exec_diag, funzione PURA (testabile offline).
    #     Il CORPO sono le prime `n_body` gambe (call, put ATM) e gli aggregati storici
    #     — straddle_delta, net_delta, half_spread_btc, half_spread_frac — si calcolano
    #     SOLO sul corpo, con la stessa aritmetica e lo stesso ordine di somma di prima:
    #     su un record a 2 gambe l'output è bit-identico. Con più di `n_body` gambe si
    #     aggiungono (e SOLO allora) i campi dell'intera struttura: n_legs, body_idx,
    #     structure_delta_all, half_spread_btc_all, half_spread_frac_all. I consumatori
    #     che misurano l'ATM leggono il corpo; chi vuole la struttura legge `_all`.
    # EN: structure aggregates for exec_diag, PURE function (offline-testable). The
    #     BODY is the first `n_body` legs (ATM call, put) and the historical aggregates
    #     — straddle_delta, net_delta, half_spread_btc, half_spread_frac — are computed
    #     on the body ONLY, same arithmetic and summation order as before: on a 2-leg
    #     record the output is bit-identical. With more than `n_body` legs, and ONLY
    #     then, whole-structure fields are added: n_legs, body_idx, structure_delta_all,
    #     half_spread_btc_all, half_spread_frac_all.
    def _agg(ls):
        d = None
        if all(l["delta"] is not None for l in ls):
            d = sum(l["delta"] for l in ls)
        hs_btc = hs_frac = None
        if all(l["bid"] is not None and l["ask"] is not None and l["mark"] is not None
               for l in ls):
            hs_btc = sum((l["ask"] - l["bid"]) / 2.0 for l in ls)
            mark_sum = sum(l["mark"] for l in ls)
            hs_frac = hs_btc / mark_sum if mark_sum > 0 else None
        return d, hs_btc, hs_frac

    body = legs[:n_body]
    d_body, hs_btc, hs_frac = _agg(body)
    out = {"straddle_delta": d_body,
           "net_delta": (side * d_body) if d_body is not None else None,
           "half_spread_btc": hs_btc, "half_spread_frac": hs_frac}
    if len(legs) > n_body:
        d_all, hs_btc_all, hs_frac_all = _agg(legs)
        out.update({"n_legs": len(legs), "body_idx": list(range(n_body)),
                    "structure_delta_all": d_all,
                    "half_spread_btc_all": hs_btc_all, "half_spread_frac_all": hs_frac_all})
    return out


def log_exec_diag(db: DeribitTestnet, path: Path = EXEC_DIAG_PATH):
    # IT: A6 (ROADMAP_VOL_BOOK, sequencing B3 step 1) — colonne SOLO diagnostiche,
    #     la regola pre-registrata resta INTATTA (nessun input al trading). A ogni
    #     tick orario logga bid/ask reali + delta teorico: (a) posizione aperta →
    #     le 2 leg in essere (serie del delta lungo l'holding → stima offline del
    #     valore dell'hedge, alimenta A1); (b) flat → lo straddle ATM che
    #     open_straddle sceglierebbe ORA (serie half-spread di entry → rilettura
    #     PnL net-of-half-spread a gate chiuso). Fail-soft: MAI un raise verso tick().
    # EN: A6 (ROADMAP_VOL_BOOK, B3 sequencing step 1) — diagnostic-ONLY columns,
    #     the pre-registered rule stays UNTOUCHED (no input to trading). Each hourly
    #     tick logs real bid/ask + theoretical delta: (a) open position → its 2 live
    #     legs (delta series over the holding → offline hedge-value estimate, feeds
    #     A1); (b) flat → the ATM straddle open_straddle would pick NOW (entry
    #     half-spread series → post-gate net-of-half-spread PnL re-read). Fail-soft:
    #     NEVER raises into tick().
    try:
        pos = load_position()
        if pos is not None:
            src, side = "position", int(pos["side"])
            strike, expiry_ms = float(pos["strike"]), int(pos["expiry_ms"])
            call, put = pos["call"], pos["put"]
        else:
            pick = db.pick_straddle(TENOR_HOURS)
            src, side = "atm_pick", 0
            strike, expiry_ms = float(pick["strike"]), int(pick["expiry_ms"])
            call, put = pick["call"], pick["put"]
        # IT: gambe della struttura: il CORPO (call, put ATM) sempre per primo, poi le
        #     eventuali gambe aggiuntive della posizione (`wings`, assenti oggi → lista
        #     vuota, record a 2 gambe come sempre). L'ordine è il contratto che i
        #     consumatori usano per isolare il corpo (`body_idx`).
        # EN: structure legs: the BODY (ATM call, put) always first, then any extra legs
        #     of the position (`wings`, absent today → empty list, 2-leg record as ever).
        #     The order is the contract consumers use to isolate the body (`body_idx`).
        extra = list((pos or {}).get("wings") or [])
        legs = [_leg_snapshot(db, inst) for inst in [call, put] + extra]

        # IT: delta di struttura + delta netto (side×struttura; 0 da flat È il dato
        #     corretto) e half-spread aggregato — il "haircut" che l'IVS ha dimostrato
        #     essere decision-relevant — calcolati sul CORPO da exec_diag_aggregate;
        #     i campi `_all` compaiono solo con più di 2 gambe.
        # EN: structure delta + net delta (side×structure; 0 when flat IS the correct
        #     datum) and aggregate half-spread — the "haircut" the IVS work proved
        #     decision-relevant — computed on the BODY by exec_diag_aggregate; the
        #     `_all` fields appear only with more than 2 legs.
        agg = exec_diag_aggregate(legs, side)

        rec = {"ts": str(pd.Timestamp.now(tz="UTC").floor("s")),
               "source": src, "side": side, "strike": strike, "expiry_ms": expiry_ms,
               "t_hours": round((expiry_ms - time.time() * 1000) / 3.6e6, 3),
               **agg,
               "legs": legs}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:
        # IT: la diagnostica non deve mai costare un tick di trading.
        # EN: diagnostics must never cost a trading tick.
        log.warning(f"exec-diag (A6) fallito/failed — tick NON impattato: "
                    f"{type(e).__name__}: {e}")


# ──────────────── funzioni gamma (A12/A13/A14 — lever v2, puri) ────────────────
def ww_band(fee: float, S: float, gamma_struct: float, lam: float,
            band_ref: float) -> float:
    # IT: A12 — half-width della no-trade band asintotica di Whalley–Wilmott (1997)
    #     sotto costi proporzionali: (3·k·S·Γ²/2λ)^(1/3), in spazio |delta_book|
    #     BTC-eq (lo stesso della banda fissa). k = fee frazione del nozionale,
    #     Γ = gamma di struttura in ∂δ/∂S (venue, × amount), λ = avversione al
    #     rischio (CONGELATA alla pre-registrazione). Clip a [band_ref/4, 4·band_ref]:
    #     un greek testnet assurdo non può né azzerare la banda (churn illimitato)
    #     né spalancarla (delta nudo) — stesso spirito del bound MINOR-2.
    # EN: A12 — asymptotic Whalley–Wilmott (1997) no-trade band half-width under
    #     proportional costs: (3·k·S·Γ²/2λ)^(1/3), in |book_delta| BTC-eq space
    #     (same as the fixed band). k = fee as notional fraction, Γ = structure
    #     gamma in ∂δ/∂S (venue, × amount), λ = risk aversion (FROZEN at
    #     pre-registration). Clipped to [band_ref/4, 4·band_ref]: an absurd
    #     testnet greek can neither zero the band (unbounded churn) nor blow it
    #     open (naked delta) — same spirit as the MINOR-2 bound.
    h = (1.5 * fee * S * gamma_struct ** 2 / lam) ** (1.0 / 3.0)
    return float(min(max(h, band_ref / 4.0), band_ref * 4.0))


def pin_close_due(strike: float, s_index: float, expiry_ms: float, now_ms: float,
                  max_hours: float, pin_band: float) -> bool:
    # IT: A13a — True se la posizione è nella pin region a ridosso della scadenza:
    #     0 < ore residue ≤ max_hours E |S−K|/S ≤ pin_band. A expiry passata (≤0)
    #     ritorna False: lì il payoff è congelato, compete a maybe_settle.
    # EN: A13a — True when the position sits in the pin region near expiry:
    #     0 < hours left ≤ max_hours AND |S−K|/S ≤ pin_band. Past expiry (≤0)
    #     returns False: payoff is frozen there, maybe_settle's jurisdiction.
    t_left_h = (expiry_ms - now_ms) / 3.6e6
    return 0.0 < t_left_h <= max_hours and abs(s_index - strike) / s_index <= pin_band


def vega_sized_amount(vega_sum_usd: float, target_vega_usd: float,
                      max_contracts: float) -> float:
    # IT: A14 — contratti per portare la vega di struttura al target: round a step
    #     0.1 (granularità opzioni Deribit), floor 0.1, cap fail-safe. Input non
    #     finiti/≤0 → 0.0 (il chiamante fa fallback alla size fissa).
    # EN: A14 — contracts bringing structure vega to target: rounded to the 0.1
    #     step (Deribit options granularity), 0.1 floor, fail-safe cap. Non-finite
    #     or ≤0 inputs → 0.0 (caller falls back to fixed size).
    if not (np.isfinite(vega_sum_usd) and vega_sum_usd > 0.0
            and np.isfinite(target_vega_usd) and target_vega_usd > 0.0):
        return 0.0
    amt = np.round(target_vega_usd / vega_sum_usd, 1)
    return float(min(max(amt, 0.1), max_contracts))


def maybe_pin_close(db: DeribitTestnet, pos: dict, pcfg: dict, execute: bool) -> bool:
    # IT: A13a (V2, INERTE senza --pin-close-hours) — chiusura anticipata nella pin
    #     region: a ≤x ore dalla scadenza con S nella banda |S−K|/S ≤ f il PnL
    #     marginale è pin-risk (coin-flip su S vs K), non più la bet RV-vs-IV →
    #     chiudi al market/mark e registra il trade con exit_mode="pin_close".
    #     ⚠ Cambia la regola hold-to-expiry pre-registrata: SOLO v2. Fail-soft:
    #     un errore lascia la posizione intatta (settlement resta il default).
    #     Ritorna True se la posizione è stata chiusa.
    # EN: A13a (V2, INERT without --pin-close-hours) — early close inside the pin
    #     region: within ≤x hours of expiry with S inside |S−K|/S ≤ f, marginal
    #     PnL is pin risk (coin-flip on S vs K), no longer the RV-vs-IV bet →
    #     close at market/mark and record the trade with exit_mode="pin_close".
    #     ⚠ Alters the pre-registered hold-to-expiry rule: v2 ONLY. Fail-soft:
    #     any error leaves the position intact (settlement stays the default).
    #     Returns True when the position was closed.
    try:
        s_idx = db.index_price()
        if not pin_close_due(float(pos["strike"]), s_idx, float(pos["expiry_ms"]),
                             time.time() * 1000, pcfg["hours"], pcfg["band"]):
            return False
        amt = float(pos.get("amount", SIZE_CONTRACTS))
        if execute:
            # IT: chiusura = verbo opposto all'entry (long chiude vendendo).
            # EN: closing = the verb opposite to entry (a long closes by selling).
            verb = "sell" if pos["side"] > 0 else "buy"
            exit_c = db.market_order(pos["call"], verb, amt)
            exit_p = db.market_order(pos["put"], verb, amt)
        else:
            exit_c = db.mark_price(pos["call"])
            exit_p = db.mark_price(pos["put"])
        exit_fee = fee_btc(exit_c, amt) + fee_btc(exit_p, amt)
        # IT: PnL = side·(premio uscita − premio entrata)·amount − fee entry − fee exit
        #     (stessa convenzione BTC/contratto di maybe_settle).
        # EN: PnL = side·(exit premium − entry premium)·amount − entry fee − exit fee
        #     (same BTC-per-contract convention as maybe_settle).
        pnl = pos["side"] * ((exit_c + exit_p) - (pos["prem_call"] + pos["prem_put"])) \
            * amt - pos["fee_btc"] - exit_fee
        rec = {**pos, "exit_prem_call": exit_c, "exit_prem_put": exit_p,
               "index_at_exit": s_idx, "exit_fee_btc": exit_fee, "pnl_btc": pnl,
               "exit_mode": "pin_close",
               "settled_ts": str(pd.Timestamp.now(tz="UTC").floor("s"))}
        with open(TRADES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        save_position(None)
        log.info(f"PIN-CLOSE {('LONG' if pos['side'] > 0 else 'SHORT')} "
                 f"K={pos['strike']:.0f} S={s_idx:.0f} → PnL={pnl:+.5f} BTC "
                 f"({'ordini REALI' if execute else 'fill SIMULATO al mark'})")
        return True
    except Exception as e:
        log.error(f"pin-close fallito/failed — posizione INTATTA, settlement resta "
                  f"il default: {type(e).__name__}: {e}", exc_info=True)
        return False


# ──────────────────── leg delta-hedge perp (V2, B2/A1) ────────────────────
def load_hedge_state() -> dict | None:
    if HEDGE_STATE_PATH.exists():
        return json.loads(HEDGE_STATE_PATH.read_text(encoding="utf-8"))
    return None


def save_hedge_state(st: dict | None):
    # IT: write atomica (.tmp + os.replace) — pattern safety-net del repo (audit
    #     MINOR-1: un crash tra fill e write lascerebbe uno stato stale → doppio hedge).
    # EN: atomic write (.tmp + os.replace) — repo safety-net pattern (MINOR-1
    #     audit: a crash between fill and write would leave stale state → double hedge).
    if st is None:
        HEDGE_STATE_PATH.unlink(missing_ok=True)
    else:
        tmp = HEDGE_STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, HEDGE_STATE_PATH)


def _hedge_ledger_append(rec: dict):
    with open(HEDGE_LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _perp_trade(db: DeribitTestnet, dh_usd: float, execute: bool) -> float:
    # IT: esegue il delta-ordine perp (USD firmati: >0 buy, <0 sell) e ritorna il
    #     prezzo di fill. Senza --execute il fill è simulato al mark del perp —
    #     stessa convenzione zero-rumore dei fill opzioni (spread perp ~1bp).
    # EN: executes the perp delta-order (signed USD: >0 buy, <0 sell) and returns
    #     the fill price. Without --execute the fill is simulated at the perp mark —
    #     same zero-noise convention as option fills (perp spread ~1bp).
    if execute:
        verb = "buy" if dh_usd > 0 else "sell"
        return db.market_order(PERP_INSTRUMENT, verb, abs(dh_usd))
    return float(db.ticker(PERP_INSTRUMENT)["mark_price"])


def _flatten_hedge(db: DeribitTestnet, st: dict, hcfg: dict, execute: bool, reason: str):
    # IT: chiude l'intera leg perp residua (settlement o cambio struttura sotto hedge).
    # EN: closes the whole residual perp leg (settlement or structure change under hedge).
    h_usd = float(st.get("h_usd", 0.0))
    if abs(h_usd) < PERP_CONTRACT_USD:
        save_hedge_state(None)
        return
    price = _perp_trade(db, -h_usd, execute)
    _hedge_ledger_append({
        "ts": str(pd.Timestamp.now(tz="UTC").floor("s")), "event": "flatten",
        "reason": reason, "dh_usd": -h_usd, "h_usd_after": 0.0,
        "fill_price": price, "fee_btc": hcfg["fee"] * abs(h_usd) / price,
        "executed": bool(execute), "position_key": st.get("position_key")})
    save_hedge_state(None)
    log.info(f"HEDGE flatten ({reason}): perp {-h_usd:+,.0f} USD @ {price:,.1f}")


def reconcile_hedge_state(db: DeribitTestnet):
    # IT: audit MINOR-1 — all'avvio con --execute allinea lo stato locale alla
    #     posizione perp REALE del venue: un crash tra fill e write dello stato
    #     non può più produrre un doppio hedge al restart. Se divergono adotta
    #     il venue (è la verità contabile) e logga l'evento nel ledger.
    # EN: MINOR-1 audit — at --execute startup, aligns local state with the
    #     venue's REAL perp position: a crash between fill and state write can
    #     no longer produce a double hedge on restart. On divergence the venue
    #     wins (it is the accounting truth) and the event is ledgered.
    try:
        h_venue = db.perp_position_usd(PERP_INSTRUMENT)
    except Exception as e:
        log.warning(f"riconciliazione hedge fallita/failed (get_position): "
                    f"{type(e).__name__}: {e} — proseguo con lo stato locale")
        return
    st = load_hedge_state()
    h_state = float(st["h_usd"]) if st else 0.0
    if abs(h_venue - h_state) < PERP_CONTRACT_USD:
        return
    log.warning(f"hedge state divergente: locale {h_state:+,.0f} USD vs venue "
                f"{h_venue:+,.0f} USD — adotto il venue / adopting venue")
    if abs(h_venue) < PERP_CONTRACT_USD:
        save_hedge_state(None)
    else:
        st = st or {}
        st.update({"h_usd": h_venue,
                   "updated_ts": str(pd.Timestamp.now(tz="UTC").floor("s"))})
        save_hedge_state(st)
    _hedge_ledger_append({
        "ts": str(pd.Timestamp.now(tz="UTC").floor("s")), "event": "reconcile",
        "h_usd_before": h_state, "h_usd_after": h_venue,
        "dh_usd": 0.0, "fill_price": None, "fee_btc": 0.0, "executed": True,
        "position_key": (st or {}).get("position_key")})


def maybe_hedge(db: DeribitTestnet, hcfg: dict, execute: bool):
    # IT: V2 (B2/A1) — mantiene delta_book ≈ 0 col perp inverse, MA solo oltre la
    #     no-trade band (isteresi anti-churn, dry-run 07-10: sull'ATM il ribilancio
    #     è drag puro). Convenzione delta dal venue (greeks del ticker), parametrica:
    #     'raw' = Σdelta leg (∂V_usd/∂S) · 'adj' = Σdelta − Σmark (BTC-terms,
    #     coerente con lo slope −0.98 sui mark mainnet, verdetto 07-08). Nozionale:
    #     delta_book (BTC-eq) = side·δ_conv·size + H_usd/S → target H*_usd =
    #     −side·δ_conv·size·S. Errori: log.error, MAI un raise verso tick()
    #     (la leg opzioni pre-registrata non deve mai perdere un tick).
    # EN: V2 (B2/A1) — keeps book delta ≈ 0 with the inverse perp, but ONLY beyond
    #     the no-trade band (anti-churn hysteresis; 07-10 dry-run: ATM rebalancing
    #     is pure drag). Venue delta convention (ticker greeks), parametric:
    #     'raw' = Σ leg delta (∂V_usd/∂S) · 'adj' = Σdelta − Σmark (BTC-terms,
    #     consistent with the −0.98 mainnet-mark slope, 07-08 verdict). Notional:
    #     book_delta (BTC-eq) = side·δ_conv·size + H_usd/S → target H*_usd =
    #     −side·δ_conv·size·S. Errors: log.error, NEVER raised into tick()
    #     (the pre-registered options leg must never lose a tick).
    try:
        pos = load_position()
        st = load_hedge_state()
        pos_key = None
        if pos is not None:
            pos_key = {"side": int(pos["side"]), "strike": float(pos["strike"]),
                       "expiry_ms": int(pos["expiry_ms"])}

        # IT: audit MAJOR-1 — a expiry passata le opzioni sono MORTE (payoff
        #     congelato al TWAP 07:30-08:00 UTC) anche se il delivery price non è
        #     ancora pubblicato e position.json esiste ancora: tenere il perp
        #     sarebbe delta NUDO attribuito alla leg hedge (bias sistematico
        #     contro il gate hedged-vs-unhedged). Flatten indipendente dal
        #     bookkeeping del settlement.
        # EN: MAJOR-1 audit — past expiry the options are DEAD (payoff frozen at
        #     the 07:30-08:00 UTC TWAP) even while the delivery price is not yet
        #     published and position.json still exists: keeping the perp would be
        #     NAKED delta charged to the hedge leg (systematic bias against the
        #     hedged-vs-unhedged gate). Flatten independently of settlement bookkeeping.
        expired = pos is not None and time.time() * 1000 >= float(pos["expiry_ms"])

        # IT: flatten se il book è flat, la struttura è cambiata sotto l'hedge,
        #     o la struttura è scaduta (vedi sopra).
        # EN: flatten if the book is flat, the structure changed under the hedge,
        #     or the structure expired (see above).
        if st is not None and (pos is None or expired
                               or st.get("position_key") != pos_key):
            _flatten_hedge(db, st, hcfg, execute,
                           reason=("settled" if pos is None else
                                   "expired" if expired else "structure_changed"))
            st = None
        if pos is None or expired:
            return

        legs = [_leg_snapshot(db, pos["call"]), _leg_snapshot(db, pos["put"])]
        if any(l["delta"] is None or l["mark"] is None or l["underlying"] is None
               for l in legs):
            # IT: meglio saltare un ribilancio che hedgiare con un delta sbagliato.
            # EN: better to skip one rebalance than hedge with a wrong delta.
            log.warning("hedge: greeks/mark assenti su una leg — ribilanciamento "
                        "saltato questo tick / missing greeks — rebalance skipped")
            return
        S = float(np.mean([l["underlying"] for l in legs]))
        d_raw = float(sum(l["delta"] for l in legs))
        m_sum = float(sum(l["mark"] for l in legs))
        d_conv = d_raw if hcfg["conv"] == "raw" else d_raw - m_sum
        # IT: audit MINOR-2 — bound analitico dello straddle: |δ_raw| ≤ 1
        #     (call∈[0,1], put∈[−1,0]), |δ_adj| ≤ 1 + Σmark; un greek testnet
        #     numericamente assurdo dimensionerebbe un hedge macroscopico →
        #     skip fail-soft (margine 0.10 per tolleranza di quotazione).
        # EN: MINOR-2 audit — analytic straddle bound: |δ_raw| ≤ 1, |δ_adj| ≤
        #     1 + Σmark; a numerically absurd testnet greek would size a
        #     macroscopic hedge → fail-soft skip (0.10 quoting-tolerance margin).
        d_bound = 1.0 + (m_sum if hcfg["conv"] == "adj" else 0.0) + 0.10
        if abs(d_conv) > d_bound:
            log.warning(f"hedge: delta implausibile |{d_conv:.3f}| > bound "
                        f"{d_bound:.3f} — ribilanciamento saltato / implausible "
                        f"delta — rebalance skipped")
            return
        side = int(pos["side"])
        amt = float(pos.get("amount", SIZE_CONTRACTS))
        h_usd_cur = float(st["h_usd"]) if st else 0.0

        # IT: A12 — banda effettiva: fissa (default, design storico) o Whalley–
        #     Wilmott gamma-scalata (∝ Γ^(2/3), vedi ww_band). Fail-soft: greeks
        #     gamma assenti → fallback alla banda fissa congelata.
        # EN: A12 — effective band: fixed (default, legacy design) or gamma-scaled
        #     Whalley–Wilmott (∝ Γ^(2/3), see ww_band). Fail-soft: missing gamma
        #     greeks → fallback to the frozen fixed band.
        band_eff = hcfg["band"]
        if hcfg.get("band_mode") == "ww":
            gammas = [l["gamma"] for l in legs]
            if all(g is not None for g in gammas):
                g_struct = float(sum(gammas)) * amt
                band_eff = ww_band(hcfg["fee"], S, g_struct,
                                   hcfg["ww_lambda"], hcfg["band"])
            else:
                log.warning("hedge ww: gamma assente su una leg — banda fissa "
                            "questo tick / missing gamma — fixed band this tick")

        # IT: delta del book in BTC-equivalenti (opzioni + perp già in essere).
        # EN: book delta in BTC-equivalents (options + perp already on).
        book_delta = side * d_conv * amt + h_usd_cur / S
        if abs(book_delta) < band_eff:
            return                                     # dentro la banda / inside the band
        h_usd_target = -side * d_conv * amt * S
        dh = h_usd_target - h_usd_cur
        dh = float(np.round(dh / PERP_CONTRACT_USD) * PERP_CONTRACT_USD)
        if abs(dh) < PERP_CONTRACT_USD:
            return
        price = _perp_trade(db, dh, execute)
        h_usd_new = h_usd_cur + dh
        save_hedge_state({"h_usd": h_usd_new, "position_key": pos_key,
                          "conv": hcfg["conv"], "last_fill_price": price,
                          "updated_ts": str(pd.Timestamp.now(tz="UTC").floor("s"))})
        _hedge_ledger_append({
            "ts": str(pd.Timestamp.now(tz="UTC").floor("s")),
            "event": "open" if abs(h_usd_cur) < PERP_CONTRACT_USD else "rebalance",
            "S_underlying": S, "delta_raw": d_raw, "mark_sum": m_sum,
            "conv": hcfg["conv"], "book_delta_pre": book_delta,
            "band_eff": band_eff, "band_mode": hcfg.get("band_mode", "fixed"),
            "h_usd_before": h_usd_cur, "h_usd_after": h_usd_new, "dh_usd": dh,
            "fill_price": price, "fee_btc": hcfg["fee"] * abs(dh) / price,
            "executed": bool(execute), "position_key": pos_key})
        log.info(f"HEDGE {'open' if abs(h_usd_cur) < PERP_CONTRACT_USD else 'rebalance'}: "
                 f"Δ_book={book_delta:+.3f} → perp {dh:+,.0f} USD @ {price:,.1f} "
                 f"(H={h_usd_new:+,.0f} USD, conv={hcfg['conv']})")
    except Exception as e:
        log.error(f"hedge leg fallita/failed — leg opzioni NON impattata: "
                  f"{type(e).__name__}: {e}", exc_info=True)


# ──────────────────────────── ciclo di trade ────────────────────────────
def maybe_settle(db: DeribitTestnet, pos: dict) -> bool:
    # IT: se l'expiry è passata e il delivery price è pubblicato: P&L cash-settled
    #     (opzioni inverse: payoff straddle = |S−K|/S_del in BTC/contratto) → trades.jsonl.
    # EN: if expiry has passed and the delivery price is out: cash-settled P&L
    #     (inverse options: straddle payoff = |S−K|/S_del in BTC/contract) → trades.jsonl.
    if time.time() * 1000 < pos["expiry_ms"]:
        return False
    dp = db.delivery_price(pos["expiry_ms"])
    if dp is None:
        log.info("expiry passata ma delivery price non ancora pubblicato — riprovo al prossimo tick")
        return False
    # IT: amount dalla posizione (A14-ready); fallback = costante v1 → bit-identico
    #     per le posizioni storiche (amount è sempre stato SIZE_CONTRACTS).
    # EN: amount from the position (A14-ready); fallback = v1 constant → bit-identical
    #     for historical positions (amount has always been SIZE_CONTRACTS).
    amt = float(pos.get("amount", SIZE_CONTRACTS))
    payoff = abs(dp - pos["strike"]) / dp * amt
    premium = (pos["prem_call"] + pos["prem_put"]) * amt
    pnl = pos["side"] * (payoff - premium) - pos["fee_btc"]
    rec = {**pos, "delivery_price": dp, "payoff_btc": payoff, "pnl_btc": pnl,
           "exit_mode": "settlement",
           "settled_ts": str(pd.Timestamp.now(tz="UTC").floor("s"))}
    with open(TRADES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    save_position(None)
    log.info(f"SETTLED {('LONG' if pos['side'] > 0 else 'SHORT')} K={pos['strike']:.0f} "
             f"S_del={dp:.0f} payoff={payoff:.5f} prem={premium:.5f} → PnL={pnl:+.5f} BTC")
    return True


def open_straddle(db: DeribitTestnet, side: int, sig: dict, execute: bool,
                  size_cfg: dict | None = None) -> dict:
    # IT: apre lo straddle (LONG side=+1 compra, SHORT side=−1 vende). Senza --execute
    #     i fill sono simulati al mark price (zero rumore di fill, pre-registrato).
    #     A14 (V2, size_cfg non-None): amount = target_vega/Σν all'entry (bet uniforme
    #     in spazio-vol); fail-soft alla size fissa se le vega venue mancano.
    # EN: opens the straddle (LONG side=+1 buys, SHORT side=−1 sells). Without
    #     --execute, fills are simulated at mark price (zero fill noise, pre-registered).
    #     A14 (V2, non-None size_cfg): amount = target_vega/Σν at entry (uniform bet
    #     in vol space); fail-soft to fixed size when venue vegas are missing.
    pick = db.pick_straddle(TENOR_HOURS)
    amount = SIZE_CONTRACTS
    if size_cfg is not None:
        legs = [_leg_snapshot(db, pick["call"]), _leg_snapshot(db, pick["put"])]
        vegas = [l["vega"] for l in legs]
        amt = vega_sized_amount(sum(v for v in vegas if v is not None)
                                if all(v is not None for v in vegas) else 0.0,
                                size_cfg["target_vega"], size_cfg["max_contracts"])
        if amt > 0.0:
            amount = amt
        else:
            log.warning("sizing vega (A14): vega venue assente/degenere — fallback "
                        f"size fissa {SIZE_CONTRACTS} / missing venue vega — fixed-size fallback")
    if execute:
        verb = "buy" if side > 0 else "sell"
        prem_c = db.market_order(pick["call"], verb, amount)
        prem_p = db.market_order(pick["put"], verb, amount)
    else:
        prem_c = db.mark_price(pick["call"])
        prem_p = db.mark_price(pick["put"])
    pos = {
        "entry_ts": str(pd.Timestamp.now(tz="UTC").floor("s")),
        "side": side, "executed": bool(execute),
        "expiry_ms": pick["expiry_ms"], "t_hours_at_entry": round(pick["t_hours"], 2),
        "strike": pick["strike"], "index_at_entry": pick["index"],
        "call": pick["call"], "put": pick["put"], "amount": amount,
        "prem_call": prem_c, "prem_put": prem_p,
        "fee_btc": fee_btc(prem_c, amount) + fee_btc(prem_p, amount),
        "edge": sig["edge"], "rv_pred": sig["rv_pred"], "var_iv": sig["var_iv"],
    }
    save_position(pos)
    log.info(f"OPEN {('LONG' if side > 0 else 'SHORT')} straddle {pick['call']}/{pick['put']} "
             f"prem={prem_c + prem_p:.5f} BTC edge={sig['edge']:+.3f} "
             f"({'ORDINI REALI testnet' if execute else 'fill SIMULATO al mark'})")
    return pos


def tick(fc: VolForecaster, db: DeribitTestnet, execute: bool,
         hedge_cfg: dict | None = None, pin_cfg: dict | None = None,
         size_cfg: dict | None = None):
    # IT: un ciclo completo: settlement → pin-close (SOLO v2) → forecast → IV →
    #     regola → log (sempre) → hedge (SOLO v2). Tutti i cfg=None = v1 bit-identico.
    # EN: one full cycle: settlement → pin-close (v2 ONLY) → forecast → IV → rule
    #     → log (always) → hedge (v2 ONLY). All cfg=None = bit-identical v1.
    pos = load_position()
    if pos is not None and maybe_settle(db, pos):
        pos = None
    # IT: A13a — dopo il settlement (expiry passata compete a maybe_settle) e prima
    #     della regola di entry: un pin-close libera il libro nello stesso tick.
    # EN: A13a — after settlement (past expiry belongs to maybe_settle) and before
    #     the entry rule: a pin-close frees the book within the same tick.
    if pos is not None and pin_cfg is not None and maybe_pin_close(db, pos, pin_cfg, execute):
        pos = None

    f = fc.forecast()
    iv = read_iv()
    row = {"candle_ts": f["candle_ts"], "mu_z": f["mu_z"], "log_rv": f["log_rv"],
           "rv_pred": f["rv_pred"], "rv_trail": f["rv_trail"], "iv_30h": np.nan,
           "var_iv": np.nan, "edge": np.nan, "action": "NO_IV", "executed": bool(execute)}
    if iv is not None:
        edge = float(np.log(f["rv_pred"] / iv["var_iv"]))
        row.update({"iv_30h": iv["iv_30h"], "var_iv": iv["var_iv"], "edge": edge})
        if pos is not None:
            row["action"] = "HOLD"
        elif edge > EDGE_THRESHOLD:
            row["action"] = "LONG"
        elif edge < -EDGE_THRESHOLD:
            row["action"] = "SHORT"
        else:
            row["action"] = "FLAT"
    append_forecast(row)
    log.info(f"tick {f['candle_ts']}: rv_pred={f['rv_pred']:.3e} "
             f"var_iv={row['var_iv']:.3e} edge={row['edge']:+.3f} → {row['action']}"
             if iv is not None else
             f"tick {f['candle_ts']}: rv_pred={f['rv_pred']:.3e} → NO_IV (poller stale/assente)")

    if row["action"] in ("LONG", "SHORT"):
        open_straddle(db, +1 if row["action"] == "LONG" else -1,
                      {"edge": row["edge"], "rv_pred": f["rv_pred"],
                       "var_iv": row["var_iv"]}, execute, size_cfg)

    # IT: A6 — dopo l'eventuale open, così la riga cattura le leg della posizione
    #     appena aperta al momento dell'ingresso (half-spread di entry reale).
    # EN: A6 — after any open, so the row captures the just-opened position's legs
    #     at entry time (real entry half-spread).
    log_exec_diag(db)

    # IT: V2 — la leg hedge gira per ULTIMA (dopo settlement/open/diagnostica):
    #     vede lo stato book definitivo del tick. Inerte se hedge_cfg è None.
    # EN: V2 — the hedge leg runs LAST (after settlement/open/diagnostics): it
    #     sees the tick's final book state. Inert when hedge_cfg is None.
    if hedge_cfg is not None:
        maybe_hedge(db, hedge_cfg, execute)


def main():
    # IT: boilerplate UTF-8 (checklist nuovo script) | EN: UTF-8 boilerplate (new-script checklist)
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Forward test vol-paper (NN-RV vs IV, testnet Deribit)")
    ap.add_argument("--once", action="store_true",
                    help="un solo tick e termina (smoke) / single tick then exit (smoke)")
    ap.add_argument("--execute", action="store_true",
                    help="piazza ordini REALI sul testnet (default: fill simulati al mark) / "
                         "place REAL testnet orders (default: simulated mark fills)")
    # IT: arch del modello da caricare (models/{arch}); flag esplicito, NON env
    #     QUANTSYS_ARCH — default itransformer = run storica bit-identica.
    # EN: model arch to load (models/{arch}); explicit flag, NOT the QUANTSYS_ARCH
    #     env var — default itransformer = bit-identical legacy run.
    ap.add_argument("--arch", default="itransformer",
                    choices=["itransformer", "nhits", "tcnmamba", "lstm"],
                    help="architettura del modello vol da caricare (models/{arch}) / "
                         "vol model architecture to load (models/{arch})")
    # IT: normalizer macro — INERTE di default. Senza il flag lo strumento e'
    #     ri-stimato whole-df a ogni bootstrap (comportamento storico bit-identico);
    #     con un path lo strumento e' PINNATO a un vintage dichiarato e varia solo
    #     lo stato. ⚠ Cambia l'input del live: attivarlo e' un atto DELIBERATO e va
    #     datato in STATUS.md se un campione forward e' aperto. Flag esplicito e MAI
    #     env, come --arch: una env residua cambierebbe l'input in silenzio.
    # EN: macro normalizer — INERT by default. Without the flag the instrument is
    #     refitted whole-df at every bootstrap (bit-identical legacy behavior); with
    #     a path the instrument is PINNED at a declared vintage and only the state
    #     moves. ⚠ It changes the live input: enabling it is a DELIBERATE act and must
    #     be dated in STATUS.md if a forward sample is open. Explicit flag and NEVER
    #     env, like --arch: a stale env would change the input silently.
    ap.add_argument("--macro-norm", default=None, metavar="PATH",
                    help="pickle del MacroNormalizer pinnato (default: ri-stima "
                         "whole-df, comportamento storico) / pinned MacroNormalizer "
                         "pickle (default: whole-df refit, legacy behavior)")
    # IT: V2 (B2/A1) — flag hedge, INERTI di default. ⚠ Attivarli SOLO post-gate
    #     n≥20 e SOLO con band/convenzione CONGELATE dalla pre-registrazione
    #     hedged-vs-unhedged in STATUS.md (i default qui sono placeholder di design).
    # EN: V2 (B2/A1) — hedge flags, INERT by default. ⚠ Enable ONLY post-gate
    #     n≥20 and ONLY with band/convention FROZEN by the hedged-vs-unhedged
    #     pre-registration in STATUS.md (defaults here are design placeholders).
    ap.add_argument("--hedge", action="store_true",
                    help="attiva la leg delta-hedge perp (v2; default OFF = v1 "
                         "bit-identico) / enable the perp delta-hedge leg (v2)")
    # IT: audit MINOR-3 — band/conv SENZA default: con --hedge vanno passati
    #     esplicitamente (fail-fast sotto), così un --hedge distratto non parte
    #     coi placeholder non congelati dalla pre-registrazione.
    # EN: MINOR-3 audit — band/conv WITHOUT defaults: with --hedge they must be
    #     passed explicitly (fail-fast below), so a careless --hedge cannot start
    #     on placeholders the pre-registration has not frozen.
    ap.add_argument("--hedge-band", type=float, default=None,
                    help="no-trade band su |delta_book| in BTC-eq (OBBLIGATORIA "
                         f"con --hedge; riferimento design {DEFAULT_HEDGE_BAND}) / "
                         "|book_delta| no-trade band (REQUIRED with --hedge)")
    ap.add_argument("--hedge-conv", choices=["raw", "adj"], default=None,
                    help="convenzione delta (OBBLIGATORIA con --hedge): raw=Σdelta "
                         "venue, adj=Σdelta−Σmark (BTC-terms) / delta convention "
                         "(REQUIRED with --hedge)")
    ap.add_argument("--hedge-fee", type=float, default=DEFAULT_HEDGE_FEE,
                    help="fee taker perp (frazione nozionale, solo contabilità "
                         "ledger) / perp taker fee (ledger accounting only)")
    # IT: A12 (V2) — modalità banda: fixed (default, design storico) o ww =
    #     Whalley–Wilmott gamma-scalata; con ww, λ è OBBLIGATORIA (pattern MINOR-3:
    #     il valore va CONGELATO dalla pre-registrazione hedged-vs-unhedged).
    # EN: A12 (V2) — band mode: fixed (default, legacy design) or ww = gamma-scaled
    #     Whalley–Wilmott; with ww, λ is REQUIRED (MINOR-3 pattern: the value must
    #     be FROZEN by the hedged-vs-unhedged pre-registration).
    ap.add_argument("--hedge-band-mode", choices=["fixed", "ww"], default="fixed",
                    help="banda no-trade: fixed (default) o ww gamma-scalata "
                         "Whalley–Wilmott / no-trade band: fixed (default) or "
                         "gamma-scaled Whalley–Wilmott")
    ap.add_argument("--hedge-ww-lambda", type=float, default=None,
                    help="avversione al rischio λ della banda ww (OBBLIGATORIA con "
                         "--hedge-band-mode ww) / ww-band risk aversion λ "
                         "(REQUIRED with --hedge-band-mode ww)")
    # IT: A13a (V2, INERTI) — early-close nella pin region: entrambe obbligatorie
    #     insieme; default None = hold-to-expiry pre-registrato INTATTO.
    # EN: A13a (V2, INERT) — pin-region early close: both required together;
    #     default None = pre-registered hold-to-expiry UNTOUCHED.
    ap.add_argument("--pin-close-hours", type=float, default=None,
                    help="chiudi anticipato se restano ≤ X ore E S è nella pin band "
                         "(v2; default OFF) / close early when ≤ X hours remain AND "
                         "S is inside the pin band (v2; default OFF)")
    ap.add_argument("--pin-close-band", type=float, default=None,
                    help="pin region |S−K|/S ≤ f per l'early-close (v2, con "
                         "--pin-close-hours) / pin region |S−K|/S ≤ f for the "
                         "early close (v2, with --pin-close-hours)")
    # IT: A14 (V2, INERTE) — sizing vega-normalizzato: amount = target/Σν all'entry.
    # EN: A14 (V2, INERT) — vega-normalized sizing: amount = target/Σν at entry.
    ap.add_argument("--size-mode", choices=["contracts", "vega"], default="contracts",
                    help="contracts = size fissa pre-registrata (default); vega = "
                         "amount vega-normalizzato (v2) / contracts = pre-registered "
                         "fixed size (default); vega = vega-normalized amount (v2)")
    ap.add_argument("--size-vega-target", type=float, default=None,
                    help="vega di struttura target in USD/vol-pt (OBBLIGATORIA con "
                         "--size-mode vega) / target structure vega in USD/vol-pt "
                         "(REQUIRED with --size-mode vega)")
    ap.add_argument("--size-max-contracts", type=float, default=10.0,
                    help="cap fail-safe sull'amount vega-normalizzato / fail-safe "
                         "cap on the vega-normalized amount")
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config target_type deve essere log_rv / must be log_rv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc = VolForecaster(cfg, device, arch=args.arch,
                      macro_norm=(args.macro_norm or MACRO_NORM_REFIT))
    db = DeribitTestnet(cfg)

    # IT: config hedge SOLO se --hedge (None = path v1, nessun file hedge toccato).
    #     Fail-fast (audit MINOR-3): band e conv esplicite = valori CONGELATI
    #     dalla pre-registrazione, mai i placeholder di design.
    # EN: hedge config ONLY with --hedge (None = v1 path, no hedge file touched).
    #     Fail-fast (MINOR-3 audit): explicit band and conv = values FROZEN by
    #     the pre-registration, never the design placeholders.
    hedge_cfg = None
    if args.hedge:
        if args.hedge_band is None or args.hedge_conv is None:
            raise SystemExit(
                "--hedge richiede --hedge-band e --hedge-conv ESPLICITI (valori "
                "congelati dalla pre-registrazione in STATUS.md) / --hedge requires "
                "EXPLICIT --hedge-band and --hedge-conv (frozen by the STATUS.md "
                "pre-registration)")
        if args.hedge_band_mode == "ww" and args.hedge_ww_lambda is None:
            raise SystemExit(
                "--hedge-band-mode ww richiede --hedge-ww-lambda ESPLICITA (valore "
                "congelato dalla pre-registrazione) / ww band mode requires an "
                "EXPLICIT --hedge-ww-lambda (frozen by the pre-registration)")
        hedge_cfg = {"band": float(args.hedge_band), "conv": args.hedge_conv,
                     "fee": float(args.hedge_fee),
                     "band_mode": args.hedge_band_mode,
                     "ww_lambda": (float(args.hedge_ww_lambda)
                                   if args.hedge_ww_lambda is not None else None)}
        log.warning(f"V2 HEDGE ATTIVO: band={hedge_cfg['band']} conv={hedge_cfg['conv']} "
                    f"fee={hedge_cfg['fee']:.1e} band_mode={hedge_cfg['band_mode']} — "
                    f"verificare che la pre-registrazione "
                    f"hedged-vs-unhedged sia CHIUSA in STATUS.md / verify the "
                    f"pre-registration is FROZEN in STATUS.md")
        # IT: audit MINOR-1 (riconciliazione) — con --execute lo stato locale
        #     viene allineato alla posizione perp REALE del venue all'avvio:
        #     un crash tra fill e write non può più produrre doppio hedge.
        # EN: MINOR-1 audit (reconciliation) — with --execute the local state is
        #     aligned to the venue's REAL perp position at startup: a crash
        #     between fill and write can no longer produce a double hedge.
        if args.execute:
            reconcile_hedge_state(db)

    # IT: A13a — pin-close: coppia di parametri obbligatoria (fail-fast, pattern
    #     MINOR-3); None = regola hold-to-expiry pre-registrata intatta.
    # EN: A13a — pin close: parameter pair required together (fail-fast, MINOR-3
    #     pattern); None = pre-registered hold-to-expiry rule untouched.
    pin_cfg = None
    if args.pin_close_hours is not None or args.pin_close_band is not None:
        if args.pin_close_hours is None or args.pin_close_band is None:
            raise SystemExit(
                "--pin-close-hours e --pin-close-band vanno passati INSIEME (valori "
                "congelati dalla pre-registrazione v2) / --pin-close-hours and "
                "--pin-close-band must be passed TOGETHER (frozen by the v2 "
                "pre-registration)")
        pin_cfg = {"hours": float(args.pin_close_hours),
                   "band": float(args.pin_close_band)}
        log.warning(f"V2 PIN-CLOSE ATTIVO: hours≤{pin_cfg['hours']} "
                    f"band={pin_cfg['band']} — la regola hold-to-expiry v1 è "
                    f"SOSPESA / the v1 hold-to-expiry rule is SUSPENDED")

    # IT: A14 — sizing vega-normalizzato (fail-fast sul target, pattern MINOR-3).
    # EN: A14 — vega-normalized sizing (fail-fast on the target, MINOR-3 pattern).
    size_cfg = None
    if args.size_mode == "vega":
        if args.size_vega_target is None:
            raise SystemExit(
                "--size-mode vega richiede --size-vega-target ESPLICITO (valore "
                "congelato dalla pre-registrazione v2) / --size-mode vega requires "
                "an EXPLICIT --size-vega-target (frozen by the v2 pre-registration)")
        size_cfg = {"target_vega": float(args.size_vega_target),
                    "max_contracts": float(args.size_max_contracts)}
        log.warning(f"V2 SIZING VEGA ATTIVO: target={size_cfg['target_vega']} "
                    f"USD/vol-pt cap={size_cfg['max_contracts']} contratti — la "
                    f"size fissa v1 è SOSPESA / the v1 fixed size is SUSPENDED")

    log.info(f"vol-paper avviato/started — soglia edge ±{EDGE_THRESHOLD}, "
             f"size {SIZE_CONTRACTS} contratti/leg, "
             f"{'ESECUZIONE TESTNET' if args.execute else 'SIMULAZIONE mark-price'}"
             f"{' + DELTA-HEDGE v2' if hedge_cfg is not None else ''}"
             f"{' + PIN-CLOSE v2' if pin_cfg is not None else ''}"
             f"{' + SIZING-VEGA v2' if size_cfg is not None else ''}")
    while True:
        try:
            tick(fc, db, args.execute, hedge_cfg, pin_cfg, size_cfg)
        except KeyboardInterrupt:
            return
        except Exception as e:
            log.error(f"tick fallito/failed: {type(e).__name__}: {e}", exc_info=True)
        if args.once:
            return
        # IT: dorme fino a hh:00:90 (chiusura candela 1h + margine propagazione REST).
        # EN: sleeps until hh:00:90 (1h candle close + REST propagation margin).
        now = time.time()
        nxt = (int(now // 3600) + 1) * 3600 + 90
        try:
            time.sleep(max(60.0, nxt - now))
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
