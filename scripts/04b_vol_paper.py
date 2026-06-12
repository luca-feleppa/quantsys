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
#     Output: results/vol_paper/{forecasts.parquet, trades.jsonl, position.json}.
#     Il log forecasts è scritto ANCHE quando flat: serve alle baseline
#     always-long/short-vol sull'intero calendario (gate pre-registrato).
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
#     Output: results/vol_paper/{forecasts.parquet, trades.jsonl, position.json}.
#     The forecasts log is written EVEN when flat: it feeds the always-long/
#     short-vol baselines over the full calendar (pre-registered gate).
import argparse
import json
import logging
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
from quantsys.data import fetch_klines, fetch_funding_rate                    # noqa: E402
from quantsys.features import FeatureBuilder, LIVE_DROP_FEATURES              # noqa: E402
from quantsys.model.ensemble import EnsembleModel                             # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vol_paper")

OUT_DIR = Path("results/vol_paper")
FORECASTS_PATH = OUT_DIR / "forecasts.parquet"
TRADES_PATH = OUT_DIR / "trades.jsonl"
POSITION_PATH = OUT_DIR / "position.json"
IV_PATH = Path("data/iv/atm_30h.parquet")

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

    # IT: ordine market sul testnet; ritorna il prezzo medio di fill (BTC/contratto).
    # EN: testnet market order; returns the average fill price (BTC/contract).
    def market_order(self, instrument: str, side: str, amount: float) -> float:
        res = self.get(f"private/{side}", {"instrument_name": instrument,
                                           "amount": amount, "type": "market"},
                       private=True)
        return float(res["order"]["average_price"])

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


# ──────────────────────────── Forecaster NN-RV ────────────────────────────
class VolForecaster:
    # IT: replica del wiring parity-blessed di FeatureAssembler (04_live_signals):
    #     FeatureBuilder configurato da config + interval/scaler/colonne INIETTATI
    #     dal PipelineState → build(fit=False) identico al training. Le candele
    #     vivono in memoria (bootstrap dal parquet + delta REST per tick).
    # EN: replica of the parity-blessed FeatureAssembler wiring (04_live_signals):
    #     FeatureBuilder configured from config + interval/scaler/columns INJECTED
    #     from PipelineState → build(fit=False) identical to training. Candles
    #     live in memory (parquet bootstrap + REST delta per tick).
    def __init__(self, cfg: dict, device, arch: str = "itransformer"):
        self.cfg = cfg
        self.device = device
        self.symbol = cfg["data"].get("symbol", "BTCUSDT")

        # IT: dir modelli parametrica via --arch (default itransformer = comportamento
        #     storico bit-identico); flag CLI esplicito, MAI QUANTSYS_ARCH (una env
        #     residua redirigerebbe il caricamento in silenzio).
        # EN: model dir parametrized via --arch (default itransformer = bit-identical
        #     legacy behavior); explicit CLI flag, NEVER QUANTSYS_ARCH (a stale env
        #     would silently redirect the loading).
        self.arch = arch
        self.model_dir = Path("models") / arch
        log.info(f"dir modelli effettiva / effective model dir: {self.model_dir} (arch={arch})")

        ps = PipelineState.load(str(self.model_dir / "pipeline_state.pkl"))
        # IT: guard contratto config↔state (pattern repo: fail-fast su mix incoerenti).
        # EN: config↔state contract guard (repo pattern: fail-fast on incoherent mixes).
        if str(cfg["data"]["interval"]) != str(ps.interval):
            raise RuntimeError(f"interval mismatch: config={cfg['data']['interval']} "
                               f"vs PipelineState={ps.interval}")
        idx = ps.scale_cols.index("target_ret")
        self.c = float(ps.scaler.center_[idx])
        self.s = float(ps.scaler.scale_[idx])
        # IT: sanity del giudice QLIKE: il centro log-RV è ≈−7; ≈0 ⇒ state direzionale stale.
        # EN: QLIKE-judge sanity: the log-RV center is ≈−7; ≈0 ⇒ stale directional state.
        assert self.c < -3, f"scaler center={self.c:.3f} ≈ 0 → PipelineState NON log-RV (stale?)"
        self.ps = ps
        self.h = int(cfg["features"].get("forecast_horizon", 30))
        self.window_size = int(cfg["model"].get("window_size", 120))

        fcfg, mcfg = cfg.get("features", {}), cfg.get("model", {})
        self.fb = FeatureBuilder(
            vp_bins          = fcfg.get("vp_bins", 30),
            vp_lookback      = fcfg.get("vp_lookback", 240),
            windows          = fcfg.get("windows", [5, 10, 20, 60]),
            lag_periods      = fcfg.get("lag_periods", 5),
            forecast_horizon = self.h,
            vp_stride        = fcfg.get("vp_stride", 1),
            frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
            use_revin        = bool(mcfg.get("use_revin", False)),
            interval_minutes = ps.interval_minutes,
        )
        self.fb.scaler             = ps.scaler
        self.fb._scale_cols        = list(ps.scale_cols)
        self.fb.scalers            = dict(ps.price_scaler_state)
        self.fb.clip_lo_           = ps.clip_lo_
        self.fb.clip_hi_           = ps.clip_hi_
        self.fb.feature_cols       = list(ps.feature_cols)
        self.fb.n_dynamic_features = ps.n_dynamic_features
        # IT: la lista canonica si deriva al primo forecast (stessi filtri di 01:
        #     exclude/C-funding/NaN/Inf su ps.feature_cols, ordine del builder) e si
        #     valida contro n_features del config.json del modello — l'npz non serve.
        # EN: the canonical list is derived on the first forecast (same filters as 01:
        #     exclude/C-funding/NaN/Inf over ps.feature_cols, builder order) and is
        #     validated against n_features in the model's config.json — no npz needed.
        self._canonical: list | None = None
        mc = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.n_feat_expected = int(mc.get("n_features", 104))
        self.n_macro_expected = int(mc.get("n_macro", 0)) if mc.get("use_macro") else 0

        self.model = EnsembleModel.load(str(self.model_dir), device)
        log.info(f"Ensemble vol caricato: {getattr(self.model, 'n_members', '?')} membri | "
                 f"scaler target: center={self.c:.3f} scale={self.s:.3f} | h={self.h}")

        # IT: macro dal parquet su disco + refit IDENTICO del MacroNormalizer (pattern
        #     dev_vols_macro_append): l'ultima riga daily-ffillata è ESATTAMENTE ciò
        #     che il training vedeva per i timestamp recenti. NO updater live (lo state
        #     vol non persiste il normalizer; il parquet è la fonte coerente col training).
        # EN: macro from the on-disk parquet + IDENTICAL MacroNormalizer refit (the
        #     dev_vols_macro_append pattern): the last daily-ffilled row is EXACTLY what
        #     training saw for recent timestamps. NO live updater (the vol state doesn't
        #     persist the normalizer; the parquet is the training-coherent source).
        self.xm = None
        if self.n_macro_expected:
            from quantsys.macro.regime import MacroNormalizer
            df_macro = pd.read_parquet("data/macro_features.parquet")
            macro_cols = list(df_macro.columns)
            assert len(macro_cols) == self.n_macro_expected, \
                f"macro: {len(macro_cols)} colonne vs n_macro={self.n_macro_expected} del modello"
            norm = MacroNormalizer()
            norm.fit_transform(df_macro, macro_cols)
            last = df_macro[macro_cols].iloc[[-1]].fillna(0.0)
            xm_np = np.clip(norm.scaler.transform(last.values.astype(np.float32)),
                            -5, 5).astype(np.float32)
            self.xm = torch.tensor(xm_np, dtype=torch.float32).to(device)
            macro_date = pd.Timestamp(df_macro.index[-1])
            age_days = (pd.Timestamp.now(tz=getattr(macro_date, 'tz', None)) - macro_date).days
            log.info(f"macro snapshot: {len(macro_cols)} feature, ultima data {macro_date.date()} "
                     f"({age_days}g fa)")
            if age_days > 7:
                log.warning(f"macro_features.parquet vecchio di {age_days}g — "
                            f"valuta un refresh (01b, sezione macro)")

        # IT: bootstrap candele + funding dal disco (entrambi refreshati per tick/avvio).
        # EN: candle + funding bootstrap from disk (both refreshed per tick/startup).
        self.candles = pd.read_parquet("data/raw_candles.parquet") \
                         .sort_values("open_time").reset_index(drop=True)
        self.funding = fetch_funding_rate(self.symbol,
                                          str(cfg["data"].get("start_time", "2019-01-01")),
                                          "data")
        log.info(f"bootstrap: {len(self.candles):,} candele 1h, {len(self.funding):,} funding obs")

    def _refresh_candles(self):
        # IT: delta REST (48 candele coprono qualsiasi gap breve), merge dedup,
        #     scarta la candela corrente non chiusa. Non riscrive il parquet su disco.
        # EN: REST delta (48 candles cover any short gap), dedup merge, drop the
        #     unfinished current candle. Does not rewrite the on-disk parquet.
        fresh = fetch_klines(self.symbol, self.cfg["data"]["interval"], limit=48)

        # IT: tutto tz-aware UTC (come il path di training: raw parquet + funding).
        # EN: everything tz-aware UTC (like the training path: raw parquet + funding).
        def _to_utc(s: pd.Series) -> pd.Series:
            s = pd.to_datetime(s)
            return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")

        fresh["open_time"] = _to_utc(fresh["open_time"])
        self.candles["open_time"] = _to_utc(self.candles["open_time"])
        merged = (pd.concat([self.candles, fresh], ignore_index=True)
                  .drop_duplicates(subset="open_time", keep="last")
                  .sort_values("open_time").reset_index(drop=True))
        cutoff = pd.Timestamp.now(tz="UTC").floor("h")
        self.candles = merged[merged["open_time"] < cutoff].reset_index(drop=True)

    def forecast(self) -> dict:
        # IT: un forecast completo: refresh candele → feature (full history, identico
        #     al training) → finestra (T,104) → ensemble → inversione completa z→raw.
        # EN: one full forecast: candle refresh → features (full history, identical
        #     to training) → (T,104) window → ensemble → full z→raw inversion.
        self._refresh_candles()
        feat = self.fb.build(self.candles, fit=False, normalize=True,
                             funding_df=self.funding)
        if self._canonical is None:
            # IT: replica ESATTA dei filtri di 01_download_data (ordine = builder):
            #     exclude non-feature → C-funding → NaN>50% → Inf. Validata sul conteggio.
            # EN: EXACT replica of the 01_download_data filters (builder order):
            #     non-feature exclude → C-funding → NaN>50% → Inf. Count-validated.
            exclude = {"open_time", "close_time", "date_utc", "pv", "cum_pv", "cum_vol",
                       "typical_price", "obv", "target_ret", "target_dir"}
            cols = [c for c in self.fb.feature_cols
                    if c not in exclude and c in feat.columns
                    and feat[c].dtype in ["float64", "float32"]
                    and c not in LIVE_DROP_FEATURES]
            cols = [c for c in cols if feat[c].isna().mean() <= 0.5]
            cols = [c for c in cols if not np.isinf(feat[c].values).any()]
            if len(cols) != self.n_feat_expected:
                raise RuntimeError(f"canonico derivato: {len(cols)} feature vs "
                                   f"n_features={self.n_feat_expected} del modello")
            self._canonical = cols
            log.info(f"lista canonica derivata e validata: {len(cols)} feature")
        feat = feat[self._canonical].dropna()
        if len(feat) < self.window_size:
            raise RuntimeError(f"righe valide {len(feat)} < window {self.window_size}")
        window = feat.tail(self.window_size).values.astype(np.float32)

        xb = torch.tensor(window[None], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            mu, _, _ = self.model(xb, self.xm) if self.xm is not None else self.model(xb)
        mu_z = float(mu.item())
        log_rv = mu_z * self.s + self.c          # IT: inversione COMPLETA | EN: FULL inversion
        rv_pred = float(np.exp(log_rv))          # IT/EN: varianza 30h dei log-return
        # IT: RV trailing h-barre — diagnostica + input della baseline naive in analisi.
        # EN: trailing h-bar RV — diagnostic + naive-baseline input at analysis time.
        lr2 = np.log(self.candles["close"] / self.candles["close"].shift(1)) ** 2
        rv_trail = float(lr2.tail(self.h).sum())
        last_ts = pd.Timestamp(self.candles["open_time"].iloc[-1])
        return {"candle_ts": last_ts, "mu_z": mu_z, "log_rv": log_rv,
                "rv_pred": rv_pred, "rv_trail": rv_trail}


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


def fee_btc(premium: float) -> float:
    # IT: fee taker per contratto, cap al 12.5% del premio (schema Deribit opzioni).
    # EN: per-contract taker fee, capped at 12.5% of premium (Deribit options schema).
    return min(FEE_PER_CONTRACT, FEE_CAP_FRAC * premium) * SIZE_CONTRACTS


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
    payoff = abs(dp - pos["strike"]) / dp * SIZE_CONTRACTS
    premium = (pos["prem_call"] + pos["prem_put"]) * SIZE_CONTRACTS
    pnl = pos["side"] * (payoff - premium) - pos["fee_btc"]
    rec = {**pos, "delivery_price": dp, "payoff_btc": payoff, "pnl_btc": pnl,
           "settled_ts": str(pd.Timestamp.now(tz="UTC").floor("s"))}
    with open(TRADES_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    save_position(None)
    log.info(f"SETTLED {('LONG' if pos['side'] > 0 else 'SHORT')} K={pos['strike']:.0f} "
             f"S_del={dp:.0f} payoff={payoff:.5f} prem={premium:.5f} → PnL={pnl:+.5f} BTC")
    return True


def open_straddle(db: DeribitTestnet, side: int, sig: dict, execute: bool) -> dict:
    # IT: apre lo straddle (LONG side=+1 compra, SHORT side=−1 vende). Senza --execute
    #     i fill sono simulati al mark price (zero rumore di fill, pre-registrato).
    # EN: opens the straddle (LONG side=+1 buys, SHORT side=−1 sells). Without
    #     --execute, fills are simulated at mark price (zero fill noise, pre-registered).
    pick = db.pick_straddle(TENOR_HOURS)
    if execute:
        verb = "buy" if side > 0 else "sell"
        prem_c = db.market_order(pick["call"], verb, SIZE_CONTRACTS)
        prem_p = db.market_order(pick["put"], verb, SIZE_CONTRACTS)
    else:
        prem_c = db.mark_price(pick["call"])
        prem_p = db.mark_price(pick["put"])
    pos = {
        "entry_ts": str(pd.Timestamp.now(tz="UTC").floor("s")),
        "side": side, "executed": bool(execute),
        "expiry_ms": pick["expiry_ms"], "t_hours_at_entry": round(pick["t_hours"], 2),
        "strike": pick["strike"], "index_at_entry": pick["index"],
        "call": pick["call"], "put": pick["put"], "amount": SIZE_CONTRACTS,
        "prem_call": prem_c, "prem_put": prem_p,
        "fee_btc": fee_btc(prem_c) + fee_btc(prem_p),
        "edge": sig["edge"], "rv_pred": sig["rv_pred"], "var_iv": sig["var_iv"],
    }
    save_position(pos)
    log.info(f"OPEN {('LONG' if side > 0 else 'SHORT')} straddle {pick['call']}/{pick['put']} "
             f"prem={prem_c + prem_p:.5f} BTC edge={sig['edge']:+.3f} "
             f"({'ORDINI REALI testnet' if execute else 'fill SIMULATO al mark'})")
    return pos


def tick(fc: VolForecaster, db: DeribitTestnet, execute: bool):
    # IT: un ciclo completo: settlement → forecast → IV → regola → log (sempre).
    # EN: one full cycle: settlement → forecast → IV → rule → log (always).
    pos = load_position()
    if pos is not None and maybe_settle(db, pos):
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
                       "var_iv": row["var_iv"]}, execute)


def main():
    # IT: boilerplate UTF-8 (checklist CLAUDE.md) | EN: UTF-8 boilerplate (CLAUDE.md checklist)
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
    args = ap.parse_args()

    cfg = load_config("config/default.yaml")
    assert cfg["features"].get("target_type") == "log_rv", \
        "config target_type deve essere log_rv / must be log_rv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fc = VolForecaster(cfg, device, arch=args.arch)
    db = DeribitTestnet(cfg)

    log.info(f"vol-paper avviato/started — soglia edge ±{EDGE_THRESHOLD}, "
             f"size {SIZE_CONTRACTS} contratti/leg, "
             f"{'ESECUZIONE TESTNET' if args.execute else 'SIMULAZIONE mark-price'}")
    while True:
        try:
            tick(fc, db, args.execute)
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
