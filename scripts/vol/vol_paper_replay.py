# IT: REPLAY OFFLINE DEL FORWARD TEST VOL-PAPER (04b) — simula i tick orari
#     che il processo live avrebbe eseguito nelle ore a PC spento, usando SOLO
#     dati su disco (post-merge VPS): candele → rv_pred (stesso modello/path
#     parity di 04b), atm_30h.parquet → var_iv (staleness ≤30 min come live),
#     chain/*.parquet → premio al mark dello snapshot ≤ t, delivery price
#     pubblico Deribit → settlement. Regola/costanti IMPORTATE da 04b
#     (nessuna copia: EDGE_THRESHOLD, TENOR_HOURS, fee_btc, ... restano
#     single-source-of-truth).
#     ⚠ SOLO ANALISI: output separati in results/vol_paper/replay/
#     (forecasts_replay.parquet, trades_replay.jsonl) — i trade replayati NON
#     entrano MAI nel gate v1 (campione pre-registrato = trade del processo
#     live). Attivazione come gap-filler ufficiale: SOLO post-chiusura gate v1
#     (STATUS.md 2026-07-14). Ogni run rigenera l'intera griglia richiesta
#     (idempotente, nessuno stato persistente).
#     Approssimazioni dichiarate vs live: (a) premio dal mark dello snapshot
#     chain a griglia 10 min invece di hh:00+90s (Δt ≤10 min); (b) macro E
#     funding as-of-t (il processo live li congela AL PROPRIO AVVIO: 04b non
#     refresha il funding per tick → le sue feature funding diventano stale con
#     l'uptime; verificato 2026-07-14 con A/B bit-identico sul path troncato —
#     il replay è il PIÙ causale dei due); (c) niente exec_diag/greeks (il
#     replay non può raccoglierli → la v2 hedged NON è replayabile, vedi
#     STATUS). Il check di parità integrato confronta mu_z/rv_pred/segnale con
#     i tick live sovrapposti e quantifica il residuo (a)+(b).
# EN: OFFLINE REPLAY OF THE VOL-PAPER FORWARD TEST (04b) — simulates the hourly
#     ticks the live process would have executed while the PC was off, using
#     ONLY on-disk data (post VPS merge): candles → rv_pred (same model/parity
#     path as 04b), atm_30h.parquet → var_iv (≤30 min staleness like live),
#     chain/*.parquet → mark premium from the snapshot ≤ t, public Deribit
#     delivery price → settlement. Rule/constants IMPORTED from 04b (no copy:
#     EDGE_THRESHOLD, TENOR_HOURS, fee_btc, ... stay single-source-of-truth).
#     ⚠ ANALYSIS ONLY: separate outputs in results/vol_paper/replay/
#     (forecasts_replay.parquet, trades_replay.jsonl) — replayed trades NEVER
#     enter the v1 gate (pre-registered sample = live-process trades). Official
#     gap-filler activation: ONLY after the v1 gate closes (STATUS.md
#     2026-07-14). Each run regenerates the whole requested grid (idempotent,
#     no persistent state).
#     Declared approximations vs live: (a) premium from the 10-min-grid chain
#     snapshot mark instead of hh:00+90s (Δt ≤10 min); (b) macro AND funding
#     as-of-t (the live process freezes them AT ITS OWN START: 04b does not
#     refresh funding per tick → its funding features go stale with uptime;
#     verified 2026-07-14 via bit-identical A/B on the truncated path — the
#     replay is the MORE causal of the two); (c) no exec_diag/greeks (the
#     replay cannot collect them → the hedged v2 is NOT replayable, see
#     STATUS). The built-in parity check compares mu_z/rv_pred/signal with
#     overlapping live ticks and quantifies (a)+(b).
import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from quantsys.utils import setup_logging, load_config          # noqa: E402
from quantsys.utils.atomic_save import atomic_save_parquet     # noqa: E402
from quantsys.features import canonical_feature_columns        # noqa: E402
from quantsys.data.deribit import delivery_price_cached as _delivery_cached  # noqa: E402

setup_logging()
log = logging.getLogger("quantsys.script.vol_replay")

# IT: import di 04b come modulo (nome con cifra iniziale → importlib): costanti
#     pre-registrate, fee e VolForecaster restano definiti in UN posto solo.
# EN: import 04b as a module (digit-leading name → importlib): pre-registered
#     constants, fee and VolForecaster stay defined in ONE place only.
_spec = importlib.util.spec_from_file_location("vol_paper_04b",
                                               ROOT / "scripts" / "04b_vol_paper.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

OUT_DIR = ROOT / "results" / "vol_paper" / "replay"
FORECASTS_OUT = OUT_DIR / "forecasts_replay.parquet"
TRADES_OUT = OUT_DIR / "trades_replay.jsonl"
DELIVERY_CACHE = OUT_DIR / "delivery_cache.json"
LIVE_FORECASTS = ROOT / "results" / "vol_paper" / "forecasts.parquet"
IV_PATH = ROOT / "data" / "iv" / "atm_30h.parquet"
CHAIN_DIR = ROOT / "data" / "iv" / "chain"
# IT: default di --start = primo tick con collector VPS attivi (deploy 2026-07-14).
# EN: --start default = first tick with VPS collectors up (2026-07-14 deploy).
DEFAULT_START = "2026-07-14T14:00:00+00:00"
# IT: staleness max dello snapshot chain per prezzare l'entry (griglia poller 10').
# EN: max chain-snapshot staleness to price the entry (10' poller grid).
CHAIN_MAX_AGE_MIN = 30.0


def read_iv_asof(iv_df: pd.DataFrame, t: pd.Timestamp) -> dict | None:
    # IT: ultimo tick IV ≤ t con età ≤ IV_MAX_AGE_MIN — stessa regola di read_iv
    #     live, valutata "as of" t invece che "now".
    # EN: latest IV tick ≤ t aged ≤ IV_MAX_AGE_MIN — same rule as the live
    #     read_iv, evaluated "as of" t instead of "now".
    sub = iv_df[iv_df["timestamp"] <= t]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    age_min = (t - row["timestamp"]).total_seconds() / 60
    if age_min > M.IV_MAX_AGE_MIN or not np.isfinite(row["iv_30h"]):
        return None
    iv = float(row["iv_30h"])
    var_iv = (iv / 100.0) ** 2 * (M.TENOR_HOURS / M.HOURS_PER_YEAR)
    return {"iv_ts": row["timestamp"], "iv_30h": iv, "var_iv": var_iv,
            "iv_age_min": age_min}


def pick_straddle_asof(t: pd.Timestamp) -> dict | None:
    # IT: replica offline di DeribitTestnet.pick_straddle: snapshot chain più
    #     recente ≤ t (età ≤ CHAIN_MAX_AGE_MIN), expiry più vicina al tenor,
    #     strike ATM sull'underlying, premio = mark_price call+put (BTC/contr.).
    # EN: offline replica of DeribitTestnet.pick_straddle: latest chain snapshot
    #     ≤ t (age ≤ CHAIN_MAX_AGE_MIN), expiry closest to tenor, ATM strike on
    #     the underlying, premium = call+put mark_price (BTC/contract).
    day_files = [CHAIN_DIR / f"btc_options_{d.strftime('%Y%m%d')}.parquet"
                 for d in (t.normalize(), (t - pd.Timedelta(days=1)).normalize())]
    frames = [pd.read_parquet(p) for p in day_files if p.exists()]
    if not frames:
        return None
    ch = pd.concat(frames, ignore_index=True)
    ch["snapshot_ts"] = pd.to_datetime(ch["snapshot_ts"], utc=True)
    ch = ch[ch["snapshot_ts"] <= t]
    if ch.empty:
        return None
    snap_ts = ch["snapshot_ts"].max()
    if (t - snap_ts).total_seconds() / 60 > CHAIN_MAX_AGE_MIN:
        return None
    snap = ch[ch["snapshot_ts"] == snap_ts].copy()
    snap["expiry"] = pd.to_datetime(snap["expiry"], utc=True)
    # IT: expiry più vicina al tenor 30h da t (stesso criterio del venue-pick).
    # EN: expiry closest to the 30h tenor from t (same criterion as venue-pick).
    exps = snap["expiry"].unique()
    exp = min(exps, key=lambda e: abs((e - t).total_seconds() / 3600 - M.TENOR_HOURS))
    sub = snap[snap["expiry"] == exp]
    und = float(sub["underlying_price"].median())
    strikes = sorted(sub["strike"].unique())
    k = min(strikes, key=lambda s: abs(s - und))
    # IT: 01c salva option_type come 'C'/'P' — matcha sull'iniziale (robusto a
    #     entrambe le convenzioni 'C'/'call').
    # EN: 01c stores option_type as 'C'/'P' — match on the initial (robust to
    #     both 'C'/'call' conventions).
    ot_up = sub["option_type"].astype(str).str.upper().str[0]
    call = sub[(sub["strike"] == k) & (ot_up == "C")]
    put = sub[(sub["strike"] == k) & (ot_up == "P")]
    if call.empty or put.empty:
        return None
    prem_c = float(call["mark_price"].iloc[0])
    prem_p = float(put["mark_price"].iloc[0])
    if not (np.isfinite(prem_c) and np.isfinite(prem_p)):
        return None
    return {"expiry_ts": pd.Timestamp(exp), "strike": float(k), "index": und,
            "snapshot_ts": snap_ts, "prem_call": prem_c, "prem_put": prem_p,
            "call": str(call["instrument_name"].iloc[0]),
            "put": str(put["instrument_name"].iloc[0]),
            "t_hours": (pd.Timestamp(exp) - t).total_seconds() / 3600}


def delivery_price_cached(expiry_ts: pd.Timestamp) -> float | None:
    # IT: delivery price production (dato di mercato) — helper condiviso C2 2ter
    #     (quantsys.data.deribit): cache DDMMMYY + paging; le vecchie chiavi
    #     YYYY-MM-DD in cache restano ignorate (refetch innocuo, valori identici).
    # EN: production delivery price (market data) — shared C2 2ter helper
    #     (quantsys.data.deribit): DDMMMYY cache + paging; old YYYY-MM-DD cache
    #     keys are ignored (harmless refetch, identical values).
    return _delivery_cached(expiry_ts, DELIVERY_CACHE)


def macro_asof(fc, t: pd.Timestamp, device):
    # IT: snapshot macro as-of t: ultima riga daily ≤ t, normalizer rifittato
    #     SOLO su righe ≤ t (niente lookahead nei parametri di scala — il live
    #     fitta sul parquet com'era a inizio processo, che finiva ~t).
    # EN: as-of-t macro snapshot: last daily row ≤ t, normalizer refit ONLY on
    #     rows ≤ t (no lookahead in the scale params — live fits on the parquet
    #     as it existed at process start, which ended ~t).
    if fc.n_macro_expected == 0:
        return None
    from quantsys.macro.regime import MacroNormalizer
    df = pd.read_parquet(ROOT / "data" / "macro_features.parquet")
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    hist = df[idx <= t]
    if hist.empty:
        return None
    cols = list(df.columns)
    norm = MacroNormalizer()
    norm.fit_transform(hist, cols)
    last = hist[cols].iloc[[-1]].fillna(0.0)
    xm = np.clip(norm.scaler.transform(last.values.astype(np.float32)),
                 -5, 5).astype(np.float32)
    return torch.tensor(xm, dtype=torch.float32).to(device)


def main() -> int:
    # IT: boilerplate UTF-8 console Windows (checklist CLAUDE.md — bug cp1252).
    # EN: Windows console UTF-8 boilerplate (CLAUDE.md checklist — cp1252 bug).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Replay offline del tick 04b / offline 04b tick replay")
    ap.add_argument("--start", default=DEFAULT_START,
                    help="inizio griglia oraria UTC / hourly grid start (UTC)")
    ap.add_argument("--end", default=None,
                    help="fine griglia (default: ultima candela chiusa) / grid end (default: last closed candle)")
    ap.add_argument("--arch", default="itransformer", help="dir modelli / model dir")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cpu di default: NON contende CUDA ai processi live / cpu default: no CUDA contention with live processes")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    # IT: riuso del wiring parity-blessed di 04b (modello+scaler+builder+candele).
    # EN: reuse of 04b's parity-blessed wiring (model+scaler+builder+candles).
    fc = M.VolForecaster(cfg, device, arch=args.arch)
    fc._refresh_candles()
    candles = fc.candles

    start = pd.Timestamp(args.start)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    last_closed = pd.Timestamp(candles["open_time"].iloc[-1]) + pd.Timedelta(hours=1)
    end = pd.Timestamp(args.end).tz_convert("UTC") if args.end else last_closed
    # IT: griglia inclusiva: start == end = un singolo tick (valido: usa candele < t).
    # EN: inclusive grid: start == end = a single tick (valid: it uses candles < t).
    if start > end:
        log.error(f"griglia vuota/empty grid: start={start} > end={end}")
        return 1
    # IT: coverage candele: il delta REST copre 48h; oltre serve 01_update_data.
    # EN: candle coverage: the REST delta spans 48h; beyond that run 01_update_data.
    if candles["open_time"].iloc[-1] < end - pd.Timedelta(hours=2):
        raise RuntimeError("candele non coprono la griglia: lancia scripts/01_update_data.py "
                           "/ candles do not cover the grid: run scripts/01_update_data.py")

    # IT: feature UNA volta su tutta la storia (builder causale, equivalenza
    #     batch↔buffer parity-blessed da 99_replay) + filtro canonico di 04b.
    # EN: features ONCE over full history (causal builder, batch↔buffer
    #     equivalence parity-blessed by 99_replay) + 04b's canonical filter.
    log.info("build feature su storia completa / building features over full history...")
    feat = fc.fb.build(candles, fit=False, normalize=True, funding_df=fc.funding)
    # IT: derivazione canonica condivisa (C2 2ter) — identica a 04b per costruzione.
    # EN: shared canonical derivation (C2 2ter) — identical to 04b by construction.
    cols = canonical_feature_columns(fc.fb.feature_cols, feat)
    if len(cols) != fc.n_feat_expected:
        raise RuntimeError(f"canonico: {len(cols)} feature vs {fc.n_feat_expected} attese/expected")
    # IT: allineamento feature↔candele PER CODA (contratto del live: feat.tail(T)
    #     = ultime T candele). Il builder scarta SOLO warmup in testa e resetta
    #     l'indice → .loc[feat.index] sarebbe sfasato di quel warmup (bug visto
    #     al primo smoke: finestre stale di 30h). NIENTE dropna globale: i NaN
    #     sporadici storici (ammessi dal filtro ≤50%) romperebbero l'alignment;
    #     la validazione NaN avviene per-finestra al tick (come il live, che
    #     consuma solo la coda).
    # EN: TAIL-based feature↔candle alignment (live contract: feat.tail(T) = the
    #     last T candles). The builder drops leading warmup ONLY and resets the
    #     index → .loc[feat.index] would be off by that warmup (bug caught in
    #     the first smoke: 30h-stale windows). NO global dropna: historical
    #     sporadic NaN (allowed by the ≤50% filter) would break the alignment;
    #     NaN validation happens per-window at tick time (like live, which only
    #     consumes the tail).
    fm = feat[cols].reset_index(drop=True)
    warmup_dropped = len(candles) - len(fm)
    assert 0 <= warmup_dropped <= 500, \
        f"drop del builder anomalo/anomalous builder drop: {warmup_dropped} righe/rows"
    ot = candles["open_time"].iloc[len(candles) - len(fm):].reset_index(drop=True)

    iv_df = pd.read_parquet(IV_PATH).sort_values("timestamp").reset_index(drop=True)
    iv_df["timestamp"] = pd.to_datetime(iv_df["timestamp"], utc=True)

    grid = pd.date_range(start.ceil("h"), end.floor("h"), freq="1h", tz="UTC")
    log.info(f"replay griglia/grid: {grid[0]} → {grid[-1]} ({len(grid)} tick, device={args.device})")

    rows, trades, pos = [], [], None
    xm_cache_day, xm = None, None
    lr2_full = np.log(candles["close"] / candles["close"].shift(1)) ** 2

    for t in grid:
        # IT: settlement PRIMA del segnale (stesso ordine del tick live).
        # EN: settlement BEFORE the signal (same order as the live tick).
        if pos is not None and t >= pos["expiry_ts"]:
            dp = delivery_price_cached(pos["expiry_ts"])
            if dp is not None:
                payoff = abs(dp - pos["strike"]) / dp * M.SIZE_CONTRACTS
                premium = (pos["prem_call"] + pos["prem_put"]) * M.SIZE_CONTRACTS
                pnl = pos["side"] * (payoff - premium) - pos["fee_btc"]
                trades.append({**{k: (str(v) if isinstance(v, pd.Timestamp) else v)
                                  for k, v in pos.items()},
                               "delivery_price": dp, "payoff_btc": payoff,
                               "pnl_btc": pnl, "settled_ts": str(t), "replay": True})
                pos = None
            # IT: dp mancante (expiry troppo recente) → posizione resta aperta.
            # EN: missing dp (too-recent expiry) → position stays open.

        # IT: finestra (T,104) con solo candele CHIUSE prima di t (open_time < t
        #     implica close ≤ t sulla griglia oraria) — identica al tick live.
        # EN: (T,104) window with only candles CLOSED before t (open_time < t
        #     implies close ≤ t on the hourly grid) — identical to the live tick.
        mask_n = int((ot < t).sum())
        if mask_n < fc.window_size:
            continue
        window = fm.iloc[mask_n - fc.window_size:mask_n].values.astype(np.float32)
        # IT: validazione NaN per-finestra (il dropna globale è vietato: rompe
        #     l'alignment per coda). Tick non valutabile → SKIP esplicito.
        # EN: per-window NaN validation (global dropna is forbidden: it breaks
        #     tail alignment). Non-evaluable tick → explicit SKIP.
        if np.isnan(window).any():
            rows.append({"candle_ts": pd.Timestamp(ot.iloc[mask_n - 1]), "tick_ts": t,
                         "mu_z": np.nan, "log_rv": np.nan, "rv_pred": np.nan,
                         "rv_trail": np.nan, "iv_30h": np.nan, "var_iv": np.nan,
                         "edge": np.nan, "action": "SKIP_NAN_WINDOW", "replay": True})
            continue
        # IT: macro as-of-t, ricalcolata solo al cambio di giorno (dato daily).
        # EN: as-of-t macro, recomputed only on day change (daily data).
        if xm_cache_day != t.date():
            xm = macro_asof(fc, t, device)
            xm_cache_day = t.date()
        xb = torch.tensor(window[None], dtype=torch.float32).to(device)
        with torch.no_grad():
            mu, _, _ = fc.model(xb, xm) if xm is not None else fc.model(xb)
        mu_z = float(mu.item())
        log_rv = mu_z * fc.s + fc.c
        rv_pred = float(np.exp(log_rv))
        rv_trail = float(lr2_full.iloc[:candles.index[candles["open_time"] < t][-1] + 1]
                         .tail(fc.h).sum())
        candle_ts = pd.Timestamp(ot.iloc[mask_n - 1])

        iv = read_iv_asof(iv_df, t)
        row = {"candle_ts": candle_ts, "tick_ts": t, "mu_z": mu_z, "log_rv": log_rv,
               "rv_pred": rv_pred, "rv_trail": rv_trail, "iv_30h": np.nan,
               "var_iv": np.nan, "edge": np.nan, "action": "NO_IV", "replay": True}
        if iv is not None:
            edge = float(np.log(rv_pred / iv["var_iv"]))
            row.update({"iv_30h": iv["iv_30h"], "var_iv": iv["var_iv"], "edge": edge})
            if pos is not None:
                row["action"] = "HOLD"
            elif edge > M.EDGE_THRESHOLD:
                row["action"] = "LONG"
            elif edge < -M.EDGE_THRESHOLD:
                row["action"] = "SHORT"
            else:
                row["action"] = "FLAT"

        if row["action"] in ("LONG", "SHORT"):
            pick = pick_straddle_asof(t)
            if pick is None:
                row["action"] = "SKIP_NO_CHAIN"
            else:
                side = +1 if row["action"] == "LONG" else -1
                pos = {"entry_ts": t, "side": side, "executed": False,
                       "expiry_ts": pick["expiry_ts"],
                       "t_hours_at_entry": round(pick["t_hours"], 2),
                       "strike": pick["strike"], "index_at_entry": pick["index"],
                       "call": pick["call"], "put": pick["put"],
                       "amount": M.SIZE_CONTRACTS,
                       "prem_call": pick["prem_call"], "prem_put": pick["prem_put"],
                       "fee_btc": M.fee_btc(pick["prem_call"]) + M.fee_btc(pick["prem_put"]),
                       "edge": row["edge"], "rv_pred": rv_pred, "var_iv": row["var_iv"],
                       "chain_snapshot_ts": pick["snapshot_ts"]}
        rows.append(row)

    # IT: output idempotenti (overwrite: il replay rigenera tutta la griglia).
    # EN: idempotent outputs (overwrite: the replay regenerates the whole grid).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    atomic_save_parquet(df, FORECASTS_OUT, index=False)
    with open(TRADES_OUT, "w", encoding="utf-8") as f:
        for tr in trades:
            f.write(json.dumps(tr, default=str) + "\n")
    n_open = 1 if pos is not None else 0
    va = df["action"].value_counts().to_dict()
    log.info(f"replay: {len(df)} tick → azioni {va} | {len(trades)} settlement "
             f"(+{n_open} aperta/open) → {TRADES_OUT}")

    # IT: check di parità sulle ore coperte ANCHE dal live: quantifica il residuo
    #     delle approssimazioni dichiarate (macro as-of, griglia chain).
    # EN: parity check on hours ALSO covered live: quantifies the residual of
    #     the declared approximations (as-of macro, chain grid).
    if LIVE_FORECASTS.exists():
        live = pd.read_parquet(LIVE_FORECASTS)
        live["candle_ts"] = pd.to_datetime(live["candle_ts"], utc=True)
        j = df.merge(live, on="candle_ts", suffixes=("_rep", "_live"))
        if len(j):
            dmu = (j["mu_z_rep"] - j["mu_z_live"]).abs().max()
            drv = (j["rv_pred_rep"] / j["rv_pred_live"] - 1).abs().max()
            agree = (j["action_rep"] == j["action_live"]).mean()
            # IT: parità sul SEGNALE puro (classificazione dell'edge, ignora lo
            #     stato posizione): l'azione diverge strutturalmente al bordo
            #     griglia (replay parte flat, il live può avere posizione aperta).
            #     Il Δmu residuo = snapshot macro del live congelato all'avvio
            #     del processo (approssimazione dichiarata nell'header).
            # EN: pure-SIGNAL parity (edge classification, ignores position
            #     state): the action structurally diverges at the grid boundary
            #     (replay starts flat, live may hold a position). The residual
            #     Δmu = live's macro snapshot frozen at process start (declared
            #     approximation in the header).
            def _sig(e):
                if not np.isfinite(e):
                    return "NO_IV"
                if e > M.EDGE_THRESHOLD:
                    return "L"
                if e < -M.EDGE_THRESHOLD:
                    return "S"
                return "F"
            sig_agree = (j["edge_rep"].map(_sig) == j["edge_live"].map(_sig)).mean()
            log.info(f"parita' vs live su {len(j)} tick sovrapposti/overlapping: "
                     f"max|Δmu_z|={dmu:.2e}, max|Δrv|/rv={drv:.2e}, "
                     f"segnale uguale/same signal {sig_agree:.0%}, azione uguale/same action {agree:.0%}")
        else:
            log.info("nessun tick sovrapposto col live / no overlapping live tick")
    return 0


if __name__ == "__main__":
    sys.exit(main())
