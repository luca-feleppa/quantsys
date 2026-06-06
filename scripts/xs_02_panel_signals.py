"""
xs_02_panel_signals.py — Harness di PANEL-INFERENCE per la sonda IC cross-sectional.

IT: Applica il modello ESISTENTE (ensemble eterogeneo iTransformer+N-HiTS+TCN+Mamba,
    nessun retraining) a ogni asset per produrre un pannello LONG di μ/σ raw predetti,
    da testare in seguito per skill di RANGO cross-sectional. Lo scaler RobustScaler
    fittato su BTC (+ target_scale) è applicato UNIFORMEMENTE a ogni asset: la
    trasformazione uniforme preserva l'ORDINAMENTO cross-sectional (è il rango che
    testiamo). Il modello predice in spazio z-score → dopo il forward si DEVE chiamare
    PipelineState.denormalize_predictions(mu, sigma) per ottenere μ/σ in spazio RAW
    (frazione di log-return) — saltarla è il bug più costoso del progetto.
EN: Applies the EXISTING model (heterogeneous ensemble, NO retraining) to each asset to
    emit a LONG panel of predicted raw μ/σ for later cross-sectional RANK-skill testing.
    The BTC-fit RobustScaler (+ target_scale) is applied UNIFORMLY to every asset: the
    uniform transform preserves the cross-sectional ORDERING (the rank is what we test).
    The model predicts in z-score space → after the forward pass we MUST call
    PipelineState.denormalize_predictions(mu, sigma) to get RAW-space μ/σ.

Schema deliverato | Delivered schema:
    generate_mu_for_symbol(raw_parquet, funding_parquet)
        -> DataFrame[open_time, mu_raw, sigma_raw, fwd_ret_30]
    Panel: data/xs/mu_panel.parquet[open_time, symbol, mu_raw, sigma_raw, fwd_ret_30]
           (open_time = datetime64[ms, UTC])

Run config PyCharm:
  Script: scripts/xs_02_panel_signals.py
  Working dir: <root del progetto>
"""
from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

# IT: Forza UTF-8 su stdout/stderr — Windows cp1252 crasha sui caratteri non-ASCII.
# EN: Force UTF-8 on stdout/stderr — Windows cp1252 crashes on non-ASCII chars.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch

# IT: Root del progetto sul path per import del package quantsys/.
# EN: Project root on sys.path so the quantsys/ package imports resolve.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quantsys.features import FeatureBuilder, get_canonical_feature_names
from quantsys.model.ensemble import EnsembleModel
from quantsys.utils import PipelineState, load_config, setup_device, setup_logging

setup_logging()
log = logging.getLogger("quantsys.script.xs_02")

# ── Costanti pipeline (single source of truth = config + PipelineState) ──────────
# IT: forecast_horizon=30 (target = somma dei prossimi 30 log-return); window=120×104.
#     La griglia step=30 (=h) rende i forward return NON sovrapposti → bet quasi-indip.
# EN: forecast_horizon=30 (target = sum of next 30 log-returns); window=120×104.
#     The step=30 (=h) grid yields NON-overlapping forward returns → quasi-independent bets.
ARCH_DIR = "itransformer"   # IT: dir checkpoint/PipelineState | EN: checkpoint/PipelineState dir
# IT: 2026-05-16 — batch 256 mantiene la VRAM <3 GiB anche col branch Mamba attivo (vedi 03_backtest).
# EN: 2026-05-16 — batch 256 keeps VRAM <3 GiB even with the Mamba branch on (see 03_backtest).
BATCH_SIZE = 256


# IT: Costruisce un FeatureBuilder con lo scaler/clip/feature_cols di BTC (no fit).
#     Replica esatta del path replay (99_replay_live_vs_training.py): la trasformazione
#     è quella di BTC, applicata uniformemente all'asset → ordinamento cross-sectional sano.
# EN: Builds a FeatureBuilder carrying BTC's scaler/clip/feature_cols (no fit). Exact
#     replica of the replay path: the BTC transform is applied uniformly to the asset →
#     keeps the cross-sectional ordering meaningful.
def _build_feature_builder(cfg: dict, ps: PipelineState) -> FeatureBuilder:
    fcfg = cfg.get("features", {})
    mcfg = cfg.get("model", {})
    fb = FeatureBuilder(
        vp_bins          = fcfg.get("vp_bins", 30),
        vp_lookback      = fcfg.get("vp_lookback", 240),
        windows          = fcfg.get("windows", [5, 10, 20, 60]),
        lag_periods      = fcfg.get("lag_periods", 5),
        forecast_horizon = fcfg.get("forecast_horizon", 1),
        vp_stride        = fcfg.get("vp_stride", 1),
        frac_diff_d      = fcfg.get("frac_diff_d", 0.0),
        use_revin        = bool(mcfg.get("use_revin", False)),
    )
    # IT: Inietta lo stato dello scaler BTC (multi-colonna + clip + legacy per-col).
    # EN: Inject BTC scaler state (multi-column + clip + legacy per-col).
    fb.scaler             = ps.scaler
    fb._scale_cols        = list(ps.scale_cols)
    fb.scalers            = dict(ps.price_scaler_state)
    fb.clip_lo_           = ps.clip_lo_
    fb.clip_hi_           = ps.clip_hi_
    fb.feature_cols       = list(ps.feature_cols)
    fb.n_dynamic_features = ps.n_dynamic_features
    return fb


# IT: Normalizza il funding_df a open_time UTC ns-naive (FeatureBuilder._funding_features
#     reindexa per ffill 8h→1m: l'indice deve essere comparabile con df["open_time"]).
# EN: Normalize funding_df to open_time UTC ns-naive (FeatureBuilder._funding_features
#     reindexes via ffill 8h→1m: the index must be comparable to df["open_time"]).
def _prepare_funding(funding_parquet_path: str | Path) -> pd.DataFrame | None:
    p = Path(funding_parquet_path)
    if not p.exists():
        return None
    fd = pd.read_parquet(p)
    if "open_time" in fd.columns:
        ot = pd.to_datetime(fd["open_time"])
        if getattr(ot.dt, "tz", None) is not None:
            ot = ot.dt.tz_convert("UTC").dt.tz_localize(None)
        fd = fd.copy()
        fd["open_time"] = ot
    return fd


# IT: Modulo-singleton del modello/stato/builder — caricati una sola volta e riusati su
#     tutti i simboli (evita di ricaricare 3 archi per ogni asset).
# EN: Module-level singletons for model/state/builder — loaded once and reused across all
#     symbols (avoids reloading 3 archs per asset).
_MODEL: "EnsembleModel | None" = None
_PS: "PipelineState | None" = None
_CFG: "dict | None" = None
_DEVICE: "torch.device | None" = None
_CANONICAL: "tuple[str, ...] | None" = None
_FEAT_BUILDER: "FeatureBuilder | None" = None


# IT: Carica (una volta) ensemble eterogeneo + PipelineState BTC + builder + canonical.
#     L'ensemble è caricato via load_heterogeneous (stesso path di backtest/live), AMP off
#     in inference (gestito internamente da EnsembleModel.__call__).
# EN: Loads (once) heterogeneous ensemble + BTC PipelineState + builder + canonical names.
#     Ensemble loaded via load_heterogeneous (same path as backtest/live); AMP off in
#     inference (handled inside EnsembleModel.__call__).
def _ensure_loaded() -> None:
    global _MODEL, _PS, _CFG, _DEVICE, _CANONICAL, _FEAT_BUILDER
    if _MODEL is not None:
        return

    _CFG = load_config("config/default.yaml")
    # IT: La policy Spectral Norm deve matchare il training per caricare i checkpoint.
    # EN: Spectral-norm policy must match training to load checkpoints correctly.
    from quantsys.model import set_sn_on_mu_only
    set_sn_on_mu_only(bool(_CFG.get("training", {}).get("sn_on_mu_only", False)))

    _DEVICE = setup_device(_CFG)

    ps_path = ROOT / "models" / ARCH_DIR / "pipeline_state.pkl"
    if not ps_path.exists():
        raise FileNotFoundError(
            f"PipelineState non trovato: {ps_path}. Esegui prima il training di {ARCH_DIR}."
        )
    _PS = PipelineState.load(str(ps_path))

    # IT: Feature canoniche (104) = single source of truth dal NPZ — stesso ordine del training.
    # EN: Canonical features (104) = single source of truth from the NPZ — training order.
    _CANONICAL = get_canonical_feature_names(str(ROOT / "data" / "lstm_dataset.npz"))

    # IT: Ensemble eterogeneo (stesso loader di 03_backtest / 04_live).
    # EN: Heterogeneous ensemble (same loader as 03_backtest / 04_live).
    _MODEL = EnsembleModel.load_heterogeneous(_DEVICE, cfg=_CFG)
    _MODEL.eval()
    log.info(
        f"Ensemble ETEROGENEO caricato: {_MODEL.n_members} archi "
        f"[{', '.join(_MODEL.arch_names)}]  device={_DEVICE}  "
        f"target_scale={_PS.target_scale:.6f}"
    )

    _FEAT_BUILDER = _build_feature_builder(_CFG, _PS)


# IT: Inferenza per UN simbolo → pannello LONG [open_time, mu_raw, sigma_raw, fwd_ret_30].
# EN: Inference for ONE symbol → LONG panel [open_time, mu_raw, sigma_raw, fwd_ret_30].
def generate_mu_for_symbol(raw_parquet_path: str | Path,
                           funding_parquet_path: str | Path) -> pd.DataFrame:
    """
    IT: Applica l'ensemble BTC alle candele dell'asset producendo μ/σ raw su una griglia
        ogni h=30 candele (forward return non sovrapposti). `open_time` = timestamp della
        candela DECISIONALE (ultima della finestra 120). `fwd_ret_30` = somma dei prossimi
        30 log-close-return dalla decisione (NaN se <30 candele future disponibili).
    EN: Applies the BTC ensemble to the asset's candles, emitting raw μ/σ on a grid every
        h=30 candles (non-overlapping forward returns). `open_time` = DECISION timestamp
        (last candle of the 120-window). `fwd_ret_30` = sum of next 30 log-close-returns
        from the decision point (NaN where <30 future candles exist).

    Args:
        raw_parquet_path:     parquet OHLCV dell'asset (schema raw_candles: open_time,
                              open/high/low/close/volume/taker_buy_vol, ...).
        funding_parquet_path: parquet funding rate dell'asset (open_time, funding_rate).

    Returns:
        DataFrame[open_time, mu_raw, sigma_raw, fwd_ret_30] (open_time datetime64[ms, UTC]).
    """
    _ensure_loaded()
    assert _MODEL is not None and _PS is not None and _CANONICAL is not None
    assert _FEAT_BUILDER is not None and _DEVICE is not None

    cfg  = _CFG
    fcfg = cfg.get("features", {})
    mcfg = cfg.get("model", {})
    window_size = int(mcfg.get("window_size", 120))
    horizon     = int(fcfg.get("forecast_horizon", _PS.forecast_horizon))
    # IT: La griglia step = horizon → forward return NON sovrapposti (quasi-indipendenti).
    # EN: Grid step = horizon → NON-overlapping forward returns (quasi-independent).
    step = horizon

    # ── 1. Carica candele raw dell'asset ────────────────────────────────────────
    raw_path = Path(raw_parquet_path)
    df_raw = pd.read_parquet(raw_path)
    # IT: open_time → UTC ns-naive ordinato. FeatureBuilder._funding_features reindexa il
    #     funding (tz-naive) su df["open_time"], quindi entrambi devono essere tz-naive
    #     (stesso contratto del path live/replay, dove buf.to_dataframe è naive). Riconverto
    #     a UTC tz-aware solo nell'output finale (schema deliverato).
    # EN: open_time → ordered UTC ns-naive. FeatureBuilder._funding_features reindexes the
    #     (tz-naive) funding onto df["open_time"], so both must be tz-naive (same contract as
    #     live/replay, where buf.to_dataframe is naive). Re-add UTC tz only in the final output.
    df_raw = df_raw.copy()
    _ot = pd.to_datetime(df_raw["open_time"], utc=True)
    df_raw["open_time"] = _ot.dt.tz_convert("UTC").dt.tz_localize(None)
    df_raw = (df_raw.sort_values("open_time")
                    .drop_duplicates(subset="open_time")
                    .reset_index(drop=True))
    if len(df_raw) < window_size + horizon + 1:
        log.warning(
            f"{raw_path.name}: solo {len(df_raw)} candele (< {window_size}+{horizon}+1) — skip."
        )
        return pd.DataFrame(columns=["open_time", "mu_raw", "sigma_raw", "fwd_ret_30"])

    # IT: Serie raw di log-close-return — base per fwd_ret_30, ALLINEATA all'asse temporale
    #     completo (PRIMA del dropna del FeatureBuilder, che taglia le ultime h righe).
    # EN: Raw log-close-return series — basis for fwd_ret_30, aligned to the FULL timeline
    #     (BEFORE FeatureBuilder's dropna, which trims the last h rows).
    raw_log_ret = np.log(df_raw["close"].to_numpy(dtype=np.float64))
    raw_log_ret = np.diff(raw_log_ret, prepend=raw_log_ret[0])   # IT: ret[0]=0 | EN: ret[0]=0
    # IT: fwd_ret_30[t] = somma dei log-return su (t, t+h] = ret[t+1] + ... + ret[t+h].
    #     cumsum + differenza scorrevole, NaN dove mancano h candele future.
    # EN: fwd_ret_30[t] = sum of log-returns over (t, t+h] = ret[t+1] + ... + ret[t+h].
    #     cumsum + shifted difference, NaN where <h future candles exist.
    csum = np.concatenate([[0.0], np.cumsum(raw_log_ret)])        # IT: csum[k]=Σ ret[:k]
    n_raw = len(raw_log_ret)
    fwd_full = np.full(n_raw, np.nan, dtype=np.float64)
    valid_end = n_raw - horizon
    if valid_end > 0:
        idx = np.arange(valid_end)
        # IT: Σ ret[idx+1 .. idx+h] = csum[idx+h+1] − csum[idx+1].
        # EN: Σ ret[idx+1 .. idx+h] = csum[idx+h+1] − csum[idx+1].
        fwd_full[idx] = csum[idx + horizon + 1] - csum[idx + 1]
    # IT: Mappa open_time → forward return (lookup robusto a tagli/gap del feature frame).
    # EN: Map open_time → forward return (robust lookup against feature-frame trims/gaps).
    fwd_by_time = pd.Series(fwd_full, index=df_raw["open_time"].to_numpy())

    # ── 2. Feature engineering con scaler BTC (uniform transform, no fit) ────────
    fd = _prepare_funding(funding_parquet_path)
    if fd is None:
        log.warning(f"{raw_path.name}: funding parquet assente — feature funding_* = 0.")
    # IT: build(normalize=True, fit=False) → applica lo scaler BTC iniettato (NO re-fit) +
    #     clip, ESATTAMENTE come il path live/replay. dropna(target_ret) interno taglia le
    #     ultime h righe ma fwd_ret_30 è già calcolato sull'asse completo (lookup per tempo).
    # EN: build(normalize=True, fit=False) → applies the injected BTC scaler (NO re-fit) +
    #     clip, EXACTLY as live/replay. The internal dropna(target_ret) trims the last h rows
    #     but fwd_ret_30 is already computed on the full timeline (time-keyed lookup).
    feat_df = _FEAT_BUILDER.build(df_raw, normalize=True, fit=False, funding_df=fd)

    missing = set(_CANONICAL) - set(feat_df.columns)
    if missing:
        log.error(
            f"{raw_path.name}: FeatureBuilder non ha prodotto {len(missing)} feature "
            f"canoniche (es. {sorted(missing)[:5]}) — skip."
        )
        return pd.DataFrame(columns=["open_time", "mu_raw", "sigma_raw", "fwd_ret_30"])

    # IT: Matrice feature canoniche (ordine fisso) + asse temporale; scarta righe con NaN
    #     residui (rolling warmup iniziale) per evitare finestre con NaN nell'inferenza.
    # EN: Canonical feature matrix (fixed order) + timeline; drop residual-NaN rows (initial
    #     rolling warmup) to avoid NaN windows in inference.
    feat_df = feat_df.copy()
    feat_df["open_time"] = pd.to_datetime(feat_df["open_time"])   # IT: resta UTC ns-naive | EN: stays UTC ns-naive
    feat_cols = list(_CANONICAL)
    nan_mask = feat_df[feat_cols].isna().any(axis=1)
    feat_df = feat_df.loc[~nan_mask].reset_index(drop=True)
    if len(feat_df) < window_size:
        log.warning(
            f"{raw_path.name}: solo {len(feat_df)} righe feature valide (< window {window_size}) — skip."
        )
        return pd.DataFrame(columns=["open_time", "mu_raw", "sigma_raw", "fwd_ret_30"])

    feat_mat = feat_df[feat_cols].to_numpy(dtype=np.float32)
    times    = feat_df["open_time"].to_numpy()
    n_feat_rows = len(feat_mat)

    # ── 3. Griglia decisionale ogni `step` candele ──────────────────────────────
    # IT: La finestra che termina alla riga j usa feat_mat[j-window+1 : j+1]; la prima riga
    #     valida è j=window-1. Campioniamo j su una griglia step=h dalla coda all'indietro,
    #     così l'ultimo punto decisionale è sempre incluso (forward più recente).
    # EN: The window ending at row j uses feat_mat[j-window+1 : j+1]; first valid j=window-1.
    #     Sample j on a step=h grid from the tail backwards so the latest decision point is
    #     always included (most recent forward).
    last_j  = n_feat_rows - 1
    first_j = window_size - 1
    decision_js = np.arange(last_j, first_j - 1, -step)[::-1]    # IT: ordine cronologico
    if len(decision_js) == 0:
        return pd.DataFrame(columns=["open_time", "mu_raw", "sigma_raw", "fwd_ret_30"])

    # IT: Stack delle finestre (n_dec, window, n_feat) — view stride-tricks poi copia.
    # EN: Stack windows (n_dec, window, n_feat) — stride-tricks view then copy.
    windows = np.empty((len(decision_js), window_size, feat_mat.shape[1]), dtype=np.float32)
    for k, j in enumerate(decision_js):
        windows[k] = feat_mat[j - window_size + 1 : j + 1]

    # ── 4. Inferenza batch (ensemble eterogeneo, AMP off interno) ───────────────
    n_dec = len(windows)
    mu_z    = np.zeros(n_dec, dtype=np.float64)
    sigma_z = np.zeros(n_dec, dtype=np.float64)
    _MODEL.eval()
    with torch.no_grad():
        for b0 in range(0, n_dec, BATCH_SIZE):
            b1 = min(b0 + BATCH_SIZE, n_dec)
            xb = torch.tensor(windows[b0:b1], dtype=torch.float32).to(_DEVICE)
            # IT: Modello has_macro=False per questi archi → forward senza x_macro
            #     (l'ensemble eterogeneo non usa branch macro; coerente con backtest).
            # EN: has_macro=False for these archs → forward without x_macro (the
            #     heterogeneous ensemble has no macro branch; consistent with backtest).
            mu_b, sigma_b, _nu_b = _MODEL(xb)
            mu_z[b0:b1]    = mu_b.squeeze(-1).cpu().numpy()
            sigma_z[b0:b1] = sigma_b.squeeze(-1).cpu().numpy()

    # ── 5. Denormalizzazione z-score → raw (INVARIANTE CRITICA) ─────────────────
    # IT: Il modello predice in spazio z-score; denormalize_predictions riporta μ/σ in spazio
    #     RAW (frazione di log-return) col target_scale di BTC. Saltarla = bug più costoso.
    # EN: Model predicts in z-score space; denormalize_predictions converts μ/σ to RAW space
    #     (log-return fraction) with BTC's target_scale. Skipping it = costliest bug.
    mu_raw, sigma_raw = _PS.denormalize_predictions(mu_z, sigma_z)
    mu_raw    = np.asarray(mu_raw, dtype=np.float64)
    sigma_raw = np.asarray(sigma_raw, dtype=np.float64)

    # IT: Safety check — σ raw ≥ 5%/min è fisicamente impossibile: denorm mancata o scaler rotto.
    # EN: Safety check — raw σ ≥ 5%/min is physically impossible: missed denorm or broken scaler.
    if np.isfinite(sigma_raw).any() and np.nanmax(sigma_raw) >= 0.05:
        raise RuntimeError(
            f"σ post-denorm = {np.nanmax(sigma_raw):.4f} >= 0.05 su {raw_path.name}. "
            "Probabile mancata denormalizzazione (target_scale=1.0?) o scaler corrotto."
        )

    # ── 6. Allinea forward return alla griglia decisionale (lookup per open_time) ─
    decision_times = times[decision_js]
    fwd_ret_30 = fwd_by_time.reindex(decision_times).to_numpy(dtype=np.float64)

    out = pd.DataFrame({
        "open_time":  pd.to_datetime(decision_times, utc=True),
        "mu_raw":     mu_raw,
        "sigma_raw":  sigma_raw,
        "fwd_ret_30": fwd_ret_30,
    })
    return out


# IT: Loop principale — itera ogni data/xs/raw/{SYMBOL}.parquet, concatena in un pannello LONG.
#     Robusto a data/xs/raw/ non ancora popolata (download in parallelo): skip + log, exit 0.
# EN: Main loop — iterates every data/xs/raw/{SYMBOL}.parquet, concatenates into a LONG panel.
#     Robust to a not-yet-populated data/xs/raw/ (parallel download): skip + log, exit 0.
def main() -> int:
    xs_dir      = ROOT / "data" / "xs"
    raw_dir     = xs_dir / "raw"
    funding_dir = xs_dir / "funding"
    out_path    = xs_dir / "mu_panel.parquet"

    if not raw_dir.exists():
        log.warning(
            f"{raw_dir} non esiste ancora — il download cross-sectional è probabilmente "
            "in corso in parallelo. Nessun pannello prodotto (skip graceful)."
        )
        return 0

    raw_files = sorted(raw_dir.glob("*.parquet"))
    if not raw_files:
        log.warning(
            f"{raw_dir} è vuota — nessun {{SYMBOL}}.parquet trovato (download in corso?). "
            "Nessun pannello prodotto (skip graceful)."
        )
        return 0

    log.info(f"Panel-inference: {len(raw_files)} simboli trovati in {raw_dir}")
    frames: list[pd.DataFrame] = []
    n_ok, n_skip = 0, 0
    for raw_f in raw_files:
        symbol      = raw_f.stem
        funding_f   = funding_dir / f"{symbol}.parquet"
        try:
            df_sym = generate_mu_for_symbol(raw_f, funding_f)
        except Exception as e:
            log.error(f"  {symbol}: errore inferenza ({e}) — skip.")
            n_skip += 1
            continue
        if df_sym.empty:
            log.warning(f"  {symbol}: pannello vuoto — skip.")
            n_skip += 1
            continue
        df_sym = df_sym.copy()
        df_sym.insert(1, "symbol", symbol)
        frames.append(df_sym)
        n_ok += 1
        log.info(
            f"  {symbol}: {len(df_sym)} punti  "
            f"μ_raw[{np.nanmin(df_sym['mu_raw']):+.2e},{np.nanmax(df_sym['mu_raw']):+.2e}]  "
            f"σ_raw_med={np.nanmedian(df_sym['sigma_raw']):.4f}"
        )

    if not frames:
        log.warning("Nessun simbolo ha prodotto segnali — pannello non scritto.")
        return 0

    panel = pd.concat(frames, ignore_index=True)
    # IT: open_time → datetime64[ms, UTC] (schema deliverato).
    # EN: open_time → datetime64[ms, UTC] (delivered schema).
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True).astype("datetime64[ms, UTC]")
    panel = panel[["open_time", "symbol", "mu_raw", "sigma_raw", "fwd_ret_30"]]

    xs_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    log.info(
        f"Pannello scritto → {out_path}  ({len(panel):,} righe, {n_ok} simboli, "
        f"{n_skip} skip)  schema={list(panel.columns)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
