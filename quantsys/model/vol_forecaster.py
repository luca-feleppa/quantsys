# VolForecaster — the vol-paper line's forecast core (C2 2ter refactor,
# 2026-07-18: PROMOTED from scripts/04b_vol_paper.py, body UNCHANGED —
# bit-perfect A/B proof in STATUS). Consumers: 04b (VPS live) and
# vol_paper_replay (gap-filler). 04b's digit-leading filename made it
# importlib-only; from here the import is clean.
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quantsys.utils import PipelineState
from quantsys.utils.atomic_save import atomic_save_parquet
from quantsys.data import fetch_klines, fetch_klines_incremental, fetch_funding_rate
from quantsys.features import FeatureBuilder, canonical_feature_columns
from quantsys.model.ensemble import EnsembleModel

log = logging.getLogger("quantsys.model.vol_forecaster")

# legacy macro-snapshot mode: refits the MacroNormalizer whole-df at every
# bootstrap. Historical behavior, and it stays the DEFAULT.
MACRO_NORM_REFIT = "refit"


def macro_snapshot(df_macro: pd.DataFrame, macro_cols: list,
                   macro_norm: str = MACRO_NORM_REFIT) -> tuple:
    # the normalized last macro row, i.e. the live `x_macro` input. TWO modes:
    # - "refit" (DEFAULT, legacy): the MacroNormalizer is refitted whole-df on the
    #   current parquet. Known, measured consequence (2026-07-31 breakpoint):
    #   extending the parquet moves median and IQR, so the measuring INSTRUMENT
    #   changes together with the state of the world it is supposed to measure. On
    #   the 31/07 breakpoint that drift was 2.7% of the total variation;
    # - <path>: PINNED normalizer, loaded from disk and only APPLIED. The
    #   instrument stays at a declared vintage and only the state moves.
    # ⚠ Inert by construction: with no explicit path the legacy branch is
    # bit-identical, because it is literally the same code as before.
    # Returns (xm_np, info), info always populated for the caller's logging.
    from quantsys.macro.regime import MacroNormalizer

    last = df_macro[macro_cols].iloc[[-1]].fillna(0.0)
    if macro_norm == MACRO_NORM_REFIT:
        norm = MacroNormalizer()
        norm.fit_transform(df_macro, macro_cols)
        info = {"mode": MACRO_NORM_REFIT, "pinned_vintage": None}
    else:
        norm = MacroNormalizer.load(str(macro_norm))
        # fail-fast guard — the pin's columns must match the live parquet's
        # ORDER INCLUDED. Without it, an added or reordered macro column would
        # apply the WRONG column's median and IQR, silently, on every live tick.
        if list(norm.feature_cols) != list(macro_cols):
            raise RuntimeError(
                f"macro normalizer pinnato incompatibile / incompatible pinned macro "
                f"normalizer: {len(norm.feature_cols)} colonne pinnate vs "
                f"{len(macro_cols)} nel parquet, o ordine diverso / different order "
                f"({macro_norm})")
        info = {"mode": str(macro_norm),
                "pinned_vintage": getattr(norm, "pinned_vintage", None)}

    xm_np = np.clip(norm.scaler.transform(last.values.astype(np.float32)),
                    -5, 5).astype(np.float32)
    info["data_vintage"] = str(pd.Timestamp(df_macro.index[-1]).date())
    return xm_np, info


class VolForecaster:
    # replica of the parity-blessed FeatureAssembler wiring (04_live_signals):
    # FeatureBuilder configured from config + interval/scaler/columns INJECTED
    # from PipelineState → build(fit=False) identical to training. Candles
    # live in memory (parquet bootstrap + REST delta per tick).
    def __init__(self, cfg: dict, device, arch: str = "itransformer",
                 macro_norm: str = MACRO_NORM_REFIT):
        self.cfg = cfg
        self.device = device
        self.symbol = cfg["data"].get("symbol", "BTCUSDT")

        # model dir parametrized via --arch (default itransformer = bit-identical
        # legacy behavior); explicit CLI flag, NEVER QUANTSYS_ARCH (a stale env
        # would silently redirect the loading).
        self.arch = arch
        self.model_dir = Path("models") / arch
        log.info(f"dir modelli effettiva / effective model dir: {self.model_dir} (arch={arch})")

        ps = PipelineState.load(str(self.model_dir / "pipeline_state.pkl"))
        # ⚠ THE SCALER GUARD (`assert_model_dataset_scaler`) DOES NOT BELONG HERE,
        # and the reason must be written down because the temptation to add it is
        # strong after reading THEORY.md §12.2. That guard compares the model scaler
        # against the npz's CANONICAL one, and applies where a model is evaluated
        # on a dataset built by someone else (the judges). The live path is not
        # that case: features are computed on the fly by a FeatureBuilder whose
        # scaler and columns are INJECTED from THIS very `ps`, and the npz is never
        # read → self-consistent by construction. A fail-fast here would stop the
        # forward test at the next bootstrap over a mismatch that does not exist
        # live — and would stop open pre-registered samples.
        # config↔state contract guard (repo pattern: fail-fast on incoherent mixes).
        if str(cfg["data"]["interval"]) != str(ps.interval):
            raise RuntimeError(f"interval mismatch: config={cfg['data']['interval']} "
                               f"vs PipelineState={ps.interval}")
        idx = ps.scale_cols.index("target_ret")
        self.c = float(ps.scaler.center_[idx])
        self.s = float(ps.scaler.scale_[idx])
        # QLIKE-judge sanity: the log-RV center is ≈−7; ≈0 ⇒ stale directional state.
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
            # A4 HAR-CJ — same config as training (live↔training parity).
            use_har_cj       = bool(fcfg.get("har_cj", False)),
        )
        self.fb.scaler             = ps.scaler
        self.fb._scale_cols        = list(ps.scale_cols)
        self.fb.scalers            = dict(ps.price_scaler_state)
        self.fb.clip_lo_           = ps.clip_lo_
        self.fb.clip_hi_           = ps.clip_hi_
        self.fb.feature_cols       = list(ps.feature_cols)
        self.fb.n_dynamic_features = ps.n_dynamic_features
        # the canonical list is derived on the first forecast (same filters as 01:
        # exclude/C-funding/NaN/Inf over ps.feature_cols, builder order) and is
        # validated against n_features in the model's config.json — no npz needed.
        self._canonical: list | None = None
        mc = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.n_feat_expected = int(mc.get("n_features", 104))
        self.n_macro_expected = int(mc.get("n_macro", 0)) if mc.get("use_macro") else 0

        self.model = EnsembleModel.load(str(self.model_dir), device)
        log.info(f"Ensemble vol caricato: {getattr(self.model, 'n_members', '?')} membri | "
                 f"scaler target: center={self.c:.3f} scale={self.s:.3f} | h={self.h}")

        # macro from the on-disk parquet. The INSTRUMENT (MacroNormalizer) is
        # chosen by `macro_norm`: default "refit" = bit-identical legacy behavior;
        # a path = normalizer pinned to a declared vintage (see `macro_snapshot`).
        # The parameter is EXPLICIT and never read from env, same principle as
        # `arch`: a stale env would silently change the live input.
        self.xm = None
        self.macro_norm = macro_norm
        if self.n_macro_expected:
            df_macro = pd.read_parquet("data/macro_features.parquet")
            macro_cols = list(df_macro.columns)
            assert len(macro_cols) == self.n_macro_expected, \
                f"macro: {len(macro_cols)} colonne vs n_macro={self.n_macro_expected} del modello"
            xm_np, minfo = macro_snapshot(df_macro, macro_cols, macro_norm)
            self.xm = torch.tensor(xm_np, dtype=torch.float32).to(device)
            macro_date = pd.Timestamp(df_macro.index[-1])
            age_days = (pd.Timestamp.now(tz=getattr(macro_date, 'tz', None)) - macro_date).days
            # the two vintages are logged SEPARATELY because they are different
            # things: the instrument (normalizer) and the state (last row).
            # Conflating them is the confusion the 31/07 breakpoint had to
            # decompose by hand.
            pinned = minfo.get("pinned_vintage")
            log.info(f"macro snapshot: {len(macro_cols)} feature | stato/state: "
                     f"{macro_date.date()} ({age_days}g fa) | strumento/instrument: "
                     f"{'REFIT whole-df (legacy)' if minfo['mode'] == MACRO_NORM_REFIT else f'PINNATO vintage {pinned}'}")
            if age_days > 7:
                log.warning(f"macro_features.parquet vecchio di {age_days}g — "
                            f"valuta un refresh (01b, sezione macro)")

        # GAP-AWARE candle bootstrap (2026-07-18 fix): fetch_klines_incremental
        # downloads the FULL delta since the parquet's last candle (not just 48h)
        # and the updated parquet is re-persisted. The old bootstrap (parquet +
        # 48-candle REST delta) left a silent HOLE whenever the parquet was older
        # than 48h: with the A3/A8 freeze at 2026-06-22 the live T=120 window
        # contained ~72 June candles at every restart (bug found at v1-gate
        # closure). Persistence touches ONLY raw_candles.parquet: the frozen
        # npz dataset is NOT regenerated.
        raw_path = Path("data/raw_candles.parquet")
        n_disk = len(pd.read_parquet(raw_path, columns=["open_time"]))
        self.candles = fetch_klines_incremental(str(raw_path), self.symbol,
                                                cfg["data"]["interval"]) \
            .sort_values("open_time").reset_index(drop=True)
        if len(self.candles) > n_disk:
            raw_cols = ["open_time", "close_time", "open", "high", "low", "close",
                        "volume", "quote_vol", "trades", "taker_buy_vol",
                        "taker_buy_quote_vol"]
            atomic_save_parquet(self.candles[raw_cols], raw_path, index=False)
            log.info(f"raw_candles.parquet esteso/extended: {n_disk:,} → "
                     f"{len(self.candles):,} candele (gap-fill bootstrap)")
        self.funding = fetch_funding_rate(self.symbol,
                                          str(cfg["data"].get("start_time", "2019-01-01")),
                                          "data")
        log.info(f"bootstrap: {len(self.candles):,} candele 1h, {len(self.funding):,} funding obs")

    def _refresh_candles(self):
        # REST delta (48 candles cover any short gap), dedup merge, drop the
        # unfinished current candle. Does not rewrite the on-disk parquet.
        fresh = fetch_klines(self.symbol, self.cfg["data"]["interval"], limit=48)

        # everything tz-aware UTC (like the training path: raw parquet + funding).
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
        # one full forecast: candle refresh → features (full history, identical
        # to training) → (T,104) window → ensemble → full z→raw inversion.
        self._refresh_candles()
        # SAFETY NET (2026-07-18 fix) — the input window MUST be contiguous:
        # a hole in the series (stale parquet + 48h delta) silently produced
        # forecasts on weeks-old candles. Fail-fast.
        tail_ot = self.candles["open_time"].tail(self.window_size)
        # SECONDS-based comparison, no pd.Timedelta — on the VPS numpy the
        # constructor emits the "generic unit" DeprecationWarning (same class
        # as the 2026-07-16 01e fix: warning today, crash-loop tomorrow).
        gap_secs = tail_ot.diff().dropna().dt.total_seconds()
        if not (gap_secs == 60.0 * int(self.ps.interval_minutes)).all():
            raise RuntimeError(
                f"finestra candele NON contigua (gap max {gap_secs.max():.0f}s) — "
                f"serie bucata, forecast rifiutato / non-contiguous candle window — "
                f"holed series, forecast refused")
        # C1 (POST_GATE_V1) — PER-TICK funding refresh (delta handled inside
        # fetch_funding_rate, 0-1 requests): it used to be frozen at startup,
        # funding features went stale with uptime (replay-vs-live Δμ residual).
        # Fail-soft: on error keep the previous series (stale funding beats a
        # lost tick).
        try:
            self.funding = fetch_funding_rate(
                self.symbol, str(self.cfg["data"].get("start_time", "2019-01-01")),
                "data")
        except Exception as e:
            log.warning(f"refresh funding fallito/failed — uso serie precedente / "
                        f"keeping previous series: {type(e).__name__}: {e}")
        feat = self.fb.build(self.candles, fit=False, normalize=True,
                             funding_df=self.funding)
        if self._canonical is None:
            # shared canonical derivation (C2 2ter: same function as
            # 01_download/replay — the "drifting duplicated list" class is dead).
            # Count-validated against the model.
            cols = canonical_feature_columns(self.fb.feature_cols, feat)
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
        log_rv = mu_z * self.s + self.c          # FULL inversion
        rv_pred = float(np.exp(log_rv))          # 30h variance of the log-returns
        # trailing h-bar RV — diagnostic + naive-baseline input at analysis time.
        lr2 = np.log(self.candles["close"] / self.candles["close"].shift(1)) ** 2
        rv_trail = float(lr2.tail(self.h).sum())
        last_ts = pd.Timestamp(self.candles["open_time"].iloc[-1])
        return {"candle_ts": last_ts, "mu_z": mu_z, "log_rv": log_rv,
                "rv_pred": rv_pred, "rv_trail": rv_trail}
