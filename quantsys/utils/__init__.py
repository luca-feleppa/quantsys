"""Utilities: config loader, logging, device setup, pipeline state."""
import json
import logging
import logging.handlers
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


# Loads YAML config with selective merge of secrets + per-interval + per-arch override.
def load_config(path: str = "config/default.yaml", arch: str = None,
                interval: str = None) -> dict:
    """
    Loads the YAML configuration file.

    If config/secrets.yaml exists in the same folder, its keys are merged
    over the default (selective override, section by section).
    secrets.yaml is gitignored and never ends up on GitHub.

    If `interval` is not None, loads config/interval/{interval}.yaml (if present)
    and merges it section by section over default + secrets: it contains ONLY
    the keys that depend on candle resolution (stride, embargo, raw thresholds).

    If `arch` is not None, loads config/arch/{arch}.yaml (if present) and
    merges it section by section over default + secrets + interval, so that
    each architecture has its own isolated parameters without affecting the
    others. The arch stays the MOST specific override (applied last).
    A call without `arch`/`interval` is identical to the previous behavior.

    If `arch`/`interval` are None, reads the QUANTSYS_ARCH /
    QUANTSYS_INTERVAL env vars (set by run_all.py).
    """
    # Resolve `arch`/`interval` from env vars (run_all.py sets them for subprocesses).
    import os as _os
    if arch is None:
        arch = _os.environ.get("QUANTSYS_ARCH")  # None → no override
    if interval is None:
        interval = _os.environ.get("QUANTSYS_INTERVAL")  # None → no override

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # secrets.yaml is gitignored, selective override of default.yaml.
    secrets_path = Path(path).parent / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path, encoding="utf-8") as f:
            secrets = yaml.safe_load(f) or {}
        for section, values in secrets.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section] = {**cfg[section], **values}
            else:
                cfg[section] = values

    # Interval overlay (candle resolution) — AFTER secrets, BEFORE arch:
    # arch stays the most specific override. Per-section shallow merge,
    # identical to the arch overlay. Missing file → warning, continue.
    if interval is not None:
        interval_path = Path(path).parent / "interval" / f"{interval}.yaml"
        if interval_path.exists():
            with open(interval_path, encoding="utf-8") as f:
                interval_cfg = yaml.safe_load(f) or {}
            for section, values in interval_cfg.items():
                if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                    cfg[section] = {**cfg[section], **values}
                else:
                    cfg[section] = values
            logging.getLogger("quantsys").info(
                f"Interval override caricato: {interval_path}"
            )
        else:
            logging.getLogger("quantsys").warning(
                f"Interval override non trovato: {interval_path} — uso solo default.yaml"
            )

    if arch is not None:
        arch_path = Path(path).parent / "arch" / f"{arch}.yaml"
        if arch_path.exists():
            with open(arch_path, encoding="utf-8") as f:
                arch_cfg = yaml.safe_load(f) or {}
            for section, values in arch_cfg.items():
                if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                    cfg[section] = {**cfg[section], **values}
                else:
                    cfg[section] = values
            logging.getLogger("quantsys").info(
                f"Arch override caricato: {arch_path}"
            )
        else:
            logging.getLogger("quantsys").warning(
                f"Arch override non trovato: {arch_path} — uso solo default.yaml"
            )

    return cfg


# Credential sections that exist ONLY because load_config merges config/secrets.yaml
# over default.yaml; no consumer ever reads them back from a persisted config.
_SECRET_SECTIONS = ("vps", "alpaca", "deribit_testnet", "binance", "binance_testnet")

# Key names that carry a credential value wherever they happen to be nested.
_SECRET_KEY_RE = re.compile(
    r"(api_key|api_secret|secret|password|token|passphrase|credential)", re.IGNORECASE
)


# Returns a redacted DEEP COPY of cfg, safe to persist inside a pickle.
def redact_config(cfg: dict) -> dict:
    """
    Strips credentials out of a config before it gets serialized.

    load_config merges config/secrets.yaml over default.yaml, so the live cfg
    carries FRED/Alpaca/Deribit credentials and the VPS host. Persisting it
    verbatim writes those values into every models/**/pipeline_state.pkl.

    Two rules:
      · whole credential sections (`_SECRET_SECTIONS`) are dropped: they exist
        only because of the secrets merge and nothing reads them;
      · any key whose NAME matches `_SECRET_KEY_RE`, at any nesting depth, gets
        its value replaced by "<redacted>".

    The copy is DEEP on purpose: dicts and lists are rebuilt, never mutated, so
    the caller's live cfg still has its credentials after the call (the old
    `dict(cfg)` was shallow — redacting in place would have blanked the config
    of the running script). Non-credential keys (`dashboard.host`, the whole
    `data`/`features`/`model` sections) are left untouched.
    """
    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: ("<redacted>" if isinstance(k, str) and _SECRET_KEY_RE.search(k)
                    else _walk(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [_walk(v) for v in node]
        return node

    if not isinstance(cfg, dict):
        return _walk(cfg)
    # Drop the credential sections first, then rebuild every surviving level.
    return _walk({k: v for k, v in cfg.items() if k not in _SECRET_SECTIONS})


# Maps Binance candle interval → minutes. Single source of truth for the
# timeframe pivot: ALL temporal windows in the code derive from this.
_INTERVAL_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                     "1h": 60, "2h": 120, "4h": 240, "6h": 360,
                     "8h": 480, "12h": 720, "1d": 1440}


# Extracts interval_minutes from cfg["data"]["interval"] (default 1 = legacy 1m).
# ValueError on unknown intervals: fail-fast beats silently wrong windows.
def interval_minutes_from_cfg(cfg: dict) -> int:
    interval = str(cfg.get("data", {}).get("interval", "1m"))
    if interval not in _INTERVAL_MINUTES:
        raise ValueError(
            f"data.interval '{interval}' non riconosciuto / unknown — "
            f"validi/valid: {sorted(_INTERVAL_MINUTES)}"
        )
    return _INTERVAL_MINUTES[interval]


# Picks CUDA or CPU based on config/hardware and enables cudnn.benchmark.
def setup_device(cfg: dict) -> torch.device:
    """
    Selects the optimal device based on config and available hardware.
    RTX 2070 Super → CUDA 12.1, VRAM 8 GB.
    """
    # cudnn.benchmark autotunes kernels for fixed shapes (training).
    requested = cfg.get("hardware", {}).get("device", "cuda")

    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu    = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.getLogger("quantsys").info(f"GPU: {gpu}  VRAM: {vram:.1f} GB")
        if cfg.get("hardware", {}).get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        if requested == "cuda":
            logging.getLogger("quantsys").warning(
                "CUDA richiesto ma non disponibile — uso CPU."
            )
    return device


# Configures logging to stdout + rotating file (max 50 MB total).
def setup_logging(level: int = logging.INFO, log_dir: str = "logs") -> None:
    """
    Configures logging to stdout + rotating file.

    RotatingFileHandler: max 10 MB per file, 5 backup files.
    Total maximum: 50 MB of logs on disk, then the oldest is overwritten.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fmt       = "%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s"
    datefmt   = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    # stdout for live visibility; rotating file for historical audit.
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"quantsys_{ts}.log"
    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes   = 10 * 1024 * 1024,   # 10 MB/file
        backupCount= 5,                   # ≤50 MB total
        encoding   = "utf-8",
    )
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logging.getLogger("quantsys").info(f"Log file: {log_path} (rotating, max 50 MB)")


# Creates the given directories if missing (idempotent recursive mkdir).
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# models root — env override QUANTSYS_MODELS_ROOT for ISOLATED EXPERIMENTS.
# Default "models" = byte-identical behavior. Lets a distill (or any train) run on
# an isolated dir (e.g. `models_exp/`) WITHOUT overwriting a LIVE model — typically
# `models/itransformer/` used by the vol-paper forward test (04b). The whole
# train→distill→judge path derives its base from here: set the env before the
# command and read/write stay consistent within the isolated sandbox.
def models_root() -> Path:
    return Path(os.environ.get("QUANTSYS_MODELS_ROOT", "models"))


# path of the CANONICAL PipelineState (written by `01_download_data` at dataset
# build time, arch-independent). It is NOT `models_root()/pipeline_state.pkl`:
# under `QUANTSYS_MODELS_ROOT` that resolves inside the sandbox, where no
# canonical exists — so the identity guard degraded to "not verifiable" in
# exactly the mode where candidates are judged, i.e. its only use case (found on
# 2026-08-04 during R1). Precedence: sandbox-local canonical if present (an
# experiment that built its OWN dataset inside the sandbox, via
# `QUANTSYS_DATASET_NPZ`), otherwise the default-root canonical.
def canonical_state_path() -> Path:
    local = models_root() / "pipeline_state.pkl"
    return local if local.exists() else Path("models") / "pipeline_state.pkl"


# dataset npz path — env override QUANTSYS_DATASET_NPZ for ISOLATED EXPERIMENTS
# (DVOL-as-feature probe, 2026-07-17 pre-reg). Default "data/lstm_dataset.npz" =
# byte-identical behavior. Consumers: 02_train.py + scripts/vol/dev_vols_qlike.py
# (train and judge must read the SAME npz: set the env before both).
def dataset_npz_path() -> Path:
    return Path(os.environ.get("QUANTSYS_DATASET_NPZ", "data/lstm_dataset.npz"))


# PipelineState — unified container for scalers + config + metadata.

class PipelineState:
    """
    Fix 6 — Unified container for the whole pipeline state.

    Previous problem:
      Price-feature scalers were handled by the FeatureBuilder, macro-feature
      scalers by the MacroNormalizer, and the model config by a separate JSON.
      At inference all of them had to be loaded manually, remembering the
      column order.

    Solution:
      PipelineState wraps everything in a single serialized file (.pkl):
        · price_scaler_state  — dict {col: RobustScaler} from the FeatureBuilder
        · macro_normalizer    — full MacroNormalizer
        · feature_cols        — ordered list of price features
        · macro_feature_cols  — ordered list of macro features
        · model_config        — dict with n_features, n_macro, etc.
        · training_config     — copy of the config used in training (credentials redacted)

    At inference (04_live_signals.py, backtest) it is enough to:
        state = PipelineState.load("models/pipeline_state.pkl")
        # All scalers and columns are already there, in the right order.
    """

    # Initializes all slots empty; populated via from_* methods.
    def __init__(self):
        self.price_scaler_state:  dict          = {}
        # multi-column scaler (current format)
        self.scaler:              Optional[Any]  = None
        self.scale_cols:          list[str]      = []
        self.macro_normalizer:    Optional[Any]  = None
        self.feature_cols:        list[str]      = []
        self.macro_feature_cols:  list[str]      = []
        self.model_config:        dict           = {}
        self.training_config:     dict           = {}
        self.n_dynamic_features:  Optional[int]  = None  # dual-stream split
        self.clip_lo_:            Optional[Any]  = None  # 0.1% percentile
        self.clip_hi_:            Optional[Any]  = None  # 99.9% percentile
        self.created_at:          str            = datetime.now().isoformat()
        # Dataset metadata (e.g. 2M candles) for traceability.
        self.dataset_start:       Optional[str]  = None
        self.dataset_end:         Optional[str]  = None
        self.n_train_samples:     Optional[int]  = None
        self.interval:            Optional[str]  = None
        # fingerprint of the MACRO VINTAGE of the npz the model was trained on
        # (M1, 2026-08-05). `None` on pre-M1 models → "not verifiable", which is
        # not "verified equal". `_source` separates "measured" (computed from the
        # npz during training) from "declared" (documentary backfill, not
        # provable): a backfill passed off as a measurement would be an inference
        # written as if it were data.
        self.macro_vintage_fp:        Optional[dict] = None
        self.macro_vintage_fp_source: Optional[str]  = None

    # Copies scalers + dual-stream config from the FeatureBuilder.
    def from_feature_builder(self, builder: Any) -> "PipelineState":
        """Copies scaler state and dual-stream configuration from the FeatureBuilder."""
        # Current format: single multi-column scaler (efficient).
        if getattr(builder, "scaler", None) is not None:
            self.scaler      = builder.scaler
            self.scale_cols  = list(getattr(builder, "_scale_cols", []))
        # Legacy per-column format (backward compat with old pkl files).
        self.price_scaler_state  = builder.scalers.copy()
        self.feature_cols        = list(builder.feature_cols)
        self.n_dynamic_features  = getattr(builder, "n_dynamic_features", None)
        self.clip_lo_            = getattr(builder, "clip_lo_", None)
        self.clip_hi_            = getattr(builder, "clip_hi_", None)
        return self

    # Copies the MacroNormalizer and the macro column ordering.
    def from_macro_normalizer(self, normalizer: Any) -> "PipelineState":
        """Copies the MacroNormalizer."""
        self.macro_normalizer   = normalizer
        self.macro_feature_cols = list(normalizer.feature_cols)
        return self

    # Stores the model config (n_features, n_macro, etc.).
    def set_model_config(self, cfg: dict) -> "PipelineState":
        self.model_config = dict(cfg)
        return self

    # Stores a redacted copy of the config used during training.
    def set_training_config(self, cfg: dict) -> "PipelineState":
        # redact_config: deep copy without credentials — the pkl is a build
        # artifact, cfg here is default.yaml already merged with secrets.yaml.
        self.training_config = redact_config(cfg)
        return self

    # Saves dataset metadata (time span, n_train, frequency).
    def set_dataset_info(self, df_feat: Any, n_train: int) -> "PipelineState":
        """
        Fix 3 — Saves dataset metadata into the PipelineState.

        With a 2M-candle dataset it is useful to see at a glance:
          · the time span of the training set
          · how many samples the training set has
          · the data frequency

        Called by scripts/01_download_data.py after window creation,
        before saving the PipelineState to disk.

        Parameters:
          df_feat : DataFrame with an "open_time" column (after feature engineering)
          n_train : number of samples in the training set (after create_windows)
        """
        try:
            import pandas as pd
            ot = df_feat["open_time"] if "open_time" in df_feat.columns else None
            if ot is not None and len(ot) > 0:
                self.dataset_start    = str(ot.iloc[0].date())
                # train_end_idx = last sample in the training set.
                train_end_idx         = min(n_train, len(ot) - 1)
                self.dataset_end      = str(ot.iloc[train_end_idx].date())
            self.n_train_samples  = n_train
            # Try to read the interval from training_config
            self.interval = (
                self.training_config.get("data", {}).get("interval", None)
                if self.training_config else None
            )
        except Exception as e:
            logging.getLogger("quantsys.utils").warning(
                f"set_dataset_info: impossibile salvare i metadati del dataset ({e})"
            )
        return self

    # TRAINING candle interval in minutes — part of the train↔inference contract.
    # Consumers (live/replay/backtest) must use THIS, not the current config:
    # a 1m-trained model with a 1h config is an invalid combination
    # (TIME-semantic windows diverge). Fallback 1 = legacy pre-pivot pkl (1m).
    @property
    def interval_minutes(self) -> int:
        interval = getattr(self, "interval", None)
        if interval is None:
            return 1
        return _INTERVAL_MINUTES.get(str(interval), 1)

    # RobustScaler scale factor to denormalize μ/σ (raw ↔ z-score).
    @property
    def target_scale(self) -> float:
        """
        Denormalization factor for the model's predicted μ/σ.

        target_ret is scaled by the global RobustScaler together with the other
        features → the model predicts in standardized space (z-score).
        To obtain predictions in raw space (log-return fraction):

            mu_raw = mu_z * state.target_scale
            sigma_raw = sigma_z * state.target_scale

        Needed because the trading layer (SignalGenerator + RiskManager) works
        in raw space: comparisons against min_expected_ret, max_sigma, SL/TP
        based on σ*price etc. Without denormalization SL/TP become
        macroscopic (σ_z=1 × price = 100% distance).

        Returns 1.0 if target_ret is not in scale_cols (safe fallback).
        """
        if self.scaler is None or not self.scale_cols:
            return 1.0
        try:
            idx = self.scale_cols.index("target_ret")
        except ValueError:
            return 1.0
        return float(self.scaler.scale_[idx])

    # Prediction horizon (candles) read from the persisted training_config.
    @property
    def forecast_horizon(self) -> int:
        """Prediction horizon in candles, read from the persisted training_config.

        Fix #22: blocks user error if it changes in config between training and backtest.
        The real config key lives under `features.forecast_horizon` (not
        `data.`). Falls back to 15 (legacy default) if absent, for backward compat.
        """
        try:
            tc = self.training_config or {}
            # Look under features first (current key), then data (legacy)
            v = tc.get("features", {}).get("forecast_horizon")
            if v is None:
                v = tc.get("data", {}).get("forecast_horizon", 15)
            return int(v)
        except (AttributeError, TypeError):
            return 15

    # Denormalizes μ/σ z-score → raw; no-op when the model uses RevIN.
    def denormalize_predictions(self, mu, sigma):
        """
        Converts μ/σ predictions from standardized space (z-score, direct model
        output) to raw space (log-return fraction, expected by the trading layer).

        Works on Python scalars (float), np.ndarray (batch) and torch.Tensor —
        preserves the input type. Idempotent with target_scale=1.0
        (safe fallback if target_ret is not in scale_cols).

        When the model uses RevIN (`training_config.model.use_revin=True`), μ is
        already denormalized internally by the RevIN module (see
        `quantsys/model/__init__.py` revin.denormalize_mu). In that case applying
        `target_scale` again produces double denormalization → hypertrophic
        signals. We return (mu, sigma) unchanged.

        Returns:
            (mu_raw, sigma_raw) in the same type as the input.
        """
        use_revin = bool(
            self.training_config.get("model", {}).get("use_revin", False)
            if self.training_config else False
        )
        if use_revin:
            return mu, sigma
        scale = self.target_scale
        return mu * scale, sigma * scale

    # ── Serialization ───────────────────────────────────────────────────────────

    # Atomically serializes the state to disk (never corrupted).
    def save(self, path: str) -> None:
        """Saves the state atomically (tmp + rename → never corrupted)."""
        from quantsys.utils.atomic_save import atomic_save_pkl
        atomic_save_pkl(self, path)
        size_kb = Path(path).stat().st_size // 1024
        logging.getLogger("quantsys.utils").info(
            f"PipelineState salvato → {path}  ({size_kb} KB)"
        )

    # Reloads the pipeline state from a pickle file (for inference/backtest).
    @classmethod
    def load(cls, path: str) -> "PipelineState":
        """Loads the state from a pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        logging.getLogger("quantsys.utils").info(
            f"PipelineState caricato da {path}  "
            f"({len(state.feature_cols)} price features, "
            f"{len(state.macro_feature_cols)} macro features)"
        )
        return state

    # ── Transformation ────────────────────────────────────────────────────────

    # Applies price-feature scalers (handles legacy + current format).
    def transform_price(self, df: Any) -> Any:
        """
        Applies the price-feature scalers to a DataFrame.
        If a column is absent from the df (e.g. a feature added after training),
        logs a warning instead of crashing silently.

        Supports two formats:
          · New (multi-column self.scaler): a single RobustScaler over the whole matrix.
          · Old (self.price_scaler_state dict): one RobustScaler per column.
        """
        import logging as _log
        import numpy as _np
        log = _log.getLogger("quantsys.utils.pipeline")
        result = df.copy()

        # Multi-column path: rebuilds sub-scaler from sliced center/scale.
        if getattr(self, "scaler", None) is not None and getattr(self, "scale_cols", []):
            cols    = self.scale_cols
            present = [c for c in cols if c in result.columns]
            missing = [c for c in cols if c not in result.columns]

            if present:
                # O(1) column lookup via dict instead of cols.index in a loop (A9).
                idx_map = {c: i for i, c in enumerate(cols)}
                col_idx = [idx_map[c] for c in present]
                X = result[present].values.astype(_np.float64)
                nan_mask = _np.isnan(X)
                X_imp = _np.where(nan_mask, 0.0, X)

                # RobustScaler.transform = (X − center_)/scale_ → apply directly, no object rebuilt
                # per call (A9). Bit-identical (with_centering/scaling=True).
                center = self.scaler.center_[col_idx]
                scale  = self.scaler.scale_[col_idx]
                X_scaled = (X_imp - center) / scale
                X_scaled[nan_mask] = _np.nan

                for i, col in enumerate(present):
                    result[col + "_scaled"] = X_scaled[:, i]

            for col in missing:
                result[col + "_scaled"] = 0.0

            if missing:
                log.warning(f"transform_price: {len(missing)} colonne mancanti → zeros: {missing[:5]}")
            return result

        # Legacy per-column path (backward compat with pre-refactor pkls).
        missing = []
        for col, scaler in self.price_scaler_state.items():
            scaled_col = col + "_scaled"
            if col not in result.columns:
                missing.append(col)
                result[scaled_col] = 0.0   # zero fallback (centered)
                continue
            mask = result[col].notna()
            if mask.any():
                result.loc[mask, scaled_col] = scaler.transform(
                    result.loc[mask, col].values.reshape(-1, 1)
                ).flatten()
        if missing:
            log.warning(f"transform_price: {len(missing)} colonne mancanti → zeros: {missing[:5]}")
        return result

    # Applies the MacroNormalizer; missing columns filled with zeros.
    def transform_macro(self, df: Any) -> Any:
        """
        Applies the MacroNormalizer to a DataFrame.
        Handles missing columns with zeros (robustness to new FRED series).
        """
        if self.macro_normalizer is None:
            raise RuntimeError("MacroNormalizer non presente nello stato.")
        # copy ONCE if any column is missing (was df.copy() per missing one) (A-minor).
        missing = [c for c in self.macro_feature_cols if c not in df.columns]
        if missing:
            df = df.copy()
            for col in missing:
                df[col] = 0.0
        return self.macro_normalizer.transform(df)

    # Compact repr with feature counts, stream type and dataset metadata.
    def __repr__(self) -> str:
        n_dyn    = getattr(self, "n_dynamic_features", None)
        n_struct = (len(self.feature_cols) - n_dyn) if n_dyn is not None else "?"
        stream   = f"dual({n_dyn}dyn+{n_struct}struct)" if n_dyn else "single"
        # Fix 3 — show dataset metadata if available
        ds_start = getattr(self, "dataset_start", None)
        ds_end   = getattr(self, "dataset_end",   None)
        n_train  = getattr(self, "n_train_samples", None)
        interval = getattr(self, "interval",       None)
        dataset_str = ""
        if ds_start and ds_end:
            n_str       = f"{n_train:,}" if n_train else "?"
            iv_str      = f"/{interval}" if interval else ""
            dataset_str = (
                f", dataset={ds_start}→{ds_end}{iv_str}"
                f", n_train={n_str}"
            )
        return (
            f"PipelineState("
            f"features={len(self.feature_cols)} [{stream}], "
            f"macro={len(self.macro_feature_cols)}, "
            f"created={self.created_at[:10]}"
            f"{dataset_str})"
        )



# ─────────────────── model↔dataset scaler identity ───────────────────
# THE SAFETY NET THAT WAS MISSING (2026-08-01). The manifesto validates
# train↔inference on `forecast_horizon` and `interval` but NOT on the scaler —
# and the scaler is what changes most often, since it is refit at every dataset
# rebuild. Concrete, already-happened consequence: `dev_vols_qlike.py` loads
# center/scale from the MODEL's `pipeline_state` and evaluates on the current
# npz, built with a different RobustScaler. Inputs come normalized under one
# scaler, the net was trained under another, μ is denormalized against a third
# reference. Nothing fails: out comes a plausible, slightly worse number, which
# is the worst way an error can present itself.
def scaler_fingerprint(state: "PipelineState") -> dict:
    # compact, comparable scaler fingerprint. Hashes cover the WHOLE
    # center_/scale_: comparing `target_scale` alone is not enough, since two
    # datasets can share the target scale and differ on the input features —
    # which is half of the mismatch.
    import hashlib
    import numpy as _np
    fp: dict = {"target_scale": None, "n_scale_cols": len(getattr(state, "scale_cols", []) or []),
                "center_md5": None, "scale_md5": None}
    try:
        fp["target_scale"] = float(state.target_scale)
    except Exception:
        pass
    sc = getattr(state, "scaler", None)
    for attr, key in (("center_", "center_md5"), ("scale_", "scale_md5")):
        v = getattr(sc, attr, None) if sc is not None else None
        if v is not None:
            fp[key] = hashlib.md5(_np.ascontiguousarray(v, dtype=_np.float64).tobytes()).hexdigest()
    return fp


def check_model_dataset_scaler(model_state: "PipelineState",
                               canonical_path: Optional[Path] = None) -> dict:
    # compares the model's scaler against the CANONICAL one, i.e. the one
    # written when the dataset was built (`canonical_state_path()`, arch- and
    # sandbox-independent). Returns a provenance dict — it does not raise: whether
    # to stop is the caller's decision, since a deliberate cross-vintage run is
    # legitimate as long as it is DECLARED.
    # `matches=None` means "not verifiable" (canonical absent, e.g. a clean
    # clone) and must never be conflated with "verified equal".
    canonical_path = Path(canonical_path) if canonical_path else canonical_state_path()
    out = {"model": scaler_fingerprint(model_state), "canonical": None,
           "canonical_path": str(canonical_path), "matches": None}
    if not canonical_path.exists():
        return out
    out["canonical"] = scaler_fingerprint(PipelineState.load(str(canonical_path)))
    out["matches"] = bool(out["model"] == out["canonical"])
    return out


# ──────────── model↔dataset MACRO VINTAGE identity (M1, 2026-08-05) ────────────
# THE SECOND VINTAGE AXIS, found on 2026-08-04. The guard above covers the PRICE
# RobustScaler and `target_scale`; MACRO normalization is not covered and cannot
# be, by the same mechanism, since it does not live in the canonical
# `PipelineState` (`01_download_data` writes it BEFORE `01b` runs; the
# `MacroNormalizer` ends up in `models/lstm/pipeline_state.pkl` for `01b`
# routing). The comparison is therefore model ↔ NPZ, not model ↔ canonical.
# Mechanism: `01b_download_macro.py` reloads the splits and replaces ONLY
# `X_macro_{split}`, leaving `X_*`, `y_*`, `t_*` untouched, and refits the
# `MacroNormalizer` WHOLE-DF — so extending the series shifts median and IQR and
# changes macro values for historical rows too. Macro is a model INPUT (90
# columns, embedding active) but enters no HAR baseline, which read RV/target
# only: hence the observed signature, the NN moving while baselines stay
# identical digit for digit. Measured gap on the published ratio: 0.0019 —
# invisible to `matches: true`, which only looks at prices.
def macro_fingerprint(npz: Any) -> Optional[dict]:
    # fingerprint of the macro block INSIDE the npz. It fingerprints the ARRAY the
    # model consumed, not the `MacroNormalizer` parameters: those are absent from
    # the canonical state, and comparing them against the model's own normalizer
    # would be circular. ⚠ M1's pre-registration fixed `X_macro_train` alone; here
    # ALL present splits are fingerprinted — strictly stronger, identical cost
    # (X_macro is 2D `(n, 90)` float32, not 3D like `X`), and it closes the corner
    # case where a refit leaves train unchanged at float32 precision but moves
    # val/test. dtype is part of the fingerprint: two arrays equal in value but of
    # different dtype are NOT the same input to the network.
    import hashlib
    import numpy as _np
    if isinstance(npz, (str, Path)):
        npz = _np.load(str(npz), allow_pickle=True)
    keys = list(getattr(npz, "files", None) or list(npz.keys()))
    splits = [k for k in ("X_macro_train", "X_macro_val", "X_macro_test") if k in keys]
    # no macro block → None = "absent", which is NOT "matches".
    if not splits:
        return None
    fp: dict = {"n_macro_features": None, "names_md5": None, "splits": {}}
    if "n_macro_features" in keys:
        try:
            fp["n_macro_features"] = int(_np.asarray(npz["n_macro_features"]).ravel()[0])
        except Exception:
            pass
    # macro column ORDER is part of the vintage: the same columns in a different
    # order are a different input to a positional embedding.
    if "macro_feature_names" in keys:
        names = [str(x) for x in _np.asarray(npz["macro_feature_names"]).ravel().tolist()]
        fp["names_md5"] = hashlib.md5("\x00".join(names).encode("utf-8")).hexdigest()
    for k in splits:
        a = _np.ascontiguousarray(npz[k])
        fp["splits"][k] = {"md5": hashlib.md5(a.tobytes()).hexdigest(),
                           "shape": list(a.shape), "dtype": str(a.dtype)}
    return fp


def check_model_dataset_macro(model_state: "PipelineState", npz_arrays: Any) -> dict:
    # compares the macro fingerprint RECORDED IN THE MODEL at training time with
    # the current npz's. Like the price guard it does not raise: it returns
    # provenance. ⚠ `matches=None` = "not verifiable" and must never become True.
    # Models trained before M1 carry no fingerprint: they stay `None` forever, and
    # that is a fact about their provenance, not a defect to paper over.
    out = {"model": getattr(model_state, "macro_vintage_fp", None),
           "model_fp_source": getattr(model_state, "macro_vintage_fp_source", None) or None,
           "dataset": None, "matches": None}
    if npz_arrays is None:
        return out
    out["dataset"] = macro_fingerprint(npz_arrays)
    if out["model"] is None or out["dataset"] is None:
        return out
    out["matches"] = bool(out["model"] == out["dataset"])
    return out


def assert_model_dataset_scaler(model_state: "PipelineState", *, model_dir: Any,
                                arch: str = "", npz: Any = "",
                                allow_mismatch: bool = False,
                                npz_arrays: Any = None,
                                allow_macro_mismatch: bool = False,
                                logger: Any = None) -> dict:
    # check + log + raise in one place, returning the provenance block to be
    # written into the report. It exists as a single function because the call
    # sites are four judges with identical wiring: duplicating the block would
    # mean the fifth one forgets it, and "the check missing in exactly one
    # place" is what produced the wrong number.
    # ⚠ Do NOT call it on the LIVE path. `VolForecaster` and `FeatureAssembler`
    # compute features on the fly, injecting scaler and columns from the
    # MODEL's PipelineState: they are self-consistent by construction and never
    # read the npz, so they have no such exposure. A fail-fast here would stop
    # the forward test at the next bootstrap over a mismatch that does not
    # exist live.
    log = logger or logging.getLogger("quantsys.utils")
    prov = check_model_dataset_scaler(model_state)
    prov.update({"arch": arch, "model_dir": str(model_dir), "npz": str(npz),
                 "allow_scaler_mismatch": bool(allow_mismatch)})
    if prov["matches"] is None:
        log.warning(f"scaler canonico assente ({prov['canonical_path']}): identita' "
                    f"modello<->dataset NON verificabile / canonical scaler absent: "
                    f"model<->dataset identity NOT verifiable")
        return _assert_macro_vintage(model_state, prov, npz_arrays=npz_arrays,
                                     model_dir=model_dir, npz=npz,
                                     allow_mismatch=allow_macro_mismatch, log=log)
    if prov["matches"]:
        log.info(f"scaler modello<->dataset: IDENTICO (target_scale="
                 f"{prov['model']['target_scale']}) / model<->dataset scaler: IDENTICAL")
        return _assert_macro_vintage(model_state, prov, npz_arrays=npz_arrays,
                                     model_dir=model_dir, npz=npz,
                                     allow_mismatch=allow_macro_mismatch, log=log)
    msg = (f"SCALER MISMATCH modello<->dataset — il modello in {model_dir} e' stato "
           f"addestrato sotto uno scaler diverso da quello con cui e' stato costruito {npz}.\n"
           f"  modello  : target_scale={prov['model']['target_scale']} "
           f"center_md5={(prov['model']['center_md5'] or '?')[:12]}\n"
           f"  canonico : target_scale={prov['canonical']['target_scale']} "
           f"center_md5={(prov['canonical']['center_md5'] or '?')[:12]}\n"
           f"  Il numero che ne uscirebbe NON e' confrontabile con gli altri report: la "
           f"differenza include un artefatto di scaler, non solo skill.\n"
           f"  Riaddestra il modello sull'npz corrente, oppure dichiara l'incomparabilita' "
           f"con --allow-scaler-mismatch.")
    if not allow_mismatch:
        raise RuntimeError(msg)
    log.warning(msg)
    log.warning("--allow-scaler-mismatch attivo: report marcato NON confrontabile / "
                "active: report flagged NOT comparable")
    return _assert_macro_vintage(model_state, prov, npz_arrays=npz_arrays,
                                 model_dir=model_dir, npz=npz,
                                 allow_mismatch=allow_macro_mismatch, log=log)


def _assert_macro_vintage(model_state: "PipelineState", prov: dict, *, npz_arrays: Any,
                          model_dir: Any, npz: Any, allow_mismatch: bool, log: Any) -> dict:
    # the guard's second stage (M1). It ALWAYS runs, even when the scaler is
    # "not verifiable" or has been declared incomparable: the two vintage axes
    # are independent and switching one off must not switch off the other.
    # It enriches `prov` with the `macro` block and fails fast on a mismatch.
    # ⚠ If `npz_arrays` is None the check was not requested by the caller: it
    # stays `None` and the report says so — never silence, never True.
    m = check_model_dataset_macro(model_state, npz_arrays)
    m["allow_macro_mismatch"] = bool(allow_mismatch)
    prov["macro"] = m
    if m["matches"] is None:
        if npz_arrays is not None:
            why = ("il modello non porta l'impronta macro (addestrato prima di M1)"
                   if m["model"] is None else "l'npz non contiene un blocco macro")
            log.warning(f"vintage macro NON verificabile: {why} / macro vintage NOT "
                        f"verifiable — provenance recorded as null, not as a match")
        return prov
    if m["matches"]:
        log.info(f"vintage macro modello<->dataset: IDENTICO "
                 f"(n_macro={m['dataset'].get('n_macro_features')}, "
                 f"fonte impronta={m['model_fp_source'] or '?'}) / macro vintage: IDENTICAL")
        return prov
    _md = (m["model"] or {}).get("splits", {}).get("X_macro_train", {}).get("md5") or "?"
    _dd = (m["dataset"] or {}).get("splits", {}).get("X_macro_train", {}).get("md5") or "?"
    msg = (f"MACRO VINTAGE MISMATCH modello<->dataset — il modello in {model_dir} e' stato "
           f"addestrato su un blocco macro diverso da quello dentro {npz}.\n"
           f"  modello  : X_macro_train md5={_md[:12]} (fonte impronta: "
           f"{m['model_fp_source'] or '?'})\n"
           f"  dataset  : X_macro_train md5={_dd[:12]}\n"
           f"  La macro e' INPUT del modello ma non entra in nessuna baseline HAR: il NN si\n"
           f"  sposta e le baseline restano identiche cifra per cifra, quindi il confronto\n"
           f"  sembra sano ed e' cross-vintage. Scarto misurato su un caso reale: 0.0019 sul\n"
           f"  rapporto pubblicato.\n"
           f"  Riaddestra sull'npz corrente, oppure dichiara l'incomparabilita' con "
           f"--allow-macro-mismatch.")
    if not allow_mismatch:
        raise RuntimeError(msg)
    log.warning(msg)
    log.warning("--allow-macro-mismatch attivo: report marcato NON confrontabile / "
                "active: report flagged NOT comparable")
    return prov
