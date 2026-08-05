"""Utilities: config loader, logging, device setup, pipeline state."""
import json
import logging
import logging.handlers
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


# IT: Carica YAML config con merge selettivo di secrets + override per-interval + per-arch.
# EN: Loads YAML config with selective merge of secrets + per-interval + per-arch override.
def load_config(path: str = "config/default.yaml", arch: str = None,
                interval: str = None) -> dict:
    """
    Carica il file YAML di configurazione.

    Se esiste config/secrets.yaml nella stessa cartella, le sue chiavi
    vengono fuse sopra il default (override selettivo, sezione per sezione).
    Il file secrets.yaml è gitignored e non finisce mai su GitHub.

    Se `interval` non è None, carica config/interval/{interval}.yaml (se esiste)
    e lo fonde sezione per sezione sopra default + secrets: contiene SOLO le
    chiavi dipendenti dalla risoluzione candela (stride, embargo, soglie raw).

    Se `arch` non è None, carica config/arch/{arch}.yaml (se esiste) e lo
    fonde sezione per sezione sopra default + secrets + interval, in modo che
    ogni architettura abbia i propri parametri isolati senza influenzare le
    altre. L'arch resta l'override PIÙ specifico (applicato per ultimo).
    Chiamata senza `arch`/`interval` è identica al comportamento precedente.

    Se `arch`/`interval` sono None, legge dalle env var QUANTSYS_ARCH /
    QUANTSYS_INTERVAL (impostate da run_all.py).
    """
    # IT: Risolve `arch`/`interval` da env var (run_all.py le setta per i subprocess).
    # EN: Resolve `arch`/`interval` from env vars (run_all.py sets them for subprocesses).
    import os as _os
    if arch is None:
        arch = _os.environ.get("QUANTSYS_ARCH")  # IT: None → no override | EN: None → no override
    if interval is None:
        interval = _os.environ.get("QUANTSYS_INTERVAL")  # IT: None → no override | EN: None → no override

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # IT: secrets.yaml è gitignored, override selettivo del default.yaml.
    # EN: secrets.yaml is gitignored, selective override of default.yaml.
    secrets_path = Path(path).parent / "secrets.yaml"
    if secrets_path.exists():
        with open(secrets_path, encoding="utf-8") as f:
            secrets = yaml.safe_load(f) or {}
        for section, values in secrets.items():
            if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                cfg[section] = {**cfg[section], **values}
            else:
                cfg[section] = values

    # IT: Overlay interval (risoluzione candela) — DOPO secrets, PRIMA dell'arch:
    #     l'arch resta l'override più specifico. Merge shallow per-sezione,
    #     identico all'overlay arch. File mancante → warning, si prosegue.
    # EN: Interval overlay (candle resolution) — AFTER secrets, BEFORE arch:
    #     arch stays the most specific override. Per-section shallow merge,
    #     identical to the arch overlay. Missing file → warning, continue.
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


# IT: Mappa intervallo candela Binance → minuti. Single source of truth per il
#     pivot timeframe: TUTTE le finestre temporali del codice derivano da qui.
# EN: Maps Binance candle interval → minutes. Single source of truth for the
#     timeframe pivot: ALL temporal windows in the code derive from this.
_INTERVAL_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                     "1h": 60, "2h": 120, "4h": 240, "6h": 360,
                     "8h": 480, "12h": 720, "1d": 1440}


# IT: Estrae interval_minutes da cfg["data"]["interval"] (default 1 = legacy 1m).
#     ValueError su intervalli sconosciuti: meglio fail-fast che finestre silenti errate.
# EN: Extracts interval_minutes from cfg["data"]["interval"] (default 1 = legacy 1m).
#     ValueError on unknown intervals: fail-fast beats silently wrong windows.
def interval_minutes_from_cfg(cfg: dict) -> int:
    interval = str(cfg.get("data", {}).get("interval", "1m"))
    if interval not in _INTERVAL_MINUTES:
        raise ValueError(
            f"data.interval '{interval}' non riconosciuto / unknown — "
            f"validi/valid: {sorted(_INTERVAL_MINUTES)}"
        )
    return _INTERVAL_MINUTES[interval]


# IT: Sceglie CUDA o CPU in base a config/hardware e abilita cudnn.benchmark.
# EN: Picks CUDA or CPU based on config/hardware and enables cudnn.benchmark.
def setup_device(cfg: dict) -> torch.device:
    """
    Seleziona il device ottimale in base alla config e all'hardware disponibile.
    RTX 2070 Super → CUDA 12.1, VRAM 8 GB.
    """
    # IT: cudnn.benchmark autotuna i kernel per shape fisse (training).
    # EN: cudnn.benchmark autotunes kernels for fixed shapes (training).
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


# IT: Configura logging su stdout + file rotante (max 50 MB totali).
# EN: Configures logging to stdout + rotating file (max 50 MB total).
def setup_logging(level: int = logging.INFO, log_dir: str = "logs") -> None:
    """
    Configura logging su stdout + file rotante.

    RotatingFileHandler: max 10 MB per file, 5 file di backup.
    Totale massimo: 50 MB di log su disco, poi il più vecchio viene sovrascritto.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fmt       = "%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s"
    datefmt   = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        return

    # IT: stdout per visibilità live; file rotante per audit storico.
    # EN: stdout for live visibility; rotating file for historical audit.
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"quantsys_{ts}.log"
    fh = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes   = 10 * 1024 * 1024,   # IT: 10 MB/file | EN: 10 MB/file
        backupCount= 5,                   # IT: ≤50 MB totali | EN: ≤50 MB total
        encoding   = "utf-8",
    )
    fh.setFormatter(formatter)
    root.addHandler(fh)

    logging.getLogger("quantsys").info(f"Log file: {log_path} (rotating, max 50 MB)")


# IT: Crea le directory passate se mancanti (mkdir ricorsivo idempotente).
# EN: Creates the given directories if missing (idempotent recursive mkdir).
def ensure_dirs(*paths: str) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# IT: root dei modelli — override via env QUANTSYS_MODELS_ROOT per ESPERIMENTI ISOLATI.
#     Default "models" = comportamento byte-identico. Permette di girare un distill
#     (o qualsiasi train) su una dir isolata (es. `models_exp/`) SENZA sovrascrivere un
#     modello LIVE — tipicamente `models/itransformer/` usato dal forward-test vol-paper
#     (04b). L'intero path train→distill→giudice deriva da qui la base: impostando l'env
#     prima del comando, scrittura e lettura restano coerenti nella sandbox isolata.
# EN: models root — env override QUANTSYS_MODELS_ROOT for ISOLATED EXPERIMENTS.
#     Default "models" = byte-identical behavior. Lets a distill (or any train) run on
#     an isolated dir (e.g. `models_exp/`) WITHOUT overwriting a LIVE model — typically
#     `models/itransformer/` used by the vol-paper forward test (04b). The whole
#     train→distill→judge path derives its base from here: set the env before the
#     command and read/write stay consistent within the isolated sandbox.
def models_root() -> Path:
    return Path(os.environ.get("QUANTSYS_MODELS_ROOT", "models"))


# IT: path del PipelineState CANONICO (scritto da `01_download_data` alla costruzione
#     del dataset, arch-independent). NON è `models_root()/pipeline_state.pkl`: sotto
#     `QUANTSYS_MODELS_ROOT` quella risolve dentro la sandbox, dove il canonico non
#     esiste — e il guard di identità degradava a "non verificabile" proprio nella
#     modalità in cui si giudicano i candidati, cioè il suo unico caso d'uso
#     (trovato il 2026-08-04 durante R1). Precedenza: canonico locale alla sandbox se
#     esiste (esperimento che ha costruito il PROPRIO dataset dentro la sandbox, con
#     `QUANTSYS_DATASET_NPZ`), altrimenti il canonico della root di default.
# EN: path of the CANONICAL PipelineState (written by `01_download_data` at dataset
#     build time, arch-independent). It is NOT `models_root()/pipeline_state.pkl`:
#     under `QUANTSYS_MODELS_ROOT` that resolves inside the sandbox, where no
#     canonical exists — so the identity guard degraded to "not verifiable" in
#     exactly the mode where candidates are judged, i.e. its only use case (found on
#     2026-08-04 during R1). Precedence: sandbox-local canonical if present (an
#     experiment that built its OWN dataset inside the sandbox, via
#     `QUANTSYS_DATASET_NPZ`), otherwise the default-root canonical.
def canonical_state_path() -> Path:
    local = models_root() / "pipeline_state.pkl"
    return local if local.exists() else Path("models") / "pipeline_state.pkl"


# IT: path del dataset npz — override via env QUANTSYS_DATASET_NPZ per ESPERIMENTI ISOLATI
#     (probe DVOL-come-feature, pre-reg 2026-07-17). Default "data/lstm_dataset.npz" =
#     comportamento byte-identico. Consumer: 02_train.py + scripts/vol/dev_vols_qlike.py
#     (train e giudice devono leggere lo STESSO npz: setta l'env prima di entrambi).
# EN: dataset npz path — env override QUANTSYS_DATASET_NPZ for ISOLATED EXPERIMENTS
#     (DVOL-as-feature probe, 2026-07-17 pre-reg). Default "data/lstm_dataset.npz" =
#     byte-identical behavior. Consumers: 02_train.py + scripts/vol/dev_vols_qlike.py
#     (train and judge must read the SAME npz: set the env before both).
def dataset_npz_path() -> Path:
    return Path(os.environ.get("QUANTSYS_DATASET_NPZ", "data/lstm_dataset.npz"))


# IT: PipelineState — contenitore unificato scaler + config + metadati.
# EN: PipelineState — unified container for scalers + config + metadata.

class PipelineState:
    """
    Fix 6 — Contenitore unificato per tutto lo stato della pipeline.

    Problema precedente:
      Gli scaler delle price features erano gestiti dal FeatureBuilder,
      quelli delle macro features dal MacroNormalizer, e la config del modello
      da un JSON separato. In inference bisognava caricarli manualmente tutti e
      ricordarsi l'ordine delle colonne.

    Soluzione:
      PipelineState wrappa tutto in un unico file serializzato (.pkl):
        · price_scaler_state  — dict {col: RobustScaler} del FeatureBuilder
        · macro_normalizer    — MacroNormalizer completo
        · feature_cols        — lista ordinata delle price features
        · macro_feature_cols  — lista ordinata delle macro features
        · model_config        — dizionario con n_features, n_macro, etc.
        · training_config     — copia della config usata in training

    In inference (04_live_signals.py, backtest) basta:
        state = PipelineState.load("models/pipeline_state.pkl")
        # Tutti gli scaler e le colonne sono già lì, nell'ordine giusto.
    """

    # IT: Inizializza tutti gli slot a vuoto; popolati via from_* methods.
    # EN: Initializes all slots empty; populated via from_* methods.
    def __init__(self):
        self.price_scaler_state:  dict          = {}
        # IT: scaler multi-colonna (formato attuale) | EN: multi-column scaler (current format)
        self.scaler:              Optional[Any]  = None
        self.scale_cols:          list[str]      = []
        self.macro_normalizer:    Optional[Any]  = None
        self.feature_cols:        list[str]      = []
        self.macro_feature_cols:  list[str]      = []
        self.model_config:        dict           = {}
        self.training_config:     dict           = {}
        self.n_dynamic_features:  Optional[int]  = None  # IT: split dual-stream | EN: dual-stream split
        self.clip_lo_:            Optional[Any]  = None  # IT: percentile 0.1% | EN: 0.1% percentile
        self.clip_hi_:            Optional[Any]  = None  # IT: percentile 99.9% | EN: 99.9% percentile
        self.created_at:          str            = datetime.now().isoformat()
        # IT: Metadati dataset (es. 2M candele) per traceability.
        # EN: Dataset metadata (e.g. 2M candles) for traceability.
        self.dataset_start:       Optional[str]  = None
        self.dataset_end:         Optional[str]  = None
        self.n_train_samples:     Optional[int]  = None
        self.interval:            Optional[str]  = None
        # IT: impronta del VINTAGE MACRO dell'npz su cui il modello è stato addestrato
        #     (M1, 2026-08-05). `None` sui modelli anteriori a M1 → "non verificabile",
        #     che non è "verificato uguale". `_source` distingue "measured" (calcolata
        #     dall'npz durante il training) da "declared" (backfill documentale, non
        #     dimostrabile): un backfill spacciato per misura sarebbe un'inferenza
        #     scritta come se fosse un dato.
        # EN: fingerprint of the MACRO VINTAGE of the npz the model was trained on
        #     (M1, 2026-08-05). `None` on pre-M1 models → "not verifiable", which is
        #     not "verified equal". `_source` separates "measured" (computed from the
        #     npz during training) from "declared" (documentary backfill, not
        #     provable): a backfill passed off as a measurement would be an inference
        #     written as if it were data.
        self.macro_vintage_fp:        Optional[dict] = None
        self.macro_vintage_fp_source: Optional[str]  = None

    # IT: Copia scaler + config dual-stream dal FeatureBuilder.
    # EN: Copies scalers + dual-stream config from the FeatureBuilder.
    def from_feature_builder(self, builder: Any) -> "PipelineState":
        """Copia lo stato degli scaler e la configurazione dual-stream dal FeatureBuilder."""
        # IT: Formato corrente: singolo scaler multi-colonna (efficiente).
        # EN: Current format: single multi-column scaler (efficient).
        if getattr(builder, "scaler", None) is not None:
            self.scaler      = builder.scaler
            self.scale_cols  = list(getattr(builder, "_scale_cols", []))
        # IT: Formato legacy per-colonna (backward compat con vecchi pkl).
        # EN: Legacy per-column format (backward compat with old pkl files).
        self.price_scaler_state  = builder.scalers.copy()
        self.feature_cols        = list(builder.feature_cols)
        self.n_dynamic_features  = getattr(builder, "n_dynamic_features", None)
        self.clip_lo_            = getattr(builder, "clip_lo_", None)
        self.clip_hi_            = getattr(builder, "clip_hi_", None)
        return self

    # IT: Copia il MacroNormalizer e l'ordine delle colonne macro.
    # EN: Copies the MacroNormalizer and the macro column ordering.
    def from_macro_normalizer(self, normalizer: Any) -> "PipelineState":
        """Copia il MacroNormalizer."""
        self.macro_normalizer   = normalizer
        self.macro_feature_cols = list(normalizer.feature_cols)
        return self

    # IT: Memorizza la config del modello (n_features, n_macro, ecc.).
    # EN: Stores the model config (n_features, n_macro, etc.).
    def set_model_config(self, cfg: dict) -> "PipelineState":
        self.model_config = dict(cfg)
        return self

    # IT: Memorizza una copia della config usata in training.
    # EN: Stores a copy of the config used during training.
    def set_training_config(self, cfg: dict) -> "PipelineState":
        self.training_config = dict(cfg)
        return self

    # IT: Salva metadati dataset (arco temporale, n_train, frequenza).
    # EN: Saves dataset metadata (time span, n_train, frequency).
    def set_dataset_info(self, df_feat: Any, n_train: int) -> "PipelineState":
        """
        Fix 3 — Salva metadati sul dataset nel PipelineState.

        Con dataset da 2M candele è utile sapere a colpo d'occhio:
          · l'arco temporale del training set
          · quanti campioni ha il training set
          · la frequenza dei dati

        Viene chiamato da scripts/01_download_data.py dopo la creazione
        delle windows, prima di salvare il PipelineState su disco.

        Parametri:
          df_feat : DataFrame con colonna "open_time" (dopo feature engineering)
          n_train : numero di campioni nel training set (dopo create_windows)
        """
        try:
            import pandas as pd
            ot = df_feat["open_time"] if "open_time" in df_feat.columns else None
            if ot is not None and len(ot) > 0:
                self.dataset_start    = str(ot.iloc[0].date())
                # IT: train_end_idx = ultimo campione del training set.
                # EN: train_end_idx = last sample in the training set.
                train_end_idx         = min(n_train, len(ot) - 1)
                self.dataset_end      = str(ot.iloc[train_end_idx].date())
            self.n_train_samples  = n_train
            # Prova a leggere l'intervallo dalla training_config
            self.interval = (
                self.training_config.get("data", {}).get("interval", None)
                if self.training_config else None
            )
        except Exception as e:
            logging.getLogger("quantsys.utils").warning(
                f"set_dataset_info: impossibile salvare i metadati del dataset ({e})"
            )
        return self

    # IT: Intervallo candela del TRAINING in minuti — parte del contratto train↔inference.
    #     I consumer (live/replay/backtest) devono usare QUESTO, non la config corrente:
    #     un modello addestrato a 1m con config a 1h è una combinazione invalida
    #     (finestre TIME-semantic divergono). Fallback 1 = legacy pkl pre-pivot (1m).
    # EN: TRAINING candle interval in minutes — part of the train↔inference contract.
    #     Consumers (live/replay/backtest) must use THIS, not the current config:
    #     a 1m-trained model with a 1h config is an invalid combination
    #     (TIME-semantic windows diverge). Fallback 1 = legacy pre-pivot pkl (1m).
    @property
    def interval_minutes(self) -> int:
        interval = getattr(self, "interval", None)
        if interval is None:
            return 1
        return _INTERVAL_MINUTES.get(str(interval), 1)

    # IT: Fattore di scala RobustScaler per denormalizzare μ/σ (raw ↔ z-score).
    # EN: RobustScaler scale factor to denormalize μ/σ (raw ↔ z-score).
    @property
    def target_scale(self) -> float:
        """
        Fattore di denormalizzazione per μ/σ predette dal modello.

        target_ret viene scalato dal RobustScaler globale insieme alle altre
        feature → il modello predice in spazio standardizzato (z-score).
        Per ottenere predizioni in spazio raw (frazione di log-return):

            mu_raw = mu_z * state.target_scale
            sigma_raw = sigma_z * state.target_scale

        Necessario perché trading layer (SignalGenerator + RiskManager) opera
        in spazio raw: confronti contro min_expected_ret, max_sigma, SL/TP
        basati su σ*price etc. Senza denormalizzazione SL/TP diventano
        macroscopici (σ_z=1 × price = 100% di distanza).

        Ritorna 1.0 se target_ret non è in scale_cols (fallback safe).
        """
        if self.scaler is None or not self.scale_cols:
            return 1.0
        try:
            idx = self.scale_cols.index("target_ret")
        except ValueError:
            return 1.0
        return float(self.scaler.scale_[idx])

    # IT: Orizzonte di predizione (candele) letto dal training_config persistito.
    # EN: Prediction horizon (candles) read from the persisted training_config.
    @property
    def forecast_horizon(self) -> int:
        """Orizzonte di predizione in candele, letto dal training_config persistito.

        Fix #22: blocca user error se cambia in config tra training e backtest.
        La chiave reale nel config sta sotto `features.forecast_horizon` (non
        `data.`). Fallback a 15 (legacy default) se non presente per backward compat.
        """
        try:
            tc = self.training_config or {}
            # Cerca prima sotto features (chiave attuale), poi data (legacy)
            v = tc.get("features", {}).get("forecast_horizon")
            if v is None:
                v = tc.get("data", {}).get("forecast_horizon", 15)
            return int(v)
        except (AttributeError, TypeError):
            return 15

    # IT: Denormalizza μ/σ z-score → raw; no-op se il modello usa RevIN.
    # EN: Denormalizes μ/σ z-score → raw; no-op when the model uses RevIN.
    def denormalize_predictions(self, mu, sigma):
        """
        Converte predizioni μ/σ da spazio standardizzato (z-score, output diretto
        del modello) a spazio raw (frazione di log-return, atteso dal trading layer).

        Funziona sia su scalari Python (float) che su np.ndarray (batch) che su
        torch.Tensor — preserva il tipo di input. Idempotente con target_scale=1.0
        (fallback safe se target_ret non in scale_cols).

        Quando il modello usa RevIN (`training_config.model.use_revin=True`), μ è
        già denormalizzato internamente dal modulo RevIN (vedi
        `quantsys/model/__init__.py` revin.denormalize_mu). In quel caso applicare
        di nuovo `target_scale` produce doppia denormalizzazione → segnali
        ipertofici. Ritorniamo (mu, sigma) invariati.

        Returns:
            (mu_raw, sigma_raw) nello stesso tipo dell'input.
        """
        use_revin = bool(
            self.training_config.get("model", {}).get("use_revin", False)
            if self.training_config else False
        )
        if use_revin:
            return mu, sigma
        scale = self.target_scale
        return mu * scale, sigma * scale

    # ── Serializzazione ───────────────────────────────────────────────────────

    # IT: Serializza lo stato su disco in modo atomico (mai corrotto).
    # EN: Atomically serializes the state to disk (never corrupted).
    def save(self, path: str) -> None:
        """Salva lo stato in modo atomico (tmp + rename → mai corrotto)."""
        from quantsys.utils.atomic_save import atomic_save_pkl
        atomic_save_pkl(self, path)
        size_kb = Path(path).stat().st_size // 1024
        logging.getLogger("quantsys.utils").info(
            f"PipelineState salvato → {path}  ({size_kb} KB)"
        )

    # IT: Ricarica lo stato pipeline da file pickle (per inference/backtest).
    # EN: Reloads the pipeline state from a pickle file (for inference/backtest).
    @classmethod
    def load(cls, path: str) -> "PipelineState":
        """Carica lo stato da file pickle."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        logging.getLogger("quantsys.utils").info(
            f"PipelineState caricato da {path}  "
            f"({len(state.feature_cols)} price features, "
            f"{len(state.macro_feature_cols)} macro features)"
        )
        return state

    # ── Trasformazione ────────────────────────────────────────────────────────

    # IT: Applica gli scaler delle price features (gestisce vecchio/nuovo formato).
    # EN: Applies price-feature scalers (handles legacy + current format).
    def transform_price(self, df: Any) -> Any:
        """
        Applica gli scaler delle price features a un DataFrame.
        Se una colonna è assente nel df (es. feature aggiunta dopo il training),
        logga un warning invece di crashare silenziosamente.

        Supporta due formati:
          · Nuovo (self.scaler multi-colonna): un singolo RobustScaler su tutta la matrice.
          · Vecchio (self.price_scaler_state dict): un RobustScaler per colonna.
        """
        import logging as _log
        import numpy as _np
        log = _log.getLogger("quantsys.utils.pipeline")
        result = df.copy()

        # IT: Path multi-colonna: ricostruisce sub-scaler con center/scale slice.
        # EN: Multi-column path: rebuilds sub-scaler from sliced center/scale.
        if getattr(self, "scaler", None) is not None and getattr(self, "scale_cols", []):
            cols    = self.scale_cols
            present = [c for c in cols if c in result.columns]
            missing = [c for c in cols if c not in result.columns]

            if present:
                # IT: lookup colonna O(1) via dict invece di cols.index in loop (A9).
                # EN: O(1) column lookup via dict instead of cols.index in a loop (A9).
                idx_map = {c: i for i, c in enumerate(cols)}
                col_idx = [idx_map[c] for c in present]
                X = result[present].values.astype(_np.float64)
                nan_mask = _np.isnan(X)
                X_imp = _np.where(nan_mask, 0.0, X)

                # IT: RobustScaler.transform = (X − center_)/scale_ → applico diretto, niente oggetto
                #     ricostruito ad ogni chiamata (A9). Bit-identico (with_centering/scaling=True).
                # EN: RobustScaler.transform = (X − center_)/scale_ → apply directly, no object rebuilt
                #     per call (A9). Bit-identical (with_centering/scaling=True).
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

        # IT: Path legacy per-colonna (backward compat pkl pre-refactor).
        # EN: Legacy per-column path (backward compat with pre-refactor pkls).
        missing = []
        for col, scaler in self.price_scaler_state.items():
            scaled_col = col + "_scaled"
            if col not in result.columns:
                missing.append(col)
                result[scaled_col] = 0.0   # IT: fallback zero (scaler centrato) | EN: zero fallback (centered)
                continue
            mask = result[col].notna()
            if mask.any():
                result.loc[mask, scaled_col] = scaler.transform(
                    result.loc[mask, col].values.reshape(-1, 1)
                ).flatten()
        if missing:
            log.warning(f"transform_price: {len(missing)} colonne mancanti → zeros: {missing[:5]}")
        return result

    # IT: Applica il MacroNormalizer; colonne mancanti riempite con zero.
    # EN: Applies the MacroNormalizer; missing columns filled with zeros.
    def transform_macro(self, df: Any) -> Any:
        """
        Applica il MacroNormalizer a un DataFrame.
        Gestisce colonne mancanti con zeros (robustezza a nuove serie FRED).
        """
        if self.macro_normalizer is None:
            raise RuntimeError("MacroNormalizer non presente nello stato.")
        # IT: copia UNA volta sola se mancano colonne (era df.copy() per ogni mancante) (A-minor).
        # EN: copy ONCE if any column is missing (was df.copy() per missing one) (A-minor).
        missing = [c for c in self.macro_feature_cols if c not in df.columns]
        if missing:
            df = df.copy()
            for col in missing:
                df[col] = 0.0
        return self.macro_normalizer.transform(df)

    # IT: Repr sintetico con conteggio feature, stream e metadati dataset.
    # EN: Compact repr with feature counts, stream type and dataset metadata.
    def __repr__(self) -> str:
        n_dyn    = getattr(self, "n_dynamic_features", None)
        n_struct = (len(self.feature_cols) - n_dyn) if n_dyn is not None else "?"
        stream   = f"dual({n_dyn}dyn+{n_struct}struct)" if n_dyn else "single"
        # Fix 3 — mostra metadati dataset se disponibili
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



# ─────────────────── identità dello scaler modello↔dataset ───────────────────
# IT: IL SAFETY NET CHE MANCAVA (2026-08-01). Il manifesto valida train↔inference
#     su `forecast_horizon` e su `interval`, ma NON sullo scaler — ed è lo scaler
#     la cosa che cambia più spesso, perché si rifitta a ogni rebuild del dataset.
#     Conseguenza concreta e già accaduta: `dev_vols_qlike.py` carica center/scale
#     dal `pipeline_state` del MODELLO e valuta sull'npz corrente, costruito con un
#     RobustScaler diverso. Gli input arrivano normalizzati con uno scaler, la rete
#     è stata addestrata con un altro, e μ viene denormalizzato con un terzo
#     riferimento. Nulla fallisce: esce un numero plausibile e leggermente peggiore,
#     che è il modo peggiore in cui un errore può presentarsi.
#     È la stessa classe del −4.94% di A8, artefatto di distribution shift
#     old-scaler che a parità di scaler valeva −0.79%.
# EN: THE SAFETY NET THAT WAS MISSING (2026-08-01). The manifesto validates
#     train↔inference on `forecast_horizon` and `interval` but NOT on the scaler —
#     and the scaler is what changes most often, since it is refit at every dataset
#     rebuild. Concrete, already-happened consequence: `dev_vols_qlike.py` loads
#     center/scale from the MODEL's `pipeline_state` and evaluates on the current
#     npz, built with a different RobustScaler. Inputs come normalized under one
#     scaler, the net was trained under another, μ is denormalized against a third
#     reference. Nothing fails: out comes a plausible, slightly worse number, which
#     is the worst way an error can present itself.
def scaler_fingerprint(state: "PipelineState") -> dict:
    # IT: impronta compatta e confrontabile dello scaler. Gli hash stanno su
    #     center_/scale_ INTERI: confrontare il solo `target_scale` non basta,
    #     perché due dataset possono condividere la scala del target e differire
    #     sulle feature di input — che è metà del disallineamento.
    # EN: compact, comparable scaler fingerprint. Hashes cover the WHOLE
    #     center_/scale_: comparing `target_scale` alone is not enough, since two
    #     datasets can share the target scale and differ on the input features —
    #     which is half of the mismatch.
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
    # IT: confronta lo scaler del modello con quello CANONICO, cioè quello scritto
    #     quando il dataset è stato costruito (`canonical_state_path()`,
    #     arch- e sandbox-independent). Ritorna un dizionario di provenienza — non solleva:
    #     la decisione se fermarsi spetta al chiamante, perché un run cross-vintage
    #     deliberato è legittimo purché sia DICHIARATO.
    #     `matches=None` significa "non verificabile" (canonico assente, p.es. clone
    #     pulito) e non deve mai essere confuso con "verificato uguale".
    # EN: compares the model's scaler against the CANONICAL one, i.e. the one
    #     written when the dataset was built (`canonical_state_path()`, arch- and
    #     sandbox-independent). Returns a provenance dict — it does not raise: whether
    #     to stop is the caller's decision, since a deliberate cross-vintage run is
    #     legitimate as long as it is DECLARED.
    #     `matches=None` means "not verifiable" (canonical absent, e.g. a clean
    #     clone) and must never be conflated with "verified equal".
    canonical_path = Path(canonical_path) if canonical_path else canonical_state_path()
    out = {"model": scaler_fingerprint(model_state), "canonical": None,
           "canonical_path": str(canonical_path), "matches": None}
    if not canonical_path.exists():
        return out
    out["canonical"] = scaler_fingerprint(PipelineState.load(str(canonical_path)))
    out["matches"] = bool(out["model"] == out["canonical"])
    return out


# ──────────── identità del VINTAGE MACRO modello↔dataset (M1, 2026-08-05) ────────────
# IT: IL SECONDO ASSE DI VINTAGE, scoperto il 2026-08-04. Il guard sopra copre il
#     RobustScaler dei PREZZI e `target_scale`; la normalizzazione MACRO non è
#     coperta e non può esserlo con lo stesso meccanismo, perché non vive nel
#     `PipelineState` canonico (`01_download_data` lo scrive PRIMA che `01b` esista;
#     il `MacroNormalizer` finisce in `models/lstm/pipeline_state.pkl` per il
#     routing di `01b`). Il confronto è quindi modello ↔ NPZ, non modello ↔ canonico.
#     Meccanismo: `01b_download_macro.py` ricarica gli split e sostituisce SOLO
#     `X_macro_{split}`, lasciando intatti `X_*`, `y_*`, `t_*`, e rifitta il
#     `MacroNormalizer` WHOLE-DF — quindi allungare la serie sposta mediana e IQR e
#     cambia i valori macro anche delle righe storiche. La macro è INPUT del modello
#     (90 colonne, embedding attivo) ma non entra in nessuna baseline HAR, che
#     leggono solo RV/target: da qui la firma osservata, il NN che si sposta e le
#     baseline identiche cifra per cifra. Scarto misurato sul rapporto pubblicato:
#     0.0019 — invisibile a `matches: true`, che guarda solo i prezzi.
# EN: THE SECOND VINTAGE AXIS, found on 2026-08-04. The guard above covers the PRICE
#     RobustScaler and `target_scale`; MACRO normalization is not covered and cannot
#     be, by the same mechanism, since it does not live in the canonical
#     `PipelineState` (`01_download_data` writes it BEFORE `01b` runs; the
#     `MacroNormalizer` ends up in `models/lstm/pipeline_state.pkl` for `01b`
#     routing). The comparison is therefore model ↔ NPZ, not model ↔ canonical.
#     Mechanism: `01b_download_macro.py` reloads the splits and replaces ONLY
#     `X_macro_{split}`, leaving `X_*`, `y_*`, `t_*` untouched, and refits the
#     `MacroNormalizer` WHOLE-DF — so extending the series shifts median and IQR and
#     changes macro values for historical rows too. Macro is a model INPUT (90
#     columns, embedding active) but enters no HAR baseline, which read RV/target
#     only: hence the observed signature, the NN moving while baselines stay
#     identical digit for digit. Measured gap on the published ratio: 0.0019 —
#     invisible to `matches: true`, which only looks at prices.
def macro_fingerprint(npz: Any) -> Optional[dict]:
    # IT: impronta del blocco macro DENTRO l'npz. Si impronta l'ARRAY che il modello
    #     ha consumato, non i parametri del `MacroNormalizer`: quelli nel canonico non
    #     ci sono affatto, e confrontarli col normalizer del modello stesso sarebbe
    #     circolare. ⚠ La pre-registrazione M1 fissava il solo `X_macro_train`; qui
    #     sono impronti TUTTI gli split presenti — strettamente più forte, costo
    #     identico (X_macro è 2D `(n, 90)` float32, non 3D come `X`), e chiude il caso
    #     di bordo in cui un rifit lasci train invariato a precisione float32 e muova
    #     val/test. Il dtype entra nell'impronta: due array uguali a valori ma di
    #     dtype diverso NON sono lo stesso input per la rete.
    # EN: fingerprint of the macro block INSIDE the npz. It fingerprints the ARRAY the
    #     model consumed, not the `MacroNormalizer` parameters: those are absent from
    #     the canonical state, and comparing them against the model's own normalizer
    #     would be circular. ⚠ M1's pre-registration fixed `X_macro_train` alone; here
    #     ALL present splits are fingerprinted — strictly stronger, identical cost
    #     (X_macro is 2D `(n, 90)` float32, not 3D like `X`), and it closes the corner
    #     case where a refit leaves train unchanged at float32 precision but moves
    #     val/test. dtype is part of the fingerprint: two arrays equal in value but of
    #     different dtype are NOT the same input to the network.
    import hashlib
    import numpy as _np
    if isinstance(npz, (str, Path)):
        npz = _np.load(str(npz), allow_pickle=True)
    keys = list(getattr(npz, "files", None) or list(npz.keys()))
    splits = [k for k in ("X_macro_train", "X_macro_val", "X_macro_test") if k in keys]
    # IT: nessun blocco macro → None = "non c'è", che NON è "combacia".
    # EN: no macro block → None = "absent", which is NOT "matches".
    if not splits:
        return None
    fp: dict = {"n_macro_features": None, "names_md5": None, "splits": {}}
    if "n_macro_features" in keys:
        try:
            fp["n_macro_features"] = int(_np.asarray(npz["n_macro_features"]).ravel()[0])
        except Exception:
            pass
    # IT: l'ORDINE delle colonne macro fa parte del vintage: stesse colonne in ordine
    #     diverso sono un input diverso per un embedding posizionale.
    # EN: macro column ORDER is part of the vintage: the same columns in a different
    #     order are a different input to a positional embedding.
    if "macro_feature_names" in keys:
        names = [str(x) for x in _np.asarray(npz["macro_feature_names"]).ravel().tolist()]
        fp["names_md5"] = hashlib.md5("\x00".join(names).encode("utf-8")).hexdigest()
    for k in splits:
        a = _np.ascontiguousarray(npz[k])
        fp["splits"][k] = {"md5": hashlib.md5(a.tobytes()).hexdigest(),
                           "shape": list(a.shape), "dtype": str(a.dtype)}
    return fp


def check_model_dataset_macro(model_state: "PipelineState", npz_arrays: Any) -> dict:
    # IT: confronta l'impronta macro REGISTRATA NEL MODELLO al training con quella
    #     dell'npz corrente. Come il guard sui prezzi non solleva: ritorna provenienza.
    #     ⚠ `matches=None` = "non verificabile" e NON deve mai diventare True. I
    #     modelli addestrati prima di M1 non hanno l'impronta: sono `None` per sempre,
    #     ed è un fatto sulla loro provenienza, non un difetto da mascherare.
    # EN: compares the macro fingerprint RECORDED IN THE MODEL at training time with
    #     the current npz's. Like the price guard it does not raise: it returns
    #     provenance. ⚠ `matches=None` = "not verifiable" and must never become True.
    #     Models trained before M1 carry no fingerprint: they stay `None` forever, and
    #     that is a fact about their provenance, not a defect to paper over.
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
    # IT: check + log + raise in un solo punto, e ritorna il blocco di provenienza
    #     da scrivere nel report. Esiste come funzione unica perche' i call-site
    #     sono quattro giudici con lo stesso identico wiring: duplicare il blocco
    #     significherebbe che il quinto lo dimentica, ed e' proprio "il controllo
    #     che manca in un posto solo" ad aver prodotto il numero sbagliato.
    #     ⚠ NON va chiamata sul path LIVE. `VolForecaster` e `FeatureAssembler`
    #     calcolano le feature al volo iniettando scaler e colonne dal
    #     PipelineState DEL MODELLO: sono self-consistent by construction e non
    #     leggono l'npz, quindi non hanno questa esposizione. Aggiungere qui un
    #     fail-fast fermerebbe il forward test al bootstrap successivo per un
    #     disallineamento che sul live non esiste.
    # EN: check + log + raise in one place, returning the provenance block to be
    #     written into the report. It exists as a single function because the call
    #     sites are four judges with identical wiring: duplicating the block would
    #     mean the fifth one forgets it, and "the check missing in exactly one
    #     place" is what produced the wrong number.
    #     ⚠ Do NOT call it on the LIVE path. `VolForecaster` and `FeatureAssembler`
    #     compute features on the fly, injecting scaler and columns from the
    #     MODEL's PipelineState: they are self-consistent by construction and never
    #     read the npz, so they have no such exposure. A fail-fast here would stop
    #     the forward test at the next bootstrap over a mismatch that does not
    #     exist live.
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
    # IT: secondo stadio del guard (M1). Gira SEMPRE, anche quando lo scaler è
    #     "non verificabile" o è stato dichiarato incomparabile: i due assi di
    #     vintage sono indipendenti e uno spento non deve spegnere l'altro.
    #     Arricchisce `prov` con il blocco `macro` e fail-fasta sul mismatch.
    #     ⚠ Se `npz_arrays` è None il controllo NON è stato richiesto dal chiamante:
    #     resta `None` e viene detto nel report — mai silenzio, mai True.
    # EN: the guard's second stage (M1). It ALWAYS runs, even when the scaler is
    #     "not verifiable" or has been declared incomparable: the two vintage axes
    #     are independent and switching one off must not switch off the other.
    #     It enriches `prov` with the `macro` block and fails fast on a mismatch.
    #     ⚠ If `npz_arrays` is None the check was not requested by the caller: it
    #     stays `None` and the report says so — never silence, never True.
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
