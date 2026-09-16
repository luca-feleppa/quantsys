"""
quantsys/macro/live_snapshot.py
================================
MacroSnapshotUpdater — periodic refresh of the macro context during live inference.

Problem solved:
  In the live engine the MacroEncoder always received x_macro = zeros,
  making the whole trained macro branch useless. The model had learned
  correlations between macro regimes and BTC moves, but did not use them in production.

Solution:
  A separate (daemon) thread refreshes the macro snapshot every hour.
  The snapshot is read thread-safely by the main WS loop.
  The sources are the same used in training: yfinance (VIX, DXY, Gold, Oil)
  + FRED for daily data (rates, credit spreads, etc.).

Architecture:
  LiveEngine.__init__() → starts MacroSnapshotUpdater in background
  MacroSnapshotUpdater → every 60 min downloads recent yfinance + FRED
                       → transforms with MacroNormalizer (same scaler as training)
                       → stores in self._snapshot (thread-safe numpy array)
  LiveEngine._predict() → reads self.macro_updater.snapshot → passes it to the LSTM

Fallback:
  If the download fails (no network, FRED not responding), the snapshot
  stays the previous one. If there has never been a snapshot, zeros are used
  (previous behaviour) with a warning at startup.

Latency:
  The update is asynchronous: it never blocks the WebSocket loop.
  The lock holds the mutex only while reading the array → microseconds.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np

log = logging.getLogger("quantsys.macro.live")


# Market features downloadable in real time from yfinance (daily),
# matching the columns the MacroNormalizer expects.
# yfinance ticker → macro column name expected by the normalizer.
_YF_TICKERS = {
    "^VIX":    "vix",
    "DX-Y.NYB":"dxy",
    "GC=F":    "gold",
    "CL=F":    "oil_wti",
    "^GSPC":   "sp500",
    "^TNX":    "treasury_10y_yf",
    "BTC-USD": "btc_daily",
}

# Daily-frequency FRED series → macro column name.
_FRED_DAILY = {
    "T10YIE":   "infl_exp_10y",     # 10Y inflation breakeven
    "T5YIE":    "infl_exp_5y",      # 5Y inflation breakeven
    "T10Y2Y":   "yield_curve_2_10", # Spread 2Y-10Y
    "T10Y3M":   "yield_curve_3m_10",# Spread 3M-10Y
    "DFII10":   "real_rate_10y",    # Real rate 10Y
    "DFEDTARU": "fed_funds_upper",  # Fed Funds target upper
    "BAA10Y":       "credit_spread_hy",
    "AAA10Y":       "credit_spread_ig",
}


# Daemon thread keeping the macro snapshot fresh for live inference, thread-safe.
class MacroSnapshotUpdater:
    """
    Daemon thread that keeps the macro snapshot up to date for live inference.

    Usage:
        updater = MacroSnapshotUpdater(normalizer, macro_feature_cols)
        updater.start()                    # starts the background thread
        ...
        xm = updater.get_tensor(device)   # normalized tensor ready for the LSTM
        ...
        updater.stop()                     # when the live engine shuts down
    """

    def __init__(
        self,
        normalizer,            # MacroNormalizer already fitted in 01b
        macro_feature_cols:    list[str],
        update_interval_sec:   int = 3600,   # refresh every hour
        fred_api_key:          str = "",
    ):
        # Init state + zeros snapshot (fallback if the first fetch fails).
        self.normalizer          = normalizer
        self.macro_feature_cols  = macro_feature_cols
        self.update_interval     = update_interval_sec
        self.fred_api_key        = fred_api_key

        # Current snapshot — normalized numpy array (n_macro_features,)
        # Initialized to zeros (fallback if the first update fails)
        self._snapshot           = np.zeros(len(macro_feature_cols), dtype=np.float32)
        self._snapshot_ts        = None   # timestamp of the last update
        self._lock               = threading.RLock()
        self._stop_event         = threading.Event()
        self._thread             = None
        self._n_updates          = 0
        self._n_errors           = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    # Starts the periodic loop + an immediate first fetch, both in background.
    def start(self) -> None:
        """Start the background update thread."""
        self._thread = threading.Thread(
            target=self._update_loop,
            name="macro-snapshot-updater",
            daemon=True,   # exits automatically with the main process
        )
        self._thread.start()
        log.info(
            f"MacroSnapshotUpdater avviato: {len(self.macro_feature_cols)} features, "
            f"update ogni {self.update_interval//60} min"
        )
        # Immediate first update in background (does not block the main thread)
        threading.Thread(target=self._do_update, daemon=True).start()

    # Signals stop and waits for the thread to finish (5s timeout).
    def stop(self) -> None:
        """Stop the update thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # Copies the snapshot under lock and returns it as a (1, n_features) tensor.
    def get_tensor(self, device=None):
        """
        Return the macro snapshot as a normalized PyTorch tensor.
        Thread-safe: uses a lightweight lock to copy the array.

        Returns:
            torch.Tensor of shape (1, n_macro_features) ready for the LSTM.
        """
        import torch
        with self._lock:
            snap = self._snapshot.copy()

        t = torch.tensor(snap[np.newaxis], dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    # True if the snapshot is newer than 2× the update interval.
    @property
    def is_fresh(self) -> bool:
        """True if the snapshot was updated in the last 2 hours."""
        if self._snapshot_ts is None:
            return False
        age = time.time() - self._snapshot_ts
        return age < self.update_interval * 2

    # Diagnostic string (snapshot age, update count, error count) for log/dashboard.
    @property
    def status(self) -> str:
        """Status string for the log/dashboard."""
        if self._snapshot_ts is None:
            return "snapshot: mai aggiornato (usando zeros)"
        age_min = int((time.time() - self._snapshot_ts) / 60)
        return (
            f"snapshot macro: {age_min} min fa | "
            f"aggiornamenti: {self._n_updates} | "
            f"errori: {self._n_errors}"
        )

    # ── Internal loop ─────────────────────────────────────────────────────────

    # Thread loop: waits the interval (interruptible) then triggers a fetch.
    def _update_loop(self) -> None:
        """Thread loop: periodically refreshes the snapshot."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.update_interval)
            if not self._stop_event.is_set():
                self._do_update()

    # Fetch → align columns to expected order → normalize → write the snapshot under lock.
    def _do_update(self) -> None:
        """Download the current macro data and update the snapshot."""
        try:
            raw = self._fetch_current_macro()
            if raw is None:
                return

            # Build a DataFrame with the columns in the order expected by the normalizer
            import pandas as pd
            row = {col: 0.0 for col in self.macro_feature_cols}
            for col, val in raw.items():
                if col in row:
                    row[col] = val

            df_row = pd.DataFrame([row])
            normalized = self.normalizer.transform(df_row)   # shape (1, n_features)
            snap = normalized[0].astype(np.float32)

            with self._lock:
                self._snapshot    = snap
                self._snapshot_ts = time.time()
                self._n_updates  += 1

            n_nonzero = int(np.count_nonzero(snap))
            log.info(
                f"Snapshot macro aggiornato: {n_nonzero}/{len(snap)} features non-zero "
                f"(ts={datetime.now().strftime('%H:%M:%S')})"
            )

        except Exception as e:
            self._n_errors += 1
            log.warning(
                f"MacroSnapshotUpdater: aggiornamento fallito ({e.__class__.__name__}: {e}). "
                f"Uso snapshot precedente."
            )

    # Downloads the latest macro values (yfinance + optional FRED) → dict {feature: value}.
    def _fetch_current_macro(self) -> Optional[dict]:
        """
        Download the latest macro values from yfinance and FRED.
        Returns a dict {feature_name: scalar_value}.
        """
        raw = {}
        errors = []

        # ── yfinance (real-time, ~15 min lag on free data) ───────────────────
        try:
            import yfinance as yf
            for ticker, name in _YF_TICKERS.items():
                try:
                    data = yf.download(ticker, period="5d", progress=False,
                                       auto_adjust=True)
                    if not data.empty:
                        last = float(data["Close"].iloc[-1])
                        raw[name] = last
                        # Compute YoY and MoM if there is enough history
                        if len(data) >= 2:
                            raw[f"{name}_1d_ret"] = float(
                                data["Close"].pct_change().iloc[-1]
                            )
                except Exception as e:
                    errors.append(f"{ticker}: {e}")
        except ImportError:
            errors.append("yfinance non installato")

        # ── VIX derivative features ───────────────────────────────────────────
        if "vix" in raw:
            raw["vix_high"] = float(raw["vix"] > 25)
            raw["vix_chg"]  = raw.get("vix_1d_ret", 0.0) * raw["vix"]

        # ── FRED daily (requires API key — skipped if absent) ─────────────────
        if self.fred_api_key:
            try:
                import requests
                base = "https://api.stlouisfed.org/fred/series/observations"
                for fred_id, name in _FRED_DAILY.items():
                    params = {
                        "series_id":  fred_id,
                        "api_key":    self.fred_api_key,
                        "sort_order": "desc",
                        "limit":      5,
                        "file_type":  "json",
                    }
                    r = requests.get(base, params=params, timeout=8)
                    if r.status_code == 200:
                        for o in r.json().get("observations", []):
                            try:
                                raw[name] = float(o["value"]); break
                            except (ValueError, TypeError):
                                continue
                    elif r.status_code == 400:
                        log.debug(f"FRED {fred_id}: serie non disponibile")
            except Exception as e:
                errors.append(f"FRED: {e}")
        else:
            log.debug("FRED API key assente — solo dati yfinance nel snapshot macro")

        # Composite yield curve
        if "yield_curve_2_10" in raw:
            raw["yield_inverted"] = float(raw["yield_curve_2_10"] < 0)

        if errors:
            log.debug(f"Fetch parziale: {len(errors)} errori — {'; '.join(errors[:3])}")

        n_filled = sum(1 for v in raw.values() if v != 0.0)
        log.debug(f"Macro fetch: {n_filled}/{len(_YF_TICKERS)+len(_FRED_DAILY)} valori")

        return raw if n_filled > 0 else None
