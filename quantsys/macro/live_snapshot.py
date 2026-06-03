"""
quantsys/macro/live_snapshot.py
================================
MacroSnapshotUpdater — aggiornamento periodico del contesto macro in inference live.

Problema risolto:
  Durante il live engine, il MacroEncoder riceveva sempre x_macro = zeros,
  rendendo inutile tutto il branch macro addestrato. Il modello aveva imparato
  correlazioni tra regimi macro e movimenti BTC, ma in produzione non le usava.

Soluzione:
  Un thread separato (daemon) aggiorna lo snapshot macro ogni ora.
  Lo snapshot viene letto in modo thread-safe dal loop principale del WS.
  Le fonti sono le stesse usate in training: yfinance (VIX, DXY, Gold, Oil)
  + FRED per i dati giornalieri (tassi, spread credito, ecc.).

Architettura:
  LiveEngine.__init__() → avvia MacroSnapshotUpdater in background
  MacroSnapshotUpdater → ogni 60 min scarica yfinance + FRED recent
                       → trasforma con MacroNormalizer (stesso scaler del training)
                       → salva in self._snapshot (array numpy thread-safe)
  LiveEngine._predict() → legge self.macro_updater.snapshot → passa alla LSTM

Fallback:
  Se il download fallisce (rete assente, FRED non risponde), lo snapshot
  rimane quello precedente. Se non c'è mai stato uno snapshot, usa zeros
  (comportamento precedente) con un warning all'avvio.

Latenza:
  L'aggiornamento è asincrono: non blocca mai il loop del WebSocket.
  Il lock acquisisce il mutex solo per la lettura dell'array → microsecondi.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np

log = logging.getLogger("quantsys.macro.live")


# Feature di mercato scaricabili real-time da yfinance (giornaliere)
# Corrispondono alle colonne che il MacroNormalizer si aspetta
# IT: Mappa ticker yfinance → nome colonna macro atteso dal normalizer.
# EN: yfinance ticker → macro column name expected by the normalizer.
_YF_TICKERS = {
    "^VIX":    "vix",
    "DX-Y.NYB":"dxy",
    "GC=F":    "gold",
    "CL=F":    "oil_wti",
    "^GSPC":   "sp500",
    "^TNX":    "treasury_10y_yf",
    "BTC-USD": "btc_daily",
}

# FRED series che cambiano ogni giorno (giornaliere)
# IT: Serie FRED a frequenza giornaliera → nome colonna macro.
# EN: Daily-frequency FRED series → macro column name.
_FRED_DAILY = {
    "T10YIE":   "infl_exp_10y",     # Breakeven inflazione 10Y
    "T5YIE":    "infl_exp_5y",      # Breakeven inflazione 5Y
    "T10Y2Y":   "yield_curve_2_10", # Spread 2Y-10Y
    "T10Y3M":   "yield_curve_3m_10",# Spread 3M-10Y
    "DFII10":   "real_rate_10y",    # Real rate 10Y
    "DFEDTARU": "fed_funds_upper",  # Fed Funds target upper
    "BAA10Y":       "credit_spread_hy",
    "AAA10Y":       "credit_spread_ig",
}


# IT: Thread daemon che aggiorna lo snapshot macro per l'inference live, thread-safe.
# EN: Daemon thread keeping the macro snapshot fresh for live inference, thread-safe.
class MacroSnapshotUpdater:
    """
    Thread daemon che mantiene aggiornato lo snapshot macro per l'inference live.

    Uso:
        updater = MacroSnapshotUpdater(normalizer, macro_feature_cols)
        updater.start()                    # avvia thread background
        ...
        xm = updater.get_tensor(device)   # tensor normalizzato pronto per la LSTM
        ...
        updater.stop()                     # alla chiusura del live engine
    """

    def __init__(
        self,
        normalizer,            # MacroNormalizer già fittato in 01b
        macro_feature_cols:    list[str],
        update_interval_sec:   int = 3600,   # aggiorna ogni ora
        fred_api_key:          str = "",
    ):
        # IT: Inizializza stato + snapshot a zeros (fallback se il 1° fetch fallisce).
        # EN: Init state + zeros snapshot (fallback if the first fetch fails).
        self.normalizer          = normalizer
        self.macro_feature_cols  = macro_feature_cols
        self.update_interval     = update_interval_sec
        self.fred_api_key        = fred_api_key

        # Snapshot corrente — array numpy (n_macro_features,) normalizzato
        # Inizializzato a zeros (fallback se il primo aggiornamento non riesce)
        self._snapshot           = np.zeros(len(macro_feature_cols), dtype=np.float32)
        self._snapshot_ts        = None   # timestamp dell'ultimo aggiornamento
        self._lock               = threading.RLock()
        self._stop_event         = threading.Event()
        self._thread             = None
        self._n_updates          = 0
        self._n_errors           = 0

    # ── API pubblica ──────────────────────────────────────────────────────────

    # IT: Avvia il loop periodico + un primo fetch immediato, entrambi in background.
    # EN: Starts the periodic loop + an immediate first fetch, both in background.
    def start(self) -> None:
        """Avvia il thread di aggiornamento in background."""
        self._thread = threading.Thread(
            target=self._update_loop,
            name="macro-snapshot-updater",
            daemon=True,   # si chiude automaticamente con il processo principale
        )
        self._thread.start()
        log.info(
            f"MacroSnapshotUpdater avviato: {len(self.macro_feature_cols)} features, "
            f"update ogni {self.update_interval//60} min"
        )
        # Primo aggiornamento immediato in background (non blocca il main thread)
        threading.Thread(target=self._do_update, daemon=True).start()

    # IT: Segnala lo stop e attende la chiusura del thread (timeout 5s).
    # EN: Signals stop and waits for the thread to finish (5s timeout).
    def stop(self) -> None:
        """Ferma il thread di aggiornamento."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # IT: Copia lo snapshot sotto lock e lo restituisce come tensor (1, n_features).
    # EN: Copies the snapshot under lock and returns it as a (1, n_features) tensor.
    def get_tensor(self, device=None):
        """
        Ritorna lo snapshot macro come tensor PyTorch normalizzato.
        Thread-safe: usa un lock leggero per copiare l'array.

        Returns:
            torch.Tensor shape (1, n_macro_features) pronto per la LSTM.
        """
        import torch
        with self._lock:
            snap = self._snapshot.copy()

        t = torch.tensor(snap[np.newaxis], dtype=torch.float32)
        if device is not None:
            t = t.to(device)
        return t

    # IT: True se lo snapshot è più recente di 2× l'intervallo di aggiornamento.
    # EN: True if the snapshot is newer than 2× the update interval.
    @property
    def is_fresh(self) -> bool:
        """True se lo snapshot è stato aggiornato nelle ultime 2 ore."""
        if self._snapshot_ts is None:
            return False
        age = time.time() - self._snapshot_ts
        return age < self.update_interval * 2

    # IT: Stringa diagnostica (età snapshot, n. aggiornamenti, n. errori) per log/dashboard.
    # EN: Diagnostic string (snapshot age, update count, error count) for log/dashboard.
    @property
    def status(self) -> str:
        """Stringa di stato per il log/dashboard."""
        if self._snapshot_ts is None:
            return "snapshot: mai aggiornato (usando zeros)"
        age_min = int((time.time() - self._snapshot_ts) / 60)
        return (
            f"snapshot macro: {age_min} min fa | "
            f"aggiornamenti: {self._n_updates} | "
            f"errori: {self._n_errors}"
        )

    # ── Loop interno ──────────────────────────────────────────────────────────

    # IT: Loop del thread: attende l'intervallo (interrompibile) e poi rilancia il fetch.
    # EN: Thread loop: waits the interval (interruptible) then triggers a fetch.
    def _update_loop(self) -> None:
        """Thread loop: aggiorna periodicamente lo snapshot."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.update_interval)
            if not self._stop_event.is_set():
                self._do_update()

    # IT: Fetch → allinea le colonne all'ordine atteso → normalizza → scrive lo snapshot sotto lock.
    # EN: Fetch → align columns to expected order → normalize → write the snapshot under lock.
    def _do_update(self) -> None:
        """Scarica i dati macro correnti e aggiorna lo snapshot."""
        try:
            raw = self._fetch_current_macro()
            if raw is None:
                return

            # Crea un DataFrame con le colonne nell'ordine atteso dal normalizer
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

    # IT: Scarica i valori macro più recenti (yfinance + FRED opzionale) → dict {feature: valore}.
    # EN: Downloads the latest macro values (yfinance + optional FRED) → dict {feature: value}.
    def _fetch_current_macro(self) -> Optional[dict]:
        """
        Scarica i valori macro più recenti da yfinance e FRED.
        Ritorna un dizionario {feature_name: valore_scalare}.
        """
        raw = {}
        errors = []

        # ── yfinance (real-time, ~15 min lag per dati gratuiti) ──────────────
        try:
            import yfinance as yf
            for ticker, name in _YF_TICKERS.items():
                try:
                    data = yf.download(ticker, period="5d", progress=False,
                                       auto_adjust=True)
                    if not data.empty:
                        last = float(data["Close"].iloc[-1])
                        raw[name] = last
                        # Calcola YoY e MoM se abbiamo abbastanza storia
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

        # ── FRED daily (richiede API key — salta se assente) ─────────────────
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

        # Yield curve composita
        if "yield_curve_2_10" in raw:
            raw["yield_inverted"] = float(raw["yield_curve_2_10"] < 0)

        if errors:
            log.debug(f"Fetch parziale: {len(errors)} errori — {'; '.join(errors[:3])}")

        n_filled = sum(1 for v in raw.values() if v != 0.0)
        log.debug(f"Macro fetch: {n_filled}/{len(_YF_TICKERS)+len(_FRED_DAILY)} valori")

        return raw if n_filled > 0 else None
