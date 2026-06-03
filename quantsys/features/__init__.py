"""Fase 2 — Feature Engineering: OHLCV → features normalizzate per LSTM."""
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

log = logging.getLogger("quantsys.features")


# IT: Single source of truth per i nomi feature canonici del modello (post C-funding).
#     Letta dal dataset NPZ generato dal training. Usata dal live engine per:
#       - allineare l'ordine delle colonne prodotte da FeatureBuilder
#       - hard-fail se una feature attesa manca (no pad/truncate posizionale)
#     Il caching evita re-letture del file NPZ a ogni chiamata.
# EN: Single source of truth for the model's canonical feature names (post C-funding).
#     Read from the NPZ dataset produced by training. Used by the live engine to:
#       - align the order of columns produced by FeatureBuilder
#       - hard-fail if an expected feature is missing (no positional pad/truncate)
#     Caching avoids re-reading the NPZ file on every call.
@lru_cache(maxsize=4)
def get_canonical_feature_names(npz_path: str = "data/lstm_dataset.npz") -> tuple[str, ...]:
    """Ritorna i nomi feature canonici del modello (104 nomi, ordine fisso).

    Single source of truth = `data/lstm_dataset.npz['feature_names']`.
    NB: `PipelineState.feature_cols` contiene 121 entries (pre-filter, include LIVE_DROP_FEATURES
    e i target target_ret/target_dir) — NON è la lista canonica del modello.

    Args:
        npz_path: percorso al dataset NPZ generato da `scripts/01_download_data.py`.

    Returns:
        tuple di stringhe (immutable per cache-friendly), ordine fisso.

    Raises:
        FileNotFoundError: se il NPZ non esiste.
        KeyError: se il NPZ non contiene 'feature_names'.
    """
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"NPZ canonico non trovato: {npz_path}. Esegui prima 01_download_data.py")
    with np.load(p, allow_pickle=True) as data:
        if "feature_names" not in data.files:
            raise KeyError(f"'feature_names' assente in {npz_path}. NPZ corrotto o vecchia versione.")
        names = tuple(str(n) for n in data["feature_names"])
    return names


# IT: Feature scartate dal set "C-funding" (single source of truth training↔live).
#     Motivo: permutation importance 2026-05-28 → ROI ≤ 0 (rumore o dannose) E/O lookback > 30g
#     non calcolabile nel buffer live. Si mantengono invece le 30d + funding (ROI positivo).
#     Vedi MODEL_IMPROVEMENTS.md "Allineamento feature live↔training (2026-05-28)".
# EN: Features dropped from the "C-funding" set (single source of truth training↔live).
#     Reason: 2026-05-28 permutation importance → ROI ≤ 0 (noise or harmful) AND/OR lookback > 30d
#     not computable in the live buffer. The 30d + funding features (positive ROI) are kept instead.
#     See MODEL_IMPROVEMENTS.md "Allineamento feature live↔training (2026-05-28)".
LIVE_DROP_FEATURES = frozenset({
    "dist_ath_90d", "dist_atl_90d", "price_pos_90d",
    "dist_ath_365d", "dist_atl_365d", "price_pos_365d",
    "momentum_90d", "momentum_7d",
    "frac_diff_close", "frac_diff_volume",
    "vp_poc_dist_long", "vp_vah_dist_long", "vp_val_dist_long", "vp_concentration_long",
    "vp_poc_convergence",
})


# IT: Pipeline feature engineering OHLCV → matrice normalizzata per il modello.
# EN: Feature-engineering pipeline OHLCV → normalized matrix for the model.
class FeatureBuilder:
    """Pipeline completa: OHLCV grezzo → array numpy pronti per la LSTM."""

    # IT: Memorizza iperparametri feature (VP, lag, horizon, FFD, RevIN, scaler).
    # EN: Stores feature hyperparameters (VP, lag, horizon, FFD, RevIN, scaler).
    def __init__(self, vp_bins: int = 30, vp_lookback: int = 240,
                 windows: list[int] = None, lag_periods: int = 5,
                 forecast_horizon: int = 1, vp_stride: int = 1,
                 frac_diff_d: float = 0.0, use_revin: bool = False):
        self.vp_bins             = vp_bins
        self.vp_lookback         = vp_lookback
        self.windows             = windows or [5, 10, 20, 60]
        self.lag_periods         = lag_periods
        self.forecast_horizon    = forecast_horizon   # IT: minuti nel futuro | EN: minutes ahead to predict
        self.vp_stride           = vp_stride          # IT: VP subsample stride | EN: VP subsample stride (O(n)→O(n/stride))
        self.frac_diff_d         = frac_diff_d         # IT: ordine FFD (0=skip) | EN: FFD order (0=skip)
        # IT: RevIN fix — escludi return raw dal RobustScaler globale: RevIN opera in
        #     scala raw e denormalize_mu allinea le predizioni col target (somma di log_ret).
        # EN: RevIN fix — exclude raw returns from the global RobustScaler so RevIN runs in
        #     raw scale and denormalize_mu yields predictions aligned to the target.
        self.use_revin           = use_revin
        self.scalers:            dict[str, RobustScaler] = {}
        self.scaler:             Optional[RobustScaler]  = None   # IT/EN: multi-column RobustScaler
        self._scale_cols:        list[str]               = []     # IT/EN: columns scaled by the multi-scaler
        self.feature_cols:       list[str] = []
        self.n_dynamic_features: int       = 0
        # IT: Clip bounds fittati su training (P0.1/P99.9 per feature) — adattivi vs ±20 fisso.
        # EN: Clip bounds fitted on training (P0.1/P99.9 per feature) — adaptive vs fixed ±20.
        self.clip_lo_: Optional[np.ndarray] = None
        self.clip_hi_: Optional[np.ndarray] = None

    # ── Log-returns ──────────────────────────────────────────────────────────
    # IT: Calcola log-return OHLCV e target multi-step (somma h candele future).
    # EN: Computes OHLCV log-returns and multi-step target (sum of next h candles).
    def _returns(self, df, forecast_horizon: int = 1):
        """
        Calcola log-return e target.

        Miglioramento — Target multi-step:
          Il target originale era log_ret.shift(-1): il rendimento del prossimo
          singolo minuto. Con commissioni dello 0.1% per trade, per essere
          profittevole il segnale deve predire movimenti di almeno 0.2%.
          Su candele a 1 minuto, movimenti così grandi sono rari e rumorosi.

          Con forecast_horizon=15: il target è la somma dei log-return delle
          prossime 15 candele = rendimento cumulato su 15 minuti.
          Questo riduce la frequenza di trading (meno commissioni), rende
          il segnale più forte e dà peso alle macro features. Movimenti
          da 0.3-1.0% su 15 min sono comuni e ben sopra le commissioni.

          Il modello vede comunque la finestra a 1 minuto — l'orizzonte
          cambia solo il target, non le feature.
        """
        df["log_ret"]      = np.log(df["close"] / df["close"].shift(1))
        df["log_ret_high"] = np.log(df["high"]  / df["high"].shift(1))
        df["log_ret_low"]  = np.log(df["low"]   / df["low"].shift(1))
        df["log_ret_vol"]  = np.log(
            df["volume"].replace(0, np.nan) /
            df["volume"].shift(1).replace(0, np.nan)
        )

        # IT: Target = somma dei log-return delle prossime h candele (rolling+shift).
        # EN: Target = sum of next h log-returns (rolling+shift, no temp Series loop).
        h = max(1, forecast_horizon)
        df["target_ret"] = df["log_ret"].rolling(h).sum().shift(-h)

        df["target_dir"] = (df["target_ret"] > 0).astype(int)
        return df

    # ── VWAP ─────────────────────────────────────────────────────────────────
    # IT: VWAP intraday + rolling 20/60 e deviazioni del prezzo dal VWAP.
    # EN: Intraday VWAP + rolling 20/60 and price-vs-VWAP deviations.
    def _vwap(self, df):
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["pv"]            = df["typical_price"] * df["volume"]
        df["date_utc"]      = df["open_time"].dt.date
        df["cum_pv"]        = df.groupby("date_utc")["pv"].cumsum()
        df["cum_vol"]       = df.groupby("date_utc")["volume"].cumsum()
        df["vwap"]          = df["cum_pv"] / df["cum_vol"].replace(0, np.nan)
        df["vwap_dev"]      = (df["close"] - df["vwap"]) / df["vwap"]
        for w in [20, 60]:
            rpv = df["pv"].rolling(w).sum()
            rv  = df["volume"].rolling(w).sum()
            df[f"vwap_{w}"]     = rpv / rv.replace(0, np.nan)
            df[f"vwap_{w}_dev"] = (df["close"] - df[f"vwap_{w}"]) / df[f"vwap_{w}"]
        return df

    # ── Volume Profile ────────────────────────────────────────────────────────
    # IT: Volume Profile per un lookback: POC/VAH/VAL/concentrazione (con stride).
    # EN: Volume Profile for one lookback: POC/VAH/VAL/concentration (strided).
    def _vp_single(self, tp_arr, vl_arr, lo_arr, hi_arr, cl_arr,
                   lookback: int, suffix: str, df_len: int,
                   vp_stride: int = 1) -> dict:
        """
        Calcola VP per un singolo lookback. Restituisce arrays per le 4 features.

        Ottimizzazione con vp_stride > 1:
          Invece di calcolare il VP per ogni singola candela (O(n × lookback)),
          lo calcola ogni `vp_stride` candele e interpola linearmente i valori
          intermedi. Con vp_stride=5 il costo scende da O(n) a O(n/5):
            · Scale 60:   2.1M × 60 / 5   = 25M  operazioni  (da 126M)
            · Scale 1440: 2.1M × 1440 / 5 = 605M operazioni  (da 3B)
          Il VP a scala 1440 cambia pochissimo in 5 minuti → l'approssimazione
          è trascurabile rispetto al rumore di mercato.

        Struttura:
          1. Calcola il VP solo sugli indici campionati (i = lookback, lookback+stride, ...)
          2. Riempie i risultati in un array full-size agli indici campionati
          3. Interpola linearmente i gap (forward/backward fill ai bordi)
        """
        poc_dist_sampled = {}
        vah_dist_sampled = {}
        val_dist_sampled = {}
        vol_conc_sampled = {}

        # IT: Indici campionati ogni vp_stride a partire da `lookback`.
        # EN: Sampled indices every vp_stride starting from `lookback`.
        sampled_indices = list(range(lookback, df_len, max(1, vp_stride)))

        for i in sampled_indices:
            sl  = slice(i - lookback, i)
            tp  = tp_arr[sl]; vol = vl_arr[sl]
            lo_ = lo_arr[sl].min(); hi_ = hi_arr[sl].max()
            if hi_ <= lo_:
                continue

            step    = (hi_ - lo_) / self.vp_bins
            idx_arr = np.clip(((tp - lo_) / step).astype(int), 0, self.vp_bins - 1)
            bin_vol = np.zeros(self.vp_bins)
            np.add.at(bin_vol, idx_arr, vol)

            poc_idx   = int(bin_vol.argmax())
            poc_price = lo_ + (poc_idx + 0.5) * step
            total     = bin_vol.sum()

            sorted_idx  = np.argsort(bin_vol)[::-1]
            cum_sorted  = np.cumsum(bin_vol[sorted_idx])
            n_va        = int(np.searchsorted(cum_sorted, 0.70 * total)) + 1
            va          = sorted_idx[:n_va]
            va_lo = lo_ + int(va.min()) * step
            va_hi = lo_ + (int(va.max()) + 1) * step

            curr = cl_arr[i]; safe = max(curr, 1e-9)
            poc_dist_sampled[i] = (curr - poc_price) / safe
            vah_dist_sampled[i] = (curr - va_hi)     / safe
            val_dist_sampled[i] = (curr - va_lo)     / safe
            vol_conc_sampled[i] = bin_vol[poc_idx]   / (total + 1e-9)

        # IT: Ricostruzione full-size con forward-fill (no look-ahead).
        # EN: Full-size reconstruction via forward-fill (no look-ahead).
        def _fill_interp(sampled_dict: dict, n: int) -> np.ndarray:
            arr = np.full(n, np.nan)
            if not sampled_dict:
                return arr
            idxs = np.array(sorted(sampled_dict.keys()), dtype=np.int64)
            vals = np.array([sampled_dict[k] for k in idxs], dtype=np.float64)
            arr[idxs] = vals
            # IT: numpy ffill — evita pd.Series temporanee (×12: 4 feat × 3 scale).
            # EN: numpy ffill — avoids ×12 temporary pd.Series (4 feats × 3 scales).
            mask = np.isnan(arr)
            idx  = np.where(~mask, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            return arr[idx]

        poc_dist = _fill_interp(poc_dist_sampled, df_len)
        vah_dist = _fill_interp(vah_dist_sampled, df_len)
        val_dist = _fill_interp(val_dist_sampled, df_len)
        vol_conc = _fill_interp(vol_conc_sampled, df_len)

        return {
            f"vp_poc_dist{suffix}": poc_dist,
            f"vp_vah_dist{suffix}": vah_dist,
            f"vp_val_dist{suffix}": val_dist,
            f"vp_concentration{suffix}": vol_conc,
        }

    # IT: Volume Profile multi-scala (1h/4h/1d) + feature di convergenza POC.
    # EN: Multi-scale Volume Profile (1h/4h/1d) + POC convergence feature.
    def _volume_profile(self, df):
        """
        Multi-scale Volume Profile: breve + medio + lungo termine.

        FIX CONCETTUALE — VP lookback fisso non si adatta al regime:
        ─────────────────────────────────────────────────────────────
        Con lookback fisso a 240 (4 ore):
          · In alta volatilità: 4 ore non bastano per i nodi di liquidità
          · In bassa volatilità: 4 ore coprono già mercato "maturo"
          · Il POC varia radicalmente in base al periodo scelto

        Soluzione — tre scale temporali:
          · Breve  (60 min  = 1 ora):   liquidità intraday recente
          · Medio  (240 min = 4 ore):   struttura di sessione (default precedente)
          · Lungo  (1440 min = 1 giorno): livelli tecnici giornalieri

        La LSTM vede tutte e tre le scale → impara quale è più rilevante
        in ogni regime. In alta vol domina il breve termine; in bassa vol
        il lungo termine è più stabile come supporto/resistenza.

        Il costo computazionale triplica ma resta accettabile (~15-30s totali).
        """
        tp_arr = df["typical_price"].values
        vl_arr = df["volume"].values
        lo_arr = df["low"].values
        hi_arr = df["high"].values
        cl_arr = df["close"].values
        n      = len(df)

        # IT: Avviso performance per dataset grandi.
        # EN: Performance warning for large datasets.
        if n > 500_000:
            log.info(
                f"Volume Profile: dataset grande ({n:,} candele), "
                f"vp_stride={self.vp_stride} "
                f"(se il calcolo è lento, aumenta vp_stride in config/default.yaml)"
            )

        # IT: Tre scale VP: breve (1h), medio (4h default), lungo (1d).
        # EN: Three VP scales: short (1h), medium (4h default), long (1d).
        scales = [
            (60,   "_short"),        # IT/EN: 1h — intraday liquidity
            (self.vp_lookback, ""),  # IT/EN: 4h — default (legacy name)
            (1440, "_long"),         # IT/EN: 1d — daily technical levels
        ]

        all_vp = {}
        for lookback, suffix in scales:
            # IT/EN: skip se dati insufficienti per la scala.
            if lookback > n - 10:
                log.warning(f"VP scale {lookback}: troppo pochi dati ({n}), skip.")
                continue
            effective_windows = max(1, (n - lookback) // self.vp_stride)
            log.debug(f"  VP lookback={lookback}{suffix} | stride={self.vp_stride} | ~{effective_windows:,} finestre")
            arrays = self._vp_single(tp_arr, vl_arr, lo_arr, hi_arr, cl_arr,
                                     lookback, suffix, n, vp_stride=self.vp_stride)
            all_vp.update(arrays)

        if all_vp:
            df = pd.concat([df, pd.DataFrame(all_vp, index=df.index)], axis=1)

        # IT: Feature composita — convergenza POC short vs long (livello forte).
        # EN: Composite — POC convergence short vs long (strong level).
        if "vp_poc_dist_short" in df.columns and "vp_poc_dist_long" in df.columns:
            df["vp_poc_convergence"] = 1.0 - np.abs(
                df["vp_poc_dist_short"].fillna(0) - df["vp_poc_dist_long"].fillna(0)
            ).clip(0, 1)

        return df

    # ── Technical indicators ──────────────────────────────────────────────────
    # IT: Microstructure zero-lag: anatomia candela, velocità, spread, skew.
    # EN: Zero-lag microstructure: candle anatomy, velocity, spread, skew.
    def _technicals(self, df):
        """
        Microstructure features — RSI, MACD, Bollinger Width e ATR rimossi.

        Rimossi perché ritardati e ridondanti:
          · RSI      → già catturato da vol_std + lag_ret
          · MACD     → già catturato da momentum + vol_ratio
          · BB Width → identico a vol_std_20 / vol_std_60
          · ATR      → già in vol_std; usato separatamente dal RiskManager

        Sostituiti con microstructure features istantanee o quasi:

          body_ratio      Forza direzionale della candela (0=doji, 1=marubozu)
          upper_shadow    Rifiuto del prezzo alto (pressione venditori)
          lower_shadow    Rifiuto del prezzo basso (pressione compratori)
          close_vs_open   Direzione e forza della singola candela
          intraday_pos    Dove chiude il prezzo nel range H-L

          price_velocity  Velocità del prezzo (close diff su 3 step normalizzata)
          price_accel     Accelerazione del prezzo (derivata della velocità)

          vwap_slope      Tendenza del VWAP negli ultimi 5 min (intraday bias)
          spread_proxy    (high-low)/volume — proxy del bid-ask spread / liquidità
          high_of_day_dist Distanza dal massimo delle ultime 4 ore (sessione)
          vwap_ret_skew   Asimmetria dei rendimenti pesata per volume (pressione)
        """
        hl = (df["high"] - df["low"]).replace(0, np.nan)

        # ── Candle anatomy (zero-lag microstructure) | Anatomia candela (zero-lag)
        df["body_ratio"]    = (df["close"] - df["open"]).abs() / hl.fillna(1)
        df["upper_shadow"]  = (df["high"] - df[["open","close"]].max(axis=1)) / hl.fillna(1)
        df["lower_shadow"]  = (df[["open","close"]].min(axis=1) - df["low"])   / hl.fillna(1)
        df["close_vs_open"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        df["intraday_pos"]  = (df["close"] - df["low"])  / hl.fillna(1)

        # ── Velocity & acceleration | Velocità e accelerazione del prezzo ────
        velocity             = df["close"].diff(3) / 3 / df["close"].shift(3).replace(0, np.nan)
        df["price_velocity"] = velocity.fillna(0)
        df["price_accel"]    = velocity.diff(1).fillna(0)

        # ── VWAP slope (intraday directional bias) | bias direzionale intraday
        if "vwap" in df.columns:
            vwap_diff        = df["vwap"].diff(5)
            df["vwap_slope"] = (vwap_diff / df["vwap"].shift(5).replace(0, np.nan)).fillna(0)
        else:
            df["vwap_slope"] = 0.0

        # IT: Spread proxy — proxy liquidità istantanea (alto = illiquido).
        # EN: Spread proxy — instant liquidity proxy (high = illiquid).
        df["spread_proxy"] = (hl / df["volume"].replace(0, np.nan)).fillna(0)

        # IT: Session position in [-0.5,+0.5] dentro il range 4h (mid_4h centrato).
        # EN: Session position in [-0.5,+0.5] within the 4h range (mid_4h centered).
        high_4h            = df["high"].rolling(240, min_periods=10).max()
        low_4h             = df["low"].rolling(240,  min_periods=10).min()
        range_4h           = (high_4h - low_4h).replace(0, np.nan)
        mid_4h             = (high_4h + low_4h) / 2
        df["session_position"] = (df["close"] - mid_4h) / range_4h

        # IT: Vol-weighted return skew (20) — >0 = pressione rialzista.
        # EN: Vol-weighted return skew (20) — >0 = bullish pressure.
        if "log_ret" in df.columns:
            vol_s        = df["volume"]
            ret_s        = df["log_ret"]
            roll_vol     = vol_s.rolling(20, min_periods=10).sum().replace(0, np.nan)
            wret_mean    = (ret_s * vol_s).rolling(20, min_periods=10).sum() / roll_vol
            dev          = ret_s - wret_mean
            roll_var     = (dev**2 * vol_s).rolling(20, min_periods=10).sum() / roll_vol
            df["vwap_ret_skew"] = (
                ((dev**3) * vol_s).rolling(20, min_periods=10).sum() /
                (roll_vol * roll_var.replace(0, np.nan)**1.5 + 1e-12)
            ).fillna(0)
        else:
            df["vwap_ret_skew"] = 0.0

        return df

    # ── Volume features ───────────────────────────────────────────────────────
    # IT: Feature di volume: taker ratio, z-score, OBV ROC, money flow.
    # EN: Volume features: taker ratio, z-score, OBV ROC, money flow.
    def _volume_features(self, df):
        df["taker_buy_ratio"] = (df["taker_buy_vol"] / df["volume"].replace(0, np.nan)).clip(0, 1)
        for w in [20, 60]:
            mu  = df["volume"].rolling(w).mean()
            sig = df["volume"].rolling(w).std().replace(0, np.nan)
            df[f"vol_zscore_{w}"] = (df["volume"] - mu) / sig

        direction = np.sign(df["close"].diff())

        # IT: OBV Rate-of-Change (stazionario) — evita drift cumulativo decennale.
        # EN: OBV Rate-of-Change (stationary) — avoids decade-long cumulative drift.
        obv_raw             = (direction * df["volume"]).cumsum()
        df["obv_roc_20"]    = obv_raw.diff(20)
        df["obv_roc_60"]    = obv_raw.diff(60)
        # IT/EN: normalizzato per volume medio | normalized by mean volume
        vol_ma_20           = df["volume"].rolling(20, min_periods=1).mean().replace(0, np.nan)
        vol_ma_60           = df["volume"].rolling(60, min_periods=1).mean().replace(0, np.nan)
        df["obv_roc_20_n"]  = df["obv_roc_20"] / (vol_ma_20 * 20)
        df["obv_roc_60_n"]  = df["obv_roc_60"] / (vol_ma_60 * 60)

        candle_sz = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-9)
        mf = direction * df["volume"] * candle_sz
        df["money_flow_norm"] = mf.rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-9)
        return df

    # IT: Cumulative Volume Delta: pressione order-flow, divergenza, accelerazione.
    # EN: Cumulative Volume Delta: order-flow pressure, divergence, acceleration.
    def _cvd_features(self, df):
        """
        Miglioramento 4 — Cumulative Volume Delta (CVD).

        Il delta del volume è la differenza tra volume di acquisto aggressivo
        (taker buy) e vendita aggressiva (taker sell). Misura la pressione
        direzionale degli operatori che "attraversano lo spread".

        CVD = Σ(taker_buy - taker_sell) cumulato nel tempo.
        Un CVD crescente mentre il prezzo è piatto = pressione nascosta al rialzo.
        Un CVD decrescente mentre il prezzo è alto = distribuzione.

        Feature derivate:
          · cvd_raw:        delta valore assoluto (in unità di BTC)
          · cvd_norm:       delta normalizzato per volume [-1, +1]
          · cvd_divergence: differenza tra trend CVD e trend prezzo
          · delta_accel:    accelerazione del delta (secondo derivata)
        """
        taker_sell = df["volume"] - df["taker_buy_vol"]
        delta      = df["taker_buy_vol"] - taker_sell
        cvd        = delta.cumsum()

        df["cvd_raw"]   = delta                                    # IT/EN: instant delta
        df["cvd_norm"]  = delta / df["volume"].replace(0, np.nan)  # IT/EN: normalized [-1,1]

        # IT: Rolling CVD 20/60 con min_periods=w (coerenza warmup vs steady-state).
        # EN: Rolling CVD 20/60 with min_periods=w (warmup vs steady-state parity).
        for w in [20, 60]:
            df[f"cvd_cum_{w}"]  = delta.rolling(w, min_periods=w).sum()
            vol_sum = df["volume"].rolling(w, min_periods=w).sum().replace(0, np.nan)
            df[f"cvd_pct_{w}"]  = df[f"cvd_cum_{w}"] / vol_sum     # IT/EN: % of volume

        # IT: Divergenza CVD norm vs log_ret norm (rolling 20) — min_periods esplicito
        #     per evitare distribution shift sui primi sample del buffer live.
        # EN: CVD-norm vs log_ret-norm divergence (rolling 20) — explicit min_periods
        #     to avoid distribution shift on the first live-buffer samples.
        cvd_trend  = delta.rolling(20, min_periods=20).sum().fillna(0)
        price_trend= df["log_ret"].rolling(20, min_periods=20).sum().fillna(0)
        cvd_std    = cvd_trend.rolling(60, min_periods=60).std().replace(0, np.nan)
        price_std  = price_trend.rolling(60, min_periods=60).std().replace(0, np.nan)
        df["cvd_divergence"] = (cvd_trend / cvd_std) - (price_trend / price_std)

        # IT/EN: accelerazione delta = order-flow momentum | delta acceleration = order-flow momentum
        df["delta_accel"] = delta.diff(5) / df["volume"].rolling(5, min_periods=5).sum().replace(0, np.nan)

        return df

    # IT: Allinea il funding rate (8h) all'indice 1m e deriva media/deviazione.
    # EN: Aligns funding rate (8h) to the 1m index and derives mean/deviation.
    def _funding_features(self, df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge funding rate features al DataFrame principale.

        Il funding rate è a frequenza 8h; viene allineato all'indice 1m
        del df principale con forward-fill. I NaN iniziali (dati pre-2020
        o gap iniziali) vengono riempiti con 0.

        Feature aggiunte:
          · funding_rate:     valore istantaneo (ffill da 8h)
          · funding_rate_1d:  media mobile 24h (1440 min) — livello di base
          · funding_rate_dev: deviazione dalla media — segnale contrarian
        """
        funding_df = funding_df.copy()
        if not isinstance(funding_df.index, pd.DatetimeIndex):
            funding_df = funding_df.set_index("open_time")

        funding_series = funding_df["funding_rate"]

        # IT/EN: allinea all'indice del df (ffill su frequenza 8h) | align to df index (ffill 8h freq)
        df_index = df["open_time"]
        aligned = funding_series.reindex(df_index, method="ffill").values

        df["funding_rate"]     = aligned
        df["funding_rate"]     = df["funding_rate"].fillna(0)

        df["funding_rate_1d"]  = df["funding_rate"].rolling(1440, min_periods=1).mean()
        df["funding_rate_dev"] = df["funding_rate"] - df["funding_rate_1d"]

        return df

    # IT: Feature strutturali di livello prezzo: dist ATH/ATL, momentum, MA200m.
    # EN: Structural price-level features: ATH/ATL distance, momentum, MA200m.
    def _structural_features(self, df):
        """
        Miglioramento 3 — Features di livello prezzo assoluto.

        La LSTM senza queste feature non sa se BTC è a $30k (vicino ai minimi)
        o $70k (vicino ai massimi storici). Il contesto strutturale è cruciale
        per capire dove si trovano i livelli di supporto/resistenza.

        Feature:
          · dist_ath_{30,90,365}:  distanza % dall'ATH del periodo
          · dist_atl_{30,90,365}:  distanza % dall'ATL del periodo
          · price_position_{30,90}: posizione nel range [ATL, ATH] → [0, 1]
          · momentum_{30,90}:       performance % vs N giorni fa
          · round_level_dist:       distanza dal livello tondo più vicino (psicologico)
                                    ($60k, $65k, $70k, ecc.)

        Nota: queste feature cambiano lentamente (settimane) — sono ideali per
        il StructuralEncoder (stream B del dual-stream) perché non hanno la
        stessa dinamica delle feature di trading (stream A).
        """
        close = df["close"]

        for days in [30, 90, 365]:
            w = days * 24 * 60   # IT/EN: candele 1m | 1m candles
            ath = close.rolling(w, min_periods=60).max()
            atl = close.rolling(w, min_periods=60).min()

            df[f"dist_ath_{days}d"]   = (close - ath) / ath.replace(0, np.nan)   # IT/EN: ≤0
            df[f"dist_atl_{days}d"]   = (close - atl) / atl.replace(0, np.nan)   # IT/EN: ≥0
            price_range = (ath - atl).replace(0, np.nan)
            df[f"price_pos_{days}d"]  = (close - atl) / price_range              # IT/EN: [0,1]

        # IT/EN: momentum = log-return vs N giorni fa | log-return vs N days ago
        for days in [7, 30, 90]:
            w = days * 24 * 60
            df[f"momentum_{days}d"] = np.log(
                close / close.shift(w).replace(0, np.nan)
            )

        # IT: Livelli psicologici tondi (multipli di $1000 per BTC).
        # EN: Round psychological levels (multiples of $1000 for BTC).
        round_level = (close / 1000).round() * 1000
        df["round_level_dist"] = (close - round_level) / close.replace(0, np.nan)

        # IT: price vs MA 200 MINUTI (~3.3h, intraday) — NON 200 giorni.
        # EN: price vs 200-MINUTE MA (~3.3h, intraday) — NOT 200 days.
        df["price_vs_ma200m"] = close / close.rolling(200, min_periods=50).mean() - 1

        return df

    # ── Volatility / regime ───────────────────────────────────────────────────
    # IT: Volatilità realizzata, rapporti tra scale, skew e curtosi dei return.
    # EN: Realized volatility, cross-scale ratios, return skew and kurtosis.
    def _volatility(self, df):
        for w in self.windows:
            df[f"vol_std_{w}"]  = df["log_ret"].rolling(w).std()
            df[f"vol_mean_{w}"] = df["log_ret"].rolling(w).mean()
        for w1, w2 in [(5, 20), (5, 60), (20, 60)]:
            c1, c2 = f"vol_std_{w1}", f"vol_std_{w2}"
            if c1 in df.columns and c2 in df.columns:
                df[f"vol_ratio_{w1}_{w2}"] = df[c1] / df[c2].replace(0, np.nan)
        df["realized_var_5"]  = (df["log_ret"] ** 2).rolling(5).mean()
        df["realized_var_20"] = (df["log_ret"] ** 2).rolling(20).mean()
        df["ret_skew_20"]     = df["log_ret"].rolling(20).skew()
        df["ret_kurt_20"]     = df["log_ret"].rolling(20).kurt()
        return df

    # ── Time features ─────────────────────────────────────────────────────────
    # IT: Encoding ciclico (ora/giorno/mese) + flag sessioni di trading.
    # EN: Cyclic encoding (hour/day/month) + trading-session flags.
    def _time_features(self, df):
        hour  = df["open_time"].dt.hour + df["open_time"].dt.minute / 60
        dow   = df["open_time"].dt.dayofweek
        month = df["open_time"].dt.month
        df["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
        df["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["session_asia"]    = ((hour >= 0)  & (hour < 8)).astype(float)
        df["session_london"]  = ((hour >= 8)  & (hour < 16)).astype(float)
        df["session_ny"]      = ((hour >= 13) & (hour < 21)).astype(float)
        df["session_overlap"] = ((hour >= 13) & (hour < 16)).astype(float)
        return df

    # ── Lag features ──────────────────────────────────────────────────────────
    # IT: Lag 1..N di return, volume z-score e taker ratio (memoria breve).
    # EN: Lags 1..N of returns, volume z-score and taker ratio (short memory).
    def _lags(self, df):
        for lag in range(1, self.lag_periods + 1):
            df[f"lag_ret_{lag}"]   = df["log_ret"].shift(lag)
            df[f"lag_vol_{lag}"]   = df["vol_zscore_20"].shift(lag)
            df[f"lag_taker_{lag}"] = df["taker_buy_ratio"].shift(lag)
        return df

    # ── Fractional Differencing (López de Prado, AFML) ──────────────────────
    # IT: Pesi binomiali troncati per FFD (stazionarietà preservando memoria).
    # EN: Truncated binomial weights for FFD (stationarity keeping memory).
    @staticmethod
    def _frac_diff_weights(d: float, thresh: float = 1e-5) -> np.ndarray:
        """
        Pesi della serie binomiale per differenziazione frazionaria.

        w_0 = 1, w_k = -w_{k-1} * (d - k + 1) / k
        I pesi vengono troncati quando |w_k| < thresh (Fixed-width FFD).
        Restituisce i pesi invertiti per convoluzione diretta con np.convolve.
        """
        w = [1.0]
        k = 1
        while True:
            w_k = -w[-1] * (d - k + 1) / k
            if abs(w_k) < thresh:
                break
            w.append(w_k)
            k += 1
            if k > 5000:   # safety cap
                break
        return np.array(w[::-1])   # reversed per convoluzione

    # IT: FFD vettorizzata di log(close) e log(volume+1); skip se d=0.
    # EN: Vectorized FFD of log(close) and log(volume+1); skipped if d=0.
    def _frac_diff(self, df):
        """
        Differenziazione frazionaria di log(close) e log(volume+1).

        Con d standard = 1.0, i log-return rimuovono TUTTA la memoria
        della serie. Con 0 < d < 1 (tipicamente 0.3-0.7) si ottiene
        stazionarietà preservando l'autocorrelazione di lungo periodo.

        Usa il metodo FFD (Fixed-width Fractional Differencing):
          frac_diff(x, d) = sum_{k=0}^{K} w_k * x_{t-k}
        dove i pesi sono troncati quando |w_k| < 1e-5.
        La convoluzione è vettorizzata con np.convolve (nessun loop Python su righe).

        Feature aggiunte:
          · frac_diff_close:  FFD di log(close)
          · frac_diff_volume: FFD di log(volume + 1)

        Skip se frac_diff_d == 0.0 (backward compatible).
        """
        d = self.frac_diff_d
        if d == 0.0:
            return df

        weights = self._frac_diff_weights(d)
        width = len(weights)
        log.info(f"  Fractional diff: d={d}, window={width} pesi")

        # ── log(close) ──────────────────────────────────────────────────
        log_close = np.log(df["close"].values.astype(np.float64))
        conv = np.convolve(log_close, weights, mode="full")[:len(log_close)]
        # Le prime (width - 1) osservazioni non hanno abbastanza storia → NaN
        result_close = np.empty(len(log_close), dtype=np.float64)
        result_close[:width - 1] = np.nan
        result_close[width - 1:] = conv[width - 1:]
        df["frac_diff_close"] = result_close

        # ── log(volume + 1) ─────────────────────────────────────────────
        log_vol = np.log1p(df["volume"].values.astype(np.float64))
        conv_v = np.convolve(log_vol, weights, mode="full")[:len(log_vol)]
        result_vol = np.empty(len(log_vol), dtype=np.float64)
        result_vol[:width - 1] = np.nan
        result_vol[width - 1:] = conv_v[width - 1:]
        df["frac_diff_volume"] = result_vol

        return df

    # ── Normalizzazione ───────────────────────────────────────────────────────
    # IT: Colonne già in scala naturale (cicliche, [0,1], binarie) — no scaler.
    # EN: Columns already in natural scale (cyclic, [0,1], binary) — no scaler.
    _NO_SCALE = {
        "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
        "session_asia","session_london","session_ny","session_overlap",
        "taker_buy_ratio","intraday_pos","target_dir",
        # Nuove microstructure già in [0,1]
        "body_ratio","upper_shadow","lower_shadow",
    }

    # IT: Set no-scale; con RevIN aggiunge i return raw (gestiti per-istanza).
    # EN: No-scale set; with RevIN adds raw returns (handled per-instance).
    def _no_scale_set(self) -> set:
        """
        Set di colonne da NON scalare con il RobustScaler globale.

        Comportamento di default: ritorna _NO_SCALE (backward compatible).

        Quando self.use_revin=True, aggiunge dinamicamente le colonne di
        return raw (log_ret, log_ret_high, log_ret_low, log_ret_vol e tutte
        le lag_ret_{N}). RevIN normalizza queste feature per-istanza, e la
        denormalizzazione delle predizioni deve riportare i mu nello stesso
        spazio del target raw (target_ret = somma di log_ret). Se il global
        scaler le standardizzasse, RevIN opererebbe su una feature già
        scalata e denormalizzerebbe in spazio scalato — disallineato con il
        target raw (~1e-4), e l'affine (gamma, beta) finirebbe per assorbire
        il mismatch invece di lasciare a RevIN il suo ruolo originale di
        rimozione del distribution shift locale.
        """
        no_scale = set(self._NO_SCALE)
        if getattr(self, "use_revin", False):
            no_scale.update({
                "log_ret", "log_ret_high", "log_ret_low", "log_ret_vol",
            })
            # lag_ret_{N} sono derivati da log_ret.shift(N): stessa scala raw
            for lag in range(1, self.lag_periods + 1):
                no_scale.add(f"lag_ret_{lag}")
        return no_scale

    # IT: Fit dello scaler solo su righe di training — evita data leakage.
    # EN: Scaler fit on training rows only — prevents data leakage.
    def fit_scaler_only(self, df: pd.DataFrame) -> "FeatureBuilder":
        """
        Fitta un singolo RobustScaler multi-colonna sulle righe di df senza trasformare nulla.
        Usato per evitare data leakage: si fitta solo sulle righe di training,
        poi si trasforma tutto il dataset con _normalize(fit=False).

        Args:
            df: DataFrame con le righe di training (indice 0..train_end)

        Returns:
            self (per chaining)
        """
        no_scale = self._no_scale_set()
        to_scale = [c for c in self.feature_cols if c not in no_scale and c in df.columns]
        self._scale_cols = to_scale
        self.scalers = {}   # IT/EN: vuoto — backward-compat con vecchi pkl | empty — back-compat with old pkl

        X = df[to_scale].values.astype(np.float64)
        # IT: Imputa NaN con mediana per-colonna solo per il fit dei quantili.
        # EN: Impute NaN with per-column median for quantile fit only.
        with np.errstate(all="ignore"):
            col_medians = np.nanmedian(X, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        X_imp = np.where(np.isnan(X), col_medians, X)
        self.scaler = RobustScaler()
        self.scaler.fit(X_imp)

        # IT: Clip [P0.1, P99.9] fittato solo su training (adattivo, no leakage).
        # EN: Clip [P0.1, P99.9] fitted on training only (adaptive, no leakage).
        X_scaled_tr = self.scaler.transform(X_imp)
        with np.errstate(all="ignore"):
            self.clip_lo_ = np.nanpercentile(X_scaled_tr, 0.1, axis=0)
            self.clip_hi_ = np.nanpercentile(X_scaled_tr, 99.9, axis=0)

        log.info(
            f"Scaler multi-colonna fittato su {len(df):,} righe di training "
            f"({len(to_scale)} colonne)"
        )
        return self

    # IT: Applica (e opz. fitta) il RobustScaler multi-colonna + clip, in-place.
    # EN: Applies (and optionally fits) the multi-column RobustScaler + clip, in-place.
    def _normalize(self, df, fit: bool = True):
        """
        Normalizza le feature in-place sovrascrivendo le colonne originali.

        FIX RAM — sovrascrittura in-place invece di colonne *_scaled:
          Il vecchio approccio creava col + "_scaled" per ogni colonna scalata,
          raddoppiando la RAM usata dal DataFrame. Con 2M righe e 55+ features,
          questo significava tenere in memoria sia i valori raw che quelli scalati.

          Ora i valori normalizzati sovrascrivono direttamente le colonne originali.
          Le colonne raw non servono dopo la normalizzazione: il backtest carica
          open/high/low/close/volume direttamente dal parquet (colonne OHLCV),
          non le feature engineered.

          Risparmio: ~55 colonne × 2M righe × 4 byte (float32) ≈ 440 MB.

        OTTIMIZZAZIONE — RobustScaler multi-colonna (singolo oggetto invece di 60+):
          Invece di 60 RobustScaler separati (uno per colonna), usa un singolo
          RobustScaler fittato su tutta la matrice. Elimina il loop Python per il
          transform e riduce da ~60 oggetti a 1 nel PipelineState.pkl.
          I NaN vengono preservati: vengono imputati con 0 per il transform
          (valore neutro dopo la centratura), poi ripristinati dalla mask.
        """
        no_scale = self._no_scale_set()
        to_scale = [c for c in self.feature_cols if c not in no_scale and c in df.columns]
        if fit:
            self._scale_cols = to_scale
            self.scalers = {}
            X = df[to_scale].values.astype(np.float64)
            with np.errstate(all="ignore"):
                col_medians = np.nanmedian(X, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            X_imp = np.where(np.isnan(X), col_medians, X)
            self.scaler = RobustScaler()
            self.scaler.fit(X_imp)
            # IT/EN: clip adattivo fit solo su training | adaptive clip fitted on training only
            X_scaled_tr = self.scaler.transform(X_imp)
            with np.errstate(all="ignore"):
                self.clip_lo_ = np.nanpercentile(X_scaled_tr, 0.1, axis=0)
                self.clip_hi_ = np.nanpercentile(X_scaled_tr, 99.9, axis=0)

        if self.scaler is not None and self._scale_cols:
            # IT/EN: interseca _scale_cols con le colonne presenti | intersect _scale_cols with present columns
            cols = [c for c in self._scale_cols if c in df.columns]
            X = df[cols].values.astype(np.float64)
            nan_mask = np.isnan(X)
            # IT: Imputa NaN con 0 = mediana RobustScaler dopo centratura (neutro).
            # EN: Impute NaN with 0 = RobustScaler median after centering (neutral).
            X_imp = np.where(nan_mask, 0.0, X)

            if cols == self._scale_cols:
                X_scaled = self.scaler.transform(X_imp)
            else:
                # IT/EN: subset scaler parziale via indici del fit | partial scaler via fit indices
                col_idx = [self._scale_cols.index(c) for c in cols]
                import sklearn.preprocessing as _skpp
                sub_scaler = _skpp.RobustScaler()
                sub_scaler.center_ = self.scaler.center_[col_idx]
                sub_scaler.scale_  = self.scaler.scale_[col_idx]
                sub_scaler.n_features_in_ = len(col_idx)
                X_scaled = sub_scaler.transform(X_imp)

            X_scaled[nan_mask] = np.nan
            # IT/EN: winsorization per-feature con bounds da training | per-feature winsorization with training bounds
            if self.clip_lo_ is not None:
                if len(self.clip_lo_) == X_scaled.shape[1]:
                    X_scaled = np.clip(X_scaled, self.clip_lo_, self.clip_hi_)
                elif cols != self._scale_cols:
                    col_idx = [self._scale_cols.index(c) for c in cols]
                    X_scaled = np.clip(X_scaled, self.clip_lo_[col_idx], self.clip_hi_[col_idx])
            df[cols] = X_scaled

        return df

    # IT: Estrae uno scaler per singola colonna (nuovo multi o vecchio per-col).
    # EN: Extracts a single-column scaler (new multi or legacy per-column).
    def _get_scaler_for_col(self, col: str) -> Optional[RobustScaler]:
        """
        Backward compatibility: restituisce un RobustScaler per singola colonna
        funzionando sia con il nuovo formato multi-colonna che con il vecchio per-colonna.

        Nuovo formato (self.scaler multi-colonna):
          Estrae centro e scala per la colonna richiesta dai parametri del multi-scaler
          e costruisce un RobustScaler "virtuale" per quella singola colonna.

        Vecchio formato (self.scalers dict):
          Restituisce direttamente self.scalers.get(col) come prima.

        Usato da PipelineState e dal live engine per transform su singola colonna.
        """
        if self.scaler is not None and col in self._scale_cols:
            idx = self._scale_cols.index(col)
            s = RobustScaler()
            s.center_         = np.array([self.scaler.center_[idx]])
            s.scale_          = np.array([self.scaler.scale_[idx]])
            s.n_features_in_  = 1
            return s
        # IT/EN: fallback per pkl pre-refactor | fallback for pre-refactor pkl
        return self.scalers.get(col)

    # ── PUBLIC ────────────────────────────────────────────────────────────────
    # IT: Orchestratore: esegue tutti gli step, split dual-stream e normalizza.
    # EN: Orchestrator: runs all steps, dual-stream split and normalization.
    def build(self, df: pd.DataFrame, normalize: bool = True, fit: bool = True,
              funding_df: "Optional[pd.DataFrame]" = None) -> pd.DataFrame:
        df = df.copy()
        steps = [
            ("log-returns",   lambda d: self._returns(d, self.forecast_horizon)),
            ("VWAP",          self._vwap),
            ("technical",     self._technicals),
            ("volume",        self._volume_features),
            ("CVD",           self._cvd_features),
            ("volatility",    self._volatility),
            ("time",          self._time_features),
            ("lags",          self._lags),
            ("frac_diff",     self._frac_diff),           # IT/EN: López de Prado FFD
            ("structural",    self._structural_features),
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented",
                                    category=pd.errors.PerformanceWarning)
            for i, (name, fn) in enumerate(steps):
                log.info(f"  → {name}")
                df = fn(df)
                if i == 4:
                    df = df.copy()
            log.info("  → volume profile (multi-scale, ~30-60s) ...")
            df = self._volume_profile(df)

        if funding_df is not None and len(funding_df) > 0:
            log.info("  → funding rate features")
            df = self._funding_features(df, funding_df)
        else:
            log.warning(
                "Funding rate non disponibile — feature funding_rate* escluse. "
                "Esegui scripts/01_download_data.py per scaricarle."
            )

        # IT/EN: interazioni di feature per regime detection | feature interactions for regime detection
        # IT: accesso colonna sicuro — su dataset corti alcune feature-base possono mancare (no KeyError).
        # EN: safe column access — on short datasets some base features may be missing (no KeyError).
        g = lambda c: df[c] if c in df.columns else 0.0
        df["vol_x_pos"]          = g("vol_ratio_5_20") * g("price_pos_30d")
        df["momentum_x_funding"] = g("momentum_30d")   * g("funding_rate_dev")
        df["cvd_x_vol"]          = g("cvd_norm")        * g("vol_std_20")

        exclude = {"open_time","close_time","date_utc","cum_pv","cum_vol","pv",
                   "typical_price","obv","obv_roc_20","obv_roc_60"}
        all_cols = [c for c in df.columns if c not in exclude]

        # IT: Dual-stream split via set esplicito (no prefix matching fragile).
        #     Stream B (structural): feature lente (giorni/ore) — contesto di mercato.
        #     Stream A (dynamic):   tutto il resto, varia ogni minuto.
        # EN: Dual-stream split via explicit set (no fragile prefix matching).
        #     Stream B (structural): slow features (days/hours) — market context.
        #     Stream A (dynamic):   everything else, changes every minute.
        _STRUCTURAL_COLS = {
            "vp_poc_dist", "vp_vah_dist", "vp_val_dist", "vp_concentration",
            "vp_poc_dist_short", "vp_vah_dist_short", "vp_val_dist_short", "vp_concentration_short",
            "vp_poc_dist_long",  "vp_vah_dist_long",  "vp_val_dist_long",  "vp_concentration_long",
            "vp_poc_convergence",
            "dist_ath_30d",   "dist_atl_30d",   "price_pos_30d",
            "dist_ath_90d",   "dist_atl_90d",   "price_pos_90d",
            "dist_ath_365d",  "dist_atl_365d",  "price_pos_365d",
            "momentum_7d", "momentum_30d", "momentum_90d",
            "round_level_dist", "price_vs_ma200m",
            "session_position",
            "funding_rate", "funding_rate_1d", "funding_rate_dev",
        }
        structural_cols = [c for c in all_cols if c in _STRUCTURAL_COLS]
        dynamic_cols    = [c for c in all_cols if c not in _STRUCTURAL_COLS]

        # IT/EN: dyn prima poi struct → prime N_DYN colonne = stream A | dyn first then struct → first N_DYN cols = stream A
        ordered_cols = dynamic_cols + structural_cols
        self.feature_cols     = ordered_cols
        self.n_dynamic_features = len(dynamic_cols)

        log.info(
            f"Features: {len(dynamic_cols)} dinamiche (stream A) + "
            f"{len(structural_cols)} strutturali (stream B) = "
            f"{len(ordered_cols)} totale"
        )

        # IT: Defragmenta prima della normalizzazione (100+ insert frammentano il DF).
        # EN: Defragment before normalization (100+ column inserts fragment the DF).
        df = df.copy()

        if normalize:
            df = self._normalize(df, fit=fit)

        n_before = len(df)
        df = df.dropna(subset=["target_ret"]).reset_index(drop=True)
        log.info(f"Rimosse {n_before - len(df)} righe NaN — {len(df)} valide")
        return df


# IT: Crea le sliding window (n, window, feat) via stride_tricks, scarta i NaN.
# EN: Builds sliding windows (n, window, feat) via stride_tricks, drops NaNs.
def create_windows(df: pd.DataFrame, feature_cols: list[str],
                   window_size: int = 60, target_col: str = "target_ret",
                   window_stride: int = 1):
    """
    Crea windows (n, window, features) per la LSTM.

    Usa numpy stride_tricks invece di un loop Python: con dataset da 2M+
    candele il loop originale richiederebbe minuti, stride_tricks è O(1)
    in tempo e crea una view (zero-copy fino al filtro NaN).

    window_stride: campiona 1 window ogni N candele.
      stride=1  → tutte le window (default, max campioni, alta RAM)
      stride=10 → 10x meno window (~7 GB per 2.8M candele)
      stride=20 → 20x meno window (~3.7 GB, consigliato per dataset >1M candele)
      Il training è equivalente perché le window adiacenti sono quasi identiche
      (differiscono solo di 1 candela) — il stride riduce la ridondanza.
    """
    scaled_cols = [c for c in feature_cols if c in df.columns]

    feat  = df[scaled_cols].values.astype(np.float32)
    tgt   = df[target_col].values.astype(np.float32)
    times = df["open_time"].values

    n, n_feat = feat.shape

    # IT/EN: stima RAM pre-alloc (warn se OOM probabile) | RAM estimate pre-alloc (warn if OOM likely)
    stride_eff   = max(1, int(window_stride))
    n_windows_est = (n - window_size) // stride_eff
    ram_gb_est    = n_windows_est * window_size * n_feat * 4 / 1e9
    if ram_gb_est > 8.0:
        log.warning(
            f"create_windows: stima RAM = {ram_gb_est:.1f} GB "
            f"({n_windows_est:,} windows × {window_size} × {n_feat} × float32). "
            f"Se OOM, aumenta window_stride nel config (ora={stride_eff})."
        )
    else:
        log.info(
            f"create_windows: stima RAM = {ram_gb_est:.1f} GB "
            f"({n_windows_est:,} windows, stride={stride_eff})"
        )

    # IT/EN: sliding_window_view zero-copy → shape (n-w+1, 1, w, f)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(feat, window_shape=(window_size, n_feat))[:, 0, :, :]

    max_idx = min(len(windows), n - window_size - 1)
    # IT/EN: stride sulla view zero-copy prima della copia finale | stride on zero-copy view before final copy
    wins  = windows[:max_idx:stride_eff]
    y_raw = tgt   [window_size: window_size + max_idx: stride_eff]
    t_raw = times [window_size: window_size + max_idx: stride_eff]

    # IT/EN: scarta finestre con NaN (righe iniziali con rolling incompleto) | drop NaN windows (incomplete rolling at start)
    valid = ~np.isnan(wins).any(axis=(1, 2))
    X = np.ascontiguousarray(wins[valid], dtype=np.float32)
    y = y_raw[valid].astype(np.float32)
    t = t_raw[valid]

    log.info(f"Windows: X={X.shape}  y={y.shape}  ({X.shape[-1]} features)  "
             f"stride={stride_eff}  ({(~valid).sum()} finestre NaN scartate)")
    return X, y, t


# IT: Split temporale train/val/test (no shuffle) per il training finale.
# EN: Temporal train/val/test split (no shuffle) for final training.
def temporal_split(X, y, t, val_frac=0.10, test_frac=0.10):
    """
    Split TEMPORALE semplice — usato in produzione per il training finale.
    Mantiene per compatibilità con 01_download_data.py.
    Per la valutazione robusta del modello usa walk_forward_folds().
    """
    n = len(X)
    iv = int(n * (1 - val_frac - test_frac))
    it = int(n * (1 - test_frac))
    return {
        "X_train": X[:iv],   "y_train": y[:iv],   "t_train": t[:iv],
        "X_val":   X[iv:it], "y_val":   y[iv:it], "t_val":   t[iv:it],
        "X_test":  X[it:],   "y_test":  y[it:],   "t_test":  t[it:],
    }


# IT: Purged walk-forward k-fold con embargo (stima robusta, no look-ahead).
# EN: Purged walk-forward k-fold with embargo (robust estimate, no look-ahead).
def walk_forward_folds(
    X:             np.ndarray,
    y:             np.ndarray,
    t:             np.ndarray,
    n_folds:       int = 3,
    embargo_steps: int = 60,
    val_frac:      float = 0.10,
) -> list[dict]:
    """
    Purged Walk-Forward k-Fold con embargo period.

    FIX CONCETTUALE — Overfitting temporale mascherato:
    ────────────────────────────────────────────────────
    Il problema dello split singolo (80/10/10):
      Il modello viene valutato su un unico periodo di test (l'ultimo 10%).
      Se quel periodo è fortunatamente favorevole (bull run, bassa volatilità),
      le metriche sembrano buone ma non sono generalizzabili.
      Viceversa, un mercato difficile nell'ultimo 10% può far sembrare
      il modello peggiore di quanto sia.

    La soluzione — Purged Walk-Forward:
      Divide il dataset in K fold temporali. Per ogni fold:
        · Train:    tutti i dati prima del fold (expanding window)
        · Embargo:  `embargo_steps` campioni scartati tra train e val
                    (evita che il gradiente dell'ultima candela di training
                     si propaghi nel validation attraverso autocorrelazioni)
        · Val:      il fold corrente
      Le metriche vengono mediate su tutti i fold → stima robusta.

    Embargo period:
      Con finestre di 60 minuti, la candela t del val set è predetta
      usando feature che includono candele fino a t-1. Se train termina
      a t-window, le ultime finestre di training e le prime di validation
      si sovrappongono parzialmente. L'embargo scarta questi campioni.

    Args:
        n_folds:       numero di fold (3 consigliato con ≤10k candele)
        embargo_steps: campioni da scartare tra fine training e inizio val
        val_frac:      frazione usata per validation all'interno di ogni fold

    Returns:
        Lista di dizionari, uno per fold:
          {fold, X_train, y_train, t_train, X_val, y_val, t_val,
           train_end_idx, val_start_idx, val_end_idx}
    """
    n      = len(X)
    folds  = []

    # IT: Fold 0 = periodo val più antico; fold K-1 = test set classico.
    # EN: Fold 0 = earliest val period; fold K-1 = classic test set.
    fold_size = n // (n_folds + 1)   # IT/EN: +1 = il primo fold deve avere training | +1 so first fold has training

    for k in range(n_folds):
        # IT/EN: fold k = finestra [val_start, val_end) | fold k = window [val_start, val_end)
        val_start = (k + 1) * fold_size
        val_end   = min(val_start + fold_size, n)

        if val_end - val_start < 50:
            log.warning(f"Fold {k}: troppo pochi campioni ({val_end-val_start}), skip.")
            continue

        # IT/EN: train = tutto prima del val, meno embargo | train = everything before val, minus embargo
        train_end = val_start - embargo_steps
        if train_end < fold_size:
            log.warning(f"Fold {k}: training troppo corto ({train_end} campioni), skip.")
            continue

        # IT/EN: val interna = ultime val_frac del train per early stopping | inner val = last val_frac for early stopping
        iv = int(train_end * (1 - val_frac))

        folds.append({
            "fold":           k,
            "X_train":        X[:iv],
            "y_train":        y[:iv],
            "t_train":        t[:iv],
            "X_val_internal": X[iv:train_end],          # IT/EN: early stopping
            "y_val_internal": y[iv:train_end],
            "t_val_internal": t[iv:train_end],
            "X_val":          X[val_start:val_end],     # IT/EN: held-out test fold
            "y_val":          y[val_start:val_end],
            "t_val":          t[val_start:val_end],
            "train_end_idx":  train_end,
            "val_start_idx":  val_start,
            "val_end_idx":    val_end,
            "embargo_steps":  embargo_steps,
        })

        log.info(
            f"Fold {k}: train=[0,{iv}] | embargo=[{iv},{val_start}] "
            f"| val=[{val_start},{val_end}] ({val_end-val_start} campioni)"
        )

    if not folds:
        raise ValueError(
            f"Nessun fold valido con n={n}, n_folds={n_folds}, embargo={embargo_steps}. "
            f"Scarica più dati (consigliato: almeno {n_folds * fold_size * 3} campioni)."
        )

    log.info(
        f"Walk-forward: {len(folds)} fold validi | "
        f"embargo={embargo_steps} candele | fold_size≈{fold_size}"
    )
    return folds
