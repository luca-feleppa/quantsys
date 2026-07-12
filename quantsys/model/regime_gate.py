"""A3 — Costruzione del gate regime causale per la testa regime-MoE.

IT: Allinea le filtered probabilities di RegimeMarkovBTC (`data/regime_probs.parquet`,
    index orario UTC, colonne regime_prob_0/1/2 — CAUSALI: Hamilton filter
    forward-only) ai timestamp dei sample del dataset via `merge_asof` BACKWARD
    (ultimo regime noto ≤ t: mai lookahead). Burn-in/gap → riga uniforme (1/3).
    Stesso meccanismo di allineamento della stratificazione val di 02_train
    (`_load_val_regimes`): unica differenza, qui si estraggono le 3 probabilità
    invece del regime dominante.
EN: Aligns the RegimeMarkovBTC filtered probabilities (`data/regime_probs.parquet`,
    hourly UTC index, columns regime_prob_0/1/2 — CAUSAL: forward-only Hamilton
    filter) to the dataset sample timestamps via BACKWARD `merge_asof` (last
    known regime ≤ t: never lookahead). Burn-in/gaps → uniform row (1/3).
    Same alignment mechanism as 02_train's val stratification
    (`_load_val_regimes`): the only difference is extracting the 3 probabilities
    instead of the dominant regime.
"""
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("quantsys.model.regime_gate")

# IT: colonne del gate nel parquet (ordine = indice regime R0/R1/R2).
# EN: gate columns in the parquet (order = regime index R0/R1/R2).
GATE_COLS = ["regime_prob_0", "regime_prob_1", "regime_prob_2"]

# IT: path canonico del parquet dei regimi (prodotto da 01b_download_macro).
# EN: canonical path of the regime parquet (produced by 01b_download_macro).
DEFAULT_REGIME_PARQUET = "data/regime_probs.parquet"


# IT: Costruisce l'array gate (N,3) float32 allineato ai timestamp dei sample.
# EN: Builds the (N,3) float32 gate array aligned to the sample timestamps.
def build_regime_gate(timestamps,
                      parquet_path: str = DEFAULT_REGIME_PARQUET) -> np.ndarray:
    """
    IT: timestamps: array-like di datetime64 (es. `t_train`/`t_val`/`t_test` del
        dataset npz). Ritorna G (N,3) con righe sul simplesso; sample senza
        regime noto (prima dell'inizio del parquet) → (1/3,1/3,1/3).
        Fail-fast su parquet mancante o colonne assenti (il gate è un input
        esterno obbligatorio del regime-MoE, non un optional).
    EN: timestamps: array-like of datetime64 (e.g. the npz dataset's
        `t_train`/`t_val`/`t_test`). Returns G (N,3) with rows on the simplex;
        samples with no known regime (before the parquet start) → (1/3,1/3,1/3).
        Fails fast on missing parquet or missing columns (the gate is a
        mandatory external input of the regime-MoE, not an optional).
    """
    import pandas as pd

    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(
            f"build_regime_gate: {p} non trovato — rigenerare con "
            f"01b_download_macro.py / not found — regenerate via 01b_download_macro.py"
        )
    df = pd.read_parquet(p)
    missing = [c for c in GATE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"build_regime_gate: colonne mancanti/missing columns {missing} in {p}")

    # IT: normalizzazione ns tz-naive su ENTRAMBI i lati (stesso helper della
    #     stratificazione val: merge_asof richiede stessa risoluzione, no tz mix).
    # EN: ns tz-naive normalization on BOTH sides (same helper as the val
    #     stratification: merge_asof needs identical resolution, no tz mix).
    def _to_ns_naive(idx_or_series):
        s = pd.to_datetime(idx_or_series)
        if getattr(s, "tz", None) is not None:
            s = s.tz_convert("UTC").tz_localize(None)
        return s.astype("datetime64[ns]")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"build_regime_gate: index di {p} non è DatetimeIndex / "
                        f"index of {p} is not a DatetimeIndex")
    df = df.sort_index()
    df.index = _to_ns_naive(df.index)

    ts = _to_ns_naive(pd.Index(pd.to_datetime(np.asarray(timestamps))))
    df_t = pd.DataFrame({"_t": ts})

    # IT: BACKWARD = ultimo regime noto ≤ t (causale, mai forward).
    # EN: BACKWARD = last known regime ≤ t (causal, never forward).
    merged = pd.merge_asof(
        df_t.sort_values("_t"),
        df[GATE_COLS].reset_index().rename(
            columns={df.index.name or "index": "_t"}),
        on="_t", direction="backward",
    )
    G = merged[GATE_COLS].to_numpy(dtype=np.float64)

    # IT: merge_asof ordina per _t — riporta le righe nell'ordine originale dei sample.
    # EN: merge_asof sorts by _t — restore the original sample ordering.
    order = np.argsort(np.argsort(ts.values))
    G = G[order]

    # IT: burn-in/gap (NaN o riga degenere) → gate uniforme; poi rinormalizza le
    #     righe al simplesso (le prob filtrate possono deviare da 1 per roundoff).
    # EN: burn-in/gap (NaN or degenerate row) → uniform gate; then renormalize
    #     rows onto the simplex (filtered probs can drift from 1 by roundoff).
    bad = ~np.isfinite(G).all(axis=1) | (np.nansum(G, axis=1) <= 0)
    if bad.any():
        G[bad] = 1.0 / len(GATE_COLS)
    G = G / G.sum(axis=1, keepdims=True)

    log.info(
        f"build_regime_gate: {len(G)} sample allineati/aligned, "
        f"{int(bad.sum())} fallback uniformi/uniform fallbacks, "
        f"medie gate/mean gate = {np.round(G.mean(axis=0), 3).tolist()}"
    )
    return G.astype(np.float32)
