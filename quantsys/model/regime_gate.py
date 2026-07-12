"""A3 — Costruzione del gate regime causale per la testa regime-MoE.

IT: Allinea le filtered probabilities di RegimeMarkovBTC (`data/regime_probs.parquet`,
    index orario UTC, colonne regime_prob_0/1/2 — CAUSALI: Hamilton filter
    forward-only) ai timestamp dei sample del dataset. ⚠ CONVENZIONE DI LABELING
    (audit BLOCKER-1, 2026-07-12): la riga etichettata `t` del parquet contiene
    l'osservazione della barra `[t, t+1h)` (log_ret/log_rv resample label-left)
    → è DISPONIBILE solo a `t+1h`. L'indice viene quindi SHIFTATO a availability
    time (+1 barra oraria — il clock del regime detector è orario by design)
    PRIMA del `merge_asof` BACKWARD: ultimo regime *disponibile* ≤ t, mai
    lookahead. Burn-in (pre-inizio) → riga uniforme (1/3); gap interni →
    last-known entro `max_age`, uniforme oltre (staleness bounded, audit
    MAJOR-1); coda stale > 20% dei sample → fail-fast.
    NB: `_load_val_regimes` di 02_train ha la convenzione exact-match SENZA
    shift — lì il regime è solo stratificazione diagnostica, NON input del
    modello: qui il rigore è d'obbligo.
EN: Aligns the RegimeMarkovBTC filtered probabilities (`data/regime_probs.parquet`,
    hourly UTC index, columns regime_prob_0/1/2 — CAUSAL: forward-only Hamilton
    filter) to the dataset sample timestamps. ⚠ LABELING CONVENTION (BLOCKER-1
    audit, 2026-07-12): the parquet row labeled `t` holds the observation of bar
    `[t, t+1h)` (label-left resample) → it becomes AVAILABLE only at `t+1h`. The
    index is therefore SHIFTED to availability time (+1 hourly bar — the regime
    detector clock is hourly by design) BEFORE the BACKWARD `merge_asof`: last
    *available* regime ≤ t, never lookahead. Burn-in (pre-start) → uniform row
    (1/3); internal gaps → last-known within `max_age`, uniform beyond (bounded
    staleness, MAJOR-1 audit); stale tail > 20% of samples → fail-fast.
    NB: 02_train's `_load_val_regimes` keeps the unshifted exact-match
    convention — there the regime is diagnostic stratification only, NOT a
    model input: here rigour is mandatory.
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


# IT: lag di disponibilità della riga regime (clock orario del detector by design:
#     riga `t` = barra [t, t+1h) → disponibile a t+1h). Audit BLOCKER-1.
# EN: availability lag of a regime row (hourly detector clock by design: row `t`
#     = bar [t, t+1h) → available at t+1h). BLOCKER-1 audit.
AVAILABILITY_LAG = "1h"

# IT: età massima dell'ultimo regime disponibile prima del fallback uniforme
#     (staleness bounded, audit MAJOR-1); oltre il 20% di sample stale → fail-fast.
# EN: max age of the last available regime before the uniform fallback (bounded
#     staleness, MAJOR-1 audit); beyond 20% stale samples → fail-fast.
DEFAULT_MAX_AGE = "168h"
STALE_FRAC_FAIL = 0.20


# IT: Costruisce l'array gate (N,3) float32 allineato ai timestamp dei sample.
# EN: Builds the (N,3) float32 gate array aligned to the sample timestamps.
def build_regime_gate(timestamps,
                      parquet_path: str = DEFAULT_REGIME_PARQUET,
                      max_age: str = DEFAULT_MAX_AGE) -> np.ndarray:
    """
    IT: timestamps: array-like di datetime64 (es. `t_train`/`t_val`/`t_test` del
        dataset npz). Ritorna G (N,3) con righe sul simplesso; sample senza
        regime DISPONIBILE (prima dell'inizio del parquet, o ultimo regime più
        vecchio di `max_age`) → (1/3,1/3,1/3). Fail-fast su parquet mancante,
        colonne assenti, o coda stale > 20% dei sample (parquet da rigenerare
        con 01b — il gate è un input esterno obbligatorio del regime-MoE).
    EN: timestamps: array-like of datetime64 (e.g. the npz dataset's
        `t_train`/`t_val`/`t_test`). Returns G (N,3) with rows on the simplex;
        samples with no AVAILABLE regime (before the parquet start, or last
        regime older than `max_age`) → (1/3,1/3,1/3). Fails fast on missing
        parquet, missing columns, or a stale tail > 20% of samples (regenerate
        the parquet via 01b — the gate is a mandatory external regime-MoE input).
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

    # IT: BLOCKER-1 — shift dell'indice ad AVAILABILITY TIME: la riga etichettata
    #     `t` contiene la barra [t, t+1h) → utilizzabile solo da t+1h. Senza lo
    #     shift, il match esatto a t incorporerebbe il return della prima ora
    #     FUTURA (r² = predittore forte della RV forward → gate QLIKE gonfiato).
    # EN: BLOCKER-1 — index shift to AVAILABILITY TIME: the row labeled `t` holds
    #     bar [t, t+1h) → usable only from t+1h. Without the shift, the exact
    #     match at t would embed the FIRST FUTURE hour's return (r² = strong
    #     forward-RV predictor → spuriously inflated QLIKE gate).
    df.index = df.index + pd.Timedelta(AVAILABILITY_LAG)

    ts = _to_ns_naive(pd.Index(pd.to_datetime(np.asarray(timestamps))))
    df_t = pd.DataFrame({"_t": ts})

    # IT: colonna availability-time duplicata: sopravvive al merge → età del match.
    # EN: duplicated availability-time column: survives the merge → match age.
    right = df[GATE_COLS].reset_index().rename(
        columns={df.index.name or "index": "_t"})
    right["_avail"] = right["_t"]

    # IT: BACKWARD = ultimo regime DISPONIBILE ≤ t (causale, mai forward).
    # EN: BACKWARD = last AVAILABLE regime ≤ t (causal, never forward).
    merged = pd.merge_asof(
        df_t.sort_values("_t"), right,
        on="_t", direction="backward",
    )
    G = merged[GATE_COLS].to_numpy(dtype=np.float64)
    age = (merged["_t"] - merged["_avail"]).to_numpy()

    # IT: merge_asof ordina per _t — riporta le righe nell'ordine originale dei sample.
    # EN: merge_asof sorts by _t — restore the original sample ordering.
    order = np.argsort(np.argsort(ts.values))
    G = G[order]
    age = age[order]

    # IT: MAJOR-1 — staleness bounded: burn-in (nessun match → NaN) O ultimo
    #     regime più vecchio di max_age → gate uniforme, MAI last-known illimitato
    #     (parquet fermo + dataset che avanza = gate congelato silenzioso).
    # EN: MAJOR-1 — bounded staleness: burn-in (no match → NaN) OR last regime
    #     older than max_age → uniform gate, NEVER unbounded last-known (stalled
    #     parquet + advancing dataset = silently frozen gate).
    max_age_td = pd.Timedelta(max_age).to_timedelta64()
    stale = np.isnat(age) | (age > max_age_td)
    bad = stale | ~np.isfinite(G).all(axis=1) | (np.nansum(G, axis=1) <= 0)
    if bad.any():
        G[bad] = 1.0 / len(GATE_COLS)
    G = G / G.sum(axis=1, keepdims=True)

    n_stale = int((stale & ~np.isnat(age)).sum())
    if n_stale:
        log.warning(
            f"build_regime_gate: {n_stale}/{len(G)} sample con regime più vecchio "
            f"di {max_age} → gate uniforme (parquet da rigenerare? 01b) / samples "
            f"with regime older than {max_age} → uniform gate (regenerate via 01b?)"
        )
    if n_stale > STALE_FRAC_FAIL * len(G):
        raise RuntimeError(
            f"build_regime_gate: {n_stale}/{len(G)} sample stale (> "
            f"{STALE_FRAC_FAIL:.0%}) — regime_probs.parquet è fermo rispetto al "
            f"dataset: rigenerare con 01b PRIMA del training / stale beyond the "
            f"fail-fast bound — regenerate regime_probs.parquet via 01b BEFORE training"
        )

    log.info(
        f"build_regime_gate: {len(G)} sample allineati/aligned "
        f"(availability lag +{AVAILABILITY_LAG}), "
        f"{int(bad.sum())} fallback uniformi/uniform fallbacks "
        f"(di cui stale/of which stale: {n_stale}), "
        f"medie gate/mean gate = {np.round(G.mean(axis=0), 3).tolist()}"
    )
    return G.astype(np.float32)
