"""
test_vp_golden.py — golden/regressione per l'ottimizzazione B2 (rolling min/max del Volume Profile).
test_vp_golden.py — golden/regression for B2 optimization (Volume Profile rolling min/max).

IT: B2 ha sostituito il `lo_arr[sl].min()/hi_arr[sl].max()` ricalcolato per-finestra in `_vp_single`
    con un rolling min/max precomputato una volta per scala. Questo test prova la BIT-IDENTITÀ delle
    4 feature VP contro un ORACOLO naive indipendente (loop per-finestra esplicito), su dati sintetici
    deterministici, per tutte le scale usate in produzione (60/240/1440). Hermetico: nessun dato esterno,
    nessun golden binario su disco.
EN: B2 replaced the per-window-recomputed `lo_arr[sl].min()/hi_arr[sl].max()` in `_vp_single` with a
    rolling min/max precomputed once per scale. This test proves BIT-IDENTITY of the 4 VP features vs an
    independent naive ORACLE (explicit per-window loop), on deterministic synthetic data, for every
    production scale (60/240/1440). Hermetic: no external data, no on-disk binary golden.
"""
import numpy as np
import pytest

from quantsys.features import FeatureBuilder


def _naive_vp_single(fb, tp_arr, vl_arr, lo_arr, hi_arr, cl_arr, lookback, suffix, df_len, vp_stride=1):
    # IT: replica FEDELE di _vp_single MA con lo_/hi_ per-finestra espliciti (pre-B2) = oracolo.
    # EN: FAITHFUL replica of _vp_single BUT with explicit per-window lo_/hi_ (pre-B2) = oracle.
    poc_d, vah_d, val_d, conc_d = {}, {}, {}, {}
    sampled = list(range(lookback, df_len, max(1, vp_stride)))
    for i in sampled:
        sl = slice(i - lookback, i)
        tp = tp_arr[sl]; vol = vl_arr[sl]
        lo_ = lo_arr[sl].min(); hi_ = hi_arr[sl].max()      # IT/EN: per-finestra (pre-B2) | per-window (pre-B2)
        if hi_ <= lo_:
            continue
        step = (hi_ - lo_) / fb.vp_bins
        idx_arr = np.clip(((tp - lo_) / step).astype(int), 0, fb.vp_bins - 1)
        bin_vol = np.bincount(idx_arr, weights=vol, minlength=fb.vp_bins)
        poc_idx = int(bin_vol.argmax()); poc_price = lo_ + (poc_idx + 0.5) * step
        total = bin_vol.sum()
        sorted_idx = np.argsort(bin_vol)[::-1]
        cum_sorted = np.cumsum(bin_vol[sorted_idx])
        n_va = int(np.searchsorted(cum_sorted, 0.70 * total)) + 1
        va = sorted_idx[:n_va]
        va_lo = lo_ + int(va.min()) * step
        va_hi = lo_ + (int(va.max()) + 1) * step
        curr = cl_arr[i]; safe = max(curr, 1e-9)
        poc_d[i] = (curr - poc_price) / safe
        vah_d[i] = (curr - va_hi) / safe
        val_d[i] = (curr - va_lo) / safe
        conc_d[i] = bin_vol[poc_idx] / (total + 1e-9)

    def fill(d):
        arr = np.full(df_len, np.nan)
        if not d:
            return arr
        idxs = np.array(sorted(d), dtype=np.int64)
        vals = np.array([d[k] for k in idxs], dtype=np.float64)
        arr[idxs] = vals
        m = np.isnan(arr)
        idx = np.where(~m, np.arange(df_len), 0)
        np.maximum.accumulate(idx, out=idx)
        return arr[idx]

    return {f"vp_poc_dist{suffix}": fill(poc_d), f"vp_vah_dist{suffix}": fill(vah_d),
            f"vp_val_dist{suffix}": fill(val_d), f"vp_concentration{suffix}": fill(conc_d)}


def _synthetic_ohlc(n=2000, seed=42):
    # IT: serie OHLC deterministica con low<=high e volume>0. | EN: deterministic OHLC, low<=high, vol>0.
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    spread = np.abs(rng.normal(0, 0.3, n)) + 0.01
    hi = close + spread
    lo = close - spread
    tp = (hi + lo + close) / 3.0
    vol = np.abs(rng.normal(100, 10, n)) + 1.0
    return tp, vol, lo, hi, close


@pytest.mark.parametrize("lookback,suffix", [(60, "_short"), (240, ""), (1440, "_long")])
def test_b2_vp_single_bit_identical_to_naive(lookback, suffix):
    # IT: l'_vp_single ottimizzato (B2) == oracolo naive per-finestra, bit-a-bit, ogni scala.
    # EN: optimized _vp_single (B2) == naive per-window oracle, bit-for-bit, every scale.
    fb = FeatureBuilder(vp_stride=1)
    tp, vol, lo, hi, cl = _synthetic_ohlc(n=2000)
    n = len(cl)
    opt = fb._vp_single(tp, vol, lo, hi, cl, lookback, suffix, n, vp_stride=1)
    ref = _naive_vp_single(fb, tp, vol, lo, hi, cl, lookback, suffix, n, vp_stride=1)
    assert set(opt) == set(ref)
    for k in opt:
        a, b = opt[k], ref[k]
        # IT: uguaglianza bit-a-bit trattando NaN==NaN come uguali | EN: bit-for-bit, NaN==NaN equal
        eq = (a == b) | (np.isnan(a) & np.isnan(b))
        assert eq.all(), f"{k}: {int((~eq).sum())} celle divergenti (scala {lookback})"


def test_b2_rolling_minmax_window_alignment():
    # IT: invariante centrale di B2: roll[i-1] copre ESATTAMENTE lo_arr[i-lookback:i] (no off-by-one).
    # EN: B2 core invariant: roll[i-1] spans EXACTLY lo_arr[i-lookback:i] (no off-by-one).
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 1500
    lo = np.cumsum(rng.normal(0, 1, n)) + 500
    hi = lo + np.abs(rng.normal(0, 1, n)) + 0.5
    for lookback in (60, 240, 1440):
        roll_lo = pd.Series(lo).rolling(lookback, min_periods=lookback).min().to_numpy()
        roll_hi = pd.Series(hi).rolling(lookback, min_periods=lookback).max().to_numpy()
        for i in range(lookback, n, 137):  # IT/EN: campione sparso | sparse sample
            assert roll_lo[i - 1] == lo[i - lookback:i].min()
            assert roll_hi[i - 1] == hi[i - lookback:i].max()
