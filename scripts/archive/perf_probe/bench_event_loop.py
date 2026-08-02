"""
Probe temporanea (PERF AUDIT) — costo per barra dell'event loop direzionale.
Temporary probe (PERF AUDIT) — per-bar cost of the directional event loop.

Il backtest 03 NON e' eseguibile end-to-end sullo stato su disco (il checkpoint
e' il modello VOL: il guard sigma fail-fasta, correttamente). Qui si isola il
solo anello Python — SignalGenerator + RiskManager sulle stesse classi di
produzione, guidate da mu/sigma sintetici — per rispondere alla domanda "vale
Numba?" senza toccare il path production.
Backtest 03 is NOT runnable end-to-end against the on-disk state (the checkpoint
is the VOL model: the sigma guard fail-fasts, correctly). Here we isolate the
Python loop alone — production SignalGenerator + RiskManager driven by synthetic
mu/sigma — to answer "is Numba worth it?" without touching the production path.

Uso / Usage: python scripts/archive/perf_probe/bench_event_loop.py
"""
import sys
import time
from pathlib import Path

import numpy as np
# IT: pandas PRIMA di torch/sklearn — obbligatorio su questa macchina: l'ordine
#     inverso fa crashare pyarrow con access violation (vedi docs/PERF_AUDIT.md).
# EN: pandas BEFORE torch/sklearn — mandatory on this box: the reverse order
#     crashes pyarrow with an access violation (see docs/PERF_AUDIT.md).
import pandas as pd  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from quantsys.utils import load_config  # noqa: E402
from quantsys.trading import RiskManager, SignalGenerator, Side  # noqa: E402
from quantsys.utils.stats import bootstrap_sharpe_ci, mdd_stats  # noqa: E402


def main():
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    cfg = load_config(str(ROOT / "config/default.yaml"))
    rcfg, bcfg = cfg["risk"], cfg["backtest"]

    # IT: prezzi reali dal parquet (l'anello tocca ATR/OHLC veri).
    # EN: real prices from the parquet (the loop touches true ATR/OHLC).

    raw = pd.read_parquet(ROOT / "data/raw_candles.parquet").tail(6485).reset_index(drop=True)
    ohlcv = raw[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    n = len(ohlcv)
    tr = np.maximum(ohlcv[:, 1] - ohlcv[:, 2],
                    np.abs(ohlcv[:, 1] - np.roll(ohlcv[:, 3], 1)))
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()

    # IT: mu/sigma sintetici in spazio RAW plausibile (h=30 barre 1h).
    # EN: synthetic mu/sigma in a plausible RAW space (h=30 1h-bars).
    rng = np.random.default_rng(42)
    mu_a = rng.normal(0.0, 0.004, n)
    sig_a = np.abs(rng.normal(0.020, 0.004, n)) + 0.005
    nu_a = np.full(n, 5.0)

    sig_gen = SignalGenerator(
        prob_threshold=bcfg["prob_threshold"],
        min_expected_ret=bcfg["min_expected_ret"],
        max_sigma=bcfg["max_sigma"],
        conviction_alpha=bcfg.get("conviction_alpha", 0.5),
    )

    def run_loop():
        rm = RiskManager(
            initial_capital=rcfg["initial_capital"],
            max_risk_per_trade=rcfg["max_risk_per_trade"],
            sl_atr_mult=rcfg["sl_atr_mult"], tp_rr_ratio=rcfg["tp_rr_ratio"],
            max_position_pct=rcfg["max_position_pct"],
            max_drawdown_stop=rcfg["max_drawdown_stop"],
            max_hold_candles=rcfg["max_hold_candles"],
            use_trailing_stop=rcfg["use_trailing_stop"],
            trailing_atr_mult=rcfg["trailing_atr_mult"],
            fee_rate=bcfg["fee_rate"], slippage_rate=bcfg["slippage_rate"],
            slippage_model=bcfg.get("slippage_model", "fixed"),
            bars_per_year=8760,
        )
        for i in range(n - 1):
            if rm.circuit_breaker:
                break
            o_c, h_c, l_c, c_c = ohlcv[i]
            o_n, h_n, l_n, c_n = ohlcv[i + 1]
            atr_i = max(atr[i], c_c * 0.0005)
            mu, sigma, nu = float(mu_a[i]), float(sig_a[i]), float(nu_a[i])
            side, dist = sig_gen.generate(mu, sigma, nu)
            # IT: stesso idioma di 03_backtest — `position` e' Optional[Position].
            # EN: same idiom as 03_backtest — `position` is Optional[Position].
            if rm.position:
                rm.update_trailing(c_c, atr_i)
                reason = rm.check_exit(h_n, l_n, c_n, i + 1, side)
                if reason is not None:
                    rm.close_position(reason, c_n, i + 1)
            if side != Side.NONE and not rm.position:
                rm.open_position(side, o_n, i + 1, atr_i, dist)
        return rm

    for _ in range(2):
        run_loop()
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        rm = run_loop()
    t = (time.perf_counter() - t0) / reps
    print(f"event loop direzionale: {n:,} barre in {t*1000:.1f} ms "
          f"→ {t/n*1e6:.2f} µs/barra   ({len(rm.trades)} trade)")
    print(f"  proiezione su test split (6.5k barre) : {t:.3f} s")
    print(f"  proiezione su dataset intero (66k)    : {t*66410/n:.2f} s")

    # ── bootstrap CI: gia' vettorizzato? ────────────────────────────────────
    pnl = [tr_.net_pnl for tr_ in rm.trades] or list(rng.normal(0, 50, 200))
    if len(pnl) < 30:
        pnl = list(rng.normal(0, 50, 200))
    t0 = time.perf_counter()
    for _ in range(5):
        bootstrap_sharpe_ci(pnl, annualize=8760)
    print(f"\nbootstrap_sharpe_ci (5000 resample, n={len(pnl)}): "
          f"{(time.perf_counter()-t0)/5*1000:.1f} ms")

    eq = np.cumsum(rng.normal(0, 10, 6485)) + 10000
    t0 = time.perf_counter()
    for _ in range(5):
        mdd_stats(eq)
    print(f"mdd_stats (loop Python su {len(eq):,} punti equity): "
          f"{(time.perf_counter()-t0)/5*1000:.2f} ms")


if __name__ == "__main__":
    main()
