"""xs_01 — Cross-sectional data acquisition: download the perp universe.

Downloads 1m klines + funding for the USDT-perp universe (PerpUniverse) over
the SAME span as data/raw_candles.parquet (BTCUSDT, 2025-05-19 → BTC history
end), saving one parquet per symbol with a schema IDENTICAL to the
single-asset one. This is the data layer for the cross-sectional IC probe:
testing whether the BTC model's μ has cross-sectional rank skill across
assets.

Output:
  data/xs/raw/{SYMBOL}.parquet     — schema == data/raw_candles.parquet
  data/xs/funding/{SYMBOL}.parquet — schema == data/funding_rate.parquet

Run: python scripts/xs_01_download.py   (working dir = project root)
"""
import logging
import sys
import time
from pathlib import Path

# force UTF-8 on stdout/stderr — avoids Windows cp1252 UnicodeEncodeError on the
# non-ASCII glyphs (→, ·) of the final report when output is redirected.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from quantsys.utils import load_config, setup_logging
from quantsys.utils.atomic_save import atomic_save_parquet
from quantsys.data import fetch_klines, fetch_funding_rate
from quantsys.data.universe import PerpUniverse

setup_logging()
log = logging.getLogger("quantsys.script.xs_01")

# Canonical raw schema (column order IDENTICAL to data/raw_candles.parquet).
RAW_COLS = [
    "open_time", "close_time", "open", "high", "low", "close", "volume",
    "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote_vol",
]

# Minimum span-coverage threshold: below it the symbol is dropped (history
# too short for a stable cross-sectional probe).
MIN_COVERAGE = 0.60

# Polite per-symbol sleep to avoid saturating Binance's weight rate-limit
# (the fetchers already handle intra-symbol pagination; this is inter-symbol).
INTER_SYMBOL_SLEEP = 1.0


# Derive the reference span from the existing single-asset BTC dataset.
def _reference_span(cfg) -> tuple[str, pd.Timestamp]:
    dcfg = cfg["data"]
    start_time = dcfg.get("start_time", "2025-05-19")
    raw_path = Path(dcfg.get("raw_path", "./data/raw_candles.parquet"))
    # span end = last open_time of BTC raw (exact alignment).
    btc = pd.read_parquet(raw_path, columns=["open_time"])
    span_end = btc["open_time"].max()
    # expected 1m candles over the span (for coverage computation).
    start_ts = pd.Timestamp(start_time, tz="UTC")
    expected = int((span_end - start_ts).total_seconds() // 60) + 1
    log.info(
        f"Span di riferimento (da BTC raw): {start_ts} → {span_end}  "
        f"(~{expected:,} candele 1m attese)."
    )
    return start_time, span_end, expected


# Download and persist a symbol's 1m klines, truncated to the BTC span.
def _download_raw(symbol: str, start_time: str, span_end: pd.Timestamp,
                  out_dir: Path) -> pd.DataFrame:
    df = fetch_klines(symbol, "1m", limit=0, start_time=start_time)
    # truncate to the BTC span end so all symbols share the same end.
    df = df[df["open_time"] <= span_end].copy()
    # re-align exactly to the canonical schema (column order + RangeIndex).
    df = df[RAW_COLS].reset_index(drop=True)
    atomic_save_parquet(df, out_dir / f"{symbol}.parquet", index=False)
    return df


# Download and persist the symbol's funding, truncated to the BTC span.
# fetch_funding_rate saves to output_dir/funding_rate.parquet; we rename.
def _download_funding(symbol: str, start_time: str, span_end: pd.Timestamp,
                      fund_dir: Path) -> int:
    target = fund_dir / f"{symbol}.parquet"
    tmp_dir = fund_dir / f"_tmp_{symbol}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    default_out = tmp_dir / "funding_rate.parquet"
    try:
        fetch_funding_rate(symbol=symbol, start_time=start_time, output_dir=str(tmp_dir))
        if not default_out.exists():
            log.warning(f"[{symbol}] funding: nessun file prodotto.")
            return 0
        fdf = pd.read_parquet(default_out)
        # truncate to the BTC span; schema == data/funding_rate.parquet.
        fdf = fdf[fdf["open_time"] <= span_end].reset_index(drop=True)
        atomic_save_parquet(fdf[["open_time", "funding_rate"]], target, index=False)
        return len(fdf)
    finally:
        # clean up the intermediate temp file.
        if default_out.exists():
            default_out.unlink()
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    cfg = load_config("config/default.yaml")
    start_time, span_end, expected = _reference_span(cfg)

    # cross-sectional output directories.
    raw_dir = Path("./data/xs/raw")
    fund_dir = Path("./data/xs/funding")
    raw_dir.mkdir(parents=True, exist_ok=True)
    fund_dir.mkdir(parents=True, exist_ok=True)

    # resolve the universe (top-N liquidity + anchors, BTCUSDT guaranteed).
    universe = PerpUniverse(n=20)
    symbols = universe.symbols()
    liq = universe.liquidity()
    log.info(f"Universo ({len(symbols)} simboli): {symbols}")

    report = []   # report rows
    skipped = []  # dropped symbols
    total_candles = 0

    for i, sym in enumerate(symbols, 1):
        # idempotent resume — skip if raw+funding already present with OK coverage.
        raw_existing = raw_dir / f"{sym}.parquet"
        fund_existing = fund_dir / f"{sym}.parquet"
        if raw_existing.exists() and fund_existing.exists():
            try:
                _ex = pd.read_parquet(raw_existing, columns=["open_time"])
                _cov = len(_ex) / expected if expected else 0.0
                if _cov >= MIN_COVERAGE:
                    _f = pd.read_parquet(fund_existing, columns=["open_time"])
                    total_candles += len(_ex)
                    report.append((sym, len(_ex), _ex["open_time"].iloc[0],
                                   _ex["open_time"].iloc[-1], _cov, len(_f)))
                    log.info(f"[{i}/{len(symbols)}] {sym}: già presente ({_cov:.1%}) — skip.")
                    continue
            except Exception:
                pass  # corrupt file → re-download.
        log.info(f"[{i}/{len(symbols)}] {sym}: download klines 1m ...")
        try:
            df = _download_raw(sym, start_time, span_end, raw_dir)
        except Exception as e:
            log.error(f"[{sym}] download klines FALLITO: {e}")
            skipped.append((sym, "download_error", str(e)))
            time.sleep(INTER_SYMBOL_SLEEP)
            continue

        n_rows = len(df)
        coverage = n_rows / expected if expected else 0.0
        first_ts = df["open_time"].iloc[0] if n_rows else None
        last_ts = df["open_time"].iloc[-1] if n_rows else None

        # insufficient coverage → drop (also remove the written parquet).
        if coverage < MIN_COVERAGE:
            log.warning(
                f"[{sym}] copertura {coverage:.1%} < {MIN_COVERAGE:.0%} "
                f"({n_rows:,}/{expected:,}) — SCARTATO."
            )
            (raw_dir / f"{sym}.parquet").unlink(missing_ok=True)
            skipped.append((sym, "low_coverage", f"{coverage:.1%}"))
            time.sleep(INTER_SYMBOL_SLEEP)
            continue

        # funding (best-effort: missing funding does not drop the symbol).
        try:
            n_fund = _download_funding(sym, start_time, span_end, fund_dir)
        except Exception as e:
            log.warning(f"[{sym}] funding FALLITO ({e}) — continuo senza.")
            n_fund = 0

        total_candles += n_rows
        report.append((sym, n_rows, first_ts, last_ts, coverage, n_fund))
        log.info(
            f"[{sym}] OK: {n_rows:,} candele ({coverage:.1%} copertura), "
            f"{n_fund:,} funding obs  [{first_ts} → {last_ts}]  "
            f"vol24h={liq.get(sym, float('nan'))/1e9:.2f}B"
        )
        time.sleep(INTER_SYMBOL_SLEEP)

    # final console report.
    print("\n" + "=" * 72)
    print("  xs_01 · CROSS-SECTIONAL DOWNLOAD · REPORT")
    print("=" * 72)
    print(f"  Span: {start_time} → {span_end}  (~{expected:,} candele 1m attese)")
    print(f"  Universo richiesto: {len(symbols)} simboli")
    print(f"  Simboli salvati   : {len(report)}")
    print(f"  Candele totali    : {total_candles:,}")
    print("-" * 72)
    print(f"  {'SYMBOL':<12}{'ROWS':>10}{'COV':>8}{'FUND':>7}  SPAN")
    for sym, n_rows, first_ts, last_ts, cov, n_fund in report:
        f = first_ts.date() if first_ts is not None else "-"
        l = last_ts.date() if last_ts is not None else "-"
        print(f"  {sym:<12}{n_rows:>10,}{cov:>7.0%}{n_fund:>7}  {f} → {l}")
    if skipped:
        print("-" * 72)
        print("  SCARTATI:")
        for sym, reason, detail in skipped:
            print(f"    {sym:<12} {reason} ({detail})")
    print("=" * 72)
    print(f"  Output: {raw_dir}/*.parquet  +  {fund_dir}/*.parquet\n")


if __name__ == "__main__":
    main()
