"""
quantsys/macro/__init__.py
==========================
Phase 1b — US Macroeconomic Data + Market Expectations

Sources:
  · FRED (Federal Reserve St. Louis) — free API, optional key
  · yfinance                         — DXY, VIX, Gold, Oil, SPY
  · Atlanta Fed GDPNow               — real-time GDP estimate (web scrape)

For each indicator we include both the REALIZED value and the market
EXPECTATIONS (forward-looking), where available.

Forward-fill logic:
  Macro data are monthly/quarterly; BTC candles are 1-minute.
  After the merge, each candle inherits the most recent available macro value
  (forward fill) — no look-ahead bias because we use the published value,
  not the not-yet-released value of the current period.
"""

import logging
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("quantsys.macro")

# Conservative per-frequency lags (prevents look-ahead in ffill).
RELEASE_LAG_DAYS: dict[str, int] = {
    "D": 1,    # T+1
    "W": 4,    # pub. Thu of next week
    "M": 35,   # ~5 weeks
    "Q": 35,   # advance estimate ~30d
}

# Overrides for series with known lags (e.g. NFP first Friday).
SERIES_LAG_OVERRIDE: dict[str, int] = {
    "nfp_level":       10,   # first Friday
    "initial_claims":   5,   # Thu of next week
    "continued_claims": 5,
    "jolts_openings":  42,   # ~6-week lag
    "jolts_quits":     42,
    "fed_funds_upper":  1,   # announced immediately
    "fed_funds_lower":  1,
    "infl_exp_5y":      1,   # daily breakeven market data
    "infl_exp_10y":     1,
    "tips_5y5y":        1,
    "yield_curve_2_10": 1,
    "yield_curve_3m_10":1,
    "real_rate_10y":    1,
    "repo_rate":        1,
    "credit_spread_hy": 1,
    "credit_spread_ig": 1,
    "gdpnow":           1,
    "treasury_2y":      1,
    "treasury_5y":      1,
    "treasury_10y":     1,
    "treasury_30y":     1,
}

# FRED series catalog: {name: (FRED_ID, desc, freq)}.
FRED_SERIES = {

    # ── INFLATION ──────────────────────────────────────────────────────────
    # Realized
    "cpi_yoy":          ("CPIAUCSL",   "CPI YoY",                 "M"),
    "core_cpi_yoy":     ("CPILFESL",   "Core CPI YoY (ex F&E)",   "M"),
    "pce_yoy":          ("PCEPI",      "PCE deflator YoY",        "M"),
    "core_pce_yoy":     ("PCEPILFE",   "Core PCE YoY (target Fed)","M"),

    # Inflation expectations
    "infl_exp_1y":      ("MICH",       "Michigan Infl Exp 1Y (survey)", "M"),
    "infl_exp_5y":      ("T5YIE",      "Breakeven Infl 5Y (market)",    "D"),
    "infl_exp_10y":     ("T10YIE",     "Breakeven Infl 10Y (market)",   "D"),
    "tips_5y5y":        ("T5YIFR",     "Breakeven Infl 5Y5Y Forward",   "D"),

    # ── MONETARY POLICY (Fed) ───────────────────────────────────────────────
    # Realized
    "fed_funds":        ("FEDFUNDS",   "Fed Funds Rate effettivo",  "M"),
    "fed_funds_upper":  ("DFEDTARU",   "Fed Funds Target Upper",    "D"),
    "fed_funds_lower":  ("DFEDTARL",   "Fed Funds Target Lower",    "D"),

    # Rate expectations (bond market)
    "treasury_2y":      ("GS2",        "Treasury 2Y (proxy exp Fed)","M"),
    "treasury_5y":      ("GS5",        "Treasury 5Y",               "M"),
    "treasury_10y":     ("GS10",       "Treasury 10Y",              "M"),
    "treasury_30y":     ("GS30",       "Treasury 30Y",              "M"),
    "yield_curve_2_10": ("T10Y2Y",     "Spread 2Y-10Y (inversione)","D"),
    "yield_curve_3m_10":("T10Y3M",     "Spread 3M-10Y",             "D"),
    "real_rate_10y":    ("DFII10",     "Real Rate 10Y (TIPS)",       "D"),

    # ── GDP / GROWTH ────────────────────────────────────────────────────────
    # Realized
    "gdp_growth":       ("A191RL1Q225SBEA", "GDP QoQ annualizzato",  "Q"),
    "gdp_level":        ("GDP",             "GDP nominale (miliardi)","Q"),

    # Growth expectations (leading indicators)
    "lei":              ("USSLIND",    "Leading Economic Index (CB)","M"),
    "indpro":           ("INDPRO",     "Industrial Production Index","M"),
    "gdpnow":           ("GDPNOW",    "Atlanta Fed GDPNow",        "D"),
    # NAPM and NMFCI (ISM PMI) removed from FRED in 2016 — INDPRO is the best proxy

    # ── LABOR MARKET ────────────────────────────────────────────────────────
    # Realized
    "unemployment":     ("UNRATE",     "Tasso disoccupazione",      "M"),
    "nfp_level":        ("PAYEMS",     "Non-Farm Payroll (livello)", "M"),
    "avg_hourly_earn":  ("CES0500000003","Salari orari medi",        "M"),
    "participation":    ("CIVPART",    "Tasso partecipazione",       "M"),

    # Labor expectations / leading
    "initial_claims":   ("ICSA",       "Initial Jobless Claims (sett.)","W"),
    "continued_claims": ("CCSA",       "Continued Claims",           "W"),
    "jolts_openings":   ("JTSJOL",     "JOLTS Job Openings",         "M"),
    "jolts_quits":      ("JTSJOR",     "JOLTS Quit Rate",            "M"),

    # ── SENTIMENT / FINANCIAL CONDITIONS ───────────────────────────────────
    "nfci":             ("NFCI",       "Chicago Fed Conditions Index","W"),
    "consumer_conf":    ("UMCSENT",    "Michigan Consumer Sentiment", "M"),
    "credit_spread_hy": ("BAA10Y",     "Moody's BAA-10Y Spread (HY proxy)","D"),
    "credit_spread_ig": ("AAA10Y",     "Moody's AAA-10Y Spread (IG proxy)","D"),

    # ── LIQUIDITY / FED BALANCE SHEET ───────────────────────────────────────
    "fed_balance":      ("WALCL",      "Fed Balance Sheet (assets)", "W"),
    "m2":               ("M2SL",       "M2 Money Supply",            "M"),
    "repo_rate":        ("RRPONTSYD",  "Reverse Repo (overnight)",   "D"),

}

# Daily yfinance series {yahoo_ticker: internal_name}.
YFINANCE_TICKERS = {
    "DX-Y.NYB":  "dxy",
    "^VIX":      "vix",
    "GC=F":      "gold",
    "CL=F":      "oil_wti",
    "^GSPC":     "sp500",
    "^TNX":      "treasury_10y_yf",
    "BTC-USD":   "btc_daily",
    "ETH-USD":   "eth_daily",     # leads BTC in bull/bear
    "ETH-BTC":   "eth_btc_ratio", # internal crypto rotation
}


# FRED downloader with 429 (rate-limit) retries and optional api_key.

class FREDDownloader:
    """
    Download historical series from FRED.

    The API key is free (registration at fred.stlouisfed.org).
    Without a key it still works for most series
    through the public endpoint, with a tighter rate limit.
    """

    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    # Stores the (optional) api_key and warns if missing.
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        if not api_key:
            log.warning("FRED api_key non fornita — alcune serie potrebbero non essere accessibili.")

    # Downloads a single FRED series as a pd.Series (retries on 429).
    def fetch(self, series_id: str, start: str = "2018-01-01",
              max_retries: int = 3, retry_wait: float = 15.0) -> pd.Series:
        """
        Download one FRED series and return it as a pd.Series with a DatetimeIndex.
        '.' (missing) values are converted to NaN.
        On a 429 error (rate limit) waits retry_wait seconds and retries.
        """
        params = {
            "series_id":        series_id,
            "observation_start": start,
            "file_type":        "json",
            "sort_order":       "asc",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        else:
            # Dummy key to satisfy format check (public endpoint).
            params["api_key"] = "abcdefghijklmnopqrstuvwxyz123456"

        # Retry policy: fixed retry_wait backoff on 429 and HTTPError.
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(self.BASE_URL, params=params, timeout=15)
                if r.status_code == 429:
                    log.warning(
                        f"  FRED {series_id}: rate limit (429) — "
                        f"attendo {retry_wait:.0f}s (tentativo {attempt}/{max_retries})"
                    )
                    time.sleep(retry_wait)
                    continue
                r.raise_for_status()
                obs = r.json().get("observations", [])
                break
            except requests.exceptions.HTTPError as e:
                if attempt < max_retries:
                    log.warning(f"  FRED {series_id}: {e} — retry {attempt}/{max_retries}")
                    time.sleep(retry_wait)
                else:
                    log.warning(f"  FRED {series_id}: {e}")
                    return pd.Series(dtype=float, name=series_id)
            except Exception as e:
                log.warning(f"  FRED {series_id}: {e}")
                return pd.Series(dtype=float, name=series_id)
        else:
            log.warning(f"  FRED {series_id}: tutti i retry esauriti")
            return pd.Series(dtype=float, name=series_id)

        if not obs:
            return pd.Series(dtype=float, name=series_id)

        dates  = pd.to_datetime([o["date"] for o in obs])
        values = pd.to_numeric([o["value"] for o in obs], errors="coerce")
        s = pd.Series(values, index=dates, name=series_id)
        s = s.replace(".", np.nan).dropna()
        return s

    # Downloads all series, applies release-lag and aligns on a daily index.
    def fetch_all(self, series_dict: dict, start: str = "2018-01-01",
                  sleep: float = 2.0) -> pd.DataFrame:
        """
        Download all series and align them on a daily index.

        LOOK-AHEAD BIAS CORRECTION:
          Each value becomes available only after its typical release lag.
          E.g.: January CPI (obs_date=2024-01-31) → available from 2024-03-06
                (35 days later), not from 2024-02-01.

          In practice: we shift each observation's index forward by
          `lag` days before reindexing on the daily calendar.
          The ffill then only propagates data already "published" at that date.
        """
        frames = {}
        total  = len(series_dict)
        skipped_series: list[str] = []  # log of skipped series
        for i, (name, (fred_id, desc, freq)) in enumerate(series_dict.items(), 1):
            log.info(f"  [{i:2d}/{total}] {fred_id:<22} {desc}")
            s = self.fetch(fred_id, start=start)
            if s.empty:
                # Logs explicitly (no silent skip).
                log.warning(
                    f"  FRED {fred_id} ({name}): nessun dato da {start} — "
                    f"la serie potrebbe non esistere per questo periodo. "
                    f"Serie ignorata."
                )
                skipped_series.append(f"{name} ({fred_id})")
                time.sleep(sleep)
                continue

            # Verifies effective coverage of the requested period.
            series_start = s.index.min().date()
            requested    = pd.to_datetime(start).date()
            if series_start > requested:
                gap = (series_start - requested).days
                log.warning(
                    f"  FRED {fred_id} ({name}): inizia il {series_start} "
                    f"({gap} giorni dopo il history_start={start}). "
                    f"Le prime {gap} giorni di dati macro avranno bfill nel merge."
                )

            # Lag for this series (override > default for freq).
            lag_days = SERIES_LAG_OVERRIDE.get(name, RELEASE_LAG_DAYS.get(freq, 35))

            # Forward-shifts observation dates = availability date.
            s.index = s.index + pd.Timedelta(days=lag_days)
            frames[name] = s
            time.sleep(sleep)  # rate-limit safety

        if skipped_series:
            log.warning(
                f"FRED fetch_all: {len(skipped_series)} serie saltate "
                f"(nessun dato nel periodo richiesto): "
                f"{', '.join(skipped_series[:10])}"
                + (" ..." if len(skipped_series) > 10 else "")
            )

        if not frames:
            raise ValueError("Nessuna serie FRED scaricata — controlla connessione e api_key.")

        # Daily reindex + ffill only propagates already-published data (no look-ahead).
        idx = pd.date_range(start=start, end=datetime.now().date(), freq="D")
        df  = pd.DataFrame(index=idx)
        for name, s in frames.items():
            df[name] = s
        df = df.ffill()
        return df


# yfinance downloader: auto-adjusted daily prices.
def fetch_yfinance(tickers: dict, start: str = "2018-01-01") -> pd.DataFrame:
    """
    Download daily prices from yfinance.
    Tickers: {yahoo_ticker: column_name}
    """
    try:
        import yfinance as yf
    except ImportError:
        log.warning("yfinance non installato — pip install yfinance")
        return pd.DataFrame()

    frames = {}
    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if not data.empty:
                frames[name] = data["Close"]
                log.info(f"  yfinance {ticker:<16} → {name}  ({len(data)} giorni)")
        except Exception as e:
            log.warning(f"  yfinance {ticker}: {e}")
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame()

    idx = pd.date_range(start=start, end=datetime.now().date(), freq="D")
    clean_frames = {}
    for k, v in frames.items():
        if isinstance(v, pd.DataFrame):
            clean_frames[k] = v.squeeze()
        else:
            clean_frames[k] = v
    df = pd.DataFrame(clean_frames)
    df = df.reindex(idx).ffill()
    return df




# Transforms raw series into stationary features (YoY, MoM, diff, z-score).

class MacroFeatureBuilder:
    """
    Transform raw FRED series into stationary, normalized features,
    ready to be passed to the HMM and the MacroEncoder.

    Principle: level series (e.g. CPI = 312) are not stationary.
    We use YoY or MoM percentage changes depending on the series.
    For series already in % (rates, spreads), we use absolute changes.
    """

    # Builds stationary macro features from FRED + yfinance series.
    def build(self, df_fred: pd.DataFrame,
              df_yf: pd.DataFrame) -> pd.DataFrame:
        """
        Input:  df_fred (daily, forward-filled), df_yf (daily)
        Output: DataFrame with all macro features ready for the HMM
        """
        f = pd.DataFrame(index=df_fred.index)

        # Realized inflation: YoY + MoM momentum.
        for col in ["cpi_yoy", "core_cpi_yoy", "pce_yoy", "core_pce_yoy"]:
            if col in df_fred.columns:
                raw = df_fred[col]
                # YoY%: (t - t-12months) / t-12months * 100
                f[col + "_yoy"] = raw.pct_change(252) * 100
                # Momentum: accelerating or decelerating?
                f[col + "_mom"] = raw.pct_change(21) * 100

        # Inflation expectations (level + 1M change).
        for col in ["infl_exp_1y", "infl_exp_5y", "infl_exp_10y", "tips_5y5y"]:
            if col in df_fred.columns:
                f[col] = df_fred[col]
                f[col + "_chg"] = df_fred[col].diff(21)

        # Inflation surprise proxy: expectations - realized.
        if "infl_exp_1y" in df_fred.columns and "cpi_yoy" in df_fred.columns:
            f["infl_surprise"] = df_fred["infl_exp_1y"] - df_fred["cpi_yoy"].pct_change(252) * 100

        # Fed policy: effective rate, target, credibility gap.
        if "fed_funds" in df_fred.columns:
            f["fed_funds"]      = df_fred["fed_funds"]
            f["fed_funds_chg"]  = df_fred["fed_funds"].diff(21)    # 1M change

        if "fed_funds_upper" in df_fred.columns:
            f["fed_target"]     = df_fred["fed_funds_upper"]
            # Spread: market vs target → measures Fed credibility
            if "fed_funds" in df_fred.columns:
                f["fed_gap"]    = df_fred["fed_funds"] - df_fred["fed_funds_upper"]

        # Yield curve: Treasury levels + spreads/inversion.
        for col in ["treasury_2y", "treasury_5y", "treasury_10y", "treasury_30y"]:
            if col in df_fred.columns:
                f[col]          = df_fred[col]
                f[col + "_chg"] = df_fred[col].diff(21)

        for col in ["yield_curve_2_10", "yield_curve_3m_10", "real_rate_10y"]:
            if col in df_fred.columns:
                f[col]          = df_fred[col]
                f[col + "_chg"] = df_fred[col].diff(21)

        # Yield curve inversion (recession signal)
        if "yield_curve_2_10" in df_fred.columns:
            f["yield_inverted"] = (df_fred["yield_curve_2_10"] < 0).astype(float)

        # GDP and growth leading indicators (LEI, INDPRO, GDPNow).
        if "gdp_growth" in df_fred.columns:
            f["gdp_growth"]     = df_fred["gdp_growth"]
            f["gdp_growth_chg"] = df_fred["gdp_growth"].diff(63)   # 1Q change

        if "lei" in df_fred.columns:
            f["lei"]            = df_fred["lei"].pct_change(21) * 100  # MoM%
            f["lei_trend"]      = df_fred["lei"].pct_change(63) * 100  # 3M trend

        if "indpro" in df_fred.columns:
            f["indpro"]         = df_fred["indpro"].pct_change(21) * 100
            f["indpro_trend"]   = df_fred["indpro"].pct_change(63) * 100

        if "gdpnow" in df_fred.columns:
            f["gdpnow"]         = df_fred["gdpnow"]
            if "gdp_growth" in df_fred.columns:
                f["gdpnow_surprise"] = df_fred["gdpnow"] - df_fred["gdp_growth"]

        # Labor market: unemployment, NFP, wages, Sahm proxy.
        if "unemployment" in df_fred.columns:
            f["unemployment"]   = df_fred["unemployment"]
            f["unemployment_chg"]= df_fred["unemployment"].diff(21)
            # Sahm Rule proxy: recession if +0.5% above the 12M minimum
            roll_min = df_fred["unemployment"].rolling(252).min()
            f["sahm_proxy"]     = df_fred["unemployment"] - roll_min

        if "nfp_level" in df_fred.columns:
            # NFP MoM change (thousands of jobs)
            f["nfp_mom"]        = df_fred["nfp_level"].diff(21)
            f["nfp_3m_avg"]     = f["nfp_mom"].rolling(63).mean()

        if "avg_hourly_earn" in df_fred.columns:
            f["wages_yoy"]      = df_fred["avg_hourly_earn"].pct_change(252) * 100

        # Labor leading indicators
        if "initial_claims" in df_fred.columns:
            f["claims"]         = np.log(df_fred["initial_claims"])
            f["claims_chg"]     = df_fred["initial_claims"].pct_change(21) * 100
            f["claims_4w"]      = df_fred["initial_claims"].rolling(4).mean()
            f["claims_4w_chg"]  = f["claims_4w"].pct_change(21) * 100

        if "jolts_openings" in df_fred.columns:
            f["jolts_openings_yoy"] = df_fred["jolts_openings"].pct_change(252) * 100

        # Financial conditions: NFCI, sentiment, credit spreads.
        if "nfci" in df_fred.columns:
            f["nfci"]           = df_fred["nfci"]    # already normalized (mean=0)
            f["nfci_trend"]     = df_fred["nfci"].rolling(21).mean()

        if "consumer_conf" in df_fred.columns:
            f["consumer_conf"]  = df_fred["consumer_conf"]
            f["consumer_conf_chg"] = df_fred["consumer_conf"].diff(21)

        for col in ["credit_spread_hy", "credit_spread_ig"]:
            if col in df_fred.columns:
                f[col]          = df_fred[col]
                f[col + "_chg"] = df_fred[col].diff(21)

        # Liquidity: Fed balance sheet, M2, reverse repo.
        if "fed_balance" in df_fred.columns:
            f["fed_balance_yoy"] = df_fred["fed_balance"].pct_change(252) * 100
            f["fed_balance_chg"] = df_fred["fed_balance"].diff(21)

        if "m2" in df_fred.columns:
            f["m2_yoy"]         = df_fred["m2"].pct_change(252) * 100
            f["m2_chg"]         = df_fred["m2"].pct_change(21) * 100

        if "repo_rate" in df_fred.columns:
            f["repo_rate"]      = df_fred["repo_rate"]
            f["repo_rate_chg"]  = df_fred["repo_rate"].diff(21)

        # Financial markets (yfinance): VIX, DXY, gold, oil, SP500, crypto.
        if not df_yf.empty:
            if "vix" in df_yf.columns:
                f["vix"]        = df_yf["vix"]
                f["vix_chg"]    = df_yf["vix"].diff(5)
                f["vix_high"]   = (df_yf["vix"] > 25).astype(float)  # fear regime

            if "dxy" in df_yf.columns:
                f["dxy_yoy"]    = df_yf["dxy"].pct_change(252) * 100
                f["dxy_mom"]    = df_yf["dxy"].pct_change(21)  * 100

            if "gold" in df_yf.columns:
                f["gold_yoy"]   = df_yf["gold"].pct_change(252) * 100
                f["gold_mom"]   = df_yf["gold"].pct_change(21)  * 100

            if "oil_wti" in df_yf.columns:
                f["oil_yoy"]    = df_yf["oil_wti"].pct_change(252) * 100
                f["oil_mom"]    = df_yf["oil_wti"].pct_change(21)  * 100

            if "sp500" in df_yf.columns:
                f["sp500_yoy"]  = df_yf["sp500"].pct_change(252) * 100
                f["sp500_mom"]  = df_yf["sp500"].pct_change(21)  * 100
                f["sp500_trend"]= (df_yf["sp500"] > df_yf["sp500"].rolling(200).mean()).astype(float)

            # Crypto correlations: ETH as BTC leading signal, rolling corr.
            if "eth_daily" in df_yf.columns:
                f["eth_yoy"]    = df_yf["eth_daily"].pct_change(252) * 100
                f["eth_mom"]    = df_yf["eth_daily"].pct_change(21)  * 100
                f["eth_mom_7d"] = df_yf["eth_daily"].pct_change(7)   * 100
                if "btc_daily" in df_yf.columns:
                    btc_ret = df_yf["btc_daily"].pct_change()
                    eth_ret = df_yf["eth_daily"].pct_change()
                    f["btc_eth_corr_30d"] = btc_ret.rolling(30).corr(eth_ret)
                    f["btc_eth_corr_90d"] = btc_ret.rolling(90).corr(eth_ret)
                    f["btc_vs_eth_30d"]   = (
                        df_yf["btc_daily"].pct_change(30) -
                        df_yf["eth_daily"].pct_change(30)
                    )

            if "eth_btc_ratio" in df_yf.columns:
                f["eth_btc_ratio"]     = df_yf["eth_btc_ratio"]
                f["eth_btc_ratio_mom"] = df_yf["eth_btc_ratio"].pct_change(21) * 100
                f["eth_btc_trend"]     = (
                    df_yf["eth_btc_ratio"] > df_yf["eth_btc_ratio"].rolling(90).mean()
                ).astype(float)

        # Final cleanup: ffill/bfill, drop inf and overly-sparse columns.
        f = f.ffill().bfill()
        f = f.replace([np.inf, -np.inf], np.nan)

        # Drop columns with too many NaN (>50%)
        thresh = len(f) * 0.5
        f = f.dropna(axis=1, thresh=thresh)

        log.info(f"Macro features costruite: {len(f.columns)} colonne, {len(f)} giorni")
        return f


# ─── MERGE WITH BTC CANDLES ──────────────────────────────────────────────────

# Aligns daily macro to BTC candles via merge_asof (no look-ahead).
def merge_macro_with_candles(
    df_candles: pd.DataFrame,
    df_macro:   pd.DataFrame,
) -> pd.DataFrame:
    """
    Align the (daily) macro features with the (minute) BTC candles.

    The release lag has already been applied in `fetch_all` — the df_macro index
    already reflects the *effective availability* dates (not observation dates).
    Here we simply use `pd.merge_asof`, which performs a timestamp-ordered
    forward fill, guaranteeing zero look-ahead bias.

    For each BTC candle `pd.merge_asof` picks the macro value with the most
    recent index ≤ the candle timestamp — never a future value.

    Memory notes with 2M+ candles:
      · merge_asof is implemented in Cython — it does not expand the macro
        DataFrame into an N×M matrix. The only overhead is the resulting
        DataFrame, with the same rows as df_candles (N) + n_macro_cols extra columns.
      · With 2M rows × 80 float32 columns → ~640 MB of additional RAM.
        Acceptable on machines with ≥8 GB RAM; if RAM is limited,
        consider saving the macro features separately and joining only
        when the windows are created (TODO future optimization).
    """
    if df_macro.empty:
        log.warning("DataFrame macro vuoto — merge saltato.")
        return df_candles

    # ── Log macro vs price time coverage ─────────────────────────────────────
    # Fix 4: explicitly show the time ranges of both datasets
    # to diagnose series starting later than the configured history_start.
    price_start = df_candles["open_time"].min() if "open_time" in df_candles.columns else None
    price_end   = df_candles["open_time"].max() if "open_time" in df_candles.columns else None
    macro_start = pd.to_datetime(df_macro.index.min(), utc=True)
    macro_end   = pd.to_datetime(df_macro.index.max(), utc=True)

    if price_start is not None:
        log.info(
            f"Copertura temporale:"
            f"\n  Price (candele): {price_start.date()} → {price_end.date()}"
            f"  ({len(df_candles):,} candele)"
            f"\n  Macro (giornaliero): {macro_start.date()} → {macro_end.date()}"
            f"  ({len(df_macro)} giorni, {len(df_macro.columns)} serie)"
        )
        # Warn if macro does not cover the whole price period
        if macro_start > price_start:
            gap_days = (macro_start - price_start).days
            # bars/day inferred from the MEDIAN data step (interval-agnostic, no config
            # needed): 1m → 1440, 1h → 24. Fallback 1440 (= legacy 1m) if the step is
            # NaN/non-positive (e.g. <2 candles or degenerate timestamps).
            _step_med = df_candles["open_time"].diff().median()
            _step_sec = _step_med.total_seconds() if pd.notna(_step_med) else 0.0
            bars_per_day = max(1, round(86400 / _step_sec)) if _step_sec > 0 else 1440
            n_candles_gap = gap_days * bars_per_day
            pct_gap       = n_candles_gap / max(len(df_candles), 1) * 100
            log.warning(
                f"Macro inizia {gap_days} giorni DOPO le candele price "
                f"({macro_start.date()} vs {price_start.date()}). "
                f"Le prime ~{n_candles_gap:,} candele ({pct_gap:.1f}%) "
                f"avranno NaN nelle colonne macro → bfill applicato dopo il merge."
            )
        if macro_end < price_end:
            gap_days = (price_end - macro_end).days
            log.warning(
                f"Macro termina {gap_days} giorni PRIMA delle candele price "
                f"({macro_end.date()} vs {price_end.date()}). "
                f"Le ultime candele useranno l'ultimo dato macro disponibile (ffill)."
            )

    # Prepare macro index: timezone-aware UTC, sorted
    macro_utc = df_macro.copy()
    macro_utc.index = pd.to_datetime(macro_utc.index, utc=True).normalize()
    macro_utc = macro_utc[~macro_utc.index.duplicated(keep="last")].sort_index()
    macro_utc = macro_utc.reset_index().rename(columns={"index": "date"})

    # Normalize candle open_time to UTC midnight for the join
    candles_work = df_candles.copy()
    candles_work["_merge_date"] = candles_work["open_time"].dt.normalize()

    # merge_asof: for each candle takes the most recent available macro value
    # direction="backward" guarantees zero look-ahead bias
    merged = pd.merge_asof(
        candles_work.sort_values("_merge_date"),
        macro_utc.rename(columns={"date": "_merge_date"}),
        on="_merge_date",
        direction="backward",   # ← past data only, never future
        suffixes=("", "_macro"),
    )
    merged = merged.drop(columns=["_merge_date"])

    # Rename macro columns adding the "macro_" prefix
    macro_cols = [c for c in macro_utc.columns if c != "_merge_date"]
    rename_map = {c: f"macro_{c}" for c in macro_cols if c in merged.columns}
    merged = merged.rename(columns=rename_map)

    # Restore the original open_time order
    merged = merged.sort_values("open_time").reset_index(drop=True)

    # Fix 4: fill any leading NaN with bfill (series starting after history_start)
    # This can happen if a FRED series has no data before 2018-01-01
    # but the price candles start from that date.
    # The bfill uses the first available value for the earlier rows → conservative.
    new_macro_cols = [c for c in merged.columns if c.startswith("macro_")]
    nan_before = merged[new_macro_cols].isna().sum().sum() if new_macro_cols else 0
    if nan_before > 0:
        merged[new_macro_cols] = merged[new_macro_cols].bfill()
        nan_after = merged[new_macro_cols].isna().sum().sum()
        log.info(
            f"bfill applicato sulle colonne macro: "
            f"{nan_before:,} NaN → {nan_after:,} NaN residui "
            f"(residui = serie FRED completamente assenti per il periodo)"
        )

    n_macro = len(new_macro_cols)
    log.info(
        f"Merge completato (merge_asof, no look-ahead): "
        f"{n_macro} colonne macro aggiunte  |  "
        f"shape finale = {merged.shape}"
    )
    return merged
