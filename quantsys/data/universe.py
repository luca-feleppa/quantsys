"""Cross-sectional perp universe selection — liquidity-ranked Binance USDT-perps.

Selects the top-N Binance USDT-perpetual symbols by liquidity (trailing-24h
quote volume from the fapi `ticker/24hr` endpoint), as the universe for a
cross-sectional Information Coefficient probe: testing whether the BTC
model's predicted μ has cross-sectional rank skill across assets.

⚠️ SURVIVORSHIP / FORWARD-LOOKING BIAS (explicit caveat)
─────────────────────────────────────────────────────────────────────────────
Selection queries the CURRENT exchange state (`exchangeInfo` →
status=TRADING, `ticker/24hr` → TODAY's volume). Therefore:
  • only symbols STILL listed today are included → survivorship bias
    (delisted/failed/illiquid assets are excluded a posteriori);
  • the liquidity ranking uses current volume, not the sampled historical
    period's volume → look-ahead in the universe COMPOSITION.
For a FIRST cross-sectional rank probe this is acceptable (we are measuring
IC, not simulating tradable PnL), BUT the result is NOT a valid backtest of
a cross-asset strategy. Must be redone with a point-in-time universe
(historical listing+volume snapshots) before any tradability claim.
"""
import logging
from typing import List, Optional

log = logging.getLogger("quantsys.data.universe")

# USDT-perpetual endpoint (fapi) — same host used by fetch_funding_rate.
BINANCE_FAPI_REST = "https://fapi.binance.com/fapi/v1"

# Always-included anchors (if listed): BTC is the sanity anchor (the model is
# trained on it), the others are the historically-stable liquid majors.
# Ordered by inclusion priority.
ANCHOR_SYMBOLS: List[str] = [
    "BTCUSDT",   # sanity anchor — model trained here
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "LTCUSDT",
]


class PerpUniverse:
    """Cross-sectional universe of Binance USDT perps, ranked by liquidity.

    Builds the symbol list for the cross-sectional IC probe: top-N by 24h
    quote volume, with anchors ALWAYS included (BTCUSDT guaranteed first).
    Fail-safe: if the network is unreachable, falls back to anchors only.

    Args:
        n: target universe size (default 20).
        anchors: symbols to force-include.
        request_timeout: HTTP timeout seconds.
    """

    def __init__(
        self,
        n: int = 20,
        anchors: Optional[List[str]] = None,
        request_timeout: int = 20,
    ) -> None:
        self.n = int(n)
        # defensive copy so we never mutate the module constant.
        self.anchors = list(anchors) if anchors is not None else list(ANCHOR_SYMBOLS)
        self.request_timeout = request_timeout
        # cache of the resolved list (lazy) to avoid duplicate network calls.
        self._symbols: Optional[List[str]] = None
        # per-symbol diagnostics (volume) populated by _resolve.
        self.volume_usd: dict = {}

    # Query the currently-TRADING USDT perps (set of valid symbols).
    def _fetch_trading_perps(self) -> set:
        import requests

        resp = requests.get(
            f"{BINANCE_FAPI_REST}/exchangeInfo", timeout=self.request_timeout
        )
        resp.raise_for_status()
        info = resp.json()
        # filter: perpetual, USDT-quoted, TRADING status (excludes HALT/BREAK).
        trading = {
            s["symbol"]
            for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        }
        log.info(f"exchangeInfo: {len(trading)} perp USDT in TRADING.")
        return trading

    # Liquidity ranking: 24h quoteVolume from ticker, restricted to valid perps.
    def _fetch_volume_ranked(self, valid: set) -> List[str]:
        import requests

        resp = requests.get(
            f"{BINANCE_FAPI_REST}/ticker/24hr", timeout=self.request_timeout
        )
        resp.raise_for_status()
        tickers = resp.json()
        # keep only valid symbols and read quote volume (USD-notional 24h).
        rows = []
        for t in tickers:
            sym = t.get("symbol")
            if sym in valid:
                try:
                    qv = float(t.get("quoteVolume", 0.0))
                except (TypeError, ValueError):
                    qv = 0.0
                rows.append((sym, qv))
                self.volume_usd[sym] = qv
        # descending liquidity order.
        rows.sort(key=lambda r: r[1], reverse=True)
        return [sym for sym, _ in rows]

    # Resolves the universe: anchors (filtered by listing) + top-N by liquidity.
    def _resolve(self) -> List[str]:
        try:
            valid = self._fetch_trading_perps()
            ranked = self._fetch_volume_ranked(valid)
        except Exception as e:
            # fail-safe — without network only anchors remain (BTC guaranteed).
            log.warning(
                f"Universe: query fapi fallita ({e}) — fallback alle sole ancore."
            )
            return [s for s in self.anchors][: self.n]

        # start from anchors STILL listed, preserving their priority order.
        selected: List[str] = []
        for sym in self.anchors:
            if sym in valid and sym not in selected:
                selected.append(sym)
            elif sym not in valid:
                log.warning(f"Anchor {sym} non è un perp USDT in TRADING — saltato.")

        # fill up to N with the most-liquid symbols not yet present.
        for sym in ranked:
            if len(selected) >= self.n:
                break
            if sym not in selected:
                selected.append(sym)

        # hard guarantee: BTCUSDT first (the model's sanity anchor).
        if "BTCUSDT" in selected:
            selected.remove("BTCUSDT")
            selected.insert(0, "BTCUSDT")
        else:
            selected.insert(0, "BTCUSDT")
            selected = selected[: self.n]

        log.info(
            f"Universe risolto: {len(selected)} simboli "
            f"(target N={self.n}, ancore={len(self.anchors)})."
        )
        return selected

    # Public API — symbol list (lazy cache: a single network query).
    def symbols(self) -> List[str]:
        """Returns the universe symbol list (BTCUSDT first)."""
        if self._symbols is None:
            self._symbols = self._resolve()
        return list(self._symbols)

    # Diagnostics — 24h volume per resolved symbol (USD-notional).
    def liquidity(self) -> dict:
        """Map {symbol: quote_volume_24h_usd} for the resolved symbols."""
        syms = self.symbols()
        return {s: self.volume_usd.get(s, float("nan")) for s in syms}
