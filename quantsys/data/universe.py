"""Cross-sectional perp universe selection — liquidity-ranked Binance USDT-perps.

IT: Seleziona i top-N simboli perpetui USDT di Binance per liquidità (quote
    volume trailing-24h fornito da `ticker/24hr` sul fapi), come universo per
    una probe di Information Coefficient cross-sezionale: testare se la μ
    predetta dal modello BTC ha skill di rango trasversale tra asset.
EN: Selects the top-N Binance USDT-perpetual symbols by liquidity (trailing-24h
    quote volume from the fapi `ticker/24hr` endpoint), as the universe for a
    cross-sectional Information Coefficient probe: testing whether the BTC
    model's predicted μ has cross-sectional rank skill across assets.

⚠️ SURVIVORSHIP / FORWARD-LOOKING BIAS (caveat esplicito · explicit caveat)
─────────────────────────────────────────────────────────────────────────────
IT: La selezione interroga lo stato CORRENTE dell'exchange (`exchangeInfo` →
    status=TRADING, `ticker/24hr` → volume di OGGI). Quindi:
      • include solo i simboli ANCORA quotati oggi → survivorship bias
        (delisting come es. asset falliti/illiquidi sono esclusi a posteriori);
      • il ranking di liquidità usa il volume corrente, non quello del periodo
        storico campionato → look-ahead nella COMPOSIZIONE dell'universo.
    Per una PRIMA probe di rango cross-sezionale questo è accettabile (stiamo
    misurando IC, non simulando un PnL tradabile), MA il risultato NON è una
    backtest valida di una strategia cross-asset. Da rifare con universo
    point-in-time (snapshot storici di listing+volume) prima di qualunque claim
    di tradabilità.
EN: Selection queries the CURRENT exchange state (`exchangeInfo` →
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

# IT: Endpoint perpetui USDT (fapi) — stesso host usato da fetch_funding_rate.
# EN: USDT-perpetual endpoint (fapi) — same host used by fetch_funding_rate.
BINANCE_FAPI_REST = "https://fapi.binance.com/fapi/v1"

# IT: Ancore sempre incluse (se quotate): BTC è la sanity anchor (il modello è
#     addestrato su di lui), gli altri sono le major liquide storicamente stabili.
#     Ordinate per priorità di inclusione.
# EN: Always-included anchors (if listed): BTC is the sanity anchor (the model is
#     trained on it), the others are the historically-stable liquid majors.
#     Ordered by inclusion priority.
ANCHOR_SYMBOLS: List[str] = [
    "BTCUSDT",   # sanity anchor — modello addestrato qui | model trained here
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
    """Universo cross-sezionale di perp USDT Binance, ranked per liquidità.

    IT: Costruisce la lista dei simboli per la probe IC cross-sezionale:
        top-N per quote volume 24h, con le ancore SEMPRE incluse (BTCUSDT
        garantito come prima posizione). Fail-safe: se la rete non è
        raggiungibile, ricade sulle sole ancore.
    EN: Builds the symbol list for the cross-sectional IC probe: top-N by 24h
        quote volume, with anchors ALWAYS included (BTCUSDT guaranteed first).
        Fail-safe: if the network is unreachable, falls back to anchors only.

    Args:
        n: dimensione target dell'universo (default 20) | target universe size.
        anchors: simboli da forzare nell'universo | symbols to force-include.
        request_timeout: timeout HTTP secondi | HTTP timeout seconds.
    """

    def __init__(
        self,
        n: int = 20,
        anchors: Optional[List[str]] = None,
        request_timeout: int = 20,
    ) -> None:
        self.n = int(n)
        # IT: copia difensiva per non mutare la costante di modulo.
        # EN: defensive copy so we never mutate the module constant.
        self.anchors = list(anchors) if anchors is not None else list(ANCHOR_SYMBOLS)
        self.request_timeout = request_timeout
        # IT: cache della lista risolta (lazy) per evitare doppie query di rete.
        # EN: cache of the resolved list (lazy) to avoid duplicate network calls.
        self._symbols: Optional[List[str]] = None
        # IT: diagnostica per-simbolo (volume) popolata da _resolve.
        # EN: per-symbol diagnostics (volume) populated by _resolve.
        self.volume_usd: dict = {}

    # IT: Interroga i perp USDT attualmente in TRADING (insieme di simboli validi).
    # EN: Query the currently-TRADING USDT perps (set of valid symbols).
    def _fetch_trading_perps(self) -> set:
        import requests

        resp = requests.get(
            f"{BINANCE_FAPI_REST}/exchangeInfo", timeout=self.request_timeout
        )
        resp.raise_for_status()
        info = resp.json()
        # IT: filtro: perpetuo, quotato in USDT, stato TRADING (esclude HALT/BREAK).
        # EN: filter: perpetual, USDT-quoted, TRADING status (excludes HALT/BREAK).
        trading = {
            s["symbol"]
            for s in info.get("symbols", [])
            if s.get("contractType") == "PERPETUAL"
            and s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
        }
        log.info(f"exchangeInfo: {len(trading)} perp USDT in TRADING.")
        return trading

    # IT: Ranking per liquidità: quoteVolume 24h dal ticker, ristretto ai perp validi.
    # EN: Liquidity ranking: 24h quoteVolume from ticker, restricted to valid perps.
    def _fetch_volume_ranked(self, valid: set) -> List[str]:
        import requests

        resp = requests.get(
            f"{BINANCE_FAPI_REST}/ticker/24hr", timeout=self.request_timeout
        )
        resp.raise_for_status()
        tickers = resp.json()
        # IT: tieni solo i simboli validi e leggi il quote volume (USD-notional 24h).
        # EN: keep only valid symbols and read quote volume (USD-notional 24h).
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
        # IT: ordine decrescente di liquidità.
        # EN: descending liquidity order.
        rows.sort(key=lambda r: r[1], reverse=True)
        return [sym for sym, _ in rows]

    # IT: Risolve l'universo: ancore (filtrate per quotazione) + top-N per liquidità.
    # EN: Resolves the universe: anchors (filtered by listing) + top-N by liquidity.
    def _resolve(self) -> List[str]:
        try:
            valid = self._fetch_trading_perps()
            ranked = self._fetch_volume_ranked(valid)
        except Exception as e:
            # IT: fail-safe — senza rete restano solo le ancore (BTC garantito).
            # EN: fail-safe — without network only anchors remain (BTC guaranteed).
            log.warning(
                f"Universe: query fapi fallita ({e}) — fallback alle sole ancore."
            )
            return [s for s in self.anchors][: self.n]

        # IT: parti dalle ancore ANCORA quotate, preservandone l'ordine di priorità.
        # EN: start from anchors STILL listed, preserving their priority order.
        selected: List[str] = []
        for sym in self.anchors:
            if sym in valid and sym not in selected:
                selected.append(sym)
            elif sym not in valid:
                log.warning(f"Anchor {sym} non è un perp USDT in TRADING — saltato.")

        # IT: riempi fino a N con i simboli più liquidi non ancora presenti.
        # EN: fill up to N with the most-liquid symbols not yet present.
        for sym in ranked:
            if len(selected) >= self.n:
                break
            if sym not in selected:
                selected.append(sym)

        # IT: garanzia hard: BTCUSDT prima posizione (sanity anchor del modello).
        # EN: hard guarantee: BTCUSDT first (the model's sanity anchor).
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

    # IT: API pubblica — lista simboli (cache lazy: una sola query di rete).
    # EN: Public API — symbol list (lazy cache: a single network query).
    def symbols(self) -> List[str]:
        """Ritorna la lista dei simboli dell'universo (BTCUSDT in testa).

        EN: Returns the universe symbol list (BTCUSDT first).
        """
        if self._symbols is None:
            self._symbols = self._resolve()
        return list(self._symbols)

    # IT: Diagnostica — volume 24h per simbolo risolto (USD-notional).
    # EN: Diagnostics — 24h volume per resolved symbol (USD-notional).
    def liquidity(self) -> dict:
        """Mappa {symbol: quote_volume_24h_usd} per i simboli risolti.

        EN: Map {symbol: quote_volume_24h_usd} for the resolved symbols.
        """
        syms = self.symbols()
        return {s: self.volume_usd.get(s, float("nan")) for s in syms}
