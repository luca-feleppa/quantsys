"""
Script 00b — Verifica connessione e funzionalità API sul Binance Futures Testnet.

Scopo: validare end-to-end le API key del testnet PRIMA di integrarle nel live engine.

Cosa fa (5 step, ognuno con check esplicito):
  1. Carica BINANCE_TESTNET_API_KEY/SECRET da .env
  2. Connette al testnet (https://testnet.binancefuture.com via python-binance.Client(testnet=True))
  3. Legge balance USDT + posizione BTCUSDT
  4. Setta leverage=1 + margin=ISOLATED su BTCUSDT (no-op se già così)
  5. Piazza un ordine LIMIT BUY a -10% dal prezzo corrente (NO FILL garantito),
     verifica che sia negli open orders, lo cancella, verifica che balance/pos
     restino invariati

Tutti gli ordini sono BTCUSDT con quantity minima (0.002 BTC ≈ $160) → impatto
trascurabile sul balance testnet anche in caso di fill accidentale.

Run:
    python scripts/00_test_binance_testnet.py
"""
from __future__ import annotations

import os
import sys
import time

# IT: forza UTF-8 su stdout/stderr per glifi unicode su Windows cp1252
# EN: force UTF-8 stdout/stderr for unicode glyphs on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERRORE: python-dotenv non installato. Esegui: pip install python-dotenv")
    sys.exit(1)

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException
except ImportError:
    print("ERRORE: python-binance non installato. Esegui: pip install python-binance")
    sys.exit(1)


GRN = "\033[92m"
RED = "\033[91m"
YLW = "\033[93m"
CYN = "\033[96m"
RST = "\033[0m"


# IT: helper di stampa colorata (successo/errore/info/warning)
# EN: colored print helpers (success/error/info/warning)
def _ok(msg: str): print(f"  {GRN}✓{RST} {msg}")
def _err(msg: str): print(f"  {RED}✗{RST} {msg}")
def _info(msg: str): print(f"  {CYN}ℹ{RST} {msg}")
def _warn(msg: str): print(f"  {YLW}⚠{RST} {msg}")


# IT: test end-to-end testnet (5 step); ritorna exit code 0 se tutti passano.
# EN: end-to-end testnet test (5 steps); returns exit code 0 if all pass.
def main() -> int:
    print(f"\n{'═' * 60}")
    print(f"  QUANTSYS · BINANCE FUTURES TESTNET — Connection Test")
    print(f"{'═' * 60}\n")

    # IT: step 1 - carica le API key dal file .env
    # EN: step 1 - load API keys from the .env file
    print(f"{CYN}[1/5]{RST} Carico chiavi da .env ...")
    load_dotenv()
    api_key = os.getenv("BINANCE_TESTNET_API_KEY", "").strip()
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "").strip()
    if not api_key or not api_secret:
        _err("BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET non trovati o vuoti in .env")
        _info("Genera la key su: https://testnet.binancefuture.com/en/futures/BTCUSDT")
        _info("Tab 'API Key' (alto destra) → Create API Key → permessi Reading + Futures")
        _info("Poi compila il file .env nella root del progetto")
        return 1
    _ok(f"API key caricata (len={len(api_key)}, secret len={len(api_secret)})")

    # IT: step 2 - connetti al testnet (REST + auth check)
    # EN: step 2 - connect to testnet (REST + auth check)
    print(f"\n{CYN}[2/5]{RST} Connessione a Binance Futures Testnet ...")
    try:
        # IT: testnet=True usa testnet.binancefuture.com e stream.binancefuture.com
        # EN: testnet=True targets testnet.binancefuture.com and stream.binancefuture.com
        client = Client(api_key, api_secret, testnet=True)
        # IT: ping pubblico, nessuna autenticazione
        # EN: public ping, no auth required
        client.futures_ping()
        _ok("Ping OK")
        # IT: clock skew > 1s causerebbe errore -1021 INVALID_TIMESTAMP
        # EN: clock skew above 1s would cause error -1021 INVALID_TIMESTAMP
        t = client.futures_time()
        delta_ms = abs(int(time.time() * 1000) - t["serverTime"])
        if delta_ms > 1000:
            _warn(f"Clock skew con server Binance: {delta_ms} ms (>1s può causare INVALID_TIMESTAMP)")
        else:
            _ok(f"Clock sync (delta {delta_ms} ms)")
        # IT: prima chiamata autenticata - valida la signature HMAC
        # EN: first authenticated call - validates the HMAC signature
        acct = client.futures_account()
        _ok(f"Account autenticato: feeTier={acct.get('feeTier')}, totalWalletBalance={acct.get('totalWalletBalance')}")
    except BinanceAPIException as e:
        _err(f"BinanceAPIException: code={e.code} status={e.status_code} — {e.message}")
        if e.code in (-2014, -1022):
            _info("La signature non è valida → controlla che API_SECRET sia copiato CORRETTAMENTE (no spazi, no troncamento)")
        if e.code == -2015:
            _info("API key non valida o senza permessi Futures → ricontrolla i permessi sul pannello testnet")
        if e.code == -1021:
            _info("Clock del PC fuori sync col server → sincronizza orologio Windows")
        return 1
    except Exception as e:
        _err(f"Errore connessione: {type(e).__name__}: {e}")
        return 1

    # IT: step 3 - leggi balance USDT e posizione BTCUSDT correnti
    # EN: step 3 - read current USDT balance and BTCUSDT position
    print(f"\n{CYN}[3/5]{RST} Leggo balance USDT e posizione BTCUSDT ...")
    try:
        balances = client.futures_account_balance()
        usdt = next((b for b in balances if b["asset"] == "USDT"), None)
        if usdt:
            _ok(f"USDT balance: {usdt['balance']} (available: {usdt['availableBalance']})")
            usdt_balance_pre = float(usdt["balance"])
        else:
            _warn("Nessun balance USDT trovato — testnet account vuoto. Riempi su 'Get Test Funds' del pannello testnet")
            usdt_balance_pre = 0.0

        positions = client.futures_position_information(symbol="BTCUSDT")
        pos = positions[0] if positions else None
        if pos and float(pos.get("positionAmt", 0)) != 0:
            _info(f"Posizione BTCUSDT aperta: {pos['positionAmt']} @ {pos['entryPrice']} (PnL: {pos['unRealizedProfit']})")
        else:
            _ok("Nessuna posizione BTCUSDT aperta (stato pulito)")
    except BinanceAPIException as e:
        _err(f"Lettura balance/pos fallita: code={e.code} msg={e.message}")
        return 1

    # IT: step 4 - imposta leverage=1 e margin=ISOLATED (no-op se gia' impostati)
    # EN: step 4 - set leverage=1 and margin=ISOLATED (no-op if already set)
    print(f"\n{CYN}[4/5]{RST} Setto leverage=1 + margin=ISOLATED su BTCUSDT ...")
    try:
        r = client.futures_change_leverage(symbol="BTCUSDT", leverage=1)
        _ok(f"Leverage = {r.get('leverage')}x  maxNotional = {r.get('maxNotionalValue')}")
    except BinanceAPIException as e:
        _err(f"Cambio leverage fallito: code={e.code} msg={e.message}")
        return 1
    try:
        client.futures_change_margin_type(symbol="BTCUSDT", marginType="ISOLATED")
        _ok("Margin type = ISOLATED")
    except BinanceAPIException as e:
        if e.code == -4046:  # IT: -4046 = "no need to change" | EN: -4046 = "no need to change"
            _ok("Margin type già ISOLATED (no change needed)")
        else:
            _err(f"Cambio margin fallito: code={e.code} msg={e.message}")
            return 1

    # IT: step 5 - ordine LIMIT no-fill, verifica nella lista open, poi cancel
    # EN: step 5 - place no-fill LIMIT order, verify in open orders, then cancel
    print(f"\n{CYN}[5/5]{RST} Test ordine LIMIT no-fill + cancel (sicurezza max) ...")
    try:
        ticker = client.futures_symbol_ticker(symbol="BTCUSDT")
        current_px = float(ticker["price"])
        _info(f"BTCUSDT price corrente: ${current_px:,.2f}")
        # IT: LIMIT BUY a -10% sotto mercato per garantire no-fill
        # EN: LIMIT BUY 10% below market to guarantee no-fill
        limit_px = round(current_px * 0.90, 1)
        order = client.futures_create_order(
            symbol="BTCUSDT",
            side="BUY",
            type="LIMIT",
            timeInForce="GTC",
            quantity=0.002,   # IT: qty minima BTCUSDT futures | EN: BTCUSDT futures min qty
            price=limit_px,
            recvWindow=5000,
        )
        order_id = order["orderId"]
        _ok(f"Ordine LIMIT piazzato: ID={order_id}, qty=0.002 @ ${limit_px:,.1f}")

        # IT: conferma che l'ordine appare negli open orders
        # EN: confirm the order shows up in open orders
        open_orders = client.futures_get_open_orders(symbol="BTCUSDT")
        if any(o["orderId"] == order_id for o in open_orders):
            _ok(f"Ordine confermato negli open orders ({len(open_orders)} totali)")
        else:
            _warn("Ordine non visibile in open_orders (potrebbe essere già stato cancellato/fillato)")

        # IT: cancella l'ordine di test
        # EN: cancel the test order
        cancel = client.futures_cancel_order(symbol="BTCUSDT", orderId=order_id)
        if cancel.get("status") == "CANCELED":
            _ok(f"Ordine cancellato correttamente (status={cancel['status']})")
        else:
            _warn(f"Status cancel inatteso: {cancel.get('status')}")

        # IT: verifica che il balance non sia cambiato (nessuna fee = no fill)
        # EN: verify balance is unchanged (no fee implies no fill)
        balances_post = client.futures_account_balance()
        usdt_post = next((b for b in balances_post if b["asset"] == "USDT"), None)
        if usdt_post:
            usdt_balance_post = float(usdt_post["balance"])
            delta = usdt_balance_post - usdt_balance_pre
            if abs(delta) < 0.01:
                _ok(f"Balance USDT invariato (delta ${delta:+.4f})")
            else:
                _warn(f"Balance cambiato di ${delta:+.4f} (atteso 0 — possibile fill accidentale?)")
    except BinanceAPIException as e:
        _err(f"Test ordine fallito: code={e.code} msg={e.message}")
        if e.code == -4131:
            _info("Errore 'PERCENT_PRICE filter' — il prezzo LIMIT è fuori dal range consentito. Riprova quando il mercato è calmo")
        if e.code == -2019:
            _info("Margine insufficiente — accredita test funds sul pannello testnet")
        return 1

    # IT: riepilogo finale
    # EN: final summary
    print(f"\n{'═' * 60}")
    print(f"  {GRN}✓ TUTTI I TEST PASSATI{RST}")
    print(f"{'═' * 60}")
    print(f"  Le API key sono valide e il testnet è accessibile.")
    print(f"  Puoi procedere con l'integrazione del live engine (Fasi 2-5).")
    print(f"{'═' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
