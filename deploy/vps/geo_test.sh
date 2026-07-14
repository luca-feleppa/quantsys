#!/usr/bin/env bash
# IT: GEO-TEST pre-deploy (da lanciare PRIMA di installare qualsiasi cosa sul VPS):
#     verifica che l'IP del datacenter NON sia geo-bloccato da Binance (HTTP 451)
#     e che Deribit prod+testnet rispondano. Se Binance fallisce, il VPS va reso
#     subito (finestra di recesso) — nessun workaround lato nostro.
# EN: Pre-deploy GEO-TEST (run BEFORE installing anything on the VPS): verifies
#     the datacenter IP is NOT geo-blocked by Binance (HTTP 451) and that Deribit
#     prod+testnet respond. If Binance fails, return the VPS immediately
#     (withdrawal window) — there is no workaround on our side.
set -u

FAIL=0

# IT: check generico — confronta lo status HTTP atteso; 451 = geo-block Binance.
# EN: generic check — compares the expected HTTP status; 451 = Binance geo-block.
check() {
    local name="$1" url="$2"
    local code
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$url")
    if [ "$code" = "200" ]; then
        echo "OK   $name ($code)"
    else
        echo "FAIL $name (HTTP $code) — $url"
        FAIL=1
    fi
}

echo "=== QUANTSYS geo-test collector VPS ==="
# IT: endpoint critici: senza questi i collector 01c/01d non funzionano.
# EN: critical endpoints: without these the 01c/01d collectors cannot work.
check "Binance REST ping"   "https://api.binance.com/api/v3/ping"
check "Binance REST depth"  "https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=5"
check "Deribit prod"        "https://www.deribit.com/api/v2/public/get_time"
check "Deribit testnet"     "https://test.deribit.com/api/v2/public/get_time"
# IT: fallback solo-market-data (klines senza geo-fence) — informativo, non critico.
# EN: market-data-only fallback (geo-fence-free klines) — informative, not critical.
check "Binance vision (fallback)" "https://data-api.binance.vision/api/v3/ping"

echo
if [ "$FAIL" -eq 0 ]; then
    echo "VERDETTO / VERDICT: PASS — l'IP è utilizzabile / the IP is usable."
else
    echo "VERDETTO / VERDICT: FAIL — NON deployare, IP geo-bloccato o rete KO."
    echo "                    Do NOT deploy: geo-blocked IP or broken network."
fi
exit "$FAIL"
