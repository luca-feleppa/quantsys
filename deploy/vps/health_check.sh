#!/usr/bin/env bash
# IT: HEALTH-CHECK del VPS collector (gira SUL VPS; lanciato da casa via
#     scripts/vps/check_vps.ps1). Controlla: servizi systemd attivi + conteggio
#     restart (crash-loop), freschezza dei parquet (poller 10' / recorder 5s),
#     disco, RAM, geo-block Binance/Deribit. Output: una riga PASS/WARN per
#     check + verdetto finale. Exit 0 = tutto PASS, 1 = almeno un WARN.
# EN: Collector VPS HEALTH-CHECK (runs ON the VPS; invoked from home via
#     scripts/vps/check_vps.ps1). Checks: systemd services active + restart
#     count (crash-loop), parquet freshness (10' poller / 5s recorder), disk,
#     RAM, Binance/Deribit geo-block. Output: one PASS/WARN line per check +
#     final verdict. Exit 0 = all PASS, 1 = at least one WARN.
set -u
ROOT="/opt/quantsys"
FAIL=0

pass() { echo "PASS  $1"; }
warn() { echo "WARN  $1"; FAIL=1; }

echo "=== QUANTSYS VPS health-check $(date -u '+%Y-%m-%d %H:%M UTC') ==="

# IT: 1) servizi attivi + restart count (NRestarts alto = crash-loop mascherato
#     da Restart=always: il servizio risulta active ma muore di continuo).
# EN: 1) services active + restart count (high NRestarts = crash-loop masked by
#     Restart=always: the unit shows active but keeps dying).
for svc in quantsys-iv quantsys-ob quantsys-trades quantsys-volpaper; do
    state=$(systemctl is-active "$svc" 2>/dev/null || true)
    nrst=$(systemctl show "$svc" -p NRestarts --value 2>/dev/null || echo "?")
    if [ "$state" = "active" ]; then
        if [ "${nrst:-0}" -gt 10 ] 2>/dev/null; then
            warn "$svc: active ma $nrst restart (possibile crash-loop / possible crash-loop)"
        else
            pass "$svc: active (restart: $nrst)"
        fi
    else
        warn "$svc: $state"
    fi
done

# IT: 2) freschezza output — mtime entro la soglia attesa per cadenza.
# EN: 2) output freshness — mtime within the cadence-expected threshold.
fresh() {  # $1=path-glob  $2=soglia-minuti/threshold-min  $3=label
    local newest
    newest=$(ls -t $1 2>/dev/null | head -1)
    if [ -z "$newest" ]; then
        warn "$3: nessun file / no file ($1)"
    elif [ -n "$(find "$newest" -mmin -"$2" 2>/dev/null)" ]; then
        pass "$3: fresco/fresh ($(basename "$newest"))"
    else
        warn "$3: STALE >$2 min ($(basename "$newest") $(stat -c '%y' "$newest" | cut -c1-16))"
    fi
}
fresh "$ROOT/data/iv/atm_30h.parquet"      30 "IV poller (atm_30h)"
fresh "$ROOT/data/iv/chain/*.parquet"      30 "IV poller (chain)"
fresh "$ROOT/data/orderbook/*.parquet"     10 "L2 recorder (orderbook)"
# IT: 01e scrive solo se ci sono trade nuovi: soglia larga (60') anti falsi-WARN.
# EN: 01e writes only when new trades exist: wide threshold (60') vs false WARNs.
fresh "$ROOT/data/deribit_trades/*.parquet" 60 "Trades recorder (deribit_trades)"
# IT: 04b appende un forecast a ogni tick orario: soglia 130' = tollera UN tick
#     fallito (retry al successivo) senza falsi-WARN.
# EN: 04b appends one forecast per hourly tick: 130' threshold = tolerates ONE
#     failed tick (next-tick retry) without false WARNs.
fresh "$ROOT/results/vol_paper/forecasts.parquet" 130 "Vol-paper 04b (forecasts)"

# IT: 3) disco e RAM (warn oltre 80% disco; RAM solo informativa).
# EN: 3) disk and RAM (warn above 80% disk; RAM informational only).
duse=$(df --output=pcent / | tail -1 | tr -dc '0-9')
if [ "${duse:-0}" -lt 80 ]; then pass "disco/disk: ${duse}% usato/used"; else warn "disco/disk: ${duse}% usato/used"; fi
echo "INFO  RAM: $(free -m | awk '/Mem:/{printf "%d/%d MB", $3, $2}')"

# IT: 4) geo-block (può comparire DOPO il deploy: Binance aggiorna le blocklist).
# EN: 4) geo-block (can appear AFTER deploy: Binance updates its blocklists).
for probe in "Binance|https://api.binance.com/api/v3/ping" \
             "Deribit|https://www.deribit.com/api/v2/public/get_time"; do
    name="${probe%%|*}"; url="${probe#*|}"
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url")
    if [ "$code" = "200" ]; then pass "$name raggiungibile/reachable (200)"; else warn "$name: HTTP $code"; fi
done

echo
if [ "$FAIL" -eq 0 ]; then
    echo "VERDETTO / VERDICT: PASS — VPS in salute / VPS healthy"
else
    echo "VERDETTO / VERDICT: WARN — controlla le righe WARN / check the WARN lines"
fi
exit "$FAIL"
