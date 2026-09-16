🇮🇹 Italiano · [🇬🇧 English](DASHBOARD_IMPROVEMENTS.md)

# Dashboard Improvements — backlog proposte

> **Verifica 2026-09-10:** backlog conservato: il terminale e il parser locale non esauriscono D1–D6. Le indicazioni temporali di luglio sotto sono storiche, non autorizzazioni: l'hedge v2 è chiuso FAIL e il suo pre-hedge non è più una priorità. D4 resta un miglioramento operativo; D1–D3 devono rispettare i vincoli dei campioni forward e dei descrittivi. D5 è opzionale, D6 ancora opportunistico. Coda e gate correnti: [STATUS.md](../STATUS.md).

> Creato 2026-07-16 (sessione pre-chiusura gate v1). Stato: **PROPOSTE, nessuna implementata**. Perimetro: `scripts/06_dashboard.py` (risk terminal Deribit, 4 tab: Surface / Chain / Risk & Greeks / Trades; single-file, GPU-free, per design SCOLLEGATO dai modelli ML). Il filo conduttore: i margini veri non sono estetici — sono i dati già su disco che nessuna tab mostra. Nessun item tocca path pre-registrati: è tutto strato di lettura.

## Item (ordinati per valore)

| # | Item | Effort | Quando |
|---|---|---|---|
| D1 | **Tab VRP monitor**: `iv_30h` (H24 dal 14/07) vs realized vol 30h successiva dalle candele = VRP realizzato nel tempo, con overlay dei trade del forward test (entry, edge, esito). Rende visibile anche il caveat orario pre/post-VPS del gate | M | post-gate v1 |
| D2 | **Tab Trades: da elenco ad analisi**: equity curve cumulata BTC; scatter edge-all'entry → PnL realizzato ("l'edge predice il PnL?" in un grafico); pannello attribution A11 da `results/vol_paper/attribution.parquet` (Δ/Γ/ν/Θ/residuo per trade, coverage dichiarata — oggi quel parquet non lo guarda nessuno) | M | post-gate v1 (primo) |
| D3 | **Pannello spread realizzati** dai dati 01e (`data/deribit_trades/`): distribuzione \|price−mark\| per moneyness/tenor/ora del giorno — consumer naturale del collector 2026-07-16, input diretto ai costi eseguibili post-gate | M | post-gate v1 |
| D4 | **Pannello infra/posizione**: freshness dei parquet canonici (heartbeat visivo IV/L2/trades senza `check_vps`), età ultimo pull; per la posizione aperta: countdown expiry + distanza pin \|S−K\|/S (metrica A13). Oggi tutto CLI+log | S | post-gate v1 (primo) |
| D5 | **Confronto δ venue vs δ BS locale** sull'ATM straddle (serie `atm_greeks.parquet` vs `bs_greeks`): validazione visiva della convenzione raw/premium-adjusted, supporta la decisione pre-hedge v2 | S | dopo C4 (`--greeks` attivo) |
| D6 | **Igiene tecnica**: consolidare la 4ª copia del parser Deribit (audit 2026-07-16: import `parse_instrument` da `quantsys/data/deribit.py`, mantenendo sessione+retry propri); opzionale fit SVI/SABR dello smile per display (massimo upgrade modellistico giustificato) | S | opportunistico |

## Non-goal (respinti con razionale)

- **Collegare i modelli ML alla dashboard** — la separazione è un design vinto (terminale di mercato, zero contesa GPU, zero rischio di contaminare il path production).
- **Dupire / Variance-Gamma / modelli misti per le greche locali** — respinto 2026-07-16: dati radi (4 expiry corte, ali illiquide) → derivate seconde instabili; il delta divergerebbe da quello del venue, che è l'unico decisionale (verdetto 07-08). Le greche `bs_greeks` restano display-only.
- **WebSocket al posto del polling REST** — complessità senza beneficio alla cadenza attuale.

## Ordine pratico suggerito

D2 e D4 subito dopo la chiusura del gate v1 (servono alla valutazione e all'operatività v2) → D1 e D3 nella stessa finestra (condividono l'infrastruttura dati) → D5 quando C4 è attivo → D6 opportunistico.
