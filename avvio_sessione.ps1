# IT: AVVIO SESSIONE (lato casa) - un solo comando alla riaccensione del PC:
#       1. pull+merge dei dati raccolti dal VPS mentre il PC era spento
#          (scripts/vps/pull_vps_data.ps1: host privato da config/secrets.yaml,
#          heartbeat staleness incluso);
#       2. check freshness regime B7 (refresh incrementale in background se
#          servono barre nuove). NESSUN processo residente parte piu' a casa:
#          01c/01d/01e e 04b vivono TUTTI sul VPS (systemd) dal 2026-07-18;
#       3. monitoraggio ricorrente linea vol (CPU-only, dal 2026-07-25):
#          derivazione MFIV incrementale + conteggio expiry qualificati
#          (--count-only) + contatori dei gate forward aperti.
#     Anti-duplicazione: se un processo e' gia' vivo NON viene rilanciato
#     (due 04b scriverebbero position/trades in conflitto).
#     NOTA encoding: file deliberatamente ASCII-only - PS 5.1 legge i .ps1
#     senza BOM come cp1252 e i caratteri unicode corrompono il parsing.
#     Uso: .\avvio_sessione.ps1 [-Days 7] [-SkipPull] [-SkipMonitor]
#     (da root progetto; il doppio click NON esegue i .ps1 - usa
#     "Esegui con PowerShell" o terminale).
# EN: SESSION STARTUP (home side) - a single command at PC power-on:
#       1. pull+merge of the data the VPS collected while the PC was off
#          (scripts/vps/pull_vps_data.ps1: private host from config/secrets.yaml,
#          staleness heartbeat included);
#       2. B7 regime freshness check (background incremental refresh when new
#          bars exist). NO resident process starts at home anymore: 01c/01d/01e
#          and 04b ALL live on the VPS (systemd) since 2026-07-18;
#       3. recurring vol-line monitoring (CPU-only, since 2026-07-25):
#          incremental MFIV derivation + qualifying-expiry count (--count-only)
#          + counters of the open forward gates.
#     Encoding note: deliberately ASCII-only - PS 5.1 reads BOM-less .ps1 as
#     cp1252 and unicode characters corrupt parsing.
#     Usage: .\avvio_sessione.ps1 [-Days 7] [-SkipPull] [-SkipMonitor]
#     (from project root; double-click does NOT run .ps1 - use "Run with
#     PowerShell" or a terminal).
param(
    [int]$Days = 7,
    [switch]$SkipPull,
    [switch]$SkipMonitor
)

$ErrorActionPreference = "Stop"
$ProjRoot = $PSScriptRoot
$Py = Join-Path $ProjRoot ".venv\Scripts\python.exe"

# IT: fail-fast se manca il venv (path esplicito: evita l'ambiguita' python->base).
# EN: fail-fast when the venv is missing (explicit path: avoids python->base ambiguity).
if (-not (Test-Path $Py)) { throw "venv non trovato/not found: $Py" }

# --- 1. Pull + merge dal VPS / VPS pull + merge ------------------------------
if (-not $SkipPull) {
    Write-Output "[sessione] pull+merge dal VPS (finestra $Days giorni)..."
    # IT: un pull fallito (VPS irraggiungibile) NON blocca l'avvio dei processi
    #     locali: la raccolta di casa deve partire comunque. Warning esplicito.
    # EN: a failed pull (unreachable VPS) does NOT block the local processes:
    #     home collection must start anyway. Explicit warning.
    try {
        & (Join-Path $ProjRoot "scripts\vps\pull_vps_data.ps1") -Days $Days
    } catch {
        Write-Warning "pull FALLITO/FAILED: $($_.Exception.Message) - proseguo con l'avvio locale / continuing with local startup"
    }
} else {
    Write-Output "[sessione] pull saltato (-SkipPull)"
}

# --- 2. Processi locali / local processes ------------------------------------
# IT: NESSUN processo residente a casa (decisione 2026-07-18 sera): 01c/01d/01e
#     e 04b vivono TUTTI sul VPS (systemd: quantsys-iv/ob/trades/volpaper). Il
#     PC e' passivo: solo pull+merge+check B7. EMERGENZA: se l'heartbeat del
#     merge stampa "WARN IV poller" (collector VPS giu'), lancia a mano finche'
#     il VPS non torna:
#       .\.venv\Scripts\python.exe scripts\01c_iv_poller.py
#     04b invece NON va MAI lanciato a casa: due --execute gestirebbero la
#     stessa posizione testnet (doppi ordini).
# EN: NO resident home process (2026-07-18 evening decision): 01c/01d/01e and
#     04b ALL live on the VPS (systemd: quantsys-iv/ob/trades/volpaper). The PC
#     is passive: pull+merge+B7 check only. EMERGENCY: if the merge heartbeat
#     prints "WARN IV poller" (VPS collector down), run manually until the VPS
#     is back:
#       .\.venv\Scripts\python.exe scripts\01c_iv_poller.py
#     04b must instead NEVER run at home: two --execute would manage the same
#     testnet position (double orders).
Write-Output "[sessione] nessun processo locale da avviare (tutto sul VPS dal 2026-07-18) / no local process to start (all on the VPS)"

# --- 3. Freshness regime probs (B7) / regime probs freshness (B7) ----------
# IT: se le candele hanno >= $RegimeStaleBars barre orarie oltre il checkpoint
#     walk-forward, lancia il refresh incrementale (01b --regime-incremental:
#     0-1 fit MLE, minuti) in background. Con candele congelate (es. span
#     esperimenti A3/A8) il check stampa "fresco" ed e' un no-op. L'append
#     incrementale NON altera le righe storiche (catena causale, scaler
#     congelato): non puo' disallineare esperimenti in corso.
# EN: if the candles are >= $RegimeStaleBars hourly bars past the walk-forward
#     checkpoint, launch the incremental refresh (01b --regime-incremental:
#     0-1 MLE fits, minutes) in the background. With frozen candles (e.g. the
#     A3/A8 experiment span) the check prints "fresh" and is a no-op. The
#     incremental append does NOT alter historical rows (causal chain, frozen
#     scaler): it cannot misalign running experiments.
$RegimeStaleBars = 168  # IT: soglia = 1 settimana di barre 1h / EN: threshold = 1 week of 1h bars
# IT: NOTA quoting: solo apici SINGOLI nel codice python - PS 5.1 strippa le
#     doppie virgolette negli argomenti ai native exe (bug classico di quoting).
# EN: quoting NOTE: SINGLE quotes only in the python code - PS 5.1 strips double
#     quotes in arguments to native exes (classic quoting bug).
$pyRegimeCheck = @'
import pickle, sys
from pathlib import Path
import pandas as pd
c = Path('data/raw_candles.parquet'); k = Path('data/regime_wf_checkpoint.pkl')
if not c.exists() or not k.exists():
    print(-1); sys.exit(0)
s = pd.read_parquet(c, columns=['open_time'])['open_time']
if not pd.api.types.is_datetime64_any_dtype(s):
    s = pd.to_datetime(s, unit='ms', utc=True)
t = s.max()
t = t.tz_localize('UTC') if t.tz is None else t.tz_convert('UTC')
with open(k, 'rb') as f:
    ts = pickle.load(f)['last_timestamp']
print(int((t - ts) / pd.Timedelta('1h')))
'@
$barsNew = & $Py -c $pyRegimeCheck 2>$null | Select-Object -Last 1
if ($barsNew -match '^-?\d+$') {
    $barsNew = [int]$barsNew
    if ($barsNew -lt 0) {
        Write-Output "[sessione] regime B7: candele o checkpoint assenti - check saltato / missing artifacts, check skipped"
    } elseif ($barsNew -ge $RegimeStaleBars) {
        # IT: anti-duplicazione: un 01b gia' vivo (full rebuild o incrementale) ha la precedenza.
        # EN: anti-duplication: an already-alive 01b (full rebuild or incremental) takes precedence.
        $alive01b = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                      Where-Object { $_.CommandLine -match "01b_download_macro" })
        if ($alive01b.Count -gt 0) {
            Write-Output "[sessione] regime B7: 01b GIA' ATTIVO (pid $($alive01b[0].ProcessId)) - non rilancio / already running"
        } else {
            Start-Process -WindowStyle Hidden -WorkingDirectory $ProjRoot -FilePath $Py `
                -ArgumentList @("scripts/01b_download_macro.py", "--regime-incremental")
            Write-Output "[sessione] regime B7: STALE ($barsNew barre nuove >= $RegimeStaleBars) - refresh incrementale AVVIATO in background (esito in logs\quantsys_*.log) / incremental refresh STARTED"
        }
    } else {
        Write-Output "[sessione] regime B7: fresco / fresh ($barsNew barre nuove / new bars < $RegimeStaleBars)"
    }
} else {
    Write-Warning "[sessione] regime B7: check fallito/failed ($barsNew) - refresh manuale: python scripts\01b_download_macro.py --regime-incremental"
}

# --- 4. Monitoraggio ricorrente linea vol / recurring vol-line monitoring ----
# IT: i tre passi che la routine di sessione richiedeva a mano (STATUS 2026-07-22
#     "monitoraggio ricorrente per sessione"). Tutti CPU-only e OFF-PATH: nessun
#     file di produzione toccato, zero GPU.
#     ATTENZIONE - DISCIPLINA ONE-SHOT (pre-reg MFIV v2): qui si lancia SOLO --count-only,
#     che calcola timestamp e NIENTE altro (nessun PnL, nessun edge, nessuna
#     correlazione); il giudice ha comunque il guard n<N_MIN -> NO_RUN e non
#     scrive report sotto soglia. Il run one-shot vero va lanciato A MANO alla
#     prima sessione con n>=40: NON automatizzarlo qui.
#     Ordine vincolante: derive_mfiv DOPO il merge (legge la chain appena
#     scaricata). Fail-soft: un errore non blocca la routine.
# EN: the three steps the session routine required manually (STATUS 2026-07-22).
#     All CPU-only and OFF-PATH: no production file touched, zero GPU.
#     WARNING - ONE-SHOT DISCIPLINE (MFIV v2 pre-reg): only --count-only runs here, which
#     computes timestamps and NOTHING else (no PnL, no edge, no correlation); the
#     judge also guards n<N_MIN -> NO_RUN and writes no report below threshold.
#     The real one-shot run must be launched BY HAND at the first session with
#     n>=40: do NOT automate it here.
#     Binding order: derive_mfiv AFTER the merge (it reads the freshly pulled
#     chain). Fail-soft: an error does not stop the routine.
if (-not $SkipMonitor) {
    # IT: nota PS 5.1 - un exe nativo che esce !=0 NON solleva eccezione (il
    #     try/catch non basta): si controlla $LASTEXITCODE esplicitamente.
    # EN: PS 5.1 note - a native exe exiting !=0 does NOT throw (try/catch is not
    #     enough): $LASTEXITCODE is checked explicitly.
    Write-Output "[sessione] monitoraggio vol: derivazione MFIV incrementale / incremental MFIV derivation..."
    & $Py (Join-Path $ProjRoot "scripts\vol\derive_mfiv.py")
    if ($LASTEXITCODE -ne 0) { Write-Warning "derive_mfiv FALLITO/FAILED (exit $LASTEXITCODE) - conteggio MFIV sotto sara' su dati non aggiornati / count below is on stale data" }

    Write-Output "[sessione] monitoraggio vol: conteggio expiry qualificati (solo timestamp) / qualifying-expiry count (timestamps only)..."
    & $Py (Join-Path $ProjRoot "scripts\vol\mfiv_comparator_judge.py") --count-only
    if ($LASTEXITCODE -ne 0) { Write-Warning "mfiv_comparator_judge --count-only FALLITO/FAILED (exit $LASTEXITCODE)" }

    # IT: continuita' del recorder L2 - NON e' la stessa cosa della freschezza gia'
    #     stampata dagli HEARTBEAT del merge. Il campione del filone order-book e' fatto
    #     di ore CONTIGUE: una finestra richiede T+h barre consecutive, quindi un'ora di
    #     buco non costa un'ora ma ~149 finestre (6.2 giorni di accumulo). Un file fresco
    #     con un buco DENTRO passava il check heartbeat senza dire nulla, e l'epoca
    #     "casa" ha gia' prodotto 32 giorni di raccolta con ZERO finestre utilizzabili.
    #     Sola lettura dei timestamp: nessun valore di feature, nessun rischio one-shot.
    # EN: L2 recorder continuity - NOT the same as the freshness already printed by the
    #     merge HEARTBEATs. The order-book sample is made of CONTIGUOUS hours: a window
    #     needs T+h consecutive bars, so one gap hour costs not one hour but ~149 windows
    #     (6.2 days of accrual). A fresh file with a gap INSIDE passed the heartbeat check
    #     silently, and the "home" epoch already produced 32 days of collection with ZERO
    #     usable windows. Timestamps only: no feature values, no one-shot risk.
    # IT: NOTA 2026-08-03 - il check gira SUBITO DOPO il pull, quindi legge una coda non
    #     consolidata: l'ora in corso e' esclusa e i buchi nelle ultime ore sono marcati
    #     PROVVISORI. Un buco e' un fatto solo se sopravvive al pull successivo.
    # EN: NOTE 2026-08-03 - the check runs RIGHT AFTER the pull, so it reads an
    #     unconsolidated tail: the in-progress hour is excluded and gaps in the last hours
    #     are flagged PROVISIONAL. A gap is a fact only if it survives the next pull.
    Write-Output "[sessione] monitoraggio vol: continuita' recorder L2 / L2 recorder continuity..."
    & $Py (Join-Path $ProjRoot "scripts\vol\l2_continuity_check.py") --days $Days
    if ($LASTEXITCODE -ne 0) { Write-Warning "l2_continuity_check FALLITO/FAILED (exit $LASTEXITCODE)" }

    # IT: contatori dei gate forward aperti (leg opzioni n>=30, hedged n>=20).
    #     Sola lettura dei ledger di 04b: nessun PnL aggregato, nessun verdetto.
    #     NOTA quoting: solo apici SINGOLI (PS 5.1 strippa le doppie virgolette
    #     negli argomenti ai native exe).
    # EN: counters of the open forward gates (option legs n>=30, hedged n>=20).
    #     Read-only over 04b ledgers: no aggregate PnL, no verdict.
    #     Quoting NOTE: SINGLE quotes only (PS 5.1 strips double quotes in native
    #     exe arguments).
    $pyGateCounters = @'
import json
from pathlib import Path
tp = Path('results/vol_paper/trades.jsonl')
hp = Path('results/vol_paper/hedge_ledger.jsonl')
n_rows = n_exec = 0
if tp.exists():
    rows = [json.loads(l) for l in tp.open(encoding='utf-8') if l.strip()]
    n_rows = len(rows)
    n_exec = sum(1 for r in rows if r.get('executed'))
# IT: il campione pre-registrato del giudice hedged conta i TRADE aperti con
#     hedge attivo, NON gli eventi di ledger (1 posizione = open + N rebalance +
#     flatten): stampare gli eventi invitava a leggere '22 >= 20' e a lanciare il
#     giudice in anticipo. Unita' di misura = position_key distinte con >=1 hedge
#     eseguito. Esclusa la posizione 19JUL26 (aperta UNHEDGED, hedge solo da
#     meta' vita: esclusione pre-dichiarata in STATUS 2026-07-18).
# EN: the hedged judge's pre-registered sample counts TRADES opened with the hedge
#     active, NOT ledger events (1 position = open + N rebalance + flatten):
#     printing events invited reading '22 >= 20' and running the judge early.
#     Unit = distinct position_key with >=1 executed hedge. The 19JUL26 position
#     is excluded (opened UNHEDGED, hedge only from mid-life: exclusion
#     pre-declared in STATUS 2026-07-18).
EXCLUDED_EXPIRY_MS = 1784448000000  # 2026-07-19 08:00 UTC
n_hedge_ev = 0
hedged_pos = set()
if hp.exists():
    for l in hp.open(encoding='utf-8'):
        if not l.strip():
            continue
        r = json.loads(l)
        n_hedge_ev += 1
        pk = r.get('position_key') or {}
        if not r.get('executed') or pk.get('expiry_ms') == EXCLUDED_EXPIRY_MS:
            continue
        hedged_pos.add(json.dumps(pk, sort_keys=True))
print(f'[gate] leg opzioni / option legs: n={n_exec} executed ({n_rows} righe/rows) - soglia/threshold n>=30')
print(f'[gate] hedged: n={len(hedged_pos)} posizioni hedge-attive/hedge-active positions ({n_hedge_ev} eventi/events nel ledger) - soglia/threshold n>=20')
'@
    & $Py -c $pyGateCounters
    if ($LASTEXITCODE -ne 0) { Write-Warning "contatori gate FALLITI/FAILED (exit $LASTEXITCODE)" }

    # IT: contatore del campione E1 stadio 2 (confermativo, expiry liquidate dopo il
    #     2026-08-01). Aggiunto il 2026-08-06 dopo aver scoperto che il numero in
    #     continuita' era fermo a 0 da cinque giorni: non era nella routine, quindi
    #     nessuno lo rileggeva, e sotto c'era un secondo difetto che lo avrebbe fatto
    #     leggere basso comunque (vedi il check di freschezza qui sotto).
    #     ATTENZIONE - DUE PRECAUZIONI, entrambe necessarie:
    #     (1) si lancia --count-only, MAI `--stage 2` nudo: il guard n<40 -> NO_RUN
    #         protegge solo SOTTO soglia, quindi a n>=40 il comando nudo calcolerebbe
    #         le tre condizioni e scriverebbe il verdetto PER AUTOMAZIONE - cioe'
    #         esattamente il run one-shot che il protocollo impone manuale;
    #     (2) il conteggio dipende dalla serie dei close (una expiry e' osservabile
    #         solo se la sua RV e' calcolabile), quindi si stampa PRIMA fin dove
    #         arriva quella serie: un raw_candles.parquet stale fa scendere n senza
    #         che nulla lo dica. Il rimedio (01_update_data.py --candles-only) NON e'
    #         automatizzato di proposito: e' una scrittura su un file di dati, e
    #         questo blocco resta a scrittura zero.
    # EN: E1 stage-2 sample counter (confirmatory, expiries settling after 2026-08-01).
    #     Added 2026-08-06 after finding the continuity-log number had been stuck at 0
    #     for five days: it was not in the routine, so nobody re-read it, and beneath
    #     it sat a second defect that would have made it read low anyway (see the
    #     freshness check below).
    #     WARNING - TWO PRECAUTIONS, both required:
    #     (1) run --count-only, NEVER a bare `--stage 2`: the n<40 -> NO_RUN guard only
    #         protects BELOW threshold, so at n>=40 the bare command would compute the
    #         three conditions and write the verdict BY AUTOMATION - exactly the
    #         one-shot run the protocol requires to be manual;
    #     (2) the count depends on the close series (an expiry is observable only if
    #         its RV is computable), so how far that series reaches is printed FIRST: a
    #         stale raw_candles.parquet lowers n with nothing saying so. The remedy
    #         (01_update_data.py --candles-only) is deliberately NOT automated: it
    #         writes to a data file, and this block stays write-free.
    $pyCloseFreshness = @'
import importlib.util
from pathlib import Path
import pandas as pd
# IT: la serie e' letta con la funzione del giudice stesso - una copia qui sarebbe
#     una seconda sorgente di verita' sulla stessa domanda.
# EN: the series is read with the judge's own function - a copy here would be a
#     second source of truth on the same question.
spec = importlib.util.spec_from_file_location(
    'e1_judge', Path('scripts/vol/edge_information_judge.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
c = m.hourly_close()
last = c.index.max()
lag = (pd.Timestamp.now(tz='UTC').floor('h') - last) / pd.Timedelta('1h')
print(f'[e1] serie close fino a/through {last:%Y-%m-%d %H:%M} UTC - ritardo/lag {lag:.0f}h')
if lag >= 6:
    print(f'[e1] ATTENZIONE/WARNING: serie close STALE - il conteggio sotto e SOTTOSTIMATO / the count below is UNDERSTATED')
    print(f'[e1] rimedio/remedy: python scripts\\01_update_data.py --candles-only  (estende SOLO raw_candles.parquet / extends raw_candles.parquet ONLY)')
'@
    Write-Output "[sessione] monitoraggio vol: campione E1 stadio 2 (solo conteggio) / E1 stage-2 sample (count only)..."
    & $Py -c $pyCloseFreshness
    if ($LASTEXITCODE -ne 0) { Write-Warning "check freschezza serie close FALLITO/FAILED (exit $LASTEXITCODE) - il conteggio E1 sotto non e' interpretabile / the E1 count below is not interpretable" }
    & $Py (Join-Path $ProjRoot "scripts\vol\edge_information_judge.py") --stage 2 --count-only
    if ($LASTEXITCODE -ne 0) { Write-Warning "edge_information_judge --stage 2 --count-only FALLITO/FAILED (exit $LASTEXITCODE)" }
} else {
    Write-Output "[sessione] monitoraggio vol saltato (-SkipMonitor)"
}

Write-Output "[sessione] pronto / ready. Log vivi: logs\quantsys_*.log (piu' recenti per mtime)"
