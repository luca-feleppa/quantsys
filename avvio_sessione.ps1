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
#     Uso: .\avvio_sessione.ps1 [-Days 7] [-SkipPull] [-SkipMonitor] [-RefreshCandles]
#     (-RefreshCandles NON e' un default: estende raw_candles.parquet, vedi 3bis / see 3bis)
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
#     Usage: .\avvio_sessione.ps1 [-Days 7] [-SkipPull] [-SkipMonitor] [-RefreshCandles]
#     (-RefreshCandles is NOT a default: it extends raw_candles.parquet, see 3bis)
#     (from project root; double-click does NOT run .ps1 - use "Run with
#     PowerShell" or a terminal).
param(
    [int]$Days = 7,
    [switch]$SkipPull,
    [switch]$SkipMonitor,
    [switch]$RefreshCandles
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

# --- 3bis. Estensione candele su richiesta esplicita / candle extension on demand -
# IT: NOTA sulla numerazione - qui i blocchi contano anche il "2. processi locali"
#     (oggi un no-op), quindi questo passo e' 3bis; nella tabella di AVVIO.md 5.3,
#     che elenca i soli tre blocchi attivi, lo stesso passo e' 2bis.
# EN: numbering NOTE - blocks here also count "2. local processes" (a no-op today),
#     so this step is 3bis; in the AVVIO.md 5.3 table, which lists the three active
#     blocks only, the same step is 2bis.
# IT: -RefreshCandles estende data/raw_candles.parquet con 01_update_data.py
#     --candles-only. NON e' un default, ed e' una decisione ogni volta: modellato
#     su -PromoteMacro, per la stessa ragione per cui quello esiste. La meccanica
#     e' sicura (no-op se non c'e' nulla di nuovo, dedup che tiene la riga
#     ESISTENTE quindi la storia non si riscrive, mai la barra in formazione,
#     scrittura atomica, file mai spedito al VPS), ma automatizzarla toglierebbe
#     la possibilita' di CONGELARE i dati per un esperimento pre-registrato:
#     l'invariante "candele/npz/regime_probs non si toccano fino a chiusura gate"
#     passerebbe da "non fare nulla" a "ricordarsi -SkipMonitor", ed e' la forma
#     esatta della promozione macro avvenuta per automazione il 2026-07-31.
#     Quando serve: prima di annotare il contatore E1 se il blocco 4 stampa il
#     warning di ritardo, e prima del run one-shot di E1 stadio 2 (prerequisito 5
#     della sua pre-registrazione). Fra i gate aperti solo E1 legge le barre: il
#     giudice hedged legge i ledger, il comparatore MFIV il chain.
#     ATTENZIONE - effetto a distanza di una sessione, voluto: estendere le
#     candele fa avanzare il contatore di staleness B7, quindi il refresh
#     incrementale del regime puo' partire al PROSSIMO avvio (blocco 3, che qui
#     sopra ha gia' girato). Le due scritture restano cosi' separate e ognuna
#     visibile per conto suo, invece di concatenarsi in silenzio.
# EN: -RefreshCandles extends data/raw_candles.parquet via 01_update_data.py
#     --candles-only. NOT a default, and a decision every time: modelled on
#     -PromoteMacro, for the same reason that one exists. The mechanics are safe
#     (no-op when nothing is new, dedup keeps the EXISTING row so history is never
#     rewritten, never the in-progress bar, atomic write, file never shipped to the
#     VPS), but automating it would remove the ability to FREEZE data for a
#     pre-registered experiment: the "candles/npz/regime_probs untouched until the
#     gate closes" invariant would go from "do nothing" to "remember -SkipMonitor",
#     which is exactly the shape of the macro promotion that happened by automation
#     on 2026-07-31.
#     When it is needed: before recording the E1 counter if block 4 prints the lag
#     warning, and before the one-shot run of E1 stage 2 (prerequisite 5 of its
#     pre-registration). Among the open gates only E1 reads the bars: the hedged
#     judge reads the ledgers, the MFIV comparator the chain.
#     WARNING - deliberate one-session delayed effect: extending the candles moves
#     the B7 staleness counter forward, so the incremental regime refresh may start
#     at the NEXT startup (block 3, which has already run above). The two writes
#     stay separate and each visible on its own, instead of chaining silently.
if ($RefreshCandles) {
    Write-Output "[sessione] -RefreshCandles: estensione di data\raw_candles.parquet (solo candele) / candle-only extension..."
    & $Py (Join-Path $ProjRoot "scripts\01_update_data.py") --candles-only
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "01_update_data --candles-only FALLITO/FAILED (exit $LASTEXITCODE) - la serie close resta dov'era, il contatore E1 sotto sara' SOTTOSTIMATO / the close series is unchanged, the E1 count below will be UNDERSTATED"
    }
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

    # IT: copertura del file di barre a 1 MINUTO. Non e' un doppione del check L2
    #     qui sopra: quello misura la continuita' del RECORDER, questo misura se
    #     esiste il TARGET con cui quelle ore verrebbero giudicate. Il giudice B1
    #     costruisce la RV da rendimenti a 1m (~180 quadrati per osservazione invece
    #     di 3), quindi le ore di L2 oltre la fine del file 1m sono raccolte ma prive
    #     di target: il campione utile e' il MINIMO fra i due, e finora nessuno dei
    #     due contatori lo diceva.
    #     PRECISIONE - il tetto vale per le analisi a target 1m (B1 a h=3, proxy del
    #     pin-close). Il contatore n_eff a h=30 stampato dal check L2 usa le barre
    #     ORARIE (target di produzione: 30 barre da rendimenti orari) e NON e' toccato
    #     da questo file: scriverlo senza qualificare l'orizzonte darebbe l'allarme
    #     sbagliato sul gate sbagliato.
    #     ATTENZIONE - il file ha TRE consumatori (l2_incremental_judge,
    #     pin_close_feasibility, edge_information_judge come sorgente di coda) e
    #     ZERO produttori: nessuno script del repo lo genera o lo estende, e' stato
    #     acquisito a mano. Quindi qui si misura soltanto, e il messaggio dice cosa
    #     manca invece di suggerire un comando che non esiste.
    #     Sola lettura di timestamp e nomi file: nessun valore di feature.
    # EN: 1-MINUTE bar file coverage. Not a duplicate of the L2 check above: that one
    #     measures RECORDER continuity, this one measures whether the TARGET those
    #     hours would be judged against exists at all. The B1 judge builds RV from 1m
    #     returns (~180 squares per observation instead of 3), so L2 hours beyond the
    #     end of the 1m file are collected but NOT judgeable: the usable sample is the
    #     MINIMUM of the two, and until now neither counter said so.
    #     WARNING - the file has THREE consumers (l2_incremental_judge,
    #     pin_close_feasibility, edge_information_judge as a tail source) and ZERO
    #     producers, BY DECISION (2026-08-10): the 1m klines are HISTORICAL and
    #     re-downloadable indefinitely, so the lag is not a maturing debt - it is a
    #     download not yet done, and waiting costs nothing. A standing producer would
    #     keep a file current that no operational path reads. So this only measures,
    #     and does not suggest a command: when a 1m-target analysis is actually
    #     reopened, the download is decided then, over the window it needs.
    #     Timestamps and file names only: no feature values.
    $pyOneMinCoverage = @'
from pathlib import Path
import pandas as pd
p = Path('data/raw_candles_1m_l2.parquet')
if not p.exists():
    print('[1m] data/raw_candles_1m_l2.parquet ASSENTE - il giudice B1 non ha target / MISSING')
else:
    d = pd.read_parquet(p, columns=['open_time'])
    t = pd.to_datetime(d['open_time'], utc=True)
    last, first = t.max(), t.min()
    lag_d = (pd.Timestamp.now(tz='UTC') - last) / pd.Timedelta('1D')
    print(f'[1m] {len(d):,} barre/bars {first:%Y-%m-%d} -> {last:%Y-%m-%d %H:%M} UTC - ritardo/lag {lag_d:.1f} giorni/days')
    # IT: giorni di L2 gia' registrati che restano senza target 1m -> tetto sul
    #     campione B1 indipendente dalla continuita' del recorder.
    # EN: already-recorded L2 days left without a 1m target -> a cap on the B1 sample
    #     independent of recorder continuity.
    days = sorted(q.stem.split('_')[-1] for q in Path('data/orderbook').glob('l2_features_*.parquet'))
    if days:
        l2_last = pd.Timestamp(days[-1], tz='UTC')
        uncov = (l2_last.normalize() - last.normalize()) / pd.Timedelta('1D')
        if uncov > 0:
            print(f'[1m] ATTENZIONE/WARNING: L2 registrata fino al/through {l2_last:%Y-%m-%d}, target 1m fino al/through {last:%Y-%m-%d}')
            print(f'[1m] -> {uncov:.0f} giorni di L2 senza target a 1 minuto / days of L2 without a 1-minute target')
            print(f'[1m]    riguarda le analisi a target 1m (B1 h=3, proxy pin-close); il contatore n_eff a h=30 usa le barre ORARIE ed e intatto')
            print(f'[1m]    affects 1m-target analyses (B1 h=3, pin-close proxies); the h=30 n_eff counter uses HOURLY bars and is unaffected')
            print(f'[1m] nessun path operativo lo legge; le kline 1m sono storiche e RI-SCARICABILI: non e un debito che matura / no operational path reads it; 1m klines are historical and RE-DOWNLOADABLE: not a maturing debt')
        else:
            print(f'[1m] copertura/coverage OK: il target 1m arriva almeno quanto la L2 registrata / 1m target reaches at least as far as recorded L2')
'@
    Write-Output "[sessione] monitoraggio vol: copertura del file barre 1m / 1m bar file coverage..."
    & $Py -c $pyOneMinCoverage
    if ($LASTEXITCODE -ne 0) { Write-Warning "check copertura 1m FALLITO/FAILED (exit $LASTEXITCODE)" }

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
