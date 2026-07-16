# IT: AVVIO SESSIONE (lato casa) - un solo comando alla riaccensione del PC:
#       1. pull+merge dei dati raccolti dal VPS mentre il PC era spento
#          (scripts/vps/pull_vps_data.ps1: host privato da config/secrets.yaml,
#          heartbeat staleness incluso);
#       2. rilancio dei 2 processi locali: 01c_iv_poller (alimenta 04b con IV
#          fresca <=30 min) e 04b_vol_paper --execute (forward test testnet).
#          01d NON viene lanciato: il recorder L2 vive sul VPS dal 2026-07-14.
#     Anti-duplicazione: se un processo e' gia' vivo NON viene rilanciato
#     (due 04b scriverebbero position/trades in conflitto).
#     NOTA encoding: file deliberatamente ASCII-only - PS 5.1 legge i .ps1
#     senza BOM come cp1252 e i caratteri unicode corrompono il parsing.
#     Uso: .\avvio_sessione.ps1 [-Days 7] [-SkipPull]  (da root progetto;
#     il doppio click NON esegue i .ps1 - usa "Esegui con PowerShell" o terminale).
# EN: SESSION STARTUP (home side) - a single command at PC power-on:
#       1. pull+merge of the data the VPS collected while the PC was off
#          (scripts/vps/pull_vps_data.ps1: private host from config/secrets.yaml,
#          staleness heartbeat included);
#       2. relaunch of the 2 local processes: 01c_iv_poller (feeds 04b with
#          <=30 min fresh IV) and 04b_vol_paper --execute (testnet forward test).
#          01d is NOT launched: the L2 recorder lives on the VPS since 2026-07-14.
#     Anti-duplication: an already-alive process is NOT relaunched (two 04b
#     would write conflicting position/trades).
#     Encoding note: deliberately ASCII-only - PS 5.1 reads BOM-less .ps1 as
#     cp1252 and unicode characters corrupt parsing.
#     Usage: .\avvio_sessione.ps1 [-Days 7] [-SkipPull]  (from project root;
#     double-click does NOT run .ps1 - use "Run with PowerShell" or a terminal).
param(
    [int]$Days = 7,
    [switch]$SkipPull
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

# --- 2. Processi locali (anti-duplicazione) / local processes (anti-dup) -----
# IT: pattern -> argomenti di lancio. 01d/01e esclusi: girano sul VPS (systemd).
# EN: pattern -> launch arguments. 01d/01e excluded: they run on the VPS (systemd).
$targets = @(
    @{ pattern = "01c_iv_poller"; args = @("scripts/01c_iv_poller.py");              name = "01c poller IV" },
    @{ pattern = "04b_vol_paper"; args = @("scripts/04b_vol_paper.py", "--execute"); name = "04b vol-paper" }
)
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
foreach ($t in $targets) {
    $alive = @($procs | Where-Object { $_.CommandLine -match $t.pattern })
    if ($alive.Count -gt 0) {
        Write-Output "[sessione] $($t.name): GIA' ATTIVO (pid $($alive[0].ProcessId)) - non rilancio / already running, not relaunching"
    } else {
        Start-Process -WindowStyle Hidden -WorkingDirectory $ProjRoot -FilePath $Py -ArgumentList $t.args
        Write-Output "[sessione] $($t.name): AVVIATO / STARTED"
    }
}

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
# IT: NOTA quoting: solo apici SINGOLI nel codice python — PS 5.1 strippa le
#     doppie virgolette negli argomenti ai native exe (bug classico di quoting).
# EN: quoting NOTE: SINGLE quotes only in the python code — PS 5.1 strips double
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

Write-Output "[sessione] pronto / ready. Log vivi: logs\quantsys_*.log (piu' recenti per mtime)"
