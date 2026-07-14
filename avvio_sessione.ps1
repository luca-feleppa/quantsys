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
# IT: pattern -> argomenti di lancio. 01d escluso: gira sul VPS (systemd).
# EN: pattern -> launch arguments. 01d excluded: it runs on the VPS (systemd).
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

Write-Output "[sessione] pronto / ready. Log vivi: logs\quantsys_*.log (piu' recenti per mtime)"
