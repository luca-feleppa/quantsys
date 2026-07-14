# IT: PULL DATI DAL VPS COLLECTOR (lato casa, Windows PowerShell 5.1+).
#     Scarica via scp in data/vps_staging/ i file dei collector 24/7:
#       - file singoli append-only: atm_30h.parquet, dvol.parquet
#       - parquet giornalieri recenti (ultimi -Days giorni): iv/chain + orderbook
#     Poi lancia il merge nella copia canonica (scripts/vps/merge_vps_data.py).
#     Prerequisito: OpenSSH client di Windows (ssh/scp) + chiave autorizzata sul VPS.
#     Host/root PRIVATI: default da config/secrets.yaml (gitignored), blocco:
#       vps:
#         host: <user@ip>
#         remote_root: /opt/quantsys
#     Uso: .\scripts\vps\pull_vps_data.ps1 [-VpsHost user@ip] [-Days 7] [-NoMerge]
# EN: PULL DATA FROM THE COLLECTOR VPS (home side, Windows PowerShell 5.1+).
#     Downloads the 24/7 collector files into data/vps_staging/ via scp:
#       - append-only single files: atm_30h.parquet, dvol.parquet
#       - recent daily parquet (last -Days days): iv/chain + orderbook
#     Then runs the merge into the canonical copy (scripts/vps/merge_vps_data.py).
#     Prerequisite: Windows OpenSSH client (ssh/scp) + authorized key on the VPS.
#     PRIVATE host/root: defaults from config/secrets.yaml (gitignored), block:
#       vps: { host: <user@ip>, remote_root: /opt/quantsys }
#     Usage: .\scripts\vps\pull_vps_data.ps1 [-VpsHost user@ip] [-Days 7] [-NoMerge]
param(
    [string]$VpsHost = "",
    [int]$Days = 7,
    [string]$RemoteRoot = "",
    [switch]$NoMerge
)

$ErrorActionPreference = "Stop"

# IT: root di progetto = due livelli sopra questo script (scripts/vps/ → root).
# EN: project root = two levels above this script (scripts/vps/ → root).
$ProjRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# IT: parser minimale del blocco `vps:` in secrets.yaml (niente modulo yaml in
#     PS 5.1): estrae `host:` e `remote_root:` dalle righe indentate del blocco.
#     I valori NON vengono mai stampati (restano privati anche nei log/chat).
# EN: minimal parser of the `vps:` block in secrets.yaml (no yaml module in
#     PS 5.1): pulls `host:` and `remote_root:` from the block's indented lines.
#     Values are NEVER printed (they stay private in logs/chat too).
if (-not $VpsHost -or -not $RemoteRoot) {
    $secretsPath = Join-Path $ProjRoot "config\secrets.yaml"
    if (Test-Path $secretsPath) {
        $inVps = $false
        foreach ($line in (Get-Content $secretsPath)) {
            if ($line -match '^vps:\s*$') { $inVps = $true; continue }
            if ($inVps -and $line -match '^\S') { $inVps = $false }
            if ($inVps -and $line -match '^\s+host:\s*(\S+)' -and -not $VpsHost) { $VpsHost = $Matches[1] }
            if ($inVps -and $line -match '^\s+remote_root:\s*(\S+)' -and -not $RemoteRoot) { $RemoteRoot = $Matches[1] }
        }
    }
}
if (-not $VpsHost) { throw "VpsHost assente: passa -VpsHost o aggiungi il blocco vps: in config/secrets.yaml / missing: pass -VpsHost or add the vps: block to config/secrets.yaml" }
if (-not $RemoteRoot) { $RemoteRoot = "/opt/quantsys" }
$Staging  = Join-Path $ProjRoot "data\vps_staging"
New-Item -ItemType Directory -Force (Join-Path $Staging "iv\chain")   | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Staging "orderbook")  | Out-Null

# IT: mai stampare l'host (privato): solo la finestra temporale.
# EN: never print the host (private): only the time window.
Write-Output "[pull] finestra/window: $Days giorni/days"

# IT: 1) file singoli append-only (piccoli: si ricopiano interi ogni volta).
# EN: 1) append-only single files (small: re-copied whole every time).
scp -q "${VpsHost}:$RemoteRoot/data/iv/atm_30h.parquet" (Join-Path $Staging "iv\")
if ($LASTEXITCODE -ne 0) { throw "scp atm_30h.parquet fallito/failed" }
scp -q "${VpsHost}:$RemoteRoot/data/iv/dvol.parquet" (Join-Path $Staging "iv\")
if ($LASTEXITCODE -ne 0) { Write-Warning "dvol.parquet non copiato (puo' non esistere ancora) / not copied (may not exist yet)" }

# IT: 2) giornalieri recenti — lista remota via find -mtime, poi scp per file
#        (niente rsync su Windows; i file sono pochi MB, il costo è trascurabile).
# EN: 2) recent dailies — remote list via find -mtime, then per-file scp
#        (no rsync on Windows; files are a few MB, cost is negligible).
$pairs = @(
    @{ remote = "$RemoteRoot/data/iv/chain";  local = (Join-Path $Staging "iv\chain") },
    @{ remote = "$RemoteRoot/data/orderbook"; local = (Join-Path $Staging "orderbook") }
)
foreach ($p in $pairs) {
    $listCmd = "find $($p.remote) -name '*.parquet' -mtime -$Days 2>/dev/null"
    $files = ssh $VpsHost $listCmd
    if ($LASTEXITCODE -ne 0) { throw "ssh find fallito/failed su $($p.remote)" }
    $files = @($files | Where-Object { $_ -and $_.Trim() })
    Write-Output "[pull] $($p.remote): $($files.Count) file recenti/recent files"
    foreach ($f in $files) {
        scp -q "${VpsHost}:$f" $p.local
        if ($LASTEXITCODE -ne 0) { throw "scp fallito/failed: $f" }
    }
}

# IT: 3) merge nella copia canonica + heartbeat staleness (salvo -NoMerge).
# EN: 3) merge into the canonical copy + staleness heartbeat (unless -NoMerge).
if (-not $NoMerge) {
    Write-Output "[pull] merge nella copia canonica / merging into canonical copy..."
    python (Join-Path $ProjRoot "scripts\vps\merge_vps_data.py")
    if ($LASTEXITCODE -ne 0) { throw "merge_vps_data.py fallito/failed" }
}
Write-Output "[pull] completato / done."
