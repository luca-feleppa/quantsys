# IT: CHECK SALUTE VPS (lato casa) - wrapper sottile: legge l'host privato da
#     config/secrets.yaml (blocco vps:, mai stampato) e lancia via ssh
#     deploy/vps/health_check.sh SUL server (servizi, freschezza dati, disco,
#     geo-block). Con -UpdateRepo fa prima git pull sul VPS (porta a bordo
#     eventuali fix del kit). Exit 0 = PASS, 1 = WARN, 2 = VPS irraggiungibile.
#     Uso: .\scripts\vps\check_vps.ps1 [-UpdateRepo]
#     NOTA encoding: ASCII-only (PS 5.1 legge i .ps1 senza BOM come cp1252).
# EN: VPS HEALTH CHECK (home side) - thin wrapper: reads the private host from
#     config/secrets.yaml (vps: block, never printed) and runs
#     deploy/vps/health_check.sh ON the server via ssh (services, data
#     freshness, disk, geo-block). With -UpdateRepo it git-pulls on the VPS
#     first (ships any kit fixes). Exit 0 = PASS, 1 = WARN, 2 = unreachable.
#     Usage: .\scripts\vps\check_vps.ps1 [-UpdateRepo]
#     Encoding note: ASCII-only (PS 5.1 reads BOM-less .ps1 as cp1252).
param(
    [switch]$UpdateRepo
)

$ErrorActionPreference = "Stop"
$ProjRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# IT: stesso mini-parser del blocco vps: usato da pull_vps_data.ps1 (valori
#     privati: mai stampati).
# EN: same vps: block mini-parser as pull_vps_data.ps1 (private values: never
#     printed).
$VpsHost = ""
$secretsPath = Join-Path $ProjRoot "config\secrets.yaml"
if (Test-Path $secretsPath) {
    $inVps = $false
    foreach ($line in (Get-Content $secretsPath)) {
        if ($line -match '^vps:\s*$') { $inVps = $true; continue }
        if ($inVps -and $line -match '^\S') { $inVps = $false }
        if ($inVps -and $line -match '^\s+host:\s*(\S+)' -and -not $VpsHost) { $VpsHost = $Matches[1] }
    }
}
if (-not $VpsHost) { throw "vps.host assente in config/secrets.yaml / missing from config/secrets.yaml" }

# IT: comando remoto: pull opzionale del repo + health check server-side.
# EN: remote command: optional repo pull + server-side health check.
$remote = "bash /opt/quantsys/deploy/vps/health_check.sh"
if ($UpdateRepo) {
    $remote = "cd /opt/quantsys; git pull --ff-only -q; " + $remote
}
ssh -o ConnectTimeout=15 -o BatchMode=yes $VpsHost $remote
$code = $LASTEXITCODE
if ($code -eq 255) {
    Write-Warning "VPS IRRAGGIUNGIBILE via ssh / VPS UNREACHABLE via ssh"
    exit 2
}
exit $code
