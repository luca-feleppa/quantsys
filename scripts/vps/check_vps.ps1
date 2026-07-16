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

# IT: parser secrets condiviso (common.ps1, step 3 refactor 2026-07-16); da qui
#     anche $VpsSshOpts: il check ora ha keepalive anti-stallo come il pull
#     (estensione deliberata del fix 2026-07-15, prima solo ConnectTimeout).
# EN: shared secrets parser (common.ps1, refactor step 3, 2026-07-16); it also
#     provides $VpsSshOpts: the check now gets the pull's anti-stall keepalive
#     (deliberate extension of the 2026-07-15 fix, previously ConnectTimeout only).
. (Join-Path $PSScriptRoot "common.ps1")
$VpsHost = (Get-VpsConfig -SecretsPath (Join-Path $ProjRoot "config\secrets.yaml")).Host
if (-not $VpsHost) { throw "vps.host assente in config/secrets.yaml / missing from config/secrets.yaml" }

# IT: comando remoto: pull opzionale del repo + health check server-side.
#     Il pull gira da root (la deploy key e' in /root/.ssh) su un repo di
#     proprieta' di quantsys -> serve safe.directory (config una-tantum) e il
#     chown post-pull ripristina l'ownership per i servizi.
# EN: remote command: optional repo pull + server-side health check. The pull
#     runs as root (deploy key lives in /root/.ssh) on a quantsys-owned repo ->
#     needs safe.directory (one-time config) and the post-pull chown restores
#     ownership for the services.
$remote = "bash /opt/quantsys/deploy/vps/health_check.sh"
if ($UpdateRepo) {
    $remote = "cd /opt/quantsys; git pull --ff-only -q; chown -R quantsys:quantsys /opt/quantsys; " + $remote
}
ssh @VpsSshOpts $VpsHost $remote
$code = $LASTEXITCODE
if ($code -eq 255) {
    Write-Warning "VPS IRRAGGIUNGIBILE via ssh / VPS UNREACHABLE via ssh"
    exit 2
}
exit $code
