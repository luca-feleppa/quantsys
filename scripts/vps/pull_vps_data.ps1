# IT: PULL DATI DAL VPS COLLECTOR (lato casa, Windows PowerShell 5.1+).
#     Scarica via scp in data/vps_staging/ i file dei collector 24/7:
#       - file singoli append-only: atm_30h.parquet, dvol.parquet
#       - parquet giornalieri recenti (ultimi -Days giorni): iv/chain + orderbook + deribit_trades
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
#       - recent daily parquet (last -Days days): iv/chain + orderbook + deribit_trades
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

# IT: parser secrets + opzioni ssh anti-stallo condivisi (common.ps1, step 3
#     refactor 2026-07-16): i valori restano privati, mai stampati.
# EN: shared secrets parser + anti-stall ssh options (common.ps1, refactor
#     step 3, 2026-07-16): values stay private, never printed.
. (Join-Path $PSScriptRoot "common.ps1")
if (-not $VpsHost -or -not $RemoteRoot) {
    $vpsCfg = Get-VpsConfig -SecretsPath (Join-Path $ProjRoot "config\secrets.yaml")
    if (-not $VpsHost)    { $VpsHost    = $vpsCfg.Host }
    if (-not $RemoteRoot) { $RemoteRoot = $vpsCfg.RemoteRoot }
}
if (-not $VpsHost) { throw "VpsHost assente: passa -VpsHost o aggiungi il blocco vps: in config/secrets.yaml / missing: pass -VpsHost or add the vps: block to config/secrets.yaml" }
if (-not $RemoteRoot) { $RemoteRoot = "/opt/quantsys" }
$SshOpts = $VpsSshOpts

$Staging  = Join-Path $ProjRoot "data\vps_staging"
New-Item -ItemType Directory -Force (Join-Path $Staging "iv\chain")        | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Staging "orderbook")       | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Staging "deribit_trades")  | Out-Null
New-Item -ItemType Directory -Force (Join-Path $Staging "vol_paper")       | Out-Null

# IT: mai stampare l'host (privato): solo la finestra temporale.
# EN: never print the host (private): only the time window.
Write-Output "[pull] finestra/window: $Days giorni/days"

# IT: 0) PUSH macro casa -> VPS (dal 2026-07-18 il VPS ospita 04b, che legge lo
#     snapshot macro al bootstrap; il refresh macro resta lato casa). Atomico:
#     scp su .tmp + mv remoto (04b non puo' leggere un file a meta'). Fail-soft:
#     un push fallito non blocca il pull (04b warna da solo a 7g di staleness).
# EN: 0) PUSH home macro -> VPS (since 2026-07-18 the VPS hosts 04b, which reads
#     the macro snapshot at bootstrap; macro refresh stays home-side). Atomic:
#     scp to .tmp + remote mv (04b can never read a half-written file).
#     Fail-soft: a failed push does not block the pull (04b warns by itself at
#     7d staleness).
$macroLocal = Join-Path $ProjRoot "data\macro_features.parquet"
if (Test-Path $macroLocal) {
    scp -q @SshOpts $macroLocal "${VpsHost}:$RemoteRoot/data/macro_features.parquet.tmp"
    if ($LASTEXITCODE -eq 0) {
        ssh @SshOpts $VpsHost "mv $RemoteRoot/data/macro_features.parquet.tmp $RemoteRoot/data/macro_features.parquet"
        if ($LASTEXITCODE -eq 0) { Write-Output "[pull] push macro_features.parquet -> VPS OK" }
        else { Write-Warning "push macro: mv remoto fallito/remote mv failed" }
    } else { Write-Warning "push macro: scp fallito/failed - 04b VPS usera' lo snapshot precedente / VPS 04b keeps the previous snapshot" }
}

# IT: 1) file singoli append-only (piccoli: si ricopiano interi ogni volta).
# EN: 1) append-only single files (small: re-copied whole every time).
scp -q @SshOpts "${VpsHost}:$RemoteRoot/data/iv/atm_30h.parquet" (Join-Path $Staging "iv\")
if ($LASTEXITCODE -ne 0) { throw "scp atm_30h.parquet fallito/failed" }
scp -q @SshOpts "${VpsHost}:$RemoteRoot/data/iv/dvol.parquet" (Join-Path $Staging "iv\")
if ($LASTEXITCODE -ne 0) { Write-Warning "dvol.parquet non copiato (puo' non esistere ancora) / not copied (may not exist yet)" }

# IT: 2) giornalieri recenti — lista remota via find -mtime, poi scp per file
#        (niente rsync su Windows; i file sono pochi MB, il costo è trascurabile).
# EN: 2) recent dailies — remote list via find -mtime, then per-file scp
#        (no rsync on Windows; files are a few MB, cost is negligible).
$pairs = @(
    @{ remote = "$RemoteRoot/data/iv/chain";       local = (Join-Path $Staging "iv\chain") },
    @{ remote = "$RemoteRoot/data/orderbook";      local = (Join-Path $Staging "orderbook") },
    @{ remote = "$RemoteRoot/data/deribit_trades"; local = (Join-Path $Staging "deribit_trades") }
)
foreach ($p in $pairs) {
    $listCmd = "find $($p.remote) -name '*.parquet' -mtime -$Days 2>/dev/null"
    $files = ssh @SshOpts $VpsHost $listCmd
    if ($LASTEXITCODE -ne 0) { throw "ssh find fallito/failed su $($p.remote)" }
    $files = @($files | Where-Object { $_ -and $_.Trim() })
    Write-Output "[pull] $($p.remote): $($files.Count) file recenti/recent files"
    foreach ($f in $files) {
        scp -q @SshOpts "${VpsHost}:$f" $p.local
        if ($LASTEXITCODE -ne 0) { throw "scp fallito/failed: $f" }
    }
}

# IT: 2b) vol-paper (04b sul VPS dal 2026-07-18): log forward test + file di
#     stato. forecasts/trades = record del giudizio (fatali se mancanti);
#     position/hedge/exec_diag = opzionali (possono legittimamente non esistere:
#     book flat, hedge mai attivato). Il marker _pulled.ok autorizza il merge a
#     specchiare la PRESENZA di position/hedge_state (assente sul VPS = flat =
#     va rimosso anche in canonico).
# EN: 2b) vol-paper (04b on the VPS since 2026-07-18): forward-test log + state
#     files. forecasts/trades = judgment record (fatal if missing);
#     position/hedge/exec_diag = optional (may legitimately not exist: flat
#     book, hedge never enabled). The _pulled.ok marker authorizes the merge to
#     mirror the PRESENCE of position/hedge_state (absent on VPS = flat = must
#     be removed from the canonical copy too).
$vpStaging = Join-Path $Staging "vol_paper"
Remove-Item (Join-Path $vpStaging "*") -Force -ErrorAction SilentlyContinue
$vpRemote = "$RemoteRoot/results/vol_paper"
foreach ($f in @("forecasts.parquet", "trades.jsonl")) {
    scp -q @SshOpts "${VpsHost}:$vpRemote/$f" $vpStaging
    if ($LASTEXITCODE -ne 0) { throw "scp vol_paper/$f fallito/failed" }
}
# IT: NIENTE redirect 2>$null sui native exe: in PS 5.1 con EAP=Stop lo stderr
#     incapsulato come ErrorRecord diventerebbe terminante (gotcha noto del repo).
# EN: NO 2>$null redirect on native exes: under PS 5.1 with EAP=Stop the stderr
#     wrapped as an ErrorRecord would become terminating (known repo gotcha).
foreach ($f in @("position.json", "hedge_state.json", "hedge_ledger.jsonl", "exec_diag.jsonl")) {
    scp -q @SshOpts "${VpsHost}:$vpRemote/$f" $vpStaging
    if ($LASTEXITCODE -ne 0) { Write-Output "[pull] vol_paper/$f assente sul VPS (ok se flat/hedge off) / absent on VPS (ok when flat/hedge off)" }
}
New-Item -ItemType File -Force (Join-Path $vpStaging "_pulled.ok") | Out-Null
Write-Output "[pull] vol_paper: staging aggiornato / staging updated"

# IT: 3) merge nella copia canonica + heartbeat staleness (salvo -NoMerge).
#     Exit 2 del merge = merge OK ma heartbeat stale (warning gia' loggato,
#     NON e' un errore del pull); solo gli altri exit != 0 sono fatali.
# EN: 3) merge into the canonical copy + staleness heartbeat (unless -NoMerge).
#     Merge exit 2 = merge OK but stale heartbeat (warning already logged,
#     NOT a pull failure); only other non-zero exits are fatal.
if (-not $NoMerge) {
    Write-Output "[pull] merge nella copia canonica / merging into canonical copy..."
    python (Join-Path $ProjRoot "scripts\vps\merge_vps_data.py")
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 2) { throw "merge_vps_data.py fallito/failed (exit $LASTEXITCODE)" }
}
Write-Output "[pull] completato / done."
