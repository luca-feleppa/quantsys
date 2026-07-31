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
#          [-PromoteMacro]  <- ripunta il canonico macro del VPS al vintage locale
#                              (atto DELIBERATO: senza il flag il canonico non si
#                               tocca mai, vedi blocco 0)
# EN: PULL DATA FROM THE COLLECTOR VPS (home side, Windows PowerShell 5.1+).
#     Downloads the 24/7 collector files into data/vps_staging/ via scp:
#       - append-only single files: atm_30h.parquet, dvol.parquet
#       - recent daily parquet (last -Days days): iv/chain + orderbook + deribit_trades
#     Then runs the merge into the canonical copy (scripts/vps/merge_vps_data.py).
#     Prerequisite: Windows OpenSSH client (ssh/scp) + authorized key on the VPS.
#     PRIVATE host/root: defaults from config/secrets.yaml (gitignored), block:
#       vps: { host: <user@ip>, remote_root: /opt/quantsys }
#     Usage: .\scripts\vps\pull_vps_data.ps1 [-VpsHost user@ip] [-Days 7] [-NoMerge]
#            [-PromoteMacro]  <- repoint the VPS macro canonical to the local
#                                vintage (DELIBERATE act: without the flag the
#                                canonical is never touched, see block 0)
param(
    [string]$VpsHost = "",
    [int]$Days = 7,
    [string]$RemoteRoot = "",
    [switch]$NoMerge,
    [switch]$PromoteMacro
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

# IT: 0) PUSH macro casa -> VPS, A VINTAGE DATATI (riscritto 2026-07-31; prima
#     sovrascriveva il canonico a OGNI pull, incondizionatamente). Il VPS ospita
#     04b, che al bootstrap notturno legge lo snapshot macro e lo CONGELA per
#     tutto il giorno: sovrascrivere il canonico durante un campione forward
#     pre-registrato ne cambia l'input silenziosamente (successo 2026-07-31,
#     breakpoint datato in STATUS.md). Disegno in tre parti:
#       a) l'archivio e' APPEND-ONLY: `data/macro/macro_features_<vintage>.parquet`,
#          dove <vintage> = ultima data dell'indice. 716 KB a copia: la retention
#          costa nulla e rende ogni decisione forward riconducibile al suo
#          vintage (oggi il file vecchio e' irrecuperabile: non e' in git);
#       b) il canonico `data/macro_features.parquet` diventa un SYMLINK
#          all'archivio, quindi il vintage live e' leggibile con un `readlink`
#          e visibile in un `ls -l`: nessun marker che possa mentire, la
#          verita' e' il puntatore stesso;
#       c) ripuntarlo richiede -PromoteMacro: il push smette di essere una
#          decisione, promuovere lo diventa. Senza il flag un vintage divergente
#          e' solo un WARNING e il live resta su quello che gia' usava.
#     Fail-soft in ogni ramo: la macro non blocca mai il pull dei collector
#     (04b warna da solo a 7g di staleness).
# EN: 0) PUSH home macro -> VPS, WITH DATED VINTAGES (rewritten 2026-07-31;
#     previously it overwrote the canonical on EVERY pull, unconditionally). The
#     VPS hosts 04b, which reads the macro snapshot at its nightly bootstrap and
#     FREEZES it for the day: overwriting the canonical inside a pre-registered
#     forward sample silently changes its input (happened 2026-07-31, breakpoint
#     dated in STATUS.md). Three-part design:
#       a) the archive is APPEND-ONLY: `data/macro/macro_features_<vintage>.parquet`,
#          <vintage> = last index date. 716 KB per copy: retention costs nothing
#          and makes every forward decision traceable to its vintage (today the
#          old file is unrecoverable: it is not in git);
#       b) the canonical `data/macro_features.parquet` becomes a SYMLINK into the
#          archive, so the live vintage is readable with `readlink` and visible
#          in `ls -l` - no marker that could lie, the pointer is the truth;
#       c) repointing it requires -PromoteMacro: the push stops being a decision,
#          promoting becomes one. Without the flag a diverging vintage is only a
#          WARNING and the live path stays on what it was already using.
#     Fail-soft on every branch: macro never blocks the collector pull (04b warns
#     by itself at 7d staleness).
$macroLocal = Join-Path $ProjRoot "data\macro_features.parquet"
if (Test-Path $macroLocal) {
    # IT: vintage locale = ultima data dell'indice (helper: stdout pulito).
    # EN: local vintage = last index date (helper: clean stdout).
    $macroVintage = (& python (Join-Path $ProjRoot "scripts\vps\macro_vintage.py") $macroLocal | Select-Object -Last 1)
    if ($LASTEXITCODE -ne 0 -or -not ($macroVintage -match '^\d{8}$')) {
        Write-Warning "macro: vintage locale non determinabile - push SALTATO / local vintage undeterminable - push SKIPPED"
    } else {
        $macroArch = "$RemoteRoot/data/macro/macro_features_$macroVintage.parquet"
        ssh @SshOpts $VpsHost "mkdir -p $RemoteRoot/data/macro"
        if ($LASTEXITCODE -ne 0) { Write-Warning "macro: mkdir remoto fallito/remote mkdir failed" }
        # IT: scp su .tmp + mv remoto: 04b non deve mai leggere un file a meta'.
        #     Ri-pushare un vintage gia' archiviato riscrive contenuto identico.
        # EN: scp to .tmp + remote mv: 04b must never read a half-written file.
        #     Re-pushing an archived vintage rewrites identical content.
        scp -q @SshOpts $macroLocal "${VpsHost}:$macroArch.tmp"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "macro: scp archivio fallito/archive scp failed - il VPS resta sul vintage corrente / VPS keeps its current vintage"
        } else {
            ssh @SshOpts $VpsHost "mv $macroArch.tmp $macroArch"
            if ($LASTEXITCODE -ne 0) { Write-Warning "macro: mv remoto archivio fallito/archive remote mv failed" }
            else { Write-Output "[pull] macro: archiviato vintage $macroVintage (append-only) / vintage $macroVintage archived" }

            # IT: vintage LIVE = target del symlink canonico. Vuoto = canonico
            #     ancora file regolare (stato pre-migrazione): va tracciato con
            #     una promozione esplicita, che al primo giro e' un no-op di
            #     contenuto se i due vintage coincidono.
            # EN: LIVE vintage = canonical symlink target. Empty = canonical is
            #     still a regular file (pre-migration state): it must be tracked
            #     with an explicit promotion, a content no-op on the first run
            #     when the two vintages coincide.
            $linkTarget = (ssh @SshOpts $VpsHost "readlink $RemoteRoot/data/macro_features.parquet" | Select-Object -Last 1)
            $liveVintage = ""
            if ($linkTarget -and $linkTarget -match 'macro_features_(\d{8})\.parquet') { $liveVintage = $Matches[1] }
            $liveLabel = if ($liveVintage) { $liveVintage } else { "NON TRACCIATO (file regolare) / UNTRACKED (regular file)" }

            if ($liveVintage -eq $macroVintage) {
                Write-Output "[pull] macro: canonico VPS gia' al vintage $macroVintage - nessuna promozione / already at vintage $macroVintage - no promotion"
            } elseif ($PromoteMacro) {
                # IT: symlink RELATIVO (risolto dalla dir del link) + mv -T =
                #     ripuntamento atomico, anche sopra un file regolare.
                # EN: RELATIVE symlink (resolved from the link's dir) + mv -T =
                #     atomic repoint, even over a regular file.
                $canon = "$RemoteRoot/data/macro_features.parquet"
                ssh @SshOpts $VpsHost "ln -sf macro/macro_features_$macroVintage.parquet $canon.tmp && mv -T $canon.tmp $canon"
                if ($LASTEXITCODE -eq 0) {
                    Write-Output "[pull] macro: PROMOSSO  $liveLabel  ->  $macroVintage"
                    Write-Output "[pull] macro: il nuovo vintage entra in vigore al prossimo bootstrap 04b (00:30 UTC) - datalo in STATUS.md se un campione forward e' aperto / effective at the next 04b bootstrap (00:30 UTC) - date it in STATUS.md if a forward sample is open"
                } else { Write-Warning "macro: promozione fallita, canonico INVARIATO / promotion failed, canonical UNCHANGED" }
            } else {
                Write-Warning "macro: vintage divergente - canonico VPS $liveLabel, locale $macroVintage. Canonico NON toccato (il live resta sul suo vintage). Per promuovere: .\scripts\vps\pull_vps_data.ps1 -PromoteMacro / diverging vintage - VPS canonical NOT touched; promote with -PromoteMacro"
            }
        }
    }
}

# IT: 1) file singoli append-only (piccoli: si ricopiano interi ogni volta).
# EN: 1) append-only single files (small: re-copied whole every time).
scp -q @SshOpts "${VpsHost}:$RemoteRoot/data/iv/atm_30h.parquet" (Join-Path $Staging "iv\")
if ($LASTEXITCODE -ne 0) { throw "scp atm_30h.parquet fallito/failed" }
scp -q @SshOpts "${VpsHost}:$RemoteRoot/data/iv/dvol.parquet" (Join-Path $Staging "iv\")
if ($LASTEXITCODE -ne 0) { Write-Warning "dvol.parquet non copiato (puo' non esistere ancora) / not copied (may not exist yet)" }
# IT: C4 (2026-07-18): greeks ATM del poller --greeks — senza questa riga il VPS
#     accumula ma casa non vede la serie (nota 2026-07-16). Fail-soft: puo'
#     mancare nei primi tick post-attivazione.
# EN: C4 (2026-07-18): --greeks poller ATM greeks — without this line the VPS
#     accumulates but home never sees the series (2026-07-16 note). Fail-soft:
#     may be missing in the first post-activation ticks.
scp -q @SshOpts "${VpsHost}:$RemoteRoot/data/iv/atm_greeks.parquet" (Join-Path $Staging "iv\")
if ($LASTEXITCODE -ne 0) { Write-Warning "atm_greeks.parquet non copiato (puo' non esistere ancora) / not copied (may not exist yet)" }

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
