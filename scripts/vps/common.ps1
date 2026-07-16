# IT: HELPER CONDIVISI lato casa per gli script VPS (dot-source: `. common.ps1`).
#     Estratti 2026-07-16 (step 3 refactor): il mini-parser del blocco `vps:` di
#     config/secrets.yaml era duplicato in pull_vps_data.ps1 e check_vps.ps1.
#     I valori (host/remote_root) sono PRIVATI: mai stamparli in log/chat.
#     $VpsSshOpts = opzioni anti-stallo comuni (fix 2026-07-15: senza keepalive
#     una TCP morta blocca per sempre; BatchMode = niente prompt interattivi).
#     NOTA encoding: ASCII-only (PS 5.1 legge i .ps1 senza BOM come cp1252).
# EN: Home-side SHARED HELPERS for the VPS scripts (dot-source: `. common.ps1`).
#     Extracted 2026-07-16 (refactor step 3): the `vps:` block mini-parser for
#     config/secrets.yaml was duplicated in pull_vps_data.ps1 and check_vps.ps1.
#     Values (host/remote_root) are PRIVATE: never print them in logs/chat.
#     $VpsSshOpts = shared anti-stall options (2026-07-15 fix: without keepalive
#     a dead TCP hangs forever; BatchMode = no interactive prompts).
#     Encoding note: ASCII-only (PS 5.1 reads BOM-less .ps1 as cp1252).

$VpsSshOpts = @("-o","ConnectTimeout=10","-o","ServerAliveInterval=10","-o","ServerAliveCountMax=3","-o","BatchMode=yes")

function Get-VpsConfig {
    # IT: parser minimale del blocco `vps:` (niente modulo yaml in PS 5.1):
    #     estrae host: e remote_root: dalle righe indentate del blocco.
    #     Ritorna una hashtable @{ Host; RemoteRoot } (stringhe vuote se assenti).
    # EN: minimal `vps:` block parser (no yaml module in PS 5.1): pulls host:
    #     and remote_root: from the block's indented lines. Returns a hashtable
    #     @{ Host; RemoteRoot } (empty strings when missing).
    param([Parameter(Mandatory=$true)][string]$SecretsPath)
    $cfg = @{ Host = ""; RemoteRoot = "" }
    if (Test-Path $SecretsPath) {
        $inVps = $false
        foreach ($line in (Get-Content $SecretsPath)) {
            if ($line -match '^vps:\s*$') { $inVps = $true; continue }
            if ($inVps -and $line -match '^\S') { $inVps = $false }
            if ($inVps -and $line -match '^\s+host:\s*(\S+)' -and -not $cfg.Host) { $cfg.Host = $Matches[1] }
            if ($inVps -and $line -match '^\s+remote_root:\s*(\S+)' -and -not $cfg.RemoteRoot) { $cfg.RemoteRoot = $Matches[1] }
        }
    }
    return $cfg
}
