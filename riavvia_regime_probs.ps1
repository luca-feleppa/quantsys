# IT: SCRIPT TEMPORANEO ONE-SHOT (2026-07-15) - rilancia la rigenerazione di
#     data/regime_probs.parquet (01b --regime-only) in un terminale VISIBILE,
#     cosi' ogni retrain del walk-forward si vede in diretta.
#     Contesto: il full-rebuild costa ~3h (30 refit MLE expanding, O(t) ciascuno)
#     e i run in background sono gia' morti 2 volte con la chiusura della
#     sessione. Questo script va lanciato quando il PC e' libero e la finestra
#     va lasciata aperta fino alla fine.
#     DA ELIMINARE dopo l'uso: il sostituto definitivo e' la modalita'
#     incrementale (backlog B7 in STATUS.md) - minuti invece di ore.
#     NON tocca 01c/04b: rilanciali normalmente con .\avvio_sessione.ps1
#     (possono girare in parallelo: questo job e' CPU-only, zero CUDA).
#     NOTA encoding: file deliberatamente ASCII-only (PS 5.1 legge i .ps1
#     senza BOM come cp1252).
#     Uso: .\riavvia_regime_probs.ps1   (da root progetto, "Esegui con
#     PowerShell" o da terminale; il doppio click NON esegue i .ps1).
# EN: ONE-SHOT TEMPORARY SCRIPT (2026-07-15) - relaunches the regeneration of
#     data/regime_probs.parquet (01b --regime-only) in a VISIBLE terminal so
#     every walk-forward retrain streams live.
#     Context: the full rebuild costs ~3h (30 expanding MLE refits, O(t) each)
#     and background runs already died twice with the session closing. Launch
#     when the PC is free and keep the window open until it finishes.
#     DELETE after use: the definitive replacement is the incremental mode
#     (backlog B7 in STATUS.md) - minutes instead of hours.
#     Does NOT touch 01c/04b: relaunch them normally via .\avvio_sessione.ps1
#     (they can run in parallel: this job is CPU-only, zero CUDA).

$ErrorActionPreference = "Stop"
$ProjRoot = $PSScriptRoot
$Py = Join-Path $ProjRoot ".venv\Scripts\python.exe"
$Host.UI.RawUI.WindowTitle = "01b --regime-only (full rebuild ~3h) - NON CHIUDERE / DO NOT CLOSE"

if (-not (Test-Path $Py)) { throw "venv non trovato/not found: $Py" }
Set-Location $ProjRoot

# IT: anti-duplicazione - un secondo 01b concorrente scriverebbe lo stesso parquet.
# EN: anti-duplication - a second concurrent 01b would write the same parquet.
$alive = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -match "01b_download_macro" })
if ($alive.Count -gt 0) {
    Write-Warning "01b GIA' ATTIVO (pid $($alive[0].ProcessId)) - non rilancio / already running, not relaunching"
    Read-Host "Premi INVIO per chiudere / Press ENTER to close"
    exit 1
}

Write-Output "======================================================================"
Write-Output " REGIME PROBS - FULL REBUILD (walk-forward Markov-Switching 2019->)"
Write-Output " Durata attesa ~3h: ~30 retrain a costo crescente (il progresso"
Write-Output " 't=NNN/65511' NON e' lineare nel tempo - la seconda meta' delle"
Write-Output " barre costa ~3/4 del tempo). NON chiudere questa finestra."
Write-Output " Expected ~3h: ~30 retrains of growing cost (bar progress is NOT"
Write-Output " linear in time). Do NOT close this window."
Write-Output "======================================================================"
Write-Output ""

# IT: output UTF-8 dal processo python (banner unicode di 01b su console Windows).
# EN: UTF-8 output from the python process (01b unicode banner on Windows console).
$env:PYTHONIOENCODING = "utf-8"

# IT: esecuzione DIRETTA nel terminale corrente (niente redirect: il logging
#     python e' su stderr e in console si vede in diretta; copia persistente
#     gia' in logs\quantsys_*.log via rotating handler).
# EN: DIRECT execution in the current terminal (no redirect: python logging is
#     on stderr and streams live; persistent copy already in logs\quantsys_*.log).
& $Py scripts\01b_download_macro.py --regime-only
$code = $LASTEXITCODE

Write-Output ""
if ($code -eq 0) {
    Write-Output "[regime] COMPLETATO - verifica finale / COMPLETED - final check:"
    # IT: verifica esito: span (atteso fino a 2026-06-22) + distribuzione regimi
    #     (prior storico ~R0 42 / R1 18 / R2 40) + mtime del parquet.
    # EN: outcome check: span (expected up to 2026-06-22) + regime distribution
    #     (historical prior ~R0 42 / R1 18 / R2 40) + parquet mtime.
    & $Py -c "import pandas as pd; rp = pd.read_parquet('data/regime_probs.parquet'); print('  span :', rp.index.min(), '->', rp.index.max(), f'({len(rp)} righe/rows)'); post = rp[~rp['regime_burn_in']] if 'regime_burn_in' in rp else rp; print('  regimi/regimes:', (post['regime_dominant'].value_counts(normalize=True).sort_index()*100).round(1).to_dict())"
    Write-Output ""
    Write-Output "[regime] Ora puoi dirlo a Claude nella prossima sessione (STATUS.md,"
    Write-Output "         sessione 2026-07-15: resta il commit dell'esito) e poi"
    Write-Output "         ELIMINARE questo script / You can now tell Claude in the"
    Write-Output "         next session and then DELETE this script."
} else {
    Write-Warning "[regime] USCITO CON ERRORE (exit $code) - controlla l'output sopra e logs\quantsys_*.log / exited with error, check output above and logs"
}
Read-Host "Premi INVIO per chiudere / Press ENTER to close"
