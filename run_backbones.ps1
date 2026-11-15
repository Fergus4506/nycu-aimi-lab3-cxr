# M1-M4 backbone queue at 512px (same settings as A2, only the model changes).
# Run alone: a concurrent predict.py once exhausted the Windows paging file (WinError 1455).
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$runs = @(
    @('M2', 'densenet121', 24),
    @('M1', 'resnet50', 24),
    @('M3', 'convnext_tiny', 24),
    @('M4', 'efficientnet_b3', 16)  # bs 24 peaks at 11 GB
)
foreach ($r in $runs) {
    $id, $model, $bs = $r
    if (Test-Path "outputs\$id\metrics.json") { Write-Host "skip $id (done)"; continue }
    Write-Host "== $id $model bs$bs $(Get-Date -Format HH:mm)"
    & .\.venv\Scripts\python.exe -u train.py --run $id --model $model --size 512 --bs $bs --note 'backbone@512' 2>&1 | Tee-Object -FilePath "outputs\$id.log"
    if (Test-Path "outputs\$id\metrics.json") { Select-String oof_macro_f1 "outputs\$id\metrics.json" | ForEach-Object Line }
    else { Get-Content "outputs\$id.log" -Tail 5 }
}
Write-Host "done $(Get-Date -Format HH:mm)"
