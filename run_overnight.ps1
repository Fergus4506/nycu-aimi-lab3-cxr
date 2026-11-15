# Overnight queue (2026-11-12). Same recipe as A2/M3 (CE, no aug, lr 1e-4, patience 7); one change per run.
# Run alone in its own window: concurrent GPU/Python jobs once exhausted the paging file (WinError 1455).
# Finished runs (metrics.json present) are skipped, so re-running this script resumes the queue.
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
# D: is an HDD; read the 768 cache from the SSD copy (robocopy of data\cache\long768_npy, verified 5,909 files / same bytes)
if (Test-Path 'E:\lab3_cache\long768_npy\manifest.json') { $env:LAB3_CACHE_ROOT = 'E:\lab3_cache' }
$runs = @(
    @('A3_r18_768',   '--model', 'resnet18', '--size', '768', '--bs', '24', '--note', 'resolution768'),
    @('M6_cnx_in22k', '--model', 'convnext_tiny.fb_in22k_ft_in1k', '--size', '512', '--bs', '24', '--note', 'in22k-weights'),
    @('M7_effv2s',    '--model', 'tf_efficientnetv2_s.in21k_ft_in1k', '--size', '512', '--bs', '16', '--note', 'new-arch'),
    @('M8_swin_t',    '--model', 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', '--size', '512', '--bs', '16', '--note', 'new-arch'),
    # 2026-11-13: 512 seed runs dropped (768 beat 512 on OOF and public); 768 seeds + ResNet34 instead
    @('S2_A3_seed1',  '--model', 'resnet18', '--size', '768', '--bs', '24', '--seed', '1', '--note', 'seed'),
    @('S2_A3_seed2',  '--model', 'resnet18', '--size', '768', '--bs', '24', '--seed', '2', '--note', 'seed'),
    @('M9_r34_768',   '--model', 'resnet34', '--size', '768', '--bs', '24', '--note', 'resnet34@768'),
    # 2026-11-13 afternoon: regularization vs overfitting (train loss ~0.001). Compare each with A2 (R18@512, 0.866).
    @('R1_wd05',      '--model', 'resnet18', '--size', '512', '--bs', '24', '--wd', '0.05', '--note', 'weight-decay0.05'),
    @('R2_lr3e5',     '--model', 'resnet18', '--size', '512', '--bs', '24', '--lr', '3e-5', '--note', 'lr3e-5'),
    @('R3_mixup',     '--model', 'resnet18', '--size', '512', '--bs', '24', '--mixup', '0.2', '--note', 'mixup0.2'),
    @('R4_geo_p12',   '--model', 'resnet18', '--size', '512', '--bs', '24', '--aug', 'geo', '--patience', '12', '--epochs', '40', '--note', 'geo-aug+patience12'),
    # 2026-11-13 evening: D2's geo+brightness may also have failed only because patience 7 stopped it early. Compare with R4.
    @('R5_geocolor_p12', '--model', 'resnet18', '--size', '512', '--bs', '24', '--aug', 'geo_color', '--patience', '12', '--epochs', '40', '--note', 'geo_color-aug+patience12'),
    # Converged recipe (R5) at 768; max epochs 50 because an R5 fold peaked at epoch 38 of 40. Compare with A3 seeds (0.8715-0.8756).
    @('R6_768_geocolor', '--model', 'resnet18', '--size', '768', '--bs', '24', '--aug', 'geo_color', '--patience', '12', '--epochs', '50', '--note', 'R5 recipe@768'),
    # Report control: patience 12 alone (no aug) separates "longer training" from "augmentation". Compare with A2 and R5.
    @('R7_p12_noaug', '--model', 'resnet18', '--size', '512', '--bs', '24', '--patience', '12', '--epochs', '40', '--note', 'patience12 only'),
    # 2026-11-14: SAM (rho 0.05) on the R5 recipe; compare with R5 (0.8745, b/v errors 548)
    @('R8_sam',       '--model', 'resnet18', '--size', '512', '--bs', '24', '--aug', 'geo_color', '--patience', '12', '--epochs', '40', '--sam', '0.05', '--note', 'R5 recipe+SAM0.05'),
    # 1024 with the plain A3 recipe (augmentation added nothing at 768); compare with A3 seeds 0.8715-0.8756. Needs E:\lab3_cache\long1024_npy.
    @('A4_r18_1024',  '--model', 'resnet18', '--size', '1024', '--bs', '24', '--note', 'resolution1024'),
    # SAM at the best resolution. No aug (it added nothing at 768 and made the CPU the bottleneck); patience 12 because SAM
    # converged later (R8 best epochs 7-35). Compare with A3 seeds 0.8715-0.8756 and R8 0.8883.
    @('R9_sam_768',   '--model', 'resnet18', '--size', '768', '--bs', '24', '--patience', '12', '--epochs', '40', '--sam', '0.05', '--note', 'SAM0.05@768'),
    # 2026-11-14 night: one change each on the R8 recipe (R18@512, geo_color, patience 12, SAM 0.05); compare with R8 0.8883
    @('R10_hflip',    '--model', 'resnet18', '--size', '512', '--bs', '24', '--aug', 'geo_color', '--patience', '12', '--epochs', '40', '--sam', '0.05', '--hflip', '--note', 'R8+hflip(+flip TTA)'),
    @('R11_norm',     '--model', 'resnet18', '--size', '512', '--bs', '24', '--aug', 'geo_color', '--patience', '12', '--epochs', '40', '--sam', '0.05', '--norm', '--note', 'R8+percentile norm'),
    @('R12_rho01',    '--model', 'resnet18', '--size', '512', '--bs', '24', '--aug', 'geo_color', '--patience', '12', '--epochs', '40', '--sam', '0.1', '--note', 'R8 with SAM rho0.1'),
    # FixRes-style: fine-tune each R8 fold model at 768 (same fold, no leakage), low lr, short, no aug/SAM. Compare with R8 0.8883.
    @('R13_fixres768', '--model', 'resnet18', '--size', '768', '--bs', '24', '--init', 'R8_sam', '--lr', '2e-5', '--epochs', '8', '--patience', '4', '--note', 'FixRes: R8 fine-tuned@768')
)
$MinRestMin = 2       # always rest at least this long (CPU heatsink settles in ~1-2 min)
$MaxRestMin = 20      # never rest longer than this
$GpuTempC = 50        # continue early once GPU temp <= this (idle is ~42 C) ...
$CpuLoadPct = 15      # ... and CPU load <= this. Windows exposes no reliable CPU temperature without admin tools.

function Cooldown {
    $start = Get-Date
    Write-Host "-- cooldown (min $MinRestMin, max $MaxRestMin min)  $(Get-Date -Format 'HH:mm')"
    Start-Sleep -Seconds ($MinRestMin * 60)
    while (((Get-Date) - $start).TotalMinutes -lt $MaxRestMin) {
        $t = [int](nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        $c = [int](Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average
        if ($t -le $GpuTempC -and $c -le $CpuLoadPct) { break }
        Write-Host "   GPU $t C, CPU load $c %: waiting"
        Start-Sleep -Seconds 60
    }
    $t = nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
    Write-Host ("-- rested {0:N0} min, GPU {1} C, continue" -f ((Get-Date) - $start).TotalMinutes, $t)
}

$trained = $false
foreach ($r in $runs) {
    $id, $rest = $r
    if (Test-Path "outputs\$id\metrics.json") { Write-Host "skip $id (done)"; continue }
    if ($trained) { Cooldown }
    $trained = $true
    Write-Host "== $id $($rest -join ' ')  $(Get-Date -Format 'MM-dd HH:mm')"
    & .\.venv\Scripts\python.exe -u train.py --run $id @rest 2>&1 | Tee-Object -FilePath "outputs\$id.log"
    if (Test-Path "outputs\$id\metrics.json") { Select-String oof_macro_f1 "outputs\$id\metrics.json" | ForEach-Object Line }
    else { Write-Host "!! $id failed, see outputs\$id.log" }
}
Write-Host "done $(Get-Date -Format 'MM-dd HH:mm')"
