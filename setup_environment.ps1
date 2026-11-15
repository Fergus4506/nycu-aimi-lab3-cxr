# Run from an ordinary PowerShell session with access to the official package sites.
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$ProjectPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ProjectPython)) {
    throw 'Project .venv is missing. Create it using Python 3.12 before continuing.'
}
Push-Location -LiteralPath $ProjectRoot
try {
    & $ProjectPython -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA package installation failed; no lockfile written.' }
    & $ProjectPython -m pip install -r requirements-bootstrap.txt
    if ($LASTEXITCODE -ne 0) { throw 'Additional package installation failed; no lockfile written.' }
    & $ProjectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed; no lockfile written.' }
    & $ProjectPython check_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment smoke test failed; no lockfile written.' }
    $Frozen = & $ProjectPython -m pip freeze
    if ($LASTEXITCODE -ne 0) { throw 'Unable to export installed versions.' }
    $LockText = "# Verified on Windows / Python 3.12; CUDA wheels require the official index.`n--extra-index-url https://download.pytorch.org/whl/cu128`n" + ($Frozen -join "`n") + "`n"
    [System.IO.File]::WriteAllText((Join-Path $ProjectRoot 'requirements.txt'), $LockText, [System.Text.UTF8Encoding]::new($false))
    Write-Host 'Environment verified. requirements.txt written.'
} finally {
    Pop-Location
}
