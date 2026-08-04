Param(
    [string]$RepoRoot = $PSScriptRoot
)

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Missing venv python at $venvPython"
    exit 1
}

Set-Location $RepoRoot
& $venvPython "$RepoRoot\spad23\spad23_trpl_viewer_v2.py" @args
