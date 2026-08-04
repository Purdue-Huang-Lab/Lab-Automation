Param(
    [string]$RepoRoot = $PSScriptRoot
)

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Missing venv python at $venvPython"
    exit 1
}

$guiScript = Join-Path $RepoRoot "measurements\pump_probe_gate_dep\gui_integrated.py"
Set-Location (Join-Path $RepoRoot "measurements\pump_probe_gate_dep")
& $venvPython $guiScript
Set-Location $RepoRoot
