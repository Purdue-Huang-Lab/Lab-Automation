Param(
    [string]$RepoRoot = $PSScriptRoot,
    [string]$DataFolder = ""
)

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Missing venv python at $venvPython"
    exit 1
}

Set-Location $RepoRoot
if ($DataFolder -ne "") {
    & $venvPython -m measurements.ta_viewer $DataFolder @args
} else {
    & $venvPython -m measurements.ta_viewer @args
}
