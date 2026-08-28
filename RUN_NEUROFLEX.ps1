$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m uv sync --dev --extra pose
}

& ".venv\Scripts\python.exe" -m neuroflex
