<#
.SYNOPSIS
    Starts the API and the Vite dev server in two windows.

.EXAMPLE
    .\scripts\dev.ps1
    .\scripts\dev.ps1 -Port 8123
#>
[CmdletBinding()]
param(
    [int]$Port = 8077,
    [switch]$NoFrontend
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root 'backend'
$venvPython = Join-Path $backend '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    throw "Ambiente Python non trovato. Esegui prima .\scripts\setup.ps1"
}

$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    throw "La porta $Port è già occupata (pid $($busy.OwningProcess)). Usa -Port <altra>."
}

Write-Host "API      → http://127.0.0.1:$Port" -ForegroundColor Cyan
Start-Process -FilePath $venvPython `
    -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port", '--reload') `
    -WorkingDirectory $backend

if (-not $NoFrontend) {
    Write-Host 'Frontend → http://localhost:5173' -ForegroundColor Cyan
    Start-Process -FilePath 'cmd.exe' `
        -ArgumentList @('/c', 'npm', 'run', 'dev') `
        -WorkingDirectory (Join-Path $root 'frontend')
}
