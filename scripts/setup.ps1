<#
.SYNOPSIS
    One-shot setup: Python venv, PyTorch, Node modules, Stockfish and piece art.

.EXAMPLE
    .\scripts\setup.ps1
    .\scripts\setup.ps1 -Cpu          # install the CPU build of PyTorch
#>
[CmdletBinding()]
param(
    [switch]$Cpu,
    [string]$CudaIndex = 'https://download.pytorch.org/whl/cu130',
    [string]$PythonVersion = '3.12'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root 'backend'
$venvPython = Join-Path $backend '.venv\Scripts\python.exe'

Write-Host '== Python environment ==' -ForegroundColor Cyan
if (-not (Test-Path $venvPython)) {
    & py "-$PythonVersion" -m venv (Join-Path $backend '.venv')
}
& $venvPython -m pip install --quiet --upgrade pip setuptools wheel

Write-Host '== PyTorch ==' -ForegroundColor Cyan
if ($Cpu) {
    & $venvPython -m pip install --retries 10 torch
} else {
    # Install the pure-Python dependencies from PyPI first: the PyTorch index
    # occasionally 503s on those small wheels.
    & $venvPython -m pip install --retries 10 filelock typing-extensions sympy networkx jinja2 fsspec markupsafe mpmath
    & $venvPython -m pip install --retries 10 --no-deps torch --index-url $CudaIndex
}

Write-Host '== Backend requirements ==' -ForegroundColor Cyan
& $venvPython -m pip install --retries 10 -r (Join-Path $backend 'requirements.txt')

Write-Host '== Assets (Stockfish + pieces) ==' -ForegroundColor Cyan
& $venvPython (Join-Path $root 'scripts\download_assets.py')

Write-Host '== Frontend ==' -ForegroundColor Cyan
Push-Location (Join-Path $root 'frontend')
try { npm install } finally { Pop-Location }

Write-Host '== Check ==' -ForegroundColor Cyan
& $venvPython -c "import torch, chess; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| chess', chess.__version__)"

Write-Host ''
Write-Host 'Setup completo. Avvia con  .\scripts\dev.ps1' -ForegroundColor Green
