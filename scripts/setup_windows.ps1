param(
    [ValidateSet('cpu', 'cu128')][string]$TorchBackend = 'cpu'
)
$ErrorActionPreference = 'Stop'
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed with exit code $LASTEXITCODE" }
}
if ($env:OS -ne 'Windows_NT') { throw 'Run this script on Windows.' }
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw 'Conda is unavailable. Open Miniconda Prompt, run conda init powershell, then reopen PowerShell.'
}
# conda run targets the existing environment even if the prompt was not initialized.
Invoke-Checked -Program conda -Arguments @('run', '-n', 'orcalab', 'python', '-c', 'import sys; assert sys.version_info[:2] == (3,12), "orcalab requires Python 3.12; inspect conda list -n orcalab python before changing it"')
Invoke-Checked -Program conda -Arguments @('run', '-n', 'orcalab', 'python', '-m', 'pip', 'install', '--upgrade', 'pip')
$TorchIndex = "https://download.pytorch.org/whl/$TorchBackend"
Invoke-Checked -Program conda -Arguments @('run', '--no-capture-output', '-n', 'orcalab', 'python', '-m', 'pip', 'install', '--upgrade', 'torch==2.8.0', 'torchvision==0.23.0', '--index-url', $TorchIndex)
$RequirementPath = Join-Path (Split-Path $PSScriptRoot -Parent) 'requirement.txt'
Invoke-Checked -Program conda -Arguments @('run', '--no-capture-output', '-n', 'orcalab', 'python', '-m', 'pip', 'install', '-r', $RequirementPath)
Invoke-Checked -Program conda -Arguments @('run', '-n', 'orcalab', 'python', '-m', 'pip', 'check')
$ProjectPath = Split-Path $PSScriptRoot -Parent
Invoke-Checked -Program conda -Arguments @('run', '-n', 'orcalab', 'python', '-m', 'pip', 'install', '--no-deps', '-e', $ProjectPath)
$VerifyPath = Join-Path $PSScriptRoot 'verify_environment.py'
$VerifyArgs = @('run', '--no-capture-output', '-n', 'orcalab', 'python', $VerifyPath)
if ($TorchBackend -eq 'cu128') { $VerifyArgs += '--require-cuda' }
Invoke-Checked -Program conda -Arguments $VerifyArgs
Write-Host 'Installation checks passed. Run conda activate orcalab in an initialized PowerShell.'
