$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'Start-Filter.ps1')
if (-not $?) { exit 1 }

function Test-Fcc {
    try {
        $null = Invoke-RestMethod 'http://127.0.0.1:8082/health' -NoProxy -TimeoutSec 3
        return $true
    } catch { return $false }
}
if (-not (Test-Fcc)) {
    $desktop = Join-Path $env:USERPROFILE '.local\bin\fcc-desktop.exe'
    if (-not (Test-Path -LiteralPath $desktop)) { throw 'Install FCC or start its server on port 8082.' }
    Start-Process -FilePath $desktop -WindowStyle Hidden | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Fcc)) {
        if ((Get-Date) -gt $deadline) { throw 'FCC did not become ready on port 8082.' }
        Start-Sleep -Seconds 1
    }
}
$repo = Split-Path $PSScriptRoot -Parent
$providerFile = Join-Path $repo 'runtime\fcc-provider.json'
if (-not (Test-Path -LiteralPath $providerFile)) {
    $python = $env:LEMON_FILTER_PYTHON
    if (-not $python) { $python = Join-Path $env:APPDATA 'uv\tools\free-claude-code\Scripts\python.exe' }
    & $python (Join-Path $PSScriptRoot 'configure_fcc.py')
    if ($LASTEXITCODE -ne 0) { throw 'FCC provider configuration failed.' }
}
$provider = Get-Content -LiteralPath $providerFile -Raw | ConvertFrom-Json
& fcc-pi --provider free-claude-code --model $provider.model --thinking off @args
exit $LASTEXITCODE
