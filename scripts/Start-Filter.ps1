$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$runtime = Join-Path $repo 'runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null

function Read-LocalJson([string]$Uri) {
    try { return Invoke-RestMethod -Uri $Uri -TimeoutSec 3 -NoProxy } catch { return $null }
}

$ready = Read-LocalJson 'http://127.0.0.1:13306/health'
if (Get-Process flm -ErrorAction SilentlyContinue) {
    throw 'FLM is still running. Stop its server to free the NPU before using the Lemonade hybrid model.'
}

if (-not (Read-LocalJson 'http://127.0.0.1:13305/api/v1/health')) {
    $lemonade = Join-Path $env:LOCALAPPDATA 'lemonade_server\bin\LemonadeServer.exe'
    if (-not (Test-Path -LiteralPath $lemonade)) { throw 'Start your local Lemonade server on port 13305 first.' }
    Start-Process -FilePath $lemonade -ArgumentList '--silent' -WorkingDirectory (Split-Path $lemonade) -WindowStyle Hidden | Out-Null
}
$lemonDeadline = (Get-Date).AddSeconds(30)
while (-not (Read-LocalJson 'http://127.0.0.1:13305/api/v1/health')) {
    if ((Get-Date) -gt $lemonDeadline) { throw 'Lemonade did not become ready on port 13305.' }
    Start-Sleep -Seconds 1
}
if ($ready -and $ready.model -eq 'lemon-hybrid') { return }

$python = $env:LEMON_FILTER_PYTHON
if (-not $python) {
    $candidates = @((Join-Path $repo '.venv\Scripts\python.exe'),
        (Join-Path $env:APPDATA 'uv\tools\free-claude-code\Scripts\python.exe'))
    $python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $python) { throw 'Create .venv and pip install -e . first, or set LEMON_FILTER_PYTHON to a Python with this project dependencies.' }

$priorPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = (Join-Path $repo 'src')
    $process = Start-Process -FilePath $python -ArgumentList @('-m','lemon_juice_filter.server') -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runtime 'server.stdout.log') -RedirectStandardError (Join-Path $runtime 'server.stderr.log') -PassThru
    $process.Id | Set-Content -LiteralPath (Join-Path $runtime 'server.pid')
} finally { $env:PYTHONPATH = $priorPath }
$deadline = (Get-Date).AddSeconds(30)
do {
    if (Read-LocalJson 'http://127.0.0.1:13306/health') { return }
    Start-Sleep -Seconds 1
} while ((Get-Date) -lt $deadline)
throw 'Adapter did not start. See runtime/server.stderr.log.'
