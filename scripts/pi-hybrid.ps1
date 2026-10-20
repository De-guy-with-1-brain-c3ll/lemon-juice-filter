& (Join-Path $PSScriptRoot 'Start-Filter.ps1')
if (-not $?) { exit 1 }
& pi --provider lemon-juice-filter --model lemon-hybrid --thinking off @args
exit $LASTEXITCODE
