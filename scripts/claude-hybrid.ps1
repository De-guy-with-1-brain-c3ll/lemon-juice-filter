& (Join-Path $PSScriptRoot 'Start-Filter.ps1')
if (-not $?) { exit 1 }
$updates = @{
    ANTHROPIC_BASE_URL = 'http://127.0.0.1:13306'
    ANTHROPIC_API_KEY = 'local-only'
    ANTHROPIC_AUTH_TOKEN = 'local-only'
    ANTHROPIC_DEFAULT_SONNET_MODEL = 'lemon-hybrid'
    ANTHROPIC_DEFAULT_OPUS_MODEL = 'lemon-hybrid'
    ANTHROPIC_DEFAULT_HAIKU_MODEL = 'lemon-hybrid'
    CLAUDE_CODE_MAX_OUTPUT_TOKENS = '4096'
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1'
    CLAUDE_CODE_USE_BEDROCK = $null
    CLAUDE_CODE_USE_VERTEX = $null
    CLAUDE_CODE_USE_FOUNDRY = $null
}
$previous = @{}
foreach ($key in $updates.Keys) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
    [Environment]::SetEnvironmentVariable($key, $updates[$key], 'Process')
}
try {
    & claude --model lemon-hybrid --tools 'Read,Write,Edit,Bash' @args
    $result = $LASTEXITCODE
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}
exit $result
