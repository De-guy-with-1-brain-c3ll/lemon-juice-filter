$ErrorActionPreference = 'Stop'
function Read-LocalJson([string]$Uri) {
    $json = & curl.exe -s --noproxy '*' --max-time 5 $Uri
    if ($LASTEXITCODE -ne 0 -or -not $json) { throw "Cannot reach $Uri" }
    return $json | ConvertFrom-Json
}
$health = Read-LocalJson 'http://127.0.0.1:13306/health'
$config = Read-LocalJson 'http://127.0.0.1:8082/admin/api/config'
$thinking = $env:LEMON_CLAUDE_THINKING -eq '1'
$baseUrl = if ($thinking) { 'http://127.0.0.1:13306/native/v1' } else { 'http://127.0.0.1:13306/native/no-thinking/v1' }
$provider = $config.custom_providers | Where-Object {
    $_.base_url.TrimEnd('/') -eq $baseUrl -and
    $_.model_ids -contains $health.upstream_model
} | Select-Object -First 1
if (-not $provider) { throw 'Register the selected native endpoint with scripts/configure_fcc.py --native (add --no-thinking for the default profile).' }
$model = $provider.provider_id + '/' + $health.upstream_model
$cwd = (Get-Location).Path.Replace('\', '/')
$platform = if ($env:OS -eq 'Windows_NT') {
    'Windows paths use forward slashes such as C:/... . Commands run in GNU Bash (Git Bash), not cmd.exe or PowerShell. Use ls, find, and pwd; cmd.exe switches such as dir /b are invalid.'
} else { 'Use the paths and shell reported by the execution environment.' }
$prompt = @"
You are a coding agent with access to the supplied tools. Carry out the user's request yourself within the existing permissions.
Use Bash with pwd, ls, find, and grep to inspect directories and discover files. Use Read to inspect file contents, Write to create files, and Edit to change existing files. Read before editing. Quote paths containing spaces. Inspect discovered directories before guessing what they contain.
Resolve relative paths against the working directory below. Do not substitute the user home directory for the working directory.
Use tool results to continue the task. Correct failed actions when possible. Follow project instructions and respect permission denials. Report verified results or a blocker concisely.
Platform: $platform
Working directory: $cwd
"@
$previous = $env:CLAUDE_CODE_MAX_OUTPUT_TOKENS
$previousThinking = $env:MAX_THINKING_TOKENS
$tempFiles = @()
function New-TempJson([string]$Json) {
    $path = Join-Path ([IO.Path]::GetTempPath()) ('lemon-claude-' + [guid]::NewGuid().ToString('N') + '.json')
    Set-Content -LiteralPath $path -Value $Json -Encoding ASCII
    $script:tempFiles += $path
    return $path
}
# PowerShell 5.1 strips quotes from inline JSON arguments, so pass files instead.
$forward = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    $arg = [string]$args[$i]
    if (($arg -eq '--mcp-config' -or $arg -eq '--settings') -and $i + 1 -lt $args.Count -and
        ([string]$args[$i + 1]).TrimStart().StartsWith('{')) {
        $forward += $arg
        $forward += New-TempJson ([string]$args[$i + 1])
        $i++
    } else {
        $forward += $args[$i]
    }
}
try {
    $env:CLAUDE_CODE_MAX_OUTPUT_TOKENS = '4096'
    $env:MAX_THINKING_TOKENS = if ($thinking) { '2048' } else { '0' }
    $settingsFile = New-TempJson (@{alwaysThinkingEnabled=$thinking} | ConvertTo-Json -Compress)
    & fcc-claude --safe-mode --model $model --settings $settingsFile --system-prompt $prompt --tools 'Read,Write,Edit,Bash' @forward
    $result = $LASTEXITCODE
} finally {
    $env:CLAUDE_CODE_MAX_OUTPUT_TOKENS = $previous
    $env:MAX_THINKING_TOKENS = $previousThinking
    $tempFiles | ForEach-Object { Remove-Item -LiteralPath $_ -Force -ErrorAction SilentlyContinue }
}
exit $result
