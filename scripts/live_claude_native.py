"""Verify native Claude Code directory discovery and a resumed user turn."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

repo = Path(__file__).resolve().parents[1]
runtime = repo / 'runtime'
runtime.mkdir(exist_ok=True)
session = str(uuid.uuid4())
launcher = repo / 'scripts' / 'claude-native.ps1'
marker = 'CLAUDE_NATIVE_VERIFIED_' + uuid.uuid4().hex[:12]


def run(root, turn, prompt):
    log = runtime / f'claude-native-{session}-turn-{turn}.jsonl'
    shell = 'pwsh' if shutil.which('pwsh') else 'powershell'
    command = [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(launcher),
        '--output-format', 'stream-json', '--verbose',
        '--max-turns', '8', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
        '--allowedTools', 'Read,Bash(ls *),Bash(find *),Bash(pwd),Bash(cat *)',
        '--session-id' if turn == 1 else '--resume', session, '-p', prompt]
    env = {**os.environ, 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1'}
    with log.open('w', encoding='utf-8') as output:
        result = subprocess.run(command, cwd=root, env=env, stdout=output,
            stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', timeout=900)
    events = [json.loads(line) for line in log.read_text(encoding='utf-8-sig').splitlines() if line.startswith('{')]
    end = next((e for e in reversed(events) if e.get('type') == 'result'), None)
    assert end is not None, result.stderr[-2000:]
    assert not end.get('is_error'), end.get('result', end)
    assert result.returncode == 0, result.stderr[-2000:]
    calls = [b['name'] for e in events if e.get('type') == 'assistant'
        for b in e.get('message', {}).get('content', []) if b.get('type') == 'tool_use']
    print(f'Turn {turn}: tools={calls}; result={end.get("result", "")}', flush=True)
    return calls, end.get('result', '')


with tempfile.TemporaryDirectory(prefix='lemon-claude-') as directory:
    root = Path(directory)
    source = root / 'Desktop' / 'VSCODE projects' / 'athena' / 'athena-source' / 'src'
    source.mkdir(parents=True)
    (source / 'index.ts').write_text(f'export const status = "{marker}";\n', encoding='utf-8')
    for index in range(30):
        (root / 'Desktop' / f'recording-{index:02d}.mp4').touch()
    calls, answer = run(root, 1,
        'Inspect the Desktop folder under the current working directory. Report the project folders you find. Stay within the current working directory.')
    assert calls and 'VSCODE projects' in answer, 'Desktop was not inspected'
    calls, answer = run(root, 2,
        'Find the VSCODE projects folder and the athena source folder in it. Read a source file to verify it, then report the source path and the exact status string.')
    assert calls and marker in answer and 'athena-source' in answer, 'Follow-up did not verify the source file'
print('Claude Code native discovery and resumed-user-turn verification: passed', flush=True)
