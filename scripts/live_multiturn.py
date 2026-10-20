"""Exercise failed-tool recovery and a later user turn with real fixture files."""
import json
import os
from pathlib import Path
import tempfile
import urllib.request

base = os.getenv('LEMON_FILTER_URL', 'http://127.0.0.1:13306').rstrip('/') + '/native/v1'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
        headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=600) as response:
        return json.load(response)


model = request('/models')['data'][0]['id']
tools = [{'type': 'function', 'function': {'name': name, 'description': description,
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}
    for name, description in [('read', 'Read a text file.'), ('list', 'List a directory.')]]

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory).resolve()
    (root / 'project').mkdir()
    (root / 'project' / 'first.txt').write_text('RECOVERY_VERIFIED_81426', encoding='utf-8')
    (root / 'project' / 'second.txt').write_text('FOLLOWUP_VERIFIED_93715', encoding='utf-8')
    messages = [{'role': 'system', 'content': f'Working directory: {root.as_posix()}. Tools resolve relative paths here.'},
        {'role': 'user', 'content': 'Read project/first.txt and report its contents.'},
        {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'failed-read', 'type': 'function',
            'function': {'name': 'read', 'arguments': '{"path":"/mnt/c/missing/project/first.txt"}'}}]},
        {'role': 'tool', 'tool_call_id': 'failed-read', 'content': 'Error: No such file or directory', 'is_error': True}]
    body = {'model': model, 'messages': messages, 'tools': tools, 'max_tokens': 2048,
        'chat_template_kwargs': {'enable_thinking': True}}

    def finish(expected):
        calls = 0
        for _ in range(6):
            item = request('/chat/completions', body)['choices'][0]['message']
            messages.append(item)
            if not item.get('tool_calls'):
                assert calls, 'Model gave up without trying a recovery tool: ' + str(item.get('content'))
                assert expected in (item.get('content') or ''), 'Final answer did not use the observed file contents: ' + str(item.get('content'))
                return calls
            for call in item['tool_calls']:
                name = call['function']['name']
                args = json.loads(call['function']['arguments'])
                print(f'Fixture tool: {name} {args}', flush=True)
                target = (root / args['path']).resolve()
                assert target.is_relative_to(root), 'Model failed to correct the invalid fixture path'
                try:
                    output = target.read_text(encoding='utf-8') if name == 'read' else '\n'.join(p.name for p in target.iterdir())
                    error = False
                except OSError as exc:
                    output, error = str(exc), True
                messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': output, 'is_error': error})
                calls += 1
        raise AssertionError('Model did not finish within six steps')

    recovery_calls = finish('RECOVERY_VERIFIED_81426')
    print(f'Failed path recovered and verified with {recovery_calls} tool call(s)', flush=True)
    messages.append({'role': 'user', 'content': 'Now read second.txt in that same folder and report its contents.'})
    followup_calls = finish('FOLLOWUP_VERIFIED_93715')
    print(f'Later user turn resolved the earlier folder with {followup_calls} tool call(s)', flush=True)
