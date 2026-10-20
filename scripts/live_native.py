"""Native reasoning/tool-result smoke test against a running filter; synthetic data only."""
import json
import os
import urllib.request

base = os.getenv('LEMON_FILTER_URL', 'http://127.0.0.1:13306').rstrip('/') + '/native/v1'
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(path, body=None):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
        headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=600) as response:
        return json.load(response)


model = call('/models')['data'][0]['id']
messages = [{'role': 'user', 'content': 'Read fixture.txt using read, then report its exact contents.'}]
body = {'model': model, 'messages': messages, 'max_tokens': 1024,
    'chat_template_kwargs': {'enable_thinking': True}, 'tools': [{'type': 'function', 'function': {
        'name': 'read', 'description': 'Read a file', 'parameters': {'type': 'object',
            'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]}
first = call('/chat/completions', body)['choices'][0]
assert first['finish_reason'] == 'tool_calls', first['finish_reason']
tool = first['message']['tool_calls'][0]
assert tool['function']['name'] == 'read'
assert json.loads(tool['function']['arguments']) == {'path': 'fixture.txt'}
assert not first['message'].get('content'), 'Tool markup leaked into assistant text'
messages += [first['message'], {'role': 'tool', 'tool_call_id': tool['id'], 'content': 'LEMON_NATIVE_VERIFIED_59371'}]
last = call('/chat/completions', body)['choices'][0]['message']
assert 'reasoning_content' in first['message'] or 'reasoning_content' in last, 'Reasoning channel missing'
assert not last.get('tool_calls'), 'Model repeated a successful read'
assert 'LEMON_NATIVE_VERIFIED_59371' in (last.get('content') or '')
print('Native tool call, canonical history, reasoning transport, and final answer: passed')
