"""Verify an early instruction survives a prompt larger than RyzenAI's old 2K limit."""
import json
import os
import urllib.request

base = os.getenv('LEMON_BASE_URL', 'http://127.0.0.1:13305/api/v1').rstrip('/')
model = os.environ['LEMON_MODEL']
marker = 'LEMON_CONTEXT_VERIFIED_71945'
body = {'model': model, 'messages': [
    {'role': 'system', 'content': 'The verification code is ' + marker + '.'},
    {'role': 'user', 'content': 'Ignore this padding: ' + 'padding ' * 2500 +
        '\nReturn only the verification code from the system instruction.'},
    {'role': 'system', 'content': '/no_think'}],
    'max_tokens': 64, 'temperature': .7, 'top_p': .8, 'top_k': 20, 'stream': False,
    'chat_template_kwargs': {'enable_thinking': False}}
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
request = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(),
    headers={'Content-Type': 'application/json'})
with opener.open(request, timeout=180) as response:
    result = json.load(response)
message = result['choices'][0]['message']
assert result['usage']['prompt_tokens'] > 2048, 'Test prompt was too short'
assert marker == (message.get('content') or '').strip(), 'Beginning of the prompt was not retained'
assert not (message.get('reasoning_content') or '').strip(), 'The no-thinking soft switch did not take effect'
print(f'Early instruction retained across {result["usage"]["prompt_tokens"]} prompt tokens; thinking disabled: passed')
