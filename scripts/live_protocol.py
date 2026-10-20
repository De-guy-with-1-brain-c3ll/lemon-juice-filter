"""Opt-in live round trip against a running local adapter. Uses synthetic data only."""
import json
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, body):
    req = urllib.request.Request('http://127.0.0.1:13306' + path,
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=240) as r:
        return json.load(r)


def run():
    for wire, name, key in [('anthropic', 'Read', 'file_path'), ('openai', 'read', 'path')]:
        schema = {'type': 'object', 'properties': {key: {'type': 'string'}}, 'required': [key]}
        tool = {'name': name, 'description': 'Read a local file', 'input_schema': schema} if wire == 'anthropic' else {
            'type': 'function', 'function': {'name': name, 'description': 'Read a local file', 'parameters': schema}}
        messages = [{'role': 'user', 'content': 'Read fixture.txt using the available tool, then report its exact contents.'}]
        body = {'model': 'lemon-hybrid', 'max_tokens': 512, 'messages': messages, 'tools': [tool]}
        path = '/v1/messages' if wire == 'anthropic' else '/v1/chat/completions'
        first = post(path, body)
        if wire == 'anthropic':
            assert first['stop_reason'] == 'tool_use', first
            call = next(c for c in first['content'] if c['type'] == 'tool_use')
            assert call['name'] == name and call['input'][key] == 'fixture.txt', call
            messages += [{'role': 'assistant', 'content': first['content']},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': call['id'], 'content': 'ADAPTER_VERIFIED_81347'}]}]
        else:
            item = first['choices'][0]
            assert item['finish_reason'] == 'tool_calls', first
            call = item['message']['tool_calls'][0]
            assert call['function']['name'] == name and json.loads(call['function']['arguments'])[key] == 'fixture.txt', call
            messages += [item['message'], {'role': 'tool', 'tool_call_id': call['id'], 'content': 'ADAPTER_VERIFIED_81347'}]
        final = post(path, body)
        text = '\n'.join(c.get('text', '') for c in final.get('content', [])) if wire == 'anthropic' else final['choices'][0]['message']['content']
        assert 'ADAPTER_VERIFIED_81347' in text, final
        print(wire + ': tool action and result continuation passed', flush=True)


if __name__ == '__main__':
    run()
