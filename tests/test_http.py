import json
import unittest
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from lemon_juice_filter.server import app


def upstream(text):
    return httpx.Response(200, json={'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 80, 'completion_tokens': 20, 'total_tokens': 100}},
        request=httpx.Request('POST', 'http://localhost/chat/completions'))


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.ctx = TestClient(app)
        self.client = self.ctx.__enter__()
        self.body = {'model': 'lemon-hybrid', 'max_tokens': 256, 'messages': [{'role': 'user', 'content': 'read x'}],
            'tools': [{'name': 'Read', 'input_schema': {'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']}}]}

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_claude_native_tool_and_result_round_trip(self):
        mock = AsyncMock(side_effect=[upstream('{"name":"Read","arguments":{"file_path":"x"}}'), upstream('{"final":"actual contents"}')])
        app.state.client.post = mock
        first = self.client.post('/v1/messages', json=self.body).json()
        block = first['content'][0]
        self.assertEqual(first['stop_reason'], 'tool_use')
        second = {**self.body, 'messages': self.body['messages'] + [
            {'role': 'assistant', 'content': [block]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': block['id'], 'content': 'actual contents'}]}]}
        response = self.client.post('/v1/messages', json=second).json()
        self.assertEqual(response['content'][0]['text'], 'actual contents')
        submitted = mock.call_args.kwargs['json']
        self.assertNotIn('tools', submitted)
        self.assertIn('actual contents', submitted['messages'][-1]['content'])

    def test_repair_once_and_never_emit_invalid_call(self):
        mock = AsyncMock(side_effect=[upstream('{"name":"Read","arguments":{"wrong":"x"}}'), upstream('{"name":"Read","arguments":{"file_path":"x"}}')])
        app.state.client.post = mock
        response = self.client.post('/v1/messages', json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock.call_count, 2)
        self.assertEqual(response.json()['content'][0]['input'], {'file_path': 'x'})

    def test_two_invalid_outputs_fail_closed(self):
        mock = AsyncMock(return_value=upstream('{"name":"undeclared","arguments":{}}'))
        app.state.client.post = mock
        response = self.client.post('/v1/messages', json=self.body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(mock.call_count, 2)
        self.assertNotIn('tool_use', response.text)

    def test_dependent_batch_is_repaired_before_any_action_is_emitted(self):
        call = {'name': 'Read', 'arguments': {'file_path': 'x'}}
        mock = AsyncMock(side_effect=[upstream(json.dumps({'tool_calls': [call, call]})), upstream(json.dumps(call))])
        app.state.client.post = mock
        response = self.client.post('/v1/messages', json=self.body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()['content']), 1)
        self.assertEqual(mock.call_count, 2)

    def test_stream_uses_input_json_and_terminal_tool_reason(self):
        app.state.client.post = AsyncMock(return_value=upstream('{"name":"Read","arguments":{"file_path":"x"}}'))
        response = self.client.post('/v1/messages?beta=true', json={**self.body, 'stream': True})
        self.assertIn('event: message_stop', response.text)
        self.assertIn('input_json_delta', response.text)
        self.assertIn('"stop_reason": "tool_use"', response.text)

    def test_fcc_openai_endpoint(self):
        app.state.client.post = AsyncMock(return_value=upstream('{"name":"Read","arguments":{"file_path":"x"}}'))
        body = {**self.body, 'stream': True, 'tools': [{'type': 'function', 'function': {'name': 'Read', 'parameters': self.body['tools'][0]['input_schema']}}]}
        response = self.client.post('/v1/chat/completions', json=body)
        self.assertIn('"finish_reason": "tool_calls"', response.text)
        self.assertIn('data: [DONE]', response.text)

    def test_unknown_model_and_browser_origin_rejected(self):
        self.assertEqual(self.client.post('/v1/messages', json={**self.body, 'model': 'cloud-model'}).status_code, 400)
        self.assertEqual(self.client.get('/health', headers={'Origin': 'https://example.com'}).status_code, 403)


if __name__ == '__main__':
    unittest.main()
