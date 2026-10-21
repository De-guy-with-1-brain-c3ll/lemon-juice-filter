import json
import unittest
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from lemon_juice_filter.server import app, CONTEXT, UPSTREAM_MODEL


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

    def test_post_tool_completion_repair_allows_final_answer(self):
        mock = AsyncMock(side_effect=[upstream('{"tool_calls":[{"next_action":"answer","directory_path":"fixture"}]}'),
            upstream('{"final":"sample-00.txt through sample-49.txt"}')])
        app.state.client.post = mock
        body = {**self.body, 'messages': self.body['messages'] + [
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'fixture', 'name': 'Read', 'input': {'file_path': 'fixture'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'fixture', 'content': 'sample-00.txt through sample-49.txt'}]}]}
        response = self.client.post('/v1/messages', json=body)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['stop_reason'], 'end_turn')
        self.assertEqual(response.json()['content'][0]['text'], 'sample-00.txt through sample-49.txt')
        correction = mock.call_args.kwargs['json']['messages'][-1]['content']
        self.assertIn('top-level key "final"', correction)

    def test_followup_repair_repeats_request_and_schemas_without_bad_answer(self):
        invalid = '{"tool_calls":[{"next_action":"find","directory_path":"fixture"}]}'
        mock = AsyncMock(side_effect=[upstream(invalid), upstream('{"name":"Read","arguments":{"file_path":"fixture"}}')])
        app.state.client.post = mock
        body = {**self.body, 'messages': [{'role': 'user', 'content': 'list files'},
            {'role': 'assistant', 'content': 'fixture is a directory'},
            {'role': 'user', 'content': 'find the source inside fixture'}]}
        response = self.client.post('/v1/messages', json=body)
        self.assertEqual(response.status_code, 200)
        sent = mock.call_args.kwargs['json']['messages']
        self.assertFalse(any(m['content'] == invalid for m in sent))
        self.assertIn('find the source inside fixture', sent[-1]['content'])
        self.assertIn('"arguments_schema"', sent[-1]['content'])
        self.assertIn('"file_path"', sent[-1]['content'])

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

    def test_lost_weights_reloads_only_configured_model_then_retries(self):
        fault = httpx.Response(500, text='Weights not loaded in MatMulNBits_8_0_0',
            request=httpx.Request('POST', 'http://localhost/chat/completions'))
        ok = httpx.Response(200, json={'status': 'success'}, request=httpx.Request('POST', 'http://localhost/load'))
        mock = AsyncMock(side_effect=[fault, ok, ok, upstream('{"final":"recovered"}')])
        app.state.client.post = mock
        self.assertEqual(self.client.post('/v1/messages', json=self.body).status_code, 200)
        calls = mock.call_args_list
        self.assertTrue(calls[1].args[0].endswith('/unload'))
        self.assertEqual(calls[1].kwargs['json'], {'model_name': UPSTREAM_MODEL})
        self.assertEqual(calls[2].kwargs['json']['ctx_size'], CONTEXT)
        self.assertEqual(calls[3].kwargs, calls[0].kwargs)
        self.assertEqual(calls[3].args, calls[0].args)

    def test_generic_failure_and_cooldown_do_not_reload(self):
        import time
        for message in ('unrelated inference error', 'Weights not loaded in MatMulNBits_8_0_0'):
            app.state.last_weight_recovery = time.monotonic() if 'Weights' in message else float('-inf')
            app.state.client.post = AsyncMock(return_value=httpx.Response(500, text=message,
                request=httpx.Request('POST', 'http://localhost/chat/completions')))
            self.assertEqual(self.client.post('/v1/messages', json=self.body).status_code, 502)
            self.assertEqual(app.state.client.post.call_count, 1)


if __name__ == '__main__':
    unittest.main()
