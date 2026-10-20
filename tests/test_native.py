import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from lemon_juice_filter.server import app


def response(command=None, text=None):
    message = {'role': 'assistant', 'content': text, 'reasoning_content': 'synthetic thought'}
    if command is not None:
        message.update(content='<tool_call>duplicate markup</tool_call>', tool_calls=[
            {'id': 'fixture', 'type': 'function', 'function': {'name': 'bash', 'arguments': json.dumps({'command': command})}}])
    return httpx.Response(200, request=httpx.Request('POST', 'http://localhost'), json={
        'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'Qwen3-4B-Hybrid',
        'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}], 'usage': {}})


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.model_patch = patch('lemon_juice_filter.server.UPSTREAM_MODEL', 'Qwen3-4B-Hybrid')
        self.model_patch.start()
        self.context = TestClient(app)
        self.client = self.context.__enter__()
        self.body = {'model': 'Qwen3-4B-Hybrid', 'messages': [{'role': 'user', 'content': 'list files'}],
            'tools': [{'type': 'function', 'function': {'name': 'bash', 'parameters': {
                'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}]}

    def tearDown(self):
        self.context.__exit__(None, None, None)
        self.model_patch.stop()

    def test_native_thinking_and_calls_survive_without_duplicate_text(self):
        app.state.client.post = AsyncMock(return_value=response('ls'))
        r = self.client.post('/native/v1/chat/completions', json={**self.body, 'stream': True})
        self.assertIn('reasoning_content', r.text)
        self.assertIn('"finish_reason": "tool_calls"', r.text)
        self.assertNotIn('duplicate markup', r.text)
        self.assertIn('[DONE]', r.text)
        payload = app.state.client.post.call_args.kwargs['json']
        self.assertEqual((payload['temperature'], payload['top_p'], payload['top_k']), (0.6, 0.95, 20))
        self.assertGreaterEqual(app.state.client.post.call_args.kwargs['timeout'].read, 500)

    def test_history_drops_reasoning_and_redundant_call_content(self):
        app.state.client.post = AsyncMock(return_value=response(text='listed files'))
        historical = response('ls').json()['choices'][0]['message']
        messages = self.body['messages'] + [historical,
            {'role': 'tool', 'tool_call_id': 'fixture', 'content': 'file.txt'}]
        r = self.client.post('/native/v1/chat/completions', json={**self.body, 'messages': messages})
        self.assertEqual(r.status_code, 200)
        sent = app.state.client.post.call_args.kwargs['json']['messages']
        assistant = next(m for m in sent if m['role'] == 'assistant')
        self.assertNotIn('reasoning_content', assistant)
        self.assertEqual(assistant['content'].count('<tool_call>'), 1)
        self.assertNotIn('tool_calls', assistant)
        self.assertEqual(sent[-1], {'role': 'user', 'content': '<tool_response>\nfile.txt\n</tool_response>'})

    @unittest.skipUnless(shutil.which('bash') or Path('C:/Program Files/Git/bin/bash.exe').exists(), 'bash unavailable')
    def test_invalid_shell_is_repaired_without_executing_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'must-not-exist'
            command = 'touch "' + marker.as_posix() + '"'
            app.state.client.post = AsyncMock(side_effect=[response("ls 'unclosed"), response(command)])
            r = self.client.post('/native/v1/chat/completions', json=self.body)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(app.state.client.post.call_count, 2)
            self.assertFalse(marker.exists())

    def test_reasoning_only_response_cannot_be_reported_as_success(self):
        app.state.client.post = AsyncMock(return_value=response())
        self.assertEqual(self.client.post('/native/v1/chat/completions', json=self.body).status_code, 400)
        self.assertEqual(app.state.client.post.call_count, 2)

    def test_required_tool_choice_rejects_premature_final_answer(self):
        app.state.client.post = AsyncMock(return_value=response(text='I would run ls'))
        r = self.client.post('/native/v1/chat/completions', json={**self.body, 'tool_choice': 'required'})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(app.state.client.post.call_count, 2)

    def test_thinking_off_uses_non_thinking_sampling_defaults(self):
        app.state.client.post = AsyncMock(return_value=response(text='files'))
        r = self.client.post('/native/v1/chat/completions', json={**self.body,
            'chat_template_kwargs': {'enable_thinking': False}})
        self.assertEqual(r.status_code, 200)
        payload = app.state.client.post.call_args.kwargs['json']
        self.assertEqual((payload['temperature'], payload['top_p']), (0.7, 0.8))

    def test_invalid_token_limit_rejected_before_inference(self):
        app.state.client.post = AsyncMock()
        r = self.client.post('/native/v1/chat/completions', json={**self.body, 'max_tokens': '8192'})
        self.assertEqual(r.status_code, 400)
        app.state.client.post.assert_not_called()

    def test_native_guidance_preserves_client_instructions_and_input(self):
        body = {**self.body, 'messages': [{'role': 'system', 'content': 'Never modify files.'}] + self.body['messages']}
        original = json.dumps(body)
        app.state.client.post = AsyncMock(return_value=response(text='files'))
        self.assertEqual(self.client.post('/native/v1/chat/completions', json=body).status_code, 200)
        sent = app.state.client.post.call_args.kwargs['json']['messages']
        self.assertTrue(sent[0]['content'].startswith('Never modify files.'))
        self.assertEqual(json.dumps(body), original)

    def test_raw_tool_markup_is_validated_and_not_displayed_as_answer(self):
        raw = '<tool_call>{"name":"bash","arguments":{"command":"ls"}}</tool_call>'
        app.state.client.post = AsyncMock(return_value=response(text=raw))
        result = self.client.post('/native/v1/chat/completions', json=self.body).json()['choices'][0]
        self.assertEqual(result['finish_reason'], 'tool_calls')
        self.assertIsNone(result['message']['content'])
        self.assertEqual(result['message']['tool_calls'][0]['function']['name'], 'bash')
