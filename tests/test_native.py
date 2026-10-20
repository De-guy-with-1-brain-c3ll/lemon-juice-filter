import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient
from lemon_juice_filter.server import app
from lemon_juice_filter.native import unresolved_failures


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
        observation = next(m for m in reversed(sent) if m['content'].startswith('<tool_response>'))
        result = json.loads(observation['content'].removeprefix('<tool_response>\n').removesuffix('\n</tool_response>'))
        self.assertEqual(result, {'tool_call_id': 'fixture', 'name': 'bash', 'output': 'file.txt'})
        self.assertEqual(sent[-1]['role'], 'system')
        self.assertIn('not user input', sent[-1]['content'])

    def test_followup_preserves_failed_call_and_all_user_turns(self):
        historical = response('ls /missing').json()['choices'][0]['message']
        messages = [{'role': 'user', 'content': 'Remember folder alpha.'},
            {'role': 'assistant', 'content': 'Understood.', 'reasoning_content': 'private'},
            {'role': 'user', 'content': 'Inspect that folder.'}, historical,
            {'role': 'tool', 'tool_call_id': 'fixture', 'content': 'No such file or directory', 'is_error': True}]
        original = json.dumps(messages)
        app.state.client.post = AsyncMock(return_value=response('pwd'))
        r = self.client.post('/native/v1/chat/completions', json={**self.body, 'messages': messages})
        self.assertEqual(r.status_code, 200)
        sent = app.state.client.post.call_args.kwargs['json']['messages']
        self.assertEqual(sent[1]['content'], 'Remember folder alpha.')
        self.assertEqual(sent[3]['content'], 'Inspect that folder.')
        self.assertNotIn('reasoning_content', sent[2])
        observation = next(m for m in reversed(sent) if m['content'].startswith('<tool_response>'))
        result = json.loads(observation['content'][len('<tool_response>\n'):-len('\n</tool_response>')])
        self.assertEqual(result['name'], 'bash')
        self.assertTrue(result['is_error'])
        self.assertEqual(result['output'], 'No such file or directory')
        self.assertEqual(json.dumps(messages), original)

    def failed_history(self, **overrides):
        historical = response('ls /missing').json()['choices'][0]['message']
        return {**self.body, 'messages': self.body['messages'] + [historical,
            {'role': 'tool', 'tool_call_id': 'fixture', 'content': 'Command exited with code 2'}], **overrides}

    def test_premature_final_after_error_gets_one_reconsideration(self):
        app.state.client.post = AsyncMock(side_effect=[response(text='Try another path yourself.'), response('pwd')])
        r = self.client.post('/native/v1/chat/completions', json=self.failed_history())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(app.state.client.post.call_count, 2)
        sent = app.state.client.post.call_args.kwargs['json']['messages']
        self.assertIn('original task is still pending', sent[-1]['content'])
        self.assertEqual(sent[-1]['role'], 'system')
        self.assertNotIn('Try another path yourself.', json.dumps(sent))

    def test_recovery_is_bounded_and_allows_a_real_blocker(self):
        app.state.client.post = AsyncMock(return_value=response(text='Access denied; cannot proceed.'))
        r = self.client.post('/native/v1/chat/completions', json=self.failed_history())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(app.state.client.post.call_count, 2)
        self.assertIn('Access denied', r.json()['choices'][0]['message']['content'])

    def test_tool_choice_none_disables_recovery(self):
        app.state.client.post = AsyncMock(return_value=response(text='Path was missing.'))
        r = self.client.post('/native/v1/chat/completions', json=self.failed_history(tool_choice='none'))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(app.state.client.post.call_count, 1)

    def test_new_user_turn_does_not_retry_an_earlier_failure(self):
        body = self.failed_history()
        body['messages'].append({'role': 'user', 'content': 'Stop searching. Say hello.'})
        app.state.client.post = AsyncMock(return_value=response(text='Hello.'))
        self.assertEqual(self.client.post('/native/v1/chat/completions', json=body).status_code, 200)
        self.assertEqual(app.state.client.post.call_count, 1)

    def test_unrelated_diagnostic_does_not_clear_a_failed_read(self):
        messages = [{'role': 'user', 'content': 'Read project/first.txt.'},
            {'role': 'assistant', 'tool_calls': [{'id': 'r', 'function': {'name': 'read'}}]},
            {'role': 'tool', 'tool_call_id': 'r', 'content': 'Missing file', 'is_error': True},
            {'role': 'assistant', 'tool_calls': [{'id': 'l', 'function': {'name': 'list'}}]},
            {'role': 'tool', 'tool_call_id': 'l', 'content': 'project'}]
        self.assertEqual(unresolved_failures(messages), {'read'})
        messages += [{'role': 'assistant', 'tool_calls': [{'id': 'r2', 'function': {'name': 'read'}}]},
            {'role': 'tool', 'tool_call_id': 'r2', 'content': 'Actual file contents'}]
        self.assertEqual(unresolved_failures(messages), set())

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

    def test_protocol_repair_is_not_a_new_user_request(self):
        app.state.client.post = AsyncMock(side_effect=[response(), response(text='Hello.')])
        self.assertEqual(self.client.post('/native/v1/chat/completions', json=self.body).status_code, 200)
        sent = app.state.client.post.call_args.kwargs['json']['messages']
        self.assertEqual(sent[-1]['role'], 'system')
        users = [m['content'] for m in sent if m['role'] == 'user']
        self.assertEqual(users, ['list files'])

    def test_new_request_remains_last_message_without_observation_cue(self):
        historical = response('ls').json()['choices'][0]['message']
        messages = self.body['messages'] + [historical,
            {'role': 'tool', 'tool_call_id': 'fixture', 'content': 'project'},
            {'role': 'assistant', 'content': 'Found project.'},
            {'role': 'user', 'content': 'Find its source folder.'}]
        app.state.client.post = AsyncMock(return_value=response('ls project'))
        self.assertEqual(self.client.post('/native/v1/chat/completions', json={**self.body, 'messages': messages}).status_code, 200)
        sent = app.state.client.post.call_args.kwargs['json']['messages']
        self.assertEqual(sent[-1], messages[-1])
