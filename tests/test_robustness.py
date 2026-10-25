import json
import unittest
from unittest.mock import AsyncMock

import httpx
from fastapi.testclient import TestClient

from lemon_juice_filter.native import unresolved_failures
from lemon_juice_filter.policy import (compact_observation, is_control_message,
    repeated_observation, turn_start)
from lemon_juice_filter.protocol import ProtocolError, Tool, parse_decision
from lemon_juice_filter.server import app, UPSTREAM_MODEL


def upstream(payload):
    return httpx.Response(200, json=payload,
        request=httpx.Request('POST', 'http://localhost/chat/completions'))


def completion(text):
    return upstream({'choices': [{'message': {'role': 'assistant', 'content': text},
        'finish_reason': 'stop'}], 'usage': {}})


def call(call_id, name, **arguments):
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.ctx = TestClient(app, raise_server_exceptions=False)
        self.client = self.ctx.__enter__()
        app.state.client.post = AsyncMock(return_value=completion('{"final":"ok"}'))
        self.messages = [{'role': 'user', 'content': 'read x'}]

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def test_malformed_json_is_a_client_error(self):
        for path in ('/v1/messages', '/v1/chat/completions', '/native/v1/chat/completions',
                     '/native/no-thinking/v1/chat/completions', '/v1/messages/count_tokens'):
            response = self.client.post(path, content=b'{"messages":',
                                        headers={'content-type': 'application/json'})
            self.assertEqual(response.status_code, 400, path)

    def test_non_object_body_is_a_client_error(self):
        for path in ('/v1/messages', '/v1/chat/completions', '/native/v1/chat/completions',
                     '/v1/messages/count_tokens'):
            self.assertEqual(self.client.post(path, json=[1, 2]).status_code, 400, path)

    def test_malformed_history_is_a_client_error(self):
        cases = [
            {'messages': {'role': 'user', 'content': 'x'}},
            {'messages': ['not a message']},
            {'messages': [{'role': 'user', 'content': [1]}]},
            {'messages': [{'role': 'user', 'content': 'x'},
                          {'role': 'assistant', 'tool_calls': [{'function': {'name': 'Read'}}]}]},
            {'messages': [{'role': 'user', 'content': 'x'},
                          {'role': 'assistant', 'tool_calls': [{'id': 'a', 'function': 'Read'}]}]},
            {'messages': self.messages, 'tools': {'Read': {}}},
            {'messages': [{'role': 'user', 'content': 'x'},
                          {'role': 'tool', 'tool_call_id': ['a'], 'content': 'out'}]},
            {'messages': [{'role': 'user', 'content': 'x'},
                          {'role': 'assistant', 'tool_calls': [call('a', 'Read', path='x')]},
                          {'role': 'tool', 'tool_call_id': 'a', 'content': 'out',
                           'tool_calls': [call('b', 'Read', path='y')]}]},
        ]
        for body in cases:
            response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid', **body})
            self.assertEqual(response.status_code, 400, body)

    def test_null_tool_parameters_use_the_default_schema(self):
        mock = AsyncMock(return_value=completion('{"final":"ok"}'))
        app.state.client.post = mock
        tools = [{'type': 'function', 'function': {'name': 'Bash', 'parameters': None}}]
        openai = self.client.post('/v1/chat/completions', json={'model': 'lemon-hybrid',
            'messages': self.messages, 'tools': tools})
        self.assertEqual(openai.status_code, 200)
        native = self.client.post('/native/v1/chat/completions', json={'model': UPSTREAM_MODEL,
            'messages': self.messages, 'tools': tools})
        self.assertEqual(native.status_code, 200)
        # The backend rejects a null schema even though the client may send one.
        sent = mock.call_args.kwargs['json']['tools']
        self.assertEqual(sent[0]['function']['parameters'], {'type': 'object'})

    def test_truncated_tool_output_is_not_retried(self):
        mock = AsyncMock(return_value=upstream({'choices': [{'message': {
            'role': 'assistant', 'content': '{"tool_calls":'}, 'finish_reason': 'length'}],
            'usage': {}}))
        app.state.client.post = mock
        response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
            'messages': self.messages, 'tools': [{'name': 'Read', 'input_schema': {'type': 'object'}}]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(mock.await_count, 1)

    def test_non_string_text_block_is_a_client_error(self):
        response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 5}]}]})
        self.assertEqual(response.status_code, 400)

    def test_null_anthropic_input_schema_is_rejected(self):
        response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
            'messages': self.messages, 'tools': [{'name': 'Bash', 'input_schema': None}]})
        self.assertEqual(response.status_code, 400)

    def test_malformed_upstream_payload_is_a_gateway_error(self):
        app.state.client.post = AsyncMock(return_value=upstream({}))
        for path in ('/v1/messages', '/native/v1/chat/completions'):
            body = {'model': 'lemon-hybrid', 'messages': self.messages}
            if path.startswith('/native'):
                body['model'] = UPSTREAM_MODEL
            self.assertEqual(self.client.post(path, json=body).status_code, 502, path)

    def test_tool_choice_none_accepts_a_plain_text_answer(self):
        app.state.client.post = AsyncMock(return_value=completion('plain answer'))
        tools = [{'name': 'Read', 'input_schema': {'type': 'object'}}]
        response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
            'messages': self.messages, 'tools': tools, 'tool_choice': {'type': 'none'}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['content'][0]['text'], 'plain answer')

    def test_null_max_tokens_uses_the_default(self):
        mock = AsyncMock(return_value=completion('{"final":"ok"}'))
        app.state.client.post = mock
        response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
            'max_tokens': None, 'messages': self.messages})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock.call_args.kwargs['json']['max_tokens'], 1024)

    def test_null_temperature_is_accepted_as_zero(self):
        mock = AsyncMock(return_value=completion('{"final":"ok"}'))
        app.state.client.post = mock
        response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
            'temperature': None, 'messages': self.messages})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock.call_args.kwargs['json']['temperature'], 0)

    def test_non_numeric_temperature_is_rejected(self):
        for path in ('/v1/messages', '/native/v1/chat/completions'):
            body = {'model': 'lemon-hybrid', 'temperature': 'hot', 'messages': self.messages}
            if path.startswith('/native'):
                body['model'] = UPSTREAM_MODEL
            self.assertEqual(self.client.post(path, json=body).status_code, 400, path)

    def test_streaming_upstream_failure_yields_an_error_event(self):
        app.state.client.post = AsyncMock(return_value=upstream({}))
        with self.client.stream('POST', '/v1/messages', json={'model': 'lemon-hybrid',
                'stream': True, 'messages': self.messages}) as response:
            self.assertEqual(response.status_code, 200)
            text = ''.join(response.iter_text())
        self.assertIn('event: error', text)

    def test_malformed_upstream_shapes_are_gateway_errors(self):
        app.state.client.post = AsyncMock(return_value=upstream(
            {'choices': [{'message': {'content': '{"final":"ok"}'}}], 'usage': 'bad'}))
        self.assertEqual(self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
            'messages': self.messages}).status_code, 200)
        for payload in ({'choices': [{'message': ['not', 'an', 'object']}], 'usage': {}},
                        {'choices': 'bad', 'usage': {}}):
            app.state.client.post = AsyncMock(return_value=upstream(payload))
            response = self.client.post('/v1/messages', json={'model': 'lemon-hybrid',
                'messages': self.messages})
            self.assertEqual(response.status_code, 502, payload)

    def test_browser_origin_is_rejected(self):
        response = self.client.get('/health', headers={'Origin': 'http://example.test'})
        self.assertEqual(response.status_code, 403)

    def test_models_reports_the_configured_upstream(self):
        model = self.client.get('/v1/models').json()['data'][0]
        self.assertIn(UPSTREAM_MODEL, model['display_name'])


class PolicyRobustnessTests(unittest.TestCase):
    def test_compact_observation_never_exceeds_its_limit(self):
        text = 'y' * 50000
        for limit in (0, 1, 2, 3, 10, 79, 100, 800, 2000, 2080, 12000):
            self.assertLessEqual(len(compact_observation(text, limit)), limit, limit)
        self.assertEqual(compact_observation(text, 0), '')
        self.assertEqual(compact_observation('short', 800), 'short')

    def test_blank_user_message_is_not_a_turn_boundary(self):
        messages = [{'role': 'user', 'content': 'find it'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Bash', command='ls')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'No such directory'},
            {'role': 'user', 'content': '   '}]
        self.assertEqual(turn_start(messages), 0)

        class Call:
            name, arguments = 'Bash', {'command': 'ls'}
        self.assertEqual(repeated_observation([Call()], messages), 'No such directory')

    def test_structured_content_is_not_a_control_message(self):
        self.assertFalse(is_control_message([{'type': 'text', 'text': 'hi'}]))
        self.assertTrue(is_control_message(None))
        self.assertTrue(is_control_message(' '))

    def test_unresolved_failures_ignores_unknown_tool_ids(self):
        messages = [{'role': 'user', 'content': 'x'},
            {'role': 'tool', 'tool_call_id': 'missing', 'content': 'output'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Bash', command='ls')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'Exit code 2'}]
        self.assertEqual(unresolved_failures(messages), {'Bash'})

    def test_malformed_tool_choice_is_rejected(self):
        tools = [Tool('Bash', '', {'type': 'object'})]
        for choice in ({'type': 'tool', 'function': 'Bash'}, {'type': 'tool'}):
            with self.assertRaises(ProtocolError):
                parse_decision('{"final":"x"}', tools, choice)


if __name__ == '__main__':
    unittest.main()
