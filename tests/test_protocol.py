import json
import unittest

from lemon_juice_filter.protocol import Call, Decision, ProtocolError, Tool, normalize_request, parse_decision, schema_for_prompt, tool_prompt, tool_result_payload
from lemon_juice_filter.server import anthropic_events, anthropic_message, openai_events, openai_message

READ = Tool('read', '', {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False})
CLAUDE = Tool('Read', '', {'type': 'object', 'properties': {'file_path': {'type': 'string'}}, 'required': ['file_path']})


class ProtocolTests(unittest.TestCase):
    def test_observed_shell_space_escape_is_preserved(self):
        shell = Tool('Bash', '', {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']})
        raw = r'<tool_call>{"name":"Bash","arguments":{"command":"ls -R ./Desktop/VSCODE\ projects"}}</tool_call>'
        result = parse_decision(raw, [shell])
        self.assertEqual(result.calls[0].arguments['command'], r'ls -R ./Desktop/VSCODE\ projects')

    def test_native_argument_string_accepts_only_lossless_escape_correction(self):
        shell = Tool('Bash', '', {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']})
        raw = json.dumps({'tool_calls': [{'function': {'name': 'Bash',
            'arguments': r'{"command":"ls VSCODE\ projects"}'}}]})
        self.assertEqual(parse_decision(raw, [shell]).calls[0].arguments['command'], r'ls VSCODE\ projects')

    def test_escape_correction_preserves_valid_escaped_quotes_and_newlines(self):
        raw = r'{"name":"read","arguments":{"path":"VSCODE\ projects\\file\"name\n.txt"}}'
        result = parse_decision(raw, [READ])
        self.assertEqual(result.calls[0].arguments['path'], 'VSCODE\\ projects\\file"name\n.txt')

    def test_escape_correction_does_not_complete_truncated_objects_or_unicode(self):
        for raw in (r'{"name":"read","arguments":{"path":"VSCODE\ projects"',
                    r'{"name":"read","arguments":{"path":"C:\user"}}'):
            with self.assertRaises(ProtocolError):
                parse_decision(raw, [READ])

    def test_tool_inventory_precedes_protocol_and_uses_client_names(self):
        prompt = tool_prompt([READ, CLAUDE])
        inventory = prompt.split('TOOL PROTOCOL:', 1)[0]
        self.assertTrue(inventory.startswith('AVAILABLE TOOLS'))
        self.assertIn('- "read":', inventory)
        self.assertIn('- "Read":', inventory)
        self.assertIn('"path" (string)', inventory)
        self.assertIn('"file_path" (string)', inventory)
        self.assertNotIn('- "locate_folder":', inventory)

    def test_observed_final_key_is_answer_text(self):
        result = parse_decision('{"final_key":"50 files"}', [READ], require_envelope=True)
        self.assertEqual(result.text, '50 files')
        self.assertFalse(result.calls)
        with self.assertRaises(ProtocolError):
            parse_decision('{"final_key":"50 files"}', [READ], choice='required', require_envelope=True)

    def test_final_key_does_not_hide_operations_or_nontext(self):
        for value in ({'final_key': 'done', 'tool_calls': [{'name': 'undeclared', 'arguments': {}}]},
                      {'final_key': {'command': 'ls'}}, {'final_key': 'done', 'command': 'ls'}):
            with self.assertRaises(ProtocolError):
                parse_decision(json.dumps(value), [READ], require_envelope=True)

    def test_missing_name_and_invalid_function_fail_as_protocol_errors(self):
        for call in ({'next_action': 'answer', 'directory_path': 'fixture'}, {'function': None}, {'function': 'read'}):
            with self.assertRaisesRegex(ProtocolError, 'name string'):
                parse_decision(json.dumps({'tool_calls': [call]}), [READ], require_envelope=True)

    def test_claude_read_display_lines_are_separate_metadata(self):
        result = tool_result_payload('one', 'Read', '1\tbase=41\n2\t')
        self.assertEqual(result['content'], 'base=41\n')
        self.assertEqual(result['first_line_number'], 1)
        self.assertTrue(result['display_line_numbers_removed'])

    def test_numbered_file_contents_preserve_their_own_numbers(self):
        result = tool_result_payload('one', 'Read', '7\t1\tactual file contents\n8\tnext')
        self.assertEqual(result['content'], '1\tactual file contents\nnext')

    def test_other_tools_and_errors_are_not_reformatted(self):
        for name, error in [('read', False), ('Bash', False), ('Read', True)]:
            self.assertEqual(tool_result_payload('one', name, '1\tdata', error)['content'], '1\tdata')
        self.assertEqual(tool_result_payload('one', 'Read', '1\tdata\nplain text')['content'], '1\tdata\nplain text')

    def test_prompt_compaction_retains_schema_constraints(self):
        schema = {'type': 'object', 'required': ['path'], 'additionalProperties': False,
            'properties': {'path': {'type': 'string', 'minLength': 1, 'description': 'x' * 800,
                'examples': ['/path/to/placeholder'], 'title': 'Path'}}}
        original = json.dumps(schema)
        compact = schema_for_prompt(schema)
        self.assertEqual(compact['required'], ['path'])
        self.assertFalse(compact['additionalProperties'])
        self.assertEqual(compact['properties']['path']['minLength'], 1)
        self.assertEqual(len(compact['properties']['path']['description']), 160)
        self.assertNotIn('examples', compact['properties']['path'])
        self.assertEqual(json.dumps(schema), original)

    def test_tool_manual_is_compacted_without_renaming(self):
        prompt = tool_prompt([Tool('Read', 'Read files.\n\n' + 'manual ' * 1000, CLAUDE.schema)])
        self.assertIn('"name": "Read"', prompt)
        self.assertIn('file_path', prompt)
        self.assertNotIn('manual manual', prompt)

    def test_compaction_preserves_fields_named_like_annotations(self):
        schema = {'type': 'object', 'properties': {
            'title': {'type': 'string'}, 'description': {'type': 'string'},
            'examples': {'type': 'array', 'items': {'type': 'string'}}},
            '$defs': {'title': {'type': 'string'}},
            'enum': [{'title': 'literal data', 'description': 'y' * 300}]}
        compact = schema_for_prompt(schema)
        self.assertEqual(set(compact['properties']), {'title', 'description', 'examples'})
        self.assertIn('title', compact['$defs'])
        self.assertEqual(compact['enum'], schema['enum'])

    def test_observed_xml(self):
        call = parse_decision('```xml\n<response>{"name":"read","arguments":{"path":"x"}}</response>\n```', [READ]).calls[0]
        self.assertEqual((call.name, call.arguments), ('read', {'path': 'x'}))

    def test_observed_extra_backticks(self):
        self.assertEqual(parse_decision('```json\n{"name":"read","arguments":{"path":"x"}}\n``````', [READ]).calls[0].name, 'read')

    def test_observed_singular_wrapper_generates_fresh_id(self):
        call = parse_decision('{"tool_call":{"id":"model-invented","name":"read","arguments":{"path":"x"}}}', [READ]).calls[0]
        self.assertEqual(call.arguments, {'path': 'x'})
        self.assertNotEqual(call.id, 'model-invented')

    def test_claude_schema_and_case(self):
        self.assertEqual(parse_decision('{"name":"Read","arguments":{"file_path":"x"}}', [CLAUDE]).calls[0].arguments, {'file_path': 'x'})
        with self.assertRaises(ProtocolError):
            parse_decision('{"name":"read","arguments":{"path":"x"}}', [CLAUDE])

    def test_missing_required_argument(self):
        with self.assertRaises(ProtocolError):
            parse_decision('{"name":"read","arguments":{"file_path":"x"}}', [READ])

    def test_undeclared_tool(self):
        with self.assertRaises(ProtocolError):
            parse_decision('{"name":"Bash","arguments":{"command":"anything"}}', [READ])

    def test_json_string_arguments_and_parallel(self):
        value = json.dumps({'tool_calls': [{'function': {'name': 'read', 'arguments': '{"path":"a"}'}}, {'name': 'read', 'arguments': {'path': 'b'}}]})
        calls = parse_decision(value, [READ]).calls
        self.assertEqual(len({c.id for c in calls}), 2)
        with self.assertRaises(ProtocolError):
            parse_decision(value, [READ], parallel=False)

    def test_truncated_call(self):
        with self.assertRaises(ProtocolError):
            parse_decision('{"name":"read","arguments":{"path":', [READ])

    def test_plain_code_is_not_a_tool_call(self):
        text = 'Here is an example: {"name":"read","arguments":{"path":"x"}}'
        self.assertFalse(parse_decision(text, [READ]).calls)

    def test_strict_mode_repairs_narrated_operations(self):
        with self.assertRaises(ProtocolError):
            parse_decision('I would read x and write y.', [READ], require_envelope=True)

    def test_final_and_choices(self):
        self.assertEqual(parse_decision('{"final":"Done"}', [READ]).text, 'Done')
        with self.assertRaises(ProtocolError):
            parse_decision('{"final":"Done"}', [READ], 'required')
        with self.assertRaises(ProtocolError):
            parse_decision('{"name":"read","arguments":{"path":"x"}}', [READ], 'none')

    def test_observed_final_answer_pseudo_tool_is_plain_text(self):
        raw = '{"tool_calls":[{"name":"Final Answer","arguments":{"answer":"result=43"}}]}'
        decision = parse_decision(raw, [READ], require_envelope=True)
        self.assertEqual(decision.text, 'result=43')
        self.assertFalse(decision.calls)
        with self.assertRaises(ProtocolError):
            parse_decision(raw, [READ], 'required')

    def test_final_answer_cannot_hide_operational_arguments(self):
        with self.assertRaises(ProtocolError):
            parse_decision('{"name":"Final Answer","arguments":{"answer":"done","command":"anything"}}', [READ])

    def test_real_declared_final_answer_tool_is_not_rewritten(self):
        final_tool = Tool('Final Answer', '', {'type': 'object', 'properties': {'answer': {'type': 'string'}}, 'required': ['answer']})
        decision = parse_decision('{"name":"Final Answer","arguments":{"answer":"done"}}', [final_tool])
        self.assertEqual(decision.calls[0].name, 'Final Answer')
        self.assertEqual(decision.text, '')

    def test_anthropic_result_history_preserves_error_and_id(self):
        body = {'messages': [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'one', 'name': 'Read', 'input': {'file_path': 'x'}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'one', 'content': 'missing', 'is_error': True}]}]}
        messages, _ = normalize_request(body, 'anthropic')
        result = json.loads(messages[1]['content'])['tool_result']
        self.assertTrue(result['is_error'])
        self.assertEqual(result['id'], 'one')

    def test_openai_history(self):
        messages, _ = normalize_request({'messages': [{'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'one', 'function': {'name': 'read', 'arguments': '{"path":"x"}'}}]},
            {'role': 'tool', 'tool_call_id': 'one', 'content': 'contents'}]}, 'openai')
        self.assertEqual(json.loads(messages[1]['content'])['tool_result']['name'], 'read')

    def test_unmatched_results_fail(self):
        with self.assertRaises(ProtocolError):
            normalize_request({'messages': [{'role': 'tool', 'tool_call_id': 'unknown', 'content': 'x'}]}, 'openai')

    def test_images_are_not_silently_dropped(self):
        with self.assertRaises(ProtocolError):
            normalize_request({'messages': [{'role': 'user', 'content': [{'type': 'image', 'source': {}}]}]}, 'anthropic')

    def test_anthropic_stream_structure(self):
        decision = Decision(calls=[Call('read', {'path': 'x'})])
        message = anthropic_message(decision, {'prompt_tokens': 10, 'completion_tokens': 5}, 'stop')
        events = ''.join(anthropic_events(message))
        self.assertEqual(message['stop_reason'], 'tool_use')
        self.assertIn('input_json_delta', events)
        self.assertIn('message_stop', events)
        self.assertEqual(events.count('event: content_block_stop'), 1)

    def test_openai_stream_has_terminal_reason_and_done(self):
        message = openai_message(Decision(calls=[Call('Read', {'file_path': 'x'})]), {}, 'stop')
        events = ''.join(openai_events(message))
        self.assertIn('"finish_reason": "tool_calls"', events)
        self.assertTrue(events.endswith('data: [DONE]\n\n'))


if __name__ == '__main__':
    unittest.main()
