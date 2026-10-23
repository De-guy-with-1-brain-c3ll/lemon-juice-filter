import json
import unittest

from lemon_juice_filter.policy import (claims_file_contents, compact_observation,
    is_control_message, needs_retry, promises_action, read_evidence,
    repeated_observation, repair_command, repetition_instruction, turn_start)


def call(call_id, name, **arguments):
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(arguments)}}


class PolicyTests(unittest.TestCase):
    def test_promises_action(self):
        self.assertTrue(promises_action('Next, I will inspect the src folder.'))
        self.assertTrue(promises_action('Let me read that file now.'))
        self.assertTrue(promises_action('The file is at a/b. I can read it and check. Let me do that now.'))
        self.assertTrue(promises_action("I'll try that path again."))
        self.assertFalse(promises_action('There are three files here.'))
        self.assertFalse(promises_action('Let me know if you need anything else.'))

    def test_claims_file_contents(self):
        self.assertTrue(claims_file_contents('The exact status string from the source file is "x".'))
        self.assertTrue(claims_file_contents('The file contains two exports.'))
        self.assertFalse(claims_file_contents('There are three files in this folder.'))

    def test_read_evidence_accepts_read_tool(self):
        messages = [{'role': 'user', 'content': 'read it'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Read', file_path='/x')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'contents'}]
        self.assertTrue(read_evidence(messages))

    def test_read_evidence_accepts_shell_reads(self):
        for command in ('cat x', 'head -20 "a b"', 'grep -n foo x', 'cd a && sed -n 1,5p x'):
            messages = [{'role': 'user', 'content': 'look'},
                {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Bash', command=command)]},
                {'role': 'tool', 'tool_call_id': 'a', 'content': 'contents'}]
            self.assertTrue(read_evidence(messages), command)

    def test_read_evidence_rejects_listing_only(self):
        messages = [{'role': 'user', 'content': 'find it'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Bash', command='ls -R')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'a b c'}]
        self.assertFalse(read_evidence(messages))

    def test_read_evidence_only_counts_the_current_turn(self):
        messages = [{'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Read', file_path='/x')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'contents'},
            {'role': 'user', 'content': 'second'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('b', 'Bash', command='ls')]},
            {'role': 'tool', 'tool_call_id': 'b', 'content': 'names'}]
        self.assertFalse(read_evidence(messages))

    def test_needs_retry(self):
        tools = [object()]
        claimed = [{'role': 'user', 'content': 'verify'}]
        self.assertEqual(needs_retry('I will read the file.', claimed, tools), 'action')
        self.assertEqual(needs_retry('The file contains "x".', claimed, tools), 'evidence')
        self.assertIsNone(needs_retry('There are three files.', claimed, tools))
        self.assertIsNone(needs_retry('The file contains "x".', claimed, []))

    def test_repair_command_quotes_unquoted_cd_paths(self):
        self.assertEqual(repair_command('cd Desktop/VSCODE projects && ls -la'),
            "cd 'Desktop/VSCODE projects' && ls -la")
        self.assertEqual(repair_command('pwd && cd "a b" && ls'), 'pwd && cd "a b" && ls')
        self.assertEqual(repair_command('cd src'), 'cd src')
        self.assertEqual(repair_command('cd Desktop/VSCODE\\ projects && ls'),
            'cd Desktop/VSCODE\\ projects && ls')
        self.assertEqual(repair_command('ls -la Desktop/VSCODE projects'),
            'ls -la Desktop/VSCODE projects')

    def test_repeated_call_returns_its_observed_result(self):
        messages = [{'role': 'user', 'content': 'find it'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Bash', command='ls')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'No such directory'}]

        class Call:
            name, arguments = 'Bash', {'command': 'ls'}
        self.assertEqual(repeated_observation([Call()], messages), 'No such directory')
        Call.arguments = {'command': 'pwd'}
        self.assertIsNone(repeated_observation([Call()], messages))

    def test_repeated_call_only_counts_the_current_turn(self):
        messages = [{'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Bash', command='ls')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'names'},
            {'role': 'user', 'content': 'second'}]

        class Call:
            name, arguments = 'Bash', {'command': 'ls'}
        self.assertIsNone(repeated_observation([Call()], messages))

    def test_client_control_message_is_not_a_turn_boundary(self):
        messages = [{'role': 'user', 'content': 'find it'},
            {'role': 'assistant', 'content': '', 'tool_calls': [call('a', 'Bash', command='ls')]},
            {'role': 'tool', 'tool_call_id': 'a', 'content': 'No such directory'},
            {'role': 'assistant', 'content': ' '},
            {'role': 'user', 'content': '<total_tokens>14998499 tokens left</total_tokens>'}]
        self.assertEqual(turn_start(messages), 0)
        self.assertTrue(is_control_message(messages[-1]['content']))
        self.assertFalse(is_control_message('Read the file.'))

        class Call:
            name, arguments = 'Bash', {'command': 'ls'}
        self.assertEqual(repeated_observation([Call()], messages), 'No such directory')

    def test_repetition_instruction_includes_the_previous_result(self):
        class Call:
            name = 'Bash'
        text = repetition_instruction(Call(), 'No such directory')
        self.assertIn('already called Bash', text)
        self.assertIn('No such directory', text)
        self.assertIn('Do not repeat', text)

    def test_compact_observation_keeps_both_ends(self):
        text = 'start' + 'm' * 20000 + 'end'
        compacted = compact_observation(text, 5000)
        self.assertLess(len(compacted), 5100)
        self.assertTrue(compacted.startswith('start'))
        self.assertTrue(compacted.endswith('end'))
        self.assertIn('characters omitted', compacted)
        self.assertEqual(compact_observation('short'), 'short')


if __name__ == '__main__':
    unittest.main()
