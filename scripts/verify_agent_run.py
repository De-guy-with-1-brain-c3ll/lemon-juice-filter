"""Verify real agent actions, filesystem results, and the final answer together."""
import argparse
import json
from pathlib import Path


def verify(agent, log_path, output_path, expected, required):
    events = [json.loads(line) for line in Path(log_path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    calls, errors, final = [], 0, ''
    api_error = False
    for event in events:
        if agent == 'pi':
            if event.get('type') == 'tool_execution_start':
                calls.append(event['toolName'])
            if event.get('type') == 'tool_execution_end':
                errors += bool(event.get('isError'))
            message = event.get('message', {})
            if event.get('type') == 'message_end' and message.get('role') == 'assistant':
                final = '\n'.join(block.get('text', '') for block in message.get('content', []))
                api_error = message.get('stopReason') in ('error', 'aborted')
        else:
            if event.get('type') == 'assistant':
                calls.extend(block['name'] for block in event.get('message', {}).get('content', []) if block.get('type') == 'tool_use')
            if event.get('type') == 'user':
                errors += sum(bool(block.get('is_error')) for block in event.get('message', {}).get('content', []) if block.get('type') == 'tool_result')
            if event.get('type') == 'result':
                final = event.get('result') or ''
                api_error = event.get('is_error', True)
    output = Path(output_path)
    actual = output.read_text(encoding='utf-8-sig').strip() if output.exists() else None
    observed = {name.casefold() for name in calls}
    report = {'agent': agent, 'tool_calls': calls, 'tool_errors_recovered_or_pending': errors,
        'required_tools_executed': set(required).issubset(observed), 'api_error': api_error,
        'file_matches_expected': actual == expected,
        'final_answer_matches_file': actual is not None and actual in final}
    report['success'] = all(report[key] for key in (
        'required_tools_executed', 'file_matches_expected', 'final_answer_matches_file')) and not api_error
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=['pi', 'claude'], required=True)
    parser.add_argument('--log', required=True)
    parser.add_argument('--file', required=True)
    parser.add_argument('--expected', default='result=43')
    parser.add_argument('--require-tools', nargs='+', default=['read', 'write', 'edit'])
    args = parser.parse_args()
    report = verify(args.agent, args.log, args.file, args.expected, args.require_tools)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['success'] else 1)


if __name__ == '__main__':
    main()
