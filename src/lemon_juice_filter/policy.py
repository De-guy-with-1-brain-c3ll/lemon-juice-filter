"""Turn-level policy for keeping a small model honest and inside its context."""
import json
import re

# The model narrating a next step instead of taking it.
_ACTION_PROMISE = re.compile(
    r"(?i)\b(?:i(?:'ll| will| am going to| need to| can)|let me|next,? i(?:'ll| will)|i'm going to)\b"
    r"[^.\n]{0,120}?\b(?:read|open|inspect|check|list|run|execute|look|verify|search|find|"
    r"create|write|edit|update|use|call|query|locate|do|proceed|try|start|begin|continue|handle)\b")

# Claude Code appends control-only user messages to the live request after a
# tool result. They are not a new user turn and must not reset its boundary.
_CONTROL_TAGS = re.compile(
    r'<(total_tokens|system-reminder)>.*?</\1>', re.S | re.I)

# A final answer asserting contents that can only come from reading a file.
_FILE_CLAIM = re.compile(
    r"(?i)(?:from the (?:source|input|target) file|the file contains|file contents|"
    r"the exact .{0,20}string|status string|i (?:read|verified|inspected) the "
    r"(?:source |input |target )?file)")

# Shell programs whose output is the file itself, not just its name.
_READ_COMMAND = re.compile(
    r"(?i)(?:^|[|;&(]\s*|\s)(?:cat|head|tail|less|more|sed|awk|grep|rg|strings|od|xxd|"
    r"wc|nl|bat|type)\b")

# A cd path split on spaces is always invalid: cd accepts one argument.
_CD_UNQUOTED = re.compile(
    r"(?m)(?P<prefix>^|&&|\|\||;|\n)(?P<space>\s*)cd\s+"
    # An escaped space (cd a\ b) is already valid and must not be quoted.
    r"(?P<path>[^\s'\"`$;&|<>]+(?:(?<!\\)\s+[^\s'\"`$;&|<>]+)+)(?=\s*(?:&&|\|\||;|\n|$))")

OBSERVATION_LIMIT = 12000
OBSERVATION_TAIL = 2000


def is_control_message(content):
    """True for a client control line that carries no user request."""
    if not isinstance(content, str) or not content.strip():
        return False
    return not _CONTROL_TAGS.sub('', content).strip()


def turn_start(messages):
    """Index of the last genuine user message, ignoring client control lines."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get('role') == 'user' and not is_control_message(message.get('content')):
            return index
    return 0


def promises_action(text):
    """True when the model narrated a next step instead of performing it."""
    return bool(text) and bool(_ACTION_PROMISE.search(text))


def claims_file_contents(text):
    """True when the answer asserts contents that require a prior read."""
    return bool(text) and bool(_FILE_CLAIM.search(text))


def _commands(messages):
    """Map tool_call_id to (name, command) for every call in the history."""
    names = {}
    for message in messages:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            argument = function.get('arguments')
            command = None
            if isinstance(argument, str):
                try:
                    command = json.loads(argument).get('command')
                except (json.JSONDecodeError, AttributeError):
                    command = argument
            elif isinstance(argument, dict):
                command = argument.get('command')
            names[call.get('id')] = (function.get('name'), command)
    return names


def read_evidence(messages):
    """True when the current user turn contains an observation of file contents."""
    start = turn_start(messages)
    names = _commands(messages)
    for message in messages[start:]:
        if message.get('role') != 'tool':
            continue
        name, command = names.get(message.get('tool_call_id'), (None, None))
        if name == 'Read':
            return True
        if name in ('Bash', 'bash') and isinstance(command, str) and _READ_COMMAND.search(command):
            return True
    return False


def repair_command(command):
    """Quote an unquoted multi-word cd path, which bash always rejects."""
    if not isinstance(command, str):
        return command
    return _CD_UNQUOTED.sub(
        lambda match: f"{match.group('prefix')}{match.group('space')}cd '{match.group('path')}'",
        command)


def _signature(name, arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return name + '\x00' + json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def repeated_observation(calls, messages):
    """Return the observation of an identical call in this turn, or None.

    Repeating a call cannot change its result, so the caller can stop the loop
    with a corrective instruction instead of executing the duplicate again.
    """
    start = turn_start(messages)
    signatures, observed = {}, {}
    for message in messages[start:]:
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            signatures[call.get('id')] = _signature(function.get('name'), function.get('arguments'))
        if message.get('role') == 'tool':
            signature = signatures.get(message.get('tool_call_id'))
            if signature is not None:
                observed[signature] = message.get('content')
    for call in calls:
        signature = _signature(call.name, call.arguments)
        if signature in observed:
            return observed[signature] or ''
    return None


def repetition_instruction(call, observation):
    """Tell the model that its identical call was already executed."""
    observed = compact_observation(str(observation), 800)
    return ('You already called ' + call.name + ' with these exact arguments in this turn and received '
            'the result below. Repeating it cannot change the result.\n'
            '<previous_result>\n' + observed + '\n</previous_result>\n'
            'Do not repeat that call. Use the result, inspect a different path, or take a different '
            'action that makes progress. A failed command must be changed, not retried unchanged. '
            'If a path was wrong, list its parent directory and copy the exact entry name from that output.')


def needs_retry(text, messages, tools):
    """Explain why a text reply should be retried, or return None.

    Retrying is bounded to one attempt by the caller, so a false positive costs
    one generation and never changes the answer that is finally returned.
    """
    if not text or not tools:
        return None
    if promises_action(text):
        return 'action'
    if claims_file_contents(text) and not read_evidence(messages):
        return 'evidence'
    return None


def retry_instruction(reason):
    if reason == 'evidence':
        return ('Your reply reports file contents, but no file was read in this turn. '
                'Call Read or a shell command that prints the file before reporting its contents.')
    return ('Your last reply announced an action but did not call a tool. '
            'Emit the tool call now with the exact arguments; do not describe or simulate it.')


def compact_observation(text, limit=OBSERVATION_LIMIT):
    """Bound one tool observation so a large result cannot crowd out the request."""
    if not text or len(text) <= limit:
        return text
    head = limit - OBSERVATION_TAIL - 80
    removed = len(text) - head - OBSERVATION_TAIL
    return text[:head] + f'\n... [{removed} characters omitted] ...\n' + text[-OBSERVATION_TAIL:]
