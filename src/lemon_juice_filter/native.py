"""Normalize native tool responses without turning them into text protocols."""
import asyncio
import json
import logging
import os
import re
from pathlib import Path
import shutil

from .policy import (compact_observation, needs_retry, repeated_observation,
    repair_command, repetition_instruction, retry_instruction, turn_start)
from .protocol import (ProtocolError, embedded_tool_decision, normalize_request, parse_decision,
    schema_for_prompt, text_content)


def unresolved_failures(messages):
    """Track failures within the current user turn, across diagnostic calls."""
    names, pending = {}, set()
    for message in messages[turn_start(messages):]:
        if not isinstance(message, dict):
            continue
        calls = message.get('tool_calls')
        if isinstance(calls, list):
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get('id'), str):
                    continue
                function = call.get('function')
                if not isinstance(function, dict):
                    function = {}
                names[call['id']] = function.get('name')
        if message.get('role') != 'tool':
            continue
        name = names.get(message.get('tool_call_id'))
        if name is None:
            continue
        failed = message.get('is_error') is True or (name in ('bash', 'Bash') and bool(re.search(
            r'(?m)^(?:Command exited with code |Exit code )[1-9][0-9]*\s*$', text_content(message.get('content')))))
        if failed:
            pending.add(name)
        else:
            pending.discard(name)
    return pending


async def validate_shell(calls):
    """Parse shell syntax without executing the command or loading startup files."""
    bash = os.getenv('LEMON_BASH')
    if not bash:
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        bash = str(git_bash) if os.name == 'nt' and git_bash.is_file() else shutil.which('bash')
    if not bash:
        return
    for call in calls:
        command = call.arguments.get('command')
        if call.name not in ('bash', 'Bash') or not isinstance(command, str):
            continue
        env = {k: v for k, v in os.environ.items() if k.upper() not in ('BASH_ENV', 'ENV', 'SHELLOPTS', 'BASHOPTS')}
        process = await asyncio.create_subprocess_exec(bash, '--noprofile', '--norc', '-n',
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL, env=env)
        try:
            await asyncio.wait_for(process.communicate(command.encode('utf-8')), timeout=3)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise ProtocolError('Shell syntax is invalid. Check matching quotes, escapes, and command delimiters.')


async def generate_native(body, app, post_generation, upstream_model):
    _, tools = normalize_request(body, 'openai')
    raw_messages = body.get('messages') or []
    if not isinstance(raw_messages, list):
        raise ProtocolError('messages must be a list.')
    pending_failures = unresolved_failures(raw_messages)
    if body.get('model') != upstream_model:
        raise ProtocolError('Select the configured upstream model for the native endpoint.')
    limit = body.get('max_tokens')
    if limit is None:
        limit = 4096
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ProtocolError('max_tokens must be a positive integer.')
    if not isinstance(body.get('chat_template_kwargs', {}), dict):
        raise ProtocolError('chat_template_kwargs must be an object.')
    for key in ('temperature', 'top_p', 'top_k', 'min_p'):
        value = body.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ProtocolError(f'{key} must be a number.')
    choice = body.get('tool_choice')
    tools_enabled = not (choice == 'none' or isinstance(choice, dict) and choice.get('type') == 'none')
    # A correction may force a call, but must not widen a client's named tool.
    forced_choice = choice if isinstance(choice, dict) and choice.get('type') in ('tool', 'function') else 'required'
    messages = []
    for original in raw_messages:
        if not isinstance(original, dict):
            raise ProtocolError('Each message must be an object.')
        message = {k: v for k, v in original.items() if k not in ('reasoning', 'reasoning_content', 'reasoning_details')}
        # RyzenAI rejects OpenAI's legal null content on assistant tool turns.
        if message.get('content') is None:
            message['content'] = ''
        if message.get('tool_calls'):
            # RyzenAI emits call markup as content alongside parsed calls.
            # Replaying both teaches the model to duplicate/escape its calls.
            message['content'] = ''
            if upstream_model.startswith('Qwen3-'):
                # RyzenAI's input path does not reliably render native tool
                # history. Supply Qwen3's canonical history markup once.
                rendered = []
                for call in message.pop('tool_calls'):
                    fn = call['function']
                    arguments = fn.get('arguments')
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            raise ProtocolError('Malformed historical tool arguments.') from exc
                    rendered.append('<tool_call>\n' + json.dumps({'name': fn['name'],
                        'arguments': arguments}, ensure_ascii=False) + '\n</tool_call>')
                message['content'] = '\n'.join(rendered)
        if message['role'] == 'tool':
            message['content'] = compact_observation(text_content(message['content']))
            if upstream_model.startswith('Qwen3-'):
                output = message['content']
                if message.get('is_error') is True:
                    output = 'Tool execution failed:\n' + output
                message = {'role': 'user', 'content': '<tool_response>\n' + output + '\n</tool_response>'}
        messages.append(message)
    payload = {k: v for k, v in body.items() if v is not None and k in (
        'tools', 'tool_choice', 'parallel_tool_calls', 'temperature', 'top_p', 'top_k',
        'min_p', 'chat_template_kwargs', 'stop', 'seed')}
    payload.update(model=upstream_model, messages=messages, stream=False,
                   max_tokens=min(limit, 4096), parallel_tool_calls=False)
    if body.get('tools') is not None:
        # The client may legally send null parameters; the backend expects an
        # object schema. Validation already accepted the null as the default.
        payload['tools'] = [{**tool, 'function': {**tool['function'],
            'parameters': tool['function'].get('parameters') or {'type': 'object'}}}
            for tool in body['tools']]
    if upstream_model.startswith('Qwen3-'):
        # CLI tool manuals contain lengthy workflow prose. Retain the full
        # validation schema locally while shortening model-facing annotations.
        payload['tools'] = [{**tool, 'function': {**tool['function'],
            'description': tool['function'].get('description', '').split('\n\n', 1)[0][:320],
            'parameters': schema_for_prompt(tool['function'].get('parameters') or {'type': 'object'})}}
            for tool in body.get('tools') or []]
        thinking = payload.get('chat_template_kwargs', {}).get('enable_thinking', True)
        # Agent loops need repeatable tool choices more than reply variety.
        payload.setdefault('temperature', 0.6 if thinking else 0.3)
        payload.setdefault('top_p', 0.95 if thinking else 0.8)
        payload.setdefault('top_k', 20)
        payload.setdefault('min_p', 0)
        if not thinking:
            # RyzenAI 1.7 ignores chat_template_kwargs. Qwen's supported soft
            # switch must also be present in the rendered conversation.
            messages.append({'role': 'system', 'content': '/no_think'})
    if os.getenv('LEMON_NATIVE_DEBUG') == '1':
        logging.getLogger('lemon_juice_filter').info('native body shapes=%s',
            [(m.get('role'), len(m.get('tool_calls') or []), m.get('tool_call_id') is not None,
              len(m['content']) if isinstance(m.get('content'), str) else -1)
             for m in raw_messages])
        logging.getLogger('lemon_juice_filter').info('native request options=%s tools=%s message_shapes=%s',
            {k: payload.get(k) for k in ('temperature', 'top_p', 'top_k', 'tool_choice', 'chat_template_kwargs')},
            [tool.name for tool in tools],
            [(m['role'], type(m.get('content')).__name__, len(m.get('content') or ''),
              len(m.get('tool_calls', []))) for m in messages])
    async with app.state.gate:
        # A duplicate call and a narrated action can occur in one turn: the
        # duplicate correction yields text, which still needs its own
        # correction. Each kind fires once, and at most two corrections fit in
        # the three generations, so the loop stays bounded.
        corrected, invalid, duplicate_fixes, failure_fixes = set(), 0, 0, 0
        # Corrections accumulate: a later protocol repair must not discard the
        # instruction that explained why the previous generation was rejected.
        corrections = []
        for attempt in range(3):
            # Buffered thinking can legitimately exceed the legacy 240s read
            # timeout. Keep a bound while allowing the declared token budget.
            response = await post_generation(payload, read_timeout=max(240, min(600, payload['max_tokens'] / 8 + 30)))
            data = response.json()
            usage = data.get('usage')
            used = usage.get('prompt_tokens') if isinstance(usage, dict) else None
            window = getattr(app.state, 'context_window', None)
            if os.getenv('LEMON_NATIVE_DEBUG') == '1':
                logging.getLogger('lemon_juice_filter').info('native usage prompt_tokens=%s window=%s', used, window)
            if isinstance(used, int) and window and used > window:
                raise ProtocolError(f'Prompt uses {used} tokens, exceeding the verified {window}-token window. '
                    'Compact the conversation before retrying; the backend may have discarded earlier instructions.')
            item = data['choices'][0]
            message = item['message']
            try:
                calls = message.get('tool_calls')
                content = message.get('content')
                if not calls and not content:
                    raise ProtocolError('Generation ended without an answer or tool call. Keep reasoning concise and produce the requested action.')
                decision = parse_decision(json.dumps({'tool_calls': calls}) if calls else content,
                    tools, body.get('tool_choice'), parallel=False)
                if not decision.calls and tools_enabled:
                    # A call printed inside a markdown fence is an action, not an
                    # answer; leaving it as text stalls the client.
                    embedded = embedded_tool_decision(decision.text, tools, body.get('tool_choice'))
                    if embedded:
                        decision = embedded
                if decision.calls:
                    for call in decision.calls:
                        if call.name in ('bash', 'Bash'):
                            call.arguments['command'] = repair_command(call.arguments.get('command'))
                    observation = repeated_observation(decision.calls, raw_messages)
                    if observation is not None and attempt < 2 and duplicate_fixes < 2:
                        # A duplicate cannot change the result. Corrections with
                        # a forced call replace an unbounded client loop.
                        duplicate_fixes += 1
                        corrected.add('duplicate')
                        if os.getenv('LEMON_NATIVE_DEBUG') == '1':
                            logging.getLogger('lemon_juice_filter').info(
                                'native duplicate call=%s arguments=%s', decision.calls[0].name,
                                decision.calls[0].arguments)
                        payload['tool_choice'] = forced_choice
                        corrections.append({'role': 'user', 'content':
                            repetition_instruction(decision.calls[0], observation)})
                        payload['messages'] = messages + corrections
                        continue
                    if observation is not None:
                        # The model insists on a call that already ran. Returning
                        # it would make the client execute it again, so end the
                        # turn with the real observation instead of looping.
                        message['content'] = ('The model repeated a tool call that was already '
                            'executed in this turn, so it was not run again. Its earlier result was:\n'
                            + compact_observation(str(observation), 2000))
                        message.pop('tool_calls', None)
                        item['finish_reason'] = 'stop'
                        break
                    await validate_shell(decision.calls)
                    # Do not expose the redundant raw call markup to clients.
                    message['content'] = None
                    message['tool_calls'] = [{'id': c.id, 'type': 'function', 'function': {
                        'name': c.name, 'arguments': json.dumps(c.arguments, ensure_ascii=False)}} for c in decision.calls]
                    item['finish_reason'] = 'tool_calls'
                else:
                    # A small model can treat the first failed command as the
                    # end of the task. Give it one bounded chance to recover;
                    # never force execution or suppress a genuine blocker.
                    reason = needs_retry(decision.text, raw_messages, tools)
                    if reason and reason not in corrected and attempt < 2 and tools_enabled:
                        # One bounded retry with the tool call forced, so the
                        # model cannot answer from assumption. The client still
                        # controls the outer loop.
                        corrected.add(reason)
                        payload['tool_choice'] = forced_choice
                        corrections.append({'role': 'system', 'content': retry_instruction(reason)})
                        payload['messages'] = messages + corrections
                        continue
                    if tools and pending_failures and attempt < 2 and failure_fixes < 2 and tools_enabled:
                        failure_fixes += 1
                        corrected.add('failure')
                        if failure_fixes > 1:
                            # A second give-up is not accepted. Force the
                            # diagnostic call so the client can make progress.
                            payload['tool_choice'] = forced_choice
                        corrections.append({'role': 'system', 'content':
                            'A tool failed and has not subsequently succeeded; the original task is still pending. '
                            'Reconsider whether a safe diagnostic or corrected tool call can resolve it. '
                            'For a missing path, inspect the working directory or home directory using a tool. '
                            'A directory listing alone does not answer a request to read a file: '
                            'inspect the discovered subdirectory and read the requested file. '
                            'Use the available tools yourself rather than suggesting commands to the user. '
                            'Respect all existing restrictions; if recovery is unsafe, unauthorized, or '
                            'blocked, explain the blocker. Do not repeat the failed call unchanged.'
                            + (' You already reported the failure; do not repeat that explanation. '
                               'Emit the diagnostic tool call now.' if failure_fixes > 1 else '')})
                        payload['messages'] = messages + corrections
                        continue
                    message['content'] = decision.text
                break
            except ProtocolError as exc:
                if os.getenv('LEMON_NATIVE_DEBUG') == '1':
                    logging.getLogger('lemon_juice_filter').warning(
                        'native invalid attempt=%s reason=%s finish=%s reasoning_len=%s usage=%s '
                        'content=%r tool_calls=%r', attempt, exc, item.get('finish_reason'),
                        len(message.get('reasoning_content') or ''), data.get('usage'),
                        (message.get('content') or '')[-3000:], message.get('tool_calls'))
                if invalid:
                    raise ProtocolError('Native output remained invalid after one repair: ' + str(exc)) from exc
                invalid += 1
                corrections.append({'role': 'system', 'content':
                    'Correct the response format or tool arguments: ' + str(exc) +
                    ' Use only the declared tools. Return an actual tool call if work remains, '
                    'or answer the user if the task is finished. Do not repeat a successful operation.'})
                payload['messages'] = messages + corrections
    app.state.completed += 1
    data['model'] = upstream_model
    return data
