"""Normalize native tool responses without turning them into text protocols."""
import asyncio
import json
import logging
import os
from pathlib import Path
import shutil

from .protocol import ProtocolError, normalize_request, parse_decision, text_content


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
    if body.get('model') != upstream_model:
        raise ProtocolError('Select the configured upstream model for the native endpoint.')
    limit = body.get('max_tokens', 4096)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ProtocolError('max_tokens must be a positive integer.')
    if not isinstance(body.get('chat_template_kwargs', {}), dict):
        raise ProtocolError('chat_template_kwargs must be an object.')
    messages = []
    for original in body.get('messages', []):
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
                    arguments = fn['arguments']
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    rendered.append('<tool_call>\n' + json.dumps({'name': fn['name'],
                        'arguments': arguments}, ensure_ascii=False) + '\n</tool_call>')
                message['content'] = '\n'.join(rendered)
        if message['role'] == 'tool' and upstream_model.startswith('Qwen3-'):
            message = {'role': 'user', 'content': '<tool_response>\n' + text_content(message['content']) + '\n</tool_response>'}
        messages.append(message)
    payload = {k: v for k, v in body.items() if k in (
        'tools', 'tool_choice', 'parallel_tool_calls', 'temperature', 'top_p', 'top_k',
        'min_p', 'chat_template_kwargs', 'stop', 'seed')}
    payload.update(model=upstream_model, messages=messages, stream=False,
                   max_tokens=min(limit, 4096), parallel_tool_calls=False)
    if upstream_model.startswith('Qwen3-'):
        guidance = ('Use concise reasoning and one tool action at a time. When choosing shell commands, '
            'prefer simple commands and quote entire paths containing spaces. Inspect directory structure '
            'before filtering by filename or assuming a programming language. After a tool error, '
            'inspect its output and correct the cause rather than repeating the same command. '
            'Once tool results answer the user, give the answer without another tool call.')
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = text_content(messages[0]['content']) + '\n\n' + guidance
        else:
            messages.insert(0, {'role': 'system', 'content': guidance})
        thinking = payload.get('chat_template_kwargs', {}).get('enable_thinking', True)
        payload.setdefault('temperature', 0.6 if thinking else 0.7)
        payload.setdefault('top_p', 0.95 if thinking else 0.8)
        payload.setdefault('top_k', 20)
        payload.setdefault('min_p', 0)
    if os.getenv('LEMON_NATIVE_DEBUG') == '1':
        logging.getLogger('lemon_juice_filter').info('native request options=%s message_shapes=%s',
            {k: payload.get(k) for k in ('temperature', 'top_p', 'top_k', 'tool_choice', 'chat_template_kwargs')},
            [(m['role'], type(m.get('content')).__name__, len(m.get('content') or ''),
              len(m.get('tool_calls', []))) for m in messages])
    async with app.state.gate:
        for attempt in range(2):
            # Buffered thinking can legitimately exceed the legacy 240s read
            # timeout. Keep a bound while allowing the declared token budget.
            response = await post_generation(payload, read_timeout=max(240, min(600, payload['max_tokens'] / 8 + 30)))
            data = response.json()
            item = data['choices'][0]
            message = item['message']
            try:
                calls = message.get('tool_calls')
                content = message.get('content')
                if not calls and not content:
                    raise ProtocolError('Generation ended without an answer or tool call. Keep reasoning concise and produce the requested action.')
                decision = parse_decision(json.dumps({'tool_calls': calls}) if calls else content,
                    tools, body.get('tool_choice'), parallel=False)
                if decision.calls:
                    await validate_shell(decision.calls)
                    # Do not expose the redundant raw call markup to clients.
                    message['content'] = None
                    message['tool_calls'] = [{'id': c.id, 'type': 'function', 'function': {
                        'name': c.name, 'arguments': json.dumps(c.arguments, ensure_ascii=False)}} for c in decision.calls]
                    item['finish_reason'] = 'tool_calls'
                else:
                    message['content'] = decision.text
                break
            except ProtocolError as exc:
                if attempt:
                    raise ProtocolError('Native output remained invalid after one repair: ' + str(exc)) from exc
                payload['messages'] = messages + [{'role': 'user', 'content':
                    'Correct the response format or tool arguments: ' + str(exc) +
                    ' Use only the declared tools. Return an actual tool call if work remains, '
                    'or answer the user if the task is finished. Do not repeat a successful operation.'}]
    app.state.completed += 1
    data['model'] = upstream_model
    return data
