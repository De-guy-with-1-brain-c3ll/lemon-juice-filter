"""Loopback-only HTTP adapter. Generation is buffered before validated SSE emission."""
import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .protocol import ProtocolError, normalize_request, parse_decision, tool_prompt

log = logging.getLogger('lemon_juice_filter')
MODEL = 'lemon-hybrid'
UPSTREAM_MODEL = os.getenv('LEMON_MODEL', 'Qwen2.5-Coder-7B-Instruct-Hybrid')
UPSTREAM_BASE = os.getenv('LEMON_BASE_URL', 'http://127.0.0.1:13305/api/v1').rstrip('/')
CONTEXT = 32768
MAX_OUTPUT = 4096


@asynccontextmanager
async def lifespan(app):
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(240, connect=5)) as client:
        app.state.client = client
        app.state.gate = asyncio.Semaphore(1)
        app.state.completed = 0
        yield


app = FastAPI(title='lemon-juice-filter', version='0.1.0', lifespan=lifespan)


@app.middleware('http')
async def local_only(request, call_next):
    if request.client and request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        return JSONResponse({'error': {'message': 'Loopback clients only.'}}, status_code=403)
    if request.headers.get('origin'):
        # Coding agents do not need browser access. Do not enable ambient browser callers.
        return JSONResponse({'error': {'message': 'Browser-origin requests are not supported.'}}, status_code=403)
    return await call_next(request)


@app.exception_handler(ProtocolError)
async def protocol_error(request, exc):
    return JSONResponse({'type': 'error', 'error': {'type': 'invalid_request_error', 'message': str(exc)}}, status_code=400)


@app.get('/health')
async def health():
    return {'status': 'ok', 'model': MODEL, 'upstream_model': UPSTREAM_MODEL,
            'context_window': CONTEXT, 'completed_requests': getattr(app.state, 'completed', 0)}


@app.get('/v1/models')
async def models():
    return {'object': 'list', 'data': [{'id': MODEL, 'type': 'model', 'object': 'model',
        'display_name': 'Lemonade Qwen2.5 Coder Hybrid (adapted)', 'owned_by': 'local',
        'created': 0, 'created_at': '1970-01-01T00:00:00Z', 'context_length': CONTEXT,
        'contextWindow': CONTEXT, 'maxCompletionTokens': MAX_OUTPUT,
        'supportsReasoning': False, 'inputModalities': ['text']}],
        'has_more': False, 'first_id': MODEL, 'last_id': MODEL}


async def generate(body, wire):
    messages, tools = normalize_request(body, wire)
    choice = body.get('tool_choice', 'auto')
    if not messages:
        raise ProtocolError('At least one message is required.')
    if body.get('model') not in (MODEL, UPSTREAM_MODEL):
        raise ProtocolError(f'Unknown model. Select {MODEL}.')
    systems = [m['content'] for m in messages if m['role'] == 'system']
    transcript = [m for m in messages if m['role'] != 'system']
    if tools:
        systems.append(tool_prompt(tools, choice))
    system = '\n\n'.join(systems)
    upstream_messages = ([{'role': 'system', 'content': system}] if system else []) + transcript
    requested = body.get('max_tokens', body.get('max_completion_tokens', 1024))
    if not isinstance(requested, int) or isinstance(requested, bool) or requested <= 0:
        raise ProtocolError('max_tokens must be a positive integer.')
    payload = {'model': UPSTREAM_MODEL, 'messages': upstream_messages, 'stream': False,
        'max_tokens': min(requested, MAX_OUTPUT), 'temperature': body.get('temperature', 0)}
    parallel = body.get('parallel_tool_calls', True) and not (
        isinstance(choice, dict) and choice.get('disable_parallel_tool_use', False))
    request_id = uuid.uuid4().hex[:16]
    started = time.monotonic()
    async with app.state.gate:
        for attempt in range(2):
            response = await app.state.client.post(UPSTREAM_BASE + '/chat/completions', json=payload)
            response.raise_for_status()
            data = response.json()
            try:
                item = data['choices'][0]
                raw = item['message'].get('content') or ''
                # Native calls, if a future backend starts emitting them, use the same validation.
                native = item['message'].get('tool_calls')
                if native:
                    raw = json.dumps({'tool_calls': native})
                if item.get('finish_reason') == 'length' and tools:
                    raise ProtocolError('Model output was truncated; reduce the requested operation or increase max_tokens.')
                decision = parse_decision(raw, tools, choice, parallel)
                break
            except ProtocolError as exc:
                if attempt:
                    raise ProtocolError('Model returned invalid tool output after one repair: ' + str(exc)) from exc
                payload['messages'] = upstream_messages + [
                    {'role': 'assistant', 'content': raw},
                    {'role': 'user', 'content': 'Protocol correction: ' + str(exc) +
                     ' Return one complete JSON decision using the declared tool schemas. Do not execute anything yourself.'}]
    usage = data.get('usage') or {}
    app.state.completed += 1
    log.info('completed request=%s wire=%s tools=%s repairs=%d seconds=%.2f',
             request_id, wire, [c.name for c in decision.calls], attempt, time.monotonic()-started)
    return decision, usage, item.get('finish_reason', 'stop')


def anthropic_message(decision, usage, finish):
    content = []
    if decision.text:
        content.append({'type': 'text', 'text': decision.text})
    content.extend({'type': 'tool_use', 'id': c.id, 'name': c.name, 'input': c.arguments} for c in decision.calls)
    if not content:
        content.append({'type': 'text', 'text': ''})
    return {'id': 'msg_' + uuid.uuid4().hex, 'type': 'message', 'role': 'assistant',
        'model': MODEL, 'content': content, 'stop_reason': 'tool_use' if decision.calls else
        ('max_tokens' if finish == 'length' else 'end_turn'), 'stop_sequence': None,
        'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': usage.get('completion_tokens', 0)}}


def openai_message(decision, usage, finish):
    message = {'role': 'assistant', 'content': decision.text or None}
    if decision.calls:
        message['tool_calls'] = [{'id': c.id, 'type': 'function', 'function': {
            'name': c.name, 'arguments': json.dumps(c.arguments, ensure_ascii=False)}} for c in decision.calls]
    return {'id': 'chatcmpl-' + uuid.uuid4().hex, 'object': 'chat.completion', 'created': int(time.time()),
        'model': MODEL, 'choices': [{'index': 0, 'message': message, 'finish_reason':
            'tool_calls' if decision.calls else ('length' if finish == 'length' else 'stop')}],
        'usage': {'prompt_tokens': usage.get('prompt_tokens', 0), 'completion_tokens': usage.get('completion_tokens', 0),
                  'total_tokens': usage.get('total_tokens', 0)}}


def sse(data, event=None):
    return (f'event: {event}\n' if event else '') + 'data: ' + json.dumps(data, ensure_ascii=False) + '\n\n'


def anthropic_events(message):
    start = {**message, 'content': [], 'stop_reason': None, 'usage': {**message['usage'], 'output_tokens': 0}}
    yield sse({'type': 'message_start', 'message': start}, 'message_start')
    for index, block in enumerate(message['content']):
        tool = block['type'] == 'tool_use'
        empty = {**block, 'input': {}} if tool else {**block, 'text': ''}
        yield sse({'type': 'content_block_start', 'index': index, 'content_block': empty}, 'content_block_start')
        delta = {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'], ensure_ascii=False)} if tool else {
            'type': 'text_delta', 'text': block['text']}
        yield sse({'type': 'content_block_delta', 'index': index, 'delta': delta}, 'content_block_delta')
        yield sse({'type': 'content_block_stop', 'index': index}, 'content_block_stop')
    yield sse({'type': 'message_delta', 'delta': {'stop_reason': message['stop_reason'], 'stop_sequence': None},
               'usage': {'output_tokens': message['usage']['output_tokens']}}, 'message_delta')
    yield sse({'type': 'message_stop'}, 'message_stop')


def openai_events(message):
    base = {k: message[k] for k in ('id', 'created', 'model')}
    base['object'] = 'chat.completion.chunk'
    delta = message['choices'][0]['message'].copy()
    if 'tool_calls' in delta:
        delta['tool_calls'] = [{'index': i, **c} for i, c in enumerate(delta['tool_calls'])]
    yield sse({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})
    yield sse({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': message['choices'][0]['finish_reason']}],
               'usage': message['usage']})
    yield 'data: [DONE]\n\n'


async def handle(body, wire):
    # Validate input before sending HTTP 200 / beginning an SSE stream.
    normalize_request(body, wire)
    if body.get('model') not in (MODEL, UPSTREAM_MODEL):
        raise ProtocolError(f'Unknown model. Select {MODEL}.')
    if not body.get('stream'):
        try:
            decision, usage, finish = await generate(body, wire)
            return JSONResponse((anthropic_message if wire == 'anthropic' else openai_message)(decision, usage, finish))
        except httpx.HTTPError:
            return JSONResponse({'type': 'error', 'error': {'type': 'api_error', 'message': 'Lemonade inference failed. Check the local backend.'}}, status_code=502)

    async def events():
        task = asyncio.create_task(generate(body, wire))
        try:
            while not task.done():
                done, _ = await asyncio.wait({task}, timeout=10)
                if not done:
                    yield sse({'type': 'ping'}, 'ping') if wire == 'anthropic' else ': ping\n\n'
            decision, usage, finish = task.result()
            message = (anthropic_message if wire == 'anthropic' else openai_message)(decision, usage, finish)
            for event in (anthropic_events(message) if wire == 'anthropic' else openai_events(message)):
                yield event
        except (ProtocolError, httpx.HTTPError, KeyError, ValueError) as exc:
            error = {'type': 'error', 'error': {'type': 'api_error', 'message':
                str(exc) if isinstance(exc, ProtocolError) else 'Local inference returned an invalid or unavailable response.'}}
            yield sse(error, 'error' if wire == 'anthropic' else None)
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
    return StreamingResponse(events(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.post('/v1/messages')
async def messages(request: Request):
    return await handle(await request.json(), 'anthropic')


@app.post('/v1/chat/completions')
async def completions(request: Request):
    return await handle(await request.json(), 'openai')


@app.post('/v1/messages/count_tokens')
async def count_tokens(request: Request):
    body = await request.json()
    normalize_request(body, 'anthropic')
    # A conservative estimate, not a tokenizer claim. Never silently trim history.
    return JSONResponse({'input_tokens': max(1, len(json.dumps(body, ensure_ascii=False)) // 3)},
                        headers={'X-Lemon-Token-Count': 'estimated'})


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host='127.0.0.1', port=int(os.getenv('LEMON_FILTER_PORT', '13306')))


if __name__ == '__main__':
    main()
