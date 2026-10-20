# Protocol research

## Actual installed model output

Backend: Lemonade `ryzenai-llm`, checkpoint `amd/Qwen2.5-Coder-7B-Instruct-onnx-ryzenai-1.7-hybrid`, configured context 32,768.

A request declaring `get_weather(city)` produced this assistant **content**, with `finish_reason: stop` and no `tool_calls`:

````text
```xml
<response>
    {"name": "get_weather", "arguments": {"city": "Shanghai"}}
</response>
```
````

An actual Pi request declaring its `read` tool produced a fenced JSON `{ "name": "read", "arguments": { "path": "smoke-secret.txt" } }` as text. FCC passed the text through; Pi did not execute a read. These are synthetic fixture prompts, not user conversations.

The model successfully retained a verification code across conversation messages and retrieved another from 2,930 input tokens. A short generation test reported about 13.5 tokens/second. That is a limited smoke test, not a general coding benchmark or validation at the full context limit.

## Pi

The installed Pi 1.2.0 client uses `@earendil-works/pi-ai`. Its Anthropic handler maps `tool_use` to internal `toolCall`, accumulates `input_json_delta.partial_json`, and emits a tool call only after a complete block. A subsequent internal `toolResult` becomes a user `tool_result` whose `tool_use_id` matches the call ID. Its tool schemas are supplied dynamically; built-in names include lowercase `read`, `write`, `edit`, and `bash`.

Public upstream reference: https://github.com/badlogic/pi-mono/tree/main/packages/ai

## Claude Code

Claude Code uses Anthropic Messages. A tool response contains a `tool_use` block with `id`, `name`, and object `input`, and `stop_reason: tool_use`. Results come back as user `tool_result` blocks. Names and schemas are client-defined (for example `Read.file_path`, `Write.file_path/content`, `Edit.file_path/old_string/new_string`, `Bash.command`). The adapter must preserve the exact supplied schema rather than guessing a casing or parameter mapping.

Streaming order: `message_start`, then start/delta/stop for each content block, then `message_delta` and `message_stop`. Tools use `input_json_delta`, not text deltas. The server may send pings while generating.

Official references:

- https://platform.claude.com/docs/en/build-with-claude/streaming
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls

## FCC

FCC can consume an OpenAI-compatible upstream. An assistant tool response must carry `message.tool_calls` with a function name and JSON-encoded arguments, `finish_reason: tool_calls`, and stable IDs. Its streaming form requires indexed `delta.tool_calls`, a terminal finish reason, and `[DONE]`. FCC then translates it to Anthropic for Pi/Claude Code.

The adapter will expose both native Anthropic and OpenAI-compatible endpoints. It will never execute tools: the calling agent retains execution and permission handling. Invalid, undeclared, or schema-mismatched calls must fail closed, with at most one model repair attempt.

## Observed completion convention

In a real Claude Code fixture, the model successfully consumed tool results but
finished with `{"tool_calls":[{"name":"Final Answer","arguments":{"answer":"result=43"}}]}`.
The adapter treats this exact one-call, answer-only form as ordinary final text,
unless the client actually declared a tool named `Final Answer`. Extra operational
arguments are rejected. Required-tool choices still reject a final answer. No
model-supplied ID is trusted or used for tool execution.

Claude's full tool manuals are much longer than Pi's and include placeholder
paths. The model-facing catalog compacts description annotations and omits
examples/titles/comments. Names, required fields, types, enums, references, and
other constraints are retained; validation uses the untouched original schema.
