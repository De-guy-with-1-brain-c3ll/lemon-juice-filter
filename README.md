# lemon-juice-filter

A local compatibility adapter for Lemonade's Qwen2.5-Coder-7B-Instruct-Hybrid model, Pi, Claude Code, and Free Claude Code (FCC).

The installed hybrid model emits requested tool actions as XML-wrapped or fenced JSON **text**. Clients require structured tool calls before they will execute anything. This project translates that boundary and preserves the tool-result conversation.

Implementation milestones:

1. Record the observed protocols and scaffold a standalone project.
2. Implement validated output parsing, history translation, and both wire protocols; add automated tests.
3. Verify real multi-step agent tasks and provide local launch/configuration helpers.

No model weights, credentials, private conversations, or machine configuration backups belong in this repository.

## Run

Start Lemonade with the already-downloaded `Qwen2.5-Coder-7B-Instruct-Hybrid` model. Its NPU cannot be shared with a running FLM model server.

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your shell, then:
python -m pip install -e .
lemon-juice-filter
```

The adapter listens only on `127.0.0.1:13306`. It uses Lemonade at `http://127.0.0.1:13305/api/v1`. Optional environment settings: `LEMON_BASE_URL`, `LEMON_MODEL`, and `LEMON_FILTER_PORT`.

### Pi

```sh
python scripts/configure_pi.py
pi --provider lemon-juice-filter --model lemon-hybrid --thinking off
```

The configuration helper backs up Pi's models file and adds a provider without changing existing defaults or credentials. The provider advertises the configured **32,768 context / 4,096 output** limits and zero local token cost. Restart Pi to load it.

### Claude Code

Point `ANTHROPIC_BASE_URL` to `http://127.0.0.1:13306`, set a dummy local `ANTHROPIC_API_KEY`, and run `claude --model lemon-hybrid`. No Anthropic credentials are needed for local inference.

On Windows / PowerShell 7, these launchers start the local services when needed:

```powershell
& ./scripts/pi-hybrid.ps1
& ./scripts/claude-hybrid.ps1
```

`Start-Filter.ps1` prefers the project's `.venv`, then an existing FCC Python environment. You can set `LEMON_FILTER_PYTHON` explicitly. The Claude launcher scopes environment changes to the child session, selects the local model for each tier, and starts with Read, Write, Edit, and Bash to fit the smaller context. Agent permission checks remain in effect.

The Claude launcher supplies a concise local-model system prompt with the real
working directory. This replaces the much larger default Claude system prompt;
project-specific instructions should be supplied in the user context as needed.
It does not add execution permissions or bypass client checks. Direct API use
can supply a different system prompt; reliability depends on that prompt and the model.

### FCC

Add a custom **OpenAI Chat Completions** provider with base URL `http://127.0.0.1:13306/v1`, any dummy local key, and model `lemon-hybrid`. Use default/no reasoning encoding. FCC handles its usual Anthropic-facing clients while this adapter supplies valid OpenAI tool calls upstream.

## Behavior and limits

- Both `/v1/messages` and `/v1/chat/completions` support complete and SSE responses. `/v1/models`, `/health`, and estimated `/v1/messages/count_tokens` are also available.
- Tool names and parameter schemas come from each request. No tool-name casing or argument aliases are guessed. This includes Pi's current structured `edit.edits[]` schema and Claude Code's `Edit.old_string/new_string` schema.
- Only one validated tool action is emitted per model response. This makes dependent file operations wait for their preceding tool result. Undeclared tools, schema failures, truncated calls, and invalid batches trigger at most one repair before an error.
- The adapter never executes commands or file operations. Pi/Claude Code perform them using their normal permissions.
- The hybrid's raw output is buffered before tool validation. SSE clients receive pings while waiting; this is not token-by-token model streaming. The first visible output arrives after a complete validated decision.
- This is text-only. Unsupported image/content types fail explicitly. Extended thinking, server-side tools, Responses API, prompt caching, and exact token counting are not implemented. History is not silently trimmed by the adapter. The 32k figure is the configured backend context, not a claim of full-window quality validation.
- Schema validation improves protocol reliability, not reasoning ability. A weaker model can still choose the wrong valid action. Keep normal agent permission prompts enabled.
- Request logs contain request IDs, tool names, timings, and repair counts; they do not intentionally log prompts, arguments, or file contents. Lemonade and the coding clients have their own logging policies.

## Tests

```sh
python -m unittest discover -s tests -v
# Requires a running adapter and the hybrid model; synthetic fixtures only:
python scripts/live_protocol.py
```

See [protocol research](docs/protocol-research.md) for the observed failure, client contracts, and primary references.
