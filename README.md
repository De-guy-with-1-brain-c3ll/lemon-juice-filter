<p align="center">
  <img src="assets/logo.png" alt="A lemon squeezed through a filter" width="200">
</p>

# lemon-juice-filter

Tool-call translation for local models and coding agents.

Some inference backends return tool calls as ordinary text. lemon-juice-filter
converts those responses into structured calls for Pi, Claude Code, and Free
Claude Code. It exposes Anthropic Messages and OpenAI Chat Completions endpoints
over an OpenAI-compatible inference backend.

Protocol translation only goes so far. The model still needs to follow tool
schemas and reason over tool results.

## Installation

Requires Python 3.11+ and a running inference server.

```sh
git clone https://github.com/De-guy-with-1-brain-c3ll/lemon-juice-filter.git
cd lemon-juice-filter
python -m pip install -e .
```

Set `LEMON_MODEL` to the model ID exposed by the inference server and
`LEMON_BASE_URL` to its OpenAI-compatible API base, then run `lemon-juice-filter`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `LEMON_MODEL` | `Qwen2.5-Coder-7B-Instruct-Hybrid` | Upstream model ID |
| `LEMON_BASE_URL` | `http://127.0.0.1:13305/api/v1` | Upstream API base |
| `LEMON_FILTER_PORT` | `13306` | Adapter port |
| `LEMON_CONTEXT_WINDOW` | `32768` | Advertised context window; `scripts/Start-Filter.ps1` lowers it to the verified backend limit |
| `LEMON_AUTO_RECOVER_WEIGHTS` | `1` | Recovery for a specific RyzenAI lost-weights fault; use `0` for other backends |

The defaults target Lemonade. Other backends must provide an unauthenticated
OpenAI-compatible `/chat/completions` endpoint; they have not been validated here.
The client-facing model ID is currently `lemon-hybrid`, independent of the upstream
ID. The advertised context window comes from `LEMON_CONTEXT_WINDOW`; the output
limit is 4,096 tokens. RyzenAI can silently cap prompts below the requested
context, so `scripts/Start-Filter.ps1` reads the backend's `max_prompt_length`
and lowers the advertised window to it. `scripts/configure_ryzenai_context.py`
raises that limit for a hybrid model snapshot (with a backup) when it is below
the value needed for agent use.

| Client | API base | Model |
| --- | --- | --- |
| Pi / Claude Code | `http://127.0.0.1:13306` (Anthropic Messages) | `lemon-hybrid` |
| FCC | `http://127.0.0.1:13306/v1` (OpenAI Chat Completions) | `lemon-hybrid` |
| FCC, native | `http://127.0.0.1:13306/native/v1` (OpenAI Chat Completions) | upstream model ID |
| FCC, native without thinking | `http://127.0.0.1:13306/native/no-thinking/v1` | upstream model ID |

Use a placeholder API key if the client requires one. The adapter binds to loopback.

Provider registration helpers: `scripts/configure_pi.py`, `scripts/configure_fcc.py`.
PowerShell launchers are in [`scripts/`](scripts/) and include Lemonade-specific
startup logic. The Claude Code launcher replaces the default system prompt with
a smaller prompt and selects Read, Write, Edit, and Bash.

For models with native tool calls and reasoning output, use the OpenAI-compatible
base `http://127.0.0.1:13306/native/v1` and the upstream model ID. Start the service
with `LEMON_MODEL` set, then run `python scripts/configure_fcc.py --native` to
register this route in FCC. FCC's reasoning policy must allow client requests.
Qwen3 uses its recommended sampling defaults and canonical tool-history markup
for RyzenAI compatibility. Shell calls are syntax-checked with `bash -n` when
Bash is installed; invalid calls get one repair attempt. A reply that announces
an action without calling a tool gets one retry asking for the call itself.
Output is buffered for validation, including reasoning, so this path does not
stream tokens as generated.

`scripts/configure_fcc.py --native --no-thinking` registers the route that
disables thinking. RyzenAI 1.7 ignores `chat_template_kwargs`, so the adapter
also appends Qwen's `/no_think` soft switch. `scripts/claude-native.ps1`
selects the route from `LEMON_CLAUDE_THINKING` (`0` disables thinking).

## Verification

The unit and HTTP suite covers protocol translation, validation, and the
native route. Integration scripts exercise a running adapter and backend with a
real client and check actual tool calls and observed file contents, not just
returned text.

```sh
python -m unittest discover -s tests -v
python scripts/live_protocol.py  # requires a running adapter and backend
python scripts/live_multiturn.py  # native route; temporary file recovery and follow-up
python scripts/live_claude_native.py  # Claude Code through FCC, two resumed turns
```

Local translation and validation add negligible time relative to inference.
Tool-call reliability depends on the upstream model and quantization; the
adapter validates arguments and retries malformed output once.

Protocol notes: [`docs/protocol-research.md`](docs/protocol-research.md).

## License

[MIT](LICENSE)
