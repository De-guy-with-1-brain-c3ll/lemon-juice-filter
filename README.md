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
| `LEMON_AUTO_RECOVER_WEIGHTS` | `1` | Recovery for a specific RyzenAI lost-weights fault; use `0` for other backends |

The defaults target Lemonade. Other backends must provide an unauthenticated
OpenAI-compatible `/chat/completions` endpoint; they have not been validated here.
The client-facing model ID is currently `lemon-hybrid`, independent of the upstream
ID. Advertised context and output limits are fixed at 32,768 and 4,096 tokens;
changing the upstream model does not adjust them automatically.

| Client | API base | Model |
| --- | --- | --- |
| Pi / Claude Code | `http://127.0.0.1:13306` (Anthropic Messages) | `lemon-hybrid` |
| FCC | `http://127.0.0.1:13306/v1` (OpenAI Chat Completions) | `lemon-hybrid` |

Use a placeholder API key if the client requires one. The adapter binds to loopback.

Provider registration helpers: `scripts/configure_pi.py`, `scripts/configure_fcc.py`.
PowerShell launchers are in [`scripts/`](scripts/) and include Lemonade-specific
startup logic. The Claude Code launcher replaces the default system prompt with
a smaller prompt and selects Read, Write, Edit, and Bash.

## Test results

Integration fixture: read an input file, write an output, edit it, read it back,
then check the final answer against the file. Backend: Qwen2.5-Coder-7B-Instruct-Hybrid
on Lemonade/RyzenAI. Snapshot: 2026-10-20.

| Check | Result |
| --- | --- |
| Unit and HTTP suite | 35 passed |
| Pi → adapter → backend | Passed, 33.4 s summed API time |
| Claude Code → adapter → backend | Passed, 168 s wall time including failure/recovery |
| Pi → FCC → adapter → backend | Passed, 33.3 s summed API time |
| Local translation and validation | 1.79 ms/request |

Processing overhead is the median of three runs of 1,000 iterations with Pi's
read/write/edit schemas, excluding HTTP transport and inference. Integration
timings are individual runs, not throughput benchmarks. Earlier runs failed on
invalid arguments after repair; this model's completion rate remains uneven.

```sh
python -m unittest discover -s tests -v
python scripts/live_protocol.py  # requires a running adapter and backend
```

Protocol notes: [`docs/protocol-research.md`](docs/protocol-research.md).

## License

[MIT](LICENSE)
