# lemon-juice-filter

A local compatibility adapter for Lemonade's Qwen2.5-Coder-7B-Instruct-Hybrid model, Pi, Claude Code, and Free Claude Code (FCC).

The installed hybrid model emits requested tool actions as XML-wrapped or fenced JSON **text**. Clients require structured tool calls before they will execute anything. This project translates that boundary and preserves the tool-result conversation.

Implementation milestones:

1. Record the observed protocols and scaffold a standalone project.
2. Implement validated output parsing, history translation, and both wire protocols; add automated tests.
3. Verify real multi-step agent tasks and provide local launch/configuration helpers.

No model weights, credentials, private conversations, or machine configuration backups belong in this repository.
