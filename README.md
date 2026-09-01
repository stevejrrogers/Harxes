# Harxes

A Rust coding agent for your terminal — plans its work, runs tools, remembers your project, and connects to MCP servers.

```bash
# one-shot
harxes "add a --verbose flag to the CLI and run the tests"

# interactive TUI
harxes
```

## Features

- **Tool-calling agent loop** — Bash, Read (with paging), Write, Edit (golden-diff review), Grep, Glob, with retry/backoff, context compaction, and guardrails (iteration + token caps). Read-only tools from one turn run concurrently.
- **Self-managed plan** — the agent tracks multi-step work with a Todo tool (atomic whole-list writes, one step in-progress at a time), rendered live in the TUI with a spinner; stale plans trigger a reminder. `/plan`, `/todo`.
- **Workspace memory** — `HARXES.md` (or `AGENTS.md`/`CLAUDE.md`) at the repo root and `~/.harxes/HARXES.md` globally are folded into the system prompt; `/init` asks the agent to write one. Shared notes live in `.harxes/agents/NOTES.md`.
- **Sub-agents** — a `Delegate` tool spawns scoped sub-agents whose tool activity nests in the UI (`└ Bash`).
- **Web fetch** — a built-in `Fetch` tool reads pages and APIs (HTML reduced to text, size-capped).
- **MCP client** — stdio MCP servers from config appear as `mcp__server__tool`; `/mcp` shows status.
- **Hooks** — shell commands around tool calls: `pre_tool` can veto (exit 2), `post_tool` feeds output back to the model.
- **Permissions** — dangerous commands prompt y/n in the TUI; config `commands.allow`/`deny` skip or hard-block by pattern; file edits show a colored diff before applying.
- **Sessions & cost** — resume with `--resume <id>` (plan included), per-model token/cost report via `/cost` and at exit.

## Providers

Anthropic and any OpenAI-compatible endpoint (LiteLLM, local servers):

```bash
OPENAI_API_KEY=sk-... harxes --provider openai \
  --base-url https://my-proxy.example/v1 --model my-model "hello"
```

`.../v1` and bare origins get `/chat/completions` appended automatically.

## Config (`~/.harxes/config.json`)

```json
{
  "commands": { "allow": ["cargo *", "git status"], "deny": ["git push *"] },
  "pricing":  { "my-model": { "input_per_mtok": 0.5, "output_per_mtok": 1.5 } },
  "mcp":      { "weather": { "command": "python3", "args": ["server.py"] } },
  "hooks":    { "pre_tool": [{ "match": "Bash", "command": "./check.sh" }] }
}
```

See [docs/AGENTS.md](docs/AGENTS.md) for architecture (hexagonal workspace: `core-domain` → `app` → `infrastructure/*` → `cli`) and the full command list (`/help`).

## Build

```bash
cargo build --release
cargo test --workspace
```
