# Harxes — Dev Notes for Future Agents

> Trạng thái dự án + hướng đi tiếp. Cập nhật khi có milestone mới.

## TL;DR

`harxes` là CLI coding agent (như Claude Code) viết bằng Rust, kiến trúc hexagonal
qua Cargo workspace. Repo đã có lịch sử commit đầy đủ trên `main`; cả workspace
compile sạch và toàn bộ test pass.

Các vòng phát triển gần nhất (từ sau khi hướng tới "ngon hơn Claude/Copilot") đã bổ
sung: **Grep/Glob/Edit tools**, **context compression** theo budget, **retry/backoff**,
**streaming output realtime (SSE)** cho Anthropic + OpenAI, và **Ctrl-C cancel** cho
turn đang chạy trong TUI.

## Kho crates

- `core-domain` — (dependency-free) entities + value objects (`Message`, `Role`,
  `ProviderId`, `ModelId`, `TokenUsage`, `ContextBudget`…), ports
  (`LlmPort`, `ShellPort`, `FileSystemPort`, `ConfigStorePort`, `SecretsVaultPort`,
  `SessionStorePort`, `PermissionDecider`, `ToolObserver`).
- `app` — usecase `AgentLoop` (tool-calling loop nhiều vòng, retry, context compress,
  streaming). Phụ thuộc port, không phụ thuộc infra.
- `infrastructure/*` — các adapter implement port (mỗi cái là một package riêng
  `harxes-infra-*`, gom trong thư mục container `crates/infrastructure/`):
  - `infrastructure/llm` — adapter Anthropic + OpenAI (`generate` non-stream + `generate_stream` SSE).
  - `infrastructure/fs` — `HostFileSystem`: read/write/replace (Edit) + glob + grep.
  - `infrastructure/shell` — tokio command shell.
  - `infrastructure/auth` / `infrastructure/session` — env keys / JSON session store.
- `cli` — TUI (ratatui) + one-shot prompt, composition root (`compose.rs`).

## Những gì đã implement trong session này (tăng "ngon hơn Claude/Copilot")

### Search tools (Grep + Glob)
- `FileSystemPort::grep(re, path, options)` → `Vec<GrepMatch>`; `glob(pattern, path)` → `Vec<String>`.
- `infrastructure/fs` bỏ qua các thư mục lớn: `.git`, `target`, `node_modules`, v.v.

### Edit tool
- `FileSystemPort::replace(old, new, path)`; tool `Edit` trong `tool_protocol.rs`.
- `parse_args` xử lý JSON-object args và fallback plain-string.

### Context window quản lý theo budget
- `compress_transcript_to_budget(transcript, budget)` — fold các turn cũ thành một
  `System` summary compact, giữ lại những content mới nhất.
- Anthropic adapter **giữ** các System message không phải đầu tiên (map thành role user,
  không bị filter drop) — quan trọng để summary compress không mất.

### Retry / backoff
- `retry_generate` trong `AgentLoop`: transient error thì backoff rồi thử lại; báo qua
  `ToolObserver::on_retry(secs)`.

### Streaming output (realtime)
- `LlmPort::generate_stream(..., StreamSink)`; `StreamSink = Arc<dyn Fn(StreamEvent) + Send + Sync>`.
  Mặc định delegate về `generate` (vẫn object-safe); provider hỗ trợ override.
- Anthropic: SSE `message_start` / `content_block_delta(text_delta)` /
  `content_block_stop(tool_use)` / `message_delta`.
- OpenAI: SSE `choices[0].delta.content` + `delta.tool_calls[]` (BTreeMap theo index), `include_usage`.
- CLI: one-shot bật streaming — text in gõ dần; TUI nhận final text.

### Ctrl-C cancel (TUI)
- `tui::run` nhận thêm callback `cancel_turn`; khi `state.processing` và bấm Ctrl-C →
  abort `JoinHandle` của turn đang chạy + `ApprovalGate::cancel_all()` để thả worker đang
  chờ approval. Hiển thị dòng `⛔ cancelled`, quay lại prompt. Ctrl-C khi idle → thoát app.

## Những việc/chỗ còn hở

- (trống — các gap trước đã đóng; thêm mục mới tại đây khi phát hiện)

## Config nâng cao (`~/.harxes/config.json`)

- `commands.allow` / `commands.deny`: rule allow/deny cho Bash tool, match theo
  wildcard `*` hoặc word-boundary prefix (vd `"cargo *"`, `"git status"`).
  Deny thắng allow; allow bỏ qua approval prompt cho lệnh nguy hiểm; deny từ
  chối thẳng không hỏi.
- `pricing`: override giá ước tính — map substring của model id →
  `{ "input_per_mtok": USD, "output_per_mtok": USD }`.
- `mcp`: MCP servers (stdio) — map tên server →
  `{ "command": "...", "args": [...], "env": {...} }`. Tool xuất hiện với model
  dưới tên `mcp__<server>__<tool>`; `/mcp` trong REPL liệt kê trạng thái.
  Adapter ở `crates/infrastructure/mcp` (JSON-RPC line-delimited, initialize →
  tools/list → tools/call, lỗi kết nối không fatal).

- `hooks`: lifecycle hooks quanh tool execution —
  `{ "pre_tool": [{"match": "Bash", "command": "..."}], "post_tool": [...] }`.
  Hook nhận `HARXES_TOOL_NAME`/`HARXES_TOOL_ARGS` qua env; pre_tool exit code 2
  chặn tool call (output thành lý do cho model thấy); post_tool stdout được nối
  vào tool result làm feedback. `match` hỗ trợ wildcard `*`.

## Fetch tool

Tool `Fetch` (adapter `crates/infrastructure/web`, port `WebPort`): HTTP GET,
HTML→text (bỏ script/style, decode entity, gộp whitespace), cap 20k chars,
chỉ nhận http(s). Read-only nên được chạy song song cùng Read/Grep/Glob.

## System prompt

`build_system_prompt()` trong `crates/cli/src/main.rs`: identity + môi trường
(cwd, OS, git branch) + kỷ luật làm việc (plan bằng Todo tool trước, verify
bằng build/test trước khi báo xong, ưu tiên tool chuyên dụng, an toàn lệnh
destructive). Dùng chung cho REPL và one-shot; workspace context (HARXES.md,
NOTES.md) được nối vào sau.

(Đã đóng: TUI live streaming — commit 62304cb; Bash child kill khi abort — process
group + RAII guard trong `infrastructure/shell`; `reasoning_content` — fallback khi
content rỗng trong adapter OpenAI; Delegate sub-agent hiển thị nested `└ Tool`
qua `NestedObserver`; cost/usage report per-model trong `/cost` + khi thoát phiên;
đã smoke-test tool-calling loop + delegate live qua LiteLLM.)

## Nguyên tắc xuyên suốt

- Mỗi phase ra binary chạy được; có test đơn vị cho core-domain/app.
- `core-domain` sạch dependency infra; `app` phụ thuộc port; `infrastructure/*` (`harxes-infra-*`) implement port.
- Guardrails an toàn (iteration cap, token cap) được duy trì trong AgentLoop.
