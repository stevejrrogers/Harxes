# Harxes — Long-term Roadmap

Mục tiêu: biến `harxes` từ skeleton thành CLI coding agent xịn như Claude Code — workspace-aware,
an toàn, có tool-calling loop và UX terminal tốt.

Kiến trúc: hexagonal (clean) qua Cargo workspace. Mỗi phase kết thúc bằng binary chạy được,
compile sạch; `core-domain` không phụ thuộc infra.

---

## Tổng quan

Roadmap theo phase dưới đây. Trạng thái hiện tại đã vượt qua hầu hết mục trong
Phase 1–5 (xem `docs/AGENTS.md`): adapter LLM Anthropic+OpenAI hoàn chỉnh (streaming SSE),
tool-calling `AgentLoop` nhiều vòng, permission/approval gate, grep/glob/edit tools,
context compress theo budget, retry/backoff, Ctrl-C cancel, TUI giàu (markdown hiển thị,
status/cost panel, plan checklist, session resume). Phần còn hở được ghi ở AGENTS.md.

## Baseline (hiện tại)

- **core-domain**: entities + value objects — `Message`, `Role`, `ProviderId`, `ModelId`,
  `TokenUsage`, `ContextBudget`; ports — `LlmPort`, `ShellPort`, `FileSystemPort` (read/write/replace/glob/grep),
  `ConfigStorePort`, `SecretsVaultPort`, `SessionStorePort`, `PermissionDecider`, `ToolObserver`
  (on_tool_start/result, on_retry, on_stream_delta).
- **app**: usecase `AgentLoop` — tool-calling loop nhiều vòng, retry/backoff, context compress,
  streaming; `LoopLimits` guardrail (iteration cap, token cap).
- **infrastructure/llm**: adapter Anthropic + OpenAI (`generate` + `generate_stream` SSE).
- **infrastructure/fs**: `HostFileSystem` (read/write/replace/glob/grep, skip dirs lớn).
- **infrastructure/shell**: tokio command shell; **infrastructure/auth** env keys; **infrastructure/session** JSON store.
- **cli**: TUI (ratatui) + one-shot prompt; composition root `compose.rs`.
- Workspace compile sạch, toàn bộ test pass; repo có commit đầy đủ trên `main`.

---

## Phase 1 — REPL chạy được + one-shot prompt

Tasks:
1. Adapter LLM Anthropic + OpenAI bằng HTTP client (`reqwest`) trong `infrastructure/llm`: map roles ↔ provider format; parse content + token usage; timeout.
2. Composition root trong cli/main.rs: vault env → registry → llm adapter → AgentRunner builder.
3. CLI args clap: subcommand mặc định nhận prompt one-shot.
4. Streaming output dạng gõ dần, in token usage cuối lượt.

Demo:
```bash
$ harxes "Giải thích hexagon pattern ngắn gọn"
harxes > [streaming...] Hệ hexagonal...
usage ~= in 500 / out 120 · anthropic/claude-sonnet-4-5
```

Done khi: compile clean, gõ prompt là trả lời streamed.
Dependency mới dự kiến: reqwest + tokio stream sink.

---

## Phase 2 — Tool-calling agent loop

Bước lớn nhất để "giống Claude Code": agent tự quyết định gọi tool nhiều vòng.

Tasks:
1. Domain mở rộng capabilities + message role Tool result; phân loại provider có hỗ trợ tool-use.
2. Registry thành strategy factory động theo capability-profile (format tool call khác nhau Anthropic vs OpenAI).
3. Adapter shell hoàn chỉnh: spawn `/bin/sh -c <cmd>` qua tokio process, stream stdout/stderr realtime, timeout configurable.
4. Adapter filesystem hoàn chỉnh read/write.
5. Usecase agent-loop:

```
user_prompt -> llm(tools declared) -> parse tool_use ->
  execute qua ShellPort/FsPort ->
  append <tool_result> -> llm tiếp ... (max N vòng)
  done khi stop reason text-only
```

6. Guardrails controller: MaxIterations (N=25 default) + total-token cap ở app layer.

Done khi: một task nhiều bước hoàn thành tự động bằng bash/read/write, in step-by-step UI màu/icons ra terminal.

---

## Phase 3 — Workspace awareness & an toàn

Tasks:
1. Permission system: white/black-list paths qua config (`~/.harxes/permissions.*`), confirm với op viết/xoá ngoài allowed list (như `/permission`).
2. Glob/grep helpers tốc độ cao cho SearchFiles/GrepContents.
3. Context/token manager cho cửa sổ ngữ cảnh dài: truncate/summarize các turn cũ khi vượt budget.
4. Session persist/replay transcripts; multi-turn continue khôi phục state sau restart.

Done khi: grep/glob trong repo tốt, không vượt quyền khi chưa confirm, quản lý context trên window lớn, resume session sau thoát app.

---

## Phase 4 — Độ tin cậy & UX pro

Tasks:
1. Retry + backoff cho rate-limit/reset network; resumable mid-turn.
2. Graceful Ctrl-C cancel giữa turn rồi hỏi tiếp tục hay không (không corrupt state).
3. Error categories rõ ràng theo từng lớp nguồn lỗi, hiển thị hữu dụng hơn stack trace thô.
4. TUI renderers giàu hơn: status bar, banner mỗi phase, cost/usage tổng phiên; chuyển sang raw terminal để kiểm soát luồng in.

Done khi: mạng chập chờn không hỏng phiên; cancel sạch sẽ; UI thân thiện với milestone hiển thị progress/cost.

---

## Phase 5 — Native edit/diff & premium feel

Tasks:
1. Edit-tool native (thay những đoạn nhỏ bằng old_string/new_string) + apply diff cục bộ, tự detection số dòng.
2. Diff review bước trước khi agent ghi file quan trọng (`harxes --diff` hoặc confirm hiện golden diff).
3. Mở rộng registry thêm provider khác (e.g. local models qua OpenAI-compatible endpoint).
4. Cân nhắc telemetry/cost report tóm tắt mỗi phiên.

Done khi: sửa file theo từng hunk có review; UX chững chạc như Claude Code.

---

## Nguyên tắc xuyên suốt

- Mỗi phase ra một milestone có binary chạy được, test đơn vị cho core-domain/app.
- `core-domain` giữ sạch không dependency infra; app phụ thuộc port, infra implement port.
- Tài liệu đi kèm mỗi phase vào `docs/{architecture,guides,patterns}` cho đúng chỗ code comment đang trỏ tới
  (`composition-root.md`, `implement-shell.md`, ...).
- Sau Phase 2 trở đi luôn có guardrails an toàn (iteration cap, token cap). Không bao giờ agent loop không biên độ.
