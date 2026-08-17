# Harxes — Roadmap: xịn hơn Claude Code (+ LiteLLM)

> Mục tiêu: biến `harxes` thành CLI coding agent **vượt trội** Claude Code.
> Điểm khác biệt cốt lõi: **provider-agnostic + LiteLLM proxy support**, token manager,
> session replay, TUI, guardrails an toàn. Kiến trúc hexagonal giữ nguyên qua Cargo workspace.

## Nguyên tắc xuyên suốt
- Mỗi milestone kết thúc bằng **binary chạy được** (`cargo run`).
- `core-domain` SẠCH dependency infra; `app` chỉ phụ thuộc port; `infrastructure/*` (`harxes-infra-*`) implement port.
- Docs đi kèm mỗi milestone vào `docs/{architecture,guides,patterns}`.
- **Chuẩn base_url đã chốt:** adapter nhận **full endpoint URL** (không tự append path).
  Registry/LiteLLM descriptor chứa full path; adapter dùng thẳng, không nối thêm `/v1/...`.

---

## Milestone 0 — Hoàn thiện baseline (fix double-path + infrastructure/llm)
Trước khi có binary chạy được, phải làm cho workspace compile sạch toàn bộ.

1. **Fix double-path base_url**
   - Registry (`provider_registry.rs`) đã chứa full path (`.../v1/messages`, `.../v1/chat/completions`).
   - Adapter Anthropic tự append `/v1/messages`, OpenAI tự append → double path.
   - **Sửa:** adapter nhận full URL từ registry/config; xoá đoạn tự nối path trong
     `AnthropicClient::url()` / `OpenAiClient::url()`. Chỉ giữ base host trim nếu cần.

2. **Hoàn thiện infrastructure/llm**
   - Thêm `pub mod providers;` vào `crates/infrastructure/llm/src/lib.rs`.
   - Viết lại phần đuôi cắt ngang của `openai.rs` (parse ResponseBody -> AgentResponse + TokenUsage),
     align với pattern anthropic.rs.
   - Kiểm tra map role cả 2 provider.

3. **Unit test core-domain/app**
   - Test value objects (`Message`, `ModelId`, `ProviderId`) — validation fail/success.
   - Test usecase dummy LLM (fake impl LlmPort) cho success + EmptyResponse + LlmError paths.

**Done:** `cargo build && cargo test` xanh toàn bộ workspace.

---

## Milestone 1 — REPL one-shot prompt streamed
CLI wiring composition root, gõ prompt là trả lời streamed.

1. **CLI args bằng clap**
   - Subcommand mặc định nhận prompt one-shot: `harxes "prompt"`.
   - Option: `--provider <id>`, `--model <id>`, config file override.

2. **Composition root trong cli/main.rs**
   - Vault env (`EnvSecretsVault`) -> đọc key theo active provider -> dựng adapter
     (Anthropic/OpenAI) -> bọc trong registry factory -> dựng AgentRunner builder.
   - Config store JSON (`~/.harxes/config.json`) quyết định active provider/model/base_url/key env.

3. **Streaming output**
   - Phải quyết định cách streaming với object-safe trait (`Arc<dyn LlmPort>`).
     Đề xuất: thêm method streaming trên LlmPort hoặc Sink-based callback để in token
     gõ dần. Nếu LLM API hỗ trợ SSE thì stream thật; tối thiểu in toàn cục + usage cuối lượt.
   - In token usage cuối lượt:
     ```
     usage ~= in 500 / out 120 · anthropic/claude-sonnet-4-5
     ```

**Done:** gõ prompt là nhận câu trả lời streamed; compile clean.

---

## Milestone 2 — Tool-calling loop (đúng nghĩa coding agent)
Bước lớn nhất để "giống Claude Code".

1. **Domain mở rộng capabilities & tool role result**
   - Mở rộng message Role/Tool với content tool_result (đã có ToolKind Bash/Read/Write/Glob).
   - Thêm capability-profile cho provider phân loại tool-use format (Anthropic vs OpenAI).

2. **Registry thành strategy factory động theo capability-profile**
   - Chuyển static ProviderRegistry sang factory có capability detection:
     provider nào hỗ trợ tool-use, format tool-call ra sao.

3. **Adapter shell hoàn chỉnh** (infrastructure/shell implement ShellPort)
   - Spawn `/bin/sh -c <cmd>` qua tokio process; stream stdout/stderr realtime;
     timeout configurable; return CommandOutput{stdout,stderr,exit_status}.

4. **Adapter filesystem hoàn chỉnh** (infrastructure/fs implement FileSystemPort)
   - Read/write file với error mapping chuẩn FsError (NotFound/PermissionDenied/Io).

5. **Usecase agent-loop** trong app layer:
```
user_prompt -> llm(tools declared) -> parse tool_use ->
  execute qua ShellPort/FsPort ->
  append <tool_result> Message(Role::Tool) ->
  llm tiếp ... (max N vòng)
done khi stop reason text-only / max iterations đạt
```

6. **Guardrails controller ở app layer**
   - MaxIterations (N=25 default), total-token cap → ngừng loop an toàn sau ngưỡng.
   - Không bao giờ agent loop không biên độ.

**Done:** một task nhiều bước hoàn thành tự động bằng bash/read/write;
in step-by-step UI ra terminal theo từng bước agent thực hiện.

---

## Milestone 3 — LiteLLM proxy support (+ multi-provider routing)
Điểm "xịn hơn" Claude Code về tính mở provider: không khóa cứng vào Anthropic/OpenAI,
cho phép route qua LiteLLM và mọi OpenAI-compatible endpoint/local model.

1. **Hướng tích hợp đã chốt:** tự implement + option dùng LiteLLM proxy —
   Harxes vẫn có adapter native Anthropic/OpenAI NHƯNG config cho phép điểm tới một endpoint
   tổng quát ("LiteLLM mode"), request format OpenAI-compatible tới proxy đó.
2. **Core-domain:** descriptor/provider profile tổng quát hóa — id không chỉ `anthropic`/`openai`
   mà còn custom như `litellm`, `local`. Config store load providers `BTreeMap` sẵn hỗ trợ
   (`ProviderConfig { id, base_url, default_model, api_key_env }`). Khớp chuẩn base_url = full URL.
3. **Adapter tổng quát OpenAI-compatible** bao phủ LiteLLM & local model server,
   cùng native Anthropic adapter riêng khi cần feature riêng của Anthropic.
4. **Router/factory** quyết định chọn adapter theo `active_provider` config +
   khả năng fallback nhiều provider khi rate-limit/lỗi auth.
5. **Docs pattern `add-provider.md`:** thêm provider = thêm ProviderConfig trong config.json,
   không cần sửa code (trừ khi cần feature riêng của provider).

**Done:** dùng model local/LiteLLM như native Claude/GPT; switch `--provider litellm --model llama3...`.

---

## Milestone 4 — Focus "xịn hơn" Claude Code (differencing features)
Bổ sung các tính năng làm harxes nổi bật so với Claude Code:

- **Permission system:** white/black-list paths qua config, confirm trước op viết/xoá ngoài allowed list.
- **Glob/grep tốc độ cao** cho SearchFiles/GrepContents tool.
- **Context/token manager:** truncate/summarize turn cũ khi vượt budget (chìa khoá xử lý long context).
- **Session persist/replay transcripts,** multi-turn continue sau restart.

Tiếp tục theo ROADMAP Phase 4+5 cho retry/cancel/diff-review/TUI rich rendering.

---

## Trạng thái triển khai (sẽ update)
| Milestone | Trạng thái |
|---|---|
| M0 Fix baseline + infrastructure/llm | ✅ DONE |
| M1 REPL one-shot streamed | ✅ DONE |
| M2 Tool-calling loop + guardrails | ✅ DONE |
| M3 LiteLLM proxy support | ✅ DONE |
| M4 Xịn hơn CC (permission/token/session) | ✅ DONE |

## Ghi chú triển khai
- **20 unit test** pass toàn workspace, clippy clean, `cargo build` xanh.
- Thêm crate mới: `harxes-infra-fs` (`crates/infrastructure/fs`, `HostFileSystem` implement `FileSystemPort`).
- `infrastructure/shell`: `TokioCommandShell` spawn `/bin/sh -c <cmd>` với timeout.
- `infrastructure/session`: thêm `JsonSessionStore` lưu transcript JSON.
- AgentLoop: multi-turn tool-loop (Bash/Read/Write), guardrails (iteration 25 / token 128k),
  context manager trim turn cũ, permission policy gate write ops, session persist.


