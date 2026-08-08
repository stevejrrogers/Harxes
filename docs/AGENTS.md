# Harxes — Dev Notes for Future Agents

> Trạng thái dự án + hướng đi tiếp. Cập nhật khi có milestone mới.

## TL;DR (đứng giữa Phase 0 → Phase 1)

`harxes` là CLI coding agent (như Claude Code) viết bằng Rust, kiến trúc hexagonal
qua Cargo workspace. **Repo chưa có commit đầu tiên** (branch `main` rỗng) — mọi file
đang ở trạng thái added/staged từ lần tạo repo ban đầu.

## Hiện trạng cụ thể

- **core-domain / app / infra-auth / infra-shell / infra-session**: baseline hoàn chỉnh và compile sạch (`cargo build`, ~77 crates).
- **infra-llm**: đang viết dở adapter LLM trong `crates/infra-llm/src/providers/`
  - Thư mục này là **untracked**.
  - `lib.rs` hiện chỉ có doc comment, **chưa khai báo `pub mod providers;`**
    → toàn bộ code trong `providers/*.rs` KHÔNG được compile hiện tại.
  - `anthropic.rs`: gần hoàn chỉnh nhưng chưa tham gia module tree.
  - `openai.rs`: **bị cắt ngang** (kết thúc giữa hàm ở cuối file, thiếu phần parse response) — cần viết lại phần đuôi.
- Mâu thuẫn đường dẫn chưa xử lý: `ProviderDescriptor.base_url` ở registry đã chứa full path
  (`.../v1/messages`, `.../v1/chat/completions`) nhưng adapter Anthropic tự append `/v1/messages`,
  adapter OpenAI tự append `/home/v1/...`. Nếu wire thẳng sẽ ra double-path → cần chọn một chuẩn
  duy nhất (khuyến nghị: adapter nhận thẳng full endpoint URL, không tự nối thêm path).

## Hướng đi tiếp (Phase làm trong session này)

Theo `docs/ROADMAP.md`:

### Phase 1 — REPL + one-shot prompt
1. Hoàn thiện adapter LLM Anthropic + OpenAI qua HTTP (`reqwest`) — map roles ↔ provider format,
   parse content + token usage, timeout/xử lý lỗi (RateLimited/Auth).
2. Bật module: thêm `pub mod providers;` vào `lib.rs`.
3. Composition root trong `cli/main.rs`: vault env → registry → llm adapter → AgentRunner builder
   (hiện main chỉ in banner "skeleton built successfully.").
4. CLI args bằng clap: subcommand nhận prompt one-shot.
5. Streaming output kiểu gõ dần + in token usage cuối lượt.

Demo target:
```bash
$ harxes "Giải thích hexagon pattern ngắn gọn"
harxes > [streaming...] Hệ hexagonal...
usage ~= in 500 / out 120 · anthropic/claude-sonnet-4-5
```

### Lưu ý kỹ thuật khi làm tiếp
- Port chi phối là `LlmPort::generate()` ở core-domain; muốn streaming realtime cần quyết định cách
  truyền callback/Sink loạt delta mà vẫn giữ object-safe (`Arc<dyn LlmPort>`).
- Đảm bảo compile clean sau từng bước.

## Nguyên tắc xuyên suốt (trích ROADMAP)
- Mỗi phase ra binary chạy được; có test đơn vị cho core-domain/app.
- core-domain sạch dependency infra; app phụ thuộc port, infra implement port.
- Sau Phase 2 luôn có guardrails an toàn (iteration cap, token cap).

---

*Note này được tạo vì các output trước đó của agent liên tiếp bị hỏng/thành văn bản rác —
thêm một file note để future agents nắm trạng thái mà không cần re-derive.*
