//! Request-side wire invariants (CXA-B287b / CXA-B317): which fields of each
//! inbound wire type are REQUIRED (absent ⇒ decode fails) vs OPTIONAL
//! (absent ⇒ sensible default). Pure serde functions over real domain types —
//! no IO, no server.
//!
//! Policy asserted here centrally: the wire surface does NOT set
//! `deny_unknown_fields`; configs/session files must survive forward
//! minor-version additions (see `unknown_field_policy`). Every type that adds
//! an optional field MUST pair it with `#[serde(default)]`.

use serde_json::json;

use harxes_core_domain::ports::HarxesConfig;
use harxes_core_domain::domain::value_objects::{Message, TodoItem};

/// Required-field absence must fail deserialization: `{}` lacks every
/// non-defaulted field of [`Message`] (`role`, `content`).
#[test]
fn message_required_fields_absent_fails() {
    let err = serde_json::from_str::<Message>("{}").unwrap_err();
    assert!(
        !err.to_string().is_empty(),
        "decoder must explain the missing field"
    );
}

/// Optional fields deserialize fine when omitted: `tool_calls` carries
/// `#[serde(default)]` and a plain text turn omits it on the wire.
#[test]
fn message_optional_tool_calls_omitted_ok() {
    let msg: Message =
        serde_json::from_value(json!({ "role": "user", "content": "hi" })).expect("omit tool_calls");
    assert!(msg.tool_calls.is_empty());
    assert_eq!(msg.content, "hi");
}

/// [`TodoItem`]: `label` and `status` are required; omitting either fails…
#[test]
fn todo_item_required_fields_absent_fail() {
    assert!(serde_json::from_value::<TodoItem>(json!({ "status": "pending" })).is_err());
    assert!(serde_json::from_value::<TodoItem>(json!({ "label": "x" })).is_err());
}

/// …while `active_form` is optional (`default`) and falls back to empty.
#[test]
fn todo_item_optional_active_form_omitted_ok() {
    let item: TodoItem = serde_json::from_value(json!({
        "label": "Run tests", "status": "in_progress"
    }))
    .expect("active_form is optional");
    assert_eq!(item.active_form, "");
}

/// Provider entries: all four identity/routing fields are required config.
#[test]
fn provider_config_required_fields_absent_fails() {
    let src = r#"{"id":"a","base_url":"https://x","default_model":"m"}"#;
    assert!(
        serde_json::from_str::<harxes_core_domain::ports::ProviderConfig>(src).is_err(),
        "api_key_env is required (no env-var-less silent provider)"
    );
}

/// Lifecycle hooks: `command` is required; the `matcher` key is renamed and
/// optional.
#[test]
fn hook_rule_command_required_matcher_optional() {
    let with_default: harxes_core_domain::ports::HookRule =
        serde_json::from_value(json!({ "command": "echo hi" })).expect("matcher defaults");
    assert_eq!(with_default.matcher, "");
    assert!(
        serde_json::from_str::<harxes_core_domain::ports::HookRule>("{}").is_err(),
        "a hook without a command is meaningless"
    );
}

/// MCP servers: `command` required, `args`/`env` optional.
#[test]
fn mcp_server_config_command_required_args_optional() {
    let cfg: harxes_core_domain::ports::McpServerConfig =
        serde_json::from_value(json!({ "command": "uvx" })).expect("args/env optional");
    assert!(cfg.args.is_empty() && cfg.env.is_empty());
}

/// Pricing rows are fully required — absent numbers would silently price at 0.
#[test]
fn model_pricing_fields_required() {
    assert!(serde_json::from_str::<harxes_core_domain::ports::ModelPricing>(
        "{\"input_per_mtok\":1.0}"
    )
    .is_err());
}

/// Session records: id/timestamp/transcript required; todos snapshot optional
/// (pre-planner sessions omit it).
#[test]
fn session_record_transcript_required_todos_optional() {
    use harxes_core_domain::ports::SessionRecord;
    assert!(serde_json::from_str::<SessionRecord>("{}").is_err());
    let rec: SessionRecord = serde_json::from_value(json!({
        "id": "s1",
        "created_at": "2026-02-14T00:00:00Z",
        "transcript": [{ "role": "user", "content": "hello" }]
    }))
    .expect("todos snapshot is optional");
    assert_eq!(rec.transcript.len(), 1);
    assert!(rec.todos.is_empty());
}

/// Whole-config document: `{}` is a VALID first-run config — every top-level
/// field is `#[serde(default)]`. Absence means "unset", not corruption.
#[test]
fn empty_config_document_decodes_to_defaults() {
    let cfg: HarxesConfig = serde_json::from_str("{}").expect("first-run config");
    assert!(cfg.active_provider.is_none());
    assert!(cfg.providers.is_empty());
    // All five limit overrides are Option and default to unset (None serializes
    // as explicit null keys — documented wire shape, not an accident).
    let limits: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&serde_json::to_string(&cfg.limits).unwrap()).unwrap();
    for (_k, v) in &limits {
        assert!(v.is_null(), "first-run limit {_k} must be unset");
    }
    assert_eq!(limits.len(), 5);
}

/// CENTRAL unknown-field policy assertion (one place, per ticket): forward
/// compatibility wins — unknown keys are tolerated, never rejected, on the
/// two documents users hand-edit or persist across versions.
#[test]
fn unknown_field_policy_forward_compatible() {
    let cfg: Result<HarxesConfig, _> =
        serde_json::from_str(r#"{"some_future_key": true}"#);
    assert!(cfg.is_ok(), "config must tolerate unknown future keys");

    let msg: Result<Message, _> = serde_json::from_value(json!({
        "role": "user", "content": "hi", "future_field": 1
    }));
    assert!(msg.is_ok(), "messages must tolerate unknown future keys");
}
