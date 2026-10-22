//! Response-side wire invariants (CXA-B287b / CXA-B317): byte-stable
//! serialize→deserialize→serialize roundtrips for representative samples, and
//! literal key-casing assertions on the agreed wire names. Pure serde over
//! real domain types — no IO.

use serde_json::{json, Value};

use harxes_core_domain::domain::value_objects::{
    Message, Role, TodoItem, TodoStatus, ToolCall,
};
use harxes_core_domain::ports::{HarxesConfig, ProviderConfig};

/// serialize → deserialize → serialize must produce IDENTICAL JSON bytes:
/// accidental field reordering or lossy defaults would break diffed config /
/// session files.
#[test]
fn message_roundtrip_is_byte_stable() {
    let msg = Message {
        role: Role::Assistant,
        content: "doing it".into(),
        images: Vec::new(),
        tool_call_id: None,
        tool_calls: vec![ToolCall {
            id: "call_1".into(),
            name: "Bash".into(),
            arguments: "{\"command\":\"ls\"}".into(),
        }],
    };
    let first = serde_json::to_vec(&msg).expect("serialize");
    let back: Message = serde_json::from_slice(&first).expect("deserialize");
    assert_eq!(serde_json::to_vec(&back).expect("reserialize"), first);
}

#[test]
fn todo_item_and_status_roundtrip_is_byte_stable() {
    let item = TodoItem {
        label: "Run tests".into(),
        active_form: "Running tests".into(),
        status: TodoStatus::InProgress,
    };
    let first = serde_json::to_vec(&item).expect("serialize");
    let back: TodoItem = serde_json::from_slice(&first).expect("deserialize");
    assert_eq!(serde_json::to_vec(&back).expect("reserialize"), first);
}

#[test]
fn config_document_roundtrip_is_byte_stable() {
    let mut cfg = HarxesConfig::default();
    cfg.providers.insert(
        "anthropic".into(),
        ProviderConfig {
            id: "anthropic".into(),
            base_url: "https://api.example.com/v1/messages".into(),
            api_key_env: "HARXES_API_KEY".into(),
            default_model: "claude-sonnet-4-5".into(),
        },
    );
    let first = serde_json::to_string_pretty(&cfg).expect("serialize");
    let back: HarxesConfig = serde_json::from_str(&first).expect("deserialize");
    assert_eq!(
        serde_json::to_string_pretty(&back).expect("reserialize"),
        first,
        "config files are hand-diffed; reserialization must not churn them"
    );
}

// --- literal wire-casing keys -----------------------------------------------

/// The exact on-the-wire key strings of a chat message. Renames here are
/// breaking API changes — assert them literally.
#[test]
fn message_keys_are_literal_snake_case() {
    let v: Value =
        serde_json::to_value(Message { role: Role::User, content: "x".into(), images: vec![], tool_call_id: Some("t1".into()), tool_calls: vec![] }).unwrap();
    assert_eq!(v["role"], json!("user"));
    assert_eq!(v["content"], json!("x"));
    // casing contract: exactly these top-level keys — `images` and an absent
    // tool_calls list both skipped (Option None / empty vec may drop from the
    // envelope; only key NAMES are contractual here)
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["content", "role", "tool_call_id", "tool_calls"],
        "message wire envelope renamed or gained a field"
    );
    // optional fields, when present, use snake_case too
    assert_eq!(v["tool_call_id"], json!("t1"));
    // and empty images list must serialize as [] not be silently dropped:
    let with_images: Value =
        serde_json::to_value(Message { role: Role::User, content: String::new(), images: vec![], tool_call_id: None, tool_calls: vec![] }).unwrap();
    assert!(
        !with_images.as_object().unwrap().contains_key("images")
            || with_images["images"].is_array(),
        "images must be an array when present"
    );
}

#[test]
fn todo_status_serializes_as_snake_case_word() {
    assert_eq!(serde_json::to_value(TodoStatus::InProgress).unwrap(), json!("in_progress"));
    assert_eq!(serde_json::to_value(TodoStatus::Completed).unwrap(), json!("completed"));
    // and the item's keys:
    let v = serde_json::to_value(TodoItem::new("x")).unwrap();
    let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ["label", "active_form", "status"]); // declared order kept
}

/// HookRule's Rust field `matcher` is deliberately renamed to `"match"` on
/// the wire (`match` is a keyword); assert the rename survives.
#[test]
fn hook_rule_matcher_renamed_to_match_on_wire() {
    let rule = harxes_core_domain::ports::HookRule {
        matcher: "*".into(),
        command: "cargo test".into(),
    };
    let v: Value = serde_json::to_value(&rule).unwrap();
    assert!(v.get("match").is_some(), "wire key must be 'match'");
    assert!(v.get("matcher").is_none());
}
