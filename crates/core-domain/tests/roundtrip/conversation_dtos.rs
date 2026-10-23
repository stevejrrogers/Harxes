//! Round-trip tests for the agent-conversation DTOs (CXA-B305a, group 1):
//! message, tool-call and todo value objects that cross the provider /
//! session-store wire.

use harxes_core_domain::domain::value_objects::{
    ImageData, Message, Role, TodoItem, TodoStatus, ToolCall,
};

use crate::expect_round_trip;

#[test]
fn roundtrip_role_all_variants() {
    for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
        expect_round_trip("Role", &role);
    }
}

#[test]
fn roundtrip_message_plain_user() {
    expect_round_trip("Message(user)", &Message::new(Role::User, "plain text turn"));
}

#[test]
fn roundtrip_message_assistant_with_tool_calls() {
    let msg = Message::assistant_with_tools(vec![ToolCall {
        id: "call_01".into(),
        name: "Bash".into(),
        arguments: r#"{"command":"cargo test"}"#.into(),
    }]);
    expect_round_trip("Message(assistant+tools)", &msg);
}

#[test]
fn roundtrip_message_tool_result_with_call_id() {
    expect_round_trip(
        "Message(tool-result)",
        &Message::tool_result("call_01", "exit=0 all green"),
    );
}

#[test]
fn roundtrip_message_user_with_images() {
    let images = vec![
        ImageData { media_type: "image/png".into(), base64: "aGVsbG8=".into() },
        ImageData { media_type: "image/jpeg".into(), base64: "d29ybGQ=".into() },
    ];
    expect_round_trip(
        "Message(user+images)",
        &Message::user_with_images("describe this", images),
    );
}

#[test]
fn roundtrip_tool_call_full() {
    let call = ToolCall {
        id: "call_42".into(),
        name: "Read".into(),
        arguments: r#"{"path":"src/lib.rs","offset":1}"#.into(),
    };
    expect_round_trip("ToolCall", &call);
}

#[test]
fn roundtrip_todo_item_every_status() {
    for status in [TodoStatus::Pending, TodoStatus::InProgress, TodoStatus::Completed] {
        let item = TodoItem {
            label: format!("item-{status:?}"),
            active_form: "working on it".into(),
            status,
        };
        expect_round_trip("TodoItem", &item);
    }
    // Default empty active_form exercises `#[serde(default)]`.
    expect_round_trip("TodoItem(bare)", &TodoItem::new("bare label"));
}
