//! Pure sample builders for the wire-surface contract guard tests
//! (CXA-B287a / CXA-B316). Every value here is constructed from REAL domain /
//! port types declared in this crate — no fabricated placeholder structs, no
//! IO. Later parts of CXA-B287 layer invariant assertions on top of these.

use harxes_core_domain::domain::value_objects::{Message, Role};
use harxes_core_domain::domain::value_objects::ProviderId;
use harxes_core_domain::domain::value_objects::ReasoningEffort;
use harxes_core_domain::domain::value_objects::{TodoItem, TodoStatus};
use harxes_core_domain::domain::value_objects::ToolCall;
use harxes_core_domain::ports::{HarxesConfig, LimitsConfig, ProviderConfig};

/// A minimal valid assistant-visible chat message.
pub fn sample_message(text: &str) -> Message {
    Message {
        role: Role::User,
        content: text.to_string(),
        images: Vec::new(),
        tool_call_id: None,
        tool_calls: Vec::new(),
    }
}

/// A tool invocation envelope as produced by the agent loop.
pub fn sample_tool_call(name: &str, args_json: &str) -> ToolCall {
    ToolCall { id: "call_1".to_string(), name: name.to_string(), arguments: args_json.to_string() }
}

/// The single in-progress plan item every todo list must have.
pub fn sample_todo(label: &str) -> TodoItem {
    TodoItem {
        label: label.to_string(),
        status: TodoStatus::InProgress,
        active_form: label.to_string(),
    }
}

/// Default reasoning dial as surfaced to providers and config.
pub fn sample_reasoning_effort() -> ReasoningEffort {
    ReasoningEffort::parse("medium").expect("valid effort")
}

/// A valid provider identity as accepted by every adapter.
pub fn sample_provider_id() -> ProviderId {
    ProviderId::new("anthropic").expect("valid provider id")
}

/// A single configured model provider entry (as persisted in `config.json`).
pub fn sample_provider_config(id: &str) -> ProviderConfig {
    ProviderConfig {
        id: id.to_string(),
        base_url: "https://api.example.com".to_string(),
        api_key_env: "HARXES_TEST_KEY".to_string(),
        default_model: "test-model".to_string(),
    }
}

/// An all-defaults configuration document — the baseline every config
/// migration/guard test starts from.
pub fn sample_config() -> HarxesConfig {
    HarxesConfig {
        limits: LimitsConfig::default(),
        ..HarxesConfig::default()
    }
}
