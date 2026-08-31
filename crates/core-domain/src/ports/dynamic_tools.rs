//! Driven port (hexagonal): a provider of dynamically-discovered tools the
//! agent can call in addition to its built-ins — e.g. tools exposed by MCP
//! servers. Specs are merged into the tool list offered to the model; calls
//! are routed here when the name is not a built-in.

use async_trait::async_trait;

use crate::domain::value_objects::ToolSpec;

/// A source of runtime-discovered tools.
#[async_trait]
pub trait DynamicToolPort: Send + Sync {
    /// The tool specifications to offer the model (names must not collide
    /// with built-ins; prefix by origin, e.g. `mcp__server__tool`).
    fn specs(&self) -> Vec<ToolSpec>;
    /// True when `name` belongs to this provider.
    fn owns(&self, name: &str) -> bool;
    /// Execute the named tool with raw JSON arguments, returning the result
    /// text to fold into the transcript (errors as readable text, not Err).
    async fn call(&self, name: &str, arguments: &str) -> String;
}
