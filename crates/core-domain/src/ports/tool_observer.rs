//! Driven port (hexagonal): lets the outer layers observe tool execution as it
//! happens so the UI can render live feedback (e.g. "⏺ Bash(cmd)") while the
//! agent runs, instead of only seeing the final text.

/// A sink for live tool-execution events.
pub trait ToolObserver: Send + Sync {
    /// Called just before a tool starts executing.
    fn on_tool_start(&self, name: &str, args_preview: &str);
    /// Called with a short summary of the result after execution.
    fn on_tool_result(&self, name: &str, result_preview: &str);
}
