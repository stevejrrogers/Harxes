//! Driven port (hexagonal): lets the outer layers observe tool execution as it
//! happens so the UI can render live feedback (e.g. "⏺ Bash(cmd)") while the
//! agent runs, instead of only seeing the final text.

/// A sink for live tool-execution events.
pub trait ToolObserver: Send + Sync {
    /// Called just before a tool starts executing.
    fn on_tool_start(&self, name: &str, args_preview: &str);
    /// Called with a short summary of the result after execution.
    fn on_tool_result(&self, name: &str, result_preview: &str);
    /// Called after a tool finishes with richer telemetry: how long it took and
    /// whether it succeeded. Defaults to [`Self::on_tool_result`] so existing
    /// observers keep working; the agent loop always calls this.
    fn on_tool_end(&self, name: &str, result_preview: &str, _duration_ms: u64, _ok: bool) {
        self.on_tool_result(name, result_preview);
    }
    /// Called once per agent-loop iteration with cumulative token usage, so a
    /// UI can show progress toward the iteration/token guardrails. Default: no-op.
    fn on_iteration(&self, _n: usize, _input_tokens: u64, _output_tokens: u64) {}
    /// Called when a transient LLM failure triggers a backoff wait before
    /// retrying. `wait_secs` is how long we will sleep.
    fn on_retry(&self, _wait_secs: u64) {}

    /// Called incrementally with chunks of hidden reasoning as a reasoning
    /// model thinks. Default: ignore.
    fn on_reasoning(&self, _text: &str) {}

    /// Called incrementally with chunks of assistant text as the model streams
    /// its reply (used for realtime feedback). Empty by default; an
    /// implementation may echo the text or buffer it for a live view.
    fn on_stream_delta(&self, _text: &str) {}
}
