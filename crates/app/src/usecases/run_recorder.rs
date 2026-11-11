//! Pure run-summary recorder: a fold over collected run telemetry that
//! accumulates a [`SweepRunSummary`] while the agent loop executes.
//!
//! Zero IO by construction (hexagonal: no `std::fs`, no `std::process`, no
//! clock access — durations are supplied by the caller from whatever monotonic
//! source the adapter layer owns). Persistence in part 2 will go through an
//! outbound port; this module stays the IO-free accumulation core.

use harxes_core_domain::domain::value_objects::{
    FinishedSummary, RunAttribution, RunEventTelemetry, SweepRunSummary,
};

/// Error returned when finishing a recorder that already finished, or whose
/// timing contract was violated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecorderFinishError(pub &'static str);

impl std::fmt::Display for RecorderFinishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for RecorderFinishError {}

/// A single-run accumulator. Pure value semantics: feed it telemetry with
/// [`RunRecorder::observe`], freeze it with [`RunRecorder::finish`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunRecorder {
    state: SweepRunSummary,
}

impl RunRecorder {
    /// Start recording a run attributed to `attribution`.
    pub fn new(attribution: RunAttribution) -> Self {
        Self { state: SweepRunSummary::new(attribution) }
    }

    /// Fold one telemetry item in. Total function — never errors, ignores
    /// post-finish stragglers (late events from a drained channel).
    pub fn observe(&mut self, ev: &RunEventTelemetry) -> &mut Self {
        self.state.observe(ev);
        self
    }

    /// Still accumulating?
    pub fn is_recording(&self) -> bool {
        self.state.is_recording()
    }

    /// Freeze the accumulated aggregates into a finished summary.
    ///
    /// `end_elapsed_ms` is the run's total wall-clock/monotonic duration as
    /// measured by the caller; `guardrail_stop` records whether a guardrail
    /// (iteration/token cap) ended the run instead of normal completion;
    /// `outcome_iterations` lets the authoritative final iteration count from
    /// the outcome override the event-derived count.
    pub fn finish(
        self,
        end_elapsed_ms: u64,
        guardrail_stop: bool,
        outcome_iterations: Option<usize>,
    ) -> Result<FinishedSummary, RecorderFinishError> {
        self.state
            .finish(end_elapsed_ms, guardrail_stop, outcome_iterations, None)
            .map(|s| match s {
                SweepRunSummary::Finished(f) => f,
                // unreachable by construction: finish() from Recording yields Finished
                SweepRunSummary::Recording(_) => {
                    unreachable!("finish on Recording must yield Finished")
                }
            })
            .map_err(|e| RecorderFinishError(e.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat() -> RunAttribution {
        RunAttribution::chat("user:test")
    }

    #[test]
    fn mixed_ok_failed_tools_reconcile_by_tool_counts() {
        let mut r = RunRecorder::new(chat());
        r.observe(&RunEventTelemetry::ToolEnd { tool: "Bash".into(), duration_ms: 12, ok: true });
        r.observe(&RunEventTelemetry::ToolEnd { tool: "Edit".into(), duration_ms: 30, ok: false });
        r.observe(&RunEventTelemetry::ToolEnd { tool: "Bash".into(), duration_ms: 8, ok: true });
        r.observe(&RunEventTelemetry::Iteration { n: 4 });
        r.observe(&RunEventTelemetry::Retry { wait_secs: 2 });
        let f = r.finish(120, false, Some(4)).unwrap();
        assert_eq!(f.tool_counters.total, 3);
        assert_eq!(f.tool_counters.ok, 2);
        assert_eq!(f.tool_counters.failed, 1);
        assert_eq!(f.tool_counters.by_tool_ok.get("Bash"), Some(&2));
        assert_eq!(f.tool_counters.by_tool_failed.get("Edit"), Some(&1));
        assert_eq!(f.tool_counters.max_duration_ms, 30);
        assert_eq!(f.retries, 1);
        assert_eq!(f.iterations, 4);
    }

    #[test]
    fn clean_run_has_no_attributed_cause() {
        let mut r = RunRecorder::new(chat());
        r.observe(&RunEventTelemetry::ToolEnd { tool: "Read".into(), duration_ms: 1, ok: true });
        let f = r.finish(50, false, Some(1)).unwrap();
        assert!(f.attribution_cause.is_none());
        assert!(!f.guardrail_stop);
    }

    #[test]
    fn failed_and_guardrail_stopped_runs_are_attributed() {
        let mut r = RunRecorder::new(RunAttribution::scheduled("cron:sweep"));
        r.observe(&RunEventTelemetry::ToolEnd { tool: "Grep".into(), duration_ms: 5, ok: false });
        let f = r.finish(300, true, Some(7)).unwrap();
        let cause = f.attribution_cause.expect("non-clean run carries a cause");
        assert_eq!(cause.rule, "Grep");
        assert!(f.guardrail_stop);
        assert_eq!(f.attribution.trigger.as_str(), "scheduled_sweep");
    }

    #[test]
    fn empty_run_records_zero_totals_without_events() {
        let r = RunRecorder::new(chat());
        let f = r.finish(0, false, Some(0)).unwrap();
        assert_eq!(f.tool_counters.total, 0);
        assert_eq!(f.iterations, 0);
        assert_eq!(f.timing.duration_elapsed_ms, 0);
        assert!(f.attribution_cause.is_none());
    }

    #[test]
    fn events_after_finish_are_ignored_not_panics() {
        let mut done = RunRecorder::new(chat());
        let finished = done.clone().finish(10, false, None).unwrap();
        assert!(finished.attribution_cause.is_none());
        // A finished summary ignores late-arriving stragglers without panicking.
        done.observe(&RunEventTelemetry::ToolEnd { tool: "Bash".into(), duration_ms: 1, ok: true });
        assert_eq!(finished.tool_counters.total, 0, "straggler did not mutate frozen aggregates");
    }

    #[test]
    fn nested_delegate_depth_tracked_as_max_once_at_top_level() {
        let mut r = RunRecorder::new(chat());
        r.observe(&RunEventTelemetry::DelegationNested { depth: 1 });
        r.observe(&RunEventTelemetry::DelegationNested { depth: 3 });
        let f = r.finish(20, false, None).unwrap();
        assert_eq!(f.delegation_depth_max, 3);
    }
}
