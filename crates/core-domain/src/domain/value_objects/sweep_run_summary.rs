//! A structured, IO-free record of one agent run's outcome.
//!
//! Pure projection: `observe`/`finish` only consume already-collected data (the
//! engine's run-event stream and the final outcome counters), so the
//! summary is testable without any host or persistence. All types derive serde —
//! part 2 persists this verbatim, so the wire shape here (`trigger` tags
//! `"chat" | "delegate" | "scheduled_sweep" | "api"`) is a contract that must
//! not be renamed casually.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Who/what started the run. Wire tag names are locked by CXA-F438 part 1;
/// renaming after persistence ships would be a migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunTrigger {
    Chat,
    Delegate,
    ScheduledSweep,
    Api,
}

impl RunTrigger {
    /// The stable lowercase wire token for this trigger.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Delegate => "delegate",
            Self::ScheduledSweep => "scheduled_sweep",
            Self::Api => "api",
        }
    }
}

impl std::fmt::Display for RunTrigger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Attribution: which trigger, and which human/system actor fired it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunAttribution {
    pub trigger: RunTrigger,
    /// e.g. `"user:bob"`, `"cron:nightly-sweep"`, `"agent:DEV-BUG"`.
    pub actor: String,
}

impl RunAttribution {
    pub fn chat(actor: impl Into<String>) -> Self {
        Self { trigger: RunTrigger::Chat, actor: actor.into() }
    }

    pub fn scheduled(actor: impl Into<String>) -> Self {
        Self { trigger: RunTrigger::ScheduledSweep, actor: actor.into() }
    }
}

/// Per-outcome counts over all tool invocations in the run.
///
/// Invariant: `ok + failed == calls`, and every entry of `by_tool_ok` +
/// `by_tool_failed` sums to exactly those totals. Upheld by construction —
/// `ToolCounters::new()` takes no arguments and mutators keep the equation,
/// tested at the bottom of this file.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCounters {
    /// Total finished tool executions (regardless of ok/failed).
    pub total: u32,
    /// Finished successfully.
    pub ok: u32,
    /// Failed.
    pub failed: u32,
    /// ok-count per distinct tool name.
    pub by_tool_ok: BTreeMap<String, u32>,
    /// failure-count per distinct tool name.
    pub by_tool_failed: BTreeMap<String, u32>,
    /// Sum of per-tool durations in milliseconds.
    pub total_duration_ms: u64,
    /// Slowest single tool execution, in milliseconds.
    pub max_duration_ms: u64,
}

impl ToolCounters {
    /// Record one finished tool execution.
    pub fn record(&mut self, tool: &str, duration_ms: u64, ok: bool) {
        self.total += 1;
        if ok {
            self.ok += 1;
            *self.by_tool_ok.entry(tool.to_string()).or_insert(0) += 1;
        } else {
            self.failed += 1;
            *self.by_tool_failed.entry(tool.to_string()).or_insert(0) += 1;
        }
        self.total_duration_ms += duration_ms;
        self.max_duration_ms = self.max_duration_ms.max(duration_ms);
    }

    #[cfg(test)]
    pub(crate) fn reconcile_check(&self) -> Result<(), String> {
        if self.ok + self.failed != self.total {
            return Err(format!("{}+{} != {}", self.ok, self.failed, self.total));
        }
        let sum_ok: u32 = self.by_tool_ok.values().sum();
        let sum_fail: u32 = self.by_tool_failed.values().sum();
        if sum_ok != self.ok || sum_fail != self.failed {
            return Err(format!(
                "per-tool sums ({sum_ok}/{sum_fail}) do not reconcile with totals ({}/{})",
                self.ok, self.failed
            ));
        }
        Ok(())
    }
}

/// Wall-clock stamping knobs. Elapsed times are measured by the recorder from
/// when observation began; hosts that want display-friendly wall-clock stamps
/// supply them as opaque strings (no chrono dependency in this workspace).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingBlock {
    /// Engine-supplied elapsed time from run start to finish, ms.
    pub duration_elapsed_ms: u64,
    /// When the run was queued before its driver first polled, if known.
    pub queued_to_start_delta_ms: Option<u64>,
    /// Host-provided completion timestamp, opaque for display only.
    pub completed_wall_clock: Option<String>,
    /// Host-provided start timestamp, opaque for display only.
    pub started_wall_clock: Option<String>,
}

/// Terminal state marker distinguishes a summarised, finished run from one
/// still being observed (never persisted until finished — the envelope carries
/// both shapes but callers must call `.finish()` to read aggregate fields).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum SweepRunSummary {
    /// Still accumulating events.
    Recording(RecordingSummary),
    /// Run reached a terminal outcome; aggregates frozen.
    Finished(FinishedSummary),
}

/// Everything accumulated during recording.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingSummary {
    pub attribution: RunAttribution,
    pub tool_counters: ToolCounters,
    pub iterations: usize,
    pub retries: u32,
    pub delegation_depth_max: u32,
    /// Most recent failing tool (drives attribution of a non-clean outcome).
    last_failed_tool: Option<String>,
    /// Start-of-run monotonic reference used to compute elapsed durations on
    /// finish (opaque — only meaningful relative to itself).
    start_elapsed_ms: u128,
}

impl RecordingSummary {
    fn start_now(attribution: RunAttribution) -> Self {
        Self {
            attribution,
            tool_counters: ToolCounters::default(),
            iterations: 0,
            retries: 0,
            delegation_depth_max: 0,
            last_failed_tool: None,
            start_elapsed_ms: 0, // run-scoped; zero-based by definition
        }
    }
}

/// The finished, immutable summary of one complete run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinishedSummary {
    pub attribution: RunAttribution,
    pub tool_counters: ToolCounters,
    pub timing: TimingBlock,
    pub iterations: usize,
    pub retries: u32,
    pub delegation_depth_max: u32,
    pub guardrail_stop: bool,
    /// Non-clean runs carry an attributed cause; clean (all-ok) runs omit it.
    pub attribution_cause: Option<AttributedCause>,
}

/// Which check/tool produced a non-clean outcome, where, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttributedCause {
    /// Which rule/check produced the failure, i.e. the tool name as recorded.
    pub rule: String,
    /// Free-form entity identifier for post-run drilldown (e.g. file path).
    pub entity: Option<String>,
    /// Error text or decision code explaining WHY it failed.
    pub reason: String,
}

/// Errors impossible by construction in the fold; surfaced only if a caller
/// violates the Recording→Finished transition contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishError(pub &'static str);

impl std::fmt::Display for FinishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for FinishError {}

impl SweepRunSummary {
    /// Begin observing a run. See also [`Self::finish`].
    pub fn new(attribution: RunAttribution) -> Self {
        Self::Recording(RecordingSummary::start_now(attribution))
    }

    /// True while still accumulating (a run that never reaches `finish()`).
    pub fn is_recording(&self) -> bool {
        matches!(self, Self::Recording(_))
    }

    /// Aggregate one event into the running summary. Ignored on `Finished`
    /// variants: events arriving after finish are late-arriving telemetry, not
    /// a protocol error (mpsc drain ordering means at most one straggler).
    pub fn observe(&mut self, ev: &RunEventTelemetry) {
        let Self::Recording(rec) = self else { return };
        rec.apply(ev);
    }

    /// Freeze the summary. Verifies monotonicity: end elapsed may not precede
    /// start elapsed (start is always 0-based, so negative elapsed is refused
    /// defensively even though the accumulator cannot produce it today).
    pub fn finish(
        self,
        end_elapsed_ms: u64,
        stop_reason_guardrail: bool,
        carried_outcome_iterations: Option<usize>,
        wall_clock: Option<(String, String)>,
    ) -> Result<Self, FinishError> {
        let Self::Recording(rec) = self else {
            return Err(FinishError("already finished"));
        };
        if (end_elapsed_ms as u128) < rec.start_elapsed_ms {
            return Err(FinishError("elapsed went backwards"));
        }
        let attribution_cause = if rec.tool_counters.failed > 0 || rec.retries > 0 {
            Some(AttributedCause {
                rule: rec
                    .last_failed_tool
                    .clone()
                    .unwrap_or_else(|| "llm_retry".to_string()),
                entity: None,
                reason: format!(
                    "{} tool failures, {} retries, delegate depth {}",
                    rec.tool_counters.failed, rec.retries, rec.delegation_depth_max
                ),
            })
        } else {
            None
        };
        Ok(Self::Finished(FinishedSummary {
            attribution: rec.attribution,
            tool_counters: rec.tool_counters,
            timing: TimingBlock {
                duration_elapsed_ms: end_elapsed_ms,
                queued_to_start_delta_ms: None,
                completed_wall_clock: wall_clock.as_ref().map(|(_, c)| c.clone()),
                started_wall_clock: wall_clock.map(|(s, _)| s),
            },
            iterations: carried_outcome_iterations.unwrap_or(rec.iterations),
            retries: rec.retries,
            delegation_depth_max: rec.delegation_depth_max,
            guardrail_stop: stop_reason_guardrail,
            attribution_cause,
        }))
    }
}

/// Minimal, engine-agnostic view of the events we care about. The engine maps
/// its rich `RunEvent` onto these few shapes; keeping this local VO instead of
/// depending on the core facade keeps core-domain free of any outward dep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunEventTelemetry {
    ToolEnd { tool: String, duration_ms: u64, ok: bool },
    Iteration { n: usize },
    Retry { wait_secs: u64 },
    DelegationNested { depth: u32 },
}

impl RecordingSummary {
    fn apply(&mut self, ev: &RunEventTelemetry) {
        match ev {
            RunEventTelemetry::ToolEnd { tool, duration_ms, ok } => {
                if !ok {
                    self.last_failed_tool = Some(tool.clone());
                }
                self.tool_counters.record(tool, *duration_ms, *ok)
            }
            RunEventTelemetry::Iteration { n } => self.iterations = self.iterations.max(*n),
            RunEventTelemetry::Retry { .. } => self.retries += 1,
            RunEventTelemetry::DelegationNested { depth } => {
                self.delegation_depth_max = self.delegation_depth_max.max(*depth)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_attr() -> RunAttribution {
        RunAttribution::chat("user:test")
    }

    #[test]
    fn trigger_wire_tags_are_locked_snake_case() {
        assert_eq!(RunTrigger::Chat.as_str(), "chat");
        assert_eq!(RunTrigger::Delegate.as_str(), "delegate");
        assert_eq!(RunTrigger::ScheduledSweep.as_str(), "scheduled_sweep");
        assert_eq!(RunTrigger::Api.as_str(), "api");
    }

    #[test]
    fn clean_run_yields_no_attribution_cause_and_reconciled_counters() {
        let mut s = SweepRunSummary::new(chat_attr());
        s.observe(&RunEventTelemetry::ToolEnd { tool: "Read".into(), duration_ms: 5, ok: true });
        s.observe(&RunEventTelemetry::ToolEnd { tool: "Bash".into(), duration_ms: 120, ok: true });
        s.observe(&RunEventTelemetry::Iteration { n: 3 });
        let fin = match s.finish(150, false, Some(3), None).unwrap() {
            SweepRunSummary::Finished(f) => f,
            other => panic!("expected Finished, got {other:?}"),
        };
        assert!(fin.attribution_cause.is_none(), "clean run has no cause");
        fin.tool_counters.reconcile_check().unwrap();
        assert_eq!(fin.tool_counters.total, 2);
        assert_eq!(fin.tool_counters.failed, 0);
        assert!(!fin.guardrail_stop);
        assert_eq!(fin.timing.duration_elapsed_ms, 150);
    }

    #[test]
    fn failed_run_carries_attributed_cause() {
        let mut s = SweepRunSummary::new(RunAttribution::scheduled("cron:sweep"));
        s.observe(&RunEventTelemetry::ToolEnd { tool: "Edit".into(), duration_ms: 10, ok: false });
        let fin = match s.finish(500, true, None, None).unwrap() {
            SweepRunSummary::Finished(f) => f,
            other => panic!("expected Finished, got {other:?}"),
        };
        let cause = fin.attribution_cause.expect("failure must be attributed");
        assert_eq!(cause.rule, "Edit");
        assert!(cause.reason.contains("1 tool failures"));
        assert!(fin.guardrail_stop);
    }

    #[test]
    fn empty_run_finishes_with_zeroed_aggregates() {
        let s = SweepRunSummary::new(chat_attr());
        let fin = match s.finish(0, false, Some(0), None).unwrap() {
            SweepRunSummary::Finished(f) => f,
            other => panic!("expected Finished, got {other:?}"),
        };
        assert_eq!(fin.tool_counters, ToolCounters::default());
        assert_eq!(fin.iterations, 0);
        assert!(fin.attribution_cause.is_none());
    }

    #[test]
    fn finish_is_idempotent_gated_by_variant() {
        let s = SweepRunSummary::new(chat_attr());
        assert!(s.finish(1, false, None, None).is_ok());
        // Re-finishing a Finished summary errors rather than double-counting.
        let done = SweepRunSummary::Finished(FinishedSummary {
            attribution: chat_attr(),
            tool_counters: ToolCounters::default(),
            timing: TimingBlock::default(),
            iterations: 0,
            retries: 0,
            delegation_depth_max: 0,
            guardrail_stop: false,
            attribution_cause: None,
        });
        assert!(done.finish(1, false, None, None).is_err());
    }

    #[test]
    fn serde_roundtrip_preserves_trigger_tag_and_finished_state() {
        let mut s = SweepRunSummary::new(RunAttribution {
            trigger: RunTrigger::Delegate,
            actor: "agent:TEST".into(),
        });
        s.observe(&RunEventTelemetry::ToolEnd { tool: "Grep".into(), duration_ms: 7, ok: false });
        let fin = s.finish(99, false, Some(1), Some(("t0".into(), "t1".into()))).unwrap();
        let json = serde_json::to_string(&fin).unwrap();
        assert!(json.contains(r#""state":"finished""#));
        assert!(json.contains(r#""trigger":"delegate""#));
        let back: SweepRunSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, fin);
    }

    #[test]
    fn delegate_depth_tracked_per_event_not_cumulative() {
        let mut s = SweepRunSummary::new(chat_attr());
        s.observe(&RunEventTelemetry::DelegationNested { depth: 3 });
        s.observe(&RunEventTelemetry::DelegationNested { depth: 2 });
        let fin = match s.finish(10, false, None, None).unwrap() {
            SweepRunSummary::Finished(f) => f,
            other => panic!("{other:?}"),
        };
        assert_eq!(fin.delegation_depth_max, 3);
    }
}
