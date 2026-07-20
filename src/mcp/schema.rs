use schemars::{schema_for, JsonSchema};
use serde_json::{json, Value};

use crate::{
    app::CorrectnessCheckResult,
    domain::run::HardwareProfile,
    error::{Error, ErrorKind, ErrorPayload, Result},
    results::report::ModelMemoryOutcome,
};

use super::tools::{
    ConfigArgs, ConfigValidationOutput, EngineInspectionOutput, EvaluationSummary, ExecutionArgs,
    GetReportArgs, InspectEngineArgs, InspectHardwareArgs, ListRunsArgs, ListRunsOutput,
    RankCandidatesArgs, RankCandidatesOutput, ValidateConfigArgs,
};

#[derive(JsonSchema)]
#[serde(untagged)]
#[allow(dead_code)]
enum ToolOutput<T> {
    Success(T),
    Error(Box<ErrorPayload>),
}

pub(super) fn tool_definitions() -> Result<Vec<Value>> {
    Ok(vec![
        tool::<InspectHardwareArgs, HardwareProfile>(
            "inspect_hardware",
            "Inspect selected local NVIDIA GPUs with bounded nvidia-smi execution. Selection mirrors runtime config: `gpus` picks the first N visible GPUs, `gpu_devices` picks exact indexes/UUIDs and must match `gpus` when both are set.",
            true,
            true,
            false,
        )?,
        tool::<InspectEngineArgs, EngineInspectionOutput>(
            "inspect_engine",
            "Resolve an immutable image and inspect its runtime serving-parameter schema. May pull the image (multi-gigabyte) when it is not already local; set offline only with an immutable image and a matching cache entry.",
            false,
            true,
            true,
        )?,
        tool::<ValidateConfigArgs, ConfigValidationOutput>(
            "validate_config",
            "Normalize a canonical config, resolve its immutable image, and validate all serving arguments against the inspected schema. May pull the image (multi-gigabyte) when it is not already local.",
            false,
            true,
            true,
        )?,
        tool::<ConfigArgs, ModelMemoryOutcome>(
            "estimate_memory",
            "Run the configured bounded hf-mem command for one canonical candidate.",
            true,
            true,
            true,
        )?,
        tool::<ExecutionArgs, CorrectnessCheckResult>(
            "check_correctness",
            "Run the configured correctness suite without a throughput benchmark and return scores, artifacts, and the durable report path.",
            false,
            false,
            true,
        )?,
        tool::<ExecutionArgs, EvaluationSummary>(
            "run_benchmark",
            "Execute one candidate with correctness disabled and return per-trial summaries plus the durable report path. Long-running: includes image pull, server startup, and the benchmark. The selected metric must be emitted by that engine image; otherwise the objective fails, but parsed metrics remain available through get_report.",
            false,
            false,
            true,
        )?,
        tool::<ExecutionArgs, EvaluationSummary>(
            "evaluate_candidate",
            "Execute one candidate through validation, optional correctness, benchmark, persistence, and report finalization. Long-running: includes image pull, server startup, correctness, and the benchmark.",
            false,
            false,
            true,
        )?,
        tool::<ExecutionArgs, EvaluationSummary>(
            "run_sweep",
            "Execute the bounded config sweep, preserve failed trials, and return per-trial summaries plus the durable report path. Long-running: every candidate starts its own server.",
            false,
            false,
            true,
        )?,
        tool::<RankCandidatesArgs, RankCandidatesOutput>(
            "rank_candidates",
            "Rank observed candidate values by correctness first and metric direction second with deterministic ID tie-breaking. Omit a candidate's correctness when it was not evaluated; failed correctness always ranks last.",
            true,
            true,
            false,
        )?,
        tool::<GetReportArgs, EvaluationSummary>(
            "get_report",
            "Summarize a durable report.json from a previous run: state, ranking, and per-trial metrics, correctness, and compact failures.",
            true,
            true,
            false,
        )?,
        tool::<ListRunsArgs, ListRunsOutput>(
            "list_runs",
            "List prior runs under a results directory (newest first) with their state, trial counts, and best values.",
            true,
            true,
            false,
        )?,
    ])
}

fn tool<I: JsonSchema, O: JsonSchema>(
    name: &str,
    description: &str,
    read_only: bool,
    idempotent: bool,
    open_world: bool,
) -> Result<Value> {
    Ok(json!({
        "name": name,
        "description": description,
        "inputSchema": schema_value::<I>()?,
        "outputSchema": schema_value::<ToolOutput<O>>()?,
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": false,
            "idempotentHint": idempotent,
            "openWorldHint": open_world
        }
    }))
}

fn schema_value<T: JsonSchema>() -> Result<Value> {
    serde_json::to_value(schema_for!(T)).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            None,
            "failed to encode generated MCP schema",
        )
        .with_source(source)
    })
}
