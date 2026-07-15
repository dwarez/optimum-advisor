use std::{cmp::Ordering, collections::HashSet, path::PathBuf};

use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    app::{
        run_correctness_check_with_cancellation, run_evaluation_with_cancellation, EvaluationResult,
    },
    cli::args::{CommandKind, Invocation},
    config::{ConfigInput, ExecutableConfig, RuntimeInput},
    domain::{
        engine::{Engine, Metric},
        run::{ExecutionBackend, PullPolicy, ResolvedImage},
    },
    error::{Error, ErrorKind, ExecutionStage, Result},
    inspection::{
        correctness::{CorrectnessResult, CorrectnessStatus},
        hardware::inspect_hardware as inspect_hardware_runtime,
        model_memory::{estimate_model_memory, resolve_hf_mem_command},
    },
    results::{
        metrics::{compare_observations, BenchmarkMetrics},
        report::{ModelMemoryOutcome, RunKind, RunState},
    },
    runtime::{
        cancel::CancellationToken,
        docker::resolve_image,
        params::{cached_parameter_schema, load_parameter_schema, ParameterSchema},
        process::ProcessExecutor,
    },
};

const DEFAULT_PARAMETER_CACHE: &str = ".optimum-advisor/params";
const DEFAULT_RESULTS_DIR: &str = ".optimum-advisor/results";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolCallRequest {
    name: String,
    #[serde(default = "empty_object")]
    arguments: Value,
}

fn empty_object() -> Value {
    json!({})
}

pub(super) fn call_tool_request(params: &Value, cancellation: &CancellationToken) -> Result<Value> {
    let request: ToolCallRequest = decode(params.clone(), "tools/call params")?;
    if !TOOL_NAMES.contains(&request.name.as_str()) {
        return Err(Error::usage(format!("unknown tool: {}", request.name)));
    }
    let call = match call_tool(&request.name, request.arguments, cancellation) {
        Ok(call) => call,
        Err(error) => ToolCall::error(serde_json::to_value(error.payload()).map_err(|source| {
            Error::new(ErrorKind::Protocol, None, "failed to encode MCP tool error")
                .with_source(source)
        })?),
    };
    tool_result(call)
}

#[derive(Debug)]
struct ToolCall {
    value: Value,
    is_error: bool,
}

impl ToolCall {
    fn success(value: Value) -> Self {
        Self {
            value,
            is_error: false,
        }
    }

    fn error(value: Value) -> Self {
        Self {
            value,
            is_error: true,
        }
    }
}

const TOOL_NAMES: &[&str] = &[
    "inspect_hardware",
    "inspect_engine",
    "validate_config",
    "estimate_memory",
    "check_correctness",
    "run_benchmark",
    "evaluate_candidate",
    "run_sweep",
    "rank_candidates",
];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct InspectHardwareArgs {
    #[serde(default)]
    #[schemars(range(min = 1))]
    gpus: Option<usize>,
    #[serde(default)]
    gpu_devices: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct InspectEngineArgs {
    engine: Engine,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    cache_dir: Option<PathBuf>,
    #[serde(default)]
    refresh: bool,
    #[serde(default)]
    offline: bool,
    #[serde(default)]
    pull_policy: PullPolicy,
    #[serde(default)]
    allow_local_image: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ValidateConfigArgs {
    config: ConfigInput,
    #[serde(default)]
    cache_dir: Option<PathBuf>,
    #[serde(default)]
    offline: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ConfigArgs {
    config: ConfigInput,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ExecutionArgs {
    config: ConfigInput,
    #[serde(default)]
    cache_dir: Option<PathBuf>,
    #[serde(default)]
    results_dir: Option<PathBuf>,
}

#[derive(Clone, Copy)]
enum EvaluationMode {
    Benchmark,
    Candidate,
    Sweep,
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(rename_all = "snake_case")]
enum RankCorrectness {
    Passed,
    Failed,
    Unknown,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RankCandidate {
    #[schemars(length(min = 1))]
    id: String,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    correctness: Option<RankCorrectness>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct RankCandidatesArgs {
    metric: Metric,
    #[schemars(length(min = 1))]
    candidates: Vec<RankCandidate>,
}

#[derive(Serialize, JsonSchema)]
pub(super) struct EngineInspectionOutput {
    image: ResolvedImage,
    schema: ParameterSchema,
}

#[derive(Serialize, JsonSchema)]
pub(super) struct ConfigValidationOutput {
    valid: bool,
    image: ResolvedImage,
    config: ExecutableConfig,
}

#[derive(Serialize, JsonSchema)]
struct BestTrialSummary {
    metrics: Option<BenchmarkMetrics>,
    correctness: Option<CorrectnessResult>,
    model_memory: ModelMemoryOutcome,
}

#[derive(Serialize, JsonSchema)]
pub(super) struct EvaluationSummary {
    report_path: PathBuf,
    run_id: String,
    kind: RunKind,
    state: RunState,
    trial_count: usize,
    succeeded: usize,
    failed: usize,
    best_trial_index: Option<usize>,
    best_winning_value: Option<f64>,
    best_config_path: Option<PathBuf>,
    best: Option<BestTrialSummary>,
    output: String,
    terminal_log: String,
}

#[derive(Serialize, JsonSchema)]
pub(super) struct RankCandidatesOutput {
    metric: Metric,
    candidates: Vec<RankedCandidate>,
}

fn call_tool(name: &str, arguments: Value, cancellation: &CancellationToken) -> Result<ToolCall> {
    match name {
        "inspect_hardware" => inspect_hardware_tool(decode(arguments, name)?, cancellation),
        "inspect_engine" => inspect_engine_tool(decode(arguments, name)?, cancellation),
        "validate_config" => validate_config_tool(decode(arguments, name)?, cancellation),
        "estimate_memory" => estimate_memory_tool(decode(arguments, name)?, cancellation),
        "check_correctness" => check_correctness_tool(decode(arguments, name)?, cancellation),
        "run_benchmark" => execute_tool(
            decode(arguments, name)?,
            EvaluationMode::Benchmark,
            cancellation,
        ),
        "evaluate_candidate" => execute_tool(
            decode(arguments, name)?,
            EvaluationMode::Candidate,
            cancellation,
        ),
        "run_sweep" => execute_tool(
            decode(arguments, name)?,
            EvaluationMode::Sweep,
            cancellation,
        ),
        "rank_candidates" => rank_candidates(decode(arguments, name)?),
        _ => unreachable!("tool name was validated"),
    }
}

fn decode<T: DeserializeOwned>(value: Value, label: &str) -> Result<T> {
    serde_json::from_value(value).map_err(|source| {
        Error::validation(format!("invalid {label} arguments")).with_source(source)
    })
}

fn inspect_hardware_tool(
    args: InspectHardwareArgs,
    cancellation: &CancellationToken,
) -> Result<ToolCall> {
    let mut input = ConfigInput::minimal(Engine::Vllm, "hardware-inspection");
    input.runtime = RuntimeInput {
        gpus: args.gpus,
        gpu_devices: args.gpu_devices,
        ..RuntimeInput::default()
    };
    let runtime = input.normalize()?.runtime;
    let profile = inspect_hardware_runtime(&runtime, &ProcessExecutor::default(), cancellation)?;
    serialized(profile)
}

fn inspect_engine_tool(
    args: InspectEngineArgs,
    cancellation: &CancellationToken,
) -> Result<ToolCall> {
    if args.offline && args.refresh {
        return Err(Error::validation(
            "offline and refresh cannot both be enabled",
        ));
    }
    let image = args
        .image
        .unwrap_or_else(|| args.engine.default_image().to_string());
    let cache = args
        .cache_dir
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PARAMETER_CACHE));
    let executor = ProcessExecutor::default();
    let (identity, schema) = if args.offline {
        require_immutable_image(&image)?;
        let schema = cached_parameter_schema(args.engine, &image, &cache)?.ok_or_else(|| {
            Error::new(
                ErrorKind::ParameterInspection,
                Some(ExecutionStage::ParameterInspection),
                "offline parameter cache entry was not found",
            )
            .with_cache_identity(&image)
        })?;
        (
            ResolvedImage {
                requested: image.clone(),
                immutable: image,
                local_only: false,
            },
            schema,
        )
    } else {
        let identity = resolve_image(
            &image,
            args.pull_policy,
            args.allow_local_image,
            &executor,
            cancellation,
        )?;
        let schema = load_parameter_schema(
            args.engine,
            &identity.immutable,
            &cache,
            args.refresh,
            &executor,
            cancellation,
            ExecutionBackend::Docker,
        )?;
        (identity.resolved(), schema)
    };
    serialized(EngineInspectionOutput {
        image: identity,
        schema,
    })
}

fn validate_config_tool(
    args: ValidateConfigArgs,
    cancellation: &CancellationToken,
) -> Result<ToolCall> {
    let normalized = args.config.normalize()?;
    let cache = args
        .cache_dir
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PARAMETER_CACHE));
    let executor = ProcessExecutor::default();
    let (identity, schema) = if args.offline {
        require_immutable_image(&normalized.image)?;
        let schema = cached_parameter_schema(normalized.engine, &normalized.image, &cache)?
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::ParameterInspection,
                    Some(ExecutionStage::ParameterInspection),
                    "offline parameter cache entry was not found",
                )
                .with_cache_identity(&normalized.image)
            })?;
        (
            ResolvedImage {
                requested: normalized.image.clone(),
                immutable: normalized.image.clone(),
                local_only: false,
            },
            schema,
        )
    } else {
        let identity = resolve_image(
            &normalized.image,
            normalized.runtime.pull_policy,
            normalized.runtime.allow_local_image,
            &executor,
            cancellation,
        )?;
        let schema = load_parameter_schema(
            normalized.engine,
            &identity.immutable,
            &cache,
            false,
            &executor,
            cancellation,
            ExecutionBackend::Docker,
        )?;
        (identity.resolved(), schema)
    };
    let config = normalized.into_executable(identity.clone());
    schema.validate(&config.serve_args)?;
    serialized(ConfigValidationOutput {
        valid: true,
        image: identity,
        config,
    })
}

fn require_immutable_image(image: &str) -> Result<()> {
    let digest = image
        .split_once("@sha256:")
        .map(|(_, digest)| digest)
        .or_else(|| image.strip_prefix("sha256:"));
    if digest.is_some_and(|value| {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }) {
        Ok(())
    } else {
        Err(Error::validation(
            "offline inspection requires a repository digest or sha256 image ID",
        ))
    }
}

fn estimate_memory_tool(args: ConfigArgs, cancellation: &CancellationToken) -> Result<ToolCall> {
    let mut input = args.config;
    input.model_memory.enabled = Some(true);
    input.model_memory.required = Some(true);
    let normalized = input.normalize()?;
    let config = normalized.clone().into_executable(ResolvedImage {
        requested: normalized.image.clone(),
        immutable: normalized.image.clone(),
        local_only: normalized.runtime.allow_local_image,
    });
    let command = resolve_hf_mem_command(config.model_memory.command.as_deref());
    let estimate =
        estimate_model_memory(&config, command, &ProcessExecutor::default(), cancellation)?;
    serialized(estimate)
}

fn check_correctness_tool(
    mut args: ExecutionArgs,
    cancellation: &CancellationToken,
) -> Result<ToolCall> {
    args.config.correctness.enabled = Some(true);
    args.config.model_memory.enabled = Some(false);
    args.config.sweep = None;
    let invocation = execution_invocation(args, RunKind::Bench);
    serialized(run_correctness_check_with_cancellation(
        invocation,
        cancellation,
    )?)
}

fn execute_tool(
    mut args: ExecutionArgs,
    mode: EvaluationMode,
    cancellation: &CancellationToken,
) -> Result<ToolCall> {
    let kind = match mode {
        EvaluationMode::Benchmark => {
            args.config.correctness.enabled = Some(false);
            args.config.sweep = None;
            RunKind::Bench
        }
        EvaluationMode::Candidate => {
            args.config.sweep = None;
            RunKind::Bench
        }
        EvaluationMode::Sweep => {
            if args.config.sweep.is_none() {
                return Err(Error::validation("run_sweep requires config.sweep"));
            }
            RunKind::Sweep
        }
    };
    let invocation = execution_invocation(args, kind);
    let mut terminal = Vec::new();
    let mut progress = Vec::new();
    let result = run_evaluation_with_cancellation(
        invocation,
        kind,
        &mut terminal,
        &mut progress,
        cancellation,
    )?
    .ok_or_else(|| {
        Error::new(
            ErrorKind::Protocol,
            Some(ExecutionStage::Benchmark),
            "MCP evaluation unexpectedly returned a dry-run result",
        )
    })?;
    evaluation_summary(result, &terminal, &progress)
}

fn execution_invocation(args: ExecutionArgs, kind: RunKind) -> Invocation {
    Invocation {
        kind: match kind {
            RunKind::Bench => CommandKind::Bench,
            RunKind::Sweep => CommandKind::Sweep,
        },
        input: args.config,
        execute: true,
        backend: ExecutionBackend::Docker,
        results_dir: args
            .results_dir
            .unwrap_or_else(|| PathBuf::from(DEFAULT_RESULTS_DIR)),
        parameter_cache_dir: args
            .cache_dir
            .unwrap_or_else(|| PathBuf::from(DEFAULT_PARAMETER_CACHE)),
        refresh_parameters: false,
        offline_parameters: false,
        cleanup_run_id: None,
        cleanup_dry_run: false,
    }
}

fn evaluation_summary(
    result: EvaluationResult,
    terminal: &[u8],
    progress: &[u8],
) -> Result<ToolCall> {
    let succeeded = result
        .report
        .trials
        .iter()
        .filter(|trial| trial.is_success())
        .count();
    let best = result
        .report
        .best_trial_index
        .and_then(|index| result.report.trials.get(index))
        .map(|trial| match trial {
            crate::results::report::TrialOutcome::Success {
                metrics,
                correctness,
                model_memory,
                ..
            } => BestTrialSummary {
                metrics: Some(metrics.clone()),
                correctness: correctness.clone(),
                model_memory: model_memory.clone(),
            },
            crate::results::report::TrialOutcome::Failed {
                metrics,
                correctness,
                model_memory,
                ..
            } => BestTrialSummary {
                metrics: metrics.clone(),
                correctness: correctness.clone(),
                model_memory: model_memory.clone(),
            },
        });
    serialized(EvaluationSummary {
        report_path: result.report_path,
        run_id: result.report.run_id,
        kind: result.report.kind,
        state: result.report.state,
        trial_count: result.report.trials.len(),
        succeeded,
        failed: result.report.trials.len() - succeeded,
        best_trial_index: result.report.best_trial_index,
        best_winning_value: result.report.best_winning_value,
        best_config_path: result.report.best_config_path,
        best,
        output: String::from_utf8_lossy(terminal).into_owned(),
        terminal_log: String::from_utf8_lossy(progress).into_owned(),
    })
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
struct RankedCandidate {
    rank: usize,
    id: String,
    value: Option<f64>,
    correctness: Option<RankCorrectness>,
}

fn rank_candidates(args: RankCandidatesArgs) -> Result<ToolCall> {
    let mut seen = HashSet::new();
    let mut ranked = args
        .candidates
        .into_iter()
        .map(|candidate| {
            let id = candidate.id.trim().to_string();
            if id.is_empty() {
                return Err(Error::validation("candidate id must not be empty"));
            }
            if !seen.insert(id.clone()) {
                return Err(Error::validation(format!("duplicate candidate id: {id}")));
            }
            Ok(RankedCandidate {
                rank: 0,
                id,
                value: candidate.value,
                correctness: candidate.correctness,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ranked.sort_by(|left, right| compare_scores(left, right, args.metric));
    for (index, candidate) in ranked.iter_mut().enumerate() {
        candidate.rank = index + 1;
    }
    serialized(RankCandidatesOutput {
        metric: args.metric,
        candidates: ranked,
    })
}

fn compare_scores(left: &RankedCandidate, right: &RankedCandidate, metric: Metric) -> Ordering {
    compare_observations(
        metric,
        correctness_status(left.correctness),
        left.value,
        correctness_status(right.correctness),
        right.value,
    )
    .reverse()
    .then_with(|| left.id.cmp(&right.id))
}

fn correctness_status(status: Option<RankCorrectness>) -> Option<CorrectnessStatus> {
    match status {
        Some(RankCorrectness::Passed) => Some(CorrectnessStatus::Passed),
        Some(RankCorrectness::Failed) => Some(CorrectnessStatus::Failed),
        Some(RankCorrectness::Unknown) | None => None,
    }
}

const MAX_TOOL_TEXT_BYTES: usize = 64 * 1024;

fn tool_result(call: ToolCall) -> Result<Value> {
    let text = serde_json::to_string_pretty(&call.value).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            None,
            "failed to render MCP tool result",
        )
        .with_source(source)
    })?;
    Ok(json!({
        "content": [{ "type": "text", "text": bounded_tool_text(text) }],
        "structuredContent": call.value,
        "isError": call.is_error,
    }))
}

fn bounded_tool_text(mut text: String) -> String {
    if text.len() <= MAX_TOOL_TEXT_BYTES {
        return text;
    }
    let mut end = MAX_TOOL_TEXT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    text.push_str("\n...[truncated; use structuredContent or report_path]");
    text
}

fn serialized(value: impl Serialize) -> Result<ToolCall> {
    let value = serde_json::to_value(value).map_err(|source| {
        Error::new(
            ErrorKind::Protocol,
            None,
            "failed to encode MCP tool result",
        )
        .with_source(source)
    })?;
    Ok(ToolCall::success(value))
}

#[cfg(test)]
mod tests {
    use std::{
        io::Cursor,
        sync::{Arc, Mutex},
    };

    use super::*;
    use crate::mcp::{
        protocol::{
            begin_request, cancel_request, finish_request, lock_in_flight, serve, InFlightState,
            MAX_REQUEST_BYTES, SERVER_NAME,
        },
        schema::tool_definitions,
    };

    #[test]
    fn stdio_initializes_lists_and_calls_tools() {
        let requests = [
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
            json!({
                "jsonrpc":"2.0",
                "id":3,
                "method":"tools/call",
                "params": {
                    "name":"rank_candidates",
                    "arguments": {
                        "metric":"tps",
                        "candidates":[
                            {"id":"a","value":1.0},
                            {"id":"b","value":2.0}
                        ]
                    }
                }
            }),
        ]
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("\n");
        let mut output = Vec::new();

        serve(Cursor::new(requests), &mut output).unwrap();

        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 3);
        assert_eq!(responses[0]["result"]["serverInfo"]["name"], SERVER_NAME);
        let tool_names = responses[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            tool_names,
            [
                "inspect_hardware",
                "inspect_engine",
                "validate_config",
                "estimate_memory",
                "check_correctness",
                "run_benchmark",
                "evaluate_candidate",
                "run_sweep",
                "rank_candidates",
            ]
        );
        assert_eq!(
            responses[2]["result"]["structuredContent"]["candidates"][0]["id"],
            "b"
        );
    }

    #[test]
    fn enforces_initialize_and_initialized_notification_lifecycle() {
        let requests = [
            json!({"jsonrpc":"2.0","id":0,"method":"tools/list"}),
            json!({"jsonrpc":"2.0","id":1,"method":"ping"}),
            json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":3,"method":"tools/list"}),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({"jsonrpc":"2.0","id":4,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","id":5,"method":"tools/list"}),
            json!({"jsonrpc":"2.0","id":6,"method":"shutdown"}),
        ]
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("\n");
        let mut output = Vec::new();

        serve(Cursor::new(requests), &mut output).unwrap();

        let responses = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 7);
        assert_eq!(responses[0]["error"]["code"], -32002);
        assert_eq!(responses[1]["result"], json!({}));
        assert_eq!(responses[2]["id"], 2);
        assert_eq!(responses[3]["error"]["code"], -32002);
        assert_eq!(responses[4]["error"]["code"], -32600);
        assert_eq!(responses[5]["result"]["tools"].as_array().unwrap().len(), 9);
        assert_eq!(responses[6]["error"]["code"], -32601);
    }

    #[test]
    fn incompatible_protocol_version_returns_error_and_disconnects() {
        let requests = [
            json!({
                "jsonrpc":"2.0",
                "id":1,
                "method":"initialize",
                "params":{"protocolVersion":"1900-01-01"}
            }),
            json!({"jsonrpc":"2.0","id":2,"method":"ping"}),
        ]
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("\n");
        let mut output = Vec::new();

        serve(Cursor::new(requests), &mut output).unwrap();

        let responses = String::from_utf8(output).unwrap();
        assert_eq!(responses.lines().count(), 1);
        assert!(responses.contains("incompatible protocol version"));
        assert!(!responses.contains("\"id\":2"));
    }

    #[test]
    fn malformed_json_returns_parse_error() {
        let mut output = Vec::new();
        serve(Cursor::new("{not-json\n"), &mut output).unwrap();
        let response: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(response["error"]["code"], -32700);
        assert!(response["id"].is_null());
    }

    #[test]
    fn ranks_direction_correctness_and_ties_deterministically() {
        let cancellation = CancellationToken::new();
        let throughput = call_tool(
            "rank_candidates",
            json!({
                "metric": "tps",
                "candidates": [
                    {"id":"failed", "value":100.0, "correctness":"failed"},
                    {"id":"b", "value":20.0, "correctness":"passed"},
                    {"id":"a", "value":20.0, "correctness":"passed"}
                ]
            }),
            &cancellation,
        )
        .unwrap();
        assert_eq!(throughput.value["candidates"][0]["id"], "a");
        assert_eq!(throughput.value["candidates"][2]["id"], "failed");

        let latency = call_tool(
            "rank_candidates",
            json!({
                "metric":"p99_ttft",
                "candidates":[
                    {"id":"slow","value":20.0},
                    {"id":"fast","value":10.0}
                ]
            }),
            &cancellation,
        )
        .unwrap();
        assert_eq!(latency.value["candidates"][0]["id"], "fast");
    }

    #[test]
    fn rejects_oversized_request_and_continues() {
        let oversized = format!("{{\"x\":\"{}\"}}\n", "a".repeat(MAX_REQUEST_BYTES));
        let valid = json!({"jsonrpc":"2.0","id":1,"method":"ping"}).to_string();
        let mut output = Vec::new();

        serve(Cursor::new(format!("{oversized}{valid}")), &mut output).unwrap();

        let responses = String::from_utf8(output).unwrap();
        assert_eq!(responses.lines().count(), 2);
        assert!(responses.contains("request exceeds"));
        assert!(responses.contains("\"id\":1"));
    }

    #[test]
    fn config_decoder_uses_canonical_shape_and_rejects_unknowns() {
        let config = decode::<ConfigArgs>(
            json!({
                "config": {
                    "engine":"sglang",
                    "model":"m",
                    "runtime":{"gpus":2},
                    "candidate":{"tensor_parallelism":2},
                    "serve_args":[{"name":"reasoning-parser","value":"deepseek"}]
                }
            }),
            "estimate_memory",
        )
        .unwrap()
        .config
        .normalize()
        .unwrap();
        assert_eq!(config.engine, Engine::Sglang);
        assert_eq!(config.candidate.tensor_parallelism, 2);
        assert_eq!(config.serve_args[0].name, "reasoning-parser");

        let error = decode::<ConfigArgs>(
            json!({"config":{"engine":"vllm","model":"m","wat":1}}),
            "estimate_memory",
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
    }

    #[test]
    fn generated_tool_schemas_cover_every_strict_input_and_output() {
        let tools = tool_definitions().unwrap();
        assert_eq!(tools.len(), TOOL_NAMES.len());
        for tool in &tools {
            assert_eq!(tool["inputSchema"]["additionalProperties"], false);
            assert!(tool.get("outputSchema").is_some());
        }

        let validation = tools
            .iter()
            .find(|tool| tool["name"] == "validate_config")
            .unwrap();
        let definitions = validation["inputSchema"]["$defs"].as_object().unwrap();
        for name in [
            "BenchmarkInput",
            "CandidateOverrides",
            "ConfigInput",
            "CorrectnessInput",
            "DynamicArg",
            "LeaderboardInput",
            "ModelMemoryInput",
            "RuntimeInput",
            "SweepSpec",
        ] {
            assert_eq!(
                definitions[name]["additionalProperties"], false,
                "{name} must reject unknown fields"
            );
        }
    }

    #[test]
    fn typed_tool_inputs_reject_unknown_fields() {
        let error = call_tool(
            "rank_candidates",
            json!({
                "metric": "tps",
                "candidates": [{"id": "a", "value": 1.0}],
                "unexpected": true
            }),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Validation);
        assert!(error
            .to_string()
            .contains("invalid rank_candidates arguments"));
    }

    #[test]
    fn cancellation_notifications_cancel_pending_request_tokens() {
        let in_flight = Arc::new(Mutex::new(InFlightState::default()));
        cancel_request(&in_flight, json!(7));

        let token = begin_request(&in_flight, &json!(7), &CancellationToken::new());

        assert!(token.is_cancelled());
        finish_request(&in_flight, &json!(7));
        assert!(lock_in_flight(&in_flight).active.is_none());
    }

    #[test]
    fn human_tool_content_is_bounded_without_truncating_structured_content() {
        let value = json!({"payload": "x".repeat(MAX_TOOL_TEXT_BYTES * 2)});
        let result = tool_result(ToolCall::success(value.clone())).unwrap();

        assert!(result["content"][0]["text"].as_str().unwrap().len() < value.to_string().len());
        assert_eq!(result["structuredContent"], value);
    }
}
