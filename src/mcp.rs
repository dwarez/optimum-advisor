use std::cmp::Ordering;
use std::io::{BufRead, Write};
use std::path::Path;

use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::cli::Setup;
use crate::config::ServingConfig;
use crate::correctness::CorrectnessStatus;
use crate::engine::{Engine, Metric, Mode};
use crate::engines::adapter_for;
use crate::results::{compare_observations, create_run_dir, write_report, ResultSet};
use crate::serve::EngineArg;
use crate::tools::{
    estimate_memory, evaluate_candidate, inspect_engine, inspect_hardware, validate_config,
    EvaluationChecks, EvaluationOptions,
};
use crate::Result;

const PROTOCOL_VERSION: &str = "2025-11-25";
const SERVER_NAME: &str = "optimum-advisor";

pub fn serve(input: impl BufRead, mut output: impl Write) -> Result<()> {
    for line in input.lines() {
        let line = line.map_err(|err| format!("failed to read MCP input: {err}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<Value>(&line) {
            Ok(request) => request,
            Err(err) => {
                write_message(
                    &mut output,
                    &rpc_error(Value::Null, -32700, format!("parse error: {err}")),
                )?;
                continue;
            }
        };
        if let Some(response) = handle_request(&request) {
            write_message(&mut output, &response)?;
        }
    }
    Ok(())
}

fn handle_request(request: &Value) -> Option<Value> {
    let id = request.get("id").cloned()?;
    let Some(method) = request.get("method").and_then(Value::as_str) else {
        return Some(rpc_error(id, -32600, "invalid request"));
    };
    let params = request.get("params").cloned().unwrap_or_else(|| json!({}));
    let result = match method {
        "initialize" => Ok(initialize(&params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => call_tool_request(&params),
        _ => return Some(rpc_error(id, -32601, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err(err) => rpc_error(id, -32602, err),
    })
}

fn initialize(params: &Value) -> Value {
    let requested = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(PROTOCOL_VERSION);
    let protocol_version = match requested {
        "2025-11-25" | "2025-06-18" | "2025-03-26" | "2024-11-05" => requested,
        _ => PROTOCOL_VERSION,
    };
    json!({
        "protocolVersion": protocol_version,
        "capabilities": { "tools": { "listChanged": false } },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Atomic inspection and evaluation tools for LLM serving configurations"
        },
        "instructions": "Inspect hardware and engine parameters, validate or memory-prune candidates, then evaluate correctness and benchmark metrics. Candidate execution failures are returned as structured tool results."
    })
}

fn call_tool_request(params: &Value) -> Result<Value> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "tools/call requires a tool name".to_string())?;
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let call = match call_tool(name, &arguments) {
        Ok(call) => call,
        Err(err) => ToolCall::error(json!({ "status": "failed", "message": err })),
    };
    tool_result(call)
}

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

fn call_tool(name: &str, arguments: &Value) -> Result<ToolCall> {
    let args = object(arguments, "tool arguments")?;
    match name {
        "inspect_hardware" => serialized(inspect_hardware()),
        "inspect_engine" => {
            let engine = Engine::parse(required_string(args, "engine")?)?;
            let image = optional_string(args, "image")?
                .map(str::to_string)
                .unwrap_or_else(|| engine.default_image().to_string());
            let cache_dir =
                optional_string(args, "param_cache_dir")?.unwrap_or(".optimum-advisor/params");
            let refresh = optional_bool(args, "refresh")?.unwrap_or(false);
            serialized(inspect_engine(
                engine,
                image,
                Path::new(cache_dir),
                refresh,
            )?)
        }
        "validate_config" => {
            let config = config_argument(args)?;
            let cache_dir =
                optional_string(args, "param_cache_dir")?.unwrap_or(".optimum-advisor/params");
            let refresh = optional_bool(args, "refresh")?.unwrap_or(false);
            serialized(validate_config(&config, Path::new(cache_dir), refresh)?)
        }
        "estimate_memory" => serialized(estimate_memory(&config_argument(args)?)),
        "check_correctness" => evaluation_call(args, EvaluationChecks::CORRECTNESS),
        "run_benchmark" => evaluation_call(args, EvaluationChecks::BENCHMARK),
        "evaluate_candidate" => evaluation_call(args, EvaluationChecks::ALL),
        "rank_candidates" => rank_candidates(args),
        _ => Err(format!("unknown tool: {name}")),
    }
}

fn evaluation_call(args: &Map<String, Value>, checks: EvaluationChecks) -> Result<ToolCall> {
    let config = config_argument(args)?;
    let cache_dir = optional_string(args, "param_cache_dir")?.unwrap_or(".optimum-advisor/params");
    let results_dir = optional_string(args, "results_dir")?.unwrap_or(".optimum-advisor/results");
    let refresh = optional_bool(args, "refresh")?.unwrap_or(false);
    let run_dir = create_run_dir(results_dir, "agent", config.engine)?;
    let mut options = EvaluationOptions::new(cache_dir, &run_dir);
    options.refresh_params = refresh;
    options.checks = checks;
    let mut terminal = Vec::new();

    match evaluate_candidate(config, options, &mut terminal) {
        Ok(evaluation) => {
            let report = if checks.benchmark {
                let mut results = ResultSet::new(evaluation.config.metric);
                results.push(evaluation.clone().into_trial_result()?);
                Some(write_report(&run_dir, "agent", &results)?)
            } else {
                None
            };
            let mut value = serde_json::to_value(evaluation)
                .map_err(|err| format!("failed to encode evaluation: {err}"))?;
            let fields = value
                .as_object_mut()
                .ok_or_else(|| "evaluation result was not an object".to_string())?;
            fields.insert(
                "terminal_log".to_string(),
                Value::String(String::from_utf8_lossy(&terminal).to_string()),
            );
            fields.insert(
                "report_path".to_string(),
                report
                    .map(|path| Value::String(path.display().to_string()))
                    .unwrap_or(Value::Null),
            );
            Ok(ToolCall::success(value))
        }
        Err(failure) => {
            let mut value = serde_json::to_value(failure)
                .map_err(|err| format!("failed to encode evaluation failure: {err}"))?;
            if let Some(fields) = value.as_object_mut() {
                fields.insert(
                    "artifact_dir".to_string(),
                    Value::String(run_dir.display().to_string()),
                );
                fields.insert(
                    "terminal_log".to_string(),
                    Value::String(String::from_utf8_lossy(&terminal).to_string()),
                );
            }
            Ok(ToolCall::error(value))
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct RankedCandidate {
    rank: usize,
    id: String,
    value: Option<f64>,
    correctness: Option<String>,
}

fn rank_candidates(args: &Map<String, Value>) -> Result<ToolCall> {
    let metric = Metric::parse(required_string(args, "metric")?)?;
    let candidates = args
        .get("candidates")
        .and_then(Value::as_array)
        .ok_or_else(|| "rank_candidates requires a candidates array".to_string())?;
    let mut ranked = candidates
        .iter()
        .map(|candidate| {
            let candidate = object(candidate, "candidate score")?;
            let id = required_string(candidate, "id")?.to_string();
            let value = optional_f64(candidate, "value")?;
            let correctness = optional_string(candidate, "correctness")?.map(str::to_string);
            if let Some(status) = correctness.as_deref() {
                if !matches!(status, "passed" | "failed" | "unknown") {
                    return Err(format!("unknown correctness status: {status}"));
                }
            }
            Ok(RankedCandidate {
                rank: 0,
                id,
                value,
                correctness,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    ranked.sort_by(|left, right| compare_scores(left, right, metric));
    for (index, candidate) in ranked.iter_mut().enumerate() {
        candidate.rank = index + 1;
    }
    serialized(json!({ "metric": metric.to_string(), "candidates": ranked }))
}

fn compare_scores(left: &RankedCandidate, right: &RankedCandidate, metric: Metric) -> Ordering {
    compare_observations(
        metric,
        correctness_status(left.correctness.as_deref()),
        left.value,
        correctness_status(right.correctness.as_deref()),
        right.value,
    )
}

fn correctness_status(status: Option<&str>) -> Option<CorrectnessStatus> {
    match status {
        Some("passed") => Some(CorrectnessStatus::Passed),
        Some("failed") => Some(CorrectnessStatus::Failed),
        Some("unknown") => Some(CorrectnessStatus::Unknown),
        _ => None,
    }
}

fn config_argument(args: &Map<String, Value>) -> Result<ServingConfig> {
    let value = args
        .get("config")
        .ok_or_else(|| "tool requires a config object".to_string())?;
    config_from_json(value)
}

fn config_from_json(value: &Value) -> Result<ServingConfig> {
    let config = object(value, "config")?;
    let engine = Engine::parse(required_string(config, "engine")?)?;
    let model = required_string(config, "model")?.trim();
    if model.is_empty() {
        return Err("config.model must not be empty".to_string());
    }

    let mut setup = Setup::default_for_mode(Mode::Bench);
    setup.engine = engine;
    setup.model = model.to_string();
    setup.image = optional_string(config, "image")?.map(str::to_string);
    if let Some(value) = optional_u64(config, "gpus")? {
        setup.gpus = usize::try_from(value).map_err(|_| "config.gpus is too large".to_string())?;
    }
    if setup.gpus == 0 {
        return Err("config.gpus must be greater than zero".to_string());
    }
    if let Some(value) = optional_string(config, "host")? {
        setup.host = value.to_string();
    }
    if let Some(value) = optional_u64(config, "port")? {
        setup.port = u16::try_from(value).map_err(|_| "config.port must fit u16".to_string())?;
    }
    if setup.port == 0 {
        return Err("config.port must be greater than zero".to_string());
    }
    if let Some(value) = optional_u64(config, "startup_timeout_secs")? {
        setup.startup_timeout_secs = value;
    }
    if let Some(value) = optional_u64(config, "max_model_len")? {
        setup.max_model_len =
            u32::try_from(value).map_err(|_| "config.max_model_len must fit u32".to_string())?;
    }
    if setup.max_model_len == 0 {
        return Err("config.max_model_len must be greater than zero".to_string());
    }
    if let Some(value) = optional_string(config, "metric")? {
        setup.metric = Metric::parse(value)?;
    }
    if let Some(benchmark) = config.get("benchmark") {
        apply_benchmark(object(benchmark, "config.benchmark")?, &mut setup)?;
    }
    if let Some(args) = config.get("serve_args") {
        setup.serve_args = parse_serve_args(args)?;
    }

    let adapter = adapter_for(engine);
    let mut candidate = adapter.initial_candidate(&setup);
    if let Some(value) = config.get("candidate") {
        apply_candidate(object(value, "config.candidate")?, &mut candidate)?;
    }
    candidate.clamp_to_gpus(setup.gpus);
    Ok(ServingConfig::from_setup_and_candidate(&setup, candidate))
}

fn apply_benchmark(values: &Map<String, Value>, setup: &mut Setup) -> Result<()> {
    if let Some(value) = optional_string(values, "dataset_name")? {
        setup.benchmark.dataset_name = value.to_string();
    }
    if let Some(value) = optional_u64(values, "num_prompts")? {
        setup.benchmark.num_prompts =
            u32::try_from(value).map_err(|_| "benchmark.num_prompts must fit u32".to_string())?;
    }
    if let Some(value) = optional_string(values, "request_rate")? {
        setup.benchmark.request_rate = value.to_string();
    }
    if let Some(value) = values.get("max_concurrency") {
        setup.benchmark.max_concurrency = if value.is_null() {
            None
        } else {
            Some(
                u32::try_from(value.as_u64().ok_or_else(|| {
                    "benchmark.max_concurrency must be an integer or null".to_string()
                })?)
                .map_err(|_| "benchmark.max_concurrency must fit u32".to_string())?,
            )
        };
    }
    if let Some(value) = optional_u64(values, "random_input_len")? {
        setup.benchmark.random_input_len = u32::try_from(value)
            .map_err(|_| "benchmark.random_input_len must fit u32".to_string())?;
    }
    if let Some(value) = optional_u64(values, "random_output_len")? {
        setup.benchmark.random_output_len = u32::try_from(value)
            .map_err(|_| "benchmark.random_output_len must fit u32".to_string())?;
    }
    Ok(())
}

fn apply_candidate(
    values: &Map<String, Value>,
    candidate: &mut crate::trial::Candidate,
) -> Result<()> {
    if let Some(value) = values.get("parallelism") {
        let values = object(value, "candidate.parallelism")?;
        if let Some(value) = optional_u64(values, "tensor")? {
            candidate.parallelism.tensor = usize::try_from(value)
                .map_err(|_| "candidate.parallelism.tensor is too large".to_string())?;
        }
        if let Some(value) = optional_u64(values, "pipeline")? {
            candidate.parallelism.pipeline = usize::try_from(value)
                .map_err(|_| "candidate.parallelism.pipeline is too large".to_string())?;
        }
        if let Some(value) = optional_u64(values, "data")? {
            candidate.parallelism.data = usize::try_from(value)
                .map_err(|_| "candidate.parallelism.data is too large".to_string())?;
        }
    }
    if let Some(value) = values.get("memory") {
        let values = object(value, "candidate.memory")?;
        if let Some(value) = optional_f64(values, "fraction")? {
            if !(0.0 < value && value <= 1.0) {
                return Err("candidate.memory.fraction must be in (0, 1]".to_string());
            }
            candidate.memory.fraction = value as f32;
        }
    }
    if let Some(value) = values.get("scheduler") {
        let values = object(value, "candidate.scheduler")?;
        if let Some(value) = optional_u64(values, "prefill_token_budget")? {
            candidate.scheduler.prefill_token_budget = u32::try_from(value)
                .map_err(|_| "candidate.scheduler.prefill_token_budget must fit u32".to_string())?;
        }
        if let Some(value) = optional_u64(values, "max_running_requests")? {
            candidate.scheduler.max_running_requests = u32::try_from(value)
                .map_err(|_| "candidate.scheduler.max_running_requests must fit u32".to_string())?;
        }
    }
    Ok(())
}

fn parse_serve_args(value: &Value) -> Result<Vec<EngineArg>> {
    value
        .as_array()
        .ok_or_else(|| "config.serve_args must be an array".to_string())?
        .iter()
        .map(|value| {
            let value = object(value, "serve argument")?;
            let name = required_string(value, "name")?;
            match value.get("value") {
                None | Some(Value::Null) => Ok(EngineArg::flag(name)),
                Some(Value::String(arg)) => Ok(EngineArg::value(name, arg)),
                Some(Value::Number(arg)) => Ok(EngineArg::value(name, arg.to_string())),
                Some(Value::Bool(arg)) => Ok(EngineArg::value(name, arg.to_string())),
                Some(_) => Err(
                    "serve argument value must be a string, number, boolean, or null".to_string(),
                ),
            }
        })
        .collect()
}

fn serialized(value: impl Serialize) -> Result<ToolCall> {
    serde_json::to_value(value)
        .map(ToolCall::success)
        .map_err(|err| format!("failed to encode tool result: {err}"))
}

fn tool_result(call: ToolCall) -> Result<Value> {
    let text = serde_json::to_string_pretty(&call.value)
        .map_err(|err| format!("failed to render tool result: {err}"))?;
    Ok(json!({
        "content": [{ "type": "text", "text": text }],
        "structuredContent": call.value,
        "isError": call.is_error
    }))
}

fn tool_definitions() -> Vec<Value> {
    let config = config_schema();
    let evaluation_properties = json!({
        "config": config.clone(),
        "param_cache_dir": { "type": "string", "default": ".optimum-advisor/params" },
        "results_dir": { "type": "string", "default": ".optimum-advisor/results" },
        "refresh": { "type": "boolean", "default": false }
    });
    vec![
        tool(
            "inspect_hardware",
            "Inspect local NVIDIA GPUs, free memory, compute capability, and CUDA visibility.",
            json!({ "type": "object", "properties": {}, "additionalProperties": false }),
            true,
        ),
        tool(
            "inspect_engine",
            "Inspect and cache the serving parameters supported by a specific engine container image.",
            json!({
                "type": "object",
                "properties": {
                    "engine": { "type": "string", "enum": ["vllm", "sglang"] },
                    "image": { "type": "string", "description": "Defaults to the engine's standard image." },
                    "param_cache_dir": { "type": "string", "default": ".optimum-advisor/params" },
                    "refresh": { "type": "boolean", "default": false }
                },
                "required": ["engine"],
                "additionalProperties": false
            }),
            false,
        ),
        tool(
            "validate_config",
            "Validate a serving configuration against parameters discovered from its engine image.",
            json!({
                "type": "object",
                "properties": {
                    "config": config.clone(),
                    "param_cache_dir": { "type": "string", "default": ".optimum-advisor/params" },
                    "refresh": { "type": "boolean", "default": false }
                },
                "required": ["config"],
                "additionalProperties": false
            }),
            false,
        ),
        tool(
            "estimate_memory",
            "Estimate model weights, KV cache, and total memory for a candidate without starting a server.",
            json!({
                "type": "object",
                "properties": { "config": config.clone() },
                "required": ["config"],
                "additionalProperties": false
            }),
            true,
        ),
        tool(
            "check_correctness",
            "Start the configured server, run only the owned correctness suite, collect artifacts, and clean up.",
            json!({
                "type": "object",
                "properties": evaluation_properties.clone(),
                "required": ["config"],
                "additionalProperties": false
            }),
            false,
        ),
        tool(
            "run_benchmark",
            "Start the configured server, run only its engine-native benchmark, parse metrics, and clean up.",
            json!({
                "type": "object",
                "properties": evaluation_properties.clone(),
                "required": ["config"],
                "additionalProperties": false
            }),
            false,
        ),
        tool(
            "evaluate_candidate",
            "Validate a candidate, run correctness and benchmark against one server lifecycle, and return structured observations.",
            json!({
                "type": "object",
                "properties": evaluation_properties,
                "required": ["config"],
                "additionalProperties": false
            }),
            false,
        ),
        tool(
            "rank_candidates",
            "Rank observed candidate values using the metric direction and correctness status.",
            json!({
                "type": "object",
                "properties": {
                    "metric": { "type": "string", "description": "Any metric accepted by optimum-advisor, such as tps or p99_ttft." },
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "value": { "type": ["number", "null"] },
                                "correctness": { "type": ["string", "null"], "enum": ["passed", "failed", "unknown", null] }
                            },
                            "required": ["id", "value"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["metric", "candidates"],
                "additionalProperties": false
            }),
            true,
        ),
    ]
}

fn tool(name: &str, description: &str, input_schema: Value, read_only: bool) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": input_schema,
        "annotations": {
            "readOnlyHint": read_only,
            "destructiveHint": false,
            "idempotentHint": read_only,
            "openWorldHint": false
        }
    })
}

fn config_schema() -> Value {
    json!({
        "type": "object",
        "description": "A normalized serving candidate. Only engine and model are required; omitted fields use CLI defaults.",
        "properties": {
            "engine": { "type": "string", "enum": ["vllm", "sglang"] },
            "model": { "type": "string" },
            "image": { "type": "string" },
            "gpus": { "type": "integer", "minimum": 1, "default": 1 },
            "host": { "type": "string", "default": "127.0.0.1" },
            "port": { "type": "integer", "minimum": 1, "maximum": 65535, "default": 8000 },
            "startup_timeout_secs": { "type": "integer", "minimum": 1, "default": 300 },
            "max_model_len": { "type": "integer", "minimum": 1, "default": 8192 },
            "metric": { "type": "string", "default": "tps" },
            "candidate": {
                "type": "object",
                "properties": {
                    "parallelism": {
                        "type": "object",
                        "properties": {
                            "tensor": { "type": "integer", "minimum": 1 },
                            "pipeline": { "type": "integer", "minimum": 1 },
                            "data": { "type": "integer", "minimum": 1 }
                        },
                        "additionalProperties": false
                    },
                    "memory": {
                        "type": "object",
                        "properties": { "fraction": { "type": "number", "exclusiveMinimum": 0, "maximum": 1 } },
                        "additionalProperties": false
                    },
                    "scheduler": {
                        "type": "object",
                        "properties": {
                            "prefill_token_budget": { "type": "integer", "minimum": 1 },
                            "max_running_requests": { "type": "integer", "minimum": 1 }
                        },
                        "additionalProperties": false
                    }
                },
                "additionalProperties": false
            },
            "serve_args": {
                "type": "array",
                "description": "Engine arguments as name/value pairs; omit or null the value for flags.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "value": { "type": ["string", "number", "boolean", "null"] }
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
            },
            "benchmark": {
                "type": "object",
                "properties": {
                    "dataset_name": { "type": "string", "default": "random" },
                    "num_prompts": { "type": "integer", "minimum": 1, "default": 100 },
                    "request_rate": { "type": "string", "default": "1" },
                    "max_concurrency": { "type": ["integer", "null"], "minimum": 1, "default": 1 },
                    "random_input_len": { "type": "integer", "minimum": 1, "default": 1024 },
                    "random_output_len": { "type": "integer", "minimum": 1, "default": 128 }
                },
                "additionalProperties": false
            }
        },
        "required": ["engine", "model"],
        "additionalProperties": false
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn required_string<'a>(values: &'a Map<String, Value>, key: &str) -> Result<&'a str> {
    values
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{key} must be a string"))
}

fn optional_string<'a>(values: &'a Map<String, Value>, key: &str) -> Result<Option<&'a str>> {
    match values.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(format!("{key} must be a string")),
    }
}

fn optional_bool(values: &Map<String, Value>, key: &str) -> Result<Option<bool>> {
    match values.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("{key} must be a boolean")),
    }
}

fn optional_u64(values: &Map<String, Value>, key: &str) -> Result<Option<u64>> {
    match values.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be a non-negative integer")),
    }
}

fn optional_f64(values: &Map<String, Value>, key: &str) -> Result<Option<f64>> {
    match values.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| format!("{key} must be a finite number")),
    }
}

fn rpc_error(id: Value, code: i32, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() }
    })
}

fn write_message(output: &mut impl Write, message: &Value) -> Result<()> {
    serde_json::to_writer(&mut *output, message)
        .map_err(|err| format!("failed to write MCP response: {err}"))?;
    output
        .write_all(b"\n")
        .and_then(|_| output.flush())
        .map_err(|err| format!("failed to flush MCP response: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn request(message: Value) -> Value {
        let input = format!("{}\n", serde_json::to_string(&message).unwrap());
        let mut output = Vec::new();
        serve(Cursor::new(input), &mut output).unwrap();
        serde_json::from_slice(&output).unwrap()
    }

    #[test]
    fn initializes_and_declares_tools() {
        let response = request(json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "test", "version": "1" }
            }
        }));
        assert_eq!(response["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert!(response["result"]["capabilities"]["tools"].is_object());

        let response = request(json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }));
        let names = response["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect::<Vec<_>>();
        assert!(names.contains(&"estimate_memory"));
        assert!(names.contains(&"check_correctness"));
        assert!(names.contains(&"run_benchmark"));
        assert!(names.contains(&"evaluate_candidate"));
    }

    #[test]
    fn ranks_throughput_and_latency_in_the_right_direction() {
        let throughput = call_tool(
            "rank_candidates",
            &json!({
                "metric": "tps",
                "candidates": [
                    { "id": "slow", "value": 10.0 },
                    { "id": "fast", "value": 20.0 }
                ]
            }),
        )
        .unwrap();
        assert_eq!(throughput.value["candidates"][0]["id"], "fast");

        let latency = call_tool(
            "rank_candidates",
            &json!({
                "metric": "p99_ttft",
                "candidates": [
                    { "id": "slow", "value": 20.0 },
                    { "id": "fast", "value": 10.0 }
                ]
            }),
        )
        .unwrap();
        assert_eq!(latency.value["candidates"][0]["id"], "fast");
    }

    #[test]
    fn config_input_uses_engine_defaults_and_explicit_overrides() {
        let config = config_from_json(&json!({
            "engine": "sglang",
            "model": "m",
            "gpus": 2,
            "metric": "tpot",
            "candidate": {
                "parallelism": { "tensor": 2 },
                "scheduler": { "prefill_token_budget": 4096 }
            },
            "serve_args": [{ "name": "kv-cache-dtype", "value": "fp8" }]
        }))
        .unwrap();
        assert_eq!(config.engine, Engine::Sglang);
        assert_eq!(config.candidate.parallelism.tensor, 2);
        assert_eq!(config.candidate.scheduler.prefill_token_budget, 4096);
        assert_eq!(
            config.serve_args[0],
            EngineArg::value("kv-cache-dtype", "fp8")
        );
    }
}
