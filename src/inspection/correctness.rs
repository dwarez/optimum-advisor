use std::{
    collections::{BTreeMap, HashSet},
    ffi::OsString,
    fs,
    net::IpAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    config::ExecutableConfig,
    domain::candidate::DynamicArg,
    error::{Error, ErrorKind, ExecutionStage, Result},
    runtime::{json::parse_unique_json, process::ProcessSpec},
};

const MAX_CORRECTNESS_JSON_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CorrectnessTask {
    pub domain: &'static str,
    pub spec: &'static str,
    pub metric: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CorrectnessSuite {
    pub id: &'static str,
    pub threshold: f64,
    pub max_samples: u32,
    pub tasks: &'static [CorrectnessTask],
}

impl CorrectnessSuite {
    pub(crate) fn task_spec(self) -> String {
        self.tasks
            .iter()
            .map(|task| task.spec)
            .collect::<Vec<_>>()
            .join(",")
    }
}

pub(crate) const DEFAULT_SUITE: CorrectnessSuite = CorrectnessSuite {
    id: "oa-fast-v1",
    threshold: 0.20,
    max_samples: 20,
    tasks: &[
        CorrectnessTask {
            domain: "math",
            spec: "gsm8k|0",
            metric: "extractive_match",
        },
        CorrectnessTask {
            domain: "instruction_following",
            spec: "ifeval|0",
            metric: "prompt_level_strict_acc",
        },
        CorrectnessTask {
            domain: "factual_qa",
            spec: "triviaqa|0",
            metric: "em",
        },
        CorrectnessTask {
            domain: "reading_comprehension",
            spec: "drop|1",
            metric: "em",
        },
    ],
};

pub(crate) fn lighteval_spec(
    config: &ExecutableConfig,
    suite: &CorrectnessSuite,
    output_dir: &Path,
) -> ProcessSpec {
    let base_url = endpoint_base_url(config);
    let cache_dir = output_dir.join("lighteval-cache");
    let model = format!(
        "provider=openai,model_name=openai/{},base_url={base_url},api_key=EMPTY,cache_dir={}",
        config.model,
        cache_dir.display(),
    );
    let args = [
        OsString::from("endpoint"),
        OsString::from("litellm"),
        OsString::from(model),
        OsString::from(suite.task_spec()),
        OsString::from("--max-samples"),
        OsString::from(suite.max_samples.to_string()),
        OsString::from("--output-dir"),
        output_dir.as_os_str().to_os_string(),
        OsString::from("--save-details"),
    ];
    let timeout = Duration::from_secs(config.correctness.timeout_secs);
    let mut spec = ProcessSpec::new("lighteval", args)
        .with_stage(ExecutionStage::Correctness)
        .with_timeout(timeout)
        .with_artifacts(
            output_dir.join("lighteval.stdout.log"),
            output_dir.join("lighteval.stderr.log"),
        )
        .with_safe_display(format!(
            "lighteval endpoint litellm <model> {} --output-dir {} --save-details",
            suite.task_spec(),
            output_dir.display()
        ));
    spec.max_stdout_bytes = config.runtime.max_process_output_bytes;
    spec.max_stderr_bytes = config.runtime.max_process_output_bytes;
    spec
}

pub(crate) fn capability_probe_spec(
    config: &ExecutableConfig,
    output_dir: &Path,
) -> Option<ProcessSpec> {
    let tool_parser = dynamic_value(&config.serve_args, "tool-call-parser").unwrap_or_default();
    let reasoning_parser =
        dynamic_value(&config.serve_args, "reasoning-parser").unwrap_or_default();
    if tool_parser.is_empty() && reasoning_parser.is_empty() {
        return None;
    }
    let timeout = Duration::from_secs(config.correctness.timeout_secs);
    let args = [
        OsString::from("-c"),
        OsString::from(CAPABILITY_PROBES_SCRIPT),
        OsString::from(endpoint_base_url(config)),
        OsString::from(&config.model),
        output_dir.join("capabilities.json").into_os_string(),
        OsString::from(tool_parser),
        OsString::from(reasoning_parser),
        OsString::from(config.correctness.timeout_secs.to_string()),
    ];
    let mut spec = ProcessSpec::new("python3", args)
        .with_stage(ExecutionStage::Correctness)
        .with_timeout(timeout)
        .with_artifacts(
            output_dir.join("capabilities.stdout.log"),
            output_dir.join("capabilities.stderr.log"),
        )
        .with_safe_display("python3 <correctness capability probes>");
    spec.max_stdout_bytes = config.runtime.max_process_output_bytes;
    spec.max_stderr_bytes = config.runtime.max_process_output_bytes;
    Some(spec)
}

fn endpoint_base_url(config: &ExecutableConfig) -> String {
    let host = match config.runtime.bind_host {
        IpAddr::V4(address) if address.is_unspecified() => "127.0.0.1".to_string(),
        IpAddr::V4(address) => address.to_string(),
        IpAddr::V6(address) if address.is_unspecified() => "[::1]".to_string(),
        IpAddr::V6(address) => format!("[{address}]"),
    };
    format!("http://{host}:{}/v1", config.runtime.port)
}

fn dynamic_value<'a>(arguments: &'a [DynamicArg], name: &str) -> Option<&'a str> {
    arguments
        .iter()
        .find(|argument| argument.name == name)
        .and_then(|argument| argument.value.as_deref())
}

const CAPABILITY_PROBES_SCRIPT: &str = r#"
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

base_url, model, artifact_path, tool_parser, reasoning_parser, timeout = sys.argv[1:]
timeout = max(1, int(timeout))
checks = []

def post(payload):
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read(1048577)
            if len(data) > 1048576:
                raise RuntimeError("capability response exceeded 1 MiB")
            return json.loads(data)
    except HTTPError as error:
        data = error.read(4096).decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {data}") from error

def run(domain, parser, probe):
    try:
        probe()
        checks.append({"domain": domain, "parser": parser, "passed": True})
    except Exception:
        checks.append({"domain": domain, "parser": parser, "passed": False})

def tool_calling():
    response = post({
        "model": model,
        "messages": [{"role": "user", "content": "Call get_temperature once for Rome."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get a city temperature.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 256,
    })
    calls = response["choices"][0]["message"].get("tool_calls") or []
    assert len(calls) == 1
    function = calls[0]["function"]
    assert function["name"] == "get_temperature"
    arguments = function["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    assert isinstance(arguments.get("city"), str)

def reasoning():
    response = post({
        "model": model,
        "messages": [{"role": "user", "content": "What is 17 multiplied by 19?"}],
        "temperature": 0,
        "max_tokens": 256,
    })
    message = response["choices"][0]["message"]
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    assert isinstance(reasoning, str) and reasoning.strip()
    content = message.get("content") or ""
    assert "<think>" not in content and "</think>" not in content

if tool_parser:
    run("tool_calling", tool_parser, tool_calling)
if reasoning_parser:
    run("reasoning", reasoning_parser, reasoning)

path = Path(artifact_path)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
with temporary.open("x") as output:
    json.dump({"schema_version": 1, "checks": checks}, output, indent=2)
    output.flush()
    os.fsync(output.fileno())
os.replace(temporary, path)
"#;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CorrectnessStatus {
    Passed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CorrectnessTaskResult {
    pub domain: String,
    pub spec: String,
    pub metric: String,
    pub score: f64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CapabilityResult {
    pub domain: String,
    pub parser: String,
    pub passed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CorrectnessArtifact {
    pub path: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CorrectnessResult {
    pub suite_id: String,
    pub status: CorrectnessStatus,
    pub threshold: f64,
    pub max_samples: u32,
    pub tasks: Vec<CorrectnessTaskResult>,
    pub capabilities: Vec<CapabilityResult>,
    pub artifacts: Vec<CorrectnessArtifact>,
}

pub(crate) fn collect_results(
    suite: &CorrectnessSuite,
    output_dir: &Path,
    serving_arguments: &[DynamicArg],
) -> Result<CorrectnessResult> {
    if !suite.threshold.is_finite() || !(0.0..=1.0).contains(&suite.threshold) {
        return Err(collection_error(
            "correctness threshold must be finite and in [0, 1]",
        ));
    }
    let mut paths = Vec::new();
    collect_json_paths(output_dir, output_dir, &mut paths)?;
    paths.sort();
    let mut scores = BTreeMap::<&str, f64>::new();
    let mut capability_document = None;
    let mut artifacts = Vec::with_capacity(paths.len());

    for path in paths {
        let metadata = fs::metadata(&path).map_err(|source| {
            collection_error("failed to inspect correctness artifact")
                .with_path(&path)
                .with_source(source)
        })?;
        if metadata.len() > MAX_CORRECTNESS_JSON_BYTES {
            return Err(
                collection_error("correctness JSON artifact exceeds 16 MiB").with_path(&path)
            );
        }
        let bytes = fs::read(&path).map_err(|source| {
            collection_error("failed to read correctness artifact")
                .with_path(&path)
                .with_source(source)
        })?;
        let text = std::str::from_utf8(&bytes).map_err(|source| {
            collection_error("correctness artifact is not UTF-8")
                .with_path(&path)
                .with_source(source)
        })?;
        let value: serde_json::Value = parse_unique_json(text).map_err(|source| {
            collection_error("correctness artifact is not strict JSON")
                .with_path(&path)
                .with_source(source)
        })?;
        let relative = path.strip_prefix(output_dir).map_err(|_| {
            collection_error("correctness artifact escaped its output directory").with_path(&path)
        })?;
        artifacts.push(CorrectnessArtifact {
            path: relative.to_string_lossy().into_owned(),
            bytes: metadata.len(),
        });

        if relative == Path::new("capabilities.json") {
            if capability_document.is_some() {
                return Err(collection_error(
                    "multiple capability result documents were found",
                ));
            }
            capability_document = Some(
                serde_json::from_value::<CapabilityDocument>(value).map_err(|source| {
                    collection_error("invalid capability result document")
                        .with_path(&path)
                        .with_source(source)
                })?,
            );
            continue;
        }
        let Some(results) = value.get("results") else {
            continue;
        };
        let results = results.as_object().ok_or_else(|| {
            collection_error("correctness results field must be an object").with_path(&path)
        })?;
        for task in suite.tasks {
            let Some(task_result) = results.get(task.spec) else {
                continue;
            };
            if scores.contains_key(task.spec) {
                return Err(collection_error(format!(
                    "duplicate correctness result for task {}",
                    task.spec
                )));
            }
            let metrics = task_result.as_object().ok_or_else(|| {
                collection_error(format!(
                    "correctness result for {} must be an object",
                    task.spec
                ))
                .with_path(&path)
            })?;
            let score = metrics
                .get(task.metric)
                .and_then(serde_json::Value::as_f64)
                .filter(|score| score.is_finite())
                .ok_or_else(|| {
                    collection_error(format!(
                        "correctness task {} is missing finite numeric metric {}",
                        task.spec, task.metric
                    ))
                    .with_path(&path)
                })?;
            scores.insert(task.spec, score);
        }
    }

    let tasks = suite
        .tasks
        .iter()
        .map(|task| {
            let score = scores.get(task.spec).copied().ok_or_else(|| {
                collection_error(format!("correctness output is missing task {}", task.spec))
            })?;
            Ok(CorrectnessTaskResult {
                domain: task.domain.to_string(),
                spec: task.spec.to_string(),
                metric: task.metric.to_string(),
                score,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let capabilities = validate_capabilities(serving_arguments, capability_document)?;
    let passed = tasks.iter().all(|task| task.score >= suite.threshold)
        && capabilities.iter().all(|capability| capability.passed);
    Ok(CorrectnessResult {
        suite_id: suite.id.to_string(),
        status: if passed {
            CorrectnessStatus::Passed
        } else {
            CorrectnessStatus::Failed
        },
        threshold: suite.threshold,
        max_samples: suite.max_samples,
        tasks,
        capabilities,
        artifacts,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapabilityDocument {
    schema_version: u32,
    checks: Vec<CapabilityResult>,
}

fn validate_capabilities(
    serving_arguments: &[DynamicArg],
    document: Option<CapabilityDocument>,
) -> Result<Vec<CapabilityResult>> {
    let expected = [
        ("tool-call-parser", "tool_calling"),
        ("reasoning-parser", "reasoning"),
    ]
    .into_iter()
    .filter_map(|(argument, domain)| {
        serving_arguments
            .iter()
            .find(|configured| configured.name == argument)
            .and_then(|configured| configured.value.as_deref())
            .map(|parser| (domain, parser))
    })
    .collect::<Vec<_>>();
    if expected.is_empty() {
        return Ok(Vec::new());
    }
    let document = document.ok_or_else(|| {
        collection_error("configured parser requires capabilities.json probe results")
    })?;
    if document.schema_version != 1 {
        return Err(collection_error(format!(
            "unsupported capability schema version {}",
            document.schema_version
        )));
    }
    let mut seen = HashSet::new();
    for check in &document.checks {
        if !seen.insert((check.domain.as_str(), check.parser.as_str())) {
            return Err(collection_error(format!(
                "duplicate capability result for {}:{}",
                check.domain, check.parser
            )));
        }
    }
    expected
        .into_iter()
        .map(|(domain, parser)| {
            document
                .checks
                .iter()
                .find(|check| check.domain == domain && check.parser == parser)
                .cloned()
                .ok_or_else(|| {
                    collection_error(format!("missing capability result for {domain}:{parser}"))
                })
        })
        .collect()
}

fn collect_json_paths(root: &Path, directory: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    let entries = fs::read_dir(directory).map_err(|source| {
        collection_error("failed to read correctness output directory")
            .with_path(directory)
            .with_source(source)
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| {
            collection_error("failed to read correctness output entry")
                .with_path(directory)
                .with_source(source)
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|source| {
            collection_error("failed to inspect correctness output entry")
                .with_path(&path)
                .with_source(source)
        })?;
        if file_type.is_symlink() {
            return Err(
                collection_error("symlinks are not allowed in correctness output").with_path(&path),
            );
        }
        if file_type.is_dir() {
            collect_json_paths(root, &path, paths)?;
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension == "json")
        {
            if !path.starts_with(root) {
                return Err(
                    collection_error("correctness artifact escaped its output directory")
                        .with_path(&path),
                );
            }
            paths.push(path);
        }
    }
    Ok(())
}

fn collection_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::Correctness,
        Some(ExecutionStage::ResultCollection),
        message,
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;
    use crate::{
        config::ConfigInput,
        domain::{candidate::DynamicArg, engine::Engine, run::ResolvedImage},
        error::{ErrorKind, ExecutionStage},
    };

    const TASKS: &[CorrectnessTask] = &[
        CorrectnessTask {
            domain: "math",
            spec: "gsm8k|0",
            metric: "extractive_match",
        },
        CorrectnessTask {
            domain: "instruction_following",
            spec: "ifeval|0",
            metric: "prompt_level_strict_acc",
        },
    ];

    fn suite() -> CorrectnessSuite {
        CorrectnessSuite {
            id: "test-suite",
            threshold: 0.5,
            max_samples: 4,
            tasks: TASKS,
        }
    }

    #[test]
    fn accepts_pinned_lighteval_0_13_default_suite_metrics() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("results.json"),
            r#"{"results":{"gsm8k|0":{"extractive_match":0.8},"ifeval|0":{"prompt_level_strict_acc":0.7},"triviaqa|0":{"em":0.6},"drop|1":{"em":0.5}}}"#,
        )
        .unwrap();

        let result = collect_results(&DEFAULT_SUITE, directory.path(), &[]).unwrap();

        assert_eq!(result.tasks.len(), DEFAULT_SUITE.tasks.len());
        assert_eq!(
            result
                .tasks
                .iter()
                .map(|task| (task.spec.as_str(), task.metric.as_str()))
                .collect::<Vec<_>>(),
            [
                ("gsm8k|0", "extractive_match"),
                ("ifeval|0", "prompt_level_strict_acc"),
                ("triviaqa|0", "em"),
                ("drop|1", "em"),
            ]
        );
    }

    #[test]
    fn accepts_only_complete_finite_expected_metrics() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("results.json"),
            r#"{"results":{"gsm8k|0":{"extractive_match":0.8,"extractive_match_stderr":0.1},"ifeval|0":{"prompt_level_strict_acc":0.7}}}"#,
        )
        .unwrap();

        let result = collect_results(&suite(), directory.path(), &[]).unwrap();

        assert_eq!(result.status, CorrectnessStatus::Passed);
        assert_eq!(result.tasks.len(), 2);
        assert!(result.tasks.iter().all(|task| task.score >= 0.5));
    }

    #[test]
    fn missing_wrong_and_duplicate_metrics_are_collection_failures() {
        for json in [
            r#"{"results":{"gsm8k|0":{"acc":0.8},"ifeval|0":{"prompt_level_strict_acc":0.7}}}"#,
            r#"{"results":{"gsm8k|0":{"extractive_match":"0.8"},"ifeval|0":{"prompt_level_strict_acc":0.7}}}"#,
            r#"{"results":{"gsm8k|0":{"extractive_match":0.8,"extractive_match":0.9},"ifeval|0":{"prompt_level_strict_acc":0.7}}}"#,
        ] {
            let directory = tempdir().unwrap();
            fs::write(directory.path().join("results.json"), json).unwrap();

            let error = collect_results(&suite(), directory.path(), &[]).unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Correctness);
            assert_eq!(error.stage(), Some(ExecutionStage::ResultCollection));
        }
    }

    #[test]
    fn duplicate_task_results_across_documents_are_rejected() {
        let directory = tempdir().unwrap();
        let result = r#"{"results":{"gsm8k|0":{"extractive_match":0.8},"ifeval|0":{"prompt_level_strict_acc":0.7}}}"#;
        fs::write(directory.path().join("first.json"), result).unwrap();
        fs::write(directory.path().join("second.json"), result).unwrap();

        assert!(collect_results(&suite(), directory.path(), &[]).is_err());
    }

    #[test]
    fn configured_capability_probe_must_have_a_typed_passing_result() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("results.json"),
            r#"{"results":{"gsm8k|0":{"extractive_match":0.8},"ifeval|0":{"prompt_level_strict_acc":0.7}},"message":"tool_calling hermes passed"}"#,
        )
        .unwrap();
        let configured = vec![DynamicArg::value("tool-call-parser", "hermes")];

        assert!(collect_results(&suite(), directory.path(), &configured).is_err());

        fs::write(
            directory.path().join("capabilities.json"),
            r#"{"schema_version":1,"checks":[{"domain":"tool_calling","parser":"hermes","passed":true}]}"#,
        )
        .unwrap();
        let result = collect_results(&suite(), directory.path(), &configured).unwrap();
        assert_eq!(result.capabilities.len(), 1);
    }

    #[test]
    fn correctness_commands_are_bounded_and_target_the_configured_endpoint() {
        let directory = tempdir().unwrap();
        let mut input = ConfigInput::minimal(Engine::Vllm, "repo/model");
        input.runtime.bind_host = Some("::1".parse().unwrap());
        input.correctness.timeout_secs = Some(42);
        let mut config = input.normalize().unwrap().into_executable(ResolvedImage {
            requested: "image:tag".into(),
            immutable:
                "image@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .into(),
            local_only: false,
        });

        let lighteval = lighteval_spec(&config, &suite(), directory.path());
        assert_eq!(lighteval.stage, Some(ExecutionStage::Correctness));
        let args = lighteval
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(args
            .iter()
            .any(|arg| arg.contains("base_url=http://[::1]:8000/v1")));
        assert!(args.iter().any(|arg| {
            arg.contains(&format!(
                "cache_dir={}",
                directory.path().join("lighteval-cache").display()
            ))
        }));
        assert!(args.contains(&suite().task_spec()));
        assert!(lighteval.deadline.is_some());

        assert!(capability_probe_spec(&config, directory.path()).is_none());
        config.serve_args = vec![DynamicArg::value("reasoning-parser", "qwen3")];
        let probe = capability_probe_spec(&config, directory.path()).unwrap();
        assert_eq!(probe.stage, Some(ExecutionStage::Correctness));
        assert!(probe.args.iter().any(|arg| arg == "qwen3"));
    }
}
