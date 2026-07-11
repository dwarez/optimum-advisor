use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Serialize;

use crate::config::ServingConfig;
use crate::runner::{BenchmarkRunOutput, ProcessSpec};
use crate::Result;

mod suite;

pub use suite::{default_suite, CorrectnessSuite, CorrectnessTask};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CorrectnessResult {
    pub suite_id: String,
    pub status: CorrectnessStatus,
    pub threshold: f64,
    pub max_samples: u32,
    pub tasks: Vec<CorrectnessTaskResult>,
    pub artifacts: Vec<CorrectnessArtifact>,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CorrectnessTaskResult {
    pub domain: String,
    pub spec: String,
    pub metric: Option<String>,
    pub score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CorrectnessArtifact {
    pub path: String,
    pub json: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectnessStatus {
    Passed,
    Failed,
    Unknown,
}

impl CorrectnessStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Passed => "passed",
            Self::Failed => "failed",
            Self::Unknown => "unknown",
        }
    }

    pub fn rank(self) -> u8 {
        match self {
            Self::Passed => 0,
            Self::Unknown => 1,
            Self::Failed => 2,
        }
    }
}

pub fn lighteval_plan(config: &ServingConfig, output_dir: impl AsRef<Path>) -> ProcessSpec {
    lighteval_plan_for_suite(default_suite(), config, output_dir)
}

pub fn ensure_lighteval_suite_ready(suite: &CorrectnessSuite) -> Result<()> {
    ensure_litellm_backend_ready()?;
    for task in suite.tasks {
        let output = Command::new("lighteval")
            .args(["tasks", "inspect", task.spec])
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()
            .map_err(|err| {
                format!(
                    "lighteval is required for correctness checks: {err}; install with `./scripts/setup-correctness-env.sh`"
                )
            })?;
        if !output.status.success() {
            return Err(format!(
                "correctness task preflight failed for {}. Install missing lighteval task dependencies, e.g. `pip install langdetect` for IFEval.\n{}",
                task.spec,
                tail(&String::from_utf8_lossy(&output.stderr), 20)
            ));
        }
    }
    Ok(())
}

fn ensure_litellm_backend_ready() -> Result<()> {
    let output = Command::new("python3")
        .args([
            "-c",
            "import importlib.metadata as m, re\nimport litellm, diskcache, pyarrow\nversion = tuple(int(x) for x in re.findall(r'\\d+', m.version('litellm'))[:3])\nassert version >= (1, 66, 0), m.version('litellm')\n",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| {
            format!(
                "failed to run lighteval LiteLLM backend preflight: {err}; install with `./scripts/setup-correctness-env.sh`"
            )
        })?;
    if output.status.success() {
        return Ok(());
    }

    Err(format!(
        "correctness backend preflight failed for lighteval endpoint litellm. Install missing backend deps with `./scripts/setup-correctness-env.sh`.\n{}",
        tail(&String::from_utf8_lossy(&output.stderr), 20)
    ))
}

pub fn lighteval_plan_for_suite(
    suite: &CorrectnessSuite,
    config: &ServingConfig,
    output_dir: impl AsRef<Path>,
) -> ProcessSpec {
    ProcessSpec::new(
        "lighteval",
        vec![
            "endpoint".to_string(),
            "litellm".to_string(),
            model_args(config),
            suite.task_spec(),
            "--max-samples".to_string(),
            suite.max_samples.to_string(),
            "--output-dir".to_string(),
            output_dir.as_ref().display().to_string(),
            "--save-details".to_string(),
        ],
    )
}

fn model_args(config: &ServingConfig) -> String {
    format!(
        "provider=openai,model_name=openai/{},base_url=http://{}:{}/v1,api_key=EMPTY",
        config.model, config.host, config.port
    )
}

pub fn collect_lighteval_result(
    suite: &CorrectnessSuite,
    output_dir: impl AsRef<Path>,
    output: BenchmarkRunOutput,
) -> Result<CorrectnessResult> {
    let artifacts = collect_artifacts(output_dir.as_ref())?;
    let tasks = suite
        .tasks
        .iter()
        .map(|task| {
            let metric = artifacts
                .iter()
                .find_map(|artifact| metric_for_task(&artifact.json, task.spec));
            CorrectnessTaskResult {
                domain: task.domain.to_string(),
                spec: task.spec.to_string(),
                metric: metric.as_ref().map(|metric| metric.name.clone()),
                score: metric.map(|metric| metric.value),
            }
        })
        .collect::<Vec<_>>();
    let status = if tasks.is_empty() || tasks.iter().any(|task| task.score.is_none()) {
        CorrectnessStatus::Unknown
    } else if tasks
        .iter()
        .all(|task| task.score.is_some_and(|score| score >= suite.threshold))
    {
        CorrectnessStatus::Passed
    } else {
        CorrectnessStatus::Failed
    };
    write_responses_artifact(suite, output_dir.as_ref())?;
    let artifacts = collect_artifacts(output_dir.as_ref())?;

    Ok(CorrectnessResult {
        suite_id: suite.id.to_string(),
        status,
        threshold: suite.threshold,
        max_samples: suite.max_samples,
        tasks,
        artifacts,
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn collect_artifacts(output_dir: &Path) -> Result<Vec<CorrectnessArtifact>> {
    let mut paths = Vec::new();
    collect_json_paths(output_dir, &mut paths)?;
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let json = fs::read_to_string(&path)
                .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
            Ok(CorrectnessArtifact {
                path: path.display().to_string(),
                json,
            })
        })
        .collect()
}

fn collect_json_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read {}: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_json_paths(&path, paths)?;
        } else if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            paths.push(path);
        }
    }
    Ok(())
}

fn write_responses_artifact(suite: &CorrectnessSuite, output_dir: &Path) -> Result<()> {
    let mut details = Vec::new();
    collect_detail_paths(output_dir, &mut details)?;
    if details.is_empty() {
        return Ok(());
    }

    let output_dir_arg = output_dir
        .to_str()
        .ok_or_else(|| format!("non-utf8 correctness output path: {}", output_dir.display()))?;
    let mut args = vec!["-c", RESPONSES_SCRIPT, output_dir_arg, suite.id];
    let specs = suite.tasks.iter().map(|task| task.spec).collect::<Vec<_>>();
    args.extend(specs);
    let output = Command::new("python3")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|err| format!("failed to build correctness responses artifact: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "failed to build correctness responses artifact\n{}",
            tail(&String::from_utf8_lossy(&output.stderr), 20)
        ));
    }

    fs::write(output_dir.join("responses.json"), output.stdout).map_err(|err| {
        format!(
            "failed to write {}: {err}",
            output_dir.join("responses.json").display()
        )
    })
}

fn collect_detail_paths(dir: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    for entry in entries {
        let entry = entry.map_err(|err| format!("failed to read {}: {err}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_detail_paths(&path, paths)?;
        } else if path.file_name().is_some_and(|name| {
            let name = name.to_string_lossy();
            name.starts_with("details_") && name.ends_with(".parquet")
        }) {
            paths.push(path);
        }
    }
    Ok(())
}

const RESPONSES_SCRIPT: &str = r#"
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

output_dir = Path(sys.argv[1])
suite_id = sys.argv[2]
specs = sys.argv[3:]
details_root = output_dir / "details"
tasks = []

for spec in specs:
    files = sorted(details_root.glob(f"**/details_{spec}_*.parquet"))
    samples = []
    for file in files:
        samples.extend(pq.read_table(file).to_pylist())
    tasks.append({
        "spec": spec,
        "files": [str(file) for file in files],
        "samples": samples,
    })

print(json.dumps({
    "schema_version": 1,
    "suite_id": suite_id,
    "tasks": tasks,
}, ensure_ascii=False))
"#;

#[derive(Clone, Debug, PartialEq)]
struct ExtractedMetric {
    name: String,
    value: f64,
}

fn metric_for_task(raw_json: &str, task: &str) -> Option<ExtractedMetric> {
    let results = object_for_key(raw_json, "results")?;
    let metrics = object_for_key(results, task)?;
    first_metric(metrics)
}

fn object_for_key<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let key = format!("\"{}\"", key.replace('\\', "\\\\").replace('"', "\\\""));
    let start = json.find(&key)? + key.len();
    let colon = json[start..].find(':')? + start;
    object_after(json, colon + 1)
}

fn object_after(json: &str, start: usize) -> Option<&str> {
    let open = json[start..].find('{')? + start;
    let bytes = json.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for index in open..bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return json.get(open + 1..index);
                }
            }
            _ => {}
        }
    }
    None
}

fn first_metric(metrics: &str) -> Option<ExtractedMetric> {
    for field in metrics.split(',') {
        let Some((name, value)) = field.split_once(':') else {
            continue;
        };
        let name = name.trim().trim_matches('"');
        if name.ends_with("_stderr") {
            continue;
        }
        let Some(value) = first_json_number(value) else {
            continue;
        };
        return Some(ExtractedMetric {
            name: name.to_string(),
            value,
        });
    }
    None
}

fn first_json_number(text: &str) -> Option<f64> {
    let text = text.trim_start();
    let end = text
        .find(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E')))
        .unwrap_or(text.len());
    text.get(..end)?
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
}

fn tail(text: &str, max_lines: usize) -> String {
    let mut lines = text.lines().rev().take(max_lines).collect::<Vec<_>>();
    lines.reverse();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BenchmarkConfig;
    use crate::engine::{Engine, Metric};
    use crate::trial::Candidate;

    fn config() -> ServingConfig {
        ServingConfig {
            engine: Engine::Vllm,
            image: "vllm/vllm-openai:latest".to_string(),
            resolved_image: None,
            model: "Qwen/Qwen3-4B-Instruct-2507".to_string(),
            gpus: 1,
            host: "127.0.0.1".to_string(),
            port: 8000,
            startup_timeout_secs: 300,
            max_model_len: 8192,
            metric: Metric::Tps,
            candidate: Candidate::default(),
            serve_args: Vec::new(),
            benchmark: BenchmarkConfig::default(),
        }
    }

    #[test]
    fn lighteval_plan_targets_served_openai_endpoint() {
        let plan = lighteval_plan(&config(), ".optimum-advisor/results/run/correctness");

        assert_eq!(plan.program, "lighteval");
        assert!(plan.args.contains(&"endpoint".to_string()));
        assert!(plan.args.contains(&"litellm".to_string()));
        assert!(plan
            .args
            .contains(&"provider=openai,model_name=openai/Qwen/Qwen3-4B-Instruct-2507,base_url=http://127.0.0.1:8000/v1,api_key=EMPTY".to_string()));
        assert!(plan.args.contains(&"--max-samples".to_string()));
        assert!(plan.args.contains(&default_suite().max_samples.to_string()));
        assert!(plan.args.contains(&"--save-details".to_string()));
    }

    #[test]
    fn extracts_scores_from_lighteval_results_json() {
        let raw = r#"{
            "results": {
                "gsm8k|0": {"em": 0.5, "em_stderr": 0.1},
                "ifeval|0": {"prompt_level_strict_acc": 0.7},
                "all": {"acc": 0.6}
            }
        }"#;

        assert_eq!(
            metric_for_task(raw, "gsm8k|0"),
            Some(ExtractedMetric {
                name: "em".to_string(),
                value: 0.5
            })
        );
        assert_eq!(
            metric_for_task(raw, "ifeval|0"),
            Some(ExtractedMetric {
                name: "prompt_level_strict_acc".to_string(),
                value: 0.7
            })
        );
        assert_eq!(
            metric_for_task(raw, "all"),
            Some(ExtractedMetric {
                name: "acc".to_string(),
                value: 0.6
            })
        );
    }

    #[test]
    fn collects_lighteval_artifacts_and_scores_suite() {
        let dir = std::env::temp_dir().join(format!(
            "optimum-advisor-correctness-test-{}",
            std::process::id()
        ));
        let results_dir = dir.join("results/model");
        std::fs::create_dir_all(&results_dir).unwrap();
        std::fs::write(
            results_dir.join("results_1.json"),
            r#"{
                "results": {
                    "gsm8k|0": {"em": 1.0, "em_stderr": 0.0},
                    "ifeval|0": {"prompt_level_strict_acc": 1.0},
                    "triviaqa|0": {"em": 1.0},
                    "drop|1": {"em": 1.0}
                }
            }"#,
        )
        .unwrap();

        let result = collect_lighteval_result(
            default_suite(),
            &dir,
            BenchmarkRunOutput {
                stdout: "stdout".to_string(),
                stderr: String::new(),
            },
        )
        .unwrap();

        assert_eq!(result.status, CorrectnessStatus::Passed);
        assert_eq!(result.artifacts.len(), 1);
        assert_eq!(result.tasks[0].metric.as_deref(), Some("em"));
        assert_eq!(
            result.tasks[1].metric.as_deref(),
            Some("prompt_level_strict_acc")
        );
        assert!(result.tasks.iter().all(|task| task.score == Some(1.0)));
    }

    #[test]
    fn finds_lighteval_detail_parquet_files() {
        let dir = std::env::temp_dir().join(format!(
            "optimum-advisor-correctness-details-test-{}",
            std::process::id()
        ));
        let details_dir = dir.join("details/model/date");
        std::fs::create_dir_all(&details_dir).unwrap();
        std::fs::write(details_dir.join("details_triviaqa|0_date.parquet"), "").unwrap();
        std::fs::write(details_dir.join("results_date.json"), "{}").unwrap();

        let mut paths = Vec::new();
        collect_detail_paths(&dir, &mut paths).unwrap();

        assert_eq!(paths.len(), 1);
        assert!(paths[0]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("details_triviaqa|0_"));
    }

    #[test]
    fn suite_status_requires_each_task_to_clear_threshold() {
        let dir = std::env::temp_dir().join(format!(
            "optimum-advisor-correctness-threshold-test-{}",
            std::process::id()
        ));
        let results_dir = dir.join("results/model");
        std::fs::create_dir_all(&results_dir).unwrap();
        std::fs::write(
            results_dir.join("results_1.json"),
            r#"{
                "results": {
                    "gsm8k|0": {"em": 1.0},
                    "ifeval|0": {"prompt_level_strict_acc": 1.0},
                    "triviaqa|0": {"em": 1.0},
                    "drop|1": {"em": 0.0}
                }
            }"#,
        )
        .unwrap();

        let result = collect_lighteval_result(
            default_suite(),
            &dir,
            BenchmarkRunOutput {
                stdout: String::new(),
                stderr: String::new(),
            },
        )
        .unwrap();

        assert_eq!(result.status, CorrectnessStatus::Failed);
    }
}
