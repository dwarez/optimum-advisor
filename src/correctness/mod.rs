use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::config::ServingConfig;
use crate::runner::{BenchmarkRunOutput, ProcessSpec};
use crate::Result;

mod suite;

pub use suite::{default_suite, CorrectnessSuite, CorrectnessTask};

#[derive(Clone, Debug, PartialEq)]
pub struct CorrectnessResult {
    pub suite_id: String,
    pub status: CorrectnessStatus,
    pub score: Option<f64>,
    pub threshold: f64,
    pub max_samples: u32,
    pub tasks: Vec<CorrectnessTaskResult>,
    pub artifacts: Vec<CorrectnessArtifact>,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CorrectnessTaskResult {
    pub domain: String,
    pub spec: String,
    pub score: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorrectnessArtifact {
    pub path: String,
    pub json: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
            "import importlib.metadata as m, re\nimport litellm, diskcache\nversion = tuple(int(x) for x in re.findall(r'\\d+', m.version('litellm'))[:3])\nassert version >= (1, 66, 0), m.version('litellm')\n",
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
        .map(|task| CorrectnessTaskResult {
            domain: task.domain.to_string(),
            spec: task.spec.to_string(),
            score: artifacts
                .iter()
                .find_map(|artifact| score_for_task(&artifact.json, task.spec)),
        })
        .collect::<Vec<_>>();
    let task_scores = tasks
        .iter()
        .filter_map(|task| task.score)
        .collect::<Vec<_>>();
    let score = if task_scores.is_empty() {
        artifacts
            .iter()
            .find_map(|artifact| score_for_task(&artifact.json, "all"))
    } else {
        Some(task_scores.iter().sum::<f64>() / task_scores.len() as f64)
    };
    let status = match score {
        Some(score) if score >= suite.threshold => CorrectnessStatus::Passed,
        Some(_) => CorrectnessStatus::Failed,
        None => CorrectnessStatus::Unknown,
    };

    Ok(CorrectnessResult {
        suite_id: suite.id.to_string(),
        status,
        score,
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

fn score_for_task(raw_json: &str, task: &str) -> Option<f64> {
    let results = object_for_key(raw_json, "results")?;
    let metrics = object_for_key(results, task)?;
    first_metric_value(metrics)
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

fn first_metric_value(metrics: &str) -> Option<f64> {
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
        return Some(value);
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

        assert_eq!(score_for_task(raw, "gsm8k|0"), Some(0.5));
        assert_eq!(score_for_task(raw, "ifeval|0"), Some(0.7));
        assert_eq!(score_for_task(raw, "all"), Some(0.6));
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
        assert_eq!(result.score, Some(1.0));
        assert_eq!(result.artifacts.len(), 1);
        assert!(result.tasks.iter().all(|task| task.score == Some(1.0)));
    }
}
