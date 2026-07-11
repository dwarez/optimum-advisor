use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::advisor::hardware::{detect_hardware, HardwareProfile};
use crate::advisor::model_memory::{estimate_model_memory, ModelMemoryEstimate};
use crate::config::ServingConfig;
use crate::correctness::{
    capability_probe_plan, collect_lighteval_result, default_suite, ensure_lighteval_suite_ready,
    lighteval_plan, CorrectnessResult,
};
use crate::engine::Engine;
use crate::engines::adapter_for;
use crate::params::{load_or_inspect, ParameterSchema};
use crate::results::{BenchmarkMetrics, TrialResult};
use crate::runner::{
    execute_evaluation_plan_with_probe, resolve_docker_image_tag, BenchmarkRunOutput,
};
use crate::serve::EngineArg;
use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationChecks {
    pub correctness: bool,
    pub benchmark: bool,
}

impl EvaluationChecks {
    pub const CORRECTNESS: Self = Self {
        correctness: true,
        benchmark: false,
    };
    pub const BENCHMARK: Self = Self {
        correctness: false,
        benchmark: true,
    };
    pub const ALL: Self = Self {
        correctness: true,
        benchmark: true,
    };
}

#[derive(Clone, Debug)]
pub struct EvaluationOptions {
    pub param_cache_dir: PathBuf,
    pub artifact_dir: PathBuf,
    pub refresh_params: bool,
    pub checks: EvaluationChecks,
    pub preflight: bool,
    pub hardware: Option<HardwareProfile>,
    pub model_memory: Option<ModelMemoryEstimate>,
}

impl EvaluationOptions {
    pub fn new(param_cache_dir: impl Into<PathBuf>, artifact_dir: impl Into<PathBuf>) -> Self {
        Self {
            param_cache_dir: param_cache_dir.into(),
            artifact_dir: artifact_dir.into(),
            refresh_params: false,
            checks: EvaluationChecks::ALL,
            preflight: true,
            hardware: None,
            model_memory: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ConfigValidation {
    pub valid: bool,
    pub config: ServingConfig,
    pub effective_args: Vec<EngineArg>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidateEvaluation {
    pub status: &'static str,
    pub artifact_dir: PathBuf,
    pub config: ServingConfig,
    pub hardware: HardwareProfile,
    pub model_memory: ModelMemoryEstimate,
    pub metrics: Option<BenchmarkMetrics>,
    pub correctness: Option<CorrectnessResult>,
    pub benchmark_stdout: Option<String>,
    pub benchmark_stderr: Option<String>,
}

impl CandidateEvaluation {
    pub fn into_trial_result(self) -> Result<TrialResult> {
        let stdout = self
            .benchmark_stdout
            .ok_or_else(|| "evaluation did not run a benchmark".to_string())?;
        let stderr = self.benchmark_stderr.unwrap_or_default();
        let metric = self.config.metric;
        let mut trial = TrialResult::new(
            self.config,
            metric,
            self.hardware,
            self.model_memory,
            stdout,
            stderr,
        );
        if let Some(correctness) = self.correctness {
            trial = trial.with_correctness(correctness);
        }
        Ok(trial)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationStage {
    Preflight,
    Validation,
    Server,
    Correctness,
    Benchmark,
    ResultCollection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EvaluationFailure {
    pub status: &'static str,
    pub stage: EvaluationStage,
    pub message: String,
}

impl EvaluationFailure {
    fn new(stage: EvaluationStage, message: impl Into<String>) -> Self {
        Self {
            status: "failed",
            stage,
            message: message.into(),
        }
    }
}

impl fmt::Display for EvaluationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

pub fn inspect_hardware() -> HardwareProfile {
    detect_hardware()
}

pub fn inspect_engine(
    engine: Engine,
    image: String,
    cache_dir: &Path,
    refresh: bool,
) -> Result<ParameterSchema> {
    load_or_inspect(adapter_for(engine), image, cache_dir, refresh)
}

pub fn validate_config(
    config: &ServingConfig,
    cache_dir: &Path,
    refresh: bool,
) -> Result<ConfigValidation> {
    let adapter = adapter_for(config.engine);
    let args = adapter.serving_args(config);
    let schema = load_or_inspect(adapter, config.image.clone(), cache_dir, refresh)?;
    match schema.validate_args(&args) {
        Ok(()) => Ok(ConfigValidation {
            valid: true,
            config: config.clone(),
            effective_args: args,
        }),
        Err(err) if !refresh => {
            let schema = load_or_inspect(adapter, config.image.clone(), cache_dir, true)?;
            schema.validate_args(&args).map_err(|_| err)?;
            Ok(ConfigValidation {
                valid: true,
                config: config.clone(),
                effective_args: args,
            })
        }
        Err(err) => Err(err),
    }
}

pub fn estimate_memory(config: &ServingConfig) -> ModelMemoryEstimate {
    estimate_model_memory(config)
}

pub fn preflight(checks: EvaluationChecks) -> Result<()> {
    if !checks.correctness && !checks.benchmark {
        return Err("evaluation requires correctness, benchmark, or both".to_string());
    }
    ensure_hf_token()?;
    if checks.correctness {
        ensure_lighteval_suite_ready(default_suite())?;
    }
    Ok(())
}

pub fn evaluate_candidate(
    mut config: ServingConfig,
    options: EvaluationOptions,
    mut out: impl Write,
) -> std::result::Result<CandidateEvaluation, EvaluationFailure> {
    if options.preflight {
        preflight(options.checks)
            .map_err(|err| EvaluationFailure::new(EvaluationStage::Preflight, err))?;
    }
    validate_config(&config, &options.param_cache_dir, options.refresh_params)
        .map_err(|err| EvaluationFailure::new(EvaluationStage::Validation, err))?;

    let adapter = adapter_for(config.engine);
    let hardware = options.hardware.unwrap_or_else(inspect_hardware);
    let model_memory = options
        .model_memory
        .unwrap_or_else(|| estimate_memory(&config));
    let correctness_dir = options.artifact_dir.join("correctness");
    let correctness_plan = options
        .checks
        .correctness
        .then(|| lighteval_plan(&config, &correctness_dir));
    let capability_probe = options
        .checks
        .correctness
        .then(|| capability_probe_plan(&config, &correctness_dir))
        .flatten();
    let plan = adapter.run_plan(&config);
    let output = execute_evaluation_plan_with_probe(
        &plan,
        capability_probe.as_ref(),
        correctness_plan.as_ref(),
        options.checks.benchmark,
        &mut out,
    )
    .map_err(|err| EvaluationFailure::new(stage_for_runtime_error(&err), err))?;

    let correctness = match output.correctness {
        Some(output) => Some(
            collect_lighteval_result(default_suite(), &correctness_dir, output)
                .map_err(|err| EvaluationFailure::new(EvaluationStage::ResultCollection, err))?,
        ),
        None => None,
    };
    config.resolved_image =
        resolve_docker_image_tag(&config.image, engine_version_package(config.engine));
    let (metrics, benchmark_stdout, benchmark_stderr) = benchmark_parts(output.benchmark);

    Ok(CandidateEvaluation {
        status: "ok",
        artifact_dir: options.artifact_dir,
        config,
        hardware,
        model_memory,
        metrics,
        correctness,
        benchmark_stdout,
        benchmark_stderr,
    })
}

fn benchmark_parts(
    output: Option<BenchmarkRunOutput>,
) -> (Option<BenchmarkMetrics>, Option<String>, Option<String>) {
    match output {
        Some(output) => {
            let metrics = BenchmarkMetrics::parse(&output.stdout);
            (Some(metrics), Some(output.stdout), Some(output.stderr))
        }
        None => (None, None, None),
    }
}

fn stage_for_runtime_error(error: &str) -> EvaluationStage {
    if error.starts_with("correct ")
        || error.starts_with("correctness probe ")
        || error.starts_with("failed to start correct")
    {
        EvaluationStage::Correctness
    } else if error.starts_with("benchmark ") || error.starts_with("failed to start benchmark") {
        EvaluationStage::Benchmark
    } else {
        EvaluationStage::Server
    }
}

fn engine_version_package(engine: Engine) -> Option<&'static str> {
    match engine {
        Engine::Vllm => Some("vllm"),
        Engine::Sglang => Some("sglang"),
    }
}

pub fn ensure_hf_token() -> Result<()> {
    if std::env::var("HF_TOKEN")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        Ok(())
    } else {
        Err("HF_TOKEN is required for executing serving/benchmark containers".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_error_stage_is_structured() {
        assert_eq!(
            stage_for_runtime_error("benchmark exited with status 1"),
            EvaluationStage::Benchmark
        );
        assert_eq!(
            stage_for_runtime_error("server exited before becoming ready"),
            EvaluationStage::Server
        );
    }

    #[test]
    fn rejects_empty_evaluation() {
        let err = preflight(EvaluationChecks {
            correctness: false,
            benchmark: false,
        })
        .unwrap_err();
        assert!(err.contains("requires correctness, benchmark, or both"));
    }
}
