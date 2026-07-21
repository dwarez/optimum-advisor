use std::path::{Component, Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    config::ExecutableConfig,
    domain::{
        candidate::normalize_zero,
        engine::{Engine, Metric},
        run::{HardwareProfile, ResolvedImage},
    },
    error::{Error, ErrorKind, ErrorPayload, ExecutionStage, Result},
    inspection::correctness::{CorrectnessResult, CorrectnessStatus},
    runtime::atomic::{atomic_write, create_private_dir},
};

use super::{
    artifact::ArtifactManifest,
    metrics::{select_best, BenchmarkMetrics, RankableObservation},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunKind {
    Bench,
    Sweep,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunState {
    Running,
    Completed,
    CompletedWithFailures,
    Failed,
    Interrupted,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
pub(crate) struct WarningRecord {
    pub kind: ErrorKind,
    pub stage: ExecutionStage,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct ModelMemoryEstimate {
    pub source: String,
    pub model: String,
    pub max_model_len: u32,
    pub batch_size: u32,
    pub kv_cache_dtype: String,
    pub weights_bytes: Option<u64>,
    pub kv_cache_bytes: Option<u64>,
    pub activation_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, JsonSchema)]
pub(crate) struct ModelMemoryOutcome {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimate: Option<ModelMemoryEstimate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<WarningRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SubmissionState {
    Accepted,
    Queued,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
pub(crate) struct SubmissionResult {
    pub state: SubmissionState,
    pub message: String,
    pub remote_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct TrialFailure {
    #[serde(flatten)]
    pub error: Box<ErrorPayload>,
    pub timed_out: bool,
    pub interrupted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stdout_tail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr_tail: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum TrialOutcome {
    Success {
        index: usize,
        config: ExecutableConfig,
        metrics: BenchmarkMetrics,
        correctness: Option<CorrectnessResult>,
        model_memory: ModelMemoryOutcome,
        artifacts: Vec<ArtifactManifest>,
    },
    Failed {
        index: usize,
        config: ExecutableConfig,
        failure: TrialFailure,
        metrics: Option<BenchmarkMetrics>,
        correctness: Option<CorrectnessResult>,
        model_memory: ModelMemoryOutcome,
        artifacts: Vec<ArtifactManifest>,
    },
}

impl TrialOutcome {
    pub(crate) fn index(&self) -> usize {
        match self {
            Self::Success { index, .. } | Self::Failed { index, .. } => *index,
        }
    }

    pub(crate) fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct ReportRequest {
    pub run_id: String,
    pub kind: RunKind,
    pub engine: Engine,
    pub winning_metric: Metric,
    pub requested_image: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct RunReport {
    pub schema_version: u32,
    pub run_id: String,
    pub kind: RunKind,
    pub state: RunState,
    pub engine: Engine,
    pub winning_metric: Metric,
    pub requested_image: String,
    pub resolved_image: Option<ResolvedImage>,
    pub selected_hardware: Option<HardwareProfile>,
    pub started_at_unix_ms: u64,
    pub ended_at_unix_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub trials: Vec<TrialOutcome>,
    pub best_trial_index: Option<usize>,
    pub best_winning_value: Option<f64>,
    pub best_config_path: Option<PathBuf>,
    pub run_failure: Option<ErrorPayload>,
    pub submission: Option<SubmissionResult>,
}

impl RunReport {
    pub(crate) fn running(request: ReportRequest, started_at_unix_ms: u64) -> Self {
        Self {
            schema_version: 2,
            run_id: request.run_id,
            kind: request.kind,
            state: RunState::Running,
            engine: request.engine,
            winning_metric: request.winning_metric,
            requested_image: request.requested_image,
            resolved_image: None,
            selected_hardware: None,
            started_at_unix_ms,
            ended_at_unix_ms: None,
            duration_ms: None,
            trials: Vec::new(),
            best_trial_index: None,
            best_winning_value: None,
            best_config_path: None,
            run_failure: None,
            submission: None,
        }
    }

    pub(crate) fn set_preflight(
        &mut self,
        resolved_image: ResolvedImage,
        selected_hardware: HardwareProfile,
    ) -> Result<()> {
        if self.state != RunState::Running || !self.trials.is_empty() {
            return Err(report_validation(
                "preflight identity must be recorded before trials and finalization",
            ));
        }
        if self.resolved_image.is_some() || self.selected_hardware.is_some() {
            return Err(report_validation(
                "preflight identity is immutable once recorded",
            ));
        }
        self.resolved_image = Some(resolved_image);
        self.selected_hardware = Some(selected_hardware);
        Ok(())
    }

    pub(crate) fn push_trial(&mut self, trial: TrialOutcome) -> Result<()> {
        let expected = self.trials.len();
        if self.state != RunState::Running {
            return Err(report_validation("cannot append a trial to a final report"));
        }
        if self.resolved_image.is_none() || self.selected_hardware.is_none() {
            return Err(report_validation(
                "cannot append a trial before preflight identity is recorded",
            ));
        }
        if trial.index() != expected {
            return Err(report_validation(format!(
                "trial index {} is out of order; expected {expected}",
                trial.index()
            )));
        }
        validate_trial(&trial, self.winning_metric)?;
        self.trials.push(trial);
        self.recompute_best();
        Ok(())
    }

    pub(crate) fn best_trial_index(&self) -> Option<usize> {
        self.best_trial_index
    }

    pub(crate) fn set_best_config_path(&mut self, path: PathBuf) -> Result<()> {
        validate_report_relative_path(&path)?;
        if self.best_trial_index.is_none() {
            return Err(report_validation(
                "best config path requires a successful winning trial",
            ));
        }
        self.best_config_path = Some(path);
        Ok(())
    }

    pub(crate) fn set_submission(&mut self, submission: SubmissionResult) -> Result<()> {
        if self.state == RunState::Running {
            return Err(report_validation(
                "leaderboard submission can only be recorded on a final report",
            ));
        }
        if self.submission.is_some() {
            return Err(report_validation(
                "leaderboard submission is already recorded",
            ));
        }
        self.submission = Some(submission);
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        state: RunState,
        ended_at_unix_ms: u64,
        failure: Option<ErrorPayload>,
    ) -> Result<()> {
        if self.state != RunState::Running {
            return Err(report_validation("final report state is immutable"));
        }
        if state == RunState::Running {
            return Err(report_validation("finish requires a final run state"));
        }
        if ended_at_unix_ms < self.started_at_unix_ms {
            return Err(report_validation("run end time precedes start time"));
        }
        let mut finished = self.clone();
        finished.recompute_best();
        finished.state = state;
        finished.ended_at_unix_ms = Some(ended_at_unix_ms);
        finished.duration_ms = Some(ended_at_unix_ms - self.started_at_unix_ms);
        finished.run_failure = failure;
        finished.validate()?;
        *self = finished;
        Ok(())
    }

    pub(crate) fn checkpoint(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let parent = path
            .parent()
            .ok_or_else(|| report_validation("report checkpoint path has no parent directory"))?;
        create_private_dir(parent)?;
        let mut bytes = serde_json::to_vec_pretty(self).map_err(|source| {
            Error::new(
                ErrorKind::Io,
                Some(ExecutionStage::Persistence),
                "failed to serialize report checkpoint",
            )
            .with_report_path(path)
            .with_source(source)
        })?;
        bytes.push(b'\n');
        atomic_write(path, 0o600, &bytes).map_err(|error| error.with_report_path(path))
    }

    fn recompute_best(&mut self) {
        let observations = self
            .trials
            .iter()
            .filter_map(|trial| match trial {
                TrialOutcome::Success {
                    index,
                    metrics,
                    correctness,
                    ..
                } => metrics
                    .value_for(self.winning_metric)
                    .map(|value| RankableObservation {
                        index: *index,
                        correctness: correctness.as_ref().map(|result| result.status),
                        value,
                    }),
                TrialOutcome::Failed { .. } => None,
            })
            .collect::<Vec<_>>();
        self.best_trial_index = select_best(self.winning_metric, &observations);
        self.best_winning_value = self.best_trial_index.and_then(|index| {
            self.trials.get(index).and_then(|trial| match trial {
                TrialOutcome::Success { metrics, .. } => {
                    metrics.value_for(self.winning_metric).map(normalize_zero)
                }
                TrialOutcome::Failed { .. } => None,
            })
        });
    }

    fn validate(&self) -> Result<()> {
        if self.schema_version != 2 {
            return Err(report_validation("report schema_version must be 2"));
        }
        if self.run_id.trim().is_empty() {
            return Err(report_validation("report run_id must not be empty"));
        }
        if self.requested_image.trim().is_empty() {
            return Err(report_validation(
                "report requested_image must not be empty",
            ));
        }
        for (expected, trial) in self.trials.iter().enumerate() {
            if trial.index() != expected {
                return Err(report_validation(format!(
                    "trial index {} is out of order; expected {expected}",
                    trial.index()
                )));
            }
            validate_trial(trial, self.winning_metric)?;
        }
        match (self.best_trial_index, self.best_winning_value) {
            (Some(index), Some(value)) => {
                if !value.is_finite() {
                    return Err(report_validation("best winning value must be finite"));
                }
                let Some(TrialOutcome::Success { metrics, .. }) = self.trials.get(index) else {
                    return Err(report_validation(
                        "best_trial_index must reference a successful trial",
                    ));
                };
                if metrics.value_for(self.winning_metric).map(normalize_zero) != Some(value) {
                    return Err(report_validation(
                        "best winning value does not match the referenced trial",
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(report_validation(
                    "best trial index and winning value must both be present or absent",
                ));
            }
        }
        if let Some(path) = &self.best_config_path {
            validate_report_relative_path(path)?;
            if self.best_trial_index.is_none() {
                return Err(report_validation(
                    "best config path requires a successful winning trial",
                ));
            }
        }

        match self.state {
            RunState::Running => {
                if self.ended_at_unix_ms.is_some()
                    || self.duration_ms.is_some()
                    || self.run_failure.is_some()
                    || self.submission.is_some()
                {
                    return Err(report_validation(
                        "running report must not contain final-only fields",
                    ));
                }
            }
            RunState::Completed | RunState::CompletedWithFailures => {
                self.validate_final_time()?;
                if self.resolved_image.is_none() || self.selected_hardware.is_none() {
                    return Err(report_validation(
                        "completed report requires resolved image and selected hardware",
                    ));
                }
                if self.best_trial_index.is_none() || self.best_config_path.is_none() {
                    return Err(report_validation(
                        "completed report requires a winning trial and installed best config",
                    ));
                }
                if self.run_failure.is_some() {
                    return Err(report_validation(
                        "completed report must not contain a run-level failure",
                    ));
                }
                let has_failed = self.trials.iter().any(|trial| !trial.is_success());
                if (self.state == RunState::CompletedWithFailures) != has_failed {
                    return Err(report_validation(
                        "completed_with_failures must exactly reflect failed trials",
                    ));
                }
            }
            RunState::Failed | RunState::Interrupted => {
                self.validate_final_time()?;
                if self.run_failure.is_none() {
                    return Err(report_validation(
                        "failed or interrupted report requires run failure context",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_final_time(&self) -> Result<()> {
        let ended = self
            .ended_at_unix_ms
            .ok_or_else(|| report_validation("final report requires ended_at_unix_ms"))?;
        let duration = self
            .duration_ms
            .ok_or_else(|| report_validation("final report requires duration_ms"))?;
        if ended < self.started_at_unix_ms || duration != ended - self.started_at_unix_ms {
            return Err(report_validation(
                "report duration does not match start and end time",
            ));
        }
        Ok(())
    }
}

pub(crate) fn install_best_config(run_dir: &Path, config: &ExecutableConfig) -> Result<PathBuf> {
    create_private_dir(run_dir)?;
    let relative = PathBuf::from("best.toml");
    let destination = run_dir.join(&relative);
    let text = config
        .best_config_toml()
        .map_err(|error| error.with_artifact_path(&relative))?;
    atomic_write(&destination, 0o600, text.as_bytes())
        .map_err(|error| error.with_artifact_path(&relative))?;
    Ok(relative)
}

fn validate_trial(trial: &TrialOutcome, selected: Metric) -> Result<()> {
    let (metrics, model_memory, artifacts) = match trial {
        TrialOutcome::Success {
            metrics,
            correctness,
            model_memory,
            artifacts,
            ..
        } => {
            if metrics
                .value_for(selected)
                .is_none_or(|value| !value.is_finite())
            {
                return Err(report_validation(format!(
                    "successful trial is missing finite selected metric {selected}"
                )));
            }
            if correctness
                .as_ref()
                .is_some_and(|result| result.status != CorrectnessStatus::Passed)
            {
                return Err(report_validation(
                    "successful trial correctness must be passed when present",
                ));
            }
            (Some(metrics), model_memory, artifacts)
        }
        TrialOutcome::Failed {
            metrics,
            model_memory,
            artifacts,
            ..
        } => (metrics.as_ref(), model_memory, artifacts),
    };
    if let Some(metrics) = metrics {
        if metrics
            .value_for(selected)
            .is_some_and(|value| !value.is_finite())
        {
            return Err(report_validation("trial metric values must be finite"));
        }
    }
    if model_memory.estimate.is_some() && model_memory.warning.is_some() {
        return Err(report_validation(
            "model-memory outcome cannot contain both estimate and warning",
        ));
    }
    for artifact in artifacts {
        validate_report_relative_path(&artifact.path)?;
    }
    Ok(())
}

fn validate_report_relative_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(report_validation(format!(
            "report path must be nonempty and run-relative: {}",
            path.display()
        )));
    }
    Ok(())
}

fn report_validation(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::Validation,
        Some(ExecutionStage::Persistence),
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn running_report_has_nullable_final_fields() {
        let report = RunReport::running(
            ReportRequest {
                run_id: "run-1".into(),
                kind: RunKind::Bench,
                engine: Engine::Vllm,
                winning_metric: Metric::Tps,
                requested_image: "repo/image:tag".into(),
            },
            1_000,
        );

        let value = serde_json::to_value(report).unwrap();

        assert_eq!(value["schema_version"], 2);
        assert!(value["ended_at_unix_ms"].is_null());
        assert!(value["duration_ms"].is_null());
    }

    #[test]
    fn ranking_preserves_history_and_failed_trials_cannot_win() {
        let mut report = running_report();
        report
            .set_preflight(resolved_image(), hardware_profile())
            .unwrap();
        report.push_trial(success(0, 1.0)).unwrap();
        report.push_trial(success(1, 1.0)).unwrap();
        report.push_trial(failed(2, 100.0)).unwrap();

        assert_eq!(report.best_trial_index(), Some(0));
        assert_eq!(report.best_winning_value, Some(1.0));
        assert_eq!(report.trials[0].index(), 0);
        assert_eq!(report.trials[2].index(), 2);

        report
            .set_best_config_path(PathBuf::from("best.toml"))
            .unwrap();
        report
            .finish(RunState::CompletedWithFailures, 1_250, None)
            .unwrap();
        assert_eq!(report.state, RunState::CompletedWithFailures);
        assert_eq!(report.duration_ms, Some(250));
    }

    #[test]
    fn invalid_finish_is_transactional_and_early_failure_needs_no_preflight() {
        let mut invalid = running_report();
        assert!(invalid.finish(RunState::Completed, 1_100, None).is_err());
        assert_eq!(invalid.state, RunState::Running);
        assert!(invalid.ended_at_unix_ms.is_none());

        let mut failed = running_report();
        failed
            .finish(
                RunState::Failed,
                1_100,
                Some(
                    Error::new(
                        ErrorKind::ProcessExit,
                        Some(ExecutionStage::Preflight),
                        "failed",
                    )
                    .payload(),
                ),
            )
            .unwrap();
        assert_eq!(failed.state, RunState::Failed);
        assert!(failed.resolved_image.is_none());
        assert_eq!(failed.duration_ms, Some(100));
        assert!(failed
            .finish(
                RunState::Interrupted,
                1_200,
                Some(Error::interrupted(ExecutionStage::Preflight).payload()),
            )
            .is_err());
    }

    #[test]
    fn checkpoint_is_valid_json_and_rejects_out_of_order_trials() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("report.json");
        let mut report = running_report();

        assert!(report.push_trial(success(0, 1.0)).is_err());
        assert!(report.trials.is_empty());
        assert!(report.push_trial(success(1, 1.0)).is_err());
        report.checkpoint(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["state"], "running");
        assert!(bytes.ends_with(b"\n"));
    }

    #[test]
    fn installs_a_parseable_winning_config_atomically() {
        let directory = tempfile::tempdir().unwrap();

        let relative = install_best_config(directory.path(), &executable_config()).unwrap();
        let text = std::fs::read_to_string(directory.path().join(&relative)).unwrap();
        let parsed =
            crate::config::ConfigInput::try_from(crate::config::parse_config_text(&text).unwrap())
                .unwrap()
                .normalize()
                .unwrap();

        assert_eq!(relative, PathBuf::from("best.toml"));
        assert_eq!(parsed.image, "repo/image@sha256:abc");
        assert!(parsed.sweep.is_none());
        assert!(!parsed.leaderboard.submit);
    }

    fn running_report() -> RunReport {
        RunReport::running(
            ReportRequest {
                run_id: "run-1".into(),
                kind: RunKind::Sweep,
                engine: Engine::Vllm,
                winning_metric: Metric::Tps,
                requested_image: "repo/image:tag".into(),
            },
            1_000,
        )
    }

    fn executable_config() -> ExecutableConfig {
        crate::config::ConfigInput::minimal(Engine::Vllm, "model")
            .normalize()
            .unwrap()
            .into_executable(resolved_image())
    }

    fn resolved_image() -> ResolvedImage {
        ResolvedImage {
            requested: "repo/image:tag".into(),
            immutable: "repo/image@sha256:abc".into(),
            local_only: false,
        }
    }

    fn hardware_profile() -> HardwareProfile {
        let gpu = crate::domain::run::GpuRecord {
            index: 0,
            uuid: "GPU-0".into(),
            name: "Test GPU".into(),
            compute_capability: Some("9.0".into()),
            memory_total_mib: 80_000,
            memory_free_mib: 79_000,
            memory_used_mib: 1_000,
        };
        HardwareProfile {
            source: "test".into(),
            cuda_visible_devices: None,
            all_gpus: vec![gpu.clone()],
            selected_gpus: vec![gpu],
            warnings: Vec::new(),
        }
    }

    fn success(index: usize, value: f64) -> TrialOutcome {
        TrialOutcome::Success {
            index,
            config: executable_config(),
            metrics: BenchmarkMetrics::parse_for(
                &format!("Output token throughput (tok/s): {value}"),
                Metric::Tps,
            )
            .unwrap(),
            correctness: None,
            model_memory: ModelMemoryOutcome::default(),
            artifacts: Vec::new(),
        }
    }

    fn failed(index: usize, value: f64) -> TrialOutcome {
        TrialOutcome::Failed {
            index,
            config: executable_config(),
            failure: TrialFailure {
                error: Box::new(
                    Error::new(
                        ErrorKind::ProcessExit,
                        Some(ExecutionStage::Benchmark),
                        "failed",
                    )
                    .payload(),
                ),
                timed_out: false,
                interrupted: false,
                stdout_tail: None,
                stderr_tail: None,
            },
            metrics: Some(
                BenchmarkMetrics::parse_for(
                    &format!("Output token throughput (tok/s): {value}"),
                    Metric::Tps,
                )
                .unwrap(),
            ),
            correctness: None,
            model_memory: ModelMemoryOutcome::default(),
            artifacts: Vec::new(),
        }
    }
}
