use std::{
    collections::HashSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use schemars::JsonSchema;
use serde::Serialize;

use crate::{
    cli::args::{self, CommandKind, Invocation, ParsedCli},
    config::{ConfigInput, ExecutableConfig, NormalizedConfig},
    domain::{
        candidate::CandidateSpec,
        engine::Engine,
        run::{ExecutionBackend, ExecutionTarget, PullPolicy, ResolvedImage},
    },
    engines::managed::{managed_run_plan, safe_display, ManagedRunPlan},
    error::{Error, ErrorKind, ErrorPayload, ExecutionStage, Result},
    inspection::{
        correctness::{
            capability_probe_spec, collect_results, lighteval_spec, CorrectnessResult,
            CorrectnessStatus, CorrectnessSuite, DEFAULT_SUITE,
        },
        hardware::{format_hardware_profile, inspect_hardware},
        model_memory::{estimate_model_memory, resolve_hf_mem_command},
    },
    leaderboard::{
        auth::{resolve_hf_token, resolve_submit_key, Secret},
        client::{infer_hf_username, LeaderboardClient},
    },
    results::{
        artifact::ArtifactManifest,
        metrics::BenchmarkMetrics,
        report::{
            install_best_config, ModelMemoryOutcome, ReportRequest, RunKind, RunReport, RunState,
            SubmissionResult, SubmissionState, TrialFailure, TrialOutcome,
        },
    },
    runtime::{
        atomic::{atomic_write, create_private_dir},
        cancel::CancellationToken,
        docker::{
            cleanup_owned_containers, immutable_reference, in_container_image_identity,
            resolve_image,
        },
        params::{cached_parameter_schema, load_parameter_schema},
        process::{
            ArtifactCapture, ProcessCapture, ProcessExecutor, ProcessFailure, ProcessOutcome,
            ProcessSpec,
        },
        server::ManagedServer,
    },
};

const LEADERBOARD_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_ARTIFACTS_PER_TRIAL: usize = 4096;
static RUN_SEQUENCE: AtomicU64 = AtomicU64::new(0);
pub(crate) struct EvaluationResult {
    pub report_path: PathBuf,
    pub report: RunReport,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CorrectnessCheckState {
    Passed,
    Failed,
    Interrupted,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct CorrectnessCheckResult {
    pub schema_version: u32,
    pub run_id: String,
    pub state: CorrectnessCheckState,
    pub started_at_unix_ms: u64,
    pub ended_at_unix_ms: u64,
    pub duration_ms: u64,
    pub config: Option<ExecutableConfig>,
    pub correctness: Option<CorrectnessResult>,
    pub failure: Option<TrialFailure>,
    pub artifacts: Vec<ArtifactManifest>,
    pub report_path: PathBuf,
}

pub fn run(
    args: impl Iterator<Item = String>,
    mut stdout: impl Write,
    mut stderr: impl Write,
) -> Result<()> {
    match args::parse(args)? {
        ParsedCli::Display(text) => write_text(&mut stdout, &text),
        ParsedCli::Invocation(invocation) => {
            let invocation = *invocation;
            match invocation.kind {
                CommandKind::Params => run_params(invocation, &mut stdout),
                CommandKind::Hardware => run_hardware(&mut stdout),
                CommandKind::Plan => run_plan(invocation, &mut stdout),
                CommandKind::Serve => run_serve(invocation, &mut stderr),
                CommandKind::Bench => {
                    if invocation.target == ExecutionTarget::HfJobs {
                        crate::hf_jobs::submit(
                            &invocation,
                            RunKind::Bench,
                            &mut stdout,
                            &mut stderr,
                        )
                    } else {
                        run_evaluation(invocation, RunKind::Bench, &mut stdout, &mut stderr)
                            .map(|_| ())
                    }
                }
                CommandKind::Sweep => {
                    if invocation.target == ExecutionTarget::HfJobs {
                        crate::hf_jobs::submit(
                            &invocation,
                            RunKind::Sweep,
                            &mut stdout,
                            &mut stderr,
                        )
                    } else {
                        run_evaluation(invocation, RunKind::Sweep, &mut stdout, &mut stderr)
                            .map(|_| ())
                    }
                }
                CommandKind::Cleanup => run_cleanup(invocation, &mut stdout),
                CommandKind::Mcp => Err(Error::usage(
                    "mcp must be served through the stdio protocol entry point",
                )),
            }
        }
    }
}

fn run_params(invocation: Invocation, out: &mut impl Write) -> Result<()> {
    let engine = invocation
        .input
        .engine
        .ok_or_else(|| Error::validation("engine is required"))?;
    let image = invocation
        .input
        .image
        .as_deref()
        .unwrap_or_else(|| engine.default_image());
    let executor = ProcessExecutor::default();
    let cancellation = cancellation_token()?;
    let (identity, schema, source) = if invocation.offline_parameters {
        let identity = immutable_reference(image)?.ok_or_else(|| {
            Error::validation("params --offline requires a repository digest or sha256 image ID")
        })?;
        let schema = cached_parameter_schema(engine, &identity, &invocation.parameter_cache_dir)?
            .ok_or_else(|| {
            Error::new(
                ErrorKind::ParameterInspection,
                Some(ExecutionStage::ParameterInspection),
                "offline parameter cache entry was not found",
            )
            .with_cache_identity(&identity)
        })?;
        (identity, schema, "cache")
    } else {
        let identity = resolve_image(
            image,
            invocation
                .input
                .runtime
                .pull_policy
                .unwrap_or(PullPolicy::Missing),
            invocation.input.runtime.allow_local_image.unwrap_or(false),
            &executor,
            &cancellation,
        )?;
        let schema = load_parameter_schema(
            engine,
            &identity.immutable,
            &invocation.parameter_cache_dir,
            invocation.refresh_parameters,
            &executor,
            &cancellation,
            invocation.backend,
        )?;
        (identity.immutable, schema, "runtime_or_cache")
    };
    writeln_checked(out, &format!("image: {identity}"))?;
    writeln_checked(out, &format!("source: {source}"))?;
    for (name, mode) in schema.parameters {
        writeln_checked(out, &format!("--{name}\t{mode:?}"))?;
    }
    Ok(())
}

fn run_hardware(out: &mut impl Write) -> Result<()> {
    let runtime = ConfigInput::minimal(Engine::Vllm, "hardware-inspection")
        .normalize()?
        .runtime;
    let cancellation = cancellation_token()?;
    let profile = inspect_hardware(&runtime, &ProcessExecutor::default(), &cancellation)?;
    write_text(out, &format_hardware_profile(&profile))
}

fn run_cleanup(invocation: Invocation, out: &mut impl Write) -> Result<()> {
    let cancellation = cancellation_token()?;
    let containers = cleanup_owned_containers(
        invocation.cleanup_run_id.as_deref(),
        invocation.cleanup_dry_run,
        &ProcessExecutor::default(),
        &cancellation,
    )?;
    if containers.is_empty() {
        return writeln_checked(out, "containers: 0");
    }
    let label = if invocation.cleanup_dry_run {
        "owned_container"
    } else {
        "removed_container"
    };
    for container in containers {
        writeln_checked(out, &format!("{label}: {container}"))?;
    }
    Ok(())
}

fn run_plan(invocation: Invocation, out: &mut impl Write) -> Result<()> {
    let normalized = invocation.input.clone().normalize()?;
    let resolved = unresolved_image(&normalized);
    let configs = executable_candidates(&normalized, &resolved, RunKind::Bench)?;
    render_plans(&invocation, &configs, out, true)
}

fn run_serve(invocation: Invocation, out: &mut impl Write) -> Result<()> {
    let normalized = invocation.input.clone().normalize()?;
    if !invocation.execute {
        let config = normalized
            .clone()
            .into_executable(unresolved_image(&normalized));
        return render_plans(&invocation, &[config], out, true);
    }

    let cancellation = cancellation_token()?;
    let token = resolve_hf_token(&ProcessExecutor::default(), &cancellation)?;
    let executor =
        ProcessExecutor::with_credentials(token.as_ref().into_iter().map(Secret::expose));
    let identity = match invocation.backend {
        ExecutionBackend::Docker => resolve_image(
            &normalized.image,
            normalized.runtime.pull_policy,
            normalized.runtime.allow_local_image,
            &executor,
            &cancellation,
        )?,
        ExecutionBackend::InContainer => in_container_image_identity(&normalized.image)?,
    };
    let hardware = inspect_hardware(&normalized.runtime, &executor, &cancellation)?;
    writeln_checked(out, &format_hardware_profile(&hardware))?;
    let config = normalized.clone().into_executable(identity.resolved());
    validate_parameter_sets(
        &invocation,
        std::slice::from_ref(&config),
        &identity.immutable,
        &executor,
        &cancellation,
    )?;
    let memory_command = resolve_hf_mem_command(config.model_memory.command.as_deref());
    let _memory = estimate_model_memory(&config, memory_command, &executor, &cancellation)?;

    let artifacts = invocation.results_dir.join("serve");
    create_private_dir(&artifacts)?;
    let mut plan = managed_run_plan(&config, "serve", &artifacts, invocation.backend)?;
    attach_hf_token(&mut plan, token.as_ref(), invocation.backend);
    let server = ManagedServer::start(
        &executor,
        &plan.server,
        plan.readiness.clone(),
        &cancellation,
    )
    .map_err(|failure| failure.error)?;
    let mut server = server
        .wait_ready(&cancellation)
        .map_err(|failure| failure.error)?;
    writeln_checked(out, "server: ready")?;
    loop {
        if cancellation.is_cancelled() {
            let _ = server.stop();
            return Err(Error::interrupted(ExecutionStage::Server));
        }
        if !server.is_running()? {
            let _ = server.stop();
            return Err(Error::new(
                ErrorKind::ProcessExit,
                Some(ExecutionStage::Server),
                "serving process exited",
            ));
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub(crate) fn run_evaluation(
    invocation: Invocation,
    kind: RunKind,
    out: &mut impl Write,
    progress: &mut impl Write,
) -> Result<Option<EvaluationResult>> {
    let cancellation = cancellation_token()?;
    run_evaluation_with_cancellation(invocation, kind, out, progress, &cancellation)
}

pub(crate) fn run_evaluation_with_cancellation(
    invocation: Invocation,
    kind: RunKind,
    out: &mut impl Write,
    progress: &mut impl Write,
    cancellation: &CancellationToken,
) -> Result<Option<EvaluationResult>> {
    let normalized = invocation.input.clone().normalize()?;
    let unresolved = unresolved_image(&normalized);
    let dry_configs = executable_candidates(&normalized, &unresolved, kind)?;
    if !invocation.execute {
        render_plans(&invocation, &dry_configs, out, false)?;
        return Ok(None);
    }

    let run_id = new_run_id(kind);
    let run_dir = invocation.results_dir.join(&run_id);
    create_private_dir(&run_dir)?;
    let report_path = run_dir.join("report.json");
    let mut report = RunReport::running(
        ReportRequest {
            run_id: run_id.clone(),
            kind,
            engine: normalized.engine,
            winning_metric: normalized.metric,
            requested_image: normalized.image.clone(),
        },
        now_millis()?,
    );
    report.checkpoint(&report_path)?;

    let preflight = evaluation_preflight(&invocation, &normalized, kind, cancellation);
    let (credentials, executor, identity, hardware, configs, memory) = match preflight {
        Ok(preflight) => preflight,
        Err(error) => {
            finish_failed_report(&mut report, &report_path, &error, cancellation)?;
            return Err(error.with_report_path(&report_path));
        }
    };
    report.set_preflight(identity.resolved(), hardware)?;
    report.checkpoint(&report_path)?;

    for (index, (config, model_memory)) in configs.into_iter().zip(memory).enumerate() {
        if cancellation.is_cancelled() {
            let error = Error::interrupted(ExecutionStage::Benchmark);
            finish_failed_report(&mut report, &report_path, &error, cancellation)?;
            return Err(error.with_report_path(&report_path));
        }
        writeln_checked(
            progress,
            &format!("trial: {}/{}", index + 1, dry_configs.len()),
        )?;
        let trial = match execute_trial(
            index,
            config,
            model_memory,
            &run_dir,
            &executor,
            credentials.hf_token.as_ref(),
            cancellation,
            invocation.backend,
        ) {
            Ok(trial) => trial,
            Err(error) => {
                finish_failed_report(&mut report, &report_path, &error, cancellation)?;
                return Err(error.with_report_path(&report_path));
            }
        };
        report.push_trial(trial)?;
        report.checkpoint(&report_path)?;
    }

    if cancellation.is_cancelled() {
        let error = Error::interrupted(ExecutionStage::Benchmark);
        finish_failed_report(&mut report, &report_path, &error, cancellation)?;
        return Err(error.with_report_path(&report_path));
    }
    let Some(best_index) = report.best_trial_index() else {
        let diagnostic = report.trials.iter().find_map(|trial| {
            let TrialOutcome::Failed { failure, .. } = trial else {
                return None;
            };
            [&failure.stderr_tail, &failure.stdout_tail]
                .into_iter()
                .flatten()
                .map(|tail| tail.trim())
                .find(|tail| !tail.is_empty())
                .or_else(|| {
                    Some(failure.error.message.trim()).filter(|message| !message.is_empty())
                })
        });
        let message = diagnostic.map_or_else(
            || "all benchmark candidates failed".to_string(),
            |diagnostic| format!("all benchmark candidates failed:\n{diagnostic}"),
        );
        let error = Error::new(
            ErrorKind::Benchmark,
            Some(ExecutionStage::Benchmark),
            message,
        );
        finish_failed_report(&mut report, &report_path, &error, cancellation)?;
        return Err(error.with_report_path(&report_path));
    };
    let best_config = match &report.trials[best_index] {
        TrialOutcome::Success { config, .. } => config.clone(),
        TrialOutcome::Failed { .. } => unreachable!("best trial is always successful"),
    };
    let best_path = install_best_config(&run_dir, &best_config)?;
    let best_output_path = run_dir.join(&best_path);
    report.set_best_config_path(best_path)?;
    let state = if report.trials.iter().any(|trial| !trial.is_success()) {
        RunState::CompletedWithFailures
    } else {
        RunState::Completed
    };
    report.finish(state, now_millis()?, None)?;
    report.checkpoint(&report_path)?;

    if let Some(submission) = credentials.submission.as_ref() {
        let client = LeaderboardClient::new(&normalized.leaderboard.url, LEADERBOARD_TIMEOUT)?;
        let report_json = fs::read_to_string(&report_path).map_err(|source| {
            Error::new(
                ErrorKind::Io,
                Some(ExecutionStage::Leaderboard),
                "failed to read final report for leaderboard submission",
            )
            .with_report_path(&report_path)
            .with_source(source)
        })?;
        match client.submit_report(
            &report_json,
            &submission.contributor,
            submission.submit_key.as_ref(),
            credentials.hf_token.as_ref(),
        ) {
            Ok(result) => {
                let state = if result.message.starts_with("Accepted:") {
                    SubmissionState::Accepted
                } else {
                    SubmissionState::Queued
                };
                let remote_id = result
                    .message
                    .split_once(':')
                    .map(|(_, value)| value.trim().to_string())
                    .filter(|value| !value.is_empty());
                report.set_submission(SubmissionResult {
                    state,
                    message: result.message,
                    remote_id,
                })?;
            }
            Err(error) => {
                report.set_submission(SubmissionResult {
                    state: SubmissionState::Failed,
                    message: error.to_string(),
                    remote_id: None,
                })?;
                report.checkpoint(&report_path)?;
                return Err(error.with_report_path(&report_path));
            }
        }
        report.checkpoint(&report_path)?;
    }

    writeln_checked(out, &format!("report: {}", report_path.display()))
        .map_err(|error| error.with_report_path(&report_path))?;
    writeln_checked(
        out,
        &format!("winning_config: {}", best_output_path.display()),
    )
    .map_err(|error| error.with_report_path(&report_path))?;
    Ok(Some(EvaluationResult {
        report_path,
        report,
    }))
}

struct CorrectnessRun {
    run_id: String,
    run_dir: PathBuf,
    report_path: PathBuf,
    started_at_unix_ms: u64,
}

pub(crate) fn run_correctness_check_with_cancellation(
    invocation: Invocation,
    cancellation: &CancellationToken,
) -> Result<CorrectnessCheckResult> {
    let normalized = invocation.input.clone().normalize()?;
    if !normalized.correctness.enabled {
        return Err(Error::validation(
            "correctness must be enabled for a correctness-only run",
        ));
    }
    if normalized.sweep.is_some() {
        return Err(Error::validation(
            "correctness-only runs accept exactly one candidate",
        ));
    }

    let run_id = new_prefixed_run_id("correctness");
    let run_dir = invocation.results_dir.join(&run_id);
    create_private_dir(&run_dir)?;
    let run = CorrectnessRun {
        report_path: run_dir.join("report.json"),
        run_id,
        run_dir,
        started_at_unix_ms: now_millis()?,
    };
    let (credentials, executor, _, _, mut configs, mut memory) =
        match evaluation_preflight(&invocation, &normalized, RunKind::Bench, cancellation) {
            Ok(preflight) => preflight,
            Err(error) => return persist_correctness_error(&run, None, Vec::new(), error),
        };
    if configs.len() != 1 || memory.len() != 1 {
        return persist_correctness_error(
            &run,
            None,
            Vec::new(),
            Error::validation("correctness-only preflight must produce exactly one candidate"),
        );
    }
    let config = configs.remove(0);
    let config_for_error = config.clone();
    let execution = match execute_trial_steps(
        TrialExecutionInput {
            index: 0,
            config,
            model_memory: memory.remove(0),
            run_benchmark: false,
        },
        &run.run_dir,
        &executor,
        credentials.hf_token.as_ref(),
        cancellation,
        invocation.backend,
    ) {
        Ok(execution) => execution,
        Err(error) => {
            return persist_correctness_error(&run, Some(config_for_error), Vec::new(), error);
        }
    };
    let artifacts =
        match collect_artifacts(&run.run_dir, &execution.relative_dir, &execution.truncated) {
            Ok(artifacts) => artifacts,
            Err(error) => {
                return persist_correctness_error(&run, Some(execution.config), Vec::new(), error);
            }
        };
    if let Some(failure) = execution.failure.as_ref() {
        let error = Error {
            kind: failure.error.kind,
            stage: failure.error.stage,
            message: failure.error.message.clone(),
            context: Box::new(failure.error.context.clone()),
            source: None,
        };
        let state = if failure.interrupted {
            CorrectnessCheckState::Interrupted
        } else {
            CorrectnessCheckState::Failed
        };
        let result = finish_correctness_result(
            &run,
            state,
            Some(execution.config),
            execution.correctness,
            execution.failure,
            artifacts,
        )?;
        return Err(error.with_report_path(result.report_path));
    }
    let correctness = match execution.correctness {
        Some(correctness) => correctness,
        None => {
            return persist_correctness_error(
                &run,
                Some(execution.config),
                artifacts,
                Error::new(
                    ErrorKind::Correctness,
                    Some(ExecutionStage::ResultCollection),
                    "correctness-only run completed without correctness results",
                ),
            );
        }
    };
    let state = match correctness.status {
        CorrectnessStatus::Passed => CorrectnessCheckState::Passed,
        CorrectnessStatus::Failed => CorrectnessCheckState::Failed,
    };
    finish_correctness_result(
        &run,
        state,
        Some(execution.config),
        Some(correctness),
        None,
        artifacts,
    )
}

fn persist_correctness_error(
    run: &CorrectnessRun,
    config: Option<ExecutableConfig>,
    artifacts: Vec<ArtifactManifest>,
    error: Error,
) -> Result<CorrectnessCheckResult> {
    let failure = error_trial_failure(error);
    let returned = Error {
        kind: failure.error.kind,
        stage: failure.error.stage,
        message: failure.error.message.clone(),
        context: Box::new(failure.error.context.clone()),
        source: None,
    };
    let state = if failure.interrupted {
        CorrectnessCheckState::Interrupted
    } else {
        CorrectnessCheckState::Failed
    };
    let result = finish_correctness_result(run, state, config, None, Some(failure), artifacts)?;
    Err(returned.with_report_path(result.report_path))
}

fn finish_correctness_result(
    run: &CorrectnessRun,
    state: CorrectnessCheckState,
    config: Option<ExecutableConfig>,
    correctness: Option<CorrectnessResult>,
    failure: Option<TrialFailure>,
    artifacts: Vec<ArtifactManifest>,
) -> Result<CorrectnessCheckResult> {
    let ended_at_unix_ms = now_millis()?;
    let result = CorrectnessCheckResult {
        schema_version: 1,
        run_id: run.run_id.clone(),
        state,
        started_at_unix_ms: run.started_at_unix_ms,
        ended_at_unix_ms,
        duration_ms: ended_at_unix_ms.saturating_sub(run.started_at_unix_ms),
        config,
        correctness,
        failure,
        artifacts,
        report_path: run.report_path.clone(),
    };
    let mut bytes = serde_json::to_vec_pretty(&result).map_err(|source| {
        Error::new(
            ErrorKind::Io,
            Some(ExecutionStage::Persistence),
            "failed to encode correctness report",
        )
        .with_source(source)
        .with_report_path(&run.report_path)
    })?;
    bytes.push(b'\n');
    atomic_write(&run.report_path, 0o600, &bytes)
        .map_err(|error| error.with_report_path(&run.report_path))?;
    Ok(result)
}

struct PreparedSubmission {
    contributor: String,
    submit_key: Option<Secret>,
}

struct RuntimeCredentials {
    hf_token: Option<Secret>,
    submission: Option<PreparedSubmission>,
}

type Preflight = (
    RuntimeCredentials,
    ProcessExecutor,
    crate::runtime::docker::DockerImageIdentity,
    crate::domain::run::HardwareProfile,
    Vec<ExecutableConfig>,
    Vec<ModelMemoryOutcome>,
);

fn evaluation_preflight(
    invocation: &Invocation,
    normalized: &NormalizedConfig,
    kind: RunKind,
    cancellation: &CancellationToken,
) -> Result<Preflight> {
    let hf_token = resolve_hf_token(&ProcessExecutor::default(), cancellation)?;
    let submission = if normalized.leaderboard.submit {
        let contributor = infer_hf_username(hf_token.as_ref())?;
        Some(PreparedSubmission {
            contributor,
            submit_key: resolve_submit_key()?,
        })
    } else {
        None
    };
    let mut redactions = hf_token
        .as_ref()
        .into_iter()
        .map(Secret::expose)
        .collect::<Vec<_>>();
    if let Some(key) = submission
        .as_ref()
        .and_then(|submission| submission.submit_key.as_ref())
    {
        redactions.push(key.expose());
    }
    let executor = ProcessExecutor::with_credentials(redactions);
    let identity = match invocation.backend {
        ExecutionBackend::Docker => resolve_image(
            &normalized.image,
            normalized.runtime.pull_policy,
            normalized.runtime.allow_local_image,
            &executor,
            cancellation,
        )?,
        ExecutionBackend::InContainer => in_container_image_identity(&normalized.image)?,
    };
    let hardware = inspect_hardware(&normalized.runtime, &executor, cancellation)?;
    let configs = executable_candidates(normalized, &identity.resolved(), kind)?;
    validate_parameter_sets(
        invocation,
        &configs,
        &identity.immutable,
        &executor,
        cancellation,
    )?;
    let memory_command = resolve_hf_mem_command(normalized.model_memory.command.as_deref());
    let memory = configs
        .iter()
        .map(|config| {
            estimate_model_memory(config, memory_command.clone(), &executor, cancellation)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        RuntimeCredentials {
            hf_token,
            submission,
        },
        executor,
        identity,
        hardware,
        configs,
        memory,
    ))
}

fn validate_parameter_sets(
    invocation: &Invocation,
    configs: &[ExecutableConfig],
    image_identity: &str,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<()> {
    let engine = configs
        .first()
        .map(|config| config.engine)
        .ok_or_else(|| Error::validation("candidate expansion returned no configurations"))?;
    let schema = load_parameter_schema(
        engine,
        image_identity,
        &invocation.parameter_cache_dir,
        invocation.refresh_parameters,
        executor,
        cancellation,
        invocation.backend,
    )?;
    for config in configs {
        schema.validate(&config.serve_args)?;
    }
    Ok(())
}

fn preview_parameter_validation(
    invocation: &Invocation,
    configs: &[ExecutableConfig],
) -> Result<&'static str> {
    let first = configs
        .first()
        .ok_or_else(|| Error::validation("candidate expansion returned no configurations"))?;
    let Some(identity) = immutable_reference(&first.image.requested)? else {
        return Ok("pending_runtime");
    };
    let Some(schema) =
        cached_parameter_schema(first.engine, &identity, &invocation.parameter_cache_dir)?
    else {
        return Ok("pending_runtime");
    };
    for config in configs {
        schema.validate(&config.serve_args)?;
    }
    Ok("cached")
}

fn executable_candidates(
    normalized: &NormalizedConfig,
    resolved: &ResolvedImage,
    kind: RunKind,
) -> Result<Vec<ExecutableConfig>> {
    let base = CandidateSpec {
        candidate: normalized.candidate.clone(),
        serve_args: normalized.serve_args.clone(),
    };
    let candidates = match kind {
        RunKind::Bench => vec![base],
        RunKind::Sweep => normalized
            .sweep
            .as_ref()
            .ok_or_else(|| Error::validation("sweep configuration is required"))?
            .candidates(&base)?,
    };
    candidates
        .into_iter()
        .map(|candidate| {
            let mut config = normalized.clone().into_executable(resolved.clone());
            config.candidate = candidate.candidate;
            config.serve_args = candidate.serve_args;
            Ok(config)
        })
        .collect()
}

fn render_plans(
    invocation: &Invocation,
    configs: &[ExecutableConfig],
    out: &mut impl Write,
    plan_labels: bool,
) -> Result<()> {
    let validation = preview_parameter_validation(invocation, configs)?;
    writeln_checked(out, &format!("validation: {validation}"))?;
    for (index, config) in configs.iter().enumerate() {
        if configs.len() > 1 {
            writeln_checked(out, &format!("trial: {}/{}", index + 1, configs.len()))?;
        }
        let plan = managed_run_plan(
            config,
            &format!("dry-{index}"),
            Path::new(".optimum-advisor/dry-run"),
            invocation.backend,
        )?;
        let server_label = if plan_labels { "serve" } else { "server" };
        let benchmark_label = if plan_labels { "bench" } else { "benchmark" };
        writeln_checked(
            out,
            &format!("{server_label}: {}", plan.server.safe_display),
        )?;
        writeln_checked(
            out,
            &format!("{benchmark_label}: {}", plan.benchmark.safe_display),
        )?;
        if config.correctness.enabled {
            let suite = correctness_suite(config.correctness.threshold);
            let directory = Path::new(".optimum-advisor/dry-run/correctness");
            writeln_checked(
                out,
                &format!(
                    "correctness: {}",
                    lighteval_spec(config, &suite, directory).safe_display
                ),
            )?;
            if let Some(probe) = capability_probe_spec(config, directory) {
                writeln_checked(out, &format!("capability_probe: {}", probe.safe_display))?;
            }
        }
    }
    Ok(())
}

struct TrialExecutionInput {
    index: usize,
    config: ExecutableConfig,
    model_memory: ModelMemoryOutcome,
    run_benchmark: bool,
}

struct TrialExecution {
    config: ExecutableConfig,
    model_memory: ModelMemoryOutcome,
    relative_dir: PathBuf,
    truncated: HashSet<PathBuf>,
    failure: Option<TrialFailure>,
    metrics: Option<BenchmarkMetrics>,
    correctness: Option<CorrectnessResult>,
}

#[allow(clippy::too_many_arguments)]
fn execute_trial(
    index: usize,
    config: ExecutableConfig,
    model_memory: ModelMemoryOutcome,
    run_dir: &Path,
    executor: &ProcessExecutor,
    hf_token: Option<&Secret>,
    cancellation: &CancellationToken,
    backend: ExecutionBackend,
) -> Result<TrialOutcome> {
    let TrialExecution {
        config,
        model_memory,
        relative_dir,
        truncated,
        failure,
        metrics,
        correctness,
    } = execute_trial_steps(
        TrialExecutionInput {
            index,
            config,
            model_memory,
            run_benchmark: true,
        },
        run_dir,
        executor,
        hf_token,
        cancellation,
        backend,
    )?;
    finish_trial(
        index,
        config,
        model_memory,
        run_dir,
        &relative_dir,
        truncated,
        failure,
        metrics,
        correctness,
    )
}

fn execute_trial_steps(
    input: TrialExecutionInput,
    run_dir: &Path,
    executor: &ProcessExecutor,
    hf_token: Option<&Secret>,
    cancellation: &CancellationToken,
    backend: ExecutionBackend,
) -> Result<TrialExecution> {
    let TrialExecutionInput {
        index,
        config,
        model_memory,
        run_benchmark,
    } = input;
    let relative_dir = PathBuf::from("trials").join(format!("{index:04}"));
    let trial_dir = run_dir.join(&relative_dir);
    create_private_dir(&trial_dir)?;
    let mut truncated = HashSet::new();
    let mut failure = None;
    let mut metrics = None;
    let mut correctness = None;

    let mut plan = match managed_run_plan(&config, &format!("trial-{index}"), &trial_dir, backend) {
        Ok(plan) => plan,
        Err(error) => {
            failure = Some(error_trial_failure(error));
            return Ok(TrialExecution {
                config,
                model_memory,
                relative_dir,
                truncated,
                failure,
                metrics,
                correctness,
            });
        }
    };
    attach_hf_token(&mut plan, hf_token, backend);
    let mut server =
        match ManagedServer::start(executor, &plan.server, plan.readiness.clone(), cancellation) {
            Ok(server) => match server.wait_ready(cancellation) {
                Ok(server) => Some(server),
                Err(process_failure) => {
                    failure = Some(capture_process_failure(
                        process_failure,
                        &plan.server,
                        run_dir,
                        &mut truncated,
                    ));
                    None
                }
            },
            Err(process_failure) => {
                failure = Some(capture_process_failure(
                    process_failure,
                    &plan.server,
                    run_dir,
                    &mut truncated,
                ));
                None
            }
        };

    if failure.is_none() && config.correctness.enabled {
        let directory = trial_dir.join("correctness");
        create_private_dir(&directory)?;
        let suite = correctness_suite(config.correctness.threshold);
        let mut lighteval = lighteval_spec(&config, &suite, &directory);
        if let Some(token) = hf_token {
            lighteval = lighteval.with_env("HF_TOKEN", token.expose());
        }
        match executor.execute(&lighteval, cancellation) {
            Ok(outcome) => {
                if let Err(problem) =
                    capture_process_outcome(outcome, &lighteval, run_dir, &mut truncated)
                {
                    failure = Some(problem);
                }
            }
            Err(problem) => {
                failure = Some(capture_process_failure(
                    problem,
                    &lighteval,
                    run_dir,
                    &mut truncated,
                ));
            }
        }
        if failure.is_none() {
            if let Some(mut probe) = capability_probe_spec(&config, &directory) {
                if let Some(token) = hf_token {
                    probe = probe.with_env("HF_TOKEN", token.expose());
                }
                match executor.execute(&probe, cancellation) {
                    Ok(outcome) => {
                        if let Err(problem) =
                            capture_process_outcome(outcome, &probe, run_dir, &mut truncated)
                        {
                            failure = Some(problem);
                        }
                    }
                    Err(problem) => {
                        failure = Some(capture_process_failure(
                            problem,
                            &probe,
                            run_dir,
                            &mut truncated,
                        ));
                    }
                }
            }
        }
        if failure.is_none() {
            match collect_results(&suite, &directory, &config.serve_args) {
                Ok(result) => {
                    failure = correctness_gate_failure(&result);
                    correctness = Some(result);
                }
                Err(error) => failure = Some(error_trial_failure(error)),
            }
        }
    }

    if failure.is_none() && run_benchmark {
        match executor.execute(&plan.benchmark, cancellation) {
            Ok(outcome) => {
                match capture_process_outcome(outcome, &plan.benchmark, run_dir, &mut truncated) {
                    Ok(_) => match read_benchmark_metrics(&plan.benchmark) {
                        Ok(parsed) => {
                            if let Err(error) = parsed.validate_for(config.metric) {
                                failure = Some(error_trial_failure(error));
                            }
                            metrics = Some(parsed);
                        }
                        Err(error) => failure = Some(error_trial_failure(error)),
                    },
                    Err(problem) => failure = Some(problem),
                }
            }
            Err(problem) => {
                failure = Some(capture_process_failure(
                    problem,
                    &plan.benchmark,
                    run_dir,
                    &mut truncated,
                ));
            }
        }
    }

    if let Some(mut active) = server.take() {
        if failure.is_none() {
            match active.is_running() {
                Ok(true) => {}
                Ok(false) => {
                    failure = Some(error_trial_failure(Error::new(
                        ErrorKind::ProcessExit,
                        Some(ExecutionStage::Server),
                        "server exited before the trial workload completed",
                    )));
                }
                Err(error) => failure = Some(error_trial_failure(error)),
            }
        }
        match active.stop() {
            Ok(outcome) => {
                if let Err(problem) =
                    capture_process_outcome(outcome, &plan.server, run_dir, &mut truncated)
                {
                    merge_trial_failure(&mut failure, problem);
                }
            }
            Err(problem) => {
                let problem =
                    capture_process_failure(problem, &plan.server, run_dir, &mut truncated);
                merge_trial_failure(&mut failure, problem);
            }
        }
    }

    Ok(TrialExecution {
        config,
        model_memory,
        relative_dir,
        truncated,
        failure,
        metrics,
        correctness,
    })
}

fn correctness_gate_failure(result: &CorrectnessResult) -> Option<TrialFailure> {
    if result.status == CorrectnessStatus::Passed {
        return None;
    }
    let failed_tasks = result
        .tasks
        .iter()
        .filter(|task| task.score < result.threshold)
        .map(|task| format!("{} {}={}", task.spec, task.metric, task.score))
        .collect::<Vec<_>>();
    let failed_capabilities = result
        .capabilities
        .iter()
        .filter(|capability| !capability.passed)
        .map(|capability| format!("{}:{}", capability.domain, capability.parser))
        .collect::<Vec<_>>();
    let message = match (failed_tasks.is_empty(), failed_capabilities.is_empty()) {
        (false, true) => format!(
            "correctness threshold {} was not met by tasks: {}",
            result.threshold,
            failed_tasks.join(", ")
        ),
        (true, false) => format!(
            "correctness capability checks failed: {}",
            failed_capabilities.join(", ")
        ),
        (false, false) => format!(
            "correctness threshold {} was not met by tasks: {}; capability checks failed: {}",
            result.threshold,
            failed_tasks.join(", "),
            failed_capabilities.join(", ")
        ),
        (true, true) => "correctness requirements were not met".to_string(),
    };
    Some(error_trial_failure(Error::new(
        ErrorKind::Correctness,
        Some(ExecutionStage::Correctness),
        message,
    )))
}

#[allow(clippy::too_many_arguments)]
fn finish_trial(
    index: usize,
    config: ExecutableConfig,
    model_memory: ModelMemoryOutcome,
    run_dir: &Path,
    relative_dir: &Path,
    truncated: HashSet<PathBuf>,
    failure: Option<TrialFailure>,
    metrics: Option<BenchmarkMetrics>,
    correctness: Option<CorrectnessResult>,
) -> Result<TrialOutcome> {
    let artifacts = collect_artifacts(run_dir, relative_dir, &truncated)?;
    Ok(match failure {
        Some(failure) => TrialOutcome::Failed {
            index,
            config,
            failure,
            metrics,
            correctness,
            model_memory,
            artifacts,
        },
        None => TrialOutcome::Success {
            index,
            config,
            metrics: metrics.ok_or_else(|| {
                Error::new(
                    ErrorKind::Benchmark,
                    Some(ExecutionStage::ResultCollection),
                    "successful trial is missing benchmark metrics",
                )
            })?,
            correctness,
            model_memory,
            artifacts,
        },
    })
}

fn read_benchmark_metrics(spec: &ProcessSpec) -> Result<BenchmarkMetrics> {
    let path = spec.stdout_artifact.as_deref().ok_or_else(|| {
        Error::new(
            ErrorKind::Io,
            Some(ExecutionStage::ResultCollection),
            "benchmark process has no stdout artifact",
        )
    })?;
    let text = fs::read_to_string(path).map_err(|source| {
        Error::new(
            ErrorKind::Io,
            Some(ExecutionStage::ResultCollection),
            "failed to read benchmark output artifact",
        )
        .with_artifact_path(path)
        .with_source(source)
    })?;
    BenchmarkMetrics::parse(&text)
}

fn attach_hf_token(plan: &mut ManagedRunPlan, token: Option<&Secret>, backend: ExecutionBackend) {
    let Some(token) = token else {
        return;
    };
    for process in [&mut plan.server, &mut plan.benchmark] {
        process
            .env_add
            .push(("HF_TOKEN".into(), token.expose().into()));
        // The Docker CLI needs an explicit `-e HF_TOKEN` to forward the host
        // variable into the container; a directly launched engine already
        // inherits it from the child environment.
        if backend == ExecutionBackend::Docker {
            process.args.insert(2, "HF_TOKEN".into());
            process.args.insert(2, "-e".into());
        }
        process.safe_display = safe_display(&process.program, &process.args);
    }
}

fn capture_process_outcome(
    outcome: ProcessOutcome,
    spec: &ProcessSpec,
    run_dir: &Path,
    truncated: &mut HashSet<PathBuf>,
) -> std::result::Result<ArtifactCapture, TrialFailure> {
    let ProcessOutcome {
        capture,
        cleanup_failure,
        ..
    } = outcome;
    let ProcessCapture::Artifacts(capture) = capture else {
        return Err(error_trial_failure(Error::new(
            ErrorKind::Protocol,
            Some(ExecutionStage::ResultCollection),
            "artifact-producing process returned secret capture",
        )));
    };
    record_truncation(spec, &capture, run_dir, truncated);
    if let Some(error) = cleanup_failure {
        return Err(payload_trial_failure(error, &capture));
    }
    Ok(capture)
}

fn capture_process_failure(
    failure: ProcessFailure,
    spec: &ProcessSpec,
    run_dir: &Path,
    truncated: &mut HashSet<PathBuf>,
) -> TrialFailure {
    let ProcessFailure {
        error,
        capture,
        cleanup_failure,
    } = failure;
    if let Some(capture) = capture.as_ref() {
        record_truncation(spec, capture, run_dir, truncated);
    }
    let mut result = match capture.as_ref() {
        Some(capture) => payload_trial_failure(error.payload(), capture),
        None => error_trial_failure(error),
    };
    if let Some(cleanup) = cleanup_failure {
        result.error.message.push_str(&format!(
            "; owned-container cleanup also failed: {}",
            cleanup.message
        ));
    }
    result
}

fn payload_trial_failure(error: ErrorPayload, capture: &ArtifactCapture) -> TrialFailure {
    TrialFailure {
        timed_out: error.kind == ErrorKind::Timeout,
        interrupted: error.kind == ErrorKind::Interrupted,
        stdout_tail: nonempty(&capture.stdout.tail),
        stderr_tail: nonempty(&capture.stderr.tail),
        error: Box::new(error),
    }
}

fn error_trial_failure(error: Error) -> TrialFailure {
    let payload = error.payload();
    TrialFailure {
        timed_out: payload.kind == ErrorKind::Timeout,
        interrupted: payload.kind == ErrorKind::Interrupted,
        error: Box::new(payload),
        stdout_tail: None,
        stderr_tail: None,
    }
}

fn merge_trial_failure(current: &mut Option<TrialFailure>, additional: TrialFailure) {
    if let Some(current) = current {
        current.error.message.push_str(&format!(
            "; cleanup also failed: {}",
            additional.error.message
        ));
        current.timed_out |= additional.timed_out;
        current.interrupted |= additional.interrupted;
    } else {
        *current = Some(additional);
    }
}

fn record_truncation(
    spec: &ProcessSpec,
    capture: &ArtifactCapture,
    run_dir: &Path,
    truncated: &mut HashSet<PathBuf>,
) {
    for (path, was_truncated) in [
        (spec.stdout_artifact.as_ref(), capture.stdout.truncated),
        (spec.stderr_artifact.as_ref(), capture.stderr.truncated),
    ] {
        if was_truncated {
            if let Some(relative) = path.and_then(|path| path.strip_prefix(run_dir).ok()) {
                truncated.insert(relative.to_path_buf());
            }
        }
    }
}

fn collect_artifacts(
    run_dir: &Path,
    root: &Path,
    truncated: &HashSet<PathBuf>,
) -> Result<Vec<ArtifactManifest>> {
    let mut paths = Vec::new();
    collect_artifact_paths(&run_dir.join(root), run_dir, 0, &mut paths)?;
    paths.sort();
    if paths.len() > MAX_ARTIFACTS_PER_TRIAL {
        return Err(Error::new(
            ErrorKind::OutputTruncated,
            Some(ExecutionStage::Persistence),
            format!("trial produced more than {MAX_ARTIFACTS_PER_TRIAL} artifacts"),
        ));
    }
    paths
        .into_iter()
        .map(|path| ArtifactManifest::from_path(run_dir, &path, truncated.contains(&path)))
        .collect()
}

fn collect_artifact_paths(
    directory: &Path,
    run_dir: &Path,
    depth: usize,
    paths: &mut Vec<PathBuf>,
) -> Result<()> {
    if depth > 16 {
        return Err(
            Error::validation("artifact directory nesting exceeds 16 levels")
                .with_artifact_path(directory),
        );
    }
    let entries = fs::read_dir(directory).map_err(|source| {
        Error::new(
            ErrorKind::Io,
            Some(ExecutionStage::Persistence),
            "failed to enumerate trial artifacts",
        )
        .with_artifact_path(directory)
        .with_source(source)
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| {
            Error::new(
                ErrorKind::Io,
                Some(ExecutionStage::Persistence),
                "failed to inspect trial artifact entry",
            )
            .with_artifact_path(directory)
            .with_source(source)
        })?;
        let file_type = entry.file_type().map_err(|source| {
            Error::new(
                ErrorKind::Io,
                Some(ExecutionStage::Persistence),
                "failed to inspect trial artifact type",
            )
            .with_artifact_path(entry.path())
            .with_source(source)
        })?;
        if file_type.is_symlink() {
            return Err(
                Error::validation("trial artifact must not be a symbolic link")
                    .with_artifact_path(entry.path()),
            );
        }
        if file_type.is_dir() {
            collect_artifact_paths(&entry.path(), run_dir, depth + 1, paths)?;
        } else if file_type.is_file() {
            let path = entry.path();
            let relative = path.strip_prefix(run_dir).map_err(|_| {
                Error::validation("trial artifact escaped the run directory")
                    .with_artifact_path(&path)
            })?;
            paths.push(relative.to_path_buf());
        }
    }
    Ok(())
}

fn finish_failed_report(
    report: &mut RunReport,
    report_path: &Path,
    error: &Error,
    cancellation: &CancellationToken,
) -> Result<()> {
    let state = if error.kind() == ErrorKind::Interrupted || cancellation.is_cancelled() {
        RunState::Interrupted
    } else {
        RunState::Failed
    };
    report.finish(state, now_millis()?, Some(error.payload()))?;
    report.checkpoint(report_path)
}

fn correctness_suite(threshold: f64) -> CorrectnessSuite {
    CorrectnessSuite {
        id: DEFAULT_SUITE.id,
        threshold,
        max_samples: DEFAULT_SUITE.max_samples,
        tasks: DEFAULT_SUITE.tasks,
    }
}

fn unresolved_image(config: &NormalizedConfig) -> ResolvedImage {
    ResolvedImage {
        requested: config.image.clone(),
        immutable: config.image.clone(),
        local_only: config.runtime.allow_local_image,
    }
}

fn cancellation_token() -> Result<CancellationToken> {
    let cancellation = CancellationToken::new();
    cancellation.register_os_signals()?;
    Ok(cancellation)
}

fn new_run_id(kind: RunKind) -> String {
    let prefix = match kind {
        RunKind::Bench => "bench",
        RunKind::Sweep => "sweep",
    };
    new_prefixed_run_id(prefix)
}

fn new_prefixed_run_id(prefix: &str) -> String {
    let sequence = RUN_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!(
        "{prefix}-{}-{}-{sequence}",
        now_millis().unwrap_or_default(),
        std::process::id()
    )
}

fn now_millis() -> Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|source| {
            Error::new(
                ErrorKind::Io,
                Some(ExecutionStage::Persistence),
                "system clock is before the Unix epoch",
            )
            .with_source(source)
        })?
        .as_millis();
    u64::try_from(millis).map_err(|_| {
        Error::new(
            ErrorKind::Io,
            Some(ExecutionStage::Persistence),
            "Unix timestamp does not fit in u64 milliseconds",
        )
    })
}

fn nonempty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}

fn write_text(out: &mut impl Write, text: &str) -> Result<()> {
    out.write_all(text.as_bytes()).map_err(write_error)
}

fn writeln_checked(out: &mut impl Write, text: &str) -> Result<()> {
    writeln!(out, "{text}").map_err(write_error)
}

fn write_error(source: std::io::Error) -> Error {
    Error::new(ErrorKind::Io, None, "failed to write command output").with_source(source)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn generated_run_ids_do_not_collide_within_one_process() {
        let ids = (0..1_024)
            .map(|_| new_run_id(RunKind::Bench))
            .collect::<HashSet<_>>();
        assert_eq!(ids.len(), 1_024);
        assert!(ids.iter().all(|id| id.starts_with("bench-")));
    }

    #[test]
    fn hugging_face_token_is_attached_only_when_present() {
        let config = ConfigInput::minimal(Engine::Vllm, "repo/model")
            .normalize()
            .unwrap()
            .into_executable(ResolvedImage {
                requested: "repo/image:tag".into(),
                immutable: "repo/image@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                local_only: false,
            });
        let directory = tempfile::tempdir().unwrap();
        let mut plan =
            managed_run_plan(&config, "run-1", directory.path(), ExecutionBackend::Docker).unwrap();

        for process in [&plan.server, &plan.benchmark] {
            assert!(!process.args.iter().any(|arg| arg == "HF_TOKEN"));
            assert!(process.env_add.is_empty());
        }

        let token = Secret::new("hf_test_secret").unwrap();
        attach_hf_token(&mut plan, Some(&token), ExecutionBackend::Docker);
        for process in [&plan.server, &plan.benchmark] {
            assert!(process.args.iter().any(|arg| arg == "HF_TOKEN"));
            assert!(process
                .env_add
                .iter()
                .any(|(name, value)| name == "HF_TOKEN" && value == token.expose()));
            assert!(!process.safe_display.contains(token.expose()));
        }
    }

    #[test]
    fn in_container_hf_token_is_passed_through_child_env_not_docker_flags() {
        let config = ConfigInput::minimal(Engine::Vllm, "repo/model")
            .normalize()
            .unwrap()
            .into_executable(ResolvedImage {
                requested: "repo/image:tag".into(),
                immutable: "repo/image@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                local_only: false,
            });
        let directory = tempfile::tempdir().unwrap();
        let mut plan = managed_run_plan(
            &config,
            "run-1",
            directory.path(),
            ExecutionBackend::InContainer,
        )
        .unwrap();

        let token = Secret::new("hf_test_secret").unwrap();
        attach_hf_token(&mut plan, Some(&token), ExecutionBackend::InContainer);
        for process in [&plan.server, &plan.benchmark] {
            // No Docker `-e HF_TOKEN` argument leaks into the engine command line.
            assert!(!process
                .args
                .iter()
                .any(|arg| arg == "-e" || arg == "HF_TOKEN"));
            assert!(process
                .env_add
                .iter()
                .any(|(name, value)| name == "HF_TOKEN" && value == token.expose()));
            assert!(!process.safe_display.contains(token.expose()));
        }
    }
}
