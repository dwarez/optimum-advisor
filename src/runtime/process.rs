use std::{
    collections::VecDeque,
    ffi::{OsStr, OsString},
    fmt,
    io::{Read, Write},
    path::PathBuf,
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

use zeroize::Zeroizing;

use crate::{
    error::{Error, ErrorKind, ErrorPayload, ExecutionStage, Result},
    runtime::{
        atomic::{create_private_dir, AtomicWriter},
        cancel::CancellationToken,
        docker::OwnedContainer,
        sanitize::StreamSanitizer,
    },
};

pub(crate) const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
pub(crate) const DEFAULT_CORRECTNESS_TIMEOUT: Duration = Duration::from_secs(600);
pub(crate) const DEFAULT_BENCHMARK_TIMEOUT: Duration = Duration::from_secs(1_800);
pub(crate) const DEFAULT_INSPECTION_TIMEOUT: Duration = Duration::from_secs(300);
pub(crate) const DEFAULT_MAX_PROCESS_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const DIAGNOSTIC_TAIL_BYTES: usize = 64 * 1024;
pub(crate) const SECRET_CAPTURE_BYTES: usize = 64 * 1024;
pub(crate) const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const DRAIN_CHUNK_BYTES: usize = 8 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CapturePolicy {
    ArtifactTails,
    Secret,
}

#[derive(Clone)]
pub(crate) struct ProcessSpec {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub env_add: Vec<(OsString, OsString)>,
    pub env_remove: Vec<OsString>,
    pub cwd: Option<PathBuf>,
    pub deadline: Option<Instant>,
    pub stage: Option<ExecutionStage>,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
    pub stdout_artifact: Option<PathBuf>,
    pub stderr_artifact: Option<PathBuf>,
    pub owned_container: Option<OwnedContainer>,
    pub safe_display: String,
    pub capture: CapturePolicy,
}

impl ProcessSpec {
    pub(crate) fn new<I, A>(program: impl Into<OsString>, args: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<OsString>,
    {
        let program = program.into();
        let safe_display = program.to_string_lossy().into_owned();
        Self {
            program,
            args: args.into_iter().map(Into::into).collect(),
            env_add: Vec::new(),
            env_remove: Vec::new(),
            cwd: None,
            deadline: None,
            stage: None,
            max_stdout_bytes: DEFAULT_MAX_PROCESS_OUTPUT_BYTES,
            max_stderr_bytes: DEFAULT_MAX_PROCESS_OUTPUT_BYTES,
            stdout_artifact: None,
            stderr_artifact: None,
            owned_container: None,
            safe_display,
            capture: CapturePolicy::ArtifactTails,
        }
    }

    pub(crate) fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub(crate) fn with_timeout(self, timeout: Duration) -> Self {
        self.with_deadline(Instant::now() + timeout)
    }

    pub(crate) fn with_stage(mut self, stage: ExecutionStage) -> Self {
        self.stage = Some(stage);
        self
    }

    pub(crate) fn with_artifacts(
        mut self,
        stdout: impl Into<PathBuf>,
        stderr: impl Into<PathBuf>,
    ) -> Self {
        self.stdout_artifact = Some(stdout.into());
        self.stderr_artifact = Some(stderr.into());
        self
    }

    #[cfg(test)]
    pub(crate) fn with_stdout_artifact(mut self, path: impl Into<PathBuf>) -> Self {
        self.stdout_artifact = Some(path.into());
        self
    }

    pub(crate) fn with_capture(mut self, capture: CapturePolicy) -> Self {
        self.capture = capture;
        if capture == CapturePolicy::Secret {
            self.stdout_artifact = None;
            self.stderr_artifact = None;
        }
        self
    }

    pub(crate) fn with_safe_display(mut self, display: impl Into<String>) -> Self {
        self.safe_display = display.into();
        self
    }

    #[allow(dead_code)]
    pub(crate) fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    pub(crate) fn with_env(
        mut self,
        name: impl Into<OsString>,
        value: impl Into<OsString>,
    ) -> Self {
        self.env_add.push((name.into(), value.into()));
        self
    }

    #[allow(dead_code)]
    pub(crate) fn without_env(mut self, name: impl Into<OsString>) -> Self {
        self.env_remove.push(name.into());
        self
    }

    pub(crate) fn with_owned_container(mut self, container: OwnedContainer) -> Self {
        self.owned_container = Some(container);
        self
    }
}

impl fmt::Debug for ProcessSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessSpec")
            .field("safe_display", &self.safe_display)
            .field("cwd", &self.cwd)
            .field("deadline", &self.deadline)
            .field("stage", &self.stage)
            .field("max_stdout_bytes", &self.max_stdout_bytes)
            .field("max_stderr_bytes", &self.max_stderr_bytes)
            .field("stdout_artifact", &self.stdout_artifact)
            .field("stderr_artifact", &self.stderr_artifact)
            .field("owned_container", &self.owned_container)
            .field("capture", &self.capture)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StreamOutcome {
    pub artifact: Option<PathBuf>,
    pub tail: String,
    pub observed_bytes: u64,
    pub persisted_bytes: u64,
    pub truncated: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ArtifactCapture {
    pub stdout: StreamOutcome,
    pub stderr: StreamOutcome,
}

pub(crate) struct SecretOutput(Zeroizing<String>);

impl SecretOutput {
    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

#[derive(Debug)]
pub(crate) enum ProcessCapture {
    Artifacts(ArtifactCapture),
    Secret(SecretOutput),
}

#[derive(Debug)]
pub(crate) struct ProcessOutcome {
    #[allow(dead_code)]
    pub status: ExitStatus,
    #[allow(dead_code)]
    pub duration: Duration,
    pub capture: ProcessCapture,
    pub cleanup_failure: Option<ErrorPayload>,
}

#[derive(Debug)]
pub(crate) struct ProcessFailure {
    pub error: Error,
    pub capture: Option<Box<ArtifactCapture>>,
    pub cleanup_failure: Option<Box<ErrorPayload>>,
}

impl ProcessFailure {
    pub(crate) fn diagnostic_tail(&self) -> Option<&str> {
        self.capture.as_deref().and_then(|capture| {
            [&capture.stderr.tail, &capture.stdout.tail]
                .into_iter()
                .map(|tail| tail.trim())
                .find(|tail| !tail.is_empty())
        })
    }
}

pub(crate) struct ProcessExecutor {
    credentials: Vec<Zeroizing<String>>,
    docker_program: OsString,
}

impl Default for ProcessExecutor {
    fn default() -> Self {
        Self {
            credentials: Vec::new(),
            docker_program: OsString::from("docker"),
        }
    }
}

impl fmt::Debug for ProcessExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcessExecutor")
            .field("registered_credentials", &self.credentials.len())
            .field("docker_program", &self.docker_program)
            .finish()
    }
}

impl ProcessExecutor {
    pub(crate) fn with_credentials<I, S>(credentials: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut values = credentials
            .into_iter()
            .map(|credential| Zeroizing::new(credential.as_ref().to_string()))
            .filter(|credential| !credential.is_empty())
            .collect::<Vec<_>>();
        values.sort_unstable_by_key(|credential| std::cmp::Reverse(credential.len()));
        values.dedup_by(|left, right| left.as_str() == right.as_str());
        Self {
            credentials: values,
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub(crate) fn with_docker_program(mut self, program: impl Into<OsString>) -> Self {
        self.docker_program = program.into();
        self
    }

    pub(crate) fn docker_program(&self) -> &OsStr {
        &self.docker_program
    }

    pub(crate) fn execute(
        &self,
        spec: &ProcessSpec,
        cancellation: &CancellationToken,
    ) -> std::result::Result<ProcessOutcome, ProcessFailure> {
        self.execute_inner(spec, cancellation)
    }

    fn execute_inner(
        &self,
        spec: &ProcessSpec,
        cancellation: &CancellationToken,
    ) -> std::result::Result<ProcessOutcome, ProcessFailure> {
        self.spawn(spec, cancellation)?.wait(cancellation)
    }

    pub(crate) fn spawn<'a>(
        &'a self,
        spec: &'a ProcessSpec,
        cancellation: &CancellationToken,
    ) -> std::result::Result<ManagedProcess<'a>, ProcessFailure> {
        if cancellation.is_cancelled() {
            return Err(ProcessFailure {
                error: Error::interrupted(spec.stage.unwrap_or(ExecutionStage::Preflight))
                    .with_process(&spec.safe_display),
                capture: None,
                cleanup_failure: None,
            });
        }
        if spec.capture == CapturePolicy::ArtifactTails
            && (spec.max_stdout_bytes == 0 || spec.max_stderr_bytes == 0)
        {
            return Err(ProcessFailure {
                error: Error::validation("process output limits must be greater than zero")
                    .with_process(&spec.safe_display),
                capture: None,
                cleanup_failure: None,
            });
        }
        if spec
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
        {
            return Err(ProcessFailure {
                error: Error::new(
                    ErrorKind::Timeout,
                    spec.stage,
                    format!(
                        "process deadline elapsed before spawn: {}",
                        spec.safe_display
                    ),
                )
                .with_process(&spec.safe_display),
                capture: None,
                cleanup_failure: None,
            });
        }

        let started = Instant::now();
        let mut command = Command::new(&spec.program);
        command
            .args(&spec.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (name, value) in &spec.env_add {
            command.env(name, value);
        }
        for name in &spec.env_remove {
            command.env_remove(name);
        }
        if let Some(cwd) = &spec.cwd {
            command.current_dir(cwd);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let mut child = command.spawn().map_err(|source| ProcessFailure {
            error: Error::new(
                ErrorKind::ProcessSpawn,
                spec.stage,
                format!("failed to spawn process: {}", spec.safe_display),
            )
            .with_process(&spec.safe_display)
            .with_source(source),
            capture: None,
            cleanup_failure: None,
        })?;
        let stdout = child.stdout.take().expect("piped stdout is available");
        let stderr = child.stderr.take().expect("piped stderr is available");
        let secret_capture = spec.capture == CapturePolicy::Secret;
        let stdout_limit = if secret_capture {
            SECRET_CAPTURE_BYTES as u64
        } else {
            spec.max_stdout_bytes
        };
        let stderr_limit = if secret_capture {
            SECRET_CAPTURE_BYTES as u64
        } else {
            spec.max_stderr_bytes
        };
        let stdout_thread = spawn_drain(
            stdout,
            stdout_limit,
            spec.stdout_artifact.clone(),
            self.credentials.clone(),
            secret_capture,
        );
        let stderr_thread = spawn_drain(
            stderr,
            stderr_limit,
            spec.stderr_artifact.clone(),
            self.credentials.clone(),
            secret_capture,
        );
        Ok(ManagedProcess {
            executor: self,
            spec,
            child,
            started,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
            finished: false,
        })
    }

    fn cleanup_container(&self, container: &OwnedContainer) -> Option<ErrorPayload> {
        let spec = ProcessSpec::new(
            self.docker_program.clone(),
            [
                OsString::from("rm"),
                OsString::from("-f"),
                OsString::from(&container.name),
            ],
        )
        .with_timeout(Duration::from_secs(30))
        .with_safe_display(format!("docker rm -f {}", container.name));
        self.execute_inner(&spec, &CancellationToken::new())
            .err()
            .map(|failure| {
                failure
                    .error
                    .with_docker_identity(container.run_id.clone())
                    .with_container(container.name.clone())
                    .payload()
            })
    }
}

pub(crate) struct ManagedProcess<'a> {
    executor: &'a ProcessExecutor,
    spec: &'a ProcessSpec,
    child: Child,
    started: Instant,
    stdout_thread: Option<thread::JoinHandle<Result<DrainResult>>>,
    stderr_thread: Option<thread::JoinHandle<Result<DrainResult>>>,
    finished: bool,
}

impl ManagedProcess<'_> {
    pub(crate) fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        self.child.try_wait().map_err(|source| {
            Error::new(
                ErrorKind::Io,
                self.spec.stage,
                format!("failed to query process: {}", self.spec.safe_display),
            )
            .with_process(&self.spec.safe_display)
            .with_source(source)
        })
    }

    pub(crate) fn wait(
        mut self,
        cancellation: &CancellationToken,
    ) -> std::result::Result<ProcessOutcome, ProcessFailure> {
        let stop = wait_for_child(&mut self.child, self.spec.deadline, cancellation);
        let (status, stop_error) = match stop {
            Ok(ChildStop::Exited(status)) => {
                terminate_remaining_group(self.child.id());
                (Some(status), None)
            }
            Ok(ChildStop::TimedOut) => {
                let status = terminate_child(&mut self.child).ok();
                let deadline_ms = self.spec.deadline.map(|deadline| {
                    deadline.saturating_duration_since(self.started).as_millis() as u64
                });
                let mut error = Error::new(
                    ErrorKind::Timeout,
                    self.spec.stage,
                    format!("process timed out: {}", self.spec.safe_display),
                )
                .with_process(&self.spec.safe_display);
                if let Some(deadline_ms) = deadline_ms {
                    error = error.with_deadline_ms(deadline_ms);
                }
                (status, Some(error))
            }
            Ok(ChildStop::Cancelled) => {
                let status = terminate_child(&mut self.child).ok();
                (
                    status,
                    Some(
                        Error::interrupted(self.spec.stage.unwrap_or(ExecutionStage::Preflight))
                            .with_process(&self.spec.safe_display),
                    ),
                )
            }
            Err(error) => {
                let status = terminate_child(&mut self.child).ok();
                (status, Some(error.with_process(&self.spec.safe_display)))
            }
        };
        self.finalize(status, stop_error, true, false)
    }

    pub(crate) fn terminate(mut self) -> std::result::Result<ProcessOutcome, ProcessFailure> {
        let (status, stop_error) = match self.child.try_wait() {
            Ok(Some(status)) => {
                terminate_remaining_group(self.child.id());
                (Some(status), None)
            }
            Ok(None) => match terminate_child(&mut self.child) {
                Ok(status) => (Some(status), None),
                Err(error) => (None, Some(error.with_process(&self.spec.safe_display))),
            },
            Err(source) => (
                None,
                Some(
                    Error::new(
                        ErrorKind::Io,
                        self.spec.stage,
                        format!("failed to query process: {}", self.spec.safe_display),
                    )
                    .with_process(&self.spec.safe_display)
                    .with_source(source),
                ),
            ),
        };
        self.finalize(status, stop_error, false, true)
    }

    fn finalize(
        mut self,
        status: Option<ExitStatus>,
        stop_error: Option<Error>,
        require_success: bool,
        cleanup_on_success: bool,
    ) -> std::result::Result<ProcessOutcome, ProcessFailure> {
        let stdout_result = join_drain(
            self.stdout_thread
                .take()
                .expect("stdout drain is available before finalization"),
            "stdout",
        );
        let stderr_result = join_drain(
            self.stderr_thread
                .take()
                .expect("stderr drain is available before finalization"),
            "stderr",
        );
        let drains = match (stdout_result, stderr_result) {
            (Ok(stdout), Ok(stderr)) => Ok((stdout, stderr)),
            (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
            (Err(first), Err(_)) => Err(first),
        };

        let mut failure_error = stop_error;
        if require_success && failure_error.is_none() {
            if let (Ok(_), Some(status)) = (&drains, status.as_ref()) {
                if !status.success() {
                    let mut error = Error::new(
                        ErrorKind::ProcessExit,
                        self.spec.stage,
                        format!("process exited unsuccessfully: {}", self.spec.safe_display),
                    )
                    .with_process(&self.spec.safe_display);
                    if let Some(code) = status.code() {
                        error = error.with_child_exit_code(code);
                    }
                    failure_error = Some(error);
                }
            }
        }

        let duration = self.started.elapsed();
        let secret_capture = self.spec.capture == CapturePolicy::Secret;
        let result = match drains {
            Err(error) => Err(ProcessFailure {
                error,
                capture: None,
                cleanup_failure: None,
            }),
            Ok((stdout, stderr)) if secret_capture => {
                if let Some(error) = failure_error {
                    Err(ProcessFailure {
                        error,
                        capture: None,
                        cleanup_failure: None,
                    })
                } else if stdout.outcome.truncated || stderr.outcome.truncated {
                    Err(ProcessFailure {
                        error: Error::new(
                            ErrorKind::OutputTruncated,
                            self.spec.stage,
                            format!(
                                "secret-producing process exceeded {SECRET_CAPTURE_BYTES} bytes"
                            ),
                        )
                        .with_process(&self.spec.safe_display),
                        capture: None,
                        cleanup_failure: None,
                    })
                } else {
                    let bytes = stdout.secret.unwrap_or_default();
                    match String::from_utf8(bytes.to_vec()) {
                        Ok(value) => Ok(ProcessOutcome {
                            status: status.expect("successful process has an exit status"),
                            duration,
                            capture: ProcessCapture::Secret(SecretOutput(Zeroizing::new(value))),
                            cleanup_failure: None,
                        }),
                        Err(_) => Err(ProcessFailure {
                            error: Error::new(
                                ErrorKind::Protocol,
                                self.spec.stage,
                                "secret-producing process returned non-UTF-8 output",
                            )
                            .with_process(&self.spec.safe_display),
                            capture: None,
                            cleanup_failure: None,
                        }),
                    }
                }
            }
            Ok((stdout, stderr)) => {
                let capture = ArtifactCapture {
                    stdout: stdout.outcome,
                    stderr: stderr.outcome,
                };
                if let Some(error) = failure_error {
                    Err(ProcessFailure {
                        error,
                        capture: Some(Box::new(capture)),
                        cleanup_failure: None,
                    })
                } else {
                    Ok(ProcessOutcome {
                        status: status.expect("successful process has an exit status"),
                        duration,
                        capture: ProcessCapture::Artifacts(capture),
                        cleanup_failure: None,
                    })
                }
            }
        };

        self.finished = true;
        match result {
            Ok(mut outcome) => {
                if cleanup_on_success {
                    if let Some(container) = &self.spec.owned_container {
                        outcome.cleanup_failure = self.executor.cleanup_container(container);
                    }
                }
                Ok(outcome)
            }
            Err(mut failure) => {
                if let Some(container) = &self.spec.owned_container {
                    failure.cleanup_failure =
                        self.executor.cleanup_container(container).map(Box::new);
                }
                Err(failure)
            }
        }
    }
}

impl Drop for ManagedProcess<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let _ = terminate_child(&mut self.child);
        if let Some(handle) = self.stdout_thread.take() {
            let _ = join_drain(handle, "stdout");
        }
        if let Some(handle) = self.stderr_thread.take() {
            let _ = join_drain(handle, "stderr");
        }
        if let Some(container) = &self.spec.owned_container {
            let _ = self.executor.cleanup_container(container);
        }
        self.finished = true;
    }
}

enum ChildStop {
    Exited(ExitStatus),
    TimedOut,
    Cancelled,
}

fn wait_for_child(
    child: &mut Child,
    deadline: Option<Instant>,
    cancellation: &CancellationToken,
) -> Result<ChildStop> {
    loop {
        if let Some(status) = child.try_wait().map_err(|source| {
            Error::new(ErrorKind::Io, None, "failed to poll child process").with_source(source)
        })? {
            return Ok(ChildStop::Exited(status));
        }
        if cancellation.is_cancelled() {
            return Ok(ChildStop::Cancelled);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Ok(ChildStop::TimedOut);
        }
        let sleep = deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
            .map_or(POLL_INTERVAL, |remaining| remaining.min(POLL_INTERVAL));
        if !sleep.is_zero() {
            thread::sleep(sleep);
        }
    }
}

#[cfg(unix)]
fn terminate_child(child: &mut Child) -> Result<ExitStatus> {
    use nix::{
        errno::Errno,
        sys::signal::{killpg, Signal},
        unistd::Pid,
    };

    let group = Pid::from_raw(child.id() as i32);
    if let Err(error) = killpg(group, Signal::SIGTERM) {
        if error != Errno::ESRCH {
            return Err(
                Error::new(ErrorKind::Io, None, "failed to terminate process group")
                    .with_source(error),
            );
        }
    }
    let grace_deadline = Instant::now() + SHUTDOWN_GRACE;
    loop {
        if let Some(status) = child.try_wait().map_err(|source| {
            Error::new(ErrorKind::Io, None, "failed to reap terminated child").with_source(source)
        })? {
            return Ok(status);
        }
        if Instant::now() >= grace_deadline {
            break;
        }
        thread::sleep(POLL_INTERVAL);
    }
    if let Err(error) = killpg(group, Signal::SIGKILL) {
        if error != Errno::ESRCH {
            return Err(
                Error::new(ErrorKind::Io, None, "failed to kill process group").with_source(error),
            );
        }
    }
    child.wait().map_err(|source| {
        Error::new(ErrorKind::Io, None, "failed to reap killed child").with_source(source)
    })
}

#[cfg(not(unix))]
fn terminate_child(child: &mut Child) -> Result<ExitStatus> {
    child.kill().map_err(|source| {
        Error::new(ErrorKind::Io, None, "failed to terminate child process").with_source(source)
    })?;
    child.wait().map_err(|source| {
        Error::new(ErrorKind::Io, None, "failed to reap terminated child").with_source(source)
    })
}

#[cfg(unix)]
fn terminate_remaining_group(id: u32) {
    use nix::{
        sys::signal::{killpg, Signal},
        unistd::Pid,
    };

    let group = Pid::from_raw(id as i32);
    let _ = killpg(group, Signal::SIGTERM);
    thread::sleep(Duration::from_millis(10));
    let _ = killpg(group, Signal::SIGKILL);
}

#[cfg(not(unix))]
fn terminate_remaining_group(_id: u32) {}

struct DrainResult {
    outcome: StreamOutcome,
    secret: Option<Zeroizing<Vec<u8>>>,
}

fn spawn_drain<R: Read + Send + 'static>(
    reader: R,
    limit: u64,
    artifact: Option<PathBuf>,
    credentials: Vec<Zeroizing<String>>,
    secret: bool,
) -> thread::JoinHandle<Result<DrainResult>> {
    thread::spawn(move || drain_stream(reader, limit, artifact, credentials, secret))
}

fn drain_stream(
    mut reader: impl Read,
    limit: u64,
    artifact: Option<PathBuf>,
    credentials: Vec<Zeroizing<String>>,
    secret: bool,
) -> Result<DrainResult> {
    let mut writer = if let Some(path) = &artifact {
        let parent = path.parent().ok_or_else(|| {
            Error::validation("process artifact path has no parent").with_artifact_path(path)
        })?;
        create_private_dir(parent)?;
        Some(AtomicWriter::create(path, 0o600)?)
    } else {
        None
    };
    let credential_refs = if secret {
        Vec::new()
    } else {
        credentials
            .iter()
            .map(|value| value.as_str())
            .collect::<Vec<_>>()
    };
    let mut sanitizer = StreamSanitizer::new(&credential_refs);
    let mut observed_bytes = 0u64;
    let mut sanitized_bytes = 0u64;
    let mut persisted_bytes = 0u64;
    let mut tail = VecDeque::with_capacity(DIAGNOSTIC_TAIL_BYTES);
    let mut secret_bytes = secret.then(|| Zeroizing::new(Vec::with_capacity(SECRET_CAPTURE_BYTES)));
    let mut buffer = [0u8; DRAIN_CHUNK_BYTES];

    loop {
        let read = reader.read(&mut buffer).map_err(|source| {
            Error::new(ErrorKind::Io, None, "failed to drain process output").with_source(source)
        })?;
        if read == 0 {
            break;
        }
        observed_bytes = observed_bytes.checked_add(read as u64).ok_or_else(|| {
            Error::new(
                ErrorKind::Io,
                None,
                "observed process output byte count overflowed",
            )
        })?;
        let sanitized = sanitizer.push(&buffer[..read]);
        consume_sanitized(
            &sanitized,
            limit,
            &mut writer,
            &mut tail,
            secret_bytes.as_mut(),
            &mut sanitized_bytes,
            &mut persisted_bytes,
        )?;
    }
    let final_bytes = sanitizer.finish();
    consume_sanitized(
        &final_bytes,
        limit,
        &mut writer,
        &mut tail,
        secret_bytes.as_mut(),
        &mut sanitized_bytes,
        &mut persisted_bytes,
    )?;
    if let Some(writer) = writer {
        writer.commit()?;
    }

    let tail = if secret {
        String::new()
    } else {
        bounded_utf8_tail(tail)
    };
    Ok(DrainResult {
        outcome: StreamOutcome {
            artifact,
            tail,
            observed_bytes,
            persisted_bytes,
            truncated: sanitized_bytes > limit,
        },
        secret: secret_bytes,
    })
}

#[allow(clippy::too_many_arguments)]
fn consume_sanitized(
    bytes: &[u8],
    limit: u64,
    writer: &mut Option<AtomicWriter>,
    tail: &mut VecDeque<u8>,
    secret: Option<&mut Zeroizing<Vec<u8>>>,
    sanitized_bytes: &mut u64,
    persisted_bytes: &mut u64,
) -> Result<()> {
    let remaining = limit.saturating_sub(*sanitized_bytes) as usize;
    let retained = bytes.len().min(remaining);
    if let Some(secret) = secret {
        secret.extend_from_slice(&bytes[..retained]);
    } else {
        for &byte in bytes {
            if tail.len() == DIAGNOSTIC_TAIL_BYTES {
                tail.pop_front();
            }
            tail.push_back(byte);
        }
        if let Some(writer) = writer {
            writer.write_all(&bytes[..retained]).map_err(|source| {
                Error::new(
                    ErrorKind::Io,
                    None,
                    "failed to write process output artifact",
                )
                .with_source(source)
            })?;
            *persisted_bytes = persisted_bytes
                .checked_add(retained as u64)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::Io,
                        None,
                        "persisted output byte count overflowed",
                    )
                })?;
        }
    }
    *sanitized_bytes = sanitized_bytes
        .checked_add(bytes.len() as u64)
        .ok_or_else(|| {
            Error::new(
                ErrorKind::Io,
                None,
                "sanitized output byte count overflowed",
            )
        })?;
    Ok(())
}

fn bounded_utf8_tail(bytes: VecDeque<u8>) -> String {
    let text = String::from_utf8_lossy(&bytes.into_iter().collect::<Vec<_>>()).into_owned();
    if text.len() <= DIAGNOSTIC_TAIL_BYTES {
        return text;
    }
    let mut start = text.len() - DIAGNOSTIC_TAIL_BYTES;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_string()
}

fn join_drain(
    handle: thread::JoinHandle<Result<DrainResult>>,
    stream: &'static str,
) -> Result<DrainResult> {
    handle.join().map_err(|_| {
        Error::new(
            ErrorKind::Io,
            None,
            format!("{stream} drain thread panicked"),
        )
        .with_stream(stream)
    })?
}

#[cfg(all(test, unix))]
mod tests {
    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        path::Path,
        thread,
        time::{Duration, Instant},
    };

    use tempfile::tempdir;

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn captures_normal_stdout_and_stderr() {
        let directory = tempdir().unwrap();
        let spec = shell("printf 'hello'; printf 'warning' >&2").with_artifacts(
            directory.path().join("stdout.log"),
            directory.path().join("stderr.log"),
        );

        let outcome = ProcessExecutor::default()
            .execute(&spec, &CancellationToken::new())
            .unwrap();
        let ProcessCapture::Artifacts(capture) = outcome.capture else {
            panic!("expected artifact capture")
        };

        assert_eq!(capture.stdout.tail, "hello");
        assert_eq!(capture.stderr.tail, "warning");
        assert_eq!(
            fs::read(directory.path().join("stdout.log")).unwrap(),
            b"hello"
        );
        assert_eq!(
            fs::read(directory.path().join("stderr.log")).unwrap(),
            b"warning"
        );
        assert!(!capture.stdout.truncated);
    }

    #[test]
    fn reports_nonzero_exit_with_sanitized_tails() {
        let spec = shell(r"printf '\033[31mboom\033[0m' >&2; exit 7");

        let failure = ProcessExecutor::default()
            .execute(&spec, &CancellationToken::new())
            .unwrap_err();

        assert_eq!(failure.error.kind(), ErrorKind::ProcessExit);
        assert_eq!(failure.error.context.child_exit_code, Some(7));
        assert_eq!(failure.capture.unwrap().stderr.tail, "boom");
    }

    #[test]
    fn deadline_terminates_a_sleeping_process_group() {
        let spec = shell("sleep 30").with_deadline(Instant::now() + Duration::from_millis(100));
        let started = Instant::now();

        let failure = ProcessExecutor::default()
            .execute(&spec, &CancellationToken::new())
            .unwrap_err();

        assert_eq!(failure.error.kind(), ErrorKind::Timeout);
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn cancellation_kills_spawned_descendants() {
        let directory = tempdir().unwrap();
        let pid_path = directory.path().join("descendant.pid");
        let script = format!("sleep 30 & echo $! > '{}'; wait", pid_path.display());
        let spec = shell(&script);
        let cancellation = CancellationToken::new();
        let worker_token = cancellation.clone();
        let worker =
            thread::spawn(move || ProcessExecutor::default().execute(&spec, &worker_token));
        wait_for_file(&pid_path);
        let pid: i32 = fs::read_to_string(&pid_path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();

        cancellation.cancel();
        let failure = worker.join().unwrap().unwrap_err();

        assert_eq!(failure.error.kind(), ErrorKind::Interrupted);
        let deadline = Instant::now() + Duration::from_secs(2);
        while process_exists(pid) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(
            !process_exists(pid),
            "descendant process {pid} survived cancellation"
        );
    }

    #[test]
    fn drains_excess_output_and_persists_only_the_cap() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("stdout.log");
        let mut spec =
            shell("i=0; while [ $i -lt 1000 ]; do printf '0123456789'; i=$((i+1)); done")
                .with_stdout_artifact(path.clone());
        spec.max_stdout_bytes = 4 * 1024;

        let outcome = ProcessExecutor::default()
            .execute(&spec, &CancellationToken::new())
            .unwrap();
        let ProcessCapture::Artifacts(capture) = outcome.capture else {
            panic!("expected artifact capture")
        };

        assert_eq!(capture.stdout.observed_bytes, 10_000);
        assert_eq!(capture.stdout.persisted_bytes, 4 * 1024);
        assert_eq!(fs::metadata(path).unwrap().len(), 4 * 1024);
        assert!(capture.stdout.truncated);
    }

    #[test]
    fn secret_capture_creates_no_artifact_and_failure_is_redacted() {
        let directory = tempdir().unwrap();
        let artifact = directory.path().join("must-not-exist");
        let success = shell("printf 'super-secret'")
            .with_stdout_artifact(artifact.clone())
            .with_capture(CapturePolicy::Secret);

        let outcome = ProcessExecutor::default()
            .execute(&success, &CancellationToken::new())
            .unwrap();
        let ProcessCapture::Secret(secret) = outcome.capture else {
            panic!("expected secret capture")
        };
        assert_eq!(secret.expose(), "super-secret");
        assert!(!artifact.exists());

        let failure = ProcessExecutor::default()
            .execute(
                &shell("printf 'super-secret'; exit 1").with_capture(CapturePolicy::Secret),
                &CancellationToken::new(),
            )
            .unwrap_err();
        assert!(!format!("{failure:?}").contains("super-secret"));
        assert!(failure.capture.is_none());
    }

    #[test]
    fn registered_credentials_are_redacted_before_persistence() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("stdout.log");
        let spec = shell("printf 'prefix-token-value-suffix'").with_stdout_artifact(path.clone());
        let executor = ProcessExecutor::with_credentials(["token-value"]);

        let outcome = executor.execute(&spec, &CancellationToken::new()).unwrap();
        let ProcessCapture::Artifacts(capture) = outcome.capture else {
            panic!("expected artifact capture")
        };

        assert_eq!(
            fs::read_to_string(path).unwrap(),
            "prefix-[REDACTED]-suffix"
        );
        assert_eq!(capture.stdout.tail, "prefix-[REDACTED]-suffix");
    }

    #[test]
    fn failed_owned_process_runs_bounded_container_cleanup() {
        let directory = tempdir().unwrap();
        let log = directory.path().join("docker.log");
        let docker = directory.path().join("docker");
        fs::write(
            &docker,
            format!("#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\n", log.display()),
        )
        .unwrap();
        let mut permissions = fs::metadata(&docker).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&docker, permissions).unwrap();
        let container = OwnedContainer::new("owned-server", "run-1", "server");
        let spec = shell("exit 9").with_owned_container(container);
        let executor = ProcessExecutor::default().with_docker_program(&docker);

        let failure = executor
            .execute(&spec, &CancellationToken::new())
            .unwrap_err();

        assert!(failure.cleanup_failure.is_none());
        assert_eq!(
            fs::read_to_string(log).unwrap().trim(),
            "rm -f owned-server"
        );
    }

    #[test]
    fn diagnostic_tails_are_valid_utf8_and_byte_bounded() {
        let bytes = VecDeque::from(vec![0xff; DIAGNOSTIC_TAIL_BYTES]);

        let tail = bounded_utf8_tail(bytes);

        assert!(tail.len() <= DIAGNOSTIC_TAIL_BYTES);
        assert!(std::str::from_utf8(tail.as_bytes()).is_ok());
    }

    #[test]
    fn managed_process_can_be_polled_and_stopped() {
        let directory = tempdir().unwrap();
        let ready = directory.path().join("ready");
        let spec = shell(&format!(
            "printf ready; : > '{}'; sleep 30",
            ready.display()
        ));
        let executor = ProcessExecutor::default();
        let cancellation = CancellationToken::new();
        let mut process = executor.spawn(&spec, &cancellation).unwrap();
        wait_for_file(&ready);

        assert!(process.try_wait().unwrap().is_none());
        let outcome = process.terminate().unwrap();
        let ProcessCapture::Artifacts(capture) = outcome.capture else {
            panic!("expected artifact capture")
        };

        assert_eq!(capture.stdout.tail, "ready");
    }

    #[test]
    fn managed_process_wait_reports_an_unexpected_exit() {
        let spec = shell("exit 7");
        let executor = ProcessExecutor::default();
        let cancellation = CancellationToken::new();
        let process = executor.spawn(&spec, &cancellation).unwrap();

        let failure = process.wait(&cancellation).unwrap_err();

        assert_eq!(failure.error.kind(), ErrorKind::ProcessExit);
        assert_eq!(failure.error.context.child_exit_code, Some(7));
    }

    fn shell(script: &str) -> ProcessSpec {
        ProcessSpec::new("sh", ["-c", script])
            .with_deadline(Instant::now() + Duration::from_secs(10))
    }

    fn wait_for_file(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !path.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        assert!(path.exists(), "fixture did not write {}", path.display());
    }

    fn process_exists(pid: i32) -> bool {
        nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid), None).is_ok()
    }
}
