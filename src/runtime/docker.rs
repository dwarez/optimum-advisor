use std::{collections::BTreeMap, ffi::OsString, time::Duration};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    domain::run::{PullPolicy, ResolvedImage},
    error::{Error, ErrorKind, ExecutionStage, Result},
    runtime::{
        cancel::CancellationToken,
        process::{
            ArtifactCapture, CapturePolicy, ProcessCapture, ProcessExecutor, ProcessFailure,
            ProcessSpec, DEFAULT_INSPECTION_TIMEOUT, DIAGNOSTIC_TAIL_BYTES,
        },
    },
};

pub(crate) const OWNED_LABEL_KEY: &str = "optimum-advisor";
pub(crate) const OWNED_LABEL_VALUE: &str = "true";
pub(crate) const RUN_ID_LABEL: &str = "optimum-advisor.run-id";
pub(crate) const ROLE_LABEL: &str = "optimum-advisor.role";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct OwnedContainer {
    pub name: String,
    pub run_id: String,
    pub role: String,
    pub labels: BTreeMap<String, String>,
}

impl OwnedContainer {
    pub(crate) fn new(
        name: impl Into<String>,
        run_id: impl Into<String>,
        role: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let run_id = run_id.into();
        let role = role.into();
        let labels = BTreeMap::from([
            (OWNED_LABEL_KEY.to_string(), OWNED_LABEL_VALUE.to_string()),
            (RUN_ID_LABEL.to_string(), run_id.clone()),
            (ROLE_LABEL.to_string(), role.clone()),
        ]);
        Self {
            name,
            run_id,
            role,
            labels,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct DockerImageIdentity {
    pub requested: String,
    pub image_id: String,
    pub repository_digest: Option<String>,
    pub immutable: String,
    pub local_only: bool,
}

impl DockerImageIdentity {
    pub(crate) fn resolved(&self) -> ResolvedImage {
        ResolvedImage {
            requested: self.requested.clone(),
            immutable: self.immutable.clone(),
            local_only: self.local_only,
        }
    }
}

pub(crate) fn resolve_image(
    requested: &str,
    pull_policy: PullPolicy,
    allow_local_image: bool,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<DockerImageIdentity> {
    if requested.trim().is_empty() {
        return Err(image_error("image reference must not be empty"));
    }
    match pull_policy {
        PullPolicy::Always => pull_image(requested, executor, cancellation)?,
        PullPolicy::Missing => {
            if !image_exists(requested, executor, cancellation)? {
                pull_image(requested, executor, cancellation)?;
            }
        }
        PullPolicy::Never => {
            if !image_exists(requested, executor, cancellation)? {
                return Err(image_error(format!(
                    "Docker image {requested:?} is unavailable under pull policy {pull_policy}"
                )));
            }
        }
    }
    let inspected = inspect_image(requested, executor, cancellation)?.ok_or_else(|| {
        image_error(format!(
            "Docker image {requested:?} disappeared before identity inspection"
        ))
    })?;

    validate_digest("Docker image ID", &inspected.id)?;
    let requested_repository = repository_from_reference(requested)?;
    let normalized_requested = normalize_repository(requested_repository)?;
    let mut matching = inspected
        .repo_digests
        .iter()
        .filter_map(|digest| canonical_matching_digest(digest, &normalized_requested))
        .collect::<Vec<_>>();
    if let Some(explicit) = explicit_digest(requested, &normalized_requested) {
        matching.push(explicit);
    }
    matching.sort();
    matching.dedup();
    if matching.len() > 1 {
        return Err(image_error(format!(
            "Docker image {requested:?} has multiple matching repository digests"
        )));
    }
    if let Some(repository_digest) = matching.pop() {
        return Ok(DockerImageIdentity {
            requested: requested.to_string(),
            image_id: inspected.id,
            immutable: repository_digest.clone(),
            repository_digest: Some(repository_digest),
            local_only: false,
        });
    }
    if !allow_local_image {
        return Err(image_error(format!(
            "Docker image {requested:?} has no matching repository digest; \
             enable explicit local-image use to accept its image ID"
        )));
    }
    Ok(DockerImageIdentity {
        requested: requested.to_string(),
        immutable: inspected.id.clone(),
        image_id: inspected.id,
        repository_digest: None,
        local_only: true,
    })
}

/// Build an image identity for in-container execution without contacting Docker.
///
/// The surrounding container already provides the engine image, so its
/// reference is treated as the immutable identity directly; no digest is
/// resolved and no pull is attempted.
pub(crate) fn in_container_image_identity(requested: &str) -> Result<DockerImageIdentity> {
    if requested.trim().is_empty() {
        return Err(image_error("image reference must not be empty"));
    }
    Ok(DockerImageIdentity {
        requested: requested.to_string(),
        image_id: requested.to_string(),
        repository_digest: None,
        immutable: requested.to_string(),
        local_only: true,
    })
}

pub(crate) fn cleanup_owned_containers(
    run_id: Option<&str>,
    dry_run: bool,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<Vec<String>> {
    if run_id.is_some_and(|value| value.trim().is_empty() || value.contains(['\r', '\n'])) {
        return Err(Error::validation(
            "cleanup run ID must be nonempty and contain no newlines",
        ));
    }

    let mut arguments = vec![
        OsString::from("ps"),
        OsString::from("-a"),
        OsString::from("--filter"),
        OsString::from(format!("label={OWNED_LABEL_KEY}={OWNED_LABEL_VALUE}")),
    ];
    if let Some(run_id) = run_id {
        arguments.extend([
            OsString::from("--filter"),
            OsString::from(format!("label={RUN_ID_LABEL}={run_id}")),
        ]);
    }
    arguments.extend([OsString::from("--format"), OsString::from("{{.Names}}")]);
    let listing = ProcessSpec::new(executor.docker_program().to_os_string(), arguments)
        .with_stage(ExecutionStage::Persistence)
        .with_timeout(Duration::from_secs(30))
        .with_capture(CapturePolicy::Secret)
        .with_safe_display("docker ps -a <owned-label-filters> --format <names>");
    let output = executor
        .execute(&listing, cancellation)
        .map_err(|failure| map_cleanup_failure("list owned Docker containers", failure))?;
    let ProcessCapture::Secret(output) = output.capture else {
        unreachable!("cleanup listing uses bounded in-memory capture");
    };
    let containers = output
        .expose()
        .lines()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if dry_run {
        return Ok(containers);
    }

    for container in &containers {
        let removal = ProcessSpec::new(
            executor.docker_program().to_os_string(),
            [
                OsString::from("rm"),
                OsString::from("-f"),
                OsString::from(container),
            ],
        )
        .with_stage(ExecutionStage::Persistence)
        .with_timeout(Duration::from_secs(30))
        .with_capture(CapturePolicy::Secret)
        .with_safe_display(format!("docker rm -f {container}"));
        executor
            .execute(&removal, cancellation)
            .map_err(|failure| map_cleanup_failure("remove owned Docker container", failure))?;
    }
    Ok(containers)
}

pub(crate) fn immutable_reference(reference: &str) -> Result<Option<String>> {
    if reference.starts_with("sha256:") {
        validate_digest("image ID", reference)?;
        return Ok(Some(reference.to_string()));
    }
    let Some((repository, digest)) = reference.rsplit_once("@sha256:") else {
        return Ok(None);
    };
    if !valid_hash(digest) {
        return Err(image_error(
            "repository digest must contain 64 lowercase hexadecimal digits",
        ));
    }
    Ok(Some(format!(
        "{}@sha256:{digest}",
        normalize_repository(repository)?
    )))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InspectOutput {
    id: String,
    #[serde(default)]
    repo_digests: Vec<String>,
}

fn image_exists(
    requested: &str,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<bool> {
    let spec = ProcessSpec::new(
        executor.docker_program().to_os_string(),
        [
            OsString::from("image"),
            OsString::from("inspect"),
            OsString::from(requested),
        ],
    )
    .with_timeout(DEFAULT_INSPECTION_TIMEOUT)
    .with_safe_display(format!("docker image inspect {requested}"));
    match executor.execute(&spec, cancellation) {
        Ok(_) => Ok(true),
        Err(failure)
            if failure.error.kind() == ErrorKind::ProcessExit
                && failure.error.context.child_exit_code == Some(1) =>
        {
            Ok(false)
        }
        Err(failure) => Err(map_docker_failure("check local Docker image", failure)),
    }
}

fn inspect_image(
    requested: &str,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<Option<InspectOutput>> {
    let template = r#"{"id":{{json .Id}},"repo_digests":{{json .RepoDigests}}}"#;
    let args = [
        OsString::from("image"),
        OsString::from("inspect"),
        OsString::from("--format"),
        OsString::from(template),
        OsString::from(requested),
    ];
    let mut spec = ProcessSpec::new(executor.docker_program().to_os_string(), args)
        .with_timeout(DEFAULT_INSPECTION_TIMEOUT)
        .with_safe_display(format!("docker image inspect {requested}"));
    spec.max_stdout_bytes = DIAGNOSTIC_TAIL_BYTES as u64;
    spec.max_stderr_bytes = DIAGNOSTIC_TAIL_BYTES as u64;
    match executor.execute(&spec, cancellation) {
        Ok(outcome) => {
            let capture = artifact_capture(outcome.capture)?;
            if capture.stdout.truncated {
                return Err(Error::new(
                    ErrorKind::OutputTruncated,
                    Some(ExecutionStage::ImageResolution),
                    "Docker image identity response was truncated",
                ));
            }
            serde_json::from_str(&capture.stdout.tail)
                .map(Some)
                .map_err(|source| {
                    Error::new(
                        ErrorKind::Docker,
                        Some(ExecutionStage::ImageResolution),
                        "Docker image inspect returned malformed identity JSON",
                    )
                    .with_source(source)
                })
        }
        Err(failure)
            if failure.error.kind() == ErrorKind::ProcessExit
                && failure.error.context.child_exit_code == Some(1) =>
        {
            Ok(None)
        }
        Err(failure) => Err(map_docker_failure("inspect Docker image", failure)),
    }
}

fn pull_image(
    requested: &str,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<()> {
    let spec = ProcessSpec::new(
        executor.docker_program().to_os_string(),
        [OsString::from("pull"), OsString::from(requested)],
    )
    .with_timeout(DEFAULT_INSPECTION_TIMEOUT)
    .with_safe_display(format!("docker pull {requested}"));
    executor
        .execute(&spec, cancellation)
        .map(|_| ())
        .map_err(|failure| map_docker_failure("pull Docker image", failure))
}

fn artifact_capture(capture: ProcessCapture) -> Result<ArtifactCapture> {
    match capture {
        ProcessCapture::Artifacts(capture) => Ok(capture),
        ProcessCapture::Secret(_) => Err(image_error(
            "Docker command unexpectedly used secret capture",
        )),
    }
}

fn map_docker_failure(operation: &'static str, failure: ProcessFailure) -> Error {
    let message = failure.diagnostic_tail().map_or_else(
        || format!("failed to {operation}"),
        |diagnostic| format!("failed to {operation}:\n{diagnostic}"),
    );
    Error::new(
        ErrorKind::Docker,
        Some(ExecutionStage::ImageResolution),
        message,
    )
    .with_operation(operation)
    .with_source(failure.error)
}

fn map_cleanup_failure(operation: &'static str, failure: ProcessFailure) -> Error {
    Error::new(
        ErrorKind::Docker,
        Some(ExecutionStage::Persistence),
        format!("failed to {operation}"),
    )
    .with_operation(operation)
    .with_source(failure.error)
}

fn repository_from_reference(reference: &str) -> Result<&str> {
    let without_digest = reference
        .split_once('@')
        .map_or(reference, |(repo, _)| repo);
    let slash = without_digest.rfind('/');
    let colon = without_digest.rfind(':');
    let repository = match (slash, colon) {
        (Some(slash), Some(colon)) if colon > slash => &without_digest[..colon],
        (None, Some(colon)) => &without_digest[..colon],
        _ => without_digest,
    };
    if repository.is_empty() {
        Err(image_error(format!(
            "invalid Docker image reference: {reference:?}"
        )))
    } else {
        Ok(repository)
    }
}

fn normalize_repository(repository: &str) -> Result<String> {
    let mut parts = repository
        .split('/')
        .filter(|part| !part.is_empty())
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return Err(image_error("Docker repository must not be empty"));
    }
    let has_registry = parts[0].contains('.') || parts[0].contains(':') || parts[0] == "localhost";
    if !has_registry {
        parts.insert(0, "docker.io".to_string());
    } else if matches!(
        parts[0].as_str(),
        "index.docker.io" | "registry-1.docker.io"
    ) {
        parts[0] = "docker.io".to_string();
    }
    if parts[0] == "docker.io" && parts.len() == 2 {
        parts.insert(1, "library".to_string());
    }
    if parts.iter().any(|part| {
        part.is_empty()
            || !part.chars().all(|character| {
                character.is_ascii_lowercase()
                    || character.is_ascii_digit()
                    || matches!(character, '.' | '_' | '-')
            })
    }) {
        return Err(image_error(format!(
            "invalid Docker repository: {repository:?}"
        )));
    }
    Ok(parts.join("/"))
}

fn canonical_matching_digest(value: &str, expected_repository: &str) -> Option<String> {
    let (repository, digest) = value.rsplit_once("@sha256:")?;
    if normalize_repository(repository).ok()?.as_str() != expected_repository || !valid_hash(digest)
    {
        return None;
    }
    Some(format!("{expected_repository}@sha256:{digest}"))
}

fn explicit_digest(reference: &str, expected_repository: &str) -> Option<String> {
    let (repository, digest) = reference.rsplit_once("@sha256:")?;
    if normalize_repository(repository).ok()?.as_str() == expected_repository && valid_hash(digest)
    {
        Some(format!("{expected_repository}@sha256:{digest}"))
    } else {
        None
    }
}

fn validate_digest(label: &str, value: &str) -> Result<()> {
    let Some(hash) = value.strip_prefix("sha256:") else {
        return Err(image_error(format!("{label} is not a sha256 digest")));
    };
    if !valid_hash(hash) {
        return Err(image_error(format!(
            "{label} must contain 64 lowercase hexadecimal digits"
        )));
    }
    Ok(())
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn image_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::Docker,
        Some(ExecutionStage::ImageResolution),
        message,
    )
}

#[cfg(all(test, unix))]
mod tests {
    use std::{fs, os::unix::fs::PermissionsExt};

    use tempfile::tempdir;

    use super::*;
    use crate::runtime::{cancel::CancellationToken, process::ProcessExecutor};

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn selects_only_the_matching_repository_digest() {
        let fake = FakeDocker::new();
        let executor = ProcessExecutor::default().with_docker_program(&fake.program);

        let identity = resolve_image(
            "repo:tag",
            PullPolicy::Missing,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .unwrap();

        assert_eq!(
            identity.immutable,
            format!("docker.io/library/repo@sha256:{DIGEST}")
        );
        assert!(!identity.local_only);
        assert_eq!(identity.image_id, format!("sha256:{DIGEST}"));
    }

    #[test]
    fn requires_explicit_permission_for_local_only_images() {
        let fake = FakeDocker::new();
        let executor = ProcessExecutor::default().with_docker_program(&fake.program);

        assert!(resolve_image(
            "local/image:dev",
            PullPolicy::Missing,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .is_err());
        let identity = resolve_image(
            "local/image:dev",
            PullPolicy::Missing,
            true,
            &executor,
            &CancellationToken::new(),
        )
        .unwrap();

        assert!(identity.local_only);
        assert_eq!(identity.immutable, format!("sha256:{DIGEST}"));
    }

    #[test]
    fn docker_hub_aliases_resolve_to_the_same_immutable_identity() {
        let fake = FakeDocker::new();
        let executor = ProcessExecutor::default().with_docker_program(&fake.program);
        let first = resolve_image(
            "repo:tag",
            PullPolicy::Missing,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .unwrap();
        let second = resolve_image(
            "docker.io/library/repo:tag",
            PullPolicy::Missing,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .unwrap();

        assert_eq!(first.immutable, second.immutable);
    }

    #[test]
    fn applies_missing_always_and_never_pull_policies() {
        let missing = FakeDocker::new();
        let executor = ProcessExecutor::default().with_docker_program(&missing.program);
        resolve_image(
            "missing/repo:tag",
            PullPolicy::Missing,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .unwrap();
        let log = missing.log();
        assert_eq!(
            log.iter().filter(|line| line.starts_with("pull ")).count(),
            1
        );
        assert_eq!(
            log.iter()
                .filter(|line| line.starts_with("image inspect "))
                .count(),
            2
        );

        let always = FakeDocker::new();
        let executor = ProcessExecutor::default().with_docker_program(&always.program);
        resolve_image(
            "repo:tag",
            PullPolicy::Always,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .unwrap();
        let log = always.log();
        assert!(log[0].starts_with("pull "));
        assert_eq!(
            log.iter()
                .filter(|line| line.starts_with("image inspect "))
                .count(),
            1
        );

        let never = FakeDocker::new();
        let executor = ProcessExecutor::default().with_docker_program(&never.program);
        assert!(resolve_image(
            "missing/repo:tag",
            PullPolicy::Never,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .is_err());
        assert!(!never.log().iter().any(|line| line.starts_with("pull ")));
    }

    #[test]
    fn reports_docker_pull_diagnostics() {
        let directory = tempdir().unwrap();
        let docker = directory.path().join("docker");
        fs::write(
            &docker,
            "#!/bin/sh\nif [ \"$1\" = image ] && [ \"$2\" = inspect ]; then\n  exit 1\nfi\nif [ \"$1\" = pull ]; then\n  printf '%s\\n' 'no space left on device' >&2\n  exit 1\nfi\nexit 2\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&docker).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&docker, permissions).unwrap();
        let executor = ProcessExecutor::default().with_docker_program(&docker);

        let error = resolve_image(
            "lmsysorg/sglang:latest",
            PullPolicy::Missing,
            false,
            &executor,
            &CancellationToken::new(),
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::Docker);
        assert!(
            error.to_string().contains("no space left on device"),
            "{error}"
        );
    }

    #[test]
    fn cleanup_lists_and_removes_only_owned_containers() {
        let fake = FakeDocker::new();
        let executor = ProcessExecutor::default().with_docker_program(&fake.program);

        let listed =
            cleanup_owned_containers(Some("run-7"), true, &executor, &CancellationToken::new())
                .unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed, ["owned-server"]);
        assert!(!fake.log().iter().any(|line| line.starts_with("rm ")));

        cleanup_owned_containers(Some("run-7"), false, &executor, &CancellationToken::new())
            .unwrap();
        let log = fake.log();
        assert!(log.iter().all(|line| !line.contains("unowned")));
        assert_eq!(
            log.iter()
                .filter(|line| line.as_str() == "rm -f owned-server")
                .count(),
            1
        );
        assert!(log
            .iter()
            .filter(|line| line.starts_with("ps "))
            .all(|line| {
                line.contains("label=optimum-advisor=true")
                    && line.contains("label=optimum-advisor.run-id=run-7")
            }));
    }

    struct FakeDocker {
        _directory: tempfile::TempDir,
        program: std::path::PathBuf,
        log: std::path::PathBuf,
    }

    impl FakeDocker {
        fn new() -> Self {
            let directory = tempdir().unwrap();
            let program = directory.path().join("docker");
            let log = directory.path().join("calls.log");
            let state = directory.path().join("pulled");
            let script = format!(
                r#"#!/bin/sh
printf '%s\n' "$*" >> '{log}'
if [ "$1" = "pull" ]; then
  : > '{state}'
  exit 0
fi
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  ref="${{5:-$3}}"
  case "$ref" in
    missing/*)
      [ -f '{state}' ] || exit 1
      repository="docker.io/missing/repo"
      ;;
    local/*)
      printf '%s\n' '{{"id":"sha256:{digest}","repo_digests":[]}}'
      exit 0
      ;;
    *)
      repository="docker.io/library/repo"
      ;;
  esac
  printf '{{"id":"sha256:{digest}","repo_digests":["unrelated.example/x@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","%s@sha256:{digest}"]}}\n' "$repository"
  exit 0
fi
if [ "$1" = "ps" ]; then
  printf '%s\n' 'owned-server'
  exit 0
fi
if [ "$1" = "rm" ]; then
  exit 0
fi
exit 2
"#,
                log = log.display(),
                state = state.display(),
                digest = DIGEST,
            );
            fs::write(&program, script).unwrap();
            let mut permissions = fs::metadata(&program).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&program, permissions).unwrap();
            Self {
                _directory: directory,
                program,
                log,
            }
        }

        fn log(&self) -> Vec<String> {
            fs::read_to_string(&self.log)
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect()
        }
    }
}
