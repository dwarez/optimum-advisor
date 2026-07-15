use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    domain::{candidate::DynamicArg, engine::Engine, run::ExecutionBackend},
    engines::parameter_introspection_script,
    error::{Error, ErrorKind, ExecutionStage, Result},
    runtime::json::parse_unique_json,
    runtime::{
        atomic::{atomic_write, create_private_dir},
        cancel::CancellationToken,
        process::{
            ProcessCapture, ProcessExecutor, ProcessFailure, ProcessSpec,
            DEFAULT_INSPECTION_TIMEOUT, DIAGNOSTIC_TAIL_BYTES,
        },
    },
};

const PARAMETER_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ParameterMode {
    Flag,
    Value,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ParameterSchema {
    pub parameters: BTreeMap<String, ParameterMode>,
}

impl ParameterSchema {
    pub(crate) fn validate(&self, arguments: &[DynamicArg]) -> Result<()> {
        for argument in arguments {
            let Some(mode) = self.parameters.get(&argument.name) else {
                return Err(parameter_error(format!(
                    "unknown serving parameter: {}",
                    argument.name
                )));
            };
            match (mode, argument.value.as_ref()) {
                (ParameterMode::Flag, None) | (ParameterMode::Value, Some(_)) => {}
                (ParameterMode::Flag, Some(_)) => {
                    return Err(parameter_error(format!(
                        "flag serving parameter {} must not have a value",
                        argument.name
                    )));
                }
                (ParameterMode::Value, None) => {
                    return Err(parameter_error(format!(
                        "serving parameter {} requires a value",
                        argument.name
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CachedParameterSchema {
    schema_version: u32,
    engine: Engine,
    image_identity: String,
    schema: ParameterSchema,
}

pub(crate) fn load_parameter_schema(
    engine: Engine,
    image_identity: &str,
    cache_root: &Path,
    refresh: bool,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
    backend: ExecutionBackend,
) -> Result<ParameterSchema> {
    let path = schema_cache_path(cache_root, engine, image_identity);
    if !refresh {
        if let Some(schema) = cached_parameter_schema(engine, image_identity, cache_root)? {
            return Ok(schema);
        }
    }
    let schema = inspect_parameter_schema(engine, image_identity, executor, cancellation, backend)?;
    create_private_dir(path.parent().expect("schema cache path has a parent"))?;
    let cached = CachedParameterSchema {
        schema_version: PARAMETER_SCHEMA_VERSION,
        engine,
        image_identity: image_identity.to_string(),
        schema: schema.clone(),
    };
    let bytes = serde_json::to_vec_pretty(&cached).map_err(|source| {
        Error::new(
            ErrorKind::ParameterInspection,
            Some(ExecutionStage::ParameterInspection),
            "failed to serialize parameter schema cache",
        )
        .with_path(&path)
        .with_source(source)
    })?;
    atomic_write(&path, 0o600, &bytes)?;
    Ok(schema)
}

pub(crate) fn cached_parameter_schema(
    engine: Engine,
    image_identity: &str,
    cache_root: &Path,
) -> Result<Option<ParameterSchema>> {
    let path = schema_cache_path(cache_root, engine, image_identity);
    match fs::metadata(&path) {
        Ok(metadata) if metadata.is_file() => {
            read_cached_schema(&path, engine, image_identity).map(Some)
        }
        Ok(_) => Err(parameter_error("parameter schema cache is not a file").with_path(path)),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(parameter_error("failed to inspect parameter schema cache")
            .with_path(path)
            .with_source(source)),
    }
}

pub(crate) fn parameter_inspection_spec(
    engine: Engine,
    image_identity: &str,
    executor: &ProcessExecutor,
    backend: ExecutionBackend,
) -> ProcessSpec {
    let script = parameter_introspection_script(engine);
    let (program, args, safe_display) = match backend {
        ExecutionBackend::Docker => (
            executor.docker_program().to_os_string(),
            vec![
                OsString::from("run"),
                OsString::from("--rm"),
                OsString::from("--gpus"),
                OsString::from("all"),
                OsString::from("--entrypoint"),
                OsString::from("python3"),
                OsString::from(image_identity),
                OsString::from("-c"),
                OsString::from(script),
            ],
            format!(
                "docker run --rm --gpus all --entrypoint python3 {image_identity} -c <parameter introspection>"
            ),
        ),
        ExecutionBackend::InContainer => (
            OsString::from("python3"),
            vec![OsString::from("-c"), OsString::from(script)],
            "python3 -c <parameter introspection>".to_string(),
        ),
    };
    let mut spec = ProcessSpec::new(program, args)
        .with_stage(ExecutionStage::ParameterInspection)
        .with_timeout(DEFAULT_INSPECTION_TIMEOUT)
        .with_safe_display(safe_display);
    spec.max_stdout_bytes = DIAGNOSTIC_TAIL_BYTES as u64;
    spec.max_stderr_bytes = DIAGNOSTIC_TAIL_BYTES as u64;
    spec
}

pub(crate) fn inspect_parameter_schema(
    engine: Engine,
    image_identity: &str,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
    backend: ExecutionBackend,
) -> Result<ParameterSchema> {
    let spec = parameter_inspection_spec(engine, image_identity, executor, backend);
    let outcome = executor
        .execute(&spec, cancellation)
        .map_err(map_inspection_failure)?;
    let ProcessCapture::Artifacts(capture) = outcome.capture else {
        return Err(parameter_error(
            "parameter introspection unexpectedly used secret capture",
        ));
    };
    if capture.stdout.truncated {
        return Err(Error::new(
            ErrorKind::OutputTruncated,
            Some(ExecutionStage::ParameterInspection),
            "parameter introspection output exceeded 64 KiB",
        ));
    }
    if capture.stdout.observed_bytes != capture.stdout.tail.len() as u64 {
        return Err(parameter_error(
            "parameter introspection output contained ANSI, invalid UTF-8, or redacted bytes",
        ));
    }
    parse_schema(&capture.stdout.tail)
}

fn parse_schema(text: &str) -> Result<ParameterSchema> {
    if text.as_bytes().contains(&0x1b) {
        return Err(parameter_error(
            "parameter schema output must not contain ANSI escapes",
        ));
    }
    let mut parameters = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 2 {
            return Err(parameter_error(format!(
                "malformed parameter schema line {}",
                index + 1
            )));
        }
        let cli_name = fields[0];
        let name = canonical_parameter_name(cli_name).ok_or_else(|| {
            parameter_error(format!(
                "invalid parameter name on schema line {}",
                index + 1
            ))
        })?;
        let mode = match fields[1] {
            "flag" => ParameterMode::Flag,
            "value" => ParameterMode::Value,
            _ => {
                return Err(parameter_error(format!(
                    "invalid parameter mode on schema line {}",
                    index + 1
                )));
            }
        };
        if parameters.insert(name.clone(), mode).is_some() {
            return Err(parameter_error(format!(
                "duplicate parameter {name:?} in inspected schema"
            )));
        }
    }
    if parameters.is_empty() {
        return Err(parameter_error(
            "parameter introspection returned an empty schema",
        ));
    }
    Ok(ParameterSchema { parameters })
}

fn canonical_parameter_name(name: &str) -> Option<String> {
    let name = name.strip_prefix("--")?;
    let canonical = name.to_ascii_lowercase().replace('_', "-");
    valid_parameter_name(&canonical).then_some(canonical)
}

fn valid_parameter_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
}

fn schema_cache_path(cache_root: &Path, engine: Engine, identity: &str) -> PathBuf {
    let digest = Sha256::digest(identity.as_bytes());
    cache_root.join("schemas").join(format!(
        "{engine}-{}.json",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn read_cached_schema(
    path: &Path,
    engine: Engine,
    image_identity: &str,
) -> Result<ParameterSchema> {
    let bytes = fs::read(path).map_err(|source| {
        parameter_error("failed to read parameter schema cache")
            .with_path(path)
            .with_source(source)
    })?;
    let text = std::str::from_utf8(&bytes).map_err(|source| {
        parameter_error("parameter schema cache is not UTF-8")
            .with_path(path)
            .with_source(source)
    })?;
    let cached: CachedParameterSchema = parse_unique_json(text).map_err(|error| {
        parameter_error("invalid parameter schema cache")
            .with_path(path)
            .with_source(error)
    })?;
    if cached.schema_version != PARAMETER_SCHEMA_VERSION
        || cached.engine != engine
        || cached.image_identity != image_identity
    {
        return Err(
            parameter_error("parameter schema cache identity or version does not match")
                .with_path(path),
        );
    }
    if cached.schema.parameters.is_empty()
        || cached
            .schema
            .parameters
            .keys()
            .any(|name| !valid_parameter_name(name))
    {
        return Err(parameter_error("parameter schema cache is invalid").with_path(path));
    }
    Ok(cached.schema)
}

fn map_inspection_failure(failure: ProcessFailure) -> Error {
    if matches!(
        failure.error.kind(),
        ErrorKind::Interrupted | ErrorKind::Timeout
    ) {
        return failure.error.with_operation("inspect serving parameters");
    }
    let message = failure.diagnostic_tail().map_or_else(
        || "parameter introspection command failed".to_string(),
        |diagnostic| format!("parameter introspection command failed:\n{diagnostic}"),
    );
    parameter_error(message)
        .with_operation("inspect serving parameters")
        .with_source(failure.error)
}

fn parameter_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::ParameterInspection,
        Some(ExecutionStage::ParameterInspection),
        message,
    )
}

#[cfg(all(test, unix))]
mod tests {
    use std::{fs, os::unix::fs::PermissionsExt};

    use tempfile::tempdir;

    use super::*;
    use crate::{
        domain::{candidate::DynamicArg, engine::Engine},
        error::ErrorKind,
        runtime::{cancel::CancellationToken, process::ProcessExecutor},
    };

    #[test]
    fn parses_strict_parameter_modes_and_validates_arguments() {
        let schema = parse_schema("--model\tvalue\n--trust-remote-code\tflag\n").unwrap();

        schema
            .validate(&[
                DynamicArg::value("model", "repo/model"),
                DynamicArg::flag("trust-remote-code"),
            ])
            .unwrap();
        assert!(schema
            .validate(&[DynamicArg {
                name: "unknown".into(),
                value: None,
            }])
            .is_err());
        assert!(schema
            .validate(&[DynamicArg {
                name: "model".into(),
                value: None,
            }])
            .is_err());
        assert!(schema
            .validate(&[DynamicArg {
                name: "trust-remote-code".into(),
                value: Some("yes".into()),
            }])
            .is_err());
    }

    #[test]
    fn rejects_ansi_malformed_and_duplicate_schema_lines() {
        assert!(parse_schema("\u{1b}[31m--model\tvalue\n").is_err());
        assert!(parse_schema("--model value\n").is_err());
        assert!(parse_schema("--model\toptional\n").is_err());
        assert!(parse_schema("--model\tvalue\n--model\tvalue\n").is_err());
        assert!(parse_schema("garbage\tflag\n").is_err());
    }

    #[test]
    fn in_container_inspection_spec_runs_python_without_docker() {
        let executor = ProcessExecutor::default();
        let spec = parameter_inspection_spec(
            Engine::Vllm,
            "repo/server@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &executor,
            ExecutionBackend::InContainer,
        );
        let args: Vec<String> = spec
            .args
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect();
        assert_eq!(spec.program.to_string_lossy(), "python3");
        assert_eq!(args.first().map(String::as_str), Some("-c"));
        assert!(!args.iter().any(|arg| arg == "run" || arg == "--gpus"));
        assert_eq!(spec.safe_display, "python3 -c <parameter introspection>");
    }

    #[test]
    fn exposes_host_gpus_to_the_introspection_container() {
        let directory = tempdir().unwrap();
        let docker = directory.path().join("docker");
        fs::write(
            &docker,
            "#!/bin/sh\nprevious=\nfor argument do\n  if [ \"$previous\" = --gpus ] && [ \"$argument\" = all ]; then\n    printf '%s\\t%s\\n' --model value\n    exit 0\n  fi\n  previous=$argument\ndone\nprintf '%s\\n' 'RuntimeError: Failed to infer device type' >&2\nexit 1\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&docker).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&docker, permissions).unwrap();
        let executor = ProcessExecutor::default().with_docker_program(&docker);

        let schema = inspect_parameter_schema(
            Engine::Vllm,
            "repo/model@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &executor,
            &CancellationToken::new(),
            ExecutionBackend::Docker,
        )
        .unwrap();

        assert!(schema.parameters.contains_key("model"));
    }

    #[test]
    fn rejects_truncated_introspection_output() {
        let directory = tempdir().unwrap();
        let docker = directory.path().join("docker");
        fs::write(
            &docker,
            "#!/bin/sh\ni=0\nwhile [ $i -lt 7000 ]; do printf '%s\\t%s\\n' --option value; i=$((i + 1)); done\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&docker).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&docker, permissions).unwrap();
        let executor = ProcessExecutor::default().with_docker_program(&docker);

        let error = inspect_parameter_schema(
            Engine::Vllm,
            "repo/model@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &executor,
            &CancellationToken::new(),
            ExecutionBackend::Docker,
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::OutputTruncated);
    }

    #[test]
    fn reports_introspection_stderr_when_the_command_fails() {
        let directory = tempdir().unwrap();
        let docker = directory.path().join("docker");
        fs::write(
            &docker,
            "#!/bin/sh\nprintf '%s\\n' 'ImportError: incompatible vLLM CLI parser' >&2\nexit 1\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&docker).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&docker, permissions).unwrap();
        let executor = ProcessExecutor::default().with_docker_program(&docker);

        let error = inspect_parameter_schema(
            Engine::Vllm,
            "repo/model@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            &executor,
            &CancellationToken::new(),
            ExecutionBackend::Docker,
        )
        .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ParameterInspection);
        assert!(
            error
                .to_string()
                .contains("ImportError: incompatible vLLM CLI parser"),
            "{error}"
        );
    }

    #[test]
    fn caches_typed_schema_by_immutable_image_identity() {
        let directory = tempdir().unwrap();
        let docker = directory.path().join("docker");
        let log = directory.path().join("calls.log");
        fs::write(
            &docker,
            format!(
                "#!/bin/sh\nprintf 'call\\n' >> '{}'\nprintf '%s\\t%s\\n' --model value\n",
                log.display()
            ),
        )
        .unwrap();
        let mut permissions = fs::metadata(&docker).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(&docker, permissions).unwrap();
        let executor = ProcessExecutor::default().with_docker_program(&docker);
        let identity =
            "docker.io/library/repo@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

        let first = load_parameter_schema(
            Engine::Vllm,
            identity,
            directory.path(),
            true,
            &executor,
            &CancellationToken::new(),
            ExecutionBackend::Docker,
        )
        .unwrap();
        let second = load_parameter_schema(
            Engine::Vllm,
            identity,
            directory.path(),
            false,
            &executor,
            &CancellationToken::new(),
            ExecutionBackend::Docker,
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(fs::read_to_string(log).unwrap().lines().count(), 1);
        assert_eq!(
            fs::read_dir(directory.path().join("schemas"))
                .unwrap()
                .count(),
            1
        );
    }
}
