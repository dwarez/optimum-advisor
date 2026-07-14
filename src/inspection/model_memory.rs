use serde::Deserialize;
use std::{
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    config::ExecutableConfig,
    error::{Error, ErrorKind, ExecutionStage, Result},
    results::report::{ModelMemoryEstimate, ModelMemoryOutcome, WarningRecord},
    runtime::{
        cancel::CancellationToken,
        json::parse_unique_json,
        process::{ProcessCapture, ProcessExecutor, ProcessSpec, DIAGNOSTIC_TAIL_BYTES},
    },
};

pub(crate) fn resolve_hf_mem_command(configured: Option<&Path>) -> Option<PathBuf> {
    let environment = std::env::var_os("OPTIMUM_ADVISOR_HF_MEM");
    let search_path = std::env::var_os("PATH");
    resolve_command_with(configured, environment.as_deref(), search_path.as_deref())
}

pub(crate) fn estimate_model_memory(
    config: &ExecutableConfig,
    command: Option<PathBuf>,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<ModelMemoryOutcome> {
    if !config.model_memory.enabled {
        return Ok(ModelMemoryOutcome {
            estimate: None,
            warning: None,
        });
    }
    let result = match command {
        Some(command) => estimate_with_command(config, command, executor, cancellation),
        None => Err(Error::new(
            ErrorKind::ProcessSpawn,
            Some(ExecutionStage::Preflight),
            "hf-mem is unavailable; configure model_memory.command, \
             OPTIMUM_ADVISOR_HF_MEM, or install hf-mem on PATH",
        )),
    };
    match result {
        Ok(estimate) => Ok(ModelMemoryOutcome {
            estimate: Some(estimate),
            warning: None,
        }),
        Err(error) if !config.model_memory.required => Ok(ModelMemoryOutcome {
            estimate: None,
            warning: Some(WarningRecord {
                kind: error.kind(),
                stage: error.stage().unwrap_or(ExecutionStage::ParameterInspection),
                message: error.to_string(),
            }),
        }),
        Err(error) => Err(error),
    }
}

fn estimate_with_command(
    config: &ExecutableConfig,
    command: PathBuf,
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<ModelMemoryEstimate> {
    let kv_cache_dtype = config
        .serve_args
        .iter()
        .rev()
        .find(|argument| argument.name == "kv-cache-dtype")
        .and_then(|argument| argument.value.clone())
        .unwrap_or_else(|| "auto".to_string());
    let args = [
        OsString::from("--model-id"),
        OsString::from(&config.model),
        OsString::from("--experimental"),
        OsString::from("--json-output"),
        OsString::from("--max-model-len"),
        OsString::from(config.runtime.max_model_len.to_string()),
        OsString::from("--batch-size"),
        OsString::from(config.benchmark.max_concurrency.to_string()),
        OsString::from("--kv-cache-dtype"),
        OsString::from(&kv_cache_dtype),
    ];
    let mut spec = ProcessSpec::new(command.as_os_str().to_os_string(), args)
        .with_stage(ExecutionStage::ParameterInspection)
        .with_timeout(Duration::from_secs(config.model_memory.timeout_secs))
        .with_safe_display(format!("{} <model memory arguments>", command.display()));
    spec.max_stdout_bytes = DIAGNOSTIC_TAIL_BYTES as u64;
    spec.max_stderr_bytes = DIAGNOSTIC_TAIL_BYTES as u64;
    let outcome = executor
        .execute(&spec, cancellation)
        .map_err(|failure| failure.error)?;
    let ProcessCapture::Artifacts(capture) = outcome.capture else {
        return Err(memory_error(
            "model-memory command unexpectedly used secret capture",
        ));
    };
    if capture.stdout.truncated {
        return Err(Error::new(
            ErrorKind::OutputTruncated,
            Some(ExecutionStage::ParameterInspection),
            "model-memory JSON output exceeded 64 KiB",
        ));
    }
    if capture.stdout.observed_bytes != capture.stdout.tail.len() as u64 {
        return Err(memory_error(
            "model-memory output contained ANSI, invalid UTF-8, or redacted bytes",
        ));
    }
    let document: HfMemDocument = parse_unique_json(&capture.stdout.tail).map_err(|source| {
        memory_error("model-memory command returned invalid JSON").with_source(source)
    })?;
    let components = document
        .memory
        .checked_add(document.kv_cache)
        .ok_or_else(|| memory_error("model-memory byte counts overflowed u64"))?;
    if document.total_memory == 0 || components > document.total_memory {
        return Err(memory_error(
            "model-memory total must cover weights and KV-cache bytes",
        ));
    }
    Ok(ModelMemoryEstimate {
        source: command.display().to_string(),
        model: config.model.clone(),
        max_model_len: config.runtime.max_model_len,
        batch_size: config.benchmark.max_concurrency,
        kv_cache_dtype,
        weights_bytes: Some(document.memory),
        kv_cache_bytes: Some(document.kv_cache),
        activation_bytes: None,
        total_bytes: Some(document.total_memory),
    })
}

#[derive(Deserialize)]
struct HfMemDocument {
    memory: u64,
    kv_cache: u64,
    total_memory: u64,
}

fn resolve_command_with(
    configured: Option<&Path>,
    environment: Option<&OsStr>,
    search_path: Option<&OsStr>,
) -> Option<PathBuf> {
    if let Some(configured) = configured.filter(|path| !path.as_os_str().is_empty()) {
        return Some(configured.to_path_buf());
    }
    if let Some(environment) = environment.filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(environment));
    }
    search_path
        .into_iter()
        .flat_map(std::env::split_paths)
        .map(|directory| directory.join("hf-mem"))
        .find(|candidate| executable_file(candidate))
}

fn executable_file(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if !metadata.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

fn memory_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::ParameterInspection,
        Some(ExecutionStage::ParameterInspection),
        message,
    )
}

#[cfg(all(test, unix))]
mod tests {
    use std::{ffi::OsStr, fs, os::unix::fs::PermissionsExt};

    use tempfile::tempdir;

    use super::*;
    use crate::{
        config::ConfigInput,
        domain::{engine::Engine, run::ResolvedImage},
        runtime::{cancel::CancellationToken, process::ProcessExecutor},
    };

    fn config(required: bool) -> crate::config::ExecutableConfig {
        let mut input = ConfigInput::minimal(Engine::Vllm, "repo/model");
        input.model_memory.enabled = Some(true);
        input.model_memory.required = Some(required);
        input.runtime.max_model_len = Some(16_384);
        input.benchmark.max_concurrency = Some(4);
        input.normalize().unwrap().into_executable(ResolvedImage {
            requested: "image:tag".into(),
            immutable:
                "image@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    .into(),
            local_only: false,
        })
    }

    #[test]
    fn executes_hf_mem_with_typed_bounded_json() {
        let directory = tempdir().unwrap();
        let command = directory.path().join("hf-mem");
        let log = directory.path().join("args.log");
        fs::write(
            &command,
            format!(
                "#!/bin/sh\nprintf '%s\\n' \"$*\" > '{}'\nprintf '%s' '{{\"memory\":10,\"kv_cache\":20,\"total_memory\":30}}'\n",
                log.display()
            ),
        )
        .unwrap();
        make_executable(&command);

        let outcome = estimate_model_memory(
            &config(true),
            Some(command),
            &ProcessExecutor::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        let estimate = outcome.estimate.unwrap();

        assert_eq!(estimate.weights_bytes, Some(10));
        assert_eq!(estimate.kv_cache_bytes, Some(20));
        assert_eq!(estimate.total_bytes, Some(30));
        let args = fs::read_to_string(log).unwrap();
        assert!(args.contains("--max-model-len 16384"));
        assert!(args.contains("--batch-size 4"));
        assert!(!args.contains("uvx"));
    }

    #[test]
    fn required_invalid_json_fails_but_optional_returns_a_typed_warning() {
        let directory = tempdir().unwrap();
        let command = directory.path().join("hf-mem");
        fs::write(
            &command,
            "#!/bin/sh\nprintf '%s' '{\"memory\":10,\"memory\":11,\"kv_cache\":20,\"total_memory\":30}'\n",
        )
        .unwrap();
        make_executable(&command);

        assert!(estimate_model_memory(
            &config(true),
            Some(command.clone()),
            &ProcessExecutor::default(),
            &CancellationToken::new(),
        )
        .is_err());
        let optional = estimate_model_memory(
            &config(false),
            Some(command),
            &ProcessExecutor::default(),
            &CancellationToken::new(),
        )
        .unwrap();
        assert!(optional.estimate.is_none());
        assert!(optional.warning.is_some());
    }

    #[test]
    fn command_resolution_prefers_explicit_then_environment_then_path() {
        let directory = tempdir().unwrap();
        let installed = directory.path().join("hf-mem");
        fs::write(&installed, "#!/bin/sh\n").unwrap();
        make_executable(&installed);

        assert_eq!(
            resolve_command_with(
                Some(std::path::Path::new("/configured/hf-mem")),
                Some(OsStr::new("/environment/hf-mem")),
                Some(directory.path().as_os_str()),
            ),
            Some(std::path::PathBuf::from("/configured/hf-mem"))
        );
        assert_eq!(
            resolve_command_with(
                None,
                Some(OsStr::new("/environment/hf-mem")),
                Some(directory.path().as_os_str()),
            ),
            Some(std::path::PathBuf::from("/environment/hf-mem"))
        );
        assert_eq!(
            resolve_command_with(None, None, Some(directory.path().as_os_str())),
            Some(installed)
        );
    }

    fn make_executable(path: &std::path::Path) {
        let mut permissions = fs::metadata(path).unwrap().permissions();
        permissions.set_mode(0o700);
        fs::set_permissions(path, permissions).unwrap();
    }
}
