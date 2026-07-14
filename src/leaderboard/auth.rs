use std::{
    fmt, fs,
    path::{Path, PathBuf},
    time::Duration,
};

use zeroize::Zeroizing;

use crate::{
    error::{Error, ErrorKind, ExecutionStage, Result},
    runtime::{
        cancel::CancellationToken,
        process::{CapturePolicy, ProcessCapture, ProcessExecutor, ProcessSpec},
    },
};

const CLI_TOKEN_TIMEOUT: Duration = Duration::from_secs(10);
#[derive(Clone)]
pub(crate) struct Secret(Zeroizing<String>);

impl Secret {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        let value = value.trim();
        if value.is_empty() {
            return Err(auth_error("credential must not be empty"));
        }
        if value.contains(['\r', '\n']) {
            return Err(auth_error("credential must be a single line"));
        }
        Ok(Self(Zeroizing::new(value.to_string())))
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted>")
    }
}

pub(crate) fn resolve_hf_token(
    executor: &ProcessExecutor,
    cancellation: &CancellationToken,
) -> Result<Option<Secret>> {
    let environment = [
        std::env::var("HF_TOKEN").ok(),
        std::env::var("HUGGING_FACE_HUB_TOKEN").ok(),
    ];
    if let Some(token) = resolve_token_from(&environment, &hf_token_paths())? {
        return Ok(Some(token));
    }

    for (program, arguments) in [
        ("hf", &["auth", "token"][..]),
        ("huggingface-cli", &["token"][..]),
    ] {
        let spec = ProcessSpec::new(program, arguments)
            .with_stage(ExecutionStage::Preflight)
            .with_timeout(CLI_TOKEN_TIMEOUT)
            .with_capture(CapturePolicy::Secret)
            .with_safe_display(format!("{program} <token>"));
        match executor.execute(&spec, cancellation) {
            Ok(output) => {
                let ProcessCapture::Secret(token) = output.capture else {
                    unreachable!("secret capture always returns secret output");
                };
                return Secret::new(token.expose()).map(Some);
            }
            Err(failure) if failure.error.kind() == ErrorKind::Interrupted => {
                return Err(failure.error);
            }
            Err(_) => {}
        }
    }
    Ok(None)
}

pub(crate) fn resolve_submit_key() -> Result<Option<Secret>> {
    std::env::var("OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT_KEY")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(Secret::new)
        .transpose()
}

fn resolve_token_from(environment: &[Option<String>], paths: &[PathBuf]) -> Result<Option<Secret>> {
    for value in environment.iter().flatten() {
        if !value.trim().is_empty() {
            return Secret::new(value.clone()).map(Some);
        }
    }
    for path in paths {
        let metadata = match fs::metadata(path) {
            Ok(metadata) => metadata,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => continue,
            Err(source) => {
                return Err(auth_error("failed to inspect Hugging Face token file")
                    .with_path(path)
                    .with_source(source));
            }
        };
        if !metadata.is_file() {
            return Err(auth_error("Hugging Face token path is not a file").with_path(path));
        }
        require_private_permissions(path, &metadata)?;
        let value = fs::read_to_string(path).map_err(|source| {
            auth_error("failed to read Hugging Face token file")
                .with_path(path)
                .with_source(source)
        })?;
        return Secret::new(value)
            .map(Some)
            .map_err(|error| error.with_path(path));
    }
    Ok(None)
}

fn hf_token_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(path) = std::env::var_os("HF_TOKEN_PATH").filter(|value| !value.is_empty()) {
        paths.push(PathBuf::from(path));
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|value| !value.is_empty()) {
        paths.push(PathBuf::from(home).join("token"));
    }
    if let Some(cache) = std::env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        paths.push(PathBuf::from(cache).join("huggingface").join("token"));
    }
    if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
        paths.push(PathBuf::from(home).join(".cache/huggingface/token"));
    }
    paths
}

fn require_private_permissions(path: &Path, metadata: &fs::Metadata) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(auth_error(
                "Hugging Face token file must not be readable or writable by group/others",
            )
            .with_path(path));
        }
    }
    Ok(())
}

fn auth_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::Validation,
        Some(ExecutionStage::Leaderboard),
        message,
    )
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn secrets_are_redacted_and_zeroizing() {
        let secret = Secret::new("hf_secret").unwrap();

        assert_eq!(secret.expose(), "hf_secret");
        assert_eq!(format!("{secret:?}"), "<redacted>");
    }

    #[test]
    fn resolves_environment_before_private_token_files() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("token");
        fs::write(&path, "hf_file\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        let token = resolve_token_from(&[Some("hf_environment".into())], &[path])
            .unwrap()
            .unwrap();

        assert_eq!(token.expose(), "hf_environment");
    }

    #[test]
    fn rejects_group_or_world_readable_token_files() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("token");
        fs::write(&path, "hf_file\n").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        assert!(resolve_token_from(&[], std::slice::from_ref(&path)).is_err());

        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(
            resolve_token_from(&[], &[path]).unwrap().unwrap().expose(),
            "hf_file"
        );
    }

    #[test]
    fn missing_optional_credentials_are_not_an_error() {
        assert!(resolve_token_from(&[], &[]).unwrap().is_none());
    }
}
