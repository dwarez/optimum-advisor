use std::{
    fs::{self, File},
    io::Read,
    path::{Component, Path, PathBuf},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, ErrorKind, ExecutionStage, Result};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct ArtifactManifest {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub truncated: bool,
}

impl ArtifactManifest {
    pub(crate) fn from_path(run_dir: &Path, path: &Path, truncated: bool) -> Result<Self> {
        let relative = validate_relative(path)?;
        let canonical_run = fs::canonicalize(run_dir)
            .map_err(|source| artifact_error("canonicalize run directory", run_dir, source))?;
        let installed = canonical_run.join(&relative);
        let canonical_installed = fs::canonicalize(&installed).map_err(|source| {
            artifact_error("canonicalize installed artifact", &installed, source)
        })?;
        if !canonical_installed.starts_with(&canonical_run) {
            return Err(Error::validation(format!(
                "artifact path escapes run directory: {}",
                relative.display()
            ))
            .with_artifact_path(relative));
        }
        let (bytes, sha256) = hash_file(&canonical_installed)?;
        Ok(Self {
            path: relative,
            bytes,
            sha256,
            truncated,
        })
    }
}

fn validate_relative(path: &Path) -> Result<PathBuf> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(Error::validation(format!(
            "artifact path must be a nonempty run-relative path: {}",
            path.display()
        ))
        .with_artifact_path(path));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(Error::validation(format!(
            "artifact path must not contain traversal components: {}",
            path.display()
        ))
        .with_artifact_path(path));
    }
    Ok(path.to_path_buf())
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let mut file = File::open(path)
        .map_err(|source| artifact_error("open installed artifact", path, source))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| artifact_error("read installed artifact", path, source))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes = bytes.checked_add(read as u64).ok_or_else(|| {
            Error::new(
                ErrorKind::Io,
                Some(ExecutionStage::Persistence),
                "artifact byte count overflowed u64",
            )
            .with_artifact_path(path)
        })?;
    }
    Ok((bytes, hex(hasher.finalize().as_slice())))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn artifact_error(operation: &'static str, path: &Path, source: std::io::Error) -> Error {
    Error::new(
        ErrorKind::Io,
        Some(ExecutionStage::Persistence),
        format!("failed to {operation}: {}", path.display()),
    )
    .with_operation(operation)
    .with_artifact_path(path)
    .with_source(source)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn hashes_existing_run_relative_artifact() {
        let dir = tempdir().unwrap();
        fs::create_dir(dir.path().join("logs")).unwrap();
        fs::write(dir.path().join("logs/server.log"), b"hello").unwrap();

        let manifest =
            ArtifactManifest::from_path(dir.path(), Path::new("logs/server.log"), false).unwrap();

        assert_eq!(manifest.path, PathBuf::from("logs/server.log"));
        assert_eq!(manifest.bytes, 5);
        assert_eq!(
            manifest.sha256,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn rejects_paths_outside_the_run_directory() {
        let dir = tempdir().unwrap();

        assert!(
            ArtifactManifest::from_path(dir.path(), &dir.path().join("../escape.log"), false,)
                .is_err()
        );
    }
}
