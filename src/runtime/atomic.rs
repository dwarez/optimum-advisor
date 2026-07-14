use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use crate::error::{Error, ErrorKind, ExecutionStage, Result};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
const MAX_TEMP_ATTEMPTS: usize = 128;

pub(crate) fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path)
        .map_err(|source| io_error("create private directory", path, source))?;
    #[cfg(unix)]
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|source| io_error("set private directory permissions", path, source))?;
    Ok(())
}

pub(crate) fn atomic_write(path: &Path, mode: u32, bytes: &[u8]) -> Result<()> {
    atomic_write_inner(path, mode, bytes, |from, to| fs::rename(from, to))
}

pub(crate) struct AtomicWriter {
    destination: PathBuf,
    parent: PathBuf,
    temporary: PathBuf,
    file: Option<File>,
    guard: TemporaryGuard,
}

impl AtomicWriter {
    pub(crate) fn create(path: &Path, mode: u32) -> Result<Self> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let file_name = path.file_name().ok_or_else(|| {
            Error::new(
                ErrorKind::Io,
                Some(ExecutionStage::Persistence),
                "atomic write destination has no file name",
            )
            .with_path(path)
        })?;
        let (temporary, file) = create_temporary(parent, file_name, mode)?;
        Ok(Self {
            destination: path.to_path_buf(),
            parent: parent.to_path_buf(),
            guard: TemporaryGuard::new(temporary.clone()),
            temporary,
            file: Some(file),
        })
    }

    pub(crate) fn commit(mut self) -> Result<()> {
        let mut file = self.file.take().expect("atomic writer file is present");
        file.flush()
            .and_then(|()| file.sync_all())
            .map_err(|source| io_error("flush and sync temporary file", &self.temporary, source))?;
        drop(file);
        fs::rename(&self.temporary, &self.destination).map_err(|source| {
            io_error("atomically replace destination", &self.destination, source)
        })?;
        self.guard.disarm();
        sync_parent(&self.parent)
    }
}

impl Write for AtomicWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.file
            .as_mut()
            .expect("atomic writer file is present")
            .write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file
            .as_mut()
            .expect("atomic writer file is present")
            .flush()
    }
}

fn atomic_write_inner(
    path: &Path,
    mode: u32,
    bytes: &[u8],
    rename: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file_name = path.file_name().ok_or_else(|| {
        Error::new(
            ErrorKind::Io,
            Some(ExecutionStage::Persistence),
            "atomic write destination has no file name",
        )
        .with_path(path)
    })?;
    let (temporary_path, mut file) = create_temporary(parent, file_name, mode)?;
    let mut guard = TemporaryGuard::new(temporary_path.clone());

    file.write_all(bytes)
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_all())
        .map_err(|source| io_error("write and sync temporary file", &temporary_path, source))?;
    drop(file);

    rename(&temporary_path, path)
        .map_err(|source| io_error("atomically replace destination", path, source))?;
    guard.disarm();
    sync_parent(parent)?;
    Ok(())
}

fn create_temporary(
    parent: &Path,
    file_name: &std::ffi::OsStr,
    mode: u32,
) -> Result<(PathBuf, File)> {
    for _ in 0..MAX_TEMP_ATTEMPTS {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            ".{}.tmp-{}-{id}",
            file_name.to_string_lossy(),
            std::process::id()
        );
        let path = parent.join(name);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(mode);
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(io_error("create temporary file", &path, source)),
        }
    }

    Err(Error::new(
        ErrorKind::Io,
        Some(ExecutionStage::Persistence),
        format!("could not allocate a unique temporary file after {MAX_TEMP_ATTEMPTS} attempts"),
    )
    .with_path(parent))
}

#[cfg(unix)]
fn sync_parent(parent: &Path) -> Result<()> {
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error("sync parent directory", parent, source))
}

#[cfg(not(unix))]
fn sync_parent(_parent: &Path) -> Result<()> {
    Ok(())
}

fn io_error(operation: &'static str, path: &Path, source: io::Error) -> Error {
    Error::new(
        ErrorKind::Io,
        Some(ExecutionStage::Persistence),
        format!("failed to {operation}: {}", path.display()),
    )
    .with_operation(operation)
    .with_path(path)
    .with_source(source)
}

struct TemporaryGuard {
    path: PathBuf,
    armed: bool,
}

impl TemporaryGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
fn atomic_write_with_rename(
    path: &Path,
    mode: u32,
    bytes: &[u8],
    rename: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> Result<()> {
    atomic_write_inner(path, mode, bytes, rename)
}

#[cfg(test)]
mod tests {
    use std::{fs, io};

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn writes_private_file_atomically() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("report.json");

        atomic_write(&path, 0o600, b"new").unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"new");
        #[cfg(unix)]
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn rename_failure_preserves_existing_destination() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("report.json");
        fs::write(&path, b"old").unwrap();

        let result = atomic_write_with_rename(&path, 0o600, b"new", |_, _| {
            Err(io::Error::other("injected rename failure"))
        });

        assert!(result.is_err());
        assert_eq!(fs::read(&path).unwrap(), b"old");
    }
}
