use std::{error::Error as StdError, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error as ThisError;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStage {
    Preflight,
    ImageResolution,
    Validation,
    ParameterInspection,
    Server,
    Correctness,
    Benchmark,
    ResultCollection,
    Persistence,
    Leaderboard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Usage,
    Configuration,
    Validation,
    Io,
    ProcessSpawn,
    ProcessExit,
    Timeout,
    Interrupted,
    OutputTruncated,
    Docker,
    ParameterInspection,
    Correctness,
    Benchmark,
    HttpTransport,
    HttpProtocol,
    Protocol,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct ErrorContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docker_identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ErrorPayload {
    pub kind: ErrorKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage: Option<ExecutionStage>,
    pub message: String,
    #[serde(flatten)]
    pub context: ErrorContext,
}

#[derive(Debug, ThisError)]
#[error("{message}")]
pub struct Error {
    pub kind: ErrorKind,
    pub stage: Option<ExecutionStage>,
    pub message: String,
    pub context: Box<ErrorContext>,
    #[source]
    pub source: Option<Box<dyn StdError + Send + Sync>>,
}

impl Error {
    pub(crate) fn usage(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Usage, None, message)
    }

    pub(crate) fn configuration(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Configuration, None, message)
    }

    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self::new(
            ErrorKind::Validation,
            Some(ExecutionStage::Validation),
            message,
        )
    }

    pub(crate) fn interrupted(stage: ExecutionStage) -> Self {
        Self::new(ErrorKind::Interrupted, Some(stage), "operation interrupted")
    }

    pub(crate) fn new(
        kind: ErrorKind,
        stage: Option<ExecutionStage>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            stage,
            message: message.into(),
            context: Box::new(ErrorContext::default()),
            source: None,
        }
    }

    pub(crate) fn with_source(mut self, source: impl StdError + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    pub(crate) fn with_operation(mut self, operation: impl Into<String>) -> Self {
        self.context.operation = Some(operation.into());
        self
    }

    pub(crate) fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.context.path = Some(path.into());
        self
    }

    pub(crate) fn with_process(mut self, process: impl Into<String>) -> Self {
        self.context.process = Some(process.into());
        self
    }

    pub(crate) fn with_deadline_ms(mut self, deadline_ms: u64) -> Self {
        self.context.deadline_ms = Some(deadline_ms);
        self
    }

    pub(crate) fn with_stream(mut self, stream: impl Into<String>) -> Self {
        self.context.stream = Some(stream.into());
        self
    }

    pub(crate) fn with_docker_identity(mut self, identity: impl Into<String>) -> Self {
        self.context.docker_identity = Some(identity.into());
        self
    }

    pub(crate) fn with_container(mut self, container: impl Into<String>) -> Self {
        self.context.container = Some(container.into());
        self
    }

    pub(crate) fn with_cache_identity(mut self, identity: impl Into<String>) -> Self {
        self.context.cache_identity = Some(identity.into());
        self
    }

    pub(crate) fn with_http_status(mut self, status: u16) -> Self {
        self.context.http_status = Some(status);
        self
    }

    pub(crate) fn with_child_exit_code(mut self, code: i32) -> Self {
        self.context.child_exit_code = Some(code);
        self
    }

    pub(crate) fn with_report_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.context.report_path = Some(path.into());
        self
    }

    pub(crate) fn with_artifact_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.context.artifact_path = Some(path.into());
        self
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn stage(&self) -> Option<ExecutionStage> {
        self.stage
    }

    pub fn payload(&self) -> ErrorPayload {
        ErrorPayload::from(self)
    }

    pub fn exit_code(&self) -> u8 {
        match self.kind {
            ErrorKind::Usage | ErrorKind::Configuration | ErrorKind::Validation => 2,
            ErrorKind::Interrupted => 130,
            _ => 1,
        }
    }
}

impl From<&Error> for ErrorPayload {
    fn from(error: &Error) -> Self {
        Self {
            kind: error.kind,
            stage: error.stage,
            message: error.message.clone(),
            context: (*error.context).clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_error_categories_to_stable_exit_codes() {
        assert_eq!(Error::usage("bad flag").exit_code(), 2);
        assert_eq!(
            Error::new(
                ErrorKind::ProcessExit,
                Some(ExecutionStage::Benchmark),
                "failed",
            )
            .exit_code(),
            1
        );
        assert_eq!(
            Error::interrupted(ExecutionStage::Benchmark).exit_code(),
            130
        );
    }
}
