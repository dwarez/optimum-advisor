use std::{fmt, str::FromStr};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PullPolicy {
    #[default]
    Missing,
    Always,
    Never,
}

impl FromStr for PullPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "missing" => Ok(Self::Missing),
            "always" => Ok(Self::Always),
            "never" => Ok(Self::Never),
            _ => Err(format!("unknown image pull policy: {value}")),
        }
    }
}

impl fmt::Display for PullPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => formatter.write_str("missing"),
            Self::Always => formatter.write_str("always"),
            Self::Never => formatter.write_str("never"),
        }
    }
}

/// Selects how engine server and benchmark commands are launched.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionBackend {
    /// Wrap each engine invocation in `docker run --gpus ... <image> ...`.
    ///
    /// Requires a local Docker daemon with the NVIDIA container runtime.
    #[default]
    Docker,
    /// Run the engine binary directly in the current process namespace.
    ///
    /// Assumes the surrounding container already provides the engine image
    /// (for example a Hugging Face Job whose image is `vllm/vllm-openai`),
    /// so no Docker daemon, image resolution, or container cleanup applies.
    InContainer,
}

/// Selects where an evaluation runs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ExecutionTarget {
    /// Execute on the local host (Docker backend).
    #[default]
    Local,
    /// Submit the evaluation to Hugging Face Jobs (in-container backend).
    HfJobs,
}

impl FromStr for ExecutionTarget {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "local" => Ok(Self::Local),
            "hf-jobs" => Ok(Self::HfJobs),
            _ => Err(format!("unknown execution target: {value}")),
        }
    }
}

impl fmt::Display for ExecutionTarget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local => formatter.write_str("local"),
            Self::HfJobs => formatter.write_str("hf-jobs"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
pub(crate) struct ResolvedImage {
    pub requested: String,
    pub immutable: String,
    pub local_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
pub(crate) struct GpuRecord {
    pub index: u32,
    pub uuid: String,
    pub name: String,
    pub compute_capability: Option<String>,
    pub memory_total_mib: u64,
    pub memory_free_mib: u64,
    pub memory_used_mib: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct HardwareProfile {
    pub source: String,
    pub cuda_visible_devices: Option<String>,
    pub all_gpus: Vec<GpuRecord>,
    pub selected_gpus: Vec<GpuRecord>,
    pub warnings: Vec<String>,
}
