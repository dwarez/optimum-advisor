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
