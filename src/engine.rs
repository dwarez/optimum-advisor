use std::fmt;

use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Vllm,
    Sglang,
}

impl Engine {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "vllm" => Ok(Self::Vllm),
            "sglang" => Ok(Self::Sglang),
            _ => Err(format!("unknown engine: {value}")),
        }
    }

    pub fn default_image(self) -> &'static str {
        match self {
            Self::Vllm => "vllm/vllm-openai:latest",
            Self::Sglang => "lmsysorg/sglang:latest",
        }
    }
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vllm => write!(f, "vllm"),
            Self::Sglang => write!(f, "sglang"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Ttft,
    Tps,
    Itl,
}

impl Metric {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "ttft" => Ok(Self::Ttft),
            "tps" | "throughput" => Ok(Self::Tps),
            "itl" => Ok(Self::Itl),
            _ => Err(format!("unknown metric: {value}")),
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ttft => write!(f, "ttft"),
            Self::Tps => write!(f, "tps"),
            Self::Itl => write!(f, "itl"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Plan,
    Params,
    Serve,
    Run,
    Advise,
}
