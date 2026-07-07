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
    Tps,
    TotalTps,
    InputTps,
    PeakTps,
    ReqS,
    Goodput,
    Ttft,
    P90Ttft,
    P95Ttft,
    P99Ttft,
    Tpot,
    P90Tpot,
    P95Tpot,
    P99Tpot,
    Itl,
    P90Itl,
    P95Itl,
    P99Itl,
    E2e,
    P90E2e,
    P95E2e,
    P99E2e,
}

impl Metric {
    pub fn parse(value: &str) -> Result<Self> {
        match value.replace('-', "_").as_str() {
            "tps" | "output_tps" | "output_throughput" | "throughput" => Ok(Self::Tps),
            "total_tps" | "total_throughput" => Ok(Self::TotalTps),
            "input_tps" | "input_throughput" => Ok(Self::InputTps),
            "peak_tps" | "peak_output_tps" => Ok(Self::PeakTps),
            "req_s" | "rps" | "request_throughput" => Ok(Self::ReqS),
            "goodput" | "request_goodput" => Ok(Self::Goodput),
            "ttft" | "mean_ttft" => Ok(Self::Ttft),
            "p90_ttft" => Ok(Self::P90Ttft),
            "p95_ttft" => Ok(Self::P95Ttft),
            "p99_ttft" => Ok(Self::P99Ttft),
            "tpot" | "mean_tpot" => Ok(Self::Tpot),
            "p90_tpot" => Ok(Self::P90Tpot),
            "p95_tpot" => Ok(Self::P95Tpot),
            "p99_tpot" => Ok(Self::P99Tpot),
            "itl" | "mean_itl" => Ok(Self::Itl),
            "p90_itl" => Ok(Self::P90Itl),
            "p95_itl" => Ok(Self::P95Itl),
            "p99_itl" => Ok(Self::P99Itl),
            "e2e" | "e2el" | "latency" | "mean_e2e" => Ok(Self::E2e),
            "p90_e2e" | "p90_e2el" => Ok(Self::P90E2e),
            "p95_e2e" | "p95_e2el" => Ok(Self::P95E2e),
            "p99_e2e" | "p99_e2el" => Ok(Self::P99E2e),
            _ => Err(format!("unknown metric: {value}")),
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tps => write!(f, "tps"),
            Self::TotalTps => write!(f, "total_tps"),
            Self::InputTps => write!(f, "input_tps"),
            Self::PeakTps => write!(f, "peak_tps"),
            Self::ReqS => write!(f, "req_s"),
            Self::Goodput => write!(f, "goodput"),
            Self::Ttft => write!(f, "ttft"),
            Self::P90Ttft => write!(f, "p90_ttft"),
            Self::P95Ttft => write!(f, "p95_ttft"),
            Self::P99Ttft => write!(f, "p99_ttft"),
            Self::Tpot => write!(f, "tpot"),
            Self::P90Tpot => write!(f, "p90_tpot"),
            Self::P95Tpot => write!(f, "p95_tpot"),
            Self::P99Tpot => write!(f, "p99_tpot"),
            Self::Itl => write!(f, "itl"),
            Self::P90Itl => write!(f, "p90_itl"),
            Self::P95Itl => write!(f, "p95_itl"),
            Self::P99Itl => write!(f, "p99_itl"),
            Self::E2e => write!(f, "e2e"),
            Self::P90E2e => write!(f, "p90_e2e"),
            Self::P95E2e => write!(f, "p95_e2e"),
            Self::P99E2e => write!(f, "p99_e2e"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Plan,
    Params,
    Hardware,
    Serve,
    Bench,
    Sweep,
    Advise,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metric_aliases() {
        assert_eq!(Metric::parse("throughput").unwrap(), Metric::Tps);
        assert_eq!(Metric::parse("req-s").unwrap(), Metric::ReqS);
        assert_eq!(Metric::parse("p99_ttft").unwrap(), Metric::P99Ttft);
        assert_eq!(Metric::parse("p95-tpot").unwrap(), Metric::P95Tpot);
        assert_eq!(Metric::parse("e2el").unwrap(), Metric::E2e);
        assert_eq!(Metric::parse("p99_e2e").unwrap(), Metric::P99E2e);
    }
}
