use std::{cmp::Ordering, collections::BTreeMap};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    domain::{candidate::normalize_zero, engine::Metric},
    error::{Error, ErrorKind, ExecutionStage, Result},
    inspection::correctness::CorrectnessStatus,
};

pub(crate) const MAX_UNRECOGNIZED_METRICS: usize = 32;
const MAX_DIAGNOSTIC_VALUE_CHARS: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub(crate) struct BenchmarkMetrics {
    pub successful_requests: Option<f64>,
    pub failed_requests: Option<f64>,
    pub max_request_concurrency: Option<f64>,
    pub request_rate_configured: Option<f64>,
    pub benchmark_duration_s: Option<f64>,
    pub total_input_tokens: Option<f64>,
    pub total_input_text_tokens: Option<f64>,
    pub total_input_vision_tokens: Option<f64>,
    pub total_generated_tokens: Option<f64>,
    pub total_generated_tokens_retokenized: Option<f64>,
    pub request_throughput: Option<f64>,
    pub request_goodput: Option<f64>,
    pub input_token_throughput: Option<f64>,
    pub output_token_throughput: Option<f64>,
    pub output_token_throughput_retokenized: Option<f64>,
    pub peak_output_token_throughput: Option<f64>,
    pub peak_concurrent_requests: Option<f64>,
    pub total_token_throughput: Option<f64>,
    pub total_token_throughput_retokenized: Option<f64>,
    pub concurrency: Option<f64>,
    pub accept_length: Option<f64>,
    pub rtfx: Option<f64>,
    pub mean_ttft_ms: Option<f64>,
    pub median_ttft_ms: Option<f64>,
    pub std_ttft_ms: Option<f64>,
    pub p90_ttft_ms: Option<f64>,
    pub p95_ttft_ms: Option<f64>,
    pub p99_ttft_ms: Option<f64>,
    pub mean_tpot_ms: Option<f64>,
    pub median_tpot_ms: Option<f64>,
    pub std_tpot_ms: Option<f64>,
    pub p90_tpot_ms: Option<f64>,
    pub p95_tpot_ms: Option<f64>,
    pub p99_tpot_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub median_itl_ms: Option<f64>,
    pub std_itl_ms: Option<f64>,
    pub p90_itl_ms: Option<f64>,
    pub p95_itl_ms: Option<f64>,
    pub p99_itl_ms: Option<f64>,
    pub max_itl_ms: Option<f64>,
    pub mean_e2e_ms: Option<f64>,
    pub median_e2e_ms: Option<f64>,
    pub std_e2e_ms: Option<f64>,
    pub p90_e2e_ms: Option<f64>,
    pub p95_e2e_ms: Option<f64>,
    pub p99_e2e_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unrecognized: BTreeMap<String, String>,
}

impl BenchmarkMetrics {
    pub(crate) fn parse(text: &str, selected: Metric) -> Result<Self> {
        let mut metrics = Self::default();
        let mut recognized: BTreeMap<String, Option<f64>> = BTreeMap::new();
        for line in text.lines() {
            let Some((raw_label, raw_value)) = line.split_once(':') else {
                continue;
            };
            let label = raw_label.trim();
            let parsed = first_finite_number(raw_value);
            if metrics.set(label, parsed) {
                // Engine benchmarks may echo a configured value both before the
                // run and inside the result block (e.g. vLLM prints "Maximum
                // request concurrency" twice). A repeat carrying the same value
                // is self-consistent; only conflicting repeats are ambiguous.
                if let Some(previous) = recognized.insert(label.to_string(), parsed) {
                    if previous != parsed {
                        return Err(metric_error(format!(
                            "benchmark output contains conflicting values for metric {label:?}"
                        )));
                    }
                }
            } else if metrics.unrecognized.len() < MAX_UNRECOGNIZED_METRICS {
                metrics.unrecognized.insert(
                    bounded_chars(label, MAX_DIAGNOSTIC_VALUE_CHARS),
                    bounded_chars(raw_value.trim(), MAX_DIAGNOSTIC_VALUE_CHARS),
                );
            }
        }

        let selected_value = metrics.value_for(selected).ok_or_else(|| {
            metric_error(format!(
                "benchmark output is missing finite selected metric {selected}"
            ))
        })?;
        if !selected_value.is_finite() {
            return Err(metric_error(format!(
                "selected metric {selected} must be finite"
            )));
        }
        if let Some(failed) = metrics.failed_requests {
            if failed < 0.0 {
                return Err(metric_error("failed requests must not be negative"));
            }
            if failed > 0.0 {
                return Err(metric_error(format!(
                    "benchmark reported {failed} failed requests"
                )));
            }
        }
        Ok(metrics)
    }

    pub(crate) fn value_for(&self, metric: Metric) -> Option<f64> {
        match metric {
            Metric::Tps => self.output_token_throughput,
            Metric::TotalTps => self.total_token_throughput,
            Metric::InputTps => self.input_token_throughput,
            Metric::PeakTps => self.peak_output_token_throughput,
            Metric::ReqS => self.request_throughput,
            Metric::Goodput => self.request_goodput,
            Metric::Ttft => self.mean_ttft_ms,
            Metric::P90Ttft => self.p90_ttft_ms,
            Metric::P95Ttft => self.p95_ttft_ms,
            Metric::P99Ttft => self.p99_ttft_ms,
            Metric::Tpot => self.mean_tpot_ms,
            Metric::P90Tpot => self.p90_tpot_ms,
            Metric::P95Tpot => self.p95_tpot_ms,
            Metric::P99Tpot => self.p99_tpot_ms,
            Metric::Itl => self.mean_itl_ms,
            Metric::P90Itl => self.p90_itl_ms,
            Metric::P95Itl => self.p95_itl_ms,
            Metric::P99Itl => self.p99_itl_ms,
            Metric::E2e => self.mean_e2e_ms,
            Metric::P90E2e => self.p90_e2e_ms,
            Metric::P95E2e => self.p95_e2e_ms,
            Metric::P99E2e => self.p99_e2e_ms,
        }
    }

    fn set(&mut self, label: &str, value: Option<f64>) -> bool {
        let value = value.map(normalize_zero);
        match label {
            "Successful requests" => self.successful_requests = value,
            "Failed requests" => self.failed_requests = value,
            "Maximum request concurrency" | "Max request concurrency" => {
                self.max_request_concurrency = value
            }
            "Request rate configured (RPS)" | "Traffic request rate" => {
                self.request_rate_configured = value
            }
            "Benchmark duration (s)" => self.benchmark_duration_s = value,
            "Total input tokens" => self.total_input_tokens = value,
            "Total input text tokens" => self.total_input_text_tokens = value,
            "Total input vision tokens" => self.total_input_vision_tokens = value,
            "Total generated tokens" => self.total_generated_tokens = value,
            "Total generated tokens (retokenized)" => {
                self.total_generated_tokens_retokenized = value
            }
            "Request throughput (req/s)" => self.request_throughput = value,
            "Request goodput (req/s)" => self.request_goodput = value,
            "Input token throughput (tok/s)" => self.input_token_throughput = value,
            "Output token throughput (tok/s)" => self.output_token_throughput = value,
            "Output token throughput (retokenized) (tok/s)" => {
                self.output_token_throughput_retokenized = value
            }
            "Peak output token throughput (tok/s)" => self.peak_output_token_throughput = value,
            "Peak concurrent requests" => self.peak_concurrent_requests = value,
            "Total token throughput (tok/s)" => self.total_token_throughput = value,
            "Total token throughput (retokenized) (tok/s)" => {
                self.total_token_throughput_retokenized = value
            }
            "Concurrency" => self.concurrency = value,
            "Accept length" => self.accept_length = value,
            "RTFx (Inverse Real-Time Factor)" => self.rtfx = value,
            "Mean TTFT (ms)" => self.mean_ttft_ms = value,
            "Median TTFT (ms)" => self.median_ttft_ms = value,
            "Std TTFT (ms)" => self.std_ttft_ms = value,
            "P90 TTFT (ms)" => self.p90_ttft_ms = value,
            "P95 TTFT (ms)" => self.p95_ttft_ms = value,
            "P99 TTFT (ms)" => self.p99_ttft_ms = value,
            "Mean TPOT (ms)" => self.mean_tpot_ms = value,
            "Median TPOT (ms)" => self.median_tpot_ms = value,
            "Std TPOT (ms)" => self.std_tpot_ms = value,
            "P90 TPOT (ms)" => self.p90_tpot_ms = value,
            "P95 TPOT (ms)" => self.p95_tpot_ms = value,
            "P99 TPOT (ms)" => self.p99_tpot_ms = value,
            "Mean ITL (ms)" => self.mean_itl_ms = value,
            "Median ITL (ms)" => self.median_itl_ms = value,
            "Std ITL (ms)" => self.std_itl_ms = value,
            "P90 ITL (ms)" => self.p90_itl_ms = value,
            "P95 ITL (ms)" => self.p95_itl_ms = value,
            "P99 ITL (ms)" => self.p99_itl_ms = value,
            "Max ITL (ms)" => self.max_itl_ms = value,
            "Mean E2E Latency (ms)" | "Mean E2EL (ms)" => self.mean_e2e_ms = value,
            "Median E2E Latency (ms)" | "Median E2EL (ms)" => self.median_e2e_ms = value,
            "Std E2E Latency (ms)" | "Std E2EL (ms)" => self.std_e2e_ms = value,
            "P90 E2E Latency (ms)" | "P90 E2EL (ms)" => self.p90_e2e_ms = value,
            "P95 E2E Latency (ms)" | "P95 E2EL (ms)" => self.p95_e2e_ms = value,
            "P99 E2E Latency (ms)" | "P99 E2EL (ms)" => self.p99_e2e_ms = value,
            _ => return false,
        }
        true
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RankableObservation {
    pub index: usize,
    pub correctness: Option<CorrectnessStatus>,
    pub value: f64,
}

pub(crate) fn compare_observations(
    metric: Metric,
    left_correctness: Option<CorrectnessStatus>,
    left_value: Option<f64>,
    right_correctness: Option<CorrectnessStatus>,
    right_value: Option<f64>,
) -> Ordering {
    let correctness = correctness_rank(left_correctness).cmp(&correctness_rank(right_correctness));
    if correctness != Ordering::Equal {
        return correctness;
    }

    match (
        left_value.filter(|value| value.is_finite()),
        right_value.filter(|value| value.is_finite()),
    ) {
        (Some(left), Some(right)) if metric.lower_is_better() => right.total_cmp(&left),
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => Ordering::Equal,
    }
}

pub(crate) fn select_best(metric: Metric, observations: &[RankableObservation]) -> Option<usize> {
    let mut best: Option<RankableObservation> = None;
    for observation in observations
        .iter()
        .copied()
        .filter(|item| item.value.is_finite())
    {
        let replace = best.is_none_or(|current| {
            compare_observations(
                metric,
                observation.correctness,
                Some(observation.value),
                current.correctness,
                Some(current.value),
            ) == Ordering::Greater
        });
        if replace {
            best = Some(observation);
        }
    }
    best.map(|observation| observation.index)
}

fn correctness_rank(status: Option<CorrectnessStatus>) -> u8 {
    match status {
        Some(CorrectnessStatus::Passed) => 2,
        None => 1,
        Some(CorrectnessStatus::Failed) => 0,
    }
}

fn first_finite_number(text: &str) -> Option<f64> {
    text.split_whitespace()
        .find_map(|word| {
            word.trim_matches(|character: char| matches!(character, ',' | ';'))
                .parse::<f64>()
                .ok()
        })
        .filter(|value| value.is_finite())
}

fn bounded_chars(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn metric_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::Benchmark,
        Some(ExecutionStage::ResultCollection),
        message,
    )
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;

    #[test]
    fn requires_a_finite_selected_metric() {
        assert!(BenchmarkMetrics::parse("Successful requests: 1", Metric::Tps).is_err());
        assert!(
            BenchmarkMetrics::parse("Output token throughput (tok/s): NaN", Metric::Tps,).is_err()
        );
    }

    #[test]
    fn rejects_positive_failed_requests() {
        let error = BenchmarkMetrics::parse(
            "Failed requests: 1\nOutput token throughput (tok/s): 10",
            Metric::Tps,
        )
        .unwrap_err();

        assert!(error.to_string().contains("failed requests"));
    }

    #[test]
    fn tolerates_self_consistent_duplicate_metrics() {
        // vLLM `bench serve` echoes the configured concurrency before the run
        // and again inside the result block; both carry the same value.
        let metrics = BenchmarkMetrics::parse(
            "Maximum request concurrency: 1\n\
             ============ Serving Benchmark Result ============\n\
             Successful requests: 4\n\
             Maximum request concurrency: 1\n\
             Output token throughput (tok/s): 10\n",
            Metric::Tps,
        )
        .unwrap();

        assert_eq!(metrics.max_request_concurrency, Some(1.0));
        assert_eq!(metrics.output_token_throughput, Some(10.0));
    }

    #[test]
    fn rejects_conflicting_duplicate_metrics() {
        let error = BenchmarkMetrics::parse(
            "Maximum request concurrency: 1\n\
             Maximum request concurrency: 2\n\
             Output token throughput (tok/s): 10\n",
            Metric::Tps,
        )
        .unwrap_err();

        assert!(error.to_string().contains("conflicting values"));
    }

    #[test]
    fn normalizes_negative_zero_and_bounds_unknown_labels() {
        let mut text = "Output token throughput (tok/s): -0.0\n".to_string();
        for index in 0..100 {
            text.push_str(&format!("unknown-{index}: {index}\n"));
        }

        let metrics = BenchmarkMetrics::parse(&text, Metric::Tps).unwrap();

        assert_eq!(metrics.output_token_throughput, Some(0.0));
        assert_eq!(
            metrics.output_token_throughput.unwrap().to_bits(),
            0.0f64.to_bits()
        );
        assert!(metrics.unrecognized.len() <= MAX_UNRECOGNIZED_METRICS);
    }

    #[test]
    fn missing_correctness_ranks_below_passed() {
        assert_eq!(
            compare_observations(
                Metric::Tps,
                Some(CorrectnessStatus::Passed),
                Some(1.0),
                None,
                Some(2.0),
            ),
            Ordering::Greater
        );
    }

    #[test]
    fn ties_keep_the_earlier_trial() {
        let observations = [
            RankableObservation {
                index: 0,
                correctness: None,
                value: 1.0,
            },
            RankableObservation {
                index: 1,
                correctness: None,
                value: 1.0,
            },
        ];

        assert_eq!(select_best(Metric::Tps, &observations), Some(0));
    }

    #[test]
    fn ranking_respects_metric_direction() {
        let observations = [
            RankableObservation {
                index: 0,
                correctness: None,
                value: 2.0,
            },
            RankableObservation {
                index: 1,
                correctness: None,
                value: 1.0,
            },
        ];

        assert_eq!(select_best(Metric::Ttft, &observations), Some(1));
        assert_eq!(select_best(Metric::Tps, &observations), Some(0));
    }
}
