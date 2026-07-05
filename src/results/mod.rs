use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::ServingConfig;
use crate::engine::{Engine, Metric};
use crate::serve::EngineArg;
use crate::Result;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BenchmarkMetrics {
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
}

impl BenchmarkMetrics {
    pub fn parse(text: &str) -> Self {
        let mut metrics = Self::default();
        for line in text.lines() {
            if let Some((label, value)) = line.split_once(':') {
                metrics.set(label.trim(), first_number(value));
            }
        }
        metrics
    }

    pub fn value_for(&self, metric: Metric) -> Option<f64> {
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

    fn set(&mut self, label: &str, value: Option<f64>) {
        match label {
            "Successful requests" => self.successful_requests = value,
            "Failed requests" => self.failed_requests = value,
            "Maximum request concurrency" | "Max request concurrency" => {
                self.max_request_concurrency = value;
            }
            "Request rate configured (RPS)" | "Traffic request rate" => {
                self.request_rate_configured = value;
            }
            "Benchmark duration (s)" => self.benchmark_duration_s = value,
            "Total input tokens" => self.total_input_tokens = value,
            "Total input text tokens" => self.total_input_text_tokens = value,
            "Total input vision tokens" => self.total_input_vision_tokens = value,
            "Total generated tokens" => self.total_generated_tokens = value,
            "Total generated tokens (retokenized)" => {
                self.total_generated_tokens_retokenized = value;
            }
            "Request throughput (req/s)" => self.request_throughput = value,
            "Request goodput (req/s)" => self.request_goodput = value,
            "Input token throughput (tok/s)" => self.input_token_throughput = value,
            "Output token throughput (tok/s)" => self.output_token_throughput = value,
            "Output token throughput (retokenized) (tok/s)" => {
                self.output_token_throughput_retokenized = value;
            }
            "Peak output token throughput (tok/s)" => self.peak_output_token_throughput = value,
            "Peak concurrent requests" => self.peak_concurrent_requests = value,
            "Total token throughput (tok/s)" => self.total_token_throughput = value,
            "Total token throughput (retokenized) (tok/s)" => {
                self.total_token_throughput_retokenized = value;
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
            _ => {}
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrialResult {
    pub config: ServingConfig,
    pub winning_metric: Metric,
    pub metrics: BenchmarkMetrics,
    pub benchmark_stdout: String,
    pub benchmark_stderr: String,
}

impl TrialResult {
    pub fn new(
        config: ServingConfig,
        winning_metric: Metric,
        benchmark_stdout: String,
        benchmark_stderr: String,
    ) -> Self {
        let metrics = BenchmarkMetrics::parse(&benchmark_stdout);
        Self {
            config,
            winning_metric,
            metrics,
            benchmark_stdout,
            benchmark_stderr,
        }
    }

    pub fn winning_value(&self) -> Option<f64> {
        self.metrics.value_for(self.winning_metric)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResultSet {
    pub winning_metric: Metric,
    pub trials: Vec<TrialResult>,
}

impl ResultSet {
    pub fn new(winning_metric: Metric) -> Self {
        Self {
            winning_metric,
            trials: Vec::new(),
        }
    }

    pub fn push(&mut self, result: TrialResult) {
        self.trials.push(result);
    }

    pub fn sort_best_first(&mut self) {
        let metric = self.winning_metric;
        self.trials
            .sort_by(|left, right| compare_results(left, right, metric));
    }

    pub fn best(&self) -> Option<&TrialResult> {
        self.trials.first()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultFiles {
    pub raw: PathBuf,
    pub summary: PathBuf,
}

pub fn create_run_dir(root: impl AsRef<Path>, kind: &str, engine: Engine) -> Result<PathBuf> {
    let root = root.as_ref();
    let dir = root.join(format!(
        "{kind}-{}-{}-{}",
        now_nanos()?,
        engine,
        std::process::id()
    ));
    fs::create_dir_all(&dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    Ok(dir)
}

pub fn write_trial_result(dir: impl AsRef<Path>, result: &TrialResult) -> Result<ResultFiles> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir).map_err(|err| format!("failed to create {}: {err}", dir.display()))?;

    let stem = format!(
        "trial-{}-{}-{}",
        now_nanos()?,
        result.config.engine,
        std::process::id()
    );
    let raw = dir.join(format!("{stem}.raw.txt"));
    let summary = dir.join(format!("{stem}.tsv"));

    fs::write(&raw, raw_text(result))
        .map_err(|err| format!("failed to write {}: {err}", raw.display()))?;
    fs::write(&summary, summary_tsv(result, &raw))
        .map_err(|err| format!("failed to write {}: {err}", summary.display()))?;

    Ok(ResultFiles { raw, summary })
}

pub fn write_best_config(dir: impl AsRef<Path>, text: &str) -> Result<PathBuf> {
    write_config_file(dir, "best.conf", text)
}

pub fn write_config_file(dir: impl AsRef<Path>, name: &str, text: &str) -> Result<PathBuf> {
    let path = dir.as_ref().join(name);
    fs::write(&path, text).map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    Ok(path)
}

fn compare_results(left: &TrialResult, right: &TrialResult, metric: Metric) -> Ordering {
    match (
        left.metrics.value_for(metric),
        right.metrics.value_for(metric),
    ) {
        (Some(left), Some(right)) if metric.lower_is_better() => left.total_cmp(&right),
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn first_number(text: &str) -> Option<f64> {
    text.split_whitespace()
        .find_map(|word| word.trim_end_matches(',').parse::<f64>().ok())
        .filter(|value| value.is_finite())
}

fn raw_text(result: &TrialResult) -> String {
    format!(
        "engine: {}\nmodel: {}\nwinning_metric: {}\n\n===== stdout =====\n{}\n===== stderr =====\n{}\n",
        result.config.engine,
        result.config.model,
        result.winning_metric,
        result.benchmark_stdout,
        result.benchmark_stderr
    )
}

fn summary_tsv(result: &TrialResult, raw_path: &Path) -> String {
    let metrics = &result.metrics;
    let columns = vec![
        ("engine", result.config.engine.to_string()),
        ("model", result.config.model.clone()),
        ("winning_metric", result.winning_metric.to_string()),
        ("winning_value", fmt_value(result.winning_value())),
        ("tp", result.config.candidate.parallelism.tensor.to_string()),
        (
            "memory_fraction",
            format!("{:.2}", result.config.candidate.memory.fraction),
        ),
        (
            "prefill_token_budget",
            result
                .config
                .candidate
                .scheduler
                .prefill_token_budget
                .to_string(),
        ),
        (
            "max_running_requests",
            result
                .config
                .candidate
                .scheduler
                .max_running_requests
                .to_string(),
        ),
        ("serve_args", format_engine_args(&result.config.serve_args)),
        (
            "successful_requests",
            fmt_value(metrics.successful_requests),
        ),
        ("failed_requests", fmt_value(metrics.failed_requests)),
        (
            "max_request_concurrency",
            fmt_value(metrics.max_request_concurrency),
        ),
        (
            "request_rate_configured",
            fmt_value(metrics.request_rate_configured),
        ),
        (
            "benchmark_duration_s",
            fmt_value(metrics.benchmark_duration_s),
        ),
        ("total_input_tokens", fmt_value(metrics.total_input_tokens)),
        (
            "total_input_text_tokens",
            fmt_value(metrics.total_input_text_tokens),
        ),
        (
            "total_input_vision_tokens",
            fmt_value(metrics.total_input_vision_tokens),
        ),
        (
            "total_generated_tokens",
            fmt_value(metrics.total_generated_tokens),
        ),
        (
            "total_generated_tokens_retokenized",
            fmt_value(metrics.total_generated_tokens_retokenized),
        ),
        ("request_throughput", fmt_value(metrics.request_throughput)),
        ("request_goodput", fmt_value(metrics.request_goodput)),
        (
            "input_token_throughput",
            fmt_value(metrics.input_token_throughput),
        ),
        (
            "output_token_throughput",
            fmt_value(metrics.output_token_throughput),
        ),
        (
            "output_token_throughput_retokenized",
            fmt_value(metrics.output_token_throughput_retokenized),
        ),
        (
            "peak_output_token_throughput",
            fmt_value(metrics.peak_output_token_throughput),
        ),
        (
            "peak_concurrent_requests",
            fmt_value(metrics.peak_concurrent_requests),
        ),
        (
            "total_token_throughput",
            fmt_value(metrics.total_token_throughput),
        ),
        (
            "total_token_throughput_retokenized",
            fmt_value(metrics.total_token_throughput_retokenized),
        ),
        ("concurrency", fmt_value(metrics.concurrency)),
        ("accept_length", fmt_value(metrics.accept_length)),
        ("rtfx", fmt_value(metrics.rtfx)),
        ("mean_ttft_ms", fmt_value(metrics.mean_ttft_ms)),
        ("median_ttft_ms", fmt_value(metrics.median_ttft_ms)),
        ("std_ttft_ms", fmt_value(metrics.std_ttft_ms)),
        ("p90_ttft_ms", fmt_value(metrics.p90_ttft_ms)),
        ("p95_ttft_ms", fmt_value(metrics.p95_ttft_ms)),
        ("p99_ttft_ms", fmt_value(metrics.p99_ttft_ms)),
        ("mean_tpot_ms", fmt_value(metrics.mean_tpot_ms)),
        ("median_tpot_ms", fmt_value(metrics.median_tpot_ms)),
        ("std_tpot_ms", fmt_value(metrics.std_tpot_ms)),
        ("p90_tpot_ms", fmt_value(metrics.p90_tpot_ms)),
        ("p95_tpot_ms", fmt_value(metrics.p95_tpot_ms)),
        ("p99_tpot_ms", fmt_value(metrics.p99_tpot_ms)),
        ("mean_itl_ms", fmt_value(metrics.mean_itl_ms)),
        ("median_itl_ms", fmt_value(metrics.median_itl_ms)),
        ("std_itl_ms", fmt_value(metrics.std_itl_ms)),
        ("p90_itl_ms", fmt_value(metrics.p90_itl_ms)),
        ("p95_itl_ms", fmt_value(metrics.p95_itl_ms)),
        ("p99_itl_ms", fmt_value(metrics.p99_itl_ms)),
        ("max_itl_ms", fmt_value(metrics.max_itl_ms)),
        ("mean_e2e_ms", fmt_value(metrics.mean_e2e_ms)),
        ("median_e2e_ms", fmt_value(metrics.median_e2e_ms)),
        ("std_e2e_ms", fmt_value(metrics.std_e2e_ms)),
        ("p90_e2e_ms", fmt_value(metrics.p90_e2e_ms)),
        ("p95_e2e_ms", fmt_value(metrics.p95_e2e_ms)),
        ("p99_e2e_ms", fmt_value(metrics.p99_e2e_ms)),
        ("raw_file", raw_path.display().to_string()),
    ];
    let header = columns
        .iter()
        .map(|(name, _)| *name)
        .collect::<Vec<_>>()
        .join("\t");
    let row = columns
        .into_iter()
        .map(|(_, value)| value)
        .collect::<Vec<_>>()
        .join("\t");
    format!("{header}\n{row}\n")
}

fn fmt_value(value: Option<f64>) -> String {
    value.map(|value| format!("{value:.4}")).unwrap_or_default()
}

fn format_engine_args(args: &[EngineArg]) -> String {
    args.iter()
        .map(|arg| match &arg.value {
            Some(value) => format!("{}={value}", arg.name),
            None => arg.name.clone(),
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn now_nanos() -> Result<u128> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .map_err(|err| format!("system clock is before unix epoch: {err}"))
}

trait MetricDirection {
    fn lower_is_better(self) -> bool;
}

impl MetricDirection for Metric {
    fn lower_is_better(self) -> bool {
        matches!(
            self,
            Metric::Ttft
                | Metric::P90Ttft
                | Metric::P95Ttft
                | Metric::P99Ttft
                | Metric::Tpot
                | Metric::P90Tpot
                | Metric::P95Tpot
                | Metric::P99Tpot
                | Metric::Itl
                | Metric::P90Itl
                | Metric::P95Itl
                | Metric::P99Itl
                | Metric::E2e
                | Metric::P90E2e
                | Metric::P95E2e
                | Metric::P99E2e
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BenchmarkConfig;
    use crate::engine::Engine;
    use crate::trial::Candidate;

    fn config() -> ServingConfig {
        ServingConfig {
            engine: Engine::Vllm,
            image: "vllm/vllm-openai:latest".to_string(),
            model: "m".to_string(),
            gpus: 1,
            host: "127.0.0.1".to_string(),
            port: 8000,
            startup_timeout_secs: 300,
            max_model_len: 8192,
            metric: Metric::Tps,
            candidate: Candidate::default(),
            serve_args: Vec::new(),
            benchmark: BenchmarkConfig::default(),
        }
    }

    #[test]
    fn parses_vllm_benchmark_summary() {
        let metrics = BenchmarkMetrics::parse(
            "Successful requests:                     4
Failed requests:                         0
Maximum request concurrency:             1
Request rate configured (RPS):           1.00
Benchmark duration (s):                  18.59
Total input tokens:                      4096
Total generated tokens:                  512
Request throughput (req/s):              0.22
Output token throughput (tok/s):           27.54
Peak output token throughput (tok/s):    30.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):            247.82
Mean TTFT (ms):                          167.28
P95 TTFT (ms):                           170.00
P99 TTFT (ms):                           172.32
Mean TPOT (ms):                          33.61
P99 TPOT (ms):                           33.62
Mean ITL (ms):                           33.61
P99 ITL (ms):                            33.99
Mean E2EL (ms):                          4301.25
P99 E2EL (ms):                           4400.00",
        );

        assert_eq!(metrics.successful_requests, Some(4.0));
        assert_eq!(metrics.failed_requests, Some(0.0));
        assert_eq!(metrics.max_request_concurrency, Some(1.0));
        assert_eq!(metrics.request_rate_configured, Some(1.0));
        assert_eq!(metrics.benchmark_duration_s, Some(18.59));
        assert_eq!(metrics.total_input_tokens, Some(4096.0));
        assert_eq!(metrics.total_generated_tokens, Some(512.0));
        assert_eq!(metrics.output_token_throughput, Some(27.54));
        assert_eq!(metrics.peak_output_token_throughput, Some(30.0));
        assert_eq!(metrics.peak_concurrent_requests, Some(2.0));
        assert_eq!(metrics.mean_ttft_ms, Some(167.28));
        assert_eq!(metrics.p95_ttft_ms, Some(170.0));
        assert_eq!(metrics.p99_ttft_ms, Some(172.32));
        assert_eq!(metrics.mean_tpot_ms, Some(33.61));
        assert_eq!(metrics.p99_tpot_ms, Some(33.62));
        assert_eq!(metrics.mean_itl_ms, Some(33.61));
        assert_eq!(metrics.p99_itl_ms, Some(33.99));
        assert_eq!(metrics.mean_e2e_ms, Some(4301.25));
        assert_eq!(metrics.p99_e2e_ms, Some(4400.0));
        assert_eq!(metrics.value_for(Metric::P99Ttft), Some(172.32));
    }

    #[test]
    fn parses_sglang_benchmark_summary() {
        let metrics = BenchmarkMetrics::parse(
            "Backend:                                sglang
Traffic request rate:                    1
Max request concurrency:                 4
Successful requests:                     8
Benchmark duration (s):                  10.00
Total input tokens:                      8192
Total input text tokens:                 8192
Total generated tokens:                  512
Total generated tokens (retokenized):    500
Request throughput (req/s):              0.80
Input token throughput (tok/s):          819.20
Output token throughput (tok/s):         51.20
Peak output token throughput (tok/s):    80.00
Peak concurrent requests:                4
Total token throughput (tok/s):          870.40
Concurrency:                             3.20
Accept length:                           1.50
Mean E2E Latency (ms):                   900.00
Median E2E Latency (ms):                 850.00
P90 E2E Latency (ms):                    1000.00
P95 E2E Latency (ms):                    1100.00
P99 E2E Latency (ms):                    1200.00
Mean TTFT (ms):                          120.00
P90 TTFT (ms):                           140.00
P95 TTFT (ms):                           150.00
P99 TTFT (ms):                           160.00
Mean TPOT (ms):                          20.00
P99 TPOT (ms):                           25.00
Mean ITL (ms):                           19.00
P99 ITL (ms):                            24.00
Max ITL (ms):                            30.00",
        );

        assert_eq!(metrics.request_rate_configured, Some(1.0));
        assert_eq!(metrics.max_request_concurrency, Some(4.0));
        assert_eq!(metrics.total_input_text_tokens, Some(8192.0));
        assert_eq!(metrics.total_generated_tokens_retokenized, Some(500.0));
        assert_eq!(metrics.input_token_throughput, Some(819.2));
        assert_eq!(metrics.output_token_throughput, Some(51.2));
        assert_eq!(metrics.total_token_throughput, Some(870.4));
        assert_eq!(metrics.concurrency, Some(3.2));
        assert_eq!(metrics.accept_length, Some(1.5));
        assert_eq!(metrics.mean_e2e_ms, Some(900.0));
        assert_eq!(metrics.p99_e2e_ms, Some(1200.0));
        assert_eq!(metrics.p99_ttft_ms, Some(160.0));
        assert_eq!(metrics.max_itl_ms, Some(30.0));
    }

    #[test]
    fn sorts_higher_tps_and_lower_tail_latency_first() {
        let slow = TrialResult::new(
            config(),
            Metric::Tps,
            "Output token throughput (tok/s): 1".to_string(),
            String::new(),
        );
        let fast = TrialResult::new(
            config(),
            Metric::Tps,
            "Output token throughput (tok/s): 2".to_string(),
            String::new(),
        );
        let mut set = ResultSet::new(Metric::Tps);
        set.push(slow);
        set.push(fast);
        set.sort_best_first();
        assert_eq!(set.best().unwrap().winning_value(), Some(2.0));

        let high = TrialResult::new(
            config(),
            Metric::Ttft,
            "Mean TTFT (ms): 100".to_string(),
            String::new(),
        );
        let low = TrialResult::new(
            config(),
            Metric::Ttft,
            "Mean TTFT (ms): 50".to_string(),
            String::new(),
        );
        let mut set = ResultSet::new(Metric::Ttft);
        set.push(high);
        set.push(low);
        set.sort_best_first();
        assert_eq!(set.best().unwrap().winning_value(), Some(50.0));

        let high = TrialResult::new(
            config(),
            Metric::P99Tpot,
            "P99 TPOT (ms): 100".to_string(),
            String::new(),
        );
        let low = TrialResult::new(
            config(),
            Metric::P99Tpot,
            "P99 TPOT (ms): 50".to_string(),
            String::new(),
        );
        let mut set = ResultSet::new(Metric::P99Tpot);
        set.push(high);
        set.push(low);
        set.sort_best_first();
        assert_eq!(set.best().unwrap().winning_value(), Some(50.0));
    }

    #[test]
    fn writes_raw_and_summary_files() {
        let result = TrialResult::new(
            config(),
            Metric::Tps,
            "Output token throughput (tok/s): 27.54".to_string(),
            String::new(),
        );
        let dir = std::env::temp_dir().join(format!(
            "optimum-advisor-results-test-{}",
            std::process::id()
        ));

        let files = write_trial_result(&dir, &result).unwrap();

        assert!(files.raw.exists());
        assert!(files.summary.exists());
        let summary = fs::read_to_string(files.summary).unwrap();
        assert!(summary.contains("winning_metric"));
        assert!(summary.contains("27.5400"));
        assert!(summary.contains("p99_ttft_ms"));
        assert!(summary.contains("peak_output_token_throughput"));
    }

    #[test]
    fn creates_run_dir_and_best_config() {
        let root = std::env::temp_dir().join(format!(
            "optimum-advisor-result-dir-test-{}",
            std::process::id()
        ));

        let dir = create_run_dir(&root, "sweep", Engine::Vllm).unwrap();
        let best = write_best_config(&dir, "engine = vllm\n").unwrap();

        assert!(dir.starts_with(&root));
        assert!(dir.file_name().unwrap().to_string_lossy().contains("sweep"));
        assert!(dir.file_name().unwrap().to_string_lossy().contains("vllm"));
        assert_eq!(best.file_name().unwrap(), "best.conf");
        assert_eq!(fs::read_to_string(best).unwrap(), "engine = vllm\n");
    }
}
