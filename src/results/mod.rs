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
    pub request_throughput: Option<f64>,
    pub output_token_throughput: Option<f64>,
    pub total_token_throughput: Option<f64>,
    pub mean_ttft_ms: Option<f64>,
    pub median_ttft_ms: Option<f64>,
    pub p99_ttft_ms: Option<f64>,
    pub mean_tpot_ms: Option<f64>,
    pub median_tpot_ms: Option<f64>,
    pub p99_tpot_ms: Option<f64>,
    pub mean_itl_ms: Option<f64>,
    pub median_itl_ms: Option<f64>,
    pub p99_itl_ms: Option<f64>,
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
            Metric::Ttft => self.mean_ttft_ms,
            Metric::Itl => self.mean_itl_ms,
        }
    }

    fn set(&mut self, label: &str, value: Option<f64>) {
        match label {
            "Successful requests" => self.successful_requests = value,
            "Failed requests" => self.failed_requests = value,
            "Request throughput (req/s)" => self.request_throughput = value,
            "Output token throughput (tok/s)" => self.output_token_throughput = value,
            "Total token throughput (tok/s)" => self.total_token_throughput = value,
            "Mean TTFT (ms)" => self.mean_ttft_ms = value,
            "Median TTFT (ms)" => self.median_ttft_ms = value,
            "P99 TTFT (ms)" => self.p99_ttft_ms = value,
            "Mean TPOT (ms)" => self.mean_tpot_ms = value,
            "Median TPOT (ms)" => self.median_tpot_ms = value,
            "P99 TPOT (ms)" => self.p99_tpot_ms = value,
            "Mean ITL (ms)" => self.mean_itl_ms = value,
            "Median ITL (ms)" => self.median_itl_ms = value,
            "P99 ITL (ms)" => self.p99_itl_ms = value,
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
    format!(
        "engine\tmodel\twinning_metric\twinning_value\ttp\tmemory_fraction\tprefill_token_budget\tmax_running_requests\tserve_args\trequest_throughput\toutput_token_throughput\ttotal_token_throughput\tmean_ttft_ms\tmean_itl_ms\traw_file\n{}\t{}\t{}\t{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        result.config.engine,
        result.config.model,
        result.winning_metric,
        fmt_value(result.winning_value()),
        result.config.candidate.parallelism.tensor,
        result.config.candidate.memory.fraction,
        result.config.candidate.scheduler.prefill_token_budget,
        result.config.candidate.scheduler.max_running_requests,
        format_engine_args(&result.config.serve_args),
        fmt_value(result.metrics.request_throughput),
        fmt_value(result.metrics.output_token_throughput),
        fmt_value(result.metrics.total_token_throughput),
        fmt_value(result.metrics.mean_ttft_ms),
        fmt_value(result.metrics.mean_itl_ms),
        raw_path.display()
    )
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
        matches!(self, Metric::Ttft | Metric::Itl)
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
Output token throughput (tok/s):           27.54
Total token throughput (tok/s):            247.82
Mean TTFT (ms):                          167.28
Mean ITL (ms):                           33.61",
        );

        assert_eq!(metrics.successful_requests, Some(4.0));
        assert_eq!(metrics.output_token_throughput, Some(27.54));
        assert_eq!(metrics.mean_ttft_ms, Some(167.28));
        assert_eq!(metrics.mean_itl_ms, Some(33.61));
    }

    #[test]
    fn sorts_higher_tps_and_lower_ttft_first() {
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
