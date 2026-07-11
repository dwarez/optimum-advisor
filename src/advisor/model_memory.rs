use std::io::ErrorKind;
use std::process::Command;

use serde::Serialize;

use crate::config::ServingConfig;
use crate::serve::EngineArg;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ModelMemoryEstimate {
    pub source: String,
    pub model: String,
    pub max_model_len: u32,
    pub batch_size: u32,
    pub kv_cache_dtype: String,
    pub weights_bytes: Option<u64>,
    pub kv_cache_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    pub raw_stdout: String,
    pub raw_stderr: String,
    pub warnings: Vec<String>,
}

pub fn estimate_model_memory(config: &ServingConfig) -> ModelMemoryEstimate {
    let mut estimate = ModelMemoryEstimate {
        source: "hf-mem".to_string(),
        model: config.model.clone(),
        max_model_len: config.max_model_len,
        batch_size: config.benchmark.max_concurrency.unwrap_or(1),
        kv_cache_dtype: kv_cache_dtype(&config.serve_args),
        ..Default::default()
    };

    match run_hf_mem(&estimate) {
        Ok(output) => {
            estimate.source = output.command;
            estimate.raw_stdout = output.stdout;
            estimate.raw_stderr = output.stderr;
            estimate.weights_bytes = json_u64(&estimate.raw_stdout, "memory");
            estimate.kv_cache_bytes = json_u64(&estimate.raw_stdout, "kv_cache");
            estimate.total_bytes = json_u64(&estimate.raw_stdout, "total_memory");
            if estimate.total_bytes.is_none() {
                estimate
                    .warnings
                    .push("hf-mem returned no total_memory field".to_string());
            }
        }
        Err(warning) => estimate.warnings.push(warning),
    }

    estimate
}

pub fn format_model_memory_estimate(estimate: &ModelMemoryEstimate) -> String {
    let mut text = format!(
        "source: {}\nmodel: {}\nmax_model_len: {}\nbatch_size: {}\nkv_cache_dtype: {}\n",
        estimate.source,
        estimate.model,
        estimate.max_model_len,
        estimate.batch_size,
        estimate.kv_cache_dtype
    );
    if let Some(value) = estimate.weights_bytes {
        text.push_str(&format!("weights_bytes: {value}\n"));
    }
    if let Some(value) = estimate.kv_cache_bytes {
        text.push_str(&format!("kv_cache_bytes: {value}\n"));
    }
    if let Some(value) = estimate.total_bytes {
        text.push_str(&format!("total_bytes: {value}\n"));
    }
    for warning in &estimate.warnings {
        text.push_str(&format!("warning: {warning}\n"));
    }
    if !estimate.raw_stderr.trim().is_empty() {
        text.push_str("stderr:\n");
        text.push_str(estimate.raw_stderr.trim());
        text.push('\n');
    }
    if !estimate.raw_stdout.trim().is_empty() {
        text.push_str("stdout:\n");
        text.push_str(estimate.raw_stdout.trim());
        text.push('\n');
    }
    text
}

pub fn summarize_model_memory(estimate: &ModelMemoryEstimate) -> String {
    if let Some(total) = estimate.total_bytes {
        let weights = estimate
            .weights_bytes
            .map(format_gib)
            .unwrap_or_else(|| "unknown".to_string());
        let kv_cache = estimate
            .kv_cache_bytes
            .map(format_gib)
            .unwrap_or_else(|| "unknown".to_string());
        return format!(
            "total={} weights={} kv_cache={} batch_size={} max_model_len={}",
            format_gib(total),
            weights,
            kv_cache,
            estimate.batch_size,
            estimate.max_model_len
        );
    }

    let warning = if estimate.warnings.is_empty() {
        "unavailable".to_string()
    } else {
        estimate.warnings.join("; ")
    };
    format!("unavailable warning={warning}")
}

struct HfMemOutput {
    command: String,
    stdout: String,
    stderr: String,
}

fn run_hf_mem(estimate: &ModelMemoryEstimate) -> Result<HfMemOutput, String> {
    let args = hf_mem_args(estimate);
    match run_command("uvx", &[vec!["hf-mem".to_string()], args.clone()].concat()) {
        Ok(output) => Ok(output),
        Err(err) if err.kind == Some(ErrorKind::NotFound) => {
            run_command("hf-mem", &args).map_err(|err| err.message)
        }
        Err(err) => Err(err.message),
    }
}

struct CommandError {
    kind: Option<ErrorKind>,
    message: String,
}

fn run_command(command: &str, args: &[String]) -> Result<HfMemOutput, CommandError> {
    let output = Command::new(command)
        .args(args)
        .output()
        .map_err(|err| CommandError {
            kind: Some(err.kind()),
            message: format!("{command} unavailable: {err}"),
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if output.status.success() {
        Ok(HfMemOutput {
            command: command.to_string(),
            stdout,
            stderr,
        })
    } else {
        Err(CommandError {
            kind: None,
            message: format!("{command} failed: {}", tail(&stderr)),
        })
    }
}

fn hf_mem_args(estimate: &ModelMemoryEstimate) -> Vec<String> {
    vec![
        "--model-id".to_string(),
        estimate.model.clone(),
        "--experimental".to_string(),
        "--json-output".to_string(),
        "--max-model-len".to_string(),
        estimate.max_model_len.to_string(),
        "--batch-size".to_string(),
        estimate.batch_size.to_string(),
        "--kv-cache-dtype".to_string(),
        estimate.kv_cache_dtype.clone(),
    ]
}

fn kv_cache_dtype(args: &[EngineArg]) -> String {
    args.iter()
        .rev()
        .find(|arg| arg.name == "--kv-cache-dtype")
        .and_then(|arg| arg.value.clone())
        .unwrap_or_else(|| "auto".to_string())
}

fn json_u64(text: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let value = text.split(&needle).nth(1)?.split_once(':')?.1.trim_start();
    let value = value.strip_prefix('"').unwrap_or(value);
    let digits = value
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse().ok()
}

fn format_gib(bytes: u64) -> String {
    format!("{:.2}GiB", bytes as f64 / 1024.0 / 1024.0 / 1024.0)
}

fn tail(text: &str) -> String {
    let text = text.trim();
    if text.len() <= 500 {
        text.to_string()
    } else {
        text[text.len() - 500..].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BenchmarkConfig;
    use crate::engine::{Engine, Metric};
    use crate::trial::Candidate;

    fn config() -> ServingConfig {
        ServingConfig {
            engine: Engine::Vllm,
            image: "vllm/vllm-openai:latest".to_string(),
            resolved_image: None,
            model: "Qwen/Qwen3-4B-Instruct-2507".to_string(),
            gpus: 1,
            host: "127.0.0.1".to_string(),
            port: 8000,
            startup_timeout_secs: 300,
            max_model_len: 8192,
            metric: Metric::Tps,
            candidate: Candidate::default(),
            serve_args: vec![EngineArg::value("kv-cache-dtype", "fp8")],
            benchmark: BenchmarkConfig {
                max_concurrency: Some(4),
                ..Default::default()
            },
        }
    }

    #[test]
    fn builds_hf_mem_args_from_config() {
        let estimate = ModelMemoryEstimate {
            model: config().model,
            max_model_len: 8192,
            batch_size: 4,
            kv_cache_dtype: "fp8".to_string(),
            ..Default::default()
        };

        let args = hf_mem_args(&estimate);

        assert!(args
            .windows(2)
            .any(|pair| pair == ["--model-id", "Qwen/Qwen3-4B-Instruct-2507"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--max-model-len", "8192"]));
        assert!(args.windows(2).any(|pair| pair == ["--batch-size", "4"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["--kv-cache-dtype", "fp8"]));
    }

    #[test]
    fn parses_hf_mem_json_numbers() {
        let text = r#"{"memory":230121630720,"kv_cache":24964497408,"total_memory":255086128128}"#;

        assert_eq!(json_u64(text, "memory"), Some(230121630720));
        assert_eq!(json_u64(text, "kv_cache"), Some(24964497408));
        assert_eq!(json_u64(text, "total_memory"), Some(255086128128));
    }

    #[test]
    fn estimate_shape_uses_serving_config() {
        let estimate = ModelMemoryEstimate {
            source: "uvx".to_string(),
            model: config().model,
            max_model_len: 8192,
            batch_size: 4,
            kv_cache_dtype: kv_cache_dtype(&config().serve_args),
            weights_bytes: Some(10),
            kv_cache_bytes: Some(20),
            total_bytes: Some(30),
            raw_stdout: String::new(),
            raw_stderr: String::new(),
            warnings: Vec::new(),
        };

        let text = format_model_memory_estimate(&estimate);

        assert!(text.contains("model: Qwen/Qwen3-4B-Instruct-2507"));
        assert!(text.contains("batch_size: 4"));
        assert!(text.contains("kv_cache_dtype: fp8"));
        assert!(text.contains("total_bytes: 30"));
    }
}
