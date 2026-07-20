use std::{
    collections::HashSet,
    net::{IpAddr, Ipv4Addr},
};

use super::{
    BenchmarkConfig, ConfigInput, CorrectnessConfig, LeaderboardConfig, ModelMemoryConfig,
    NormalizedConfig, RuntimeConfig, DEFAULT_LEADERBOARD_URL,
};
use crate::{
    domain::{
        candidate::{canonical_name, validate_dynamic_name, Candidate, CandidateSpec, DynamicArg},
        engine::{Engine, Metric},
        run::PullPolicy,
    },
    error::{Error, Result},
    runtime::process::{
        DEFAULT_BENCHMARK_TIMEOUT, DEFAULT_CORRECTNESS_TIMEOUT, DEFAULT_INSPECTION_TIMEOUT,
        DEFAULT_MAX_PROCESS_OUTPUT_BYTES, DEFAULT_STARTUP_TIMEOUT,
    },
};

pub(super) fn normalize(input: ConfigInput) -> Result<NormalizedConfig> {
    if let Some(version) = input.schema_version {
        if version != 2 {
            return Err(Error::validation(format!(
                "unsupported schema_version {version}; expected 2"
            )));
        }
    }
    let engine = input
        .engine
        .ok_or_else(|| Error::validation("engine is required after configuration merge"))?;
    let model = required_trimmed(input.model, "model")?;
    let metric = input.metric.unwrap_or_else(|| default_metric(&model));
    let image = match input.image {
        Some(image) => required_trimmed(Some(image), "image")?,
        None => engine.default_image().to_string(),
    };

    let gpus = input.runtime.gpus.unwrap_or(1);
    positive(gpus, "runtime.gpus")?;
    let gpu_devices = normalize_gpu_devices(input.runtime.gpu_devices, gpus)?;
    let port = input.runtime.port.unwrap_or(8000);
    positive(port, "runtime.port")?;
    let max_model_len = input.runtime.max_model_len.unwrap_or(8_192);
    positive(max_model_len, "runtime.max_model_len")?;
    let startup_timeout_secs = input
        .runtime
        .startup_timeout_secs
        .unwrap_or(DEFAULT_STARTUP_TIMEOUT.as_secs());
    positive(startup_timeout_secs, "runtime.startup_timeout_secs")?;
    let benchmark_timeout_secs = input
        .runtime
        .benchmark_timeout_secs
        .unwrap_or(DEFAULT_BENCHMARK_TIMEOUT.as_secs());
    positive(benchmark_timeout_secs, "runtime.benchmark_timeout_secs")?;
    let max_process_output_bytes = input
        .runtime
        .max_process_output_bytes
        .unwrap_or(DEFAULT_MAX_PROCESS_OUTPUT_BYTES);
    positive(max_process_output_bytes, "runtime.max_process_output_bytes")?;
    let runtime = RuntimeConfig {
        gpus,
        gpu_devices,
        pull_policy: input.runtime.pull_policy.unwrap_or(PullPolicy::Missing),
        allow_local_image: input.runtime.allow_local_image.unwrap_or(false),
        bind_host: input
            .runtime
            .bind_host
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST)),
        port,
        max_model_len,
        startup_timeout_secs,
        benchmark_timeout_secs,
        max_process_output_bytes,
    };

    let dataset_name = required_trimmed(
        Some(
            input
                .benchmark
                .dataset_name
                .unwrap_or_else(|| "random".to_string()),
        ),
        "benchmark.dataset_name",
    )?;
    let num_prompts = input.benchmark.num_prompts.unwrap_or(100);
    positive(num_prompts, "benchmark.num_prompts")?;
    let request_rate = normalize_request_rate(
        input
            .benchmark
            .request_rate
            .unwrap_or_else(|| "1".to_string()),
    )?;
    let max_concurrency = input.benchmark.max_concurrency.unwrap_or(1);
    positive(max_concurrency, "benchmark.max_concurrency")?;
    let random_input_len = input.benchmark.random_input_len.unwrap_or(1024);
    positive(random_input_len, "benchmark.random_input_len")?;
    let random_output_len = input.benchmark.random_output_len.unwrap_or(128);
    positive(random_output_len, "benchmark.random_output_len")?;
    let benchmark = BenchmarkConfig {
        dataset_name,
        num_prompts,
        request_rate,
        max_concurrency,
        random_input_len,
        random_output_len,
    };

    let mut candidate = default_candidate(engine, metric);
    input.candidate.apply_to(&mut candidate);
    validate_candidate(&candidate, gpus)?;
    let serve_args = normalize_dynamic_args(input.serve_args)?;

    let correctness = CorrectnessConfig {
        enabled: input.correctness.enabled.unwrap_or(true),
        threshold: input.correctness.threshold.unwrap_or(0.2),
        timeout_secs: input
            .correctness
            .timeout_secs
            .unwrap_or(DEFAULT_CORRECTNESS_TIMEOUT.as_secs()),
    };
    if !correctness.threshold.is_finite() || !(0.0..=1.0).contains(&correctness.threshold) {
        return Err(Error::validation(
            "correctness.threshold must be finite and in [0, 1]",
        ));
    }
    positive(correctness.timeout_secs, "correctness.timeout_secs")?;

    let model_memory = ModelMemoryConfig {
        enabled: input.model_memory.enabled.unwrap_or(true),
        required: input.model_memory.required.unwrap_or(false),
        command: input.model_memory.command,
        timeout_secs: input
            .model_memory
            .timeout_secs
            .unwrap_or(DEFAULT_INSPECTION_TIMEOUT.as_secs()),
    };
    if model_memory.required && !model_memory.enabled {
        return Err(Error::validation(
            "model_memory.required cannot be true when model_memory.enabled is false",
        ));
    }
    if model_memory
        .command
        .as_ref()
        .is_some_and(|command| command.as_os_str().is_empty())
    {
        return Err(Error::validation("model_memory.command must not be empty"));
    }
    positive(model_memory.timeout_secs, "model_memory.timeout_secs")?;

    let leaderboard = LeaderboardConfig {
        submit: input.leaderboard.submit.unwrap_or(false),
        url: required_trimmed(
            Some(
                input
                    .leaderboard
                    .url
                    .unwrap_or_else(|| DEFAULT_LEADERBOARD_URL.to_string()),
            ),
            "leaderboard.url",
        )?,
    };

    if let Some(sweep) = &input.sweep {
        let base = CandidateSpec {
            candidate: candidate.clone(),
            serve_args: serve_args.clone(),
        };
        for generated in sweep.candidates(&base)? {
            validate_candidate(&generated.candidate, gpus)?;
            normalize_dynamic_args(generated.serve_args)?;
        }
    }

    Ok(NormalizedConfig {
        engine,
        image,
        model,
        metric,
        runtime,
        benchmark,
        candidate,
        correctness,
        model_memory,
        leaderboard,
        serve_args,
        sweep: input.sweep,
    })
}

fn default_candidate(engine: Engine, metric: Metric) -> Candidate {
    let memory_fraction = match engine {
        Engine::Vllm => 0.9,
        Engine::Sglang => 0.88,
    };
    let prefill_token_budget = match (engine, metric) {
        (Engine::Vllm, Metric::Ttft | Metric::P90Ttft | Metric::P95Ttft | Metric::P99Ttft) => {
            16_384
        }
        (
            _,
            Metric::Tpot
            | Metric::P90Tpot
            | Metric::P95Tpot
            | Metric::P99Tpot
            | Metric::Itl
            | Metric::P90Itl
            | Metric::P95Itl
            | Metric::P99Itl,
        ) => 2_048,
        _ => 8_192,
    };
    Candidate {
        tensor_parallelism: 1,
        memory_fraction,
        prefill_token_budget,
        max_running_requests: 256,
    }
}

pub(crate) fn validate_candidate(candidate: &Candidate, gpus: usize) -> Result<()> {
    if candidate.tensor_parallelism == 0 {
        return Err(Error::validation(
            "candidate.tensor_parallelism must be greater than zero",
        ));
    }
    if candidate.tensor_parallelism > gpus {
        return Err(Error::validation(format!(
            "candidate.tensor_parallelism {} must not exceed requested GPU count {gpus}",
            candidate.tensor_parallelism
        )));
    }
    if !candidate.memory_fraction.is_finite()
        || candidate.memory_fraction <= 0.0
        || candidate.memory_fraction > 1.0
    {
        return Err(Error::validation(
            "candidate.memory_fraction must be finite and in (0, 1]",
        ));
    }
    positive(
        candidate.prefill_token_budget,
        "candidate.prefill_token_budget",
    )?;
    positive(
        candidate.max_running_requests,
        "candidate.max_running_requests",
    )?;
    Ok(())
}

fn normalize_dynamic_args(arguments: Vec<DynamicArg>) -> Result<Vec<DynamicArg>> {
    let mut normalized = Vec::with_capacity(arguments.len());
    let mut seen = HashSet::with_capacity(arguments.len());
    for argument in arguments {
        let name = canonical_name(&argument.name);
        validate_dynamic_name(&name)?;
        if !seen.insert(name.clone()) {
            return Err(Error::validation(format!(
                "duplicate canonical engine argument: {name}"
            )));
        }
        normalized.push(DynamicArg {
            name,
            value: argument.value,
        });
    }
    normalized.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(normalized)
}

fn normalize_gpu_devices(devices: Option<Vec<String>>, gpus: usize) -> Result<Vec<String>> {
    let Some(devices) = devices else {
        return Ok(Vec::new());
    };
    if devices.len() != gpus {
        return Err(Error::validation(format!(
            "runtime.gpu_devices has {} entries but runtime.gpus is {gpus}",
            devices.len()
        )));
    }
    let mut normalized = Vec::with_capacity(devices.len());
    let mut seen = HashSet::with_capacity(devices.len());
    for device in devices {
        let device = device.trim().to_string();
        if device.is_empty() {
            return Err(Error::validation(
                "runtime.gpu_devices entries must not be empty",
            ));
        }
        if !seen.insert(device.clone()) {
            return Err(Error::validation(format!(
                "duplicate runtime.gpu_devices entry: {device}"
            )));
        }
        normalized.push(device);
    }
    Ok(normalized)
}

fn default_metric(model: &str) -> Metric {
    const MAX_LATENCY_DEFAULT_BILLIONS: f64 = 3.0;

    let parameter_billions = model
        .split(['/', '-'])
        .filter_map(parameter_count_billions)
        .next_back();
    if parameter_billions.is_some_and(|size| size <= MAX_LATENCY_DEFAULT_BILLIONS) {
        Metric::Tpot
    } else {
        Metric::Tps
    }
}

fn parameter_count_billions(segment: &str) -> Option<f64> {
    let (number, scale) = if let Some(number) = segment
        .strip_suffix('B')
        .or_else(|| segment.strip_suffix('b'))
    {
        (number, 1.0)
    } else {
        let number = segment
            .strip_suffix('M')
            .or_else(|| segment.strip_suffix('m'))?;
        (number, 0.001)
    };
    let decimal;
    let number = if number.contains('_') {
        decimal = number.replace('_', ".");
        decimal.as_str()
    } else {
        number
    };
    let value = number.parse::<f64>().ok()? * scale;
    (value.is_finite() && value > 0.0).then_some(value)
}

fn normalize_request_rate(value: String) -> Result<String> {
    let value = value.trim();
    if value == "inf" {
        return Ok(value.to_string());
    }
    let parsed = value.parse::<f64>().map_err(|_| {
        Error::validation("benchmark.request_rate must be a positive finite number or 'inf'")
    })?;
    if !parsed.is_finite() || parsed <= 0.0 {
        return Err(Error::validation(
            "benchmark.request_rate must be a positive finite number or 'inf'",
        ));
    }
    Ok(value.to_string())
}

fn required_trimmed(value: Option<String>, label: &str) -> Result<String> {
    let value = value.ok_or_else(|| Error::validation(format!("{label} is required")))?;
    let value = value.trim();
    if value.is_empty() {
        return Err(Error::validation(format!("{label} must not be empty")));
    }
    Ok(value.to_string())
}

fn positive<T>(value: T, label: &str) -> Result<()>
where
    T: Copy + PartialEq + From<u8>,
{
    if value == T::from(0) {
        Err(Error::validation(format!(
            "{label} must be greater than zero"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::candidate::{CandidateOverrides, DynamicArg};

    #[test]
    fn tiny_model_defaults_to_decode_latency_metric() {
        let normalized = ConfigInput::minimal(Engine::Vllm, "Qwen/Qwen3-0.6B")
            .normalize()
            .unwrap();

        assert_eq!(normalized.metric, Metric::Tpot);
    }

    #[test]
    fn common_sub_three_billion_notations_default_to_decode_latency() {
        for model in [
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            "stabilityai/stablelm-2-1_6b",
        ] {
            let normalized = ConfigInput::minimal(Engine::Vllm, model)
                .normalize()
                .unwrap();

            assert_eq!(normalized.metric, Metric::Tpot, "{model}");
        }
    }

    #[test]
    fn larger_or_unversioned_model_defaults_to_throughput_metric() {
        for model in ["Qwen/Qwen3-4B-Instruct-2507", "repo/model"] {
            let normalized = ConfigInput::minimal(Engine::Vllm, model)
                .normalize()
                .unwrap();

            assert_eq!(normalized.metric, Metric::Tps, "{model}");
        }
    }

    #[test]
    fn explicit_metric_overrides_model_aware_default() {
        let mut input = ConfigInput::minimal(Engine::Vllm, "Qwen/Qwen3-0.6B");
        input.metric = Some(Metric::ReqS);

        assert_eq!(input.normalize().unwrap().metric, Metric::ReqS);
    }

    #[test]
    fn explicit_candidate_values_survive_engine_defaults() {
        let mut input = ConfigInput::minimal(Engine::Sglang, "m");
        input.candidate = CandidateOverrides {
            memory_fraction: Some(0.73),
            prefill_token_budget: Some(4096),
            ..CandidateOverrides::default()
        };

        let normalized = input.normalize().unwrap();

        assert_eq!(normalized.candidate.memory_fraction, 0.73);
        assert_eq!(normalized.candidate.prefill_token_budget, 4096);
    }

    #[test]
    fn allows_parallelism_that_does_not_divide_gpu_pool() {
        let mut input = ConfigInput::minimal(Engine::Vllm, "m");
        input.runtime.gpus = Some(4);
        input.candidate.tensor_parallelism = Some(3);

        assert!(input.normalize().is_ok());
    }

    #[test]
    fn rejects_parallelism_instead_of_clamping() {
        let mut input = ConfigInput::minimal(Engine::Vllm, "m");
        input.runtime.gpus = Some(2);
        input.candidate.tensor_parallelism = Some(3);

        let error = input.normalize().unwrap_err();

        assert!(error.to_string().contains("must not exceed"));
    }

    #[test]
    fn rejects_dynamic_arguments_owned_by_normalized_scheduler_config() {
        let mut input = ConfigInput::minimal(Engine::Vllm, "m");
        input.serve_args = vec![DynamicArg::value("max-num-batched-tokens", "4096")];

        let error = input.normalize().unwrap_err();

        assert!(error.to_string().contains("owned"));
    }

    #[test]
    fn owned_dynamic_arguments_name_the_canonical_config_field() {
        for (name, replacement) in [
            ("max-num-seqs", "candidate.max_running_requests"),
            ("max-model-len", "runtime.max_model_len"),
            ("gpu-memory-utilization", "candidate.memory_fraction"),
        ] {
            let mut input = ConfigInput::minimal(Engine::Vllm, "m");
            input.serve_args = vec![DynamicArg::value(name, "1")];

            let error = input.normalize().unwrap_err();

            assert!(
                error
                    .to_string()
                    .contains(&format!("set {replacement} instead")),
                "{error}"
            );
        }
    }

    #[test]
    fn max_model_length_is_explicit_and_positive() {
        let normalized = ConfigInput::minimal(Engine::Vllm, "m").normalize().unwrap();
        assert_eq!(normalized.runtime.max_model_len, 8_192);

        let mut invalid = ConfigInput::minimal(Engine::Vllm, "m");
        invalid.runtime.max_model_len = Some(0);
        assert!(invalid.normalize().is_err());
    }
}
