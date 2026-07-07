use std::fs;
use std::io::Write;
use std::path::Path;

use crate::advisor::hardware::{detect_hardware, format_hardware_profile, summarize_hardware};
use crate::advisor::model_memory::{estimate_model_memory, summarize_model_memory};
use crate::cli::parse_args;
use crate::config::ServingConfig;
use crate::engine::Mode;
use crate::engines::{adapter_for, EngineAdapter};
use crate::logs::classify_log;
use crate::params::{inspect_command, load_cached_or_hint, load_or_inspect};
use crate::results::{
    create_run_dir, write_best_config, write_config_file, write_report, ResultSet, TrialResult,
};
use crate::runner::{execute_run_plan, execute_server_plan};
use crate::serve::EngineArg;
use crate::terminal;
use crate::Result;

pub fn run(args: impl Iterator<Item = String>, mut out: impl Write) -> Result<()> {
    let setup = parse_args(args)?;
    match setup.mode {
        Mode::Plan => print_plan(&setup, &mut out),
        Mode::Params => print_params(&setup, &mut out),
        Mode::Hardware => print_hardware(&mut out),
        Mode::Serve => serve(&setup, &mut out),
        Mode::Bench => bench_once(&setup, &mut out),
        Mode::Sweep => run_sweep(&setup, &mut out),
        Mode::Advise => advise(&setup, &mut out),
    }
}

fn print_hardware(out: &mut impl Write) -> Result<()> {
    let profile = detect_hardware();
    write!(out, "{}", format_hardware_profile(&profile)).map_err(write_error)
}

fn print_plan(setup: &crate::cli::Setup, out: &mut impl Write) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let candidate = adapter.initial_candidate(setup);
    let config = ServingConfig::from_setup_and_candidate(setup, candidate);
    if setup.validate_params {
        let schema = if setup.execute {
            load_or_inspect(
                adapter,
                config.image.clone(),
                Path::new(&setup.param_cache_dir),
                setup.refresh_params,
            )?
        } else {
            load_cached_or_hint(
                adapter,
                config.image.clone(),
                Path::new(&setup.param_cache_dir),
            )?
        };
        schema.validate_args(&adapter.serving_args(&config))?;
    }
    let plan = adapter.run_plan(&config);
    writeln!(out, "engine: {}", config.engine).map_err(write_error)?;
    writeln!(out, "image: {}", config.image).map_err(write_error)?;
    writeln!(out, "model: {}", config.model).map_err(write_error)?;
    writeln!(out, "max_model_len: {}", config.max_model_len).map_err(write_error)?;
    writeln!(out, "metric: {}", config.metric).map_err(write_error)?;
    writeln!(
        out,
        "candidate: {}",
        adapter.describe_candidate(&config.candidate)
    )
    .map_err(write_error)?;
    writeln!(
        out,
        "benchmark: dataset={}, num_prompts={}, request_rate={}, max_concurrency={}, random_input_len={}, random_output_len={}",
        config.benchmark.dataset_name,
        config.benchmark.num_prompts,
        config.benchmark.request_rate,
        config
            .benchmark
            .max_concurrency
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unbounded".to_string()),
        config.benchmark.random_input_len,
        config.benchmark.random_output_len
    )
    .map_err(write_error)?;
    writeln!(out, "serve: {}", plan.server.shell()).map_err(write_error)?;
    writeln!(out, "bench: {}", plan.benchmark.shell()).map_err(write_error)?;
    Ok(())
}

fn print_params(setup: &crate::cli::Setup, out: &mut impl Write) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let image = image_for(setup);
    if !setup.execute {
        let command = inspect_command(adapter, image);
        writeln!(out, "inspect: {}", command.shell()).map_err(write_error)?;
        writeln!(out, "source: runtime only").map_err(write_error)?;
        writeln!(
            out,
            "add --execute to inspect the container and cache the schema"
        )
        .map_err(write_error)?;
        return Ok(());
    }

    writeln!(out, "inspecting image parameters: {image}").map_err(write_error)?;
    let schema = load_or_inspect(
        adapter,
        image,
        Path::new(&setup.param_cache_dir),
        setup.refresh_params,
    )?;
    writeln!(out, "source: runtime").map_err(write_error)?;
    writeln!(out, "parameters: {}", schema.parameters.len()).map_err(write_error)?;
    for spec in schema.parameters {
        writeln!(out, "{}\t{:?}", spec.cli, spec.kind).map_err(write_error)?;
    }
    Ok(())
}

fn serve(setup: &crate::cli::Setup, out: &mut impl Write) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let candidate = adapter.initial_candidate(setup);
    let config = ServingConfig::from_setup_and_candidate(setup, candidate);
    let plan = adapter.run_plan(&config);
    if !setup.execute {
        writeln!(out, "{}", plan.server.shell()).map_err(write_error)?;
        return Ok(());
    }

    ensure_hf_token()?;
    validate_serving_args(setup, &config)?;
    execute_server_plan(&plan, out)
}

fn bench_once(setup: &crate::cli::Setup, out: &mut impl Write) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let candidate = adapter.initial_candidate(setup);
    let config = ServingConfig::from_setup_and_candidate(setup, candidate);
    if !setup.execute {
        if setup.validate_params {
            validate_serving_configs_from_cache(setup, adapter, std::slice::from_ref(&config))?;
        }
        print_run_plans(adapter, &[config], out)?;
        return Ok(());
    }

    ensure_hf_token()?;
    let run_dir = create_run_dir(&setup.results_dir, "bench", setup.engine)?;
    let hardware = detect_hardware();
    terminal::info(
        out,
        "model-mem",
        format!(
            "estimating model={} max_model_len={}",
            config.model, config.max_model_len
        ),
    )?;
    let model_memory = estimate_model_memory(&config);
    terminal::info(out, "bench", describe_config(adapter, &config))?;
    terminal::info(out, "hardware", summarize_hardware(&hardware))?;
    terminal::info(out, "model-mem", summarize_model_memory(&model_memory))?;
    terminal::info(out, "results", format!("dir={}", run_dir.display()))?;
    validate_serving_args(setup, &config)?;
    let plan = adapter.run_plan(&config);
    let output = execute_run_plan(&plan, &mut *out)?;
    let result = TrialResult::new(
        config,
        setup.metric,
        hardware,
        model_memory,
        output.stdout,
        output.stderr,
    );
    let mut results = ResultSet::new(setup.metric);
    results.push(result);
    let report_file = write_report(&run_dir, "bench", &results)?;
    let best = results.best().ok_or("no benchmark results produced")?;
    let config_file = write_config_file(
        &run_dir,
        "config.conf",
        &executable_config_text(setup, adapter, &best.config),
    )?;
    terminal::ok(out, "metrics", metrics_line(best))?;
    terminal::info(
        out,
        "saved",
        format!(
            "report={}, config={}",
            report_file.display(),
            config_file.display()
        ),
    )?;
    Ok(())
}

fn run_sweep(setup: &crate::cli::Setup, out: &mut impl Write) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let configs = benchmark_configs(setup, adapter);
    if !setup.execute {
        if setup.validate_params {
            validate_serving_configs_from_cache(setup, adapter, &configs)?;
        }
        print_run_plans(adapter, &configs, out)?;
        return Ok(());
    }
    ensure_hf_token()?;
    let mut results = ResultSet::new(setup.metric);
    let total = configs.len();
    let run_dir = create_run_dir(&setup.results_dir, "sweep", setup.engine)?;
    let hardware = detect_hardware();
    let first_config = configs
        .first()
        .ok_or("sweep produced no candidate configs")?;
    terminal::info(
        out,
        "model-mem",
        format!(
            "estimating model={} max_model_len={}",
            first_config.model, first_config.max_model_len
        ),
    )?;
    let model_memory = estimate_model_memory(first_config);
    terminal::info(
        out,
        "sweep",
        format!("{total} trial(s), optimizing {}", setup.metric),
    )?;
    terminal::info(out, "hardware", summarize_hardware(&hardware))?;
    terminal::info(out, "model-mem", summarize_model_memory(&model_memory))?;
    terminal::info(out, "results", format!("dir={}", run_dir.display()))?;

    for (index, config) in configs.into_iter().enumerate() {
        terminal::info(
            out,
            "trial",
            format!(
                "{}/{} remaining={} {}",
                index + 1,
                total,
                total.saturating_sub(index + 1),
                describe_config(adapter, &config)
            ),
        )?;
        validate_serving_args(setup, &config)?;
        let plan = adapter.run_plan(&config);
        let output = execute_run_plan(&plan, &mut *out)?;
        let result = TrialResult::new(
            config,
            setup.metric,
            hardware.clone(),
            model_memory.clone(),
            output.stdout,
            output.stderr,
        );
        terminal::ok(out, "metrics", metrics_line(&result))?;
        results.push(result);
    }

    results.sort_best_first();
    let best = results.best().ok_or("no benchmark results produced")?;
    let report_file = write_report(&run_dir, "sweep", &results)?;
    let best_config = write_best_config(
        &run_dir,
        &executable_config_text(setup, adapter, &best.config),
    )?;
    terminal::ok(
        out,
        "best",
        format!(
            "{}={} {}",
            setup.metric,
            best.winning_value()
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "unavailable".to_string()),
            describe_config(adapter, &best.config)
        ),
    )?;
    terminal::info(
        out,
        "saved",
        format!(
            "report={}, best_config={}",
            report_file.display(),
            best_config.display(),
        ),
    )?;
    Ok(())
}

fn benchmark_configs(setup: &crate::cli::Setup, adapter: &dyn EngineAdapter) -> Vec<ServingConfig> {
    let base = adapter.initial_candidate(setup);
    let serve_sweeps = setup.serve_sweep.combinations();
    let mut configs = Vec::new();

    for candidate in setup.sweep.candidates(&base, setup.gpus).into_iter() {
        for serve_args in &serve_sweeps {
            let mut config = ServingConfig::from_setup_and_candidate(setup, candidate.clone());
            config.serve_args.extend(serve_args.clone());
            configs.push(config);
        }
    }

    configs
}

fn print_run_plans(
    adapter: &dyn EngineAdapter,
    configs: &[ServingConfig],
    out: &mut impl Write,
) -> Result<()> {
    for (index, config) in configs.iter().enumerate() {
        let plan = adapter.run_plan(config);
        if configs.len() > 1 {
            writeln!(
                out,
                "trial: {}/{} config: {}",
                index + 1,
                configs.len(),
                describe_config(adapter, config)
            )
            .map_err(write_error)?;
        }
        writeln!(out, "server: {}", plan.server.shell()).map_err(write_error)?;
        writeln!(out, "benchmark: {}", plan.benchmark.shell()).map_err(write_error)?;
    }
    Ok(())
}

fn describe_config(adapter: &dyn EngineAdapter, config: &ServingConfig) -> String {
    format!(
        "server_args={}",
        describe_engine_args(&display_server_args(adapter, config))
    )
}

fn display_server_args(adapter: &dyn EngineAdapter, config: &ServingConfig) -> Vec<EngineArg> {
    adapter
        .serving_args(config)
        .into_iter()
        .filter(|arg| !is_identity_server_arg(&arg.name))
        .collect()
}

fn is_identity_server_arg(name: &str) -> bool {
    matches!(name, "--model" | "--model-path" | "--host" | "--port")
}

fn describe_engine_args(args: &[EngineArg]) -> String {
    args.iter()
        .map(|arg| match &arg.value {
            Some(value) => format!("{}={value}", arg.name),
            None => arg.name.clone(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn metrics_line(result: &TrialResult) -> String {
    let mut parts = Vec::new();
    if let Some(value) = result.winning_value() {
        parts.push(format!("{}={value:.4}", result.winning_metric));
    }
    for (label, value) in [
        ("tps", result.metrics.output_token_throughput),
        ("total_tps", result.metrics.total_token_throughput),
        ("input_tps", result.metrics.input_token_throughput),
        ("peak_tps", result.metrics.peak_output_token_throughput),
        ("ttft_ms", result.metrics.mean_ttft_ms),
        ("p99_ttft_ms", result.metrics.p99_ttft_ms),
        ("tpot_ms", result.metrics.mean_tpot_ms),
        ("p99_tpot_ms", result.metrics.p99_tpot_ms),
        ("itl_ms", result.metrics.mean_itl_ms),
        ("p99_itl_ms", result.metrics.p99_itl_ms),
        ("e2e_ms", result.metrics.mean_e2e_ms),
        ("p99_e2e_ms", result.metrics.p99_e2e_ms),
        ("req_s", result.metrics.request_throughput),
        ("failed", result.metrics.failed_requests),
    ] {
        let Some(value) = value else {
            continue;
        };
        let item = format!("{label}={value:.4}");
        if !parts.contains(&item) {
            parts.push(item);
        }
    }
    if parts.is_empty() {
        "unavailable".to_string()
    } else {
        parts.join(", ")
    }
}

fn executable_config_text(
    setup: &crate::cli::Setup,
    adapter: &dyn EngineAdapter,
    config: &ServingConfig,
) -> String {
    let mut text = format!(
        "# Generated by optimum-advisor from the winning trial.\nengine = {}\nimage = {}\nmodel = {}\ngpus = {}\nhost = {}\nport = {}\nstartup_timeout_secs = {}\nmax_model_len = {}\nmetric = {}\nresults_dir = {}\n\n[benchmark]\ndataset_name = {}\nnum_prompts = {}\nrequest_rate = {}\n",
        config.engine,
        config.image,
        config.model,
        config.gpus,
        config.host,
        config.port,
        config.startup_timeout_secs,
        config.max_model_len,
        config.metric,
        setup.results_dir,
        config.benchmark.dataset_name,
        config.benchmark.num_prompts,
        config.benchmark.request_rate,
    );
    if let Some(max_concurrency) = config.benchmark.max_concurrency {
        text.push_str(&format!("max_concurrency = {max_concurrency}\n"));
    }
    text.push_str(&format!(
        "random_input_len = {}\nrandom_output_len = {}\n\n[serve]\n",
        config.benchmark.random_input_len, config.benchmark.random_output_len
    ));
    for arg in adapter.serving_args(config) {
        let key = arg.name.trim_start_matches("--");
        match arg.value {
            Some(value) => text.push_str(&format!("{key} = {value}\n")),
            None => text.push_str(&format!("{key} = true\n")),
        }
    }
    text
}

fn advise(setup: &crate::cli::Setup, out: &mut impl Write) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let path = setup
        .log_file
        .as_ref()
        .ok_or("advise requires --log-file <path>")?;
    let log = fs::read_to_string(path).map_err(|err| format!("failed to read {path}: {err}"))?;
    let outcome = classify_log(&log);
    let current = adapter.initial_candidate(setup);
    let next = adapter.next_candidate(setup, &current, outcome);
    let config = ServingConfig::from_setup_and_candidate(setup, next);
    let plan = adapter.run_plan(&config);
    writeln!(out, "outcome: {:?}", outcome).map_err(write_error)?;
    writeln!(
        out,
        "next candidate: {}",
        adapter.describe_candidate(&config.candidate)
    )
    .map_err(write_error)?;
    writeln!(out, "serve: {}", plan.server.shell()).map_err(write_error)?;
    Ok(())
}

fn image_for(setup: &crate::cli::Setup) -> String {
    setup
        .image
        .clone()
        .unwrap_or_else(|| setup.engine.default_image().to_string())
}

fn validate_serving_args(setup: &crate::cli::Setup, config: &ServingConfig) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let schema = load_or_inspect(
        adapter,
        config.image.clone(),
        Path::new(&setup.param_cache_dir),
        setup.refresh_params,
    )?;
    let args = adapter.serving_args(config);
    match schema.validate_args(&args) {
        Ok(()) => Ok(()),
        Err(err) if !setup.refresh_params => {
            let schema = load_or_inspect(
                adapter,
                config.image.clone(),
                Path::new(&setup.param_cache_dir),
                true,
            )?;
            schema.validate_args(&args).map_err(|_| err)
        }
        Err(err) => Err(err),
    }
}

fn validate_serving_configs_from_cache(
    setup: &crate::cli::Setup,
    adapter: &dyn EngineAdapter,
    configs: &[ServingConfig],
) -> Result<()> {
    let Some(config) = configs.first() else {
        return Ok(());
    };
    let schema = load_cached_or_hint(
        adapter,
        config.image.clone(),
        Path::new(&setup.param_cache_dir),
    )?;
    for config in configs {
        schema.validate_args(&adapter.serving_args(config))?;
    }
    Ok(())
}

fn write_error(err: std::io::Error) -> String {
    format!("failed to write output: {err}")
}

fn ensure_hf_token() -> Result<()> {
    if std::env::var("HF_TOKEN")
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
    {
        Ok(())
    } else {
        Err("HF_TOKEN is required for executing serving/benchmark containers".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advisor::hardware::HardwareProfile;

    #[test]
    fn plan_writes_serve_and_bench_commands() {
        let mut out = Vec::new();
        run(
            [
                "plan",
                "--engine",
                "vllm",
                "--model",
                "meta-llama/Llama-3.1-8B-Instruct",
            ]
            .into_iter()
            .map(String::from),
            &mut out,
        )
        .unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("serve: docker run"));
        assert!(text.contains("bench: docker run"));
        assert!(text.contains("--entrypoint vllm"));
    }

    #[test]
    fn params_prints_inspection_command_without_execute() {
        let mut out = Vec::new();
        run(
            ["params", "--engine", "sglang"]
                .into_iter()
                .map(String::from),
            &mut out,
        )
        .unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("inspect: docker run"));
        assert!(text.contains("--entrypoint python3"));
    }

    #[test]
    fn hardware_profile_prints_gpu_shape() {
        let profile = HardwareProfile {
            source: "nvidia-smi".to_string(),
            cuda_visible_devices: Some("0,1".to_string()),
            gpus: vec![crate::advisor::hardware::GpuInfo {
                index: 0,
                name: "NVIDIA L4".to_string(),
                uuid: "GPU-abc".to_string(),
                compute_capability: Some("8.9".to_string()),
                memory_total_mib: 23034,
                memory_free_mib: 22000,
                memory_used_mib: 1034,
            }],
            warnings: vec!["test warning".to_string()],
        };
        let mut out = Vec::new();

        write!(out, "{}", format_hardware_profile(&profile)).unwrap();

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("source: nvidia-smi"));
        assert!(text.contains("cuda_visible_devices: 0,1"));
        assert!(text.contains("gpus: 1"));
        assert!(text.contains("compute_capability=8.9"));
        assert!(text.contains("memory_total_mib=23034"));
        assert!(text.contains("warning: test warning"));
    }

    #[test]
    fn config_description_uses_effective_server_args() {
        let mut setup = crate::cli::Setup::default_for_mode(Mode::Bench);
        setup.model = "m".to_string();
        setup.gpus = 2;
        setup
            .serve_args
            .push(EngineArg::value("tensor-parallel-size", "2"));
        setup
            .serve_args
            .push(EngineArg::value("gpu-memory-utilization", "0.80"));
        let adapter = adapter_for(setup.engine);
        let candidate = adapter.initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);

        let description = describe_config(adapter, &config);

        assert!(description.contains("--tensor-parallel-size=2"));
        assert!(description.contains("--gpu-memory-utilization=0.80"));
        assert!(!description.contains("tp=1"));
        assert!(!description.contains("gpu_memory_utilization=0.90"));
    }

    #[test]
    fn best_config_records_effective_winning_config() {
        let mut setup = crate::cli::Setup::default_for_mode(Mode::Bench);
        setup.model = "m".to_string();
        setup.gpus = 2;
        setup.benchmark.num_prompts = 4;
        setup
            .serve_args
            .push(EngineArg::value("tensor-parallel-size", "2"));
        setup
            .serve_args
            .push(EngineArg::value("gpu-memory-utilization", "0.80"));
        let adapter = adapter_for(setup.engine);
        let candidate = adapter.initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);

        let text = executable_config_text(&setup, adapter, &config);

        assert!(text.contains("engine = vllm"));
        assert!(text.contains("model = m"));
        assert!(text.contains("[benchmark]"));
        assert!(text.contains("num_prompts = 4"));
        assert!(text.contains("[serve]"));
        assert!(text.contains("tensor-parallel-size = 2"));
        assert!(text.contains("gpu-memory-utilization = 0.80"));
    }
}
