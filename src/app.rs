use std::fs;
use std::io::Write;
use std::path::Path;

use crate::cli::parse_args;
use crate::config::ServingConfig;
use crate::engine::Mode;
use crate::engines::adapter_for;
use crate::logs::classify_log;
use crate::params::{inspect_command, load_cached_or_hint, load_or_inspect};
use crate::runner::{execute_run_plan, execute_server_plan};
use crate::Result;

pub fn run(args: impl Iterator<Item = String>, mut out: impl Write) -> Result<()> {
    let setup = parse_args(args)?;
    match setup.mode {
        Mode::Plan => print_plan(&setup, &mut out),
        Mode::Params => print_params(&setup, &mut out),
        Mode::Serve => serve(&setup, &mut out),
        Mode::Run => run_benchmark(&setup, &mut out),
        Mode::Advise => advise(&setup, &mut out),
    }
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
            "run with --execute to inspect the container and cache the schema"
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

fn run_benchmark(setup: &crate::cli::Setup, out: &mut impl Write) -> Result<()> {
    let adapter = adapter_for(setup.engine);
    let candidate = adapter.initial_candidate(setup);
    let config = ServingConfig::from_setup_and_candidate(setup, candidate);
    let plan = adapter.run_plan(&config);
    if !setup.execute {
        writeln!(out, "server: {}", plan.server.shell()).map_err(write_error)?;
        writeln!(out, "benchmark: {}", plan.benchmark.shell()).map_err(write_error)?;
        return Ok(());
    }
    ensure_hf_token()?;
    validate_serving_args(setup, &config)?;
    execute_run_plan(&plan, out)
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
}
