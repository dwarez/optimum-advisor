use std::fs;
use std::path::Path;

use crate::engine::{Engine, Metric};
use crate::serve::{EngineArg, ServingParamSweep};
use crate::Result;

use super::Setup;

pub fn apply_config_file(setup: &mut Setup, path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read config file {}: {err}", path.display()))?;
    parse_config_text(setup, &text)
}

pub fn parse_usize_list(value: &str, label: &str) -> Result<Vec<usize>> {
    let values = parse_list(value, label)?;
    if values.iter().all(|value| *value > 0) {
        Ok(values)
    } else {
        Err(format!("{label} values must be greater than zero"))
    }
}

pub fn parse_u32_list(value: &str, label: &str) -> Result<Vec<u32>> {
    let values = parse_list(value, label)?;
    if values.iter().all(|value| *value > 0) {
        Ok(values)
    } else {
        Err(format!("{label} values must be greater than zero"))
    }
}

pub fn parse_memory_fraction_list(value: &str, label: &str) -> Result<Vec<f32>> {
    let values: Vec<f32> = parse_list(value, label)?;
    if values
        .iter()
        .all(|value| value.is_finite() && *value > 0.0 && *value <= 1.0)
    {
        Ok(values)
    } else {
        Err(format!("{label} values must be in (0, 1]"))
    }
}

fn parse_config_text(setup: &mut Setup, text: &str) -> Result<()> {
    let mut section = Section::Run;
    for (index, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            section = Section::parse(name.trim(), index + 1)?;
            continue;
        }

        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("invalid config line {}: expected key=value", index + 1))?;
        let key = normalize_key(key);
        let value = clean_value(value);
        match section {
            Section::Run => apply_run_key(setup, &key, &value, index + 1)?,
            Section::Benchmark => apply_benchmark_key(setup, &key, &value, index + 1)?,
            Section::Serve => apply_serve_key(setup, &key, &value)?,
            Section::Sweep => apply_sweep_key(&mut setup.serve_sweep, &key, &value)?,
        }
    }
    Ok(())
}

fn apply_run_key(setup: &mut Setup, key: &str, value: &str, line: usize) -> Result<()> {
    match key {
        "engine" => setup.engine = Engine::parse(value)?,
        "model" => setup.model = value.to_string(),
        "image" => setup.image = Some(value.to_string()),
        "gpus" => setup.gpus = parse_value(value, key)?,
        "host" => setup.host = value.to_string(),
        "port" => setup.port = parse_value(value, key)?,
        "startup_timeout_secs" => setup.startup_timeout_secs = parse_value(value, key)?,
        "max_model_len" => setup.max_model_len = parse_value(value, key)?,
        "param_cache_dir" => setup.param_cache_dir = value.to_string(),
        "refresh_params" => setup.refresh_params = parse_bool(value, key)?,
        "validate_params" => setup.validate_params = parse_bool(value, key)?,
        "results_dir" => setup.results_dir = value.to_string(),
        "metric" => setup.metric = Metric::parse(value)?,
        "execute" => setup.execute = parse_bool(value, key)?,
        "tp" => setup.candidate.parallelism.tensor = parse_value(value, key)?,
        "memory_fraction" => setup.candidate.memory.fraction = parse_value(value, key)?,
        "prefill_token_budget" => {
            setup.candidate.scheduler.prefill_token_budget = parse_value(value, key)?;
        }
        "max_running_requests" | "max_concurrency" => {
            setup.candidate.scheduler.max_running_requests = parse_value(value, key)?;
        }
        _ => return Err(format!("unknown run config key '{key}' on line {line}")),
    }
    Ok(())
}

fn apply_benchmark_key(setup: &mut Setup, key: &str, value: &str, line: usize) -> Result<()> {
    match key {
        "dataset_name" => setup.benchmark.dataset_name = value.to_string(),
        "num_prompts" => setup.benchmark.num_prompts = parse_value(value, key)?,
        "request_rate" => setup.benchmark.request_rate = value.to_string(),
        "max_concurrency" | "benchmark_max_concurrency" => {
            setup.benchmark.max_concurrency = Some(parse_value(value, key)?);
        }
        "random_input_len" => setup.benchmark.random_input_len = parse_value(value, key)?,
        "random_output_len" => setup.benchmark.random_output_len = parse_value(value, key)?,
        _ => {
            return Err(format!(
                "unknown benchmark config key '{key}' on line {line}"
            ))
        }
    }
    Ok(())
}

fn apply_serve_key(setup: &mut Setup, key: &str, value: &str) -> Result<()> {
    match parse_optional_engine_arg(key, value)? {
        Some(arg) => setup.serve_args.push(arg),
        None => {}
    }
    Ok(())
}

fn apply_sweep_key(sweep: &mut ServingParamSweep, key: &str, value: &str) -> Result<()> {
    let values = parse_string_list(value, key)?;
    sweep.push(engine_param_name(key), values);
    Ok(())
}

fn parse_optional_engine_arg(key: &str, value: &str) -> Result<Option<EngineArg>> {
    match value {
        "true" => Ok(Some(EngineArg::flag(&engine_param_name(key)))),
        "false" => Ok(None),
        _ => Ok(Some(EngineArg::value(engine_param_name(key), value))),
    }
}

fn parse_string_list(value: &str, label: &str) -> Result<Vec<String>> {
    let values = value
        .split(',')
        .map(clean_value)
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    if values.is_empty() {
        Err(format!("{label} requires at least one value"))
    } else {
        Ok(values)
    }
}

fn parse_list<T: std::str::FromStr>(value: &str, label: &str) -> Result<Vec<T>> {
    parse_string_list(value, label)?
        .into_iter()
        .map(|item| {
            item.parse()
                .map_err(|_| format!("{label} has invalid value: {item}"))
        })
        .collect()
}

fn parse_value<T: std::str::FromStr>(value: &str, label: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| format!("{label} has invalid value: {value}"))
}

fn parse_bool(value: &str, label: &str) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{label} has invalid boolean value: {value}")),
    }
}

fn normalize_key(key: &str) -> String {
    key.trim().trim_start_matches('-').replace('-', "_")
}

fn engine_param_name(key: &str) -> String {
    key.trim().trim_start_matches('-').replace('_', "-")
}

fn clean_value(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2
        && ((value.starts_with('"') && value.ends_with('"'))
            || (value.starts_with('\'') && value.ends_with('\'')))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

#[derive(Clone, Copy)]
enum Section {
    Run,
    Benchmark,
    Serve,
    Sweep,
}

impl Section {
    fn parse(value: &str, line: usize) -> Result<Self> {
        match normalize_key(value).as_str() {
            "run" => Ok(Self::Run),
            "benchmark" => Ok(Self::Benchmark),
            "serve" | "serving" => Ok(Self::Serve),
            "sweep" => Ok(Self::Sweep),
            _ => Err(format!("unknown config section [{value}] on line {line}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Mode;

    fn setup() -> Setup {
        Setup::default_for_mode(Mode::Run)
    }

    #[test]
    fn parses_full_run_config() {
        let mut setup = setup();
        parse_config_text(
            &mut setup,
            "
            engine = vllm
            model = Qwen/Qwen3-4B-Instruct-2507
            gpus = 2
            metric = ttft

            [benchmark]
            num_prompts = 4
            request_rate = inf

            [serve]
            kv-cache-dtype = fp8
            disable-log-stats = true

            [sweep]
            tensor-parallel-size = 1,2
            gpu-memory-utilization = 0.8,0.9
            ",
        )
        .unwrap();

        assert_eq!(setup.engine, Engine::Vllm);
        assert_eq!(setup.model, "Qwen/Qwen3-4B-Instruct-2507");
        assert_eq!(setup.gpus, 2);
        assert_eq!(setup.metric, Metric::Ttft);
        assert_eq!(setup.benchmark.num_prompts, 4);
        assert_eq!(setup.benchmark.request_rate, "inf");
        assert_eq!(
            setup.serve_args,
            vec![
                EngineArg::value("kv-cache-dtype", "fp8"),
                EngineArg::flag("disable-log-stats")
            ]
        );
        assert_eq!(setup.serve_sweep.parameters.len(), 2);
    }

    #[test]
    fn old_abstract_sweep_lists_still_parse_for_cli_overrides() {
        assert_eq!(parse_usize_list("1,2", "tp").unwrap(), vec![1, 2]);
        assert_eq!(parse_u32_list("4,8", "tokens").unwrap(), vec![4, 8]);
        assert_eq!(
            parse_memory_fraction_list("0.8,0.9", "memory").unwrap(),
            vec![0.8, 0.9]
        );
    }

    #[test]
    fn rejects_unknown_config_keys() {
        let mut setup = setup();
        let err = parse_config_text(&mut setup, "wat = nope").unwrap_err();
        assert!(err.contains("unknown run config key"));
    }
}
