use crate::config::BenchmarkConfig;
use crate::engine::{Engine, Metric, Mode};
use crate::serve::EngineArg;
use crate::trial::Candidate;
use crate::Result;

#[derive(Debug)]
pub struct Setup {
    pub mode: Mode,
    pub engine: Engine,
    pub model: String,
    pub image: Option<String>,
    pub gpus: usize,
    pub host: String,
    pub port: u16,
    pub startup_timeout_secs: u64,
    pub max_model_len: u32,
    pub param_cache_dir: String,
    pub refresh_params: bool,
    pub validate_params: bool,
    pub metric: Metric,
    pub execute: bool,
    pub log_file: Option<String>,
    pub candidate: Candidate,
    pub serve_args: Vec<EngineArg>,
    pub benchmark: BenchmarkConfig,
}

pub fn parse_args(args: impl Iterator<Item = String>) -> Result<Setup> {
    let mut args = args.peekable();
    let mode = match args.next().as_deref() {
        Some("plan") => Mode::Plan,
        Some("params") => Mode::Params,
        Some("serve") => Mode::Serve,
        Some("run") => Mode::Run,
        Some("advise") => Mode::Advise,
        Some("-h" | "--help") | None => return Err(usage()),
        Some(other) => return Err(format!("unknown command: {other}\n\n{}", usage())),
    };

    let mut setup = Setup {
        mode,
        engine: Engine::Vllm,
        model: String::new(),
        image: None,
        gpus: 1,
        host: "127.0.0.1".to_string(),
        port: 8000,
        startup_timeout_secs: 300,
        max_model_len: 8192,
        param_cache_dir: ".optimum-advisor/params".to_string(),
        refresh_params: false,
        validate_params: false,
        metric: Metric::Tps,
        execute: false,
        log_file: None,
        candidate: Candidate::default(),
        serve_args: Vec::new(),
        benchmark: BenchmarkConfig::default(),
    };

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--engine" => setup.engine = Engine::parse(&take_value(&mut args, "--engine")?)?,
            "--model" => setup.model = take_value(&mut args, "--model")?,
            "--image" => setup.image = Some(take_value(&mut args, "--image")?),
            "--gpus" => setup.gpus = parse_value(&take_value(&mut args, "--gpus")?, "--gpus")?,
            "--host" => setup.host = take_value(&mut args, "--host")?,
            "--port" => setup.port = parse_value(&take_value(&mut args, "--port")?, "--port")?,
            "--startup-timeout-secs" => {
                setup.startup_timeout_secs = parse_value(
                    &take_value(&mut args, "--startup-timeout-secs")?,
                    "--startup-timeout-secs",
                )?;
            }
            "--max-model-len" => {
                setup.max_model_len = parse_value(
                    &take_value(&mut args, "--max-model-len")?,
                    "--max-model-len",
                )?;
            }
            "--param-cache-dir" => {
                setup.param_cache_dir = take_value(&mut args, "--param-cache-dir")?
            }
            "--refresh-params" => setup.refresh_params = true,
            "--validate-params" => setup.validate_params = true,
            "--metric" => setup.metric = Metric::parse(&take_value(&mut args, "--metric")?)?,
            "--execute" => setup.execute = true,
            "--log-file" => setup.log_file = Some(take_value(&mut args, "--log-file")?),
            "--serve-arg" => setup.serve_args.push(EngineArg::assignment(&take_value(
                &mut args,
                "--serve-arg",
            )?)?),
            "--serve-flag" => setup
                .serve_args
                .push(EngineArg::flag(&take_value(&mut args, "--serve-flag")?)),
            "--dataset-name" => {
                setup.benchmark.dataset_name = take_value(&mut args, "--dataset-name")?
            }
            "--num-prompts" => {
                setup.benchmark.num_prompts =
                    parse_value(&take_value(&mut args, "--num-prompts")?, "--num-prompts")?;
            }
            "--request-rate" => {
                setup.benchmark.request_rate = take_value(&mut args, "--request-rate")?
            }
            "--benchmark-max-concurrency" => {
                setup.benchmark.max_concurrency = Some(parse_value(
                    &take_value(&mut args, "--benchmark-max-concurrency")?,
                    "--benchmark-max-concurrency",
                )?);
            }
            "--tp" => {
                setup.candidate.parallelism.tensor =
                    parse_value(&take_value(&mut args, "--tp")?, "--tp")?;
            }
            "--memory-fraction" => {
                setup.candidate.memory.fraction = parse_value(
                    &take_value(&mut args, "--memory-fraction")?,
                    "--memory-fraction",
                )?;
            }
            "--prefill-token-budget" => {
                setup.candidate.scheduler.prefill_token_budget = parse_value(
                    &take_value(&mut args, "--prefill-token-budget")?,
                    "--prefill-token-budget",
                )?;
            }
            "--max-concurrency" => {
                setup.candidate.scheduler.max_running_requests = parse_value(
                    &take_value(&mut args, "--max-concurrency")?,
                    "--max-concurrency",
                )?;
            }
            "--max-num-batched-tokens" => {
                setup.candidate.scheduler.prefill_token_budget = parse_value(
                    &take_value(&mut args, "--max-num-batched-tokens")?,
                    "--max-num-batched-tokens",
                )?;
            }
            "--gpu-memory-utilization" => {
                setup.candidate.memory.fraction = parse_value(
                    &take_value(&mut args, "--gpu-memory-utilization")?,
                    "--gpu-memory-utilization",
                )?;
            }
            "--chunked-prefill-size" => {
                setup.candidate.scheduler.prefill_token_budget = parse_value(
                    &take_value(&mut args, "--chunked-prefill-size")?,
                    "--chunked-prefill-size",
                )?;
            }
            "--mem-fraction-static" => {
                setup.candidate.memory.fraction = parse_value(
                    &take_value(&mut args, "--mem-fraction-static")?,
                    "--mem-fraction-static",
                )?;
            }
            "--max-running-requests" => {
                setup.candidate.scheduler.max_running_requests = parse_value(
                    &take_value(&mut args, "--max-running-requests")?,
                    "--max-running-requests",
                )?;
            }
            "-h" | "--help" => return Err(usage()),
            unknown if unknown.starts_with("--") => {
                if let Some(value) = args.next_if(|value| !value.starts_with("--")) {
                    setup.serve_args.push(EngineArg::value(unknown, value));
                } else {
                    setup.serve_args.push(EngineArg::flag(unknown));
                }
            }
            _ => return Err(format!("unknown argument: {arg}")),
        }
    }

    if setup.model.is_empty() && setup.mode != Mode::Params {
        return Err("--model is required".to_string());
    }
    if setup.gpus == 0 {
        return Err("--gpus must be greater than zero".to_string());
    }
    setup.candidate.clamp_to_gpus(setup.gpus);
    Ok(setup)
}

fn take_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_value<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| format!("{flag} has invalid value: {value}"))
}

fn usage() -> String {
    "usage:
  optimum-advisor plan --engine vllm|sglang --model MODEL [--gpus N] [--max-model-len N] [--metric ttft|tps|itl]
  optimum-advisor params --engine vllm|sglang [--image IMAGE] [--execute]
  optimum-advisor serve --engine vllm|sglang --model MODEL [--gpus N] [--serve-arg NAME=VALUE] [--execute]
  optimum-advisor run --engine vllm|sglang --model MODEL [--gpus N] [--max-model-len N] [--num-prompts N] [--request-rate R] --execute
  optimum-advisor advise --engine vllm|sglang --model MODEL --log-file PATH [--gpus N] [--tp N]"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_model() {
        let err = parse_args(["plan"].into_iter().map(String::from)).unwrap_err();
        assert_eq!(err, "--model is required");
    }

    #[test]
    fn clamps_tp_to_available_gpus() {
        let setup = parse_args(
            [
                "plan", "--engine", "vllm", "--model", "m", "--gpus", "2", "--tp", "4",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(setup.candidate.parallelism.tensor, 2);
    }

    #[test]
    fn accepts_version_specific_serve_args() {
        let setup = parse_args(
            [
                "plan",
                "--engine",
                "vllm",
                "--model",
                "m",
                "--serve-arg",
                "kv-cache-dtype=fp8",
                "--serve-flag",
                "disable-log-stats",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();
        assert_eq!(setup.serve_args.len(), 2);
    }

    #[test]
    fn unknown_long_flags_become_engine_args() {
        let setup = parse_args(
            [
                "plan",
                "--engine",
                "vllm",
                "--model",
                "m",
                "--kv-cache-dtype",
                "fp8",
                "--disable-log-stats",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(setup.serve_args[0].name, "--kv-cache-dtype");
        assert_eq!(setup.serve_args[0].value.as_deref(), Some("fp8"));
        assert_eq!(setup.serve_args[1].name, "--disable-log-stats");
        assert_eq!(setup.serve_args[1].value, None);
    }

    #[test]
    fn params_does_not_require_model() {
        let setup = parse_args(
            ["params", "--engine", "sglang"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert_eq!(setup.engine, Engine::Sglang);
    }
}
