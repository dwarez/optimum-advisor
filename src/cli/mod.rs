use crate::config::BenchmarkConfig;
use crate::engine::{Engine, Metric, Mode};
use crate::serve::{EngineArg, ServingParamSweep};
use crate::trial::{Candidate, CandidateSweep};
use crate::Result;

mod config_file;

use config_file::{
    apply_config_file, parse_memory_fraction_list, parse_u32_list, parse_usize_list,
};

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
    pub results_dir: String,
    pub metric: Metric,
    pub execute: bool,
    pub log_file: Option<String>,
    pub candidate: Candidate,
    pub sweep: CandidateSweep,
    pub serve_sweep: ServingParamSweep,
    pub serve_args: Vec<EngineArg>,
    pub benchmark: BenchmarkConfig,
}

impl Setup {
    pub fn default_for_mode(mode: Mode) -> Self {
        Self {
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
            results_dir: ".optimum-advisor/results".to_string(),
            metric: Metric::Tps,
            execute: false,
            log_file: None,
            candidate: Candidate::default(),
            sweep: CandidateSweep::default(),
            serve_sweep: ServingParamSweep::default(),
            serve_args: Vec::new(),
            benchmark: BenchmarkConfig::default(),
        }
    }
}

pub fn parse_args(args: impl Iterator<Item = String>) -> Result<Setup> {
    let mut args = args.peekable();
    let mode = match args.next().as_deref() {
        Some("plan") => Mode::Plan,
        Some("params") => Mode::Params,
        Some("serve") => Mode::Serve,
        Some("bench") => Mode::Bench,
        Some("sweep") => Mode::Sweep,
        Some("advise") => Mode::Advise,
        Some("-h" | "--help") | None => return Err(usage()),
        Some(other) => return Err(format!("unknown command: {other}\n\n{}", usage())),
    };

    let mut setup = Setup::default_for_mode(mode);
    if matches!(mode, Mode::Bench | Mode::Sweep) {
        setup.execute = true;
    }

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
            "--results-dir" => setup.results_dir = take_value(&mut args, "--results-dir")?,
            "--config" => apply_config_file(&mut setup, take_value(&mut args, "--config")?)?,
            "--sweep-file" => {
                apply_config_file(&mut setup, take_value(&mut args, "--sweep-file")?)?;
            }
            "--metric" => setup.metric = Metric::parse(&take_value(&mut args, "--metric")?)?,
            "--execute" => setup.execute = true,
            "--dry-run" => setup.execute = false,
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
            "--random-input-len" => {
                setup.benchmark.random_input_len = parse_value(
                    &take_value(&mut args, "--random-input-len")?,
                    "--random-input-len",
                )?;
            }
            "--random-output-len" => {
                setup.benchmark.random_output_len = parse_value(
                    &take_value(&mut args, "--random-output-len")?,
                    "--random-output-len",
                )?;
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
            "--sweep-tp" => {
                setup.sweep.tensor_parallelism =
                    parse_usize_list(&take_value(&mut args, "--sweep-tp")?, "--sweep-tp")?;
            }
            "--sweep-memory-fraction" => {
                setup.sweep.memory_fraction = parse_memory_fraction_list(
                    &take_value(&mut args, "--sweep-memory-fraction")?,
                    "--sweep-memory-fraction",
                )?;
            }
            "--sweep-prefill-token-budget" => {
                setup.sweep.prefill_token_budget = parse_u32_list(
                    &take_value(&mut args, "--sweep-prefill-token-budget")?,
                    "--sweep-prefill-token-budget",
                )?;
            }
            "--sweep-max-running-requests" => {
                setup.sweep.max_running_requests = parse_u32_list(
                    &take_value(&mut args, "--sweep-max-running-requests")?,
                    "--sweep-max-running-requests",
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
    if setup.mode == Mode::Bench && has_sweep(&setup) {
        return Err(
            "bench accepts one configuration; use sweep for [sweep] or --sweep-*".to_string(),
        );
    }
    if setup.mode == Mode::Sweep && !has_sweep(&setup) {
        return Err(
            "sweep requires [sweep] or --sweep-*; use bench for one configuration".to_string(),
        );
    }
    setup.candidate.clamp_to_gpus(setup.gpus);
    Ok(setup)
}

fn has_sweep(setup: &Setup) -> bool {
    !setup.sweep.tensor_parallelism.is_empty()
        || !setup.sweep.memory_fraction.is_empty()
        || !setup.sweep.prefill_token_budget.is_empty()
        || !setup.sweep.max_running_requests.is_empty()
        || !setup.serve_sweep.parameters.is_empty()
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
  optimum-advisor plan --engine vllm|sglang --model MODEL [--gpus N] [--max-model-len N] [--metric tps|total_tps|req_s|ttft|p99_ttft|tpot|p99_tpot|itl|p99_itl|e2e|p99_e2e]
  optimum-advisor params --engine vllm|sglang [--image IMAGE] [--execute]
  optimum-advisor serve --engine vllm|sglang --model MODEL [--gpus N] [--serve-arg NAME=VALUE] [--execute]
  optimum-advisor sweep --config PATH [--dry-run]
  optimum-advisor bench --config PATH [--dry-run]
  optimum-advisor bench --engine vllm|sglang --model MODEL [--gpus N] [--metric tps|total_tps|req_s|ttft|p99_ttft|tpot|p99_tpot|itl|p99_itl|e2e|p99_e2e] [--results-dir DIR] [--num-prompts N] [--request-rate R] [--dry-run]
  optimum-advisor advise --engine vllm|sglang --model MODEL --log-file PATH [--gpus N] [--tp N]"
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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

    #[test]
    fn accepts_results_dir() {
        let setup = parse_args(
            [
                "bench",
                "--engine",
                "vllm",
                "--model",
                "m",
                "--results-dir",
                "results",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(setup.results_dir, "results");
    }

    #[test]
    fn accepts_sweep_lists() {
        let setup = parse_args(
            [
                "sweep",
                "--engine",
                "vllm",
                "--model",
                "m",
                "--sweep-tp",
                "1,2",
                "--sweep-memory-fraction",
                "0.8,0.9",
                "--sweep-prefill-token-budget",
                "2048,8192",
                "--sweep-max-running-requests",
                "64,128",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(setup.sweep.tensor_parallelism, vec![1, 2]);
        assert_eq!(setup.sweep.memory_fraction, vec![0.8, 0.9]);
        assert_eq!(setup.sweep.prefill_token_budget, vec![2048, 8192]);
        assert_eq!(setup.sweep.max_running_requests, vec![64, 128]);
    }

    #[test]
    fn accepts_config_file() {
        let path = std::env::temp_dir().join(format!(
            "optimum-advisor-config-{}.conf",
            std::process::id()
        ));
        fs::write(
            &path,
            "engine = vllm\nmodel = m\ngpus = 2\nmetric = ttft\n[benchmark]\nnum_prompts = 4\n[sweep]\ntensor-parallel-size = 1,2\ngpu-memory-utilization = 0.8,0.9\n",
        )
        .unwrap();

        let setup = parse_args(
            [
                "sweep",
                "--config",
                path.to_str().unwrap(),
                "--sweep-memory-fraction",
                "0.7",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap();

        assert_eq!(setup.engine, Engine::Vllm);
        assert_eq!(setup.model, "m");
        assert_eq!(setup.gpus, 2);
        assert_eq!(setup.metric, Metric::Ttft);
        assert_eq!(setup.benchmark.num_prompts, 4);
        assert_eq!(setup.serve_sweep.parameters.len(), 2);
        assert_eq!(setup.sweep.memory_fraction, vec![0.7]);
    }

    #[test]
    fn bench_rejects_sweep_parameters() {
        let err = parse_args(
            [
                "bench",
                "--engine",
                "vllm",
                "--model",
                "m",
                "--sweep-tp",
                "1,2",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap_err();

        assert!(err.contains("bench accepts one configuration"));
    }

    #[test]
    fn bench_executes_by_default_but_supports_dry_run() {
        let setup = parse_args(
            ["bench", "--engine", "vllm", "--model", "m"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(setup.execute);

        let setup = parse_args(
            ["bench", "--engine", "vllm", "--model", "m", "--dry-run"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(!setup.execute);
    }

    #[test]
    fn sweep_executes_by_default_but_supports_dry_run() {
        let path =
            std::env::temp_dir().join(format!("optimum-advisor-sweep-{}.conf", std::process::id()));
        fs::write(
            &path,
            "engine = vllm\nmodel = m\n[sweep]\ntensor-parallel-size = 1,2\n",
        )
        .unwrap();

        let setup = parse_args(
            ["sweep", "--config", path.to_str().unwrap()]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(setup.execute);

        let setup = parse_args(
            ["sweep", "--config", path.to_str().unwrap(), "--dry-run"]
                .into_iter()
                .map(String::from),
        )
        .unwrap();
        assert!(!setup.execute);
    }

    #[test]
    fn sweep_rejects_single_run_configs() {
        let err = parse_args(
            ["sweep", "--engine", "vllm", "--model", "m"]
                .into_iter()
                .map(String::from),
        )
        .unwrap_err();

        assert!(err.contains("sweep requires"));
    }
}
