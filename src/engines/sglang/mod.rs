use crate::cli::Setup;
use crate::config::ServingConfig;
use crate::engine::{Engine, Metric};
use crate::logs::Outcome;
use crate::runner::{ProcessSpec, RunPlan};
use crate::serve::EngineArg;
use crate::trial::{next_tensor_parallelism, Candidate};

use super::{append_engine_args, docker_server_args, readiness, EngineAdapter};

pub(super) static SGLANG: SglangAdapter = SglangAdapter;

pub(super) struct SglangAdapter;

impl EngineAdapter for SglangAdapter {
    fn engine(&self) -> Engine {
        Engine::Sglang
    }

    fn help_command(&self, image: String) -> ProcessSpec {
        ProcessSpec::new(
            "docker",
            vec![
                "run".to_string(),
                "--rm".to_string(),
                "--gpus".to_string(),
                "all".to_string(),
                "--entrypoint".to_string(),
                "python3".to_string(),
                image,
                "-m".to_string(),
                "sglang.launch_server".to_string(),
                "--help".to_string(),
            ],
        )
    }

    fn initial_candidate(&self, setup: &Setup) -> Candidate {
        let mut candidate = setup.candidate.clone();
        candidate.memory.fraction = 0.88;
        candidate.scheduler.prefill_token_budget = match setup.metric {
            Metric::Itl => 2_048,
            Metric::Ttft | Metric::Tps => 8_192,
        };
        candidate.clamp_to_gpus(setup.gpus);
        candidate
    }

    fn next_candidate(&self, setup: &Setup, last: &Candidate, outcome: Outcome) -> Candidate {
        let mut next = last.clone();
        match outcome {
            Outcome::Oom => {
                if let Some(tp) = next_tensor_parallelism(last.parallelism.tensor, setup.gpus) {
                    next.parallelism.tensor = tp;
                } else {
                    next.scheduler.prefill_token_budget =
                        (next.scheduler.prefill_token_budget / 2).max(1024);
                    next.memory.fraction = (next.memory.fraction - 0.05).max(0.70);
                }
            }
            Outcome::KvPressure => {
                if next.memory.fraction < 0.95 {
                    next.memory.fraction = (next.memory.fraction + 0.03).min(0.95);
                } else {
                    next.scheduler.max_running_requests =
                        (next.scheduler.max_running_requests / 2).max(1);
                }
            }
            Outcome::Ready | Outcome::Unknown => {}
        }
        next
    }

    fn describe_candidate(&self, candidate: &Candidate) -> String {
        format!(
            "tp={}, mem_fraction_static={:.2}, chunked_prefill_size={}, max_running_requests={}",
            candidate.parallelism.tensor,
            candidate.memory.fraction,
            candidate.scheduler.prefill_token_budget,
            candidate.scheduler.max_running_requests
        )
    }

    fn serving_args(&self, config: &ServingConfig) -> Vec<EngineArg> {
        let mut args = vec![
            EngineArg::value("--model-path", config.model.clone()),
            EngineArg::value("--host", "0.0.0.0"),
            EngineArg::value("--port", "30000"),
            EngineArg::value("--tp-size", config.candidate.parallelism.tensor.to_string()),
            EngineArg::value(
                "--mem-fraction-static",
                format!("{:.2}", config.candidate.memory.fraction),
            ),
            EngineArg::value(
                "--chunked-prefill-size",
                config.candidate.scheduler.prefill_token_budget.to_string(),
            ),
            EngineArg::value(
                "--max-running-requests",
                config.candidate.scheduler.max_running_requests.to_string(),
            ),
        ];
        args.extend(config.serve_args.clone());
        args
    }

    fn run_plan(&self, config: &ServingConfig) -> RunPlan {
        let mut server_args = docker_server_args(config, "30000");
        server_args.extend([
            "python3".to_string(),
            "-m".to_string(),
            "sglang.launch_server".to_string(),
        ]);
        append_engine_args(&mut server_args, self.serving_args(config));

        RunPlan {
            server: ProcessSpec::new("docker", server_args),
            benchmark: ProcessSpec::new(
                "python3",
                vec![
                    "-m".to_string(),
                    "sglang.bench_serving".to_string(),
                    "--backend".to_string(),
                    "sglang".to_string(),
                    "--model".to_string(),
                    config.model.clone(),
                    "--dataset-name".to_string(),
                    config.benchmark.dataset_name.clone(),
                    "--num-prompts".to_string(),
                    config.benchmark.num_prompts.to_string(),
                ],
            ),
            readiness: readiness(config, config.port),
        }
    }
}
