use crate::cli::Setup;
use crate::config::ServingConfig;
use crate::engine::{Engine, Metric};
use crate::logs::Outcome;
use crate::runner::{ProcessSpec, RunPlan};
use crate::serve::EngineArg;
use crate::trial::{next_tensor_parallelism, Candidate};

use super::{
    append_engine_args, docker_server_args, http_readiness, push_default_arg,
    server_container_name, EngineAdapter,
};

const VLLM_ARGPARSE_INTROSPECTION: &str = r#"
import argparse
from vllm.entrypoints.openai.cli_args import make_arg_parser

parser = make_arg_parser(argparse.ArgumentParser(prog="vllm serve"))
flag_types = (
    argparse._StoreTrueAction,
    argparse._StoreFalseAction,
    argparse._StoreConstAction,
    argparse.BooleanOptionalAction,
)
for action in parser._actions:
    kind = "bool flag" if isinstance(action, flag_types) or action.nargs == 0 else "value"
    for option in action.option_strings:
        if option.startswith("--"):
            print(f"{option}\t{kind}")
"#;

pub(super) static VLLM: VllmAdapter = VllmAdapter;

pub(super) struct VllmAdapter;

impl EngineAdapter for VllmAdapter {
    fn engine(&self) -> Engine {
        Engine::Vllm
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
                "-c".to_string(),
                VLLM_ARGPARSE_INTROSPECTION.to_string(),
            ],
        )
    }

    fn initial_candidate(&self, setup: &Setup) -> Candidate {
        let mut candidate = setup.candidate.clone();
        candidate.scheduler.prefill_token_budget = match setup.metric {
            Metric::Ttft => 16_384,
            Metric::Itl => 2_048,
            Metric::Tps => 8_192,
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
                } else if let Some(tp) =
                    next_tensor_parallelism(last.parallelism.tensor, setup.gpus)
                {
                    next.parallelism.tensor = tp;
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
            "tp={}, gpu_memory_utilization={:.2}, max_num_batched_tokens={}, max_running_requests={}",
            candidate.parallelism.tensor,
            candidate.memory.fraction,
            candidate.scheduler.prefill_token_budget,
            candidate.scheduler.max_running_requests
        )
    }

    fn serving_args(&self, config: &ServingConfig) -> Vec<EngineArg> {
        let mut args = Vec::new();
        push_default_arg(&mut args, config, "--model", config.model.clone());
        push_default_arg(
            &mut args,
            config,
            "--tensor-parallel-size",
            config.candidate.parallelism.tensor.to_string(),
        );
        push_default_arg(
            &mut args,
            config,
            "--gpu-memory-utilization",
            format!("{:.2}", config.candidate.memory.fraction),
        );
        push_default_arg(
            &mut args,
            config,
            "--max-model-len",
            config.max_model_len.to_string(),
        );
        push_default_arg(
            &mut args,
            config,
            "--max-num-batched-tokens",
            config.candidate.scheduler.prefill_token_budget.to_string(),
        );
        args.extend(config.serve_args.clone());
        args
    }

    fn run_plan(&self, config: &ServingConfig) -> RunPlan {
        let mut server_args = docker_server_args(config, "8000");
        append_engine_args(&mut server_args, self.serving_args(config));

        RunPlan {
            server: ProcessSpec::new("docker", server_args),
            benchmark: ProcessSpec::new("docker", benchmark_args(config)),
            readiness: http_readiness(config, config.port, "/v1/models"),
            server_container: Some(server_container_name(config)),
        }
    }
}

fn benchmark_args(config: &ServingConfig) -> Vec<String> {
    vec![
        "run".to_string(),
        "--rm".to_string(),
        "--gpus".to_string(),
        "all".to_string(),
        "-e".to_string(),
        "HF_TOKEN".to_string(),
        "--network".to_string(),
        "host".to_string(),
        "--entrypoint".to_string(),
        "vllm".to_string(),
        config.image.clone(),
        "bench".to_string(),
        "serve".to_string(),
        "--backend".to_string(),
        "vllm".to_string(),
        "--model".to_string(),
        config.model.clone(),
        "--host".to_string(),
        config.host.clone(),
        "--port".to_string(),
        config.port.to_string(),
        "--endpoint".to_string(),
        "/v1/completions".to_string(),
        "--dataset-name".to_string(),
        config.benchmark.dataset_name.clone(),
        "--num-prompts".to_string(),
        config.benchmark.num_prompts.to_string(),
        "--request-rate".to_string(),
        config.benchmark.request_rate.clone(),
        "--max-concurrency".to_string(),
        config
            .benchmark
            .max_concurrency
            .unwrap_or(config.benchmark.num_prompts)
            .to_string(),
    ]
}
