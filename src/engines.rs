use std::time::Duration;

use crate::cli::Setup;
use crate::engine::{Engine, Metric};
use crate::logs::Outcome;
use crate::runner::{ProcessSpec, Readiness, RunPlan};
use crate::trial::{next_tensor_parallelism, Candidate};

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

pub trait EngineAdapter {
    fn engine(&self) -> Engine;
    fn help_command(&self, image: String) -> ProcessSpec;
    fn initial_candidate(&self, setup: &Setup) -> Candidate;
    fn next_candidate(&self, setup: &Setup, last: &Candidate, outcome: Outcome) -> Candidate;
    fn describe_candidate(&self, candidate: &Candidate) -> String;
    fn run_plan(&self, setup: &Setup, candidate: &Candidate) -> RunPlan;
}

pub fn adapter_for(engine: Engine) -> &'static dyn EngineAdapter {
    match engine {
        Engine::Vllm => &VLLM,
        Engine::Sglang => &SGLANG,
    }
}

struct VllmAdapter;
struct SglangAdapter;

static VLLM: VllmAdapter = VllmAdapter;
static SGLANG: SglangAdapter = SglangAdapter;

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

    fn run_plan(&self, setup: &Setup, candidate: &Candidate) -> RunPlan {
        let image = setup
            .image
            .clone()
            .unwrap_or_else(|| setup.engine.default_image().to_string());
        let mut server_args = docker_base_args(setup, "8000", image);
        server_args.extend([
            "--model".to_string(),
            setup.model.clone(),
            "--tensor-parallel-size".to_string(),
            candidate.parallelism.tensor.to_string(),
            "--gpu-memory-utilization".to_string(),
            format!("{:.2}", candidate.memory.fraction),
            "--max-num-batched-tokens".to_string(),
            candidate.scheduler.prefill_token_budget.to_string(),
        ]);
        append_extra_args(setup, &mut server_args);

        RunPlan {
            server: ProcessSpec::new("docker", server_args),
            benchmark: ProcessSpec::new(
                "vllm",
                vec![
                    "bench".to_string(),
                    "serve".to_string(),
                    "--backend".to_string(),
                    "vllm".to_string(),
                    "--model".to_string(),
                    setup.model.clone(),
                    "--endpoint".to_string(),
                    "/v1/completions".to_string(),
                    "--dataset-name".to_string(),
                    "random".to_string(),
                    "--num-prompts".to_string(),
                    "100".to_string(),
                ],
            ),
            readiness: readiness(setup, setup.port),
        }
    }
}

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

    fn run_plan(&self, setup: &Setup, candidate: &Candidate) -> RunPlan {
        let image = setup
            .image
            .clone()
            .unwrap_or_else(|| setup.engine.default_image().to_string());
        let mut server_args = docker_base_args(setup, "30000", image);
        server_args.extend([
            "python3".to_string(),
            "-m".to_string(),
            "sglang.launch_server".to_string(),
            "--model-path".to_string(),
            setup.model.clone(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            "30000".to_string(),
            "--tp-size".to_string(),
            candidate.parallelism.tensor.to_string(),
            "--mem-fraction-static".to_string(),
            format!("{:.2}", candidate.memory.fraction),
            "--chunked-prefill-size".to_string(),
            candidate.scheduler.prefill_token_budget.to_string(),
            "--max-running-requests".to_string(),
            candidate.scheduler.max_running_requests.to_string(),
        ]);
        append_extra_args(setup, &mut server_args);

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
                    setup.model.clone(),
                    "--dataset-name".to_string(),
                    "random".to_string(),
                    "--num-prompts".to_string(),
                    "100".to_string(),
                ],
            ),
            readiness: readiness(setup, setup.port),
        }
    }
}

fn docker_base_args(setup: &Setup, container_port: &str, image: String) -> Vec<String> {
    vec![
        "run".to_string(),
        "--rm".to_string(),
        "--gpus".to_string(),
        "all".to_string(),
        "--ipc=host".to_string(),
        "-p".to_string(),
        format!("{}:{container_port}", setup.port),
        image,
    ]
}

fn append_extra_args(setup: &Setup, args: &mut Vec<String>) {
    for arg in &setup.serve_args {
        arg.append_to(args);
    }
}

fn readiness(setup: &Setup, port: u16) -> Readiness {
    Readiness {
        host: setup.host.clone(),
        port,
        timeout: Duration::from_secs(setup.startup_timeout_secs),
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::Setup;
    use crate::engine::{Metric, Mode};
    use crate::serve::EngineArg;

    use super::*;

    fn setup(engine: Engine) -> Setup {
        Setup {
            mode: Mode::Plan,
            engine,
            model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            image: None,
            gpus: 1,
            host: "127.0.0.1".to_string(),
            port: 8000,
            startup_timeout_secs: 300,
            param_cache_dir: ".optimum-advisor/params".to_string(),
            refresh_params: false,
            validate_params: false,
            metric: Metric::Tps,
            execute: false,
            log_file: None,
            candidate: Candidate::default(),
            serve_args: Vec::new(),
        }
    }

    #[test]
    fn vllm_renders_vllm_parameter_names() {
        let setup = setup(Engine::Vllm);
        let candidate = adapter_for(Engine::Vllm).initial_candidate(&setup);
        let server = adapter_for(Engine::Vllm)
            .run_plan(&setup, &candidate)
            .server;
        assert!(server.args.contains(&"--tensor-parallel-size".to_string()));
        assert!(server
            .args
            .contains(&"--max-num-batched-tokens".to_string()));
    }

    #[test]
    fn sglang_renders_sglang_parameter_names() {
        let setup = setup(Engine::Sglang);
        let candidate = adapter_for(Engine::Sglang).initial_candidate(&setup);
        let server = adapter_for(Engine::Sglang)
            .run_plan(&setup, &candidate)
            .server;
        assert!(server.args.contains(&"--tp-size".to_string()));
        assert!(server.args.contains(&"--chunked-prefill-size".to_string()));
    }

    #[test]
    fn extra_args_are_appended_for_version_specific_flags() {
        let mut setup = setup(Engine::Vllm);
        setup
            .serve_args
            .push(EngineArg::assignment("kv-cache-dtype=fp8").unwrap());
        let candidate = adapter_for(Engine::Vllm).initial_candidate(&setup);
        let server = adapter_for(Engine::Vllm)
            .run_plan(&setup, &candidate)
            .server;
        assert!(server
            .args
            .ends_with(&["--kv-cache-dtype".to_string(), "fp8".to_string()]));
    }
}
