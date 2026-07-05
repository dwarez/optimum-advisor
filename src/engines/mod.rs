use std::time::Duration;

use crate::cli::Setup;
use crate::config::ServingConfig;
use crate::engine::Engine;
use crate::logs::Outcome;
use crate::runner::{Readiness, RunPlan, OWNED_CONTAINER_LABEL, SERVER_CONTAINER_LABEL};
use crate::serve::EngineArg;
use crate::trial::Candidate;

mod sglang;
mod vllm;

pub trait EngineAdapter {
    fn engine(&self) -> Engine;
    fn help_command(&self, image: String) -> crate::runner::ProcessSpec;
    fn initial_candidate(&self, setup: &Setup) -> Candidate;
    fn next_candidate(&self, setup: &Setup, last: &Candidate, outcome: Outcome) -> Candidate;
    fn describe_candidate(&self, candidate: &Candidate) -> String;
    fn serving_args(&self, config: &ServingConfig) -> Vec<EngineArg>;
    fn run_plan(&self, config: &ServingConfig) -> RunPlan;
}

pub fn adapter_for(engine: Engine) -> &'static dyn EngineAdapter {
    match engine {
        Engine::Vllm => &vllm::VLLM,
        Engine::Sglang => &sglang::SGLANG,
    }
}

pub(crate) fn docker_server_args(config: &ServingConfig, container_port: &str) -> Vec<String> {
    vec![
        "run".to_string(),
        "--rm".to_string(),
        "--gpus".to_string(),
        "all".to_string(),
        "-e".to_string(),
        "HF_TOKEN".to_string(),
        "--ipc=host".to_string(),
        "-p".to_string(),
        format!("{}:{container_port}", config.port),
        "--name".to_string(),
        server_container_name(config),
        "--label".to_string(),
        OWNED_CONTAINER_LABEL.to_string(),
        "--label".to_string(),
        SERVER_CONTAINER_LABEL.to_string(),
        "--label".to_string(),
        format!("optimum-advisor.engine={}", config.engine),
        "--label".to_string(),
        format!("optimum-advisor.port={}", config.port),
        config.image.clone(),
    ]
}

pub(crate) fn server_container_name(config: &ServingConfig) -> String {
    format!(
        "optimum-advisor-{}-{}-{}-server",
        config.engine,
        config.port,
        std::process::id()
    )
}

pub(crate) fn append_engine_args(args: &mut Vec<String>, engine_args: Vec<EngineArg>) {
    for arg in engine_args {
        arg.append_to(args);
    }
}

pub(crate) fn http_readiness(config: &ServingConfig, port: u16, path: &str) -> Readiness {
    Readiness {
        host: config.host.clone(),
        port,
        timeout: Duration::from_secs(config.startup_timeout_secs),
        http_path: Some(path.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::Setup;
    use crate::config::{BenchmarkConfig, ServingConfig};
    use crate::engine::{Metric, Mode};
    use crate::serve::EngineArg;
    use crate::trial::Candidate;

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
        }
    }

    #[test]
    fn vllm_renders_vllm_parameter_names() {
        let setup = setup(Engine::Vllm);
        let candidate = adapter_for(Engine::Vllm).initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);
        let server = adapter_for(Engine::Vllm).run_plan(&config).server;
        assert!(server.args.contains(&"--tensor-parallel-size".to_string()));
        assert!(server.args.contains(&"--max-model-len".to_string()));
        assert!(server.args.contains(&"8192".to_string()));
        assert!(server
            .args
            .contains(&"--max-num-batched-tokens".to_string()));
    }

    #[test]
    fn vllm_benchmark_runs_from_the_engine_image() {
        let setup = setup(Engine::Vllm);
        let candidate = adapter_for(Engine::Vllm).initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);
        let benchmark = adapter_for(Engine::Vllm).run_plan(&config).benchmark;

        assert_eq!(benchmark.program, "docker");
        assert!(benchmark.args.contains(&"--network".to_string()));
        assert!(benchmark.args.contains(&"host".to_string()));
        assert!(benchmark.args.contains(&"--entrypoint".to_string()));
        assert!(benchmark.args.contains(&"-e".to_string()));
        assert!(benchmark.args.contains(&"HF_TOKEN".to_string()));
        assert!(benchmark.args.contains(&"vllm".to_string()));
        assert!(benchmark.args.contains(&"bench".to_string()));
        assert!(benchmark.args.contains(&"serve".to_string()));
        assert!(benchmark.args.contains(&"--port".to_string()));
        assert!(benchmark.args.contains(&"8000".to_string()));
    }

    #[test]
    fn server_containers_are_named_and_labeled_for_cleanup() {
        let setup = setup(Engine::Sglang);
        let candidate = adapter_for(Engine::Sglang).initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);
        let plan = adapter_for(Engine::Sglang).run_plan(&config);

        let container = plan.server_container.as_deref().unwrap();
        assert!(container.starts_with("optimum-advisor-sglang-8000-"));
        assert!(container.ends_with("-server"));
        assert!(plan.server.args.contains(&"--name".to_string()));
        assert!(plan.server.args.contains(&container.to_string()));
        assert!(plan.server.args.contains(&"--label".to_string()));
        assert!(plan
            .server
            .args
            .contains(&OWNED_CONTAINER_LABEL.to_string()));
        assert!(plan
            .server
            .args
            .contains(&SERVER_CONTAINER_LABEL.to_string()));
    }

    #[test]
    fn vllm_waits_for_models_endpoint() {
        let setup = setup(Engine::Vllm);
        let candidate = adapter_for(Engine::Vllm).initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);
        let plan = adapter_for(Engine::Vllm).run_plan(&config);

        assert_eq!(plan.readiness.http_path.as_deref(), Some("/v1/models"));
    }

    #[test]
    fn sglang_renders_sglang_parameter_names() {
        let setup = setup(Engine::Sglang);
        let candidate = adapter_for(Engine::Sglang).initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);
        let server = adapter_for(Engine::Sglang).run_plan(&config).server;
        assert!(server.args.contains(&"--tp-size".to_string()));
        assert!(server.args.contains(&"--chunked-prefill-size".to_string()));
    }

    #[test]
    fn sglang_benchmark_runs_from_the_engine_image() {
        let setup = setup(Engine::Sglang);
        let candidate = adapter_for(Engine::Sglang).initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);
        let plan = adapter_for(Engine::Sglang).run_plan(&config);

        assert_eq!(plan.benchmark.program, "docker");
        assert!(plan.benchmark.args.contains(&"--network".to_string()));
        assert!(plan.benchmark.args.contains(&"host".to_string()));
        assert!(plan
            .benchmark
            .args
            .contains(&"sglang.bench_serving".to_string()));
        assert!(plan
            .benchmark
            .args
            .contains(&"--random-output-len".to_string()));
        assert_eq!(plan.readiness.http_path.as_deref(), Some("/v1/models"));
    }

    #[test]
    fn extra_args_are_appended_for_version_specific_flags() {
        let mut setup = setup(Engine::Vllm);
        setup
            .serve_args
            .push(EngineArg::assignment("kv-cache-dtype=fp8").unwrap());
        let candidate = adapter_for(Engine::Vllm).initial_candidate(&setup);
        let config = ServingConfig::from_setup_and_candidate(&setup, candidate);
        let server = adapter_for(Engine::Vllm).run_plan(&config).server;
        assert!(server
            .args
            .ends_with(&["--kv-cache-dtype".to_string(), "fp8".to_string()]));
    }
}
