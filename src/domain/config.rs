use crate::cli::Setup;
use crate::engine::{Engine, Metric};
use crate::serve::EngineArg;
use crate::trial::Candidate;

#[derive(Clone, Debug, PartialEq)]
pub struct ServingConfig {
    pub engine: Engine,
    pub image: String,
    pub model: String,
    pub gpus: usize,
    pub host: String,
    pub port: u16,
    pub startup_timeout_secs: u64,
    pub max_model_len: u32,
    pub metric: Metric,
    pub candidate: Candidate,
    pub serve_args: Vec<EngineArg>,
    pub benchmark: BenchmarkConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub dataset_name: String,
    pub num_prompts: u32,
    pub request_rate: String,
    pub max_concurrency: Option<u32>,
    pub random_input_len: u32,
    pub random_output_len: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            dataset_name: "random".to_string(),
            num_prompts: 100,
            request_rate: "1".to_string(),
            max_concurrency: Some(1),
            random_input_len: 1024,
            random_output_len: 128,
        }
    }
}

impl ServingConfig {
    pub fn from_setup_and_candidate(setup: &Setup, candidate: Candidate) -> Self {
        Self {
            engine: setup.engine,
            image: setup
                .image
                .clone()
                .unwrap_or_else(|| setup.engine.default_image().to_string()),
            model: setup.model.clone(),
            gpus: setup.gpus,
            host: setup.host.clone(),
            port: setup.port,
            startup_timeout_secs: setup.startup_timeout_secs,
            max_model_len: setup.max_model_len,
            metric: setup.metric,
            candidate,
            serve_args: setup.serve_args.clone(),
            benchmark: setup.benchmark.clone(),
        }
    }
}
