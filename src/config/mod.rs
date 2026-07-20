mod file;
mod validate;

use std::{collections::BTreeMap, net::IpAddr, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    domain::{
        candidate::{Candidate, CandidateOverrides, DynamicArg, SweepSpec},
        engine::{Engine, Metric},
        run::{PullPolicy, ResolvedImage},
    },
    error::{Error, ErrorKind, ExecutionStage, Result},
};

#[cfg(test)]
pub(crate) use file::parse_config_text;
pub(crate) use file::{load_config, ConfigFile};

pub(crate) const DEFAULT_LEADERBOARD_URL: &str =
    "https://hf-dwarez-optimum-advisor-leaderboard.hf.space";

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct RuntimeInput {
    #[schemars(range(min = 1))]
    pub gpus: Option<usize>,
    #[schemars(length(min = 1))]
    pub gpu_devices: Option<Vec<String>>,
    pub pull_policy: Option<PullPolicy>,
    pub allow_local_image: Option<bool>,
    pub bind_host: Option<IpAddr>,
    #[schemars(range(min = 1))]
    pub port: Option<u16>,
    /// Maximum model context length. Set this instead of passing
    /// `max-model-len` in `serve_args`.
    #[schemars(range(min = 1))]
    pub max_model_len: Option<u32>,
    #[schemars(range(min = 1))]
    pub startup_timeout_secs: Option<u64>,
    #[schemars(range(min = 1))]
    pub benchmark_timeout_secs: Option<u64>,
    #[schemars(range(min = 1))]
    pub max_process_output_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct BenchmarkInput {
    pub dataset_name: Option<String>,
    #[schemars(range(min = 1))]
    pub num_prompts: Option<u32>,
    pub request_rate: Option<String>,
    #[schemars(range(min = 1))]
    pub max_concurrency: Option<u32>,
    #[schemars(range(min = 1))]
    pub random_input_len: Option<u32>,
    #[schemars(range(min = 1))]
    pub random_output_len: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct CorrectnessInput {
    pub enabled: Option<bool>,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub threshold: Option<f64>,
    #[schemars(range(min = 1))]
    pub timeout_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ModelMemoryInput {
    pub enabled: Option<bool>,
    pub required: Option<bool>,
    pub command: Option<PathBuf>,
    #[schemars(range(min = 1))]
    pub timeout_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct LeaderboardInput {
    pub submit: Option<bool>,
    pub url: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ConfigInput {
    /// Optional in JSON contexts (MCP tools); when present it must be 2, so
    /// configs copied from schema-v2 TOML files decode unchanged.
    #[schemars(range(min = 2, max = 2))]
    pub schema_version: Option<u32>,
    #[schemars(required)]
    pub engine: Option<Engine>,
    pub image: Option<String>,
    #[schemars(required)]
    pub model: Option<String>,
    /// Optimization objective. When omitted, models whose IDs declare at most 3B
    /// parameters default to `tpot` (decode latency); all others default to `tps`.
    /// Set this explicitly when the model name does not encode its parameter count
    /// or when the intended workload favors a different objective.
    pub metric: Option<Metric>,
    pub runtime: RuntimeInput,
    pub benchmark: BenchmarkInput,
    pub candidate: CandidateOverrides,
    pub correctness: CorrectnessInput,
    pub model_memory: ModelMemoryInput,
    pub leaderboard: LeaderboardInput,
    /// Extra engine-specific serving arguments only. Do not duplicate fields
    /// managed by normalized configuration: use `candidate.max_running_requests`
    /// for `max-num-seqs`, `runtime.max_model_len` for `max-model-len`, and
    /// `candidate.memory_fraction` for `gpu-memory-utilization`.
    pub serve_args: Vec<DynamicArg>,
    pub sweep: Option<SweepSpec>,
}

impl ConfigInput {
    pub(crate) fn minimal(engine: Engine, model: impl Into<String>) -> Self {
        Self {
            engine: Some(engine),
            model: Some(model.into()),
            ..Self::default()
        }
    }

    pub(crate) fn overlay(mut self, higher: Self) -> Self {
        macro_rules! replace {
            ($target:expr, $source:expr) => {
                if $source.is_some() {
                    $target = $source;
                }
            };
        }

        replace!(self.schema_version, higher.schema_version);
        replace!(self.engine, higher.engine);
        replace!(self.image, higher.image);
        replace!(self.model, higher.model);
        replace!(self.metric, higher.metric);
        replace!(self.runtime.gpus, higher.runtime.gpus);
        replace!(self.runtime.gpu_devices, higher.runtime.gpu_devices);
        replace!(self.runtime.pull_policy, higher.runtime.pull_policy);
        replace!(
            self.runtime.allow_local_image,
            higher.runtime.allow_local_image
        );
        replace!(self.runtime.bind_host, higher.runtime.bind_host);
        replace!(self.runtime.port, higher.runtime.port);
        replace!(self.runtime.max_model_len, higher.runtime.max_model_len);
        replace!(
            self.runtime.startup_timeout_secs,
            higher.runtime.startup_timeout_secs
        );
        replace!(
            self.runtime.benchmark_timeout_secs,
            higher.runtime.benchmark_timeout_secs
        );
        replace!(
            self.runtime.max_process_output_bytes,
            higher.runtime.max_process_output_bytes
        );
        replace!(self.benchmark.dataset_name, higher.benchmark.dataset_name);
        replace!(self.benchmark.num_prompts, higher.benchmark.num_prompts);
        replace!(self.benchmark.request_rate, higher.benchmark.request_rate);
        replace!(
            self.benchmark.max_concurrency,
            higher.benchmark.max_concurrency
        );
        replace!(
            self.benchmark.random_input_len,
            higher.benchmark.random_input_len
        );
        replace!(
            self.benchmark.random_output_len,
            higher.benchmark.random_output_len
        );
        replace!(
            self.candidate.tensor_parallelism,
            higher.candidate.tensor_parallelism
        );
        replace!(
            self.candidate.memory_fraction,
            higher.candidate.memory_fraction
        );
        replace!(
            self.candidate.prefill_token_budget,
            higher.candidate.prefill_token_budget
        );
        replace!(
            self.candidate.max_running_requests,
            higher.candidate.max_running_requests
        );
        replace!(self.correctness.enabled, higher.correctness.enabled);
        replace!(self.correctness.threshold, higher.correctness.threshold);
        replace!(
            self.correctness.timeout_secs,
            higher.correctness.timeout_secs
        );
        replace!(self.model_memory.enabled, higher.model_memory.enabled);
        replace!(self.model_memory.required, higher.model_memory.required);
        replace!(self.model_memory.command, higher.model_memory.command);
        replace!(
            self.model_memory.timeout_secs,
            higher.model_memory.timeout_secs
        );
        replace!(self.leaderboard.submit, higher.leaderboard.submit);
        replace!(self.leaderboard.url, higher.leaderboard.url);

        for argument in higher.serve_args {
            if let Some(existing) = self
                .serve_args
                .iter_mut()
                .find(|existing| existing.name == argument.name)
            {
                *existing = argument;
            } else {
                self.serve_args.push(argument);
            }
        }
        if higher.sweep.is_some() {
            self.sweep = higher.sweep;
        }
        self
    }

    pub(crate) fn normalize(self) -> Result<NormalizedConfig> {
        validate::normalize(self)
    }
}

impl TryFrom<ConfigFile> for ConfigInput {
    type Error = crate::error::Error;

    fn try_from(file: ConfigFile) -> Result<Self> {
        file::into_input(file)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct RuntimeConfig {
    pub gpus: usize,
    pub gpu_devices: Vec<String>,
    pub pull_policy: PullPolicy,
    pub allow_local_image: bool,
    pub bind_host: IpAddr,
    pub port: u16,
    pub max_model_len: u32,
    pub startup_timeout_secs: u64,
    pub benchmark_timeout_secs: u64,
    pub max_process_output_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct BenchmarkConfig {
    pub dataset_name: String,
    pub num_prompts: u32,
    pub request_rate: String,
    pub max_concurrency: u32,
    pub random_input_len: u32,
    pub random_output_len: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct CorrectnessConfig {
    pub enabled: bool,
    pub threshold: f64,
    pub timeout_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct ModelMemoryConfig {
    pub enabled: bool,
    pub required: bool,
    pub command: Option<PathBuf>,
    pub timeout_secs: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct LeaderboardConfig {
    pub submit: bool,
    pub url: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct NormalizedConfig {
    pub engine: Engine,
    pub image: String,
    pub model: String,
    pub metric: Metric,
    pub runtime: RuntimeConfig,
    pub benchmark: BenchmarkConfig,
    pub candidate: Candidate,
    pub correctness: CorrectnessConfig,
    pub model_memory: ModelMemoryConfig,
    pub leaderboard: LeaderboardConfig,
    pub serve_args: Vec<DynamicArg>,
    #[serde(skip)]
    #[schemars(skip)]
    pub sweep: Option<SweepSpec>,
}

impl NormalizedConfig {
    pub(crate) fn into_executable(self, resolved_image: ResolvedImage) -> ExecutableConfig {
        ExecutableConfig {
            engine: self.engine,
            image: resolved_image,
            model: self.model,
            metric: self.metric,
            runtime: self.runtime,
            benchmark: self.benchmark,
            candidate: self.candidate,
            correctness: self.correctness,
            model_memory: self.model_memory,
            leaderboard: self.leaderboard,
            serve_args: self.serve_args,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub(crate) struct ExecutableConfig {
    pub engine: Engine,
    pub image: ResolvedImage,
    pub model: String,
    pub metric: Metric,
    pub runtime: RuntimeConfig,
    pub benchmark: BenchmarkConfig,
    pub candidate: Candidate,
    pub correctness: CorrectnessConfig,
    pub model_memory: ModelMemoryConfig,
    pub leaderboard: LeaderboardConfig,
    pub serve_args: Vec<DynamicArg>,
}

impl ExecutableConfig {
    pub(crate) fn best_config_toml(&self) -> Result<String> {
        #[derive(Serialize)]
        struct BestRuntime<'a> {
            gpus: usize,
            #[serde(skip_serializing_if = "Option::is_none")]
            gpu_devices: Option<&'a [String]>,
            pull_policy: PullPolicy,
            allow_local_image: bool,
            bind_host: IpAddr,
            port: u16,
            max_model_len: u32,
            startup_timeout_secs: u64,
            benchmark_timeout_secs: u64,
            max_process_output_bytes: u64,
        }

        #[derive(Serialize)]
        struct BestConfigDocument<'a> {
            schema_version: u32,
            engine: Engine,
            image: &'a str,
            model: &'a str,
            metric: Metric,
            runtime: BestRuntime<'a>,
            benchmark: &'a BenchmarkConfig,
            candidate: &'a Candidate,
            correctness: &'a CorrectnessConfig,
            model_memory: &'a ModelMemoryConfig,
            leaderboard: LeaderboardConfig,
            #[serde(skip_serializing_if = "BTreeMap::is_empty")]
            serve: BTreeMap<String, toml::Value>,
        }

        let runtime = BestRuntime {
            gpus: self.runtime.gpus,
            gpu_devices: (!self.runtime.gpu_devices.is_empty())
                .then_some(self.runtime.gpu_devices.as_slice()),
            pull_policy: if self.image.local_only {
                PullPolicy::Never
            } else {
                PullPolicy::Missing
            },
            allow_local_image: self.image.local_only,
            bind_host: self.runtime.bind_host,
            port: self.runtime.port,
            max_model_len: self.runtime.max_model_len,
            startup_timeout_secs: self.runtime.startup_timeout_secs,
            benchmark_timeout_secs: self.runtime.benchmark_timeout_secs,
            max_process_output_bytes: self.runtime.max_process_output_bytes,
        };
        let mut leaderboard = self.leaderboard.clone();
        leaderboard.submit = false;
        let serve = self
            .serve_args
            .iter()
            .map(|argument| {
                (
                    argument.name.clone(),
                    argument
                        .value
                        .clone()
                        .map(toml::Value::String)
                        .unwrap_or(toml::Value::Boolean(true)),
                )
            })
            .collect();
        let document = BestConfigDocument {
            schema_version: 2,
            engine: self.engine,
            image: &self.image.immutable,
            model: &self.model,
            metric: self.metric,
            runtime,
            benchmark: &self.benchmark,
            candidate: &self.candidate,
            correctness: &self.correctness,
            model_memory: &self.model_memory,
            leaderboard,
            serve,
        };
        toml::to_string_pretty(&document).map_err(|source| {
            Error::new(
                ErrorKind::Configuration,
                Some(ExecutionStage::Persistence),
                "failed to serialize winning configuration",
            )
            .with_operation("serialize winning configuration")
            .with_source(source)
        })
    }
}

#[cfg(test)]
mod best_config_tests {
    use super::*;

    #[test]
    fn winning_config_round_trips_as_a_single_immutable_candidate() {
        let mut input = ConfigInput::minimal(Engine::Vllm, "model");
        input.runtime.pull_policy = Some(PullPolicy::Always);
        input.candidate.memory_fraction = Some(0.73);
        input.serve_args = vec![
            DynamicArg::flag("disable-log-stats"),
            DynamicArg::value("kv-cache-dtype", "fp8"),
        ];
        let executable = input.normalize().unwrap().into_executable(ResolvedImage {
            requested: "repo/image:tag".into(),
            immutable: "repo/image@sha256:abc".into(),
            local_only: false,
        });

        let text = executable.best_config_toml().unwrap();
        let round_trip = ConfigInput::try_from(parse_config_text(&text).unwrap())
            .unwrap()
            .normalize()
            .unwrap();

        assert_eq!(round_trip.image, "repo/image@sha256:abc");
        assert_eq!(round_trip.runtime.pull_policy, PullPolicy::Missing);
        assert!(!round_trip.runtime.allow_local_image);
        assert_eq!(round_trip.candidate, executable.candidate);
        assert_eq!(round_trip.serve_args, executable.serve_args);
        assert!(!round_trip.leaderboard.submit);
        assert!(round_trip.sweep.is_none());
    }

    #[test]
    fn zero_correctness_threshold_is_a_valid_execution_only_gate() {
        let mut input = ConfigInput::minimal(Engine::Vllm, "model");
        input.correctness.enabled = Some(true);
        input.correctness.threshold = Some(0.0);

        let normalized = input.clone().normalize().unwrap();

        assert!(normalized.correctness.enabled);
        assert_eq!(normalized.correctness.threshold, 0.0);
        for invalid in [-f64::EPSILON, f64::NAN, 1.0 + f64::EPSILON] {
            input.correctness.threshold = Some(invalid);
            assert!(input.clone().normalize().is_err(), "{invalid}");
        }
    }
}
