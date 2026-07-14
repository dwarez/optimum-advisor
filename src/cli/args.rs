use std::{collections::HashSet, ffi::OsString, net::IpAddr, path::PathBuf};

use clap::{error::ErrorKind as ClapErrorKind, Args, Parser, Subcommand};

use crate::{
    config::{
        load_config, BenchmarkInput, ConfigInput, CorrectnessInput, LeaderboardInput,
        ModelMemoryInput, RuntimeInput,
    },
    domain::{
        candidate::{canonical_name, validate_dynamic_name, CandidateOverrides, DynamicArg},
        engine::{Engine, Metric},
        run::PullPolicy,
    },
    error::{Error, Result},
};

const DEFAULT_RESULTS_DIR: &str = ".optimum-advisor/results";
const DEFAULT_PARAMETER_CACHE: &str = ".optimum-advisor/params";

#[derive(Debug)]
pub(crate) enum ParsedCli {
    Display(String),
    Invocation(Box<Invocation>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CommandKind {
    Plan,
    Params,
    Hardware,
    Serve,
    Bench,
    Sweep,
    Cleanup,
    Mcp,
}

#[derive(Debug)]
pub(crate) struct Invocation {
    pub kind: CommandKind,
    pub input: ConfigInput,
    pub execute: bool,
    pub results_dir: PathBuf,
    pub parameter_cache_dir: PathBuf,
    pub refresh_parameters: bool,
    pub offline_parameters: bool,
    pub cleanup_run_id: Option<String>,
    pub cleanup_dry_run: bool,
}

#[derive(Debug, Parser)]
#[command(
    name = "optimum-advisor",
    version,
    about = "Benchmark production LLM serving configurations"
)]
struct Cli {
    #[command(subcommand)]
    command: CliCommand,
}

#[derive(Debug, Subcommand)]
enum CliCommand {
    /// Render one validated, non-executing serving and benchmark preview.
    Plan {
        #[arg(long)]
        config: Option<PathBuf>,
        #[command(flatten)]
        overrides: ConfigOverrides,
    },
    /// Resolve an engine image and inspect its serving-parameter schema.
    Params {
        #[arg(long)]
        engine: Engine,
        #[arg(long)]
        image: Option<String>,
        #[arg(long, default_value = "missing")]
        pull_policy: PullPolicy,
        #[arg(long)]
        allow_local_image: bool,
        #[arg(long, default_value = DEFAULT_PARAMETER_CACHE)]
        cache_dir: PathBuf,
        #[arg(long, conflicts_with = "offline")]
        refresh: bool,
        #[arg(long)]
        offline: bool,
    },
    /// Inspect local NVIDIA hardware.
    Hardware,
    /// Run one validated serving container until it exits or is interrupted.
    Serve {
        #[arg(long)]
        config: Option<PathBuf>,
        #[command(flatten)]
        overrides: ConfigOverrides,
    },
    /// Evaluate one serving candidate.
    Bench {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long, default_value = DEFAULT_RESULTS_DIR)]
        results_dir: PathBuf,
        #[arg(long)]
        dry_run: bool,
        #[command(flatten)]
        overrides: ConfigOverrides,
    },
    /// Evaluate the bounded sweep declared in a v2 TOML file.
    Sweep {
        #[arg(long)]
        config: PathBuf,
        #[arg(long, default_value = DEFAULT_RESULTS_DIR)]
        results_dir: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
    /// List or remove only Optimum Advisor-owned Docker containers.
    Cleanup {
        #[arg(long)]
        run_id: Option<String>,
        #[arg(long)]
        dry_run: bool,
    },
    /// Serve the strict newline-delimited MCP protocol on stdin/stdout.
    Mcp,
}

#[derive(Clone, Debug, Default, Args)]
struct ConfigOverrides {
    #[arg(long)]
    engine: Option<Engine>,
    #[arg(long)]
    image: Option<String>,
    #[arg(long)]
    pull_policy: Option<PullPolicy>,
    #[arg(long)]
    allow_local_image: bool,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    metric: Option<Metric>,
    #[arg(long)]
    gpus: Option<usize>,
    #[arg(long = "gpu-device")]
    gpu_devices: Vec<String>,
    #[arg(long)]
    bind_host: Option<IpAddr>,
    #[arg(long)]
    port: Option<u16>,
    #[arg(long)]
    startup_timeout_secs: Option<u64>,
    #[arg(long)]
    benchmark_timeout_secs: Option<u64>,
    #[arg(long)]
    max_process_output_bytes: Option<u64>,
    #[arg(long)]
    dataset_name: Option<String>,
    #[arg(long)]
    num_prompts: Option<u32>,
    #[arg(long)]
    request_rate: Option<String>,
    #[arg(long)]
    benchmark_max_concurrency: Option<u32>,
    #[arg(long)]
    random_input_len: Option<u32>,
    #[arg(long)]
    random_output_len: Option<u32>,
    #[arg(long)]
    tensor_parallelism: Option<usize>,
    #[arg(long)]
    memory_fraction: Option<f64>,
    #[arg(long)]
    prefill_token_budget: Option<u32>,
    #[arg(long)]
    max_running_requests: Option<u32>,
    #[arg(long)]
    no_correctness: bool,
    #[arg(long)]
    correctness_threshold: Option<f64>,
    #[arg(long)]
    correctness_timeout_secs: Option<u64>,
    #[arg(long, conflicts_with = "require_model_memory")]
    no_model_memory: bool,
    #[arg(long, conflicts_with = "no_model_memory")]
    require_model_memory: bool,
    #[arg(long)]
    hf_mem_command: Option<PathBuf>,
    #[arg(long)]
    hf_mem_timeout_secs: Option<u64>,
    #[arg(long)]
    leaderboard_submit: bool,
    #[arg(long)]
    leaderboard_url: Option<String>,
    #[arg(long = "serve-arg", value_name = "NAME=VALUE")]
    serve_args: Vec<String>,
    #[arg(long = "serve-flag", value_name = "NAME")]
    serve_flags: Vec<String>,
}

pub(crate) fn parse(args: impl Iterator<Item = String>) -> Result<ParsedCli> {
    let argv = std::iter::once(OsString::from("optimum-advisor"))
        .chain(args.map(OsString::from))
        .collect::<Vec<_>>();
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(error)
            if matches!(
                error.kind(),
                ClapErrorKind::DisplayHelp | ClapErrorKind::DisplayVersion
            ) =>
        {
            return Ok(ParsedCli::Display(error.to_string()))
        }
        Err(error) => return Err(Error::usage(error.to_string())),
    };

    let invocation = match cli.command {
        CliCommand::Plan { config, overrides } => Invocation {
            kind: CommandKind::Plan,
            input: merged_input(config.as_deref(), overrides)?,
            execute: false,
            ..Invocation::default_paths()
        },
        CliCommand::Params {
            engine,
            image,
            pull_policy,
            allow_local_image,
            cache_dir,
            refresh,
            offline,
        } => Invocation {
            kind: CommandKind::Params,
            input: ConfigInput {
                engine: Some(engine),
                image,
                runtime: RuntimeInput {
                    pull_policy: Some(pull_policy),
                    allow_local_image: Some(allow_local_image),
                    ..RuntimeInput::default()
                },
                ..ConfigInput::default()
            },
            execute: true,
            parameter_cache_dir: cache_dir,
            refresh_parameters: refresh,
            offline_parameters: offline,
            ..Invocation::default_paths()
        },
        CliCommand::Hardware => Invocation {
            kind: CommandKind::Hardware,
            ..Invocation::default_paths()
        },
        CliCommand::Serve { config, overrides } => Invocation {
            kind: CommandKind::Serve,
            input: merged_input(config.as_deref(), overrides)?,
            execute: true,
            ..Invocation::default_paths()
        },
        CliCommand::Bench {
            config,
            results_dir,
            dry_run,
            overrides,
        } => Invocation {
            kind: CommandKind::Bench,
            input: merged_input(config.as_deref(), overrides)?,
            execute: !dry_run,
            results_dir,
            ..Invocation::default_paths()
        },
        CliCommand::Sweep {
            config,
            results_dir,
            dry_run,
        } => {
            let input = ConfigInput::try_from(load_config(&config)?)?;
            if input.sweep.is_none() {
                return Err(Error::validation(
                    "sweep requires a nonempty [sweep] configuration",
                ));
            }
            Invocation {
                kind: CommandKind::Sweep,
                input,
                execute: !dry_run,
                results_dir,
                ..Invocation::default_paths()
            }
        }
        CliCommand::Cleanup { run_id, dry_run } => Invocation {
            kind: CommandKind::Cleanup,
            cleanup_run_id: run_id,
            cleanup_dry_run: dry_run,
            ..Invocation::default_paths()
        },
        CliCommand::Mcp => Invocation {
            kind: CommandKind::Mcp,
            ..Invocation::default_paths()
        },
    };
    Ok(ParsedCli::Invocation(Box::new(invocation)))
}

impl Invocation {
    fn default_paths() -> Self {
        Self {
            kind: CommandKind::Hardware,
            input: ConfigInput::default(),
            execute: false,
            results_dir: PathBuf::from(DEFAULT_RESULTS_DIR),
            parameter_cache_dir: PathBuf::from(DEFAULT_PARAMETER_CACHE),
            refresh_parameters: false,
            offline_parameters: false,
            cleanup_run_id: None,
            cleanup_dry_run: false,
        }
    }
}

fn merged_input(
    config: Option<&std::path::Path>,
    overrides: ConfigOverrides,
) -> Result<ConfigInput> {
    let input = config
        .map(load_config)
        .transpose()?
        .map(ConfigInput::try_from)
        .transpose()?
        .unwrap_or_default();
    Ok(input.overlay(overrides.into_input()?))
}

impl ConfigOverrides {
    fn into_input(self) -> Result<ConfigInput> {
        let mut dynamic = self
            .serve_args
            .iter()
            .map(|value| parse_assignment(value, "--serve-arg"))
            .chain(
                self.serve_flags
                    .iter()
                    .map(|value| parse_flag(value, "--serve-flag")),
            )
            .collect::<Result<Vec<_>>>()?;
        reject_duplicate_dynamic(&dynamic)?;
        dynamic.sort_by(|left, right| left.name.cmp(&right.name));

        Ok(ConfigInput {
            engine: self.engine,
            image: self.image,
            model: self.model,
            metric: self.metric,
            runtime: RuntimeInput {
                gpus: self.gpus,
                gpu_devices: (!self.gpu_devices.is_empty()).then_some(self.gpu_devices),
                pull_policy: self.pull_policy,
                allow_local_image: self.allow_local_image.then_some(true),
                bind_host: self.bind_host,
                port: self.port,
                startup_timeout_secs: self.startup_timeout_secs,
                benchmark_timeout_secs: self.benchmark_timeout_secs,
                max_process_output_bytes: self.max_process_output_bytes,
                ..RuntimeInput::default()
            },
            benchmark: BenchmarkInput {
                dataset_name: self.dataset_name,
                num_prompts: self.num_prompts,
                request_rate: self.request_rate,
                max_concurrency: self.benchmark_max_concurrency,
                random_input_len: self.random_input_len,
                random_output_len: self.random_output_len,
            },
            candidate: CandidateOverrides {
                tensor_parallelism: self.tensor_parallelism,
                memory_fraction: self.memory_fraction,
                prefill_token_budget: self.prefill_token_budget,
                max_running_requests: self.max_running_requests,
            },
            correctness: CorrectnessInput {
                enabled: self.no_correctness.then_some(false),
                threshold: self.correctness_threshold,
                timeout_secs: self.correctness_timeout_secs,
            },
            model_memory: ModelMemoryInput {
                enabled: self.no_model_memory.then_some(false),
                required: self.require_model_memory.then_some(true),
                command: self.hf_mem_command,
                timeout_secs: self.hf_mem_timeout_secs,
            },
            leaderboard: LeaderboardInput {
                submit: self.leaderboard_submit.then_some(true),
                url: self.leaderboard_url,
            },
            serve_args: dynamic,
            sweep: None,
        })
    }
}

fn parse_assignment(value: &str, flag: &str) -> Result<DynamicArg> {
    let (name, value) = value
        .split_once('=')
        .ok_or_else(|| Error::usage(format!("{flag} expects NAME=VALUE")))?;
    if value.is_empty() {
        return Err(Error::usage(format!("{flag} value must not be empty")));
    }
    let name = canonical_name(name);
    validate_dynamic_name(&name)?;
    Ok(DynamicArg::value(name, value))
}

fn parse_flag(value: &str, flag: &str) -> Result<DynamicArg> {
    if value.contains('=') {
        return Err(Error::usage(format!("{flag} expects NAME without a value")));
    }
    let name = canonical_name(value);
    validate_dynamic_name(&name)?;
    Ok(DynamicArg::flag(name))
}

fn reject_duplicate_dynamic(arguments: &[DynamicArg]) -> Result<()> {
    let mut seen = HashSet::new();
    for argument in arguments {
        if !seen.insert(argument.name.as_str()) {
            return Err(Error::usage(format!(
                "duplicate CLI serving parameter: --{}",
                argument.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_help_without_an_error_exit() {
        let parsed = parse(["--help".to_string()].into_iter()).unwrap();
        assert!(matches!(parsed, ParsedCli::Display(_)));
    }

    #[test]
    fn rejects_duplicate_dynamic_arguments() {
        let error = parse(
            [
                "plan",
                "--engine",
                "vllm",
                "--model",
                "m",
                "--serve-arg",
                "kv-cache-dtype=fp8",
                "--serve-flag",
                "kv_cache_dtype",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .unwrap_err();
        assert_eq!(error.exit_code(), 2);
    }

    #[test]
    fn does_not_read_operational_configuration_from_the_environment() {
        std::env::set_var("OPTIMUM_ADVISOR_MODEL", "environment/model");
        let parsed = parse(
            ["plan", "--engine", "vllm", "--model", "cli/model"]
                .into_iter()
                .map(str::to_string),
        )
        .unwrap();
        std::env::remove_var("OPTIMUM_ADVISOR_MODEL");

        let ParsedCli::Invocation(invocation) = parsed else {
            panic!("expected invocation");
        };
        assert_eq!(invocation.input.model.as_deref(), Some("cli/model"));
    }
}
