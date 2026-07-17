use std::{collections::BTreeMap, fs, path::Path};

use serde::Deserialize;

use super::{
    BenchmarkInput, ConfigInput, CorrectnessInput, LeaderboardInput, ModelMemoryInput, RuntimeInput,
};
use crate::{
    domain::{
        candidate::{
            canonical_name, validate_dynamic_name, CandidateOverrides, DynamicArg, SweepSpec,
        },
        engine::{Engine, Metric},
    },
    error::{Error, ErrorKind, Result},
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConfigFile {
    pub schema_version: u32,
    pub engine: Option<Engine>,
    pub image: Option<String>,
    pub model: Option<String>,
    pub metric: Option<Metric>,
    #[serde(default)]
    pub runtime: RuntimeInput,
    #[serde(default)]
    pub benchmark: BenchmarkInput,
    #[serde(default)]
    pub candidate: CandidateOverrides,
    #[serde(default)]
    pub correctness: CorrectnessInput,
    #[serde(default)]
    pub model_memory: ModelMemoryInput,
    #[serde(default)]
    pub leaderboard: LeaderboardInput,
    #[serde(default)]
    pub serve: toml::Table,
    pub sweep: Option<SweepInput>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SweepInput {
    pub max_trials: Option<usize>,
    pub tensor_parallelism: Option<Vec<usize>>,
    pub memory_fraction: Option<Vec<f64>>,
    pub prefill_token_budget: Option<Vec<u32>>,
    pub max_running_requests: Option<Vec<u32>>,
    pub serve: toml::Table,
}

pub(crate) fn parse_config_text(text: &str) -> Result<ConfigFile> {
    let file = toml::from_str::<ConfigFile>(text).map_err(|source| {
        let mut error = Error::configuration(format!("invalid schema-v2 TOML: {source}"));
        error.source = Some(Box::new(source));
        error
    })?;
    if file.schema_version != 2 {
        return Err(Error::configuration(format!(
            "unsupported schema_version {}; expected 2",
            file.schema_version
        )));
    }
    Ok(file)
}

pub(crate) fn load_config(path: &Path) -> Result<ConfigFile> {
    let text = fs::read_to_string(path).map_err(|source| {
        let mut error = Error::new(
            ErrorKind::Io,
            None,
            format!("failed to read configuration {}", path.display()),
        );
        error.context.operation = Some("read configuration".to_string());
        error.context.path = Some(path.to_path_buf());
        error.source = Some(Box::new(source));
        error
    })?;
    parse_config_text(&text).map_err(|mut error| {
        error.context.path = Some(path.to_path_buf());
        error
    })
}

pub(super) fn into_input(file: ConfigFile) -> Result<ConfigInput> {
    if file.schema_version != 2 {
        return Err(Error::configuration(format!(
            "unsupported schema_version {}; expected 2",
            file.schema_version
        )));
    }
    let serve_args = parse_serve_table(file.serve, "serve")?;
    let sweep = file.sweep.map(parse_sweep).transpose()?;
    Ok(ConfigInput {
        schema_version: Some(file.schema_version),
        engine: file.engine,
        image: file.image,
        model: file.model,
        metric: file.metric,
        runtime: file.runtime,
        benchmark: file.benchmark,
        candidate: file.candidate,
        correctness: file.correctness,
        model_memory: file.model_memory,
        leaderboard: file.leaderboard,
        serve_args,
        sweep,
    })
}

fn parse_sweep(input: SweepInput) -> Result<SweepSpec> {
    let mut serve = BTreeMap::new();
    for (raw_name, value) in input.serve {
        let name = canonical_name(&raw_name);
        validate_dynamic_name(&name)?;
        let values = match value {
            toml::Value::Array(values) => values,
            _ => {
                return Err(Error::configuration(format!(
                    "sweep.serve.{raw_name} must be an array"
                )))
            }
        };
        let values = values
            .into_iter()
            .map(|value| scalar_value(value, &format!("sweep.serve.{raw_name}")))
            .collect::<Result<Vec<_>>>()?;
        if serve.insert(name.clone(), values).is_some() {
            return Err(Error::configuration(format!(
                "duplicate canonical sweep argument: {name}"
            )));
        }
    }
    Ok(SweepSpec {
        max_trials: input.max_trials.unwrap_or(256),
        tensor_parallelism: input.tensor_parallelism,
        memory_fraction: input.memory_fraction,
        prefill_token_budget: input.prefill_token_budget,
        max_running_requests: input.max_running_requests,
        serve,
    })
}

fn parse_serve_table(table: toml::Table, section: &str) -> Result<Vec<DynamicArg>> {
    let mut arguments = BTreeMap::new();
    for (raw_name, value) in table {
        let name = canonical_name(&raw_name);
        validate_dynamic_name(&name)?;
        let value = scalar_value(value, &format!("{section}.{raw_name}"))?;
        if arguments.insert(name.clone(), value).is_some() {
            return Err(Error::configuration(format!(
                "duplicate canonical engine argument: {name}"
            )));
        }
    }
    Ok(arguments
        .into_iter()
        .map(|(name, value)| DynamicArg { name, value })
        .collect())
}

fn scalar_value(value: toml::Value, location: &str) -> Result<Option<String>> {
    match value {
        toml::Value::String(value) => Ok(Some(value)),
        toml::Value::Integer(value) => Ok(Some(value.to_string())),
        toml::Value::Float(value) if value.is_finite() => Ok(Some(value.to_string())),
        toml::Value::Boolean(true) => Ok(None),
        toml::Value::Boolean(false) => Ok(Some("false".to_string())),
        toml::Value::Float(_) => Err(Error::configuration(format!("{location} must be finite"))),
        _ => Err(Error::configuration(format!(
            "{location} must be a string, number, or boolean"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unknown_and_duplicate_toml_keys() {
        assert!(
            parse_config_text("schema_version=2\nengine='vllm'\nmodel='m'\nunknown=1").is_err()
        );
        assert!(
            parse_config_text("schema_version=2\nengine='vllm'\nengine='sglang'\nmodel='m'")
                .is_err()
        );
    }

    #[test]
    fn permits_engine_and_model_to_arrive_from_a_later_source() {
        let file = parse_config_text("schema_version=2").unwrap();

        assert!(file.engine.is_none());
        assert!(file.model.is_none());
    }
}
