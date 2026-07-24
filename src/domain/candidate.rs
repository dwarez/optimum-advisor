use std::collections::{BTreeMap, HashSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Candidate {
    #[schemars(range(min = 1))]
    pub tensor_parallelism: usize,
    #[schemars(range(min = 0.0, max = 1.0))]
    pub memory_fraction: f64,
    #[schemars(range(min = 1))]
    pub prefill_token_budget: u32,
    #[schemars(range(min = 1))]
    pub max_running_requests: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct CandidateOverrides {
    /// Number of GPUs assigned to one engine replica. This owns engine flags
    /// such as `--tensor-parallel-size` and `--tp-size`.
    #[schemars(range(min = 1))]
    pub tensor_parallelism: Option<usize>,
    /// Fraction of GPU memory available to the engine. Set this instead of
    /// passing `gpu-memory-utilization` or `mem-fraction-static` in `serve_args`.
    #[schemars(range(min = 0.0, max = 1.0))]
    pub memory_fraction: Option<f64>,
    /// Maximum tokens admitted in one prefill batch. Set this instead of
    /// passing `max-num-batched-tokens` or `chunked-prefill-size` in `serve_args`.
    #[schemars(range(min = 1))]
    pub prefill_token_budget: Option<u32>,
    /// Maximum concurrent requests admitted by the engine. Set this instead of
    /// passing `max-num-seqs` or `max-running-requests` in `serve_args`.
    #[schemars(range(min = 1))]
    pub max_running_requests: Option<u32>,
}

impl CandidateOverrides {
    pub(crate) fn apply_to(&self, candidate: &mut Candidate) {
        if let Some(value) = self.tensor_parallelism {
            candidate.tensor_parallelism = value;
        }
        if let Some(value) = self.memory_fraction {
            candidate.memory_fraction = normalize_zero(value);
        }
        if let Some(value) = self.prefill_token_budget {
            candidate.prefill_token_budget = value;
        }
        if let Some(value) = self.max_running_requests {
            candidate.max_running_requests = value;
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DynamicArg {
    #[schemars(length(min = 1))]
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

impl DynamicArg {
    pub(crate) fn value(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: canonical_name(&name.into()),
            value: Some(value.into()),
        }
    }

    pub(crate) fn flag(name: impl Into<String>) -> Self {
        Self {
            name: canonical_name(&name.into()),
            value: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CandidateSpec {
    #[serde(flatten)]
    pub candidate: Candidate,
    #[serde(default)]
    pub serve_args: Vec<DynamicArg>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SweepSpec {
    #[schemars(range(min = 1))]
    pub max_trials: usize,
    #[schemars(range(min = 1))]
    pub max_parallel_trials: Option<usize>,
    pub tensor_parallelism: Option<Vec<usize>>,
    pub memory_fraction: Option<Vec<f64>>,
    pub prefill_token_budget: Option<Vec<u32>>,
    pub max_running_requests: Option<Vec<u32>>,
    pub serve: BTreeMap<String, Vec<Option<String>>>,
}

impl SweepSpec {
    pub(crate) fn candidates(&self, base: &CandidateSpec) -> Result<Vec<CandidateSpec>> {
        if self.max_trials == 0 {
            return Err(Error::validation(
                "sweep max_trials must be greater than zero",
            ));
        }

        if self.max_parallel_trials == Some(0) {
            return Err(Error::validation(
                "sweep.max_parallel_trials must be greater than zero",
            ));
        }
        let mut dimension_sizes = Vec::new();
        let mut has_real_dimension = false;
        collect_dimension_size(
            self.tensor_parallelism.as_deref(),
            "tensor_parallelism",
            &mut dimension_sizes,
            &mut has_real_dimension,
        )?;
        collect_float_dimension_size(
            self.memory_fraction.as_deref(),
            "memory_fraction",
            &mut dimension_sizes,
            &mut has_real_dimension,
        )?;
        collect_dimension_size(
            self.prefill_token_budget.as_deref(),
            "prefill_token_budget",
            &mut dimension_sizes,
            &mut has_real_dimension,
        )?;
        collect_dimension_size(
            self.max_running_requests.as_deref(),
            "max_running_requests",
            &mut dimension_sizes,
            &mut has_real_dimension,
        )?;

        let mut canonical_serve = BTreeMap::new();
        for (name, values) in &self.serve {
            if values.is_empty() {
                return Err(Error::validation(format!(
                    "sweep.serve.{name} must not be empty"
                )));
            }
            let canonical = canonical_name(name);
            validate_dynamic_name(&canonical)?;
            if canonical_serve
                .insert(canonical.clone(), values.clone())
                .is_some()
            {
                return Err(Error::validation(format!(
                    "duplicate canonical sweep argument: {canonical}"
                )));
            }
            let unique = values.iter().collect::<HashSet<_>>().len();
            has_real_dimension |= unique > 1;
            dimension_sizes.push(values.len());
        }

        if dimension_sizes.is_empty() || !has_real_dimension {
            return Err(Error::validation(
                "sweep requires at least one dimension with multiple distinct values",
            ));
        }

        checked_trial_product(dimension_sizes, self.max_trials)?;

        let mut candidates = vec![base.clone()];
        if let Some(values) = &self.tensor_parallelism {
            candidates = expand(candidates, values, |spec, value| {
                spec.candidate.tensor_parallelism = *value;
            });
        }
        if let Some(values) = &self.memory_fraction {
            candidates = expand(candidates, values, |spec, value| {
                spec.candidate.memory_fraction = normalize_zero(*value);
            });
        }
        if let Some(values) = &self.prefill_token_budget {
            candidates = expand(candidates, values, |spec, value| {
                spec.candidate.prefill_token_budget = *value;
            });
        }
        if let Some(values) = &self.max_running_requests {
            candidates = expand(candidates, values, |spec, value| {
                spec.candidate.max_running_requests = *value;
            });
        }
        for (name, values) in canonical_serve {
            candidates = expand(candidates, &values, |spec, value| {
                set_dynamic_arg(spec, &name, value.clone());
            });
        }

        let mut seen = HashSet::with_capacity(candidates.len());
        candidates.retain(|candidate| seen.insert(candidate_key(candidate)));
        Ok(candidates)
    }
}

fn collect_dimension_size<T: Eq + std::hash::Hash>(
    values: Option<&[T]>,
    name: &str,
    sizes: &mut Vec<usize>,
    has_real_dimension: &mut bool,
) -> Result<()> {
    let Some(values) = values else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(Error::validation(format!("sweep.{name} must not be empty")));
    }
    *has_real_dimension |= values.iter().collect::<HashSet<_>>().len() > 1;
    sizes.push(values.len());
    Ok(())
}

fn collect_float_dimension_size(
    values: Option<&[f64]>,
    name: &str,
    sizes: &mut Vec<usize>,
    has_real_dimension: &mut bool,
) -> Result<()> {
    let Some(values) = values else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(Error::validation(format!("sweep.{name} must not be empty")));
    }
    let unique = values
        .iter()
        .map(|value| normalize_zero(*value).to_bits())
        .collect::<HashSet<_>>()
        .len();
    *has_real_dimension |= unique > 1;
    sizes.push(values.len());
    Ok(())
}

fn expand<T, F>(current: Vec<CandidateSpec>, values: &[T], mut apply: F) -> Vec<CandidateSpec>
where
    F: FnMut(&mut CandidateSpec, &T),
{
    let mut expanded = Vec::with_capacity(current.len().saturating_mul(values.len()));
    for candidate in current {
        for value in values {
            let mut next = candidate.clone();
            apply(&mut next, value);
            expanded.push(next);
        }
    }
    expanded
}

fn set_dynamic_arg(spec: &mut CandidateSpec, name: &str, value: Option<String>) {
    if let Some(argument) = spec.serve_args.iter_mut().find(|arg| arg.name == name) {
        argument.value = value;
    } else {
        spec.serve_args.push(DynamicArg {
            name: name.to_string(),
            value,
        });
        spec.serve_args
            .sort_by(|left, right| left.name.cmp(&right.name));
    }
}

fn checked_trial_product(
    sizes: impl IntoIterator<Item = usize>,
    max_trials: usize,
) -> Result<usize> {
    let product = sizes.into_iter().try_fold(1usize, |total, size| {
        total
            .checked_mul(size)
            .ok_or_else(|| Error::validation("sweep candidate count overflowed usize"))
    })?;
    if product > max_trials {
        return Err(Error::validation(format!(
            "sweep expands to {product} candidates, exceeding max_trials {max_trials}"
        )));
    }
    Ok(product)
}

fn candidate_key(spec: &CandidateSpec) -> String {
    let mut key = format!(
        "{}:{}:{}:{}",
        spec.candidate.tensor_parallelism,
        normalize_zero(spec.candidate.memory_fraction).to_bits(),
        spec.candidate.prefill_token_budget,
        spec.candidate.max_running_requests
    );
    for argument in &spec.serve_args {
        key.push('\u{1f}');
        key.push_str(&argument.name);
        key.push('\u{1e}');
        if let Some(value) = &argument.value {
            key.push_str(value);
        }
    }
    key
}

pub(crate) fn canonical_name(name: &str) -> String {
    name.trim()
        .trim_start_matches('-')
        .to_ascii_lowercase()
        .replace('_', "-")
}

pub(crate) fn validate_dynamic_name(name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '-')
    {
        return Err(Error::validation(format!(
            "invalid dynamic engine argument name: {name:?}"
        )));
    }
    if is_reserved_dynamic_name(name) {
        let message = replacement_for_reserved_dynamic_name(name).map_or_else(
            || format!("dynamic argument {name:?} is owned by normalized configuration"),
            |field| {
                format!(
                    "dynamic argument {name:?} is owned by normalized configuration; \
                     set {field} instead"
                )
            },
        );
        return Err(Error::validation(message));
    }
    Ok(())
}

fn replacement_for_reserved_dynamic_name(name: &str) -> Option<&'static str> {
    match name {
        "model" | "model-path" | "served-model-name" => Some("model"),
        "port" => Some("runtime.port"),
        "tensor-parallel-size" | "tensor-parallelism" | "tp-size" | "tp" => {
            Some("candidate.tensor_parallelism")
        }
        "mem-fraction-static" | "memory-fraction" | "gpu-memory-utilization" => {
            Some("candidate.memory_fraction")
        }
        "max-model-len" => Some("runtime.max_model_len"),
        "chunked-prefill-size" | "max-num-batched-tokens" | "prefill-token-budget" => {
            Some("candidate.prefill_token_budget")
        }
        "max-num-seqs" | "max-running-requests" => Some("candidate.max_running_requests"),
        _ => None,
    }
}

fn is_reserved_dynamic_name(name: &str) -> bool {
    matches!(
        name,
        "model"
            | "model-path"
            | "served-model-name"
            | "host"
            | "port"
            | "tensor-parallel-size"
            | "tensor-parallelism"
            | "tp-size"
            | "tp"
            | "pipeline-parallel-size"
            | "data-parallel-size"
            | "mem-fraction-static"
            | "memory-fraction"
            | "gpu-memory-utilization"
            | "max-model-len"
            | "chunked-prefill-size"
            | "max-num-batched-tokens"
            | "prefill-token-budget"
            | "max-num-seqs"
            | "max-running-requests"
    )
}

pub(crate) fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expansion_is_deterministic_and_last_dimension_varies_fastest() {
        let base = CandidateSpec {
            candidate: Candidate {
                tensor_parallelism: 1,
                memory_fraction: 0.9,
                prefill_token_budget: 8192,
                max_running_requests: 256,
            },
            serve_args: Vec::new(),
        };
        let sweep = SweepSpec {
            max_trials: 256,
            max_parallel_trials: None,
            tensor_parallelism: Some(vec![1, 2]),
            memory_fraction: Some(vec![0.8, 0.9]),
            prefill_token_budget: None,
            max_running_requests: None,
            serve: BTreeMap::from([(
                "kv-cache-dtype".to_string(),
                vec![Some("auto".to_string()), Some("fp8".to_string())],
            )]),
        };

        let candidates = sweep.candidates(&base).unwrap();

        assert_eq!(candidates.len(), 8);
        assert_eq!(candidates[0].candidate.tensor_parallelism, 1);
        assert_eq!(candidates[0].candidate.memory_fraction, 0.8);
        assert_eq!(
            candidates[0].serve_args,
            vec![DynamicArg::value("kv-cache-dtype", "auto")]
        );
        assert_eq!(
            candidates[1].serve_args,
            vec![DynamicArg::value("kv-cache-dtype", "fp8")]
        );
        assert_eq!(candidates[2].candidate.memory_fraction, 0.9);
        assert_eq!(candidates[4].candidate.tensor_parallelism, 2);
    }

    #[test]
    fn rejects_zero_parallel_trial_cap() {
        let sweep = SweepSpec {
            max_trials: 1,
            max_parallel_trials: Some(0),
            ..SweepSpec::default()
        };
        let base = CandidateSpec {
            candidate: Candidate {
                tensor_parallelism: 1,
                memory_fraction: 0.9,
                prefill_token_budget: 8192,
                max_running_requests: 256,
            },
            serve_args: Vec::new(),
        };

        let error = sweep.candidates(&base).unwrap_err();

        assert!(error.to_string().contains("max_parallel_trials"));
    }
    #[test]
    fn rejects_trial_count_overflow_before_allocation() {
        let error = checked_trial_product([usize::MAX, 2], usize::MAX).unwrap_err();

        assert!(error.to_string().contains("overflow"));
    }
}
