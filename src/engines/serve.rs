use crate::Result;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineArg {
    pub name: String,
    pub value: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingSweepParam {
    pub name: String,
    pub values: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ServingParamSweep {
    pub parameters: Vec<ServingSweepParam>,
}

impl EngineArg {
    pub fn value(name: impl Into<String>, value: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            name: normalize_arg_name(&name),
            value: Some(value.into()),
        }
    }

    pub fn assignment(value: &str) -> Result<Self> {
        let (name, arg_value) = value
            .split_once('=')
            .ok_or_else(|| format!("--serve-arg expects NAME=VALUE, got {value}"))?;
        Ok(Self::value(name, arg_value))
    }

    pub fn flag(value: &str) -> Self {
        Self {
            name: normalize_arg_name(value),
            value: None,
        }
    }

    pub fn append_to(&self, args: &mut Vec<String>) {
        args.push(self.name.clone());
        if let Some(value) = &self.value {
            args.push(value.clone());
        }
    }
}

impl ServingSweepParam {
    pub fn new(name: impl Into<String>, values: Vec<String>) -> Self {
        let name = name.into();
        Self {
            name: normalize_arg_name(&name),
            values,
        }
    }
}

impl ServingParamSweep {
    pub fn push(&mut self, name: impl Into<String>, values: Vec<String>) {
        self.parameters.push(ServingSweepParam::new(name, values));
    }

    pub fn combinations(&self) -> Vec<Vec<EngineArg>> {
        let mut combinations = vec![Vec::new()];
        for param in &self.parameters {
            let mut next = Vec::new();
            for existing in &combinations {
                for value in &param.values {
                    let mut combo = existing.clone();
                    combo.push(EngineArg::value(param.name.clone(), value.clone()));
                    next.push(combo);
                }
            }
            combinations = next;
        }
        combinations
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Value,
    Flag,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSpec {
    pub canonical: String,
    pub cli: String,
    pub kind: ParamKind,
}

impl ParameterSpec {
    pub fn new(canonical: impl Into<String>, cli: impl Into<String>, kind: ParamKind) -> Self {
        Self {
            canonical: canonical.into(),
            cli: cli.into(),
            kind,
        }
    }
}

fn normalize_arg_name(value: &str) -> String {
    if value.starts_with("--") {
        value.to_string()
    } else {
        format!("--{value}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_assignment() {
        let arg = EngineArg::assignment("dtype=bfloat16").unwrap();
        assert_eq!(arg.name, "--dtype");
        assert_eq!(arg.value.as_deref(), Some("bfloat16"));
    }

    #[test]
    fn serving_sweep_builds_combinations() {
        let mut sweep = ServingParamSweep::default();
        sweep.push(
            "tensor-parallel-size",
            vec!["1".to_string(), "2".to_string()],
        );
        sweep.push(
            "gpu-memory-utilization",
            vec!["0.8".to_string(), "0.9".to_string()],
        );

        let combinations = sweep.combinations();

        assert_eq!(combinations.len(), 4);
        assert!(combinations.iter().any(|combo| {
            combo
                == &vec![
                    EngineArg::value("tensor-parallel-size", "2"),
                    EngineArg::value("gpu-memory-utilization", "0.9"),
                ]
        }));
    }
}
