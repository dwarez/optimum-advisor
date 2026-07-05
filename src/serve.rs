use crate::Result;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineArg {
    pub name: String,
    pub value: Option<String>,
}

impl EngineArg {
    pub fn assignment(value: &str) -> Result<Self> {
        let (name, arg_value) = value
            .split_once('=')
            .ok_or_else(|| format!("--serve-arg expects NAME=VALUE, got {value}"))?;
        Ok(Self {
            name: normalize_arg_name(name),
            value: Some(arg_value.to_string()),
        })
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
}
