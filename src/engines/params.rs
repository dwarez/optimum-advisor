use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Serialize;

use crate::engine::Engine;
use crate::engines::EngineAdapter;
use crate::runner::ProcessSpec;
use crate::serve::{EngineArg, ParamKind, ParameterSpec};
use crate::Result;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ParameterSchema {
    pub engine: Engine,
    pub image: String,
    pub parameters: Vec<ParameterSpec>,
}

impl ParameterSchema {
    pub fn validate_args(&self, args: &[EngineArg]) -> Result<()> {
        for arg in args {
            if !self.parameters.iter().any(|spec| spec.cli == arg.name) {
                return Err(format!(
                    "unknown serving parameter for {}: {}",
                    self.engine, arg.name
                ));
            }
        }
        Ok(())
    }
}

pub fn inspect_command(adapter: &dyn EngineAdapter, image: String) -> ProcessSpec {
    adapter.help_command(image)
}

pub fn load_or_inspect(
    adapter: &dyn EngineAdapter,
    image: String,
    cache_dir: &Path,
    refresh: bool,
) -> Result<ParameterSchema> {
    let path = cache_path(cache_dir, adapter.engine(), &image);
    if !refresh && path.exists() {
        let cached = load_cached(adapter.engine(), image.clone(), &path)?;
        if has_serving_parameters(&cached.parameters) {
            return Ok(cached);
        }
        eprintln!("ignoring stale parameter cache: {}", path.display());
    }

    ensure_image_available(&image)?;
    let command = adapter.help_command(image.clone());
    let output = Command::new(&command.program)
        .args(&command.args)
        .output()
        .map_err(|err| format!("failed to inspect serving parameters: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "parameter inspection failed with status {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let mut text = String::from_utf8_lossy(&output.stdout).to_string();
    if !output.stderr.is_empty() {
        text.push('\n');
        text.push_str(&String::from_utf8_lossy(&output.stderr));
    }
    let parameters = parse_parameter_output(&text);
    if parameters.is_empty() {
        return Err("parameter inspection returned no long flags".to_string());
    }

    fs::create_dir_all(cache_dir).map_err(|err| {
        format!(
            "failed to create parameter cache {}: {err}",
            cache_dir.display()
        )
    })?;
    fs::write(&path, serialize_specs(&parameters))
        .map_err(|err| format!("failed to write parameter cache {}: {err}", path.display()))?;

    Ok(ParameterSchema {
        engine: adapter.engine(),
        image,
        parameters,
    })
}

pub fn load_cached_or_hint(
    adapter: &dyn EngineAdapter,
    image: String,
    cache_dir: &Path,
) -> Result<ParameterSchema> {
    let path = cache_path(cache_dir, adapter.engine(), &image);
    if path.exists() {
        return load_cached(adapter.engine(), image, &path);
    }
    Err(format!(
        "no cached parameter schema for {} image {image}; run `optimum-advisor params --engine {} --image {image} --execute` first, or add --execute to this command",
        adapter.engine(),
        adapter.engine()
    ))
}

pub fn parse_help_parameters(help: &str) -> Vec<ParameterSpec> {
    let mut specs = BTreeMap::new();
    let help = strip_ansi(help);
    for line in help.lines() {
        let trimmed = line.trim_start();
        for cli in long_flags(trimmed) {
            if cli == "--help" {
                continue;
            }
            let canonical = cli.trim_start_matches("--").replace('-', "_");
            let kind = if cli.starts_with("--no-") || trimmed.contains(" bool flag") {
                ParamKind::Flag
            } else {
                ParamKind::Value
            };
            specs
                .entry(cli.clone())
                .or_insert_with(|| ParameterSpec::new(canonical, cli, kind));
        }
    }
    specs.into_values().collect()
}

fn parse_parameter_output(output: &str) -> Vec<ParameterSpec> {
    let structured = parse_tab_parameters(output);
    if structured.is_empty() {
        parse_help_parameters(output)
    } else {
        structured
    }
}

fn parse_tab_parameters(output: &str) -> Vec<ParameterSpec> {
    let mut specs = BTreeMap::new();
    for line in output.lines() {
        let Some((cli, kind)) = line.trim().split_once('\t') else {
            continue;
        };
        if !cli.starts_with("--") || cli == "--help" {
            continue;
        }
        let canonical = cli.trim_start_matches("--").replace('-', "_");
        let kind = if kind == "bool flag" {
            ParamKind::Flag
        } else {
            ParamKind::Value
        };
        specs
            .entry(cli.to_string())
            .or_insert_with(|| ParameterSpec::new(canonical, cli.to_string(), kind));
    }
    specs.into_values().collect()
}

fn has_serving_parameters(parameters: &[ParameterSpec]) -> bool {
    parameters.iter().any(|spec| spec.cli != "--help")
}

fn long_flags(line: &str) -> Vec<String> {
    let chars = line.chars().collect::<Vec<_>>();
    let mut flags = Vec::new();
    let mut index = 0;
    while index + 1 < chars.len() {
        if chars[index] != '-' || chars[index + 1] != '-' {
            index += 1;
            continue;
        }

        let start = index;
        index += 2;
        while index < chars.len()
            && (chars[index].is_ascii_alphanumeric() || chars[index] == '-' || chars[index] == '_')
        {
            index += 1;
        }
        if index > start + 2 {
            flags.push(chars[start..index].iter().collect());
        }
    }
    flags
}

fn strip_ansi(text: &str) -> String {
    let mut stripped = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for next in chars.by_ref() {
                if ('@'..='~').contains(&next) {
                    break;
                }
            }
            continue;
        }
        stripped.push(ch);
    }
    stripped
}

pub fn cache_path(cache_dir: &Path, engine: Engine, image: &str) -> PathBuf {
    cache_dir.join(format!("{}-{}.tsv", engine, sanitize(image)))
}

fn serialize_specs(specs: &[ParameterSpec]) -> String {
    specs
        .iter()
        .map(|spec| {
            let kind = match spec.kind {
                ParamKind::Value => "value",
                ParamKind::Flag => "flag",
            };
            format!("{kind}\t{}\t{}", spec.cli, spec.canonical)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn deserialize_specs(text: &str) -> Vec<ParameterSpec> {
    text.lines()
        .filter_map(|line| {
            let mut parts = line.splitn(3, '\t');
            let kind = match parts.next()? {
                "flag" => ParamKind::Flag,
                _ => ParamKind::Value,
            };
            let cli = parts.next()?.to_string();
            let canonical = parts.next()?.to_string();
            Some(ParameterSpec::new(canonical, cli, kind))
        })
        .collect()
}

fn load_cached(engine: Engine, image: String, path: &Path) -> Result<ParameterSchema> {
    let text = fs::read_to_string(path)
        .map_err(|err| format!("failed to read parameter cache {}: {err}", path.display()))?;
    Ok(ParameterSchema {
        engine,
        image,
        parameters: deserialize_specs(&text),
    })
}

fn ensure_image_available(image: &str) -> Result<()> {
    let local = Command::new("docker")
        .args(["image", "inspect", image])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|err| format!("failed to check Docker image {image}: {err}"))?;
    if local.success() {
        eprintln!("using local image: {image}");
        return Ok(());
    }

    eprintln!("pulling image: {image}");
    let pulled = Command::new("docker")
        .args(["pull", image])
        .status()
        .map_err(|err| format!("failed to pull Docker image {image}: {err}"))?;
    if pulled.success() {
        Ok(())
    } else {
        Err(format!("docker pull failed for {image}: {pulled}"))
    }
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_long_flags_from_help() {
        let specs = parse_help_parameters(
            "
usage: vllm serve [-h] [--model MODEL] [--enable-lora] [--kv-cache-dtype {auto,fp8}]
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
",
        );

        assert!(specs.iter().any(|spec| spec.cli == "--model"));
        assert!(specs
            .iter()
            .any(|spec| spec.cli == "--tensor-parallel-size"));
        assert!(specs.iter().any(|spec| spec.cli == "--kv-cache-dtype"));
    }

    #[test]
    fn parses_decorated_help_flags() {
        let specs = parse_help_parameters(
            "\x1b[1;36m--model\x1b[0m TEXT\n[--kv-cache-dtype={auto,fp8}]\n--help.",
        );

        assert!(specs.iter().any(|spec| spec.cli == "--model"));
        assert!(specs.iter().any(|spec| spec.cli == "--kv-cache-dtype"));
        assert!(!specs.iter().any(|spec| spec.cli == "--help"));
    }

    #[test]
    fn parses_structured_runtime_parameters() {
        let specs = parse_parameter_output(
            "warning: ignore me --not-a-param\n--model\tvalue\n--enable-lora\tbool flag\n--help\tbool flag\n",
        );

        assert!(specs.iter().any(|spec| spec.cli == "--model"));
        assert!(specs
            .iter()
            .any(|spec| { spec.cli == "--enable-lora" && matches!(spec.kind, ParamKind::Flag) }));
        assert!(!specs.iter().any(|spec| spec.cli == "--not-a-param"));
        assert!(!specs.iter().any(|spec| spec.cli == "--help"));
    }

    #[test]
    fn validates_known_arguments() {
        let schema = ParameterSchema {
            engine: Engine::Vllm,
            image: "img".to_string(),
            parameters: vec![ParameterSpec::new("model", "--model", ParamKind::Value)],
        };

        assert!(schema
            .validate_args(&[EngineArg::assignment("model=m").unwrap()])
            .is_ok());
        assert!(schema
            .validate_args(&[EngineArg::assignment("nope=x").unwrap()])
            .is_err());
    }
}
