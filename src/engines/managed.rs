use std::{ffi::OsString, net::IpAddr, path::Path, time::Duration};

use crate::{
    config::ExecutableConfig,
    domain::{candidate::DynamicArg, engine::Engine},
    error::{Error, ExecutionStage, Result},
    runtime::{docker::OwnedContainer, process::ProcessSpec, server::ReadinessProbe},
};

pub(crate) struct ManagedRunPlan {
    pub server: ProcessSpec,
    pub benchmark: ProcessSpec,
    pub readiness: ReadinessProbe,
}

pub(crate) fn managed_run_plan(
    config: &ExecutableConfig,
    run_id: &str,
    artifact_dir: &Path,
) -> Result<ManagedRunPlan> {
    validate_run_id(run_id)?;
    let server_container =
        OwnedContainer::new(format!("optimum-advisor-{run_id}-server"), run_id, "server");
    let benchmark_container = OwnedContainer::new(
        format!("optimum-advisor-{run_id}-benchmark"),
        run_id,
        "benchmark",
    );

    let mut server_args = docker_run_prefix(config, &server_container);
    server_args.extend([
        OsString::from("--ipc=host"),
        OsString::from("-p"),
        OsString::from(publish_binding(
            config.runtime.bind_host,
            config.runtime.port,
            container_port(config.engine),
        )),
    ]);
    if config.engine == Engine::Sglang {
        server_args.extend([OsString::from("--entrypoint"), OsString::from("python3")]);
    }
    server_args.push(OsString::from(&config.image.immutable));
    append_server_command(config, &mut server_args);
    let mut server = ProcessSpec::new("docker", server_args)
        .with_stage(ExecutionStage::Server)
        .with_artifacts(
            artifact_dir.join("server.stdout.log"),
            artifact_dir.join("server.stderr.log"),
        )
        .with_owned_container(server_container.clone());
    server.max_stdout_bytes = config.runtime.max_process_output_bytes;
    server.max_stderr_bytes = config.runtime.max_process_output_bytes;
    server.safe_display = safe_display(&server.program, &server.args);

    let mut benchmark_args = docker_run_prefix(config, &benchmark_container);
    benchmark_args.extend([
        OsString::from("--network"),
        OsString::from("host"),
        OsString::from("--entrypoint"),
        OsString::from(match config.engine {
            Engine::Vllm => "vllm",
            Engine::Sglang => "python3",
        }),
        OsString::from(&config.image.immutable),
    ]);
    append_benchmark_command(config, &mut benchmark_args);
    let mut benchmark = ProcessSpec::new("docker", benchmark_args)
        .with_stage(ExecutionStage::Benchmark)
        .with_timeout(Duration::from_secs(config.runtime.benchmark_timeout_secs))
        .with_artifacts(
            artifact_dir.join("benchmark.stdout.log"),
            artifact_dir.join("benchmark.stderr.log"),
        )
        .with_owned_container(benchmark_container);
    benchmark.max_stdout_bytes = config.runtime.max_process_output_bytes;
    benchmark.max_stderr_bytes = config.runtime.max_process_output_bytes;
    benchmark.safe_display = safe_display(&benchmark.program, &benchmark.args);

    Ok(ManagedRunPlan {
        server,
        benchmark,
        readiness: ReadinessProbe::new(
            config.runtime.bind_host,
            config.runtime.port,
            Some("/v1/models".to_string()),
            Duration::from_secs(config.runtime.startup_timeout_secs),
        ),
    })
}

fn docker_run_prefix(config: &ExecutableConfig, container: &OwnedContainer) -> Vec<OsString> {
    let gpu_selector = if config.runtime.gpu_devices.is_empty() {
        config.runtime.gpus.to_string()
    } else {
        format!("device={}", config.runtime.gpu_devices.join(","))
    };
    let mut args = vec![
        OsString::from("run"),
        OsString::from("--rm"),
        OsString::from("--gpus"),
        OsString::from(gpu_selector),
        OsString::from("--name"),
        OsString::from(&container.name),
    ];
    for (name, value) in &container.labels {
        args.extend([
            OsString::from("--label"),
            OsString::from(format!("{name}={value}")),
        ]);
    }
    args
}

fn append_server_command(config: &ExecutableConfig, args: &mut Vec<OsString>) {
    match config.engine {
        Engine::Vllm => {
            append_value(args, "--model", &config.model);
            append_value(
                args,
                "--tensor-parallel-size",
                config.candidate.tensor_parallelism,
            );
            append_value(
                args,
                "--gpu-memory-utilization",
                config.candidate.memory_fraction,
            );
            append_value(args, "--max-model-len", config.runtime.max_model_len);
            append_value(
                args,
                "--max-num-batched-tokens",
                config.candidate.prefill_token_budget,
            );
            append_value(
                args,
                "--max-num-seqs",
                config.candidate.max_running_requests,
            );
        }
        Engine::Sglang => {
            args.extend([OsString::from("-m"), OsString::from("sglang.launch_server")]);
            append_value(args, "--model-path", &config.model);
            append_value(args, "--host", "0.0.0.0");
            append_value(args, "--port", 30_000);
            append_value(args, "--context-length", config.runtime.max_model_len);
            append_value(args, "--tp-size", config.candidate.tensor_parallelism);
            append_value(
                args,
                "--mem-fraction-static",
                config.candidate.memory_fraction,
            );
            append_value(
                args,
                "--chunked-prefill-size",
                config.candidate.prefill_token_budget,
            );
            append_value(
                args,
                "--max-running-requests",
                config.candidate.max_running_requests,
            );
        }
    }
    append_dynamic(args, &config.serve_args);
}

fn append_benchmark_command(config: &ExecutableConfig, args: &mut Vec<OsString>) {
    match config.engine {
        Engine::Vllm => args.extend([
            OsString::from("bench"),
            OsString::from("serve"),
            OsString::from("--backend"),
            OsString::from("vllm"),
        ]),
        Engine::Sglang => args.extend([
            OsString::from("-m"),
            OsString::from("sglang.bench_serving"),
            OsString::from("--backend"),
            OsString::from("sglang"),
        ]),
    }
    append_value(args, "--model", &config.model);
    append_value(args, "--host", readiness_host(config.runtime.bind_host));
    append_value(args, "--port", config.runtime.port);
    if config.engine == Engine::Vllm {
        append_value(args, "--endpoint", "/v1/completions");
    }
    append_value(args, "--dataset-name", &config.benchmark.dataset_name);
    append_value(args, "--num-prompts", config.benchmark.num_prompts);
    append_value(args, "--request-rate", &config.benchmark.request_rate);
    append_value(args, "--max-concurrency", config.benchmark.max_concurrency);
    append_value(
        args,
        "--random-input-len",
        config.benchmark.random_input_len,
    );
    append_value(
        args,
        "--random-output-len",
        config.benchmark.random_output_len,
    );
    append_value(args, "--random-range-ratio", 0);
}

fn append_dynamic(args: &mut Vec<OsString>, dynamic: &[DynamicArg]) {
    for argument in dynamic {
        args.push(OsString::from(format!("--{}", argument.name)));
        if let Some(value) = &argument.value {
            args.push(OsString::from(value));
        }
    }
}

fn append_value(args: &mut Vec<OsString>, name: &str, value: impl ToString) {
    args.extend([OsString::from(name), OsString::from(value.to_string())]);
}

fn container_port(engine: Engine) -> u16 {
    match engine {
        Engine::Vllm => 8_000,
        Engine::Sglang => 30_000,
    }
}

fn readiness_host(host: IpAddr) -> String {
    match host {
        IpAddr::V4(address) if address.is_unspecified() => "127.0.0.1".to_string(),
        IpAddr::V6(address) if address.is_unspecified() => "::1".to_string(),
        address => address.to_string(),
    }
}

fn publish_binding(host: IpAddr, host_port: u16, container_port: u16) -> String {
    match host {
        IpAddr::V4(host) => format!("{host}:{host_port}:{container_port}"),
        IpAddr::V6(host) => format!("[{host}]:{host_port}:{container_port}"),
    }
}

fn validate_run_id(run_id: &str) -> Result<()> {
    if run_id.is_empty()
        || run_id.len() > 40
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(Error::validation(
            "run ID must be 1-40 ASCII letters, digits, hyphens, or underscores",
        ));
    }
    Ok(())
}

pub(crate) fn safe_display(program: &std::ffi::OsStr, args: &[OsString]) -> String {
    std::iter::once(program)
        .chain(args.iter().map(OsString::as_os_str))
        .map(|value| {
            let value = value.to_string_lossy();
            if value
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-_./:=,@[]".contains(&byte))
            {
                value.into_owned()
            } else {
                format!("'{}'", value.replace('\'', "'\"'\"'"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::{
        config::ConfigInput,
        domain::{engine::Engine, run::ResolvedImage},
    };

    #[test]
    fn every_docker_command_uses_the_same_immutable_image_and_owned_labels() {
        let mut input = ConfigInput::minimal(Engine::Vllm, "repo/model");
        input.runtime.gpu_devices = Some(vec!["1".into(), "3".into()]);
        input.runtime.gpus = Some(2);
        let config = input
            .normalize()
            .unwrap()
            .into_executable(ResolvedImage {
                requested: "repo/server:latest".into(),
                immutable: "repo/server@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                local_only: false,
            });
        let directory = tempdir().unwrap();

        let plan = managed_run_plan(&config, "run-1", directory.path()).unwrap();
        let server = strings(&plan.server.args);
        let benchmark = strings(&plan.benchmark.args);

        for arguments in [&server, &benchmark] {
            assert!(arguments.contains(&config.image.immutable));
            assert!(!arguments.contains(&config.image.requested));
        }
        let entrypoint = benchmark
            .iter()
            .position(|value| value == "--entrypoint")
            .unwrap();
        let image = benchmark
            .iter()
            .position(|value| value == &config.image.immutable)
            .unwrap();
        assert_eq!(benchmark[entrypoint + 1], "vllm");
        assert!(entrypoint < image);
        assert!(server.contains(&"device=1,3".to_string()));
        assert!(server.contains(&"optimum-advisor=true".to_string()));
        assert!(server.contains(&"optimum-advisor.run-id=run-1".to_string()));
        assert!(server.contains(&"optimum-advisor.role=server".to_string()));
        assert_eq!(
            plan.server.owned_container.as_ref().unwrap().run_id,
            "run-1"
        );
    }

    #[test]
    fn renders_ipv6_publish_syntax_and_engine_specific_candidate_flags() {
        let mut input = ConfigInput::minimal(Engine::Sglang, "repo/model");
        input.runtime.bind_host = Some("::1".parse().unwrap());
        input.candidate.tensor_parallelism = Some(2);
        input.runtime.gpus = Some(2);
        let config = input
            .normalize()
            .unwrap()
            .into_executable(ResolvedImage {
                requested: "server:tag".into(),
                immutable: "docker.io/library/server@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
                local_only: false,
            });
        let directory = tempdir().unwrap();

        let plan = managed_run_plan(&config, "run-2", directory.path()).unwrap();
        let server = strings(&plan.server.args);

        assert!(server.contains(&"[::1]:8000:30000".to_string()));
        let entrypoint = server
            .iter()
            .position(|value| value == "--entrypoint")
            .unwrap();
        let image = server
            .iter()
            .position(|value| value == &config.image.immutable)
            .unwrap();
        assert_eq!(server[entrypoint + 1], "python3");
        assert!(entrypoint < image);
        assert!(server.contains(&"--tp-size".to_string()));
        assert!(server.contains(&"--context-length".to_string()));
        assert!(server.contains(&"8192".to_string()));
        assert!(server.contains(&"--mem-fraction-static".to_string()));
        assert!(server.contains(&"--chunked-prefill-size".to_string()));
        assert!(server.contains(&"--max-running-requests".to_string()));
    }

    fn strings(values: &[std::ffi::OsString]) -> Vec<String> {
        values
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect()
    }
}
