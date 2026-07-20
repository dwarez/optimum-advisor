use std::{
    fs,
    io::Write,
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use serde_json::{json, Value};
use tempfile::TempDir;

fn command() -> Command {
    Command::new(env!("CARGO_BIN_EXE_optimum-advisor"))
}

fn run(args: &[&str]) -> Output {
    command()
        .args(args)
        .output()
        .expect("failed to run optimum-advisor")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn write_config(directory: &TempDir, body: &str) -> PathBuf {
    let path = directory.path().join("config.toml");
    fs::write(&path, body).unwrap();
    path
}

fn report_path(results: &Path) -> PathBuf {
    let run_dirs = fs::read_dir(results)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(run_dirs.len(), 1, "expected exactly one run directory");
    run_dirs[0].join("report.json")
}

#[test]
fn root_help_lists_the_canonical_commands() {
    let output = run(&["--help"]);
    let text = stdout(&output);

    assert!(output.status.success(), "{}", stderr(&output));
    for command in [
        "plan", "params", "hardware", "serve", "bench", "sweep", "cleanup", "mcp",
    ] {
        assert!(text.contains(command), "missing {command} in {text}");
    }
    assert!(!text.contains("--max-model-len"));
}

#[test]
fn missing_command_and_unknown_root_flags_are_usage_errors() {
    let missing = run(&[]);
    assert_eq!(missing.status.code(), Some(2));
    assert!(stderr(&missing).contains("Usage:"));

    let unknown = run(&["--wat"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(stderr(&unknown).contains("unexpected argument '--wat'"));
}

#[test]
fn plan_is_a_secret_safe_deterministic_dry_run() {
    let output = run(&[
        "plan",
        "--engine",
        "vllm",
        "--model",
        "repo/model",
        "--image",
        "repo/image@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "--tensor-parallelism",
        "2",
        "--gpus",
        "2",
        "--serve-arg",
        "reasoning-parser=deepseek",
    ]);
    let text = stdout(&output);

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(text.contains("serve: docker run"));
    assert!(text.contains("--tensor-parallel-size 2"));
    assert!(text.contains("--reasoning-parser deepseek"));
    assert!(text.contains("bench: docker run"));
    assert!(text.contains("correctness: lighteval endpoint"));
    assert!(!text.contains("hf_"));
}

const HF_JOBS_CONFIG: &str = r#"
schema_version = 2
engine = "vllm"
model = "repo/model"
metric = "tps"

[runtime]
gpus = 1
max_model_len = 4096

[benchmark]
num_prompts = 4

[candidate]
tensor_parallelism = 1

[correctness]
enabled = true
threshold = 0.5
"#;

#[test]
fn hf_jobs_submission_is_a_rendered_dry_run() {
    let directory = TempDir::new().unwrap();
    let config = write_config(&directory, HF_JOBS_CONFIG);
    let config = config.to_str().unwrap();

    // Without a bucket: results are dumped to the job logs.
    let output = run(&[
        "bench",
        "--config",
        config,
        "--on",
        "hf-jobs",
        "--hf-flavor",
        "a10g-large",
        "--dry-run",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(
        text.starts_with("hf jobs run --flavor a10g-large"),
        "{text}"
    );
    assert!(text.contains("vllm/vllm-openai:latest sh -c"));
    assert!(text.contains("urllib.request.urlretrieve"));
    assert!(text.contains(&format!(
        "releases/download/v{}/optimum-advisor-x86_64-unknown-linux-musl",
        env!("CARGO_PKG_VERSION")
    )));
    assert!(text.contains("base64 -d > /tmp/optimum-advisor-config.toml"));
    assert!(text.contains("bench --in-container --config /tmp/optimum-advisor-config.toml"));
    assert!(text.contains("--results-dir /tmp/optimum-advisor-results"));
    assert!(text.contains("uv pip install --quiet"));
    assert!(text.contains("lighteval==0.13.0"));
    assert!(text.contains("find /tmp/optimum-advisor-results -name report.json"));

    // With a bucket: mounted read-write, no log dump.
    let output = run(&[
        "bench",
        "--config",
        config,
        "--on",
        "hf-jobs",
        "--hf-flavor",
        "a10g-large",
        "--results-bucket",
        "hf://buckets/me/oa",
        "--dry-run",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("-v hf://buckets/me/oa:/results:rw"));
    // The run stages locally; one bulk copy lands in the mounted bucket at the
    // end, and it also runs for failed evaluations.
    assert!(text.contains("--results-dir /tmp/optimum-advisor-results"));
    assert!(!text.contains("--results-dir /results"));
    assert!(text.contains("cp -r /tmp/optimum-advisor-results/. /results/"));
    assert!(!text.contains("find /tmp/optimum-advisor-results -name report.json"));
}

#[test]
fn hf_jobs_image_override_selects_and_forwards_the_job_image() {
    let directory = TempDir::new().unwrap();
    let config = write_config(&directory, HF_JOBS_CONFIG);
    let config = config.to_str().unwrap();

    let output = run(&[
        "bench",
        "--config",
        config,
        "--on",
        "hf-jobs",
        "--hf-flavor",
        "a10g-large",
        "--image",
        "repo/custom-vllm:2",
        "--dry-run",
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    // The override is the job container image...
    assert!(text.contains("repo/custom-vllm:2 sh -c"), "{text}");
    assert!(!text.contains("vllm/vllm-openai:latest"));
    // ...and is forwarded in-job so the report identity matches it.
    assert!(text.contains("--image repo/custom-vllm:2"));
}

#[test]
fn hf_jobs_requires_flavor_config_and_forbids_overrides() {
    let directory = TempDir::new().unwrap();
    let config = write_config(&directory, HF_JOBS_CONFIG);
    let config = config.to_str().unwrap();

    let no_flavor = run(&["bench", "--config", config, "--on", "hf-jobs", "--dry-run"]);
    assert_eq!(no_flavor.status.code(), Some(2));
    assert!(stderr(&no_flavor).contains("requires --hf-flavor"));

    let local_with_hf = run(&["bench", "--config", config, "--hf-flavor", "a10g-large"]);
    assert_eq!(local_with_hf.status.code(), Some(2));
    assert!(stderr(&local_with_hf).contains("require --on hf-jobs"));

    let override_conflict = run(&[
        "bench",
        "--config",
        config,
        "--on",
        "hf-jobs",
        "--hf-flavor",
        "a10g-large",
        "--model",
        "other/model",
        "--dry-run",
    ]);
    assert_eq!(override_conflict.status.code(), Some(2));
    assert!(stderr(&override_conflict).contains("supports only the --image CLI override"));
}

#[test]
fn schema_v2_config_and_cli_overlay_use_one_canonical_model() {
    let directory = TempDir::new().unwrap();
    let config = write_config(
        &directory,
        r#"
schema_version = 2
engine = "sglang"
model = "file/model"
metric = "tpot"

[runtime]
gpus = 2
max_model_len = 4096

[candidate]
tensor_parallelism = 2
prefill_token_budget = 2048

[serve]
reasoning_parser = "deepseek-r1"
"#,
    );
    let output = command()
        .args([
            "plan",
            "--config",
            config.to_str().unwrap(),
            "--model",
            "cli/model",
        ])
        .output()
        .unwrap();
    let text = stdout(&output);

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(text.contains("sglang.launch_server"));
    assert!(text.contains("--model-path cli/model"));
    assert!(text.contains("--tp-size 2"));
    assert!(text.contains("--reasoning-parser deepseek-r1"));
}

#[test]
fn merge_precedence_is_config_then_cli_and_ignores_operational_environment() {
    let directory = TempDir::new().unwrap();
    let config = write_config(
        &directory,
        r#"
schema_version = 2
engine = "vllm"
model = "file/model"

[runtime]
max_model_len = 4096
"#,
    );
    let output = command()
        .args([
            "plan",
            "--config",
            config.to_str().unwrap(),
            "--model",
            "cli/model",
        ])
        .env("OPTIMUM_ADVISOR_MODEL", "environment/model")
        .env("OPTIMUM_ADVISOR_MAX_MODEL_LEN", "2048")
        .output()
        .unwrap();
    let text = stdout(&output);

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(text.contains("--model cli/model"));
    assert!(text.contains("--max-model-len 4096"));
    assert!(!text.contains("file/model"));
    assert!(!text.contains("environment/model"));
}

#[test]
fn config_schema_version_and_unknown_keys_are_rejected() {
    let directory = TempDir::new().unwrap();
    let old = write_config(&directory, "schema_version=1\nengine='vllm'\nmodel='m'\n");
    let output = command()
        .args(["plan", "--config", old.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("unsupported schema_version 1"));

    fs::write(
        &old,
        "schema_version=2\nengine='vllm'\nmodel='m'\nunknown=1\n",
    )
    .unwrap();
    let output = command()
        .args(["plan", "--config", old.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(stderr(&output).contains("unknown field"));
}
#[test]
fn sweep_expands_bounded_candidates_without_execution() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sweep-v2.toml");
    let output = command()
        .args(["sweep", "--config", fixture.to_str().unwrap(), "--dry-run"])
        .output()
        .unwrap();
    let text = stdout(&output);

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(text.contains("trial: 1/2"));
    assert!(text.contains("trial: 2/2"));
    assert!(text.contains("--gpu-memory-utilization 0.8"));
    assert!(text.contains("--gpu-memory-utilization 0.9"));
}

#[test]
fn plan_and_bench_previews_do_not_require_host_tools_or_credentials() {
    let directory = TempDir::new().unwrap();
    let empty_path = directory.path().join("empty-path");
    fs::create_dir(&empty_path).unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bench-v2.toml");

    for args in [
        vec!["plan", "--config", fixture.to_str().unwrap()],
        vec!["bench", "--config", fixture.to_str().unwrap(), "--dry-run"],
    ] {
        let output = command()
            .args(args)
            .env("PATH", &empty_path)
            .env_remove("HF_TOKEN")
            .env_remove("HUGGING_FACE_HUB_TOKEN")
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains("docker run"));
        assert!(!stdout(&output).contains("HF_TOKEN"));
    }
}

#[test]
fn execution_without_credentials_reaches_docker_preflight() {
    let directory = TempDir::new().unwrap();
    let results = directory.path().join("results");
    let home = directory.path().join("home");
    let empty_path = directory.path().join("empty-path");
    fs::create_dir(&home).unwrap();
    fs::create_dir(&empty_path).unwrap();
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bench-v2.toml");
    let output = command()
        .args([
            "bench",
            "--config",
            fixture.to_str().unwrap(),
            "--results-dir",
            results.to_str().unwrap(),
        ])
        .env("PATH", &empty_path)
        .env("HOME", &home)
        .env("HF_HOME", home.join("hf"))
        .env("XDG_CACHE_HOME", home.join("cache"))
        .env_remove("HF_TOKEN")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .env_remove("HF_TOKEN_PATH")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("failed to check local Docker image"));
    assert!(!stderr(&output).contains("missing Hugging Face token"));
    let report = fs::read_to_string(report_path(&results)).unwrap();
    let report: Value = serde_json::from_str(&report).unwrap();
    assert_eq!(report["state"], "failed");
    assert_eq!(report["run_failure"]["stage"], "image_resolution");
    assert!(report["resolved_image"].is_null());
    assert!(report["trials"].as_array().unwrap().is_empty());
}

#[test]
fn mcp_stdio_is_protocol_clean_and_returns_structured_results() {
    let mut child = command()
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let requests = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
        json!({
            "jsonrpc":"2.0",
            "id":3,
            "method":"tools/call",
            "params":{
                "name":"rank_candidates",
                "arguments":{
                    "metric":"p99_ttft",
                    "candidates":[
                        {"id":"slow","value":20.0},
                        {"id":"fast","value":10.0}
                    ]
                }
            }
        }),
    ];
    {
        let stdin = child.stdin.as_mut().unwrap();
        for request in requests {
            writeln!(stdin, "{request}").unwrap();
        }
    }
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();

    assert!(output.status.success(), "{}", stderr(&output));
    assert!(output.stderr.is_empty(), "{}", stderr(&output));
    let responses = stdout(&output)
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 3);
    assert_eq!(
        responses[0]["result"]["serverInfo"]["name"],
        "optimum-advisor"
    );
    assert!(responses[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "evaluate_candidate"));
    assert_eq!(
        responses[2]["result"]["structuredContent"]["candidates"][0]["id"],
        "fast"
    );
}
#[cfg(unix)]
mod execution {
    use std::{
        os::unix::fs::PermissionsExt,
        thread,
        time::{Duration, Instant},
    };

    use super::*;

    const TOKEN: &str = "hf_production_secret_fixture";

    struct FakeRuntime {
        directory: TempDir,
        bin: PathBuf,
        server: PathBuf,
        config: PathBuf,
        results: PathBuf,
        events: PathBuf,
        server_events: PathBuf,
        port: u16,
    }

    impl FakeRuntime {
        fn new() -> Self {
            let directory = TempDir::new().unwrap();
            let bin = directory.path().join("bin");
            fs::create_dir(&bin).unwrap();
            let server = directory.path().join("server.py");
            fs::write(&server, include_str!("fixtures/fake_server.py")).unwrap();
            let docker = bin.join("docker");
            fs::write(
                &docker,
                r#"#!/bin/sh
set -eu

if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
    case "$*" in
        *--format*) printf '%s\n' '{"id":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repo_digests":["repo/image@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]}' ;;
    esac
    exit 0
fi
if [ "${1:-}" = "ps" ]; then
    printf '%s\n' 'owned-run-7-server'
    exit 0
fi
if [ "${1:-}" = "pull" ] || [ "${1:-}" = "rm" ]; then
    exit 0
fi
if [ "${1:-}" != "run" ]; then
    echo "unexpected docker command: $*" >&2
    exit 64
fi
case "$*" in
    *make_arg_parser*|*ServerArgs.add_cli_args*)
        printf '%s\n' '--reasoning-parser	value' '--trust-remote-code	flag'
        exit 0
        ;;
esac
benchmark=false
binding=
previous=
for argument in "$@"; do
    if [ "$argument" = "--network" ]; then
        benchmark=true
    fi
    if [ "$previous" = "-p" ]; then
        binding=$argument
    fi
    previous=$argument
done
if [ "$benchmark" = true ]; then
    printf '%s\n' started > "$FAKE_EVENTS"
    if [ "${SLOW_BENCH:-0}" = "1" ]; then
        sleep 30
    fi
    if [ "${FAIL_BENCH:-0}" = "1" ]; then
        printf 'benchmark credential=%s\n' "${HF_TOKEN:-missing}" >&2
        exit 9
    fi
    printf '%s\n' \
        'Successful requests: 4' \
        'Failed requests: 0' \
        'Request throughput (req/s): 1.25' \
        'Output token throughput (tok/s): 42.5' \
        'Mean TTFT (ms): 24.95' \
        'Mean TPOT (ms): 6.09' \
        'Mean ITL (ms): 6.09'
    exit 0
fi
if [ -n "$binding" ]; then
    port=${binding#*:}
    port=${port%%:*}
    printf '%s\n' started > "$FAKE_SERVER_EVENTS"
    exec python3 "$FAKE_SERVER" "$port"
fi
echo "unrecognized docker run: $*" >&2
exit 64
"#,
            )
            .unwrap();
            fs::set_permissions(&docker, fs::Permissions::from_mode(0o700)).unwrap();

            let lighteval = bin.join("lighteval");
            fs::write(
                &lighteval,
                r#"#!/bin/sh
set -eu
output=
previous=
for argument in "$@"; do
    if [ "$previous" = "--output-dir" ]; then
        output=$argument
    fi
    previous=$argument
done
mkdir -p "$output"
if [ "${FAIL_CORRECTNESS:-0}" = "1" ]; then
    printf '%s\n' '{"results":{"gsm8k|0":{"extractive_match":0.0},"ifeval|0":{"prompt_level_strict_acc":0.0},"triviaqa|0":{"em":0.0},"drop|1":{"em":0.0}}}' > "$output/results.json"
else
    printf '%s\n' '{"results":{"gsm8k|0":{"extractive_match":1.0},"ifeval|0":{"prompt_level_strict_acc":1.0},"triviaqa|0":{"em":1.0},"drop|1":{"em":1.0}}}' > "$output/results.json"
fi
"#,
            )
            .unwrap();
            fs::set_permissions(&lighteval, fs::Permissions::from_mode(0o700)).unwrap();

            let nvidia_smi = bin.join("nvidia-smi");
            fs::write(
                &nvidia_smi,
                "#!/bin/sh\nprintf '%s\\n' '0, Test GPU, GPU-test, 9.0, 24576, 24000, 576'\n",
            )
            .unwrap();
            fs::set_permissions(&nvidia_smi, fs::Permissions::from_mode(0o700)).unwrap();

            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            let results = directory.path().join("results");
            let events = directory.path().join("events");
            let server_events = directory.path().join("server-events");
            let config = directory.path().join("config.toml");
            fs::write(
                &config,
                format!(
                    r#"schema_version = 2
engine = "vllm"
image = "repo/image:tag"
model = "repo/model"
metric = "tps"

[runtime]
pull_policy = "never"
port = {port}
startup_timeout_secs = 5
benchmark_timeout_secs = 5
max_process_output_bytes = 1048576

[benchmark]
num_prompts = 4

[correctness]
enabled = false

[model_memory]
enabled = false
"#
                ),
            )
            .unwrap();
            Self {
                directory,
                bin,
                server,
                config,
                results,
                events,
                server_events,
                port,
            }
        }

        fn command(&self, fail_benchmark: bool) -> Command {
            let mut path = self.bin.as_os_str().to_os_string();
            path.push(":");
            path.push(std::env::var_os("PATH").unwrap_or_default());
            let mut command = super::command();
            command.args([
                "bench",
                "--config",
                self.config.to_str().unwrap(),
                "--results-dir",
                self.results.to_str().unwrap(),
            ]);
            command
                .env("PATH", path)
                .env("HF_TOKEN", TOKEN)
                .env("FAKE_SERVER", &self.server)
                .env("FAKE_EVENTS", &self.events)
                .env("FAKE_SERVER_EVENTS", &self.server_events)
                .env("HOME", self.directory.path())
                .current_dir(self.directory.path())
                .env("FAIL_BENCH", if fail_benchmark { "1" } else { "0" });
            command
        }

        fn mcp_command(&self, fail_benchmark: bool) -> Command {
            let mut path = self.bin.as_os_str().to_os_string();
            path.push(":");
            path.push(std::env::var_os("PATH").unwrap_or_default());
            let mut command = super::command();
            command.arg("mcp");
            command
                .env("PATH", path)
                .env("HF_TOKEN", TOKEN)
                .env("FAKE_SERVER", &self.server)
                .env("FAKE_EVENTS", &self.events)
                .env("FAKE_SERVER_EVENTS", &self.server_events)
                .env("HOME", self.directory.path())
                .current_dir(self.directory.path())
                .env("FAIL_BENCH", if fail_benchmark { "1" } else { "0" });
            command
        }
    }

    #[test]
    fn bench_executes_end_to_end_and_persists_a_valid_private_report() {
        let runtime = FakeRuntime::new();
        let output = runtime.command(false).output().unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains("winning_config:"));
        assert!(stderr(&output).contains("trial: 1/1"));
        assert!(!stdout(&output).contains("trial:"));
        let path = report_path(&runtime.results);
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains(TOKEN));
        let report: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(report["state"], "completed");
        assert_eq!(report["best_trial_index"], 0);
        assert_eq!(
            report["trials"][0]["metrics"]["output_token_throughput"],
            42.5
        );
        assert!(report["best_config_path"]
            .as_str()
            .unwrap()
            .ends_with("best.toml"));
        assert!(report["trials"][0]["artifacts"].as_array().unwrap().len() >= 4);
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(path.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert!(runtime.port > 0);
    }

    #[test]
    fn omitted_metric_uses_tiny_model_latency_default_end_to_end() {
        let runtime = FakeRuntime::new();
        let config = fs::read_to_string(&runtime.config)
            .unwrap()
            .replace("model = \"repo/model\"", "model = \"Qwen/Qwen3-0.6B\"")
            .replace("metric = \"tps\"\n", "");
        fs::write(&runtime.config, config).unwrap();

        let output = runtime.command(false).output().unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        let report: Value =
            serde_json::from_str(&fs::read_to_string(report_path(&runtime.results)).unwrap())
                .unwrap();
        assert_eq!(report["winning_metric"], "tpot");
        assert_eq!(report["best_winning_value"], 6.09);
    }

    #[test]
    fn missing_selected_metric_preserves_observed_benchmark_metrics() {
        let runtime = FakeRuntime::new();
        let config = fs::read_to_string(&runtime.config)
            .unwrap()
            .replace("metric = \"tps\"", "metric = \"e2e\"");
        fs::write(&runtime.config, config).unwrap();

        let output = runtime.command(false).output().unwrap();

        assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
        assert!(stderr(&output).contains("selected metric e2e was not emitted"));
        assert!(stderr(&output).contains("available finite metrics: tps, req_s, ttft, tpot, itl"));
        let report: Value =
            serde_json::from_str(&fs::read_to_string(report_path(&runtime.results)).unwrap())
                .unwrap();
        assert_eq!(report["state"], "failed");
        assert_eq!(report["trials"][0]["status"], "failed");
        assert_eq!(
            report["trials"][0]["metrics"]["output_token_throughput"],
            42.5
        );
        assert_eq!(report["trials"][0]["metrics"]["mean_ttft_ms"], 24.95);
        assert_eq!(report["trials"][0]["metrics"]["mean_tpot_ms"], 6.09);
        assert_eq!(report["trials"][0]["metrics"]["mean_itl_ms"], 6.09);
    }

    #[test]
    fn below_threshold_correctness_is_a_failed_trial_not_an_invalid_success() {
        let runtime = FakeRuntime::new();
        let config = fs::read_to_string(&runtime.config).unwrap().replace(
            "[correctness]\nenabled = false",
            "[correctness]\nenabled = true\nthreshold = 0.5",
        );
        fs::write(&runtime.config, config).unwrap();
        let output = runtime
            .command(false)
            .env("FAIL_CORRECTNESS", "1")
            .output()
            .unwrap();

        assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
        assert!(
            stderr(&output).contains("correctness threshold 0.5 was not met"),
            "{}",
            stderr(&output)
        );
        let report: Value =
            serde_json::from_str(&fs::read_to_string(report_path(&runtime.results)).unwrap())
                .unwrap();
        assert_eq!(report["state"], "failed");
        assert_eq!(report["trials"][0]["status"], "failed");
        assert_eq!(report["trials"][0]["failure"]["kind"], "correctness");
        assert_eq!(report["trials"][0]["failure"]["stage"], "correctness");
        assert_eq!(report["trials"][0]["correctness"]["status"], "failed");
        assert!(report["trials"][0]["metrics"].is_null());
        assert!(
            !runtime.events.exists(),
            "benchmark ran after correctness failed"
        );
    }

    #[test]
    fn public_model_execution_does_not_require_hugging_face_credentials() {
        let runtime = FakeRuntime::new();
        let output = runtime
            .command(false)
            .env_remove("HF_TOKEN")
            .env_remove("HUGGING_FACE_HUB_TOKEN")
            .env_remove("HF_TOKEN_PATH")
            .output()
            .unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        assert_eq!(
            serde_json::from_str::<Value>(
                &fs::read_to_string(report_path(&runtime.results)).unwrap()
            )
            .unwrap()["state"],
            "completed"
        );
    }

    #[test]
    fn interrupt_finalizes_the_report_and_stops_active_processes() {
        let runtime = FakeRuntime::new();
        let mut command = runtime.command(false);
        command
            .env("SLOW_BENCH", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();

        let deadline = Instant::now() + Duration::from_secs(10);
        while !runtime.events.exists() {
            if Instant::now() >= deadline {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!("benchmark did not start: {}", stderr(&output));
            }
            thread::sleep(Duration::from_millis(10));
        }
        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(child.id() as i32),
            nix::sys::signal::Signal::SIGINT,
        )
        .unwrap();
        let output = child.wait_with_output().unwrap();

        assert_eq!(output.status.code(), Some(130), "{}", stderr(&output));
        let report: Value =
            serde_json::from_str(&fs::read_to_string(report_path(&runtime.results)).unwrap())
                .unwrap();
        assert_eq!(report["state"], "interrupted");
        assert_eq!(report["run_failure"]["kind"], "interrupted");
        assert!(!report["ended_at_unix_ms"].is_null());
    }

    #[test]
    fn cleanup_cli_lists_owned_containers_without_removing_in_dry_run() {
        let runtime = FakeRuntime::new();
        let mut path = runtime.bin.as_os_str().to_os_string();
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let output = super::command()
            .args(["cleanup", "--run-id", "run-7", "--dry-run"])
            .env("PATH", path)
            .current_dir(runtime.directory.path())
            .output()
            .unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        assert_eq!(stdout(&output), "owned_container: owned-run-7-server\n");
        assert!(stderr(&output).is_empty());
    }

    #[test]
    fn params_inspects_an_engine_through_the_runtime() {
        let runtime = FakeRuntime::new();
        let mut path = runtime.bin.as_os_str().to_os_string();
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let cache = runtime.directory.path().join("params");
        let output = super::command()
            .args([
                "params",
                "--engine",
                "vllm",
                "--image",
                "repo/image:tag",
                "--pull-policy",
                "never",
                "--cache-dir",
                cache.to_str().unwrap(),
            ])
            .env("PATH", path)
            .current_dir(runtime.directory.path())
            .output()
            .unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains("source: runtime_or_cache"));
        assert!(stdout(&output).contains("--reasoning-parser"));
        assert!(stderr(&output).is_empty());
    }

    #[test]
    fn hardware_prints_selected_local_gpu_data() {
        let runtime = FakeRuntime::new();
        let mut path = runtime.bin.as_os_str().to_os_string();
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let output = super::command()
            .arg("hardware")
            .env("PATH", path)
            .current_dir(runtime.directory.path())
            .output()
            .unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains("Test GPU"));
        assert!(stderr(&output).is_empty());
    }

    #[test]
    fn serve_runs_until_interrupted_and_does_not_require_a_token() {
        let runtime = FakeRuntime::new();
        let mut path = runtime.bin.as_os_str().to_os_string();
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut child = super::command()
            .args(["serve", "--config", runtime.config.to_str().unwrap()])
            .env("PATH", path)
            .env("FAKE_SERVER", &runtime.server)
            .env("FAKE_SERVER_EVENTS", &runtime.server_events)
            .env("HOME", runtime.directory.path())
            .env_remove("HF_TOKEN")
            .env_remove("HUGGING_FACE_HUB_TOKEN")
            .env_remove("HF_TOKEN_PATH")
            .current_dir(runtime.directory.path())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        wait_for_path(&runtime.server_events, &mut child);

        nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(child.id() as i32),
            nix::sys::signal::Signal::SIGINT,
        )
        .unwrap();
        let output = child.wait_with_output().unwrap();

        assert_eq!(output.status.code(), Some(130), "{}", stderr(&output));
        assert!(stdout(&output).is_empty());
        assert!(stderr(&output).contains("Test GPU"));
        assert!(stderr(&output).contains("interrupted"));
        assert!(!stderr(&output).contains("HF_TOKEN"));
    }

    #[test]
    fn failed_candidate_is_recorded_and_credentials_are_redacted_everywhere() {
        let runtime = FakeRuntime::new();
        let output = runtime.command(true).output().unwrap();

        assert_eq!(output.status.code(), Some(1), "{}", stderr(&output));
        assert!(stderr(&output).contains("all benchmark candidates failed"));
        assert!(stderr(&output).contains("benchmark credential=[REDACTED]"));
        assert!(!stderr(&output).contains(TOKEN));
        let path = report_path(&runtime.results);
        assert!(stderr(&output).contains(path.to_str().unwrap()));
        let text = fs::read_to_string(&path).unwrap();
        assert!(!text.contains(TOKEN));
        let report: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(report["state"], "failed");
        assert_eq!(report["trials"][0]["status"], "failed");
        assert_eq!(report["trials"][0]["failure"]["kind"], "process_exit");
        assert!(report["trials"][0]["failure"]["stderr_tail"]
            .as_str()
            .unwrap()
            .contains("[REDACTED]"));

        for entry in walk_files(path.parent().unwrap()) {
            let bytes = fs::read(&entry).unwrap();
            assert!(
                !bytes
                    .windows(TOKEN.len())
                    .any(|window| window == TOKEN.as_bytes()),
                "secret leaked to {}",
                entry.display()
            );
        }
    }

    #[test]
    fn mcp_correctness_check_does_not_run_the_benchmark() {
        let runtime = FakeRuntime::new();
        let mut child = runtime
            .mcp_command(true)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let requests = [
            json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
            json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            json!({
                "jsonrpc":"2.0",
                "id":2,
                "method":"tools/call",
                "params":{
                    "name":"check_correctness",
                    "arguments":{
                        "results_dir":runtime.results,
                        "config":{
                            "engine":"vllm",
                            "image":"repo/image:tag",
                            "model":"repo/model",
                            "metric":"tps",
                            "runtime":{
                                "gpus":1,
                                "pull_policy":"never",
                                "port":runtime.port,
                                "startup_timeout_secs":5,
                                "benchmark_timeout_secs":5,
                                "max_process_output_bytes":1048576
                            },
                            "benchmark":{"num_prompts":4},
                            "candidate":{
                                "tensor_parallelism":1,
                                "memory_fraction":0.9,
                                "prefill_token_budget":1024,
                                "max_running_requests":1
                            },
                            "correctness":{"enabled":true,"threshold":0.2,"timeout_secs":5},
                            "model_memory":{"enabled":false},
                            "leaderboard":{"submit":false}
                        }
                    }
                }
            }),
        ];
        {
            let stdin = child.stdin.as_mut().unwrap();
            for request in requests {
                writeln!(stdin, "{request}").unwrap();
            }
        }
        drop(child.stdin.take());

        let output = child.wait_with_output().unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        assert!(output.stderr.is_empty(), "{}", stderr(&output));
        let responses = stdout(&output)
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[1]["result"]["isError"], false);
        assert_eq!(
            responses[1]["result"]["structuredContent"]["state"],
            "passed"
        );
        assert_eq!(
            responses[1]["result"]["structuredContent"]["correctness"]["status"],
            "passed"
        );
        assert!(responses[1]["result"]["structuredContent"]["report_path"]
            .as_str()
            .is_some_and(|path| Path::new(path).is_file()));
        assert!(
            !runtime.events.exists(),
            "benchmark command unexpectedly ran"
        );
    }
    fn wait_for_path(path: &Path, child: &mut std::process::Child) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !path.exists() {
            if Instant::now() >= deadline {
                let _ = child.kill();
                panic!("timed out waiting for {}", path.display());
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn walk_files(root: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let mut directories = vec![root.to_path_buf()];
        while let Some(directory) = directories.pop() {
            for entry in fs::read_dir(directory).unwrap() {
                let entry = entry.unwrap();
                if entry.file_type().unwrap().is_dir() {
                    directories.push(entry.path());
                } else {
                    files.push(entry.path());
                }
            }
        }
        files
    }
}
