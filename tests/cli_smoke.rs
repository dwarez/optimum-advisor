use std::fs;
use std::process::{Command, Output};

use optimum_advisor::engine::Engine;
use optimum_advisor::params::cache_path;

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_optimum-advisor"))
        .args(args)
        .output()
        .expect("failed to run optimum-advisor")
}

fn run_without_hf_token(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_optimum-advisor"))
        .args(args)
        .env_remove("HF_TOKEN")
        .output()
        .expect("failed to run optimum-advisor")
}

fn run_without_hf_token_or_hf_login(args: &[&str]) -> Output {
    let empty_path =
        std::env::temp_dir().join(format!("optimum-advisor-empty-path-{}", std::process::id()));
    fs::create_dir_all(&empty_path).unwrap();
    Command::new(env!("CARGO_BIN_EXE_optimum-advisor"))
        .args(args)
        .env_remove("HF_TOKEN")
        .env("PATH", empty_path)
        .output()
        .expect("failed to run optimum-advisor")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

#[test]
fn plan_renders_vllm_server_and_benchmark_commands() {
    let output = run(&[
        "plan",
        "--engine",
        "vllm",
        "--model",
        "meta-llama/Llama-3.1-8B-Instruct",
        "--metric",
        "ttft",
        "--max-model-len",
        "8192",
        "--num-prompts",
        "4",
        "--request-rate",
        "2",
        "--benchmark-max-concurrency",
        "2",
    ]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("serve: docker run"));
    assert!(text.contains("--tensor-parallel-size 1"));
    assert!(text.contains("--max-model-len 8192"));
    assert!(text.contains("--max-num-batched-tokens 16384"));
    assert!(text.contains("bench: docker run"));
    assert!(text.contains("--entrypoint vllm"));
    assert!(text.contains("bench serve"));
    assert!(text.contains("--num-prompts 4"));
    assert!(text.contains("--request-rate 2"));
    assert!(text.contains("--max-concurrency 2"));
}

#[test]
fn params_dry_run_prints_container_inspection_command() {
    let output = run(&["params", "--engine", "sglang"]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("inspect: docker run"));
    assert!(text.contains("--gpus all"));
    assert!(text.contains("--entrypoint python3"));
    assert!(text.contains("ServerArgs.add_cli_args"));
    assert!(text.contains("source: runtime only"));
}

#[test]
fn bench_dry_run_prints_server_and_benchmark_commands() {
    let output = run(&[
        "bench",
        "--engine",
        "sglang",
        "--model",
        "m",
        "--num-prompts",
        "4",
        "--random-output-len",
        "32",
        "--dry-run",
    ]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("server: docker run"));
    assert!(text.contains("benchmark: docker run"));
    assert!(text.contains("sglang.bench_serving"));
    assert!(text.contains("--num-prompts 4"));
    assert!(text.contains("--random-output-len 32"));
}

#[test]
fn sweep_dry_run_prints_sweep_trials() {
    let output = run(&[
        "sweep",
        "--engine",
        "vllm",
        "--model",
        "m",
        "--gpus",
        "2",
        "--sweep-tp",
        "1,2",
        "--sweep-memory-fraction",
        "0.8,0.9",
        "--dry-run",
    ]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("trial: 1/4"));
    assert!(text.contains("trial: 4/4"));
    assert!(text.contains("--tensor-parallel-size 2"));
    assert!(text.contains("--gpu-memory-utilization 0.80"));
}

#[test]
fn bench_dry_run_accepts_full_config_file() {
    let output = run(&["bench", "--config", "examples/bench.conf", "--dry-run"]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(!text.contains("trial:"));
    assert!(text.contains("--tensor-parallel-size 1"));
    assert!(text.contains("--gpu-memory-utilization 0.90"));
    assert!(text.contains("Qwen/Qwen3-4B-Instruct-2507"));
    assert!(text.contains("correctness_suite: id=oa-fast-v1"));
    assert!(text.contains("correctness: lighteval endpoint litellm"));
    assert!(text.contains("base_url=http://127.0.0.1:8000/v1"));
}

#[test]
fn bench_dry_run_accepts_sglang_config_file() {
    let output = run(&[
        "bench",
        "--config",
        "examples/sglang-bench.conf",
        "--dry-run",
    ]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("sglang.launch_server"));
    assert!(text.contains("sglang.bench_serving"));
    assert!(text.contains("--tp-size 1"));
    assert!(text.contains("--mem-fraction-static 0.88"));
    assert!(text.contains("--chunked-prefill-size 8192"));
    assert!(text.contains("--random-input-len 1024"));
    assert!(text.contains("--random-output-len 128"));
}

#[test]
fn bench_rejects_sweep_config_file() {
    let output = run(&["bench", "--config", "examples/sweep.conf", "--dry-run"]);

    assert!(!output.status.success());
    assert!(stderr(&output).contains("bench accepts one configuration"));
}

#[test]
fn sweep_dry_run_accepts_full_config_file() {
    let output = run(&["sweep", "--config", "examples/sweep.conf", "--dry-run"]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("trial: 1/4"));
    assert!(text.contains("--tensor-parallel-size 2"));
    assert!(text.contains("--gpu-memory-utilization 0.80"));
}

#[test]
fn direct_engine_flags_are_forwarded_to_the_server() {
    let output = run(&[
        "bench",
        "--engine",
        "vllm",
        "--model",
        "m",
        "--kv-cache-dtype",
        "fp8",
        "--disable-log-stats",
        "--dry-run",
    ]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("--kv-cache-dtype fp8"));
    assert!(text.contains("--disable-log-stats"));
}

#[test]
fn bench_execute_requires_hf_token() {
    let output = run_without_hf_token(&["bench", "--engine", "vllm", "--model", "m"]);

    assert!(!output.status.success());
    assert!(stderr(&output).contains("HF_TOKEN is required"));
}

#[test]
fn leaderboard_submit_requires_hf_login_before_hf_token() {
    let output = run_without_hf_token_or_hf_login(&[
        "bench",
        "--engine",
        "vllm",
        "--model",
        "m",
        "--leaderboard-submit",
    ]);

    assert!(!output.status.success());
    let err = stderr(&output);
    assert!(
        err.contains("leaderboard submit requires Hugging Face login"),
        "{err}"
    );
    assert!(!err.contains("HF_TOKEN is required"), "{err}");
}

#[test]
fn sweep_requires_hf_token_by_default() {
    let output = run_without_hf_token(&["sweep", "--config", "examples/sweep.conf"]);

    assert!(!output.status.success());
    assert!(stderr(&output).contains("HF_TOKEN is required"));
}

#[test]
fn validate_params_rejects_unknown_cached_arg_without_docker() {
    let cache_dir =
        std::env::temp_dir().join(format!("optimum-advisor-smoke-{}", std::process::id()));
    fs::create_dir_all(&cache_dir).unwrap();
    fs::write(
        cache_path(&cache_dir, Engine::Vllm, "vllm/vllm-openai:latest"),
        "value\t--model\tmodel\n",
    )
    .unwrap();

    let output = run(&[
        "plan",
        "--engine",
        "vllm",
        "--model",
        "m",
        "--param-cache-dir",
        cache_dir.to_str().unwrap(),
        "--validate-params",
        "--serve-arg",
        "definitely-not-a-real-param=x",
    ]);

    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown serving parameter"));
}

#[test]
fn validate_params_rejects_unknown_config_sweep_arg_without_docker() {
    let cache_dir = std::env::temp_dir().join(format!(
        "optimum-advisor-config-smoke-{}",
        std::process::id()
    ));
    fs::create_dir_all(&cache_dir).unwrap();
    fs::write(
        cache_path(&cache_dir, Engine::Vllm, "vllm/vllm-openai:latest"),
        "value\t--model\tmodel\nvalue\t--tensor-parallel-size\ttensor_parallel_size\n",
    )
    .unwrap();
    let config = cache_dir.join("sweep.conf");
    fs::write(
        &config,
        "engine = vllm\nmodel = m\n[sweep]\ndefinitely-not-real = 1,2\n",
    )
    .unwrap();

    let output = run(&[
        "sweep",
        "--config",
        config.to_str().unwrap(),
        "--param-cache-dir",
        cache_dir.to_str().unwrap(),
        "--validate-params",
        "--dry-run",
    ]);

    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown serving parameter"));
}
