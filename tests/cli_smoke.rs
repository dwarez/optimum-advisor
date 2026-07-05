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
    assert!(text.contains("sglang.launch_server --help"));
    assert!(text.contains("source: runtime only"));
}

#[test]
fn run_dry_run_prints_server_and_benchmark_commands() {
    let output = run(&["run", "--engine", "sglang", "--model", "m"]);

    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("server: docker run"));
    assert!(text.contains("benchmark: python3 -m sglang.bench_serving"));
}

#[test]
fn run_execute_requires_hf_token() {
    let output = run_without_hf_token(&["run", "--engine", "vllm", "--model", "m", "--execute"]);

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
