#![cfg(unix)]

use std::{
    env,
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    thread,
    time::Duration,
};

use nix::{
    sys::signal::{kill, Signal},
    unistd::Pid,
};
use serde_json::Value;
use tempfile::TempDir;

#[test]
fn acceptance_path_resolves_repo_correctness_tool_without_activation() {
    use std::os::unix::fs::PermissionsExt;

    let repo = TempDir::new().unwrap();
    let bin = repo.path().join(".venv/bin");
    fs::create_dir_all(&bin).unwrap();
    let lighteval = bin.join("lighteval");
    fs::write(&lighteval, "#!/bin/sh\nexit 0\n").unwrap();
    let mut permissions = fs::metadata(&lighteval).unwrap().permissions();
    permissions.set_mode(0o700);
    fs::set_permissions(&lighteval, permissions).unwrap();

    let inherited = repo.path().join("inherited");
    fs::create_dir(&inherited).unwrap();
    let output = Command::new("lighteval")
        .env(
            "PATH",
            acceptance_path_with(repo.path(), Some(inherited.as_os_str())),
        )
        .output()
        .unwrap();

    assert!(output.status.success());
}

#[test]
fn acceptance_config_tests_execution_without_a_model_quality_gate() {
    let directory = TempDir::new().unwrap();
    let empty_path = directory.path().join("empty-path");
    fs::create_dir(&empty_path).unwrap();
    for (operation, sweep) in [("bench", false), ("sweep", true)] {
        let config_path = directory.path().join(format!("{operation}.toml"));
        let text = config(
            "sglang",
            Some("lmsysorg/sglang:latest"),
            "Qwen/Qwen3-0.6B",
            sweep,
        );
        fs::write(&config_path, &text).unwrap();

        let value = toml::from_str::<toml::Value>(&text).unwrap();
        assert_eq!(value["correctness"]["enabled"].as_bool(), Some(true));
        assert_eq!(value["correctness"]["threshold"].as_float(), Some(0.0));

        let output = command()
            .arg(operation)
            .arg("--config")
            .arg(&config_path)
            .arg("--dry-run")
            .env("PATH", &empty_path)
            .env_remove("HF_TOKEN")
            .env_remove("HUGGING_FACE_HUB_TOKEN")
            .output()
            .unwrap();

        assert!(output.status.success(), "{}", stderr(&output));
        assert!(stdout(&output).contains("docker run"));
    }
}

#[test]
#[ignore = "requires an explicitly enabled Docker/NVIDIA GPU host"]
fn production_gpu_host_acceptance() {
    assert_eq!(
        env::var("OPTIMUM_ADVISOR_GPU_ACCEPTANCE").as_deref(),
        Ok("1"),
        "set OPTIMUM_ADVISOR_GPU_ACCEPTANCE=1 to acknowledge real GPU execution"
    );
    let model = env::var("OPTIMUM_ADVISOR_GPU_ACCEPTANCE_MODEL").expect(
        "OPTIMUM_ADVISOR_GPU_ACCEPTANCE_MODEL must name a runnable public or authorized model",
    );
    let engine =
        env::var("OPTIMUM_ADVISOR_GPU_ACCEPTANCE_ENGINE").unwrap_or_else(|_| "vllm".to_string());
    assert!(matches!(engine.as_str(), "vllm" | "sglang"));
    let image = env::var("OPTIMUM_ADVISOR_GPU_ACCEPTANCE_IMAGE").ok();
    assert_correctness_environment();
    let workspace = TempDir::new().unwrap();
    let hf_cache = workspace.path().join("huggingface");
    let hf_hub_cache = hf_cache.join("hub");
    let hf_datasets_cache = hf_cache.join("datasets");
    let hf_xet_cache = hf_cache.join("xet");
    fs::create_dir_all(&hf_hub_cache).unwrap();
    fs::create_dir_all(&hf_datasets_cache).unwrap();
    fs::create_dir_all(&hf_xet_cache).unwrap();
    env::set_var("HF_HOME", hf_cache);
    env::set_var("HF_HUB_CACHE", hf_hub_cache);
    env::set_var("HF_DATASETS_CACHE", hf_datasets_cache);
    env::set_var("HF_XET_CACHE", hf_xet_cache);
    let cache = workspace.path().join("params");
    let bench_config = workspace.path().join("bench.toml");
    fs::write(
        &bench_config,
        config(&engine, image.as_deref(), &model, false),
    )
    .unwrap();
    let sweep_config = workspace.path().join("sweep.toml");
    fs::write(
        &sweep_config,
        config(&engine, image.as_deref(), &model, true),
    )
    .unwrap();
    let bench_results = workspace.path().join("bench-results");
    let sweep_results = workspace.path().join("sweep-results");
    for (operation, path) in [("bench", &bench_config), ("sweep", &sweep_config)] {
        run_owned(&[
            operation.to_string(),
            "--config".to_string(),
            path.display().to_string(),
            "--dry-run".to_string(),
        ]);
    }

    let before = run(&["cleanup", "--dry-run"]);

    let hardware = run(&["hardware"]);
    assert!(
        stdout(&hardware).contains("gpu["),
        "hardware output did not include a GPU record\nstdout:\n{}\nstderr:\n{}",
        stdout(&hardware),
        stderr(&hardware),
    );

    let mut params = vec![
        "params".to_string(),
        "--engine".to_string(),
        engine.clone(),
        "--cache-dir".to_string(),
        cache.display().to_string(),
        "--refresh".to_string(),
    ];
    if let Some(image) = image.as_ref() {
        params.extend(["--image".to_string(), image.clone()]);
    }
    let params = run_owned(&params);
    assert!(stdout(&params).contains("image: sha256:") || stdout(&params).contains("@sha256:"));

    run_owned(&[
        "bench".to_string(),
        "--config".to_string(),
        bench_config.display().to_string(),
        "--results-dir".to_string(),
        bench_results.display().to_string(),
    ]);
    let bench_report = read_report(&bench_results);
    assert_terminal_success(&bench_report, 1);
    assert!(bench_report["best_config_path"].as_str().is_some());

    run_owned(&[
        "sweep".to_string(),
        "--config".to_string(),
        sweep_config.display().to_string(),
        "--results-dir".to_string(),
        sweep_results.display().to_string(),
    ]);
    let sweep_report = read_report(&sweep_results);
    assert_terminal_success(&sweep_report, 2);
    assert!(sweep_report["best_config_path"].as_str().is_some());
    if env::var("OPTIMUM_ADVISOR_GPU_ACCEPTANCE_GPUS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|gpus| gpus >= 2)
    {
        assert_parallel_sweep_acceptance(&workspace, &engine, image.as_deref(), &model);
    }

    let child = command()
        .args(["serve", "--config"])
        .arg(&bench_config)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    thread::sleep(Duration::from_secs(10));
    kill(Pid::from_raw(child.id() as i32), Signal::SIGINT).unwrap();
    let interrupted = child.wait_with_output().unwrap();
    assert_eq!(
        interrupted.status.code(),
        Some(130),
        "{}",
        stderr(&interrupted)
    );
    assert!(stderr(&interrupted).contains("interrupted"));

    let after = run(&["cleanup", "--dry-run"]);
    assert_eq!(
        stdout(&after),
        stdout(&before),
        "owned Docker containers leaked"
    );
}

fn assert_parallel_sweep_acceptance(
    workspace: &TempDir,
    engine: &str,
    image: Option<&str>,
    model: &str,
) {
    let mut roots = Vec::new();
    let mut reports = Vec::new();
    for cap in [1, 2] {
        let config_path = workspace.path().join(format!("sweep-cap-{cap}.toml"));
        let text = config(engine, image, model, true)
            .replace("gpus = 1", "gpus = 2")
            .replace(
                "[sweep]\n",
                &format!("[sweep]\nmax_parallel_trials = {cap}\n"),
            );
        fs::write(&config_path, text).unwrap();
        let results = workspace.path().join(format!("sweep-cap-{cap}-results"));
        run_owned(&[
            "sweep".to_string(),
            "--config".to_string(),
            config_path.display().to_string(),
            "--results-dir".to_string(),
            results.display().to_string(),
        ]);
        let report = read_report(&results);
        assert_terminal_success(&report, 2);
        roots.push(results);
        reports.push(report);
    }

    let sequential = &reports[0];
    let parallel = &reports[1];
    for index in 0..2 {
        assert_eq!(
            sequential["trials"][index]["config"]["candidate"],
            parallel["trials"][index]["config"]["candidate"]
        );
        assert_eq!(parallel["trials"][index]["index"], index);
        assert_eq!(parallel["trials"][index]["status"], "success");
    }
    let first = parallel["trials"][0]["allocation"]["gpu_devices"]
        .as_array()
        .unwrap();
    let second = parallel["trials"][1]["allocation"]["gpu_devices"]
        .as_array()
        .unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(second.len(), 1);
    assert_ne!(first, second, "parallel trials must lease disjoint GPUs");
    assert_ne!(
        parallel["trials"][0]["allocation"]["host_port"],
        parallel["trials"][1]["allocation"]["host_port"]
    );
    assert!(sequential["best_trial_index"].as_u64().is_some());
    assert!(parallel["best_trial_index"].as_u64().is_some());
    for root in roots {
        let report_path = fs::read_dir(&root)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path()
            .join("best.toml");
        assert!(!fs::read_to_string(report_path)
            .unwrap()
            .contains("gpu_devices"));
    }
}

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_optimum-advisor"));
    command.env("PATH", acceptance_path());
    command
}

fn acceptance_path() -> OsString {
    acceptance_path_with(
        Path::new(env!("CARGO_MANIFEST_DIR")),
        env::var_os("PATH").as_deref(),
    )
}

fn acceptance_path_with(repo: &Path, inherited: Option<&OsStr>) -> OsString {
    let mut paths = vec![repo.join(".venv/bin")];
    if let Some(inherited) = inherited {
        paths.extend(env::split_paths(inherited));
    }
    env::join_paths(paths).expect("acceptance PATH entries must not contain path separators")
}

fn assert_correctness_environment() {
    let path = acceptance_path();
    let available = env::split_paths(&path)
        .map(|directory| directory.join("lighteval"))
        .any(|candidate| {
            use std::os::unix::fs::PermissionsExt;

            fs::metadata(candidate).is_ok_and(|metadata| {
                metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
            })
        });
    assert!(
        available,
        "lighteval is unavailable; run ./scripts/setup-correctness-env.sh"
    );
}

fn run(arguments: &[&str]) -> Output {
    let output = command().args(arguments).output().unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    output
}

fn run_owned(arguments: &[String]) -> Output {
    let output = command().args(arguments).output().unwrap();
    assert!(output.status.success(), "{}", stderr(&output));
    output
}

fn config(engine: &str, image: Option<&str>, model: &str, sweep: bool) -> String {
    let image = image
        .map(|value| format!("image = {value:?}\n"))
        .unwrap_or_default();
    let sweep = if sweep {
        "\n[sweep]\nmax_trials = 2\nmemory_fraction = [0.80, 0.90]\n"
    } else {
        ""
    };
    format!(
        "schema_version = 2\nengine = {engine:?}\n{image}model = {model:?}\nmetric = \"tps\"\n\n[runtime]\ngpus = 1\nmax_model_len = 1024\nstartup_timeout_secs = 600\nbenchmark_timeout_secs = 600\n\n[benchmark]\nnum_prompts = 1\nrequest_rate = \"1\"\nmax_concurrency = 1\nrandom_input_len = 128\nrandom_output_len = 16\n\n[candidate]\ntensor_parallelism = 1\nmemory_fraction = 0.90\nprefill_token_budget = 1024\nmax_running_requests = 1\n\n[correctness]\nenabled = true\nthreshold = 0.0\ntimeout_secs = 600\n\n[model_memory]\nenabled = false\n\n[leaderboard]\nsubmit = false\n{sweep}"
    )
}

fn read_report(root: &Path) -> Value {
    let mut reports = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path().join("report.json"))
        .filter(|path| path.is_file())
        .collect::<Vec<PathBuf>>();
    reports.sort();
    assert_eq!(
        reports.len(),
        1,
        "expected exactly one report under {}",
        root.display()
    );
    serde_json::from_str(&fs::read_to_string(&reports[0]).unwrap()).unwrap()
}

fn assert_terminal_success(report: &Value, expected_trials: usize) {
    assert_eq!(report["schema_version"], 2);
    assert!(matches!(
        report["state"].as_str(),
        Some("completed" | "completed_with_failures")
    ));
    assert_eq!(report["trials"].as_array().unwrap().len(), expected_trials);
    assert!(report["ended_at_unix_ms"].as_u64().is_some());
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
