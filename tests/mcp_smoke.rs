#![cfg(unix)]

use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::TcpListener,
    os::unix::fs::PermissionsExt,
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};

use serde_json::{json, Value};
use tempfile::TempDir;

#[test]
fn cancellation_keeps_the_binary_server_ready() {
    let fixture = McpFixture::new();
    let mut child = fixture.command();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if sender.send(line.unwrap()).is_err() {
                break;
            }
        }
    });

    send(
        &mut stdin,
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
    );
    send(
        &mut stdin,
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    );
    send(
        &mut stdin,
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}),
    );

    let initialized = response(&receiver);
    assert_eq!(initialized["id"], 1);
    assert_eq!(initialized["result"]["protocolVersion"], "2025-11-25");
    let listed = response(&receiver);
    assert_eq!(listed["id"], 2);
    let tools = listed["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 11);
    assert!(tools.iter().all(|tool| {
        tool["inputSchema"]["additionalProperties"] == false && tool["outputSchema"].is_object()
    }));

    send(
        &mut stdin,
        json!({
            "jsonrpc":"2.0",
            "id":3,
            "method":"tools/call",
            "params":{
                "name":"inspect_engine",
                "arguments":{
                    "engine":"vllm",
                    "image":"repo/image:tag",
                    "cache_dir":fixture.directory.path().join("cache"),
                    "pull_policy":"never"
                }
            }
        }),
    );
    wait_for_path(&fixture.started, &mut child);
    send(
        &mut stdin,
        json!({
            "jsonrpc":"2.0",
            "method":"notifications/cancelled",
            "params":{"requestId":3,"reason":"test cancellation"}
        }),
    );

    let cancelled = response(&receiver);
    assert_eq!(cancelled["id"], 3);
    assert_eq!(cancelled["result"]["isError"], true);
    assert_eq!(
        cancelled["result"]["structuredContent"]["kind"],
        "interrupted"
    );

    send(
        &mut stdin,
        json!({
            "jsonrpc":"2.0",
            "id":4,
            "method":"tools/call",
            "params":{
                "name":"rank_candidates",
                "arguments":{
                    "metric":"tps",
                    "candidates":[
                        {"id":"slow","value":10.0,"correctness":"passed"},
                        {"id":"fast","value":20.0,"correctness":"passed"}
                    ]
                }
            }
        }),
    );
    let ranked = response(&receiver);
    assert_eq!(ranked["id"], 4);
    assert_eq!(ranked["result"]["isError"], false);
    assert_eq!(
        ranked["result"]["structuredContent"]["candidates"][0]["id"],
        "fast"
    );

    drop(stdin);
    let status = wait_for_exit(&mut child);
    assert!(status.success());
    reader.join().unwrap();
    let stderr = child.wait_with_output().unwrap().stderr;
    assert!(stderr.is_empty(), "{}", String::from_utf8_lossy(&stderr));
}

#[test]
fn server_startup_failure_returns_an_error_and_keeps_mcp_ready() {
    let fixture = McpFixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let mut child = fixture.failing_server_command();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (sender, receiver) = mpsc::channel();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if sender.send(line.unwrap()).is_err() {
                break;
            }
        }
    });

    send(
        &mut stdin,
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
    );
    send(
        &mut stdin,
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    );
    assert_eq!(response(&receiver)["id"], 1);
    send(
        &mut stdin,
        json!({
            "jsonrpc":"2.0",
            "id":2,
            "method":"tools/call",
            "params":{
                "name":"run_benchmark",
                "arguments":{
                    "results_dir":fixture.directory.path().join("results"),
                    "config":{
                        "engine":"vllm",
                        "image":"repo/image:tag",
                        "model":"repo/model",
                        "metric":"tps",
                        "runtime":{
                            "port":port,
                            "startup_timeout_secs":5,
                            "benchmark_timeout_secs":5
                        },
                        "benchmark":{"num_prompts":1},
                        "correctness":{"enabled":false},
                        "model_memory":{"enabled":false}
                    }
                }
            }
        }),
    );

    let failed = response(&receiver);
    assert_eq!(failed["id"], 2);
    assert_eq!(failed["result"]["isError"], true);
    let error = &failed["result"]["structuredContent"];
    assert_eq!(error["kind"], "benchmark", "{failed}");
    assert_eq!(error["stage"], "benchmark");
    assert!(error["message"]
        .as_str()
        .unwrap()
        .contains("fatal server configuration"));
    let report_path = error["report_path"].as_str().unwrap();
    let report: Value = serde_json::from_str(&fs::read_to_string(report_path).unwrap()).unwrap();
    assert_eq!(report["state"], "failed");
    assert_eq!(report["trials"][0]["failure"]["stage"], "server");
    assert!(report["trials"][0]["failure"]["stderr_tail"]
        .as_str()
        .unwrap()
        .contains("fatal server configuration"));

    send(
        &mut stdin,
        json!({
            "jsonrpc":"2.0",
            "id":3,
            "method":"tools/call",
            "params":{
                "name":"rank_candidates",
                "arguments":{
                    "metric":"tps",
                    "candidates":[{"id":"candidate","value":1.0}]
                }
            }
        }),
    );
    assert_eq!(response(&receiver)["result"]["isError"], false);

    drop(stdin);
    assert!(wait_for_exit(&mut child).success());
    reader.join().unwrap();
    let stderr = child.wait_with_output().unwrap().stderr;
    assert!(stderr.is_empty(), "{}", String::from_utf8_lossy(&stderr));
}

struct McpFixture {
    directory: TempDir,
    started: std::path::PathBuf,
}

impl McpFixture {
    fn new() -> Self {
        let directory = TempDir::new().unwrap();
        let bin = directory.path().join("bin");
        fs::create_dir(&bin).unwrap();
        let started = directory.path().join("introspection-started");
        let docker = bin.join("docker");
        fs::write(
            &docker,
            r#"#!/bin/sh
set -eu
if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
    printf '%s\n' '{"id":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repo_digests":["repo/image@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]}'
    exit 0
fi
if [ "${1:-}" = "run" ]; then
    case "$*" in
        *make_arg_parser*|*ServerArgs.add_cli_args*)
            if [ "${MCP_FAIL_SERVER:-0}" = "1" ]; then
                printf '%s\n' '--reasoning-parser	value' '--trust-remote-code	flag'
                exit 0
            fi
            printf '%s\n' started > "$MCP_INTROSPECTION_STARTED"
            trap 'exit 130' TERM INT
            while :; do sleep 1; done
            ;;
    esac
    if [ "${MCP_FAIL_SERVER:-0}" = "1" ]; then
        printf '%s\n' 'fatal server configuration' >&2
        exit 17
    fi
fi
echo "unexpected docker command: $*" >&2
exit 64
"#,
        )
        .unwrap();
        fs::set_permissions(&docker, fs::Permissions::from_mode(0o700)).unwrap();
        let nvidia_smi = bin.join("nvidia-smi");
        fs::write(
            &nvidia_smi,
            "#!/bin/sh\nprintf '%s\\n' '0, Test GPU, GPU-test, 9.0, 24576, 24000, 576'\n",
        )
        .unwrap();
        fs::set_permissions(&nvidia_smi, fs::Permissions::from_mode(0o700)).unwrap();
        Self { directory, started }
    }

    fn failing_server_command(&self) -> Child {
        self.command_with_server_failure(true)
    }

    fn command(&self) -> Child {
        self.command_with_server_failure(false)
    }

    fn command_with_server_failure(&self, fail_server: bool) -> Child {
        let mut path = self.directory.path().join("bin").into_os_string();
        path.push(":");
        path.push(std::env::var_os("PATH").unwrap_or_default());
        let mut command = Command::new(env!("CARGO_BIN_EXE_optimum-advisor"));
        command
            .arg("mcp")
            .env("PATH", path)
            .env("MCP_INTROSPECTION_STARTED", &self.started)
            .current_dir(self.directory.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if fail_server {
            command.env("MCP_FAIL_SERVER", "1");
        }
        command.spawn().unwrap()
    }
}

fn send(stdin: &mut impl Write, message: Value) {
    writeln!(stdin, "{message}").unwrap();
    stdin.flush().unwrap();
}

fn response(receiver: &Receiver<String>) -> Value {
    let line = receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("timed out waiting for MCP response");
    assert!(
        !line.contains('\u{1b}'),
        "MCP stdout contained ANSI: {line:?}"
    );
    serde_json::from_str(&line).unwrap()
}

fn wait_for_path(path: &Path, child: &mut Child) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(
            child.try_wait().unwrap().is_none(),
            "MCP exited before introspection started"
        );
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn wait_for_exit(child: &mut Child) -> std::process::ExitStatus {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            panic!("timed out waiting for MCP to exit after EOF");
        }
        thread::sleep(Duration::from_millis(10));
    }
}
