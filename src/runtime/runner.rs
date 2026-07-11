use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::{terminal, Result};

pub const OWNED_CONTAINER_LABEL: &str = "optimum-advisor=true";
pub const SERVER_CONTAINER_LABEL: &str = "optimum-advisor.role=server";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessSpec {
    pub program: String,
    pub args: Vec<String>,
}

impl ProcessSpec {
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
        }
    }

    pub fn shell(&self) -> String {
        let mut parts = Vec::with_capacity(self.args.len() + 1);
        parts.push(self.program.clone());
        parts.extend(self.args.clone());
        shell_join(&parts)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunPlan {
    pub server: ProcessSpec,
    pub benchmark: ProcessSpec,
    pub readiness: Readiness,
    pub server_container: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Readiness {
    pub host: String,
    pub port: u16,
    pub timeout: Duration,
    pub http_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BenchmarkRunOutput {
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedRunOutput {
    pub benchmark: BenchmarkRunOutput,
    pub correctness: Option<BenchmarkRunOutput>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvaluationRunOutput {
    pub benchmark: Option<BenchmarkRunOutput>,
    pub correctness: Option<BenchmarkRunOutput>,
}

pub fn execute_run_plan(plan: &RunPlan, mut out: impl Write) -> Result<BenchmarkRunOutput> {
    execute_evaluation_plan(plan, None, true, &mut out)?
        .benchmark
        .ok_or_else(|| "benchmark execution did not produce output".to_string())
}

pub fn execute_run_plan_with_correctness(
    plan: &RunPlan,
    correctness: &ProcessSpec,
    mut out: impl Write,
) -> Result<ManagedRunOutput> {
    let output = execute_evaluation_plan(plan, Some(correctness), true, &mut out)?;
    Ok(ManagedRunOutput {
        benchmark: output
            .benchmark
            .ok_or_else(|| "benchmark execution did not produce output".to_string())?,
        correctness: output.correctness,
    })
}

pub fn execute_evaluation_plan(
    plan: &RunPlan,
    correctness: Option<&ProcessSpec>,
    run_benchmark: bool,
    mut out: impl Write,
) -> Result<EvaluationRunOutput> {
    if correctness.is_none() && !run_benchmark {
        return Err("evaluation requires correctness, benchmark, or both".to_string());
    }
    ensure_port_free(&plan.readiness)?;
    let (server_log, server_log_path) = open_server_log()?;
    terminal::info(&mut out, "server", "starting")?;
    let mut server = Command::new(&plan.server.program)
        .args(&plan.server.args)
        .stdout(Stdio::from(
            server_log
                .try_clone()
                .map_err(|err| format!("failed to clone server log: {err}"))?,
        ))
        .stderr(Stdio::from(server_log))
        .spawn()
        .map_err(|err| format!("failed to start server: {err}"))?;

    let result = (|| {
        wait_for_readiness(&plan.readiness, &mut server)?;
        terminal::ok(&mut out, "server", "ready")?;
        let correctness = correctness
            .map(|spec| run_child("correct", spec, &mut out))
            .transpose()?;
        let benchmark = run_benchmark
            .then(|| run_child("benchmark", &plan.benchmark, &mut out))
            .transpose()?;
        Ok(EvaluationRunOutput {
            benchmark,
            correctness,
        })
    })();

    stop_server(&mut server, plan.server_container.as_deref());
    match result {
        Ok(output) => {
            let _ = std::fs::remove_file(server_log_path);
            Ok(output)
        }
        Err(err) => Err(with_server_log(err, &server_log_path)),
    }
}

pub fn execute_process(
    label: &str,
    spec: &ProcessSpec,
    mut out: impl Write,
) -> Result<BenchmarkRunOutput> {
    run_child(label, spec, &mut out)
}

pub fn resolve_docker_image_tag(image: &str, version_package: Option<&str>) -> Option<String> {
    if image.contains('@') {
        return None;
    }
    if !uses_latest_tag(image) {
        return Some(image.to_string());
    }
    let repo = image_repo(image)?;
    if let Some(tag) =
        inspect_repo_tags(image).and_then(|tags| parse_resolved_image_tag(&tags, repo))
    {
        return Some(tag);
    }
    version_package
        .and_then(|package| inspect_python_package_version(image, package))
        .map(|version| version_to_image_tag(repo, &version))
}

fn inspect_repo_tags(image: &str) -> Option<String> {
    let output = Command::new("docker")
        .args([
            "image",
            "inspect",
            image,
            "--format",
            "{{range .RepoTags}}{{println .}}{{end}}",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

fn inspect_python_package_version(image: &str, package: &str) -> Option<String> {
    let script = format!("import importlib.metadata as m; print(m.version({package:?}))");
    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image,
            "-c",
            &script,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(str::trim)
        .filter(|version| !version.is_empty())
        .map(str::to_string)
}

fn parse_resolved_image_tag(text: &str, repo: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .find(|tag| image_repo(tag) == Some(repo) && !uses_latest_tag(tag))
        .map(str::to_string)
}

fn version_to_image_tag(repo: &str, version: &str) -> String {
    format!("{repo}:v{}", version.trim_start_matches('v'))
}

fn uses_latest_tag(image: &str) -> bool {
    image_tag(image).map(|tag| tag == "latest").unwrap_or(true)
}

fn image_repo(image: &str) -> Option<&str> {
    let image = image.split_once('@').map(|(repo, _)| repo).unwrap_or(image);
    let slash = image.rfind('/');
    let colon = image.rfind(':');
    if colon.is_some_and(|index| slash.map(|slash| index > slash).unwrap_or(true)) {
        return colon.map(|index| &image[..index]);
    }
    Some(image)
}

fn image_tag(image: &str) -> Option<&str> {
    if image.contains('@') {
        return None;
    }
    let slash = image.rfind('/');
    let colon = image.rfind(':')?;
    if slash.map(|slash| colon < slash).unwrap_or(false) {
        return None;
    }
    Some(&image[colon + 1..])
}

fn run_child(label: &str, spec: &ProcessSpec, out: &mut impl Write) -> Result<BenchmarkRunOutput> {
    terminal::info(out, label, "running")?;
    let output = Command::new(&spec.program)
        .args(&spec.args)
        .output()
        .map_err(|err| format!("failed to start {label}: {err}"))?;

    if output.status.success() {
        Ok(BenchmarkRunOutput {
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        })
    } else {
        Err(process_failure(label, &output))
    }
}

pub fn execute_server_plan(plan: &RunPlan, mut out: impl Write) -> Result<()> {
    ensure_port_free(&plan.readiness)?;
    writeln!(out, "starting: {}", plan.server.shell()).map_err(write_error)?;
    let mut server = Command::new(&plan.server.program)
        .args(&plan.server.args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| format!("failed to start server: {err}"))?;

    let result = match server.wait() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("server exited with status {status}")),
        Err(err) => Err(format!("failed to wait for server: {err}")),
    };
    cleanup_server_container(plan.server_container.as_deref());
    result
}

fn wait_for_readiness(readiness: &Readiness, server: &mut std::process::Child) -> Result<()> {
    let deadline = Instant::now() + readiness.timeout;
    let addr = readiness_addr(readiness)?;

    loop {
        let ready = if let Some(path) = &readiness.http_path {
            http_ready(&addr, &readiness.host, path)
        } else {
            TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok()
        };
        if ready {
            return Ok(());
        }
        if let Some(status) = server
            .try_wait()
            .map_err(|err| format!("failed to inspect server process: {err}"))?
        {
            return Err(format!("server exited before becoming ready: {status}"));
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "server did not become ready on {}:{} within {}s",
                readiness.host,
                readiness.port,
                readiness.timeout.as_secs()
            ));
        }
        sleep(Duration::from_millis(500));
    }
}

fn ensure_port_free(readiness: &Readiness) -> Result<()> {
    let addr = readiness_addr(readiness)?;
    if TcpStream::connect_timeout(&addr, Duration::from_millis(250)).is_ok() {
        Err(format!(
            "port {}:{} is already in use; stop the existing server/container or choose a different --port",
            readiness.host, readiness.port
        ))
    } else {
        Ok(())
    }
}

fn stop_server(server: &mut std::process::Child, container: Option<&str>) {
    let _ = server.kill();
    cleanup_server_container(container);
    let _ = server.wait();
    cleanup_server_container(container);
}

fn cleanup_server_container(container: Option<&str>) {
    if let Some(container) = container {
        let _ = cleanup_container(container);
    }
}

fn cleanup_container(container: &str) -> bool {
    Command::new("docker")
        .args(["rm", "-f", container])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn open_server_log() -> Result<(File, PathBuf)> {
    let path = std::env::temp_dir().join(format!(
        "optimum-advisor-server-{}-{}.log",
        std::process::id(),
        now_nanos()
    ));
    let file =
        File::create(&path).map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    Ok((file, path))
}

fn process_failure(label: &str, output: &std::process::Output) -> String {
    let mut message = format!("{label} exited with status {}", output.status);
    let tail = output_tail(&output.stdout, &output.stderr);
    if !tail.trim().is_empty() {
        message.push_str(&format!("\n--- {label} output tail ---\n"));
        message.push_str(tail.trim_end());
    }
    message
}

fn output_tail(stdout: &[u8], stderr: &[u8]) -> String {
    let mut text = String::new();
    push_labeled_tail(&mut text, "stdout", stdout);
    push_labeled_tail(&mut text, "stderr", stderr);
    text
}

fn push_labeled_tail(out: &mut String, label: &str, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let text = String::from_utf8_lossy(bytes);
    let tail = tail_lines(&text, 40);
    if !tail.trim().is_empty() {
        out.push_str(label);
        out.push_str(":\n");
        out.push_str(&tail);
        out.push('\n');
    }
}

fn with_server_log(mut err: String, path: &Path) -> String {
    err.push_str("\nserver_log: ");
    err.push_str(&path.display().to_string());
    if let Ok(tail) = tail_file(path, 64 * 1024) {
        if !tail.trim().is_empty() {
            err.push_str("\n--- server log tail ---\n");
            err.push_str(tail.trim_end());
        }
    }
    err
}

fn tail_file(path: &Path, max_bytes: u64) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len > max_bytes {
        file.seek(SeekFrom::End(-(max_bytes as i64)))?;
    }
    let mut text = String::new();
    file.read_to_string(&mut text)?;
    if len > max_bytes {
        if let Some((_, rest)) = text.split_once('\n') {
            return Ok(format!("...\n{rest}"));
        }
    }
    Ok(text)
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let mut lines = text.lines().rev().take(max_lines).collect::<Vec<_>>();
    lines.reverse();
    lines.join("\n")
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default()
}

fn readiness_addr(readiness: &Readiness) -> Result<std::net::SocketAddr> {
    (readiness.host.as_str(), readiness.port)
        .to_socket_addrs()
        .map_err(|err| format!("invalid readiness address: {err}"))?
        .next()
        .ok_or("readiness address resolved to nothing".to_string())
}

fn http_ready(addr: &std::net::SocketAddr, host: &str, path: &str) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(addr, Duration::from_millis(250)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
    let request = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0; 32];
    match stream.read(&mut response) {
        Ok(n) => String::from_utf8_lossy(&response[..n]).starts_with("HTTP/1.1 200"),
        Err(_) => false,
    }
}

pub fn shell_join(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            if arg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_./:=,".contains(c))
            {
                arg.clone()
            } else {
                format!("'{}'", arg.replace('\'', "'\"'\"'"))
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn write_error(err: std::io::Error) -> String {
    format!("failed to write output: {err}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn quotes_shell_arguments() {
        assert_eq!(shell_join(&["a b".to_string()]), "'a b'");
    }

    #[test]
    fn detects_occupied_readiness_port() {
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(err) => panic!("failed to bind test listener: {err}"),
        };
        let readiness = Readiness {
            host: "127.0.0.1".to_string(),
            port: listener.local_addr().unwrap().port(),
            timeout: Duration::from_secs(1),
            http_path: None,
        };

        assert!(ensure_port_free(&readiness).is_err());
    }

    #[test]
    fn execute_run_plan_keeps_child_output_out_of_terminal() {
        if Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(err) => panic!("failed to bind test listener: {err}"),
        };
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let plan = RunPlan {
            server: ProcessSpec::new(
                "python3",
                vec![
                    "-m".to_string(),
                    "http.server".to_string(),
                    port.to_string(),
                    "--bind".to_string(),
                    "127.0.0.1".to_string(),
                ],
            ),
            benchmark: ProcessSpec::new(
                "sh",
                vec![
                    "-c".to_string(),
                    "printf 'Output token throughput (tok/s): 7\\n'; printf 'noisy stderr\\n' >&2"
                        .to_string(),
                ],
            ),
            readiness: Readiness {
                host: "127.0.0.1".to_string(),
                port,
                timeout: Duration::from_secs(5),
                http_path: None,
            },
            server_container: None,
        };
        let mut terminal = Vec::new();

        let output = execute_run_plan(&plan, &mut terminal).unwrap();

        let terminal = String::from_utf8(terminal).unwrap();
        assert!(terminal.contains("server"));
        assert!(terminal.contains("starting"));
        assert!(terminal.contains("benchmark"));
        assert!(terminal.contains("running"));
        assert!(!terminal.contains("Output token throughput"));
        assert!(!terminal.contains("noisy stderr"));
        assert!(output.stdout.contains("Output token throughput"));
        assert!(output.stderr.contains("noisy stderr"));
    }

    #[test]
    fn execute_run_plan_can_run_correctness_before_benchmark() {
        if Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(err) => panic!("failed to bind test listener: {err}"),
        };
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let plan = RunPlan {
            server: ProcessSpec::new(
                "python3",
                vec![
                    "-m".to_string(),
                    "http.server".to_string(),
                    port.to_string(),
                    "--bind".to_string(),
                    "127.0.0.1".to_string(),
                ],
            ),
            benchmark: ProcessSpec::new("sh", vec!["-c".to_string(), "printf ok".to_string()]),
            readiness: Readiness {
                host: "127.0.0.1".to_string(),
                port,
                timeout: Duration::from_secs(5),
                http_path: None,
            },
            server_container: None,
        };
        let correctness = ProcessSpec::new("sh", vec!["-c".to_string(), "printf ok".to_string()]);
        let mut terminal = Vec::new();

        let output = execute_run_plan_with_correctness(&plan, &correctness, &mut terminal).unwrap();

        let terminal = String::from_utf8(terminal).unwrap();
        assert!(terminal.contains("correct"));
        assert!(!terminal.contains("Output token throughput"));
        assert_eq!(output.correctness.unwrap().stdout, "ok");
        assert!(output.benchmark.stdout.contains("ok"));
    }

    #[test]
    fn evaluation_plan_can_run_correctness_without_benchmark() {
        if Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_err()
        {
            return;
        }
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(listener) => listener,
            Err(err) if err.kind() == std::io::ErrorKind::PermissionDenied => return,
            Err(err) => panic!("failed to bind test listener: {err}"),
        };
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let plan = RunPlan {
            server: ProcessSpec::new(
                "python3",
                vec![
                    "-m".to_string(),
                    "http.server".to_string(),
                    port.to_string(),
                    "--bind".to_string(),
                    "127.0.0.1".to_string(),
                ],
            ),
            benchmark: ProcessSpec::new("sh", vec!["-c".to_string(), "exit 99".to_string()]),
            readiness: Readiness {
                host: "127.0.0.1".to_string(),
                port,
                timeout: Duration::from_secs(5),
                http_path: None,
            },
            server_container: None,
        };
        let correctness = ProcessSpec::new("sh", vec!["-c".to_string(), "printf ok".to_string()]);

        let output = execute_evaluation_plan(&plan, Some(&correctness), false, Vec::new()).unwrap();

        assert_eq!(output.correctness.unwrap().stdout, "ok");
        assert!(output.benchmark.is_none());
    }

    #[test]
    fn execute_process_keeps_child_output_out_of_terminal() {
        let spec = ProcessSpec::new(
            "sh",
            vec!["-c".to_string(), "printf ok; printf noisy >&2".to_string()],
        );
        let mut terminal = Vec::new();

        let output = execute_process("correct", &spec, &mut terminal).unwrap();

        let terminal = String::from_utf8(terminal).unwrap();
        assert!(terminal.contains("correct"));
        assert!(terminal.contains("running"));
        assert!(!terminal.contains("ok"));
        assert!(!terminal.contains("noisy"));
        assert_eq!(output.stdout, "ok");
        assert_eq!(output.stderr, "noisy");
    }

    #[test]
    fn benchmark_failure_keeps_only_a_tail() {
        let output = std::process::Output {
            status: Command::new("sh").arg("-c").arg("exit 1").status().unwrap(),
            stdout: (0..50)
                .map(|i| format!("out{i}\n"))
                .collect::<String>()
                .into_bytes(),
            stderr: b"err\n".to_vec(),
        };

        let message = process_failure("benchmark", &output);

        assert!(message.contains("benchmark exited with status"));
        assert!(!message.contains("out0"));
        assert!(message.contains("out49"));
        assert!(message.contains("err"));
    }

    #[test]
    fn parses_resolved_docker_image_tag() {
        let text = "vllm/vllm-openai:latest\nvllm/vllm-openai:v0.22.0\n";

        assert_eq!(
            parse_resolved_image_tag(text, "vllm/vllm-openai"),
            Some("vllm/vllm-openai:v0.22.0".to_string())
        );
    }

    #[test]
    fn ignores_digest_when_resolving_human_image_tag() {
        assert_eq!(
            parse_resolved_image_tag("vllm/vllm-openai@sha256:abc123\n", "vllm/vllm-openai"),
            None
        );
    }

    #[test]
    fn builds_version_tag_from_package_version() {
        assert_eq!(
            version_to_image_tag("vllm/vllm-openai", "0.22.0"),
            "vllm/vllm-openai:v0.22.0"
        );
    }
}
