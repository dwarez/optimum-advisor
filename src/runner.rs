use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::Result;

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

pub fn execute_run_plan(plan: &RunPlan, mut out: impl Write) -> Result<()> {
    ensure_port_free(&plan.readiness)?;
    writeln!(out, "starting: {}", plan.server.shell()).map_err(write_error)?;
    let mut server = Command::new(&plan.server.program)
        .args(&plan.server.args)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|err| format!("failed to start server: {err}"))?;

    let result = (|| {
        wait_for_readiness(&plan.readiness, &mut server)?;
        writeln!(out, "benchmark: {}", plan.benchmark.shell()).map_err(write_error)?;
        let status = Command::new(&plan.benchmark.program)
            .args(&plan.benchmark.args)
            .status()
            .map_err(|err| format!("failed to start benchmark: {err}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("benchmark exited with status {status}"))
        }
    })();

    stop_server(&mut server, plan.server_container.as_deref());
    result
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
}
