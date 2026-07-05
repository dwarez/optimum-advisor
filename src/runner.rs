use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

use crate::Result;

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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Readiness {
    pub host: String,
    pub port: u16,
    pub timeout: Duration,
    pub http_path: Option<String>,
}

pub fn execute_run_plan(plan: &RunPlan, mut out: impl Write) -> Result<()> {
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

    let _ = server.kill();
    let _ = server.wait();
    result
}

fn wait_for_readiness(readiness: &Readiness, server: &mut std::process::Child) -> Result<()> {
    let deadline = Instant::now() + readiness.timeout;
    let addr = (readiness.host.as_str(), readiness.port)
        .to_socket_addrs()
        .map_err(|err| format!("invalid readiness address: {err}"))?
        .next()
        .ok_or("readiness address resolved to nothing")?;

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

    #[test]
    fn quotes_shell_arguments() {
        assert_eq!(shell_join(&["a b".to_string()]), "'a b'");
    }
}
