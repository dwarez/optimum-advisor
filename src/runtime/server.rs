use std::{
    io::{Read, Write},
    net::{IpAddr, SocketAddr, TcpStream},
    thread,
    time::{Duration, Instant},
};

use crate::{
    error::{Error, ErrorKind, ExecutionStage, Result},
    runtime::{
        cancel::CancellationToken,
        process::{ManagedProcess, ProcessExecutor, ProcessFailure, ProcessOutcome, ProcessSpec},
    },
};

const READINESS_POLL_INTERVAL: Duration = Duration::from_millis(100);
const READINESS_CONNECT_TIMEOUT: Duration = Duration::from_millis(250);
const READINESS_IO_TIMEOUT: Duration = Duration::from_millis(500);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ReadinessProbe {
    pub address: SocketAddr,
    pub host_header: String,
    pub http_path: Option<String>,
    pub deadline: Instant,
}

impl ReadinessProbe {
    pub(crate) fn new(
        bind_host: IpAddr,
        port: u16,
        http_path: Option<String>,
        timeout: Duration,
    ) -> Self {
        let connect_host = match bind_host {
            IpAddr::V4(address) if address.is_unspecified() => {
                IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
            }
            IpAddr::V6(address) if address.is_unspecified() => {
                IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
            }
            address => address,
        };
        let host_header = match connect_host {
            IpAddr::V4(address) => format!("{address}:{port}"),
            IpAddr::V6(address) => format!("[{address}]:{port}"),
        };
        Self {
            address: SocketAddr::new(connect_host, port),
            host_header,
            http_path,
            deadline: Instant::now() + timeout,
        }
    }
}

pub(crate) struct ManagedServer<'a> {
    process: Option<ManagedProcess<'a>>,
    readiness: ReadinessProbe,
}

impl<'a> ManagedServer<'a> {
    pub(crate) fn start(
        executor: &'a ProcessExecutor,
        spec: &'a ProcessSpec,
        readiness: ReadinessProbe,
        cancellation: &CancellationToken,
    ) -> std::result::Result<Self, ProcessFailure> {
        if TcpStream::connect_timeout(&readiness.address, READINESS_CONNECT_TIMEOUT).is_ok() {
            return Err(ProcessFailure {
                error: Error::new(
                    ErrorKind::Validation,
                    Some(ExecutionStage::Server),
                    format!(
                        "server readiness address {} is already in use",
                        readiness.address
                    ),
                ),
                capture: None,
                cleanup_failure: None,
            });
        }
        let process = executor.spawn(spec, cancellation)?;
        Ok(Self {
            process: Some(process),
            readiness,
        })
    }

    pub(crate) fn wait_ready(&mut self, cancellation: &CancellationToken) -> Result<()> {
        loop {
            if cancellation.is_cancelled() {
                return Err(Error::interrupted(ExecutionStage::Server));
            }
            let process = self
                .process
                .as_mut()
                .expect("managed server exists until shutdown");
            if process.try_wait()?.is_some() {
                let shutdown = self
                    .process
                    .take()
                    .expect("managed server exists until shutdown")
                    .terminate();
                return match shutdown {
                    Ok(_) => Err(Error::new(
                        ErrorKind::ProcessExit,
                        Some(ExecutionStage::Server),
                        "server exited before becoming ready",
                    )),
                    Err(failure) => Err(failure.error),
                };
            }
            if readiness_satisfied(&self.readiness) {
                return Ok(());
            }
            if Instant::now() >= self.readiness.deadline {
                return Err(Error::new(
                    ErrorKind::Timeout,
                    Some(ExecutionStage::Server),
                    format!(
                        "server did not become ready at {} before the startup deadline",
                        self.readiness.address
                    ),
                ));
            }
            thread::sleep(READINESS_POLL_INTERVAL);
        }
    }

    pub(crate) fn is_running(&mut self) -> Result<bool> {
        self.process
            .as_mut()
            .expect("managed server exists until shutdown")
            .try_wait()
            .map(|status| status.is_none())
    }

    pub(crate) fn stop(mut self) -> std::result::Result<ProcessOutcome, ProcessFailure> {
        self.process
            .take()
            .expect("managed server exists until shutdown")
            .terminate()
    }
}

fn readiness_satisfied(readiness: &ReadinessProbe) -> bool {
    let Some(path) = readiness.http_path.as_deref() else {
        return TcpStream::connect_timeout(&readiness.address, READINESS_CONNECT_TIMEOUT).is_ok();
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&readiness.address, READINESS_CONNECT_TIMEOUT)
    else {
        return false;
    };
    if stream.set_read_timeout(Some(READINESS_IO_TIMEOUT)).is_err()
        || stream
            .set_write_timeout(Some(READINESS_IO_TIMEOUT))
            .is_err()
    {
        return false;
    }
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        readiness.host_header
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = [0_u8; 64];
    let Ok(length) = stream.read(&mut response) else {
        return false;
    };
    response[..length].starts_with(b"HTTP/1.1 200 ")
        || response[..length].starts_with(b"HTTP/1.0 200 ")
}

#[cfg(all(test, unix))]
mod tests {
    use std::{net::TcpListener, sync::mpsc};

    use super::*;

    #[test]
    fn readiness_fails_immediately_when_server_exits() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let spec = ProcessSpec::new("sh", ["-c", "exit 0"])
            .with_stage(ExecutionStage::Server)
            .with_timeout(Duration::from_secs(5));
        let executor = ProcessExecutor::default();
        let cancellation = CancellationToken::new();
        let started = Instant::now();
        let mut server = ManagedServer::start(
            &executor,
            &spec,
            ReadinessProbe::new(
                "127.0.0.1".parse().unwrap(),
                port,
                None,
                Duration::from_secs(5),
            ),
            &cancellation,
        )
        .unwrap();

        let error = server.wait_ready(&cancellation).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ProcessExit);
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn keeps_server_managed_until_work_finishes() {
        let reservation = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = reservation.local_addr().unwrap();
        drop(reservation);
        let (stop_sender, stop_receiver) = mpsc::channel();
        let fixture = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            let listener = TcpListener::bind(address).unwrap();
            listener.set_nonblocking(true).unwrap();
            while stop_receiver.try_recv().is_err() {
                let _ = listener.accept();
                thread::sleep(Duration::from_millis(10));
            }
        });
        let spec = ProcessSpec::new("sh", ["-c", "sleep 30"])
            .with_stage(ExecutionStage::Server)
            .with_timeout(Duration::from_secs(10));
        let executor = ProcessExecutor::default();
        let cancellation = CancellationToken::new();
        let mut server = ManagedServer::start(
            &executor,
            &spec,
            ReadinessProbe::new(address.ip(), address.port(), None, Duration::from_secs(2)),
            &cancellation,
        )
        .unwrap();

        server.wait_ready(&cancellation).unwrap();
        assert!(server.is_running().unwrap());
        server.stop().unwrap();
        stop_sender.send(()).unwrap();
        fixture.join().unwrap();
    }

    #[test]
    fn rejects_an_occupied_port_before_spawning() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let spec = ProcessSpec::new("sh", ["-c", "exit 99"]).with_stage(ExecutionStage::Server);
        let executor = ProcessExecutor::default();
        let cancellation = CancellationToken::new();

        let failure = ManagedServer::start(
            &executor,
            &spec,
            ReadinessProbe::new(address.ip(), address.port(), None, Duration::from_secs(1)),
            &cancellation,
        )
        .err()
        .expect("occupied port must fail");

        assert_eq!(failure.error.kind(), ErrorKind::Validation);
    }
}
