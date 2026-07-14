use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, ErrorKind, ExecutionStage, Result},
    runtime::{
        cancel::CancellationToken,
        process::{CapturePolicy, ProcessCapture, ProcessExecutor, ProcessSpec},
        sanitize::StreamSanitizer,
    },
};

use super::auth::Secret;

const MAX_RESPONSE_BYTES: u64 = 1024 * 1024;
const LOGIN_TIMEOUT: Duration = Duration::from_secs(10);
const USER_AGENT: &str = concat!("optimum-advisor/", env!("CARGO_PKG_VERSION"));
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LeaderboardSubmission {
    pub message: String,
}

#[derive(Clone, Debug)]
pub(crate) struct LeaderboardClient {
    agent: ureq::Agent,
    base_url: String,
}

impl LeaderboardClient {
    pub(crate) fn new(base_url: &str, timeout: Duration) -> Result<Self> {
        validate_base_url(base_url)?;
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .max_redirects(0)
            .http_status_as_error(false)
            .user_agent(USER_AGENT)
            .build();
        Ok(Self {
            agent: config.into(),
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    pub(crate) fn submit_report(
        &self,
        report_json: &str,
        contributor: &str,
        submit_key: Option<&Secret>,
        hf_token: Option<&Secret>,
    ) -> Result<LeaderboardSubmission> {
        let payload = SubmitPayload {
            data: [
                report_json,
                contributor,
                submit_key.map_or("", Secret::expose),
            ],
        };
        let url = format!("{}/gradio_api/call/submit_report", self.base_url);
        let mut request = self.agent.post(&url);
        if let Some(token) = hf_token {
            request = request.header("Authorization", format!("Bearer {}", token.expose()));
        }
        let response = request
            .send_json(&payload)
            .map_err(|source| transport_error("leaderboard submission request failed", source))?;
        let event: EventId = read_json_response(response, &[submit_key, hf_token])?;
        if event.event_id.is_empty()
            || !event
                .event_id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(protocol_error("leaderboard returned an invalid event_id"));
        }

        let stream_url = format!("{url}/{}", event.event_id);
        let mut request = self.agent.get(&stream_url);
        if let Some(token) = hf_token {
            request = request.header("Authorization", format!("Bearer {}", token.expose()));
        }
        let response = request
            .call()
            .map_err(|source| transport_error("leaderboard event request failed", source))?;
        let stream = read_text_response(response, &[submit_key, hf_token])?;
        parse_submit_stream(&stream)
    }
}

#[derive(Serialize)]
struct SubmitPayload<'a> {
    data: [&'a str; 3],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EventId {
    event_id: String,
}

#[derive(Deserialize)]
struct WhoAmI {
    name: String,
}

pub(crate) fn infer_hf_username(token: Option<&Secret>) -> Result<String> {
    if let Some(token) = token {
        let client = LeaderboardClient::new("https://huggingface.co", LOGIN_TIMEOUT)?;
        let mut response = client
            .agent
            .get("https://huggingface.co/api/whoami-v2")
            .header("Authorization", format!("Bearer {}", token.expose()))
            .call()
            .map_err(|source| transport_error("Hugging Face whoami request failed", source))?;
        if response.status().is_success() {
            let whoami: WhoAmI = response
                .body_mut()
                .with_config()
                .limit(MAX_RESPONSE_BYTES)
                .read_json()
                .map_err(|source| {
                    protocol_error("invalid Hugging Face whoami response").with_source(source)
                })?;
            if !whoami.name.trim().is_empty() {
                return Ok(whoami.name);
            }
        }
    }

    let executor = ProcessExecutor::default();
    let cancellation = CancellationToken::new();
    for (program, arguments) in [
        ("hf", &["auth", "whoami"][..]),
        ("huggingface-cli", &["whoami"][..]),
    ] {
        let spec = ProcessSpec::new(program, arguments)
            .with_stage(ExecutionStage::Leaderboard)
            .with_timeout(LOGIN_TIMEOUT)
            .with_capture(CapturePolicy::Secret)
            .with_safe_display(format!("{program} <whoami>"));
        if let Ok(output) = executor.execute(&spec, &cancellation) {
            if let ProcessCapture::Secret(text) = output.capture {
                if let Some(username) = parse_whoami_username(text.expose()) {
                    return Ok(username);
                }
            }
        }
    }
    Err(auth_error(
        "leaderboard submit requires Hugging Face login; run `hf auth login`",
    ))
}

fn read_json_response<T: for<'de> Deserialize<'de>>(
    response: http::Response<ureq::Body>,
    secrets: &[Option<&Secret>],
) -> Result<T> {
    let text = read_text_response(response, secrets)?;
    serde_json::from_str(&text)
        .map_err(|source| protocol_error("leaderboard returned invalid JSON").with_source(source))
}

fn read_text_response(
    mut response: http::Response<ureq::Body>,
    secrets: &[Option<&Secret>],
) -> Result<String> {
    let status = response.status().as_u16();
    let body = response
        .body_mut()
        .with_config()
        .limit(MAX_RESPONSE_BYTES)
        .read_to_string()
        .map_err(|source| {
            protocol_error("leaderboard response exceeded its limit or was unreadable")
                .with_source(source)
        })?;
    if !(200..300).contains(&status) {
        let detail = sanitize_detail(&body, secrets);
        return Err(
            protocol_error(format!("leaderboard HTTP {status}: {detail}")).with_http_status(status),
        );
    }
    Ok(body)
}

fn sanitize_detail(text: &str, secrets: &[Option<&Secret>]) -> String {
    let exposed = secrets
        .iter()
        .flatten()
        .map(|secret| secret.expose())
        .collect::<Vec<_>>();
    let mut sanitizer = StreamSanitizer::new(&exposed);
    let mut bytes = sanitizer.push(text.as_bytes());
    bytes.extend(sanitizer.finish());
    String::from_utf8_lossy(&bytes).chars().take(4096).collect()
}

fn parse_submit_stream(text: &str) -> Result<LeaderboardSubmission> {
    for event in text.replace("\r\n", "\n").split("\n\n") {
        let mut kind = None;
        let mut data = String::new();
        for line in event.lines() {
            if let Some(value) = line.strip_prefix("event:") {
                kind = Some(value.trim());
            } else if let Some(value) = line.strip_prefix("data:") {
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(value.trim_start());
            }
        }
        if kind != Some("complete") || data.is_empty() {
            continue;
        }
        let values: Vec<String> = serde_json::from_str(&data).map_err(|source| {
            protocol_error("leaderboard completion event was not a string array")
                .with_source(source)
        })?;
        let Some(message) = values.first() else {
            continue;
        };
        if message.starts_with("Rejected:") {
            return Err(protocol_error(format!(
                "leaderboard rejected report: {message}"
            )));
        }
        if message.starts_with("Accepted:") || message.starts_with("Queued for review:") {
            return Ok(LeaderboardSubmission {
                message: message.clone(),
            });
        }
    }
    Err(protocol_error(
        "leaderboard submit returned no accepted or queued completion event",
    ))
}

fn parse_whoami_username(text: &str) -> Option<String> {
    let lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
    for line in lines {
        if line.to_ascii_lowercase().contains("not logged") {
            return None;
        }
        if let Some(username) = line
            .strip_prefix("Username:")
            .or_else(|| line.strip_prefix("username:"))
            .or_else(|| line.strip_prefix("user:"))
        {
            let username = username.split_whitespace().next()?;
            if !username.is_empty() {
                return Some(username.to_string());
            }
        }
    }
    text.lines()
        .map(str::trim)
        .find(|line| {
            !line.is_empty()
                && *line != "✓"
                && !line.to_ascii_lowercase().contains("logged in")
                && !line.to_ascii_lowercase().starts_with("orgs:")
        })
        .and_then(|line| line.split_whitespace().next())
        .map(str::to_string)
}

fn validate_base_url(url: &str) -> Result<()> {
    let trimmed = url.trim();
    let secure = trimmed.starts_with("https://");
    let loopback = trimmed
        .strip_prefix("http://")
        .and_then(|rest| rest.split(['/', ':']).next())
        .is_some_and(|host| matches!(host, "127.0.0.1" | "localhost" | "[::1]"));
    if (!secure && !loopback) || trimmed.contains(['\r', '\n']) {
        return Err(Error::validation(
            "leaderboard URL must use HTTPS (HTTP is allowed only for loopback tests)",
        ));
    }
    Ok(())
}

fn transport_error(message: impl Into<String>, source: ureq::Error) -> Error {
    Error::new(
        ErrorKind::HttpTransport,
        Some(ExecutionStage::Leaderboard),
        message,
    )
    .with_source(source)
}

fn protocol_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::HttpProtocol,
        Some(ExecutionStage::Leaderboard),
        message,
    )
}

fn auth_error(message: impl Into<String>) -> Error {
    Error::new(
        ErrorKind::Validation,
        Some(ExecutionStage::Leaderboard),
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        thread,
        time::Instant,
    };

    #[test]
    fn parses_only_typed_completion_events() {
        let accepted =
            parse_submit_stream("event: complete\ndata: [\"Accepted: abc123\"]\n\n").unwrap();
        assert_eq!(accepted.message, "Accepted: abc123");
        assert!(
            parse_submit_stream("event: complete\ndata: [\"Rejected: bad report\"]\n\n",).is_err()
        );
        assert!(parse_submit_stream("Accepted: substring only").is_err());
    }

    #[test]
    fn native_client_rejects_redirects_and_does_not_forward_auth() {
        let target = TcpListener::bind("127.0.0.1:0").unwrap();
        target.set_nonblocking(true).unwrap();
        let target_address = target.local_addr().unwrap();
        let source = TcpListener::bind("127.0.0.1:0").unwrap();
        let source_address = source.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = source.accept().unwrap();
            let request = read_http_request(&mut stream);
            assert!(String::from_utf8_lossy(&request)
                .to_ascii_lowercase()
                .contains("authorization: bearer hf_secret"));
            write!(
                stream,
                "HTTP/1.1 302 Found\r\nLocation: http://{target_address}/stolen\r\nContent-Length: 0\r\n\r\n"
            )
            .unwrap();
        });
        let client =
            LeaderboardClient::new(&format!("http://{source_address}"), Duration::from_secs(1))
                .unwrap();
        let token = Secret::new("hf_secret").unwrap();
        let error = client
            .submit_report("{}", "user", None, Some(&token))
            .unwrap_err();
        server.join().unwrap();

        assert_eq!(error.context.http_status, Some(302));
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            assert!(target.accept().is_err());
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut chunk = [0_u8; 1024];
        loop {
            let count = stream.read(&mut chunk).unwrap();
            assert!(count > 0, "connection closed before request completed");
            request.extend_from_slice(&chunk[..count]);
            let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .map(str::trim)
                .map(str::parse::<usize>)
                .transpose()
                .unwrap()
                .unwrap_or_default();
            if request.len() >= header_end + 4 + content_length {
                return request;
            }
        }
    }

    #[test]
    fn rejects_non_tls_remote_urls() {
        assert!(LeaderboardClient::new("http://example.com", Duration::from_secs(1)).is_err());
        assert!(LeaderboardClient::new("https://example.com", Duration::from_secs(1)).is_ok());
    }

    #[test]
    fn parses_cli_login_without_accepting_logged_out_output() {
        assert_eq!(
            parse_whoami_username("Username: hf-dwarez\norgs: test\n").as_deref(),
            Some("hf-dwarez")
        );
        assert_eq!(parse_whoami_username("Not logged in"), None);
    }
}
