use std::fmt;
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::Result;

pub const DEFAULT_LEADERBOARD_URL: &str = "https://hf-dwarez-optimum-advisor-leaderboard.hf.space";

#[derive(Clone, PartialEq, Eq)]
pub struct LeaderboardConfig {
    pub submit: bool,
    pub url: String,
    pub submit_key: String,
}

impl Default for LeaderboardConfig {
    fn default() -> Self {
        Self {
            submit: false,
            url: DEFAULT_LEADERBOARD_URL.to_string(),
            submit_key: String::new(),
        }
    }
}

impl fmt::Debug for LeaderboardConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LeaderboardConfig")
            .field("submit", &self.submit)
            .field("url", &self.url)
            .field(
                "submit_key",
                &if self.submit_key.is_empty() {
                    "<empty>"
                } else {
                    "<redacted>"
                },
            )
            .finish()
    }
}

impl LeaderboardConfig {
    pub fn apply_env(&mut self) {
        self.apply_from(|name| std::env::var(name).ok());
    }

    fn apply_from(&mut self, get: impl Fn(&str) -> Option<String>) {
        if truthy(get("OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT").as_deref()) {
            self.submit = true;
        }
        if let Some(value) = get("OPTIMUM_ADVISOR_LEADERBOARD_URL") {
            if !value.trim().is_empty() {
                self.url = value;
            }
        }
        if let Some(value) = get("OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT_KEY") {
            self.submit_key = value;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaderboardSubmission {
    pub message: String,
}

#[derive(Clone, PartialEq, Eq)]
pub struct PreparedLeaderboardSubmission {
    config: LeaderboardConfig,
    contributor: String,
    hf_token: Option<String>,
}

impl PreparedLeaderboardSubmission {
    pub fn url(&self) -> &str {
        &self.config.url
    }
}

impl fmt::Debug for PreparedLeaderboardSubmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLeaderboardSubmission")
            .field("config", &self.config)
            .field("contributor", &self.contributor)
            .field(
                "hf_token",
                &if self.hf_token.is_some() {
                    "<redacted>"
                } else {
                    "<empty>"
                },
            )
            .finish()
    }
}

pub fn prepare_submission(
    config: &LeaderboardConfig,
) -> Result<Option<PreparedLeaderboardSubmission>> {
    prepare_submission_with(config, infer_hf_username, infer_hf_token)
}

fn prepare_submission_with(
    config: &LeaderboardConfig,
    infer_username: impl FnOnce() -> Result<String>,
    infer_token: impl FnOnce() -> Result<String>,
) -> Result<Option<PreparedLeaderboardSubmission>> {
    if !config.submit {
        return Ok(None);
    }

    let contributor = infer_username()?;
    let hf_token = match infer_token() {
        Ok(token) => Some(token),
        Err(error) if config.submit_key.is_empty() => return Err(error),
        Err(_) => None,
    };

    Ok(Some(PreparedLeaderboardSubmission {
        config: config.clone(),
        contributor,
        hf_token,
    }))
}

pub fn submit_report_file(
    report_path: impl AsRef<Path>,
    submission: &PreparedLeaderboardSubmission,
) -> Result<LeaderboardSubmission> {
    let report_json = fs::read_to_string(report_path.as_ref())
        .map_err(|err| format!("failed to read {}: {err}", report_path.as_ref().display()))?;
    submit_report_json(&report_json, submission)
}

fn submit_report_json(
    report_json: &str,
    submission: &PreparedLeaderboardSubmission,
) -> Result<LeaderboardSubmission> {
    let payload = submit_payload(
        report_json,
        &submission.contributor,
        &submission.config.submit_key,
    );
    let post_url = submit_url(&submission.config.url);
    let auth_header = authorization_header(submission.hf_token.as_deref());
    let mut post_args = vec![
        "-sS",
        "-X",
        "POST",
        &post_url,
        "-H",
        "Content-Type: application/json",
    ];
    if let Some(header) = auth_header.as_deref() {
        post_args.extend(["-H", header]);
    }
    post_args.extend(["--data-binary", "@-"]);
    let post = run_curl(&post_args, Some(&payload))?;
    let event_id = parse_event_id(&post)?;
    let stream_url = format!("{post_url}/{event_id}");
    let mut stream_args = vec!["-sS", "-N", &stream_url];
    if let Some(header) = auth_header.as_deref() {
        stream_args.extend(["-H", header]);
    }
    let stream = run_curl(&stream_args, None)?;
    parse_submit_stream(&stream)
}

fn infer_hf_username() -> Result<String> {
    for (command, args) in [
        ("hf", &["auth", "whoami"][..]),
        ("huggingface-cli", &["whoami"][..]),
    ] {
        if let Ok(output) = Command::new(command).args(args).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(username) = parse_whoami_username(&stdout) {
                    return Ok(username);
                }
            }
        }
    }
    Err(
        "leaderboard submit requires Hugging Face login; run `hf auth login` or `huggingface-cli login`"
            .to_string(),
    )
}

fn parse_whoami_username(text: &str) -> Option<String> {
    for line in text.lines().map(str::trim).filter(|line| !line.is_empty()) {
        let line = line
            .strip_prefix("Username:")
            .or_else(|| line.strip_prefix("username:"))
            .unwrap_or(line)
            .trim();
        if line.to_ascii_lowercase().contains("not logged") {
            return None;
        }
        let username = line.split_whitespace().next().unwrap_or("");
        if !username.is_empty() {
            return Some(username.to_string());
        }
    }
    None
}

fn infer_hf_token() -> Result<String> {
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(token) = clean_token(std::env::var(name).ok().as_deref()) {
            return Ok(token);
        }
    }
    for (command, args) in [
        ("hf", &["auth", "token"][..]),
        ("huggingface-cli", &["token"][..]),
    ] {
        if let Ok(output) = Command::new(command).args(args).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(token) = clean_token(Some(&stdout)) {
                    return Ok(token);
                }
            }
        }
    }
    for path in hf_token_paths() {
        if let Ok(token) = fs::read_to_string(path) {
            if let Some(token) = clean_token(Some(&token)) {
                return Ok(token);
            }
        }
    }
    Err("leaderboard submit requires a Hugging Face token; run `hf auth login`".to_string())
}

fn hf_token_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(path) = std::env::var("HF_TOKEN_PATH") {
        paths.push(PathBuf::from(path));
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        paths.push(PathBuf::from(home).join("token"));
    }
    if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
        paths.push(PathBuf::from(cache).join("huggingface").join("token"));
    }
    if let Ok(home) = std::env::var("HOME") {
        paths.push(PathBuf::from(home).join(".cache/huggingface/token"));
    }
    paths
}

fn clean_token(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(str::to_string)
}

fn authorization_header(token: Option<&str>) -> Option<String> {
    clean_token(token).map(|token| format!("Authorization: Bearer {token}"))
}

fn run_curl(args: &[&str], stdin: Option<&str>) -> Result<String> {
    let mut command = Command::new("curl");
    command.args(args);
    command.args(["-w", "\n%{http_code}"]);
    if stdin.is_some() {
        command.stdin(Stdio::piped());
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("failed to run curl: {err}"))?;
    if let Some(input) = stdin {
        child
            .stdin
            .as_mut()
            .ok_or("failed to open curl stdin")?
            .write_all(input.as_bytes())
            .map_err(|err| format!("failed to write curl request body: {err}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|err| format!("failed to wait for curl: {err}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !output.status.success() {
        Err(format!(
            "curl failed: {}",
            if stderr.is_empty() {
                stdout.trim()
            } else {
                &stderr
            }
        ))
    } else {
        checked_http_body(&stdout, &stderr)
    }
}

fn checked_http_body(stdout: &str, stderr: &str) -> Result<String> {
    let (body, status) = stdout
        .rsplit_once('\n')
        .ok_or_else(|| "curl returned no HTTP status".to_string())?;
    let status = status
        .trim()
        .parse::<u16>()
        .map_err(|_| "curl returned an invalid HTTP status".to_string())?;
    if (200..300).contains(&status) {
        return Ok(body.to_string());
    }
    let detail = body.trim();
    if detail.is_empty() {
        Err(format!("leaderboard HTTP {status}: {stderr}"))
    } else {
        Err(format!("leaderboard HTTP {status}: {detail}"))
    }
}

fn submit_url(base_url: &str) -> String {
    format!(
        "{}/gradio_api/call/submit_report",
        base_url.trim_end_matches('/')
    )
}

fn submit_payload(report_json: &str, contributor: &str, submit_key: &str) -> String {
    format!(
        "{{\"data\":[{},{},{}]}}",
        json_string(report_json),
        json_string(contributor),
        json_string(submit_key)
    )
}

fn parse_event_id(text: &str) -> Result<String> {
    json_string_field(text, "event_id")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "leaderboard submit did not return an event_id".to_string())
}

fn parse_submit_stream(text: &str) -> Result<LeaderboardSubmission> {
    if let Some(message) = prefixed_message(text, "Rejected:") {
        return Err(format!("leaderboard rejected report: {message}"));
    }
    for prefix in ["Accepted:", "Queued for review:"] {
        if let Some(message) = prefixed_message(text, prefix) {
            return Ok(LeaderboardSubmission { message });
        }
    }
    Err("leaderboard submit returned no accepted/queued result".to_string())
}

fn prefixed_message(text: &str, prefix: &str) -> Option<String> {
    let start = text.find(prefix)?;
    let rest = &text[start..];
    let end = rest
        .find(|ch| matches!(ch, '\n' | '\r' | '"' | ']'))
        .unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn truthy(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn json_string_field(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let value = text.split(&needle).nth(1)?.split_once(':')?.1.trim_start();
    let value = value.strip_prefix('"')?;
    let mut out = String::new();
    let mut escaped = false;
    for ch in value.chars() {
        if escaped {
            out.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(out);
        } else {
            out.push(ch);
        }
    }
    None
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_contains_empty_key_when_missing() {
        let payload = submit_payload("{\"x\":1}", "hf-dwarez", "");

        assert_eq!(payload, "{\"data\":[\"{\\\"x\\\":1}\",\"hf-dwarez\",\"\"]}");
        assert!(!payload.contains("submit_key"));
    }

    #[test]
    fn authorization_header_uses_bearer_token() {
        assert_eq!(
            authorization_header(Some("hf_test")).as_deref(),
            Some("Authorization: Bearer hf_test")
        );
        assert_eq!(authorization_header(Some(" ")), None);
    }

    #[test]
    fn submit_url_uses_gradio_api_mount() {
        assert_eq!(
            submit_url("https://example.test/"),
            "https://example.test/gradio_api/call/submit_report"
        );
    }

    #[test]
    fn http_errors_include_response_body() {
        let err = checked_http_body(
            "{\"type\":\"error\",\"error\":{\"message\":\"Method Not Allowed\"}}\n405",
            "",
        )
        .unwrap_err();

        assert!(err.contains("405"));
        assert!(err.contains("Method Not Allowed"));
    }

    #[test]
    fn parses_accepted_and_queued_streams() {
        let accepted =
            parse_submit_stream("event: complete\ndata: [\"Accepted: abc123\"]").unwrap();
        let queued =
            parse_submit_stream("event: complete\ndata: [\"Queued for review: xyz\"]").unwrap();

        assert_eq!(accepted.message, "Accepted: abc123");
        assert_eq!(queued.message, "Queued for review: xyz");
    }

    #[test]
    fn rejected_stream_is_an_error() {
        let err =
            parse_submit_stream("event: complete\ndata: [\"Rejected: bad report\"]").unwrap_err();

        assert!(err.contains("Rejected: bad report"));
    }

    #[test]
    fn env_fallbacks_enable_submit_without_key() {
        let mut config = LeaderboardConfig::default();
        config.apply_from(|name| match name {
            "OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT" => Some("1".to_string()),
            _ => None,
        });

        assert!(config.submit);
        assert_eq!(config.submit_key, "");
    }

    #[test]
    fn parses_hf_login_username() {
        assert_eq!(
            parse_whoami_username("hf-dwarez\n"),
            Some("hf-dwarez".to_string())
        );
        assert_eq!(
            parse_whoami_username("Username: hf-dwarez\norgs: test\n"),
            Some("hf-dwarez".to_string())
        );
        assert_eq!(parse_whoami_username("Not logged in"), None);
    }

    #[test]
    fn prepare_skips_login_when_submit_disabled() {
        let prepared = prepare_submission_with(
            &LeaderboardConfig::default(),
            || panic!("disabled submissions must not inspect HF login"),
            || panic!("disabled submissions must not inspect HF token"),
        )
        .unwrap();

        assert!(prepared.is_none());
    }

    #[test]
    fn prepare_fails_fast_when_submit_enabled_without_login() {
        let config = LeaderboardConfig {
            submit: true,
            ..Default::default()
        };

        let err = prepare_submission_with(
            &config,
            || Err("missing login".to_string()),
            || panic!("login failure should stop before token lookup"),
        )
        .unwrap_err();

        assert_eq!(err, "missing login");
    }

    #[test]
    fn prepare_requires_token_without_submit_key() {
        let config = LeaderboardConfig {
            submit: true,
            ..Default::default()
        };

        let err = prepare_submission_with(
            &config,
            || Ok("hf-dwarez".to_string()),
            || Err("missing token".to_string()),
        )
        .unwrap_err();

        assert_eq!(err, "missing token");
    }

    #[test]
    fn debug_redacts_hf_token() {
        let config = LeaderboardConfig {
            submit: true,
            ..Default::default()
        };
        let prepared = prepare_submission_with(
            &config,
            || Ok("hf-dwarez".to_string()),
            || Ok("hf_secret".to_string()),
        )
        .unwrap()
        .unwrap();

        let debug = format!("{prepared:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("hf_secret"));
    }

    #[test]
    fn debug_redacts_submit_key() {
        let config = LeaderboardConfig {
            submit_key: "secret".to_string(),
            ..Default::default()
        };

        let debug = format!("{config:?}");

        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("secret"));
    }
}
