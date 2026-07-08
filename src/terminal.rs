use std::io::Write;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Info,
    Ok,
    Error,
}

impl Level {
    fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Ok => "OK",
            Self::Error => "ERROR",
        }
    }

    fn color(self) -> &'static str {
        match self {
            Self::Info => "34",
            Self::Ok => "32",
            Self::Error => "31",
        }
    }
}

pub fn info(out: &mut (impl Write + ?Sized), event: &str, message: impl AsRef<str>) -> Result<()> {
    log(out, Level::Info, event, message)
}

pub fn ok(out: &mut (impl Write + ?Sized), event: &str, message: impl AsRef<str>) -> Result<()> {
    log(out, Level::Ok, event, message)
}

pub fn error(out: &mut (impl Write + ?Sized), event: &str, message: impl AsRef<str>) -> Result<()> {
    log(out, Level::Error, event, message)
}

fn log(
    out: &mut (impl Write + ?Sized),
    level: Level,
    event: &str,
    message: impl AsRef<str>,
) -> Result<()> {
    out.write_all(format_line(level, event, message.as_ref(), color_enabled()).as_bytes())
        .map_err(|err| format!("failed to write output: {err}"))
}

fn format_line(level: Level, event: &str, message: &str, color: bool) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_line_at(level, event, message, color, now)
}

fn format_line_at(level: Level, event: &str, message: &str, color: bool, now: Duration) -> String {
    let timestamp = style(&timestamp(now), "90", color);
    let level = style(&format!("{:<5}", level.label()), level.color(), color);
    let event = style(&format!("{:<9}", event), "36", color);
    format!("{timestamp} {level} {event} {message}\n")
}

fn color_enabled() -> bool {
    std::env::var_os("NO_COLOR").is_none()
        && std::env::var("TERM")
            .map(|term| term != "dumb")
            .unwrap_or(true)
}

fn style(text: &str, code: &str, color: bool) -> String {
    if color {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn timestamp(duration: Duration) -> String {
    let millis = duration.as_millis();
    let seconds = (millis / 1000) % 86_400;
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60,
        millis % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_timestamp_from_epoch_duration() {
        assert_eq!(timestamp(Duration::from_millis(45_296_789)), "12:34:56.789");
    }

    #[test]
    fn plain_line_keeps_message_readable() {
        let line = format_line_at(
            Level::Info,
            "trial",
            "1/4 remaining=3",
            false,
            Duration::ZERO,
        );
        assert_eq!(line, "00:00:00.000 INFO  trial     1/4 remaining=3\n");
    }

    #[test]
    fn colored_line_wraps_prefixes_only() {
        let line = format_line_at(Level::Ok, "metrics", "tps=1.0000", true, Duration::ZERO);
        assert!(line.contains("\x1b[32mOK   \x1b[0m"));
        assert!(line.ends_with("tps=1.0000\n"));
    }
}
