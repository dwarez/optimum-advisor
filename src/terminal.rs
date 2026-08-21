use std::{
    io::{IsTerminal, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::error::{Error, ErrorKind, Result};

pub fn error(out: &mut (impl Write + ?Sized), event: &str, message: impl AsRef<str>) -> Result<()> {
    out.write_all(format!("{event}: {}\n", message.as_ref()).as_bytes())
        .map_err(|source| {
            Error::new(ErrorKind::Io, None, "failed to write terminal output").with_source(source)
        })
}

const SPINNER_FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPINNER_TICK: Duration = Duration::from_millis(100);

/// Animated single-line status on stderr. Active only when stderr is an
/// interactive terminal; otherwise `start` returns `None` and callers emit
/// plain lines only.
pub(crate) struct Spinner {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Spinner {
    pub(crate) fn start(message: &str) -> Option<Self> {
        let dumb_terminal = std::env::var("TERM").is_ok_and(|value| value == "dumb");
        if dumb_terminal || !std::io::stderr().is_terminal() {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let message = message.to_string();
        let started = Instant::now();
        let handle = std::thread::spawn(move || {
            let mut stderr = std::io::stderr();
            let mut frame = 0usize;
            while !thread_stop.load(Ordering::Relaxed) {
                let _ = write!(
                    stderr,
                    "\r\x1b[K{} {} ({})",
                    SPINNER_FRAMES[frame % SPINNER_FRAMES.len()],
                    message,
                    format_elapsed(started.elapsed())
                );
                let _ = stderr.flush();
                frame += 1;
                std::thread::sleep(SPINNER_TICK);
            }
            let _ = write!(stderr, "\r\x1b[K");
            let _ = stderr.flush();
        });
        Some(Self {
            stop,
            handle: Some(handle),
        })
    }

    pub(crate) fn stop(mut self) {
        self.halt();
    }

    fn halt(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Spinner {
    fn drop(&mut self) {
        self.halt();
    }
}

pub(crate) fn format_elapsed(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    if seconds >= 60 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_one_deterministic_plain_line() {
        let mut output = Vec::new();
        error(&mut output, "error", "invalid config").unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "error: invalid config\n"
        );
    }

    #[test]
    fn formats_elapsed_seconds_below_a_minute() {
        assert_eq!(format_elapsed(Duration::from_secs(0)), "0s");
        assert_eq!(format_elapsed(Duration::from_secs(59)), "59s");
    }

    #[test]
    fn formats_elapsed_minutes_and_seconds() {
        assert_eq!(format_elapsed(Duration::from_secs(60)), "1m00s");
        assert_eq!(format_elapsed(Duration::from_secs(192)), "3m12s");
    }
}
