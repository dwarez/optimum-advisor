const REDACTED: &[u8] = b"[REDACTED]";

#[derive(Clone, Copy, Debug)]
enum AnsiState {
    Text,
    Escape,
    Csi,
    Osc,
    OscEscape,
    ControlString,
    ControlEscape,
}

pub(crate) struct StreamSanitizer {
    ansi: AnsiState,
    secrets: Vec<Vec<u8>>,
    pending: Vec<u8>,
}

impl StreamSanitizer {
    pub(crate) fn new(secrets: &[&str]) -> Self {
        let mut secrets = secrets
            .iter()
            .filter(|secret| !secret.is_empty())
            .map(|secret| secret.as_bytes().to_vec())
            .collect::<Vec<_>>();
        secrets.sort_unstable_by_key(|secret| std::cmp::Reverse(secret.len()));
        secrets.dedup();
        Self {
            ansi: AnsiState::Text,
            secrets,
            pending: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<u8> {
        for &byte in bytes {
            self.ansi = match self.ansi {
                AnsiState::Text if byte == 0x1b => AnsiState::Escape,
                AnsiState::Text => {
                    if byte >= 0x20 || matches!(byte, b'\n' | b'\r' | b'\t') {
                        self.pending.push(byte);
                    }
                    AnsiState::Text
                }
                AnsiState::Escape => match byte {
                    b'[' => AnsiState::Csi,
                    b']' => AnsiState::Osc,
                    b'P' | b'X' | b'^' | b'_' => AnsiState::ControlString,
                    _ => AnsiState::Text,
                },
                AnsiState::Csi if (0x40..=0x7e).contains(&byte) => AnsiState::Text,
                AnsiState::Csi => AnsiState::Csi,
                AnsiState::Osc if byte == 0x07 => AnsiState::Text,
                AnsiState::Osc if byte == 0x1b => AnsiState::OscEscape,
                AnsiState::Osc => AnsiState::Osc,
                AnsiState::OscEscape if byte == b'\\' => AnsiState::Text,
                AnsiState::OscEscape if byte == 0x1b => AnsiState::OscEscape,
                AnsiState::OscEscape => AnsiState::Osc,
                AnsiState::ControlString if byte == 0x1b => AnsiState::ControlEscape,
                AnsiState::ControlString => AnsiState::ControlString,
                AnsiState::ControlEscape if byte == b'\\' => AnsiState::Text,
                AnsiState::ControlEscape if byte == 0x1b => AnsiState::ControlEscape,
                AnsiState::ControlEscape => AnsiState::ControlString,
            };
        }
        self.drain_redacted(false)
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.drain_redacted(true)
    }

    fn drain_redacted(&mut self, finish: bool) -> Vec<u8> {
        let retain = self
            .secrets
            .first()
            .map_or(0, |secret| secret.len().saturating_sub(1));
        let limit = if finish {
            self.pending.len()
        } else {
            self.pending.len().saturating_sub(retain)
        };
        let mut consumed = 0;
        let mut output = Vec::with_capacity(limit);
        while consumed < limit {
            if let Some(secret) = self
                .secrets
                .iter()
                .find(|secret| self.pending[consumed..].starts_with(secret))
            {
                output.extend_from_slice(REDACTED);
                consumed += secret.len();
            } else {
                output.push(self.pending[consumed]);
                consumed += 1;
            }
        }
        self.pending.drain(..consumed);
        output
    }
}

#[cfg(test)]
pub(crate) fn sanitize_bytes(bytes: &[u8], secrets: &[&str]) -> Vec<u8> {
    let mut sanitizer = StreamSanitizer::new(secrets);
    let mut output = sanitizer.push(bytes);
    output.extend(sanitizer.finish());
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_csi_and_osc_sequences_and_redacts_secrets() {
        let input = b"\x1b[31mred\x1b[0m \x1b]0;title\x07 token";

        assert_eq!(sanitize_bytes(input, &["token"]), b"red  [REDACTED]");
    }

    #[test]
    fn redacts_credentials_split_across_chunks() {
        let mut sanitizer = StreamSanitizer::new(&["token-value"]);
        let mut output = sanitizer.push(b"prefix-token-");
        output.extend(sanitizer.push(b"value-suffix"));
        output.extend(sanitizer.finish());

        assert_eq!(output, b"prefix-[REDACTED]-suffix");
    }
}
