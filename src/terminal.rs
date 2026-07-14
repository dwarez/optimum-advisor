use std::io::Write;

use crate::error::{Error, ErrorKind, Result};

pub fn error(out: &mut (impl Write + ?Sized), event: &str, message: impl AsRef<str>) -> Result<()> {
    out.write_all(format!("{event}: {}\n", message.as_ref()).as_bytes())
        .map_err(|source| {
            Error::new(ErrorKind::Io, None, "failed to write terminal output").with_source(source)
        })
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
}
