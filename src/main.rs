use std::{env, io, process::ExitCode};

fn main() -> ExitCode {
    match optimum_advisor::run(env::args().skip(1), io::stdout()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let _ = optimum_advisor::terminal::error(&mut io::stderr(), "error", err);
            ExitCode::from(2)
        }
    }
}
