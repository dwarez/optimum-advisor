use std::{env, io, process::ExitCode};

fn main() -> ExitCode {
    match optimum_advisor::run(env::args().skip(1), io::stdout()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}
