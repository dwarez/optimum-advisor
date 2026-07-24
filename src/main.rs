mod app;
mod cli;
mod config;
mod domain;
mod engines;
mod error;
mod hf_jobs;
mod inspection;
mod leaderboard;
mod mcp;
mod results;
mod runtime;
mod sweep;
mod terminal;

use std::{env, io, process::ExitCode};

fn main() -> ExitCode {
    let mut args = env::args().skip(1).peekable();
    if args.peek().map(String::as_str) == Some("mcp") {
        args.next();
        if args.next().is_some() {
            let _ = terminal::error(
                &mut io::stderr(),
                "error",
                "mcp accepts no arguments; it communicates over stdin/stdout",
            );
            return ExitCode::from(2);
        }
        return finish(mcp::serve(
            io::BufReader::new(io::stdin()),
            io::stdout().lock(),
        ));
    }

    finish(app::run(args, io::stdout(), io::stderr()))
}

fn finish(result: error::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let code = err.exit_code();
            let mut message = err.to_string();
            if let Some(report_path) = &err.context.report_path {
                message.push_str(&format!("\nreport: {}", report_path.display()));
            }
            let _ = terminal::error(&mut io::stderr(), "error", message);
            ExitCode::from(code)
        }
    }
}
