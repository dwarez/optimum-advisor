use std::{env, io, process::ExitCode};

fn main() -> ExitCode {
    let mut args = env::args().skip(1).peekable();
    let result = if args.peek().map(String::as_str) == Some("mcp") {
        args.next();
        if args.next().is_some() {
            Err("mcp accepts no arguments; it communicates over stdin/stdout".to_string())
        } else {
            optimum_advisor::mcp::serve(io::stdin().lock(), io::stdout().lock())
        }
    } else {
        optimum_advisor::run(args, io::stdout())
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let _ = optimum_advisor::terminal::error(&mut io::stderr(), "error", err);
            ExitCode::from(2)
        }
    }
}
