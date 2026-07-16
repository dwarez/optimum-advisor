//! Launcher that offloads an evaluation to Hugging Face Jobs.
//!
//! The local process never runs the engine here; it submits an `hf jobs run`
//! whose container downloads the prebuilt binary, materializes the config, and
//! runs `optimum-advisor <bench|sweep> --in-container`. Results are always
//! written to a local directory inside the job (atomic checkpoints stay on a
//! real filesystem) and then transferred once at the end — copied into the
//! mounted read-write bucket when `--results-bucket` is given, otherwise the
//! report is emitted to the job logs. The transfer also runs when the
//! evaluation fails, so failed-trial evidence is preserved.

use std::{
    ffi::{OsStr, OsString},
    io::Write,
    path::Path,
    process::Command,
};

use crate::{
    cli::args::Invocation,
    engines::managed::safe_display,
    error::{Error, ErrorKind, ExecutionStage, Result},
    leaderboard::auth::resolve_hf_token,
    results::report::RunKind,
    runtime::{cancel::CancellationToken, process::ProcessExecutor},
};

const BINARY_PATH: &str = "/tmp/optimum-advisor";
const CONFIG_PATH: &str = "/tmp/optimum-advisor-config.toml";
const CORRECTNESS_VENV: &str = "/tmp/optimum-advisor-correctness";
/// One-entry directory prepended to PATH so `lighteval` resolves from the
/// isolated venv while `python3` keeps resolving to the base image's
/// interpreter (which the engine introspection imports vllm/sglang from).
const SHIM_DIR: &str = "/tmp/optimum-advisor-bin";
const RESULTS_MOUNT: &str = "/results";
/// Local staging directory inside the job. Checkpoints need atomic same-dir
/// rename and directory fsync, which bucket FUSE mounts do not reliably
/// provide, so the run never writes to the mount directly.
const LOCAL_RESULTS: &str = "/tmp/optimum-advisor-results";

/// Pinned correctness tools, mirroring `scripts/setup-correctness-env.sh`.
const CORRECTNESS_PACKAGES: [&str; 4] = [
    "lighteval==0.13.0",
    "litellm==1.66.0",
    "diskcache==5.6.3",
    "langdetect==1.0.9",
];

pub(crate) fn submit(
    invocation: &Invocation,
    kind: RunKind,
    out: &mut impl Write,
    progress: &mut impl Write,
) -> Result<()> {
    let normalized = invocation.input.clone().normalize()?;
    let settings = &invocation.hf_jobs;
    let flavor = settings
        .flavor
        .as_deref()
        .ok_or_else(|| Error::usage("--on hf-jobs requires --hf-flavor"))?;
    let config_path = invocation
        .config_path
        .as_deref()
        .ok_or_else(|| Error::usage("--on hf-jobs requires --config"))?;
    let config_b64 = encode_config(config_path)?;

    if let Some(bucket) = &settings.results_bucket {
        validate_bucket(bucket)?;
    }

    let subcommand = match kind {
        RunKind::Bench => "bench",
        RunKind::Sweep => "sweep",
    };
    validate_binary_url(&settings.binary_url)?;
    // Forward the effective image to the in-job bench so the report's image
    // identity always matches the container the job actually runs on (the
    // shipped config file may carry a different or absent `image`). `sweep`
    // takes no CLI overrides, so its config image is the job image already.
    let forwarded_image = match kind {
        RunKind::Bench => {
            validate_shell_safe(&normalized.image, "image reference")?;
            Some(normalized.image.as_str())
        }
        RunKind::Sweep => None,
    };
    let persist_to_bucket = settings.results_bucket.is_some();
    let bootstrap = build_bootstrap(
        &settings.binary_url,
        &config_b64,
        subcommand,
        forwarded_image,
        normalized.correctness.enabled,
        persist_to_bucket,
    );

    let mut args: Vec<OsString> = vec![
        OsString::from("jobs"),
        OsString::from("run"),
        OsString::from("--flavor"),
        OsString::from(flavor),
    ];
    if let Some(timeout) = &settings.timeout {
        args.push(OsString::from("--timeout"));
        args.push(OsString::from(timeout));
    }
    if let Some(namespace) = &settings.namespace {
        args.push(OsString::from("--namespace"));
        args.push(OsString::from(namespace));
    }
    if token_present() {
        // `hf` reads the local token by name and encrypts it server-side.
        args.push(OsString::from("--secrets"));
        args.push(OsString::from("HF_TOKEN"));
    }
    if let Some(bucket) = &settings.results_bucket {
        args.push(OsString::from("-v"));
        args.push(OsString::from(format!("{bucket}:{RESULTS_MOUNT}:rw")));
    }
    if settings.detach {
        args.push(OsString::from("--detach"));
    }
    args.push(OsString::from(&normalized.image));
    args.push(OsString::from("sh"));
    args.push(OsString::from("-c"));
    args.push(OsString::from(bootstrap));

    // `--dry-run` renders the exact submission without contacting Hugging Face.
    if !invocation.execute {
        writeln_checked(out, &safe_display(OsStr::new("hf"), &args))?;
        return Ok(());
    }

    writeln_checked(
        progress,
        &format!("submitting {subcommand} to Hugging Face Jobs (flavor {flavor})"),
    )?;
    // Surface the pinned binary so a stale local checkout (whose version pins
    // an older release) is visible at submit time instead of failing in-job.
    writeln_checked(progress, &format!("in-job binary: {}", settings.binary_url))?;
    run_hf(&args)?;
    if let Some(bucket) = &settings.results_bucket {
        writeln_checked(out, &format!("results: {bucket}"))?;
    } else {
        writeln_checked(
            out,
            "results: emitted to the job logs (pass --results-bucket to persist them)",
        )?;
    }
    Ok(())
}

fn build_bootstrap(
    binary_url: &str,
    config_b64: &str,
    subcommand: &str,
    forwarded_image: Option<&str>,
    correctness: bool,
    persist_to_bucket: bool,
) -> String {
    let mut script = String::new();
    script.push_str("set -eu\n");
    // python3 is the one guaranteed tool in the engine images (the parameter
    // probes already depend on it), so it also performs the download.
    script.push_str(&format!(
        "python3 -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' '{binary_url}' {BINARY_PATH}\n"
    ));
    script.push_str(&format!("chmod +x {BINARY_PATH}\n"));
    script.push_str(&format!(
        "printf %s '{config_b64}' | base64 -d > {CONFIG_PATH}\n"
    ));
    if correctness {
        // Install the correctness tools in an isolated venv so vLLM/SGLang's
        // own environment is never mutated. Expose ONLY the `lighteval`
        // entrypoint through a one-entry shim directory: prepending the whole
        // venv bin would shadow `python3`, and the engine parameter
        // introspection and capability probes must keep resolving the base
        // image's python (the venv interpreter cannot import vllm/sglang).
        script.push_str(&format!("uv venv {CORRECTNESS_VENV}\n"));
        script.push_str(&format!(
            "VIRTUAL_ENV={CORRECTNESS_VENV} uv pip install --quiet {}\n",
            CORRECTNESS_PACKAGES
                .iter()
                .map(|package| format!("'{package}'"))
                .collect::<Vec<_>>()
                .join(" ")
        ));
        script.push_str(&format!("mkdir -p {SHIM_DIR}\n"));
        script.push_str(&format!(
            "ln -sf {CORRECTNESS_VENV}/bin/lighteval {SHIM_DIR}/lighteval\n"
        ));
        script.push_str(&format!("export PATH=\"{SHIM_DIR}:$PATH\"\n"));
    }
    script.push_str(&format!("mkdir -p {LOCAL_RESULTS}\n"));
    let image_argument = forwarded_image
        .map(|image| format!(" --image {image}"))
        .unwrap_or_default();
    // Capture the exit code instead of aborting so results are transferred and
    // failed-trial evidence survives; the job still exits with the real status.
    script.push_str("rc=0\n");
    script.push_str(&format!(
        "{BINARY_PATH} {subcommand} --in-container --config {CONFIG_PATH} --results-dir {LOCAL_RESULTS}{image_argument} || rc=$?\n"
    ));
    if persist_to_bucket {
        // One bulk copy at the end; plain writes only, no rename/fsync games
        // against the FUSE mount. A failed copy must fail the job, but never
        // mask the evaluation's own exit code.
        script.push_str(&format!(
            "if ! cp -r {LOCAL_RESULTS}/. {RESULTS_MOUNT}/; then\n  if [ \"$rc\" -eq 0 ]; then rc=1; fi\nfi\n"
        ));
    } else {
        script.push_str(&format!(
            "find {LOCAL_RESULTS} -name report.json -exec cat {{}} + || true\n"
        ));
    }
    script.push_str("exit \"$rc\"\n");
    script
}

fn validate_binary_url(url: &str) -> Result<()> {
    if !url.starts_with("https://") && !url.starts_with("http://") {
        return Err(Error::usage("--hf-binary-url must be an http(s) URL"));
    }
    validate_shell_safe(url, "--hf-binary-url")
}

/// Rejects values that could break out of the single-quoted or bare positions
/// they are interpolated into inside the job bootstrap script.
fn validate_shell_safe(value: &str, what: &str) -> Result<()> {
    if value.is_empty()
        || value.bytes().any(|byte| {
            byte.is_ascii_control() || matches!(byte, b' ' | b'\'' | b'"' | b'\\' | b'$' | b'`')
        })
    {
        return Err(Error::usage(format!(
            "{what} must be nonempty without spaces, quotes, backslashes, `$`, backticks, or control characters"
        )));
    }
    Ok(())
}

fn run_hf(args: &[OsString]) -> Result<()> {
    let status = Command::new("hf").args(args).status().map_err(|source| {
        Error::new(
            ErrorKind::ProcessSpawn,
            Some(ExecutionStage::Preflight),
            "failed to launch `hf`; install the Hugging Face CLI \
             (`pip install huggingface_hub`, `brew install hf`, or `uv tool install hf`)",
        )
        .with_source(source)
    })?;
    if !status.success() {
        return Err(Error::new(
            ErrorKind::ProcessExit,
            Some(ExecutionStage::Benchmark),
            format!("Hugging Face Jobs submission failed ({status})"),
        ));
    }
    Ok(())
}

fn token_present() -> bool {
    resolve_hf_token(&ProcessExecutor::default(), &CancellationToken::new())
        .ok()
        .flatten()
        .is_some()
}

fn validate_bucket(bucket: &str) -> Result<()> {
    if bucket.starts_with("hf://buckets/") {
        Ok(())
    } else {
        Err(Error::usage(
            "--results-bucket must be a Hugging Face bucket URI (hf://buckets/<namespace>/<name>[/<path>])",
        ))
    }
}

fn encode_config(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|source| {
        Error::new(
            ErrorKind::Io,
            Some(ExecutionStage::Preflight),
            "failed to read configuration file for Hugging Face Jobs submission",
        )
        .with_path(path)
        .with_source(source)
    })?;
    Ok(base64_encode(&bytes))
}

fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let triple = ((chunk[0] as u32) << 16)
            | ((*chunk.get(1).unwrap_or(&0) as u32) << 8)
            | (*chunk.get(2).unwrap_or(&0) as u32);
        encoded.push(ALPHABET[((triple >> 18) & 63) as usize] as char);
        encoded.push(ALPHABET[((triple >> 12) & 63) as usize] as char);
        encoded.push(if chunk.len() > 1 {
            ALPHABET[((triple >> 6) & 63) as usize] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            ALPHABET[(triple & 63) as usize] as char
        } else {
            '='
        });
    }
    encoded
}

fn writeln_checked(out: &mut impl Write, text: &str) -> Result<()> {
    writeln!(out, "{text}").map_err(|source| {
        Error::new(ErrorKind::Io, None, "failed to write command output").with_source(source)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn bootstrap_stages_locally_and_copies_to_bucket_even_on_failure() {
        let bucket = build_bootstrap(
            "https://x/oa",
            "Zm9v",
            "bench",
            Some("repo/custom:1"),
            true,
            true,
        );
        assert!(bucket.contains("urllib.request.urlretrieve"));
        assert!(bucket.contains("uv venv /tmp/optimum-advisor-correctness"));
        assert!(bucket.contains("lighteval==0.13.0"));
        // Only `lighteval` is exposed; `python3` must keep resolving to the
        // base image (the venv interpreter cannot import the engine).
        assert!(bucket
            .contains("ln -sf /tmp/optimum-advisor-correctness/bin/lighteval /tmp/optimum-advisor-bin/lighteval"));
        assert!(bucket.contains("export PATH=\"/tmp/optimum-advisor-bin:$PATH\""));
        assert!(!bucket.contains("PATH=\"/tmp/optimum-advisor-correctness/bin"));
        // The run itself never writes to the FUSE mount.
        assert!(bucket.contains("--results-dir /tmp/optimum-advisor-results"));
        assert!(!bucket.contains("--results-dir /results"));
        // Effective image is forwarded so the report identity matches the job.
        assert!(bucket.contains(" --image repo/custom:1"));
        // Exit code is captured, results copied once, real status preserved.
        assert!(bucket.contains("|| rc=$?"));
        assert!(bucket.contains("cp -r /tmp/optimum-advisor-results/. /results/"));
        assert!(bucket.contains("exit \"$rc\""));
        assert!(!bucket.contains("find "));

        let logs = build_bootstrap("https://x/oa", "Zm9v", "sweep", None, false, false);
        assert!(!logs.contains("uv venv"));
        assert!(!logs.contains("lighteval"));
        assert!(!logs.contains("--image"));
        assert!(logs.contains("sweep --in-container"));
        assert!(logs.contains("find /tmp/optimum-advisor-results -name report.json -exec cat"));
        assert!(!logs.contains("cp -r"));
        assert!(logs.contains("exit \"$rc\""));
    }

    #[test]
    fn binary_url_and_shell_interpolations_are_validated() {
        assert!(validate_binary_url("https://github.com/x/releases/v1/oa").is_ok());
        assert!(validate_binary_url("ftp://host/oa").is_err());
        assert!(validate_binary_url("https://host/oa with space").is_err());
        assert!(validate_binary_url("https://host/oa'; rm -rf /'").is_err());
        assert!(validate_shell_safe("vllm/vllm-openai@sha256:abc", "image reference").is_ok());
        assert!(validate_shell_safe("bad image", "image reference").is_err());
        assert!(validate_shell_safe("bad`image`", "image reference").is_err());
        assert!(validate_shell_safe("", "image reference").is_err());
    }

    #[test]
    fn bucket_uri_is_validated() {
        assert!(validate_bucket("hf://buckets/me/results").is_ok());
        assert!(validate_bucket("s3://me/results").is_err());
    }
}
