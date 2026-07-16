# Optimum Advisor

Optimum Advisor is a production-hardened runner for comparing vLLM and
SGLang serving configurations on NVIDIA GPU hosts — locally via Docker, or
remotely on Hugging Face Jobs. It resolves container images to immutable
identities, validates engine arguments against the selected image, runs
correctness and engine-native benchmarks, preserves failed trials, and writes
an atomic schema-v2 report.

It is **not** a production advisor or cluster scheduler. Candidate execution is
sequential. Selection ranks only observed candidates; it does not yet invent
configurations or enforce service-level objectives.

**Contents**

- [Getting started](#getting-started)
  — [Requirements](#requirements)
  · [Installation](#installation)
  · [Safe first run](#safe-first-run)
- [Configuration](#configuration)
  — [Schema-v2 TOML](#schema-v2-toml)
  · [Sweeps](#sweeps)
  · [Precedence](#precedence)
  · [GPU selection](#gpu-selection)
- [Running evaluations](#running-evaluations)
  — [Commands](#commands)
  · [Execution backends](#execution-backends)
  · [Running on Hugging Face Jobs](#running-on-hugging-face-jobs)
  · [Timeouts, cancellation, and cleanup](#timeouts-cancellation-and-cleanup)
  · [Reports and artifacts](#reports-and-artifacts)
  · [Correctness](#correctness)
  · [Secrets and leaderboard submission](#secrets-and-leaderboard-submission)
- [MCP server](#mcp-server)
- [Development](#development)
  — [Development gates](#development-gates)
  · [Real GPU-host acceptance](#real-gpu-host-acceptance)
  · [Releases](#releases)
- [Explicit limitations](#explicit-limitations)

## Getting started

### Requirements

- Docker with the NVIDIA container runtime (`docker run --gpus ...`) for the
  default execution backend. The `--in-container` backend instead expects the
  engine CLI (`vllm`, or `python3 -m sglang...`) on `PATH` inside the current
  container; see [Execution backends](#execution-backends).
- NVIDIA drivers and `nvidia-smi` on execution hosts.
- Rust **1.85.0** or newer to build from source (not needed for the prebuilt
  binary).
- The correctness environment when correctness is enabled:
  `./scripts/setup-correctness-env.sh`.
- Optional: `hf-mem`, `uvx hf-mem`, or a configured command for model-memory
  estimates. Missing optional estimation is recorded as a typed warning;
  `model_memory.required = true` makes it a preflight failure.
- Optional: a Hugging Face token for gated/private models or leaderboard
  submission. Public model execution does not require a token.
- Optional: the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
  and a logged-in account to submit runs with `--on hf-jobs`; see
  [Running on Hugging Face Jobs](#running-on-hugging-face-jobs).

### Installation

#### Prebuilt binary (Linux x86_64)

Every [release](https://github.com/dwarez/optimum-advisor/releases) attaches a
statically linked binary that runs on any x86_64 Linux distribution — no Rust
toolchain, no glibc requirement. Download it, verify the checksum, and install:

```bash
curl -fsSLO https://github.com/dwarez/optimum-advisor/releases/latest/download/optimum-advisor-x86_64-unknown-linux-musl
curl -fsSLO https://github.com/dwarez/optimum-advisor/releases/latest/download/optimum-advisor-x86_64-unknown-linux-musl.sha256
sha256sum -c optimum-advisor-x86_64-unknown-linux-musl.sha256
install -m 0755 optimum-advisor-x86_64-unknown-linux-musl ~/.local/bin/optimum-advisor
```

Pin a specific version by replacing `latest/download` with
`download/v<version>`. This is the same artifact the `--on hf-jobs` launcher
downloads inside jobs.

#### From source (any platform)

From this checkout:

```bash
./scripts/install.sh
```

Equivalent command:

```bash
cargo install --path . --locked --force
```

From Git:

```bash
cargo install --git https://github.com/dwarez/optimum-advisor.git --locked --force
```

Uninstall with `cargo uninstall optimum-advisor`.

### Safe first run

Dry-run planning performs local invariant checks and renders the exact commands.
It does not pull images, inspect hardware, start Docker containers, contact the
leaderboard, or require credentials:

```bash
optimum-advisor plan --config examples/bench.toml
optimum-advisor bench --config examples/bench.toml --dry-run
optimum-advisor sweep --config examples/sweep.toml --dry-run
```

Inspect the host and the selected engine image before execution:

```bash
optimum-advisor hardware
optimum-advisor params \
  --engine vllm \
  --image vllm/vllm-openai:latest \
  --cache-dir .optimum-advisor/params \
  --refresh
```

`params` resolves a tag to a repository digest or image ID before using it as a
cache identity. `params --offline` performs no Docker or network operation and
requires both an immutable `repo@sha256:...`/`sha256:...` reference and an exact
cached schema. `--offline` and `--refresh` are mutually exclusive.

Execute one candidate or a bounded sweep:

```bash
optimum-advisor bench --config examples/bench.toml
optimum-advisor bench --config examples/sglang-bench.toml
optimum-advisor sweep --config examples/sweep.toml
```

## Configuration

### Schema-v2 TOML

Configuration files are strict TOML. `schema_version = 2` is required; unknown
and duplicate keys fail. Operational output directories and dry-run mode remain
CLI options rather than persisted configuration.

```toml
schema_version = 2
engine = "vllm"
model = "Qwen/Qwen3-4B-Instruct-2507"
metric = "tps"

[runtime]
gpus = 1
max_model_len = 8192
startup_timeout_secs = 600
benchmark_timeout_secs = 900
max_process_output_bytes = 16777216

[benchmark]
dataset_name = "random"
num_prompts = 4
request_rate = "1"
max_concurrency = 1
random_input_len = 1024
random_output_len = 128

[candidate]
tensor_parallelism = 1
memory_fraction = 0.90
prefill_token_budget = 8192
max_running_requests = 8

[correctness]
enabled = true
threshold = 0.80
timeout_secs = 900

[model_memory]
enabled = true
required = false
timeout_secs = 300

[leaderboard]
submit = false

[serve]
dtype = "auto"
```

`[candidate]` contains portable dimensions translated into the selected
engine's current CLI. `[serve]` contains explicit engine-specific arguments.
Names are canonicalized, deduplicated, and validated against the schema obtained
from the immutable image. Repeatable CLI equivalents are `--serve-arg
NAME=VALUE` and `--serve-flag NAME`.

### Sweeps

A sweep adds bounded arrays. Expansion is deterministic, overflow-checked, and
fails before execution when its Cartesian product exceeds `max_trials`:

```toml
[sweep]
max_trials = 2
tensor_parallelism = [1, 2]

[sweep.serve]
# Engine-specific dimensions, when needed:
# kv-cache-dtype = ["auto", "fp8"]
```

See `examples/bench.toml`, `examples/sglang-bench.toml`, and
`examples/sweep.toml` for runnable configurations.

### Precedence

The merge order is:

1. schema-v2 TOML;
2. explicit CLI overrides.

Arbitrary `OPTIMUM_ADVISOR_*` variables do not change model, engine, runtime, or
candidate settings. Environment access is limited to external integration and
secret discovery: `CUDA_VISIBLE_DEVICES`, `HF_TOKEN`,
`HUGGING_FACE_HUB_TOKEN`, Hugging Face token locations, `HF_TOKEN_PATH`,
`HF_HOME`, `OPTIMUM_ADVISOR_HF_MEM`, and
`OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT_KEY`.

### GPU selection

`runtime.gpus` selects a count. `runtime.gpu_devices` or repeated
`--gpu-device` values select explicit GPU indexes/UUIDs. Explicit devices must
be nonempty and unique, and their count must match `gpus`. Tensor parallelism
must be nonzero, cannot exceed the selected count, and must divide it evenly.

Under `--in-container`, explicit `gpu_devices` are exported to the server as
`CUDA_VISIBLE_DEVICES`; count-based `gpus` selection uses whatever GPUs the
surrounding container exposes.

## Running evaluations

### Commands

```text
plan      Render one validated, non-executing server/benchmark preview
params    Resolve an image and inspect/cache its engine parameter schema
hardware  Inspect selected local NVIDIA GPUs
serve     Run one validated owned serving container until exit/interruption
bench     Evaluate one candidate
sweep     Evaluate the bounded sweep from a v2 TOML file
cleanup   List or remove only Optimum Advisor-owned containers
mcp       Serve newline-delimited MCP JSON-RPC over stdin/stdout
```

Run `optimum-advisor <command> --help` for the exact options, except `mcp`,
which intentionally accepts no arguments and reserves stdout for protocol
frames.

### Execution backends

`plan`, `serve`, `bench`, and `sweep` launch the engine server and benchmark one
of two ways, selected with `--in-container`:

- **Docker (default):** every engine invocation is wrapped in `docker run --gpus
  ... <image> ...` on the local host. The image is resolved to an immutable
  identity, the server port is published, and the run owns and cleans up its
  containers.
- **In-container (`--in-container`):** the engine binaries (`vllm` /
  `python3 -m sglang...`) run directly as child processes bound on loopback,
  with no Docker daemon, image resolution, or container cleanup. Use it inside a
  container that already provides the engine image — for example a Hugging Face
  Job whose image is `vllm/vllm-openai` — where nested Docker is unavailable.

Preview the exact commands for either backend without executing anything:

```bash
optimum-advisor plan --config examples/bench.toml
optimum-advisor plan --config examples/bench.toml --in-container
```

Both backends share the same validation, correctness suite, ranking, schema-v2
report, and cancellation paths; only server and benchmark launch and image
handling differ. Under `--in-container` the Hugging Face token, when present, is
passed through the child environment rather than a `docker run -e` flag, and the
engine parameter schema is inspected by running `python3` directly instead of
through Docker.

### Running on Hugging Face Jobs

`bench` and `sweep` accept `--on hf-jobs` to run on Hugging Face Jobs instead of
the local host. The evaluation runs inside a single GPU container (the engine
image) through the in-container backend, so no local Docker or GPU is required —
only the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), a
logged-in account (`hf auth login`) with a positive credit balance, and a
published release of the prebuilt Linux binary.

```bash
optimum-advisor bench --config examples/bench.toml \
  --on hf-jobs --hf-flavor a10g-large \
  --results-bucket hf://buckets/<namespace>/<name>
```

The launcher submits an `hf jobs run` whose container downloads the prebuilt
binary, materializes the config, and runs the evaluation with `--in-container`.
The job container image is the config's `image` (defaulting to the engine's own
image, e.g. `vllm/vllm-openai:latest`); for `bench`, `--image` overrides it —
useful for custom builds that bundle the engine — and is forwarded to the in-job
run so the report records the image the job actually ran on. Add `--dry-run` to
print the exact `hf jobs run` command without submitting.

Options are valid only with `--on hf-jobs`:

| Flag | Meaning |
| --- | --- |
| `--hf-flavor` | Hardware flavor (required), e.g. `a10g-large`, `a100-large`. |
| `--hf-timeout` | Maximum job duration (`90m`, `2h`); the Jobs default is 30 minutes. |
| `--hf-namespace` | Organization namespace to run the job under. |
| `--results-bucket` | Persist results to `hf://buckets/<namespace>/<name>[/<path>]`. |
| `--hf-detach` | Submit in the background and print only the job ID. |
| `--hf-binary-url` | Override the prebuilt binary URL downloaded inside the job (defaults to the GitHub release matching this binary's version). |

Constraints and behavior:

- Requires `--config`; the only supported CLI override is `--image` (see above),
  so put every other setting in the file.
- A local Hugging Face token, when present, is forwarded with `hf jobs run
  --secrets HF_TOKEN` (encrypted server-side) for gated models and bucket access.
- Results are written to a local directory inside the job and transferred once
  at the end — also for failed runs, preserving failed-trial evidence. With
  `--results-bucket` the bucket is mounted read-write and receives a bulk copy
  of `report.json`, `best.toml`, and per-trial artifacts; without it, results
  live only in the ephemeral container and the final `report.json` is printed to
  the job logs.
- When `[correctness] enabled = true`, the pinned correctness tools are installed
  into an isolated `uv` environment inside the job, leaving the engine's own
  dependencies untouched.

### Timeouts, cancellation, and cleanup

Startup, benchmark, correctness, model-memory, HTTP, and output limits are
bounded. SIGINT/SIGTERM cancellation propagates through active subprocesses.
On Unix, child process groups receive TERM, a bounded grace period, and then
KILL. Docker cleanup targets only containers carrying all Optimum Advisor
ownership labels, including the run ID and role.

Inspect without removing:

```bash
optimum-advisor cleanup --dry-run
optimum-advisor cleanup --run-id <run-id> --dry-run
```

Remove matching owned containers:

```bash
optimum-advisor cleanup
optimum-advisor cleanup --run-id <run-id>
```

The cleanup command never targets an unlabeled or partially labeled container.

### Reports and artifacts

Every executed `bench` or `sweep` creates a private directory under
`.optimum-advisor/results` unless `--results-dir` overrides it. The directory is
created before external preflight and immediately receives `report.json`.

`report.json` is schema version 2 and is atomically checkpointed after
preflight, after each trial, after winning-config selection, and after
leaderboard submission. Terminal states are:

- `completed`;
- `completed_with_failures`;
- `failed`;
- `interrupted`.

Expected candidate failures do not abort a sweep. Each failed trial records a
typed stage/kind, timeout/interruption flags, and bounded UTF-8-safe stdout and
stderr tails. Ranking considers correctness first, then the selected metric's
direction, then stable trial order. Missing or non-finite selected metrics
cannot win. If every candidate fails, the process fails but the terminal report
and all trial evidence remain available.

Important paths:

- `report.json`: durable source of truth;
- `best.toml`: runnable winning schema-v2 configuration, written only after a
  successful winner exists;
- `trials/NNNN/`: per-trial correctness, benchmark, and bounded diagnostic
  artifacts recorded in the report manifest.

All result directories are mode `0700` and sensitive files are mode `0600` on
Unix.

### Correctness

When enabled, correctness runs against the same owned server before its
benchmark. The suite records task scores and, when configured, probes OpenAI
chat-completion tool-call and reasoning-parser behavior. The finite threshold
must be in `[0, 1]`; `0` validates execution and complete metric collection
without imposing a model-quality floor. A missed positive threshold fails the
trial and prevents it from winning.

Install the pinned external tools in a repository-local environment:

```bash
./scripts/setup-correctness-env.sh
source .venv/bin/activate
```

### Secrets and leaderboard submission

Tokens and submission keys use redacted wrappers, are attached only to child
environments or HTTPS authorization, and are not included in safe command
rendering, reports, artifacts, URLs, or persisted configuration. Captured
process streams are sanitized before storage.

Leaderboard publishing is opt-in through `[leaderboard] submit = true` or
`--leaderboard-submit`. The default endpoint is
`https://hf-dwarez-optimum-advisor-leaderboard.hf.space`; set
`leaderboard.url` or `--leaderboard-url` to override it. The client requires
HTTPS, uses rustls plus platform certificate verification, bounds response
bodies, and applies connect/read/overall deadlines.

Authenticate with `hf auth login` or `HF_TOKEN`. Contributor identity is
inferred from that credential. An optional administrative submit key is read
from `OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT_KEY`. Neither credential is persisted.

## MCP server

Configure an MCP client to start the binary directly:

```json
{
  "mcpServers": {
    "optimum-advisor": {
      "command": "/absolute/path/to/optimum-advisor",
      "args": ["mcp"]
    }
  }
}
```

The transport is strict newline-delimited JSON-RPC 2.0. Clients initialize,
receive protocol version `2025-11-25`, send `notifications/initialized`, and
then call tools. Requests are processed sequentially; `ping` remains available
during initialization. `notifications/cancelled` cancels a matching queued or
active tool request. Input lines are capped at 1 MiB, responses at 256 KiB, and
human-readable tool text at 64 KiB. Stdout contains protocol frames only.

Tools:

- `inspect_hardware`;
- `inspect_engine`;
- `validate_config`;
- `estimate_memory`;
- `check_correctness`;
- `run_benchmark`;
- `evaluate_candidate`;
- `run_sweep`;
- `rank_candidates`.

Input and output JSON Schemas are generated from the same strict Rust DTOs used
for decoding. Unknown fields fail. Execution tools use the same cancellation,
immutable image, validation, persistence, and cleanup path as the CLI. They
return bounded summaries plus durable report paths. Tool-domain failures remain
successful JSON-RPC responses with `isError: true` and a typed payload;
malformed envelopes and unknown methods use JSON-RPC errors.

## Development

### Development gates

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --release --all-features --locked
cargo +1.85.0 test --all-targets --all-features --locked
sh -n scripts/install.sh scripts/setup-correctness-env.sh
```

CI also runs RustSec `cargo audit` and ShellCheck.

### Real GPU-host acceptance

The ignored acceptance test performs real hardware selection, immutable image
and parameter inspection, one tiny correctness-plus-benchmark candidate, a
two-candidate sweep, SIGINT cleanup, report-v2 checks, winning-config checks,
and owned-container leak comparison. It does not fake GPU success.

Prepare the correctness environment, choose a model appropriate for the host,
and opt in explicitly:

```bash
./scripts/setup-correctness-env.sh
source .venv/bin/activate

OPTIMUM_ADVISOR_GPU_ACCEPTANCE=1 \
OPTIMUM_ADVISOR_GPU_ACCEPTANCE_MODEL=Qwen/Qwen3-0.6B \
cargo test --test gpu_acceptance -- --ignored --nocapture
```

Optional overrides:

```bash
OPTIMUM_ADVISOR_GPU_ACCEPTANCE_ENGINE=sglang
OPTIMUM_ADVISOR_GPU_ACCEPTANCE_IMAGE=repo/image@sha256:<digest>
```

Run this only on a disposable or controlled GPU host: it pulls/starts real
containers and performs real inference.

### Releases

Releases are fully automated with [release-plz](https://release-plz.dev) in
git-only mode (the crate is not published to crates.io):

1. Every push to `main` opens or updates a release PR that bumps the version
   and `CHANGELOG.md` from the commits since the last tag.
2. Merging that PR makes CI create the `v<version>` git tag and the GitHub
   release, then dispatches the workflow that builds and attaches the prebuilt
   static Linux binary — the artifact the `--on hf-jobs` launcher downloads.

Conventional commits drive the bump: `feat:` bumps minor (also on `0.x`),
`fix:` and other messages bump patch, and `!`/`BREAKING CHANGE` marks a
breaking release. This repository's `add:`/`change:`/`refactor:` prefixes are
grouped in the changelog and produce patch bumps. Manually pushed `v*` tags
still trigger the binary build directly.

One-time repository setting: enable "Allow GitHub Actions to create and
approve pull requests" (Settings → Actions → General) so the release PR can be
opened with the default `GITHUB_TOKEN`. Release PRs opened with that token do
not trigger PR CI; the gates run when the merge lands on `main`.

## Explicit limitations

- execution is sequential, one candidate at a time, not distributed;
- the default Docker backend requires Docker and NVIDIA GPU support for real
  serving runs; the `--in-container` backend drops the Docker dependency but
  still needs the engine CLI and visible GPUs inside the container;
- model-memory estimation depends on an optional external `hf-mem` command;
- correctness depends on the separately installed pinned Python tools;
- no advisor heuristics currently generate candidates from hardware or memory
  budgets;
- no latency-ceiling or minimum-throughput constraint solver exists;
- GPU-host acceptance is opt-in and is not simulated in ordinary CI.
