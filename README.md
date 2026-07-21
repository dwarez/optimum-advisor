# Optimum Advisor

How you serve a model — engine, parallelism, memory budget, batching, cache
settings — can move throughput and latency by large factors, and the only way
to know what works on *your* model and *your* hardware is to measure it.
Optimum Advisor makes that measurement systematic: declare candidate serving
configurations, and it serves each one, checks output quality, runs the
engine's own benchmark, ranks the candidates, and leaves you a durable report
plus a ready-to-run winning configuration.

Today it drives **vLLM** and **SGLang**, running either on your own NVIDIA GPU
host via Docker or on [Hugging Face Jobs](#run-on-hugging-face-jobs) with no
GPU of your own. It is not a cluster scheduler or an automatic tuner: local
Docker sweeps can pack candidates across a selected multi-GPU pool, while
HF Jobs/in-container sweeps remain sequential; ranking covers only the
candidates you declare.

**Jump to:**
[Quickstart](#quickstart) ·
[Configuration](#configuration) ·
[Running benchmarks](#running-benchmarks) ·
[Hugging Face Jobs](#run-on-hugging-face-jobs) ·
[Results](#results) ·
[Correctness](#correctness-checks) ·
[MCP server](#mcp-server) ·
[Limitations](#limitations) ·
[Contributing](CONTRIBUTING.md)

## Quickstart

### 1. Install

Prebuilt binaries (Linux x86_64, macOS arm64/x86_64):

```bash
target=$(case "$(uname -s)-$(uname -m)" in
  (Linux-x86_64)  echo x86_64-unknown-linux-musl ;;
  (Darwin-arm64)  echo aarch64-apple-darwin ;;
  (Darwin-x86_64) echo x86_64-apple-darwin ;;
esac)
curl -fsSLO "https://github.com/dwarez/optimum-advisor/releases/latest/download/optimum-advisor-$target"
curl -fsSLO "https://github.com/dwarez/optimum-advisor/releases/latest/download/optimum-advisor-$target.sha256"
shasum -a 256 -c "optimum-advisor-$target.sha256"   # on Linux: sha256sum -c
install -m 0755 "optimum-advisor-$target" ~/.local/bin/optimum-advisor
```

Or build from source (Rust 1.85+): `cargo install --git
https://github.com/dwarez/optimum-advisor.git --locked`. Pin a version by
replacing `latest/download` with `download/v<version>`.

### 2. Write a config

```toml
# bench.toml
schema_version = 2
engine = "vllm"                          # or "sglang"
model = "Qwen/Qwen3-4B-Instruct-2507"
metric = "tps"                           # what to optimize; see Configuration

[runtime]
gpus = 1

[correctness]
enabled = false                          # see "Correctness checks" to enable
```

### 3. Run it

**On a GPU host** (needs Docker with the NVIDIA container runtime and
`nvidia-smi`):

```bash
optimum-advisor bench --config bench.toml
```

**On Hugging Face Jobs** (no local GPU or Docker; needs the
[`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), `hf auth
login`, and a positive credit balance):

```bash
optimum-advisor bench --config bench.toml \
  --on hf-jobs --hf-flavor a10g-large \
  --results-bucket hf://buckets/<namespace>/<bucket>
```

Either way you get a run directory containing `report.json` (every metric,
every trial, including failures) and — once a candidate wins — `best.toml`, a
runnable config of the winner. Locally it lands under
`.optimum-advisor/results/`; on HF Jobs it is copied into your bucket.

Try a sweep next: declare arrays under `[sweep]` and run `optimum-advisor
sweep` to compare candidates — see [Sweeps](#sweeps).

## Preview before you run

These commands validate and print exactly what would execute, without pulling
images, starting containers, or requiring credentials:

```bash
optimum-advisor plan  --config bench.toml         # render server/benchmark commands
optimum-advisor bench --config bench.toml --dry-run
optimum-advisor sweep --config sweep.toml --dry-run
optimum-advisor hardware                          # inspect local GPUs
optimum-advisor params --engine vllm --image vllm/vllm-openai:latest --refresh
```

`params` resolves the image to an immutable digest and caches the engine's
accepted CLI arguments; `--offline` reuses only the cache (mutually exclusive
with `--refresh`). With `--on hf-jobs`, `--dry-run` prints the exact `hf jobs
run` submission.

## Configuration

Configs are strict schema-v2 TOML: `schema_version = 2` is required, and
unknown or duplicate keys are rejected. The quickstart config above is
complete; everything else has defaults. Full reference:

<details>
<summary>Full annotated example</summary>

```toml
schema_version = 2
engine = "vllm"                # vllm | sglang
model = "Qwen/Qwen3-4B-Instruct-2507"
metric = "tps"                 # optional objective: throughput/latency and p90/p95/p99 variants
                               # omit: tpot for model IDs up to 3B, tps otherwise
image = "vllm/vllm-openai:latest"  # optional; defaults to the engine's image

[runtime]
gpus = 1                       # GPU count; or explicit devices:
# gpu_devices = ["0", "1"]     # indexes/UUIDs; count must equal `gpus`
max_model_len = 8192
startup_timeout_secs = 600
benchmark_timeout_secs = 900
max_process_output_bytes = 16777216

[benchmark]                    # engine-native benchmark shape
dataset_name = "random"
num_prompts = 100
request_rate = "1"
max_concurrency = 1
random_input_len = 1024
random_output_len = 128

[candidate]                    # portable dimensions, translated per engine
tensor_parallelism = 1
memory_fraction = 0.90
prefill_token_budget = 8192
max_running_requests = 8

[correctness]                  # quality gate before the benchmark
enabled = true                 # default: true
threshold = 0.0                # default: 0.2; see "Correctness checks"
timeout_secs = 900

[model_memory]                 # optional hf-mem estimate
enabled = true
required = false               # true makes a missing estimate a failure
timeout_secs = 300

[leaderboard]
submit = false                 # opt-in public leaderboard submission

[serve]                        # explicit engine-specific server arguments
dtype = "auto"
```

</details>

Key rules:

- `[candidate]` holds portable dimensions translated into the selected engine's
  CLI; `[serve]` holds raw engine arguments. Names are canonicalized,
  deduplicated, and validated against the schema of the actual image. CLI
  equivalents: `--serve-arg NAME=VALUE`, `--serve-flag NAME`.
- Tensor parallelism must be nonzero and cannot exceed the GPU count. In a
  local Docker sweep, each trial leases exactly that many GPUs from the
  selected pool. Explicit `gpu_devices` must be unique and match `gpus`.
- Precedence: config file first, explicit CLI overrides win. Environment
  variables never change benchmark settings; only integration/secret lookups
  are read (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH`, `HF_HOME`,
  `CUDA_VISIBLE_DEVICES`, `OPTIMUM_ADVISOR_HF_MEM`,
  `OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT_KEY`).

### Sweeps

`[sweep]` declares bounded arrays; the Cartesian product expands
deterministically and fails fast if it exceeds `max_trials`:

```toml
[sweep]
max_trials = 4
max_parallel_trials = 2        # optional cap; omit for automatic GPU packing
tensor_parallelism = [1, 2]

[sweep.serve]                  # engine-specific dimensions
kv-cache-dtype = ["auto", "fp8"]
```

Local Docker sweeps schedule trials concurrently when their tensor-parallel
GPU demands fit disjoint subsets of the selected `runtime.gpus` /
`runtime.gpu_devices` pool. Pending trials stay in declaration order, with
greedy backfill when an earlier trial is temporarily too large for the free
subset. `--max-parallel-trials 1` forces sequential execution; higher values
cap concurrency without permitting GPU oversubscription. Each active trial
gets a distinct host port. HF Jobs and `--in-container` sweeps stay sequential
because one job/container owns one engine process namespace.

Parallelism is primarily a search-throughput optimization, not measurement
isolation. Active trials receive disjoint GPUs, but their engine processes
still share host resources such as CPU, memory, storage and cache I/O,
networking, PCIe bandwidth, and power or thermal headroom. Contention can
shift benchmark metrics and usually prevents perfectly linear wall-time
speedups. When candidates rank closely, rerun the finalists with
`max_parallel_trials = 1` before treating the ranking as conclusive.

Runnable examples live in [`examples/`](examples/).

## Running benchmarks

| Command | What it does |
| --- | --- |
| `bench` | Evaluate one candidate |
| `sweep` | Evaluate a bounded sweep and rank the candidates |
| `plan` | Print validated commands without executing |
| `serve` | Run one validated serving container until interrupted |
| `params` | Inspect/cache an engine image's accepted arguments |
| `hardware` | Inspect local NVIDIA GPUs |
| `cleanup` | List/remove only Optimum Advisor-owned containers |
| `mcp` | Serve MCP JSON-RPC over stdio |

`optimum-advisor <command> --help` lists every option.

### Where it runs

- **Local Docker (default).** Every engine invocation is wrapped in
  `docker run --gpus ...`; images are resolved to immutable digests, and the
  run owns and cleans up its containers.
  Multi-GPU sweeps lease disjoint GPU subsets and distinct host ports per
  active trial; use `--max-parallel-trials` to cap concurrency.
- **Inside a container (`--in-container`).** The engine binaries run as direct
  child processes — no Docker daemon needed. Use it in an environment that
  already provides the engine image (this is how jobs execute on HF Jobs).
  Explicit `gpu_devices` are applied via `CUDA_VISIBLE_DEVICES`.
- **Hugging Face Jobs (`--on hf-jobs`).** See below.

Both backends share the same validation, correctness, ranking, and report
paths; compare them with `plan --config … [--in-container]`.

### Stopping and cleaning up

All stages are bounded by timeouts. Ctrl-C (SIGINT/SIGTERM) cancels cleanly:
child process groups get TERM, a grace period, then KILL, and the report is
finalized as `interrupted`. Docker cleanup only ever touches containers
carrying the full set of Optimum Advisor ownership labels:

```bash
optimum-advisor cleanup --dry-run     # inspect
optimum-advisor cleanup               # remove owned containers
```

## Run on Hugging Face Jobs

`bench` and `sweep` accept `--on hf-jobs` to run the whole evaluation in a
single GPU job — no local Docker or GPU. The launcher submits an `hf jobs run`
whose container downloads the Optimum Advisor binary, materializes your config,
and executes in-container.

```bash
optimum-advisor bench --config bench.toml \
  --on hf-jobs --hf-flavor a10g-large --hf-timeout 90m \
  --results-bucket hf://buckets/<namespace>/<bucket>
```

| Flag | Meaning |
| --- | --- |
| `--hf-flavor` | Hardware flavor (required), e.g. `a10g-large`, `a100-large` |
| `--hf-timeout` | Max job duration (`90m`, `2h`); the Jobs default is 30 minutes |
| `--hf-namespace` | Organization to run the job under |
| `--results-bucket` | Copy results to `hf://buckets/<namespace>/<name>[/<path>]` |
| `--hf-detach` | Submit in the background, print only the job ID |
| `--hf-binary-url` | Override the in-job binary download URL |

Behavior and constraints:

- **Job image.** Defaults to the config's `image` (or the engine's own image).
  For `bench`, `--image` overrides it — the only CLI override accepted with
  `--on hf-jobs`; put everything else in the config file. The override is
  forwarded in-job so the report records the image that actually ran.
- **Results.** Written locally inside the job and transferred once at the end —
  also for failed runs, so failure evidence survives. With `--results-bucket`
  the bucket receives `report.json`, `best.toml`, and per-trial artifacts;
  without it, the final `report.json` is printed to the job logs.
- **Auth.** A local Hugging Face token is forwarded as an encrypted job secret
  for gated models and bucket access.
- **Correctness.** When enabled, the pinned tools are installed into an
  isolated environment inside the job; the engine's dependencies are never
  touched.
- **Versioning.** The job downloads the release binary **matching your CLI's
  version** (printed at submit time as `in-job binary: …`), so submitter and
  job always agree on flags and config schema. After upgrading, make sure your
  installed CLI matches the newest release before resubmitting.

## Results

Every executed run creates a private directory (default under
`.optimum-advisor/results/`, override with `--results-dir`):

- `report.json` — the durable source of truth, atomically checkpointed after
  preflight, every trial, winner selection, and leaderboard submission.
  Terminal states: `completed`, `completed_with_failures`, `failed`,
  `interrupted`.
- `best.toml` — a runnable schema-v2 config of the winner (only written when a
  candidate succeeds).
- `trials/NNNN/` — per-trial correctness, benchmark, and bounded diagnostic
  artifacts.
  Local parallel sweeps record each trial's leased GPU IDs, host port, and
  execution lane; any preflight port fallback is recorded under top-level
  `warnings`. These physical leases are diagnostic only and are not written
  into `best.toml`.

A failed candidate does not abort a sweep: each failure records a typed
stage/kind and bounded stdout/stderr tails. Ranking considers correctness
first, then the selected metric, then stable trial order; a candidate with a
missing or non-finite objective cannot win. Metrics the benchmark did emit
remain in `report.json`, and the error lists the finite metric names available
from that engine image. Directories are mode `0700`, sensitive files `0600`.

## Correctness checks

When enabled (default), a small lighteval suite (gsm8k, ifeval, triviaqa, drop)
plus optional tool-call/reasoning probes runs against the served model before
its benchmark. The `threshold` (in `[0, 1]`) applies **per task**; a missed
positive threshold fails the trial.

Calibration matters: `triviaqa` and `drop` are scored by strict exact match and
score near zero for verbose instruct models, so any positive threshold fails
healthy configurations of that model class. Use `threshold = 0.0` to validate
execution and metric collection without a quality floor, and set a positive
threshold only when every suite task can realistically meet it.

Running locally requires the pinned tools once:

```bash
./scripts/setup-correctness-env.sh && source .venv/bin/activate
```

(On HF Jobs this happens automatically inside the job.)

## Leaderboard and secrets

Submission is opt-in (`[leaderboard] submit = true` or `--leaderboard-submit`)
and posts to `https://hf-dwarez-optimum-advisor-leaderboard.hf.space` (override
with `leaderboard.url` / `--leaderboard-url`; HTTPS enforced). Contributor
identity comes from your Hugging Face credential (`hf auth login` or
`HF_TOKEN`).

Tokens and submit keys are held in redacted wrappers, passed only to child
environments or HTTPS authorization, and never appear in rendered commands,
reports, artifacts, or persisted configuration. Captured process output is
sanitized before storage.

## MCP server

Point an MCP client at the binary:

```json
{
  "mcpServers": {
    "optimum-advisor": { "command": "/path/to/optimum-advisor", "args": ["mcp"] }
  }
}
```

Transport is strict newline-delimited JSON-RPC 2.0 (protocol `2025-11-25`,
negotiating down to `2025-06-18` and `2025-03-26` clients per the MCP spec);
stdout carries protocol frames only. Tools: `inspect_hardware`,
`inspect_engine`, `validate_config`, `estimate_memory`, `check_correctness`,
`run_benchmark`, `evaluate_candidate`, `run_sweep`, `rank_candidates`,
`get_report`, `list_runs`. Schemas are generated from the same strict types the
CLI uses, and JSON configs use the same shape as schema-v2 TOML files.
Execution tools share the CLI's validation, cancellation, persistence, and
cleanup paths and return per-trial summaries (metrics, correctness, compact
failures) plus durable report paths; `get_report` and `list_runs` read prior
results back without filesystem access. Requests run sequentially;
`notifications/cancelled` cancels a queued or active request; input lines are
capped at 1 MiB and responses at 256 KiB. `scripts/mcp-probe.py` is a minimal
stdio client for inspecting the server by hand.

## Limitations

- Execution is sequential — one candidate at a time, not distributed.
- The default backend needs Docker + NVIDIA GPUs; `--in-container` drops Docker
  but still needs the engine CLI and visible GPUs; `--on hf-jobs` needs a
  Hugging Face account with credits.
- Model-memory estimation depends on the optional external `hf-mem` command.
- Correctness depends on separately installed pinned Python tools.
- No heuristics generate candidates from hardware or memory budgets, and no
  latency/throughput constraint solver exists — you declare the candidates.

## Contributing

Development gates, the GPU acceptance test, and the release process are
documented in [CONTRIBUTING.md](CONTRIBUTING.md).
