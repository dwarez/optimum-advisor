# Production Hardening Design

**Date:** 2026-07-13

**Status:** Approved for implementation

## Goal

Turn Optimum Advisor from an experimental benchmark prototype into a reliable, production-grade local CLI and stdio MCP server for running sequential LLM-serving evaluations on a Docker/NVIDIA host.

“Production-grade” here means deterministic configuration, faithful execution of requested settings, bounded and cancellable subprocesses, scoped container cleanup, durable sweep results, strict machine interfaces, secret-safe network calls, atomic artifacts, and automated verification. It does not mean converting the project into a daemon, distributed scheduler, or scientifically validated optimizer.

## Decisions

- Use a clean cutover. In-repository callers migrate together; there are no v1 compatibility parsers, deprecated aliases, re-exports, or report shims.
- Keep execution synchronous and single-process. Do not add Tokio or background job scheduling.
- Preserve the CLI and stdio MCP product surfaces, but replace their current input contracts with strict v2 contracts.
- Delete the prototype `advise` command, log substring classifier, and engine `next_candidate` heuristics. The project must not present unvalidated heuristics as production advice.
- Continue a sweep after candidate-scoped failures. Do not retry candidates automatically.
- Do not commit or push any implementation, specification, or plan changes.
- Preserve the pre-existing untracked `Todos.md` unchanged.

## Current Risks Driving the Refactor

1. Engine defaults overwrite explicit user candidate values in `src/engines/vllm/mod.rs` and `src/engines/sglang/mod.rs`.
2. `src/app.rs` aborts a sweep on the first failed candidate and writes its report only after all trials complete.
3. Subprocesses in `src/runtime/runner.rs`, correctness checks, image inspection, and leaderboard submission have incomplete or absent timeouts and cancellation.
4. The correctness suite declares `timeout_secs` but no executor enforces it.
5. Leaderboard bearer tokens are passed to `curl` in process arguments, where local process inspection can expose them.
6. Missing correctness currently ranks equal to passed correctness.
7. Results, correctness artifacts, leaderboard payloads, and model-memory output use hand-written JSON parsing or serialization despite the existing `serde_json` dependency.
8. Model-memory estimation can invoke `uvx hf-mem`, implicitly downloading and executing unpinned code.
9. Docker publishes model ports on every interface and exposes all GPUs regardless of the requested count.
10. Parameter caches are keyed by mutable image names, can collide after filename sanitization, and are written non-atomically.
11. CLI and MCP decoding duplicate defaults and validation while silently clamping or ignoring invalid values.
12. Reports embed unbounded logs and correctness JSON, and subprocess output is collected in memory without a bound.
13. Redirected and MCP logs can contain ANSI escapes.
14. Strict Clippy currently fails on five warnings; the repository has no CI.

## Architecture

The normalized execution path is:

```text
CLI / TOML / MCP DTO
        |
        v
source-specific decoding
        |
        v
normalization + invariant validation
        |
        v
ExecutableConfig
        |
        v
engine RunPlan
        |
        v
managed process/container executor
        |
        v
TrialOutcome (success or typed failure)
        |
        v
atomic report checkpoint
        |
        v
ranking and reproducible winning config
```

All input surfaces use the same domain validator. Engine adapters consume only validated `ExecutableConfig`; they do not mutate user input or apply hidden fallback behavior.

`plan` and `--dry-run` operate on a validated preview that retains the requested image reference and performs no pull or container execution. They clearly label runtime parameter validation as cached or pending. The execution path resolves that preview into `ExecutableConfig` with an immutable image identity before building an executable `RunPlan`.

## Configuration Contract

### Strict TOML

Configuration files use `toml` plus `serde` with unknown-field rejection. Parsing is transactional: an invalid file produces no partially mutated setup.

The v2 shape is:

```toml
schema_version = 2
engine = "vllm"
image = "vllm/vllm-openai:latest"
model = "Qwen/Qwen3-4B-Instruct-2507"
metric = "tps"

[runtime]
gpus = 1
# Optional explicit Docker device identifiers. When present, its length must
# equal `gpus`. Values may be NVIDIA indices or UUIDs.
gpu_devices = ["0"]
pull_policy = "missing" # "missing", "always", or "never"
allow_local_image = false
bind_host = "127.0.0.1"
port = 8000
startup_timeout_secs = 300
benchmark_timeout_secs = 1800
max_process_output_bytes = 16777216

[benchmark]
dataset_name = "random"
num_prompts = 100
request_rate = "1"
max_concurrency = 1
random_input_len = 1024
random_output_len = 128

[candidate]
tensor_parallelism = 1
memory_fraction = 0.90
prefill_token_budget = 8192
max_running_requests = 256

[correctness]
enabled = true
threshold = 0.20
timeout_secs = 600

[model_memory]
enabled = true
required = false
command = "hf-mem"
timeout_secs = 300

[leaderboard]
submit = false
url = "https://hf-dwarez-optimum-advisor-leaderboard.hf.space"

[serve]
kv-cache-dtype = "fp8"
disable-log-stats = true

[sweep]
max_trials = 256
tensor_parallelism = [1, 2]
memory_fraction = [0.80, 0.90]
prefill_token_budget = [4096, 8192]

[sweep.serve]
kv-cache-dtype = ["auto", "fp8"]
```

Top-level and fixed-section fields, including `[sweep]`, reject unknown keys. `[serve]` and `[sweep.serve]` are intentionally dynamic, but every name and flag/value shape is validated against the resolved image’s runtime parameter schema before execution.

Secrets are not accepted in TOML, ordinary CLI values, or generated winning configs. Hugging Face tokens may come from environment variables, user-private Hugging Face token files, or the bounded secret-output mode of a supported HF CLI. The leaderboard submit key comes only from `OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT_KEY`.

### Defaults and overrides

Engine and metric defaults produce a complete base candidate. Optional user overrides are represented separately and applied exactly once after default derivation. Validation follows override application.

No code silently clamps an invalid value. Validation rejects:

- empty model or image references;
- zero GPU, prompt, token, concurrency, timeout, or output-limit values;
- non-finite or out-of-range memory fractions and correctness thresholds;
- malformed request rates other than a positive finite number or the literal `inf`;
- tensor parallelism that is zero, exceeds the requested GPU count, or does not divide that count evenly;
- an explicit GPU-device list whose length differs from `gpus`;
- duplicate engine arguments with conflicting values;
- sweep-product overflow or more than `max_trials` candidates;
- dynamic parameters absent from the resolved image schema;
- using a value for a runtime-declared flag or omitting a value for a runtime-declared value parameter;
- dynamic engine arguments that attempt to override model, host, port, parallelism, memory, or scheduler fields owned by the normalized config.

`bind_host` must parse as an IP address, `port` must be nonzero, and IPv6 publish syntax is rendered explicitly. GPU identifiers are normalized and must be nonempty and unique. The correctness threshold is finite and lies in `(0, 1]`.

### CLI

Use `clap` for command structure, usage, help, version, value parsing, and successful `--help` behavior. Retained commands are:

- `plan`
- `params`
- `hardware`
- `serve`
- `bench`
- `sweep`
- `cleanup`
- `mcp` (parsed as a normal `clap` subcommand, then switched to ANSI-free stdio protocol mode)

The prototype `advise` command is removed.

Engine-specific CLI parameters are explicit repeatable `--serve-arg NAME=VALUE` and `--serve-flag NAME` options. Unknown CLI options are errors rather than guessed engine parameters. Runtime introspection remains the source of truth for validating those explicit options.

`plan`, `bench --dry-run`, and `sweep --dry-run` preserve host-tool-free previews. They perform local invariant validation and may use a parameter cache only when the requested image reference is already immutable and the cache identity matches exactly. Floating image references are labeled as pending runtime validation; a preview never pulls an image merely to print a plan.

#### CLI inputs, precedence, and output

Built-in defaults apply first, then a v2 TOML file, then explicit CLI overrides. A duplicate fixed option or dynamic engine argument within one source is an error. A CLI dynamic argument with the same canonical name as one TOML `[serve]` argument replaces the TOML value; all other duplicate/conflicting normalized arguments fail. Environment variables supply secrets and the documented `CUDA_VISIBLE_DEVICES`/`OPTIMUM_ADVISOR_HF_MEM` fallbacks only; they do not silently enable execution or leaderboard submission.

The retained commands have these contracts:

- `plan [--config PATH] [configuration overrides]` requires `engine` and `model` after merging and prints one validated, non-executing preview.
- `params --engine ENGINE [--image IMAGE] [--pull-policy POLICY] [--cache-dir PATH] [--refresh | --offline]` resolves and inspects by default; `--offline` requires an immutable image reference and an exact cache hit, while `--refresh` forces inspection.
- `hardware` performs local inspection and never requires Docker or a token.
- `serve [--config PATH] [configuration overrides]` starts the validated server and runs until it exits, times out during readiness, or is interrupted.
- `bench [--config PATH] [configuration overrides] [--results-dir PATH] [--dry-run]` executes one candidate unless `--dry-run` is present.
- `sweep --config PATH [--results-dir PATH] [--dry-run]` takes sweep dimensions only from TOML, preventing a second ad hoc sweep grammar in CLI flags.
- `cleanup [--run-id ID] [--dry-run]` lists or removes only labeled owned containers.
- `mcp` accepts no operational arguments and reserves stdout exclusively for newline-delimited JSON-RPC.

Configuration overrides are exactly `--engine`, `--image`, `--pull-policy`, `--allow-local-image`, `--model`, `--metric`, `--gpus`, repeatable `--gpu-device`, `--bind-host`, `--port`, `--startup-timeout-secs`, `--benchmark-timeout-secs`, `--max-process-output-bytes`, benchmark fields (`--dataset-name`, `--num-prompts`, `--request-rate`, `--benchmark-max-concurrency`, `--random-input-len`, `--random-output-len`), candidate fields (`--tensor-parallelism`, `--memory-fraction`, `--prefill-token-budget`, `--max-running-requests`), correctness fields (`--no-correctness`, `--correctness-threshold`, `--correctness-timeout-secs`), model-memory fields (`--no-model-memory`, `--require-model-memory`, `--hf-mem-command`, `--hf-mem-timeout-secs`), leaderboard fields (`--leaderboard-submit`, `--leaderboard-url`), and repeatable `--serve-arg NAME=VALUE` / `--serve-flag NAME`.

Human progress and diagnostics go to stderr. `plan`, `params`, and `hardware` put their requested data on stdout. Successful `bench` and `sweep` print the final report path and, when present, winning-config path on stdout. Errors go to stderr. MCP stdout contains protocol frames only; its operational diagnostics are bounded and returned inside tool results.

## Domain Types and Errors

Replace `pub type Result<T> = Result<T, String>` with a crate error type using `thiserror`. Variants carry structured context for:

- usage and configuration decoding;
- invariant validation;
- filesystem operation and path;
- process spawn, exit, timeout, cancellation, and output truncation;
- Docker image resolution and container cleanup;
- parameter inspection/cache;
- correctness and benchmark collection;
- HTTP transport/protocol;
- MCP/JSON-RPC protocol.

Execution failures expose a typed `ExecutionStage` rather than inferring the stage from message prefixes. CLI rendering remains concise and includes an error source chain. Exit codes are `0` for success, `2` for usage/configuration errors, `1` for runtime/evaluation failures, and `130` for interruption. MCP serializes the same structured fields in tool errors.

## Image and Parameter Integrity

Apply the configured image pull policy first: `missing` uses a matching local image or pulls when absent, `always` pulls first, and `never` requires a local image. Inspect the resulting requested reference once to obtain its exact image ID and repository digests, then select the digest whose Docker-normalized repository matches the requested repository. Every subsequent container uses that selected immutable reference, eliminating a retag race.

If no matching repository digest exists, execution fails by default. With explicit `allow_local_image = true`, it may use the immutable local image ID, records `local_only: true`, and forces the generated winning config to `pull_policy = "never"`. Such a config is reproducible only while that image ID remains in the local Docker store; the CLI states this limitation and leaderboard submission rejects local-only runs.

Server, benchmark, correctness support processes that use the image, and parameter inspection must all consume the same resolved identity. Delete package-version-to-image-tag guessing.

Parameter cache keys include only the engine plus a SHA-256 digest of the immutable image identity. Cache contents use typed serialization and include a cache schema version, engine, immutable identity, and parameter list; requested image aliases are informational and do not invalidate an identity match. Writes are atomic. A malformed, mismatched, or old cache is rejected and refreshed when execution permits it.

Execution validation accepts only structured engine introspection. The vLLM and SGLang scripts inspect their Python argument-parser actions and emit parameter names, canonical name, value mode (`none`, `required`, or `optional`), repeatability, and finite choices when available. A version that cannot produce this schema fails inspection; production execution never falls back to scraping human `--help` text. Validation enforces known names and syntactic value mode. Cross-argument and model-dependent semantics remain the engine’s responsibility and surface as typed startup failures.

A lookup map replaces repeated linear scans.

## Managed Process and Container Runtime

### Process specification

`ProcessSpec` includes:

- program and argument vector;
- explicitly added or removed environment variables;
- working directory when needed;
- optional execution deadline;
- maximum persisted bytes per stdout/stderr stream;
- optional owned-container metadata;
- a human-safe display form that redacts secrets.

Short-lived commands always have a positive deadline. An evaluation server has a separate positive readiness deadline and remains managed until its bounded correctness/benchmark children finish. The interactive `serve` command intentionally has no lifetime deadline after readiness and exits only when the server exits or cancellation is requested.

Secret-bearing environment values are added without being copied into the display form. Empty optional secrets, including `HF_TOKEN`, are omitted rather than treated as a universal preflight requirement; public models can run without a token, while a gated-model authentication failure remains a typed server-start failure.

Every external invocation uses one executor. Direct `Command::output()` calls disappear from production modules.

### Bounded output

The executor drains stdout and stderr concurrently so child pipe buffers cannot deadlock. Each drain writes at most `max_process_output_bytes` to its artifact and maintains a 64 KiB tail for diagnostics. Further bytes are drained and discarded, and the outcome records truncation. Full in-memory `Output` capture is not used.

A truncated stdout stream that is parsed as JSON, a parameter schema, an identity response, or benchmark metrics is invalid and fails that operation; parsers never accept a complete-looking prefix. Truncated diagnostic stderr remains valid but is marked. Secret-producing commands use a separate 64 KiB in-memory capture policy: they write no artifact or diagnostic tail, reject truncation, wrap the parsed secret in zeroizing/redacted storage, and redact all failure output.

Run directories are private to the current user on supported Unix hosts (`0700` directories and `0600` files). Known credentials are redacted before diagnostic tails or persisted logs are finalized, and checksums are computed from the sanitized artifact bytes.

Default limits are:

- startup: 300 seconds;
- correctness: 600 seconds;
- benchmark: 1800 seconds;
- image/parameter inspection: 300 seconds;
- persisted bytes per output stream: 16 MiB;
- diagnostic tail per stream: 64 KiB;
- graceful shutdown before force kill: 10 seconds.

Every configured duration and byte limit is positive. The benchmark, correctness, model-memory, and output limits are configurable in v2 TOML and corresponding MCP inputs.

### Cancellation and cleanup

`signal-hook` installs `SIGINT` and `SIGTERM` handling. A cancellation token is polled while waiting for children. Cancellation stops scheduling new trials, terminates the current process, force-removes its owned container if present, waits for the child, atomically checkpoints the interrupted run, and exits with the interruption code.

On supported Unix hosts, each non-container child starts in its own process group. Timeout or cancellation sends termination to the group, waits up to 10 seconds, then force-kills the group so descendants cannot retain output pipes. Owned Docker containers are force-removed before the Docker client is reaped. Output-drain threads are joined only after the group/container is gone, making termination bounded. Linux is the production execution platform; macOS supports host-only development and tests.

The managed-child guard performs the same scoped cleanup on ordinary early returns. Cleanup errors are recorded and surfaced; they are not silently swallowed. `SIGKILL` cannot be intercepted, so the `cleanup` command removes only containers carrying `optimum-advisor=true`, optionally filtered by run ID. It never deletes arbitrary Docker resources.

Server, benchmark, and inspection containers receive unique names and labels containing role, engine, and run ID. Docker `--rm` remains a secondary safeguard.

### Network and GPU exposure

Model ports publish on the validated `bind_host`, which defaults to `127.0.0.1`; IPv6 addresses use Docker’s bracketed publish syntax. Before spawn, the target address must be free. Each adapter declares `/v1/models` as its readiness endpoint. Readiness requires the managed child to remain alive, an HTTP 2xx JSON response, and a model entry matching the configured model. A different or malformed service on the port is never accepted as ready.

GPU selection is deterministic. Explicit `gpu_devices` wins; otherwise a nonempty `CUDA_VISIBLE_DEVICES` supplies candidates; otherwise hardware inspection selects the first `gpus` devices by ascending NVIDIA index. Indices resolve to UUIDs, all normalized selections must be unique and present, and execution fails if fewer than `gpus` devices exist. Docker receives the exact selected device identifiers rather than a count or `all`, and reports record those identifiers. Standalone `hardware` inspection may still return warnings instead of failing.

## Evaluation and Sweep Semantics

Preflight, immutable image resolution, sweep expansion, and validation of every generated candidate occur before GPU work starts. A run-scoped preflight, configuration, image, or report-I/O failure aborts before trials or terminates the run. A candidate-scoped startup, correctness, benchmark, metric-collection, timeout, or OOM failure becomes a failed `TrialOutcome`, checkpoints the report, and allows the next candidate.

There are no automatic retries.

The run directory and an initial `running` report checkpoint are created before external preflight. A later run-scoped failure updates that checkpoint to `failed` whenever report I/O is still available.

Sweep expansion:

- rejects a `sweep` command with no real dimension or with an empty dimension array;
- applies fixed dimensions in this order: tensor parallelism, memory fraction, prefill token budget, max running requests;
- applies `[sweep.serve]` keys in lexical canonical-name order and each key’s values in TOML order;
- treats a swept value as a replacement for the corresponding base `[candidate]` or `[serve]` value;
- makes the last listed dimension vary fastest;
- detects multiplication overflow before allocation;
- deduplicates normalized candidates while preserving the first generated occurrence;
- defaults to at most 256 trials and requires an explicit higher `max_trials` to exceed it.

A successful trial requires:

- the requested stages to complete;
- requested correctness to produce a known status and clear its threshold;
- benchmark output to contain a finite value for the selected winning metric;
- no positive `failed_requests` value when that field is present;
- artifact collection to complete.

Ranking never reorders stored trial history. It computes `best_trial_index` over successful trials. Missing correctness ranks as unknown, never passed. Failed trials cannot win. Lower-is-better and higher-is-better metric direction remains explicit in `Metric`.

Parsed `-0.0` is normalized to `0.0`. Equal metric values retain the lower original trial index, so ties are deterministic.

Run states are exactly `running`, `completed`, `completed_with_failures`, `failed`, and `interrupted`. A sweep with a winner exits successfully and returns `completed_with_failures` when any other trial failed. An all-failed sweep returns `failed` and exit code 1. Cancellation returns `interrupted` and exit code 130.

If no trial succeeds, the sweep finishes with a durable report and does not write a winning config. On success, `best.conf` is a complete v2 single-candidate TOML document: it contains the exact normalized candidate and dynamic serve arguments, retains correctness and model-memory policy, removes every `[sweep]` section, forces `leaderboard.submit = false`, contains no secret or results path, and uses the resolved repository digest with `pull_policy = "missing"`. An explicitly allowed local-only image instead uses its image ID with `allow_local_image = true` and `pull_policy = "never"`.

## Report and Artifact Contract

Report schema v2 is serialized with `serde_json`, never string concatenation. Its top-level contract contains:

- `schema_version: 2`;
- run ID, kind, state, engine, required start Unix milliseconds, nullable end Unix milliseconds, and nullable duration while `state = "running"`;
- requested winning metric;
- requested image reference plus nullable resolved image identity and nullable selected hardware until preflight discovers them;
- ordered `trials`;
- nullable `best_trial_index` and `best_winning_value`;
- run-level failure/interruption when applicable.

Final states require end time and duration. `running` requires both to be null. Completed states require resolved image and selected hardware; an early `failed` or `interrupted` report may retain null preflight fields. `best_trial_index` references the original ordered trial array.

Each trial contains:

- index and `success` or `failed` status;
- normalized executable config;
- metrics and correctness summary when available;
- model-memory estimate or warning;
- typed failure with stage, kind, message, optional exit code, timeout/interruption flags, and output tails;
- an artifact manifest.

Artifact manifest entries contain a run-directory-relative path, byte count, SHA-256, and truncation flag. Reports do not embed arbitrary full logs or correctness artifact bodies. Paths escaping the run directory are rejected.

Reports, winning configs, parameter caches, and generated correctness artifacts are written to a same-directory temporary file, flushed, `sync_all`ed, atomically renamed, and followed by a parent-directory sync on platforms that support it. A failed replacement leaves the previous checkpoint readable.

Artifact/config files are fully written, sanitized, synced, renamed, and parent-synced before a report checkpoint may reference them. The report is committed last. A crash can leave an unreferenced artifact, but never a readable report pointing to an uninstalled artifact or marking a trial successful before its manifest is durable. `best.conf` is installed before the final completed report that references it.

## Correctness and Metric Parsing

Use typed `serde_json` structures or `serde_json::Value` navigation for lighteval, capability probes, benchmark JSON, model-memory JSON, leaderboard responses, and SSE payloads. Delete all hand-written JSON scanners/escapers.

Every owned correctness task declares its expected metric explicitly (`em` for GSM8K, TriviaQA, and DROP; `prompt_level_strict_acc` for IFEval). Typed `serde_json` navigation selects that exact finite number. A missing task, missing metric, duplicate/invalid result shape, or non-finite score yields a typed collection failure when correctness was requested; there is no “first number” fallback.

Correctness passes only when every configured task score is greater than or equal to the threshold and every requested capability probe passes. Any score below the threshold yields `failed`; missing or malformed results are collection failures rather than `unknown`. `unknown` is reserved for comparisons where correctness was not requested.

Benchmark text parsing retains the known engine labels but returns a parse result containing recognized and unrecognized fields. Absence of the selected metric is an error. Parsing and diagnostic tails are UTF-8-safe.

## Model-Memory Integration

Run only an explicitly installed `hf-mem` binary, configurable by path or `OPTIMUM_ADVISOR_HF_MEM`. Never invoke `uvx` or implicitly download executable code.

Parse JSON output with `serde_json`. Missing or failed optional estimation becomes a typed warning attached to the trial. A caller may opt into requiring an estimate, in which case it is a preflight failure. Secret or token values are not included in command displays or persisted output.

## Leaderboard Integration

Use `ureq` 3 with rustls and platform certificate verification. JSON request bodies use `serde_json`. Bearer tokens and submit keys use zeroizing, redacted wrappers and never appear in argv, `Debug`, logs, errors, reports, or generated configs.

HTTP limits are:

- 10 seconds for DNS/connect/response headers;
- 60 seconds overall for ordinary requests;
- 300 seconds overall for streamed completion;
- 5-second response-body read slices so OS/MCP cancellation is observed within one slice;
- 1 MiB for ordinary successful JSON bodies;
- 64 KiB for non-2xx diagnostic bodies;
- 256 KiB per SSE event and 1 MiB accumulated streamed response;
- 16 MiB maximum report upload.

Credentialed plaintext HTTP is rejected except for an explicit loopback test endpoint. Follow at most three same-origin HTTPS redirects and never forward credentials across an origin change. Gradio event IDs and SSE `data` fields are parsed structurally; oversized or malformed events fail without unbounded accumulation.

Submission remains explicit through TOML or `--leaderboard-submit`. Authentication prefers private token sources and may use supported HF CLI identity/token commands through the managed executor. Token-returning commands use secret-output capture and never create artifacts. A cancellation during HTTP waits at most the current five-second body slice or ten-second connect/header deadline, then returns the interrupted result.

Submission occurs after the evaluation report and winning config are durable. Its accepted, queued, or failed result is added to the report in one final atomic checkpoint. A requested submission failure preserves the winner and artifacts, changes a previously `completed` run to `completed_with_failures`, and returns exit code 1 (or an MCP `isError` result) with the report path.

## MCP Contract

The MCP stdio server supports protocol version `2025-11-25` and uses strict typed request/tool DTOs. Server-side decoding rejects unknown fields even if a client ignores JSON Schema.

Lifecycle states are `awaiting_initialize`, `awaiting_initialized_notification`, and `ready`. Except for `ping`, `initialize` is the first request. The server responds with `2025-11-25`; a client that requested an incompatible version must disconnect. Tool requests are rejected until `notifications/initialized` transitions the server to `ready`. Repeated initialization is an invalid request, EOF is clean transport shutdown, and no nonstandard shutdown method is added.

A dedicated bounded reader thread continues framing stdin while the dispatcher executes at most one tool call. It recognizes `notifications/cancelled` with a matching in-flight `requestId` and sets that request’s cancellation token; `initialize` is not cancellable. Other requests remain queued in arrival order. Child execution polls the token, HTTP body reads poll it between five-second slices, and cancellation returns a structured interrupted tool result while leaving the MCP server ready. OS `SIGINT`/`SIGTERM` instead cancels active work, checkpoints it, and terminates the server.

Framing accepts a final non-newline-terminated frame at EOF. A frame exceeding 1 MiB is drained through its newline without further allocation, returns one `-32700` error with null ID because its envelope is untrusted, and allows the next frame. Notifications never receive responses. Each response is one JSON object followed by one LF byte and is flushed.

Protocol parse/invalid-request/method-not-found/envelope-parameter failures use JSON-RPC error codes. Once a valid `tools/call` envelope names a known tool, strict tool-input or execution failures return MCP `isError` tool results, not JSON-RPC protocol errors. All run-scoped failures, all-failed sweeps, and cancellation include their report/artifact path when one exists.

The v2 tool set is:

- `inspect_hardware {}` → selected-capable `HardwareProfile`;
- `inspect_engine { engine, image?, pull_policy?, allow_local_image?, cache_dir?, refresh?, offline? }` → `ParameterSchema`;
- `validate_config { config, cache_dir?, offline? }` → normalized validation result;
- `estimate_memory { config }` → estimate or typed warning;
- `check_correctness`, `run_benchmark`, and `evaluate_candidate { config, results_dir?, cache_dir? }` → one candidate outcome;
- `run_sweep { config, results_dir?, cache_dir? }` → report summary and path;
- `rank_candidates { metric, candidates[] }` → stable ordered ranks.

`config` uses the same nested v2 DTO and defaults as TOML, excluding `schema_version`, `[sweep]` where the tool is single-candidate, and all secrets. Candidate score IDs must be nonempty and unique. Tool schemas are generated from these DTOs with `schemars`; annotations mark filesystem, Docker, process, and network behavior accurately.

## Terminal Behavior

Color is enabled only when the destination is a real terminal and `NO_COLOR` is absent. MCP buffers, redirected output, files, tests, and report tails contain no ANSI sequences.

Human progress output remains stable and concise. Machine-readable report, TOML, and MCP outputs never depend on terminal formatting.

## File and Module Boundaries

The implementation should converge on these responsibilities without creating one-file wrappers:

- `src/error.rs`: crate errors, execution stages, error kinds, CLI exit mapping.
- `src/cli/`: `clap` DTOs and conversion into normalized requests.
- `src/config/`: strict TOML DTOs, defaults/overrides, shared normalization and validation.
- `src/inspection/`: hardware discovery and explicitly configured model-memory estimation.
- `src/domain/`: validated engine, metric, candidate, sweep, runtime, and trial-outcome types.
- `src/engines/`: engine-specific parameter defaults and command construction only.
- `src/runtime/`: managed process execution, cancellation, Docker image identity, owned-container cleanup, bounded output.
- `src/results/`: ranking, report-v2 DTOs, artifact manifests, atomic persistence.
- `src/correctness/`: correctness plans and typed artifact collection.
- `src/leaderboard/`: redacted credentials, Hugging Face identity lookup, and native HTTP protocol.
- `src/mcp/`: bounded stdio framing, JSON-RPC lifecycle, strict tool DTOs, and generated schemas.
- `src/app.rs`: thin command orchestration; no parsing, JSON building, or subprocess mechanics.
- `src/evaluation.rs`: the shared candidate-evaluation service used directly by CLI and MCP orchestration.

The crate becomes binary-only: remove `src/lib.rs` and move integration tests to the executable surface. All implementation modules are internal, avoiding an accidental third “embedder” compatibility contract. Thin wrappers from the current `tools.rs` are removed rather than moved.

Existing tiny modules should remain combined when splitting would only create a delegating wrapper.

## Dependencies

Expected runtime additions:

- `clap` with derive support;
- `toml`;
- `schemars` for MCP schemas generated from strict DTOs;
- `thiserror`;
- `ureq` 3 with `json`, `rustls`, and platform-verifier support;
- `signal-hook`;
- `nix` with process/signal support for safe process-group termination;
- `sha2`;
- `zeroize`.

Expected development addition:

- `tempfile`.

Do not add an async runtime, general dependency-injection framework, logging facade, or compatibility framework. Before finalizing dependency versions, verify their current maintenance status, MSRV, default features, TLS behavior, and transitive footprint. Disable unused default features.

## Verification

### Focused contract tests

Tests must cover:

- user overrides winning over engine/metric defaults;
- invalid values rejected rather than clamped;
- strict TOML duplicate/unknown keys and generated-config round trips;
- strict MCP unknown fields and numeric bounds;
- missing correctness ranked below passed correctness;
- lower/higher metric direction;
- missing or non-finite selected metrics;
- failed-request rejection;
- deterministic, bounded, overflow-safe sweep expansion;
- candidate failure continuation and per-trial checkpointing;
- all-failed and interrupted run reports;
- secret redaction across debug, error, and HTTP paths;
- UTF-8-safe bounded tails;
- parameter cache identity and flag/value validation;
- atomic-write preservation after an injected replacement failure.

### Runtime tests

Local fixture processes verify normal exit, non-zero exit, timeout, cancellation, concurrent output draining, truncation, and cleanup ordering. Docker command construction is testable without Docker. A fake Docker executable verifies digest resolution, cache keys, and scoped cleanup calls.

### CLI and MCP smoke tests

Exercise successful help/version behavior, every retained CLI command in dry-run/local-only form, malformed JSON-RPC, initialization order, message-size limits, tool listing, strict tool input, structured execution failures, and ANSI-free framing.

### GPU-host acceptance

Provide one opt-in ignored acceptance command for a real NVIDIA host. It must run:

1. hardware/device selection;
2. immutable image and parameter inspection;
3. a tiny correctness plus benchmark candidate;
4. a two-candidate sweep with both candidates checkpointed;
5. interruption during a running child;
6. owned-container leak verification;
7. report-v2 and winning-config validation.

Host-only CI must not fake GPU success.

### Continuous integration

Add jobs for:

- `cargo fmt --all -- --check`;
- `cargo clippy --all-targets --all-features -- -D warnings`;
- `cargo test --all-targets --all-features`;
- release build;
- declared-MSRV build/test;
- RustSec dependency audit;
- shell syntax/static checks for maintained scripts.

Declare and verify an MSRV compatible with selected dependencies rather than assuming the developer’s current toolchain.

## Documentation and Delivery

After behavior passes focused smoke tests, update README and examples for v2 TOML, strict engine arguments, report v2, timeout/cancellation, cleanup, GPU selection, secret handling, MCP initialization, and the GPU acceptance command. State exact limitations: sequential local execution, Docker/NVIDIA requirement, optional externally installed `hf-mem`, and no production advisor heuristics.

Completion requires:

- all in-repository callers and examples migrated;
- no obsolete aliases, manual JSON helpers, string-prefix stage classifier, `uvx` fallback, or prototype advisor code;
- focused and full verification passing;
- no tracked or untracked user work overwritten;
- no commit and no push.
