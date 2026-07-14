# Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` and execute these checkbox (`- [ ]`) steps in order. This plan is intentionally configured for inline execution in the current working tree. Do not commit or push at any point.

**Goal:** Replace the prototype’s permissive parsing, fragile process lifecycle, lossy sweep behavior, hand-written JSON, and secret-unsafe integrations with the approved strict v2 local CLI/MCP design.

**Architecture:** Decode CLI, TOML, and MCP into strict source DTOs, normalize through one validator, resolve an immutable Docker image into `ExecutableConfig`, execute through one bounded/cancellable process runtime, and checkpoint typed `TrialOutcome` values through atomic report-v2 persistence. Keep the runtime synchronous; use a reader thread only where blocking I/O must remain cancellable.

**Tech Stack:** Rust 2021 with MSRV 1.85; `serde`, `serde_json`, `clap` 4.6.1, `toml` 1.1.2, `schemars` 1.2.1, `thiserror` 2.0.18, `ureq` 3.3.0, `signal-hook` 0.4.4, `nix` 0.31.3, `sha2` 0.11.0, `zeroize` 1.9.0, and dev-only `tempfile` 3.27.0.

## Global Constraints

- Clean cutover only. Migrate every in-repository caller; leave no v1 parser, alias, deprecated flag, compatibility re-export, or report shim.
- Keep a synchronous, single-process CLI and stdio MCP server. Do not add Tokio or a daemon.
- Production execution is Linux; macOS must retain host-only development and tests.
- Never silently clamp invalid user input.
- Never invoke `uvx` or implicitly download executable helper code.
- Never place tokens or submit keys in argv, ordinary CLI values, TOML, debug output, errors, logs, reports, or generated configs.
- Preserve the pre-existing untracked `Todos.md` byte-for-byte.
- Baseline `Todos.md` SHA-256: `e736408f314a0c5c0647fa542e437552a5bcb023a1f27d7c8fe69da7be4f01b2`.
- Do not commit or push.
- Do not run project-wide format/lint/test commands until the focused behavioral smoke gate in Task 10.
- Cleanup documentation/CI work is deliberately not decomposed before the behavior smoke gate; append exact cleanup tasks only after Task 10 proves the cutover works, per the repository execution policy.

## Target File Structure

```text
src/
  main.rs
  app.rs
  error.rs
  evaluation.rs
  cli/
    mod.rs
  config/
    mod.rs
    file.rs
    validate.rs
  domain/
    mod.rs
    engine.rs
    candidate.rs
    run.rs
  engines/
    mod.rs
    params.rs
    serve.rs
    vllm/mod.rs
    sglang/mod.rs
  runtime/
    mod.rs
    atomic.rs
    cancel.rs
    docker.rs
    process.rs
  results/
    mod.rs
    artifact.rs
    metrics.rs
    report.rs
  correctness/
    mod.rs
    suite.rs
  inspection/
    mod.rs
    hardware.rs
    model_memory.rs
  leaderboard/
    mod.rs
    auth.rs
    client.rs
  mcp/
    mod.rs
    protocol.rs
    tools.rs
    schema.rs
tests/
  cli_smoke.rs
  mcp_smoke.rs
```

Delete after migration:

```text
src/lib.rs
src/tools.rs
src/domain/config.rs
src/domain/logs.rs
src/domain/trial.rs
src/advisor/mod.rs
src/advisor/hardware.rs
src/advisor/model_memory.rs
```

---

### Task 1: Error and dependency foundation

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock` through Cargo
- Create: `src/error.rs`
- Modify: `src/lib.rs` to expose `error` internally until Task 9 moves all module declarations to the binary root

**Interfaces:**
- Produces: `crate::error::{Error, ErrorKind, ExecutionStage, Result}`
- Produces: `Error::exit_code() -> u8`
- Consumes: no new internal interfaces

- [ ] **Step 1: Add a failing exit-code contract test**

Add unit tests in `src/error.rs` before defining the types:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_error_categories_to_stable_exit_codes() {
        assert_eq!(Error::usage("bad flag").exit_code(), 2);
        assert_eq!(Error::runtime(ExecutionStage::Benchmark, "failed").exit_code(), 1);
        assert_eq!(Error::interrupted(ExecutionStage::Benchmark).exit_code(), 130);
    }
}
```

- [ ] **Step 2: Run the focused test and confirm it fails to compile**

Run:

```bash
cargo test --lib error::tests::maps_error_categories_to_stable_exit_codes
```

Expected: compile failure because `Error`, constructors, and `ExecutionStage` do not exist.

- [ ] **Step 3: Add exact dependency and package metadata**

Set `rust-version = "1.85"` and add:

```toml
clap = { version = "4.6.1", features = ["derive"] }
nix = { version = "0.31.3", default-features = false, features = ["process", "signal"] }
schemars = "1.2.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
sha2 = "0.11.0"
signal-hook = "0.4.4"
thiserror = "2.0.18"
toml = "1.1.2"
ureq = { version = "3.3.0", default-features = false, features = ["json", "rustls", "platform-verifier"] }
zeroize = { version = "1.9.0", features = ["derive"] }

[dev-dependencies]
tempfile = "3.27.0"
```

Use `cargo update` only to resolve the edited manifest; do not upgrade unrelated dependencies manually.

Verify the selected versions against official crate metadata, then run:

```bash
cargo tree -e features
cargo tree --duplicates
rustup run 1.85.0 cargo check --all-targets
```

Expected: required features are explicit; ureq uses rustls plus `platform-verifier` and not native TLS; nix enables only process/signal; the dependency graph has no unexplained duplicate major versions; the crate checks on Rust 1.85.0. Install only that minimal Rust toolchain if it is absent.

- [ ] **Step 4: Implement typed errors**

Implement these shapes in `src/error.rs`; builders set only fields relevant to the operation, and MCP serializes `ErrorPayload::from(&error)`:

```rust
use std::{error::Error as StdError, path::PathBuf};
use thiserror::Error as ThisError;

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionStage {
    Preflight,
    ImageResolution,
    Validation,
    Server,
    Correctness,
    Benchmark,
    ResultCollection,
    Persistence,
    Leaderboard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ErrorKind {
    Usage,
    Configuration,
    Validation,
    Io,
    ProcessSpawn,
    ProcessExit,
    Timeout,
    Interrupted,
    OutputTruncated,
    Docker,
    ParameterInspection,
    Correctness,
    Benchmark,
    HttpTransport,
    HttpProtocol,
    Protocol,
}

#[derive(Clone, Debug, Default, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct ErrorContext {
    #[serde(skip_serializing_if = "Option::is_none")] pub operation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")] pub process: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub deadline_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")] pub stream: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub docker_identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub container: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub cache_identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub http_status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")] pub child_exit_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")] pub report_path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")] pub artifact_path: Option<PathBuf>,
}

#[derive(Debug, ThisError)]
#[error("{message}")]
pub(crate) struct Error {
    pub kind: ErrorKind,
    pub stage: Option<ExecutionStage>,
    pub message: String,
    pub context: ErrorContext,
    #[source]
    pub source: Option<Box<dyn StdError + Send + Sync>>,
}

#[derive(Clone, Debug, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct ErrorPayload {
    pub kind: ErrorKind,
    #[serde(skip_serializing_if = "Option::is_none")] pub stage: Option<ExecutionStage>,
    pub message: String,
    #[serde(flatten)] pub context: ErrorContext,
}
```

Add `usage`, `configuration`, `runtime`, and `interrupted` constructors, context builder methods, `kind()`, `stage()`, `payload()`, and stable exit-code mapping. Runtime/all-failed/interrupted builders accept optional report and artifact paths; process, Docker, cache, and HTTP boundaries fill their structured context rather than encoding it into prefixes. Implement `From<serde_json::Error>` only at explicit protocol/serialization boundaries so it cannot discard the execution stage.

- [ ] **Step 5: Run the focused test**

Run:

```bash
cargo test --lib error::tests::maps_error_categories_to_stable_exit_codes
```

Expected: PASS.

---

### Task 2: Strict v2 configuration and domain invariants

**Files:**
- Create: `src/config/mod.rs`
- Create: `src/config/file.rs`
- Create: `src/config/validate.rs`
- Create: `src/domain/candidate.rs`
- Create: `src/domain/run.rs`
- Modify: `src/domain/mod.rs`
- Modify: `src/domain/engine.rs`
- Modify: `src/lib.rs` to declare `config`; retain the old domain modules only until their in-repository callers move in Tasks 5 and 8

**Interfaces:**
- Produces: `ConfigFile`, `ConfigInput`, `ConfigOverrides`, `NormalizedConfig`, `ExecutableConfig`
- Produces: `RuntimeConfig`, `BenchmarkConfig`, `CorrectnessConfig`, `ModelMemoryConfig`, `LeaderboardConfig`, `Candidate`, `CandidateOverrides`, `CandidateSpec`, `DynamicArg`, `SweepSpec`, `ResolvedImage`, `GpuRecord`, `HardwareProfile`
- Produces: `parse_config_text(&str) -> Result<ConfigFile>` and `load_config(&Path) -> Result<ConfigFile>`
- Produces: `ConfigInput::normalize() -> Result<NormalizedConfig>`
- Produces: `SweepSpec::candidates(&CandidateSpec, usize) -> Result<Vec<CandidateSpec>>`
- Consumes: `Engine`, `Metric`, typed errors

- [ ] **Step 1: Write failing strict-TOML and override tests**

Add tests in `src/config/file.rs` and `src/config/validate.rs` that assert:

```rust
#[test]
fn rejects_unknown_and_duplicate_toml_keys() {
    assert!(parse_config_text("schema_version=2\nengine='vllm'\nmodel='m'\nunknown=1").is_err());
    assert!(parse_config_text("schema_version=2\nengine='vllm'\nengine='sglang'\nmodel='m'").is_err());
}

#[test]
fn explicit_candidate_values_survive_engine_defaults() {
    let input = ConfigInput::minimal(Engine::Sglang, "m")
        .with_candidate_overrides(CandidateOverrides {
            memory_fraction: Some(0.73),
            prefill_token_budget: Some(4096),
            ..CandidateOverrides::default()
        });
    let normalized = input.normalize().unwrap();
    assert_eq!(normalized.candidate.memory_fraction, 0.73);
    assert_eq!(normalized.candidate.prefill_token_budget, 4096);
}

#[test]
fn rejects_parallelism_instead_of_clamping() {
    let mut input = ConfigInput::minimal(Engine::Vllm, "m");
    input.runtime.gpus = 2;
    input.candidate.tensor_parallelism = Some(3);
    assert!(input.normalize().unwrap_err().to_string().contains("divide"));
}
```

Add sweep tests for deterministic dimension order, empty arrays, overflow, deduplication, reserved dynamic names, and the 256 default limit.

- [ ] **Step 2: Run focused configuration tests and confirm failure**

Run:

```bash
cargo test --lib config::
```

Expected: compile failure because the new modules/types do not exist.

- [ ] **Step 3: Implement serializable domain enums**

Update `Engine` and `Metric` to derive `Serialize`, `Deserialize`, and `JsonSchema`; use `FromStr` instead of ad hoc `parse` entry points while retaining one canonical alias table for metrics. Remove `Mode` from the domain; command choice belongs to `cli`.

Define:

```rust
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Candidate {
    pub tensor_parallelism: usize,
    pub memory_fraction: f64,
    pub prefill_token_budget: u32,
    pub max_running_requests: u32,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct CandidateOverrides {
    pub tensor_parallelism: Option<usize>,
    pub memory_fraction: Option<f64>,
    pub prefill_token_budget: Option<u32>,
    pub max_running_requests: Option<u32>,
}
```

```rust
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct DynamicArg {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct CandidateSpec {
    #[serde(flatten)]
    pub candidate: Candidate,
    pub serve_args: Vec<DynamicArg>,
}
```

Engine defaults return a complete `Candidate`; apply `CandidateOverrides` afterward exactly once.

- [ ] **Step 4: Implement strict file DTOs**

Use `#[serde(deny_unknown_fields)]` on every fixed section, require `schema_version == 2`, and represent `[serve]` and `[sweep.serve]` as ordered maps whose scalar/list values convert explicitly into `EngineArgInput`.

The primary file DTO is:

```rust
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ConfigFile {
    pub schema_version: u32,
    pub engine: Option<Engine>,
    pub image: Option<String>,
    pub model: Option<String>,
    pub metric: Option<Metric>,
    #[serde(default)] pub runtime: RuntimeInput,
    #[serde(default)] pub benchmark: BenchmarkInput,
    #[serde(default)] pub candidate: CandidateOverrides,
    #[serde(default)] pub correctness: CorrectnessInput,
    #[serde(default)] pub model_memory: ModelMemoryInput,
    #[serde(default)] pub leaderboard: LeaderboardInput,
    #[serde(default)] pub serve: toml::Table,
    pub sweep: Option<SweepInput>,
}
```

The normalized/report-facing infrastructure records are pure domain values so later Docker and inspection modules populate, rather than own, their serialized contracts:

```rust
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct ResolvedImage {
    pub requested: String,
    pub immutable: String,
    pub local_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct GpuRecord {
    pub index: u32,
    pub uuid: String,
    pub name: String,
    pub compute_capability: Option<String>,
    pub memory_total_mib: u64,
    pub memory_free_mib: u64,
    pub memory_used_mib: u64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, schemars::JsonSchema)]
pub(crate) struct HardwareProfile {
    pub source: String,
    pub cuda_visible_devices: Option<String>,
    pub all_gpus: Vec<GpuRecord>,
    pub selected_gpus: Vec<GpuRecord>,
    pub warnings: Vec<String>,
}
```

`parse_config_text(text)` decodes one TOML document transactionally for unit tests. `load_config(path)` reads once, calls the text decoder, and adds path/span context to `ErrorKind::Configuration`. Missing `engine` or `model` remains representable until the TOML and CLI sources have merged; normalization then requires both.

- [ ] **Step 5: Implement one normalization validator**

`ConfigInput::normalize()` must:

1. derive engine/metric defaults;
2. apply file DTO values;
3. apply CLI overrides;
4. canonicalize dynamic names;
5. reject same-source duplicates and reserved engine-owned names;
6. validate model/image, IP/port, positive bounds, finite fractions, request rate, unique GPU IDs, parallelism divisibility, threshold `(0,1]`, and output limits;
7. produce `NormalizedConfig` without an immutable image identity.

Do not mutate a partially validated object visible to callers.

- [ ] **Step 6: Implement deterministic sweep expansion**

Use checked multiplication before allocation. Fixed dimension order is tensor, memory, prefill, max-running; dynamic keys sort lexically and TOML array order is retained. The last dimension varies fastest. Expand to `CandidateSpec` so each fixed candidate retains its canonical dynamic `serve_args`. Normalize `-0.0` to `0.0`, reject empty dimensions, reject no-op sweep specs, and deduplicate the complete fixed-plus-dynamic candidate with a key set while retaining first occurrence.

- [ ] **Step 7: Run focused configuration/domain tests**

Run:

```bash
cargo test --lib config::
cargo test --lib domain::candidate::
```

Expected: PASS.

---

### Task 3: Typed metrics, report v2, and atomic persistence

**Files:**
- Create: `src/runtime/atomic.rs`
- Create: `src/results/artifact.rs`
- Create: `src/results/metrics.rs`
- Create: `src/results/report.rs`
- Modify: `src/results/mod.rs` to declare typed submodules; delete its legacy top-level serializer/ranking code in Task 8 when callers migrate

**Interfaces:**
- Produces: `atomic_write(path: &Path, mode: u32, bytes: &[u8]) -> Result<()>`
- Produces: `BenchmarkMetrics::parse(text: &str, selected: Metric) -> Result<BenchmarkMetrics>`
- Produces: `TrialOutcome::{Success, Failed}`, `ReportRequest`, and `RunReport`
- Produces: `RunReport::best_trial_index() -> Option<usize>`, `checkpoint(&Path) -> Result<()>`, and `finish(RunState, u64, Option<ErrorPayload>) -> Result<()>`
- Produces: `ArtifactManifest::from_path(run_dir: &Path, path: &Path, truncated: bool) -> Result<ArtifactManifest>`

- [ ] **Step 1: Write failing ranking/report/atomic tests**

Cover these observable contracts:

```rust
#[test]
fn missing_correctness_ranks_below_passed() {
    assert_eq!(
        compare_observations(
            Metric::Tps,
            Some(CorrectnessStatus::Passed),
            Some(1.0),
            None,
            Some(2.0),
        ),
        Ordering::Greater
    );
}

#[test]
fn ties_keep_the_earlier_trial() {
    let observations = [
        RankableObservation { index: 0, correctness: None, value: 1.0 },
        RankableObservation { index: 1, correctness: None, value: 1.0 },
    ];
    assert_eq!(select_best(Metric::Tps, &observations), Some(0));
}

#[test]
fn running_report_has_nullable_final_fields() {
    let report = RunReport::running(
        ReportRequest {
            run_id: "run-1".into(),
            kind: RunKind::Bench,
            engine: Engine::Vllm,
            winning_metric: Metric::Tps,
            requested_image: "repo/image:tag".into(),
        },
        1_000,
    );
    let value = serde_json::to_value(report).unwrap();
    assert_eq!(value["schema_version"], 2);
    assert!(value["ended_at_unix_ms"].is_null());
    assert!(value["duration_ms"].is_null());
}
```

Add an injected atomic-rename failure test proving an existing destination remains readable. Add path-escape, SHA-256, and generated `best.conf` round-trip tests.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
cargo test --lib results::
cargo test --lib runtime::atomic::
```

Expected: compile failure for missing modules/types.

- [ ] **Step 3: Implement atomic writes**

Create same-directory temporary files with `create_new`, mode `0600`, write/flush/`sync_all`, rename, then sync the parent directory on Unix. Private run directories use `0700`. Expose an internal rename injection only under `#[cfg(test)]`; production has no fault-injection flag.

- [ ] **Step 4: Replace hand-built report JSON with DTO serialization**

Define the complete report DTO; `CorrectnessResult` remains the typed correctness summary and `ResolvedImage`/`HardwareProfile` come from Task 2:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunKind { Bench, Sweep }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunState { Running, Completed, CompletedWithFailures, Failed, Interrupted }

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct WarningRecord {
    pub kind: ErrorKind,
    pub stage: ExecutionStage,
    pub message: String,
}
#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct ModelMemoryEstimate {
    pub source: String,
    pub model: String,
    pub max_model_len: u32,
    pub batch_size: u32,
    pub kv_cache_dtype: String,
    pub weights_bytes: Option<u64>,
    pub kv_cache_bytes: Option<u64>,
    pub activation_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SubmissionState { Accepted, Queued, Failed }

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct SubmissionResult {
    pub state: SubmissionState,
    pub message: String,
    pub remote_id: Option<String>,
}


#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct ModelMemoryOutcome {
    #[serde(skip_serializing_if = "Option::is_none")] pub estimate: Option<ModelMemoryEstimate>,
    #[serde(skip_serializing_if = "Option::is_none")] pub warning: Option<WarningRecord>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct TrialFailure {
    #[serde(flatten)] pub error: ErrorPayload,
    pub timed_out: bool,
    pub interrupted: bool,
    #[serde(skip_serializing_if = "Option::is_none")] pub stdout_tail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")] pub stderr_tail: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum TrialOutcome {
    Success {
        index: usize,
        config: ExecutableConfig,
        metrics: BenchmarkMetrics,
        correctness: Option<CorrectnessResult>,
        model_memory: ModelMemoryOutcome,
        artifacts: Vec<ArtifactManifest>,
    },
    Failed {
        index: usize,
        config: ExecutableConfig,
        failure: TrialFailure,
        metrics: Option<BenchmarkMetrics>,
        correctness: Option<CorrectnessResult>,
        model_memory: ModelMemoryOutcome,
        artifacts: Vec<ArtifactManifest>,
    },
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct ReportRequest {
    pub run_id: String,
    pub kind: RunKind,
    pub engine: Engine,
    pub winning_metric: Metric,
    pub requested_image: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub(crate) struct RunReport {
    pub schema_version: u32,
    pub run_id: String,
    pub kind: RunKind,
    pub state: RunState,
    pub engine: Engine,
    pub winning_metric: Metric,
    pub requested_image: String,
    pub resolved_image: Option<ResolvedImage>,
    pub selected_hardware: Option<HardwareProfile>,
    pub started_at_unix_ms: u64,
    pub ended_at_unix_ms: Option<u64>,
    pub duration_ms: Option<u64>,
    pub trials: Vec<TrialOutcome>,
    pub best_trial_index: Option<usize>,
    pub best_winning_value: Option<f64>,
    pub best_config_path: Option<PathBuf>,
    pub run_failure: Option<ErrorPayload>,
    pub submission: Option<SubmissionResult>,
}
```

`RunReport::running(request, started_ms)` sets schema 2, state `running`, nullable preflight/final/best/submission fields, and no trials. `set_preflight(resolved, hardware)` records immutable identity and selected devices. `push_trial` requires the next ordered index and recomputes best index/value without reordering. `checkpoint(path)` validates and serializes through `serde_json::to_vec_pretty` plus `atomic_write`. `finish(state, ended_ms, failure)` rejects `running`, requires `ended_ms >= started`, calculates duration, enforces resolved image/hardware for completed states, records failure/interruption context, and recomputes the winner. `set_submission` stores the final leaderboard result before one last checkpoint.

- [ ] **Step 5: Make benchmark parsing strict**

Move the known label table into `metrics.rs`. `BenchmarkMetrics::parse(text, selected)` returns an error when `selected` is absent or non-finite; normalize negative zero; fail a trial when `failed_requests > 0`. Preserve unknown labels in a bounded diagnostic map rather than dropping them silently.

- [ ] **Step 6: Implement artifact manifests and checkpoint order**

Reject absolute/outside-run paths. Strip ANSI, redact registered credentials, then atomically install the sanitized artifact; compute SHA-256 from those installed bytes. Flush/sync artifacts before the report checkpoint. Install `best.conf` before the completed report that references it. Store only relative paths, lengths, hashes, and truncation in report JSON.

- [ ] **Step 7: Run focused result tests**

Run:

```bash
cargo test --lib results::
cargo test --lib runtime::atomic::
```

Expected: PASS.

---

### Task 4: Bounded process runtime, cancellation, Docker identity, and parameter schema

**Files:**
- Replace: `src/runtime/mod.rs`
- Create: `src/runtime/cancel.rs`
- Create: `src/runtime/process.rs`
- Create: `src/runtime/docker.rs`
- Replace: `src/engines/params.rs`
- Test in: `src/runtime/process.rs` `#[cfg(test)]` module so the binary-only crate exposes no test API

**Interfaces:**
- Produces: `CancellationToken`
- Produces: `ProcessSpec`, `CapturePolicy`, `ProcessOutcome`, `ProcessFailure`, `ProcessExecutor`
- Produces: `DockerImageIdentity`, `PullPolicy`, `OwnedContainer`, `resolve_image()`
- Produces: `ParameterSchema`, `ParameterSpec`, `ValueMode`

- [ ] **Step 1: Write failing process lifecycle tests**

Use local shell fixtures only in Unix-gated unit tests inside `src/runtime/process.rs`. Verify:

- normal stdout/stderr capture;
- nonzero exit code and tails;
- 100 ms deadline terminates a sleeping process group;
- a spawned descendant does not survive cancellation;
- output exceeding a 4 KiB test cap is drained, persisted only to the cap, and marked truncated;
- secret capture creates no artifact and redacts errors.

Run:

```bash
cargo test --lib runtime::process::tests
```

Expected: compile failure for missing runtime interfaces.

- [ ] **Step 2: Implement cancellation and process groups**

`CancellationToken` wraps `Arc<AtomicBool>`. Register OS flags once in `main`; tests use an unregistered token. On Unix, call `CommandExt::process_group(0)`, then use safe `nix::sys::signal::killpg` for TERM/KILL. Poll child status/deadline/cancellation at 50 ms intervals. Terminate, wait at most 10 seconds, force-kill, then join drain threads.

Implement the shared specification and constants exactly:

```rust
pub(crate) const DEFAULT_STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
pub(crate) const DEFAULT_CORRECTNESS_TIMEOUT: Duration = Duration::from_secs(600);
pub(crate) const DEFAULT_BENCHMARK_TIMEOUT: Duration = Duration::from_secs(1_800);
pub(crate) const DEFAULT_INSPECTION_TIMEOUT: Duration = Duration::from_secs(300);
pub(crate) const DEFAULT_MAX_PROCESS_OUTPUT_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const DIAGNOSTIC_TAIL_BYTES: usize = 64 * 1024;
pub(crate) const SECRET_CAPTURE_BYTES: usize = 64 * 1024;
pub(crate) const SHUTDOWN_GRACE: Duration = Duration::from_secs(10);

#[derive(Clone, Debug)]
pub(crate) struct ProcessSpec {
    pub program: OsString,
    pub args: Vec<OsString>,
    pub env_add: Vec<(OsString, OsString)>,
    pub env_remove: Vec<OsString>,
    pub cwd: Option<PathBuf>,
    pub deadline: Option<Instant>,
    pub max_stdout_bytes: u64,
    pub max_stderr_bytes: u64,
    pub stdout_artifact: Option<PathBuf>,
    pub stderr_artifact: Option<PathBuf>,
    pub owned_container: Option<OwnedContainer>,
    pub safe_display: String,
    pub capture: CapturePolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CapturePolicy { ArtifactTails, Secret }
```

- [ ] **Step 3: Implement concurrent bounded drains**

Each drain thread reads 8 KiB chunks, writes at most the configured cap through a streaming ANSI stripper and registered-credential redactor, keeps a UTF-8-safe 64 KiB sanitized tail, and drains/discards excess. It flushes/syncs a same-directory temporary artifact and atomically installs it before returning. Checksums are calculated later from these sanitized final bytes. Machine-readable stdout consumers reject `stdout.truncated`; diagnostic stderr truncation remains representable.

Use these result shapes:

```rust
pub(crate) struct StreamOutcome {
    pub artifact: Option<PathBuf>,
    pub tail: String,
    pub observed_bytes: u64,
    pub persisted_bytes: u64,
    pub truncated: bool,
}

pub(crate) struct ArtifactCapture {
    pub stdout: StreamOutcome,
    pub stderr: StreamOutcome,
}

pub(crate) struct SecretOutput(Zeroizing<String>);

impl SecretOutput {
    pub(crate) fn expose(&self) -> &str { &self.0 }
}

impl std::fmt::Debug for SecretOutput {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result { f.write_str("<redacted>") }
}

pub(crate) enum ProcessCapture {
    Artifacts(ArtifactCapture),
    Secret(SecretOutput),
}

pub(crate) struct ProcessOutcome {
    pub status: ExitStatus,
    pub duration: Duration,
    pub capture: ProcessCapture,
    pub cleanup_failure: Option<ErrorPayload>,
}

pub(crate) struct ProcessFailure {
    pub error: Error,
    pub capture: Option<ArtifactCapture>,
    pub cleanup_failure: Option<ErrorPayload>,
}
```

`ProcessExecutor::execute(&self, spec: &ProcessSpec, cancellation: &CancellationToken) -> std::result::Result<ProcessOutcome, ProcessFailure>` is the sole child-process entry point. `CapturePolicy::Secret` keeps at most 64 KiB in memory, never creates an artifact or diagnostic tail, rejects truncation, and returns `ProcessCapture::Secret` only on a successful child exit. Secret failures contain no child output.

- [ ] **Step 4: Implement owned-container cleanup**

`OwnedContainer` includes name, run ID, role, and labels. On timeout/cancellation/early return, execute bounded `docker rm -f NAME`, record cleanup failure, then reap the Docker client. Build `cleanup --dry-run` around `docker ps -a --filter label=optimum-advisor=true`; never use an unfiltered delete.

- [ ] **Step 5: Write failing image/cache tests with fake Docker**

A temporary fake `docker` executable must simulate:

- requested ref resolving to ID plus a matching repo digest;
- unrelated repo digests being ignored;
- no digest rejected unless `allow_local_image`;
- local-only generated identity;
- aliases sharing one immutable cache key;
- `missing`, `always`, and `never` pull behavior.

- [ ] **Step 6: Implement immutable image resolution**

After applying pull policy, run one formatted inspect for image ID and JSON `RepoDigests`. Normalize Docker Hub aliases before repository comparison. Use the matching `repository@sha256:<64 lowercase hexadecimal digits>`; otherwise require explicit local-image permission and use image ID. Every later plan gets the immutable reference.

- [ ] **Step 7: Replace parameter help scraping for execution**

Update embedded vLLM/SGLang Python introspection to emit JSON records with aliases, canonical name, `none|required|optional`, repeatability, and choices. Delete the production `--help` fallback. Typed cache schema includes `schema_version`, engine, immutable image identity, and specs. Validate name plus value mode through a lookup map. Cache writes use `atomic_write` and identity-hash filenames.

- [ ] **Step 8: Run focused runtime and parameter tests**

Run:

```bash
cargo test --lib runtime::
cargo test --lib engines::params::
```

Expected: PASS.

---

### Task 5: Engine plans and durable evaluation/sweeps

**Files:**
- Replace: `src/engines/mod.rs`
- Modify: `src/engines/serve.rs`
- Replace: `src/engines/vllm/mod.rs`
- Replace: `src/engines/sglang/mod.rs`
- Create: `src/evaluation.rs`
- Modify: `src/lib.rs` to declare `evaluation`
- Retain `src/domain/trial.rs` and `src/domain/logs.rs` until Task 8 migrates the CLI/app callers, then delete both in that task

**Interfaces:**
- Produces: narrowed `EngineAdapter`
- Produces: `PreviewPlan`, `RunPlan`, `ReadinessProbe`, `EvaluationContext`
- Produces: `prepare_run(config: NormalizedConfig, executor: &ProcessExecutor, cancellation: &CancellationToken) -> Result<PreparedRun>`
- Produces: `evaluate_candidate(context: &EvaluationContext, config: &ExecutableConfig, index: usize) -> TrialOutcome`
- Produces: `run_sweep_with(configs: Vec<ExecutableConfig>, report: &mut RunReport, report_path: &Path, cancellation: &CancellationToken, checkpoint: impl FnMut(&RunReport) -> Result<()>, evaluate: impl FnMut(usize, &ExecutableConfig) -> TrialOutcome) -> Result<()>`

- [ ] **Step 1: Write failing explicit-override and Docker-plan tests**

Assert SGLang `memory_fraction = 0.73` and `prefill_token_budget = 4096` survive defaults and render as `--mem-fraction-static 0.73` and `--chunked-prefill-size 4096`. Assert vLLM overrides survive. Assert published ports bind the configured IP, IPv6 renders as `[address]:host_port:container_port`, occupied host ports fail before Docker spawn, exact GPU UUIDs are passed, and server/benchmark/inspection containers are named/labeled.

- [ ] **Step 2: Narrow the adapter contract**

Use:

```rust
pub(crate) trait EngineAdapter: Sync {
    fn engine(&self) -> Engine;
    fn default_image(&self) -> &'static str;
    fn default_candidate(&self, metric: Metric) -> Candidate;
    fn introspection_plan(&self, image: &DockerImageIdentity) -> ProcessSpec;
    fn effective_args(&self, config: &ExecutableConfig) -> Vec<EngineArg>;
    fn run_plan(&self, config: &ExecutableConfig, run_id: &str, trial: usize) -> RunPlan;
}
```

Delete `next_candidate`, `Outcome`, `classify_log`, and the prototype advisor path. Engine defaults never read user overrides.

- [ ] **Step 3: Build secure deterministic Docker plans**

Before spawn, bind the validated `SocketAddr` once and fail if occupied; release immediately before Docker executes. Render Docker publish values as `host:port:container_port` for IPv4 and `[host]:port:container_port` for IPv6. Server, benchmark, correctness, and inspection plans use the immutable image, exact selected GPU UUIDs, owned names/labels, optional nonempty `HF_TOKEN` by environment name, and positive short-lived command deadlines. Reserve model/host/port/tensor/memory/scheduler arguments so dynamic args cannot override them.

`ReadinessProbe` targets `/v1/models`, has the positive startup deadline, and requires a 2xx JSON model list containing the configured model while the managed server remains alive. The server child itself has no lifetime deadline: `serve` keeps it managed after readiness until child exit or cancellation; evaluation keeps it managed only while bounded correctness/benchmark children run. Readiness timeout maps to exit 1, OS/user interruption to 130.

- [ ] **Step 4: Write failing failure-tolerant sweep tests**

Use `run_sweep_with` and a closure returning success/failure/success. Assert all three trials are retained, report state is `completed_with_failures`, the winner ignores the failed trial, and checkpoint count increments after each result. Add all-failed and cancellation cases.

Define the evaluation context used by both CLI and MCP:

```rust
pub(crate) struct EvaluationContext<'a> {
    pub executor: &'a ProcessExecutor,
    pub cancellation: &'a CancellationToken,
    pub adapter: &'static dyn EngineAdapter,
    pub resolved_image: &'a ResolvedImage,
    pub hardware: &'a HardwareProfile,
    pub parameter_schema: &'a ParameterSchema,
    pub run_dir: &'a Path,
}
```

- [ ] **Step 5: Implement candidate evaluation**

`prepare_run` creates the initial running report directory/checkpoint, resolves image/hardware/parameter schema, expands and validates every executable config, and returns `PreparedRun` before GPU work. For each prepared candidate, `evaluate_candidate`:

1. evaluates optional model memory with required/optional policy;
2. creates a private trial directory;
3. starts the managed server without a lifetime deadline;
4. enforces the readiness deadline and model-identity response;
5. runs requested correctness and/or benchmark children with their positive deadlines;
6. parses typed results and selected metric;
7. terminates/reaps the server and owned container;
8. installs sanitized artifacts before manifests;
9. returns `TrialOutcome::Success` or a stage-typed `TrialOutcome::Failed`.

Candidate startup, correctness, benchmark, metric-collection, timeout, and OOM failures return `TrialOutcome::Failed`; they do not unwind the sweep and there are no retries. Run-scoped report I/O, config, image, hardware, or parameter-schema failures return `Err` carrying the durable report path and finalize the run report as failed when persistence remains available.

- [ ] **Step 6: Implement durable sequential sweep orchestration**

Create the initial running report before external preflight. Append/checkpoint each outcome. Stop scheduling on cancellation and finalize `interrupted`. Compute a stable winner without reordering history. Serialize `best.conf` from the complete winning `ExecutableConfig`: exact candidate and canonical dynamic serve args; retained runtime, benchmark, correctness, and model-memory policy; no sweep/results path/secret; `leaderboard.submit = false`; repository digest plus `pull_policy = "missing"`, or local image ID plus `allow_local_image = true` and `pull_policy = "never"`. Atomically install `best.conf`, then final report. Return success for `completed_with_failures` with a winner; return an `Error` containing report path for all-failed/interrupted.

- [ ] **Step 7: Run focused engine/evaluation tests**

Run:

```bash
cargo test --lib engines::
cargo test --lib evaluation::
```

Expected: PASS.

---

### Task 6: Correctness and inspection cutover

**Files:**
- Replace: `src/correctness/mod.rs`
- Modify: `src/correctness/suite.rs`
- Create: `src/inspection/mod.rs`
- Move/rewrite: `src/advisor/hardware.rs` → `src/inspection/hardware.rs`
- Move/rewrite: `src/advisor/model_memory.rs` → `src/inspection/model_memory.rs`
- Delete: `src/advisor/mod.rs`

**Interfaces:**
- Produces: `CorrectnessTask { domain, spec, metric }`
- Produces: `parse_unique_json(&str) -> Result<serde_json::Value>` and typed lighteval/capability collection
- Produces: populated selected-device `HardwareProfile`
- Produces: safe `ModelMemoryOutcome` using `ModelMemoryEstimate` or typed warning

- [ ] **Step 1: Write failing exact-metric correctness tests**

Tests must reject missing configured metrics, duplicate JSON keys at any depth, select exact `em`/`prompt_level_strict_acc`, require every task score `>= threshold`, fail capability probes, and never scan JSON text manually.

- [ ] **Step 2: Replace correctness JSON scanners**

Decode every correctness document with `parse_unique_json(text: &str) -> Result<serde_json::Value>`, backed by a recursive custom Serde visitor whose `visit_map` keeps a `HashSet<String>` and returns `de::Error::custom(format!("duplicate JSON key: {key}"))` before inserting a repeated key; `visit_seq` recursively decodes the same wrapper. Navigate that unique-key value into typed results. `CorrectnessTask` names the expected metric. Missing tasks, missing/nonfinite metrics, duplicates, and invalid shapes are `ResultCollection` failures. Enforce the configured timeout through `ProcessSpec`; remove dead `timeout_secs` behavior.

- [ ] **Step 3: Keep generated helper artifacts atomic and bounded**

Run Python response conversion through the managed executor, reject truncated machine stdout, validate generated JSON, strip ANSI/redact credentials, then atomically install it. Artifact records contain relative paths, byte counts, hashes, and truncation rather than JSON bodies.

- [ ] **Step 4: Write failing hardware/device tests**

Test explicit devices over environment, environment over automatic selection, duplicate rejection, index-to-UUID resolution, insufficient devices, and standalone warning behavior.

- [ ] **Step 5: Implement deterministic selected-device hardware**

Parse `nvidia-smi` once into all GPUs, normalize explicit/environment IDs to UUIDs, select first indices only when neither source is present, and return both all detected and selected devices as required by the report contract. Production execution fails insufficient selection; `hardware` can report warnings.

- [ ] **Step 6: Remove implicit `uvx` and hand-written model JSON**

Resolve the executable from explicit config, then `OPTIMUM_ADVISOR_HF_MEM`, then the already-installed `hf-mem` on `PATH`; never invoke `uvx`. If unavailable/failed and optional, return `ModelMemoryOutcome { estimate: None, warning: Some(WarningRecord { kind: ErrorKind::ParameterInspection, stage: ExecutionStage::Preflight, message }) }`; if required, fail preflight. Execute with the 300-second default or configured positive deadline through `ProcessExecutor`, reject truncated stdout, and deserialize `ModelMemoryEstimate` with `serde_json`.

- [ ] **Step 7: Run focused correctness/inspection tests**

Run:

```bash
cargo test --lib correctness::
cargo test --lib inspection::
```

Expected: PASS.

---

### Task 7: Secret-safe native leaderboard client

**Files:**
- Move/rewrite: `src/leaderboard.rs` → `src/leaderboard/mod.rs`
- Create: `src/leaderboard/auth.rs`
- Create: `src/leaderboard/client.rs`

**Interfaces:**
- Produces: `Secret`, `PreparedSubmission`, and report-owned `SubmissionResult`
- Produces: `prepare_submission(config: &LeaderboardConfig, report: &RunReport, executor: &ProcessExecutor, cancellation: &CancellationToken) -> Result<Option<PreparedSubmission>>`
- Produces: `submit_report(prepared: &PreparedSubmission, report_json: &[u8], cancellation: &CancellationToken) -> SubmissionResult`
- Consumes: managed secret process capture, v2 report identity, environment/private token sources, cancellation token

- [ ] **Step 1: Write failing redaction/transport tests**

Use a local TCP HTTP fixture. Assert:

- `Debug` and error rendering never contain token/key values;
- credentialed non-loopback HTTP is rejected;
- loopback HTTP works only in test mode;
- redirects do not forward authorization across origin;
- response/body/SSE limits are enforced;
- cancellation is observed within a body-read slice;
- submit payload uses `serde_json` and accepted/queued/rejected SSE parsing is structural.

- [ ] **Step 2: Implement zeroizing credentials**

`Secret(Zeroizing<String>)` has no `Display`, redacted `Debug`, explicit `expose()` scoped borrow, and zeroizes on drop. Resolve token in this order: nonempty `HF_TOKEN`, nonempty `HUGGING_FACE_HUB_TOKEN`, user-only token files under `HF_HOME`/the platform Hugging Face cache, then bounded secret-output `hf auth token` or `huggingface-cli token`. Reject group/world-readable token files on Unix. Resolve contributor from the token-backed bounded whoami API first, then bounded `hf auth whoami`/`huggingface-cli whoami`. Submit key comes only from `OPTIMUM_ADVISOR_LEADERBOARD_SUBMIT_KEY`.

- [ ] **Step 3: Build the configured ureq agent**

Use rustls with platform verifier. Ordinary requests have 10-second DNS/connect/response-header limits, a 60-second overall deadline, and 5-second receive-body slices; streamed completion has a 300-second overall deadline. Follow at most three same-origin HTTPS redirects and never forward authorization on an origin change. Limit ordinary successful JSON to 1 MiB and non-2xx diagnostics to 64 KiB through bounded readers rather than default `read_to_string()`.

- [ ] **Step 4: Implement bounded Gradio JSON/SSE flow**

Reject submission before credential discovery when `report.resolved_image.local_only == true`. POST report JSON only when ≤16 MiB, parse an event ID from ≤1 MiB JSON, stream structurally parsed SSE with ≤256 KiB per event and ≤1 MiB accumulated data, and stop at the 300-second deadline or cancellation. `submit_report` returns accepted, queued, or failed without exposing credentials. Atomically add that result to the durable report; a failed requested submission preserves winner/artifacts, changes `completed` to `completed_with_failures`, and returns an `Error` carrying report path so CLI exits 1 and MCP emits `isError`.

- [ ] **Step 5: Run focused leaderboard tests**

Run:

```bash
cargo test --lib leaderboard::
```

Expected: PASS.

---

### Task 8: Clap CLI and thin application orchestration

**Files:**
- Replace: `src/cli/mod.rs`
- Delete: `src/cli/config_file.rs`
- Replace: `src/app.rs`
- Update: `tests/cli_smoke.rs`
- Create: `tests/fixtures/bench-v2.toml`
- Create: `tests/fixtures/sweep-v2.toml`
- Modify: `src/terminal.rs`
- Delete after migration: `src/domain/config.rs`
- Delete after migration: `src/domain/trial.rs`
- Delete after migration: `src/domain/logs.rs`
- Remove legacy top-level code from `src/results/mod.rs` while retaining its new module declarations

**Interfaces:**
- Produces: `Cli`, `Command`, `ConfigOverrides`, `AppIo`
- Produces: `app::run(command: Command, io: &mut AppIo, cancellation: &CancellationToken) -> Result<()>`
- Consumes: config normalization, evaluation, report, runtime cleanup
- `AppIo<'a> { stdout: &'a mut dyn Write, stderr: &'a mut dyn Write, stdout_color: bool, stderr_color: bool }`

- [ ] **Step 1: Rewrite smoke tests to the v2 contract and verify they fail**

Cover `--help`/`--version` exit 0, unknown flags exit 2, strict v2 config, precedence, no direct guessed engine flags, dry-run without host tools, retained commands, removed `advise`, stdout/stderr separation, and cleanup dry-run label scoping.

Run:

```bash
cargo test --test cli_smoke
```

Expected: failures against the old parser.

- [ ] **Step 2: Define Clap DTOs**

Use derive subcommands for `plan`, `params`, `hardware`, `serve`, `bench`, `sweep`, `cleanup`, and `mcp`. Flatten one `ConfigOverrides` into applicable commands. Make `sweep --config` required and reject operational args for `mcp`. Let Clap own help/version/usage exit behavior.

Implement these exact command shapes; `ConfigOverrides` contains no secret:

```rust
#[derive(Debug, clap::Parser)]
#[command(name = "optimum-advisor", version, about)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum Command {
    Plan { #[arg(long)] config: Option<PathBuf>, #[command(flatten)] overrides: ConfigOverrides },
    Params {
        #[arg(long)] engine: Engine,
        #[arg(long)] image: Option<String>,
        #[arg(long, default_value = "missing")] pull_policy: PullPolicy,
        #[arg(long)] cache_dir: Option<PathBuf>,
        #[arg(long, conflicts_with = "offline")] refresh: bool,
        #[arg(long)] offline: bool,
    },
    Hardware,
    Serve { #[arg(long)] config: Option<PathBuf>, #[command(flatten)] overrides: ConfigOverrides },
    Bench {
        #[arg(long)] config: Option<PathBuf>,
        #[arg(long)] results_dir: Option<PathBuf>,
        #[arg(long)] dry_run: bool,
        #[command(flatten)] overrides: ConfigOverrides,
    },
    Sweep { #[arg(long)] config: PathBuf, #[arg(long)] results_dir: Option<PathBuf>, #[arg(long)] dry_run: bool },
    Cleanup { #[arg(long)] run_id: Option<String>, #[arg(long)] dry_run: bool },
    Mcp,
}

#[derive(Clone, Debug, Default, clap::Args)]
pub(crate) struct ConfigOverrides {
    #[arg(long)] pub engine: Option<Engine>,
    #[arg(long)] pub image: Option<String>,
    #[arg(long)] pub pull_policy: Option<PullPolicy>,
    #[arg(long)] pub allow_local_image: bool,
    #[arg(long)] pub model: Option<String>,
    #[arg(long)] pub metric: Option<Metric>,
    #[arg(long)] pub gpus: Option<usize>,
    #[arg(long = "gpu-device")] pub gpu_devices: Vec<String>,
    #[arg(long)] pub bind_host: Option<IpAddr>,
    #[arg(long)] pub port: Option<u16>,
    #[arg(long)] pub startup_timeout_secs: Option<u64>,
    #[arg(long)] pub benchmark_timeout_secs: Option<u64>,
    #[arg(long)] pub max_process_output_bytes: Option<u64>,
    #[arg(long)] pub dataset_name: Option<String>,
    #[arg(long)] pub num_prompts: Option<u32>,
    #[arg(long)] pub request_rate: Option<String>,
    #[arg(long)] pub benchmark_max_concurrency: Option<u32>,
    #[arg(long)] pub random_input_len: Option<u32>,
    #[arg(long)] pub random_output_len: Option<u32>,
    #[arg(long)] pub tensor_parallelism: Option<usize>,
    #[arg(long)] pub memory_fraction: Option<f64>,
    #[arg(long)] pub prefill_token_budget: Option<u32>,
    #[arg(long)] pub max_running_requests: Option<u32>,
    #[arg(long)] pub no_correctness: bool,
    #[arg(long)] pub correctness_threshold: Option<f64>,
    #[arg(long)] pub correctness_timeout_secs: Option<u64>,
    #[arg(long)] pub no_model_memory: bool,
    #[arg(long)] pub require_model_memory: bool,
    #[arg(long)] pub hf_mem_command: Option<PathBuf>,
    #[arg(long)] pub hf_mem_timeout_secs: Option<u64>,
    #[arg(long)] pub leaderboard_submit: bool,
    #[arg(long)] pub leaderboard_url: Option<String>,
    #[arg(long = "serve-arg", value_name = "NAME=VALUE")] pub serve_args: Vec<String>,
    #[arg(long = "serve-flag", value_name = "NAME")] pub serve_flags: Vec<String>,
}
```

Create `tests/fixtures/bench-v2.toml` with:

```toml
schema_version = 2
engine = "vllm"
image = "vllm/vllm-openai:latest"
model = "test/model"
metric = "tps"

[runtime]
gpus = 1

[benchmark]
dataset_name = "random"
num_prompts = 2
request_rate = "1"
max_concurrency = 1
random_input_len = 8
random_output_len = 4

[candidate]
tensor_parallelism = 1
memory_fraction = 0.9
prefill_token_budget = 1024
max_running_requests = 8

[correctness]
enabled = false

[model_memory]
enabled = false
```

Create `tests/fixtures/sweep-v2.toml` with the same top-level/runtime/benchmark/correctness/model-memory content and:

```toml
[candidate]
tensor_parallelism = 1
memory_fraction = 0.9
prefill_token_budget = 1024
max_running_requests = 8

[sweep]
memory_fraction = [0.8, 0.9]
max_trials = 256
```

- [ ] **Step 3: Implement source merge and output routing**

Open the optional config exactly once, convert Clap values to source-aware overrides, merge built-ins < TOML < CLI, and normalize. Duplicate fixed options within CLI remain Clap errors; duplicate dynamic CLI names fail before merge, while one canonical CLI dynamic name replaces the TOML value. Progress/errors use stderr; plan/params/hardware data and final report/config paths use stdout.

`plan`, `bench --dry-run`, and `sweep --dry-run` never invoke Docker, Python, `hf-mem`, or network tools. For a repository digest or image ID, they derive the immutable cache key and use only an exact typed cache hit, reporting `validation = "cached"`; a floating reference reports `validation = "pending_runtime"` and never pulls merely to print. `params` resolves/inspects by default; `--refresh` forces resolution and introspection; `--offline` rejects floating references and requires an immutable exact cache hit. A malformed, mismatched, or old cache is an offline error and is refreshed only in online execution.

Construct `AppIo` in `main` using `std::io::IsTerminal`: color may be true only for that specific real terminal and only when `NO_COLOR` is absent. Tests force both flags false. `terminal.rs` never emits ANSI when the corresponding flag is false; report/output-tail sanitizers strip ANSI unconditionally. JSON, TOML, MCP, files, redirected streams, and test buffers therefore remain plain.

- [ ] **Step 4: Shrink app orchestration**

Each command function only coordinates domain services. `serve` performs normalized preflight, starts the managed server, waits through the startup deadline, then waits without a lifetime deadline until server exit or cancellation. `params` follows the cache semantics above. `cleanup` lists/removes only the ownership label and optional exact run ID. `bench`/`sweep` invoke the shared evaluation service and requested leaderboard finalization. Remove config parsing, hand JSON, subprocess calls, report string assembly, candidate heuristics, and duplicated display logic from `app.rs`; delete `src/cli/config_file.rs`, `src/domain/config.rs`, `src/domain/trial.rs`, `src/domain/logs.rs`, and the legacy `src/results/mod.rs` helpers after all callsites move.

- [ ] **Step 5: Run focused CLI smoke tests**

Run:

```bash
cargo test --test cli_smoke
```

Expected: PASS.

---

### Task 9: Strict cancellable MCP and binary-only cutover

**Files:**
- Move/rewrite: `src/mcp.rs` → `src/mcp/mod.rs`
- Create: `src/mcp/protocol.rs`
- Create: `src/mcp/tools.rs`
- Create: `src/mcp/schema.rs`
- Replace: `src/main.rs`
- Delete: `src/lib.rs`
- Delete: `src/tools.rs`
- Create: `tests/mcp_smoke.rs`

**Interfaces:**
- Produces: `mcp::serve_stdio(input: impl BufRead + Send + 'static, output: impl Write, os_cancellation: CancellationToken) -> Result<()>`
- Produces: strict request/tool DTOs, `ToolAnnotations`, and generated schemas
- Consumes: the same config/evaluation/report services as CLI; integration tests invoke only `CARGO_BIN_EXE_optimum-advisor`

- [ ] **Step 1: Write failing MCP lifecycle/framing tests**

Test:

- malformed JSON `-32700`;
- invalid request `-32600`;
- method not found `-32601`;
- invalid envelope params `-32602`;
- tool requests rejected before initialize and before initialized notification;
- one-MiB overflow drained and next frame processed;
- final frame at EOF without newline;
- notifications receive no response;
- cancellation reaches an in-flight fixture tool and server remains ready;
- unknown tool fields become `isError` tool results;
- no ANSI on stdout;
- all schemas use `additionalProperties: false` equivalents generated from DTOs.
- zero timeout/output limits, invalid fractions/thresholds, and out-of-range counts are rejected by server decoding/normalization;
- incompatible protocol version disconnects, repeated initialize is invalid, `ping` works before initialize, and clean EOF returns success.

- [ ] **Step 2: Implement bounded reader and lifecycle state machine**

The reader thread frames with `fill_buf`/`consume` and never allocates beyond 1 MiB; it drains an oversized line through LF and sends one null-ID parse error event. It accepts one final EOF frame and treats empty EOF as clean shutdown. Events go through `std::sync::mpsc`; the dispatcher queues arrival order and executes at most one tool call while the reader recognizes matching `notifications/cancelled`. States are exactly `awaiting_initialize`, `awaiting_initialized_notification`, and `ready`: only `ping` precedes initialize, the server replies with `2025-11-25`, incompatible requested versions send the initialization error then disconnect, repeated initialize is `-32600`, tools wait for `notifications/initialized`, and there is no custom shutdown method. OS cancellation cancels/checkpoints active work and exits; request cancellation returns interrupted and restores `ready`.

- [ ] **Step 3: Implement strict tool DTOs and schemas**

Use `Deserialize + JsonSchema + deny_unknown_fields`. Provide exactly `inspect_hardware`, `inspect_engine`, `validate_config`, `estimate_memory`, `check_correctness`, `run_benchmark`, `evaluate_candidate`, `run_sweep`, and `rank_candidates`. Shared `ConfigInputDto` excludes secrets and sweep for single-candidate tools; only `run_sweep` accepts `SweepInput`. Validate all numeric bounds through the same normalizer and require nonempty unique rank IDs. Generate schemas via `schemars`; do not maintain a second hand-written field list. Attach conservative `ToolAnnotations { read_only_hint, destructive_hint, idempotent_hint, open_world_hint }`: rank/hardware are read-only; validation/inspection may read/write caches or pull images; execution tools create files, processes, and owned containers; any image pull, model access, or leaderboard-capable operation is open-world.

- [ ] **Step 4: Route valid tool failures correctly**

Malformed JSON-RPC envelopes use protocol errors. After a valid known `tools/call`, strict input and execution errors return `{ content, structuredContent, isError: true }`. Clip human content and each diagnostic tail to 64 KiB and return only report summaries/paths rather than embedding reports, keeping a complete tool response under 256 KiB. Include structured report/artifact paths when created. Cancellation leaves MCP ready; OS signal exits.

- [ ] **Step 5: Convert to binary-only crate**

Declare every internal module from `main.rs`, parse `mcp` through Clap, remove `lib.rs` and `src/tools.rs`, and keep integration tests strictly on `CARGO_BIN_EXE_optimum-advisor`. Unit-only runtime tests remain inside the binary target. Ensure MCP stdout is reserved before constructing terminal/log output.

- [ ] **Step 6: Run focused MCP tests**

Run:

```bash
cargo test --test mcp_smoke
```

Expected: PASS.

---

### Task 10: Behavioral smoke gate

**Files:**
- No planned source edits unless a focused check reveals a real defect

**Interfaces:**
- Verifies the complete local, host-tool-free contract before any cleanup/documentation/CI work is planned

- [ ] **Step 1: Run focused module and integration suites together**

Run:

```bash
cargo test --bin optimum-advisor
cargo test --test cli_smoke
cargo test --test mcp_smoke
```

Expected: every focused test PASS.

- [ ] **Step 2: Exercise observable CLI scenarios**

Run exact host-tool-free scenarios against the strict test fixtures:

```bash
cargo run -- --help
cargo run -- plan --engine vllm --model test/model --tensor-parallelism 1
cargo run -- bench --config tests/fixtures/bench-v2.toml --dry-run
cargo run -- sweep --config tests/fixtures/sweep-v2.toml --dry-run
```

Expected: help exits 0; previews use no Docker/Python/HF tool; fixtures are strict schema-v2 documents; redirected output contains no ANSI. Label-scoped cleanup remains covered by the fake-Docker CLI smoke test rather than requiring a host Docker daemon at this gate.

- [ ] **Step 3: Exercise MCP lifecycle and cancellation**

Run `tests/mcp_smoke.rs` with its fake Docker executable: initialize, send `notifications/initialized`, list tools, validate strict schemas, start a deliberately waiting `inspect_engine` introspection child, send `notifications/cancelled`, receive an interrupted tool result, then successfully call `rank_candidates`.

Expected: one-line JSON framing, no ANSI, server remains ready after cancellation.

- [ ] **Step 4: Inspect working-tree preservation**

Run `shasum -a 256 Todos.md` and require `e736408f314a0c5c0647fa542e437552a5bcb023a1f27d7c8fe69da7be4f01b2`. Confirm it remains untracked. Do not commit or push.

- [ ] **Step 5: Open the cleanup gate**

Only after Steps 1–4 pass, append exact final cleanup tasks for project-wide format/Clippy/tests, CI, README/examples, obsolete-file deletion verification, dependency audit, and GPU acceptance documentation. Derive those edits from the behavior that just passed; do not guess them earlier.
