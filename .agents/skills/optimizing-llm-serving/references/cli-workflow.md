# Optimum Advisor CLI Workflow

This reference describes the current CLI. Do not substitute imagined plan files, detached servers, JSON output flags, search strategies, or resume controls.

## Command Contracts

```text
plan      Validate one configuration and print exact server/benchmark commands
params    Resolve an image and inspect/cache legal serving argument names and modes
hardware  Inspect locally selected NVIDIA GPUs
serve     Run one serving container until exit or interruption
bench     Own and evaluate one server/correctness/benchmark lifecycle
sweep     Own and sequentially evaluate the explicit bounded TOML sweep
cleanup   List or remove only Optimum Advisor-owned containers
mcp       Serve JSON-RPC; not needed for this workflow
```

`plan`, `bench --dry-run`, and `sweep --dry-run` do not pull images, inspect hardware, start containers, use the network, or require credentials.

`bench` accepts a schema-v2 file plus explicit CLI overrides. `sweep` requires a schema-v2 file containing a nonempty `[sweep]`. Configuration precedence is TOML followed by explicit CLI overrides. Arbitrary environment variables do not override experiment settings.

Fixed engine-specific arguments belong under `[serve]`; valued engine-specific sweep arrays belong under `[sweep.serve]`. Add them only after `params` validates each canonical name and mode. In `[serve]`, `name = true` emits a validated flag; omit the key to leave that flag off, and never encode absence as `false`. To compare a flag off versus on, use two separately planned and dry-run bounded configurations—one omitting the flag and one setting it to `true`—and count both evaluations against the overall budget. Scalar values under `[serve]` and arrays under `[sweep.serve]` represent valued arguments.

Private-model credentials come from supported authentication discovery such as `HF_TOKEN` or `hf auth login`. Never put tokens or submit keys in TOML, CLI arguments, result paths, or copied output.

## Complete Bounded Example

This is a syntax example, not a universal recommendation. Replace the model, image, workload, quality floor, candidate values, and budget with the declared experiment contract.

```toml
schema_version = 2
engine = "vllm"
image = "vllm/vllm-openai:latest"
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
num_prompts = 128
request_rate = "inf"
max_concurrency = 32
random_input_len = 1024
random_output_len = 256

[candidate]
tensor_parallelism = 1
memory_fraction = 0.90
prefill_token_budget = 8192
max_running_requests = 32

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

[sweep]
max_trials = 6
memory_fraction = [0.85, 0.90, 0.94]
max_running_requests = [16, 32]
```

Save it as `/tmp/optimum-advisor-sweep.toml`, then pass all local gates:

```bash
optimum-advisor plan --config /tmp/optimum-advisor-sweep.toml
optimum-advisor bench --config /tmp/optimum-advisor-sweep.toml --dry-run
optimum-advisor sweep --config /tmp/optimum-advisor-sweep.toml --dry-run
```

On the intended GPU host, inspect the hardware and exact image:

```bash
optimum-advisor hardware
optimum-advisor params \
  --engine vllm \
  --image vllm/vllm-openai:latest \
  --cache-dir .optimum-advisor/params \
  --refresh
```

After applying any `params`-validated argument changes, rerun `plan`, `bench --dry-run`, and `sweep --dry-run`. Then establish the baseline from the file's fixed `[candidate]`:

```bash
optimum-advisor bench \
  --config /tmp/optimum-advisor-sweep.toml \
  --results-dir .optimum-advisor/baselines
```

Inspect the printed baseline report. Continue only if it contains a successful, correctness-valid result with a finite selected metric. Then run the sweep into a distinct results root:

```bash
optimum-advisor sweep \
  --config /tmp/optimum-advisor-sweep.toml \
  --results-dir .optimum-advisor/agent-runs
```

Successful execution prints:

```text
report: <run-directory>/report.json
winning_config: <run-directory>/best.toml
```

Use those printed paths. Do not guess the timestamped run directory.

## Report Interpretation

`report.json` is the durable schema-v2 source of truth. Read at least:

```text
schema_version
run_id
state
engine
winning_metric
resolved_image
selected_hardware
trials
best_trial_index
best_winning_value
best_config_path
run_failure
```

For every trial, inspect:

```text
status
config
metrics
correctness
model_memory
failure
artifacts
```

Terminal run states are `completed`, `completed_with_failures`, `failed`, and `interrupted`. `completed_with_failures` may still have a valid winner. A failed trial's partial metric never overrides eligibility. Preserve the report even when the command exits nonzero.

## Refinement Loop

1. Record candidate fingerprints from the baseline and all prior `trials[].config` values.
2. Hold the workload and objective fixed.
3. Maintain one incumbent across the baseline and every preserved report by comparing only eligible selected metrics in the configured metric direction. A later run-local `best.toml` does not replace the incumbent when it regresses.
4. Select a small neighborhood around that incumbent.
5. Change one or two dimensions; keep the Cartesian product within `max_trials`.
6. Dry-run the new sweep before GPU execution.
7. Run it into a new results root; never overwrite prior evidence.
8. Stop when the budget is exhausted, no meaningful improvement remains, or failures show a feasibility boundary.

Optimum Advisor does not calculate statistical significance or a global search optimum. If repeated measurements are required, run explicit confirmation benches and report their individual evidence rather than inventing aggregation fields.

## Final Confirmation

Use the exact incumbent configuration path recorded by its baseline or sweep report:

```bash
optimum-advisor plan --config <incumbent-config-path>
optimum-advisor bench \
  --config <incumbent-config-path> \
  --results-dir .optimum-advisor/confirmations
```

The confirmation must use the same model, workload, correctness policy, hardware selection, and ranking metric. If it fails or materially regresses, report the disagreement; do not promote the incumbent as confirmed.

## Cleanup

Normal evaluation owns its containers and cleans them up. After interruption or suspected leakage, inspect first:

```bash
optimum-advisor cleanup --run-id <run-id> --dry-run
```

Remove only that owned run after reviewing the list:

```bash
optimum-advisor cleanup --run-id <run-id>
```

Never use generic Docker pruning as part of this workflow.
