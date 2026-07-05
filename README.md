# Optimum Advisor

> **Big warning:** this project is still super early and experimental. The API,
> CLI, heuristics, cache format, and engine integrations are expected to change.
> Treat it as a working prototype, not a production optimizer yet.

Optimum Advisor is a Rust CLI for exploring LLM serving configurations. Given a
serving engine, model, hardware shape, target metric, and constraints, the tool
is meant to run candidate serving configurations, benchmark them, and suggest
better configurations.

For now the focus is on vLLM and SGLang.

## Configuration

A configuration is the complete executable setup for one serving attempt. It is
not just the serving engine flags.

For now it contains:

- engine name, such as `vllm` or `sglang`
- container image
- model id or model path
- GPU count
- `HF_TOKEN` from the environment for execution; the token is forwarded into
  serving and benchmark containers without printing its value
- max model length, defaulting to `8192` for now so long-context models do not
  force huge KV-cache allocation during smoke runs
- host, port, and startup timeout
- optimization metric, such as `tps`, `total_tps`, `req_s`, `ttft`,
  `p99_ttft`, `tpot`, `p99_tpot`, `itl`, `p99_itl`, `e2e`, or `p99_e2e`
- abstract candidate knobs: tensor/pipeline/data parallelism, memory fraction,
  prefill token budget, and max running requests
- extra engine-specific serving args from `--serve-arg` and `--serve-flag`
- direct engine-specific flags, such as `--kv-cache-dtype fp8`; unknown
  `--long-flags` are forwarded to the selected engine and validated against
  the introspected schema when executing
- benchmark settings: dataset name, number of prompts, request rate, and max
  concurrency, plus random input/output lengths for synthetic benchmark data
- result settings: winning metric from `--metric`; each execution writes a
  subdirectory under `.optimum-advisor/results` unless `--results-dir` is set
- engine-specific serving args from `[serve]`; sweep configs additionally use a
  `[sweep]` section

The engine adapter turns that configuration into concrete commands. For example,
vLLM maps the abstract candidate to `--tensor-parallel-size`,
`--gpu-memory-utilization`, `--max-model-len`, and
`--max-num-batched-tokens`, then runs `vllm bench serve` from inside the same
vLLM image. SGLang maps the same abstract candidate to `--tp-size`,
`--mem-fraction-static`, `--chunked-prefill-size`, and
`--max-running-requests`, then runs `python3 -m sglang.bench_serving` from
inside the same SGLang image.

## What Exists

- Rust CLI backbone with `plan`, `params`, `serve`, `bench`, `sweep`, and
  `advise` modes.
- First-class executable serving configuration in code.
- Engine-specific adapter folders for vLLM and SGLang under `src/engines/`.
- Abstract candidate configuration for parallelism, memory budget, and scheduler
  budget, rendered into engine-specific serving flags.
- Full configs through `--config`: `bench` consumes one exact config, while
  `sweep` consumes configs with engine-specific serving-parameter sweeps.
- Runtime parameter introspection from the selected container image.
- Cached parameter schemas under `.optimum-advisor/params`.
- Basic validation for extra serving args against the introspected schema.
- Docker command construction for serving containers, including GPU passthrough.
- Serving containers are named/labeled per CLI process and cleaned up after
  `serve --execute`, `bench`, or `sweep` finish.
- vLLM benchmark invocation through `vllm bench serve` inside the selected vLLM
  image.
- SGLang benchmark invocation through `python3 -m sglang.bench_serving`
  inside the selected SGLang image.
- Benchmark result capture with raw output, one-row TSV summaries, and
  `best.conf` for the winning config in each result subdirectory.
- Initial log classification for OOM and KV-cache pressure.
- A small sync helper for sending the repo to the GPU machine:
  `scripts/sync-to-gpu.sh`.
- Unit and smoke tests for the current atomic behavior.

## Quick Checks

Local checks that do not need a GPU:

```bash
cargo test
cargo run -- plan --engine vllm --model Qwen/Qwen3-4B-Instruct-2507 --max-model-len 8192 --num-prompts 4 --request-rate 1 --benchmark-max-concurrency 1
cargo run -- bench --config examples/bench.conf --dry-run
cargo run -- bench --engine vllm --model Qwen/Qwen3-4B-Instruct-2507 --kv-cache-dtype fp8 --dry-run
cargo run -- bench --engine sglang --model Qwen/Qwen3-4B-Instruct-2507 --num-prompts 4 --request-rate 1 --benchmark-max-concurrency 1 --random-output-len 32 --dry-run
cargo run -- sweep --config examples/sweep.conf --dry-run
```

`bench --dry-run` prints one server/benchmark pair. `sweep --dry-run` prints one
pair per candidate.

Config files can replace the CLI args. A single-benchmark config has `[serve]` and no
`[sweep]`; a sweep config adds `[sweep]`. Values in `[sweep]` are real engine
serving parameters and are validated against the selected engine image when
executing:

```text
engine = vllm
model = Qwen/Qwen3-4B-Instruct-2507
gpus = 2
metric = tps

[benchmark]
num_prompts = 4
request_rate = 1

[sweep]
tensor-parallel-size = 1,2
gpu-memory-utilization = 0.80,0.90
```

GPU smoke benchmark:

```bash
export HF_TOKEN=hf_...
cargo run -- params --engine vllm --image vllm/vllm-openai:latest --execute --refresh-params
cargo run -- sweep --config examples/sweep.conf
cargo run -- bench --config .optimum-advisor/results/<sweep-dir>/best.conf
cargo run -- bench --engine sglang --model Qwen/Qwen3-4B-Instruct-2507 --gpus 1 --num-prompts 4 --request-rate 1 --benchmark-max-concurrency 1 --random-output-len 32
```

## Missing / TODO

- Make SGLang parameter introspection as robust as vLLM's argparse-based path.
- Expand structured benchmark metrics and engine-specific parsers beyond the
  current common throughput/latency summary fields.
- Persist trials, outcomes, configs, and metrics in a real execution history.
- Make candidate search failure-tolerant so OOM/bad candidates are recorded and
  skipped instead of aborting the whole sweep.
- Improve engine-specific heuristics for OOM, KV pressure, batching, tensor
  parallelism, pipeline parallelism, and memory utilization.
- Add hardware/model discovery instead of requiring most setup details manually.
- Add constraints to the optimizer, such as latency ceilings or minimum
  throughput.
- Expand Docker lifecycle handling with better logs, volumes, automatic port
  selection, and CUDA-host integration coverage.
- Expand integration tests on an actual CUDA host.
