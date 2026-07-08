# Optimum Advisor

> **Early experimental project.** The CLI, config format, search logic, and
> engine integrations are still changing. Treat this as a prototype for
> benchmarking serving setups, not a production optimizer.

Optimum Advisor is a Rust CLI for testing LLM serving configurations. It starts
serving-engine containers, runs engine-native benchmarks, records a structured
report, and helps compare configurations by metrics such as throughput, TTFT,
TPOT, ITL, and E2E latency.

Current engine focus: **vLLM** and **SGLang**.

## Install

Local install from this checkout:

```bash
./scripts/install.sh
```

Equivalent raw command:

```bash
cargo install --path . --locked --force
```

Install from git:

```bash
cargo install --git https://github.com/dwarez/optimum-advisor.git --locked --force
```

Uninstall:

```bash
cargo uninstall optimum-advisor
```

## Requirements

- Rust toolchain with `cargo`
- Docker
- NVIDIA runtime for GPU execution (`docker run --gpus ...`)
- `HF_TOKEN` for benchmark execution
- correctness env: `./scripts/setup-correctness-env.sh`
- Optional: `uvx hf-mem` or `hf-mem` for model-memory estimates in reports

## Quick Start

Dry-run a single benchmark command:

```bash
cargo run -- bench --config examples/bench.conf --dry-run
```

Run a sweep on a GPU host:

```bash
export HF_TOKEN=hf_...
optimum-advisor params --engine vllm --image vllm/vllm-openai:latest --execute --refresh-params
optimum-advisor sweep --config examples/sweep.conf
```

Inspect the result:

```bash
cat .optimum-advisor/results/<sweep-dir>/report.json
optimum-advisor bench --config .optimum-advisor/results/<sweep-dir>/best.conf
```

## Config Files

Use config files for real runs. CLI flags are useful for quick checks, but
config files are easier to repeat and commit.

Single benchmark configs use `[serve]`:

```text
engine = vllm
model = Qwen/Qwen3-4B-Instruct-2507
gpus = 1
max_model_len = 8192
metric = tps

[benchmark]
num_prompts = 4
request_rate = 1
max_concurrency = 1

[serve]
gpu-memory-utilization = 0.90
max-num-batched-tokens = 8192
```

Sweep configs add `[sweep]`. Values in `[sweep]` are real engine serving
parameters and are validated against the selected engine image at execution
time.

```text
engine = vllm
model = Qwen/Qwen3-4B-Instruct-2507
gpus = 2
max_model_len = 8192
metric = tps

[benchmark]
num_prompts = 4
request_rate = 1
max_concurrency = 1

[sweep]
tensor-parallel-size = 1,2
gpu-memory-utilization = 0.80,0.90
max-num-batched-tokens = 4096,8192
```

## Commands

```bash
optimum-advisor plan --engine vllm --model Qwen/Qwen3-4B-Instruct-2507
optimum-advisor params --engine vllm --image vllm/vllm-openai:latest --execute
optimum-advisor bench --config examples/bench.conf
optimum-advisor sweep --config examples/sweep.conf
optimum-advisor hardware
```

`bench --dry-run` prints one server/benchmark pair. `sweep --dry-run` prints one
pair per candidate without starting containers. Dry-runs also show the owned
lighteval endpoint correctness suite that runs against the same server.

## Results

Each execution writes a directory under `.optimum-advisor/results` unless
`--results-dir` is set.

Main artifact:

- `report.json`: source of truth with hardware, model-memory estimate, tested
  configs, benchmark metrics, correctness results, stdout, stderr, winning
  metric, and best trial

Convenience artifacts:

- `config.conf`: runnable config produced by `bench`
- `best.conf`: runnable best config produced by `sweep`
- `correctness/responses.json`: per-sample correctness prompts, responses, and metrics when correctness details are available

## Current Scope

Implemented:

- vLLM serving and `vllm bench serve`
- SGLang serving and `sglang.bench_serving`
- runtime serving-parameter introspection from container images
- Docker lifecycle cleanup for owned server containers
- hardware detection through `nvidia-smi`
- optional model-memory estimation through `hf-mem`
- owned lighteval endpoint correctness suite captured in `report.json`
- structured benchmark reports and basic best-result selection

Still missing:

- failure-tolerant sweeps that record bad/OOM candidates instead of aborting
- baseline-vs-candidate correctness degradation scoring
- advisor heuristics using hardware and model-memory budgets
- richer constraints such as latency ceilings and minimum throughput
- stronger SGLang parameter introspection
- broader CUDA-host integration coverage
