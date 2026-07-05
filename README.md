# Optimum Advisor

> **Big warning:** this project is still super early and experimental. The API,
> CLI, heuristics, cache format, and engine integrations are expected to change.
> Treat it as a working prototype, not a production optimizer yet.

Optimum Advisor is a Rust CLI for exploring LLM serving configurations. Given a
serving engine, model, hardware shape, target metric, and constraints, the tool
is meant to run candidate serving configurations, benchmark them, and suggest
better configurations.

For now the focus is on vLLM and SGLang.

## What Exists

- Rust CLI backbone with `plan`, `params`, `serve`, `run`, and `advise` modes.
- Engine-specific adapters for vLLM and SGLang.
- Abstract candidate configuration for parallelism, memory budget, and scheduler
  budget, rendered into engine-specific serving flags.
- Runtime parameter introspection from the selected container image.
- Cached parameter schemas under `.optimum-advisor/params`.
- Basic validation for extra serving args against the introspected schema.
- Docker command construction for serving containers, including GPU passthrough.
- Initial log classification for OOM and KV-cache pressure.
- A small sync helper for sending the repo to the GPU machine:
  `scripts/sync-to-gpu.sh`.
- Unit and smoke tests for the current atomic behavior.

## Missing / TODO

- Make SGLang parameter introspection as robust as vLLM's argparse-based path.
- Run benchmarks inside containers or a controlled environment instead of
  assuming benchmark CLIs exist on the host.
- Capture structured benchmark metrics such as TTFT, ITL, TPS, throughput, and
  error rates.
- Persist trials, outcomes, configs, and metrics in a real run history.
- Add a proper search loop over candidates instead of one-step advice.
- Improve engine-specific heuristics for OOM, KV pressure, batching, tensor
  parallelism, pipeline parallelism, and memory utilization.
- Add hardware/model discovery instead of requiring most setup details manually.
- Add constraints to the optimizer, such as latency ceilings or minimum
  throughput.
- Add safer Docker lifecycle handling, cleanup, logs, names, volumes, and port
  management.
- Expand integration tests on an actual CUDA host.

