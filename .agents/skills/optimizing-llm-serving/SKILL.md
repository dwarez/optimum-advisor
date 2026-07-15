---
name: optimizing-llm-serving
description: Use when asked to benchmark, sweep, compare, tune, or select vLLM or SGLang serving configurations with Optimum Advisor, especially when MCP is unavailable and only the CLI can be used.
compatibility: Requires the optimum-advisor CLI. Real execution requires Docker, the NVIDIA container runtime, and a GPU; planning and dry-runs do not.
---

# Optimizing LLM Serving

## Overview

Use Optimum Advisor as a bounded experiment runner. The agent defines and refines candidates; the CLI validates, executes, records, and ranks only candidates it actually observes.

**Core principle:** report the *best observed valid configuration for the declared workload and trial budget*. Never claim a global optimum.

Before running commands, read [references/cli-workflow.md](references/cli-workflow.md). It contains the exact CLI contract and a complete example. Run `optimum-advisor <command> --help` when the installed version and the reference disagree.

## Required Workflow

1. **Define the experiment.** Obtain the engine, model, GPU target, ranking metric, representative input/output lengths, request rate/concurrency, correctness threshold, constraints, maximum trial budget, and minimum meaningful improvement. Do not invent missing SLOs, traffic characteristics, or stop rules.
2. **Draft and validate.** Start from `examples/bench.toml`, `examples/sglang-bench.toml`, or `examples/sweep.toml`. Run `plan`, then `bench --dry-run` or `sweep --dry-run`. These require no Docker, GPU, network, or credentials.
3. **Inspect the execution host.** On the GPU host, run `hardware`, then `params` for the exact engine and image. `params` validates serving-argument names and flag/value modes; it does not generate candidate values or a search domain. Put only validated fixed arguments under `[serve]` and validated valued-argument arrays under `[sweep.serve]`. After any edit, rerun `plan` and the applicable execution `--dry-run` before real execution.
4. **Establish the incumbent.** Run one `bench`. `bench` owns server startup, readiness, correctness, benchmarking, persistence, and cleanup; do not compose a detached `serve` plus `bench` workflow. Seed the incumbent only from an eligible baseline result.
5. **Search in bounded stages.** Put explicit arrays under `[sweep]`, set `max_trials`, and vary one or two high-impact dimensions first. After each `sweep`, inspect every trial in `report.json` and compare all eligible results across the baseline and every preserved report using the configured metric direction. Keep the overall incumbent even when a later stage regresses. Create the next bounded sweep around that incumbent and do not repeat observed candidates without a confirmation reason.
6. **Respect validity.** A correctness failure, timeout, interruption, missing/non-finite winning metric, or failed trial cannot win. Keep correctness enabled unless the user explicitly changes the experiment contract.
7. **Confirm the incumbent.** Run `bench --config <incumbent-config-path>` under the same workload. The best observed candidate remains provisional until this confirmation succeeds.
8. **Report evidence.** Include workload, hardware, resolved image, metric, attempted/succeeded/failed trials, budget, best value, incumbent configuration path, every report path, confirmation result, and remaining uncertainty.

## Quick Reference

| Need | Command |
|---|---|
| Validate/render | `optimum-advisor plan --config <file>` |
| Safe execution preview | `optimum-advisor bench --config <file> --dry-run` or `optimum-advisor sweep --config <file> --dry-run` |
| Inspect GPUs | `optimum-advisor hardware` |
| Validate image arguments | `optimum-advisor params --engine <engine> --image <image> --refresh` |
| Evaluate one candidate | `optimum-advisor bench --config <file> --results-dir <dir>` |
| Evaluate explicit candidates | `optimum-advisor sweep --config <file> --results-dir <dir>` |
| Inspect possible cleanup | `optimum-advisor cleanup --run-id <id> --dry-run` |

## Red Flags

Stop and correct the workflow if it invents CLI flags, treats `params` as a value recommender, omits `max_trials`, launches concurrent trials, scrapes progress instead of reading `report.json`, selects a failed row, silently disables correctness, runs destructive cleanup without `--dry-run`, or says “optimal” without the workload and observed-search qualifier.

## Common Mistakes

| Mistake | Correction |
|---|---|
| “I will fix the option names after checking help” | Check help first; never present hypothetical commands as runnable. |
| `params` defines an exhaustive value domain | It reports names and modes only; propose explicit bounded values from workload evidence. |
| `plan.json`, `--plan`, `--output`, or `--strategy` | These interfaces do not exist; use schema-v2 TOML and current `--help`. |
| Separate `serve` and `bench` for each trial | Use `bench` or `sweep`; they own the complete lifecycle. |
| Huge Cartesian product | Use staged bounded sweeps and refine from durable evidence. |
| Fastest numeric row wins | Only successful, correctness-valid trials with finite selected metrics are eligible. |
| Laptop has no GPU, so nothing is testable | Run `plan` and execution `--dry-run`; defer hardware, image inspection, and execution. |
