# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6](https://github.com/dwarez/optimum-advisor/compare/v0.1.5...v0.1.6) - 2026-07-21

### Added

- improving mcp server capabilities for true agentic usage ([#34](https://github.com/dwarez/optimum-advisor/pull/34))
- improved and tested mcp server ([#31](https://github.com/dwarez/optimum-advisor/pull/31))

### Documentation

- readme and contributing ([#30](https://github.com/dwarez/optimum-advisor/pull/30))

### Added

- model-aware metric defaults: `tpot` for model IDs up to 3B and `tps` otherwise
- actionable MCP schema guidance for normalized serving fields

### Fixed

- preserve parsed benchmark metrics when the selected objective is unavailable
  and report the finite objectives emitted by the engine
- return captured server-startup failures through MCP instead of terminating the
  MCP process and leaving reports in `running`

## [0.1.5](https://github.com/dwarez/optimum-advisor/compare/v0.1.4...v0.1.5) - 2026-07-16

### Added

- logging release version

## [0.1.4](https://github.com/dwarez/optimum-advisor/compare/v0.1.3...v0.1.4) - 2026-07-16

### Fixed

- change example correctness thresholds

## [0.1.3](https://github.com/dwarez/optimum-advisor/compare/v0.1.2...v0.1.3) - 2026-07-16

### Fixed

- git on path for hf jobs during lighteval subprocess ([#26](https://github.com/dwarez/optimum-advisor/pull/26))

## [0.1.2](https://github.com/dwarez/optimum-advisor/compare/v0.1.1...v0.1.2) - 2026-07-16

### Fixed

- manual release flow

### Other

- Macos release support ([#24](https://github.com/dwarez/optimum-advisor/pull/24))

## [0.1.1](https://github.com/dwarez/optimum-advisor/compare/v0.1.0...v0.1.1) - 2026-07-16

### Added

- release support for macos ([#23](https://github.com/dwarez/optimum-advisor/pull/23))

### Changed

- better readme ([#21](https://github.com/dwarez/optimum-advisor/pull/21))

## [0.1.0](https://github.com/dwarez/optimum-advisor/releases/tag/v0.1.0) - 2026-07-15

### Added

- auto release with release-plz ([#17](https://github.com/dwarez/optimum-advisor/pull/17))
- execution paths for dockerized and in-container ([#16](https://github.com/dwarez/optimum-advisor/pull/16))
- ci for binary release ([#15](https://github.com/dwarez/optimum-advisor/pull/15))
- optimum-advisor skill ([#14](https://github.com/dwarez/optimum-advisor/pull/14))
- correctness support for parsing think and tc ([#11](https://github.com/dwarez/optimum-advisor/pull/11))
- sglang introspection, bench example ([#9](https://github.com/dwarez/optimum-advisor/pull/9))
- sgalng benchamrk tool integration
- vllm bench integration
- repository init, core logic, readme, setup

### Changed

- decoupling python code in external py srcs ([#13](https://github.com/dwarez/optimum-advisor/pull/13))
- production-harden configuration, execution, reporting, and MCP
- large refactor of internals to enable MCP server ([#10](https://github.com/dwarez/optimum-advisor/pull/10))

### Fixed

- parameters in config are now correctly built dynamically

### Other

- Adding support to leaderboard submissions directly from the tool ([#7](https://github.com/dwarez/optimum-advisor/pull/7))
- Adding Correctness checks before/after running a benchmark or sweep ([#8](https://github.com/dwarez/optimum-advisor/pull/8))
- Installer script for binary ([#6](https://github.com/dwarez/optimum-advisor/pull/6))
- Backbone for advisor's memory budget capability ([#5](https://github.com/dwarez/optimum-advisor/pull/5))
- sweep support, more metrics, better logging ([#4](https://github.com/dwarez/optimum-advisor/pull/4))
