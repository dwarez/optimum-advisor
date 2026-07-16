# Contributing

## Development gates

Every change must pass:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
cargo build --release --all-features --locked
cargo +1.85.0 test --all-targets --all-features --locked   # declared MSRV
sh -n scripts/install.sh scripts/setup-correctness-env.sh
```

CI additionally runs RustSec `cargo audit`, ShellCheck, and a static
`x86_64-unknown-linux-musl` release build (the release-critical target).

## Real GPU-host acceptance

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

## Releases

Releases are fully automated with [release-plz](https://release-plz.dev) in
git-only mode (the crate is not published to crates.io):

1. A push to `main` containing a release-worthy commit opens or updates a
   release PR that bumps the version and `CHANGELOG.md` from the commits since
   the last tag.
2. Merging that PR makes CI create the `v<version>` git tag and the GitHub
   release, then dispatches the workflow that builds and attaches the prebuilt
   binaries (Linux x86_64 musl, macOS arm64, macOS x86_64) with `.sha256`
   checksums.

### Commit types drive everything

| Squash-commit type | Effect |
| --- | --- |
| `feat:` | release PR; minor bump (also on `0.x`) |
| `fix:`, `add:`, `remove:`, `refactor:`, `perf:` | release PR; patch bump |
| `!` / `BREAKING CHANGE` | breaking release |
| `change:`, `doc:`, `chore:`, `ci:`, `test:`, untyped | no release PR; rides along in the next one |

Notes:

- The gate (`release_commits` in `release-plz.toml`) only sees commits that
  touch files — an **empty commit cannot force a release**.
- To cut a release when no release-worthy commit is pending:
  `gh workflow run release-plz.yml -f force=true`, then merge the release PR it
  opens.
- Manually pushed `v*` tags trigger the binary build directly, and
  `gh workflow run release.yml -f tag=v<version>` (re)attaches binaries to an
  existing release.

### Version coupling with `--on hf-jobs`

The default in-job binary URL is pinned at compile time to
`releases/download/v{CARGO_PKG_VERSION}/…`, so the submitting CLI and the
in-job binary always agree on flags and config schema. Consequences:

- Every release tag must ship the binary assets (the automation above does).
- A fix that changes in-job behavior only reaches jobs after it is released
  *and* the submitting CLI is rebuilt/updated to that version.

### Config gotchas (learned the hard way)

- `Cargo.toml` must **not** set `publish = false`: release-plz's `release`
  command silently skips unpublishable packages, even in git-only mode.
- `git_only = true` does **not** imply `publish = false` in `release-plz.toml`;
  both keys are required or `release` attempts a real `cargo publish`.

### One-time repository setting

Enable "Allow GitHub Actions to create and approve pull requests"
(Settings → Actions → General) so the release PR can be opened with the default
`GITHUB_TOKEN`. Release PRs opened with that token do not trigger PR CI; the
gates run when the merge lands on `main`.
