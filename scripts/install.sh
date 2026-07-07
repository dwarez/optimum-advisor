#!/usr/bin/env sh
set -eu

log() {
  printf '[install] %s\n' "$*"
}

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
install_root=${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}
bin_dir="$install_root/bin"

log "repo: $repo_dir"
log "cargo: $(command -v cargo)"
log "target bin dir: $bin_dir"
log "running: cargo install --path . --locked --force"

cd "$repo_dir"
cargo install --path . --locked --force

log "installed: $bin_dir/optimum-advisor"
if command -v optimum-advisor >/dev/null 2>&1; then
  log "available on PATH: $(command -v optimum-advisor)"
else
  log "not on PATH yet; add $bin_dir"
fi
