#!/usr/bin/env sh
set -eu

log() {
  printf '[correctness-env] %s\n' "$*"
}

repo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
venv_dir="$repo_dir/.venv"

command -v uv >/dev/null 2>&1 || {
  log "uv not found; install it first: https://docs.astral.sh/uv/"
  exit 1
}

log "repo: $repo_dir"
log "venv: $venv_dir"

cd "$repo_dir"
log "running: uv venv .venv"
uv venv .venv

VIRTUAL_ENV="$venv_dir"
PATH="$venv_dir/bin:$PATH"
export VIRTUAL_ENV PATH

log 'running: uv pip install lighteval==0.13.0 litellm==1.66.0 diskcache==5.6.3 langdetect==1.0.9'
uv pip install \
  "lighteval==0.13.0" \
  "litellm==1.66.0" \
  "diskcache==5.6.3" \
  "langdetect==1.0.9"

log "verifying imports"
python -c "import diskcache, langdetect, lighteval, litellm"

log "done"
log "activate with: source .venv/bin/activate"
