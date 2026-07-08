#!/usr/bin/env sh
set -eu

log() {
  printf '[python-env] %s\n' "$*"
}

usage() {
  printf 'usage: %s [vllm|sglang|all]\n' "$0" >&2
}

engine=${1:-vllm}
if [ "$engine" != "vllm" ] && [ "$engine" != "sglang" ] && [ "$engine" != "all" ]; then
  usage
  exit 2
fi

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
venv_dir="$repo_dir/.venv"

command -v uv >/dev/null 2>&1 || {
  log "uv not found; install it first: https://docs.astral.sh/uv/"
  exit 1
}

log "repo: $repo_dir"
log "venv: $venv_dir"
log "engine deps: $engine"
log "running: uv venv .venv"

cd "$repo_dir"
uv venv .venv

VIRTUAL_ENV="$venv_dir"
PATH="$venv_dir/bin:$PATH"
export VIRTUAL_ENV PATH

install_vllm() {
  uv pip install \
    "lighteval[vllm,extended_tasks]==0.13.0" \
    "vllm==0.10.1.1" \
    "transformers==4.56.2" \
    "tokenizers==0.22.1"
}

install_sglang() {
  uv pip install \
    "lighteval[extended_tasks]==0.13.0" \
    "sglang==0.5.14"
}

case "$engine" in
  vllm)
    log 'running: uv pip install lighteval[vllm,extended_tasks]==0.13.0 vllm==0.10.1.1 transformers==4.56.2 tokenizers==0.22.1'
    install_vllm
    ;;
  sglang)
    log 'running: uv pip install lighteval[extended_tasks]==0.13.0 sglang==0.5.14'
    install_sglang
    ;;
  all)
    log 'running: vllm and sglang pinned installs'
    install_vllm
    install_sglang
    ;;
esac

log "done"
log "activate with: source .venv/bin/activate"
