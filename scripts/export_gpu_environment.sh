#!/usr/bin/env bash
# Snapshot the host GPU stack after a successful preflight.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
report_path="$project_root/gpu_environment_report.txt"

# Make the CUDA toolkit shipped in the project venv visible to this shell.
# The Python preflight performs the same discovery for its own process, but
# callers source this script so nvcc and extension builders must also inherit it.
cuda_home="$({
  cd "$project_root"
  uv run python -c 'from robotnav.gpu_environment import ensure_cuda_toolkit_environment; print(ensure_cuda_toolkit_environment() or "")'
})"
if [[ -n "$cuda_home" ]]; then
  export CUDA_HOME="$cuda_home"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

{
  cd "$project_root"
  uv run check-gpu-environment
  UV_CACHE_DIR=/tmp/robotnav_uv_cache uv run python -c '
import gsplat
import torch
print()
print("Package versions:")
print("torch=" + torch.__version__)
print("torch.cuda=" + str(torch.version.cuda))
print("gsplat=" + str(getattr(gsplat, "__version__", "unknown")))
'
} > "$report_path"

printf 'Wrote %s\n' "$report_path"
