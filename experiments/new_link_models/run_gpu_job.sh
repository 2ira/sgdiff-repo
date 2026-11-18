#!/usr/bin/env bash
# Local/ACP job payload. Queue submission is handled by the user's ACP launcher.
set -euo pipefail

: "${NEW_LINK_DATA_ROOT:?Set NEW_LINK_DATA_ROOT to the prepared Planetoid directory}"
: "${NEW_LINK_SOURCE_ROOT:?Set NEW_LINK_SOURCE_ROOT to the pinned author checkouts}"
: "${NEW_LINK_OUTPUT_ROOT:?Set NEW_LINK_OUTPUT_ROOT to the shared experiment output}"

export DGLBACKEND="${DGLBACKEND:-pytorch}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"

NEW_LINK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$NEW_LINK_REPO_ROOT"
python -m experiments.new_link_models.sources --root "$NEW_LINK_SOURCE_ROOT" --verify-only
python -c 'import torch; assert torch.cuda.is_available(), "A CUDA GPU is required"; print(torch.cuda.get_device_properties(0))'
python -m experiments.new_link_models.suite \
  --data-root "$NEW_LINK_DATA_ROOT" --source-root "$NEW_LINK_SOURCE_ROOT" \
  --output "$NEW_LINK_OUTPUT_ROOT" --device cuda --jobs "${NEW_LINK_JOBS:-1}" "$@"
