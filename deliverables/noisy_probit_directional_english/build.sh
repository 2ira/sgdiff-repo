#!/usr/bin/env bash
set -euo pipefail
artifact_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$artifact_dir/../.." && pwd)"
cd "$repo_dir"
"${PYTHON_BIN:-python3}" -m experiments.noisy_probit.directional_channel.export
cd "$artifact_dir"
"${TECTONIC_BIN:-tectonic}" -k --keep-logs slides.tex
"${TECTONIC_BIN:-tectonic}" -k --keep-logs report.tex
