#!/usr/bin/env bash
# Download the Physics-IQ Verified benchmark dataset (README "A. Download
# Physics-IQ Verified"). Run from the physics-IQ-benchmark repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

pip install -U huggingface_hub
hf download Anates-Labs-Research/Physics-IQ-Verified \
  --repo-type dataset \
  --local-dir physics-IQ-benchmark-verified
