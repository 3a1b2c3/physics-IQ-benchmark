#!/usr/bin/env bash
# Stage Zing-0.5 generations into Physics-IQ Verified, then evaluate.
#
# Same shape as MIND's scripts/drive_<model>.sh: this wrapper resolves paths and
# checks preconditions, physiq/drive_zing.py does the work, and every extra
# argument is forwarded through.
#
#   bash scripts/drive_zing.sh --run-name zing-0.5-op-run_01 --limit 1
#   bash scripts/drive_zing.sh --run-name zing-0.5-op-run_01
#   bash scripts/drive_zing.sh --run-name zing-0.5-op-run_02 --seed 1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

ZING_ROOT="${ZING_ROOT:-$(cd "$HERE/.." && pwd)/zing-world-model}"
DRIVER="$HERE/physiq/drive_zing.py"

# Zing runs in its own venv -- zing_v0_5 is not importable from physiq's env.
# The driver launches that interpreter itself; this only needs a python able to
# read a CSV and write JSONL, so prefer physiq's venv and fall back to python3.
if [ -x "$HERE/.venv/bin/python" ]; then
  PY="$HERE/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

if [ ! -f "$DRIVER" ]; then
  echo "ERROR: driver not found: $DRIVER" >&2
  exit 2
fi
if [ ! -d "$ZING_ROOT" ]; then
  echo "ERROR: zing-world-model checkout not found: $ZING_ROOT" >&2
  echo "       set ZING_ROOT=/path/to/zing-world-model" >&2
  exit 2
fi
if [ ! -d "$HERE/physics-IQ-benchmark-verified/switch-frames" ]; then
  echo "ERROR: benchmark data missing: $HERE/physics-IQ-benchmark-verified/switch-frames" >&2
  echo "       run bash download_verified_data.sh" >&2
  exit 2
fi

echo "============================================================"
echo "Zing-0.5 staging into Physics-IQ Verified"
echo "============================================================"
echo "  benchmark : $HERE"
echo "  zing      : $ZING_ROOT"
echo "  driver    : $DRIVER"
echo "============================================================"

"$PY" "$DRIVER" --benchmark-root "$HERE" --zing-root "$ZING_ROOT" "$@"

echo
echo "Videos -> $HERE/generated_videos_5s/"
echo "Now evaluate:  uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/<run> --output_folder <out> --descriptions_file descriptions/descriptions_original.csv"
