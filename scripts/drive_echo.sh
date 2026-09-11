#!/usr/bin/env bash
# Stage Echo-WM generations into Physics-IQ Verified, then evaluate.
#
# Same shape as MIND's scripts/drive_<model>.sh: this wrapper resolves paths and
# checks preconditions, physiq/drive_echo.py does the work, and every extra
# argument is forwarded through.
#
#   bash scripts/drive_echo.sh --run-name echo-op-run_01 --limit 1
#   bash scripts/drive_echo.sh --run-name echo-op-run_01
#   bash scripts/drive_echo.sh --run-name echo-op-run_02 --seed 1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

ECHO_ROOT="${ECHO_ROOT:-$(cd "$HERE/.." && pwd)/JoyAI-Echo}"
DRIVER="$HERE/physiq/drive_echo.py"

# Zing runs in its own venv -- the model package is not importable from physiq's env.
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
if [ ! -d "$ECHO_ROOT" ]; then
  echo "ERROR: JoyAI-Echo checkout not found: $ECHO_ROOT" >&2
  echo "       set ECHO_ROOT=/path/to/JoyAI-Echo" >&2
  exit 2
fi
if [ ! -d "$HERE/physics-IQ-benchmark-verified/switch-frames" ]; then
  echo "ERROR: benchmark data missing: $HERE/physics-IQ-benchmark-verified/switch-frames" >&2
  echo "       run bash download_verified_data.sh" >&2
  exit 2
fi

echo "============================================================"
echo "Echo-WM staging into Physics-IQ Verified"
echo "============================================================"
echo "  benchmark : $HERE"
echo "  echo      : $ECHO_ROOT"
echo "  driver    : $DRIVER"
echo "============================================================"

"$PY" "$DRIVER" --benchmark-root "$HERE" --echo-root "$ECHO_ROOT" "$@"

echo
echo "Videos -> $HERE/generated_videos_5s/"
echo "Now evaluate:  uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/<run> --output_folder <out> --descriptions_file descriptions/descriptions_original.csv"
