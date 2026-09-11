#!/usr/bin/env bash
# Stage ABot-World generations into Physics-IQ Verified, then evaluate.
#
# Same shape as MIND's scripts/drive_<model>.sh: this wrapper resolves paths and
# checks preconditions, physiq/drive_abot.py does the work, and every extra
# argument is forwarded through.
#
#   bash scripts/drive_abot.sh --run-name abot-op-run_01 --limit 1
#   bash scripts/drive_abot.sh --run-name abot-op-run_01
#   bash scripts/drive_abot.sh --run-name abot-op-run_02 --seed 1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

ABOT_ROOT="${ABOT_ROOT:-$(cd "$HERE/.." && pwd)/ABot-World}"
DRIVER="$HERE/physiq/drive_abot.py"

# Zing runs in its own venv -- the ABot package is not importable from physiq's env.
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
if [ ! -d "$ABOT_ROOT" ]; then
  echo "ERROR: ABot-World checkout not found: $ABOT_ROOT" >&2
  echo "       set ABOT_ROOT=/path/to/ABot-World" >&2
  exit 2
fi
if [ ! -d "$HERE/physics-IQ-benchmark-verified/switch-frames" ]; then
  echo "ERROR: benchmark data missing: $HERE/physics-IQ-benchmark-verified/switch-frames" >&2
  echo "       run bash download_verified_data.sh" >&2
  exit 2
fi

echo "============================================================"
echo "ABot-World staging into Physics-IQ Verified"
echo "============================================================"
echo "  benchmark : $HERE"
echo "  abot      : $ABOT_ROOT"
echo "  driver    : $DRIVER"
echo "============================================================"

"$PY" "$DRIVER" --benchmark-root "$HERE" --abot-root "$ABOT_ROOT" "$@"

echo
echo "Videos -> $HERE/generated_videos_5s/"
echo "Now evaluate:  uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/<run> --output_folder <out> --descriptions_file descriptions/descriptions_original.csv"
