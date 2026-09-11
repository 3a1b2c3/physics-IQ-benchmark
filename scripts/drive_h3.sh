#!/usr/bin/env bash
# Stage H3-World generations into Physics-IQ Verified, then evaluate.
#
# Same shape as drive_zing.sh / MIND's scripts/drive_<model>.sh: this wrapper
# resolves paths and checks preconditions, physiq/drive_h3.py does the work
# (one infer.py subprocess per scenario, not a single batched call -- H3-World
# has no JSONL/session mode), and every extra argument is forwarded through.
#
#   bash scripts/drive_h3.sh --run-name h3-world-op-run_01 --limit 1
#   bash scripts/drive_h3.sh --run-name h3-world-op-run_01
#   bash scripts/drive_h3.sh --run-name h3-world-op-run_02 --seed 1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

H3_ROOT="${H3_ROOT:-$(cd "$HERE/.." && pwd)/H3-World}"
DRIVER="$HERE/physiq/drive_h3.py"

# H3-World runs in its own venv -- its DiffSynth-Studio-h3-v2 fork and
# code/abot modules are not importable from physiq's env. The driver launches
# that interpreter itself per sample; this only needs a python able to read a
# CSV and shell out, so prefer physiq's venv and fall back to python3.
if [ -x "$HERE/.venv/bin/python" ]; then
  PY="$HERE/.venv/bin/python"
else
  PY="$(command -v python3)"
fi

if [ ! -f "$DRIVER" ]; then
  echo "ERROR: driver not found: $DRIVER" >&2
  exit 2
fi
if [ ! -d "$H3_ROOT" ]; then
  echo "ERROR: H3-World checkout not found: $H3_ROOT" >&2
  echo "       set H3_ROOT=/path/to/H3-World" >&2
  exit 2
fi
if [ ! -f "$H3_ROOT/code/abot/infer.py" ]; then
  echo "ERROR: H3-World checkout looks incomplete, missing code/abot/infer.py: $H3_ROOT" >&2
  exit 2
fi
if [ ! -d "$HERE/physics-IQ-benchmark-verified/switch-frames" ]; then
  echo "ERROR: benchmark data missing: $HERE/physics-IQ-benchmark-verified/switch-frames" >&2
  echo "       run bash download_verified_data.sh" >&2
  exit 2
fi

echo "============================================================"
echo "H3-World staging into Physics-IQ Verified"
echo "============================================================"
echo "  benchmark : $HERE"
echo "  h3-world  : $H3_ROOT"
echo "  driver    : $DRIVER"
echo "============================================================"

"$PY" "$DRIVER" --benchmark-root "$HERE" --h3-root "$H3_ROOT" "$@"

echo
echo "Videos -> $HERE/generated_videos_5s/"
echo "Now evaluate:  uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/<run> --output_folder <out> --descriptions_file descriptions/descriptions_original.csv"
