#!/usr/bin/env python3
"""Drive LingBot-World-v2 over the Physics-IQ Verified benchmark (i2v mode).

generate.py's --mind_batch takes a manifest of {image, action_path, out_path,
prompt} and loops it in one process so the ~14B model loads once. The per-run
alternative pays ~2.5 min of model load plus fp8 quantize on every sample, which
across 198 clips is roughly eight hours of pure reloading.

batch_generate.py also loads once but names outputs
output/<exname>_<timestamp>.mp4, which the benchmark evaluator cannot match --
it keys on the "0001_" ID prefix. --mind_batch takes an explicit out_path, so
clips land already correctly named.

Camera: an all-zero [frame_num, 4] action array. Physics-IQ is filmed with a
locked-off camera -- every description ends "Static shot with no camera
movement." -- so any non-zero channel would pan the viewpoint and make the clip
incomparable to the ground truth.

Frame count: LingBot emits sample_fps=16 (wan/configs/shared_config.py) and
requires frame_num == 4n+1. 5.00s at 16fps is 80 frames, which is not 4n+1, so
this generates 81 (5.06s) and conform_run trims to the benchmark's 120 @ 24fps.
Trimming the tail is safe; the extra frame is past the 5s window being scored.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_lingbot.py --run-name lingbot-op-run_01 --limit 1
    python physiq/drive_lingbot.py --run-name lingbot-op-run_01
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physiq_common import (TARGET_FPS, TARGET_FRAMES, conform_run, find_switch_frames,
                           fmt_duration, iter_samples, load_take1_rows, print_banner,
                           resolve_paths)

LINGBOT_FPS = 16     # wan/configs/shared_config.py sample_fps
ACTION_CHANNELS = 4  # examples/*/action.npy is [frames, 4]


def valid_frame_count(minimum: int) -> int:
    """Smallest 4n+1 >= minimum (generate.py's --frame_num constraint)."""
    n = minimum
    while (n - 1) % 4:
        n += 1
    return n


def resolve_lingbot_python(root: Path) -> Path:
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe", "venv/bin/python"):
        candidate = root / rel
        if candidate.exists():
            return candidate
    return root / ".venv" / "bin" / "python"


def write_static_actions(python: Path, path: Path, frames: int, channels: int) -> bool:
    """All-zero [frames, channels] int32 action array, matching examples/*/action.npy.

    Written through LingBot's interpreter rather than importing numpy here: this
    driver lives in the benchmark repo, whose environment has no numpy.
    """
    code = (f"import numpy as np; "
            f"np.save(r'{path}', np.zeros(({frames}, {channels}), dtype=np.int32))")
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: could not write action array: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", type=Path, default=repo_root)
    ap.add_argument("--lingbot-root", type=Path, default=repo_root.parent / "lingbot-world-v2",
                    help="lingbot-world-v2 checkout")
    ap.add_argument("--lingbot-python", type=Path, default=None)
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="checkpoint dir (default: the lingbot checkout itself)")
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. lingbot-op-run_01")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--task", default="i2v-A14B")
    ap.add_argument("--infer-mode", default="causal_fast",
                    choices=["causal_fast", "causal_pretrain"])
    ap.add_argument("--size", default="1280*720")
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp8", action=argparse.BooleanOptionalAction, default=True,
                    help="fp8 pipeline; default on (what the 14B fits in)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest-only", action="store_true",
                    help="write the manifest and print the command, run nothing")
    args = ap.parse_args()

    bench_root = args.benchmark_root.resolve()
    lingbot_root = args.lingbot_root.resolve()
    lingbot_python = args.lingbot_python or resolve_lingbot_python(lingbot_root)
    ckpt_dir = args.ckpt_dir or lingbot_root
    switch_dir, csv_path, out_dir = resolve_paths(bench_root, args.descriptions,
                                                  args.out_root, args.run_name)

    for label, path in [("switch-frames", switch_dir), ("descriptions", csv_path),
                        ("lingbot checkout", lingbot_root)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    native_frames = valid_frame_count(TARGET_FRAMES * LINGBOT_FPS // TARGET_FPS)

    frames_map = find_switch_frames(switch_dir)
    rows = load_take1_rows(csv_path)
    if args.limit:
        rows = rows[:args.limit]
    samples = list(iter_samples(rows, frames_map))
    if not samples:
        print("ERROR: no samples", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    action_path = work / f"static_{native_frames}.npy"
    if not write_static_actions(lingbot_python, action_path, native_frames, ACTION_CHANNELS):
        return 2

    manifest = []
    for _, stem, description, frame in samples:
        target = out_dir / f"{stem}.mp4"
        if args.skip_existing and target.exists():
            continue
        manifest.append({
            "image": str(frame.resolve()),
            "action_path": str(action_path.resolve()),
            "out_path": str(target.resolve()),
            "prompt": description,
        })

    manifest_path = work / "mind_batch.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_banner("LingBot-World-v2", csv_path, len(frames_map), len(samples),
                 native_frames, *args.size.split("*"), args.seed, out_dir,
                 {"lingbot": lingbot_root, "task": args.task,
                  "infer_mode": args.infer_mode,
                  "actions": f"all-zero [{native_frames}, {ACTION_CHANNELS}]",
                  "conform": f"{native_frames}@{LINGBOT_FPS} -> {TARGET_FRAMES}@{TARGET_FPS} (5.00s)",
                  "pending": f"{len(manifest)} of {len(samples)}"})

    if not manifest:
        print("\n[physiq-lingbot] nothing to generate; all outputs exist")
        rewritten, total = conform_run(out_dir, TARGET_FRAMES)
        print(f"[physiq-lingbot] conformed {rewritten}/{total} clip(s)")
        return 0

    cmd = [str(lingbot_python), "generate.py",
           "--mind_batch", str(manifest_path.resolve()),
           "--task", args.task,
           "--ckpt_dir", str(ckpt_dir),
           "--infer_mode", args.infer_mode,
           "--size", args.size,
           "--frame_num", str(native_frames),
           "--chunk_size", str(args.chunk_size),
           "--base_seed", str(args.seed)]
    if args.fp8:
        cmd.append("--fp8")

    if args.manifest_only:
        print(f"\n[physiq-lingbot] --manifest-only; manifest: {manifest_path}")
        print("  " + " ".join(cmd))
        return 0

    print()
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(lingbot_root))
    if result.returncode != 0:
        print(f"[physiq-lingbot] generate.py exited {result.returncode}", file=sys.stderr)
    elapsed = time.monotonic() - t0

    rewritten, total = conform_run(out_dir, TARGET_FRAMES)
    produced = sorted(out_dir.glob("*.mp4"))
    print(f"\n[physiq-lingbot] {len(produced)}/{len(samples)} clip(s) in {out_dir}")
    print(f"[physiq-lingbot] generation took {fmt_duration(elapsed)}"
          + (f" ({elapsed / len(manifest):.1f}s per clip)" if manifest else ""))
    print(f"[physiq-lingbot] conformed {rewritten}/{total} to {TARGET_FRAMES} frames @ {TARGET_FPS}fps")
    print(f"\nNext: evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {csv_path}")
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
