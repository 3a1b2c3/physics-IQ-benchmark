#!/usr/bin/env python3
"""Drive ABot-World over the Physics-IQ Verified benchmark (image-to-video mode).

Unlike the other drivers this one launches a single process for all 198 samples.
scripts/inference.py's --mind-batch takes a manifest of
{ref_image, action_json, prompt, out_path, fps_blocks} and loops it with
get_pipeline lru_cached, so the ~24GB model loads once instead of 198 times.
That is the difference between a run that finishes and one that does not.

Camera: an action JSON with every key false for every frame. ABot's
_parse_action_json samples one [W,A,S,D,I,J,K,L] per block, so an all-false file
yields no movement in any block -- the locked-off camera Physics-IQ needs, since
every benchmark description ends "Static shot with no camera movement."

Frame rate: ABot emits VIDEO_FPS=12 (web_client/config.py) with 12 frames per
block, so 5.00s is 60 frames = 5 blocks. The benchmark wants 5s at 24fps, so
conform_run re-encodes 60@12 to 120@24. That duplicates each frame rather than
inventing motion -- the temporal content is unchanged, the container just
matches what the evaluator expects.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_abot.py --run-name abot-op-run_01 --limit 1
    python physiq/drive_abot.py --run-name abot-op-run_01
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physiq_common import (TARGET_FPS, TARGET_FRAMES, conform_run, find_switch_frames,
                           fmt_duration, iter_samples, load_take1_rows, print_banner,
                           resolve_paths)

KEY_ORDER = ["W", "A", "S", "D", "I", "J", "K", "L"]
ABOT_FPS = 12           # web_client/config.py VIDEO_FPS
FRAMES_PER_BLOCK = 12   # num_fpb * frames_per_latent


def resolve_abot_python(abot_root: Path) -> Path:
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        candidate = abot_root / rel
        if candidate.exists():
            return candidate
    return abot_root / ".venv" / "bin" / "python"


def write_static_action_json(path: Path, n_frames: int, fps: int) -> None:
    """Per-frame action file with every key released."""
    released = {key: False for key in KEY_ORDER}
    path.write_text(json.dumps({
        "total_frames": n_frames,
        "fps": fps,
        "frames": [{"keys": dict(released), "frame_id": f"{i:06d}"} for i in range(n_frames)],
    }), encoding="utf-8")


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", type=Path, default=repo_root)
    ap.add_argument("--abot-root", type=Path, default=repo_root.parent / "ABot-World",
                    help="ABot-World checkout")
    ap.add_argument("--abot-python", type=Path, default=None)
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. abot-op-run_01")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--frames-per-block", type=int, default=FRAMES_PER_BLOCK)
    ap.add_argument("--quant-type", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--manifest-only", action="store_true",
                    help="write the manifest and print the command, run nothing")
    args = ap.parse_args()

    bench_root = args.benchmark_root.resolve()
    abot_root = args.abot_root.resolve()
    abot_python = args.abot_python or resolve_abot_python(abot_root)
    switch_dir, csv_path, out_dir = resolve_paths(bench_root, args.descriptions,
                                                  args.out_root, args.run_name)

    for label, path in [("switch-frames", switch_dir), ("descriptions", csv_path),
                        ("ABot-World checkout", abot_root)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    # 5.00s at ABot's native 12fps, expressed in whole blocks.
    native_frames = TARGET_FRAMES * ABOT_FPS // TARGET_FPS
    blocks = max(1, native_frames // args.frames_per_block)

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

    action_path = work / f"static_{native_frames}f.json"
    write_static_action_json(action_path, native_frames, ABOT_FPS)

    manifest = []
    pending = 0
    for _, stem, description, frame in samples:
        target = out_dir / f"{stem}.mp4"
        if args.skip_existing and target.exists():
            continue
        manifest.append({
            "ref_image": str(frame.resolve()),
            "action_json": str(action_path.resolve()),
            "prompt": description,
            "out_path": str(target.resolve()),
            "fps_blocks": blocks,
        })
        pending += 1

    manifest_path = work / "mind_batch.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_banner("ABot-World", csv_path, len(frames_map), len(samples),
                 native_frames, "native", "native", args.seed, out_dir,
                 {"abot": abot_root,
                  "blocks": f"{blocks} x {args.frames_per_block} frames @ {ABOT_FPS}fps",
                  "actions": f"all keys released, {native_frames} frames",
                  "conform": f"{native_frames}@{ABOT_FPS} -> {TARGET_FRAMES}@{TARGET_FPS} (5.00s)",
                  "pending": f"{pending} of {len(samples)} (rest already generated)"})

    if not manifest:
        print("\n[physiq-abot] nothing to generate; all outputs exist")
        rewritten, total = conform_run(out_dir, TARGET_FRAMES)
        print(f"[physiq-abot] conformed {rewritten}/{total} clip(s)")
        return 0

    cmd = [str(abot_python), "scripts/inference.py",
           "--mind-batch", str(manifest_path.resolve()),
           "--quant-type", args.quant_type]

    # run_example_000.bat sets these; without them the run picks a different
    # attention backend and offload policy than the one that was validated.
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(abot_root),
        "PYTHONUNBUFFERED": "1",
        "TORCHDYNAMO_DISABLE": "1",
        "ABOT_FLEX_BYPASS": "1",
        "ABOT_ATTN_BACKEND": "sdpa",
        "ABOT_NO_OFFLOAD": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    })

    if args.manifest_only:
        print(f"\n[physiq-abot] --manifest-only; manifest: {manifest_path}")
        print("  " + " ".join(cmd))
        return 0

    print()
    _t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(abot_root), env=env)
    if result.returncode != 0:
        print(f"[physiq-abot] inference exited {result.returncode}", file=sys.stderr)

    rewritten, total = conform_run(out_dir, TARGET_FRAMES)
    produced = sorted(out_dir.glob("*.mp4"))
    print(f"\n[physiq-abot] {len(produced)}/{len(samples)} clip(s) in {out_dir}")
    print(f"[physiq-abot] conformed {rewritten}/{total} to {TARGET_FRAMES} frames @ {TARGET_FPS}fps")
    print(f"\nNext: evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {csv_path}")
    return 0 if produced else 1


if __name__ == "__main__":
    raise SystemExit(main())
