#!/usr/bin/env python3
"""Drive any flashdreams Cam2V model over Physics-IQ Verified (i2v mode).

LingBot World, HY-WorldPlay and SANA-WM all register against the same shared
Cam2V application, take the same conditioning (image, prompt, pose trace,
intrinsics) and differ only in resolution and pipeline defaults. One driver
therefore covers all of them; --app selects which.

    flashdreams-run-v2 <app> --mode mp4 --output-path OUT.mp4 \\
        -- --image-path FRAME --prompt TEXT --pose-path POSES --intrinsic-path K

There is no batch entrypoint -- Cam2V is built for interactive streaming and
mp4 is its only non-interactive output -- so this is one process per clip.

Camera: a static pose trace, the identity transform repeated for every frame.
Physics-IQ is filmed with a locked-off camera (every description ends "Static
shot with no camera movement."), so any camera motion would make the clip
incomparable to the ground truth. world_scale is left at each model's default
because it scales camera motion, of which there is none here.

Intrinsics are a real calibration and cannot be synthesised, so
--intrinsic-path defaults to a packaged example's intrinsics.npy.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_cam2v.py --app cam2v-lingbot --run-name lingbot-op-run_01 --limit 1
    python physiq/drive_cam2v.py --app cam2v-hy-worldplay --run-name hy-worldplay-op-run_01
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physiq_common import (TARGET_FPS, TARGET_FRAMES, RunTimer, conform_run,
                           find_switch_frames, iter_samples, load_take1_rows,
                           print_banner, resolve_paths)

# Mirrors each integration's Cam2VApplicationDefaults. Only values this driver
# needs; everything else stays at the application default.
CAM2V_MODELS = {
    "cam2v-lingbot":           {"fps": 16, "width": 832,  "height": 464},
    "cam2v-hy-worldplay":      {"fps": 16, "width": 1280, "height": 704},
    "cam2v-sana-wm-streaming": {"fps": 16, "width": 1280, "height": 704},
    "cam2v-dummy":             {"fps": 16, "width": 640,  "height": 360},
}
FRAMES_PER_BLOCK = 4  # autoregressive chunk size


def write_static_poses(path: Path, frames: int) -> bool:
    """Identity 4x4 transform per frame: a camera that never moves."""
    code = (f"import numpy as np; "
            f"np.save(r'{path}', np.tile(np.eye(4, dtype=np.float32), ({frames}, 1, 1)))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: could not write pose trace: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", required=True, choices=sorted(CAM2V_MODELS),
                    help="registered Cam2V application")
    ap.add_argument("--benchmark-root", type=Path, default=repo_root)
    ap.add_argument("--flashdreams-root", type=Path,
                    default=repo_root.parent / "flashdream_public",
                    help="flashdream checkout containing integrations_v2/")
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. lingbot-op-run_01")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--intrinsic-path", type=Path, default=None,
                    help="camera calibration .npy; defaults to "
                         "../lingbot-world-v2/examples/00/intrinsics.npy")
    ap.add_argument("--width", type=int, default=None, help="override model default")
    ap.add_argument("--height", type=int, default=None, help="override model default")
    ap.add_argument("--frames-per-block", type=int, default=FRAMES_PER_BLOCK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first command and exit")
    args = ap.parse_args()

    spec = CAM2V_MODELS[args.app]
    fps = spec["fps"]
    width = args.width or spec["width"]
    height = args.height or spec["height"]

    bench_root = args.benchmark_root.resolve()
    fd_root = args.flashdreams_root.resolve()
    switch_dir, csv_path, out_dir = resolve_paths(bench_root, args.descriptions,
                                                  args.out_root, args.run_name)
    # cam2v-<name> -> integrations_v2/<name>; lingbot and dummy are the exceptions
    # whose integration dir is not simply the suffix.
    integration_name = args.app.removeprefix("cam2v-").replace("-", "_")
    integration = fd_root / "integrations_v2" / integration_name

    for label, path in [("switch-frames", switch_dir), ("descriptions", csv_path),
                        ("flashdreams checkout", fd_root)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2
    if not integration.exists():
        print(f"WARNING: integration dir not found: {integration} "
              f"(the app may still be registered elsewhere)")

    intrinsic_path = (args.intrinsic_path
                      or repo_root.parent / "lingbot-world-v2" / "examples" / "00" / "intrinsics.npy")
    if not intrinsic_path.exists():
        print(f"ERROR: intrinsics not found: {intrinsic_path}", file=sys.stderr)
        print("       pass --intrinsic-path; a calibration cannot be synthesised.",
              file=sys.stderr)
        return 2

    native_frames = TARGET_FRAMES * fps // TARGET_FPS
    total_blocks = max(1, native_frames // args.frames_per_block)

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

    pose_path = work / f"static_{native_frames}_poses.npy"
    if not write_static_poses(pose_path, native_frames):
        return 2

    print_banner(args.app, csv_path, len(frames_map), len(samples),
                 native_frames, width, height, args.seed, out_dir,
                 {"integration": integration,
                  "blocks": f"{total_blocks} x {args.frames_per_block} frames @ {fps}fps",
                  "poses": "identity per frame (static)",
                  "intrinsics": intrinsic_path,
                  "conform": f"{native_frames}@{fps} -> {TARGET_FRAMES}@{TARGET_FPS} (5.00s)"},
                 fps=fps)

    generated = skipped = failed = 0
    timer = RunTimer(len(samples))

    for _, stem, description, frame in samples:
        target = out_dir / f"{stem}.mp4"
        if args.skip_existing and target.exists():
            skipped += 1
            continue

        cmd = ["uv", "run", "--no-sync", "flashdreams-run-v2", args.app,
               "--mode", "mp4", "--output-path", str(target.resolve()),
               "--fps", str(fps),
               "--",
               "--image-path", str(frame.resolve()),
               "--prompt", description,
               "--pose-path", str(pose_path.resolve()),
               "--intrinsic-path", str(intrinsic_path.resolve()),
               "--no-example-data",
               "--total-blocks", str(total_blocks),
               "--seed", str(args.seed),
               "--no-ui"]
        if args.device:
            cmd += ["--device", args.device]

        if args.dry_run:
            print("\n[physiq-cam2v] --dry-run; first command:")
            print("  " + " ".join(cmd))
            return 0

        timer.start_item()
        result = subprocess.run(cmd, cwd=str(fd_root))
        ok = result.returncode == 0 and target.exists()
        print("[physiq-cam2v] " + timer.finish_item(stem, ok))
        if not ok:
            failed += 1
            continue
        generated += 1

    rewritten, total = conform_run(out_dir, TARGET_FRAMES)
    print(f"\n[physiq-cam2v] {args.app}: generated {generated}, skipped {skipped}, failed {failed}")
    print("[physiq-cam2v] " + timer.summary())
    print(f"[physiq-cam2v] conformed {rewritten}/{total} to {TARGET_FRAMES} frames @ {TARGET_FPS}fps")
    print(f"\nNext: evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {csv_path}")
    return 1 if failed and not generated else 0


if __name__ == "__main__":
    raise SystemExit(main())
