#!/usr/bin/env python3
"""Drive Echo-WM over the Physics-IQ Verified benchmark (image-to-video mode).

Echo's inference_wm.py is I2V-only and one-shot: one process per clip. That is
slow (it reloads ~47.8GB plus Gemma every time) but it is also the only path
that completes -- the load-once worker in MIND's drive_echo.py OOM'd at 249 GiB
on a 249 GiB card. There is no batch entrypoint to amortise the reload against,
so this driver pays it 198 times.

Camera: --action-str "none-<frames>". Echo's parse_action_string treats the
literal "none" as an empty key set, which is exactly the locked-off camera
Physics-IQ needs -- every benchmark description ends "Static shot with no camera
movement." Any wasd/ijkl key would pan the viewpoint and make the clip
incomparable to the ground truth.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_echo.py --run-name echo-op-run_01 --limit 1
    python physiq/drive_echo.py --run-name echo-op-run_01
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from physiq_common import (TARGET_FPS, TARGET_FRAMES, RunTimer, conform_run,
                           find_switch_frames, iter_samples, load_take1_rows,
                           print_banner, prune_sidecars, resolve_paths, run_clips)


def valid_frame_count(minimum: int) -> int:
    """Smallest 8k+1 >= minimum.

    Echo quantises --num-frames down to its latent stride: asking for 120 yields
    113 (8*14+1) and a 4.71s clip, which the benchmark rejects and which
    conform cannot fix, since trimming cannot lengthen. Ask for 121 instead and
    trim the one extra frame off.
    """
    n = max(minimum, 1)
    while (n - 1) % 8:
        n += 1
    return n


def resolve_echo_python(echo_root: Path) -> Path:
    """Echo runs in its own venv; the physiq env cannot import its deps."""
    for rel in ("echo_wm/.venv/bin/python", "echo_wm/.venv/Scripts/python.exe",
                ".venv/bin/python", ".venv/Scripts/python.exe"):
        candidate = echo_root / rel
        if candidate.exists():
            return candidate
    return echo_root / "echo_wm" / ".venv" / "bin" / "python"


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", type=Path, default=repo_root)
    ap.add_argument("--echo-root", type=Path, default=repo_root.parent / "JoyAI-Echo",
                    help="JoyAI-Echo checkout")
    ap.add_argument("--echo-python", type=Path, default=None)
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. echo-op-run_01")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--frames", type=int, default=TARGET_FRAMES,
                    help=f"target frames after conform; {TARGET_FRAMES} = 5.0s at "
                         f"{TARGET_FPS}fps. Generation requests the next 8k+1 above "
                         f"this, because Echo rounds down to that stride.")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N samples -- use --limit 1 to smoke-test")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                    help="skip samples whose output already exists; default on so an "
                         "interrupted run resumes instead of restarting")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first command and exit")
    args = ap.parse_args()

    bench_root = args.benchmark_root.resolve()
    echo_root = args.echo_root.resolve()
    echo_python = args.echo_python or resolve_echo_python(echo_root)
    switch_dir, csv_path, out_dir = resolve_paths(bench_root, args.descriptions,
                                                  args.out_root, args.run_name)

    for label, path in [("switch-frames", switch_dir), ("descriptions", csv_path),
                        ("echo checkout", echo_root)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    gen_frames = valid_frame_count(args.frames)

    frames_map = find_switch_frames(switch_dir)
    rows = load_take1_rows(csv_path)
    if args.limit:
        rows = rows[:args.limit]
    samples = list(iter_samples(rows, frames_map))
    if not samples:
        print("ERROR: no samples", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    print_banner("Echo-WM", csv_path, len(frames_map), len(samples), gen_frames,
                 args.width, args.height, args.seed, out_dir,
                 {"echo": echo_root, "steps": args.steps,
                  "action": f"none-{gen_frames}",
                  "conform": f"{gen_frames} -> {args.frames} frames "
                             f"({args.frames / TARGET_FPS:.2f}s)"})

    entry = echo_root / "echo_wm" / "inference_wm.py"
    generated = skipped = failed = 0
    timer = RunTimer(len(samples))

    for index, (bench_id, stem, description, frame) in enumerate(samples, 1):
        target = out_dir / f"{stem}.mp4"
        if args.skip_existing and target.exists():
            skipped += 1
            continue

        cmd = [str(echo_python), str(entry),
               "--image", str(frame.resolve()),
               "--prompt", description,
               "--action-str", f"none-{gen_frames}",
               "--num-frames", str(gen_frames),
               "--fps", str(TARGET_FPS),
               "--width", str(args.width),
               "--height", str(args.height),
               "--steps", str(args.steps),
               "--seed", str(args.seed),
               "--output", str(target)]

        if args.dry_run:
            print("\n[physiq-echo] --dry-run; first command:")
            print("  " + " ".join(cmd))
            return 0

        timer.start_item()
        result = subprocess.run(cmd, cwd=str(echo_root / "echo_wm"))
        ok = result.returncode == 0 and target.exists()
        print(f"[physiq-echo] " + timer.finish_item(stem, ok))
        if not ok:
            failed += 1
            continue
        generated += 1

    rewritten, total = conform_run(out_dir, args.frames)
    print(f"\n[physiq-echo] generated {generated}, skipped {skipped}, failed {failed}")
    print(f"[physiq-echo] conformed {rewritten}/{total} clip(s) to "
          f"{args.frames} frames @ {TARGET_FPS}fps")
    print(f"\nNext: evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {csv_path}")
    return 1 if failed and not generated else 0


if __name__ == "__main__":
    raise SystemExit(main())
