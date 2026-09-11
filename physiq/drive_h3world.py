#!/usr/bin/env python3
"""Drive H3-World over the Physics-IQ Verified benchmark (image-to-video mode).

Physics-IQ gives a single "switch frame" per scenario and asks the model to
predict the next 5 seconds. H3-World is not an interactive/batched world
model like Zing -- code/abot/infer.py is a single-purpose CLI that loads the
whole pipeline and produces exactly one clip per invocation (see its own
docstring). This driver therefore shells out to infer.py once per benchmark
sample rather than building one batched request, which is slow (198 full
pipeline loads) but is the only interface infer.py exposes.

Physics-IQ scenarios are filmed with a locked-off camera -- every description
in the CSV ends "Static shot with no camera movement." -- so every sample
uses --action-preset still (the all-zero action). Any other preset would pan
the viewpoint and make the generated clip incomparable to the ground truth.

Frame count: infer.py requires --num-frames = 17k+5, and keyframe_indices=[0]
means the given first-frame becomes output frame 0 itself (not a separate
frame prepended in addition, unlike Zing). The benchmark needs exactly 5.00s
predicting *after* the switch frame, so this driver generates the smallest
valid frame count that leaves at least 120 frames once frame 0 (the given
switch frame) is dropped -- 124 (17*7+5) for the default 120-frame (5.0s @
24fps) request -- then re-encodes to drop frame 0 and keep exactly 120.

H3-World lives in its own conda env (minimax_h3, see README.md), not a venv
next to the checkout like Zing, so this driver launches it via `conda run`
by default rather than resolving a fixed interpreter path.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_h3world.py --run-name h3world-op-run_01 --limit 1
    python physiq/drive_h3world.py --run-name h3world-op-run_01
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ID_RE = re.compile(r"^(\d{4})_")


def find_switch_frames(switch_dir: Path) -> dict[str, Path]:
    """Map the 4-digit benchmark ID to its switch frame."""
    frames: dict[str, Path] = {}
    for path in sorted(switch_dir.glob("*.jpg")):
        match = ID_RE.match(path.name)
        if match:
            frames[match.group(1)] = path
    return frames


def load_take1_rows(descriptions_csv: Path) -> list[dict]:
    """The 198 take-1 rows -- the only ones needed for generation."""
    with open(descriptions_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if "take-1" in r["scenario"]]


def build_samples(rows: list[dict], frames: dict[str, Path]) -> tuple[list[dict], list[str]]:
    samples = []
    missing = []
    for row in rows:
        out_name = row["generated_video_name"]
        match = ID_RE.match(out_name)
        if not match:
            missing.append(f"{out_name} (no ID prefix)")
            continue
        bench_id = match.group(1)
        frame_path = frames.get(bench_id)
        if frame_path is None:
            missing.append(f"{out_name} (no switch frame for ID {bench_id})")
            continue
        samples.append({
            "sample_id": Path(out_name).stem,
            "description": row["description"],
            "first_frame": frame_path,
        })
    return samples, missing


def raw_frames_for(final_frames: int) -> int:
    """Smallest 17k+5 that leaves >= final_frames once frame 0 is dropped."""
    k = 0
    while 17 * k + 5 < final_frames + 1:
        k += 1
    return 17 * k + 5


def drop_first_frame(video: Path, keep_frames: int, fps: int) -> bool:
    """Drop the given switch frame (output frame 0) and keep the next
    keep_frames frames, so the export scores only what the model predicted
    after the switch, not a copy of what it was given. Re-encodes rather
    than stream-copying because dropping the first frame moves the keyframe.
    Returns True if the file was rewritten.
    """
    probe = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True)
    if probe.returncode != 0:
        return False
    try:
        n_frames = int(probe.stdout.strip())
    except ValueError:
        return False
    if n_frames == keep_frames:
        return False  # already correct

    tmp = video.with_suffix(".trim.mp4")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", r"select=gte(n\,1),setpts=PTS-STARTPTS",
         "-frames:v", str(keep_frames), "-r", str(fps), "-an", str(tmp)],
        capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(video)
    return True


def resolve_h3_python(h3_root: Path, conda_env: str, h3_python: Path | None) -> list[str]:
    """Command prefix to run a python script inside H3-World's environment.

    A direct interpreter path (--h3-python) is used verbatim if given;
    otherwise this shells out via `conda run`, since H3-World's README sets
    it up as a named conda env (minimax_h3) rather than a venv next to the
    checkout the way zing-world-model is.
    """
    if h3_python is not None:
        return [str(h3_python)]
    return ["conda", "run", "-n", conda_env, "--no-capture-output", "python3"]


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", type=Path, default=repo_root,
                    help="physics-IQ-benchmark checkout (default: this repo)")
    ap.add_argument("--h3-root", type=Path,
                    default=repo_root.parent / "H3-World",
                    help="H3-World checkout")
    ap.add_argument("--conda-env", default="minimax_h3",
                    help="conda env H3-World was installed into (default: minimax_h3)")
    ap.add_argument("--h3-python", type=Path, default=None,
                    help="direct interpreter path, bypassing `conda run` "
                         "(e.g. <conda-envs>/minimax_h3/python.exe)")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="LoRA checkpoint (default: <h3-root>/checkpoints/H3-World/step-10000.safetensors)")
    ap.add_argument("--descriptions", type=Path, default=None,
                    help="descriptions CSV (default: <benchmark-root>/descriptions/"
                         "descriptions_original.csv, i.e. the 'op' prompt setting)")
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. h3world-op-run_01. The benchmark "
                         "expects <model>-<op|bpp>-run_<NN> and averages run_01..run_04.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="where run folders are written (default: "
                         "<benchmark-root>/generated_videos_5s)")
    ap.add_argument("--frames", type=int, default=120,
                    help="final output frames after dropping the switch frame; "
                         "120 = 5.0s @ 24fps, the benchmark's required duration")
    ap.add_argument("--fps", type=int, default=24, help="H3-World's fixed output fps")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--subject", choices=("man", "car"), default="man",
                    help="noun used in the auto-generated action clauses (default: man, "
                         "matches the trained vocabulary; see infer.py --subject help)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="vary per run so run_01..run_04 are independent")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N samples -- use --limit 1 to smoke-test")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-sample commands, run nothing")
    args = ap.parse_args()

    bench_root = args.benchmark_root.resolve()
    verified = bench_root / "physics-IQ-benchmark-verified"
    switch_dir = verified / "switch-frames"
    descriptions = args.descriptions or bench_root / "descriptions" / "descriptions_original.csv"
    out_root = args.out_root or bench_root / "generated_videos_5s"
    h3_root = args.h3_root.resolve()
    checkpoint = args.checkpoint or h3_root / "checkpoints" / "H3-World" / "step-10000.safetensors"
    infer_script = h3_root / "code" / "abot" / "infer.py"

    for label, path in [("switch-frames", switch_dir), ("descriptions", descriptions),
                        ("H3-World checkout", h3_root), ("infer.py", infer_script)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2
    if not args.dry_run and not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    frames = find_switch_frames(switch_dir)
    rows = load_take1_rows(descriptions)
    if args.limit:
        rows = rows[:args.limit]

    samples, missing = build_samples(rows, frames)
    if missing:
        print(f"[physiq-h3world] WARNING: skipped {len(missing)} sample(s):", file=sys.stderr)
        for item in missing[:5]:
            print(f"    {item}", file=sys.stderr)
    if not samples:
        print("ERROR: no samples built", file=sys.stderr)
        return 2

    raw_frames = raw_frames_for(args.frames)
    out_dir = out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    py_cmd = resolve_h3_python(h3_root, args.conda_env, args.h3_python)

    print("=" * 60)
    print("H3-World -> Physics-IQ Verified (i2v, static camera)")
    print("=" * 60)
    print(f"  descriptions : {descriptions}")
    print(f"  switch frames: {len(frames)} found")
    print(f"  samples      : {len(samples)}")
    print(f"  raw frames   : {raw_frames} (17k+5) -> trimmed to {args.frames} "
          f"@ {args.fps}fps = {args.frames / args.fps:.2f}s, {args.width}x{args.height}")
    print(f"  actions      : still (static camera)")
    print(f"  subject      : {args.subject}")
    print(f"  seed         : {args.seed}")
    print(f"  h3-root      : {h3_root}")
    print(f"  checkpoint   : {checkpoint}")
    print(f"  runner       : {' '.join(py_cmd)}")
    print(f"  out_dir      : {out_dir}")
    print("=" * 60)

    env = os.environ.copy()
    env["ABOT_HEIGHT"] = str(args.height)
    env["ABOT_WIDTH"] = str(args.width)

    failed = []
    produced = 0
    for i, sample in enumerate(samples, 1):
        out_path = out_dir / f"{sample['sample_id']}.mp4"
        cmd = py_cmd + [
            str(infer_script),
            "--checkpoint", str(checkpoint),
            "--first-frame", str(sample["first_frame"]),
            "--scene-prompt", sample["description"],
            "--action-preset", "still",
            "--subject", args.subject,
            "--num-frames", str(raw_frames),
            "--steps", str(args.steps),
            "--cfg-scale", str(args.cfg_scale),
            "--seed", str(args.seed),
            "--out", str(out_path),
        ]

        print(f"\n[{i}/{len(samples)}] {sample['sample_id']}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            continue

        result = subprocess.run(cmd, cwd=str(h3_root), env=env)
        if result.returncode != 0:
            print(f"[physiq-h3world] WARNING: generation failed for {sample['sample_id']} "
                  f"(exit {result.returncode})", file=sys.stderr)
            failed.append(sample["sample_id"])
            continue

        if not drop_first_frame(out_path, args.frames, args.fps):
            print(f"[physiq-h3world] WARNING: could not trim {out_path} to "
                  f"{args.frames} frames", file=sys.stderr)
            failed.append(sample["sample_id"])
            continue
        produced += 1

    if args.dry_run:
        return 0

    print(f"\n[physiq-h3world] {produced}/{len(samples)} clip(s) in {out_dir}")
    if failed:
        print(f"[physiq-h3world] {len(failed)} sample(s) failed:", file=sys.stderr)
        for name in failed[:5]:
            print(f"    {name}", file=sys.stderr)

    print("\nNext: verify duration is exactly 5.00s, then evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {descriptions}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
