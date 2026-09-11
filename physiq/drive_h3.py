#!/usr/bin/env python3
"""Drive H3-World over the Physics-IQ Verified benchmark (image-to-video mode).

Physics-IQ gives a single "switch frame" per scenario and asks the model to
predict the next 5 seconds. Unlike Zing (an interactive world model driven by
a batched JSONL of sessions), H3-World's code/abot/infer.py is a plain,
single-sample CLI: one process per (first frame, scene prompt, action)
triple. So this driver loops over the take-1 rows and shells out to infer.py
once per scenario instead of building one batch file.

Every sample uses --action-preset still. Physics-IQ scenarios are filmed with
a locked-off camera -- every description in the CSV ends "Static shot with no
camera movement." -- so any W/A/S/D/I/J/K/L would pan or translate the
viewpoint and make the generated clip incomparable to the ground truth.
infer.py's ACTION_PRESETS maps "still" to an empty key tuple, i.e. every row
of the 9-key action grid (abot_action.ACTIVE_KEY_COLS) stays zero for the
whole clip -- this is a real, first-class preset the model ships with, not a
hand-built all-zero array standing in for one. It is also off-distribution
for a model trained on continuously-moving gameplay footage: if rollouts
drift or hallucinate motion, that is a real property of the model on this
benchmark, not a bug in this script.

Frame count: infer.py writes output at a fixed 24fps (write_video_audio(...,
fps=24) in infer.py, and the README's "--num-frames 124" example is
documented as producing "a 5.2-second" clip: 124 / 24 = 5.167s, which matches
only at 24fps). The benchmark requires exactly 5.0s, i.e. 120 frames at
24fps. But infer.py also requires --num-frames to satisfy (num_frames - 5) %
17 == 0 (LATENT_T packing), and 120 is not of that form -- the nearest valid
values are 107 (4.458s) and 124 (5.167s). There is no --num-frames that lands
on exactly 5.0s, so this driver over-generates (124 frames, the same value
used by H3-World's own run.sh/README example) and trims the output down to
exactly 120 frames with ffmpeg, the same re-encode-after-trim approach Zing's
driver uses in drop_reference_frame().

The trim also has to account for keyframe injection: infer.py's pipe() call
passes keyframes=[first_frame], keyframe_indices=[0], which binds the given
first frame to output frame 0 -- i.e. frame 0 of the num_frames-length clip
is (a reconstruction of) the input switch frame, not a genuinely predicted
frame, the same situation Zing's drop_reference_frame() handles. This driver
therefore defaults --frame-offset to 1 (drop frame 0, keep frames 1..120 of
the window) so the scored clip is 5.00s of *predicted* frames, not
input-frame-plus-4.958s-of-prediction. This has NOT been visually confirmed
against a real checkpoint (none was available while writing this driver) --
only inferred from the keyframe_indices=[0] call site and infer.py's docs.
Before trusting a real run, verify with --limit 1: ffprobe the raw clip's
frame count (should be 124) and spot-check whether frame 0 is visually
identical to the switch frame. If it is not (i.e. H3-World does NOT
reconstruct the keyframe into the output), rerun with --frame-offset 0.

sample_id is written as the benchmark's required output filename minus its
extension, and the driver writes directly to "<sample_id>.mp4" via --out, so
no separate renaming pass is needed (unlike Zing, which relies on
zing_v0_5's own output-stem sanitizer).

This lives in the benchmark repo and drives an external H3-World checkout, so
it runs H3-World's own venv interpreter rather than the one running this
script -- H3-World's DiffSynth-Studio-h3-v2 fork and its action/text modules
are not importable from the physiq environment.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_h3.py --run-name h3-world-op-run_01 --limit 1
    python physiq/drive_h3.py --run-name h3-world-op-run_01
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ID_RE = re.compile(r"^(\d{4})_")

OUTPUT_FPS = 24               # infer.py's write_video_audio(..., fps=24)
TARGET_SECONDS = 5.0
TARGET_FRAMES = round(TARGET_SECONDS * OUTPUT_FPS)   # 120
DEFAULT_NUM_FRAMES = 124      # smallest value satisfying (n-5)%17==0 with
                               # n - 1 >= TARGET_FRAMES; matches run.sh's own example


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


def build_jobs(rows: list[dict], frames: dict[str, Path], base_seed: int) -> list[dict]:
    """One infer.py invocation's worth of arguments per row.

    Seed is base_seed + row index rather than one shared seed for every
    sample: infer.py is invoked once per scenario (unlike Zing's single
    batched call), so a constant seed across all 198 samples in a run would
    correlate their initial noise. run_01..run_04 stay independent because
    --seed (the base) differs per run.
    """
    jobs = []
    missing = []
    for i, row in enumerate(rows):
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

        jobs.append({
            "sample_id": Path(out_name).stem,
            "first_frame": frame_path.resolve(),
            "scene_prompt": row["description"],
            "seed": base_seed + i,
        })

    if missing:
        print(f"[physiq-h3] WARNING: skipped {len(missing)} sample(s):", file=sys.stderr)
        for item in missing[:5]:
            print(f"    {item}", file=sys.stderr)
    return jobs


def trim_to_window(video: Path, offset: int, target_frames: int, fps: int) -> bool:
    """Cut `video` down to exactly target_frames, dropping `offset` leading
    frame(s) first.

    Same technique as Zing's drop_reference_frame(): re-encodes rather than
    stream-copying because dropping leading frames moves the keyframe.
    Returns True if the file was rewritten, False if it was already correct
    or if ffprobe/ffmpeg failed (caller is expected to notice via the final
    frame count if that happens).
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
    if offset == 0 and n_frames == target_frames:
        return False  # already correct
    if n_frames < offset + target_frames:
        print(f"[physiq-h3] WARNING: {video.name} has only {n_frames} frames, need "
              f"{offset + target_frames} (offset {offset} + target {target_frames}); "
              "leaving as-is", file=sys.stderr)
        return False

    tmp = video.with_suffix(".trim.mp4")
    vf = f"select=gte(n\\,{offset}),setpts=PTS-STARTPTS" if offset else "setpts=PTS-STARTPTS"
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", vf, "-frames:v", str(target_frames), "-r", str(fps), str(tmp)],
        capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(video)
    return True


def resolve_h3_python(h3_root: Path) -> Path:
    """H3-World's venv interpreter (see setup.sh: python3 -m venv .venv).
    infer.py's DiffSynth-Studio-h3-v2 fork is not importable from physiq's
    env, so sys.executable is the wrong thing to launch."""
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        candidate = h3_root / rel
        if candidate.exists():
            return candidate
    return h3_root / ".venv" / "bin" / "python"


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
    ap.add_argument("--h3-python", type=Path, default=None,
                    help="H3-World venv interpreter (default: <h3-root>/.venv/bin/python)")
    ap.add_argument("--descriptions", type=Path, default=None,
                    help="descriptions CSV (default: <benchmark-root>/descriptions/"
                         "descriptions_original.csv, i.e. the 'op' prompt setting)")
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. h3-world-op-run_01. The benchmark "
                         "expects <model>-<op|bpp>-run_<NN> and averages run_01..run_04.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="where run folders are written (default: "
                         "<benchmark-root>/generated_videos_5s)")
    ap.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                    help="frames requested from infer.py; must satisfy (n-5)%%17==0. "
                         f"Trimmed down to {TARGET_FRAMES} frames ({TARGET_SECONDS:.2f}s "
                         f"@ {OUTPUT_FPS}fps) afterward (default: {DEFAULT_NUM_FRAMES})")
    ap.add_argument("--frame-offset", type=int, default=1,
                    help="leading frames to drop from each raw clip before trimming to "
                         "the target window (default: 1, dropping the reconstructed "
                         "keyframe at output index 0 -- see module docstring; pass 0 if "
                         "--limit 1 verification shows H3-World does not do this)")
    ap.add_argument("--steps", type=int, default=50, help="infer.py --steps")
    ap.add_argument("--cfg-scale", type=float, default=1.0, help="infer.py --cfg-scale")
    ap.add_argument("--subject", choices=("man", "car"), default="man",
                    help="infer.py --subject; 'man' matches the trained vocabulary "
                         "(default), 'car' is out-of-distribution -- see infer.py's help")
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; row i uses seed + i, so vary this per run so "
                         "run_01..run_04 are independent")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N samples -- use --limit 1 to smoke-test")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="LoRA checkpoint (default: <h3-root>/checkpoints/H3-World/"
                         "step-10000.safetensors)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-sample infer.py commands, run nothing (H3-World "
                         "has no batch file to write, so this is the equivalent of "
                         "Zing's --jsonl-only)")
    args = ap.parse_args()

    if (args.num_frames - 5) % 17:
        ap.error(f"--num-frames must be 17k+5 (124, 243, 481, ...), got {args.num_frames}")
    if args.num_frames - args.frame_offset < TARGET_FRAMES:
        ap.error(f"--num-frames {args.num_frames} minus --frame-offset {args.frame_offset} "
                  f"leaves fewer than {TARGET_FRAMES} frames to trim down to")

    bench_root = args.benchmark_root.resolve()
    verified = bench_root / "physics-IQ-benchmark-verified"
    switch_dir = verified / "switch-frames"
    descriptions = args.descriptions or bench_root / "descriptions" / "descriptions_original.csv"
    out_root = args.out_root or bench_root / "generated_videos_5s"
    h3_root = args.h3_root.resolve()
    h3_python = args.h3_python or resolve_h3_python(h3_root)
    infer_py = h3_root / "code" / "abot" / "infer.py"
    checkpoint = args.checkpoint or h3_root / "checkpoints" / "H3-World" / "step-10000.safetensors"

    for label, path in [("switch-frames", switch_dir), ("descriptions", descriptions),
                        ("H3-World checkout", h3_root), ("infer.py", infer_py)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    frames = find_switch_frames(switch_dir)
    rows = load_take1_rows(descriptions)
    if args.limit:
        rows = rows[:args.limit]

    jobs = build_jobs(rows, frames, args.seed)
    if not jobs:
        print("ERROR: no jobs built", file=sys.stderr)
        return 2

    out_dir = out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H3-World -> Physics-IQ Verified (i2v, static camera)")
    print("=" * 60)
    print(f"  descriptions : {descriptions}")
    print(f"  switch frames: {len(frames)} found")
    print(f"  jobs         : {len(jobs)}")
    print(f"  output       : {args.num_frames} frames requested @ {OUTPUT_FPS}fps, "
          f"trimmed to {TARGET_FRAMES} frames ({TARGET_SECONDS:.2f}s), frame-offset {args.frame_offset}")
    print(f"  action       : --action-preset still (static camera)")
    print(f"  base seed    : {args.seed} (per-row: seed + row index)")
    print(f"  h3-world     : {h3_root}")
    print(f"  checkpoint   : {checkpoint}")
    print(f"  out_dir      : {out_dir}")
    print("=" * 60)

    def build_cmd(job: dict, out_path: Path) -> list[str]:
        return [str(h3_python), str(infer_py),
                "--checkpoint", str(checkpoint),
                "--first-frame", str(job["first_frame"]),
                "--scene-prompt", job["scene_prompt"],
                "--action-preset", "still",
                "--num-frames", str(args.num_frames),
                "--steps", str(args.steps),
                "--cfg-scale", str(args.cfg_scale),
                "--subject", args.subject,
                "--seed", str(job["seed"]),
                "--out", str(out_path)]

    if args.dry_run:
        print(f"\n[physiq-h3] --dry-run; {len(jobs)} command(s), e.g.:")
        preview = jobs[:1] if jobs else []
        for job in preview:
            out_path = out_dir / f"{job['sample_id']}.mp4"
            print("  " + " ".join(build_cmd(job, out_path)))
        if len(jobs) > 1:
            print(f"  ... and {len(jobs) - 1} more")
        return 0

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    print()
    produced = []
    trimmed = 0
    for i, job in enumerate(jobs):
        out_path = out_dir / f"{job['sample_id']}.mp4"
        cmd = build_cmd(job, out_path)
        print(f"[physiq-h3] ({i + 1}/{len(jobs)}) {job['sample_id']}")
        result = subprocess.run(cmd, cwd=str(h3_root))
        if result.returncode != 0:
            print(f"ERROR: infer.py failed for {job['sample_id']} (exit {result.returncode})",
                  file=sys.stderr)
            continue
        if not out_path.exists():
            print(f"ERROR: infer.py reported success but {out_path} is missing", file=sys.stderr)
            continue
        if trim_to_window(out_path, args.frame_offset, TARGET_FRAMES, OUTPUT_FPS):
            trimmed += 1
        produced.append(out_path.name)

    if trimmed:
        print(f"[physiq-h3] trimmed {trimmed} clip(s) to {TARGET_FRAMES} frames")

    print(f"\n[physiq-h3] {len(produced)}/{len(jobs)} clip(s) in {out_dir}")
    for name in produced[:3]:
        print(f"    {name}")
    print("\nNext: verify duration is exactly 5.00s, then evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {descriptions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
