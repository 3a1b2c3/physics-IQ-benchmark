#!/usr/bin/env python3
"""Drive Zing-0.5 over the Physics-IQ Verified benchmark (image-to-video mode).

Physics-IQ gives a single "switch frame" per scenario and asks the model to
predict the next 5 seconds. Zing is an interactive world model, so each sample
becomes a session: the switch frame as the reference frame, the benchmark
description as the text prompt, and one keyboard control row per output frame.

Every control row is all-zero. Physics-IQ scenarios are filmed with a locked-off
camera -- every description in the CSV ends "Static shot with no camera
movement." -- so any W/A/S/D would pan the viewpoint and make the generated
clip incomparable to the ground truth. All-zero is the faithful mapping, and it
is off-distribution for a model trained on near-constant gameplay motion: if
rollouts drift or stall, that is a real property of the model on this benchmark,
not a bug in this script.

Frame count: Zing's config/zing.yaml sets output_fps: 24, and the benchmark
requires exactly 5 seconds, so 120 frames lands on 5.0s with no resampling.

sample_id is written as the benchmark's required output filename minus its
extension. zing_v0_5's io.output_stem() sanitises sample_id to [A-Za-z0-9._-]
and writes "<stem>.mp4", and benchmark names use only those characters, so the
generated files already carry the "0001_" ID prefix the evaluator matches on --
no renaming pass.

This lives in the benchmark repo and drives an external Zing checkout, so it
runs Zing's own venv interpreter rather than the one running this script --
zing_v0_5 is not importable from the physiq environment.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_zing.py --run-name zing-0.5-op-run_01 --limit 1
    python physiq/drive_zing.py --run-name zing-0.5-op-run_01
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ACTION_KEYS = ["w", "a", "s", "d", "i", "j", "k", "l"]
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


def build_sessions(rows: list[dict], frames: dict[str, Path], n_frames: int,
                   height: int, width: int, bench_root: Path) -> list[dict]:
    sessions = []
    missing = []
    # One all-zero row per frame, shared by every sample -- json.dump copies it
    # by value, so a single list is safe and keeps the file ~5x smaller.
    idle = [[0] * len(ACTION_KEYS) for _ in range(n_frames)]

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

        sessions.append({
            "schema_version": 2,
            "sample_id": Path(out_name).stem,
            "messages": [
                {"role": "user", "type": "text", "content": row["description"]},
                {
                    "role": "target",
                    "type": "video",
                    "uri": str(frame_path.resolve()),
                    "reference_frame_count": 1,
                    "output": {"frames": n_frames, "height": height, "width": width},
                    "controls": [{
                        "type": "keyboard_direction_frame_interval",
                        "action_keys": ACTION_KEYS,
                        "actions": idle,
                    }],
                },
            ],
        })

    if missing:
        print(f"[physiq-zing] WARNING: skipped {len(missing)} sample(s):", file=sys.stderr)
        for item in missing[:5]:
            print(f"    {item}", file=sys.stderr)
    return sessions


def resolve_zing_assets(base: Path) -> tuple[Path, Path]:
    """Locate (pretrained_dir, checkpoint) for a Zing-0.5 install.

    Two different directories are needed and they are not the same one:
    InferencePipeline requires pretrained_dir to contain text_encoder/,
    tokenizer/ and vae/, while the checkpoint is generator/model.pt one level
    up. In the HF layout that download_models.sh produces those sit at
    models--seedleap--zing-0.5/snapshots/<sha>/{pretrained,generator}, so
    pointing --pretrained-dir at the snapshot root fails with
    "pretrained directory is missing text_encoder/".
    """
    roots = []
    if (base / "generator" / "model.pt").exists():
        roots.append(base)
    roots.extend(p.parent.parent
                 for p in sorted(base.glob("models--*/snapshots/*/generator/model.pt")))

    for root in roots:
        checkpoint = root / "generator" / "model.pt"
        for candidate in (root / "pretrained", root):
            if (candidate / "text_encoder").is_dir():
                return candidate, checkpoint

    return base, base / "generator" / "model.pt"


def resolve_zing_python(zing_root: Path) -> Path:
    """Zing's venv interpreter. zing_v0_5 is not importable from physiq's env,
    so sys.executable is the wrong thing to launch."""
    for rel in (".venv/bin/python", ".venv/Scripts/python.exe"):
        candidate = zing_root / rel
        if candidate.exists():
            return candidate
    return zing_root / ".venv" / "bin" / "python"


def main() -> int:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-root", type=Path, default=repo_root,
                    help="physics-IQ-benchmark checkout (default: this repo)")
    ap.add_argument("--zing-root", type=Path,
                    default=repo_root.parent / "zing-world-model",
                    help="zing-world-model checkout")
    ap.add_argument("--zing-python", type=Path, default=None,
                    help="Zing venv interpreter (default: <zing-root>/.venv/bin/python)")
    ap.add_argument("--descriptions", type=Path, default=None,
                    help="descriptions CSV (default: <benchmark-root>/descriptions/"
                         "descriptions_original.csv, i.e. the 'op' prompt setting)")
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. zing-0.5-op-run_01. The benchmark "
                         "expects <model>-<op|bpp>-run_<NN> and averages run_01..run_04.")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="where run folders are written (default: "
                         "<benchmark-root>/generated_videos_5s)")
    ap.add_argument("--frames", type=int, default=120,
                    help="output frames; 120 = 5.0s at Zing's output_fps of 24")
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0,
                    help="vary per run so run_01..run_04 are independent")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N samples -- use --limit 1 to smoke-test")
    ap.add_argument("--pretrained-dir", type=Path, default=None,
                    help="default: <zing-root>/pretrained_models")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="generator checkpoint (default: <pretrained-dir>/generator/model.pt)")
    ap.add_argument("--local-attn-size", type=int, default=97)
    ap.add_argument("--sink-size", type=int, default=9)
    ap.add_argument("--jsonl-only", action="store_true",
                    help="write the JSONL and print the command, run nothing")
    args = ap.parse_args()

    bench_root = args.benchmark_root.resolve()
    verified = bench_root / "physics-IQ-benchmark-verified"
    switch_dir = verified / "switch-frames"
    descriptions = args.descriptions or bench_root / "descriptions" / "descriptions_original.csv"
    out_root = args.out_root or bench_root / "generated_videos_5s"
    zing_root = args.zing_root.resolve()
    zing_python = args.zing_python or resolve_zing_python(zing_root)
    pretrained_dir, found_checkpoint = resolve_zing_assets(
        args.pretrained_dir or zing_root / "pretrained_models")
    checkpoint = args.checkpoint or found_checkpoint

    for label, path in [("switch-frames", switch_dir), ("descriptions", descriptions),
                        ("zing checkout", zing_root)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    frames = find_switch_frames(switch_dir)
    rows = load_take1_rows(descriptions)
    if args.limit:
        rows = rows[:args.limit]

    sessions = build_sessions(rows, frames, args.frames, args.height, args.width, bench_root)
    if not sessions:
        print("ERROR: no sessions built", file=sys.stderr)
        return 2

    out_dir = out_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "_messages.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session) + "\n")

    print("=" * 60)
    print("Zing-0.5 -> Physics-IQ Verified (i2v, static camera)")
    print("=" * 60)
    print(f"  descriptions : {descriptions}")
    print(f"  switch frames: {len(frames)} found")
    print(f"  sessions     : {len(sessions)}")
    print(f"  output       : {args.frames} frames @ 24fps = {args.frames / 24:.2f}s, "
          f"{args.width}x{args.height}")
    print(f"  actions      : all-zero (static camera)")
    print(f"  seed         : {args.seed}")
    print(f"  zing         : {zing_root}")
    print(f"  checkpoint   : {checkpoint}")
    print(f"  messages     : {jsonl_path}")
    print(f"  out_dir      : {out_dir}")
    print("=" * 60)

    cmd = [str(zing_python), "-m", "zing_v0_5",
           "--pretrained-dir", str(pretrained_dir),
           "--checkpoint", str(checkpoint),
           "--messages", str(jsonl_path),
           "--output-dir", str(out_dir),
           "--local-attn-size", str(args.local_attn_size),
           "--sink-size", str(args.sink_size),
           "--seed", str(args.seed)]

    if args.jsonl_only:
        print("\n[physiq-zing] --jsonl-only; run:")
        print("  " + " ".join(cmd))
        return 0

    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    # zing_v0_5 lives in <zing-root>/src, so it is not importable from the repo
    # root that `python -m zing_v0_5` is run from unless the package happens to
    # be pip-installed into that venv. Put src on PYTHONPATH so this works
    # either way rather than depending on how the venv was built.
    env = os.environ.copy()
    src_dir = str(zing_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing}" if existing else src_dir

    print()
    result = subprocess.run(cmd, cwd=str(zing_root), env=env)
    if result.returncode != 0:
        return result.returncode

    produced = sorted(p.name for p in out_dir.glob("*.mp4"))
    print(f"\n[physiq-zing] {len(produced)}/{len(sessions)} clip(s) in {out_dir}")
    for name in produced[:3]:
        print(f"    {name}")
    print("\nNext: verify duration is exactly 5.00s, then evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {descriptions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
