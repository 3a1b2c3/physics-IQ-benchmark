#!/usr/bin/env python3
"""Drive H3-World over the Physics-IQ Verified benchmark (image-to-video mode).

H3-World is a LoRA on MiniMax-H3, driven by a [num_frames, 17] action matrix.
For Physics-IQ that matrix is all zeros: every benchmark description ends
"Static shot with no camera movement.", so any key column would pan the camera
and make the clip incomparable to the ground truth.

Frame count: infer.py writes at fps=24 and requires num_frames == 17k+5, which
cannot equal the 120 frames that 5.00s needs (120-5=115, not a multiple of 17).
The nearest valid count above is 124, so this generates 124 and conforms down to
120 afterwards -- generating 107 (the nearest below) would fall short of the 5s
the benchmark requires and could not be padded honestly.

LICENSING: the MiniMax H3 Community License grants rights only within its
"Applicable Territory", which excludes the USA, EU, UK and South Korea (SS I.5,
II), and SS V.4 extends that to the model's *Outputs*. Generated clips and any
score derived from them fall under it. Check with legal before publishing or
submitting H3-World results from an excluded territory. Not legal advice; see
the LICENSE shipped with the weights.

Usage (from the physics-IQ-benchmark checkout):
    python physiq/drive_h3world.py --run-name h3world-op-run_01 --limit 1
    python physiq/drive_h3world.py --run-name h3world-op-run_01
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

# abot_action.ACTION_DIM. Hardcoded rather than imported so this driver needs no
# H3-World dependencies of its own.
ACTION_DIM = 17


def write_static_actions(h3_python: Path, path: Path, frames: int, dim: int) -> bool:
    """Write an all-zero [frames, dim] action matrix.

    Done through H3-World's interpreter rather than importing numpy here: this
    driver lives in the benchmark repo, whose environment has no numpy, and
    adding one just to allocate a zero array would be a dependency for nothing.
    """
    code = (f"import numpy as np; "
            f"np.save(r'{path}', np.zeros(({frames}, {dim}), dtype=np.float32))")
    result = subprocess.run([str(h3_python), "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: could not write action matrix: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def valid_frame_count(minimum: int) -> int:
    """Smallest 17k+5 >= minimum (infer.py's latent_t_for constraint)."""
    n = minimum
    while (n - 5) % 17:
        n += 1
    return n


def resolve_h3_python(h3_root: Path) -> Path:
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
    ap.add_argument("--benchmark-root", type=Path, default=repo_root)
    ap.add_argument("--h3-root", type=Path, default=repo_root.parent / "H3-World",
                    help="H3-World checkout")
    ap.add_argument("--h3-python", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="LoRA checkpoint (default: <h3-root>/checkpoints/H3-World/step-10000.safetensors)")
    ap.add_argument("--descriptions", type=Path, default=None)
    ap.add_argument("--run-name", required=True,
                    help="model-run folder, e.g. h3world-op-run_01")
    ap.add_argument("--out-root", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bench_root = args.benchmark_root.resolve()
    h3_root = args.h3_root.resolve()
    h3_python = args.h3_python or resolve_h3_python(h3_root)
    checkpoint = args.checkpoint or h3_root / "checkpoints" / "H3-World" / "step-10000.safetensors"
    switch_dir, csv_path, out_dir = resolve_paths(bench_root, args.descriptions,
                                                  args.out_root, args.run_name)

    for label, path in [("switch-frames", switch_dir), ("descriptions", csv_path),
                        ("H3-World checkout", h3_root)]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    gen_frames = valid_frame_count(TARGET_FRAMES)

    frames_map = find_switch_frames(switch_dir)
    rows = load_take1_rows(csv_path)
    if args.limit:
        rows = rows[:args.limit]
    samples = list(iter_samples(rows, frames_map))
    if not samples:
        print("ERROR: no samples", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    actions_dir = out_dir / "_actions"
    actions_dir.mkdir(exist_ok=True)

    # One all-zero matrix serves every sample: no keys pressed, no rotation or
    # translation, for the whole clip.
    action_path = actions_dir / f"static_{gen_frames}.npy"
    if not write_static_actions(h3_python, action_path, gen_frames, ACTION_DIM):
        return 2

    print_banner("H3-World", csv_path, len(frames_map), len(samples), gen_frames,
                 "native", "native", args.seed, out_dir,
                 {"h3": h3_root, "checkpoint": checkpoint.name, "steps": args.steps,
                  "actions": f"all-zero [{gen_frames}, {ACTION_DIM}]",
                  "conform": f"{gen_frames} -> {TARGET_FRAMES} frames (5.00s)"})

    entry = h3_root / "code" / "abot" / "infer.py"
    generated = skipped = failed = 0
    timer = RunTimer(len(samples))

    for index, (bench_id, stem, description, frame) in enumerate(samples, 1):
        target = out_dir / f"{stem}.mp4"
        if args.skip_existing and target.exists():
            skipped += 1
            continue

        cmd = [str(h3_python), str(entry),
               "--checkpoint", str(checkpoint),
               "--first-frame", str(frame.resolve()),
               "--scene-prompt", description,
               "--action-file", str(action_path.resolve()),
               "--num-frames", str(gen_frames),
               "--steps", str(args.steps),
               "--cfg-scale", str(args.cfg_scale),
               "--seed", str(args.seed),
               "--out", str(target)]

        if args.dry_run:
            print("\n[physiq-h3world] --dry-run; first command:")
            print("  " + " ".join(cmd))
            return 0

        timer.start_item()
        result = subprocess.run(cmd, cwd=str(h3_root))
        ok = result.returncode == 0 and target.exists()
        print(f"[physiq-h3world] " + timer.finish_item(stem, ok))
        if not ok:
            failed += 1
            continue
        generated += 1

    rewritten, total = conform_run(out_dir, TARGET_FRAMES)
    print(f"\n[physiq-h3world] generated {generated}, skipped {skipped}, failed {failed}")
    print(f"[physiq-h3world] conformed {rewritten}/{total} clip(s) from {gen_frames} to "
          f"{TARGET_FRAMES} frames @ {TARGET_FPS}fps")
    print(f"\nNext: evaluate with")
    print(f"  uv run physiq/run_physics_iq.py --input_folders {out_dir} "
          f"--output_folder <dir> --descriptions_file {csv_path}")
    return 1 if failed and not generated else 0


if __name__ == "__main__":
    raise SystemExit(main())
