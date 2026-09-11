#!/usr/bin/env python3
"""Shared plumbing for the per-model Physics-IQ generation drivers.

Every driver has the same three jobs regardless of model: pair the 198 take-1
benchmark rows with their switch frames, name outputs so the evaluator can match
them by ID prefix, and conform each clip to exactly 5.00s. Only the model
invocation differs, so that is all a driver should have to write.
"""
from __future__ import annotations

import csv
import re
import subprocess
import time
from pathlib import Path

ID_RE = re.compile(r"^(\d{4})_")

# The benchmark evaluates 5 seconds. Models that emit 24fps therefore need 120
# frames; models whose frame count is constrained (H3-World requires 17k+5)
# generate the next valid count up and get trimmed back to this.
TARGET_FPS = 24
TARGET_FRAMES = 120


def find_switch_frames(switch_dir: Path) -> dict[str, Path]:
    """Map the 4-digit benchmark ID to its switch frame."""
    frames: dict[str, Path] = {}
    for path in sorted(switch_dir.glob("*.jpg")):
        match = ID_RE.match(path.name)
        if match:
            frames[match.group(1)] = path
    return frames


def load_take1_rows(descriptions_csv: Path) -> list[dict]:
    """The 198 take-1 rows -- the only ones needed for generation.

    take-2 exists to define the benchmark's physical variance (how much two real
    recordings of the same event differ), which is what makes 100 the ceiling.
    It is not something a model generates against.
    """
    with open(descriptions_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [r for r in rows if "take-1" in r["scenario"]]


def iter_samples(rows: list[dict], frames: dict[str, Path]):
    """Yield (bench_id, out_stem, description, switch_frame) per scoreable row.

    out_stem is the benchmark's required output filename minus its extension;
    writing the clip as "<out_stem>.mp4" keeps the "0001_" ID prefix the
    evaluator matches on, so no renaming pass is needed.
    """
    skipped = []
    for row in rows:
        out_name = row["generated_video_name"]
        match = ID_RE.match(out_name)
        if not match:
            skipped.append(f"{out_name} (no ID prefix)")
            continue
        bench_id = match.group(1)
        frame = frames.get(bench_id)
        if frame is None:
            skipped.append(f"{out_name} (no switch frame for ID {bench_id})")
            continue
        yield bench_id, Path(out_name).stem, row["description"], frame
    if skipped:
        print(f"[physiq] WARNING: skipped {len(skipped)} row(s); first few:")
        for item in skipped[:5]:
            print(f"    {item}")


def probe_frame_count(video: Path) -> int | None:
    result = subprocess.run(
        ["ffprobe", "-hide_banner", "-loglevel", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def conform_video(video: Path, target_frames: int = TARGET_FRAMES,
                  drop_leading: bool = False) -> bool:
    """Make `video` exactly target_frames long at TARGET_FPS, in place.

    Physics-IQ rejects any duration other than exactly 5 seconds, and models
    overshoot for different reasons: Zing prepends the reference frame it was
    given (so 120 requested -> 121), H3-World can only emit 17k+5 frames (so 124
    is the nearest above 120). drop_leading handles the former -- that first
    frame is model *input*, and scoring it against predicted ground truth would
    compare a copy of the conditioning frame.

    Re-encodes rather than stream-copying: dropping a leading frame moves the
    keyframe. Returns True if the file was rewritten.
    """
    n_frames = probe_frame_count(video)
    if n_frames is None or n_frames == target_frames:
        return False

    select = r"select=gte(n\,1)," if drop_leading else ""
    tmp = video.with_suffix(".conform.mp4")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         # setpts rebases timestamps; -vsync/-fps_mode must not be combined with
         # -r on ffmpeg 7 ("contradictory"), and sources here are already CFR.
         "-vf", f"{select}setpts=PTS-STARTPTS",
         "-frames:v", str(target_frames), "-r", str(TARGET_FPS), str(tmp)],
        capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        print(f"[physiq] WARNING: could not conform {video.name}: "
              f"{result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'ffmpeg failed'}")
        return False
    tmp.replace(video)
    return True


def conform_run(out_dir: Path, target_frames: int = TARGET_FRAMES,
                drop_leading: bool = False) -> tuple[int, int]:
    """Conform every clip in a run folder. Returns (rewritten, total)."""
    clips = sorted(out_dir.glob("*.mp4"))
    rewritten = sum(1 for c in clips if conform_video(c, target_frames, drop_leading))
    return rewritten, len(clips)


def resolve_paths(bench_root: Path, descriptions: Path | None, out_root: Path | None,
                  run_name: str) -> tuple[Path, Path, Path]:
    """Standard benchmark path layout shared by every driver."""
    switch_dir = bench_root / "physics-IQ-benchmark-verified" / "switch-frames"
    csv_path = descriptions or bench_root / "descriptions" / "descriptions_original.csv"
    out_dir = (out_root or bench_root / "generated_videos_5s") / run_name
    return switch_dir, csv_path, out_dir


def print_banner(model: str, csv_path: Path, n_frames_found: int, n_sessions: int,
                 frames: int, width: int, height: int, seed: int, out_dir: Path,
                 extra: dict | None = None) -> None:
    print("=" * 60)
    print(f"{model} -> Physics-IQ Verified (i2v, static camera)")
    print("=" * 60)
    print(f"  descriptions : {csv_path}")
    print(f"  switch frames: {n_frames_found} found")
    print(f"  samples      : {n_sessions}")
    print(f"  output       : {frames} frames @ {TARGET_FPS}fps, {width}x{height}")
    print(f"  actions      : static camera (no movement)")
    print(f"  seed         : {seed}")
    for key, value in (extra or {}).items():
        print(f"  {key:<13}: {value}")
    print(f"  out_dir      : {out_dir}")
    print("=" * 60)


def fmt_duration(seconds: float) -> str:
    """H:MM:SS for anything over an hour, else M:SS -- run lengths here span
    both (a 1-clip smoke test and a 198-clip pass)."""
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class RunTimer:
    """Per-sample timing with a running mean and ETA.

    The mean is over completed samples rather than the last one: generation time
    varies enough between clips that a single-sample estimate swings wildly and
    is useless for deciding whether to wait.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.started = time.monotonic()
        self.done = 0
        self._last = self.started

    def start_item(self) -> None:
        self._last = time.monotonic()

    def finish_item(self, label: str = "", ok: bool = True) -> str:
        now = time.monotonic()
        item = now - self._last
        self.done += 1
        elapsed = now - self.started
        mean = elapsed / self.done
        remaining = max(self.total - self.done, 0)
        eta = f", eta {fmt_duration(mean * remaining)}" if remaining else ""
        status = "" if ok else " FAILED"
        return (f"[{self.done}/{self.total}] {label}{status} "
                f"{item:.1f}s (mean {mean:.1f}s, elapsed {fmt_duration(elapsed)}{eta})")

    def summary(self) -> str:
        elapsed = time.monotonic() - self.started
        if not self.done:
            return f"total {fmt_duration(elapsed)}"
        return (f"total {fmt_duration(elapsed)} for {self.done} clip(s), "
                f"mean {elapsed / self.done:.1f}s each")
