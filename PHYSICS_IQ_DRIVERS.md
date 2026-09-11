# Driving world models through Physics-IQ Verified

Six models are wired up. Each generates 198 clips per run, scores with
`physiq/run_physics_iq.py`, and reports a verified Physics-IQ score.

## Layout

```
physiq/physiq_common.py     shared: CSV rows, switch-frame pairing, naming, 5.00s conform
physiq/drive_zing.py        Zing-0.5          (standalone zing-world-model checkout)
physiq/drive_echo.py        Echo-WM           (JoyAI-Echo checkout)
physiq/drive_h3world.py     H3-World          (H3-World checkout)
physiq/drive_abot.py        ABot-World        (ABot-World checkout)
physiq/drive_cam2v.py       LingBot / HY-WorldPlay / SANA-WM  (flashdreams integrations_v2)
scripts/drive_<model>.sh    wrapper per model; .bat siblings for the Windows-native ones
```

Wrappers pin the model, so which script you start selects what runs -- there is
no `--app`-style flag to forget. Everything after the script name is forwarded
to the driver.

## Running

```bash
bash scripts/drive_echo.sh        --run-name echo-op-run_01 --limit 1   # smoke test
bash scripts/drive_echo.sh        --run-name echo-op-run_01             # full 198
bash scripts/drive_h3world.sh     --run-name h3world-op-run_01
bash scripts/drive_zing.sh        --run-name zing-0.5-op-run_01
bash scripts/drive_abot.sh        --run-name abot-op-run_01
bash scripts/drive_lingbot.sh     --run-name lingbot-op-run_01
bash scripts/drive_hy_worldplay.sh --run-name hy-worldplay-op-run_01
bash scripts/drive_sana_wm.sh     --run-name sana-wm-op-run_01
```

`--skip-existing` is on by default, so an interrupted run resumes rather than
restarting. **That also means stale bad clips survive a rerun** -- delete them
first if a previous attempt produced wrong output.

For a leaderboard score, generate four runs per model (`run_01`..`run_04`) with
different `--seed`, then aggregate:

```bash
uv run physiq/aggregate_runs_from_csvs.py <out>/<model>-op-run_0{1,2,3,4}.csv --score-type verified
```

## The static camera

Physics-IQ is filmed with a locked-off camera -- every description in
`descriptions_original.csv` ends *"Static shot with no camera movement."* Any
camera motion makes the clip incomparable to the ground truth, so every driver
maps "do not move" into its model's own idiom:

| model | static camera |
| --- | --- |
| Zing | all-zero `[frames, 8]` keyboard control rows |
| Echo | `--action-str none-<frames>` (`none` = empty key set) |
| H3-World | all-zero `[frames, 17]` action matrix |
| ABot | action JSON with all 8 keys `false` for every frame |
| Cam2V models | identity 4x4 pose per frame |

This is off-distribution for models trained on near-constant gameplay motion.
If rollouts drift or stall, that is a property of the model on this benchmark,
not a bug in the driver.

**Intrinsics cannot be synthesised.** The Cam2V drivers need a real calibration
and default to `../lingbot-world-v2/examples/00/intrinsics.npy`. Whether that
calibration suits Physics-IQ footage is a judgement worth making before a full
run.

## Frame counts: every model needs conforming

The benchmark requires **exactly 5.00 seconds** and rejects anything else. No
model emits that natively:

| model | native | why | conform |
| --- | --- | --- | --- |
| Zing | 121 @ 24fps | prepends the given reference frame | drop leading frame -> 120 |
| Echo | 121 @ 24fps | quantises to 8k+1 | trim -> 120 |
| H3-World | 124 @ 24fps | requires 17k+5 | trim -> 120 |
| ABot | 60 @ 12fps | 5 blocks x 12 frames | resample -> 120 @ 24 |
| Cam2V | 80 @ 16fps | 20 blocks x 4 (SANA-WM: 10 blocks) | resample -> 120 @ 24 |

`conform_run()` handles all of it at the end of a run. Two rules it encodes:

- **Zing's leading frame is dropped, not trimmed from the tail.** That frame is
  the switch frame the model was *given*; scoring it against predicted ground
  truth would compare a copy of the input.
- **Duration, not frame count, decides whether a clip is short.** 80 frames at
  16fps is a full 5.00s and resamples fine. A clip under 5.00s cannot be fixed
  and is reported rather than silently passed on.

### Echo's 8k+1 trap

Asking Echo for 120 frames yields **113** (8x14+1) and a 4.71s clip -- it rounds
*down* to its latent stride. A whole 198-clip run was generated that way and had
to be discarded. `valid_frame_count()` now requests the next 8k+1 at or above
the target (121), and conform trims one frame.

Check this for any new model: request N, verify you got N.

### Echo's sidecar files

`inference_wm.py` writes `<name>_action.mp4` beside every clip -- an overlay
carrying the same `0001_` benchmark ID prefix the evaluator matches on, so each
scenario would have two candidate files. `prune_sidecars()` removes them after
generation; `run_clips()` excludes them from every count.

## Batching

Model load dominates wall clock. Where a batch entrypoint exists, the driver
uses it:

| model | mode | note |
| --- | --- | --- |
| Zing | one process | `--mind_batch` manifest |
| ABot | one process | `--mind-batch` manifest, ~24GB loaded once |
| LingBot (standalone) | one process | `--mind_batch`, ~14B loaded once |
| Echo | per clip | no batch entrypoint; reloads ~47.8GB + Gemma each time |
| H3-World | per clip | no batch entrypoint |
| Cam2V models | per clip | Cam2V is built for interactive streaming; mp4 is its only non-interactive output |

Echo runs ~52s/clip, so a full 198 is roughly 2h45m.

## One GPU job at a time

Concurrent generation is what OOM'd this box before, and an OOM there is not
recoverable by killing processes: ViPE/driver threads wedge inside the CUDA
driver in `D` state, `kill -9` leaves unreapable zombies holding VRAM, and only
a reboot clears it. Check before starting:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

## Checkpoints

```bash
bash scripts/download_hy_worldplay.sh   # tencent/HY-WorldPlay + Wan-AI/Wan2.2-TI2V-5B
bash scripts/download_sana_wm.sh        # Efficient-Large-Model/SANA-WM_{streaming,bidirectional}
```
(both live in the flashdreams checkout). They check the HF cache first and reuse
anything present. `HF_TOKEN` only matters for gated repos.

## Licensing

**H3-World results need a legal check before publication.** It is a LoRA on
MiniMax-H3, whose Community License grants rights only within its "Applicable
Territory" -- which excludes the USA, EU, UK and South Korea -- and SS V.4
extends that restriction to the model's *Outputs*, meaning generated clips and
any score derived from them. See the LICENSE shipped with the weights. The other
models carry no comparable restriction.

## Status

| model | driver | validated |
| --- | --- | --- |
| Zing | drive_zing.py | one clip generated and inspected |
| Echo | drive_echo.py | full run in progress |
| H3-World | drive_h3world.py | partial run in progress |
| ABot | drive_abot.py | logic only -- never invoked the model |
| LingBot / HY-WorldPlay / SANA-WM | drive_cam2v.py | dry-run only -- never invoked the model |

No model has been through `run_physics_iq.py` yet; there are no Physics-IQ
scores.
