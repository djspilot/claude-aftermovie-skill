---
name: aftermovie
description: Create GoPro-style beat-synced aftermovies on macOS from a folder of mixed footage (GoPro clips, iPhone video, Live Photos, drone footage, etc.) plus a song. Triggers whenever the user wants to make a montage, highlight reel, aftermovie, recap video, travel video, sports edit, or beat-synced video, and whenever they mention assembling clips automatically. Use this skill even if the user does not say the word "aftermovie" — phrases like "make a video from my trip clips", "edit this folder to music", "GoPro-style edit", "highlight my best moments", or "auto-edit these clips" all mean this skill. Runs locally on the user's Mac via ffmpeg + librosa + GPMF telemetry, no cloud.
---

# Aftermovie

A skill that turns a folder of video clips and a song into a beat-synced highlight video — the GoPro Quik recipe, running natively on the user's Mac.

## When this skill applies

Use this skill whenever the user wants to assemble multiple clips into a single edited video paced to music. The user does not need to use the word "aftermovie." Trigger phrases include:

- "Make me a video from my trip footage"
- "Edit this folder to <song>"
- "Make a GoPro-style edit"
- "Auto-cut these clips to the beat"
- "Highlight reel of <event>"

Do NOT use this skill for single-clip operations (trimming one video, applying a filter to one file, converting formats). Use plain `ffmpeg` for those.

## Quick start

The skill exposes one Python CLI at `scripts/aftermovie.py`. The fastest path is the `auto` subcommand — give it a folder and a song, get back a video.

```bash
python3 scripts/aftermovie.py auto \
  --clips ~/Movies/Iceland2026 \
  --song ~/Music/song.mp3 \
  --output ~/Movies/iceland_aftermovie.mp4
```

Behind the scenes this runs three stages: `analyze`, `score`, and `render`. The user can also run them separately if they want to inspect the plan first (see "Inspecting the plan" below).

## First-run setup

On the first invocation, run `scripts/setup.sh`. It checks for `ffmpeg`, `python3 ≥ 3.10`, and installs the Python deps into a local venv under `.skills-data/aftermovie/venv` so the user's global Python stays clean. The setup script is idempotent — safe to run again.

```bash
bash scripts/setup.sh
```

If `ffmpeg` is missing, the setup script prints the Homebrew install command and exits. Don't try to install ffmpeg automatically — confirm with the user first.

## How the pipeline works

The skill follows the editing recipe described in `references/recipe.md` (read that file when the user asks why a specific choice was made, or when tuning the output). The short version:

1. **Analyze** — scan every video file in the input folder. For each clip, extract:
   - For GoPro files: GPMF telemetry (gyro, accel, GPS speed) + HiLight tags from the HMMT atom.
   - For all clips: scene-cut points (ffmpeg `select` filter), audio RMS energy (voices/cheering proxy), motion magnitude (frame diff), and metadata-declared frame rate (so high-fps clips can be flagged for slow-mo).
   - Write everything to `catalog.json` with one entry per discovered sub-clip candidate (1-5 seconds each).

2. **Score** — analyze the song with `librosa`: tempo, beat positions, downbeats, structural boundaries (intro/verse/chorus). Score each candidate sub-clip and assign winners to beat positions using a greedy fill, with the first cut on the first downbeat after the intro. Output: `plan.json`.

3. **Render** — execute the plan via ffmpeg. Each picked sub-clip is trimmed, optionally speed-ramped (high-fps clips get auto-slowed at action peaks), color-graded with a `.cube` LUT, then concatenated. Music goes on top, original clip audio ducked underneath at -18 dB.

## Defaults to lean on

When the user just says "make me a video" without specifying anything else, pick sensible defaults:

- **Output length**: trim to the song length, or cap at 90 seconds, whichever is shorter.
- **Aspect ratio**: 16:9 unless the user mentions Instagram/TikTok/Reels (then 9:16 vertical).
- **LUT**: `assets/luts/cinematic.cube` — neutral, slightly cool, lifted blacks. Good default for action footage.
- **Music volume**: -8 dB. Clip audio ducked to -18 dB. Tweak with `--music-db` and `--clip-db`.
- **Resolution**: 1080p. Use `--res 4k` for 2160p if the user wants it.

For Live Photos / Motion Photos: these come in as short ~3s videos. The analyzer detects them by duration + frame rate and the scorer treats them as "punctuation" — short freeze-frames between longer clips, not main beats.

## Inspecting the plan before rendering

If the user wants to see what the edit *will* look like before committing to a render (which can take a few minutes for long source folders), split the workflow:

```bash
python3 scripts/aftermovie.py analyze --clips <folder> --out catalog.json
python3 scripts/aftermovie.py score --catalog catalog.json --song <track> --out plan.json
# show plan.json to user, let them tweak
python3 scripts/aftermovie.py render --plan plan.json --output out.mp4
```

`plan.json` is human-readable: each entry is `{source, start_ms, end_ms, speed, beat_time, score, reason}`. The `reason` field explains why this clip was picked (e.g. `"hilight_tag"`, `"high_accel_jump"`, `"smile_detected"`, `"motion_peak"`) — useful for explaining to the user why their favorite clip didn't make the cut, or for manual edits.

## Tuning knobs

The `auto` command takes optional flags:

- `--theme cinematic|punchy|chill|nostalgic` — preset pacing + LUT bundle (see `references/themes.md`)
- `--max-length 60` — cap output length in seconds
- `--aspect 16:9|9:16|1:1` — output aspect ratio
- `--lut <path>` — override the color LUT
- `--no-speed-ramp` — disable auto slow-motion on high-fps clips
- `--seed 42` — for reproducible runs (the greedy scorer has a small randomness budget to avoid identical edits)

If the user wants something the flags don't cover (custom transition style, specific clip ordering, locked-in clips), edit `plan.json` directly and re-run the `render` stage — much faster than tweaking flags.

## Failure modes to watch for

- **No clips found**: the folder probably has clips in unsupported formats. The analyzer supports `.mp4`, `.mov`, `.m4v`, `.heic` (Live Photos), `.insv`, `.lrv`. Print the formats it found and ignored.
- **Song shorter than the target output**: the scorer falls back to the song's actual length and ignores `--max-length`. Tell the user.
- **All clips have very similar motion profiles** (e.g. all locked-off interview shots): the scorer will produce a flat-feeling edit. Suggest the user either mix in B-roll or pick a more varied song.
- **Mixed frame rates causing judder**: the renderer normalizes everything to 30 fps by default. Override with `--fps 24` or `--fps 60` if the user has a preference.

## Why the skill is structured this way

The `analyze → score → render` split exists because analysis is slow (1-2 minutes per 10 GB of source footage) but the score and render stages are fast (seconds). Caching the catalog means iterating on the look (different song, different theme) doesn't repeat the expensive work.

For the actual editing logic (why we cut on downbeats, why slow-mo lands on action peaks, why the first cut waits for the drop), see `references/recipe.md`. For theme presets, see `references/themes.md`. For the GPMF telemetry format, see `references/gpmf.md`.
