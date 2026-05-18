# Aftermovie — domain language

This file defines the terms the codebase uses. Match them when extending or refactoring; coin a new term only when the conversation has run out of ways to use the existing ones.

## Domain concepts

- **Aftermovie** — the final mp4 output: a beat-synced highlight cut from one **Source folder** + one **Song**. Always exactly one of each per render. ("Edit," "video," "reel," "montage" — don't use these; we made the user pick *aftermovie* because the recipe is GoPro Quik's.)

- **Source folder** — a directory of mixed footage the user points at. Contents are: native video clips, paired Live Photo MOVs, standalone HEIC/JPG/PNG stills, and (sometimes) the song itself.

- **Song** — exactly one audio file used both for the music bed AND as the timing reference for beat sync. Not "soundtrack," not "background music."

- **Clip** — a single source file (video OR a materialized still). Always referenced by its file path. **Not** a sub-segment — that's a *Candidate*. Not a finished cut — that's an *Entry*.

- **Catalog** — the analyze-step output: one ClipInfo per Clip with per-second feature lists (motion, audio, voice, accel, GPS speed) plus metadata. JSON on disk; the input to scoring.

- **Candidate** — a scored sub-clip of a Clip, in source-time (start_s, end_s). 1–2 second windows by default; the scorer ranks them.

- **Plan** — the score-step output. Ordered list of Entries (a.k.a. plan entries) the renderer will emit. JSON on disk.

- **Entry** — one slot in the Plan: which source, which window, what speed, what transition into it, what audio interest, what beat it lands on. (Not "cut," not "shot.")

- **Cut** — the *act* of placing an Entry on a beat. "Build N cuts" = the planner produced N entries. The boundary itself between two entries is also a *cut* (vs. a *crossfade*, *whip*, etc.).

- **Beat slot** — the time interval between two adjacent beat points the planner uses. An Entry's `out_duration_s` is the length of its beat slot.

- **Transition** — what happens at a cut boundary. Kinds: `cut` (hard), `crossfade` (xfade=fade), `whip` (xfade=wipe). Each has a `duration_s`.

- **Pace** — how dense cuts are: `fast` (every beat), `medium` (every 2nd), `slow` (every 4th = downbeats), `auto` (energy-aware, varies per song section).

- **Theme** — a preset bundle of look-and-feel knobs (LUT, music_db, pace, transitions, audio_mix). Names: `cinematic`, `punchy`, `chill`, `nostalgic`. Not "style" or "preset."

- **Live Photo** — an iPhone HEIC paired with a MOV that carries the brief motion. Pairs can be **paired-file** (`IMG_xxxx.HEIC` + same-stem `IMG_xxxx.MOV`) or **single-file** (HEIC with the MOV in a metadata box; requires exiftool).

- **Still variant** — the camera-move applied when a still becomes a 2.5s mp4. Variants: `live`, `push`, `pull`, `pan_h`, `fit_pad`, `blurred_bg`, `shake`. Picked deterministically per filename.

- **Ducked mix** — the audio mode that sidechain-compresses the Song against the clip-audio voice band so people talking in the source files surface over the music. Other modes: `music_only`, `clip_only`.

- **Audio interest** — the per-Entry mean voice-band RMS (200–3000 Hz). The renderer mutes clip audio for Entries below threshold so the duck trigger stays clean.

- **HiLight tag** — a millisecond timestamp emitted by GoPro firmware when the user presses the highlight button mid-record. Stored in the HMMT atom and parsed at analyze time.

- **GPMF** — GoPro Metadata Format: per-frame telemetry (accel, gyro, GPS speed) embedded in the MP4. Parsed at analyze time and contributes to scoring.

- **Score component** — one named signal that contributes to a Candidate's total score (`motion`, `audio`, `hilight_tag`, `accl_jump`, `gps_speed`, `face`, `user_favorite`, `blurry`, `poor_exposure`). Declared once in `score/components.py`; the scorer threads `ScoreComponent` instances (not raw strings) into the `Candidate.components` / `PlanEntry.components` breakdown so the "why this entry won?" inspector has a fixed vocabulary. Not "score field", not "signal" (signal is the analyze-time input — `motion_energy`, `accl_peaks`, ...; the *component* is what the scorer derives from it).

## Surfaces

- **CLI** — `aftermovie` command, argparse subcommands (`analyze`, `score`, `render`, `auto`, `doctor`, `init-config`, `show-config`).
- **MCP server** — `aftermovie-mcp` binary exposing the same flow as Claude Code tools (`analyze_folder`, `propose_plan`, `render_plan`, ...). Both surfaces should drive the same `pipeline_runner`.
- **Skill** — the `aftermovie` Claude Code skill (`~/.claude/skills/aftermovie/`) that documents user-facing recipes.

## Pipeline phases (the three-act structure of every render)

1. **Analyze** — walk the source folder, materialize stills with their picked variant, extract GPMF + HiLight + per-second motion/audio/voice energy, write a Catalog.
2. **Score** — read the Catalog and the Song's tempo/beats/energy curve, generate Candidates, rank them, allocate to beat slots, decide speed ramps, write a Plan.
3. **Render** — read the Plan, prerender each Entry as an mp4 (aspect + LUT + speed + audio fade + audio-interest gate), assemble via concat or filter_complex (xfade + acrossfade), final-mux against the Song with the chosen audio mode, write the Aftermovie.

## What this project does NOT have

- Multi-song edits.
- Real-time preview / scrubbing.
- Cloud rendering.
- Speech-to-text captions.
- Color grading beyond a 3D LUT.

If you find code that implies these exist, treat it as a bug. If a user asks for them, treat the request as out-of-scope (or write an ADR before saying yes).
