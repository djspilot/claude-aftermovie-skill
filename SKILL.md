---
name: aftermovie
description: Create GoPro-style beat-synced aftermovies from a folder of mixed footage (GoPro clips, iPhone video, Live Photos, drone, screen recording) plus a song. Triggers whenever the user wants to make a montage, highlight reel, aftermovie, recap, travel video, sports edit, or beat-synced video, and whenever they mention assembling clips automatically. Use this skill even if the user does not say "aftermovie" — "make a video from my trip clips", "edit this folder to music", "GoPro-style edit", "highlight my best moments", or "auto-edit these clips" all count. Runs locally on the user's machine via ffmpeg + librosa + GPMF telemetry, no cloud. Exposes both an MCP server (preferred when connected) and a CLI fallback.
---

# Aftermovie

A skill that turns a folder of video clips and a song into a beat-synced highlight video — the GoPro Quik recipe, running natively.

## When this skill applies

Use this skill whenever the user wants to assemble multiple clips into a single edited video paced to music. Trigger phrases:

- "Make me a video from my trip footage"
- "Edit this folder to <song>"
- "Make a GoPro-style edit"
- "Auto-cut these clips to the beat"
- "Highlight reel of <event>"

Do NOT use this skill for single-clip operations (trimming one video, applying a filter, format conversion). Use plain `ffmpeg` for those.

## First-run setup

If the user hasn't installed yet, point them at:

```bash
bash scripts/setup.sh
```

The setup script creates a venv at `~/.skills-data/aftermovie/venv`, installs the package with MCP support, registers an entry in `~/.claude/.mcp.json`, and runs `aftermovie doctor`. Idempotent.

After setup the user must restart Claude Code once for the MCP server to spawn.

## Preferred workflow — MCP

When the MCP server is connected, tools prefixed `mcp__aftermovie__*` are available. Use them for the whole flow. Do NOT fall back to Bash if MCP is available.

1. `mcp__aftermovie__list_themes` — if the user is vague, show the four themes and let them pick.
2. `mcp__aftermovie__analyze_folder({path})` — returns `{job_id, catalog_id, cached}`. If `job_id` is set, poll `get_job` until `status == "done"`. If `cached` is true, the catalog already exists and you can skip straight to step 3.
3. `mcp__aftermovie__propose_plan({catalog_id, song_path, theme, target_length_s?, aspect})` — synchronous, returns `{plan_id, summary: {n_cuts, total_length_s, sources_used, bpm}}`. Show the summary to the user.
4. `mcp__aftermovie__get_plan({plan_id})` — fetch full entries if the user wants to see them.
5. If the user wants changes, `mcp__aftermovie__tweak_plan({plan_id, ops})` returns a new `plan_id`. Supported ops:
   - `{op: "exclude_source", source}` — drop all cuts from one source file
   - `{op: "set", path, value}` — e.g. `path="music_db", value=-12` or `path="aspect", value="9:16"`
   - `{op: "swap", beat_index, with_candidate_rank}` — replace one cut with the next-highest-scoring alternative
   - `{op: "pin", beat_index, source, start_s, end_s}` — lock a specific source segment to a beat
6. `mcp__aftermovie__render_plan({plan_id, output_path})` — returns `{job_id}`. Poll `get_job` until done. Result includes `output_path`, `duration_s`, and `streams`.

Use `cancel_job` if the user changes their mind during a long render.

## CLI fallback

If the MCP server is not connected, use the CLI via Bash. Same pipeline, slightly more verbose.

```bash
~/.skills-data/aftermovie/venv/bin/aftermovie auto \
  --clips <folder> --song <track> --output <out.mp4> [--theme <name>]
```

Or staged:

```bash
aftermovie analyze --clips <folder> --out catalog.json
aftermovie score   --catalog catalog.json --song <track> --out plan.json
aftermovie render  --plan plan.json --output out.mp4
```

## Theme-prompt parsing

When the user describes a vibe rather than naming a theme, map it:

- "epic" / "hype" / "intense" → `punchy`
- "cinematic" / "film" / "moody" → `cinematic`
- "vibe" / "chill" / "aesthetic" / "lofi" → `chill`
- "throwback" / "memories" / "old school" → `nostalgic`
- "Instagram" / "TikTok" / "Reels" / "vertical" → `aspect: "9:16"`
- "YouTube" / "widescreen" → `aspect: "16:9"`
- "square" → `aspect: "1:1"`

## When to ask vs. assume

Ask up-front ONLY if missing:

- Clip folder path
- Song path

Everything else has a sensible default. Show the plan summary (n_cuts, total_length_s, bpm, sources_used) before rendering so the user can intervene. Don't ask "should I render now?" — propose, summarize, render.

## User-editable defaults (env file)

`aftermovie init-config` writes `~/.aftermovie/aftermovie.env` with every knob the user might want to set as a default: theme, aspect, pace, transitions, audio_mix, music_db, etc. The CLI loads this file at startup so bare `aftermovie auto --clips ... --song ... --output ...` uses the user's preferences.

CLI flags always win over the env file; the env file wins over built-in defaults. Run `aftermovie show-config` to inspect the effective values.

If the user says "make all my auto edits ..." (e.g. "always vertical", "always punchy"), point them at this file — that's the right place to set it.

## Output length

The full song does NOT have to be used. Default is `min(song_duration, 90s)` — pass `target_length_s` (MCP) or `--length`/`--max-length` (CLI) to cap shorter. Typical good lengths:

- 30-45s — Instagram/Reels post
- 60-90s — TikTok / YouTube Short
- 2-3 min — travel/event highlight reel

If the user gives no length cue, propose 60s and ask "longer or shorter?" only after they see the first plan. If candidates run out before the target length, the plan just gets shorter — that's fine, no need to retry; tell the user the count.

## Stills + Live Photos

iPhone photo dumps mix video clips, Live Photos, and still HEIC/JPG. The analyzer handles all three:

- **Native video** (`.mp4 .mov .m4v .insv .lrv`) — used directly.
- **Live Photo pairs** — when a `*.HEIC`/`*.JPG` has a same-stem `*.MOV` next to it, the still is dropped (the MOV already carries the motion).
- **Standalone stills** (`.heic .heif .jpg .jpeg .png`) — auto-materialized to 2.5-second mp4 clips with a subtle Ken Burns zoom, cached under `~/.skills-data/aftermovie/cache/stills/`. They participate in scoring like any other short clip.

Pass `include_stills: false` (MCP) or `--no-stills` (CLI) to ignore photos. `--still-duration N` controls the per-still clip length.

> Single-file Live Photos (HEIC with the MOV embedded as a metadata box) currently degrade to the first frame as a still. Extracting the embedded MOV needs `exiftool` and isn't wired up yet — mention this if a user's "Live Photo" looks frozen in the edit.

## Iterating on the plan

If the user pushes back on the proposal — "too many cuts from clip_07.mp4" or "make it shorter" — apply tweak ops and re-render:

```
tweak_plan({plan_id, ops: [{op: "exclude_source", source: "/Users/.../clip_07.mp4"}]})
tweak_plan({plan_id, ops: [{op: "set", path: "target_length_s", value: 60}]})
```

If they want a totally different feel, just call `propose_plan` again with a different theme — it's cheap (the catalog is cached).

## Defaults

- **Length**: song duration, capped at 90s.
- **Aspect**: 16:9.
- **Theme**: `cinematic` if not specified.
- **Resolution**: 1080p. `--res 3840x2160` for 4K.
- **FPS**: 30.

## Pipeline summary

1. **Analyze** — scan every video file. For GoPro files extract GPMF telemetry (gyro, accel, GPS speed) + HiLight tags from the HMMT atom. For all clips compute per-second motion energy (signalstats YDIF) and audio RMS. Write `catalog.json`.
2. **Score** — librosa for tempo + beats + downbeats + intro boundary. Score each candidate sub-clip (motion ×1.5, audio ×1.0, accel jump +3, GPS speed peak +2, HiLight +10, repetition penalty). Greedy-fill beat slots starting after the intro. Write `plan.json`.
3. **Render** — per-cut ffmpeg passes (trim, scale/crop to aspect, optional speed ramp for slow-mo, LUT). Concat. Mix music on top at `music_db`. Write the MP4.

See `references/recipe.md` for the editorial reasoning, `references/themes.md` for theme specs, `references/gpmf.md` for telemetry format.

## Vertical output (TikTok / Reels / Shorts)

When the user mentions Instagram, TikTok, Reels, or vertical, set `aspect: "9:16"` in `propose_plan`. If face detection is available (mediapipe installed + model present — `aftermovie doctor` reports it), the renderer auto-reframes each cut to keep faces centered as they move. Pass `reframe: false` to fall back to center-crop.

## Failure modes to watch for

- **No clips found** — analyzer skipped everything. Tell the user which extensions are supported (`.mp4 .mov .m4v .insv .lrv`).
- **Song shorter than `target_length_s`** — output is clipped to song length; mention this.
- **MCP server not registered** — run `setup.sh` again, restart Claude Code.
- **Slow analyze step** — expected. ~5-10s per minute of source. Reuse `catalog_id` for follow-up `propose_plan` calls.
- **All clips look the same** — flat-feeling edit. Suggest mixing in B-roll or trying a different song.
- **Mediapipe unavailable** — face detection / 9:16 reframing degrade to centered crop. Run `aftermovie doctor` to confirm. The skill still works; just no smart reframing.
