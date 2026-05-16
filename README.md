# Installing the aftermovie skill on your Mac

This skill lives at `~/.claude/skills/aftermovie/` so Claude Code picks it up automatically.

## One-time install

```bash
# 1. Unzip this folder to your user skills directory
mkdir -p ~/.claude/skills
# (move the unzipped aftermovie/ folder there, so you have ~/.claude/skills/aftermovie/SKILL.md)

# 2. Install ffmpeg if you don't have it
brew install ffmpeg

# 3. Run the setup script (creates a local venv, installs Python deps)
bash ~/.claude/skills/aftermovie/scripts/setup.sh
```

The setup script is idempotent — safe to run again any time.

## Using it

Just ask Claude Code something like:

> Make me an aftermovie from `~/Movies/Iceland2026` using `~/Music/song.mp3`

Claude will read the skill, run the CLI, and drop the result in your output path. You can also call the CLI directly:

```bash
~/.skills-data/aftermovie/venv/bin/python \
  ~/.claude/skills/aftermovie/scripts/aftermovie.py auto \
  --clips ~/Movies/Iceland2026 \
  --song ~/Music/song.mp3 \
  --output ~/Movies/iceland.mp4 \
  --theme cinematic
```

## What it produces

A single MP4: H.264 video, AAC audio, 1080p by default (or 4K with `--res 3840x2160`), 30 fps, with your song mixed in at -8 dB.

For each cut, the scorer logs *why* it picked that clip (motion peak, accel jump, HiLight tag, etc.). Look at `plan.json` in the working dir after a run if you want to see the reasoning.

## File layout

```
aftermovie/
├── SKILL.md                  ← entrypoint Claude reads
├── scripts/
│   ├── aftermovie.py         ← the CLI (analyze, score, render, auto)
│   └── setup.sh              ← one-time dep installer
├── references/
│   ├── recipe.md             ← scoring weights + editing philosophy
│   ├── themes.md             ← the 4 theme presets
│   └── gpmf.md               ← GoPro telemetry format primer
└── assets/
    └── luts/
        ├── cinematic.cube    ← neutral, cool shadows, lifted blacks
        ├── punchy.cube       ← contrasty, saturated, deep blacks
        ├── chill.cube        ← desaturated, warm, soft
        └── nostalgic.cube    ← film-look fade, yellow highlights
```

## Quick troubleshooting

**"ffmpeg: command not found"** — `brew install ffmpeg`.

**"librosa not installed"** — Re-run `bash scripts/setup.sh`. It creates a venv at `~/.skills-data/aftermovie/venv/`.

**Output is just one clip repeating** — Source folder is probably too uniform (same shot many times). The scorer caps repetition at 3 per source, so if you have <4 unique clips that's the ceiling.

**Slow analyze step** — Expected for large folders. Each minute of source footage takes ~5-10 seconds to analyze. Use the staged workflow (`analyze` then `score` then `render`) so you only pay this cost once per source folder.
