# Theme Presets

Themes are bundles of `(LUT + music level + speed-ramp behavior)` chosen to give a coherent feel. Pass `--theme <name>` to the `auto` command.

## cinematic (default)

- LUT: `cinematic.cube` — neutral with slightly cool shadows, gentle S-curve contrast, lifted blacks
- Music: -10 dB (the visuals carry more weight than the song)
- Speed ramping: on
- Use for: travel, landscape, drone, anything where you want a "movie trailer" feel

## punchy

- LUT: `punchy.cube` — heavier contrast, slightly more saturation, deeper blacks
- Music: -6 dB (loud, the music drives the edit)
- Speed ramping: on
- Use for: sports, action sports, mountain biking, surfing, snowboarding — anything where you want the cuts to hit hard

## chill

- LUT: `chill.cube` — desaturated, warm, soft contrast
- Music: -10 dB
- Speed ramping: off (clips play at 1x for a more relaxed feel)
- Use for: lifestyle, B-roll, food, slow travel, hangouts

## nostalgic

- LUT: `nostalgic.cube` — yellow-shifted highlights, lifted blacks, slight fade, film-look
- Music: -10 dB
- Speed ramping: on
- Use for: family video, throwback edits, retrospectives

## Custom themes

Drop a `.cube` file in `assets/luts/` and reference it by stem name (e.g. `mytheme.cube` → `--lut mytheme`). Or pass an absolute path.

If you want a full custom theme (LUT + audio levels + ramp policy), edit the `themes` dict in `scripts/aftermovie.py` inside `main()`.

## Why these defaults

The four presets cover the common requests: "make it look like a movie," "make it hit hard," "make it feel chill," "make it feel like an old memory." If a user asks for "epic," map to `punchy`. "Cinematic" or "professional" → `cinematic`. "Vibe" or "aesthetic" → `chill`. "Throwback" or "memories" → `nostalgic`.
