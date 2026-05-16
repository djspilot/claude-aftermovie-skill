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

If you want a full custom theme (LUT + audio levels + ramp policy), edit the `THEMES` dict in `src/aftermovie/config.py`.

## Phase-3 additions

Beyond LUT + music level + speed-ramp, plans now carry three more fields that the CLI and MCP both surface:

- `audio_mix` ∈ `{"music_only", "ducked", "clip_only"}` — `ducked` keeps clip audio and side-chain-compresses the music under the voice band, so voices stay intelligible. Default per theme:
  - cinematic, punchy → `music_only`
  - chill, nostalgic → `ducked`
- `transitions` ∈ `{"cut", "auto"}` — `auto` lets the scorer place crossfades and whip wipes on structural beats (max ~3 whips per edit). Pure-cut plans stay on the fast concat path; any non-cut switches the render to filter_complex (slower but allows transitions and titles).
- `titles` — list of `{kind, text, duration_s, at_s?}` entries. `kind` is `intro`, `outro`, or a free-floating timed card. Titles are rendered as PIL PNGs and ffmpeg-overlaid, so they work even on minimal ffmpeg builds without libfreetype.

Per-cut, `entries[i].transition_in = {kind, duration_s, direction?}` decides how a single cut blends with the previous one.

## Theme typography

When a title is rendered, the active theme picks the typeface, size, color, and shadow. Drop your own TTF/OTF fonts into `assets/fonts/` (a file containing `display` in the filename is preferred for big intro cards; one containing `sans` is preferred for chill captions). If `assets/fonts/` is empty, we fall back to a system font (Helvetica/Arial/DejaVu).

## Why these defaults

The four presets cover the common requests: "make it look like a movie," "make it hit hard," "make it feel chill," "make it feel like an old memory." If a user asks for "epic," map to `punchy`. "Cinematic" or "professional" → `cinematic`. "Vibe" or "aesthetic" → `chill`. "Throwback" or "memories" → `nostalgic`.
