# Render Performance & UX Plan

Drive `aftermovie` from "burns the CPU, opaque progress, fixed 90s cap" to
"uses Apple Silicon's media engine, shows per-stage progress, fills the
whole song intelligently." Born from a real session on an M5 Pro (15
cores, 48 GB) where `Render Final` ran sequential `libx264 -preset fast`
prerenders, spun up the fans, and surfaced no progress to the GUI.

## Diagnosis (what we observed)

- Pipeline: per-Entry **prerender** (one ffmpeg subprocess each,
  sequential, `libx264 -preset fast -crf 20`) → **assemble** (filter_complex
  with xfade + acrossfade, another x264 pass) → **mux** (audio mix +
  container).
- ~80–95 % of wall-clock is in the prerender phase.
- All encoding is pure CPU. M5 Pro's media engine, GPU and Neural Engine
  sit idle.
- ffmpeg's `-progress pipe:1` output is never read, so the GUI never sees
  intermediate progress — only the final "done" / "error" state from the
  worker.
- `DEFAULT_TARGET_LEN_S` caps at `min(song, 90)` so even a 2:36 song renders
  to ~90 s by default.
- Song path is a CLI-only argument (`aftermovie select --song …`) — the
  GUI can't swap songs without a restart.

## Goals

- Visible per-stage progress in the GUI with an ETA.
- Use the Apple Silicon media engine for encode/decode by default.
- Render in seconds, not minutes, on a typical 90s aftermovie.
- Default to the full song duration; planner stretches/compresses
  content to fit intelligently.
- Pick the song from the GUI without restarting the server.

## Non-Goals

- Multi-song renders.
- Cloud / remote render farms.
- Full NLE features (trim handles, keyframe envelopes, …).
- Real-time scrubbing.

---

## Phase A — Visibility (progress %, energy attribution)

Cheapest big UX win. No render-quality change.

### Deliverables

- **A1** Parse ffmpeg `-progress pipe:1` per subprocess. Emit
  `frames_done / frames_total` + `current_time_s` upstream via a callback.
- **A2** Extend `RenderJob` with `stage` (`analyze` / `score` /
  `prerender_N_of_M` / `assemble` / `mux`), `progress_percent`, `eta_s`,
  `current_ffmpeg_pid`.
- **A3** GUI status area becomes a progress bar with stage label + ETA
  + Preview/Final badge. Existing `/api/status/<job_id>` already has the
  poll loop — extend the payload, render the new fields.
- **A4** Per-stage CPU-time accounting (`time.process_time()` for the
  worker, `subprocess.Popen.cpu_seconds` heuristic for ffmpeg). Surface
  `cpu_seconds_used` on the status payload so the user can see which
  stage is the energy hog.

### Acceptance

- Status payload after first frame includes `progress_percent ≥ 0` and
  `stage` matching the running phase.
- GUI bar advances visibly during prerender (the ~80 % of time today
  with no feedback).
- `cpu_seconds_used` correlates with wall-clock when CPU encoder is in
  use, and is much lower when VideoToolbox is in use (validates B).

---

## Phase B — Apple Silicon hardware utilization

Largest perf + energy win. Single biggest factor in "fans loud" complaint.

### Deliverables

- **B1** Swap `libx264` → `hevc_videotoolbox` (or `h264_videotoolbox`)
  behind a config flag `AFTERMOVIE_VIDEO_CODEC ∈ {x264, h264_vt, hevc_vt}`.
  Default = `hevc_vt` on `darwin/arm64` when the encoder is available.
- **B2** Hardware decode for input clips:
  `-hwaccel videotoolbox -hwaccel_output_format videotoolbox`. Lets the
  GPU decode HERO9 HEVC files instead of CPU.
- **B3** Chip detection: `sysctl -n machdep.cpu.brand_string` →
  Apple-Silicon vs Intel vs other. Choose codec + parallelism profile
  based on the chip family (M1 / M-Pro / M-Max all have different media
  engine counts).
- **B4** Parallel prerender. `concurrent.futures.ProcessPoolExecutor`
  with `max_workers = media_engines * 2` for VT, or `P-cores // 2` for
  x264 fallback. Each entry prerendered independently into the workdir,
  then assembled in plan-order. Depends on A1 (progress per subprocess).
- **B5** Pixel-format guards. VideoToolbox has stricter input
  requirements — insert `format=nv12` / `format=yuv420p` where needed so
  the encoder doesn't reject mixed-format inputs.
- **B6** `aftermovie bench --plan PATH`: compare x264-CPU vs VT-GPU on
  the same plan, report wall-clock + cpu-seconds + a fan-noise proxy
  (cumulative power via `pmset -g rawlog` parse).

### Acceptance

- Default render on M-series Mac uses VideoToolbox; logs `encoder=hevc_vt`.
- Single-clip prerender wall-clock drops ≥3× vs `libx264 -preset fast` at
  matching visual quality.
- 30-entry prerender wall-clock drops ≥5× with B4 enabled.
- CPU usage peak during render drops from ~800–1000 % (10 cores @ 100 %)
  to ~50–150 %.
- `aftermovie bench` produces a reproducible table.

---

## Phase C — Dynamic length + smart song-to-content mapping

### Deliverables

- **C1** Drop the `min(song, 90)` cap. New default `target_length_s =
  song_duration`. `--max-length` keeps overriding.
- **C2** GUI: length dropdown (`Full song (M:SS)` / `1 min` / `30 sec` /
  `Custom`). `Full song` is default. Posted to `/api/render` as
  `max_length`.
- **C3** Planner **stretch mode**: when the candidate pool is too small
  for the target duration, bump `source_cap` and/or `still_duration`
  automatically and log the decision (`stretched stills 2.5s → 3.8s to
  fill 156s`). Builds on the existing `_underfilled_plan` machinery in
  `score/scorer.py`.
- **C4** Planner **subset mode**: when the pool is much larger than
  needed (200 candidates for 90 s of slots), pick top-N by score, not
  the first-N by time.
- **C5** Pacing-aware fill: detect Song sections (intro / verse / build
  / drop / outro) and allocate density accordingly — tighter cuts on
  drops, longer shots in verses. Closes out Phase 4 of the original
  `IMPROVEMENT_PLAN.md`.

### Acceptance

- A 2:36 song renders to ~2:36 of aftermovie by default.
- A 30-clip pool covering 60 s of unique content still fills a 156 s
  target via stretch + judicious repeats.
- A 200-clip pool produces a 90 s aftermovie composed of the top
  ~30–45 candidates by score.
- `pace=auto` emits visibly different cut density across quiet vs
  energetic song sections.

---

## Phase D — Song selector in GUI

### Deliverables

- **D1** Song input field showing the current song path; `Browse…`
  triggers an HTML file picker → `POST /api/upload-song` copies the
  file into a server cache and returns the absolute path the renderer
  uses. (Browsers don't expose absolute paths from file pickers, hence
  the copy-through.)
- **D2** Auto-detect: scan the clips folder for audio files
  (`*.mp3 *.wav *.m4a *.flac *.aac *.ogg`) and surface them as
  one-click suggestions under the song input.
- **D3** After song change: fetch + display `tempo_bpm`, `duration`,
  and a tiny energy-curve sparkline. Use the existing `score.song`
  analyzer; cache `song.json` keyed by file content hash so re-picking
  is instant.
- **D4** Recent-songs dropdown — last N song paths in
  `~/.aftermovie/recent-songs.json`. Click a recent → switch song
  without browsing.

### Acceptance

- Picking a song in the GUI changes the render target without a server
  restart.
- A previously analyzed song re-loads in <1 s.
- Recents list survives a server restart.

---

## Phase E — Render cache (bonus)

Makes "tweak knob → re-render" cycles practically instant.

### Deliverables

- **E1** Per-clip prerender cache keyed by
  SHA1(source-mtime, start_s, end_s, speed, aspect, target_res, fps,
  lut, audio_interest_gate). Cache hit → skip ffmpeg, hard-link out of
  the cache.
- **E2** Cache TTL + max-size config
  (`AFTERMOVIE_PRERENDER_CACHE_MAX_GB`) with LRU eviction.
- **E3** `aftermovie cache stats` and `aftermovie cache clear`
  subcommands.

### Acceptance

- A second render of the same plan with only the LUT changed reuses
  ≥80 % of prerendered clips.
- Cache directory size stays within the configured cap.
- `cache stats` prints hit/miss ratios and disk usage.

---

## Expected impact

| Metric | Now (x264 CPU) | After B1+B2 | After B1+B2+B4 |
|---|---|---|---|
| Wall-clock for 90 s / 30 entries | ~120–180 s | ~15–25 s | ~8–15 s |
| CPU usage peak | 800–1000 % | ~50–100 % | ~150–300 % |
| Energy / fans | high, audible | near-silent | mild |
| RSS memory | ~500 MB | ~600 MB | ~1.5 GB |

Phase E1 makes re-renders after knob tweaks ~instant.

## Roadmap

### Batch 1 (parallel)

1. **A1+A2+A3+A4** — progress UX
2. **B1+B2+B3+B5** — VideoToolbox encode + decode + chip detect + pixfmt guards
3. **C1+C2+C3+C4** — full-song default, GUI dropdown, stretch + subset modes
4. **D1+D2+D3+D4** — song picker, auto-detect, recents, song-analysis cache

### Batch 2 (sequential, after Batch 1)

5. **B4** — parallel prerender (depends on A1's per-subprocess progress contract)
6. **C5** — pacing-aware section fill (depends on C1)
7. **B6** — `aftermovie bench`
8. **E1+E2+E3** — render cache

## Success metrics

- Default M-series render: <30 s for a 2:30 song with VideoToolbox + parallel.
- Fans inaudible during normal render on M-Pro / M-Max.
- Progress bar reaches 100 % before "done" lands, with an ETA accurate
  to ±20 % after the first 5 % is done.
- Picking a song in the GUI works, length defaults to song duration,
  and the planner gracefully fills it with the available material.
