# The Aftermovie Editing Recipe

Read this file when:
- The user asks why a specific clip was/wasn't chosen
- The user wants to understand or change the scoring weights
- You're tuning the output for a specific genre or mood
- Adding a new feature to the scorer

## The five forces shaping the output

Every cut in an aftermovie answers a question: *given this beat in the music, which clip should be on screen?* The scorer combines five signals to answer:

### 1. User intent (HiLight tags)

Weight: **10x** — overwhelms everything else.

When a GoPro user pressed the side button during recording, they explicitly said "this moment matters." That signal beats any algorithmic guess. If a HiLight tag falls inside a candidate window, that candidate wins almost regardless of other scores. iPhone footage and most other sources don't have HiLight tags, so this only fires on GoPro originals.

### 2. Telemetry-driven action (GoPro only)

Weight: **1.5x – 3x**, depending on intensity.

The accelerometer magnitude at rest is ~9.8 m/s² (gravity alone). A jump, drop, or impact produces a brief spike to 15-30+. The scorer treats anything above 15 m/s² as a "high accel" event and above 12 as "moderate."

GPS speed peaks (the fastest moment in a clip relative to the rest of that same clip) get a 2x bonus. This catches the top of a run, the apex of a jump, the climax of a turn.

### 3. Universal motion energy

Weight: **1.5x** of the per-second motion magnitude.

For iPhone, drone, mirrorless — anything without GPMF — we fall back to frame-to-frame luma difference (`signalstats YDIF` in ffmpeg). It's a crude but reliable proxy for "stuff is happening on screen." A locked-off interview shot scores low; a tracking shot of a moving subject scores high.

### 4. Audio energy

Weight: **1x** of normalized audio RMS.

Cheering, voices, impact sounds. We don't classify what the audio *is*, just how loud it is relative to baseline. This nicely catches moments humans react to even when the camera was still.

### 5. Repetition penalty

After a source file has contributed three sub-clips to the edit, further candidates from that file are skipped. This is what prevents the edit from being 90% one clip, even if that clip is great.

## Pacing — why cuts land where they do

The scorer uses the song's beat track from librosa as the cut grid. Two rules shape the timeline:

1. **The first cut waits for the intro to end.** Librosa's onset strength curve tells us when the music actually starts (the percussive entry, the bass drop, whatever). Cutting on the song's first sub-second beat — which is often a quiet intro — feels wrong. We hold on something neutral until the song's energy peaks above its 70th percentile.

2. **Downbeats earn the "hero" clips.** Roughly every fourth beat is a downbeat. These mark the start of a musical bar and they feel structurally important. The highest-scoring clip lands on the first downbeat. Slow-motion is reserved for downbeats too — speed-ramping on a sub-beat looks accidental, but on a downbeat it reads as deliberate emphasis.

## Speed ramping logic

A clip gets auto-slowed to 0.5x playback when **all three** conditions hold:

- Source was shot at ≥ 90 fps (so slowed-down playback is still smooth, not stuttery)
- Lands on a downbeat
- Was picked because of a motion peak, accel spike, or HiLight tag (something the slow-mo is actually emphasizing)

If any condition fails, the clip plays at 1.0x. This keeps slow-motion meaningful instead of constant.

## Why these weights and not others

The weights were tuned to match the felt pacing of GoPro Quik outputs and YouTube aftermovie compilations. They are not sacred:

- For wedding/family video, lower the accel weight to 0.5x — there are no jumps and the algorithm starts mistaking camera shake for action.
- For sports content, raise speed_peak and accel weights to 3-4x — those events *are* the story.
- For travel B-roll where nothing dramatic happens, lean harder on audio and motion-energy (raise both to 2.5x) since they're the only real signal.

The user can change these by editing `scripts/aftermovie.py`, in the `score_window` function. Surface that as an option if the user is unhappy with the picks.

## What this skill deliberately doesn't do

- **Face / smile detection.** GoPro Quik uses it. We skip it in the MVP because it adds a heavy ML dependency (MediaPipe or CoreML) and the motion + audio signals already catch most human-centered moments well enough. Add it later if results feel face-blind.
- **Scene classification.** Quik knows "this is snow, use the snow LUT." We use one LUT for the whole edit. The user can override per-edit.
- **Object tracking and reframing.** Quik can crop a horizontal clip to vertical while keeping a face centered. We do center-crop only. For serious vertical output the user is better served by Final Cut or CapCut.
- **Original clip audio mixed under the music.** The MVP strips clip audio and lays the music on top. Mixing both back in cleanly (ducking the music when a voice is detected, etc.) is doable with `ffmpeg sidechaincompress` but adds enough complexity it's worth waiting until the user asks for it.

## Debugging a bad edit

If the output doesn't feel right, walk this list:

1. **Open `plan.json`.** Each entry has `reasons` — that's why the scorer picked it. If everything says `motion_peak` and nothing else, the GPMF telemetry isn't being read. Re-run analyze with verbose logging to confirm.
2. **Compare `score` values across entries.** If all scores are within 10% of each other, the scorer has no signal to work with (too-uniform footage) — the edit will feel random. Suggest the user vary their source material or accept that nothing will look perfect.
3. **Check the `beat_time_s` field.** If cuts are landing on weird fractional seconds clustered together, the beat detection didn't lock onto the song. Some songs (very ambient or hip-hop with sparse drums) defeat librosa's tracker. Try a different song or pre-process it through a beat-detection model like `madmom`.
4. **Look at `intro_end_s` in `song_meta`.** If it's `0.0` for a song that clearly has a long intro, the onset detector failed; the first cut will land in silence. Worth raising the percentile threshold from 70 to 85 in `analyze_song`.
