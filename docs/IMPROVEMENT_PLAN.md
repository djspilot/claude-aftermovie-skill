# Aftermovie Improvement Plan

This plan describes how to move `aftermovie` from a useful GoPro Quik-style generator to a much stronger editing tool: faster to iterate, better at finding the right moments, and more controllable from the GUI.

The current pipeline is:

1. **Analyze** a Source folder into a Catalog.
2. **Score** Candidates against the Song and write a Plan.
3. **Render** Plan Entries into the final Aftermovie.

The main product gap is not “more render settings.” The main gap is a tight feedback loop: the user needs to preview, steer, and re-render without paying the full Analyze cost each time.

## Goals

- Produce aftermovies that feel closer to GoPro Quik: energetic, musical, and focused on the right moments.
- Make iteration fast enough that subjective tuning is practical.
- Let the user steer moment choice without editing JSON.
- Keep the pipeline local, deterministic, and inspectable.
- Preserve the CLI-first architecture while improving the GUI.

## Non-Goals

- Multi-song edits.
- Cloud rendering.
- Full NLE replacement.
- Real-time video scrubbing in the first pass.
- Captioning or speech-to-text.

## Current Problems

### Slow Iteration

The GUI currently starts a full Analyze to Score to Render cycle. Analyze is expensive because it materializes stills and extracts motion/audio/faces. This makes it hard to compare edits quickly.

### Weak Moment Selection

The scorer uses motion, audio, HiLight tags, faces, and metadata, but it still misses some human-obvious moments. It can also over-collapse nearby burst shots, causing selected moments to disappear.

### Limited Edit Steering

The GUI can include/exclude source files, but it cannot yet express:

- “Use this moment, not that moment.”
- “Use more like this.”
- “This clip is important.”
- “This clip can repeat.”
- “Try earlier/later in the source.”

### Plan Visibility

The Plan is the most important artifact, but it is invisible in the GUI. Users cannot see the chosen Entries, their order, why they were chosen, or how the Song structure drives the timeline.

### Renderer Polish

The renderer is functional, but several details still affect perceived quality:

- speed ramp data exists but is not fully rendered as a real ramp
- transition decisions are still heuristic
- color can vary sharply across Entries
- stills and short videos can feel over-held if planning underfills
- clip audio can be too noisy or too absent depending on source material

## Phase 1: Fast Preview Loop

**Objective:** reduce iteration time from tens of seconds or minutes to a few seconds once Analyze has run.

### Deliverables

- Add a reusable on-disk project state:
  - `catalog.json`
  - `song.json` or cached Song analysis
  - `plan.json`
  - GUI selection state
  - render settings
- Add `aftermovie auto --from-plan <plan.json>` or equivalent.
- Add GUI buttons:
  - `Render preview`
  - `Render final`
  - `Reuse analysis`
- Add Preview mode defaults:
  - 480p or 720p
  - 24 fps
  - no LUT
  - no face reframe unless needed
  - faster encoder settings

### Implementation Notes

- Reuse the existing `pipeline_runner` as the single orchestration layer.
- Preview mode should never mutate the final render settings permanently.
- Store temporary project state under a predictable cache directory keyed by Source folder + Song path + selection hash.
- If the Source folder changes, invalidate Catalog only for changed files where possible.

### Acceptance Criteria

- After one full Analyze, a GUI preview re-render avoids re-analyzing the Source folder.
- Preview render completes materially faster than full render on the same Plan.
- GUI clearly shows whether it is using cached Analyze data.
- CLI and GUI use the same project-state mechanism.

## Phase 2: Plan Timeline in the GUI

**Objective:** make the Plan visible and editable.

### Deliverables

- Add a timeline panel below the source grid.
- Show one tile per Plan Entry:
  - thumbnail
  - source filename
  - Entry duration
  - transition kind
  - score reason badges
  - audio-interest indicator
- Add actions:
  - remove Entry
  - pin Entry
  - favorite source
  - ban source
  - move Entry earlier/later
  - replace Entry with another Candidate from the same source
- Add a “why this?” inspection panel using existing score reasons.

### Implementation Notes

- Keep the Plan as JSON. The GUI should edit Plan JSON through server endpoints.
- Avoid building a full video scrubber first. A thumbnail timeline is enough.
- A pinned Entry should survive re-scoring unless the source file disappears.
- User edits should be stored separately from generated Plan data so the scorer can be rerun without losing preferences.

### Acceptance Criteria

- User can see exactly which Entries will render before starting ffmpeg.
- User can remove a bad Entry and preview without full Analyze.
- User can pin a good Entry and rerun scoring without losing it.
- The GUI displays score reasons for each Entry.

## Phase 3: Better Moment Ranking

**Objective:** improve Candidate scoring so the first generated Plan contains better moments.

### New Signals

- **Sharpness:** penalize blurry frames.
- **Exposure:** penalize very dark, blown-out, or low-contrast shots.
- **Composition:** reward faces/subjects near useful framing zones.
- **Action peaks:** improve motion-peak localization from optical flow and GPMF.
- **Audio events:** detect cheers, impacts, loud crowd moments, and sudden onsets.
- **Visual duplicates:** identify near-identical frames across sources, not only timestamp bursts.
- **User preference:** boost sources or Candidates previously liked/pinned.

### Candidate Model Changes

- Store multiple Candidates per Clip even for short-form videos where useful.
- Store representative thumbnails per Candidate, not just per source.
- Add per-Candidate quality fields:
  - `sharpness`
  - `exposure_score`
  - `face_score`
  - `action_score`
  - `audio_event_score`
  - `duplicate_group`
  - `user_boost`

### Acceptance Criteria

- Plan Entries include score components, not just a single score and reason list.
- The GUI can explain why an Entry won.
- Blurry/dark shots are selected less often unless explicitly pinned.
- Nearby burst shots are deduplicated by visual similarity rather than only timestamp.

## Phase 4: Song Structure and Musical Planning

**Objective:** make pacing follow the Song structure instead of a flat beat grid.

### Deliverables

- Detect Song sections:
  - intro
  - verse
  - build
  - drop/hook
  - outro
- Allocate Entry density by section:
  - slower cuts in low-energy sections
  - tighter cuts around drops and hooks
  - climax tail near the end
- Snap final Aftermovie length to a musical phrase boundary.
- Prefer strong visual moments on downbeats and section transitions.

### Implementation Notes

- Extend Song analysis with an energy curve and section boundaries.
- Keep `pace=auto` as the user-facing mode for structure-aware pacing.
- Preserve manual `fast`, `medium`, and `slow` modes for predictable behavior.

### Acceptance Criteria

- `pace=auto` emits visibly different cut density across quiet and energetic sections.
- The final Entry lands near a musically satisfying phrase boundary.
- Drops/hooks get stronger or more action-heavy Entries than verses.

## Phase 5: Renderer Polish

**Objective:** reduce artifacts and make the final output feel more professional.

### Deliverables

- Implement real two-segment speed ramps from `speed_start` to `speed_end`.
- Improve transition placement:
  - avoid whips on static stills
  - prefer hard cuts on high-action moments
  - use soft crossfades for still-heavy sequences
- Add basic color matching:
  - normalize exposure/contrast between adjacent Entries
  - keep LUT as the final creative look
- Improve still handling:
  - avoid excessive holds
  - favor gentle motion for photos
  - avoid awkward crop on faces
- Improve audio:
  - better clip-audio gate
  - optional impact/transition sound design
  - smoother ducking around speech or cheers

### Acceptance Criteria

- Speed ramps are visible in rendered output, not only represented in Plan data.
- Transition choices avoid obvious bad cases.
- Adjacent Entries have less jarring exposure/color shifts.
- Clip audio feels intentional rather than random.

## Phase 6: Feedback and Learning

**Objective:** let the tool improve within a project based on user choices.

### Deliverables

- Project preference file:
  - liked sources
  - banned sources
  - pinned Entries
  - preferred theme/settings
  - previous render history
- GUI actions:
  - thumbs up/down
  - “more like this”
  - “less like this”
  - “shuffle using same favorites”
- A/B render comparison:
  - render two Plans
  - choose winner
  - store winner signals

### Acceptance Criteria

- Rerunning the same project after user feedback changes the Plan in expected ways.
- Feedback survives app restarts.
- User can reset project preferences.

## Phase 7: Packaging and Open Source Readiness

**Objective:** make the project easy for other people to install, run, and contribute to.

### Deliverables

- Update README with:
  - GUI usage
  - CLI recipes
  - preview workflow
  - supported source types
  - troubleshooting
- Add screenshots or a short demo GIF.
- Add issue templates:
  - bug report
  - render quality issue
  - feature request
- Add contribution guide:
  - setup
  - tests
  - architecture overview
- Add sample fixture project small enough for CI.
- Add CI for tests and lint.

### Acceptance Criteria

- A new user can clone the repo, run setup, open the GUI, and render a sample.
- CI runs on pull requests.
- The README explains what “good source material” means.

## Proposed Roadmap

### Milestone 1: Fast Iteration

1. Project-state cache.
2. Render from existing Plan.
3. GUI Preview button.
4. GUI Plan visibility read-only.

### Milestone 2: User Steering

1. Editable timeline.
2. Pin/ban/favorite.
3. Candidate replacement.
4. Preference persistence.

### Milestone 3: Better First Draft

1. Sharpness and exposure scoring.
2. Better audio-event detection.
3. Visual duplicate grouping.
4. Structure-aware `pace=auto`.

### Milestone 4: Final Polish

1. Real speed ramps.
2. Better transition rules.
3. Color matching.
4. Improved audio mix.

### Milestone 5: Public Product Quality

1. Documentation refresh.
2. Demo assets.
3. CI.
4. Contribution guide.

## Recommended Next 10 Issues

1. Add project-state cache keyed by Source folder, Song, and selection hash.
2. Add `aftermovie render-from-plan` or `auto --from-plan`.
3. Add GUI `Render preview` using cached Analyze data.
4. Add `/api/plan` endpoint returning the current Plan.
5. Add read-only GUI timeline for Plan Entries.
6. Add pin/ban/favorite state to project preferences.
7. Add sharpness and exposure metrics to Analyze.
8. Add score component breakdown to Candidate and Entry output.
9. Add visual duplicate grouping.
10. Implement real speed ramps in Render.

## Success Metrics

- Preview render time after cached Analyze: under 10 seconds for a typical phone/GoPro Source folder.
- First generated Plan requires fewer manual removals.
- User can identify and fix bad Entries from the GUI without touching JSON.
- Punchy theme preserves requested target duration despite transitions.
- Public users can install and run the GUI from README instructions.

