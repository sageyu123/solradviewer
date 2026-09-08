# Generalized SAD tracking with anchor-frame editing

Status: implementation spec, user-approved decisions baked in. Replaces the
sample-event-only tracking (markpos.pickle seeds) with general tracking for
any dataset. The user's existing in-session tracks are experimental and need
no migration.

Approved decisions: click-to-seed plus optional suggested seeds; the tracker
matches on the LEFT panel base layer's exact displayed processing chain
(operation + temporal filter + radial settings as configured in its
inspector — LP, BP, raw, whatever is active); auto-track with validity
stop; outputs include smoothed velocities; anchor-frame editing with
constrained re-track between anchors; the anchor timeline lives in a
draggable floating "Track editor" card.

## Data model

A track: { id, label, color, anchors: [ { frameIndex, mjd, x, y } ... ],
points: [ { frameIndex, mjd, x, y, confidence, isAnchor } ... ],
state: active|stopped-low-confidence|stopped-edge }.
Coordinates in context-image pixels (world derivable via WCS as today).
Anchors are authoritative user-placed truths; points between anchors are
solver output. Session JSON round-trips tracks+anchors; CSV export gains
velocity columns (see Output). Undo/redo: a bounded (50) history of track
mutations (add/move/delete anchor, delete track, re-track results),
Cmd/Ctrl-Z / Shift-Z while the Track editor card is focused/open.

## Seeding

- Seed mode: a toggle button in the Feature Tracking rail section
  ("Add seeds", crosshair-pencil icon, cyan when armed). While armed,
  plain click on the LEFT panel places a seed (world->pixel via the
  existing mapping); each seed becomes a new track with one anchor at the
  current frame. Esc or toggle disarms. The pixel-probe click is
  suppressed while armed (same suppression pattern as alignment gestures).
- Suggested seeds (optional assist): a "Suggest seeds" button, enabled
  when an ROI exists: backend detects dark descending candidates in the
  CURRENT processed frame inside the ROI (local minima below a percentile
  threshold in the ratio image, minimum separation ~8 px, top N=20 by
  darkness). Suggestions render as hollow markers; click accepts one into
  a track, "Accept all" / "Clear suggestions" buttons. Detection endpoint
  reuses the layer-processed frame (same params as the frame route).
- The legacy markpos.pickle path ("Refine Seeds") remains functional when
  a manifest provides seeds, relabeled "Load seeds from file".

## Tracking engine (backend)

- Endpoint: POST .../track/auto with { trackId(s), layer params (the left
  panel base layer's full science chain), direction: forward|backward|both,
  startFrame, range bounds, patchRadius (default 6 px), searchRadius
  (default 18 px) }.
- Per step: normalized cross-correlation template matching of the patch
  from frame i in frame i+1, search window CENTERED on the
  velocity-predicted position (linear prediction from the last 2-3 track
  points; zero velocity at track birth). Sub-pixel refinement via
  parabolic fit of the correlation peak. Template updates each step
  (patch from the newest matched position).
- Validity stop: correlation peak below threshold (default 0.5) for 2
  consecutive frames, or predicted position leaves the image/ROI ->
  track state stopped-* at the last confident frame. Confidence stored
  per point.
- Matching operates on the processed array served by the same code path
  as the frame route with the left base layer's exact params (temporal
  filters included) — reuse the disk/memory caches; do NOT reimplement
  the processing.
- Progress: auto-track registers in the progress registry
  (label 'Tracking N seeds', done/total = frames processed).

## Anchor editing (constrained re-track)

- Editing gesture: with a track selected (see card), dragging its point
  marker in the LEFT panel at the current frame moves that point and
  promotes it to an ANCHOR. Every manual move creates/updates an anchor.
- Constrained re-track: after an anchor changes, the segments between it
  and its neighboring anchors (or track ends) re-run the tracking engine
  BIDIRECTIONALLY: forward from the earlier anchor and backward from the
  later anchor, blending the two passes with linear weights across the
  segment so the path passes exactly through both anchors. Endpoint:
  POST .../track/retrack-segment.
- Deleting an anchor re-tracks across the merged segment. Deleting all
  anchors of a track deletes the track (confirm via undo, not dialog).

## Track editor floating card

Draggable card (shared hook), opened from a button in the Feature Tracking
section and auto-opened when a track is selected. Contents:
1. Track list row: select, rename (pencil), color swatch, visibility eye,
   delete; per-track state badge (active / stopped @ frame).
2. Anchor timeline: a horizontal strip spanning the master range, cursor
   marker, anchor diamonds for the selected track (all tracks' anchors
   dimmed behind). Click a diamond -> jump cursor to that frame; drag a
   diamond horizontally -> move the anchor in TIME (re-track affected
   segments); Delete key removes the selected diamond; [ / ] hop to
   prev/next anchor.
3. Buttons: Auto-track (direction select: forward/backward/both),
   Stop, Undo/Redo, Suggest seeds, Export CSV.
4. Esc closes; position transient.

## Panel rendering

Selected track: full-opacity trail with per-point markers, anchors as
diamonds (larger), current-frame point emphasized; unselected tracks:
dimmed trails. Trails render in the existing tracks draw pass (vector,
crisp). Dragging markers uses pointer capture; pan/lasso/probe unaffected
when no marker is under the pointer.

## Output

CSV per track point: track_id, label, frame_index, time_utc, time_mjd,
x_px, y_px, x_arcsec, y_arcsec, confidence, is_anchor, vx_arcsec_s,
vy_arcsec_s, speed_arcsec_s (velocities from a centered 5-point
Savitzky-Golay-style smooth of the world positions; NaN at track ends
where the stencil is incomplete). Speed also displayable in km/s using
the arcsec-to-km factor at 1 AU (725 km/arcsec) — include a speed_km_s
column.

## Definition of done

- Fresh 20250328 session: arm Add seeds, click 3 points on the CME front,
  Auto-track both directions -> trails appear with confidence-based stop;
  drag a mid-track point -> anchor created, segment re-tracks through it;
  timeline diamonds navigate/delete correctly; undo/redo reverses each
  mutation; CSV contains positions + velocities; session JSON round-trips
  tracks/anchors; suggest-seeds proposes plausible dark candidates inside
  an ROI.
- pytest: engine unit tests on synthetic moving-blob sequences (tracks
  follow a known trajectory within 0.5 px; validity stop triggers when
  the blob disappears; constrained re-track passes through anchors
  exactly; velocity columns match the analytic motion within tolerance).
- npm run build passes; no new deps; no regressions to the established
  feature set (playback, contours, inspectors, warm/progress systems).
