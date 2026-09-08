# SAD–Radio Temporal Correlation Panel

## Science goal

For the 2025-03-28 dataset, multiple supra-arcade downflows (SADs) appear
simultaneously and impinge on the loop-top region from different directions.
The classic single-slit time–distance diagram does not scale: one slit per SAD
is tedious and the resulting diagrams cannot be composed into a single
correlation argument against the radio dynamic spectrum.

The tracker already produces per-SAD trajectories r(t), v(t). That removes the
need for slits entirely: a time–distance diagram is only a means of extracting
a trajectory from image data. With trajectories as data, we compute the
physically meaningful quantity directly — **distance to the loop-top region** —
and collapse all SADs into one figure that is time-aligned with the EOVSA
spectrogram.

## User decisions (2026-08-09)

- Arrival target: a **drawn boundary** (polygon/lasso on the image panel),
  drawn once by the user around the loop-top region. Distance = signed
  distance to that boundary; arrival = first crossing.
- Expected SAD count: **~6–15** → superposed-epoch statistics are viable as
  a phase 2.
- Deliverable: **in-app panel first** (interactive lane attached to the
  spectrogram; recorder-exportable). Publication script later, after the
  method settles.

## Design

### 1. Loop-top target boundary

- Drawn on the left image panel with the existing lasso mechanism (same
  gesture family as the ROI/lasso tools; stored in solar arcsec coordinates
  so it is resolution- and orientation-independent).
- Exactly one target per session (redrawable; persisted in the session and
  exported/imported with the session JSON).
- Rendered as a dashed closed polygon with a distinct color (amber) on both
  image panels, mirroring the existing track-overlay pattern.

### 2. Correlation card (floating)

This is a per-analysis tool, not a per-dataset fixture, so it lives in a
**floating, draggable, closable card** — same family and interaction pattern
as the Track editor and Channel inspector cards. Opened from a button in the
Track editor / tracking section; nothing is rendered anywhere when the card
is closed and no target is defined.

The card body is the distance-to-looptop plot:

- **X axis synced to the spectrogram's visible time window** (pan/zoom of the
  master spectrogram updates the card, so the two stay visually comparable
  across the screen), with a toggle to lock to the full master range instead.
- Resizable card; plot height ~160–240 px.

- One curve per track: d_i(t) = min distance from track position to the
  target boundary polygon (positive outside; clamped at 0 after first
  crossing). Units: arcsec on the left axis, Mm on the right axis
  (1 arcsec = 725 km).
- Curves colored by approach position angle (hue wheel), matching the track
  polyline colors already drawn on the image panels — one color identity per
  SAD everywhere.
- The master time cursor is drawn in the card at the same MJD as the
  spectrogram cursor; clicking/dragging in the card plot moves the master
  time, same gesture as the spectrogram.
- Hover on a curve highlights the corresponding track on the image panels
  (and vice versa when the Track editor selects a track).

### 3. Arrival ticks on the spectrogram

- Arrival time t_i = first boundary crossing of track i (linear interpolation
  between the last outside and first inside sample). If a track never
  crosses, no tick (lane curve still drawn).
- Rendered as short labeled ticks on the spectrogram's bottom edge
  (track color, track label), with a subtle full-height dashed line
  toggleable from the card header. Ticks are drawn only while the card is
  open OR the "pin ticks" toggle in the card is on — closing the card with
  ticks unpinned leaves the spectrogram untouched.
- Secondary marker (optional toggle): peak-deceleration time from the
  Savitzky-Golay velocity series, drawn as an open triangle — impact can
  begin before geometric boundary crossing.

### 4. Phase 2 (not in first implementation): statistics

- Superposed-epoch: median spectrogram cut around each arrival tick
  (t_i − 60 s … t_i + 120 s), stacked and averaged; displayed as a small
  companion card.
- Impact-power proxy: P(t) = Σ_i v_i²(t) over active tracks, drawn as a
  faint white curve overlaid on the lane, for lag cross-correlation against
  a chosen radio band.

### 5. Export

- Recorder: when the card is open, the workspace layout may optionally
  include it as a capture surface (checkbox in recorder options, default
  off). Pinned arrival ticks on the spectrogram are burned in automatically
  since they are part of the spectrogram canvas. Phase 1 may ship without
  the card-capture option if it complicates the recorder; ticks alone
  already carry the correlation into exports.
- CSV addendum: the tracks CSV gains per-row `dist_to_target_arcsec` and a
  per-track `arrival_utc` in the header block, when a target is defined.
- Card plot PNG export button (one click, current view) for quick sharing.

## Integration points (from code scout, 2026-08-09)

- **Tracks**: `SadTrack` / `TrackPoint` types (App.tsx:406–430); state
  `sadTracks` with undo history; points are context-image **pixels**, world
  arcsec derived via `applyAffine(..., pixelToWorldAffine)`. Track colors in
  `TRACK_COLORS` (App.tsx:564) — the card must reuse `track.color`.
- **Velocities**: 5-point quadratic LSQ in `tracking_csv_rows`
  (data.py:1910); frontend mirror `displayedTrackSpeedKmS` (App.tsx:7881).
  Peak-deceleration marker derives from the same series.
- **Floating card**: follow `TrackEditorCard` (App.tsx:7683) +
  `useDraggableFloatingCard()` (App.tsx:7603); root
  `section.radio-alignment-card` pattern.
- **Lasso**: `ImagePanel` lasso gesture emits image-pixel polygons via
  `onLassoComplete(panel, points, sourceId, role, additive, channelSelect)`
  (App.tsx:7493); dispatch at App.tsx:5060. Target drawing adds a third
  armed mode alongside `lassoEnabled` / `channelLassoArmed`. Authoritative
  storage in **arcsec** like `roiWorld` (App.tsx:1902); render with
  `drawWorldPath` (App.tsx:8498).
- **No frontend point-in-polygon / distance-to-polygon helper exists** —
  distance-to-boundary must be written new (point-to-segment min over the
  polygon, in arcsec space).
- **Spectrogram**: `SpectrogramPanel` (App.tsx:5245); time→x mapping is the
  inline formula used by `cursorX` (App.tsx:5556) over the visible
  `displayMin/Max`; ticks drawn on the main canvas inside `draw()`
  (App.tsx:5714). Arrival ticks are a new vector pass there, gated on
  card-open ∨ pinned.
- **Card x-sync**: consume the same `timeRange` state the spectrogram pan/
  zoom updates via `onTimeRangeChange` (App.tsx:6124).
- **Session round-trip**: save payload keys at App.tsx:3314; backend
  `state.get(...)` load at data.py:3891; add a `correlationTarget` key
  (arcsec vertices) both ways.
- **CSV**: `TRACKING_FIELDS` (data.py:88, 15 columns) — append
  `dist_to_target_arcsec`; `write_tracking_csv` (data.py:5297) writes
  `feature_tracks.csv` + `sad_tracks.csv`.

## Verification plan

- Draw a target around the loop-top, confirm dashed polygon on both panels
  and persistence across reload.
- Card opens/closes/drags like the Track editor card; closing it (with
  ticks unpinned) leaves no residue on the spectrogram or image panels
  except the target polygon.
- With ≥3 seeded tracks, confirm one colored curve per track in the card,
  monotone-decreasing distance for an approaching SAD, and an arrival tick
  where the curve reaches zero; tick UTC must equal the interpolated
  crossing within one track sample interval.
- X-range sync: pan/zoom the spectrogram → the card window follows.
- Color identity: card curve, image-panel polyline, and spectrogram tick all
  share the track color.
- Pinned ticks appear in the recorder's spectrogram capture (decoded-frame
  check, same method as the spectrogram axes verification).
- CSV round-trip: `dist_to_target_arcsec` present and finite; `arrival_utc`
  matches the tick.
