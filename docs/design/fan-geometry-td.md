# Fan Geometry & Multi-Curve Time-Distance Analysis

## Science goal

SADs converge on the flare arcade from many directions. A single slit samples
one path; the user wants a *family* of slits spanning the convergence fan,
built from one drawn shape, each yielding a time-distance diagram against any
loaded data source — including the radio imaging cube with explicit frequency
selection — so SAD arrivals and radio brightenings can be compared along and
across the fan.

## User decisions (2026-08-10)

- The user draws **two boundary curves** (displayed red); the tool constructs
  the ruled patch between them and generates **N equally spaced intermediate
  curves** (displayed green, N adjustable). End closures (blue) are visual
  only — no cross-curve slits.
- Curve selection: **click a curve on the panel or in the card list**;
  selected curves become slits (multiple can be promoted; each is a normal
  slit entry).
- Radio frequency handling: extraction is **per-frequency** (one (npix,
  ntime) map per selected channel, switchable in the lane), and **additional
  selected frequencies render as contour overlays on the TD lane** in their
  channel-palette colors (e.g. AIA TD base map + radio TD contours).

## Geometry construction

- Boundary curves A and B: hand-drawn, Catmull-Rom smoothed, resampled to a
  common sample count M (max of the two resampled lengths at ~1 px spacing).
- Family curve at fraction f ∈ [0,1]: pointwise linear interpolation
  P_f(i) = (1-f)·A(i) + f·B(i), i = 0..M-1 (ruled surface). Boundaries are
  f=0 and f=1; intermediates at f = k/(N+1), k = 1..N.
- If the drawn curves run in opposite directions, auto-align B's
  parameterization (reverse if endpoint distance sum is smaller reversed).
- Each family curve inherits the slit origin convention (lower-left end =
  distance 0) applied to the whole family consistently (all curves oriented
  the same way, using boundary A's orientation after the lower-left rule).
- Stored in arcsec in session + slim JSON: the two boundary polylines, N,
  and which family members are promoted to slits.

## UI

- Fan tool lives in the Slit inspector card: "Draw fan" arms a two-stroke
  gesture (draw boundary 1, then boundary 2; Esc cancels between strokes).
- Patch renders on panels: red boundaries, green intermediates, thin blue
  end closures; non-selected curves faint, selected/promoted curves full
  opacity with the origin dot + arrowhead.
- N control (scrubbable number, default 3).
- Clicking a curve (14 px hit radius, same as markers) or its list row
  selects it; "Promote to slit" (or double-click) creates a slit entry bound
  to the current image layer — from then on it behaves exactly like a
  hand-drawn slit (extract, shift, swap, export).

## Radio frequency selection

- When a slit's bound layer is a radio source, the slit card shows a
  frequency multi-select (chips or checkboxes fed by the channel list; the
  channel palette colors each chip).
- Extraction runs per selected channel: results stored per (slit, freqIndex)
  with the same disk-cache identity extended by freqIndex.
- Lane: a base-map selector (which extracted map renders as the image —
  any freq, or the AIA/context map if the slit also has one) plus contour
  toggles per additional selected frequency: contours of that channel's TD
  map at a configurable level (percent-of-max scrubbable, default 70%),
  stroked in the channel color over the base map.
- NPZ export: intensity becomes (nfreq, npix, ntime) with freq_ghz axis
  when multiple channels are extracted; single-channel stays 2D for
  backward compatibility (or always 3D with nfreq=1 — implementer's choice,
  documented in the file).

## Efficiency

- Family curves share the extraction machinery and caches; per-freq radio
  extraction reads planes from the decoded-plane store (one decompression
  per cube ever), so extracting several curves × several channels is
  read-cheap.
- Patch editing (moving N) regenerates intermediates geometrically without
  touching extractions; only promoted slits ever extract.

## Verification plan

- Synthetic: two straight parallel boundaries → intermediates must be
  exactly equally spaced straight lines (report max deviation); opposite
  drawn directions must auto-align.
- Fan on real data: draw the SAD fan, N=3, promote one green curve, extract
  on AIA (report npix/ntime) and on the radio layer with 3 channels selected
  (report per-channel map count and cache keys distinct by freqIndex).
- Lane contours: base = one freq, contours from two other freqs; decode lane
  pixels to confirm contour strokes in the two channel colors.
- NPZ shapes with freq axis; session round-trip restores the fan (boundary
  vertices, N, promotions).
