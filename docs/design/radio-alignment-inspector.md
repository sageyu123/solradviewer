# Radio alignment inspector — per-channel x/y offsets

Status: implementation spec, user-approved decisions baked in. Purpose: the
user suspects a systematic pointing shift in part of the radio band (e.g.
spw 1 ≈ channels 30+); this tool makes per-channel offsets visible and
adjustable with immediate visual feedback.

Approved decisions: offsets are a **source-level calibration** applied
everywhere the source is used (contours, radio image layers, extraction);
UI is a **floating card**; persistence is **session JSON + CSV
export/import**; selection is **click + box + spw group buttons**.

## Data model

- The radio source gains a per-channel offset table:
  `channelOffsets: { dx: number[], dy: number[] }` in **arcsec**, length =
  nfreq (52), default all zeros.
- Authority: the backend session stores the table (new
  `POST /api/sessions/{id}/sources/{sourceId}/channel-offsets` with the full
  table; debounced from the frontend). All backend consumers read it:
  - contour overlay: per-band offset = global (xOffsetArcsec,
    yOffsetArcsec) + (dx[i], dy[i]) applied in the per-band transform to
    AIA pixels;
  - extraction (`extract_eovsa_sources`): reported source positions apply
    the same per-band offsets;
  - single-band frame rendering stays unshifted server-side; the frontend
    applies `worldOffset = global + (dx[freqIndex], dy[freqIndex])` for
    radio image layers (it already applies the global offset this way).
- Frontend keeps a mirror of the table in state; session JSON round-trips
  it (absent in old exports → zeros). Contour overlay cache keys include a
  hash/version of the table.

## Floating card

Opened from: a small target/crosshair icon button on the radio source row
(Data section) and in radio-source layer inspectors. Dismiss: × or Esc.
Wide card anchored over the workspace (light-curve-card styling family).

Contents:
1. **Two stacked plots** (~460×130 each, plain canvas): dx vs channel and
   dy vs channel. Points colored with the contour frequency palette (same
   mapping as the overlay colorbar). Zero line drawn. Y-range auto with
   minimum span ±5 arcsec.
2. **Selection**: click a point toggles that channel; drag a rectangle
   selects a range (either plot selects globally); spw group buttons —
   derive groups from the frequency table (split at the largest cfreqs
   discontinuity; if no clear gap, split at nfreq/2; label buttons with
   the actual channel ranges, e.g. `spw0 · 0–25`, `spw1 · 26–51`) — plus
   `All` and `Clear`. Selected points get a ring highlight.
3. **Numeric fine-tune**: two CommitInputs (dx, dy, arcsec) showing the
   selected channels' common value (blank when mixed); committing writes
   the value to all selected channels.
4. **Actions**: `Zero selected`, `Zero all`, `Export CSV`, `Import CSV`
   (frontend-side CSV: header `channel,freq_ghz,dx_arcsec,dy_arcsec`; on
   import validate length/finite values, unknown channels ignored with a
   toast), close.

## Image-panel interaction (while the card is open)

- **Shift+drag on either image panel** moves the SELECTED channels'
  offsets by the drag delta converted to arcsec via the panel transform
  (same math as the lasso/ROI world mapping). Live feedback: the contour
  overlay re-requests with the updated table, debounced ~150 ms (the
  existing scheduler handles cancellation); the plots update immediately.
- **Highlighting**: while the card is open, the contour overlay request
  gains `highlightChannels=<comma list of selected>`; the backend draws
  highlighted bands at full opacity/thicker stroke (2px) and dims
  non-selected bands (~35% of the layer's contour opacity), so you can see
  exactly which contours you're moving. Param absent → normal rendering
  (cache identity for the normal path unchanged).
- Shift+drag must not conflict with existing gestures: plain drag still
  pans, space-pan unchanged, lasso unchanged (alignment shift+drag takes
  precedence only while the card is open).

## Lasso channel selection (increment 2 — after the base tool lands)

User-approved addition: while the alignment card is open, the existing
lasso tool selects channels by region — draw a mask on either image
panel; the channels whose contours fall inside it become the selection.

- Mechanism: the per-band contour geometry exists only server-side, so a
  new endpoint takes the lasso polygon (in the panel's base-layer image
  pixel coords, same mapping the ROI lasso already produces) plus the
  current contour params and the offset table, and returns the band
  indices whose contours lie inside.
- Containment rule: a band is selected if ANY of its contour polylines has
  its centroid inside the polygon (`matplotlib.path.Path.contains_points`
  on centroids; centroids computed after applying the current per-band
  offsets so the test matches what the user sees).
- Frontend: while the card is open, completing a lasso routes to channel
  selection instead of ROI (the ROI lasso behavior resumes when the card
  closes); the returned channels replace the selection (shift+lasso adds
  to it). Plots and highlight rendering update as with any selection.
- Failure path: empty result → brief status note "no contours inside the
  region", selection unchanged.

## Part B (same run): stronger radial enhancement formula

Replace the coronal-enhancement factor (currently
`clip(r/R_sun, 1, 2.5)^gamma`, too weak over cutout radii ≈ 1.0–1.3 R_sun)
with an exponential scale-height compensation:

```
factor(r) = exp( gamma * max(r - R_sun, 0) / H ),  H = 0.2 * R_sun
factor capped at 1e3; on-disk (r <= R_sun) stays exactly 1
```

Same 0–3 slider and `radialGamma` param (reinterpreted; no schema change).
At gamma 3: ×20 at 1.2 R_sun, ×90 at 1.3 R_sun — matching JHelioviewer's
visible lift. Update the radial-map cache (keyed values change), the
existing radial tests' expectations (gamma=0 identity and on-disk
invariance unchanged; cap test updated to 1e3), and the design doc note in
enhancement-filters.md. Best visual results pair with the new sqrt/asinh
scales; no autofill changes.

## Definition of done

- Card opens from both entry points; plots render 52 zero points colored
  by the palette; click/box/spw/All/Clear selection works; ring highlight
  visible.
- Shift+drag on a panel with spw1 selected shifts those contours live and
  plots update; unselected contours dim while selected ones highlight;
  closing the card restores normal contour rendering.
- Numeric fine-tune, Zero selected/all work.
- CSV export→import round-trips the table; session JSON export→load
  restores the table (verify via the load-json state echo).
- Extraction results reflect the offsets (backend test: synthetic source
  with known offset table → extracted positions shifted accordingly).
- Radial Part B: anchor behaviors (gamma 0 identity, on-disk untouched,
  cap) tested; visual check shows strong off-limb lift at gamma 2–3 with
  sqrt scale.
- pytest + npm run build pass; no new dependencies.
