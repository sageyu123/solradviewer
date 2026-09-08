# Pixel light-curve probe

Status: implementation spec. Purpose: click a pixel in an image panel to see
that pixel's data time series (raw difference/ratio + temporally smoothed
curve), to guide vmin/vmax and temporal-smoothing σ choices.

## Interaction

- With ROI lasso OFF, a plain left-click on an image panel (pointer up
  without drag movement > 4 px, space-pan not held) probes the clicked
  pixel. Clicking again re-probes; Esc or the card's × closes the card.
- The probe belongs to the clicked panel's **base layer** and uses that
  layer's exact settings: source, operation, reference, lag, sampling
  policy, and temporal σ values.
- A small marker (crosshair dot) is drawn at the probed pixel on the panel
  while the card is open; it tracks pan/zoom via the existing transform.

## Backend

New route:

```
GET /api/sessions/{id}/sources/{sourceId}/timeseries
  ?x=<imagePixelX>&y=<imagePixelY>          # base-layer image pixel coords
  &patchRadius=1                            # patch mean over (2r+1)^2 px, default 1 (3x3)
  &startMjd=<mjd>&endMjd=<mjd>              # master range
  &maxPoints=400                            # stride-decimate to <= this many samples
  &freqIndex=<int>                          # radio cubes only
  &differenceOperation/Reference/Mode/diffSeconds/meanStartMjd/meanEndMjd   # as frame routes
  &temporalMode=lowpass|bandpass&temporalSigmaShort=<s>&temporalSigmaLong=<s>  # optional
```

Response JSON:

```json
{
  "mjd": [...], "raw": [...], "smoothed": [...],   // smoothed absent when temporalMode omitted
  "stats": { "raw": {"min":, "max":, "p1":, "p99":},
             "smoothed": {"min":, "max":, "p1":, "p99":} },
  "cadenceSeconds": <median>, "nTotal": <native samples in range>, "stride": <int>
}
```

- Raw series: for each selected native index, the SAME difference/ratio
  computation as the frame route, evaluated only on the patch (read the
  patch from cached frames when present; else compute the full difference
  frame through the existing cached path — do NOT invent a separate
  partial-file reader).
- Smoothed series: the design-doc temporal filter (Δt-weighted Gaussian,
  same decimation/cap rules) applied to the raw patch series — computed on
  the series directly (cheap, 1-D), not via filtered frames.
- Stride: if native samples in [startMjd, endMjd] > maxPoints, take every
  k-th index; report k. Stats computed on the STRIDED series.
- NaN patches (off-detector) return nulls in arrays; stats ignore nulls.
- Cost note: worst case (cold caches, 400 AIA points) is 400 × ~30 ms —
  acceptable as an explicit user action; the response is cacheable in the
  session texture-cache keyed by the full param tuple. No prefetching.

## Frontend

- Click mapping: reuse the existing panel transform + `pixelToWorldAffine`
  to convert the click to base-layer image pixel coordinates (the inverse
  of the lasso/ROI mapping already present).
- Card: a dismissible floating card anchored bottom-right of the workspace
  (same layer/styling family as the error toast, wider). Contents:
  - Header: `<layer label> @ (x, y)` + patch size + × close.
  - Canvas chart (~460×160): raw series as a thin line (muted), smoothed
    series as an accent line (cyan), current master-cursor time as a
    vertical line; y-axis min/max labels, x-axis start/end UTC labels.
    Plain canvas, no new dependencies.
  - Stats row: `raw p1/p99` and `smoothed p1/p99` values.
  - Buttons: `Apply smoothed p1/p99 as vmin/vmax` (falls back to raw
    p1/p99 when no temporal filter is active) — writes vmin/vmax into the
    probed layer via the existing updatePanelLayer path; `Refresh`
    (re-fetch after the user changes layer settings).
- The card re-fetches automatically when the probed layer's operation,
  reference, lag, temporal settings, or freqIndex change (debounced
  ~300 ms), so tuning σ in the inspector live-updates the smoothed curve.
- The probe is transient state: not persisted to session JSON.

## Definition of done

- Clicking a pixel on either panel opens the card with both curves within
  ~2 s warm; clicking elsewhere re-probes; Esc/× closes; marker tracks
  pan/zoom.
- Curves visibly correspond to the panel (probe the CME front on the ratio
  layer: raw shows the transient + flicker; smoothed shows the low-pass
  version with the current σ).
- Apply-button sets the layer's vmin/vmax and the panel re-renders with a
  sensible display range (the washed-out-gray failure mode is gone).
- Changing σ_short in the inspector updates the smoothed curve without
  re-clicking.
- Backend: route validated (bad x/y → 422; out-of-range window → empty
  arrays not 500); pytest additions: patch mean correctness on a synthetic
  sequence, stride math, stats p1/p99, temporal smoothing of a known
  series matches the frame-filter weights. `npm run build` + backend tests
  pass. No new dependencies.
