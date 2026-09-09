# Enhancement filters: radial coronal enhancement + temporal denoise

Status: design spec. Two phases: **Phase I (investigate/prototype)** on real
data, then **Phase R (implement)** after the supervisor reviews Phase I
numbers. Test dataset: `./run_app.sh 20250328` sources (AIA cutouts
1667×754 at 12 s cadence, EOVSA 52-band 256×256 at ~1 s cadence).

## Feature 1 — Radial coronal enhancement (JHelioviewer-style)

Goal: an inspector slider (0.0–3.0) on image layers that boosts off-limb
coronal structure, like JHelioviewer's radial filter.

### Algorithm

For each pixel with radial distance `r` from disk center (arcsec, from WCS):

```
factor(r) = exp(gamma * max(r - R_sun, 0) / H), H = 0.2 * R_sun
factor capped at 1e3; on-disk (r <= R_sun) stays exactly 1
I_enhanced = I * factor(r)
```

- `r` from the source WCS: center offset via `CRPIX/CRVAL`, scale via
  `CDELT`; `R_sun` from `RSUN_OBS`/`RSUN_REF` header or
  `sunpy.map` `rsun_obs` (Phase I decides which is present in these files).
  The AIA files are cutouts — disk center lies outside the array; the
  formula needs no on-frame center, only coordinates.
- `clip(..., 1.0, ...)`: on-disk pixels (`r <= R_sun`) are never modified.
- The exponential scale height is `H = 0.2 * R_sun`; cap the factor at `1e3`
  so noise at large r does not blow up the normalization.
- Applied to the frame array AFTER the difference operation (or the raw
  frame when operation = none), BEFORE normalization/colormap. It is
  scientifically meaningful mainly for `operation: none`; it remains
  allowed for diff/ratio (useful for off-limb running-diff CME fronts) —
  document, don't forbid.

### Wiring

- Radius-factor map cached per `(sourceId, shape, gamma)` — gamma quantized
  to 0.1 steps to keep the cache small; the `r/R_sun` base map cached once
  per source.
- New query param `radialGamma: float = 0.0` on the frame route; part of
  the texture cache key. 0.0 = identity, param omitted from URLs when 0.
- Layer schema: `display.radialGamma` (default 0). Saved-state round-trip.
- UI: "Coronal enhancement" slider 0.0–3.0 step 0.1 in the layer inspector,
  image layers only, with a numeric readout like the opacity slider.

### Phase I questions (prototype in scratchpad, no repo changes)

1. Which R_sun/center keywords do these AIA cutout headers actually carry?
2. Visual check: render frame ~193 original with gamma 0 / 1.5 / 3.0 —
   save PNGs side by side; the CME/off-limb loops at the east limb should
   emerge without saturating the disk.
3. vmin/vmax interaction: enhanced off-limb values change the histogram —
   does the default 0.5–1.5 ratio range still work, and what default
   vmin/vmax should an enhanced `none` layer use? Report a recommendation.
4. Cost: ms per frame for the multiply + map lookup at 1667×754.

## Feature 2 — Temporal denoise (band-pass) for subtract/ratio series

Goal: suppress per-pixel high-frequency noise in difference/ratio image
series while preserving real evolution (CME front, dimming).

### Algorithm — irregular-cadence Gaussian band-pass (time domain, not FFT)

For layer frames at native times `t_j` (seconds), difference/ratio series
`D_j` (computed exactly as today), the filtered frame at index `i`:

```
lowpass(sigma)_i = sum_j w_j * D_j / sum_j w_j
    where w_j = exp(-(t_j - t_i)^2 / (2 sigma^2)),
    j restricted to |t_j - t_i| <= 3 sigma, NaN pixels dropped per-pixel
    (weights renormalized over finite samples)

mode lowpass:   F_i = lowpass(sigma_short)_i
mode bandpass:  F_i = lowpass(sigma_short)_i - lowpass(sigma_long)_i
```

- Time-domain Gaussians, not FFT: robust to irregular cadence and gaps
  (weights use actual Δt), local (only ±3σ neighbor frames needed —
  existing frame caches and prefetch warm them), edge-safe (weights
  renormalize at series ends), NaN-aware.
- `sigma = period / (2 * pi)` conversion is NOT used; expose cutoffs
  directly as Gaussian σ in seconds to keep semantics simple, labeled
  "smoothing σ". Defaults (Phase I validates): AIA σ_short = 18 s
  (±3σ ≈ ±4-5 frames), σ_long = 300 s; EOVSA σ_short = 3 s, σ_long = 60 s.
- Note the bandpass output is signed and zero-centered even for ratio
  input; display defaults must switch to a symmetric range around 0
  (e.g. coolwarm, vmin=-x, vmax=+x). Phase I recommends x.

### Wiring

- Backend: `temporal_filter(mode, sigma_short, sigma_long)` applied at the
  frame level after the difference computation, before render. Filtered
  frames cached keyed by `(idx, diff-params, mode, sigma_short, sigma_long)`.
  Neighbor difference frames come from the existing per-frame path.
- Query params: `temporalMode: none|lowpass|bandpass`,
  `temporalSigmaShort: float`, `temporalSigmaLong: float`; texture and
  frame cache keys extended; omitted from URLs when mode = none.
- Layer schema: `temporal: { mode, sigmaShort, sigmaLong }` (default none).
- UI: inspector block for image layers with operation subtract/ratio:
  mode select + two σ inputs (long only for bandpass), seconds.
- Cost control: a cold frame now needs up to ~2×(3σ/cadence)+1 difference
  frames. For σ_short=18 s on AIA that is ±5 frames — fine. Cap the window
  at ±15 frames server-side regardless of σ; report in headers when capped.

### Phase I questions (prototype in scratchpad, no repo changes)

1. Extract a per-pixel time series from the AIA ratio series (a ~20×20
   patch on the CME front near frame 193, plus a quiet patch): quantify
   the noise — temporal power spectrum or lag-1 autocorrelation, and the
   fraction of variance above ~1/(2·cadence) Hz.
2. Apply the Gaussian low-pass and band-pass with the default σ values:
   report variance reduction in the quiet patch vs signal retention on the
   CME-front patch (e.g. peak amplitude of the transient before/after).
   Save before/after PNG pairs of a full frame.
3. Sweep σ_short in {12, 18, 30, 60} s: which best suppresses flicker
   without visibly smearing the front between consecutive frames?
4. Cost: ms per filtered frame with warm caches; frames needed cold.
5. Sanity: NaN handling at series edges and across the data gap (if any).

## Phase I outcomes (reviewed; these amend the sections above)

- **Feature 1 accepted as specified.** Cost: 0.14 ms/frame warm (cached
  factor map), 3.2 ms uncached. Display: when the user first sets
  `radialGamma > 0` on an `operation: none` layer, the frontend auto-fills
  scale = log, vmin = 10, vmax = 8000 (editable; ≈ frame p99.9 in DN).
  Do not implement per-frame percentile normalization — it defeats caching
  and causes brightness pumping during playback.
- **Feature 2 defaults revised from Phase I metrics** (σ_short = 18 s kept
  only 23% of the CME-front peak; per-pixel ratio series is spectrally
  white, so fast fronts attenuate almost like noise):
  - `sigmaShort` default **12 s** (82% quiet-variance reduction, 36% front
    retention); `sigmaLong` default **120 s**.
  - Long arm MUST be decimated: sample the ±3σ_long window with a stride
    chosen so ≤ 25 difference frames are used (Gaussian σ=120 s needs no
    12 s sampling). Short arm capped at ±15 frames.
  - Filtered frames MUST be cached per
    `(idx, diff-params, mode, sigmaShort, sigmaLong)` so scrub revisits
    skip recomputation (prototype cost without this: ~600 ms/frame).
  - Bandpass display default: cmap coolwarm, vmin = −0.2, vmax = +0.2
    (editable).
  - UI copy note: label the feature "Temporal smoothing" with modes
    Low-pass / Band-pass; tooltip states that fast-moving features are
    attenuated along with noise and the filter is most useful for slowly
    evolving structures and playback flicker.
- NaN handling validated (per-pixel weight renormalization; no new NaNs).

## Phase R (implement — only after Phase I review)

Implementation order: backend params + filters → frontend layer
schema/URL/inspector controls → saved-state round-trip. All existing DoD
conventions apply: `npm run build` and `pytest solradviewer/backend/tests/`
pass; new backend tests: radial factor map (on-disk untouched, cap
respected, gamma=0 identity) and temporal filter (constant series is a
fixed point; single-spike suppression ratio matches the analytic Gaussian
weight; NaN frames excluded from weights). Frame endpoints keep byte-stable
output when the new params are absent.

## Out of scope

MGN/WOW-style multiscale spatial filters (JHelioviewer's other options) —
a possible later feature; FFT/Butterworth filtering; applying temporal
filters to the spectrogram; contour-path filtering.
