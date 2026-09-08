# Efficiency Improvement Plan — Time-Slider Performance

Status: **plan only, no code changed yet.**
Basis: two independent investigations (Claude multi-agent code analysis + live
profiling; Codex CLI profiling) on the `20250328` dataset
(`./run_app.sh 20250328`), which agreed on the diagnosis.

## 1. Diagnosis

Perceived slowness while sliding over time comes from two compounding causes:

1. **Frontend request pileup.** Every drag pixel commits `timeIndex` and fires
   up to 11 requests (current AIA + radio + contours, plus 4 neighboring AIA
   and 4 neighboring radio prefetches). Stale requests are visually ignored
   but **never cancelled**, so the backend keeps processing them and the
   browser's 6-connections-per-origin limit queues the frame the user actually
   wants behind discarded 1.5 MB downloads.
   Measured: a solo current-frame request completes in ~0.19 s; the same
   request inside a realistic current+prefetch burst takes **~2.4 s (12×
   penalty)**.
2. **Backend per-frame cost is dominated by PNG encoding, not science math.**

### Measured cost breakdown (per fresh AIA frame, ~180–200 ms total)

| Step | Time | Share |
|---|---|---|
| 2× FITS read + Rice decompress (current + reference frame) | ~53 ms | ~29% |
| Running-ratio difference arithmetic | ~3 ms | ~2% |
| Normalize + colormap | ~14 ms | ~8% |
| **PNG encode (PIL default level, 1667×754 RGBA → 1.5 MB)** | **~109 ms** | **~62%** |

### Radio (EOVSA) path

| Work | Time |
|---|---|
| Full 52-band cube FITS decompress | ~190 ms per file |
| Single-band FITS section read (alternative, measured) | ~10 ms |
| Cube subtraction | <1 ms |
| 52-band contour tracing | ~11 ms |

Backend caches work well once warm (repeat hits ~5 ms), so the problem is
cold-frame cost and request storms, not steady-state.

## 2. Implementation phases

Ordered by (impact ÷ effort). Phases 1–4 are small, independent, and should
take a cold slider step from ~2.4 s worst case to well under 100 ms.

### Phase 1 — Request cancellation + current-frame priority (frontend)

Largest visible win: eliminates the 12× burst penalty.

- **Abort stale image requests.** `ImagePanel` loads frames via `new Image()`
  and only ignores late results via `imageRequestRef`
  ([App.tsx:2634](frontend/src/App.tsx:2634), overlay twin at
  [App.tsx:2673](frontend/src/App.tsx:2673)). Switch to `fetch(url, {signal})`
  + `AbortController` (create object URL / `createImageBitmap` for display),
  aborting the previous controller whenever the URL changes. Minimum fallback:
  clear the old `Image.src` so the browser can drop the socket.
- **Decouple slider position from fetch commit.** The range input calls
  `setTimeIndex` on every drag pixel ([App.tsx:1732](frontend/src/App.tsx:1732)).
  Keep a local visual index for the thumb; commit the fetch-driving index via
  rAF-batching or a ~50–100 ms debounce.
- **Prefetch only when idle.** The ±2-frame prefetch
  ([App.tsx:1215](frontend/src/App.tsx:1215)) resets a 40 ms timer every tick
  (so it never fires mid-drag) yet still bursts on every pause. Start prefetch
  only after the current frame has loaded (or via `requestIdleCallback`), and
  also prefetch the contour overlay when contours are shown (currently
  missed).
- **Postpone contours during active drags** — request them when the pointer
  pauses or releases.

Verification: drag rapidly across ≥30 uncached frames with DevTools network
open; confirm stale requests show as cancelled, the settled frame renders in
<300 ms, and total requests per drag ≈ frames paused on, not pixels moved.

### Phase 2 — Single-band radio reads (backend)

`EovsaSequence` reads and diffs the entire `(52, 256, 256)` cube to display
one band ([data.py:908](sad_eovsa_tool/backend/data.py:908),
[data.py:941](sad_eovsa_tool/backend/data.py:941)); measured 190 ms vs 10 ms
for an astropy section read of one band.

- For the single-band texture path (`texture_for_aia_time`,
  [data.py:1003](sad_eovsa_tool/backend/data.py:1003)), read only `[fidx]` via
  the compressed-HDU section API and compute the difference for that band
  only.
- Keep full-cube reads for the all-band contour path
  ([data.py:1803](sad_eovsa_tool/backend/data.py:1803)) and for
  peak-cache/extraction scans.
- Cache key: extend `_data_cache` to key on `(idx, band)` for section reads,
  or keep a separate per-band cache, so single-band and full-cube entries
  don't collide.

Verification: fresh radio frame endpoint time drops from ~200 ms to ≲25 ms;
contours and extraction results unchanged (byte-compare a few PNGs and one
extraction table before/after).

### Phase 3 — Faster PNG settings (backend)

One-line-scale change in `_render_png`
([data.py:328](sad_eovsa_tool/backend/data.py:328)); PNG encode is ~62% of
AIA frame cost.

- Set PIL `compress_level=1`: measured ~102→~30-36 ms, output ~5–70% larger
  (noisy ratio frames compress poorly regardless).
- Optional follow-ups, in order of preference:
  - **Grayscale mode**: the AIA panel uses `cmap=gray`; encoding mode `L`
    instead of RGBA quarters the raw pixel bytes (encode + transfer + browser
    decode all shrink). Needs a small branch on colormap.
  - **JPEG for the AIA panel** (measured 4 ms, 0.84 MB): only if lossless is
    not required and no alpha is needed; contours overlay must stay PNG
    (alpha).
- WebP measured ~158 ms steady-state — not worth it here.

Verification: endpoint timing (curl `time_total`) for a fresh frame ≈
90–110 ms after this phase alone; visual spot-check of frames.

### Phase 4 — In-flight FITS read deduplication (backend)

The radio frame and contour endpoints can request the same FITS file
concurrently on separate threadpool threads, both miss `_data_cache`, and
decompress the same file twice.

- Guard `_read_file` with a per-index in-progress registry (e.g. dict of
  `Future`/`threading.Event` under a lock): first caller reads, concurrent
  callers wait on the same result.
- Apply the same pattern to `AiaFitsSequence._read_file`
  ([data.py:630](sad_eovsa_tool/backend/data.py:630)).

Verification: log or counter on actual `fits.open` calls; slider tick with
contours on triggers one read per file, not two.

### Phase 5 — Resolution-aware AIA rendering; FOV crop after settle (backend + frontend)

The original "skip pixels outside the FOV" idea — valid, with a measured
caveat: the AIA files use row-oriented compression tiles, so cropping barely
reduces decompression time; the savings come from generating, shipping, and
decoding fewer PNG pixels (full PNG ~113 ms / 1.54 MB vs half-size ~28–30 ms
/ 0.44–0.63 MB).

- Add optional `maxWidth`/`maxHeight` (or `scale`) query params to the frame
  endpoints; downsample after the difference computation, before
  normalize/colormap, to ≈ the panel's displayed pixel size.
- Frontend sends its panel size; requests a full-FOV downsampled frame during
  scrubbing, and only requests a cropped/full-res FOV **after zoom/pan
  settles** (debounced), never per mouse move.
- Cache note: crop/zoom params fragment both the backend texture cache and
  the browser HTTP cache — quantize the crop box (e.g. snap to a grid and to
  a few zoom steps) to keep hit rates up.

Verification: scrub at 1× zoom — payloads ≈0.4–0.6 MB and encode ≤30 ms;
zoom in — after settle, displayed resolution matches full-res crop.

### Phase 6 — Cache hygiene (backend)

Lower urgency; matters for long back-and-forth scrubbing sessions.

- **FIFO → LRU** for `AiaFitsSequence._data_cache`
  ([data.py:640](sad_eovsa_tool/backend/data.py:640)) and
  `EovsaSequence._data_cache`
  ([data.py:841](sad_eovsa_tool/backend/data.py:841)): 32-entry FIFO evicts
  frames still in active use when scrubbing spans >32 timesteps; reuse the
  `OrderedDict`/`move_to_end` pattern already used by `_texture_cache`.
- **Decouple frame data from render params.** Texture cache keys include
  vmin/vmax/cmap/scale, so a display tweak recomputes the difference frame.
  Add a frame-level cache keyed on data params only; `_render_png` re-runs on
  display changes (~14 ms + encode).
- **Cache the EOVSA→AIA affine** (`_eovsa_to_aia_affine`,
  [data.py:1785](sad_eovsa_tool/backend/data.py:1785)) keyed on
  `(time_index, x_offset, y_offset)` — currently rebuilds two sunpy Maps +
  WCS transforms per contour request.
- **Add a cache for `AiaCube._mean_reference`**
  ([data.py:433](sad_eovsa_tool/backend/data.py:433)) mirroring the
  `AiaFitsSequence` 8-slot version (HDF5 backend + mean mode only).
- **Byte-budgeted caches**: the rendered-texture cache (384 entries) and
  browser image cache (96 entries) should be bounded by bytes rather than
  entry count before anyone shrinks them blindly — too-small caches make
  backward sliding slower.
- Remove or wire in the dead `EovsaSequence.data_cube()` fast path
  ([data.py:855](sad_eovsa_tool/backend/data.py:855)) — never called;
  Phase 2's section reads largely supersede it, so removal is the likely
  outcome.

### Phase 7 — Frontend render-path cleanups (minor)

- Dependency-less effects recreate a `ResizeObserver` + window listener every
  render ([App.tsx:2714](frontend/src/App.tsx:2714),
  [App.tsx:1975](frontend/src/App.tsx:1975)) — give them `[]`.
- rAF-coalesce the canvas `draw()` effect
  ([App.tsx:2731](frontend/src/App.tsx:2731)).
- Playback uses a fixed 180 ms `setInterval`
  ([App.tsx:1207](frontend/src/App.tsx:1207)) — faster than a cold frame can
  be served; advance only when the current frame has loaded (chain on image
  `onload`), which Phases 2–3 make fast enough for smooth ~5 fps playback.
- Optional: `React.memo` on `ImagePanel`/`SpectrogramPanel`, memoize URL
  builders — small, stacks with the above.

## 3. Non-goals / explicitly checked and fine

- **No matplotlib-figure-per-request antipattern** — PNGs are produced via
  PIL; matplotlib is used only as a stateless colormap lookup.
- **URL stability is good**: identical settings regenerate byte-identical
  URLs, so browser HTTP caching (`Cache-Control: max-age=3600`) works;
  no cache-busting bug to fix.
- **Difference arithmetic is cheap** (~3 ms AIA, <1 ms radio) — no need to
  optimize the science math itself.
- Concurrency across endpoints is healthy (parallel AIA+EOVSA ≈ max of the
  two, not the sum).

## 4. Known issue to track separately (correctness, not speed)

"Global" contour level references use opportunistically accumulated peaks
([data.py:1114](sad_eovsa_tool/backend/data.py:1114)): levels can shift as
more frames get visited in a session unless a full peak-cache refresh has
run. Worth fixing or documenting alongside Phase 6, but it is not a
performance item.

## 5. Acceptance criteria for the whole plan

Using `./run_app.sh 20250328`:

1. Dragging the time slider across 50 uncached frames and releasing shows the
   settled frame in <300 ms (was: up to ~2.4 s under burst).
2. Fresh single-frame endpoint times: AIA ≤110 ms (Phase 3) or ≤40 ms
   (Phase 5 at panel resolution); radio ≤25 ms (Phase 2).
3. Playback at default cadence no longer skips/queues on uncached spans.
4. Contour overlays, extraction tables, and exports are bit-identical (or
   visually identical where lossy encoding is explicitly chosen) to
   pre-change outputs.
