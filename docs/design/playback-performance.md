# Playback performance: play from memory, not from the network

Status: implementation spec. Goal: JHelioviewer/OVRO-RFR-class playback —
the playback loop draws decoded frames from memory; fetching and decoding
happen ahead of the cursor, at panel resolution, never on it.

## Part A — Backend audit + completion (do FIRST, report findings)

Audit solradviewer/backend/data.py + app.py for whether these exist; any
missing item gets implemented now:

1. PNG encode speed: PIL save with compress_level=1 (or faster) in
   _render_png; additionally, when the colormap is grayscale ("gray" /
   "gray_r"), encode mode "L" instead of RGBA (quarter the raw bytes).
2. Single-band radio section reads for the single-band frame path
   (texture_for_aia_time-equivalent): reads only the requested band, not
   the 52-band cube.
3. In-flight FITS read dedup (per-index lock/future so concurrent requests
   decompress a file once).

Report each as PRESENT (file:line) or IMPLEMENTED-NOW.

## Part B — Resolution-capped frames

- Frame routes gain maxWidth/maxHeight int params: after all science
  processing (difference, radial, temporal) and BEFORE normalization/
  colormap, downsample the array with area-averaging (numpy block mean or
  cv-free stride tricks; anti-aliased enough for display) so the output
  fits within the cap while preserving aspect. Stats headers (X-Data-*)
  computed on the full-res array as today. Params absent = full res,
  byte-identical legacy behavior. Cache keys include the cap.
- Frontend: each ImagePanel requests frames capped at
  ceil(panelCssSize * devicePixelRatio) QUANTIZED UP to steps of 256 px
  (so resizes rarely change the cap and caches stay hot). Zooming past 1:1
  (panel zoom scale > 1) switches that layer's requests to full resolution
  (simple two-tier: capped for overview, full when zoomed in beyond 1x) —
  hysteresis so it does not flap at the boundary.
- Contour overlays are small already (line art) — leave uncapped.

## Part C — Playback buffer + byte-budgeted cache

- Replace the decoded-bitmap cache's count limit with a BYTE budget
  (default 500 MB; estimate bitmap bytes as width*height*4). LRU eviction
  by bytes. Playback-buffer entries and scrub entries share the one cache.
- While playing: maintain a lookahead window of
  N = clamp(playbackFps * 3, 10, 60) frames AHEAD of the cursor in the
  playback direction, across every visible layer's request identity, at
  bounded concurrency (6) via the existing scheduler; cancel lookahead
  outside the window when direction/fps changes. The existing ±neighbor
  prefetch stays for scrubbing.
- The playback advance (load-chained today) becomes cache-first: if the
  next frame's bitmaps are all cached, advance on schedule (setTimeout at
  1000/fps); if not, wait for arrival (current behavior) — with the buffer
  this should be rare after spin-up.
- HUD: while playing, if buffer occupancy ahead is < 50%, show a subtle
  "buffering" note in the spectrogram transport area (no layout shift).

## Verification (Definition of done)

- Backend: pytest passes with new tests: downsampled frame has correct
  shape/aspect and matches block-mean of the full array on synthetic data;
  legacy (no-cap) requests byte-identical; Part A items each covered by an
  existing or new test.
- Measured (report numbers): warm-cache frame latency at panel resolution
  vs full resolution (curl); decoded-bitmap memory per frame before/after.
- Behavioral: playback at 10 fps over a previously-uncached 100-frame span
  reaches steady cache-fed advancement after spin-up (no per-frame stalls);
  scrubbing still works; zoomed-in view still sharp (full-res tier kicks
  in); npm run build passes.
- No new dependencies; no regressions to: mirrors, masks, channel
  inspector, gutter, time axis, availability policy, radial/temporal
  filters, sfu levels, session round-trip.
