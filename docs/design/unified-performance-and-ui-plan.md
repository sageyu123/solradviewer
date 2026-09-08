# Unified plan: performance, independent timelines, and layered panels

Status: design + implementation plan. **Supersedes** both
[`EFFICIENCY_IMPROVEMENT_PLAN.md`](../../EFFICIENCY_IMPROVEMENT_PLAN.md) and
[`independent-timeline-and-layered-panels.md`](independent-timeline-and-layered-panels.md),
which remain as background. No code has been changed yet.

Why one plan: both documents converge on the same subsystem — how frame
requests are keyed, scheduled, and cached. Implementing the efficiency fixes
against today's three hardcoded URL builders and then rebuilding them for the
layer model would do the work twice. Instead, the request scheduler and cache
keys are built once, shaped for the layer world, and the backend quick wins
land first because they are independent of any UI decision.

## 0. Instructions for the implementing agent

- **Repos.** This app: the repo containing this file. Reference app (read-only
  inspiration, never modified):
  `/path/to/projects/ovrolwa-rfr-corr-app`.
  All `PreviewCanvas.tsx` / `ContextLayerStack.tsx` / `TimeSeriesChart.tsx` /
  `radioColormaps.ts` citations below are paths under that repo's
  `frontend/src/`.
- **Order.** Implement **P0, then P1, then stop and report** with the
  acceptance measurements (§3, §2). P2–P6 are separate efforts; do not start
  them in the same change set. One phase = one commit (or one commit per
  numbered item within P0).
- **Scope guardrails.** No new frontend dependencies (the app and the
  reference app both use plain React + vanilla CSS; keep it that way). No
  repo restructuring, no file renames, no drive-by refactors beyond what a
  phase names. Keep all existing `timeIndex`-based routes working unchanged
  until §4 explicitly aliases them. Match existing code style; backend
  docstrings follow the existing Sphinx style.
- **Verify with.** Backend tests: `pytest sad_eovsa_tool/backend/tests/` (all
  must pass; add tests named in each phase's DoD). Manual/perf checks: run
  `./run_app.sh 20250328` (requires `/path/to/data` mounted) and use
  the curl timing pattern
  `curl -s -o /dev/null -w '%{http_code} %{time_total}s %{size_download}B\n' '<url>'`
  against `http://127.0.0.1:8010` with a session created by the app. The
  frontend has no test suite; verify frontend changes by building
  (`npm run build` in `frontend/`) and by the behavioral checks in each DoD.
- **When the plan is ambiguous**, prefer the smallest change that satisfies
  the phase's DoD, and record the ambiguity in your report rather than
  expanding scope.

Evidence base (all verified against source or measured on `./run_app.sh 20250328`):

- Measured hot path: fresh AIA frame ≈ 180 ms (62% PNG encode), radio full-cube
  decompress ≈ 190 ms vs ≈ 10 ms single-band section read, contour overlay
  ≈ 50–550 ms cold / ~ms warm; a current-frame request inside the uncancelled
  prefetch burst degrades ~0.19 s → ~2.4 s.
- The reference app (`ovrolwa-rfr-corr-app`), whose UI and scrubbing speed we
  are adopting, achieves responsiveness **without** client-side science
  rendering — it is server-backed plus disciplined request management
  (§2.1). Its layer-stack and status UI are the interaction spec for §5.
- Structural audit of `App.tsx` confirms `ImagePanel` is already nearly
  layer-agnostic; the real rework is App-level state (three global role slots →
  per-panel layer lists) and the master-clock couplings listed in §6.1.

---

## 1. Target architecture

Three separated concepts (kept from the Codex proposal):

1. **Source** — dataset descriptor: files, metadata, WCS, native time/frequency axes.
2. **Layer** — render recipe: `sourceId` + operation/reference + sampling policy
   + display settings. `kind: image | contours | spectrogram`.
3. **Panel** — ordered composition: one base image layer + N overlay layers.

Plus one cross-cutting subsystem both plans need:

4. **Request scheduler** (frontend) + **source-native frame API** (backend) —
   every displayed layer resolves the timeline cursor to a *source-native
   sample index* first, and all requests, caches, dedup, and suppression are
   keyed on the resolved identity:
   `(sessionId, sourceId, resolvedIndex, operation, reference, displayParams, kind)`.

The resolved-identity key is the linchpin that serves both goals:

- **Performance**: today `eovsa/frame.png` is keyed by the *AIA* `timeIndex`
  (`App.tsx:898`, resolved server-side), so multiple AIA indices that map to
  the same radio frame produce distinct URLs — duplicate requests and
  duplicate cache entries for identical images. Keying by resolved native
  index collapses them and makes "suppress request when the resolved index
  didn't change" trivial.
- **UI**: a radio-master timeline stepping at 1 s cadence resolves the AIA
  layer to the *same* AIA frame for ~12 consecutive steps; with resolved-key
  suppression those steps cost zero AIA requests.

## 2. Phase P1 — Frontend request scheduler (fast sliding)

Build once, using the reference app's proven blueprint
(`ovrolwa-rfr-corr-app/frontend/src/components/PreviewCanvas.tsx`), keyed as
in §1 so it survives the later layer migration unchanged.

Blueprint, with reference citations:

| Pattern | Reference implementation | Adopt as |
|---|---|---|
| Debounced cursor commit | `FRAME_REQUEST_DEBOUNCE_MS = 16` (PreviewCanvas.tsx:33) | ~16 ms debounce between slider motion and fetch; slider thumb moves instantly on local state |
| Stale-request invalidation | sequence tokens (`sceneSequenceRef`, checked before applying results) **and** `AbortController` per effect | Replace `new Image()` loads (`App.tsx:2634`, `:2673`) with `fetch(url, {signal}` → blob → `createImageBitmap`; abort the previous controller on every new commit |
| Current-first, overlays later | radio scene fetched immediately, context layers delayed `CONTEXT_REQUEST_DELAY_MS = 120` (:32) | Base layers fetch immediately; contour overlays deferred ~120 ms and skipped entirely while the pointer is dragging (rendered on settle) |
| Bounded caches | module-level `Map` LRU: scenes 24, images 12, context records 8192; explicit `clearPreviewCaches()` on dataset change | Keep the existing 96-entry image cache but key it by resolved identity; clear on session change; consider byte-budget rather than count |
| Idle neighbor prefetch | `PREFETCH_RADIUS = 4`, `PREFETCH_IDLE_MS = 80` (:996-1051) | Prefetch starts only after the current frame has rendered; bounded by visible layers and playback direction; includes contour overlays (today they are never prefetched, `App.tsx:1219-1225`) |
| Background warm-up pool | 4-worker queue prioritized by distance from cursor, 3 retries (:1350-1434) | Optional later: warm the ± window during idle at bounded concurrency |
| RAF-coalesced drawing | `scheduleDraw` (:903-909) | Wrap `ImagePanel`'s `draw()` effect (`App.tsx:2731`) in RAF |
| Honest staleness | "showing stale frame N while loading M" message | Keep last-good frame on canvas with a subtle stale badge instead of flicker/blank |
| Playback | `setTimeout` at `1000/playbackFps`, fps user-selectable 0.5–50 (App.tsx:505-522, :1375-1382) | Replace the fixed 180 ms `setInterval` (`App.tsx:1207`) with fps-selectable, load-chained advance (next step waits for current frame's arrival) |

Also from the audit (cheap, same phase): give the dependency-less
resize-observer effects (`App.tsx:2714`, `:1975`) a `[]` dependency array;
memoize URL builders.

**P1 definition of done**
- Dragging across 50 uncached frames and releasing renders the settled frame
  < 300 ms; requests per drag ≈ frames paused on, not pixels moved; DevTools
  network tab shows stale requests as cancelled.
- The scheduler lives in its own module (e.g. `frontend/src/frameScheduler.ts`),
  not inline in `App.tsx`, with the cache keyed by a single exported
  `requestKey(...)` function shaped as §1 — this is the piece P4 must reuse
  unchanged.
- Behavior checks: pausing mid-drag shows the frame under the thumb;
  releasing after a fast drag never leaves a stale frame on screen; playback
  with an empty cache advances without skipping (load-chained); prefetch
  fires only after the current frame rendered, and includes the contour
  overlay when contours are enabled.
- `npm run build` passes; no new dependencies in `frontend/package.json`.

## 3. Phase P0 — Backend quick wins (land first, independent of all UI work)

Carried over from the efficiency plan; unchanged in content, restated as one
phase because none of them depend on the UI decisions:

1. **Faster PNG encode** — `compress_level=1` in `_render_png`
   ([data.py:328](../../sad_eovsa_tool/backend/data.py)): ~109 → ~30 ms.
   Optional: grayscale-mode (`L`) PNGs for `cmap=gray`; JPEG only if lossless
   is waived.
2. **Single-band radio section reads** — read only `[freqIndex]` for the
   single-band texture path (`texture_for_aia_time`, data.py:1003): 190 → 10 ms.
   Full-cube reads remain for contours/extraction.
3. **In-flight FITS read dedup** — per-index future/lock in `_read_file`
   (data.py:630, :837) so concurrent frame + contour requests decompress once.
4. **Cache hygiene** — FIFO → LRU for both `_data_cache`s; split frame-data
   cache from rendered-texture cache (display-param changes stop recomputing
   science arrays); cache the EOVSA→AIA affine (data.py:1785) on
   `(timeIndex, xOffset, yOffset)`; add the missing `AiaCube._mean_reference`
   cache (data.py:433); delete the dead `data_cube()` path (data.py:855) once
   section reads land.

**P0 definition of done**
- Curl timings on a fresh session (uncached indices): AIA frame ≤ 110 ms,
  radio frame ≤ 25 ms; contour overlay not regressed.
- Item 3 verified by a counter/log on `fits.open` calls: a slider tick with
  contours enabled opens each radio FITS file at most once.
- Existing tests in `sad_eovsa_tool/backend/tests/` all pass. New tests:
  single-band section read returns arrays identical to slicing the full-cube
  read (`np.array_equal`) for a few `(timeIndex, freqIndex)` pairs and for
  each difference operation; LRU eviction keeps the most-recently-used entry
  under scripted access patterns that FIFO would evict.
- PNG output visually identical (same pixels, different compression);
  contour PNGs and extraction tables byte-identical before/after items 2–4.
- No public function signatures removed; `data_cube()` deletion only if
  nothing references it (verify by grep, not assumption).

## 4. Phase P2 — Source-native time addressing (backend contract)

The API bridge between performance and the timeline UI. Add to the frame and
overlay routes (keep `timeIndex` routes as aliases during migration):

Routes gaining the new addressing (all keep their current `timeIndex` form
working as a legacy alias; a request may use `timeIndex` **or** `sampleMjd`,
never both — reject both-present with 422):

| Route | New params |
|---|---|
| `GET /api/sessions/{id}/sources/{sourceId}/frame.png` | `sampleMjd: float`, `samplingPolicy: nearest\|previous\|next` (default `nearest`), `maxOffsetSeconds: float` (optional) |
| `GET /api/sessions/{id}/sources/{sourceId}/overlay-contours.png` | same three |
| `POST .../roi`, `.../roi/projection` | accept `sampleMjd` alongside legacy `timeIndex` |

Resolution semantics (unit-test these exactly):

```text
nearest:  argmin |S[i] - t|; on a tie, the lower index wins
previous: greatest i with S[i] <= t; none exists -> out-of-tolerance
next:     least i with S[i] >= t;    none exists -> out-of-tolerance
tolerance: if maxOffsetSeconds given and |S[i] - t| > it -> out-of-tolerance
```

- Every 200 response (including legacy `timeIndex` requests) carries
  resolution headers: `X-Resolved-Index` (int, native to `sourceId`),
  `X-Resolved-Mjd` (float), `X-Offset-Seconds` (signed float,
  `resolved − requested`). The frontend keys caches by resolved identity and
  suppresses repeat requests with these.
- CORS: add the three headers to `Access-Control-Expose-Headers`, or the
  frontend cannot read them.
- Out-of-tolerance → **204 No Content** with the headers omitted; the layer
  is marked unavailable client-side. Never silently extrapolate a frame.
- Operation references resolve on the **layer source's own axis** (a radio
  `previous` uses the prior radio sample; the mean reference is stored as MJD
  start/end, master-grid independent). This fixes a deep existing bug-shaped
  coupling: today even the radio source's mean-reference indices are indices
  into `meta.aia.timeMjd` (`App.tsx:864`, `:1327-1328`, and every
  `normalizeDifference(..., meta.aia.times.length)` call site).
- Optional `maxWidth`/`maxHeight` params land here too (downsample after the
  science operation, before colormap): scrub at panel resolution
  (~28 ms/0.5 MB), full-res crop after zoom settles, crop box quantized to
  protect cache hit rates.

## 5. Phase P3 — Master timeline; Phase P4 — layered panels (the UI)

### P3: Timeline control

As proposed by Codex, confirmed feasible by the audit, with these
refinements:

- Master = any source with a time axis (Context/AIA, Radio/EOVSA, Dynamic
  Spectrum). Default **context** for compatibility; persisted per saved
  session. Playback and arrow keys step the master's native grid.
- Per the reference app's actual design (which is simpler than "independent
  timelines" suggests): there is **one cursor on one master axis**; every
  layer independently *resolves* against its own native axis with its
  sampling policy. No merged/synthetic grids in v1.
- Cursor readout: UTC + master position/count; each visible layer shows its
  resolved native timestamp and signed Δt (in the panel header or layer
  inspector). Out-of-tolerance layers dim with an "unavailable" badge — hold
  the last frame visually, never blank silently, never fake a sample.
- **Scrub on the spectrogram.** The spectrogram panel is already a
  time-frequency chart with a cursor; adopt the reference
  `TimeSeriesChart` interactions (drag the cursor line to scrub, wheel to
  zoom time, drag to pan, keyboard Left/Right/Home/End) instead of keeping
  the slider as the only scrub surface. `selectSpectrogramTime`
  (`App.tsx:1386`) already snaps clicks to the clock — generalize it to set
  the master cursor.

### P4: Per-panel layer composition

Feasibility (from the audit): `ImagePanel` already takes generic
`imageUrl` + one `overlayUrl` + its own WCS affine — it is ~90% layer-agnostic.
The rework is:

- **State**: introduce
  `panels: Record<panelId, { baseLayerId, overlayLayerIds: string[] }>` +
  a `layers: Record<layerId, Layer>` map. Layer schema (canonical, inlined
  here so this document stands alone):

  ```json
  {
    "id": "aia-ratio",
    "sourceId": "aia-131",
    "kind": "image",            // image | contours | spectrogram
    "operation": "ratio",       // none | subtract | ratio
    "reference": "previous",    // previous | base | mean
    "cadenceSeconds": 24,
    "meanStartMjd": null,        // MJD pair, only when reference == "mean"
    "meanEndMjd": null,
    "sampling": { "policy": "nearest", "maxOffsetSeconds": 6 },
    "display": { "vmin": 0.5, "vmax": 1.5, "cmap": "gray",
                 "scale": "linear", "opacity": 1 }
  }
  ```

  Image layers may add `frequencyIndex` (radio cubes). Contour layers carry
  all-band level/reference/filled/opacity settings plus a frequency palette
  id. Legacy `differenceMode`/`useRunningDiff` remain accepted input aliases
  that normalize into `operation`/`reference` on load. Layers are
  **per-panel instances** that share a `sourceId` (decision #4, §7) with a
  "copy to other panel" affordance.
- **URL building**: collapse the three ~duplicate builders
  (`aiaFrameUrl`/`eovsaFrameUrl`/`eovsaContourUrl`, `App.tsx:843-895`) into
  one function parameterized by layer.
- **ImagePanel**: `overlayUrl: string` → ordered `overlayLayers[]`; the
  single overlay-load effect becomes keyed per layer; draw in order inside
  the existing clip block. The one instrument-coupled branch
  (`App.tsx:2771-2772`: tracks on "aia", sources on "eovsa") becomes
  layer-role-driven. `colorbar` becomes per-overlay.
- **Layer stack UI**: adopt the reference `ContextLayerStack` row design
  verbatim as the interaction spec — eye toggle, opacity slider (0–1 step
  0.05, disabled when hidden), drag handle **plus** up/down arrow buttons
  (keyboard accessibility), remove button, add-layer dropdown grouped by
  source kind, sensible layer cap (8). Reference:
  `ContextLayerStack.tsx:188-393`; layer type `types.ts:203-209`.
- **Compositing note**: the reference app composites context layers with
  additive `'lighter'` blending — appropriate for its overlaid imagery, not
  for ours. Use `source-over` with per-layer alpha for image layers;
  contours draw last.
- Panels keep the **shared** `solarView` (synchronized pan/zoom is the point
  of a comparison tool) — add an unlink toggle later only if requested.
  Spatial WCS alignment remains one shared solar frame.

## 6. Migration and saved state

### 6.1 Coupling punch list (must-fix during P3/P4; from the structural audit)

| Coupling | Site | Migration |
|---|---|---|
| Radio/mean difference indices live on the AIA axis | every `normalizeDifference(..., meta.aia.times.length)`; `differenceParams(radioDifference, meta.aia.timeMjd)` (`App.tsx:864`) | Store mean references as MJD pairs (P2 contract); convert legacy saved indices via `meta.aia.timeMjd[i]` on load |
| Tracking overwrites the clock | `stepFeature` → `setTimeIndex(row.frame_index)` (`App.tsx:1377`) | Tracking rows carry MJD; stepping moves the master cursor via MJD, valid under any master |
| Extraction window is AIA-index space | `runEovsaExtraction` sends `startIndex/endIndex` (`App.tsx:1329-1330`) | Send MJD bounds; backend resolves on the radio axis |
| Spectrogram click snaps the AIA index | `selectSpectrogramTime` (`App.tsx:1386-1389`) | Sets master cursor (MJD) instead |
| ROI keyed by `timeIndex`/`freqIndex` | `refreshRoiProjection` (`App.tsx:1179`), `saveRoi` (`App.tsx:1264`) | Key by MJD + layer; backend resolves per source |
| One source per role, hardcoded | role slots (`App.tsx:812-819`), role `<select>` (`:1541`) | Roles stay as *defaults* that seed the initial layer composition; panels/layers become the composition authority |

### 6.2 Saved-state rules (kept from the Codex doc, tightened)

1. New optional `ui.timeline`, `ui.layers`, `ui.panels`; version-1 fields
   written alongside for ≥ one release.
2. Legacy inference: master = context; cursor = `aia.timeMjd[ui.timeIndex]`;
   left = context base + contour overlay iff `showEovsaContours`; right =
   radio base. Roles/differences seed the layer instances.
3. **Index→MJD conversion is mandatory** for mean references and extraction
   windows (not just the `differenceMode` alias conversion the original doc
   listed).
4. Missing master source on load → fall back to context with a non-blocking
   warning; never substitute silently.
5. Manifests stay source declarations; optional layer/panel defaults later; a
   manifest without them must produce the legacy view.

## 7. Colormaps (Phase P5) and resolved open decisions

Colormap plan adopted from the Codex doc unchanged (scalar registry gains
`parula`/`inferno`/`rdylbu`; separate frequency-palette registry for radio
contours with **low-frequency = warm** contract; reference endpoint fixtures
Parula `#f9fb0e`→`#352a87`, RdYlBu `#a50026`→`#313695`, Viridis
`#fde725`→`#440154` — all verified against
`radioColormaps.ts` / `radioColormaps.test.ts`). Implementation note: Parula
is not in matplotlib; adopt the reference app's inline 11-stop LUT
(`radioColormaps.ts:29-33`, dependency-free, test-pinned) as the canonical
shared definition, registered identically in the backend.

Answers to the original doc's open decisions:

1. **Master default**: context, always; persisted per session. Auto-densest
   is a surprise, not a feature.
2. **Max offset default**: half the layer source's native cadence (AIA 12 s →
   6 s). Out-of-range: hold last frame, dimmed, with signed Δt badge — never
   blank, never silent.
3. **`previous` semantics**: elapsed-time lag on the source's native axis
   (matches the existing `diffSeconds` behavior); no interpolation.
4. **Layer identity**: per-panel instances sharing `sourceId` (copy-on-edit
   is unsurprising); a "copy to other panel" button covers the shared case.
5. **Parula LUT**: the reference app's inline table; provenance comment in
   the shared module.
6. **Warm-cool convention scope**: frequency-coded contours and their
   colorbar only. Scalar intensity images keep canonical order unless the
   user picks a reversed map explicitly.

## 8. Phase order, dependencies, and acceptance

```
P0 backend quick wins ──────────────┐  (no dependencies; land immediately)
P1 request scheduler ───────────────┤  (independent of P0; keys shaped for layers)
P2 source-native addressing ────────┤  (backend; enables P3 + resolved-key caching)
P3 master timeline ── needs P1+P2 ──┤
P4 layered panels ─── needs P3 ─────┤
P5 colormaps ──────── independent ──┤  (can land any time after P4 UI exists)
P6 polish: design tokens, status centralization, keyboard map, reduced-motion
```

Per-phase acceptance criteria: P0/P1 as in §2–§3. P2: resolution-semantics
unit tests (tie-break, previous/next at array edges, tolerance → 204),
headers present and CORS-exposed, legacy `timeIndex` requests byte-identical
to before. P3: radio-master playback visits every native radio timestamp
while the AIA layer shows nearest with a visible Δt; legacy version-1 JSON
still opens with AIA master and identical frame choices. P4: the left/right
example (left = AIA ratio + contours, right = AIA original + contours) can
be composed without touching global roles; layer order/visibility/opacity/
operation/display/frequency survive export → reload; ROI and tracks stay
WCS-aligned on both panels. P5: palette endpoint/direction tests pass
(fixtures in §7); scalar maps unaffected.
Whole-plan gate: a representative radio-master playback of the 20250328
event visits every native radio timestamp with median settled-frame latency
no worse than today's cold baseline, and repeated resolved indices issue
zero duplicate requests.

## 9. UI polish adopted from the reference app (P6, cheap wins)

These are part of why the reference UI reads as "better"; none block the
phases above:

- CSS custom-property design tokens (`--panel`, `--border`, `--text`,
  accent colors) instead of scattered hex values; `prefers-reduced-motion`.
- One centralized status affordance: topbar status dot + determinate
  progress (`completed/total`) for long operations, toast-style dismissible
  errors — replacing the bottom status text bar.
- Keyboard map: Left/Right frame step (guarded for inputs/dialogs),
  Home/End, Space play/pause; document in a tooltip.
- Empty/first-run states: explicit "no dataset" panel with a load CTA.
- `beforeunload` guard when unexported ROI/tracks/extractions exist.
