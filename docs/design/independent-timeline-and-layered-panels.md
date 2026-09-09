# Independent timelines and layered image panels

Status: design proposal (no application code changes are implied by this document).

## Why this change

The comparison workspace currently treats the AIA/context sequence as the clock. Radio data are sampled around that clock, which hides the radio sequence's finer cadence and makes the two image panels less reusable. This proposal separates three concepts:

1. a dataset source (files, metadata, WCS and native samples),
2. a render layer (source plus operation and visualization), and
3. a panel composition (the ordered layers shown in one image panel).

The result should allow a user to choose a radio-native timeline, while independently composing, for example:

| Panel | Base layer | Overlay |
| --- | --- | --- |
| Left | AIA running ratio / previous | EOVSA all-band contours |
| Right | AIA original | EOVSA all-band contours |

The same source can therefore appear more than once with different operations or panel-specific visualization settings.

## Current state and evidence

### Time is AIA-driven

`frontend/src/App.tsx:783-829` stores one `timeIndex` and derives `currentMjd` from `meta.aia.timeMjd`. `eovsaTimeIndex` is then the nearest radio sample (`:823-827`). Frame URLs for both images and the contour overlay are generated from that AIA index (`:843-899`). The slider and playback also operate on AIA indices (`:1721-1741`, with a 180 ms playback interval at `:1207-1213`). A radio-only step cannot currently visit every native radio time.

The backend metadata already exposes native source axes: AIA, EOVSA and spectrogram `timeMjd`/timestamps, plus EOVSA `freqGhz` (`solradviewer/backend/data.py:1526-1584`). The route implementation still presents a context-oriented `timeIndex`; source-specific nearest-time resolution should be made explicit rather than inferred from the AIA index.

### Source roles are global, not panel composition

The frontend keeps one selected source and role overrides (`sourceRoles`, `selectedSourceId`) and derives one context source, one radio source and one spectrogram source (`App.tsx:779-819`). Differences are also held per source (`sourceDifferences`, `:821-822`), but there is only one context image URL, one radio image URL and one global contour toggle (`:897-900`). The left panel receives the contour overlay (`:1834-1860`); the right panel cannot independently select that or another layer. The source rail's controls are useful dataset-level defaults, but they should not remain the only way to compose a view.

### Colormap gaps and frequency convention

Generic intensity controls currently offer Gray, Gray R, Viridis, Turbo, Magma and Coolwarm (`App.tsx:2386-2407`). Contour controls expose a smaller set (`App.tsx:256-268`). The backend already aliases `rdylbu` to Matplotlib `RdYlBu` (`solradviewer/backend/data.py:70-82`), but Parula and Inferno are not exposed consistently in the frontend. These are scalar-display palettes, not automatically frequency-aware palettes.

The reference app establishes the required frequency convention in `ovrolwa-rfr-corr-app/frontend/src/radioColormaps.ts:9-14`: low frequency is warm and high frequency is cool. Its palette tests pin the endpoints for Parula, Viridis and RdYlBu (`radioColormaps.test.ts:4-35`). The proposal adopts that direction for frequency-coded radio contours and their colorbar, while preserving ordinary low-value-to-high-value semantics for scalar intensity images.

### Useful reference patterns

The reference `ContextLayerStack` already models a layer as an id, visibility, opacity and optional cadence (`frontend/src/components/ContextLayerStack.tsx:9-31`, `:155-188`). Its preview resolves context samples to a requested frame with explicit cadence and nearest-frame selection (`frontend/src/components/PreviewCanvas.tsx:116-189`, `:233-265`). This is the right interaction pattern to adapt, but the comparison tool needs a general source + operation + visualization layer rather than a context-only list.

## Proposed user experience

### Timeline controls

Replace the implicit AIA clock with a compact Timeline control containing:

- **Master timeline**: a source selector (Context/AIA, Radio/EOVSA, Dynamic Spectrum, or any source with a time axis).
- **Sampling**: a global default (`Nearest`, `Previous`, `Next`) and an optional per-layer override.
- **Maximum time offset**: optional tolerance in seconds. If a source has no acceptable sample, the layer is marked unavailable rather than silently extrapolated.
- **Cursor**: UTC timestamp, master position/count and, for each visible layer, the resolved native timestamp and signed `Δt` in the panel header or layer inspector.

Selecting Radio/EOVSA as master makes playback and arrow keys step through every radio timestamp. AIA layers may repeat their nearest frame between radio samples; repeated image requests should be suppressed when the resolved AIA index has not changed. Selecting AIA preserves the familiar legacy behavior. Version 1 uses only the selected source's native timestamp grid; merged or synthetic grids are outside this change.

For a cursor MJD `t` and layer source axis `S`, resolution is deterministic:

```text
nearest: argmin |S[i] - t| (lower index wins ties)
previous: max S[i] <= t
next: min S[i] >= t
```

The layer's operation reference is resolved on the layer source's native MJD axis. For example, a radio `previous` difference uses the prior radio sample at the configured lag; it must not use the prior AIA index. A mean reference stores MJD start/end and is independent of the selected master grid.

### Independent panel composition

Each image-panel header gets a **Base layer** dropdown and a **Layers** menu. The base dropdown selects an image layer; the menu adds, removes, reorders and toggles overlay layers. A compact layer inspector exposes operation/reference, opacity, colormap, scale/range, radio frequency, contour levels and sampling policy for the selected layer. Panel state is independent: changing the right panel never changes the left panel's base layer or overlay visibility.

Suggested panel state:

```json
{
  "left":  { "baseLayerId": "aia-ratio", "overlayLayerIds": ["radio-contours"] },
  "right": { "baseLayerId": "aia-original", "overlayLayerIds": ["radio-contours"] }
}
```

The same layer id may be referenced by both panels when its settings are identical. If each panel needs a different opacity, contour level or sampling policy, create separate layer instances that share the source id. Spatial alignment/WCS remains the shared solar coordinate system; content selection must not create a second, silently shifted coordinate system.

### Layer data model

Sources remain dataset descriptors. A layer is a render recipe:

```json
{
  "id": "aia-ratio",
  "sourceId": "aia-131",
  "kind": "image",
  "operation": "ratio",
  "reference": "previous",
  "cadenceSeconds": 24,
  "sampling": { "policy": "nearest", "maxOffsetSeconds": 6 },
  "display": {
    "vmin": 0.5,
    "vmax": 1.5,
    "cmap": "gray",
    "scale": "linear",
    "opacity": 1
  }
}
```

`kind` is `image`, `contours`, or `spectrogram`. Image layers may specify `frequencyIndex` for radio cubes. Contour layers reference a radio source and carry all-band level/reference/filled/opacity settings plus a frequency color map. Spectrogram remains a distinct time-frequency layer. The existing operation vocabulary (`none|subtract|ratio` with `previous|base|mean`) is retained; legacy `differenceMode` remains an input compatibility alias.

### API direction

Keep existing `timeIndex` routes during migration, but add source-native addressing to layer render requests:

```text
source frame:       sourceId + sampleMjd + samplingPolicy + display/layer parameters
contour overlay:    sourceId + targetPanel/layer + sampleMjd + layer parameters
```

The backend resolves `sampleMjd` to a source-native index and returns (or exposes in metadata headers) the resolved index, timestamp and offset. A temporary frontend-only implementation may resolve against `sources[].time.timeMjd`, but backend resolution is the authority for FITS/HDF data and prevents index assumptions from leaking into saved sessions. A contour layer should no longer be hard-wired to the context target; it must accept the panel's base/world transform while preserving WCS alignment.

## Colormap design

Add the following ids to the shared scalar display registry and both frontend/backend validation paths:

- `parula` — MATLAB Parula, canonical scalar order;
- `inferno` — Matplotlib/Inferno, canonical scalar order;
- `rdylbu` — `RdYlBu`, canonical scalar order.

Keep one explicit frequency palette registry for radio-coded contours. Its contract is **low frequency / long wavelength = warm**, **high frequency / short wavelength = cool**. For palettes whose canonical order is cool-to-warm, reverse the LUT before contour assignment; `RdYlBu` is already warm-to-cool. The radio colorbar must label the low and high GHz endpoints in that same direction. Do not apply this reversal to scalar intensity maps unless the user explicitly chooses a reversed intensity map.

The reference's deterministic endpoint examples are useful acceptance fixtures: Parula low `#f9fb0e`, high `#352a87`; RdYlBu low `#a50026`, high `#313695`; Viridis low `#fde725`, high `#440154` when used as a frequency palette (`radioColormaps.test.ts:14-27`). Exact Parula LUT provenance should be recorded in the implementation module so exported sessions remain reproducible.

## Saved-state and migration plan

Introduce a new optional `timeline`, `layers` and `panels` object in the saved `ui` state. Keep the current version-1 fields while clients migrate:

1. If `ui.layers`/`ui.panels` exist, load them after validating source ids and capabilities.
2. Otherwise infer the legacy composition: master source = context source; cursor = `aia.timeMjd[ui.timeIndex]`; left base = context with `sourceDifferences[context]`; right base = radio with `sourceDifferences[radio]`; attach the existing `showEovsaContours` overlay to the left panel only.
3. Preserve `timeIndex`, `freqIndex`, `startIndex`, `endIndex`, ROI/tracks, offsets, `solarView`, display states and spectrogram ranges. Keep writing these aliases for at least one release so older clients can load newer exports.
4. Convert a legacy `differenceMode`/`useRunningDiff` only when no independent layer operation is present. New layers always write the explicit operation/reference fields.
5. When a saved master source is missing, fall back to context and show a non-blocking migration warning; never substitute a different dataset silently.

The manifest remains a source declaration. Layer/panel defaults may be added later as optional manifest fields, but a manifest without them must produce the same legacy view.

## Performance implications and safeguards

An independent radio master can increase cursor positions from the AIA count to the radio count. The existing local measurements make request control important:

- 2025 AIA ratio frames are roughly 144–172 ms cold, with PNG encoding dominant; EOVSA frames are roughly 200–550 ms cold, dominated by full compressed FITS-cube decompression.
- Cold EOVSA contour overlays are roughly 0.43–0.55 s; warm overlay cache hits are sub-millisecond.
- Current slider changes issue immediate image requests and a 40 ms prefetch of four neighboring frames; backend frame/overlay caches are plain dictionaries and concurrent frame/contour requests can duplicate FITS decompression.

The layer scheduler should therefore:

- throttle slider/playback commits to the latest cursor (or commit on pointer-up),
- deduplicate requests by `(session, source, resolvedIndex, operation, reference, display, layer kind)`,
- suppress requests when a layer resolves to the same native index as the previous cursor,
- defer expensive contour layers while dragging and render them for the settled cursor,
- share in-flight backend reads for a given EOVSA FITS file, and
- bound prefetch by visible layers and playback direction.

Compositing multiple layers in one backend response is an optional later optimization; it should not be required for the first schema/UI increment because it complicates independent cacheability and error reporting.

## Phased acceptance criteria

### Phase 1 — timeline contract

- A timeline control can choose AIA or EOVSA as master.
- Radio master playback visits every native radio timestamp; AIA displays the nearest sample with a visible offset.
- Nearest/previous/next resolution, tie-breaking, out-of-range handling and native-operation references have unit tests.
- Legacy version-1 JSON still opens with AIA as master and produces the same cursor/frame choices.

### Phase 2 — layer and panel composition

- Each image-panel header has an independent base-layer selector and overlay menu.
- The left/right example above can be created without changing source roles globally.
- Layer order, visibility, opacity, operation/reference, display controls and radio frequency survive export/reload.
- Contours can target either panel's WCS-aligned base layer; existing ROI/tracks remain aligned.

### Phase 3 — colormap contract

- Parula, Inferno and RdYlBu appear in scalar display controls and render through the backend.
- Frequency-coded contours expose the same maps with low-frequency warm/high-frequency cool ordering.
- Endpoint/direction tests cover the palette registry and colorbar labels; scalar intensity ordering remains explicit.

### Phase 4 — workload verification

- Slider/playback request counts are bounded and stale requests do not replace settled content.
- Repeated resolved indices do not issue duplicate frame requests.
- A representative 2025 radio-master playback does not regress median settled-frame latency beyond the current cold-cache baseline; contour rendering remains deferred or cached.

## Open decisions

1. Should `Master native` default to Context for all sessions (maximum compatibility) or automatically choose the source with the densest cadence (better radio exploration)?
2. What default maximum time offset is scientifically acceptable for each source, and should an out-of-range layer be blank, held at the last frame, or hidden?
3. Should layer operations such as `previous` be defined by native sample order or by elapsed-time lag with interpolation when a sample is missing?
4. Should layer settings be immutable definitions referenced by panels, or should each panel own a copied instance by default to make independent edits unsurprising?
5. Which exact Parula LUT (and license/provenance) should be the canonical shared frontend/backend definition?
6. Is the warm-to-cool radio convention limited to frequency-coded contours/colorbars, or should a radio intensity image with frequency encoding receive the same treatment?
