import { type Dispatch, type SetStateAction, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Anchor,
  ArrowLeftRight,
  ArrowDown,
  ArrowUp,
  ArrowDownToLine,
  Copy,
  ChevronLeft,
  ChevronRight,
  ChevronFirst,
  ChevronLast,
  Crosshair,
  Database,
  Droplet,
  Download,
  Eye,
  EyeOff,
  Flame,
  FolderOpen,
  LassoSelect,
  Link2,
  Palette,
  Pause,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Redo2,
  Settings2,
  Sparkles,
  Square,
  StepForward,
  Sun,
  Trash2,
  Undo2,
  Video,
  Upload
} from "lucide-react";
import {
  OVERLAY_REQUEST_DELAY_MS,
  PREFETCH_IDLE_MS,
  PREFETCH_RADIUS,
  advanceFrameTime,
  areFramesCached,
  clearFrameCache,
  frameCacheStateVersion,
  frameTimesForTimeRange,
  frameRequestState,
  frameRequestGroupKey,
  frameResolution,
  frameStats as frameStatsForKey,
  getFrameCacheBudgetBytes,
  loadFrame,
  nearestCachedFrame,
  planFrameCacheWarm,
  prefetchFrames,
  predictDisplayResolution,
  requestKey,
  setFrameCacheBudgetBytes,
  setPinnedFrameRequests,
  snapFrameTime,
  timeStepGridPosition,
  waitForFrameDelay,
  warmFrameCache,
  type ContourGeometry,
  type FrameData,
  type FrameResolution,
  type FrameStats,
  type ScheduledFrameRequest
} from "./frameScheduler";
import {
  COLORMAP_OPTIONS,
  INSTRUMENT_COLORMAP_OPTIONS,
  colormapGradient,
  colormapStops,
  normalizeColormap,
  sampleColormap,
  type ColormapId
} from "./radioColormaps";
import { predictResolution } from "./timeResolution";

type PanelId = "aia" | "eovsa";
type PanelSlotId = "left" | "right";
type SourceRole = "context" | "radio" | "spectrogram" | string;
type ScaleMode = "linear" | "log" | "sqrt" | "asinh";
type FrequencyScaleMode = "linear" | "log";
type SpectrogramNormalization = "none" | "divide" | "subtract";
type DifferenceMode = "none" | "running" | "base";
type DifferenceOperation = "none" | "subtract" | "ratio";
type DifferenceReference = "previous" | "base" | "mean";
type TemporalMode = "none" | "lowpass" | "bandpass";
type ContourColormap = ColormapId;
type Affine = number[][];
type ChannelOffsets = { dx: number[]; dy: number[]; masked: boolean[] };
type AlignmentXAxisMode = "channel" | "frequency";
type SpwGroup = { id: number; start: number; end: number };

type LayerKind = "image" | "contours" | "spectrogram";
type PanelComposition = { baseLayerId: string; overlayLayerIds: string[] };
type SelectedLayer = { slot: PanelSlotId; id: string };
type FrameRequestCap = { maxWidth: number; maxHeight: number } | null;
type PlaybackBufferState = { ahead: number; total: number };
type RecordingSource = "left" | "right" | "both" | "workspace";
type RecordingRange = "visible" | "master";
// "native" composites at the source panels' actual backing-store resolution (devicePixelRatio-scaled,
// i.e. exactly what's on screen) - this used to be called "onscreen" even though the math was already
// DPR-aware; the rename makes the option honest about what it does. "2x" doubles that (subject to the
// RECORDER_HARD_MAX_DIMENSION clamp and encoder negotiation below).
type RecordingResolution = "native" | "2x";
// "auto" walks the existing negotiation chain (pickRecorderConfig's guess, then legacy avc1, then vp9/vp8/webm).
// "mp4"/"webm" pin the negotiation to just that family - see startRecording's attemptChain construction.
type RecordingFormatChoice = "auto" | "mp4" | "webm";
type RecordingOptions = {
  source: RecordingSource;
  range: RecordingRange;
  fps: number;
  resolution: RecordingResolution;
  burnTimestamp: boolean;
  formatChoice: RecordingFormatChoice;
};
type RecordingStatus = {
  phase: "preparing" | "recording";
  done: number;
  total: number;
  fps: number;
  startedAt: number;
};
type CaptureSurface = "spectrogram" | PanelSlotId;
type RecordingSurface = CaptureSurface | "spectrogram-gutter";
type RecordingFormat = { label: "MP4 (H.264)" | "WebM"; mimeType: string; extension: "mp4" | "webm" };
type FileSystemEntry = { name: string; path: string; isDir: boolean; size: number; mtime: number };
type FileSystemListing = { path: string; parent: string; entries: FileSystemEntry[] };

type LayerState = {
  id: string;
  mirrorOf?: string;
  label: string;
  labelEdited: boolean;
  sourceId: string;
  /** Immutable role/capability snapshot captured when the layer is created. */
  sourceRoleSnapshot: SourceRole;
  kind: LayerKind;
  visible: boolean;
  opacity: number;
  operation: DifferenceOperation;
  reference: DifferenceReference;
  cadenceSeconds: string;
  meanStartMjd?: number;
  meanEndMjd?: number;
  temporal: TemporalState;
  samplingPolicy: "nearest" | "previous" | "next";
  maxOffsetSeconds?: number;
  display: DisplayState;
  freqIndex: number;
  contourLevelPercent: string;
  contourLevelKelvin: string;
  contourLevelSfu: string;
  contourLevelMode: "percent" | "kelvin" | "sfu";
  contourLevelReference: "current" | "global";
  contourFilled: boolean;
  contourOpacity: string;
  contourCmap: ContourColormap;
};

type RenderLayer = {
  layer: LayerState;
  request: ScheduledFrameRequest;
  shape: [number, number];
  pixelToWorldAffine: Affine;
  worldOffset: [number, number];
  colorbar?: RadioColorbar;
};

function layerOrigin(layer: LayerState, allLayers: LayerState[]): LayerState {
  let origin = layer;
  const visited = new Set<string>([layer.id]);
  while (origin.mirrorOf && !visited.has(origin.mirrorOf)) {
    const next = allLayers.find((candidate) => candidate.id === origin.mirrorOf);
    if (!next) break;
    visited.add(next.id);
    origin = next;
  }
  return origin;
}

function resolvedLayer(layer: LayerState, allLayers: LayerState[]): LayerState {
  const origin = layerOrigin(layer, allLayers);
  return {
    ...origin,
    id: layer.id,
    mirrorOf: layer.mirrorOf,
    display: { ...origin.display },
    temporal: { ...origin.temporal }
  };
}

function independentLayer(layer: LayerState, id = layer.id, mirrorOf?: string): LayerState {
  return {
    ...layer,
    id,
    mirrorOf,
    display: { ...layer.display },
    temporal: { ...layer.temporal }
  };
}

type SolarView = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

type LayoutState = {
  railWidth: number;
  spectrogramHeight: number;
  slitLaneHeight: number;
  imageSplit: number;
};

type FloatingCardKey = "trackEditor" | "correlation" | "slitInspector" | "channelInspector" | "spectrogramDisplay" | "recordingOptions" | "pixelProbe" | "filePicker";
type FloatingCardSize = { width: number; height: number };
type FloatingCardSizes = Partial<Record<FloatingCardKey, FloatingCardSize>>;

type SessionMeta = {
  sessionId: string;
  prewarm?: PrewarmStatus;
  sources?: SourceMeta[];
  radioPeakCache?: Record<string, number[]>;
  radioPeakTableCache?: Record<string, number[][]>;
  aia: {
    shape: [number, number];
    times: string[];
    timeMjd: number[];
    cornersArcsec: { xMin: number; xMax: number; yMin: number; yMax: number };
  };
  eovsa: {
    shape: [number, number];
    times: string[];
    timeMjd: number[];
    freqGhz: number[];
    channelOffsets?: ChannelOffsets;
    channelMask?: boolean[];
  };
  wcs: {
    aia: { pixelToWorldAffine: Affine };
    eovsa: { pixelToWorldAffine: Affine };
  };
  spectrogram: {
    shape: [number, number];
    timeMjd: number[];
    freqGhz: number[];
    defaults: {
      vmin: number;
      vmax: number;
      cmap: string;
      scale: ScaleMode;
    };
  };
  defaults: {
    timeStartIndex: number;
    timeEndIndex: number;
    timeIndex: number;
    freqIndex: number;
    xOffsetArcsec: number;
    yOffsetArcsec: number;
    diffSeconds: number;
  };
  paths: Record<string, string>;
};

type PrewarmStatus = {
  done: number;
  total: number;
  active: boolean;
};

type ProgressOperation = {
  opId: string;
  label: string;
  done: number;
  total: number | null;
  startedAt: number;
  ended?: boolean;
  endedAt?: number | null;
};

type ProgressResponse = {
  operations: ProgressOperation[];
  prewarm: PrewarmStatus;
};

function isActiveProgressOperation(operation: ProgressOperation): boolean {
  const hasEnded = operation.ended === true
    || (operation.endedAt !== undefined && operation.endedAt !== null);
  return !hasEnded && (operation.total === null || operation.done < operation.total);
}

type WarmCacheStatus = {
  active: boolean;
  done: number;
  total: number;
  retained: number;
  requestedFrames: number;
  targetFrames: number;
  limited: boolean;
  startedAt: number;
};

type WarmCacheRecord = {
  sessionId: string;
  windowSignature: string;
  settingsSignature: string;
};

type SourceMeta = {
  id: string;
  role: SourceRole;
  label: string;
  instrument?: string;
  format?: string;
  shape?: number[];
  time?: {
    count?: number;
    times?: string[];
    timeMjd?: number[];
    start?: string;
    end?: string;
  };
  freqGhz?: number[];
  channelOffsets?: ChannelOffsets;
  channelMask?: boolean[];
  pol?: string[];
  capabilities?: Record<string, boolean>;
  status?: string;
  paths?: Record<string, string>;
};

type DisplayState = {
  vmin: string;
  vmax: string;
  cmap: string;
  scale: ScaleMode;
  radialGamma: number;
};

type TemporalState = {
  mode: TemporalMode;
  sigmaShort: string;
  sigmaLong: string;
};

type FrequencyRangeState = {
  min: string;
  max: string;
};

type TimeRangeState = {
  min: string;
  max: string;
};

type RadioColorbar = {
  minGhz: number;
  maxGhz: number;
  cmap: ContourColormap;
};

type DifferenceState = {
  operation: DifferenceOperation;
  reference: DifferenceReference;
  cadenceSeconds: string;
  meanStartIndex: number;
  meanEndIndex: number;
  meanStartMjd?: number;
  meanEndMjd?: number;
};

type SavedUiState = {
  panels?: Record<PanelSlotId, { layerIds?: string[]; baseLayerId?: string; overlayLayerIds?: string[] }>;
  /** v2 canonical record; v1 array remains accepted as a migration input. */
  layers?: Record<string, LayerState | Record<string, unknown>> | Array<LayerState | Record<string, unknown>>;
  timeline?: {
    masterSourceId?: string;
    cursorMjd?: number;
    startMjd?: number;
    endMjd?: number;
  };
  selectedSourceId?: string;
  sourceRoles?: Record<string, SourceRole>;
  sourceDifferences?: Record<string, DifferenceState>;
  timeIndex?: number;
  startIndex?: number;
  endIndex?: number;
  freqIndex?: number;
  diffSeconds?: string | number;
  useRunningDiff?: boolean | string | number;
  showEovsaContours?: boolean | string | number;
  contourFilled?: boolean | string | number;
  contourLevelPercent?: string | number;
  contourLevelKelvin?: string | number;
  contourLevelSfu?: string | number;
  contourLevelMode?: string;
  contourLevelReference?: string;
  contourOpacity?: string | number;
  contourCmap?: string;
  xOffsetArcsec?: string | number;
  yOffsetArcsec?: string | number;
  view?: ViewState;
  solarView?: SolarView;
  layout?: LayoutState;
  aiaDisplay?: DisplayState;
  eovsaDisplay?: DisplayState;
  spectrogramDisplay?: DisplayState;
  spectrogramNormalization?: SpectrogramNormalization;
  spectrogramFrequencyScale?: FrequencyScaleMode;
  spectrogramFrequencyInverted?: boolean;
  spectrogramFrequencyRange?: FrequencyRangeState;
  spectrogramTimeRange?: TimeRangeState;
  playbackFps?: number;
  timeStepSeconds?: number;
  smoothPlayback?: boolean | string | number;
  showCacheCoverage?: boolean | string | number;
  frameCacheGb?: number | string;
  collapsedSections?: string[];
  trackEditorOpen?: boolean;
  trackingSourceId?: string;
  selectedTrackId?: string;
  trackingDirection?: TrackDirection;
  correlationCardOpen?: boolean;
  correlationFullRange?: boolean;
  correlationPinTicks?: boolean;
  correlationFullHeightTicks?: boolean;
  correlationPeakDecel?: boolean;
  slitInspectorOpen?: boolean;
  selectedSlitId?: string;
  slitSourceId?: string;
  slitPinLane?: boolean;
  slitExtractAll?: boolean;
  slitSmoothPx?: number;
  selectedFanMember?: number;
  floatingCardSizes?: FloatingCardSizes;
  /** Loosely typed: values come from arbitrary persisted JSON and are migrated on load (see applyLoadedSession). */
  recordingOptions?: {
    source?: string;
    range?: string;
    fps?: number | string;
    resolution?: string;
    burnTimestamp?: boolean | string | number;
    formatChoice?: string;
  };
};

type TrackDirection = "forward" | "backward" | "both";

type TrackAnchor = {
  frameIndex: number;
  mjd: number;
  x: number;
  y: number;
};

type TrackPoint = TrackAnchor & {
  confidence: number;
  isAnchor: boolean;
};

type SadTrack = {
  id: string;
  label: string;
  sourceId: string;
  color: string;
  visible: boolean;
  anchors: TrackAnchor[];
  points: TrackPoint[];
  state: "active" | "stopped-low-confidence" | "stopped-edge";
};

type TrackingSourceOption = {
  sourceId: string;
  label: string;
  sides: PanelSlotId[];
  layer: LayerState;
};

type SlitBindingKind = "image" | "contours" | "raw";
type SlitSourceOption = TrackingSourceOption & { bindingKind: SlitBindingKind };

type SlitDisplayState = {
  vmin: number;
  vmax: number;
  cmap: ColormapId;
  scale: "linear" | "sqrt" | "log";
};

type SlitDefinition = {
  id: string;
  /** Id of another slit whose geometry this one follows (see resolveSlitGeometry). */
  linkedTo?: string;
  name: string;
  color: string;
  visible: boolean;
  sourceId: string;
  layerId: string;
  bindingKind: SlitBindingKind;
  // FIXED PHYSICAL slit width, in arcsec (USER DESIGN DECISION - not
  // bound-source pixels): different sources have different pixel scales, so
  // a pixel width is meaningless across layers, while an arcsec width lets a
  // slit and its linked twin (resolveSlitGeometry - same spine, different
  // binding) sample the SAME sky corridor. Converted to each bound source's
  // own pixel width only at request-build time (see extractSlit/
  // extractAllSlits/buildFanMemberSlit's callers) - the backend's extraction
  // contract is unchanged and still receives a pixel width. A legacy session
  // may carry an old pixel `width` field instead; normalizeSlits migrates it
  // to widthArcsec on load and never writes `width` back out.
  widthArcsec: number;
  shiftSeconds: number;
  curveArcsec: [number, number][];
  inputVertexCount: number;
  // PER-SLIT RE-SMOOTHING: the jitter-trimmed hand-drawn stroke, captured
  // BEFORE gaussian smoothing, projected to arcsec with the same per-point
  // affine used to produce curveArcsec (see trimAndProjectStroke) - i.e. the
  // input to smoothAndResampleCurve rather than its output. Paired with
  // smoothPx and drawnPanel so resmoothCurveFromRaw can regenerate
  // curveArcsec from scratch at a new smoothing sigma. Optional/undefined
  // for slits created before this feature (session round-trip) or derived
  // from a fan family (buildFanMemberSlit never sets it, since those curves
  // are computed from the fan's own boundaries, not drawn) - the "Smooth"
  // control is disabled for those (see SlitInspectorCard).
  rawCurveArcsec?: [number, number][];
  // Gaussian smoothing sigma, in "pixels" of whichever panel (drawnPanel)
  // captured rawCurveArcsec - the same unit as the global draw-time Smooth
  // control. Re-expressed in arcsec at re-smooth time via
  // affinePixelScaleArcsec(that panel's live affine) rather than frozen at
  // creation - see resmoothCurveFromRaw.
  smoothPx?: number;
  /** Which panel's own pixel scale smoothPx is expressed in - see rawCurveArcsec/resmoothCurveFromRaw. */
  drawnPanel?: PanelId;
  display: SlitDisplayState;
  freqIndices: number[];
  baseFreqIndex: number | null;
  contourFreqIndices: number[];
  contourLevelPercent: number;
};

// A "linked twin" slit (linkedTo set) borrows its curve/width geometry live
// from another slit instead of storing its own - the mirrorOf pattern used
// for LayerState (see layerOrigin/resolvedLayer above), applied to slits.
// slitOriginId walks the link chain (defensively - normal use never creates
// chains longer than one hop, since createLinkedTwin always links to the
// ultimate root) to the unlinked origin; resolveSlitGeometry returns that
// origin's curveArcsec/inputVertexCount/width. A twin's own stored copies of
// those three fields are kept as a best-effort snapshot only (written at
// creation time, and re-stamped whenever the original is deleted - see
// deleteSlit/deleteFan's materialization, and normalizeSlits' orphan
// handling on session load) so the type can stay non-optional; every
// render/extract/export call site must go through this helper rather than
// read slit.curveArcsec/inputVertexCount/width directly, or it will see a
// stale snapshot instead of the original's live geometry.
function slitOriginId(slit: SlitDefinition, allSlits: SlitDefinition[]): string {
  let current = slit;
  const visited = new Set<string>([slit.id]);
  while (current.linkedTo && !visited.has(current.linkedTo)) {
    const next = allSlits.find((candidate) => candidate.id === current.linkedTo);
    if (!next) break;
    visited.add(next.id);
    current = next;
  }
  return current.id;
}

function resolveSlitGeometry(
  slit: SlitDefinition,
  allSlits: SlitDefinition[]
): { curveArcsec: [number, number][]; inputVertexCount: number; widthArcsec: number } {
  if (!slit.linkedTo) return { curveArcsec: slit.curveArcsec, inputVertexCount: slit.inputVertexCount, widthArcsec: slit.widthArcsec };
  const origin = allSlits.find((candidate) => candidate.id === slitOriginId(slit, allSlits)) ?? slit;
  // A twin's widthArcsec is inherited from the origin verbatim (it is a
  // PHYSICAL corridor width, not a pixel count) - so origin and twin always
  // sample the identical sky corridor even though they convert it to
  // different per-source pixel widths at extraction time (see
  // extractSlit/extractAllSlits) and even though their drawn swaths (arcsec,
  // not per-source pixels - see drawSlitWorldPath) are now literally
  // identical, which the panel draw loop exploits to avoid double-painting
  // the overlapping fill (see ImagePanel's slit draw loop).
  return { curveArcsec: origin.curveArcsec, inputVertexCount: origin.inputVertexCount, widthArcsec: origin.widthArcsec };
}

// Average pixel scale (arcsec/px) implied by a source's pixel->world affine.
// affine[0] is the world-space image of a unit step along the pixel-x axis,
// affine[1] the image of a unit step along pixel-y (see applyAffine); their
// norms are each source's per-axis plate scale (AIA ~0.6"/px, EOVSA ~2"/px).
// The two are averaged rather than kept separate since slit curves are drawn
// and resampled as scalar arc length, not per-axis.
function affinePixelScaleArcsec(affine: Affine): number {
  const scaleX = Math.hypot(affine[0][0], affine[0][1]);
  const scaleY = Math.hypot(affine[1][0], affine[1][1]);
  return (scaleX + scaleY) / 2;
}

// Walks a polyline's cumulative arc length to linearly interpolate the point
// at `targetDistance` from the start. Helper for resampleCurveForSource only.
function sampleAlongCurve(
  curveArcsec: [number, number][],
  segmentLengths: number[],
  targetDistance: number
): [number, number] {
  let traveled = 0;
  for (let i = 0; i < segmentLengths.length; i++) {
    const length = segmentLengths[i];
    if (traveled + length >= targetDistance || i === segmentLengths.length - 1) {
      const t = length > 0 ? Math.min(1, Math.max(0, (targetDistance - traveled) / length)) : 0;
      const [x0, y0] = curveArcsec[i];
      const [x1, y1] = curveArcsec[i + 1];
      return [x0 + (x1 - x0) * t, y0 + (y1 - y0) * t];
    }
    traveled += length;
  }
  return curveArcsec[curveArcsec.length - 1];
}

// Resamples a stored slit curve (an arcsec polyline, dense at ~1px of
// whichever panel it was drawn on - see completeSlitDraw) down/up to uniform
// arc-length spacing at `sourcePixelScaleArcsec`, the native pixel scale of
// the source it is about to be EXTRACTED against. This is extraction-only:
// panel drawing keeps rendering the original dense stored curve untouched
// (see resolveSlitGeometry call sites used for drawing) for visual
// smoothness. Without this, a radio-bound (EOVSA, ~2"/px) slit or twin
// samples the ~0.6"/px-dense AIA-drawn curve ~3.3x too densely (wasted
// compute, correlated samples, a distance axis implying false resolution),
// while a curve drawn on a radio panel but extracted against AIA would be
// undersampled (aliasing). Endpoints are always preserved exactly.
function resampleCurveForSource(
  curveArcsec: [number, number][],
  sourcePixelScaleArcsec: number
): [number, number][] {
  if (curveArcsec.length < 2) return curveArcsec;
  const step = Number.isFinite(sourcePixelScaleArcsec) && sourcePixelScaleArcsec > 0 ? sourcePixelScaleArcsec : 1;
  const segmentLengths: number[] = [];
  let totalLength = 0;
  for (let i = 1; i < curveArcsec.length; i++) {
    const length = Math.hypot(curveArcsec[i][0] - curveArcsec[i - 1][0], curveArcsec[i][1] - curveArcsec[i - 1][1]);
    segmentLengths.push(length);
    totalLength += length;
  }
  const first = curveArcsec[0];
  const last = curveArcsec[curveArcsec.length - 1];
  if (totalLength <= 0) return [first, last];
  // Shorter than 2 source pixels: essentially no resolvable structure at
  // this source's scale. Keep endpoints plus a midpoint instead of forcing
  // the minimum-sample-count floor below onto a near-point curve.
  if (totalLength < 2 * step) {
    return [first, sampleAlongCurve(curveArcsec, segmentLengths, totalLength / 2), last];
  }
  const MIN_SAMPLES = 8;
  const sampleCount = Math.max(MIN_SAMPLES, Math.round(totalLength / step) + 1);
  const resampled: [number, number][] = [first];
  for (let i = 1; i < sampleCount - 1; i++) {
    resampled.push(sampleAlongCurve(curveArcsec, segmentLengths, (totalLength * i) / (sampleCount - 1)));
  }
  resampled.push(last);
  return resampled;
}

type SlitMapResult = {
  npix: number;
  ntime: number;
  intensity: (number | null)[][];
  distanceArcsec: number[];
  timeMjd: number[];
  curveVerticesArcsec: [number, number][];
  dataMin: number;
  dataMax: number;
  dataP1: number;
  dataP99: number;
  cacheHit: boolean;
  wallSeconds: number;
  mapCacheHit?: boolean;
  mapWallSeconds?: number;
  cacheKey: string;
  freqIndex: number | null;
  freqGhz: number | null;
  // Server-resolved contour trigger level for this channel, matching the
  // bound contour layer's level settings at extraction time (see
  // slitLayerParams). Null when the map isn't radio, the slit isn't
  // contour-bound, or the threshold couldn't be resolved (e.g. no global
  // peak table yet); the lane then falls back to a local estimate and the
  // level-scrub path refreshes this live without re-extracting.
  contourThreshold?: number | null;
};

type SlitResult = SlitMapResult & {
  slitId: string;
  sourceId: string;
  width: number;
  additionalMaps: SlitMapResult[];
};

type FanDefinition = {
  id: string;
  sourceId: string;
  layerId: string;
  bindingKind: SlitBindingKind;
  boundaryA: [number, number][];
  boundaryB: [number, number][];
  intermediateCount: number;
  inputVertexCounts: [number, number];
  promotedMembers: Record<string, string>;
  // PER-FAN RE-SMOOTHING: mirrors SlitDefinition.rawCurveArcsec/smoothPx/
  // drawnPanel, one raw stroke per boundary (they can be drawn on different
  // panels - see armFanRedraw). Re-smoothing regenerates both boundaries
  // from their raw strokes at the fan's single smoothPx, then re-derives the
  // whole family (fanFamilyCurves) and updates any promoted members' curves,
  // the same way completeFanRedraw does for a full boundary redraw. Optional
  // for fans created before this feature.
  rawBoundaryA?: [number, number][];
  rawBoundaryB?: [number, number][];
  boundaryADrawnPanel?: PanelId;
  boundaryBDrawnPanel?: PanelId;
  smoothPx?: number;
};

type SeedSuggestion = { x: number; y: number; value: number };

type EovsaSource = {
  time_mjd: number;
  spw_index: number;
  freq_ghz: number;
  x_centroid_display_pix: number;
  y_centroid_display_pix: number;
  x_peak_display_pix: number;
  y_peak_display_pix: number;
  x_centroid_arcsec: number;
  y_centroid_arcsec: number;
  x_peak_arcsec?: number;
  y_peak_arcsec?: number;
  snr: number;
};

type PanelTransform = {
  scale: number;
  offsetX: number;
  offsetY: number;
  plotWidth: number;
  plotHeight: number;
  imageWidth: number;
  imageHeight: number;
  solarView: SolarView;
  pixelToWorldAffine: Affine;
  worldOffset: [number, number];
};

type ViewState = {
  zoom: number;
  center: [number, number];
};

type PanState = {
  pointerId: number;
  startX: number;
  startY: number;
  solarView: SolarView;
  startPixel: [number, number] | null;
  moved: boolean;
};

type AlignmentDragState = {
  pointerId: number;
  startWorld: [number, number];
  totalWorld: [number, number];
};

type TrackMarkerDragState = {
  pointerId: number;
  trackId: string;
  frameIndex: number;
  startPixel: [number, number];
  currentPixel: [number, number];
  moved: boolean;
};

const TRACK_MARKER_HIT_RADIUS = 14;

type TrackingClickState = {
  pointerId: number;
  startPixel: [number, number];
  suggestion?: SeedSuggestion;
  trackId?: string;
  moved: boolean;
};

type PixelProbe = {
  slot: PanelSlotId;
  panel: PanelId;
  layerId: string;
  pixel: [number, number];
  patchRadius: number;
};

type ProbeStats = {
  min: number | null;
  max: number | null;
  p1: number | null;
  p99: number | null;
};

type PixelProbeResponse = {
  mjd: number[];
  raw: (number | null)[];
  smoothed?: (number | null)[];
  stats: { raw: ProbeStats; smoothed?: ProbeStats };
  cadenceSeconds: number;
  nTotal: number;
  stride: number;
};

type LayoutDragState = {
  kind: "rail" | "spectrogram" | "slitLane" | "image";
  startX: number;
  startY: number;
  layout: LayoutState;
};

type RoiProjectionValues = {
  xOffset?: string;
  yOffset?: string;
  diffSeconds?: string;
};

type LoadedState = {
  ui?: SavedUiState;
  channelOffsets?: ChannelOffsets;
  roiWorld?: [number, number][];
  roiAia?: [number, number][];
  roiEovsa?: [number, number][];
  sadTracks?: SadTrack[];
  eovsaSources?: EovsaSource[];
  featureTracks?: SadTrack[];
  tracks?: SadTrack[];
  radioSources?: EovsaSource[];
  correlationTarget?: [number, number][];
  slits?: SlitDefinition[];
  fan?: FanDefinition;
  savedSources?: SourceMeta[];
};

type CorrelationTick = {
  trackId: string;
  label: string;
  color: string;
  arrivalMjd: number;
  peakDecelMjd?: number;
  bracket?: [number, number];
};

type CorrelationSeries = {
  track: SadTrack;
  points: { mjd: number; distance: number }[];
  arrival?: CorrelationTick;
};

type LoadSessionResponse = SessionMeta & {
  loadedState?: LoadedState;
};

const API = "";
const INITIAL_MANIFEST_JSON = import.meta.env.VITE_INITIAL_MANIFEST_JSON as string | undefined;
const INITIAL_MANIFEST_NAME = import.meta.env.VITE_INITIAL_MANIFEST_NAME as string | undefined;
const IMAGE_CACHE_LIMIT = 96;
const DEFAULT_VIEW: ViewState = { zoom: 1, center: [0.5, 0.5] };
const DEFAULT_SOLAR_VIEW: SolarView = { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
const DEFAULT_LAYOUT: LayoutState = { railWidth: 330, spectrogramHeight: 150, slitLaneHeight: 190, imageSplit: 0.5 };
const DEFAULT_DIFF_SECONDS = 60;
const DEFAULT_TEMPORAL_SIGMA_SHORT = 12;
const DEFAULT_TEMPORAL_SIGMA_LONG = 120;
const DEFAULT_PLAYBACK_FPS = 5;
// Bounds for the "Frame cache [GB]" scrubber (SpectrogramDisplayCard); mirrors
// frameScheduler's DEFAULT_FRAME_CACHE_BYTE_BUDGET of 2 GB as the fallback.
const FRAME_CACHE_GB_MIN = 0.25;
const FRAME_CACHE_GB_MAX = 8;
// The backend keeps its legacy half-cadence default when this parameter is
// omitted; this finite transport sentinel disables that legacy tolerance.
const UNBOUNDED_DISPLAY_OFFSET_SECONDS = Number.MAX_VALUE;
const DEFAULT_RECORDING_OPTIONS: RecordingOptions = { source: "both", range: "visible", fps: 20, resolution: "native", burnTimestamp: false, formatChoice: "auto" };
const TRACK_COLORS = ["#56c7d9", "#ee65be", "#ffcc66", "#8fd694", "#a99bff", "#ff7b72"];
const TRACK_HISTORY_LIMIT = 50;
const TRACK_TIMELINE_MIN_ZOOM_SPAN = 10;
const DEFAULT_SLIT_SMOOTH_PX = 6;
const SLIT_SMOOTH_MAX_PX = 100;
// Sanity ceiling for a legacy on-disk pixel `width` (normalizeSlits'
// migration input) - deliberately generous (matches the backend's own
// MAX_SLIT_WIDTH_PX in data.py); only guards against corrupt/adversarial
// session JSON before it is converted to arcsec.
const SLIT_WIDTH_HARD_CAP_PX = 4096;
// USER DESIGN DECISION: slit width is a fixed PHYSICAL quantity in arcsec
// (widthArcsec), not bound-source pixels - see SlitDefinition.widthArcsec.
// Default width for a freshly drawn slit, and the fallback when neither a
// legacy px width nor an arcsec width is present on disk.
const DEFAULT_SLIT_WIDTH_ARCSEC = 2;
// Sanity ceiling for widthArcsec values coming off disk (normalizeSlits) -
// deliberately generous (well above any realistic field of view) since this
// only guards against corrupt/adversarial session JSON; the real, tighter
// per-slit bound shown in the UI is slitWidthArcsecBounds' dynamic max (the
// largest loaded source's own field of view).
const SLIT_WIDTH_ARCSEC_HARD_CAP = 4096;
// Static fallback bounds for the Width [″] control (slitWidthArcsecBounds)
// while meta hasn't loaded real pixel scales/shapes yet - see FEATURE 2 /
// slitWidthArcsecBounds. ~0.3″ is half AIA's ~0.6″/px native scale; ~1000″
// is comfortably above AIA's ~4096px x 0.6″/px field of view.
const FALLBACK_SLIT_WIDTH_ARCSEC_MIN = 0.3;
const FALLBACK_SLIT_WIDTH_ARCSEC_MAX = 1000;
const DEFAULT_SLIT_LANE_HEIGHT = DEFAULT_LAYOUT.slitLaneHeight;
// The lane's border leaves a roughly 90 px drawable canvas at this clamp.
const MIN_SLIT_LANE_HEIGHT = 92;
const MIN_PANEL_GRID_HEIGHT = 200;
const PLAYBACK_FPS_OPTIONS = [0.5, 1, 2, 5, 10, 20] as const;
const PLAYBACK_PINNED_FRAME_COUNT = 3;
const PLAYBACK_LOOKAHEAD_MAX_FRAMES = 24;
const FRAME_CAP_QUANTUM = 256;
const FRAME_CAP_ZOOM_IN = 1.05;
const FRAME_CAP_ZOOM_OUT = 1.0;
const DEFAULT_FRAME_REQUEST_CAP = { maxWidth: 1024, maxHeight: 1024 };
// Grace period after playback pauses or a scrub ends before the motion
// resolution ladder settles back to full-res identities. Keeps a quick
// play/pause tap or a jittery scrub release from firing a full-res fetch
// only to abandon it a moment later.
const MOTION_RESOLUTION_IDLE_MS = 300;
// Cache-coverage strip: 3px-wide columns along the top edge of the
// spectrogram plot, refreshed no faster than 1 Hz (frameCacheStateVersion
// poll) plus immediately on identity/time-range changes.
const CACHE_COVERAGE_COLUMN_PX = 3;
const CACHE_COVERAGE_BAR_HEIGHT_PX = 3;
const CACHE_COVERAGE_POLL_MS = 1000;
const SCRUB_PREFETCH_RADIUS = 6;
const IDENTITY_AFFINE: Affine = [[1, 0], [0, 1], [0, 0]];
const CONTOUR_COLORMAPS: readonly { id: ContourColormap; label: string }[] = COLORMAP_OPTIONS;
const imageCache = new Map<string, HTMLImageElement>();
let initialManifestRequested = false;

function positiveTimeStepSeconds(value: unknown): number {
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric > 0 ? numeric : 0;
}

function sensibleCadenceSeconds(value: number): number {
  if (!(value > 0) || !Number.isFinite(value)) return 0;
  const digits = value < 1 ? 3 : value < 10 ? 2 : 1;
  return Number(value.toFixed(digits));
}

function clampedTimeStepSeconds(value: unknown, minimumCadence: number): number {
  const requested = positiveTimeStepSeconds(value);
  if (!requested) return 0;
  return Math.max(requested, sensibleCadenceSeconds(minimumCadence));
}

function browserRecordingFormat(): RecordingFormat {
  const supported = (mimeType: string) => typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(mimeType);
  const mp4MimeType = [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4;codecs=avc1.4D401E",
    "video/mp4;codecs=h264"
  ].find(supported);
  if (mp4MimeType) return { label: "MP4 (H.264)", mimeType: mp4MimeType, extension: "mp4" };
  const webmMimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(supported) ?? "";
  return { label: "WebM", mimeType: webmMimeType, extension: "webm" };
}

// Conservative ceiling shared by canvas/GPU texture limits and current browsers' software H.264 encoders.
// Chrome, Firefox and Safari all guarantee at least an 8192px canvas dimension; above that, canvas
// rendering itself becomes unreliable well before any codec limit is reached, so this is enforced
// independently of which codec is ultimately negotiated.
const RECORDER_HARD_MAX_DIMENSION = 8192;

type RecorderLevelTier = { mimeType: string; maxWidth: number; maxHeight: number; maxMacroblocks: number };

// H.264 level dimension/macroblock ceilings, per ITU-T H.264 (Rec. ITU-T H.264, latest ed.) Table A-1
// "Level limits" (MaxFS is in 16x16 macroblocks). "avc1.640034" is High profile, level_idc 0x34 = 52
// (Level 5.2, MaxFS 36864). The existing "avc1.42E01E" string this app already used is Baseline profile,
// level_idc 0x1E = 30 (Level 3.0, MaxFS 1620) - kept as the second tier so small captures negotiate
// exactly as before. Width is additionally capped at 4096px on every tier: this mirrors the practical
// ceiling widely reported for Chromium's avc1 MediaRecorder path (both the OpenH264 software encoder and
// most hardware encoders top out there), which is tighter than the raw macroblock math would otherwise
// allow for very wide/short frames like a workspace composite.
const RECORDER_H264_TIERS: RecorderLevelTier[] = [
  { mimeType: "video/mp4;codecs=avc1.640034", maxWidth: 4096, maxHeight: 2304, maxMacroblocks: 36864 }, // High@5.2
  { mimeType: "video/mp4;codecs=avc1.42E01E", maxWidth: 720, maxHeight: 576, maxMacroblocks: 1620 } // Baseline@3.0 (legacy string)
];

function evenFloor(value: number): number {
  return Math.max(2, Math.floor(value / 2) * 2);
}

/** Shared by pickRecorderConfig and the "explicit MP4" format-choice UI warning so both agree on what fits. */
function mp4TierFor(width: number, height: number): RecorderLevelTier | undefined {
  return RECORDER_H264_TIERS.find((candidate) => (
    width <= candidate.maxWidth && height <= candidate.maxHeight && Math.ceil(width / 16) * Math.ceil(height / 16) <= candidate.maxMacroblocks
  ));
}

/**
 * Pure negotiation of a MediaRecorder mimeType plus safe output dimensions for a given (unscaled) canvas
 * size. Deliberately touches neither MediaRecorder nor the DOM so it can be table-tested outside a
 * browser. This only picks the best a-priori guess - the caller (startRecording) still wraps the actual
 * `new MediaRecorder(...)` and `.start()` calls in try/catch and steps further down the chain (the other
 * legacy avc1 string, vp9, vp8, plain webm) because real browsers don't always honor these declared
 * limits exactly.
 */
function pickRecorderConfig(width: number, height: number): { mimeType: string; scaledWidth: number; scaledHeight: number; note: string } {
  const notes: string[] = [];
  const roundedWidth = Math.max(1, Math.round(width));
  const roundedHeight = Math.max(1, Math.round(height));
  let w = evenFloor(roundedWidth);
  let h = evenFloor(roundedHeight);
  if (w !== roundedWidth || h !== roundedHeight) notes.push("evened odd dimensions");
  if (w > RECORDER_HARD_MAX_DIMENSION || h > RECORDER_HARD_MAX_DIMENSION) {
    const scale = RECORDER_HARD_MAX_DIMENSION / Math.max(w, h);
    w = evenFloor(w * scale);
    h = evenFloor(h * scale);
    notes.push(`downscaled to fit the ${RECORDER_HARD_MAX_DIMENSION}px encoder/canvas limit`);
  }
  const tier = mp4TierFor(w, h);
  if (tier) return { mimeType: tier.mimeType, scaledWidth: w, scaledHeight: h, note: notes.join("; ") };
  notes.push("H.264 level limits exceeded at this resolution - using WebM (VP9)");
  return { mimeType: "video/webm;codecs=vp9", scaledWidth: w, scaledHeight: h, note: notes.join("; ") };
}

/**
 * Does an MP4/H.264 tier exist for this (already-rounded) composite size? Used by the "explicit MP4"
 * format choice to warn/disable Start before the user even hits the encoder - pickRecorderConfig only
 * discovers this at negotiation time, which is too late for a pre-flight UI warning.
 */
function mp4FitsAtSize(width: number, height: number): boolean {
  const w = evenFloor(Math.max(1, Math.round(width)));
  const h = evenFloor(Math.max(1, Math.round(height)));
  if (w > RECORDER_HARD_MAX_DIMENSION || h > RECORDER_HARD_MAX_DIMENSION) return false;
  return mp4TierFor(w, h) !== undefined;
}

// Explicit MediaRecorder bitrate. Left at its default, Chrome/Firefox target ~2.5 Mbps regardless of
// resolution or content - fine for talking-head webcam footage, but nowhere near enough for AIA
// difference imagery, where nearly every pixel changes frame-to-frame (no large flat regions for the
// encoder to skip) and the default heuristic crushes that entropy into blocky, "blurry-looking"
// macroblock artifacts. 0.12 bits/pixel/frame is a deliberately generous constant for this kind of
// high-motion, high-entropy synthetic content (typical well-encoded natural video sits around
// 0.05-0.1 bits/px/frame; noisy scientific difference frames warrant roughly 1.5-2x that). Clamped to
// [8, 80] Mbps: 8 Mbps is a usable floor even for small single-panel captures at low fps, 80 Mbps keeps
// large workspace/2x captures within what browser software encoders and MediaRecorder will reliably sustain.
const RECORDING_BITS_PER_PIXEL_PER_FRAME = 0.12;
const RECORDING_MIN_BITRATE = 8_000_000;
const RECORDING_MAX_BITRATE = 80_000_000;
function recordingBitrate(width: number, height: number, fps: number): number {
  return clamp(width * height * Math.max(1, fps) * RECORDING_BITS_PER_PIXEL_PER_FRAME, RECORDING_MIN_BITRATE, RECORDING_MAX_BITRATE);
}

function recordingDatasetToken(value: string): string {
  return value.replace(/\.[^.]+$/, "").replace(/[^a-z0-9._-]+/gi, "_").replace(/^_+|_+$/g, "") || "dataset";
}

function recordingTimeToken(mjd: number): string {
  const iso = new Date((mjd - 40587) * 86400000).toISOString();
  return iso.replace(/[-:]/g, "").replace(/\.\d{3}/, "");
}

function formatRecordingClock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}`;
}

function waitForRecordingTick(targetTime: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(new DOMException("Recording canceled.", "AbortError"));
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, Math.max(0, targetTime - performance.now()));
    const abort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("Recording canceled.", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
  });
}

function recordedVideoDuration(url: string): Promise<number | null> {
  return new Promise((resolve) => {
    const video = document.createElement("video");
    const finish = (duration: number | null) => {
      window.clearTimeout(timeout);
      video.onloadedmetadata = null;
      video.onerror = null;
      video.removeAttribute("src");
      video.load();
      resolve(duration);
    };
    const timeout = window.setTimeout(() => finish(null), 5000);
    video.preload = "metadata";
    video.onloadedmetadata = () => finish(Number.isFinite(video.duration) ? video.duration : null);
    video.onerror = () => finish(null);
    video.src = url;
  });
}

class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(detail: string, status: number) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

class SessionRestoredError extends Error {
  constructor() {
    super("Session restored after backend restart.");
    this.name = "SessionRestoredError";
  }
}

type SessionRecovery = () => Promise<unknown>;

async function apiJson<T>(url: string, options?: RequestInit, onSessionNotFound?: SessionRecovery): Promise<T> {
  const response = await fetch(`${API}${url}`, options);
  let detail = response.statusText;
  if (!response.ok) {
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : detail;
    } catch {
      // Keep status text.
    }
    if (response.status === 404 && /^Session not found\b/i.test(detail) && onSessionNotFound) {
      await onSessionNotFound();
      throw new SessionRestoredError();
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) {
    throw new Error("Requested sample unavailable (204).");
  }
  return response.json() as Promise<T>;
}

function numberValue(text: string, fallback: number): number {
  const value = Number(text);
  return Number.isFinite(value) ? value : fallback;
}

function formatDeltaSeconds(value: number | null | undefined, digits = 2): string {
  const delta = Number(value ?? 0);
  if (!Number.isFinite(delta)) return "Δ—";
  return `Δ${delta >= 0 ? "+" : ""}${delta.toFixed(digits)}s`;
}

function formatResolvedTimestamp(resolvedMjd: number | null | undefined, offsetSeconds: number | null | undefined, fallback: string): string {
  return Number.isFinite(resolvedMjd) ? `${mjdToUtc(resolvedMjd as number)} · ${formatDeltaSeconds(offsetSeconds)}` : fallback;
}

function statusProgress(message: string): { completed: number; total: number } | null {
  const match = message.match(/(?:^|\s)(\d+)\s*\/\s*(\d+)(?:\s|$)/);
  if (!match) return null;
  const completed = Number(match[1]);
  const total = Number(match[2]);
  return total > 0 && completed >= 0 && completed <= total ? { completed, total } : null;
}

function contourColormap(value: unknown): ContourColormap {
  return normalizeColormap(value, "turbo");
}

function contourColormapGradient(cmap: ContourColormap, direction = "90deg"): string {
  return colormapGradient(cmap, "frequency").replace("90deg", direction);
}

function frequencyBounds(freqGhz: number[]): [number, number] {
  const finite = freqGhz.filter(Number.isFinite);
  if (!finite.length) return [1, 18];
  return [Math.min(...finite), Math.max(...finite)];
}

function frequencyYMapping(min: number, max: number, scale: FrequencyScaleMode, inverted = false) {
  const logarithmic = scale === "log" && min > 0 && max > 0;
  const axisMin = logarithmic ? Math.log(min) : min;
  const axisMax = logarithmic ? Math.log(max) : max;
  const axisSpan = Math.max(1e-12, axisMax - axisMin);
  const toAxis = (frequency: number) => logarithmic ? Math.log(frequency) : frequency;
  const fromAxis = (value: number) => logarithmic ? Math.exp(value) : value;
  const baseFractionAtFrequency = (frequency: number) => (toAxis(frequency) - axisMin) / axisSpan;
  const fractionAtFrequency = (frequency: number) => inverted ? 1 - baseFractionAtFrequency(frequency) : baseFractionAtFrequency(frequency);
  const frequencyAtFraction = (fraction: number) => fromAxis(axisMin + (inverted ? 1 - fraction : fraction) * axisSpan);
  return {
    fractionAtFrequency,
    frequencyAtFraction,
    yAtFrequency: (frequency: number, top: number, height: number) => top + (1 - fractionAtFrequency(frequency)) * height,
    frequencyAtY: (y: number, top: number, height: number) => frequencyAtFraction(1 - clamp((y - top) / Math.max(1, height), 0, 1))
  };
}

function spectrogramNormalizationValue(value: unknown): SpectrogramNormalization {
  return value === "divide" || value === "subtract" ? value : "none";
}

function zeroChannelOffsets(count: number): ChannelOffsets {
  return {
    dx: Array(Math.max(0, count)).fill(0),
    dy: Array(Math.max(0, count)).fill(0),
    masked: Array(Math.max(0, count)).fill(false)
  };
}

function normalizeChannelOffsets(value: unknown, count: number): ChannelOffsets {
  const fallback = zeroChannelOffsets(count);
  if (!value || typeof value !== "object") return fallback;
  const raw = value as { dx?: unknown; dy?: unknown; masked?: unknown; channelMask?: unknown };
  const normalize = (axis: unknown) => Array.from({ length: count }, (_, index) => {
    const numeric = Array.isArray(axis) ? Number(axis[index]) : 0;
    return Number.isFinite(numeric) ? numeric : 0;
  });
  const masked = Array.from({ length: count }, (_, index) => Boolean(
    Array.isArray(raw.masked) ? raw.masked[index] : Array.isArray(raw.channelMask) ? raw.channelMask[index] : false
  ));
  return { dx: normalize(raw.dx), dy: normalize(raw.dy), masked };
}

function deriveSpwGroups(freqGhz: number[]): SpwGroup[] {
  const count = freqGhz.length;
  if (!count) return [];
  if (count === 1) return [{ id: 0, start: 0, end: 0 }];
  const gaps = freqGhz.slice(1).map((frequency, index) => Math.abs(frequency - freqGhz[index]));
  const finiteGaps = gaps.filter(Number.isFinite).sort((left, right) => left - right);
  const median = finiteGaps.length ? finiteGaps[Math.floor(finiteGaps.length / 2)] : 0;
  const largest = Math.max(...gaps);
  const split = Number.isFinite(largest) && median > 0 && largest > median * 1.05
    ? gaps.indexOf(largest) + 1
    : Math.floor(count / 2);
  return [
    { id: 0, start: 0, end: Math.max(0, split - 1) },
    { id: 1, start: split, end: count - 1 }
  ].filter((group) => group.start <= group.end);
}

function selectedCommonValue(values: number[], selected: number[]): string {
  if (!selected.length) return "";
  const first = values[selected[0]];
  return selected.every((index) => Object.is(values[index], first)) ? String(first) : "";
}

function coerceFrequencyRange(value: Partial<FrequencyRangeState> | undefined, freqGhz: number[]): FrequencyRangeState {
  const [fullMin, fullMax] = frequencyBounds(freqGhz);
  const requestedMin = numberValue(String(value?.min ?? fullMin), fullMin);
  const requestedMax = numberValue(String(value?.max ?? fullMax), fullMax);
  const min = clamp(requestedMin, fullMin, fullMax);
  const max = clamp(requestedMax, fullMin, fullMax);
  if (max <= min) return { min: String(fullMin), max: String(fullMax) };
  return { min: String(Number(min.toFixed(4))), max: String(Number(max.toFixed(4))) };
}

function timeBounds(timeMjd: number[]): [number, number] {
  const finite = timeMjd.filter(Number.isFinite);
  if (!finite.length) return [0, 1];
  return [Math.min(...finite), Math.max(...finite)];
}

function mjdToUtc(value: number): string {
  if (!Number.isFinite(value)) return "";
  const date = new Date(Math.round((value - 40587) * 86400000));
  if (!Number.isFinite(date.getTime())) return "";
  const iso = date.toISOString();
  const precision = iso.endsWith(".000Z") ? iso.slice(0, 19) : iso.slice(0, 23);
  return `${precision.replace("T", " ")} UTC`;
}

function parseTimeMjd(value: unknown, fallback: number): number {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) return numeric;
  const text = String(value ?? "").trim();
  if (!text) return fallback;
  const normalized = text.replace(/\s+UTC$/i, "Z").replace(" ", "T");
  const millis = Date.parse(normalized);
  return Number.isFinite(millis) ? millis / 86400000 + 40587 : fallback;
}

function coerceTimeRange(value: Partial<TimeRangeState> | undefined, timeMjd: number[]): TimeRangeState {
  const [fullMin, fullMax] = timeBounds(timeMjd);
  const requestedMin = parseTimeMjd(value?.min, fullMin);
  const requestedMax = parseTimeMjd(value?.max, fullMax);
  const min = clamp(requestedMin, fullMin, fullMax);
  const max = clamp(requestedMax, fullMin, fullMax);
  if (max <= min) return { min: mjdToUtc(fullMin), max: mjdToUtc(fullMax) };
  return { min: mjdToUtc(min), max: mjdToUtc(max) };
}

function timeRangeValues(value: Partial<TimeRangeState> | undefined, timeMjd: number[]): [number, number] {
  const [fullMin, fullMax] = timeBounds(timeMjd);
  const min = clamp(parseTimeMjd(value?.min, fullMin), fullMin, fullMax);
  const max = clamp(parseTimeMjd(value?.max, fullMax), fullMin, fullMax);
  return max > min ? [min, max] : [fullMin, fullMax];
}

function spectrogramTimePlotRect(rect: DOMRect, channelGutterVisible: boolean) {
  const left = channelGutterVisible ? 68 : 54;
  const right = 10;
  return { x: left, width: Math.max(1, rect.width - left - right) };
}

function timeToSharedPlotX(timeMjd: number, minMjd: number, maxMjd: number, plot: { x: number; width: number }): number {
  return plot.x + (timeMjd - minMjd) / Math.max(1e-12, maxMjd - minMjd) * plot.width;
}

function timeFromSharedPlotX(x: number, minMjd: number, maxMjd: number, plot: { x: number; width: number }): number {
  const fraction = clamp((x - plot.x) / Math.max(1, plot.width), 0, 1);
  return minMjd + fraction * (maxMjd - minMjd);
}

// Shared wheel-zoom math for the shared time window: zooms [currentMin,
// currentMax] around anchorMjd (the pointer's time position) by an amount
// derived from deltaY, clamped to [fullMin, fullMax]. Used by both the
// spectrogram's own wheel handler and the time-distance lane's, so the two
// canvases zoom the shared window with identical semantics/factor.
function zoomTimeWindow(anchorMjd: number, currentMin: number, currentMax: number, fullMin: number, fullMax: number, deltaY: number): [number, number] {
  const factor = Math.exp(Math.max(-1.2, Math.min(1.2, deltaY * 0.0015)));
  const nextSpan = Math.min(fullMax - fullMin, Math.max(1e-12, (currentMax - currentMin) * factor));
  const fraction = clamp((anchorMjd - currentMin) / Math.max(1e-12, currentMax - currentMin), 0, 1);
  const nextMin = clamp(anchorMjd - nextSpan * fraction, fullMin, fullMax - nextSpan);
  const nextMax = clamp(nextMin + nextSpan, fullMin, fullMax);
  return [nextMin, nextMax];
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (["true", "1", "yes", "on"].includes(text)) return true;
    if (["false", "0", "no", "off"].includes(text)) return false;
  }
  return fallback;
}

function playbackFpsValue(value: unknown): number {
  const numeric = Number(value);
  return PLAYBACK_FPS_OPTIONS.some((fps) => fps === numeric) ? numeric : DEFAULT_PLAYBACK_FPS;
}

// navigator.deviceMemory (Device Memory API) isn't in TypeScript's DOM lib
// and isn't supported by every browser (notably Safari/Firefox); treat it as
// an optional, display-only hint rather than something to gate behavior on.
function deviceMemoryGb(): number | undefined {
  if (typeof navigator === "undefined") return undefined;
  const value = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function imageUrl(base: string, params: Record<string, string | number>): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) query.set(key, String(value));
  return `${base}?${query.toString()}`;
}

function displayParamsWithoutAddress(params: Record<string, string | number>): Record<string, string | number> {
  const displayParams = { ...params };
  delete displayParams.timeIndex;
  delete displayParams.sampleMjd;
  return displayParams;
}

function frameCapParams(cap: FrameRequestCap | undefined): Record<string, number> {
  return cap ? { maxWidth: cap.maxWidth, maxHeight: cap.maxHeight } : {};
}

// Motion resolution ladder: halve an existing panel cap (never invent one -
// the "full resolution" tier reports cap === null, and there is no cap
// value to halve there, so that tier is left untouched by the ladder).
// Rounded down to an even pixel count on each axis.
function halveFrameCap(cap: FrameRequestCap): FrameRequestCap {
  if (!cap) return cap;
  const half = (value: number) => {
    const halved = Math.floor(value / 2);
    return Math.max(2, halved - (halved % 2));
  };
  return { maxWidth: half(cap.maxWidth), maxHeight: half(cap.maxHeight) };
}

function scheduledFrameRequest(
  url: string,
  identity: Parameters<typeof requestKey>[0],
  cursorMjd: number,
  predictedResolution: FrameResolution
): ScheduledFrameRequest {
  const bitmapIdentity = {
    ...identity,
    // Bitmap identity is native-source resolved, never the master cursor's
    // nearest index. Unavailable predictions get a distinct sentinel key.
    resolvedIndex: predictedResolution.unavailable
      ? -1
      : predictedResolution.resolvedIndex ?? identity.resolvedIndex
  };
  const bitmapKey = requestKey(bitmapIdentity);
  const cursorKey = Number.isFinite(cursorMjd) ? cursorMjd.toPrecision(15) : "nan";
  return {
    url,
    bitmapKey,
    key: `${bitmapKey}|cursor=${cursorKey}`,
    predictedResolution,
    kind: identity.kind
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampIndex(value: unknown, fallback: number, length: number): number {
  const max = Math.max(0, length - 1);
  const numeric = Number(value);
  return clamp(Number.isFinite(numeric) ? Math.round(numeric) : fallback, 0, max);
}

function scrubPrefetchOffsets(direction: number): number[] {
  const forward = direction < 0 ? -1 : 1;
  const offsets: number[] = [];
  let ahead = 1;
  let behind = 1;
  while (ahead <= SCRUB_PREFETCH_RADIUS || behind <= SCRUB_PREFETCH_RADIUS) {
    for (let count = 0; count < 2 && ahead <= SCRUB_PREFETCH_RADIUS; count += 1) {
      offsets.push(forward * ahead++);
    }
    if (behind <= SCRUB_PREFETCH_RADIUS) offsets.push(-forward * behind++);
  }
  return offsets;
}

function closestIndex(values: number[], target: number): number {
  if (!values.length) return 0;
  let best = 0;
  let bestDistance = Math.abs(values[0] - target);
  for (let index = 1; index < values.length; index += 1) {
    const distance = Math.abs(values[index] - target);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best;
}

function rememberImage(url: string, image: HTMLImageElement) {
  imageCache.delete(url);
  imageCache.set(url, image);
  while (imageCache.size > IMAGE_CACHE_LIMIT) {
    const oldest = imageCache.keys().next().value;
    if (oldest === undefined) break;
    imageCache.delete(oldest);
  }
}

function normalizeSolarView(view: SolarView): SolarView {
  const xMin = Math.min(view.xMin, view.xMax);
  const xMax = Math.max(view.xMin, view.xMax);
  const yMin = Math.min(view.yMin, view.yMax);
  const yMax = Math.max(view.yMin, view.yMax);
  return {
    xMin,
    xMax: xMax > xMin ? xMax : xMin + 1,
    yMin,
    yMax: yMax > yMin ? yMax : yMin + 1
  };
}

function defaultSolarView(meta: SessionMeta): SolarView {
  return normalizeSolarView(meta.aia.cornersArcsec);
}

function sourcesForMeta(meta: SessionMeta | null): SourceMeta[] {
  if (!meta) return [];
  if (Array.isArray(meta.sources) && meta.sources.length) return meta.sources;
  return [
    {
      id: "context",
      role: "context",
      label: "AIA 131 Å",
      instrument: "AIA",
      format: "hdf",
      shape: [meta.aia.shape[0], meta.aia.shape[1], meta.aia.times.length],
      time: { count: meta.aia.times.length, times: meta.aia.times, timeMjd: meta.aia.timeMjd, start: meta.aia.times[0], end: meta.aia.times[meta.aia.times.length - 1] },
      capabilities: { render: true, difference: true, tracking: true, roi: true },
      status: "ready"
    },
    {
      id: "radio",
      role: "radio",
      label: "EOVSA",
      instrument: "EOVSA",
      format: "fits",
      shape: [meta.eovsa.freqGhz.length, meta.eovsa.times.length, meta.eovsa.shape[0], meta.eovsa.shape[1]],
      time: { count: meta.eovsa.times.length, times: meta.eovsa.times, timeMjd: meta.eovsa.timeMjd, start: meta.eovsa.times[0], end: meta.eovsa.times[meta.eovsa.times.length - 1] },
      freqGhz: meta.eovsa.freqGhz,
      capabilities: { render: true, difference: true, overlay: true, centroid: true, roi: true },
      status: "ready"
    },
    {
      id: "spectrogram",
      role: "spectrogram",
      label: "EOVSA Dynamic Spectrum",
      instrument: "EOVSA",
      format: "fits",
      shape: meta.spectrogram.shape,
      time: { count: meta.spectrogram.timeMjd.length, timeMjd: meta.spectrogram.timeMjd },
      freqGhz: meta.spectrogram.freqGhz,
      capabilities: { render: true },
      status: "ready"
    }
  ];
}

/** Keep exported source metadata useful without embedding native time axes. */
function slimSourceForSession(source: SourceMeta): Record<string, unknown> {
  const saved = { ...source } as Record<string, unknown>;
  const time = source.time;
  if (time) {
    const times = Array.isArray(time.times) ? time.times : [];
    const mjd = Array.isArray(time.timeMjd) ? time.timeMjd.filter(Number.isFinite) : [];
    const count = Number.isFinite(Number(time.count)) ? Number(time.count) : Math.max(times.length, time.timeMjd?.length ?? 0);
    saved.time = {
      count,
      start: time.start ?? times[0] ?? (mjd.length ? mjdToUtc(mjd[0]) : ""),
      end: time.end ?? times[times.length - 1] ?? (mjd.length ? mjdToUtc(mjd[mjd.length - 1]) : "")
    };
  }
  return saved;
}

function sourceRole(source: SourceMeta | undefined, overrides: Record<string, SourceRole>): SourceRole {
  if (!source) return "";
  if (source.role === "context" || source.role === "radio" || source.role === "spectrogram") return source.role;
  return overrides[source.id] ?? source.role;
}

function sourceInfo(source: SourceMeta): string {
  const shape = source.shape?.length ? source.shape.join(" x ") : "unknown shape";
  const time = source.time?.start && source.time?.end ? `${source.time.start}-${source.time.end}` : `${source.time?.count ?? 0} frames`;
  return `${source.format ?? "unknown"} · ${shape} · ${time}`;
}

function sourceTimeMjd(source: SourceMeta | undefined, meta: SessionMeta | null, overrides: Record<string, SourceRole> = {}): number[] {
  if (!source || !meta) return [];
  const role = sourceRole(source, overrides);
  const values = role === "context" ? meta.aia.timeMjd : role === "radio" ? meta.eovsa.timeMjd : role === "spectrogram" ? meta.spectrogram.timeMjd : source.time?.timeMjd ?? [];
  return values.filter(Number.isFinite).slice().sort((left, right) => left - right);
}

function nativeCadenceSeconds(times: number[]): number {
  const deltas = times
    .filter(Number.isFinite)
    .slice(1)
    .map((value, index) => (value - times[index]) * 86400)
    .filter((value) => value > 0 && Number.isFinite(value))
    .sort((left, right) => left - right);
  return deltas.length ? deltas[Math.floor(deltas.length / 2)] : 0;
}

function defaultDifference(role: SourceRole, cadenceSeconds: string | number = DEFAULT_DIFF_SECONDS, timeCount = 1): DifferenceState {
  return {
    operation: role === "spectrogram" ? "none" : role === "context" ? "ratio" : "subtract",
    reference: "previous",
    cadenceSeconds: String(cadenceSeconds),
    meanStartIndex: 0,
    meanEndIndex: Math.min(4, Math.max(0, timeCount - 1))
  };
}

function normalizeTemporal(value: unknown, fallback: TemporalState = { mode: "none", sigmaShort: String(DEFAULT_TEMPORAL_SIGMA_SHORT), sigmaLong: String(DEFAULT_TEMPORAL_SIGMA_LONG) }): TemporalState {
  if (!value || typeof value !== "object") return { ...fallback };
  const candidate = value as Partial<TemporalState>;
  const mode: TemporalMode = candidate.mode === "lowpass" || candidate.mode === "bandpass" ? candidate.mode : "none";
  const sigmaShort = numberValue(String(candidate.sigmaShort ?? fallback.sigmaShort), DEFAULT_TEMPORAL_SIGMA_SHORT);
  const sigmaLong = numberValue(String(candidate.sigmaLong ?? fallback.sigmaLong), DEFAULT_TEMPORAL_SIGMA_LONG);
  return {
    mode,
    sigmaShort: String(sigmaShort > 0 ? sigmaShort : DEFAULT_TEMPORAL_SIGMA_SHORT),
    sigmaLong: String(sigmaLong > 0 ? sigmaLong : DEFAULT_TEMPORAL_SIGMA_LONG)
  };
}

function normalizeScale(value: unknown, fallback: ScaleMode): ScaleMode {
  return value === "linear" || value === "log" || value === "sqrt" || value === "asinh" ? value : fallback;
}

function defaultLayer(id: string, label: string, sourceId: string, kind: LayerKind, role: SourceRole, display: DisplayState, difference: DifferenceState, freqIndex = 0): LayerState {
  return {
    id,
    label,
    labelEdited: false,
    sourceId,
    sourceRoleSnapshot: role,
    kind,
    visible: true,
    opacity: kind === "contours" ? 0.8 : 1,
    operation: kind === "contours" ? "none" : difference.operation,
    reference: kind === "contours" ? "previous" : difference.reference,
    cadenceSeconds: difference.cadenceSeconds,
    meanStartMjd: difference.meanStartMjd,
    meanEndMjd: difference.meanEndMjd,
    temporal: normalizeTemporal(undefined),
    samplingPolicy: "nearest",
    maxOffsetSeconds: undefined,
    display: { ...display, radialGamma: Math.max(0, numberValue(String(display.radialGamma ?? 0), 0)) },
    freqIndex,
    contourLevelPercent: "50",
    contourLevelKelvin: "1000000",
    contourLevelSfu: "1.0",
    contourLevelMode: "percent",
    contourLevelReference: "current",
    contourFilled: false,
    contourOpacity: "0.35",
    contourCmap: "turbo"
  };
}

function layerKindRank(kind: LayerKind): number {
  return kind === "contours" ? 1 : kind === "spectrogram" ? 2 : 0;
}

function normalizeLayerOrder(layers: LayerState[]): LayerState[] {
  return layers
    .map((layer, index) => ({ layer, index }))
    .sort((left, right) => layerKindRank(left.layer.kind) - layerKindRank(right.layer.kind) || left.index - right.index)
    .map(({ layer }) => layer);
}

function normalizeLayer(value: Partial<LayerState>, fallback: LayerState): LayerState {
  const raw = value as Partial<LayerState> & {
    sampling?: { policy?: string; maxOffsetSeconds?: number };
    display?: Partial<DisplayState> & { opacity?: number };
    temporal?: Partial<TemporalState>;
    contour?: Partial<LayerState> & {
      levelMode?: string;
      levelReference?: string;
      levelPercent?: string | number;
      levelKelvin?: string | number;
      levelSfu?: string | number;
      filled?: boolean;
      opacity?: string | number;
      cmap?: string;
    };
  };
  const sampling = raw.sampling;
  const display = raw.display ?? value.display;
  const temporal = normalizeTemporal(raw.temporal, fallback.temporal);
  const contour = raw.contour;
  const parsedMaxOffset = Number(sampling?.maxOffsetSeconds ?? value.maxOffsetSeconds);
  const maxOffsetSeconds = Number.isFinite(parsedMaxOffset) && parsedMaxOffset >= 0
    ? parsedMaxOffset
    : fallback.maxOffsetSeconds;
  const samplingPolicy: LayerState["samplingPolicy"] = sampling?.policy === "previous" || sampling?.policy === "next" ? sampling.policy : value.samplingPolicy === "previous" || value.samplingPolicy === "next" ? value.samplingPolicy : fallback.samplingPolicy;
  const displayCmap = COLORMAP_OPTIONS.some((option) => option.id === display?.cmap) ? String(display?.cmap) : normalizeColormap(display?.cmap, normalizeColormap(fallback.display.cmap));
  const displayScale = normalizeScale(display?.scale, fallback.display.scale);
  const contourReference = (value.contourLevelReference ?? contour?.contourLevelReference ?? contour?.levelReference) === "global" ? "global" : "current";
  const savedContourMode = value.contourLevelMode ?? contour?.contourLevelMode ?? contour?.levelMode;
  const contourMode = contourReference === "current"
    ? "percent"
    : savedContourMode === "kelvin" ? "kelvin" : savedContourMode === "sfu" ? "sfu" : "percent";
  const normalized: LayerState = {
    ...fallback,
    ...value,
    id: String(value.id ?? fallback.id),
    mirrorOf: typeof value.mirrorOf === "string" && value.mirrorOf ? value.mirrorOf : undefined,
    labelEdited: Boolean(value.labelEdited ?? fallback.labelEdited ?? false),
    sourceId: String(value.sourceId ?? fallback.sourceId),
    sourceRoleSnapshot: String(value.sourceRoleSnapshot ?? fallback.sourceRoleSnapshot),
    kind: value.kind === "contours" || value.kind === "spectrogram" ? value.kind : "image",
    visible: value.visible !== false,
    operation: value.operation === "subtract" || value.operation === "ratio" ? value.operation : contour?.operation === "subtract" || contour?.operation === "ratio" ? contour.operation : "none",
    reference: value.reference === "base" || value.reference === "mean" ? value.reference : contour?.reference === "base" || contour?.reference === "mean" ? contour.reference : "previous",
    samplingPolicy,
    maxOffsetSeconds,
    display: {
      ...fallback.display,
      ...(display ?? {}),
      cmap: displayCmap,
      scale: displayScale,
      radialGamma: Math.max(0, numberValue(String(display?.radialGamma ?? fallback.display.radialGamma ?? 0), fallback.display.radialGamma ?? 0))
    } as DisplayState,
    temporal,
    opacity: clamp(Number((display as (Partial<DisplayState> & { opacity?: number }) | undefined)?.opacity ?? value.opacity ?? fallback.opacity), 0, 1),
    freqIndex: Math.max(0, Math.round(Number(value.freqIndex ?? (value as unknown as { frequencyIndex?: number }).frequencyIndex ?? fallback.freqIndex))),
    contourLevelPercent: String(value.contourLevelPercent ?? contour?.contourLevelPercent ?? contour?.levelPercent ?? fallback.contourLevelPercent),
    contourLevelKelvin: String(value.contourLevelKelvin ?? contour?.contourLevelKelvin ?? contour?.levelKelvin ?? fallback.contourLevelKelvin),
    contourLevelSfu: String(value.contourLevelSfu ?? contour?.contourLevelSfu ?? contour?.levelSfu ?? fallback.contourLevelSfu),
    contourLevelMode: contourMode,
    contourLevelReference: contourReference,
    contourFilled: Boolean(value.contourFilled ?? contour?.contourFilled ?? contour?.filled ?? fallback.contourFilled),
    contourOpacity: String(value.contourOpacity ?? contour?.contourOpacity ?? contour?.opacity ?? fallback.contourOpacity),
    contourCmap: contourColormap(value.contourCmap ?? contour?.contourCmap ?? contour?.cmap ?? fallback.contourCmap)
  };
  if (normalized.kind === "contours" && normalized.opacity < 1) {
    normalized.contourOpacity = String(clamp(numberValue(normalized.contourOpacity, 1) * normalized.opacity, 0, 1));
    normalized.opacity = 1;
  }
  return normalized;
}

function serializeLayer(layer: LayerState, settings: LayerState = layer) {
  return {
    id: layer.id,
    ...(layer.mirrorOf ? { mirrorOf: layer.mirrorOf } : {}),
    label: settings.label,
    labelEdited: settings.labelEdited,
    sourceId: settings.sourceId,
    sourceRoleSnapshot: settings.sourceRoleSnapshot,
    kind: settings.kind,
    visible: settings.visible,
    operation: settings.operation,
    reference: settings.reference,
    cadenceSeconds: settings.cadenceSeconds,
    meanStartMjd: settings.meanStartMjd,
    meanEndMjd: settings.meanEndMjd,
    sampling: { policy: settings.samplingPolicy, maxOffsetSeconds: settings.maxOffsetSeconds },
    temporal: { ...settings.temporal },
    display: { ...settings.display, opacity: settings.opacity },
    frequencyIndex: settings.freqIndex,
    contour: {
      levelMode: settings.contourLevelMode,
      levelReference: settings.contourLevelReference,
      levelPercent: settings.contourLevelPercent,
      levelKelvin: settings.contourLevelKelvin,
      levelSfu: settings.contourLevelSfu,
      filled: settings.contourFilled,
      opacity: settings.contourOpacity,
      cmap: settings.contourCmap
    }
  };
}

function normalizeDifference(
  value: unknown,
  role: SourceRole,
  cadenceSeconds: string | number = DEFAULT_DIFF_SECONDS,
  timeCount = 1,
  legacyTimeMjd: number[] = []
): DifferenceState {
  const fallback = defaultDifference(role, cadenceSeconds, timeCount);
  if (!value || typeof value !== "object") return fallback;
  const candidate = value as Partial<DifferenceState> & { mode?: DifferenceMode };
  const operation = candidate.operation === "none" || candidate.operation === "subtract" || candidate.operation === "ratio"
    ? candidate.operation
    : candidate.mode === "none"
      ? "none"
      : candidate.mode === "base"
        ? "subtract"
        : candidate.mode === "running"
          ? role === "context" ? "ratio" : "subtract"
          : fallback.operation;
  const reference = candidate.reference === "previous" || candidate.reference === "base" || candidate.reference === "mean"
    ? candidate.reference
    : candidate.mode === "base" ? "base" : fallback.reference;
  const legacyStartMjd = Number(candidate.meanStartMjd);
  const legacyEndMjd = Number(candidate.meanEndMjd);
  const hasExplicitMean = Number.isFinite(legacyStartMjd) || Number.isFinite(legacyEndMjd);
  const indexCount = !hasExplicitMean && legacyTimeMjd.length ? legacyTimeMjd.length : timeCount;
  const startIndex = clampIndex(candidate.meanStartIndex, fallback.meanStartIndex, indexCount);
  const endIndex = clampIndex(candidate.meanEndIndex, fallback.meanEndIndex, indexCount);
  return {
    operation,
    reference,
    cadenceSeconds: String(candidate.cadenceSeconds ?? cadenceSeconds),
    meanStartIndex: startIndex,
    meanEndIndex: endIndex,
    meanStartMjd: Number.isFinite(legacyStartMjd) ? legacyStartMjd : legacyTimeMjd[startIndex],
    meanEndMjd: Number.isFinite(legacyEndMjd) ? legacyEndMjd : legacyTimeMjd[endIndex]
  };
}

function defaultDifferences(sources: SourceMeta[], cadenceSeconds: string | number = DEFAULT_DIFF_SECONDS, timeCount = 1): Record<string, DifferenceState> {
  return Object.fromEntries(sources.map((source) => [source.id, defaultDifference(source.role, cadenceSeconds, timeCount)]));
}

function differenceLabel(state: DifferenceState): string {
  if (state.operation === "none") return "Original";
  const operation = state.operation === "ratio" ? "Ratio" : "Subtraction";
  const reference = state.reference === "previous" ? "Previous" : state.reference === "base" ? "Base" : "Mean";
  return `${reference} ${operation}`;
}

function legacyDifferenceMode(state: DifferenceState): DifferenceMode {
  if (state.operation === "none") return "none";
  return state.reference === "base" ? "base" : "running";
}

function differenceParams(state: DifferenceState, timeMjd: number[]) {
  const startIndex = clampIndex(Math.min(state.meanStartIndex, state.meanEndIndex), 0, timeMjd.length);
  const endIndex = clampIndex(Math.max(state.meanStartIndex, state.meanEndIndex), startIndex, timeMjd.length);
  return {
    differenceOperation: state.operation,
    differenceReference: state.reference,
    differenceMode: legacyDifferenceMode(state),
    diffSeconds: numberValue(state.cadenceSeconds, DEFAULT_DIFF_SECONDS),
    meanStartMjd: typeof state.meanStartMjd === "number" && Number.isFinite(state.meanStartMjd) ? state.meanStartMjd : timeMjd[startIndex] ?? 0,
    meanEndMjd: typeof state.meanEndMjd === "number" && Number.isFinite(state.meanEndMjd) ? state.meanEndMjd : timeMjd[endIndex] ?? timeMjd[startIndex] ?? 0,
    useRunningDiff: state.operation !== "none" && state.reference === "previous" ? 1 : 0
  };
}

function filterParams(layer: LayerState | undefined): Record<string, string | number> {
  if (!layer || layer.kind !== "image") return {};
  const params: Record<string, string | number> = {};
  const gamma = Math.max(0, numberValue(String(layer.display.radialGamma ?? 0), 0));
  if (gamma > 0) params.radialGamma = Number(gamma.toFixed(1));
  const temporal = normalizeTemporal(layer.temporal);
  if (temporal.mode !== "none") {
    params.temporalMode = temporal.mode;
    params.temporalSigmaShort = numberValue(temporal.sigmaShort, DEFAULT_TEMPORAL_SIGMA_SHORT);
    params.temporalSigmaLong = numberValue(temporal.sigmaLong, DEFAULT_TEMPORAL_SIGMA_LONG);
  }
  return params;
}

function differenceForLayer(layer: LayerState | undefined, fallback: DifferenceState): DifferenceState {
  if (!layer) return fallback;
  return {
    operation: layer.operation,
    reference: layer.reference,
    cadenceSeconds: layer.cadenceSeconds,
    meanStartIndex: 0,
    meanEndIndex: 0,
    meanStartMjd: layer.meanStartMjd,
    meanEndMjd: layer.meanEndMjd
  };
}

function radioPeakCacheKey(state: DifferenceState): string {
  const legacyMode = state.operation === "none" ? "none" : state.reference === "base" ? "base" : "running";
  return `${legacyMode}:dt=${numberValue(state.cadenceSeconds, DEFAULT_DIFF_SECONDS).toFixed(3)}`;
}

function roleSource(sources: SourceMeta[], overrides: Record<string, SourceRole>, role: SourceRole): SourceMeta | undefined {
  return sources.find((source) => sourceRole(source, overrides) === role);
}

function normalizeTracks(value: unknown, defaultSourceId = "context"): SadTrack[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((candidate, index) => {
    if (!candidate || typeof candidate !== "object") return [];
    const track = candidate as Partial<SadTrack>;
    if (!Array.isArray(track.anchors) || !Array.isArray(track.points)) return [];
    const anchors = track.anchors.flatMap((anchor) => {
      const frameIndex = Number(anchor?.frameIndex);
      const mjd = Number(anchor?.mjd);
      const x = Number(anchor?.x);
      const y = Number(anchor?.y);
      return [frameIndex, mjd, x, y].every(Number.isFinite) ? [{ frameIndex, mjd, x, y }] : [];
    }).sort((left, right) => left.frameIndex - right.frameIndex);
    const anchorFrames = new Set(anchors.map((anchor) => anchor.frameIndex));
    const points = track.points.flatMap((point) => {
      const frameIndex = Number(point?.frameIndex);
      const mjd = Number(point?.mjd);
      const x = Number(point?.x);
      const y = Number(point?.y);
      const confidence = Number(point?.confidence ?? 1);
      return [frameIndex, mjd, x, y, confidence].every(Number.isFinite)
        ? [{ frameIndex, mjd, x, y, confidence, isAnchor: anchorFrames.has(frameIndex) || Boolean(point?.isAnchor) }]
        : [];
    }).sort((left, right) => left.frameIndex - right.frameIndex);
    const state = track.state === "stopped-edge" || track.state === "stopped-low-confidence" ? track.state : "active";
    return [{
      id: String(track.id || `track-${index + 1}`),
      label: String(track.label || `Track ${index + 1}`),
      sourceId: String(track.sourceId || defaultSourceId),
      color: String(track.color || "#56c7d9"),
      visible: track.visible !== false,
      anchors,
      points,
      state
    }];
  });
}

// Shared point-array parser for persisted session JSON: curveArcsec,
// rawCurveArcsec, and (in normalizeFan) boundaryA/B/rawBoundaryA/B all take
// the same "array of finite [x, y] pairs, drop anything malformed" shape.
function parsePointArray(input: unknown): [number, number][] {
  return Array.isArray(input) ? input.flatMap((point) => {
    const tuple = point as [unknown, unknown];
    const x = Number(tuple?.[0]);
    const y = Number(tuple?.[1]);
    return Number.isFinite(x) && Number.isFinite(y) ? [[x, y] as [number, number]] : [];
  }) : [];
}

function normalizePanelId(value: unknown): PanelId | undefined {
  return value === "eovsa" ? "eovsa" : value === "aia" ? "aia" : undefined;
}

// MIGRATION APPROACH: normalizeSlits runs at session-load time, before this
// module's own React state (meta/sourceAffine) has committed the just-loaded
// source's pixel scales - so it cannot call the component's own sourceAffine
// closure (which would still read the PREVIOUS session's affines mid-call).
// Instead the caller (applyLoadedSession, where the freshly loaded
// LoadSessionResponse itself carries wcs.aia/eovsa.pixelToWorldAffine) is
// restructured to build a `scaleForSource` lookup from that fresh data and
// pass it in here, rather than normalizeSlits reading ambient state or
// deferring the conversion to first use. A legacy slit with a raw pixel
// `width` and no widthArcsec is converted once, here, via
// widthArcsec = width * scaleForSource(slit.sourceId); an already-migrated
// slit's own widthArcsec is trusted (sanity-clamped) and read directly.
function normalizeSlits(value: unknown, scaleForSource: (sourceId: string) => number): SlitDefinition[] {
  if (!Array.isArray(value)) return [];
  const normalized: SlitDefinition[] = value.flatMap((candidate, index) => {
    if (!candidate || typeof candidate !== "object") return [];
    const slit = candidate as Partial<SlitDefinition> & { width?: unknown };
    const curveArcsec = parsePointArray(slit.curveArcsec);
    if (curveArcsec.length < 2) return [];
    const sourceId = String(slit.sourceId || "context");
    const explicitWidthArcsec = Number(slit.widthArcsec);
    const legacyWidthPx = Number(slit.width);
    // Legacy `width` field tolerated on read (pre-arcsec sessions), never
    // written back - see SlitDefinition.widthArcsec.
    const widthArcsec = Number.isFinite(explicitWidthArcsec) && explicitWidthArcsec > 0
      ? clamp(explicitWidthArcsec, 1e-3, SLIT_WIDTH_ARCSEC_HARD_CAP)
      : Number.isFinite(legacyWidthPx) && legacyWidthPx > 0
        ? clamp(
            clamp(Math.round(legacyWidthPx), 1, SLIT_WIDTH_HARD_CAP_PX) * scaleForSource(sourceId),
            1e-3,
            SLIT_WIDTH_ARCSEC_HARD_CAP
          )
        : DEFAULT_SLIT_WIDTH_ARCSEC;
    // rawCurveArcsec/smoothPx/drawnPanel are optional (absent for slits
    // saved before per-slit re-smoothing existed, or derived from a fan
    // family - see SlitDefinition's docstring): the per-slit Smooth control
    // is disabled without a raw stroke to re-smooth from, so a malformed or
    // too-short raw array is treated the same as "absent" rather than
    // dropping the whole slit.
    const rawCurveArcsec = parsePointArray(slit.rawCurveArcsec);
    const display = slit.display ?? { vmin: 0, vmax: 1, cmap: "magma", scale: "linear" };
    const scale: SlitDisplayState["scale"] = display.scale === "sqrt" || display.scale === "log" ? display.scale : "linear";
    return [{
      id: String(slit.id || `slit-${index + 1}`),
      linkedTo: typeof slit.linkedTo === "string" && slit.linkedTo ? slit.linkedTo : undefined,
      name: String(slit.name || `Slit ${index + 1}`),
      color: String(slit.color || TRACK_COLORS[index % TRACK_COLORS.length]),
      visible: slit.visible !== false,
      sourceId,
      layerId: String(slit.layerId || ""),
      bindingKind: slit.bindingKind === "contours" || slit.bindingKind === "raw" ? slit.bindingKind : "image",
      widthArcsec,
      shiftSeconds: Number.isFinite(Number(slit.shiftSeconds)) ? Number(slit.shiftSeconds) : 0,
      curveArcsec,
      inputVertexCount: Math.max(2, Math.round(Number(slit.inputVertexCount ?? curveArcsec.length))),
      rawCurveArcsec: rawCurveArcsec.length >= 2 ? rawCurveArcsec : undefined,
      smoothPx: Number.isFinite(Number(slit.smoothPx)) ? clamp(Number(slit.smoothPx), 0, SLIT_SMOOTH_MAX_PX) : undefined,
      drawnPanel: normalizePanelId(slit.drawnPanel),
      freqIndices: Array.isArray(slit.freqIndices)
        ? [...new Set(slit.freqIndices.map(Number).filter((value) => Number.isInteger(value) && value >= 0))]
        : [],
      baseFreqIndex: Number.isInteger(Number(slit.baseFreqIndex)) ? Number(slit.baseFreqIndex) : null,
      contourFreqIndices: Array.isArray(slit.contourFreqIndices)
        ? [...new Set(slit.contourFreqIndices.map(Number).filter((value) => Number.isInteger(value) && value >= 0))]
        : [],
      contourLevelPercent: clamp(Number(slit.contourLevelPercent ?? 70), 1, 99),
      display: {
        vmin: Number.isFinite(Number(display.vmin)) ? Number(display.vmin) : 0,
        vmax: Number.isFinite(Number(display.vmax)) ? Number(display.vmax) : 1,
        cmap: normalizeColormap(display.cmap, "magma"),
        scale
      }
    }];
  });
  // A twin (linkedTo set) whose original didn't survive normalization - a
  // dangling link, e.g. an old session file, or the original was itself
  // dropped for having too short a curve - has no live source left to
  // resolve geometry from. Materialize it into an independent slit by
  // dropping the link; its own curveArcsec/inputVertexCount/width (already
  // parsed above from its saved snapshot, same as any other slit) become its
  // real geometry going forward. Mirrors how orphaned mirrorOf layers fall
  // back to independentLayer() on restore (see the panels-normalization pass
  // that calls it).
  const ids = new Set(normalized.map((slit) => slit.id));
  return normalized.map((slit) => (
    slit.linkedTo && (slit.linkedTo === slit.id || !ids.has(slit.linkedTo))
      ? { ...slit, linkedTo: undefined }
      : slit
  ));
}

function normalizeFan(value: unknown): FanDefinition | null {
  if (!value || typeof value !== "object") return null;
  const candidate = value as Partial<FanDefinition>;
  const boundaryA = parsePointArray(candidate.boundaryA);
  const boundaryB = parsePointArray(candidate.boundaryB);
  if (boundaryA.length < 2 || boundaryB.length < 2) return null;
  // rawBoundaryA/B/smoothPx/boundaryADrawnPanel/boundaryBDrawnPanel are
  // optional - absent for fans saved before per-fan re-smoothing existed;
  // the fan "Smooth" control is disabled without both raw boundaries (see
  // FanDefinition's docstring and SlitInspectorCard).
  const rawBoundaryA = parsePointArray(candidate.rawBoundaryA);
  const rawBoundaryB = parsePointArray(candidate.rawBoundaryB);
  const rawPromotions = candidate.promotedMembers && typeof candidate.promotedMembers === "object" ? candidate.promotedMembers : {};
  return {
    id: String(candidate.id || "fan-1"),
    sourceId: String(candidate.sourceId || "context"),
    layerId: String(candidate.layerId || ""),
    bindingKind: candidate.bindingKind === "contours" || candidate.bindingKind === "raw" ? candidate.bindingKind : "image",
    boundaryA,
    boundaryB,
    intermediateCount: clamp(Math.round(Number(candidate.intermediateCount ?? 3)), 1, 20),
    inputVertexCounts: [
      Math.max(2, Math.round(Number(candidate.inputVertexCounts?.[0] ?? boundaryA.length))),
      Math.max(2, Math.round(Number(candidate.inputVertexCounts?.[1] ?? boundaryB.length)))
    ],
    promotedMembers: Object.fromEntries(Object.entries(rawPromotions).map(([key, slitId]) => [key, String(slitId)])),
    rawBoundaryA: rawBoundaryA.length >= 2 ? rawBoundaryA : undefined,
    rawBoundaryB: rawBoundaryB.length >= 2 ? rawBoundaryB : undefined,
    boundaryADrawnPanel: normalizePanelId(candidate.boundaryADrawnPanel),
    boundaryBDrawnPanel: normalizePanelId(candidate.boundaryBDrawnPanel),
    smoothPx: Number.isFinite(Number(candidate.smoothPx)) ? clamp(Number(candidate.smoothPx), 0, SLIT_SMOOTH_MAX_PX) : undefined
  };
}

function resampleCurveCount(points: [number, number][], count: number): [number, number][] {
  if (points.length < 2) return [];
  const cumulative = [0];
  for (let index = 1; index < points.length; index += 1) {
    cumulative.push(cumulative[index - 1] + Math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]));
  }
  const total = cumulative.at(-1) ?? 0;
  if (!(total > 0)) return [];
  return Array.from({ length: Math.max(2, count) }, (_, index) => {
    const distance = total * index / Math.max(1, count - 1);
    let upper = cumulative.findIndex((value) => value >= distance);
    if (upper <= 0) upper = 1;
    const lower = upper - 1;
    const fraction = (distance - cumulative[lower]) / Math.max(1e-12, cumulative[upper] - cumulative[lower]);
    return [
      points[lower][0] + (points[upper][0] - points[lower][0]) * fraction,
      points[lower][1] + (points[upper][1] - points[lower][1]) * fraction
    ];
  });
}

function createFanDefinition(
  binding: SlitSourceOption,
  boundaryAInput: [number, number][],
  boundaryBInput: [number, number][],
  inputVertexCounts: [number, number],
  intermediateCount: number
): { fan: FanDefinition; reversedBoundaryB: boolean } | null {
  const count = Math.max(2, boundaryAInput.length, boundaryBInput.length);
  const boundaryA = resampleCurveCount(boundaryAInput, count);
  let boundaryB = resampleCurveCount(boundaryBInput, count);
  if (boundaryA.length < 2 || boundaryB.length < 2) return null;
  const forward = Math.hypot(boundaryA[0][0] - boundaryB[0][0], boundaryA[0][1] - boundaryB[0][1])
    + Math.hypot(boundaryA.at(-1)![0] - boundaryB.at(-1)![0], boundaryA.at(-1)![1] - boundaryB.at(-1)![1]);
  const reverse = Math.hypot(boundaryA[0][0] - boundaryB.at(-1)![0], boundaryA[0][1] - boundaryB.at(-1)![1])
    + Math.hypot(boundaryA.at(-1)![0] - boundaryB[0][0], boundaryA.at(-1)![1] - boundaryB[0][1]);
  const reversedBoundaryB = reverse < forward;
  if (reversedBoundaryB) boundaryB = [...boundaryB].reverse();
  return {
    reversedBoundaryB,
    fan: {
      id: `fan-${crypto.randomUUID().slice(0, 8)}`,
      sourceId: binding.sourceId,
      layerId: binding.layer.id,
      bindingKind: binding.bindingKind,
      boundaryA,
      boundaryB,
      intermediateCount: clamp(Math.round(intermediateCount), 1, 20),
      inputVertexCounts,
      promotedMembers: {}
    }
  };
}

function fanFamilyCurves(fan: FanDefinition): [number, number][][] {
  const count = fan.intermediateCount + 2;
  return Array.from({ length: count }, (_, index) => {
    const fraction = index / Math.max(1, count - 1);
    return fan.boundaryA.map((point, vertex) => [
      point[0] + (fan.boundaryB[vertex][0] - point[0]) * fraction,
      point[1] + (fan.boundaryB[vertex][1] - point[1]) * fraction
    ]);
  });
}

// Pointer-up / pointer-jitter artifacts: a hand-drawn stroke's trailing (or
// leading) captured point can double back on itself -- either because the
// pointerup event fires after a stray small movement, or because two
// consecutive samples land on (almost) the same pixel. appendPoint() only
// gates capture on a MINIMUM distance (App.tsx ~1981-1985), so it never
// filters out a point that reverses direction. Left untreated, that reversed
// point becomes a genuine vertex for the Catmull-Rom spline below, and since
// the spline is only uniformly-parameterized (not chordal/centripetal), a
// short or reversed final segment makes the interpolated curve overshoot
// past the intended endpoint and hook back -- a visible cusp/hairpin the
// user never drew. Trim such points before smoothing.
const JITTER_NEAR_DUPLICATE_PX = 0.75;
const JITTER_REVERSAL_DIST_PX = 3;
const JITTER_REVERSAL_ANGLE_DEG = 90;
const JITTER_EPSILON = 1e-6;

function pointDistance(a: [number, number], b: [number, number]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function turnAngleDegrees(a: [number, number], b: [number, number], c: [number, number]): number {
  const v1 = [b[0] - a[0], b[1] - a[1]];
  const v2 = [c[0] - b[0], c[1] - b[1]];
  const l1 = Math.hypot(v1[0], v1[1]);
  const l2 = Math.hypot(v2[0], v2[1]);
  if (l1 < 1e-9 || l2 < 1e-9) return 180;
  const dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2);
  return Math.acos(clamp(dot, -1, 1)) * 180 / Math.PI;
}

function trimStrokeJitter(points: [number, number][]): [number, number][] {
  let result = points;
  while (result.length > 3) {
    const n = result.length;
    const d = pointDistance(result[n - 1], result[n - 2]);
    if (d <= JITTER_NEAR_DUPLICATE_PX + JITTER_EPSILON) { result = result.slice(0, n - 1); continue; }
    if (d <= JITTER_REVERSAL_DIST_PX + JITTER_EPSILON && turnAngleDegrees(result[n - 3], result[n - 2], result[n - 1]) > JITTER_REVERSAL_ANGLE_DEG) {
      result = result.slice(0, n - 1);
      continue;
    }
    break;
  }
  while (result.length > 3) {
    const n = result.length;
    const d = pointDistance(result[0], result[1]);
    if (d <= JITTER_NEAR_DUPLICATE_PX + JITTER_EPSILON) { result = result.slice(1); continue; }
    if (d <= JITTER_REVERSAL_DIST_PX + JITTER_EPSILON && turnAngleDegrees(result[2], result[1], result[0]) > JITTER_REVERSAL_ANGLE_DEG) {
      result = result.slice(1);
      continue;
    }
    break;
  }
  return result;
}

// True APPROXIMATING smoothing stage, run before the Catmull-Rom pass below.
// Catmull-Rom is an interpolating spline -- it threads through every vertex
// it is given, so on its own it preserves all hand-drawn jitter verbatim.
// This convolves each of x(s) and y(s) (arc-length-parameterized along the
// jitter-trimmed captured polyline) with a Gaussian of the given sigma, in
// pixels of stroke arc length, so the resulting vertices approximate the
// drawn shape instead of tracing every wobble.
//
// Endpoint anchoring: each output point's convolution window is centered on
// that point's own index and radius-limited to the stroke's own length
// (never reaching past a single reflection of the far endpoint), using
// "odd" reflection padding -- pointAt(-j) = 2*p0 - pointAt(j), and the
// mirror image about the far endpoint for j >= n. For i = 0 that window is
// exactly [-(n-1), n-1], symmetric about s = 0; every weighted pair
// (+d, -d) then sums to 2*w*p0 (since pointAt(-j) + pointAt(j) = 2*p0 by
// construction), so the normalized result is exactly p0 regardless of
// sigma. The mirror argument is identical at the other end for i = n - 1.
// This keeps the smoothed curve anchored to the same (jitter-trimmed)
// stroke endpoints the endpoint-hook fix relies on.
function gaussianSmoothArcLength(points: [number, number][], sigmaPx: number): [number, number][] {
  const n = points.length;
  if (!(sigmaPx > 0) || n < 3) return points;
  const s: number[] = [0];
  for (let index = 1; index < n; index += 1) s.push(s[index - 1] + pointDistance(points[index - 1], points[index]));
  const total = s[n - 1];
  if (!(total > 0)) return points;

  const extent = n - 1;
  const pointAt = (index: number): [number, number] => {
    if (index >= 0 && index < n) return points[index];
    if (index < 0) {
      const mirror = points[clamp(-index, 0, n - 1)];
      const anchor = points[0];
      return [2 * anchor[0] - mirror[0], 2 * anchor[1] - mirror[1]];
    }
    const mirror = points[clamp(2 * extent - index, 0, n - 1)];
    const anchor = points[n - 1];
    return [2 * anchor[0] - mirror[0], 2 * anchor[1] - mirror[1]];
  };
  const arcAt = (index: number): number => {
    if (index >= 0 && index < n) return s[index];
    if (index < 0) return -s[clamp(-index, 0, n - 1)];
    return 2 * total - s[clamp(2 * extent - index, 0, n - 1)];
  };

  const twoSigmaSq = 2 * sigmaPx * sigmaPx;
  return points.map((_, i) => {
    const si = s[i];
    let sumW = 0;
    let sumX = 0;
    let sumY = 0;
    for (let j = i - extent; j <= i + extent; j += 1) {
      const d = si - arcAt(j);
      const w = Math.exp(-(d * d) / twoSigmaSq);
      if (w < 1e-6) continue;
      const p = pointAt(j);
      sumW += w;
      sumX += w * p[0];
      sumY += w * p[1];
    }
    if (!(sumW > 0)) return points[i];
    return [sumX / sumW, sumY / sumW];
  });
}

// Shared core of the smoothing pipeline: gaussian-smooths `points` (sigma in
// the SAME 2D units as the points - pixels or arcsec, whichever the caller
// is working in), then fits a Catmull-Rom spline through the smoothed
// vertices and resamples it at ~`resampleStep` unit spacing (arc length, same
// units). Used both by smoothResampleSlit (drawn-panel pixel space, called
// with resampleStep=1 to reproduce the pre-refactor "~1px" density exactly)
// and by resmoothCurveFromRaw (arcsec space, for the per-slit/per-fan
// re-smooth controls - see SlitDefinition.rawCurveArcsec). `clampBounds`, if
// given, clamps every Catmull-Rom sample into [0, width-1] x [0, height-1] -
// only meaningful in drawn-panel pixel space (keeps the curve inside the
// source image against spline overshoot near the endpoints); the arcsec
// re-smooth path omits it; see resmoothCurveFromRaw's docstring for why that
// is safe to skip. Returns points in the same units/space as the input,
// unprojected - callers apply their own world projection/orientation.
function smoothAndResampleCurve(
  points: [number, number][],
  sigma: number,
  resampleStep: number,
  clampBounds?: { width: number; height: number }
): [number, number][] {
  const smoothed = gaussianSmoothArcLength(points, sigma);
  const dense: [number, number][] = [];
  // Phantom control points beyond the real endpoints use reflection (mirroring
  // the neighboring point through the endpoint) rather than clamped
  // duplication, giving the standard "clamped end tangent" for Catmull-Rom
  // (matches the interior central-difference convention) instead of an
  // underweighted duplicate-point tangent at the two ends of the stroke.
  const valueAt = (index: number): [number, number] => {
    if (index < 0) {
      const p0 = smoothed[0];
      const p1 = smoothed[1] ?? smoothed[0];
      return [2 * p0[0] - p1[0], 2 * p0[1] - p1[1]];
    }
    if (index >= smoothed.length) {
      const pn = smoothed[smoothed.length - 1];
      const pnm1 = smoothed[smoothed.length - 2] ?? pn;
      return [2 * pn[0] - pnm1[0], 2 * pn[1] - pnm1[1]];
    }
    return smoothed[index];
  };
  const clampPoint = (x: number, y: number): [number, number] => clampBounds
    ? [clamp(x, 0, clampBounds.width - 1), clamp(y, 0, clampBounds.height - 1)]
    : [x, y];
  for (let segment = 0; segment < smoothed.length - 1; segment += 1) {
    const p0 = valueAt(segment - 1);
    const p1 = valueAt(segment);
    const p2 = valueAt(segment + 1);
    const p3 = valueAt(segment + 2);
    const samples = Math.max(4, Math.ceil(Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 2));
    for (let sample = 0; sample < samples; sample += 1) {
      const t = sample / samples;
      const t2 = t * t;
      const t3 = t2 * t;
      dense.push(clampPoint(
        0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3),
        0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
      ));
    }
  }
  dense.push(smoothed.at(-1)!);
  const cumulative = [0];
  for (let index = 1; index < dense.length; index += 1) {
    cumulative.push(cumulative[index - 1] + Math.hypot(dense[index][0] - dense[index - 1][0], dense[index][1] - dense[index - 1][1]));
  }
  const total = cumulative.at(-1) ?? 0;
  if (!(total > 0)) return [];
  const step = Number.isFinite(resampleStep) && resampleStep > 0 ? resampleStep : 1;
  const result: [number, number][] = [];
  for (let distance = 0; distance <= total; distance += step) {
    let upper = cumulative.findIndex((value) => value >= distance);
    if (upper <= 0) upper = 1;
    const lower = upper - 1;
    const span = Math.max(1e-9, cumulative[upper] - cumulative[lower]);
    const fraction = clamp((distance - cumulative[lower]) / span, 0, 1);
    result.push([
      dense[lower][0] + (dense[upper][0] - dense[lower][0]) * fraction,
      dense[lower][1] + (dense[upper][1] - dense[lower][1]) * fraction
    ]);
  }
  const lastCovered = Math.floor(total / step) * step;
  if (result.length < 2 || Math.abs(total - lastCovered) > 1e-6) {
    result.push(dense.at(-1)!);
  }
  return result;
}

// Trims jitter from a raw hand-drawn stroke and projects it to arcsec via
// the same per-point affine smoothResampleSlit uses for its final output -
// but stops BEFORE gaussian smoothing/resampling. This is what gets stored
// as SlitDefinition.rawCurveArcsec/FanDefinition.rawBoundaryA/B, so a later
// re-smooth (resmoothCurveFromRaw) starts from the same trimmed input the
// original draw-time smoothing did.
function trimAndProjectStroke(rawPoints: [number, number][], transform: PanelTransform): [number, number][] {
  if (rawPoints.length < 2) return [];
  const trimmed = rawPoints.length > 3 ? trimStrokeJitter(rawPoints) : rawPoints;
  if (trimmed.length < 2) return [];
  return trimmed.map((point) => applyAffine(point, transform.pixelToWorldAffine, transform.worldOffset));
}

// Canonicalizes a world/arcsec curve's direction so its first point is the
// one nearer the lower-left corner of the given solar view - the same rule
// smoothResampleSlit applies at draw time (there expressed via canvas
// distance from the drawn panel's own lower-left; worldToCanvas is a
// uniform-scale, axis-monotonic map of world coordinates, so comparing
// canvas distance from that corner is exactly equivalent to comparing world
// distance from (solarView.xMin, solarView.yMin) - this is that same
// comparison done directly in world space, without needing a panel
// transform). Used both by smoothResampleSlit (refactored to call this) and
// by resmoothCurveFromRaw, which has no panel transform to work with after
// the fact.
function reorientLowerLeft(curveArcsec: [number, number][], solarView: SolarView): [number, number][] {
  if (curveArcsec.length < 2) return curveArcsec;
  const lowerLeft: [number, number] = [solarView.xMin, solarView.yMin];
  const first = curveArcsec[0];
  const last = curveArcsec.at(-1)!;
  return pointDistance(last, lowerLeft) < pointDistance(first, lowerLeft) ? [...curveArcsec].reverse() : curveArcsec;
}

function smoothResampleSlit(
  rawPoints: [number, number][],
  transform: PanelTransform,
  orientLowerLeft = true,
  smoothSigmaPx: number = DEFAULT_SLIT_SMOOTH_PX
): [number, number][] {
  if (rawPoints.length < 2) return [];
  const trimmed = rawPoints.length > 3 ? trimStrokeJitter(rawPoints) : rawPoints;
  if (trimmed.length < 2) return [];
  const densePixels = smoothAndResampleCurve(trimmed, smoothSigmaPx, 1, { width: transform.imageWidth, height: transform.imageHeight });
  if (densePixels.length < 1) return [];
  const result = densePixels.map((pixel) => applyAffine(pixel, transform.pixelToWorldAffine, transform.worldOffset));
  return orientLowerLeft ? reorientLowerLeft(result, transform.solarView) : result;
}

// Re-runs the gaussian-smooth + Catmull-Rom-resample stage directly in
// arcsec, from a slit or fan boundary's stored pre-smoothing raw stroke
// (rawCurveArcsec/rawBoundaryA/B) - the per-slit/per-fan "Smooth" controls'
// implementation. There is no drawn-panel pixel transform available after
// the fact (the raw stroke is stored already projected to arcsec - see
// trimAndProjectStroke), so this re-expresses `smoothPx` (still meant as
// "pixels of whichever panel captured the stroke") in arcsec via
// `scaleArcsecPerPx`, the drawn panel's OWN live pixel scale
// (affinePixelScaleArcsec of its current affine, looked up by the caller
// from the slit/fan's stored drawnPanel - not a frozen snapshot, consistent
// with how every other geometry helper here re-reads the current affine
// rather than a stashed copy; see resolveSlitGeometry). This is the "adapt
// cleanly" choice over storing the raw stroke in pixel space plus a frozen
// affine: gaussianSmoothArcLength's output is a weighted affine combination
// of the input points (weights sum to 1), and an affine map distributes over
// affine combinations, so smoothing directly in arcsec with sigma scaled by
// the same factor as the points gives the identical result smoothing in
// pixel space and then projecting would - PROVIDED the affine's per-axis
// scale is isotropic, which affinePixelScaleArcsec already assumes
// (averages the two axes) for every other use of pixel scale in this file.
// The resample spacing is likewise expressed in arcsec via the same factor,
// so re-smoothing at the slit's original smoothPx reproduces close to the
// original curveArcsec's sample density. One fidelity difference from
// smoothResampleSlit: there is no per-pixel image-bounds clamp here (would
// need the drawn source's pixel-space image bounds, which the arcsec-only
// storage doesn't retain) - harmless in practice, since Catmull-Rom only
// interpolates between real (already on-image) smoothed points, and any
// endpoint overshoot is a fraction of a source pixel.
function resmoothCurveFromRaw(
  rawCurveArcsec: [number, number][],
  smoothPx: number,
  scaleArcsecPerPx: number,
  solarView: SolarView
): [number, number][] {
  if (rawCurveArcsec.length < 2) return rawCurveArcsec;
  const scale = Number.isFinite(scaleArcsecPerPx) && scaleArcsecPerPx > 0 ? scaleArcsecPerPx : 1;
  const sigmaArcsec = Math.max(0, smoothPx) * scale;
  const resampled = smoothAndResampleCurve(rawCurveArcsec, sigmaArcsec, scale);
  return resampled.length >= 2 ? reorientLowerLeft(resampled, solarView) : rawCurveArcsec;
}

function reverseSlitMapResult<T extends SlitMapResult>(result: T): T {
  const distanceMax = result.distanceArcsec.at(-1) ?? 0;
  result.intensity.reverse();
  result.distanceArcsec.reverse();
  result.distanceArcsec.forEach((distance, index) => {
    result.distanceArcsec[index] = distanceMax - distance;
  });
  result.curveVerticesArcsec.reverse();
  return {
    ...result
  };
}

function reverseSlitResult(result: SlitResult): SlitResult {
  const additionalMaps = result.additionalMaps.map((map) => reverseSlitMapResult(map));
  return { ...reverseSlitMapResult(result), additionalMaps };
}

function slitResultMaps(result: SlitResult): SlitMapResult[] {
  return [result, ...result.additionalMaps];
}

function coerceSolarView(value: unknown, fallback: SolarView): SolarView {
  if (!value || typeof value !== "object") return fallback;
  const candidate = value as Partial<SolarView>;
  const values = [candidate.xMin, candidate.xMax, candidate.yMin, candidate.yMax];
  if (values.every((item) => typeof item === "number" && Number.isFinite(item))) {
    return normalizeSolarView(candidate as SolarView);
  }
  return fallback;
}

function coerceLayout(value: unknown): LayoutState {
  if (!value || typeof value !== "object") return DEFAULT_LAYOUT;
  const candidate = value as Partial<LayoutState>;
  return {
    railWidth: clamp(Number(candidate.railWidth ?? DEFAULT_LAYOUT.railWidth), 260, 520),
    spectrogramHeight: clamp(Number(candidate.spectrogramHeight ?? DEFAULT_LAYOUT.spectrogramHeight), 80, 320),
    slitLaneHeight: clamp(Number(candidate.slitLaneHeight ?? DEFAULT_LAYOUT.slitLaneHeight), MIN_SLIT_LANE_HEIGHT, 480),
    imageSplit: clamp(Number(candidate.imageSplit ?? DEFAULT_LAYOUT.imageSplit), 0.25, 0.75)
  };
}

function applyAffine(point: [number, number], affine: Affine, offset: [number, number] = [0, 0]): [number, number] {
  const x = point[0];
  const y = point[1];
  return [
    x * affine[0][0] + y * affine[1][0] + affine[2][0] + offset[0],
    x * affine[0][1] + y * affine[1][1] + affine[2][1] + offset[1]
  ];
}

function invertAffine(point: [number, number], affine: Affine, offset: [number, number] = [0, 0]): [number, number] {
  const a = affine[0][0];
  const b = affine[1][0];
  const c = affine[2][0] + offset[0];
  const d = affine[0][1];
  const e = affine[1][1];
  const f = affine[2][1] + offset[1];
  const det = a * e - b * d;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-10) return [NaN, NaN];
  const wx = point[0] - c;
  const wy = point[1] - f;
  return [
    (wx * e - b * wy) / det,
    (a * wy - d * wx) / det
  ];
}

function worldToCanvas(point: [number, number], t: PanelTransform): [number, number] {
  return [
    t.offsetX + (point[0] - t.solarView.xMin) * t.scale,
    t.offsetY + (t.solarView.yMax - point[1]) * t.scale
  ];
}

function canvasToWorld(point: [number, number], t: PanelTransform): [number, number] {
  return [
    t.solarView.xMin + (point[0] - t.offsetX) / t.scale,
    t.solarView.yMax - (point[1] - t.offsetY) / t.scale
  ];
}

function pixelToCanvas(point: [number, number], t: PanelTransform): [number, number] {
  return worldToCanvas(applyAffine(point, t.pixelToWorldAffine, t.worldOffset), t);
}

function canvasToPixel(point: [number, number], t: PanelTransform): [number, number] {
  return invertAffine(canvasToWorld(point, t), t.pixelToWorldAffine, t.worldOffset);
}

const MIN_SOLAR_ZOOM = 0.5;
const MAX_SOLAR_ZOOM = 40;

function zoomSolarView(
  view: SolarView,
  anchor: [number, number],
  factor: number,
  fitView: SolarView = DEFAULT_SOLAR_VIEW
): SolarView {
  const base = normalizeSolarView(fitView);
  const current = normalizeSolarView(view);
  const nextFactor = Number.isFinite(factor) && factor > 0 ? factor : 1;
  const baseWidth = Math.max(1e-12, base.xMax - base.xMin);
  const baseHeight = Math.max(1e-12, base.yMax - base.yMin);
  const currentWidth = Math.max(1e-12, current.xMax - current.xMin);
  const currentHeight = Math.max(1e-12, current.yMax - current.yMin);
  const currentZoom = Math.max(baseWidth / currentWidth, baseHeight / currentHeight);
  const nextZoom = clamp(currentZoom / nextFactor, MIN_SOLAR_ZOOM, MAX_SOLAR_ZOOM);
  const appliedFactor = currentZoom / nextZoom;
  return normalizeSolarView({
    xMin: anchor[0] + (current.xMin - anchor[0]) * appliedFactor,
    xMax: anchor[0] + (current.xMax - anchor[0]) * appliedFactor,
    yMin: anchor[1] + (current.yMin - anchor[1]) * appliedFactor,
    yMax: anchor[1] + (current.yMax - anchor[1]) * appliedFactor
  });
}

function panSolarView(view: SolarView, dx: number, dy: number, scale: number): SolarView {
  const xShift = -dx / scale;
  const yShift = dy / scale;
  return {
    xMin: view.xMin + xShift,
    xMax: view.xMax + xShift,
    yMin: view.yMin + yShift,
    yMax: view.yMax + yShift
  };
}

function expandSolarViewToAspect(solarView: SolarView, viewportAspect: number): SolarView {
  const view = normalizeSolarView(solarView);
  const width = view.xMax - view.xMin;
  const height = view.yMax - view.yMin;
  const safeAspect = Number.isFinite(viewportAspect) && viewportAspect > 0 ? viewportAspect : width / height;
  const viewAspect = width / height;
  if (viewAspect > safeAspect) {
    const expandedHeight = width / safeAspect;
    const center = (view.yMin + view.yMax) / 2;
    return { ...view, yMin: center - expandedHeight / 2, yMax: center + expandedHeight / 2 };
  }
  if (viewAspect < safeAspect) {
    const expandedWidth = height * safeAspect;
    const center = (view.xMin + view.xMax) / 2;
    return { ...view, xMin: center - expandedWidth / 2, xMax: center + expandedWidth / 2 };
  }
  return view;
}

function computePanelTransform(
  rect: DOMRect,
  shape: [number, number],
  solarView: SolarView,
  pixelToWorldAffine: Affine,
  worldOffset: [number, number]
): PanelTransform {
  const plotHeight = Math.max(1, rect.height);
  const plotWidth = Math.max(1, rect.width);
  const view = expandSolarViewToAspect(solarView, plotWidth / plotHeight);
  const scale = plotWidth / Math.max(1e-6, view.xMax - view.xMin);
  const displayWidth = plotWidth;
  const displayHeight = plotHeight;
  return {
    scale,
    offsetX: 0,
    offsetY: 0,
    plotWidth,
    plotHeight,
    imageWidth: shape[1],
    imageHeight: shape[0],
    solarView: view,
    pixelToWorldAffine,
    worldOffset
  };
}

function drawImageWithWcs(ctx: CanvasRenderingContext2D, image: HTMLImageElement | ImageBitmap, t: PanelTransform) {
  const affine = t.pixelToWorldAffine;
  const ny = t.imageHeight;
  const sourceWidth = image instanceof HTMLImageElement ? image.naturalWidth : image.width;
  const sourceHeight = image instanceof HTMLImageElement ? image.naturalHeight : image.height;
  const sx = (t.imageWidth - 1) / Math.max(1, sourceWidth);
  const sy = (t.imageHeight - 1) / Math.max(1, sourceHeight);
  const wx0 = affine[1][0] * (ny - 1) + affine[2][0] + t.worldOffset[0];
  const wy0 = affine[1][1] * (ny - 1) + affine[2][1] + t.worldOffset[1];
  ctx.save();
  ctx.transform(
    t.scale * affine[0][0] * sx,
    -t.scale * affine[0][1] * sx,
    -t.scale * affine[1][0] * sy,
    t.scale * affine[1][1] * sy,
    t.offsetX + t.scale * (wx0 - t.solarView.xMin),
    t.offsetY + t.scale * (t.solarView.yMax - wy0)
  );
  ctx.drawImage(image, 0, 0);
  ctx.restore();
}

function isContourGeometry(data: FrameData): data is ContourGeometry {
  return !(data instanceof ImageBitmap) && Array.isArray(data.bands);
}

function contourBandColor(cmap: ContourColormap, frequency: number, frequencies: number[]): string {
  const [minimum, maximum] = frequencyBounds(frequencies);
  const fraction = maximum > minimum ? clamp((frequency - minimum) / (maximum - minimum), 0, 1) : 0;
  return sampleColormap(cmap, fraction * 1023, 1024, "frequency");
}

function drawContourGeometry(
  ctx: CanvasRenderingContext2D,
  geometry: ContourGeometry,
  transform: PanelTransform,
  layer: LayerState,
  frequencies: number[],
  offsets: ChannelOffsets,
  globalOffset: [number, number],
  selectedChannels: number[],
  selectionActive: boolean,
  selectionPreview: [number, number]
): void {
  const selected = new Set(selectedChannels);
  const alpha = clamp(numberValue(layer.contourOpacity, 0.35), 0, 1);
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const band of geometry.bands) {
    const channel = Number(band.channel);
    if (!Number.isInteger(channel) || offsets.masked[channel]) continue;
    const isSelected = selectionActive && selected.has(channel);
    const preview = isSelected ? selectionPreview : [0, 0];
    const bandTransform: PanelTransform = {
      ...transform,
      worldOffset: [
        transform.worldOffset[0] + globalOffset[0] + (offsets.dx[channel] ?? 0) + preview[0],
        transform.worldOffset[1] + globalOffset[1] + (offsets.dy[channel] ?? 0) + preview[1]
      ]
    };
    ctx.strokeStyle = contourBandColor(layer.contourCmap, Number(band.freqGhz), frequencies);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.globalAlpha = alpha * (selectionActive && !isSelected ? 0.35 : 1);
    ctx.lineWidth = isSelected ? 2.25 : 1.25;
    for (const polyline of band.polylines) {
      if (polyline.length < 3) continue;
      ctx.beginPath();
      const first = pixelToCanvas(polyline[0], bandTransform);
      ctx.moveTo(first[0], first[1]);
      for (const point of polyline.slice(1)) {
        const canvasPoint = pixelToCanvas(point, bandTransform);
        ctx.lineTo(canvasPoint[0], canvasPoint[1]);
      }
      const last = polyline[polyline.length - 1];
      const closed = Math.hypot(polyline[0][0] - last[0], polyline[0][1] - last[1]) < 3;
      if (layer.contourFilled && closed) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();
    }
  }
  ctx.restore();
}

function appendPoint(points: [number, number][], point: [number, number], minDistance = 2): [number, number][] {
  const last = points[points.length - 1];
  if (last && Math.hypot(last[0] - point[0], last[1] - point[1]) < minDistance) return points;
  return [...points, point];
}

type RecordingPlacement = {
  surface: RecordingSurface;
  canvas: HTMLCanvasElement;
  headerElement: HTMLElement | null;
  sourceWidth: number;
  sourceHeight: number;
  x: number;
  y: number;
  width: number;
  height: number;
};
type RecordingGeometry = {
  bounds: { width: number; height: number };
  placements: Array<{ surface: RecordingSurface; x: number; y: number; width: number; height: number }>;
  headerStrips: Array<{ surface: CaptureSurface; x: number; y: number; width: number; height: number }>;
  divider: { x: number; y: number; width: number; height: number } | null;
};

function createRecordingComposite(
  source: RecordingSource,
  resolution: RecordingResolution,
  canvases: Record<CaptureSurface, HTMLCanvasElement | null>
): { canvas: HTMLCanvasElement; draw: (timestamp: string | null) => void; geometry: RecordingGeometry; note: string } {
  const required = source === "left" ? ["left"] as CaptureSurface[]
    : source === "right" ? ["right"] as CaptureSurface[]
      : source === "both" ? ["left", "right"] as CaptureSurface[]
        : ["spectrogram", "left", "right"] as CaptureSurface[];
  const measured: Array<{
    surface: RecordingSurface;
    canvas: HTMLCanvasElement;
    rect: DOMRect;
    layoutRect: { left: number; top: number; right: number; bottom: number };
    pixelRatio: number;
  }> = required.map((surface) => {
    const canvas = canvases[surface];
    if (!canvas) throw new Error(`The ${surface} canvas is unavailable.`);
    const rect = canvas.getBoundingClientRect();
    const panelRect = canvas.closest(".image-panel")?.getBoundingClientRect() ?? rect;
    return {
      surface,
      canvas,
      rect,
      layoutRect: {
        left: panelRect.left,
        top: rect.top,
        right: panelRect.right,
        bottom: rect.bottom
      },
      pixelRatio: Math.max(1, canvas.width / Math.max(1, rect.width || canvas.width))
    };
  });
  if (source === "workspace") {
    const gutter = canvases.spectrogram?.closest(".spectrogram-body")?.querySelector<HTMLCanvasElement>(".spectrogram-channel-gutter");
    if (gutter) {
      const rect = gutter.getBoundingClientRect();
      measured.push({
        surface: "spectrogram-gutter",
        canvas: gutter,
        rect,
        layoutRect: { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom },
        pixelRatio: Math.max(1, gutter.width / Math.max(1, rect.width || gutter.width))
      });
    }
  }
  const contentBounds = measured.reduce((bounds, item) => ({
    left: Math.min(bounds.left, item.layoutRect.left),
    top: Math.min(bounds.top, item.layoutRect.top),
    right: Math.max(bounds.right, item.layoutRect.right),
    bottom: Math.max(bounds.bottom, item.layoutRect.bottom)
  }), { left: Infinity, top: Infinity, right: -Infinity, bottom: -Infinity });
  const workspaceRect = source === "workspace" ? measured[0].canvas.closest(".workspace")?.getBoundingClientRect() : null;
  const bounds = workspaceRect && workspaceRect.width > 0 && workspaceRect.height > 0
    ? { left: workspaceRect.left, top: workspaceRect.top, right: workspaceRect.right, bottom: workspaceRect.bottom }
    : contentBounds;
  const logicalWidth = Math.max(1, bounds.right - bounds.left);
  const logicalHeight = Math.max(1, bounds.bottom - bounds.top);
  const placements: RecordingPlacement[] = measured.map((item) => ({
    surface: item.surface,
    canvas: item.canvas,
    headerElement: item.surface === "spectrogram-gutter" ? null : item.canvas.closest(".image-panel")?.querySelector<HTMLElement>("header") ?? null,
    sourceWidth: item.canvas.width,
    sourceHeight: item.canvas.height,
    x: item.rect.left - bounds.left,
    y: item.rect.top - bounds.top,
    width: Math.max(1, item.rect.width || item.canvas.width),
    height: Math.max(1, item.rect.height || item.canvas.height)
  }));
  const dividerElement = required.includes("left") && required.includes("right")
    ? canvases.left?.closest(".panel-grid")?.querySelector<HTMLElement>(".image-resizer")
    : null;
  const dividerRect = dividerElement?.getBoundingClientRect();
  const divider = dividerElement && dividerRect ? {
    x: dividerRect.left - bounds.left,
    y: dividerRect.top - bounds.top,
    width: dividerRect.width,
    height: dividerRect.height,
    color: window.getComputedStyle(dividerElement).backgroundColor
  } : null;
  let pixelRatio = Math.max(...measured.map((item) => item.pixelRatio)) * (resolution === "2x" ? 2 : 1);
  const rawOutputWidth = logicalWidth * pixelRatio;
  const rawOutputHeight = logicalHeight * pixelRatio;
  let note = "";
  if (rawOutputWidth > RECORDER_HARD_MAX_DIMENSION || rawOutputHeight > RECORDER_HARD_MAX_DIMENSION) {
    // The composite would exceed the shared canvas/GPU/encoder dimension ceiling (e.g. a wide workspace
    // capture at 2x). Scale the whole render down rather than letting captureStream/MediaRecorder fail.
    pixelRatio *= RECORDER_HARD_MAX_DIMENSION / Math.max(rawOutputWidth, rawOutputHeight);
    note = `downscaled composite to fit the ${RECORDER_HARD_MAX_DIMENSION}px encoder/canvas limit`;
  }
  const output = document.createElement("canvas");
  output.width = Math.max(2, Math.floor(logicalWidth * pixelRatio / 2) * 2);
  output.height = Math.max(2, Math.floor(logicalHeight * pixelRatio / 2) * 2);
  const context = output.getContext("2d", { alpha: false });
  if (!context) throw new Error("Could not create the recording canvas.");
  const headerHeight = 24;
  const headerPlacements = source === "both" || source === "workspace"
    ? placements.filter((placement): placement is RecordingPlacement & { surface: CaptureSurface } => placement.surface !== "spectrogram-gutter")
    : [];
  const headerText = (placement: RecordingPlacement) => {
    const title = placement.headerElement?.querySelector<HTMLElement>(".image-panel-title, span")?.textContent?.trim() ?? "";
    const rawTimestamp = placement.headerElement?.querySelector<HTMLElement>(".spectrogram-tool-state, small")?.textContent?.trim() ?? "";
    return { title, timestamp: placement.surface === "spectrogram" ? rawTimestamp.split(" · ")[0] : rawTimestamp };
  };
  const draw = (timestamp: string | null) => {
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.fillStyle = "#050607";
    context.fillRect(0, 0, output.width, output.height);
    if (divider) {
      context.fillStyle = divider.color;
      context.fillRect(
        Math.round(divider.x * pixelRatio),
        Math.round(divider.y * pixelRatio),
        Math.round(divider.width * pixelRatio),
        Math.round(divider.height * pixelRatio)
      );
    }
    context.imageSmoothingQuality = "high";
    placements.forEach((placement) => {
      const destWidth = Math.round(placement.width * pixelRatio);
      const destHeight = Math.round(placement.height * pixelRatio);
      // Each source canvas already carries its own backing-store resolution (placement.sourceWidth/Height =
      // canvas.width/height, i.e. devicePixelRatio-scaled). When the composite's pixelRatio matches that
      // source's own ratio exactly (the common case in "native" mode), this blit is a true 1:1 pixel copy -
      // resampling it would only soften it, so smoothing is off. It's only turned on when the blit is an
      // actual up/downscale (e.g. "2x" mode, or a surface whose own backing ratio differs from the
      // composite's, such as the spectrogram gutter on a >2x display where its DPR is capped elsewhere).
      const isNativeBlit = Math.abs(destWidth - placement.sourceWidth) <= 1 && Math.abs(destHeight - placement.sourceHeight) <= 1;
      context.imageSmoothingEnabled = !isNativeBlit;
      context.drawImage(
        placement.canvas,
        0,
        0,
        placement.sourceWidth,
        placement.sourceHeight,
        Math.round(placement.x * pixelRatio),
        Math.round(placement.y * pixelRatio),
        destWidth,
        destHeight
      );
    });
    headerPlacements.forEach((placement) => {
      const text = headerText(placement);
      const x = Math.round(placement.x * pixelRatio);
      const y = Math.round(placement.y * pixelRatio);
      const width = Math.round(placement.width * pixelRatio);
      const height = Math.round(headerHeight * pixelRatio);
      const padding = Math.max(6, Math.round(7 * pixelRatio));
      context.fillStyle = "rgba(5, 6, 7, 0.78)";
      context.fillRect(x, y, width, height);
      context.textBaseline = "middle";
      context.fillStyle = "rgba(231, 233, 235, 0.94)";
      context.font = `600 ${Math.max(11, Math.round(12 * pixelRatio))}px Inter, ui-sans-serif, sans-serif`;
      context.textAlign = "left";
      const timestampFont = `${Math.max(10, Math.round(11 * pixelRatio))}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
      context.save();
      context.font = timestampFont;
      const timestampWidth = context.measureText(text.timestamp).width;
      context.restore();
      context.save();
      context.beginPath();
      context.rect(x + padding, y, Math.max(1, width - timestampWidth - padding * 3), height);
      context.clip();
      context.fillText(text.title, x + padding, y + height / 2);
      context.restore();
      context.font = timestampFont;
      context.textAlign = "right";
      context.fillStyle = "rgba(183, 190, 199, 0.94)";
      context.fillText(text.timestamp, x + width - padding, y + height / 2);
    });
    if (!timestamp) return;
    const padding = Math.max(4, Math.round(5 * pixelRatio));
    const inset = Math.max(7, Math.round(8 * pixelRatio));
    context.font = `${Math.max(10, Math.round(11 * pixelRatio))}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
    context.textAlign = "left";
    context.textBaseline = "bottom";
    const metrics = context.measureText(timestamp);
    const boxHeight = Math.ceil((metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent) || 12 * pixelRatio) + padding * 2;
    const boxWidth = Math.ceil(metrics.width) + padding * 2;
    context.fillStyle = "rgba(0, 0, 0, 0.58)";
    context.fillRect(inset, output.height - inset - boxHeight, boxWidth, boxHeight);
    context.fillStyle = "rgba(255, 255, 255, 0.9)";
    context.fillText(timestamp, inset + padding, output.height - inset - padding);
  };
  return {
    canvas: output,
    draw,
    geometry: {
      bounds: { width: logicalWidth, height: logicalHeight },
      placements: placements.map(({ surface, x, y, width, height }) => ({ surface, x, y, width, height })),
      headerStrips: headerPlacements.map(({ surface, x, y, width }) => ({ surface, x, y, width, height: headerHeight })),
      divider: divider ? { x: divider.x, y: divider.y, width: divider.width, height: divider.height } : null
    },
    note
  };
}

function App() {
  const [meta, setMeta] = useState<SessionMeta | null>(null);
  const [prewarmStatus, setPrewarmStatus] = useState<PrewarmStatus | null>(null);
  const [progressOperations, setProgressOperations] = useState<ProgressOperation[]>([]);
  const [warmCacheStatus, setWarmCacheStatus] = useState<WarmCacheStatus>({
    active: false,
    done: 0,
    total: 0,
    retained: 0,
    requestedFrames: 0,
    targetFrames: 0,
    limited: false,
    startedAt: 0
  });
  const [warmCacheRecord, setWarmCacheRecord] = useState<WarmCacheRecord | null>(null);
  const warmCacheControllerRef = useRef<AbortController | null>(null);
  const [message, setMessage] = useState("Load sample data or a source manifest to begin.");
  const [datasetName, setDatasetName] = useState(INITIAL_MANIFEST_NAME ?? "sample");
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ id: number; message: string } | null>(null);
  const [dirty, setDirty] = useState(false);
  const baselineSignatureRef = useRef("");
  const baselinePendingRef = useRef(false);
  const sessionRestoreRef = useRef<Promise<string> | null>(null);
  const restoredSessionIdRef = useRef("");

  function reportError(error: unknown, fallback = "Operation failed") {
    const detail = error instanceof Error ? error.message : String(error);
    setToast({ id: Date.now(), message: detail || fallback });
    setMessage(fallback);
  }

  function confirmDatasetReplace(): boolean {
    return !dirty || window.confirm("Discard unsaved analysis and load another dataset?");
  }
  const [playing, setPlaying] = useState(false);
  const [solarView, setSolarView] = useState<SolarView>(DEFAULT_SOLAR_VIEW);
  const [layout, setLayout] = useState<LayoutState>(DEFAULT_LAYOUT);
  const [spaceDown, setSpaceDown] = useState(false);
  const pointerOverImageRef = useRef(false);
  const layoutDragRef = useRef<LayoutDragState | null>(null);
  const workspaceRef = useRef<HTMLElement | null>(null);
  const panelGridRef = useRef<HTMLDivElement | null>(null);
  const jsonInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedSourceId, setSelectedSourceId] = useState("");
  const [alignmentOpen, setAlignmentOpen] = useState(false);
  const [spectrogramDisplayOpen, setSpectrogramDisplayOpen] = useState(false);
  const [channelOffsets, setChannelOffsets] = useState<ChannelOffsets>({ dx: [], dy: [], masked: [] });
  const [selectedChannels, setSelectedChannels] = useState<number[]>([]);
  const alignmentPostKeyRef = useRef("");
  const alignmentLoadedSessionRef = useRef("");
  const [sourceRoles, setSourceRoles] = useState<Record<string, SourceRole>>({});
  const [sourceDifferences, setSourceDifferences] = useState<Record<string, DifferenceState>>({});
  const [panelLayers, setPanelLayers] = useState<Record<PanelSlotId, LayerState[]>>({ left: [], right: [] });
  const [panelCompositions, setPanelCompositions] = useState<Record<PanelSlotId, PanelComposition>>({ left: { baseLayerId: "", overlayLayerIds: [] }, right: { baseLayerId: "", overlayLayerIds: [] } });
  const [panelFrameCaps, setPanelFrameCaps] = useState<Record<PanelSlotId, FrameRequestCap>>({
    left: { ...DEFAULT_FRAME_REQUEST_CAP },
    right: { ...DEFAULT_FRAME_REQUEST_CAP }
  });
  const contourLevelModeRef = useRef<Record<string, LayerState["contourLevelMode"]>>({});
  const [layerFrameStats, setLayerFrameStats] = useState<Record<string, { key: string; stats: FrameStats }>>({});
  const [pixelProbe, setPixelProbe] = useState<PixelProbe | null>(null);
  const [pixelProbeData, setPixelProbeData] = useState<PixelProbeResponse | null>(null);
  const [pixelProbeLoading, setPixelProbeLoading] = useState(false);
  const [pixelProbeError, setPixelProbeError] = useState("");
  const [pixelProbeRefresh, setPixelProbeRefresh] = useState(0);
  const [selectedLayerId, setSelectedLayerId] = useState<SelectedLayer | null>(null);
  const [collapsedSections, setCollapsedSections] = useState<string[]>([]);
  const [sourceDraft, setSourceDraft] = useState({ role: "context", label: "", format: "fits", path: "" });
  const [filePickerOpen, setFilePickerOpen] = useState(false);
  const [lastBrowsedDirectory, setLastBrowsedDirectory] = useState("");
  const [masterSourceId, setMasterSourceId] = useState("context");
  const [masterCursorMjd, setMasterCursorMjd] = useState<number | null>(null);
  const [masterStartMjd, setMasterStartMjd] = useState<number | null>(null);
  const [masterEndMjd, setMasterEndMjd] = useState<number | null>(null);
  const [masterWarning, setMasterWarning] = useState("");
  const [timeIndex, setTimeIndex] = useState(0);
  const [timeSliderMjd, setTimeSliderMjd] = useState(0);
  const [timeScrubbing, setTimeScrubbing] = useState(false);
  const [playbackFps, setPlaybackFps] = useState(DEFAULT_PLAYBACK_FPS);
  const [timeStepSeconds, setTimeStepSeconds] = useState(0);
  const [playbackBuffer, setPlaybackBuffer] = useState<PlaybackBufferState>({ ahead: 0, total: 0 });
  // Feature: motion resolution ladder. `motionResolutionActive` tracks
  // whether half-res identities are currently in effect; it flips on the
  // instant playback/scrubbing starts, and only flips back off after
  // MOTION_RESOLUTION_IDLE_MS of no motion (see the effect below).
  const [smoothPlaybackEnabled, setSmoothPlaybackEnabled] = useState(true);
  const [motionResolutionActive, setMotionResolutionActive] = useState(false);
  const motionIdleTimerRef = useRef(0);
  // Feature: cache coverage bar toggle (rendering lives in SpectrogramPanel).
  const [showCacheCoverage, setShowCacheCoverage] = useState(true);
  // Feature: decoded-frame bitmap cache byte budget, user-scrubbable in the
  // spectrogram display card. frameCacheGb mirrors frameScheduler's module-
  // level budget (see setFrameCacheBudgetBytes); every commit pushes the new
  // ceiling into the scheduler, which evicts immediately if it shrank.
  const [frameCacheGb, setFrameCacheGbState] = useState(() => getFrameCacheBudgetBytes() / 1024 ** 3);
  function applyFrameCacheGb(gb: number) {
    const clamped = clamp(gb, FRAME_CACHE_GB_MIN, FRAME_CACHE_GB_MAX);
    setFrameCacheGbState(clamped);
    setFrameCacheBudgetBytes(clamped * 1024 ** 3);
  }
  const timeCommitRafRef = useRef(0);
  const timeSliderMjdRef = useRef(0);
  const scrubDirectionRef = useRef(1);
  const lastCursorMjdRef = useRef(0);
  const playbackDirectionRef = useRef(1);
  const playbackPrefetchGroupsRef = useRef<ScheduledFrameRequest[][]>([]);
  const playbackImminentRequestsRef = useRef<ScheduledFrameRequest[]>([]);
  const playbackPrefetchProgressRef = useRef<() => void>(() => undefined);
  const [recordingOptionsOpen, setRecordingOptionsOpen] = useState(false);
  const [recordingOptions, setRecordingOptions] = useState<RecordingOptions>(DEFAULT_RECORDING_OPTIONS);
  const [recordingStatus, setRecordingStatus] = useState<RecordingStatus | null>(null);
  const [recordingDrawNonce, setRecordingDrawNonce] = useState(0);
  const recordingControllerRef = useRef<AbortController | null>(null);
  const captureCanvasRefs = useRef<Record<CaptureSurface, HTMLCanvasElement | null>>({ spectrogram: null, left: null, right: null });
  const captureDrawSerialRef = useRef<Record<CaptureSurface, number>>({ spectrogram: 0, left: 0, right: 0 });
  const captureDrawListenersRef = useRef(new Set<(surface: CaptureSurface, mjd: number, serial: number) => void>());
  const [freqIndex, setFreqIndex] = useState(0);
  const [xOffset, setXOffset] = useState("7");
  const [yOffset, setYOffset] = useState("0");
  const [lassoEnabled, setLassoEnabled] = useState(false);
  const [channelLassoArmed, setChannelLassoArmed] = useState(false);
  const [roiWorld, setRoiWorld] = useState<[number, number][]>([]);
  const [roiAia, setRoiAia] = useState<[number, number][]>([]);
  const [roiEovsa, setRoiEovsa] = useState<[number, number][]>([]);
  const [correlationTarget, setCorrelationTarget] = useState<[number, number][]>([]);
  const [correlationCardOpen, setCorrelationCardOpen] = useState(false);
  const [correlationCardDismissed, setCorrelationCardDismissed] = useState(false);
  const [correlationFullRange, setCorrelationFullRange] = useState(false);
  const [correlationPinTicks, setCorrelationPinTicks] = useState(false);
  const [correlationFullHeightTicks, setCorrelationFullHeightTicks] = useState(false);
  const [correlationPeakDecel, setCorrelationPeakDecel] = useState(false);
  const [correlationHoverTrackId, setCorrelationHoverTrackId] = useState("");
  const [slits, setSlits] = useState<SlitDefinition[]>([]);
  const [selectedSlitId, setSelectedSlitId] = useState("");
  const [slitSourceId, setSlitSourceId] = useState("");
  const [slitDraftWidthArcsec, setSlitDraftWidthArcsec] = useState(DEFAULT_SLIT_WIDTH_ARCSEC);
  const [slitDraftSmoothPx, setSlitDraftSmoothPx] = useState(DEFAULT_SLIT_SMOOTH_PX);
  const [slitInspectorOpen, setSlitInspectorOpen] = useState(false);
  const [slitPinLane, setSlitPinLane] = useState(false);
  // "All slits" toggle beside Extract in the Slit inspector: when on,
  // Extract runs every visible slit in one shared-source-batched operation
  // (see extractAllSlits) instead of just the selected slit.
  const [slitExtractAllMode, setSlitExtractAllMode] = useState(false);
  const [slitDrawArmed, setSlitDrawArmed] = useState(false);
  const [slitExtracting, setSlitExtracting] = useState(false);
  const [slitResults, setSlitResults] = useState<Record<string, SlitResult>>({});
  const [fan, setFan] = useState<FanDefinition | null>(null);
  const [fanDrawStage, setFanDrawStage] = useState<0 | 1 | 2>(0);
  const [fanBoundaryDraft, setFanBoundaryDraft] = useState<{ curve: [number, number][]; inputVertexCount: number; rawCurveArcsec: [number, number][]; drawnPanel: PanelId } | null>(null);
  const [fanIntermediateCount, setFanIntermediateCount] = useState(3);
  const [selectedFanMember, setSelectedFanMember] = useState(0);
  // Single-stroke redraw of one existing fan boundary (distinct from the
  // two-stage fanDrawStage used to create a brand-new fan from scratch).
  const [fanRedrawTarget, setFanRedrawTarget] = useState<"A" | "B" | null>(null);
  const slitLaneCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [floatingCardSizes, setFloatingCardSizes] = useState<FloatingCardSizes>({});
  const [targetDrawArmed, setTargetDrawArmed] = useState(false);
  const [sadTracks, setSadTracks] = useState<SadTrack[]>([]);
  const [selectedTrackId, setSelectedTrackId] = useState("");
  const [trackingSourceId, setTrackingSourceId] = useState("");
  const [trackEditorOpen, setTrackEditorOpen] = useState(false);
  const [seedMode, setSeedMode] = useState(false);
  const [seedSuggestions, setSeedSuggestions] = useState<SeedSuggestion[]>([]);
  const [trackingDirection, setTrackingDirection] = useState<TrackDirection>("both");
  const [trackingActive, setTrackingActive] = useState(false);
  const trackingStopRequestedRef = useRef(false);
  const [selectedAnchorFrame, setSelectedAnchorFrame] = useState<number | null>(null);
  const trackHistoryRef = useRef<{ past: SadTrack[][]; future: SadTrack[][] }>({ past: [], future: [] });
  const [trackHistoryVersion, setTrackHistoryVersion] = useState(0);
  const [eovsaSources, setEovsaSources] = useState<EovsaSource[]>([]);
  const [aiaDisplay, setAiaDisplay] = useState<DisplayState>({ vmin: "0.5", vmax: "1.5", cmap: "gray", scale: "linear", radialGamma: 0 });
  const [eovsaDisplay, setEovsaDisplay] = useState<DisplayState>({ vmin: "-1000000", vmax: "5000000", cmap: "turbo", scale: "linear", radialGamma: 0 });
  const [spectrogramDisplay, setSpectrogramDisplay] = useState<DisplayState>({ vmin: "0.5", vmax: "150", cmap: "viridis", scale: "log", radialGamma: 0 });
  const [spectrogramNormalization, setSpectrogramNormalization] = useState<SpectrogramNormalization>("none");
  const [spectrogramFrequencyScale, setSpectrogramFrequencyScale] = useState<FrequencyScaleMode>("linear");
  const [spectrogramFrequencyInverted, setSpectrogramFrequencyInverted] = useState(false);
  const [spectrogramFrequencyRange, setSpectrogramFrequencyRange] = useState<FrequencyRangeState>({ min: "1.1", max: "18" });
  const [spectrogramTimeRange, setSpectrogramTimeRange] = useState<TimeRangeState>({ min: "", max: "" });

  const sessionId = meta?.sessionId ?? "";
  useEffect(() => {
    setSeedMode(false);
    setSeedSuggestions([]);
    setSelectedAnchorFrame(null);
    trackHistoryRef.current = { past: [], future: [] };
    setTrackHistoryVersion((value) => value + 1);
    recordingControllerRef.current?.abort();
    recordingControllerRef.current = null;
    setRecordingStatus(null);
    setRecordingOptionsOpen(false);
      setCorrelationCardOpen(false);
      setCorrelationCardDismissed(false);
      setCorrelationHoverTrackId("");
    setSlitDrawArmed(false);
    setFanDrawStage(0);
    setFanBoundaryDraft(null);
    setFanRedrawTarget(null);
    setSlitExtracting(false);
    setSlitResults({});
    warmCacheControllerRef.current?.abort();
    warmCacheControllerRef.current = null;
    setWarmCacheRecord(null);
    setWarmCacheStatus((current) => ({ ...current, active: false, done: 0, total: 0 }));
    contourLevelModeRef.current = {};
    if (alignmentLoadedSessionRef.current !== sessionId) {
      alignmentPostKeyRef.current = "";
      alignmentLoadedSessionRef.current = sessionId;
    }
    setLayerFrameStats({});
    if (restoredSessionIdRef.current && restoredSessionIdRef.current === sessionId) {
      restoredSessionIdRef.current = "";
      sessionRestoreRef.current = null;
    }
  }, [sessionId]);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const initial = meta?.prewarm ?? null;
    setPrewarmStatus(initial);
    setProgressOperations([]);
    if (!sessionId || (!loading && !initial?.active)) return () => undefined;

    const poll = async () => {
      try {
        const response = await apiJson<ProgressResponse>(`/api/sessions/${sessionId}/progress`);
        if (cancelled) return;
        const operations = response.operations.filter(isActiveProgressOperation);
        setProgressOperations(operations);
        setPrewarmStatus(response.prewarm);
        const prewarmActive = response.prewarm.active && response.prewarm.done < response.prewarm.total;
        if (loading || operations.length > 0 || prewarmActive) {
          timer = window.setTimeout(poll, 750);
        } else {
          setProgressOperations([]);
        }
      } catch {
        if (!cancelled) {
          setProgressOperations([]);
          setPrewarmStatus(null);
        }
      }
    };
    timer = window.setTimeout(poll, loading ? 100 : 750);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [sessionId, loading]);

  const sources = useMemo(() => sourcesForMeta(meta), [meta]);
  const allPanelLayers = useMemo(() => [...panelLayers.left, ...panelLayers.right], [panelLayers.left, panelLayers.right]);
  const dirtySignature = useMemo(() => meta ? JSON.stringify({
    sessionId,
    roiWorld,
    correlationTarget,
    roiAia,
    roiEovsa,
    sadTracks,
    slits,
    fan,
    eovsaSources,
    channelOffsets
  }) : "", [channelOffsets, correlationTarget, eovsaSources, fan, roiAia, roiEovsa, roiWorld, sadTracks, sessionId, slits]);
  const contextSource = useMemo(() => roleSource(sources, sourceRoles, "context"), [sourceRoles, sources]);
  const radioSource = useMemo(() => roleSource(sources, sourceRoles, "radio"), [sourceRoles, sources]);
  const spectrogramSource = useMemo(() => roleSource(sources, sourceRoles, "spectrogram"), [sourceRoles, sources]);
  const selectedSource = sources.find((source) => source.id === selectedSourceId) ?? sources[0];
  const contextSourceId = contextSource?.id ?? "context";
  const radioSourceId = radioSource?.id ?? "radio";
  const spectrogramSourceId = spectrogramSource?.id ?? "spectrogram";
  const radioFreqGhz = meta?.eovsa.freqGhz ?? radioSource?.freqGhz ?? [];
  const spwGroups = useMemo(() => deriveSpwGroups(radioFreqGhz), [radioFreqGhz]);
  const alignmentPalette = useMemo(() => {
    const contour = allPanelLayers.map((layer) => resolvedLayer(layer, allPanelLayers)).find((layer) => layer.kind === "contours");
    return contour?.contourCmap ?? "turbo";
  }, [allPanelLayers]);

  useEffect(() => {
    if (!sessionId || !radioFreqGhz.length || channelOffsets.dx.length !== radioFreqGhz.length) return undefined;
    const key = JSON.stringify(channelOffsets);
    if (key === alignmentPostKeyRef.current) return undefined;
    const timer = window.setTimeout(() => {
      void apiJson<{ channelOffsets: ChannelOffsets }>(
        `/api/sessions/${sessionId}/sources/${radioSourceId}/channel-offsets`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(channelOffsets)
        },
      ).then(() => {
        alignmentPostKeyRef.current = key;
      }).catch((error) => {
        if (!(error instanceof SessionRestoredError)) reportError(error, "Could not save channel offsets");
      });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [channelOffsets, radioFreqGhz.length, radioSourceId, sessionId]);

  useEffect(() => {
    if (!alignmentOpen) setChannelLassoArmed(false);
  }, [alignmentOpen]);
  const masterSource = sources.find((source) => source.id === masterSourceId) ?? contextSource;
  const masterTimes = useMemo(() => sourceTimeMjd(masterSource, meta, sourceRoles), [masterSource, meta, sourceRoles]);
  const masterFullBounds = timeBounds(masterTimes);
  const effectiveMasterMjd = masterCursorMjd ?? meta?.aia.timeMjd[timeIndex] ?? 0;
  const masterCursorIndex = closestIndex(masterTimes, effectiveMasterMjd);
  const masterMin = masterStartMjd ?? masterFullBounds[0];
  const masterMax = masterEndMjd ?? masterFullBounds[1];
  const minimumTimeStepSeconds = useMemo(() => sensibleCadenceSeconds(Math.min(
    ...sources
      .filter((source) => source.status !== "placeholder")
      .map((source) => nativeCadenceSeconds(sourceTimeMjd(source, meta, sourceRoles)))
      .filter((cadence) => cadence > 0 && Number.isFinite(cadence))
  )), [meta, sourceRoles, sources]);
  const timePosition = timeStepSeconds > 0
    ? timeStepGridPosition(masterMin, masterMax, effectiveMasterMjd, timeStepSeconds)
    : { index: masterCursorIndex, total: Math.max(1, masterTimes.length) };
  const timeCount = meta?.aia.times.length ?? 1;
  const radioTimeCount = meta?.eovsa.times.length ?? timeCount;
  const contextDifference = sourceDifferences[contextSourceId] ?? defaultDifference("context", meta?.defaults.diffSeconds ?? DEFAULT_DIFF_SECONDS, timeCount);
  const radioDifference = sourceDifferences[radioSourceId] ?? defaultDifference("radio", meta?.defaults.diffSeconds ?? DEFAULT_DIFF_SECONDS, radioTimeCount);
  const probedLayer = useMemo(() => {
    if (!pixelProbe) return null;
    const layers = panelLayers[pixelProbe.slot];
    const composition = compositionFor(layers, panelCompositions[pixelProbe.slot]?.baseLayerId, panelCompositions[pixelProbe.slot]?.overlayLayerIds);
    if (composition.baseLayerId !== pixelProbe.layerId) return null;
    const layer = layers.find((candidate) => candidate.id === pixelProbe.layerId);
    return layer && resolvedLayer(layer, allPanelLayers).kind === "image" ? resolvedLayer(layer, allPanelLayers) : null;
  }, [allPanelLayers, panelCompositions, panelLayers, pixelProbe]);
  const currentMjd = effectiveMasterMjd;
  const eovsaTimeIndex = meta ? closestIndex(meta.eovsa.timeMjd, currentMjd) : 0;
  const eovsaAvailable = meta ? meta.eovsa.timeMjd.length > 0 : false;
  const eovsaResolvedAvailable = meta && eovsaAvailable
    ? Math.abs(meta.eovsa.timeMjd[eovsaTimeIndex] - currentMjd) <= nativeCadenceSeconds(meta.eovsa.timeMjd) / 2 / 86400
    : false;
  const aiaTimestamp = meta ? mjdToUtc(currentMjd) : "--";
  const eovsaTimestamp = meta
    ? (eovsaResolvedAvailable
      ? formatResolvedTimestamp(meta.eovsa.timeMjd[eovsaTimeIndex], (meta.eovsa.timeMjd[eovsaTimeIndex] - currentMjd) * 86400, "--")
      : "No EOVSA data · unavailable")
    : "--";
  const timeMin = meta ? masterMin : 0;
  const timeMax = meta ? masterMax : 0;
  const selectedTrack = sadTracks.find((track) => track.id === selectedTrackId) ?? null;
  const leftBaseLayer = useMemo(() => {
    const composition = compositionFor(panelLayers.left, panelCompositions.left?.baseLayerId, panelCompositions.left?.overlayLayerIds);
    const layer = panelLayers.left.find((candidate) => candidate.id === composition.baseLayerId);
    return layer ? resolvedLayer(layer, allPanelLayers) : null;
  }, [allPanelLayers, panelCompositions.left, panelLayers.left]);
  const rightBaseLayer = useMemo(() => {
    const composition = compositionFor(panelLayers.right, panelCompositions.right?.baseLayerId, panelCompositions.right?.overlayLayerIds);
    const layer = panelLayers.right.find((candidate) => candidate.id === composition.baseLayerId);
    return layer ? resolvedLayer(layer, allPanelLayers) : null;
  }, [allPanelLayers, panelCompositions.right, panelLayers.right]);
  const trackingSourceOptions = useMemo(() => {
    const options = new Map<string, TrackingSourceOption>();
    for (const slot of ["left", "right"] as PanelSlotId[]) {
      const composition = compositionFor(panelLayers[slot], panelCompositions[slot]?.baseLayerId, panelCompositions[slot]?.overlayLayerIds);
      const ordered = [
        panelLayers[slot].find((layer) => layer.id === composition.baseLayerId),
        ...panelLayers[slot].filter((layer) => layer.id !== composition.baseLayerId)
      ].filter((layer): layer is LayerState => Boolean(layer));
      for (const entry of ordered) {
        const layer = resolvedLayer(entry, allPanelLayers);
        if (layer.kind !== "image" || !["context", "radio"].includes(layer.sourceRoleSnapshot)) continue;
        const existing = options.get(layer.sourceId);
        if (existing) {
          if (!existing.sides.includes(slot)) existing.sides.push(slot);
        } else {
          options.set(layer.sourceId, { sourceId: layer.sourceId, label: layer.label, sides: [slot], layer });
        }
      }
    }
    return [...options.values()];
  }, [allPanelLayers, panelCompositions, panelLayers]);
  const slitSourceOptions = useMemo(() => {
    const options = new Map<string, SlitSourceOption>();
    const panelEntries = (kind: "image" | "contours") => {
      for (const slot of ["left", "right"] as PanelSlotId[]) {
        const composition = compositionFor(panelLayers[slot], panelCompositions[slot]?.baseLayerId, panelCompositions[slot]?.overlayLayerIds);
        const ordered = [
          panelLayers[slot].find((layer) => layer.id === composition.baseLayerId),
          ...composition.overlayLayerIds.map((id) => panelLayers[slot].find((layer) => layer.id === id))
        ].filter((layer): layer is LayerState => Boolean(layer));
        for (const entry of ordered) {
          const layer = resolvedLayer(entry, allPanelLayers);
          if (layer.kind !== kind || !["context", "radio"].includes(layer.sourceRoleSnapshot)) continue;
          const existing = options.get(layer.sourceId);
          if (existing) {
            if (!existing.sides.includes(slot)) existing.sides.push(slot);
            continue;
          }
          const source = sources.find((candidate) => candidate.id === layer.sourceId);
          options.set(layer.sourceId, {
            sourceId: layer.sourceId,
            label: kind === "contours" ? `${source?.label ?? layer.label} - contours` : layer.label,
            sides: [slot],
            layer,
            bindingKind: kind
          });
        }
      }
    };
    // A displayed image is the most specific contract; contours are eligible
    // only when that radio source is otherwise absent as an image layer.
    panelEntries("image");
    panelEntries("contours");
    for (const source of sources) {
      const role = sourceRole(source, sourceRoles);
      if (options.has(source.id) || !["context", "radio"].includes(role) || source.capabilities?.render === false) continue;
      const times = sourceTimeMjd(source, meta, sourceRoles);
      const sourceDifference = sourceDifferences[source.id]
        ?? defaultDifference(role, meta?.defaults.diffSeconds ?? DEFAULT_DIFF_SECONDS, times.length);
      const display = role === "radio"
        ? { ...eovsaDisplay, radialGamma: 0 }
        : { ...aiaDisplay, radialGamma: 0 };
      const layer = defaultLayer(
        `slit-raw-${source.id}`,
        source.label,
        source.id,
        "image",
        role,
        display,
        { ...sourceDifference, operation: "none", reference: "previous" },
        role === "radio" ? freqIndex : 0
      );
      options.set(source.id, {
        sourceId: source.id,
        label: `${source.label} (not displayed - raw)`,
        sides: [],
        layer,
        bindingKind: "raw"
      });
    }
    return [...options.values()];
  }, [aiaDisplay, allPanelLayers, eovsaDisplay, freqIndex, meta, panelCompositions, panelLayers, sourceDifferences, sourceRoles, sources]);
  const trackingSourceOption = trackingSourceOptions.find((option) => option.sourceId === trackingSourceId) ?? null;
  const trackingAvailable = Boolean(meta && trackingSourceOption && [leftBaseLayer?.sourceId, rightBaseLayer?.sourceId].includes(trackingSourceOption.sourceId));
  const targetDrawingAvailable = Boolean(meta && leftBaseLayer?.sourceRoleSnapshot === "context");
  const allTrackSourcesAvailable = sadTracks.every((track) => trackingSourceOptions.some((option) => (
    option.sourceId === track.sourceId && [leftBaseLayer?.sourceId, rightBaseLayer?.sourceId].includes(option.sourceId)
  )));
  const selectedTrackSourceId = selectedTrack?.sourceId ?? trackingSourceOption?.sourceId ?? contextSourceId;
  const selectedTrackSource = sources.find((source) => source.id === selectedTrackSourceId);
  const selectedTrackTimes = sourceTimeMjd(selectedTrackSource, meta, sourceRoles);
  const currentTrackFrame = closestIndex(selectedTrackTimes, currentMjd);
  const trackingSeedTimes = sourceTimeMjd(
    sources.find((source) => source.id === trackingSourceOption?.sourceId),
    meta,
    sourceRoles
  );
  const currentTrackingSeedFrame = closestIndex(trackingSeedTimes, currentMjd);
  const sourceAffine = (sourceId: string): Affine => sourceId === radioSourceId
    ? meta?.wcs.eovsa.pixelToWorldAffine ?? IDENTITY_AFFINE
    : meta?.wcs.aia.pixelToWorldAffine ?? IDENTITY_AFFINE;
  // Live per-axis-averaged pixel scale of whichever panel (aia/eovsa) a raw
  // stroke was captured on - see SlitDefinition.drawnPanel/resmoothCurveFromRaw.
  // Distinct from sourceAffine (keyed by bound sourceId): a slit's draw
  // panel and its extraction binding are independent choices, but both only
  // ever resolve to one of these same two underlying affines.
  const panelAffine = (panel: PanelId): Affine => panel === "eovsa"
    ? meta?.wcs.eovsa.pixelToWorldAffine ?? IDENTITY_AFFINE
    : meta?.wcs.aia.pixelToWorldAffine ?? IDENTITY_AFFINE;
  // Arcsec-per-pixel scale for a slit's BOUND SOURCE - used only to render
  // the muted "≈ N px (source)" hint next to the (now physical, arcsec)
  // Width control, and for the legacy px->arcsec migration in
  // normalizeSlits (via applyLoadedSession's own copy, not this closure -
  // see that migration's docstring for why). NOT used by drawSlitWorldPath
  // any more - the drawn swath is widthArcsec directly, independent of any
  // source's pixel scale (USER DESIGN DECISION - see SlitDefinition.widthArcsec).
  const slitWidthScaleArcsec = (sourceId: string): number => affinePixelScaleArcsec(sourceAffine(sourceId));
  // Largest image dimension of a BOUND SOURCE, in that source's own pixels
  // (its field of view edge to edge) - used by slitWidthArcsecBounds below
  // to compute the Width control's arcsec ceiling, and by the muted px hint.
  // Falls back to a generous constant while meta hasn't loaded a real shape
  // yet - see FALLBACK_SLIT_WIDTH_ARCSEC_MAX.
  const sourceMaxWidthPx = (sourceId: string): number => {
    const shape = sourceId === radioSourceId ? meta?.eovsa.shape : meta?.aia.shape;
    return shape && shape.length === 2 && shape[0] > 0 && shape[1] > 0
      ? Math.round(Math.max(shape[0], shape[1]))
      : 0;
  };
  // Physical bounds for the Width [″] control (FEATURE 2 / USER DESIGN
  // DECISION: width is a fixed arcsec quantity, not bound-source pixels).
  // GLOBAL across both underlying affines (aia/eovsa - see sourceAffine),
  // not keyed by the slit's current binding, so re-binding a slit to a
  // different source never invalidates its current width:
  //  - min: half the FINEST loaded source's native pixel scale, so the
  //    corridor is never narrower than what that source can meaningfully
  //    resolve (~0.3″ for AIA's ~0.6″/px).
  //  - max: the LARGEST loaded source's own field of view edge to edge
  //    (~1000-2500″ for AIA), so a slit can span a full image.
  // Falls back to FALLBACK_SLIT_WIDTH_ARCSEC_MIN/MAX while meta hasn't
  // loaded real scales/shapes yet.
  const slitWidthArcsecBounds = (): [number, number] => {
    const scales = [slitWidthScaleArcsec(contextSourceId), slitWidthScaleArcsec(radioSourceId)]
      .filter((value) => Number.isFinite(value) && value > 0);
    const fovs = [contextSourceId, radioSourceId]
      .map((sourceId) => sourceMaxWidthPx(sourceId) * slitWidthScaleArcsec(sourceId))
      .filter((value) => Number.isFinite(value) && value > 0);
    const min = scales.length ? 0.5 * Math.min(...scales) : FALLBACK_SLIT_WIDTH_ARCSEC_MIN;
    const max = fovs.length ? Math.max(...fovs) : FALLBACK_SLIT_WIDTH_ARCSEC_MAX;
    return [min, Math.max(min, max)];
  };
  useEffect(() => {
    if (!meta || trackingSourceOptions.some((option) => option.sourceId === trackingSourceId)) return;
    const defaultOption = trackingSourceOptions.find((option) => (
      option.sourceId === leftBaseLayer?.sourceId && option.layer.sourceRoleSnapshot === "context"
    )) ?? trackingSourceOptions.find((option) => option.layer.sourceRoleSnapshot === "context") ?? trackingSourceOptions[0];
    setTrackingSourceId(defaultOption?.sourceId ?? "");
  }, [leftBaseLayer?.sourceId, meta, trackingSourceId, trackingSourceOptions]);
  useEffect(() => {
    if (!meta || slitSourceOptions.some((option) => option.sourceId === slitSourceId)) return;
    setSlitSourceId(slitSourceOptions.find((option) => option.layer.sourceRoleSnapshot === "context")?.sourceId ?? slitSourceOptions[0]?.sourceId ?? "");
  }, [meta, slitSourceId, slitSourceOptions]);
  useEffect(() => {
    if (slits.length && !slits.some((slit) => slit.id === selectedSlitId)) setSelectedSlitId(slits[0].id);
    if (!slits.length && selectedSlitId) setSelectedSlitId("");
  }, [selectedSlitId, slits]);
  const selectedSlit = slits.find((slit) => slit.id === selectedSlitId) ?? null;
  const selectedSlitResult = selectedSlit ? slitResults[selectedSlit.id] ?? null : null;
  // Resolves a slit's LIVE bound contour layer (see TimeDistanceLane) - null
  // for slits not bound through a "contours"-kind layer, or whose bound
  // layer no longer exists in the current panel composition. Reading the
  // slit's ACTUAL bound layer (not a copy captured at extraction time) means
  // scrubbing the layer inspector's level control updates the lane without
  // a re-extraction. Generalized (any slit, not just the selected one) so
  // TWIN OVERLAY COMPOSITING below can resolve each contour-bound twin's own
  // layer independently.
  function contourLayerForSlit(candidate: SlitDefinition | null): LayerState | null {
    if (!candidate || candidate.bindingKind !== "contours") return null;
    const saved = allPanelLayers.find((layer) => layer.id === candidate.layerId);
    if (!saved) return null;
    const resolved = resolvedLayer(saved, allPanelLayers);
    return resolved.kind === "contours" ? resolved : null;
  }
  function contourDiffParamsForSlit(candidate: SlitDefinition | null, layer: LayerState | null) {
    if (!layer || !candidate || !meta) return null;
    const source = sources.find((entry) => entry.id === candidate.sourceId);
    if (!source) return null;
    const times = sourceTimeMjd(source, meta, sourceRoles);
    const fallback = sourceDifferences[candidate.sourceId] ?? defaultDifference(sourceRole(source, sourceRoles), meta.defaults.diffSeconds, times.length);
    const difference = differenceForLayer(layer, fallback);
    return differenceParams(difference, times);
  }
  // Resolves a slit's LIVE bound image layer (mirrors contourLayerForSlit) -
  // the source of layer-linked raster display settings (LAYER-LINKED RASTER
  // SCALE below). null for "raw"-bound slits (never displayed in a panel, so
  // there is nothing to link to) and for "image"-bound slits whose bound
  // layer isn't currently in the panel composition - both cases fall back to
  // the slit's own stored display (card controls stay visible for them).
  function imageLayerForSlit(candidate: SlitDefinition | null): LayerState | null {
    if (!candidate || candidate.bindingKind !== "image") return null;
    const saved = allPanelLayers.find((layer) => layer.id === candidate.layerId);
    if (!saved) return null;
    const resolved = resolvedLayer(saved, allPanelLayers);
    return resolved.kind === "image" ? resolved : null;
  }
  // A layer's scale can be "asinh" (image panels only); the TD lane's own
  // client-side raster shading only implements linear/sqrt/log fraction
  // curves (see the raster fill loop in TimeDistanceLane), and a slit's own
  // stored display can never be asinh (SlitDisplayState excludes it) - this
  // is only reachable via a linked image layer. Approximated with log, the
  // closest available stretch.
  function slitScaleFromLayerScale(scale: ScaleMode): SlitDisplayState["scale"] {
    return scale === "sqrt" || scale === "log" ? scale : scale === "asinh" ? "log" : "linear";
  }
  // Effective raster display for a slit: the bound image layer's live
  // vmin/vmax/cmap/scale when linked (same processed-data domain the image
  // panel itself uses), else the slit's own stored display.
  function effectiveSlitDisplay(candidate: SlitDefinition): SlitDisplayState {
    const layer = imageLayerForSlit(candidate);
    if (!layer) return candidate.display;
    return {
      vmin: numberValue(layer.display.vmin, candidate.display.vmin),
      vmax: numberValue(layer.display.vmax, candidate.display.vmax),
      cmap: normalizeColormap(layer.display.cmap, candidate.display.cmap),
      scale: slitScaleFromLayerScale(layer.display.scale)
    };
  }
  const selectedSlitContourLayer = useMemo(() => contourLayerForSlit(selectedSlit), [selectedSlit, allPanelLayers]);
  const selectedSlitContourDiffParams = useMemo(
    () => contourDiffParamsForSlit(selectedSlit, selectedSlitContourLayer),
    [meta, selectedSlit, selectedSlitContourLayer, sourceDifferences, sourceRoles, sources]
  );
  const selectedSlitImageLayer = useMemo(() => imageLayerForSlit(selectedSlit), [selectedSlit, allPanelLayers]);
  // TWIN OVERLAY COMPOSITING - the full linked-pair family: the unlinked
  // origin plus every twin sharing it (see SlitDefinition.linkedTo /
  // slitOriginId). Selecting EITHER member of a linked pair resolves to the
  // same family, so the lane composites identically regardless of which
  // member is selected.
  const selectedSlitFamily = useMemo(() => {
    if (!selectedSlit) return [] as SlitDefinition[];
    const originId = slitOriginId(selectedSlit, slits);
    return slits.filter((candidate) => candidate.id === originId || candidate.linkedTo === originId);
  }, [selectedSlit, slits]);
  // Resolves the family into a "base + contour families" composite. Only
  // the unambiguous shapes composite: exactly one image-bound member (the
  // base raster) plus any number of contour-bound members (each its own
  // contour family, "More than one twin: composite base + all contour-bound
  // twins' families"), or a lone contour-bound member with no image-bound
  // sibling at all. Two image-bound members (no unambiguous base) or
  // two-plus contour-bound members with no base member fall back to null -
  // TimeDistanceLane then renders only the selected member, exactly as
  // before this feature ("no raster-over-raster").
  const twinComposite = useMemo(() => {
    if (!selectedSlit || selectedSlitFamily.length < 2) return null;
    const imageMembers = selectedSlitFamily.filter((candidate) => candidate.bindingKind === "image");
    const contourMembers = selectedSlitFamily.filter((candidate) => candidate.bindingKind === "contours");
    if (imageMembers.length === 1) return { base: imageMembers[0] as SlitDefinition | null, contours: contourMembers };
    if (imageMembers.length === 0 && contourMembers.length === 1) return { base: null as SlitDefinition | null, contours: contourMembers };
    return null;
  }, [selectedSlit, selectedSlitFamily]);
  // Raster participant: the composite's base member when part of a genuine
  // pair; the lone contour-bound member when the family has no image-bound
  // member at all (preserves that member's own solo behavior - a
  // contour-bound slit with its own baseFreqIndex chosen shows its own
  // raster too, see hideRasterControls above - even though it is
  // technically part of a family via an unrelated raw-bound sibling); else
  // the selected slit itself (today's behavior - a slit can be its own
  // raster source regardless of bindingKind, exactly as before this
  // feature; contourLayerForSlit still gates contour-bound vs legacy
  // toggle-based contour rendering per member below).
  const twinRasterSlit = twinComposite
    ? (twinComposite.base ?? (twinComposite.contours.length === 1 ? twinComposite.contours[0] : null))
    : selectedSlit;
  const twinRasterResult = twinRasterSlit ? slitResults[twinRasterSlit.id] ?? null : null;
  const twinRaster = useMemo(
    () => (twinRasterSlit && twinRasterResult
      ? { slit: twinRasterSlit, result: twinRasterResult, display: effectiveSlitDisplay(twinRasterSlit) }
      : null),
    [twinRasterSlit, twinRasterResult, allPanelLayers]
  );
  // Contour participants: every contour-bound twin, PLUS the raster
  // participant itself (deduped) - a non-"contours"-bound raster member can
  // still carry its own legacy per-channel contour toggle
  // (slit.contourFreqIndices) alongside its raster, exactly as before this
  // feature; dropping it here would silently lose that overlay merely
  // because the slit is now part of a linked family.
  const twinContourSlits = useMemo(() => {
    if (!twinComposite) return selectedSlit ? [selectedSlit] : [];
    const byId = new Map<string, SlitDefinition>();
    if (twinRasterSlit) byId.set(twinRasterSlit.id, twinRasterSlit);
    for (const member of twinComposite.contours) byId.set(member.id, member);
    return [...byId.values()];
  }, [twinComposite, twinRasterSlit, selectedSlit]);
  const twinContours = useMemo(() => twinContourSlits.flatMap((member) => {
    const result = slitResults[member.id];
    if (!result) return [];
    const contourLayer = contourLayerForSlit(member);
    return [{ slit: member, result, contourLayer, contourDiffParams: contourDiffParamsForSlit(member, contourLayer) }];
  }), [twinContourSlits, slitResults, allPanelLayers, meta, sourceDifferences, sourceRoles, sources]);
  // "If both extracted results exist, composite; if only one extracted,
  // render what exists (current behavior) plus a subtle hint in the lane
  // corner naming the missing member." Only meaningful once a genuine pair
  // (or larger family) is resolved.
  const twinMissingHint = useMemo(() => {
    if (!twinComposite) return null;
    const missing: string[] = [];
    if (twinComposite.base && !slitResults[twinComposite.base.id]) missing.push(twinComposite.base.name);
    for (const member of twinComposite.contours) if (!slitResults[member.id]) missing.push(member.name);
    return missing.length ? `twin not extracted: ${missing.join(", ")}` : null;
  }, [twinComposite, slitResults]);
  const slitLaneHasContent = Boolean(selectedSlitResult || twinRaster || twinContours.length);
  const fanCurves = useMemo(() => fan ? fanFamilyCurves(fan) : [], [fan]);
  const slitTimeRange = timeRangeValues(spectrogramTimeRange, meta?.spectrogram.timeMjd ?? masterTimes);
  // Was gated on selectedSlitResult alone; now also shows the lane when only
  // an unselected twin has been extracted ("selecting EITHER member shows
  // the composite" - the selected member itself need not have a result).
  const slitLaneVisible = Boolean((slitInspectorOpen || slitPinLane) && selectedSlit && slitLaneHasContent && !recordingStatus);
  const correlationSeries = useMemo(() => {
    if (!meta || correlationTarget.length < 3) return [];
    return sadTracks.map((track) => correlationSeriesForTrack(track, correlationTarget, sourceAffine(track.sourceId)));
  }, [correlationTarget, contextSourceId, meta, radioSourceId, sadTracks]);
  const correlationTicks = useMemo(() => correlationSeries.map((item) => item.arrival).filter((tick): tick is CorrelationTick => Boolean(tick)), [correlationSeries]);
  const correlationXRange = correlationFullRange
    ? [masterMin, masterMax] as [number, number]
    : timeRangeValues(spectrogramTimeRange, meta?.spectrogram.timeMjd ?? masterTimes);
  const correlationTicksVisible = correlationTarget.length >= 3 && (correlationCardOpen || correlationPinTicks);

  const visibleSources = useMemo(() => {
    if (!meta || !eovsaAvailable) return [];
    const tolerance = nativeCadenceSeconds(meta.eovsa.timeMjd) / 2 / 86400;
    return eovsaSources.filter(
      (source) => source.spw_index === freqIndex && Math.abs(source.time_mjd - currentMjd) <= tolerance
    );
  }, [currentMjd, eovsaAvailable, eovsaSources, freqIndex, meta]);

  function initialPanelLayers(
    data: SessionMeta,
    sourceList: SourceMeta[],
    roles: Record<string, SourceRole>,
    differences: Record<string, DifferenceState>,
    includeContours = false,
    legacyContour?: Partial<Pick<LayerState, "contourFilled" | "contourLevelPercent" | "contourLevelKelvin" | "contourLevelSfu" | "contourLevelMode" | "contourLevelReference" | "contourOpacity" | "contourCmap">>
  ): Record<PanelSlotId, LayerState[]> {
    const context = sourceList.find((source) => source.id === (roleSource(sourceList, roles, "context")?.id ?? "context"));
    const radio = sourceList.find((source) => source.id === (roleSource(sourceList, roles, "radio")?.id ?? "radio"));
    const contextId = context?.id ?? "context";
    const radioId = radio?.id ?? "radio";
    const contextDiff = differences[contextId] ?? defaultDifference("context", data.defaults.diffSeconds, data.aia.timeMjd.length);
    const radioDiff = differences[radioId] ?? defaultDifference("radio", data.defaults.diffSeconds, data.eovsa.timeMjd.length);
    const contextLayer = defaultLayer("context-base", context?.label ?? "Context", contextId, "image", "context", { vmin: "0.5", vmax: "1.5", cmap: "gray", scale: "linear", radialGamma: 0 }, contextDiff);
    const radioLayer = defaultLayer("radio-base", radio?.label ?? "Radio", radioId, "image", "radio", { vmin: "-1000000", vmax: "5000000", cmap: "turbo", scale: "linear", radialGamma: 0 }, radioDiff);
    const contourLayer = {
      ...defaultLayer("radio-contours", "Radio contours", radioId, "contours", "radio", radioLayer.display, radioDiff, freqIndex),
      contourFilled: legacyContour?.contourFilled ?? false,
      contourLevelPercent: legacyContour?.contourLevelPercent ?? "50",
      contourLevelKelvin: legacyContour?.contourLevelKelvin ?? "1000000",
      contourLevelSfu: legacyContour?.contourLevelSfu ?? "1.0",
      contourLevelMode: legacyContour?.contourLevelMode ?? "percent",
      contourLevelReference: legacyContour?.contourLevelReference ?? "current",
      contourOpacity: legacyContour?.contourOpacity ?? "0.35",
      contourCmap: legacyContour?.contourCmap ?? "turbo"
    };
    return { left: includeContours ? [contextLayer, contourLayer] : [contextLayer], right: [radioLayer] };
  }

  function panelVisibleSources(slot: PanelSlotId): EovsaSource[] {
    const composition = compositionFor(panelLayers[slot], panelCompositions[slot]?.baseLayerId, panelCompositions[slot]?.overlayLayerIds);
    const rawBase = panelLayers[slot].find((layer) => layer.id === composition.baseLayerId);
    const base = rawBase ? resolvedLayer(rawBase, allPanelLayers) : undefined;
    const frequencies = base?.visible && base.sourceRoleSnapshot === "radio" ? new Set([base.freqIndex]) : new Set<number>();
    const tolerance = meta ? nativeCadenceSeconds(meta.eovsa.timeMjd) / 2 / 86400 : 0;
    return eovsaSources.filter((source) => frequencies.has(source.spw_index) && Math.abs(source.time_mjd - currentMjd) <= tolerance);
  }

  function compositionFor(layers: LayerState[], preferredBaseId?: string, preferredOverlayIds?: string[]): PanelComposition {
    const unique = normalizeLayerOrder(layers.filter((layer, index, all) => all.findIndex((candidate) => candidate.id === layer.id) === index));
    const base = unique.find((layer) => layer.id === preferredBaseId && layer.kind === "image") ?? unique.find((layer) => layer.kind === "image");
    const overlays = unique.filter((layer) => layer.id !== base?.id && layer.kind !== "spectrogram");
    const preferred = preferredOverlayIds?.map((id) => overlays.find((layer) => layer.id === id)).filter((layer): layer is LayerState => Boolean(layer)) ?? [];
    const ordered = normalizeLayerOrder([...preferred, ...overlays.filter((layer) => !preferred.some((candidate) => candidate.id === layer.id))]);
    return { baseLayerId: base?.id ?? "", overlayLayerIds: ordered.map((layer) => layer.id) };
  }

  function effectivePanelLayer(slot: PanelSlotId, layerId: string): LayerState | undefined {
    const layer = panelLayers[slot].find((candidate) => candidate.id === layerId);
    return layer ? resolvedLayer(layer, allPanelLayers) : undefined;
  }

  function setPanelBase(slot: PanelSlotId, layerId: string) {
    setPanelCompositions((value) => ({
      ...value,
      [slot]: compositionFor(panelLayers[slot], layerId, value[slot].overlayLayerIds)
    }));
  }

  function toggleSection(sectionId: string) {
    setCollapsedSections((value) => value.includes(sectionId) ? value.filter((id) => id !== sectionId) : [...value, sectionId]);
  }

  function setMasterCursor(mjd: number, stepSeconds = timeStepSeconds) {
    if (!meta || !Number.isFinite(mjd)) return;
    const next = clamp(mjd, masterMin, masterMax);
    const snapped = masterTimes.length
      ? snapFrameTime(masterTimes, next, stepSeconds, masterMin, masterMax, masterMin)
      : next;
    if (!Number.isFinite(snapped)) return;
    setMasterCursorMjd(snapped);
    setTimeIndex(closestIndex(meta.aia.timeMjd, snapped));
  }

  function setMasterSource(nextSourceId: string) {
    if (!meta) return;
    const nextSource = sources.find((source) => source.id === nextSourceId);
    const nextTimes = sourceTimeMjd(nextSource, meta, sourceRoles);
    if (!nextTimes.length) {
      setMasterWarning(`${nextSource?.label ?? nextSourceId} has no time axis; keeping Context master.`);
      const contextTimes = sourceTimeMjd(contextSource, meta, sourceRoles);
      const fallbackTimes = contextTimes.length ? contextTimes : sources.flatMap((source) => sourceTimeMjd(source, meta, sourceRoles)).filter(Number.isFinite).sort((left, right) => left - right);
      const fallbackBounds = timeBounds(fallbackTimes);
      const fallbackMjd = fallbackTimes[closestIndex(fallbackTimes, currentMjd)] ?? fallbackBounds[0];
      setMasterSourceId(contextSourceId);
      setTimeStepSeconds(0);
      setMasterStartMjd(fallbackBounds[0]);
      setMasterEndMjd(fallbackBounds[1]);
      setMasterCursorMjd(fallbackMjd);
      setTimeIndex(closestIndex(meta.aia.timeMjd, fallbackMjd));
      return;
    }
    setMasterWarning("");
    setMasterSourceId(nextSourceId);
    setTimeStepSeconds(0);
    const nextBounds = timeBounds(nextTimes);
    setMasterStartMjd(nextBounds[0]);
    setMasterEndMjd(nextBounds[1]);
    const nextMjd = clamp(currentMjd, nextBounds[0], nextBounds[1]);
    const snapped = nextTimes[closestIndex(nextTimes, nextMjd)] ?? nextMjd;
    setMasterCursorMjd(snapped);
    setTimeIndex(closestIndex(meta.aia.timeMjd, snapped));
  }

  function applyLayerPatch(layer: LayerState, patch: Partial<LayerState>): LayerState {
    const next = { ...layer, ...patch };
    if (layer.kind === "contours") {
      const requestedMode = patch.contourLevelMode;
      if (requestedMode && next.contourLevelReference === "global") {
        contourLevelModeRef.current[layer.id] = requestedMode;
      }
      if (patch.contourLevelReference === "current") {
        contourLevelModeRef.current[layer.id] = requestedMode ?? next.contourLevelMode;
        next.contourLevelMode = "percent";
      } else if (patch.contourLevelReference === "global") {
        next.contourLevelMode = requestedMode ?? contourLevelModeRef.current[layer.id] ?? next.contourLevelMode;
      } else if (next.contourLevelReference === "current") {
        next.contourLevelMode = "percent";
      }
    }
    if (patch.display) {
      const nextDisplay = { ...layer.display, ...patch.display };
      const parsedGamma = Number(nextDisplay.radialGamma ?? 0);
      const nextGamma = Number.isFinite(parsedGamma) ? Math.max(0, parsedGamma) : 0;
      if (layer.operation === "none" && (layer.display.radialGamma ?? 0) === 0 && nextGamma > 0) {
        nextDisplay.scale = "log";
        nextDisplay.vmin = "10";
        nextDisplay.vmax = "8000";
      }
      next.display = { ...nextDisplay, radialGamma: nextGamma };
    }
    if (patch.temporal) {
      next.temporal = normalizeTemporal(patch.temporal, layer.temporal);
      if (layer.operation !== "none" && layer.temporal.mode !== "bandpass" && next.temporal.mode === "bandpass") {
        next.display = { ...next.display, cmap: "coolwarm", vmin: "-0.2", vmax: "0.2", scale: "linear" };
      }
    }
    if (patch.sourceId) {
      const source = sources.find((candidate) => candidate.id === patch.sourceId);
      const immutableRole = sourceRole(source, {});
      if (!next.labelEdited) next.label = source?.label ?? next.label;
      next.sourceRoleSnapshot = immutableRole;
    }
    return next;
  }

  function updatePanelLayer(slot: PanelSlotId, layerId: string, patch: Partial<LayerState>) {
    setPanelLayers((value) => {
      const currentLayers = [...value.left, ...value.right];
      const requestedLayer = currentLayers.find((layer) => layer.id === layerId);
      const targetId = requestedLayer ? layerOrigin(requestedLayer, currentLayers).id : layerId;
      const next = {
        ...value,
        left: value.left.map((layer) => layer.id === targetId ? applyLayerPatch(layer, patch) : layer),
        right: value.right.map((layer) => layer.id === targetId ? applyLayerPatch(layer, patch) : layer)
      };
      return next;
    });
  }

  function radioWorldOffset(index: number): [number, number] {
    return [
      numberValue(xOffset, 7) + (channelOffsets.dx[index] ?? 0),
      numberValue(yOffset, 0) + (channelOffsets.dy[index] ?? 0)
    ];
  }

  function updateSelectedChannelOffsets(deltaX: number, deltaY: number) {
    if (!selectedChannels.length || !Number.isFinite(deltaX) || !Number.isFinite(deltaY)) return;
    setChannelOffsets((current) => {
      const next = { dx: [...current.dx], dy: [...current.dy], masked: [...current.masked] };
      for (const index of selectedChannels) {
        if (index < 0 || index >= next.dx.length) continue;
        next.dx[index] += deltaX;
        next.dy[index] += deltaY;
      }
      return next;
    });
  }

  function replaceSelectedChannelOffsets(next: ChannelOffsets) {
    setChannelOffsets(normalizeChannelOffsets(next, radioFreqGhz.length));
  }

  function closeAlignmentCard() {
    setAlignmentOpen(false);
    setChannelLassoArmed(false);
    setSelectedChannels([]);
  }

  function exportChannelOffsetsCsv() {
    const lines = ["channel,freq_ghz,dx_arcsec,dy_arcsec,masked"];
    radioFreqGhz.forEach((frequency, index) => {
      lines.push(`${index},${Number(frequency).toFixed(6)},${channelOffsets.dx[index] ?? 0},${channelOffsets.dy[index] ?? 0},${channelOffsets.masked[index] ? 1 : 0}`);
    });
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "radio-channel-offsets.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  function importChannelOffsetsCsv(csv: string) {
    const rows = csv.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    const header = rows[0]?.split(",").map((value) => value.trim()).join(",");
    const legacyHeader = "channel,freq_ghz,dx_arcsec,dy_arcsec";
    const maskedHeader = "channel,freq_ghz,dx_arcsec,dy_arcsec,masked";
    if (header !== legacyHeader && header !== maskedHeader) {
      setToast({ id: Date.now(), message: "CSV header must be channel,freq_ghz,dx_arcsec,dy_arcsec,masked" });
      return;
    }
    const includesMask = header === maskedHeader;
    const next = { dx: [...channelOffsets.dx], dy: [...channelOffsets.dy], masked: [...channelOffsets.masked] };
    const imported: number[] = [];
    let unknown = 0;
    let invalid = 0;
    for (const row of rows.slice(1)) {
      const fields = row.split(",").map((value) => value.trim());
      const channel = Number(fields[0]);
      const frequency = Number(fields[1]);
      const dx = Number(fields[2]);
      const dy = Number(fields[3]);
      const masked = fields[4];
      if (fields.length !== (includesMask ? 5 : 4) || !Number.isInteger(channel) || !Number.isFinite(frequency) || !Number.isFinite(dx) || !Number.isFinite(dy) || (includesMask && masked !== "0" && masked !== "1")) {
        invalid += 1;
        continue;
      }
      if (channel < 0 || channel >= radioFreqGhz.length) {
        unknown += 1;
        continue;
      }
      next.dx[channel] = dx;
      next.dy[channel] = dy;
      if (includesMask) next.masked[channel] = masked === "1";
      imported.push(channel);
    }
    if (invalid || !imported.length) {
      setToast({ id: Date.now(), message: invalid ? `CSV contains ${invalid} invalid row${invalid === 1 ? "" : "s"}.` : "CSV contains no known channel rows." });
      if (!imported.length) return;
    }
    setChannelOffsets(next);
    setSelectedChannels([...new Set(imported)].sort((left, right) => left - right));
    const missing = Math.max(0, radioFreqGhz.length - new Set(imported).size);
    setMessage(`Imported ${imported.length} channel offsets${unknown ? `; ignored ${unknown} unknown` : ""}${missing ? `; ${missing} channel${missing === 1 ? "" : "s"} unchanged` : ""}.`);
  }

  function closePixelProbe() {
    setPixelProbe(null);
    setPixelProbeData(null);
    setPixelProbeError("");
    setPixelProbeLoading(false);
  }

  function openPixelProbe(slot: PanelSlotId, panel: PanelId, pixel: [number, number], layerId: string) {
    const composition = compositionFor(panelLayers[slot], panelCompositions[slot]?.baseLayerId, panelCompositions[slot]?.overlayLayerIds);
    if (composition.baseLayerId !== layerId) return;
    setPixelProbe({ slot, panel, layerId, pixel, patchRadius: 1 });
    setPixelProbeData(null);
    setPixelProbeError("");
    setPixelProbeRefresh((value) => value + 1);
  }

  function addPanelLayer(slot: PanelSlotId, kind: LayerKind) {
    if (!meta || panelLayers[slot].length >= 8) return;
    const source = kind === "contours" ? radioSource : kind === "spectrogram" ? spectrogramSource : contextSource;
    if (!source) return;
    const role = sourceRole(source, {});
    if (!sourceSupportsLayer(source, kind, role)) return;
    const axis = sourceTimeMjd(source, meta, sourceRoles);
    const difference = sourceDifferences[source.id] ?? defaultDifference(role, meta.defaults.diffSeconds, axis.length || 1);
    const display = role === "radio" ? eovsaDisplay : aiaDisplay;
    const id = `${slot}-${kind}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    const layer = defaultLayer(id, kind === "contours" ? `${source.label} contours` : source.label, source.id, kind, role, display, difference, freqIndex);
    setPanelLayers((value) => {
      const layers = [...value[slot], layer];
      setPanelCompositions((compositions) => ({ ...compositions, [slot]: compositionFor(layers, compositions[slot].baseLayerId) }));
      return { ...value, [slot]: layers };
    });
    setSelectedLayerId({ slot, id });
  }

  function movePanelLayer(slot: PanelSlotId, layerId: string, delta: number) {
    setPanelLayers((value) => {
      const index = value[slot].findIndex((layer) => layer.id === layerId);
      const nextIndex = index + delta;
      if (index < 0 || nextIndex < 0 || nextIndex >= value[slot].length) return value;
      if (value[slot][index].kind === "contours" && value[slot][nextIndex].kind === "image") return value;
      const layers = [...value[slot]];
      [layers[index], layers[nextIndex]] = [layers[nextIndex], layers[index]];
      const orderedLayers = normalizeLayerOrder(layers);
      setPanelCompositions((compositions) => ({ ...compositions, [slot]: compositionFor(orderedLayers, compositions[slot].baseLayerId) }));
      return { ...value, [slot]: orderedLayers };
    });
  }

  function reorderPanelLayer(slot: PanelSlotId, fromId: string, toId: string) {
    setPanelLayers((value) => {
      const layers = [...value[slot]];
      const from = layers.findIndex((layer) => layer.id === fromId);
      const to = layers.findIndex((layer) => layer.id === toId);
      if (from < 0 || to < 0 || from === to) return value;
      if (layers[from].kind === "contours" && layers[to].kind === "image") return value;
      const [moved] = layers.splice(from, 1);
      layers.splice(to, 0, moved);
      const orderedLayers = normalizeLayerOrder(layers);
      setPanelCompositions((compositions) => ({ ...compositions, [slot]: compositionFor(orderedLayers, compositions[slot].baseLayerId) }));
      return { ...value, [slot]: orderedLayers };
    });
  }

  function removePanelLayer(slot: PanelSlotId, layerId: string) {
    const currentLayers = panelLayers[slot];
    const targetIndex = currentLayers.findIndex((layer) => layer.id === layerId);
    const nextSelection = targetIndex >= 0
      ? currentLayers[targetIndex + 1]?.id ?? currentLayers[targetIndex - 1]?.id ?? null
      : null;
    setPanelLayers((value) => {
      const currentAllLayers = [...value.left, ...value.right];
      const target = currentAllLayers.find((layer) => layer.id === layerId);
      const absorbedMirrors = target && !target.mirrorOf
        ? new Map(currentAllLayers
          .filter((layer) => layer.mirrorOf === target.id)
          .map((layer) => [layer.id, independentLayer(resolvedLayer(layer, currentAllLayers))]))
        : new Map<string, LayerState>();
      const next: Record<PanelSlotId, LayerState[]> = {
        left: value.left.filter((layer) => layer.id !== layerId).map((layer) => absorbedMirrors.get(layer.id) ?? layer),
        right: value.right.filter((layer) => layer.id !== layerId).map((layer) => absorbedMirrors.get(layer.id) ?? layer)
      };
      setPanelCompositions((compositions) => ({
        left: compositionFor(next.left, compositions.left.baseLayerId, compositions.left.overlayLayerIds),
        right: compositionFor(next.right, compositions.right.baseLayerId, compositions.right.overlayLayerIds)
      }));
      return next;
    });
    setSelectedLayerId((value) => value?.slot === slot && value.id === layerId && nextSelection ? { slot, id: nextSelection } : value?.slot === slot && value.id === layerId ? null : value);
  }

  function copyPanelLayer(slot: PanelSlotId, layerId: string) {
    const other: PanelSlotId = slot === "left" ? "right" : "left";
    const sourceLayer = panelLayers[slot].find((layer) => layer.id === layerId);
    const sourceSettings = sourceLayer ? resolvedLayer(sourceLayer, allPanelLayers) : undefined;
    if (!sourceLayer || !sourceSettings || panelLayers[other].length >= 8) return;
    const occupied = new Set(allPanelLayers.map((layer) => layer.id));
    const occupiedLabels = new Set(allPanelLayers.map((layer) => resolvedLayer(layer, allPanelLayers).label));
    let copyId = `${other}-${sourceSettings.kind}-${Date.now()}`;
    let suffix = 1;
    while (occupied.has(copyId)) copyId = `${other}-${sourceSettings.kind}-${Date.now()}-${suffix++}`;
    const stem = sourceSettings.label.replace(/ \(\d+\)$/, "");
    let labelSuffix = 2;
    let copyLabel = `${stem} (${labelSuffix})`;
    while (occupiedLabels.has(copyLabel)) copyLabel = `${stem} (${++labelSuffix})`;
    const copy = independentLayer({ ...sourceSettings, label: copyLabel, labelEdited: true }, copyId);
    setPanelLayers((value) => {
      const layers = normalizeLayerOrder([...value[other], copy]);
      setPanelCompositions((compositions) => ({ ...compositions, [other]: compositionFor(layers, compositions[other].baseLayerId) }));
      return { ...value, [other]: layers };
    });
    setSelectedLayerId({ slot: other, id: copy.id });
  }

  function mirrorPanelLayer(slot: PanelSlotId, layerId: string) {
    const other: PanelSlotId = slot === "left" ? "right" : "left";
    const sourceLayer = panelLayers[slot].find((layer) => layer.id === layerId);
    const sourceSettings = sourceLayer ? resolvedLayer(sourceLayer, allPanelLayers) : undefined;
    const origin = sourceLayer ? layerOrigin(sourceLayer, allPanelLayers) : undefined;
    if (!sourceLayer || !sourceSettings || !origin || panelLayers[other].length >= 8) return;
    const occupied = new Set(allPanelLayers.map((layer) => layer.id));
    let mirrorId = `${other}-${sourceSettings.kind}-mirror-${Date.now()}`;
    let suffix = 1;
    while (occupied.has(mirrorId)) mirrorId = `${other}-${sourceSettings.kind}-mirror-${Date.now()}-${suffix++}`;
    const mirror = independentLayer(sourceSettings, mirrorId, origin.id);
    setPanelLayers((value) => {
      const layers = normalizeLayerOrder([...value[other], mirror]);
      setPanelCompositions((compositions) => ({ ...compositions, [other]: compositionFor(layers, compositions[other].baseLayerId) }));
      return { ...value, [other]: layers };
    });
    setSelectedLayerId({ slot: other, id: mirror.id });
  }

  function unlinkPanelLayer(slot: PanelSlotId, layerId: string) {
    setPanelLayers((value) => {
      const currentAllLayers = [...value.left, ...value.right];
      const target = currentAllLayers.find((layer) => layer.id === layerId);
      if (!target?.mirrorOf) return value;
      const independent = independentLayer(resolvedLayer(target, currentAllLayers));
      return {
        ...value,
        [slot]: value[slot].map((layer) => layer.id === layerId ? independent : layer)
      };
    });
  }

  function aiaFrameRequest(index: number, sampleMjd = currentMjd, layer?: LayerState, requestCap?: FrameRequestCap): ScheduledFrameRequest {
    const aiaTimes = meta?.aia.timeMjd ?? [];
    const display = layer?.display ?? aiaDisplay;
    const difference = differenceForLayer(layer, contextDifference);
    const sourceId = layer?.sourceId ?? contextSourceId;
    const params: Record<string, string | number> = {
      sampleMjd,
      samplingPolicy: "nearest",
      maxOffsetSeconds: UNBOUNDED_DISPLAY_OFFSET_SECONDS,
      vmin: numberValue(display.vmin, 0.5),
      vmax: numberValue(display.vmax, 1.5),
      cmap: display.cmap,
      scale: display.scale,
      orientation: "solar",
      ...frameCapParams(requestCap),
      ...differenceParams(difference, aiaTimes),
      ...filterParams(layer)
    };
    const url = imageUrl(`/api/sessions/${sessionId}/sources/${sourceId}/frame.png`, params);
    return scheduledFrameRequest(url, {
      sessionId,
      sourceId,
      resolvedIndex: closestIndex(aiaTimes, sampleMjd),
      operation: difference.operation,
      reference: difference.reference,
      displayParams: displayParamsWithoutAddress(params),
      kind: "image"
    }, sampleMjd, predictDisplayResolution(aiaTimes, sampleMjd));
  }

  function eovsaFrameRequest(index: number, sampleMjd = currentMjd, layer?: LayerState, fullCube = false, requestCap?: FrameRequestCap): ScheduledFrameRequest {
    const radioTimes = meta?.eovsa.timeMjd ?? [];
    const display = layer?.display ?? eovsaDisplay;
    const difference = differenceForLayer(layer, radioDifference);
    const sourceId = layer?.sourceId ?? radioSourceId;
    const params = {
      sampleMjd,
      samplingPolicy: "nearest",
      maxOffsetSeconds: UNBOUNDED_DISPLAY_OFFSET_SECONDS,
      freqIndex: layer?.freqIndex ?? freqIndex,
      vmin: numberValue(display.vmin, -1e6),
      vmax: numberValue(display.vmax, 5e6),
      cmap: display.cmap,
      scale: display.scale,
      orientation: "solar",
      fullCube: fullCube ? 1 : 0,
      ...frameCapParams(requestCap),
      ...differenceParams(difference, radioTimes),
      ...filterParams(layer)
    };
    const resolvedIndex = meta ? closestIndex(radioTimes, sampleMjd) : index;
    const url = imageUrl(`/api/sessions/${sessionId}/sources/${sourceId}/frame.png`, params);
    return scheduledFrameRequest(url, {
      sessionId,
      sourceId,
      resolvedIndex,
      operation: difference.operation,
      reference: difference.reference,
      displayParams: displayParamsWithoutAddress(params),
      kind: "image"
    }, sampleMjd, predictDisplayResolution(radioTimes, sampleMjd));
  }

  function eovsaContourRequest(index: number, sampleMjd = currentMjd, layer?: LayerState, targetSourceId = contextSourceId, targetRole?: SourceRole): ScheduledFrameRequest {
    const radioTimes = meta?.eovsa.timeMjd ?? [];
    const samplingPolicy = "nearest";
    const radioPrediction = predictDisplayResolution(radioTimes, sampleMjd);
    const resolvedTargetRole = targetRole ?? sourceRole(sources.find((source) => source.id === targetSourceId), sourceRoles);
    const targetTimes = resolvedTargetRole === "radio" ? (meta?.eovsa.timeMjd ?? []) : (meta?.aia.timeMjd ?? []);
    const targetMjd = radioPrediction.resolvedMjd ?? sampleMjd;
    const targetTolerance = UNBOUNDED_DISPLAY_OFFSET_SECONDS;
    const targetPrediction = predictResolution(targetTimes, targetMjd, targetTolerance, "nearest");
    const contourDifference = differenceForLayer(layer, radioDifference);
    const params: Record<string, string | number> = {
      targetSourceId,
      sampleMjd,
      samplingPolicy,
      maxOffsetSeconds: UNBOUNDED_DISPLAY_OFFSET_SECONDS,
      ...differenceParams(contourDifference, radioTimes),
      levelPercent: numberValue(layer?.contourLevelPercent ?? "50", 50),
      levelKelvin: numberValue(layer?.contourLevelKelvin ?? "1000000", 1e6),
      levelSfu: numberValue(layer?.contourLevelSfu ?? "1.0", 1.0),
      levelMode: layer?.contourLevelReference === "global" ? layer.contourLevelMode : "percent",
      levelReference: layer?.contourLevelReference ?? "current"
    };
    const resolvedIndex = meta ? closestIndex(radioTimes, sampleMjd) : index;
    const url = imageUrl(`/api/sessions/${sessionId}/sources/${layer?.sourceId ?? radioSourceId}/contour-geometry`, params);
    return scheduledFrameRequest(url, {
      sessionId,
      sourceId: layer?.sourceId ?? radioSourceId,
      resolvedIndex,
      operation: contourDifference.operation,
      reference: contourDifference.reference,
      displayParams: {
        ...displayParamsWithoutAddress(params),
        targetResolvedIndex: targetPrediction.unavailable ? -1 : targetPrediction.resolvedIndex ?? -1
      },
      kind: "contours"
    }, sampleMjd, radioPrediction);
  }

  function requestForLayer(layer: LayerState, sampleMjd = currentMjd, targetSourceId = contextSourceId, fullCube = false, targetRole?: SourceRole, requestCap?: FrameRequestCap): ScheduledFrameRequest | undefined {
    if (!meta || !layer.visible) return undefined;
    const role = layer.sourceRoleSnapshot;
    if (layer.kind === "contours") return role === "radio" ? eovsaContourRequest(masterCursorIndex, sampleMjd, layer, targetSourceId, targetRole) : undefined;
    if (role === "context") return aiaFrameRequest(masterCursorIndex, sampleMjd, layer, requestCap);
    if (role === "radio") return eovsaFrameRequest(masterCursorIndex, sampleMjd, layer, fullCube, requestCap);
    return undefined;
  }

  function updatePanelFrameCap(slot: PanelSlotId, cap: FrameRequestCap) {
    setPanelFrameCaps((current) => {
      const previous = current[slot];
      const unchanged = previous === null
        ? cap === null
        : cap !== null && previous.maxWidth === cap.maxWidth && previous.maxHeight === cap.maxHeight;
      return unchanged ? current : { ...current, [slot]: cap };
    });
  }

  // Motion resolution ladder: half-res the instant motion starts, full-res
  // only after MOTION_RESOLUTION_IDLE_MS of no motion. `playing`/`timeScrubbing`
  // are the two motion sources; releasing either one arms the idle timer
  // instead of switching immediately, so quick pause/resume or scrub
  // click-release cycles don't thrash between resolutions.
  useEffect(() => {
    const motionActive = playing || timeScrubbing;
    if (motionActive) {
      if (motionIdleTimerRef.current) {
        window.clearTimeout(motionIdleTimerRef.current);
        motionIdleTimerRef.current = 0;
      }
      setMotionResolutionActive(true);
      return undefined;
    }
    motionIdleTimerRef.current = window.setTimeout(() => {
      motionIdleTimerRef.current = 0;
      setMotionResolutionActive(false);
    }, MOTION_RESOLUTION_IDLE_MS);
    return () => {
      if (motionIdleTimerRef.current) {
        window.clearTimeout(motionIdleTimerRef.current);
        motionIdleTimerRef.current = 0;
      }
    };
  }, [playing, timeScrubbing]);

  const useHalfResMotionCaps = smoothPlaybackEnabled && motionResolutionActive;
  // Per-slot caps actually used by playback/scrub identities while motion is
  // active; the coverage bar and every other consumer keep using
  // `panelFrameCaps` (full-res) directly via requestsForMasterTime's default.
  // Memoized so its identity is stable across unrelated re-renders (this
  // object sits in several effects' dependency arrays; panelFrameCaps itself
  // is already reference-stable across no-op cap updates).
  const motionPanelFrameCaps: Record<PanelSlotId, FrameRequestCap> = useMemo(() => (
    useHalfResMotionCaps
      ? { left: halveFrameCap(panelFrameCaps.left), right: halveFrameCap(panelFrameCaps.right) }
      : panelFrameCaps
  ), [useHalfResMotionCaps, panelFrameCaps]);

  function geometryForLayer(layer: LayerState, targetSourceId = layer.sourceId, targetRole?: SourceRole): Pick<RenderLayer, "shape" | "pixelToWorldAffine" | "worldOffset"> {
    const geometrySourceId = layer.kind === "contours" ? targetSourceId : layer.sourceId;
    const role = layer.kind === "contours"
      ? targetRole ?? sourceRole(sources.find((source) => source.id === geometrySourceId), sourceRoles)
      : layer.sourceRoleSnapshot;
    if (role === "radio") {
      return {
        shape: meta?.eovsa.shape ?? [1, 1],
        pixelToWorldAffine: meta?.wcs.eovsa.pixelToWorldAffine ?? IDENTITY_AFFINE,
        worldOffset: layer.kind === "image" ? radioWorldOffset(layer.freqIndex) : [0, 0]
      };
    }
    return {
      shape: meta?.aia.shape ?? [1, 1],
      pixelToWorldAffine: meta?.wcs.aia.pixelToWorldAffine ?? IDENTITY_AFFINE,
      worldOffset: [0, 0]
    };
  }

  function renderPanelLayersAt(sampleMjd: number, capsOverride: Record<PanelSlotId, FrameRequestCap> = panelFrameCaps): Record<PanelSlotId, RenderLayer[]> {
    if (!meta) return { left: [], right: [] } as Record<PanelSlotId, RenderLayer[]>;
    const render = (slot: PanelSlotId) => {
      const composition = compositionFor(panelLayers[slot], panelCompositions[slot]?.baseLayerId, panelCompositions[slot]?.overlayLayerIds);
      const byId = new Map(panelLayers[slot].map((layer) => [layer.id, layer]));
      const orderedLayerEntries = [byId.get(composition.baseLayerId), ...composition.overlayLayerIds.map((id) => byId.get(id))]
        .filter((layer): layer is LayerState => Boolean(layer))
        .filter((layer, index, all) => all.findIndex((candidate) => candidate.id === layer.id) === index)
        .sort((left, right) => layerKindRank(left.kind) - layerKindRank(right.kind));
      const orderedLayers = orderedLayerEntries.map((layer) => resolvedLayer(layer, allPanelLayers));
      const targetLayer = orderedLayers.find((layer) => layer.id === composition.baseLayerId) ?? orderedLayers.find((layer) => layer.kind === "image");
      if (!targetLayer || targetLayer.kind !== "image") return [];
      const targetSourceId = targetLayer?.sourceId ?? contextSourceId;
      const targetRole = targetLayer?.sourceRoleSnapshot;
      const fullCube = orderedLayers.some((layer) => layer.visible && layer.kind === "contours");
      return orderedLayerEntries
      .map((entry, index) => ({ entry, layer: orderedLayers[index] }))
      .filter(({ layer }) => layer.visible)
      .flatMap(({ entry, layer }) => {
        const request = requestForLayer(layer, sampleMjd, targetSourceId, fullCube, targetRole, capsOverride[slot]);
        if (!request) return [];
        const geometry = geometryForLayer(layer, targetSourceId, targetRole);
        const radio = layer.sourceRoleSnapshot === "radio";
        const colorbar = layer.kind === "contours" && radio && meta ? {
          minGhz: frequencyBounds(meta.eovsa.freqGhz)[0],
          maxGhz: frequencyBounds(meta.eovsa.freqGhz)[1],
          cmap: layer.contourCmap
        } : undefined;
        const baseRenderLayer = {
          layer: { ...layer, id: entry.id, mirrorOf: entry.mirrorOf },
          request,
          ...geometry,
          colorbar
        } as RenderLayer;
        return [baseRenderLayer];
      });
    };
    return { left: render("left"), right: render("right") };
  }

  const panelRenderLayers = useMemo(() => {
    // Motion resolution ladder: draw at half the panel's resolution cap
    // while playback/scrubbing is active (or within the idle grace period
    // after either stops); motionPanelFrameCaps collapses back to the raw,
    // zoom-derived panelFrameCaps otherwise, so this is a no-op when the
    // ladder is off or the view is static.
    return renderPanelLayersAt(currentMjd, motionPanelFrameCaps);
  }, [allPanelLayers, channelOffsets, currentMjd, contextDifference, eovsaDisplay, eovsaAvailable, freqIndex, meta, panelCompositions, panelFrameCaps, motionPanelFrameCaps, panelLayers, radioDifference, radioFreqGhz, sessionId, sourceRoles, sources, xOffset, yOffset, aiaDisplay, contextSourceId, radioSourceId, masterCursorIndex]);
  const scrubCursorMjd = snapFrameTime(masterTimes, timeSliderMjd, timeStepSeconds, masterMin, masterMax, masterMin);
  const scrubPanelRenderLayers = timeScrubbing ? renderPanelLayersAt(scrubCursorMjd, motionPanelFrameCaps) : panelRenderLayers;

  const [readyLayerKeys, setReadyLayerKeys] = useState<Record<string, string>>({});
  const [layerStatuses, setLayerStatuses] = useState<Record<string, FrameResolution>>({});

  function eovsaSpectrogramUrl() {
    return imageUrl(`/api/sessions/${sessionId}/sources/${spectrogramSourceId}/spectrogram.png`, {
      vmin: numberValue(spectrogramDisplay.vmin, 0.5),
      vmax: numberValue(spectrogramDisplay.vmax, 150),
      cmap: spectrogramDisplay.cmap,
      scale: spectrogramDisplay.scale,
      frequencyScale: spectrogramFrequencyScale,
      frequencyMinGhz: numberValue(spectrogramFrequencyRange.min, frequencyBounds(meta?.spectrogram.freqGhz ?? [])[0]),
      frequencyMaxGhz: numberValue(spectrogramFrequencyRange.max, frequencyBounds(meta?.spectrogram.freqGhz ?? [])[1]),
      normalization: spectrogramNormalization
    });
  }

  const panelReady = (Object.values(panelRenderLayers).flat()).length > 0 && Object.values(panelRenderLayers).flat().every((renderLayer) => readyLayerKeys[renderLayer.layer.id] === renderLayer.request.key);
  const spectrogramRenderable = spectrogramSource?.capabilities?.render !== false;
  const spectrogramUrl = meta && spectrogramRenderable ? eovsaSpectrogramUrl() : "";

  useEffect(() => {
    if (pixelProbe && !probedLayer) {
      closePixelProbe();
    }
  }, [pixelProbe, probedLayer]);

  useEffect(() => {
    if (!meta || !pixelProbe || !probedLayer || !Number.isFinite(masterMin) || !Number.isFinite(masterMax)) return undefined;
    const source = sources.find((candidate) => candidate.id === probedLayer.sourceId);
    const axis = sourceTimeMjd(source, meta, sourceRoles);
    const fallback = probedLayer.sourceRoleSnapshot === "radio" ? radioDifference : contextDifference;
    const difference = differenceForLayer(probedLayer, fallback);
    const params = new URLSearchParams({
      x: String(pixelProbe.pixel[0]),
      y: String(pixelProbe.pixel[1]),
      patchRadius: String(pixelProbe.patchRadius),
      startMjd: String(masterMin),
      endMjd: String(masterMax),
      maxPoints: "400",
      freqIndex: String(probedLayer.freqIndex),
      ...Object.fromEntries(Object.entries(differenceParams(difference, axis)).map(([key, value]) => [key, String(value)]))
    });
    if (probedLayer.maxOffsetSeconds !== undefined) params.set("maxOffsetSeconds", String(probedLayer.maxOffsetSeconds));
    params.set("samplingPolicy", probedLayer.samplingPolicy);
    const temporal = normalizeTemporal(probedLayer.temporal);
    if (temporal.mode !== "none") {
      params.set("temporalMode", temporal.mode);
      params.set("temporalSigmaShort", temporal.sigmaShort);
      params.set("temporalSigmaLong", temporal.sigmaLong);
    }
    const controller = new AbortController();
    setPixelProbeData(null);
    setPixelProbeError("");
    setPixelProbeLoading(true);
    const timer = window.setTimeout(() => {
      void apiJson<PixelProbeResponse>(
        `/api/sessions/${sessionId}/sources/${encodeURIComponent(probedLayer.sourceId)}/timeseries?${params.toString()}`,
        { signal: controller.signal },
        restoreSessionAfterRestart
      ).then((data) => {
        if (controller.signal.aborted) return;
        setPixelProbeData(data);
        setPixelProbeLoading(false);
      }).catch((error) => {
        if (controller.signal.aborted || error instanceof DOMException && error.name === "AbortError" || error instanceof SessionRestoredError) return;
        setPixelProbeError(error instanceof Error ? error.message : String(error));
        setPixelProbeLoading(false);
      });
    }, 300);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [contextDifference, masterMax, masterMin, meta, pixelProbe, pixelProbeRefresh, probedLayer, radioDifference, sessionId, sourceRoles, sources]);

  useEffect(() => {
    if (effectiveMasterMjd !== lastCursorMjdRef.current) {
      playbackDirectionRef.current = effectiveMasterMjd > lastCursorMjdRef.current ? 1 : -1;
      lastCursorMjdRef.current = effectiveMasterMjd;
    }
    if (timeScrubbing) return;
    timeSliderMjdRef.current = effectiveMasterMjd;
    setTimeSliderMjd(effectiveMasterMjd);
  }, [effectiveMasterMjd, timeScrubbing]);

  useEffect(() => {
    setTimeStepSeconds((value) => clampedTimeStepSeconds(value, minimumTimeStepSeconds));
  }, [minimumTimeStepSeconds]);

  useEffect(() => {
    if (!datasetName) return;
    try {
      if (correlationTarget.length >= 3) {
        const encoded = JSON.stringify(correlationTarget);
        for (const key of new Set([INITIAL_MANIFEST_NAME, datasetName].filter(Boolean))) localStorage.setItem(`sad-eovsa:correlation-target:${key}`, encoded);
      }
    } catch { /* storage is optional */ }
  }, [correlationTarget, datasetName]);

  useEffect(() => {
    if (!datasetName || correlationTarget.length >= 3) return;
    try {
      const raw = [INITIAL_MANIFEST_NAME, datasetName].filter(Boolean).map((key) => localStorage.getItem(`sad-eovsa:correlation-target:${key}`)).find(Boolean);
      const parsed = raw ? JSON.parse(raw) : [];
      if (Array.isArray(parsed) && parsed.length >= 3) {
        setCorrelationTarget(parsed as [number, number][]);
        void apiJson(`/api/sessions/${sessionId}/correlation-target`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ target: parsed })
        }).catch(() => undefined);
      }
    } catch { /* storage is optional */ }
  }, [correlationTarget.length, datasetName, sessionId]);

  useEffect(() => () => window.cancelAnimationFrame(timeCommitRafRef.current), []);

  useEffect(() => {
    if (!meta) {
      baselineSignatureRef.current = "";
      baselinePendingRef.current = false;
      setDirty(false);
      return;
    }
    if (baselinePendingRef.current || !baselineSignatureRef.current) {
      baselineSignatureRef.current = dirtySignature;
      baselinePendingRef.current = false;
      setDirty(false);
      return;
    }
    setDirty(dirtySignature !== baselineSignatureRef.current);
  }, [dirtySignature, meta]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(null), 8000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    function beforeUnload(event: BeforeUnloadEvent) {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = "";
    }
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [dirty]);

  useEffect(() => {
    if (!pixelProbe) return undefined;
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      closePixelProbe();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [pixelProbe]);

  useEffect(() => {
    if (!fanDrawStage) return undefined;
    function cancelFanDraw(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setFanDrawStage(0);
      setFanBoundaryDraft(null);
      setMessage("Fan drawing cancelled.");
    }
    window.addEventListener("keydown", cancelFanDraw);
    return () => window.removeEventListener("keydown", cancelFanDraw);
  }, [fanDrawStage]);

  useEffect(() => {
    if (!fanRedrawTarget) return undefined;
    function cancelFanRedraw(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setFanRedrawTarget(null);
      setMessage("Boundary redraw cancelled.");
    }
    window.addEventListener("keydown", cancelFanRedraw);
    return () => window.removeEventListener("keydown", cancelFanRedraw);
  }, [fanRedrawTarget]);

  useEffect(() => {
    if (!seedMode && !trackEditorOpen) return undefined;
    function trackingKeys(event: KeyboardEvent) {
      const editable = event.target instanceof Element && Boolean(event.target.closest("input, select, textarea, [contenteditable='true']"));
      if (event.key === "Escape") {
        event.preventDefault();
        setSeedMode(false);
        if (trackEditorOpen) setTrackEditorOpen(false);
        return;
      }
      if (!trackEditorOpen || editable) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) void redoTracks();
        else void undoTracks();
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        if (selectedAnchorFrame !== null) {
          event.preventDefault();
          void deleteSelectedAnchor();
        }
        return;
      }
      if ((event.key === "[" || event.key === "]") && selectedTrack?.anchors.length) {
        event.preventDefault();
        const anchors = selectedTrack.anchors;
        const anchor = event.key === "]"
          ? anchors.find((candidate) => candidate.frameIndex > currentTrackFrame)
          : [...anchors].reverse().find((candidate) => candidate.frameIndex < currentTrackFrame);
        if (!anchor) return;
        setSelectedAnchorFrame(anchor.frameIndex);
        setMasterCursor(anchor.mjd, 0);
      }
    }
    window.addEventListener("keydown", trackingKeys);
    return () => window.removeEventListener("keydown", trackingKeys);
  }, [currentTrackFrame, seedMode, selectedAnchorFrame, selectedTrack, trackEditorOpen, trackHistoryVersion]);

  useEffect(() => {
    clearFrameCache();
    imageCache.clear();
    setReadyLayerKeys({});
    setLayerStatuses({});
    setPixelProbe(null);
    setPixelProbeData(null);
    setPixelProbeError("");
  }, [sessionId]);

  useEffect(() => {
    const options = { capture: true };
    function keyDown(event: KeyboardEvent) {
      if (event.code !== "Space" && event.key !== " ") return;
      if (event.target instanceof HTMLElement && event.target.closest("input, select, textarea, button, a, [contenteditable='true'], dialog, [role='dialog']")) return;
      if (!pointerOverImageRef.current) return;
      event.preventDefault();
      event.stopPropagation();
      setSpaceDown(true);
    }
    function keyUp(event: KeyboardEvent) {
      if (event.code !== "Space" && event.key !== " ") return;
      if (event.target instanceof HTMLElement && event.target.closest("input, select, textarea, button, a, [contenteditable='true'], dialog, [role='dialog']")) return;
      if (pointerOverImageRef.current || spaceDown) {
        event.preventDefault();
        event.stopPropagation();
      }
      setSpaceDown(false);
    }
    function resetSpace() {
      setSpaceDown(false);
    }
    window.addEventListener("keydown", keyDown, options);
    window.addEventListener("keyup", keyUp, options);
    window.addEventListener("blur", resetSpace);
    return () => {
      window.removeEventListener("keydown", keyDown, options);
      window.removeEventListener("keyup", keyUp, options);
      window.removeEventListener("blur", resetSpace);
    };
  }, [spaceDown]);

  useEffect(() => {
    function ignoredTarget(target: EventTarget | null, event: KeyboardEvent): boolean {
      if (!(target instanceof Element)) return false;
      if (target.closest(".spectrogram-canvas")) return event.key !== " " && event.code !== "Space";
      return Boolean(target.closest("input, select, textarea, button, a, [contenteditable='true'], dialog, [role='dialog']"));
    }
    function keyDown(event: KeyboardEvent) {
      if (event.defaultPrevented || ignoredTarget(event.target, event)) return;
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      if (recordingStatus) return;
      if (event.key === " " || event.code === "Space") {
        if (pointerOverImageRef.current) return;
        event.preventDefault();
        setPlaying((value) => !value);
        return;
      }
      if (!meta || !masterTimes.length) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        const direction = event.key === "ArrowRight" ? 1 : -1;
        const step = event.shiftKey ? 10 : 1;
        stepTime(direction * step);
      } else if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        const rangeIndex = closestIndex(masterTimes, event.key === "End" ? masterMax : masterMin);
        setMasterCursor(masterTimes[rangeIndex] ?? currentMjd);
      }
    }
    window.addEventListener("keydown", keyDown);
    return () => window.removeEventListener("keydown", keyDown);
  }, [currentMjd, masterCursorIndex, masterMax, masterMin, masterTimes, meta, recordingStatus, timeStepSeconds]);

  useEffect(() => {
    function pointerMove(event: PointerEvent) {
      const drag = layoutDragRef.current;
      if (!drag) return;
      if (drag.kind === "rail") {
        setLayout({ ...drag.layout, railWidth: clamp(drag.layout.railWidth + event.clientX - drag.startX, 260, 520) });
      } else if (drag.kind === "spectrogram") {
        setLayout({ ...drag.layout, spectrogramHeight: clamp(drag.layout.spectrogramHeight + event.clientY - drag.startY, 80, 320) });
      } else if (drag.kind === "slitLane") {
        const rect = workspaceRef.current?.getBoundingClientRect();
        if (!rect) return;
        const maximum = Math.max(MIN_SLIT_LANE_HEIGHT, rect.height - drag.layout.spectrogramHeight - 12 - MIN_PANEL_GRID_HEIGHT);
        setLayout({
          ...drag.layout,
          slitLaneHeight: clamp(drag.layout.slitLaneHeight + event.clientY - drag.startY, MIN_SLIT_LANE_HEIGHT, maximum)
        });
      } else {
        const rect = panelGridRef.current?.getBoundingClientRect();
        if (!rect) return;
        setLayout({ ...drag.layout, imageSplit: clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0.25, 0.75) });
      }
      event.preventDefault();
    }
    function pointerUp() {
      layoutDragRef.current = null;
    }
    window.addEventListener("pointermove", pointerMove);
    window.addEventListener("pointerup", pointerUp);
    window.addEventListener("pointercancel", pointerUp);
    return () => {
      window.removeEventListener("pointermove", pointerMove);
      window.removeEventListener("pointerup", pointerUp);
      window.removeEventListener("pointercancel", pointerUp);
    };
  }, []);

  function startLayoutDrag(kind: LayoutDragState["kind"], event: React.PointerEvent<HTMLElement>) {
    layoutDragRef.current = {
      kind,
      startX: event.clientX,
      startY: event.clientY,
      layout
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  async function loadDefault() {
    if (!confirmDatasetReplace()) return;
    setLoading(true);
    setMessage("Loading sample AIA H5 and EOVSA FITS sequence...");
    try {
      const data = await apiJson<SessionMeta>("/api/sessions/sample", { method: "POST" });
      baselinePendingRef.current = true;
      alignmentLoadedSessionRef.current = data.sessionId;
      setMeta(data);
      const loadedSources = sourcesForMeta(data);
      setSelectedSourceId(loadedSources[0]?.id ?? "");
      setSourceRoles({});
      const defaultMasterId = loadedSources.find((source) => source.role === "context")?.id ?? loadedSources[0]?.id ?? "context";
      const defaultMasterTimes = sourceTimeMjd(loadedSources.find((source) => source.id === defaultMasterId), data);
      const defaultMasterBounds = timeBounds(defaultMasterTimes);
      setMasterSourceId(defaultMasterId);
      setMasterStartMjd(defaultMasterBounds[0]);
      setMasterEndMjd(defaultMasterBounds[1]);
      setMasterCursorMjd(data.aia.timeMjd[data.defaults.timeIndex] ?? defaultMasterBounds[0]);
      setMasterWarning("");
      const nextDifferences = Object.fromEntries(loadedSources.map((source) => {
        const role = sourceRole(source, {});
        const times = sourceTimeMjd(source, data);
        const count = times.length || data.aia.times.length;
        const state = defaultDifference(role, data.defaults.diffSeconds, count);
        state.meanStartMjd = times[state.meanStartIndex] ?? data.aia.timeMjd[state.meanStartIndex];
        state.meanEndMjd = times[state.meanEndIndex] ?? data.aia.timeMjd[state.meanEndIndex];
        return [source.id, state];
      }));
      setSourceDifferences(nextDifferences);
      const defaults = initialPanelLayers(data, loadedSources, {}, nextDifferences);
      setPanelLayers(defaults);
      setPanelCompositions({ left: compositionFor(defaults.left), right: compositionFor(defaults.right) });
      setTrackingSourceId(defaults.left.find((layer) => layer.kind === "image" && layer.sourceRoleSnapshot === "context")?.sourceId ?? "");
      setSelectedLayerId(null);
      setCollapsedSections([]);
      setFilePickerOpen(false);
      setLastBrowsedDirectory("");
      setTimeIndex(data.defaults.timeIndex);
      setFreqIndex(data.defaults.freqIndex);
      const radioMeta = loadedSources.find((source) => source.role === "radio");
      const loadedOffsets = normalizeChannelOffsets(
        data.eovsa.channelOffsets ?? (radioMeta ? { ...radioMeta.channelOffsets, channelMask: radioMeta.channelMask } : { channelMask: data.eovsa.channelMask }),
        data.eovsa.freqGhz.length
      );
      alignmentPostKeyRef.current = JSON.stringify(loadedOffsets);
      setChannelOffsets(loadedOffsets);
      setAlignmentOpen(false);
      setSpectrogramDisplayOpen(false);
      setXOffset(String(data.defaults.xOffsetArcsec));
      setYOffset(String(data.defaults.yOffsetArcsec));
      setSadTracks([]);
      setSelectedTrackId("");
      setTrackEditorOpen(false);
      setEovsaSources([]);
      setRoiWorld([]);
      setRoiAia([]);
      setRoiEovsa([]);
      setCorrelationTarget([]);
      setCorrelationCardOpen(false);
      setCorrelationCardDismissed(false);
      setSlits([]);
      setSelectedSlitId("");
      setSlitSourceId(defaults.left.find((layer) => layer.kind === "image")?.sourceId ?? "");
      setSlitDraftWidthArcsec(DEFAULT_SLIT_WIDTH_ARCSEC);
      setSlitDraftSmoothPx(DEFAULT_SLIT_SMOOTH_PX);
      setSlitInspectorOpen(false);
      setSlitPinLane(false);
      setSlitDrawArmed(false);
      setSlitResults({});
      setFloatingCardSizes({});
      setCorrelationPinTicks(false);
      setCorrelationFullHeightTicks(false);
      setCorrelationPeakDecel(false);
      setPlaying(false);
      setPlaybackFps(DEFAULT_PLAYBACK_FPS);
      setTimeStepSeconds(0);
      setDatasetName("sample");
      setSolarView(defaultSolarView(data));
      setLayout(DEFAULT_LAYOUT);
      setSpectrogramDisplay({
        vmin: String(data.spectrogram.defaults.vmin),
        vmax: String(data.spectrogram.defaults.vmax),
        cmap: data.spectrogram.defaults.cmap,
        scale: normalizeScale(data.spectrogram.defaults.scale, "log"),
        radialGamma: 0
      });
      setSpectrogramNormalization("none");
      setSpectrogramFrequencyScale("linear");
      setSpectrogramFrequencyInverted(false);
      setSpectrogramFrequencyRange(coerceFrequencyRange(undefined, data.spectrogram.freqGhz));
      setSpectrogramTimeRange(coerceTimeRange(undefined, data.spectrogram.timeMjd));
      setMessage("Sample data loaded. Draw a ROI, then track features or extract radio sources.");
    } catch (error) {
      baselinePendingRef.current = false;
      reportError(error);
    } finally {
      setLoading(false);
    }
  }

  function applyLoadedSession(data: LoadSessionResponse) {
    baselinePendingRef.current = true;
    alignmentLoadedSessionRef.current = data.sessionId;
    const loaded = data.loadedState;
    const ui = loaded?.ui ?? {};
    const loadedSources = sourcesForMeta(data);
    setMeta(data);
    setSelectedSourceId(ui.selectedSourceId ?? loadedSources[0]?.id ?? "");
    setSourceRoles(ui.sourceRoles ?? {});
    const savedMasterId = ui.timeline?.masterSourceId;
    const savedMaster = loadedSources.find((source) => source.id === savedMasterId);
    const savedMasterTimes = sourceTimeMjd(savedMaster, data, ui.sourceRoles ?? {});
    const fallbackMaster = loadedSources.find((source) => sourceRole(source, ui.sourceRoles ?? {}) === "context" && sourceTimeMjd(source, data, ui.sourceRoles ?? {}).length)
      ?? loadedSources.find((source) => sourceTimeMjd(source, data, ui.sourceRoles ?? {}).length > 0);
    const inferredMasterId = savedMaster && savedMasterTimes.length ? savedMaster.id : fallbackMaster?.id ?? "context";
    const inferredMaster = loadedSources.find((source) => source.id === inferredMasterId);
    const inferredTimes = sourceTimeMjd(inferredMaster, data, ui.sourceRoles ?? {});
    const inferredBounds = timeBounds(inferredTimes);
    setMasterSourceId(inferredMasterId);
    setMasterWarning(savedMasterId && inferredMasterId !== savedMasterId
      ? "Saved master source is unavailable; using Context."
      : !inferredMaster || !inferredTimes.length
        ? "No master time axis is available; choose another master source."
        : "");
    const savedStart = Number(ui.timeline?.startMjd);
    const savedEnd = Number(ui.timeline?.endMjd);
    const savedMasterSummary = (loaded?.savedSources ?? []).find((source) => (
      source.id === inferredMasterId || sourceRole(source, ui.sourceRoles ?? {}) === sourceRole(inferredMaster, ui.sourceRoles ?? {})
    ));
    const savedSummaryMjd = (value: unknown, fallback: number) => {
      const numeric = Number(value);
      if (Number.isFinite(numeric)) return numeric;
      const text = String(value ?? "").trim();
      if (!text) return fallback;
      const datePrefix = mjdToUtc(inferredBounds[0]).slice(0, 10);
      return parseTimeMjd(text.includes("-") ? text : `${datePrefix} ${text}`, fallback);
    };
    const savedSummaryStart = savedSummaryMjd(savedMasterSummary?.time?.start, Number.NaN);
    const savedSummaryEnd = savedSummaryMjd(savedMasterSummary?.time?.end, Number.NaN);
    const dataExtendedEarlier = Number.isFinite(savedSummaryStart) && inferredBounds[0] < savedSummaryStart - 1e-9;
    const dataExtendedLater = Number.isFinite(savedSummaryEnd) && inferredBounds[1] > savedSummaryEnd + 1e-9;
    const normalizedStart = dataExtendedEarlier
      ? inferredBounds[0]
      : clamp(Number.isFinite(savedStart) ? savedStart : inferredBounds[0], inferredBounds[0], inferredBounds[1]);
    const normalizedEnd = dataExtendedLater
      ? inferredBounds[1]
      : clamp(Number.isFinite(savedEnd) ? savedEnd : inferredBounds[1], inferredBounds[0], inferredBounds[1]);
    const rangeStart = Math.min(normalizedStart, normalizedEnd);
    const rangeEnd = Math.max(normalizedStart, normalizedEnd);
    setMasterStartMjd(rangeStart);
    setMasterEndMjd(rangeEnd);
    let nextDifferences: Record<string, DifferenceState>;
    if (ui.sourceDifferences) {
      nextDifferences = Object.fromEntries(loadedSources.map((source) => {
        const role = sourceRole(source, ui.sourceRoles ?? {});
        const nativeAxis = sourceTimeMjd({ ...source, role }, data, ui.sourceRoles ?? {});
        const savedDifference = ui.sourceDifferences?.[source.id] as (Partial<DifferenceState> & { meanStartMjd?: number; meanEndMjd?: number }) | undefined;
        const hasExplicitMean = Number.isFinite(Number(savedDifference?.meanStartMjd)) || Number.isFinite(Number(savedDifference?.meanEndMjd));
        // v1 radio mean indices were AIA-frame indices; preserve their actual MJD
        // before projecting the canonical values onto the radio-native controls.
        const legacyAxis = role === "radio" && !hasExplicitMean ? data.aia.timeMjd : nativeAxis;
        const normalized = normalizeDifference(
          savedDifference,
          role,
          ui.diffSeconds ?? data.defaults.diffSeconds,
          nativeAxis.length || data.aia.times.length,
          legacyAxis
        );
        if (role === "radio" && !hasExplicitMean && nativeAxis.length) {
          normalized.meanStartIndex = closestIndex(nativeAxis, normalized.meanStartMjd ?? nativeAxis[0]);
          normalized.meanEndIndex = closestIndex(nativeAxis, normalized.meanEndMjd ?? nativeAxis[Math.min(4, nativeAxis.length - 1)]);
        }
        return [source.id, normalized];
      }));
    } else {
      const legacyMode: DifferenceMode = booleanValue(ui.useRunningDiff, true) ? "running" : "none";
      nextDifferences = Object.fromEntries(loadedSources.map((source) => [
        source.id,
        normalizeDifference(
          { mode: sourceRole(source, ui.sourceRoles ?? {}) === "spectrogram" ? "none" : legacyMode },
          sourceRole(source, ui.sourceRoles ?? {}),
          ui.diffSeconds ?? data.defaults.diffSeconds,
          sourceTimeMjd(source, data, ui.sourceRoles ?? {}).length || data.aia.times.length,
          sourceRole(source, ui.sourceRoles ?? {}) === "radio" ? data.eovsa.timeMjd : sourceRole(source, ui.sourceRoles ?? {}) === "spectrogram" ? data.spectrogram.timeMjd : data.aia.timeMjd
        )
      ]));
    }
    setSourceDifferences(nextDifferences);
    const canonicalLayerState = Boolean(ui.layers);
    const legacyContour = !canonicalLayerState && booleanValue(ui.showEovsaContours, false)
      ? {
          contourFilled: booleanValue(ui.contourFilled, false),
          contourLevelPercent: String(ui.contourLevelPercent ?? "50"),
          contourLevelKelvin: String(ui.contourLevelKelvin ?? "1000000"),
          contourLevelSfu: String(ui.contourLevelSfu ?? "1.0"),
          contourLevelMode: ui.contourLevelMode === "kelvin" ? "kelvin" as const : ui.contourLevelMode === "sfu" ? "sfu" as const : "percent" as const,
          contourLevelReference: ui.contourLevelReference === "global" ? "global" as const : "current" as const,
          contourOpacity: String(ui.contourOpacity ?? "0.35"),
          contourCmap: contourColormap(ui.contourCmap)
        }
      : undefined;
    const defaults = initialPanelLayers(data, loadedSources, ui.sourceRoles ?? {}, nextDifferences, Boolean(legacyContour), legacyContour);
    const savedLayers = Array.isArray(ui.layers)
      ? ui.layers
      : ui.layers && typeof ui.layers === "object"
        ? Object.entries(ui.layers).map(([id, layer]) => ({ ...(layer as Record<string, unknown>), id: (layer as Record<string, unknown>).id ?? id }))
        : [];
    const layerById = new Map(savedLayers.map((layer) => [String(layer.id), layer]));
    const normalizedPanels: Record<PanelSlotId, LayerState[]> = { left: [], right: [] };
    let missingLayerSource = false;
    const usedLayerIds = new Set<string>();
    for (const slot of ["left", "right"] as PanelSlotId[]) {
      const panelState = ui.panels?.[slot];
      const canonicalIds = [panelState?.baseLayerId, ...(panelState?.overlayLayerIds ?? [])].filter((id): id is string => Boolean(id));
      const savedIds = panelState?.layerIds ?? canonicalIds;
      const fallback = defaults[slot];
      const selected = savedIds.map((id) => layerById.get(id)).filter((layer): layer is LayerState => Boolean(layer));
      const useSavedPanel = canonicalLayerState && Boolean(panelState);
      normalizedPanels[slot] = (useSavedPanel ? selected : selected.length ? selected : fallback).map((layer, index) => {
        const fallbackLayer = fallback.find((candidate) => candidate.id === layer.id) ?? fallback[index % Math.max(1, fallback.length)] ?? defaults.left[0];
        let normalized = normalizeLayer(layer, fallbackLayer);
        if (usedLayerIds.has(normalized.id)) {
          const baseId = normalized.id;
          let suffix = 1;
          let candidateId = `${baseId}-${slot}`;
          while (usedLayerIds.has(candidateId)) candidateId = `${baseId}-${slot}-${suffix++}`;
          normalized = { ...normalized, id: candidateId };
        }
        usedLayerIds.add(normalized.id);
        const actualSource = loadedSources.find((source) => source.id === normalized.sourceId);
        if (!actualSource) {
          missingLayerSource = true;
          return {
            ...normalized,
            sourceId: fallbackLayer.sourceId,
            sourceRoleSnapshot: fallbackLayer.sourceRoleSnapshot,
            label: fallbackLayer.label
          };
        }
        // Canonical layers retain their saved capability snapshot, but a source
        // metadata role is authoritative when a malformed/legacy snapshot lies.
        if (actualSource.role === "context" || actualSource.role === "radio" || actualSource.role === "spectrogram") {
          normalized = { ...normalized, sourceRoleSnapshot: actualSource.role };
        }
        return normalized;
      }).filter((layer): layer is LayerState => Boolean(layer)).slice(0, 8);
      normalizedPanels[slot] = normalizeLayerOrder(normalizedPanels[slot]);
      if (!canonicalLayerState && !normalizedPanels[slot].some((layer) => layer.kind === "image")) {
        const fallbackBase = fallback.find((layer) => layer.kind === "image");
        if (fallbackBase) {
          let fallbackId = fallbackBase.id;
          let suffix = 1;
          while (usedLayerIds.has(fallbackId)) fallbackId = `${fallbackBase.id}-${slot}-${suffix++}`;
          const insertedBase = fallbackId === fallbackBase.id ? fallbackBase : { ...fallbackBase, id: fallbackId };
          usedLayerIds.add(insertedBase.id);
          normalizedPanels[slot].unshift(insertedBase);
        }
      }
    }
    const normalizedLayerIds = new Set([...normalizedPanels.left, ...normalizedPanels.right].map((layer) => layer.id));
    for (const slot of ["left", "right"] as PanelSlotId[]) {
      normalizedPanels[slot] = normalizedPanels[slot].map((layer) =>
        layer.mirrorOf && !normalizedLayerIds.has(layer.mirrorOf)
          ? independentLayer(layer)
          : layer
      );
    }
    setPanelLayers(normalizedPanels);
    setPanelCompositions({
      left: compositionFor(normalizedPanels.left, ui.panels?.left?.baseLayerId, ui.panels?.left?.overlayLayerIds),
      right: compositionFor(normalizedPanels.right, ui.panels?.right?.baseLayerId, ui.panels?.right?.overlayLayerIds)
    });
    const restoredTrackingSources = new Set([...normalizedPanels.left, ...normalizedPanels.right]
      .map((layer) => resolvedLayer(layer, [...normalizedPanels.left, ...normalizedPanels.right]))
      .filter((layer) => layer.kind === "image" && ["context", "radio"].includes(layer.sourceRoleSnapshot))
      .map((layer) => layer.sourceId));
    const defaultTrackingSource = normalizedPanels.left
      .map((layer) => resolvedLayer(layer, [...normalizedPanels.left, ...normalizedPanels.right]))
      .find((layer) => layer.kind === "image" && layer.sourceRoleSnapshot === "context")?.sourceId
      ?? [...restoredTrackingSources][0]
      ?? "";
    setTrackingSourceId(ui.trackingSourceId && restoredTrackingSources.has(ui.trackingSourceId) ? ui.trackingSourceId : defaultTrackingSource);
    setSelectedLayerId(null);
    if (missingLayerSource) setMasterWarning((value) => value ? `${value} Missing layer sources were replaced.` : "Some saved layer sources are unavailable; defaults were substituted.");
    const legacyCursor = inferredTimes[clampIndex(ui.timeIndex, closestIndex(inferredTimes, inferredBounds[0]), inferredTimes.length)] ?? inferredBounds[0];
    const requestedCursor = Number(ui.timeline?.cursorMjd);
    const fallbackCursor = Number.isFinite(legacyCursor) ? legacyCursor : rangeStart;
    const cursor = clamp(Number.isFinite(requestedCursor) ? requestedCursor : fallbackCursor, rangeStart, rangeEnd);
    const savedUsesVirtualGrid = positiveTimeStepSeconds(ui.timeStepSeconds) > 0;
    setMasterCursorMjd(savedUsesVirtualGrid || !inferredTimes.length ? cursor : inferredTimes[closestIndex(inferredTimes, cursor)]);
    setTimeIndex(closestIndex(data.aia.timeMjd, cursor));
    setFreqIndex(clampIndex(ui.freqIndex, data.defaults.freqIndex, data.eovsa.freqGhz.length));
    const radioMeta = loadedSources.find((source) => source.role === "radio");
    const loadedOffsets = normalizeChannelOffsets(
      loaded?.channelOffsets ?? data.eovsa.channelOffsets ?? (radioMeta ? { ...radioMeta.channelOffsets, channelMask: radioMeta.channelMask } : { channelMask: data.eovsa.channelMask }),
      data.eovsa.freqGhz.length
    );
    alignmentPostKeyRef.current = JSON.stringify(loadedOffsets);
    setChannelOffsets(loadedOffsets);
    setAlignmentOpen(false);
    setSpectrogramDisplayOpen(false);
    setFilePickerOpen(false);
    setLastBrowsedDirectory("");
    setXOffset(String(ui.xOffsetArcsec ?? data.defaults.xOffsetArcsec));
    setYOffset(String(ui.yOffsetArcsec ?? data.defaults.yOffsetArcsec));
    setCollapsedSections(Array.isArray(ui.collapsedSections) ? ui.collapsedSections : []);
    setAiaDisplay(ui.aiaDisplay ? { ...ui.aiaDisplay, cmap: normalizeColormap(ui.aiaDisplay.cmap, "gray"), scale: normalizeScale(ui.aiaDisplay.scale, "linear"), radialGamma: clamp(Number(ui.aiaDisplay.radialGamma ?? 0), 0, 3) } : { vmin: "0.5", vmax: "1.5", cmap: "gray", scale: "linear", radialGamma: 0 });
    setEovsaDisplay(ui.eovsaDisplay ? { ...ui.eovsaDisplay, cmap: normalizeColormap(ui.eovsaDisplay.cmap, "turbo"), scale: normalizeScale(ui.eovsaDisplay.scale, "linear"), radialGamma: clamp(Number(ui.eovsaDisplay.radialGamma ?? 0), 0, 3) } : { vmin: "-1000000", vmax: "5000000", cmap: "turbo", scale: "linear", radialGamma: 0 });
    setSpectrogramDisplay(ui.spectrogramDisplay ? {
      ...ui.spectrogramDisplay,
      cmap: normalizeColormap(ui.spectrogramDisplay.cmap, "viridis"),
      scale: normalizeScale(ui.spectrogramDisplay.scale, data.spectrogram.defaults.scale),
      radialGamma: 0
    } : {
      vmin: String(data.spectrogram.defaults.vmin),
      vmax: String(data.spectrogram.defaults.vmax),
      cmap: normalizeColormap(data.spectrogram.defaults.cmap, "viridis"),
      scale: normalizeScale(data.spectrogram.defaults.scale, "log"),
      radialGamma: 0
    });
    setSpectrogramNormalization(spectrogramNormalizationValue(ui.spectrogramNormalization));
    setSpectrogramFrequencyScale(ui.spectrogramFrequencyScale === "log" ? "log" : "linear");
    setSpectrogramFrequencyInverted(Boolean(ui.spectrogramFrequencyInverted));
    setSpectrogramFrequencyRange(coerceFrequencyRange(ui.spectrogramFrequencyRange, data.spectrogram.freqGhz));
    setSpectrogramTimeRange(coerceTimeRange(undefined, data.spectrogram.timeMjd));
    setPlaybackFps(playbackFpsValue(ui.playbackFps));
    setSmoothPlaybackEnabled(booleanValue(ui.smoothPlayback, true));
    setShowCacheCoverage(booleanValue(ui.showCacheCoverage, true));
    applyFrameCacheGb(numberValue(String(ui.frameCacheGb ?? ""), 2));
    const restoredMinimumCadence = sensibleCadenceSeconds(Math.min(
      ...loadedSources
        .filter((source) => source.status !== "placeholder")
        .map((source) => nativeCadenceSeconds(sourceTimeMjd(source, data, ui.sourceRoles ?? {})))
        .filter((cadence) => cadence > 0 && Number.isFinite(cadence))
    ));
    setTimeStepSeconds(clampedTimeStepSeconds(ui.timeStepSeconds, restoredMinimumCadence));
    setSolarView(coerceSolarView(ui.solarView, defaultSolarView(data)));
    setLayout(coerceLayout(ui.layout));
    setRoiWorld(loaded?.roiWorld ?? []);
    setRoiAia(loaded?.roiAia ?? []);
    setRoiEovsa(loaded?.roiEovsa ?? []);
    let storedTarget: [number, number][] = [];
    try {
      const raw = [INITIAL_MANIFEST_NAME, datasetName].filter(Boolean).map((key) => localStorage.getItem(`sad-eovsa:correlation-target:${key}`)).find(Boolean);
      const parsed = raw ? JSON.parse(raw) : [];
      if (Array.isArray(parsed)) storedTarget = parsed as [number, number][];
    } catch { /* storage is optional */ }
    const restoredTarget = loaded?.correlationTarget?.length ? loaded.correlationTarget : storedTarget;
    setCorrelationTarget(restoredTarget);
    setCorrelationCardOpen(Boolean(ui.correlationCardOpen) && restoredTarget.length >= 3);
    setCorrelationCardDismissed(false);
    setCorrelationFullRange(Boolean(ui.correlationFullRange));
    setCorrelationPinTicks(Boolean(ui.correlationPinTicks));
    setCorrelationFullHeightTicks(Boolean(ui.correlationFullHeightTicks));
    setCorrelationPeakDecel(Boolean(ui.correlationPeakDecel));
    // MIGRATION (see normalizeSlits' docstring): a legacy slit's on-disk
    // pixel `width` needs its bound source's pixel scale to become
    // widthArcsec. Built here from `data`/`ui.sourceRoles` directly (NOT the
    // component's own sourceAffine/radioSourceId closures above, which still
    // reflect the PREVIOUS session - setMeta/setSourceRoles just queued for
    // this render, not yet committed) so the conversion always uses the
    // session actually being loaded.
    const loadedRadioSourceId = roleSource(loadedSources, ui.sourceRoles ?? {}, "radio")?.id ?? "radio";
    const scaleForLoadedSlitSource = (sourceId: string): number => affinePixelScaleArcsec(
      sourceId === loadedRadioSourceId ? data.wcs.eovsa.pixelToWorldAffine : data.wcs.aia.pixelToWorldAffine
    );
    const restoredSlits = normalizeSlits(loaded?.slits, scaleForLoadedSlitSource);
    setSlits(restoredSlits);
    const restoredFan = normalizeFan(loaded?.fan);
    setFan(restoredFan);
    setFanIntermediateCount(restoredFan?.intermediateCount ?? 3);
    setSelectedFanMember(clamp(Math.round(Number(ui.selectedFanMember ?? 0)), -1, (restoredFan?.intermediateCount ?? 0) + 1));
    setFanDrawStage(0);
    setFanBoundaryDraft(null);
    setFanRedrawTarget(null);
    const restoredSelectedSlitId = restoredSlits.some((slit) => slit.id === ui.selectedSlitId)
      ? String(ui.selectedSlitId)
      : restoredSlits[0]?.id ?? "";
    setSelectedSlitId(restoredSelectedSlitId);
    setSlitSourceId(ui.slitSourceId && restoredTrackingSources.has(ui.slitSourceId)
      ? ui.slitSourceId
      : restoredSlits.find((slit) => slit.id === restoredSelectedSlitId)?.sourceId ?? defaultTrackingSource);
    const restoredSelectedSlit = restoredSlits.find((slit) => slit.id === restoredSelectedSlitId);
    setSlitDraftWidthArcsec(restoredSelectedSlit ? resolveSlitGeometry(restoredSelectedSlit, restoredSlits).widthArcsec : DEFAULT_SLIT_WIDTH_ARCSEC);
    setSlitDraftSmoothPx(clamp(numberValue(String(ui.slitSmoothPx ?? DEFAULT_SLIT_SMOOTH_PX), DEFAULT_SLIT_SMOOTH_PX), 0, SLIT_SMOOTH_MAX_PX));
    setSlitInspectorOpen(Boolean(ui.slitInspectorOpen));
    setSlitPinLane(Boolean(ui.slitPinLane));
    setSlitExtractAllMode(Boolean(ui.slitExtractAll));
    setSlitDrawArmed(false);
    setSlitResults({});
    setFloatingCardSizes(ui.floatingCardSizes ?? {});
    const rawRecordingOptions = ui.recordingOptions ?? {};
    setRecordingOptions({
      source: rawRecordingOptions.source === "left" || rawRecordingOptions.source === "right" || rawRecordingOptions.source === "workspace"
        ? rawRecordingOptions.source
        : DEFAULT_RECORDING_OPTIONS.source,
      range: rawRecordingOptions.range === "master" ? "master" : "visible",
      fps: clamp(Math.round(Number(rawRecordingOptions.fps) || DEFAULT_RECORDING_OPTIONS.fps), 1, 60),
      // Migrates the pre-rename "onscreen" value (and anything else unrecognized) to "native" - same option, honest name.
      resolution: rawRecordingOptions.resolution === "2x" ? "2x" : "native",
      burnTimestamp: Boolean(rawRecordingOptions.burnTimestamp),
      // Sessions saved before the Format dropdown existed have no formatChoice at all - default to "auto",
      // which reproduces their previous (only) behavior exactly.
      formatChoice: rawRecordingOptions.formatChoice === "mp4" || rawRecordingOptions.formatChoice === "webm"
        ? rawRecordingOptions.formatChoice
        : "auto"
    });
    setTargetDrawArmed(false);
    if (restoredTarget.length >= 3 && datasetName) {
      try { localStorage.setItem(`sad-eovsa:correlation-target:${datasetName}`, JSON.stringify(restoredTarget)); } catch { /* storage is optional */ }
    }
    const restoredTracks = normalizeTracks(
      loaded?.tracks ?? loaded?.sadTracks ?? loaded?.featureTracks,
      loadedSources.find((source) => sourceRole(source, ui.sourceRoles ?? {}) === "context")?.id ?? "context"
    );
    setSadTracks(restoredTracks);
    const restoredSelectedTrackId = restoredTracks.some((track) => track.id === ui.selectedTrackId) ? String(ui.selectedTrackId) : "";
    setSelectedTrackId(restoredSelectedTrackId);
    setTrackEditorOpen(Boolean(ui.trackEditorOpen));
    setTrackingDirection(ui.trackingDirection === "forward" || ui.trackingDirection === "backward" ? ui.trackingDirection : "both");
    setSelectedAnchorFrame(null);
    setEovsaSources(loaded?.eovsaSources ?? loaded?.radioSources ?? []);
    setPlaying(false);
  }

  function serializeSessionState(currentMeta: SessionMeta): Record<string, unknown> {
    const canonicalLayers = [...panelLayers.left, ...panelLayers.right];
    const canonicalRadio = canonicalLayers
      .map((layer) => resolvedLayer(layer, canonicalLayers))
      .find((layer) => layer.kind === "image" && layer.sourceRoleSnapshot === "radio");
    return {
      version: 2,
      sources: sources.map(slimSourceForSession),
      channelOffsets,
      data: {
        aiaIntensity: currentMeta.paths.aiaIntensity,
        aiaDiff: currentMeta.paths.aiaDiff,
        seeds: currentMeta.paths.seeds,
        eovsaFits: currentMeta.paths.eovsaFits,
        eovsaSpectrogram: currentMeta.paths.eovsaSpectrogram
      },
      ui: {
        selectedSourceId: selectedSource?.id ?? "",
        sourceRoles,
        sourceDifferences,
        layers: Object.fromEntries(
          normalizeLayerOrder(canonicalLayers)
            .filter((layer, index, all) => all.findIndex((candidate) => candidate.id === layer.id) === index)
            .map((layer) => [layer.id, serializeLayer(layer, resolvedLayer(layer, canonicalLayers))])
        ),
        panels: {
          left: { baseLayerId: panelCompositions.left.baseLayerId, overlayLayerIds: panelCompositions.left.overlayLayerIds },
          right: { baseLayerId: panelCompositions.right.baseLayerId, overlayLayerIds: panelCompositions.right.overlayLayerIds }
        },
        timeline: { masterSourceId, cursorMjd: currentMjd, startMjd: masterMin, endMjd: masterMax },
        timeIndex,
        startIndex: closestIndex(currentMeta.aia.timeMjd, masterMin),
        endIndex: closestIndex(currentMeta.aia.timeMjd, masterMax),
        freqIndex,
        diffSeconds: canonicalRadio?.cadenceSeconds ?? radioDifference.cadenceSeconds,
        useRunningDiff: canonicalRadio ? canonicalRadio.operation !== "none" && canonicalRadio.reference === "previous" : radioDifference.operation !== "none" && radioDifference.reference === "previous",
        xOffsetArcsec: xOffset,
        yOffsetArcsec: yOffset,
        solarView,
        layout,
        aiaDisplay,
        eovsaDisplay,
        spectrogramDisplay,
        spectrogramNormalization,
        spectrogramFrequencyScale,
        spectrogramFrequencyInverted,
        spectrogramFrequencyRange,
        playbackFps,
        timeStepSeconds,
        smoothPlayback: smoothPlaybackEnabled,
        showCacheCoverage,
        frameCacheGb,
        collapsedSections,
        trackEditorOpen,
        trackingSourceId,
        selectedTrackId,
        trackingDirection,
        correlationCardOpen,
        correlationFullRange,
        correlationPinTicks,
        correlationFullHeightTicks,
        correlationPeakDecel,
        slitInspectorOpen,
        selectedSlitId,
        slitSourceId,
        slitPinLane,
        slitExtractAll: slitExtractAllMode,
        slitSmoothPx: slitDraftSmoothPx,
        selectedFanMember,
        floatingCardSizes,
        recordingOptions
      },
      roiWorld,
      roiAia,
      roiEovsa,
      correlationTarget,
      slits,
      fan,
      radioSources: eovsaSources,
      tracks: sadTracks,
      eovsaSources
    };
  }

  async function postLoadedSession(state: Record<string, unknown>): Promise<LoadSessionResponse> {
    const isSessionJson = Boolean(state.ui && typeof state.ui === "object");
    const endpoint = isSessionJson ? "/api/sessions/load-json" : "/api/sessions/load-manifest";
    return apiJson<LoadSessionResponse>(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state)
    });
  }

  async function restoreSessionAfterRestart(): Promise<string> {
    const existing = sessionRestoreRef.current;
    if (existing) return existing;
    const restore = (async () => {
      setLoading(true);
      setMessage("Restoring session after backend restart...");
      try {
        if (!meta) throw new Error("No active session is available to restore.");
        const data = await postLoadedSession(serializeSessionState(meta));
        applyLoadedSession(data);
        restoredSessionIdRef.current = data.sessionId;
        setMessage("Session restored after backend restart");
        return data.sessionId;
      } catch (error) {
        reportError(error, "Could not restore session after backend restart");
        throw error;
      } finally {
        setLoading(false);
      }
    })();
    sessionRestoreRef.current = restore;
    restore.catch(() => {
      if (sessionRestoreRef.current === restore) sessionRestoreRef.current = null;
    }).catch(() => undefined);
    return restore;
  }

  async function loadJsonState(state: Record<string, unknown>, label: string) {
    if (!confirmDatasetReplace()) return;
    setLoading(true);
    setMessage(`Loading ${label}...`);
    try {
      const data = await postLoadedSession(state);
      applyLoadedSession(data);
      setDatasetName(recordingDatasetToken(label));
      const isSessionJson = Boolean(state.ui && typeof state.ui === "object");
      setMessage(`Loaded ${isSessionJson ? "JSON session" : "manifest"} with ${sourcesForMeta(data).length} data sources.`);
    } catch (error) {
      reportError(error);
    } finally {
      setLoading(false);
    }
  }

  async function loadJsonFile(file: File) {
    try {
      await loadJsonState(JSON.parse(await file.text()), file.name);
    } catch (error) {
      reportError(error);
    }
  }

  useEffect(() => {
    if (!INITIAL_MANIFEST_JSON || initialManifestRequested) return;
    initialManifestRequested = true;
    try {
      void loadJsonState(JSON.parse(INITIAL_MANIFEST_JSON), INITIAL_MANIFEST_NAME ?? "launch manifest");
    } catch (error) {
      reportError(error, "Could not parse launch manifest");
    }
  }, []);

  async function saveSessionJson() {
    if (!meta) return;
    let latestMeta = meta;
    try {
      latestMeta = await apiJson<SessionMeta>(`/api/sessions/${sessionId}/meta`, undefined, restoreSessionAfterRestart);
      setMeta(latestMeta);
    } catch {
      // Save the current UI state even if the cache metadata refresh fails.
    }
    const state = serializeSessionState(latestMeta);
    const blob = new Blob([JSON.stringify(state)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `solradviewer_session_${meta.aia.times[timeIndex] ?? "state"}.json`;
    link.click();
    URL.revokeObjectURL(url);
    baselineSignatureRef.current = dirtySignature;
    setDirty(false);
  }

  async function refreshRoiProjection(targetPanel: PanelId, values: RoiProjectionValues = {}, sourceOverride?: string, freqOverride?: number, roleOverride?: SourceRole) {
    if (!meta) return;
    const sourceId = sourceOverride ?? (targetPanel === "aia" ? contextSourceId : radioSourceId);
    const role = roleOverride ?? sourceRole(sources.find((source) => source.id === sourceId), sourceRoles);
    const nativeAxis = role === "radio" ? meta.eovsa.timeMjd : meta.aia.timeMjd;
    const roiOffset = role === "radio" ? radioWorldOffset(freqOverride ?? freqIndex) : [numberValue(values.xOffset ?? xOffset, 7), numberValue(values.yOffset ?? yOffset, 0)];
    const response = await apiJson<{ points: [number, number][] }>(`/api/sessions/${sessionId}/roi/projection`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        panel: targetPanel,
        sourceId,
        sampleMjd: currentMjd,
        samplingPolicy: "nearest",
        maxOffsetSeconds: nativeCadenceSeconds(nativeAxis) / 2,
        freqIndex: freqOverride ?? freqIndex,
        xOffsetArcsec: roiOffset[0],
        yOffsetArcsec: roiOffset[1],
        diffSeconds: numberValue(values.diffSeconds ?? radioDifference.cadenceSeconds, DEFAULT_DIFF_SECONDS)
      })
    });
    if (targetPanel === "aia") setRoiAia(response.points);
    else setRoiEovsa(response.points);
  }

  async function refreshBothRoiProjections(values: RoiProjectionValues = {}) {
    if (!meta) return;
    try {
      await Promise.all([refreshRoiProjection("aia", values), refreshRoiProjection("eovsa", values)]);
    } catch (error) {
      reportError(error);
    }
  }

  function playbackTimeAtOffset(mjd: number, offset: number, startMjd: number, endMjd: number): number {
    let next = mjd;
    const direction = offset < 0 ? -1 : 1;
    for (let step = 0; step < Math.abs(offset); step += 1) {
      next = advanceFrameTime(masterTimes, next, direction, timeStepSeconds, startMjd, endMjd, true, masterMin);
    }
    return next;
  }

  // Same identity builder used for the on-screen render, prefetch/lookahead,
  // cache warming, and the cache-coverage bar. `capsOverride` defaults to the
  // raw zoom-derived panelFrameCaps (full-res) so every existing caller
  // (warmSettingsSignature, warmVisibleRange, the coverage bar) is unaffected;
  // only the playback/scrub prefetch effects pass the motion (half-res) caps.
  function requestsForMasterTime(sampleMjd: number, capsOverride: Record<PanelSlotId, FrameRequestCap> = panelFrameCaps): ScheduledFrameRequest[] {
    if (!meta) return [];
    if (!Number.isFinite(sampleMjd)) return [];
    const requests: ScheduledFrameRequest[] = [];
    for (const [slot, layers] of [["left", panelLayers.left], ["right", panelLayers.right]] as [PanelSlotId, LayerState[]][]) {
      const composition = compositionFor(layers, panelCompositions[slot]?.baseLayerId, panelCompositions[slot]?.overlayLayerIds);
      const targetLayer = layers.find((layer) => layer.id === composition.baseLayerId);
      const effectiveTarget = targetLayer ? resolvedLayer(targetLayer, allPanelLayers) : undefined;
      const targetSourceId = effectiveTarget?.sourceId ?? contextSourceId;
      const targetRole = effectiveTarget?.sourceRoleSnapshot;
      const effectiveLayers = layers.map((layer) => resolvedLayer(layer, allPanelLayers));
      const fullCube = effectiveLayers.some((layer) => layer.visible && layer.kind === "contours");
      for (const effectiveLayer of effectiveLayers) {
        if (!effectiveLayer.visible) continue;
        const request = requestForLayer(
          effectiveLayer,
          sampleMjd,
          targetSourceId,
          fullCube,
          targetRole,
          capsOverride[slot]
        );
        if (request) requests.push(request);
      }
    }
    return requests.sort((left, right) => {
      if (left.kind === right.kind) return 0;
      return left.kind === "image" ? -1 : 1;
    });
  }

  function prioritizedRequestsForMasterTime(mjd: number, distance: number, directionBias: number, capsOverride?: Record<PanelSlotId, FrameRequestCap>): ScheduledFrameRequest[] {
    return requestsForMasterTime(mjd, capsOverride).map((request) => ({
      ...request,
      priority: { distance, directionBias }
    }));
  }

  const warmWindow = timeRangeValues(spectrogramTimeRange, meta?.spectrogram.timeMjd ?? []);
  const warmWindowSignature = warmWindow.map((value) => value.toPrecision(15)).join(":");
  const warmSettingsSignature = meta ? JSON.stringify(
    [...new Set(requestsForMasterTime(currentMjd).map(frameRequestGroupKey))].sort()
  ) : "";
  const warmStaleReason: "window" | "settings" | null = warmCacheRecord?.sessionId === sessionId
    ? warmCacheRecord.settingsSignature !== warmSettingsSignature
      ? "settings"
      : warmCacheRecord.windowSignature !== warmWindowSignature
        ? "window"
        : null
    : null;
  const warmCacheTitle = warmCacheStatus.active
    ? `Cancel cache warming (${warmCacheStatus.done}/${warmCacheStatus.total})`
    : warmStaleReason === "settings"
      ? "Warm cache for the visible time range — settings changed; full re-warm required"
      : warmStaleReason === "window"
        ? "Warm cache for the visible time range — window changed; click to rebuild"
        : warmCacheStatus.limited && warmCacheRecord?.sessionId === sessionId
          ? `Window exceeds cache budget — warmed nearest ${warmCacheStatus.targetFrames} frames`
          : "Warm cache for the visible time range";

  async function warmVisibleRange() {
    if (warmCacheStatus.active) {
      // Cancel-and-clear: sever the ref and zero the progress UI synchronously
      // so a subsequent click can never be blocked, clobbered, or have its
      // fresh plan mixed with stragglers from the run being canceled below.
      const canceledController = warmCacheControllerRef.current;
      warmCacheControllerRef.current = null;
      canceledController?.abort();
      setMessage(`Cache warming canceled at ${warmCacheStatus.done}/${warmCacheStatus.total}.`);
      setWarmCacheStatus({
        active: false,
        done: 0,
        total: 0,
        retained: 0,
        requestedFrames: 0,
        targetFrames: 0,
        limited: false,
        startedAt: 0
      });
      return;
    }
    if (!meta || !masterTimes.length) return;
    const [windowMin, windowMax] = warmWindow;
    const orderedIndices = masterTimes
      .map((mjd, index) => ({ mjd, index }))
      .filter(({ mjd }) => Number.isFinite(mjd) && mjd >= windowMin && mjd <= windowMax)
      .sort((left, right) => (
        Math.abs(left.mjd - currentMjd) - Math.abs(right.mjd - currentMjd)
        || left.index - right.index
      ));
    const plan = planFrameCacheWarm(orderedIndices.map(({ mjd }) => requestsForMasterTime(mjd)));
    const startedAt = Date.now() / 1000;
    setWarmCacheStatus({
      active: plan.missingIdentities > 0,
      done: 0,
      total: plan.missingIdentities,
      retained: plan.retainedIdentities,
      requestedFrames: plan.requestedFrames,
      targetFrames: plan.targetFrames,
      limited: plan.limited,
      startedAt
    });
    if (!plan.missingIdentities) {
      setWarmCacheRecord({ sessionId, windowSignature: warmWindowSignature, settingsSignature: warmSettingsSignature });
      setMessage(plan.limited
        ? `Visible cache already warm for the nearest ${plan.targetFrames} frames allowed by the budget.`
        : `Visible cache already warm (${plan.retainedIdentities} retained identities; 0 missing).`);
      return;
    }

    const controller = new AbortController();
    warmCacheControllerRef.current = controller;
    // A prior run only reaches its own completion/cleanup code after this
    // point if it was still in flight when canceled above; isCurrent() keeps
    // any such straggler from writing progress/messages/records that belong
    // to this fresh plan.
    const isCurrent = () => warmCacheControllerRef.current === controller;
    let done = 0;
    try {
      // Tell the backend these requests are cache prewarm work.  It keeps the
      // atomic tempfile+replace cache semantics but avoids an fsync per PNG;
      // interactive renders retain durable writes.
      const warmRequests = plan.requests.map((request) => ({
        ...request,
        url: `${request.url}${request.url.includes("?") ? "&" : "?"}warm=1`
      }));
      await warmFrameCache(
        warmRequests,
        controller.signal,
        () => {
          done += 1;
          if (isCurrent()) setWarmCacheStatus((current) => ({ ...current, done }));
        },
        restoreSessionAfterRestart
      );
      if (controller.signal.aborted) {
        if (isCurrent()) setMessage(`Cache warming canceled at ${done}/${plan.missingIdentities}.`);
        return;
      }
      if (isCurrent()) {
        setWarmCacheRecord({ sessionId, windowSignature: warmWindowSignature, settingsSignature: warmSettingsSignature });
        setMessage(plan.limited
          ? `Warmed ${done} missing identities nearest the cursor across ${plan.targetFrames} frames; window exceeds the cache budget.`
          : `Warmed ${done} missing identities; retained ${plan.retainedIdentities} already-cached identities.`);
      }
    } catch (error) {
      if (controller.signal.aborted) {
        if (isCurrent()) setMessage(`Cache warming canceled at ${done}/${plan.missingIdentities}.`);
      } else {
        reportError(error, "Could not warm the visible cache");
      }
    } finally {
      if (isCurrent()) {
        warmCacheControllerRef.current = null;
        setWarmCacheStatus((current) => ({ ...current, active: false, done }));
      }
    }
  }

  useEffect(() => {
    if (!meta || !timeScrubbing || !masterTimes.length) return undefined;
    const controller = new AbortController();
    const lower = Math.min(closestIndex(masterTimes, masterMin), closestIndex(masterTimes, masterMax));
    const upper = Math.max(closestIndex(masterTimes, masterMin), closestIndex(masterTimes, masterMax));
    const timeSliderIndex = closestIndex(masterTimes, timeSliderMjd);
    const indices = [
      timeSliderIndex,
      ...scrubPrefetchOffsets(scrubDirectionRef.current).map((offset) => clamp(timeSliderIndex + offset, lower, upper))
    ].filter((index, position, all) => all.indexOf(index) === position);
    const direction = scrubDirectionRef.current < 0 ? -1 : 1;
    // Scrub prefetch issues half-res requests while the pointer is down (the
    // motion resolution ladder); the on-screen preview still shows whatever
    // is nearest-cached first via scrubPanelRenderLayers/nearestCachedFrame.
    const requests = indices.flatMap((index) => prioritizedRequestsForMasterTime(
      masterTimes[index],
      Math.abs(index - timeSliderIndex),
      (index - timeSliderIndex) * direction >= 0 ? 0 : 1,
      motionPanelFrameCaps
    ));
    void prefetchFrames(requests, controller.signal, restoreSessionAfterRestart);
    return () => controller.abort();
  }, [
    meta,
    sessionId,
    masterSourceId,
    masterTimes,
    masterMin,
    masterMax,
    timeScrubbing,
    timeSliderMjd,
    warmSettingsSignature,
    motionPanelFrameCaps
  ]);

  useEffect(() => {
    if (!meta || !playing || timeScrubbing || masterTimes.length < 2) {
      return undefined;
    }
    const intervalMs = 1000 / playbackFps;
    let nextDeadline = performance.now() + intervalMs;
    let playbackCursor = currentMjd;
    let timer = 0;
    const advance = () => {
      const now = performance.now();
      const elapsedSteps = Math.max(1, Math.floor((now - nextDeadline) / intervalMs) + 1);
      playbackCursor = playbackTimeAtOffset(playbackCursor, elapsedSteps, masterMin, masterMax);
      setMasterCursor(playbackCursor);
      nextDeadline += elapsedSteps * intervalMs;
      timer = window.setTimeout(advance, Math.max(0, nextDeadline - performance.now()));
    };
    timer = window.setTimeout(advance, intervalMs);
    return () => window.clearTimeout(timer);
  }, [masterMax, masterMin, masterTimes, meta, playbackFps, playing, timeScrubbing, timeStepSeconds]);

  useEffect(() => {
    if (!meta || timeScrubbing || (!playing && !panelReady)) {
      playbackPrefetchGroupsRef.current = [];
      playbackImminentRequestsRef.current = [];
      playbackPrefetchProgressRef.current = () => undefined;
      setPinnedFrameRequests([]);
      return undefined;
    }
    const rangeStartIndex = closestIndex(masterTimes, masterMin);
    const rangeEndIndex = closestIndex(masterTimes, masterMax);
    const direction = playing ? 1 : playbackDirectionRef.current || 1;
    const candidates = playing
      ? Array.from(
          { length: clamp(Math.ceil(playbackFps), 4, PLAYBACK_LOOKAHEAD_MAX_FRAMES) },
          (_, offset) => ({
            mjd: playbackTimeAtOffset(currentMjd, direction * (offset + 1), masterMin, masterMax),
            distance: offset + 1,
            directionBias: 0
          })
        )
      : Array.from({ length: PREFETCH_RADIUS }, (_, index) => index + 1)
          .flatMap((distance) => [
            { offset: direction * distance, distance, directionBias: 0 },
            { offset: -direction * distance, distance, directionBias: 1 }
          ])
          .map((candidate) => ({
            ...candidate,
            mjd: masterTimes[clamp(masterCursorIndex + candidate.offset, Math.min(rangeStartIndex, rangeEndIndex), Math.max(rangeStartIndex, rangeEndIndex))]
          }))
          .filter((candidate) => candidate.mjd !== currentMjd);
    const uniqueCandidates = candidates.filter((candidate, position, all) => (
      all.findIndex((other) => other.mjd === candidate.mjd) === position
    ));
    // Motion resolution ladder: playback lookahead/imminent requests (and the
    // on-screen position pinned alongside them) use the half-res motion caps.
    // The non-playing branch below is the idle radius prefetch around a
    // parked cursor - not motion, so it keeps requesting full-res.
    const capsForPrefetch = playing ? motionPanelFrameCaps : panelFrameCaps;
    const requestGroups = uniqueCandidates.map((candidate) => prioritizedRequestsForMasterTime(
      candidate.mjd,
      candidate.distance,
      candidate.directionBias,
      capsForPrefetch
    ));
    const updateBuffer = () => {
      if (!playing) return;
      const groups = playbackPrefetchGroupsRef.current;
      const ahead = groups.filter((requests) => requests.length > 0 && areFramesCached(requests)).length;
      const total = groups.length;
      setPlaybackBuffer((current) => current.ahead === ahead && current.total === total ? current : { ahead, total });
    };
    if (playing) {
      const currentRequests = requestsForMasterTime(currentMjd, motionPanelFrameCaps);
      playbackPrefetchGroupsRef.current = requestGroups;
      // The identities for the position already on screen need the same
      // imminence priority as the next lookahead frame's - otherwise a
      // slower-to-resolve kind (contour geometry vs. the base frame.png)
      // never wins a reserved slot until it is already stale, and playback
      // is left drawing whatever contour last happened to land. Distance 0
      // ranks the on-screen position ahead of the distance-1 lookahead frame
      // in the imminent queue's priority sort.
      const currentImminentRequests = currentRequests.map((request) => ({
        ...request,
        priority: { distance: 0, directionBias: 0 }
      }));
      playbackImminentRequestsRef.current = [...currentImminentRequests, ...(requestGroups[0] ?? [])];
      playbackPrefetchProgressRef.current = updateBuffer;
      setPinnedFrameRequests(
        [...currentRequests, ...requestGroups.slice(0, PLAYBACK_PINNED_FRAME_COUNT).flat()],
        currentRequests
      );
      updateBuffer();
      return undefined;
    } else {
      playbackPrefetchGroupsRef.current = [];
      playbackImminentRequestsRef.current = [];
      playbackPrefetchProgressRef.current = () => undefined;
      setPinnedFrameRequests([]);
      setPlaybackBuffer((current) => current.ahead === 0 && current.total === 0 ? current : { ahead: 0, total: 0 });
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void prefetchFrames(requestGroups.flat(), controller.signal, restoreSessionAfterRestart);
    }, PREFETCH_IDLE_MS);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [
    meta,
    sessionId,
    timeIndex,
    masterSourceId,
    currentMjd,
    masterCursorIndex,
    masterTimes,
    masterMin,
    masterMax,
    timeScrubbing,
    panelReady,
    playing,
    playbackFps,
    timeStepSeconds,
    warmSettingsSignature,
    motionPanelFrameCaps
  ]);

  useEffect(() => {
    if (!meta || !playing || timeScrubbing) return undefined;
    const controller = new AbortController();
    const imminentController = new AbortController();
    void prefetchFrames(
      () => playbackImminentRequestsRef.current,
      imminentController.signal,
      restoreSessionAfterRestart,
      () => playbackPrefetchProgressRef.current(),
      "imminent"
    );
    void prefetchFrames(
      () => playbackPrefetchGroupsRef.current.flat(),
      controller.signal,
      restoreSessionAfterRestart,
      () => playbackPrefetchProgressRef.current()
    );
    return () => {
      controller.abort();
      imminentController.abort();
      setPinnedFrameRequests([]);
    };
  }, [meta, playing, sessionId, timeScrubbing]);

  async function saveRoi(panel: PanelId, points: [number, number][], sourceOverride?: string, freqOverride?: number, roleOverride?: SourceRole) {
    if (!meta) return;
    const sourceId = sourceOverride ?? (panel === "aia" ? contextSourceId : radioSourceId);
    const sourceRoleValue = roleOverride ?? sourceRole(sources.find((source) => source.id === sourceId), sourceRoles);
    const nativeAxis = sourceRoleValue === "radio" ? meta.eovsa.timeMjd : meta.aia.timeMjd;
    const roiOffset = sourceRoleValue === "radio" ? radioWorldOffset(freqOverride ?? freqIndex) : [numberValue(xOffset, 7), numberValue(yOffset, 0)];
    setMessage(`Saving ${panel.toUpperCase()} lasso ROI...`);
    try {
      const response = await apiJson<{ roiWorld: [number, number][] }>(`/api/sessions/${sessionId}/roi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
          panel,
          sourceId,
        points,
          sampleMjd: currentMjd,
          samplingPolicy: "nearest",
          maxOffsetSeconds: nativeCadenceSeconds(nativeAxis) / 2,
          freqIndex: freqOverride ?? freqIndex,
          xOffsetArcsec: roiOffset[0],
          yOffsetArcsec: roiOffset[1],
          diffSeconds: numberValue(radioDifference.cadenceSeconds, DEFAULT_DIFF_SECONDS)
        })
      });
      setRoiWorld(response.roiWorld);
      if (panel === "aia") setRoiAia(points);
      else setRoiEovsa(points);
      await refreshRoiProjection(panel, {}, sourceId, freqOverride, sourceRoleValue);
      await refreshBothRoiProjections();
      setMessage("ROI saved in Solar-X/Solar-Y coordinates.");
    } catch (error) {
      reportError(error);
    }
  }

  async function saveCorrelationTarget(points: [number, number][]) {
    if (!meta || points.length < 3) return;
    const world = points.map((point) => applyAffine(point, meta.wcs.aia.pixelToWorldAffine));
    setCorrelationTarget(world);
    setTargetDrawArmed(false);
    try {
      await apiJson<{ correlationTarget: [number, number][] }>(`/api/sessions/${sessionId}/correlation-target`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: world })
      }, restoreSessionAfterRestart);
      setMessage("Loop-top target saved in solar arcseconds.");
    } catch (error) {
      reportError(error, "Could not save loop-top target");
    }
  }

  function completeSlitDraw(
    curveArcsec: [number, number][],
    inputVertexCount: number,
    rawCurveArcsec: [number, number][],
    drawnPanel: PanelId
  ) {
    setSlitDrawArmed(false);
    if (curveArcsec.length < 2) return;
    const binding = slitSourceOptions.find((option) => option.sourceId === slitSourceId);
    if (!binding) {
      setMessage("Choose a default slit binding before drawing.");
      return;
    }
    const id = `slit-${crypto.randomUUID().slice(0, 8)}`;
    const radioChannel = binding.layer.sourceRoleSnapshot === "radio" ? binding.layer.freqIndex : null;
    const slit: SlitDefinition = {
      id,
      name: `Slit ${slits.length + 1}`,
      color: TRACK_COLORS[slits.length % TRACK_COLORS.length],
      visible: true,
      sourceId: binding.sourceId,
      layerId: binding.layer.id,
      bindingKind: binding.bindingKind,
      widthArcsec: slitDraftWidthArcsec,
      shiftSeconds: 0,
      curveArcsec,
      inputVertexCount,
      rawCurveArcsec: rawCurveArcsec.length >= 2 ? rawCurveArcsec : undefined,
      smoothPx: slitDraftSmoothPx,
      drawnPanel,
      freqIndices: radioChannel === null ? [] : [radioChannel],
      baseFreqIndex: radioChannel,
      contourFreqIndices: [],
      contourLevelPercent: 70,
      display: { vmin: 0, vmax: 1, cmap: "magma", scale: "linear" }
    };
    setSlits((current) => [...current, slit]);
    setSelectedSlitId(id);
    setMessage(`Saved ${slit.name}: ${inputVertexCount} hand points → ${curveArcsec.length} smoothed samples.`);
  }

  function completeFanBoundary(
    curveArcsec: [number, number][],
    inputVertexCount: number,
    rawCurveArcsec: [number, number][],
    drawnPanel: PanelId
  ) {
    if (curveArcsec.length < 2) return;
    if (fanDrawStage === 1 || !fanBoundaryDraft) {
      setFanBoundaryDraft({ curve: curveArcsec, inputVertexCount, rawCurveArcsec, drawnPanel });
      setFanDrawStage(2);
      setMessage("Fan boundary A saved. Draw boundary B on either image panel; Esc cancels.");
      return;
    }
    const binding = slitSourceOptions.find((option) => option.sourceId === slitSourceId);
    if (!binding) {
      setMessage("Choose a default slit binding before drawing.");
      return;
    }
    const built = createFanDefinition(
      binding,
      fanBoundaryDraft.curve,
      curveArcsec,
      [fanBoundaryDraft.inputVertexCount, inputVertexCount],
      fanIntermediateCount
    );
    if (!built) return;
    setFan({
      ...built.fan,
      rawBoundaryA: fanBoundaryDraft.rawCurveArcsec.length >= 2 ? fanBoundaryDraft.rawCurveArcsec : undefined,
      rawBoundaryB: rawCurveArcsec.length >= 2 ? rawCurveArcsec : undefined,
      boundaryADrawnPanel: fanBoundaryDraft.drawnPanel,
      boundaryBDrawnPanel: drawnPanel,
      smoothPx: slitDraftSmoothPx
    });
    setSelectedFanMember(0);
    setFanDrawStage(0);
    setFanBoundaryDraft(null);
    setMessage(`Fan saved: ${built.fan.intermediateCount + 2} curves × ${built.fan.boundaryA.length} samples${built.reversedBoundaryB ? "; boundary B auto-aligned" : ""}.`);
  }

  function armFanDrawing() {
    if (!slitSourceId) return;
    setSlitDrawArmed(false);
    setFanDrawStage(1);
    setFanBoundaryDraft(null);
    setFanRedrawTarget(null);
    setTargetDrawArmed(false);
    setSeedMode(false);
    setLassoEnabled(false);
    setChannelLassoArmed(false);
    closePixelProbe();
    setMessage("Draw fan boundary A on either image panel.");
  }

  // Arms a single-stroke redraw of one existing fan boundary. The next
  // completed stroke on either panel replaces only that boundary; the fan
  // family regenerates in place (same intermediateCount) and any promoted
  // slits derived from this fan have their geometry updated -- see
  // completeFanRedraw().
  function armFanRedraw(target: "A" | "B") {
    if (!fan) return;
    setFanRedrawTarget(target);
    setFanDrawStage(0);
    setFanBoundaryDraft(null);
    setSlitDrawArmed(false);
    setTargetDrawArmed(false);
    setSeedMode(false);
    setLassoEnabled(false);
    setChannelLassoArmed(false);
    closePixelProbe();
    setMessage(`Draw the replacement for boundary ${target} on either image panel; Esc cancels.`);
  }

  function findFanBinding(current: FanDefinition): SlitSourceOption | undefined {
    return slitSourceOptions.find((option) => (
      option.sourceId === current.sourceId && option.bindingKind === current.bindingKind
    )) ?? slitSourceOptions.find((option) => option.sourceId === current.sourceId);
  }

  function buildFanMemberSlit(current: FanDefinition, memberIndex: number, curve: [number, number][], binding: SlitSourceOption): SlitDefinition {
    const radioChannel = binding.layer.sourceRoleSnapshot === "radio" ? binding.layer.freqIndex : null;
    return {
      id: `slit-${crypto.randomUUID().slice(0, 8)}`,
      name: `Fan curve ${memberIndex + 1}`,
      color: memberIndex === 0 || memberIndex === current.intermediateCount + 1 ? "#ff5b5b" : "#53d769",
      visible: true,
      sourceId: current.sourceId,
      layerId: current.layerId,
      bindingKind: current.bindingKind,
      widthArcsec: slitDraftWidthArcsec,
      shiftSeconds: 0,
      curveArcsec: curve.map((point) => [...point] as [number, number]),
      inputVertexCount: curve.length,
      freqIndices: radioChannel === null ? [] : [radioChannel],
      baseFreqIndex: radioChannel,
      contourFreqIndices: [],
      contourLevelPercent: 70,
      display: { vmin: 0, vmax: 1, cmap: "magma", scale: "linear" }
    };
  }

  function promoteFanMember(memberIndex: number) {
    if (!fan) return;
    const curve = fanFamilyCurves(fan)[memberIndex];
    if (!curve) return;
    const existingId = fan.promotedMembers[String(memberIndex)];
    if (existingId && slits.some((slit) => slit.id === existingId)) {
      setSelectedSlitId(existingId);
      return;
    }
    const binding = findFanBinding(fan);
    if (!binding) return;
    const slit = buildFanMemberSlit(fan, memberIndex, curve, binding);
    setSlits((current) => [...current, slit]);
    setFan((current) => current ? { ...current, promotedMembers: { ...current.promotedMembers, [String(memberIndex)]: slit.id } } : current);
    setSelectedFanMember(memberIndex);
    setSelectedSlitId(slit.id);
    setMessage(`Promoted fan curve ${memberIndex + 1} to ${slit.name}.`);
  }

  // One-gesture "promote (if needed) and extract" for a fan family curve --
  // addresses discoverability: users could draw fan curves but not find how
  // to get a time-distance lane out of them.
  async function extractFanMember(memberIndex: number) {
    if (!fan) return;
    const curve = fanFamilyCurves(fan)[memberIndex];
    if (!curve) return;
    const existingId = fan.promotedMembers[String(memberIndex)];
    const existingSlit = existingId ? slits.find((slit) => slit.id === existingId) : undefined;
    if (existingSlit) {
      setSelectedFanMember(memberIndex);
      setSelectedSlitId(existingSlit.id);
      await extractSlit(existingSlit);
      return;
    }
    const binding = findFanBinding(fan);
    if (!binding) {
      setMessage("Choose a fan binding before extracting.");
      return;
    }
    const slit = buildFanMemberSlit(fan, memberIndex, curve, binding);
    setSlits((current) => [...current, slit]);
    setFan((current) => current ? { ...current, promotedMembers: { ...current.promotedMembers, [String(memberIndex)]: slit.id } } : current);
    setSelectedFanMember(memberIndex);
    setSelectedSlitId(slit.id);
    setMessage(`Promoted fan curve ${memberIndex + 1} to ${slit.name}; extracting...`);
    await extractSlit(slit);
  }

  // Replaces one boundary of the current fan with a freshly-drawn stroke
  // (already smoothed/jitter-trimmed and auto-oriented by the caller, same
  // as the two-stage fan creation flow). The fan family regenerates from the
  // new boundary pair at the same intermediateCount; promoted slits derived
  // from this fan keep their id/binding/width/shift but get their geometry
  // updated in place, and their cached extraction result is dropped so the
  // next Extract recomputes it.
  function completeFanRedraw(
    curveArcsec: [number, number][],
    inputVertexCount: number,
    rawCurveArcsec: [number, number][],
    drawnPanel: PanelId
  ) {
    const target = fanRedrawTarget;
    setFanRedrawTarget(null);
    if (!target || !fan || curveArcsec.length < 2) return;
    const binding = findFanBinding(fan);
    if (!binding) {
      setMessage(`Could not redraw boundary ${target}: its source binding is no longer available.`);
      return;
    }
    const boundaryAInput = target === "A" ? curveArcsec : fan.boundaryA;
    const boundaryBInput = target === "B" ? curveArcsec : fan.boundaryB;
    const newInputVertexCounts: [number, number] = [
      target === "A" ? inputVertexCount : fan.inputVertexCounts[0],
      target === "B" ? inputVertexCount : fan.inputVertexCounts[1]
    ];
    const built = createFanDefinition(binding, boundaryAInput, boundaryBInput, newInputVertexCounts, fan.intermediateCount);
    if (!built) {
      setMessage(`Redrawn boundary ${target} could not be resampled; fan left unchanged.`);
      return;
    }
    const rawCurveArcsecTrimmed = rawCurveArcsec.length >= 2 ? rawCurveArcsec : undefined;
    const updatedFan: FanDefinition = {
      ...built.fan,
      id: fan.id,
      sourceId: fan.sourceId,
      layerId: fan.layerId,
      bindingKind: fan.bindingKind,
      promotedMembers: fan.promotedMembers,
      rawBoundaryA: target === "A" ? rawCurveArcsecTrimmed : fan.rawBoundaryA,
      rawBoundaryB: target === "B" ? rawCurveArcsecTrimmed : fan.rawBoundaryB,
      boundaryADrawnPanel: target === "A" ? drawnPanel : fan.boundaryADrawnPanel,
      boundaryBDrawnPanel: target === "B" ? drawnPanel : fan.boundaryBDrawnPanel,
      smoothPx: fan.smoothPx ?? slitDraftSmoothPx
    };
    const newFamily = fanFamilyCurves(updatedFan);
    const promotedIds = new Set(Object.values(updatedFan.promotedMembers));
    setFan(updatedFan);
    if (promotedIds.size) {
      setSlits((current) => current.map((slit) => {
        if (!promotedIds.has(slit.id)) return slit;
        const memberIndexEntry = Object.entries(updatedFan.promotedMembers).find(([, slitId]) => slitId === slit.id);
        const memberIndex = memberIndexEntry ? Number(memberIndexEntry[0]) : -1;
        const curve = newFamily[memberIndex];
        if (!curve) return slit;
        return { ...slit, curveArcsec: curve.map((point) => [...point] as [number, number]), inputVertexCount: curve.length };
      }));
      setSlitResults((current) => {
        let changed = false;
        const next = { ...current };
        // Also drop any linked twin's cached results - its geometry is
        // resolved from the promoted slit whose curve just moved.
        const staleIds = new Set(promotedIds);
        for (const slit of slits) {
          if (slit.linkedTo && promotedIds.has(slit.linkedTo)) staleIds.add(slit.id);
        }
        for (const slitId of staleIds) {
          if (slitId in next) { delete next[slitId]; changed = true; }
        }
        return changed ? next : current;
      });
    }
    setSelectedFanMember((current) => clamp(current, -1, updatedFan.intermediateCount + 1));
    setMessage(`Boundary ${target} redrawn; fan family regenerated${built.reversedBoundaryB ? "; boundary B auto-aligned" : ""}.${promotedIds.size ? ` ${promotedIds.size} promoted slit${promotedIds.size === 1 ? "" : "s"} updated - Extract to refresh.` : ""}`);
  }

  // Per-fan re-smoothing (FEATURE 1): regenerates both boundaries from their
  // stored raw strokes (fan.rawBoundaryA/B) at a new smoothPx, then rebuilds
  // the whole family and updates promoted members exactly like
  // completeFanRedraw's tail does for a full boundary redraw - re-smoothing
  // is "redraw both boundaries with the same raw input, different sigma".
  // Disabled in the UI (SlitInspectorCard) when either raw boundary is
  // missing (legacy fan - see FanDefinition's docstring).
  function reSmoothFan(smoothPx: number) {
    if (!fan || !fan.rawBoundaryA || !fan.rawBoundaryB || fan.rawBoundaryA.length < 2 || fan.rawBoundaryB.length < 2) return;
    const binding = findFanBinding(fan);
    if (!binding) {
      setMessage("Could not re-smooth fan: its source binding is no longer available.");
      return;
    }
    const clampedSmoothPx = clamp(smoothPx, 0, SLIT_SMOOTH_MAX_PX);
    const scaleA = affinePixelScaleArcsec(panelAffine(fan.boundaryADrawnPanel ?? "aia"));
    const scaleB = affinePixelScaleArcsec(panelAffine(fan.boundaryBDrawnPanel ?? "aia"));
    const boundaryA = resmoothCurveFromRaw(fan.rawBoundaryA, clampedSmoothPx, scaleA, solarView);
    const boundaryB = resmoothCurveFromRaw(fan.rawBoundaryB, clampedSmoothPx, scaleB, solarView);
    const built = createFanDefinition(binding, boundaryA, boundaryB, fan.inputVertexCounts, fan.intermediateCount);
    if (!built) {
      setMessage("Re-smoothed boundaries could not be resampled; fan left unchanged.");
      return;
    }
    const updatedFan: FanDefinition = {
      ...built.fan,
      id: fan.id,
      sourceId: fan.sourceId,
      layerId: fan.layerId,
      bindingKind: fan.bindingKind,
      promotedMembers: fan.promotedMembers,
      rawBoundaryA: fan.rawBoundaryA,
      rawBoundaryB: fan.rawBoundaryB,
      boundaryADrawnPanel: fan.boundaryADrawnPanel,
      boundaryBDrawnPanel: fan.boundaryBDrawnPanel,
      smoothPx: clampedSmoothPx
    };
    const newFamily = fanFamilyCurves(updatedFan);
    const promotedIds = new Set(Object.values(updatedFan.promotedMembers));
    setFan(updatedFan);
    if (promotedIds.size) {
      setSlits((current) => current.map((slit) => {
        if (!promotedIds.has(slit.id)) return slit;
        const memberIndexEntry = Object.entries(updatedFan.promotedMembers).find(([, slitId]) => slitId === slit.id);
        const memberIndex = memberIndexEntry ? Number(memberIndexEntry[0]) : -1;
        const curve = newFamily[memberIndex];
        if (!curve) return slit;
        return { ...slit, curveArcsec: curve.map((point) => [...point] as [number, number]), inputVertexCount: curve.length };
      }));
      setSlitResults((current) => {
        let changed = false;
        const next = { ...current };
        // Also drop any linked twin's cached results - its geometry is
        // resolved from the promoted slit whose curve just moved.
        const staleIds = new Set(promotedIds);
        for (const slit of slits) {
          if (slit.linkedTo && promotedIds.has(slit.linkedTo)) staleIds.add(slit.id);
        }
        for (const slitId of staleIds) {
          if (slitId in next) { delete next[slitId]; changed = true; }
        }
        return changed ? next : current;
      });
    }
    setMessage(`Fan re-smoothed at ${clampedSmoothPx.toFixed(1)} px; family regenerated${promotedIds.size ? ` - ${promotedIds.size} promoted slit${promotedIds.size === 1 ? "" : "s"} updated, Extract to refresh.` : "."}`);
  }

  function setFanCurveCount(value: number) {
    const normalized = clamp(Math.round(value), 1, 20);
    setFanIntermediateCount(normalized);
    setFan((current) => current ? { ...current, intermediateCount: normalized } : current);
    setSelectedFanMember((current) => current < 0 ? current : clamp(current, 0, normalized + 1));
  }

  function deleteSlit(slitId: string) {
    const promotedMember = fan
      ? Object.entries(fan.promotedMembers).find(([, promotedId]) => promotedId === slitId)
      : undefined;
    // Deleting a plain twin is a plain deletion (nothing depends on it).
    // Deleting the ORIGINAL of one or more linked twins must not cascade-
    // delete the twins' own user data (name/binding/channels/results): each
    // dependent twin is materialized into an independent slit first, its
    // curveArcsec/inputVertexCount/width stamped from the original (still
    // present in `current` at this point, since filter/map below read the
    // same pre-filter array) so it keeps rendering/extracting exactly as it
    // did a moment ago.
    setSlits((current) => current
      .filter((slit) => slit.id !== slitId)
      .map((slit) => slit.linkedTo === slitId
        ? { ...slit, linkedTo: undefined, ...resolveSlitGeometry(slit, current) }
        : slit));
    setSlitResults((current) => {
      const next = { ...current };
      delete next[slitId];
      return next;
    });
    if (selectedSlitId === slitId) setSelectedSlitId("");
    if (!promotedMember) return;
    const memberIndex = Number(promotedMember[0]);
    setFan((current) => current ? {
      ...current,
      promotedMembers: Object.fromEntries(
        Object.entries(current.promotedMembers).filter(([, promotedId]) => promotedId !== slitId)
      )
    } : current);
    setSelectedFanMember((current) => current === memberIndex ? -1 : current);
  }

  // Clones a slit's geometry/binding (curve, width, shift, source) so a user
  // who already has an AIA-bound slit with an extracted TD map can get a
  // radio/contour TD map "from the same curve" without losing the AIA
  // result: the copy starts with the identical binding (rebind it via the
  // row's source dropdown - onSlitSourceChange/rebindSlit already lists any
  // contour overlay through slitSourceOptions) and no slitResults entry, so
  // it always needs its own Extract and never touches the original's cached
  // map.
  function duplicateSlit(slitId: string) {
    const original = slits.find((candidate) => candidate.id === slitId);
    if (!original) return;
    // Always materializes (resolveSlitGeometry) rather than copying
    // original.curveArcsec/width directly, so duplicating a linked twin
    // yields a plain independent copy of its current geometry instead of a
    // second twin pointed at the wrong root or a stale snapshot.
    const geometry = resolveSlitGeometry(original, slits);
    const id = `slit-${crypto.randomUUID().slice(0, 8)}`;
    const paletteIndex = TRACK_COLORS.indexOf(original.color);
    const color = paletteIndex >= 0
      ? TRACK_COLORS[(paletteIndex + 1) % TRACK_COLORS.length]
      : TRACK_COLORS[slits.length % TRACK_COLORS.length];
    const duplicate: SlitDefinition = {
      ...original,
      id,
      linkedTo: undefined,
      name: `${original.name} copy`,
      color,
      curveArcsec: geometry.curveArcsec.map((point) => [...point] as [number, number]),
      inputVertexCount: geometry.inputVertexCount,
      widthArcsec: geometry.widthArcsec,
      freqIndices: [...original.freqIndices],
      contourFreqIndices: [...original.contourFreqIndices],
      display: { ...original.display }
    };
    setSlits((current) => [...current, duplicate]);
    setSelectedSlitId(id);
    setSlitDraftWidthArcsec(duplicate.widthArcsec);
    setMessage(`Duplicated ${original.name} → ${duplicate.name}; rebind or Extract to build its own map.`);
  }

  // Creates a "linked twin" of a slit: a new slit whose curveArcsec/
  // inputVertexCount/width are RESOLVED live from the original (see
  // resolveSlitGeometry) rather than copied, so later edits to the
  // original's geometry (reverse, fan-boundary redraw) instantly apply to
  // the twin's rendering and its next extraction too - see
  // invalidateTwinResults for how its stale cached results get dropped when
  // that happens. If `original` is itself a twin, the new twin links to the
  // same ultimate root (slitOriginId) rather than chaining through it.
  //
  // Binding defaults to "one click from an AIA slit to its radio twin":
  // the inspector's current "Default binding" picker when it differs from
  // the original's own binding, else any available contours-kind binding,
  // else (nothing better available) the original's own binding - same as
  // duplicateSlit in that last-resort case. freqIndices/baseFreqIndex reuse
  // rebindSlit's defaulting (slitDefaultChannelsForBinding).
  function createLinkedTwin(slitId: string) {
    const original = slits.find((candidate) => candidate.id === slitId);
    if (!original) return;
    const defaultBinding = slitSourceOptions.find((option) => option.sourceId === slitSourceId);
    const contourBinding = slitSourceOptions.find((option) => option.bindingKind === "contours");
    const binding = (defaultBinding && defaultBinding.sourceId !== original.sourceId)
      ? defaultBinding
      : contourBinding ?? slitSourceOptions.find((option) => option.sourceId === original.sourceId);
    if (!binding) return;
    const rootId = slitOriginId(original, slits);
    const id = `slit-${crypto.randomUUID().slice(0, 8)}`;
    const paletteIndex = TRACK_COLORS.indexOf(original.color);
    const color = paletteIndex >= 0
      ? TRACK_COLORS[(paletteIndex + 1) % TRACK_COLORS.length]
      : TRACK_COLORS[slits.length % TRACK_COLORS.length];
    const { freqIndices, baseFreqIndex } = slitDefaultChannelsForBinding(binding);
    const twin: SlitDefinition = {
      ...original,
      id,
      linkedTo: rootId,
      name: `${original.name} twin`,
      color,
      sourceId: binding.sourceId,
      layerId: binding.layer.id,
      bindingKind: binding.bindingKind,
      shiftSeconds: 0,
      // Seed geometry with the original's current curve as a fallback
      // snapshot only (see resolveSlitGeometry's docstring) - every real
      // read of this twin's geometry resolves through linkedTo instead.
      curveArcsec: original.curveArcsec.map((point) => [...point] as [number, number]),
      inputVertexCount: original.inputVertexCount,
      widthArcsec: original.widthArcsec,
      // A twin never has a hand-drawn stroke of its own - its geometry
      // (and hence its Smooth control) resolves live from the original via
      // linkedTo/resolveSlitGeometry, not from a stored raw stroke. Clear
      // these rather than let the spread above carry over the ORIGINAL's
      // raw/smoothPx/drawnPanel under the twin's own id, which would be
      // misleading in an export and is never read (the Smooth/Width
      // controls are both disabled for a selected twin).
      rawCurveArcsec: undefined,
      smoothPx: undefined,
      drawnPanel: undefined,
      freqIndices,
      baseFreqIndex,
      contourFreqIndices: [],
      display: { ...original.display }
    };
    setSlits((current) => [...current, twin]);
    setSelectedSlitId(id);
    setSlitDraftWidthArcsec(twin.widthArcsec);
    setMessage(`Linked ${twin.name} to ${original.name}; its geometry follows the original automatically.`);
  }

  function deleteFan() {
    if (!fan && !fanBoundaryDraft) return;
    const promotedIds = new Set(Object.values(fan?.promotedMembers ?? {}));
    // Same twin-materialization as deleteSlit: a promoted slit being removed
    // here may itself be the original of a linked twin, which must survive
    // as an independent slit rather than being cascade-deleted.
    setSlits((current) => current
      .filter((slit) => !promotedIds.has(slit.id))
      .map((slit) => slit.linkedTo && promotedIds.has(slit.linkedTo)
        ? { ...slit, linkedTo: undefined, ...resolveSlitGeometry(slit, current) }
        : slit));
    setSlitResults((current) => {
      const next = { ...current };
      promotedIds.forEach((slitId) => delete next[slitId]);
      return next;
    });
    if (promotedIds.has(selectedSlitId)) setSelectedSlitId("");
    setFan(null);
    setFanDrawStage(0);
    setFanBoundaryDraft(null);
    setFanRedrawTarget(null);
    setSelectedFanMember(-1);
    setMessage("Fan geometry and its promoted slits deleted.");
  }

  function patchSlit(slitId: string, patch: Partial<SlitDefinition>) {
    setSlits((current) => current.map((slit) => slit.id === slitId
      ? { ...slit, ...patch, display: patch.display ? { ...slit.display, ...patch.display } : slit.display }
      : slit));
  }

  // Drops any cached extraction results belonging to linked twins of
  // `originId` - called from every mutation that changes an original
  // slit's resolved geometry (reverse, fan-boundary redraw for promoted
  // slits, width) so a twin never shows a time-distance map extracted from
  // geometry the original has since moved away from. This is the chosen
  // invalidation approach: explicit clearing at each geometry-mutating call
  // site, rather than keying slitResults itself on a geometry hash - the
  // mutation sites are few and already enumerable (see call sites below).
  function invalidateTwinResults(originId: string) {
    const twinIds = slits.filter((slit) => slit.linkedTo === originId).map((slit) => slit.id);
    if (!twinIds.length) return;
    setSlitResults((current) => {
      let changed = false;
      const next = { ...current };
      for (const id of twinIds) {
        if (id in next) { delete next[id]; changed = true; }
      }
      return changed ? next : current;
    });
  }

  // Per-slit re-smoothing (FEATURE 1): re-runs the gaussian+resample stage
  // from the slit's stored raw stroke (rawCurveArcsec) at a new smoothPx,
  // replacing curveArcsec in place and invalidating its own cached result
  // plus any linked twins' (same invalidation as width changes - see
  // onWidthChange in SlitInspectorCard's wiring). Not called directly for a
  // linked twin (its curve is resolved from its origin - see
  // resolveSlitGeometry - so there is nothing of its own to re-smooth) or
  // for a fan-promoted member whose fan still exists (re-smoothed via
  // reSmoothFan instead, which regenerates the whole family) - see
  // commitSlitSmooth's dispatch, the sole caller.
  //
  // LEGACY ADOPTION: a slit with no captured raw stroke - session data from
  // before per-slit re-smoothing existed, or a fan-promoted slit whose
  // parent fan has since been deleted (buildFanMemberSlit never sets
  // rawCurveArcsec to begin with) - has nothing to re-derive from. Rather
  // than leave the control disabled for these (the old behavior), the first
  // commit ADOPTS the slit's CURRENT curve as its raw baseline and proceeds
  // exactly as if that had always been the raw stroke. Accepted per user
  // request: the adopted baseline is already smoothed, so this only
  // smooths further from here - lowering below whatever smoothness is
  // already baked into the adopted curve cannot recover the original
  // jitter.
  function reSmoothSlit(slitId: string, smoothPx: number) {
    const slit = slits.find((candidate) => candidate.id === slitId);
    if (!slit || slit.linkedTo) return;
    const hasRaw = Boolean(slit.rawCurveArcsec && slit.rawCurveArcsec.length >= 2);
    const rawCurveArcsec = hasRaw ? slit.rawCurveArcsec! : slit.curveArcsec;
    if (rawCurveArcsec.length < 2) return;
    const drawnPanel = slit.drawnPanel ?? "aia";
    const clampedSmoothPx = clamp(smoothPx, 0, SLIT_SMOOTH_MAX_PX);
    const scale = affinePixelScaleArcsec(panelAffine(drawnPanel));
    const curveArcsec = resmoothCurveFromRaw(rawCurveArcsec, clampedSmoothPx, scale, solarView);
    if (curveArcsec.length < 2) return;
    patchSlit(slitId, { curveArcsec, smoothPx: clampedSmoothPx, rawCurveArcsec, drawnPanel });
    setSlitResults((current) => {
      if (!(slitId in current)) return current;
      const next = { ...current };
      delete next[slitId];
      return next;
    });
    invalidateTwinResults(slitId);
    setMessage(`${slit.name} re-smoothed at ${clampedSmoothPx.toFixed(1)} px.${hasRaw ? "" : " (Adopted its current curve as the re-smooth baseline.)"}`);
  }

  // Single entry point for the inspector's one Smooth [px] control (UX
  // consolidation - the old separate "Smooth (new)" draft control is gone,
  // see slitDraftSmoothPx). Every commit updates the draft default too, so
  // the last value used here seeds the next drawn slit/fan boundary; with
  // nothing selected that IS the whole effect. With a slit selected, the
  // edit is routed to whichever geometry actually owns the smoothness:
  //   - a linked twin has no geometry of its own (resolveSlitGeometry) - the
  //     edit redirects to its origin (slitOriginId walks the whole chain,
  //     though createLinkedTwin never actually produces more than one hop).
  //   - a fan-promoted member (its id is a value in the live fan's
  //     promotedMembers) is generated from the fan's two boundaries -
  //     re-smoothing it re-smooths the FAN (reSmoothFan), regenerating
  //     every member curve, not just this one. If the fan has since been
  //     deleted, the promoted slit is just a plain slit now and falls
  //     through to reSmoothSlit below (which will adopt its current curve -
  //     see reSmoothSlit's docstring).
  //   - anything else (plain slit with or without a captured raw stroke)
  //     goes through reSmoothSlit directly.
  // SlitInspectorCard mirrors this exact resolution to decide what value to
  // display (smoothTarget/smoothTargetFanPromoted there).
  function commitSlitSmooth(smoothPx: number) {
    const clampedSmoothPx = clamp(smoothPx, 0, SLIT_SMOOTH_MAX_PX);
    setSlitDraftSmoothPx(clampedSmoothPx);
    if (!selectedSlit) return;
    const targetId = slitOriginId(selectedSlit, slits);
    const target = slits.find((candidate) => candidate.id === targetId) ?? selectedSlit;
    if (fan && Object.values(fan.promotedMembers).includes(target.id)) {
      reSmoothFan(clampedSmoothPx);
      return;
    }
    reSmoothSlit(target.id, clampedSmoothPx);
  }

  // Shared by rebindSlit and createLinkedTwin: a contours-kind binding can
  // carry many channels at once - every one of them becomes its own map at
  // extraction (see extractSlit's isContourFamily path) - unlike a
  // single-image binding, which only ever shows the one channel on the
  // panel. Defaults a freshly (re)bound contour slit's freqIndices to
  // whatever channels are already selected in the channel picker/alignment
  // tool (selectedChannels), or every available radio channel when nothing
  // is selected there; a plain image or raw binding keeps the prior
  // single-channel-or-none behavior.
  function slitDefaultChannelsForBinding(binding: SlitSourceOption): { freqIndices: number[]; baseFreqIndex: number | null } {
    const radioChannel = binding.layer.sourceRoleSnapshot === "radio" ? binding.layer.freqIndex : null;
    const contourDefaultChannels = selectedChannels.filter((index) => index >= 0 && index < radioFreqGhz.length);
    const freqIndices = binding.bindingKind === "contours"
      ? (contourDefaultChannels.length ? contourDefaultChannels : radioFreqGhz.map((_, index) => index))
      : radioChannel === null ? [] : [radioChannel];
    // Contour-family bindings default to no base map (contours on dark);
    // the layer's displayed channel must not leak in as a base raster.
    return { freqIndices, baseFreqIndex: binding.bindingKind === "contours" ? null : radioChannel };
  }

  function rebindSlit(slitId: string, sourceId: string) {
    const slit = slits.find((candidate) => candidate.id === slitId);
    const binding = slitSourceOptions.find((candidate) => candidate.sourceId === sourceId);
    if (!slit || !binding) return;
    const { freqIndices, baseFreqIndex } = slitDefaultChannelsForBinding(binding);
    patchSlit(slitId, {
      sourceId: binding.sourceId,
      layerId: binding.layer.id,
      bindingKind: binding.bindingKind,
      freqIndices,
      baseFreqIndex,
      contourFreqIndices: []
    });
    setSlitResults((current) => {
      const next = { ...current };
      delete next[slitId];
      return next;
    });
    setMessage(`Rebound ${slit.name} to ${binding.label}; extract again to refresh its map.`);
  }

  function rebindFan(sourceId: string) {
    const binding = slitSourceOptions.find((candidate) => candidate.sourceId === sourceId);
    if (!binding) return;
    setFan((current) => current ? {
      ...current,
      sourceId: binding.sourceId,
      layerId: binding.layer.id,
      bindingKind: binding.bindingKind
    } : current);
    setMessage(`Fan default binding changed to ${binding.label}; existing promoted slits keep their own bindings.`);
  }

  async function reverseSlitDirection(slitId: string) {
    const slit = slits.find((candidate) => candidate.id === slitId);
    if (!slit || !sessionId) return;
    patchSlit(slitId, { curveArcsec: [...slit.curveArcsec].reverse() });
    // Geometry just changed under any linked twins (their results are
    // resolved from this slit's curve) - drop their stale cached maps so
    // the next Extract picks up the reversed curve instead of silently
    // showing the pre-reverse distance axis.
    invalidateTwinResults(slitId);
    const cachedResult = slitResults[slitId];
    const reversedResult = cachedResult ? reverseSlitResult(cachedResult) : null;
    if (reversedResult) setSlitResults((current) => ({ ...current, [slitId]: reversedResult }));
    if (!cachedResult) {
      setMessage(`Reversed ${slit.name}.`);
      return;
    }
    try {
      await apiJson(`/api/sessions/${sessionId}/slits/${encodeURIComponent(slitId)}/reverse`, { method: "POST" }, restoreSessionAfterRestart);
      setMessage(`Reversed ${slit.name}; the cached distance map was flipped without extraction.`);
    } catch (error) {
      patchSlit(slitId, { curveArcsec: slit.curveArcsec });
      const restoredResult = cachedResult ? reverseSlitResult(cachedResult) : null;
      if (restoredResult) setSlitResults((current) => ({ ...current, [slitId]: restoredResult }));
      reportError(error, `Could not reverse ${slit.name}`);
    }
  }

  function armSlitDrawing() {
    if (!slitSourceId) return;
    setSlitDrawArmed(true);
    setFanRedrawTarget(null);
    setTargetDrawArmed(false);
    setSeedMode(false);
    setLassoEnabled(false);
    setChannelLassoArmed(false);
    closePixelProbe();
    setMessage("Draw an open slit curve on either image panel.");
  }

  // Applies one freshly extracted slit result to state exactly the way a
  // single-slit extraction always has (range/base-channel stamping rules
  // included) - factored out of extractSlit so extractAllSlits (which
  // receives its per-slit results from a batch endpoint instead) can share
  // the same application logic bit-for-bit rather than re-deriving it.
  function applyExtractedSlitResult(targetSlit: SlitDefinition, result: SlitResult, hadResult: boolean) {
    setSlitResults((current) => ({ ...current, [targetSlit.id]: result }));
    const resultMaps = slitResultMaps(result);
    const availableMapIndices = resultMaps.flatMap((map) => map.freqIndex === null ? [] : [map.freqIndex]);
    // Every selected channel is always part of the contour family for a
    // contour-bound radio slit (see TimeDistanceLane) - there is no
    // separate include/exclude toggle to maintain for it. Multi-channel
    // contour-bound slits default their heat-map base to None (contours on
    // a dark background); everything else keeps the prior single-image
    // default of "first selected channel is the base".
    const isContourFamily = targetSlit.bindingKind === "contours" && availableMapIndices.length > 1;
    // A freshly-drawn slit pre-seeds baseFreqIndex from whichever single
    // channel was active on the panel when the user drew it (see
    // completeSlitDraw/buildFanMemberSlit) - that channel is normally
    // still among the selected freqIndices by the first extraction, so a
    // plain "was the prior base valid" check would never see it as
    // invalid and would never reach the None default. Force None only on
    // this very first extraction of a multi-channel contour-bound slit;
    // every later re-extraction keeps whatever the user has since chosen
    // (a specific channel, or None) via priorBaseValid below.
    const priorBaseValid = !hadResult && isContourFamily
      ? false
      : targetSlit.baseFreqIndex === null
        ? isContourFamily
        : availableMapIndices.includes(targetSlit.baseFreqIndex);
    const nextBaseFreqIndex = priorBaseValid ? targetSlit.baseFreqIndex : (isContourFamily ? null : availableMapIndices[0] ?? null);
    const baseChanged = nextBaseFreqIndex !== targetSlit.baseFreqIndex;
    if (availableMapIndices.length && (baseChanged || (isContourFamily && targetSlit.contourFreqIndices.length !== availableMapIndices.length))) {
      patchSlit(targetSlit.id, {
        baseFreqIndex: nextBaseFreqIndex,
        contourFreqIndices: isContourFamily ? availableMapIndices : availableMapIndices.filter((index) => index !== nextBaseFreqIndex)
      });
    }
    // Range defaults to the robust p1..p99 of whichever map is now the
    // base heat map (falling back to the primary/first-requested map when
    // there is no base, e.g. None) - not the raw dataMin/dataMax, and not
    // necessarily the first-requested channel's stats if the base ended up
    // being a different channel. Only (re)stamped on first load or when
    // extraction just changed which channel is the base, so a user's own
    // range edits survive later re-extractions of the same base channel.
    if (!hadResult || baseChanged) {
      const rangeMap = resultMaps.find((map) => map.freqIndex === nextBaseFreqIndex) ?? result;
      patchSlit(targetSlit.id, {
        display: {
          ...targetSlit.display,
          vmin: rangeMap.dataP1,
          vmax: rangeMap.dataP99 > rangeMap.dataP1 ? rangeMap.dataP99 : rangeMap.dataMax
        }
      });
    }
  }

  // Shared by the "Extract" button (targetSlit = selectedSlit) and the fan
  // family's "Extract this curve" one-click action (targetSlit = a slit
  // object built synchronously in extractFanMember, so it works even before
  // the promotion's setSlits/setSelectedSlitId state updates have committed).
  async function extractSlit(targetSlit: SlitDefinition | null) {
    if (!targetSlit || !sessionId || slitExtracting) return;
    setLoading(true);
    setSlitExtracting(true);
    setMessage(`Extracting ${targetSlit.name} over its native source frames...`);
    try {
      const hadResult = Boolean(slitResults[targetSlit.id]);
      // Always send the RESOLVED geometry (the twin's own curveArcsec/
      // widthArcsec fields are only a best-effort snapshot - see
      // resolveSlitGeometry), both so a twin's map matches the original's
      // live curve and so the server's content-addressed extraction cache
      // key (built from this request body) naturally lands on the same
      // cache entry as any other slit - twin or not - sharing that resolved
      // geometry + binding + channels.
      const geometry = resolveSlitGeometry(targetSlit, slits);
      // Resample the resolved (possibly AIA-dense) curve to THIS slit's own
      // bound source's native pixel scale before sending it - a twin bound
      // to radio must resample at radio scale even when its origin lives on
      // an AIA-bound slit (see resampleCurveForSource/affinePixelScaleArcsec
      // above). This changes the request body, so it also changes the
      // server's content-addressed cache key for radio-bound
      // slits/twins that previously sent the raw AIA-dense curve - a
      // one-time re-extraction, expected and harmless.
      const targetScaleArcsec = affinePixelScaleArcsec(sourceAffine(targetSlit.sourceId));
      const extractionCurveArcsec = resampleCurveForSource(geometry.curveArcsec, targetScaleArcsec);
      // EXTRACTION CONTRACT UNCHANGED: the backend still receives a pixel
      // width (its own MAX_SLIT_WIDTH_PX/cache-key logic is untouched) - the
      // physical widthArcsec is converted to THIS slit's own bound source's
      // pixel width only here, at request-build time. A twin bound to a
      // different source than its origin therefore sends a DIFFERENT px
      // width for the SAME physical corridor - correct (see
      // SlitDefinition.widthArcsec) - and a migrated slit whose px width
      // happens to round-trip identically keeps hitting the same server
      // cache entry; no further cache-preservation effort is made beyond that.
      const widthPx = Math.max(1, Math.round(geometry.widthArcsec / targetScaleArcsec));
      const result = await apiJson<SlitResult>(`/api/sessions/${sessionId}/slits/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          slitId: targetSlit.id,
          name: targetSlit.name,
          sourceId: targetSlit.sourceId,
          curveArcsec: extractionCurveArcsec,
          width: widthPx,
          layerParams: slitLayerParams(targetSlit),
          freqIndices: targetSlit.freqIndices,
          contourLevel: slitContourLevelParams(targetSlit)
        })
      }, restoreSessionAfterRestart);
      applyExtractedSlitResult(targetSlit, result, hadResult);
      setMessage(`${targetSlit.name}: ${result.npix} × ${result.ntime} in ${result.wallSeconds.toFixed(2)} s${result.cacheHit ? " (cached)" : ""}.`);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) setMessage("Time-distance extraction cancelled.");
      else if (!(error instanceof SessionRestoredError)) reportError(error, "Could not extract the slit");
    } finally {
      setSlitExtracting(false);
      setLoading(false);
    }
  }

  async function extractSelectedSlit() {
    await extractSlit(selectedSlit);
  }

  // "All slits" mode: batch-extracts every VISIBLE slit (skips nothing else -
  // hidden slits are left alone, same population the lane/panel already
  // filter on elsewhere). Groups by (sourceId, serialized layerParams) so
  // slits that share one canonical source AND one processing snapshot ride
  // one /slits/extract-batch request - the backend then reads each native
  // source frame once for the whole group (see extract_slit_batch in
  // data.py) instead of once per slit. Groups run sequentially (normally at
  // most two: AIA context and radio), each its own progress op server-side;
  // this function's own status message is the single "Extracting N slits"
  // op the user sees, updated per group.
  async function extractAllSlits() {
    if (!sessionId || slitExtracting) return;
    const visibleSlits = slits.filter((slit) => slit.visible);
    if (!visibleSlits.length) {
      setMessage("No visible slits to extract.");
      return;
    }
    setLoading(true);
    setSlitExtracting(true);
    const slitWord = visibleSlits.length === 1 ? "slit" : "slits";
    setMessage(`Extracting ${visibleSlits.length} ${slitWord}...`);
    type BatchEntry = {
      slitId: string;
      name: string;
      curveArcsec: [number, number][];
      width: number;
      freqIndices?: number[];
      contourLevel?: Record<string, unknown>;
    };
    type PreparedSlit = { slit: SlitDefinition; hadResult: boolean; entry: BatchEntry };
    type Group = { sourceId: string; layerParams: Record<string, unknown>; items: PreparedSlit[] };
    let completed = 0;
    try {
      // Same per-slit inputs extractSlit sends (resolved twin geometry,
      // resampled to this slit's own bound source scale) - see its comments
      // above - so a batched slit's cache key lands on the exact same entry
      // a single extraction of it would.
      const groups = new Map<string, Group>();
      const groupOrder: string[] = [];
      for (const slit of visibleSlits) {
        const geometry = resolveSlitGeometry(slit, slits);
        const slitScaleArcsec = affinePixelScaleArcsec(sourceAffine(slit.sourceId));
        const extractionCurveArcsec = resampleCurveForSource(geometry.curveArcsec, slitScaleArcsec);
        // EXTRACTION CONTRACT UNCHANGED - see extractSlit's comment: convert
        // the physical widthArcsec to THIS slit's own bound source's pixel
        // width here, at request-build time. Twins in the same batch may
        // land in different groups (different sourceId) and send different
        // px widths for the same physical corridor.
        const widthPx = Math.max(1, Math.round(geometry.widthArcsec / slitScaleArcsec));
        const layerParams = slitLayerParams(slit);
        const groupKey = `${slit.sourceId}|${JSON.stringify(layerParams)}`;
        if (!groups.has(groupKey)) {
          groups.set(groupKey, { sourceId: slit.sourceId, layerParams, items: [] });
          groupOrder.push(groupKey);
        }
        groups.get(groupKey)!.items.push({
          slit,
          hadResult: Boolean(slitResults[slit.id]),
          entry: {
            slitId: slit.id,
            name: slit.name,
            curveArcsec: extractionCurveArcsec,
            width: widthPx,
            freqIndices: slit.freqIndices,
            contourLevel: slitContourLevelParams(slit)
          }
        });
      }
      for (let groupIndex = 0; groupIndex < groupOrder.length; groupIndex++) {
        const group = groups.get(groupOrder[groupIndex])!;
        setMessage(groupOrder.length > 1
          ? `Extracting ${visibleSlits.length} ${slitWord} (group ${groupIndex + 1}/${groupOrder.length})...`
          : `Extracting ${visibleSlits.length} ${slitWord}...`);
        const response = await apiJson<{ results: Record<string, SlitResult> }>(
          `/api/sessions/${sessionId}/slits/extract-batch`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sourceId: group.sourceId,
              layerParams: group.layerParams,
              slits: group.items.map((item) => item.entry),
              groupIndex,
              groupCount: groupOrder.length
            })
          },
          restoreSessionAfterRestart
        );
        for (const item of group.items) {
          const result = response.results[item.slit.id];
          if (!result) continue;
          applyExtractedSlitResult(item.slit, result, item.hadResult);
          completed += 1;
        }
      }
      setMessage(`Extracted ${completed} of ${visibleSlits.length} ${slitWord}.`);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setMessage(completed > 0
          ? `Time-distance extraction cancelled after ${completed} of ${visibleSlits.length} ${slitWord}.`
          : "Time-distance extraction cancelled.");
      } else if (!(error instanceof SessionRestoredError)) {
        reportError(error, "Could not extract all slits");
      }
    } finally {
      setSlitExtracting(false);
      setLoading(false);
    }
  }

  async function cancelSlitExtraction() {
    if (!sessionId) return;
    try {
      await apiJson(`/api/sessions/${sessionId}/slits/cancel`, { method: "POST" });
      setMessage("Cancelling time-distance extraction...");
    } catch (error) {
      reportError(error, "Could not cancel slit extraction");
    }
  }

  function exportSlitLanePng() {
    if (!selectedSlit) return;
    slitLaneCanvasRef.current?.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${recordingDatasetToken(datasetName)}_${recordingDatasetToken(selectedSlit.name)}_time_distance.png`;
      link.click();
      URL.revokeObjectURL(url);
    }, "image/png");
  }

  async function selectChannelsByLasso(panel: PanelId, points: [number, number][], sourceOverride?: string, roleOverride?: SourceRole, additive = false) {
    if (!meta || !alignmentOpen) return;
    const targetSourceId = sourceOverride ?? (panel === "aia" ? contextSourceId : radioSourceId);
    const targetRole = roleOverride ?? sourceRole(sources.find((source) => source.id === targetSourceId), sourceRoles);
    const targetSlot = (Object.keys(panelLayers) as PanelSlotId[]).find((slot) => {
      const composition = panelCompositions[slot];
      const rawBase = panelLayers[slot].find((layer) => layer.id === composition.baseLayerId);
      const base = rawBase ? resolvedLayer(rawBase, allPanelLayers) : undefined;
      return base?.sourceId === targetSourceId;
    }) ?? (panel === "aia" ? "left" : "right");
    const contourLayer = panelLayers[targetSlot]
      .map((layer) => resolvedLayer(layer, allPanelLayers))
      .find((layer) => layer.kind === "contours" && layer.sourceRoleSnapshot === "radio");
    const request = eovsaContourRequest(masterCursorIndex, currentMjd, contourLayer, targetSourceId, targetRole);
    const query = Object.fromEntries(new URL(request.url, window.location.href).searchParams.entries());
    setMessage("Selecting radio channels inside the lasso...");
    try {
      const response = await apiJson<{ channels: number[] }>(`/api/sessions/${sessionId}/sources/${radioSourceId}/select-channels`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...query,
          panel,
          sourceId: radioSourceId,
          points,
          xOffsetArcsec: numberValue(xOffset, 7),
          yOffsetArcsec: numberValue(yOffset, 0)
        })
      });
      const channels = [...new Set(response.channels.filter((index) => Number.isInteger(index) && index >= 0 && index < radioFreqGhz.length))].sort((left, right) => left - right);
      if (!channels.length) {
        setMessage("no contours inside the region");
        return;
      }
      setSelectedChannels((current) => additive
        ? [...new Set([...current, ...channels])].sort((left, right) => left - right)
        : channels);
      setMessage(`Selected ${channels.length} radio channel${channels.length === 1 ? "" : "s"}.`);
    } catch (error) {
      reportError(error, "Could not select radio channels from lasso");
    }
  }

  function trackingLayerParams(sourceId: string): Record<string, unknown> {
    const option = trackingSourceOptions.find((candidate) => candidate.sourceId === sourceId);
    const source = sources.find((candidate) => candidate.id === sourceId);
    if (!meta || !option || !source) throw new Error("The trajectory source must be present as an image layer.");
    const times = sourceTimeMjd(source, meta, sourceRoles);
    const fallback = sourceDifferences[sourceId] ?? defaultDifference(sourceRole(source, sourceRoles), meta.defaults.diffSeconds, times.length);
    const difference = differenceForLayer(option.layer, fallback);
    const temporal = normalizeTemporal(option.layer.temporal);
    return {
      sourceId,
      freqIndex: option.layer.freqIndex,
      ...differenceParams(difference, times),
      radialGamma: Number(option.layer.display.radialGamma ?? 0),
      temporalMode: temporal.mode,
      temporalSigmaShort: numberValue(temporal.sigmaShort, DEFAULT_TEMPORAL_SIGMA_SHORT),
      temporalSigmaLong: numberValue(temporal.sigmaLong, DEFAULT_TEMPORAL_SIGMA_LONG)
    };
  }

  function slitLayerParams(slit: SlitDefinition): Record<string, unknown> {
    const savedLayer = allPanelLayers.find((candidate) => candidate.id === slit.layerId);
    const layer = savedLayer && slit.bindingKind !== "raw"
      ? resolvedLayer(savedLayer, allPanelLayers)
      : slitSourceOptions.find((candidate) => (
          candidate.sourceId === slit.sourceId && candidate.bindingKind === slit.bindingKind
        ))?.layer;
    const source = sources.find((candidate) => candidate.id === slit.sourceId);
    if (!meta || !layer || !source) throw new Error("The slit source is not available for extraction.");
    const times = sourceTimeMjd(source, meta, sourceRoles);
    const fallback = sourceDifferences[slit.sourceId] ?? defaultDifference(sourceRole(source, sourceRoles), meta.defaults.diffSeconds, times.length);
    const difference = differenceForLayer(layer, fallback);
    const temporal = normalizeTemporal(layer.temporal);
    const worldOffset = layer.sourceRoleSnapshot === "radio"
      ? [numberValue(xOffset, 7), numberValue(yOffset, 0)]
      : [0, 0];
    return {
      sourceId: slit.sourceId,
      bindingKind: slit.bindingKind,
      freqIndex: layer.freqIndex,
      ...differenceParams(difference, times),
      radialGamma: Number(layer.display.radialGamma ?? 0),
      temporalMode: temporal.mode,
      temporalSigmaShort: numberValue(temporal.sigmaShort, DEFAULT_TEMPORAL_SIGMA_SHORT),
      temporalSigmaLong: numberValue(temporal.sigmaLong, DEFAULT_TEMPORAL_SIGMA_LONG),
      xOffsetArcsec: worldOffset[0],
      yOffsetArcsec: worldOffset[1]
    };
  }

  // Deliberately kept OUT of slitLayerParams: these level fields feed only
  // the display-side contour threshold (see backend radio_contour_threshold)
  // and must never enter the extraction disk-cache key, or scrubbing the
  // bound layer's contour level would force a full re-extraction instead of
  // a display-only re-render.
  function slitContourLevelParams(slit: SlitDefinition): Record<string, unknown> | undefined {
    if (slit.bindingKind !== "contours") return undefined;
    const savedLayer = allPanelLayers.find((candidate) => candidate.id === slit.layerId);
    const layer = savedLayer ? resolvedLayer(savedLayer, allPanelLayers) : undefined;
    if (!layer || layer.kind !== "contours") return undefined;
    const source = sources.find((candidate) => candidate.id === slit.sourceId);
    if (!meta || !source) return undefined;
    const times = sourceTimeMjd(source, meta, sourceRoles);
    const fallback = sourceDifferences[slit.sourceId] ?? defaultDifference(sourceRole(source, sourceRoles), meta.defaults.diffSeconds, times.length);
    const difference = differenceForLayer(layer, fallback);
    return {
      levelMode: layer.contourLevelMode,
      levelReference: layer.contourLevelReference,
      levelPercent: numberValue(layer.contourLevelPercent, 50),
      levelKelvin: numberValue(layer.contourLevelKelvin, 1e6),
      levelSfu: numberValue(layer.contourLevelSfu, 1.0),
      ...differenceParams(difference, times)
    };
  }

  function cloneTracks(tracks: SadTrack[]): SadTrack[] {
    return JSON.parse(JSON.stringify(tracks)) as SadTrack[];
  }

  function rememberTrackMutation(snapshot = sadTracks) {
    const history = trackHistoryRef.current;
    history.past = [...history.past, cloneTracks(snapshot)].slice(-TRACK_HISTORY_LIMIT);
    history.future = [];
    setTrackHistoryVersion((value) => value + 1);
  }

  async function syncTrackSnapshot(tracks: SadTrack[]) {
    if (!sessionId) return;
    await apiJson<{ tracks: SadTrack[] }>(`/api/sessions/${sessionId}/track/sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tracks, correlationTarget })
    }, restoreSessionAfterRestart);
  }

  async function undoTracks() {
    const history = trackHistoryRef.current;
    const previous = history.past.at(-1);
    if (!previous) return;
    history.past = history.past.slice(0, -1);
    history.future = [cloneTracks(sadTracks), ...history.future].slice(0, TRACK_HISTORY_LIMIT);
    const restored = cloneTracks(previous);
    setSadTracks(restored);
    const restoredSelected = restored.find((track) => track.id === selectedTrackId);
    if (!restoredSelected) {
      setSelectedTrackId("");
      setSelectedAnchorFrame(null);
    } else if (selectedAnchorFrame !== null && !restoredSelected.anchors.some((anchor) => anchor.frameIndex === selectedAnchorFrame)) {
      setSelectedAnchorFrame(null);
    }
    setTrackHistoryVersion((value) => value + 1);
    try {
      await syncTrackSnapshot(restored);
      setMessage("Undid track mutation.");
    } catch (error) {
      reportError(error, "Could not persist track undo");
    }
  }

  async function redoTracks() {
    const history = trackHistoryRef.current;
    const next = history.future[0];
    if (!next) return;
    history.future = history.future.slice(1);
    history.past = [...history.past, cloneTracks(sadTracks)].slice(-TRACK_HISTORY_LIMIT);
    const restored = cloneTracks(next);
    setSadTracks(restored);
    const restoredSelected = restored.find((track) => track.id === selectedTrackId);
    if (!restoredSelected) {
      setSelectedTrackId("");
      setSelectedAnchorFrame(null);
    } else if (selectedAnchorFrame !== null && !restoredSelected.anchors.some((anchor) => anchor.frameIndex === selectedAnchorFrame)) {
      setSelectedAnchorFrame(null);
    }
    setTrackHistoryVersion((value) => value + 1);
    try {
      await syncTrackSnapshot(restored);
      setMessage("Redid track mutation.");
    } catch (error) {
      reportError(error, "Could not persist track redo");
    }
  }

  function selectTrack(trackId: string) {
    setSelectedTrackId(trackId);
    setSelectedAnchorFrame(null);
    if (trackId) setTrackEditorOpen(true);
  }

  async function addSeed(pixel: [number, number]) {
    if (!meta || !trackingAvailable || !trackingSourceOption || !trackingSeedTimes.length) return;
    rememberTrackMutation();
    try {
      const result = await apiJson<{ track: SadTrack; tracks: SadTrack[] }>(`/api/sessions/${sessionId}/track/seed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sourceId: trackingSourceOption.sourceId,
          frameIndex: currentTrackingSeedFrame,
          x: pixel[0],
          y: pixel[1],
          color: TRACK_COLORS[sadTracks.length % TRACK_COLORS.length]
        })
      }, restoreSessionAfterRestart);
      const tracks = normalizeTracks(result.tracks, contextSourceId);
      setSadTracks(tracks);
      selectTrack(result.track.id);
      setSeedMode(false);
      setMessage(`Added ${result.track.label} on ${trackingSourceOption.label} at frame ${currentTrackingSeedFrame}.`);
    } catch (error) {
      trackHistoryRef.current.past.pop();
      setTrackHistoryVersion((value) => value + 1);
      reportError(error, "Could not add tracking seed");
    }
  }

  async function suggestSeeds() {
    if (!meta || !trackingAvailable || !trackingSourceOption || !roiWorld.length) return;
    setMessage("Finding dark feature candidates inside the ROI...");
    try {
      const result = await apiJson<{ suggestions: SeedSuggestion[] }>(`/api/sessions/${sessionId}/track/suggest-seeds`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frameIndex: currentTrackingSeedFrame, layerParams: trackingLayerParams(trackingSourceOption.sourceId) })
      }, restoreSessionAfterRestart);
      setSeedSuggestions(result.suggestions);
      setMessage(`Suggested ${result.suggestions.length} dark seed candidate${result.suggestions.length === 1 ? "" : "s"}.`);
    } catch (error) {
      reportError(error, "Could not suggest seeds");
    }
  }

  async function acceptSuggestion(suggestion: SeedSuggestion) {
    setSeedSuggestions((current) => current.filter((item) => item !== suggestion));
    await addSeed([suggestion.x, suggestion.y]);
  }

  async function acceptAllSuggestions() {
    if (!seedSuggestions.length || !trackingSourceOption) return;
    const suggestions = [...seedSuggestions];
    const before = sadTracks;
    rememberTrackMutation(before);
    let latest = sadTracks;
    try {
      for (const [index, suggestion] of suggestions.entries()) {
        const result = await apiJson<{ track: SadTrack; tracks: SadTrack[] }>(`/api/sessions/${sessionId}/track/seed`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sourceId: trackingSourceOption?.sourceId,
            frameIndex: currentTrackingSeedFrame,
            x: suggestion.x,
            y: suggestion.y,
            color: TRACK_COLORS[(before.length + index) % TRACK_COLORS.length]
          })
        }, restoreSessionAfterRestart);
        latest = normalizeTracks(result.tracks, contextSourceId);
      }
      setSadTracks(latest);
      setSeedSuggestions([]);
      if (latest.length) selectTrack(latest.at(-1)?.id ?? "");
      setMessage(`Accepted ${suggestions.length} suggested seeds.`);
    } catch (error) {
      reportError(error, "Could not accept all suggestions");
    }
  }

  async function runAutoTrack(
    trackIds = selectedTrackId ? [selectedTrackId] : sadTracks.map((track) => track.id),
    oneFrame = false
  ) {
    if (!meta || !trackIds.length || trackingActive) return;
    const requestedTracks = sadTracks.filter((track) => trackIds.includes(track.id));
    if (oneFrame && requestedTracks.length !== 1) return;
    const groups = [...requestedTracks.reduce((grouped, track) => {
      const ids = grouped.get(track.sourceId) ?? [];
      ids.push(track.id);
      grouped.set(track.sourceId, ids);
      return grouped;
    }, new Map<string, string[]>())];
    if (!groups.length) return;
    rememberTrackMutation();
    setLoading(true);
    setTrackingActive(true);
    trackingStopRequestedRef.current = false;
    const effectiveDirection: TrackDirection = oneFrame
      ? trackingDirection === "backward" ? "backward" : "forward"
      : trackingDirection;
    setMessage(oneFrame
      ? `Tracking ${requestedTracks[0].label} one frame ${effectiveDirection}...`
      : `Auto-tracking ${trackIds.length} seed${trackIds.length === 1 ? "" : "s"} ${effectiveDirection}...`);
    try {
      let latest = sadTracks;
      let processedFrames = 0;
      let steppedFrame: number | null = null;
      let steppedMjd: number | null = null;
      let steppedConfidence: number | null = null;
      for (const [sourceId, sourceTrackIds] of groups) {
        if (trackingStopRequestedRef.current) break;
        const source = sources.find((candidate) => candidate.id === sourceId);
        const times = sourceTimeMjd(source, meta, sourceRoles);
        if (!times.length) throw new Error(`Tracking source ${sourceId} has no native timeline.`);
        let requestStart = closestIndex(times, currentMjd);
        let requestRangeStart = closestIndex(times, masterMin);
        let requestRangeEnd = closestIndex(times, masterMax);
        if (oneFrame) {
          const track = requestedTracks[0];
          const pointFrames = track.points.map((point) => point.frameIndex);
          if (!pointFrames.length) throw new Error(`${track.label} has no point to advance.`);
          requestStart = effectiveDirection === "backward" ? Math.min(...pointFrames) : Math.max(...pointFrames);
          const target = requestStart + (effectiveDirection === "backward" ? -1 : 1);
          if (target < 0 || target >= times.length) throw new Error(`${track.label} is already at the source timeline boundary.`);
          requestRangeStart = Math.min(requestStart, target);
          requestRangeEnd = Math.max(requestStart, target);
          steppedFrame = target;
          steppedMjd = times[target];
        }
        const result = await apiJson<{ tracks: SadTrack[]; processedFrames: number }>(`/api/sessions/${sessionId}/track/auto`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            trackIds: sourceTrackIds,
            layerParams: trackingLayerParams(sourceId),
            direction: effectiveDirection,
            startFrame: requestStart,
            rangeStart: requestRangeStart,
            rangeEnd: requestRangeEnd,
            patchRadius: 6,
            searchRadius: 18
          })
        }, restoreSessionAfterRestart);
        latest = normalizeTracks(result.tracks, contextSourceId);
        processedFrames += result.processedFrames;
        setSadTracks(latest);
        if (oneFrame && steppedFrame !== null) {
          steppedConfidence = latest
            .find((track) => track.id === requestedTracks[0].id)
            ?.points.find((point) => point.frameIndex === steppedFrame)
            ?.confidence ?? null;
        }
      }
      if (oneFrame) {
        if (steppedFrame === null || steppedMjd === null || steppedConfidence === null) {
          throw new Error(`No confident point was accepted for ${requestedTracks[0].label}.`);
        }
        setSelectedAnchorFrame(null);
        setMasterCursor(steppedMjd, 0);
        setMessage(`Tracked ${requestedTracks[0].label} to frame ${steppedFrame} · confidence ${steppedConfidence.toFixed(3)}.`);
      } else {
        setMessage(`Tracked ${trackIds.length} seed${trackIds.length === 1 ? "" : "s"} across ${processedFrames} frame steps.`);
      }
    } catch (error) {
      reportError(error, "Could not auto-track features");
    } finally {
      setTrackingActive(false);
      setLoading(false);
    }
  }

  async function stopAutoTrack() {
    if (!trackingActive) return;
    trackingStopRequestedRef.current = true;
    try {
      await apiJson<{ stopped: boolean }>(`/api/sessions/${sessionId}/track/stop`, { method: "POST" });
      setMessage("Stopping auto-track at the current confident point...");
    } catch (error) {
      reportError(error, "Could not stop tracking");
    }
  }

  async function retrackEditedTrack(edited: SadTrack, affectedFrame: number) {
    if (!meta) return;
    rememberTrackMutation();
    setSadTracks((tracks) => tracks.map((track) => track.id === edited.id ? edited : track));
    setLoading(true);
    try {
      const result = await apiJson<{ track: SadTrack; tracks: SadTrack[] }>(`/api/sessions/${sessionId}/track/retrack-segment`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track: edited, layerParams: trackingLayerParams(edited.sourceId), affectedFrame })
      }, restoreSessionAfterRestart);
      setSadTracks(normalizeTracks(result.tracks, contextSourceId));
      setSelectedAnchorFrame(affectedFrame);
      setMessage(`Re-tracked constrained segments through anchor frame ${affectedFrame}.`);
    } catch (error) {
      reportError(error, "Could not re-track edited anchor segments");
    } finally {
      setLoading(false);
    }
  }

  async function promotePointToAnchor(trackId: string, frameIndex: number, pixel: [number, number]) {
    if (!meta) return;
    const track = sadTracks.find((candidate) => candidate.id === trackId);
    if (!track) return;
    const times = sourceTimeMjd(sources.find((source) => source.id === track.sourceId), meta, sourceRoles);
    const mjd = times[frameIndex];
    if (!Number.isFinite(mjd)) return;
    const anchor: TrackAnchor = { frameIndex, mjd, x: pixel[0], y: pixel[1] };
    const edited: SadTrack = {
      ...track,
      anchors: [...track.anchors.filter((item) => item.frameIndex !== frameIndex), anchor].sort((left, right) => left.frameIndex - right.frameIndex),
      points: [...track.points.filter((point) => point.frameIndex !== frameIndex), { ...anchor, confidence: 1, isAnchor: true }].sort((left, right) => left.frameIndex - right.frameIndex)
    };
    await retrackEditedTrack(edited, frameIndex);
  }

  async function moveAnchorInTime(frameIndex: number, nextFrame: number) {
    if (!meta || !selectedTrack) return;
    const anchor = selectedTrack.anchors.find((item) => item.frameIndex === frameIndex);
    if (!anchor) return;
    const times = sourceTimeMjd(sources.find((source) => source.id === selectedTrack.sourceId), meta, sourceRoles);
    const targetFrame = clampIndex(nextFrame, frameIndex, times.length);
    const moved = { ...anchor, frameIndex: targetFrame, mjd: times[targetFrame] };
    const edited = {
      ...selectedTrack,
      anchors: [...selectedTrack.anchors.filter((item) => item.frameIndex !== frameIndex && item.frameIndex !== targetFrame), moved].sort((left, right) => left.frameIndex - right.frameIndex),
      points: [
        ...selectedTrack.points.filter((point) => point.frameIndex !== frameIndex && point.frameIndex !== targetFrame),
        { ...moved, confidence: 1, isAnchor: true }
      ].sort((left, right) => left.frameIndex - right.frameIndex)
    };
    setSelectedAnchorFrame(targetFrame);
    await retrackEditedTrack(edited, targetFrame);
  }

  async function deleteSelectedAnchor() {
    if (!selectedTrack || selectedAnchorFrame === null) return;
    const anchors = selectedTrack.anchors.filter((anchor) => anchor.frameIndex !== selectedAnchorFrame);
    if (!anchors.length) {
      await mutateTrackMetadata(sadTracks.filter((track) => track.id !== selectedTrack.id), "Deleted track after its final anchor was removed.");
      setSelectedTrackId("");
      setSelectedAnchorFrame(null);
      return;
    }
    const removedFrame = selectedAnchorFrame;
    const edited = {
      ...selectedTrack,
      anchors,
      points: selectedTrack.points.filter((point) => point.frameIndex !== removedFrame)
    };
    setSelectedAnchorFrame(null);
    await retrackEditedTrack(edited, removedFrame);
  }

  // Commits a key frame at the current frame using whatever position the ghost
  // marker is currently showing there: the real tracked point if one exists,
  // otherwise the same interpolated/extrapolated prediction the ghost renderer
  // uses (predictedTrackPoint). Reuses promotePointToAnchor - the identical
  // helper the ghost-drag-drop path (ImagePanel markerDragRef pointerUp ->
  // onAnchorMove) already calls - so the constrained re-track/extend behavior
  // and undo history are unchanged.
  async function addKeyFrameAtCurrentFrame() {
    if (!selectedTrack) return;
    const frameIndex = currentTrackFrame;
    if (selectedTrack.anchors.some((anchor) => anchor.frameIndex === frameIndex)) return;
    const existingPoint = selectedTrack.points.find((point) => point.frameIndex === frameIndex);
    const pixel: [number, number] | null = existingPoint
      ? [existingPoint.x, existingPoint.y]
      : (() => {
          const predicted = predictedTrackPoint(selectedTrack, frameIndex);
          return predicted ? [predicted.x, predicted.y] : null;
        })();
    if (!pixel) return;
    await promotePointToAnchor(selectedTrack.id, frameIndex, pixel);
  }

  async function mutateTrackMetadata(next: SadTrack[], successMessage: string) {
    rememberTrackMutation();
    setSadTracks(next);
    try {
      const result = await apiJson<{ tracks: SadTrack[] }>(`/api/sessions/${sessionId}/track/sync`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks: next, correlationTarget })
      }, restoreSessionAfterRestart);
      setSadTracks(normalizeTracks(result.tracks, contextSourceId));
      setMessage(successMessage);
    } catch (error) {
      reportError(error, "Could not update track");
    }
  }

  async function runSadTracking() {
    if (!meta) return;
    setLoading(true);
    setMessage("Loading and refining seed tracks from the configured file...");
    try {
      const result = await apiJson<{ count: number; tracks: SadTrack[] }>(`/api/sessions/${sessionId}/extract/aia-sads`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ searchRadius: 8 })
      });
      rememberTrackMutation();
      const tracks = normalizeTracks(result.tracks, contextSourceId);
      setSadTracks(tracks);
      setSelectedTrackId(tracks[0]?.id ?? "");
      setTrackEditorOpen(Boolean(tracks.length));
      setMessage(`Loaded ${result.count} seed points from file.`);
    } catch (error) {
      reportError(error);
    } finally {
      setLoading(false);
    }
  }

  async function runEovsaExtraction() {
    if (!meta) return;
    setLoading(true);
    setMessage("Extracting radio per-band peak/centroid sources inside the ROI...");
    try {
      const result = await apiJson<{ count: number; rows: EovsaSource[] }>(`/api/sessions/${sessionId}/extract/radio-sources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          xOffsetArcsec: numberValue(xOffset, 7),
          yOffsetArcsec: numberValue(yOffset, 0),
          diffSeconds: numberValue(radioDifference.cadenceSeconds, DEFAULT_DIFF_SECONDS),
          useRunningDiff: radioDifference.operation !== "none" && radioDifference.reference === "previous",
          differenceOperation: radioDifference.operation,
          differenceReference: radioDifference.reference,
          meanStartMjd: radioDifference.meanStartMjd ?? meta.eovsa.timeMjd[Math.min(radioDifference.meanStartIndex, radioDifference.meanEndIndex)],
          meanEndMjd: radioDifference.meanEndMjd ?? meta.eovsa.timeMjd[Math.max(radioDifference.meanStartIndex, radioDifference.meanEndIndex)],
          startMjd: masterMin,
          endMjd: masterMax,
          stride: 4,
          minSnr: 5
        })
      });
      setEovsaSources(result.rows);
      setMessage(`Extracted ${result.count} accepted radio source positions.`);
    } catch (error) {
      reportError(error);
    } finally {
      setLoading(false);
    }
  }

  function exportHref(name: string): string {
    return sessionId ? `/api/sessions/${sessionId}/exports/${name}` : "#";
  }

  function queueTimeSlider(mjd: number) {
    if (!meta) return;
    const next = clamp(mjd, masterMin, masterMax);
    if (next !== timeSliderMjdRef.current) scrubDirectionRef.current = next > timeSliderMjdRef.current ? 1 : -1;
    timeSliderMjdRef.current = next;
    setTimeSliderMjd(next);
    if (timeCommitRafRef.current) return;
    timeCommitRafRef.current = window.requestAnimationFrame(() => {
      timeCommitRafRef.current = 0;
      setMasterCursor(timeSliderMjdRef.current);
    });
  }

  function beginTimeScrub() {
    if (!meta) return;
    setPlaying(false);
    setTimeScrubbing(true);
  }

  function finishTimeScrub() {
    setTimeScrubbing(false);
  }

  function scrubSpectrogramTime(mjd: number) {
    if (!meta || !masterTimes.length || !Number.isFinite(mjd)) return;
    queueTimeSlider(clamp(mjd, masterMin, masterMax));
  }

  function stepTime(delta: number) {
    if (!meta) return;
    let next = currentMjd;
    const direction = delta < 0 ? -1 : 1;
    for (let step = 0; step < Math.abs(delta); step += 1) {
      next = advanceFrameTime(masterTimes, next, direction, timeStepSeconds, masterMin, masterMax, false, masterMin);
    }
    setMasterCursor(next);
  }

  function stepSelectedTrackFrame(delta: number) {
    if (!selectedTrackTimes.length) return;
    const nextFrame = clamp(currentTrackFrame + delta, 0, selectedTrackTimes.length - 1);
    setMasterCursor(selectedTrackTimes[nextFrame], 0);
  }

  function recordingSurfaces(source: RecordingSource): CaptureSurface[] {
    if (source === "left" || source === "right") return [source];
    return source === "workspace" ? ["spectrogram", "left", "right"] : ["left", "right"];
  }

  function noteCaptureDraw(surface: CaptureSurface, mjd: number) {
    const serial = captureDrawSerialRef.current[surface] + 1;
    captureDrawSerialRef.current[surface] = serial;
    captureDrawListenersRef.current.forEach((listener) => listener(surface, mjd, serial));
  }

  function waitForCaptureDraws(surfaces: CaptureSurface[], targetMjd: number, signal: AbortSignal): Promise<void> {
    const baseline = Object.fromEntries(surfaces.map((surface) => [surface, captureDrawSerialRef.current[surface]])) as Record<CaptureSurface, number>;
    const pending = new Set(surfaces);
    return new Promise((resolve, reject) => {
      const cleanup = () => {
        captureDrawListenersRef.current.delete(listener);
        signal.removeEventListener("abort", abort);
      };
      const listener = (surface: CaptureSurface, drawnMjd: number, serial: number) => {
        if (!pending.has(surface) || serial <= baseline[surface] || Math.abs(drawnMjd - targetMjd) > 1e-10) return;
        pending.delete(surface);
        if (!pending.size) {
          cleanup();
          resolve();
        }
      };
      const abort = () => {
        cleanup();
        reject(new DOMException("Recording canceled.", "AbortError"));
      };
      if (signal.aborted) return abort();
      captureDrawListenersRef.current.add(listener);
      signal.addEventListener("abort", abort, { once: true });
    });
  }

  async function presentRecordingFrame(mjd: number, surfaces: CaptureSurface[], signal: AbortSignal) {
    const drawn = waitForCaptureDraws(surfaces, mjd, signal);
    setMasterCursorMjd(mjd);
    if (meta) setTimeIndex(closestIndex(meta.aia.timeMjd, mjd));
    setRecordingDrawNonce((value) => value + 1);
    await drawn;
  }

  function cancelRecording() {
    recordingControllerRef.current?.abort();
    setMessage("Canceling recording...");
  }

  async function startRecording(requestedOptions: RecordingOptions) {
    if (!meta || recordingControllerRef.current || !masterTimes.length) return;
    if (typeof MediaRecorder === "undefined" || typeof HTMLCanvasElement.prototype.captureStream !== "function") {
      reportError(new Error("This browser does not support canvas recording."), "Could not start recording");
      return;
    }
    const options = {
      ...requestedOptions,
      fps: clamp(Math.round(Number(requestedOptions.fps) || DEFAULT_RECORDING_OPTIONS.fps), 1, 60)
    };
    const selectedRange = options.range === "visible"
      ? timeRangeValues(spectrogramTimeRange, meta.spectrogram.timeMjd)
      : [masterMin, masterMax] as [number, number];
    const frameTimes = frameTimesForTimeRange(masterTimes, selectedRange[0], selectedRange[1], timeStepSeconds, masterMin);
    if (!frameTimes.length) {
      reportError(new Error("The selected recording range has no master frames."), "Could not start recording");
      return;
    }

    const controller = new AbortController();
    recordingControllerRef.current = controller;
    const originalCursor = currentMjd;
    const surfaces = recordingSurfaces(options.source);
    let recorder: MediaRecorder | null = null;
    let stream: MediaStream | null = null;
    let stopPromise: Promise<void> | null = null;
    let completed = false;
    setPlaying(false);
    setRecordingOptionsOpen(false);
    try {
      const warmPlan = planFrameCacheWarm(frameTimes.map((mjd) => requestsForMasterTime(mjd)));
      let prepared = warmPlan.retainedIdentities;
      setRecordingStatus({ phase: "preparing", done: prepared, total: warmPlan.totalIdentities, fps: options.fps, startedAt: Date.now() / 1000 });
      setMessage(`Preparing recording ${prepared}/${warmPlan.totalIdentities}`);
      if (warmPlan.missingIdentities) {
        const warmRequests = warmPlan.requests.map((request) => ({
          ...request,
          url: `${request.url}${request.url.includes("?") ? "&" : "?"}warm=1`
        }));
        await warmFrameCache(
          warmRequests,
          controller.signal,
          () => {
            prepared += 1;
            setRecordingStatus((status) => status?.phase === "preparing" ? { ...status, done: prepared } : status);
            setMessage(`Preparing recording ${prepared}/${warmPlan.totalIdentities}`);
          },
          restoreSessionAfterRestart
        );
      }
      if (controller.signal.aborted) throw new DOMException("Recording canceled.", "AbortError");

      const composite = createRecordingComposite(options.source, options.resolution, captureCanvasRefs.current);
      const predictedFormat = browserRecordingFormat();
      if (composite.note) console.info("[recording-composite]", composite.note, `${composite.canvas.width}x${composite.canvas.height}`);
      stream = composite.canvas.captureStream(0);
      const videoTrack = stream.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
      if (!videoTrack) throw new Error("The recording canvas did not create a video track.");
      const chunks: Blob[] = [];

      // Negotiate the actual encoder: pickRecorderConfig gives the best a-priori mimeType for these
      // dimensions, then we still try it (and successive fallbacks) against the real MediaRecoder,
      // because MediaRecorder.isTypeSupported() only validates the codec string in the abstract - the
      // dimension mismatch that produces "The given encoder configuration is not supported by the
      // encoder." only surfaces once construction/start is attempted with the real canvas size.
      // The candidate chain depends on the explicit Format choice: "auto" walks the full negotiation
      // chain (unchanged from before the Format dropdown existed); "mp4"/"webm" pin it to just that
      // family - the RecordingOptionsCard warning already keeps the user from reaching an unsatisfiable
      // "mp4" choice, but the explicit chain here is the actual enforcement of "skip MP4 tiers entirely"
      // for webm and "attempt only MP4 tiers" for mp4.
      const negotiated = pickRecorderConfig(composite.canvas.width, composite.canvas.height);
      const mp4Chain = ["video/mp4;codecs=avc1.640034", "video/mp4;codecs=avc1.42E01E", "video/mp4;codecs=avc1.4D401E"];
      const webmChain = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
      const attemptChain = Array.from(new Set(
        options.formatChoice === "mp4" ? mp4Chain
          : options.formatChoice === "webm" ? webmChain
            : [negotiated.mimeType, ...mp4Chain.slice(1), ...webmChain]
      )).filter((mimeType) => typeof MediaRecorder === "undefined" || MediaRecorder.isTypeSupported(mimeType));
      const bitrate = recordingBitrate(composite.canvas.width, composite.canvas.height, options.fps);
      let usedMimeType = "";
      let attemptedMp4 = false;
      for (const mimeType of attemptChain.length ? attemptChain : [""]) {
        try {
          const candidate = mimeType
            ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond: bitrate })
            : new MediaRecorder(stream, { videoBitsPerSecond: bitrate });
          candidate.start();
          recorder = candidate;
          usedMimeType = mimeType;
          break;
        } catch (negotiationError) {
          attemptedMp4 = attemptedMp4 || mimeType.includes("mp4");
          recorder = null;
        }
      }
      if (!recorder) throw new Error("The given encoder configuration is not supported by the encoder.");
      if (options.formatChoice === "auto" && (predictedFormat.mimeType.includes("mp4") || attemptedMp4) && !usedMimeType.includes("mp4")) {
        setToast({
          id: Date.now(),
          message: `H.264 limit exceeded at ${options.resolution === "2x" ? "2x" : "this size"} - recording as WebM (VP9).`
        });
      }
      stopPromise = new Promise<void>((resolve) => { if (recorder) recorder.onstop = () => resolve(); });
      recorder.ondataavailable = (event) => { if (event.data.size > 0) chunks.push(event.data); };
      let recorderError: Error | null = null;
      recorder.onerror = (event) => {
        const error = (event as Event & { error?: DOMException }).error;
        recorderError = error ?? new Error("The browser video encoder failed.");
      };

      setRecordingStatus({ phase: "recording", done: 0, total: frameTimes.length, fps: options.fps, startedAt: Date.now() / 1000 });
      setMessage(`Recording 0/${frameTimes.length}`);
      const recordingStartedAt = performance.now();
      for (let frame = 0; frame < frameTimes.length; frame += 1) {
        const frameMjd = frameTimes[frame];
        await presentRecordingFrame(frameMjd, surfaces, controller.signal);
        composite.draw(options.burnTimestamp ? mjdToUtc(frameMjd) : null);
        videoTrack.requestFrame();
        setRecordingStatus((status) => status?.phase === "recording" ? { ...status, done: frame + 1 } : status);
        setMessage(`Recording ${frame + 1}/${frameTimes.length}`);
        await waitForRecordingTick(recordingStartedAt + (frame + 1) * 1000 / options.fps, controller.signal);
        if (recorderError) throw recorderError;
      }
      recorder.stop();
      await stopPromise;
      if (controller.signal.aborted) throw new DOMException("Recording canceled.", "AbortError");
      const mimeType = recorder.mimeType || usedMimeType || chunks[0]?.type || "video/webm";
      const extension = mimeType.includes("mp4") ? "mp4" : "webm";
      const blob = new Blob(chunks, { type: mimeType });
      if (!blob.size) throw new Error("The browser produced an empty recording.");
      const startMjd = frameTimes[0];
      const endMjd = frameTimes[frameTimes.length - 1];
      const filename = `${recordingDatasetToken(datasetName)}_${recordingTimeToken(startMjd)}_${recordingTimeToken(endMjd)}.${extension}`;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      const measuredDuration = await recordedVideoDuration(url);
      if (controller.signal.aborted) {
        URL.revokeObjectURL(url);
        throw new DOMException("Recording canceled.", "AbortError");
      }
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 0);
      console.info("[recording-complete]", JSON.stringify({
        filename,
        mimeType,
        size: blob.size,
        durationSeconds: measuredDuration,
        expectedDurationSeconds: frameTimes.length / options.fps,
        frames: frameTimes.length,
        source: options.source,
        resolution: `${composite.canvas.width}x${composite.canvas.height}`,
        geometry: composite.geometry,
        stepSeconds: timeStepSeconds
      }));
      setMessage(`Downloaded ${filename} · ${blob.size} bytes${measuredDuration === null ? "" : ` · ${measuredDuration.toFixed(2)} s`} · ${mimeType}.`);
      completed = true;
    } catch (error) {
      if (controller.signal.aborted || error instanceof DOMException && error.name === "AbortError") {
        setMasterCursorMjd(originalCursor);
        setTimeIndex(closestIndex(meta.aia.timeMjd, originalCursor));
        setRecordingDrawNonce((value) => value + 1);
        setMessage("Recording canceled.");
      } else {
        reportError(error, "Could not record video");
      }
    } finally {
      if (recorder?.state !== "inactive") recorder?.stop();
      if (!completed && stopPromise) await stopPromise.catch(() => undefined);
      stream?.getTracks().forEach((track) => track.stop());
      if (recordingControllerRef.current === controller) recordingControllerRef.current = null;
      setRecordingStatus(null);
    }
  }

  function selectSpectrogramTime(mjd: number) {
    if (!meta) return;
    if (timeCommitRafRef.current) window.cancelAnimationFrame(timeCommitRafRef.current);
    timeCommitRafRef.current = 0;
    setMasterCursor(mjd);
  }

function updateSourceDifference(sourceId: string, role: SourceRole, patch: Partial<DifferenceState>) {
    setSourceDifferences((value) => {
      const axis = role === "radio" ? (meta?.eovsa.timeMjd ?? []) : role === "spectrogram" ? (meta?.spectrogram.timeMjd ?? []) : (meta?.aia.timeMjd ?? []);
      const count = axis.length || 1;
      const current = value[sourceId] ?? defaultDifference(role, meta?.defaults.diffSeconds ?? DEFAULT_DIFF_SECONDS, count);
      const next = { ...current, ...patch };
      if (patch.meanStartIndex !== undefined) next.meanStartMjd = axis[clampIndex(patch.meanStartIndex, current.meanStartIndex, axis.length)] ?? next.meanStartMjd;
      if (patch.meanEndIndex !== undefined) next.meanEndMjd = axis[clampIndex(patch.meanEndIndex, current.meanEndIndex, axis.length)] ?? next.meanEndMjd;
      return { ...value, [sourceId]: next };
    });
  }

  async function addSourceFromPath() {
    if (!meta) {
      setMessage("Load sample data or a manifest before adding a local source path.");
      return;
    }
    if (!confirmDatasetReplace()) return;
    const path = sourceDraft.path.trim();
    if (!path) {
      setMessage("Enter a backend-local file or directory path for the source.");
      return;
    }
    setLoading(true);
    setMessage("Adding local source path...");
    try {
      const label = sourceDraft.label.trim() || path.split("/").filter(Boolean).pop() || "Local source";
      const data = await apiJson<SessionMeta & { addedSource?: SourceMeta }>(`/api/sessions/${sessionId}/sources`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role: sourceDraft.role,
          label,
          format: sourceDraft.format,
          path
        })
      });
      setMeta(data);
      const addedId = data.addedSource?.id ?? sourcesForMeta(data).at(-1)?.id ?? "";
      setSelectedSourceId(addedId);
      setSourceDifferences((value) => ({
        ...defaultDifferences(sourcesForMeta(data), meta.defaults.diffSeconds, meta.aia.times.length),
        ...value,
        ...(addedId ? { [addedId]: defaultDifference(sourceDraft.role, meta.defaults.diffSeconds, meta.aia.times.length) } : {})
      }));
      setSourceDraft({ role: "context", label: "", format: "fits", path: "" });
      setMessage("Added source path as a placeholder. Rendering requires a matching loader adapter.");
    } catch (error) {
      reportError(error);
    } finally {
      setLoading(false);
    }
  }

  async function refreshRadioPeakValues(layer?: LayerState) {
    if (!meta) return;
    setLoading(true);
    setMessage("Refreshing true global radio peak values...");
    try {
      const difference = differenceForLayer(layer, radioDifference);
      const response = await apiJson<{ key: string; values: number[]; radioPeakCache: Record<string, number[]>; radioPeakTableCache: Record<string, number[][]> }>(`/api/sessions/${sessionId}/radio/peak-cache/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          diffSeconds: numberValue(difference.cadenceSeconds, DEFAULT_DIFF_SECONDS),
          differenceMode: legacyDifferenceMode(difference),
          differenceOperation: difference.operation,
          differenceReference: difference.reference,
          meanStartMjd: difference.meanStartMjd ?? meta.eovsa.timeMjd[Math.min(difference.meanStartIndex, difference.meanEndIndex)],
          meanEndMjd: difference.meanEndMjd ?? meta.eovsa.timeMjd[Math.max(difference.meanStartIndex, difference.meanEndIndex)]
        })
      });
      setMeta({ ...meta, radioPeakCache: response.radioPeakCache, radioPeakTableCache: response.radioPeakTableCache });
      setMessage(`Refreshed true global radio peak table for ${response.key}.`);
    } catch (error) {
      reportError(error);
    } finally {
      setLoading(false);
    }
  }

  function panelHeaderTitle(slot: PanelSlotId, fallback: string): string {
    const layers = panelLayers[slot];
    const composition = compositionFor(layers, panelCompositions[slot]?.baseLayerId, panelCompositions[slot]?.overlayLayerIds);
    const rawBaseLayer = layers.find((layer) => layer.id === composition.baseLayerId);
    const baseLayer = rawBaseLayer ? resolvedLayer(rawBaseLayer, allPanelLayers) : undefined;
    const contoursVisible = composition.overlayLayerIds.some((id) => {
      const layer = layers.find((candidate) => candidate.id === id);
      return layer ? resolvedLayer(layer, allPanelLayers).kind === "contours" && resolvedLayer(layer, allPanelLayers).visible : false;
    });
    return `${baseLayer?.label || fallback}${contoursVisible ? " + contours" : ""}`;
  }

  const statusOperations = progressOperations.filter(isActiveProgressOperation);
  if (warmCacheStatus.active && warmCacheStatus.done < warmCacheStatus.total) {
    statusOperations.push({
      opId: "visible-cache-warm",
      label: "Warming visible range",
      done: warmCacheStatus.done,
      total: warmCacheStatus.total,
      startedAt: warmCacheStatus.startedAt
    });
  }
  if (recordingStatus?.phase === "preparing" && recordingStatus.done < recordingStatus.total) {
    statusOperations.push({
      opId: "recording-preparation",
      label: `Preparing recording ${recordingStatus.done}/${recordingStatus.total}`,
      done: recordingStatus.done,
      total: recordingStatus.total,
      startedAt: recordingStatus.startedAt
    });
  }
  if (
    prewarmStatus?.active
    && prewarmStatus.done < prewarmStatus.total
    && !statusOperations.some((operation) => operation.label === "Warming render cache")
  ) {
    statusOperations.push({
      opId: "legacy-prewarm",
      label: "Warming render cache",
      done: prewarmStatus.done,
      total: prewarmStatus.total,
      startedAt: Date.now() / 1000
    });
  }
  if (loading && !statusOperations.length) {
    const parsed = statusProgress(message);
    if (parsed?.total === undefined || parsed.completed < parsed.total) {
      statusOperations.push({
        opId: "frontend-operation",
        label: message.replace(/\.{3}$/, ""),
        done: parsed?.completed ?? 0,
        total: parsed?.total ?? null,
        startedAt: Date.now() / 1000
      });
    }
  }
  const activeProgress = statusOperations.reduce<ProgressOperation | null>((latest, operation) => (
    !latest || operation.startedAt >= latest.startedAt ? operation : latest
  ), null);
  const progressPercent = activeProgress?.total && activeProgress.total > 0
    ? Math.round(100 * activeProgress.done / activeProgress.total)
    : null;
  const progressLabel = activeProgress
    ? `${activeProgress.label}${statusOperations.length > 1 ? ` +${statusOperations.length - 1} more` : ""}`
    : "";
  const progressDetail = activeProgress
    ? `${progressLabel} ${progressPercent !== null
      ? `${progressPercent}%`
      : `${Math.max(0, Math.floor(Date.now() / 1000 - activeProgress.startedAt))}s`}`
    : "";
  const statusBusy = loading || recordingStatus !== null || statusOperations.length > 0;

  // Pre-flight check for the "explicit MP4" format choice: probe the composite dimensions the current
  // Source/Resolution options would actually produce (reusing createRecordingComposite - it's cheap to
  // call without invoking .draw()) and warn/disable Start before the user ever hits the encoder, rather
  // than discovering the H.264 level mismatch only once startRecording negotiates it.
  let recordingFormatWarning = "";
  if (recordingOptionsOpen && meta && !recordingStatus && recordingOptions.formatChoice === "mp4") {
    try {
      const probe = createRecordingComposite(recordingOptions.source, recordingOptions.resolution, captureCanvasRefs.current);
      if (!mp4FitsAtSize(probe.canvas.width, probe.canvas.height)) {
        recordingFormatWarning = "Too large for MP4 at this resolution - use Auto or WebM";
      }
    } catch {
      // Source canvas not mounted yet (e.g. panel still loading) - nothing to warn about until it is.
    }
  }

  return (
    <main className="app-shell" style={{ gridTemplateColumns: `${layout.railWidth}px 6px minmax(0, 1fr)` }}>
      <header className="topbar">
        <div className="brand">
          <Sparkles size={18} />
          <div>
            <strong>SolRadViewer</strong>
            <span>Radio and context imaging workbench</span>
          </div>
        </div>
        <div className="topbar-spacer" />
        <div className="topbar-status" role="status" aria-live="polite" aria-busy={statusBusy}>
          <span className={`status-dot ${toast ? "error" : statusBusy ? "busy" : "idle"}`} aria-hidden="true" />
          <span className="topbar-status-copy">
            <span className="topbar-status-text">{message}</span>
            <span className={`topbar-operation ${activeProgress ? "visible" : ""}`} aria-hidden={!activeProgress}>
              <span className="topbar-operation-label">{progressDetail || "Idle"}</span>
              <span
              className={`topbar-progress ${progressPercent !== null ? "determinate" : ""}`}
              role="progressbar"
              aria-label={progressLabel || "Operation progress"}
              aria-valuemin={progressPercent !== null ? 0 : undefined}
              aria-valuemax={activeProgress?.total ?? undefined}
              aria-valuenow={progressPercent !== null ? activeProgress?.done : undefined}
              aria-valuetext={progressPercent !== null ? `${activeProgress?.done}/${activeProgress?.total}` : "In progress"}
            >
                {progressPercent !== null && <span className="topbar-progress-fill" style={{ width: `${progressPercent}%` }} />}
              </span>
            </span>
          </span>
        </div>
        {dirty && <span className="dirty-indicator" title="Unsaved analysis changes">Unsaved</span>}
        <button
          className="keyboard-help"
          type="button"
          aria-label="Keyboard shortcuts"
          title="Keyboard shortcuts: Left/Right frame step · Shift+Left/Right ×10 · Space play/pause · Home/End range start/end"
        >?
        </button>
      </header>
      {toast && <div className="toast-error" role="alert" aria-live="assertive"><span>{toast.message}</span><button type="button" aria-label="Dismiss error" onClick={() => setToast(null)}>×</button></div>}
      <aside className="control-rail">
        <section
          className="tool-panel data-panel"
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = "copy";
          }}
          onDrop={(event) => {
            event.preventDefault();
            const file = event.dataTransfer.files?.[0];
            if (file) void loadJsonFile(file);
          }}
        >
          <button className="panel-title section-header" type="button" aria-expanded={!collapsedSections.includes("data")} onClick={() => toggleSection("data")}>
            <span className="section-chevron" aria-hidden="true">{collapsedSections.includes("data") ? "▸" : "▾"}</span>
            <Database size={15} /> Data
          </button>
          <div className={`section-content ${collapsedSections.includes("data") ? "collapsed" : ""}`}>
          <input
            ref={jsonInputRef}
            className="hidden-input"
            type="file"
            accept="application/json,.json"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void loadJsonFile(file);
            }}
          />
          <button className="button" onClick={() => jsonInputRef.current?.click()} disabled={loading}>
            <Upload size={15} /> Load Manifest / JSON
          </button>
          <div className="drop-hint">Drop a local manifest JSON here</div>
          <div className="add-source-form">
            <div className="two-col">
              <label>Role
                <select value={sourceDraft.role} onChange={(event) => setSourceDraft((value) => ({ ...value, role: event.target.value }))}>
                  <option value="context">Context</option>
                  <option value="radio">Radio</option>
                  <option value="spectrogram">Spectrogram</option>
                </select>
              </label>
              <label>Format
                <select value={sourceDraft.format} onChange={(event) => setSourceDraft((value) => ({ ...value, format: event.target.value }))}>
                  <option value="fits">FITS/FTS</option>
                  <option value="hdf">HDF</option>
                  <option value="npz">NPZ</option>
                  <option value="unknown">Unknown</option>
                </select>
              </label>
            </div>
            <label>Label
              <input value={sourceDraft.label} onChange={(event) => setSourceDraft((value) => ({ ...value, label: event.target.value }))} placeholder="Optional source label" />
            </label>
            <label>Local path
              <span className="local-path-input-row">
                <input value={sourceDraft.path} onChange={(event) => setSourceDraft((value) => ({ ...value, path: event.target.value }))} placeholder="/path/to/file.fits or /path/to/fits_dir" />
                <button className="button local-path-browse-button" type="button" title="Browse server files" aria-label="Browse server files" onClick={() => setFilePickerOpen(true)}>
                  <FolderOpen size={15} aria-hidden="true" /> Browse
                </button>
              </span>
            </label>
            <button className="button" onClick={() => void addSourceFromPath()} disabled={loading || !meta}>
              <Plus size={15} /> Add Source Path
            </button>
          </div>
          {meta && (
            <div className="source-stack">
              {sources.map((source) => {
                const role = sourceRole(source, sourceRoles);
                return (
                  <div
                    key={source.id}
                    className={`source-row ${selectedSource?.id === source.id ? "active" : ""}`}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedSourceId(source.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setSelectedSourceId(source.id);
                      }
                    }}
                  >
                    <span className={`role-badge role-${role}`}>{role}</span>
                    <span className="source-main">
                      <strong>{source.label}</strong>
                      <small>{sourceInfo(source)}</small>
                    </span>
                    {role === "radio" && <button className="button icon-button source-alignment-button" type="button" aria-label={`Open ${source.label} channel inspector`} title="Open channel inspector" onClick={(event) => { event.stopPropagation(); setSelectedSourceId(source.id); setAlignmentOpen(true); }}><Crosshair size={15} aria-hidden="true" /></button>}
                    <span className={`source-status ${source.status === "placeholder" ? "placeholder" : ""}`}>{source.status ?? "ready"}</span>
                  </div>
                );
              })}
              {selectedSource && (
                <div className="source-knobs">
                  <label>Source role
                    <select
                      value={sourceRole(selectedSource, sourceRoles)}
                      disabled={selectedSource.role === "context" || selectedSource.role === "radio" || selectedSource.role === "spectrogram" || selectedSource.status !== "placeholder"}
                      onChange={(event) => {
                        const nextRole = event.target.value;
                        setSourceRoles((value) => ({ ...value, [selectedSource.id]: nextRole }));
                        updateSourceDifference(selectedSource.id, nextRole, defaultDifference(nextRole, meta.defaults.diffSeconds, meta.aia.times.length));
                      }}
                    >
                      <option value="context">Context</option>
                      <option value="radio">Radio</option>
                      <option value="spectrogram">Spectrogram</option>
                    </select>
                  </label>
                </div>
              )}
            </div>
          )}
          </div>
        </section>

        <section className="tool-panel layers-panel">
          <button className="panel-title section-header" type="button" aria-expanded={!collapsedSections.includes("layers")} onClick={() => toggleSection("layers")}>
            <span className="section-chevron" aria-hidden="true">{collapsedSections.includes("layers") ? "▸" : "▾"}</span>
            Layers
          </button>
          <div className={`section-content ${collapsedSections.includes("layers") ? "collapsed" : ""}`}>
            <LayerRail
              panelLayers={panelLayers}
              allLayers={allPanelLayers}
              panelCompositions={panelCompositions}
              sources={sources}
              statuses={Object.fromEntries([...panelRenderLayers.left, ...panelRenderLayers.right].map((renderLayer) => [renderLayer.layer.id, layerStatuses[renderLayer.layer.id] ?? frameResolution(renderLayer.request.key)]))}
              frameStats={Object.fromEntries([...panelRenderLayers.left, ...panelRenderLayers.right].map((renderLayer) => {
                const entry = layerFrameStats[renderLayer.layer.id];
                return [renderLayer.layer.id, entry?.key === renderLayer.request.key ? entry.stats : undefined];
              }))}
              freqGhz={meta?.eovsa.freqGhz ?? []}
              selectedLayerId={selectedLayerId}
              loading={loading}
              onSelect={setSelectedLayerId}
              onChange={updatePanelLayer}
              onAdd={addPanelLayer}
              onMove={movePanelLayer}
              onReorder={reorderPanelLayer}
              onRemove={removePanelLayer}
              onCopy={copyPanelLayer}
              onMirror={mirrorPanelLayer}
              onSetBase={setPanelBase}
              onUnlink={unlinkPanelLayer}
              onRefreshGlobalPeak={(layer) => void refreshRadioPeakValues(layer)}
              onOpenAlignment={() => setAlignmentOpen(true)}
            />
          </div>
        </section>

        <section className="tool-panel">
          <button className="panel-title section-header" type="button" aria-expanded={!collapsedSections.includes("time")} onClick={() => toggleSection("time")}>
            <span className="section-chevron" aria-hidden="true">{collapsedSections.includes("time") ? "▸" : "▾"}</span>
            <Activity size={15} /> Time
          </button>
          <div className={`section-content ${collapsedSections.includes("time") ? "collapsed" : ""}`}>
          <div className="master-clock-row">
            <label>Master clock
              <select value={masterSourceId} onChange={(event) => setMasterSource(event.target.value)} disabled={!meta || recordingStatus !== null}>
                {sources.filter((source) => sourceTimeMjd(source, meta, sourceRoles).length > 0).map((source) => (
                  <option key={source.id} value={source.id}>{source.label} ({sourceRole(source, sourceRoles)})</option>
                ))}
              </select>
            </label>
            <label className="time-step-control">Δt [s]
              <CommitInput
                type="number"
                min={0}
                value={timeStepSeconds}
                disabled={!meta || recordingStatus !== null}
                ariaLabel="Custom time step in seconds"
                scrubStep={1}
                title={`0 uses the native master cadence. Minimum loaded-source cadence: ${minimumTimeStepSeconds || "unavailable"} s`}
                onCommit={(value) => {
                  const next = clampedTimeStepSeconds(value, minimumTimeStepSeconds);
                  setTimeStepSeconds(next);
                  setMasterCursor(currentMjd, next);
                }}
              />
            </label>
          </div>
          {masterWarning && <div className="warning-text">{masterWarning}</div>}
          <label>
            Time {meta ? mjdToUtc(timeScrubbing ? timeSliderMjd : currentMjd) : "--"}
            <div className="slider-row slider-row-time">
              <button className={`button icon-button ${playing ? "active" : ""}`} aria-label={playing ? "Pause time playback" : "Play time sequence"} onClick={() => setPlaying((value) => !value)} disabled={!meta || recordingStatus !== null} title={playing ? "Pause time playback" : "Play time sequence"}>
                {playing ? <Pause size={15} /> : <Play size={15} />}
              </button>
              <button className="button icon-button" aria-label="Previous frame" onClick={() => stepTime(-1)} disabled={!meta || recordingStatus !== null || timePosition.index <= 0} title="Previous frame">
                <ChevronLeft size={16} />
              </button>
              <input
                aria-label="Current time frame"
                type="range"
                min={timeMin}
                max={timeMax}
                step="any"
                value={timeSliderMjd || timeMin}
                onChange={(event) => queueTimeSlider(Number(event.target.value))}
                onPointerDown={beginTimeScrub}
                onPointerUp={finishTimeScrub}
                onPointerCancel={finishTimeScrub}
                onBlur={() => { if (timeScrubbing) finishTimeScrub(); }}
                disabled={!meta || recordingStatus !== null}
              />
              <button className="button icon-button" aria-label="Next frame" onClick={() => stepTime(1)} disabled={!meta || recordingStatus !== null || timePosition.index >= timePosition.total - 1} title="Next frame">
                <ChevronRight size={16} />
              </button>
            </div>
          </label>
          <div className="master-range-readout">Range: {mjdToUtc(masterMin)} → {mjdToUtc(masterMax)} · {timePosition.index + 1}/{timePosition.total}</div>
          <div className="two-col master-range-controls">
            <label>Master start<CommitInput value={mjdToUtc(masterMin)} onCommit={(value) => setMasterStartMjd(clamp(parseTimeMjd(value, masterMin), masterFullBounds[0], masterMax))} disabled={!meta} /></label>
            <label>Master end<CommitInput value={mjdToUtc(masterMax)} onCommit={(value) => setMasterEndMjd(clamp(parseTimeMjd(value, masterMax), masterMin, masterFullBounds[1]))} disabled={!meta} /></label>
          </div>
          <div className="time-output-row">
            <label className="playback-speed-control">
              Playback speed
              <select value={playbackFps} onChange={(event) => setPlaybackFps(Number(event.target.value))} disabled={!meta || recordingStatus !== null}>
                {PLAYBACK_FPS_OPTIONS.map((fps) => <option key={fps} value={fps}>{fps} fps</option>)}
              </select>
            </label>
            <button className="button record-options-button" type="button" onClick={() => setRecordingOptionsOpen(true)} disabled={!meta || recordingStatus !== null}>
              <Video size={15} aria-hidden="true" /> Record
            </button>
          </div>
          {recordingStatus && (
            <div className={`recording-indicator ${recordingStatus.phase}`} role="status" aria-live="polite">
              <span className="recording-dot" aria-hidden="true" />
              <strong>{recordingStatus.phase === "preparing" ? "Preparing" : "Recording"}</strong>
              <span>{recordingStatus.phase === "recording"
                ? `${formatRecordingClock(recordingStatus.done / recordingStatus.fps)} / ${formatRecordingClock(recordingStatus.total / recordingStatus.fps)}`
                : `${recordingStatus.done}/${recordingStatus.total}`}</span>
              <button className="button" type="button" onClick={cancelRecording}>Cancel</button>
            </div>
          )}
          </div>
        </section>

        <section className="tool-panel">
          <button className="panel-title section-header" type="button" aria-expanded={!collapsedSections.includes("roi")} onClick={() => toggleSection("roi")}>
            <span className="section-chevron" aria-hidden="true">{collapsedSections.includes("roi") ? "▸" : "▾"}</span>
            <LassoSelect size={15} /> ROI / Alignment
          </button>
          <div className={`section-content ${collapsedSections.includes("roi") ? "collapsed" : ""}`}>
          <button className={`button ${lassoEnabled ? "active" : ""}`} onClick={() => setLassoEnabled((value) => !value)} disabled={!meta}>
            <LassoSelect size={15} /> {lassoEnabled ? "Lasso Active" : "Lasso ROI"}
          </button>
          <div className="two-col">
            <label>X offset [arcsec]<CommitInput type="number" scrubStep={0.5} value={xOffset} onCommit={(value) => { setXOffset(value); void refreshBothRoiProjections({ xOffset: value }); }} /></label>
            <label>Y offset [arcsec]<CommitInput type="number" scrubStep={0.5} value={yOffset} onCommit={(value) => { setYOffset(value); void refreshBothRoiProjections({ yOffset: value }); }} /></label>
          </div>
          <button className="button" onClick={() => { setRoiWorld([]); setRoiAia([]); setRoiEovsa([]); }} disabled={!roiWorld.length && !roiAia.length && !roiEovsa.length}>
            <RotateCcw size={15} /> Clear ROI
          </button>
          </div>
        </section>

        <section className="tool-panel">
          <button className="panel-title section-header" type="button" aria-expanded={!collapsedSections.includes("tracking")} onClick={() => toggleSection("tracking")}>
            <span className="section-chevron" aria-hidden="true">{collapsedSections.includes("tracking") ? "▸" : "▾"}</span>
            <ArrowDownToLine size={15} /> Feature Tracking
          </button>
          <div className={`section-content ${collapsedSections.includes("tracking") ? "collapsed" : ""}`}>
          <div className="tracking-entry-grid">
            <button
              className={`button tracking-seed-button ${seedMode ? "active" : ""}`}
              type="button"
              onClick={() => {
                const armed = !seedMode;
                setSeedMode(armed);
                if (armed) {
                  setLassoEnabled(false);
                  setChannelLassoArmed(false);
                  closePixelProbe();
                }
              }}
              disabled={!trackingAvailable || loading}
              title={`Plain-click a ${trackingSourceOption?.label ?? "selected source"} base panel to add one anchored track`}
            >
              <span className="tracking-seed-icon"><Crosshair size={15} /><Pencil size={9} /></span>
              {seedMode ? "Adding seeds" : "Add seeds"}
            </button>
            <button className="button" type="button" onClick={() => setTrackEditorOpen(true)} disabled={!meta}>
              <Settings2 size={15} /> Track editor
            </button>
            <button
              className={`button ${targetDrawArmed ? "active" : ""}`}
              type="button"
              onClick={() => {
                const armed = !targetDrawArmed;
                setTargetDrawArmed(armed);
                setSeedMode(false);
                setLassoEnabled(false);
                setChannelLassoArmed(false);
                setSlitDrawArmed(false);
                setFanRedrawTarget(null);
                if (armed) closePixelProbe();
              }}
              disabled={!targetDrawingAvailable || loading}
              title="Draw one closed loop-top boundary in the left image panel"
            >
              <LassoSelect size={15} /> {targetDrawArmed ? "Draw target…" : "Loop-top target"}
            </button>
            <button className="button" type="button" onClick={() => { setCorrelationCardDismissed(false); setCorrelationCardOpen(true); }} disabled={!meta || correlationTarget.length < 3}>
              <Activity size={15} /> Correlation
            </button>
            <button className="button" type="button" onClick={() => setSlitInspectorOpen(true)} disabled={!meta || !slitSourceOptions.length}>
              <Activity size={15} /> Slits / Time-distance
            </button>
            {correlationTarget.length >= 3 && <button className="button" type="button" onClick={() => { setCorrelationTarget([]); setCorrelationCardOpen(false); try { for (const key of new Set([INITIAL_MANIFEST_NAME, datasetName].filter(Boolean))) localStorage.removeItem(`sad-eovsa:correlation-target:${key}`); } catch { /* storage is optional */ } void apiJson(`/api/sessions/${sessionId}/correlation-target`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target: [] }) }).catch(() => undefined); }}>Clear target</button>}
          </div>
          <div className="tracking-entry-grid">
            <button className="button" type="button" onClick={() => void suggestSeeds()} disabled={!trackingAvailable || !roiWorld.length || loading}>
              <Sparkles size={15} /> Suggest seeds
            </button>
            <button className="button primary" type="button" onClick={() => void runAutoTrack(sadTracks.map((track) => track.id))} disabled={!allTrackSourcesAvailable || !sadTracks.length || loading}>
              <Play size={15} /> Auto-track all
            </button>
          </div>
          {seedSuggestions.length > 0 && <div className="tracking-suggestion-actions">
            <button className="button" type="button" onClick={() => void acceptAllSuggestions()}>Accept all ({seedSuggestions.length})</button>
            <button className="button" type="button" onClick={() => setSeedSuggestions([])}>Clear suggestions</button>
          </div>}
          <button className="button" onClick={runSadTracking} disabled={!meta || loading} title="Load and refine the legacy markpos.pickle points configured by this manifest.">
            <Upload size={15} /> Load seeds from file
          </button>
          <button className="button" onClick={runEovsaExtraction} disabled={!meta || loading}><Sparkles size={15} /> Extract Radio</button>
          <div className="metrics">
            <span><strong>{sadTracks.length}</strong> tracks</span>
            <span><strong>{sadTracks.reduce((count, track) => count + track.points.length, 0)}</strong> points</span>
            <span><strong>{eovsaSources.length}</strong> radio sources</span>
          </div>
          </div>
        </section>

        <section className="tool-panel">
          <button className="panel-title section-header" type="button" aria-expanded={!collapsedSections.includes("exports")} onClick={() => toggleSection("exports")}>
            <span className="section-chevron" aria-hidden="true">{collapsedSections.includes("exports") ? "▸" : "▾"}</span>
            <Download size={15} /> Exports
          </button>
          <div className={`section-content ${collapsedSections.includes("exports") ? "collapsed" : ""}`}>
          <div className="export-grid">
            <a className="button" href={exportHref("sad_tracks.csv")}>SAD CSV</a>
            <a className="button" href={exportHref("eovsa_sources.csv")}>EOVSA CSV</a>
            <a className="button" href={exportHref("feature_tracks.csv")}>Feature CSV</a>
            <a className="button" href={exportHref("radio_sources.csv")}>Radio CSV</a>
            <a className="button" href={exportHref("aia_sad_track_map.png")}>AIA PNG</a>
            <a className="button" href={exportHref("eovsa_source_time_map.png")}>EOVSA PNG</a>
            <button className="button" onClick={() => void saveSessionJson()} disabled={!meta}>
              <Download size={15} /> JSON
            </button>
          </div>
          </div>
        </section>
      </aside>

      <div className="vertical-resizer rail-resizer" onPointerDown={(event) => startLayoutDrag("rail", event)} title="Resize controls" />

      <section ref={workspaceRef} className={`workspace ${recordingStatus ? "recording-active" : ""}`} aria-busy={loading || recordingStatus !== null} style={{ gridTemplateRows: slitLaneVisible
        ? `${layout.spectrogramHeight}px 6px ${layout.slitLaneHeight}px 6px minmax(${MIN_PANEL_GRID_HEIGHT}px, 1fr)`
        : `${layout.spectrogramHeight}px 6px minmax(0, 1fr)` }}>
        {!meta ? <div className="empty-workspace" role="status">
          <Database size={34} aria-hidden="true" />
          <h1>Load a solar dataset</h1>
          <p>Open a manifest or saved session to compare context and radio imagery.</p>
          <button className="button primary" type="button" onClick={() => jsonInputRef.current?.click()}><Upload size={15} /> Load Manifest / JSON</button>
        </div> : <>
        <SpectrogramPanel
          title={spectrogramSource?.label ?? "Dynamic Spectrum"}
          imageUrl={spectrogramUrl}
          timestamp={aiaTimestamp}
          timeMjd={meta?.spectrogram.timeMjd ?? []}
          availableTimeMjd={meta?.eovsa.timeMjd ?? []}
          radioFreqGhz={meta?.eovsa.freqGhz ?? []}
          freqGhz={meta?.spectrogram.freqGhz ?? []}
          frequencyScale={spectrogramFrequencyScale}
          frequencyInverted={spectrogramFrequencyInverted}
          frequencyRange={spectrogramFrequencyRange}
          alignmentOpen={alignmentOpen}
          selectedChannels={selectedChannels}
          maskedChannels={channelOffsets.masked}
          channelPalette={alignmentPalette}
          onChannelSelectionChange={setSelectedChannels}
          onChannelMaskToggle={(index) => setChannelOffsets((current) => ({
            ...current,
            masked: current.masked.map((value, channel) => channel === index ? !value : value)
          }))}
          timeRange={spectrogramTimeRange}
          currentMjd={currentMjd}
          onTimeSelect={selectSpectrogramTime}
          onTimeScrubStart={beginTimeScrub}
          onTimeScrub={scrubSpectrogramTime}
          onTimeScrubEnd={finishTimeScrub}
          playing={playing}
          buffering={playing && playbackBuffer.total > 0 && playbackBuffer.ahead * 2 < playbackBuffer.total}
          playbackFps={playbackFps}
          onPlaybackToggle={() => setPlaying((value) => !value)}
          onPlaybackFpsChange={setPlaybackFps}
          onStepTime={stepTime}
          warmCacheStatus={warmCacheStatus}
          warmCacheStale={warmStaleReason !== null}
          warmCacheTitle={warmCacheTitle}
          onWarmCache={() => void warmVisibleRange()}
          onOpenDisplay={() => setSpectrogramDisplayOpen(true)}
          onTimeRangeChange={(value) => setSpectrogramTimeRange(coerceTimeRange(value, meta?.spectrogram.timeMjd ?? []))}
          onFrequencyRangeChange={(value) => setSpectrogramFrequencyRange(coerceFrequencyRange(value, meta?.spectrogram.freqGhz ?? []))}
          arrivalTicks={correlationTicks}
          arrivalTicksVisible={correlationTicksVisible}
          arrivalFullHeight={correlationFullHeightTicks}
          showPeakDecel={correlationPeakDecel}
          coverageEnabled={showCacheCoverage}
          coverageSettingsSignature={warmSettingsSignature}
          masterTimeMjd={masterTimes}
          coverageRequestsAt={requestsForMasterTime}
          captureCanvasRef={captureCanvasRefs.current}
          drawNonce={recordingDrawNonce}
          onDrawComplete={(mjd) => noteCaptureDraw("spectrogram", mjd)}
        />
        <div className="horizontal-resizer" onPointerDown={(event) => startLayoutDrag("spectrogram", event)} title="Resize spectrogram" />
        {slitLaneVisible && selectedSlit && (
          <TimeDistanceLane
            slit={selectedSlit}
            raster={twinRaster}
            contours={twinContours}
            missingTwinHint={twinMissingHint}
            xRange={slitTimeRange}
            timeMjd={meta?.spectrogram.timeMjd ?? masterTimes}
            channelGutterVisible={alignmentOpen && (meta?.eovsa.freqGhz.length ?? 0) > 0}
            currentMjd={currentMjd}
            radioFrequencyCount={radioFreqGhz.length}
            radioFreqGhz={radioFreqGhz}
            channelPalette={alignmentPalette}
            sessionId={sessionId}
            radioSourceId={radioSourceId}
            canvasRef={slitLaneCanvasRef}
            onTimeSelect={selectSpectrogramTime}
            onTimeScrubStart={beginTimeScrub}
            onTimeScrub={scrubSpectrogramTime}
            onTimeScrubEnd={finishTimeScrub}
            onTimeRangeChange={(value) => setSpectrogramTimeRange(coerceTimeRange(value, meta?.spectrogram.timeMjd ?? []))}
          />
        )}
        {slitLaneVisible && <div className="horizontal-resizer slit-lane-resizer" onPointerDown={(event) => startLayoutDrag("slitLane", event)} title="Resize time-distance lane" />}
        <div
          ref={panelGridRef}
          className="panel-grid"
          style={{ gridTemplateColumns: `minmax(0, ${layout.imageSplit}fr) 6px minmax(0, ${1 - layout.imageSplit}fr)` }}
        >
          <div className="panel-column">
          <ImagePanel
            sessionId={sessionId}
            onSessionNotFound={restoreSessionAfterRestart}
            panel={effectivePanelLayer("left", panelCompositions.left.baseLayerId)?.sourceRoleSnapshot === "radio" ? "eovsa" : "aia"}
            title={panelHeaderTitle("left", contextSource?.label ?? "Context Image")}
            timestamp={aiaTimestamp}
            renderLayers={panelRenderLayers.left}
            scrubRenderLayers={scrubPanelRenderLayers.left}
            scrubCursorMjd={scrubCursorMjd}
            scrubbing={timeScrubbing}
            onLayerReady={(layerId, key, status, stats) => {
              setReadyLayerKeys((value) => ({ ...value, [layerId]: key }));
              if (status) setLayerStatuses((value) => ({ ...value, [layerId]: status }));
              if (stats) setLayerFrameStats((value) => ({ ...value, [layerId]: { key, stats } }));
            }}
            shape={meta?.aia.shape ?? [1, 1]}
            lassoEnabled={lassoEnabled}
            targetDrawArmed={targetDrawArmed}
            slitDrawArmed={slitDrawArmed}
            slitSmoothPx={slitDraftSmoothPx}
            fanDrawStage={fanDrawStage}
            fanRedrawTarget={fanRedrawTarget}
            channelLassoArmed={channelLassoArmed}
            roi={roiAia}
            roiWorld={roiWorld}
            targetWorld={correlationTarget}
            slits={slits}
            selectedSlitId={selectedSlitId}
            fan={fan}
            fanBoundaryDraft={fanBoundaryDraft?.curve ?? []}
            selectedFanMember={selectedFanMember}
            tracks={sadTracks.filter((track) => track.sourceId === leftBaseLayer?.sourceId)}
            selectedTrackId={correlationHoverTrackId || selectedTrackId}
            currentFrameIndex={closestIndex(sourceTimeMjd(sources.find((source) => source.id === leftBaseLayer?.sourceId), meta, sourceRoles), currentMjd)}
            seedMode={seedMode && trackingSourceId === leftBaseLayer?.sourceId}
            seedSuggestions={trackingSourceId === leftBaseLayer?.sourceId ? seedSuggestions : []}
            currentMjd={currentMjd}
            timeMin={timeMin}
            timeMax={timeMax}
            solarView={solarView}
            fitView={meta ? defaultSolarView(meta) : solarView}
            pixelToWorldAffine={meta?.wcs.aia.pixelToWorldAffine ?? IDENTITY_AFFINE}
            worldOffset={[0, 0]}
            sources={panelVisibleSources("left")}
            probePixel={pixelProbe?.slot === "left" ? pixelProbe.pixel : undefined}
            spaceDown={spaceDown}
            alignmentActive={alignmentOpen && selectedChannels.length > 0}
            selectedChannels={selectedChannels}
            channelOffsets={channelOffsets}
            radioFreqGhz={radioFreqGhz}
            contourGlobalOffset={[numberValue(xOffset, 7), numberValue(yOffset, 0)]}
            onRequestCapChange={(cap) => updatePanelFrameCap("left", cap)}
            onImageHoverChange={(value) => { pointerOverImageRef.current = value; }}
            onSolarViewChange={setSolarView}
            onProbe={(panel, pixel, layerId) => openPixelProbe("left", panel, pixel, layerId)}
            onSeed={(pixel) => { setSeedMode(false); void addSeed(pixel); }}
            onSuggestionAccept={(suggestion) => void acceptSuggestion(suggestion)}
            onTrackSelect={selectTrack}
            onAnchorMove={(trackId, frameIndex, pixel) => void promotePointToAnchor(trackId, frameIndex, pixel)}
            onSlitComplete={completeSlitDraw}
            onFanBoundaryComplete={completeFanBoundary}
            onFanRedrawComplete={completeFanRedraw}
            onFanSelect={setSelectedFanMember}
            onFanPromote={promoteFanMember}
            onLassoComplete={(panel, points, sourceId, sourceRoleValue, additive, channelSelect) => targetDrawArmed && panel === "aia"
              ? saveCorrelationTarget(points)
              : channelSelect
              ? selectChannelsByLasso(panel, points, sourceId, sourceRoleValue, additive)
              : saveRoi(panel, points, sourceId, effectivePanelLayer("left", panelCompositions.left.baseLayerId)?.freqIndex, sourceRoleValue)}
            onAlignmentDrag={(_, delta) => updateSelectedChannelOffsets(delta[0], delta[1])}
            captureSurface="left"
            captureCanvasRef={captureCanvasRefs.current}
            drawNonce={recordingDrawNonce}
            onDrawComplete={(mjd) => noteCaptureDraw("left", mjd)}
          />
          </div>
          <div className="vertical-resizer image-resizer" onPointerDown={(event) => startLayoutDrag("image", event)} title="Resize panels" />
          <div className="panel-column">
          <ImagePanel
            sessionId={sessionId}
            onSessionNotFound={restoreSessionAfterRestart}
            panel={effectivePanelLayer("right", panelCompositions.right.baseLayerId)?.sourceRoleSnapshot === "radio" ? "eovsa" : "aia"}
            title={panelHeaderTitle("right", radioSource?.label ?? "Radio")}
            timestamp={eovsaTimestamp}
            renderLayers={panelRenderLayers.right}
            scrubRenderLayers={scrubPanelRenderLayers.right}
            scrubCursorMjd={scrubCursorMjd}
            scrubbing={timeScrubbing}
            onLayerReady={(layerId, key, status, stats) => {
              setReadyLayerKeys((value) => ({ ...value, [layerId]: key }));
              if (status) setLayerStatuses((value) => ({ ...value, [layerId]: status }));
              if (stats) setLayerFrameStats((value) => ({ ...value, [layerId]: { key, stats } }));
            }}
            shape={meta?.eovsa.shape ?? [1, 1]}
            lassoEnabled={lassoEnabled}
            targetDrawArmed={false}
            slitDrawArmed={slitDrawArmed}
            slitSmoothPx={slitDraftSmoothPx}
            fanDrawStage={fanDrawStage}
            fanRedrawTarget={fanRedrawTarget}
            channelLassoArmed={channelLassoArmed}
            roi={roiEovsa}
            roiWorld={roiWorld}
            targetWorld={correlationTarget}
            slits={slits}
            selectedSlitId={selectedSlitId}
            fan={fan}
            fanBoundaryDraft={fanBoundaryDraft?.curve ?? []}
            selectedFanMember={selectedFanMember}
            tracks={sadTracks.filter((track) => track.sourceId === rightBaseLayer?.sourceId)}
            selectedTrackId={correlationHoverTrackId || selectedTrackId}
            currentFrameIndex={closestIndex(sourceTimeMjd(sources.find((source) => source.id === rightBaseLayer?.sourceId), meta, sourceRoles), currentMjd)}
            seedMode={seedMode && trackingSourceId === rightBaseLayer?.sourceId}
            seedSuggestions={trackingSourceId === rightBaseLayer?.sourceId ? seedSuggestions : []}
            sources={panelVisibleSources("right")}
            currentMjd={currentMjd}
            timeMin={timeMin}
            timeMax={timeMax}
            solarView={solarView}
            fitView={meta ? defaultSolarView(meta) : solarView}
            pixelToWorldAffine={meta?.wcs.eovsa.pixelToWorldAffine ?? IDENTITY_AFFINE}
            worldOffset={[numberValue(xOffset, 7), numberValue(yOffset, 0)]}
            probePixel={pixelProbe?.slot === "right" ? pixelProbe.pixel : undefined}
            spaceDown={spaceDown}
            alignmentActive={alignmentOpen && selectedChannels.length > 0}
            selectedChannels={selectedChannels}
            channelOffsets={channelOffsets}
            radioFreqGhz={radioFreqGhz}
            contourGlobalOffset={[numberValue(xOffset, 7), numberValue(yOffset, 0)]}
            onRequestCapChange={(cap) => updatePanelFrameCap("right", cap)}
            onImageHoverChange={(value) => { pointerOverImageRef.current = value; }}
            onSolarViewChange={setSolarView}
            onProbe={(panel, pixel, layerId) => openPixelProbe("right", panel, pixel, layerId)}
            onSeed={(pixel) => { setSeedMode(false); void addSeed(pixel); }}
            onSuggestionAccept={(suggestion) => void acceptSuggestion(suggestion)}
            onTrackSelect={selectTrack}
            onAnchorMove={(trackId, frameIndex, pixel) => void promotePointToAnchor(trackId, frameIndex, pixel)}
            onSlitComplete={completeSlitDraw}
            onFanBoundaryComplete={completeFanBoundary}
            onFanRedrawComplete={completeFanRedraw}
            onFanSelect={setSelectedFanMember}
            onFanPromote={promoteFanMember}
            onLassoComplete={(panel, points, sourceId, sourceRoleValue, additive, channelSelect) => channelSelect
              ? selectChannelsByLasso(panel, points, sourceId, sourceRoleValue, additive)
              : saveRoi(panel, points, sourceId, effectivePanelLayer("right", panelCompositions.right.baseLayerId)?.freqIndex, sourceRoleValue)}
            onAlignmentDrag={(_, delta) => updateSelectedChannelOffsets(delta[0], delta[1])}
            captureSurface="right"
            captureCanvasRef={captureCanvasRefs.current}
            drawNonce={recordingDrawNonce}
            onDrawComplete={(mjd) => noteCaptureDraw("right", mjd)}
          />
          </div>
        </div>
        </>}
        {filePickerOpen && (
          <FilePickerCard
            initialPath={lastBrowsedDirectory}
            onDirectoryChange={setLastBrowsedDirectory}
            onSelect={(path) => {
              setSourceDraft((value) => ({ ...value, path }));
              setFilePickerOpen(false);
            }}
            cardSize={floatingCardSizes.filePicker}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, filePicker: size }))}
            onClose={() => setFilePickerOpen(false)}
          />
        )}
        {correlationCardOpen && !correlationCardDismissed && meta && correlationTarget.length >= 3 && (
          <CorrelationCard
            series={correlationSeries}
            xRange={correlationXRange}
            currentMjd={currentMjd}
            fullRange={correlationFullRange}
            pinTicks={correlationPinTicks}
            fullHeightTicks={correlationFullHeightTicks}
            peakDecel={correlationPeakDecel}
            selectedTrackId={selectedTrackId}
            onFullRange={setCorrelationFullRange}
            onPinTicks={setCorrelationPinTicks}
            onFullHeightTicks={setCorrelationFullHeightTicks}
            onPeakDecel={setCorrelationPeakDecel}
            onSelectTrack={selectTrack}
            onHoverTrack={setCorrelationHoverTrackId}
            onTimeSelect={setMasterCursor}
            cardSize={floatingCardSizes.correlation}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, correlation: size }))}
            onClose={() => { setCorrelationCardDismissed(true); setCorrelationCardOpen(false); setCorrelationHoverTrackId(""); }}
          />
        )}
        {slitInspectorOpen && meta && (
          <SlitInspectorCard
            slits={slits}
            selectedSlitId={selectedSlitId}
            sourceOptions={slitSourceOptions}
            defaultSourceId={slitSourceId}
            draftWidthArcsec={selectedSlit ? resolveSlitGeometry(selectedSlit, slits).widthArcsec : slitDraftWidthArcsec}
            widthArcsecBounds={slitWidthArcsecBounds()}
            draftSmoothPx={slitDraftSmoothPx}
            selectedResult={selectedSlitResult}
            imageLinked={Boolean(selectedSlit?.bindingKind === "image" && selectedSlitImageLayer)}
            laneHasContent={slitLaneHasContent}
            fan={fan}
            selectedFanMember={selectedFanMember}
            fanIntermediateCount={fan?.intermediateCount ?? fanIntermediateCount}
            fanDrawStage={fanDrawStage}
            fanRedrawTarget={fanRedrawTarget}
            radioFreqGhz={radioFreqGhz}
            channelPalette={alignmentPalette}
            extracting={slitExtracting}
            drawArmed={slitDrawArmed}
            pinLane={slitPinLane}
            extractAll={slitExtractAllMode}
            npzHref={selectedSlitResult && selectedSlit
              ? `/api/sessions/${sessionId}/slits/${encodeURIComponent(selectedSlit.id)}/export.npz?shiftSeconds=${encodeURIComponent(selectedSlit.shiftSeconds)}`
              : ""}
            slitWidthScaleArcsec={slitWidthScaleArcsec}
            onSelect={(slitId) => {
              const slit = slits.find((candidate) => candidate.id === slitId);
              setSelectedSlitId(slitId);
              if (slit) {
                setSlitDraftWidthArcsec(resolveSlitGeometry(slit, slits).widthArcsec);
              }
            }}
            onRename={(slitId, name) => patchSlit(slitId, { name })}
            onColor={(slitId, color) => patchSlit(slitId, { color })}
            onVisibility={(slitId) => {
              const slit = slits.find((candidate) => candidate.id === slitId);
              if (slit) patchSlit(slitId, { visible: !slit.visible });
            }}
            onReverse={(slitId) => void reverseSlitDirection(slitId)}
            onDuplicate={duplicateSlit}
            onLink={createLinkedTwin}
            onDelete={deleteSlit}
            onDeleteFan={deleteFan}
            onDefaultSourceChange={setSlitSourceId}
            onSlitSourceChange={rebindSlit}
            onFanSourceChange={rebindFan}
            onWidthChange={(widthArcsec) => {
              const [minArcsec, maxArcsec] = slitWidthArcsecBounds();
              const normalized = Math.round(clamp(widthArcsec, minArcsec, maxArcsec) * 10) / 10;
              setSlitDraftWidthArcsec(normalized);
              // Width is geometry-adjacent (a linked twin follows the
              // original's width, same as its curve) - the Width control is
              // disabled for a selected twin (see SlitInspectorCard), but
              // guard here too in case that ever changes.
              if (selectedSlit && !selectedSlit.linkedTo) {
                patchSlit(selectedSlit.id, { widthArcsec: normalized });
                setSlitResults((current) => { const next = { ...current }; delete next[selectedSlit.id]; return next; });
                invalidateTwinResults(selectedSlit.id);
              }
            }}
            onSmoothCommit={commitSlitSmooth}
            onFanSmoothChange={(smoothPx) => reSmoothFan(smoothPx)}
            onDraw={armSlitDrawing}
            onDrawFan={armFanDrawing}
            onFanCount={setFanCurveCount}
            onSelectFan={setSelectedFanMember}
            onPromoteFan={promoteFanMember}
            onRedrawFanBoundary={armFanRedraw}
            onExtractFanCurve={(memberIndex) => void extractFanMember(memberIndex)}
            onExtract={() => void (slitExtractAllMode ? extractAllSlits() : extractSelectedSlit())}
            onExtractAllChange={setSlitExtractAllMode}
            onCancel={() => void cancelSlitExtraction()}
            onShift={(shiftSeconds) => selectedSlit && patchSlit(selectedSlit.id, { shiftSeconds })}
            onDisplay={(display) => selectedSlit && patchSlit(selectedSlit.id, { display })}
            onFreqSelection={(freqIndices) => {
              if (!selectedSlit) return;
              patchSlit(selectedSlit.id, {
                freqIndices,
                baseFreqIndex: freqIndices.includes(selectedSlit.baseFreqIndex ?? -1) ? selectedSlit.baseFreqIndex : freqIndices[0] ?? null,
                contourFreqIndices: selectedSlit.contourFreqIndices.filter((index) => freqIndices.includes(index))
              });
              setSlitResults((current) => { const next = { ...current }; delete next[selectedSlit.id]; return next; });
            }}
            onBaseFreq={(baseFreqIndex) => {
              if (!selectedSlit) return;
              // Resync the range to the newly chosen base map's own p1..p99
              // so switching channels doesn't leave a mismatched vmin/vmax
              // from whichever channel the range last belonged to (the
              // washed-out-display bug: e.g. a wide-range channel's vmax
              // left in place while displaying a much narrower channel).
              const map = selectedSlitResult ? slitResultMaps(selectedSlitResult).find((candidate) => candidate.freqIndex === baseFreqIndex) : undefined;
              patchSlit(selectedSlit.id, {
                baseFreqIndex,
                ...(map ? { display: { ...selectedSlit.display, vmin: map.dataP1, vmax: map.dataP99 > map.dataP1 ? map.dataP99 : map.dataMax } } : {})
              });
            }}
            onContourFreq={(freqIndexValue) => selectedSlit && patchSlit(selectedSlit.id, {
              contourFreqIndices: selectedSlit.contourFreqIndices.includes(freqIndexValue)
                ? selectedSlit.contourFreqIndices.filter((index) => index !== freqIndexValue)
                : [...selectedSlit.contourFreqIndices, freqIndexValue]
            })}
            onContourLevel={(contourLevelPercent) => selectedSlit && patchSlit(selectedSlit.id, { contourLevelPercent })}
            onPinLane={setSlitPinLane}
            onExportPng={exportSlitLanePng}
            cardSize={floatingCardSizes.slitInspector}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, slitInspector: size }))}
            onClose={() => { setSlitInspectorOpen(false); setSlitDrawArmed(false); setFanDrawStage(0); setFanBoundaryDraft(null); setFanRedrawTarget(null); }}
          />
        )}
        {trackEditorOpen && meta && (
          <TrackEditorCard
            tracks={sadTracks}
            trackingSources={trackingSourceOptions}
            trackingSourceId={trackingSourceId}
            selectedTrackId={selectedTrackId}
            selectedAnchorFrame={selectedAnchorFrame}
            currentFrame={currentTrackFrame}
            currentMjd={timeScrubbing ? timeSliderMjd : currentMjd}
            timeMjd={selectedTrackTimes}
            pixelToWorldAffine={sourceAffine(selectedTrackSourceId)}
            rangeStart={0}
            rangeEnd={Math.max(0, selectedTrackTimes.length - 1)}
            direction={trackingDirection}
            trackingActive={trackingActive}
            canUndo={trackHistoryRef.current.past.length > 0}
            canRedo={trackHistoryRef.current.future.length > 0}
            canSuggest={trackingAvailable && roiWorld.length > 0}
            exportHref={exportHref("feature_tracks.csv")}
            onTrackingSourceChange={(sourceId) => {
              setTrackingSourceId(sourceId);
              setSeedSuggestions([]);
            }}
            onSelectTrack={selectTrack}
            onRename={(trackId, label) => void mutateTrackMetadata(sadTracks.map((track) => track.id === trackId ? { ...track, label } : track), "Renamed track.")}
            onColor={(trackId, color) => void mutateTrackMetadata(sadTracks.map((track) => track.id === trackId ? { ...track, color } : track), "Changed track color.")}
            onVisibility={(trackId) => void mutateTrackMetadata(sadTracks.map((track) => track.id === trackId ? { ...track, visible: !track.visible } : track), "Changed track visibility.")}
            onDeleteTrack={(trackId) => {
              void mutateTrackMetadata(sadTracks.filter((track) => track.id !== trackId), "Deleted track; Undo restores it.");
              if (selectedTrackId === trackId) setSelectedTrackId("");
            }}
            onSelectAnchor={(anchor) => {
              setSelectedAnchorFrame(anchor.frameIndex);
              setMasterCursor(anchor.mjd, 0);
            }}
            onStepFrame={stepSelectedTrackFrame}
            onTimeSelect={queueTimeSlider}
            onTimeScrubStart={beginTimeScrub}
            onTimeScrubEnd={finishTimeScrub}
            onMoveAnchor={(frameIndex, nextFrame) => void moveAnchorInTime(frameIndex, nextFrame)}
            onDeleteAnchor={() => void deleteSelectedAnchor()}
            onAddKeyFrame={() => void addKeyFrameAtCurrentFrame()}
            onDirectionChange={setTrackingDirection}
            onAutoTrack={() => {
              if (selectedTrackId) void runAutoTrack([selectedTrackId]);
            }}
            onStepTrack={() => {
              if (selectedTrackId) void runAutoTrack([selectedTrackId], true);
            }}
            onStop={() => void stopAutoTrack()}
            onUndo={() => void undoTracks()}
            onRedo={() => void redoTracks()}
            onSuggest={() => void suggestSeeds()}
            cardSize={floatingCardSizes.trackEditor}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, trackEditor: size }))}
            onClose={() => setTrackEditorOpen(false)}
          />
        )}
        {pixelProbe && probedLayer && (
          <PixelProbeCard
            layerLabel={probedLayer.label}
            pixel={pixelProbe.pixel}
            patchRadius={pixelProbe.patchRadius}
            data={pixelProbeData}
            loading={pixelProbeLoading}
            error={pixelProbeError}
            currentMjd={currentMjd}
            onClose={closePixelProbe}
            onRefresh={() => setPixelProbeRefresh((value) => value + 1)}
            cardSize={floatingCardSizes.pixelProbe}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, pixelProbe: size }))}
            onApply={() => {
              if (!pixelProbeData) return;
              const filtered = pixelProbeData.stats.smoothed;
              const stats = filtered?.p1 !== null && filtered?.p1 !== undefined && filtered?.p99 !== null && filtered?.p99 !== undefined
                ? filtered
                : pixelProbeData.stats.raw;
              if (stats.p1 === null || stats.p99 === null || !Number.isFinite(stats.p1) || !Number.isFinite(stats.p99)) return;
              updatePanelLayer(pixelProbe.slot, pixelProbe.layerId, {
                display: { ...probedLayer.display, vmin: String(stats.p1), vmax: String(stats.p99) }
              });
            }}
          />
        )}
        {alignmentOpen && meta && (
          <RadioAlignmentCard
            freqGhz={radioFreqGhz}
            offsets={channelOffsets}
            selectedChannels={selectedChannels}
            lassoArmed={channelLassoArmed}
            palette={alignmentPalette}
            spwGroups={spwGroups}
            onSelectionChange={setSelectedChannels}
            onChange={replaceSelectedChannelOffsets}
            onZeroSelected={() => {
              const next = { dx: [...channelOffsets.dx], dy: [...channelOffsets.dy], masked: [...channelOffsets.masked] };
              selectedChannels.forEach((index) => { next.dx[index] = 0; next.dy[index] = 0; });
              setChannelOffsets(next);
            }}
            onZeroAll={() => setChannelOffsets({ ...zeroChannelOffsets(radioFreqGhz.length), masked: [...channelOffsets.masked] })}
            onMaskSelected={(masked) => setChannelOffsets((current) => ({
              ...current,
              masked: current.masked.map((value, index) => selectedChannels.includes(index) ? masked : value)
            }))}
            onToggleMask={(index) => setChannelOffsets((current) => ({
              ...current,
              masked: current.masked.map((value, channel) => channel === index ? !value : value)
            }))}
            onExport={exportChannelOffsetsCsv}
            onImport={importChannelOffsetsCsv}
            onToggleLasso={() => setChannelLassoArmed((value) => !value)}
            cardSize={floatingCardSizes.channelInspector}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, channelInspector: size }))}
            onClose={closeAlignmentCard}
          />
        )}
        {spectrogramDisplayOpen && meta && (
          <SpectrogramDisplayCard
            display={spectrogramDisplay}
            normalization={spectrogramNormalization}
            frequencyScale={spectrogramFrequencyScale}
            frequencyInverted={spectrogramFrequencyInverted}
            frequencyRange={spectrogramFrequencyRange}
            onDisplayChange={setSpectrogramDisplay}
            onNormalizationChange={setSpectrogramNormalization}
            onFrequencyScaleChange={setSpectrogramFrequencyScale}
            onFrequencyInvertedChange={setSpectrogramFrequencyInverted}
            onFrequencyRangeChange={(value) => setSpectrogramFrequencyRange(coerceFrequencyRange(value, meta.spectrogram.freqGhz))}
            smoothPlayback={smoothPlaybackEnabled}
            onSmoothPlaybackChange={setSmoothPlaybackEnabled}
            showCacheCoverage={showCacheCoverage}
            onShowCacheCoverageChange={setShowCacheCoverage}
            frameCacheGb={frameCacheGb}
            onFrameCacheGbChange={applyFrameCacheGb}
            cardSize={floatingCardSizes.spectrogramDisplay}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, spectrogramDisplay: size }))}
            onClose={() => setSpectrogramDisplayOpen(false)}
          />
        )}
        {recordingOptionsOpen && meta && !recordingStatus && (
          <RecordingOptionsCard
            options={recordingOptions}
            predictedFormat={browserRecordingFormat()}
            formatWarning={recordingFormatWarning}
            onChange={setRecordingOptions}
            onStart={(options) => void startRecording(options)}
            cardSize={floatingCardSizes.recordingOptions}
            onCardSizeChange={(size) => setFloatingCardSizes((current) => ({ ...current, recordingOptions: size }))}
            onClose={() => setRecordingOptionsOpen(false)}
          />
        )}
      </section>
    </main>
  );
}

function SpectrogramPanel({
  title,
  imageUrl,
  timestamp,
  timeMjd,
  availableTimeMjd,
  radioFreqGhz,
  freqGhz,
  frequencyScale,
  frequencyInverted,
  frequencyRange,
  alignmentOpen,
  selectedChannels,
  maskedChannels,
  channelPalette,
  onChannelSelectionChange,
  onChannelMaskToggle,
  timeRange,
  currentMjd,
  onTimeSelect,
  onTimeScrubStart,
  onTimeScrub,
  onTimeScrubEnd,
  playing,
  buffering,
  playbackFps,
  onPlaybackToggle,
  onPlaybackFpsChange,
  onStepTime,
  warmCacheStatus,
  warmCacheStale,
  warmCacheTitle,
  onWarmCache,
  onOpenDisplay,
  onTimeRangeChange,
  onFrequencyRangeChange,
  arrivalTicks = [],
  arrivalTicksVisible = false,
  arrivalFullHeight = false,
  showPeakDecel = false,
  coverageEnabled = false,
  coverageSettingsSignature = "",
  masterTimeMjd = [],
  coverageRequestsAt,
  captureCanvasRef,
  drawNonce,
  onDrawComplete
}: {
  title: string;
  imageUrl: string;
  timestamp: string;
  timeMjd: number[];
  availableTimeMjd: number[];
  radioFreqGhz: number[];
  freqGhz: number[];
  frequencyScale: FrequencyScaleMode;
  frequencyInverted: boolean;
  frequencyRange: FrequencyRangeState;
  alignmentOpen: boolean;
  selectedChannels: number[];
  maskedChannels: boolean[];
  channelPalette: ContourColormap;
  onChannelSelectionChange: (channels: number[]) => void;
  onChannelMaskToggle: (index: number) => void;
  timeRange: TimeRangeState;
  currentMjd: number;
  onTimeSelect: (mjd: number) => void;
  onTimeScrubStart: () => void;
  onTimeScrub: (mjd: number) => void;
  onTimeScrubEnd: () => void;
  playing: boolean;
  buffering: boolean;
  playbackFps: number;
  onPlaybackToggle: () => void;
  onPlaybackFpsChange: (fps: number) => void;
  onStepTime: (delta: number) => void;
  warmCacheStatus: WarmCacheStatus;
  warmCacheStale: boolean;
  warmCacheTitle: string;
  onWarmCache: () => void;
  onOpenDisplay: () => void;
  onTimeRangeChange: (range: TimeRangeState) => void;
  onFrequencyRangeChange: (range: FrequencyRangeState) => void;
  arrivalTicks?: CorrelationTick[];
  arrivalTicksVisible?: boolean;
  arrivalFullHeight?: boolean;
  showPeakDecel?: boolean;
  /** Cache coverage bar toggle (Spectrogram display card, default on). */
  coverageEnabled?: boolean;
  /** Cheap "did the requested identity set change" signature (reuses
   * App's warmSettingsSignature) so the coverage strip recomputes on
   * settings changes without diffing full request objects every poll. */
  coverageSettingsSignature?: string;
  /** Master timeline used to snap each sampled column to "nearest master
   * frame" before deriving its identity set - the same axis playback and
   * prefetch schedule against. */
  masterTimeMjd?: number[];
  /** Same identity builder the panels actually request through
   * (App's requestsForMasterTime, called with no cap override so it is
   * always the full-res identity set regardless of the motion ladder). */
  coverageRequestsAt?: (sampleMjd: number) => ScheduledFrameRequest[];
  captureCanvasRef: Record<CaptureSurface, HTMLCanvasElement | null>;
  drawNonce: number;
  onDrawComplete: (mjd: number) => void;
}) {
  type BoxSelection = { x0: number; y0: number; x1: number; y1: number };
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const gutterCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const imageRequestRef = useRef(0);
  const draggingTimeRef = useRef(false);
  const draggingBoxRef = useRef(false);
  const panRef = useRef<{ pointerId: number; startX: number; startMin: number; startMax: number; moved: boolean } | null>(null);
  const boxSelectionRef = useRef<BoxSelection | null>(null);
  const clickGestureRef = useRef<{ timestamp: number; x: number; y: number; originalMjd: number } | null>(null);
  const gutterDragRef = useRef<{ pointerId: number; startY: number; lastY: number; moved: boolean; additive: boolean; altKey: boolean } | null>(null);
  const suppressGestureRef = useRef(false);
  const spectrogramCursorRef = useRef<"" | "ew-resize">("");
  const cursorGrabWidth = 10;
  const [boxMode, setBoxMode] = useState(false);
  const [boxSelection, setBoxSelection] = useState<BoxSelection | null>(null);
  const channelGutterVisible = alignmentOpen && radioFreqGhz.length > 0;
  // Cache coverage bar: draw() only ever reads coverageColumnsRef (a plain
  // boolean-per-column array), so the per-tick draw cost stays a handful of
  // fillRects - no playback-pacing regression. The actual identity lookups
  // (which call the same requestsForMasterTime App uses for real requests)
  // happen in the throttled effect below, at most once per second unless a
  // genuine settings/time-range change forces an immediate recompute.
  const coverageColumnsRef = useRef<Uint8Array>(new Uint8Array(0));
  const coverageSignatureRef = useRef("");
  const coverageRequestsAtRef = useRef<(sampleMjd: number) => ScheduledFrameRequest[]>(() => []);
  coverageRequestsAtRef.current = coverageRequestsAt ?? (() => []);

  useEffect(() => {
    if (!coverageEnabled) {
      coverageColumnsRef.current = new Uint8Array(0);
      coverageSignatureRef.current = "";
      draw();
      return undefined;
    }
    let cancelled = false;
    const recompute = () => {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0) return;
      const plot = plotRect(rect);
      const [displayTimeMin, displayTimeMax] = timeRangeValues(timeRange, timeMjd);
      const version = frameCacheStateVersion();
      const signature = `${version}|${coverageSettingsSignature}|${displayTimeMin.toPrecision(15)}|${displayTimeMax.toPrecision(15)}|${Math.round(plot.width)}`;
      if (signature === coverageSignatureRef.current) return;
      coverageSignatureRef.current = signature;
      const columnCount = Math.max(1, Math.floor(plot.width / CACHE_COVERAGE_COLUMN_PX));
      const columns = new Uint8Array(columnCount);
      if (displayTimeMax > displayTimeMin) {
        const masterAxis = masterTimeMjd.length ? masterTimeMjd : timeMjd;
        for (let index = 0; index < columnCount; index += 1) {
          const fraction = (index + 0.5) / columnCount;
          const sampleMjd = displayTimeMin + fraction * (displayTimeMax - displayTimeMin);
          const nearestMjd = masterAxis.length ? masterAxis[closestIndex(masterAxis, sampleMjd)] : sampleMjd;
          // A layer that is not displayable at this time under the
          // bracketed-nearest availability rule (frameRequestState ===
          // "unavailable", the same predicate the panels' own draw loop
          // uses to skip rendering it - see the `unavailable` continue
          // above renderLayer drawing) contributes nothing on screen, so it
          // must not block the "instant" verdict below. Only requests for
          // layers that are actually displayable at this column's time
          // count toward "must all be cached".
          const displayableRequests = coverageRequestsAtRef.current(nearestMjd).filter(
            (request) => frameRequestState(request) !== "unavailable"
          );
          columns[index] = displayableRequests.length > 0 && displayableRequests.every((request) => frameRequestState(request) === "cached") ? 1 : 0;
        }
      }
      coverageColumnsRef.current = columns;
      draw();
    };
    recompute();
    const timer = window.setInterval(recompute, CACHE_COVERAGE_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [coverageEnabled, coverageSettingsSignature, timeRange.min, timeRange.max, timeMjd, masterTimeMjd]);

  useEffect(() => {
    const requestId = imageRequestRef.current + 1;
    imageRequestRef.current = requestId;
    if (!imageUrl) {
      imageRef.current = null;
      draw();
      return;
    }
    let cached = imageCache.get(imageUrl);
    if (cached?.complete && cached.naturalWidth <= 0) {
      imageCache.delete(imageUrl);
      cached = undefined;
    }
    if (cached?.complete && cached.naturalWidth > 0) {
      imageRef.current = cached;
      rememberImage(imageUrl, cached);
      draw();
      return;
    }
    const image = cached ?? new Image();
    image.decoding = "async";
    image.onload = () => {
      if (imageRequestRef.current !== requestId) return;
      rememberImage(imageUrl, image);
      imageRef.current = image;
      draw();
    };
    image.onerror = () => {
      if (imageRequestRef.current !== requestId) return;
      imageCache.delete(imageUrl);
      imageRef.current = null;
      draw();
    };
    if (!cached) {
      image.src = imageUrl;
      rememberImage(imageUrl, image);
    }
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    let frame = 0;
    const resize = () => {
      if (frame) window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(draw);
    };
    const observer = canvas && typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    if (canvas) observer?.observe(canvas);
    window.addEventListener("resize", resize);
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener("resize", resize);
    };
  });

  useEffect(() => draw(), [alignmentOpen, currentMjd, timeMjd, availableTimeMjd, radioFreqGhz, freqGhz, frequencyScale, frequencyInverted, frequencyRange.min, frequencyRange.max, timeRange.min, timeRange.max, boxSelection, arrivalTicks, arrivalTicksVisible, arrivalFullHeight, showPeakDecel, coverageEnabled, drawNonce]);

  useEffect(() => {
    drawGutter();
  }, [alignmentOpen, channelPalette, freqGhz, frequencyRange.min, frequencyRange.max, frequencyScale, frequencyInverted, maskedChannels, radioFreqGhz, selectedChannels]);

  function plotRect(rect: DOMRect) {
    const horizontal = spectrogramTimePlotRect(rect, channelGutterVisible);
    const bottomPad = 18;
    return {
      x: horizontal.x,
      y: 4,
      width: horizontal.width,
      height: Math.max(1, rect.height - bottomPad - 4)
    };
  }

  function frequencyWindow(): [number, number] {
    const [fullMin, fullMax] = frequencyBounds(freqGhz);
    const min = clamp(numberValue(frequencyRange.min, fullMin), fullMin, fullMax);
    const max = clamp(numberValue(frequencyRange.max, fullMax), min, fullMax);
    return max > min ? [min, max] : [fullMin, fullMax];
  }

  function frequencyAxis() {
    const [frequencyMin, frequencyMax] = frequencyWindow();
    return frequencyYMapping(frequencyMin, frequencyMax, frequencyScale, frequencyInverted);
  }

  function frequencyFraction(frequency: number): number | null {
    if (!Number.isFinite(frequency)) return null;
    const fraction = frequencyAxis().fractionAtFrequency(frequency);
    return Number.isFinite(fraction) ? fraction : null;
  }

  function frequencyAtGutterY(y: number, plot: ReturnType<typeof plotRect>): number {
    return frequencyAxis().frequencyAtY(y, plot.y, plot.height);
  }

  function gutterPlotPoint(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const plot = plotRect(rect);
    return { y: clamp(event.clientY - rect.top, plot.y, plot.y + plot.height), plot };
  }

  function channelAtGutterY(y: number, plot: ReturnType<typeof plotRect>): number | null {
    let nearest: number | null = null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    radioFreqGhz.forEach((frequency, index) => {
      const fraction = frequencyFraction(frequency);
      if (fraction === null || fraction < 0 || fraction > 1) return;
      const channelY = plot.y + (1 - fraction) * plot.height;
      const distance = Math.abs(channelY - y);
      if (distance < nearestDistance) {
        nearest = index;
        nearestDistance = distance;
      }
    });
    return nearestDistance <= 7 ? nearest : null;
  }

  function channelsInGutterRange(startY: number, endY: number, plot: ReturnType<typeof plotRect>): number[] {
    const startFrequency = frequencyAtGutterY(startY, plot);
    const endFrequency = frequencyAtGutterY(endY, plot);
    const low = Math.min(startFrequency, endFrequency);
    const high = Math.max(startFrequency, endFrequency);
    return radioFreqGhz
      .map((frequency, index) => ({ frequency, index }))
      .filter(({ frequency }) => Number.isFinite(frequency) && frequency >= low && frequency <= high)
      .map(({ index }) => index);
  }

  function drawGutter() {
    const gutter = gutterCanvasRef.current;
    const canvas = canvasRef.current;
    if (!gutter || !canvas || !channelGutterVisible) return;
    const rect = gutter.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (gutter.width !== width || gutter.height !== height) {
      gutter.width = width;
      gutter.height = height;
    }
    const ctx = gutter.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "rgba(11, 15, 20, 0.96)";
    ctx.fillRect(0, 0, rect.width, rect.height);
    const plot = plotRect(canvas.getBoundingClientRect());
    const axis = frequencyAxis();
    radioFreqGhz.forEach((frequency, index) => {
      const fraction = frequencyFraction(frequency);
      if (fraction === null || fraction < 0 || fraction > 1) return;
      const y = axis.yAtFrequency(frequency, plot.y, plot.height);
      const selected = selectedChannels.includes(index);
      const masked = maskedChannels[index] ?? false;
      const color = sampleColormap(channelPalette, index, Math.max(1, radioFreqGhz.length), "frequency");
      const tickWidth = selected ? 11 : masked ? 7 : 5;
      const tickHeight = selected ? 4 : masked ? 4 : 2;
      const x = (rect.width - tickWidth) / 2;
      if (masked) {
        ctx.globalAlpha = selected ? 0.85 : 0.48;
        ctx.strokeStyle = color;
        ctx.lineWidth = selected ? 1.5 : 1;
        ctx.strokeRect(x + 0.5, y - tickHeight / 2 + 0.5, tickWidth - 1, tickHeight - 1);
      } else {
        ctx.globalAlpha = selected ? 1 : 0.82;
        ctx.fillStyle = color;
        ctx.fillRect(x, y - tickHeight / 2, tickWidth, tickHeight);
      }
      if (selected) {
        ctx.globalAlpha = 0.95;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 1, y - tickHeight / 2 - 1, tickWidth + 2, tickHeight + 2);
      }
    });
    ctx.globalAlpha = 1;
  }

  function radioFootprint() {
    if (availableTimeMjd.length < 2 || !radioFreqGhz.length) return null;
    const [timeMin, timeMax] = timeBounds(availableTimeMjd);
    const [frequencyMin, frequencyMax] = frequencyBounds(radioFreqGhz);
    if (!(timeMax > timeMin) || !(frequencyMax > frequencyMin)) return null;
    return { timeMin, timeMax, frequencyMin, frequencyMax };
  }

  function footprintRect(plot: ReturnType<typeof plotRect>) {
    const footprint = radioFootprint();
    if (!footprint) return null;
    const [displayTimeMin, displayTimeMax] = timeRangeValues(timeRange, timeMjd);
    const timeSpan = Math.max(1e-12, displayTimeMax - displayTimeMin);
    const x0 = plot.x + clamp((footprint.timeMin - displayTimeMin) / timeSpan, 0, 1) * plot.width;
    const x1 = plot.x + clamp((footprint.timeMax - displayTimeMin) / timeSpan, 0, 1) * plot.width;
    const axis = frequencyAxis();
    const frequencyY0 = clamp(axis.yAtFrequency(footprint.frequencyMax, plot.y, plot.height), plot.y, plot.y + plot.height);
    const frequencyY1 = clamp(axis.yAtFrequency(footprint.frequencyMin, plot.y, plot.height), plot.y, plot.y + plot.height);
    const y0 = Math.min(frequencyY0, frequencyY1);
    const y1 = Math.max(frequencyY0, frequencyY1);
    if (x1 <= x0 || y1 <= y0) return null;
    return { x0, x1, y0, y1, ...footprint };
  }

  function pointerPoint(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const plot = plotRect(rect);
    return {
      x: clamp(event.clientX - rect.left, plot.x, plot.x + plot.width),
      y: clamp(event.clientY - rect.top, plot.y, plot.y + plot.height),
      plot
    };
  }

  function cursorX(plot: ReturnType<typeof plotRect>): number | null {
    if (timeMjd.length < 2) return null;
    const [displayMin, displayMax] = timeRangeValues(timeRange, timeMjd);
    if (currentMjd < displayMin || currentMjd > displayMax) return null;
    return timeToSharedPlotX(currentMjd, displayMin, displayMax, plot);
  }

  function setSpectrogramCursor(canvas: HTMLCanvasElement, cursor: "" | "ew-resize") {
    if (spectrogramCursorRef.current === cursor) return;
    spectrogramCursorRef.current = cursor;
    canvas.style.cursor = cursor;
  }

  function updateSpectrogramCursor(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = pointerPoint(event);
    const lineX = point ? cursorX(point.plot) : null;
    const nearCursor = !boxMode && !panRef.current && !draggingBoxRef.current && point !== null && lineX !== null && Math.abs(point.x - lineX) <= cursorGrabWidth;
    setSpectrogramCursor(event.currentTarget, draggingTimeRef.current || nearCursor ? "ew-resize" : "");
  }

  function timeFromPointer(event: React.PointerEvent<HTMLCanvasElement>): number | null {
    const canvas = canvasRef.current;
    if (!canvas || timeMjd.length < 2) return null;
    const point = pointerPoint(event);
    if (!point) return null;
    const [tMin, tMax] = timeRangeValues(timeRange, timeMjd);
    return timeFromSharedPlotX(point.x, tMin, tMax, point.plot);
  }

  function selectTimeFromPointer(event: React.PointerEvent<HTMLCanvasElement>, select = onTimeSelect) {
    const mjd = timeFromPointer(event);
    if (mjd !== null) select(mjd);
  }

  function completeBoxSelection(selection: BoxSelection) {
    const pointWidth = Math.abs(selection.x1 - selection.x0);
    const pointHeight = Math.abs(selection.y1 - selection.y0);
    if (pointWidth < 4 || pointHeight < 4) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const plot = plotRect(canvas.getBoundingClientRect());
    const [displayMin, displayMax] = timeRangeValues(timeRange, timeMjd);
    const x0 = clamp((Math.min(selection.x0, selection.x1) - plot.x) / Math.max(1, plot.width), 0, 1);
    const x1 = clamp((Math.max(selection.x0, selection.x1) - plot.x) / Math.max(1, plot.width), 0, 1);
    const nextTimeMin = displayMin + x0 * (displayMax - displayMin);
    const nextTimeMax = displayMin + x1 * (displayMax - displayMin);
    const axis = frequencyAxis();
    const selectedFrequency0 = axis.frequencyAtY(selection.y0, plot.y, plot.height);
    const selectedFrequency1 = axis.frequencyAtY(selection.y1, plot.y, plot.height);
    const nextFrequencyMin = Math.min(selectedFrequency0, selectedFrequency1);
    const nextFrequencyMax = Math.max(selectedFrequency0, selectedFrequency1);
    onTimeRangeChange({ min: mjdToUtc(nextTimeMin), max: mjdToUtc(nextTimeMax) });
    onFrequencyRangeChange({
      min: String(Number(nextFrequencyMin.toFixed(4))),
      max: String(Number(nextFrequencyMax.toFixed(4)))
    });
  }

  function fitView() {
    onTimeRangeChange(coerceTimeRange(undefined, timeMjd));
    onFrequencyRangeChange(coerceFrequencyRange(undefined, freqGhz));
    boxSelectionRef.current = null;
    setBoxSelection(null);
  }

  function rangesMatch(left: [number, number], right: [number, number], tolerance: number): boolean {
    return Math.abs(left[0] - right[0]) <= tolerance && Math.abs(left[1] - right[1]) <= tolerance;
  }

  function handleDoubleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    const point = canvas ? (() => {
      const rect = canvas.getBoundingClientRect();
      const plot = plotRect(rect);
      return {
        x: clamp(event.clientX - rect.left, plot.x, plot.x + plot.width),
        y: clamp(event.clientY - rect.top, plot.y, plot.y + plot.height),
        plot
      };
    })() : null;
    const footprint = point ? footprintRect(point.plot) : null;
    const insideFootprint = Boolean(footprint && point && point.x >= footprint.x0 && point.x <= footprint.x1 && point.y >= footprint.y0 && point.y <= footprint.y1);
    const currentTime = timeRangeValues(timeRange, timeMjd);
    const currentFrequency = frequencyWindow();
    const fullTime = timeBounds(timeMjd);
    const fullFrequency = frequencyBounds(freqGhz);
    const isFullView = rangesMatch(currentTime, fullTime, 2e-5) && rangesMatch(currentFrequency, fullFrequency, 1e-4);
    const isFootprintView = Boolean(footprint && rangesMatch(currentTime, [footprint.timeMin, footprint.timeMax], 2e-5) && rangesMatch(currentFrequency, [footprint.frequencyMin, footprint.frequencyMax], 1e-4));
    event.preventDefault();
    draggingBoxRef.current = false;
    draggingTimeRef.current = false;
    boxSelectionRef.current = null;
    suppressGestureRef.current = false;
    clickGestureRef.current = null;
    setBoxSelection(null);
    if (insideFootprint && footprint && isFootprintView) {
      fitView();
    } else if (insideFootprint && footprint && isFullView) {
      onTimeRangeChange({ min: mjdToUtc(footprint.timeMin), max: mjdToUtc(footprint.timeMax) });
      onFrequencyRangeChange({
        min: String(Number(footprint.frequencyMin.toFixed(4))),
        max: String(Number(footprint.frequencyMax.toFixed(4)))
      });
    } else {
      fitView();
    }
  }

  function zoomOut() {
    const [fullTimeMin, fullTimeMax] = timeBounds(timeMjd);
    const [currentTimeMin, currentTimeMax] = timeRangeValues(timeRange, timeMjd);
    const timeCenter = (currentTimeMin + currentTimeMax) / 2;
    const timeSpan = Math.max(1e-12, currentTimeMax - currentTimeMin) * 1.35;
    const nextTimeMin = clamp(timeCenter - timeSpan / 2, fullTimeMin, fullTimeMax);
    const nextTimeMax = clamp(timeCenter + timeSpan / 2, fullTimeMin, fullTimeMax);
    const [fullFrequencyMin, fullFrequencyMax] = frequencyBounds(freqGhz);
    const currentFrequencyMin = clamp(numberValue(frequencyRange.min, fullFrequencyMin), fullFrequencyMin, fullFrequencyMax);
    const currentFrequencyMax = clamp(numberValue(frequencyRange.max, fullFrequencyMax), fullFrequencyMin, fullFrequencyMax);
    const axis = frequencyYMapping(currentFrequencyMin, currentFrequencyMax, frequencyScale);
    const nextFrequencyMin = clamp(axis.frequencyAtFraction(-0.175), fullFrequencyMin, fullFrequencyMax);
    const nextFrequencyMax = clamp(axis.frequencyAtFraction(1.175), fullFrequencyMin, fullFrequencyMax);
    onTimeRangeChange({ min: mjdToUtc(nextTimeMin), max: mjdToUtc(nextTimeMax) });
    onFrequencyRangeChange({
      min: String(Number(nextFrequencyMin.toFixed(4))),
      max: String(Number(nextFrequencyMax.toFixed(4)))
    });
    boxSelectionRef.current = null;
    setBoxSelection(null);
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    if (!timeMjd.length) return;
    event.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const plot = plotRect(rect);
    const point = { x: clamp(event.clientX - rect.left, plot.x, plot.x + plot.width), plot };
    const [fullMin, fullMax] = timeBounds(timeMjd);
    const [currentMin, currentMax] = timeRangeValues(timeRange, timeMjd);
    const anchor = currentMin + clamp((point.x - point.plot.x) / Math.max(1, point.plot.width), 0, 1) * (currentMax - currentMin);
    const [nextMin, nextMax] = zoomTimeWindow(anchor, currentMin, currentMax, fullMin, fullMax, event.deltaY);
    onTimeRangeChange({ min: mjdToUtc(nextMin), max: mjdToUtc(nextMax) });
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLCanvasElement>) {
    if (!timeMjd.length) return;
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.key === "Home") return onTimeSelect(timeMjd[0]);
    if (event.key === "End") return onTimeSelect(timeMjd[timeMjd.length - 1]);
    const step = event.shiftKey ? 10 : 1;
    onStepTime((event.key === "ArrowRight" ? 1 : -1) * step);
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, rect.width, rect.height);
    const plot = plotRect(rect);
    const plotX = plot.x;
    const plotY = plot.y;
    const plotW = plot.width;
    const plotH = plot.height;
    const [fullTimeMin, fullTimeMax] = timeBounds(timeMjd);
    const [displayTimeMin, displayTimeMax] = timeRangeValues(timeRange, timeMjd);
    if (imageRef.current) {
      const sourceWidth = Math.max(1, imageRef.current.naturalWidth || imageRef.current.width);
      const sourceHeight = Math.max(1, imageRef.current.naturalHeight || imageRef.current.height);
      const sourceStart = clamp((displayTimeMin - fullTimeMin) / Math.max(1e-12, fullTimeMax - fullTimeMin), 0, 1);
      const sourceEnd = clamp((displayTimeMax - fullTimeMin) / Math.max(1e-12, fullTimeMax - fullTimeMin), sourceStart, 1);
      const sourceX = sourceStart * sourceWidth;
      const sourceW = Math.max(1, (sourceEnd - sourceStart) * sourceWidth);
      if (frequencyInverted) {
        ctx.save();
        ctx.translate(0, 2 * plotY + plotH);
        ctx.scale(1, -1);
        ctx.drawImage(imageRef.current, sourceX, 0, sourceW, sourceHeight, plotX, plotY, plotW, plotH);
        ctx.restore();
      } else {
        ctx.drawImage(imageRef.current, sourceX, 0, sourceW, sourceHeight, plotX, plotY, plotW, plotH);
      }
    } else {
      ctx.fillStyle = "#101010";
      ctx.fillRect(plotX, plotY, plotW, plotH);
    }
    const footprint = footprintRect(plot);
    if (footprint) {
      ctx.fillStyle = "rgba(255, 178, 46, 0.11)";
      ctx.fillRect(footprint.x0, footprint.y0, footprint.x1 - footprint.x0, footprint.y1 - footprint.y0);
      ctx.strokeStyle = "rgba(255, 204, 102, 0.95)";
      ctx.lineWidth = 1.4;
      ctx.strokeRect(footprint.x0, footprint.y0, footprint.x1 - footprint.x0, footprint.y1 - footprint.y0);
    }
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, plotY, plotW, plotH);
    if (coverageEnabled) {
      const columns = coverageColumnsRef.current;
      if (columns.length) {
        ctx.fillStyle = "rgba(52, 211, 118, 0.92)";
        const columnWidth = plotW / columns.length;
        for (let index = 0; index < columns.length; index += 1) {
          if (!columns[index]) continue;
          ctx.fillRect(plotX + index * columnWidth, plotY, columnWidth + 0.6, CACHE_COVERAGE_BAR_HEIGHT_PX);
        }
      }
    }
    if (timeMjd.length > 1 && displayTimeMax > displayTimeMin) {
      const tickStepsSeconds = [5, 10, 30, 60, 120, 300, 600, 900, 1800, 3600];
      const displayStartSeconds = (displayTimeMin - 40587) * 86400;
      const displayEndSeconds = (displayTimeMax - 40587) * 86400;
      ctx.font = "9px SF Pro Text, Helvetica, Arial";
      const ticksForStep = (stepSeconds: number) => {
        const ticks: { x: number; label: string }[] = [];
        const firstTickSeconds = Math.ceil((displayStartSeconds - 1e-3) / stepSeconds) * stepSeconds;
        for (let tickSeconds = firstTickSeconds; tickSeconds <= displayEndSeconds + 1e-3; tickSeconds += stepSeconds) {
          const fraction = (tickSeconds - displayStartSeconds) / Math.max(1e-9, displayEndSeconds - displayStartSeconds);
          const x = plotX + fraction * plotW;
          const label = new Date(tickSeconds * 1000).toISOString().slice(11, stepSeconds >= 60 ? 16 : 19);
          const labelHalfWidth = ctx.measureText(label).width / 2;
          if (x - labelHalfWidth < plotX || x + labelHalfWidth > plotX + plotW) continue;
          ticks.push({ x, label });
        }
        return ticks;
      };
      const candidates = tickStepsSeconds.map((stepSeconds) => ({ stepSeconds, ticks: ticksForStep(stepSeconds) }));
      const selectedCandidate = candidates.find(({ ticks }) => ticks.length >= 4 && ticks.length <= 7)
        ?? candidates.reduce((best, candidate) => {
          const distance = candidate.ticks.length < 4 ? 4 - candidate.ticks.length : candidate.ticks.length - 7;
          const bestDistance = best.ticks.length < 4 ? 4 - best.ticks.length : best.ticks.length - 7;
          return distance < bestDistance ? candidate : best;
        });
      const axisY = plotY + plotH;
      ctx.strokeStyle = "#8a93a4";
      ctx.fillStyle = "#bdbdbd";
      ctx.lineWidth = 1;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      selectedCandidate.ticks.forEach(({ x, label }) => {
        ctx.beginPath();
        ctx.moveTo(x, axisY);
        ctx.lineTo(x, axisY + 3);
        ctx.stroke();
        ctx.fillText(label, x, axisY + 5);
      });
    }
    if (arrivalTicksVisible && arrivalTicks.length && displayTimeMax > displayTimeMin) {
      const axisY = plotY + plotH;
      ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      arrivalTicks.forEach((tick) => {
        if (tick.arrivalMjd < displayTimeMin || tick.arrivalMjd > displayTimeMax) return;
        const x = plotX + clamp((tick.arrivalMjd - displayTimeMin) / Math.max(1e-9, displayTimeMax - displayTimeMin), 0, 1) * plotW;
        ctx.strokeStyle = tick.color;
        ctx.fillStyle = tick.color;
        ctx.lineWidth = 1.5;
        if (arrivalFullHeight) {
          ctx.setLineDash([4, 4]);
          ctx.beginPath(); ctx.moveTo(x, plotY); ctx.lineTo(x, axisY); ctx.stroke();
          ctx.setLineDash([]);
        }
        ctx.beginPath(); ctx.moveTo(x, axisY); ctx.lineTo(x, axisY - 9); ctx.stroke();
        ctx.fillText(tick.label, x, axisY - 11);
        if (showPeakDecel && Number.isFinite(tick.peakDecelMjd) && (tick.peakDecelMjd as number) >= displayTimeMin && (tick.peakDecelMjd as number) <= displayTimeMax) {
          const peakX = plotX + clamp(((tick.peakDecelMjd as number) - displayTimeMin) / Math.max(1e-9, displayTimeMax - displayTimeMin), 0, 1) * plotW;
          ctx.beginPath(); ctx.moveTo(peakX, axisY - 2); ctx.lineTo(peakX - 4, axisY - 8); ctx.lineTo(peakX + 4, axisY - 8); ctx.closePath(); ctx.stroke();
        }
      });
    }
    if (timeMjd.length > 1 && currentMjd >= displayTimeMin && currentMjd <= displayTimeMax) {
      const x = timeToSharedPlotX(currentMjd, displayTimeMin, displayTimeMax, plot);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, plotY + plotH);
      ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = "rgba(0, 0, 0, 0.72)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x - 5, 3);
      ctx.lineTo(x + 5, 3);
      ctx.lineTo(x, 10);
      ctx.closePath();
      ctx.moveTo(x - 5, plotY + plotH);
      ctx.lineTo(x + 5, plotY + plotH);
      ctx.lineTo(x, plotY + plotH - 7);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    if (freqGhz.length) {
      const axis = frequencyAxis();
      const tickCount = plotH < 100 ? 3 : 4;
      ctx.font = "10px SF Pro Text, Helvetica, Arial";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let index = 0; index < tickCount; index += 1) {
        const fraction = index / Math.max(1, tickCount - 1);
        const frequency = axis.frequencyAtFraction(fraction);
        const y = axis.yAtFrequency(frequency, plotY, plotH);
        ctx.strokeStyle = "rgba(138, 147, 164, 0.18)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(plotX, y);
        ctx.lineTo(plotX + plotW, y);
        ctx.stroke();
        ctx.fillStyle = "#bdbdbd";
        ctx.fillText(`${frequency.toFixed(1)}`, plotX - (channelGutterVisible ? 22 : 8), clamp(y, plotY + 6, plotY + plotH - 6));
      }
      ctx.save();
      ctx.fillStyle = "#8a93a4";
      ctx.font = "9px SF Pro Text, Helvetica, Arial";
      ctx.textAlign = "center";
      ctx.translate(9, plotY + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Frequency (GHz)", 0, 0);
      ctx.restore();
    }
    if (boxSelection) {
      const x = Math.min(boxSelection.x0, boxSelection.x1);
      const y = Math.min(boxSelection.y0, boxSelection.y1);
      const width = Math.abs(boxSelection.x1 - boxSelection.x0);
      const height = Math.abs(boxSelection.y1 - boxSelection.y0);
      ctx.fillStyle = "rgba(86, 199, 217, 0.18)";
      ctx.fillRect(x, y, width, height);
      ctx.strokeStyle = "rgba(126, 224, 237, 0.95)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);
    }
    drawGutter();
    onDrawComplete(currentMjd);
  }

  function gutterPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    event.stopPropagation();
    if (event.button !== 0 || !channelGutterVisible) return;
    const point = gutterPlotPoint(event);
    gutterDragRef.current = {
      pointerId: event.pointerId,
      startY: point.y,
      lastY: point.y,
      moved: false,
      additive: event.shiftKey,
      altKey: event.altKey
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  }

  function gutterPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    event.stopPropagation();
    const drag = gutterDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const point = gutterPlotPoint(event);
    drag.lastY = point.y;
    if (Math.abs(point.y - drag.startY) > 2) drag.moved = true;
    event.preventDefault();
  }

  function gutterPointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    event.stopPropagation();
    const drag = gutterDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    gutterDragRef.current = null;
    const point = gutterPlotPoint(event);
    if (!drag.moved) {
      const channel = channelAtGutterY(point.y, point.plot);
      if (channel !== null) {
        if (drag.altKey || event.altKey) onChannelMaskToggle(channel);
        else {
          const next = selectedChannels.includes(channel)
            ? selectedChannels.filter((index) => index !== channel)
            : [...selectedChannels, channel];
          onChannelSelectionChange(next.sort((left, right) => left - right));
        }
      }
    } else {
      const channels = channelsInGutterRange(drag.startY, point.y, point.plot);
      const next = drag.additive
        ? [...new Set([...selectedChannels, ...channels])]
        : channels;
      onChannelSelectionChange(next.sort((left, right) => left - right));
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    event.preventDefault();
  }

  function gutterPointerCancel(event: React.PointerEvent<HTMLCanvasElement>) {
    event.stopPropagation();
    gutterDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    event.preventDefault();
  }

  function gutterWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    event.preventDefault();
    event.stopPropagation();
  }

  return (
    <section className="image-panel spectrogram-panel">
      <header>
        <div className="spectrogram-transport" role="group" aria-label="Spectrogram transport">
          <button
            type="button"
            className={playing ? "active" : ""}
            onClick={onPlaybackToggle}
            disabled={!timeMjd.length}
            aria-label={playing ? "Pause playback" : "Play playback"}
            title={playing ? "Pause playback" : "Play playback"}
          >
            {playing ? <Pause size={13} /> : <Play size={13} />}
          </button>
          <select
            aria-label="Playback speed"
            value={playbackFps}
            onChange={(event) => onPlaybackFpsChange(Number(event.target.value))}
            disabled={!timeMjd.length}
          >
            {PLAYBACK_FPS_OPTIONS.map((fps) => <option key={fps} value={fps}>{fps} fps</option>)}
          </select>
          <button type="button" onClick={() => onStepTime(-1)} disabled={!timeMjd.length} aria-label="Previous frame" title="Previous frame">
            <ChevronLeft size={14} />
          </button>
          <button type="button" onClick={() => onStepTime(1)} disabled={!timeMjd.length} aria-label="Next frame" title="Next frame">
            <ChevronRight size={14} />
          </button>
          {buffering && <small className="playback-buffer-note">buffering</small>}
        </div>
        <span className="image-panel-title">{title}</span>
        <small className="spectrogram-tool-state">{timestamp} · {boxMode ? "Box zoom" : "Click / drag to select time"}</small>
        <div className="spectrogram-view-controls" role="group" aria-label="Spectrogram view controls">
          <button
            type="button"
            className={`warm-cache-button ${warmCacheStatus.active ? "active warming" : ""} ${warmCacheStale ? "stale" : ""}`}
            onClick={onWarmCache}
            disabled={!timeMjd.length}
            aria-label={warmCacheStatus.active ? "Cancel cache warming" : "Warm cache for the visible time range"}
            aria-pressed={warmCacheStatus.active}
            title={warmCacheTitle}
          >
            {warmCacheStatus.active
              ? <span className="warm-cache-percent">{Math.round(100 * warmCacheStatus.done / Math.max(1, warmCacheStatus.total))}%</span>
              : <Flame size={14} aria-hidden="true" />}
            {warmCacheStale && !warmCacheStatus.active && <span className="warm-cache-stale-dot" aria-hidden="true" />}
          </button>
          <button type="button" onClick={onOpenDisplay} aria-label="Spectrogram display" title="Spectrogram display"><Settings2 size={14} aria-hidden="true" /></button>
          <button type="button" onClick={zoomOut} disabled={!timeMjd.length || !freqGhz.length} aria-label="Zoom out spectrogram">−</button>
          <button type="button" className={boxMode ? "active" : ""} onClick={() => { setBoxMode((value) => !value); boxSelectionRef.current = null; setBoxSelection(null); }} aria-pressed={boxMode} aria-label="Toggle box zoom mode">Box</button>
          <button type="button" className="fit-button" onClick={fitView} disabled={!timeMjd.length || !freqGhz.length}>Fit</button>
        </div>
      </header>
      <div className="spectrogram-body">
        <canvas
          ref={(canvas) => {
            canvasRef.current = canvas;
            captureCanvasRef.spectrogram = canvas;
          }}
          data-window-start={timeRangeValues(timeRange, timeMjd)[0]}
          data-window-end={timeRangeValues(timeRange, timeMjd)[1]}
          data-frequency-inverted={frequencyInverted ? "true" : "false"}
          data-frequency-min={frequencyWindow()[0]}
          data-frequency-max={frequencyWindow()[1]}
          data-plot-left={channelGutterVisible ? 68 : 54}
          data-plot-right={10}
          tabIndex={0}
          aria-label="Dynamic spectrum timeline. Use arrow keys to move, Home/End for bounds, and mouse wheel to zoom time."
          className={`spectrogram-canvas ${boxMode ? "box-zoom-mode" : ""}`}
        onPointerDown={(event) => {
          if (event.button === 2) return;
          const point = pointerPoint(event);
          if (!point) return;
          const now = performance.now();
          const previous = clickGestureRef.current;
          const isDoubleClick = Boolean(previous && now - previous.timestamp < 450 && Math.hypot(point.x - previous.x, point.y - previous.y) < 12);
          if (isDoubleClick && previous) {
            suppressGestureRef.current = true;
            draggingBoxRef.current = false;
            draggingTimeRef.current = false;
            boxSelectionRef.current = null;
            setBoxSelection(null);
            onTimeSelect(previous.originalMjd);
            updateSpectrogramCursor(event);
            return;
          }
          clickGestureRef.current = { timestamp: now, x: point.x, y: point.y, originalMjd: currentMjd };
          const lineX = cursorX(point.plot);
          const nearCursor = lineX !== null && Math.abs(point.x - lineX) <= cursorGrabWidth;
          if (!boxMode && (event.button === 1 || event.shiftKey || event.altKey || !nearCursor)) {
            const [startMin, startMax] = timeRangeValues(timeRange, timeMjd);
            panRef.current = { pointerId: event.pointerId, startX: point.x, startMin, startMax, moved: false };
            event.currentTarget.setPointerCapture(event.pointerId);
            updateSpectrogramCursor(event);
            event.preventDefault();
            return;
          }
          event.currentTarget.setPointerCapture(event.pointerId);
          if (boxMode) {
            draggingBoxRef.current = true;
            const selection = { x0: point.x, y0: point.y, x1: point.x, y1: point.y };
            boxSelectionRef.current = selection;
            setBoxSelection(selection);
            updateSpectrogramCursor(event);
          } else {
            draggingTimeRef.current = true;
            setSpectrogramCursor(event.currentTarget, "ew-resize");
            onTimeScrubStart();
            selectTimeFromPointer(event, onTimeScrub);
          }
        }}
        onPointerMove={(event) => {
          updateSpectrogramCursor(event);
          if (panRef.current?.pointerId === event.pointerId) {
            const point = pointerPoint(event);
            if (point) {
              const dx = point.x - panRef.current.startX;
              if (!panRef.current.moved && Math.abs(dx) < 2) return;
              panRef.current.moved = true;
              const span = panRef.current.startMax - panRef.current.startMin;
              const delta = dx / Math.max(1, point.plot.width) * span;
              const [fullMin, fullMax] = timeBounds(timeMjd);
              const nextMin = clamp(panRef.current.startMin - delta, fullMin, fullMax - span);
              onTimeRangeChange({ min: mjdToUtc(nextMin), max: mjdToUtc(nextMin + span) });
            }
            event.preventDefault();
          } else if (draggingBoxRef.current) {
            const point = pointerPoint(event);
            if (point) {
              setBoxSelection((selection) => {
                if (!selection) return selection;
                const next = { ...selection, x1: point.x, y1: point.y };
                boxSelectionRef.current = next;
                return next;
              });
            }
          } else if (draggingTimeRef.current) {
            selectTimeFromPointer(event, onTimeScrub);
          }
        }}
        onPointerUp={(event) => {
          if (panRef.current?.pointerId === event.pointerId) {
            const pan = panRef.current;
            panRef.current = null;
            if (!pan.moved && event.button === 0) selectTimeFromPointer(event);
            if (pan.moved) clickGestureRef.current = null;
            if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
            updateSpectrogramCursor(event);
            event.preventDefault();
            return;
          }
          if (suppressGestureRef.current) {
            suppressGestureRef.current = false;
            if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
            updateSpectrogramCursor(event);
            return;
          }
          const selection = boxSelectionRef.current;
          if (draggingBoxRef.current && selection) completeBoxSelection(selection);
          if (draggingTimeRef.current) onTimeScrubEnd();
          draggingBoxRef.current = false;
          draggingTimeRef.current = false;
          boxSelectionRef.current = null;
          if (boxMode) setBoxSelection(null);
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
          updateSpectrogramCursor(event);
        }}
        onPointerCancel={(event) => {
          panRef.current = null;
          suppressGestureRef.current = false;
          draggingBoxRef.current = false;
          if (draggingTimeRef.current) onTimeScrubEnd();
          draggingTimeRef.current = false;
          boxSelectionRef.current = null;
          clickGestureRef.current = null;
          setBoxSelection(null);
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
          updateSpectrogramCursor(event);
        }}
        onPointerLeave={(event) => {
          if (!draggingTimeRef.current) setSpectrogramCursor(event.currentTarget, "");
        }}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
          onKeyDown={handleKeyDown}
        />
        {channelGutterVisible && (
          <canvas
            ref={gutterCanvasRef}
            className="spectrogram-channel-gutter"
            aria-label="Radio channel gutter"
            title="Radio channels — click/drag to select"
            onPointerDown={gutterPointerDown}
            onPointerMove={gutterPointerMove}
            onPointerUp={gutterPointerUp}
            onPointerCancel={gutterPointerCancel}
            onWheel={gutterWheel}
          />
        )}
      </div>
    </section>
  );
}

function LayerRail({
  panelLayers,
  allLayers,
  panelCompositions,
  sources,
  statuses = {},
  frameStats = {},
  freqGhz = [],
  selectedLayerId,
  loading,
  onSelect,
  onChange,
  onAdd,
  onMove,
  onReorder,
  onRemove,
  onCopy,
  onMirror,
  onSetBase,
  onUnlink,
  onRefreshGlobalPeak,
  onOpenAlignment
}: {
  panelLayers: Record<PanelSlotId, LayerState[]>;
  allLayers: LayerState[];
  panelCompositions: Record<PanelSlotId, PanelComposition>;
  sources: SourceMeta[];
  statuses?: Record<string, FrameResolution | undefined>;
  frameStats?: Record<string, FrameStats | undefined>;
  freqGhz?: number[];
  selectedLayerId: SelectedLayer | null;
  loading: boolean;
  onSelect: (selection: SelectedLayer | null) => void;
  onChange: (slot: PanelSlotId, layerId: string, patch: Partial<LayerState>) => void;
  onAdd: (slot: PanelSlotId, kind: LayerKind) => void;
  onMove: (slot: PanelSlotId, layerId: string, delta: number) => void;
  onReorder: (slot: PanelSlotId, fromId: string, toId: string) => void;
  onRemove: (slot: PanelSlotId, layerId: string) => void;
  onCopy: (slot: PanelSlotId, layerId: string) => void;
  onMirror: (slot: PanelSlotId, layerId: string) => void;
  onSetBase: (slot: PanelSlotId, layerId: string) => void;
  onUnlink: (slot: PanelSlotId, layerId: string) => void;
  onRefreshGlobalPeak: (layer: LayerState) => void;
  onOpenAlignment: () => void;
}) {
  return (
    <div className="layer-rail">
      {(["left", "right"] as PanelSlotId[]).map((slot) => (
        <LayerRailGroup
          key={slot}
          slot={slot}
          layers={panelLayers[slot]}
          allLayers={allLayers}
          baseLayerId={panelCompositions[slot].baseLayerId}
          sources={sources}
          statuses={statuses}
          frameStats={frameStats}
          freqGhz={freqGhz}
          selectedLayerId={selectedLayerId}
          loading={loading}
          onSelect={onSelect}
          onChange={onChange}
          onAdd={onAdd}
          onMove={onMove}
          onReorder={onReorder}
          onRemove={onRemove}
          onCopy={onCopy}
          onMirror={onMirror}
          onSetBase={onSetBase}
          onUnlink={onUnlink}
          onRefreshGlobalPeak={onRefreshGlobalPeak}
          onOpenAlignment={onOpenAlignment}
        />
      ))}
    </div>
  );
}

function layerStatusInfo(layer: LayerState, layers: LayerState[], sources: SourceMeta[], status?: FrameResolution) {
  const unavailable = (layer.kind === "contours" && !layers.some((candidate) => candidate.kind === "image")) || isLayerUnavailable(layer, sources) || status?.unavailable;
  if (unavailable) return { unavailable: true, row: "unavailable", full: "Unavailable" };
  if (status && Number.isFinite(status.resolvedMjd)) {
    const delta = "Δ" + Number(status.offsetSeconds ?? 0).toFixed(1) + "s";
    return { unavailable: false, row: delta, full: mjdToUtc(status.resolvedMjd as number) + " · " + delta };
  }
  return { unavailable: false, row: "pending", full: "Pending" };
}

function isLayerUnavailable(layer: LayerState, sources: SourceMeta[]): boolean {
  if (layer.kind === "spectrogram") return true;
  const source = sources.find((candidate) => candidate.id === layer.sourceId);
  if (!source || source.capabilities?.render === false) return true;
  if (layer.kind === "contours") return layer.sourceRoleSnapshot !== "radio" || source.capabilities?.overlay === false;
  return layer.sourceRoleSnapshot !== "context" && layer.sourceRoleSnapshot !== "radio";
}

function sourceSupportsLayer(source: SourceMeta, kind: LayerKind, role = sourceRole(source, {})): boolean {
  if (source.capabilities?.render === false) return false;
  if (kind === "contours") return role === "radio" && source.capabilities?.overlay !== false;
  if (kind === "spectrogram") return role === "spectrogram";
  return role === "context" || role === "radio";
}

function LayerRailGroup({
  slot,
  layers,
  allLayers,
  baseLayerId,
  sources,
  statuses,
  frameStats,
  freqGhz,
  selectedLayerId,
  loading,
  onSelect,
  onChange,
  onAdd,
  onMove,
  onReorder,
  onRemove,
  onCopy,
  onMirror,
  onSetBase,
  onUnlink,
  onRefreshGlobalPeak,
  onOpenAlignment
}: {
  slot: PanelSlotId;
  layers: LayerState[];
  allLayers: LayerState[];
  baseLayerId: string;
  sources: SourceMeta[];
  statuses: Record<string, FrameResolution | undefined>;
  frameStats: Record<string, FrameStats | undefined>;
  freqGhz: number[];
  selectedLayerId: SelectedLayer | null;
  loading: boolean;
  onSelect: (selection: SelectedLayer | null) => void;
  onChange: (slot: PanelSlotId, layerId: string, patch: Partial<LayerState>) => void;
  onAdd: (slot: PanelSlotId, kind: LayerKind) => void;
  onMove: (slot: PanelSlotId, layerId: string, delta: number) => void;
  onReorder: (slot: PanelSlotId, fromId: string, toId: string) => void;
  onRemove: (slot: PanelSlotId, layerId: string) => void;
  onCopy: (slot: PanelSlotId, layerId: string) => void;
  onMirror: (slot: PanelSlotId, layerId: string) => void;
  onSetBase: (slot: PanelSlotId, layerId: string) => void;
  onUnlink: (slot: PanelSlotId, layerId: string) => void;
  onRefreshGlobalPeak: (layer: LayerState) => void;
  onOpenAlignment: () => void;
}) {
  const selectedId = selectedLayerId?.slot === slot ? selectedLayerId.id : null;
  const selectedLayer = selectedId
    ? layers.find((layer) => layer.id === selectedId)
    : undefined;
  const selectedDisplayLayer = selectedLayer ? resolvedLayer(selectedLayer, allLayers) : undefined;
  const displayLayers = layers.map((layer) => resolvedLayer(layer, allLayers));
  const otherSlot: PanelSlotId = slot === "left" ? "right" : "left";
  const [editingLayerId, setEditingLayerId] = useState<string | null>(null);

  function beginRename(layer: LayerState) {
    setEditingLayerId(layer.id);
    onSelect({ slot, id: layer.id });
  }

  function commitRename(layer: LayerState, label: string) {
    const nextLabel = label.trim();
    if (nextLabel) onChange(slot, layer.id, { label: nextLabel, labelEdited: true });
    setEditingLayerId(null);
  }

  return (
    <div className="layer-group" aria-label={slot + " panel layers"}>
      <div className="layer-group-header">
        <strong>{slot === "left" ? "Left panel" : "Right panel"}</strong>
        <span className="layer-group-actions">
          <button className="button" type="button" onClick={() => onAdd(slot, "image")} disabled={layers.length >= 8}>+ Image</button>
          <button className="button" type="button" onClick={() => onAdd(slot, "contours")} disabled={layers.length >= 8}>+ Overlay</button>
        </span>
      </div>
      <div className="layer-stack-list" role="list">
        {layers.length ? layers.map((layer) => {
          const displayLayer = resolvedLayer(layer, allLayers);
          const displayLayers = layers.map((candidate) => resolvedLayer(candidate, allLayers));
          const status = layerStatusInfo(displayLayer, displayLayers, sources, statuses[layer.id]);
          const linked = Boolean(layer.mirrorOf) || allLayers.some((candidate) => candidate.mirrorOf === layer.id);
          const isSelected = selectedId === layer.id;
          return (
            <div
              className={"layer-stack-row " + (isSelected ? "active" : "")}
              key={layer.id}
              role="listitem"
              draggable
              onDragStart={(event) => event.dataTransfer.setData("text/plain", slot + "|" + layer.id)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                const value = event.dataTransfer.getData("text/plain");
                const prefix = slot + "|";
                if (value.startsWith(prefix)) onReorder(slot, value.slice(prefix.length), layer.id);
              }}
            >
              <span className="layer-drag-handle" title="Drag to reorder" aria-label={"Drag " + displayLayer.label + " to reorder"}>⋮⋮</span>
              <button type="button" className="layer-eye" aria-label={(displayLayer.visible ? "Hide " : "Show ") + displayLayer.label} onClick={() => onChange(slot, layer.id, { visible: !displayLayer.visible })}>{displayLayer.visible ? "◉" : "○"}</button>
              {editingLayerId === layer.id ? <>
                <InlineLayerLabelInput
                  value={displayLayer.label}
                  ariaLabel={`Rename ${displayLayer.label}`}
                  className="layer-name-input"
                  onCommit={(value) => commitRename(layer, value)}
                  onCancel={() => setEditingLayerId(null)}
                />
                {linked && <Link2 className="layer-link-glyph" size={11} aria-label="Mirrored layer" />}
              </> : <button
                type="button"
                className="layer-name"
                onClick={() => onSelect(isSelected ? null : { slot, id: layer.id })}
                onDoubleClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  beginRename(layer);
                }}
                title="Double-click to rename"
              >
                <span>{displayLayer.label}</span>
                {linked && <Link2 className="layer-link-glyph" size={11} aria-label="Mirrored layer" />}
                {baseLayerId === layer.id && displayLayer.kind === "image" && <small className="layer-base-badge">base</small>}
              </button>}
              <small className={"layer-status " + (status.unavailable ? "unavailable" : "")} aria-label={displayLayer.label + " status"}>{status.row}</small>
              <button type="button" className="layer-delete" aria-label={"Delete " + displayLayer.label} onClick={() => onRemove(slot, layer.id)}>×</button>
            </div>
          );
        }) : <div className="layer-empty">no layers — add an image layer</div>}
      </div>
      {selectedLayer && (
        <LayerInspector
          slot={slot}
          otherSlot={otherSlot}
          layers={layers}
          layer={selectedDisplayLayer!}
          baseLayerId={baseLayerId}
          sources={sources}
          status={layerStatusInfo(selectedDisplayLayer!, displayLayers, sources, statuses[selectedLayer.id])}
          frameStats={frameStats[selectedLayer.id]}
          freqGhz={freqGhz}
          loading={loading}
          onChange={onChange}
          onMove={onMove}
          onRemove={onRemove}
          onCopy={onCopy}
          onMirror={onMirror}
          onSetBase={onSetBase}
          onUnlink={onUnlink}
          onRefreshGlobalPeak={onRefreshGlobalPeak}
          onOpenAlignment={onOpenAlignment}
        />
      )}
    </div>
  );
}

function LayerInspector({
  slot,
  otherSlot,
  layers,
  layer,
  baseLayerId,
  sources,
  status,
  frameStats,
  freqGhz,
  loading,
  onChange,
  onMove,
  onRemove,
  onCopy,
  onMirror,
  onSetBase,
  onUnlink,
  onRefreshGlobalPeak,
  onOpenAlignment
}: {
  slot: PanelSlotId;
  otherSlot: PanelSlotId;
  layers: LayerState[];
  layer: LayerState;
  baseLayerId: string;
  sources: SourceMeta[];
  status: { unavailable: boolean; row: string; full: string };
  frameStats?: FrameStats;
  freqGhz: number[];
  loading: boolean;
  onChange: (slot: PanelSlotId, layerId: string, patch: Partial<LayerState>) => void;
  onMove: (slot: PanelSlotId, layerId: string, delta: number) => void;
  onRemove: (slot: PanelSlotId, layerId: string) => void;
  onCopy: (slot: PanelSlotId, layerId: string) => void;
  onMirror: (slot: PanelSlotId, layerId: string) => void;
  onSetBase: (slot: PanelSlotId, layerId: string) => void;
  onUnlink: (slot: PanelSlotId, layerId: string) => void;
  onRefreshGlobalPeak: (layer: LayerState) => void;
  onOpenAlignment: () => void;
}) {
  const index = layers.findIndex((candidate) => candidate.id === layer.id);
  const source = sources.find((candidate) => candidate.id === layer.sourceId);
  const role = sourceRole(source, {}) || layer.sourceRoleSnapshot;
  const sourceOptions = sources.filter((candidate) => sourceSupportsLayer(candidate, layer.kind, sourceRole(candidate, {})));
  if (source && !sourceOptions.some((candidate) => candidate.id === source.id)) sourceOptions.push(source);
  const patch = (value: Partial<LayerState>) => onChange(slot, layer.id, value);
  const patchDisplay = (key: keyof DisplayState, value: string | number) => patch({ display: { ...layer.display, [key]: value } });
  const isImage = layer.kind === "image";
  const isRadioImage = isImage && role === "radio";
  const hasDifference = layer.operation !== "none";
  const effectiveLevelMode: LayerState["contourLevelMode"] = layer.contourLevelReference === "global" ? layer.contourLevelMode : "percent";
  const [moreOpen, setMoreOpen] = useState(false);
  const [editingLabel, setEditingLabel] = useState(false);
  const processingSignature = [
    layer.operation,
    layer.reference,
    layer.temporal.mode,
    layer.temporal.sigmaShort,
    layer.temporal.sigmaLong,
    Number(layer.display.radialGamma ?? 0)
  ].join("|");
  const makeRangeBounds = (stats: FrameStats | undefined, vmin: number, vmax: number): { min: number; max: number } => {
    const p1 = stats?.p1 ?? Math.min(vmin, vmax);
    const p99 = stats?.p99 ?? Math.max(vmin, vmax);
    const min = Math.min(p1, vmin);
    const max = Math.max(p99, vmax);
    if (max > min) return { min, max };
    return { min: min - 1, max: max + 1 };
  };
  const currentVmin = numberValue(layer.display.vmin, 0);
  const currentVmax = numberValue(layer.display.vmax, 1);
  // Bounds are derived for the track only, and are STICKY: the stats
  // snapshot freezes per (layer, processing settings) so per-frame stats
  // during scrubbing/playback cannot jitter the track. Fit re-reads live
  // stats via fitLevels.
  const frozenStatsRef = useRef<{ key: string; stats: FrameStats | undefined }>({ key: "", stats: undefined });
  const statsFreezeKey = layer.id + "|" + processingSignature;
  if (frozenStatsRef.current.key !== statsFreezeKey || (!frozenStatsRef.current.stats && frameStats)) {
    frozenStatsRef.current = { key: statsFreezeKey, stats: frameStats };
  }
  const boundsStats = frozenStatsRef.current.stats;
  const rangeBounds = useMemo(
    () => makeRangeBounds(boundsStats, currentVmin, currentVmax),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [currentVmax, currentVmin, boundsStats, processingSignature]
  );

  useEffect(() => {
    setMoreOpen(false);
    setEditingLabel(false);
  }, [layer.id]);

  const fitLevels = () => {
    if (!frameStats) return;
    const nextMin = String(frameStats.p1);
    const nextMax = String(frameStats.p99);
    patchDisplay("vmin", nextMin);
    patchDisplay("vmax", nextMax);
  };

  return (
    <div className="layer-inspector-card">
      <div className="layer-inspector-header">
        <div className="layer-inspector-title">
          {editingLabel ? <>
            <span>Inspector: {slot === "left" ? "Left" : "Right"} ·</span>
            <InlineLayerLabelInput
              value={layer.label}
              ariaLabel="Layer label"
              className="layer-label-input"
              onCommit={(value) => {
                const nextLabel = value.trim();
                if (nextLabel) patch({ label: nextLabel, labelEdited: true });
                setEditingLabel(false);
              }}
              onCancel={() => setEditingLabel(false)}
            />
          </> : <>
            <strong>Inspector: {slot === "left" ? "Left" : "Right"} · {layer.label}</strong>
            <button className="button icon-button layer-label-edit" type="button" aria-label={`Rename ${layer.label}`} title="Rename layer" onClick={() => setEditingLabel(true)}>
              <Pencil size={13} aria-hidden="true" />
            </button>
            {layer.mirrorOf && <span className="layer-mirror-status">mirrored</span>}
            {layer.mirrorOf && <button className="button layer-unlink-button" type="button" onClick={() => onUnlink(slot, layer.id)} title="Break mirror link and make an independent copy">Unlink</button>}
          </>}
        </div>
        <small>{status.full}</small>
      </div>
      <div className="layer-inspector-grid">
        <InspectorRow label="Src" title="Source" control={<div className="source-control">
          <select value={layer.sourceId} onChange={(event) => patch({ sourceId: event.target.value })}>
            {sourceOptions.map((candidate) => <option key={candidate.id} value={candidate.id}>{candidate.label}</option>)}
          </select>
          {layer.kind === "contours" && <small>(all bands)</small>}
        </div>} />
        {isRadioImage && <InspectorRow label="Band" title="Band" control={<select value={layer.freqIndex} onChange={(event) => patch({ freqIndex: Math.max(0, Number(event.target.value)) })}>
          {(freqGhz.length ? freqGhz : [undefined]).map((frequency, band) => <option key={band} value={band}>Band {band} · {Number.isFinite(frequency) ? Number(frequency).toFixed(2) : "—"} GHz</option>)}
        </select>} />}
        <InspectorRow label="Op" title="Operation" control={<div className="operation-control">
          <div className="mode-row">
            <ModeButton active={layer.operation === "none"} title="Original" onClick={() => patch({ operation: "none" })}>Orig</ModeButton>
            <ModeButton active={layer.operation === "subtract"} title="Subtract" onClick={() => patch({ operation: "subtract" })}>Sub</ModeButton>
            <ModeButton active={layer.operation === "ratio"} title="Ratio" onClick={() => patch({ operation: "ratio" })}>Ratio</ModeButton>
          </div>
          <div className="operation-reference-controls">
            <select aria-label="Reference" title="Reference" value={layer.reference} onChange={(event) => patch({ reference: event.target.value as DifferenceReference })}>
              <option value="previous">Previous</option>
              <option value="base">Base</option>
              <option value="mean">Mean</option>
            </select>
            {layer.reference === "previous" && <CommitInput type="number" ariaLabel="Lag [s]" title="Lag [s]" value={layer.cadenceSeconds} onCommit={(value) => patch({ cadenceSeconds: value })} />}
          </div>
        </div>} />
        {isImage && <InspectorRow label={<Droplet size={15} aria-hidden="true" />} title="Layer opacity" control={<input type="range" min="0" max="1" step="0.05" value={layer.opacity} disabled={!layer.visible} onChange={(event) => patch({ opacity: Number(event.target.value) })} />} value={<CommitInput type="number" min={0} max={1} scrubStep={0.01} ariaLabel="Layer opacity" value={layer.opacity} disabled={!layer.visible} onCommit={(value) => patch({ opacity: Number(value) })} />} />}
        {isImage && <DualRangeRow
          bounds={rangeBounds}
          minValue={layer.display.vmin}
          maxValue={layer.display.vmax}
          disabled={!layer.visible}
          onMinCommit={(value) => patchDisplay("vmin", value)}
          onMaxCommit={(value) => patchDisplay("vmax", value)}
          onFit={fitLevels}
          fitDisabled={!frameStats}
        />}
        {isImage && <InspectorRow label={<Palette size={15} aria-hidden="true" />} title="Color map / scale" control={<div className="color-control">
          <select value={layer.display.cmap} onChange={(event) => patchDisplay("cmap", event.target.value)}>
            <optgroup label="Standard">
              {COLORMAP_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </optgroup>
            <optgroup label="Instrument">
              {INSTRUMENT_COLORMAP_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </optgroup>
          </select>
          <div className="mode-row scale-mode-row">
            <ModeButton active={layer.display.scale === "linear"} title="Linear" onClick={() => patchDisplay("scale", "linear")}>Linear</ModeButton>
            <ModeButton active={layer.display.scale === "log"} title="Logarithmic" onClick={() => patchDisplay("scale", "log")}>Log</ModeButton>
            <ModeButton active={layer.display.scale === "sqrt"} title="Sqrt — JHelioviewer-like stretch" onClick={() => patchDisplay("scale", "sqrt")}>Sqrt</ModeButton>
            <ModeButton active={layer.display.scale === "asinh"} title="Asinh stretch" onClick={() => patchDisplay("scale", "asinh")}>Asinh</ModeButton>
          </div>
        </div>} />}
        {isImage && <InspectorRow label={<Sun size={15} aria-hidden="true" />} title="Coronal enhancement — radial filter, γ 0–3" control={<input type="range" min="0" max="3" step="0.1" value={layer.display.radialGamma} onChange={(event) => patchDisplay("radialGamma", Number(event.target.value))} />} value={<CommitInput type="number" min={0} max={3} scrubStep={0.05} ariaLabel="Radial gamma" value={Number(layer.display.radialGamma ?? 0).toFixed(1)} onCommit={(value) => patchDisplay("radialGamma", Number(value))} />} />}
        {isImage && hasDifference && <InspectorRow label={<Activity size={15} aria-hidden="true" />} title="Temporal smoothing — smoothing can attenuate transient variations; larger σ increases attenuation" control={<div className="mode-row">
          <ModeButton active={layer.temporal.mode === "none"} title="None" onClick={() => patch({ temporal: { ...layer.temporal, mode: "none" } })}>None</ModeButton>
          <ModeButton active={layer.temporal.mode === "lowpass"} title="Low-pass" onClick={() => patch({ temporal: { ...layer.temporal, mode: "lowpass" } })}>LP</ModeButton>
          <ModeButton active={layer.temporal.mode === "bandpass"} title="Band-pass" onClick={() => patch({ temporal: { ...layer.temporal, mode: "bandpass" } })}>BP</ModeButton>
        </div>} />}
        {isImage && hasDifference && layer.temporal.mode !== "none" && <InspectorRow label="σ [s]" title="Temporal σ short [s]" control={<input type="range" min="1" max="120" step="1" value={layer.temporal.sigmaShort} onChange={(event) => patch({ temporal: { ...layer.temporal, sigmaShort: event.target.value } })} />} value={<CommitInput type="number" min={1} max={120} scrubStep={1} ariaLabel="Temporal sigma short in seconds" value={layer.temporal.sigmaShort} onCommit={(value) => patch({ temporal: { ...layer.temporal, sigmaShort: value } })} />} />}
        {layer.kind === "contours" && <>
          <InspectorRow label={<Palette size={15} aria-hidden="true" />} title="Contour map" control={<span className="contour-colormap-input">
            <i aria-hidden="true" style={{ backgroundImage: contourColormapGradient(layer.contourCmap) }} />
            <select aria-label="Contour colormap" value={layer.contourCmap} onChange={(event) => patch({ contourCmap: contourColormap(event.target.value) })}>
              {CONTOUR_COLORMAPS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
            </select>
          </span>} />
          <InspectorRow label="Peak" title="Peak reference" control={<div className="mode-row peak-reference-row">
            <ModeButton active={layer.contourLevelReference === "current"} title="Current" onClick={() => patch({ contourLevelReference: "current" })}>Cur</ModeButton>
            <span className="global-mode-control">
              <ModeButton active={layer.contourLevelReference === "global"} title="Global" onClick={() => patch({ contourLevelReference: "global" })}>Glob</ModeButton>
              <button className="button icon-button" type="button" aria-label="Refresh global peak" onClick={() => onRefreshGlobalPeak(layer)} disabled={loading || layer.contourLevelReference !== "global" || effectiveLevelMode !== "percent"} title="Compute true full-data global peak values"><RotateCcw size={13} aria-hidden="true" /></button>
            </span>
          </div>} />
          {layer.contourLevelReference === "global" && <InspectorRow label="Units" title="Level mode" className="indented-row" control={<div className="mode-row">
            <ModeButton active={effectiveLevelMode === "percent"} title="% peak" onClick={() => patch({ contourLevelMode: "percent" })}>%</ModeButton>
            <ModeButton active={effectiveLevelMode === "kelvin"} title="Kelvin" onClick={() => patch({ contourLevelMode: "kelvin" })}>K</ModeButton>
            <ModeButton active={effectiveLevelMode === "sfu"} title="Flux density per pixel [sfu]; frequency-independent threshold" onClick={() => patch({ contourLevelMode: "sfu" })}>sfu</ModeButton>
          </div>} />}
          {effectiveLevelMode === "percent" ? <InspectorRow label="Lvl" title="Contour level [% peak]" control={<input type="range" min="1" max="99" step="1" value={layer.contourLevelPercent} onChange={(event) => patch({ contourLevelPercent: event.target.value })} />} value={<CommitInput type="number" min={1} max={99} scrubStep={0.5} ariaLabel="Contour level percent" value={layer.contourLevelPercent} onCommit={(value) => patch({ contourLevelPercent: value })} />} />
            : effectiveLevelMode === "kelvin" ? <InspectorRow label="Lvl" title="Contour level Tb [K]" control={<CommitInput type="number" min={0} scrubStep={1.01} scrubMode="multiplicative" ariaLabel="Contour level Kelvin" value={layer.contourLevelKelvin} onCommit={(value) => patch({ contourLevelKelvin: value })} />} />
              : <InspectorRow label="Lvl" title="Contour level [sfu — flux density per pixel]" control={<CommitInput type="number" min={0} scrubStep={1.01} scrubMode="multiplicative" ariaLabel="Contour level sfu" value={layer.contourLevelSfu} onCommit={(value) => patch({ contourLevelSfu: value })} />} />}
          <InspectorRow label={<Droplet size={15} aria-hidden="true" />} title="Contour opacity" control={<input type="range" min="0" max="1" step="0.05" value={layer.contourOpacity} disabled={!layer.visible} onChange={(event) => patch({ contourOpacity: event.target.value })} />} value={<CommitInput type="number" min={0} max={1} scrubStep={0.01} ariaLabel="Contour opacity" value={layer.contourOpacity} disabled={!layer.visible} onCommit={(value) => patch({ contourOpacity: value })} />} />
          <InspectorRow label="Fill" title="Fill style" control={<div className="mode-row">
            <ModeButton active={!layer.contourFilled} title="Open" onClick={() => patch({ contourFilled: false })}>Open</ModeButton>
            <ModeButton active={layer.contourFilled} title="Filled" onClick={() => patch({ contourFilled: true })}>Fill</ModeButton>
          </div>} />
        </>}
      </div>
      <button className="button layer-more-button" type="button" aria-expanded={moreOpen} onClick={() => setMoreOpen((value) => !value)}>
        More adjustments {moreOpen ? "▴" : "▾"}
      </button>
      {moreOpen && <div className="layer-more-adjustments">
        <InspectorRow label="Sample" title="Sample policy" control={<select value={layer.samplingPolicy} onChange={(event) => patch({ samplingPolicy: event.target.value as LayerState["samplingPolicy"] })}>
          <option value="nearest">Nearest</option>
          <option value="previous">Previous</option>
          <option value="next">Next</option>
        </select>} />
        <InspectorRow label="Tol [s]" title="Tolerance [s]" control={<CommitInput type="number" min={0} value={String(layer.maxOffsetSeconds ?? "")} onCommit={(value) => patch({ maxOffsetSeconds: Number(value) })} />} />
        {layer.reference === "mean" && <>
          <InspectorRow label="Start" title="Mean start" control={<CommitInput value={mjdToUtc(layer.meanStartMjd ?? 0)} onCommit={(value) => patch({ meanStartMjd: parseTimeMjd(value, layer.meanStartMjd ?? 0) })} />} />
          <InspectorRow label="End" title="Mean end" control={<CommitInput value={mjdToUtc(layer.meanEndMjd ?? layer.meanStartMjd ?? 0)} onCommit={(value) => patch({ meanEndMjd: parseTimeMjd(value, layer.meanEndMjd ?? layer.meanStartMjd ?? 0) })} />} />
        </>}
        {isImage && hasDifference && layer.temporal.mode === "bandpass" && <InspectorRow label="σₗ [s]" title="Temporal σ long [s]" control={<input type="range" min="10" max="600" step="1" value={layer.temporal.sigmaLong} onChange={(event) => patch({ temporal: { ...layer.temporal, sigmaLong: event.target.value } })} />} value={<CommitInput type="number" min={10} max={600} scrubStep={1} ariaLabel="Temporal sigma long in seconds" value={layer.temporal.sigmaLong} onCommit={(value) => patch({ temporal: { ...layer.temporal, sigmaLong: value } })} />} />}
      </div>}
      <div className="layer-action-row">
        {role === "radio" && <button className="button icon-button" type="button" aria-label="Open radio channel inspector" title="Open channel inspector" onClick={onOpenAlignment}><Crosshair size={14} aria-hidden="true" /></button>}
        {layer.kind === "image" && <button className="button icon-button" type="button" aria-label="Set as base layer" title="Set as base layer" onClick={() => onSetBase(slot, layer.id)} disabled={baseLayerId === layer.id}><Anchor size={14} aria-hidden="true" /></button>}
        <button className="button icon-button" type="button" aria-label={`Copy layer to ${otherSlot} panel`} title="Copy — independent duplicate" onClick={() => onCopy(slot, layer.id)}>
          <span className="layer-copy-icon" aria-hidden="true"><Copy size={14} /><span>{slot === "left" ? <ChevronRight size={10} /> : <ChevronLeft size={10} />}</span></span>
        </button>
        <button className="button icon-button" type="button" aria-label="Mirror — settings stay synced" title="Mirror — settings stay synced" onClick={() => onMirror(slot, layer.id)}>
          <Link2 size={14} aria-hidden="true" />
        </button>
        <button className="button icon-button" type="button" aria-label="Move layer up" title="Move layer up" onClick={() => onMove(slot, layer.id, -1)} disabled={index <= 0}><ArrowUp size={14} aria-hidden="true" /></button>
        <button className="button icon-button" type="button" aria-label="Move layer down" title="Move layer down" onClick={() => onMove(slot, layer.id, 1)} disabled={index < 0 || index >= layers.length - 1}><ArrowDown size={14} aria-hidden="true" /></button>
        <button className="button icon-button danger" type="button" aria-label="Delete layer" title="Delete layer" onClick={() => onRemove(slot, layer.id)}><Trash2 size={14} aria-hidden="true" /></button>
      </div>
    </div>
  );
}

function InspectorRow({ label, title, control, value, className = "" }: { label: React.ReactNode; title?: string; control: React.ReactNode; value?: React.ReactNode; className?: string }) {
  return <div className={`layer-inspector-row ${className}`}>
    <span className="layer-inspector-label" title={title} aria-label={title}>{label}</span>
    <div className="layer-inspector-control">{control}</div>
    {/* Column tracks are explicitly sized by grid-template-columns (not
        content-driven), so omitting this wrapper when unused is visually
        identical to rendering it empty for every existing caller - it just
        stops SpectrogramDisplayCard's 2-column rows (no `value`) from
        claiming a phantom 3rd grid track. */}
    {value !== undefined && <div className="layer-inspector-value">{value}</div>}
  </div>;
}

function ModeButton({ active, title, onClick, children }: { active: boolean; title?: string; onClick: () => void; children: React.ReactNode }) {
  return <button className={`button mode-button ${active ? "active" : ""}`} type="button" title={title} aria-label={title} onClick={onClick}>{children}</button>;
}

function InlineLayerLabelInput({ value, ariaLabel, className, onCommit, onCancel }: {
  value: string;
  ariaLabel: string;
  className: string;
  onCommit: (value: string) => void;
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    setDraft(value);
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [value]);

  function commit() {
    onCommit(draft);
  }

  function cancel() {
    cancelledRef.current = true;
    onCancel();
  }

  return <input
    ref={inputRef}
    className={className}
    value={draft}
    aria-label={ariaLabel}
    onChange={(event) => setDraft(event.target.value)}
    onBlur={() => {
      if (cancelledRef.current) {
        cancelledRef.current = false;
      } else {
        commit();
      }
    }}
    onKeyDown={(event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commit();
        event.currentTarget.blur();
      } else if (event.key === "Escape") {
        event.preventDefault();
        cancel();
        event.currentTarget.blur();
      }
    }}
  />;
}

function DualRangeRow({
  bounds,
  minValue,
  maxValue,
  disabled,
  onMinCommit,
  onMaxCommit,
  onFit,
  fitDisabled = false
}: {
  bounds: { min: number; max: number };
  minValue: string;
  maxValue: string;
  disabled: boolean;
  onMinCommit: (value: string) => void;
  onMaxCommit: (value: string) => void;
  onFit?: () => void;
  fitDisabled?: boolean;
}) {
  // These are render-only knob positions. Bounds changes never call either
  // commit callback; only an input event below can update display.vmin/vmax.
  const min = clamp(numberValue(minValue, bounds.min), bounds.min, bounds.max);
  const max = clamp(numberValue(maxValue, bounds.max), bounds.min, bounds.max);
  const scrubStep = Math.max(bounds.max - bounds.min, Math.abs(max - min), Number.EPSILON) / 300;
  const position = (value: number) => `${((value - bounds.min) / Math.max(bounds.max - bounds.min, Number.EPSILON)) * 100}%`;
  return <div className="layer-inspector-row layer-range-row">
    <span className="layer-inspector-label" title="Display range (vmin / vmax)" aria-label="Display range (vmin / vmax)">Range</span>
    <div className={`layer-inspector-control dual-range-control ${onFit ? "" : "no-fit"}`}>
      <div className="dual-range-track" style={{ background: `linear-gradient(90deg, var(--border) ${position(min)}, var(--cyan) ${position(min)}, var(--cyan) ${position(max)}, var(--border) ${position(max)})` }}>
        <input className="dual-range-input dual-range-min" aria-label="Minimum level" type="range" min={bounds.min} max={bounds.max} step="any" value={min} disabled={disabled} onChange={(event) => onMinCommit(event.target.value)} />
        <input className="dual-range-input dual-range-max" aria-label="Maximum level" type="range" min={bounds.min} max={bounds.max} step="any" value={max} disabled={disabled} onChange={(event) => onMaxCommit(event.target.value)} />
      </div>
      {onFit && <button className="button icon-button dual-range-fit" type="button" aria-label="Fit display range to current frame" onClick={onFit} disabled={disabled || fitDisabled} title="Fit levels to the current frame"> <RotateCcw size={13} aria-hidden="true" /></button>}
    </div>
    <div className="layer-range-values">
      {/* Stacked top-to-bottom: top box is the upper limit (max), bottom box
          is the lower limit (min), matching vertical intuition. Handlers are
          unchanged - only the visual order of the two boxes is swapped. */}
      <CommitInput type="number" scrubStep={scrubStep} ariaLabel="Maximum level value" value={maxValue} disabled={disabled} onCommit={onMaxCommit} />
      <CommitInput type="number" scrubStep={scrubStep} ariaLabel="Minimum level value" value={minValue} disabled={disabled} onCommit={onMinCommit} />
    </div>
  </div>;
}


function CommitInput({
  value,
  onCommit,
  type = "text",
  disabled = false,
  min,
  max,
  ariaLabel,
  title,
  scrubStep,
  scrubMode = "linear"
}: {
  value: string | number;
  onCommit: (value: string) => void;
  type?: "text" | "number";
  disabled?: boolean;
  min?: number;
  max?: number;
  ariaLabel?: string;
  title?: string;
  scrubStep?: number;
  scrubMode?: "linear" | "multiplicative";
}) {
  const [draft, setDraft] = useState(String(value));
  const committedRef = useRef(String(value));
  const inputRef = useRef<HTMLInputElement | null>(null);
  const onCommitRef = useRef(onCommit);
  const dragCleanupRef = useRef<((cancel: boolean) => void) | null>(null);
  const suppressClickRef = useRef(false);

  onCommitRef.current = onCommit;

  useEffect(() => {
    const next = String(value);
    setDraft(next);
    committedRef.current = next;
  }, [value]);

  function commit() {
    if (draft === committedRef.current) return;
    committedRef.current = draft;
    onCommit(draft);
  }

  useEffect(() => {
    const input = inputRef.current;
    if (!input || type !== "number" || disabled || !Number.isFinite(scrubStep) || Number(scrubStep) <= 0) return undefined;
    const containingLabel = input.closest("label");
    const inspectorRow = input.closest(".layer-inspector-row");
    const inspectorLabel = inspectorRow?.querySelector<HTMLElement>(".layer-inspector-label") ?? null;
    const rowScrubInputs = inspectorRow?.querySelectorAll(".scrubbable-number-input").length ?? 0;
    const zones: HTMLElement[] = containingLabel instanceof HTMLElement
      ? [containingLabel]
      : [input, ...(inspectorLabel && rowScrubInputs === 1 ? [inspectorLabel] : [])];

    zones.forEach((zone) => zone.classList.add("scrubbable-number-zone"));

    const suppressDraggedClick = (event: MouseEvent) => {
      if (!suppressClickRef.current) return;
      suppressClickRef.current = false;
      event.preventDefault();
      event.stopPropagation();
    };

    const begin = (event: PointerEvent) => {
      if (event.button !== 0 || dragCleanupRef.current) return;
      if (event.target === input && document.activeElement === input) return;
      const startValue = Number(committedRef.current);
      if (!Number.isFinite(startValue)) return;
      const startText = committedRef.current;
      let moved = false;
      let lastText = startText;

      const finish = (cancel: boolean) => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", pointerUp);
        window.removeEventListener("pointercancel", pointerCancel);
        window.removeEventListener("keydown", keyDown, true);
        document.body.classList.remove("number-input-scrubbing");
        input.classList.remove("is-scrubbing");
        if (input.hasPointerCapture(event.pointerId)) input.releasePointerCapture(event.pointerId);
        dragCleanupRef.current = null;
        if (cancel && lastText !== startText) {
          setDraft(startText);
          committedRef.current = startText;
          onCommitRef.current(startText);
        } else if (!moved) {
          input.focus({ preventScroll: true });
        }
      };

      const move = (moveEvent: PointerEvent) => {
        if (moveEvent.pointerId !== event.pointerId) return;
        const dx = moveEvent.clientX - event.clientX;
        const dy = moveEvent.clientY - event.clientY;
        if (!moved && Math.hypot(dx, dy) < 3) return;
        moved = true;
        const modifier = (moveEvent.shiftKey ? 10 : 1) * (moveEvent.altKey ? 0.1 : 1);
        const raw = scrubMode === "multiplicative"
          ? startValue * Math.pow(Number(scrubStep), dx * modifier)
          : startValue + dx * Number(scrubStep) * modifier;
        const bounded = clamp(raw, min ?? Number.NEGATIVE_INFINITY, max ?? Number.POSITIVE_INFINITY);
        const nextText = String(Number(bounded.toPrecision(12)));
        if (nextText === lastText) return;
        lastText = nextText;
        setDraft(nextText);
        committedRef.current = nextText;
        onCommitRef.current(nextText);
        moveEvent.preventDefault();
      };

      const pointerUp = (upEvent: PointerEvent) => {
        if (upEvent.pointerId !== event.pointerId) return;
        suppressClickRef.current = moved;
        finish(false);
        if (moved) window.setTimeout(() => { suppressClickRef.current = false; }, 0);
      };
      const pointerCancel = (cancelEvent: PointerEvent) => {
        if (cancelEvent.pointerId === event.pointerId) finish(true);
      };
      const keyDown = (keyEvent: KeyboardEvent) => {
        if (keyEvent.key !== "Escape") return;
        keyEvent.preventDefault();
        keyEvent.stopPropagation();
        suppressClickRef.current = true;
        finish(true);
        window.setTimeout(() => { suppressClickRef.current = false; }, 0);
      };

      dragCleanupRef.current = finish;
      document.body.classList.add("number-input-scrubbing");
      input.classList.add("is-scrubbing");
      try { input.setPointerCapture(event.pointerId); } catch { /* Pointer remains covered by window listeners. */ }
      window.addEventListener("pointermove", move, { passive: false });
      window.addEventListener("pointerup", pointerUp);
      window.addEventListener("pointercancel", pointerCancel);
      window.addEventListener("keydown", keyDown, true);
      event.preventDefault();
    };

    zones.forEach((zone) => {
      zone.addEventListener("pointerdown", begin);
      zone.addEventListener("click", suppressDraggedClick, true);
    });
    return () => {
      zones.forEach((zone) => {
        zone.classList.remove("scrubbable-number-zone");
        zone.removeEventListener("pointerdown", begin);
        zone.removeEventListener("click", suppressDraggedClick, true);
      });
    };
  }, [disabled, max, min, scrubMode, scrubStep, type]);

  useEffect(() => () => dragCleanupRef.current?.(true), []);

  return (
    <input
      ref={inputRef}
      type={type}
      value={draft}
      min={min}
      max={max}
      aria-label={ariaLabel}
      title={title}
      className={type === "number" && scrubStep ? "scrubbable-number-input" : undefined}
      data-scrub-step={type === "number" && scrubStep ? scrubStep : undefined}
      data-scrub-mode={type === "number" && scrubStep ? scrubMode : undefined}
      disabled={disabled}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        commit();
        event.currentTarget.blur();
      }}
    />
  );
}

function ImagePanel({
  sessionId,
  onSessionNotFound,
  panel,
  title,
  renderLayers = [],
  scrubRenderLayers = [],
  scrubCursorMjd = 0,
  scrubbing = false,
  timestamp,
  imageRequest,
  overlayRequest,
  onImageReady = () => undefined,
  onOverlayReady = () => undefined,
  onLayerReady = () => undefined,
  colorbar,
  shape,
  lassoEnabled,
  targetDrawArmed = false,
  slitDrawArmed = false,
  slitSmoothPx = DEFAULT_SLIT_SMOOTH_PX,
  fanDrawStage = 0,
  fanRedrawTarget = null,
  channelLassoArmed = false,
  roi,
  roiWorld = [],
  targetWorld = [],
  slits = [],
  selectedSlitId = "",
  fan = null,
  fanBoundaryDraft = [],
  selectedFanMember = 0,
  tracks = [],
  selectedTrackId = "",
  currentFrameIndex = 0,
  seedMode = false,
  seedSuggestions = [],
  sources = [],
  currentMjd,
  timeMin,
  timeMax,
  solarView,
  fitView,
  pixelToWorldAffine,
  worldOffset,
  probePixel,
  spaceDown,
  alignmentActive = false,
  selectedChannels = [],
  channelOffsets = { dx: [], dy: [], masked: [] },
  radioFreqGhz = [],
  contourGlobalOffset = [0, 0],
  onImageHoverChange,
  onSolarViewChange,
  onProbe,
  onSeed = () => undefined,
  onSuggestionAccept = () => undefined,
  onTrackSelect = () => undefined,
  onAnchorMove = () => undefined,
  onSlitComplete = () => undefined,
  onFanBoundaryComplete = () => undefined,
  onFanRedrawComplete = () => undefined,
  onFanSelect = () => undefined,
  onFanPromote = () => undefined,
  onLassoComplete,
  onAlignmentDrag = () => undefined,
  onRequestCapChange = () => undefined,
  captureSurface,
  captureCanvasRef,
  drawNonce,
  onDrawComplete
}: {
  sessionId: string;
  onSessionNotFound?: () => Promise<unknown>;
  panel: PanelId;
  title: string;
  timestamp: string;
  imageRequest?: ScheduledFrameRequest;
  overlayRequest?: ScheduledFrameRequest;
  onImageReady?: (key: string) => void;
  onOverlayReady?: (key: string) => void;
  renderLayers?: RenderLayer[];
  scrubRenderLayers?: RenderLayer[];
  scrubCursorMjd?: number;
  scrubbing?: boolean;
  onLayerReady?: (layerId: string, key: string, status?: FrameResolution, stats?: FrameStats) => void;
  colorbar?: RadioColorbar;
  shape: [number, number];
  lassoEnabled: boolean;
  targetDrawArmed?: boolean;
  slitDrawArmed?: boolean;
  slitSmoothPx?: number;
  fanDrawStage?: 0 | 1 | 2;
  fanRedrawTarget?: "A" | "B" | null;
  channelLassoArmed?: boolean;
  roi: [number, number][];
  roiWorld?: [number, number][];
  targetWorld?: [number, number][];
  slits?: SlitDefinition[];
  selectedSlitId?: string;
  fan?: FanDefinition | null;
  fanBoundaryDraft?: [number, number][];
  selectedFanMember?: number;
  tracks?: SadTrack[];
  selectedTrackId?: string;
  currentFrameIndex?: number;
  seedMode?: boolean;
  seedSuggestions?: SeedSuggestion[];
  sources?: EovsaSource[];
  currentMjd: number;
  timeMin: number;
  timeMax: number;
  solarView: SolarView;
  fitView: SolarView;
  pixelToWorldAffine: Affine;
  worldOffset: [number, number];
  probePixel?: [number, number];
  spaceDown: boolean;
  alignmentActive?: boolean;
  selectedChannels?: number[];
  channelOffsets?: ChannelOffsets;
  radioFreqGhz?: number[];
  contourGlobalOffset?: [number, number];
  onImageHoverChange: (value: boolean) => void;
  onSolarViewChange: Dispatch<SetStateAction<SolarView>>;
  onProbe: (panel: PanelId, pixel: [number, number], layerId: string) => void;
  onSeed?: (pixel: [number, number]) => void;
  onSuggestionAccept?: (suggestion: SeedSuggestion) => void;
  onTrackSelect?: (trackId: string) => void;
  onAnchorMove?: (trackId: string, frameIndex: number, pixel: [number, number]) => void;
  onSlitComplete?: (curveArcsec: [number, number][], inputVertexCount: number, rawCurveArcsec: [number, number][], drawnPanel: PanelId) => void;
  onFanBoundaryComplete?: (curveArcsec: [number, number][], inputVertexCount: number, rawCurveArcsec: [number, number][], drawnPanel: PanelId) => void;
  onFanRedrawComplete?: (curveArcsec: [number, number][], inputVertexCount: number, rawCurveArcsec: [number, number][], drawnPanel: PanelId) => void;
  onFanSelect?: (memberIndex: number) => void;
  onFanPromote?: (memberIndex: number) => void;
  onLassoComplete: (panel: PanelId, points: [number, number][], sourceId?: string, sourceRole?: SourceRole, additive?: boolean, channelSelect?: boolean) => void;
  onAlignmentDrag?: (panel: PanelId, deltaWorld: [number, number]) => void;
  onRequestCapChange?: (cap: FrameRequestCap) => void;
  captureSurface: PanelSlotId;
  captureCanvasRef: Record<CaptureSurface, HTMLCanvasElement | null>;
  drawNonce: number;
  onDrawComplete: (mjd: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const layerDataRef = useRef<Map<string, FrameData>>(new Map());
  const layerRequestKeyRef = useRef<Map<string, string>>(new Map());
  const lastDisplayedDataRef = useRef<Map<string, FrameData>>(new Map());
  const layerUnavailableRef = useRef<Set<string>>(new Set());
  const transformRef = useRef<PanelTransform>({
    scale: 1,
    offsetX: 0,
    offsetY: 0,
    plotWidth: 1,
    plotHeight: 1,
    imageWidth: shape[1],
    imageHeight: shape[0],
    solarView,
    pixelToWorldAffine,
    worldOffset
  });
  const onImageReadyRef = useRef<(key: string) => void>(() => undefined);
  const onOverlayReadyRef = useRef<(key: string) => void>(() => undefined);
  const onLayerReadyRef = useRef<(layerId: string, key: string, status?: FrameResolution, stats?: FrameStats) => void>(() => undefined);
  const onRequestCapChangeRef = useRef<(cap: FrameRequestCap) => void>(() => undefined);
  const onDrawCompleteRef = useRef<(mjd: number) => void>(() => undefined);
  const fullResolutionTierRef = useRef(false);
  const requestCapKeyRef = useRef("");
  const solarViewRef = useRef(solarView);
  const fitViewRef = useRef(fitView);
  const drawLatestRef = useRef<() => void>(() => undefined);
  const rafRef = useRef(0);
  const drawFallbackTimerRef = useRef(0);
  const drawingRef = useRef(false);
  const panRef = useRef<PanState | null>(null);
  const alignmentDragRef = useRef<AlignmentDragState | null>(null);
  const markerDragRef = useRef<TrackMarkerDragState | null>(null);
  const trackingClickRef = useRef<TrackingClickState | null>(null);
  const lassoAdditiveRef = useRef(false);
  const lassoChannelSelectRef = useRef(false);
  const draftRef = useRef<[number, number][]>([]);
  const [draft, setDraft] = useState<[number, number][]>([]);
  const [alignmentVectorPreview, setAlignmentVectorPreview] = useState<[number, number]>([0, 0]);
  const [markerPreview, setMarkerPreview] = useState<{ trackId: string; frameIndex: number; pixel: [number, number] } | null>(null);
  const [hoveredTrackMarkerId, setHoveredTrackMarkerId] = useState("");
  const [hoveredFanMember, setHoveredFanMember] = useState<number | null>(null);
  const [markerDragging, setMarkerDragging] = useState(false);
  const [imageUnavailable, setImageUnavailable] = useState(false);
  const [overlayUnavailable, setOverlayUnavailable] = useState(false);
  const [scrubDisplayedFrame, setScrubDisplayedFrame] = useState<{ resolvedMjd: number; offsetSeconds: number } | null>(null);
  onImageReadyRef.current = typeof onImageReady === "function" ? onImageReady : () => undefined;
  onOverlayReadyRef.current = typeof onOverlayReady === "function" ? onOverlayReady : () => undefined;
  onLayerReadyRef.current = typeof onLayerReady === "function" ? onLayerReady : () => undefined;
  onRequestCapChangeRef.current = typeof onRequestCapChange === "function" ? onRequestCapChange : () => undefined;
  onDrawCompleteRef.current = typeof onDrawComplete === "function" ? onDrawComplete : () => undefined;
  solarViewRef.current = solarView;
  fitViewRef.current = fitView;

  const requestSignature = renderLayers.map((renderLayer) => `${renderLayer.layer.id}:${renderLayer.request.key}`).join("|");
  const renderSignature = JSON.stringify({
    requests: requestSignature,
    layers: renderLayers.map(({ layer }) => ({
      id: layer.id,
      opacity: layer.opacity,
      contourCmap: layer.contourCmap,
      contourOpacity: layer.contourOpacity,
      contourFilled: layer.contourFilled
    })),
    alignmentActive,
    selectedChannels,
    channelOffsets,
    contourGlobalOffset,
    selectedTrackId,
    currentFrameIndex,
    seedMode,
    seedSuggestions,
    slits,
    selectedSlitId,
    fan,
    fanBoundaryDraft,
    selectedFanMember
  });
  const scrubLayerSignature = scrubRenderLayers.map((renderLayer) => `${renderLayer.layer.id}:${renderLayer.request.key}`).join("|");

  useEffect(() => {
    layerDataRef.current.clear();
    layerRequestKeyRef.current.clear();
    lastDisplayedDataRef.current.clear();
    layerUnavailableRef.current.clear();
    setImageUnavailable(false);
    setOverlayUnavailable(false);
    setAlignmentVectorPreview([0, 0]);
    setMarkerPreview(null);
  }, [sessionId]);

  useLayoutEffect(() => {
    if (!scrubbing) {
      setScrubDisplayedFrame(null);
      return;
    }
    const primaryLayerId = (scrubRenderLayers.find((renderLayer) => renderLayer.layer.kind === "image") ?? scrubRenderLayers[0])?.layer.id;
    let primaryFrame: { resolvedMjd: number; offsetSeconds: number } | null = null;
    for (const renderLayer of scrubRenderLayers) {
      const cached = nearestCachedFrame(renderLayer.request);
      if (!cached) continue;
      layerDataRef.current.set(renderLayer.layer.id, cached.data);
      layerUnavailableRef.current.delete(renderLayer.layer.id);
      if (renderLayer.layer.id === primaryLayerId && Number.isFinite(cached.resolvedMjd)) {
        primaryFrame = {
          resolvedMjd: cached.resolvedMjd as number,
          offsetSeconds: ((cached.resolvedMjd as number) - scrubCursorMjd) * 86400
        };
      }
    }
    setScrubDisplayedFrame((current) => current?.resolvedMjd === primaryFrame?.resolvedMjd && current?.offsetSeconds === primaryFrame?.offsetSeconds ? current : primaryFrame);
    scheduleDraw();
  }, [scrubCursorMjd, scrubLayerSignature, scrubbing]);

  useEffect(() => {
    if (!alignmentActive) setAlignmentVectorPreview([0, 0]);
  }, [alignmentActive]);

  useEffect(() => {
    const controllers = new Map<string, AbortController>();
    const primaryLayerId = (renderLayers.find((renderLayer) => renderLayer.layer.kind === "image") ?? renderLayers[0])?.layer.id;
    setImageUnavailable(false);
    setOverlayUnavailable(false);
    for (const renderLayer of renderLayers) {
      if (layerRequestKeyRef.current.get(renderLayer.layer.id) !== renderLayer.request.key) {
        layerRequestKeyRef.current.delete(renderLayer.layer.id);
      }
      if (frameRequestState(renderLayer.request) === "unavailable") {
        layerUnavailableRef.current.add(renderLayer.layer.id);
      } else {
        layerUnavailableRef.current.delete(renderLayer.layer.id);
      }
      const controller = new AbortController();
      controllers.set(renderLayer.layer.id, controller);
      // The overlay debounce exists to coalesce contour-geometry requests
      // while a user is actively dragging the time slider (many identities
      // per second, only the settled one matters). Playback ticks are
      // already paced by the frame interval, not a rapid drag - applying
      // the same fixed delay there just eats into that interval, and once
      // fps outpaces the delay the request is aborted by the next tick
      // before it is ever sent. Gate the delay to scrubbing only.
      const start = renderLayer.layer.kind === "contours" && scrubbing
        ? waitForFrameDelay(OVERLAY_REQUEST_DELAY_MS, controller.signal)
        : Promise.resolve();
      void start.then(() => loadFrame(renderLayer.request, controller.signal, onSessionNotFound)).then((data) => {
        if (controller.signal.aborted) return;
        const unavailable = !data && frameRequestState(renderLayer.request) === "unavailable";
        if (data) layerDataRef.current.set(renderLayer.layer.id, data);
        layerRequestKeyRef.current.set(renderLayer.layer.id, renderLayer.request.key);
        if (unavailable) layerUnavailableRef.current.add(renderLayer.layer.id);
        else layerUnavailableRef.current.delete(renderLayer.layer.id);
        if (renderLayer.layer.kind === "image") {
          setImageUnavailable(unavailable);
          if (typeof onImageReadyRef.current === "function") onImageReadyRef.current(renderLayer.request.key);
        } else {
          setOverlayUnavailable(unavailable);
          if (typeof onOverlayReadyRef.current === "function") onOverlayReadyRef.current(renderLayer.request.key);
        }
        const resolution = frameResolution(renderLayer.request.key);
        if (scrubbing && data && renderLayer.layer.id === primaryLayerId && Number.isFinite(resolution?.resolvedMjd)) {
          const next = { resolvedMjd: resolution?.resolvedMjd as number, offsetSeconds: resolution?.offsetSeconds ?? 0 };
          setScrubDisplayedFrame((current) => current?.resolvedMjd === next.resolvedMjd && current?.offsetSeconds === next.offsetSeconds ? current : next);
        }
        if (typeof onLayerReadyRef.current === "function") onLayerReadyRef.current(renderLayer.layer.id, renderLayer.request.key, resolution, frameStatsForKey(renderLayer.request.bitmapKey));
        scheduleDraw();
      }).catch((error) => {
        if (controller.signal.aborted || error instanceof DOMException && error.name === "AbortError") return;
        console.error(`Layer ${renderLayer.layer.id} load failed.`, error);
        layerRequestKeyRef.current.set(renderLayer.layer.id, renderLayer.request.key);
        layerUnavailableRef.current.delete(renderLayer.layer.id);
        if (renderLayer.layer.kind === "image") setImageUnavailable(false);
        else setOverlayUnavailable(false);
        if (typeof onLayerReadyRef.current === "function") onLayerReadyRef.current(renderLayer.layer.id, renderLayer.request.key, {
          resolvedIndex: renderLayer.request.predictedResolution?.resolvedIndex ?? null,
          resolvedMjd: renderLayer.request.predictedResolution?.resolvedMjd ?? null,
          offsetSeconds: renderLayer.request.predictedResolution?.offsetSeconds ?? null,
          // A transport/decode failure is not a display-policy miss. Keep the
          // last frame on screen and leave the layer eligible for retry.
          unavailable: renderLayer.request.predictedResolution?.unavailable ?? false
        });
        scheduleDraw();
      });
    }
    return () => {
      for (const controller of controllers.values()) controller.abort();
    };
  }, [requestSignature]);

  drawLatestRef.current = draw;

  useEffect(() => {
    const canvas = canvasRef.current;
    const resize = () => {
      scheduleDraw();
      updateFrameRequestCap();
    };
    const observer = canvas && typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    if (canvas) observer?.observe(canvas);
    window.addEventListener("resize", resize);
    return () => {
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
      window.clearTimeout(drawFallbackTimerRef.current);
      rafRef.current = 0;
      drawFallbackTimerRef.current = 0;
      observer?.disconnect();
      window.removeEventListener("resize", resize);
    };
  }, []);

  useEffect(() => updateFrameRequestCap(), [solarView, fitView]);

  useEffect(() => scheduleDraw(), [alignmentVectorPreview, markerPreview, hoveredTrackMarkerId, roi, roiWorld, targetWorld, tracks, selectedTrackId, currentFrameIndex, seedSuggestions, sources, draft, probePixel, currentMjd, timeMin, timeMax, shape, solarView, pixelToWorldAffine, worldOffset, colorbar?.minGhz, colorbar?.maxGhz, colorbar?.cmap, renderSignature, imageUnavailable, overlayUnavailable, targetDrawArmed, drawNonce]);

  function scheduleDraw() {
    if (rafRef.current || drawFallbackTimerRef.current) return;
    const flush = () => {
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
      window.clearTimeout(drawFallbackTimerRef.current);
      rafRef.current = 0;
      drawFallbackTimerRef.current = 0;
      drawLatestRef.current();
    };
    rafRef.current = window.requestAnimationFrame(flush);
    drawFallbackTimerRef.current = window.setTimeout(flush, 50);
  }

  function updateFrameRequestCap() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const view = normalizeSolarView(solarViewRef.current);
    const fit = normalizeSolarView(fitViewRef.current);
    const zoom = Math.max(
      (fit.xMax - fit.xMin) / Math.max(1e-12, view.xMax - view.xMin),
      (fit.yMax - fit.yMin) / Math.max(1e-12, view.yMax - view.yMin)
    );
    if (!fullResolutionTierRef.current && zoom > FRAME_CAP_ZOOM_IN) {
      fullResolutionTierRef.current = true;
    } else if (fullResolutionTierRef.current && zoom <= FRAME_CAP_ZOOM_OUT) {
      fullResolutionTierRef.current = false;
    }
    if (fullResolutionTierRef.current) {
      if (requestCapKeyRef.current !== "full") {
        requestCapKeyRef.current = "full";
        onRequestCapChangeRef.current(null);
      }
      return;
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const cap = {
      maxWidth: Math.max(FRAME_CAP_QUANTUM, Math.ceil(Math.ceil(rect.width * dpr) / FRAME_CAP_QUANTUM) * FRAME_CAP_QUANTUM),
      maxHeight: Math.max(FRAME_CAP_QUANTUM, Math.ceil(Math.ceil(rect.height * dpr) / FRAME_CAP_QUANTUM) * FRAME_CAP_QUANTUM)
    };
    const key = `${cap.maxWidth}x${cap.maxHeight}`;
    if (requestCapKeyRef.current === key) return;
    requestCapKeyRef.current = key;
    onRequestCapChangeRef.current(cap);
  }

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, rect.width, rect.height);

    const interaction = renderLayers.find((renderLayer) => renderLayer.layer.kind === "image") ?? renderLayers[0];
    const interactionShape = interaction?.shape ?? shape;
    const interactionAffine = interaction?.pixelToWorldAffine ?? pixelToWorldAffine;
    const interactionOffset = interaction?.worldOffset ?? worldOffset;
    const transform = computePanelTransform(rect, interactionShape, solarView, interactionAffine, interactionOffset);
    transformRef.current = transform;
    const displayWidth = (transform.solarView.xMax - transform.solarView.xMin) * transform.scale;
    const displayHeight = (transform.solarView.yMax - transform.solarView.yMin) * transform.scale;

    ctx.fillStyle = "#111";
    ctx.fillRect(transform.offsetX, transform.offsetY, displayWidth, displayHeight);
    if (!renderLayers.length) {
      ctx.fillStyle = "#8a93a4";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("No layers — add one in the Layers panel", rect.width / 2, rect.height / 2);
      // Falls through instead of returning: an empty image-layer set (e.g. a
      // panel holding only a contours overlay with no base image, which the
      // radio-bound "contours" slit binding makes easy to end up with) must
      // not short-circuit the user-drawn overlays below - the slit curve,
      // ROI, and fan patch are independent annotations, not tied to there
      // being a displayable base frame. This previously unreachable branch
      // is exactly why the colorbar-for-empty-layers case a few lines down
      // already special-cased renderLayers.length === 0: this draw pass
      // used to return before ever reaching it.
    }
    for (const renderLayer of renderLayers) {
      const layerTransform = computePanelTransform(rect, renderLayer.shape, solarView, renderLayer.pixelToWorldAffine, renderLayer.worldOffset);
      const layerWidth = (layerTransform.solarView.xMax - layerTransform.solarView.xMin) * layerTransform.scale;
      const layerHeight = (layerTransform.solarView.yMax - layerTransform.solarView.yMin) * layerTransform.scale;
      // A policy-unavailable request must be blank, but a pending/failed
      // request keeps the last frame visible while its replacement loads.
      // Do not use layerUnavailableRef here: it is updated transiently while
      // scrub requests transition and used to suppress the hold-last frame.
      if (frameRequestState(renderLayer.request) === "unavailable") continue;
      let data = layerDataRef.current.get(renderLayer.layer.id)
        ?? lastDisplayedDataRef.current.get(renderLayer.layer.id)
        ?? nearestCachedFrame(renderLayer.request)?.data;
      if (renderLayer.layer.kind === "contours") {
        // layerDataRef only advances when this layer's own per-tick fetch
        // settles before the next advance aborts it, and it never expires -
        // so once starved it holds the first-ever-resolved geometry forever,
        // not just transiently. A different consumer (the imminent/general
        // playback prefetch) can land the *current* identity in the shared
        // cache before this layer's own attempt does. Prefer that exact
        // match so the draw swaps in each newly resolved geometry as it
        // lands; hold-last (above) only covers the gap before anything for
        // this identity exists.
        const exact = nearestCachedFrame(renderLayer.request);
        if (exact && exact.bitmapKey === renderLayer.request.bitmapKey) data = exact.data;
      }
      if (!data) continue;
      ctx.save();
      ctx.beginPath();
      ctx.rect(layerTransform.offsetX, layerTransform.offsetY, layerWidth, layerHeight);
      ctx.clip();
      if (renderLayer.layer.kind === "contours" && isContourGeometry(data)) {
        drawContourGeometry(
          ctx,
          data,
          layerTransform,
          renderLayer.layer,
          radioFreqGhz,
          channelOffsets,
          contourGlobalOffset,
          selectedChannels,
          alignmentActive,
          alignmentVectorPreview
        );
      } else if (data instanceof ImageBitmap) {
        ctx.globalAlpha = clamp(renderLayer.layer.opacity, 0, 1);
        drawImageWithWcs(ctx, data, layerTransform);
      }
      ctx.restore();
      lastDisplayedDataRef.current.set(renderLayer.layer.id, data);
      if (renderLayer.colorbar) drawRadioColorbar(ctx, rect.width, rect.height, renderLayer.colorbar);
    }
    ctx.strokeStyle = "#5a5a5a";
    ctx.lineWidth = 1;
    ctx.strokeRect(transform.offsetX, transform.offsetY, displayWidth, displayHeight);

    if (roiWorld.length) drawWorldPath(ctx, roiWorld, transform, "#ffcc66", true);
    if (targetWorld.length) drawWorldPath(ctx, targetWorld, transform, "#ffc857", true, true);
    else drawPath(ctx, roi, transform, "#ffcc66", true);
    if (fan) drawFanPatch(ctx, fan, transform, selectedFanMember);
    if (fanBoundaryDraft.length) drawWorldPath(ctx, fanBoundaryDraft, transform, "#ff5b5b", false);
    for (const slit of slits) {
      if (!slit.visible) continue;
      // A linked twin's own curveArcsec/widthArcsec fields are only a
      // fallback snapshot (see resolveSlitGeometry) - draw the resolved
      // geometry so the twin's path on the panel always tracks the original
      // live.
      const geometry = resolveSlitGeometry(slit, slits);
      // DOUBLE-SWATH SUPPRESSION: an origin and every twin linked (directly
      // or through a chain - slitOriginId) to it now draw the IDENTICAL
      // physical corridor (widthArcsec is inherited verbatim - see
      // resolveSlitGeometry), so painting each member's translucent fill/
      // dashed edges separately would double-paint the same patch of sky.
      // CHOICE: when more than one member of that group is visible, only
      // ONE of them draws the swath (fill + dashed edges) - preferring
      // whichever member is currently SELECTED (so selecting either the
      // origin or a twin still highlights the shared corridor in its own
      // color), falling back to the origin when neither is selected. Every
      // member still draws its own spine/arrowhead unconditionally, so
      // distinct twins remain identifiable by color on the panel.
      const originId = slit.linkedTo ? slitOriginId(slit, slits) : slit.id;
      const overlapGroup = slits.filter((candidate) => (
        candidate.visible && (candidate.id === originId || (candidate.linkedTo && slitOriginId(candidate, slits) === originId))
      ));
      const swathOwnerId = overlapGroup.some((candidate) => candidate.id === selectedSlitId) ? selectedSlitId : originId;
      const drawSwath = overlapGroup.length <= 1 || slit.id === swathOwnerId;
      drawSlitWorldPath(ctx, { ...slit, ...geometry }, transform, slit.id === selectedSlitId, drawSwath);
    }
    drawPath(ctx, draft, transform, fanDrawStage || fanRedrawTarget ? "#ff5b5b" : "#7cf0ff", false);
    if (probePixel) drawProbeMarker(ctx, probePixel, transform);
    const baseRenderLayer = renderLayers.find((renderLayer) => renderLayer.layer.kind === "image");
    if (baseRenderLayer) {
      drawTracks(ctx, tracks, transform, selectedTrackId, currentFrameIndex, markerPreview, hoveredTrackMarkerId);
      drawSeedSuggestions(ctx, seedSuggestions, transform);
    }
    if (baseRenderLayer?.layer.sourceRoleSnapshot === "radio") drawSources(ctx, sources, transform);
    if (!renderLayers.length && colorbar) drawRadioColorbar(ctx, rect.width, rect.height, colorbar);
    if (renderLayers.every((renderLayer) => layerRequestKeyRef.current.get(renderLayer.layer.id) === renderLayer.request.key)) {
      onDrawCompleteRef.current(currentMjd);
    }
  }

  function pointerToImage(event: React.PointerEvent<HTMLCanvasElement>): [number, number] | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const point: [number, number] = [event.clientX - rect.left, event.clientY - rect.top];
    const t = transformRef.current;
    const pixel = canvasToPixel(point, t);
    if (pixel[0] < 0 || pixel[1] < 0 || pixel[0] >= t.imageWidth || pixel[1] >= t.imageHeight) return null;
    return pixel;
  }

  function pointerToWorld(event: React.PointerEvent<HTMLCanvasElement>): [number, number] | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return canvasToWorld([event.clientX - rect.left, event.clientY - rect.top], transformRef.current);
  }

  function pointerCanvasPoint(event: React.PointerEvent<HTMLCanvasElement>): [number, number] | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  function hitTrackMarker(event: React.PointerEvent<HTMLCanvasElement>): { track: SadTrack; point: TrackPoint; ghost: boolean } | null {
    const canvasPoint = pointerCanvasPoint(event);
    if (!canvasPoint) return null;
    const ordered = [...tracks].sort((left, right) => Number(left.id === selectedTrackId) - Number(right.id === selectedTrackId));
    for (const track of ordered.reverse()) {
      if (!track.visible) continue;
      const exact = track.points.find((candidate) => candidate.frameIndex === currentFrameIndex);
      const point = exact ?? (track.id === selectedTrackId ? predictedTrackPoint(track, currentFrameIndex) : null);
      if (!point) continue;
      const marker = pixelToCanvas([point.x, point.y], transformRef.current);
      if (Math.hypot(marker[0] - canvasPoint[0], marker[1] - canvasPoint[1]) <= TRACK_MARKER_HIT_RADIUS) return { track, point, ghost: !exact };
    }
    return null;
  }

  function hitSeedSuggestion(event: React.PointerEvent<HTMLCanvasElement>): SeedSuggestion | null {
    const canvasPoint = pointerCanvasPoint(event);
    if (!canvasPoint) return null;
    return seedSuggestions.find((suggestion) => {
      const marker = pixelToCanvas([suggestion.x, suggestion.y], transformRef.current);
      return Math.hypot(marker[0] - canvasPoint[0], marker[1] - canvasPoint[1]) <= TRACK_MARKER_HIT_RADIUS;
    }) ?? null;
  }

  function hitFanCurveAt(canvasPoint: [number, number]): number | null {
    if (!fan) return null;
    let best: { index: number; distance: number } | null = null;
    const curves = fanFamilyCurves(fan);
    for (let memberIndex = 0; memberIndex < curves.length; memberIndex += 1) {
      const curve = curves[memberIndex];
      for (let index = 1; index < curve.length; index += 1) {
        const left = worldToCanvas(curve[index - 1], transformRef.current);
        const right = worldToCanvas(curve[index], transformRef.current);
        const dx = right[0] - left[0];
        const dy = right[1] - left[1];
        const span = dx * dx + dy * dy;
        const fraction = span > 0 ? clamp(((canvasPoint[0] - left[0]) * dx + (canvasPoint[1] - left[1]) * dy) / span, 0, 1) : 0;
        const distance = Math.hypot(canvasPoint[0] - left[0] - fraction * dx, canvasPoint[1] - left[1] - fraction * dy);
        if (distance <= 14 && (!best || distance < best.distance)) best = { index: memberIndex, distance };
      }
    }
    return best ? best.index : null;
  }

  function hitFanCurve(event: React.PointerEvent<HTMLCanvasElement>): number | null {
    const point = pointerCanvasPoint(event);
    return point ? hitFanCurveAt(point) : null;
  }

  function updateMarkerHover(event: React.PointerEvent<HTMLCanvasElement>) {
    if (seedMode || spaceDown || lassoEnabled || channelLassoArmed || targetDrawArmed || slitDrawArmed || fanDrawStage || fanRedrawTarget || markerDragRef.current || trackingClickRef.current || panRef.current || drawingRef.current) {
      setHoveredTrackMarkerId("");
      setHoveredFanMember(null);
      return;
    }
    const hit = hitTrackMarker(event);
    setHoveredTrackMarkerId(hit?.track.id === selectedTrackId ? hit.track.id : "");
    setHoveredFanMember(hit ? null : hitFanCurve(event));
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const point: [number, number] = [event.clientX - rect.left, event.clientY - rect.top];
    const factor = Math.exp(Math.max(-1.25, Math.min(1.25, event.deltaY * 0.0015)));
    onSolarViewChange((current) => {
      const view = expandSolarViewToAspect(current, rect.width / Math.max(1, rect.height));
      const anchor: [number, number] = [
        view.xMin + clamp(point[0] / Math.max(1, rect.width), 0, 1) * (view.xMax - view.xMin),
        view.yMax - clamp(point[1] / Math.max(1, rect.height), 0, 1) * (view.yMax - view.yMin)
      ];
      return zoomSolarView(view, anchor, factor, fitView);
    });
  }

  function zoomAroundCenter(factor: number) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    onSolarViewChange((current) => {
      const view = expandSolarViewToAspect(current, rect.width / Math.max(1, rect.height));
      return zoomSolarView(view, [
        (view.xMin + view.xMax) / 2,
        (view.yMin + view.yMax) / 2
      ], factor, fitView);
    });
  }

  function pointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (event.button !== 0) return;
    if (alignmentActive && event.shiftKey && !channelLassoArmed) {
      const world = pointerToWorld(event);
      if (!world) return;
      alignmentDragRef.current = { pointerId: event.pointerId, startWorld: world, totalWorld: [0, 0] };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (!spaceDown) {
      if (!seedMode && !lassoEnabled && !channelLassoArmed && !targetDrawArmed && !slitDrawArmed && !fanDrawStage && !fanRedrawTarget) {
        const fanMember = hitFanCurve(event);
        if (fanMember !== null) {
          onFanSelect(fanMember);
          event.preventDefault();
          return;
        }
      }
      const hit = hitTrackMarker(event);
      if (hit) {
        const pixel = pointerToImage(event);
        if (!pixel) return;
        if (hit.track.id === selectedTrackId) {
          markerDragRef.current = {
            pointerId: event.pointerId,
            trackId: hit.track.id,
            frameIndex: hit.point.frameIndex,
            startPixel: pixel,
            currentPixel: [hit.point.x, hit.point.y],
            moved: false
          };
          setMarkerDragging(true);
        } else {
          trackingClickRef.current = { pointerId: event.pointerId, startPixel: pixel, trackId: hit.track.id, moved: false };
        }
        event.currentTarget.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
      }
      const suggestion = hitSeedSuggestion(event);
      if (suggestion) {
        const pixel = pointerToImage(event);
        if (!pixel) return;
        trackingClickRef.current = { pointerId: event.pointerId, startPixel: pixel, suggestion, moved: false };
        event.currentTarget.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
      }
      if (seedMode) {
        const pixel = pointerToImage(event);
        if (!pixel) return;
        trackingClickRef.current = { pointerId: event.pointerId, startPixel: pixel, moved: false };
        event.currentTarget.setPointerCapture(event.pointerId);
        event.preventDefault();
        return;
      }
    }
    const lassoArmed = lassoEnabled || channelLassoArmed || targetDrawArmed || slitDrawArmed || Boolean(fanDrawStage) || Boolean(fanRedrawTarget);
    if (spaceDown || !lassoArmed) {
      panRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        solarView: transformRef.current.solarView,
        startPixel: !spaceDown && !lassoArmed ? pointerToImage(event) : null,
        moved: false
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (!lassoArmed) return;
    const pixel = pointerToImage(event);
    if (!pixel) return;
    drawingRef.current = true;
    lassoAdditiveRef.current = event.shiftKey;
    lassoChannelSelectRef.current = channelLassoArmed;
    draftRef.current = [pixel];
    setDraft([pixel]);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function pointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    updateMarkerHover(event);
    if (alignmentDragRef.current?.pointerId === event.pointerId) {
      const world = pointerToWorld(event);
      if (!world) return;
      alignmentDragRef.current.totalWorld = [
        world[0] - alignmentDragRef.current.startWorld[0],
        world[1] - alignmentDragRef.current.startWorld[1]
      ];
      setAlignmentVectorPreview(alignmentDragRef.current.totalWorld);
      event.preventDefault();
      return;
    }
    if (markerDragRef.current?.pointerId === event.pointerId) {
      const pixel = pointerToImage(event);
      if (!pixel) return;
      const drag = markerDragRef.current;
      drag.currentPixel = pixel;
      drag.moved = drag.moved || Math.hypot(pixel[0] - drag.startPixel[0], pixel[1] - drag.startPixel[1]) > 0.5;
      setHoveredTrackMarkerId(drag.trackId);
      setMarkerPreview({ trackId: drag.trackId, frameIndex: drag.frameIndex, pixel });
      event.preventDefault();
      return;
    }
    if (trackingClickRef.current?.pointerId === event.pointerId) {
      const pixel = pointerToImage(event);
      if (pixel && Math.hypot(pixel[0] - trackingClickRef.current.startPixel[0], pixel[1] - trackingClickRef.current.startPixel[1]) > 3) {
        trackingClickRef.current.moved = true;
      }
      event.preventDefault();
      return;
    }
    if (panRef.current?.pointerId === event.pointerId) {
      const t = transformRef.current;
      const dx = event.clientX - panRef.current.startX;
      const dy = event.clientY - panRef.current.startY;
      if (!panRef.current.moved && Math.hypot(dx, dy) <= 4) return;
      panRef.current.moved = true;
      onSolarViewChange(panSolarView(panRef.current.solarView, dx, dy, t.scale));
      event.preventDefault();
      return;
    }
    if (!drawingRef.current) return;
    const pixel = pointerToImage(event);
    if (!pixel) return;
    const updated = appendPoint(draftRef.current, pixel);
    draftRef.current = updated;
    setDraft(updated);
  }

  function pointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    if (alignmentDragRef.current?.pointerId === event.pointerId) {
      const drag = alignmentDragRef.current;
      alignmentDragRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      setAlignmentVectorPreview([0, 0]);
      onAlignmentDrag(panel, drag.totalWorld);
      event.preventDefault();
      return;
    }
    if (markerDragRef.current?.pointerId === event.pointerId) {
      const drag = markerDragRef.current;
      markerDragRef.current = null;
      setMarkerDragging(false);
      setMarkerPreview(null);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      if (drag.moved) onAnchorMove(drag.trackId, drag.frameIndex, drag.currentPixel);
      else onTrackSelect(drag.trackId);
      event.preventDefault();
      return;
    }
    if (trackingClickRef.current?.pointerId === event.pointerId) {
      const click = trackingClickRef.current;
      trackingClickRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      if (!click.moved) {
        if (click.trackId) onTrackSelect(click.trackId);
        else if (click.suggestion) onSuggestionAccept(click.suggestion);
        else onSeed(click.startPixel);
      }
      event.preventDefault();
      return;
    }
    if (panRef.current?.pointerId === event.pointerId) {
      const pan = panRef.current;
      panRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      if (!pan.moved && pan.startPixel) {
        const baseLayer = renderLayers.find((renderLayer) => renderLayer.layer.kind === "image");
        if (baseLayer) onProbe(panel, pan.startPixel, baseLayer.layer.id);
      }
      event.preventDefault();
      return;
    }
    if (!drawingRef.current) return;
    drawingRef.current = false;
    const points = draftRef.current;
    const additive = lassoAdditiveRef.current;
    const channelSelect = lassoChannelSelectRef.current;
    draftRef.current = [];
    lassoAdditiveRef.current = false;
    lassoChannelSelectRef.current = false;
    setDraft([]);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (fanDrawStage && points.length >= 2) {
      onFanBoundaryComplete(
        smoothResampleSlit(points, transformRef.current, fanDrawStage === 1, slitSmoothPx),
        points.length,
        trimAndProjectStroke(points, transformRef.current),
        panel
      );
    } else if (fanRedrawTarget && points.length >= 2) {
      onFanRedrawComplete(
        smoothResampleSlit(points, transformRef.current, fanRedrawTarget === "A", slitSmoothPx),
        points.length,
        trimAndProjectStroke(points, transformRef.current),
        panel
      );
    } else if (slitDrawArmed && points.length >= 2) {
      onSlitComplete(
        smoothResampleSlit(points, transformRef.current, true, slitSmoothPx),
        points.length,
        trimAndProjectStroke(points, transformRef.current),
        panel
      );
    } else if (points.length >= 3) {
      const baseLayer = renderLayers.find((renderLayer) => renderLayer.layer.kind === "image") ?? renderLayers[0];
      onLassoComplete(panel, points, baseLayer?.layer.sourceId, baseLayer?.layer.sourceRoleSnapshot, additive, channelSelect);
    }
  }

  function pointerCancel(event: React.PointerEvent<HTMLCanvasElement>) {
    alignmentDragRef.current = null;
    setAlignmentVectorPreview([0, 0]);
    panRef.current = null;
    markerDragRef.current = null;
    trackingClickRef.current = null;
    setMarkerDragging(false);
    setHoveredTrackMarkerId("");
    setHoveredFanMember(null);
    setMarkerPreview(null);
    drawingRef.current = false;
    lassoAdditiveRef.current = false;
    lassoChannelSelectRef.current = false;
    draftRef.current = [];
    setDraft([]);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const primaryRequest = renderLayers.find((renderLayer) => renderLayer.layer.kind === "image")?.request ?? renderLayers[0]?.request;
  const resolvedInfo = primaryRequest ? frameResolution(primaryRequest.key) ?? primaryRequest.predictedResolution : undefined;
  const resolvedTimestamp = scrubbing && scrubDisplayedFrame
    ? formatResolvedTimestamp(scrubDisplayedFrame.resolvedMjd, scrubDisplayedFrame.offsetSeconds, timestamp)
    : formatResolvedTimestamp(resolvedInfo?.resolvedMjd, resolvedInfo?.offsetSeconds, timestamp);

  function doubleClickFan(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!fan || fanDrawStage || slitDrawArmed || fanRedrawTarget) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const member = hitFanCurveAt([event.clientX - rect.left, event.clientY - rect.top]);
    if (member === null) return;
    event.preventDefault();
    onFanSelect(member);
    onFanPromote(member);
  }

  return (
    <section className="image-panel">
      <header>
        <span>{title}</span>
        <small>{resolvedTimestamp}</small>
        <div className="image-view-controls" role="group" aria-label={`${panel} view controls`}>
          <button
            type="button"
            aria-label="Zoom out"
            title="Zoom out"
            onClick={() => zoomAroundCenter(1.35)}
            disabled={!renderLayers.length}
          >−</button>
          <button
            type="button"
            className="fit-button"
            aria-label="Fit image"
            title="Fit image"
            onClick={() => onSolarViewChange(fitView)}
            disabled={!renderLayers.length}
          >Fit</button>
          <button
            type="button"
            aria-label="Zoom in"
            title="Zoom in"
            onClick={() => zoomAroundCenter(1 / 1.35)}
            disabled={!renderLayers.length}
          >+</button>
        </div>
      </header>
      <canvas
        ref={(canvas) => {
          canvasRef.current = canvas;
          captureCanvasRef[captureSurface] = canvas;
        }}
        className={seedMode ? "seed-mode" : alignmentActive ? "alignment-mode" : spaceDown || !(lassoEnabled || channelLassoArmed || targetDrawArmed || slitDrawArmed || fanDrawStage || fanRedrawTarget) ? "pan-mode" : "lasso-mode"}
        style={{ cursor: markerDragging ? "grabbing" : hoveredTrackMarkerId ? "grab" : hoveredFanMember !== null ? "pointer" : undefined }}
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={pointerCancel}
        onDoubleClick={doubleClickFan}
        onPointerEnter={(event) => { onImageHoverChange(true); updateMarkerHover(event); }}
        onPointerLeave={() => { onImageHoverChange(false); setHoveredTrackMarkerId(""); setHoveredFanMember(null); }}
        onWheel={handleWheel}
      />
    </section>
  );
}

type FloatingCardPosition = { x: number; y: number };

function clampFloatingCardPosition(
  candidate: FloatingCardPosition,
  positionAtMeasurement: FloatingCardPosition,
  headerRect: DOMRect,
  safeTop: number
): FloatingCardPosition {
  const minimumVisible = 40;
  const baseLeft = headerRect.left - positionAtMeasurement.x;
  const baseRight = headerRect.right - positionAtMeasurement.x;
  const baseTop = headerRect.top - positionAtMeasurement.y;
  const visibleWidth = Math.min(minimumVisible, headerRect.width);
  const visibleHeight = Math.min(minimumVisible, headerRect.height);
  const maximumHeaderTop = Math.max(safeTop, window.innerHeight - visibleHeight);
  return {
    x: clamp(candidate.x, visibleWidth - baseRight, window.innerWidth - visibleWidth - baseLeft),
    y: clamp(candidate.y, safeTop - baseTop, maximumHeaderTop - baseTop)
  };
}

type FloatingCardHookOptions = {
  minWidth?: number;
  minHeight?: number;
  size?: FloatingCardSize;
  onSizeChange?: (size: FloatingCardSize) => void;
};

function useDraggableFloatingCard(options: FloatingCardHookOptions = {}) {
  const cardRef = useRef<HTMLElement | null>(null);
  const headerRef = useRef<HTMLElement | null>(null);
  const positionRef = useRef<FloatingCardPosition>({ x: 0, y: 0 });
  const resizeRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startWidth: number;
    startHeight: number;
  } | null>(null);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startY: number;
    startPosition: FloatingCardPosition;
    headerRect: DOMRect;
  } | null>(null);
  const [position, setPosition] = useState<FloatingCardPosition>({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [size, setSize] = useState<FloatingCardSize | undefined>(options.size);
  const minWidth = options.minWidth ?? 320;
  const minHeight = options.minHeight ?? 240;

  useEffect(() => {
    setSize(options.size);
  }, [options.size?.height, options.size?.width]);

  const safeTop = () => Math.max(0, cardRef.current?.offsetParent?.getBoundingClientRect().top ?? 0);
  const updatePosition = (next: FloatingCardPosition) => {
    positionRef.current = next;
    setPosition(next);
  };

  useEffect(() => {
    const reclamp = () => {
      const headerRect = headerRef.current?.getBoundingClientRect();
      if (!headerRect) return;
      const current = positionRef.current;
      const next = clampFloatingCardPosition(current, current, headerRect, safeTop());
      if (next.x !== current.x || next.y !== current.y) updatePosition(next);
    };
    reclamp();
    window.addEventListener("resize", reclamp);
    return () => window.removeEventListener("resize", reclamp);
  }, []);

  function onPointerDown(event: React.PointerEvent<HTMLElement>) {
    if (event.button !== 0 || (event.target as Element).closest("button, input, select, textarea, a, [role='button']")) return;
    const headerRect = headerRef.current?.getBoundingClientRect();
    if (!headerRect) return;
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startPosition: positionRef.current,
      headerRect
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    setDragging(true);
  }

  function onPointerMove(event: React.PointerEvent<HTMLElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    updatePosition(clampFloatingCardPosition({
      x: drag.startPosition.x + event.clientX - drag.startX,
      y: drag.startPosition.y + event.clientY - drag.startY
    }, drag.startPosition, drag.headerRect, safeTop()));
  }

  function stopDragging(event: React.PointerEvent<HTMLElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setDragging(false);
  }

  function onResizePointerDown(event: React.PointerEvent<HTMLElement>) {
    if (event.button !== 0) return;
    const rect = cardRef.current?.getBoundingClientRect();
    if (!rect) return;
    resizeRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      startWidth: rect.width,
      startHeight: rect.height
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
  }

  function onResizePointerMove(event: React.PointerEvent<HTMLElement>) {
    const resize = resizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    const rect = cardRef.current?.getBoundingClientRect();
    const maxWidth = Math.max(minWidth, window.innerWidth - (rect?.left ?? 0) - 20);
    const maxHeight = Math.max(minHeight, window.innerHeight - (rect?.top ?? 0) - 20);
    const next = {
      width: clamp(resize.startWidth + event.clientX - resize.startX, minWidth, maxWidth),
      height: clamp(resize.startHeight + event.clientY - resize.startY, minHeight, maxHeight)
    };
    setSize(next);
    options.onSizeChange?.(next);
    event.preventDefault();
    event.stopPropagation();
  }

  function stopResizing(event: React.PointerEvent<HTMLElement>) {
    const resize = resizeRef.current;
    if (!resize || resize.pointerId !== event.pointerId) return;
    resizeRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    event.stopPropagation();
  }

  return {
    cardRef,
    headerRef,
    dragging,
    style: {
      transform: `translate3d(${position.x}px, ${position.y}px, 0)`,
      minWidth,
      minHeight,
      ...(size ? { width: size.width, height: size.height } : {})
    },
    headerHandlers: {
      onPointerDown,
      onPointerMove,
      onPointerUp: stopDragging,
      onPointerCancel: stopDragging
    },
    resizeHandleHandlers: {
      onPointerDown: onResizePointerDown,
      onPointerMove: onResizePointerMove,
      onPointerUp: stopResizing,
      onPointerCancel: stopResizing
    }
  };
}

function filesystemBreadcrumbs(path: string): { label: string; path: string }[] {
  const parts = path.split("/").filter(Boolean);
  let current = "";
  return [
    { label: "/", path: "/" },
    ...parts.map((part) => {
      current += `/${part}`;
      return { label: part, path: current };
    })
  ];
}

function fileSizeLabel(bytes: number): string {
  if (!(bytes >= 0) || !Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function FilePickerCard({
  initialPath,
  onDirectoryChange,
  onSelect,
  onClose,
  cardSize,
  onCardSizeChange
}: {
  initialPath: string;
  onDirectoryChange: (path: string) => void;
  onSelect: (path: string) => void;
  onClose: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const floatingCard = useDraggableFloatingCard({ minWidth: 420, minHeight: 300, size: cardSize, onSizeChange: onCardSizeChange });
  const listRef = useRef<HTMLDivElement | null>(null);
  const [listing, setListing] = useState<FileSystemListing | null>(null);
  const [highlightedPath, setHighlightedPath] = useState("");
  const [showHidden, setShowHidden] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadDirectory(path: string) {
    setLoading(true);
    setError("");
    const params = new URLSearchParams();
    if (path) params.set("path", path);
    if (showHidden) params.set("showHidden", "1");
    try {
      const next = await apiJson<FileSystemListing>(`/api/fs/list${params.size ? `?${params.toString()}` : ""}`);
      setListing(next);
      setHighlightedPath(next.entries[0]?.path ?? "");
      onDirectoryChange(next.path);
      window.requestAnimationFrame(() => listRef.current?.focus());
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.detail : reason instanceof Error ? reason.message : "Could not list directory");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadDirectory(listing?.path ?? initialPath);
  }, [showHidden]);

  const highlightedEntry = listing?.entries.find((entry) => entry.path === highlightedPath);

  function activate(entry: FileSystemEntry | undefined) {
    if (!entry) return;
    if (entry.isDir) void loadDirectory(entry.path);
    else onSelect(entry.path);
  }

  function handleListKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const entries = listing?.entries ?? [];
    const currentIndex = Math.max(0, entries.findIndex((entry) => entry.path === highlightedPath));
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!entries.length) return;
      const delta = event.key === "ArrowDown" ? 1 : -1;
      const nextIndex = clamp(currentIndex + delta, 0, entries.length - 1);
      setHighlightedPath(entries[nextIndex].path);
      document.getElementById(`file-picker-entry-${nextIndex}`)?.scrollIntoView({ block: "nearest" });
    } else if (event.key === "Enter") {
      event.preventDefault();
      activate(entries[currentIndex]);
    } else if (event.key === "Backspace") {
      event.preventDefault();
      if (listing?.parent) void loadDirectory(listing.parent);
    }
  }

  return (
    <section ref={floatingCard.cardRef} className="radio-alignment-card file-picker-card" role="dialog" aria-label="Browse server files" style={floatingCard.style}>
      <header ref={floatingCard.headerRef} className={`radio-alignment-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
        <div><strong>Browse server files</strong><small>FITS, HDF5, JSON, and NPZ sources</small></div>
        <button className="probe-close" type="button" aria-label="Close file picker" onClick={onClose}>×</button>
      </header>
      <nav className="file-picker-breadcrumbs" aria-label="Current directory">
        {(listing ? filesystemBreadcrumbs(listing.path) : []).map((crumb, index) => (
          <span key={crumb.path}>
            {index > 0 && <span aria-hidden="true">›</span>}
            <button type="button" title={crumb.path} onClick={() => void loadDirectory(crumb.path)}>{crumb.label}</button>
          </span>
        ))}
      </nav>
      <div className="file-picker-toolbar">
        <button className="button" type="button" onClick={() => listing?.parent && void loadDirectory(listing.parent)} disabled={!listing || listing.parent === listing.path} title="Parent directory (Backspace)"><ChevronLeft size={14} /> Up</button>
        <span title={listing?.path}>{listing?.path ?? (loading ? "Loading…" : "")}</span>
        <label><input type="checkbox" checked={showHidden} onChange={(event) => setShowHidden(event.target.checked)} /> Hidden</label>
      </div>
      <div
        ref={listRef}
        className="file-picker-list"
        role="listbox"
        tabIndex={0}
        aria-label="Directory entries"
        aria-activedescendant={highlightedPath ? `file-picker-entry-${Math.max(0, listing?.entries.findIndex((entry) => entry.path === highlightedPath) ?? 0)}` : undefined}
        onKeyDown={handleListKeyDown}
      >
        {loading && !listing && <div className="file-picker-status">Loading directory…</div>}
        {error && <div className="file-picker-status error">{error}</div>}
        {!loading && !error && listing?.entries.length === 0 && <div className="file-picker-status">No supported files or directories.</div>}
        {listing?.entries.map((entry, index) => (
          <button
            id={`file-picker-entry-${index}`}
            className={`file-picker-entry ${highlightedPath === entry.path ? "selected" : ""}`}
            type="button"
            role="option"
            aria-selected={highlightedPath === entry.path}
            tabIndex={-1}
            key={entry.path}
            title={entry.path}
            onClick={() => entry.isDir ? void loadDirectory(entry.path) : setHighlightedPath(entry.path)}
            onDoubleClick={() => { if (!entry.isDir) onSelect(entry.path); }}
          >
            <FolderOpen size={14} aria-hidden="true" className={entry.isDir ? "" : "file-picker-file-icon"} />
            <span className="file-picker-entry-name">{entry.name}</span>
            <span>{entry.isDir ? "Folder" : fileSizeLabel(entry.size)}</span>
            <time dateTime={new Date(entry.mtime * 1000).toISOString()}>{new Date(entry.mtime * 1000).toLocaleString()}</time>
          </button>
        ))}
      </div>
      <footer className="file-picker-actions">
        <button className="button" type="button" onClick={() => listing && onSelect(listing.path)} disabled={!listing}>Select this folder</button>
        <span />
        <button className="button" type="button" onClick={onClose}>Cancel</button>
        <button className="button primary" type="button" onClick={() => activate(highlightedEntry)} disabled={!highlightedEntry || highlightedEntry.isDir}>Select</button>
      </footer>
      <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
    </section>
  );
}

function CorrelationCard({
  series,
  xRange,
  currentMjd,
  fullRange,
  pinTicks,
  fullHeightTicks,
  peakDecel,
  selectedTrackId,
  onFullRange,
  onPinTicks,
  onFullHeightTicks,
  onPeakDecel,
  onSelectTrack,
  onHoverTrack,
  onTimeSelect,
  onClose,
  cardSize,
  onCardSizeChange
}: {
  series: CorrelationSeries[];
  xRange: [number, number];
  currentMjd: number;
  fullRange: boolean;
  pinTicks: boolean;
  fullHeightTicks: boolean;
  peakDecel: boolean;
  selectedTrackId: string;
  onFullRange: (value: boolean) => void;
  onPinTicks: (value: boolean) => void;
  onFullHeightTicks: (value: boolean) => void;
  onPeakDecel: (value: boolean) => void;
  onSelectTrack: (trackId: string) => void;
  onHoverTrack: (trackId: string) => void;
  onTimeSelect: (mjd: number) => void;
  onClose: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const floatingCard = useDraggableFloatingCard({ minWidth: 320, minHeight: 300, size: cardSize, onSizeChange: onCardSizeChange });
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragTimeRef = useRef<number | null>(null);
  const geometryRef = useRef<{ id: string; points: [number, number][] }[]>([]);
  const [hoverTrackId, setHoverTrackId] = useState("");
  const [plotSize, setPlotSize] = useState({ width: 520, height: 220 });
  const minX = xRange[0];
  const maxX = Math.max(minX + 1e-9, xRange[1]);
  const maxDistance = Math.max(1, ...series.flatMap((item) => item.points.map((point) => point.distance)).filter(Number.isFinite));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      setPlotSize({ width: Math.max(260, rect.width), height: Math.max(160, rect.height) });
    };
    resize();
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    observer?.observe(canvas);
    return () => observer?.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(plotSize.width * dpr);
    canvas.height = Math.floor(plotSize.height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, plotSize.width, plotSize.height);
    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, plotSize.width, plotSize.height);
    const margin = { left: 48, right: 42, top: 12, bottom: 30 };
    const width = Math.max(1, plotSize.width - margin.left - margin.right);
    const height = Math.max(1, plotSize.height - margin.top - margin.bottom);
    const xAt = (mjd: number) => margin.left + clamp((mjd - minX) / (maxX - minX), 0, 1) * width;
    const yAt = (distance: number) => margin.top + (1 - clamp(distance / maxDistance, 0, 1)) * height;
    ctx.strokeStyle = "#56606b";
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, width, height);
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.fillStyle = "#aeb6c1";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(`${maxDistance.toFixed(1)} arcsec`, margin.left - 5, margin.top + 4);
    ctx.fillText("0 arcsec", margin.left - 5, margin.top + height);
    ctx.textAlign = "left";
    ctx.fillText(`${(maxDistance * 0.725).toFixed(1)} Mm`, margin.left + width + 5, margin.top + 4);
    ctx.fillText("0 Mm", margin.left + width + 5, margin.top + height);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(mjdToUtc(minX).slice(11, 19), margin.left, margin.top + height + 7);
    ctx.fillText(mjdToUtc(maxX).slice(11, 19), margin.left + width, margin.top + height + 7);
    geometryRef.current = [];
    for (const item of series) {
      const path: [number, number][] = [];
      ctx.strokeStyle = item.track.color;
      ctx.globalAlpha = item.track.id === (hoverTrackId || selectedTrackId) ? 1 : 0.78;
      ctx.lineWidth = item.track.id === (hoverTrackId || selectedTrackId) ? 2.6 : 1.5;
      ctx.beginPath();
      item.points.forEach((point, index) => {
        const xy: [number, number] = [xAt(point.mjd), yAt(point.distance)];
        path.push(xy);
        if (index === 0) ctx.moveTo(xy[0], xy[1]); else ctx.lineTo(xy[0], xy[1]);
      });
      ctx.stroke();
      geometryRef.current.push({ id: item.track.id, points: path });
      if (item.arrival && item.arrival.arrivalMjd >= minX && item.arrival.arrivalMjd <= maxX) {
        const x = xAt(item.arrival.arrivalMjd);
        ctx.setLineDash([3, 3]);
        ctx.strokeStyle = item.track.color;
        ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + height); ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.globalAlpha = 1;
    if (currentMjd >= minX && currentMjd <= maxX) {
      const x = xAt(currentMjd);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + height); ctx.stroke();
    }
  }, [currentMjd, hoverTrackId, maxDistance, maxX, minX, plotSize, selectedTrackId, series]);

  function timeFromEvent(event: React.PointerEvent<HTMLCanvasElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = clamp((event.clientX - rect.left - 48) / Math.max(1, rect.width - 90), 0, 1);
    return minX + fraction * (maxX - minX);
  }

  function hoverFromEvent(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let nearest = "";
    let distance = 10;
    geometryRef.current.forEach((path) => path.points.forEach((point) => {
      const value = Math.hypot(point[0] - x, point[1] - y);
      if (value < distance) { distance = value; nearest = path.id; }
    }));
    setHoverTrackId(nearest);
    onHoverTrack(nearest);
  }

  function exportPng() {
    canvasRef.current?.toBlob((blob) => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a"); link.href = url; link.download = "sad_correlation.png"; link.click(); URL.revokeObjectURL(url);
    }, "image/png");
  }

  return <section ref={floatingCard.cardRef} className="radio-alignment-card correlation-card" role="dialog" aria-label="SAD distance correlation" tabIndex={-1} style={floatingCard.style}>
    <header ref={floatingCard.headerRef} className={`radio-alignment-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
      <div><strong>Distance to loop-top</strong><small>{series.length} track{series.length === 1 ? "" : "s"}</small></div>
      <button className="probe-close" type="button" aria-label="Close correlation card" onPointerDown={(event) => { event.stopPropagation(); onClose(); }} onPointerUp={(event) => event.stopPropagation()} onMouseDown={(event) => { event.stopPropagation(); onClose(); }} onClick={(event) => event.stopPropagation()}>×</button>
    </header>
    <div className="correlation-controls">
      <label><input type="checkbox" checked={fullRange} onChange={(event) => onFullRange(event.target.checked)} /> Full master range</label>
      <label><input type="checkbox" checked={pinTicks} onChange={(event) => onPinTicks(event.target.checked)} /> Pin ticks</label>
      <label><input type="checkbox" checked={fullHeightTicks} onChange={(event) => onFullHeightTicks(event.target.checked)} /> Full-height lines</label>
      <label><input type="checkbox" checked={peakDecel} onChange={(event) => onPeakDecel(event.target.checked)} /> Peak deceleration</label>
      <button className="button" type="button" onClick={exportPng}><Download size={13} /> PNG</button>
      <span className="correlation-window">{mjdToUtc(minX).slice(11, 19)} → {mjdToUtc(maxX).slice(11, 19)}</span>
    </div>
    <canvas ref={canvasRef} className="correlation-plot" onPointerMove={(event) => { hoverFromEvent(event); if (dragTimeRef.current === event.pointerId) onTimeSelect(timeFromEvent(event)); }} onPointerLeave={() => { setHoverTrackId(""); onHoverTrack(""); }} onPointerDown={(event) => { dragTimeRef.current = event.pointerId; event.currentTarget.setPointerCapture(event.pointerId); onTimeSelect(timeFromEvent(event)); }} onPointerUp={(event) => { if (dragTimeRef.current === event.pointerId) dragTimeRef.current = null; if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); onTimeSelect(timeFromEvent(event)); }} onPointerCancel={() => { dragTimeRef.current = null; }} />
    <div className="correlation-legend">{series.map((item) => { const first = item.points[0]?.distance; const last = item.points.at(-1)?.distance; return <span key={item.track.id} style={{ color: item.track.color }} onMouseEnter={() => { setHoverTrackId(item.track.id); onHoverTrack(item.track.id); }} onMouseLeave={() => { setHoverTrackId(""); onHoverTrack(""); }}><i style={{ background: item.track.color }} />{item.track.label} · n={item.points.length}{typeof first === "number" && typeof last === "number" && Number.isFinite(first) && Number.isFinite(last) ? ` · d ${first.toFixed(2)}→${last.toFixed(2)}″` : ""}{item.arrival ? ` · ${mjdToUtc(item.arrival.arrivalMjd).slice(11, 19)}` : ""}</span>; })}</div>
    <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
  </section>;
}

type TimeDistanceContourSegment = [[number, number], [number, number]];

// Shared marching-squares core: contourSegmentsAtThreshold takes an already
// resolved threshold value (used by the layer-driven radio contour family),
// while timeDistanceContourSegments keeps the legacy percent-of-own-peak
// entry point (used for non-radio/base-map slits only - see TimeDistanceLane).
function contourSegmentsAtThreshold(map: SlitMapResult, threshold: number): TimeDistanceContourSegment[] {
  const segments: TimeDistanceContourSegment[] = [];
  for (let distance = 0; distance < map.npix - 1; distance += 1) {
    for (let time = 0; time < map.ntime - 1; time += 1) {
      const corners = [
        { point: [time, distance] as [number, number], value: map.intensity[distance]?.[time] },
        { point: [time + 1, distance] as [number, number], value: map.intensity[distance]?.[time + 1] },
        { point: [time + 1, distance + 1] as [number, number], value: map.intensity[distance + 1]?.[time + 1] },
        { point: [time, distance + 1] as [number, number], value: map.intensity[distance + 1]?.[time] }
      ];
      if (corners.some((corner) => typeof corner.value !== "number" || !Number.isFinite(corner.value))) continue;
      const crossings: [number, number][] = [];
      for (let edge = 0; edge < 4; edge += 1) {
        const left = corners[edge];
        const right = corners[(edge + 1) % 4];
        const leftValue = left.value as number;
        const rightValue = right.value as number;
        if ((leftValue >= threshold) === (rightValue >= threshold)) continue;
        const fraction = clamp((threshold - leftValue) / Math.max(1e-12, rightValue - leftValue), 0, 1);
        crossings.push([
          left.point[0] + (right.point[0] - left.point[0]) * fraction,
          left.point[1] + (right.point[1] - left.point[1]) * fraction
        ]);
      }
      if (crossings.length === 2) segments.push([crossings[0], crossings[1]]);
      else if (crossings.length === 4) {
        segments.push([crossings[0], crossings[1]], [crossings[2], crossings[3]]);
      }
    }
  }
  return segments;
}

function timeDistanceContourSegments(map: SlitMapResult, levelPercent: number): TimeDistanceContourSegment[] {
  const threshold = map.dataMax * clamp(levelPercent, 1, 99) / 100;
  return contourSegmentsAtThreshold(map, threshold);
}

// TWIN OVERLAY COMPOSITING: the "raster" the lane draws (at most one - "no
// raster-over-raster") - the composite base member's own extraction result,
// together with its effective display (LAYER-LINKED RASTER SCALE: the bound
// image layer's live vmin/vmax/cmap/scale when linked, else the slit's own
// stored display - see effectiveSlitDisplay in the parent).
type TimeDistanceRasterSource = { slit: SlitDefinition; result: SlitResult; display: SlitDisplayState };
// One contour-bound family drawn on top of the raster (or alone, on the dark
// background, when there is no raster) - a composite may stack several of
// these (one per contour-bound twin), each contoured at ITS OWN bound
// layer's thresholds/colors and shifted by ITS OWN shiftSeconds.
type TimeDistanceContourSource = {
  slit: SlitDefinition;
  result: SlitResult;
  contourLayer: LayerState | null;
  contourDiffParams: ReturnType<typeof differenceParams> | null;
};

function TimeDistanceLane({
  slit,
  raster,
  contours,
  missingTwinHint,
  xRange,
  timeMjd,
  channelGutterVisible,
  currentMjd,
  radioFrequencyCount,
  radioFreqGhz,
  channelPalette,
  sessionId,
  radioSourceId,
  canvasRef,
  onTimeSelect,
  onTimeScrubStart,
  onTimeScrub,
  onTimeScrubEnd,
  onTimeRangeChange
}: {
  slit: SlitDefinition;
  raster: TimeDistanceRasterSource | null;
  contours: TimeDistanceContourSource[];
  missingTwinHint: string | null;
  xRange: [number, number];
  timeMjd: number[];
  channelGutterVisible: boolean;
  currentMjd: number;
  radioFrequencyCount: number;
  radioFreqGhz: number[];
  channelPalette: ContourColormap;
  sessionId: string;
  radioSourceId: string;
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>;
  onTimeSelect: (mjd: number) => void;
  onTimeScrubStart: () => void;
  onTimeScrub: (mjd: number) => void;
  onTimeScrubEnd: () => void;
  onTimeRangeChange: (range: TimeRangeState) => void;
}) {
  const localCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<number | null>(null);
  const panRef = useRef<{ pointerId: number; startX: number; startMin: number; startMax: number; moved: boolean } | null>(null);
  const laneCursorRef = useRef<"" | "ew-resize">("");
  const cursorGrabWidth = 10;
  // Keyed by "<contour member slit id>:<freqIndex>" rather than bare
  // freqIndex - a composite can stack several contour-bound twins whose
  // channel numbering overlaps, and each needs its own cached segments.
  const contourCacheRef = useRef(new Map<string, { intensity: SlitMapResult["intensity"]; threshold: number; segments: TimeDistanceContourSegment[] }>());
  // Global-reference thresholds (sfu conversion, global-percent peak table)
  // are resolved server-side (see radio_contour_threshold) and fetched here,
  // debounced, so scrubbing the bound layer's level control re-renders the
  // lane from the already-extracted maps without a re-extraction. Current-
  // reference percent and kelvin thresholds are cheap enough to resolve
  // inline in the draw effect below and never need this fetch. Keyed by
  // contour member slit id first, then freqIndex - composited twins may be
  // bound to different contour layers/levels even when their own frequency
  // indices overlap.
  const [globalThresholds, setGlobalThresholds] = useState<Record<string, Record<number, number | null>>>({});

  // Per contour-bound family member: its own channel maps, its own base
  // (for legacy same-slit exclusion), and the contour channels it
  // contributes - identical math to the pre-composite single-slit
  // computation, just resolved once per participant instead of once for the
  // (always singular, pre-composite) selected slit.
  const contourGroups = contours.map((entry) => {
    const maps = slitResultMaps(entry.result);
    const isContourBound = Boolean(entry.contourLayer);
    const soleImageMap = maps.length === 1 && maps[0].freqIndex === null ? maps[0] : null;
    const ownBaseMap = soleImageMap ?? (entry.slit.baseFreqIndex === null ? null : maps.find((map) => map.freqIndex === entry.slit.baseFreqIndex) ?? null);
    const geometryMap = ownBaseMap ?? maps[0];
    // A contour-bound radio slit contours every selected channel regardless
    // of which one (if any) is also shown as the base heat map - "EXACTLY
    // the same as the overlaid visualization" contours all active channels.
    // The legacy path (non-contour-bound multi-map slits) keeps the old
    // explicit include/exclude toggle list against a single base map.
    const contourMaps = geometryMap
      ? (isContourBound
          ? maps.filter((map) => map.freqIndex !== null)
          : maps.filter((map) => map.freqIndex !== geometryMap.freqIndex && map.freqIndex !== null && entry.slit.contourFreqIndices.includes(map.freqIndex)))
      : [];
    return { ...entry, isContourBound, maps: contourMaps, geometryMap };
  });
  const globalContourGroupsKey = contourGroups.map((group) => {
    const layer = group.contourLayer;
    if (!group.isContourBound || !layer || layer.contourLevelReference !== "global") return "";
    return [
      group.slit.id,
      group.maps.map((map) => map.freqIndex).join(","),
      layer.contourLevelMode, layer.contourLevelPercent, layer.contourLevelKelvin, layer.contourLevelSfu,
      group.contourDiffParams ? JSON.stringify(group.contourDiffParams) : ""
    ].join(":");
  }).join(";");

  useEffect(() => {
    if (!sessionId) return undefined;
    const handle = window.setTimeout(() => {
      for (const group of contourGroups) {
        if (!group.isContourBound || !group.contourLayer || group.contourLayer.contourLevelReference !== "global" || !group.contourDiffParams) continue;
        const channels = group.maps.map((map) => map.freqIndex).filter((value): value is number => value !== null);
        if (!channels.length) continue;
        const layer = group.contourLayer;
        const diffParams = group.contourDiffParams;
        const memberId = group.slit.id;
        const params = new URLSearchParams({
          freqIndices: channels.join(","),
          levelMode: layer.contourLevelMode,
          levelReference: layer.contourLevelReference,
          levelPercent: String(numberValue(layer.contourLevelPercent, 50)),
          levelKelvin: String(numberValue(layer.contourLevelKelvin, 1e6)),
          levelSfu: String(numberValue(layer.contourLevelSfu, 1)),
          diffSeconds: String(diffParams.diffSeconds),
          differenceMode: String(diffParams.differenceMode),
          useRunningDiff: String(Boolean(diffParams.useRunningDiff)),
          differenceOperation: String(diffParams.differenceOperation),
          differenceReference: String(diffParams.differenceReference),
          meanStartMjd: String(diffParams.meanStartMjd),
          meanEndMjd: String(diffParams.meanEndMjd)
        });
        apiJson<{ thresholds: Record<string, number | null> }>(`/api/sessions/${sessionId}/sources/${radioSourceId}/radio-contour-thresholds?${params.toString()}`)
          .then((response) => {
            setGlobalThresholds((current) => {
              const next = { ...current, [memberId]: { ...current[memberId] } };
              for (const [key, value] of Object.entries(response.thresholds)) next[memberId][Number(key)] = value;
              return next;
            });
          })
          .catch(() => undefined);
      }
    }, 200);
    return () => window.clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, radioSourceId, globalContourGroupsKey]);

  // Resolves one channel's contour trigger level from its own group's bound
  // layer's live settings, matching the panel overlay's threshold exactly
  // for kelvin/sfu/global-percent (server-resolved, see the effect above and
  // radio_contour_threshold on the backend). The "current" percent
  // reference has no single native frame to fall back to for a whole
  // time-distance map, so it uses the channel's own map peak - see
  // radio_contour_threshold's docstring for the same reasoning applied
  // server-side to the extraction-time snapshot in map.contourThreshold.
  function resolveContourThreshold(group: (typeof contourGroups)[number], map: SlitMapResult): number | null {
    if (!group.contourLayer || map.freqIndex === null) return null;
    const reference = group.contourLayer.contourLevelReference === "global" ? "global" : "current";
    if (reference !== "global") {
      if (!Number.isFinite(map.dataMax)) return null;
      const percent = clamp(numberValue(group.contourLayer.contourLevelPercent, 50), 1, 99);
      return map.dataMax * percent / 100;
    }
    const fetched = globalThresholds[group.slit.id]?.[map.freqIndex];
    if (typeof fetched === "number") return Number.isFinite(fetched) ? fetched : null;
    if (fetched === null) return null;
    return typeof map.contourThreshold === "number" && Number.isFinite(map.contourThreshold) ? map.contourThreshold : null;
  }

  const rasterRef = useRef<{
    intensity: SlitMapResult["intensity"];
    firstRow: SlitMapResult["intensity"][number] | undefined;
    lastRow: SlitMapResult["intensity"][number] | undefined;
    displayKey: string;
    canvas: HTMLCanvasElement;
  } | null>(null);
  const [size, setSize] = useState({ width: 900, height: DEFAULT_SLIT_LANE_HEIGHT });
  const minX = xRange[0];
  const maxX = Math.max(minX + 1e-9, xRange[1]);

  useEffect(() => {
    const canvas = localCanvasRef.current;
    if (!canvas) return undefined;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      setSize({ width: Math.max(320, rect.width), height: Math.max(1, rect.height) });
    };
    resize();
    const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(resize) : null;
    observer?.observe(canvas);
    return () => observer?.disconnect();
  }, []);

  const rasterMaps = raster ? slitResultMaps(raster.result) : [];
  // Same "sole no-frequency-axis map is unambiguously the base" / explicit
  // None handling as the pre-composite single-slit computation (see
  // contourGroups above) - just resolved for the composite's raster
  // participant instead.
  const rasterSoleImageMap = rasterMaps.length === 1 && rasterMaps[0].freqIndex === null ? rasterMaps[0] : null;
  const baseMap = raster
    ? (rasterSoleImageMap ?? (raster.slit.baseFreqIndex === null ? null : rasterMaps.find((map) => map.freqIndex === raster.slit.baseFreqIndex) ?? null))
    : null;
  // The lane always has geometry (npix/ntime/distance/time axes) whenever
  // ANY participant has a result, even with no base heat map (None) and no
  // raster participant at all (a lone contour-bound twin, no image sibling).
  const overallGeometryMap = baseMap ?? rasterMaps[0] ?? contourGroups.find((group) => group.geometryMap)?.geometryMap ?? null;

  useEffect(() => {
    const canvas = localCanvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(size.width * dpr);
    canvas.height = Math.round(size.height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#080b10";
    ctx.fillRect(0, 0, size.width, size.height);
    const horizontal = spectrogramTimePlotRect(canvas.getBoundingClientRect(), channelGutterVisible);
    const margin = { left: horizontal.x, top: 20, bottom: 8 };
    const plotW = horizontal.width;
    const plotH = Math.max(1, size.height - margin.top - margin.bottom);
    // Multi-channel radio slits (and composites with no image-bound member)
    // default to no base map (contours on a dark background); the raster
    // block below is simply skipped in that case - the earlier fillRect
    // already painted the dark background.
    if (raster && baseMap) {
      // LAYER-LINKED RASTER SCALE: raster.display already carries either the
      // bound image layer's live settings or the slit's own stored display
      // (see effectiveSlitDisplay in the parent) - this block itself does
      // not care which.
      const display = raster.display;
      const displayKey = `${raster.slit.id}|${display.vmin}|${display.vmax}|${display.cmap}|${display.scale}`;
      const previousRaster = rasterRef.current;
      const sameIntensity = previousRaster?.intensity === baseMap.intensity;
      const sameRaster = sameIntensity && previousRaster?.displayKey === displayKey
        && previousRaster.firstRow === baseMap.intensity[0]
        && previousRaster.lastRow === baseMap.intensity.at(-1);
      const reversedRaster = previousRaster?.displayKey === displayKey
        && sameIntensity
        && previousRaster.intensity.length === baseMap.intensity.length
        && previousRaster.firstRow === baseMap.intensity.at(-1)
        && previousRaster.lastRow === baseMap.intensity[0];
      let image = previousRaster?.canvas ?? document.createElement("canvas");
      if (!sameRaster) {
        const nextImage = document.createElement("canvas");
        nextImage.width = Math.max(1, baseMap.ntime);
        nextImage.height = Math.max(1, baseMap.npix);
        const imageContext = nextImage.getContext("2d");
        if (imageContext && reversedRaster && previousRaster) {
          imageContext.translate(0, nextImage.height);
          imageContext.scale(1, -1);
          imageContext.drawImage(previousRaster.canvas, 0, 0);
        } else if (imageContext) {
          const pixels = imageContext.createImageData(nextImage.width, nextImage.height);
          const lower = Math.min(display.vmin, display.vmax);
          const upper = Math.max(lower + Number.EPSILON, Math.max(display.vmin, display.vmax));
          for (let distanceIndex = 0; distanceIndex < baseMap.npix; distanceIndex += 1) {
            const row = baseMap.intensity[distanceIndex] ?? [];
            for (let timeIndex = 0; timeIndex < baseMap.ntime; timeIndex += 1) {
              const value = row[timeIndex];
              let fraction = typeof value === "number" && Number.isFinite(value) ? clamp((value - lower) / (upper - lower), 0, 1) : 0;
              if (display.scale === "sqrt") fraction = Math.sqrt(fraction);
              else if (display.scale === "log") fraction = Math.log1p(999 * fraction) / Math.log(1000);
              const color = sampleColormap(display.cmap, Math.round(fraction * 1023), 1024);
              const offset = ((baseMap.npix - 1 - distanceIndex) * baseMap.ntime + timeIndex) * 4;
              pixels.data[offset] = Number.parseInt(color.slice(1, 3), 16);
              pixels.data[offset + 1] = Number.parseInt(color.slice(3, 5), 16);
              pixels.data[offset + 2] = Number.parseInt(color.slice(5, 7), 16);
              pixels.data[offset + 3] = typeof value === "number" && Number.isFinite(value) ? 255 : 0;
            }
          }
          imageContext.putImageData(pixels, 0, 0);
        }
        image = nextImage;
        rasterRef.current = {
          intensity: baseMap.intensity,
          firstRow: baseMap.intensity[0],
          lastRow: baseMap.intensity.at(-1),
          displayKey,
          canvas: image
        };
      }
      if (image.getContext("2d")) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(margin.left, margin.top, plotW, plotH);
        ctx.clip();
        ctx.imageSmoothingEnabled = true;
        // The raster's OWN shift - independent of any contour-bound twin's
        // shift below and of the selected member's shift (the white cursor
        // line) - see "each member's own shiftSeconds applied to its own
        // layer of the composite" (useful for lag work between members).
        const shiftDays = raster.slit.shiftSeconds / 86400;
        for (let timeIndex = 0; timeIndex < baseMap.ntime; timeIndex += 1) {
          const time = baseMap.timeMjd[timeIndex] + shiftDays;
          const previous = baseMap.timeMjd[Math.max(0, timeIndex - 1)] + shiftDays;
          const next = baseMap.timeMjd[Math.min(baseMap.ntime - 1, timeIndex + 1)] + shiftDays;
          const leftEdge = timeIndex === 0 ? time - (next - time) / 2 : (previous + time) / 2;
          const rightEdge = timeIndex === baseMap.ntime - 1 ? time + (time - previous) / 2 : (time + next) / 2;
          const destinationX = timeToSharedPlotX(leftEdge, minX, maxX, horizontal);
          const destinationRight = timeToSharedPlotX(rightEdge, minX, maxX, horizontal);
          if (destinationRight < margin.left || destinationX > margin.left + plotW) continue;
          ctx.drawImage(image, timeIndex, 0, 1, image.height, destinationX, margin.top, Math.max(0.5, destinationRight - destinationX), plotH);
        }
        ctx.restore();
      }
    }
    const drawnGroups = contourGroups.filter((group) => group.maps.length);
    if (drawnGroups.length) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(margin.left, margin.top, plotW, plotH);
      ctx.clip();
      for (const group of drawnGroups) {
        // Each contour-bound member's OWN shift - see the raster comment
        // above; a composite twin pair may be shifted differently from each
        // other and from the raster.
        const shiftDays = group.slit.shiftSeconds / 86400;
        for (const map of group.maps) {
          const frequencyIndex = map.freqIndex ?? 0;
          const cacheKey = `${group.slit.id}:${frequencyIndex}`;
          // Layer-driven threshold for a contour-bound radio slit (same
          // levelMode/levelReference/level as the panel overlay - see
          // resolveContourThreshold); legacy percent-of-own-peak otherwise.
          const threshold = group.isContourBound ? resolveContourThreshold(group, map) : map.dataMax * clamp(group.slit.contourLevelPercent, 1, 99) / 100;
          if (threshold === null || !Number.isFinite(threshold)) continue;
          const cached = contourCacheRef.current.get(cacheKey);
          const segments = cached?.intensity === map.intensity && cached.threshold === threshold
            ? cached.segments
            : contourSegmentsAtThreshold(map, threshold);
          if (!cached || cached.intensity !== map.intensity || cached.threshold !== threshold) {
            contourCacheRef.current.set(cacheKey, { intensity: map.intensity, threshold, segments });
          }
          const timeAtIndex = (value: number) => {
            const lower = clamp(Math.floor(value), 0, map.ntime - 1);
            const upper = clamp(Math.ceil(value), 0, map.ntime - 1);
            const fraction = value - Math.floor(value);
            return map.timeMjd[lower] + (map.timeMjd[upper] - map.timeMjd[lower]) * fraction + shiftDays;
          };
          // Same channel-index -> color mapping as the panel overlay: the
          // frequency-value-normalized colormap sample (contourBandColor),
          // not an index/count sample, when the layer settings are live.
          ctx.strokeStyle = group.isContourBound && group.contourLayer
            ? contourBandColor(group.contourLayer.contourCmap, map.freqGhz ?? 0, radioFreqGhz)
            : sampleColormap(channelPalette, frequencyIndex, Math.max(1, radioFrequencyCount), "frequency");
          ctx.lineWidth = 1.35;
          ctx.beginPath();
          for (const [start, end] of segments) {
            const startX = timeToSharedPlotX(timeAtIndex(start[0]), minX, maxX, horizontal);
            const endX = timeToSharedPlotX(timeAtIndex(end[0]), minX, maxX, horizontal);
            const startY = margin.top + plotH * (1 - start[1] / Math.max(1, map.npix - 1));
            const endY = margin.top + plotH * (1 - end[1] / Math.max(1, map.npix - 1));
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
          }
          ctx.stroke();
        }
      }
      ctx.restore();
      // Compact frequency legend, bottom-left of the plot - reads
      // identically to the panel overlay's GHz colorbar. Rows are prefixed
      // with the member's own name only when the composite stacks more than
      // one contour-bound family, disambiguating which twin each row is.
      const legendGroups = drawnGroups.filter((group) => group.isContourBound && group.contourLayer);
      if (legendGroups.length) {
        const rows = legendGroups.flatMap((group) => group.maps.map((map) => ({ group, map })));
        const multiMember = legendGroups.length > 1;
        const swatch = 8;
        const rowHeight = 12;
        const legendWidth = multiMember ? 132 : 78;
        const legendHeight = rows.length * rowHeight + 6;
        const startX = margin.left + 6;
        const startY = clamp(margin.top + plotH - legendHeight - 2, margin.top + 4, margin.top + plotH - legendHeight);
        ctx.fillStyle = "rgba(5, 7, 10, 0.72)";
        ctx.fillRect(startX - 4, startY - 2, legendWidth, legendHeight);
        ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        rows.forEach(({ group, map }, index) => {
          const y = startY + index * rowHeight;
          const color = contourBandColor(group.contourLayer!.contourCmap, map.freqGhz ?? 0, radioFreqGhz);
          ctx.fillStyle = color;
          ctx.fillRect(startX, y, swatch, swatch);
          ctx.fillStyle = "#e7ebf0";
          const label = multiMember ? `${group.slit.name} · ${(map.freqGhz ?? 0).toFixed(2)} GHz` : `${(map.freqGhz ?? 0).toFixed(2)} GHz`;
          ctx.fillText(label, startX + swatch + 4, y - 1);
        });
      }
    }
    ctx.strokeStyle = "#56606b";
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, plotW, plotH);
    ctx.fillStyle = "#aeb6c1";
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    const distanceMax = overallGeometryMap?.distanceArcsec.at(-1) ?? 0;
    ctx.fillText(`far · ${distanceMax.toFixed(1)}″`, margin.left - 5, margin.top + 4);
    ctx.fillText("origin · 0″", margin.left - 5, margin.top + plotH);
    ctx.textAlign = "right";
    ctx.fillStyle = "rgba(5, 7, 10, 0.72)";
    ctx.fillRect(margin.left + plotW - 68, margin.top, 68, 15);
    ctx.fillRect(margin.left + plotW - 46, margin.top + plotH - 15, 46, 15);
    ctx.fillStyle = "#aeb6c1";
    ctx.fillText(`${(distanceMax * 0.725).toFixed(1)} Mm`, margin.left + plotW - 4, margin.top + 4);
    ctx.fillText("0 Mm", margin.left + plotW - 4, margin.top + plotH - 4);
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    // Composite header: one line per distinct participating member (raster
    // first, then each drawn contour family), each in its own color with
    // its own shift - "document this: shifts may differ, which is
    // scientifically useful for lag work". A solo (non-composite) lane has
    // exactly one participant (raster and contour share the same slit), so
    // this reproduces the single-line header unchanged.
    const seenLabelIds = new Set<string>();
    const labelEntries: { name: string; color: string; shiftSeconds: number }[] = [];
    if (raster) {
      labelEntries.push({ name: raster.slit.name, color: raster.slit.color, shiftSeconds: raster.slit.shiftSeconds });
      seenLabelIds.add(raster.slit.id);
    }
    for (const group of drawnGroups) {
      if (seenLabelIds.has(group.slit.id)) continue;
      seenLabelIds.add(group.slit.id);
      labelEntries.push({ name: group.slit.name, color: group.slit.color, shiftSeconds: group.slit.shiftSeconds });
    }
    const labelLineHeight = 13;
    const labelBoxHeight = Math.max(16, labelEntries.length * labelLineHeight + 3);
    ctx.fillStyle = "rgba(5, 7, 10, 0.76)";
    ctx.fillRect(margin.left + 4, margin.top + 4, 200, labelBoxHeight);
    labelEntries.forEach((entry, index) => {
      ctx.fillStyle = entry.color;
      ctx.fillText(`${entry.name} · shift ${entry.shiftSeconds >= 0 ? "+" : ""}${entry.shiftSeconds.toFixed(0)} s`, margin.left + 8, margin.top + 7 + index * labelLineHeight);
    });
    // The cursor line follows the SELECTED member's own shift (unchanged
    // from before this feature) - the selected member is always one of the
    // composited participants, so this stays a meaningful reference point.
    const shiftedCursorMjd = currentMjd + slit.shiftSeconds / 86400;
    if (shiftedCursorMjd >= minX && shiftedCursorMjd <= maxX) {
      const x = timeToSharedPlotX(shiftedCursorMjd, minX, maxX, horizontal);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, margin.top + plotH);
      ctx.stroke();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [channelGutterVisible, channelPalette, contours, currentMjd, globalThresholds, maxX, minX, radioFreqGhz, radioFrequencyCount, raster, size, slit]);

  function timeAt(event: React.PointerEvent<HTMLCanvasElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const plot = spectrogramTimePlotRect(rect, channelGutterVisible);
    return timeFromSharedPlotX(event.clientX - rect.left, minX, maxX, plot) - slit.shiftSeconds / 86400;
  }

  function cursorX(plot: { x: number; width: number }): number | null {
    const shiftedCursorMjd = currentMjd + slit.shiftSeconds / 86400;
    if (shiftedCursorMjd < minX || shiftedCursorMjd > maxX) return null;
    return timeToSharedPlotX(shiftedCursorMjd, minX, maxX, plot);
  }

  function setLaneCursor(canvas: HTMLCanvasElement, cursor: "" | "ew-resize") {
    if (laneCursorRef.current === cursor) return;
    laneCursorRef.current = cursor;
    canvas.style.cursor = cursor;
  }

  function updateLaneCursor(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const plot = spectrogramTimePlotRect(rect, channelGutterVisible);
    const x = clamp(event.clientX - rect.left, plot.x, plot.x + plot.width);
    const lineX = cursorX(plot);
    const nearCursor = !panRef.current && dragRef.current === null && lineX !== null && Math.abs(x - lineX) <= cursorGrabWidth;
    setLaneCursor(event.currentTarget, dragRef.current === event.pointerId || nearCursor ? "ew-resize" : "");
  }

  // Zooms the SHARED spectrogram time window (xRange/minX..maxX) around the
  // pointer's time position, reusing the exact zoom math the spectrogram's
  // own wheel handler uses (zoomTimeWindow). Because the window is shared
  // (spectrogramTimeRange), the spectrogram canvas re-renders in sync.
  // Anchor is computed in the unshifted xRange domain (no slit.shiftSeconds
  // subtraction) since minX/maxX - the values being zoomed - are unshifted.
  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    if (!timeMjd.length) return;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const plot = spectrogramTimePlotRect(rect, channelGutterVisible);
    const anchor = timeFromSharedPlotX(event.clientX - rect.left, minX, maxX, plot);
    const [fullMin, fullMax] = timeBounds(timeMjd);
    const [nextMin, nextMax] = zoomTimeWindow(anchor, minX, maxX, fullMin, fullMax, event.deltaY);
    onTimeRangeChange({ min: mjdToUtc(nextMin), max: mjdToUtc(nextMax) });
  }

  return <section
    className="time-distance-lane"
    data-window-start={minX}
    data-window-end={maxX}
    data-shift-seconds={slit.shiftSeconds}
    data-distance-max-arcsec={overallGeometryMap?.distanceArcsec.at(-1) ?? 0}
    data-npix={overallGeometryMap?.npix ?? 0}
    data-ntime={overallGeometryMap?.ntime ?? 0}
    data-base-freq-index={baseMap?.freqIndex ?? "none"}
    data-contour-map-count={contourGroups.reduce((sum, group) => sum + group.maps.length, 0)}
    data-twin-composite={raster && contours.some((entry) => entry.slit.id !== raster.slit.id) ? "true" : contours.length > 1 ? "true" : "false"}
    data-plot-left={channelGutterVisible ? 68 : 54}
    data-plot-right={10}
  >
    <canvas
      ref={(canvas) => { localCanvasRef.current = canvas; canvasRef.current = canvas; }}
      aria-label={`Time-distance lane for ${slit.name}`}
      onPointerDown={(event) => {
        if (event.button === 2) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const plot = spectrogramTimePlotRect(rect, channelGutterVisible);
        const x = clamp(event.clientX - rect.left, plot.x, plot.x + plot.width);
        const lineX = cursorX(plot);
        const nearCursor = lineX !== null && Math.abs(x - lineX) <= cursorGrabWidth;
        if (event.button === 1 || event.shiftKey || event.altKey || !nearCursor) {
          panRef.current = { pointerId: event.pointerId, startX: x, startMin: minX, startMax: maxX, moved: false };
          event.currentTarget.setPointerCapture(event.pointerId);
          updateLaneCursor(event);
          event.preventDefault();
          return;
        }
        dragRef.current = event.pointerId;
        event.currentTarget.setPointerCapture(event.pointerId);
        onTimeScrubStart();
        onTimeScrub(timeAt(event));
        updateLaneCursor(event);
      }}
      onPointerMove={(event) => {
        updateLaneCursor(event);
        if (panRef.current?.pointerId === event.pointerId) {
          const rect = event.currentTarget.getBoundingClientRect();
          const plot = spectrogramTimePlotRect(rect, channelGutterVisible);
          const x = clamp(event.clientX - rect.left, plot.x, plot.x + plot.width);
          const dx = x - panRef.current.startX;
          if (!panRef.current.moved && Math.abs(dx) < 2) return;
          panRef.current.moved = true;
          const span = panRef.current.startMax - panRef.current.startMin;
          const delta = dx / Math.max(1, plot.width) * span;
          const [fullMin, fullMax] = timeBounds(timeMjd);
          const nextMin = clamp(panRef.current.startMin - delta, fullMin, fullMax - span);
          onTimeRangeChange({ min: mjdToUtc(nextMin), max: mjdToUtc(nextMin + span) });
          event.preventDefault();
        } else if (dragRef.current === event.pointerId) {
          onTimeScrub(timeAt(event));
        }
      }}
      onPointerUp={(event) => {
        if (panRef.current?.pointerId === event.pointerId) {
          const pan = panRef.current;
          panRef.current = null;
          if (!pan.moved && event.button === 0) onTimeSelect(timeAt(event));
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
          updateLaneCursor(event);
          event.preventDefault();
          return;
        }
        if (dragRef.current !== event.pointerId) return;
        dragRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        onTimeSelect(timeAt(event));
        onTimeScrubEnd();
        updateLaneCursor(event);
      }}
      onPointerCancel={(event) => {
        panRef.current = null;
        if (dragRef.current === event.pointerId) onTimeScrubEnd();
        dragRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        updateLaneCursor(event);
      }}
      onPointerEnter={updateLaneCursor}
      onPointerLeave={(event) => {
        if (dragRef.current !== event.pointerId && !panRef.current) setLaneCursor(event.currentTarget, "");
      }}
      onWheel={handleWheel}
    />
    {missingTwinHint && <div className="time-distance-lane-hint">{missingTwinHint}</div>}
  </section>;
}

function SlitInspectorCard({
  slits,
  selectedSlitId,
  sourceOptions,
  defaultSourceId,
  draftWidthArcsec,
  widthArcsecBounds,
  draftSmoothPx,
  selectedResult,
  imageLinked,
  laneHasContent,
  fan,
  selectedFanMember,
  fanIntermediateCount,
  fanDrawStage,
  fanRedrawTarget,
  radioFreqGhz,
  channelPalette,
  extracting,
  drawArmed,
  pinLane,
  extractAll,
  npzHref,
  slitWidthScaleArcsec,
  onSelect,
  onRename,
  onColor,
  onVisibility,
  onReverse,
  onDuplicate,
  onLink,
  onDelete,
  onDefaultSourceChange,
  onSlitSourceChange,
  onFanSourceChange,
  onWidthChange,
  onSmoothCommit,
  onFanSmoothChange,
  onDraw,
  onDrawFan,
  onDeleteFan,
  onFanCount,
  onSelectFan,
  onPromoteFan,
  onRedrawFanBoundary,
  onExtractFanCurve,
  onExtract,
  onExtractAllChange,
  onCancel,
  onShift,
  onDisplay,
  onFreqSelection,
  onBaseFreq,
  onContourFreq,
  onContourLevel,
  onPinLane,
  onExportPng,
  onClose,
  cardSize,
  onCardSizeChange
}: {
  slits: SlitDefinition[];
  selectedSlitId: string;
  sourceOptions: SlitSourceOption[];
  defaultSourceId: string;
  /** Draft width in arcsec (USER DESIGN DECISION: a fixed physical quantity, not bound-source pixels) for the default binding, shown when nothing is selected. */
  draftWidthArcsec: number;
  /** [min, max] bounds in arcsec for the Width [″] control - see slitWidthArcsecBounds in the parent. GLOBAL, not keyed by the current binding. */
  widthArcsecBounds: [number, number];
  draftSmoothPx: number;
  selectedResult: SlitResult | null;
  /** Whether the selected slit is image-bound to a live, linked panel layer (see LAYER-LINKED RASTER SCALE / imageLayerForSlit). */
  imageLinked: boolean;
  /** Whether the lane has anything to show (selected slit's own result, or a linked twin's - see TWIN OVERLAY COMPOSITING). */
  laneHasContent: boolean;
  fan: FanDefinition | null;
  selectedFanMember: number;
  fanIntermediateCount: number;
  fanDrawStage: 0 | 1 | 2;
  fanRedrawTarget: "A" | "B" | null;
  radioFreqGhz: number[];
  channelPalette: ContourColormap;
  extracting: boolean;
  drawArmed: boolean;
  pinLane: boolean;
  /** "All slits" toggle: when true, Extract batch-extracts every visible slit instead of just the selected one. */
  extractAll: boolean;
  npzHref: string;
  /** Perpendicular-width swath scale for a given BOUND source, in arcsec/px - see slitWidthScaleArcsec in the parent. A linked twin has its own sourceId/binding, so it gets its own scale even though it shares its origin's width in px. */
  slitWidthScaleArcsec: (sourceId: string) => number;
  onSelect: (slitId: string) => void;
  onRename: (slitId: string, name: string) => void;
  onColor: (slitId: string, color: string) => void;
  onVisibility: (slitId: string) => void;
  onReverse: (slitId: string) => void;
  onDuplicate: (slitId: string) => void;
  onLink: (slitId: string) => void;
  onDelete: (slitId: string) => void;
  onDefaultSourceChange: (sourceId: string) => void;
  onSlitSourceChange: (slitId: string, sourceId: string) => void;
  onFanSourceChange: (sourceId: string) => void;
  onWidthChange: (widthArcsec: number) => void;
  /** Commits the single Smooth [px] control (FEATURE 1/UX consolidation): always updates slitDraftSmoothPx (the hidden draw-time default) as a side effect, then - if a slit is selected - re-smooths it, redirecting through a linked twin to its origin and through a fan-promoted member to the fan itself. See commitSlitSmooth in the parent. */
  onSmoothCommit: (smoothPx: number) => void;
  /** Re-smooths the fan's two boundaries from their stored raw strokes (FEATURE 1) - see reSmoothFan. Independent of the per-slit Smooth [px] control - this is the Fan family section's own control. */
  onFanSmoothChange: (smoothPx: number) => void;
  onDraw: () => void;
  onDrawFan: () => void;
  onDeleteFan: () => void;
  onFanCount: (count: number) => void;
  onSelectFan: (memberIndex: number) => void;
  onPromoteFan: (memberIndex: number) => void;
  onRedrawFanBoundary: (target: "A" | "B") => void;
  onExtractFanCurve: (memberIndex: number) => void;
  onExtract: () => void;
  onExtractAllChange: (value: boolean) => void;
  onCancel: () => void;
  onShift: (seconds: number) => void;
  onDisplay: (display: SlitDisplayState) => void;
  onFreqSelection: (freqIndices: number[]) => void;
  onBaseFreq: (freqIndex: number | null) => void;
  onContourFreq: (freqIndex: number) => void;
  onContourLevel: (level: number) => void;
  onPinLane: (value: boolean) => void;
  onExportPng: () => void;
  onClose: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const floatingCard = useDraggableFloatingCard({ minWidth: 420, minHeight: 480, size: cardSize, onSizeChange: onCardSizeChange });
  const selected = slits.find((slit) => slit.id === selectedSlitId) ?? null;
  const selectedMaps = selectedResult ? slitResultMaps(selectedResult) : [];
  const baseMap = selectedMaps.find((map) => map.freqIndex === selected?.baseFreqIndex) ?? selectedResult;
  const selectedSource = sourceOptions.find((option) => option.sourceId === selected?.sourceId);
  const radioSelected = selectedSource?.layer.sourceRoleSnapshot === "radio";
  // A radio-bound slit with zero selected channels has nothing to extract:
  // the backend's freqIndices fallback treats an empty array as "unset" and
  // silently re-extracts a single layerParams channel instead of erroring
  // (see SlitExtractRequest/extract_slit in backend/app.py), so an empty
  // selection must be blocked client-side rather than sent through.
  const noChannelsSelected = radioSelected && (selected?.freqIndices.length ?? 0) === 0;
  const visibleSlitCount = slits.filter((slit) => slit.visible).length;
  // The old percent-of-max contour control and its per-channel include/
  // exclude checkboxes are replaced by layer-driven rendering only for
  // slits actually bound through a contours-kind layer (bindingKind
  // "contours"); an image-bound radio slit (no contour layer to mirror)
  // keeps the legacy controls untouched.
  const isContourBoundSelected = selected?.bindingKind === "contours";
  // A contour-bound slit with no base map chosen (baseFreqIndex === null)
  // renders as a pure contour family: every visual (thresholds, colors) is
  // inherited live from the bound overlay layer, and there is no raster to
  // apply a Range/Colormap/Scale to - those controls are meaningless (and,
  // before this, confusingly carried over whatever map's stats they last
  // showed). Choosing a base map for a contour-bound slit brings its own
  // raster back, so the controls return along with it (see onBaseFreq's
  // p1/p99 resync in the parent).
  //
  // LAYER-LINKED RASTER SCALE: an image-bound slit with a live bound panel
  // layer (imageLinked, from imageLayerForSlit in the parent) takes its
  // raster's vmin/vmax/cmap/scale from that layer's own display settings
  // instead - same rationale as the contour case above (live, not a
  // snapshot), so the card's own Range/Colormap/Scale controls are
  // meaningless for it too and are hidden the same way.
  const hideRasterControls = (isContourBoundSelected && selected?.baseFreqIndex === null) || imageLinked;
  const selectedGeometry = selected ? resolveSlitGeometry(selected, slits) : null;
  // Single Smooth [px] control's displayed value (FEATURE 1/UX
  // consolidation) - mirrors the exact resolution commitSlitSmooth performs
  // in the parent, so the number shown is always the one a commit would
  // actually edit: with a twin selected, follow linkedTo to its origin
  // (slitOriginId); if that resolved target is a live fan's promoted
  // member, show the FAN's smoothPx (a commit re-smooths the fan, not the
  // member); otherwise show the target's own smoothPx. With nothing
  // selected, show the hidden draw-time default (draftSmoothPx) - a commit
  // there only ever updates that default.
  const smoothTarget = selected ? (slits.find((slit) => slit.id === slitOriginId(selected, slits)) ?? selected) : null;
  const smoothTargetFanPromoted = Boolean(fan && smoothTarget && Object.values(fan.promotedMembers).includes(smoothTarget.id));
  const smoothValue = !selected
    ? draftSmoothPx
    : smoothTargetFanPromoted
    ? fan!.smoothPx ?? draftSmoothPx
    : smoothTarget?.smoothPx ?? draftSmoothPx;
  // Width control value, in arcsec (USER DESIGN DECISION - the control IS
  // arcsec now, not bound-source pixels): selectedGeometry's widthArcsec
  // when a slit is selected, else the draft width for the default binding.
  // The muted px hint below still converts through the row's OWN binding's
  // scale (slitWidthScaleArcsec) so it reflects what a request-build-time
  // conversion (extractSlit) would actually send for that specific binding.
  const widthValueArcsec = selectedGeometry ? selectedGeometry.widthArcsec : draftWidthArcsec;
  const widthHintSourceId = selected?.sourceId ?? defaultSourceId;
  const widthHintPx = Math.max(1, Math.round(widthValueArcsec / slitWidthScaleArcsec(widthHintSourceId)));
  const widthHintSourceLabel = (sourceOptions.find((option) => option.sourceId === widthHintSourceId)?.label ?? "")
    .replace(/ \(not displayed - raw\)$/, "")
    .replace(/ - contours$/, "");
  const bounds = baseMap
    ? { min: baseMap.dataMin, max: Math.max(baseMap.dataMin + Number.EPSILON, baseMap.dataMax) }
    : { min: 0, max: 1 };
  const familyCurves = fan ? fanFamilyCurves(fan) : [];
  const selectedFanCurveValid = Boolean(fan) && selectedFanMember >= 0 && selectedFanMember < familyCurves.length;
  const selectedFanCurvePromoted = selectedFanCurveValid && Boolean(fan!.promotedMembers[String(selectedFanMember)]);
  return <section ref={floatingCard.cardRef} className="radio-alignment-card slit-inspector-card" role="dialog" aria-label="Slit inspector" tabIndex={-1} style={floatingCard.style}>
    <header ref={floatingCard.headerRef} className={`radio-alignment-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
      <div><strong>Slit inspector</strong><small>{slits.length} slit{slits.length === 1 ? "" : "s"}</small></div>
      <button className="probe-close" type="button" aria-label="Close slit inspector" onPointerDown={(event) => { event.stopPropagation(); onClose(); }}>×</button>
    </header>
    <label className="tracking-source-picker">Default binding
      <select aria-label="Default slit binding" value={defaultSourceId} onChange={(event) => onDefaultSourceChange(event.target.value)}>
        {sourceOptions.map((option) => <option key={option.sourceId} value={option.sourceId}>{option.label}{option.sides.length ? ` · ${option.sides.map((side) => side === "left" ? "Left" : "Right").join("+")}` : ""}</option>)}
      </select>
    </label>
    <div className="fan-controls">
      <label>Intermediate curves<CommitInput type="number" min={1} max={20} scrubStep={1} value={fanIntermediateCount} ariaLabel="Fan intermediate curve count" onCommit={(value) => onFanCount(clamp(Math.round(numberValue(value, 3)), 1, 20))} /></label>
      <button className={`button ${fanDrawStage ? "active" : ""}`} type="button" onClick={onDrawFan} disabled={!defaultSourceId || Boolean(fanRedrawTarget)}>
        <Pencil size={14} /> {fanDrawStage === 2 ? "Draw boundary B…" : fanDrawStage === 1 ? "Draw boundary A…" : "Draw fan"}
      </button>
      <button className="button icon-button" type="button" onClick={onDeleteFan} disabled={!fan && !fanDrawStage} title="Delete fan" aria-label="Delete fan">
        <Trash2 size={14} />
      </button>
    </div>
    {fan && <>
      <div className="track-editor-section-heading">Fan family - {fan.intermediateCount + 2}</div>
      <label className="tracking-source-picker">Fan binding
        <select aria-label="Fan source binding" value={fan.sourceId} onChange={(event) => onFanSourceChange(event.target.value)}>
          {sourceOptions.map((option) => <option key={option.sourceId} value={option.sourceId}>{option.label}{option.sides.length ? ` · ${option.sides.map((side) => side === "left" ? "Left" : "Right").join("+")}` : ""}</option>)}
        </select>
      </label>
      <label>Smooth [px]<CommitInput
        type="number"
        min={0}
        max={SLIT_SMOOTH_MAX_PX}
        scrubStep={0.5}
        value={fan.smoothPx ?? draftSmoothPx}
        disabled={!fan.rawBoundaryA || !fan.rawBoundaryB}
        ariaLabel="Fan boundary smoothing sigma in pixels"
        title={!fan.rawBoundaryA || !fan.rawBoundaryB
          ? "Redraw both boundaries to enable re-smoothing"
          : "Re-smooths both boundaries from their original hand-drawn strokes and regenerates the fan family; promoted slits update in place."}
        onCommit={(value) => onFanSmoothChange(clamp(numberValue(value, fan.smoothPx ?? DEFAULT_SLIT_SMOOTH_PX), 0, SLIT_SMOOTH_MAX_PX))}
      /></label>
      {fanRedrawTarget && <p className="track-editor-empty">Draw the replacement for Boundary {fanRedrawTarget} on either image panel; Esc cancels.</p>}
      <div className="fan-member-list" role="list" aria-label="Fan curves">
        {familyCurves.map((curve, index, family) => {
          const boundary = index === 0 || index === family.length - 1;
          const boundaryLabel = index === 0 ? "A" : "B";
          const promoted = Boolean(fan.promotedMembers[String(index)]);
          return <div
            key={index}
            className={`fan-member-row ${selectedFanMember === index ? "selected" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => onSelectFan(index)}
            onDoubleClick={() => onPromoteFan(index)}
          >
            <span className="track-color-dot" style={{ background: boundary ? "#ff5353" : "#56dc72" }} />
            <span>{boundary ? `Boundary ${boundaryLabel}` : `Curve ${index}`}</span>
            <small>f={(index / Math.max(1, family.length - 1)).toFixed(2)} · {curve.length} samples</small>
            {boundary && <button
              className={`track-row-icon ${fanRedrawTarget === boundaryLabel ? "active" : ""}`}
              type="button"
              disabled={fanDrawStage !== 0}
              title="Redraw this boundary"
              aria-label={`Redraw boundary ${boundaryLabel}`}
              onClick={(event) => { event.stopPropagation(); onRedrawFanBoundary(boundaryLabel); }}
            ><Pencil size={13} /></button>}
            <button className="track-row-icon" type="button" disabled={promoted} title={promoted ? "Already promoted" : "Promote to slit"} aria-label={`Promote fan curve ${index + 1}`} onClick={(event) => { event.stopPropagation(); onPromoteFan(index); }}><Plus size={13} /></button>
          </div>;
        })}
      </div>
      {selectedFanCurveValid && <div className="fan-selected-actions">
        <button
          className="button"
          type="button"
          disabled={selectedFanCurvePromoted}
          title={selectedFanCurvePromoted ? "Already promoted" : "Promote to slit"}
          aria-label="Promote selected fan curve to slit"
          onClick={() => onPromoteFan(selectedFanMember)}
        ><Plus size={14} /> Promote to slit</button>
        <button
          className="button primary"
          type="button"
          disabled={extracting}
          title="Extract this curve"
          aria-label="Extract this fan curve"
          onClick={() => onExtractFanCurve(selectedFanMember)}
        ><Activity size={14} /> Extract this curve</button>
      </div>}
    </>}
    <div className="track-editor-section-heading">Slits - {slits.length}</div>
    <div className="slit-list" role="list" aria-label="Slits">
      {slits.map((slit) => {
        const isTwin = Boolean(slit.linkedTo);
        const geometry = resolveSlitGeometry(slit, slits);
        return <div key={slit.id} className={`slit-row ${slit.id === selectedSlitId ? "selected" : ""}`} role="button" tabIndex={0} onClick={() => onSelect(slit.id)}>
        <button className="track-select" type="button" aria-label={`Select ${slit.name}`}><span className="track-color-dot" style={{ background: slit.color }} /></button>
        <span className="slit-row-name">
          <CommitInput value={slit.name} ariaLabel={`Rename ${slit.name}`} onCommit={(name) => onRename(slit.id, name || slit.name)} />
          {isTwin && <Link2 className="slit-link-glyph" size={11} aria-label="Linked twin - geometry follows another slit" />}
        </span>
        <select
          aria-label={`${slit.name} source binding`}
          value={slit.sourceId}
          onClick={(event) => event.stopPropagation()}
          onChange={(event) => onSlitSourceChange(slit.id, event.target.value)}
        >{sourceOptions.map((option) => <option key={option.sourceId} value={option.sourceId}>{option.label}</option>)}</select>
        <input type="color" aria-label={`${slit.name} color`} value={slit.color} onChange={(event) => onColor(slit.id, event.target.value)} />
        <button className="track-row-icon" type="button" aria-label={`${slit.visible ? "Hide" : "Show"} ${slit.name}`} onClick={(event) => { event.stopPropagation(); onVisibility(slit.id); }}>{slit.visible ? <Eye size={13} /> : <EyeOff size={13} />}</button>
        <small title="Width is a fixed physical corridor in arcsec (shared with any linked twin); ≈px is that width converted through THIS row's own binding's pixel scale.">{geometry.curveArcsec.length} px · w {geometry.widthArcsec.toFixed(1)}″ · ≈{Math.max(1, Math.round(geometry.widthArcsec / slitWidthScaleArcsec(slit.sourceId)))} px</small>
        {isTwin
          ? <span className="track-row-icon" aria-hidden="true" />
          : <button className="track-row-icon" type="button" aria-label={`Reverse slit direction for ${slit.name}`} title="Reverse slit direction" onClick={(event) => { event.stopPropagation(); onReverse(slit.id); }}><ArrowLeftRight size={13} /></button>}
        <button className="track-row-icon" type="button" aria-label={`Create linked twin of ${slit.name}`} title="Create linked twin" onClick={(event) => { event.stopPropagation(); onLink(slit.id); }}><Link2 size={13} /></button>
        <button className="track-row-icon" type="button" aria-label={`Duplicate ${slit.name}`} title="Duplicate slit" onClick={(event) => { event.stopPropagation(); onDuplicate(slit.id); }}><Copy size={13} /></button>
        <button className="track-row-icon delete" type="button" aria-label={`Delete ${slit.name}`} onClick={(event) => { event.stopPropagation(); onDelete(slit.id); }}><Trash2 size={13} /></button>
      </div>;
      })}
      {!slits.length && <p className="track-editor-empty">Draw a slit, or promote a fan curve, then Extract.</p>}
    </div>
    {radioSelected && <div className="slit-frequency-controls">
      <div className="track-editor-section-heading slit-frequency-heading">
        <span>Radio channels - {selected?.freqIndices.length ?? 0}</span>
        <span className="slit-frequency-bulk-actions">
          <button
            type="button"
            className="slit-frequency-bulk-btn"
            title="Select all channels"
            aria-label="Select all radio channels"
            onClick={() => selected && onFreqSelection(radioFreqGhz.map((_, index) => index))}
          >All</button>
          <button
            type="button"
            className="slit-frequency-bulk-btn"
            title="Deselect all channels"
            aria-label="Deselect all radio channels"
            onClick={() => selected && onFreqSelection([])}
          >None</button>
        </span>
      </div>
      <div className="slit-frequency-chips">
        {radioFreqGhz.map((frequency, index) => {
          const active = selected?.freqIndices.includes(index) ?? false;
          const color = sampleColormap(channelPalette, index, Math.max(1, radioFreqGhz.length), "frequency");
          return <button
            key={index}
            type="button"
            className={`slit-frequency-chip ${active ? "active" : ""}`}
            style={{ borderColor: color, color: active ? "#05070a" : color, background: active ? color : undefined }}
            aria-pressed={active}
            aria-label={`${active ? "Remove" : "Add"} ${frequency.toFixed(3)} GHz`}
            onClick={() => selected && onFreqSelection(active ? selected.freqIndices.filter((value) => value !== index) : [...selected.freqIndices, index].sort((a, b) => a - b))}
          >{frequency.toFixed(2)}</button>;
        })}
      </div>
      {selectedMaps.length > 0 && <div className="slit-map-controls">
        <label>Base map<select
          value={selected?.baseFreqIndex ?? ""}
          onChange={(event) => onBaseFreq(event.target.value === "" ? null : Number(event.target.value))}
        >
          <option value="">None</option>
          {selectedMaps.map((map) => <option key={map.freqIndex ?? -1} value={map.freqIndex ?? 0}>{map.freqGhz?.toFixed(3) ?? "image"} GHz</option>)}
        </select></label>
        {isContourBoundSelected
          ? <span className="slit-contour-family-note" title="Every selected radio channel is contoured at the bound overlay layer's level - scrub it in the layer inspector.">
              Contours: layer-driven ({selectedMaps.filter((map) => map.freqIndex !== null).length} ch)
            </span>
          : <>
              <label>Contour level [%]<CommitInput type="number" min={1} max={99} scrubStep={0.5} value={selected?.contourLevelPercent ?? 70} ariaLabel="Time-distance contour percent of maximum" onCommit={(value) => onContourLevel(clamp(numberValue(value, 70), 1, 99))} /></label>
              <div className="slit-contour-toggles">{selectedMaps.filter((map) => map.freqIndex !== selected?.baseFreqIndex).map((map) => {
                const index = map.freqIndex ?? 0;
                const checked = selected?.contourFreqIndices.includes(index) ?? false;
                const color = sampleColormap(channelPalette, index, Math.max(1, radioFreqGhz.length), "frequency");
                return <label key={index} style={{ color }}><input type="checkbox" checked={checked} onChange={() => onContourFreq(index)} /> {map.freqGhz?.toFixed(2)} GHz</label>;
              })}</div>
            </>}
      </div>}
    </div>}
    <div className="slit-geometry-controls">
      <label>Width [″]<CommitInput
        type="number"
        min={widthArcsecBounds[0]}
        max={widthArcsecBounds[1]}
        scrubStep={0.5}
        value={Number(widthValueArcsec.toFixed(1))}
        disabled={Boolean(selected?.linkedTo)}
        ariaLabel="Slit width in arcsec"
        title={selected?.linkedTo
          ? "Width follows the linked original - select it to change width"
          : `Fixed physical corridor width in arcsec (${widthArcsecBounds[0].toFixed(1)}–${widthArcsecBounds[1].toFixed(0)}″, up to the largest loaded source's field of view) - a linked twin samples the SAME sky corridor at its own binding's pixel scale. Shift-drag for x10, Alt-drag for x0.1.`}
        onCommit={(value) => onWidthChange(clamp(numberValue(value, DEFAULT_SLIT_WIDTH_ARCSEC), widthArcsecBounds[0], widthArcsecBounds[1]))}
      /></label>
      <span className="slit-width-readout" title="Width converted through this row's own binding's pixel scale - see the per-slit ≈px hint above for what a request-build-time conversion would send.">≈ {widthHintPx} px{widthHintSourceLabel ? ` (${widthHintSourceLabel})` : ""}</span>
      <label>Smooth [px]<CommitInput
        type="number"
        min={0}
        max={SLIT_SMOOTH_MAX_PX}
        scrubStep={0.5}
        value={smoothValue}
        ariaLabel="Smoothing sigma in pixels for the selected slit, or the default for new slits when nothing is selected"
        title="Smoothing sigma in pixels. Applies to the selected slit (twins follow their origin); with nothing selected, sets the default for new slits."
        onCommit={(value) => onSmoothCommit(clamp(numberValue(value, smoothValue), 0, SLIT_SMOOTH_MAX_PX))}
      /></label>
      <button className={`button ${drawArmed ? "active" : ""}`} type="button" onClick={onDraw} disabled={!defaultSourceId}><Pencil size={14} /> {drawArmed ? "Draw on either panel…" : "Draw slit"}</button>
      <button
        className="button primary"
        type="button"
        onClick={onExtract}
        disabled={extractAll ? extracting || !visibleSlitCount : (!selected || extracting || noChannelsSelected)}
        title={extractAll ? (visibleSlitCount ? undefined : "No visible slits to extract") : (noChannelsSelected ? "Select at least one channel" : undefined)}
      ><Activity size={14} /> {extractAll ? "Extract all" : "Extract"}</button>
      {extracting && <button className="button" type="button" onClick={onCancel}><Square size={13} /> Cancel</button>}
      <label
        className="slit-extract-all-toggle"
        title="When on, Extract runs every visible slit in one shared-source-batched operation instead of just the selected slit."
      >
        <input
          type="checkbox"
          checked={extractAll}
          onChange={(event) => onExtractAllChange(event.target.checked)}
        /> All slits
      </label>
    </div>
    <div className="slit-display-controls">
      <label>Shift [s]<CommitInput type="number" scrubStep={1} value={selected?.shiftSeconds ?? 0} disabled={!selected} ariaLabel="Slit time shift in seconds" onCommit={(value) => onShift(numberValue(value, 0))} /></label>
      <label><input type="checkbox" checked={pinLane} onChange={(event) => onPinLane(event.target.checked)} /> Pin lane</label>
      {!hideRasterControls && <>
        <label>Colormap<select value={selected?.display.cmap ?? "magma"} disabled={!selected} onChange={(event) => selected && onDisplay({ ...selected.display, cmap: normalizeColormap(event.target.value, "magma") })}>{COLORMAP_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}</select></label>
        <label>Scale<select value={selected?.display.scale ?? "linear"} disabled={!selected} onChange={(event) => selected && onDisplay({ ...selected.display, scale: event.target.value as SlitDisplayState["scale"] })}><option value="linear">Linear</option><option value="sqrt">Sqrt</option><option value="log">Log</option></select></label>
      </>}
    </div>
    {hideRasterControls
      ? <p className="slit-contour-family-note">{isContourBoundSelected ? "Contours linked to the radio overlay layer (levels, colors)." : "Map display linked to the image layer (range, colormap)."}</p>
      : selected && <DualRangeRow
          bounds={bounds}
          minValue={String(selected.display.vmin)}
          maxValue={String(selected.display.vmax)}
          disabled={!selectedResult}
          onMinCommit={(value) => onDisplay({ ...selected.display, vmin: numberValue(value, bounds.min) })}
          onMaxCommit={(value) => onDisplay({ ...selected.display, vmax: numberValue(value, bounds.max) })}
        />}
    <div
      className="slit-result-summary"
      data-map-cache-keys={selectedMaps.map((map) => map.cacheKey).join(",")}
      data-map-wall-seconds={selectedMaps.map((map) => map.mapWallSeconds ?? map.wallSeconds).join(",")}
      data-map-cache-hits={selectedMaps.map((map) => map.mapCacheHit ?? map.cacheHit).join(",")}
    >{selectedResult
      ? `${selectedResult.npix} px × ${selectedResult.ntime} frames × ${selectedMaps.length} map${selectedMaps.length === 1 ? "" : "s"} · ${selectedResult.wallSeconds.toFixed(2)} s${selectedResult.cacheHit ? " · cached" : ""}`
      : "Extract a slit to render its time-distance lane."}</div>
    <footer className="slit-export-actions">
      <a className={`button icon-button ${npzHref ? "" : "disabled"}`} href={npzHref || undefined} aria-disabled={!npzHref} download title="Export time-distance NPZ" aria-label="Export time-distance NPZ"><Download size={14} /></a>
      <button className="button icon-button" type="button" onClick={onExportPng} disabled={!laneHasContent} title="Export lane PNG" aria-label="Export time-distance PNG"><Download size={14} /></button>
    </footer>
    <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
  </section>;
}

function TrackEditorCard({
  tracks,
  trackingSources,
  trackingSourceId,
  selectedTrackId,
  selectedAnchorFrame,
  currentFrame,
  currentMjd,
  timeMjd,
  pixelToWorldAffine,
  rangeStart,
  rangeEnd,
  direction,
  trackingActive,
  canUndo,
  canRedo,
  canSuggest,
  exportHref,
  onTrackingSourceChange,
  onSelectTrack,
  onRename,
  onColor,
  onVisibility,
  onDeleteTrack,
  onSelectAnchor,
  onStepFrame,
  onTimeSelect,
  onTimeScrubStart,
  onTimeScrubEnd,
  onMoveAnchor,
  onDeleteAnchor,
  onAddKeyFrame,
  onDirectionChange,
  onAutoTrack,
  onStepTrack,
  onStop,
  onUndo,
  onRedo,
  onSuggest,
  onClose,
  cardSize,
  onCardSizeChange
}: {
  tracks: SadTrack[];
  trackingSources: TrackingSourceOption[];
  trackingSourceId: string;
  selectedTrackId: string;
  selectedAnchorFrame: number | null;
  currentFrame: number;
  currentMjd: number;
  timeMjd: number[];
  pixelToWorldAffine: Affine;
  rangeStart: number;
  rangeEnd: number;
  direction: TrackDirection;
  trackingActive: boolean;
  canUndo: boolean;
  canRedo: boolean;
  canSuggest: boolean;
  exportHref: string;
  onTrackingSourceChange: (sourceId: string) => void;
  onSelectTrack: (trackId: string) => void;
  onRename: (trackId: string, label: string) => void;
  onColor: (trackId: string, color: string) => void;
  onVisibility: (trackId: string) => void;
  onDeleteTrack: (trackId: string) => void;
  onSelectAnchor: (anchor: TrackAnchor) => void;
  onStepFrame: (delta: number) => void;
  onTimeSelect: (mjd: number) => void;
  onTimeScrubStart: () => void;
  onTimeScrubEnd: () => void;
  onMoveAnchor: (frameIndex: number, nextFrame: number) => void;
  onDeleteAnchor: () => void;
  onAddKeyFrame: () => void;
  onDirectionChange: (direction: TrackDirection) => void;
  onAutoTrack: () => void;
  onStepTrack: () => void;
  onStop: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onSuggest: () => void;
  onClose: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const floatingCard = useDraggableFloatingCard({ minWidth: 460, minHeight: 420, size: cardSize, onSizeChange: onCardSizeChange });
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const anchorDragRef = useRef<{ pointerId: number; frameIndex: number; nextFrame: number } | null>(null);
  const timeDragRef = useRef<number | null>(null);
  const [previewFrame, setPreviewFrame] = useState<number | null>(null);
  const selected = tracks.find((track) => track.id === selectedTrackId) ?? tracks[0] ?? null;
  const minimum = Math.min(rangeStart, rangeEnd);
  const maximum = Math.max(rangeStart, rangeEnd);
  const [viewWindow, setViewWindow] = useState<[number, number]>([minimum, maximum]);
  const viewStart = clamp(Math.min(viewWindow[0], viewWindow[1]), minimum, maximum);
  const viewEnd = clamp(Math.max(viewWindow[0], viewWindow[1]), viewStart, maximum);
  const viewSpan = Math.max(1, viewEnd - viewStart);
  const inspectedFrame = selectedAnchorFrame ?? currentFrame;
  const inspectedPoint = selected?.points.find((point) => point.frameIndex === inspectedFrame);
  const speedKmS = selected ? displayedTrackSpeedKmS(selected, inspectedFrame, pixelToWorldAffine) : null;
  const position = (frame: number) => `${clamp((frame - viewStart) / viewSpan, 0, 1) * 100}%`;
  const minimumMjd = timeMjd[viewStart] ?? timeMjd[minimum] ?? timeMjd[0] ?? 0;
  const maximumMjd = timeMjd[viewEnd] ?? timeMjd[maximum] ?? timeMjd.at(-1) ?? minimumMjd;
  // The strip's x-axis is FRAME INDEX (diamonds, 0..N labels, cursor readout).
  // Map MJD <-> fractional frame index by interpolating between bracketing
  // native frames so the playhead line lands on the same axis as the diamonds
  // even when the source cadence is irregular.
  const fractionalIndexAt = (mjd: number): number => {
    if (mjd <= minimumMjd) return viewStart;
    if (mjd >= maximumMjd) return viewEnd;
    for (let index = viewStart; index < viewEnd; index += 1) {
      const t0 = timeMjd[index];
      const t1 = timeMjd[index + 1];
      if (t0 <= mjd && mjd <= t1) return t1 > t0 ? index + (mjd - t0) / (t1 - t0) : index;
    }
    return viewEnd;
  };
  const mjdAtFractionalIndex = (value: number): number => {
    const lower = clamp(Math.floor(value), viewStart, viewEnd);
    const upper = Math.min(lower + 1, viewEnd);
    const t0 = timeMjd[lower] ?? minimumMjd;
    const t1 = timeMjd[upper] ?? maximumMjd;
    return t0 + (t1 - t0) * clamp(value - lower, 0, 1);
  };
  const timePosition = (mjd: number) => position(fractionalIndexAt(mjd));

  useEffect(() => {
    floatingCard.cardRef.current?.focus();
  }, []);

  useEffect(() => {
    if (tracks.length && !selectedTrackId) onSelectTrack(tracks[0].id);
  }, [onSelectTrack, selectedTrackId, tracks]);

  useEffect(() => {
    setViewWindow([minimum, maximum]);
  }, [maximum, minimum, selected?.id]);

  function frameAtPointer(event: React.PointerEvent<HTMLElement>): number {
    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect) return viewStart;
    return Math.round(viewStart + clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1) * viewSpan);
  }

  function timeAtPointer(event: React.PointerEvent<HTMLElement>): number {
    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect) return minimumMjd;
    const fraction = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return mjdAtFractionalIndex(viewStart + fraction * viewSpan);
  }

  function timelineWheel(event: React.WheelEvent<HTMLDivElement>) {
    event.preventDefault();
    const rect = timelineRef.current?.getBoundingClientRect();
    const fullSpan = maximum - minimum;
    if (!rect || fullSpan <= 0) return;
    const fraction = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const anchor = viewStart + fraction * viewSpan;
    const factor = Math.exp(Math.max(-1.2, Math.min(1.2, event.deltaY * 0.0015)));
    const minimumSpan = Math.min(TRACK_TIMELINE_MIN_ZOOM_SPAN, fullSpan);
    const nextSpan = Math.round(clamp(viewSpan * factor, minimumSpan, fullSpan));
    const nextStart = clamp(Math.round(anchor - fraction * nextSpan), minimum, maximum - nextSpan);
    setViewWindow([nextStart, nextStart + nextSpan]);
  }

  function resetTimelineZoom(event: React.MouseEvent<HTMLDivElement>) {
    if ((event.target as Element).closest(".track-anchor-diamond")) return;
    event.preventDefault();
    setViewWindow([minimum, maximum]);
  }

  function timelinePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || (event.target as Element).closest(".track-anchor-diamond")) return;
    timeDragRef.current = event.pointerId;
    event.currentTarget.setPointerCapture(event.pointerId);
    onTimeScrubStart();
    onTimeSelect(timeAtPointer(event));
    event.preventDefault();
  }

  function timelinePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (timeDragRef.current !== event.pointerId) return;
    onTimeSelect(timeAtPointer(event));
    event.preventDefault();
  }

  function timelinePointerUp(event: React.PointerEvent<HTMLDivElement>) {
    if (timeDragRef.current !== event.pointerId) return;
    timeDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    onTimeScrubEnd();
    event.preventDefault();
  }

  function timelinePointerCancel(event: React.PointerEvent<HTMLDivElement>) {
    if (timeDragRef.current !== event.pointerId) return;
    timeDragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    onTimeScrubEnd();
  }

  function startAnchorDrag(anchor: TrackAnchor, event: React.PointerEvent<HTMLButtonElement>) {
    anchorDragRef.current = { pointerId: event.pointerId, frameIndex: anchor.frameIndex, nextFrame: anchor.frameIndex };
    setPreviewFrame(anchor.frameIndex);
    onSelectAnchor(anchor);
    event.currentTarget.setPointerCapture(event.pointerId);
    event.stopPropagation();
  }

  function moveAnchorDrag(event: React.PointerEvent<HTMLButtonElement>) {
    const drag = anchorDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    drag.nextFrame = frameAtPointer(event);
    setPreviewFrame(drag.nextFrame);
    event.preventDefault();
  }

  function finishAnchorDrag(event: React.PointerEvent<HTMLButtonElement>) {
    const drag = anchorDragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    anchorDragRef.current = null;
    setPreviewFrame(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (drag.nextFrame !== drag.frameIndex) onMoveAnchor(drag.frameIndex, drag.nextFrame);
    event.stopPropagation();
  }

  function navigateAnchor(delta: number) {
    // Always cursor-relative: jump to the nearest key frame strictly
    // before/after the current cursor frame, regardless of selection.
    if (!selected?.anchors.length) return;
    const target = delta > 0
      ? selected.anchors.find((anchor) => anchor.frameIndex > currentFrame)
      : [...selected.anchors].reverse().find((anchor) => anchor.frameIndex < currentFrame);
    if (target) onSelectAnchor(target);
  }

  const selectedAnchorIndex = selected?.anchors.findIndex((anchor) => anchor.frameIndex === selectedAnchorFrame) ?? -1;
  const previousAnchorDisabled = !selected?.anchors.some((anchor) => anchor.frameIndex < currentFrame);
  const nextAnchorDisabled = !selected?.anchors.some((anchor) => anchor.frameIndex > currentFrame);
  const currentFrameHasAnchor = selected?.anchors.some((anchor) => anchor.frameIndex === currentFrame) ?? false;
  const addKeyFrameDisabled = !selected || currentFrameHasAnchor;
  const addKeyFrameTitle = !selected
    ? "Select a trajectory to add a key frame"
    : currentFrameHasAnchor
      ? "Key frame already at this frame"
      : "Add key frame at current frame (ghost position)";
  const selectedPointFrames = selected?.points.map((point) => point.frameIndex) ?? [];
  const stepDirection = direction === "backward" ? -1 : 1;
  const stepOrigin = selectedPointFrames.length
    ? stepDirection < 0 ? Math.min(...selectedPointFrames) : Math.max(...selectedPointFrames)
    : null;
  const canStepTrack = stepOrigin !== null && stepOrigin + stepDirection >= minimum && stepOrigin + stepDirection <= maximum;

  return (
    <section ref={floatingCard.cardRef} className="radio-alignment-card track-editor-card" role="dialog" aria-label="Track editor" tabIndex={-1} style={floatingCard.style}>
      <header ref={floatingCard.headerRef} className={`radio-alignment-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
        <div><strong>Track editor</strong><small>{tracks.length} track{tracks.length === 1 ? "" : "s"}</small></div>
        <button className="probe-close" type="button" aria-label="Close Track editor" onClick={onClose}>×</button>
      </header>

      <label className="tracking-source-picker">
        <span>Tracking source</span>
        <select aria-label="Tracking source" value={trackingSourceId} onChange={(event) => onTrackingSourceChange(event.target.value)}>
          {trackingSources.map((option) => <option key={option.sourceId} value={option.sourceId}>
            {option.label} · {option.sides.map((side) => side === "left" ? "Left" : "Right").join(" + ")}
          </option>)}
        </select>
      </label>

      <div className="track-editor-section-heading">Trajectories - {tracks.length}</div>
      <div className="track-editor-list" role="list" aria-label="Tracks">
        {tracks.length === 0 && <p className="track-editor-empty">Add a seed on the selected source to begin.</p>}
        {tracks.map((track) => {
          const stoppedFrame = track.points.at(-1)?.frameIndex;
          const trackState = track.state === "active" ? "active" : "stopped";
          return <div
            key={track.id}
            className={`track-editor-row ${track.id === selected?.id ? "selected" : ""}`}
            role="listitem"
            aria-selected={track.id === selected?.id}
            tabIndex={0}
            onClick={() => onSelectTrack(track.id)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelectTrack(track.id);
              }
            }}
          >
            <button className="track-select" type="button" aria-label={`Select ${track.label}`} onClick={() => onSelectTrack(track.id)}>
              <span className="track-color-dot" style={{ background: track.color }} />
            </button>
            <CommitInput value={track.label} ariaLabel={`Rename ${track.label}`} onCommit={(label) => { const next = label.trim(); if (next && next !== track.label) onRename(track.id, next); }} />
            <Pencil size={11} className="track-pencil" aria-hidden="true" />
            <input key={`${track.id}:${track.color}`} className="track-color-input" type="color" defaultValue={track.color} aria-label={`${track.label} color`} onBlur={(event) => { if (event.currentTarget.value !== track.color) onColor(track.id, event.currentTarget.value); }} />
            <button className="track-row-icon" type="button" aria-label={`${track.visible ? "Hide" : "Show"} ${track.label}`} onClick={() => onVisibility(track.id)}>{track.visible ? <Eye size={14} /> : <EyeOff size={14} />}</button>
            <span className={`track-state-badge ${track.state}`}>{track.state === "active" ? "active" : `${track.state === "stopped-edge" ? "edge" : "low conf"} @ ${stoppedFrame ?? "—"}`}</span>
            <button className="track-row-icon delete" type="button" aria-label={`Delete ${track.label}`} onClick={() => onDeleteTrack(track.id)}><Trash2 size={14} /></button>
            <span className="track-row-summary">{trackingSources.find((option) => option.sourceId === track.sourceId)?.label ?? track.sourceId} · {track.points.length} {track.points.length === 1 ? "point" : "points"} - {track.anchors.length} {track.anchors.length === 1 ? "key frame" : "key frames"} - {trackState}</span>
          </div>;
        })}
      </div>

      <div className="track-timeline-heading">
        <span style={{ color: selected?.color }}>Key frames - {selected?.label ?? "—"}</span>
        <div>
          <button type="button" aria-label="Previous frame" title="Previous frame" onClick={() => onStepFrame(-1)}><ChevronLeft size={14} /></button>
          <button type="button" aria-label="Next frame" title="Next frame" onClick={() => onStepFrame(1)}><ChevronRight size={14} /></button>
          <span className="track-row-summary" title="Selected key frame / total key frames">{selected?.anchors.length ? `${selectedAnchorIndex >= 0 ? selectedAnchorIndex + 1 : "–"}/${selected.anchors.length}` : "0/0"}</span>
          <button type="button" aria-label="Previous key frame ([)" title="Previous key frame ([)" onClick={() => navigateAnchor(-1)} disabled={previousAnchorDisabled}><ChevronFirst size={14} /></button>
          <button type="button" aria-label="Next key frame (])" title="Next key frame (])" onClick={() => navigateAnchor(1)} disabled={nextAnchorDisabled}><ChevronLast size={14} /></button>
          <button type="button" aria-label="Add key frame" title={addKeyFrameTitle} onClick={onAddKeyFrame} disabled={addKeyFrameDisabled}><Plus size={14} /></button>
          <button type="button" aria-label="Delete selected anchor" onClick={onDeleteAnchor} disabled={selectedAnchorFrame === null}><Trash2 size={12} /></button>
        </div>
      </div>
      <div
        ref={timelineRef}
        className="track-anchor-timeline"
        aria-label="Anchor timeline"
        style={{ borderColor: selected?.color }}
        onPointerDown={timelinePointerDown}
        onPointerMove={timelinePointerMove}
        onPointerUp={timelinePointerUp}
        onPointerCancel={timelinePointerCancel}
        onWheel={timelineWheel}
        onDoubleClick={resetTimelineZoom}
      >
        <span className="track-timeline-cursor" style={{ left: timePosition(currentMjd), background: selected?.color, boxShadow: selected?.color ? `0 0 5px ${selected.color}` : undefined }} />
        {selected?.anchors.filter((anchor) => anchor.frameIndex >= viewStart && anchor.frameIndex <= viewEnd).map((anchor) => {
          const displayFrame = selectedAnchorFrame === anchor.frameIndex && previewFrame !== null ? previewFrame : anchor.frameIndex;
          return <button
            key={`${selected.id}:${anchor.frameIndex}`}
            className={`track-anchor-diamond ${selectedAnchorFrame === anchor.frameIndex ? "selected" : ""}`}
            type="button"
            aria-label={`${selected.label} anchor frame ${anchor.frameIndex}`}
            title={`Frame ${displayFrame}`}
            style={{ left: position(displayFrame), background: selected.color }}
            onClick={() => onSelectAnchor(anchor)}
            onPointerDown={(event) => startAnchorDrag(anchor, event)}
            onPointerMove={moveAnchorDrag}
            onPointerUp={finishAnchorDrag}
            onPointerCancel={finishAnchorDrag}
          />;
        })}
      </div>
      <div className="track-timeline-range"><span>{viewStart}</span><span>cursor {currentFrame}</span><span>{viewEnd}</span></div>
      {selected && <div className="track-point-readout">
        <span>Frame <strong>{inspectedFrame}</strong></span>
        <span>Confidence <strong>{inspectedPoint ? inspectedPoint.confidence.toFixed(3) : "—"}</strong></span>
        <span>Speed <strong>{speedKmS === null ? "—" : `${speedKmS.toFixed(1)} km/s`}</strong></span>
      </div>}

      <div className="track-editor-controls">
        <select aria-label="Auto-track direction" value={direction} onChange={(event) => onDirectionChange(event.target.value as TrackDirection)}>
          <option value="forward">Forward</option>
          <option value="backward">Backward</option>
          <option value="both">Both</option>
        </select>
        <button className="button icon-button primary" type="button" aria-label={`Auto-track: ${selected?.label ?? "no trajectory selected"}`} title={`Auto-track: ${selected?.label ?? "—"}`} onClick={onAutoTrack} disabled={!selected || trackingActive}><Play size={14} /></button>
        <button className="button icon-button" type="button" aria-label={`Track one frame: ${selected?.label ?? "no trajectory selected"}`} title="Track one frame" onClick={onStepTrack} disabled={!selected || !canStepTrack || trackingActive}><StepForward size={14} /></button>
        <button className="button icon-button" type="button" aria-label="Stop auto-track" title="Stop" onClick={onStop} disabled={!trackingActive}><Square size={13} /></button>
      </div>
      <div className="track-editor-controls secondary">
        <button className="button icon-button" type="button" aria-label="Undo track mutation" title="Undo Cmd/Ctrl-Z" onClick={onUndo} disabled={!canUndo}><Undo2 size={14} /></button>
        <button className="button icon-button" type="button" aria-label="Redo track mutation" title="Redo Cmd/Ctrl-Shift-Z" onClick={onRedo} disabled={!canRedo}><Redo2 size={14} /></button>
        <button className="button icon-button" type="button" aria-label="Suggest tracking seeds" title="Suggest seeds" onClick={onSuggest} disabled={!canSuggest}><Sparkles size={14} /></button>
        <a className="button icon-button" href={exportHref} aria-label="Export tracking CSV" title="Export CSV"><Download size={14} /></a>
      </div>
      <p className="track-editor-hint">Each trajectory is tracked and edited independently.</p>
      <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
    </section>
  );
}

function displayedTrackSpeedKmS(track: SadTrack, frameIndex: number, affine: Affine): number | null {
  const points = [...track.points].sort((left, right) => left.frameIndex - right.frameIndex);
  const center = points.findIndex((point) => point.frameIndex === frameIndex);
  if (center < 2 || center > points.length - 3) return null;
  const stencil = points.slice(center - 2, center + 3);
  const centerMjd = stencil[2].mjd;
  const times = stencil.map((point) => (point.mjd - centerMjd) * 86400);
  const design = times.map((time) => [1, time, time * time]);
  const normal = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const rhs = [[0, 0], [0, 0], [0, 0]];
  stencil.forEach((point, row) => {
    const world = applyAffine([point.x, point.y], affine);
    for (let left = 0; left < 3; left += 1) {
      rhs[left][0] += design[row][left] * world[0];
      rhs[left][1] += design[row][left] * world[1];
      for (let right = 0; right < 3; right += 1) normal[left][right] += design[row][left] * design[row][right];
    }
  });
  const determinant = normal[0][0] * (normal[1][1] * normal[2][2] - normal[1][2] * normal[2][1])
    - normal[0][1] * (normal[1][0] * normal[2][2] - normal[1][2] * normal[2][0])
    + normal[0][2] * (normal[1][0] * normal[2][1] - normal[1][1] * normal[2][0]);
  if (!Number.isFinite(determinant) || Math.abs(determinant) < 1e-12) return null;
  const inverse = [
    [normal[1][1] * normal[2][2] - normal[1][2] * normal[2][1], normal[0][2] * normal[2][1] - normal[0][1] * normal[2][2], normal[0][1] * normal[1][2] - normal[0][2] * normal[1][1]],
    [normal[1][2] * normal[2][0] - normal[1][0] * normal[2][2], normal[0][0] * normal[2][2] - normal[0][2] * normal[2][0], normal[0][2] * normal[1][0] - normal[0][0] * normal[1][2]],
    [normal[1][0] * normal[2][1] - normal[1][1] * normal[2][0], normal[0][1] * normal[2][0] - normal[0][0] * normal[2][1], normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]]
  ].map((row) => row.map((value) => value / determinant));
  const vx = inverse[1].reduce((sum, value, index) => sum + value * rhs[index][0], 0);
  const vy = inverse[1].reduce((sum, value, index) => sum + value * rhs[index][1], 0);
  return Number.isFinite(vx) && Number.isFinite(vy) ? Math.hypot(vx, vy) * 725 : null;
}

function RecordingOptionsCard({
  options,
  predictedFormat,
  formatWarning,
  onChange,
  onStart,
  onClose,
  cardSize,
  onCardSizeChange
}: {
  options: RecordingOptions;
  predictedFormat: RecordingFormat;
  formatWarning: string;
  onChange: (options: RecordingOptions) => void;
  onStart: (options: RecordingOptions) => void;
  onClose: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const floatingCard = useDraggableFloatingCard({ minWidth: 320, minHeight: 280, size: cardSize, onSizeChange: onCardSizeChange });
  const patch = (value: Partial<RecordingOptions>) => onChange({ ...options, ...value });
  const supported = typeof MediaRecorder !== "undefined" && typeof HTMLCanvasElement.prototype.captureStream === "function";
  return (
    <section ref={floatingCard.cardRef} className="radio-alignment-card recording-options-card" role="dialog" aria-label="Video recording options" style={floatingCard.style}>
      <header ref={floatingCard.headerRef} className={`radio-alignment-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
        <div><strong>Record video</strong><small>Canvas capture</small></div>
        <button className="probe-close" type="button" aria-label="Close recording options" onClick={onClose}>×</button>
      </header>
      <div className="recording-options-grid">
        <label>Source
          <select value={options.source} onChange={(event) => {
            const source = event.target.value as RecordingSource;
            patch({ source, burnTimestamp: source === "left" || source === "right" });
          }}>
            <option value="left">Left panel</option>
            <option value="right">Right panel</option>
            <option value="both">Both side-by-side</option>
            <option value="workspace">Workspace (spectrogram + panels)</option>
          </select>
        </label>
        <label>Range
          <select value={options.range} onChange={(event) => patch({ range: event.target.value as RecordingRange })}>
            <option value="visible">Spectrogram visible window</option>
            <option value="master">Master start-end range</option>
          </select>
        </label>
        <label>Output fps
          <CommitInput type="number" min={1} max={60} scrubStep={1} ariaLabel="Recording output fps" value={options.fps} onCommit={(value) => patch({ fps: clamp(Math.round(Number(value) || 20), 1, 60) })} />
        </label>
        <label>Resolution
          <select value={options.resolution} onChange={(event) => patch({ resolution: event.target.value as RecordingResolution })}>
            <option value="native">Native (as displayed)</option>
            <option value="2x">2x native</option>
          </select>
        </label>
      </div>
      <label className="recording-checkbox">
        <input type="checkbox" checked={options.burnTimestamp} onChange={(event) => patch({ burnTimestamp: event.target.checked })} />
        Burn UTC timestamp
      </label>
      <label>Format
        <select value={options.formatChoice} onChange={(event) => patch({ formatChoice: event.target.value as RecordingFormatChoice })}>
          <option value="auto">Auto (recommended)</option>
          <option value="mp4">MP4 (H.264)</option>
          <option value="webm">WebM (VP9)</option>
        </select>
      </label>
      {options.formatChoice === "auto" && (
        <div className="recording-format-line" aria-label={`Auto format currently predicts ${predictedFormat.label}`}>Auto currently predicts: <strong>{predictedFormat.label}</strong></div>
      )}
      {formatWarning && (
        <div className="warning-text" role="alert">{formatWarning}</div>
      )}
      <button className="button primary recording-start-button" type="button" onClick={() => onStart(options)} disabled={!supported || Boolean(formatWarning)}>Start</button>
      <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
    </section>
  );
}

function SpectrogramDisplayCard({
  display,
  normalization,
  frequencyScale,
  frequencyInverted,
  frequencyRange,
  onDisplayChange,
  onNormalizationChange,
  onFrequencyScaleChange,
  onFrequencyInvertedChange,
  onFrequencyRangeChange,
  smoothPlayback,
  onSmoothPlaybackChange,
  showCacheCoverage,
  onShowCacheCoverageChange,
  frameCacheGb,
  onFrameCacheGbChange,
  onClose,
  cardSize,
  onCardSizeChange
}: {
  display: DisplayState;
  normalization: SpectrogramNormalization;
  frequencyScale: FrequencyScaleMode;
  frequencyInverted: boolean;
  frequencyRange: FrequencyRangeState;
  onDisplayChange: (display: DisplayState) => void;
  onNormalizationChange: (normalization: SpectrogramNormalization) => void;
  onFrequencyScaleChange: (scale: FrequencyScaleMode) => void;
  onFrequencyInvertedChange: (inverted: boolean) => void;
  onFrequencyRangeChange: (range: FrequencyRangeState) => void;
  smoothPlayback: boolean;
  onSmoothPlaybackChange: (value: boolean) => void;
  showCacheCoverage: boolean;
  onShowCacheCoverageChange: (value: boolean) => void;
  frameCacheGb: number;
  onFrameCacheGbChange: (value: number) => void;
  onClose: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const floatingCard = useDraggableFloatingCard({ minWidth: 420, minHeight: 300, size: cardSize, onSizeChange: onCardSizeChange });
  const presetBounds = normalization === "divide"
    ? { min: 0, max: 3 }
    : normalization === "subtract" ? { min: -150, max: 150 } : { min: 0, max: 150 };
  const currentMin = numberValue(display.vmin, presetBounds.min);
  const currentMax = numberValue(display.vmax, presetBounds.max);
  const rangeBounds = {
    min: Math.min(presetBounds.min, currentMin, currentMax),
    max: Math.max(presetBounds.max, currentMin, currentMax)
  };
  const normalizationTooltip = "flattens per-channel background (RFI bands, quiet Sun)";
  const patchDisplay = (key: "vmin" | "vmax" | "cmap" | "scale", value: string) => {
    onDisplayChange({ ...display, [key]: value });
  };
  const deviceMemGb = deviceMemoryGb();
  const frameCacheExceedsHalfDeviceMemory = deviceMemGb !== undefined && frameCacheGb > deviceMemGb / 2;

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  return (
    <section ref={floatingCard.cardRef} className="radio-alignment-card spectrogram-display-card" role="dialog" aria-label="Spectrogram display" style={floatingCard.style}>
      <header ref={floatingCard.headerRef} className={`radio-alignment-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
        <div><strong>Spectrogram display</strong></div>
        <button className="probe-close" type="button" aria-label="Close spectrogram display" onClick={onClose}>×</button>
      </header>
      <div className="layer-inspector-grid spectrogram-display-grid">
        <div className="track-editor-section-heading">Intensity</div>
        <InspectorRow label="Scale" title="Color scale" control={<div className="mode-row spectrogram-two-mode-row">
          <ModeButton active={display.scale === "linear"} title="Linear color scale" onClick={() => patchDisplay("scale", "linear")}>Linear</ModeButton>
          <ModeButton active={display.scale === "log"} title="Logarithmic color scale" onClick={() => patchDisplay("scale", "log")}>Log</ModeButton>
        </div>} />
        <DualRangeRow
          bounds={rangeBounds}
          minValue={display.vmin}
          maxValue={display.vmax}
          disabled={false}
          onMinCommit={(value) => patchDisplay("vmin", value)}
          onMaxCommit={(value) => patchDisplay("vmax", value)}
        />
        <InspectorRow label="Colormap" title="Colormap" control={<select aria-label="Spectrogram colormap" value={display.cmap} onChange={(event) => patchDisplay("cmap", event.target.value)}>
          {COLORMAP_OPTIONS.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
        </select>} />

        <div className="track-editor-section-heading">Frequency axis</div>
        <InspectorRow label="Scale" title="Frequency axis" control={<div className="mode-row">
          <ModeButton active={frequencyScale === "linear"} title="Linear frequency axis" onClick={() => onFrequencyScaleChange("linear")}>Linear</ModeButton>
          <ModeButton active={frequencyScale === "log"} title="Logarithmic frequency axis" onClick={() => onFrequencyScaleChange("log")}>Log</ModeButton>
          <ModeButton active={frequencyInverted} title="Invert frequency axis" onClick={() => onFrequencyInvertedChange(!frequencyInverted)}>Invert</ModeButton>
        </div>} />
        <InspectorRow label="Range [GHz]" title="Frequency range [GHz]" control={
          <div className="spectrogram-freq-range-control">
            <CommitInput type="number" ariaLabel="Minimum frequency [GHz]" title="Minimum frequency [GHz]" value={frequencyRange.min} onCommit={(value) => onFrequencyRangeChange({ ...frequencyRange, min: value })} />
            <span className="spectrogram-freq-range-arrow" aria-hidden="true">→</span>
            <CommitInput type="number" ariaLabel="Maximum frequency [GHz]" title="Maximum frequency [GHz]" value={frequencyRange.max} onCommit={(value) => onFrequencyRangeChange({ ...frequencyRange, max: value })} />
          </div>
        } />
        <InspectorRow label="Normalization" title={normalizationTooltip} control={<div className="mode-row spectrogram-normalization-row">
          <ModeButton active={normalization === "none"} title={`None — ${normalizationTooltip}`} onClick={() => onNormalizationChange("none")}>None</ModeButton>
          <ModeButton active={normalization === "divide"} title={`Divide by row median — ${normalizationTooltip}`} onClick={() => onNormalizationChange("divide")}>Divide</ModeButton>
          <ModeButton active={normalization === "subtract"} title={`Subtract row median — ${normalizationTooltip}`} onClick={() => onNormalizationChange("subtract")}>Subtract</ModeButton>
        </div>} />

        <div className="track-editor-section-heading">Performance</div>
        <InspectorRow label={<span className="spectrogram-row-label-stack"><span>Smooth playback</span><span className="spectrogram-row-label-subtitle">half resolution during motion</span></span>} title="Halve the requested frame resolution while the timeline is moving (playback or scrubbing), then swap in the full-res frame once it settles" control={
          <label className="spectrogram-toggle-row"><input type="checkbox" aria-label="Smooth playback (half resolution during motion)" checked={smoothPlayback} onChange={(event) => onSmoothPlaybackChange(event.target.checked)} /></label>
        } />
        <InspectorRow label="Frame cache [GB]" title="Decoded-frame bitmap cache byte budget. Lowering it evicts cached frames immediately; playback and prefetch behavior are otherwise unaffected." control={
          <div className="frame-cache-budget-control">
            <CommitInput
              type="number"
              min={FRAME_CACHE_GB_MIN}
              max={FRAME_CACHE_GB_MAX}
              scrubStep={0.05}
              ariaLabel="Frame cache budget in gigabytes"
              title="Frame cache budget [GB]"
              value={frameCacheGb.toFixed(1)}
              onCommit={(value) => onFrameCacheGbChange(numberValue(value, frameCacheGb))}
            />
            {frameCacheExceedsHalfDeviceMemory && <div className="frame-cache-warning">exceeds half of device memory</div>}
          </div>
        } />
        <InspectorRow label="Cache coverage bar" title="Green strip along the top of the spectrogram time axis showing which columns are already decoded and cached at full resolution" control={
          <label className="spectrogram-toggle-row"><input type="checkbox" aria-label="Cache coverage bar" checked={showCacheCoverage} onChange={(event) => onShowCacheCoverageChange(event.target.checked)} /></label>
        } />
      </div>
      <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
    </section>
  );
}

function RadioAlignmentCard({
  freqGhz,
  offsets,
  selectedChannels,
  lassoArmed,
  palette,
  spwGroups,
  onSelectionChange,
  onChange,
  onZeroSelected,
  onZeroAll,
  onMaskSelected,
  onToggleMask,
  onExport,
  onImport,
  onToggleLasso,
  onClose,
  cardSize,
  onCardSizeChange
}: {
  freqGhz: number[];
  offsets: ChannelOffsets;
  selectedChannels: number[];
  lassoArmed: boolean;
  palette: ContourColormap;
  spwGroups: SpwGroup[];
  onSelectionChange: (channels: number[]) => void;
  onChange: (offsets: ChannelOffsets) => void;
  onZeroSelected: () => void;
  onZeroAll: () => void;
  onMaskSelected: (masked: boolean) => void;
  onToggleMask: (index: number) => void;
  onExport: () => void;
  onImport: (csv: string) => void;
  onToggleLasso: () => void;
  onClose: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const dxCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dyCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const dragRef = useRef<{ axis: "dx" | "dy"; start: number; last: number; moved: boolean; altKey: boolean } | null>(null);
  const [box, setBox] = useState<{ axis: "dx" | "dy"; start: number; end: number } | null>(null);
  const [xAxisMode, setXAxisMode] = useState<AlignmentXAxisMode>("channel");
  const floatingCard = useDraggableFloatingCard({ minWidth: 360, minHeight: 330, size: cardSize, onSizeChange: onCardSizeChange });

  useEffect(() => {
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  function drawPlot(canvas: HTMLCanvasElement | null, values: number[], axis: "dx" | "dy") {
    if (!canvas) return;
    const width = 460;
    const height = 130;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const left = 36;
    const right = 8;
    const top = 10;
    const bottom = 22;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const finite = values.filter(Number.isFinite);
    const limit = Math.max(5, ...(finite.length ? finite.map((value) => Math.abs(value)) : [5]));
    const [frequencyMin, frequencyMax] = frequencyBounds(freqGhz);
    const xValue = (index: number) => xAxisMode === "frequency" ? Number(freqGhz[index]) : index;
    const xMin = xAxisMode === "frequency" ? frequencyMin : 0;
    const xMax = xAxisMode === "frequency" ? frequencyMax : Math.max(0, values.length - 1);
    const xFor = (index: number) => left + (xMax > xMin ? (xValue(index) - xMin) / (xMax - xMin) : 0.5) * plotWidth;
    const yFor = (value: number) => top + (1 - (clamp(value, -limit, limit) + limit) / (2 * limit)) * plotHeight;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = "#3b4046";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, yFor(0));
    ctx.lineTo(width - right, yFor(0));
    ctx.stroke();
    ctx.fillStyle = "#979ea6";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(`+${limit.toFixed(1)}″`, width - right - 2, top + 4);
    ctx.fillText(`−${limit.toFixed(1)}″`, width - right - 2, height - bottom + 4);
    ctx.textAlign = "left";
    ctx.fillText(axis, 7, top + 13);
    ctx.textAlign = "center";
    ctx.fillText(xAxisMode === "frequency" ? "GHz" : "channel", width / 2, height - 4);
    if (xAxisMode === "frequency") {
      ctx.textAlign = "left";
      ctx.fillText(frequencyMin.toFixed(2), left, height - 4);
      ctx.textAlign = "right";
      ctx.fillText(frequencyMax.toFixed(2), width - right, height - 4);
    }
    if (box?.axis === axis) {
      const x0 = Math.min(xFor(box.start), xFor(box.end));
      const x1 = Math.max(xFor(box.start), xFor(box.end));
      ctx.fillStyle = "rgba(86, 199, 217, 0.16)";
      ctx.fillRect(x0, top, Math.max(1, x1 - x0), plotHeight);
      ctx.strokeStyle = "#56c7d9";
      ctx.strokeRect(x0, top, Math.max(1, x1 - x0), plotHeight);
    }
    values.forEach((value, index) => {
      const x = xFor(index);
      const y = yFor(Number.isFinite(value) ? value : 0);
      const selected = selectedChannels.includes(index);
      const masked = offsets.masked[index] ?? false;
      const color = sampleColormap(palette, index, Math.max(1, values.length), "frequency");
      if (masked) {
        ctx.globalAlpha = 0.5;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, selected ? 4.5 : 3.8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, selected ? 3.2 : 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      if (selected) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, 5.4, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
  }

  useEffect(() => {
    drawPlot(dxCanvasRef.current, offsets.dx, "dx");
    drawPlot(dyCanvasRef.current, offsets.dy, "dy");
  }, [box, freqGhz, offsets, palette, selectedChannels, xAxisMode]);

  function indexFor(event: React.PointerEvent<HTMLCanvasElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    const fraction = clamp((event.clientX - rect.left - 36) / Math.max(1, rect.width - 44), 0, 1);
    if (xAxisMode === "channel") return Math.round(fraction * Math.max(0, freqGhz.length - 1));
    const [frequencyMin, frequencyMax] = frequencyBounds(freqGhz);
    const target = frequencyMin + fraction * (frequencyMax - frequencyMin);
    return freqGhz.reduce((nearest, frequency, index) => (
      Math.abs(frequency - target) < Math.abs(freqGhz[nearest] - target) ? index : nearest
    ), 0);
  }

  function pointerDown(axis: "dx" | "dy", event: React.PointerEvent<HTMLCanvasElement>) {
    if (event.button !== 0 || !freqGhz.length) return;
    const index = indexFor(event);
    dragRef.current = { axis, start: index, last: index, moved: false, altKey: event.altKey };
    setBox({ axis, start: index, end: index });
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function pointerMove(axis: "dx" | "dy", event: React.PointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag || drag.axis !== axis) return;
    const index = indexFor(event);
    drag.last = index;
    if (index !== drag.start) drag.moved = true;
    setBox({ axis, start: drag.start, end: index });
  }

  function pointerUp(axis: "dx" | "dy", event: React.PointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag || drag.axis !== axis) return;
    dragRef.current = null;
    const end = indexFor(event);
    if (!drag.moved && (drag.altKey || event.altKey)) {
      onToggleMask(end);
    } else if (drag.moved) {
      const start = Math.min(drag.start, end);
      const stop = Math.max(drag.start, end);
      onSelectionChange(Array.from({ length: stop - start + 1 }, (_, index) => start + index));
    } else {
      const next = selectedChannels.includes(end)
        ? selectedChannels.filter((index) => index !== end)
        : [...selectedChannels, end];
      onSelectionChange(next.sort((left, right) => left - right));
    }
    setBox(null);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  function commitAxis(axis: "dx" | "dy", value: string) {
    const numeric = Number(value);
    if (!selectedChannels.length || !Number.isFinite(numeric)) return;
    const next = { dx: [...offsets.dx], dy: [...offsets.dy], masked: [...offsets.masked] };
    selectedChannels.forEach((index) => { next[axis][index] = numeric; });
    onChange(next);
  }

  const dxValue = selectedCommonValue(offsets.dx, selectedChannels);
  const dyValue = selectedCommonValue(offsets.dy, selectedChannels);
  const maskedCount = offsets.masked.filter(Boolean).length;

  return (
    <section ref={floatingCard.cardRef} className="radio-alignment-card" role="dialog" aria-label="Channel inspector" style={floatingCard.style}>
      <header ref={floatingCard.headerRef} className={`radio-alignment-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
        <div><strong>Channel inspector</strong><small>{freqGhz.length} channels · arcsec{maskedCount ? ` · ${maskedCount} masked` : ""}</small></div>
        <div className="radio-alignment-axis-toggle" role="group" aria-label="Plot x-axis mode">
          <button className={`button mode-button ${xAxisMode === "channel" ? "active" : ""}`} type="button" title="Channel index" aria-pressed={xAxisMode === "channel"} onClick={() => setXAxisMode("channel")}>Ch</button>
          <button className={`button mode-button ${xAxisMode === "frequency" ? "active" : ""}`} type="button" title="Frequency [GHz]" aria-pressed={xAxisMode === "frequency"} onClick={() => setXAxisMode("frequency")}>GHz</button>
        </div>
        <button className="probe-close" type="button" aria-label="Close radio alignment inspector" onClick={onClose}>×</button>
      </header>
      <div className="radio-alignment-plots">
        <canvas ref={dxCanvasRef} className="radio-alignment-plot" aria-label="X offset by channel" onPointerDown={(event) => pointerDown("dx", event)} onPointerMove={(event) => pointerMove("dx", event)} onPointerUp={(event) => pointerUp("dx", event)} onPointerCancel={() => { dragRef.current = null; setBox(null); }} />
        <canvas ref={dyCanvasRef} className="radio-alignment-plot" aria-label="Y offset by channel" onPointerDown={(event) => pointerDown("dy", event)} onPointerMove={(event) => pointerMove("dy", event)} onPointerUp={(event) => pointerUp("dy", event)} onPointerCancel={() => { dragRef.current = null; setBox(null); }} />
      </div>
      <div className="radio-alignment-selection">
        <span>{selectedChannels.length ? `${selectedChannels.length} selected` : "Select channels"}</span>
        <button className="button mode-button" type="button" onClick={() => onSelectionChange(Array.from({ length: freqGhz.length }, (_, index) => index))}>All</button>
        <button className="button mode-button" type="button" onClick={() => onSelectionChange([])}>Clear</button>
        <button className={`button mode-button radio-alignment-lasso-button ${lassoArmed ? "armed" : ""}`} type="button" aria-pressed={lassoArmed} onClick={onToggleLasso}><LassoSelect size={13} aria-hidden="true" /> {lassoArmed ? "Lasso armed" : "Lasso select"}</button>
        {spwGroups.map((group) => <button className="button mode-button" type="button" key={group.id} onClick={() => onSelectionChange(Array.from({ length: group.end - group.start + 1 }, (_, index) => group.start + index))}>spw{group.id} · {group.start}–{group.end}</button>)}
      </div>
      <div className="two-col radio-alignment-finetune">
        <label>dx [arcsec]<CommitInput type="number" scrubStep={0.25} ariaLabel="Selected channel dx in arcseconds" value={dxValue} disabled={!selectedChannels.length} onCommit={(value) => commitAxis("dx", value)} /></label>
        <label>dy [arcsec]<CommitInput type="number" scrubStep={0.25} ariaLabel="Selected channel dy in arcseconds" value={dyValue} disabled={!selectedChannels.length} onCommit={(value) => commitAxis("dy", value)} /></label>
      </div>
      <div className="radio-alignment-actions">
        <button className="button" type="button" onClick={onZeroSelected} disabled={!selectedChannels.length}>Zero selected</button>
        <button className="button" type="button" onClick={() => onMaskSelected(true)} disabled={!selectedChannels.length}>Mask selected</button>
        <button className="button" type="button" onClick={() => onMaskSelected(false)} disabled={!selectedChannels.length}>Unmask selected</button>
        <button className="button" type="button" onClick={onZeroAll}>Zero all</button>
        <button className="button" type="button" onClick={onExport}>Export CSV</button>
        <button className="button" type="button" onClick={() => inputRef.current?.click()}>Import CSV</button>
        <input ref={inputRef} className="hidden-input" type="file" accept="text/csv,.csv" onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; if (file) void file.text().then(onImport); }} />
      </div>
      <small className="radio-alignment-help">Lasso select channels in either image. Alt+click toggles one mask; masked channels stay selectable.</small>
      <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
    </section>
  );
}

function PixelProbeCard({
  layerLabel,
  pixel,
  patchRadius,
  data,
  loading,
  error,
  currentMjd,
  onClose,
  onRefresh,
  onApply,
  cardSize,
  onCardSizeChange
}: {
  layerLabel: string;
  pixel: [number, number];
  patchRadius: number;
  data: PixelProbeResponse | null;
  loading: boolean;
  error: string;
  currentMjd: number;
  onClose: () => void;
  onRefresh: () => void;
  onApply: () => void;
  cardSize?: FloatingCardSize;
  onCardSizeChange?: (size: FloatingCardSize) => void;
}) {
  const chartRef = useRef<HTMLCanvasElement | null>(null);
  const floatingCard = useDraggableFloatingCard({ minWidth: 320, minHeight: 280, size: cardSize, onSizeChange: onCardSizeChange });

  useEffect(() => {
    const canvas = chartRef.current;
    if (!canvas) return;
    const width = 460;
    const height = 160;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0d1015";
    ctx.fillRect(0, 0, width, height);
    const left = 42;
    const right = 8;
    const top = 10;
    const bottom = 30;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    const times = data?.mjd ?? [];
    const raw = data?.raw ?? [];
    const smoothed = data?.smoothed ?? [];
    const finite = [...raw, ...smoothed].filter((value): value is number => value !== null && Number.isFinite(value));
    const yMin = finite.length ? Math.min(...finite) : 0;
    const yMax = finite.length ? Math.max(...finite) : 1;
    const ySpan = Math.max(1e-12, yMax - yMin);
    const paddedMin = yMin - ySpan * 0.05;
    const paddedMax = yMax + ySpan * 0.05;
    const start = times[0] ?? 0;
    const end = times[times.length - 1] ?? start + 1;
    const xSpan = Math.max(1e-12, end - start);
    const xPosition = (value: number) => left + ((value - start) / xSpan) * plotWidth;
    const yPosition = (value: number) => top + (1 - (value - paddedMin) / Math.max(1e-12, paddedMax - paddedMin)) * plotHeight;

    ctx.strokeStyle = "#4a5057";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top);
    ctx.lineTo(left, top + plotHeight);
    ctx.lineTo(left + plotWidth, top + plotHeight);
    ctx.stroke();
    if (!data) {
      ctx.fillStyle = "#979ea6";
      ctx.font = "11px ui-sans-serif";
      ctx.fillText(loading ? "Loading…" : error || "No samples", left + 8, top + plotHeight / 2);
      return;
    }
    const drawSeries = (values: (number | null)[], color: string, lineWidth: number) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      let drawing = false;
      values.forEach((value, index) => {
        if (value === null || !Number.isFinite(value) || !Number.isFinite(times[index])) {
          drawing = false;
          return;
        }
        const pointX = xPosition(times[index]);
        const pointY = yPosition(value);
        if (!drawing) ctx.moveTo(pointX, pointY);
        else ctx.lineTo(pointX, pointY);
        drawing = true;
      });
      ctx.stroke();
    };
    drawSeries(raw, "#87919b", 1);
    if (data.smoothed) drawSeries(data.smoothed, "#56c7d9", 1.8);
    if (Number.isFinite(currentMjd) && currentMjd >= start && currentMjd <= end) {
      const cursorX = xPosition(currentMjd);
      ctx.strokeStyle = "#ee65be";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cursorX, top);
      ctx.lineTo(cursorX, top + plotHeight);
      ctx.stroke();
    }
    ctx.fillStyle = "#979ea6";
    ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.fillText(formatProbeNumber(yMax), left - 5, top + 3);
    ctx.fillText(formatProbeNumber(yMin), left - 5, top + plotHeight);
    ctx.textAlign = "left";
    ctx.fillText(probeUtcLabel(start), left, height - 8);
    ctx.textAlign = "right";
    ctx.fillText(probeUtcLabel(end), left + plotWidth, height - 8);
  }, [currentMjd, data, error, loading]);

  const rawStats = data?.stats.raw;
  const smoothedStats = data?.stats.smoothed;
  const valueLabel = (value: number | null | undefined) => value === null || value === undefined || !Number.isFinite(value) ? "—" : formatProbeNumber(value);
  return (
    <aside ref={floatingCard.cardRef} className="pixel-probe-card" aria-label="Pixel light-curve probe" style={floatingCard.style}>
      <header ref={floatingCard.headerRef} className={`pixel-probe-header ${floatingCard.dragging ? "is-dragging" : ""}`} {...floatingCard.headerHandlers}>
        <strong>{layerLabel} @ ({Math.round(pixel[0])}, {Math.round(pixel[1])})</strong>
        <span>{2 * patchRadius + 1}×{2 * patchRadius + 1} patch</span>
        <button className="probe-close" type="button" aria-label="Close pixel probe" onClick={onClose}>×</button>
      </header>
      <canvas ref={chartRef} className="pixel-probe-chart" aria-label="Raw and smoothed pixel light curves" />
      {error && <div className="pixel-probe-error" role="alert">{error}</div>}
      <div className="pixel-probe-stats">
        <span>raw p1/p99 <strong>{valueLabel(rawStats?.p1)} / {valueLabel(rawStats?.p99)}</strong></span>
        <span>smoothed p1/p99 <strong>{valueLabel(smoothedStats?.p1)} / {valueLabel(smoothedStats?.p99)}</strong></span>
      </div>
      <div className="pixel-probe-actions">
        <button className="button primary" type="button" onClick={onApply} disabled={!data || loading || (rawStats?.p1 == null && smoothedStats?.p1 == null)}>Apply smoothed p1/p99 as vmin/vmax</button>
        <button className="button" type="button" onClick={onRefresh} disabled={loading}>Refresh</button>
      </div>
      {data && <small className="pixel-probe-meta">{data.nTotal} native samples · stride {data.stride} · median cadence {data.cadenceSeconds.toFixed(2)} s{loading ? " · updating…" : ""}</small>}
      <div className="floating-card-resize-handle" {...floatingCard.resizeHandleHandlers} aria-hidden="true" />
    </aside>
  );
}

function formatProbeNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 || Math.abs(value) < 0.01 ? value.toExponential(3) : value.toFixed(3);
}

function probeUtcLabel(value: number): string {
  const text = mjdToUtc(value);
  return text.length > 22 ? text.slice(11, 22) : text;
}

function drawRadioColorbar(ctx: CanvasRenderingContext2D, width: number, height: number, colorbar: RadioColorbar) {
  const barHeight = Math.min(150, Math.max(86, height * 0.3));
  const boxWidth = 76;
  const boxHeight = barHeight + 34;
  const boxX = Math.max(8, width - boxWidth - 12);
  const boxY = Math.max(8, height - boxHeight - 12);
  const barX = boxX + 10;
  const barY = boxY + 23;
  const barWidth = 11;
  const stops = colormapStops(colorbar.cmap, "frequency");
  const gradient = ctx.createLinearGradient(0, barY, 0, barY + barHeight);
  stops.forEach((color, index) => gradient.addColorStop(1 - index / Math.max(1, stops.length - 1), color));

  ctx.save();
  ctx.fillStyle = "rgb(5 9 12 / 82%)";
  ctx.strokeStyle = "rgb(91 108 118 / 58%)";
  ctx.lineWidth = 1;
  ctx.fillRect(boxX, boxY, boxWidth, boxHeight);
  ctx.strokeRect(boxX + 0.5, boxY + 0.5, boxWidth - 1, boxHeight - 1);
  ctx.fillStyle = "#c8d0d8";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "middle";
  ctx.fillText("GHz", barX, boxY + 12);
  ctx.fillStyle = gradient;
  ctx.fillRect(barX, barY, barWidth, barHeight);
  ctx.strokeStyle = "#87919b";
  ctx.strokeRect(barX + 0.5, barY + 0.5, barWidth - 1, barHeight - 1);
  ctx.fillStyle = "#d7dde3";
  ctx.fillText(colorbar.maxGhz.toFixed(2), barX + 18, barY + 4);
  ctx.fillText(((colorbar.minGhz + colorbar.maxGhz) / 2).toFixed(2), barX + 18, barY + barHeight / 2);
  ctx.fillText(colorbar.minGhz.toFixed(2), barX + 18, barY + barHeight - 4);
  ctx.restore();
}

function drawPath(ctx: CanvasRenderingContext2D, points: [number, number][], t: PanelTransform, color: string, closed: boolean) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.setLineDash(closed ? [] : [5, 4]);
  ctx.beginPath();
  const first = pixelToCanvas(points[0], t);
  ctx.moveTo(first[0], first[1]);
  for (const point of points.slice(1)) {
    const [x, y] = pixelToCanvas(point, t);
    ctx.lineTo(x, y);
  }
  if (closed) ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function drawWorldPath(ctx: CanvasRenderingContext2D, points: [number, number][], t: PanelTransform, color: string, closed: boolean, dashed = false) {
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.setLineDash(dashed || !closed ? [5, 4] : []);
  ctx.beginPath();
  const first = worldToCanvas(points[0], t);
  ctx.moveTo(first[0], first[1]);
  for (const point of points.slice(1)) {
    const [x, y] = worldToCanvas(point, t);
    ctx.lineTo(x, y);
  }
  if (closed) ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function drawSlitWorldPath(
  ctx: CanvasRenderingContext2D,
  slit: SlitDefinition,
  transform: PanelTransform,
  selected: boolean,
  drawSwath: boolean = true
) {
  if (slit.curveArcsec.length < 2) return;
  const canvasPoints = slit.curveArcsec.map((point) => worldToCanvas(point, transform));
  const trace = (lineWidth: number, strokeStyle: string, alpha: number) => {
    ctx.save();
    ctx.strokeStyle = strokeStyle;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    canvasPoints.forEach(([x, y], index) => {
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();
  };
  // FEATURE 2 (width edge lines): the extraction swath's actual half-width,
  // in arcsec, is widthArcsec/2 DIRECTLY - no per-source pixel-scale
  // multiplication (USER DESIGN DECISION: width is now a fixed physical
  // arcsec quantity, not bound-source pixels - see
  // SlitDefinition.widthArcsec) - so an origin and any linked twin, which
  // inherit widthArcsec verbatim (resolveSlitGeometry), draw the IDENTICAL
  // corridor regardless of which source each is bound to. `drawSwath` lets
  // the caller suppress this fill/edge pass for all but one member of such
  // an overlapping group (see the draw loop's DOUBLE-SWATH SUPPRESSION
  // comment) - the spine/arrowhead below are drawn unconditionally either
  // way. Computed as an offset polyline directly in world/arcsec space
  // (tangent/normal from neighboring curveArcsec points) and only then
  // projected to canvas, rather than offsetting in canvas-pixel space using
  // the PANEL's own affine as the old translucent-only band did - that
  // meant a slit's rendered width could silently depend on which panel it
  // was drawn on rather than what it is bound to. Same physical swath now
  // backs both the (kept, slightly stronger) translucent fill and the new
  // dashed edge lines.
  if (slit.widthArcsec > 0 && drawSwath) {
    const halfWidthArcsec = Math.max(1e-6, slit.widthArcsec / 2);
    const left: [number, number][] = [];
    const right: [number, number][] = [];
    slit.curveArcsec.forEach((point, index) => {
      const before = slit.curveArcsec[Math.max(0, index - 1)];
      const after = slit.curveArcsec[Math.min(slit.curveArcsec.length - 1, index + 1)];
      const dx = after[0] - before[0];
      const dy = after[1] - before[1];
      const length = Math.max(1e-9, Math.hypot(dx, dy));
      const nx = -dy / length;
      const ny = dx / length;
      left.push([point[0] + nx * halfWidthArcsec, point[1] + ny * halfWidthArcsec]);
      right.push([point[0] - nx * halfWidthArcsec, point[1] - ny * halfWidthArcsec]);
    });
    const leftCanvas = left.map((point) => worldToCanvas(point, transform));
    const rightCanvas = right.map((point) => worldToCanvas(point, transform));
    ctx.save();
    ctx.fillStyle = slit.color;
    ctx.globalAlpha = selected ? 0.26 : 0.18;
    ctx.beginPath();
    leftCanvas.forEach(([x, y], index) => index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
    [...rightCanvas].reverse().forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    const traceEdge = (points: [number, number][]) => {
      ctx.save();
      ctx.strokeStyle = slit.color;
      ctx.globalAlpha = slit.visible ? 0.85 : 0.3;
      ctx.lineWidth = 1.2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      points.forEach(([x, y], index) => index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y));
      ctx.stroke();
      ctx.restore();
    };
    traceEdge(leftCanvas);
    traceEdge(rightCanvas);
  }
  trace(selected ? 2.4 : 1.7, slit.color, slit.visible ? 0.98 : 0.35);
  const origin = worldToCanvas(slit.curveArcsec[0], transform);
  const far = worldToCanvas(slit.curveArcsec.at(-1)!, transform);
  const previous = worldToCanvas(slit.curveArcsec.at(-2)!, transform);
  const dx = far[0] - previous[0];
  const dy = far[1] - previous[1];
  const length = Math.max(1e-9, Math.hypot(dx, dy));
  const ux = dx / length;
  const uy = dy / length;
  const arrowLength = selected ? 9 : 8;
  const arrowHalfWidth = selected ? 4.5 : 4;
  ctx.save();
  ctx.globalAlpha = slit.visible ? 1 : 0.4;
  ctx.fillStyle = slit.color;
  ctx.strokeStyle = "rgba(5, 7, 10, 0.82)";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.arc(origin[0], origin[1], selected ? 5 : 4.25, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(far[0], far[1]);
  ctx.lineTo(far[0] - ux * arrowLength - uy * arrowHalfWidth, far[1] - uy * arrowLength + ux * arrowHalfWidth);
  ctx.lineTo(far[0] - ux * arrowLength + uy * arrowHalfWidth, far[1] - uy * arrowLength - ux * arrowHalfWidth);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawFanPatch(
  ctx: CanvasRenderingContext2D,
  fan: FanDefinition,
  transform: PanelTransform,
  selectedMember: number
) {
  const curves = fanFamilyCurves(fan);
  if (curves.length < 2) return;
  curves.forEach((curve, index) => {
    const boundary = index === 0 || index === curves.length - 1;
    const emphasized = index === selectedMember || Boolean(fan.promotedMembers[String(index)]);
    ctx.save();
    ctx.strokeStyle = boundary ? "#ff5353" : "#56dc72";
    ctx.globalAlpha = emphasized ? 1 : 0.64;
    ctx.lineWidth = emphasized ? 2.4 : boundary ? 1.9 : 1.35;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    curve.forEach((point, vertex) => {
      const canvasPoint = worldToCanvas(point, transform);
      if (vertex === 0) ctx.moveTo(canvasPoint[0], canvasPoint[1]);
      else ctx.lineTo(canvasPoint[0], canvasPoint[1]);
    });
    ctx.stroke();
    ctx.restore();
    if (emphasized) drawSlitWorldPath(ctx, {
      id: `fan-member-${index}`,
      name: `Fan curve ${index + 1}`,
      color: boundary ? "#ff5353" : "#56dc72",
      visible: true,
      sourceId: fan.sourceId,
      layerId: fan.layerId,
      bindingKind: fan.bindingKind,
      widthArcsec: 0,
      shiftSeconds: 0,
      curveArcsec: curve,
      inputVertexCount: curve.length,
      freqIndices: [],
      baseFreqIndex: null,
      contourFreqIndices: [],
      contourLevelPercent: 70,
      display: { vmin: 0, vmax: 1, cmap: "magma", scale: "linear" }
    }, transform, true);
  });
  ctx.save();
  ctx.strokeStyle = "#4d89ff";
  ctx.globalAlpha = 0.9;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  [0, curves[0].length - 1].forEach((vertex) => {
    const a = worldToCanvas(curves[0][vertex], transform);
    const b = worldToCanvas(curves.at(-1)![vertex], transform);
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
  });
  ctx.stroke();
  ctx.restore();
}

function pointInsidePolygon(point: [number, number], polygon: [number, number][]): boolean {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const left = polygon[previous];
    const right = polygon[index];
    const denominator = right[1] - left[1];
    if ((left[1] > point[1]) !== (right[1] > point[1]) && Math.abs(denominator) > 1e-12 && point[0] < (right[0] - left[0]) * (point[1] - left[1]) / denominator + left[0]) inside = !inside;
  }
  return inside;
}

function distanceToPolygonBoundary(point: [number, number], polygon: [number, number][]): number {
  let minimum = Number.POSITIVE_INFINITY;
  for (let index = 0; index < polygon.length; index += 1) {
    const left = polygon[index === 0 ? polygon.length - 1 : index - 1];
    const right = polygon[index];
    const dx = right[0] - left[0];
    const dy = right[1] - left[1];
    const denominator = dx * dx + dy * dy;
    const fraction = denominator > 0 ? clamp(((point[0] - left[0]) * dx + (point[1] - left[1]) * dy) / denominator, 0, 1) : 0;
    const x = left[0] + fraction * dx;
    const y = left[1] + fraction * dy;
    minimum = Math.min(minimum, Math.hypot(point[0] - x, point[1] - y));
  }
  return Number.isFinite(minimum) ? minimum : 0;
}

function correlationSeriesForTrack(track: SadTrack, target: [number, number][], affine: Affine): CorrelationSeries {
  const orderedPoints = [...track.points].sort((left, right) => left.mjd - right.mjd);
  const samples = orderedPoints.map((point) => {
    const world = applyAffine([point.x, point.y], affine);
    return { mjd: point.mjd, distance: distanceToPolygonBoundary(world, target) };
  });
  let arrival: CorrelationTick | undefined;
  for (let index = 0; index < samples.length; index += 1) {
    const world = applyAffine([orderedPoints[index]?.x ?? 0, orderedPoints[index]?.y ?? 0], affine);
    if (pointInsidePolygon(world, target) || samples[index].distance <= 1e-6) {
      const previous = samples[index - 1];
      const current = samples[index];
      const fraction = previous && previous.distance > 1e-6
        ? clamp(previous.distance / Math.max(1e-9, previous.distance + current.distance), 0, 1)
        : 0;
      arrival = {
        trackId: track.id,
        label: track.label,
        color: track.color,
        arrivalMjd: previous ? previous.mjd + fraction * (current.mjd - previous.mjd) : current.mjd,
        bracket: previous ? [previous.mjd, current.mjd] : [current.mjd, current.mjd]
      };
      for (let tail = index; tail < samples.length; tail += 1) samples[tail].distance = 0;
      break;
    }
  }
  if (arrival && samples.length >= 5) {
    let peak: { mjd: number; slope: number } | undefined;
    const speeds = orderedPoints.map((point) => displayedTrackSpeedKmS(track, point.frameIndex, affine));
    for (let index = 1; index < samples.length - 1; index += 1) {
      const previous = samples[index - 1];
      const current = samples[index];
      const next = samples[index + 1];
      const previousSpeed = speeds[index - 1];
      const currentSpeed = speeds[index];
      const nextSpeed = speeds[index + 1];
      if (previousSpeed === null || currentSpeed === null || nextSpeed === null) continue;
      const slope = ((nextSpeed - currentSpeed) / Math.max(1e-9, (next.mjd - current.mjd) * 86400)) - ((currentSpeed - previousSpeed) / Math.max(1e-9, (current.mjd - previous.mjd) * 86400));
      if (!peak || slope < peak.slope) peak = { mjd: current.mjd, slope };
    }
    if (peak) arrival.peakDecelMjd = peak.mjd;
  }
  return { track, points: samples, arrival };
}

function drawProbeMarker(ctx: CanvasRenderingContext2D, pixel: [number, number], t: PanelTransform) {
  const [x, y] = pixelToCanvas(pixel, t);
  ctx.save();
  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#56c7d9";
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x - 8, y);
  ctx.lineTo(x + 8, y);
  ctx.moveTo(x, y - 8);
  ctx.lineTo(x, y + 8);
  ctx.stroke();
  ctx.restore();
}

function predictedTrackPoint(track: SadTrack, frameIndex: number): TrackPoint | null {
  const points = [...track.points].sort((left, right) => left.frameIndex - right.frameIndex);
  if (!points.length) return null;
  const exact = points.find((point) => point.frameIndex === frameIndex);
  if (exact) return exact;
  if (points.length === 1) return { ...points[0], frameIndex, isAnchor: false };
  let left: TrackPoint;
  let right: TrackPoint;
  if (frameIndex < points[0].frameIndex) {
    left = points[0];
    right = points[1];
  } else if (frameIndex > points[points.length - 1].frameIndex) {
    left = points[points.length - 2];
    right = points[points.length - 1];
  } else {
    const rightIndex = points.findIndex((point) => point.frameIndex > frameIndex);
    if (rightIndex <= 0) return { ...points[0], frameIndex, isAnchor: false };
    left = points[rightIndex - 1];
    right = points[rightIndex];
  }
  const frameSpan = right.frameIndex - left.frameIndex;
  if (!(frameSpan > 0)) return { ...left, frameIndex, isAnchor: false };
  const fraction = (frameIndex - left.frameIndex) / frameSpan;
  return {
    frameIndex,
    mjd: left.mjd + (right.mjd - left.mjd) * fraction,
    x: left.x + (right.x - left.x) * fraction,
    y: left.y + (right.y - left.y) * fraction,
    confidence: Math.min(left.confidence, right.confidence),
    isAnchor: false
  };
}

function drawTracks(
  ctx: CanvasRenderingContext2D,
  tracks: SadTrack[],
  t: PanelTransform,
  selectedTrackId: string,
  currentFrameIndex: number,
  markerPreview: { trackId: string; frameIndex: number; pixel: [number, number] } | null,
  hoveredTrackMarkerId = ""
) {
  ctx.save();
  for (const track of [...tracks].sort((left, right) => Number(left.id === selectedTrackId) - Number(right.id === selectedTrackId))) {
    if (!track.visible || !track.points.length) continue;
    const selected = track.id === selectedTrackId;
    const points = [...track.points].sort((left, right) => left.frameIndex - right.frameIndex);
    ctx.globalAlpha = selected ? 1 : 0.32;
    ctx.strokeStyle = track.color;
    ctx.lineWidth = selected ? 2.1 : 1.1;
    ctx.beginPath();
    points.forEach((point, index) => {
      const preview = markerPreview?.trackId === track.id && markerPreview.frameIndex === point.frameIndex ? markerPreview.pixel : null;
      const [x, y] = pixelToCanvas(preview ?? [point.x, point.y], t);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    for (const point of points) {
      const preview = markerPreview?.trackId === track.id && markerPreview.frameIndex === point.frameIndex ? markerPreview.pixel : null;
      const [x, y] = pixelToCanvas(preview ?? [point.x, point.y], t);
      const current = point.frameIndex === currentFrameIndex;
      const anchor = point.isAnchor || track.anchors.some((candidate) => candidate.frameIndex === point.frameIndex);
      const radius = current ? 5.5 : anchor ? 4.5 : selected ? 3 : 2.3;
      ctx.fillStyle = track.color;
      ctx.strokeStyle = current ? "#ffffff" : "#101317";
      ctx.lineWidth = current ? 1.8 : 1.1;
      ctx.beginPath();
      if (anchor) {
        ctx.moveTo(x, y - radius - 1);
        ctx.lineTo(x + radius + 1, y);
        ctx.lineTo(x, y + radius + 1);
        ctx.lineTo(x - radius - 1, y);
        ctx.closePath();
      } else {
        ctx.arc(x, y, radius, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.stroke();
      if (hoveredTrackMarkerId === track.id && current) {
        ctx.save();
        ctx.globalAlpha = 0.72;
        ctx.strokeStyle = track.color;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }
    const exactCurrent = points.find((point) => point.frameIndex === currentFrameIndex);
    if (selected && !exactCurrent) {
      const ghost = predictedTrackPoint(track, currentFrameIndex);
      if (ghost) {
        const preview = markerPreview?.trackId === track.id && markerPreview.frameIndex === currentFrameIndex ? markerPreview.pixel : [ghost.x, ghost.y] as [number, number];
        const [x, y] = pixelToCanvas(preview, t);
        ctx.save();
        ctx.globalAlpha = 0.95;
        ctx.strokeStyle = track.color;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.arc(x, y, 6.5, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        if (hoveredTrackMarkerId === track.id) {
          ctx.globalAlpha = 0.72;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(x, y, 10.5, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }
    }
  }
  ctx.restore();
}

function drawSeedSuggestions(ctx: CanvasRenderingContext2D, suggestions: SeedSuggestion[], t: PanelTransform) {
  ctx.save();
  ctx.strokeStyle = "#7cf0ff";
  ctx.lineWidth = 1.4;
  for (const suggestion of suggestions) {
    const [x, y] = pixelToCanvas([suggestion.x, suggestion.y], t);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 9, y);
    ctx.lineTo(x - 5, y);
    ctx.moveTo(x + 5, y);
    ctx.lineTo(x + 9, y);
    ctx.moveTo(x, y - 9);
    ctx.lineTo(x, y - 5);
    ctx.moveTo(x, y + 5);
    ctx.lineTo(x, y + 9);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSources(ctx: CanvasRenderingContext2D, sources: EovsaSource[], t: PanelTransform) {
  ctx.save();
  for (const source of sources) {
    const centroidWorld: [number, number] = Number.isFinite(source.x_centroid_arcsec) && Number.isFinite(source.y_centroid_arcsec)
      ? [source.x_centroid_arcsec, source.y_centroid_arcsec]
      : applyAffine([source.x_centroid_display_pix, source.y_centroid_display_pix], t.pixelToWorldAffine, t.worldOffset);
    const [x, y] = worldToCanvas(centroidWorld, t);
    ctx.strokeStyle = "#ffffff";
    ctx.fillStyle = "#ff553f";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    const peakWorld: [number, number] = Number.isFinite(source.x_peak_arcsec) && Number.isFinite(source.y_peak_arcsec)
      ? [source.x_peak_arcsec as number, source.y_peak_arcsec as number]
      : applyAffine([source.x_peak_display_pix, source.y_peak_display_pix], t.pixelToWorldAffine, t.worldOffset);
    const [px, py] = worldToCanvas(peakWorld, t);
    ctx.strokeStyle = "#ffe36e";
    ctx.beginPath();
    ctx.moveTo(px - 5, py);
    ctx.lineTo(px + 5, py);
    ctx.moveTo(px, py - 5);
    ctx.lineTo(px, py + 5);
    ctx.stroke();
  }
  ctx.restore();
}

export default App;
