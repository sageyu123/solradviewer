export type FrameKind = "image" | "contours";

const SECONDS_PER_DAY = 86_400;

function boundedClosestIndex(values: number[], target: number, lower: number, upper: number): number {
  let best = lower;
  let bestDistance = Math.abs(values[lower] - target);
  for (let index = lower + 1; index <= upper; index += 1) {
    const distance = Math.abs(values[index] - target);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return best;
}

const GRID_EPSILON = 1e-5;

function gridIndexBounds(rangeStartMjd: number, rangeEndMjd: number, stepSeconds: number, anchorMjd: number): [number, number] {
  const lowerMjd = Math.min(rangeStartMjd, rangeEndMjd);
  const upperMjd = Math.max(rangeStartMjd, rangeEndMjd);
  return [
    Math.ceil((lowerMjd - anchorMjd) * SECONDS_PER_DAY / stepSeconds - GRID_EPSILON),
    Math.floor((upperMjd - anchorMjd) * SECONDS_PER_DAY / stepSeconds + GRID_EPSILON)
  ];
}

function gridTimeMjd(anchorMjd: number, index: number, stepSeconds: number): number {
  return anchorMjd + index * stepSeconds / SECONDS_PER_DAY;
}

/** Advance the cursor through native samples or an exact custom-time grid. */
export function advanceFrameTime(
  timesMjd: number[],
  currentMjd: number,
  direction: number,
  stepSeconds: number,
  rangeStartMjd = timesMjd[0],
  rangeEndMjd = timesMjd[timesMjd.length - 1],
  wrap = false,
  gridAnchorMjd = rangeStartMjd
): number {
  if (!timesMjd.length || !direction) return currentMjd;
  const lowerMjd = Math.min(rangeStartMjd, rangeEndMjd);
  const upperMjd = Math.max(rangeStartMjd, rangeEndMjd);
  const sign = direction < 0 ? -1 : 1;
  if (stepSeconds > 0 && Number.isFinite(stepSeconds)) {
    const [minimum, maximum] = gridIndexBounds(lowerMjd, upperMjd, stepSeconds, gridAnchorMjd);
    if (minimum > maximum) return Math.max(lowerMjd, Math.min(upperMjd, currentMjd));
    const position = (currentMjd - gridAnchorMjd) * SECONDS_PER_DAY / stepSeconds;
    const candidate = sign > 0
      ? Math.floor(position + GRID_EPSILON) + 1
      : Math.ceil(position - GRID_EPSILON) - 1;
    if (candidate < minimum || candidate > maximum) {
      if (wrap) return gridTimeMjd(gridAnchorMjd, sign > 0 ? minimum : maximum, stepSeconds);
      return Math.max(lowerMjd, Math.min(upperMjd, currentMjd));
    }
    return gridTimeMjd(gridAnchorMjd, candidate, stepSeconds);
  }

  const lower = boundedClosestIndex(timesMjd, lowerMjd, 0, timesMjd.length - 1);
  const upper = boundedClosestIndex(timesMjd, upperMjd, lower, timesMjd.length - 1);
  const current = boundedClosestIndex(timesMjd, currentMjd, lower, upper);
  const adjacent = (skipDuplicateMjd: boolean) => {
    let candidate = current + sign;
    while (skipDuplicateMjd && candidate >= lower && candidate <= upper && timesMjd[candidate] === timesMjd[current]) candidate += sign;
    if (candidate >= lower && candidate <= upper) return timesMjd[candidate];
    return wrap ? timesMjd[sign > 0 ? lower : upper] : timesMjd[current];
  };
  return adjacent(false);
}

/** Snap direct cursor placement to the active native or custom grid. */
export function snapFrameTime(
  timesMjd: number[],
  targetMjd: number,
  stepSeconds: number,
  rangeStartMjd = timesMjd[0],
  rangeEndMjd = timesMjd[timesMjd.length - 1],
  gridAnchorMjd = rangeStartMjd
): number {
  if (!timesMjd.length) return targetMjd;
  const lowerMjd = Math.min(rangeStartMjd, rangeEndMjd);
  const upperMjd = Math.max(rangeStartMjd, rangeEndMjd);
  const clamped = Math.max(lowerMjd, Math.min(upperMjd, targetMjd));
  if (!(stepSeconds > 0) || !Number.isFinite(stepSeconds)) {
    const lower = boundedClosestIndex(timesMjd, lowerMjd, 0, timesMjd.length - 1);
    const upper = boundedClosestIndex(timesMjd, upperMjd, lower, timesMjd.length - 1);
    return timesMjd[boundedClosestIndex(timesMjd, clamped, lower, upper)];
  }
  const [minimum, maximum] = gridIndexBounds(lowerMjd, upperMjd, stepSeconds, gridAnchorMjd);
  const index = Math.max(minimum, Math.min(maximum, Math.round((clamped - gridAnchorMjd) * SECONDS_PER_DAY / stepSeconds)));
  return gridTimeMjd(gridAnchorMjd, index, stepSeconds);
}

export function timeStepGridPosition(rangeStartMjd: number, rangeEndMjd: number, cursorMjd: number, stepSeconds: number): { index: number; total: number } {
  const [minimum, maximum] = gridIndexBounds(rangeStartMjd, rangeEndMjd, stepSeconds, rangeStartMjd);
  const index = Math.max(minimum, Math.min(maximum, Math.round((cursorMjd - rangeStartMjd) * SECONDS_PER_DAY / stepSeconds)));
  return { index: index - minimum, total: Math.max(1, maximum - minimum + 1) };
}

/** Enumerate the single forward pass used by recording. */
export function frameTimesForTimeRange(
  timesMjd: number[],
  startMjd: number,
  endMjd: number,
  stepSeconds: number,
  gridAnchorMjd = startMjd
): number[] {
  if (!timesMjd.length) return [];
  const lowerMjd = Math.min(startMjd, endMjd);
  const upperMjd = Math.max(startMjd, endMjd);
  if (!(stepSeconds > 0) || !Number.isFinite(stepSeconds)) {
    const lower = boundedClosestIndex(timesMjd, lowerMjd, 0, timesMjd.length - 1);
    const upper = boundedClosestIndex(timesMjd, upperMjd, lower, timesMjd.length - 1);
    return timesMjd.slice(lower, upper + 1);
  }
  const [minimum, maximum] = gridIndexBounds(lowerMjd, upperMjd, stepSeconds, gridAnchorMjd);
  if (minimum > maximum) return [];
  const frames = [gridTimeMjd(gridAnchorMjd, minimum, stepSeconds)];
  while (frames.length <= maximum - minimum) {
    const current = frames[frames.length - 1];
    const next = advanceFrameTime(timesMjd, current, 1, stepSeconds, lowerMjd, upperMjd, false, gridAnchorMjd);
    if (next <= current + GRID_EPSILON / SECONDS_PER_DAY) break;
    frames.push(next);
  }
  return frames;
}

export type ContourBandGeometry = {
  channel: number;
  freqGhz: number;
  level: number;
  polylines: [number, number][][];
};

export type ContourGeometry = {
  bands: ContourBandGeometry[];
  resolvedIndex: number;
  resolvedMjd: number;
};

export type FrameData = ImageBitmap | ContourGeometry;

export type FrameRequestIdentity = {
  sessionId: string;
  sourceId: string;
  resolvedIndex: number;
  operation: string;
  reference: string;
  displayParams: Record<string, unknown>;
  kind: FrameKind;
};

export type ScheduledFrameRequest = {
  url: string;
  /** Per-cursor identity used for readiness/resolution state. */
  key: string;
  /** Bitmap identity; excludes sample address and resolved headers. */
  bitmapKey: string;
  predictedResolution?: FrameResolution;
  kind: FrameKind;
  priority?: {
    distance: number;
    directionBias: number;
  };
};

export type FrameResolution = {
  resolvedIndex: number | null;
  resolvedMjd: number | null;
  offsetSeconds: number | null;
  unavailable: boolean;
};

const BRACKET_EPSILON_MJD = 1e-9;

/**
 * Display-only availability: require a native sample on both sides of the
 * cursor and select the nearest native sample. Bracketed samples remain
 * displayable regardless of their offset; extraction/tracking policies are
 * separate.
 */
export function predictDisplayResolution(
  values: number[],
  requestedMjd: number
): FrameResolution {
  const finite = values
    .map((value, index) => ({ value, index }))
    .filter((entry) => Number.isFinite(entry.value))
    .sort((left, right) => left.value - right.value || left.index - right.index);
  if (!finite.length || !Number.isFinite(requestedMjd)) {
    return { resolvedIndex: null, resolvedMjd: null, offsetSeconds: null, unavailable: true };
  }
  // Allow the virtual-grid/slider arithmetic to land within sub-millisecond
  // rounding of an endpoint that is mathematically the same sample.
  const before = finite.filter((entry) => entry.value <= requestedMjd + BRACKET_EPSILON_MJD).at(-1);
  const after = finite.find((entry) => entry.value >= requestedMjd - BRACKET_EPSILON_MJD);
  if (!before || !after) {
    return { resolvedIndex: null, resolvedMjd: null, offsetSeconds: null, unavailable: true };
  }
  const beforeDistance = Math.abs(before.value - requestedMjd);
  const afterDistance = Math.abs(after.value - requestedMjd);
  const selected = beforeDistance <= afterDistance ? before : after;
  const offsetSeconds = (selected.value - requestedMjd) * SECONDS_PER_DAY;
  return {
    resolvedIndex: selected.index,
    resolvedMjd: selected.value,
    offsetSeconds,
    unavailable: false
  };
}

export type FrameStats = {
  min: number;
  max: number;
  p1: number;
  p99: number;
};

export const OVERLAY_REQUEST_DELAY_MS = 120;
export const PREFETCH_IDLE_MS = 80;
export const PREFETCH_RADIUS = 4;
export const PREFETCH_CONCURRENCY = 6;
// Reserve fetch lanes for the identities playback needs right now: the
// imminent set covers both the on-screen position and the next lookahead
// frame, each of which can carry an image and a contour-geometry request.
// One lane serializes those behind the image every tick, which starves the
// contour overlay; two gives it real (if still bounded) concurrency.
export const IMMINENT_PREFETCH_CONCURRENCY = 2;
// Persistent decoded cubes remove the serial FITS bottleneck. Four warm
// fetches keep rendering busy on a 16-core workstation while reserving ample
// CPU and request-pool capacity for interactive frames.
export const WARM_PREFETCH_CONCURRENCY = 4;
export const DEFAULT_FRAME_CACHE_BYTE_BUDGET = 2 * 1024 ** 3;
// Mutable so the user can raise/lower the decoded-frame bitmap cache's byte
// budget at runtime (see setFrameCacheBudgetBytes). Lowering it evicts down
// to the new ceiling immediately via evictFrameCacheToBudget.
let frameCacheByteBudget = DEFAULT_FRAME_CACHE_BYTE_BUDGET;
export function getFrameCacheBudgetBytes(): number {
  return frameCacheByteBudget;
}
export function setFrameCacheBudgetBytes(bytes: number): void {
  const next = Math.max(1, Math.round(bytes));
  if (next === frameCacheByteBudget) return;
  frameCacheByteBudget = next;
  evictFrameCacheToBudget();
}
export const FRAME_REQUEST_TIMEOUT_MS = 12_000;

type FetchPriority = "imminent" | "interactive" | "warm";

export type FrameCacheWarmPlan = {
  requests: ScheduledFrameRequest[];
  requestedFrames: number;
  targetFrames: number;
  totalIdentities: number;
  retainedIdentities: number;
  missingIdentities: number;
  limited: boolean;
  estimatedBytes: number;
};

const RESOLUTION_CACHE_LIMIT = 192;
const PREFETCH_RETRY_LIMIT = 3;
const PREFETCH_RETRY_BACKOFF_MS = 250;
const DYNAMIC_QUEUE_IDLE_MS = 50;
const frameCache = new Map<string, FrameData>();
const frameCacheSizes = new Map<string, number>();
const frameCacheMetadata = new Map<string, {
  groupKey: string;
  resolvedIndex: number;
  resolvedMjd: number | null;
}>();
let frameCacheBytes = 0;
let pinnedFrameCacheKeys = new Set<string>();
let currentFrameCacheKeys = new Set<string>();
// Bumped on every frameCache put/evict (cacheWrite covers both paths) and on
// clearFrameCache. Consumers that want to react to cache contents changing
// (e.g. the cache-coverage strip) poll this cheaply instead of subscribing
// to every write, which would fire far too often during playback/prefetch.
let frameCacheVersion = 0;
export function frameCacheStateVersion(): number {
  return frameCacheVersion;
}
const resolutionCache = new Map<string, FrameResolution>();
const statsCache = new Map<string, FrameStats>();
type TerminalFrameResult = { status: "unavailable" | "failed"; detail?: string };
const terminalFrameResults = new Map<string, TerminalFrameResult>();
type FrameResult = { data: FrameData; byteSize: number; resolution: FrameResolution; stats?: FrameStats };
type InFlightRequest = {
  promise: Promise<FrameResult>;
  controller: AbortController;
  consumers: number;
  priority: { value: FetchPriority };
};
const inFlight = new Map<string, InFlightRequest>();
let interactiveFetches = 0;
let warmFetches = 0;

function isBitmapData(value: FrameData): value is ImageBitmap {
  return !("bands" in value);
}

function absoluteRequestUrl(url: string): URL {
  const base = typeof window === "undefined" ? "http://localhost/" : window.location.href;
  return new URL(url, base);
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, stableValue(item)])
    );
  }
  return value;
}

export function requestKey(identity: FrameRequestIdentity): string {
  return JSON.stringify(stableValue(identity));
}

function cacheRead(key: string): FrameData | undefined {
  const value = frameCache.get(key);
  if (!value) return undefined;
  frameCache.delete(key);
  frameCache.set(key, value);
  return value;
}

function cacheIdentity(bitmapKey: string): { groupKey: string; resolvedIndex: number } | null {
  try {
    const identity = JSON.parse(bitmapKey) as Record<string, unknown>;
    const resolvedIndex = Number(identity.resolvedIndex);
    if (!Number.isFinite(resolvedIndex)) return null;
    delete identity.resolvedIndex;
    if (identity.displayParams && typeof identity.displayParams === "object") {
      const displayParams = { ...(identity.displayParams as Record<string, unknown>) };
      delete displayParams.targetResolvedIndex;
      identity.displayParams = displayParams;
    }
    return { groupKey: JSON.stringify(stableValue(identity)), resolvedIndex };
  } catch {
    return null;
  }
}

export function frameRequestGroupKey(request: ScheduledFrameRequest): string {
  return cacheIdentity(request.bitmapKey)?.groupKey ?? request.bitmapKey;
}

function estimatedFrameBytes(request: ScheduledFrameRequest): number {
  const exact = frameCacheSizes.get(request.bitmapKey);
  if (exact !== undefined) return exact;
  const groupKey = frameRequestGroupKey(request);
  const groupSizes = [...frameCacheMetadata]
    .filter(([, metadata]) => metadata.groupKey === groupKey)
    .map(([key]) => frameCacheSizes.get(key) ?? 0)
    .filter((size) => size > 0);
  if (groupSizes.length) return Math.max(...groupSizes);
  if (request.kind === "contours") return 256 * 1024;
  try {
    const url = absoluteRequestUrl(request.url);
    const width = Math.max(1, Number(url.searchParams.get("maxWidth")) || 1024);
    const height = Math.max(1, Number(url.searchParams.get("maxHeight")) || 1024);
    return width * height * 4;
  } catch {
    return 1024 * 1024 * 4;
  }
}

export function planFrameCacheWarm(requestGroups: ScheduledFrameRequest[][]): FrameCacheWarmPlan {
  const groups = requestGroups.map((requests) => [...new Map(
    requests
      .filter((request) => !request.predictedResolution?.unavailable)
      .filter((request) => {
        const state = frameRequestState(request);
        return state === "cached" || state === "pending";
      })
      .map((request) => [request.bitmapKey, request])
  ).values()]);
  const selected = new Map<string, ScheduledFrameRequest>();
  let estimatedBytes = 0;
  let targetFrames = 0;

  for (const group of groups) {
    const additions = group.filter((request) => !selected.has(request.bitmapKey));
    const additionBytes = additions.reduce((total, request) => total + estimatedFrameBytes(request), 0);
    if (estimatedBytes + additionBytes > frameCacheByteBudget) break;
    additions.forEach((request) => selected.set(request.bitmapKey, request));
    estimatedBytes += additionBytes;
    targetFrames += 1;
  }

  let retainedIdentities = 0;
  const requests: ScheduledFrameRequest[] = [];
  for (const request of selected.values()) {
    if (frameCache.has(request.bitmapKey)) {
      cacheRead(request.bitmapKey);
      retainedIdentities += 1;
    } else {
      requests.push(request);
    }
  }
  return {
    requests,
    requestedFrames: groups.length,
    targetFrames,
    totalIdentities: selected.size,
    retainedIdentities,
    missingIdentities: requests.length,
    limited: targetFrames < groups.length,
    estimatedBytes
  };
}

// Evicts the single oldest unpinned (or stale-pinned) cached frame, mirroring
// the Map's insertion/access order maintained by cacheRead's touch-on-read.
// Returns false when nothing is left that's safe to evict. Shared by
// cacheWrite's make-room-for-the-incoming-frame loop and
// evictFrameCacheToBudget's shrink-to-a-lowered-budget loop.
function evictOldestCachedFrame(): boolean {
  let oldest: string | undefined;
  for (const candidate of frameCache.keys()) {
    if (pinnedFrameCacheKeys.has(candidate)) continue;
    oldest = candidate;
    break;
  }
  oldest ??= [...pinnedFrameCacheKeys]
    .reverse()
    .find((candidate) => !currentFrameCacheKeys.has(candidate) && frameCache.has(candidate));
  if (oldest === undefined) return false;
  frameCacheBytes -= frameCacheSizes.get(oldest) ?? 0;
  frameCache.delete(oldest);
  frameCacheSizes.delete(oldest);
  frameCacheMetadata.delete(oldest);
  statsCache.delete(oldest);
  return true;
}

// Called whenever the budget shrinks (setFrameCacheBudgetBytes) to evict
// immediately down to the new ceiling, rather than waiting for the next
// cacheWrite to notice. Bumps frameCacheVersion once iff it evicted anything,
// so the cache-coverage strip refreshes.
function evictFrameCacheToBudget(): void {
  let evicted = false;
  while (frameCacheBytes > frameCacheByteBudget) {
    if (!evictOldestCachedFrame()) break;
    evicted = true;
  }
  if (evicted) frameCacheVersion += 1;
}

function cacheWrite(request: ScheduledFrameRequest, value: FrameData, resolution?: FrameResolution, byteSize?: number): void {
  const key = request.bitmapKey;
  const existing = frameCache.get(key);
  const existingBytes = frameCacheSizes.get(key) ?? 0;
  if (existing) {
    frameCacheBytes -= existingBytes;
    frameCache.delete(key);
    frameCacheSizes.delete(key);
    frameCacheMetadata.delete(key);
  }
  const bytes = Math.max(1, Math.round(byteSize ?? (isBitmapData(value) ? value.width * value.height * 4 : 256 * 1024)));
  if (bytes > frameCacheByteBudget) return;
  while (frameCacheBytes + bytes > frameCacheByteBudget) {
    if (!evictOldestCachedFrame()) return;
  }
  frameCache.set(key, value);
  frameCacheSizes.set(key, bytes);
  const identity = cacheIdentity(key);
  if (identity) {
    frameCacheMetadata.set(key, {
      ...identity,
      resolvedMjd: resolution?.resolvedMjd ?? request.predictedResolution?.resolvedMjd ?? null
    });
  }
  frameCacheBytes += bytes;
  frameCacheVersion += 1;
}

function terminalWrite(key: string, value: TerminalFrameResult): void {
  terminalFrameResults.delete(key);
  terminalFrameResults.set(key, value);
  while (terminalFrameResults.size > RESOLUTION_CACHE_LIMIT) {
    const oldest = terminalFrameResults.keys().next().value;
    if (oldest === undefined) break;
    terminalFrameResults.delete(oldest);
  }
}

export function setPinnedFrameRequests(
  requests: ScheduledFrameRequest[],
  currentRequests: ScheduledFrameRequest[] = []
): void {
  pinnedFrameCacheKeys = new Set(
    requests
      .filter((request) => !request.predictedResolution?.unavailable)
      .map((request) => request.bitmapKey)
  );
  currentFrameCacheKeys = new Set(
    currentRequests
      .filter((request) => !request.predictedResolution?.unavailable)
      .map((request) => request.bitmapKey)
  );
}

function resolutionWrite(key: string, value: FrameResolution): void {
  resolutionCache.delete(key);
  resolutionCache.set(key, value);
  while (resolutionCache.size > RESOLUTION_CACHE_LIMIT) {
    const oldest = resolutionCache.keys().next().value;
    if (oldest === undefined) break;
    resolutionCache.delete(oldest);
  }
}

export function clearFrameCache(): void {
  for (const value of frameCache.values()) {
    if (isBitmapData(value)) value.close();
  }
  for (const pending of inFlight.values()) pending.controller.abort();
  frameCache.clear();
  frameCacheSizes.clear();
  frameCacheMetadata.clear();
  frameCacheBytes = 0;
  pinnedFrameCacheKeys.clear();
  currentFrameCacheKeys.clear();
  resolutionCache.clear();
  statsCache.clear();
  terminalFrameResults.clear();
  inFlight.clear();
  frameCacheVersion += 1;
}

export function frameResolution(key: string): FrameResolution | undefined {
  return resolutionCache.get(key);
}

export function frameStats(key: string): FrameStats | undefined {
  return statsCache.get(key);
}

export function hasFrameResult(key: string): boolean {
  return frameCache.has(key);
}

export function frameRequestState(request: ScheduledFrameRequest): "cached" | "unavailable" | "failed" | "pending" {
  if (request.predictedResolution?.unavailable) return "unavailable";
  if (frameCache.has(request.bitmapKey)) return "cached";
  return terminalFrameResults.get(request.bitmapKey)?.status ?? "pending";
}

export function nearestCachedFrame(request: ScheduledFrameRequest): {
  data: FrameData;
  bitmapKey: string;
  resolvedIndex: number;
  resolvedMjd: number | null;
} | undefined {
  const target = cacheIdentity(request.bitmapKey);
  if (!target) return undefined;
  let bestKey = "";
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestIndex = Number.POSITIVE_INFINITY;
  for (const [key, metadata] of frameCacheMetadata) {
    if (metadata.groupKey !== target.groupKey) continue;
    const distance = Math.abs(metadata.resolvedIndex - target.resolvedIndex);
    if (distance < bestDistance || distance === bestDistance && metadata.resolvedIndex < bestIndex) {
      bestKey = key;
      bestDistance = distance;
      bestIndex = metadata.resolvedIndex;
    }
  }
  const metadata = frameCacheMetadata.get(bestKey);
  const data = bestKey ? cacheRead(bestKey) : undefined;
  if (!metadata || !data) return undefined;
  return { data, bitmapKey: bestKey, resolvedIndex: metadata.resolvedIndex, resolvedMjd: metadata.resolvedMjd };
}

export function areFramesCached(requests: ScheduledFrameRequest[]): boolean {
  return requests.every((request) => frameRequestState(request) !== "pending");
}

class SessionNotFoundFrameError extends Error {
  constructor() {
    super("Session not found.");
    this.name = "SessionNotFoundFrameError";
  }
}

class UnavailableFrameError extends Error {
  constructor(message = "Frame unavailable (204).") {
    super(message);
    this.name = "UnavailableFrameError";
  }
}

class FrameRequestTimeoutError extends Error {
  constructor() {
    super(`Frame request timed out after ${FRAME_REQUEST_TIMEOUT_MS} ms.`);
    this.name = "FrameRequestTimeoutError";
  }
}

function parseContourGeometry(value: unknown): ContourGeometry {
  if (!value || typeof value !== "object" || !Array.isArray((value as ContourGeometry).bands)) {
    throw new Error("Contour geometry response is malformed.");
  }
  const geometry = value as ContourGeometry;
  if (!Number.isFinite(Number(geometry.resolvedIndex)) || !Number.isFinite(Number(geometry.resolvedMjd))) {
    throw new Error("Contour geometry response is missing its resolved sample.");
  }
  return geometry;
}

async function fetchFrameData(
  request: ScheduledFrameRequest,
  signal: AbortSignal,
  priority: { value: FetchPriority }
): Promise<FrameResult> {
  if (priority.value === "warm") {
    while (
      priority.value === "warm"
      && !signal.aborted
      && interactiveFetches + warmFetches >= PREFETCH_CONCURRENCY
    ) {
      await waitForFrameDelay(DYNAMIC_QUEUE_IDLE_MS, signal);
    }
  }
  if (signal.aborted) throw new DOMException("Frame request aborted.", "AbortError");
  const countedPriority = priority.value;
  if (countedPriority === "warm") {
    warmFetches += 1;
  } else {
    interactiveFetches += 1;
  }
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort(signal.reason);
  signal.addEventListener("abort", abort, { once: true });
  const timeout = globalThis.setTimeout(() => {
    timedOut = true;
    controller.abort(new FrameRequestTimeoutError());
  }, FRAME_REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(request.url, { signal: controller.signal });
    if (response.status === 204) throw new UnavailableFrameError();
    const unavailableReason = response.headers.get("X-Overlay-Unavailable");
    if (unavailableReason) throw new UnavailableFrameError(`Overlay unavailable: ${unavailableReason}`);
    if (response.status === 404) {
      let detail = "";
      try {
        const body = await response.json() as { detail?: unknown };
        detail = typeof body.detail === "string" ? body.detail : "";
      } catch {
        // Keep the generic status handling below for non-session 404s.
      }
      if (/^Session not found\b/i.test(detail)) throw new SessionNotFoundFrameError();
    }
    if (!response.ok) {
      throw new Error(`Frame request failed (${response.status} ${response.statusText}).`);
    }
    const headerIndex = Number(response.headers.get("X-Resolved-Index"));
    const headerMjd = Number(response.headers.get("X-Resolved-Mjd"));
    const headerOffset = Number(response.headers.get("X-Offset-Seconds"));
    const dataStats = ["X-Data-Min", "X-Data-Max", "X-Data-P1", "X-Data-P99"].map((name) => Number(response.headers.get(name)));
    if (request.kind === "contours") {
      const text = await response.text();
      const geometry = parseContourGeometry(JSON.parse(text));
      const resolvedMjd = Number(geometry.resolvedMjd);
      return {
        data: geometry,
        byteSize: Math.max(1024, text.length * 2),
        resolution: {
          resolvedIndex: Number.isFinite(headerIndex) ? headerIndex : Number(geometry.resolvedIndex),
          resolvedMjd,
          offsetSeconds: Number.isFinite(headerOffset)
            ? headerOffset
            : Number.isFinite(resolvedMjd) ? (resolvedMjd - Number(absoluteRequestUrl(request.url).searchParams.get("sampleMjd"))) * 86400 : null,
          unavailable: false
        }
      };
    }
    const bitmap = await createImageBitmap(await response.blob());
    return {
      data: bitmap,
      byteSize: bitmap.width * bitmap.height * 4,
      resolution: {
        resolvedIndex: Number.isFinite(headerIndex) ? headerIndex : null,
        resolvedMjd: Number.isFinite(headerMjd) ? headerMjd : null,
        offsetSeconds: Number.isFinite(headerOffset) ? headerOffset : null,
        unavailable: false
      },
      stats: dataStats.every(Number.isFinite) ? {
        min: dataStats[0],
        max: dataStats[1],
        p1: dataStats[2],
        p99: dataStats[3]
      } : undefined
    };
  } catch (error) {
    if (timedOut) throw new FrameRequestTimeoutError();
    throw error;
  } finally {
    if (countedPriority === "warm") warmFetches = Math.max(0, warmFetches - 1);
    else interactiveFetches = Math.max(0, interactiveFetches - 1);
    globalThis.clearTimeout(timeout);
    signal.removeEventListener("abort", abort);
  }
}

async function raceWithAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) throw new DOMException("Frame request aborted.", "AbortError");
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(new DOMException("Frame request aborted.", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", abort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", abort);
        reject(error);
      }
    );
  });
}

export async function loadFrame(
  request: ScheduledFrameRequest,
  signal: AbortSignal,
  onSessionNotFound?: () => Promise<unknown>,
  priority: FetchPriority = "interactive"
): Promise<FrameData | null> {
  if (request.predictedResolution?.unavailable) {
    resolutionWrite(request.key, request.predictedResolution);
    return null;
  }
  const terminal = terminalFrameResults.get(request.bitmapKey);
  if (terminal) {
    resolutionWrite(request.key, {
      ...(request.predictedResolution ?? {}),
      resolvedIndex: request.predictedResolution?.resolvedIndex ?? null,
      resolvedMjd: request.predictedResolution?.resolvedMjd ?? null,
      offsetSeconds: request.predictedResolution?.offsetSeconds ?? null,
      unavailable: true
    });
    return null;
  }
  const cached = cacheRead(request.bitmapKey);
  if (cached) {
    resolutionWrite(request.key, {
      ...(request.predictedResolution ?? {}),
      resolvedIndex: request.predictedResolution?.resolvedIndex ?? null,
      resolvedMjd: request.predictedResolution?.resolvedMjd ?? null,
      offsetSeconds: request.predictedResolution?.offsetSeconds ?? null,
      unavailable: false
    });
    return cached;
  }
  let pending = inFlight.get(request.bitmapKey);
  if (!pending) {
    const controller = new AbortController();
    const priorityRef = { value: priority };
    const promise = fetchFrameData(request, controller.signal, priorityRef).then((result) => {
      terminalFrameResults.delete(request.bitmapKey);
      cacheWrite(request, result.data, result.resolution, result.byteSize);
      return result;
    });
    pending = { promise, controller, consumers: 0, priority: priorityRef };
    inFlight.set(request.bitmapKey, pending);
    promise.finally(() => {
      if (inFlight.get(request.bitmapKey) === pending) inFlight.delete(request.bitmapKey);
    }).catch(() => undefined);
  } else if (priority === "imminent" || priority === "interactive" && pending.priority.value === "warm") {
    pending.priority.value = priority;
  }
  pending.consumers += 1;
  try {
    const result = await raceWithAbort(pending.promise, signal);
    const resolution = request.predictedResolution
      ? { ...result.resolution, ...request.predictedResolution, unavailable: false }
      : result.resolution;
    resolutionWrite(request.key, resolution);
    if (result.stats) statsCache.set(request.bitmapKey, result.stats);
    if (!frameCache.has(request.bitmapKey)) cacheWrite(request, result.data, result.resolution, result.byteSize);
    return cacheRead(request.bitmapKey) ?? result.data;
  } catch (error) {
    if (error instanceof SessionNotFoundFrameError && onSessionNotFound) {
      await raceWithAbort(onSessionNotFound(), signal);
      return null;
    }
    if (error instanceof UnavailableFrameError) {
      terminalWrite(request.bitmapKey, { status: "unavailable", detail: error.message });
      resolutionWrite(request.key, {
        ...(request.predictedResolution ?? {}),
        resolvedIndex: request.predictedResolution?.resolvedIndex ?? null,
        resolvedMjd: request.predictedResolution?.resolvedMjd ?? null,
        offsetSeconds: request.predictedResolution?.offsetSeconds ?? null,
        unavailable: true
      });
      return null;
    }
    throw error;
  } finally {
    pending.consumers -= 1;
    if (signal.aborted && pending.consumers === 0 && inFlight.get(request.bitmapKey) === pending) {
      pending.controller.abort();
    }
  }
}

export async function waitForFrameDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) throw new DOMException("Frame request aborted.", "AbortError");
  await new Promise<void>((resolve, reject) => {
    const timer = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, milliseconds);
    const abort = () => {
      globalThis.clearTimeout(timer);
      reject(new DOMException("Frame request aborted.", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
  });
}

export async function prefetchFrames(
  requests: ScheduledFrameRequest[] | (() => ScheduledFrameRequest[]),
  signal: AbortSignal,
  onSessionNotFound?: () => Promise<unknown>,
  onProgress?: () => void,
  priority: FetchPriority = "interactive"
): Promise<void> {
  const dynamic = typeof requests === "function";
  const claimed = new Set<string>();
  const completed = new Set<string>();

  function queue(): ScheduledFrameRequest[] {
    const pending = typeof requests === "function" ? requests() : requests;
    return [...new Map(pending.map((request) => [request.bitmapKey, request])).values()]
      .filter((request) => (
        !claimed.has(request.bitmapKey)
        && !completed.has(request.bitmapKey)
        && (priority === "warm" || !areFramesCached([request]))
      ))
      .sort((left, right) => {
        const distance = (left.priority?.distance ?? Number.POSITIVE_INFINITY)
          - (right.priority?.distance ?? Number.POSITIVE_INFINITY);
        if (distance) return distance;
        const direction = (left.priority?.directionBias ?? Number.POSITIVE_INFINITY)
          - (right.priority?.directionBias ?? Number.POSITIVE_INFINITY);
        if (direction) return direction;
        if (left.kind !== right.kind) return left.kind === "image" ? -1 : 1;
        return 0;
      });
  }

  async function prefetch(request: ScheduledFrameRequest): Promise<void> {
    let lastError: unknown;
    for (let retry = 0; retry <= PREFETCH_RETRY_LIMIT; retry += 1) {
      try {
        await loadFrame(request, signal, onSessionNotFound, priority);
        return;
      } catch (error) {
        if (signal.aborted) return;
        lastError = error;
        if (retry === PREFETCH_RETRY_LIMIT) break;
        try {
          await waitForFrameDelay(PREFETCH_RETRY_BACKOFF_MS * (retry + 1), signal);
        } catch {
          return;
        }
      }
    }
    const detail = lastError instanceof Error ? `${lastError.name}: ${lastError.message}` : String(lastError);
    terminalWrite(request.bitmapKey, { status: "failed", detail });
    console.warn("[frame-prefetch] final failure; playback will skip", JSON.stringify({
      key: request.key,
      bitmapKey: request.bitmapKey,
      url: request.url,
      attempts: PREFETCH_RETRY_LIMIT + 1,
      error: detail
    }));
  }

  async function worker(): Promise<void> {
    while (!signal.aborted) {
      const request = queue()[0];
      if (!request) {
        if (!dynamic) return;
        try {
          await waitForFrameDelay(DYNAMIC_QUEUE_IDLE_MS, signal);
        } catch {
          return;
        }
        continue;
      }
      claimed.add(request.bitmapKey);
      try {
        await prefetch(request);
      } finally {
        claimed.delete(request.bitmapKey);
        // The imminent queue must recover from eviction while a resolved
        // identity stays current. Lookahead keeps its tombstones to avoid
        // repeatedly fetching distant frames that do not fit the cache.
        if (dynamic && priority === "imminent") completed.delete(request.bitmapKey);
        else completed.add(request.bitmapKey);
        if (!signal.aborted) onProgress?.();
      }
    }
  }
  const workerLimit = priority === "warm"
    ? WARM_PREFETCH_CONCURRENCY
    : priority === "imminent" ? IMMINENT_PREFETCH_CONCURRENCY : PREFETCH_CONCURRENCY - IMMINENT_PREFETCH_CONCURRENCY;
  const workers = dynamic ? workerLimit : Math.min(workerLimit, queue().length);
  await Promise.all(Array.from({ length: workers }, () => worker()));
}

export async function warmFrameCache(
  requests: ScheduledFrameRequest[],
  signal: AbortSignal,
  onProgress?: () => void,
  onSessionNotFound?: () => Promise<unknown>
): Promise<void> {
  await prefetchFrames(requests, signal, onSessionNotFound, onProgress, "warm");
}
