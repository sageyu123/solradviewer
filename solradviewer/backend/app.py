"""FastAPI app for the interactive SolRadViewer workbench."""

from __future__ import annotations

import math
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, model_validator

from .data import (
    AIA_DISPLAY_ORIENTATION,
    DEFAULT_DIFF_SECONDS,
    DEFAULT_TEMPORAL_SIGMA_LONG,
    DEFAULT_TEMPORAL_SIGMA_SHORT,
    EOVSA_DISPLAY_ORIENTATION,
    MAX_SLIT_WIDTH_PX,
    OverlayUnavailableError,
    ProgressRegistry,
    RENDER_DISK_CACHE,
    SolRadSession,
    SlitExtractionCancelled,
    data_stats_headers,
    resolve_time_index,
    temporal_window_cap_status,
)


DisplayScale = Literal["linear", "log", "sqrt", "asinh"]
SpectrogramNormalization = Literal["none", "divide", "subtract"]

_BROWSABLE_SOURCE_EXTENSIONS = {".fits", ".fts", ".fit", ".h5", ".hdf5", ".json", ".npz"}


class RoiRequest(BaseModel):
    panel: Literal["aia", "eovsa"] = "aia"
    sourceId: str | None = None
    points: list[list[float]]
    timeIndex: int | None = None
    sampleMjd: float | None = None
    samplingPolicy: Literal["nearest", "previous", "next"] = "nearest"
    maxOffsetSeconds: float | None = None
    freqIndex: int
    xOffsetArcsec: float = 7.0
    yOffsetArcsec: float = 0.0
    diffSeconds: float = DEFAULT_DIFF_SECONDS


class RoiProjectionRequest(BaseModel):
    panel: Literal["aia", "eovsa"] = "aia"
    sourceId: str | None = None
    timeIndex: int | None = None
    sampleMjd: float | None = None
    samplingPolicy: Literal["nearest", "previous", "next"] = "nearest"
    maxOffsetSeconds: float | None = None
    freqIndex: int
    xOffsetArcsec: float = 7.0
    yOffsetArcsec: float = 0.0
    diffSeconds: float = DEFAULT_DIFF_SECONDS


class ChannelLassoRequest(BaseModel):
    """Radio-contour lasso selection addressed in the target panel pixels."""

    panel: Literal["aia", "eovsa"] = "aia"
    sourceId: str | None = None
    targetSourceId: str = "context"
    points: list[list[float]]
    timeIndex: int | None = None
    sampleMjd: float | None = None
    samplingPolicy: Literal["nearest", "previous", "next"] = "nearest"
    maxOffsetSeconds: float | None = None
    freqIndex: int = 0
    diffSeconds: float = DEFAULT_DIFF_SECONDS
    useRunningDiff: bool = True
    xOffsetArcsec: float = 7.0
    yOffsetArcsec: float = 0.0
    levelPercent: float = 50.0
    filled: bool = False
    opacity: float = 0.35
    differenceMode: str | None = None
    levelReference: str = "current"
    differenceOperation: str | None = None
    differenceReference: str = "previous"
    meanStartMjd: float | None = None
    meanEndMjd: float | None = None
    levelMode: str = "percent"
    levelKelvin: float = 1_000_000.0
    levelSfu: float = 1.0
    contourCmap: str = "turbo"

    @model_validator(mode="after")
    def validate_polygon(self) -> "ChannelLassoRequest":
        """Require a finite polygon with at least three vertices."""
        if len(self.points) < 3 or any(
            len(point) != 2 or not all(math.isfinite(value) for value in point)
            for point in self.points
        ):
            raise ValueError("points must contain at least three finite x/y vertices")
        return self


class SadExtractRequest(BaseModel):
    searchRadius: int = 8


class EovsaExtractRequest(BaseModel):
    xOffsetArcsec: float = 7.0
    yOffsetArcsec: float = 0.0
    diffSeconds: float = DEFAULT_DIFF_SECONDS
    useRunningDiff: bool = True
    startIndex: int | None = None
    endIndex: int | None = None
    startMjd: float | None = None
    endMjd: float | None = None
    stride: int = 4
    minSnr: float = 5.0
    differenceOperation: str | None = None
    differenceReference: str = "previous"
    meanStartMjd: float | None = None
    meanEndMjd: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "EovsaExtractRequest":
        """Require one complete legacy or source-native extraction range."""
        has_index = self.startIndex is not None or self.endIndex is not None
        has_mjd = self.startMjd is not None or self.endMjd is not None
        complete_index = self.startIndex is not None and self.endIndex is not None
        complete_mjd = self.startMjd is not None and self.endMjd is not None
        if has_index and has_mjd:
            raise ValueError("Provide index bounds or MJD bounds, not both")
        if not complete_index and not complete_mjd:
            raise ValueError("Provide complete startIndex/endIndex or startMjd/endMjd bounds")
        if complete_mjd:
            assert self.startMjd is not None and self.endMjd is not None
            if not (math.isfinite(self.startMjd) and math.isfinite(self.endMjd)):
                raise ValueError("startMjd and endMjd must be finite")
        return self


class FeatureStepRequest(BaseModel):
    sourceId: str = "context"
    frameIndex: int | None = None
    sampleMjd: float | None = None
    maxOffsetSeconds: float | None = None
    direction: int
    point: list[float] | None = None
    patchRadius: int = 6
    searchRadius: int = 18

    @model_validator(mode="after")
    def validate_address(self) -> "FeatureStepRequest":
        """Require exactly one legacy index or canonical timeline timestamp."""
        if (self.frameIndex is None) == (self.sampleMjd is None):
            raise ValueError("Provide either frameIndex or sampleMjd, not both")
        if self.maxOffsetSeconds is not None and (
            not math.isfinite(self.maxOffsetSeconds) or self.maxOffsetSeconds < 0
        ):
            raise ValueError("maxOffsetSeconds must be a finite non-negative number")
        return self


class TrackDeleteRequest(BaseModel):
    rowIndex: int


class TrackSeedRequest(BaseModel):
    """One user-authored seed in tracked-source image pixels."""

    sourceId: str = "context"
    frameIndex: int
    x: float
    y: float
    label: str | None = None
    color: str = "#56c7d9"


class TrackSyncRequest(BaseModel):
    """Complete client-side track graph for undo, redo, and lightweight edits."""

    tracks: list[dict[str, object]]
    correlationTarget: list[list[float]] | None = None


class CorrelationTargetRequest(BaseModel):
    target: list[list[float]] = []


class TrackAutoRequest(BaseModel):
    """NCC auto-track request bound to one source-layer science chain."""

    trackIds: list[str]
    layerParams: dict[str, object]
    direction: Literal["forward", "backward", "both"] = "both"
    startFrame: int
    rangeStart: int
    rangeEnd: int
    patchRadius: int = 6
    searchRadius: int = 18
    confidenceThreshold: float = 0.5

    @model_validator(mode="after")
    def validate_tracking_values(self) -> "TrackAutoRequest":
        """Require selected tracks and finite, usable tracker radii."""
        if not self.trackIds:
            raise ValueError("trackIds must contain at least one track")
        if self.patchRadius < 2 or self.searchRadius <= self.patchRadius:
            raise ValueError("searchRadius must be larger than patchRadius >= 2")
        if not math.isfinite(self.confidenceThreshold):
            raise ValueError("confidenceThreshold must be finite")
        return self


class TrackRetrackRequest(BaseModel):
    """Edited track graph for constrained anchor-to-anchor re-tracking."""

    track: dict[str, object]
    layerParams: dict[str, object]
    affectedFrame: int | None = None
    patchRadius: int = 6
    searchRadius: int = 18
    confidenceThreshold: float = 0.5


class TrackSuggestionRequest(BaseModel):
    """Dark local-minimum suggestion request for one processed context frame."""

    frameIndex: int
    layerParams: dict[str, object]
    percentile: float = 20.0
    minimumSeparation: float = 8.0
    limit: int = 20


class SlitExtractRequest(BaseModel):
    """One native-cadence time-distance extraction request."""

    slitId: str
    name: str = "Slit"
    sourceId: str
    curveArcsec: list[list[float]]
    width: int = 3
    layerParams: dict[str, object]
    freqIndices: list[int] | None = None
    # The bound contour layer's level settings, sent separately from
    # layerParams so they never enter the extraction disk-cache key: level
    # settings only affect how an already-extracted map is thresholded for
    # display, never the extracted pixels themselves. See radio_contour_threshold.
    contourLevel: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_slit(self) -> "SlitExtractRequest":
        """Require a finite open curve and the supported averaging width."""
        if len(self.curveArcsec) < 2 or any(
            len(point) != 2 or not all(math.isfinite(value) for value in point)
            for point in self.curveArcsec
        ):
            raise ValueError("curveArcsec must contain at least two finite x/y vertices")
        if self.width < 1 or self.width > MAX_SLIT_WIDTH_PX:
            raise ValueError(f"width must be between 1 and {MAX_SLIT_WIDTH_PX} pixels")
        if self.freqIndices is not None and any(index < 0 for index in self.freqIndices):
            raise ValueError("freqIndices must contain non-negative channel indexes")
        return self


class SlitBatchEntry(BaseModel):
    """One slit's curve/width/channels within a shared-source batch request."""

    slitId: str
    name: str = "Slit"
    curveArcsec: list[list[float]]
    width: int = 3
    freqIndices: list[int] | None = None
    # See SlitExtractRequest.contourLevel: display-only, never part of the
    # extraction disk-cache key.
    contourLevel: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_entry(self) -> "SlitBatchEntry":
        """Require a finite open curve and the supported averaging width."""
        if len(self.curveArcsec) < 2 or any(
            len(point) != 2 or not all(math.isfinite(value) for value in point)
            for point in self.curveArcsec
        ):
            raise ValueError("curveArcsec must contain at least two finite x/y vertices")
        if self.width < 1 or self.width > MAX_SLIT_WIDTH_PX:
            raise ValueError(f"width must be between 1 and {MAX_SLIT_WIDTH_PX} pixels")
        if self.freqIndices is not None and any(index < 0 for index in self.freqIndices):
            raise ValueError("freqIndices must contain non-negative channel indexes")
        return self


class SlitBatchExtractRequest(BaseModel):
    """A batch of slits sharing one canonical source and processing snapshot.

    The client groups every visible slit by (sourceId, serialized
    layerParams) before issuing one request per group (see extractAllSlits
    in App.tsx) so the frame-outer batch pass in
    :meth:`SolRadSession.extract_slit_batch` can read each native source
    frame once and sample every slit's (and, for radio, every channel's)
    curve from it.
    """

    sourceId: str
    layerParams: dict[str, object]
    slits: list[SlitBatchEntry]
    # 1-based-in-spirit progress labeling only ("group i/N"); not otherwise
    # used server-side.
    groupIndex: int = 0
    groupCount: int = 1

    @model_validator(mode="after")
    def validate_batch(self) -> "SlitBatchExtractRequest":
        """Require at least one slit entry."""
        if not self.slits:
            raise ValueError("At least one slit is required")
        return self


class SourceAddRequest(BaseModel):
    id: str | None = None
    role: str = "context"
    label: str = ""
    format: str = "fits"
    path: str = ""


class RadioPeakRefreshRequest(BaseModel):
    diffSeconds: float = DEFAULT_DIFF_SECONDS
    differenceMode: str = "running"
    differenceOperation: str | None = None
    differenceReference: str = "previous"
    meanStartMjd: float | None = None
    meanEndMjd: float | None = None


class ChannelOffsetsRequest(BaseModel):
    """Complete per-channel radio alignment table in arcsec."""

    dx: list[float]
    dy: list[float]
    masked: list[bool] | None = None
    channelMask: list[bool] | None = None

    @model_validator(mode="after")
    def validate_values(self) -> "ChannelOffsetsRequest":
        """Require equal-length finite numeric offset arrays."""
        if len(self.dx) != len(self.dy):
            raise ValueError("channelOffsets.dx and channelOffsets.dy must have equal length")
        if not all(math.isfinite(value) for value in [*self.dx, *self.dy]):
            raise ValueError("channelOffsets values must be finite")
        if self.masked is not None and self.channelMask is not None and self.masked != self.channelMask:
            raise ValueError("Provide only one of channelOffsets.masked or channelMask")
        mask = self.masked if self.masked is not None else self.channelMask
        if mask is not None and len(mask) != len(self.dx):
            raise ValueError("channelOffsets.masked must have the same length as dx and dy")
        return self


SESSIONS: dict[str, SolRadSession] = {}
SESSION_LOCK = Lock()
FOREGROUND_LOCK = Lock()
FOREGROUND_RENDERS = 0
FRAME_CACHE_HEADERS = {"Cache-Control": "public, max-age=3600"}
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="SolRadViewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Resolved-Index",
        "X-Resolved-Mjd",
        "X-Offset-Seconds",
        "X-Temporal-Window-Capped",
        "X-Overlay-Unavailable",
        "X-Data-Min",
        "X-Data-Max",
        "X-Data-P1",
        "X-Data-P99",
    ],
)


@app.middleware("http")
async def track_foreground_renders(request: Request, call_next):
    """Let background pre-warm work yield between interactive PNG requests."""
    global FOREGROUND_RENDERS
    is_render = request.url.path.endswith(".png")
    if is_render:
        with FOREGROUND_LOCK:
            FOREGROUND_RENDERS += 1
    try:
        return await call_next(request)
    finally:
        if is_render:
            with FOREGROUND_LOCK:
                FOREGROUND_RENDERS = max(0, FOREGROUND_RENDERS - 1)


def _foreground_active() -> bool:
    with FOREGROUND_LOCK:
        return FOREGROUND_RENDERS > 0


def _register_session(session: SolRadSession, prewarm: bool = False) -> None:
    with SESSION_LOCK:
        previous = SESSIONS.get(session.session_id)
        SESSIONS[session.session_id] = session
    if previous is not None and previous is not session:
        cancel = getattr(previous, "cancel_prewarm", None)
        if callable(cancel):
            cancel()
    if prewarm:
        session.start_prewarm(_foreground_active)

def _resolution_headers(index: int, resolved_mjd: float, offset_seconds: float) -> dict[str, str]:
    return {
        "X-Resolved-Index": str(int(index)),
        "X-Resolved-Mjd": f"{float(resolved_mjd):.12f}",
        "X-Offset-Seconds": f"{float(offset_seconds):.9f}",
    }


def _temporal_headers(
    source: object,
    index: int,
    mode: str,
    sigma_short: float,
    sigma_long: float,
) -> dict[str, str]:
    """Return a header when the server had to cap temporal neighbors."""
    if str(mode).lower() == "none":
        return {}
    times = getattr(source, "times", None)
    if times is None:
        return {}
    capped = temporal_window_cap_status(times, index, mode, sigma_short, sigma_long)
    return {"X-Temporal-Window-Capped": capped} if capped else {}


def _frame_stats(source: object, method_name: str, *args: object, **kwargs: object) -> dict[str, str]:
    method = getattr(source, method_name, None)
    if not callable(method):
        return {}
    return data_stats_headers(method(*args, **kwargs))


def _resolve_request(
    session: SolRadSession,
    source_id: str,
    time_index: int | None,
    sample_mjd: float | None,
    sampling_policy: str,
    max_offset_seconds: float | None,
) -> tuple[str, int, float, float] | None:
    """Resolve either a legacy index or native MJD for a renderable source."""
    if time_index is not None and sample_mjd is not None:
        raise HTTPException(status_code=422, detail="Provide either timeIndex or sampleMjd, not both")
    if max_offset_seconds is not None:
        try:
            valid_offset = math.isfinite(float(max_offset_seconds)) and float(max_offset_seconds) >= 0.0
        except (TypeError, ValueError):
            valid_offset = False
        if not valid_offset:
            raise HTTPException(status_code=422, detail="maxOffsetSeconds must be a finite non-negative number")
    panel = _panel_for_source(session, "aia", source_id)
    if panel == "eovsa":
        axis = session.eovsa.times
    elif panel == "aia":
        axis = session.aia.times
    else:
        raise HTTPException(status_code=400, detail=f"Unknown renderable source: {source_id}")
    if sample_mjd is not None:
        effective_max_offset = max_offset_seconds
        if effective_max_offset is None:
            cadence = _native_cadence_seconds(axis)
            effective_max_offset = None if cadence is None else cadence / 2.0
        try:
            resolved = resolve_time_index(axis, sample_mjd, sampling_policy, effective_max_offset)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if resolved is None:
            return None
        index, resolved_mjd, offset_seconds = resolved
        return panel, index, resolved_mjd, offset_seconds
    if time_index is None:
        raise HTTPException(status_code=422, detail="Either timeIndex or sampleMjd is required")
    index = int(time_index)
    # The pre-P2 radio route used an AIA index and resolved it to the radio
    # axis internally. Keep that alias byte-compatible while exposing the
    # native radio resolution in response headers.
    if panel == "eovsa":
        aia_values = session.aia.times.mjd
        if index < 0 or index >= len(aia_values):
            raise HTTPException(status_code=400, detail="timeIndex is outside the context time axis")
        requested_mjd = float(aia_values[index])
        try:
            resolved = resolve_time_index(axis, requested_mjd, "nearest", max_offset_seconds)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return None if resolved is None else (panel, *resolved)
    values = session.aia.times.mjd
    if index < 0 or index >= len(values):
        raise HTTPException(status_code=400, detail="timeIndex is outside the source time axis")
    resolved_mjd = float(values[index])
    return panel, index, resolved_mjd, 0.0


def _resolve_panel_time(
    session: SolRadSession,
    panel: str,
    time_index: int | None,
    sample_mjd: float | None,
    sampling_policy: str,
    max_offset_seconds: float | None,
) -> tuple[int, int, float, float] | None:
    """Resolve ROI addressing while retaining the native radio index.

    The first tuple item is the legacy AIA index used only for compatibility;
    the second is the layer-native index used to select radio data exactly.
    A radio tolerance never constrains the compatibility-only AIA lookup.
    """
    source_id = session.radio_source_id if panel == "eovsa" else session.context_source_id
    resolved = _resolve_request(session, source_id, time_index, sample_mjd, sampling_policy, max_offset_seconds)
    if resolved is None:
        return None
    resolved_panel, index, resolved_mjd, offset_seconds = resolved
    if resolved_panel == "aia":
        return index, index, resolved_mjd, offset_seconds
    aia = resolve_time_index(session.aia.times, resolved_mjd, "nearest")
    return None if aia is None else (aia[0], index, resolved_mjd, offset_seconds)


def _session(session_id: str) -> SolRadSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session


def _progress_registry(session: object) -> ProgressRegistry:
    """Return the session registry, including lightweight test sessions."""
    registry = getattr(session, "progress_registry", None)
    if not isinstance(registry, ProgressRegistry):
        registry = ProgressRegistry()
        setattr(session, "progress_registry", registry)
    return registry


def _panel_for_source(session: SolRadSession, panel: str, source_id: str | None) -> str:
    if source_id in {session.radio_source_id, "radio", "eovsa"}:
        return "eovsa"
    if source_id in {session.context_source_id, "context", "aia"}:
        return "aia"
    return panel


def _native_cadence_seconds(times: object) -> float | None:
    """Return the median positive cadence of a source-native axis."""
    values = getattr(times, "mjd", times)
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    deltas = [right - left for left, right in zip(finite, finite[1:]) if right > left]
    return None if not deltas else float(sorted(deltas)[len(deltas) // 2] * 86400.0)


def _validate_source_id(session: SolRadSession, source_id: str | None) -> None:
    """Reject an explicit unknown source instead of falling back to panel."""
    if source_id is None:
        return
    known = {
        session.context_source_id,
        session.radio_source_id,
        getattr(session, "spectrogram_source_id", "spectrogram"),
        "context",
        "aia",
        "radio",
        "eovsa",
        "spectrogram",
        "eovsa-spectrogram",
    }
    if source_id not in known:
        raise HTTPException(status_code=422, detail=f"Unknown sourceId: {source_id}")


def _parse_highlight_channels(value: str | None) -> list[int] | None:
    """Parse an optional comma-separated contour highlight channel list."""
    if value is None:
        return None
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(token))
        except ValueError:
            continue
    return result


def _parse_channels(value: str | None) -> list[int] | None:
    """Parse an optional comma-separated contour band subset."""
    if value is None:
        return None
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            result.append(int(token))
        except ValueError:
            continue
    return result


def _effective_overlay_channels(session: object, channels: list[int] | None) -> list[int] | None:
    """Drop masked radio channels while preserving the legacy no-mask query shape."""
    eovsa = getattr(session, "eovsa", None)
    nfreq = int(getattr(eovsa, "nfreq", 0))
    raw_mask = getattr(session, "channel_mask", None)
    table = getattr(session, "channel_offsets", None)
    table_mask = table.get("masked") if isinstance(table, dict) else None
    if nfreq <= 0:
        nfreq = len(raw_mask) if isinstance(raw_mask, list) else len(table_mask) if isinstance(table_mask, list) else 0
    mask = [
        (bool(raw_mask[index]) if isinstance(raw_mask, list) and index < len(raw_mask) else False)
        or (bool(table_mask[index]) if isinstance(table_mask, list) and index < len(table_mask) else False)
        for index in range(nfreq)
    ]
    if not any(mask):
        return channels
    requested = range(nfreq) if channels is None else channels
    return [int(index) for index in requested if 0 <= int(index) < nfreq and not mask[int(index)]]


def _filesystem_listing(path: str | None, show_hidden: bool = False) -> dict[str, object]:
    """Return one filtered server-local directory listing.

    Parameters
    ----------
    path
        Absolute directory path. ``None`` selects the current user's home directory.
    show_hidden
        Include dot-prefixed entries when true.

    Returns
    -------
    dict
        Resolved directory, parent, and sortable entry metadata.

    Raises
    ------
    HTTPException
        If the requested path is relative, missing, inaccessible, or not a directory.
    """
    requested = Path.home() if path is None or not path.strip() else Path(path).expanduser()
    if not requested.is_absolute():
        raise HTTPException(status_code=400, detail="Filesystem path must be absolute")
    try:
        current = requested.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Directory not found: {requested}") from exc
    except OSError as exc:
        raise HTTPException(status_code=403, detail=f"Directory is not accessible: {requested}") from exc
    if not current.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {current}")

    try:
        candidates = list(current.iterdir())
    except OSError:
        candidates = []
    entries: list[dict[str, object]] = []
    for candidate in candidates:
        if not show_hidden and candidate.name.startswith("."):
            continue
        try:
            is_dir = candidate.is_dir()
            if not is_dir and candidate.suffix.lower() not in _BROWSABLE_SOURCE_EXTENSIONS:
                continue
            metadata = candidate.stat()
        except OSError:
            continue
        entries.append({
            "name": candidate.name,
            "path": str(candidate.absolute()),
            "isDir": is_dir,
            "size": int(metadata.st_size),
            "mtime": float(metadata.st_mtime),
        })
    entries.sort(key=lambda entry: (not bool(entry["isDir"]), str(entry["name"]).casefold(), str(entry["name"])))
    return {
        "path": str(current),
        "parent": str(current.parent),
        "entries": entries,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/fs/list")
def filesystem_list(
    path: str | None = Query(default=None),
    show_hidden: bool = Query(default=False, alias="showHidden"),
) -> dict[str, object]:
    """List source-compatible files in one server-local directory."""
    return _filesystem_listing(path, show_hidden)


@app.get("/api/cache-stats")
def cache_stats() -> dict[str, int]:
    return RENDER_DISK_CACHE.stats()


@app.post("/api/sessions/default")
def create_default_session() -> dict[str, object]:
    try:
        session = SolRadSession.create_default()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load default event: {exc}") from exc
    _register_session(session)
    return session.api_meta()


@app.post("/api/sessions/sample")
def create_sample_session() -> dict[str, object]:
    try:
        session = SolRadSession.create_sample()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load sample data: {exc}") from exc
    _register_session(session)
    return session.api_meta()


@app.post("/api/sessions/load-manifest")
def load_manifest_session(manifest: dict[str, object]) -> dict[str, object]:
    session: SolRadSession | None = None
    try:
        session = SolRadSession.create_from_manifest(manifest)
        loaded_state = session.api_loaded_state(manifest)
        session.update_load_progress(5)
        _register_session(session, prewarm=True)
        payload = session.api_meta()
        payload["loadedState"] = loaded_state
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load manifest: {exc}") from exc
    finally:
        if session is not None:
            session.finish_load_progress()


@app.post("/api/sessions/load-json")
def load_json_session(state: dict[str, object]) -> dict[str, object]:
    session: SolRadSession | None = None
    try:
        session = SolRadSession.create_from_state(state)
        loaded_state = session.api_loaded_state(state)
        session.update_load_progress(5)
        _register_session(session, prewarm=True)
        payload = session.api_meta()
        payload["loadedState"] = loaded_state
        return payload
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load JSON session: {exc}") from exc
    finally:
        if session is not None:
            session.finish_load_progress()


@app.get("/api/sessions/{session_id}/meta")
def session_meta(session_id: str) -> dict[str, object]:
    return _session(session_id).api_meta()


@app.get("/api/sessions/{session_id}/prewarm-status")
def prewarm_status(session_id: str) -> dict[str, int | bool]:
    return _session(session_id).prewarm_status()


@app.get("/api/sessions/{session_id}/progress")
def session_progress(session_id: str) -> dict[str, object]:
    """Return active operations and the legacy pre-warm status snapshot."""
    session = _session(session_id)
    operations = [
        operation
        for operation in _progress_registry(session).snapshot()
        if operation["total"] is None or int(operation["done"]) < int(operation["total"])
    ]
    return {
        "operations": operations,
        "prewarm": session.prewarm_status(),
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    with SESSION_LOCK:
        session = SESSIONS.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    session.cancel_prewarm()
    return {"deleted": True}


@app.post("/api/sessions/{session_id}/sources/{source_id}/channel-offsets")
def set_channel_offsets(session_id: str, source_id: str, request: ChannelOffsetsRequest) -> dict[str, object]:
    """Store the complete source-level radio channel-offset calibration.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param source_id: Radio source identifier or legacy radio alias.
    :type source_id: str
    :param request: Complete x/y offset table in arcsec.
    :type request: ChannelOffsetsRequest
    :returns: The canonical stored table.
    :rtype: dict[str, object]
    :raises fastapi.HTTPException: If the source or table is invalid.
    """
    session = _session(session_id)
    if source_id not in {session.radio_source_id, "radio", "eovsa"}:
        raise HTTPException(status_code=422, detail=f"Unknown radio source: {source_id}")
    if len(request.dx) != session.eovsa.nfreq:
        raise HTTPException(status_code=422, detail=f"channelOffsets must contain exactly {session.eovsa.nfreq} channels")
    try:
        offsets = session.set_channel_offsets({
            "dx": request.dx,
            "dy": request.dy,
            **({"masked": request.masked} if request.masked is not None else {}),
            **({"channelMask": request.channelMask} if request.channelMask is not None else {}),
        })
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"channelOffsets": offsets, "channelMask": session.channel_mask_payload() if hasattr(session, "channel_mask_payload") else offsets.get("masked", [])}


@app.post("/api/sessions/{session_id}/sources/{source_id}/select-channels")
def select_channels_by_lasso(session_id: str, source_id: str, request: ChannelLassoRequest) -> Response:
    """Select radio bands whose shifted contour centroids enter a lasso."""
    session = _session(session_id)
    if source_id not in {session.radio_source_id, "radio", "eovsa"}:
        raise HTTPException(status_code=422, detail=f"Unknown contour source: {source_id}")
    if request.targetSourceId in {session.context_source_id, "context", "aia"}:
        target_panel = "aia"
    elif request.targetSourceId in {session.radio_source_id, "radio", "eovsa"}:
        target_panel = "eovsa"
    else:
        raise HTTPException(status_code=422, detail=f"Unknown contour target source: {request.targetSourceId}")
    resolved = _resolve_request(
        session, source_id, request.timeIndex, request.sampleMjd,
        request.samplingPolicy, request.maxOffsetSeconds,
    )
    if resolved is None:
        return Response(status_code=204)
    _, native_index, resolved_mjd, offset_seconds = resolved
    if target_panel == "eovsa":
        target_time_index = native_index
    elif request.sampleMjd is None:
        if request.timeIndex is None:
            raise HTTPException(status_code=422, detail="AIA lasso selection requires timeIndex or sampleMjd")
        target_time_index = int(request.timeIndex)
    else:
        aia_cadence = _native_cadence_seconds(session.aia.times)
        aia_max_offset = None if aia_cadence is None else aia_cadence / 2.0
        aia_resolved = resolve_time_index(session.aia.times, resolved_mjd, "nearest", aia_max_offset)
        if aia_resolved is None:
            return Response(status_code=204)
        target_time_index = aia_resolved[0]
    try:
        channels = session.eovsa_channels_in_polygon(
            target_time_index,
            request.diffSeconds,
            request.useRunningDiff,
            request.xOffsetArcsec,
            request.yOffsetArcsec,
            request.levelPercent,
            request.points,
            request.differenceMode,
            request.levelReference,
            request.differenceOperation,
            request.differenceReference,
            request.meanStartMjd,
            request.meanEndMjd,
            request.levelMode,
            request.levelKelvin,
            request.levelSfu,
            eovsa_index=native_index,
            target_panel=target_panel,
        )
    except OverlayUnavailableError as exc:
        raise HTTPException(status_code=503, detail=exc.reason) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content={"channels": channels},
        headers=_resolution_headers(native_index, resolved_mjd, offset_seconds),
    )


@app.get("/api/sessions/{session_id}/aia/frame.png")
def aia_frame(
    session_id: str,
    timeIndex: int,
    vmin: float = 0.5,
    vmax: float = 1.5,
    cmap: str = "gray",
    scale: DisplayScale = "linear",
    orientation: str = AIA_DISPLAY_ORIENTATION,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    useRunningDiff: bool = True,
    differenceMode: str | None = None,
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
    radialGamma: float = 0.0,
    temporalMode: str = "none",
    temporalSigmaShort: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
    temporalSigmaLong: float = DEFAULT_TEMPORAL_SIGMA_LONG,
    maxWidth: int | None = None,
    maxHeight: int | None = None,
) -> Response:
    """Render a legacy-addressed AIA frame.

    :param maxWidth: Optional maximum PNG width after science processing.
    :type maxWidth: int or None
    :param maxHeight: Optional maximum PNG height after science processing.
    :type maxHeight: int or None
    :returns: PNG response with full-resolution science-statistic headers.
    :rtype: fastapi.responses.Response
    """
    session = _session(session_id)
    try:
        difference_mode_value = differenceMode if differenceMode is not None else ("running" if useRunningDiff else "none")
        stats_headers = _frame_stats(
            session.aia,
            "frame",
            timeIndex,
            difference_mode=difference_mode_value,
            difference_operation=differenceOperation,
            difference_reference=differenceReference,
            diff_seconds=diffSeconds,
            mean_start_mjd=meanStartMjd,
            mean_end_mjd=meanEndMjd,
            radial_gamma=radialGamma,
            temporal_mode=temporalMode,
            temporal_sigma_short=temporalSigmaShort,
            temporal_sigma_long=temporalSigmaLong,
        )
        content = session.aia.texture(
            timeIndex, vmin, vmax, cmap, scale, orientation, useRunningDiff, differenceMode,
            differenceOperation, differenceReference, diffSeconds, meanStartMjd, meanEndMjd,
            radialGamma, temporalMode, temporalSigmaShort, temporalSigmaLong,
            maxWidth, maxHeight,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={
            **FRAME_CACHE_HEADERS,
            **stats_headers,
            **_temporal_headers(session.aia, timeIndex, temporalMode, temporalSigmaShort, temporalSigmaLong),
        },
    )


@app.get("/api/sessions/{session_id}/eovsa/frame.png")
def eovsa_frame(
    session_id: str,
    timeIndex: int,
    freqIndex: int,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    vmin: float = -1.0e6,
    vmax: float = 5.0e6,
    cmap: str = "turbo",
    scale: DisplayScale = "linear",
    orientation: str = EOVSA_DISPLAY_ORIENTATION,
    useRunningDiff: bool = True,
    differenceMode: str | None = None,
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
    fullCube: bool = False,
    radialGamma: float = 0.0,
    temporalMode: str = "none",
    temporalSigmaShort: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
    temporalSigmaLong: float = DEFAULT_TEMPORAL_SIGMA_LONG,
    maxWidth: int | None = None,
    maxHeight: int | None = None,
) -> Response:
    """Render a legacy-addressed radio frame.

    :param maxWidth: Optional maximum PNG width after science processing.
    :type maxWidth: int or None
    :param maxHeight: Optional maximum PNG height after science processing.
    :type maxHeight: int or None
    :returns: PNG response with full-resolution science-statistic headers.
    :rtype: fastapi.responses.Response
    """
    session = _session(session_id)
    try:
        aia_mjd = session.aia.times[int(timeIndex)].mjd
        difference_mode_value = differenceMode if differenceMode is not None else ("running" if useRunningDiff else "none")
        content = session.eovsa.texture_for_aia_time(
            aia_mjd, freqIndex, diffSeconds, vmin, vmax, cmap, scale, orientation,
            useRunningDiff, differenceMode, differenceOperation, differenceReference, meanStartMjd, meanEndMjd,
            fullCube, None, radialGamma, temporalMode, temporalSigmaShort, temporalSigmaLong,
            maxWidth, maxHeight,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    nearest_time_index = getattr(session.eovsa, "nearest_time_index", None)
    if callable(nearest_time_index):
        eovsa_index = int(nearest_time_index(aia_mjd))
    else:
        eovsa_index = min(
            range(len(session.eovsa.times)),
            key=lambda index: abs(float(session.eovsa.times[index].mjd) - float(aia_mjd)),
        )
    stats_headers = _frame_stats(
        session.eovsa,
        "frame_for_aia_time",
        float(aia_mjd),
        freqIndex,
        diffSeconds,
        use_running_diff=useRunningDiff,
        difference_mode=difference_mode_value,
        difference_operation=differenceOperation,
        difference_reference=differenceReference,
        mean_start_mjd=meanStartMjd,
        mean_end_mjd=meanEndMjd,
        full_cube=fullCube,
        eovsa_index=eovsa_index,
        radial_gamma=radialGamma,
        temporal_mode=temporalMode,
        temporal_sigma_short=temporalSigmaShort,
        temporal_sigma_long=temporalSigmaLong,
    )
    return Response(
        content=content,
        media_type="image/png",
        headers={
            **FRAME_CACHE_HEADERS,
            **stats_headers,
            **_temporal_headers(session.eovsa, eovsa_index, temporalMode, temporalSigmaShort, temporalSigmaLong),
        },
    )


@app.get("/api/sessions/{session_id}/eovsa/aia-contours.png")
def eovsa_aia_contours(
    session_id: str,
    timeIndex: int,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    useRunningDiff: bool = True,
    xOffsetArcsec: float = 7.0,
    yOffsetArcsec: float = 0.0,
    levelPercent: float = 50.0,
    filled: bool = False,
    opacity: float = 0.35,
    differenceMode: str | None = None,
    levelReference: str = "current",
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
    levelMode: str = "percent",
    levelKelvin: float = 1_000_000.0,
    levelSfu: float = 1.0,
    contourCmap: str = "turbo",
    highlightChannels: str | None = None,
    offsetsRev: int | None = None,
    channels: str | None = None,
) -> Response:
    del offsetsRev
    session = _session(session_id)
    try:
        content = session.eovsa_all_band_contours_on_aia(
            timeIndex,
            diffSeconds,
            useRunningDiff,
            xOffsetArcsec,
            yOffsetArcsec,
            levelPercent,
            filled,
            opacity,
            differenceMode,
            levelReference,
            differenceOperation,
            differenceReference,
            meanStartMjd,
            meanEndMjd,
            levelMode,
            levelKelvin,
            levelSfu,
            contourCmap,
            highlight_channels=_parse_highlight_channels(highlightChannels),
            channels=_effective_overlay_channels(session, _parse_channels(channels)),
        )
    except HTTPException:
        raise
    except OverlayUnavailableError as exc:
        return Response(
            content=session.transparent_overlay_png("aia"),
            media_type="image/png",
            headers={**FRAME_CACHE_HEADERS, "X-Overlay-Unavailable": exc.reason},
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=content, media_type="image/png", headers=FRAME_CACHE_HEADERS)


@app.get("/api/sessions/{session_id}/eovsa/spectrogram.png")
def eovsa_spectrogram(
    session_id: str,
    vmin: float = 0.5,
    vmax: float = 150.0,
    scale: DisplayScale = "log",
    cmap: str = "viridis",
    frequencyScale: str = "linear",
    frequencyMinGhz: float | None = None,
    frequencyMaxGhz: float | None = None,
    normalization: SpectrogramNormalization = "none",
) -> Response:
    session = _session(session_id)
    try:
        content = session.spectrogram.texture(
            vmin, vmax, cmap, scale, frequencyScale, frequencyMinGhz, frequencyMaxGhz, normalization
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=content, media_type="image/png", headers=FRAME_CACHE_HEADERS)


@app.get("/api/sessions/{session_id}/sources/{source_id}/frame.png")
def source_frame(
    session_id: str,
    source_id: str,
    timeIndex: int | None = None,
    sampleMjd: float | None = None,
    samplingPolicy: Literal["nearest", "previous", "next"] = "nearest",
    maxOffsetSeconds: float | None = None,
    freqIndex: int = 0,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    vmin: float = 0.5,
    vmax: float = 1.5,
    cmap: str = "gray",
    scale: DisplayScale = "linear",
    orientation: str = "solar",
    useRunningDiff: bool = True,
    differenceMode: str | None = None,
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
    fullCube: bool = False,
    radialGamma: float = 0.0,
    temporalMode: str = "none",
    temporalSigmaShort: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
    temporalSigmaLong: float = DEFAULT_TEMPORAL_SIGMA_LONG,
    maxWidth: int | None = None,
    maxHeight: int | None = None,
    warm: bool = False,
) -> Response:
    """Render one source frame resolved on its native time axis.

    ``timeIndex`` remains the legacy alias; callers must provide exactly one
    of it or ``sampleMjd``. Successful responses include native-resolution
    headers, while a tolerance miss returns HTTP 204.

    :param maxWidth: Optional maximum PNG width after science processing.
    :type maxWidth: int or None
    :param maxHeight: Optional maximum PNG height after science processing.
    :type maxHeight: int or None
    :returns: Resolved PNG response, or an empty 204 response when unavailable.
    :rtype: fastapi.responses.Response
    """
    session = _session(session_id)
    _validate_source_id(session, source_id)
    if source_id not in {
        session.context_source_id,
        session.radio_source_id,
        "context",
        "aia",
        "radio",
        "eovsa",
    }:
        raise HTTPException(status_code=400, detail=f"Unknown renderable source: {source_id}")
    resolved = _resolve_request(
        session, source_id, timeIndex, sampleMjd, samplingPolicy, maxOffsetSeconds
    )
    if resolved is None:
        return Response(status_code=204)
    panel, native_index, resolved_mjd, offset_seconds = resolved
    try:
        difference_mode_value = differenceMode if differenceMode is not None else ("running" if useRunningDiff else "none")
        if panel == "aia":
            stats_headers = _frame_stats(
                session.aia,
                "frame",
                native_index,
                difference_mode=difference_mode_value,
                difference_operation=differenceOperation,
                difference_reference=differenceReference,
                diff_seconds=diffSeconds,
                mean_start_mjd=meanStartMjd,
                mean_end_mjd=meanEndMjd,
                radial_gamma=radialGamma,
                temporal_mode=temporalMode,
                temporal_sigma_short=temporalSigmaShort,
                temporal_sigma_long=temporalSigmaLong,
            )
            texture_kwargs = {"cache_durable": False} if warm else {}
            content = session.aia.texture(
                native_index, vmin, vmax, cmap, scale, orientation, useRunningDiff, differenceMode,
                differenceOperation, differenceReference, diffSeconds, meanStartMjd, meanEndMjd,
                radialGamma, temporalMode, temporalSigmaShort, temporalSigmaLong,
                maxWidth, maxHeight,
                **texture_kwargs,
            )
        else:
            stats_headers = _frame_stats(
                session.eovsa,
                "frame_for_aia_time",
                resolved_mjd,
                freqIndex,
                diffSeconds,
                use_running_diff=useRunningDiff,
                difference_mode=difference_mode_value,
                difference_operation=differenceOperation,
                difference_reference=differenceReference,
                mean_start_mjd=meanStartMjd,
                mean_end_mjd=meanEndMjd,
                full_cube=fullCube,
                eovsa_index=native_index,
                radial_gamma=radialGamma,
                temporal_mode=temporalMode,
                temporal_sigma_short=temporalSigmaShort,
                temporal_sigma_long=temporalSigmaLong,
            )
            texture_kwargs = {"cache_durable": False} if warm else {}
            content = session.eovsa.texture_for_aia_time(
                resolved_mjd,
                freqIndex,
                diffSeconds,
                vmin,
                vmax,
                cmap,
                scale,
                orientation,
                useRunningDiff,
                differenceMode,
                differenceOperation,
                differenceReference,
                meanStartMjd,
                meanEndMjd,
                fullCube,
                eovsa_index=native_index,
                radial_gamma=radialGamma,
                temporal_mode=temporalMode,
                temporal_sigma_short=temporalSigmaShort,
                temporal_sigma_long=temporalSigmaLong,
                max_width=maxWidth,
                max_height=maxHeight,
                **texture_kwargs,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={
            **FRAME_CACHE_HEADERS,
            **_resolution_headers(native_index, resolved_mjd, offset_seconds),
            **stats_headers,
            **_temporal_headers(
                session.aia if panel == "aia" else session.eovsa,
                native_index,
                temporalMode,
                temporalSigmaShort,
                temporalSigmaLong,
            ),
        },
    )


@app.get("/api/sessions/{session_id}/sources/{source_id}/timeseries")
def source_timeseries(
    session_id: str,
    source_id: str,
    x: float = Query(..., description="Base-layer image pixel x coordinate"),
    y: float = Query(..., description="Base-layer image pixel y coordinate"),
    patchRadius: int = Query(1, ge=0),
    startMjd: float = Query(...),
    endMjd: float = Query(...),
    maxPoints: int = Query(400, ge=1),
    freqIndex: int = Query(0, ge=0),
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    differenceMode: str | None = None,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
    useRunningDiff: bool = True,
    samplingPolicy: Literal["nearest", "previous", "next"] = "nearest",
    maxOffsetSeconds: float | None = None,
    temporalMode: Literal["lowpass", "bandpass"] | None = None,
    temporalSigmaShort: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
    temporalSigmaLong: float = DEFAULT_TEMPORAL_SIGMA_LONG,
) -> JSONResponse:
    """Return a native source pixel light curve and optional temporal smoothing.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param source_id: Renderable context or radio source identifier.
    :type source_id: str
    :param x: Base-layer image pixel x coordinate.
    :type x: float
    :param y: Base-layer image pixel y coordinate.
    :type y: float
    :returns: JSON light-curve payload matching the pixel-probe schema.
    :rtype: JSONResponse
    :raises HTTPException: If the source or probe parameters are invalid.
    """
    session = _session(session_id)
    numeric_values = {
        "x": x,
        "y": y,
        "startMjd": startMjd,
        "endMjd": endMjd,
        "diffSeconds": diffSeconds,
        "temporalSigmaShort": temporalSigmaShort,
        "temporalSigmaLong": temporalSigmaLong,
    }
    for name, value in numeric_values.items():
        if not math.isfinite(float(value)):
            raise HTTPException(status_code=422, detail=f"{name} must be finite")
    if maxOffsetSeconds is not None and (
        not math.isfinite(float(maxOffsetSeconds)) or float(maxOffsetSeconds) < 0
    ):
        raise HTTPException(status_code=422, detail="maxOffsetSeconds must be finite and non-negative")
    if temporalSigmaShort <= 0 or temporalSigmaLong <= 0:
        raise HTTPException(status_code=422, detail="Temporal sigmas must be positive")
    try:
        payload = session.pixel_timeseries(
            source_id=source_id,
            x=x,
            y=y,
            patch_radius=patchRadius,
            start_mjd=startMjd,
            end_mjd=endMjd,
            max_points=maxPoints,
            freq_index=freqIndex,
            difference_mode=differenceMode,
            difference_operation=differenceOperation,
            difference_reference=differenceReference,
            diff_seconds=diffSeconds,
            mean_start_mjd=meanStartMjd,
            mean_end_mjd=meanEndMjd,
            use_running_diff=useRunningDiff,
            temporal_mode=temporalMode,
            temporal_sigma_short=temporalSigmaShort,
            temporal_sigma_long=temporalSigmaLong,
            sampling_policy=samplingPolicy,
            max_offset_seconds=maxOffsetSeconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=payload, headers=FRAME_CACHE_HEADERS)


@app.get("/api/sessions/{session_id}/sources/{source_id}/spectrogram.png")
def source_spectrogram(
    session_id: str,
    source_id: str,
    vmin: float = 0.5,
    vmax: float = 150.0,
    scale: DisplayScale = "log",
    cmap: str = "viridis",
    frequencyScale: str = "linear",
    frequencyMinGhz: float | None = None,
    frequencyMaxGhz: float | None = None,
    normalization: SpectrogramNormalization = "none",
) -> Response:
    session = _session(session_id)
    try:
        if source_id not in {session.spectrogram_source_id, "spectrogram", "eovsa-spectrogram"}:
            raise ValueError(f"Unknown spectrogram source: {source_id}")
        content = session.spectrogram.texture(
            vmin, vmax, cmap, scale, frequencyScale, frequencyMinGhz, frequencyMaxGhz, normalization
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=content, media_type="image/png", headers=FRAME_CACHE_HEADERS)


@app.get("/api/sessions/{session_id}/sources/{source_id}/overlay-contours.png")
def source_overlay_contours(
    session_id: str,
    source_id: str,
    targetSourceId: str = "context",
    timeIndex: int | None = None,
    sampleMjd: float | None = None,
    samplingPolicy: Literal["nearest", "previous", "next"] = "nearest",
    maxOffsetSeconds: float | None = None,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    useRunningDiff: bool = True,
    xOffsetArcsec: float = 7.0,
    yOffsetArcsec: float = 0.0,
    levelPercent: float = 50.0,
    filled: bool = False,
    opacity: float = 0.35,
    differenceMode: str | None = None,
    levelReference: str = "current",
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
    levelMode: str = "percent",
    levelKelvin: float = 1_000_000.0,
    levelSfu: float = 1.0,
    contourCmap: str = "turbo",
    highlightChannels: str | None = None,
    offsetsRev: int | None = None,
    channels: str | None = None,
) -> Response:
    """Render radio contours on a context- or radio-native target grid.

    ``targetSourceId`` identifies the panel's base image source, independent
    of the panel slot. The default retains the legacy context/AIA target.
    Successful responses remain addressed by the contour source's native
    radio index and timestamp.
    """
    del offsetsRev
    session = _session(session_id)
    try:
        if source_id not in {session.radio_source_id, "radio", "eovsa"}:
            raise HTTPException(status_code=422, detail=f"Unknown contour source: {source_id}")
        if targetSourceId in {session.context_source_id, "context", "aia"}:
            target_panel = "aia"
        elif targetSourceId in {session.radio_source_id, "radio", "eovsa"}:
            target_panel = "eovsa"
        else:
            raise HTTPException(status_code=422, detail=f"Unknown contour target source: {targetSourceId}")
        resolved = _resolve_request(
            session, source_id, timeIndex, sampleMjd, samplingPolicy, maxOffsetSeconds
        )
        if resolved is None:
            return Response(status_code=204)
        _, native_index, resolved_mjd, offset_seconds = resolved
        if target_panel == "eovsa":
            target_time_index = native_index
        elif sampleMjd is None:
            target_time_index = int(timeIndex)  # validated as an AIA alias above
        else:
            aia_cadence = _native_cadence_seconds(session.aia.times)
            aia_max_offset = None if aia_cadence is None else aia_cadence / 2.0
            aia_resolved = resolve_time_index(session.aia.times, resolved_mjd, "nearest", aia_max_offset)
            if aia_resolved is None:
                return Response(status_code=204)
            target_time_index = aia_resolved[0]
        content = session.eovsa_all_band_contours_on_aia(
            target_time_index,
            diffSeconds,
            useRunningDiff,
            xOffsetArcsec,
            yOffsetArcsec,
            levelPercent,
            filled,
            opacity,
            differenceMode,
            levelReference,
            differenceOperation,
            differenceReference,
            meanStartMjd,
            meanEndMjd,
            levelMode,
            levelKelvin,
            levelSfu,
            contourCmap,
            eovsa_index=native_index,
            target_panel=target_panel,
            highlight_channels=_parse_highlight_channels(highlightChannels),
            channels=_effective_overlay_channels(session, _parse_channels(channels)),
        )
    except HTTPException:
        raise
    except OverlayUnavailableError as exc:
        return Response(
            content=session.transparent_overlay_png(target_panel),
            media_type="image/png",
            headers={
                **FRAME_CACHE_HEADERS,
                **_resolution_headers(native_index, resolved_mjd, offset_seconds),
                "X-Overlay-Unavailable": exc.reason,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=content,
        media_type="image/png",
        headers={**FRAME_CACHE_HEADERS, **_resolution_headers(native_index, resolved_mjd, offset_seconds)},
    )


@app.get("/api/sessions/{session_id}/sources/{source_id}/contour-geometry")
def source_contour_geometry(
    session_id: str,
    source_id: str,
    targetSourceId: str = "context",
    timeIndex: int | None = None,
    sampleMjd: float | None = None,
    samplingPolicy: Literal["nearest", "previous", "next"] = "nearest",
    maxOffsetSeconds: float | None = None,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    useRunningDiff: bool = True,
    levelPercent: float = 50.0,
    differenceMode: str | None = None,
    levelReference: str = "current",
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
    levelMode: str = "percent",
    levelKelvin: float = 1_000_000.0,
    levelSfu: float = 1.0,
    warm: bool = False,
) -> Response:
    """Return unshifted radio contour polylines on a target-native grid.

    Alignment offsets and presentation parameters are deliberately absent;
    the browser applies them when drawing the cached vectors.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param source_id: Radio contour source identifier.
    :type source_id: str
    :param targetSourceId: Base image source receiving the contours.
    :type targetSourceId: str
    :returns: Per-band contour geometry plus the resolved native sample.
    :rtype: fastapi.responses.Response
    """
    session = _session(session_id)
    try:
        if source_id not in {session.radio_source_id, "radio", "eovsa"}:
            raise HTTPException(status_code=422, detail=f"Unknown contour source: {source_id}")
        if targetSourceId in {session.context_source_id, "context", "aia"}:
            target_panel = "aia"
        elif targetSourceId in {session.radio_source_id, "radio", "eovsa"}:
            target_panel = "eovsa"
        else:
            raise HTTPException(status_code=422, detail=f"Unknown contour target source: {targetSourceId}")
        resolved = _resolve_request(
            session, source_id, timeIndex, sampleMjd, samplingPolicy, maxOffsetSeconds
        )
        if resolved is None:
            return Response(status_code=204)
        _, native_index, resolved_mjd, offset_seconds = resolved
        if target_panel == "eovsa":
            target_time_index = native_index
        elif sampleMjd is None:
            target_time_index = int(timeIndex)
        else:
            aia_cadence = _native_cadence_seconds(session.aia.times)
            aia_max_offset = None if aia_cadence is None else aia_cadence / 2.0
            aia_resolved = resolve_time_index(session.aia.times, resolved_mjd, "nearest", aia_max_offset)
            if aia_resolved is None:
                return Response(status_code=204)
            target_time_index = aia_resolved[0]
        geometry = session.eovsa_contour_geometry(
            target_time_index,
            diffSeconds,
            useRunningDiff,
            levelPercent,
            differenceMode,
            levelReference,
            differenceOperation,
            differenceReference,
            meanStartMjd,
            meanEndMjd,
            levelMode,
            levelKelvin,
            levelSfu,
            eovsa_index=native_index,
            target_panel=target_panel,
            cache_durable=not warm,
        )
    except HTTPException:
        raise
    except OverlayUnavailableError as exc:
        return JSONResponse(
            content={"bands": [], "resolvedIndex": native_index, "resolvedMjd": resolved_mjd},
            headers={
                **FRAME_CACHE_HEADERS,
                **_resolution_headers(native_index, resolved_mjd, offset_seconds),
                "X-Overlay-Unavailable": exc.reason,
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = {
        "bands": geometry.get("bands", []),
        "resolvedIndex": native_index,
        "resolvedMjd": resolved_mjd,
    }
    return JSONResponse(
        content=payload,
        headers={**FRAME_CACHE_HEADERS, **_resolution_headers(native_index, resolved_mjd, offset_seconds)},
    )


@app.get("/api/sessions/{session_id}/sources/{source_id}/radio-contour-thresholds")
def radio_contour_thresholds(
    session_id: str,
    source_id: str,
    freqIndices: str,
    levelPercent: float = 50.0,
    levelReference: str = "current",
    levelMode: str = "percent",
    levelKelvin: float = 1_000_000.0,
    levelSfu: float = 1.0,
    diffSeconds: float = DEFAULT_DIFF_SECONDS,
    differenceMode: str | None = None,
    useRunningDiff: bool = True,
    differenceOperation: str | None = None,
    differenceReference: str = "previous",
    meanStartMjd: float | None = None,
    meanEndMjd: float | None = None,
) -> dict[str, object]:
    """Resolve per-channel contour thresholds without extracting a map.

    This lets a time-distance lane's contour family re-render live as the
    bound layer's level settings change, matching the image-panel overlay's
    threshold exactly for the ``kelvin``, ``sfu``, and ``global`` percent
    cases without repeating a slit extraction. The ``current`` percent
    reference is intentionally not resolved here: it takes each channel's own
    already-extracted map peak, which the client already holds.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param source_id: Radio contour source identifier.
    :type source_id: str
    :param freqIndices: Comma-separated native radio channel indexes.
    :type freqIndices: str
    :returns: Per-channel resolved thresholds, ``None`` where unavailable.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    if source_id not in {session.radio_source_id, "radio", "eovsa"}:
        raise HTTPException(status_code=422, detail=f"Unknown contour source: {source_id}")
    channels = _parse_channels(freqIndices) or []
    thresholds: dict[str, float | None] = {}
    for channel in channels:
        try:
            thresholds[str(channel)] = session.radio_contour_threshold(
                channel,
                levelMode,
                levelReference,
                levelPercent,
                levelKelvin,
                levelSfu,
                float("nan"),
                diff_seconds=diffSeconds,
                difference_mode=differenceMode,
                use_running_diff=useRunningDiff,
                difference_operation=differenceOperation,
                difference_reference=differenceReference,
                mean_start_mjd=meanStartMjd,
                mean_end_mjd=meanEndMjd,
            )
        except Exception:
            thresholds[str(channel)] = None
    return {"thresholds": thresholds}


@app.post("/api/sessions/{session_id}/sources")
def add_source(session_id: str, request: SourceAddRequest) -> dict[str, object]:
    session = _session(session_id)
    spec: dict[str, object] = {
        "role": request.role,
        "label": request.label,
        "format": request.format,
        "path": request.path,
    }
    if request.id:
        spec["id"] = request.id
    try:
        source = session.add_source_spec(spec)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = session.api_meta()
    payload["addedSource"] = source
    return payload


@app.post("/api/sessions/{session_id}/radio/peak-cache/refresh")
def refresh_radio_peak_cache(session_id: str, request: RadioPeakRefreshRequest) -> dict[str, object]:
    session = _session(session_id)
    try:
        key, values = session.radio_global_peaks(
            request.diffSeconds,
            request.differenceMode,
            refresh=True,
            difference_operation=request.differenceOperation,
            difference_reference=request.differenceReference,
            mean_start_mjd=request.meanStartMjd,
            mean_end_mjd=request.meanEndMjd,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "key": key,
        "values": values,
        "radioPeakCache": session.radio_peak_cache,
        "radioPeakTableCache": session.radio_peak_table_cache,
    }


@app.post("/api/sessions/{session_id}/roi")
def set_roi(session_id: str, request: RoiRequest) -> Response:
    """Set an ROI from pixels addressed by legacy index or native MJD."""
    session = _session(session_id)
    _validate_source_id(session, request.sourceId)
    if request.sourceId in {session.spectrogram_source_id, "spectrogram", "eovsa-spectrogram"}:
        raise HTTPException(status_code=422, detail="Spectrogram sources do not support image ROIs")
    panel = _panel_for_source(session, request.panel, request.sourceId)
    resolved = _resolve_panel_time(
        session,
        panel,
        request.timeIndex,
        request.sampleMjd,
        request.samplingPolicy,
        request.maxOffsetSeconds,
    )
    if resolved is None:
        return Response(status_code=204)
    resolved_time_index, native_index, resolved_mjd, offset_seconds = resolved
    try:
        payload = session.set_roi_from_pixels(
            panel,
            request.points,
            resolved_time_index,
            request.freqIndex,
            request.xOffsetArcsec,
            request.yOffsetArcsec,
            request.diffSeconds,
            native_index if panel == "eovsa" else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=payload, headers=_resolution_headers(native_index, resolved_mjd, offset_seconds))


@app.post("/api/sessions/{session_id}/roi/projection")
def roi_projection(session_id: str, request: RoiProjectionRequest) -> Response:
    """Project the stored ROI using the requested native sample."""
    session = _session(session_id)
    _validate_source_id(session, request.sourceId)
    if request.sourceId in {session.spectrogram_source_id, "spectrogram", "eovsa-spectrogram"}:
        raise HTTPException(status_code=422, detail="Spectrogram sources do not support image ROIs")
    panel = _panel_for_source(session, request.panel, request.sourceId)
    resolved = _resolve_panel_time(
        session,
        panel,
        request.timeIndex,
        request.sampleMjd,
        request.samplingPolicy,
        request.maxOffsetSeconds,
    )
    if resolved is None:
        return Response(status_code=204)
    resolved_time_index, native_index, resolved_mjd, offset_seconds = resolved
    try:
        points = session.roi_pixels_for_panel(
            panel,
            resolved_time_index,
            request.freqIndex,
            request.xOffsetArcsec,
            request.yOffsetArcsec,
            request.diffSeconds,
            native_index if panel == "eovsa" else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(
        content={"points": points},
        headers=_resolution_headers(native_index, resolved_mjd, offset_seconds),
    )


@app.post("/api/sessions/{session_id}/extract/aia-sads")
def extract_aia_sads(session_id: str, request: SadExtractRequest) -> dict[str, object]:
    session = _session(session_id)
    try:
        rows = session.extract_sads(search_radius=request.searchRadius)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(rows), "rows": rows, "tracks": session.tracks}


@app.post("/api/sessions/{session_id}/track/seed")
def add_track_seed(session_id: str, request: TrackSeedRequest) -> dict[str, object]:
    """Create one authoritative anchor and its containing track.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param request: Context-pixel seed request.
    :type request: TrackSeedRequest
    :returns: The created track and complete session track list.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    try:
        track = session.add_track_seed(
            request.sourceId,
            request.frameIndex,
            request.x,
            request.y,
            request.label,
            request.color,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"track": track, "tracks": session.tracks}


@app.post("/api/sessions/{session_id}/track/sync")
def sync_tracks(session_id: str, request: TrackSyncRequest) -> dict[str, object]:
    """Persist a complete track snapshot after a reversible UI mutation.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param request: Complete canonical track snapshot.
    :type request: TrackSyncRequest
    :returns: Normalized session tracks.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    try:
        if request.correlationTarget is not None:
            session.set_correlation_target(request.correlationTarget)
        tracks = session.set_tracks(request.tracks)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tracks": tracks}


@app.post("/api/sessions/{session_id}/correlation-target")
def sync_correlation_target(session_id: str, request: CorrelationTargetRequest) -> dict[str, object]:
    session = _session(session_id)
    try:
        target = session.set_correlation_target(request.target)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"correlationTarget": target}


@app.post("/api/sessions/{session_id}/track/auto")
def auto_track(session_id: str, request: TrackAutoRequest) -> dict[str, object]:
    """Run velocity-predicted NCC tracking on selected session tracks.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param request: Track IDs, processing chain, direction, and frame bounds.
    :type request: TrackAutoRequest
    :returns: Updated tracks and number of attempted frame steps.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    registry = _progress_registry(session)
    lower = min(int(request.rangeStart), int(request.rangeEnd))
    upper = max(int(request.rangeStart), int(request.rangeEnd))
    selected = set(request.trackIds)
    total = 0
    for track in session.tracks:
        if str(track.get("id")) not in selected:
            continue
        points = [point for point in track.get("points", []) if isinstance(point, dict)]
        if not points:
            continue
        origin = min(points, key=lambda point: abs(int(point["frameIndex"]) - int(request.startFrame)))
        origin_frame = int(origin["frameIndex"])
        if request.direction in {"forward", "both"}:
            total += max(0, upper - origin_frame)
        if request.direction in {"backward", "both"}:
            total += max(0, origin_frame - lower)
    total = max(1, total)
    op_id = registry.start(f"Tracking {len(request.trackIds)} seeds", total=total)
    done = 0

    def report() -> None:
        nonlocal done
        done += 1
        registry.update(op_id, done=min(done, total), total=total)

    try:
        tracks = session.auto_track(
            request.trackIds,
            request.layerParams,
            request.direction,
            request.startFrame,
            request.rangeStart,
            request.rangeEnd,
            request.patchRadius,
            request.searchRadius,
            request.confidenceThreshold,
            report,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        registry.finish(op_id)
    return {"tracks": tracks, "processedFrames": done}


@app.post("/api/sessions/{session_id}/track/stop")
def stop_tracking(session_id: str) -> dict[str, bool]:
    """Request cooperative cancellation of the active tracking pass.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :returns: Cancellation acknowledgement.
    :rtype: dict[str, bool]
    """
    _session(session_id).cancel_tracking()
    return {"stopped": True}


@app.post("/api/sessions/{session_id}/track/retrack-segment")
def retrack_segment(session_id: str, request: TrackRetrackRequest) -> dict[str, object]:
    """Re-solve an edited track between its authoritative anchors.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param request: Edited track and exact left-layer processing chain.
    :type request: TrackRetrackRequest
    :returns: Re-solved track, all session tracks, and affected frame.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    try:
        track = session.retrack_track(
            request.track,
            request.layerParams,
            request.patchRadius,
            request.searchRadius,
            request.confidenceThreshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"track": track, "tracks": session.tracks, "affectedFrame": request.affectedFrame}


@app.post("/api/sessions/{session_id}/track/suggest-seeds")
def suggest_track_seeds(session_id: str, request: TrackSuggestionRequest) -> dict[str, object]:
    """Suggest dark, separated local minima inside the stored ROI.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param request: Frame, processing chain, and candidate-selection limits.
    :type request: TrackSuggestionRequest
    :returns: Ordered context-pixel candidate positions.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    try:
        suggestions = session.suggest_tracking_seeds(
            request.frameIndex,
            request.layerParams,
            request.percentile,
            request.minimumSeparation,
            request.limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"suggestions": suggestions}


@app.post("/api/sessions/{session_id}/track/step")
def track_feature_step(session_id: str, request: FeatureStepRequest) -> Response:
    """Advance context tracking from a legacy frame or timeline timestamp.

    Native timestamp requests use the context source's default half-cadence
    tolerance unless ``maxOffsetSeconds`` is supplied. A tolerance miss returns
    HTTP 204; successful native requests include P2 resolution headers.
    """
    session = _session(session_id)
    frame_index = request.frameIndex
    resolution_headers: dict[str, str] = {}
    if request.sampleMjd is not None:
        resolved = _resolve_request(
            session,
            session.context_source_id,
            None,
            request.sampleMjd,
            "nearest",
            request.maxOffsetSeconds,
        )
        if resolved is None:
            return Response(status_code=204)
        _, frame_index, resolved_mjd, offset_seconds = resolved
        resolution_headers = _resolution_headers(frame_index, resolved_mjd, offset_seconds)
    if frame_index is None:  # Guard direct calls that bypass request validation.
        raise HTTPException(status_code=422, detail="Either frameIndex or sampleMjd is required")
    try:
        row = session.track_feature_step(
            source_id=request.sourceId,
            frame_index=frame_index,
            direction=request.direction,
            point=request.point,
            patch_radius=request.patchRadius,
            search_radius=request.searchRadius,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content={"row": row}, headers=resolution_headers)


@app.post("/api/sessions/{session_id}/track/delete-point")
def delete_track_point(session_id: str, request: TrackDeleteRequest) -> dict[str, object]:
    session = _session(session_id)
    try:
        rows = session.delete_track_point(request.rowIndex)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"count": len(rows), "rows": rows}


@app.post("/api/sessions/{session_id}/extract/eovsa-sources")
def extract_eovsa_sources(session_id: str, request: EovsaExtractRequest) -> dict[str, object]:
    """Extract radio sources over legacy AIA-index or native-MJD bounds."""
    session = _session(session_id)
    registry = _progress_registry(session)
    op_id = registry.start("Extracting radio sources")

    def report(done: int, total: int) -> None:
        registry.update(op_id, done=done, total=total)

    try:
        rows = session.extract_eovsa_sources(
            x_offset=request.xOffsetArcsec,
            y_offset=request.yOffsetArcsec,
            diff_seconds=request.diffSeconds,
            start_index=request.startIndex,
            end_index=request.endIndex,
            start_mjd=request.startMjd,
            end_mjd=request.endMjd,
            stride=request.stride,
            min_snr=request.minSnr,
            use_running_diff=request.useRunningDiff,
            difference_operation=request.differenceOperation,
            difference_reference=request.differenceReference,
            mean_start_mjd=request.meanStartMjd,
            mean_end_mjd=request.meanEndMjd,
            on_progress=report,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        registry.finish(op_id)
    return {"count": len(rows), "rows": rows}


@app.post("/api/sessions/{session_id}/extract/radio-sources")
def extract_radio_sources(session_id: str, request: EovsaExtractRequest) -> dict[str, object]:
    return extract_eovsa_sources(session_id, request)


def _slit_map_payload(
    session: object,
    result: dict[str, object],
    cache_hit: bool,
    wall_seconds: float,
    frequency_index: int,
    frequency_ghz: float | None,
    contour_level: dict[str, object] | None,
) -> dict[str, object]:
    """Build one client-facing map payload from an extracted slit result.

    Shared by the single-slit and batch slit extraction endpoints so the
    contour-threshold and NaN-safe stat logic (see
    :meth:`SolRadSession.radio_contour_threshold`) has one definition.

    :param session: Owning session, used for the radio contour threshold.
    :type session: object
    :param result: One extracted map record (from :meth:`SolRadSession.extract_slit`
        or :meth:`SolRadSession.extract_radio_slit_channels`/``extract_slit_batch``).
    :type result: dict[str, object]
    :param cache_hit: Whether this map came from the disk cache.
    :type cache_hit: bool
    :param wall_seconds: Wall time attributed to this map.
    :type wall_seconds: float
    :param frequency_index: Native radio channel index, or -1 for non-radio.
    :type frequency_index: int
    :param frequency_ghz: Physical frequency in GHz, or ``None`` for non-radio.
    :type frequency_ghz: float or None
    :param contour_level: Optional bound-layer contour level settings.
    :type contour_level: dict[str, object] or None
    :returns: Client-facing map payload.
    :rtype: dict[str, object]
    """
    intensity = np.asarray(result["intensity"], dtype=float)
    finite = intensity[np.isfinite(intensity)]
    table = [
        [float(value) if math.isfinite(float(value)) else None for value in row]
        for row in intensity
    ]
    data_max = float(np.max(finite)) if finite.size else 1.0
    contour_threshold: float | None = None
    # contourLevel rides separately from layerParams (never in the
    # extraction cache key - see SlitExtractRequest) and is populated
    # only for radio-bound slits (bindingKind "contours"); its absence
    # for image/raw bindings and non-radio maps leaves the threshold
    # None, and the client falls back to a local percent-of-peak estimate.
    if frequency_index >= 0 and contour_level is not None:
        try:
            contour_threshold = session.radio_contour_threshold(
                frequency_index,
                str(contour_level.get("levelMode", "percent")),
                str(contour_level.get("levelReference", "current")),
                float(contour_level.get("levelPercent", 50.0)),
                float(contour_level.get("levelKelvin", 1_000_000.0)),
                float(contour_level.get("levelSfu", 1.0)),
                data_max,
                diff_seconds=float(contour_level.get("diffSeconds", contour_level.get("cadenceSeconds", DEFAULT_DIFF_SECONDS))),
                difference_mode=contour_level.get("differenceMode"),
                use_running_diff=bool(contour_level.get("useRunningDiff", True)),
                difference_operation=contour_level.get("differenceOperation"),
                difference_reference=str(contour_level.get("differenceReference", "previous")),
                mean_start_mjd=contour_level.get("meanStartMjd"),
                mean_end_mjd=contour_level.get("meanEndMjd"),
            )
        except Exception:
            contour_threshold = None
    return {
        "npix": int(intensity.shape[0]),
        "ntime": int(intensity.shape[1]),
        "intensity": table,
        "distanceArcsec": np.asarray(result["distance_arcsec"], dtype=float).tolist(),
        "timeMjd": np.asarray(result["time_mjd"], dtype=float).tolist(),
        "curveVerticesArcsec": np.asarray(result["curve_vertices_arcsec"], dtype=float).tolist(),
        "dataMin": float(np.min(finite)) if finite.size else 0.0,
        "dataMax": data_max,
        "dataP1": float(np.percentile(finite, 1)) if finite.size else 0.0,
        "dataP99": float(np.percentile(finite, 99)) if finite.size else 1.0,
        "cacheHit": cache_hit,
        "wallSeconds": wall_seconds,
        "mapCacheHit": cache_hit,
        "mapWallSeconds": wall_seconds,
        "cacheKey": str(result["cache_key"]),
        "freqIndex": frequency_index if frequency_index >= 0 else None,
        "freqGhz": frequency_ghz,
        "contourThreshold": contour_threshold,
    }


@app.post("/api/sessions/{session_id}/slits/extract")
def extract_slit(session_id: str, request: SlitExtractRequest) -> dict[str, object]:
    """Extract one time-distance map through the displayed image pipeline.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param request: Slit geometry, width, source, and processing snapshot.
    :type request: SlitExtractRequest
    :returns: Native axes, intensity map, cache status, and timing.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    try:
        canonical_source_id, times, _ = session._tracking_source_geometry(request.sourceId)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    is_radio = canonical_source_id == session.radio_source_id
    requested_frequencies = request.freqIndices if is_radio and request.freqIndices else [
        int(request.layerParams.get("freqIndex", 0)) if is_radio else -1
    ]
    frequency_indices = list(dict.fromkeys(int(value) for value in requested_frequencies))
    if is_radio and any(value >= int(session.eovsa.nfreq) for value in frequency_indices):
        raise HTTPException(status_code=422, detail="freqIndices contains a channel outside the radio cube")
    registry = _progress_registry(session)
    is_multi_channel = is_radio and len(frequency_indices) > 1
    # extract_radio_slit_channels makes one time-major pass over all
    # channels (see data.py), so its progress denominator is the native
    # frame count, not frames x channels; the single-channel path below
    # still reports one frame-count unit per frame.
    total_frames = len(times) if is_multi_channel else len(times) * len(frequency_indices)
    op_id = registry.start(f"Extracting {request.name}", total=total_frames)

    session._slit_extract_cancel.clear()
    results: list[dict[str, object]] = []
    cache_hits: list[bool] = []
    wall_times: list[float] = []

    try:
        if is_multi_channel:
            def report(done: int, _total: int) -> None:
                registry.update(op_id, done=done, total=total_frames)

            results, cache_hits, wall_times, bundle = session.extract_radio_slit_channels(
                request.slitId,
                request.curveArcsec,
                request.width,
                request.layerParams,
                frequency_indices,
                report,
            )
        else:
            result, cache_hit, wall_seconds = session.extract_slit(
                request.slitId,
                request.sourceId,
                request.curveArcsec,
                request.width,
                request.layerParams,
                lambda done, _total: registry.update(op_id, done=done, total=total_frames),
                reset_cancel=False,
            )
            results = [result]
            cache_hits = [cache_hit]
            wall_times = [wall_seconds]
            frequencies_ghz = [
                float(session.eovsa.freqs_hz[frequency_indices[0]] / 1e9)
            ] if is_radio else []
            bundle = session.store_slit_result_bundle(
                request.slitId, results, frequency_indices if is_radio else [], frequencies_ghz
            )
    except SlitExtractionCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        registry.finish(op_id)

    frequencies_ghz = [
        float(session.eovsa.freqs_hz[index] / 1e9) for index in frequency_indices
    ] if is_radio else []

    maps = [
        _slit_map_payload(
            session,
            result,
            cache_hits[index],
            wall_times[index],
            frequency_indices[index],
            frequencies_ghz[index] if frequencies_ghz else None,
            request.contourLevel,
        )
        for index, result in enumerate(results)
    ]
    primary = maps[0]
    return {
        "slitId": request.slitId,
        "sourceId": str(bundle["source_id"]),
        "width": int(bundle["width"]),
        **primary,
        "additionalMaps": maps[1:],
        "cacheHit": all(cache_hits),
        "wallSeconds": float(sum(wall_times)),
    }


@app.post("/api/sessions/{session_id}/slits/extract-batch")
def extract_slit_batch(session_id: str, request: SlitBatchExtractRequest) -> dict[str, object]:
    """Extract many slits sharing one source and processing snapshot in one pass.

    Companion to ``/slits/extract`` for the "All slits" extraction mode: the
    client groups every visible slit by (sourceId, serialized layerParams)
    and issues one request per group, each of which reads every native
    source frame exactly once (see :meth:`SolRadSession.extract_slit_batch`)
    instead of once per slit.

    :param session_id: In-memory session identifier.
    :type session_id: str
    :param request: One shared-source, shared-processing batch of slits.
    :type request: SlitBatchExtractRequest
    :returns: ``{"results": {slitId: <same shape as /slits/extract>}}``.
    :rtype: dict[str, object]
    """
    session = _session(session_id)
    try:
        canonical_source_id, times, _ = session._tracking_source_geometry(request.sourceId)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    is_radio = canonical_source_id == session.radio_source_id
    if is_radio:
        default_freq = [int(request.layerParams.get("freqIndex", 0))]
        for entry in request.slits:
            requested = entry.freqIndices if entry.freqIndices else default_freq
            if any(value >= int(session.eovsa.nfreq) for value in requested):
                raise HTTPException(status_code=422, detail="freqIndices contains a channel outside the radio cube")

    registry = _progress_registry(session)
    total_frames = len(times)
    slit_count = len(request.slits)
    label = f"Extracting {slit_count} slit{'s' if slit_count != 1 else ''}..."
    if request.groupCount > 1:
        label = f"Extracting {slit_count} slit{'s' if slit_count != 1 else ''} (group {request.groupIndex + 1}/{request.groupCount})..."
    op_id = registry.start(label, total=total_frames)

    session._slit_extract_cancel.clear()
    entries = [
        {
            "slitId": entry.slitId,
            "sourceId": request.sourceId,
            "curveArcsec": entry.curveArcsec,
            "width": entry.width,
            "layerParams": dict(request.layerParams),
            "freqIndices": entry.freqIndices,
        }
        for entry in request.slits
    ]
    try:
        bundles = session.extract_slit_batch(
            entries,
            lambda done, _total: registry.update(op_id, done=done, total=total_frames),
        )
    except SlitExtractionCancelled as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        registry.finish(op_id)

    contour_by_slit = {entry.slitId: entry.contourLevel for entry in request.slits}
    responses: dict[str, object] = {}
    for entry in request.slits:
        info = bundles[entry.slitId]
        results = info["results"]
        cache_hits = info["cache_hits"]
        wall_times = info["wall_times"]
        bundle = info["bundle"]
        freq_indices = info["freq_indices"]
        frequencies_ghz = [
            float(session.eovsa.freqs_hz[index] / 1e9) for index in freq_indices
        ] if is_radio else []
        maps = [
            _slit_map_payload(
                session,
                result,
                cache_hits[index],
                wall_times[index],
                freq_indices[index] if is_radio else -1,
                frequencies_ghz[index] if frequencies_ghz else None,
                contour_by_slit[entry.slitId],
            )
            for index, result in enumerate(results)
        ]
        primary = maps[0]
        responses[entry.slitId] = {
            "slitId": entry.slitId,
            "sourceId": str(bundle["source_id"]),
            "width": int(bundle["width"]),
            **primary,
            "additionalMaps": maps[1:],
            "cacheHit": all(cache_hits),
            "wallSeconds": float(sum(wall_times)),
        }
    return {"results": responses}


@app.post("/api/sessions/{session_id}/slits/cancel")
def cancel_slit_extraction(session_id: str) -> dict[str, bool]:
    """Request cooperative cancellation of the active slit extraction."""
    _session(session_id).cancel_slit_extraction()
    return {"cancelled": True}


@app.post("/api/sessions/{session_id}/slits/{slit_id}/reverse")
def reverse_slit_direction(session_id: str, slit_id: str) -> dict[str, bool]:
    """Flip a cached slit result onto its opposite distance orientation."""
    try:
        _session(session_id).reverse_slit_result(slit_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"reversed": True}


@app.get("/api/sessions/{session_id}/slits/{slit_id}/export.npz")
def export_slit_npz(
    session_id: str,
    slit_id: str,
    shiftSeconds: float = 0.0,
) -> Response:
    """Download one extracted slit with its display shift and metadata."""
    if not math.isfinite(shiftSeconds):
        raise HTTPException(status_code=422, detail="shiftSeconds must be finite")
    session = _session(session_id)
    try:
        content = session.slit_result_npz(slit_id, shiftSeconds)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in slit_id)
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="time_distance_{safe_name}.npz"'},
    )


@app.get("/api/sessions/{session_id}/exports/{name}")
def export_file(session_id: str, name: str):
    session = _session(session_id)
    try:
        path = session.export_path(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown export: {name}") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Export has not been generated yet: {name}")
    media_type = "text/csv" if path.suffix == ".csv" else "image/png"
    return FileResponse(path, media_type=media_type, filename=path.name)


def configure_frontend(application: FastAPI, web_root: Path = WEB_ROOT) -> None:
    """Serve the bundled frontend when a release includes built web assets.

    A source checkout can run the backend before the frontend has been built;
    in that case the historical JSON response at ``/`` remains available.
    API routes are registered separately and are never handled by the
    frontend's static fallback.
    """
    root = Path(web_root)
    assets = root / "assets"
    if assets.is_dir():
        application.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    def index() -> FileResponse | dict[str, str]:
        index_path = root / "index.html"
        if index_path.is_file():
            return FileResponse(index_path, media_type="text/html")
        return {"app": "SolRadViewer", "frontend": "http://127.0.0.1:5174"}

    application.add_api_route("/", index, include_in_schema=False, response_model=None)


configure_frontend(app)
