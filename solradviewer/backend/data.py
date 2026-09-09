"""Data loading, rendering, and feature extraction for SolRadViewer."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import time
import uuid
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait as wait_futures
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread, local

import astropy.units as u
import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from matplotlib import colormaps
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass, map_coordinates, minimum_filter
from skimage.feature import match_template
from skimage import measure
from sunpy.map import Map

try:  # Registers SDO/AIA colormaps when available.
    import sunpy.visualization.colormaps  # noqa: F401
except Exception:  # pragma: no cover - display still works with standard maps.
    pass


PROJECT_ROOT = Path.cwd()
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "EOVSA_20220118_Mflare"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "solradviewer"


def _first_environment_value(*names: str) -> str | None:
    """Return the first non-empty environment value in ``names`` order."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _configured_path(default: Path, *environment_names: str) -> Path:
    """Resolve a path from environment configuration, expanding ``~``."""
    value = _first_environment_value(*environment_names)
    return Path(value).expanduser() if value is not None else Path(default).expanduser()


# The sample defaults are opt-in data locations.  A fresh checkout can start
# without any science files; manifests remain the normal way to load data.
DATA_ROOT = _configured_path(
    DEFAULT_DATA_ROOT,
    "SOLRADVIEWER_DATA_ROOT",
)
DEFAULT_AIA_INTENSITY = DATA_ROOT / "AIA/mapseq_AIA/ssw_cutout/mapseq_20220118_1705-1805UT_131_bin1.h5"
DEFAULT_AIA_DIFF = DATA_ROOT / "AIA/mapseq_AIA/ssw_cutout/mapseq_20220118_1705-1805UT_131_bin1_rratio.h5"
DEFAULT_SEEDS = DATA_ROOT / "AIA/SADs_pos_from_Xiaoyan/markpos.pickle"
DEFAULT_EOVSA_DIR = DATA_ROOT / "EOVSA/qlookallbdfits_1s_1719-1745UT"
DEFAULT_EOVSA_SPECTROGRAM = DATA_ROOT / "EOVSA/eovsa.spec_xp.flare_id_202201181736.fits"
DEFAULT_OUTPUT_ROOT = _configured_path(
    PROJECT_ROOT / "outputs",
    "SOLRADVIEWER_OUTPUT_ROOT",
)
# ``SOLRADVIEWER_CACHE_DIR`` is the direct cache directory.  The existing
# legacy render-cache name remains accepted so local launch configurations keep
# working.
CACHE_ROOT = _configured_path(
    DEFAULT_CACHE_ROOT,
    "SOLRADVIEWER_CACHE_DIR",
    "SAD_EOVSA_RENDER_CACHE_DIR",
)
# Retain the old module symbol for callers that inspected the cache location.
EXTERNAL_CACHE_ROOT = CACHE_ROOT
AIA_DISPLAY_ORIENTATION = "solar"
EOVSA_DISPLAY_ORIENTATION = "solar"
TEXTURE_CACHE_LIMIT = 384
DATA_CACHE_LIMIT = 32
EOVSA_DATA_CACHE_LIMIT = 32
# A decoded 256x256 float32 radio plane is roughly 256 KiB.  This budget
# retains about 256 planes (several complete 52-channel cubes) for contour
# scrubbing without allowing the interactive process to grow unbounded.
EOVSA_PLANE_CACHE_BYTES = 64 * 1024**2
WARM_DATA_CACHE_LIMIT = 64
MEAN_CACHE_LIMIT = 8
DEFAULT_DIFF_SECONDS = 60.0
DEFAULT_TEMPORAL_SIGMA_SHORT = 12.0
DEFAULT_TEMPORAL_SIGMA_LONG = 120.0
# Perpendicular slit-averaging width upper bound, in source pixels. Was a
# hardcoded 15 (a reasonable ceiling for a spine-hugging profile), raised to
# cover the full field of view of the largest bound source (AIA, ~1667px)
# with headroom - see the frontend's per-slit max (sourceMaxWidthPx in
# App.tsx), which derives its own cap from the bound source's actual pixel
# shape and is always <= this value. Extraction cost scales linearly with
# width (one perpendicular sample averaged per curve point per unit width),
# so a large width is legitimate but slower; this bound only guards against
# malformed/out-of-range requests, not a usability limit.
MAX_SLIT_WIDTH_PX = 4096
RADIAL_R_MAX = 2.5
TEMPORAL_SHORT_FRAME_RADIUS = 15
TEMPORAL_LONG_SAMPLE_LIMIT = 25
TEMPORAL_CACHE_LIMIT = 96
# A temporal output reuses nearly the entire +/-3-sigma window of processed
# source frames.  Keep a separate bounded cache for those frames so the
# ordinary display-data LRU retains its existing semantics and size.
TEMPORAL_RAW_FRAME_CACHE_LIMIT = 64
TEMPORAL_WORKERS = 6
# Cache warming is deliberately less parallel than interactive rendering.  Two
# workers keep FITS decompression moving without monopolising the server CPU.
PREWARM_WORKERS = 2
MAX_SPECTROGRAM_TIME_COLUMNS = 8192
DEFAULT_RENDER_CACHE_BYTES = 2 * 1024**3
# The decoded-plane store holds one whole decoded radio cube per native FITS
# time step (all frequency channels, ~13 MB each for a 52 x 256 x 256 float32
# EOVSA cube). A full-range extraction over a multi-hour flare touches every
# native time step at least once, so the budget must cover the FULL dataset's
# working set (frames x cube bytes) or the store evicts its own oldest
# entries mid-pass and every subsequent pass re-decodes from FITS instead of
# hitting the store -- see docs/design/fan-geometry-td.md's "one decompression
# per cube ever" invariant, which only holds if the budget actually fits the
# dataset. 24 GiB comfortably covers the reference EOVSA session used to
# diagnose this (1456 frames x ~13 MB ~= 19 GiB working set) with headroom for
# larger datasets; override with SAD_EOVSA_DECODED_STORE_GB for bigger cubes.
DEFAULT_DECODED_STORE_GB = 24
DEFAULT_DECODED_STORE_BYTES = DEFAULT_DECODED_STORE_GB * 1024**3
SAD_FIELDS = [
    "sad_id", "time_utc", "time_mjd", "frame_index", "x_arcsec", "y_arcsec",
    "x_pix", "y_pix", "vx_arcsec_s", "vy_arcsec_s", "speed_km_s", "quality",
]
EOVSA_FIELDS = [
    "time_utc", "time_mjd", "eovsa_index", "spw_index", "freq_ghz",
    "x_peak_arcsec", "y_peak_arcsec", "x_centroid_arcsec", "y_centroid_arcsec",
    "x_peak_pix", "y_peak_pix", "x_centroid_pix", "y_centroid_pix",
    "x_peak_display_pix", "y_peak_display_pix", "x_centroid_display_pix", "y_centroid_display_pix",
    "peak_tb", "snr", "accepted", "x_offset_arcsec", "y_offset_arcsec",
]
FEATURE_FIELDS = [
    "track_id", "source_id", "time_utc", "time_mjd", "frame_index",
    "x_arcsec", "y_arcsec", "x_pix", "y_pix", "quality", "score", "roi_id",
]
TRACKING_FIELDS = [
    "track_id", "label", "source_id", "frame_index", "time_utc", "time_mjd",
    "x_px", "y_px", "x_arcsec", "y_arcsec", "confidence", "is_anchor",
    "vx_arcsec_s", "vy_arcsec_s", "speed_arcsec_s", "speed_km_s",
    "dist_to_target_arcsec", "arrival_utc",
]


_PROGRESS_UNCHANGED = object()


class ProgressRegistry:
    """Thread-safe registry of operations currently active for one session."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._operations: dict[str, dict[str, object]] = {}

    def start(self, label: str, total: int | None = None) -> str:
        """Register an operation and return its opaque identifier."""
        op_id = uuid.uuid4().hex
        normalized_total = None if total is None else max(0, int(total))
        with self._lock:
            self._operations[op_id] = {
                "opId": op_id,
                "label": str(label),
                "done": 0,
                "total": normalized_total,
                "startedAt": time.time(),
            }
        return op_id

    def update(
        self,
        op_id: str,
        *,
        done: int | None = None,
        total: int | None | object = _PROGRESS_UNCHANGED,
    ) -> None:
        """Update an active operation if it has not already finished."""
        with self._lock:
            operation = self._operations.get(op_id)
            if operation is None:
                return
            if total is not _PROGRESS_UNCHANGED:
                operation["total"] = None if total is None else max(0, int(total))
            if done is not None:
                normalized_done = max(0, int(done))
                operation_total = operation["total"]
                if isinstance(operation_total, int):
                    normalized_done = min(normalized_done, operation_total)
                operation["done"] = normalized_done

    def advance(self, op_id: str, amount: int = 1) -> None:
        """Atomically advance an active operation's completed count."""
        with self._lock:
            operation = self._operations.get(op_id)
            if operation is None:
                return
            normalized_done = max(0, int(operation["done"]) + int(amount))
            operation_total = operation["total"]
            if isinstance(operation_total, int):
                normalized_done = min(normalized_done, operation_total)
            operation["done"] = normalized_done

    def finish(self, op_id: str) -> None:
        """Remove a completed or cancelled operation."""
        with self._lock:
            self._operations.pop(op_id, None)

    def snapshot(self) -> list[dict[str, object]]:
        """Return detached active-operation records ordered by start time."""
        with self._lock:
            operations = [dict(operation) for operation in self._operations.values()]
        return sorted(operations, key=lambda operation: float(operation["startedAt"]))


class RenderDiskCache:
    """Thread-safe byte-budgeted LRU cache for encoded render payloads."""

    def __init__(self, directory: Path, max_bytes: int = DEFAULT_RENDER_CACHE_BYTES) -> None:
        self.directory = Path(directory)
        self.max_bytes = max(0, int(max_bytes))
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._index_ready = False
        self._sizes: dict[Path, tuple[int, int]] = {}
        self._total_bytes = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(identity: object) -> str:
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str, extension: str = "png") -> Path:
        """Return the cache path for one encoded payload.

        :param key: Stable hexadecimal cache key.
        :type key: str
        :param extension: Payload filename extension without a leading dot.
        :type extension: str
        :returns: Cache entry path.
        :rtype: pathlib.Path
        """
        suffix = str(extension).strip().lstrip(".") or "png"
        return self.directory / key[:2] / f"{key}.{suffix}"

    def _entries(self) -> list[tuple[int, int, Path]]:
        entries: list[tuple[int, int, Path]] = []
        for path in self.directory.glob("*/*"):
            if not path.is_file() or path.name.startswith("."):
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            entries.append((int(stat.st_mtime_ns), int(stat.st_size), path))
        return entries

    def _ensure_index(self) -> None:
        """Build the on-disk size index once instead of rescanning per write."""
        if self._index_ready:
            return
        entries = self._entries()
        self._sizes = {path: (mtime_ns, size) for mtime_ns, size, path in entries}
        self._total_bytes = sum(size for _, size in self._sizes.values())
        self._index_ready = True

    def get(self, key: str, extension: str = "png") -> bytes | None:
        """Read one payload and update its LRU timestamp.

        :param key: Stable hexadecimal cache key.
        :type key: str
        :param extension: Payload filename extension without a leading dot.
        :type extension: str
        :returns: Encoded payload, or ``None`` on a cache miss.
        :rtype: bytes or None
        """
        path = self._path(key, extension)
        with self._lock:
            self._ensure_index()
            try:
                content = path.read_bytes()
                os.utime(path, None)
                try:
                    stat = path.stat()
                    self._sizes[path] = (int(stat.st_mtime_ns), int(stat.st_size))
                except FileNotFoundError:
                    self._sizes.pop(path, None)
            except FileNotFoundError:
                self._misses += 1
                return None
            self._hits += 1
            return content

    def put(
        self,
        key: str,
        content: bytes,
        extension: str = "png",
        *,
        durable: bool = True,
    ) -> bytes:
        """Atomically store one payload and enforce the shared byte budget.

        :param key: Stable hexadecimal cache key.
        :type key: str
        :param content: Encoded payload bytes.
        :type content: bytes
        :param extension: Payload filename extension without a leading dot.
        :type extension: str
        :param durable: Flush the temporary file to stable storage before the
            atomic replace. Warm-cache writes may disable this durability step;
            the tempfile-plus-replace single-writer guarantee is retained.
        :type durable: bool
        :returns: The stored payload.
        :rtype: bytes
        """
        path = self._path(key, extension)
        with self._lock:
            self._ensure_index()
            previous = self._sizes.pop(path, None)
            if previous is not None:
                self._total_bytes -= previous[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as handle:
                    handle.write(content)
                    handle.flush()
                    if durable:
                        os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            try:
                stat = path.stat()
                self._sizes[path] = (int(stat.st_mtime_ns), int(stat.st_size))
                self._total_bytes += int(stat.st_size)
            except FileNotFoundError:
                pass
            entries = sorted(
                ((mtime_ns, size, candidate) for candidate, (mtime_ns, size) in self._sizes.items()),
                key=lambda item: (item[0], str(item[2])),
            )
            for _, size, candidate in entries:
                if self._total_bytes <= self.max_bytes:
                    break
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    self._sizes.pop(candidate, None)
                    continue
                self._sizes.pop(candidate, None)
                self._total_bytes -= size
        return content

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._ensure_index()
            entries = self._entries()
            return {
                "hits": self._hits,
                "misses": self._misses,
                "bytes": sum(size for _, size, _ in entries),
                "entries": len(entries),
                "maxBytes": self.max_bytes,
            }


class DecodedPlaneStore:
    """Thread-safe byte-budgeted LRU store for memmappable float32 arrays."""

    def __init__(self, directory: Path, max_bytes: int = DEFAULT_DECODED_STORE_BYTES) -> None:
        """Create a decoded-array store.

        :param directory: Directory containing hashed ``.npy`` entries.
        :type directory: pathlib.Path
        :param max_bytes: Maximum on-disk bytes retained by LRU eviction.
        :type max_bytes: int
        """
        self.directory = Path(directory)
        self.max_bytes = max(0, int(max_bytes))
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._index_ready = False
        self._sizes: dict[Path, tuple[int, int]] = {}
        self._total_bytes = 0
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def source_key(path: Path, decoder: object = None) -> str:
        """Return a cache key invalidated by path, size, and nanosecond mtime.

        :param path: Source file supplying the decoded array.
        :type path: pathlib.Path
        :param decoder: Optional HDU/shape discriminator for the decoder.
        :type decoder: object
        :returns: Stable hexadecimal cache key.
        :rtype: str
        """
        source = Path(path)
        stat = source.stat()
        return RenderDiskCache.make_key({
            "version": 1,
            "path": str(source.resolve()),
            "size": int(stat.st_size),
            "mtimeNs": int(stat.st_mtime_ns),
            "decoder": decoder,
            "dtype": "float32",
        })

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.npy"

    def _entries(self) -> list[tuple[int, int, Path]]:
        entries: list[tuple[int, int, Path]] = []
        for path in self.directory.glob("*/*.npy"):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            entries.append((int(stat.st_mtime_ns), int(stat.st_size), path))
        return entries

    def _ensure_index(self) -> None:
        if self._index_ready:
            return
        entries = self._entries()
        self._sizes = {path: (mtime_ns, size) for mtime_ns, size, path in entries}
        self._total_bytes = sum(size for _, size in self._sizes.values())
        self._index_ready = True

    def _drop(self, path: Path) -> None:
        previous = self._sizes.pop(path, None)
        if previous is not None:
            self._total_bytes -= previous[1]
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _open(path: Path, expected_shape: tuple[int, ...] | None) -> np.memmap | None:
        try:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
        except (EOFError, OSError, ValueError):
            return None
        if not isinstance(value, np.memmap) or value.dtype != np.float32:
            if isinstance(value, np.memmap):
                value._mmap.close()
            return None
        if expected_shape is not None and tuple(int(size) for size in value.shape) != expected_shape:
            value._mmap.close()
            return None
        return value

    @staticmethod
    def _materialize(value: np.memmap) -> np.ndarray:
        """Copy a memory map into the bounded source LRU, then close its file.

        :param value: Read-only decoded-store memory map.
        :type value: numpy.memmap
        :returns: Detached float32 array.
        :rtype: numpy.ndarray
        """
        try:
            return np.array(value, dtype=np.float32, copy=True)
        finally:
            value._mmap.close()

    def get(self, key: str, expected_shape: tuple[int, ...] | None = None) -> np.memmap | None:
        """Open one decoded array through NumPy memory mapping.

        :param key: Stable source-signature key.
        :type key: str
        :param expected_shape: Optional exact array shape validation.
        :type expected_shape: tuple of int or None
        :returns: Read-only float32 memory map, or ``None`` on a miss.
        :rtype: numpy.memmap or None
        """
        path = self._path(key)
        with self._lock:
            self._ensure_index()
            value = self._open(path, expected_shape)
            if value is None:
                if path.exists():
                    self._drop(path)
                self._misses += 1
                return None
            try:
                os.utime(path, None)
                stat = path.stat()
                self._sizes[path] = (int(stat.st_mtime_ns), int(stat.st_size))
            except FileNotFoundError:
                value._mmap.close()
                self._misses += 1
                return None
            self._hits += 1
            return value

    def read(self, key: str, expected_shape: tuple[int, ...] | None = None) -> np.ndarray | None:
        """Read through a short-lived memory map and close it immediately.

        :param key: Stable source-signature key.
        :type key: str
        :param expected_shape: Optional exact array shape validation.
        :type expected_shape: tuple of int or None
        :returns: Detached float32 array, or ``None`` on a miss.
        :rtype: numpy.ndarray or None
        """
        mapped = self.get(key, expected_shape)
        return None if mapped is None else self._materialize(mapped)

    def put(self, key: str, data: np.ndarray) -> np.memmap:
        """Atomically store an array and return the new read-only memory map.

        Writes use a same-directory temporary file plus ``os.replace``. The
        temporary file is flushed through Python but deliberately not fsynced,
        matching warm render-cache writes.

        :param key: Stable source-signature key.
        :type key: str
        :param data: Decoded source array.
        :type data: numpy.ndarray
        :returns: Stored read-only float32 memory map.
        :rtype: numpy.memmap
        """
        value = np.asarray(data, dtype=np.float32)
        path = self._path(key)
        with self._lock:
            self._ensure_index()
            existing = self._open(path, tuple(int(size) for size in value.shape))
            if existing is not None:
                os.utime(path, None)
                stat = path.stat()
                self._sizes[path] = (int(stat.st_mtime_ns), int(stat.st_size))
                return existing
            previous = self._sizes.pop(path, None)
            if previous is not None:
                self._total_bytes -= previous[1]
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with temporary.open("xb") as handle:
                    np.save(handle, value, allow_pickle=False)
                    handle.flush()
                os.replace(temporary, path)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            stat = path.stat()
            self._sizes[path] = (int(stat.st_mtime_ns), int(stat.st_size))
            self._total_bytes += int(stat.st_size)
            entries = sorted(
                ((mtime_ns, size, candidate) for candidate, (mtime_ns, size) in self._sizes.items()),
                key=lambda item: (item[0], str(item[2])),
            )
            for _, size, candidate in entries:
                if self._total_bytes <= self.max_bytes:
                    break
                if candidate == path and len(entries) > 1:
                    continue
                self._drop(candidate)
            stored = self._open(path, tuple(int(size) for size in value.shape))
            if stored is None:
                raise OSError(f"Decoded cache entry was evicted before opening: {path}")
            return stored

    def write(self, key: str, data: np.ndarray) -> np.ndarray:
        """Persist an array, then render from a short-lived store memory map.

        :param key: Stable source-signature key.
        :type key: str
        :param data: Newly decoded float32 array.
        :type data: numpy.ndarray
        :returns: Detached float32 array read back from the store.
        :rtype: numpy.ndarray
        """
        return self._materialize(self.put(key, data))

    def stats(self) -> dict[str, int]:
        """Return cache counters and current on-disk accounting.

        :returns: Hit, miss, byte, entry, and budget counters.
        :rtype: dict
        """
        with self._lock:
            self._ensure_index()
            return {
                "hits": self._hits,
                "misses": self._misses,
                "bytes": self._total_bytes,
                "entries": len(self._sizes),
                "maxBytes": self.max_bytes,
            }


def _configured_render_cache_bytes() -> int:
    raw = _first_environment_value(
        "SOLRADVIEWER_RENDER_CACHE_BYTES",
        "SAD_EOVSA_RENDER_CACHE_BYTES",
    )
    if raw is None:
        return DEFAULT_RENDER_CACHE_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RENDER_CACHE_BYTES


def _configured_decoded_store_bytes() -> int:
    """Resolve the decoded-plane store's byte budget from the environment.

    ``SOLRADVIEWER_DECODED_STORE_GB`` (whole or fractional GiB) is the primary
    override, sized to fit a dataset's full working set -- see
    :data:`DEFAULT_DECODED_STORE_GB`. The ``SAD_EOVSA_*`` names remain
    supported for compatibility.

    :returns: Byte budget, never negative.
    :rtype: int
    """
    gib_raw = _first_environment_value(
        "SOLRADVIEWER_DECODED_STORE_GB",
        "SAD_EOVSA_DECODED_STORE_GB",
    )
    if gib_raw is not None:
        try:
            return max(0, int(float(gib_raw) * 1024**3))
        except ValueError:
            pass
    raw = _first_environment_value(
        "SOLRADVIEWER_DECODED_CACHE_BYTES",
        "SAD_EOVSA_DECODED_CACHE_BYTES",
    )
    if raw is None:
        return DEFAULT_DECODED_STORE_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_DECODED_STORE_BYTES


def _default_render_cache_directory() -> Path:
    """Return the configured cache directory, defaulting to user-local storage."""
    return CACHE_ROOT


RENDER_DISK_CACHE = RenderDiskCache(
    _default_render_cache_directory(),
    _configured_render_cache_bytes(),
)
DECODED_PLANE_STORE = DecodedPlaneStore(
    RENDER_DISK_CACHE.directory / "decoded-planes",
    _configured_decoded_store_bytes(),
)
TEMPORAL_EXECUTOR = ThreadPoolExecutor(max_workers=TEMPORAL_WORKERS, thread_name_prefix="sad-temporal")


class OverlayUnavailableError(RuntimeError):
    """Raised when an overlay cannot be computed for the requested recipe."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


class SlitExtractionCancelled(RuntimeError):
    """Raised when a cooperative time-distance extraction is cancelled."""


def _resample_slit_world_curve(
    curve_arcsec: np.ndarray,
    reference_pixels: np.ndarray,
    spacing_pixels: float = 1.0,
) -> np.ndarray:
    """Resample a world-coordinate polyline at uniform reference-pixel spacing.

    :param curve_arcsec: Input ``N x 2`` solar-coordinate polyline.
    :type curve_arcsec: numpy.ndarray
    :param reference_pixels: The same vertices projected into a reference frame.
    :type reference_pixels: numpy.ndarray
    :param spacing_pixels: Desired sample spacing in reference image pixels.
    :type spacing_pixels: float
    :returns: Uniformly sampled solar-coordinate vertices.
    :rtype: numpy.ndarray
    :raises ValueError: If the curve has fewer than two finite, distinct points.
    """
    world = np.asarray(curve_arcsec, dtype=float).reshape(-1, 2)
    pixels = np.asarray(reference_pixels, dtype=float).reshape(-1, 2)
    finite = np.isfinite(world).all(axis=1) & np.isfinite(pixels).all(axis=1)
    world = world[finite]
    pixels = pixels[finite]
    if world.shape[0] < 2:
        raise ValueError("Slit curve must contain at least two finite vertices")
    segment_lengths = np.linalg.norm(np.diff(pixels, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    world = world[keep]
    cumulative = cumulative[keep]
    if world.shape[0] < 2 or cumulative[-1] <= 0:
        raise ValueError("Slit curve must span at least one image pixel")
    spacing = max(0.25, float(spacing_pixels))
    samples = np.arange(0.0, cumulative[-1], spacing, dtype=float)
    if samples.size == 0 or samples[-1] < cumulative[-1]:
        samples = np.r_[samples, cumulative[-1]]
    return np.column_stack([
        np.interp(samples, cumulative, world[:, axis]) for axis in range(2)
    ])


def resample_polyline_count(curve: object, count: int) -> np.ndarray:
    """Resample a two-dimensional polyline to an exact vertex count.

    :param curve: Input ``N x 2`` polyline.
    :type curve: array-like
    :param count: Requested output vertex count, at least two.
    :type count: int
    :returns: Arc-length-resampled ``count x 2`` vertices.
    :rtype: numpy.ndarray
    :raises ValueError: If the input curve is not finite and non-degenerate.
    """
    points = np.asarray(curve, dtype=float).reshape(-1, 2)
    if points.shape[0] < 2 or not np.isfinite(points).all():
        raise ValueError("Fan boundaries require at least two finite vertices")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    keep = np.r_[True, np.diff(cumulative) > 1e-9]
    points = points[keep]
    cumulative = cumulative[keep]
    if points.shape[0] < 2 or cumulative[-1] <= 0:
        raise ValueError("Fan boundaries must be non-degenerate")
    samples = np.linspace(0.0, cumulative[-1], max(2, int(count)))
    return np.column_stack([
        np.interp(samples, cumulative, points[:, axis]) for axis in range(2)
    ])


def fan_family_curves(
    boundary_a: object,
    boundary_b: object,
    intermediate_count: int = 3,
) -> tuple[list[np.ndarray], bool]:
    """Construct an endpoint-aligned ruled family between two boundaries.

    Boundary B is reversed when that minimizes the two paired endpoint
    distances.  Every returned member uses the oriented boundary-A direction.

    :param boundary_a: First sampled boundary vertices.
    :type boundary_a: array-like
    :param boundary_b: Second sampled boundary vertices.
    :type boundary_b: array-like
    :param intermediate_count: Number of equally spaced interior curves.
    :type intermediate_count: int
    :returns: Family including both boundaries, and whether B was reversed.
    :rtype: tuple[list[numpy.ndarray], bool]
    """
    a_input = np.asarray(boundary_a, dtype=float).reshape(-1, 2)
    b_input = np.asarray(boundary_b, dtype=float).reshape(-1, 2)
    count = max(2, a_input.shape[0], b_input.shape[0])
    a = resample_polyline_count(a_input, count)
    b = resample_polyline_count(b_input, count)
    forward = np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[-1] - b[-1])
    reverse = np.linalg.norm(a[0] - b[-1]) + np.linalg.norm(a[-1] - b[0])
    reversed_b = bool(reverse < forward)
    if reversed_b:
        b = b[::-1].copy()
    members = max(0, int(intermediate_count)) + 2
    return [a + (b - a) * fraction for fraction in np.linspace(0.0, 1.0, members)], reversed_b


def _slit_sample_grid(curve_pixels: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Precompute the flat bilinear sample coordinates for a slit curve.

    The tangent/normal/offset geometry depends only on the curve's pixel
    coordinates, not on any particular frame's data.  Callers that reuse the
    same projected curve across many frames (e.g. one radio channel sampled
    at every native time step) should compute this once and reuse it, rather
    than recomputing it per frame inside :func:`sample_slit_profile`.

    :param curve_pixels: Uniform slit samples in source pixel coordinates.
    :type curve_pixels: numpy.ndarray
    :param width: Number of perpendicular one-pixel samples, from 1 through MAX_SLIT_WIDTH_PX.
    :type width: int
    :returns: Flat ``sample_y``, flat ``sample_x``, the clamped width, and the
        number of along-slit pixels.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, int, int]
    :raises ValueError: If the curve has fewer than two points.
    """
    points = np.asarray(curve_pixels, dtype=float).reshape(-1, 2)
    if points.shape[0] < 2:
        raise ValueError("Slit sampling requires at least two curve points")
    normalized_width = int(np.clip(int(width), 1, MAX_SLIT_WIDTH_PX))
    tangent = np.gradient(points, axis=0)
    length = np.linalg.norm(tangent, axis=1)
    length[length < 1e-9] = 1.0
    normals = np.column_stack([-tangent[:, 1] / length, tangent[:, 0] / length])
    offsets = np.arange(normalized_width, dtype=float) - (normalized_width - 1) / 2.0
    sample_x = points[:, 0][None, :] + offsets[:, None] * normals[:, 0][None, :]
    sample_y = points[:, 1][None, :] + offsets[:, None] * normals[:, 1][None, :]
    return sample_y.reshape(-1), sample_x.reshape(-1), normalized_width, points.shape[0]


def _reduce_slit_samples(sampled_flat: np.ndarray, width: int, npix: int) -> np.ndarray:
    """Collapse a flat perpendicular-width bilinear sample back to one profile.

    :param sampled_flat: ``map_coordinates`` output for a ``(width, npix)`` grid.
    :type sampled_flat: numpy.ndarray
    :param width: Number of perpendicular samples used to build the grid.
    :type width: int
    :param npix: Number of along-slit pixels used to build the grid.
    :type npix: int
    :returns: One intensity value per slit-distance sample.
    :rtype: numpy.ndarray
    """
    sampled = np.asarray(sampled_flat).reshape(width, npix)
    finite_count = np.sum(np.isfinite(sampled), axis=0)
    summed = np.nansum(sampled, axis=0)
    return np.divide(
        summed,
        finite_count,
        out=np.full(npix, np.nan, dtype=float),
        where=finite_count > 0,
    ).astype(np.float32)


def sample_slit_profile(frame: np.ndarray, curve_pixels: np.ndarray, width: int = 3) -> np.ndarray:
    """Bilinearly sample one frame along a slit with perpendicular averaging.

    :param frame: Processed two-dimensional source frame.
    :type frame: numpy.ndarray
    :param curve_pixels: Uniform slit samples in source pixel coordinates.
    :type curve_pixels: numpy.ndarray
    :param width: Number of perpendicular one-pixel samples, from 1 through MAX_SLIT_WIDTH_PX.
    :type width: int
    :returns: One intensity value per slit-distance sample.
    :rtype: numpy.ndarray
    :raises ValueError: If the frame or curve geometry is invalid.
    """
    values = np.asarray(frame, dtype=float)
    if values.ndim != 2:
        raise ValueError("Slit sampling requires a 2-D frame and at least two curve points")
    sample_y, sample_x, normalized_width, npix = _slit_sample_grid(curve_pixels, width)
    sampled = map_coordinates(
        values,
        [sample_y, sample_x],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    return _reduce_slit_samples(sampled, normalized_width, npix)


def extract_time_distance_map(
    frame_getter: object,
    times_mjd: object,
    curve_arcsec: object,
    world_to_pixel: object,
    width: int = 3,
    on_progress: object | None = None,
    is_cancelled: object | None = None,
) -> dict[str, np.ndarray]:
    """Extract a native-cadence time-distance map from processed image frames.

    :param frame_getter: Callable returning one processed 2-D frame by index.
    :type frame_getter: callable
    :param times_mjd: Native source timestamps in MJD.
    :type times_mjd: array-like
    :param curve_arcsec: Slit polyline in solar arcseconds.
    :type curve_arcsec: array-like
    :param world_to_pixel: Callable projecting ``(frame_index, world_points)``.
    :type world_to_pixel: callable
    :param width: Perpendicular averaging width in image pixels.
    :type width: int
    :param on_progress: Optional ``(done, total)`` callback.
    :type on_progress: callable or None
    :param is_cancelled: Optional zero-argument cooperative cancellation check.
    :type is_cancelled: callable or None
    :returns: Intensity, distance, time, and uniformly sampled curve arrays.
    :rtype: dict[str, numpy.ndarray]
    :raises SlitExtractionCancelled: If cancellation is requested.
    """
    if not callable(frame_getter) or not callable(world_to_pixel):
        raise ValueError("Slit extraction requires frame and WCS callables")
    times = np.asarray(times_mjd, dtype=float).reshape(-1)
    if times.size == 0:
        raise ValueError("Slit source has no native frames")
    input_curve = np.asarray(curve_arcsec, dtype=float).reshape(-1, 2)
    reference_pixels = np.asarray(world_to_pixel(0, input_curve), dtype=float)
    sampled_curve = _resample_slit_world_curve(input_curve, reference_pixels)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(sampled_curve, axis=0), axis=1))]
    intensity = np.empty((sampled_curve.shape[0], times.size), dtype=np.float32)
    for index in range(times.size):
        if callable(is_cancelled) and bool(is_cancelled()):
            raise SlitExtractionCancelled("Time-distance extraction cancelled")
        pixels = np.asarray(world_to_pixel(index, sampled_curve), dtype=float)
        intensity[:, index] = sample_slit_profile(
            np.asarray(frame_getter(index), dtype=float), pixels, width
        )
        if callable(on_progress):
            on_progress(index + 1, int(times.size))
    return {
        "intensity": intensity,
        "distance_arcsec": distance.astype(np.float32),
        "time_mjd": times.astype(np.float64),
        "curve_vertices_arcsec": sampled_curve.astype(np.float64),
    }


def _encode_slit_result_npz(
    result: dict[str, object],
    source_id: str,
    layer_params: dict[str, object],
    shift_seconds: float = 0.0,
) -> bytes:
    """Encode one time-distance result using stable, pickle-free NPZ fields."""
    maps = result.get("maps")
    map_results = list(maps) if isinstance(maps, list) and maps else [result]
    intensity_maps = [np.asarray(item["intensity"], dtype=np.float32) for item in map_results]
    multi_frequency = len(intensity_maps) > 1
    intensity = np.stack(intensity_maps) if multi_frequency else intensity_maps[0]
    freq_ghz = np.asarray(result.get("freq_ghz", []), dtype=np.float64).reshape(-1)
    output = BytesIO()
    np.savez_compressed(
        output,
        intensity=intensity,
        distance_arcsec=np.asarray(map_results[0]["distance_arcsec"], dtype=np.float32),
        time_mjd=np.asarray(map_results[0]["time_mjd"], dtype=np.float64),
        applied_shift_seconds=np.asarray(float(shift_seconds), dtype=np.float64),
        curve_vertices_arcsec=np.asarray(map_results[0]["curve_vertices_arcsec"], dtype=np.float64),
        layer_params_snapshot=np.asarray(json.dumps(layer_params, sort_keys=True, separators=(",", ":"))),
        source_id=np.asarray(str(source_id)),
        freq_ghz=freq_ghz,
        layout=np.asarray("frequency_distance_time" if multi_frequency else "distance_time"),
    )
    return output.getvalue()


def _decode_slit_result_npz(content: bytes) -> dict[str, np.ndarray]:
    """Decode a cached, pickle-free time-distance NPZ payload."""
    with np.load(BytesIO(content), allow_pickle=False) as archive:
        return {
            "intensity": np.asarray(archive["intensity"], dtype=np.float32),
            "distance_arcsec": np.asarray(archive["distance_arcsec"], dtype=np.float32),
            "time_mjd": np.asarray(archive["time_mjd"], dtype=np.float64),
            "curve_vertices_arcsec": np.asarray(archive["curve_vertices_arcsec"], dtype=np.float64),
        }


def _canonical_slit_curve(curve_arcsec: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a direction-independent curve and whether the request is reversed."""
    curve = np.asarray(curve_arcsec, dtype=float).reshape(-1, 2)
    rounded = np.round(curve, 8)
    reversed_is_canonical = tuple(rounded[::-1].reshape(-1)) < tuple(rounded.reshape(-1))
    return (curve[::-1].copy(), True) if reversed_is_canonical else (curve.copy(), False)


def _reverse_slit_result(result: dict[str, object]) -> dict[str, object]:
    """Flip one extracted result onto the opposite distance orientation."""
    maps = result.get("maps")
    if isinstance(maps, list) and maps:
        flipped_maps = [_reverse_slit_result(dict(item)) for item in maps]
        flipped = {**result, **flipped_maps[0], "maps": flipped_maps}
        flipped["freq_indices"] = list(result.get("freq_indices", []))
        flipped["freq_ghz"] = list(result.get("freq_ghz", []))
        return flipped
    distance = np.asarray(result["distance_arcsec"], dtype=np.float32)
    distance_max = float(distance[-1]) if distance.size else 0.0
    flipped = dict(result)
    flipped["intensity"] = np.asarray(result["intensity"], dtype=np.float32)[::-1].copy()
    flipped["distance_arcsec"] = (distance_max - distance[::-1]).astype(np.float32)
    flipped["curve_vertices_arcsec"] = np.asarray(
        result["curve_vertices_arcsec"], dtype=np.float64
    )[::-1].copy()
    return flipped


def _sfu_to_tb_thresholds(level_sfu: float, freqs_hz: object, header: fits.Header) -> np.ndarray:
    """Convert a flux-density contour level to per-band brightness temperatures.

    EOVSA image pixels use the FITS ``CDELT1``/``CDELT2`` scale in arcsec.
    The Rayleigh-Jeans conversion is
    ``T_thresh(nu_i) = S_sfu * 1e-19 * c^2 / (2 * k_B * nu_i^2 * Omega_pix)``
    in cgs units, where ``Omega_pix = (CDELT_arcsec * pi / 180 / 3600)^2``.

    :param level_sfu: Flux density per pixel in solar flux units.
    :type level_sfu: float
    :param freqs_hz: Per-band center frequencies in Hz.
    :type freqs_hz: object
    :param header: EOVSA image FITS header containing the pixel scale.
    :type header: astropy.io.fits.Header
    :returns: One brightness-temperature threshold in K per frequency.
    :rtype: numpy.ndarray
    :raises ValueError: If the header scale is missing, non-square, or not in arcsec.
    """
    unit1 = str(header.get("CUNIT1", "")).strip().lower()
    unit2 = str(header.get("CUNIT2", "")).strip().lower()
    if unit1 != "arcsec" or unit2 != "arcsec":
        raise ValueError(f"EOVSA CDELT units must be arcsec; found CUNIT1={unit1!r}, CUNIT2={unit2!r}")
    try:
        cdelt1_arcsec = abs(float(header["CDELT1"]))
        cdelt2_arcsec = abs(float(header["CDELT2"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("EOVSA FITS header must provide numeric CDELT1 and CDELT2") from exc
    if not np.isfinite(cdelt1_arcsec) or not np.isfinite(cdelt2_arcsec) or cdelt1_arcsec <= 0 or cdelt2_arcsec <= 0:
        raise ValueError("EOVSA FITS CDELT1 and CDELT2 must be finite and nonzero")
    if not np.isclose(cdelt1_arcsec, cdelt2_arcsec):
        raise ValueError("EOVSA FITS CDELT1 and CDELT2 must match for the square-pixel sfu conversion")

    c_cgs = 2.99792458e10
    k_b_cgs = 1.380649e-16
    omega_pix = (cdelt1_arcsec * np.pi / 180.0 / 3600.0) ** 2
    frequencies = np.asarray(freqs_hz, dtype=float)
    return float(level_sfu) * 1e-19 * c_cgs**2 / (2.0 * k_b_cgs * frequencies**2 * omega_pix)


def _contour_plane(data: np.ndarray, peak: float, threshold: float, absolute_level: bool) -> np.ndarray | None:
    """Return a finite plane when a contour threshold crosses its data range."""
    finite = data[np.isfinite(data)]
    if finite.size == 0 or not np.isfinite(threshold) or (not absolute_level and peak <= 0):
        return None
    floor = float(np.nanmin(finite)) - abs(peak) - 1.0
    cleaned = np.nan_to_num(data, nan=floor, posinf=peak, neginf=floor)
    if threshold <= float(np.nanmin(cleaned)) or threshold >= float(np.nanmax(cleaned)):
        return None
    return cleaned


def _contour_polylines(
    cleaned: np.ndarray,
    threshold: float,
    transform: np.ndarray | None = None,
    display_height: int | None = None,
) -> list[list[list[float]]]:
    """Trace one band and optionally transform it into display-image pixels."""
    polylines: list[list[list[float]]] = []
    for contour in measure.find_contours(cleaned, threshold):
        if contour.shape[0] < 3:
            continue
        pixels = np.column_stack([contour[:, 1], contour[:, 0]])
        if transform is not None:
            pixels = np.column_stack([pixels, np.ones(pixels.shape[0])]) @ transform
        if display_height is not None:
            pixels[:, 1] = display_height - 1 - pixels[:, 1]
        finite_points = pixels[np.isfinite(pixels).all(axis=1)]
        if finite_points.shape[0] < 3:
            continue
        polylines.append([[float(x), float(y)] for x, y in finite_points])
    return polylines


CMAP_ALIASES = {
    "aia": "sdoaia131",
    "aia94": "sdoaia94",
    "aia131": "sdoaia131",
    "aia171": "sdoaia171",
    "aia193": "sdoaia193",
    "aia211": "sdoaia211",
    "aia304": "sdoaia304",
    "aia335": "sdoaia335",
    "gray": "gray",
    "gray_r": "gray_r",
    "viridis": "viridis",
    "magma": "magma",
    "turbo": "turbo",
    "coolwarm": "coolwarm",
    "parula": "parula",
    "parula_r": "parula_r",
    "inferno": "inferno",
    "inferno_r": "inferno_r",
    "viridis_r": "viridis_r",
    "rdylbu": "RdYlBu",
    "rdylbu_r": "RdYlBu",
    "rdbu": "RdBu_r",
}

# Canonical Parula stops copied from
# ovrolwa-rfr-corr-app/frontend/src/radioColormaps.ts. Keeping the exact table
# in source makes saved palette ids reproducible across Matplotlib releases.
PARULA_STOPS = (
    "#352a87", "#1e42ba", "#0575b7", "#079bad", "#07b792", "#2ec971",
    "#7bd151", "#bdd242", "#ecd14d", "#f9ba59", "#f9fb0e",
)
PARULA_COLORMAP = LinearSegmentedColormap.from_list("parula", PARULA_STOPS)
CUSTOM_COLORMAPS: dict[str, Colormap] = {
    "parula": PARULA_COLORMAP,
    "parula_r": PARULA_COLORMAP.reversed(name="parula_r"),
}


def _difference_mode(mode: str | None, use_running_diff: bool = True) -> str:
    if mode:
        cleaned = str(mode).strip().lower()
        if cleaned in {"none", "running", "base"}:
            return cleaned
    return "running" if use_running_diff else "none"


def _difference_operation(operation: str | None) -> str | None:
    """Normalize the independent frame operation, preserving legacy mode APIs."""
    if operation is None:
        return None
    cleaned = str(operation).strip().lower()
    return cleaned if cleaned in {"none", "subtract", "ratio"} else "none"


def _temporal_mode(mode: str | None) -> str:
    cleaned = str(mode or "none").strip().lower()
    return cleaned if cleaned in {"none", "lowpass", "bandpass"} else "none"


def _difference_reference(reference: str | None) -> str:
    cleaned = str(reference or "previous").strip().lower()
    return cleaned if cleaned in {"previous", "base", "mean"} else "previous"


def _radio_layer_is_identity(layer: dict[str, object]) -> bool:
    """Return whether a tracked radio layer resolves to a raw decoded plane.

    Mirrors the default-resolution in
    :meth:`SolRadSession.processed_tracking_frame` and the
    operation/temporal/radial-gamma normalization in
    :meth:`EovsaSequence.frame_for_aia_time` so that callers can bypass the
    full processing pipeline exactly when it would have been a no-op,
    without risking any divergence from its output.

    :param layer: Tracked source-layer science parameters.
    :type layer: dict[str, object]
    :returns: ``True`` when no difference operation, temporal filter, or
        radial weighting would be applied.
    :rtype: bool
    """
    operation_raw = str(layer.get("differenceOperation", layer.get("operation", "ratio"))).strip().lower()
    operation = operation_raw if operation_raw in {"none", "subtract", "ratio"} else "none"
    if operation != "none":
        return False
    temporal_raw = str(layer.get("temporalMode", "none")).strip().lower()
    temporal = temporal_raw if temporal_raw in {"none", "lowpass", "bandpass"} else "none"
    if temporal != "none":
        return False
    return float(layer.get("radialGamma", 0.0)) <= 0.0


def _frequency_scale(scale: str | None) -> str:
    cleaned = str(scale or "linear").strip().lower()
    return cleaned if cleaned in {"linear", "log"} else "linear"


def _spectrogram_normalization(mode: str | None) -> str:
    cleaned = str(mode or "none").strip().lower()
    if cleaned not in {"none", "divide", "subtract"}:
        raise ValueError(f"Unsupported spectrogram normalization: {mode!r}")
    return cleaned


def _spectrogram_row_medians(data: np.ndarray) -> np.ndarray:
    """Return one finite-sample median per frequency row."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Spectrogram data must be two-dimensional")
    medians = np.full(values.shape[0], np.nan, dtype=np.float32)
    for index, row in enumerate(values):
        finite = row[np.isfinite(row)]
        if finite.size:
            medians[index] = np.median(finite)
    return medians


def _normalize_spectrogram_rows(data: np.ndarray, medians: np.ndarray, mode: str) -> np.ndarray:
    """Flatten frequency-row backgrounds using cached full-time medians."""
    normalization = _spectrogram_normalization(mode)
    values = np.asarray(data, dtype=np.float32)
    if normalization == "none":
        return values
    row_medians = np.asarray(medians, dtype=np.float32).reshape(-1)
    if values.ndim != 2 or len(row_medians) != values.shape[0]:
        raise ValueError("Spectrogram row medians must match the frequency dimension")
    if normalization == "divide":
        divisor = np.where(np.isfinite(row_medians) & (row_medians != 0), row_medians, 1.0)
        return (values / divisor[:, None]).astype(np.float32)
    baseline = np.where(np.isfinite(row_medians), row_medians, 0.0)
    return (values - baseline[:, None]).astype(np.float32)


def _display_scale(scale: str | None) -> str:
    """Validate and normalize a display intensity scale.

    :param scale: Requested display scale.
    :type scale: str | None
    :returns: One of the supported display scale names.
    :rtype: str
    :raises ValueError: If the display scale is unsupported.
    """
    cleaned = str(scale or "linear").strip().lower()
    if cleaned not in {"linear", "log", "sqrt", "asinh"}:
        raise ValueError(f"Unsupported display scale: {scale!r}")
    return cleaned


def _resample_frequency_axis(
    data: np.ndarray,
    freqs_ghz: np.ndarray,
    scale: str,
    frequency_min_ghz: float | None = None,
    frequency_max_ghz: float | None = None,
) -> np.ndarray:
    """Resample spectrogram rows onto an evenly spaced linear or log frequency grid."""
    values = np.asarray(data, dtype=np.float32)
    freqs = np.asarray(freqs_ghz, dtype=float)
    if values.ndim != 2 or len(freqs) != values.shape[0] or len(freqs) < 2:
        return values
    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    sorted_values = values[order]
    unique_freqs, unique_indices = np.unique(sorted_freqs, return_index=True)
    sorted_values = sorted_values[unique_indices]
    if len(unique_freqs) < 2:
        return sorted_values
    frequency_min = float(unique_freqs[0] if frequency_min_ghz is None else frequency_min_ghz)
    frequency_max = float(unique_freqs[-1] if frequency_max_ghz is None else frequency_max_ghz)
    frequency_min = float(np.clip(frequency_min, unique_freqs[0], unique_freqs[-1]))
    frequency_max = float(np.clip(frequency_max, unique_freqs[0], unique_freqs[-1]))
    if frequency_max <= frequency_min:
        raise ValueError("Frequency maximum must be greater than frequency minimum")
    mode = _frequency_scale(scale)
    if mode == "log" and frequency_min > 0:
        target_freqs = np.geomspace(frequency_min, frequency_max, values.shape[0])
    else:
        target_freqs = np.linspace(frequency_min, frequency_max, values.shape[0])
    positions = np.interp(target_freqs, unique_freqs, np.arange(len(unique_freqs), dtype=float))
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(unique_freqs) - 1)
    weight = (positions - lower).astype(np.float32)[:, None]
    return (sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight).astype(np.float32)


def _regularize_spectrogram_time_axis(
    data: np.ndarray,
    times_mjd: np.ndarray,
    max_columns: int = MAX_SPECTROGRAM_TIME_COLUMNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Place spectrogram columns on a uniform elapsed-time grid.

    The nominal cadence is estimated from positive timestamp differences and
    deliberately ignores a dominant long-gap tail.  Source samples are placed
    in their nearest grid bins without interpolation; bins that fall inside a
    timestamp gap remain NaN so renderers can display an explicit blank band.

    :param data: Spectrogram values shaped ``(frequency, time)``.
    :type data: numpy.ndarray
    :param times_mjd: Modified Julian Date value for each input column.
    :type times_mjd: numpy.ndarray
    :param max_columns: Maximum output columns for pathological spans.
    :type max_columns: int
    :returns: Uniform-grid values and corresponding MJD timestamps.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    values = np.asarray(data, dtype=np.float32)
    times = np.asarray(times_mjd, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(times) or len(times) < 2:
        return values, times
    finite = np.isfinite(times)
    if finite.sum() < 2:
        return values, times
    values = values[:, finite]
    times = times[finite]
    order = np.argsort(times)
    times = times[order]
    values = values[:, order]
    differences = np.diff(times)
    positive = differences[np.isfinite(differences) & (differences > 0)]
    if not len(positive):
        return values, times
    nominal = float(np.median(positive))
    # A severe spread is the signature of one or more missing intervals. Use
    # the lower half to avoid letting the gap itself define the cadence, while
    # ignoring timestamp duplicates that would otherwise force an enormous
    # output grid.
    if len(positive) >= 2 and float(np.max(positive) / np.min(positive)) > 4.0:
        lower_half = positive[positive <= nominal]
        lower_half = lower_half[lower_half >= nominal * 0.1]
        if not len(lower_half):
            lower_half = np.sort(positive)[: max(1, (len(positive) + 1) // 2)]
        nominal = float(np.median(lower_half))
    if not np.isfinite(nominal) or nominal <= 0:
        return values, times
    start = float(times[0])
    end = float(times[-1])
    span = max(0.0, end - start)
    limit = max(2, int(max_columns))
    column_count = max(2, int(np.round(span / nominal)) + 1)
    if column_count > limit:
        column_count = limit
    # Include both endpoints exactly; this keeps the displayed gap aligned
    # with the metadata even when MJD arithmetic introduces tiny round-off.
    nominal = span / max(1, column_count - 1)
    grid = start + np.arange(column_count, dtype=float) * nominal
    output_sum = np.zeros((values.shape[0], column_count), dtype=np.float32)
    output_count = np.zeros((values.shape[0], column_count), dtype=np.uint16)
    positions = np.rint((times - start) / nominal).astype(int)
    distance = np.abs(grid[np.clip(positions, 0, column_count - 1)] - times)
    keep = (positions >= 0) & (positions < column_count) & (distance <= nominal * 0.6)
    for source_index, column_index in enumerate(positions):
        if not keep[source_index]:
            continue
        column = values[:, source_index]
        valid = np.isfinite(column)
        if not np.any(valid):
            continue
        output_sum[valid, column_index] += column[valid]
        output_count[valid, column_index] += 1
    output = np.full(output_sum.shape, np.nan, dtype=np.float32)
    valid = output_count > 0
    output[valid] = output_sum[valid] / output_count[valid]
    return output, grid


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return a ratio while masking zero and numerically unsafe denominators."""
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    finite = np.isfinite(denominator)
    scale = float(np.nanmax(np.abs(denominator[finite]))) if finite.any() else 0.0
    epsilon = max(1.0e-12, scale * 1.0e-7)
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=finite & (np.abs(denominator) > epsilon))
    return result


def radial_factor_map(
    radius_ratio: np.ndarray,
    gamma: float,
    r_max: float = RADIAL_R_MAX,
) -> np.ndarray:
    """Return the exponential radial enhancement factor map.

    :param radius_ratio: Pixel radius divided by the observed solar radius.
    :type radius_ratio: numpy.ndarray
    :param gamma: Non-negative enhancement strength; the UI slider covers 0--3.
    :type gamma: float
    :param r_max: Retained for compatibility; the exponential map uses a 1e3 cap.
    :type r_max: float
    :returns: Float32 factors, with all on-disk values equal to one and all
        off-limb values capped at 1e3.
    :rtype: numpy.ndarray
    """
    del r_max
    exponent = float(gamma)
    if not np.isfinite(exponent):
        exponent = 0.0
    exponent = max(0.0, exponent)
    radius = np.asarray(radius_ratio, dtype=np.float32)
    excess = np.maximum(radius - 1.0, 0.0)
    factor = np.exp(exponent * excess / 0.2).astype(np.float32)
    factor = np.minimum(factor, np.float32(1.0e3))
    return np.where(radius <= 1.0, np.float32(1.0), factor).astype(np.float32)


def _quantized_radial_gamma(gamma: float) -> float:
    try:
        exponent = float(gamma)
    except (TypeError, ValueError):
        exponent = 0.0
    if not np.isfinite(exponent):
        exponent = 0.0
    return round(max(0.0, exponent) * 10.0) / 10.0


def _time_seconds(times: object) -> np.ndarray:
    values = np.asarray(getattr(times, "unix", times), dtype=float).reshape(-1)
    if values.size and 30000.0 <= np.nanmedian(np.abs(values)) <= 100000.0:
        # Plain numeric MJD inputs are convenient in unit tests and public
        # helpers; Astropy Time inputs have already been converted to Unix.
        values = values * 86400.0
    return values


def _temporal_neighbor_indices(
    times: object,
    target_index: int,
    sigma: float,
    *,
    frame_radius: int | None = None,
    sample_limit: int | None = None,
) -> np.ndarray:
    values = _time_seconds(times)
    if not len(values):
        return np.array([], dtype=int)
    sigma_value = max(float(sigma), np.finfo(float).eps)
    target = int(np.clip(target_index, 0, len(values) - 1))
    delta = values - values[target]
    candidates = np.flatnonzero(np.isfinite(delta) & (np.abs(delta) <= 3.0 * sigma_value + 1.0e-9))
    if frame_radius is not None:
        candidates = candidates[np.abs(candidates - target) <= int(frame_radius)]
    if sample_limit is None or len(candidates) <= int(sample_limit):
        return candidates.astype(int)
    limit = max(1, int(sample_limit))
    stride = max(1, int(math.ceil(len(candidates) / limit)))
    sampled = candidates[::stride]
    if sampled[-1] != candidates[-1] and len(sampled) < limit:
        sampled = np.append(sampled, candidates[-1])
    if target not in sampled:
        replacement = int(np.argmax(np.abs(sampled - target)))
        sampled[replacement] = target
    return np.unique(sampled).astype(int)


def temporal_window_cap_status(
    times: object,
    target_index: int,
    mode: str,
    sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
    sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
) -> str | None:
    """Describe temporal neighbor caps reached for one filtered frame.

    :param times: Native frame timestamps as seconds, MJD values, or Astropy
        ``Time`` values.
    :type times: object
    :param target_index: Index of the frame being rendered.
    :type target_index: int
    :param mode: ``none``, ``lowpass``, or ``bandpass``.
    :type mode: str
    :param sigma_short: Short-arm Gaussian sigma in seconds.
    :type sigma_short: float
    :param sigma_long: Long-arm Gaussian sigma in seconds.
    :type sigma_long: float
    :returns: ``short``, ``long``, ``short,long``, or ``None``.
    :rtype: str or None
    """
    normalized = _temporal_mode(mode)
    if normalized == "none":
        return None
    values = _time_seconds(times)
    target = int(np.clip(target_index, 0, max(0, len(values) - 1))) if len(values) else 0
    capped: list[str] = []
    if len(_temporal_neighbor_indices(values, target, sigma_short)) > 0:
        short_uncapped = _temporal_neighbor_indices(values, target, sigma_short)
        short_capped = _temporal_neighbor_indices(
            values, target, sigma_short, frame_radius=TEMPORAL_SHORT_FRAME_RADIUS
        )
        if len(short_uncapped) != len(short_capped):
            capped.append("short")
    if normalized == "bandpass":
        long_uncapped = _temporal_neighbor_indices(values, target, sigma_long)
        if len(long_uncapped) > TEMPORAL_LONG_SAMPLE_LIMIT:
            capped.append("long")
    return ",".join(capped) or None


def _gaussian_average(
    data: np.ndarray,
    times: np.ndarray,
    target_index: int,
    indices: np.ndarray,
    sigma: float,
) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    sigma_value = max(float(sigma), np.finfo(float).eps)
    target_time = float(times[int(target_index)])
    numerator = np.zeros(values.shape[1:], dtype=np.float64)
    denominator = np.zeros(values.shape[1:], dtype=np.float64)
    for index in np.asarray(indices, dtype=int):
        delta = float(times[int(index)] - target_time)
        weight = math.exp(-(delta * delta) / (2.0 * sigma_value * sigma_value))
        frame = values[int(index)]
        finite = np.isfinite(frame)
        numerator[finite] += frame[finite] * weight
        denominator[finite] += weight
    result = np.full(values.shape[1:], np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


def temporal_filter(
    data: np.ndarray,
    times: object,
    target_index: int,
    mode: str = "none",
    sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
    sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
    *,
    short_indices: np.ndarray | None = None,
    long_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the NaN-aware irregular-cadence Gaussian temporal filter.

    :param data: Difference or ratio frames shaped ``(time, y, x)``.
    :type data: numpy.ndarray
    :param times: Frame timestamps as seconds, MJD values, or Astropy
        ``Time`` values.
    :type times: object
    :param target_index: Index of the frame to return.
    :type target_index: int
    :param mode: ``none``, ``lowpass``, or ``bandpass``.
    :type mode: str
    :param sigma_short: Short-arm Gaussian sigma in seconds.
    :type sigma_short: float
    :param sigma_long: Long-arm Gaussian sigma in seconds.
    :type sigma_long: float
    :param short_indices: Optional preselected short-arm frame indices.
    :type short_indices: numpy.ndarray or None
    :param long_indices: Optional preselected long-arm frame indices.
    :type long_indices: numpy.ndarray or None
    :returns: The filtered target frame.
    :rtype: numpy.ndarray
    """
    values = np.asarray(data, dtype=np.float32)
    if values.ndim < 2:
        raise ValueError("Temporal filter data must have a time axis and pixel axes")
    if len(values) == 0:
        raise ValueError("Temporal filter data cannot be empty")
    normalized = _temporal_mode(mode)
    target = int(np.clip(target_index, 0, len(values) - 1))
    if normalized == "none":
        return values[target]
    times_seconds = _time_seconds(times)
    if len(times_seconds) != len(values):
        raise ValueError("Temporal filter times must match the data time axis")
    short = (
        _temporal_neighbor_indices(
            times_seconds,
            target,
            sigma_short,
            frame_radius=TEMPORAL_SHORT_FRAME_RADIUS,
        )
        if short_indices is None
        else np.asarray(short_indices, dtype=int)
    )
    short_result = _gaussian_average(values, times_seconds, target, short, sigma_short)
    if normalized == "lowpass":
        return short_result
    long = (
        _temporal_neighbor_indices(
            times_seconds,
            target,
            sigma_long,
            sample_limit=TEMPORAL_LONG_SAMPLE_LIMIT,
        )
        if long_indices is None
        else np.asarray(long_indices, dtype=int)
    )
    long_result = _gaussian_average(values, times_seconds, target, long, sigma_long)
    result = np.full(values.shape[1:], np.nan, dtype=np.float32)
    finite = np.isfinite(short_result) & np.isfinite(long_result)
    result[finite] = short_result[finite] - long_result[finite]
    return result


def patch_mean(data: np.ndarray, x: float, y: float, patch_radius: int = 1) -> float:
    """Return the finite mean in a clipped square pixel patch.

    :param data: Two-dimensional image values shaped ``(y, x)``.
    :type data: numpy.ndarray
    :param x: Image pixel x coordinate.
    :type x: float
    :param y: Image pixel y coordinate.
    :type y: float
    :param patch_radius: Radius of the square patch in pixels.
    :type patch_radius: int
    :returns: The patch mean, or NaN when no finite pixels are present.
    :rtype: float
    :raises ValueError: If the data is not two-dimensional or the radius is negative.
    """
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Pixel probe data must be two-dimensional")
    radius = int(patch_radius)
    if radius < 0:
        raise ValueError("patch_radius must be non-negative")
    center_x = int(np.clip(np.rint(float(x)), 0, values.shape[1] - 1))
    center_y = int(np.clip(np.rint(float(y)), 0, values.shape[0] - 1))
    y0 = max(0, center_y - radius)
    y1 = min(values.shape[0], center_y + radius + 1)
    x0 = max(0, center_x - radius)
    x1 = min(values.shape[1], center_x + radius + 1)
    finite = values[y0:y1, x0:x1]
    usable = finite[np.isfinite(finite)]
    return float(np.mean(usable)) if usable.size else float("nan")


def stride_indices(count: int, max_points: int) -> tuple[np.ndarray, int]:
    """Return native indices decimated to the requested maximum length.

    :param count: Number of native samples in the requested range.
    :type count: int
    :param max_points: Maximum number of samples to return.
    :type max_points: int
    :returns: Selected indices and the reported stride.
    :rtype: tuple[numpy.ndarray, int]
    :raises ValueError: If the maximum point count is not positive.
    """
    if int(max_points) <= 0:
        raise ValueError("max_points must be positive")
    native_count = max(0, int(count))
    stride = max(1, int(math.ceil(native_count / int(max_points)))) if native_count else 1
    return np.arange(native_count, dtype=int)[::stride], stride


def series_stats(values: object) -> dict[str, float | None]:
    """Return finite min/max and percentile statistics for a numeric series.

    :param values: Numeric values, with NaNs representing missing samples.
    :type values: object
    :returns: Null-safe ``min``, ``max``, ``p1``, and ``p99`` statistics.
    :rtype: dict[str, float or None]
    """
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"min": None, "max": None, "p1": None, "p99": None}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p1": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
    }


def temporal_filter_series(
    values: object,
    times: object,
    mode: str,
    sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
    sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
) -> np.ndarray:
    """Apply the frame temporal filter weights to a one-dimensional series.

    :param values: Raw series values, with NaNs representing missing samples.
    :type values: object
    :param times: Sample timestamps as seconds, MJD values, or Astropy times.
    :type times: object
    :param mode: ``lowpass`` or ``bandpass``.
    :type mode: str
    :param sigma_short: Short-arm Gaussian sigma in seconds.
    :type sigma_short: float
    :param sigma_long: Long-arm Gaussian sigma in seconds.
    :type sigma_long: float
    :returns: Smoothed series with one value per input sample.
    :rtype: numpy.ndarray
    :raises ValueError: If the values and timestamps have different lengths.
    """
    raw = np.asarray(values, dtype=np.float32).reshape(-1)
    times_seconds = _time_seconds(times)
    if len(raw) != len(times_seconds):
        raise ValueError("Temporal series times must match the data length")
    normalized = _temporal_mode(mode)
    if normalized == "none" or not len(raw):
        return raw.copy()
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    series_data = raw[:, None]
    for target in range(len(raw)):
        short_indices = _temporal_neighbor_indices(
            times_seconds,
            target,
            sigma_short,
            frame_radius=TEMPORAL_SHORT_FRAME_RADIUS,
        )
        short_result = _gaussian_average(
            series_data, times_seconds, target, short_indices, sigma_short
        )[0]
        if normalized == "lowpass":
            result[target] = short_result
            continue
        long_indices = _temporal_neighbor_indices(
            times_seconds,
            target,
            sigma_long,
            sample_limit=TEMPORAL_LONG_SAMPLE_LIMIT,
        )
        long_result = _gaussian_average(
            series_data, times_seconds, target, long_indices, sigma_long
        )[0]
        if np.isfinite(short_result) and np.isfinite(long_result):
            result[target] = short_result - long_result
    return result


def _meta_value(meta: object, *keys: str) -> object | None:
    for key in keys:
        try:
            value = meta.get(key)  # type: ignore[union-attr]
        except AttributeError:
            value = None
        if value is not None:
            return value
        try:
            for existing in meta.keys():  # type: ignore[union-attr]
                if str(existing).upper() == key.upper():
                    return meta[existing]  # type: ignore[index]
        except AttributeError:
            pass
    return None


def _radius_ratio_map(shape: tuple[int, int], meta: object) -> np.ndarray:
    """Build a WCS-derived pixel radius ratio for one image source."""
    ny, nx = shape
    try:
        smap = Map(np.zeros(shape, dtype=np.float32), meta)
        world_x, world_y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
        world = smap.pixel_to_world(world_x * u.pix, world_y * u.pix)
        rsun_value = float(smap.rsun_obs.to_value(u.arcsec))
        if np.isfinite(rsun_value) and rsun_value > 0:
            radius = np.hypot(world.Tx.to_value(u.arcsec), world.Ty.to_value(u.arcsec))
            return (radius / rsun_value).astype(np.float32)
    except Exception:
        pass
    rsun = _meta_value(meta, "RSUN_OBS")
    if rsun is None:
        rsun_ref = _meta_value(meta, "RSUN_REF")
        dsun_obs = _meta_value(meta, "DSUN_OBS")
        try:
            rsun = float(rsun_ref) / float(dsun_obs) * (180.0 / math.pi) * 3600.0
        except (TypeError, ValueError, ZeroDivisionError):
            rsun = None
    try:
        rsun_value = float(rsun)
        crpix1 = float(_meta_value(meta, "CRPIX1"))
        crpix2 = float(_meta_value(meta, "CRPIX2"))
        crval1 = float(_meta_value(meta, "CRVAL1") or 0.0)
        crval2 = float(_meta_value(meta, "CRVAL2") or 0.0)
        cdelt1 = float(_meta_value(meta, "CDELT1"))
        cdelt2 = float(_meta_value(meta, "CDELT2"))
        if not np.isfinite(rsun_value) or rsun_value <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return np.ones(shape, dtype=np.float32)
    unit1 = str(_meta_value(meta, "CUNIT1") or "arcsec").lower()
    unit2 = str(_meta_value(meta, "CUNIT2") or "arcsec").lower()
    if "deg" in unit1:
        cdelt1 *= 3600.0
    if "deg" in unit2:
        cdelt2 *= 3600.0
    pixel_x, pixel_y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    x_arcsec = (pixel_x + 1.0 - crpix1) * cdelt1 + crval1
    y_arcsec = (pixel_y + 1.0 - crpix2) * cdelt2 + crval2
    return np.hypot(x_arcsec, y_arcsec).astype(np.float32) / np.float32(rsun_value)


def _source_radial_factor(source: object, meta: object, shape: tuple[int, int], gamma: float) -> np.ndarray:
    exponent = _quantized_radial_gamma(gamma)
    cache = getattr(source, "_radial_factor_cache")
    key = (tuple(shape), exponent)
    cached = _ordered_get(cache, key)
    if cached is not None:
        return cached
    base = getattr(source, "_radial_base_map")
    if base is None or base.shape != shape:
        base = _radius_ratio_map(shape, meta)
        setattr(source, "_radial_base_map", base)
    return _ordered_put(cache, key, radial_factor_map(base, exponent), TEXTURE_CACHE_LIMIT)


def _gather_temporal_frames(frame_getter: object, indices: np.ndarray) -> np.ndarray:
    """Gather an ordered temporal window with bounded concurrent reads."""
    selected = [int(index) for index in np.asarray(indices, dtype=int)]
    getter = frame_getter
    if len(selected) <= 1:
        frames = [getter(index) for index in selected]  # type: ignore[operator]
    else:
        frames = list(TEMPORAL_EXECUTOR.map(getter, selected))  # type: ignore[arg-type]
    return np.stack(frames).astype(np.float32, copy=False)


def _temporal_frame_for_source(
    source: object,
    target_index: int,
    times: object,
    frame_getter: object,
    cache_key_parts: tuple[object, ...],
    mode: str,
    sigma_short: float,
    sigma_long: float,
) -> np.ndarray:
    normalized = _temporal_mode(mode)
    times_seconds = _time_seconds(times)
    target = int(np.clip(target_index, 0, len(times_seconds) - 1))
    short_indices = _temporal_neighbor_indices(
        times_seconds, target, sigma_short, frame_radius=TEMPORAL_SHORT_FRAME_RADIUS
    )
    long_indices = _temporal_neighbor_indices(
        times_seconds, target, sigma_long, sample_limit=TEMPORAL_LONG_SAMPLE_LIMIT
    )
    selected = short_indices if normalized == "lowpass" else np.unique(np.concatenate([short_indices, long_indices]))
    selected = np.asarray(selected, dtype=int)
    local_target = int(np.flatnonzero(selected == target)[0])
    local_short = np.flatnonzero(np.isin(selected, short_indices)).astype(int)
    local_long = np.flatnonzero(np.isin(selected, long_indices)).astype(int)
    key = _cache_key(target, *cache_key_parts, normalized, sigma_short, sigma_long)
    cache = getattr(source, "_temporal_cache")
    cached = _ordered_get(cache, key)
    if cached is not None:
        return cached
    # The temporal window moves one native sample at a time during cache
    # warming.  Cache the processed source frames (for example, running-ratio
    # frames) separately from the ordinary display-data LRU so those shared
    # neighbours are read and differenced only once.  The bounded cache keeps
    # memory use finite for long recordings and does not alter the filter
    # arithmetic or its output bytes.
    raw_cache = getattr(source, "_temporal_raw_frame_cache", None)
    if raw_cache is None:
        raw_cache = OrderedDict()
        setattr(source, "_temporal_raw_frame_cache", raw_cache)
    raw_lock = getattr(source, "_temporal_raw_frame_lock", None)
    if raw_lock is None:
        raw_lock = Lock()
        setattr(source, "_temporal_raw_frame_lock", raw_lock)
    raw_key_parts = tuple(cache_key_parts)

    def cached_frame(index: int) -> np.ndarray:
        raw_key = _cache_key(int(index), *raw_key_parts)
        with raw_lock:
            cached_raw = _ordered_get(raw_cache, raw_key)
        if cached_raw is not None:
            return cached_raw
        frame = frame_getter(int(index))  # type: ignore[operator]
        with raw_lock:
            return _ordered_put(raw_cache, raw_key, frame, TEMPORAL_RAW_FRAME_CACHE_LIMIT)

    frames = _gather_temporal_frames(cached_frame, selected)
    result = temporal_filter(
        frames,
        times_seconds[selected],
        local_target,
        normalized,
        sigma_short,
        sigma_long,
        short_indices=local_short,
        long_indices=local_long,
    )
    return _ordered_put(cache, key, result, TEMPORAL_CACHE_LIMIT)


def _parse_mjd(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        return float(Time(str(value)).mjd)
    except Exception:
        return None


def resolve_time_index(
    times: Time | np.ndarray | list[float],
    requested_mjd: float,
    sampling_policy: str = "nearest",
    max_offset_seconds: float | None = None,
) -> tuple[int, float, float] | None:
    """Resolve a requested MJD on a source-native time axis.

    :param times: Native source samples as an Astropy ``Time`` or MJD values.
    :type times: astropy.time.Time or numpy.ndarray or list[float]
    :param requested_mjd: Requested sample time in MJD.
    :type requested_mjd: float
    :param sampling_policy: ``nearest``, ``previous``, or ``next``.
    :type sampling_policy: str
    :param max_offset_seconds: Optional absolute tolerance for the match.
    :type max_offset_seconds: float or None
    :returns: ``(index, resolved_mjd, signed_offset_seconds)`` or ``None``
        when the policy has no sample or exceeds the tolerance.
    :raises ValueError: If the policy or tolerance is invalid.
    :rtype: tuple[int, float, float] or None
    """
    policy = str(sampling_policy).lower()
    if policy not in {"nearest", "previous", "next"}:
        raise ValueError(f"Unknown sampling policy: {sampling_policy}")
    if max_offset_seconds is not None:
        max_offset = float(max_offset_seconds)
        if not np.isfinite(max_offset) or max_offset < 0:
            raise ValueError("maxOffsetSeconds must be a finite non-negative number")
    values = np.asarray(getattr(times, "mjd", times), dtype=float).reshape(-1)
    valid = np.flatnonzero(np.isfinite(values))
    requested = float(requested_mjd)
    if not np.isfinite(requested):
        raise ValueError("sampleMjd must be finite")
    if valid.size == 0:
        return None
    candidates = valid
    if policy == "previous":
        candidates = valid[values[valid] <= requested]
        if candidates.size == 0:
            return None
        best_value = np.max(values[candidates])
        candidates = candidates[values[candidates] == best_value]
        index = int(np.max(candidates))
        resolved = float(values[index])
        offset_seconds = (resolved - requested) * 86400.0
        if max_offset_seconds is not None and abs(offset_seconds) > float(max_offset_seconds):
            return None
        return index, resolved, float(offset_seconds)
    elif policy == "next":
        candidates = valid[values[valid] >= requested]
        if candidates.size == 0:
            return None
        best_value = np.min(values[candidates])
        candidates = candidates[values[candidates] == best_value]
    else:
        distances = np.abs(values[valid] - requested)
        minimum = float(np.min(distances))
        candidates = valid[distances == minimum]
        if candidates.size == 0:
            candidates = np.array([valid[int(np.argmin(distances))]])
    index = int(np.min(candidates))
    resolved = float(values[index])
    offset_seconds = (resolved - requested) * 86400.0
    if max_offset_seconds is not None and abs(offset_seconds) > float(max_offset_seconds):
        return None
    return index, resolved, float(offset_seconds)


def _coerce_meta_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text in {"True", "False"}:
        return text == "True"
    if not text or re.search(r"[T:/_{}]", text):
        return value
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return value


def _coerce_meta(meta: dict[str, object]) -> dict[str, object]:
    cleaned = {key: _coerce_meta_value(value) for key, value in meta.items()}
    if isinstance(cleaned.get("keycomments"), str):
        cleaned.pop("keycomments", None)
    return cleaned


def _map_indices(group: h5py.Group) -> list[int]:
    indices = []
    for key in group.keys():
        if key.startswith("map_"):
            indices.append(int(key.split("_", 1)[1]))
    return sorted(indices)


def _cache_key(*parts: object) -> tuple[object, ...]:
    key: list[object] = []
    for part in parts:
        if isinstance(part, float):
            key.append(round(part, 6))
        else:
            key.append(part)
    return tuple(key)


def _source_render_identity(source: object) -> dict[str, object]:
    cached = getattr(source, "_render_source_identity", None)
    if isinstance(cached, dict):
        return cached
    raw_paths: list[Path] = []
    files = getattr(source, "files", None)
    if isinstance(files, list):
        raw_paths.extend(Path(path) for path in files if isinstance(path, (str, os.PathLike)))
    else:
        for name in ("intensity_path", "diff_path", "path"):
            value = getattr(source, name, None)
            if value is not None:
                raw_paths.append(Path(value))
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in raw_paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            stat = path.stat()
            records.append({"path": resolved, "mtimeNs": int(stat.st_mtime_ns), "size": int(stat.st_size)})
        except FileNotFoundError:
            records.append({"path": resolved, "missing": True})
    identity = {
        "type": type(source).__name__,
        "files": records,
        "pattern": getattr(source, "pattern", None) or getattr(source, "file_pattern", None),
        "extension": getattr(source, "extension", None),
    }
    if not records:
        identity["ephemeralId"] = id(source)
    setattr(source, "_render_source_identity", identity)
    return identity


def _render_disk_key(source: object, render_kind: str, params: object, extra: object = None) -> str:
    sources = source if isinstance(source, (list, tuple)) else [source]
    return RENDER_DISK_CACHE.make_key({
        "version": 1,
        "sources": [_source_render_identity(item) for item in sources],
        "render": render_kind,
        "params": params,
        "extra": extra,
    })


def _finite_float_list(values: object) -> list[float]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in arr]


def _finite_float_table(values: object) -> list[list[float]]:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim != 2:
        return []
    return [[float(value) for value in row] for row in arr]


def _zero_channel_offsets(nfreq: int) -> dict[str, list[float] | list[bool]]:
    """Return a zero-filled source-level channel-offset table."""
    count = max(0, int(nfreq))
    return {"dx": [0.0] * count, "dy": [0.0] * count, "masked": [False] * count}


def _normalize_channel_mask(value: object, nfreq: int, strict: bool = False) -> list[bool]:
    """Normalize a persisted or API-provided per-channel display mask."""
    count = max(0, int(nfreq))
    if not isinstance(value, list):
        if strict:
            raise ValueError("channelOffsets.masked must be an array")
        return [False] * count
    if strict and len(value) != count:
        raise ValueError(f"channelOffsets.masked must contain exactly {count} values")
    result: list[bool] = []
    for item in value[:count]:
        if isinstance(item, (bool, np.bool_)):
            result.append(bool(item))
        elif isinstance(item, (int, float)) and item in {0, 1}:
            result.append(bool(item))
        elif strict:
            raise ValueError("channelOffsets.masked must contain only boolean or 0/1 values")
        else:
            result.append(False)
    return result + [False] * (count - len(result))


def _normalize_channel_offsets(value: object, nfreq: int, strict: bool = False) -> dict[str, list[float] | list[bool]]:
    """Normalize a persisted or API-provided per-channel offset table.

    :param value: Mapping containing ``dx`` and ``dy`` sequences in arcsec.
    :type value: object
    :param nfreq: Required number of radio frequency channels.
    :type nfreq: int
    :param strict: Require both exact lengths and finite numeric values.
    :type strict: bool
    :returns: Canonical ``{"dx": [...], "dy": [...], "masked": [...]}`` table.
    :rtype: dict[str, list[float] | list[bool]]
    :raises ValueError: If strict validation fails.
    """
    count = max(0, int(nfreq))
    if not isinstance(value, dict):
        if strict:
            raise ValueError("channelOffsets must be an object with dx and dy arrays")
        return _zero_channel_offsets(count)
    result: dict[str, list[float]] = {}
    for axis in ("dx", "dy"):
        raw = value.get(axis)
        if not isinstance(raw, list):
            if strict:
                raise ValueError(f"channelOffsets.{axis} must be an array")
            raw = []
        if strict and len(raw) != count:
            raise ValueError(f"channelOffsets.{axis} must contain exactly {count} values")
        values: list[float] = []
        for item in raw[:count]:
            try:
                numeric = float(item)
            except (TypeError, ValueError) as exc:
                if strict:
                    raise ValueError(f"channelOffsets.{axis} contains a non-numeric value") from exc
                numeric = 0.0
            if not np.isfinite(numeric):
                if strict:
                    raise ValueError(f"channelOffsets.{axis} contains a non-finite value")
                numeric = 0.0
            values.append(numeric)
        result[axis] = values + [0.0] * (count - len(values))
    result["masked"] = _normalize_channel_mask(value.get("masked", value.get("channelMask")), count, strict=False)
    if strict and ("masked" in value or "channelMask" in value):
        result["masked"] = _normalize_channel_mask(value.get("masked", value.get("channelMask")), count, strict=True)
    return result


def _channel_mask_for_session(session: object) -> list[bool]:
    """Return the current source-level mask, tolerating legacy test/session objects."""
    eovsa = getattr(session, "eovsa", None)
    nfreq = int(getattr(eovsa, "nfreq", 0))
    table = _normalize_channel_offsets(getattr(session, "channel_offsets", None), nfreq)
    table_mask = list(table["masked"])  # type: ignore[arg-type]
    raw_mask = getattr(session, "channel_mask", None)
    if isinstance(raw_mask, list) and len(raw_mask) == nfreq and any(raw_mask):
        return _normalize_channel_mask(raw_mask, nfreq)
    return table_mask


def _cache_get(cache: OrderedDict[tuple[object, ...], bytes], key: tuple[object, ...]) -> bytes | None:
    content = cache.get(key)
    if content is not None:
        cache.move_to_end(key)
    return content


def _cache_put(cache: OrderedDict[tuple[object, ...], bytes], key: tuple[object, ...], content: bytes) -> bytes:
    cache[key] = content
    cache.move_to_end(key)
    while len(cache) > TEXTURE_CACHE_LIMIT:
        cache.popitem(last=False)
    return content


def _ordered_get(cache: OrderedDict, key: object):
    """Return a cached value and mark it as recently used."""
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _ordered_put(cache: OrderedDict, key: object, value: object, limit: int):
    """Store a value in an LRU cache and evict the oldest entries."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max(1, int(limit)):
        cache.popitem(last=False)
    return value


def _display_array(data: np.ndarray, orientation: str) -> np.ndarray:
    arr = np.asarray(data)
    if orientation in {"solar", "flip_y"}:
        return arr[::-1, :]
    return arr


def _resolve_colormap(cmap: str, fallback: str) -> tuple[str, Colormap]:
    """Resolve a normalized colormap name and implementation.

    :param cmap: Canonical or legacy colormap name.
    :type cmap: str
    :param fallback: Matplotlib fallback name for an unknown colormap.
    :type fallback: str
    :returns: Normalized cache name and resolved colormap.
    :rtype: tuple[str, matplotlib.colors.Colormap]
    """
    requested = str(cmap).strip()
    cmap_name = CMAP_ALIASES.get(requested.lower(), requested)
    custom = CUSTOM_COLORMAPS.get(cmap_name)
    if custom is not None:
        return cmap_name, custom
    try:
        return cmap_name, colormaps[cmap_name]
    except KeyError:
        return fallback, colormaps[fallback]


def _normalize_display_values(data: np.ndarray, vmin: float, vmax: float, scale: str = "linear") -> np.ndarray:
    """Normalize display values using the selected intensity scale.

    :param data: Values to normalize.
    :type data: numpy.ndarray
    :param vmin: Lower display bound.
    :type vmin: float
    :param vmax: Upper display bound.
    :type vmax: float
    :param scale: Display scale name.
    :type scale: str
    :returns: Values in the inclusive ``[0, 1]`` display range.
    :rtype: numpy.ndarray
    :raises ValueError: If the display scale is unsupported.
    """
    mode = _display_scale(scale)
    if vmax <= vmin:
        vmax = vmin + 1.0
    span = vmax - vmin
    values = np.asarray(data, dtype=float)
    clipped = np.clip((values - vmin) / span, 0.0, 1.0)
    if mode == "log":
        return np.log1p(99.0 * clipped) / np.log1p(99.0)
    if mode == "sqrt":
        return np.sqrt(clipped)
    if mode == "asinh":
        soft = span / 100.0
        transformed = np.arcsinh(np.clip(values - vmin, 0.0, span) / soft)
        return np.clip(transformed / np.arcsinh(span / soft), 0.0, 1.0)
    return clipped


def downsample_to_cap(
    data: np.ndarray,
    max_width: int | None = None,
    max_height: int | None = None,
) -> np.ndarray:
    """Area-average a two-dimensional array to an optional size cap.

    A single integer block factor is used on both axes so pixels retain their
    aspect ratio. Partial edge blocks average only their source samples.

    :param data: Full-resolution science array shaped ``(height, width)``.
    :type data: numpy.ndarray
    :param max_width: Maximum output width, or ``None`` for no width cap.
    :type max_width: int or None
    :param max_height: Maximum output height, or ``None`` for no height cap.
    :type max_height: int or None
    :returns: The original array when no reduction is needed, otherwise block
        means that fit within both supplied caps.
    :rtype: numpy.ndarray
    :raises ValueError: If a supplied cap is not positive or data is not 2-D.
    """
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("Capped frame data must be two-dimensional")
    if max_width is not None and int(max_width) < 1:
        raise ValueError("max_width must be positive")
    if max_height is not None and int(max_height) < 1:
        raise ValueError("max_height must be positive")
    height, width = values.shape
    factor = max(
        1,
        math.ceil(width / int(max_width)) if max_width is not None else 1,
        math.ceil(height / int(max_height)) if max_height is not None else 1,
    )
    if factor == 1:
        return values
    output_height = math.ceil(height / factor)
    output_width = math.ceil(width / factor)
    padded = np.pad(
        np.asarray(values, dtype=np.float64),
        ((0, output_height * factor - height), (0, output_width * factor - width)),
        mode="constant",
        constant_values=np.nan,
    )
    blocks = padded.reshape(output_height, factor, output_width, factor)
    finite = np.isfinite(blocks)
    totals = np.where(finite, blocks, 0.0).sum(axis=(1, 3))
    counts = finite.sum(axis=(1, 3))
    result = np.full((output_height, output_width), np.nan, dtype=np.float64)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def _render_png(data: np.ndarray, vmin: float, vmax: float, cmap: str, scale: str = "linear", orientation: str = "raw") -> bytes:
    """Render display values as a fast PNG.

    :param data: Science values before display normalization.
    :type data: numpy.ndarray
    :param vmin: Lower display bound.
    :type vmin: float
    :param vmax: Upper display bound.
    :type vmax: float
    :param cmap: Matplotlib colormap name.
    :type cmap: str
    :param scale: Display normalization scale.
    :type scale: str
    :param orientation: Display orientation transform.
    :type orientation: str
    :returns: PNG-encoded image bytes.
    :rtype: bytes
    """
    arr = np.asarray(_display_array(data, orientation), dtype=float)
    missing = ~np.isfinite(arr)
    arr = np.nan_to_num(arr, nan=vmin, posinf=vmax, neginf=vmin)
    normed = _normalize_display_values(arr, vmin, vmax, scale)
    cmap_name, cm = _resolve_colormap(cmap, "gray")
    rgba = cm(normed, bytes=True)
    if cmap_name.lower() in {"gray", "gray_r"}:
        luminance = rgba[..., 0]
        luminance[missing] = 0
        image = Image.fromarray(luminance)
    else:
        # Keep missing samples transparent. The frontend paints the plot black,
        # so this produces a true dark gap rather than a colormap-minimum stripe.
        rgba[missing, 3] = 0
        image = Image.fromarray(rgba)
    out = BytesIO()
    image.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def data_stats_headers(data: np.ndarray) -> dict[str, str]:
    """Return robust statistics for a rendered, pre-normalization array."""
    values = np.asarray(data, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "X-Data-Min": "nan",
            "X-Data-Max": "nan",
            "X-Data-P1": "nan",
            "X-Data-P99": "nan",
        }
    p1, p99 = np.percentile(finite, [1.0, 99.0])
    return {
        "X-Data-Min": f"{float(np.min(finite)):.12g}",
        "X-Data-Max": f"{float(np.max(finite)):.12g}",
        "X-Data-P1": f"{float(p1):.12g}",
        "X-Data-P99": f"{float(p99):.12g}",
    }


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _tracking_point(
    frame_index: int,
    x: float,
    y: float,
    confidence: float,
    is_anchor: bool = False,
    mjd: float | None = None,
) -> dict[str, object]:
    """Build one canonical solver or anchor point.

    :param frame_index: Native context frame index.
    :type frame_index: int
    :param x: Context-image x coordinate in pixels.
    :type x: float
    :param y: Context-image y coordinate in pixels.
    :type y: float
    :param confidence: Normalized cross-correlation confidence.
    :type confidence: float
    :param is_anchor: Whether the point is authoritative.
    :type is_anchor: bool
    :param mjd: Optional native sample time.
    :type mjd: float or None
    :returns: Canonical JSON-ready point.
    :rtype: dict[str, object]
    """
    return {
        "frameIndex": int(frame_index),
        "mjd": float(mjd) if mjd is not None else float("nan"),
        "x": float(x),
        "y": float(y),
        "confidence": float(confidence),
        "isAnchor": bool(is_anchor),
    }


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    """Return the sub-pixel offset of a sampled parabolic maximum.

    :param values: One-dimensional correlation samples through the peak.
    :type values: numpy.ndarray
    :param index: Integer peak index.
    :type index: int
    :returns: Refined offset in the closed interval ``[-1, 1]``.
    :rtype: float
    """
    if index <= 0 or index >= values.size - 1:
        return 0.0
    left, center, right = (float(values[index - 1]), float(values[index]), float(values[index + 1]))
    denominator = left - 2.0 * center + right
    if not np.isfinite([left, center, right, denominator]).all() or abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _sample_tracking_patch(data: np.ndarray, center: tuple[float, float], radius: int) -> np.ndarray:
    """Sample a square patch at a floating-point image position.

    :param data: Processed two-dimensional image.
    :type data: numpy.ndarray
    :param center: Floating-point ``(x, y)`` patch center.
    :type center: tuple[float, float]
    :param radius: Half-width of the returned square patch.
    :type radius: int
    :returns: Bilinearly sampled ``(2 * radius + 1)`` square patch.
    :rtype: numpy.ndarray
    """
    offsets = np.arange(-radius, radius + 1, dtype=float)
    yy, xx = np.meshgrid(center[1] + offsets, center[0] + offsets, indexing="ij")
    finite = np.isfinite(data)
    fill = float(np.nanmedian(data[finite])) if finite.any() else 0.0
    clean = np.where(finite, data, fill)
    return map_coordinates(clean, [yy, xx], order=1, mode="nearest", prefilter=False)


def _tracking_prediction(history: list[dict[str, object]], target_frame: int) -> np.ndarray:
    """Predict a position from the latest two or three confident points.

    :param history: Accepted points in tracking order.
    :type history: list[dict[str, object]]
    :param target_frame: Native frame index to predict.
    :type target_frame: int
    :returns: Predicted ``(x, y)`` context-image position.
    :rtype: numpy.ndarray
    """
    recent = history[-3:]
    if len(recent) < 2:
        return np.array([float(recent[-1]["x"]), float(recent[-1]["y"])], dtype=float)
    frames = np.asarray([int(point["frameIndex"]) for point in recent], dtype=float)
    if np.unique(frames).size < 2:
        return np.array([float(recent[-1]["x"]), float(recent[-1]["y"])], dtype=float)
    positions = np.asarray([[float(point["x"]), float(point["y"])] for point in recent], dtype=float)
    design = np.column_stack([frames, np.ones(frames.size)])
    coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
    return np.asarray([float(target_frame), 1.0]) @ coefficients


def _point_inside_tracking_roi(point: np.ndarray, roi_pixels: np.ndarray | None) -> bool:
    """Return whether a predicted center is inside an optional pixel ROI.

    :param point: Candidate ``(x, y)`` position.
    :type point: numpy.ndarray
    :param roi_pixels: Optional context-pixel polygon.
    :type roi_pixels: numpy.ndarray or None
    :returns: ``True`` when no ROI is active or the point lies inside it.
    :rtype: bool
    """
    if roi_pixels is None or roi_pixels.shape[0] < 3:
        return True
    return bool(MplPath(roi_pixels).contains_point((float(point[0]), float(point[1])), radius=1e-7))


def track_ncc_pass(
    frame_getter: object,
    start_point: dict[str, object],
    stop_frame: int,
    direction: int,
    patch_radius: int = 6,
    search_radius: int = 18,
    confidence_threshold: float = 0.5,
    roi_pixels: np.ndarray | None = None,
    times_mjd: object | None = None,
    on_progress: object | None = None,
    should_stop: object | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Track one feature with velocity-predicted normalized cross-correlation.

    The template is sampled from the last confident position and refreshed
    after every accepted match. Two consecutive sub-threshold peaks stop the
    pass at its last confident point.

    :param frame_getter: Callable returning a processed two-dimensional frame.
    :type frame_getter: callable
    :param start_point: Canonical point containing ``frameIndex``, ``x``, and ``y``.
    :type start_point: dict[str, object]
    :param stop_frame: Inclusive final frame bound.
    :type stop_frame: int
    :param direction: Positive for forward tracking, negative for backward.
    :type direction: int
    :param patch_radius: Template radius in pixels.
    :type patch_radius: int
    :param search_radius: Search radius around the predicted position.
    :type search_radius: int
    :param confidence_threshold: Minimum accepted NCC peak.
    :type confidence_threshold: float
    :param roi_pixels: Optional polygon in the processed frame's pixel space.
    :type roi_pixels: numpy.ndarray or None
    :param times_mjd: Optional native MJD time axis.
    :type times_mjd: object or None
    :param on_progress: Optional callback invoked once per attempted frame.
    :type on_progress: callable or None
    :param should_stop: Optional cancellation predicate.
    :type should_stop: callable or None
    :returns: Confident points including the start point and a stop state.
    :rtype: tuple[list[dict[str, object]], str]
    """
    if not callable(frame_getter):
        raise TypeError("frame_getter must be callable")
    step = 1 if direction >= 0 else -1
    start_frame = int(start_point["frameIndex"])
    terminal = int(stop_frame)
    if (terminal - start_frame) * step < 0:
        raise ValueError("stop_frame must lie in the requested direction")
    pr = max(2, int(patch_radius))
    sr = max(pr + 2, int(search_radius))
    axis = None if times_mjd is None else np.asarray(times_mjd, dtype=float)
    initial = dict(start_point)
    initial["frameIndex"] = start_frame
    initial["x"] = float(initial["x"])
    initial["y"] = float(initial["y"])
    initial["confidence"] = float(initial.get("confidence", 1.0))
    if axis is not None and 0 <= start_frame < axis.size:
        initial["mjd"] = float(axis[start_frame])
    points = [initial]
    low_confidence_count = 0
    current_frame = np.asarray(frame_getter(start_frame), dtype=float)
    template = _sample_tracking_patch(current_frame, (float(initial["x"]), float(initial["y"])), pr)

    for target_frame in range(start_frame + step, terminal + step, step):
        if callable(should_stop) and bool(should_stop()):
            return points, "active"
        predicted = _tracking_prediction(points, target_frame)
        target = np.asarray(frame_getter(target_frame), dtype=float)
        if target.ndim != 2:
            raise ValueError("Tracking frames must be two-dimensional")
        ny, nx = target.shape
        if (
            predicted[0] < pr or predicted[1] < pr
            or predicted[0] > nx - pr - 1 or predicted[1] > ny - pr - 1
            or not _point_inside_tracking_roi(predicted, roi_pixels)
        ):
            return points, "stopped-edge"

        x0 = max(0, int(math.floor(predicted[0])) - sr - pr)
        x1 = min(nx, int(math.floor(predicted[0])) + sr + pr + 2)
        y0 = max(0, int(math.floor(predicted[1])) - sr - pr)
        y1 = min(ny, int(math.floor(predicted[1])) + sr + pr + 2)
        search = target[y0:y1, x0:x1]
        clean_template = np.nan_to_num(template, nan=float(np.nanmedian(template)))
        clean_search = np.nan_to_num(search, nan=float(np.nanmedian(search)))
        if clean_search.shape[0] < clean_template.shape[0] or clean_search.shape[1] < clean_template.shape[1]:
            return points, "stopped-edge"
        correlation = match_template(clean_search, clean_template, pad_input=False)
        if correlation.size == 0 or not np.isfinite(correlation).any():
            confidence = -1.0
            peak_y = peak_x = 0
        else:
            peak_y, peak_x = np.unravel_index(int(np.nanargmax(correlation)), correlation.shape)
            confidence = float(correlation[peak_y, peak_x])
        if callable(on_progress):
            on_progress()
        if confidence < float(confidence_threshold):
            low_confidence_count += 1
            if low_confidence_count >= 2:
                return points, "stopped-low-confidence"
            continue
        low_confidence_count = 0
        dx = _parabolic_peak(correlation[peak_y, :], peak_x)
        dy = _parabolic_peak(correlation[:, peak_x], peak_y)
        matched = np.array([
            x0 + peak_x + pr + dx,
            y0 + peak_y + pr + dy,
        ], dtype=float)
        if not _point_inside_tracking_roi(matched, roi_pixels):
            return points, "stopped-edge"
        point = _tracking_point(
            target_frame,
            float(matched[0]),
            float(matched[1]),
            confidence,
            False,
            float(axis[target_frame]) if axis is not None and 0 <= target_frame < axis.size else None,
        )
        points.append(point)
        template = _sample_tracking_patch(target, (float(matched[0]), float(matched[1])), pr)
    return points, "active"


def constrained_ncc_segment(
    frame_getter: object,
    earlier_anchor: dict[str, object],
    later_anchor: dict[str, object],
    patch_radius: int = 6,
    search_radius: int = 18,
    confidence_threshold: float = 0.5,
    roi_pixels: np.ndarray | None = None,
    times_mjd: object | None = None,
) -> list[dict[str, object]]:
    """Re-track and blend a segment so both anchors pass through exactly.

    :param frame_getter: Callable returning one processed image frame.
    :type frame_getter: callable
    :param earlier_anchor: Authoritative earlier anchor.
    :type earlier_anchor: dict[str, object]
    :param later_anchor: Authoritative later anchor.
    :type later_anchor: dict[str, object]
    :param patch_radius: Template radius in pixels.
    :type patch_radius: int
    :param search_radius: Prediction-centered search radius in pixels.
    :type search_radius: int
    :param confidence_threshold: Minimum accepted NCC peak.
    :type confidence_threshold: float
    :param roi_pixels: Optional ROI polygon in pixels.
    :type roi_pixels: numpy.ndarray or None
    :param times_mjd: Optional native MJD axis.
    :type times_mjd: object or None
    :returns: Blended points with the two anchors copied exactly.
    :rtype: list[dict[str, object]]
    """
    start = int(earlier_anchor["frameIndex"])
    end = int(later_anchor["frameIndex"])
    if end <= start:
        raise ValueError("Constrained anchors must have increasing frame indices")
    first = _tracking_point(
        start,
        float(earlier_anchor["x"]),
        float(earlier_anchor["y"]),
        float(earlier_anchor.get("confidence", 1.0)),
        bool(earlier_anchor.get("isAnchor", True)),
        float(earlier_anchor.get("mjd", float("nan"))),
    )
    last = _tracking_point(
        end,
        float(later_anchor["x"]),
        float(later_anchor["y"]),
        float(later_anchor.get("confidence", 1.0)),
        bool(later_anchor.get("isAnchor", True)),
        float(later_anchor.get("mjd", float("nan"))),
    )
    forward, _ = track_ncc_pass(
        frame_getter, first, end, 1, patch_radius, search_radius,
        confidence_threshold, roi_pixels, times_mjd,
    )
    backward, _ = track_ncc_pass(
        frame_getter, last, start, -1, patch_radius, search_radius,
        confidence_threshold, roi_pixels, times_mjd,
    )
    forward_by_frame = {int(point["frameIndex"]): point for point in forward}
    backward_by_frame = {int(point["frameIndex"]): point for point in backward}
    result: list[dict[str, object]] = [first]
    axis = None if times_mjd is None else np.asarray(times_mjd, dtype=float)
    for frame_index in range(start + 1, end):
        left = forward_by_frame.get(frame_index)
        right = backward_by_frame.get(frame_index)
        if left is None and right is None:
            continue
        if left is None:
            point = dict(right or {})
        elif right is None:
            point = dict(left)
        else:
            weight = (frame_index - start) / (end - start)
            point = _tracking_point(
                frame_index,
                (1.0 - weight) * float(left["x"]) + weight * float(right["x"]),
                (1.0 - weight) * float(left["y"]) + weight * float(right["y"]),
                (1.0 - weight) * float(left["confidence"]) + weight * float(right["confidence"]),
                False,
                float(axis[frame_index]) if axis is not None and frame_index < axis.size else None,
            )
        point["isAnchor"] = False
        result.append(point)
    result.append(last)
    return result


def tracking_csv_rows(
    tracks: list[dict[str, object]],
    times_mjd: object,
    pixel_to_world: object,
    km_per_arcsec: float = 725.0,
    target: list[list[float]] | None = None,
) -> list[dict[str, object]]:
    """Flatten tracks and derive centered five-point local-polynomial velocities.

    A quadratic least-squares fit over each complete centered five-point
    stencil is evaluated at the center time. Track ends are emitted as NaN.

    :param tracks: Canonical nested tracks.
    :type tracks: list[dict[str, object]]
    :param times_mjd: Native MJD time axis.
    :type times_mjd: object
    :param pixel_to_world: Callable accepting ``(frame_index, points)``.
    :type pixel_to_world: callable
    :param km_per_arcsec: Arcsec-to-km factor, fixed to 725 at 1 AU by default.
    :type km_per_arcsec: float
    :returns: Rows ordered by track then frame for ``TRACKING_FIELDS``.
    :rtype: list[dict[str, object]]
    """
    axis = np.asarray(times_mjd, dtype=float)
    target_array = np.asarray(target, dtype=float) if target and len(target) >= 3 else None

    def point_inside(point: np.ndarray) -> bool:
        if target_array is None:
            return False
        result = False
        for index in range(len(target_array)):
            left = target_array[index - 1]
            right = target_array[index]
            denominator = right[1] - left[1]
            if ((left[1] > point[1]) != (right[1] > point[1])) and abs(denominator) > 1e-12 and point[0] < (right[0] - left[0]) * (point[1] - left[1]) / denominator + left[0]:
                result = not result
        return result

    def boundary_distance(point: np.ndarray) -> float:
        if target_array is None:
            return float("nan")
        values: list[float] = []
        for index in range(len(target_array)):
            left = target_array[index - 1]
            right = target_array[index]
            segment = right - left
            denominator = float(np.dot(segment, segment))
            fraction = float(np.clip(np.dot(point - left, segment) / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
            values.append(float(np.linalg.norm(point - (left + fraction * segment))))
        return min(values, default=float("nan"))
    rows: list[dict[str, object]] = []
    for track in tracks:
        points = sorted(
            [point for point in track.get("points", []) if isinstance(point, dict)],
            key=lambda point: int(point.get("frameIndex", 0)),
        )
        if not points:
            continue
        world = np.asarray([
            np.asarray(pixel_to_world(
                int(point["frameIndex"]),
                np.asarray([[float(point["x"]), float(point["y"])]], dtype=float),
            ), dtype=float)[0]
            for point in points
        ])
        times = np.asarray([
            float(point.get("mjd", axis[int(point["frameIndex"])]))
            if np.isfinite(float(point.get("mjd", float("nan"))))
            else float(axis[int(point["frameIndex"])])
            for point in points
        ])
        raw_distances = np.asarray([boundary_distance(point) for point in world], dtype=float)
        distances = raw_distances.copy()
        arrival_mjd: float | None = None
        if target_array is not None:
            for index, point in enumerate(world):
                if point_inside(point) or raw_distances[index] <= 1e-9:
                    if index > 0 and raw_distances[index - 1] > 1e-9:
                        previous = float(times[index - 1])
                        current = float(times[index])
                        fraction = float(np.clip(raw_distances[index - 1] / max(1e-12, raw_distances[index - 1] + raw_distances[index]), 0.0, 1.0))
                        arrival_mjd = previous + fraction * (current - previous)
                    else:
                        arrival_mjd = float(times[index])
                    distances[index:] = 0.0
                    break
        velocity = np.full((len(points), 2), np.nan, dtype=float)
        for index in range(2, len(points) - 2):
            stencil_times = (times[index - 2:index + 3] - times[index]) * 86400.0
            design = np.column_stack([np.ones(5), stencil_times, stencil_times**2])
            velocity[index] = np.linalg.lstsq(design, world[index - 2:index + 3], rcond=None)[0][1]
        for index, point in enumerate(points):
            frame_index = int(point["frameIndex"])
            speed = float(np.hypot(*velocity[index])) if np.isfinite(velocity[index]).all() else float("nan")
            rows.append({
                "track_id": str(track.get("id", "")),
                "label": str(track.get("label", track.get("id", ""))),
                "source_id": str(track.get("sourceId", "context")),
                "frame_index": frame_index,
                "time_utc": Time(times[index], format="mjd").isot,
                "time_mjd": float(times[index]),
                "x_px": float(point["x"]),
                "y_px": float(point["y"]),
                "x_arcsec": float(world[index, 0]),
                "y_arcsec": float(world[index, 1]),
                "confidence": float(point.get("confidence", 1.0)),
                "is_anchor": bool(point.get("isAnchor", False)),
                "vx_arcsec_s": float(velocity[index, 0]),
                "vy_arcsec_s": float(velocity[index, 1]),
                "speed_arcsec_s": speed,
                "speed_km_s": speed * float(km_per_arcsec),
                "dist_to_target_arcsec": float(distances[index]) if np.isfinite(distances[index]) else "",
                "arrival_utc": Time(arrival_mjd, format="mjd").isot if arrival_mjd is not None else "",
            })
    return rows


def _time_color_values(times_mjd: list[float]) -> tuple[np.ndarray, mdates.DateFormatter]:
    dates = mdates.date2num(Time(times_mjd, format="mjd").to_datetime())
    return dates, mdates.DateFormatter("%H:%M")


def _sample_pixels(shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    return np.array(
        [
            [0.0, 0.0],
            [nx - 1.0, 0.0],
            [0.0, ny - 1.0],
            [nx - 1.0, ny - 1.0],
            [(nx - 1.0) / 2.0, (ny - 1.0) / 2.0],
        ],
        dtype=float,
    )


def _fit_pixel_to_world_affine(pixels: np.ndarray, world_arcsec: np.ndarray) -> np.ndarray:
    design = np.column_stack([pixels, np.ones(pixels.shape[0])])
    return np.linalg.lstsq(design, world_arcsec, rcond=None)[0]


@dataclass
class AiaCube:
    """AIA running-ratio cube and WCS metadata."""

    intensity_path: Path
    diff_path: Path
    intensity_cube: np.ndarray = field(init=False)
    diff_cube: np.ndarray = field(init=False)
    metas: list[dict[str, object]] = field(init=False)
    times: Time = field(init=False)
    shape: tuple[int, int] = field(init=False)
    _mean_cache: OrderedDict[tuple[float | None, float | None], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _radial_base_map: np.ndarray | None = field(default=None, init=False, repr=False)
    _radial_factor_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _temporal_raw_frame_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _temporal_raw_frame_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _temporal_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _texture_cache: OrderedDict[tuple[object, ...], bytes] = field(default_factory=OrderedDict, init=False)

    def __post_init__(self) -> None:
        self.diff_cube, self.metas, self.times = self._load_mapseq(self.diff_path)
        try:
            self.intensity_cube, _, _ = self._load_mapseq(self.intensity_path)
        except Exception:
            self.intensity_cube = self.diff_cube
        self.shape = tuple(int(v) for v in self.diff_cube.shape[:2])

    @staticmethod
    def _load_mapseq(path: Path) -> tuple[np.ndarray, list[dict[str, object]], Time]:
        if not path.exists():
            raise FileNotFoundError(path)
        frames: list[np.ndarray] = []
        metas: list[dict[str, object]] = []
        times: list[str] = []
        with h5py.File(path, "r") as handle:
            maps = handle["map_sequence"]
            for index in _map_indices(maps):
                item = maps[f"map_{index}"]
                meta = _coerce_meta(json.loads(item.attrs["meta"]))
                data = np.asarray(item["data"], dtype=np.float32)
                frames.append(data)
                metas.append(meta)
                times.append(str(meta.get("date-obs") or meta.get("date_obs") or meta.get("t_obs")))
        cube = np.stack(frames, axis=2)
        return cube, metas, Time(times, format="isot")

    @property
    def nt(self) -> int:
        return int(self.diff_cube.shape[2])

    def map_for_frame(self, time_index: int, data: np.ndarray | None = None):
        idx = int(np.clip(time_index, 0, self.nt - 1))
        return Map(self.diff_cube[:, :, idx] if data is None else data, self.metas[idx])

    def _mean_reference(self, start_mjd: float | None, end_mjd: float | None) -> np.ndarray:
        cache_key = (
            None if start_mjd is None else float(start_mjd),
            None if end_mjd is None else float(end_mjd),
        )
        cached = _ordered_get(self._mean_cache, cache_key)
        if cached is not None:
            return cached
        times = np.asarray(self.times.mjd, dtype=float)
        start = float(np.nanmin(times) if start_mjd is None else start_mjd)
        end = float(np.nanmax(times) if end_mjd is None else end_mjd)
        if start > end:
            start, end = end, start
        selected = np.flatnonzero((times >= start) & (times <= end))
        if selected.size == 0:
            selected = np.arange(self.nt)
        result = np.nanmean(self.intensity_cube[:, :, selected], axis=2)
        return _ordered_put(self._mean_cache, cache_key, result, MEAN_CACHE_LIMIT)

    def _independent_frame(
        self,
        time_index: int,
        operation: str,
        reference: str,
        diff_seconds: float,
        mean_start_mjd: float | None,
        mean_end_mjd: float | None,
    ) -> np.ndarray:
        idx = int(np.clip(time_index, 0, self.nt - 1))
        current = self.intensity_cube[:, :, idx]
        if operation == "none":
            return current
        ref = _difference_reference(reference)
        if ref == "base":
            denominator = self.intensity_cube[:, :, 0]
        elif ref == "mean":
            denominator = self._mean_reference(mean_start_mjd, mean_end_mjd)
        else:
            target = self.times.mjd[idx] - float(diff_seconds) / 86400.0
            previous = int(np.nanargmin(np.abs(self.times.mjd - target)))
            denominator = self.intensity_cube[:, :, previous]
        if operation == "ratio":
            return _safe_ratio(current, denominator)
        return current - denominator

    def _legacy_frame(
        self,
        time_index: int,
        difference_mode: str = "running",
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
    ) -> np.ndarray:
        operation = _difference_operation(difference_operation)
        if operation is not None:
            return self._independent_frame(time_index, operation, difference_reference, diff_seconds, mean_start_mjd, mean_end_mjd)
        idx = int(np.clip(time_index, 0, self.nt - 1))
        if difference_mode == "none":
            return self.intensity_cube[:, :, idx]
        if difference_mode == "base":
            return self.intensity_cube[:, :, idx] - self.intensity_cube[:, :, 0]
        return self.diff_cube[:, :, idx]

    def frame(
        self,
        time_index: int,
        difference_mode: str = "running",
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        radial_gamma: float = 0.0,
        temporal_mode: str = "none",
        temporal_sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
        temporal_sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
    ) -> np.ndarray:
        """Return one AIA frame with optional radial and temporal filters.

        :param radial_gamma: Radial coronal enhancement exponent.
        :type radial_gamma: float
        :param temporal_mode: ``none``, ``lowpass``, or ``bandpass``.
        :type temporal_mode: str
        :param temporal_sigma_short: Short-arm Gaussian sigma in seconds.
        :type temporal_sigma_short: float
        :param temporal_sigma_long: Long-arm Gaussian sigma in seconds.
        :type temporal_sigma_long: float
        :returns: The selected, optionally filtered image array.
        :rtype: numpy.ndarray
        """
        idx = int(np.clip(time_index, 0, self.nt - 1))
        operation = _difference_operation(difference_operation)
        mode = _temporal_mode(temporal_mode)
        if mode != "none" and operation != "none":
            data = _temporal_frame_for_source(
                self,
                idx,
                self.times,
                lambda index: self._legacy_frame(
                    index,
                    difference_mode,
                    difference_operation,
                    difference_reference,
                    diff_seconds,
                    mean_start_mjd,
                    mean_end_mjd,
                ),
                (
                    difference_mode,
                    operation,
                    difference_reference,
                    diff_seconds,
                    mean_start_mjd,
                    mean_end_mjd,
                ),
                mode,
                temporal_sigma_short,
                temporal_sigma_long,
            )
        else:
            data = self._legacy_frame(
                idx,
                difference_mode,
                difference_operation,
                difference_reference,
                diff_seconds,
                mean_start_mjd,
                mean_end_mjd,
            )
        if float(radial_gamma) > 0.0:
            factor = _source_radial_factor(self, self.metas[0], self.shape, radial_gamma)
            return np.multiply(data, factor, dtype=np.float32)
        return data

    def texture(
        self,
        time_index: int,
        vmin: float,
        vmax: float,
        cmap: str,
        scale: str,
        orientation: str = AIA_DISPLAY_ORIENTATION,
        use_difference: bool = True,
        difference_mode: str | None = None,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        radial_gamma: float = 0.0,
        temporal_mode: str = "none",
        temporal_sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
        temporal_sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
        max_width: int | None = None,
        max_height: int | None = None,
        cache_durable: bool = True,
    ) -> bytes:
        """Render an AIA cube frame as a PNG.

        :param max_width: Maximum rendered width, or ``None`` for full size.
        :type max_width: int or None
        :param max_height: Maximum rendered height, or ``None`` for full size.
        :type max_height: int or None
        :param cache_durable: Flush the cache file before atomic replacement.
        :type cache_durable: bool
        :returns: PNG bytes for the processed frame.
        :rtype: bytes
        """
        if not cache_durable:
            self._warm_cache_active = True
        idx = int(np.clip(time_index, 0, self.nt - 1))
        mode = _difference_mode(difference_mode, use_difference)
        operation = _difference_operation(difference_operation)
        radial = _quantized_radial_gamma(radial_gamma)
        temporal = _temporal_mode(temporal_mode)
        key = _cache_key(
            idx,
            mode,
            operation,
            difference_reference,
            diff_seconds,
            mean_start_mjd,
            mean_end_mjd,
            radial,
            temporal,
            temporal_sigma_short,
            temporal_sigma_long,
            vmin,
            vmax,
            cmap,
            scale,
            orientation,
            max_width,
            max_height,
        )
        cached = _cache_get(self._texture_cache, key)
        if cached is not None:
            return cached
        disk_key = _render_disk_key(self, "aia-frame", key)
        cached = RENDER_DISK_CACHE.get(disk_key)
        if cached is not None:
            return _cache_put(self._texture_cache, key, cached)
        frame_data = self.frame(
            idx,
            mode,
            operation,
            difference_reference,
            diff_seconds,
            mean_start_mjd,
            mean_end_mjd,
            radial,
            temporal,
            temporal_sigma_short,
            temporal_sigma_long,
        )
        content = _render_png(
            downsample_to_cap(frame_data, max_width, max_height),
            vmin,
            vmax,
            cmap,
            scale,
            orientation,
        )
        RENDER_DISK_CACHE.put(disk_key, content, durable=cache_durable)
        return _cache_put(self._texture_cache, key, content)

    def pixel_to_world(self, time_index: int, points: np.ndarray) -> np.ndarray:
        smap = self.map_for_frame(time_index)
        x = points[:, 0] * u.pix
        y = points[:, 1] * u.pix
        coords = smap.pixel_to_world(x, y)
        return np.column_stack([coords.Tx.to_value(u.arcsec), coords.Ty.to_value(u.arcsec)])

    def world_to_pixel(self, time_index: int, points_arcsec: np.ndarray) -> np.ndarray:
        smap = self.map_for_frame(time_index)
        coords = SkyCoord(points_arcsec[:, 0] * u.arcsec, points_arcsec[:, 1] * u.arcsec, frame=smap.coordinate_frame)
        pix = smap.world_to_pixel(coords)
        return np.column_stack([pix.x.to_value(u.pix), pix.y.to_value(u.pix)])

    def corners_arcsec(self) -> dict[str, float]:
        ny, nx = self.shape
        corners = self.pixel_to_world(0, np.array([[0, 0], [nx - 1, ny - 1]], dtype=float))
        return {
            "xMin": float(corners[0, 0]),
            "xMax": float(corners[1, 0]),
            "yMin": float(corners[0, 1]),
            "yMax": float(corners[1, 1]),
        }

    def pixel_to_world_affine(self, time_index: int = 0) -> np.ndarray:
        pixels = _sample_pixels(self.shape)
        world = self.pixel_to_world(time_index, pixels)
        return _fit_pixel_to_world_affine(pixels, world)

    def km_per_arcsec(self) -> float:
        meta = self.metas[0]
        try:
            return float(meta["rsun_ref"]) / float(meta["rsun_obs"]) / 1000.0
        except Exception:
            return 725.0


@dataclass
class AiaFitsSequence:
    """Lazy AIA FITS sequence loader used by ``aia-fits-sequence`` manifests."""

    directory: Path
    pattern: str = "*.fits*"
    wavelength: object | None = None
    extension: int = 1
    time_key: str = "T_OBS"
    files: list[Path] = field(init=False)
    times: Time = field(init=False)
    metas: list[fits.Header] = field(init=False)
    header: fits.Header = field(init=False)
    shape: tuple[int, int] = field(init=False)
    _data_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict, init=False)
    _data_inflight: dict[int, Future] = field(default_factory=dict, init=False, repr=False)
    _data_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _mean_cache: OrderedDict[tuple[float | None, float | None], np.ndarray] = field(default_factory=OrderedDict, init=False)
    _radial_base_map: np.ndarray | None = field(default=None, init=False, repr=False)
    _radial_factor_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _temporal_raw_frame_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _temporal_raw_frame_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _temporal_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _texture_cache: OrderedDict[tuple[object, ...], bytes] = field(default_factory=OrderedDict, init=False)

    def __post_init__(self) -> None:
        if not self.directory.exists() or not self.directory.is_dir():
            raise FileNotFoundError(self.directory)
        pattern = str(self.pattern or "*.fits*")
        if "{wavelength}" in pattern or "{wave}" in pattern:
            wavelength = self.wavelength if self.wavelength is not None else "*"
            pattern = pattern.replace("{wavelength}", str(wavelength)).replace("{wave}", str(wavelength))
        candidates = sorted(self.directory.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No AIA FITS files in {self.directory} matching {pattern}")
        records: list[tuple[Path, float, fits.Header, tuple[int, int]]] = []
        for path in candidates:
            with fits.open(path, memmap=False) as hdul:
                ext = int(self.extension)
                if ext < 0 or ext >= len(hdul) or hdul[ext].shape is None:
                    ext = 0
                shape = tuple(int(v) for v in (hdul[ext].shape or ()))
                if len(shape) != 2:
                    raise ValueError(f"AIA FITS image must be 2-D: {path}")
                header = hdul[ext].header.copy()
                value = None
                for candidate_header in (header, hdul[0].header):
                    for key in (self.time_key, "T_OBS", "DATE-OBS", "DATE_OBS"):
                        if key and key in candidate_header:
                            value = candidate_header[key]
                            break
                        for existing in candidate_header.keys():
                            if str(existing).upper() == str(key).upper():
                                value = candidate_header[existing]
                                break
                        if value is not None:
                            break
                    if value is not None:
                        break
                parsed = _parse_mjd(value)
                if parsed is None:
                    raise ValueError(f"Missing valid {self.time_key}/DATE-OBS in {path}")
                records.append((path, parsed, header, (shape[0], shape[1])))
        records.sort(key=lambda item: (item[1], item[0].name))
        self.files = [item[0] for item in records]
        self.times = Time([item[1] for item in records], format="mjd")
        self.metas = [item[2] for item in records]
        self.header = self.metas[0].copy()
        self.shape = records[0][3]

    @property
    def nt(self) -> int:
        return len(self.files)

    @property
    def intensity_path(self) -> Path:
        return self.directory

    @property
    def diff_path(self) -> Path:
        return self.directory

    @property
    def format(self) -> str:
        return "aia-fits-sequence"

    def _read_file(self, index: int) -> np.ndarray:
        idx = int(np.clip(index, 0, self.nt - 1))
        with self._data_lock:
            cached = _ordered_get(self._data_cache, idx)
            if cached is not None:
                return cached
            pending = self._data_inflight.get(idx)
            if pending is None:
                pending = Future()
                self._data_inflight[idx] = pending
                owner = True
            else:
                owner = False
        if not owner:
            return pending.result()
        try:
            decoded_key = DECODED_PLANE_STORE.source_key(
                self.files[idx],
                {"source": "aia-fits", "hdu": int(self.extension), "shape": self.shape},
            )
            data = DECODED_PLANE_STORE.read(decoded_key, self.shape)
            if data is None:
                with fits.open(self.files[idx], memmap=False) as hdul:
                    ext = int(self.extension)
                    if ext < 0 or ext >= len(hdul) or hdul[ext].data is None:
                        ext = 0
                    decoded = np.asarray(hdul[ext].data, dtype=np.float32)
                data = DECODED_PLANE_STORE.write(decoded_key, decoded)
            with self._data_lock:
                cache_limit = WARM_DATA_CACHE_LIMIT if getattr(self, "_warm_cache_active", False) else DATA_CACHE_LIMIT
                _ordered_put(self._data_cache, idx, data, cache_limit)
            pending.set_result(data)
            return data
        except Exception as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._data_lock:
                self._data_inflight.pop(idx, None)

    def map_for_frame(self, time_index: int, data: np.ndarray | None = None):
        idx = int(np.clip(time_index, 0, self.nt - 1))
        return Map(self._read_file(idx) if data is None else data, self.metas[idx])

    def pixel_to_world(self, time_index: int, points: np.ndarray) -> np.ndarray:
        smap = self.map_for_frame(time_index)
        coords = smap.pixel_to_world(points[:, 0] * u.pix, points[:, 1] * u.pix)
        return np.column_stack([coords.Tx.to_value(u.arcsec), coords.Ty.to_value(u.arcsec)])

    def world_to_pixel(self, time_index: int, points_arcsec: np.ndarray) -> np.ndarray:
        smap = self.map_for_frame(time_index)
        coords = SkyCoord(points_arcsec[:, 0] * u.arcsec, points_arcsec[:, 1] * u.arcsec, frame=smap.coordinate_frame)
        pix = smap.world_to_pixel(coords)
        return np.column_stack([pix.x.to_value(u.pix), pix.y.to_value(u.pix)])

    def corners_arcsec(self) -> dict[str, float]:
        ny, nx = self.shape
        corners = self.pixel_to_world(0, np.array([[0, 0], [nx - 1, ny - 1]], dtype=float))
        return {"xMin": float(corners[0, 0]), "xMax": float(corners[1, 0]), "yMin": float(corners[0, 1]), "yMax": float(corners[1, 1])}

    def pixel_to_world_affine(self, time_index: int = 0) -> np.ndarray:
        pixels = _sample_pixels(self.shape)
        return _fit_pixel_to_world_affine(pixels, self.pixel_to_world(time_index, pixels))

    def km_per_arcsec(self) -> float:
        try:
            return float(self.metas[0]["RSUN_REF"]) / float(self.metas[0]["RSUN_OBS"]) / 1000.0
        except Exception:
            return 725.0

    def _mean_reference(self, start_mjd: float | None, end_mjd: float | None) -> np.ndarray:
        cache_key = (None if start_mjd is None else float(start_mjd), None if end_mjd is None else float(end_mjd))
        cached = _ordered_get(self._mean_cache, cache_key)
        if cached is not None:
            return cached
        times = np.asarray(self.times.mjd, dtype=float)
        start = float(np.nanmin(times) if start_mjd is None else start_mjd)
        end = float(np.nanmax(times) if end_mjd is None else end_mjd)
        if start > end:
            start, end = end, start
        selected = np.flatnonzero((times >= start) & (times <= end))
        if selected.size == 0:
            selected = np.arange(self.nt)
        first = self._read_file(int(selected[0]))
        total = np.zeros(first.shape, dtype=np.float64)
        count = np.zeros(first.shape, dtype=np.int32)
        for selected_index in selected:
            frame = self._read_file(int(selected_index))
            finite = np.isfinite(frame)
            total[finite] += frame[finite]
            count[finite] += 1
        result = np.full(first.shape, np.nan, dtype=np.float32)
        np.divide(total, count, out=result, where=count > 0)
        return _ordered_put(self._mean_cache, cache_key, result, MEAN_CACHE_LIMIT)

    def _legacy_frame(
        self,
        time_index: int,
        difference_mode: str = "running",
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
    ) -> np.ndarray:
        idx = int(np.clip(time_index, 0, self.nt - 1))
        operation = _difference_operation(difference_operation)
        if operation is None:
            mode = _difference_mode(difference_mode)
            if mode == "none":
                return self._read_file(idx)
            operation = "subtract"
            difference_reference = "base" if mode == "base" else "previous"
        current = self._read_file(idx)
        if operation == "none":
            return current
        reference = _difference_reference(difference_reference)
        if reference == "base":
            denominator = self._read_file(0)
        elif reference == "mean":
            denominator = self._mean_reference(mean_start_mjd, mean_end_mjd)
        else:
            target = self.times.mjd[idx] - float(diff_seconds) / 86400.0
            previous = int(np.nanargmin(np.abs(self.times.mjd - target)))
            denominator = self._read_file(previous)
        return _safe_ratio(current, denominator) if operation == "ratio" else current - denominator

    def frame(
        self,
        time_index: int,
        difference_mode: str = "running",
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        radial_gamma: float = 0.0,
        temporal_mode: str = "none",
        temporal_sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
        temporal_sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
    ) -> np.ndarray:
        """Return one lazy AIA FITS frame with optional enhancements.

        :param radial_gamma: Radial coronal enhancement exponent.
        :type radial_gamma: float
        :param temporal_mode: ``none``, ``lowpass``, or ``bandpass``.
        :type temporal_mode: str
        :param temporal_sigma_short: Short-arm Gaussian sigma in seconds.
        :type temporal_sigma_short: float
        :param temporal_sigma_long: Long-arm Gaussian sigma in seconds.
        :type temporal_sigma_long: float
        :returns: The selected, optionally filtered image array.
        :rtype: numpy.ndarray
        """
        idx = int(np.clip(time_index, 0, self.nt - 1))
        operation = _difference_operation(difference_operation)
        mode = _temporal_mode(temporal_mode)
        if mode != "none" and operation != "none":
            data = _temporal_frame_for_source(
                self,
                idx,
                self.times,
                lambda index: self._legacy_frame(
                    index,
                    difference_mode,
                    difference_operation,
                    difference_reference,
                    diff_seconds,
                    mean_start_mjd,
                    mean_end_mjd,
                ),
                (
                    difference_mode,
                    operation,
                    difference_reference,
                    diff_seconds,
                    mean_start_mjd,
                    mean_end_mjd,
                ),
                mode,
                temporal_sigma_short,
                temporal_sigma_long,
            )
        else:
            data = self._legacy_frame(
                idx,
                difference_mode,
                difference_operation,
                difference_reference,
                diff_seconds,
                mean_start_mjd,
                mean_end_mjd,
            )
        if float(radial_gamma) > 0.0:
            factor = _source_radial_factor(self, self.metas[0], self.shape, radial_gamma)
            return np.multiply(data, factor, dtype=np.float32)
        return data

    def texture(
        self,
        time_index: int,
        vmin: float,
        vmax: float,
        cmap: str,
        scale: str,
        orientation: str = AIA_DISPLAY_ORIENTATION,
        use_difference: bool = True,
        difference_mode: str | None = None,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        radial_gamma: float = 0.0,
        temporal_mode: str = "none",
        temporal_sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
        temporal_sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
        max_width: int | None = None,
        max_height: int | None = None,
        cache_durable: bool = True,
    ) -> bytes:
        """Render a lazy AIA FITS frame as a PNG.

        :param max_width: Maximum rendered width, or ``None`` for full size.
        :type max_width: int or None
        :param max_height: Maximum rendered height, or ``None`` for full size.
        :type max_height: int or None
        :param cache_durable: Flush the cache file before atomic replacement.
        :type cache_durable: bool
        :returns: PNG bytes for the processed frame.
        :rtype: bytes
        """
        if not cache_durable:
            self._warm_cache_active = True
        idx = int(np.clip(time_index, 0, self.nt - 1))
        mode = _difference_mode(difference_mode, use_difference)
        operation = _difference_operation(difference_operation)
        radial = _quantized_radial_gamma(radial_gamma)
        temporal = _temporal_mode(temporal_mode)
        key = _cache_key(
            idx,
            mode,
            operation,
            difference_reference,
            diff_seconds,
            mean_start_mjd,
            mean_end_mjd,
            radial,
            temporal,
            temporal_sigma_short,
            temporal_sigma_long,
            vmin,
            vmax,
            cmap,
            scale,
            orientation,
            max_width,
            max_height,
        )
        cached = _cache_get(self._texture_cache, key)
        if cached is not None:
            return cached
        disk_key = _render_disk_key(self, "aia-frame", key)
        cached = RENDER_DISK_CACHE.get(disk_key)
        if cached is not None:
            return _cache_put(self._texture_cache, key, cached)
        data = self.frame(
            idx,
            mode,
            operation,
            difference_reference,
            diff_seconds,
            mean_start_mjd,
            mean_end_mjd,
            radial,
            temporal,
            temporal_sigma_short,
            temporal_sigma_long,
        )
        data = downsample_to_cap(data, max_width, max_height)
        content = _render_png(data, vmin, vmax, cmap, scale, orientation)
        RENDER_DISK_CACHE.put(disk_key, content, durable=cache_durable)
        return _cache_put(self._texture_cache, key, content)


@dataclass
class EovsaSequence:
    """EOVSA all-band FITS sequence with lazy frame reads."""

    fits_dir: Path
    file_pattern: str | None = None
    pattern: str | None = None
    files: list[Path] = field(init=False)
    times: Time = field(init=False)
    header: fits.Header = field(init=False)
    freqs_hz: np.ndarray = field(init=False)
    freq_widths_hz: np.ndarray = field(init=False)
    shape: tuple[int, int] = field(init=False)
    _data_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict, init=False)
    _band_cache: OrderedDict[tuple[int, int], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _band_cache_bytes: int = field(default=0, init=False, repr=False)
    _data_inflight: dict[int, Future] = field(default_factory=dict, init=False, repr=False)
    _band_cube_inflight: dict[int, Future] = field(default_factory=dict, init=False, repr=False)
    _data_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _mean_cache: OrderedDict[tuple[float | None, float | None], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _mean_band_cache: OrderedDict[tuple[int, float | None, float | None], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _radial_base_map: np.ndarray | None = field(default=None, init=False, repr=False)
    _radial_factor_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _temporal_raw_frame_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _temporal_raw_frame_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _temporal_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _cube_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _none_peak_rows: np.ndarray | None = field(default=None, init=False, repr=False)
    _none_peak_valid: np.ndarray | None = field(default=None, init=False, repr=False)
    _loaded_peak_rows: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _loaded_peak_valid: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _peak_table_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _peak_progress_local: object = field(default_factory=local, init=False, repr=False)
    _texture_cache: OrderedDict[tuple[object, ...], bytes] = field(default_factory=OrderedDict, init=False)

    def __post_init__(self) -> None:
        file_pattern = self.file_pattern or self.pattern or "EOVSA.20220118T*.500.allbd.fits"
        self.files = sorted(self.fits_dir.glob(file_pattern))
        if not self.files:
            raise FileNotFoundError(f"No EOVSA FITS files in {self.fits_dir} matching {file_pattern}")
        dates = []
        with fits.open(self.files[0]) as hdul:
            self.header = hdul[1].header.copy()
            self.shape = tuple(int(v) for v in hdul[1].shape[-2:])
            table = hdul[2].data
            self.freqs_hz = np.asarray(table["cfreqs"], dtype=float)
            self.freq_widths_hz = np.asarray(table["cdelts"], dtype=float)
        for path in self.files:
            with fits.open(path) as hdul:
                header = hdul[1].header
                dates.append(header.get("DATE-OBS") or header.get("T_OBS") or header.get("DATE_OBS"))
        self.times = Time(dates, format="isot")

    @property
    def nfreq(self) -> int:
        return int(len(self.freqs_hz))

    def _ensure_none_peak_rows(self) -> tuple[np.ndarray, np.ndarray]:
        if self._none_peak_rows is None or self._none_peak_rows.shape != (len(self.files), self.nfreq):
            self._none_peak_rows = np.full((len(self.files), self.nfreq), np.nan, dtype=np.float32)
            self._none_peak_valid = np.zeros(len(self.files), dtype=bool)
        assert self._none_peak_valid is not None
        return self._none_peak_rows, self._none_peak_valid

    def _remember_none_peak_row(self, index: int, data: np.ndarray) -> None:
        rows, valid = self._ensure_none_peak_rows()
        idx = int(index)
        if 0 <= idx < len(valid) and not valid[idx]:
            rows[idx] = np.nanmax(data, axis=(1, 2))
            valid[idx] = True

    def _ensure_loaded_peak_rows(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        rows = self._loaded_peak_rows.get(key)
        valid = self._loaded_peak_valid.get(key)
        if rows is None or valid is None or rows.shape != (len(self.files), self.nfreq):
            rows = np.full((len(self.files), self.nfreq), np.nan, dtype=np.float32)
            valid = np.zeros(len(self.files), dtype=bool)
            self._loaded_peak_rows[key] = rows
            self._loaded_peak_valid[key] = valid
        return rows, valid

    def _remember_loaded_peak_row(self, key: str, index: int, data: np.ndarray) -> None:
        rows, valid = self._ensure_loaded_peak_rows(key)
        idx = int(index)
        if 0 <= idx < len(valid) and not valid[idx]:
            rows[idx] = np.nanmax(data, axis=(1, 2))
            valid[idx] = True

    def _read_file(self, index: int) -> np.ndarray:
        idx = int(np.clip(index, 0, len(self.files) - 1))
        if self._cube_cache is not None and self._cube_cache.shape[0] == len(self.files):
            self._remember_none_peak_row(idx, self._cube_cache[idx])
            return self._cube_cache[idx]
        with self._data_lock:
            cached = _ordered_get(self._data_cache, idx)
            if cached is not None:
                self._remember_none_peak_row(idx, cached)
                return cached
            pending = self._data_inflight.get(idx)
            if pending is None:
                pending = Future()
                self._data_inflight[idx] = pending
                owner = True
            else:
                owner = False
        if not owner:
            return pending.result()
        try:
            expected_shape = (self.nfreq, *self.shape)
            decoded_key = DECODED_PLANE_STORE.source_key(
                self.files[idx],
                {"source": "eovsa-fits", "hdu": 1, "shape": expected_shape},
            )
            data = DECODED_PLANE_STORE.read(decoded_key, expected_shape)
            if data is None:
                with fits.open(self.files[idx]) as hdul:
                    decoded = np.asarray(hdul[1].data, dtype=np.float32)
                data = DECODED_PLANE_STORE.write(decoded_key, decoded)
            self._remember_none_peak_row(idx, data)
            with self._data_lock:
                cache_limit = WARM_DATA_CACHE_LIMIT if getattr(self, "_warm_cache_active", False) else EOVSA_DATA_CACHE_LIMIT
                _ordered_put(self._data_cache, idx, data, cache_limit)
            pending.set_result(data)
            return data
        except Exception as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._data_lock:
                self._data_inflight.pop(idx, None)

    def _read_band(self, index: int, freq_index: int) -> np.ndarray:
        """Read one EOVSA frequency plane through the decoded-cube store.

        :param index: Native EOVSA time-file index.
        :type index: int
        :param freq_index: Frequency-plane index.
        :type freq_index: int
        :returns: The requested ``(ny, nx)`` image plane.
        :rtype: numpy.ndarray
        """
        idx = int(np.clip(index, 0, len(self.files) - 1))
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        key = (idx, fidx)
        if self._cube_cache is not None and self._cube_cache.shape[0] == len(self.files):
            self._remember_none_peak_row(idx, self._cube_cache[idx])
            return self._cube_cache[idx, fidx]
        with self._data_lock:
            cached = _ordered_get(self._band_cache, key)
            if cached is not None:
                return cached
            full = _ordered_get(self._data_cache, idx)
            if full is not None:
                self._remember_none_peak_row(idx, full)
                data = full[fidx]
            else:
                data = None
        if data is None:
            data = self._read_file(idx)[fidx]
        return self._cache_band_plane(key, data)

    def _cache_band_plane(self, key: tuple[int, int], data: np.ndarray) -> np.ndarray:
        """Insert one decoded radio plane into the byte-bounded LRU."""
        # Copy views from a decoded cube so evicted planes do not keep the
        # entire FITS cube alive through NumPy's base pointer.
        value = np.array(data, dtype=np.float32, copy=True)
        size = int(value.nbytes)
        if size > EOVSA_PLANE_CACHE_BYTES:
            return value
        with self._data_lock:
            previous = self._band_cache.pop(key, None)
            if previous is not None:
                self._band_cache_bytes -= int(previous.nbytes)
            while self._band_cache and self._band_cache_bytes + size > EOVSA_PLANE_CACHE_BYTES:
                _, evicted = self._band_cache.popitem(last=False)
                self._band_cache_bytes -= int(evicted.nbytes)
            self._band_cache[key] = value
            self._band_cache_bytes += size
        return value

    def _ensure_band_planes(self, index: int, freq_indices: object | None = None) -> None:
        """Decode one radio cube once and retain its frequency planes.

        Contour requests consume every frequency band for a native time. A
        compressed FITS extension may decompress the complete cube even for a
        ``section`` read; decoding once and splitting the result avoids one
        decompression per band while retaining the exact float32 values.

        On a genuine miss the full cube is still decoded once and persisted
        whole (unchanged behavior, needed regardless of how many planes were
        requested). But when the decoded-plane store already has the cube on
        disk, this reads only the requested planes directly off the store's
        memory map instead of materializing (copying) the whole cube -- see
        the narrow-channel fast path below -- since :meth:`DecodedPlaneStore.
        read` would otherwise copy every one of the 52 channels just to
        discard the ones this call did not ask for.

        :param index: Native EOVSA time-file index.
        :type index: int
        :param freq_indices: Optional iterable of required frequency indices.
            Missing cubes are decoded and persisted once, after which every
            requested plane is read from the memory-mapped cube.
        :type freq_indices: object or None
        """
        idx = int(np.clip(index, 0, len(self.files) - 1))
        requested = sorted({
            int(np.clip(value, 0, self.nfreq - 1))
            for value in (range(self.nfreq) if freq_indices is None else freq_indices)
        })
        if not requested:
            return
        with self._data_lock:
            if all((idx, fidx) in self._band_cache for fidx in requested):
                for fidx in requested:
                    self._band_cache.move_to_end((idx, fidx))
                return
            pending = self._band_cube_inflight.get(idx)
            if pending is None:
                pending = Future()
                self._band_cube_inflight[idx] = pending
                owner = True
            else:
                owner = False
        if not owner:
            pending.result()
            return self._ensure_band_planes(idx, requested)
        try:
            with self._data_lock:
                cube_cache_hit = self._cube_cache is not None and self._cube_cache.shape[0] == len(self.files)
                cached_cube = None if cube_cache_hit else _ordered_get(self._data_cache, idx)
            if cube_cache_hit:
                self._remember_none_peak_row(idx, self._cube_cache[idx])
                cube = self._cube_cache[idx]
                for fidx in requested:
                    if fidx < int(cube.shape[0]):
                        self._cache_band_plane((idx, fidx), cube[fidx])
            elif cached_cube is not None:
                self._remember_none_peak_row(idx, cached_cube)
                for fidx in requested:
                    if fidx < int(cached_cube.shape[0]):
                        self._cache_band_plane((idx, fidx), cached_cube[fidx])
            else:
                # Neither the whole-cube warm cache nor the small in-memory
                # LRU has this frame. Try a plane-sparing store hit before
                # paying for a full decode: DecodedPlaneStore.read()/
                # _materialize() np.array()-copies the ENTIRE mapped cube
                # (all 52 channels, ~13 MB) even when only a couple of
                # planes are requested, which is exactly the cost a narrow
                # (2-6 channel) time-distance extraction should not have to
                # pay on every native frame once the cube is already on
                # disk. Slicing the still-open memmap ourselves reads only
                # the requested planes' bytes (~0.25 MB each).
                expected_shape = (self.nfreq, *self.shape)
                decoded_key = DECODED_PLANE_STORE.source_key(
                    self.files[idx],
                    {"source": "eovsa-fits", "hdu": 1, "shape": expected_shape},
                )
                mapped = DECODED_PLANE_STORE.get(decoded_key, expected_shape)
                if mapped is not None:
                    # The none-peak-row bookkeeping (see loaded_global_band_peaks)
                    # is intentionally skipped on this fast path: it is
                    # already a lazy, best-effort cache with its own
                    # explicit backfill sweep (the ``refresh`` branch of
                    # peak_table), not a per-touch correctness requirement.
                    try:
                        for fidx in requested:
                            if fidx < int(mapped.shape[0]):
                                self._cache_band_plane(
                                    (idx, fidx), np.array(mapped[fidx], dtype=np.float32, copy=True)
                                )
                    finally:
                        mapped._mmap.close()
                else:
                    cube = self._read_file(idx)
                    for fidx in requested:
                        if fidx < int(cube.shape[0]):
                            self._cache_band_plane((idx, fidx), cube[fidx])
            pending.set_result(None)
        except Exception as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._data_lock:
                self._band_cube_inflight.pop(idx, None)

    def nearest_time_index(self, mjd: float) -> int:
        return int(np.nanargmin(np.abs(self.times.mjd - mjd)))

    def previous_index(self, index: int, diff_seconds: float) -> int:
        target = self.times.mjd[int(index)] - float(diff_seconds) / 86400.0
        return int(np.nanargmin(np.abs(self.times.mjd - target)))

    def previous_indices(self, diff_seconds: float) -> np.ndarray:
        times = np.asarray(self.times.mjd, dtype=float)
        targets = times - float(diff_seconds) / 86400.0
        right = np.searchsorted(times, targets, side="left")
        right = np.clip(right, 0, len(times) - 1)
        left = np.clip(right - 1, 0, len(times) - 1)
        use_right = np.abs(times[right] - targets) < np.abs(times[left] - targets)
        return np.where(use_right, right, left).astype(int)

    def frame(self, eovsa_index: int, freq_index: int) -> np.ndarray:
        idx = int(np.clip(eovsa_index, 0, len(self.files) - 1))
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        return self._read_file(idx)[fidx]

    def diff_frame(self, eovsa_index: int, freq_index: int, diff_seconds: float) -> np.ndarray:
        idx = int(np.clip(eovsa_index, 0, len(self.files) - 1))
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        prev_idx = self.previous_index(idx, diff_seconds)
        return self._read_file(idx)[fidx] - self._read_file(prev_idx)[fidx]

    def diff_frame_for_aia(self, aia_time_mjd: float, freq_index: int, diff_seconds: float) -> tuple[np.ndarray, int]:
        idx = self.nearest_time_index(aia_time_mjd)
        return self.diff_frame(idx, freq_index, diff_seconds), idx

    def mode_frame(self, eovsa_index: int, freq_index: int, diff_seconds: float, difference_mode: str) -> np.ndarray:
        idx = int(np.clip(eovsa_index, 0, len(self.files) - 1))
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        return self.mode_data(idx, diff_seconds, difference_mode)[fidx]

    def _band_mode_data(self, eovsa_index: int, freq_index: int, diff_seconds: float, difference_mode: str) -> np.ndarray:
        """Compute a legacy difference mode from one-band FITS reads."""
        idx = int(np.clip(eovsa_index, 0, len(self.files) - 1))
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        mode = _difference_mode(difference_mode)
        current = self._read_band(idx, fidx)
        if mode == "none":
            return current
        reference = self._read_band(0 if mode == "base" else self.previous_index(idx, diff_seconds), fidx)
        return current - reference

    def mode_data(self, eovsa_index: int, diff_seconds: float, difference_mode: str) -> np.ndarray:
        idx = int(np.clip(eovsa_index, 0, len(self.files) - 1))
        mode = _difference_mode(difference_mode)
        if mode == "none":
            return self._read_file(idx)
        if mode == "base":
            data = self._read_file(idx) - self._read_file(0)
        else:
            data = self._read_file(idx) - self._read_file(self.previous_index(idx, diff_seconds))
        self._remember_loaded_peak_row(self.peak_table_cache_key(diff_seconds, mode), idx, data)
        return data

    def _mean_data(self, start_mjd: float | None, end_mjd: float | None) -> np.ndarray:
        cache_key = (
            None if start_mjd is None else float(start_mjd),
            None if end_mjd is None else float(end_mjd),
        )
        cached = _ordered_get(self._mean_cache, cache_key)
        if cached is not None:
            return cached
        times = np.asarray(self.times.mjd, dtype=float)
        start = float(np.nanmin(times) if start_mjd is None else start_mjd)
        end = float(np.nanmax(times) if end_mjd is None else end_mjd)
        if start > end:
            start, end = end, start
        selected = np.flatnonzero((times >= start) & (times <= end))
        if selected.size == 0:
            selected = np.arange(len(self.files))
        if self._cube_cache is not None and self._cube_cache.shape[0] == len(self.files):
            return np.nanmean(self._cube_cache[selected], axis=0)
        total: np.ndarray | None = None
        count = 0
        for selected_index in selected:
            frame = self._read_file(int(selected_index))
            total = frame.astype(np.float64) if total is None else total + frame
            count += 1
        if total is None or count == 0:
            raise ValueError("No EOVSA frames available for mean reference")
        return _ordered_put(self._mean_cache, cache_key, (total / count).astype(np.float32), MEAN_CACHE_LIMIT)

    def _mean_band(self, freq_index: int, start_mjd: float | None, end_mjd: float | None) -> np.ndarray:
        """Compute and cache a mean reference for one frequency plane."""
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        cache_key = (
            fidx,
            None if start_mjd is None else float(start_mjd),
            None if end_mjd is None else float(end_mjd),
        )
        cached = _ordered_get(self._mean_band_cache, cache_key)
        if cached is not None:
            return cached
        times = np.asarray(self.times.mjd, dtype=float)
        start = float(np.nanmin(times) if start_mjd is None else start_mjd)
        end = float(np.nanmax(times) if end_mjd is None else end_mjd)
        if start > end:
            start, end = end, start
        selected = np.flatnonzero((times >= start) & (times <= end))
        if selected.size == 0:
            selected = np.arange(len(self.files))
        first = self._read_band(int(selected[0]), fidx)
        total = np.zeros(first.shape, dtype=np.float64)
        count = np.zeros(first.shape, dtype=np.int32)
        for selected_index in selected:
            frame = self._read_band(int(selected_index), fidx)
            finite = np.isfinite(frame)
            total[finite] += frame[finite]
            count[finite] += 1
        result = np.full(first.shape, np.nan, dtype=np.float32)
        np.divide(total, count, out=result, where=count > 0)
        return _ordered_put(self._mean_band_cache, cache_key, result, MEAN_CACHE_LIMIT)

    def _band_operation_data(
        self,
        eovsa_index: int,
        freq_index: int,
        diff_seconds: float,
        operation: str,
        reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
    ) -> np.ndarray:
        """Compute an independent operation from targeted one-band reads."""
        idx = int(np.clip(eovsa_index, 0, len(self.files) - 1))
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        current = self._read_band(idx, fidx)
        operation = _difference_operation(operation) or "none"
        if operation == "none":
            return current
        ref = _difference_reference(reference)
        if ref == "base":
            denominator = self._read_band(0, fidx)
        elif ref == "mean":
            denominator = self._mean_band(fidx, mean_start_mjd, mean_end_mjd)
        else:
            denominator = self._read_band(self.previous_index(idx, diff_seconds), fidx)
        if operation == "ratio":
            return _safe_ratio(current, denominator)
        return current - denominator

    def operation_data(
        self,
        eovsa_index: int,
        diff_seconds: float,
        operation: str,
        reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        remember_peak: bool = True,
    ) -> np.ndarray:
        """Return one full-cube difference operation.

        ``remember_peak`` is disabled by contour rendering because the
        current-frame contour level computes its own per-band maxima.  Avoiding
        that duplicate reduction preserves the array result while removing a
        sizeable scan from every warm contour identity.

        :param remember_peak: Populate the lazy peak-table row cache.
        :type remember_peak: bool
        """
        idx = int(np.clip(eovsa_index, 0, len(self.files) - 1))
        current = self._read_file(idx)
        operation = _difference_operation(operation) or "none"
        if operation == "none":
            return current
        ref = _difference_reference(reference)
        if ref == "base":
            denominator = self._read_file(0)
        elif ref == "mean":
            denominator = self._mean_data(mean_start_mjd, mean_end_mjd)
        else:
            denominator = self._read_file(self.previous_index(idx, diff_seconds))
        if operation == "ratio":
            result = _safe_ratio(current, denominator)
        else:
            result = current - denominator
        key = self._operation_peak_table_cache_key(
            diff_seconds, operation, reference, mean_start_mjd, mean_end_mjd
        )
        if remember_peak:
            self._remember_loaded_peak_row(key, idx, result)
        return result

    def loaded_global_band_peaks(self, diff_seconds: float, difference_mode: str) -> list[float]:
        mode = _difference_mode(difference_mode)
        if mode == "none":
            rows, valid = self._ensure_none_peak_rows()
        else:
            rows, valid = self._ensure_loaded_peak_rows(self.peak_table_cache_key(diff_seconds, mode))
        if not np.any(valid):
            return [0.0] * self.nfreq
        return _finite_float_list(np.nanmax(rows[valid], axis=0))

    def map_for_data(self, data: np.ndarray):
        return Map(data, self.header)

    def pixel_to_world(self, data: np.ndarray, points: np.ndarray, x_offset: float, y_offset: float) -> np.ndarray:
        smap = self.map_for_data(data)
        coords = smap.pixel_to_world(points[:, 0] * u.pix, points[:, 1] * u.pix)
        return np.column_stack([
            coords.Tx.to_value(u.arcsec) + float(x_offset),
            coords.Ty.to_value(u.arcsec) + float(y_offset),
        ])

    def world_to_pixel(self, data: np.ndarray, points_arcsec: np.ndarray, x_offset: float, y_offset: float) -> np.ndarray:
        smap = self.map_for_data(data)
        shifted = np.column_stack([
            points_arcsec[:, 0] - float(x_offset),
            points_arcsec[:, 1] - float(y_offset),
        ])
        coords = SkyCoord(shifted[:, 0] * u.arcsec, shifted[:, 1] * u.arcsec, frame=smap.coordinate_frame)
        pix = smap.world_to_pixel(coords)
        return np.column_stack([pix.x.to_value(u.pix), pix.y.to_value(u.pix)])

    def frame_for_aia_time(
        self,
        aia_time_mjd: float,
        freq_index: int,
        diff_seconds: float,
        use_running_diff: bool = True,
        difference_mode: str | None = None,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        full_cube: bool = False,
        eovsa_index: int | None = None,
        radial_gamma: float = 0.0,
        temporal_mode: str = "none",
        temporal_sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
        temporal_sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
    ) -> np.ndarray:
        """Return one processed radio frame before display normalization."""
        idx = (
            int(np.clip(eovsa_index, 0, len(self.files) - 1))
            if eovsa_index is not None
            else self.nearest_time_index(aia_time_mjd)
        )
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        mode = _difference_mode(difference_mode, use_running_diff)
        operation = _difference_operation(difference_operation)
        temporal = _temporal_mode(temporal_mode)
        if temporal != "none":
            if operation is None:
                frame_getter = lambda index: self._band_mode_data(index, fidx, diff_seconds, mode)
            else:
                frame_getter = lambda index: self._band_operation_data(
                    index,
                    fidx,
                    diff_seconds,
                    operation,
                    difference_reference,
                    mean_start_mjd,
                    mean_end_mjd,
                )
            data = _temporal_frame_for_source(
                self,
                idx,
                self.times,
                frame_getter,
                (
                    fidx,
                    diff_seconds,
                    mode,
                    operation,
                    difference_reference,
                    mean_start_mjd,
                    mean_end_mjd,
                ),
                temporal,
                temporal_sigma_short,
                temporal_sigma_long,
            )
        elif full_cube and operation is None:
            data = self.mode_frame(idx, fidx, diff_seconds, mode)
        elif full_cube:
            data = self.operation_data(idx, diff_seconds, operation, difference_reference, mean_start_mjd, mean_end_mjd)[fidx]
        elif operation is None:
            data = self._band_mode_data(idx, fidx, diff_seconds, mode)
        else:
            data = self._band_operation_data(
                idx,
                fidx,
                diff_seconds,
                operation,
                difference_reference,
                mean_start_mjd,
                mean_end_mjd,
            )
        if float(radial_gamma) > 0.0:
            factor = _source_radial_factor(self, self.header, self.shape, radial_gamma)
            data = np.multiply(data, factor, dtype=np.float32)
        return data

    def texture_for_aia_time(
        self,
        aia_time_mjd: float,
        freq_index: int,
        diff_seconds: float,
        vmin: float,
        vmax: float,
        cmap: str,
        scale: str,
        orientation: str = EOVSA_DISPLAY_ORIENTATION,
        use_running_diff: bool = True,
        difference_mode: str | None = None,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        full_cube: bool = False,
        eovsa_index: int | None = None,
        radial_gamma: float = 0.0,
        temporal_mode: str = "none",
        temporal_sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
        temporal_sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
        max_width: int | None = None,
        max_height: int | None = None,
        cache_durable: bool = True,
    ) -> bytes:
        """Render one radio texture, optionally using a resolved native index.

        :param aia_time_mjd: Legacy context time used when ``eovsa_index`` is
            omitted.
        :type aia_time_mjd: float
        :param eovsa_index: Already-resolved native radio index.
        :type eovsa_index: int or None
        :param max_width: Maximum rendered width, or ``None`` for full size.
        :type max_width: int or None
        :param max_height: Maximum rendered height, or ``None`` for full size.
        :type max_height: int or None
        :param cache_durable: Flush the cache file before atomic replacement.
        :type cache_durable: bool
        :returns: PNG bytes for the selected radio plane.
        :rtype: bytes
        """
        if not cache_durable:
            self._warm_cache_active = True
        idx = (
            int(np.clip(eovsa_index, 0, len(self.files) - 1))
            if eovsa_index is not None
            else self.nearest_time_index(aia_time_mjd)
        )
        fidx = int(np.clip(freq_index, 0, self.nfreq - 1))
        mode = _difference_mode(difference_mode, use_running_diff)
        operation = _difference_operation(difference_operation)
        radial = _quantized_radial_gamma(radial_gamma)
        temporal = _temporal_mode(temporal_mode)
        key = _cache_key(
            idx,
            fidx,
            diff_seconds,
            mode,
            operation,
            difference_reference,
            mean_start_mjd,
            mean_end_mjd,
            full_cube,
            radial,
            temporal,
            temporal_sigma_short,
            temporal_sigma_long,
            vmin,
            vmax,
            cmap,
            scale,
            orientation,
            max_width,
            max_height,
        )
        cached = _cache_get(self._texture_cache, key)
        if cached is not None:
            return cached
        disk_key = _render_disk_key(self, "radio-frame", key)
        cached = RENDER_DISK_CACHE.get(disk_key)
        if cached is not None:
            return _cache_put(self._texture_cache, key, cached)
        data = self.frame_for_aia_time(
            aia_time_mjd,
            freq_index,
            diff_seconds,
            use_running_diff,
            difference_mode,
            difference_operation,
            difference_reference,
            mean_start_mjd,
            mean_end_mjd,
            full_cube,
            eovsa_index,
            radial_gamma,
            temporal_mode,
            temporal_sigma_short,
            temporal_sigma_long,
        )
        data = downsample_to_cap(data, max_width, max_height)
        content = _render_png(data, vmin, vmax, cmap, scale, orientation)
        RENDER_DISK_CACHE.put(disk_key, content, durable=cache_durable)
        return _cache_put(self._texture_cache, key, content)

    @staticmethod
    def peak_table_cache_key(diff_seconds: float, difference_mode: str) -> str:
        return f"{_difference_mode(difference_mode)}:dt={float(diff_seconds):.3f}"

    @staticmethod
    def _operation_peak_table_cache_key(
        diff_seconds: float,
        operation: str,
        reference: str,
        mean_start_mjd: float | None,
        mean_end_mjd: float | None,
    ) -> str:
        return f"op={operation}:ref={_difference_reference(reference)}:dt={float(diff_seconds):.3f}:mean={mean_start_mjd}:{mean_end_mjd}"

    def peak_table(
        self,
        diff_seconds: float,
        difference_mode: str,
        refresh: bool = False,
        on_progress: object = None,
    ) -> np.ndarray:
        mode = _difference_mode(difference_mode)
        key = self.peak_table_cache_key(diff_seconds, mode)
        cached = self._peak_table_cache.get(key)
        if cached is not None and not refresh:
            return cached
        total = len(self.files)
        progress_callback = on_progress if callable(on_progress) else getattr(self._peak_progress_local, "callback", None)

        def report(done: int) -> None:
            if callable(progress_callback):
                progress_callback(done, total)

        cache_result = True
        if mode == "none":
            table, valid = self._ensure_none_peak_rows()
            done = int(np.count_nonzero(valid))
            report(done)
            if refresh:
                missing = np.flatnonzero(~valid)
                for idx in missing:
                    self._read_file(int(idx))
                    done += 1
                    report(done)
            cache_result = bool(np.all(valid))
            table = table.copy()
        elif self._cube_cache is not None and self._cube_cache.shape[0] == len(self.files):
            cube = self._cube_cache
            table = np.empty((cube.shape[0], self.nfreq), dtype=np.float32)
            chunk_size = 128
            if mode == "base":
                reference = cube[0]
                for start in range(0, cube.shape[0], chunk_size):
                    stop = min(cube.shape[0], start + chunk_size)
                    diff = cube[start:stop] - reference[None, :, :, :]
                    table[start:stop] = np.nanmax(diff, axis=(2, 3))
                    report(stop)
            else:
                previous = self.previous_indices(diff_seconds)
                for start in range(0, cube.shape[0], chunk_size):
                    stop = min(cube.shape[0], start + chunk_size)
                    diff = cube[start:stop] - cube[previous[start:stop]]
                    table[start:stop] = np.nanmax(diff, axis=(2, 3))
                    report(stop)
        else:
            table, valid = self._ensure_loaded_peak_rows(key)
            done = int(np.count_nonzero(valid))
            report(done)
            if refresh:
                missing = np.flatnonzero(~valid)
                for idx in missing:
                    self.mode_data(int(idx), diff_seconds, mode)
                    done += 1
                    report(done)
            cache_result = bool(np.all(valid))
            table = table.copy()
        if cache_result:
            self._peak_table_cache[key] = table
        return table

    def operation_peak_table(
        self,
        diff_seconds: float,
        operation: str,
        reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        refresh: bool = False,
        on_progress: object = None,
    ) -> np.ndarray:
        operation = _difference_operation(operation) or "none"
        key = self._operation_peak_table_cache_key(
            diff_seconds, operation, reference, mean_start_mjd, mean_end_mjd
        )
        cached = self._peak_table_cache.get(key)
        if cached is not None and not refresh:
            return cached
        progress_callback = on_progress if callable(on_progress) else getattr(self._peak_progress_local, "callback", None)
        table = np.empty((len(self.files), self.nfreq), dtype=np.float32)
        for idx in range(len(self.files)):
            data = self.operation_data(idx, diff_seconds, operation, reference, mean_start_mjd, mean_end_mjd)
            table[idx] = np.nanmax(data, axis=(1, 2))
            if callable(progress_callback):
                progress_callback(idx + 1, len(self.files))
        self._peak_table_cache[key] = table
        return table

    def loaded_global_operation_peaks(
        self,
        diff_seconds: float,
        operation: str,
        reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
    ) -> list[float]:
        """Return per-band peaks from operation frames already read.

        :param diff_seconds: Cadence used for a previous-frame reference.
        :type diff_seconds: float
        :param operation: Difference operation, such as ``subtract`` or ``ratio``.
        :type operation: str
        :param reference: Reference image selection.
        :type reference: str
        :param mean_start_mjd: Optional mean-reference start time.
        :type mean_start_mjd: float or None
        :param mean_end_mjd: Optional mean-reference end time.
        :type mean_end_mjd: float or None
        :returns: Maximum value for each frequency band among loaded frames.
        :rtype: list[float]
        """
        key = self._operation_peak_table_cache_key(
            diff_seconds, _difference_operation(operation) or "none", reference,
            mean_start_mjd, mean_end_mjd,
        )
        rows, valid = self._ensure_loaded_peak_rows(key)
        if not np.any(valid):
            return [0.0] * self.nfreq
        return _finite_float_list(np.nanmax(rows[valid], axis=0))

    def global_operation_peaks(
        self,
        diff_seconds: float,
        operation: str,
        reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        refresh: bool = False,
    ) -> list[float]:
        table = self.operation_peak_table(diff_seconds, operation, reference, mean_start_mjd, mean_end_mjd, refresh=refresh)
        return _finite_float_list(np.nanmax(table, axis=0))

    def global_band_peaks(self, diff_seconds: float, difference_mode: str, refresh: bool = False) -> list[float]:
        table = self.peak_table(diff_seconds, difference_mode, refresh=refresh)
        return _finite_float_list(np.nanmax(table, axis=0))

    def pixel_to_world_affine(self) -> np.ndarray:
        pixels = _sample_pixels(self.shape)
        data = np.zeros(self.shape, dtype=np.float32)
        world = self.pixel_to_world(data, pixels, 0.0, 0.0)
        return _fit_pixel_to_world_affine(pixels, world)


@dataclass
class EovsaSpectrogram:
    """EOVSA total-power dynamic spectrum used by the top overview panel."""

    path: Path
    data: np.ndarray = field(init=False)
    freqs_ghz: np.ndarray = field(init=False)
    times: Time = field(init=False)
    shape: tuple[int, int] = field(init=False)
    _regularized_data: np.ndarray = field(init=False, repr=False)
    _regularized_times_mjd: np.ndarray = field(init=False, repr=False)
    _row_medians: np.ndarray = field(init=False, repr=False)
    _texture_cache: OrderedDict[tuple[object, ...], bytes] = field(default_factory=OrderedDict, init=False)

    def __post_init__(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        with fits.open(self.path) as hdul:
            self.data = np.asarray(hdul[0].data, dtype=np.float32)
            self.freqs_ghz = np.asarray(hdul[1].data["FGHZ"], dtype=float)
            self.times = Time(np.asarray(hdul[2].data["TIME"], dtype=float), format="jd")
        self.shape = tuple(int(v) for v in self.data.shape)
        self._regularized_data, self._regularized_times_mjd = _regularize_spectrogram_time_axis(
            self.data,
            np.asarray(self.times.mjd, dtype=float),
        )
        self._row_medians = _spectrogram_row_medians(self.data)

    def texture(
        self,
        vmin: float = 0.5,
        vmax: float = 150.0,
        cmap: str = "viridis",
        scale: str = "log",
        frequency_scale: str = "linear",
        frequency_min_ghz: float | None = None,
        frequency_max_ghz: float | None = None,
        normalization: str = "none",
    ) -> bytes:
        frequency_scale = _frequency_scale(frequency_scale)
        normalization = _spectrogram_normalization(normalization)
        key = _cache_key(vmin, vmax, cmap, scale, frequency_scale, frequency_min_ghz, frequency_max_ghz, normalization)
        cached = _cache_get(self._texture_cache, key)
        if cached is not None:
            return cached
        disk_key = _render_disk_key(self, "spectrogram", key)
        cached = RENDER_DISK_CACHE.get(disk_key)
        if cached is not None:
            return _cache_put(self._texture_cache, key, cached)
        data = _resample_frequency_axis(
            _normalize_spectrogram_rows(self._regularized_data, self._row_medians, normalization),
            self.freqs_ghz,
            frequency_scale,
            frequency_min_ghz,
            frequency_max_ghz,
        )
        content = _render_png(data, vmin, vmax, cmap, scale, "flip_y")
        RENDER_DISK_CACHE.put(disk_key, content)
        return _cache_put(self._texture_cache, key, content)


@dataclass
class PlaceholderSpectrogram:
    """Metadata-only spectrogram placeholder for unsupported manifest formats."""

    path: Path
    format: str = "unknown"
    shape: tuple[int, int] = (0, 0)
    freqs_ghz: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    times: Time = field(default_factory=lambda: Time([], format="mjd"))

    def texture(self, *args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise ValueError(f"Spectrogram format is not renderable yet: {self.format}")


@dataclass
class SolRadSession:
    """One in-memory app session."""

    session_id: str
    aia: AiaCube | AiaFitsSequence
    eovsa: EovsaSequence
    spectrogram: EovsaSpectrogram | PlaceholderSpectrogram
    seed_path: Path
    output_dir: Path
    progress_registry: ProgressRegistry = field(default_factory=ProgressRegistry, repr=False)
    context_source_id: str = "context"
    radio_source_id: str = "radio"
    spectrogram_source_id: str = "spectrogram"
    context_label: str = "AIA 131 Å"
    radio_label: str = "EOVSA"
    spectrogram_label: str = "EOVSA Dynamic Spectrum"
    source_specs: list[dict[str, object]] = field(default_factory=list)
    roi_world: list[list[float]] | None = None
    correlation_target: list[list[float]] | None = None
    sad_tracks: list[dict[str, object]] = field(default_factory=list)
    eovsa_sources: list[dict[str, object]] = field(default_factory=list)
    feature_tracks: list[dict[str, object]] = field(default_factory=list)
    tracks: list[dict[str, object]] = field(default_factory=list)
    radio_peak_cache: dict[str, list[float]] = field(default_factory=dict)
    radio_peak_table_cache: dict[str, list[list[float]]] = field(default_factory=dict)
    channel_offsets: dict[str, list[float] | list[bool]] = field(default_factory=dict)
    channel_mask: list[bool] = field(default_factory=list)
    _overlay_cache: OrderedDict[tuple[object, ...], bytes] = field(default_factory=OrderedDict, init=False)
    _contour_geometry_cache: OrderedDict[tuple[object, ...], bytes] = field(default_factory=OrderedDict, init=False, repr=False)
    _affine_cache: OrderedDict[tuple[object, ...], np.ndarray] = field(default_factory=OrderedDict, init=False, repr=False)
    _timeseries_cache: OrderedDict[tuple[object, ...], dict[str, object]] = field(default_factory=OrderedDict, init=False, repr=False)
    # Metadata time axes are immutable for a loaded session; avoid reformatting them on every /meta call.
    _meta_time_payload_cache: dict[str, tuple[list[str], list[float]]] = field(default_factory=dict, init=False, repr=False)
    _radio_peak_build_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _radio_peak_inflight: dict[str, Event] = field(default_factory=dict, init=False, repr=False)
    _prewarm_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _prewarm_cancel: Event | None = field(default=None, init=False, repr=False)
    _prewarm_thread: Thread | None = field(default=None, init=False, repr=False)
    _prewarm_done: int = field(default=0, init=False, repr=False)
    _prewarm_total: int = field(default=0, init=False, repr=False)
    _prewarm_active: bool = field(default=False, init=False, repr=False)
    _prewarm_progress_id: str | None = field(default=None, init=False, repr=False)
    _load_progress_id: str | None = field(default=None, init=False, repr=False)
    _tracking_cancel: Event = field(default_factory=Event, init=False, repr=False)
    _slit_extract_cancel: Event = field(default_factory=Event, init=False, repr=False)
    _slit_results: dict[str, dict[str, object]] = field(default_factory=dict, init=False, repr=False)
    view_start_mjd: float | None = None
    view_end_mjd: float | None = None
    default_diff_seconds: float = DEFAULT_DIFF_SECONDS
    legacy_defaults: bool = False
    channel_offsets_version: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the source-level radio offset table for this session."""
        table = _normalize_channel_offsets(self.channel_offsets, self.eovsa.nfreq)
        if self.channel_mask:
            table["masked"] = _normalize_channel_mask(self.channel_mask, self.eovsa.nfreq)
        self.channel_offsets = table
        self.channel_mask = list(table["masked"])  # type: ignore[arg-type]

    def prewarm_status(self) -> dict[str, int | bool]:
        """Return a snapshot of this session's background render progress."""
        with self._prewarm_lock:
            return {
                "done": self._prewarm_done,
                "total": self._prewarm_total,
                "active": self._prewarm_active,
            }

    def cancel_prewarm(self) -> None:
        """Request cancellation of this session's background pre-warm."""
        with self._prewarm_lock:
            cancel = self._prewarm_cancel
        if cancel is not None:
            cancel.set()

    def update_load_progress(self, done: int) -> None:
        """Advance the coarse manifest/session restoration phases."""
        if self._load_progress_id is not None:
            self.progress_registry.update(self._load_progress_id, done=done)

    def finish_load_progress(self) -> None:
        """Remove the coarse load operation after the API payload is ready."""
        if self._load_progress_id is not None:
            self.progress_registry.finish(self._load_progress_id)
            self._load_progress_id = None

    def start_prewarm(
        self,
        foreground_active: object,
        max_width: int = 1024,
        max_height: int = 1024,
        throttle_seconds: float = 0.02,
    ) -> None:
        """Start one throttled, cancellable render-cache pre-warm thread."""
        self.cancel_prewarm()
        start, end = self._default_time_indices()
        indices = list(range(start, end + 1))
        cancel = Event()
        with self._prewarm_lock:
            self._prewarm_cancel = cancel
            self._prewarm_done = 0
            self._prewarm_total = len(indices) * 3
            self._prewarm_active = bool(indices)
            self._prewarm_progress_id = (
                self.progress_registry.start("Warming render cache", total=len(indices) * 3)
                if indices else None
            )

        def wait_for_foreground() -> bool:
            while callable(foreground_active) and bool(foreground_active()):
                if cancel.wait(0.05):
                    return False
            return not cancel.is_set()

        def render_one(kind: str, aia_index: int) -> None:
            if kind == "context-none":
                self.aia.texture(
                    aia_index, 0.5, 1.5, "gray", "linear", AIA_DISPLAY_ORIENTATION,
                    use_difference=False, difference_mode="none", difference_operation="none",
                    diff_seconds=self.default_diff_seconds, max_width=max_width, max_height=max_height,
                )
            elif kind == "context-ratio":
                self.aia.texture(
                    aia_index, 0.5, 1.5, "gray", "linear", AIA_DISPLAY_ORIENTATION,
                    use_difference=True, difference_mode="running", difference_operation="ratio",
                    difference_reference="previous", diff_seconds=self.default_diff_seconds,
                    max_width=max_width, max_height=max_height,
                )
            else:
                aia_mjd = float(self.aia.times[aia_index].mjd)
                eovsa_index = self.eovsa.nearest_time_index(aia_mjd)
                self.eovsa.texture_for_aia_time(
                    aia_mjd, min(9, self.eovsa.nfreq - 1), self.default_diff_seconds,
                    -1.0e6, 5.0e6, "turbo", "linear", EOVSA_DISPLAY_ORIENTATION,
                    use_running_diff=True, difference_mode="running", difference_operation="subtract",
                    difference_reference="previous", eovsa_index=eovsa_index,
                    max_width=max_width, max_height=max_height,
                )

        def run() -> None:
            executor: ThreadPoolExecutor | None = None
            try:
                # Keep the three recipes interleaved by native time so the
                # first visible samples become useful immediately.  A small
                # bounded pool avoids the serial warm-up while leaving CPU and
                # FITS decompression headroom for interactive requests.
                tasks = [
                    (kind, aia_index)
                    for aia_index in indices
                    for kind in ("context-none", "context-ratio", "radio")
                ]
                executor = ThreadPoolExecutor(
                    max_workers=PREWARM_WORKERS,
                    thread_name_prefix=f"sad-prewarm-worker-{self.session_id[:8]}",
                )
                pending: set[Future[None]] = set()
                task_index = 0
                while pending or task_index < len(tasks):
                    if cancel.is_set():
                        for future in pending:
                            future.cancel()
                        break
                    while task_index < len(tasks) and len(pending) < PREWARM_WORKERS:
                        if not wait_for_foreground():
                            break
                        kind, aia_index = tasks[task_index]
                        task_index += 1
                        pending.add(executor.submit(render_one, kind, aia_index))
                    if not pending:
                        if cancel.is_set() or task_index >= len(tasks):
                            break
                        continue
                    finished, pending = wait_futures(
                        pending,
                        timeout=max(0.01, float(throttle_seconds)),
                        return_when=FIRST_COMPLETED,
                    )
                    for future in finished:
                        try:
                            future.result()
                        except Exception:
                            # A missing/corrupt frame should not stop the
                            # other recipes or strand the progress operation.
                            pass
                        with self._prewarm_lock:
                            if self._prewarm_cancel is cancel:
                                self._prewarm_done += 1
                                if self._prewarm_progress_id is not None:
                                    self.progress_registry.update(
                                        self._prewarm_progress_id,
                                        done=self._prewarm_done,
                                    )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
                with self._prewarm_lock:
                    if self._prewarm_cancel is cancel:
                        self._prewarm_active = False
                        if self._prewarm_progress_id is not None:
                            self.progress_registry.finish(self._prewarm_progress_id)
                            self._prewarm_progress_id = None

        thread = Thread(target=run, name=f"sad-prewarm-{self.session_id[:8]}", daemon=True)
        with self._prewarm_lock:
            self._prewarm_thread = thread
        thread.start()

    def channel_mask_payload(self) -> list[bool]:
        """Return a detached canonical copy of the radio display mask."""
        return _channel_mask_for_session(self)

    def channel_offsets_payload(self) -> dict[str, list[float] | list[bool]]:
        """Return a detached canonical copy of the radio offset table.

        :returns: Per-channel x/y offsets and display mask.
        :rtype: dict[str, list[float] | list[bool]]
        """
        table = _normalize_channel_offsets(self.channel_offsets, self.eovsa.nfreq)
        table["masked"] = self.channel_mask_payload()
        return {axis: list(values) for axis, values in table.items()}

    def channel_offset(self, freq_index: int) -> tuple[float, float]:
        """Return one channel's x/y calibration offset in arcsec.

        :param freq_index: Zero-based EOVSA channel index.
        :type freq_index: int
        :returns: ``(dx, dy)`` in arcsec, or zeros for an invalid index.
        :rtype: tuple[float, float]
        """
        index = int(freq_index)
        if index < 0 or index >= self.eovsa.nfreq:
            return 0.0, 0.0
        table = _normalize_channel_offsets(getattr(self, "channel_offsets", None), self.eovsa.nfreq)
        return float(table["dx"][index]), float(table["dy"][index])

    def set_channel_offsets(self, value: object) -> dict[str, list[float] | list[bool]]:
        """Replace the complete source-level radio offset table.

        :param value: Full ``channelOffsets`` mapping in arcsec.
        :type value: object
        :returns: Canonical stored table.
        :rtype: dict[str, list[float]]
        :raises ValueError: If the table is not complete and finite.
        """
        if not isinstance(value, dict):
            raise ValueError("channelOffsets must be an object with dx and dy arrays")
        table = _normalize_channel_offsets(value, self.eovsa.nfreq, strict=True)
        if "masked" not in value and "channelMask" not in value:
            table["masked"] = self.channel_mask_payload()
        self.channel_offsets = table
        self.channel_mask = list(table["masked"])  # type: ignore[arg-type]
        self.channel_offsets_version += 1
        self._overlay_cache.clear()
        self._affine_cache.clear()
        return self.channel_offsets_payload()

    @classmethod
    def create_default(cls) -> "SolRadSession":
        session_id = uuid.uuid4().hex
        output_dir = DEFAULT_OUTPUT_ROOT / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        session = cls(
            session_id=session_id,
            aia=AiaCube(DEFAULT_AIA_INTENSITY, DEFAULT_AIA_DIFF),
            eovsa=EovsaSequence(DEFAULT_EOVSA_DIR),
            spectrogram=EovsaSpectrogram(DEFAULT_EOVSA_SPECTROGRAM),
            seed_path=DEFAULT_SEEDS,
            output_dir=output_dir,
            legacy_defaults=True,
        )
        return session

    @classmethod
    def create_sample(cls) -> "SolRadSession":
        return cls.create_default()

    @staticmethod
    def _manifest_source(manifest: dict[str, object], role: str) -> dict[str, object]:
        sources = manifest.get("sources")
        if not isinstance(sources, list):
            return {}
        for source in sources:
            if isinstance(source, dict) and str(source.get("role", "")).lower() == role:
                return source
        return {}

    @staticmethod
    def _manifest_paths(source: dict[str, object]) -> dict[str, object]:
        paths = source.get("paths")
        if isinstance(paths, dict):
            return paths
        path = source.get("path")
        return {"path": path} if path else {}

    @staticmethod
    def _manifest_time_bounds(manifest: dict[str, object]) -> tuple[float | None, float | None]:
        event = manifest.get("event") if isinstance(manifest.get("event"), dict) else {}
        view_start = event.get("viewStart") or manifest.get("viewStart")
        view_end = event.get("viewEnd") or manifest.get("viewEnd")
        return _parse_mjd(view_start), _parse_mjd(view_end)

    @staticmethod
    def radio_peak_cache_key(
        diff_seconds: float,
        difference_mode: str,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
    ) -> str:
        operation = _difference_operation(difference_operation)
        if operation is not None:
            return f"op={operation}:ref={_difference_reference(difference_reference)}:dt={float(diff_seconds):.3f}:mean={mean_start_mjd}:{mean_end_mjd}"
        mode = _difference_mode(difference_mode)
        return f"{mode}:dt={float(diff_seconds):.3f}"

    @staticmethod
    def _read_peak_cache(cache: object) -> dict[str, list[float]]:
        if not isinstance(cache, dict):
            return {}
        return {
            str(key): _finite_float_list(values)
            for key, values in cache.items()
            if isinstance(values, list)
        }

    @staticmethod
    def _read_peak_table_cache(cache: object) -> dict[str, list[list[float]]]:
        if not isinstance(cache, dict):
            return {}
        return {
            str(key): table
            for key, values in cache.items()
            if isinstance(values, list) and (table := _finite_float_table(values))
        }

    @classmethod
    def create_from_manifest(cls, manifest: dict[str, object]) -> "SolRadSession":
        progress_registry = ProgressRegistry()
        load_progress_id = progress_registry.start("Loading session", total=5)
        context = cls._manifest_source(manifest, "context")
        radio = cls._manifest_source(manifest, "radio")
        spectrogram_source = cls._manifest_source(manifest, "spectrogram")
        progress_registry.update(load_progress_id, done=1)
        root_paths_value = manifest.get("paths")
        if not isinstance(root_paths_value, dict):
            # Saved session JSON calls this map ``data``; use it when a source
            # entry has no per-source path metadata to fall back to.
            root_paths_value = manifest.get("data")
        root_paths = root_paths_value if isinstance(root_paths_value, dict) else {}
        context_paths = cls._manifest_paths(context)
        radio_paths = cls._manifest_paths(radio)
        spectrogram_paths = cls._manifest_paths(spectrogram_source)

        context_format = str(context.get("format") or "hdf").strip().lower()
        context_directory = context_paths.get("directory") or context_paths.get("dir") or context_paths.get("fitsDir") or context_paths.get("path") or context.get("directory") or context.get("dir")
        context_pattern = context.get("pattern") or context.get("filePattern") or context_paths.get("pattern") or context_paths.get("filenamePattern") or context_paths.get("filePattern") or context_paths.get("glob") or "*.fits*"
        context_wavelength = context.get("wavelength") or context_paths.get("wavelength")
        context_extension = context.get("hdu") or context.get("extension") or context_paths.get("hdu") or context_paths.get("extension") or 1
        context_time_key = context.get("timeKey") or context_paths.get("timeKey") or "T_OBS"

        aia_intensity = Path(str(
            context_paths.get("intensity")
            or context_paths.get("aiaIntensity")
            or root_paths.get("aiaIntensity")
            or DEFAULT_AIA_INTENSITY
        ))
        aia_diff = Path(str(
            context_paths.get("diff")
            or context_paths.get("runningDiff")
            or context_paths.get("runningRatio")
            or context_paths.get("aiaDiff")
            or context_paths.get("path")
            or root_paths.get("aiaDiff")
            or DEFAULT_AIA_DIFF
        ))
        eovsa_dir = Path(str(
            radio_paths.get("fitsDir")
            or radio_paths.get("directory")
            or radio_paths.get("path")
            or radio.get("fitsDir")
            or radio.get("directory")
            or radio.get("path")
            or root_paths.get("eovsaFits")
            or DEFAULT_EOVSA_DIR
        ))
        seeds_value = manifest.get("seeds") or root_paths.get("seeds")
        seeds = Path(str(seeds_value)) if seeds_value else Path("missing-seeds.pickle")
        spectrogram_value = spectrogram_paths.get("path") or spectrogram_paths.get("fits") or spectrogram_source.get("path") or spectrogram_source.get("fits") or root_paths.get("eovsaSpectrogram")
        spectrogram_path = Path(str(spectrogram_value)) if spectrogram_value else Path("missing-spectrogram.fits")

        spectrogram_format = str(spectrogram_source.get("format") or spectrogram_path.suffix.lstrip(".") or "unknown").lower()
        if spectrogram_path.exists() and spectrogram_format in {"fits", "fit", "fts"}:
            spectrogram: EovsaSpectrogram | PlaceholderSpectrogram = EovsaSpectrogram(spectrogram_path)
        else:
            spectrogram = PlaceholderSpectrogram(spectrogram_path, format=spectrogram_format)
        progress_registry.update(load_progress_id, done=2)

        session_id = uuid.uuid4().hex
        output_dir = DEFAULT_OUTPUT_ROOT / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        if context_format in {"aia-fits-sequence", "aia_fits_sequence", "aiafits"}:
            if not context_directory:
                raise ValueError("aia-fits-sequence context source requires a directory/path")
            aia: AiaCube | AiaFitsSequence = AiaFitsSequence(
                Path(str(context_directory)),
                pattern=str(context_pattern),
                wavelength=context_wavelength,
                extension=int(context_extension),
                time_key=str(context_time_key),
            )
        else:
            aia = AiaCube(aia_intensity, aia_diff)
        radio_pattern = radio.get("pattern") or radio.get("filePattern") or radio_paths.get("pattern") or radio_paths.get("filenamePattern") or radio_paths.get("filePattern") or radio_paths.get("glob")
        eovsa = EovsaSequence(eovsa_dir, file_pattern=str(radio_pattern) if radio_pattern else None)
        progress_registry.update(load_progress_id, done=3)
        view_start_mjd, view_end_mjd = cls._manifest_time_bounds(manifest)
        defaults = manifest.get("defaults") if isinstance(manifest.get("defaults"), dict) else {}
        default_diff_seconds = float(defaults.get("diffSeconds", DEFAULT_DIFF_SECONDS)) if defaults.get("diffSeconds") is not None else DEFAULT_DIFF_SECONDS
        session = cls(
            session_id=session_id,
            aia=aia,
            eovsa=eovsa,
            spectrogram=spectrogram,
            seed_path=seeds,
            output_dir=output_dir,
            progress_registry=progress_registry,
            context_source_id=str(context.get("id") or "context"),
            radio_source_id=str(radio.get("id") or "radio"),
            spectrogram_source_id=str(spectrogram_source.get("id") or "spectrogram"),
            context_label=str(context.get("label") or context.get("instrument") or "Context Image"),
            radio_label=str(radio.get("label") or radio.get("instrument") or "Radio Cube"),
            spectrogram_label=str(spectrogram_source.get("label") or spectrogram_source.get("instrument") or "Dynamic Spectrum"),
            source_specs=[source for source in manifest.get("sources", []) if isinstance(source, dict)] if isinstance(manifest.get("sources"), list) else [],
            view_start_mjd=view_start_mjd,
            view_end_mjd=view_end_mjd,
            default_diff_seconds=default_diff_seconds,
        )
        session._load_progress_id = load_progress_id
        return session

    @classmethod
    def create_from_state(cls, state: dict[str, object]) -> "SolRadSession":
        if isinstance(state.get("sources"), list):
            session = cls.create_from_manifest(state)
            persisted_offsets = state.get("channelOffsets")
            session.channel_offsets = _normalize_channel_offsets(persisted_offsets, session.eovsa.nfreq)
            for source in state.get("sources", []):
                if isinstance(source, dict) and str(source.get("role", "")).lower() == "radio":
                    if "channelOffsets" in source and "channelOffsets" not in state:
                        session.channel_offsets = _normalize_channel_offsets(source.get("channelOffsets"), session.eovsa.nfreq)
                    if "channelMask" in source and isinstance(session.channel_offsets, dict) and not isinstance(persisted_offsets, dict):
                        session.channel_offsets["masked"] = _normalize_channel_mask(source.get("channelMask"), session.eovsa.nfreq)
                    break
            session.channel_mask = list(session.channel_offsets["masked"])  # type: ignore[arg-type]
            roi_world = state.get("roiWorld")
            if isinstance(roi_world, list):
                session.roi_world = roi_world
            correlation_target = state.get("correlationTarget")
            if isinstance(correlation_target, list):
                session.correlation_target = correlation_target
            sad_tracks = state.get("sadTracks") or state.get("featureTracks")
            if isinstance(sad_tracks, list):
                session.sad_tracks = [row for row in sad_tracks if isinstance(row, dict)]
                session.feature_tracks = [row for row in sad_tracks if isinstance(row, dict)]
            tracks = state.get("tracks")
            if isinstance(tracks, list):
                session.set_tracks([track for track in tracks if isinstance(track, dict)])
            eovsa_sources = state.get("eovsaSources") or state.get("radioSources")
            if isinstance(eovsa_sources, list):
                session.eovsa_sources = [row for row in eovsa_sources if isinstance(row, dict)]
            session.radio_peak_cache = cls._read_peak_cache(state.get("radioPeakCache"))
            session.radio_peak_table_cache = cls._read_peak_table_cache(state.get("radioPeakTableCache"))
            session.write_loaded_outputs()
            session.update_load_progress(4)
            return session
        paths = state.get("data") if isinstance(state.get("data"), dict) else state.get("paths")
        paths = paths if isinstance(paths, dict) else {}
        progress_registry = ProgressRegistry()
        load_progress_id = progress_registry.start("Loading session", total=5)
        progress_registry.update(load_progress_id, done=1)
        session_id = uuid.uuid4().hex
        output_dir = DEFAULT_OUTPUT_ROOT / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        session = cls(
            session_id=session_id,
            aia=AiaCube(
                Path(str(paths.get("aiaIntensity") or DEFAULT_AIA_INTENSITY)),
                Path(str(paths.get("aiaDiff") or DEFAULT_AIA_DIFF)),
            ),
            eovsa=EovsaSequence(Path(str(paths.get("eovsaFits") or DEFAULT_EOVSA_DIR))),
            spectrogram=EovsaSpectrogram(Path(str(paths.get("eovsaSpectrogram") or DEFAULT_EOVSA_SPECTROGRAM))),
            seed_path=Path(str(paths.get("seeds") or DEFAULT_SEEDS)),
            output_dir=output_dir,
            progress_registry=progress_registry,
        )
        session._load_progress_id = load_progress_id
        session.update_load_progress(3)
        session.channel_offsets = _normalize_channel_offsets(state.get("channelOffsets"), session.eovsa.nfreq)
        session.channel_mask = list(session.channel_offsets["masked"])  # type: ignore[arg-type]
        roi_world = state.get("roiWorld")
        if isinstance(roi_world, list):
            session.roi_world = roi_world
        correlation_target = state.get("correlationTarget")
        if isinstance(correlation_target, list):
            session.correlation_target = correlation_target
        sad_tracks = state.get("sadTracks")
        if isinstance(sad_tracks, list):
            session.sad_tracks = [row for row in sad_tracks if isinstance(row, dict)]
            session.feature_tracks = [row for row in sad_tracks if isinstance(row, dict)]
        tracks = state.get("tracks")
        if isinstance(tracks, list):
            session.set_tracks([track for track in tracks if isinstance(track, dict)])
        eovsa_sources = state.get("eovsaSources")
        if isinstance(eovsa_sources, list):
            session.eovsa_sources = [row for row in eovsa_sources if isinstance(row, dict)]
        session.radio_peak_cache = cls._read_peak_cache(state.get("radioPeakCache"))
        session.radio_peak_table_cache = cls._read_peak_table_cache(state.get("radioPeakTableCache"))
        session.write_loaded_outputs()
        session.update_load_progress(4)
        return session

    def _default_time_indices(self) -> tuple[int, int]:
        aia_times = np.asarray(self.aia.times.mjd, dtype=float)
        if aia_times.size == 0:
            return 0, 0
        if self.legacy_defaults:
            start_target = _parse_mjd("2022-01-18T17:30:00") or float(aia_times[0])
            end_target = _parse_mjd("2022-01-18T17:40:00") or float(aia_times[-1])
        else:
            radio_times = np.asarray(self.eovsa.times.mjd, dtype=float)
            start_target = max(float(np.nanmin(aia_times)), float(np.nanmin(radio_times))) if radio_times.size else float(np.nanmin(aia_times))
            end_target = min(float(np.nanmax(aia_times)), float(np.nanmax(radio_times))) if radio_times.size else float(np.nanmax(aia_times))
            if self.view_start_mjd is not None:
                start_target = max(start_target, self.view_start_mjd)
            if self.view_end_mjd is not None:
                end_target = min(end_target, self.view_end_mjd)
            if start_target > end_target:
                start_target = float(np.nanmin(aia_times))
                end_target = float(np.nanmax(aia_times))
        start = int(np.nanargmin(np.abs(aia_times - start_target)))
        end = int(np.nanargmin(np.abs(aia_times - end_target)))
        return (min(start, end), max(start, end))

    def _cached_meta_time_payload(self, key: str, times: object) -> tuple[list[str], list[float]]:
        cached = self._meta_time_payload_cache.get(key)
        if cached is not None:
            return cached
        datetime_values = times.to_datetime() if hasattr(times, "to_datetime") else []
        formatted = [value.strftime("%H:%M:%S") for value in datetime_values]
        mjd = np.asarray(getattr(times, "mjd", []), dtype=float).tolist()
        cached = (formatted, mjd)
        self._meta_time_payload_cache[key] = cached
        return cached

    def api_meta(self) -> dict[str, object]:
        aia_times, aia_time_mjd = self._cached_meta_time_payload("aia", self.aia.times)
        eovsa_times, eovsa_time_mjd = self._cached_meta_time_payload("eovsa", self.eovsa.times)
        spectrogram_times, spectrogram_time_mjd = self._cached_meta_time_payload("spectrogram", self.spectrogram.times)
        default_start, default_end = self._default_time_indices()
        aia_affine = self.aia.pixel_to_world_affine(default_start).tolist()
        eovsa_affine = self.eovsa.pixel_to_world_affine().tolist()
        spectrogram_renderable = isinstance(self.spectrogram, EovsaSpectrogram)
        if isinstance(self.aia, AiaFitsSequence):
            aia_paths: dict[str, object] = {
                "directory": str(self.aia.directory),
                "pattern": self.aia.pattern,
                "hdu": self.aia.extension,
                "timeKey": self.aia.time_key,
            }
            if self.aia.wavelength is not None:
                aia_paths["wavelength"] = self.aia.wavelength
        else:
            aia_paths = {"intensity": str(self.aia.intensity_path), "diff": str(self.aia.diff_path)}
        radio_paths: dict[str, object] = {"fitsDir": str(self.eovsa.fits_dir)}
        if self.eovsa.file_pattern or self.eovsa.pattern:
            radio_paths["pattern"] = self.eovsa.file_pattern or self.eovsa.pattern
        sources = [
            {
                "id": self.context_source_id,
                "role": "context",
                "label": self.context_label,
                "instrument": "AIA" if "AIA" in self.context_label.upper() else self.context_label,
                "format": getattr(self.aia, "format", None) or self.aia.diff_path.suffix.lstrip(".") or "hdf",
                "shape": [self.aia.shape[0], self.aia.shape[1], self.aia.nt],
                "time": {
                    "count": self.aia.nt,
                    "times": aia_times,
                    "timeMjd": aia_time_mjd,
                    "start": aia_times[0] if aia_times else "",
                    "end": aia_times[-1] if aia_times else "",
                },
                "wcs": {"pixelToWorldAffine": aia_affine},
                "capabilities": {"render": True, "difference": True, "tracking": True, "roi": True},
                "status": "ready",
                "paths": aia_paths,
            },
            {
                "id": self.radio_source_id,
                "role": "radio",
                "label": self.radio_label,
                "instrument": "EOVSA" if "EOVSA" in self.radio_label.upper() else self.radio_label,
                "format": "fits",
                "shape": [self.eovsa.nfreq, len(self.eovsa.files), self.eovsa.shape[0], self.eovsa.shape[1]],
                "time": {
                    "count": len(self.eovsa.files),
                    "times": eovsa_times,
                    "timeMjd": eovsa_time_mjd,
                    "start": eovsa_times[0] if eovsa_times else "",
                    "end": eovsa_times[-1] if eovsa_times else "",
                },
                "freqGhz": (self.eovsa.freqs_hz / 1e9).round(3).tolist(),
                "channelOffsets": self.channel_offsets_payload(),
                "channelMask": self.channel_mask_payload(),
                "pol": ["I"],
                "wcs": {"pixelToWorldAffine": eovsa_affine},
                "capabilities": {"render": True, "difference": True, "overlay": True, "centroid": True, "tracking": True, "roi": True},
                "status": "ready",
                "paths": radio_paths,
            },
            {
                "id": self.spectrogram_source_id,
                "role": "spectrogram",
                "label": self.spectrogram_label,
                "instrument": "EOVSA" if "EOVSA" in self.spectrogram_label.upper() else self.spectrogram_label,
                "format": self.spectrogram.path.suffix.lstrip(".") or getattr(self.spectrogram, "format", "unknown"),
                "shape": [self.spectrogram.shape[0], self.spectrogram.shape[1]],
                "time": {
                    "count": int(len(self.spectrogram.times)),
                    "timeMjd": spectrogram_time_mjd,
                    "start": spectrogram_times[0] if spectrogram_times else "",
                    "end": spectrogram_times[-1] if spectrogram_times else "",
                },
                "freqGhz": self.spectrogram.freqs_ghz.round(4).tolist(),
                "capabilities": {"render": spectrogram_renderable},
                "status": "ready" if spectrogram_renderable else "placeholder",
                "paths": {"path": str(self.spectrogram.path)},
            },
        ]
        seen_sources = {str(source["id"]) for source in sources}
        for spec in self.source_specs:
            source_id = str(spec.get("id") or f"source-{len(sources) + 1}")
            if source_id in seen_sources:
                continue
            paths = self._manifest_paths(spec)
            sources.append({
                "id": source_id,
                "role": str(spec.get("role") or "context"),
                "label": str(spec.get("label") or spec.get("instrument") or source_id),
                "instrument": str(spec.get("instrument") or ""),
                "format": str(spec.get("format") or Path(str(paths.get("path") or "")).suffix.lstrip(".") or "unknown"),
                "shape": spec.get("shape") if isinstance(spec.get("shape"), list) else [],
                "time": spec.get("time") if isinstance(spec.get("time"), dict) else {"count": 0},
                "capabilities": {"render": False},
                "status": "placeholder",
                "paths": paths,
            })
            seen_sources.add(source_id)
        return {
            "sessionId": self.session_id,
            "prewarm": self.prewarm_status(),
            "sources": sources,
            "aia": {
                "shape": [self.aia.shape[0], self.aia.shape[1]],
                "times": aia_times,
                "timeMjd": aia_time_mjd,
                "cornersArcsec": self.aia.corners_arcsec(),
            },
            "eovsa": {
                "shape": [self.eovsa.shape[0], self.eovsa.shape[1]],
                "times": eovsa_times,
                "timeMjd": eovsa_time_mjd,
                "freqGhz": (self.eovsa.freqs_hz / 1e9).round(3).tolist(),
                "channelOffsets": self.channel_offsets_payload(),
                "channelMask": self.channel_mask_payload(),
            },
            "wcs": {
                "aia": {
                    "pixelToWorldAffine": aia_affine,
                },
                "eovsa": {
                    "pixelToWorldAffine": eovsa_affine,
                },
            },
            "spectrogram": {
                "shape": [self.spectrogram.shape[0], self.spectrogram.shape[1]],
                "timeMjd": spectrogram_time_mjd,
                "freqGhz": self.spectrogram.freqs_ghz.round(4).tolist(),
                "defaults": {
                    "vmin": 0.5,
                    "vmax": 150.0,
                    "scale": "log",
                    "cmap": "viridis",
                },
            },
            "defaults": {
                "timeStartIndex": default_start,
                "timeEndIndex": default_end,
                "timeIndex": default_start,
                "freqIndex": min(9, self.eovsa.nfreq - 1),
                "xOffsetArcsec": 7.0,
                "yOffsetArcsec": 0.0,
                "diffSeconds": self.default_diff_seconds,
                "selectedContextSourceId": self.context_source_id,
                "selectedRadioSourceId": self.radio_source_id,
                "selectedSpectrogramSourceId": self.spectrogram_source_id,
            },
            "paths": {
                "aiaIntensity": str(self.aia.intensity_path),
                "aiaDiff": str(self.aia.diff_path),
                "seeds": str(self.seed_path),
                "eovsaFits": str(self.eovsa.fits_dir),
                "eovsaSpectrogram": str(self.spectrogram.path),
                "outputs": str(self.output_dir),
            },
            "radioPeakCache": self.radio_peak_cache,
            "radioPeakTableCache": self.radio_peak_table_cache,
            "channelOffsets": self.channel_offsets_payload(),
        }

    def api_loaded_state(self, state: dict[str, object]) -> dict[str, object]:
        """Reconstruct client state with canonical timeline values preferred.

        Version-1 AIA index aliases remain in the returned UI mapping. Canonical
        values are finite, ranges are sorted, and values are clamped to the
        selected master's axis. When a timeline cursor is present, ROI
        projections use separately resolved AIA and EOVSA samples so a dense
        radio master does not lose its native index.

        :param state: Saved session payload containing an optional ``ui`` map.
        :type state: dict[str, object]
        :returns: Restored UI aliases, ROI projections, tracks, and sources.
        :rtype: dict[str, object]
        """
        raw_ui = state.get("ui") if isinstance(state.get("ui"), dict) else {}
        ui = dict(raw_ui)
        defaults = self.api_meta()["defaults"]
        time_index = int(ui.get("timeIndex", defaults["timeIndex"]))
        freq_index = int(ui.get("freqIndex", defaults["freqIndex"]))
        x_offset = float(ui.get("xOffsetArcsec", ui.get("xOffset", 7.0)))
        y_offset = float(ui.get("yOffsetArcsec", ui.get("yOffset", 0.0)))
        diff_seconds = float(ui.get("diffSeconds", DEFAULT_DIFF_SECONDS))
        eovsa_index: int | None = None

        has_timeline = isinstance(ui.get("timeline"), dict)
        timeline = dict(ui["timeline"]) if has_timeline else {}
        master_source_id = str(timeline.get("masterSourceId") or self.context_source_id)
        if master_source_id in {self.radio_source_id, "radio", "eovsa"}:
            master_times = self.eovsa.times
        elif master_source_id in {self.spectrogram_source_id, "spectrogram", "eovsa-spectrogram"}:
            master_times = getattr(self.spectrogram, "times", self.aia.times)
        else:
            master_times = self.aia.times
        master_values = np.asarray(getattr(master_times, "mjd", master_times), dtype=float).reshape(-1)
        master_values = master_values[np.isfinite(master_values)]
        master_min = float(np.min(master_values)) if master_values.size else None
        master_max = float(np.max(master_values)) if master_values.size else None

        def normalized_timeline_mjd(value: object) -> float | None:
            parsed = _parse_mjd(value)
            if parsed is None or not np.isfinite(parsed):
                return None
            if master_min is not None and master_max is not None:
                return float(np.clip(parsed, master_min, master_max))
            return float(parsed)

        cursor_mjd = normalized_timeline_mjd(timeline.get("cursorMjd"))
        if cursor_mjd is not None:
            timeline["cursorMjd"] = cursor_mjd
            aia_cursor = resolve_time_index(self.aia.times, cursor_mjd, "nearest")
            radio_cursor = resolve_time_index(self.eovsa.times, cursor_mjd, "nearest")
            if aia_cursor is not None:
                time_index = aia_cursor[0]
                ui["timeIndex"] = time_index
            if radio_cursor is not None:
                eovsa_index = radio_cursor[0]
        else:
            timeline.pop("cursorMjd", None)

        start_mjd = normalized_timeline_mjd(timeline.get("startMjd"))
        end_mjd = normalized_timeline_mjd(timeline.get("endMjd"))
        if start_mjd is not None and end_mjd is not None:
            start_mjd, end_mjd = sorted((start_mjd, end_mjd))
            timeline["startMjd"] = start_mjd
            timeline["endMjd"] = end_mjd
            aia_start = resolve_time_index(self.aia.times, start_mjd, "nearest")
            aia_end = resolve_time_index(self.aia.times, end_mjd, "nearest")
            if aia_start is not None and aia_end is not None:
                ui["startIndex"] = aia_start[0]
                ui["endIndex"] = aia_end[0]
        else:
            timeline.pop("startMjd", None)
            timeline.pop("endMjd", None)
        if has_timeline:
            ui["timeline"] = timeline
        offsets_getter = getattr(self, "channel_offsets_payload", None)
        if callable(offsets_getter):
            offsets_payload = offsets_getter()
        else:
            nfreq = int(getattr(self.eovsa, "nfreq", 0))
            offsets_payload = _normalize_channel_offsets(getattr(self, "channel_offsets", None), nfreq)
            offsets_payload["masked"] = _channel_mask_for_session(self)
        return {
            "ui": ui,
            # Preserve only the slim source summaries supplied by the saved
            # payload so the frontend can detect a newly extended data extent
            # while rebuilding its timeline from fresh meta times.
            "savedSources": [source for source in state.get("sources", []) if isinstance(source, dict)] if isinstance(state.get("sources"), list) else [],
            "channelOffsets": offsets_payload,
            "roiWorld": self.roi_world or [],
            "correlationTarget": getattr(self, "correlation_target", None) or [],
            "roiAia": self.roi_pixels_for_panel("aia", time_index, freq_index, x_offset, y_offset, diff_seconds),
            "roiEovsa": self.roi_pixels_for_panel(
                "eovsa",
                time_index,
                freq_index,
                x_offset,
                y_offset,
                diff_seconds,
                eovsa_index,
            ),
            "sadTracks": self.sad_tracks,
            "eovsaSources": self.eovsa_sources,
            "featureTracks": self.feature_tracks or self.sad_tracks,
            "tracks": getattr(self, "tracks", []),
            "radioSources": self.eovsa_sources,
            "slits": [slit for slit in state.get("slits", []) if isinstance(slit, dict)] if isinstance(state.get("slits"), list) else [],
            "fan": dict(state["fan"]) if isinstance(state.get("fan"), dict) else None,
        }

    def pixel_timeseries(
        self,
        source_id: str,
        x: float,
        y: float,
        patch_radius: int,
        start_mjd: float,
        end_mjd: float,
        max_points: int,
        freq_index: int = 0,
        difference_mode: str | None = None,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        use_running_diff: bool = True,
        temporal_mode: str | None = None,
        temporal_sigma_short: float = DEFAULT_TEMPORAL_SIGMA_SHORT,
        temporal_sigma_long: float = DEFAULT_TEMPORAL_SIGMA_LONG,
        sampling_policy: str = "nearest",
        max_offset_seconds: float | None = None,
    ) -> dict[str, object]:
        """Return a probed source pixel series using the frame route semantics.

        :param source_id: Renderable context or radio source identifier.
        :type source_id: str
        :param x: Image pixel x coordinate.
        :type x: float
        :param y: Image pixel y coordinate.
        :type y: float
        :param patch_radius: Radius of the square pixel patch to average.
        :type patch_radius: int
        :param start_mjd: Inclusive lower bound on the source-native time axis.
        :type start_mjd: float
        :param end_mjd: Inclusive upper bound on the source-native time axis.
        :type end_mjd: float
        :param max_points: Maximum number of returned native samples.
        :type max_points: int
        :param freq_index: Radio frequency-plane index.
        :type freq_index: int
        :param difference_mode: Legacy frame difference mode.
        :type difference_mode: str or None
        :param difference_operation: Independent frame operation.
        :type difference_operation: str or None
        :param difference_reference: Reference frame selection.
        :type difference_reference: str
        :param diff_seconds: Previous-reference lag in seconds.
        :type diff_seconds: float
        :param mean_start_mjd: Optional mean-reference lower bound.
        :type mean_start_mjd: float or None
        :param mean_end_mjd: Optional mean-reference upper bound.
        :type mean_end_mjd: float or None
        :param use_running_diff: Legacy fallback for ``difference_mode``.
        :type use_running_diff: bool
        :param temporal_mode: Optional ``lowpass`` or ``bandpass`` filter.
        :type temporal_mode: str or None
        :param temporal_sigma_short: Short temporal Gaussian sigma in seconds.
        :type temporal_sigma_short: float
        :param temporal_sigma_long: Long temporal Gaussian sigma in seconds.
        :type temporal_sigma_long: float
        :param sampling_policy: Layer sampling policy retained in the cache key.
        :type sampling_policy: str
        :param max_offset_seconds: Layer sampling tolerance retained in the cache key.
        :type max_offset_seconds: float or None
        :returns: JSON-ready MJD, raw, optional smoothed, statistics, cadence,
            native count, and stride fields.
        :rtype: dict[str, object]
        :raises ValueError: If the source, coordinate, or numeric parameters are invalid.
        """
        if source_id in {self.context_source_id, "context", "aia"}:
            source = self.aia
            role = "context"
            times = np.asarray(self.aia.times.mjd, dtype=float)
        elif source_id in {self.radio_source_id, "radio", "eovsa"}:
            source = self.eovsa
            role = "radio"
            times = np.asarray(self.eovsa.times.mjd, dtype=float)
        else:
            raise ValueError(f"Unknown renderable source: {source_id}")
        if not np.isfinite(float(x)) or not np.isfinite(float(y)):
            raise ValueError("Pixel coordinates must be finite")
        if not 0.0 <= float(x) < float(source.shape[1]) or not 0.0 <= float(y) < float(source.shape[0]):
            raise ValueError("Pixel coordinates are outside the source image")
        if int(patch_radius) < 0:
            raise ValueError("patchRadius must be non-negative")
        if int(max_points) <= 0:
            raise ValueError("maxPoints must be positive")
        if not np.isfinite(float(start_mjd)) or not np.isfinite(float(end_mjd)):
            raise ValueError("startMjd and endMjd must be finite")
        if role == "radio" and not 0 <= int(freq_index) < self.eovsa.nfreq:
            raise ValueError("freqIndex is outside the radio frequency axis")
        normalized_temporal = _temporal_mode(temporal_mode)
        operation = _difference_operation(difference_operation)
        mode = _difference_mode(difference_mode, use_running_diff)
        reference = _difference_reference(difference_reference)
        key = _cache_key(
            source_id,
            float(x),
            float(y),
            int(patch_radius),
            float(start_mjd),
            float(end_mjd),
            int(max_points),
            int(freq_index),
            mode,
            operation,
            reference,
            float(diff_seconds),
            mean_start_mjd,
            mean_end_mjd,
            normalized_temporal,
            float(temporal_sigma_short),
            float(temporal_sigma_long),
            sampling_policy,
            max_offset_seconds,
        )
        cached = _ordered_get(self._timeseries_cache, key)
        if cached is not None:
            return cached

        native = np.flatnonzero(
            np.isfinite(times) & (times >= float(start_mjd)) & (times <= float(end_mjd))
        )
        selected_offsets, stride = stride_indices(len(native), int(max_points))
        selected = native[selected_offsets]
        if role == "context":
            frame_getter = lambda index: source.frame(
                int(index),
                mode,
                operation,
                reference,
                diff_seconds,
                mean_start_mjd,
                mean_end_mjd,
                temporal_mode="none",
            )
        elif operation is None:
            frame_getter = lambda index: self.eovsa.mode_data(int(index), diff_seconds, mode)[int(freq_index)]
        else:
            frame_getter = lambda index: self.eovsa.operation_data(
                int(index),
                diff_seconds,
                operation,
                reference,
                mean_start_mjd,
                mean_end_mjd,
            )[int(freq_index)]
        raw_values = np.asarray(
            [patch_mean(frame_getter(int(index)), x, y, int(patch_radius)) for index in selected],
            dtype=np.float32,
        )
        selected_times = times[selected]
        if len(selected_times) > 1:
            cadence_deltas = np.diff(selected_times) * 86400.0
            positive = cadence_deltas[np.isfinite(cadence_deltas) & (cadence_deltas > 0)]
            cadence_seconds = float(np.median(positive)) if len(positive) else 0.0
        else:
            cadence_seconds = 0.0
        payload: dict[str, object] = {
            "mjd": [float(value) for value in selected_times],
            "raw": [float(value) if np.isfinite(value) else None for value in raw_values],
            "stats": {"raw": series_stats(raw_values)},
            "cadenceSeconds": cadence_seconds,
            "nTotal": int(len(native)),
            "stride": int(stride),
        }
        if normalized_temporal != "none":
            smoothed = temporal_filter_series(
                raw_values,
                selected_times,
                normalized_temporal,
                temporal_sigma_short,
                temporal_sigma_long,
            )
            payload["smoothed"] = [float(value) if np.isfinite(value) else None for value in smoothed]
            payload["stats"]["smoothed"] = series_stats(smoothed)  # type: ignore[index]
        return _ordered_put(self._timeseries_cache, key, payload, TEXTURE_CACHE_LIMIT)

    def write_loaded_outputs(self) -> None:
        if self.tracks:
            self.write_tracking_csv()
        elif self.sad_tracks:
            _write_csv(self.output_dir / "sad_tracks.csv", self.sad_tracks, SAD_FIELDS)
            _write_csv(self.output_dir / "feature_tracks.csv", self.sad_tracks, SAD_FIELDS)
            self.write_aia_track_map()
        if self.eovsa_sources:
            _write_csv(self.output_dir / "eovsa_sources.csv", self.eovsa_sources, EOVSA_FIELDS)
            _write_csv(self.output_dir / "radio_sources.csv", self.eovsa_sources, EOVSA_FIELDS)
            self.write_eovsa_source_map()

    def add_source_spec(self, spec: dict[str, object]) -> dict[str, object]:
        source_id = str(spec.get("id") or f"source-{uuid.uuid4().hex[:8]}")
        paths = self._manifest_paths(spec)
        path = str(paths.get("path") or paths.get("fitsDir") or paths.get("directory") or "")
        source = {
            "id": source_id,
            "role": str(spec.get("role") or "context"),
            "label": str(spec.get("label") or Path(path).name or source_id),
            "instrument": str(spec.get("instrument") or ""),
            "format": str(spec.get("format") or Path(path).suffix.lstrip(".") or "unknown"),
            "shape": spec.get("shape") if isinstance(spec.get("shape"), list) else [],
            "time": spec.get("time") if isinstance(spec.get("time"), dict) else {"count": 0},
            "capabilities": {"render": False},
            "status": "placeholder",
            "paths": paths,
        }
        self.source_specs = [item for item in self.source_specs if str(item.get("id")) != source_id]
        self.source_specs.append(source)
        return source

    def radio_global_peaks(
        self,
        diff_seconds: float,
        difference_mode: str,
        refresh: bool = False,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
    ) -> tuple[str, list[float]]:
        mode = _difference_mode(difference_mode)
        operation = _difference_operation(difference_operation)
        key = self.radio_peak_cache_key(diff_seconds, mode, operation, difference_reference, mean_start_mjd, mean_end_mjd)
        if not refresh and key in self.radio_peak_cache:
            return key, self.radio_peak_cache[key]
        build_lock = getattr(self, "_radio_peak_build_lock", None)
        if build_lock is None:
            build_lock = Lock()
            self._radio_peak_build_lock = build_lock
            self._radio_peak_inflight = {}
        with build_lock:
            pending = self._radio_peak_inflight.get(key)
            owner = pending is None
            if owner:
                pending = Event()
                self._radio_peak_inflight[key] = pending
        assert pending is not None
        if not owner:
            pending.wait()
            if key in self.radio_peak_cache:
                return key, self.radio_peak_cache[key]
            return self.radio_global_peaks(
                diff_seconds,
                difference_mode,
                refresh=refresh,
                difference_operation=difference_operation,
                difference_reference=difference_reference,
                mean_start_mjd=mean_start_mjd,
                mean_end_mjd=mean_end_mjd,
            )
        try:
            if not refresh and key in self.radio_peak_table_cache:
                table = np.asarray(self.radio_peak_table_cache[key], dtype=np.float32)
            else:
                registry = getattr(self, "progress_registry", None)
                if not isinstance(registry, ProgressRegistry):
                    registry = ProgressRegistry()
                    self.progress_registry = registry
                op_id = registry.start("Computing global peaks", total=len(self.eovsa.files))

                def report(done: int, total: int) -> None:
                    registry.update(op_id, done=done, total=total)

                progress_local = getattr(self.eovsa, "_peak_progress_local", None)
                if progress_local is None:
                    progress_local = local()
                    self.eovsa._peak_progress_local = progress_local
                previous_callback = getattr(progress_local, "callback", None)
                progress_local.callback = report
                try:
                    table = (
                        self.eovsa.peak_table(
                            diff_seconds,
                            mode,
                            refresh=refresh,
                        )
                        if operation is None
                        else self.eovsa.operation_peak_table(
                            diff_seconds,
                            operation,
                            difference_reference,
                            mean_start_mjd,
                            mean_end_mjd,
                            refresh=refresh,
                        )
                    )
                finally:
                    if previous_callback is None:
                        del progress_local.callback
                    else:
                        progress_local.callback = previous_callback
                    registry.finish(op_id)
                self.radio_peak_table_cache[key] = _finite_float_table(table)
            self.radio_peak_cache[key] = _finite_float_list(np.nanmax(table, axis=0))
            self._overlay_cache.clear()
        finally:
            with build_lock:
                self._radio_peak_inflight.pop(key, None)
                pending.set()
        return key, self.radio_peak_cache[key]

    def set_roi_from_pixels(
        self,
        panel: str,
        points: list[list[float]],
        time_index: int,
        freq_index: int,
        x_offset: float,
        y_offset: float,
        diff_seconds: float,
        eovsa_index: int | None = None,
    ) -> dict[str, object]:
        """Convert panel pixels to world coordinates and store the ROI.

        ``eovsa_index`` preserves a source-native radio resolution selected by
        the API.  It is optional so legacy AIA-index callers retain their
        historical behavior.

        :param eovsa_index: Optional native radio sample index.
        :type eovsa_index: int or None
        :returns: Stored world-coordinate ROI payload.
        :rtype: dict[str, object]
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            raise ValueError("ROI needs at least three x/y points.")
        if panel == "eovsa":
            if eovsa_index is None:
                data, _ = self.eovsa.diff_frame_for_aia(self.aia.times[int(time_index)].mjd, freq_index, diff_seconds)
            else:
                data = self.eovsa.diff_frame(int(eovsa_index), freq_index, diff_seconds)
            world = self.eovsa.pixel_to_world(data, pts, x_offset, y_offset)
        else:
            world = self.aia.pixel_to_world(time_index, pts)
        self.roi_world = world.tolist()
        return {"roiWorld": self.roi_world}

    def roi_pixels_for_panel(
        self,
        panel: str,
        time_index: int,
        freq_index: int,
        x_offset: float,
        y_offset: float,
        diff_seconds: float,
        eovsa_index: int | None = None,
    ) -> list[list[float]]:
        """Project the stored world ROI into a panel's native pixel frame.

        :param eovsa_index: Optional native radio sample index.
        :type eovsa_index: int or None
        :rtype: list[list[float]]
        """
        if not self.roi_world:
            return []
        world = np.asarray(self.roi_world, dtype=float)
        if panel == "eovsa":
            if eovsa_index is None:
                data, _ = self.eovsa.diff_frame_for_aia(self.aia.times[int(time_index)].mjd, freq_index, diff_seconds)
            else:
                data = self.eovsa.diff_frame(int(eovsa_index), freq_index, diff_seconds)
            pix = self.eovsa.world_to_pixel(data, world, x_offset, y_offset)
        else:
            pix = self.aia.world_to_pixel(time_index, world)
        return pix.tolist()

    def _roi_path(self) -> MplPath | None:
        if not self.roi_world:
            return None
        return MplPath(np.asarray(self.roi_world, dtype=float))

    def _inside_roi(self, points_arcsec: np.ndarray) -> np.ndarray:
        path = self._roi_path()
        if path is None:
            return np.ones(points_arcsec.shape[0], dtype=bool)
        return path.contains_points(points_arcsec)

    def _eovsa_to_aia_affine(self, time_index: int, x_offset: float, y_offset: float) -> np.ndarray:
        idx = int(np.clip(time_index, 0, self.aia.nt - 1))
        cache_key = ("aia", idx, float(x_offset), float(y_offset))
        cached = _ordered_get(self._affine_cache, cache_key)
        if cached is not None:
            return cached
        ny, nx = self.eovsa.shape
        src = np.array(
            [
                [0.0, 0.0],
                [nx - 1.0, 0.0],
                [0.0, ny - 1.0],
                [nx - 1.0, ny - 1.0],
                [(nx - 1.0) / 2.0, (ny - 1.0) / 2.0],
            ],
            dtype=float,
        )
        data = np.zeros((ny, nx), dtype=np.float32)
        world = self.eovsa.pixel_to_world(data, src, x_offset, y_offset)
        dst = self.aia.world_to_pixel(idx, world)
        design = np.column_stack([src, np.ones(src.shape[0])])
        return _ordered_put(self._affine_cache, cache_key, np.linalg.lstsq(design, dst, rcond=None)[0], TEXTURE_CACHE_LIMIT)

    def _eovsa_to_eovsa_affine(self, freq_index: int, x_offset: float, y_offset: float) -> np.ndarray:
        """Map one radio band's native pixels into the shifted radio grid.

        :param freq_index: Zero-based radio channel index.
        :type freq_index: int
        :param x_offset: X shift to apply in source-world arcsec.
        :type x_offset: float
        :param y_offset: Y shift to apply in source-world arcsec.
        :type y_offset: float
        :returns: Affine matrix mapping ``[x, y, 1]`` to target pixels.
        :rtype: numpy.ndarray
        """
        if not hasattr(self, "_affine_cache"):
            self._affine_cache = OrderedDict()
        fidx = int(np.clip(freq_index, 0, self.eovsa.nfreq - 1))
        cache_key = ("eovsa", fidx, float(x_offset), float(y_offset))
        cached = _ordered_get(self._affine_cache, cache_key)
        if cached is not None:
            return cached
        ny, nx = self.eovsa.shape
        src = np.array(
            [
                [0.0, 0.0], [nx - 1.0, 0.0], [0.0, ny - 1.0],
                [nx - 1.0, ny - 1.0], [(nx - 1.0) / 2.0, (ny - 1.0) / 2.0],
            ],
            dtype=float,
        )
        data = np.zeros((ny, nx), dtype=np.float32)
        world = self.eovsa.pixel_to_world(data, src, x_offset, y_offset)
        dst = self.eovsa.world_to_pixel(data, world, 0.0, 0.0)
        design = np.column_stack([src, np.ones(src.shape[0])])
        return _ordered_put(self._affine_cache, cache_key, np.linalg.lstsq(design, dst, rcond=None)[0], TEXTURE_CACHE_LIMIT)

    def transparent_overlay_png(self, target_panel: str = "aia") -> bytes:
        target = str(target_panel).lower()
        shape = self.aia.shape if target == "aia" else self.eovsa.shape
        ny, nx = shape
        out = BytesIO()
        Image.new("RGBA", (nx, ny), (0, 0, 0, 0)).save(out, format="PNG")
        return out.getvalue()

    def eovsa_channels_in_polygon(
        self,
        time_index: int,
        diff_seconds: float,
        use_running_diff: bool,
        x_offset: float,
        y_offset: float,
        level_percent: float,
        polygon: list[list[float]],
        difference_mode: str | None = None,
        level_reference: str = "current",
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        level_mode: str = "percent",
        level_kelvin: float = 1_000_000.0,
        level_sfu: float = 1.0,
        eovsa_index: int | None = None,
        target_panel: str = "aia",
    ) -> list[int]:
        """Return radio bands whose shifted contour centroid enters a polygon.

        :param time_index: Target AIA or radio-native frame index.
        :type time_index: int
        :param diff_seconds: Difference cadence in seconds.
        :type diff_seconds: float
        :param use_running_diff: Legacy running-difference switch.
        :type use_running_diff: bool
        :param x_offset: Global x offset in arcsec.
        :type x_offset: float
        :param y_offset: Global y offset in arcsec.
        :type y_offset: float
        :param level_percent: Percent contour threshold for relative levels.
        :type level_percent: float
        :param polygon: Target-panel pixel polygon in x/y coordinates.
        :type polygon: list[list[float]]
        :param eovsa_index: Optional source-native radio frame index.
        :type eovsa_index: int or None
        :param target_panel: Native target grid, either ``"aia"`` or ``"eovsa"``.
        :type target_panel: str
        :returns: Sorted zero-based radio channel indices.
        :rtype: list[int]
        """
        target = str(target_panel).lower()
        if target not in {"aia", "eovsa"}:
            raise ValueError(f"Unknown contour target panel: {target_panel}")
        polygon_array = np.asarray(polygon, dtype=float)
        if polygon_array.ndim != 2 or polygon_array.shape[1] != 2 or polygon_array.shape[0] < 3 or not np.isfinite(polygon_array).all():
            raise ValueError("polygon must contain at least three finite x/y vertices")
        polygon_path = MplPath(polygon_array)
        if target == "aia":
            target_index = int(np.clip(time_index, 0, self.aia.nt - 1))
            eidx = (
                int(np.clip(eovsa_index, 0, len(self.eovsa.files) - 1))
                if eovsa_index is not None
                else self.eovsa.nearest_time_index(self.aia.times[target_index].mjd)
            )
            target_shape = self.aia.shape
        else:
            eidx = int(np.clip(
                time_index if eovsa_index is None else eovsa_index,
                0,
                len(self.eovsa.files) - 1,
            ))
            target_index = eidx
            target_shape = self.eovsa.shape
        mode = _difference_mode(difference_mode, use_running_diff)
        operation = _difference_operation(difference_operation)
        reference = "global" if str(level_reference).lower() == "global" else "current"
        requested_level_mode = str(level_mode).strip().lower()
        absolute_level_mode = requested_level_mode if reference == "global" and requested_level_mode in {"kelvin", "sfu"} else "percent"
        absolute_level = absolute_level_mode in {"kelvin", "sfu"}
        sfu_thresholds = (
            _sfu_to_tb_thresholds(level_sfu, self.eovsa.freqs_hz, self.eovsa.header)
            if absolute_level_mode == "sfu"
            else None
        )
        if reference == "global" and not absolute_level:
            peak_cache_key = self.radio_peak_cache_key(
                diff_seconds, mode, operation, difference_reference,
                mean_start_mjd, mean_end_mjd,
            )
            if peak_cache_key in self.radio_peak_cache:
                global_peaks = self.radio_peak_cache[peak_cache_key]
            else:
                _, global_peaks = self.radio_global_peaks(
                    diff_seconds,
                    mode,
                    refresh=True,
                    difference_operation=operation,
                    difference_reference=difference_reference,
                    mean_start_mjd=mean_start_mjd,
                    mean_end_mjd=mean_end_mjd,
                )
            if len(global_peaks) != self.eovsa.nfreq or not any(
                np.isfinite(value) and abs(value) > 0.0 for value in global_peaks
            ):
                raise OverlayUnavailableError("global peak table contains no finite signal")
        else:
            global_peaks = []
        level = float(np.clip(level_percent, 1.0, 99.0))
        selected: set[int] = set()
        for fidx in range(self.eovsa.nfreq):
            data = all_band_data[fidx]
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                continue
            peak = float(global_peaks[fidx]) if reference == "global" and fidx < len(global_peaks) else float(np.nanmax(finite))
            if absolute_level_mode == "kelvin":
                threshold = float(level_kelvin)
            elif absolute_level_mode == "sfu":
                if sfu_thresholds is None or fidx >= len(sfu_thresholds):
                    continue
                threshold = float(sfu_thresholds[fidx])
            else:
                threshold = peak * level / 100.0
            if not np.isfinite(threshold) or (not absolute_level and peak <= 0):
                continue
            floor = float(np.nanmin(finite)) - abs(peak) - 1.0
            cleaned = np.nan_to_num(data, nan=floor, posinf=peak, neginf=floor)
            if threshold <= float(np.nanmin(cleaned)) or threshold >= float(np.nanmax(cleaned)):
                continue
            dx, dy = self.channel_offset(fidx)
            if target == "aia":
                transform = self._eovsa_to_aia_affine(target_index, x_offset + dx, y_offset + dy)
            elif dx == 0.0 and dy == 0.0:
                transform = None
            else:
                transform = self._eovsa_to_eovsa_affine(fidx, dx, dy)
            target_height = target_shape[0]
            for contour in measure.find_contours(cleaned, threshold):
                if contour.shape[0] < 3:
                    continue
                eovsa_pix = np.column_stack([contour[:, 1], contour[:, 0]])
                target_pix = (
                    np.column_stack([eovsa_pix, np.ones(eovsa_pix.shape[0])]) @ transform
                    if transform is not None
                    else eovsa_pix
                )
                target_pix[:, 1] = target_height - 1 - target_pix[:, 1]
                finite_points = target_pix[np.isfinite(target_pix).all(axis=1)]
                if finite_points.shape[0] >= 3 and polygon_path.contains_point(np.mean(finite_points, axis=0)):
                    selected.add(fidx)
                    break
        return sorted(selected)

    def eovsa_contour_geometry(
        self,
        time_index: int,
        diff_seconds: float,
        use_running_diff: bool,
        level_percent: float,
        difference_mode: str | None = None,
        level_reference: str = "current",
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        level_mode: str = "percent",
        level_kelvin: float = 1_000_000.0,
        level_sfu: float = 1.0,
        eovsa_index: int | None = None,
        target_panel: str = "aia",
        cache_durable: bool = True,
    ) -> dict[str, object]:
        """Return unshifted per-band radio contour geometry.

        Coordinates use the target source's native, unflipped pixel frame.
        Global and per-channel alignment offsets are intentionally omitted so
        the client can apply them while drawing.

        :param time_index: Target AIA or radio-native frame index.
        :type time_index: int
        :param diff_seconds: Previous-frame cadence in seconds.
        :type diff_seconds: float
        :param use_running_diff: Legacy running-difference switch.
        :type use_running_diff: bool
        :param level_percent: Relative contour level in percent.
        :type level_percent: float
        :param difference_mode: Legacy difference mode.
        :type difference_mode: str or None
        :param level_reference: ``"current"`` or ``"global"`` peak reference.
        :type level_reference: str
        :param difference_operation: Difference operation.
        :type difference_operation: str or None
        :param difference_reference: Difference reference selection.
        :type difference_reference: str
        :param mean_start_mjd: Optional mean-reference start MJD.
        :type mean_start_mjd: float or None
        :param mean_end_mjd: Optional mean-reference end MJD.
        :type mean_end_mjd: float or None
        :param level_mode: ``"percent"``, ``"kelvin"``, or ``"sfu"``.
        :type level_mode: str
        :param level_kelvin: Absolute brightness-temperature threshold.
        :type level_kelvin: float
        :param level_sfu: Absolute flux-density threshold per pixel.
        :type level_sfu: float
        :param eovsa_index: Optional resolved radio-native frame index.
        :type eovsa_index: int or None
        :param target_panel: Native target grid, ``"aia"`` or ``"eovsa"``.
        :type target_panel: str
        :param cache_durable: Flush the cache file before atomic replacement.
        :type cache_durable: bool
        :returns: Mapping containing unshifted per-band contour polylines.
        :rtype: dict[str, object]
        :raises ValueError: If the requested target panel is unknown.
        :raises OverlayUnavailableError: If a required global peak table is unavailable.
        """
        if not cache_durable:
            self.eovsa._warm_cache_active = True
        target = str(target_panel).lower()
        if target not in {"aia", "eovsa"}:
            raise ValueError(f"Unknown contour target panel: {target_panel}")
        if target == "aia":
            target_index = int(np.clip(time_index, 0, self.aia.nt - 1))
            eidx = (
                int(np.clip(eovsa_index, 0, len(self.eovsa.files) - 1))
                if eovsa_index is not None
                else self.eovsa.nearest_time_index(self.aia.times[target_index].mjd)
            )
        else:
            eidx = int(np.clip(
                time_index if eovsa_index is None else eovsa_index,
                0,
                len(self.eovsa.files) - 1,
            ))
            target_index = eidx

        mode = _difference_mode(difference_mode, use_running_diff)
        operation = _difference_operation(difference_operation)
        reference = "global" if str(level_reference).lower() == "global" else "current"
        requested_level_mode = str(level_mode).strip().lower()
        absolute_level_mode = requested_level_mode if reference == "global" and requested_level_mode in {"kelvin", "sfu"} else "percent"
        absolute_level = absolute_level_mode in {"kelvin", "sfu"}
        sfu_thresholds = (
            _sfu_to_tb_thresholds(level_sfu, self.eovsa.freqs_hz, self.eovsa.header)
            if absolute_level_mode == "sfu"
            else None
        )
        if reference == "global" and not absolute_level:
            peak_cache_key = self.radio_peak_cache_key(
                diff_seconds,
                mode,
                operation,
                difference_reference,
                mean_start_mjd,
                mean_end_mjd,
            )
            if peak_cache_key in self.radio_peak_cache:
                global_peaks = self.radio_peak_cache[peak_cache_key]
            else:
                try:
                    _, global_peaks = self.radio_global_peaks(
                        diff_seconds,
                        mode,
                        refresh=True,
                        difference_operation=operation,
                        difference_reference=difference_reference,
                        mean_start_mjd=mean_start_mjd,
                        mean_end_mjd=mean_end_mjd,
                    )
                except Exception as exc:
                    raise OverlayUnavailableError(f"global peak table unavailable: {exc}") from exc
            if len(global_peaks) != self.eovsa.nfreq or not any(
                np.isfinite(value) and abs(value) > 0.0 for value in global_peaks
            ):
                raise OverlayUnavailableError("global peak table contains no finite signal")
        else:
            global_peaks = []

        level = float(np.clip(level_percent, 1.0, 99.0))
        kelvin_level = float(level_kelvin)
        sfu_level = float(level_sfu)
        contour_level = kelvin_level if absolute_level_mode == "kelvin" else sfu_level if absolute_level_mode == "sfu" else level
        peak_key = tuple(round(value, 6) for value in global_peaks) if reference == "global" and not absolute_level else ()
        channel_mask = tuple(_channel_mask_for_session(self))
        active_band_indices = [
            fidx for fidx in range(self.eovsa.nfreq)
            if fidx >= len(channel_mask) or not channel_mask[fidx]
        ]
        key = _cache_key(
            target,
            target_index,
            eidx,
            diff_seconds,
            mode,
            operation,
            difference_reference,
            mean_start_mjd,
            mean_end_mjd,
            reference,
            absolute_level_mode,
            peak_key,
            contour_level,
            channel_mask,
        )
        if not hasattr(self, "_contour_geometry_cache"):
            self._contour_geometry_cache = OrderedDict()
        cached = _cache_get(self._contour_geometry_cache, key)
        if cached is not None:
            return json.loads(cached)
        disk_key = _render_disk_key(
            [self.aia, self.eovsa],
            "radio-contour-geometry",
            key,
        )
        cached = RENDER_DISK_CACHE.get(disk_key, "json")
        if cached is not None:
            _cache_put(self._contour_geometry_cache, key, cached)
            return json.loads(cached)

        # Contours consume one plane per band.  Decode each required cube once
        # and retain its planes in the bounded LRU; arithmetic stays in the
        # existing per-band helpers so the contour values remain bit-identical.
        if isinstance(self.eovsa, EovsaSequence):
            self.eovsa._ensure_band_planes(eidx, active_band_indices)
            if operation is None:
                if mode == "base":
                    self.eovsa._ensure_band_planes(0, active_band_indices)
                elif mode != "none":
                    self.eovsa._ensure_band_planes(self.eovsa.previous_index(eidx, diff_seconds), active_band_indices)
                all_band_data = [
                    self.eovsa._band_mode_data(eidx, fidx, diff_seconds, mode)
                    for fidx in range(self.eovsa.nfreq)
                ]
            else:
                ref = _difference_reference(difference_reference)
                if ref == "base":
                    self.eovsa._ensure_band_planes(0, active_band_indices)
                elif ref == "previous":
                    self.eovsa._ensure_band_planes(self.eovsa.previous_index(eidx, diff_seconds), active_band_indices)
                all_band_data = [
                    self.eovsa._band_operation_data(
                        eidx,
                        fidx,
                        diff_seconds,
                        operation,
                        difference_reference,
                        mean_start_mjd,
                        mean_end_mjd,
                    )
                    for fidx in range(self.eovsa.nfreq)
                ]
        else:
            # Small test doubles and compatibility sources expose the legacy
            # full-cube methods only; retain that contract for them.
            all_band_data = (
                self.eovsa.mode_data(eidx, diff_seconds, mode)
                if operation is None
                else self.eovsa.operation_data(
                    eidx,
                    diff_seconds,
                    operation,
                    difference_reference,
                    mean_start_mjd,
                    mean_end_mjd,
                    remember_peak=False,
                )
            )

        # The affine cache is the regrid reuse layer: target-grid geometry is
        # keyed by native time and offsets, so changing only a contour level
        # reuses this transform and the decoded planes above.
        target_transform = self._eovsa_to_aia_affine(target_index, 0.0, 0.0) if target == "aia" else None
        frequencies = np.asarray(self.eovsa.freqs_hz, dtype=float).reshape(-1)
        bands: list[dict[str, object]] = []
        for fidx in range(self.eovsa.nfreq):
            if fidx < len(channel_mask) and channel_mask[fidx]:
                continue
            data = all_band_data[fidx]
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                continue
            peak = float(global_peaks[fidx]) if reference == "global" and fidx < len(global_peaks) else float(np.nanmax(finite))
            if absolute_level_mode == "kelvin":
                threshold = kelvin_level
            elif absolute_level_mode == "sfu":
                if sfu_thresholds is None or fidx >= len(sfu_thresholds):
                    continue
                threshold = float(sfu_thresholds[fidx])
            else:
                threshold = peak * level / 100.0
            cleaned = _contour_plane(data, peak, threshold, absolute_level)
            if cleaned is None:
                continue
            polylines = _contour_polylines(cleaned, threshold, target_transform)
            if not polylines:
                continue
            bands.append({
                "channel": fidx,
                "freqGhz": float(frequencies[fidx] / 1e9),
                "level": float(threshold),
                "polylines": polylines,
            })
        payload: dict[str, object] = {"bands": bands}
        content = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        RENDER_DISK_CACHE.put(disk_key, content, "json", durable=cache_durable)
        _cache_put(self._contour_geometry_cache, key, content)
        return payload

    def eovsa_all_band_contours_on_aia(
        self,
        time_index: int,
        diff_seconds: float,
        use_running_diff: bool,
        x_offset: float,
        y_offset: float,
        level_percent: float,
        filled: bool,
        opacity: float,
        difference_mode: str | None = None,
        level_reference: str = "current",
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        level_mode: str = "percent",
        level_kelvin: float = 1_000_000.0,
        level_sfu: float = 1.0,
        contour_cmap: str = "turbo",
        eovsa_index: int | None = None,
        target_panel: str = "aia",
        highlight_channels: list[int] | None = None,
        channels: list[int] | None = None,
    ) -> bytes:
        """Render all-band radio contours in a target-native pixel frame.

        :param contour_cmap: Matplotlib colormap used to encode radio frequency.
        :type contour_cmap: str
        :param level_sfu: Absolute flux-density contour level in sfu per pixel.
        :type level_sfu: float
        :param eovsa_index: Optional source-native radio index already resolved
            by the API.  When omitted, the legacy AIA-time lookup is used.
        :type eovsa_index: int or None
        :param target_panel: Native target grid, either ``"aia"`` or
            ``"eovsa"``. The default preserves the legacy AIA overlay.
        :type target_panel: str
        :param highlight_channels: Optional selected channel indices. Selected
        contours are rendered at full opacity; all other bands are dimmed.
        :type highlight_channels: list[int] or None
        :param channels: Optional channel subset to render; omitted renders all
            bands.
        :type channels: list[int] or None
        For ``level_reference="global"`` and ``level_mode="sfu"``, the
        per-band threshold uses the Rayleigh-Jeans relation
        ``T_thresh(nu_i) = S_sfu * 1e-19 * c^2 / (2 * k_B * nu_i^2 * Omega_pix)``
        with ``Omega_pix = (CDELT_arcsec * pi / 180 / 3600)^2``.
        :returns: RGBA PNG bytes containing the contour overlay.
        :rtype: bytes
        """
        target = str(target_panel).lower()
        if target not in {"aia", "eovsa"}:
            raise ValueError(f"Unknown contour target panel: {target_panel}")
        if target == "aia":
            target_index = int(np.clip(time_index, 0, self.aia.nt - 1))
            eidx = (
                int(np.clip(eovsa_index, 0, len(self.eovsa.files) - 1))
                if eovsa_index is not None
                else self.eovsa.nearest_time_index(self.aia.times[target_index].mjd)
            )
        else:
            eidx = int(np.clip(
                time_index if eovsa_index is None else eovsa_index,
                0,
                len(self.eovsa.files) - 1,
            ))
            target_index = eidx
        mode = _difference_mode(difference_mode, use_running_diff)
        operation = _difference_operation(difference_operation)
        reference = "global" if str(level_reference).lower() == "global" else "current"
        requested_level_mode = str(level_mode).strip().lower()
        absolute_level_mode = requested_level_mode if reference == "global" and requested_level_mode in {"kelvin", "sfu"} else "percent"
        absolute_level = absolute_level_mode in {"kelvin", "sfu"}
        sfu_thresholds = (
            _sfu_to_tb_thresholds(level_sfu, self.eovsa.freqs_hz, self.eovsa.header)
            if absolute_level_mode == "sfu"
            else None
        )
        all_band_data = self.eovsa.mode_data(eidx, diff_seconds, mode) if operation is None else self.eovsa.operation_data(eidx, diff_seconds, operation, difference_reference, mean_start_mjd, mean_end_mjd)
        if reference == "global" and not absolute_level:
            peak_cache_key = self.radio_peak_cache_key(
                diff_seconds, mode, operation, difference_reference,
                mean_start_mjd, mean_end_mjd,
            )
            if peak_cache_key in self.radio_peak_cache:
                global_peaks = self.radio_peak_cache[peak_cache_key]
            else:
                try:
                    # A loaded-row summary is not a global peak table: on a
                    # fresh session it contains no rows and yields zero
                    # thresholds. Reuse the explicit refresh path so the
                    # exact difference recipe is scanned and cached once.
                    _, global_peaks = self.radio_global_peaks(
                        diff_seconds,
                        mode,
                        refresh=True,
                        difference_operation=operation,
                        difference_reference=difference_reference,
                        mean_start_mjd=mean_start_mjd,
                        mean_end_mjd=mean_end_mjd,
                    )
                except Exception as exc:
                    raise OverlayUnavailableError(f"global peak table unavailable: {exc}") from exc
            if len(global_peaks) != self.eovsa.nfreq or not any(
                np.isfinite(value) and abs(value) > 0.0 for value in global_peaks
            ):
                raise OverlayUnavailableError("global peak table contains no finite signal")
        else:
            global_peaks = []
        peak_key = tuple(round(value, 6) for value in global_peaks) if reference == "global" and not absolute_level else ()
        level = float(np.clip(level_percent, 1.0, 99.0))
        kelvin_level = float(level_kelvin)
        sfu_level = float(level_sfu)
        alpha = int(np.clip(opacity, 0.05, 1.0) * 255)
        cmap_name, contour_colormap = _resolve_colormap(contour_cmap, "turbo")
        contour_level = kelvin_level if absolute_level_mode == "kelvin" else sfu_level if absolute_level_mode == "sfu" else level
        normalized_highlights = None if highlight_channels is None else tuple(sorted({
            int(index) for index in highlight_channels if 0 <= int(index) < self.eovsa.nfreq
        }))
        channel_mask = _channel_mask_for_session(self)
        normalized_channels = (
            None
            if channels is None and not any(channel_mask)
            else tuple(sorted({
                int(index)
                for index in (range(self.eovsa.nfreq) if channels is None else channels)
                if 0 <= int(index) < self.eovsa.nfreq and not channel_mask[int(index)]
            }))
        )
        active_band_indices = list(normalized_channels) if normalized_channels is not None else [
            fidx for fidx in range(self.eovsa.nfreq)
            if fidx >= len(channel_mask) or not channel_mask[fidx]
        ]
        key = _cache_key(target, target_index, eidx, diff_seconds, mode, operation, difference_reference, mean_start_mjd, mean_end_mjd, reference, absolute_level_mode, peak_key, x_offset, y_offset, contour_level, filled, opacity, cmap_name, self.channel_offsets_version, normalized_highlights, normalized_channels)
        cached = _cache_get(self._overlay_cache, key)
        if cached is not None:
            return cached
        disk_params = (*key[:-3], *key[-2:])
        disk_key = _render_disk_key(
            [self.aia, self.eovsa],
            "radio-contours",
            disk_params,
            {
                **_normalize_channel_offsets(getattr(self, "channel_offsets", None), self.eovsa.nfreq),
                "masked": _channel_mask_for_session(self),
            },
        )
        cached = RENDER_DISK_CACHE.get(disk_key)
        if cached is not None:
            return _cache_put(self._overlay_cache, key, cached)

        if isinstance(self.eovsa, EovsaSequence):
            self.eovsa._ensure_band_planes(eidx, active_band_indices)
            if operation is None:
                if mode == "base":
                    self.eovsa._ensure_band_planes(0, active_band_indices)
                elif mode != "none":
                    self.eovsa._ensure_band_planes(self.eovsa.previous_index(eidx, diff_seconds), active_band_indices)
                all_band_data = [
                    self.eovsa._band_mode_data(eidx, fidx, diff_seconds, mode)
                    for fidx in range(self.eovsa.nfreq)
                ]
            else:
                ref = _difference_reference(difference_reference)
                if ref == "base":
                    self.eovsa._ensure_band_planes(0, active_band_indices)
                elif ref == "previous":
                    self.eovsa._ensure_band_planes(self.eovsa.previous_index(eidx, diff_seconds), active_band_indices)
                all_band_data = [
                    self.eovsa._band_operation_data(
                        eidx,
                        fidx,
                        diff_seconds,
                        operation,
                        difference_reference,
                        mean_start_mjd,
                        mean_end_mjd,
                    )
                    for fidx in range(self.eovsa.nfreq)
                ]
        else:
            all_band_data = (
                self.eovsa.mode_data(eidx, diff_seconds, mode)
                if operation is None
                else self.eovsa.operation_data(
                    eidx,
                    diff_seconds,
                    operation,
                    difference_reference,
                    mean_start_mjd,
                    mean_end_mjd,
                )
            )

        ny, nx = self.aia.shape if target == "aia" else self.eovsa.shape
        overlay = Image.new("RGBA", (nx, ny), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        color_samples = contour_colormap(np.linspace(0.0, 1.0, self.eovsa.nfreq))
        colors = np.clip(np.rint(color_samples * 255.0), 0, 255).astype(np.uint8)
        band_indices = range(self.eovsa.nfreq) if normalized_channels is None else normalized_channels
        for fidx in band_indices:
            data = all_band_data[fidx]
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                continue
            peak = float(global_peaks[fidx]) if reference == "global" and fidx < len(global_peaks) else float(np.nanmax(finite))
            if absolute_level_mode == "kelvin":
                threshold = kelvin_level
            elif absolute_level_mode == "sfu":
                if sfu_thresholds is None or fidx >= len(sfu_thresholds):
                    continue
                threshold = float(sfu_thresholds[fidx])
            else:
                threshold = peak * level / 100.0
            cleaned = _contour_plane(data, peak, threshold, absolute_level)
            if cleaned is None:
                continue
            color = tuple(int(v) for v in colors[fidx][:3])
            dx, dy = self.channel_offset(fidx)
            eovsa_to_target = (
                self._eovsa_to_aia_affine(target_index, x_offset + dx, y_offset + dy)
                if target == "aia"
                else None if dx == 0.0 and dy == 0.0 else self._eovsa_to_eovsa_affine(fidx, dx, dy)
            )
            highlighted = normalized_highlights is None or fidx in normalized_highlights
            band_alpha = alpha if highlighted else max(1, int(alpha * 0.35))
            for polyline in _contour_polylines(cleaned, threshold, eovsa_to_target, ny):
                points = [(point[0], point[1]) for point in polyline]
                if filled and np.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 3.0:
                    draw.polygon(points, fill=(*color, band_alpha))
                    draw.line(points + [points[0]], fill=(*color, min(255, band_alpha + 60)), width=2 if highlighted else 1)
                else:
                    draw.line(points, fill=(*color, band_alpha), width=2 if highlighted else 1, joint="curve")
        out = BytesIO()
        overlay.save(out, format="PNG")
        content = out.getvalue()
        RENDER_DISK_CACHE.put(disk_key, content)
        return _cache_put(self._overlay_cache, key, content)

    def _tracking_source(self, source_id: object = None) -> tuple[str, object]:
        """Resolve a track source alias to its canonical renderable source.

        :param source_id: Saved or request source identifier.
        :type source_id: object
        :returns: Canonical source id and native image sequence.
        :rtype: tuple[str, object]
        :raises ValueError: If the source is not a renderable tracking source.
        """
        requested = str(source_id or self.context_source_id)
        if requested in {self.context_source_id, "context", "aia"}:
            return self.context_source_id, self.aia
        if requested in {self.radio_source_id, "radio", "eovsa"}:
            return self.radio_source_id, self.eovsa
        raise ValueError(f"Tracking source is not renderable: {requested}")

    def _tracking_source_geometry(self, source_id: object) -> tuple[str, np.ndarray, tuple[int, int]]:
        """Return the canonical id, MJD axis, and image shape for tracking.

        :param source_id: Saved or request source identifier.
        :type source_id: object
        :returns: Canonical id, one-dimensional MJD axis, and ``(ny, nx)``.
        :rtype: tuple[str, numpy.ndarray, tuple[int, int]]
        """
        canonical, source = self._tracking_source(source_id)
        times = np.asarray(source.times.mjd, dtype=float)
        return canonical, times, tuple(int(value) for value in source.shape)

    def _tracking_pixel_to_world(
        self,
        source_id: object,
        frame_index: int,
        points: np.ndarray,
    ) -> np.ndarray:
        """Convert tracked-source pixels to solar arcseconds.

        :param source_id: Canonical or aliased tracking source id.
        :type source_id: object
        :param frame_index: Native source frame index.
        :type frame_index: int
        :param points: ``N x 2`` tracked-source pixel coordinates.
        :type points: numpy.ndarray
        :returns: ``N x 2`` solar coordinates in arcseconds.
        :rtype: numpy.ndarray
        """
        canonical, _ = self._tracking_source(source_id)
        if canonical == self.context_source_id:
            return np.asarray(self.aia.pixel_to_world(frame_index, points), dtype=float)
        data = np.zeros(self.eovsa.shape, dtype=np.float32)
        return np.asarray(self.eovsa.pixel_to_world(data, points, 0.0, 0.0), dtype=float)

    def _tracking_world_to_pixel(
        self,
        source_id: object,
        frame_index: int,
        points: np.ndarray,
    ) -> np.ndarray:
        """Project solar arcseconds into one tracked source's pixels.

        :param source_id: Canonical or aliased tracking source id.
        :type source_id: object
        :param frame_index: Native source frame index.
        :type frame_index: int
        :param points: ``N x 2`` solar coordinates in arcseconds.
        :type points: numpy.ndarray
        :returns: ``N x 2`` tracked-source pixel coordinates.
        :rtype: numpy.ndarray
        """
        canonical, _ = self._tracking_source(source_id)
        if canonical == self.context_source_id:
            return np.asarray(self.aia.world_to_pixel(frame_index, points), dtype=float)
        data = np.zeros(self.eovsa.shape, dtype=np.float32)
        return np.asarray(self.eovsa.world_to_pixel(data, points, 0.0, 0.0), dtype=float)

    def _normalize_tracking_track(self, value: dict[str, object]) -> dict[str, object]:
        """Validate and canonicalize one client or solver track.

        :param value: Client, restored, or solver-produced track payload.
        :type value: dict[str, object]
        :returns: Canonical nested track with authoritative anchors.
        :rtype: dict[str, object]
        """
        track_id = str(value.get("id") or f"track-{uuid.uuid4().hex[:8]}")
        source_id, times, _ = self._tracking_source_geometry(value.get("sourceId"))
        last_frame = max(0, len(times) - 1)
        anchors: list[dict[str, object]] = []
        for raw in value.get("anchors", []):
            if not isinstance(raw, dict):
                continue
            frame_index = int(np.clip(int(raw.get("frameIndex", 0)), 0, last_frame))
            x, y = float(raw.get("x", float("nan"))), float(raw.get("y", float("nan")))
            if not np.isfinite([x, y]).all():
                continue
            anchors.append({
                "frameIndex": frame_index,
                "mjd": float(times[frame_index]),
                "x": x,
                "y": y,
            })
        anchors = list({int(anchor["frameIndex"]): anchor for anchor in anchors}.values())
        anchors.sort(key=lambda anchor: int(anchor["frameIndex"]))
        anchor_by_frame = {int(anchor["frameIndex"]): anchor for anchor in anchors}
        points: list[dict[str, object]] = []
        for raw in value.get("points", []):
            if not isinstance(raw, dict):
                continue
            frame_index = int(np.clip(int(raw.get("frameIndex", 0)), 0, last_frame))
            anchor = anchor_by_frame.get(frame_index)
            x = float(anchor["x"] if anchor else raw.get("x", float("nan")))
            y = float(anchor["y"] if anchor else raw.get("y", float("nan")))
            if not np.isfinite([x, y]).all():
                continue
            points.append(_tracking_point(
                frame_index,
                x,
                y,
                1.0 if anchor else float(raw.get("confidence", 1.0)),
                anchor is not None,
                float(times[frame_index]),
            ))
        point_by_frame = {int(point["frameIndex"]): point for point in points}
        for frame_index, anchor in anchor_by_frame.items():
            point_by_frame[frame_index] = _tracking_point(
                frame_index, float(anchor["x"]), float(anchor["y"]), 1.0, True,
                float(times[frame_index]),
            )
        points = sorted(point_by_frame.values(), key=lambda point: int(point["frameIndex"]))
        state = str(value.get("state", "active"))
        if state not in {"active", "stopped-low-confidence", "stopped-edge"}:
            state = "active"
        return {
            "id": track_id,
            "label": str(value.get("label") or track_id),
            "sourceId": source_id,
            "color": str(value.get("color") or "#56c7d9"),
            "visible": bool(value.get("visible", True)),
            "anchors": anchors,
            "points": points,
            "state": state,
        }

    def set_tracks(self, tracks: list[dict[str, object]]) -> list[dict[str, object]]:
        """Replace session tracks and refresh their CSV exports.

        :param tracks: Nested track payloads from the client or solver.
        :type tracks: list[dict[str, object]]
        :returns: Canonical tracks stored by the session.
        :rtype: list[dict[str, object]]
        """
        self.tracks = [self._normalize_tracking_track(track) for track in tracks]
        self.write_tracking_csv()
        return self.tracks

    def set_correlation_target(self, target: list[list[float]] | None) -> list[list[float]]:
        """Persist one loop-top boundary in solar arcseconds and refresh exports."""
        normalized: list[list[float]] = []
        for point in target or []:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                x, y = float(point[0]), float(point[1])
                if np.isfinite([x, y]).all():
                    normalized.append([x, y])
        self.correlation_target = normalized if len(normalized) >= 3 else None
        if self.tracks:
            self.write_tracking_csv()
        return self.correlation_target or []

    def write_tracking_csv(self) -> list[dict[str, object]]:
        """Write canonical tracking points and centered velocity estimates.

        :returns: Flattened rows written to both feature CSV aliases.
        :rtype: list[dict[str, object]]
        """
        rows: list[dict[str, object]] = []
        for track in self.tracks:
            source_id, times, _ = self._tracking_source_geometry(track.get("sourceId"))
            rows.extend(tracking_csv_rows(
                [track],
                times,
                lambda frame_index, points, tracked_source=source_id: self._tracking_pixel_to_world(
                    tracked_source, frame_index, points
                ),
                725.0,
                self.correlation_target,
            ))
        _write_csv(self.output_dir / "feature_tracks.csv", rows, TRACKING_FIELDS)
        _write_csv(self.output_dir / "sad_tracks.csv", rows, TRACKING_FIELDS)
        return rows

    def add_track_seed(
        self,
        source_id: str,
        frame_index: int,
        x: float,
        y: float,
        label: str | None = None,
        color: str = "#56c7d9",
    ) -> dict[str, object]:
        """Create a one-anchor track at a source-native image pixel.

        :param source_id: Renderable context or radio source identifier.
        :type source_id: str
        :param frame_index: Native tracked-source frame index.
        :type frame_index: int
        :param x: Tracked-source x pixel.
        :type x: float
        :param y: Tracked-source y pixel.
        :type y: float
        :param label: Optional display label.
        :type label: str or None
        :param color: CSS track color.
        :type color: str
        :returns: Newly stored canonical track.
        :rtype: dict[str, object]
        :raises ValueError: If the seed lies outside the image or stored ROI.
        """
        canonical_source_id, times, shape = self._tracking_source_geometry(source_id)
        index = int(np.clip(frame_index, 0, len(times) - 1))
        point = np.asarray([float(x), float(y)], dtype=float)
        ny, nx = shape
        if not np.isfinite(point).all() or point[0] < 0 or point[1] < 0 or point[0] >= nx or point[1] >= ny:
            raise ValueError("Seed must lie inside the tracked source image")
        world = self._tracking_pixel_to_world(canonical_source_id, index, point.reshape(1, 2))
        if not bool(self._inside_roi(world)[0]):
            raise ValueError("Seed must lie inside the ROI")
        number = len(self.tracks) + 1
        track_id = f"track-{uuid.uuid4().hex[:8]}"
        anchor = {"frameIndex": index, "mjd": float(times[index]), "x": float(point[0]), "y": float(point[1])}
        track = self._normalize_tracking_track({
            "id": track_id,
            "label": label or f"Track {number}",
            "sourceId": canonical_source_id,
            "color": color,
            "visible": True,
            "anchors": [anchor],
            "points": [{**anchor, "confidence": 1.0, "isAnchor": True}],
            "state": "active",
        })
        self.tracks.append(track)
        self.write_tracking_csv()
        return track

    def processed_tracking_frame(self, frame_index: int, layer: dict[str, object]) -> np.ndarray:
        """Return the displayed-processing numeric frame for a track source.

        :param frame_index: Native tracked-source frame index.
        :type frame_index: int
        :param layer: Tracked source-layer science parameters.
        :type layer: dict[str, object]
        :returns: Full-resolution processed frame array.
        :rtype: numpy.ndarray
        :raises ValueError: If the layer does not address a tracking source.
        """
        source_id, times, _ = self._tracking_source_geometry(layer.get("sourceId"))
        index = int(np.clip(frame_index, 0, len(times) - 1))
        operation = str(layer.get("differenceOperation", layer.get("operation", "ratio")))
        reference = str(layer.get("differenceReference", layer.get("reference", "previous")))
        legacy_mode = str(layer.get("differenceMode", "none" if operation == "none" else "base" if reference == "base" else "running"))
        diff_seconds = float(layer.get("diffSeconds", layer.get("cadenceSeconds", self.default_diff_seconds)))
        mean_start_mjd = float(layer["meanStartMjd"]) if layer.get("meanStartMjd") is not None else None
        mean_end_mjd = float(layer["meanEndMjd"]) if layer.get("meanEndMjd") is not None else None
        radial_gamma = float(layer.get("radialGamma", 0.0))
        temporal_mode = str(layer.get("temporalMode", "none"))
        temporal_sigma_short = float(layer.get("temporalSigmaShort", DEFAULT_TEMPORAL_SIGMA_SHORT))
        temporal_sigma_long = float(layer.get("temporalSigmaLong", DEFAULT_TEMPORAL_SIGMA_LONG))
        if source_id == self.context_source_id:
            data = self.aia.frame(
                index, legacy_mode, operation, reference, diff_seconds,
                mean_start_mjd, mean_end_mjd, radial_gamma, temporal_mode,
                temporal_sigma_short, temporal_sigma_long,
            )
        else:
            data = self.eovsa.frame_for_aia_time(
                float(times[index]),
                int(layer.get("freqIndex", 0)),
                diff_seconds,
                use_running_diff=operation != "none" and reference == "previous",
                difference_mode=legacy_mode,
                difference_operation=operation,
                difference_reference=reference,
                mean_start_mjd=mean_start_mjd,
                mean_end_mjd=mean_end_mjd,
                full_cube=bool(layer.get("fullCube", False)),
                eovsa_index=index,
                radial_gamma=radial_gamma,
                temporal_mode=temporal_mode,
                temporal_sigma_short=temporal_sigma_short,
                temporal_sigma_long=temporal_sigma_long,
            )
        return np.asarray(data, dtype=float)

    def _slit_radio_layer(self, layer: dict[str, object], freq_index: int) -> dict[str, object]:
        """Return a radio slit layer with its channel alignment applied.

        The request layer carries only the global panel alignment.  Per-channel
        calibration offsets are authoritative session state shared with contour
        rendering and are applied here before WCS sampling and cache lookup.

        :param layer: Base slit-processing and global-alignment parameters.
        :type layer: dict[str, object]
        :param freq_index: Native radio frequency-plane index.
        :type freq_index: int
        :returns: Detached layer parameters with effective alignment metadata.
        :rtype: dict[str, object]
        """
        frequency_index = int(freq_index)
        dx, dy = self.channel_offset(frequency_index)
        canonical = dict(layer)
        canonical["sourceId"] = self.radio_source_id
        canonical["freqIndex"] = frequency_index
        canonical["xOffsetArcsec"] = float(layer.get("xOffsetArcsec", 0.0)) + dx
        canonical["yOffsetArcsec"] = float(layer.get("yOffsetArcsec", 0.0)) + dy
        canonical["channelOffsetDxArcsec"] = dx
        canonical["channelOffsetDyArcsec"] = dy
        canonical["offsetsRev"] = int(self.channel_offsets_version)
        return canonical

    def radio_contour_threshold(
        self,
        freq_index: int,
        level_mode: str,
        level_reference: str,
        level_percent: float,
        level_kelvin: float,
        level_sfu: float,
        data_max: float,
        diff_seconds: float = DEFAULT_DIFF_SECONDS,
        difference_mode: str | None = None,
        use_running_diff: bool = True,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
    ) -> float | None:
        """Resolve one radio channel's contour trigger level for a slit map.

        Mirrors :meth:`eovsa_contour_geometry`'s per-band threshold resolution
        bit-for-bit for the ``kelvin``, ``sfu``, and ``global`` percent cases,
        so a slit's time-distance contour family fires at the same level the
        image-panel overlay contours use for the identical layer settings.
        A time-distance map has no single native frame to stand in for the
        ``current`` percent reference, so that case uses the map's own peak
        (``data_max``) in place of one frame's peak - the same percent-of-peak
        rule applied to the one quantity a whole extracted map actually has.

        :param freq_index: Native radio frequency-plane index.
        :type freq_index: int
        :param level_mode: ``"percent"``, ``"kelvin"``, or ``"sfu"``.
        :type level_mode: str
        :param level_reference: ``"current"`` or ``"global"`` peak reference.
        :type level_reference: str
        :param level_percent: Relative contour level in percent.
        :type level_percent: float
        :param level_kelvin: Absolute brightness-temperature threshold.
        :type level_kelvin: float
        :param level_sfu: Absolute flux-density threshold per pixel.
        :type level_sfu: float
        :param data_max: The extracted map's own finite maximum intensity.
        :type data_max: float
        :param diff_seconds: Previous-frame cadence in seconds.
        :type diff_seconds: float
        :param difference_mode: Legacy difference mode.
        :type difference_mode: str or None
        :param use_running_diff: Legacy running-difference switch.
        :type use_running_diff: bool
        :param difference_operation: Difference operation.
        :type difference_operation: str or None
        :param difference_reference: Difference reference selection.
        :type difference_reference: str
        :param mean_start_mjd: Optional mean-reference start MJD.
        :type mean_start_mjd: float or None
        :param mean_end_mjd: Optional mean-reference end MJD.
        :type mean_end_mjd: float or None
        :returns: The resolved threshold, or ``None`` when it cannot be determined.
        :rtype: float or None
        """
        reference = "global" if str(level_reference).lower() == "global" else "current"
        requested_level_mode = str(level_mode).strip().lower()
        absolute_level_mode = (
            requested_level_mode if reference == "global" and requested_level_mode in {"kelvin", "sfu"} else "percent"
        )
        level = float(np.clip(level_percent, 1.0, 99.0))
        if absolute_level_mode == "kelvin":
            return float(level_kelvin)
        if absolute_level_mode == "sfu":
            thresholds = _sfu_to_tb_thresholds(level_sfu, self.eovsa.freqs_hz, self.eovsa.header)
            if freq_index < 0 or freq_index >= len(thresholds):
                return None
            return float(thresholds[freq_index])
        if reference != "global":
            # No single native frame backs a time-distance map; its own peak
            # substitutes for the "current frame" peak the overlay would use.
            if not np.isfinite(data_max):
                return None
            return float(data_max) * level / 100.0
        mode = _difference_mode(difference_mode, use_running_diff)
        operation = _difference_operation(difference_operation)
        peak_cache_key = self.radio_peak_cache_key(
            diff_seconds, mode, operation, difference_reference, mean_start_mjd, mean_end_mjd,
        )
        if peak_cache_key in self.radio_peak_cache:
            global_peaks = self.radio_peak_cache[peak_cache_key]
        else:
            try:
                _, global_peaks = self.radio_global_peaks(
                    diff_seconds,
                    mode,
                    refresh=True,
                    difference_operation=operation,
                    difference_reference=difference_reference,
                    mean_start_mjd=mean_start_mjd,
                    mean_end_mjd=mean_end_mjd,
                )
            except Exception:
                return None
        if freq_index < 0 or freq_index >= len(global_peaks) or not np.isfinite(global_peaks[freq_index]):
            return None
        return float(global_peaks[freq_index]) * level / 100.0

    def extract_slit(
        self,
        slit_id: str,
        source_id: str,
        curve_arcsec: list[list[float]],
        width: int,
        layer: dict[str, object],
        on_progress: object | None = None,
        reset_cancel: bool = True,
    ) -> tuple[dict[str, object], bool, float]:
        """Extract or restore one processed native-cadence time-distance map.

        :param slit_id: Client-stable slit identifier used for later export.
        :type slit_id: str
        :param source_id: Renderable image source identifier.
        :type source_id: str
        :param curve_arcsec: Smoothed slit polyline in solar arcseconds.
        :type curve_arcsec: list[list[float]]
        :param width: Perpendicular averaging width in image pixels.
        :type width: int
        :param layer: Displayed-processing layer parameter snapshot.
        :type layer: dict[str, object]
        :param on_progress: Optional ``(done, total)`` progress callback.
        :type on_progress: callable or None
        :param reset_cancel: Clear a prior cancellation before this map starts.
        :type reset_cancel: bool
        :returns: Result arrays, cache-hit flag, and wall time in seconds.
        :rtype: tuple[dict[str, object], bool, float]
        """
        started = time.perf_counter()
        canonical_source_id, times, _ = self._tracking_source_geometry(source_id)
        canonical_layer = dict(layer)
        canonical_layer["sourceId"] = canonical_source_id
        if canonical_source_id == self.radio_source_id:
            canonical_layer = self._slit_radio_layer(
                canonical_layer, int(canonical_layer.get("freqIndex", 0))
            )
        _, source = self._tracking_source(canonical_source_id)
        curve = np.asarray(curve_arcsec, dtype=float).reshape(-1, 2)
        if curve.shape[0] < 2 or not np.isfinite(curve).all():
            raise ValueError("Slit curve must contain at least two finite solar-coordinate vertices")
        canonical_curve, requested_is_reversed = _canonical_slit_curve(curve)
        normalized_width = int(np.clip(int(width), 1, MAX_SLIT_WIDTH_PX))
        cache_key = RENDER_DISK_CACHE.make_key({
            "version": 4,
            "render": "time-distance-slit",
            "source": _source_render_identity(source),
            "sourceId": canonical_source_id,
            "timeMjd": hashlib.sha256(np.asarray(times, dtype=np.float64).tobytes()).hexdigest(),
            "curveArcsec": np.round(canonical_curve, 8).tolist(),
            "width": normalized_width,
            "freqIndex": int(canonical_layer.get("freqIndex", -1)),
            "offsetsRev": int(canonical_layer.get("offsetsRev", 0)),
            "layerParams": canonical_layer,
        })
        cached = RENDER_DISK_CACHE.get(cache_key, "npz")
        cache_hit = cached is not None
        if cached is not None:
            canonical_result: dict[str, object] = _decode_slit_result_npz(cached)
            if callable(on_progress):
                on_progress(len(times), len(times))
        else:
            if reset_cancel:
                self._slit_extract_cancel.clear()
            x_offset = float(canonical_layer.get("xOffsetArcsec", 0.0))
            y_offset = float(canonical_layer.get("yOffsetArcsec", 0.0))

            def world_to_pixel(frame_index: int, points: np.ndarray) -> np.ndarray:
                if canonical_source_id == self.context_source_id:
                    return np.asarray(self.aia.world_to_pixel(frame_index, points), dtype=float)
                data = np.zeros(self.eovsa.shape, dtype=np.float32)
                return np.asarray(self.eovsa.world_to_pixel(data, points, x_offset, y_offset), dtype=float)

            canonical_result = extract_time_distance_map(
                lambda frame_index: self.processed_tracking_frame(frame_index, canonical_layer),
                times,
                canonical_curve,
                world_to_pixel,
                normalized_width,
                on_progress,
                self._slit_extract_cancel.is_set,
            )
            cached = _encode_slit_result_npz(canonical_result, canonical_source_id, canonical_layer)
            RENDER_DISK_CACHE.put(cache_key, cached, "npz", durable=False)
        result = _reverse_slit_result(canonical_result) if requested_is_reversed else canonical_result
        stored = {
            **result,
            "source_id": canonical_source_id,
            "layer_params": canonical_layer,
            "width": normalized_width,
            "cache_key": cache_key,
        }
        self._slit_results[str(slit_id)] = stored
        return stored, cache_hit, time.perf_counter() - started

    def store_slit_result_bundle(
        self,
        slit_id: str,
        results: list[dict[str, object]],
        freq_indices: list[int],
        freq_ghz: list[float],
    ) -> dict[str, object]:
        """Store ordered per-frequency maps as one exportable slit result.

        :param slit_id: Client-stable slit identifier.
        :type slit_id: str
        :param results: Ordered extracted map records.
        :type results: list[dict[str, object]]
        :param freq_indices: Native radio frequency indexes in map order.
        :type freq_indices: list[int]
        :param freq_ghz: Physical frequencies in GHz in map order.
        :type freq_ghz: list[float]
        :returns: Stored bundle whose first map supplies common geometry fields.
        :rtype: dict[str, object]
        """
        if not results:
            raise ValueError("At least one slit map is required")
        bundle = {
            **results[0],
            "maps": results,
            "freq_indices": [int(value) for value in freq_indices],
            "freq_ghz": [float(value) for value in freq_ghz],
        }
        self._slit_results[str(slit_id)] = bundle
        return bundle

    def extract_radio_slit_channels(
        self,
        slit_id: str,
        curve_arcsec: list[list[float]],
        width: int,
        layer: dict[str, object],
        freq_indices: list[int],
        on_progress: object | None = None,
    ) -> tuple[list[dict[str, object]], list[bool], list[float], dict[str, object]]:
        """Extract radio slit maps in one time-major pass over all channels.

        Each frequency keeps the exact single-map cache identity and numeric
        sampling path; only cache misses enter the shared loop.  The loop
        order is frames-outer, channels-inner: for each native time step the
        decoded cube is read once (:meth:`EovsaSequence._ensure_band_planes`
        decodes and caches every requested plane from a single cube read),
        and every missing channel is sampled from it before moving to the
        next frame.  This replaces the previous shape of one full time pass
        per channel with one full time pass total, so selecting more
        channels no longer multiplies wall time by the channel count.

        Two additional per-channel invariants are hoisted out of the frame
        loop, since neither depends on frame content or index:

        * The curve-to-pixel projection (``world_to_pixel``) depends only on
          the fixed instrument WCS header and this channel's static
          (dx, dy) offset, so it is computed once per channel instead of
          once per channel per frame.
        * The perpendicular-width bilinear sample grid derived from that
          projection (:func:`_slit_sample_grid`) is likewise computed once
          per channel.

        Channels whose layer resolves to a raw decoded plane (no difference
        operation, no temporal filter, no radial weighting -- see
        :func:`_radio_layer_is_identity`) skip the full processing pipeline
        and sample the decoded plane directly; channels with non-identity
        processing still call :meth:`processed_tracking_frame` per frame
        (reusing the precomputed projection), so their processing path and
        numeric output are unchanged.

        :param slit_id: Client-stable slit identifier.
        :type slit_id: str
        :param curve_arcsec: Smoothed slit vertices in solar arcseconds.
        :type curve_arcsec: list[list[float]]
        :param width: Perpendicular averaging width in pixels.
        :type width: int
        :param layer: Display-processing parameter snapshot.
        :type layer: dict[str, object]
        :param freq_indices: Ordered native radio channel indexes.
        :type freq_indices: list[int]
        :param on_progress: Optional ``(frames_done, frames_total)`` progress
            callback, reported once per native time step of the single pass
            (not once per channel-frame pair).
        :type on_progress: callable or None
        :returns: Ordered results, cache flags, per-channel walls, and bundle.
        :rtype: tuple[list[dict[str, object]], list[bool], list[float], dict[str, object]]
        """
        canonical_source_id, times, _ = self._tracking_source_geometry(self.radio_source_id)
        _, source = self._tracking_source(canonical_source_id)
        curve = np.asarray(curve_arcsec, dtype=float).reshape(-1, 2)
        if curve.shape[0] < 2 or not np.isfinite(curve).all():
            raise ValueError("Slit curve must contain at least two finite solar-coordinate vertices")
        canonical_curve, requested_is_reversed = _canonical_slit_curve(curve)
        normalized_width = int(np.clip(int(width), 1, MAX_SLIT_WIDTH_PX))
        ordered_indices = list(dict.fromkeys(int(value) for value in freq_indices))
        if not ordered_indices:
            raise ValueError("At least one radio channel is required")
        if any(value < 0 or value >= int(self.eovsa.nfreq) for value in ordered_indices):
            raise ValueError("Radio slit channel is outside the source cube")

        configurations: list[dict[str, object]] = []
        cache_hits: list[bool] = []
        channel_walls = [0.0 for _ in ordered_indices]
        frames_total = len(times)
        for map_index, frequency_index in enumerate(ordered_indices):
            canonical_layer = self._slit_radio_layer(layer, frequency_index)
            cache_key = RENDER_DISK_CACHE.make_key({
                "version": 4,
                "render": "time-distance-slit",
                "source": _source_render_identity(source),
                "sourceId": canonical_source_id,
                "timeMjd": hashlib.sha256(np.asarray(times, dtype=np.float64).tobytes()).hexdigest(),
                "curveArcsec": np.round(canonical_curve, 8).tolist(),
                "width": normalized_width,
                "freqIndex": frequency_index,
                "offsetsRev": int(canonical_layer["offsetsRev"]),
                "layerParams": canonical_layer,
            })
            lookup_started = time.perf_counter()
            cached = RENDER_DISK_CACHE.get(cache_key, "npz")
            channel_walls[map_index] += time.perf_counter() - lookup_started
            canonical_result = _decode_slit_result_npz(cached) if cached is not None else None
            cache_hit = canonical_result is not None
            cache_hits.append(cache_hit)
            configurations.append({
                "map_index": map_index,
                "freq_index": frequency_index,
                "layer": canonical_layer,
                "cache_key": cache_key,
                "canonical_result": canonical_result,
            })

        missing = [configuration for configuration in configurations if configuration["canonical_result"] is None]
        if not missing:
            # Every channel was already cached: the single pass has nothing
            # left to do, so report it complete without a fabricated
            # per-channel progress ramp.
            if callable(on_progress):
                on_progress(frames_total, frames_total)
        else:
            self._slit_extract_cancel.clear()
            eovsa = self.eovsa
            # WCS projection only needs an array of the right shape; its
            # values never reach the transform (see world_to_pixel), so one
            # zero-filled reference plane can be reused for every channel's
            # one-time projection instead of allocating (or reading) real
            # frame data just to throw it away.
            reference_plane = np.zeros(eovsa.shape, dtype=np.float32)
            for configuration in missing:
                canonical_layer = configuration["layer"]
                x_offset = float(canonical_layer.get("xOffsetArcsec", 0.0))
                y_offset = float(canonical_layer.get("yOffsetArcsec", 0.0))
                reference_pixels = np.asarray(
                    eovsa.world_to_pixel(reference_plane, canonical_curve, x_offset, y_offset), dtype=float
                )
                sampled_curve = _resample_slit_world_curve(canonical_curve, reference_pixels)
                # Time-invariant: same header, same shape, same fixed
                # per-channel offset on every frame. Computed once here
                # instead of once per frame inside the time loop below.
                pixels = np.asarray(
                    eovsa.world_to_pixel(reference_plane, sampled_curve, x_offset, y_offset), dtype=float
                )
                sample_y, sample_x, sample_width, npix = _slit_sample_grid(pixels, normalized_width)
                configuration["sampled_curve"] = sampled_curve
                configuration["distance_arcsec"] = np.r_[
                    0.0, np.cumsum(np.linalg.norm(np.diff(sampled_curve, axis=0), axis=1))
                ].astype(np.float32)
                configuration["sample_grid"] = (sample_y, sample_x, sample_width, npix)
                configuration["intensity"] = np.empty((npix, len(times)), dtype=np.float32)
                configuration["identity"] = _radio_layer_is_identity(canonical_layer)

            # The decode-once identity fast path below reaches into
            # EovsaSequence internals (_ensure_band_planes, _read_band,
            # .files); small test doubles and compatibility sources expose
            # only frame_for_aia_time, so route everything through the full
            # processed_tracking_frame path for those (see the same
            # isinstance guard around radio contour extraction).
            use_fast_path = isinstance(eovsa, EovsaSequence)
            identity_channels = [
                configuration for configuration in missing if use_fast_path and configuration["identity"]
            ]
            processed_channels = [
                configuration for configuration in missing if not (use_fast_path and configuration["identity"])
            ]
            identity_freqs = sorted({int(configuration["freq_index"]) for configuration in identity_channels})
            last_native_index = len(eovsa.files) - 1 if identity_freqs else 0

            for frame_index in range(len(times)):
                if self._slit_extract_cancel.is_set():
                    raise SlitExtractionCancelled("Time-distance extraction cancelled")
                native_index = int(np.clip(frame_index, 0, last_native_index))

                if identity_freqs:
                    # One cube read serves every identity channel at this
                    # frame; _ensure_band_planes decodes it once and caches
                    # each requested plane, so the per-channel _read_band
                    # calls below are cache hits.
                    eovsa._ensure_band_planes(native_index, identity_freqs)
                    for configuration in identity_channels:
                        map_index = int(configuration["map_index"])
                        channel_started = time.perf_counter()
                        plane = np.asarray(
                            eovsa._read_band(native_index, int(configuration["freq_index"])), dtype=float
                        )
                        sample_y, sample_x, sample_width, npix = configuration["sample_grid"]
                        sampled = map_coordinates(
                            plane, [sample_y, sample_x], order=1, mode="constant", cval=np.nan, prefilter=False,
                        )
                        configuration["intensity"][:, frame_index] = _reduce_slit_samples(sampled, sample_width, npix)
                        channel_walls[map_index] += time.perf_counter() - channel_started

                for configuration in processed_channels:
                    map_index = int(configuration["map_index"])
                    channel_started = time.perf_counter()
                    canonical_layer = configuration["layer"]
                    data = np.asarray(self.processed_tracking_frame(frame_index, canonical_layer), dtype=float)
                    sample_y, sample_x, sample_width, npix = configuration["sample_grid"]
                    sampled = map_coordinates(
                        data, [sample_y, sample_x], order=1, mode="constant", cval=np.nan, prefilter=False,
                    )
                    configuration["intensity"][:, frame_index] = _reduce_slit_samples(sampled, sample_width, npix)
                    channel_walls[map_index] += time.perf_counter() - channel_started

                if callable(on_progress):
                    on_progress(frame_index + 1, frames_total)

            for configuration in missing:
                canonical_result = {
                    "intensity": configuration["intensity"],
                    "distance_arcsec": configuration["distance_arcsec"],
                    "time_mjd": np.asarray(times, dtype=np.float64),
                    "curve_vertices_arcsec": np.asarray(configuration["sampled_curve"], dtype=np.float64),
                }
                write_started = time.perf_counter()
                content = _encode_slit_result_npz(
                    canonical_result, canonical_source_id, dict(configuration["layer"])
                )
                RENDER_DISK_CACHE.put(str(configuration["cache_key"]), content, "npz", durable=False)
                channel_walls[int(configuration["map_index"])] += time.perf_counter() - write_started
                configuration["canonical_result"] = canonical_result

        results: list[dict[str, object]] = []
        for configuration in configurations:
            canonical_result = dict(configuration["canonical_result"])
            result = _reverse_slit_result(canonical_result) if requested_is_reversed else canonical_result
            results.append({
                **result,
                "source_id": canonical_source_id,
                "layer_params": dict(configuration["layer"]),
                "width": normalized_width,
                "cache_key": str(configuration["cache_key"]),
            })
        frequencies_ghz = [float(self.eovsa.freqs_hz[index] / 1e9) for index in ordered_indices]
        bundle = self.store_slit_result_bundle(
            slit_id, results, ordered_indices, frequencies_ghz
        )
        return results, cache_hits, channel_walls, bundle

    def extract_slit_batch(
        self,
        entries: list[dict[str, object]],
        on_progress: object | None = None,
    ) -> dict[str, dict[str, object]]:
        """Extract many slits sharing one source and processing snapshot in one pass.

        Companion to :meth:`extract_slit` and :meth:`extract_radio_slit_channels`
        for the "extract all slits" batch path.  Callers (see the
        ``/slits/extract-batch`` endpoint) must already have grouped the
        entries by canonical source and by a serialized processing-layer
        snapshot, mirroring the grouping the client performs before issuing
        one batch request per (sourceId, layerParams) group -- every entry
        here is assumed to share the same canonical source and, other than
        its own curve/width/channels, the same ``layerParams``.

        Each ``(slit, channel)`` pairing keeps the exact single-map cache
        identity and numeric sampling path as :meth:`extract_slit` /
        :meth:`extract_radio_slit_channels`: the same disk-cache key formula,
        the same bilinear sampling, the same identity-plane fast path for
        radio.  Every configuration's cache entry is checked before any
        frame is read; only genuine misses enter the shared frame-outer
        loop, so a batch of all cache hits returns immediately and a batch
        with some misses reads each native frame exactly once no matter how
        many slits (and, for radio, channels) sample it.

        Non-radio (context/AIA) slits additionally share one canonical layer
        across the whole batch (since ``layerParams`` is a shared grouping
        key), so the processed frame itself -- the expensive part -- is
        computed once per native time step and reused by every missing
        slit's projection and sampling; only the per-slit curve-to-pixel
        projection repeats per frame, because :meth:`AiaSequence.world_to_pixel`
        is frame-dependent (differential-rotation tracking) and cannot be
        hoisted out of the frame loop the way radio's fixed-header
        projection can.  Radio slits sharing one frequency index similarly
        share one processed (or identity-plane) frame per time step.

        :param entries: Per-slit extraction requests sharing one canonical
            source and processing-layer snapshot.  Each is a dict with
            ``slitId``, ``sourceId``, ``curveArcsec``, ``width``,
            ``layerParams``, and (for radio sources) ``freqIndices``.
        :type entries: list[dict[str, object]]
        :param on_progress: Optional ``(frames_done, frames_total)``
            callback, reported once per native time step of the single
            shared pass (not once per slit or per channel).
        :type on_progress: callable or None
        :returns: Mapping from slit id to its ``results``/``cache_hits``/
            ``wall_times``/``bundle``/``freq_indices``, in the same shapes
            :meth:`extract_slit` and :meth:`extract_radio_slit_channels`
            hand their callers.
        :rtype: dict[str, dict[str, object]]
        :raises ValueError: If ``entries`` is empty, a curve is degenerate,
            entries disagree on the canonical source, or a radio channel is
            out of range.
        """
        if not entries:
            raise ValueError("At least one slit entry is required")
        canonical_source_id, times, _ = self._tracking_source_geometry(entries[0]["sourceId"])
        for entry in entries[1:]:
            other_source_id, _, _ = self._tracking_source_geometry(entry["sourceId"])
            if other_source_id != canonical_source_id:
                raise ValueError("All batched slits must share one canonical source")
        _, source = self._tracking_source(canonical_source_id)
        is_radio = canonical_source_id == self.radio_source_id
        frames_total = len(times)
        source_identity = _source_render_identity(source)
        times_hash = hashlib.sha256(np.asarray(times, dtype=np.float64).tobytes()).hexdigest()

        # One configuration per (slit, channel) pairing -- a radio slit
        # expands to one configuration per requested frequency index (the
        # granularity extract_radio_slit_channels caches at); a non-radio
        # slit is exactly one configuration.
        configurations: list[dict[str, object]] = []
        slit_map_indices: dict[str, list[int]] = {}
        for entry in entries:
            slit_id = str(entry["slitId"])
            curve = np.asarray(entry["curveArcsec"], dtype=float).reshape(-1, 2)
            if curve.shape[0] < 2 or not np.isfinite(curve).all():
                raise ValueError(f"Slit curve for {slit_id!r} must contain at least two finite vertices")
            canonical_curve, requested_is_reversed = _canonical_slit_curve(curve)
            normalized_width = int(np.clip(int(entry.get("width", 3)), 1, MAX_SLIT_WIDTH_PX))
            base_layer = dict(entry["layerParams"])
            base_layer["sourceId"] = canonical_source_id
            if is_radio:
                requested = entry.get("freqIndices") or [int(base_layer.get("freqIndex", 0))]
                freq_list = list(dict.fromkeys(int(value) for value in requested))
                if not freq_list:
                    raise ValueError(f"Slit {slit_id!r} requires at least one radio channel")
                if any(value < 0 or value >= int(self.eovsa.nfreq) for value in freq_list):
                    raise ValueError(f"Slit {slit_id!r} channel is outside the radio cube")
            else:
                freq_list = [-1]
            slit_map_indices[slit_id] = []
            for freq_index in freq_list:
                canonical_layer = self._slit_radio_layer(base_layer, freq_index) if is_radio else base_layer
                cache_key = RENDER_DISK_CACHE.make_key({
                    "version": 4,
                    "render": "time-distance-slit",
                    "source": source_identity,
                    "sourceId": canonical_source_id,
                    "timeMjd": times_hash,
                    "curveArcsec": np.round(canonical_curve, 8).tolist(),
                    "width": normalized_width,
                    "freqIndex": int(canonical_layer.get("freqIndex", -1)),
                    "offsetsRev": int(canonical_layer.get("offsetsRev", 0)),
                    "layerParams": canonical_layer,
                })
                cached = RENDER_DISK_CACHE.get(cache_key, "npz")
                canonical_result = _decode_slit_result_npz(cached) if cached is not None else None
                map_index = len(configurations)
                slit_map_indices[slit_id].append(map_index)
                configurations.append({
                    "slit_id": slit_id,
                    "map_index": map_index,
                    "freq_index": freq_index,
                    "layer": canonical_layer,
                    "curve": canonical_curve,
                    "width": normalized_width,
                    "requested_is_reversed": requested_is_reversed,
                    "cache_key": cache_key,
                    "cache_hit": canonical_result is not None,
                    "canonical_result": canonical_result,
                    "wall_seconds": 0.0,
                })

        missing = [configuration for configuration in configurations if configuration["canonical_result"] is None]
        if not missing:
            # Every (slit, channel) pairing was already cached: nothing left
            # to read, report complete without a fabricated progress ramp.
            if callable(on_progress):
                on_progress(frames_total, frames_total)
        elif is_radio:
            self._slit_extract_cancel.clear()
            eovsa = self.eovsa
            reference_plane = np.zeros(eovsa.shape, dtype=np.float32)
            for configuration in missing:
                canonical_layer = configuration["layer"]
                x_offset = float(canonical_layer.get("xOffsetArcsec", 0.0))
                y_offset = float(canonical_layer.get("yOffsetArcsec", 0.0))
                reference_pixels = np.asarray(
                    eovsa.world_to_pixel(reference_plane, configuration["curve"], x_offset, y_offset), dtype=float
                )
                sampled_curve = _resample_slit_world_curve(configuration["curve"], reference_pixels)
                # Time-invariant projection (fixed header, fixed per-channel
                # offset on every frame): computed once here per (slit,
                # channel), same hoisting as extract_radio_slit_channels.
                pixels = np.asarray(
                    eovsa.world_to_pixel(reference_plane, sampled_curve, x_offset, y_offset), dtype=float
                )
                sample_y, sample_x, sample_width, npix = _slit_sample_grid(pixels, configuration["width"])
                configuration["sampled_curve"] = sampled_curve
                configuration["distance_arcsec"] = np.r_[
                    0.0, np.cumsum(np.linalg.norm(np.diff(sampled_curve, axis=0), axis=1))
                ].astype(np.float32)
                configuration["sample_grid"] = (sample_y, sample_x, sample_width, npix)
                configuration["intensity"] = np.empty((npix, len(times)), dtype=np.float32)
                configuration["identity"] = _radio_layer_is_identity(canonical_layer)

            use_fast_path = isinstance(eovsa, EovsaSequence)
            identity_channels = [
                configuration for configuration in missing if use_fast_path and configuration["identity"]
            ]
            processed_channels = [
                configuration for configuration in missing if not (use_fast_path and configuration["identity"])
            ]
            identity_freqs = sorted({int(configuration["freq_index"]) for configuration in identity_channels})
            last_native_index = len(eovsa.files) - 1 if identity_freqs else 0
            # Slits sharing one frequency index share one identity plane read
            # (below) and, for processed channels, one processed_tracking_frame
            # call per native time step -- grouped by the exact canonical
            # layer content so no assumption about the batch's grouping
            # contract is required for correctness.
            processed_groups: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
            for configuration in processed_channels:
                key = json.dumps(configuration["layer"], sort_keys=True, default=str)
                processed_groups.setdefault(key, (configuration["layer"], []))[1].append(configuration)

            for frame_index in range(len(times)):
                if self._slit_extract_cancel.is_set():
                    raise SlitExtractionCancelled("Time-distance extraction cancelled")
                native_index = int(np.clip(frame_index, 0, last_native_index))

                if identity_freqs:
                    eovsa._ensure_band_planes(native_index, identity_freqs)
                    plane_cache: dict[int, np.ndarray] = {}
                    for configuration in identity_channels:
                        freq_index = int(configuration["freq_index"])
                        channel_started = time.perf_counter()
                        if freq_index not in plane_cache:
                            plane_cache[freq_index] = np.asarray(
                                eovsa._read_band(native_index, freq_index), dtype=float
                            )
                        plane = plane_cache[freq_index]
                        sample_y, sample_x, sample_width, npix = configuration["sample_grid"]
                        sampled = map_coordinates(
                            plane, [sample_y, sample_x], order=1, mode="constant", cval=np.nan, prefilter=False,
                        )
                        configuration["intensity"][:, frame_index] = _reduce_slit_samples(sampled, sample_width, npix)
                        configuration["wall_seconds"] += time.perf_counter() - channel_started

                for canonical_layer, group_configs in processed_groups.values():
                    frame_started = time.perf_counter()
                    data = np.asarray(self.processed_tracking_frame(frame_index, canonical_layer), dtype=float)
                    shared_elapsed = time.perf_counter() - frame_started
                    for configuration in group_configs:
                        channel_started = time.perf_counter()
                        sample_y, sample_x, sample_width, npix = configuration["sample_grid"]
                        sampled = map_coordinates(
                            data, [sample_y, sample_x], order=1, mode="constant", cval=np.nan, prefilter=False,
                        )
                        configuration["intensity"][:, frame_index] = _reduce_slit_samples(sampled, sample_width, npix)
                        configuration["wall_seconds"] += shared_elapsed / len(group_configs) + (
                            time.perf_counter() - channel_started
                        )

                if callable(on_progress):
                    on_progress(frame_index + 1, frames_total)
        else:
            self._slit_extract_cancel.clear()
            aia = self.aia
            for configuration in missing:
                reference_pixels = np.asarray(aia.world_to_pixel(0, configuration["curve"]), dtype=float)
                sampled_curve = _resample_slit_world_curve(configuration["curve"], reference_pixels)
                configuration["sampled_curve"] = sampled_curve
                configuration["distance_arcsec"] = np.r_[
                    0.0, np.cumsum(np.linalg.norm(np.diff(sampled_curve, axis=0), axis=1))
                ].astype(np.float32)
                configuration["intensity"] = np.empty((sampled_curve.shape[0], len(times)), dtype=np.float32)

            # Every missing configuration in a non-radio batch shares one
            # canonical layer (the group's shared processing snapshot -- see
            # the grouping contract above), so the processed frame itself is
            # computed once per native time step and reused across slits;
            # only the per-slit curve projection (frame-dependent for AIA --
            # see AiaSequence.world_to_pixel) and sampling repeat per slit.
            layer_groups: dict[str, tuple[dict[str, object], list[dict[str, object]]]] = {}
            for configuration in missing:
                key = json.dumps(configuration["layer"], sort_keys=True, default=str)
                layer_groups.setdefault(key, (configuration["layer"], []))[1].append(configuration)

            for frame_index in range(len(times)):
                if self._slit_extract_cancel.is_set():
                    raise SlitExtractionCancelled("Time-distance extraction cancelled")
                for canonical_layer, group_configs in layer_groups.values():
                    frame_started = time.perf_counter()
                    data = np.asarray(self.processed_tracking_frame(frame_index, canonical_layer), dtype=float)
                    shared_elapsed = time.perf_counter() - frame_started
                    for configuration in group_configs:
                        slit_started = time.perf_counter()
                        pixels = np.asarray(aia.world_to_pixel(frame_index, configuration["sampled_curve"]), dtype=float)
                        configuration["intensity"][:, frame_index] = sample_slit_profile(
                            data, pixels, configuration["width"]
                        )
                        configuration["wall_seconds"] += shared_elapsed / len(group_configs) + (
                            time.perf_counter() - slit_started
                        )
                if callable(on_progress):
                    on_progress(frame_index + 1, frames_total)

        for configuration in missing:
            canonical_result = {
                "intensity": configuration["intensity"],
                "distance_arcsec": configuration["distance_arcsec"],
                "time_mjd": np.asarray(times, dtype=np.float64),
                "curve_vertices_arcsec": np.asarray(configuration["sampled_curve"], dtype=np.float64),
            }
            write_started = time.perf_counter()
            content = _encode_slit_result_npz(canonical_result, canonical_source_id, dict(configuration["layer"]))
            RENDER_DISK_CACHE.put(str(configuration["cache_key"]), content, "npz", durable=False)
            configuration["wall_seconds"] += time.perf_counter() - write_started
            configuration["canonical_result"] = canonical_result

        bundles: dict[str, dict[str, object]] = {}
        for entry in entries:
            slit_id = str(entry["slitId"])
            results: list[dict[str, object]] = []
            cache_hits: list[bool] = []
            wall_times: list[float] = []
            freq_indices_out: list[int] = []
            for map_index in slit_map_indices[slit_id]:
                configuration = configurations[map_index]
                canonical_result = dict(configuration["canonical_result"])
                result = (
                    _reverse_slit_result(canonical_result)
                    if configuration["requested_is_reversed"]
                    else canonical_result
                )
                results.append({
                    **result,
                    "source_id": canonical_source_id,
                    "layer_params": dict(configuration["layer"]),
                    "width": configuration["width"],
                    "cache_key": str(configuration["cache_key"]),
                })
                cache_hits.append(bool(configuration["cache_hit"]))
                wall_times.append(float(configuration["wall_seconds"]))
                freq_indices_out.append(int(configuration["freq_index"]))
            frequencies_ghz = [float(self.eovsa.freqs_hz[index] / 1e9) for index in freq_indices_out] if is_radio else []
            bundle = self.store_slit_result_bundle(
                slit_id, results, freq_indices_out if is_radio else [], frequencies_ghz
            )
            bundles[slit_id] = {
                "results": results,
                "cache_hits": cache_hits,
                "wall_times": wall_times,
                "bundle": bundle,
                "freq_indices": freq_indices_out,
            }
        return bundles

    def reverse_slit_result(self, slit_id: str) -> None:
        """Reverse an extracted slit in memory without starting extraction.

        :param slit_id: Client-stable identifier for the extracted slit.
        :type slit_id: str
        :raises ValueError: If this session has no extracted result for the slit.
        """
        key = str(slit_id)
        result = self._slit_results.get(key)
        if result is None:
            raise ValueError("Extract the slit before reversing it")
        self._slit_results[key] = _reverse_slit_result(result)

    def cancel_slit_extraction(self) -> None:
        """Request cooperative cancellation of the active slit extraction."""
        self._slit_extract_cancel.set()

    def slit_result_npz(self, slit_id: str, shift_seconds: float = 0.0) -> bytes:
        """Return an export NPZ for the most recently opened slit result.

        :param slit_id: Client-stable slit identifier.
        :type slit_id: str
        :param shift_seconds: Display-only time shift recorded in the export.
        :type shift_seconds: float
        :returns: Pickle-free compressed NPZ bytes.
        :rtype: bytes
        :raises ValueError: If the slit has not been extracted in this session.
        """
        result = self._slit_results.get(str(slit_id))
        if result is None:
            raise ValueError("Extract the slit before exporting it")
        return _encode_slit_result_npz(
            result,
            str(result["source_id"]),
            dict(result["layer_params"]),
            float(shift_seconds),
        )

    def _tracking_roi_pixels(self, source_id: object, frame_index: int) -> np.ndarray | None:
        """Project the stored world ROI into one tracked-source frame.

        :param source_id: Renderable context or radio source identifier.
        :type source_id: object
        :param frame_index: Native tracked-source frame index.
        :type frame_index: int
        :returns: Tracked-source pixel polygon, or ``None`` when no ROI is stored.
        :rtype: numpy.ndarray or None
        """
        if not self.roi_world:
            return None
        return self._tracking_world_to_pixel(
            source_id, frame_index, np.asarray(self.roi_world, dtype=float)
        )

    def auto_track(
        self,
        track_ids: list[str],
        layer: dict[str, object],
        direction: str,
        start_frame: int,
        range_start: int,
        range_end: int,
        patch_radius: int = 6,
        search_radius: int = 18,
        confidence_threshold: float = 0.5,
        on_progress: object | None = None,
    ) -> list[dict[str, object]]:
        """Auto-track selected session tracks within inclusive native bounds.

        :param track_ids: Track identifiers to update.
        :type track_ids: list[str]
        :param layer: Tracked source-layer processing parameters.
        :type layer: dict[str, object]
        :param direction: ``forward``, ``backward``, or ``both``.
        :type direction: str
        :param start_frame: Preferred starting frame.
        :type start_frame: int
        :param range_start: Inclusive lower frame bound.
        :type range_start: int
        :param range_end: Inclusive upper frame bound.
        :type range_end: int
        :param patch_radius: Template radius in pixels.
        :type patch_radius: int
        :param search_radius: Prediction-centered search radius in pixels.
        :type search_radius: int
        :param confidence_threshold: Minimum accepted NCC peak.
        :type confidence_threshold: float
        :param on_progress: Optional progress callback.
        :type on_progress: callable or None
        :returns: All canonical session tracks after updating the selection.
        :rtype: list[dict[str, object]]
        """
        if direction not in {"forward", "backward", "both"}:
            raise ValueError("direction must be forward, backward, or both")
        source_id, times, _ = self._tracking_source_geometry(layer.get("sourceId"))
        last_frame = max(0, len(times) - 1)
        lower = int(np.clip(min(range_start, range_end), 0, last_frame))
        upper = int(np.clip(max(range_start, range_end), 0, last_frame))
        selected = set(track_ids)
        mismatched = [
            str(track.get("id"))
            for track in self.tracks
            if str(track.get("id")) in selected
            and self._tracking_source(track.get("sourceId"))[0] != source_id
        ]
        if mismatched:
            raise ValueError("Auto-track request mixes trajectories from different sources")
        self._tracking_cancel.clear()
        frame_getter = lambda index: self.processed_tracking_frame(index, layer)
        updated: list[dict[str, object]] = []
        for original in self.tracks:
            if str(original.get("id")) not in selected:
                updated.append(original)
                continue
            track = self._normalize_tracking_track(original)
            points = [point for point in track["points"] if isinstance(point, dict)]
            if not points:
                updated.append(track)
                continue
            origin = min(points, key=lambda point: abs(int(point["frameIndex"]) - int(start_frame)))
            merged = {int(point["frameIndex"]): dict(point) for point in points}
            states: list[str] = []
            roi_pixels = self._tracking_roi_pixels(source_id, int(origin["frameIndex"]))
            if direction in {"forward", "both"} and int(origin["frameIndex"]) < upper:
                forward, state = track_ncc_pass(
                    frame_getter, origin, upper, 1, patch_radius, search_radius,
                    confidence_threshold, roi_pixels, times,
                    on_progress, self._tracking_cancel.is_set,
                )
                merged.update({int(point["frameIndex"]): point for point in forward})
                states.append(state)
            if direction in {"backward", "both"} and int(origin["frameIndex"]) > lower:
                backward, state = track_ncc_pass(
                    frame_getter, origin, lower, -1, patch_radius, search_radius,
                    confidence_threshold, roi_pixels, times,
                    on_progress, self._tracking_cancel.is_set,
                )
                merged.update({int(point["frameIndex"]): point for point in backward})
                states.append(state)
            anchor_by_frame = {int(anchor["frameIndex"]): anchor for anchor in track["anchors"]}
            for frame_index, anchor in anchor_by_frame.items():
                merged[frame_index] = _tracking_point(frame_index, float(anchor["x"]), float(anchor["y"]), 1.0, True, float(anchor["mjd"]))
            state = next((item for item in states if item != "active"), "active")
            track["points"] = sorted(merged.values(), key=lambda point: int(point["frameIndex"]))
            track["state"] = state
            updated.append(self._normalize_tracking_track(track))
        self.tracks = updated
        self.write_tracking_csv()
        return self.tracks

    def cancel_tracking(self) -> None:
        """Request cancellation of the active auto-track operation.

        :returns: ``None``.
        :rtype: None
        """
        self._tracking_cancel.set()

    def retrack_track(
        self,
        track_value: dict[str, object],
        layer: dict[str, object],
        patch_radius: int = 6,
        search_radius: int = 18,
        confidence_threshold: float = 0.5,
    ) -> dict[str, object]:
        """Constrained re-track all segments of one edited anchor graph.

        :param track_value: Track containing the authoritative edited anchors.
        :type track_value: dict[str, object]
        :param layer: Tracked source-layer processing parameters.
        :type layer: dict[str, object]
        :param patch_radius: Template radius in pixels.
        :type patch_radius: int
        :param search_radius: Prediction-centered search radius in pixels.
        :type search_radius: int
        :param confidence_threshold: Minimum accepted NCC peak.
        :type confidence_threshold: float
        :returns: Canonical re-tracked session track.
        :rtype: dict[str, object]
        :raises ValueError: If the edited track has no anchors.
        """
        track = self._normalize_tracking_track(track_value)
        source_id, times, _ = self._tracking_source_geometry(track.get("sourceId"))
        layer_source_id = self._tracking_source(layer.get("sourceId"))[0]
        if layer_source_id != source_id:
            raise ValueError("Re-track layer does not match the trajectory source")
        anchors = [anchor for anchor in track["anchors"] if isinstance(anchor, dict)]
        if not anchors:
            raise ValueError("A track must retain at least one anchor")
        old_points = [point for point in track["points"] if isinstance(point, dict)]
        lower = min([int(point["frameIndex"]) for point in old_points] + [int(anchors[0]["frameIndex"])])
        upper = max([int(point["frameIndex"]) for point in old_points] + [int(anchors[-1]["frameIndex"])])
        frame_getter = lambda index: self.processed_tracking_frame(index, layer)
        roi_pixels = self._tracking_roi_pixels(source_id, int(anchors[0]["frameIndex"]))
        merged: dict[int, dict[str, object]] = {}
        point_by_frame = {int(point["frameIndex"]): point for point in old_points}
        if lower < int(anchors[0]["frameIndex"]):
            segment = constrained_ncc_segment(
                frame_getter,
                point_by_frame[lower],
                anchors[0],
                patch_radius,
                search_radius,
                confidence_threshold,
                roi_pixels,
                times,
            )
            merged.update({int(point["frameIndex"]): point for point in segment})
        for earlier, later in zip(anchors, anchors[1:]):
            segment = constrained_ncc_segment(
                frame_getter, earlier, later, patch_radius, search_radius,
                confidence_threshold, roi_pixels, times,
            )
            merged.update({int(point["frameIndex"]): point for point in segment})
        if int(anchors[-1]["frameIndex"]) < upper:
            segment = constrained_ncc_segment(
                frame_getter,
                anchors[-1],
                point_by_frame[upper],
                patch_radius,
                search_radius,
                confidence_threshold,
                roi_pixels,
                times,
            )
            merged.update({int(point["frameIndex"]): point for point in segment})
        for anchor in anchors:
            frame_index = int(anchor["frameIndex"])
            merged[frame_index] = _tracking_point(
                frame_index, float(anchor["x"]), float(anchor["y"]), 1.0, True,
                float(anchor["mjd"]),
            )
        track["points"] = sorted(merged.values(), key=lambda point: int(point["frameIndex"]))
        track["state"] = "active"
        track = self._normalize_tracking_track(track)
        self.tracks = [track if str(item.get("id")) == str(track["id"]) else item for item in self.tracks]
        if not any(str(item.get("id")) == str(track["id"]) for item in self.tracks):
            self.tracks.append(track)
        self.write_tracking_csv()
        return track

    def suggest_tracking_seeds(
        self,
        frame_index: int,
        layer: dict[str, object],
        percentile: float = 20.0,
        minimum_separation: float = 8.0,
        limit: int = 20,
    ) -> list[dict[str, float]]:
        """Find separated dark local minima in the processed frame and ROI.

        :param frame_index: Native tracked-source frame index.
        :type frame_index: int
        :param layer: Tracked source-layer processing parameters.
        :type layer: dict[str, object]
        :param percentile: Maximum darkness percentile to consider.
        :type percentile: float
        :param minimum_separation: Minimum accepted center separation in pixels.
        :type minimum_separation: float
        :param limit: Maximum number of suggestions.
        :type limit: int
        :returns: Suggested pixel centers ordered darkest first.
        :rtype: list[dict[str, float]]
        :raises ValueError: If no ROI is stored.
        """
        source_id, times, _ = self._tracking_source_geometry(layer.get("sourceId"))
        index = int(np.clip(frame_index, 0, len(times) - 1))
        roi = self._tracking_roi_pixels(source_id, index)
        if roi is None or roi.shape[0] < 3:
            raise ValueError("Suggested seeds require an ROI")
        data = self.processed_tracking_frame(index, layer)
        ny, nx = data.shape
        yy, xx = np.indices(data.shape)
        pixels = np.column_stack([xx.ravel(), yy.ravel()])
        inside = MplPath(roi).contains_points(pixels, radius=1e-7).reshape(data.shape)
        finite = np.isfinite(data) & inside
        if not finite.any():
            return []
        threshold = float(np.nanpercentile(data[finite], float(np.clip(percentile, 0.0, 100.0))))
        local_minimum = data <= minimum_filter(np.where(np.isfinite(data), data, np.inf), size=5, mode="nearest")
        candidates = np.column_stack(np.nonzero(finite & local_minimum & (data <= threshold)))
        candidates = sorted(candidates, key=lambda item: float(data[int(item[0]), int(item[1])]))
        accepted: list[dict[str, float]] = []
        separation = max(1.0, float(minimum_separation))
        for y, x in candidates:
            if any(math.hypot(float(x) - item["x"], float(y) - item["y"]) < separation for item in accepted):
                continue
            accepted.append({"x": float(x), "y": float(y), "value": float(data[int(y), int(x)])})
            if len(accepted) >= max(1, int(limit)):
                break
        return accepted

    def extract_sads(self, search_radius: int = 8) -> list[dict[str, object]]:
        """Load and refine legacy ``markpos.pickle`` feature seeds.

        :param search_radius: Local refinement radius in context pixels.
        :type search_radius: int
        :returns: Legacy flat rows, with canonical nested tracks also stored.
        :rtype: list[dict[str, object]]
        :raises FileNotFoundError: If the manifest has no usable seed file.
        """
        if not self.seed_path.exists():
            raise FileNotFoundError(f"No seed file configured for this session: {self.seed_path}")
        with self.seed_path.open("rb") as handle:
            readme, uttime, pos1, pos2, pos3, pos4, v1, v2, v3, v4 = pickle.load(handle)
        del readme, v1, v2, v3, v4
        seed_tracks = [np.asarray(pos, dtype=float) for pos in (pos1, pos2, pos3, pos4)]
        event_date = self.aia.times[0].datetime.date().isoformat()
        seed_values = [str(item) for item in uttime]
        seed_times = Time(
            [value if "T" in value or re.match(r"\d{4}-\d{2}-\d{2}", value) else f"{event_date}T{value}" for value in seed_values],
            format="isot",
        )
        scale = float(self.aia.map_for_frame(0).scale.axis1.to_value(u.arcsec / u.pix))
        ref = np.array([750.0, 160.0])
        km_per_arcsec = self.aia.km_per_arcsec()
        rows: list[dict[str, object]] = []

        for sad_idx, seeds in enumerate(seed_tracks, start=1):
            sad_rows: list[dict[str, object]] = []
            for seed_idx, seed in enumerate(seeds):
                if seed[0] <= 0 or seed[1] <= 0:
                    continue
                time_mjd = seed_times[seed_idx].mjd
                frame_idx = int(np.nanargmin(np.abs(self.aia.times.mjd - time_mjd)))
                seed_world = ref + seed * scale
                seed_pix = self.aia.world_to_pixel(frame_idx, seed_world.reshape(1, 2))[0]
                y0 = max(0, int(round(seed_pix[1])) - search_radius)
                y1 = min(self.aia.shape[0], int(round(seed_pix[1])) + search_radius + 1)
                x0 = max(0, int(round(seed_pix[0])) - search_radius)
                x1 = min(self.aia.shape[1], int(round(seed_pix[0])) + search_radius + 1)
                frame = self.aia.frame(frame_idx)
                sub = frame[y0:y1, x0:x1]
                quality = "seed"
                if sub.size and np.isfinite(sub).any():
                    darkness = np.clip(1.0 - np.nan_to_num(sub, nan=1.0), 0.0, None)
                    if np.nansum(darkness) > 0:
                        cy, cx = center_of_mass(darkness)
                        refined_pix = np.array([x0 + cx, y0 + cy], dtype=float)
                        quality = "weighted_min"
                    else:
                        iy, ix = np.unravel_index(int(np.nanargmin(sub)), sub.shape)
                        refined_pix = np.array([x0 + ix, y0 + iy], dtype=float)
                        quality = "local_min"
                else:
                    refined_pix = seed_pix
                    quality = "seed_no_signal"
                refined_world = self.aia.pixel_to_world(frame_idx, refined_pix.reshape(1, 2))[0]
                if not bool(self._inside_roi(refined_world.reshape(1, 2))[0]):
                    continue
                sad_rows.append({
                    "sad_id": sad_idx,
                    "time_utc": seed_times[seed_idx].isot,
                    "time_mjd": float(time_mjd),
                    "frame_index": frame_idx,
                    "x_arcsec": float(refined_world[0]),
                    "y_arcsec": float(refined_world[1]),
                    "x_pix": float(refined_pix[0]),
                    "y_pix": float(refined_pix[1]),
                    "vx_arcsec_s": "",
                    "vy_arcsec_s": "",
                    "speed_km_s": "",
                    "quality": quality,
                })
            for prev, cur in zip(sad_rows, sad_rows[1:]):
                dt = (float(cur["time_mjd"]) - float(prev["time_mjd"])) * 86400.0
                if dt > 0:
                    vx = (float(cur["x_arcsec"]) - float(prev["x_arcsec"])) / dt
                    vy = (float(cur["y_arcsec"]) - float(prev["y_arcsec"])) / dt
                    cur["vx_arcsec_s"] = vx
                    cur["vy_arcsec_s"] = vy
                    cur["speed_km_s"] = math.hypot(vx, vy) * km_per_arcsec
            rows.extend(sad_rows)

        self.sad_tracks = rows
        self.feature_tracks = rows
        palette = ["#56c7d9", "#ee65be", "#ffcc66", "#8fd694"]
        legacy_tracks: list[dict[str, object]] = []
        for sad_id in sorted({int(row["sad_id"]) for row in rows}):
            group = [row for row in rows if int(row["sad_id"]) == sad_id]
            anchors = [{
                "frameIndex": int(row["frame_index"]),
                "mjd": float(row["time_mjd"]),
                "x": float(row["x_pix"]),
                "y": float(row["y_pix"]),
            } for row in group]
            legacy_tracks.append(self._normalize_tracking_track({
                "id": f"file-{sad_id}",
                "label": f"File track {sad_id}",
                "color": palette[(sad_id - 1) % len(palette)],
                "anchors": anchors,
                "points": [{**anchor, "confidence": 1.0, "isAnchor": True} for anchor in anchors],
                "state": "active",
            }))
        self.tracks = legacy_tracks
        _write_csv(self.output_dir / "sad_tracks.csv", rows, SAD_FIELDS)
        _write_csv(self.output_dir / "feature_tracks.csv", rows, SAD_FIELDS)
        self.write_tracking_csv()
        self.write_aia_track_map()
        return rows

    def _default_feature_point(self, frame_index: int) -> np.ndarray:
        frame_rows = [row for row in self.feature_tracks or self.sad_tracks if int(row.get("frame_index", -1)) == int(frame_index)]
        if frame_rows:
            row = frame_rows[-1]
            return np.array([float(row["x_pix"]), float(row["y_pix"])], dtype=float)
        if self.feature_tracks or self.sad_tracks:
            row = sorted(self.feature_tracks or self.sad_tracks, key=lambda item: abs(int(item.get("frame_index", frame_index)) - int(frame_index)))[0]
            world = np.array([[float(row["x_arcsec"]), float(row["y_arcsec"])]], dtype=float)
            return self.aia.world_to_pixel(frame_index, world)[0]
        if self.roi_world:
            world = np.asarray(self.roi_world, dtype=float)
            return self.aia.world_to_pixel(frame_index, np.nanmean(world, axis=0).reshape(1, 2))[0]
        raise ValueError("Feature step tracking needs an existing feature point or ROI.")

    def track_feature_step(
        self,
        source_id: str,
        frame_index: int,
        direction: int,
        point: list[float] | None = None,
        patch_radius: int = 6,
        search_radius: int = 18,
    ) -> dict[str, object]:
        if source_id not in {self.context_source_id, "aia", "context"}:
            raise ValueError("Step tracking is currently available for the context image source.")
        current_idx = int(np.clip(frame_index, 0, self.aia.nt - 1))
        target_idx = int(np.clip(current_idx + (1 if direction >= 0 else -1), 0, self.aia.nt - 1))
        if target_idx == current_idx:
            raise ValueError("No adjacent frame is available in that direction.")
        seed = np.asarray(point, dtype=float) if point is not None else self._default_feature_point(current_idx)
        if seed.shape != (2,) or not np.isfinite(seed).all():
            raise ValueError("Feature point must be an x/y pixel pair.")

        current = np.asarray(self.aia.frame(current_idx), dtype=float)
        target = np.asarray(self.aia.frame(target_idx), dtype=float)
        ny, nx = current.shape
        px = int(round(seed[0]))
        py = int(round(seed[1]))
        pr = max(2, int(patch_radius))
        sr = max(pr + 2, int(search_radius))
        x0 = max(0, px - pr)
        x1 = min(nx, px + pr + 1)
        y0 = max(0, py - pr)
        y1 = min(ny, py + pr + 1)
        patch = current[y0:y1, x0:x1]
        if patch.size < 9 or not np.isfinite(patch).any():
            raise ValueError("Current feature patch has no usable signal.")
        sx0 = max(0, px - sr)
        sx1 = min(nx, px + sr + 1)
        sy0 = max(0, py - sr)
        sy1 = min(ny, py + sr + 1)
        search = target[sy0:sy1, sx0:sx1]
        if search.shape[0] < patch.shape[0] or search.shape[1] < patch.shape[1]:
            raise ValueError("Search window is smaller than the feature patch.")
        patch = np.nan_to_num(patch, nan=float(np.nanmedian(patch)))
        search = np.nan_to_num(search, nan=float(np.nanmedian(search)))
        result = match_template(search, patch, pad_input=False)
        iy, ix = np.unravel_index(int(np.nanargmax(result)), result.shape)
        score = float(result[iy, ix])
        refined_pix = np.array([sx0 + ix + (patch.shape[1] - 1) / 2.0, sy0 + iy + (patch.shape[0] - 1) / 2.0], dtype=float)
        refined_world = self.aia.pixel_to_world(target_idx, refined_pix.reshape(1, 2))[0]
        if not bool(self._inside_roi(refined_world.reshape(1, 2))[0]):
            raise ValueError("Tracked feature moved outside the ROI.")
        row = {
            "track_id": 1,
            "source_id": self.context_source_id,
            "time_utc": self.aia.times[target_idx].isot,
            "time_mjd": float(self.aia.times[target_idx].mjd),
            "frame_index": target_idx,
            "x_arcsec": float(refined_world[0]),
            "y_arcsec": float(refined_world[1]),
            "x_pix": float(refined_pix[0]),
            "y_pix": float(refined_pix[1]),
            "quality": "template_match",
            "score": score,
            "roi_id": "roi-1" if self.roi_world else "",
        }
        self.feature_tracks.append(row)
        sad_row = {
            "sad_id": row["track_id"],
            "time_utc": row["time_utc"],
            "time_mjd": row["time_mjd"],
            "frame_index": row["frame_index"],
            "x_arcsec": row["x_arcsec"],
            "y_arcsec": row["y_arcsec"],
            "x_pix": row["x_pix"],
            "y_pix": row["y_pix"],
            "vx_arcsec_s": "",
            "vy_arcsec_s": "",
            "speed_km_s": "",
            "quality": row["quality"],
        }
        self.sad_tracks.append(sad_row)
        _write_csv(self.output_dir / "feature_tracks.csv", self.feature_tracks, FEATURE_FIELDS)
        _write_csv(self.output_dir / "sad_tracks.csv", self.sad_tracks, SAD_FIELDS)
        self.write_aia_track_map()
        return row

    def delete_track_point(self, row_index: int) -> list[dict[str, object]]:
        idx = int(row_index)
        if idx < 0 or idx >= len(self.sad_tracks):
            raise ValueError("Track point index is out of range.")
        self.sad_tracks = [row for index, row in enumerate(self.sad_tracks) if index != idx]
        self.feature_tracks = list(self.sad_tracks)
        _write_csv(self.output_dir / "sad_tracks.csv", self.sad_tracks, SAD_FIELDS)
        _write_csv(self.output_dir / "feature_tracks.csv", self.sad_tracks, SAD_FIELDS)
        self.write_aia_track_map()
        return self.sad_tracks

    def extract_eovsa_sources(
        self,
        x_offset: float,
        y_offset: float,
        diff_seconds: float,
        start_index: int | None,
        end_index: int | None,
        stride: int = 4,
        min_snr: float = 5.0,
        use_running_diff: bool = True,
        difference_operation: str | None = None,
        difference_reference: str = "previous",
        mean_start_mjd: float | None = None,
        mean_end_mjd: float | None = None,
        start_mjd: float | None = None,
        end_mjd: float | None = None,
        on_progress: object = None,
    ) -> list[dict[str, object]]:
        """Extract radio sources inside a legacy or source-native time range.

        :param start_index: Legacy inclusive start index on the AIA axis.
        :type start_index: int or None
        :param end_index: Legacy inclusive end index on the AIA axis.
        :type end_index: int or None
        :param start_mjd: Canonical inclusive lower time bound in MJD.
        :type start_mjd: float or None
        :param end_mjd: Canonical inclusive upper time bound in MJD.
        :type end_mjd: float or None
        :returns: Accepted radio-source measurements in the requested range.
        :rtype: list[dict[str, object]]
        """
        native_bounds: tuple[int, int] | None = None
        legacy_bounds: tuple[float, float] | None = None
        if start_mjd is not None and end_mjd is not None:
            native_start = resolve_time_index(self.eovsa.times, start_mjd, "next")
            native_end = resolve_time_index(self.eovsa.times, end_mjd, "previous")
            if native_start is None or native_end is None or native_start[0] > native_end[0]:
                raise ValueError("The MJD range contains no radio samples")
            native_bounds = (native_start[0], native_end[0])
        else:
            if start_index is None or end_index is None:
                raise ValueError("Legacy extraction requires startIndex and endIndex")
            legacy_bounds = (
                float(self.aia.times[int(start_index)].mjd),
                float(self.aia.times[int(end_index)].mjd),
            )
        rows: list[dict[str, object]] = []
        roi = self._roi_path()
        stride_value = max(1, int(stride))
        total_frames = sum(
            1
            for eidx, time_mjd in enumerate(self.eovsa.times.mjd)
            if eidx % stride_value == 0 and (
                native_bounds is not None
                and native_bounds[0] <= eidx <= native_bounds[1]
                or native_bounds is None
                and legacy_bounds is not None
                and legacy_bounds[0] <= time_mjd <= legacy_bounds[1]
            )
        )
        processed_frames = 0
        if callable(on_progress):
            on_progress(processed_frames, total_frames)
        for eidx, time_mjd in enumerate(self.eovsa.times.mjd):
            if native_bounds is not None:
                outside_range = eidx < native_bounds[0] or eidx > native_bounds[1]
            else:
                assert legacy_bounds is not None
                outside_range = time_mjd < legacy_bounds[0] or time_mjd > legacy_bounds[1]
            if outside_range or eidx % stride_value != 0:
                continue
            channel_mask = _channel_mask_for_session(self)
            for fidx in range(self.eovsa.nfreq):
                if channel_mask[fidx]:
                    continue
                if difference_operation is not None:
                    data = self.eovsa.operation_data(eidx, diff_seconds, difference_operation, difference_reference, mean_start_mjd, mean_end_mjd)[fidx]
                else:
                    data = self.eovsa.diff_frame(eidx, fidx, diff_seconds) if use_running_diff else self.eovsa.frame(eidx, fidx)
                offset_getter = getattr(self, "channel_offset", None)
                if callable(offset_getter):
                    channel_dx, channel_dy = offset_getter(fidx)
                else:
                    table = _normalize_channel_offsets(getattr(self, "channel_offsets", None), self.eovsa.nfreq)
                    channel_dx, channel_dy = table["dx"][fidx], table["dy"][fidx]
                effective_x_offset = float(x_offset) + channel_dx
                effective_y_offset = float(y_offset) + channel_dy
                ny, nx = data.shape
                yy, xx = np.mgrid[0:ny, 0:nx]
                pix = np.column_stack([xx.ravel(), yy.ravel()])
                world = self.eovsa.pixel_to_world(data, pix, effective_x_offset, effective_y_offset)
                if roi is not None:
                    roi_mask = roi.contains_points(world).reshape(ny, nx)
                else:
                    roi_mask = np.ones((ny, nx), dtype=bool)
                if not np.any(roi_mask):
                    continue
                rms_region = data[: min(20, ny), :]
                rms = float(np.nanstd(rms_region[np.isfinite(rms_region)]))
                if not np.isfinite(rms) or rms <= 0:
                    rms = float(np.nanstd(data[np.isfinite(data)]))
                masked = np.where(roi_mask, data, np.nan)
                if not np.isfinite(masked).any():
                    continue
                peak = float(np.nanmax(masked))
                minval = float(np.nanmin(masked))
                snr = peak / rms if rms > 0 else np.nan
                accepted = bool(np.isfinite(snr) and snr >= min_snr and peak > 2.0 * abs(minval))
                if not accepted:
                    continue
                y_peak, x_peak = np.unravel_index(int(np.nanargmax(masked)), masked.shape)
                peak_world = self.eovsa.pixel_to_world(data, np.array([[x_peak, y_peak]], dtype=float), effective_x_offset, effective_y_offset)[0]
                source_cut = roi_mask & np.isfinite(data) & (data >= max(np.nanpercentile(masked, 98), peak * 0.55))
                if np.any(source_cut):
                    weights = np.clip(data[source_cut], 0.0, None)
                    coords = np.column_stack(np.where(source_cut))
                    if np.sum(weights) > 0:
                        cy = float(np.average(coords[:, 0], weights=weights))
                        cx = float(np.average(coords[:, 1], weights=weights))
                    else:
                        cy, cx = float(y_peak), float(x_peak)
                else:
                    cy, cx = float(y_peak), float(x_peak)
                centroid_world = self.eovsa.pixel_to_world(data, np.array([[cx, cy]], dtype=float), effective_x_offset, effective_y_offset)[0]
                peak_display_pix = self.eovsa.world_to_pixel(data, peak_world.reshape(1, 2), 0.0, 0.0)[0]
                centroid_display_pix = self.eovsa.world_to_pixel(data, centroid_world.reshape(1, 2), 0.0, 0.0)[0]
                rows.append({
                    "time_utc": self.eovsa.times[eidx].isot,
                    "time_mjd": float(time_mjd),
                    "eovsa_index": eidx,
                    "spw_index": fidx,
                    "freq_ghz": float(self.eovsa.freqs_hz[fidx] / 1e9),
                    "x_peak_arcsec": float(peak_world[0]),
                    "y_peak_arcsec": float(peak_world[1]),
                    "x_centroid_arcsec": float(centroid_world[0]),
                    "y_centroid_arcsec": float(centroid_world[1]),
                    "x_peak_pix": float(x_peak),
                    "y_peak_pix": float(y_peak),
                    "x_centroid_pix": float(cx),
                    "y_centroid_pix": float(cy),
                    "x_peak_display_pix": float(peak_display_pix[0]),
                    "y_peak_display_pix": float(peak_display_pix[1]),
                    "x_centroid_display_pix": float(centroid_display_pix[0]),
                    "y_centroid_display_pix": float(centroid_display_pix[1]),
                    "peak_tb": peak,
                    "snr": float(snr),
                    "accepted": True,
                    "x_offset_arcsec": effective_x_offset,
                    "y_offset_arcsec": effective_y_offset,
                })
            processed_frames += 1
            if callable(on_progress):
                on_progress(processed_frames, total_frames)
        self.eovsa_sources = rows
        _write_csv(self.output_dir / "eovsa_sources.csv", rows, EOVSA_FIELDS)
        _write_csv(self.output_dir / "radio_sources.csv", rows, EOVSA_FIELDS)
        self.write_eovsa_source_map()
        return rows

    def write_aia_track_map(self) -> Path:
        path = self.output_dir / "aia_sad_track_map.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        mid = int(np.nanmedian([row["frame_index"] for row in self.sad_tracks])) if self.sad_tracks else self.aia.nt // 2
        extent = self.aia.corners_arcsec()
        ax.imshow(
            self.aia.frame(mid),
            origin="lower",
            extent=[extent["xMin"], extent["xMax"], extent["yMin"], extent["yMax"]],
            cmap="gray",
            vmin=0.5,
            vmax=1.5,
        )
        if self.roi_world:
            roi = np.asarray(self.roi_world)
            ax.plot(np.r_[roi[:, 0], roi[0, 0]], np.r_[roi[:, 1], roi[0, 1]], color="white", lw=1.0)
        if self.sad_tracks:
            df = pd.DataFrame(self.sad_tracks)
            cvals, formatter = _time_color_values(df["time_mjd"].tolist())
            norm = plt.Normalize(cvals.min(), cvals.max())
            cmap = colormaps["turbo"]
            for sad_id, grp in df.groupby("sad_id"):
                grp = grp.sort_values("time_mjd")
                dates, _ = _time_color_values(grp["time_mjd"].tolist())
                colors = cmap(norm(dates))
                ax.scatter(grp["x_arcsec"], grp["y_arcsec"], c=colors, s=34, edgecolors="black", linewidths=0.3, label=f"SAD {sad_id}")
                xs = grp["x_arcsec"].to_numpy()
                ys = grp["y_arcsec"].to_numpy()
                for idx in range(len(xs) - 1):
                    ax.annotate("", xy=(xs[idx + 1], ys[idx + 1]), xytext=(xs[idx], ys[idx]), arrowprops=dict(arrowstyle="->", color=colors[idx], lw=1.2))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, ax=ax, label="Time [UT]")
            cbar.ax.yaxis.set_major_formatter(formatter)
            ax.legend(loc="upper right", framealpha=0.45, fontsize=8)
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        ax.set_title("AIA 131 Å SAD Tracks")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def write_eovsa_source_map(self) -> Path:
        path = self.output_dir / "eovsa_source_time_map.png"
        fig, ax = plt.subplots(figsize=(7, 6))
        extent = self.aia.corners_arcsec()
        ax.imshow(
            self.aia.frame(self.aia.nt // 2),
            origin="lower",
            extent=[extent["xMin"], extent["xMax"], extent["yMin"], extent["yMax"]],
            cmap="gray",
            vmin=0.5,
            vmax=1.5,
        )
        if self.roi_world:
            roi = np.asarray(self.roi_world)
            ax.plot(np.r_[roi[:, 0], roi[0, 0]], np.r_[roi[:, 1], roi[0, 1]], color="white", lw=1.0)
        if self.eovsa_sources:
            df = pd.DataFrame(self.eovsa_sources)
            dates, formatter = _time_color_values(df["time_mjd"].tolist())
            scatter = ax.scatter(
                df["x_centroid_arcsec"],
                df["y_centroid_arcsec"],
                c=dates,
                s=np.clip((df["freq_ghz"].to_numpy() / df["freq_ghz"].max()) * 34, 10, 34),
                cmap="turbo",
                alpha=0.72,
                edgecolors="black",
                linewidths=0.2,
            )
            cbar = fig.colorbar(scatter, ax=ax, label="Time [UT]")
            cbar.ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel("Solar-X [arcsec]")
        ax.set_ylabel("Solar-Y [arcsec]")
        ax.set_title("EOVSA Source Centroids")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def export_path(self, name: str) -> Path:
        allowed = {
            "feature_tracks.csv",
            "radio_sources.csv",
            "sad_tracks.csv",
            "eovsa_sources.csv",
            "aia_sad_track_map.png",
            "eovsa_source_time_map.png",
        }
        if name not in allowed:
            raise ValueError(name)
        return self.output_dir / name
