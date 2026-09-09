"""Synthetic time-distance slit extraction tests."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from astropy.time import Time
from fastapi.testclient import TestClient

from solradviewer.backend import app as api
from solradviewer.backend import data
from solradviewer.backend.data import (
    RenderDiskCache,
    SolRadSession,
    extract_time_distance_map,
    fan_family_curves,
)


class _SyntheticSequence:
    """Small identity-WCS image sequence for slit tests."""

    def __init__(self, frames: np.ndarray) -> None:
        self.frames = np.asarray(frames, dtype=np.float32)
        self.shape = tuple(int(value) for value in self.frames.shape[1:])
        self.nt = int(self.frames.shape[0])
        self.times = Time(60000.0 + np.arange(self.nt) * 2.0 / 86400.0, format="mjd")

    def frame(self, index: int, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        return self.frames[index]

    def pixel_to_world(self, _index: int, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float)

    def world_to_pixel(self, _index: int, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float)


class _SyntheticRadioSequence:
    """Small multi-frequency identity-WCS radio sequence."""

    def __init__(self, frames: np.ndarray) -> None:
        self.frames = np.asarray(frames, dtype=np.float32)
        self.nt, self.nfreq, ny, nx = self.frames.shape
        self.shape = (ny, nx)
        self.times = Time(60000.0 + np.arange(self.nt) / 86400.0, format="mjd")
        self.freqs_hz = np.linspace(3.0e9, 9.0e9, self.nfreq)

    def frame_for_aia_time(self, _mjd: float, freq_index: int, *args: object, **kwargs: object) -> np.ndarray:
        del args
        return self.frames[int(kwargs.get("eovsa_index", 0)), freq_index]

    def world_to_pixel(
        self, _data: np.ndarray, points: np.ndarray, x_offset: float, y_offset: float
    ) -> np.ndarray:
        return np.asarray(points, dtype=float) - np.asarray([x_offset, y_offset], dtype=float)

    def pixel_to_world(
        self, _data: np.ndarray, points: np.ndarray, _x_offset: float, _y_offset: float
    ) -> np.ndarray:
        return np.asarray(points, dtype=float)


def _session(frames: np.ndarray, output_dir: Path) -> SolRadSession:
    context = _SyntheticSequence(frames)
    radio = SimpleNamespace(nfreq=0, times=Time([], format="mjd"), shape=(1, 1))
    return SolRadSession(
        session_id="slit-synthetic",
        aia=context,  # type: ignore[arg-type]
        eovsa=radio,  # type: ignore[arg-type]
        spectrogram=None,  # type: ignore[arg-type]
        seed_path=output_dir / "unused.pickle",
        output_dir=output_dir,
    )


def test_slit_endpoint_recovers_horizontal_gradient_and_cached_export(tmp_path: Path) -> None:
    """A horizontal slit returns the analytic gradient at every native time."""
    yy, xx = np.indices((40, 64), dtype=float)
    frames = np.stack([xx + 10.0 * index for index in range(6)])
    session = _session(frames, tmp_path)
    cache = RenderDiskCache(tmp_path / "render-cache", 32 * 1024**2)
    api.SESSIONS[session.session_id] = session
    request = {
        "slitId": "gradient",
        "name": "Gradient slit",
        "sourceId": "context",
        "curveArcsec": [[5.0, 20.0], [58.0, 20.0]],
        "width": 3,
        "layerParams": {"sourceId": "context", "differenceOperation": "none"},
    }
    try:
        with patch.object(data, "RENDER_DISK_CACHE", cache):
            client = TestClient(api.app)
            first = client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
            cached = client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
            reversed_request = {
                **request,
                "slitId": "gradient-reversed",
                "curveArcsec": list(reversed(request["curveArcsec"])),
            }
            reversed_cached = client.post(
                f"/api/sessions/{session.session_id}/slits/extract", json=reversed_request
            )
            exported_before = client.get(
                f"/api/sessions/{session.session_id}/slits/gradient/export.npz?shiftSeconds=60"
            )
            reversed_response = client.post(
                f"/api/sessions/{session.session_id}/slits/gradient/reverse"
            )
            exported_after = client.get(
                f"/api/sessions/{session.session_id}/slits/gradient/export.npz?shiftSeconds=60"
            )
    finally:
        api.SESSIONS.pop(session.session_id, None)
    assert first.status_code == 200
    assert cached.status_code == 200
    assert reversed_cached.status_code == 200
    assert exported_before.status_code == 200
    assert reversed_response.status_code == 200
    assert exported_after.status_code == 200
    payload = first.json()
    intensity = np.asarray(payload["intensity"], dtype=float)
    expected = np.arange(5.0, 59.0)
    gradient_error = float(np.max(np.abs(intensity[:, 0] - expected)))
    assert intensity.shape == (54, 6)
    assert gradient_error < 1e-6
    assert first.json()["cacheHit"] is False
    assert cached.json()["cacheHit"] is True
    assert reversed_cached.json()["cacheHit"] is True
    np.testing.assert_allclose(
        np.asarray(reversed_cached.json()["intensity"], dtype=float), intensity[::-1]
    )
    with np.load(BytesIO(exported_before.content), allow_pickle=False) as before_archive:
        before = {key: np.asarray(before_archive[key]) for key in before_archive.files}
    with np.load(BytesIO(exported_after.content), allow_pickle=False) as archive:
        assert set(archive.files) == {
            "intensity", "distance_arcsec", "time_mjd", "applied_shift_seconds",
            "curve_vertices_arcsec", "layer_params_snapshot", "source_id", "freq_ghz", "layout",
        }
        assert str(archive["layout"]) == "distance_time"
        assert float(archive["applied_shift_seconds"]) == 60.0
        np.testing.assert_allclose(archive["intensity"], before["intensity"][::-1])
        np.testing.assert_allclose(
            archive["distance_arcsec"],
            float(before["distance_arcsec"][-1]) - before["distance_arcsec"][::-1],
        )
        np.testing.assert_allclose(
            archive["curve_vertices_arcsec"], before["curve_vertices_arcsec"][::-1]
        )
    print(
        f"gradient_profile_max_error={gradient_error:.9f} "
        f"shape={intensity.shape} cached={cached.json()['cacheHit']} "
        f"reverse_cache_hit={reversed_cached.json()['cacheHit']} "
        f"row_mean_before={float(np.mean(before['intensity'][0])):.6f} "
        f"row_mean_after={float(np.mean(before['intensity'][-1])):.6f}"
    )


def test_perpendicular_width_averaging_reduces_noise_variance() -> None:
    """A five-pixel perpendicular mean suppresses independent row noise."""
    rng = np.random.default_rng(20250328)
    yy, xx = np.indices((48, 72), dtype=float)
    frames = np.stack([xx + rng.normal(0.0, 4.0, xx.shape) for _ in range(40)])
    times = 60000.0 + np.arange(frames.shape[0]) / 86400.0
    curve = np.asarray([[8.0, 24.0], [63.0, 24.0]])
    project = lambda _index, points: points
    narrow = extract_time_distance_map(frames.__getitem__, times, curve, project, width=1)
    wide = extract_time_distance_map(frames.__getitem__, times, curve, project, width=5)
    truth = np.arange(8.0, 64.0)[:, None]
    variance_width_1 = float(np.var(np.asarray(narrow["intensity"]) - truth))
    variance_width_5 = float(np.var(np.asarray(wide["intensity"]) - truth))
    assert variance_width_5 < variance_width_1 * 0.35
    print(
        f"slit_noise_variance_width1={variance_width_1:.6f} "
        f"width5={variance_width_5:.6f} ratio={variance_width_5 / variance_width_1:.6f}"
    )


def test_fan_parallel_geometry_and_opposite_direction_alignment() -> None:
    """Parallel boundaries produce exact spacing and reverse an opposite stroke."""
    boundary_a = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    boundary_b_opposite = np.asarray([[10.0, 8.0], [0.0, 8.0]])
    curves, reversed_b = fan_family_curves(boundary_a, boundary_b_opposite, 3)
    expected_y = np.linspace(0.0, 8.0, 5)
    deviation = max(
        float(np.max(np.abs(curve[:, 1] - expected_y[index])))
        for index, curve in enumerate(curves)
    )
    spacing_deviation = float(np.max(np.abs(np.diff([curve[0, 1] for curve in curves]) - 2.0)))
    assert reversed_b is True
    assert deviation < 1e-12
    assert spacing_deviation < 1e-12
    np.testing.assert_allclose(curves[-1][0], [0.0, 8.0])
    np.testing.assert_allclose(curves[-1][-1], [10.0, 8.0])
    print(
        f"fan_parallel_max_deviation={deviation:.12g} "
        f"spacing_max_deviation={spacing_deviation:.12g} reversed_boundary_b={reversed_b}"
    )


def test_radio_multichannel_slit_cache_keys_and_npz_layout(tmp_path: Path) -> None:
    """Each radio frequency has a distinct cache identity and ordered NPZ axis."""
    yy, xx = np.indices((24, 32), dtype=float)
    frames = np.stack([
        np.stack([xx + 100.0 * channel + time for channel in range(3)])
        for time in range(5)
    ])
    context = _SyntheticSequence(frames[:, 0])
    radio = _SyntheticRadioSequence(frames)
    session = SolRadSession(
        session_id="slit-radio-multi",
        aia=context,  # type: ignore[arg-type]
        eovsa=radio,  # type: ignore[arg-type]
        spectrogram=None,  # type: ignore[arg-type]
        seed_path=tmp_path / "unused.pickle",
        output_dir=tmp_path,
    )
    cache = RenderDiskCache(tmp_path / "render-cache", 64 * 1024**2)
    api.SESSIONS[session.session_id] = session
    request = {
        "slitId": "radio-three",
        "name": "Radio fan slit",
        "sourceId": "radio",
        "curveArcsec": [[3.0, 12.0], [27.0, 12.0]],
        "width": 3,
        "freqIndices": [0, 1, 2],
        "layerParams": {"sourceId": "radio", "freqIndex": 0, "differenceOperation": "none"},
    }
    try:
        with patch.object(data, "RENDER_DISK_CACHE", cache):
            client = TestClient(api.app)
            response = client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
            exported = client.get(f"/api/sessions/{session.session_id}/slits/radio-three/export.npz")
    finally:
        api.SESSIONS.pop(session.session_id, None)
    assert response.status_code == 200, response.text
    assert exported.status_code == 200
    payload = response.json()
    maps = [payload, *payload["additionalMaps"]]
    cache_keys = [item["cacheKey"] for item in maps]
    assert len(set(cache_keys)) == 3
    assert [item["freqIndex"] for item in maps] == [0, 1, 2]
    with np.load(BytesIO(exported.content), allow_pickle=False) as archive:
        assert archive["intensity"].shape == (3, 25, 5)
        assert archive["freq_ghz"].shape == (3,)
        assert str(archive["layout"]) == "frequency_distance_time"
        np.testing.assert_allclose(archive["freq_ghz"], [3.0, 6.0, 9.0])
    print(
        f"radio_maps={len(maps)} distinct_cache_keys={len(set(cache_keys))} "
        f"npz_shape={(3, 25, 5)} layout=frequency_distance_time"
    )


def test_radio_slit_channel_offset_shifts_profile_and_cache_revision(tmp_path: Path) -> None:
    """Radio slit WCS sampling follows channel offsets and their revision."""
    yy, xx = np.indices((24, 40), dtype=float)
    feature = np.exp(-0.5 * ((xx - 13.0) / 2.0) ** 2) * 100.0
    frames = np.stack([feature[None, :, :] + time for time in range(4)])
    context = _SyntheticSequence(frames[:, 0])
    radio = _SyntheticRadioSequence(frames)
    session = SolRadSession(
        session_id="slit-radio-offset",
        aia=context,  # type: ignore[arg-type]
        eovsa=radio,  # type: ignore[arg-type]
        spectrogram=None,  # type: ignore[arg-type]
        seed_path=tmp_path / "unused.pickle",
        output_dir=tmp_path,
    )
    cache = RenderDiskCache(tmp_path / "render-cache", 64 * 1024**2)
    api.SESSIONS[session.session_id] = session
    request = {
        "slitId": "radio-offset",
        "name": "Offset proof",
        "sourceId": "radio",
        "curveArcsec": [[2.0, 12.0], [37.0, 12.0]],
        "width": 1,
        "freqIndices": [0],
        "layerParams": {
            "sourceId": "radio", "freqIndex": 0,
            "differenceOperation": "none", "xOffsetArcsec": 0.0, "yOffsetArcsec": 0.0,
        },
    }
    try:
        with patch.object(data, "RENDER_DISK_CACHE", cache):
            client = TestClient(api.app)
            zero = client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
            session.set_channel_offsets({"dx": [10.0], "dy": [0.0], "masked": [False]})
            shifted = client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
            session.set_channel_offsets({"dx": [10.0], "dy": [0.0], "masked": [False]})
            revised = client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
    finally:
        api.SESSIONS.pop(session.session_id, None)
    assert zero.status_code == shifted.status_code == revised.status_code == 200
    before = np.nan_to_num(np.asarray(zero.json()["intensity"], dtype=float)[:, 0])
    after = np.nan_to_num(np.asarray(shifted.json()["intensity"], dtype=float)[:, 0])
    lag_pixels = int(np.argmax(np.correlate(after - after.mean(), before - before.mean(), mode="full")) - (before.size - 1))
    spacing_arcsec = float(np.median(np.diff(np.asarray(zero.json()["distanceArcsec"], dtype=float))))
    lag_arcsec = lag_pixels * spacing_arcsec
    keys = [zero.json()["cacheKey"], shifted.json()["cacheKey"], revised.json()["cacheKey"]]
    assert abs(lag_arcsec - 10.0) <= 1.0
    assert len(set(keys)) == 3
    assert session.channel_offsets_version == 2
    print(
        f"radio_offset_lag_pixels={lag_pixels} lag_arcsec={lag_arcsec:.6f} "
        f"direction=positive distinct_revision_keys={len(set(keys))} offsets_rev={session.channel_offsets_version}"
    )


def test_slit_batch_endpoint_matches_sequential_singles_context(tmp_path: Path) -> None:
    """Batch-extracting two context slits over one shared gradient matches per-slit extraction."""
    yy, xx = np.indices((40, 64), dtype=float)
    frames = np.stack([xx + 10.0 * index for index in range(6)])
    session = _session(frames, tmp_path)
    layer_params = {"sourceId": "context", "differenceOperation": "none"}
    batch_request = {
        "sourceId": "context",
        "layerParams": layer_params,
        "slits": [
            {
                "slitId": "batch-a",
                "name": "Batch A",
                "curveArcsec": [[5.0, 12.0], [58.0, 12.0]],
                "width": 3,
            },
            {
                "slitId": "batch-b",
                "name": "Batch B",
                "curveArcsec": [[8.0, 28.0], [50.0, 28.0]],
                "width": 5,
            },
        ],
    }
    single_requests = [
        {
            "slitId": entry["slitId"],
            "name": entry["name"],
            "sourceId": "context",
            "curveArcsec": entry["curveArcsec"],
            "width": entry["width"],
            "layerParams": layer_params,
        }
        for entry in batch_request["slits"]
    ]
    cache_batch = RenderDiskCache(tmp_path / "render-cache-batch", 32 * 1024**2)
    cache_single = RenderDiskCache(tmp_path / "render-cache-single", 32 * 1024**2)
    api.SESSIONS[session.session_id] = session
    try:
        client = TestClient(api.app)
        with patch.object(data, "RENDER_DISK_CACHE", cache_batch):
            batch_response = client.post(
                f"/api/sessions/{session.session_id}/slits/extract-batch", json=batch_request
            )
            batch_response_warm = client.post(
                f"/api/sessions/{session.session_id}/slits/extract-batch", json=batch_request
            )
        with patch.object(data, "RENDER_DISK_CACHE", cache_single):
            single_responses = [
                client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
                for request in single_requests
            ]
    finally:
        api.SESSIONS.pop(session.session_id, None)
    assert batch_response.status_code == 200, batch_response.text
    assert batch_response_warm.status_code == 200
    for response in single_responses:
        assert response.status_code == 200, response.text
    batch_payload = batch_response.json()["results"]
    warm_payload = batch_response_warm.json()["results"]
    assert set(batch_payload.keys()) == {"batch-a", "batch-b"}
    # A cold batch computes both misses; a second identical batch call hits
    # every per-slit disk-cache entry and returns immediately.
    assert batch_payload["batch-a"]["cacheHit"] is False
    assert batch_payload["batch-b"]["cacheHit"] is False
    assert warm_payload["batch-a"]["cacheHit"] is True
    assert warm_payload["batch-b"]["cacheHit"] is True
    for single_response, entry in zip(single_responses, batch_request["slits"]):
        slit_id = entry["slitId"]
        batch_intensity = np.asarray(batch_payload[slit_id]["intensity"], dtype=float)
        single_intensity = np.asarray(single_response.json()["intensity"], dtype=float)
        np.testing.assert_array_equal(batch_intensity, single_intensity)
        np.testing.assert_array_equal(
            np.asarray(batch_payload[slit_id]["distanceArcsec"], dtype=float),
            np.asarray(single_response.json()["distanceArcsec"], dtype=float),
        )
        assert batch_payload[slit_id]["cacheKey"] == single_response.json()["cacheKey"]
    print(
        f"context_batch_cache_hit={batch_payload['batch-a']['cacheHit']},{batch_payload['batch-b']['cacheHit']} "
        f"warm_cache_hit={warm_payload['batch-a']['cacheHit']},{warm_payload['batch-b']['cacheHit']}"
    )


def test_slit_batch_endpoint_matches_sequential_singles_radio(tmp_path: Path) -> None:
    """Batch-extracting two multi-channel radio slits matches per-slit extraction."""
    yy, xx = np.indices((24, 32), dtype=float)
    frames = np.stack([
        np.stack([xx + 100.0 * channel + time for channel in range(3)])
        for time in range(5)
    ])
    context = _SyntheticSequence(frames[:, 0])
    radio = _SyntheticRadioSequence(frames)
    session = SolRadSession(
        session_id="slit-batch-radio",
        aia=context,  # type: ignore[arg-type]
        eovsa=radio,  # type: ignore[arg-type]
        spectrogram=None,  # type: ignore[arg-type]
        seed_path=tmp_path / "unused.pickle",
        output_dir=tmp_path,
    )
    layer_params = {"sourceId": "radio", "freqIndex": 0, "differenceOperation": "none"}
    batch_request = {
        "sourceId": "radio",
        "layerParams": layer_params,
        "slits": [
            {
                "slitId": "radio-batch-a",
                "curveArcsec": [[3.0, 10.0], [27.0, 10.0]],
                "width": 3,
                "freqIndices": [0, 2],
            },
            {
                "slitId": "radio-batch-b",
                "curveArcsec": [[4.0, 16.0], [25.0, 16.0]],
                "width": 1,
                "freqIndices": [1, 2],
            },
        ],
    }
    single_requests = [
        {
            "slitId": entry["slitId"],
            "sourceId": "radio",
            "curveArcsec": entry["curveArcsec"],
            "width": entry["width"],
            "freqIndices": entry["freqIndices"],
            "layerParams": layer_params,
        }
        for entry in batch_request["slits"]
    ]
    cache_batch = RenderDiskCache(tmp_path / "render-cache-radio-batch", 64 * 1024**2)
    cache_single = RenderDiskCache(tmp_path / "render-cache-radio-single", 64 * 1024**2)
    api.SESSIONS[session.session_id] = session
    try:
        client = TestClient(api.app)
        with patch.object(data, "RENDER_DISK_CACHE", cache_batch):
            batch_response = client.post(
                f"/api/sessions/{session.session_id}/slits/extract-batch", json=batch_request
            )
        with patch.object(data, "RENDER_DISK_CACHE", cache_single):
            single_responses = [
                client.post(f"/api/sessions/{session.session_id}/slits/extract", json=request)
                for request in single_requests
            ]
    finally:
        api.SESSIONS.pop(session.session_id, None)
    assert batch_response.status_code == 200, batch_response.text
    for response in single_responses:
        assert response.status_code == 200, response.text
    batch_payload = batch_response.json()["results"]
    for single_response, entry in zip(single_responses, batch_request["slits"]):
        slit_id = entry["slitId"]
        batch_maps = [batch_payload[slit_id], *batch_payload[slit_id]["additionalMaps"]]
        single_maps = [single_response.json(), *single_response.json()["additionalMaps"]]
        assert [item["freqIndex"] for item in batch_maps] == [item["freqIndex"] for item in single_maps]
        for batch_map, single_map in zip(batch_maps, single_maps):
            np.testing.assert_array_equal(
                np.asarray(batch_map["intensity"], dtype=float),
                np.asarray(single_map["intensity"], dtype=float),
            )
            assert batch_map["cacheKey"] == single_map["cacheKey"]
    print(
        f"radio_batch_maps_a={len(batch_payload['radio-batch-a']['additionalMaps']) + 1} "
        f"radio_batch_maps_b={len(batch_payload['radio-batch-b']['additionalMaps']) + 1}"
    )
