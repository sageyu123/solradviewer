"""Synthetic tests for generalized SAD feature tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.time import Time

from solradviewer.backend.data import (
    SolRadSession,
    constrained_ncc_segment,
    track_ncc_pass,
    tracking_csv_rows,
)


def _moving_blob_sequence(
    count: int = 16,
    velocity: tuple[float, float] = (0.7, -0.35),
    disappear_at: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    yy, xx = np.indices((72, 72), dtype=float)
    centers = np.asarray([[22.0 + velocity[0] * index, 38.0 + velocity[1] * index] for index in range(count)])
    frames = []
    for index, (x, y) in enumerate(centers):
        if disappear_at is not None and index >= disappear_at:
            frames.append(np.zeros_like(xx))
            continue
        blob = np.exp(-((xx - x) ** 2 / 7.0 + (yy - y) ** 2 / 10.0))
        shoulder = 0.22 * np.exp(-((xx - x - 3.2) ** 2 / 3.0 + (yy - y + 1.7) ** 2 / 5.0))
        frames.append(blob + shoulder)
    return frames, centers


def test_ncc_tracks_subpixel_trajectory_within_half_pixel() -> None:
    frames, centers = _moving_blob_sequence()
    start = {
        "frameIndex": 0,
        "mjd": 60000.0,
        "x": float(centers[0, 0]),
        "y": float(centers[0, 1]),
        "confidence": 1.0,
        "isAnchor": True,
    }
    points, state = track_ncc_pass(
        frames.__getitem__, start, len(frames) - 1, 1,
        patch_radius=5, search_radius=10, confidence_threshold=0.5,
        times_mjd=60000.0 + np.arange(len(frames)) / 86400.0,
    )
    tracked = np.asarray([[point["x"], point["y"]] for point in points])
    error = np.linalg.norm(tracked - centers[:len(points)], axis=1)
    print(f"trajectory_max_error_px={float(np.max(error)):.6f}")
    assert state == "active"
    assert len(points) == len(frames)
    assert float(np.max(error)) <= 0.5


def test_ncc_stops_after_two_low_confidence_frames() -> None:
    frames, centers = _moving_blob_sequence(count=14, disappear_at=8)
    start = {
        "frameIndex": 0,
        "x": float(centers[0, 0]),
        "y": float(centers[0, 1]),
        "confidence": 1.0,
        "isAnchor": True,
    }
    points, state = track_ncc_pass(
        frames.__getitem__, start, len(frames) - 1, 1,
        patch_radius=5, search_radius=10, confidence_threshold=0.5,
    )
    assert state == "stopped-low-confidence"
    assert int(points[-1]["frameIndex"]) == 7


def test_constrained_retrack_passes_through_anchors_exactly() -> None:
    frames, centers = _moving_blob_sequence(count=13)
    earlier = {"frameIndex": 2, "mjd": 60000.0 + 2 / 86400.0, "x": centers[2, 0], "y": centers[2, 1]}
    later = {"frameIndex": 11, "mjd": 60000.0 + 11 / 86400.0, "x": centers[11, 0] + 0.4, "y": centers[11, 1] - 0.3}
    points = constrained_ncc_segment(
        frames.__getitem__, earlier, later,
        patch_radius=5, search_radius=10, confidence_threshold=0.5,
        times_mjd=60000.0 + np.arange(len(frames)) / 86400.0,
    )
    first_error = float(np.hypot(points[0]["x"] - earlier["x"], points[0]["y"] - earlier["y"]))
    last_error = float(np.hypot(points[-1]["x"] - later["x"], points[-1]["y"] - later["y"]))
    print(f"anchor_max_error_px={max(first_error, last_error):.6f}")
    assert first_error == 0.0
    assert last_error == 0.0
    assert points[0]["isAnchor"] is True
    assert points[-1]["isAnchor"] is True


def test_tracking_csv_velocity_matches_analytic_motion() -> None:
    count = 11
    cadence_seconds = 2.0
    times = 60000.0 + np.arange(count) * cadence_seconds / 86400.0
    x = 10.0 + np.arange(count) * 0.6
    y = 30.0 - np.arange(count) * 0.4
    track = {
        "id": "track-test",
        "label": "Synthetic",
        "points": [
            {"frameIndex": index, "mjd": times[index], "x": x[index], "y": y[index], "confidence": 0.9, "isAnchor": index == 0}
            for index in range(count)
        ],
    }

    def pixel_to_world(_frame_index: int, points: np.ndarray) -> np.ndarray:
        return np.column_stack([2.0 * points[:, 0] + 5.0, -3.0 * points[:, 1] + 7.0])

    rows = tracking_csv_rows([track], times, pixel_to_world)
    expected = np.asarray([0.6, 0.6])
    measured = np.asarray([[rows[index]["vx_arcsec_s"], rows[index]["vy_arcsec_s"]] for index in range(2, count - 2)], dtype=float)
    velocity_error = float(np.max(np.abs(measured - expected)))
    print(f"velocity_max_error_arcsec_s={velocity_error:.9f}")
    assert velocity_error < 1e-5
    assert np.isnan(float(rows[0]["speed_arcsec_s"]))
    assert np.isnan(float(rows[1]["speed_arcsec_s"]))
    assert np.isnan(float(rows[-1]["speed_arcsec_s"]))
    assert np.isnan(float(rows[-2]["speed_arcsec_s"]))
    assert np.isclose(float(rows[5]["speed_km_s"]), np.hypot(*expected) * 725.0)


class _SyntheticContextSequence:
    """Small in-memory context sequence for source-bound session tests."""

    def __init__(self, frames: list[np.ndarray], times: Time) -> None:
        self.frames = frames
        self.times = times
        self.nt = len(frames)
        self.shape = frames[0].shape

    def frame(self, index: int, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        return self.frames[index]

    def pixel_to_world(self, _index: int, points: np.ndarray) -> np.ndarray:
        return np.column_stack([2.0 * points[:, 0] - 40.0, 2.0 * points[:, 1] - 60.0])

    def world_to_pixel(self, _index: int, points: np.ndarray) -> np.ndarray:
        return np.column_stack([(points[:, 0] + 40.0) / 2.0, (points[:, 1] + 60.0) / 2.0])


class _SyntheticRadioSequence:
    """Small in-memory radio sequence with the production frame API."""

    def __init__(self, frames: list[np.ndarray], times: Time) -> None:
        self.frames = frames
        self.times = times
        self.files = [Path(f"radio-{index:03d}.fits") for index in range(len(frames))]
        self.shape = frames[0].shape
        self.nfreq = 1

    def frame_for_aia_time(self, _mjd: float, _freq_index: int, _diff_seconds: float, **kwargs: object) -> np.ndarray:
        return self.frames[int(kwargs["eovsa_index"])]

    def pixel_to_world(self, _data: np.ndarray, points: np.ndarray, x_offset: float, y_offset: float) -> np.ndarray:
        return np.column_stack([3.0 * points[:, 0] + 100.0 + x_offset, -2.0 * points[:, 1] + 80.0 + y_offset])

    def world_to_pixel(self, _data: np.ndarray, points: np.ndarray, x_offset: float, y_offset: float) -> np.ndarray:
        return np.column_stack([(points[:, 0] - 100.0 - x_offset) / 3.0, -(points[:, 1] - 80.0 - y_offset) / 2.0])


def test_session_tracks_context_and_radio_sources_independently(tmp_path: Path) -> None:
    """Radio tracking uses its native frames and affine without mutating AIA tracks."""
    aia_frames, aia_centers = _moving_blob_sequence(count=8, velocity=(0.3, -0.1))
    radio_frames, radio_centers = _moving_blob_sequence(count=12, velocity=(0.45, 0.2))
    aia_times = Time(60000.0 + np.arange(len(aia_frames)) * 12.0 / 86400.0, format="mjd")
    radio_times = Time(60000.0 + np.arange(len(radio_frames)) * 4.0 / 86400.0, format="mjd")
    session = SolRadSession(
        session_id="mixed-source-test",
        aia=_SyntheticContextSequence(aia_frames, aia_times),  # type: ignore[arg-type]
        eovsa=_SyntheticRadioSequence(radio_frames, radio_times),  # type: ignore[arg-type]
        spectrogram=None,  # type: ignore[arg-type]
        seed_path=tmp_path / "unused-seeds.pickle",
        output_dir=tmp_path,
        context_source_id="aia-131",
        radio_source_id="eovsa-1s",
    )
    session.correlation_target = [[180.0, -20.0], [210.0, -20.0], [210.0, 20.0], [180.0, 20.0]]
    aia_track = session.add_track_seed(
        "aia-131", 2, float(aia_centers[2, 0]), float(aia_centers[2, 1]), label="AIA trajectory"
    )
    aia_before = len(aia_track["points"])
    radio_track = session.add_track_seed(
        "eovsa-1s", 0, float(radio_centers[0, 0]), float(radio_centers[0, 1]), label="Radio trajectory"
    )
    tracks = session.auto_track(
        [str(radio_track["id"])],
        {"sourceId": "eovsa-1s", "freqIndex": 0, "differenceOperation": "none", "differenceMode": "none"},
        "forward",
        0,
        0,
        10,
        patch_radius=5,
        search_radius=10,
        confidence_threshold=0.5,
    )
    resolved_aia = next(track for track in tracks if track["id"] == aia_track["id"])
    resolved_radio = next(track for track in tracks if track["id"] == radio_track["id"])
    radio_confidence = np.asarray([point["confidence"] for point in resolved_radio["points"]], dtype=float)
    assert resolved_radio["sourceId"] == "eovsa-1s"
    assert len(resolved_radio["points"]) == 11
    assert len(resolved_aia["points"]) == aia_before
    assert [point["frameIndex"] for point in resolved_radio["points"]] == list(range(11))
    rows = session.write_tracking_csv()
    radio_rows = [row for row in rows if row["track_id"] == radio_track["id"]]
    assert len(radio_rows) == 11
    assert all(row["source_id"] == "eovsa-1s" for row in radio_rows)
    assert np.isclose(float(radio_rows[0]["x_arcsec"]), 3.0 * radio_centers[0, 0] + 100.0)
    radio_distances = np.asarray([row["dist_to_target_arcsec"] for row in radio_rows], dtype=float)
    assert radio_distances[-1] < radio_distances[0]
    print(
        "mixed_source_radio_points=11 "
        f"confidence_range={float(np.min(radio_confidence)):.6f}..{float(np.max(radio_confidence)):.6f} "
        f"aia_points_before_after={aia_before}/{len(resolved_aia['points'])} "
        f"radio_distance_arcsec={radio_distances[0]:.6f}..{radio_distances[-1]:.6f}"
    )
