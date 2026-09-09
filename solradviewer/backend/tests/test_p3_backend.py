"""Integration tests for P3 backend timeline addressing and migration."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.time import Time
from fastapi.testclient import TestClient

from solradviewer.backend import app as api
from solradviewer.backend.data import SolRadSession


class _P3Session:
    context_source_id = "context"
    radio_source_id = "radio"
    spectrogram_source_id = "spectrogram"

    def __init__(self) -> None:
        self.aia = type(
            "FakeAia",
            (),
            {"times": Time([60000.0, 60000.0 + 10.0 / 86400.0, 60000.0 + 20.0 / 86400.0], format="mjd")},
        )()
        self.eovsa = type(
            "FakeEovsa",
            (),
            {"times": Time([60000.0 + value / 86400.0 for value in (1, 3, 5, 7, 9, 11)], format="mjd")},
        )()
        self.track_calls: list[int] = []
        self.extract_calls: list[tuple[int | None, int | None, float | None, float | None]] = []
        self.roi_calls: list[tuple[str, int, int | None]] = []

    def track_feature_step(
        self,
        source_id: str,
        frame_index: int,
        direction: int,
        **_: object,
    ) -> dict[str, object]:
        self.track_calls.append(frame_index)
        target = frame_index + (1 if direction >= 0 else -1)
        return {
            "source_id": source_id,
            "frame_index": target,
            "time_mjd": float(self.aia.times[target].mjd),
        }

    def extract_eovsa_sources(
        self,
        *,
        start_index: int | None,
        end_index: int | None,
        start_mjd: float | None = None,
        end_mjd: float | None = None,
        **_: object,
    ) -> list[dict[str, object]]:
        self.extract_calls.append((start_index, end_index, start_mjd, end_mjd))
        return [{"time_mjd": start_mjd if start_mjd is not None else float(self.aia.times[start_index].mjd)}]

    def set_roi_from_pixels(
        self,
        panel: str,
        points: list[list[float]],
        time_index: int,
        freq_index: int,
        *args: object,
    ) -> dict[str, object]:
        self.roi_calls.append((panel, time_index, args[-1] if args else None))
        return {"roiWorld": points}

    def roi_pixels_for_panel(
        self,
        panel: str,
        time_index: int,
        freq_index: int,
        *args: object,
    ) -> list[list[float]]:
        self.roi_calls.append((panel, time_index, args[-1] if args else None))
        return [[1.0, 2.0]]


class _DenseRadio:
    def __init__(self) -> None:
        self.times = Time([60000.0 + value / 86400.0 for value in (1, 3, 5, 7, 9, 11)], format="mjd")
        self.nfreq = 1
        self.freqs_hz = np.array([3.0e9])

    def diff_frame(self, index: int, freq_index: int, diff_seconds: float) -> np.ndarray:
        data = np.zeros((4, 4), dtype=float)
        data[2, 2] = 100.0 + index
        return data

    def pixel_to_world(
        self,
        data: np.ndarray,
        points: np.ndarray,
        x_offset: float,
        y_offset: float,
    ) -> np.ndarray:
        return np.asarray(points, dtype=float) + np.array([x_offset, y_offset], dtype=float)

    def world_to_pixel(
        self,
        data: np.ndarray,
        points: np.ndarray,
        x_offset: float,
        y_offset: float,
    ) -> np.ndarray:
        return np.asarray(points, dtype=float)


class _DenseExtractionSession:
    def __init__(self, output_dir: Path) -> None:
        self.aia = type(
            "FakeAia",
            (),
            {"times": Time([60000.0, 60000.0 + 10.0 / 86400.0, 60000.0 + 20.0 / 86400.0], format="mjd")},
        )()
        self.eovsa = _DenseRadio()
        self.channel_offsets = {"dx": [1.25], "dy": [-0.75]}
        self.eovsa_sources: list[dict[str, object]] = []
        self.output_dir = output_dir

    def _roi_path(self) -> None:
        return None

    def write_eovsa_source_map(self) -> Path:
        return self.output_dir / "eovsa_source_time_map.png"


class _RestoreSession:
    context_source_id = "context"
    radio_source_id = "radio"
    spectrogram_source_id = "spectrogram"

    def __init__(self) -> None:
        self.aia = type(
            "FakeAia",
            (),
            {"times": Time([60000.0, 60000.0 + 12.0 / 86400.0, 60000.0 + 24.0 / 86400.0], format="mjd")},
        )()
        self.eovsa = type(
            "FakeEovsa",
            (),
            {"times": Time([60000.0 + value / 86400.0 for value in (1, 3, 5, 7, 9, 11)], format="mjd")},
        )()
        self.roi_world = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
        self.sad_tracks: list[dict[str, object]] = []
        self.feature_tracks: list[dict[str, object]] = []
        self.eovsa_sources: list[dict[str, object]] = []
        self.projection_calls: list[tuple[str, int, int | None]] = []

    def api_meta(self) -> dict[str, object]:
        return {"defaults": {"timeIndex": 0, "freqIndex": 0}}

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
        self.projection_calls.append((panel, time_index, eovsa_index))
        return [[float(time_index), float(-1 if eovsa_index is None else eovsa_index)]]


class P3BackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = _P3Session()
        api.SESSIONS["p3-test"] = self.session  # type: ignore[assignment]
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        api.SESSIONS.pop("p3-test", None)

    def test_tracking_accepts_context_native_mjd_and_returns_target_mjd(self) -> None:
        response = self.client.post(
            "/api/sessions/p3-test/track/step",
            json={
                "sourceId": "context",
                "sampleMjd": float(self.session.aia.times[1].mjd),
                "direction": 1,
                "point": [4.0, 5.0],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.session.track_calls, [1])
        self.assertEqual(response.json()["row"]["frame_index"], 2)
        self.assertAlmostEqual(response.json()["row"]["time_mjd"], float(self.session.aia.times[2].mjd))

        legacy = self.client.post(
            "/api/sessions/p3-test/track/step",
            json={"sourceId": "context", "frameIndex": 1, "direction": 1, "point": [4.0, 5.0]},
        )
        self.assertEqual(legacy.status_code, 200)
        self.assertEqual(legacy.json(), response.json())

        for payload in (
            {"sourceId": "context", "direction": 1},
            {"sourceId": "context", "frameIndex": 0, "sampleMjd": float(self.session.aia.times[0].mjd), "direction": 1},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(
                    self.client.post("/api/sessions/p3-test/track/step", json=payload).status_code,
                    422,
                )

    def test_radio_roi_does_not_reuse_radio_tolerance_for_aia_alias(self) -> None:
        self.session.aia.times = Time(
            [60000.0 + value / 86400.0 for value in (0, 12, 24)],
            format="mjd",
        )
        self.session.eovsa.times = Time(
            [60000.0 + value / 86400.0 for value in range(13)],
            format="mjd",
        )
        sample_mjd = float(self.session.eovsa.times[5].mjd)

        roi = self.client.post(
            "/api/sessions/p3-test/roi",
            json={
                "panel": "eovsa",
                "sampleMjd": sample_mjd,
                "maxOffsetSeconds": 0.5,
                "points": [[0, 0], [1, 0], [1, 1]],
                "freqIndex": 0,
            },
        )
        projection = self.client.post(
            "/api/sessions/p3-test/roi/projection",
            json={
                "panel": "eovsa",
                "sampleMjd": sample_mjd,
                "maxOffsetSeconds": 0.5,
                "freqIndex": 0,
            },
        )

        self.assertEqual(roi.status_code, 200)
        self.assertEqual(projection.status_code, 200)
        self.assertEqual(roi.headers["X-Resolved-Index"], "5")
        self.assertEqual(projection.headers["X-Resolved-Index"], "5")
        self.assertEqual(self.session.roi_calls, [("eovsa", 0, 5), ("eovsa", 0, 5)])

    def test_tracking_mjd_uses_context_tolerance_and_resolution_headers(self) -> None:
        self.session.aia.times = Time(
            [60000.0 + value / 86400.0 for value in (0, 12, 24, 120)],
            format="mjd",
        )
        available = self.client.post(
            "/api/sessions/p3-test/track/step",
            json={
                "sampleMjd": 60000.0 + 13.0 / 86400.0,
                "maxOffsetSeconds": 2.0,
                "direction": 1,
                "point": [4.0, 5.0],
            },
        )
        unavailable_gap = self.client.post(
            "/api/sessions/p3-test/track/step",
            json={
                "sampleMjd": 60000.0 + 60.0 / 86400.0,
                "direction": 1,
                "point": [4.0, 5.0],
            },
        )
        unavailable_explicit = self.client.post(
            "/api/sessions/p3-test/track/step",
            json={
                "sampleMjd": 60000.0 + 15.0 / 86400.0,
                "maxOffsetSeconds": 2.0,
                "direction": 1,
                "point": [4.0, 5.0],
            },
        )

        self.assertEqual(available.status_code, 200)
        self.assertEqual(available.headers["X-Resolved-Index"], "1")
        self.assertAlmostEqual(float(available.headers["X-Offset-Seconds"]), -1.0, places=4)
        self.assertEqual(unavailable_gap.status_code, 204)
        self.assertEqual(unavailable_explicit.status_code, 204)
        self.assertNotIn("X-Resolved-Index", unavailable_gap.headers)
        self.assertEqual(self.session.track_calls, [1])
        invalid_tolerance = self.client.post(
            "/api/sessions/p3-test/track/step",
            json={"sampleMjd": 60000.0, "maxOffsetSeconds": -1, "direction": 1},
        )
        self.assertEqual(invalid_tolerance.status_code, 422)

    def test_radio_extraction_accepts_one_complete_bound_form(self) -> None:
        start_mjd = float(self.session.eovsa.times[1].mjd)
        end_mjd = float(self.session.eovsa.times[4].mjd)
        canonical = self.client.post(
            "/api/sessions/p3-test/extract/radio-sources",
            json={"startMjd": start_mjd, "endMjd": end_mjd},
        )
        legacy = self.client.post(
            "/api/sessions/p3-test/extract/eovsa-sources",
            json={"startIndex": 0, "endIndex": 1},
        )

        self.assertEqual(canonical.status_code, 200)
        self.assertEqual(legacy.status_code, 200)
        self.assertEqual(self.session.extract_calls[0], (None, None, start_mjd, end_mjd))
        self.assertEqual(self.session.extract_calls[1], (0, 1, None, None))

        invalid_payloads = [
            {"startMjd": start_mjd},
            {"startIndex": 0},
            {"startIndex": 0, "endIndex": 1, "startMjd": start_mjd, "endMjd": end_mjd},
            {"startIndex": 0, "endMjd": end_mjd},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                response = self.client.post(
                    "/api/sessions/p3-test/extract/radio-sources",
                    json=payload,
                )
                self.assertEqual(response.status_code, 422)

    def test_mjd_extraction_bounds_resolve_on_dense_radio_axis(self) -> None:
        with TemporaryDirectory() as directory:
            session = _DenseExtractionSession(Path(directory))
            rows = SolRadSession.extract_eovsa_sources(  # type: ignore[arg-type]
                session,
                x_offset=0.0,
                y_offset=0.0,
                diff_seconds=60.0,
                start_index=None,
                end_index=None,
                stride=1,
                min_snr=1.0,
                start_mjd=60000.0 + 2.0 / 86400.0,
                end_mjd=60000.0 + 8.0 / 86400.0,
            )

        self.assertEqual([row["eovsa_index"] for row in rows], [1, 2, 3])

    def test_radio_extraction_applies_source_channel_offsets(self) -> None:
        with TemporaryDirectory() as directory:
            session = _DenseExtractionSession(Path(directory))
            rows = SolRadSession.extract_eovsa_sources(  # type: ignore[arg-type]
                session,
                x_offset=0.0,
                y_offset=0.0,
                diff_seconds=60.0,
                start_index=None,
                end_index=None,
                stride=1,
                min_snr=1.0,
                start_mjd=60000.0 + 1.0 / 86400.0,
                end_mjd=60000.0 + 1.0 / 86400.0,
            )

        self.assertTrue(rows)
        self.assertAlmostEqual(float(rows[0]["x_peak_arcsec"]), 3.25)
        self.assertAlmostEqual(float(rows[0]["y_peak_arcsec"]), 1.25)
        self.assertAlmostEqual(float(rows[0]["x_offset_arcsec"]), 1.25)
        self.assertAlmostEqual(float(rows[0]["y_offset_arcsec"]), -0.75)

    def test_radio_extraction_skips_masked_channels(self) -> None:
        with TemporaryDirectory() as directory:
            session = _DenseExtractionSession(Path(directory))
            session.channel_mask = [True]
            rows = SolRadSession.extract_eovsa_sources(  # type: ignore[arg-type]
                session,
                x_offset=0.0,
                y_offset=0.0,
                diff_seconds=60.0,
                start_index=None,
                end_index=None,
                stride=1,
                min_snr=1.0,
                start_mjd=60000.0 + 1.0 / 86400.0,
                end_mjd=60000.0 + 1.0 / 86400.0,
            )

        self.assertEqual(rows, [])

    def test_loaded_state_prefers_timeline_mjd_and_preserves_v1_aliases(self) -> None:
        session = _RestoreSession()
        state = {
            "ui": {
                "timeline": {
                    "masterSourceId": "radio",
                    "cursorMjd": 60000.0 + 5.0 / 86400.0,
                    "startMjd": 60000.0 + 1.0 / 86400.0,
                    "endMjd": 60000.0 + 11.0 / 86400.0,
                },
                "timeIndex": 2,
                "startIndex": 2,
                "endIndex": 2,
                "freqIndex": 0,
            }
        }

        loaded = SolRadSession.api_loaded_state(session, state)  # type: ignore[arg-type]

        self.assertEqual(loaded["ui"]["timeIndex"], 0)
        self.assertEqual(loaded["ui"]["startIndex"], 0)
        self.assertEqual(loaded["ui"]["endIndex"], 1)
        self.assertEqual(session.projection_calls, [("aia", 0, None), ("eovsa", 0, 2)])

        session.projection_calls.clear()
        legacy_ui = {"timeIndex": 1, "startIndex": 0, "endIndex": 1, "freqIndex": 0}
        legacy = SolRadSession.api_loaded_state(session, {"ui": legacy_ui})  # type: ignore[arg-type]
        self.assertEqual(legacy["ui"], legacy_ui)
        self.assertEqual(session.projection_calls, [("aia", 1, None), ("eovsa", 1, None)])

    def test_loaded_state_sorts_clamps_and_rejects_nonfinite_timeline_values(self) -> None:
        session = _RestoreSession()
        clamped = SolRadSession.api_loaded_state(  # type: ignore[arg-type]
            session,
            {
                "ui": {
                    "timeline": {
                        "masterSourceId": "radio",
                        "cursorMjd": 60001.0,
                        "startMjd": 60001.0,
                        "endMjd": 59999.0,
                    },
                    "timeIndex": 2,
                    "startIndex": 2,
                    "endIndex": 2,
                    "freqIndex": 0,
                }
            },
        )
        radio_start = float(session.eovsa.times[0].mjd)
        radio_end = float(session.eovsa.times[-1].mjd)
        self.assertAlmostEqual(clamped["ui"]["timeline"]["cursorMjd"], radio_end)
        self.assertAlmostEqual(clamped["ui"]["timeline"]["startMjd"], radio_start)
        self.assertAlmostEqual(clamped["ui"]["timeline"]["endMjd"], radio_end)
        self.assertEqual(clamped["ui"]["timeIndex"], 1)
        self.assertEqual((clamped["ui"]["startIndex"], clamped["ui"]["endIndex"]), (0, 1))
        self.assertEqual(session.projection_calls[-1], ("eovsa", 1, 5))

        session.projection_calls.clear()
        invalid = SolRadSession.api_loaded_state(  # type: ignore[arg-type]
            session,
            {
                "ui": {
                    "timeline": {
                        "masterSourceId": "radio",
                        "cursorMjd": float("nan"),
                        "startMjd": float("inf"),
                        "endMjd": radio_end,
                    },
                    "timeIndex": 2,
                    "startIndex": 1,
                    "endIndex": 2,
                    "freqIndex": 0,
                }
            },
        )
        self.assertNotIn("cursorMjd", invalid["ui"]["timeline"])
        self.assertNotIn("startMjd", invalid["ui"]["timeline"])
        self.assertNotIn("endMjd", invalid["ui"]["timeline"])
        self.assertEqual((invalid["ui"]["timeIndex"], invalid["ui"]["startIndex"], invalid["ui"]["endIndex"]), (2, 1, 2))
        self.assertEqual(session.projection_calls, [("aia", 2, None), ("eovsa", 2, None)])


if __name__ == "__main__":
    unittest.main()
