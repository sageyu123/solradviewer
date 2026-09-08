"""Focused tests for source-native time addressing in the API."""

from __future__ import annotations

import unittest

import numpy as np
from astropy.time import Time
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sad_eovsa_tool.backend import app as api
from sad_eovsa_tool.backend.data import resolve_time_index


class _FakeAia:
    def __init__(self, times: Time) -> None:
        self.times = times
        self.calls: list[int] = []

    def texture(self, index: int, *args: object) -> bytes:
        self.calls.append(int(index))
        return b"aia-frame"


class _FakeRadio:
    def __init__(self, times: Time) -> None:
        self.times = times
        self.calls: list[float] = []

    def texture_for_aia_time(self, mjd: float, *args: object, **kwargs: object) -> bytes:
        self.calls.append(float(mjd))
        native_index = kwargs.get("eovsa_index")
        return f"radio-frame-{native_index}".encode()


class _FakeSession:
    context_source_id = "context"
    radio_source_id = "radio"
    spectrogram_source_id = "spectrogram"

    def __init__(self) -> None:
        self.aia = _FakeAia(Time(["2025-03-28T00:00:00", "2025-03-28T00:00:10"], format="isot"))
        self.eovsa = _FakeRadio(Time(["2025-03-28T00:00:01", "2025-03-28T00:00:11"], format="isot"))
        self.contour_calls: list[int] = []
        self.roi_calls: list[tuple[str, int, int | None]] = []

    def eovsa_all_band_contours_on_aia(self, index: int, *args: object, **kwargs: object) -> bytes:
        self.contour_calls.append(int(index))
        return b"contours"

    def set_roi_from_pixels(self, panel: str, points: list[list[float]], time_index: int, freq_index: int, *args: object) -> dict[str, object]:
        self.roi_calls.append((panel, int(time_index), args[-1] if args else None))
        return {"roiWorld": points}

    def roi_pixels_for_panel(self, panel: str, time_index: int, freq_index: int, *args: object) -> list[list[float]]:
        self.roi_calls.append((panel, int(time_index), args[-1] if args else None))
        return [[1.0, 2.0], [3.0, 4.0]]


class SourceNativeTimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = _FakeSession()
        api.SESSIONS["native-test"] = self.session  # type: ignore[assignment]

    def tearDown(self) -> None:
        api.SESSIONS.pop("native-test", None)

    def client(self) -> TestClient:
        return TestClient(api.app)

    def test_resolution_policy_edges_tie_and_tolerance(self) -> None:
        values = np.array([10.0, 10.0 + 2.0 / 86400.0, 10.0 + 4.0 / 86400.0])
        nearest = resolve_time_index(values, 10.0 + 1.0 / 86400.0)
        self.assertIsNotNone(nearest)
        assert nearest is not None
        self.assertEqual(nearest[0], 0)
        self.assertAlmostEqual(nearest[2], -1.0, places=4)
        self.assertIsNone(resolve_time_index(values, 9.0, "previous"))
        self.assertIsNone(resolve_time_index(values, 11.0, "next"))
        self.assertEqual(resolve_time_index(values, 10.0 + 1.0 / 86400.0, "previous")[0], 0)
        self.assertEqual(resolve_time_index(values, 10.0 + 1.0 / 86400.0, "next")[0], 1)
        self.assertIsNone(resolve_time_index(values, 10.0 + 1.0 / 86400.0, max_offset_seconds=0.1))
        self.assertEqual(resolve_time_index(values, 10.0, max_offset_seconds=0.0)[0], 0)
        duplicates = np.array([10.0, 10.0, 11.0])
        self.assertEqual(resolve_time_index(duplicates, 10.0, "previous")[0], 1)
        self.assertEqual(resolve_time_index(duplicates, 10.0)[0], 0)

    def test_context_frame_has_resolution_headers_and_legacy_body(self) -> None:
        legacy = api.source_frame("native-test", "context", timeIndex=1)
        native = api.source_frame("native-test", "context", sampleMjd=float(self.session.aia.times[1].mjd))
        self.assertEqual(legacy.status_code, 200)
        self.assertEqual(legacy.body, b"aia-frame")
        self.assertEqual(native.body, legacy.body)
        self.assertEqual(legacy.headers["X-Resolved-Index"], "1")
        self.assertAlmostEqual(float(legacy.headers["X-Offset-Seconds"]), 0.0)

        legacy_radio = api.source_frame("native-test", "radio", timeIndex=1)
        native_radio = api.source_frame("native-test", "radio", sampleMjd=float(self.session.eovsa.times[1].mjd))
        self.assertEqual(legacy_radio.body, native_radio.body)

    def test_duplicate_native_time_body_matches_resolved_header(self) -> None:
        self.session.eovsa.times = Time(
            ["2025-03-28T00:00:00", "2025-03-28T00:00:00", "2025-03-28T00:00:10"], format="isot"
        )
        response = api.source_frame(
            "native-test", "radio", sampleMjd=float(self.session.eovsa.times[0].mjd), samplingPolicy="previous"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Resolved-Index"], "1")
        self.assertEqual(response.body, b"radio-frame-1")

    def test_radio_native_sampling_and_out_of_tolerance(self) -> None:
        requested = float(self.session.eovsa.times[1].mjd) + 2.0 / 86400.0
        response = api.source_frame(
            "native-test", "radio", sampleMjd=requested, samplingPolicy="previous", maxOffsetSeconds=3.0
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Resolved-Index"], "1")
        self.assertLess(float(response.headers["X-Offset-Seconds"]), 0.0)
        unavailable = api.source_frame(
            "native-test", "radio", sampleMjd=requested, samplingPolicy="previous", maxOffsetSeconds=1.0
        )
        self.assertEqual(unavailable.status_code, 204)
        self.assertNotIn("X-Resolved-Index", unavailable.headers)

    def test_both_addressing_forms_are_rejected_and_overlay_resolves_native_axis(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            api.source_frame("native-test", "context", timeIndex=0, sampleMjd=1.0)
        self.assertEqual(raised.exception.status_code, 422)
        response = api.source_overlay_contours(
            "native-test", "radio", sampleMjd=float(self.session.eovsa.times[0].mjd)
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Resolved-Index"], "0")
        self.assertEqual(self.session.contour_calls, [0])

    def test_cors_exposes_resolution_headers(self) -> None:
        options = next(m.kwargs for m in api.app.user_middleware if m.cls is api.CORSMiddleware)
        self.assertEqual(
            options["expose_headers"],
            [
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

    def test_testclient_validates_queries_headers_cors_and_roi_native_index(self) -> None:
        client = self.client()
        sample = float(self.session.eovsa.times[1].mjd)
        response = client.get(
            "/api/sessions/native-test/sources/radio/frame.png",
            params={"sampleMjd": sample},
            headers={"Origin": "http://localhost:5174"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Resolved-Index"], "1")
        self.assertIn("X-Resolved-Index", response.headers["Access-Control-Expose-Headers"])
        self.assertEqual(response.content, b"radio-frame-1")

        both = client.get(
            "/api/sessions/native-test/sources/context/frame.png",
            params={"timeIndex": 0, "sampleMjd": float(self.session.aia.times[0].mjd)},
        )
        self.assertEqual(both.status_code, 422)
        unknown = client.get(
            "/api/sessions/native-test/sources/not-a-source/frame.png",
            params={"sampleMjd": sample},
        )
        self.assertEqual(unknown.status_code, 422)
        invalid_mjd = client.get(
            "/api/sessions/native-test/sources/context/frame.png", params={"sampleMjd": "nan"}
        )
        self.assertEqual(invalid_mjd.status_code, 422)
        invalid_offset = client.get(
            "/api/sessions/native-test/sources/context/frame.png",
            params={"sampleMjd": float(self.session.aia.times[0].mjd), "maxOffsetSeconds": "NaN"},
        )
        self.assertEqual(invalid_offset.status_code, 422)
        negative_offset = client.get(
            "/api/sessions/native-test/sources/context/frame.png",
            params={"sampleMjd": float(self.session.aia.times[0].mjd), "maxOffsetSeconds": -1},
        )
        self.assertEqual(negative_offset.status_code, 422)
        invalid_legacy_offset = client.get(
            "/api/sessions/native-test/sources/context/frame.png",
            params={"timeIndex": 0, "maxOffsetSeconds": "NaN"},
        )
        self.assertEqual(invalid_legacy_offset.status_code, 422)
        no_time = client.get("/api/sessions/native-test/sources/radio/overlay-contours.png")
        self.assertEqual(no_time.status_code, 422)
        unknown_contour = client.get(
            "/api/sessions/native-test/sources/not-a-source/overlay-contours.png",
            params={"sampleMjd": sample},
        )
        self.assertEqual(unknown_contour.status_code, 422)

        roi = client.post(
            "/api/sessions/native-test/roi",
            json={
                "panel": "eovsa",
                "sampleMjd": sample,
                "points": [[0, 0], [1, 0], [1, 1]],
                "freqIndex": 0,
            },
        )
        self.assertEqual(roi.status_code, 200)
        self.assertEqual(roi.headers["X-Resolved-Index"], "1")
        self.assertEqual(self.session.roi_calls[-1], ("eovsa", 1, 1))

        projection = client.post(
            "/api/sessions/native-test/roi/projection",
            json={"panel": "eovsa", "sampleMjd": sample, "freqIndex": 0},
        )
        self.assertEqual(projection.status_code, 200)
        self.assertIn("X-Resolved-Mjd", projection.headers)

    def test_dense_radio_roi_keeps_native_index_and_overlay_rejects_unavailable_target(self) -> None:
        self.session.aia.times = Time(["2025-03-28T00:00:00", "2025-03-28T00:01:00"], format="isot")
        self.session.eovsa.times = Time(
            ["2025-03-28T00:00:00", "2025-03-28T00:00:01", "2025-03-28T00:00:02"], format="isot"
        )
        client = self.client()
        sample = float(self.session.eovsa.times[2].mjd)
        roi = client.post(
            "/api/sessions/native-test/roi",
            json={"panel": "eovsa", "sampleMjd": sample, "samplingPolicy": "previous", "points": [[0, 0], [1, 0], [1, 1]], "freqIndex": 0},
        )
        self.assertEqual(roi.status_code, 200)
        self.assertEqual(self.session.roi_calls[-1], ("eovsa", 0, 2))

        self.session.eovsa.times = Time(["2025-03-28T00:00:00", "2025-03-28T00:02:00"], format="isot")
        unavailable = client.get(
            "/api/sessions/native-test/sources/radio/overlay-contours.png",
            params={"sampleMjd": float(self.session.eovsa.times[1].mjd)},
        )
        self.assertEqual(unavailable.status_code, 204)
        self.assertNotIn("X-Resolved-Index", unavailable.headers)


if __name__ == "__main__":
    unittest.main()
