"""Tests for the pixel light-curve probe helpers and API route."""

from __future__ import annotations

from collections import OrderedDict
import unittest

import numpy as np
from astropy.time import Time
from fastapi.testclient import TestClient

from sad_eovsa_tool.backend import app as api
from sad_eovsa_tool.backend.data import (
    SadEovsaSession,
    patch_mean,
    series_stats,
    stride_indices,
    temporal_filter,
    temporal_filter_series,
)


class _ProbeAia:
    def __init__(self, values: np.ndarray, times: Time) -> None:
        self.values = values
        self.times = times
        self.shape = values.shape[1:]

    def frame(self, index: int, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        return self.values[int(index)]


class _ProbeRadio:
    nfreq = 1

    def __init__(self, values: np.ndarray, times: Time) -> None:
        self.values = values[:, None, :, :]
        self.times = times
        self.shape = values.shape[1:]

    def mode_data(self, index: int, *args: object) -> np.ndarray:
        del args
        return self.values[int(index)]

    def operation_data(self, index: int, *args: object) -> np.ndarray:
        del args
        return self.values[int(index)]


def _probe_session() -> SadEovsaSession:
    times = Time([60000.0 + value / 86400.0 for value in (0.0, 1.0, 2.0)], format="mjd")
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    session = object.__new__(SadEovsaSession)
    session.session_id = "probe-test"
    session.context_source_id = "context"
    session.radio_source_id = "radio"
    session.aia = _ProbeAia(values, times)
    session.eovsa = _ProbeRadio(values, times)
    session._timeseries_cache = OrderedDict()
    return session


class PixelLightcurveProbeTest(unittest.TestCase):
    def test_patch_mean_correctness_on_synthetic_sequence(self) -> None:
        values = np.arange(25, dtype=np.float32).reshape(5, 5)
        self.assertAlmostEqual(patch_mean(values, 2, 2, 1), 12.0)
        values[1:4, 1:4] = np.nan
        self.assertTrue(np.isnan(patch_mean(values, 2, 2, 1)))

    def test_stride_math_limits_native_samples(self) -> None:
        selected, stride = stride_indices(10, 4)
        np.testing.assert_array_equal(selected, np.array([0, 3, 6, 9]))
        self.assertEqual(stride, 3)
        self.assertEqual(len(stride_indices(4, 4)[0]), 4)

    def test_series_stats_percentiles_ignore_nan(self) -> None:
        stats = series_stats([np.nan, 1.0, 2.0, 3.0, 100.0])
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 100.0)
        self.assertAlmostEqual(float(stats["p1"]), 1.03, places=6)
        self.assertAlmostEqual(float(stats["p99"]), 97.09, places=6)

    def test_temporal_series_smoothing_matches_frame_filter_weights(self) -> None:
        values = np.array([0.0, 1.0, 4.0, 2.0, 0.0], dtype=np.float32)
        times = np.arange(len(values), dtype=float)
        series = temporal_filter_series(values, times, "lowpass", 1.5, 8.0)
        expected = np.array([
            temporal_filter(values[:, None, None], times, index, "lowpass", 1.5, 8.0)[0, 0]
            for index in range(len(values))
        ])
        np.testing.assert_allclose(series, expected, rtol=0.0, atol=1e-6)

    def test_pixel_probe_rejects_out_of_bounds_coordinates(self) -> None:
        api.SESSIONS["probe-test"] = _probe_session()  # type: ignore[assignment]
        try:
            response = TestClient(api.app).get(
                "/api/sessions/probe-test/sources/context/timeseries",
                params={"x": 3, "y": 1, "startMjd": 60000, "endMjd": 60000.0001},
            )
            self.assertEqual(response.status_code, 422)
        finally:
            api.SESSIONS.pop("probe-test", None)

    def test_pixel_probe_out_of_range_window_returns_empty_arrays(self) -> None:
        api.SESSIONS["probe-test"] = _probe_session()  # type: ignore[assignment]
        try:
            response = TestClient(api.app).get(
                "/api/sessions/probe-test/sources/context/timeseries",
                params={"x": 1, "y": 1, "startMjd": 60001, "endMjd": 59999},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["mjd"], [])
            self.assertEqual(response.json()["raw"], [])
            self.assertEqual(response.json()["nTotal"], 0)
        finally:
            api.SESSIONS.pop("probe-test", None)


if __name__ == "__main__":
    unittest.main()
