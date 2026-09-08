"""Tests for radial enhancement and temporal denoise filters."""

from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from astropy.io import fits
from fastapi.testclient import TestClient
from PIL import Image

from sad_eovsa_tool.backend import app as api
from sad_eovsa_tool.backend.data import (
    AiaFitsSequence,
    _temporal_frame_for_source,
    downsample_to_cap,
    radial_factor_map,
    temporal_filter,
)


def _header(timestamp: str) -> fits.Header:
    header = fits.Header()
    header["T_OBS"] = timestamp
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRPIX1"] = 1.0
    header["CRPIX2"] = 1.0
    header["CRVAL1"] = 0.0
    header["CRVAL2"] = 0.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    return header


class EnhancementFilterTest(unittest.TestCase):
    def test_resolution_cap_preserves_aspect_and_matches_block_mean(self) -> None:
        full = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
        capped = downsample_to_cap(full, max_width=4, max_height=3)
        expected = full.reshape(3, 2, 4, 2).mean(axis=(1, 3))

        self.assertEqual(capped.shape, (3, 4))
        self.assertEqual(capped.shape[1] / capped.shape[0], full.shape[1] / full.shape[0])
        np.testing.assert_array_equal(capped, expected)

    def test_capped_frame_uses_full_resolution_stats_and_no_cap_is_byte_identical(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(values, header=_header("2025-03-28T00:00:00")),
            ]).writeto(directory / "frame_0.fits")
            sequence = AiaFitsSequence(directory, pattern="frame_*.fits")
            session = type("Session", (), {"aia": sequence})()
            api.SESSIONS["frame-cap-test"] = session  # type: ignore[assignment]
            try:
                expected = sequence.texture(0, 0.0, 47.0, "gray", "linear", difference_mode="none")
                legacy = TestClient(api.app).get(
                    "/api/sessions/frame-cap-test/aia/frame.png",
                    params={"timeIndex": 0, "vmin": 0, "vmax": 47, "differenceMode": "none"},
                )
                capped = TestClient(api.app).get(
                    "/api/sessions/frame-cap-test/aia/frame.png",
                    params={
                        "timeIndex": 0,
                        "vmin": 0,
                        "vmax": 47,
                        "differenceMode": "none",
                        "maxWidth": 4,
                        "maxHeight": 3,
                    },
                )
            finally:
                api.SESSIONS.pop("frame-cap-test", None)

        self.assertEqual(legacy.content, expected)
        self.assertEqual(Image.open(BytesIO(capped.content)).size, (4, 3))
        self.assertEqual(float(capped.headers["X-Data-Min"]), 0.0)
        self.assertEqual(float(capped.headers["X-Data-Max"]), 47.0)

    def test_aia_frame_response_exposes_consistent_robust_data_stats(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(values, header=_header("2025-03-28T00:00:00")),
            ]).writeto(directory / "frame_0.fits")
            sequence = AiaFitsSequence(directory, pattern="frame_*.fits")
            session = type("Session", (), {"aia": sequence})()
            api.SESSIONS["frame-stats-test"] = session  # type: ignore[assignment]
            try:
                response = TestClient(api.app).get(
                    "/api/sessions/frame-stats-test/aia/frame.png",
                    params={"timeIndex": 0, "differenceMode": "none", "useRunningDiff": "false"},
                )
            finally:
                api.SESSIONS.pop("frame-stats-test", None)

        self.assertEqual(response.status_code, 200)
        data_min = float(response.headers["X-Data-Min"])
        data_max = float(response.headers["X-Data-Max"])
        p1 = float(response.headers["X-Data-P1"])
        p99 = float(response.headers["X-Data-P99"])
        self.assertLessEqual(data_min, p1)
        self.assertLess(p1, p99)
        self.assertLessEqual(p99, data_max)
        self.assertEqual(data_min, 1.0)
        self.assertEqual(data_max, 4.0)

    def test_radial_factor_map_gamma_zero_is_identity_and_cap_is_respected(self) -> None:
        radius = np.array([[0.5, 1.0, 2.0, 2.5, 4.0]], dtype=np.float32)
        identity = radial_factor_map(radius, 0.0)
        np.testing.assert_array_equal(identity, np.ones_like(radius))

        factors = radial_factor_map(radius, 2.0)
        np.testing.assert_array_equal(factors[0, :2], np.ones(2, dtype=np.float32))
        self.assertAlmostEqual(float(factors[0, 2]), 1000.0, places=5)
        self.assertAlmostEqual(float(factors[0, 3]), 1000.0, places=5)
        self.assertAlmostEqual(float(factors[0, 4]), 1000.0, places=5)

        typed_outside_slider = radial_factor_map(radius, 5.0)
        self.assertAlmostEqual(float(typed_outside_slider[0, 3]), 1000.0, places=5)

    def test_radial_factor_map_matches_exponential_scale_height_anchors(self) -> None:
        radius = np.array([1.0, 1.2, 1.3], dtype=np.float32)
        factors = radial_factor_map(radius, 3.0)
        np.testing.assert_allclose(factors, [1.0, np.exp(3.0), np.exp(4.5)], rtol=1e-5, atol=1e-5)

    def test_temporal_filter_constant_series_is_a_fixed_point(self) -> None:
        data = np.full((7, 2, 2), 3.5, dtype=np.float32)
        times = np.arange(7, dtype=float) * 2.0
        lowpass = temporal_filter(data, times, 3, "lowpass", 3.0, 12.0)
        bandpass = temporal_filter(data, times, 3, "bandpass", 3.0, 12.0)
        np.testing.assert_allclose(lowpass, 3.5, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(bandpass, 0.0, rtol=0.0, atol=1e-6)

    def test_temporal_filter_single_spike_matches_gaussian_weight(self) -> None:
        data = np.zeros((3, 1, 1), dtype=np.float32)
        data[1, 0, 0] = 1.0
        times = np.array([0.0, 1.0, 2.0])
        result = temporal_filter(data, times, 1, "lowpass", 1.0, 10.0)
        weight = np.exp(-0.5)
        expected = 1.0 / (1.0 + 2.0 * weight)
        self.assertAlmostEqual(float(result[0, 0]), float(expected), places=6)

    def test_temporal_filter_excludes_nan_frames_from_weights(self) -> None:
        data = np.array([[[np.nan]], [[2.0]], [[4.0]]], dtype=np.float32)
        times = np.array([0.0, 1.0, 2.0])
        result = temporal_filter(data, times, 1, "lowpass", 1.0, 10.0)
        weight = np.exp(-0.5)
        expected = (2.0 + 4.0 * weight) / (1.0 + weight)
        self.assertAlmostEqual(float(result[0, 0]), float(expected), places=6)

    def test_parallel_temporal_window_matches_serial_result(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(17, 5, 4)).astype(np.float32)
        data[3, 1, 2] = np.nan
        times = np.arange(17, dtype=float) * 12.0
        source = type("SyntheticSource", (), {"_temporal_cache": OrderedDict()})()

        parallel = _temporal_frame_for_source(
            source,
            8,
            times,
            lambda index: data[index],
            ("synthetic",),
            "lowpass",
            31.0,
            120.0,
        )
        serial = temporal_filter(data, times, 8, "lowpass", 31.0, 120.0)

        np.testing.assert_array_equal(parallel, serial)

    def test_aia_frame_endpoint_is_byte_stable_without_new_parameters(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, value in enumerate((1.0, 2.0)):
                fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(
                        np.full((2, 2), value, dtype=np.float32),
                        header=_header(f"2025-03-28T00:00:0{index}"),
                    ),
                ]).writeto(directory / f"frame_{index}.fits")
            sequence = AiaFitsSequence(directory, pattern="frame_*.fits")
            session = type("Session", (), {"aia": sequence})()
            api.SESSIONS["enhancement-byte-test"] = session  # type: ignore[assignment]
            try:
                expected = sequence.texture(1, 0.5, 1.5, "gray", "linear")
                response = TestClient(api.app).get(
                    "/api/sessions/enhancement-byte-test/aia/frame.png",
                    params={"timeIndex": 1},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, expected)
            finally:
                api.SESSIONS.pop("enhancement-byte-test", None)


if __name__ == "__main__":
    unittest.main()
