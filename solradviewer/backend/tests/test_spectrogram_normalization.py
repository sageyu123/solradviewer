"""Tests for per-row dynamic-spectrum normalization."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits
from fastapi.testclient import TestClient

from solradviewer.backend import app as api
from solradviewer.backend.data import (
    EovsaSpectrogram,
    RenderDiskCache,
    _normalize_spectrogram_rows,
    _spectrogram_row_medians,
)


def _write_spectrogram(path: Path) -> None:
    values = np.array([
        [2.0, 4.0, 6.0, 8.0],
        [-1.0, 0.0, 0.0, 2.0],
        [1.0, 3.0, 5.0, 7.0],
    ], dtype=np.float32)
    frequencies = fits.BinTableHDU.from_columns([
        fits.Column(name="FGHZ", format="E", array=np.array([1.0, 2.0, 4.0], dtype=np.float32)),
    ])
    times = fits.BinTableHDU.from_columns([
        fits.Column(name="TIME", format="D", array=2460762.5 + np.arange(4, dtype=float) / 86400.0),
    ])
    fits.HDUList([fits.PrimaryHDU(values), frequencies, times]).writeto(path)


class SpectrogramNormalizationTest(unittest.TestCase):
    def test_divide_uses_full_row_medians_and_guards_zero_and_nan(self) -> None:
        values = np.array([
            [2.0, 4.0, 6.0],
            [-1.0, 0.0, 2.0],
            [np.nan, 2.0, 4.0],
            [np.nan, np.nan, np.nan],
        ], dtype=np.float32)
        medians = _spectrogram_row_medians(values)

        result = _normalize_spectrogram_rows(values, medians, "divide")

        np.testing.assert_allclose(medians[:3], [4.0, 0.0, 3.0])
        self.assertTrue(np.isnan(medians[3]))
        np.testing.assert_allclose(result[0], [0.5, 1.0, 1.5])
        np.testing.assert_allclose(result[1], values[1])
        np.testing.assert_allclose(result[2], [np.nan, 2.0 / 3.0, 4.0 / 3.0], equal_nan=True)
        np.testing.assert_allclose(result[3], values[3], equal_nan=True)

    def test_subtract_uses_full_row_medians_and_guards_nan(self) -> None:
        values = np.array([
            [2.0, 4.0, 6.0],
            [-1.0, 0.0, 2.0],
            [np.nan, 2.0, 4.0],
            [np.nan, np.nan, np.nan],
        ], dtype=np.float32)
        medians = _spectrogram_row_medians(values)

        result = _normalize_spectrogram_rows(values, medians, "subtract")

        np.testing.assert_allclose(result[0], [-2.0, 0.0, 2.0])
        np.testing.assert_allclose(result[1], values[1])
        np.testing.assert_allclose(result[2], [np.nan, -1.0, 1.0], equal_nan=True)
        np.testing.assert_allclose(result[3], values[3], equal_nan=True)

    def test_texture_cache_keys_include_normalization_and_medians_are_reused(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_spectrogram(root / "spectrogram.fits")
            source = EovsaSpectrogram(root / "spectrogram.fits")
            medians = source._row_medians
            cache = RenderDiskCache(root / "cache", max_bytes=1024 * 1024)
            with patch("solradviewer.backend.data.RENDER_DISK_CACHE", cache):
                source.texture(normalization="none")
                source.texture(normalization="divide")
                source.texture(normalization="subtract")

        self.assertIs(source._row_medians, medians)
        self.assertEqual({key[-1] for key in source._texture_cache}, {"none", "divide", "subtract"})

    def test_endpoint_rejects_unknown_normalization(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_spectrogram(root / "spectrogram.fits")
            source = EovsaSpectrogram(root / "spectrogram.fits")
            session = type("Session", (), {"spectrogram_source_id": "spectrogram", "spectrogram": source})()
            api.SESSIONS["spectrogram-normalization-test"] = session  # type: ignore[assignment]
            try:
                response = TestClient(api.app).get(
                    "/api/sessions/spectrogram-normalization-test/sources/spectrogram/spectrogram.png",
                    params={"normalization": "unknown"},
                )
            finally:
                api.SESSIONS.pop("spectrogram-normalization-test", None)

        self.assertEqual(response.status_code, 422)

    def test_absent_normalization_is_byte_identical_to_explicit_none(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_spectrogram(root / "spectrogram.fits")
            source = EovsaSpectrogram(root / "spectrogram.fits")
            session = type("Session", (), {"spectrogram_source_id": "spectrogram", "spectrogram": source})()
            api.SESSIONS["spectrogram-legacy-test"] = session  # type: ignore[assignment]
            cache = RenderDiskCache(root / "cache", max_bytes=1024 * 1024)
            try:
                with patch("solradviewer.backend.data.RENDER_DISK_CACHE", cache):
                    legacy = TestClient(api.app).get(
                        "/api/sessions/spectrogram-legacy-test/sources/spectrogram/spectrogram.png"
                    )
                    explicit = TestClient(api.app).get(
                        "/api/sessions/spectrogram-legacy-test/sources/spectrogram/spectrogram.png",
                        params={"normalization": "none"},
                    )
            finally:
                api.SESSIONS.pop("spectrogram-legacy-test", None)

        self.assertEqual(legacy.status_code, 200)
        self.assertEqual(explicit.status_code, 200)
        self.assertEqual(legacy.content, explicit.content)


if __name__ == "__main__":
    unittest.main()
