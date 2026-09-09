"""Tests for unshifted client-rendered radio contour geometry."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from astropy.io import fits
from astropy.time import Time
from fastapi.testclient import TestClient
import numpy as np
from skimage import measure

from solradviewer.backend import app as api
from solradviewer.backend.data import RenderDiskCache, SolRadSession, _sfu_to_tb_thresholds


def _header() -> fits.Header:
    header = fits.Header()
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    return header


def _session(bands: np.ndarray) -> SolRadSession:
    session = SolRadSession.__new__(SolRadSession)
    session.aia = SimpleNamespace(
        nt=1,
        times=Time([60000.0], format="mjd"),
        shape=(8, 9),
    )
    session.eovsa = Mock()
    session.eovsa.nfreq = bands.shape[0]
    session.eovsa.shape = bands.shape[1:]
    session.eovsa.files = [object()]
    session.eovsa.times = Time([60000.0], format="mjd")
    session.eovsa.freqs_hz = np.linspace(1.0e9, 2.0e9, bands.shape[0])
    session.eovsa.header = _header()
    session.eovsa.nearest_time_index.return_value = 0
    session.eovsa.mode_data.return_value = bands
    session.eovsa.operation_data.return_value = bands
    session.radio_peak_cache = {}
    session.radio_peak_table_cache = {}
    session.channel_offsets = {
        "dx": [100.0] * bands.shape[0],
        "dy": [-50.0] * bands.shape[0],
        "masked": [False] * bands.shape[0],
    }
    session.channel_mask = [False] * bands.shape[0]
    session._overlay_cache = OrderedDict()
    session._contour_geometry_cache = OrderedDict()
    session._affine_cache = OrderedDict()
    session._eovsa_to_aia_affine = Mock(
        return_value=np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    )
    return session


class ContourGeometryTest(unittest.TestCase):
    def test_polylines_match_find_contours_in_unshifted_target_pixels(self) -> None:
        data = np.zeros((1, 5, 6), dtype=np.float32)
        data[0, 1:4, 2:5] = 10.0
        session = _session(data)
        expected = measure.find_contours(data[0], 5.0)[0]
        expected_xy = np.column_stack([expected[:, 1] + 2.0, expected[:, 0] + 3.0])

        with TemporaryDirectory() as temporary, patch(
            "solradviewer.backend.data.RENDER_DISK_CACHE",
            RenderDiskCache(Path(temporary)),
        ):
            payload = session.eovsa_contour_geometry(
                0, 30.0, False, 50.0, target_panel="aia", eovsa_index=0
            )

        self.assertEqual(len(payload["bands"]), 1)
        band = payload["bands"][0]
        np.testing.assert_allclose(band["polylines"][0], expected_xy)
        session._eovsa_to_aia_affine.assert_called_once_with(0, 0.0, 0.0)

    def test_masked_channels_are_excluded(self) -> None:
        data = np.zeros((2, 5, 6), dtype=np.float32)
        data[:, 1:4, 2:5] = 10.0
        session = _session(data)
        session.channel_mask = [False, True]

        with TemporaryDirectory() as temporary, patch(
            "solradviewer.backend.data.RENDER_DISK_CACHE",
            RenderDiskCache(Path(temporary)),
        ):
            payload = session.eovsa_contour_geometry(
                0, 30.0, False, 50.0, target_panel="eovsa", eovsa_index=0
            )

        self.assertEqual([band["channel"] for band in payload["bands"]], [0])

    def test_geometry_json_disk_cache_survives_memory_cache_clear(self) -> None:
        data = np.zeros((1, 5, 6), dtype=np.float32)
        data[0, 1:4, 2:5] = 10.0
        session = _session(data)

        with TemporaryDirectory() as temporary, patch(
            "solradviewer.backend.data.RENDER_DISK_CACHE",
            RenderDiskCache(Path(temporary)),
        ) as cache:
            expected = session.eovsa_contour_geometry(
                0, 30.0, False, 50.0, target_panel="eovsa", eovsa_index=0
            )
            session._contour_geometry_cache.clear()
            with patch(
                "solradviewer.backend.data.measure.find_contours",
                side_effect=AssertionError("geometry disk cache miss"),
            ):
                actual = session.eovsa_contour_geometry(
                    0, 30.0, False, 50.0, target_panel="eovsa", eovsa_index=0
                )

            self.assertEqual(actual, expected)
            self.assertEqual(cache.stats()["hits"], 1)
            self.assertEqual(len(list(Path(temporary).rglob("*.json"))), 1)

    def test_percent_kelvin_and_sfu_levels_match_png_path(self) -> None:
        cases = [
            ({"level_reference": "current", "level_percent": 40.0}, 4.0),
            ({"level_reference": "global", "level_mode": "kelvin", "level_kelvin": 6.0}, 6.0),
            (
                {"level_reference": "global", "level_mode": "sfu", "level_sfu": 1.0e-9},
                float(_sfu_to_tb_thresholds(1.0e-9, [1.0e9], _header())[0]),
            ),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                data = np.zeros((1, 5, 6), dtype=np.float32)
                data[0, 1:4, 2:5] = 10.0
                session = _session(data)
                session.channel_offsets = {
                    "dx": [0.0], "dy": [0.0], "masked": [False]
                }
                seen: list[float] = []

                def trace(_data: np.ndarray, threshold: float) -> list[np.ndarray]:
                    seen.append(float(threshold))
                    return []

                with TemporaryDirectory() as temporary, patch(
                    "solradviewer.backend.data.RENDER_DISK_CACHE",
                    RenderDiskCache(Path(temporary)),
                ), patch(
                    "solradviewer.backend.data.measure.find_contours", side_effect=trace
                ):
                    session.eovsa_contour_geometry(
                        0,
                        30.0,
                        False,
                        float(kwargs.get("level_percent", 50.0)),
                        level_reference=str(kwargs.get("level_reference", "current")),
                        level_mode=str(kwargs.get("level_mode", "percent")),
                        level_kelvin=float(kwargs.get("level_kelvin", 1_000_000.0)),
                        level_sfu=float(kwargs.get("level_sfu", 1.0)),
                        target_panel="eovsa",
                        eovsa_index=0,
                    )
                    session.eovsa_all_band_contours_on_aia(
                        0,
                        30.0,
                        False,
                        0.0,
                        0.0,
                        float(kwargs.get("level_percent", 50.0)),
                        False,
                        1.0,
                        level_reference=str(kwargs.get("level_reference", "current")),
                        level_mode=str(kwargs.get("level_mode", "percent")),
                        level_kelvin=float(kwargs.get("level_kelvin", 1_000_000.0)),
                        level_sfu=float(kwargs.get("level_sfu", 1.0)),
                        target_panel="eovsa",
                        eovsa_index=0,
                    )

                self.assertEqual(len(seen), 2)
                self.assertAlmostEqual(seen[0], expected)
                self.assertAlmostEqual(seen[1], expected)


class _EndpointAxis:
    def __init__(self) -> None:
        self.times = Time([60000.0], format="mjd")


class _EndpointSession:
    context_source_id = "context-source"
    radio_source_id = "radio-source"

    def __init__(self) -> None:
        self.aia = _EndpointAxis()
        self.eovsa = _EndpointAxis()

    def eovsa_contour_geometry(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "bands": [{
                "channel": 0,
                "freqGhz": 1.4,
                "level": 5.0,
                "polylines": [[[1.0, 2.0], [2.0, 3.0], [3.0, 2.0]]],
            }]
        }


class ContourGeometryEndpointTest(unittest.TestCase):
    def test_endpoint_returns_band_shape_and_resolution(self) -> None:
        api.SESSIONS["geometry-test"] = _EndpointSession()  # type: ignore[assignment]
        try:
            response = TestClient(api.app).get(
                "/api/sessions/geometry-test/sources/radio-source/contour-geometry",
                params={"targetSourceId": "radio-source", "sampleMjd": 60000.0},
            )
        finally:
            api.SESSIONS.pop("geometry-test", None)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json()), {"bands", "resolvedIndex", "resolvedMjd"})
        self.assertEqual(response.json()["resolvedIndex"], 0)
        self.assertEqual(response.json()["bands"][0]["channel"], 0)
        self.assertEqual(response.json()["bands"][0]["polylines"][0][0], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
