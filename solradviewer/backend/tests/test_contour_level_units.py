from __future__ import annotations

from collections import OrderedDict
from io import BytesIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
from astropy.io import fits

from solradviewer.backend.data import SolRadSession, _sfu_to_tb_thresholds


class ContourLevelUnitsTest(unittest.TestCase):
    @staticmethod
    def _eovsa_header(cdelt_arcsec: float = 2.0) -> fits.Header:
        header = fits.Header()
        header["CDELT1"] = cdelt_arcsec
        header["CDELT2"] = cdelt_arcsec
        header["CUNIT1"] = "arcsec"
        header["CUNIT2"] = "arcsec"
        return header

    @staticmethod
    def _session() -> SolRadSession:
        session = SolRadSession.__new__(SolRadSession)
        session.aia = SimpleNamespace(nt=1, times=[SimpleNamespace(mjd=0.0)], shape=(10, 10))
        session.eovsa = Mock()
        session.eovsa.nearest_time_index.return_value = 0
        session.eovsa.mode_data.return_value = np.array(
            [[[0.0, 0.0, 10.0, 10.0]] * 4], dtype=np.float32
        )
        session.eovsa.operation_data.return_value = session.eovsa.mode_data.return_value
        session.eovsa.nfreq = 1
        session.eovsa.shape = (4, 4)
        session.eovsa.files = [object()]
        session.radio_peak_cache = {}
        session.radio_peak_table_cache = {}
        session._overlay_cache = OrderedDict()
        session._eovsa_to_aia_affine = Mock(return_value=np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))
        return session

    def test_global_kelvin_level_does_not_require_global_peaks(self) -> None:
        session = self._session()
        session.eovsa.loaded_global_band_peaks.side_effect = AssertionError("Kelvin levels must not load peak data")

        content = session.eovsa_all_band_contours_on_aia(
            0, 30.0, False, 0.0, 0.0, 50.0, False, 1.0,
            level_reference="global", level_mode="kelvin", level_kelvin=5.0,
        )

        alpha = np.asarray(Image.open(BytesIO(content)).convert("RGBA"))[:, :, 3]
        self.assertGreater(int(alpha.max()), 0)
        session.eovsa.loaded_global_band_peaks.assert_not_called()

    def test_sfu_one_sfu_at_1_4_ghz_and_2_arcsec_is_about_1_8e9_kelvin(self) -> None:
        threshold = _sfu_to_tb_thresholds(
            1.0,
            np.array([1.4e9]),
            self._eovsa_header(),
        )[0]

        self.assertAlmostEqual(threshold, 1.8e9, delta=0.05 * 1.8e9)

    def test_sfu_thresholds_follow_inverse_square_frequency(self) -> None:
        thresholds = _sfu_to_tb_thresholds(
            1.0,
            np.array([1.4e9, 2.8e9]),
            self._eovsa_header(),
        )

        self.assertAlmostEqual(thresholds[0] / thresholds[1], 4.0, places=6)

    def test_sfu_mode_uses_each_band_frequency_without_global_peaks(self) -> None:
        session = self._session()
        session.eovsa.nfreq = 2
        session.eovsa.shape = (4, 4)
        session.eovsa.freqs_hz = np.array([1.4e9, 2.8e9])
        session.eovsa.header = self._eovsa_header()
        session.eovsa.mode_data.return_value = np.full((2, 4, 4), 3.0e9, dtype=np.float32)
        session.eovsa.mode_data.return_value[:, 0, 0] = 0.0
        session.eovsa.loaded_global_band_peaks.side_effect = AssertionError("sfu levels must not load peak data")

        thresholds = []

        def record_threshold(data: np.ndarray, level: float) -> list[object]:
            del data
            thresholds.append(float(level))
            return []

        with unittest.mock.patch(
            "solradviewer.backend.data.measure.find_contours",
            side_effect=record_threshold,
        ):
            session.eovsa_all_band_contours_on_aia(
                0, 30.0, False, 0.0, 0.0, 50.0, False, 1.0,
                level_reference="global", level_mode="sfu", level_sfu=1.0,
                target_panel="eovsa",
            )

        np.testing.assert_allclose(thresholds, _sfu_to_tb_thresholds(1.0, session.eovsa.freqs_hz, session.eovsa.header))
        session.eovsa.loaded_global_band_peaks.assert_not_called()

    def test_global_percent_operation_uses_exact_cached_peaks(self) -> None:
        session = self._session()
        key = session.radio_peak_cache_key(30.0, "none", "subtract", "previous", None, None)
        session.radio_peak_cache[key] = [10.0]
        session.eovsa.loaded_global_operation_peaks.side_effect = AssertionError("Cached rendering must not reload peak data")

        content = session.eovsa_all_band_contours_on_aia(
            0, 30.0, False, 0.0, 0.0, 50.0, False, 1.0,
            level_reference="global", difference_operation="subtract",
            difference_reference="previous",
        )

        alpha = np.asarray(Image.open(BytesIO(content)).convert("RGBA"))[:, :, 3]
        self.assertGreater(int(alpha.max()), 0)
        session.eovsa.loaded_global_operation_peaks.assert_not_called()

    def test_global_percent_missing_peak_entry_computes_exact_table(self) -> None:
        session = self._session()
        session.eovsa.peak_table.return_value = np.array([[10.0]], dtype=np.float32)

        content = session.eovsa_all_band_contours_on_aia(
            0, 30.0, False, 0.0, 0.0, 50.0, False, 1.0,
            level_reference="global", level_mode="percent",
        )

        alpha = np.asarray(Image.open(BytesIO(content)).convert("RGBA"))[:, :, 3]
        self.assertGreater(int(alpha.max()), 0)
        session.eovsa.peak_table.assert_called_once_with(30.0, "none", refresh=True)

    def test_contour_colormap_changes_overlay_colors(self) -> None:
        session = self._session()
        gray = session.eovsa_all_band_contours_on_aia(
            0, 60.0, False, 0.0, 0.0, 50.0, False, 1.0,
            contour_cmap="gray",
        )
        turbo = session.eovsa_all_band_contours_on_aia(
            0, 60.0, False, 0.0, 0.0, 50.0, False, 1.0,
            contour_cmap="turbo",
        )

        gray_pixels = np.asarray(Image.open(BytesIO(gray)).convert("RGBA"))
        turbo_pixels = np.asarray(Image.open(BytesIO(turbo)).convert("RGBA"))
        gray_visible = gray_pixels[gray_pixels[:, :, 3] > 0, :3]
        turbo_visible = turbo_pixels[turbo_pixels[:, :, 3] > 0, :3]
        self.assertTrue(np.all(gray_visible[:, 0] == gray_visible[:, 1]))
        self.assertTrue(np.all(gray_visible[:, 1] == gray_visible[:, 2]))
        self.assertTrue(np.any(turbo_visible[:, 0] != turbo_visible[:, 1]))

    def test_channels_subset_renders_only_requested_bands(self) -> None:
        session = self._session()
        session.eovsa.nfreq = 3
        session.eovsa.mode_data.return_value = np.stack([
            np.array([[0.0, 0.0, 10.0, 10.0]] * 4, dtype=np.float32),
            np.array([[0.0, 0.0, 20.0, 20.0]] * 4, dtype=np.float32),
            np.array([[0.0, 0.0, 30.0, 30.0]] * 4, dtype=np.float32),
        ])
        seen: list[np.ndarray] = []

        def record_band(data: np.ndarray, threshold: float) -> list[object]:
            del threshold
            seen.append(data.copy())
            return []

        with unittest.mock.patch(
            "solradviewer.backend.data.measure.find_contours",
            side_effect=record_band,
        ):
            session.eovsa_all_band_contours_on_aia(
                0, 30.0, False, 0.0, 0.0, 50.0, False, 1.0,
                channels=[1],
            )

        self.assertEqual(len(seen), 1)
        np.testing.assert_array_equal(seen[0], np.array([[0.0, 0.0, 20.0, 20.0]] * 4, dtype=np.float32))

    def test_target_native_grid_and_explicit_context_preserve_legacy_bytes(self) -> None:
        session = self._session()
        legacy = session.eovsa_all_band_contours_on_aia(
            0, 60.0, False, 0.0, 0.0, 50.0, False, 1.0,
            contour_cmap="gray",
        )
        explicit_context = session.eovsa_all_band_contours_on_aia(
            0, 60.0, False, 0.0, 0.0, 50.0, False, 1.0,
            contour_cmap="gray", target_panel="aia",
        )
        session._eovsa_to_aia_affine.reset_mock()
        radio = session.eovsa_all_band_contours_on_aia(
            0, 60.0, False, 0.0, 0.0, 50.0, False, 1.0,
            contour_cmap="gray", target_panel="eovsa", eovsa_index=0,
        )

        self.assertEqual(explicit_context, legacy)
        self.assertEqual(Image.open(BytesIO(legacy)).size, (10, 10))
        self.assertEqual(Image.open(BytesIO(radio)).size, (4, 4))
        self.assertNotEqual(radio, legacy)
        session._eovsa_to_aia_affine.assert_not_called()

    def test_frequency_colormaps_match_frontend_endpoint_fixtures(self) -> None:
        session = self._session()
        bands = np.zeros((2, 4, 6), dtype=np.float32)
        bands[0, :, 2:] = 10.0
        bands[1, :, 4:] = 10.0
        session.eovsa.mode_data.return_value = bands
        session.eovsa.operation_data.return_value = bands
        session.eovsa.nfreq = 2
        session.eovsa.shape = (4, 6)
        fixtures = {
            "parula_r": {(249, 251, 14), (53, 42, 135)},
            "inferno_r": {(252, 255, 164), (0, 0, 4)},
            "viridis_r": {(253, 231, 37), (68, 1, 84)},
            "rdylbu": {(165, 0, 38), (49, 54, 149)},
        }
        for cmap, expected in fixtures.items():
            with self.subTest(cmap=cmap):
                content = session.eovsa_all_band_contours_on_aia(
                    0, 60.0, False, 0.0, 0.0, 50.0, False, 1.0,
                    contour_cmap=cmap, target_panel="eovsa", eovsa_index=0,
                )
                pixels = np.asarray(Image.open(BytesIO(content)).convert("RGBA"))
                visible_colors = {
                    tuple(int(value) for value in pixel[:3])
                    for pixel in pixels.reshape(-1, 4)
                    if pixel[3]
                }
                self.assertTrue(expected.issubset(visible_colors))


if __name__ == "__main__":
    unittest.main()
