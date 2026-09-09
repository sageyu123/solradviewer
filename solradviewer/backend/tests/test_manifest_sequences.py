"""Focused tests for manifest-configured FITS sequence loaders."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
import time

import numpy as np
from astropy.io import fits

from solradviewer.backend.data import (
    AiaFitsSequence,
    DATA_CACHE_LIMIT,
    EOVSA_DATA_CACHE_LIMIT,
    DecodedPlaneStore,
    EovsaSequence,
)


def _solar_header(timestamp: str) -> fits.Header:
    header = fits.Header()
    header["T_OBS"] = timestamp
    header["DATE-OBS"] = timestamp
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


class ManifestSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.decoded_temporary = TemporaryDirectory()
        self.decoded_patch = patch(
            "solradviewer.backend.data.DECODED_PLANE_STORE",
            DecodedPlaneStore(Path(self.decoded_temporary.name), max_bytes=64 * 1024**2),
        )
        self.decoded_patch.start()

    def tearDown(self) -> None:
        self.decoded_patch.stop()
        self.decoded_temporary.cleanup()

    def test_aia_fits_sequence_is_lazy_and_supports_ratio_mean(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index, value in enumerate((1.0, 2.0, 4.0)):
                data = np.full((3, 3), value, dtype=np.float32)
                fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(data, header=_solar_header(f"2025-03-28T00:0{index}:00")),
                ]).writeto(directory / f"aia_131_{index:02d}.fits.gz")

            sequence = AiaFitsSequence(directory, pattern="*131*.fits.gz", wavelength=131)
            self.assertEqual(sequence.nt, 3)
            self.assertEqual(sequence._data_cache, {})
            ratio = sequence.frame(2, difference_operation="ratio", difference_reference="base")
            np.testing.assert_allclose(ratio, 4.0)
            mean_subtract = sequence.frame(1, difference_operation="subtract", difference_reference="mean")
            np.testing.assert_allclose(mean_subtract, -1.0 / 3.0, atol=1e-6)

    def test_aia_data_cache_is_lru_bounded(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for index in range(DATA_CACHE_LIMIT + 1):
                fits.HDUList([
                    fits.PrimaryHDU(),
                    fits.ImageHDU(
                        np.full((1, 1), index, dtype=np.float32),
                        header=_solar_header(f"2025-03-28T00:{index // 60:02d}:{index % 60:02d}"),
                    ),
                ]).writeto(directory / f"aia_cache_{index:02d}.fits")

            sequence = AiaFitsSequence(directory, pattern="aia_cache_*.fits")
            for index in range(DATA_CACHE_LIMIT):
                sequence._read_file(index)
            sequence._read_file(0)
            sequence._read_file(DATA_CACHE_LIMIT)

            self.assertEqual(len(sequence._data_cache), DATA_CACHE_LIMIT)
            self.assertIn(0, sequence._data_cache)
            self.assertNotIn(1, sequence._data_cache)

    def test_aia_inflight_read_opens_each_file_once(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    np.ones((2, 2), dtype=np.float32),
                    header=_solar_header("2025-03-28T00:00:00"),
                ),
            ]).writeto(directory / "aia_once_00.fits")
            sequence = AiaFitsSequence(directory, pattern="aia_once_*.fits")
            original_open = fits.open
            calls = []

            def tracked_open(*args, **kwargs):
                calls.append(args[0])
                time.sleep(0.02)
                return original_open(*args, **kwargs)

            with patch("solradviewer.backend.data.fits.open", side_effect=tracked_open):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(lambda _index: sequence._read_file(0), range(4)))

            self.assertEqual(len(calls), 1)
            self.assertTrue(all(result is results[0] for result in results))

    def test_eovsa_manifest_pattern_overrides_legacy_glob(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="cfreqs", format="D", array=np.array([1.0e9, 2.0e9])),
                fits.Column(name="cdelts", format="D", array=np.array([1.0e7, 1.0e7])),
            ])
            for index, value in enumerate((1.0, 2.0)):
                image = fits.ImageHDU(np.full((2, 2, 2), value, dtype=np.float32), header=_solar_header(f"2025-03-28T00:0{index}:00"))
                fits.HDUList([fits.PrimaryHDU(), image, table]).writeto(directory / f"custom_{index:02d}.fits")

            sequence = EovsaSequence(directory, pattern="custom_*.fits")
            self.assertEqual(len(sequence.files), 2)
            np.testing.assert_allclose(sequence.operation_data(1, 60, "subtract", "base"), 1.0)

    def test_eovsa_lazy_data_cache_is_bounded(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="cfreqs", format="D", array=np.array([1.0e9])),
                fits.Column(name="cdelts", format="D", array=np.array([1.0e7])),
            ])
            for index in range(EOVSA_DATA_CACHE_LIMIT + 1):
                image = fits.ImageHDU(
                    np.full((1, 1, 1), index, dtype=np.float32),
                    header=_solar_header(f"2025-03-28T00:00:{index:02d}"),
                )
                fits.HDUList([fits.PrimaryHDU(), image, table]).writeto(directory / f"cache_{index:02d}.fits")

            sequence = EovsaSequence(directory, pattern="cache_*.fits")
            for index in range(EOVSA_DATA_CACHE_LIMIT):
                sequence._read_file(index)
            sequence._read_file(0)
            sequence._read_file(EOVSA_DATA_CACHE_LIMIT)

            self.assertEqual(len(sequence._data_cache), EOVSA_DATA_CACHE_LIMIT)
            self.assertNotIn(1, sequence._data_cache)
            self.assertIn(0, sequence._data_cache)
            self.assertIn(EOVSA_DATA_CACHE_LIMIT, sequence._data_cache)

    def test_eovsa_single_band_reads_match_full_operations(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="cfreqs", format="D", array=np.array([1.0e9, 2.0e9, 3.0e9])),
                fits.Column(name="cdelts", format="D", array=np.array([1.0e7, 1.0e7, 1.0e7])),
            ])
            for index, offset in enumerate((1.0, 2.0, 4.0)):
                values = np.arange(3 * 3 * 3, dtype=np.float32).reshape(3, 3, 3) + offset
                image = fits.CompImageHDU(
                    values,
                    header=_solar_header(f"2025-03-28T00:0{index}:00"),
                    compression_type="RICE_1",
                    tile_shape=(1, 3, 3),
                )
                fits.HDUList([fits.PrimaryHDU(), image, table]).writeto(directory / f"band_{index:02d}.fits")

            sequence = EovsaSequence(directory, pattern="band_*.fits")
            time_index, freq_index = 2, 1
            np.testing.assert_array_equal(
                sequence._read_band(time_index, freq_index),
                sequence._read_file(time_index)[freq_index],
            )
            for mode in ("none", "base", "running"):
                np.testing.assert_array_equal(
                    sequence._band_mode_data(time_index, freq_index, 60.0, mode),
                    sequence.mode_data(time_index, 60.0, mode)[freq_index],
                )
            for operation, reference in (("none", "previous"), ("subtract", "base"), ("subtract", "previous"), ("ratio", "base"), ("ratio", "previous"), ("subtract", "mean"), ("ratio", "mean")):
                kwargs = {"mean_start_mjd": sequence.times.mjd[0], "mean_end_mjd": sequence.times.mjd[-1]} if reference == "mean" else {}
                np.testing.assert_array_equal(
                    sequence._band_operation_data(time_index, freq_index, 60.0, operation, reference, **kwargs),
                    sequence.operation_data(time_index, 60.0, operation, reference, **kwargs)[freq_index],
                )

    def test_eovsa_full_cube_hint_selects_full_data_path(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="cfreqs", format="D", array=np.array([1.0e9, 2.0e9])),
                fits.Column(name="cdelts", format="D", array=np.array([1.0e7, 1.0e7])),
            ])
            for index in range(2):
                image = fits.ImageHDU(
                    np.full((2, 2, 2), index + 1, dtype=np.float32),
                    header=_solar_header(f"2025-03-28T00:0{index}:00"),
                )
                fits.HDUList([fits.PrimaryHDU(), image, table]).writeto(directory / f"hint_{index:02d}.fits")

            sequence = EovsaSequence(directory, pattern="hint_*.fits")
            mjd = float(sequence.times.mjd[1])
            sequence.texture_for_aia_time(mjd, 1, 60.0, -1.0, 3.0, "gray", "linear", difference_mode="none")
            self.assertEqual(len(sequence._data_cache), 1)
            self.assertIn((1, 1), sequence._band_cache)

            sequence._texture_cache.clear()
            sequence._band_cache.clear()
            sequence._data_cache.clear()
            with patch("solradviewer.backend.data.fits.open", wraps=fits.open) as open_mock:
                sequence.texture_for_aia_time(mjd, 1, 60.0, -1.0, 3.0, "gray", "linear", difference_mode="none", full_cube=True)
                sequence.mode_data(1, 60.0, "none")
            self.assertEqual(open_mock.call_count, 0)
            self.assertIn(1, sequence._data_cache)

    def test_eovsa_inflight_read_opens_each_file_once(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="cfreqs", format="D", array=np.array([1.0e9])),
                fits.Column(name="cdelts", format="D", array=np.array([1.0e7])),
            ])
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(np.ones((1, 2, 2), dtype=np.float32), header=_solar_header("2025-03-28T00:00:00")),
                table,
            ]).writeto(directory / "once_00.fits")
            sequence = EovsaSequence(directory, pattern="once_*.fits")
            original_open = fits.open
            calls = []

            def tracked_open(*args, **kwargs):
                calls.append(args[0])
                time.sleep(0.02)
                return original_open(*args, **kwargs)

            with patch("solradviewer.backend.data.fits.open", side_effect=tracked_open):
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(lambda _index: sequence._read_file(0), range(4)))

            self.assertEqual(len(calls), 1)
            self.assertTrue(all(result is results[0] for result in results))


if __name__ == "__main__":
    unittest.main()
