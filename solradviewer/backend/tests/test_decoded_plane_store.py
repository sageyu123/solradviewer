"""Focused tests for the persistent decoded-array store."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from solradviewer.backend.data import AiaFitsSequence, DecodedPlaneStore, EovsaSequence


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


class DecodedPlaneStoreTest(unittest.TestCase):
    """Verify memmap round trips, invalidation, eviction, and source reuse."""

    def test_round_trip_is_memmapped_and_bit_identical(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.fits"
            source.write_bytes(b"source")
            store = DecodedPlaneStore(root / "decoded", max_bytes=1024 * 1024)
            key = store.source_key(source, {"hdu": 1})
            original = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            stored = store.put(key, original)
            reopened = store.get(key, original.shape)
            self.assertIsInstance(stored, np.memmap)
            self.assertIsInstance(reopened, np.memmap)
            assert reopened is not None
            self.assertEqual(hashlib.sha256(original.tobytes()).hexdigest(), hashlib.sha256(reopened.tobytes()).hexdigest())
            self.assertEqual(store.stats()["entries"], 1)
            self.assertEqual(list((root / "decoded").rglob("*.tmp")), [])

    def test_source_signature_changes_with_size_or_mtime(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.fits"
            source.write_bytes(b"one")
            first = DecodedPlaneStore.source_key(source)
            stat = source.stat()
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            changed_mtime = DecodedPlaneStore.source_key(source)
            source.write_bytes(b"longer")
            changed_size = DecodedPlaneStore.source_key(source)
            self.assertEqual(len({first, changed_mtime, changed_size}), 3)

    def test_byte_budget_evicts_least_recent_array(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DecodedPlaneStore(root, max_bytes=400)
            keys = [f"{index:064x}" for index in range(3)]
            store.put(keys[0], np.arange(10, dtype=np.float32))
            store.put(keys[1], np.arange(10, dtype=np.float32))
            self.assertIsNotNone(store.get(keys[1]))
            store.put(keys[2], np.arange(10, dtype=np.float32))
            self.assertIsNone(store.get(keys[0]))
            self.assertIsNotNone(store.get(keys[1]))
            self.assertIsNotNone(store.get(keys[2]))
            self.assertLessEqual(store.stats()["bytes"], 400)

    def test_concurrent_writers_leave_one_complete_array(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DecodedPlaneStore(root, max_bytes=1024 * 1024)
            key = "a" * 64
            values = [np.full((8, 8), index, dtype=np.float32) for index in range(8)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(lambda value: store.put(key, value), values))
            result = store.get(key)
            assert result is not None
            self.assertTrue(any(np.array_equal(result, value) for value in values))
            self.assertEqual(len(list(root.rglob("*.npy"))), 1)
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_radio_and_aia_sources_reuse_store_after_restart(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DecodedPlaneStore(root / "decoded", max_bytes=16 * 1024**2)
            table = fits.BinTableHDU.from_columns([
                fits.Column(name="cfreqs", format="D", array=np.array([1.0e9, 2.0e9])),
                fits.Column(name="cdelts", format="D", array=np.array([1.0e7, 1.0e7])),
            ])
            radio_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.CompImageHDU(radio_values, header=_solar_header("2025-03-28T00:00:00"), compression_type="RICE_1"),
                table,
            ]).writeto(root / "radio_00.fits")
            aia_values = np.arange(12, dtype=np.float32).reshape(3, 4)
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(aia_values, header=_solar_header("2025-03-28T00:00:00")),
            ]).writeto(root / "aia_00.fits")

            with patch("solradviewer.backend.data.DECODED_PLANE_STORE", store):
                radio_first = EovsaSequence(root, pattern="radio_*.fits")._read_file(0)
                aia_first = AiaFitsSequence(root, pattern="aia_*.fits")._read_file(0)
                radio_restarted = EovsaSequence(root, pattern="radio_*.fits")
                aia_restarted = AiaFitsSequence(root, pattern="aia_*.fits")
                with patch("solradviewer.backend.data.fits.open", side_effect=AssertionError("decoded-store miss")):
                    radio_second = radio_restarted._read_file(0)
                    aia_second = aia_restarted._read_file(0)

            self.assertIsInstance(radio_second, np.ndarray)
            self.assertIsInstance(aia_second, np.ndarray)
            self.assertNotIsInstance(radio_second, np.memmap)
            self.assertNotIsInstance(aia_second, np.memmap)
            np.testing.assert_array_equal(radio_second, radio_first)
            np.testing.assert_array_equal(aia_second, aia_first)
            self.assertEqual(store.stats()["entries"], 2)


if __name__ == "__main__":
    unittest.main()
