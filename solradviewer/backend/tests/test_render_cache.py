"""Tests for the persistent encoded-render cache."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from solradviewer.backend.data import AiaFitsSequence, RenderDiskCache


def _header() -> fits.Header:
    header = fits.Header()
    header["T_OBS"] = "2025-03-28T00:00:00"
    return header


class RenderDiskCacheTest(unittest.TestCase):
    def test_round_trip_updates_stats(self) -> None:
        with TemporaryDirectory() as temporary:
            cache = RenderDiskCache(Path(temporary), max_bytes=100)
            key = cache.make_key({"frame": 3, "cmap": "gray"})
            self.assertIsNone(cache.get(key))
            cache.put(key, b"png-bytes")
            self.assertEqual(cache.get(key), b"png-bytes")
            self.assertEqual(cache.stats(), {
                "hits": 1,
                "misses": 1,
                "bytes": 9,
                "entries": 1,
                "maxBytes": 100,
            })

    def test_render_parameter_change_invalidates_key(self) -> None:
        with TemporaryDirectory() as temporary:
            cache = RenderDiskCache(Path(temporary), max_bytes=100)
            first = cache.make_key({"frame": 3, "vmax": 1.5})
            changed = cache.make_key({"frame": 3, "vmax": 2.0})
            cache.put(first, b"first")
            self.assertNotEqual(first, changed)
            self.assertIsNone(cache.get(changed))

    def test_byte_budget_evicts_least_recent_entry(self) -> None:
        with TemporaryDirectory() as temporary:
            cache = RenderDiskCache(Path(temporary), max_bytes=7)
            oldest = cache.make_key("oldest")
            newest = cache.make_key("newest")
            cache.put(oldest, b"1234")
            os.utime(cache._path(oldest), ns=(1, 1))
            cache.put(newest, b"5678")
            self.assertIsNone(cache.get(oldest))
            self.assertEqual(cache.get(newest), b"5678")
            self.assertLessEqual(cache.stats()["bytes"], 7)

    def test_atomic_write_survives_concurrent_writers(self) -> None:
        with TemporaryDirectory() as temporary:
            cache = RenderDiskCache(Path(temporary), max_bytes=1024)
            key = cache.make_key("shared")
            payloads = [f"payload-{index}".encode() for index in range(16)]
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(lambda payload: cache.put(key, payload), payloads))
            self.assertIn(cache.get(key), payloads)
            self.assertEqual(list(Path(temporary).rglob("*.tmp")), [])
            self.assertEqual(len(list(Path(temporary).rglob("*.png"))), 1)

    def test_texture_disk_hit_survives_new_source_instance(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(np.arange(4, dtype=np.float32).reshape(2, 2), header=_header()),
            ]).writeto(root / "frame.fits")
            cache = RenderDiskCache(root / "cache", max_bytes=1024 * 1024)
            with patch("solradviewer.backend.data.RENDER_DISK_CACHE", cache):
                first_source = AiaFitsSequence(root, pattern="frame.fits")
                expected = first_source.texture(
                    0, 0.0, 3.0, "gray", "linear",
                    use_difference=False, difference_mode="none", difference_operation="none",
                    max_width=2, max_height=2,
                )
                restarted_source = AiaFitsSequence(root, pattern="frame.fits")
                with patch.object(restarted_source, "frame", side_effect=AssertionError("disk miss")):
                    actual = restarted_source.texture(
                        0, 0.0, 3.0, "gray", "linear",
                        use_difference=False, difference_mode="none", difference_operation="none",
                        max_width=2, max_height=2,
                    )
            self.assertEqual(actual, expected)
            self.assertEqual(cache.stats()["hits"], 1)

    def test_source_mtime_change_invalidates_texture(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_path = root / "frame.fits"
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(np.ones((2, 2), dtype=np.float32), header=_header()),
            ]).writeto(frame_path)
            cache = RenderDiskCache(root / "cache", max_bytes=1024 * 1024)
            with patch("solradviewer.backend.data.RENDER_DISK_CACHE", cache):
                first_source = AiaFitsSequence(root, pattern="frame.fits")
                first_source.texture(0, 0.0, 1.0, "gray", "linear", difference_mode="none")
                stat = frame_path.stat()
                os.utime(frame_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                changed_source = AiaFitsSequence(root, pattern="frame.fits")
                with patch.object(changed_source, "frame", wraps=changed_source.frame) as render:
                    changed_source.texture(0, 0.0, 1.0, "gray", "linear", difference_mode="none")
            render.assert_called_once()


if __name__ == "__main__":
    unittest.main()
