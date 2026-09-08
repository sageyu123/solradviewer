"""Smoke tests for the default AIA/EOVSA workflow."""

from __future__ import annotations

import numpy as np
import unittest

from sad_eovsa_tool.backend.data import DEFAULT_AIA_DIFF, DEFAULT_EOVSA_DIR, DEFAULT_SEEDS, SadEovsaSession


DATA_AVAILABLE = DEFAULT_AIA_DIFF.exists() and DEFAULT_EOVSA_DIR.exists() and DEFAULT_SEEDS.exists()


class WorkflowSmokeTest(unittest.TestCase):
    @unittest.skipUnless(DATA_AVAILABLE, "Default 2022-01-18 data products are not available on this machine.")
    def test_default_session_renders_and_extracts_short_window(self) -> None:
        session = SadEovsaSession.create_default()
        meta = session.api_meta()
        self.assertIn("sources", meta)
        self.assertEqual([source["role"] for source in meta["sources"][:3]], ["context", "radio", "spectrogram"])
        self.assertIn("pixelToWorldAffine", meta["wcs"]["aia"])
        self.assertIn("pixelToWorldAffine", meta["wcs"]["eovsa"])
        self.assertEqual(meta["spectrogram"]["shape"], [451, 4198])
        start = int(meta["defaults"]["timeStartIndex"])
        end = start + 8
        freq = int(meta["defaults"]["freqIndex"])

        self.assertGreater(len(session.aia.texture(start, 0.5, 1.5, "gray", "linear")), 1000)
        self.assertGreater(len(session.aia.texture(start, -100, 100, "gray", "linear", difference_mode="base")), 1000)
        self.assertGreater(
            len(session.eovsa.texture_for_aia_time(session.aia.times[start].mjd, freq, 30, -1e6, 5e6, "turbo", "linear")),
            1000,
        )
        self.assertGreater(
            len(session.eovsa.texture_for_aia_time(session.aia.times[start].mjd, freq, 30, -1e6, 5e6, "turbo", "linear", difference_mode="base")),
            1000,
        )
        self.assertGreater(len(session.spectrogram.texture()), 1000)

        source = session.add_source_spec({"role": "context", "label": "Placeholder FITS", "format": "fits", "path": "/tmp/context.fits"})
        self.assertEqual(source["status"], "placeholder")
        self.assertIn("Placeholder FITS", [item["label"] for item in session.api_meta()["sources"]])
        original_files = session.eovsa.files
        try:
            session.eovsa.files = original_files[:1]
            key, peaks = session.radio_global_peaks(30, "none", refresh=True)
        finally:
            session.eovsa.files = original_files
            session.eovsa._cube_cache = None
            session.eovsa._peak_table_cache.clear()
            session.eovsa._data_cache.clear()
        self.assertEqual(len(peaks), session.eovsa.nfreq)
        self.assertIn(key, session.api_meta()["radioPeakCache"])
        self.assertIn(key, session.api_meta()["radioPeakTableCache"])

        roi_world = np.array([[740, 115], [850, 115], [850, 245], [740, 245]], dtype=float)
        roi_pix = session.aia.world_to_pixel(start, roi_world).tolist()
        session.set_roi_from_pixels("aia", roi_pix, start, freq, 7, 0, 30)

        sad_rows = session.extract_sads()
        sad_count = len(sad_rows)
        sad_rows = session.delete_track_point(0)
        eovsa_rows = session.extract_eovsa_sources(7, 0, 30, start, end, stride=20, min_snr=5)

        self.assertTrue(sad_rows)
        self.assertEqual(len(sad_rows), sad_count - 1)
        self.assertTrue(eovsa_rows)
        self.assertTrue((session.output_dir / "sad_tracks.csv").exists())
        self.assertTrue((session.output_dir / "feature_tracks.csv").exists())
        self.assertTrue((session.output_dir / "eovsa_sources.csv").exists())
        self.assertTrue((session.output_dir / "radio_sources.csv").exists())
        self.assertTrue((session.output_dir / "aia_sad_track_map.png").exists())
        self.assertTrue((session.output_dir / "eovsa_source_time_map.png").exists())
