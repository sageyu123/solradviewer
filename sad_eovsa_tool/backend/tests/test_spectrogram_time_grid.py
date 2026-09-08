"""Tests for elapsed-time regularization of dynamic-spectrum textures."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from astropy.io import fits
from astropy.time import Time
from PIL import Image

from sad_eovsa_tool.backend.data import EovsaSpectrogram, _regularize_spectrogram_time_axis


class SpectrogramTimeGridTest(unittest.TestCase):
    def test_gap_columns_remain_nan_on_uniform_elapsed_grid(self) -> None:
        times = 60000.0 + np.array([0.0, 1.0, 600.0, 601.0]) / 86400.0
        data = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)

        regularized, grid = _regularize_spectrogram_time_axis(data, times)

        self.assertEqual(regularized.shape, (2, 602))
        np.testing.assert_allclose(np.diff(grid), 1.0 / 86400.0, rtol=1e-5)
        self.assertTrue(np.all(np.isfinite(regularized[:, :2])))
        self.assertTrue(np.all(np.isfinite(regularized[:, -2:])))
        self.assertTrue(np.all(np.isnan(regularized[:, 2:-2])))

    def test_texture_encodes_gap_as_transparent_pixels(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "gap.fits"
            mjd = 60000.0 + np.array([0.0, 1.0, 600.0, 601.0]) / 86400.0
            table_freq = fits.BinTableHDU.from_columns([
                fits.Column(name="FGHZ", format="D", array=np.array([1.0, 2.0])),
            ])
            table_time = fits.BinTableHDU.from_columns([
                fits.Column(name="TIME", format="D", array=Time(mjd, format="mjd").jd),
            ])
            fits.HDUList([
                fits.PrimaryHDU(np.ones((2, 4), dtype=np.float32)),
                table_freq,
                table_time,
            ]).writeto(path)

            spectrogram = EovsaSpectrogram(path)
            image = Image.open(BytesIO(spectrogram.texture(vmin=0.0, vmax=5.0))).convert("RGBA")
            alpha = np.asarray(image)[..., 3]

            self.assertEqual(image.size, (602, 2))
            self.assertTrue(np.all(alpha[:, 0] == 255))
            self.assertTrue(np.all(alpha[:, 300] == 0))
            self.assertTrue(np.all(alpha[:, -1] == 255))


if __name__ == "__main__":
    unittest.main()
