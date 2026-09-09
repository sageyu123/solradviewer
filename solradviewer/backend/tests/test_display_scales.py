"""Tests for display intensity scale normalization."""

from __future__ import annotations

import unittest

import numpy as np

from solradviewer.backend.data import _normalize_display_values


class DisplayScaleTest(unittest.TestCase):
    def test_sqrt_and_asinh_normalization_are_monotonic_and_bounded(self) -> None:
        values = np.linspace(-5.0, 15.0, 101)
        for scale in ("sqrt", "asinh"):
            with self.subTest(scale=scale):
                normalized = _normalize_display_values(values, 0.0, 10.0, scale)
                self.assertTrue(np.all(np.diff(normalized) >= 0.0))
                self.assertGreaterEqual(float(np.min(normalized)), 0.0)
                self.assertLessEqual(float(np.max(normalized)), 1.0)
                self.assertAlmostEqual(float(normalized[25]), 0.0)
                self.assertAlmostEqual(float(normalized[75]), 1.0)


if __name__ == "__main__":
    unittest.main()
