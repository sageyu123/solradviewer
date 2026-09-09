"""Tests for dynamic-spectrum frequency-axis rendering."""

from __future__ import annotations

import unittest

import numpy as np

from solradviewer.backend.data import _resample_frequency_axis


class SpectrogramFrequencyScaleTest(unittest.TestCase):
    def test_linear_and_log_frequency_grids_are_distinct(self) -> None:
        data = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
        frequencies = np.array([1.0, 2.0, 4.0])

        linear = _resample_frequency_axis(data, frequencies, "linear")
        logarithmic = _resample_frequency_axis(data, frequencies, "log")

        np.testing.assert_allclose(linear[:, 0], [0.0, 12.5, 20.0])
        np.testing.assert_allclose(logarithmic[:, 0], [0.0, 10.0, 20.0])

    def test_descending_input_frequencies_are_sorted(self) -> None:
        data = np.array([[20], [10], [0]], dtype=np.float32)
        frequencies = np.array([4.0, 2.0, 1.0])

        result = _resample_frequency_axis(data, frequencies, "log")

        np.testing.assert_allclose(result[:, 0], [0.0, 10.0, 20.0])

    def test_frequency_window_is_resampled_to_the_full_texture(self) -> None:
        data = np.array([[0], [10], [20]], dtype=np.float32)
        frequencies = np.array([1.0, 2.0, 4.0])

        result = _resample_frequency_axis(data, frequencies, "linear", 2.0, 4.0)

        np.testing.assert_allclose(result[:, 0], [10.0, 15.0, 20.0])

    def test_frequency_window_requires_increasing_bounds(self) -> None:
        data = np.array([[0], [10], [20]], dtype=np.float32)
        frequencies = np.array([1.0, 2.0, 4.0])

        with self.assertRaisesRegex(ValueError, "maximum"):
            _resample_frequency_axis(data, frequencies, "linear", 3.0, 3.0)


if __name__ == "__main__":
    unittest.main()
