"""Exact scalar PNG colormap contract tests."""

from __future__ import annotations

from io import BytesIO
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from solradviewer.backend.data import CMAP_ALIASES, _render_png, _resolve_colormap


def _endpoint_rgb(cmap: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    content = _render_png(np.array([[0.0, 1.0]]), 0.0, 1.0, cmap)
    pixels = np.asarray(Image.open(BytesIO(content)).convert("RGB"))[0]
    return tuple(int(value) for value in pixels[0]), tuple(int(value) for value in pixels[1])


class ScalarColormapTest(unittest.TestCase):
    def test_grayscale_png_uses_luminance_mode_and_fast_compression(self) -> None:
        with patch("PIL.Image.Image.save", autospec=True) as save_mock:
            _render_png(np.array([[0.0, np.nan, 1.0]]), 0.0, 1.0, "gray_r")

        image = save_mock.call_args.args[0]
        self.assertEqual(image.mode, "L")
        self.assertEqual(save_mock.call_args.kwargs["compress_level"], 1)

    def test_instrument_aia_aliases_resolve_to_registered_sunpy_maps(self) -> None:
        expected = {
            "aia94": "sdoaia94",
            "aia131": "sdoaia131",
            "aia171": "sdoaia171",
            "aia193": "sdoaia193",
            "aia211": "sdoaia211",
            "aia304": "sdoaia304",
            "aia335": "sdoaia335",
        }
        for alias, registered in expected.items():
            with self.subTest(alias=alias):
                self.assertEqual(CMAP_ALIASES[alias], registered)
                self.assertEqual(_resolve_colormap(alias, "gray")[0], registered)

    def test_parula_uses_exact_canonical_scalar_endpoints(self) -> None:
        self.assertEqual(_endpoint_rgb("parula"), ((53, 42, 135), (249, 251, 14)))

    def test_case_and_legacy_aliases_resolve_to_canonical_scalar_maps(self) -> None:
        aliases = {
            "Parula": ((53, 42, 135), (249, 251, 14)),
            "Inferno": ((0, 0, 3), (252, 254, 164)),
            "RdYlBu": ((165, 0, 38), (49, 54, 149)),
            "RdYlBu_r": ((165, 0, 38), (49, 54, 149)),
        }
        for alias, endpoints in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(_endpoint_rgb(alias), endpoints)

    def test_frequency_backend_ids_are_warm_low_and_cool_high(self) -> None:
        frequency_maps = {
            "parula_r": ((249, 251, 14), (53, 42, 135)),
            "inferno_r": ((252, 254, 164), (0, 0, 3)),
            "viridis_r": ((253, 231, 36), (68, 1, 84)),
            "rdylbu": ((165, 0, 38), (49, 54, 149)),
        }
        for cmap, endpoints in frequency_maps.items():
            with self.subTest(cmap=cmap):
                self.assertEqual(_endpoint_rgb(cmap), endpoints)

    def test_existing_scalar_maps_retain_matplotlib_endpoints(self) -> None:
        existing = {
            "gray": ((0, 0, 0), (255, 255, 255)),
            "gray_r": ((255, 255, 255), (0, 0, 0)),
            "viridis": ((68, 1, 84), (253, 231, 36)),
            "turbo": ((48, 18, 59), (122, 4, 2)),
            "magma": ((0, 0, 3), (251, 252, 191)),
            "coolwarm": ((58, 76, 192), (179, 3, 38)),
        }
        for cmap, endpoints in existing.items():
            with self.subTest(cmap=cmap):
                self.assertEqual(_endpoint_rgb(cmap), endpoints)


if __name__ == "__main__":
    unittest.main()
