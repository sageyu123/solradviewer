"""Integration tests for P4 target-aware contour and panel addressing."""

from __future__ import annotations

from io import BytesIO
import unittest

from astropy.time import Time
from fastapi.testclient import TestClient
from PIL import Image

from sad_eovsa_tool.backend import app as api


class _P4Axis:
    def __init__(self, seconds: list[float]) -> None:
        self.times = Time([60000.0 + value / 86400.0 for value in seconds], format="mjd")


class _P4Session:
    context_source_id = "aia-131"
    radio_source_id = "eovsa-1s"
    spectrogram_source_id = "spectrogram"

    def __init__(self) -> None:
        self.aia = _P4Axis([0.0, 12.0])
        self.eovsa = _P4Axis([1.0, 7.0, 25.0])
        self.contour_calls: list[tuple[int, str, int | None]] = []
        self.contour_channel_subsets: list[list[int] | None] = []
        self.roi_calls: list[tuple[str, int, int, int | None]] = []

    def eovsa_all_band_contours_on_aia(
        self,
        time_index: int,
        *args: object,
        eovsa_index: int | None = None,
        target_panel: str = "aia",
        **kwargs: object,
    ) -> bytes:
        self.contour_channel_subsets.append(kwargs.get("channels"))
        del args, kwargs
        self.contour_calls.append((int(time_index), target_panel, eovsa_index))
        size = (5, 4) if target_panel == "eovsa" else (9, 8)
        out = BytesIO()
        Image.new("RGBA", size, (255, 0, 0, 255)).save(out, format="PNG")
        return out.getvalue()

    def set_roi_from_pixels(
        self,
        panel: str,
        points: list[list[float]],
        time_index: int,
        freq_index: int,
        *args: object,
    ) -> dict[str, object]:
        self.roi_calls.append((panel, int(time_index), int(freq_index), args[-1] if args else None))
        return {"roiWorld": points}

    def roi_pixels_for_panel(
        self,
        panel: str,
        time_index: int,
        freq_index: int,
        *args: object,
    ) -> list[list[float]]:
        self.roi_calls.append((panel, int(time_index), int(freq_index), args[-1] if args else None))
        return [[1.0, 2.0]]


class P4BackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.session = _P4Session()
        api.SESSIONS["p4-test"] = self.session  # type: ignore[assignment]
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        api.SESSIONS.pop("p4-test", None)

    def test_contour_recipe_targets_context_layers_and_preserves_legacy_bytes(self) -> None:
        sample_mjd = float(self.session.eovsa.times[0].mjd)
        canonical = self.client.get(
            "/api/sessions/p4-test/sources/eovsa-1s/overlay-contours.png",
            params={"targetSourceId": "aia-131", "sampleMjd": sample_mjd},
        )
        context_alias = self.client.get(
            "/api/sessions/p4-test/sources/eovsa-1s/overlay-contours.png",
            params={"targetSourceId": "context", "sampleMjd": sample_mjd},
        )
        legacy = self.client.get(
            "/api/sessions/p4-test/eovsa/aia-contours.png",
            params={"timeIndex": 0},
        )

        self.assertEqual(canonical.status_code, 200)
        self.assertEqual(context_alias.content, canonical.content)
        self.assertEqual(legacy.content, canonical.content)
        self.assertEqual(Image.open(BytesIO(canonical.content)).size, (9, 8))
        self.assertTrue(all(call[1] == "aia" for call in self.session.contour_calls))

    def test_radio_target_uses_native_grid_without_aia_availability(self) -> None:
        sample_mjd = float(self.session.eovsa.times[2].mjd)
        radio = self.client.get(
            "/api/sessions/p4-test/sources/eovsa-1s/overlay-contours.png",
            params={"targetSourceId": "eovsa-1s", "sampleMjd": sample_mjd},
        )
        unavailable_context = self.client.get(
            "/api/sessions/p4-test/sources/eovsa-1s/overlay-contours.png",
            params={"targetSourceId": "aia-131", "sampleMjd": sample_mjd},
        )

        self.assertEqual(radio.status_code, 200)
        self.assertEqual(radio.headers["X-Resolved-Index"], "2")
        self.assertEqual(Image.open(BytesIO(radio.content)).size, (5, 4))
        self.assertEqual(self.session.contour_calls[-1], (2, "eovsa", 2))
        self.assertEqual(unavailable_context.status_code, 204)
        self.assertNotIn("X-Resolved-Index", unavailable_context.headers)

    def test_contour_rejects_unknown_target_source(self) -> None:
        response = self.client.get(
            "/api/sessions/p4-test/sources/eovsa-1s/overlay-contours.png",
            params={"targetSourceId": "not-a-target", "sampleMjd": float(self.session.eovsa.times[0].mjd)},
        )
        self.assertEqual(response.status_code, 422)

    def test_contour_channels_query_forwards_subset(self) -> None:
        response = self.client.get(
            "/api/sessions/p4-test/sources/eovsa-1s/overlay-contours.png",
            params={
                "targetSourceId": "aia-131",
                "sampleMjd": float(self.session.eovsa.times[0].mjd),
                "channels": "2, 0, 2",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.session.contour_channel_subsets[-1], [2, 0, 2])

    def test_contour_overlay_drops_masked_channels_from_effective_band_list(self) -> None:
        self.session.channel_mask = [False, True, False]
        response = self.client.get(
            "/api/sessions/p4-test/sources/eovsa-1s/overlay-contours.png",
            params={"targetSourceId": "aia-131", "sampleMjd": float(self.session.eovsa.times[0].mjd)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.session.contour_channel_subsets[-1], [0, 2])

    def test_roi_source_identity_overrides_legacy_panel_slot_and_keeps_frequency(self) -> None:
        radio_roi = self.client.post(
            "/api/sessions/p4-test/roi",
            json={
                "panel": "aia",
                "sourceId": "eovsa-1s",
                "sampleMjd": float(self.session.eovsa.times[1].mjd),
                "points": [[0, 0], [1, 0], [1, 1]],
                "freqIndex": 3,
            },
        )
        context_projection = self.client.post(
            "/api/sessions/p4-test/roi/projection",
            json={
                "panel": "eovsa",
                "sourceId": "aia-131",
                "sampleMjd": float(self.session.aia.times[1].mjd),
                "freqIndex": 2,
            },
        )

        self.assertEqual(radio_roi.status_code, 200)
        self.assertEqual(context_projection.status_code, 200)
        self.assertEqual(self.session.roi_calls[0], ("eovsa", 1, 3, 1))
        self.assertEqual(self.session.roi_calls[1], ("aia", 1, 2, None))


if __name__ == "__main__":
    unittest.main()
