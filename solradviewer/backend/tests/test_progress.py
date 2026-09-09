"""Progress registry and long-running operation reporting tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event as ThreadEvent, Lock
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from solradviewer.backend import app as api, data
from solradviewer.backend.data import ProgressRegistry, SolRadSession


class ProgressRegistryTest(unittest.TestCase):
    def test_manifest_session_keeps_load_operation_handle_until_finished(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(data, "DEFAULT_OUTPUT_ROOT", Path(temp_dir)),
                patch.object(data, "AiaCube", return_value=SimpleNamespace()),
                patch.object(data, "EovsaSequence", return_value=SimpleNamespace(nfreq=2)),
            ):
                session = SolRadSession.create_from_manifest({})

        operation = session.progress_registry.snapshot()[0]
        self.assertEqual(operation["label"], "Loading session")
        self.assertEqual((operation["done"], operation["total"]), (3, 5))
        self.assertIsNotNone(session._load_progress_id)

        session.update_load_progress(5)
        session.finish_load_progress()

        self.assertEqual(session.progress_registry.snapshot(), [])
        self.assertIsNone(session._load_progress_id)

    def test_registry_updates_are_thread_safe_and_snapshots_are_detached(self) -> None:
        registry = ProgressRegistry()
        op_id = registry.start("Scanning", total=400)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(lambda _: registry.advance(op_id), range(400)))

        snapshot = registry.snapshot()
        self.assertEqual(snapshot[0]["opId"], op_id)
        self.assertEqual(snapshot[0]["done"], 400)
        snapshot[0]["done"] = -1
        self.assertEqual(registry.snapshot()[0]["done"], 400)
        registry.finish(op_id)
        self.assertEqual(registry.snapshot(), [])

    def test_progress_endpoint_returns_active_operations_and_prewarm_shape(self) -> None:
        registry = ProgressRegistry()
        op_id = registry.start("Computing global peaks", total=12)
        registry.update(op_id, done=3)
        completed_id = registry.start("Completed operation", total=2)
        registry.update(completed_id, done=2)
        session = SimpleNamespace(
            progress_registry=registry,
            prewarm_status=lambda: {"done": 0, "total": 0, "active": False},
        )
        api.SESSIONS["progress-test"] = session  # type: ignore[assignment]
        try:
            response = TestClient(api.app).get("/api/sessions/progress-test/progress")
        finally:
            api.SESSIONS.pop("progress-test", None)

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["prewarm"], {"done": 0, "total": 0, "active": False})
        self.assertEqual(len(payload["operations"]), 1)
        self.assertEqual(payload["operations"][0]["opId"], op_id)
        self.assertEqual(
            set(payload["operations"][0]),
            {"opId", "label", "done", "total", "startedAt"},
        )

    def test_radio_global_peak_computation_reports_each_scanned_frame(self) -> None:
        registry = ProgressRegistry()
        snapshots: list[tuple[int, int | None]] = []

        class _Radio:
            files = ["one", "two", "three"]

            def peak_table(
                self,
                diff_seconds: float,
                difference_mode: str,
                refresh: bool = False,
                on_progress=None,
            ) -> np.ndarray:
                del diff_seconds, difference_mode, refresh
                on_progress = on_progress or self._peak_progress_local.callback
                for done in range(1, 4):
                    on_progress(done, 3)
                    operation = registry.snapshot()[0]
                    snapshots.append((operation["done"], operation["total"]))
                return np.asarray([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], dtype=np.float32)

        fake = SimpleNamespace(
            eovsa=_Radio(),
            progress_registry=registry,
            radio_peak_cache_key=SolRadSession.radio_peak_cache_key,
            radio_peak_cache={},
            radio_peak_table_cache={},
            _overlay_cache={},
        )
        key, peaks = SolRadSession.radio_global_peaks(fake, 61.0, "running", refresh=True)

        self.assertEqual(key, "running:dt=61.000")
        self.assertEqual(peaks, [3.0, 4.0])
        self.assertEqual(snapshots, [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(registry.snapshot(), [])

    def test_overlapping_peak_refreshes_share_one_computation(self) -> None:
        registry = ProgressRegistry()
        computation_started = ThreadEvent()
        release_computation = ThreadEvent()
        waiter_started = ThreadEvent()

        class _TrackingEvent:
            def __init__(self) -> None:
                self._event = ThreadEvent()

            def wait(self) -> None:
                waiter_started.set()
                self._event.wait()

            def set(self) -> None:
                self._event.set()

        class _Radio:
            files = ["one", "two"]
            calls = 0

            def peak_table(self, *args, **kwargs) -> np.ndarray:
                del args, kwargs
                self.calls += 1
                computation_started.set()
                release_computation.wait()
                callback = self._peak_progress_local.callback
                callback(2, 2)
                return np.asarray([[1.0], [2.0]], dtype=np.float32)

        radio = _Radio()
        fake = SimpleNamespace(
            eovsa=radio,
            progress_registry=registry,
            radio_peak_cache_key=SolRadSession.radio_peak_cache_key,
            radio_peak_cache={},
            radio_peak_table_cache={},
            _overlay_cache={},
            _radio_peak_build_lock=Lock(),
            _radio_peak_inflight={},
        )
        with patch("solradviewer.backend.data.Event", _TrackingEvent):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(SolRadSession.radio_global_peaks, fake, 62.0, "running", True)
                self.assertTrue(computation_started.wait(1.0))
                second = executor.submit(SolRadSession.radio_global_peaks, fake, 62.0, "running", True)
                self.assertTrue(waiter_started.wait(1.0))
                release_computation.set()
                self.assertEqual(first.result(timeout=1.0), second.result(timeout=1.0))

        self.assertEqual(radio.calls, 1)
        self.assertEqual(registry.snapshot(), [])


if __name__ == "__main__":
    unittest.main()
