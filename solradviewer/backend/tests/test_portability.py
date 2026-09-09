"""Tests for environment-based paths used by SolRadViewer."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from solradviewer.backend import data


class PortabilityConfigurationTest(unittest.TestCase):
    def test_new_data_root_setting_takes_precedence_and_expands_home(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SOLRADVIEWER_DATA_ROOT": "~/science/solradviewer",
            },
            clear=True,
        ):
            resolved = data._configured_path(
                data.DEFAULT_DATA_ROOT,
                "SOLRADVIEWER_DATA_ROOT",
            )

        self.assertEqual(resolved, Path.home() / "science/solradviewer")

    def test_cache_byte_settings_accept_new_and_legacy_names(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SOLRADVIEWER_RENDER_CACHE_BYTES": "1234",
                "SAD_EOVSA_RENDER_CACHE_BYTES": "5678",
            },
            clear=True,
        ):
            self.assertEqual(data._configured_render_cache_bytes(), 1234)

        with patch.dict(os.environ, {"SAD_EOVSA_RENDER_CACHE_BYTES": "5678"}, clear=True):
            self.assertEqual(data._configured_render_cache_bytes(), 5678)

    def test_default_cache_directory_is_user_local_and_configurable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            default = data._configured_path(
                data.DEFAULT_CACHE_ROOT,
                "SOLRADVIEWER_CACHE_DIR",
                "SAD_EOVSA_RENDER_CACHE_DIR",
            )
        self.assertEqual(default, Path.home() / ".cache" / "solradviewer")

        with patch.dict(os.environ, {"SOLRADVIEWER_CACHE_DIR": "~/cache"}, clear=True):
            configured = data._configured_path(
                data.DEFAULT_CACHE_ROOT,
                "SOLRADVIEWER_CACHE_DIR",
                "SAD_EOVSA_RENDER_CACHE_DIR",
            )
        self.assertEqual(configured, Path.home() / "cache")

    def test_subprocess_import_uses_configured_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "science"
            cache_root = root / "cache"
            output_root = root / "outputs"
            environment = os.environ.copy()
            environment.update(
                {
                    "SOLRADVIEWER_DATA_ROOT": str(data_root),
                    "SOLRADVIEWER_CACHE_DIR": str(cache_root),
                    "SOLRADVIEWER_OUTPUT_ROOT": str(output_root),
                }
            )
            for name in (
                "SAD_EOVSA_RENDER_CACHE_DIR",
                "SAD_EOVSA_DATA_ROOT",
                "SAD_EOVSA_OUTPUT_ROOT",
            ):
                environment.pop(name, None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from solradviewer.backend import data; "
                    "print(data.DATA_ROOT); "
                    "print(data.DEFAULT_OUTPUT_ROOT); "
                    "print(data.RENDER_DISK_CACHE.directory)",
                ],
                cwd=Path(__file__).resolve().parents[3],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue(cache_root.is_dir())

        self.assertEqual(
            result.stdout.splitlines(),
            [str(data_root), str(output_root), str(cache_root)],
        )


if __name__ == "__main__":
    unittest.main()
