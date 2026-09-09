"""Tests for the installable CLI and bundled frontend behavior."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from solradviewer import cli
from solradviewer.backend import app as api


class PackageCliTest(unittest.TestCase):
    def test_cli_passes_host_and_port_to_uvicorn(self) -> None:
        output = StringIO()
        with patch.object(cli.uvicorn, "run") as run, redirect_stdout(output):
            self.assertEqual(cli.main(["--host", "0.0.0.0", "--port", "9123"]), 0)

        run.assert_called_once_with(
            "solradviewer.backend.app:app",
            host="0.0.0.0",
            port=9123,
        )
        self.assertIn("http://127.0.0.1:9123", output.getvalue())

    def test_cli_rejects_invalid_port(self) -> None:
        with self.assertRaises(SystemExit):
            cli.main(["--port", "70000"])


class BundledFrontendTest(unittest.TestCase):
    def test_frontend_and_api_routes_coexist_without_api_fallback(self) -> None:
        with TemporaryDirectory() as temporary:
            web_root = Path(temporary)
            assets = web_root / "assets"
            assets.mkdir()
            (web_root / "index.html").write_text("<html>release</html>", encoding="utf-8")
            (assets / "app.js").write_text("console.log('release');", encoding="utf-8")

            test_app = FastAPI()
            api.configure_frontend(test_app, web_root)
            test_app.add_api_route("/api/health", api.health)
            client = TestClient(test_app)

            self.assertEqual(client.get("/").status_code, 200)
            self.assertIn("release", client.get("/").text)
            self.assertEqual(client.get("/assets/app.js").status_code, 200)
            self.assertEqual(client.get("/api/health").json(), {"status": "ok"})
            self.assertEqual(client.get("/api/does-not-exist").status_code, 404)

    def test_source_checkout_keeps_json_root_fallback(self) -> None:
        with TemporaryDirectory() as temporary:
            test_app = FastAPI()
            api.configure_frontend(test_app, Path(temporary))
            client = TestClient(test_app)
            self.assertEqual(
                client.get("/").json(),
                {"app": "SolRadViewer", "frontend": "http://127.0.0.1:5174"},
            )


if __name__ == "__main__":
    unittest.main()
