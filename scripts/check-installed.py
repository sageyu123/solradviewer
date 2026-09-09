"""Smoke-test an installed SolRadViewer release outside its source checkout."""

import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


with tempfile.TemporaryDirectory(prefix="solradviewer-smoke-") as temporary:
    root = Path(temporary)
    environment = os.environ.copy()
    environment["SOLRADVIEWER_CACHE_DIR"] = str(root / "cache")
    environment["SOLRADVIEWER_OUTPUT_ROOT"] = str(root / "outputs")
    environment.pop("PYTHONPATH", None)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "solradviewer", "--port", str(port)],
            cwd=root, env=environment, stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read().decode())
                try:
                    with urllib.request.urlopen(base + "/api/health", timeout=1) as response:
                        assert json.load(response)["status"] == "ok"
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise RuntimeError("Installed application did not become ready")
            with urllib.request.urlopen(base + "/", timeout=5) as response:
                assert response.headers.get_content_type() == "text/html"
                html = response.read().decode()
            assert "SolRadViewer" in html
            asset_paths = re.findall(r'(?:src|href)="(/assets/[^\"]+)"', html)
            assert asset_paths, "Installed frontend has no bundled assets"
            for asset in asset_paths:
                with urllib.request.urlopen(base + asset, timeout=5) as response:
                    assert response.read(), f"Empty bundled asset: {asset}"
            try:
                urllib.request.urlopen(base + "/api/not-a-route", timeout=5)
            except urllib.error.HTTPError as error:
                assert error.code == 404
            else:
                raise AssertionError("Unknown API route did not return 404")
            print("Installed release: CLI, frontend, assets, health, and API 404 checks passed")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
