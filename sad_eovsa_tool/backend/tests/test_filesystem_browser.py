"""Tests for the read-only server filesystem browser endpoint."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from sad_eovsa_tool.backend import app as api


class FilesystemBrowserTest(unittest.TestCase):
    """Exercise filtering, ordering, navigation errors, and the home default."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "Zoo").mkdir()
        (self.root / "alpha").mkdir()
        (self.root / "A.FITS").write_bytes(b"fits")
        (self.root / "b.npz").write_bytes(b"npz")
        (self.root / "c.HDF5").write_bytes(b"hdf5")
        (self.root / "ignored.txt").write_text("not a source", encoding="utf-8")
        (self.root / ".hidden.fit").write_bytes(b"hidden")
        (self.root / ".hidden-dir").mkdir()
        self.client = TestClient(api.app)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_filters_and_sorts_entries(self) -> None:
        response = self.client.get("/api/fs/list", params={"path": str(self.root)})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["path"], str(self.root.resolve()))
        self.assertEqual(payload["parent"], str(self.root.resolve().parent))
        self.assertEqual(
            [(entry["name"], entry["isDir"]) for entry in payload["entries"]],
            [("alpha", True), ("Zoo", True), ("A.FITS", False), ("b.npz", False), ("c.HDF5", False)],
        )
        self.assertTrue(all(Path(entry["path"]).is_absolute() for entry in payload["entries"]))
        self.assertTrue(all(isinstance(entry["size"], int) for entry in payload["entries"]))
        self.assertTrue(all(isinstance(entry["mtime"], float) for entry in payload["entries"]))

    def test_show_hidden_and_follow_directory_symlink(self) -> None:
        link = self.root / "linked-dir"
        try:
            link.symlink_to(self.root / "alpha", target_is_directory=True)
        except OSError:
            link = None
        response = self.client.get("/api/fs/list", params={"path": str(self.root), "showHidden": 1})
        self.assertEqual(response.status_code, 200)
        entries = {entry["name"]: entry for entry in response.json()["entries"]}
        self.assertTrue(entries[".hidden-dir"]["isDir"])
        self.assertFalse(entries[".hidden.fit"]["isDir"])
        if link is not None:
            self.assertTrue(entries["linked-dir"]["isDir"])

    def test_rejects_missing_file_and_relative_paths(self) -> None:
        missing = self.client.get("/api/fs/list", params={"path": str(self.root / "missing")})
        regular_file = self.client.get("/api/fs/list", params={"path": str(self.root / "A.FITS")})
        relative = self.client.get("/api/fs/list", params={"path": "relative"})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(regular_file.status_code, 400)
        self.assertEqual(relative.status_code, 400)

    def test_defaults_to_home_directory(self) -> None:
        with patch.object(Path, "home", return_value=self.root):
            response = self.client.get("/api/fs/list")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["path"], str(self.root.resolve()))


if __name__ == "__main__":
    unittest.main()
