# Publishing SolRadViewer

Release distributions contain the Python backend and the prebuilt browser interface. End users install with `python -m pip install solradviewer` and launch with `solradviewer`.

## First-time PyPI setup

The GitHub publishing workflow uses PyPI Trusted Publishing; no persistent upload token is stored in the repository.

While signed in to PyPI, add a pending GitHub publisher at [account publishing settings](https://pypi.org/manage/account/publishing/):

| Field | Value |
| --- | --- |
| PyPI project name | `solradviewer` |
| Owner | `sageyu123` |
| Repository | `solradviewer` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

The matching GitHub workflow is `.github/workflows/publish.yml`. A pending publisher creates the PyPI project on the first successful upload. See [PyPI's setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Build and validate

```bash
python -m pip install -e '.[test]' build twine
python -m unittest discover -s solradviewer/backend/tests
./scripts/build-release.sh
python -m twine check --strict dist/*
```

The build script runs the frontend build and packages its output into the Python wheel and source distribution. Verify that neither archive includes observation files, local manifests, sessions, outputs, credentials, or the former Python namespace. Install the wheel outside the checkout and verify the CLI, `/`, a bundled JavaScript asset, and `/api/health`.

## Publish

1. Update the version in `pyproject.toml` and `solradviewer/__init__.py` for subsequent releases. PyPI does not allow replacing an existing release file.
2. Commit and push the verified release changes.
3. Run **Publish to PyPI** from GitHub Actions with **publish** enabled, or publish a GitHub release. A manual run without **publish** only builds and validates the artifacts.
4. Verify the project and version on PyPI, and install the published version in a fresh environment.

The workflow builds and tests in a job without publishing permissions. A separate job receives the validated artifacts and obtains a short-lived PyPI upload credential for the `pypi` environment.
