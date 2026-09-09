# SolRadViewer package migration

The application and Python package now share the name **SolRadViewer** / `solradviewer`. This migration supersedes the earlier RadioView branding proposal and its temporary decision to retain the original Python namespace.

## Current layout

- `solradviewer/backend/data.py`: data readers, rendering, and analysis.
- `solradviewer/backend/app.py`: FastAPI application and routes.
- `solradviewer/backend/tests/`: backend regression tests.
- `SolRadSession`: the main analysis-session class.
- `solradviewer.backend.app:app`: the backend application target.

The package directory, Python imports, test patch targets, setuptools discovery, launcher, and documentation references move together. There is no compatibility package under the former import name.

## Update an existing installation

Stop the running backend before updating. From the repository root, activate its Python environment and reinstall the editable package:

```bash
source .venv/bin/activate
python -m pip install -e ".[test]"
./run_app.sh
```

Update custom scripts to use the new package and class:

```python
from solradviewer.backend.data import SolRadSession
```

For a manually launched backend, use:

```bash
python -m uvicorn solradviewer.backend.app:app --host 127.0.0.1 --port 8010
```

## Data and session compatibility

API routes, manifest roles, observation formats, and saved-session fields keep their existing contracts. AIA and EOVSA reader names remain instrument-specific. Legacy environment-variable aliases and analysis export filenames remain accepted for existing workflows. Saved JSON sessions do not require a namespace migration.

## Verification

```bash
python -m unittest discover -s solradviewer/backend/tests
npm run build --prefix frontend
```

Package verification also checks that built wheels contain the new namespace and exclude backend tests. Historical analysis design notes retain their scientific terminology, with source paths updated to the current package.
