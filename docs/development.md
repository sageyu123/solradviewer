# Development

## Run a source checkout

Requires **Python 3.10–3.12**, **Node.js 22.12+** (or Node 20.19+), npm, and Git. The launch scripts use Bash, `curl`, and `lsof`; on Windows, use WSL. Science dependencies are installed by pip.

```bash
git clone https://github.com/sageyu123/solradviewer.git
cd solradviewer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
npm install --prefix frontend
./run_app.sh
```

Open **[http://127.0.0.1:5174](http://127.0.0.1:5174)**. The backend listens on port **8010**. Press `Ctrl+C` in the launch terminal to stop both services.

To use an existing Python environment, activate it and skip virtual-environment creation. You can also select an interpreter explicitly:

```bash
PYTHON=/path/to/python ./run_app.sh
```

For a checkout on a cloud-synced drive, keeping the Python environment outside that drive can improve startup speed. Individual launchers `./run_backend.sh` and `./run_frontend.sh` are available for debugging. The combined launcher stops existing listeners on ports 8010 and 5174 before starting; choose those ports only for this app.

## Development and checks

```bash
python -m unittest discover -s solradviewer/backend/tests
npm run build --prefix frontend
```

Most backend checks create small synthetic data files. The original observational workflow tests skip when the optional sample dataset is unavailable. With the app running, check both services:

```bash
curl http://127.0.0.1:8010/api/health
curl http://127.0.0.1:5174/api/health
```

The backend uses FastAPI, NumPy/SciPy, Astropy, SunPy, and h5py; the frontend uses React, TypeScript, and Vite. Reader implementations live in [`solradviewer/backend/data.py`](https://github.com/sageyu123/solradviewer/blob/main/solradviewer/backend/data.py), and API routes in [`solradviewer/backend/app.py`](https://github.com/sageyu123/solradviewer/blob/main/solradviewer/backend/app.py).

The application, Python package, and Python/npm distributions use **SolRadViewer** / `solradviewer`. The main analysis-session class is `SolRadSession`. API routes, data formats, and saved-session fields remain compatible.

After updating an existing checkout, stop the backend, rerun `python -m pip install -e ".[test]"` in its environment, and restart with `./run_app.sh`. Custom Python scripts must use imports such as `from solradviewer.backend.data import SolRadSession`; the former Python namespace is no longer provided. See the [package migration notes](https://github.com/sageyu123/solradviewer/blob/main/docs/design/solradviewer-rename.md).

## Build a release

From a source checkout with Python and Node.js installed:

```bash
python -m pip install build twine
./scripts/build-release.sh
python -m twine check --strict dist/*
```

The build bundles the frontend into both the wheel and source distribution. Generated assets, local science data, sessions, and caches are not committed. See the [release guide](https://github.com/sageyu123/solradviewer/blob/main/docs/releasing.md) for publication steps.
