#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to build the bundled frontend." >&2
  exit 1
fi

python_cmd="${PYTHON:-python}"
frontend_dir="$repo_root/frontend"
web_dir="$repo_root/solradviewer/web"
frontend_dist="$frontend_dir/dist"

if [[ ! -f "$frontend_dir/package-lock.json" ]]; then
  echo "Missing frontend/package-lock.json; cannot run a reproducible frontend build." >&2
  exit 1
fi

npm --prefix "$frontend_dir" ci
VITE_INITIAL_MANIFEST_JSON= VITE_INITIAL_MANIFEST_NAME= npm --prefix "$frontend_dir" run build

if [[ ! -f "$frontend_dist/index.html" ]]; then
  echo "Frontend build did not produce frontend/dist/index.html." >&2
  exit 1
fi

# This is the only generated directory the release script removes. The
# frontend build is copied into the Python package so wheels run without Node.
rm -rf "$web_dir"
mkdir -p "$web_dir"
cp -R "$frontend_dist"/. "$web_dir"/

if ! "$python_cmd" -m build; then
  echo "Python build failed. Install the 'build' package and retry." >&2
  exit 1
fi
