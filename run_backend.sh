#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${PYTHON:-}" && -x "$(pwd)/.venv/bin/python" ]]; then
  PYTHON="$(pwd)/.venv/bin/python"
fi

if [[ -z "${PYTHON:-}" ]]; then
  PYTHON="$(command -v python 2>/dev/null || true)"
  if [[ -z "$PYTHON" ]]; then
    PYTHON="$(command -v python3 2>/dev/null || true)"
  fi
fi

if [[ -z "$PYTHON" ]]; then
  echo "Could not find Python. Set PYTHON to a Python executable." >&2
  exit 1
fi

exec "$PYTHON" -m uvicorn solradviewer.backend.app:app --reload --reload-dir solradviewer --host 127.0.0.1 --port 8010 --timeout-graceful-shutdown 3
