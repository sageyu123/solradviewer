#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if (( $# > 1 )); then
  echo "Usage: ./run_app.sh [dataset-key]" >&2
  exit 2
fi

dataset_key="${1:-}"
if [[ -n "$dataset_key" ]]; then
  if [[ ! "$dataset_key" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Dataset key may contain only letters, numbers, underscores, and hyphens." >&2
    exit 2
  fi

  shopt -s nullglob
  manifest_matches=(manifests/*"$dataset_key"*.json)
  shopt -u nullglob
  if (( ${#manifest_matches[@]} == 0 )); then
    echo "No manifest matches dataset key: $dataset_key" >&2
    echo "Available manifests:" >&2
    for manifest in manifests/*.json; do
      [[ -e "$manifest" ]] && echo "  ${manifest#manifests/}" >&2
    done
    exit 2
  fi
  if (( ${#manifest_matches[@]} > 1 )); then
    echo "Dataset key is ambiguous: $dataset_key" >&2
    printf '  %s\n' "${manifest_matches[@]}" >&2
    exit 2
  fi

  manifest_path="${manifest_matches[0]}"
  export VITE_INITIAL_MANIFEST_JSON="$(< "$manifest_path")"
  export VITE_INITIAL_MANIFEST_NAME="${manifest_path#manifests/}"
  echo "Dataset: $VITE_INITIAL_MANIFEST_NAME"
else
  # A plain launch keeps the existing manual-load workflow even when the
  # caller's shell happens to retain launch variables from an earlier run.
  unset VITE_INITIAL_MANIFEST_JSON VITE_INITIAL_MANIFEST_NAME
fi

backend_pid=""
frontend_pid=""

# A wedged server from a previous run keeps listening on 8010/5174 and makes
# the health check pass against a dead session (or blocks the new bind).
for stale_port in 8010 5174; do
  stale_pids="$(lsof -nP -iTCP:"$stale_port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [[ -n "$stale_pids" ]]; then
    echo "Port $stale_port is held by pid(s): $stale_pids — stopping them first." >&2
    # shellcheck disable=SC2086
    kill $stale_pids 2>/dev/null || true
    sleep 2
    stale_pids="$(lsof -nP -iTCP:"$stale_port" -sTCP:LISTEN -t 2>/dev/null || true)"
    if [[ -n "$stale_pids" ]]; then
      # shellcheck disable=SC2086
      kill -9 $stale_pids 2>/dev/null || true
      sleep 1
    fi
  fi
done

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
    kill "$backend_pid" 2>/dev/null || true
  fi
  if [[ -n "$frontend_pid" ]] && kill -0 "$frontend_pid" 2>/dev/null; then
    kill "$frontend_pid" 2>/dev/null || true
  fi
  [[ -z "$backend_pid" ]] || wait "$backend_pid" 2>/dev/null || true
  [[ -z "$frontend_pid" ]] || wait "$frontend_pid" 2>/dev/null || true
}

wait_for_backend() {
  # First start imports the science stack (astropy/sunpy/pandas); allow for a
  # cold filesystem cache. Override with BACKEND_HEALTH_TIMEOUT if needed.
  local deadline=$((SECONDS + ${BACKEND_HEALTH_TIMEOUT:-90}))
  local waited=0
  while :; do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
      wait "$backend_pid" 2>/dev/null || true
      backend_pid=""
      echo "Backend exited before becoming ready." >&2
      return 1
    fi
    if curl --fail --silent --show-error --max-time 1 http://127.0.0.1:8010/api/health >/dev/null 2>&1; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for backend health at http://127.0.0.1:8010/api/health" >&2
      echo "Hint: check for a wedged server holding the port:" >&2
      echo "  lsof -nP -iTCP:8010 -sTCP:LISTEN" >&2
      return 1
    fi
    if (( SECONDS - waited >= 10 )); then
      waited=$SECONDS
      echo "  ... still starting backend (${waited}s)"
    fi
    sleep 0.25
  done
}

trap cleanup EXIT
trap 'exit 130' INT TERM

./run_backend.sh &
backend_pid=$!
if ! wait_for_backend; then
  exit 1
fi

./run_frontend.sh &
frontend_pid=$!

echo "Backend: http://127.0.0.1:8010"
echo "App:     http://127.0.0.1:5174"
echo "Press Ctrl+C to stop both services."

while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 1
done

status=0
if ! kill -0 "$backend_pid" 2>/dev/null; then
  wait "$backend_pid" || status=$?
  backend_pid=""
else
  wait "$frontend_pid" || status=$?
  frontend_pid=""
fi

exit "$status"
