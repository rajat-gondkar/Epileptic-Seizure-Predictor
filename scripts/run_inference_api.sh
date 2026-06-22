#!/usr/bin/env bash
# Start the EEG seizure-prediction inference API.
# Run from the project root:  bash scripts/run_inference_api.sh
set -e
cd "$(dirname "$0")/.."
PY=./venv/bin/python
if [ ! -x "$PY" ]; then PY=python3; fi
echo "Starting inference API on http://localhost:8000 (docs at /docs)"
exec "$PY" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
