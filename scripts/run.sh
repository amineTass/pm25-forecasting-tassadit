#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
project_python=""
for candidate in python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1; then
    project_python="$candidate"
    break
  fi
done
if [[ -z "$project_python" ]]; then
  echo "Install Python 3.11 or 3.12 first."
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  "$project_python" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s scripts/tests -v
.venv/bin/python -m src.run
