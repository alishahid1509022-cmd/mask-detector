#!/usr/bin/env bash
# Convenience launcher for the Streamlit app.
# Usage: ./scripts/run_app.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "No .env found - copying .env.example to .env"
  cp .env.example .env
fi

# Ensure `import mask_detector` resolves even without `pip install -e .`.
export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"

streamlit run src/mask_detector/app.py
