#!/usr/bin/env bash
set -euo pipefail

# Create the project virtualenv (.venv) and install microscope with all extras.
# Requires Python 3.11 (see pyproject: requires-python >=3.11,<3.12).

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "error: ${PYTHON_BIN} not found; set PYTHON_BIN to a Python 3.11 interpreter." >&2
    exit 1
fi

# Fetch the RelP submodule (read-only vendor dependency).
git submodule update --init --recursive

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[train,viz,dev]"

# Auto-strip notebook output on commit.
nbstripout --install

echo "Environment ready. Activate with: source .venv/bin/activate"
