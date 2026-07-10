#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python -m pip install --no-build-isolation --no-deps --force-reinstall -e .
