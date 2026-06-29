#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python code/plotting/make_figures.py
