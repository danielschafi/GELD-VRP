#!/usr/bin/env bash
# End-to-end smoke test: stage-1 SL, stage-2 SIL, minimal synthetic eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[smoke] syncing dependencies"
uv sync --extra dev

echo "[smoke] running pipeline"
uv run python -m tests.support.smoke_pipeline "$@"

echo "[smoke] done"
