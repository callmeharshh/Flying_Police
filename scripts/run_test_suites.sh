#!/usr/bin/env bash
# Run unit and validation test suites.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate.sh 2>/dev/null || source venv/bin/activate 2>/dev/null || true

echo "=== Unit tests ==="
python -m pytest tests/unit -m unit -v --tb=short "$@"

echo ""
echo "=== Validation tests (fixture regression) ==="
python -m pytest tests/validation -m "validation and not integration" -v --tb=short "$@"

echo ""
echo "=== Integration tests (BLIP + PS QA — optional, loads model) ==="
echo "    Run separately: pytest tests/integration -m integration -v"
echo ""
echo "Unit + validation suites passed."
