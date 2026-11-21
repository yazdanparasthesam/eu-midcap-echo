#!/usr/bin/env bash
# Convenience wrapper for cron / docker hosts.
# Usage: ops/run.sh [full|daily]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-daily}"
if [ "$MODE" = "full" ]; then
    python scripts/run_pipeline.py --from-drive || python scripts/run_pipeline.py --full
else
    python scripts/daily_predict.py
fi
