#!/usr/bin/env bash
set -e

echo "=== Running Full Benchmark ==="
echo ""

cd "$(dirname "$0")/.."

PYTHONPATH=. python3 -m framework.benchmark_runner

echo ""
echo "=== Benchmark complete ==="
echo "Results saved to analysis/"
ls -la analysis/*.csv 2>/dev/null || echo "No CSV files found in analysis/"
