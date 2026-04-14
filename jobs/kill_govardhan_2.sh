#!/usr/bin/env bash
# Kill govardhan_gemini_only_2 tmux session on igpu15
# Also check Job 16 output at /lab/kiran/transcript-pipeline/outputs/test_pipeline_16.txt
set -euo pipefail

echo "=== kill_govardhan_2.sh — $(hostname) — $(date) ==="

# Kill the stuck tmux session (soft kill first, then hard)
if tmux has-session -t govardhan_gemini_only_2 2>/dev/null; then
    echo ">>> Found tmux session govardhan_gemini_only_2 — killing it"
    tmux kill-session -t govardhan_gemini_only_2
    echo ">>> Killed"
else
    echo ">>> tmux session govardhan_gemini_only_2 not found (may already be dead)"
fi

# Also hard-kill by PID if still running
if kill -0 61378 2>/dev/null; then
    echo ">>> PID 61378 still alive — sending kill -9"
    kill -9 61378 && echo ">>> PID 61378 killed" || echo ">>> kill -9 failed (may have already exited)"
else
    echo ">>> PID 61378 not found (already dead)"
fi

echo ""
echo "=== Checking Job 16 output ==="
OUTPUT_FILE="/lab/kiran/transcript-pipeline/outputs/test_pipeline_16.txt"
if [[ -f "$OUTPUT_FILE" ]]; then
    SIZE=$(wc -c < "$OUTPUT_FILE")
    LINES=$(wc -l < "$OUTPUT_FILE")
    echo "EXISTS: $OUTPUT_FILE"
    echo "Size: ${SIZE} bytes, ${LINES} lines"
    echo "--- last 20 lines ---"
    tail -20 "$OUTPUT_FILE"
else
    echo "NOT FOUND: $OUTPUT_FILE"
fi

echo ""
echo "=== Done — $(date) ==="
