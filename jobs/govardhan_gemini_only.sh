#!/usr/bin/env bash
# Job: Transcribe Govardhan 2025 MP3s using Gemini Flash only
# Input:  /lab/kiran/govardhan/*.mp3
# Output: /lab/kiran/govardhan_transcripts/<stem>/transcript.json
# Stops after transcription — no WhisperX, PyAnnote, or downstream stages.
set -euo pipefail

# Use hardcoded vdabase Python path — conda activate silently fails in non-interactive SSH
CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"

REPO_DIR="$HOME/transcript-pipeline"
INPUT_DIR="/lab/kiran/govardhan"
OUTPUT_DIR="/lab/kiran/govardhan_transcripts"

cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  GOVARDHAN GEMINI TRANSCRIPTION"
echo "  $(hostname) — $(date)"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

MP3_FILES=( "$INPUT_DIR"/*.mp3 )
TOTAL=${#MP3_FILES[@]}
echo ">>> Found $TOTAL MP3 files"
echo ""

DONE=0
SKIP=0
FAIL=0

for mp3 in "${MP3_FILES[@]}"; do
    stem="$(basename "$mp3" .mp3)"
    out_json="$OUTPUT_DIR/${stem}/transcript.json"

    if [[ -f "$out_json" ]]; then
        echo "[SKIP] $stem"
        (( SKIP++ )) || true
        continue
    fi

    echo ">>> [$((DONE+SKIP+FAIL+1))/$TOTAL] $stem"
    if "$PYTHON" "$REPO_DIR/02_transcription/gemini_transcribe.py" "$mp3" --output_dir "$OUTPUT_DIR/$stem"; then
        (( DONE++ )) || true
    else
        (( FAIL++ )) || true
        echo "[FAIL] $stem"
    fi
done

echo ""
echo "============================================"
echo "  DONE — $(date)"
echo "  Transcribed: $DONE  Skipped: $SKIP  Failed: $FAIL"
echo "  Results in: $OUTPUT_DIR"
echo "============================================"
