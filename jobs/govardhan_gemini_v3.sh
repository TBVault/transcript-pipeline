#!/usr/bin/env bash
# Job: Govardhan Gemini tier2 transcription — gemini-3-flash
# Input:  /lab/kiran/govardhan/*.mp3
# Output: /lab/kiran/govardhan_transcripts/<stem>/transcript.json
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh && conda activate vdabase

REPO_DIR="/lab/kiran/transcript-pipeline"
INPUT_DIR="/lab/kiran/govardhan"
OUTPUT_DIR="/lab/kiran/govardhan_transcripts"

cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true

echo "============================================"
echo "  GOVARDHAN GEMINI V3 (gemini-3-flash)"
echo "  $(hostname) — $(date)"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

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
    if python "$REPO_DIR/02_transcription/gemini_transcribe.py" "$mp3" --output_dir "$OUTPUT_DIR/$stem"; then
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
echo "=== DONE ==="
