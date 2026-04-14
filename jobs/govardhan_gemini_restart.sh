#!/usr/bin/env bash
# Restart Govardhan Gemini transcription after quota reset (midnight Pacific / 3 AM EDT)
# Skips already-completed transcripts automatically via gemini_transcribe.py's skip logic
set -euo pipefail

CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"
PIP="$CONDAROOT/anaconda3/envs/vdabase/bin/pip"

REPO_DIR="/lab/kiran/transcript-pipeline"
INPUT_DIR="/lab/kiran/govardhan"
OUTPUT_DIR="/lab/kiran/govardhan_transcripts"

cd "$REPO_DIR"
git pull --ff-only 2>/dev/null || true

echo "============================================"
echo "  GOVARDHAN GEMINI RESTART (post-quota-reset)"
echo "  $(hostname) — $(date)"
echo "============================================"

# Fix aiohttp for Python 3.9
echo ">>> Fixing aiohttp for Python 3.9 compatibility..."
"$PIP" install --quiet "aiohttp==3.9.5" 2>&1 | tail -3
echo ">>> aiohttp fix applied"
echo ""

mkdir -p "$OUTPUT_DIR"

MP3_FILES=( "$INPUT_DIR"/*.mp3 )
TOTAL=${#MP3_FILES[@]}
echo ">>> Found $TOTAL MP3 files total (will skip completed ones)"
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
    # Sleep between files to respect 15 RPM rate limit
    sleep 5
done

echo ""
echo "============================================"
echo "  DONE — $(date)"
echo "  Transcribed: $DONE  Skipped: $SKIP  Failed: $FAIL"
echo "  Results in: $OUTPUT_DIR"
echo "============================================"
