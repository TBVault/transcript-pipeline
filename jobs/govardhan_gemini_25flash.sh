#!/bin/bash
echo "=== govardhan_gemini_25flash on $(hostname) at $(date) ==="

# Load env + API key
source ~/.bashrc
source /home3/kiran/anaconda3/etc/profile.d/conda.sh && conda activate vdabase
cd /lab/kiran/transcript-pipeline

# Pull latest code (has model fix: gemini-2.5-flash)
git pull --ff-only 2>/dev/null || true

# Verify model fix is in place
echo ">>> Model name in script:"
grep MODEL_NAME 02_transcription/gemini_transcribe.py

# Verify key is set
python -c "import os; k=os.environ.get('GOOGLE_API_KEY',''); print(f'Key length: {len(k)}, starts with: {k[:4]}...')"

# Quick API test with gemini-2.5-flash
echo ">>> Testing API with gemini-2.5-flash..."
python -c "import google.generativeai as genai; import os; genai.configure(api_key=os.environ['GOOGLE_API_KEY']); print(genai.GenerativeModel('gemini-2.5-flash').generate_content('Say hello').text)"
if [ $? -ne 0 ]; then echo "API TEST FAILED — aborting"; exit 1; fi

set -euo pipefail

REPO_DIR="/lab/kiran/transcript-pipeline"
INPUT_DIR="/lab/kiran/govardhan"
OUTPUT_DIR="/lab/kiran/govardhan_transcripts"

cd "$REPO_DIR"

echo "============================================"
echo "  GOVARDHAN GEMINI 2.5-FLASH"
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
        # Only skip if file has actual content (not empty from failed runs)
        if [[ -s "$out_json" ]] && python -c "import json,sys; d=json.load(open(sys.argv[1])); assert len(d)>0" "$out_json" 2>/dev/null; then
            echo "[SKIP] $stem"
            (( SKIP++ )) || true
            continue
        else
            echo "[REDO] $stem (empty/invalid transcript)"
            rm -f "$out_json"
        fi
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
