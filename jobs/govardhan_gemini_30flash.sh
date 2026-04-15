#!/bin/bash
# Govardhan full batch — Gemini 3.0 Flash only (google.genai SDK)
set -eo pipefail
set +u  # bridge wrapper already ran conda init; avoid bashrc unbound-var errors
echo "=== govardhan_gemini_30flash on $(hostname) at $(date) ==="

cd /lab/kiran/transcript-pipeline

git pull --ff-only 2>/dev/null || true

pip install --quiet 'google-genai>=1.0'

echo ">>> Model name in script:"
grep MODEL_NAME 02_transcription/gemini_transcribe.py

echo ">>> Quick API test with gemini-3.0-flash..."
python -c "
from google import genai
import os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
print(client.models.generate_content(model='gemini-3.0-flash', contents='Say hello').text)
"
if [ $? -ne 0 ]; then echo "API TEST FAILED — aborting"; exit 1; fi

REPO_DIR="/lab/kiran/transcript-pipeline"
INPUT_DIR="/lab/kiran/govardhan"
OUTPUT_DIR="/lab/kiran/govardhan_transcripts"

cd "$REPO_DIR"

echo "============================================"
echo "  GOVARDHAN GEMINI 3.0-FLASH"
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

