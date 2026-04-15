#!/bin/bash
set +u
echo "=== test_gemini3_single on $(hostname) at $(date) ==="

# Fix .bashrc unbound variable issue
set +u
source ~/.bashrc 2>/dev/null || true
set -u

# Conda setup
CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
export PATH=$CONDAROOT/anaconda3/bin:$CONDAROOT/anaconda3/condabin:$PATH
source $CONDAROOT/anaconda3/etc/profile.d/conda.sh
conda activate vdabase

cd /lab/kiran/transcript-pipeline

# Pull latest to get the new google.genai script
git checkout -- 02_transcription/gemini_transcribe.py 2>/dev/null || true
git pull --ff-only origin main || true

echo ""
echo ">>> Step 1: Check google-genai SDK is installed"
pip show google-genai 2>&1 | grep -E "^(Name|Version)" || {
    echo "google-genai not found, installing..."
    pip install google-genai
}

echo ""
echo ">>> Step 2: Check model in script"
grep "MODEL_NAME" 02_transcription/gemini_transcribe.py

echo ""
echo ">>> Step 3: Verify API key"
python -c "import os; k=os.environ.get('GOOGLE_API_KEY',''); print(f'Key length: {len(k)}, starts with: {k[:4]}...')"

echo ""
echo ">>> Step 4: Test text call with google.genai + gemini-3.0-flash"
python -c "
from google import genai
import os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
resp = client.models.generate_content(model='gemini-3.0-flash', contents='Say hello in one word')
print('Text test OK:', resp.text[:100])
"
if [ $? -ne 0 ]; then echo "TEXT API TEST FAILED — aborting"; exit 1; fi

echo ""
echo ">>> Step 5: Single-file audio transcription test"
INPUT_DIR="/lab/kiran/govardhan"
OUTPUT_DIR="/lab/kiran/govardhan_transcripts_test"
mkdir -p "$OUTPUT_DIR"

FIRST_MP3=$(ls "$INPUT_DIR"/*.mp3 2>/dev/null | head -1)
if [[ -z "$FIRST_MP3" ]]; then
    echo "No MP3 files found in $INPUT_DIR!"
    exit 1
fi

STEM=$(basename "$FIRST_MP3" .mp3)
echo "Testing with: $FIRST_MP3"
echo "Output dir: $OUTPUT_DIR/$STEM"

# Remove any previous test output so we get a fresh run
rm -rf "$OUTPUT_DIR/$STEM"

python 02_transcription/gemini_transcribe.py "$FIRST_MP3" --output_dir "$OUTPUT_DIR/$STEM"
EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]] && [[ -f "$OUTPUT_DIR/$STEM/transcript.json" ]]; then
    SEG_COUNT=$(python -c "import json; print(len(json.load(open('$OUTPUT_DIR/$STEM/transcript.json'))))")
    echo "SUCCESS: $SEG_COUNT segments transcribed"
    echo "First 500 chars of transcript:"
    head -c 500 "$OUTPUT_DIR/$STEM/transcript.json"
else
    echo "FAILED with exit code $EXIT_CODE"
fi

echo ""
echo "=== DONE ==="
