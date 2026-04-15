#!/bin/bash
# Test Gemini 3.0 Flash transcription with a single Govardhan MP3  # re-run 2026-04-15 01:04:13
set -eo pipefail
set +u  # bridge wrapper already ran conda init; avoid bashrc unbound-var errors
echo "=== Gemini 3.0 Flash Single-File Test on $(hostname) at $(date) ==="

cd /lab/kiran/transcript-pipeline

git pull --ff-only 2>/dev/null || true

# Install new SDK if needed
pip install --quiet 'google-genai>=1.0'

echo ""
echo ">>> Step 1: List available Gemini models for this API key"
python -c "
from google import genai
import os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
print('Available Gemini models:')
for m in client.models.list():
    if 'gemini' in m.name.lower():
        print(f'  {m.name}')
"

echo ""
echo ">>> Step 2: Confirm gemini-3.0-flash is accessible"
python -c "
from google import genai
import os
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
resp = client.models.generate_content(model='gemini-3.0-flash', contents='Say hello in one sentence.')
print(f'gemini-3.0-flash responded: {resp.text}')
"
if [ $? -ne 0 ]; then echo "FATAL: gemini-3.0-flash not accessible — aborting"; exit 1; fi

echo ""
echo ">>> Step 3: Verify model name in transcription script"
grep MODEL_NAME 02_transcription/gemini_transcribe.py

echo ""
echo ">>> Step 4: Pick first Govardhan MP3 and transcribe"
INPUT_DIR="/lab/kiran/govardhan"
OUTPUT_DIR="/lab/kiran/govardhan_transcripts_30test"
mkdir -p "$OUTPUT_DIR"

MP3=$(ls "$INPUT_DIR"/*.mp3 | head -1)
STEM="$(basename "$MP3" .mp3)"
echo "Selected: $STEM"
echo "Input:    $MP3"
echo "Output:   $OUTPUT_DIR/$STEM/"

echo ""
echo ">>> Loading audio..."
ls -lh "$MP3"

echo ""
echo ">>> Calling Gemini 3.0 Flash transcription..."
START_T=$(date +%s)
python 02_transcription/gemini_transcribe.py "$MP3" --output_dir "$OUTPUT_DIR/$STEM"
END_T=$(date +%s)
ELAPSED=$((END_T - START_T))

echo ""
echo ">>> Step 5: Verify output"
OUT_JSON="$OUTPUT_DIR/$STEM/transcript.json"
if [[ -f "$OUT_JSON" ]] && [[ -s "$OUT_JSON" ]]; then
    SEGS=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$OUT_JSON")
    echo "SUCCESS: $SEGS segments in $ELAPSED seconds"
    echo "First 3 segments:"
    python -c "import json,sys; d=json.load(open(sys.argv[1])); [print(f'  {s}') for s in d[:3]]" "$OUT_JSON"
else
    echo "FAILED: transcript.json missing or empty"
    exit 1
fi

echo ""
echo "=== TEST PASSED — Gemini 3.0 Flash is working ==="
echo "=== Safe to run full Govardhan batch ==="

