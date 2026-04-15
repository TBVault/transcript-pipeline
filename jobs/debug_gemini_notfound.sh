#!/usr/bin/env bash
# Debug the NotFound error from govardhan_gemini_only_2.sh
set -euo pipefail

CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
PYTHON="$CONDAROOT/anaconda3/envs/vdabase/bin/python"
PIP="$CONDAROOT/anaconda3/envs/vdabase/bin/pip"

REPO_DIR="/lab/kiran/transcript-pipeline"
INPUT_DIR="/lab/kiran/govardhan"

cd "$REPO_DIR"

echo "============================================"
echo "  DEBUG: Gemini NotFound Error"
echo "  $(hostname) — $(date)"
echo "============================================"

# 1. Check SDK version
echo ""
echo ">>> google-generativeai SDK version:"
"$PIP" show google-generativeai 2>&1 | grep -E "^(Name|Version)"

# 2. Check model name in script
echo ""
echo ">>> MODEL_NAME in gemini_transcribe.py:"
grep MODEL_NAME 02_transcription/gemini_transcribe.py

# 3. List available models to see if gemini-2.5-flash exists
echo ""
echo ">>> Available Gemini models (flash):"
"$PYTHON" -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY', ''))
for m in genai.list_models():
    if 'flash' in m.name.lower() or '2.5' in m.name:
        print(f'  {m.name}  (methods: {m.supported_generation_methods})')
" 2>&1 || echo "FAILED to list models"

# 4. Try a simple text-only call with the model
echo ""
echo ">>> Test text-only call with gemini-2.5-flash:"
"$PYTHON" -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv('GOOGLE_API_KEY', ''))
model = genai.GenerativeModel('gemini-2.5-flash')
resp = model.generate_content('Say hello in one word')
print('Response:', resp.text[:100])
" 2>&1 || echo "FAILED"

# 5. Check if input files exist
echo ""
echo ">>> Input MP3 files in $INPUT_DIR:"
ls "$INPUT_DIR"/*.mp3 2>/dev/null | head -5
echo "Total: $(ls "$INPUT_DIR"/*.mp3 2>/dev/null | wc -l) files"

# 6. Test with actual audio — single chunk from first file
echo ""
echo ">>> Test transcribe first 10s of first MP3:"
FIRST_MP3=$(ls "$INPUT_DIR"/*.mp3 2>/dev/null | head -1)
if [[ -n "$FIRST_MP3" ]]; then
    echo "Using: $FIRST_MP3"
    "$PYTHON" -c "
import google.generativeai as genai
import os, subprocess, tempfile
genai.configure(api_key=os.getenv('GOOGLE_API_KEY', ''))

audio_path = '$FIRST_MP3'
with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
    tmp_path = tmp.name
subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ss', '0', '-t', '10',
                 '-acodec', 'libmp3lame', '-q:a', '4', tmp_path],
                capture_output=True, check=True)
with open(tmp_path, 'rb') as f:
    audio_bytes = f.read()
print(f'Chunk size: {len(audio_bytes)} bytes')

model = genai.GenerativeModel('gemini-2.5-flash')
try:
    resp = model.generate_content([
        {'mime_type': 'audio/mpeg', 'data': audio_bytes},
        'Transcribe this audio briefly.'
    ])
    print('SUCCESS:', resp.text[:200])
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
os.unlink(tmp_path)
" 2>&1
else
    echo "No MP3 files found!"
fi

# 7. Check existing outputs / what was already transcribed
echo ""
echo ">>> Existing transcripts:"
ls /lab/kiran/govardhan_transcripts/ 2>/dev/null | head -10
echo "Total dirs: $(ls -d /lab/kiran/govardhan_transcripts/*/ 2>/dev/null | wc -l)"

echo ""
echo "============================================"
echo "  DEBUG COMPLETE — $(date)"
echo "============================================"
