#!/bin/bash
echo "=== STEP 1: Check API key ==="
echo "GOOGLE_API_KEY starts with: ${GOOGLE_API_KEY:0:10}..."
echo "GOOGLE_API_KEY length: ${#GOOGLE_API_KEY}"

echo ""
echo "=== STEP 2: List available flash models ==="
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
for m in genai.list_models():
    if 'flash' in m.name.lower():
        print(m.name, '|', getattr(m, 'supported_generation_methods', []))
" 2>&1

echo ""
echo "=== STEP 3: Test a tiny API call with gemini-2.0-flash ==="
python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-2.0-flash')
r = model.generate_content('Say hello in one word')
print('SUCCESS:', r.text)
" 2>&1

echo ""
echo "=== STEP 4: Check what model the transcription script uses ==="
grep -n 'model\|Model\|gemini' /lab/kiran/transcript-pipeline/02_transcription/gemini_transcribe.py | head -15

echo ""
echo "=== STEP 5: Check if any govardhan tmux sessions are still running ==="
tmux list-sessions 2>&1 | grep -i govar || echo "No govardhan sessions found"

echo ""
echo "=== DONE ==="
