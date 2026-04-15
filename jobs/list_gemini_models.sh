#!/bin/bash
echo "=== list_gemini_models on $(hostname) at $(date) ==="

# Match the working govardhan_gemini_final.sh pattern exactly
source ~/.bashrc
source /home3/kiran/anaconda3/etc/profile.d/conda.sh && conda activate vdabase

# Verify key is set
python -c "import os; k=os.environ.get('GOOGLE_API_KEY',''); print(f'Key length: {len(k)}, starts with: {k[:4]}...')"

echo ""
echo ">>> Listing all available Gemini models (filtering for flash and 3.0):"
python3 -c "
import os
from google import genai

client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
all_models = []
for m in client.models.list():
    all_models.append(m.name)

print('--- Models containing flash or 3.0 ---')
for name in sorted(all_models):
    if 'flash' in name.lower() or '3.0' in name or '3-0' in name:
        print(name)

print()
print('--- Full model list ---')
for name in sorted(all_models):
    print(name)
"

echo ""
echo "=== DONE ==="
