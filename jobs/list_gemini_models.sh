#!/bin/bash
set +u
echo "=== list_gemini_models on $(hostname) at $(date) ==="

source ~/.bashrc 2>/dev/null || true

CONDAROOT=$(cat /lab/kiran/envs/$(hostname).txt)
export PATH=$CONDAROOT/anaconda3/bin:$CONDAROOT/anaconda3/condabin:$PATH
source $CONDAROOT/anaconda3/etc/profile.d/conda.sh
conda activate vdabase
source /lab/kiran/.gemini_env 2>/dev/null || true

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
