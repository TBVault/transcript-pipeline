#!/usr/bin/env bash
# Job: List available Gemini Flash models
set -euo pipefail

source ~/anaconda3/etc/profile.d/conda.sh && conda activate vdabase

echo "=== Listing Gemini Flash models on $(hostname) at $(date) ==="

python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
for m in genai.list_models():
    if 'flash' in m.name.lower():
        print(m.name, '—', m.display_name)
"

echo "=== DONE ==="
