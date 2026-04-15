#!/bin/bash
set +u
echo "=== Listing Gemini Flash models on $(hostname) at $(date) ==="

# Try all possible env file locations
for f in /lab/kiran/.gemini_env ~/.gemini_env /home3/kiran/.gemini_env; do
    if [ -f "$f" ]; then
        echo "Found env file: $f"
        source "$f"
        break
    fi
done

# Also try bashrc
source ~/.bashrc 2>/dev/null || true

# Check if key is available
if [ -z "${GOOGLE_API_KEY:-}" ]; then
    echo "ERROR: GOOGLE_API_KEY not set after sourcing all env files"
    echo "Checking if .gemini_env exists anywhere:"
    find /lab/kiran /home3/kiran -maxdepth 2 -name "*.gemini*" -o -name "*gemini_env*" 2>/dev/null
    echo "=== DONE ==="
    exit 1
fi

echo "Key found (length: ${#GOOGLE_API_KEY})"

# Activate conda
source /home3/kiran/anaconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate vdabase 2>/dev/null

python3 -c "
import os
from google import genai
client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])
for m in sorted([m.name for m in client.models.list()]):
    if 'flash' in m.lower() or '3.0' in m or '3-0' in m:
        print(m)
"

echo ""
echo "=== DONE ==="
